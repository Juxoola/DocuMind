"""
Тесты main.py (FastAPI).

Используем TestClient + моки на уровне sys.modules,
чтобы не требовать llama_index, torch, chromadb, GPU.
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(scope="module")
def client():
    """
    TestClient с полным моком зависимостей.

    Стратегия:
    1. Удаляем из sys.modules ВСЕ src. и main модули (чтобы не было кеша).
    2. Подменяем sys.modules для всех тяжёлых зависимостей (llama_index, torch, chromadb).
    3. Импортируем main — его импорты получат замоканные модули.
    """
    # ── 1. Чистим кеш модулей проекта ──
    for mod_name in list(sys.modules.keys()):
        if mod_name.startswith("main") or mod_name.startswith("src."):
            sys.modules.pop(mod_name, None)

    # ── 2. Создаём моки ──
    mock_rag = MagicMock()
    mock_rag.build_index = MagicMock()
    mock_rag.retrieve_nodes = MagicMock(return_value=[])
    mock_rag.build_file_context = MagicMock(return_value=([], ""))
    mock_rag.make_prompt = MagicMock(return_value="prompt")
    mock_rag.close_all_clients = MagicMock()
    mock_rag.preload_all_models = MagicMock()
    mock_rag.unload_rag_models = MagicMock()

    mock_gguf = MagicMock()
    mock_gguf.get_gguf_llm = MagicMock(return_value="http://127.0.0.1:49152")
    mock_gguf.get_gguf_embedding_url = MagicMock(return_value="http://127.0.0.1:49153")
    mock_gguf.preload_gguf_llm = MagicMock(return_value={"status": "ready", "port": 49152})
    mock_gguf.get_llm_status = MagicMock(return_value={"state": "idle", "port": None})
    mock_gguf.unload_all_models = MagicMock()
    mock_gguf.kill_stray_servers = MagicMock()
    mock_gguf.count_running_servers = MagicMock(return_value=0)
    mock_gguf.get_loaded_models = MagicMock(return_value=[])
    mock_gguf.detect_model_family = MagicMock(return_value="qwen")
    mock_gguf.stream_gguf_chat = MagicMock()

    mock_manager = MagicMock()
    mock_manager.scan_gguf_dirs = MagicMock(return_value=[])

    mock_ingest = MagicMock()
    mock_ingest.ingest_file = MagicMock(return_value=[])

    mock_bookmarks = MagicMock()
    mock_bookmarks.list_bookmarks = MagicMock(return_value=[])
    mock_bookmarks.get_bookmark = MagicMock(return_value=None)
    mock_bookmarks.create_bookmark = MagicMock(return_value={"id": "test"})
    mock_bookmarks.update_bookmark = MagicMock(return_value={"id": "test"})
    mock_bookmarks.delete_bookmark = MagicMock(return_value=True)
    mock_bookmarks.mark_stale_for_file = MagicMock(return_value=0)

    # ── 3. Все моки в словарь ──
    mocks = {
        "src.rag_pipeline": mock_rag,
        "src.gguf_direct": mock_gguf,
        "src.gguf_manager": mock_manager,
        "src.ingestion": mock_ingest,
        "src.bookmarks": mock_bookmarks,
        # llama_index — полностью мокаем, каждый подмодуль отдельно
        "llama_index": MagicMock(),
        "llama_index.core": MagicMock(),
        "llama_index.core.schema": MagicMock(),
        "llama_index.core.settings": MagicMock(),
        "llama_index.core.storage": MagicMock(),
        "llama_index.core.storage.storage_context": MagicMock(),
        "llama_index.core.vector_stores": MagicMock(),
        "llama_index.core.vector_stores.types": MagicMock(),
        "llama_index.core.node_parser": MagicMock(),
        "llama_index.core.retrievers": MagicMock(),
        "llama_index.llms.openai": MagicMock(),
        "llama_index.embeddings.openai": MagicMock(),
        "llama_index.vector_stores.chroma": MagicMock(),
        "llama_index.readers.file": MagicMock(),
        "llama_index.retrievers.bm25": MagicMock(),
        # Тяжёлые ML-библиотеки
        "torch": MagicMock(),
        "chromadb": MagicMock(),
        "cv2": MagicMock(),
        "whisperx": MagicMock(),
    }

    # ── 4. Применяем подмену ──
    with patch.dict(sys.modules, mocks, clear=False):
        # Теперь импортируем main — все его зависимости уже замоканы
        import main as app_module

        app = app_module.app
        client = TestClient(app)
        yield client
    # Восстановление sys.modules происходит автоматически при выходе из patch.dict


# ── Тесты ────────────────────────────────────────────────────────────


class TestNotebookEndpoints:
    """CRUD для ноутбуков."""

    def test_get_notebooks_empty(self, client):
        resp = client.get("/api/notebooks")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_create_notebook(self, client):
        resp = client.post("/api/notebooks", json={"name": "Test Notebook"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "Test Notebook"
        assert "id" in data
        assert "created_at" in data

    def test_create_and_list(self, client):
        resp = client.post("/api/notebooks", json={"name": "My Notes"})
        nb = resp.json()
        nb_id = nb["id"]
        resp2 = client.get("/api/notebooks")
        ids = [n["id"] for n in resp2.json()]
        assert nb_id in ids


class TestConfigEndpoints:
    """Эндпоинты конфигурации."""

    def test_get_gguf_config(self, client):
        resp = client.get("/api/gguf-config")
        assert resp.status_code == 200
        data = resp.json()
        assert "search_dirs" in data
        assert "default_ctx_size" in data

    def test_get_rag_config(self, client):
        resp = client.get("/api/rag-config")
        assert resp.status_code == 200
        data = resp.json()
        assert "embedding_model" in data
        assert "top_k_per_file" in data
        assert "use_reranker" in data

    def test_update_rag_config(self, client):
        resp = client.post(
            "/api/update-rag-config",
            json={
                "embedding_model": "test.gguf",
                "reranker_model": "test-reranker.gguf",
                "top_k_per_file": 10,
                "rerank_pool": 30,
                "final_top_n": 15,
                "use_reranker": True,
            },
        )
        assert resp.status_code == 200


class TestGGUFEndpoints:
    """Эндпоинты управления GGUF."""

    def test_gguf_models(self, client):
        resp = client.get("/api/gguf-models")
        assert resp.status_code == 200
        assert "models" in resp.json()

    def test_gguf_loaded(self, client):
        resp = client.get("/api/gguf-loaded")
        assert resp.status_code == 200
        assert resp.json() == {"loaded_models": []}

    def test_gguf_status(self, client):
        resp = client.get("/api/gguf-status")
        assert resp.status_code == 200
        assert resp.json()["running_count"] == 0

    def test_gguf_unload(self, client):
        resp = client.post("/api/gguf-unload")
        assert resp.status_code == 200

    def test_get_llm_status(self, client):
        resp = client.get("/api/llm-status")
        assert resp.status_code == 200
        assert resp.json()["state"] == "idle"


class TestUploadValidation:
    """Валидация загрузки файлов."""

    def test_upload_invalid_extension(self, client):
        resp = client.post(
            "/api/upload?notebook_id=test",
            files={"file": ("virus.exe", b"x" * 100, "application/octet-stream")},
        )
        assert resp.status_code == 415

    def test_upload_pdf_valid(self, client):
        resp = client.post(
            "/api/upload?notebook_id=test",
            files={"file": ("doc.pdf", b"%PDF-1.4 test", "application/pdf")},
        )
        assert resp.status_code in (200, 400, 500)


class TestBookmarkEndpoints:
    """Закладки."""

    def test_list_bookmarks(self, client):
        resp = client.get("/api/bookmarks?notebook_id=test")
        assert resp.status_code == 200
        assert resp.json() == {"bookmarks": []}

    def test_create_bookmark(self, client):
        resp = client.post(
            "/api/bookmarks",
            json={
                "notebook_id": "test",
                "question": "What is RAG?",
                "answer": "Retrieval Augmented Generation",
            },
        )
        assert resp.status_code == 200
        assert "id" in resp.json()


class TestChatEndpoint:
    """Чат — базовая валидация."""

    def test_chat_no_files(self, client):
        resp = client.post(
            "/api/chat",
            json={
                "query": "test",
                "allowed_files": [],
                "notebook_id": "test",
                "max_tokens": 256,
            },
        )
        assert resp.status_code == 200
        assert resp.headers.get("content-type", "").startswith("text/event-stream")
        assert "[DONE]" in resp.text
