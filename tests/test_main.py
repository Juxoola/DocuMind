"""
Тесты main.py (FastAPI).

Используем TestClient + комбинацию подходов:
- Внешние пакеты (torch, chromadb, llama_index) — patch.dict(sys.modules, ...),
  т.к. они импортируются на уровне модулей src.* до того, как @patch может вмешаться.
- Проектные модули (src.rag.*, src.gguf.*, etc.) — импортируем и
  назначаем атрибуты напрямую, без sys.modules.
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Внешние пакеты, которые импортируются на top-level уровне в src.*.
# Их нельзя перехватить через @patch — только через sys.modules.
_HEAVY_PACKAGES = {
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
    "torch": MagicMock(),
    "chromadb": MagicMock(),
    "cv2": MagicMock(),
    "whisperx": MagicMock(),
}


@pytest.fixture(scope="module")
def client():
    """
    TestClient с моками зависимостей.

    Стратегия:
    1. Удаляем из sys.modules кеш проекта (чтобы не было stale-ссылок).
    2. Мокаем внешние тяжёлые пакеты через patch.dict(sys.modules, ...).
    3. Импортируем проектные модули и назначаем их атрибуты напрямую.
    4. Импортируем main.
    """
    # ── 1. Чистим кеш модулей проекта ──
    for mod_name in list(sys.modules.keys()):
        if mod_name.startswith("main") or mod_name.startswith("src."):
            sys.modules.pop(mod_name, None)

    # ── 2. Внешние пакеты — только sys.modules ──
    with patch.dict(sys.modules, _HEAVY_PACKAGES, clear=False):
        # ── 3. Проектные модули — импортируем и назначаем атрибуты ──
        import src.rag.bm25
        import src.rag.indexing
        import src.rag.models
        import src.rag.prompt
        import src.rag.retrieval

        src.rag.retrieval.retrieve_nodes = MagicMock(return_value=[])
        src.rag.prompt.build_file_context = MagicMock(return_value=([], ""))
        src.rag.prompt.make_prompt = MagicMock(return_value="prompt")
        src.rag.indexing.build_index = MagicMock()
        src.rag.indexing.close_all_clients = MagicMock()
        src.rag.models.preload_all_models = MagicMock()
        src.rag.models.unload_rag_models = MagicMock()
        src.rag.indexing.get_vector_store = MagicMock()
        src.rag.prompt.get_embedding_url = MagicMock()
        src.rag.bm25.flush_bm25_rebuild = MagicMock()

        import src.gguf.server

        src.gguf.server.get_gguf_llm = MagicMock(return_value="http://127.0.0.1:49152")
        src.gguf.server.get_gguf_embedding_url = MagicMock(return_value="http://127.0.0.1:49153")
        src.gguf.server.preload_gguf_llm = MagicMock(
            return_value={"status": "ready", "port": 49152}
        )
        src.gguf.server.get_llm_status = MagicMock(return_value={"state": "idle", "port": None})
        src.gguf.server.unload_all_models = MagicMock()
        src.gguf.server.kill_stray_servers = MagicMock()
        src.gguf.server.count_running_servers = MagicMock(return_value=0)
        src.gguf.server.get_loaded_models = MagicMock(return_value=[])
        src.gguf.models = MagicMock()
        src.gguf.models.detect_model_family = MagicMock(return_value="qwen")
        src.gguf.streaming = MagicMock()
        src.gguf.streaming.stream_gguf_chat = MagicMock()

        import src.gguf.scanner

        src.gguf.scanner.scan_gguf_dirs = MagicMock(return_value=[])

        import src.ingestion

        src.ingestion.ingest_file = MagicMock(return_value=[])
        src.ingestion.unload_whisper_model = MagicMock()
        src.ingestion.kill_subprocesses = MagicMock(return_value=0)
        src.ingestion.IngestionCancelled = RuntimeError

        import src.bookmarks

        src.bookmarks.list_bookmarks = MagicMock(return_value=[])
        src.bookmarks.get_bookmark = MagicMock(return_value=None)
        src.bookmarks.create_bookmark = MagicMock(return_value={"id": "test"})
        src.bookmarks.update_bookmark = MagicMock(return_value={"id": "test"})
        src.bookmarks.delete_bookmark = MagicMock(return_value=True)
        src.bookmarks.mark_stale_for_file = MagicMock(return_value=0)

        # ── 4. Импортируем main ──
        import main as app_module

        yield TestClient(app_module.app)


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
