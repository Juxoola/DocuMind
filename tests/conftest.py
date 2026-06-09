"""
Общие фикстуры pytest для проекта NotebookLM Local Clone.

Стратегия:
- Тесты модульные: не трогаем GGUF-сервера, ChromaDB, внешние API.
- test_config.py — изолированные тесты конфига (чистые функции + парсинг).
- test_rag_pipeline.py — _rrf_fuse, _rrf_fuse_across_files (чистые функции).
- test_gguf_direct.py — detect_model_family, CACHE_TYPE_MAP (чистые функции).
- test_main.py — FastAPI TestClient (монтируем только нужные эндпоинты).
"""

import os
import shutil
import sys
import tempfile

import pytest

# Добавляем корень проекта в PYTHONPATH
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


@pytest.fixture(autouse=True)
def temp_notebooks_dir(monkeypatch):
    """Временно перенаправляем NOTEBOOKS_DIR в temp-директорию.
    Автоиспользование: каждый тест получает изолированную ФС.
    """
    tmp = tempfile.mkdtemp(prefix="nb_test_")
    import config as cfg

    monkeypatch.setattr(cfg, "NOTEBOOKS_DIR", tmp)
    monkeypatch.setattr(cfg, "BASE_DIR", PROJECT_ROOT)
    # Создаём поддиректории
    os.makedirs(tmp, exist_ok=True)
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture(autouse=True)
def sanitize_env(monkeypatch):
    """Фиксим env-переменные, чтобы тесты не зависели от реальных путей."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("GGUF_SEARCH_DIRS", "/dev/null")
    monkeypatch.setenv("LM_STUDIO_URL", "http://localhost:9999/v1")
    monkeypatch.setenv("UPLOAD_MAX_SIZE_MB", "500")
    # Перенаправляем rag_config.json во временную папку (тесты не загрязняют прод)
    import config as cfg

    monkeypatch.setattr(
        cfg,
        "RAG_CONFIG_FILE",
        os.path.join(tempfile.gettempdir(), f"rag_config_test_{os.getpid()}.json"),
    )
