"""ChromaDB операции: построение индекса, клиенты, close."""

import logging
import os

import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.core.storage.storage_context import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore

import config
from src.rag.bm25 import _schedule_bm25_rebuild
from src.rag.models import init_settings
from src.rag.state import _client_cache

logger = logging.getLogger(__name__)


def build_index(nodes, notebook_id: str):
    """Построение VectorStoreIndex из нод и сохранение в ChromaDB."""
    init_settings()
    vector_store = get_vector_store(notebook_id)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(nodes, storage_context=storage_context)

    # Debounced BM25 rebuild: одна пересборка на batch
    paths = config.get_notebook_paths(notebook_id)
    db_path = paths["chroma_db"]
    _schedule_bm25_rebuild(notebook_id, db_path)
    return index


def get_vector_store(notebook_id: str):
    """ChromaVectorStore для ноутбука (с кешированием клиента)."""
    global _client_cache
    paths = config.get_notebook_paths(notebook_id)
    db_path = paths["chroma_db"]
    os.makedirs(db_path, exist_ok=True)

    if db_path not in _client_cache:
        _client_cache[db_path] = chromadb.PersistentClient(path=db_path)

    db = _client_cache[db_path]
    chroma_collection = db.get_or_create_collection("multimodal_rag")
    return ChromaVectorStore(chroma_collection=chroma_collection)


def close_all_clients():
    """Явно закрывает все открытые клиенты ChromaDB для снятия блокировок файлов."""
    global _client_cache
    for path, client in _client_cache.items():
        try:
            client.close()
        except Exception as e:
            logger.debug(f"Ошибка закрытия ChromaDB клиента {path}: {e}")
    _client_cache.clear()


def close_notebook_client(notebook_id: str):
    """Закрыть ChromaDB клиент для указанного ноутбука. Не трогает другие."""
    global _client_cache
    from config import get_notebook_paths

    db_path = get_notebook_paths(notebook_id)["chroma_db"]
    client = _client_cache.pop(db_path, None)
    if client is not None:
        try:
            client.close()
            logger.debug(f"ChromaDB клиент закрыт для {notebook_id} ({db_path})")
        except Exception as e:
            logger.debug(f"Ошибка закрытия ChromaDB клиента {db_path}: {e}")
            _client_cache[db_path] = client  # возвращаем, если не удалось закрыть
    else:
        logger.debug(f"Нет открытого ChromaDB клиента для {notebook_id}")
