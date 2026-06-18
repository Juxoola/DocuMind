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
from src.rag.state import _client_cache, _client_cache_lock

logger = logging.getLogger(__name__)


# Построение векторного индекса из nodes и запуск фоновой
# пересборки BM25. Вызывается после обработки загруженного файла.
def build_index(nodes, notebook_id: str):
    init_settings()
    vector_store = get_vector_store(notebook_id)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(nodes, storage_context=storage_context)

    paths = config.get_notebook_paths(notebook_id)
    db_path = paths["chroma_db"]
    _schedule_bm25_rebuild(notebook_id, db_path, new_nodes=nodes)
    return index


def get_vector_store(notebook_id: str):
    global _client_cache
    paths = config.get_notebook_paths(notebook_id)
    db_path = paths["chroma_db"]
    os.makedirs(db_path, exist_ok=True)

    with _client_cache_lock:
        if db_path not in _client_cache:
            _client_cache[db_path] = chromadb.PersistentClient(path=db_path)
        db = _client_cache[db_path]

    chroma_collection = db.get_or_create_collection("multimodal_rag")
    return ChromaVectorStore(chroma_collection=chroma_collection)


def close_all_clients():
    global _client_cache
    with _client_cache_lock:
        for path, client in _client_cache.items():
            try:
                client.close()
            except Exception as e:
                logger.debug(f"Ошибка закрытия ChromaDB клиента {path}: {e}")
        _client_cache.clear()


def close_notebook_client(notebook_id: str):
    global _client_cache
    from config import get_notebook_paths

    db_path = get_notebook_paths(notebook_id)["chroma_db"]
    with _client_cache_lock:
        client = _client_cache.pop(db_path, None)
    if client is not None:
        try:
            try:
                col = client.get_collection("multimodal_rag")
                col.delete()
                client.delete_collection("multimodal_rag")
            except Exception:
                pass
            try:
                client.close()
            except Exception:
                pass
            logger.debug(f"ChromaDB клиент закрыт для {notebook_id} ({db_path})")
        except Exception as e:
            logger.debug(f"Ошибка закрытия ChromaDB клиента {db_path}: {e}")
    else:
        logger.debug(f"Нет открытого ChromaDB клиента для {notebook_id}")
