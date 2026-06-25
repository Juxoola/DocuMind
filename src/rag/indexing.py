"""ChromaDB операции: построение индекса, клиенты, close."""

import asyncio
import logging
import os

import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.core.storage.storage_context import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore

import config
from src.rag.bm25 import _schedule_bm25_rebuild
from src.rag.models import init_settings
from src.rag.state import _CLIENT_CACHE_MAXSIZE, _client_cache

logger = logging.getLogger(__name__)

_client_cache_lock = asyncio.Lock()


# Макс. символов на чанк для embedding — запас 12000 (≈3000 токенов) от лимита контекста
_MAX_EMBED_CHARS = 12000


# Построение векторного индекса из nodes и запуск фоновой пересборки BM25
async def build_index(nodes, notebook_id: str):
    await init_settings()

    # Defensive truncate: обрезаем чанки > лимита embedding-контекста (SemanticSplitter даёт >4000 токенов)
    for node in nodes:
        if len(node.text) > _MAX_EMBED_CHARS:
            logger.warning(
                f"[Indexing] Чанк обрезан: {len(node.text)} → {_MAX_EMBED_CHARS} симв. "
                f"(file={node.metadata.get('file_name', '?')})"
            )
            node.text = node.text[:_MAX_EMBED_CHARS]

    vector_store = await get_vector_store(notebook_id)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    def _build():
        return VectorStoreIndex(nodes, storage_context=storage_context)

    index = await asyncio.to_thread(_build)

    paths = config.get_notebook_paths(notebook_id)
    db_path = paths["chroma_db"]
    await _schedule_bm25_rebuild(notebook_id, db_path, new_nodes=nodes)
    return index


async def get_vector_store(notebook_id: str):
    global _client_cache
    paths = config.get_notebook_paths(notebook_id)
    db_path = paths["chroma_db"]

    async with _client_cache_lock:
        if db_path in _client_cache:
            _client_cache.move_to_end(db_path)
            _, vector_store = _client_cache[db_path]
            return vector_store
        if len(_client_cache) >= _CLIENT_CACHE_MAXSIZE:
            _oldest_path, (_oldest_client, _) = _client_cache.popitem(last=False)
            try:
                _oldest_client.close()
            except Exception:
                pass
            logger.debug(f"LRU eviction: закрыт ChromaDB клиент {_oldest_path}")

    def _create_chroma_client():
        os.makedirs(db_path, exist_ok=True)
        db = chromadb.PersistentClient(path=db_path)
        collection = db.get_or_create_collection("multimodal_rag")
        return db, collection

    db, chroma_collection = await asyncio.to_thread(_create_chroma_client)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    async with _client_cache_lock:
        if db_path in _client_cache:
            try:
                db.close()
            except Exception:
                pass
            _, vector_store = _client_cache[db_path]
            _client_cache.move_to_end(db_path)
        else:
            _client_cache[db_path] = (db, vector_store)

    return vector_store


async def close_all_clients():
    global _client_cache
    async with _client_cache_lock:
        for path, (client, _) in _client_cache.items():
            try:
                client.close()
            except Exception as e:
                logger.debug(f"Ошибка закрытия ChromaDB клиента {path}: {e}")
        _client_cache.clear()


async def close_notebook_client(notebook_id: str):
    global _client_cache
    from config import get_notebook_paths

    db_path = get_notebook_paths(notebook_id)["chroma_db"]
    async with _client_cache_lock:
        entry = _client_cache.pop(db_path, None)
    if entry is not None:
        client, _ = entry
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
