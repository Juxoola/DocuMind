"""Жизненный цикл BM25-индекса: инкрементальная сборка, debounce, отмена и проверка готовности."""

import asyncio
import logging
import os

import aiofiles.os
from llama_index.core.schema import TextNode

import config
from src.rag.state import (
    _BM25_DEBOUNCE_SEC,
    _bm25_node_cache,
    _bm25_pending_dbpath,
    _bm25_pending_nodes,
    _bm25_pending_timers,
    _bm25_rebuilding,
)

# ── Локальные asyncio.Lock для конкурентного доступа ──
# Локальные asyncio.Lock вместо импортированных threading-based блокировок
_bm25_pending_lock = asyncio.Lock()
_bm25_rebuilding_lock = asyncio.Lock()
_bm25_node_cache_lock = asyncio.Lock()

logger = logging.getLogger(__name__)
_bm25_tasks: set[asyncio.Task] = set()


# Инкрементальная сборка BM25: из кэша узлов или полная загрузка из ChromaDB
# ── Инкрементальная сборка BM25: из кэша узлов или полная загрузка из ChromaDB ──
async def _rebuild_bm25_bg(notebook_id: str, db_path: str, new_nodes: list = None):
    _PAGE_SIZE = 2000
    try:
        paths = config.get_notebook_paths(notebook_id)
        bm25_dir = os.path.join(paths["base"], "bm25")
        await aiofiles.os.makedirs(bm25_dir, exist_ok=True)

        new_nodes = new_nodes or []
        async with _bm25_node_cache_lock:
            old_nodes = _bm25_node_cache.get(notebook_id, [])

        if old_nodes or new_nodes:
            full_corpus = old_nodes + new_nodes
            logger.info(
                f"[RAG] BM25 инкрементальная сборка: "
                f"{len(old_nodes)} кеш + {len(new_nodes)} новых = {len(full_corpus)} узлов"
            )
        else:
            import chromadb as _chromadb

            tmp_client = await asyncio.to_thread(_chromadb.PersistentClient, path=db_path)
            try:
                collection = tmp_client.get_or_create_collection("multimodal_rag")
                full_corpus = []
                offset = 0
                while True:
                    result = collection.get(limit=_PAGE_SIZE, offset=offset)
                    ids = result.get("ids", [])
                    if not ids:
                        break
                    documents = result.get("documents", []) or []
                    metadatas = result.get("metadatas", []) or []
                    for i, doc_id in enumerate(ids):
                        text = documents[i] if i < len(documents) else ""
                        meta = metadatas[i] if i < len(metadatas) else {}
                        if meta is None:
                            meta = {}
                        fname = meta.get("file_name", "")
                        page = meta.get("page", "")
                        t = meta.get("start", meta.get("time", ""))
                        coord_parts = []
                        if fname:
                            coord_parts.append(str(fname))
                        if page not in ("", None):
                            coord_parts.append(f"стр.{page}")
                        elif t not in ("", None):
                            coord_parts.append(f"@{t}")
                        if coord_parts:
                            text = f"[{'. '.join(coord_parts)}]: {text}"
                        full_corpus.append(TextNode(text=text, id_=doc_id, metadata=meta))
                    if len(ids) < _PAGE_SIZE:
                        break
                    offset += _PAGE_SIZE
            finally:
                try:
                    tmp_client.close()
                except Exception:
                    pass
            logger.info(f"[RAG] BM25 холодная сборка из ChromaDB: {len(full_corpus)} узлов")

        if full_corpus:

            def _build_retriever():
                from llama_index.retrievers.bm25 import BM25Retriever

                return BM25Retriever.from_defaults(
                    nodes=full_corpus,
                    similarity_top_k=config.rag.top_k_per_file,
                    language="russian",
                )

            retriever = await asyncio.to_thread(_build_retriever)

            await asyncio.to_thread(retriever.persist, bm25_dir)

            async with _bm25_node_cache_lock:
                _bm25_node_cache[notebook_id] = full_corpus
            logger.info(f"[RAG] ✅ BM25 обновлён: {len(full_corpus)} узлов.")
    except Exception as e:
        logger.warning(f"[RAG] Ошибка фоновой сборки BM25: {e}")


# Планирование rebuild с debounce: повторные вызовы сбрасывают предыдущий таймер
# ── Планирование rebuild с debounce: повторные вызовы сбрасывают предыдущий таймер ──
async def _schedule_bm25_rebuild(notebook_id: str, db_path: str, new_nodes: list = None):
    async with _bm25_pending_lock:
        old = _bm25_pending_timers.get(notebook_id)
        if old is not None:
            try:
                old.cancel()
            except Exception as e:
                logger.debug(f"Не удалось отменить таймер debounce для {notebook_id}: {e}")
        _bm25_pending_dbpath[notebook_id] = db_path

        if new_nodes:
            existing = _bm25_pending_nodes.setdefault(notebook_id, [])
            existing.extend(new_nodes)
            logger.debug(f"[RAG] +{len(new_nodes)} узлов в BM25 pending (всего: {len(existing)})")

    async def _fire():
        await asyncio.sleep(_BM25_DEBOUNCE_SEC)
        async with _bm25_pending_lock:
            _bm25_pending_timers.pop(notebook_id, None)
            path = _bm25_pending_dbpath.pop(notebook_id, None)
            pending = _bm25_pending_nodes.pop(notebook_id, [])
        if path is None:
            return
        async with _bm25_rebuilding_lock:
            _bm25_rebuilding.add(notebook_id)
        try:
            await _rebuild_bm25_bg(notebook_id, path, new_nodes=pending)
        finally:
            async with _bm25_rebuilding_lock:
                _bm25_rebuilding.discard(notebook_id)

    task = asyncio.create_task(_fire())
    _bm25_pending_timers[notebook_id] = task
    logger.info(
        f"[RAG] ⏱ BM25 rebuild запланирован через {_BM25_DEBOUNCE_SEC:.0f}с "
        "(можно сбросить через flush_bm25_rebuild)"
    )


# ── Отмена запланированного rebuild ──
async def cancel_bm25_rebuild(notebook_id: str):
    async with _bm25_pending_lock:
        task = _bm25_pending_timers.pop(notebook_id, None)
        _bm25_pending_dbpath.pop(notebook_id, None)
        _bm25_pending_nodes.pop(notebook_id, None)
        if task is not None:
            try:
                task.cancel()
            except Exception as e:
                logger.debug(f"[cancel_bm25_rebuild] task.cancel: {e}")


# Принудительный flush: немедленный rebuild без ожидания debounce
# ── Принудительный flush: немедленный rebuild без ожидания debounce ──
async def flush_bm25_rebuild(
    notebook_id: str,
    db_path: str = None,
    wait: bool = False,
    timeout: float = 120.0,
):
    async with _bm25_pending_lock:
        task = _bm25_pending_timers.pop(notebook_id, None)
        if task is not None:
            try:
                task.cancel()
            except Exception as e:
                logger.debug(f"Не удалось отменить задачу flush_bm25 для {notebook_id}: {e}")
        path = _bm25_pending_dbpath.pop(notebook_id, None)
        pending = _bm25_pending_nodes.pop(notebook_id, [])
        if path is None and db_path is not None:
            path = db_path
        if path is None:
            paths = config.get_notebook_paths(notebook_id)
            path = paths["chroma_db"]
    if path is None:
        return
    if not wait:

        async def _bg():
            try:
                await _rebuild_bm25_bg(notebook_id, path, new_nodes=pending)
            finally:
                async with _bm25_rebuilding_lock:
                    _bm25_rebuilding.discard(notebook_id)

        async with _bm25_rebuilding_lock:
            _bm25_rebuilding.add(notebook_id)
        _t = asyncio.create_task(_bg())
        _bm25_tasks.add(_t)
        _t.add_done_callback(_bm25_tasks.discard)
        return
    async with _bm25_rebuilding_lock:
        _bm25_rebuilding.add(notebook_id)
    try:
        await _rebuild_bm25_bg(notebook_id, path, new_nodes=pending)
    finally:
        async with _bm25_rebuilding_lock:
            _bm25_rebuilding.discard(notebook_id)


# Проверка готовности BM25: файл существует, нет pending-таймеров и нет активной сборки
# ── Проверка готовности BM25-индекса ──
async def is_bm25_ready(notebook_id: str) -> bool:
    paths = config.get_notebook_paths(notebook_id)
    bm25_dir = os.path.join(paths["base"], "bm25")
    exists = await aiofiles.os.path.exists(os.path.join(bm25_dir, "retriever.json"))
    async with _bm25_pending_lock:
        has_pending = notebook_id in _bm25_pending_timers
    async with _bm25_rebuilding_lock:
        is_rebuilding = notebook_id in _bm25_rebuilding
    return exists and not has_pending and not is_rebuilding
