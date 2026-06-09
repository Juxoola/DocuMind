"""BM25 lifecycle: сборка, debounce, отмена, проверка готовности.

Вынесено из rag_pipeline.py при рефакторинге.
"""

import logging
import os
import threading

from llama_index.core.schema import TextNode

import config
from src.rag.state import (
    _BM25_DEBOUNCE_SEC,
    _bm25_pending_dbpath,
    _bm25_pending_lock,
    _bm25_pending_timers,
    _bm25_rebuilding,
)

logger = logging.getLogger(__name__)


def _rebuild_bm25_bg(notebook_id: str, db_path: str):
    """Перестройка BM25-индекса в фоновом потоке. Не блокирует основной поток.

    F-fix #bm25-pagination: раньше collection.get() без limit тянул ВСЮ коллекцию
    в память за один вызов. На ноутбуке с 50k+ чанков это 1-2 ГБ RAM-спайк.
    Теперь читаем порциями по _BM25_PAGE_SIZE, аккумулируя в список.
    """
    _PAGE_SIZE = 2000
    try:
        paths = config.get_notebook_paths(notebook_id)
        bm25_dir = os.path.join(paths["base"], "bm25")
        os.makedirs(bm25_dir, exist_ok=True)
        # Отдельный клиент для фонового потока (нельзя использовать общий _client_cache)
        import chromadb as _chromadb

        tmp_client = _chromadb.PersistentClient(path=db_path)
        collection = tmp_client.get_or_create_collection("multimodal_rag")
        bm25_nodes = []
        offset = 0
        total_seen = 0
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
                # F1: BM25 видит координаты чанка
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
                    text = f"[{' '.join(coord_parts)}]: {text}"
                bm25_nodes.append(TextNode(text=text, id_=doc_id, metadata=meta))
            total_seen += len(ids)
            if len(ids) < _PAGE_SIZE:
                break
            offset += _PAGE_SIZE
        if bm25_nodes:
            from llama_index.retrievers.bm25 import BM25Retriever

            retriever = BM25Retriever.from_defaults(
                nodes=bm25_nodes,
                similarity_top_k=config.RAG_TOP_K_PER_FILE,
                language="russian",
            )
            retriever.persist(bm25_dir)
            logger.info(f"[RAG] ✅ BM25 обновлён в фоне: {len(bm25_nodes)} узлов.")
    except Exception as e:
        logger.warning(f"[RAG] Ошибка фоновой сборки BM25: {e}")


def _schedule_bm25_rebuild(notebook_id: str, db_path: str):
    """Отложить rebuild BM25. Каждый новый вызов сбрасывает таймер на _BM25_DEBOUNCE_SEC."""
    with _bm25_pending_lock:
        old = _bm25_pending_timers.get(notebook_id)
        if old is not None:
            try:
                old.cancel()
            except Exception:
                pass
        _bm25_pending_dbpath[notebook_id] = db_path

        def _fire():
            with _bm25_pending_lock:
                _bm25_pending_timers.pop(notebook_id, None)
                path = _bm25_pending_dbpath.pop(notebook_id, None)
            if path is None:
                return
            _bm25_rebuilding.add(notebook_id)
            try:
                _rebuild_bm25_bg(notebook_id, path)
            finally:
                _bm25_rebuilding.discard(notebook_id)

        t = threading.Timer(_BM25_DEBOUNCE_SEC, _fire)
        t.daemon = True
        _bm25_pending_timers[notebook_id] = t
        t.start()
        logger.info(
            f"[RAG] ⏱ BM25 rebuild запланирован через {_BM25_DEBOUNCE_SEC:.0f}с "
            "(можно сбросить через flush_bm25_rebuild)"
        )


def cancel_bm25_rebuild(notebook_id: str):
    """Отменить отложенный BM25 rebuild для ноутбука БЕЗ немедленного rebuild.

    F-fix #notebook-delete: при удалении ноутбука нужно гарантировать, что
    фоновый таймер (если есть) НЕ дёрнется через 30с и не попытается открыть
    уже удалённый chroma_db с FileNotFoundError.
    """
    with _bm25_pending_lock:
        timer = _bm25_pending_timers.pop(notebook_id, None)
        _bm25_pending_dbpath.pop(notebook_id, None)
        if timer is not None:
            try:
                timer.cancel()
            except Exception as e:
                logger.debug(f"[cancel_bm25_rebuild] timer.cancel: {e}")


def flush_bm25_rebuild(
    notebook_id: str,
    db_path: str = None,
    wait: bool = False,
    timeout: float = 120.0,
):
    """Форсировать немедленную пересборку BM25. Используется в конце batch-upload.

    Args:
        notebook_id: ID ноутбука.
        db_path: путь к ChromaDB (если None — берётся из config).
        wait: если True — блокирует вызывающий поток до завершения rebuild.
        timeout: макс. время ожидания при wait=True.
    """
    with _bm25_pending_lock:
        timer = _bm25_pending_timers.pop(notebook_id, None)
        if timer is not None:
            try:
                timer.cancel()
            except Exception:
                pass
        path = _bm25_pending_dbpath.pop(notebook_id, None)
        if path is None and db_path is not None:
            path = db_path
        if path is None:
            paths = config.get_notebook_paths(notebook_id)
            path = paths["chroma_db"]
    if path is None:
        return
    if not wait:
        _bm25_rebuilding.add(notebook_id)

        def _bg():
            try:
                _rebuild_bm25_bg(notebook_id, path)
            finally:
                _bm25_rebuilding.discard(notebook_id)

        threading.Thread(target=_bg, daemon=True, name=f"bm25-flush-{notebook_id}").start()
        return
    # wait=True: синхронный rebuild в текущем потоке
    _bm25_rebuilding.add(notebook_id)
    try:
        _rebuild_bm25_bg(notebook_id, path)
    finally:
        _bm25_rebuilding.discard(notebook_id)


def is_bm25_ready(notebook_id: str) -> bool:
    """True если BM25-индекс существует на диске и нет pending/rebuilding."""
    paths = config.get_notebook_paths(notebook_id)
    bm25_dir = os.path.join(paths["base"], "bm25")
    exists = os.path.exists(os.path.join(bm25_dir, "bm25_retriever_params.json"))
    with _bm25_pending_lock:
        has_pending = notebook_id in _bm25_pending_timers
    is_rebuilding = notebook_id in _bm25_rebuilding
    return exists and not has_pending and not is_rebuilding
