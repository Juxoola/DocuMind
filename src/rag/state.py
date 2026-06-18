"""Глобальное состояние RAG-подсистемы: кеш моделей, клиентов, блокировки и debounce BM25."""

import logging
import threading

import requests

import config
from routers.shared import make_http_session

logger = logging.getLogger(__name__)

_init_lock = threading.Lock()

_model_cache: dict = {}

_client_cache: dict = {}

# HTTP-сессия реранкера: отдельный pool чтобы не конкурировать с основными запросами
_rerank_session: requests.Session | None = None
_rerank_session_lock = threading.Lock()


def _get_rerank_session() -> requests.Session:
    global _rerank_session
    if _rerank_session is None:
        with _rerank_session_lock:
            if _rerank_session is None:
                _rerank_session = make_http_session(config.HTTP_POOL_SIZE_RERANK)
    return _rerank_session


# Debounce BM25: несколько вызовов подряд сбрасывают таймер (10с покоя)
_BM25_DEBOUNCE_SEC = 10.0
_bm25_pending_timers: dict = {}
_bm25_pending_dbpath: dict = {}
_bm25_pending_lock = threading.Lock()
_bm25_rebuilding: set = set()
_bm25_node_cache: dict[str, list] = {}
_bm25_pending_nodes: dict[str, list] = {}
