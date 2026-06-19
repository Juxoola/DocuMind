"""Глобальное состояние RAG-подсистемы: кеш моделей, клиентов, блокировки и debounce BM25."""

import logging
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import requests

import config
from routers.shared import make_http_session

logger = logging.getLogger(__name__)

_init_lock = threading.Lock()

_model_cache: dict = {}
_model_cache_lock = threading.Lock()
_CLIENT_CACHE_MAXSIZE = 20
_client_cache: OrderedDict = OrderedDict()
_client_cache_lock = threading.Lock()

# Кеш VectorStoreIndex — не пересоздаётся при каждом запросе
_INDEX_CACHE_MAXSIZE = 50
_index_cache: OrderedDict = OrderedDict()
_index_cache_lock = threading.Lock()

# Выделенный пул потоков для RAG-операций (ChromaDB + BM25 + reranking).
# Отдельный от default executor чтобы не блокировать другие async-задачи.
RAG_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="rag")

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
_bm25_rebuilding_lock = threading.Lock()
_bm25_node_cache: dict[str, list] = {}
_bm25_node_cache_lock = threading.Lock()
_bm25_pending_nodes: dict[str, list] = {}
