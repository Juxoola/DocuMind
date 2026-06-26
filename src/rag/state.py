"""Глобальное состояние RAG-подсистемы: кеш моделей, клиентов, блокировки и debounce BM25."""

import asyncio
import logging
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

_init_lock = asyncio.Lock()

_model_cache: dict = {}
_model_cache_lock = asyncio.Lock()
_CLIENT_CACHE_MAXSIZE = 20
_client_cache: OrderedDict = OrderedDict()
_client_cache_lock = asyncio.Lock()

# Кеш VectorStoreIndex — не пересоздаётся при каждого запроса
_INDEX_CACHE_MAXSIZE = 50
_index_cache: OrderedDict = OrderedDict()
_index_cache_lock = threading.Lock()

# Выделенный пул потоков для RAG-операций (ChromaDB + BM25 + reranking).
# Отдельный от default executor чтобы не блокировать другие async-задачи.
RAG_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="rag")

# Debounce BM25: несколько вызовов подряд сбрасывают таймер (10с покоя)
_BM25_DEBOUNCE_SEC = 10.0
_bm25_pending_timers: dict = {}
_bm25_pending_dbpath: dict = {}
_bm25_pending_lock = asyncio.Lock()
_bm25_rebuilding: set = set()
_bm25_rebuilding_lock = asyncio.Lock()
_bm25_node_cache: dict[str, list] = {}
_bm25_node_cache_lock = asyncio.Lock()
_bm25_pending_nodes: dict[str, list] = {}
