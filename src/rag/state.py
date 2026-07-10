"""Глобальное состояние RAG-подсистемы: кэши, блокировки и фоновые задачи."""

import asyncio
import logging
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

# Кэши моделей и ChromaDB-клиентов с LRU-эвикцией
_model_cache: dict = {}
_model_cache_lock = asyncio.Lock()
_CLIENT_CACHE_MAXSIZE = 20
_client_cache: OrderedDict = OrderedDict()

_INDEX_CACHE_MAXSIZE = 50
_index_cache: OrderedDict = OrderedDict()
_index_cache_lock = asyncio.Lock()

# Выделенный пул потоков для CPU-интенсивных RAG-операций (embedding, reranking, BM25)
# ── Выделенный пул потоков для CPU-интенсивных RAG-операций ──
RAG_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="rag")

# Debounce-механизм фоновой пересборки BM25: повторные вызовы сбрасывают таймер
_BM25_DEBOUNCE_SEC = 10.0
_bm25_pending_timers: dict = {}
_bm25_pending_dbpath: dict = {}
_bm25_rebuilding: set = set()
_bm25_node_cache: dict[str, list] = {}
_bm25_pending_nodes: dict[str, list] = {}
