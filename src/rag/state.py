"""Глобальное состояние RAG-подсистемы: кеш моделей, клиентов, блокировки и debounce BM25."""

# Синглтоны и блокировки для параллельного доступа к моделям,
# HTTP-пулам и фоновой пересборке BM25. Всё общее состояние RAG живёт здесь.

import logging
import threading

import requests

import config
from routers.shared import make_http_session

logger = logging.getLogger(__name__)

# Блокировка init-фазы — предотвращает гонку при загрузке моделей
_init_lock = threading.Lock()

_model_cache: dict = {}

_client_cache: dict = {}

# HTTP-сессия для реранкера: отдельный connection pool,
# чтобы не конкурировать с основными запросами приложения.
# Ленивая инициализация — сессия создаётся при первом запросе, а не при импорте.
_rerank_session: requests.Session | None = None
_rerank_session_lock = threading.Lock()


def _get_rerank_session() -> requests.Session:
    global _rerank_session
    if _rerank_session is None:
        with _rerank_session_lock:
            if _rerank_session is None:
                _rerank_session = make_http_session(config.HTTP_POOL_SIZE_RERANK)
    return _rerank_session


# Debounce-механизм для BM25: несколько вызовов _schedule_bm25_rebuild
# подряд сбрасывают таймер, чтобы пересборка запускалась только после
# последнего изменения индекса (по умолчанию 30 секунд покоя)
_BM25_DEBOUNCE_SEC = 30.0
_bm25_pending_timers: dict = {}
_bm25_pending_dbpath: dict = {}
_bm25_pending_lock = threading.Lock()
_bm25_rebuilding: set = set()
# Кеш узлов BM25 в памяти: при debounce новые узлы аккумулируются здесь,
# при срабатывании таймера пересборка идёт из кеша + новых узлов → без чтения ChromaDB.
_bm25_node_cache: dict[str, list] = {}
_bm25_pending_nodes: dict[str, list] = {}

# ВНИМАНИЕ: _model_cache и _client_cache не защищены RWLock — все
# обращения идут через однопоточный ASGI-цикл или защищены _init_lock.
