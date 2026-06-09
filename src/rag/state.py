"""Глобальное состояние RAG-подсистемы: кеш моделей, клиентов, блокировки и debounce BM25."""

# Синглтоны и блокировки для параллельного доступа к моделям,
# HTTP-пулам и фоновой пересборке BM25. Всё общее состояние RAG живёт здесь.

import logging
import threading

import requests
import requests.adapters

import config

logger = logging.getLogger(__name__)

# Блокировка init-фазы — предотвращает гонку при загрузке моделей
_init_lock = threading.Lock()

_model_cache: dict = {}

_client_cache: dict = {}

# HTTP-сессия для реранкера: отдельный connection pool,
# чтобы не конкурировать с основными запросами приложения
_rerank_session = requests.Session()
_rerank_session.mount(
    "http://",
    requests.adapters.HTTPAdapter(
        pool_connections=config.HTTP_POOL_SIZE_RERANK,
        pool_maxsize=config.HTTP_POOL_SIZE_RERANK,
    ),
)
_rerank_session.mount(
    "https://",
    requests.adapters.HTTPAdapter(
        pool_connections=config.HTTP_POOL_SIZE_RERANK,
        pool_maxsize=config.HTTP_POOL_SIZE_RERANK,
    ),
)

# Debounce-механизм для BM25: несколько вызовов _schedule_bm25_rebuild
# подряд сбрасывают таймер, чтобы пересборка запускалась только после
# последнего изменения индекса (по умолчанию 30 секунд покоя)
_BM25_DEBOUNCE_SEC = 30.0
_bm25_pending_timers: dict = {}
_bm25_pending_dbpath: dict = {}
_bm25_pending_lock = threading.Lock()
_bm25_rebuilding: set = set()

# ВНИМАНИЕ: _model_cache и _client_cache не защищены RWLock — все
# обращения идут через однопоточный ASGI-цикл или защищены _init_lock.
