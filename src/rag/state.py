"""Глобальное состояние RAG-подсистемы: кеш моделей, клиентов, блокировки и debounce BM25."""

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

_BM25_DEBOUNCE_SEC = 30.0
_bm25_pending_timers: dict = {}
_bm25_pending_dbpath: dict = {}
_bm25_pending_lock = threading.Lock()
_bm25_rebuilding: set = set()
