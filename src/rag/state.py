"""Глобальное состояние RAG-подсистемы.

Вынесено из rag_pipeline.py при рефакторинге.

Содержит:
- _model_cache: кеш загруженных моделей (эмбеддинг, реранкер)
- _init_lock: блокировка инициализации (предотвращает race при загрузке)
- _client_cache: кеш ChromaDB PersistentClient
- _rerank_session: HTTP session для реранкера
- BM25 debounce state: таймеры, блокировки, флаги
"""

import logging
import threading

import requests
import requests.adapters

import config

logger = logging.getLogger(__name__)

# F-fix #6: предотвращает гонку при конкурентной загрузке моделей
_init_lock = threading.Lock()

# Кеш загруженных моделей (ключи: "embed_model", "reranker")
_model_cache: dict = {}

# Кеш ChromaDB PersistentClient по пути к БД
_client_cache: dict = {}

# F-fix #15: Session для rerank-запросов (keep-alive)
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

# ── BM25 debounce state ─────────────────────────────────────────────
# F-fix #4: debounce для batch-загрузки (один rebuild на пачку, не N)
_BM25_DEBOUNCE_SEC = 30.0
_bm25_pending_timers: dict = {}  # notebook_id → threading.Timer
_bm25_pending_dbpath: dict = {}  # notebook_id → str (chroma_db path)
_bm25_pending_lock = threading.Lock()
_bm25_rebuilding: set = set()  # notebook_id которые сейчас строят BM25
