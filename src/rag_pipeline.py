"""Trampoline для обратной совместимости.

Весь код RAG переехал в пакет src/rag/. Этот файл реэкспортирует
все публичные и приватные символы, чтобы старые импорты
  from src.rag_pipeline import ...
продолжали работать без изменений.
"""

from src.rag.bm25 import (
    _rebuild_bm25_bg,
    _schedule_bm25_rebuild,
    cancel_bm25_rebuild,
    flush_bm25_rebuild,
    is_bm25_ready,
)
from src.rag.indexing import (
    build_index,
    close_all_clients,
    close_notebook_client,
    get_vector_store,
)
from src.rag.models import init_settings, preload_all_models, unload_rag_models
from src.rag.prompt import build_file_context, get_embedding_url, make_prompt
from src.rag.retrieval import (
    _QUERY_GEN_PROMPT,
    _get_qe_llm,
    _rrf_fuse,
    _rrf_fuse_across_files,
    retrieve_nodes,
)
from src.rag.state import (
    _BM25_DEBOUNCE_SEC,
    _bm25_pending_dbpath,
    _bm25_pending_lock,
    _bm25_pending_timers,
    _bm25_rebuilding,
    _client_cache,
    _init_lock,
    _model_cache,
    _rerank_session,
)

__all__ = [
    "_BM25_DEBOUNCE_SEC",
    "_QUERY_GEN_PROMPT",
    "_bm25_pending_dbpath",
    "_bm25_pending_lock",
    "_bm25_pending_timers",
    "_bm25_rebuilding",
    "_client_cache",
    "_get_qe_llm",
    "_init_lock",
    "_model_cache",
    "_rebuild_bm25_bg",
    "_rerank_session",
    "_rrf_fuse",
    "_rrf_fuse_across_files",
    "_schedule_bm25_rebuild",
    "build_file_context",
    "build_index",
    "cancel_bm25_rebuild",
    "close_all_clients",
    "close_notebook_client",
    "flush_bm25_rebuild",
    "get_embedding_url",
    "get_vector_store",
    "init_settings",
    "is_bm25_ready",
    "make_prompt",
    "preload_all_models",
    "retrieve_nodes",
    "unload_rag_models",
]
