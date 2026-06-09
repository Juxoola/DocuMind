"""Trampoline для обратной совместимости.

Весь GGUF-код переехал в пакет src/gguf/. Этот файл реэкспортирует
все публичные и приватные символы, чтобы старые импорты
  from src.gguf_direct import ...
продолжали работать без изменений.
"""

import requests  # noqa: F401 — нужно для patch("src.gguf_direct.requests.get") в тестах

from src.gguf.models import detect_model_family
from src.gguf.server import (
    _start_llm_server_sync,
    count_running_servers,
    get_active_embedding_parallel,
    get_active_llm_url,
    get_gguf_embedding_url,
    get_gguf_llm,
    get_loaded_models,
    get_llm_status,
    is_server_ready,
    kill_stray_servers,
    preload_gguf_llm,
    unload_all_models,
    unload_rag_models_safe,
)
from src.gguf.state import (
    CACHE_TYPE_MAP,
    SERVER_EXE,
    _assign_to_job,
    _llm_load_state,
    _lock,
    _server_configs,
    _server_ports,
    _server_processes,
    _server_roles,
    _win32_job,
)
from src.gguf.streaming import stream_gguf_chat

__all__ = [
    "CACHE_TYPE_MAP",
    "SERVER_EXE",
    "_assign_to_job",
    "_llm_load_state",
    "_lock",
    "_server_configs",
    "_server_ports",
    "_server_processes",
    "_server_roles",
    "_start_llm_server_sync",
    "_win32_job",
    "count_running_servers",
    "detect_model_family",
    "get_active_embedding_parallel",
    "get_active_llm_url",
    "get_gguf_embedding_url",
    "get_gguf_llm",
    "get_loaded_models",
    "get_llm_status",
    "is_server_ready",
    "kill_stray_servers",
    "preload_gguf_llm",
    "stream_gguf_chat",
    "unload_all_models",
    "unload_rag_models_safe",
]
