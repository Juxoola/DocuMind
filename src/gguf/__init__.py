"""GGUF subsystem: server lifecycle, scanning, streaming, model detection."""

from src.gguf.models import detect_model_family
from src.gguf.server import (
    count_running_servers,
    get_active_embedding_parallel,
    get_active_llm_url,
    get_gguf_embedding_url,
    get_gguf_llm,
    get_llm_status,
    get_loaded_models,
    is_server_ready,
    kill_stray_servers,
    preload_gguf_llm,
    unload_all_models,
    unload_rag_models_safe,
)
from src.gguf.state import CACHE_TYPE_MAP, SERVER_EXE
from src.gguf.streaming import stream_gguf_chat

__all__ = [
    "CACHE_TYPE_MAP",
    "SERVER_EXE",
    "count_running_servers",
    "detect_model_family",
    "get_active_embedding_parallel",
    "get_active_llm_url",
    "get_gguf_embedding_url",
    "get_gguf_llm",
    "get_llm_status",
    "get_loaded_models",
    "is_server_ready",
    "kill_stray_servers",
    "preload_gguf_llm",
    "stream_gguf_chat",
    "unload_all_models",
    "unload_rag_models_safe",
]
