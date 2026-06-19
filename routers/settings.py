"""Роутер: настройки RAG и GGUF."""

import asyncio
import logging
import os

from fastapi import APIRouter
from pydantic import BaseModel, Field, field_validator

import config
from src.config_manager import save_rag_config

logger = logging.getLogger(__name__)
router = APIRouter(tags=["settings"])


@router.get("/api/gguf-config")
async def api_get_gguf_config():
    return {
        "search_dirs": config.GGUF_SEARCH_DIRS,
        "default_ctx_size": config.GGUF_CTX_SIZE,
        "default_gpu_layers": config.GGUF_GPU_LAYERS,
        "default_threads": config.GGUF_THREADS,
    }


# Валидация пути поиска моделей GGUF
class UpdateModelDirsRequest(BaseModel):
    dirs: str = Field(..., min_length=1, max_length=2000)

    @field_validator("dirs")
    @classmethod
    def validate_dirs(cls, v: str) -> str:
        parts = [p.strip() for p in v.split(";")]
        parts = [p for p in parts if p]
        if len(parts) > 20:
            raise ValueError("максимум 20 каталогов")
        for part in parts:
            if ".." in part:
                raise ValueError(f"запрещён path traversal: {part}")
            if os.path.isabs(part):
                raise ValueError(f"абсолютные пути запрещены: {part}")
        return v


@router.post("/api/update-model-dirs")
async def update_model_dirs(req: UpdateModelDirsRequest):
    with config._config_lock:
        config.GGUF_SEARCH_DIRS = req.dirs
        data = config._collect_rag_config()
    await save_rag_config(config.RAG_CONFIG_FILE, data)
    config.resolve_model_path.cache_clear()
    try:
        from src.gguf.scanner import invalidate_scan_cache

        invalidate_scan_cache()
    except Exception:
        logger.debug("settings: не удалось инвалидировать кэш сканирования")
    return {"status": "ok", "new_dirs": config.GGUF_SEARCH_DIRS}


@router.get("/api/rag-config")
async def get_rag_config():
    return {
        "embedding_model": config.EMBEDDING_MODEL_NAME,
        "reranker_model": config.RERANKER_MODEL_NAME,
        "top_k_per_file": config.RAG_TOP_K_PER_FILE,
        "rerank_pool": config.RAG_RERANK_POOL,
        "final_top_n": config.RAG_FINAL_TOP_N,
        "use_reranker": config.USE_RERANKER,
    }


# Валидация параметров конфигурации RAG
class UpdateRagConfigRequest(BaseModel):
    embedding_model: str = Field(..., min_length=1, max_length=256)
    reranker_model: str = Field(..., min_length=1, max_length=256)
    top_k_per_file: int = Field(..., ge=1, le=100)
    rerank_pool: int = Field(..., ge=1, le=200)
    final_top_n: int = Field(..., ge=1, le=50)
    use_reranker: bool

    @field_validator("embedding_model", "reranker_model")
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        if "/" in v:
            raise ValueError("имя модели не должно содержать '/'")
        if ".." in v:
            raise ValueError("имя модели не должно содержать '..'")
        if os.path.isabs(v):
            raise ValueError("абсолютные пути запрещены")
        return v


@router.post("/api/update-rag-config")
async def update_rag_config(req: UpdateRagConfigRequest):
    from src.rag.models import unload_rag_models

    with config._config_lock:
        config.EMBEDDING_MODEL_NAME = req.embedding_model
        config.RERANKER_MODEL_NAME = req.reranker_model
        config.RAG_TOP_K_PER_FILE = req.top_k_per_file
        config.RAG_RERANK_POOL = req.rerank_pool
        config.RAG_FINAL_TOP_N = req.final_top_n
        config.USE_RERANKER = req.use_reranker
        data = config._collect_rag_config()
    await save_rag_config(config.RAG_CONFIG_FILE, data)
    await asyncio.to_thread(unload_rag_models)
    config.resolve_model_path.cache_clear()
    return {"status": "ok"}
