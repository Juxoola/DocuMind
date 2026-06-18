"""Роутер: настройки RAG и GGUF."""

import logging

from fastapi import APIRouter
from pydantic import BaseModel

import config

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


class UpdateModelDirsRequest(BaseModel):
    dirs: str


@router.post("/api/update-model-dirs")
async def update_model_dirs(req: UpdateModelDirsRequest):
    with config._config_lock:
        config.GGUF_SEARCH_DIRS = req.dirs
        config.save_rag_config()
    try:
        from src.gguf_manager import invalidate_scan_cache

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


class UpdateRagConfigRequest(BaseModel):
    embedding_model: str
    reranker_model: str
    top_k_per_file: int
    rerank_pool: int
    final_top_n: int
    use_reranker: bool


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
        config.save_rag_config()
        unload_rag_models()
    return {"status": "ok"}
