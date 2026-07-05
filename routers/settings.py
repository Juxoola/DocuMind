"""Роутер: настройки RAG и GGUF."""

import logging

from fastapi import APIRouter
from pydantic import BaseModel, Field, field_validator

import config
from src.config_manager import save_rag_config

logger = logging.getLogger(__name__)
router = APIRouter(tags=["settings"])


# ── Конфигурация GGUF: пути поиска и параметры по умолчанию ──
@router.get("/api/gguf-config")
async def api_get_gguf_config() -> dict:
    return {
        "search_dirs": config.rag.gguf_search_dirs,
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
        return v


@router.post("/api/update-model-dirs")
async def update_model_dirs(req: UpdateModelDirsRequest):
    with config._config_lock:
        config.rag.gguf_search_dirs = req.dirs
        data = config._collect_rag_config()
    await save_rag_config(config.RAG_CONFIG_FILE, data)
    config.invalidate_model_cache()
    try:
        from src.gguf.scanner import invalidate_scan_cache

        await invalidate_scan_cache()
    except Exception:
        logger.debug("settings: не удалось инвалидировать кэш сканирования")
    return {"status": "ok", "new_dirs": config.rag.gguf_search_dirs}


# ── Конфигурация RAG: модели, параметры поиска ──
@router.get("/api/rag-config")
async def get_rag_config():
    return {
        "embedding_model": config.rag.embedding_model,
        "reranker_model": config.rag.reranker_model,
        "embedding_n_parallel": config.rag.embedding_n_parallel,
        "top_k_per_file": config.rag.top_k_per_file,
        "rerank_pool": config.rag.rerank_pool,
        "final_top_n": config.rag.final_top_n,
        "use_reranker": config.rag.use_reranker,
        "query_expansion": config.rag.query_expansion,
        "rerank_score_threshold": config.rag.rerank_score_threshold,
        "surya_mode": config.rag.surya_mode,
    }


# Валидация параметров конфигурации RAG
class UpdateRagConfigRequest(BaseModel):
    embedding_model: str = Field(default="")
    reranker_model: str = Field(default="")
    embedding_n_parallel: int = Field(default=2, ge=1, le=8)
    top_k_per_file: int = Field(default=5, ge=1, le=100)
    rerank_pool: int = Field(default=30, ge=1, le=200)
    final_top_n: int = Field(default=10, ge=1, le=50)
    use_reranker: bool = True
    query_expansion: bool = True
    rerank_score_threshold: float = Field(default=0.1, ge=0.0, le=1.0)
    surya_mode: str = Field(default="layout_only")

    @field_validator("embedding_model", "reranker_model")
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        v = v.strip()
        if ".." in v:
            raise ValueError("путь не должен содержать '..'")
        if not v:
            raise ValueError("имя модели не может быть пустым")
        return v

    @field_validator("surya_mode")
    @classmethod
    def validate_surya_mode(cls, v: str) -> str:
        if v not in ("disabled", "layout_only", "full"):
            raise ValueError("surya_mode должен быть disabled, layout_only или full")
        return v


# ── Применение новой конфигурации RAG с перезагрузкой моделей ──
@router.post("/api/update-rag-config")
async def update_rag_config(req: UpdateRagConfigRequest) -> dict:
    from src.rag.models import preload_all_models, unload_rag_models

    old_embedding = config.rag.embedding_model
    old_reranker = config.rag.reranker_model

    with config._config_lock:
        config.update_rag_config(req.model_dump())
        data = config._collect_rag_config()
    await save_rag_config(config.RAG_CONFIG_FILE, data)

    embedding_changed = req.embedding_model and old_embedding != req.embedding_model
    reranker_changed = req.reranker_model and old_reranker != req.reranker_model

    if embedding_changed or reranker_changed:
        logger.info("[Settings] Модели изменились — выгрузка старых и загрузка новых...")
        try:
            await preload_all_models()
            logger.info("[Settings] Новые модели загружены.")
        except Exception as e:
            logger.warning(f"[Settings] Ошибка предзагрузки моделей: {e}")
    else:
        unload_rag_models(hard=True)
        config.invalidate_model_cache()

    return {"status": "ok"}
