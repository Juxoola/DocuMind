"""Инициализация и lifecycle RAG-моделей: embedding, reranker, LLM."""

import asyncio
import logging

import torch
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI

import config
from src.rag.state import _model_cache

_init_lock = asyncio.Lock()
_model_cache_lock = asyncio.Lock()

# Кэш последнего max_tokens: пропускает пересоздание Settings.llm при совпадении
_last_max_tokens: int = 0

logger = logging.getLogger(__name__)


# Инициализация глобальных Settings (embedding-модель + LLM) с кэшированием
async def init_settings(max_tokens=1024):
    global _model_cache, _last_max_tokens

    async with _model_cache_lock:
        embed_ready = "embed_model" in _model_cache
        tokens_match = max_tokens == _last_max_tokens
    if embed_ready and tokens_match:
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"

    async with _init_lock:
        async with _model_cache_lock:
            if "embed_model" not in _model_cache:
                await _init_embed_model()
            Settings.embed_model = _model_cache["embed_model"]

    _last_max_tokens = max_tokens
    Settings.llm = OpenAI(
        api_base=config.LM_STUDIO_URL,
        api_key=config.LLM_DEFAULT_API_KEY,
        model=config.LLM_DEFAULT_MODEL,
        temperature=config.CHAT_TEMPERATURE,
        max_tokens=max_tokens,
    )


# Загрузка GGUF embedding-модели через локальный сервер с автоопределением parallelism
async def _init_embed_model():
    model_name = config.EMBEDDING_MODEL_NAME

    if not config.validate_gguf_path(model_name):
        raise RuntimeError(
            "Поддерживаются только GGUF-модели эмбеддингов. "
            "Укажите путь к .gguf файлу в config.EMBEDDING_MODEL_NAME.\n"
            f"Текущее значение: {model_name}"
        )
    logger.info(f"Инициализация GGUF эмбеддингов: {model_name}")
    from llama_index.embeddings.openai import OpenAIEmbedding

    from src.gguf.server import get_gguf_embedding_url

    model_path = config.resolve_model_path(model_name)
    url = await get_gguf_embedding_url(model_path, n_parallel=config.EMBEDDING_N_PARALLEL)
    try:
        from src.gguf.server import get_active_embedding_parallel

        n_parallel = await get_active_embedding_parallel(model_path)
    except Exception:
        n_parallel = 1
    logger.info(
        f"[RAG] GGUF embedding server --parallel={n_parallel} → embed_batch_size={n_parallel}"
    )

    _model_cache["embed_model"] = OpenAIEmbedding(
        api_base=f"{url}/v1",
        api_key=config.EMBEDDING_DEFAULT_API_KEY,
        model=config.EMBEDDING_DEFAULT_MODEL,
        timeout=120.0,
        embed_batch_size=n_parallel,
        query_header=(
            "Instruct: Given a web search query, retrieve relevant passages "
            "that answer the query\nQuery: "
        ),
    )


# Предзагрузка embedding и реранкера при старте приложения (lazily для LLM)
async def preload_all_models():
    logger.info("[RAG] Предзагрузка моделей...")
    try:
        await init_settings()
    except Exception as e:
        logger.warning(f"  [RAG] ⚠ Эмбеддинги не загружены (будут загружены lazily): {e}")
    if config.RERANKER_MODEL_NAME:
        try:
            if not config.validate_gguf_path(config.RERANKER_MODEL_NAME):
                logger.warning(
                    f"  [RAG] ⚠ Реранкер пропущен: неверный формат ({config.RERANKER_MODEL_NAME})"
                )
            else:
                logger.info(f"  [RAG] Предзагрузка GGUF реранкера: {config.RERANKER_MODEL_NAME}")
                from src.gguf.server import get_gguf_embedding_url

                model_path = config.resolve_model_path(config.RERANKER_MODEL_NAME)
                await get_gguf_embedding_url(model_path, is_reranker=True)
        except Exception as e:
            logger.warning(f"  [RAG] ⚠ Реранкер не загружен (будет загружен lazily): {e}")

    logger.info("[RAG] Предзагрузка завершена.")


# Выгрузка моделей и очистка GPU-памяти (hard — полная, soft — без эмбеддингов)
def unload_rag_models(hard=True):
    global _model_cache
    if not _model_cache:
        return

    if hard:
        logger.info("[RAG] Выгрузка всех моделей (Embedding, Reranker)...")
        _model_cache.clear()
    else:
        logger.info("[RAG] Мягкая очистка (Эмбеддинги и Реранкер остаются в памяти)...")

    from src.ingestion.utils import cleanup_gpu

    cleanup_gpu()
    logger.info("[RAG] Память очищена.")
