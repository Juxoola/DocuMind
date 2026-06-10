"""Инициализация, предзагрузка и выгрузка RAG-моделей (embedding, reranker, LLM)."""

# Загрузка GGUF-моделей эмбеддингов и реранкера через
# локальный сервер (gguf_direct), настройка llama_index Settings.

import gc
import logging

import torch
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI

import config
from src.rag.state import _init_lock, _model_cache

logger = logging.getLogger(__name__)


def init_settings(max_tokens=1024):
    global _model_cache
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with _init_lock:
        # Инициализация эмбеддингов однократно: проверяем формат GGUF,
        # запускаем локальный сервер через gguf_direct и создаём OpenAIEmbedding-клиент
        if "embed_model" not in _model_cache:
            model_name = config.EMBEDDING_MODEL_NAME

            if not config.validate_gguf_path(model_name):
                raise RuntimeError(
                    "Поддерживаются только GGUF-модели эмбеддингов. "
                    "Укажите путь к .gguf файлу в config.EMBEDDING_MODEL_NAME.\n"
                    f"Текущее значение: {model_name}"
                )
            logger.info(f"Инициализация GGUF эмбеддингов: {model_name}")
            from llama_index.embeddings.openai import OpenAIEmbedding

            from src.gguf_direct import get_gguf_embedding_url

            model_path = config.resolve_model_path(model_name)
            url = get_gguf_embedding_url(model_path, n_parallel=config.EMBEDDING_N_PARALLEL)
            try:
                from src.gguf_direct import get_active_embedding_parallel

                n_parallel = get_active_embedding_parallel(model_path)
            except Exception:
                n_parallel = 1
            logger.info(
                f"[RAG] GGUF embedding server --parallel={n_parallel}"
                f" → embed_batch_size={n_parallel}"
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

    Settings.embed_model = _model_cache["embed_model"]

    # LLM (chat) всегда инициализируется заново — max_tokens может
    # меняться между запросами. Подключается к LM Studio или GGUF-серверу.
    Settings.llm = OpenAI(
        api_base=config.LM_STUDIO_URL,
        api_key=config.LLM_DEFAULT_API_KEY,
        model=config.LLM_DEFAULT_MODEL,
        temperature=config.CHAT_TEMPERATURE,
        max_tokens=max_tokens,
    )


# Предзагрузка всех моделей при старте приложения.
# Игнорирует ошибки — если что-то не загрузилось, подгрузится lazily.
def preload_all_models():
    logger.info("[RAG] Предзагрузка моделей...")
    try:
        init_settings()
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
                from src.gguf_direct import get_gguf_embedding_url

                model_path = config.resolve_model_path(config.RERANKER_MODEL_NAME)
                get_gguf_embedding_url(model_path, is_reranker=True)
        except Exception as e:
            logger.warning(f"  [RAG] ⚠ Реранкер не загружен (будет загружен lazily): {e}")

    logger.info("[RAG] Предзагрузка завершена.")


# Выгрузка моделей из памяти. hard=True — полная очистка кеша,
# hard=False — только сборка мусора, модели остаются в памяти процесса.
def unload_rag_models(hard=True):
    global _model_cache
    if not _model_cache:
        return

    with _init_lock:
        if hard:
            logger.info("[RAG] Выгрузка всех моделей (Embedding, Reranker)...")
            _model_cache.clear()
        else:
            logger.info("[RAG] Мягкая очистка (Эмбеддинги и Реранкер остаются в памяти)...")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("[RAG] Память очищена.")
