"""Семантический и Sentence-сплиттер для нарезки текста на чанки."""

import logging
import threading

logger = logging.getLogger(__name__)

_SPLITTER_LOCK = threading.Lock()
_splitter_cache = None


def _get_splitter():
    global _splitter_cache

    if _splitter_cache is not None:
        return _splitter_cache

    with _SPLITTER_LOCK:
        if _splitter_cache is not None:
            return _splitter_cache

        try:
            from llama_index.core.node_parser import SemanticSplitterNodeParser
            from llama_index.embeddings.openai import OpenAIEmbedding

            from src.rag.prompt import get_embedding_url

            url = get_embedding_url()
            if url:
                embed_model = OpenAIEmbedding(
                    api_base=url,
                    api_key="lm-studio",
                    model="text-embedding-ada-002",
                )
                _splitter_cache = SemanticSplitterNodeParser(
                    embed_model=embed_model,
                    buffer_size=1,
                    breakpoint_percentile_threshold=95,
                )
                logger.info(f"[Splitter] SemanticSplitterNodeParser (embedding: {url})")
                return _splitter_cache
        except Exception as e:
            logger.warning(
                f"[Splitter] SemanticSplitter недоступен, fallback на SentenceSplitter: {e}"
            )

        from llama_index.core.node_parser import SentenceSplitter

        _splitter_cache = SentenceSplitter(chunk_size=2048, chunk_overlap=256)
        return _splitter_cache


def _invalidate_splitter_cache():
    global _splitter_cache
    with _SPLITTER_LOCK:
        _splitter_cache = None
