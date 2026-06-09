"""Семантический и Sentence-сплиттер для нарезки текста на чанки."""

import logging

logger = logging.getLogger(__name__)

_SEMANTIC_SPLITTER_CACHE = None


def _get_splitter():
    """Возвращает SemanticSplitterNodeParser или SentenceSplitter (fallback)."""
    global _SEMANTIC_SPLITTER_CACHE
    if _SEMANTIC_SPLITTER_CACHE is not None:
        return _SEMANTIC_SPLITTER_CACHE

    try:
        from llama_index.core.node_parser import SemanticSplitterNodeParser
        from llama_index.embeddings.openai import OpenAIEmbedding

        from src.rag_pipeline import get_embedding_url

        url = get_embedding_url()
        if url:
            embed_model = OpenAIEmbedding(
                api_base=url,
                api_key="lm-studio",
                model="text-embedding-ada-002",
            )
            _SEMANTIC_SPLITTER_CACHE = SemanticSplitterNodeParser(
                embed_model=embed_model,
                buffer_size=1,
                breakpoint_percentile_threshold=95,
            )
            logger.info(f"[Splitter] SemanticSplitterNodeParser (embedding: {url})")
            return _SEMANTIC_SPLITTER_CACHE
    except Exception as e:
        logger.warning(f"[Splitter] SemanticSplitter недоступен, fallback на SentenceSplitter: {e}")

    from llama_index.core.node_parser import SentenceSplitter

    fallback = SentenceSplitter(chunk_size=2048, chunk_overlap=256)
    _SEMANTIC_SPLITTER_CACHE = fallback
    return fallback
