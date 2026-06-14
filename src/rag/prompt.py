"""Построение контекста, промптов и получение URL эмбеддинг-сервера."""

import logging
import os

import config
from src.rag.prompts import get_system_prompt
from src.rag.state import _model_cache

logger = logging.getLogger(__name__)


def build_file_context(nodes, notebook_id: str):
    paths = config.get_notebook_paths(notebook_id)

    sources = []
    context_parts = []

    for i, node in enumerate(nodes, 1):
        meta = node.node.metadata
        fname = meta.get("file_name", "Неизвестный источник")
        img_path = meta.get("image_path", None)
        img_url = (
            f"/files/{notebook_id}/images/" + os.path.basename(img_path)
            if img_path and os.path.exists(img_path)
            else None
        )
        text = node.node.get_content()

        page_str = ""
        if meta.get("page") not in (None, ""):
            page_str = f", стр. {meta['page']}"
        elif meta.get("start") not in (None, ""):
            page_str = f", @{meta['start']}"

        sources.append(
            {
                "id": i,
                "file_name": fname,
                "text": text,
                "image_url": img_url,
                "page": meta.get("page"),
                "time": meta.get("start") or meta.get("time"),
            }
        )
        context_parts.append(f"[{i}] Файл «{fname}»{page_str}:\n{text}")

    context_str = "\n\n" + ("=" * 40 + "\n\n").join(context_parts)
    return sources, context_str


# Формирует финальный промпт: системный промпт (с правилами ответа
# и цитирования), контекст из источников и вопрос пользователя.
def make_prompt(
    query: str,
    context_str: str,
    thinking_mode: bool = False,
    max_tokens: int = 1024,
    answer_mode: str = None,
) -> str:
    return (
        get_system_prompt(answer_mode)
        + "\n"
        + f"Доступные источники:\n{context_str}\n\n"
        + f"Вопрос пользователя: {query}\n\n"
        + "Ответ:"
    )


def get_embedding_url() -> str | None:
    global _model_cache
    embed = _model_cache.get("embed_model")
    if embed is not None:
        try:
            return embed.api_base
        except Exception:
            return None
    reranker = _model_cache.get("reranker")
    if reranker is not None:
        return reranker
    return None
