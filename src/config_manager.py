"""Менеджер конфигурации RAG: async чтение/запись rag_config.json."""

import logging
import os

import aiofiles
import orjson

from config import _config_lock

logger = logging.getLogger(__name__)


async def save_rag_config(rag_config_file: str, config_data: dict) -> None:
    with _config_lock:
        try:
            async with aiofiles.open(rag_config_file, "wb") as f:
                await f.write(orjson.dumps(config_data, option=orjson.OPT_INDENT_2))
        except Exception as e:
            logger.warning(f"Не удалось сохранить RAG config: {e}")


async def load_rag_config(rag_config_file: str, defaults: dict) -> dict:
    with _config_lock:
        try:
            if os.path.exists(rag_config_file):
                async with aiofiles.open(rag_config_file, "rb") as f:
                    data = orjson.loads(await f.read())
                    result = {}
                    for key, default_val in defaults.items():
                        val = data.get(key, default_val)
                        if key == "rerank_score_threshold":
                            val = float(val)
                        result[key] = val
                    return result
        except Exception as e:
            logger.warning(f"Не удалось загрузить RAG config: {e}")
        return dict(defaults)
