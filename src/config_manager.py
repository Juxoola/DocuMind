"""Менеджер конфигурации RAG: sync + async чтение/запись rag_config.json."""

import json
import logging
import os
import threading
from asyncio import Lock as AsyncLock

import aiofiles
import orjson

logger = logging.getLogger(__name__)

_config_lock = threading.RLock()
_async_config_lock = AsyncLock()


# ── Sync версии (для инициализации при import config.py) ──


def save_rag_config_sync(rag_config_file: str, config_data: dict) -> None:
    with _config_lock:
        try:
            with open(rag_config_file, "w", encoding="utf-8") as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Не удалось сохранить RAG config: {e}")


def load_rag_config_sync(rag_config_file: str, defaults: dict) -> dict:
    with _config_lock:
        try:
            if os.path.exists(rag_config_file):
                with open(rag_config_file, encoding="utf-8") as f:
                    data = json.load(f)
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


# ── Async версии (для роутеров) ──


async def save_rag_config(rag_config_file: str, config_data: dict) -> None:
    async with _async_config_lock:
        try:
            async with aiofiles.open(rag_config_file, "wb") as f:
                await f.write(orjson.dumps(config_data, option=orjson.OPT_INDENT_2))
        except Exception as e:
            logger.warning(f"Не удалось сохранить RAG config: {e}")


async def load_rag_config(rag_config_file: str, defaults: dict) -> dict:
    async with _async_config_lock:
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
