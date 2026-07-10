"""Менеджер конфигурации RAG: async чтение/запись rag_config.json."""

# ── Импорты ──
import asyncio
import logging

import aiofiles
import orjson

logger = logging.getLogger(__name__)

_config_lock = asyncio.Lock()


# ── Сохранение конфигурации RAG в JSON-файл ──
async def save_rag_config(rag_config_file: str, config_data: dict) -> None:
    async with _config_lock:
        try:
            async with aiofiles.open(rag_config_file, "wb") as f:
                await f.write(orjson.dumps(config_data, option=orjson.OPT_INDENT_2))
        except Exception as e:
            logger.warning(f"Не удалось сохранить RAG config: {e}")


# ── Загрузка конфигурации RAG с мержем значений по умолчанию ──
async def load_rag_config(rag_config_file: str, defaults: dict) -> dict:
    async with _config_lock:
        try:
            if await aiofiles.os.path.exists(rag_config_file):
                async with aiofiles.open(rag_config_file, "rb") as f:
                    data = orjson.loads(await f.read())
                    result = {}
                    for key, default_val in defaults.items():
                        val = data.get(key, default_val)
                        if key == "rerank_score_threshold":
                            try:
                                val = float(val)
                            except (TypeError, ValueError):
                                val = float(default_val)
                        result[key] = val
                    return result
        except Exception as e:
            logger.warning(f"Не удалось загрузить RAG config: {e}")
    return dict(defaults)
