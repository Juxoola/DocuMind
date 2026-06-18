"""Менеджер конфигурации RAG: чтение/запись rag_config.json."""

import json
import logging
import os
import threading

logger = logging.getLogger(__name__)

_config_lock = threading.RLock()


def save_rag_config(rag_config_file: str, config_data: dict) -> None:
    with _config_lock:
        try:
            with open(rag_config_file, "w", encoding="utf-8") as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Не удалось сохранить RAG config: {e}")


def load_rag_config(rag_config_file: str, defaults: dict) -> dict:
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
