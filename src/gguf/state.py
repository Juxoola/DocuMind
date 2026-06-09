"""Глобальное состояние GGUF-подсистемы.

Объединяет глобалы из gguf_direct.py и gguf_manager.py.
"""

import ctypes
import logging
import os
import threading
from ctypes import wintypes

import config

logger = logging.getLogger(__name__)

SERVER_EXE = os.path.join(config.BASE_DIR, "bin", "llama-server.exe")

# Мультисерверное состояние
_server_processes: dict[str, "subprocess.Popen"] = {}
_server_ports: dict[str, int] = {}
_server_configs: dict[str, dict] = {}
_server_roles: dict[str, str] = {}  # gguf_path → role (llm/embedding/reranker)
_lock = threading.Lock()

# Состояние последней загрузки LLM (для UI)
_llm_load_state: dict = {
    "state": "idle",
    "model": None,
    "port": None,
    "task_id": None,
    "started_at": None,
    "ready_at": None,
    "last_load_seconds": None,
    "error": None,
    "phase": None,
}

# Windows Job Object
_win32_job = None
if os.name == "nt":
    try:
        _win32_job = ctypes.windll.kernel32.CreateJobObjectW(None, None)
        limit_info = (wintypes.DWORD * 36)()
        limit_info[4] = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        ctypes.windll.kernel32.SetInformationJobObject(
            _win32_job, 9, ctypes.byref(limit_info), ctypes.sizeof(limit_info)
        )
    except Exception as e:
        logger.error(f"[GGUF] Ошибка инициализации Windows Job Object: {e}")


def _assign_to_job(process):
    """Привязывает процесс к Job Object (только для Windows)."""
    if _win32_job is not None:
        try:
            ctypes.windll.kernel32.AssignProcessToJobObject(
                _win32_job, wintypes.HANDLE(int(process._handle))
            )
        except Exception as e:
            logger.debug(f"AssignProcessToJobObject failed (non-critical): {e}")


# CACHE_TYPE_MAP — маппинг типов квантования KV-кэша
CACHE_TYPE_MAP = {
    0: "f16",
    1: "f32",
    2: "q4_0",
    3: "q4_1",
    4: "q4_0",   # было "q4_k" — невалидно
    6: "q5_0",   # было "q5_k" — невалидно
    8: "q8_0",
}

# Persistent scan cache
_GGUF_CACHE_FILE = os.path.join(config.BASE_DIR, "_gguf_scan_cache.json")
_GGUF_CACHE_TTL_SEC = 300.0
_gguf_cache_lock = threading.Lock()
