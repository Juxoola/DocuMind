"""Глобальное состояние GGUF-подсистемы."""

from __future__ import annotations

import ctypes
import logging
import os
import subprocess
import threading
from ctypes import wintypes

import config

logger = logging.getLogger(__name__)

SERVER_EXE = os.path.join(config.BASE_DIR, "bin", "llama-server.exe")

_server_processes: dict[str, subprocess.Popen] = {}
_server_ports: dict[str, int] = {}
_server_configs: dict[str, dict] = {}
_server_roles: dict[str, str] = {}
_lock = threading.Lock()

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

_win32_job = None
# Win32 Job Object: привязывает llama-server к job, чтобы ОС убила при аварии родителя
if os.name == "nt":
    try:
        _win32_job = ctypes.windll.kernel32.CreateJobObjectW(None, None)
        limit_info = (wintypes.DWORD * 36)()
        limit_info[4] = 0x2000
        ctypes.windll.kernel32.SetInformationJobObject(
            _win32_job, 9, ctypes.byref(limit_info), ctypes.sizeof(limit_info)
        )
    except Exception as e:
        logger.error(f"[GGUF] Ошибка инициализации Windows Job Object: {e}")


def _assign_to_job(process):

    if _win32_job is not None:
        try:
            ctypes.windll.kernel32.AssignProcessToJobObject(
                _win32_job, wintypes.HANDLE(int(process._handle))
            )
        except Exception as e:
            logger.debug(f"AssignProcessToJobObject failed (non-critical): {e}")


# Маппер KV-кеша: числовые коды llama.cpp → строковые флаги
CACHE_TYPE_MAP = {
    0: "f16",
    1: "f32",
    2: "q4_0",
    3: "q4_1",
    4: "q4_0",
    6: "q5_0",
    8: "q8_0",
}

_GGUF_CACHE_FILE = os.path.join(config.BASE_DIR, "_gguf_scan_cache.json")
_GGUF_CACHE_TTL_SEC = 300.0
_gguf_cache_lock = threading.Lock()
