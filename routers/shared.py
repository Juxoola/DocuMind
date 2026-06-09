"""
Общие состояние и утилиты для роутеров.

Вынесено из main.py при рефакторинге.

Содержит:
- HTTP session (keep-alive)
- Статус загрузки (ingestion_status, upload_cancel_flags)
- Background tasks reference
- safe_filename, robust_rmtree, _schedule_delete_on_reboot
"""

import asyncio
import ctypes
import gc
import logging
import os
import shutil
import stat
import subprocess
import sys
import time
from ctypes import wintypes

import requests
import requests.adapters

import config

logger = logging.getLogger(__name__)

# ── HTTP session (keep-alive) ──
_http_session = requests.Session()
_http_session.mount(
    "http://",
    requests.adapters.HTTPAdapter(
        pool_connections=config.HTTP_POOL_SIZE_MAIN,
        pool_maxsize=config.HTTP_POOL_SIZE_MAIN,
    ),
)
_http_session.mount(
    "https://",
    requests.adapters.HTTPAdapter(
        pool_connections=config.HTTP_POOL_SIZE_MAIN,
        pool_maxsize=config.HTTP_POOL_SIZE_MAIN,
    ),
)

# ── Статус ингеста ──
# F-fix #25: ключ — уникальный task_id (UUID), а не notebook_id

ingestion_status: dict = {}
upload_cancel_flags: dict = {}
_background_tasks: "set[asyncio.Task]" = set()


def safe_filename(filename: str) -> str:
    """Валидация имени файла (path traversal, null-байты, reserved names)."""
    from fastapi import HTTPException

    if not filename or not isinstance(filename, str):
        raise HTTPException(status_code=400, detail="Пустое имя файла")
    clean = os.path.basename(filename.replace("\\", "/"))
    if (
        clean != filename
        or not clean
        or clean.startswith(".")
        or "\x00" in clean
        or any(ord(c) < 32 for c in clean)
        or clean.upper()
        in {"CON", "PRN", "AUX", "NUL", "COM1", "COM2", "COM3", "COM4", "LPT1", "LPT2", "LPT3"}
    ):
        raise HTTPException(status_code=400, detail=f"Недопустимое имя файла: {filename!r}")
    return clean


def _schedule_delete_on_reboot(path: str) -> None:
    """MoveFileExW с MOVEFILE_DELAY_UNTIL_REBOOT — удаление после перезагрузки."""
    if sys.platform != "win32":
        raise OSError("MoveFileExW доступен только на Windows")
    MOVEFILE_DELAY_UNTIL_REBOOT = 0x00000004
    MOVEFILE_WRITE_THROUGH = 0x00000008
    path_w = ctypes.c_wchar_p(path)
    kernel32 = ctypes.windll.kernel32
    kernel32.MoveFileExW.restype = wintypes.BOOL
    kernel32.MoveFileExW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD]
    success = kernel32.MoveFileExW(
        path_w, None, MOVEFILE_DELAY_UNTIL_REBOOT | MOVEFILE_WRITE_THROUGH
    )
    if not success:
        err = ctypes.get_last_error()
        raise OSError(f"MoveFileExW failed, WinError={err}: {ctypes.FormatError(err)}")


def robust_rmtree(path: str, max_retries: int = 10, delay: float = 1.0) -> tuple:
    """Надёжное удаление директории (shutil → cmd → rename → MoveFileExW)."""
    if not os.path.exists(path):
        return True, None

    for root, dirs, files in os.walk(path):
        for f in files:
            try:
                os.chmod(os.path.join(root, f), stat.S_IWRITE)
            except Exception:
                pass
        for d in dirs:
            try:
                os.chmod(os.path.join(root, d), stat.S_IWRITE)
            except Exception:
                pass

    # Попытка 1: shutil.rmtree с backoff
    last_err = None
    for i in range(max_retries):
        try:
            gc.collect()
            gc.collect()
            shutil.rmtree(path)
            return True, None
        except PermissionError as e:
            last_err = e
            if i < max_retries - 1:
                time.sleep(delay + i * 0.5)
        except Exception as e:
            last_err = e
            if i < max_retries - 1:
                time.sleep(delay + i * 0.5)

    # Попытка 2: cmd.exe rmdir
    if sys.platform == "win32":
        try:
            result = subprocess.run(
                ["cmd.exe", "/c", "rmdir", "/s", "/q", path],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if not os.path.exists(path):
                return True, None
        except Exception:
            pass

    # Попытка 3: soft-delete (rename в .pending_delete_<ts>)
    ts = int(time.time())
    deferred = f"{path}.pending_delete_{ts}"
    try:
        os.rename(path, deferred)
        return True, None
    except Exception:
        if sys.platform == "win32":
            try:
                subprocess.run(
                    ["cmd.exe", "/c", "move", path, deferred],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if not os.path.exists(path):
                    return True, None
            except Exception:
                pass

        # Попытка 4: MoveFileExW с DELAY_UNTIL_REBOOT
        if sys.platform == "win32":
            try:
                _schedule_delete_on_reboot(path)
                return True, None
            except Exception:
                pass

        err_msg = (
            f"Не удалось удалить {path}: {last_err}. "
            f"Вероятно, процесс (ChromaDB/HNSW) держит mmap-дескриптор."
        )
        return False, err_msg
