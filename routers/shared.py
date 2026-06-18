"""Общие состояние и утилиты для роутеров."""

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

import httpx
import requests
import requests.adapters

import config

logger = logging.getLogger(__name__)


def make_http_session(pool_size: int = 10) -> requests.Session:
    s = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=pool_size, pool_maxsize=pool_size)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s


# HTTP-сессия с пулом соединений — используется всеми роутерами для внешних API-вызовов (LLM, эмбеддинги).
_http_session = make_http_session(config.HTTP_POOL_SIZE_MAIN)

# Асинхронная HTTP-сессия — для async-эндпоинтов (chat streaming, vision OCR)
_async_http: httpx.AsyncClient | None = None


def get_async_http() -> httpx.AsyncClient:
    global _async_http
    if _async_http is None or _async_http.is_closed:
        _async_http = httpx.AsyncClient(
            timeout=httpx.Timeout(config.LM_STUDIO_HTTP_TIMEOUT),
            limits=httpx.Limits(
                max_connections=config.HTTP_POOL_SIZE_MAIN,
                max_keepalive_connections=config.HTTP_POOL_SIZE_MAIN,
            ),
        )
    return _async_http


ingestion_status: dict = {}
upload_cancel_flags: dict = {}
_background_tasks: "set[asyncio.Task]" = set()
_INGESTION_STATUS_TTL_SEC = 3600  # 1 час


def _cleanup_ingestion_status():
    now = time.time()
    expired = [
        k
        for k, v in ingestion_status.items()
        if not v.get("is_uploading") and now - v.get("updated_at", 0) > _INGESTION_STATUS_TTL_SEC
    ]
    for k in expired:
        ingestion_status.pop(k, None)


def safe_filename(filename: str) -> str:
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


def robust_rmtree(path: str, max_retries: int = 3, delay: float = 0.5) -> tuple:
    if not os.path.exists(path):
        return True, None

    for root, dirs, files in os.walk(path):
        for f in files:
            try:
                os.chmod(os.path.join(root, f), stat.S_IWRITE)
            except Exception:
                logger.debug("robust_rmtree: не удалось снять readonly c %s", f)
        for d in dirs:
            try:
                os.chmod(os.path.join(root, d), stat.S_IWRITE)
            except Exception:
                logger.debug("robust_rmtree: не удалось снять readonly c %s", d)

    last_err = None
    for i in range(max_retries):
        try:
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

    if sys.platform == "win32":
        try:
            result = subprocess.run(
                ["cmd.exe", "/c", "rmdir", "/s", "/q", path],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if not os.path.exists(path):
                return True, None
        except Exception:
            logger.debug("robust_rmtree: cmd rmdir не удался для %s", path)

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
                logger.debug("robust_rmtree: cmd move не удался для %s", path)

        if sys.platform == "win32":
            try:
                _schedule_delete_on_reboot(path)
                return True, None
            except Exception:
                logger.debug("robust_rmtree: MoveFileExW не удался для %s", path)

        err_msg = (
            f"Не удалось удалить {path}: {last_err}. "
            f"Вероятно, процесс (ChromaDB/HNSW) держит mmap-дескриптор."
        )
        return False, err_msg


def safe_extract_llm_response(data: dict) -> str | None:
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        logger.warning("LLM API вернул неожиданный формат: %s", str(data)[:200])
        return None
