"""Trampoline для обратной совместимости.

Код менеджера GGUF переехал в пакет src/gguf/. Этот файл реэкспортирует
все публичные и приватные символы, чтобы старые импорты
  from src.gguf_manager import ...
продолжали работать без изменений.
"""
#
# Файл: gguf_manager.py — trampoline + legacy-функции.
# Кроме реэкспорта из src/gguf/, содержит устаревшие standalone-функции
# (start_gguf_server, stop_gguf_server) для обратной совместимости.
#

import logging
import os

import config
from src.gguf.scanner import (
    _dir_mtime,
    _scan_gguf_dirs_uncached,
    find_gguf_by_name,
    invalidate_scan_cache,
    scan_gguf_dirs,
)
from src.gguf.state import (
    SERVER_EXE,
    _gguf_cache_lock,
)

logger = logging.getLogger(__name__)

_server_process = None
_server_info = {}


# Legacy-функция: проверяет, жив ли старый standalone-сервер (не из пула src/gguf/).
def get_server_status() -> dict:
    global _server_process, _server_info
    if _server_process is None or _server_process.poll() is not None:
        return {"running": False, "info": {}}
    try:
        import requests

        url = f"http://127.0.0.1:{_server_info['port']}/health"
        r = requests.get(url, timeout=1)
        if r.status_code == 200:
            return {"running": True, "info": _server_info}
    except Exception:
        logger.debug("gguf_manager: health-check не удался")
    return {"running": True, "info": _server_info, "status": "initializing"}


# Legacy-функция: запускает отдельный llama-server на свободном порту (устаревший API).
def start_gguf_server(
    gguf_path: str,
    mmproj_path: str | None = None,
    ctx_size: int | None = None,
    gpu_layers: int | None = None,
    threads: int | None = None,
    mtp_enabled: bool = False,
) -> dict:
    global _server_process, _server_info
    if _server_process:
        stop_gguf_server()
    import socket as _socket
    import subprocess as _subprocess

    s = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()

    cmd = [
        SERVER_EXE,
        "-m",
        os.path.normpath(gguf_path),
        "--port",
        str(port),
        "-c",
        str(ctx_size or config.GGUF_CTX_SIZE),
        "-ngl",
        str(gpu_layers if gpu_layers is not None else config.GGUF_GPU_LAYERS),
        "-b",
        "512",
        "-ub",
        "256",
        "--parallel",
        "1",
        "--no-context-shift",
        "--jinja",
        "-n",
        "2048",
        "--flash-attn",
        "on",
    ]
    if mtp_enabled:
        cmd.extend(["--spec-type", "draft-mtp", "--spec-draft-n-max", "2"])
    if mmproj_path:
        cmd.extend(["--mmproj", os.path.normpath(mmproj_path)])

    logger.info(f"[GGUF Manager] Запуск сервера: {' '.join(cmd)}")
    _server_process = _subprocess.Popen(
        cmd,
        stdout=_subprocess.PIPE,
        stderr=_subprocess.PIPE,
        text=True,
        creationflags=0x08000000,
    )
    _server_info = {
        "port": port,
        "url": f"http://127.0.0.1:{port}",
        "model": os.path.basename(gguf_path),
    }

    for _ in range(config.GGUF_SERVER_STARTUP_TIMEOUT):
        __import__("time").sleep(1)
        try:
            import requests

            if (
                requests.get(
                    f"http://127.0.0.1:{port}/health", timeout=config.GGUF_HEALTH_CHECK_TIMEOUT
                ).status_code
                == 200
            ):
                logger.info(f"[GGUF Manager] Сервер готов на порту {port}")
                return {"status": "ok", "url": f"http://127.0.0.1:{port}/v1", "info": _server_info}
        except Exception:
            logger.debug("gguf_manager: сервер ещё не готов, ждём...")
        if _server_process.poll() is not None:
            return {"status": "error", "msg": "Процесс сервера завершился ошибкой"}
    return {"status": "error", "msg": "Таймаут запуска сервера"}


# Legacy-функция: останавливает ранее запущенный standalone-сервер.
def stop_gguf_server() -> dict:
    global _server_process, _server_info
    if _server_process:
        _server_process.terminate()
        try:
            _server_process.wait(timeout=config.GGUF_SERVER_STOP_TIMEOUT)
        except Exception:
            try:
                _server_process.kill()
            except Exception:
                pass
    _server_process = None
    _server_info = {}
    return {"status": "ok"}


# Legacy-функция: возвращает URL запущенного standalone-сервера или None.
def get_gguf_server_url() -> str | None:
    status = get_server_status()
    if status["running"]:
        return f"{status['info']['url']}/v1"
    return None


# Полный список публичного API для удобства импорта и рефакторинга.
__all__ = [
    "_dir_mtime",
    "_gguf_cache_lock",
    "_scan_gguf_dirs_uncached",
    "_server_info",
    "_server_process",
    "find_gguf_by_name",
    "get_gguf_server_url",
    "get_server_status",
    "invalidate_scan_cache",
    "scan_gguf_dirs",
    "start_gguf_server",
    "stop_gguf_server",
]
