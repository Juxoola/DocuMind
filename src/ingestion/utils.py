"""Утилиты ингеста: безопасный вывод, GPU cleanup, subprocess registry, исключения."""

import gc
import inspect as _inspect_module
import logging
import os
import sys
import threading
import warnings

import requests
import requests.adapters
import torch

import config

logger = logging.getLogger(__name__)

# Подавляем шумные предупреждения
warnings.filterwarnings("ignore", message="Module 'speechbrain")
warnings.filterwarnings("ignore", message="torchcodec is not installed")
warnings.filterwarnings("ignore", message="TensorFloat-32")
warnings.filterwarnings("ignore", message=".*speechbrain.*deprecated", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Lightning automatically upgraded.*")

logging.getLogger("lightning.pytorch.utilities.migration").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
logging.getLogger("whisperx").setLevel(logging.WARNING)

# Фикс: inspect.stack()
_orig_getmodule = _inspect_module.getmodule


def _safe_getmodule(obj, filename=None):
    try:
        return _orig_getmodule(obj, filename)
    except Exception:
        return None


_inspect_module.getmodule = _safe_getmodule

# Фикс для Windows DLL
try:
    lib_dir = os.path.join(os.path.dirname(torch.__file__), "lib")
    if os.path.exists(lib_dir):
        os.add_dll_directory(lib_dir)
except Exception:
    pass  # best-effort

# HTTP session (keep-alive)
_http_session = requests.Session()
_http_session.mount(
    "http://",
    requests.adapters.HTTPAdapter(
        pool_connections=config.HTTP_POOL_SIZE_INGEST,
        pool_maxsize=config.HTTP_POOL_SIZE_INGEST,
    ),
)
_http_session.mount(
    "https://",
    requests.adapters.HTTPAdapter(
        pool_connections=config.HTTP_POOL_SIZE_INGEST,
        pool_maxsize=config.HTTP_POOL_SIZE_INGEST,
    ),
)

# Subprocess registry
_active_subprocesses: dict = {}


def register_subprocess(notebook_id, popen):
    _active_subprocesses.setdefault(notebook_id, []).append(popen)


def unregister_subprocess(notebook_id, popen):
    lst = _active_subprocesses.get(notebook_id)
    if lst and popen in lst:
        try:
            lst.remove(popen)
        except Exception as e:
            logger.debug(f"unregister_subprocess: popen already removed: {e}")
    if lst is not None and not lst:
        _active_subprocesses.pop(notebook_id, None)


def kill_subprocesses(notebook_id):
    """Убивает все subprocess-ы блокнота."""
    procs = _active_subprocesses.pop(notebook_id, [])
    for p in procs:
        try:
            if p.poll() is None:
                p.terminate()
        except Exception:
            pass  # best-effort
    return len(procs)


# Исключение


class IngestionCancelled(Exception):
    """Поднимается, когда пользователь запросил отмену обработки файла."""
    pass


# Утилиты


def _safe_print(msg):
    """Безопасный вывод с защитой от cp1251."""
    try:
        logger.info(msg)
    except UnicodeEncodeError:
        try:
            sys.stdout.buffer.write((str(msg) + "\n").encode("utf-8", errors="replace"))
            sys.stdout.buffer.flush()
        except Exception:
            logger.error(msg.encode("ascii", errors="replace").decode("ascii"))


def cleanup_gpu():
    """Принудительная очистка видеопамяти перед тяжелыми задачами."""
    try:
        from src.rag_pipeline import unload_rag_models
        unload_rag_models(hard=False)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("[GPU] Память полностью очищена для анализа.")
    except Exception as e:
        logger.error(f"[GPU] Ошибка при очистке: {e}")


def format_seconds(s):
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = int(s % 60)
    return f"{h}:{m:02d}:{sec:02d}" if h > 0 else f"{m}:{sec:02d}"
