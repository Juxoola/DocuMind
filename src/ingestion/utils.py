"""Утилиты ингеста: безопасный вывод, GPU cleanup, subprocess registry, исключения."""

import gc
import inspect as _inspect_module
import logging
import os
import threading

import torch

logger = logging.getLogger(__name__)


# Патчинг inspect.getmodule для подавления ошибок + настройка DLL-директории torch
_orig_getmodule = _inspect_module.getmodule


def _safe_getmodule(obj, filename=None):
    try:
        return _orig_getmodule(obj, filename)
    except Exception:
        return None


_inspect_module.getmodule = _safe_getmodule

try:
    lib_dir = os.path.join(os.path.dirname(torch.__file__), "lib")
    if os.path.exists(lib_dir):
        os.add_dll_directory(lib_dir)
except Exception:
    pass


# Реестр дочерних процессов для отслеживания и завершения
_active_subprocesses: dict = {}
_subprocesses_lock = threading.Lock()


def register_subprocess(notebook_id, popen):
    with _subprocesses_lock:
        _active_subprocesses.setdefault(notebook_id, []).append(popen)


def unregister_subprocess(notebook_id, popen):
    with _subprocesses_lock:
        lst = _active_subprocesses.get(notebook_id)
        if lst and popen in lst:
            try:
                lst.remove(popen)
            except Exception as e:
                logger.debug(f"unregister_subprocess: popen already removed: {e}")
        if lst is not None and not lst:
            _active_subprocesses.pop(notebook_id, None)


# Принудительное завершение процессов блокнота при отмене
def kill_subprocesses(notebook_id):
    with _subprocesses_lock:
        procs = _active_subprocesses.pop(notebook_id, [])
    for p in procs:
        try:
            if p.poll() is None:
                p.kill()
        except Exception:
            pass
    return len(procs)


# Исключение отмены операции ингестации
class IngestionCancelled(Exception):
    pass


# Очистка видеопамяти перед тяжёлыми задачами
def cleanup_gpu():

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def format_seconds(s):
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = int(s % 60)
    return f"{h}:{m:02d}:{sec:02d}" if h > 0 else f"{m}:{sec:02d}"
