"""Сканирование и кеширование GGUF-файлов в поисковых директориях.

Вынесено из gguf_manager.py при рефакторинге.
"""

# Файл: scanner.py — рекурсивный поиск .gguf и .mmproj файлов в
# директориях из config.GGUF_SEARCH_DIRS. Результаты кешируются
# в JSON с проверкой по TTL и mtime корневых папок.

import json
import logging
import os
import threading
import time

import config
from src.gguf.state import _GGUF_CACHE_FILE, _GGUF_CACHE_TTL_SEC, _gguf_cache_lock

logger = logging.getLogger(__name__)


def _dir_mtime(root: str) -> float:

    try:
        st = os.stat(root)
        return st.st_mtime
    except OSError:
        return 0.0


def _scan_gguf_dirs_uncached() -> list[dict]:

    # Реальный обход файловой системы: для каждой корневой директории
    # из конфига os.walk ищет .gguf файлы, разделяя модели (.gguf)
    # и проекционные файлы (.mmproj.gguf).
    results = []
    search_dirs = [d.strip() for d in config.GGUF_SEARCH_DIRS.split(";") if d.strip()]

    for base_dir in search_dirs:
        if not os.path.exists(base_dir):
            continue
        for dirpath, dirnames, filenames in os.walk(base_dir):
            gguf_files = sorted(
                [
                    f
                    for f in filenames
                    if f.lower().endswith(".gguf")
                    and not any(x in f.lower() for x in [".mmproj", ".proj"])
                ]
            )
            mmproj_files = sorted(
                [
                    f
                    for f in filenames
                    if f.lower().endswith(".gguf")
                    and any(x in f.lower() for x in [".mmproj", ".proj"])
                ]
            )
            if gguf_files or mmproj_files:
                results.append(
                    {
                        "dir": dirpath,
                        "dir_name": os.path.basename(dirpath),
                        "gguf_files": gguf_files,
                        "mmproj_files": mmproj_files,
                    }
                )
    return results


def scan_gguf_dirs() -> list[dict]:

    # Сначала пытаемся загрузить закешированный результат.
    # Если кеш свежий (TTL < 300 с) и mtime корней не изменились —
    # возвращаем сохранённые данные, иначе делаем реальный обход.
    with _gguf_cache_lock:
        cached = None
        try:
            if os.path.exists(_GGUF_CACHE_FILE):
                with open(_GGUF_CACHE_FILE, encoding="utf-8") as f:
                    cached = json.load(f)
        except Exception:
            cached = None

        if cached:
            try:
                saved_at = float(cached.get("saved_at", 0))
                age = time.time() - saved_at
                cached_mtimes = cached.get("dir_mtimes", {}) or {}
                roots = [d.strip() for d in config.GGUF_SEARCH_DIRS.split(";") if d.strip()]
                roots_valid = all(
                    cached_mtimes.get(r) == _dir_mtime(r) for r in roots
                ) and len(cached_mtimes) == len(roots)
                if age < _GGUF_CACHE_TTL_SEC and roots_valid:
                    return cached.get("results", [])
            except Exception:
                pass

        results = _scan_gguf_dirs_uncached()
        try:
            roots = [d.strip() for d in config.GGUF_SEARCH_DIRS.split(";") if d.strip()]
            payload = {
                "saved_at": time.time(),
                "dir_mtimes": {r: _dir_mtime(r) for r in roots},
                "results": results,
            }
            # Атомарная запись: сначала .tmp, затем os.replace —
            # чтобы не повредить кеш при сбое в середине записи.
            tmp = _GGUF_CACHE_FILE + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
            os.replace(tmp, _GGUF_CACHE_FILE)
        except Exception as e:
            logger.warning(f"не удалось сохранить scan cache: {e}")
        return results


def invalidate_scan_cache():

    with _gguf_cache_lock:
        try:
            if os.path.exists(_GGUF_CACHE_FILE):
                os.remove(_GGUF_CACHE_FILE)
        except Exception:
            pass


def find_gguf_by_name(filename: str) -> str | None:

    if not filename:
        return None
    name = os.path.basename(filename)
    for entry in scan_gguf_dirs():
        for f in (entry.get("gguf_files") or []) + (entry.get("mmproj_files") or []):
            if f == name:
                return os.path.join(entry["dir"], name)
    return None
