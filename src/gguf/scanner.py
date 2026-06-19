"""Сканирование и кеширование GGUF-файлов в поисковых директориях."""

import asyncio
import logging
import os
import time

import aiofiles
import aiofiles.os
import orjson

import config
from src.gguf.state import _GGUF_CACHE_FILE, _GGUF_CACHE_TTL_SEC, _gguf_cache_lock

logger = logging.getLogger(__name__)


async def _dir_mtime(root: str) -> float:

    try:
        st = await asyncio.to_thread(os.stat, root)
        return st.st_mtime
    except OSError:
        return 0.0


async def _scan_gguf_dirs_uncached() -> list[dict]:

    results = []
    search_dirs = [d.strip() for d in config.GGUF_SEARCH_DIRS.split(";") if d.strip()]

    for base_dir in search_dirs:
        if not await aiofiles.os.path.exists(base_dir):
            continue
        for dirpath, dirnames, filenames in await asyncio.to_thread(
            lambda: list(os.walk(base_dir))
        ):
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


async def scan_gguf_dirs() -> list[dict]:

    with _gguf_cache_lock:
        cached = None
        try:
            if await aiofiles.os.path.exists(_GGUF_CACHE_FILE):
                async with aiofiles.open(_GGUF_CACHE_FILE, encoding="utf-8") as f:
                    cached = orjson.loads(await f.read())
        except Exception:
            cached = None

        if cached:
            try:
                saved_at = float(cached.get("saved_at", 0))
                age = time.time() - saved_at
                cached_mtimes = cached.get("dir_mtimes", {}) or {}
                roots = [d.strip() for d in config.GGUF_SEARCH_DIRS.split(";") if d.strip()]
                roots_valid = all(cached_mtimes.get(r) == await _dir_mtime(r) for r in roots) and len(
                    cached_mtimes
                ) == len(roots)
                if age < _GGUF_CACHE_TTL_SEC and roots_valid:
                    return cached.get("results", [])
            except Exception:
                pass

        results = await _scan_gguf_dirs_uncached()
        try:
            roots = [d.strip() for d in config.GGUF_SEARCH_DIRS.split(";") if d.strip()]
            payload = {
                "saved_at": time.time(),
                "dir_mtimes": {r: await _dir_mtime(r) for r in roots},
                "results": results,
            }
            tmp = _GGUF_CACHE_FILE + ".tmp"
            async with aiofiles.open(tmp, "w", encoding="utf-8") as f:
                await f.write(orjson.dumps(payload).decode())
            await asyncio.to_thread(os.replace, tmp, _GGUF_CACHE_FILE)
        except Exception as e:
            logger.warning(f"не удалось сохранить scan cache: {e}")
        return results


async def invalidate_scan_cache():

    with _gguf_cache_lock:
        try:
            if await aiofiles.os.path.exists(_GGUF_CACHE_FILE):
                await aiofiles.os.remove(_GGUF_CACHE_FILE)
        except Exception:
            pass


async def find_gguf_by_name(filename: str) -> str | None:

    if not filename:
        return None
    name = os.path.basename(filename)
    for entry in await scan_gguf_dirs():
        for f in (entry.get("gguf_files") or []) + (entry.get("mmproj_files") or []):
            if f == name:
                return os.path.join(entry["dir"], name)
    return None


def find_gguf_by_name_sync(filename: str) -> str | None:
    """Sync-версия для вызова из sync-контекстов (config.resolve_model_path)."""
    if not filename:
        return None
    name = os.path.basename(filename)
    search_dirs = [d.strip() for d in config.GGUF_SEARCH_DIRS.split(";") if d.strip()]
    for base_dir in search_dirs:
        if not os.path.exists(base_dir):
            continue
        for dirpath, _dirnames, filenames in os.walk(base_dir):
            if name in filenames:
                return os.path.join(dirpath, name)
    return None
