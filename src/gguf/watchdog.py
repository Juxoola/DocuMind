"""Watchdog-мониторинг: контроль потребления памяти GGUF-серверов."""

import asyncio
import logging
import os

import psutil

from src.gguf.state import _lock, _server_configs

logger = logging.getLogger(__name__)

_watchdog_tasks: set[asyncio.Task] = set()


def _proc_alive(proc) -> bool:
    if hasattr(proc, "poll"):
        return proc.poll() is None
    return proc.returncode is None


# Лимиты RSS по ролям (MB): при превышении — auto-restart
_WATCHDOG_LIMITS = {
    "llm": 12000,
    "vision": 4000,
    "embedding": 2000,
    "reranker": 2000,
}


async def _watchdog_memory(server_key: str, process, role: str):
    # Lazy import server.py to avoid circular import
    from src.gguf.server import (
        _kill_server_process,
        _start_llm_server,
        get_gguf_embedding_url,
        get_vision_server,
    )

    limit_mb = _WATCHDOG_LIMITS.get(role, 8000)
    while _proc_alive(process):
        await asyncio.sleep(30)
        try:
            proc = psutil.Process(process.pid)
            rss_mb = proc.memory_info().rss // (1024 * 1024)
            if rss_mb > limit_mb:
                logger.warning(
                    f"[GGUF Watchdog] {role}/{os.path.basename(server_key)}: "
                    f"RSS {rss_mb}MB > {limit_mb}MB limit. Restarting..."
                )
                if role == "vision":
                    from src.ingestion.vision import set_vision_url

                    set_vision_url(None)
                await _kill_server_process(process)
                try:
                    await asyncio.wait_for(process.wait(), timeout=5)
                except Exception as e:
                    logger.debug(f"process.wait() timeout при auto-restart watchdog ({role}): {e}")
                await asyncio.sleep(2)
                async with _lock:
                    cfg = _server_configs.get(server_key)
                if cfg is None:
                    return
                try:
                    if role == "llm":
                        url = await _start_llm_server(server_key, cfg.get("mmproj"), cfg)
                    elif role in ("embedding", "reranker"):
                        url = await get_gguf_embedding_url(
                            server_key,
                            is_reranker=(role == "reranker"),
                            n_parallel=cfg.get("n_parallel", 1),
                        )
                    elif role == "vision":
                        gguf_path = server_key.split(":", 1)[1] if ":" in server_key else server_key
                        url = await get_vision_server(
                            gguf_path,
                            mmproj_path=cfg.get("mmproj"),
                            ctx_size=cfg.get("ctx_size", 4096),
                            gpu_layers=cfg.get("gpu_layers", -1),
                            n_batch=cfg.get("n_batch", 512),
                            n_ubatch=cfg.get("n_ubatch", 256),
                            flash_attn=cfg.get("flash_attn", True),
                            n_parallel=cfg.get("n_parallel", 1),
                        )
                        from src.ingestion.vision import set_vision_url

                        set_vision_url(url)
                    else:
                        return
                    logger.info(f"[GGUF Watchdog] Restarted {role} on {url}")
                except Exception as e:
                    logger.error(f"[GGUF Watchdog] Failed to restart {role}: {e}")
                return
        except psutil.NoSuchProcess:
            break
        except psutil.AccessDenied:
            break
