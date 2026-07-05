"""Жизненный цикл GGUF-серверов (запуск, остановка, получение URL)."""

import asyncio
import logging
import os
import socket
import subprocess
import time

import httpx
import psutil
import torch

import config
from src.gguf.state import (
    CACHE_TYPE_MAP,
    SERVER_EXE,
    _assign_to_job,
    _llm_load_state,
    _lock,
    _server_configs,
    _server_ports,
    _server_processes,
    _server_roles,
)

logger = logging.getLogger(__name__)
_watchdog_tasks: set[asyncio.Task] = set()

# ── Переиспользуемый HTTP-клиент для health-check (один коннект на все polling-циклы) ──
_health_http: httpx.AsyncClient | None = None
_health_http_lock = asyncio.Lock()


async def _get_health_client() -> httpx.AsyncClient:
    global _health_http
    if _health_http is None or _health_http.is_closed:
        async with _health_http_lock:
            if _health_http is None or _health_http.is_closed:
                _health_http = httpx.AsyncClient(timeout=1)
    return _health_http


# ── Утилиты: выделение порта, убийство процесса, ожидание готовности сервера ──
def _allocate_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


async def _kill_server_process(process) -> None:
    if os.name == "nt":
        try:
            proc = await asyncio.create_subprocess_exec(
                "taskkill",
                "/F",
                "/T",
                "/PID",
                str(process.pid),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            await asyncio.wait_for(proc.communicate(), timeout=5)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass
    else:
        process.kill()


async def _wait_for_server(
    port: int,
    process,
    role: str = "llm",
    server_key: str | None = None,
    server_config: dict | None = None,
) -> str:
    start_wait = time.time()
    backoff = 0.05
    while time.time() - start_wait < 60:
        if await is_server_ready(port):
            logger.info(f"[GGUF Server] {role.capitalize()} готов!")
            if server_key and server_config is not None:
                async with _lock:
                    _server_processes[server_key] = process
                    _server_ports[server_key] = port
                    _server_configs[server_key] = server_config
                    _server_roles[server_key] = role
                if role in _WATCHDOG_LIMITS:
                    _wd_task = asyncio.create_task(_watchdog_memory(server_key, process, role))
                    _watchdog_tasks.add(_wd_task)
                    _wd_task.add_done_callback(_watchdog_tasks.discard)
            return f"http://127.0.0.1:{port}"
        if process.returncode is not None:
            try:
                _, stderr = await asyncio.wait_for(process.communicate(), timeout=1)
            except Exception:
                stderr = b""
            stderr_text = (stderr or b"").decode("utf-8", errors="ignore")[:500]
            raise RuntimeError(
                f"{role.capitalize()} сервер упал при запуске (pid={process.pid}, "
                f"retcode={process.returncode}). stderr: {stderr_text}"
            )
        await asyncio.sleep(backoff)
        backoff = min(backoff * 2, 1.0)
    try:
        await _kill_server_process(process)
    except Exception:
        pass
    try:
        await asyncio.wait_for(process.wait(), timeout=5)
    except Exception:
        pass
    raise TimeoutError(f"{role.capitalize()} сервер не ответил за 60 секунд")


def _proc_alive(proc) -> bool:

    if hasattr(proc, "poll"):
        return proc.poll() is None
    return proc.returncode is None


async def is_server_ready(port: int) -> bool:
    try:
        client = await _get_health_client()
        r = await client.get(f"http://127.0.0.1:{port}/health")
        return r.status_code == 200
    except Exception:
        return False


# Лимиты RSS по ролям (MB): при превышении — auto-restart
_WATCHDOG_LIMITS = {
    "llm": 12000,
    "vision": 4000,
    "embedding": 2000,
    "reranker": 2000,
}


async def _watchdog_memory(server_key: str, process, role: str):
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
                # Сигнал vision-запросам: ждите, сервер умирает
                if role == "vision":
                    from src.ingestion.vision import set_vision_url

                    set_vision_url(None)
                await _kill_server_process(process)
                try:
                    await asyncio.wait_for(process.wait(), timeout=5)
                except Exception:
                    pass
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
        except Exception as e:
            logger.debug(f"[GGUF Watchdog] {server_key}: {e}")


async def _start_llm_server(gguf_path: str, mmproj_path: str, current_config: dict) -> str:
    port = _allocate_port()

    total_ctx = current_config["ctx_size"] * current_config["n_parallel"]
    type_k_str = CACHE_TYPE_MAP.get(current_config["type_k"], "f16")
    type_v_str = CACHE_TYPE_MAP.get(current_config["type_v"], "f16")

    cmd = [
        SERVER_EXE,
        "-m",
        gguf_path,
        "--port",
        str(port),
        "-c",
        str(total_ctx),
        "-ngl",
        str(current_config["gpu_layers"]),
        "-b",
        str(current_config["n_batch"]),
        "-ub",
        str(current_config["n_ubatch"]),
        "--parallel",
        str(current_config["n_parallel"]),
        "--cont-batching",
        "--jinja",
        "--cache-type-k",
        type_k_str,
        "--cache-type-v",
        type_v_str,
        "--no-mmap",
        "-n",
        str(current_config["max_tokens"]),
    ]

    if current_config["mtp_enabled"]:
        cmd.extend(["--spec-type", "draft-mtp", "--spec-draft-n-max", "2"])

    if not current_config["enable_thinking"]:
        cmd.extend(["--reasoning", "off", "--reasoning-format", "none", "--reasoning-budget", "0"])
    else:
        cmd.extend(
            ["--reasoning", "on", "--reasoning-budget", str(current_config["thinking_budget"])]
        )

    if current_config["flash_attn"]:
        if not any(
            "-fa" in str(arg) or "--flash-attn" in str(arg)
            for arg in (current_config.get("custom_args") or [])
        ):
            cmd.extend(["--flash-attn", "on"])

    if current_config.get("_n_threads") and current_config["_n_threads"] > 0:
        cmd.extend(["-t", str(current_config["_n_threads"])])

    if current_config.get("mmproj") and await asyncio.to_thread(os.path.exists, current_config["mmproj"]):
        cmd.extend(["--mmproj", os.path.normpath(current_config["mmproj"])])
        logger.info(
            f"[GGUF Server] С поддержкой Vision: {os.path.basename(current_config['mmproj'])}"
        )

    cmd.extend(["--metrics"])
    if current_config.get("custom_args"):
        cmd.extend(current_config["custom_args"])

    logger.info(f"[GGUF Server] Запуск: {os.path.basename(gguf_path)} на порту {port}...")
    logger.info(f"[GGUF Server]   cmd: {' '.join(cmd)}")

    creationflags = 0x08000000
    process = await asyncio.create_subprocess_exec(
        *cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=creationflags
    )
    _assign_to_job(process)

    return await _wait_for_server(
        port, process, role="llm", server_key=gguf_path, server_config=current_config
    )


async def unload_rag_models_safe():
    try:
        from src.rag.models import unload_rag_models

        unload_rag_models(hard=False)
    except Exception:
        pass


def _build_llm_config(
    gguf_path: str,
    mmproj_path: str = None,
    temperature: float = 0.1,
    ctx_size: int = None,
    gpu_layers: int = -1,
    n_threads: int = None,
    n_batch: int = 512,
    flash_attn: bool = True,
    max_tokens: int = 4096,
    type_k: int = 2,
    type_v: int = 2,
    enable_thinking: bool = True,
    thinking_budget: int = 1024,
    n_parallel: int = 1,
    custom_args: list[str] | None = None,
    mtp_enabled: bool = False,
    n_ubatch: int = 256,
) -> dict:
    return {
        "mmproj": mmproj_path or None,
        "ctx_size": int(ctx_size or config.GGUF_CTX_SIZE),
        "gpu_layers": int(gpu_layers if gpu_layers is not None else -1),
        "n_batch": int(n_batch or 512),
        "flash_attn": bool(flash_attn),
        "max_tokens": int(max_tokens or 4096),
        "type_k": int(type_k or 2),
        "type_v": int(type_v or 2),
        "enable_thinking": bool(enable_thinking),
        "thinking_budget": int(thinking_budget if thinking_budget is not None else 1024),
        "n_parallel": int(n_parallel or 1),
        "custom_args": custom_args if custom_args is not None else [],
        "mtp_enabled": bool(mtp_enabled),
        "n_ubatch": int(n_ubatch or 256),
        "_n_threads": n_threads,
    }


# ── Нормализация путей и построение конфига: общая логика для sync и background ──
def _resolve_llm_config(
    gguf_path: str, mmproj_path: str = None, **kwargs
) -> tuple[str, str | None, dict]:
    gguf_path = os.path.normpath(config.resolve_model_path(gguf_path)).lower()
    if mmproj_path:
        mmproj_path = os.path.normpath(config.resolve_model_path(mmproj_path)).lower()
    return gguf_path, mmproj_path, _build_llm_config(gguf_path, mmproj_path, **kwargs)


async def _load_llm(gguf_path: str, mmproj_path: str, current_config: dict) -> str:
    async with _lock:
        _llm_load_state.update(
            {
                "state": "loading",
                "model": gguf_path,
                "port": None,
                "task_id": None,
                "started_at": time.time(),
                "ready_at": None,
                "error": None,
                "phase": "starting",
            }
        )
        _server_configs[gguf_path] = current_config
    await unload_rag_models_safe()
    await unload_all_models(role="llm")
    if not await asyncio.to_thread(os.path.exists, gguf_path):
        async with _lock:
            _llm_load_state.update({"state": "error", "error": f"Model not found: {gguf_path}"})
        raise FileNotFoundError(f"GGUF модель не найдена: {gguf_path}")
    try:
        url = await _start_llm_server(gguf_path, mmproj_path, current_config)
        elapsed = time.time() - (_llm_load_state.get("started_at") or time.time())
        async with _lock:
            _llm_load_state.update(
                {
                    "state": "ready",
                    "port": _server_ports.get(gguf_path),
                    "ready_at": time.time(),
                    "last_load_seconds": elapsed,
                    "phase": "ready",
                    "error": None,
                }
            )
        return url
    except Exception as e:
        async with _lock:
            _server_configs.pop(gguf_path, None)
            _llm_load_state.update({"state": "error", "error": str(e)[:300], "phase": None})
        raise


async def get_gguf_llm(gguf_path: str, mmproj_path: str = None, **kwargs) -> str:
    gguf_path, mmproj_path, current_config = _resolve_llm_config(gguf_path, mmproj_path, **kwargs)
    async with _lock:
        if gguf_path in _server_processes:
            if (
                _proc_alive(_server_processes[gguf_path])
                and _server_configs.get(gguf_path) == current_config
            ):
                _llm_load_state.update(
                    {
                        "state": "ready",
                        "model": gguf_path,
                        "port": _server_ports[gguf_path],
                        "error": None,
                    }
                )
                return f"http://127.0.0.1:{_server_ports[gguf_path]}"
            logger.info(
                f"[GGUF Server] Настройки изменились, перезапуск {os.path.basename(gguf_path)}..."
            )
    return await _load_llm(gguf_path, mmproj_path, current_config)


_preload_tasks: set = set()


async def preload_gguf_llm(gguf_path: str, mmproj_path: str = None, **kwargs) -> dict:
    import uuid

    gguf_path, mmproj_path, current_config = _resolve_llm_config(gguf_path, mmproj_path, **kwargs)
    task_id = str(uuid.uuid4())[:8]
    async with _lock:
        if gguf_path in _server_processes:
            if (
                _proc_alive(_server_processes[gguf_path])
                and _server_configs.get(gguf_path) == current_config
            ):
                _llm_load_state.update(
                    {
                        "state": "ready",
                        "model": gguf_path,
                        "port": _server_ports[gguf_path],
                        "task_id": task_id,
                        "started_at": time.time(),
                        "ready_at": time.time(),
                        "phase": "ready",
                        "error": None,
                    }
                )
                return {"status": "ready", "port": _server_ports[gguf_path], "task_id": task_id}
    async with _lock:
        _llm_load_state.update(
            {
                "state": "loading",
                "model": gguf_path,
                "port": None,
                "task_id": task_id,
                "started_at": time.time(),
                "ready_at": None,
                "error": None,
                "phase": "freeing",
            }
        )
        _server_configs[gguf_path] = current_config

    async def _worker():
        try:
            await unload_rag_models_safe()
            await unload_all_models(role="llm")
            if not await asyncio.to_thread(os.path.exists, gguf_path):
                raise FileNotFoundError(f"GGUF модель не найдена: {gguf_path}")
            url = await _start_llm_server(gguf_path, mmproj_path, current_config)
            elapsed = time.time() - _llm_load_state["started_at"]
            async with _lock:
                _llm_load_state.update(
                    {
                        "state": "ready",
                        "port": _server_ports.get(gguf_path),
                        "ready_at": time.time(),
                        "last_load_seconds": elapsed,
                        "phase": "ready",
                        "error": None,
                    }
                )
            logger.info(f"[preload] OK: loaded in {elapsed:.1f}s")
        except Exception as e:
            async with _lock:
                _server_configs.pop(gguf_path, None)
                _llm_load_state.update({"state": "error", "error": str(e)[:300], "phase": None})
            logger.info(f"[preload] ERROR: {e}")

    _task = asyncio.create_task(_worker(), name=f"preload-llm-{task_id}")
    _preload_tasks.add(_task)
    _task.add_done_callback(_preload_tasks.discard)
    return {"status": "loading", "task_id": task_id, "model": os.path.basename(gguf_path)}


async def get_llm_status() -> dict:
    async with _lock:
        state = _llm_load_state.copy()
    if state.get("started_at") and state.get("state") == "loading":
        elapsed = time.time() - state["started_at"]
        state["elapsed"] = round(elapsed, 1)
        last = state.get("last_load_seconds")
        if last:
            state["eta"] = round(max(0, last - elapsed), 1)
        else:
            try:
                size_mb = (
                    os.path.getsize(state["model"]) / (1024 * 1024) if state.get("model") else 0
                )
                state["eta"] = round(max(5, size_mb / 100), 1)
            except Exception:
                state["eta"] = None
    elif state.get("state") == "ready":
        state["elapsed"] = round(
            (state.get("ready_at") or time.time()) - (state.get("started_at") or time.time()), 1
        )
        state["eta"] = 0
    return state


async def get_gguf_embedding_url(
    gguf_path: str, n_threads: int = None, is_reranker: bool = False, n_parallel: int = 1
) -> str:
    role = "reranker" if is_reranker else "embedding"
    n_parallel = max(1, int(n_parallel or 1))

    current_config = {
        "n_threads": n_threads,
        "is_reranker": is_reranker,
        "n_parallel": n_parallel,
    }

    async with _lock:
        if gguf_path in _server_processes:
            if (
                _proc_alive(_server_processes[gguf_path])
                and _server_configs.get(gguf_path) == current_config
            ):
                return f"http://127.0.0.1:{_server_ports[gguf_path]}"
            else:
                logger.info(f"[GGUF Server] Перезапуск {role} {os.path.basename(gguf_path)}...")

    await unload_rag_models_safe()

    await unload_all_models(role=role)

    if not await asyncio.to_thread(os.path.exists, gguf_path):
        raise FileNotFoundError(f"GGUF модель не найдена: {gguf_path}")

    port = _allocate_port()

    cmd = [SERVER_EXE, "-m", gguf_path, "--port", str(port), "--parallel", str(n_parallel)]

    if not is_reranker:
        cmd.extend(["--embedding"])
        if "qwen" in os.path.basename(gguf_path).lower():
            cmd.extend(["--override-kv", "tokenizer.ggml.suffix_token_id=int:151643"])
    else:
        cmd.extend(["--reranking"])

    min_ctx_per_slot = 4096
    ctx = str(max(4096, n_parallel * min_ctx_per_slot))
    if is_reranker:
        b_size, ub_size = "2048", "2048"
    else:
        b_size, ub_size = "512", "512"
    cmd.extend(["-c", ctx, "-b", b_size, "-ub", ub_size])

    cmd.extend(["--cache-type-k", "q4_0", "--cache-type-v", "q4_0"])

    if config.GGUF_GPU_LAYERS != 0:
        cmd.extend(["-ngl", str(config.GGUF_GPU_LAYERS)])
    cmd.extend(["--flash-attn", "on", "--no-mmap"])

    if n_threads and n_threads > 0:
        cmd.extend(["-t", str(n_threads)])

    logger.info(
        f"[GGUF Server] Запуск {role}: {os.path.basename(gguf_path)} на порту {port} (parallel={n_parallel})..."
    )
    logger.info(f"[GGUF Server]   cmd: {' '.join(cmd)}")

    creationflags = 0x08000000
    process = await asyncio.create_subprocess_exec(
        *cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=creationflags
    )
    _assign_to_job(process)

    return await _wait_for_server(
        port, process, role=role, server_key=gguf_path, server_config=current_config
    )


async def get_vision_server(
    gguf_path: str,
    mmproj_path: str = None,
    ctx_size: int = 4096,
    gpu_layers: int = -1,
    n_threads: int = None,
    n_batch: int = 512,
    n_ubatch: int = 256,
    flash_attn: bool = True,
    n_parallel: int = 1,
    custom_args: list[str] | None = None,
) -> str:
    server_key = f"vision:{gguf_path}"

    current_config = {
        "mmproj": mmproj_path or None,
        "ctx_size": int(ctx_size or 4096),
        "gpu_layers": int(gpu_layers if gpu_layers is not None else -1),
        "n_batch": int(n_batch or 512),
        "n_ubatch": int(n_ubatch or 256),
        "flash_attn": bool(flash_attn),
        "n_parallel": int(n_parallel or 1),
        "custom_args": custom_args or [],
        "_n_threads": n_threads,
    }

    async with _lock:
        if server_key in _server_processes:
            if (
                _proc_alive(_server_processes[server_key])
                and _server_configs.get(server_key) == current_config
            ):
                logger.info(f"[GGUF Server] Vision-сервер уже готов: {_server_ports[server_key]}")
                return f"http://127.0.0.1:{_server_ports[server_key]}"
            else:
                logger.info("[GGUF Server] Vision: настройки изменились, перезапуск...")

    # Сигнал vision-запросам: сервер перезагружается
    from src.ingestion.vision import set_vision_url

    set_vision_url(None)

    await unload_all_models(role="vision")

    if not await asyncio.to_thread(os.path.exists, gguf_path):
        raise FileNotFoundError(f"Vision модель не найдена: {gguf_path}")

    port = _allocate_port()

    total_ctx = current_config["ctx_size"] * current_config["n_parallel"]
    cmd = [
        SERVER_EXE,
        "-m",
        gguf_path,
        "--port",
        str(port),
        "-c",
        str(total_ctx),
        "-ngl",
        str(current_config["gpu_layers"]),
        "-b",
        str(current_config["n_batch"]),
        "-ub",
        str(current_config["n_ubatch"]),
        "--parallel",
        str(current_config["n_parallel"]),
        "--jinja",
        "--cache-type-k",
        "q4_0",
        "--cache-type-v",
        "q4_0",
        "-n",
        "2048",
        "--reasoning",
        "off",
        "--reasoning-format",
        "none",
        "--reasoning-budget",
        "0",
    ]

    if current_config["flash_attn"]:
        cmd.extend(["--flash-attn", "on", "--no-mmap"])

    if current_config.get("_n_threads") and current_config["_n_threads"] > 0:
        cmd.extend(["-t", str(current_config["_n_threads"])])

    if current_config.get("mmproj") and await asyncio.to_thread(os.path.exists, current_config["mmproj"]):
        cmd.extend(["--mmproj", os.path.normpath(current_config["mmproj"])])
        logger.info(
            f"[GGUF Server] Vision с поддержкой mmproj: {os.path.basename(current_config['mmproj'])}"
        )

    cmd.extend(["--metrics"])
    if current_config.get("custom_args"):
        cmd.extend(current_config["custom_args"])

    logger.info(f"[GGUF Server] Запуск vision: {os.path.basename(gguf_path)} на порту {port}...")
    logger.info(f"[GGUF Server]   cmd: {' '.join(cmd)}")

    creationflags = 0x08000000
    process = await asyncio.create_subprocess_exec(
        *cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=creationflags
    )
    _assign_to_job(process)

    return await _wait_for_server(
        port, process, role="vision", server_key=server_key, server_config=current_config
    )


async def get_active_embedding_parallel(gguf_path: str = None) -> int:
    async with _lock:
        if gguf_path:
            cfg = _server_configs.get(gguf_path)
            if cfg and cfg.get("n_parallel"):
                return int(cfg["n_parallel"])
            return 1
        for path, role in _server_roles.items():
            if role == "embedding":
                cfg = _server_configs.get(path, {})
                return int(cfg.get("n_parallel", 1))
        return 1


async def kill_stray_servers():

    async with _lock:
        if not _server_processes:
            logger.debug("[GGUF Server] Нет отслеживаемых процессов для завершения.")
            return
        processes_copy = dict(_server_processes)
        _server_processes.clear()
        _server_ports.clear()
        _server_configs.clear()
        _server_roles.clear()

    for path, process in processes_copy.items():
        try:
            if _proc_alive(process):
                await _kill_server_process(process)
                await asyncio.wait_for(process.wait(), timeout=5)
            logger.debug(f"[GGUF Server] Остановлен: {os.path.basename(path)} (PID {process.pid})")
        except Exception as e:
            logger.debug(f"[GGUF Server] Не удалось остановить {os.path.basename(path)}: {e}")


async def count_running_servers() -> int:
    async with _lock:
        alive = 0
        for path, proc in _server_processes.items():
            if _proc_alive(proc):
                alive += 1
        return alive


# ── Выгрузка всех процессов и очистка GPU ──


async def unload_all_models(role: str = None):

    async with _lock:
        if not _server_processes:
            return

        to_remove = []
        to_kill = []
        for path, process in _server_processes.items():
            if role is not None and _server_roles.get(path) != role:
                continue

            logger.info(
                f"[GGUF Server] Выгрузка модели ({_server_roles.get(path, 'unknown')}): {os.path.basename(path)}"
            )
            to_remove.append(path)

            if _proc_alive(process):
                to_kill.append((path, process, _server_ports.get(path)))

    async def _stop_one(path, process, port):
        if port:
            try:
                client = await _get_health_client()
                await client.post(f"http://127.0.0.1:{port}/slots/0/clear")
            except Exception:
                pass
        try:
            await _kill_server_process(process)
            try:
                await asyncio.wait_for(process.wait(), timeout=10)
            except Exception:
                pass
            for _ in range(10):
                if not _proc_alive(process):
                    break
                await asyncio.sleep(0.5)
        except Exception as e:
            logger.error(f"[GGUF Server] Ошибка при остановке {os.path.basename(path)}: {e}")

    await asyncio.gather(
        *[_stop_one(p, proc, port) for p, proc, port in to_kill],
        return_exceptions=True,
    )

    async with _lock:
        for path in to_remove:
            _server_processes.pop(path, None)
            _server_ports.pop(path, None)
            _server_configs.pop(path, None)
            _server_roles.pop(path, None)

    from src.ingestion.utils import cleanup_gpu

    cleanup_gpu()
    if torch.cuda.is_available():
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
    await asyncio.sleep(0.1)


async def get_loaded_models():
    async with _lock:
        return [
            path for path in _server_processes.keys() if _server_roles.get(path, "llm") == "llm"
        ]


async def get_active_llm_url() -> str | None:
    async with _lock:
        for path, process in _server_processes.items():
            if _server_roles.get(path) == "llm" and _proc_alive(process):
                port = _server_ports.get(path)
                if port:
                    return f"http://127.0.0.1:{port}"
    return None
