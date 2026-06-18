"""Жизненный цикл GGUF-серверов (запуск, остановка, получение URL)."""

import logging
import os
import socket
import subprocess
import sys
import threading
import time

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


def is_server_ready(port: int) -> bool:

    try:
        import httpx

        with httpx.Client(timeout=1) as client:
            r = client.get(f"http://127.0.0.1:{port}/health")
            return r.status_code == 200
    except Exception:
        return False


# Выбор свободного порта через bind, сборка CLI, запуск subprocess с ожиданием ready через health-check


def _start_llm_server_sync(gguf_path: str, mmproj_path: str, current_config: dict) -> str:

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()

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

    if current_config.get("mmproj") and os.path.exists(current_config["mmproj"]):
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
    process = subprocess.Popen(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=creationflags
    )
    _assign_to_job(process)

    start_wait = time.time()
    backoff = 0.05
    while time.time() - start_wait < 60:
        if is_server_ready(port):
            logger.info("[GGUF Server] Готов!")
            with _lock:
                _server_processes[gguf_path] = process
                _server_ports[gguf_path] = port
                _server_configs[gguf_path] = current_config
                _server_roles[gguf_path] = "llm"
            return f"http://127.0.0.1:{port}"

        if process.poll() is not None:
            try:
                _, stderr = process.communicate(timeout=1)
            except Exception:
                stderr = b""
                logger.debug("llama-server communicate failed")
            stderr_text = (stderr or b"").decode("utf-8", errors="ignore")[:500]
            retcode = process.returncode
            raise RuntimeError(
                f"Сервер llama-server упал при запуске (pid={process.pid}, retcode={retcode}). "
                f"stderr: {stderr_text}"
            )
        time.sleep(backoff)
        backoff = min(backoff * 2, 1.0)

    try:
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(process.pid)], capture_output=True, timeout=5
            )
        else:
            process.kill()
    except Exception as e:
        try:
            process.kill()
        except Exception:
            pass
    try:
        process.wait(timeout=5)
    except Exception:
        pass
    raise TimeoutError("Сервер не ответил за 60 секунд")


def unload_rag_models_safe():

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


# Основной entry-point для загрузки LLM: проверяет кеш, выгружает старое, запускает новое


def get_gguf_llm(
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
) -> str:

    gguf_path = os.path.normpath(config.resolve_model_path(gguf_path)).lower()
    if mmproj_path:
        mmproj_path = os.path.normpath(config.resolve_model_path(mmproj_path)).lower()

    current_config = _build_llm_config(
        gguf_path,
        mmproj_path,
        temperature=temperature,
        ctx_size=ctx_size,
        gpu_layers=gpu_layers,
        n_threads=n_threads,
        n_batch=n_batch,
        flash_attn=flash_attn,
        max_tokens=max_tokens,
        type_k=type_k,
        type_v=type_v,
        enable_thinking=enable_thinking,
        thinking_budget=thinking_budget,
        n_parallel=n_parallel,
        custom_args=custom_args,
        mtp_enabled=mtp_enabled,
        n_ubatch=n_ubatch,
    )

    with _lock:
        if gguf_path in _server_processes:
            if (
                _server_processes[gguf_path].poll() is None
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
            else:
                logger.info(
                    f"[GGUF Server] Настройки изменились или сервер упал. Перезапуск {os.path.basename(gguf_path)}..."
                )

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
        unload_rag_models_safe()
        unload_all_models(role="llm")

        if not os.path.exists(gguf_path):
            _llm_load_state.update({"state": "error", "error": f"Model not found: {gguf_path}"})
            raise FileNotFoundError(f"GGUF модель не найдена: {gguf_path}")

    try:
        url = _start_llm_server_sync(gguf_path, mmproj_path, current_config)
        elapsed = time.time() - (_llm_load_state.get("started_at") or time.time())
        with _lock:
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
        with _lock:
            _llm_load_state.update({"state": "error", "error": str(e)[:300], "phase": None})
        raise


def preload_gguf_llm(
    gguf_path: str,
    mmproj_path: str = None,
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

    import uuid

    gguf_path = os.path.normpath(config.resolve_model_path(gguf_path)).lower()
    if mmproj_path:
        mmproj_path = os.path.normpath(config.resolve_model_path(mmproj_path)).lower()

    current_config = _build_llm_config(
        gguf_path,
        mmproj_path,
        ctx_size=ctx_size,
        gpu_layers=gpu_layers,
        n_threads=n_threads,
        n_batch=n_batch,
        flash_attn=flash_attn,
        max_tokens=max_tokens,
        type_k=type_k,
        type_v=type_v,
        enable_thinking=enable_thinking,
        thinking_budget=thinking_budget,
        n_parallel=n_parallel,
        custom_args=custom_args,
        mtp_enabled=mtp_enabled,
        n_ubatch=n_ubatch,
    )

    task_id = str(uuid.uuid4())[:8]

    with _lock:
        if gguf_path in _server_processes:
            if (
                _server_processes[gguf_path].poll() is None
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

    def _worker():
        try:
            _llm_load_state["phase"] = "starting"
            unload_rag_models_safe()
            unload_all_models(role="llm")
            if not os.path.exists(gguf_path):
                raise FileNotFoundError(f"GGUF модель не найдена: {gguf_path}")
            _llm_load_state["phase"] = "loading_model"
            url = _start_llm_server_sync(gguf_path, mmproj_path, current_config)
            elapsed = time.time() - _llm_load_state["started_at"]
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
            _llm_load_state.update({"state": "error", "error": str(e)[:300], "phase": None})
            logger.info(f"[preload] ERROR: {e}")

    thread = threading.Thread(target=_worker, daemon=True, name=f"preload-llm-{task_id}")
    thread.start()
    return {"status": "loading", "task_id": task_id, "model": os.path.basename(gguf_path)}


def get_llm_status() -> dict:

    with _lock:
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


def get_gguf_embedding_url(
    gguf_path: str, n_threads: int = None, is_reranker: bool = False, n_parallel: int = 1
) -> str:

    role = "reranker" if is_reranker else "embedding"
    n_parallel = max(1, int(n_parallel or 1))

    current_config = {
        "n_threads": n_threads,
        "is_reranker": is_reranker,
        "n_parallel": n_parallel,
    }

    with _lock:
        if gguf_path in _server_processes:
            if (
                _server_processes[gguf_path].poll() is None
                and _server_configs.get(gguf_path) == current_config
            ):
                return f"http://127.0.0.1:{_server_ports[gguf_path]}"
            else:
                logger.info(f"[GGUF Server] Перезапуск {role} {os.path.basename(gguf_path)}...")

        unload_rag_models_safe()
        unload_all_models(role=role)

        if not os.path.exists(gguf_path):
            raise FileNotFoundError(f"GGUF модель не найдена: {gguf_path}")

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()

    cmd = [SERVER_EXE, "-m", gguf_path, "--port", str(port), "--parallel", str(n_parallel)]

    if not is_reranker:
        cmd.extend(["--embedding"])
        if "qwen" in os.path.basename(gguf_path).lower():
            cmd.extend(["--override-kv", "tokenizer.ggml.suffix_token_id=int:151643"])
    else:
        cmd.extend(["--reranking"])

    min_ctx_per_slot = 2048
    ctx = str(max(4096, n_parallel * min_ctx_per_slot))
    if is_reranker:
        b_size, ub_size = "2048", "2048"
    else:
        b_size, ub_size = "512", "512"
    cmd.extend(["-c", ctx, "-b", b_size, "-ub", ub_size])

    cmd.extend(["--cache-type-k", "q8_0", "--cache-type-v", "q8_0"])

    if config.GGUF_GPU_LAYERS != 0:
        cmd.extend(["-ngl", str(config.GGUF_GPU_LAYERS)])
    cmd.extend(["--flash-attn", "on"])

    if n_threads and n_threads > 0:
        cmd.extend(["-t", str(n_threads)])

    logger.info(
        f"[GGUF Server] Запуск {role}: {os.path.basename(gguf_path)} на порту {port} (parallel={n_parallel})..."
    )
    logger.info(f"[GGUF Server]   cmd: {' '.join(cmd)}")

    creationflags = 0x08000000
    process = subprocess.Popen(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=creationflags
    )
    _assign_to_job(process)

    start_wait = time.time()
    backoff = 0.05
    while time.time() - start_wait < 60:
        if is_server_ready(port):
            logger.info(f"[GGUF Server] {role.capitalize()} готов!")
            with _lock:
                _server_processes[gguf_path] = process
                _server_ports[gguf_path] = port
                _server_configs[gguf_path] = current_config
                _server_roles[gguf_path] = role
            return f"http://127.0.0.1:{port}"

        if process.poll() is not None:
            raise RuntimeError(f"{role.capitalize()} сервер упал при запуске")
        time.sleep(backoff)
        backoff = min(backoff * 2, 1.0)

    process.terminate()
    raise TimeoutError(f"{role.capitalize()} сервер не ответил за 60 секунд")


def get_active_embedding_parallel(gguf_path: str = None) -> int:

    with _lock:
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


def kill_stray_servers():

    with _lock:
        if not _server_processes:
            logger.debug("[GGUF Server] Нет отслеживаемых процессов для завершения.")
            return
        processes_copy = dict(_server_processes)
        _server_processes.clear()
        _server_ports.clear()
        _server_configs.clear()
        _server_roles.clear()

    logger.info("[GGUF Server] Завершение отслеживаемых процессов llama-server...")
    for path, process in processes_copy.items():
        try:
            if process.poll() is None:
                if os.name == "nt":
                    subprocess.run(
                        ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                        capture_output=True,
                        timeout=5,
                    )
                else:
                    process.kill()
                process.wait(timeout=5)
            logger.debug(f"[GGUF Server] Остановлен: {os.path.basename(path)} (PID {process.pid})")
        except Exception as e:
            logger.debug(f"[GGUF Server] Не удалось остановить {os.path.basename(path)}: {e}")


def count_running_servers() -> int:

    with _lock:
        alive = 0
        for path, proc in _server_processes.items():
            if proc.poll() is None:
                alive += 1
        return alive


# Выгрузка процессов через taskkill/pkill + cleanup GPU через torch.cuda.empty_cache


def unload_all_models(role: str = None):

    if not _server_processes:
        return

    to_remove = []
    for path, process in _server_processes.items():
        if role is not None and _server_roles.get(path) != role:
            continue

        logger.info(
            f"[GGUF Server] Выгрузка модели ({_server_roles.get(path, 'unknown')}): {os.path.basename(path)}"
        )
        to_remove.append(path)

        if process.poll() is None:
            try:
                port = _server_ports.get(path)
                if port:
                    try:
                        import httpx

                        with httpx.Client(timeout=0.5) as client:
                            client.post(f"http://127.0.0.1:{port}/slots/0/clear")
                    except Exception:
                        pass
            except Exception:
                pass
            try:
                if sys.platform == "win32":
                    subprocess.run(
                        ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                        capture_output=True,
                        timeout=5,
                    )
                else:
                    process.kill()
                try:
                    process.wait(timeout=5)
                except Exception:
                    pass
            except Exception as e:
                logger.error(f"[GGUF Server] Ошибка при остановке {os.path.basename(path)}: {e}")

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
    time.sleep(0.1)


def get_loaded_models():

    return [path for path in _server_processes.keys() if _server_roles.get(path, "llm") == "llm"]


def get_active_llm_url() -> str | None:

    with _lock:
        for path, process in _server_processes.items():
            if _server_roles.get(path) == "llm" and process.poll() is None:
                port = _server_ports.get(path)
                if port:
                    return f"http://127.0.0.1:{port}"
    return None
