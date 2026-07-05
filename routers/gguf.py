"""Роутер: управление GGUF-моделями, VRAM, предзагрузка LLM."""

import asyncio
import logging
import os
import shutil

import orjson
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import config
from src.gguf.scanner import scan_gguf_dirs
from src.gguf.server import (
    count_running_servers,
    get_llm_status,
    get_loaded_models,
    kill_stray_servers,
    preload_gguf_llm,
    unload_all_models,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["gguf"])


# ── Сканирование и статус GGUF-моделей ──
@router.get("/api/gguf-models")
async def api_scan_gguf_models():
    try:
        models = await scan_gguf_dirs()
        return {"models": models}
    except Exception:
        raise HTTPException(status_code=500, detail="Ошибка сканирования GGUF-моделей")


@router.get("/api/gguf-loaded")
async def api_gguf_loaded_models():
    loaded = await get_loaded_models()
    return {"loaded_models": [os.path.basename(p) for p in loaded]}


@router.get("/api/gguf-status")
async def api_gguf_status():
    count = await count_running_servers()
    return {"running_count": count}


async def _get_gguf_servers_info() -> list:
    servers = []
    try:
        from src.gguf.state import _lock, _server_ports, _server_processes, _server_roles

        async with _lock:
            for path, proc in _server_processes.items():
                alive = True
                if hasattr(proc, "poll"):
                    alive = proc.poll() is None
                else:
                    alive = proc.returncode is None
                servers.append(
                    {
                        "model": os.path.basename(path),
                        "role": _server_roles.get(path, "?"),
                        "port": _server_ports.get(path),
                        "alive": alive,
                    }
                )
    except Exception:
        logger.debug("gguf: не удалось получить информацию о серверах")
    return servers


@router.post("/api/gguf-unload")
async def api_gguf_unload_all():
    await unload_all_models()
    return {"status": "ok", "msg": "Все модели выгружены"}


@router.post("/api/gguf-kill-all")
async def api_gguf_kill_all():
    await kill_stray_servers()
    await unload_all_models()
    return {"status": "ok", "msg": "Все процессы llama-server завершены"}


async def _run_nvidia_smi(query_args: list[str], timeout: float = 3) -> str | None:
    try:
        proc = await asyncio.create_subprocess_exec(
            "nvidia-smi",
            *query_args,
            "--format=csv,noheader,nounits",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        if proc.returncode == 0 and stdout:
            return stdout.decode()
    except Exception:
        logger.debug(f"gguf: nvidia-smi не удался ({query_args[0]})")
    return None


# ── Мониторинг VRAM и процессов GPU ──
@router.get("/api/vram")
async def api_vram():
    if not shutil.which("nvidia-smi"):
        return {
            "gpu": {
                "name": "n/a",
                "used_mib": 0,
                "free_mib": 0,
                "total_mib": 0,
                "used_gb": 0,
                "free_gb": 0,
                "total_gb": 0,
                "utilization_pct": 0,
            },
            "per_process": [],
            "gguf_servers": await _get_gguf_servers_info(),
        }
    gpu_name, used_mib, free_mib, total_mib = "unknown", 0, 0, 0
    gpu_out = await _run_nvidia_smi(["--query-gpu=name,memory.used,memory.free,memory.total"])
    if gpu_out and gpu_out.strip():
        parts = [p.strip() for p in gpu_out.strip().split(",")]
        if len(parts) >= 4:
            try:
                gpu_name, used_mib, free_mib, total_mib = (
                    parts[0],
                    int(parts[1]),
                    int(parts[2]),
                    int(parts[3]),
                )
            except (ValueError, IndexError):
                logger.debug("gguf: nvidia-smi GPU output parse failed")

    per_process = []
    proc_out = await _run_nvidia_smi(["--query-compute-apps=pid,process_name,used_memory"])
    if proc_out:
        for line in proc_out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                try:
                    per_process.append(
                        {"pid": int(parts[0]), "name": parts[1], "vram_mib": int(parts[2])}
                    )
                except (ValueError, IndexError):
                    continue
    return {
        "gpu": {
            "name": gpu_name,
            "used_mib": used_mib,
            "free_mib": free_mib,
            "total_mib": total_mib,
            "used_gb": round(used_mib / 1024, 2),
            "free_gb": round(free_mib / 1024, 2),
            "total_gb": round(total_mib / 1024, 2),
            "utilization_pct": round(used_mib / max(total_mib, 1) * 100, 1),
        },
        "per_process": per_process,
        "gguf_servers": await _get_gguf_servers_info(),
    }


class PreloadLlmRequest(BaseModel):
    gguf_model_path: str
    gguf_mmproj_path: str | None = None
    gguf_ctx_size: int | None = None
    gguf_gpu_layers: int | None = None
    gguf_threads: int | None = None
    gguf_batch_size: int | None = None
    gguf_ubatch_size: int | None = None
    gguf_flash_attn: str | None = "true"
    max_tokens: int | None = None
    gguf_kv_quant: int | None = 2
    thinking_mode: bool | None = True
    thinking_budget: int | None = 1024
    mtp_enabled: bool | None = False


# ── Предзагрузка и управление LLM ──
@router.post("/api/preload-llm")
async def api_preload_llm(request: PreloadLlmRequest):
    if not config.validate_gguf_path(request.gguf_model_path):
        raise HTTPException(status_code=400, detail="Некорректный путь GGUF-модели")
    try:
        result = await preload_gguf_llm(
            gguf_path=request.gguf_model_path,
            mmproj_path=request.gguf_mmproj_path or None,
            ctx_size=request.gguf_ctx_size,
            gpu_layers=request.gguf_gpu_layers,
            n_threads=request.gguf_threads,
            n_batch=request.gguf_batch_size,
            flash_attn=(request.gguf_flash_attn == "true"),
            max_tokens=request.max_tokens,
            type_k=request.gguf_kv_quant,
            type_v=request.gguf_kv_quant,
            enable_thinking=request.thinking_mode,
            thinking_budget=request.thinking_budget,
            mtp_enabled=request.mtp_enabled,
            n_ubatch=request.gguf_ubatch_size,
        )
        return result
    except FileNotFoundError:
        return {"status": "error", "error": "Файл модели не найден"}
    except Exception as e:
        logger.error(f"Ошибка предзагрузки LLM: {e}")
        return {"status": "error", "error": "Ошибка загрузки модели"}


@router.get("/api/llm-status")
async def api_llm_status():
    return await get_llm_status()


@router.get("/api/context-usage")
async def api_context_usage():
    try:
        st = await get_llm_status()
        port = st.get("port")
        state = st.get("state")
        if not port or state not in ("ready", "loading"):
            return {"used": 0, "total": 0, "pct": 0}
        import httpx

        async with httpx.AsyncClient(timeout=3) as client:
            try:
                resp = await client.get(f"http://127.0.0.1:{port}/metrics")
                if resp.status_code == 200:
                    text = resp.text
                    ratio = None
                    n_ctx = None
                    n_used = None
                    for line in text.splitlines():
                        if line.startswith("llamacpp:kv_cache_usage_ratio"):
                            parts = line.split()
                            if len(parts) >= 2:
                                ratio = float(parts[1])
                        elif line.startswith("llamacpp:kv_cache_tokens"):
                            parts = line.split()
                            if len(parts) >= 2:
                                n_used = int(float(parts[1]))
                        elif line.startswith("llamacpp:kv_cache_size"):
                            parts = line.split()
                            if len(parts) >= 2:
                                n_ctx = int(float(parts[1]))
                    if n_used is not None and n_ctx:
                        return {"used": n_used, "total": n_ctx, "pct": round(n_used / n_ctx * 100, 1)}
                    if ratio is not None and n_ctx:
                        n_used = round(ratio * n_ctx)
                        return {"used": n_used, "total": n_ctx, "pct": round(ratio * 100, 1)}
            except Exception:
                pass

            resp = await client.get(f"http://127.0.0.1:{port}/slots")
            if resp.status_code == 200:
                data = resp.json()
                slots = data if isinstance(data, list) else data.get("slots", data.get("data", []))
                if isinstance(slots, list) and len(slots) > 0:
                    slot = slots[0]
                    n_past = int(slot.get("n_past", 0) or 0)
                    n_ctx = int(slot.get("n_ctx", 0) or 1)
                    if n_past > 0:
                        return {
                            "used": n_past,
                            "total": n_ctx,
                            "pct": round(n_past / max(n_ctx, 1) * 100, 1),
                        }
            return {"used": 0, "total": 0, "pct": 0}
    except Exception:
        return {"used": 0, "total": 0, "pct": 0}


# Стриминг статуса LLM в реальном времени через SSE — для отслеживания прогресса предзагрузки.
@router.get("/api/llm-status/stream")
async def api_llm_status_stream():
    async def event_gen():
        last_key = None
        try:
            while True:
                st = await get_llm_status()
                key = (st.get("state"), st.get("phase"), st.get("port"), st.get("error"))
                if key != last_key:
                    payload = orjson.dumps(st).decode()
                    yield f"data: {payload}\n\n"
                    last_key = key
                await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            pass

    return StreamingResponse(event_gen(), media_type="text/event-stream")
