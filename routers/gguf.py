"""
Роутер: управление GGUF-моделями, VRAM, предзагрузка LLM.
"""
import os
import json
import subprocess
import logging
from typing import Optional
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncio

import config
from .shared import _http_session

from src.gguf_direct import (
    get_gguf_llm, preload_gguf_llm, get_llm_status,
    unload_all_models, kill_stray_servers, count_running_servers,
    get_loaded_models,
)
from src.gguf_manager import scan_gguf_dirs

logger = logging.getLogger(__name__)
router = APIRouter(tags=["gguf"])


# ── Список моделей ──

@router.get("/api/gguf-models")
async def api_scan_gguf_models():
    try:
        models = scan_gguf_dirs()
        return {"models": models}
    except Exception as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/gguf-loaded")
async def api_gguf_loaded_models():
    loaded = get_loaded_models()
    return {"loaded_models": [os.path.basename(p) for p in loaded]}


@router.get("/api/gguf-status")
async def api_gguf_status():
    count = count_running_servers()
    return {"running_count": count}


# ── Управление ──

@router.post("/api/gguf-unload")
async def api_gguf_unload_all():
    unload_all_models()
    return {"status": "ok", "msg": "Все модели выгружены"}


@router.post("/api/gguf-kill-all")
async def api_gguf_kill_all():
    kill_stray_servers()
    unload_all_models()
    return {"status": "ok", "msg": "Все процессы llama-server завершены"}


# ── VRAM ──

@router.get("/api/vram")
async def api_vram():
    used_mib = 0
    total_mib = 0
    free_mib = 0
    gpu_name = "unknown"
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.free,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3,
        )
        if out.returncode == 0 and out.stdout.strip():
            parts = [p.strip() for p in out.stdout.strip().split(",")]
            if len(parts) >= 4:
                gpu_name = parts[0]
                used_mib = int(parts[1])
                free_mib = int(parts[2])
                total_mib = int(parts[3])
    except Exception:
        pass
    per_process = []
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3,
        )
        if out.returncode == 0:
            for line in out.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 3:
                    try:
                        per_process.append({"pid": int(parts[0]), "name": parts[1], "vram_mib": int(parts[2])})
                    except (ValueError, IndexError):
                        continue
    except Exception:
        pass
    known_servers = []
    try:
        from src.gguf_direct import _server_processes, _server_ports, _server_roles
        for path, proc in _server_processes.items():
            known_servers.append({
                "model": os.path.basename(path),
                "role": _server_roles.get(path, "?"),
                "port": _server_ports.get(path),
                "alive": proc.poll() is None,
            })
    except Exception:
        pass
    return {
        "gpu": {
            "name": gpu_name, "used_mib": used_mib, "free_mib": free_mib, "total_mib": total_mib,
            "used_gb": round(used_mib / 1024, 2), "free_gb": round(free_mib / 1024, 2),
            "total_gb": round(total_mib / 1024, 2),
            "utilization_pct": round(used_mib / max(total_mib, 1) * 100, 1),
        },
        "per_process": per_process,
        "gguf_servers": known_servers,
    }


# ── Preload LLM ──

class PreloadLlmRequest(BaseModel):
    gguf_model_path: str
    gguf_mmproj_path: Optional[str] = None
    gguf_ctx_size: Optional[int] = None
    gguf_gpu_layers: Optional[int] = None
    gguf_threads: Optional[int] = None
    gguf_batch_size: Optional[int] = None
    gguf_ubatch_size: Optional[int] = None
    gguf_flash_attn: Optional[str] = "true"
    max_tokens: Optional[int] = None
    gguf_kv_quant: Optional[int] = 2
    thinking_mode: Optional[bool] = True
    thinking_budget: Optional[int] = 1024
    mtp_enabled: Optional[bool] = False


@router.post("/api/preload-llm")
async def api_preload_llm(request: PreloadLlmRequest):
    try:
        result = await asyncio.to_thread(
            preload_gguf_llm,
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
    except FileNotFoundError as e:
        return {"status": "error", "error": str(e)}
    except Exception as e:
        return {"status": "error", "error": str(e)[:300]}


@router.get("/api/llm-status")
async def api_llm_status():
    return get_llm_status()


@router.get("/api/llm-status/stream")
async def api_llm_status_stream():
    async def event_gen():
        last_key = None
        try:
            while True:
                st = get_llm_status()
                key = (st.get("state"), st.get("phase"), st.get("port"), st.get("error"))
                if key != last_key:
                    payload = json.dumps(st, ensure_ascii=False)
                    yield f"data: {payload}\n\n"
                    last_key = key
                await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            pass
    return StreamingResponse(event_gen(), media_type="text/event-stream")
