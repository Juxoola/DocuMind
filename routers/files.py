"""Роутер: загрузка, удаление файлов, метаданные."""

import asyncio
import gc
import json
import logging
import os
import queue
import threading
import time
import uuid as _uuid

import cv2
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

import config

from .shared import (
    _background_tasks,
    ingestion_status,
    robust_rmtree,
    safe_filename,
    upload_cancel_flags,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["files"])


@router.get("/api/files")
async def get_files(notebook_id: str):
    from routers.notebooks import validate_nb_id

    notebook_id = validate_nb_id(notebook_id)
    paths = config.get_notebook_paths(notebook_id)
    if os.path.exists(paths["data"]):
        files_list = [f for f in os.listdir(paths["data"]) if not f.endswith(".json")]
    else:
        files_list = []
    return {"files": files_list}


@router.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...),
    notebook_id: str = Query(...),
    current_idx: int = Query(1),
    total_count: int = Query(1),
    llm_url: str = Query(None),
    llm_api_key: str | None = None,
    llm_model: str | None = None,
    use_gguf: str | None = None,
    gguf_model_path: str | None = None,
    gguf_mmproj_path: str | None = None,
    vision_model_path: str | None = None,
    vision_mmproj_path: str | None = None,
    vision_temperature: float | None = 0.1,
    vision_ctx_size: int | None = 4096,
    vision_gpu_layers: int | None = -1,
    vision_threads: int | None = 8,
    vision_batch_size: int | None = 512,
    vision_ubatch_size: int | None = 256,
    vision_flash_attn: str | None = "true",
    vision_max_tokens: int | None = 4096,
    vision_repeat_penalty: float | None = 1.2,
    vision_top_p: float | None = 0.9,
    vision_min_p: float | None = 0.05,
    vision_presence_penalty: float | None = 0.0,
    vision_frequency_penalty: float | None = 0.0,
    vision_concurrency: int | None = 1,
    vision_kv_quant: int | None = 2,
    vision_mtp_enabled: bool | None = False,
):
    logger.info(
        f"[API] Новый запрос загрузки для блокнота {notebook_id}. Файл: {file.filename} ({current_idx}/{total_count})"
    )
    _ext = os.path.splitext(file.filename or "")[1].lower()
    if _ext not in config.ALLOWED_UPLOAD_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Тип файла '{_ext}' не поддерживается. Разрешено: {', '.join(sorted(config.ALLOWED_UPLOAD_EXTENSIONS))}.",
        )
    paths = config.get_notebook_paths(notebook_id)
    os.makedirs(paths["data"], exist_ok=True)
    file_path = os.path.join(paths["data"], file.filename)

    def save_upload():
        written = 0
        with open(file_path, "wb") as f:
            while True:
                chunk = file.file.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > config.UPLOAD_MAX_SIZE_BYTES:
                    f.close()
                    try:
                        os.remove(file_path)
                    except Exception:
                        logger.debug("upload: не удалось удалить недописанный файл")
                    raise HTTPException(
                        status_code=413,
                        detail=f"Файл превысил {config.UPLOAD_MAX_SIZE_MB} МБ во время записи.",
                    )
                f.write(chunk)

    await asyncio.to_thread(save_upload)

    q = queue.Queue()
    effective_llm_url = llm_url
    effective_llm_api_key = llm_api_key
    effective_llm_model = llm_model
    use_gguf_direct = False

    if use_gguf == "true" and gguf_model_path:
        use_gguf_direct = True
        effective_llm_model = os.path.basename(gguf_model_path)

    llm_settings = {
        "llm_url": effective_llm_url,
        "llm_api_key": effective_llm_api_key,
        "llm_model": effective_llm_model,
        "use_gguf_direct": use_gguf_direct,
        "gguf_model_path": gguf_model_path if use_gguf_direct else None,
        "gguf_mmproj_path": gguf_mmproj_path if use_gguf_direct else None,
        "vision_model_path": vision_model_path,
        "vision_mmproj_path": vision_mmproj_path,
        "vision_temperature": vision_temperature,
        "vision_ctx_size": vision_ctx_size,
        "vision_gpu_layers": vision_gpu_layers,
        "vision_threads": vision_threads,
        "vision_batch_size": vision_batch_size,
        "vision_ubatch_size": vision_ubatch_size,
        "vision_flash_attn": vision_flash_attn,
        "vision_max_tokens": vision_max_tokens,
        "vision_repeat_penalty": vision_repeat_penalty,
        "vision_top_p": vision_top_p,
        "vision_min_p": vision_min_p,
        "vision_presence_penalty": vision_presence_penalty,
        "vision_frequency_penalty": vision_frequency_penalty,
        "vision_concurrency": vision_concurrency,
        "vision_kv_quant": vision_kv_quant,
        "vision_mtp_enabled": vision_mtp_enabled,
    }

    task_id = _uuid.uuid4().hex
    q.put({"type": "started", "task_id": task_id, "filename": file.filename})

    def process_task():

        start_time = time.time()
        cancel_event = upload_cancel_flags.setdefault(task_id, threading.Event())
        cancel_event.clear()
        ingestion_status[notebook_id] = {
            "is_uploading": True,
            "progress": 0,
            "batch_progress": (current_idx - 1) / total_count * 100,
            "current_file": current_idx,
            "total_files": total_count,
            "status": "Подготовка...",
            "task_id": task_id,
        }
        try:
            from src.ingestion import IngestionCancelled
        except ImportError:
            IngestionCancelled = RuntimeError
        try:

            def prog(pct, msg):
                q.put({"type": "progress", "pct": pct, "msg": msg})
                if notebook_id in ingestion_status:
                    ingestion_status[notebook_id].update({"progress": pct, "status": msg})

            prog(5, "Файл сохранён, подготовка...")
            is_last_in_batch = current_idx >= total_count
            from src.ingestion import ingest_file
            from src.rag.indexing import build_index

            nodes = ingest_file(
                file_path,
                notebook_id,
                progress_cb=prog,
                llm_settings=llm_settings,
                cancel_check=cancel_event.is_set,
                keep_vision_alive=not is_last_in_batch,
                keep_whisper_alive=not is_last_in_batch,
            )
            prog(90, "Построение индекса (ChromaDB)...")
            build_index(nodes, notebook_id)

            elapsed = time.time() - start_time
            mins = int(elapsed // 60)
            secs = int(elapsed % 60)
            time_str = f"{mins}м {secs}с" if mins > 0 else f"{secs}с"

            if is_last_in_batch:
                logger.info(f"[INGESTION] Пачка завершена. {total_count} файлов обработано.")
                try:
                    from src.gguf.server import unload_all_models

                    unload_all_models(role="llm")
                except Exception as llm_err:
                    logger.error(f"[INGESTION] Ошибка выгрузки vision-сервера: {llm_err}")
                try:
                    from src.ingestion import unload_whisper_model

                    unload_whisper_model()
                except Exception as whisper_err:
                    logger.error(f"[INGESTION] Ошибка выгрузки WhisperX: {whisper_err}")
                try:
                    from src.rag.bm25 import flush_bm25_rebuild

                    flush_bm25_rebuild(notebook_id)
                except Exception as bm25_err:
                    logger.info(f"[INGESTION] Не удалось форсировать BM25 rebuild: {bm25_err}")
                ingestion_status[notebook_id] = {"is_uploading": False}
            else:
                ingestion_status[notebook_id].update(
                    {
                        "batch_progress": current_idx / total_count * 100,
                        "status": f"Готово: {file.filename}",
                    }
                )

            q.put(
                {
                    "type": "done",
                    "filename": file.filename,
                    "elapsed": time_str,
                    "elapsed_sec": elapsed,
                }
            )
            logger.info(f"[INGESTION] Готово: {file.filename} ({time_str})")
        except IngestionCancelled:
            logger.info(f"[INGESTION] Загрузка отменена пользователем: {file.filename}")
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception:
                logger.debug("cancel: не удалось удалить %s", file_path)
            sidecar = os.path.join(os.path.dirname(file_path), f"{file.filename}.json")
            try:
                if os.path.exists(sidecar):
                    os.remove(sidecar)
            except Exception:
                logger.debug("cancel: не удалось удалить sidecar %s", sidecar)
            try:
                images_dir = paths.get("images")
                if images_dir and os.path.exists(images_dir):
                    stem = os.path.splitext(file.filename)[0]
                    for f in os.listdir(images_dir):
                        if f.startswith("p_") or f.startswith("v_") or stem in f:
                            try:
                                os.remove(os.path.join(images_dir, f))
                            except Exception:
                                logger.debug("cancel: не удалось удалить %s", f)
            except Exception:
                logger.debug("cancel: ошибка при чистке images")
            try:
                from src.rag.indexing import get_vector_store

                vector_store = get_vector_store(notebook_id)
                collection = vector_store._collection
                collection.delete(where={"file_name": file.filename})
            except Exception:
                logger.debug("cancel: не удалось очистить векторные индексы для %s", file.filename)
            ingestion_status[notebook_id] = {"is_uploading": False, "cancelled": True}
            q.put({"type": "cancelled", "filename": file.filename})
        except Exception as e:
            logger.error("Ошибка при обработке загрузки", exc_info=True)
            ingestion_status[notebook_id] = {"is_uploading": False, "error": str(e)}
            q.put({"type": "error", "msg": str(e)})
        finally:
            upload_cancel_flags.pop(task_id, None)
            from src.ingestion import cleanup_gpu

            cleanup_gpu()

    _task = asyncio.create_task(asyncio.to_thread(process_task))
    _background_tasks.add(_task)
    _task.add_done_callback(_background_tasks.discard)

    async def event_generator():
        loop = asyncio.get_running_loop()
        while True:
            msg = await loop.run_in_executor(None, q.get)
            yield f"data: {json.dumps(msg, ensure_ascii=False)}\n\n"
            if msg["type"] in ("done", "error", "cancelled"):
                break

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/api/upload/cancel")
async def cancel_upload(notebook_id: str = Query(...), task_id: str = Query(None)):
    try:
        from src.ingestion import kill_subprocesses

        killed = kill_subprocesses(notebook_id)
        if killed:
            logger.info(f"[CANCEL] Убито {killed} активных subprocess-ов для {notebook_id}")
    except Exception as e:
        logger.error(f"[CANCEL] Ошибка при kill_subprocesses: {e}")
    if task_id:
        evt = upload_cancel_flags.get(task_id)
        if evt is None:
            return {"status": "no_active_upload"}
        evt.set()
        return {"status": "cancel_requested", "task_id": task_id}
    found = False
    for tid, evt in list(upload_cancel_flags.items()):
        evt.set()
        found = True
    return {"status": "cancel_requested" if found else "no_active_upload"}


@router.delete("/api/files/{filename}")
async def delete_file(filename: str, notebook_id: str):
    from routers.notebooks import validate_nb_id

    notebook_id = validate_nb_id(notebook_id)
    filename = safe_filename(filename)
    paths = config.get_notebook_paths(notebook_id)
    file_path = os.path.join(paths["data"], filename)
    if os.path.exists(file_path):
        if filename.lower().endswith((".mp4", ".avi", ".mov")):
            cap = cv2.VideoCapture(file_path)
            try:
                cap.get(cv2.CAP_PROP_FPS)
            finally:
                cap.release()
        gc.collect()
        for i in range(10):
            try:
                os.remove(file_path)
                break
            except PermissionError:
                if i < 4:
                    time.sleep(1)
                    continue
                # Fallback: переименовываем и удаляем (обходит лок Windows)
                try:
                    tmp = file_path + f".del{i}"
                    os.rename(file_path, tmp)
                    os.remove(tmp)
                    break
                except Exception:
                    if i == 9:
                        logger.error(f"Не удалось удалить {filename}: файл заблокирован")
                        raise
                    time.sleep(1)
    from src.rag.indexing import get_vector_store

    def _delete_chromadb_entries():
        vs = get_vector_store(notebook_id)
        vs._collection.delete(where={"file_name": filename})

    await asyncio.to_thread(_delete_chromadb_entries)
    try:
        from src.bookmarks import mark_stale_for_file

        stale_count = mark_stale_for_file(notebook_id, filename)
        if stale_count:
            logger.info(
                f"[BOOKMARKS] {stale_count} закладок помечены как stale после удаления {filename}"
            )
    except Exception as e:
        logger.info(f"[BOOKMARKS] Не удалось пометить stale: {e}")
    return {"status": "ok"}


@router.get("/api/source_content")
async def get_source_content(filename: str, notebook_id: str):
    filename = safe_filename(filename)
    try:
        from src.rag.indexing import get_vector_store

        vector_store = await asyncio.to_thread(get_vector_store, notebook_id)
        collection = vector_store._collection
        result = await asyncio.to_thread(collection.get, where={"file_name": filename})
        if result and result.get("documents"):
            full_text = "\n\n---\n\n".join(result["documents"])
            return {"text": full_text}
        return {"text": "Содержимое документа не найдено в базе данных."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/video_metadata")
async def get_video_metadata(filename: str, notebook_id: str):
    filename = safe_filename(filename)
    paths = config.get_notebook_paths(notebook_id)
    json_path = os.path.join(paths["data"], f"{filename}.json")

    def read_json():
        if os.path.exists(json_path):
            with open(json_path, encoding="utf-8") as f:
                return json.load(f)
        return None

    data = await asyncio.to_thread(read_json)
    if data is not None:
        return JSONResponse(content=data, headers={"Cache-Control": "public, max-age=300"})
    return {"error": "Метаданные не найдены"}


@router.delete("/api/clear")
async def clear_notebook(notebook_id: str):
    from src.rag.indexing import close_all_clients as _close_all

    def _clear_data():
        _close_all()
        paths = config.get_notebook_paths(notebook_id)
        for d in ("data", "chroma_db", "images"):
            p = paths[d]
            if os.path.exists(p):
                robust_rmtree(p)
            os.makedirs(p, exist_ok=True)

    await asyncio.to_thread(_clear_data)
    return {"status": "ok"}


@router.get("/api/ingestion_status")
async def get_ingestion_status(notebook_id: str):
    return ingestion_status.get(notebook_id, {"is_uploading": False})
