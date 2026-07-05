"""Роутер: загрузка, удаление файлов, метаданные."""

import asyncio
import gc
import logging
import os
import queue
import threading
import time
import uuid as _uuid

import aiofiles
import aiofiles.os
import cv2
import orjson
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse

import config
from src.rag.state import RAG_POOL

from .shared import (
    _background_tasks,
    _cleanup_ingestion_status,
    _ingestion_lock,
    ingestion_status,
    robust_rmtree,
    safe_filename,
    sse_event,
    upload_cancel_flags,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["files"])


# ── Получение списка файлов блокнота ──


@router.get("/api/files")
async def get_files(notebook_id: str):
    from routers.notebooks import validate_nb_id

    notebook_id = validate_nb_id(notebook_id)
    paths = config.get_notebook_paths(notebook_id)
    if await aiofiles.os.path.exists(paths["data"]):
        files_list = [
            f for f in await aiofiles.os.listdir(paths["data"]) if not f.endswith(".json")
        ]
    else:
        files_list = []
    return {"files": files_list}


# ── Загрузка файлов в блокнот ──
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
    vision_concurrency: int | None = None,
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
    await aiofiles.os.makedirs(paths["data"], exist_ok=True)
    file_path = os.path.join(paths["data"], safe_filename(file.filename))

    async def save_upload():
        written = 0
        async with aiofiles.open(file_path, "wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > config.UPLOAD_MAX_SIZE_BYTES:
                    try:
                        await aiofiles.os.remove(file_path)
                    except Exception:
                        logger.debug("upload: не удалось удалить недописанный файл")
                    raise HTTPException(
                        status_code=413,
                        detail=f"Файл превысил {config.UPLOAD_MAX_SIZE_MB} МБ во время записи.",
                    )
                await f.write(chunk)

    await save_upload()

    q: queue.Queue = queue.Queue()
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

    async def process_task():
        start_time = time.time()
        cancel_event = upload_cancel_flags.setdefault(task_id, threading.Event())
        cancel_event.clear()
        async with _ingestion_lock:
            ingestion_status[notebook_id] = {
                "is_uploading": True,
                "progress": 0,
                "batch_progress": (current_idx - 1) / total_count * 100,
                "current_file": current_idx,
                "total_files": total_count,
                "status": "Подготовка...",
                "task_id": task_id,
                "updated_at": time.time(),
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

            nodes = await ingest_file(
                file_path,
                notebook_id,
                progress_cb=prog,
                llm_settings=llm_settings,
                cancel_check=cancel_event.is_set,
                keep_vision_alive=not is_last_in_batch,
                keep_whisper_alive=False,
            )
            prog(90, "Построение индекса (ChromaDB)...")
            await build_index(nodes, notebook_id)
            from src.rag.retrieval import invalidate_index_cache

            await invalidate_index_cache(notebook_id)

            elapsed = time.time() - start_time
            mins = int(elapsed // 60)
            secs = int(elapsed % 60)
            time_str = f"{mins}м {secs}с" if mins > 0 else f"{secs}с"

            if is_last_in_batch:
                logger.info(f"[INGESTION] Пачка завершена. {total_count} файлов обработано.")
                try:
                    from src.gguf.server import unload_all_models

                    await unload_all_models(role="llm")
                except Exception as llm_err:
                    logger.error(f"[INGESTION] Ошибка выгрузки vision-сервера: {llm_err}")
                try:
                    from src.ingestion import unload_whisper_model

                    await unload_whisper_model()
                except Exception as whisper_err:
                    logger.error(f"[INGESTION] Ошибка выгрузки WhisperX: {whisper_err}")
                try:
                    from src.rag.bm25 import flush_bm25_rebuild

                    await flush_bm25_rebuild(notebook_id)
                except Exception as bm25_err:
                    logger.info(f"[INGESTION] Не удалось форсировать BM25 rebuild: {bm25_err}")
                async with _ingestion_lock:
                    ingestion_status[notebook_id] = {
                        "is_uploading": False,
                        "updated_at": time.time(),
                    }
            else:
                async with _ingestion_lock:
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
                from src.gguf.server import kill_stray_servers

                await kill_stray_servers()
            except Exception:
                logger.debug("cancel: не удалось убить llama-server")
            try:
                if await aiofiles.os.path.exists(file_path):
                    await aiofiles.os.remove(file_path)
            except Exception:
                logger.debug("cancel: не удалось удалить %s", file_path)
            sidecar = os.path.join(os.path.dirname(file_path), f"{file.filename}.json")
            try:
                if await aiofiles.os.path.exists(sidecar):
                    await aiofiles.os.remove(sidecar)
            except Exception:
                logger.debug("cancel: не удалось удалить sidecar %s", sidecar)
            try:
                images_dir = paths.get("images")
                if images_dir and await aiofiles.os.path.exists(images_dir):
                    stem = os.path.splitext(file.filename)[0]
                    for f in await aiofiles.os.listdir(images_dir):
                        if f.startswith("p_") or f.startswith("v_") or stem in f:
                            try:
                                await aiofiles.os.remove(os.path.join(images_dir, f))
                            except Exception:
                                logger.debug("cancel: не удалось удалить %s", f)
            except Exception:
                logger.debug("cancel: ошибка при чистке images")
            try:
                from src.rag.indexing import get_vector_store

                vector_store = await get_vector_store(notebook_id)
                collection = vector_store._collection
                collection.delete(where={"file_name": file.filename})
            except Exception:
                logger.debug("cancel: не удалось очистить векторные индексы для %s", file.filename)
            async with _ingestion_lock:
                ingestion_status[notebook_id] = {
                    "is_uploading": False,
                    "cancelled": True,
                    "updated_at": time.time(),
                }
            q.put({"type": "cancelled", "filename": file.filename})
        except Exception as e:
            logger.error("Ошибка при обработке загрузки", exc_info=True)
            async with _ingestion_lock:
                ingestion_status[notebook_id] = {
                    "is_uploading": False,
                    "error": "Внутренняя ошибка обработки файла",
                    "updated_at": time.time(),
                }
            q.put({"type": "error", "msg": "Ошибка обработки файла"})
        finally:
            upload_cancel_flags.pop(task_id, None)
            try:
                from src.ingestion import cleanup_gpu

                cleanup_gpu()
            except Exception:
                logger.debug("finally: не удалось вызвать cleanup_gpu")

    _task = asyncio.create_task(process_task())
    _background_tasks.add(_task)
    _task.add_done_callback(_background_tasks.discard)

    async def event_generator():
        loop = asyncio.get_running_loop()
        while True:
            msg = await loop.run_in_executor(RAG_POOL, q.get)
            yield sse_event(msg)
            if msg["type"] in ("done", "error", "cancelled"):
                break

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ── Отмена загрузки ──


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


# ── Удаление файла и очистка индексов ──


@router.delete("/api/files/{filename}")
async def delete_file(filename: str, notebook_id: str):
    from routers.notebooks import validate_nb_id

    notebook_id = validate_nb_id(notebook_id)
    filename = safe_filename(filename)
    _source_content_cache.pop(f"{notebook_id}:{filename}", None)
    paths = config.get_notebook_paths(notebook_id)
    file_path = os.path.join(paths["data"], filename)
    if await aiofiles.os.path.exists(file_path):
        if filename.lower().endswith((".mp4", ".avi", ".mov")):
            cap = cv2.VideoCapture(file_path)
            try:
                cap.get(cv2.CAP_PROP_FPS)
            finally:
                cap.release()
        gc.collect()

        async def _sync_remove_with_retry(fp: str):
            for i in range(5):
                try:
                    await aiofiles.os.remove(fp)
                    return
                except PermissionError:
                    if i < 2:
                        await asyncio.sleep(0.2)
                        continue
                    try:
                        tmp = fp + f".del{i}"
                        await aiofiles.os.rename(fp, tmp)
                        await aiofiles.os.remove(tmp)
                        return
                    except Exception:
                        if i == 4:
                            raise
                        await asyncio.sleep(0.2)

        await _sync_remove_with_retry(file_path)
        # Удаляем кеш предварительного извлечения
        extracted = os.path.join(paths["data"], f"{filename}.extracted.md")
        if await aiofiles.os.path.exists(extracted):
            try:
                await aiofiles.os.remove(extracted)
            except Exception:
                pass
        metadata_path = os.path.join(paths["data"], f"{filename}.json")
        if await aiofiles.os.path.exists(metadata_path):
            try:
                async with aiofiles.open(metadata_path, "r", encoding="utf-8") as f:
                    meta = orjson.loads(await f.read())
                for frame in meta.get("frames", []):
                    img = frame.get("image_path") or frame.get("path")
                    if img and await aiofiles.os.path.exists(img):
                        try:
                            await aiofiles.os.remove(img)
                        except Exception:
                            pass
            except Exception:
                pass
            try:
                await aiofiles.os.remove(metadata_path)
            except Exception:
                pass
    from src.rag.indexing import get_vector_store

    vector_store = await get_vector_store(notebook_id)
    await asyncio.to_thread(vector_store._collection.delete, where={"file_name": filename})
    from src.rag.retrieval import invalidate_index_cache

    await invalidate_index_cache(notebook_id)
    try:
        from src.bookmarks import mark_stale_for_file

        stale_count = await mark_stale_for_file(notebook_id, filename)
        if stale_count:
            logger.info(
                f"[BOOKMARKS] {stale_count} закладок помечены как stale после удаления {filename}"
            )
    except Exception as e:
        logger.info(f"[BOOKMARKS] Не удалось пометить stale: {e}")
    return {"status": "ok"}


# ── Кеш source_content: {key: (timestamp, text)} — TTL 5 минут ──
_source_content_cache: dict[str, tuple[float, str]] = {}
_CACHE_TTL = 300


def _get_cached_source(key: str) -> str | None:
    entry = _source_content_cache.get(key)
    if entry and (time.time() - entry[0]) < _CACHE_TTL:
        return entry[1]
    return None


def _set_cached_source(key: str, text: str):
    _source_content_cache[key] = (time.time(), text)


@router.get("/api/source_content")
async def get_source_content(filename: str, notebook_id: str):
    filename = safe_filename(filename)
    ext = os.path.splitext(filename)[1].lower()
    paths = config.get_notebook_paths(notebook_id)
    file_path = os.path.join(paths["data"], filename)

    cache_key = f"{notebook_id}:{filename}"
    cached = _get_cached_source(cache_key)
    if cached is not None:
        return {"text": cached}

    # ── Быстрый путь: читаем предварительно извлечённый .extracted.md ──
    if ext == ".pdf":
        extracted_path = os.path.join(paths["data"], f"{filename}.extracted.md")
        if await aiofiles.os.path.exists(extracted_path):
            async with aiofiles.open(extracted_path, encoding="utf-8") as f:
                text = await f.read()
            if text:
                _set_cached_source(cache_key, text)
                return {"text": text}

    # ── ChromaDB: текст + описания, чередуя по страницам ──
    try:
        from src.rag.indexing import get_vector_store

        vector_store = await get_vector_store(notebook_id)
        collection = vector_store._collection
        result = await asyncio.to_thread(
            collection.get, where={"file_name": filename}, include=["documents", "metadatas"]
        )
        if result and result.get("documents"):
            # Группируем чанки по странице, текст и описания чередуются
            from collections import defaultdict

            by_page: dict[int, list[str]] = defaultdict(list)
            for doc, meta in zip(result["documents"], result["metadatas"]):
                page = meta.get("page", 0)
                by_page[page].append(doc)
            pages_sorted = sorted(by_page.items())
            parts = []
            for page_num, chunks in pages_sorted:
                parts.append(f"\n\n--- Стр. {page_num} ---\n\n" + "\n\n".join(chunks))
            full_text = "\n\n".join(parts)
            _set_cached_source(cache_key, full_text)
            return {"text": full_text}
    except Exception:
        pass

    return {"text": "Содержимое документа не найдено."}


_PDF_CSS = """
body { font-family: sans-serif; font-size: 10pt; line-height: 1.5; color: #1a1a1a; }
h1 { font-size: 22pt; margin-bottom: 20px; border-bottom: 2px solid #333; padding-bottom: 8px; }
h2 { font-size: 16pt; margin-top: 24px; color: #222; }
h3 { font-size: 13pt; margin-top: 18px; color: #333; }
p { margin: 6px 0; text-align: justify; }
pre { background: #f4f4f4; padding: 10px; border-radius: 4px; font-size: 9pt; }
code { font-family: monospace; background: #f0f0f0; padding: 1px 4px; border-radius: 2px; }
table { border-collapse: collapse; width: 100%; margin: 12px 0; }
th, td { border: 1px solid #ccc; padding: 6px 8px; text-align: left; font-size: 9pt; }
th { background: #f0f0f0; font-weight: bold; }
blockquote { border-left: 3px solid #999; padding-left: 12px; color: #555; }
"""


def _build_pdf(title: str, text: str) -> bytes:

    import fitz
    import markdown

    html_body = markdown.markdown(text, extensions=["tables", "fenced_code"])
    html = f"<h1>{title}</h1>\n{html_body}"

    import tempfile as _tf
    _out_fd, out_path = _tf.mkstemp(suffix=".pdf", prefix="_export_")
    os.close(_out_fd)
    compressed_path = None
    try:
        writer = fitz.DocumentWriter(out_path)
        story = fitz.Story(html=html, user_css=_PDF_CSS)

        def contentfn(positions):
            return html

        def rectfn(page_num, filled):
            return fitz.Rect(0, 0, 595, 842), fitz.Rect(50, 50, 545, 790), fitz.Identity

        story.write_stabilized(writer, contentfn, rectfn, em=10)
        writer.close()

        _c_fd, compressed_path = _tf.mkstemp(suffix=".pdf", prefix="_export_c_")
        os.close(_c_fd)
        doc = fitz.open(out_path)
        doc.save(compressed_path, garbage=4, deflate=True)
        doc.close()

        with open(compressed_path, "rb") as f:
            pdf_bytes = f.read()

    finally:
        for _f in (out_path, compressed_path):
            if _f is not None:
                try:
                    os.unlink(_f)
                except OSError:
                    pass
    return pdf_bytes


def _content_disposition(name: str, ext: str) -> str:

    from urllib.parse import quote

    full = f"{name}.{ext}"
    encoded = quote(full)
    ascii_safe = full.encode("ascii", "replace").decode().replace("?", "_")
    return f"attachment; filename=\"{ascii_safe}\"; filename*=UTF-8''{encoded}"


# ── Экспорт текста и видео-метаданные ──
@router.get("/api/export_text")
async def export_text(filename: str, notebook_id: str, fmt: str = "txt"):

    filename = safe_filename(filename)
    ext = os.path.splitext(filename)[1].lower()
    is_video = ext in (".mp4", ".avi", ".mov", ".mkv")
    is_audio = ext in (".mp3", ".wav", ".m4a", ".aac", ".flac")
    is_image = ext in (".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp")
    is_media = is_video or is_audio

    text = ""
    stem = os.path.splitext(filename)[0]
    paths = config.get_notebook_paths(notebook_id)

    if is_media or is_image:
        json_path = os.path.join(paths["data"], f"{filename}.json")
        if await aiofiles.os.path.exists(json_path):
            async with aiofiles.open(json_path, "rb") as f:
                data = orjson.loads(await f.read())
            if is_media:
                transcript = data.get("transcript", [])
                parts = []
                if transcript:
                    lines = []
                    for seg in transcript:
                        start = seg.get("start", 0)
                        m, s = divmod(int(start), 60)
                        h, m = divmod(m, 60)
                        ts = f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
                        lines.append(f"[{ts}] {seg.get('text', '')}")
                    parts.append("\n".join(lines))
                frames = data.get("frames", [])
                if frames:
                    frame_lines = []
                    for fr in frames:
                        t = fr.get("time", 0)
                        desc = fr.get("description", "")
                        if desc:
                            m, s = divmod(int(t), 60)
                            h, m = divmod(m, 60)
                            ts = f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
                            frame_lines.append(f"[{ts}] {desc}")
                    if frame_lines:
                        parts.append("\n\n## Описания кадров\n\n" + "\n\n".join(frame_lines))
                text = "\n\n".join(parts)
            elif is_image:
                frames = data.get("frames", [])
                if frames and frames[0].get("description"):
                    text = frames[0]["description"]
        if not text:
            text = f"Описание/транскрипт для {filename} не найдены."
    else:
        file_path = os.path.join(paths["data"], filename)

        # Пытаемся взять текст из тех же источников, что и source_content
        cache_key = f"{notebook_id}:{filename}"
        cached = _get_cached_source(cache_key)
        if cached is not None:
            text = cached
        elif ext == ".pdf":
            # .extracted.md
            extracted_path = os.path.join(paths["data"], f"{filename}.extracted.md")
            if await aiofiles.os.path.exists(extracted_path):
                async with aiofiles.open(extracted_path, encoding="utf-8") as ef:
                    text = await ef.read()
        if not text:
            # ChromaDB: текст + описания, чередуя по страницам
            try:
                from src.rag.indexing import get_vector_store

                vector_store = await get_vector_store(notebook_id)
                collection = vector_store._collection
                result = await asyncio.to_thread(
                    collection.get,
                    where={"file_name": filename},
                    include=["documents", "metadatas"],
                )
                if result and result.get("documents"):
                    from collections import defaultdict

                    by_page = defaultdict(list)
                    for doc, meta in zip(result["documents"], result["metadatas"]):
                        page = meta.get("page", 0)
                        by_page[page].append(doc)
                    pages_sorted = sorted(by_page.items())
                    parts = []
                    for page_num, chunks in pages_sorted:
                        parts.append(f"\n\n--- Стр. {page_num} ---\n\n" + "\n\n".join(chunks))
                    text = "\n\n".join(parts)
            except Exception:
                pass
        if not text:
            text = f"Содержимое {filename} не найдено."

    base_name = stem
    ext_map = {"pdf": "pdf", "md": "md"}
    file_ext = ext_map.get(fmt, "txt")
    if fmt == "pdf":
        pdf_bytes = await asyncio.to_thread(_build_pdf, stem, text)
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": _content_disposition(base_name, "pdf")},
        )
    elif fmt == "md":
        content = f"# {stem}\n\n{text}"
        media_type = "text/markdown"
    else:
        content = text
        media_type = "text/plain; charset=utf-8"

    return Response(
        content=content.encode("utf-8"),
        media_type=media_type,
        headers={"Content-Disposition": _content_disposition(base_name, file_ext)},
    )


@router.get("/api/video_metadata")
async def get_video_metadata(filename: str, notebook_id: str):
    filename = safe_filename(filename)
    paths = config.get_notebook_paths(notebook_id)
    json_path = os.path.join(paths["data"], f"{filename}.json")

    if not await aiofiles.os.path.exists(json_path):
        return {"error": "Метаданные не найдены"}
    async with aiofiles.open(json_path, "rb") as f:
        data = orjson.loads(await f.read())
    return JSONResponse(content=data, headers={"Cache-Control": "public, max-age=300"})


@router.delete("/api/clear")
async def clear_notebook(notebook_id: str):
    from routers.notebooks import validate_nb_id
    notebook_id = validate_nb_id(notebook_id)

    from src.rag.indexing import close_all_clients as _close_all

    # Очищаем кеш source_content для этого блокнота
    _prefix = f"{notebook_id}:"
    for k in [k for k in _source_content_cache if k.startswith(_prefix)]:
        del _source_content_cache[k]

    async def _clear_data():
        await _close_all()
        paths = config.get_notebook_paths(notebook_id)
        for d in ("data", "chroma_db", "images"):
            p = paths[d]
            if await aiofiles.os.path.exists(p):
                await robust_rmtree(p)
            await aiofiles.os.makedirs(p, exist_ok=True)

    await _clear_data()
    return {"status": "ok"}


@router.get("/api/ingestion_status")
async def get_ingestion_status(notebook_id: str):
    from routers.notebooks import validate_nb_id
    notebook_id = validate_nb_id(notebook_id)

    await _cleanup_ingestion_status()
    async with _ingestion_lock:
        return ingestion_status.get(notebook_id, {"is_uploading": False})
