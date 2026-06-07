import os
import shutil
import json
import time
import uuid
import logging
import requests
import traceback
import threading
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
import config
import gc
import stat
import requests
import requests.adapters

import asyncio

logger = logging.getLogger(__name__)
from src.ingestion import ingest_file
from src.rag_pipeline import build_index, retrieve_nodes, build_file_context, make_prompt, make_messages, close_all_clients, preload_all_models
from src.gguf_manager import scan_gguf_dirs
from src.gguf_direct import (
    get_gguf_llm, preload_gguf_llm, get_llm_status, unload_all_models, get_loaded_models,
    detect_model_family, stream_gguf_chat, kill_stray_servers, count_running_servers
)
from src.rag_pipeline import unload_rag_models # Импортируем для очистки
from fastapi.middleware.cors import CORSMiddleware
from llama_index.core import Settings
from contextlib import asynccontextmanager

_lifespan_cleanup_done = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _lifespan_cleanup_done
    # При запуске - на всякий случай чистим мусор
    kill_stray_servers()

    # Фоновая предзагрузка моделей (сервер запустится мгновенно)
    import threading
    from src.rag_pipeline import preload_all_models
    threading.Thread(target=preload_all_models, daemon=True).start()

    yield

    # Завершение: выгрузка моделей. Под локом — чтобы при uvicorn --reload
    # (который дёргает lifespan несколько раз) cleanup не дёргался повторно.
    if not _lifespan_cleanup_done:
        _lifespan_cleanup_done = True
        print("[SERVER] Остановка системы...")
        unload_all_models()
        kill_stray_servers()

# Регистрация atexit для надежности на Windows
import atexit
from src.gguf_direct import unload_all_models, kill_stray_servers
atexit.register(unload_all_models)
atexit.register(kill_stray_servers)

app = FastAPI(title="NotebookLM Local Clone", lifespan=lifespan)

# F-fix #15: общий HTTP session для chat/vision completion запросов и slot clear.
# Каждый запрос — TCP handshake. Session переиспользует keep-alive соединения.
_http_session = requests.Session()
_http_session.mount("http://", requests.adapters.HTTPAdapter(pool_connections=10, pool_maxsize=10))
_http_session.mount("https://", requests.adapters.HTTPAdapter(pool_connections=10, pool_maxsize=10))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:8000", "http://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def safe_filename(filename: str) -> str:
    """Валидация имени файла для защиты от path traversal."""
    clean = Path(filename).name
    if clean != filename or not clean or clean.startswith('.'):
        raise HTTPException(status_code=400, detail=f"Недопустимое имя файла: {filename}")
    return clean

# Монтируем статику
os.makedirs(os.path.join(config.BASE_DIR, "static"), exist_ok=True)
app.mount("/static", StaticFiles(directory=os.path.join(config.BASE_DIR, "static")), name="static")
app.mount("/files", StaticFiles(directory=config.NOTEBOOKS_DIR), name="notebooks")

# Миграция старых данных в "default" ноутбук
def migrate_old_data():
    old_data = os.path.join(config.BASE_DIR, "data")
    old_db = os.path.join(config.BASE_DIR, "chroma_db")
    old_imgs = os.path.join(config.BASE_DIR, "images")
    
    if os.path.exists(old_data) or os.path.exists(old_db) or os.path.exists(old_imgs):
        print("Обнаружены старые данные. Миграция в ноутбук 'default'...")
        paths = config.get_notebook_paths("default")
        os.makedirs(paths["base"], exist_ok=True)
        if os.path.exists(old_data): shutil.move(old_data, paths["data"])
        if os.path.exists(old_db): shutil.move(old_db, paths["chroma_db"])
        if os.path.exists(old_imgs): shutil.move(old_imgs, paths["images"])
        
        # Создаем meta.json
        with open(os.path.join(paths["base"], "meta.json"), "w", encoding="utf-8") as f:
            json.dump({"id": "default", "name": "Мой первый блокнот", "created_at": time.time()}, f)

migrate_old_data()

def robust_rmtree(path, max_retries=5, delay=0.5):
    """Надежное удаление директории для Windows (с обработкой блокировок ChromaDB)."""
    if not os.path.exists(path):
        return

    # Сначала пытаемся снять атрибут 'только для чтения'
    for root, dirs, files in os.walk(path):
        for f in files:
            try: os.chmod(os.path.join(root, f), stat.S_IWRITE)
            except Exception: pass
        for d in dirs:
            try: os.chmod(os.path.join(root, d), stat.S_IWRITE)
            except Exception: pass

    for i in range(max_retries):
        try:
            gc.collect() # Принудительно закрываем дескрипторы файлов
            shutil.rmtree(path)
            return True
        except PermissionError:
            if i < max_retries - 1:
                time.sleep(delay)
            else:
                raise
        except Exception as e:
            print(f"Ошибка при удалении {path}: {e}")
            if i < max_retries - 1:
                time.sleep(delay)
            else:
                raise



# ── Управление блокнотами ──

@app.get("/api/notebooks")
async def get_notebooks():
    nbs = []
    if os.path.exists(config.NOTEBOOKS_DIR):
        for nb_id in os.listdir(config.NOTEBOOKS_DIR):
            meta_path = os.path.join(config.NOTEBOOKS_DIR, nb_id, "meta.json")
            if os.path.exists(meta_path):
                with open(meta_path, "r", encoding="utf-8") as f:
                    nbs.append(json.load(f))
    return nbs

class CreateNotebookRequest(BaseModel):
    name: str

@app.post("/api/notebooks")
async def create_notebook(req: CreateNotebookRequest):
    nb_id = str(uuid.uuid4())[:8]
    paths = config.get_notebook_paths(nb_id)
    os.makedirs(paths["data"], exist_ok=True)
    os.makedirs(paths["chroma_db"], exist_ok=True)
    os.makedirs(paths["images"], exist_ok=True)
    
    meta = {"id": nb_id, "name": req.name, "created_at": time.time()}
    with open(os.path.join(paths["base"], "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f)
    return meta

@app.delete("/api/notebooks/{nb_id}")
async def delete_notebook(nb_id: str):
    close_all_clients() # Закрываем базу перед удалением
    paths = config.get_notebook_paths(nb_id)
    if os.path.exists(paths["base"]):
        robust_rmtree(paths["base"])
    return {"status": "ok"}

# ── Операции с файлами ──

@app.get("/api/files")
async def get_files(notebook_id: str):
    paths = config.get_notebook_paths(notebook_id)
    if os.path.exists(paths["data"]):
        files = [f for f in os.listdir(paths["data"]) if not f.endswith(".json")]
    else:
        files = []
    return {"files": files}

# Глобальный статус загрузки для каждого блокнота
ingestion_status = {}
# Флаги отмены активной загрузки (per notebook)
upload_cancel_flags: dict = {}

@app.get("/api/ingestion_status")
async def get_ingestion_status(notebook_id: str):
    return ingestion_status.get(notebook_id, {"is_uploading": False})

@app.post("/api/upload/cancel")
async def cancel_upload(notebook_id: str = Query(...)):
    """Сигнализирует активному процессу загрузки в этом блокноте остановиться."""
    evt = upload_cancel_flags.get(notebook_id)
    # Мгновенно убиваем все subprocess-ы (ffmpeg и пр.), даже если cancel_check ещё не сработал
    try:
        from src.ingestion import kill_subprocesses
        killed = kill_subprocesses(notebook_id)
        if killed:
            print(f"[CANCEL] Убито {killed} активных subprocess-ов для {notebook_id}")
    except Exception as e:
        print(f"[CANCEL] Ошибка при kill_subprocesses: {e}")
    if evt is None:
        return {"status": "no_active_upload"}
    evt.set()
    return {"status": "cancel_requested"}

@app.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...), 
    notebook_id: str = Query(...),
    current_idx: int = Query(1),
    total_count: int = Query(1),
    llm_url: str = Query(None),
    llm_api_key: Optional[str] = None,
    llm_model: Optional[str] = None,
    use_gguf: Optional[str] = None,
    gguf_model_path: Optional[str] = None,
    gguf_mmproj_path: Optional[str] = None,
    vision_model_path: Optional[str] = None,
    vision_mmproj_path: Optional[str] = None,
    vision_temperature: Optional[float] = 0.1,
    vision_ctx_size: Optional[int] = 4096,
    vision_gpu_layers: Optional[int] = -1,
    vision_threads: Optional[int] = 8,
    vision_batch_size: Optional[int] = 512,
    vision_ubatch_size: Optional[int] = 256,
    vision_flash_attn: Optional[str] = "true",
    vision_max_tokens: Optional[int] = 4096,
    vision_repeat_penalty: Optional[float] = 1.2,
    vision_top_p: Optional[float] = 0.9,
    vision_min_p: Optional[float] = 0.05,
    vision_presence_penalty: Optional[float] = 0.0,
    vision_frequency_penalty: Optional[float] = 0.0,
    vision_concurrency: Optional[int] = 1,
    vision_kv_quant: Optional[int] = 2,
    vision_mtp_enabled: Optional[bool] = False, # Multi-Token Prediction для Vision модели
):
    print(f"[API] Новый запрос загрузки для блокнота {notebook_id}. Файл: {file.filename} ({current_idx}/{total_count})")
    paths = config.get_notebook_paths(notebook_id)
    os.makedirs(paths["data"], exist_ok=True)
    file_path = os.path.join(paths["data"], file.filename)
    def save_upload():
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
            
    await asyncio.to_thread(save_upload)

    import threading
    import queue
    

    q = queue.Queue()
    
    # Определяем эффективный LLM — если выбрана GGUF модель, используем прямой API
    effective_llm_url = llm_url
    effective_llm_api_key = llm_api_key
    effective_llm_model = llm_model
    use_gguf_direct = False
    
    if use_gguf == "true" and gguf_model_path:
        # Используем прямой API вместо сервера
        use_gguf_direct = True
        effective_llm_model = os.path.basename(gguf_model_path)
        # Сохраняем как последнюю удачную конфигурацию
        config.save_last_model(gguf_model_path, gguf_mmproj_path)

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

    def process_task():
        import time
        start_time = time.time()
        # Инициализируем cancel-флаг для этого блокнота
        cancel_event = upload_cancel_flags.setdefault(notebook_id, threading.Event())
        cancel_event.clear()
        # Инициализируем статус пачки
        ingestion_status[notebook_id] = {
            "is_uploading": True,
            "progress": 0,
            "batch_progress": (current_idx - 1) / total_count * 100,
            "current_file": current_idx,
            "total_files": total_count,
            "status": "Подготовка..."
        }
        try:
            from src.ingestion import IngestionCancelled
        except ImportError:
            IngestionCancelled = RuntimeError  # fallback, если модуль не успел загрузиться
        try:
            def prog(pct, msg):
                q.put({"type": "progress", "pct": pct, "msg": msg})
                # Обновляем глобальный статус
                if notebook_id in ingestion_status:
                    ingestion_status[notebook_id].update({
                        "progress": pct,
                        "status": msg
                    })

            prog(5, "Файл сохранён, подготовка...")
            # F-fix #7: держим Vision-сервер живым между файлами в batch.
            # keep_vision_alive=True означает, что process_pdf/process_audio_video
            # НЕ выгрузит vision-сервер по окончании OCR — это сделает main.py
            # после последнего файла. Экономим 4× старт 4B модели (~2-3 минуты)
            # и предотвращаем 4 потенциальных orphan CUDA-контекста на Windows.
            # F-fix #11: то же самое для WhisperX — в batch'е аудио/видео модель
            # грузится 1 раз, а не N раз. Экономия ~30 сек на каждый файл после первого.
            is_last_in_batch = (current_idx >= total_count)
            nodes = ingest_file(
                file_path, notebook_id,
                progress_cb=prog, llm_settings=llm_settings,
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

            # Если это последний файл в пачке — форсируем пересборку BM25 сейчас,
            # иначе debounce в build_index подождёт _BM25_DEBOUNCE_SEC.
            if is_last_in_batch:
                print(f"[INGESTION] Пачка завершена. {total_count} файлов обработано.")
                # Vision-сервер удерживался живым всё время batch — выгружаем сейчас один раз.
                try:
                    from src.gguf_direct import unload_all_models
                    unload_all_models(role="llm")
                    print(f"[INGESTION] Vision-сервер выгружен после batch.")
                except Exception as llm_err:
                    print(f"[INGESTION] Ошибка выгрузки vision-сервера: {llm_err}")
                # F-fix #11: WhisperX тоже удерживался живым в batch'е аудио/видео — выгружаем.
                try:
                    from src.ingestion import unload_whisper_model
                    unload_whisper_model()
                except Exception as whisper_err:
                    print(f"[INGESTION] Ошибка выгрузки WhisperX: {whisper_err}")
                try:
                    from src.rag_pipeline import flush_bm25_rebuild
                    flush_bm25_rebuild(notebook_id)
                except Exception as bm25_err:
                    print(f"[INGESTION] Не удалось форсировать BM25 rebuild: {bm25_err}")
                ingestion_status[notebook_id] = {"is_uploading": False}
            else:
                # Обновляем прогресс пачки
                ingestion_status[notebook_id].update({
                    "batch_progress": current_idx / total_count * 100,
                    "status": f"Готово: {file.filename}"
                })

            q.put({"type": "done", "filename": file.filename, "elapsed": time_str, "elapsed_sec": elapsed})
            print(f"[INGESTION] Готово: {file.filename} ({time_str})")
        except IngestionCancelled:
            print(f"[INGESTION] Загрузка отменена пользователем: {file.filename}")
            # Комплексная очистка, чтобы не оставалось "призраков":
            # 1) главный файл
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"[INGESTION] Файл удалён после отмены: {file_path}")
            except Exception as e:
                print(f"[INGESTION] Не удалось удалить {file_path}: {e}")
            # 2) sidecar .json (мог быть записан process_audio_video/process_pdf до того, как cancel долетел)
            sidecar = os.path.join(os.path.dirname(file_path), f"{file.filename}.json")
            try:
                if os.path.exists(sidecar):
                    os.remove(sidecar)
                    print(f"[INGESTION] Sidecar удалён: {sidecar}")
            except Exception as e:
                print(f"[INGESTION] Не удалось удалить sidecar: {e}")
            # 3) изображения страниц/кадров в images_dir
            try:
                images_dir = paths.get("images")
                if images_dir and os.path.exists(images_dir):
                    stem = os.path.splitext(file.filename)[0]
                    for f in os.listdir(images_dir):
                        if f.startswith("p_") or f.startswith("v_") or stem in f:
                            try: os.remove(os.path.join(images_dir, f))
                            except Exception: pass
            except Exception as e:
                print(f"[INGESTION] Ошибка очистки images: {e}")
            # 4) частичные эмбеддинги в ChromaDB (если build_index был вызван до cancel — не должно быть,
            #    но на всякий случай)
            try:
                from src.rag_pipeline import get_vector_store
                vector_store = get_vector_store(notebook_id)
                collection = vector_store._collection
                collection.delete(where={"file_name": file.filename})
                print(f"[INGESTION] ChromaDB записи для {file.filename} удалены")
            except Exception as e:
                print(f"[INGESTION] Очистка ChromaDB не удалась (это нормально, если индекс не строился): {e}")
            ingestion_status[notebook_id] = {"is_uploading": False, "cancelled": True}
            q.put({"type": "cancelled", "filename": file.filename})
        except Exception as e:
            import traceback
            traceback.print_exc()
            ingestion_status[notebook_id] = {"is_uploading": False, "error": str(e)}
            q.put({"type": "error", "msg": str(e)})
        finally:
            # Освобождаем cancel-флаг после завершения задачи
            upload_cancel_flags.pop(notebook_id, None)
            # После загрузки файла оставляем модели в памяти, но чистим кэш
            import gc, torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    asyncio.create_task(asyncio.to_thread(process_task))

    async def event_generator():
        while True:
            while q.empty():
                await asyncio.sleep(0.1)
            msg = q.get()
            yield f"data: {json.dumps(msg, ensure_ascii=False)}\n\n"
            if msg["type"] in ["done", "error", "cancelled"]:
                break

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.delete("/api/files/{filename}")
async def delete_file(filename: str, notebook_id: str):
    import cv2
    filename = safe_filename(filename)
    paths = config.get_notebook_paths(notebook_id)
    file_path = os.path.join(paths["data"], filename)
    
    if os.path.exists(file_path):
        # Если это видео, пытаемся освободить дескриптор, если он был занят
        if filename.lower().endswith(('.mp4', '.avi', '.mov')):
            cap = cv2.VideoCapture(file_path)
            try:
                fps = cap.get(cv2.CAP_PROP_FPS) or 25
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration_sec = total_frames / fps if fps > 0 else 0
            finally:
                cap.release()
                
        gc.collect()
        for i in range(5):
            try:
                os.remove(file_path)
                break
            except PermissionError:
                if i == 4: raise
                time.sleep(0.5)
    
    # Удаляем из ChromaDB
    from src.rag_pipeline import get_vector_store
    vector_store = get_vector_store(notebook_id)
    collection = vector_store._collection
    collection.delete(where={"file_name": filename})
    # Помечаем связанные закладки как stale (не удаляем — пользователь может захотеть увидеть историю)
    try:
        stale_count = mark_stale_for_file(notebook_id, filename)
        if stale_count:
            print(f"[BOOKMARKS] {stale_count} закладок помечены как stale после удаления {filename}")
    except Exception as e:
        print(f"[BOOKMARKS] Не удалось пометить stale: {e}")
    return {"status": "ok"}

@app.get("/api/source_content")
async def get_source_content(filename: str, notebook_id: str):
    filename = safe_filename(filename)
    try:
        from src.rag_pipeline import get_vector_store
        vector_store = await asyncio.to_thread(get_vector_store, notebook_id)
        collection = vector_store._collection
        # Выполняем тяжелый запрос к БД в потоке
        result = await asyncio.to_thread(collection.get, where={"file_name": filename})
        if result and result.get("documents"):
            full_text = "\n\n---\n\n".join(result["documents"])
            return {"text": full_text}
        return {"text": "Содержимое документа не найдено в базе данных."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/video_metadata")
async def get_video_metadata(filename: str, notebook_id: str):
    filename = safe_filename(filename)
    paths = config.get_notebook_paths(notebook_id)
    json_path = os.path.join(paths["data"], f"{filename}.json")

    def read_json():
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    data = await asyncio.to_thread(read_json)
    if data is not None:
        from fastapi.responses import JSONResponse
        return JSONResponse(content=data, headers={"Cache-Control": "public, max-age=300"})
    return {"error": "Метаданные не найдены"}

@app.delete("/api/clear")
async def clear_notebook(notebook_id: str):
    close_all_clients() # Закрываем базу перед очисткой
    paths = config.get_notebook_paths(notebook_id)
    for d in ["data", "chroma_db", "images"]:
        p = paths[d]
        if os.path.exists(p):
            robust_rmtree(p)
        os.makedirs(p, exist_ok=True)
    return {"status": "ok"}

# ── Закладки (Q&A) ──

from src.bookmarks import (
    list_bookmarks, get_bookmark, create_bookmark,
    update_bookmark, delete_bookmark, mark_stale_for_file,
)


@app.get("/api/bookmarks")
async def api_list_bookmarks(notebook_id: str = Query(...)):
    return {"bookmarks": list_bookmarks(notebook_id)}


@app.get("/api/bookmarks/{bookmark_id}")
async def api_get_bookmark(bookmark_id: str, notebook_id: str = Query(...)):
    bm = get_bookmark(notebook_id, bookmark_id)
    if bm is None:
        raise HTTPException(status_code=404, detail="Закладка не найдена")
    return bm


class CreateBookmarkRequest(BaseModel):
    notebook_id: str
    question: str
    answer: str
    sources: List[dict] = []
    model: Optional[str] = ""
    answer_mode: Optional[str] = "concise"
    thinking_mode: Optional[bool] = False
    title: Optional[str] = ""
    tags: List[str] = []


@app.post("/api/bookmarks")
async def api_create_bookmark(req: CreateBookmarkRequest):
    try:
        bm = create_bookmark(req.notebook_id, req.dict())
        return bm
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


class UpdateBookmarkRequest(BaseModel):
    notebook_id: str
    title: Optional[str] = None
    tags: Optional[List[str]] = None


@app.patch("/api/bookmarks/{bookmark_id}")
async def api_update_bookmark(bookmark_id: str, req: UpdateBookmarkRequest):
    patch = {k: v for k, v in req.dict().items() if k != "notebook_id" and v is not None}
    bm = update_bookmark(req.notebook_id, bookmark_id, patch)
    if bm is None:
        raise HTTPException(status_code=404, detail="Закладка не найдена")
    return bm


@app.delete("/api/bookmarks/{bookmark_id}")
async def api_delete_bookmark(bookmark_id: str, notebook_id: str = Query(...)):
    ok = delete_bookmark(notebook_id, bookmark_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Закладка не найдена")
    return {"status": "ok"}

# ── Управление GGUF моделями ──

@app.get("/api/gguf-models")
async def api_scan_gguf_models():
    """Сканирует директории и возвращает список доступных GGUF моделей."""
    try:
        models = scan_gguf_dirs()
        return {"models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/gguf-loaded")
async def api_gguf_loaded_models():
    """Возвращает список загруженных в память GGUF моделей."""
    loaded = get_loaded_models()
    return {"loaded_models": [os.path.basename(p) for p in loaded]}

@app.get("/api/gguf-status")
async def api_gguf_status():
    """Возвращает количество запущенных серверов."""
    count = count_running_servers()
    return {"running_count": count}

@app.post("/api/gguf-unload")
async def api_gguf_unload_all():
    """Выгружает все GGUF модели из памяти."""
    unload_all_models()
    return {"status": "ok", "msg": "Все модели выгружены"}

@app.post("/api/gguf-kill-all")
async def api_gguf_kill_all():
    """Принудительно завершает все процессы llama-server.exe в системе."""
    kill_stray_servers()
    unload_all_models() # Очищаем внутренний стейт тоже
    return {"status": "ok", "msg": "Все процессы llama-server завершены"}


@app.get("/api/vram")
async def api_vram():
    """Возвращает текущее использование VRAM + список загруженных моделей.

    F-fix #9: после инцидента с 6.5GB утечки пользователь не видел, что именно
    держит память. Этот endpoint даёт прозрачность: какие процессы llama-server
    запущены, сколько VRAM они занимают (по данным nvidia-smi), и каков общий
    расход vs свободная VRAM.
    """
    import subprocess
    used_mib = 0
    total_mib = 0
    free_mib = 0
    gpu_name = "unknown"
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.free,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3
        )
        if out.returncode == 0 and out.stdout.strip():
            parts = [p.strip() for p in out.stdout.strip().split(",")]
            if len(parts) >= 4:
                gpu_name = parts[0]
                used_mib = int(parts[1])
                free_mib = int(parts[2])
                total_mib = int(parts[3])
    except Exception as e:
        logger.debug(f"[VRAM] nvidia-smi query failed: {e}")

    # Per-process VRAM (если nvidia-smi поддерживает --query-compute-apps)
    per_process = []
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3
        )
        if out.returncode == 0:
            for line in out.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 3:
                    try:
                        per_process.append({
                            "pid": int(parts[0]),
                            "name": parts[1],
                            "vram_mib": int(parts[2]),
                        })
                    except (ValueError, IndexError):
                        continue
    except Exception:
        pass

    # Какие GGUF-серверы мы знаем (по нашему внутреннему реестру)
    from src.gguf_direct import get_loaded_models
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
        "gguf_servers": known_servers,
    }


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


@app.post("/api/preload-llm")
async def api_preload_llm(request: PreloadLlmRequest):
    """
    Hot-swap LLM модели: запускает в фоне, возвращает сразу.
    UI следит за прогрессом через /api/llm-status/stream.
    """
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


@app.get("/api/llm-status")
async def api_llm_status():
    """Возвращает текущее состояние LLM (idle/loading/ready/error) + прогресс."""
    return get_llm_status()


@app.get("/api/llm-status/stream")
async def api_llm_status_stream():
    """SSE поток обновлений статуса LLM (отправляет только при изменениях)."""
    from fastapi.responses import StreamingResponse
    import json as _json

    async def event_gen():
        last_key = None
        try:
            while True:
                st = get_llm_status()
                # Ключ — это то, что влияет на UI
                key = (st.get("state"), st.get("phase"), st.get("port"), st.get("error"))
                if key != last_key:
                    payload = _json.dumps(st, ensure_ascii=False)
                    yield f"data: {payload}\n\n"
                    last_key = key
                    # Если достигли терминального состояния — закрываем поток
                    if st.get("state") in ("ready", "error") and not (st.get("state") == "ready" and st.get("phase") == "loading"):
                        # При ready продолжаем слать, чтобы при следующей загрузке фронт получил сигнал
                        pass
                await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            pass

    return StreamingResponse(event_gen(), media_type="text/event-stream")

@app.get("/api/gguf-config")
async def api_get_gguf_config():
    """Возвращает текущие настройки GGUF из конфига."""
    return {
        "search_dirs": config.GGUF_SEARCH_DIRS,
        "default_ctx_size": config.GGUF_CTX_SIZE,
        "default_gpu_layers": config.GGUF_GPU_LAYERS,
        "default_threads": config.GGUF_THREADS,
    }

class UpdateModelDirsRequest(BaseModel):
    dirs: str

@app.post("/api/update-model-dirs")
async def update_model_dirs(req: UpdateModelDirsRequest):
    """Обновляет директории поиска моделей в реальном времени."""
    config.GGUF_SEARCH_DIRS = req.dirs
    config.save_rag_config() # Сохраняем на диск
    # Кеш scan-а привязан к старым путям — сбрасываем, чтобы новые директории попали в результат.
    try:
        from src.gguf_manager import invalidate_scan_cache
        invalidate_scan_cache()
    except Exception:
        pass
    return {"status": "ok", "new_dirs": config.GGUF_SEARCH_DIRS}

# ── Настройки RAG ──

@app.get("/api/rag-config")
async def get_rag_config():
    return {
        "embedding_model": config.EMBEDDING_MODEL_NAME,
        "reranker_model": config.RERANKER_MODEL_NAME,
        "quantization": config.QUANTIZATION,
        "top_k_per_file": config.RAG_TOP_K_PER_FILE,
        "rerank_pool": config.RAG_RERANK_POOL,
        "final_top_n": config.RAG_FINAL_TOP_N,
        "use_reranker": config.USE_RERANKER
    }

class UpdateRagConfigRequest(BaseModel):
    embedding_model: str
    reranker_model: str
    quantization: str
    top_k_per_file: int
    rerank_pool: int
    final_top_n: int
    use_reranker: bool

@app.post("/api/update-rag-config")
async def update_rag_config(req: UpdateRagConfigRequest):
    from src.rag_pipeline import unload_rag_models
    config.EMBEDDING_MODEL_NAME = req.embedding_model
    config.RERANKER_MODEL_NAME = req.reranker_model
    config.QUANTIZATION = req.quantization
    config.RAG_TOP_K_PER_FILE = req.top_k_per_file
    config.RAG_RERANK_POOL = req.rerank_pool
    config.RAG_FINAL_TOP_N = req.final_top_n
    config.USE_RERANKER = req.use_reranker
    config.save_rag_config() # Сохраняем на диск
    unload_rag_models() # Выгружаем старые модели, чтобы новые загрузились при следующем запросе
    return {"status": "ok"}

# ── Чат ──

class ChatRequest(BaseModel):
    query: str
    allowed_files: List[str]
    max_tokens: int = 2048
    notebook_id: str
    thinking_mode: bool = False
    llm_url: Optional[str] = None
    llm_api_key: Optional[str] = "lm-studio"
    llm_model: Optional[str] = "gpt-4o"
    image_base64: Optional[str] = None # Поле для фото
    # Режим ответа: "concise" (сначала коротко, потом объяснение) или
    # "detailed" (сразу развёрнуто). По умолчанию — concise.
    answer_mode: Optional[str] = "concise"

    # Расширенные параметры
    gguf_kv_quant: Optional[int] = 2 # 2=Q4_K, 8=Q8_0
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    repeat_penalty: Optional[float] = 1.1
    top_p: Optional[float] = 0.95
    min_p: Optional[float] = 0.05
    # GGUF параметры
    use_gguf: Optional[str] = None
    gguf_model_path: Optional[str] = None
    gguf_mmproj_path: Optional[str] = None
    gguf_temperature: Optional[float] = 0.7
    gguf_ctx_size: Optional[int] = 32768
    gguf_gpu_layers: Optional[int] = -1
    gguf_threads: Optional[int] = 8
    gguf_batch_size: Optional[int] = 512
    gguf_ubatch_size: Optional[int] = 256
    gguf_flash_attn: Optional[str] = "true"
    thinking_budget: Optional[int] = 1024 # -1 = без ограничений
    context_strategy: Optional[str] = "sliding" # sliding | rag_priority
    mtp_enabled: Optional[bool] = False # Multi-Token Prediction (--spec-type draft-mtp)

@app.post("/api/chat")
async def chat(request: ChatRequest):
    import time
    global_start_time = time.time()
    print(f"DEBUG: Запрос чата. Стратегия контекста: {request.context_strategy}, Лимит токенов: {request.max_tokens}, Бюджет рассуждений: {request.thinking_budget}")
    
    if not request.allowed_files:
        async def no_files():
            yield f"data: {json.dumps({'type': 'sources', 'sources': []})}\n\n"
            yield f"data: {json.dumps({'type': 'chunk', 'text': 'Пожалуйста, выберите хотя бы один источник.'})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(no_files(), media_type="text/event-stream")

    # 1. Сначала выполняем RAG (поиск чанков), пока LLM еще не заняла всю память
    from src.rag_pipeline import retrieve_nodes, build_file_context
    
    query_for_rag = request.query
    nodes = []
    sources = []
    context = ""
    
    # Пропускаем ранний RAG если есть картинка (чтобы не грузить RAG-модели дважды),
    # так как после OCR мы всё равно выполним RAG-поиск с объединенным запросом.
    skip_initial_rag = request.image_base64 and request.use_gguf == "true" and request.gguf_model_path
    
    if query_for_rag.strip() and not skip_initial_rag:
        print(f"DEBUG: Запуск RAG поиска для: {query_for_rag[:50]}...")
        # Выполняем тяжелый поиск в отдельном потоке, чтобы не блокировать Event Loop
        nodes = await asyncio.to_thread(retrieve_nodes, query_for_rag, request.notebook_id, request.allowed_files)
        sources, context = await asyncio.to_thread(build_file_context, nodes, request.notebook_id)
        print(f"DEBUG: RAG нашёл {len(nodes)} фрагментов.")

    # 2. Теперь определяем, какой LLM использовать и загружаем его
    active_llm = None
    use_direct_gguf = False
    
    if request.use_gguf == "true" and request.gguf_model_path:
        use_direct_gguf = True
        try:
            print(f"DEBUG: Подготовка GGUF модели: {os.path.basename(request.gguf_model_path)}")
            active_llm = await asyncio.to_thread(
                get_gguf_llm,
                gguf_path=request.gguf_model_path,
                mmproj_path=request.gguf_mmproj_path if request.gguf_mmproj_path else None,
                temperature=request.gguf_temperature,
                ctx_size=request.gguf_ctx_size,
                gpu_layers=request.gguf_gpu_layers,
                n_threads=request.gguf_threads,
                n_batch=request.gguf_batch_size,
                n_ubatch=request.gguf_ubatch_size,
                flash_attn=True if request.gguf_flash_attn == "true" else False,
                max_tokens=request.max_tokens,
                type_k=request.gguf_kv_quant,
                type_v=request.gguf_kv_quant,
                enable_thinking=request.thinking_mode,
                thinking_budget=request.thinking_budget,
                mtp_enabled=request.mtp_enabled
            )
            config.save_last_model(request.gguf_model_path, request.gguf_mmproj_path)
        except Exception as e:
            error_msg = f"Ошибка загрузки LLM: {str(e)}"
            async def error_gen():
                yield f"data: {json.dumps({'type': 'error', 'text': error_msg}, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(error_gen(), media_type="text/event-stream")
    elif request.llm_url:
        from llama_index.llms.openai import OpenAI
        active_llm = OpenAI(
            api_base=request.llm_url,
            api_key=request.llm_api_key or "lm-studio",
            model=request.llm_model or "gpt-4o",
            temperature=0.1,
            max_tokens=request.max_tokens
        )
    else:
        active_llm = Settings.llm

    # 3. Генерация ответа
    async def generate():
        start_time = time.time()
        nonlocal query_for_rag, sources, context
        token_count = 0
        try:
            # Обработка изображения (OCR), всегда добавляем текст с картинки в запрос
            if request.image_base64 and use_direct_gguf:
                vision_messages = [
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{request.image_base64}"}},
                        {"type": "text", "text": "Ты — профессиональный инструмент распознавания текста и анализа изображений. Сделай точный OCR всего текста на изображении, включая все вопросы, задания, формулы и таблицы. Если на картинке есть важные графики, схемы или диаграммы, кратко опиши их суть текстом. Выведи только результат распознавания без лишних слов."}
                    ]}
                ]
                try:
                    v_payload = {"messages": vision_messages, "stream": False, "max_tokens": 1024}
                    r_vision = await asyncio.to_thread(_http_session.post, f"{active_llm}/v1/chat/completions", json=v_payload, timeout=60)
                    extracted = r_vision.json()["choices"][0]["message"]["content"].strip()
                    
                    print(f"\n  [OCR] 📷 Распознанный текст с изображения:")
                    print("=" * 60)
                    print(extracted)
                    print("=" * 60 + "\n")
                    
                    # Чистый запрос для RAG (без инструкций, чтобы не ломать поиск)
                    if request.query.strip():
                        search_query = f"{request.query.strip()} {extracted}"
                        query_for_rag = f"{request.query.strip()}\n\nТекст на картинке:\n{extracted}"
                    else:
                        search_query = extracted
                        query_for_rag = f"Пожалуйста, подробно ответь на вопросы или выполни задания, которые представлены на изображении. Вот распознанный текст для удобства:\n{extracted}"
                        
                    nodes = await asyncio.to_thread(retrieve_nodes, search_query, request.notebook_id, request.allowed_files)
                    sources, context = await asyncio.to_thread(build_file_context, nodes, request.notebook_id)
                except Exception as ve: print(f"DEBUG: Ошибка OCR: {ve}")

            if not query_for_rag or not query_for_rag.strip():
                query_for_rag = request.query or "Опиши содержимое"

            yield f"data: {json.dumps({'type': 'sources', 'sources': sources}, ensure_ascii=False)}\n\n"

            if use_direct_gguf:
                sys_prompt = config.get_system_prompt(request.answer_mode) + f"\n\nДоступные источники:\n{context}"
                
                if request.image_base64:
                    user_content = [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{request.image_base64}"}},
                        {"type": "text", "text": query_for_rag}
                    ]
                    messages_for_chat = [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_content}
                    ]
                else:
                    messages_for_chat = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": query_for_rag}]
                model_family = detect_model_family(request.gguf_model_path)
                OPEN_TAG, CLOSE_TAG = ("<|channel|>", "<channel|>") if model_family == "gemma4" else ("<think>", "</think>")
                
                # ВСЕГДА начинаем с режима детекции, чтобы не пропустить начало ответа, 
                # если модель решила не рассуждать или если теги приходят позже.
                phase = "think_detect"
                
                buf = ""
                # Запускаем стриминг в потоке и оборачиваем в асинхронный генератор
                async_gen = stream_gguf_chat(
                    llm_url=active_llm, messages=messages_for_chat, enable_thinking=request.thinking_mode,
                    max_tokens=request.max_tokens, temperature=request.gguf_temperature,
                    repeat_penalty=request.repeat_penalty, top_p=request.top_p, min_p=request.min_p,
                    model_family=model_family
                )
                
                async for delta in async_gen:
                    if not delta: continue
                    token_count += 1
                    buf += delta
                    if phase == "think_detect":
                        if OPEN_TAG in buf:
                            if request.thinking_mode:
                                buf = buf[buf.index(OPEN_TAG) + len(OPEN_TAG):]
                                yield f"data: {json.dumps({'type': 'thinking_start'}, ensure_ascii=False)}\n\n"; phase = "thinking"
                            else:
                                # Если режим выключен, но тег пришел — переходим в режим игнорирования мыслей
                                buf = buf[buf.index(OPEN_TAG) + len(OPEN_TAG):]
                                phase = "thinking_ignore"
                        elif len(buf) > 10:
                            phase = "answer"; yield f"data: {json.dumps({'type': 'chunk', 'text': buf}, ensure_ascii=False)}\n\n"; buf = ""
                    
                    if phase == "thinking_ignore":
                        if CLOSE_TAG in buf:
                            _, _, rest = buf.partition(CLOSE_TAG)
                            buf = rest.lstrip("\n")
                            phase = "answer"
                        else:
                            # Просто очищаем буфер, так как это "мысли", которые мы не хотим показывать
                            if len(buf) > len(CLOSE_TAG):
                                buf = buf[-len(CLOSE_TAG):]
                            continue

                    if phase == "thinking":
                        if CLOSE_TAG in buf:
                            think_part, _, rest = buf.partition(CLOSE_TAG)
                            if think_part: yield f"data: {json.dumps({'type': 'thinking_chunk', 'text': think_part}, ensure_ascii=False)}\n\n"
                            yield f"data: {json.dumps({'type': 'thinking_done'}, ensure_ascii=False)}\n\n"; phase = "answer"
                            buf = rest.lstrip("\n")
                            if buf: yield f"data: {json.dumps({'type': 'chunk', 'text': buf}, ensure_ascii=False)}\n\n"; buf = ""
                        else:
                            safe = buf[:-len(CLOSE_TAG)] if len(buf) > len(CLOSE_TAG) else ""
                            if safe: yield f"data: {json.dumps({'type': 'thinking_chunk', 'text': safe}, ensure_ascii=False)}\n\n"; buf = buf[len(safe):]
                    elif phase == "answer":
                        yield f"data: {json.dumps({'type': 'chunk', 'text': buf}, ensure_ascii=False)}\n\n"
                        buf = ""

                # Дочищаем буфер
                if buf and phase == "thinking":
                    yield f"data: {json.dumps({'type': 'thinking_chunk', 'text': buf}, ensure_ascii=False)}\n\n"
                    yield f"data: {json.dumps({'type': 'thinking_done'}, ensure_ascii=False)}\n\n"
                elif buf and phase == "answer":
                    yield f"data: {json.dumps({'type': 'chunk', 'text': buf}, ensure_ascii=False)}\n\n"
                print("[GGUF Chat] Генерация завершена.")

            else:
                # Для API (LM Studio) используем стандартный prompt
                full_response = ""
                prompt = make_prompt(request.query, context, thinking_mode=request.thinking_mode, max_tokens=request.max_tokens, answer_mode=request.answer_mode)
                for chunk in active_llm.stream_complete(prompt):
                    if chunk.delta:
                        token_count += 1 
                        full_response += chunk.delta
                        yield f"data: {json.dumps({'type': 'chunk', 'text': chunk.delta}, ensure_ascii=False)}\n\n"
            # print(f"\n[CHAT] Ответ модели:\n{full_response}\n")

            elapsed = time.time() - start_time
            yield f"data: {json.dumps({
                'type': 'stats', 
                'elapsed_sec': round(elapsed, 2),
                'total_tokens': token_count,
                'tokens_per_sec': round(token_count / elapsed, 1) if elapsed > 0 else 0
            })}\n\n"
        
            yield "data: [DONE]\n\n"

        except Exception as e:
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'text': str(e)}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        finally:
            # Очистка только KV-слота в llama-server, не выгружая модели и не трогая CUDA-кэш.
            # gc.collect() + torch.cuda.empty_cache() УБРАНЫ: они занимают 200-800 мс на КАЖДЫЙ чат
            # и не нужны — модели живут в VRAM постоянно, а KV-cache освобождается через /slots/0/clear.
            if use_direct_gguf and active_llm:
                try:
                    # Сбрасываем состояние слота чтобы контекст не копился между запросами,
                    # но НЕ убиваем процесс сервера.
                    _http_session.post(f"{active_llm}/slots/0/clear", timeout=1)
                except Exception: pass

    return StreamingResponse(generate(), media_type="text/event-stream")

# Старый shutdown удалён

if __name__ == "__main__":
    import uvicorn
    # Отключаем логирование каждого HTTP-запроса (access_log=False)
    # и оставляем только предупреждения и ошибки (log_level="warning")
    uvicorn.run("main:app", host=config.HOST, port=config.PORT, reload=True, access_log=False, log_level="warning")
