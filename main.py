import os
import shutil
import json
import time
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
import config
import gc
import stat

from src.ingestion import ingest_file
from src.rag_pipeline import build_index, retrieve_nodes, build_file_context, make_prompt, make_messages, close_all_clients, preload_all_models
from src.gguf_manager import scan_gguf_dirs
from src.gguf_direct import (
    get_gguf_llm, unload_all_models, get_loaded_models,
    detect_model_family, stream_gguf_chat
)
from fastapi.middleware.cors import CORSMiddleware
from llama_index.core import Settings
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Фоновая предзагрузка моделей (сервер запустится мгновенно)
    import threading
    from src.rag_pipeline import preload_all_models
    threading.Thread(target=preload_all_models, daemon=True).start()
    
    yield
    
    # Shutdown: Выгрузка
    print("[SERVER] Остановка системы...")
    unload_all_models()

app = FastAPI(title="NotebookLM Local Clone", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
            except: pass
        for d in dirs:
            try: os.chmod(os.path.join(root, d), stat.S_IWRITE)
            except: pass

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

@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open(os.path.join(config.BASE_DIR, "static", "index.html"), "r", encoding="utf-8") as f:
        return f.read()

# ── Notebook Management ──

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

# ── File Operations ──

@app.get("/api/files")
async def get_files(notebook_id: str):
    paths = config.get_notebook_paths(notebook_id)
    if os.path.exists(paths["data"]):
        files = [f for f in os.listdir(paths["data"]) if not f.endswith(".json")]
    else:
        files = []
    return {"files": files}

@app.post("/api/upload")
async def upload_file(
    notebook_id: str, 
    file: UploadFile = File(...),
    llm_url: Optional[str] = None,
    llm_api_key: Optional[str] = None,
    llm_model: Optional[str] = None,
    use_gguf: Optional[str] = None,
    gguf_model_path: Optional[str] = None,
    gguf_mmproj_path: Optional[str] = None,
    vision_model_path: Optional[str] = None,
    vision_mmproj_path: Optional[str] = None,
    vision_temperature: Optional[float] = 0.2,
    vision_ctx_size: Optional[int] = 8192,
    vision_gpu_layers: Optional[int] = -1,
    vision_threads: Optional[int] = 8,
    vision_batch_size: Optional[int] = 2048,
    vision_flash_attn: Optional[str] = "true",
    vision_max_tokens: Optional[int] = 512,
    vision_concurrency: Optional[int] = 1,
):
    paths = config.get_notebook_paths(notebook_id)
    os.makedirs(paths["data"], exist_ok=True)
    file_path = os.path.join(paths["data"], file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    import threading
    import queue
    import asyncio

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
        "vision_flash_attn": vision_flash_attn,
        "vision_max_tokens": vision_max_tokens,
        "vision_concurrency": vision_concurrency,
    }

    def process_task():
        import time
        start_time = time.time()
        try:
            def prog(pct, msg):
                q.put({"type": "progress", "pct": pct, "msg": msg})

            prog(5, "Файл сохранён, подготовка...")
            nodes = ingest_file(file_path, notebook_id, progress_cb=prog, llm_settings=llm_settings)
            
            prog(90, "Построение индекса (ChromaDB)...")
            build_index(nodes, notebook_id)
            
            elapsed = time.time() - start_time
            mins = int(elapsed // 60)
            secs = int(elapsed % 60)
            time_str = f"{mins}м {secs}с" if mins > 0 else f"{secs}с"
            print(f"[INGESTION] Файл '{file.filename}' успешно добавлен в базу. Затрачено времени: {time_str}")
            
            q.put({"type": "done", "filename": file.filename, "elapsed": time_str, "elapsed_sec": elapsed})
        except Exception as e:
            import traceback
            traceback.print_exc()
            q.put({"type": "error", "msg": str(e)})

    threading.Thread(target=process_task, daemon=True).start()

    async def event_generator():
        while True:
            while q.empty():
                await asyncio.sleep(0.1)
            msg = q.get()
            yield f"data: {json.dumps(msg, ensure_ascii=False)}\n\n"
            if msg["type"] in ["done", "error"]:
                break

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.delete("/api/files/{filename}")
async def delete_file(filename: str, notebook_id: str):
    import gc, time
    import cv2
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
    return {"status": "ok"}

@app.get("/api/source_content")
async def get_source_content(filename: str, notebook_id: str):
    try:
        from src.rag_pipeline import get_vector_store
        vector_store = get_vector_store(notebook_id)
        collection = vector_store._collection
        result = collection.get(where={"file_name": filename})
        if result and result.get("documents"):
            full_text = "\n\n---\n\n".join(result["documents"])
            return {"text": full_text}
        return {"text": "Содержимое документа не найдено в базе данных."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/video_metadata")
async def get_video_metadata(filename: str, notebook_id: str):
    paths = config.get_notebook_paths(notebook_id)
    json_path = os.path.join(paths["data"], f"{filename}.json")
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"error": "Metadata not found"}

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

# ── GGUF Model Management ──

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

@app.post("/api/gguf-unload")
async def api_gguf_unload_all():
    """Выгружает все GGUF модели из памяти."""
    unload_all_models()
    return {"status": "ok", "msg": "Все модели выгружены"}

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
    return {"status": "ok", "new_dirs": config.GGUF_SEARCH_DIRS}

# ── RAG Configuration ──

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

# ── Chat ──

class ChatRequest(BaseModel):
    query: str
    allowed_files: List[str]
    max_tokens: int = 1024
    notebook_id: str
    thinking_mode: bool = False
    llm_url: Optional[str] = None
    llm_api_key: Optional[str] = "lm-studio"
    llm_model: Optional[str] = "gpt-4o"
    image_base64: Optional[str] = None # Новое поле для фото
    
    # Расширенные параметры
    gguf_kv_quant: Optional[int] = 2 # 2=Q4_K, 8=Q8_0
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    repeat_penalty: Optional[float] = 1.1
    top_p: Optional[float] = 0.9
    min_p: Optional[float] = 0.05
    # GGUF параметры
    use_gguf: Optional[str] = None
    gguf_model_path: Optional[str] = None
    gguf_mmproj_path: Optional[str] = None
    gguf_temperature: Optional[float] = 0.1
    gguf_ctx_size: Optional[int] = 8192
    gguf_gpu_layers: Optional[int] = -1
    gguf_threads: Optional[int] = 8
    gguf_batch_size: Optional[int] = 2048
    gguf_flash_attn: Optional[str] = "false"

@app.post("/api/chat")
async def chat(request: ChatRequest):
    import time
    global_start_time = time.time()
    # Определяем, какой LLM использовать
    effective_llm_url = request.llm_url
    effective_llm_api_key = request.llm_api_key
    effective_llm_model = request.llm_model
    
    # Если выбрана GGUF модель — используем прямой API
    if request.use_gguf == "true" and request.gguf_model_path:
        try:
            print(f"DEBUG: Загрузка GGUF модели через прямой API: {request.gguf_model_path}")
            active_llm = get_gguf_llm(
                gguf_path=request.gguf_model_path,
                mmproj_path=request.gguf_mmproj_path if request.gguf_mmproj_path else None,
                temperature=request.gguf_temperature,
                ctx_size=request.gguf_ctx_size,
                gpu_layers=request.gguf_gpu_layers,
                n_threads=request.gguf_threads,
                n_batch=request.gguf_batch_size,
                flash_attn=True if request.gguf_flash_attn == "true" else False,
                max_tokens=request.max_tokens,
                type_k=request.gguf_kv_quant,
                type_v=request.gguf_kv_quant,
                enable_thinking=request.thinking_mode
            )
            config.save_last_model(request.gguf_model_path, request.gguf_mmproj_path)
            
            # RAG: Получаем релевантные чанки
            from src.rag_pipeline import retrieve_nodes, build_file_context
            
            query_for_rag = request.query
            if not query_for_rag.strip() and request.image_base64:
                print("DEBUG: Текст запроса пуст, извлекаем задание из изображения...")
                vision_messages = [
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{request.image_base64}"}},
                        {"type": "text", "text": "Твоя задача — выполнить точный OCR (распознавание текста) с изображения. Перепиши ВЕСЬ текст вопроса и вариантов ответа, который видишь на картинке. Не добавляй никаких пояснений, только сам текст задания."}
                    ]}
                ]
                try:
                    res = active_llm.create_chat_completion(messages=vision_messages, stream=False, max_tokens=150)
                    extracted_query = res["choices"][0]["message"]["content"].strip()
                    
                    # Очистка от "болтливости" модели (если она начала объяснять, что делает)
                    if "The user wants" in extracted_query or "Image Analysis" in extracted_query:
                        import re
                        # Ищем последнюю строку в кавычках или после двоеточия
                        lines = [l.strip() for l in extracted_query.split("\n") if l.strip()]
                        for line in reversed(lines):
                            if ":" in line and not line.startswith("http"):
                                extracted_query = line.split(":", 1)[1].strip().strip('"')
                                break
                            elif '"' in line:
                                matches = re.findall(r'"([^"]*)"', line)
                                if matches:
                                    extracted_query = matches[-1]
                                    break
                        else:
                            # Если ничего не нашли, берем последнюю осмысленную строку
                            extracted_query = lines[-1].strip().strip('"')

                    print(f"DEBUG: Извлеченное задание: {extracted_query}")
                    query_for_rag = extracted_query
                except Exception as ve:
                    print(f"DEBUG: Ошибка при извлечении текста из фото: {ve}")
            
            # Если запрос все еще пуст, используем дефолтный промпт для RAG
            if not query_for_rag or not query_for_rag.strip():
                query_for_rag = "Опиши содержимое изображения и найди связанные инструкции в документах"
            
            nodes = retrieve_nodes(query_for_rag, request.notebook_id, request.allowed_files)
            sources, context = build_file_context(nodes, request.notebook_id)
            
            # Формируем системный промпт с RAG контекстом
            sys_prompt = (
                "Ты — умный и точный AI-помощник. ОТВЕЧАЙ СТРОГО НА РУССКОМ ЯЗЫКЕ.\n"
                "Используй Markdown для форматирования.\n\n"
                "ПРАВИЛА ОТВЕТА:\n"
                "1. Всегда отвечай на вопрос СРАЗУ.\n"
                "2. Если вопрос содержит варианты ответа (тест) — СНАЧАЛА напиши правильный вариант.\n"
                "3. После прямого ответа приведи подробное объяснение на основе источников.\n\n"
                "ПРАВИЛО ЦИТИРОВАНИЯ:\n"
                "- Каждое утверждение ДОЛЖНО завершаться ссылкой [N].\n"
                "- Если ответа нет в источниках — скажи \"В документах этого нет\".\n\n"
                f"Доступные источники:\n{context}"
            )
            user_text = query_for_rag if query_for_rag else "Ответь на вопрос по изображению"
            print(f"DEBUG: RAG нашел {len(nodes)} фрагментов. Контекст подготовлен.")

            # Определяем семейство модели — для логирования
            model_family = detect_model_family(request.gguf_model_path)
            messages_for_chat = [
                {"role": "system", "content": sys_prompt},
                {"role": "user",   "content": user_text},
            ]
            print(f"DEBUG: Семейство модели: {model_family} | thinking: {request.thinking_mode}")

            def generate():
                try:
                    import time as _time
                    _gen_start = global_start_time
                    _token_count = 0
                    _answer_chars = 0
                    print("DEBUG: Запуск генерации...")
                    yield f"data: {json.dumps({'type': 'sources', 'sources': sources}, ensure_ascii=False)}\n\n"

                    if model_family == "gemma4":
                        OPEN_TAG  = "<|channel>"  # Gemma 4: \u043c\u044b\u0448\u043b\u0435\u043d\u0438\u0435 \u043d\u0430\u0447\u0438\u043d\u0430\u0435\u0442\u0441\u044f
                        CLOSE_TAG = "<channel|>"  # Gemma 4: \u043c\u044b\u0448\u043b\u0435\u043d\u0438\u0435 \u0437\u0430\u043a\u0430\u043d\u0447\u0438\u0432\u0430\u0435\u0442\u0441\u044f
                    else:
                        OPEN_TAG  = "<think>"
                        CLOSE_TAG = "</think>"
                    thinking_models = ("qwen", "deepseek", "generic")  # gemma4: think_detect

                    if request.thinking_mode and model_family in thinking_models:
                        phase = "thinking"
                        yield f"data: {json.dumps({'type': 'thinking_start'}, ensure_ascii=False)}\n\n"
                    else:
                        phase = "think_detect"
                    buf = ""

                    for delta in stream_gguf_chat(
                        llm=active_llm,
                        messages=messages_for_chat,
                        enable_thinking=request.thinking_mode,
                        max_tokens=request.max_tokens,
                        temperature=request.gguf_temperature,
                        repeat_penalty=request.repeat_penalty,
                        top_p=request.top_p,
                        min_p=request.min_p,
                    ):
                        if not delta:
                            continue
                        _token_count += 1
                        _answer_chars += len(delta)
                        buf += delta

                        if phase == "think_detect":
                            if OPEN_TAG in buf:
                                buf = buf[buf.index(OPEN_TAG) + len(OPEN_TAG):]
                                yield f"data: {json.dumps({'type': 'thinking_start'}, ensure_ascii=False)}\n\n"
                                phase = "thinking"
                            elif len(buf) > len(OPEN_TAG) + 4:
                                phase = "answer"
                                yield f"data: {json.dumps({'type': 'chunk', 'text': buf}, ensure_ascii=False)}\n\n"
                                buf = ""

                        if phase == "thinking":
                            if CLOSE_TAG in buf:
                                think_part, _, rest = buf.partition(CLOSE_TAG)
                                if think_part:
                                    yield f"data: {json.dumps({'type': 'thinking_chunk', 'text': think_part}, ensure_ascii=False)}\n\n"
                                yield f"data: {json.dumps({'type': 'thinking_done'}, ensure_ascii=False)}\n\n"
                                phase = "answer"
                                buf = rest.lstrip("\n")  # убираем переводы строки между </think> и ответом
                                if buf:
                                    yield f"data: {json.dumps({'type': 'chunk', 'text': buf}, ensure_ascii=False)}\n\n"
                                    buf = ""
                            else:
                                safe = buf[:-len(CLOSE_TAG)] if len(buf) > len(CLOSE_TAG) else ""
                                if safe:
                                    yield f"data: {json.dumps({'type': 'thinking_chunk', 'text': safe}, ensure_ascii=False)}\n\n"
                                    buf = buf[len(safe):]

                        elif phase == "answer":
                            yield f"data: {json.dumps({'type': 'chunk', 'text': buf}, ensure_ascii=False)}\n\n"
                            buf = ""

                    # Дочищаем буфер
                    if buf and phase == "thinking":
                        yield f"data: {json.dumps({'type': 'thinking_chunk', 'text': buf}, ensure_ascii=False)}\n\n"
                        yield f"data: {json.dumps({'type': 'thinking_done'}, ensure_ascii=False)}\n\n"
                    elif buf and phase == "answer":
                        yield f"data: {json.dumps({'type': 'chunk', 'text': buf}, ensure_ascii=False)}\n\n"

                    _elapsed = _time.time() - _gen_start
                    # Приближенное кол-во токенов через длину (÷4 англ/÷2 рус)
                    _est_tokens = max(_token_count, _answer_chars // 3)
                    yield f"data: {json.dumps({'type': 'stats', 'elapsed_sec': round(_elapsed, 2), 'total_tokens': _est_tokens, 'tokens_per_sec': round(_est_tokens / _elapsed, 1) if _elapsed > 0 else 0}, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"
                    print(f"DEBUG: Генерация завершена. {_est_tokens} tok, {round(_elapsed,1)}s, {round(_est_tokens/_elapsed,1) if _elapsed>0 else 0} tok/s")
                except Exception as e:
                    yield f"data: {json.dumps({'type': 'error', 'text': str(e)}, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"
            
            return StreamingResponse(generate(), media_type="text/event-stream")
        except Exception as e:
            error_msg = str(e)
            async def error_gen():
                yield f"data: {json.dumps({'type': 'error', 'text': error_msg}, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(error_gen(), media_type="text/event-stream")
    elif effective_llm_url:
        from llama_index.llms.openai import OpenAI
        print(f"DEBUG: Используем LLM: {effective_llm_url} (model: {effective_llm_model})")
        active_llm = OpenAI(
            api_base=effective_llm_url,
            api_key=effective_llm_api_key or "lm-studio",
            model=effective_llm_model or "gpt-4o",
            temperature=0.1,
            max_tokens=request.max_tokens
        )
    else:
        print(f"DEBUG: Используем системный LLM")
        active_llm = Settings.llm

    if not request.allowed_files:
        async def no_files():
            yield f"data: {json.dumps({'type': 'sources', 'sources': []})}\n\n"
            yield f"data: {json.dumps({'type': 'chunk', 'text': 'Пожалуйста, выберите хотя бы один источник.'})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(no_files(), media_type="text/event-stream")

    async def generate():
        start_time = global_start_time
        token_count = 0
        try:
            nodes = retrieve_nodes(
                query=request.query,
                notebook_id=request.notebook_id,
                allowed_files=request.allowed_files,
                max_tokens=request.max_tokens
            )
            sources, context_str = build_file_context(nodes, request.notebook_id)
            
            yield f"data: {json.dumps({'type': 'sources', 'sources': sources}, ensure_ascii=False)}\n\n"

            full_response = ""
            
            if request.use_gguf == "true":
                messages_for_chat = make_messages(request.query, context_str)
                model_family = detect_model_family(request.gguf_model_path)
                print(f"[GGUF Chat] Семейство: {model_family} | thinking: {request.thinking_mode}")

                if model_family == "gemma4":
                    OPEN_TAG  = "<|channel>"  # Gemma 4: \u043c\u044b\u0448\u043b\u0435\u043d\u0438\u0435 \u043d\u0430\u0447\u0438\u043d\u0430\u0435\u0442\u0441\u044f
                    CLOSE_TAG = "<channel|>"  # Gemma 4: \u043c\u044b\u0448\u043b\u0435\u043d\u0438\u0435 \u0437\u0430\u043a\u0430\u043d\u0447\u0438\u0432\u0430\u0435\u0442\u0441\u044f
                else:
                    OPEN_TAG  = "<think>"
                    CLOSE_TAG = "</think>"
                thinking_models = ("qwen", "deepseek", "generic")

                if request.thinking_mode and model_family in thinking_models:
                    phase = "thinking"
                    yield f"data: {json.dumps({'type': 'thinking_start'}, ensure_ascii=False)}\n\n"
                else:
                    phase = "think_detect"
                buf = ""

                for delta in stream_gguf_chat(
                    llm=active_llm,
                    messages=messages_for_chat,
                    enable_thinking=request.thinking_mode,
                    max_tokens=request.max_tokens,
                    temperature=request.gguf_temperature,
                    repeat_penalty=request.repeat_penalty,
                    top_p=request.top_p,
                    min_p=request.min_p,
                ):
                    if not delta:
                        continue
                    buf += delta
                    token_count += 1   # каждый delta ≈ 1 токен
                    full_response += delta

                    if phase == "think_detect":
                        if OPEN_TAG in buf:
                            buf = buf[buf.index(OPEN_TAG) + len(OPEN_TAG):]
                            yield f"data: {json.dumps({'type': 'thinking_start'}, ensure_ascii=False)}\n\n"
                            phase = "thinking"
                        elif len(buf) > len(OPEN_TAG) + 4:
                            phase = "answer"
                            yield f"data: {json.dumps({'type': 'chunk', 'text': buf}, ensure_ascii=False)}\n\n"
                            buf = ""

                    if phase == "thinking":
                        if CLOSE_TAG in buf:
                            think_part, _, rest = buf.partition(CLOSE_TAG)
                            if think_part:
                                yield f"data: {json.dumps({'type': 'thinking_chunk', 'text': think_part}, ensure_ascii=False)}\n\n"
                            yield f"data: {json.dumps({'type': 'thinking_done'}, ensure_ascii=False)}\n\n"
                            phase = "answer"
                            buf = rest.lstrip("\n")
                            if buf:
                                yield f"data: {json.dumps({'type': 'chunk', 'text': buf}, ensure_ascii=False)}\n\n"
                                buf = ""
                        else:
                            safe = buf[:-len(CLOSE_TAG)] if len(buf) > len(CLOSE_TAG) else ""
                            if safe:
                                yield f"data: {json.dumps({'type': 'thinking_chunk', 'text': safe}, ensure_ascii=False)}\n\n"
                                buf = buf[len(safe):]

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
                prompt = make_prompt(request.query, context_str, thinking_mode=request.thinking_mode, max_tokens=request.max_tokens)
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
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'text': str(e)}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

# Убрали старый shutdown

if __name__ == "__main__":
    import uvicorn
    # Отключаем логирование каждого HTTP-запроса (access_log=False)
    # и оставляем только предупреждения и ошибки (log_level="warning")
    uvicorn.run("main:app", host=config.HOST, port=config.PORT, reload=True, access_log=False, log_level="warning")
