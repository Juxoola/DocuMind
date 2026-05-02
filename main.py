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
from src.rag_pipeline import build_index, retrieve_nodes, build_file_context, make_prompt, close_all_clients, preload_all_models
from src.gguf_manager import scan_gguf_dirs
from src.gguf_direct import get_gguf_llm, unload_all_models, get_loaded_models
from fastapi.middleware.cors import CORSMiddleware
from llama_index.core import Settings

app = FastAPI(title="NotebookLM Local Clone")

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

    llm_settings = {
        "llm_url": effective_llm_url,
        "llm_api_key": effective_llm_api_key,
        "llm_model": effective_llm_model,
        "use_gguf_direct": use_gguf_direct,
        "gguf_model_path": gguf_model_path if use_gguf_direct else None,
        "gguf_mmproj_path": gguf_mmproj_path if use_gguf_direct else None,
    }

    def process_task():
        try:
            def prog(pct, msg):
                q.put({"type": "progress", "pct": pct, "msg": msg})

            prog(5, "Файл сохранён, подготовка...")
            nodes = ingest_file(file_path, notebook_id, progress_cb=prog, llm_settings=llm_settings)
            
            prog(90, "Построение индекса (ChromaDB)...")
            build_index(nodes, notebook_id)
            
            q.put({"type": "done", "filename": file.filename})
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
    paths = config.get_notebook_paths(notebook_id)
    file_path = os.path.join(paths["data"], filename)
    if os.path.exists(file_path):
        os.remove(file_path)
    
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

# ── Chat ──

class ChatRequest(BaseModel):
    query: str
    allowed_files: List[str]
    max_tokens: int = 1024
    notebook_id: str
    llm_url: Optional[str] = None
    llm_api_key: Optional[str] = "lm-studio"
    llm_model: Optional[str] = "gpt-4o"
    # GGUF параметры
    use_gguf: Optional[str] = None
    gguf_model_path: Optional[str] = None
    gguf_mmproj_path: Optional[str] = None

@app.post("/api/chat")
async def chat(request: ChatRequest):
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
                temperature=0.1,
                max_tokens=request.max_tokens,
            )
        except Exception as e:
            async def error_gen():
                error_msg = f"Ошибка загрузки GGUF модели: {str(e)}"
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
        start_time = time.time()
        token_count = 0
        try:
            nodes = retrieve_nodes(
                query=request.query,
                notebook_id=request.notebook_id,
                allowed_files=request.allowed_files,
                max_tokens=request.max_tokens
            )
            sources, context_str = build_file_context(nodes, request.notebook_id)
            prompt = make_prompt(request.query, context_str)

            yield f"data: {json.dumps({'type': 'sources', 'sources': sources}, ensure_ascii=False)}\n\n"

            full_response = ""
            for chunk in active_llm.stream_complete(prompt):
                if chunk.delta:
                    token_count += 1 
                    full_response += chunk.delta
                    yield f"data: {json.dumps({'type': 'chunk', 'text': chunk.delta}, ensure_ascii=False)}\n\n"
            
            print(f"\n[CHAT] Ответ модели:\n{full_response}\n")

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

@app.on_event("shutdown")
async def shutdown_event():
    """Выгружаем GGUF модели из памяти при выключении."""
    unload_all_models()

if __name__ == "__main__":
    import uvicorn
    # Предзагрузка моделей перед запуском
    try:
        preload_all_models()
    except Exception as e:
        print(f"[ERROR] Ошибка предзагрузки моделей: {e}")
        
    uvicorn.run("main:app", host=config.HOST, port=config.PORT, reload=True)
