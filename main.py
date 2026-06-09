"""
NotebookLM Local Clone — основной модуль.

Запуск: python main.py

После рефакторинга (v2) — только создание приложения,
lifespan, middleware и подключение роутеров.
"""
import os
import sys
import logging
import atexit
import threading
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

import config
from routers.shared import _background_tasks

logger = logging.getLogger(__name__)

# ── Контроль повторного cleanup при --reload ──
_lifespan_cleanup_done = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _lifespan_cleanup_done

    # Cleanup остатков .pending_delete_* от прошлой сессии
    try:
        import glob
        for pending in glob.glob(os.path.join(config.NOTEBOOKS_DIR, "*.pending_delete_*")):
            print(f"[STARTUP] Удаляю отложенную папку: {pending}")
            from routers.shared import robust_rmtree
            success, err = robust_rmtree(pending)
            if not success:
                print(f"[STARTUP] Не удалось удалить {pending}: {err}")
    except Exception as e:
        print(f"[STARTUP] Ошибка cleanup pending_delete: {e}")

    # Миграция старых данных
    from routers.notebooks import migrate_old_data
    migrate_old_data()

    # Фоновая предзагрузка моделей
    threading.Thread(target=preload_all_models, daemon=True).start()

    yield

    if not _lifespan_cleanup_done:
        _lifespan_cleanup_done = True
        print("[SERVER] Остановка системы...")
        from src.gguf_direct import unload_all_models, kill_stray_servers
        unload_all_models()
        kill_stray_servers()


def preload_all_models():
    """Предзагрузка эмбеддингов и реранкера в фоне."""
    try:
        from src.rag_pipeline import preload_all_models as _preload
        _preload()
    except Exception as e:
        print(f"[STARTUP] Предзагрузка моделей не удалась: {e}")


# atexit — гарантированная очистка на Windows
from src.gguf_direct import unload_all_models, kill_stray_servers
atexit.register(unload_all_models)
atexit.register(kill_stray_servers)

# ── Создаём приложение ──
app = FastAPI(title="NotebookLM Local Clone", lifespan=lifespan)

# ── Middleware: лимит загрузки ──
@app.middleware("http")
async def enforce_upload_size(request, call_next):
    if request.url.path.startswith("/api/upload"):
        cl = request.headers.get("content-length")
        if cl and cl.isdigit() and int(cl) > config.UPLOAD_MAX_SIZE_BYTES:
            mb = int(cl) / (1024 * 1024)
            return JSONResponse(
                status_code=413,
                content={
                    "detail": (
                        f"Файл слишком большой: {mb:.1f} МБ. "
                        f"Лимит: {config.UPLOAD_MAX_SIZE_MB} МБ."
                    )
                },
            )
    return await call_next(request)


# ── CORS ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Статика ──
os.makedirs(os.path.join(config.BASE_DIR, "static"), exist_ok=True)
app.mount("/static", StaticFiles(directory=os.path.join(config.BASE_DIR, "static")), name="static")
app.mount("/files", StaticFiles(directory=config.NOTEBOOKS_DIR), name="notebooks")

# ── Роутеры ──
from routers.notebooks import router as notebooks_router
from routers.files import router as files_router
from routers.chat import router as chat_router
from routers.gguf import router as gguf_router
from routers.bookmarks import router as bookmarks_router
from routers.settings import router as settings_router

app.include_router(notebooks_router)
app.include_router(files_router)
app.include_router(chat_router)
app.include_router(gguf_router)
app.include_router(bookmarks_router)
app.include_router(settings_router)

# ── Точка входа ──
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=config.HOST,
        port=config.PORT,
        reload=config.RELOAD,
        access_log=False,
        log_level="warning",
    )
