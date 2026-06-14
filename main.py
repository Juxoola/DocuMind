"""DocuMind — основной модуль."""

import logging
import os
import sys
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

import config

_LOG_DIR = os.path.join(config.BASE_DIR, "logs")
os.makedirs(_LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(os.path.join(_LOG_DIR, "server.log"), encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)
logging.getLogger("llama_index").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Lifespan-менеджер: cleanup pending_delete, миграция старых данных, фоновая предзагрузка моделей.

    logger.info("Запуск сервера...")

    try:
        import glob

        for pending in glob.glob(os.path.join(config.NOTEBOOKS_DIR, "*.pending_delete_*")):
            logger.info(f"Удаляю отложенную папку: {pending}")
            from routers.shared import robust_rmtree

            success, err = robust_rmtree(pending)
            if not success:
                logger.warning(f"Не удалось удалить {pending}: {err}")
    except Exception as e:
        logger.warning(f"Ошибка cleanup pending_delete: {e}")

    from routers.notebooks import migrate_old_data

    migrate_old_data()

    threading.Thread(target=preload_all_models, daemon=True).start()

    yield

    if not getattr(app.state, "cleanup_done", False):
        app.state.cleanup_done = True
        logger.info("Остановка системы...")
        from src.gguf.server import kill_stray_servers, unload_all_models

        unload_all_models()
        kill_stray_servers()


def preload_all_models():
    try:
        from src.rag_pipeline import preload_all_models as _preload

        _preload()
    except Exception as e:
        logger.warning(f"Предзагрузка моделей не удалась: {e}")


def _shutdown_models():
    try:
        from src.gguf.server import kill_stray_servers, unload_all_models

        unload_all_models()
        kill_stray_servers()
        logger.info("Модели выгружены.")
    except Exception as e:
        logger.error(f"Ошибка при выгрузке моделей: {e}")


# Windows Console Control Handler — перехватывает CTRL_CLOSE/CTRL_BREAK для graceful shutdown.
if os.name == "nt":
    try:
        import ctypes
        from ctypes import wintypes

        _console_handler = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.DWORD)

        def _handler(dwCtrlType: int) -> bool:
            if dwCtrlType in (0, 1, 2, 5):
                logger.info("Получен сигнал закрытия консоли Windows, выгрузка моделей...")
                _shutdown_models()
            return False

        ctypes.windll.kernel32.SetConsoleCtrlHandler(_console_handler(_handler), True)
        logger.debug("Windows Console Control Handler установлен.")
    except Exception as e:
        logger.debug(f"SetConsoleCtrlHandler не удался (не критично): {e}")

app = FastAPI(title="DocuMind", lifespan=lifespan)


# Middleware: проверяет Content-Length для /api/upload до передачи в роутер — отклоняет файлы больше лимита.
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
                        f"Файл слишком большой: {mb:.1f} МБ. Лимит: {config.UPLOAD_MAX_SIZE_MB} МБ."
                    )
                },
            )
    return await call_next(request)


app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok", "app": "DocuMind"}


os.makedirs(os.path.join(config.BASE_DIR, "static"), exist_ok=True)
app.mount("/static", StaticFiles(directory=os.path.join(config.BASE_DIR, "static")), name="static")
app.mount("/files", StaticFiles(directory=config.NOTEBOOKS_DIR), name="notebooks")

from routers.bookmarks import router as bookmarks_router
from routers.chat import router as chat_router
from routers.files import router as files_router
from routers.gguf import router as gguf_router
from routers.notebooks import router as notebooks_router
from routers.settings import router as settings_router

app.include_router(notebooks_router)
app.include_router(files_router)
app.include_router(chat_router)
app.include_router(gguf_router)
app.include_router(bookmarks_router)
app.include_router(settings_router)

if __name__ == "__main__":
    import uvicorn

    logger.info(f"Сервер запускается на {config.HOST}:{config.PORT}")
    uvicorn.run(
        "main:app",
        host=config.HOST,
        port=config.PORT,
        reload=config.RELOAD,
        access_log=False,
        log_level="warning",
    )
