"""DocuMind — основной модуль."""

import asyncio
import contextvars
import logging
import os
import signal
import sys
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import config

# ── Contextvars для request tracing ──
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="")

# ── Структурированное JSON-логирование ──
_LOG_DIR = os.path.join(config.BASE_DIR, "logs")
os.makedirs(_LOG_DIR, exist_ok=True)


class _RequestIDFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get("")
        return True


try:
    from pythonjsonlogger.json import JsonFormatter as _JsonFormatter

    _json_formatter = _JsonFormatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(message)s %(request_id)s",
        rename_fields={"asctime": "time", "levelname": "level", "name": "logger"},
        static_fields={"request_id": ""},
        datefmt="%Y-%m-%dT%H:%M:%S",
        ensure_ascii=False,
    )
except Exception:
    # fallback — простой JSON без сторонней библиотеки
    class _FallbackFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            import json as _json

            return _json.dumps(
                {
                    "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
                    "level": record.levelname,
                    "logger": record.name,
                    "message": record.getMessage(),
                    "request_id": getattr(record, "request_id", ""),
                },
                ensure_ascii=False,
            )

    _json_formatter = _FallbackFormatter()

_root_handler_file = logging.FileHandler(os.path.join(_LOG_DIR, "server.log"), encoding="utf-8")
_root_handler_file.setFormatter(_json_formatter)
_root_handler_file.addFilter(_RequestIDFilter())

# UTF-8 для Windows console
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
_root_handler_stream = logging.StreamHandler(sys.stdout)
_root_handler_stream.setFormatter(_json_formatter)
_root_handler_stream.addFilter(_RequestIDFilter())

logging.basicConfig(
    level=logging.INFO,
    handlers=[_root_handler_file, _root_handler_stream],
)
del _root_handler_file, _root_handler_stream
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)
logging.getLogger("llama_index").setLevel(logging.WARNING)
logging.getLogger("bm25s.utils.benchmark").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
logging.getLogger("speechbrain").setLevel(logging.ERROR)
logging.getLogger("pyannote").setLevel(logging.WARNING)
logging.getLogger("whisperx").setLevel(logging.WARNING)

import warnings
from datetime import UTC

warnings.filterwarnings("ignore", message="(?s).*torchcodec is not installed")
warnings.filterwarnings("ignore", message="(?s).*Could not load libtorchcodec")
warnings.filterwarnings("ignore", message="(?s).*list_audio_backends")
warnings.filterwarnings("ignore", message="(?s).*Lightning automatically upgraded")
warnings.filterwarnings("ignore", message="TensorFloat-32")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="speechbrain")

logger = logging.getLogger(__name__)


# ── Lifespan: инициализация и завершение приложения ──
@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info("Запуск сервера...")

    try:
        import glob

        for pending in glob.glob(os.path.join(config.NOTEBOOKS_DIR, "*.pending_delete_*")):
            logger.info(f"Удаляю отложенную папку: {pending}")
            from routers.shared import robust_rmtree

            success, err = await robust_rmtree(pending)
            if not success:
                logger.warning(f"Не удалось удалить {pending}: {err}")
    except Exception as e:
        logger.warning(f"Ошибка cleanup pending_delete: {e}")

    from routers.notebooks import migrate_old_data

    await migrate_old_data()

    async def _preload_task():
        try:
            from src.rag.models import preload_all_models as _preload

            await _preload()
        except Exception as e:
            logger.warning(f"Предзагрузка моделей не удалась: {e}")

    _bg_tasks: set[asyncio.Task] = set()
    _bg_tasks.add(asyncio.create_task(_preload_task()))

    yield

    if not getattr(app.state, "cleanup_done", False):
        app.state.cleanup_done = True
        logger.info("Остановка системы...")
        from src.gguf.server import kill_stray_servers, unload_all_models

        await unload_all_models()
        await kill_stray_servers()


# ── Выгрузка моделей при завершении ──
def _shutdown_models():
    try:
        from src.gguf.server import kill_stray_servers, unload_all_models

        async def _do_shutdown():
            await unload_all_models()
            await kill_stray_servers()

        asyncio.run(_do_shutdown())
        logger.info("Модели выгружены.")
    except Exception as e:
        logger.error(f"Ошибка при выгрузке моделей: {e}")


# ── Graceful shutdown: обработка SIGTERM/SIGINT ──
def _graceful_signal_handler(signum, frame):
    logger.info(f"Получен сигнал {signum}, запуск graceful shutdown...")
    _shutdown_models()
    sys.exit(0)


for _sig in (signal.SIGINT, signal.SIGTERM):
    try:
        signal.signal(_sig, _graceful_signal_handler)
    except (OSError, ValueError):
        # Windows не поддерживает SIGTERM; fallback ниже
        pass

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


# Rate limiting: простой sliding-window по IP. Лимиты: upload=10/min, chat=30/min, other=60/min.
_rate_store: dict[str, list[float]] = {}
_RATE_LIMITS = {
    "/api/upload": (10, 60),
    "/api/chat": (30, 60),
}
_DEFAULT_RATE = (500, 60)


def _get_client_ip(request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


@app.middleware("http")
async def rate_limit_middleware(request, call_next):
    if not request.url.path.startswith("/api/"):
        return await call_next(request)

    client_ip = _get_client_ip(request)
    now = time.time()

    for pattern, (limit, window) in _RATE_LIMITS.items():
        if request.url.path.startswith(pattern):
            break
    else:
        limit, window = _DEFAULT_RATE

    key = f"{client_ip}:{request.url.path.split('/')[2] if len(request.url.path.split('/')) > 2 else 'root'}"
    if len(_rate_store) > 10000:
        stale = {k for k, v in _rate_store.items() if not v or now - v[-1] >= 120}
        for k in stale:
            _rate_store.pop(k, None)
    timestamps = _rate_store.get(key, [])
    timestamps = [t for t in timestamps if now - t < window]
    if len(timestamps) >= limit:
        return JSONResponse(
            status_code=429,
            content={"detail": "Слишком много запросов. Попробуйте позже."},
        )
    timestamps.append(now)
    _rate_store[key] = timestamps

    return await call_next(request)


# ── Request tracing middleware ──
@app.middleware("http")
async def request_tracing_middleware(request, call_next):
    req_id = uuid.uuid4().hex
    token = request_id_var.set(req_id)
    try:
        response = await call_next(request)
    finally:
        request_id_var.reset(token)
    response.headers["X-Request-ID"] = req_id
    return response


app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "PATCH"],
    allow_headers=["Content-Type"],
)


@app.get("/health")
async def health():
    return {"status": "ok", "app": "DocuMind"}


# ── Deep health check: проверка всех компонентов ──
@app.get("/health/deep")
async def health_deep():
    from datetime import datetime

    components: dict = {}
    overall = "ok"

    try:
        import chromadb

        _client = chromadb.Client()
        _client.heartbeat()
        components["chromadb"] = {"status": "ok"}
    except Exception as e:
        components["chromadb"] = {"status": "error", "detail": str(e)}
        overall = "degraded"

    try:
        from src.gguf.server import get_active_llm_url

        url = await asyncio.wait_for(get_active_llm_url(), timeout=5)
        if url:
            components["llm"] = {"status": "ok"}
        else:
            components["llm"] = {"status": "not_loaded"}
    except Exception as e:
        components["llm"] = {"status": "unavailable", "detail": str(e)}
        overall = "degraded"

    try:
        import torch

        if torch.cuda.is_available():
            _dev = torch.cuda.get_device_properties(0)
            _total = _dev.total_mem / (1024**3)
            _free = torch.cuda.mem_get_info(0)[0] / (1024**3)
            components["gpu"] = {
                "status": "ok",
                "device": _dev.name,
                "total_gb": round(_total, 2),
                "free_gb": round(_free, 2),
            }
        else:
            components["gpu"] = {"status": "unavailable", "detail": "CUDA not available"}
    except Exception as e:
        components["gpu"] = {"status": "unavailable", "detail": str(e)}

    if any(c.get("status") == "error" for c in components.values()):
        overall = "degraded"
    if all(c.get("status") in ("unavailable", "error") for c in components.values()):
        overall = "error"

    return {
        "status": overall,
        "app": "DocuMind",
        "timestamp": datetime.now(UTC).isoformat(),
        "components": components,
    }


os.makedirs(os.path.join(config.BASE_DIR, "static"), exist_ok=True)

from starlette.staticfiles import StaticFiles as _StaticFiles


class _CachedStaticFiles(_StaticFiles):
    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        if response.status_code == 200:
            if path.endswith((".js", ".css", ".wasm")):
                response.headers["Cache-Control"] = "public, max-age=604800, immutable"
            elif path.endswith((".woff2", ".woff", ".ttf")):
                response.headers["Cache-Control"] = "public, max-age=2592000, immutable"
        return response


app.mount(
    "/static", _CachedStaticFiles(directory=os.path.join(config.BASE_DIR, "static")), name="static"
)


app.mount("/files", _StaticFiles(directory=config.NOTEBOOKS_DIR), name="notebooks")


# ── Подключение роутеров ──
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
