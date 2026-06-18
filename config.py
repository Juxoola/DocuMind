import logging
import os
from functools import lru_cache

# Отключаем онлайн-проверки Hugging Face (используем только локальный кэш)
os.environ["HF_HUB_OFFLINE"] = os.getenv("HF_HUB_OFFLINE", "1")

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NOTEBOOKS_DIR = os.path.join(BASE_DIR, "notebooks")

os.makedirs(NOTEBOOKS_DIR, exist_ok=True)


# Пути для хранения данных каждого notebook: data, chroma_db, images.
def get_notebook_paths(notebook_id: str):
    nb_path = os.path.join(NOTEBOOKS_DIR, notebook_id)
    return {
        "base": nb_path,
        "data": os.path.join(nb_path, "data"),
        "chroma_db": os.path.join(nb_path, "chroma_db"),
        "images": os.path.join(nb_path, "images"),
    }


HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))
RELOAD = os.getenv("RELOAD", "false").lower() in ("1", "true", "yes")
CORS_ORIGINS = [
    o.strip()
    for o in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,http://localhost:8000,http://127.0.0.1:8000",
    ).split(",")
    if o.strip()
]

GGUF_SERVER_STARTUP_TIMEOUT = int(os.getenv("GGUF_SERVER_STARTUP_TIMEOUT", 30))
GGUF_HEALTH_CHECK_TIMEOUT = int(os.getenv("GGUF_HEALTH_CHECK_TIMEOUT", 2))
GGUF_SERVER_STOP_TIMEOUT = int(os.getenv("GGUF_SERVER_STOP_TIMEOUT", 5))
LM_STUDIO_HTTP_TIMEOUT = int(os.getenv("LM_STUDIO_HTTP_TIMEOUT", 60))
HTTP_POOL_SIZE_MAIN = int(os.getenv("HTTP_POOL_SIZE_MAIN", 10))
HTTP_POOL_SIZE_INGEST = int(os.getenv("HTTP_POOL_SIZE_INGEST", 10))
HTTP_POOL_SIZE_RERANK = int(os.getenv("HTTP_POOL_SIZE_RERANK", 4))

LM_STUDIO_URL = os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1")

LLM_DEFAULT_API_KEY = os.getenv("LLM_DEFAULT_API_KEY", "lm-studio")
LLM_DEFAULT_MODEL = os.getenv("LLM_DEFAULT_MODEL", "gpt-4o")
EMBEDDING_DEFAULT_API_KEY = os.getenv("EMBEDDING_DEFAULT_API_KEY", "lm-studio")
EMBEDDING_DEFAULT_MODEL = os.getenv("EMBEDDING_DEFAULT_MODEL", "text-embedding-ada-002")

UPLOAD_MAX_SIZE_MB = int(os.getenv("UPLOAD_MAX_SIZE_MB", "500"))
UPLOAD_MAX_SIZE_BYTES = UPLOAD_MAX_SIZE_MB * 1024 * 1024

ALLOWED_UPLOAD_EXTENSIONS = frozenset(
    {
        ".pdf",
        ".pptx",
        ".docx",
        ".mp4",
        ".avi",
        ".mkv",
        ".mov",
        ".mp3",
        ".wav",
        ".m4a",
        ".txt",
        ".md",
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
    }
)

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "Qwen3-Embedding-0.6B-Q8_0.gguf")
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "qwen3-reranker-0.6b-q8_0.gguf")
EMBEDDING_N_PARALLEL = int(os.getenv("EMBEDDING_N_PARALLEL", "2"))

# Параметры RAG-пайплайна: сколько чанков искать на файл (top_k),
# пул для реранкера, итоговое число, пороговые фильтры.
RAG_TOP_K_PER_FILE = int(os.getenv("RAG_TOP_K_PER_FILE", 5))
RAG_RERANK_POOL = int(os.getenv("RAG_RERANK_POOL", 30))
RAG_FINAL_TOP_N = int(os.getenv("RAG_FINAL_TOP_N", 15))
USE_RERANKER = os.getenv("USE_RERANKER", "true").lower() == "true"
RAG_QUERY_EXPANSION = os.getenv("RAG_QUERY_EXPANSION", "true").lower() == "true"
RERANK_SCORE_THRESHOLD = float(os.getenv("RERANK_SCORE_THRESHOLD", "0.1"))
MIN_FINAL_CHUNKS = int(os.getenv("MIN_FINAL_CHUNKS", "5"))
RAG_RRF_K = int(os.getenv("RAG_RRF_K", 60))
RAG_TOP_K_RATIO = float(os.getenv("RAG_TOP_K_RATIO", "0.1"))

GGUF_SEARCH_DIRS = os.getenv("GGUF_SEARCH_DIRS", "F:/llm;" + os.path.join(BASE_DIR, "models"))
GGUF_SERVER_PORT = int(os.getenv("GGUF_SERVER_PORT", 8081))
GGUF_SERVER_HOST = os.getenv("GGUF_SERVER_HOST", "127.0.0.1")
GGUF_THREADS = int(os.getenv("GGUF_THREADS", 0))
GGUF_CTX_SIZE = int(os.getenv("GGUF_CTX_SIZE", 16384))
GGUF_CTX_EMBED_CHARS = int(os.getenv("GGUF_CTX_EMBED_CHARS", 4096))
GGUF_GPU_LAYERS = int(os.getenv("GGUF_GPU_LAYERS", -1))

VISION_TEMPERATURE = float(os.getenv("VISION_TEMPERATURE", 0.1))
VISION_REPEAT_PENALTY = float(os.getenv("VISION_REPEAT_PENALTY", 1.3))
VISION_TOP_P = float(os.getenv("VISION_TOP_P", 0.9))
VISION_MIN_P = float(os.getenv("VISION_MIN_P", 0.05))
VISION_CONCURRENCY = int(os.getenv("VISION_CONCURRENCY", 4))

CHAT_TEMPERATURE = float(os.getenv("CHAT_TEMPERATURE", 0.7))
CHAT_REPEAT_PENALTY = float(os.getenv("CHAT_REPEAT_PENALTY", 1.1))
CHAT_TOP_P = float(os.getenv("CHAT_TOP_P", 0.95))
CHAT_MIN_P = float(os.getenv("CHAT_MIN_P", 0.05))

RAG_CONFIG_FILE = os.path.join(BASE_DIR, "rag_config.json")

from src.config_manager import load_rag_config as _cm_load
from src.config_manager import save_rag_config as _cm_save


def _apply_rag_config(data: dict) -> None:
    global \
        EMBEDDING_MODEL_NAME, \
        RERANKER_MODEL_NAME, \
        RAG_TOP_K_PER_FILE, \
        RAG_RERANK_POOL, \
        RAG_FINAL_TOP_N, \
        USE_RERANKER, \
        GGUF_SEARCH_DIRS, \
        RAG_QUERY_EXPANSION, \
        RERANK_SCORE_THRESHOLD
    EMBEDDING_MODEL_NAME = data.get("embedding_model", EMBEDDING_MODEL_NAME)
    RERANKER_MODEL_NAME = data.get("reranker_model", RERANKER_MODEL_NAME)
    RAG_TOP_K_PER_FILE = data.get("top_k_per_file", RAG_TOP_K_PER_FILE)
    RAG_RERANK_POOL = data.get("rerank_pool", RAG_RERANK_POOL)
    RAG_FINAL_TOP_N = data.get("final_top_n", RAG_FINAL_TOP_N)
    USE_RERANKER = data.get("use_reranker", USE_RERANKER)
    GGUF_SEARCH_DIRS = data.get("gguf_search_dirs", GGUF_SEARCH_DIRS)
    RAG_QUERY_EXPANSION = data.get("query_expansion", RAG_QUERY_EXPANSION)
    RERANK_SCORE_THRESHOLD = float(data.get("rerank_score_threshold", RERANK_SCORE_THRESHOLD))


def _collect_rag_config() -> dict:
    return {
        "embedding_model": EMBEDDING_MODEL_NAME,
        "reranker_model": RERANKER_MODEL_NAME,
        "top_k_per_file": RAG_TOP_K_PER_FILE,
        "rerank_pool": RAG_RERANK_POOL,
        "final_top_n": RAG_FINAL_TOP_N,
        "use_reranker": USE_RERANKER,
        "gguf_search_dirs": GGUF_SEARCH_DIRS,
        "query_expansion": RAG_QUERY_EXPANSION,
        "rerank_score_threshold": RERANK_SCORE_THRESHOLD,
    }


def load_rag_config():
    defaults = _collect_rag_config()
    data = _cm_load(RAG_CONFIG_FILE, defaults)
    _apply_rag_config(data)


def save_rag_config():
    data = _collect_rag_config()
    _cm_save(RAG_CONFIG_FILE, data)


load_rag_config()


# Поиск GGUF-файла: сначала абсолютный путь, потом по имени через
# gguf_manager, затем рекурсивный обход GGUF_SEARCH_DIRS.
# Результат кешируется (lru_cache на 64 записи).
@lru_cache(maxsize=64)
def resolve_model_path(path_or_filename: str) -> str:
    if not path_or_filename:
        return ""

    if os.path.isabs(path_or_filename) and os.path.exists(path_or_filename):
        return os.path.normpath(path_or_filename).lower()

    try:
        from src.gguf_manager import find_gguf_by_name

        hit = find_gguf_by_name(path_or_filename)
        if hit:
            logger.info(f"[CONFIG] Модель найдена: {hit}")
            return os.path.normpath(hit).lower()
    except Exception as e:
        logger.debug(f"[CONFIG] gguf_manager.find_gguf_by_name failed: {e}")

    search_dirs = [d.strip() for d in GGUF_SEARCH_DIRS.split(";") if d.strip()]
    filename = os.path.basename(path_or_filename)

    for base_dir in search_dirs:
        for dirpath, _dirnames, filenames in os.walk(base_dir):
            if filename in filenames:
                full_path = os.path.join(dirpath, filename)
                logger.info(f"[CONFIG] Модель найдена (fallback walk): {full_path}")
                return os.path.normpath(full_path).lower()

    return path_or_filename


def validate_gguf_path(name: str) -> bool:
    """Проверяет, что имя или путь модели — .gguf файл."""
    return bool(name.lower().endswith(".gguf") or (os.path.isabs(name) and os.path.exists(name)))
