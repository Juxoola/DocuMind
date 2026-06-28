"""Конфигурация приложения: переменные окружения, пути и параметры RAG-пайплайна."""

import logging
import os
import threading
import time
from dataclasses import dataclass

# Отключаем онлайн-проверки Hugging Face (используем только локальный кэш)
os.environ["HF_HUB_OFFLINE"] = os.getenv("HF_HUB_OFFLINE", "1")

logger = logging.getLogger(__name__)

import orjson

_config_lock = threading.RLock()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NOTEBOOKS_DIR = os.path.join(BASE_DIR, "notebooks")

os.makedirs(NOTEBOOKS_DIR, exist_ok=True)


# Пути для хранения данных каждого notebook: data, chroma_db, images.
_notebook_paths_cache: dict[str, dict] = {}


def get_notebook_paths(notebook_id: str):
    cached = _notebook_paths_cache.get(notebook_id)
    if cached is not None:
        return cached
    nb_path = os.path.join(NOTEBOOKS_DIR, notebook_id)
    paths = {
        "base": nb_path,
        "data": os.path.join(nb_path, "data"),
        "chroma_db": os.path.join(nb_path, "chroma_db"),
        "images": os.path.join(nb_path, "images"),
    }
    _notebook_paths_cache[notebook_id] = paths
    return paths


HOST = os.getenv("HOST", "127.0.0.1")
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

LM_STUDIO_HTTP_TIMEOUT = int(os.getenv("LM_STUDIO_HTTP_TIMEOUT", 60))
HTTP_POOL_SIZE_MAIN = int(os.getenv("HTTP_POOL_SIZE_MAIN", 10))
HTTP_POOL_SIZE_INGEST = int(os.getenv("HTTP_POOL_SIZE_INGEST", 10))

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


@dataclass(frozen=True)
class RAGConfig:
    """Неизменяемый снимок RAG-параметров. Атомарно заменяется при обновлении."""

    embedding_model: str = os.getenv("EMBEDDING_MODEL_NAME", "Qwen3-Embedding-0.6B-Q8_0.gguf")
    reranker_model: str = os.getenv("RERANKER_MODEL_NAME", "qwen3-reranker-0.6b-q8_0.gguf")
    embedding_n_parallel: int = int(os.getenv("EMBEDDING_N_PARALLEL", "2"))
    top_k_per_file: int = int(os.getenv("RAG_TOP_K_PER_FILE", "5"))
    rerank_pool: int = int(os.getenv("RAG_RERANK_POOL", "30"))
    final_top_n: int = int(os.getenv("RAG_FINAL_TOP_N", "15"))
    use_reranker: bool = os.getenv("USE_RERANKER", "true").lower() == "true"
    query_expansion: bool = os.getenv("RAG_QUERY_EXPANSION", "true").lower() == "true"
    rerank_score_threshold: float = float(os.getenv("RERANK_SCORE_THRESHOLD", "0.1"))
    min_final_chunks: int = int(os.getenv("MIN_FINAL_CHUNKS", "5"))
    rrf_k: int = int(os.getenv("RAG_RRF_K", "60"))
    top_k_ratio: float = float(os.getenv("RAG_TOP_K_RATIO", "0.1"))
    surya_mode: str = os.getenv("SURYA_MODE", "layout_only")  # disabled | layout_only | full
    gguf_search_dirs: str = os.getenv(
        "GGUF_SEARCH_DIRS",
        "F:/llm;" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"),
    )


# Атомарно заменяемый снимок — читатели всегда видят консистентное состояние
rag = RAGConfig()


def update_rag_config(data: dict) -> None:
    global rag
    rag = RAGConfig(
        embedding_model=data.get("embedding_model", rag.embedding_model),
        reranker_model=data.get("reranker_model", rag.reranker_model),
        embedding_n_parallel=int(data.get("embedding_n_parallel", rag.embedding_n_parallel)),
        top_k_per_file=data.get("top_k_per_file", rag.top_k_per_file),
        rerank_pool=data.get("rerank_pool", rag.rerank_pool),
        final_top_n=data.get("final_top_n", rag.final_top_n),
        use_reranker=data.get("use_reranker", rag.use_reranker),
        gguf_search_dirs=data.get("gguf_search_dirs", rag.gguf_search_dirs),
        query_expansion=data.get("query_expansion", rag.query_expansion),
        rerank_score_threshold=float(
            data.get("rerank_score_threshold", rag.rerank_score_threshold)
        ),
        min_final_chunks=int(data.get("min_final_chunks", rag.min_final_chunks)),
        rrf_k=int(data.get("rrf_k", rag.rrf_k)),
        top_k_ratio=float(data.get("top_k_ratio", rag.top_k_ratio)),
        surya_mode=data.get("surya_mode", rag.surya_mode),
    )


def collect_rag_config() -> dict:
    from dataclasses import asdict

    return asdict(rag)


EMBEDDING_MODEL_NAME = rag.embedding_model
RERANKER_MODEL_NAME = rag.reranker_model
EMBEDDING_N_PARALLEL = rag.embedding_n_parallel
RAG_TOP_K_PER_FILE = rag.top_k_per_file
RAG_RERANK_POOL = rag.rerank_pool
RAG_FINAL_TOP_N = rag.final_top_n
USE_RERANKER = rag.use_reranker
RAG_QUERY_EXPANSION = rag.query_expansion
RERANK_SCORE_THRESHOLD = rag.rerank_score_threshold
MIN_FINAL_CHUNKS = rag.min_final_chunks
RAG_RRF_K = rag.rrf_k
RAG_TOP_K_RATIO = rag.top_k_ratio
GGUF_THREADS = int(os.getenv("GGUF_THREADS", "0"))
GGUF_GPU_LAYERS = int(os.getenv("GGUF_GPU_LAYERS", "-1"))
VISION_TEMPERATURE = float(os.getenv("VISION_TEMPERATURE", "0.1"))
VISION_REPEAT_PENALTY = float(os.getenv("VISION_REPEAT_PENALTY", "1.3"))
VISION_TOP_P = float(os.getenv("VISION_TOP_P", "0.9"))
VISION_MIN_P = float(os.getenv("VISION_MIN_P", "0.05"))
CHAT_TEMPERATURE = float(os.getenv("CHAT_TEMPERATURE", "0.7"))
GGUF_SEARCH_DIRS = rag.gguf_search_dirs
GGUF_CTX_SIZE = int(os.getenv("GGUF_CTX_SIZE", "16384"))
GGUF_CTX_EMBED_CHARS = int(os.getenv("GGUF_CTX_EMBED_CHARS", "4096"))
VISION_CONCURRENCY = int(os.getenv("VISION_CONCURRENCY", "4"))
SURYA_MODE = rag.surya_mode


def _apply_rag_config(data: dict) -> None:
    update_rag_config(data)
    global EMBEDDING_MODEL_NAME, RERANKER_MODEL_NAME, EMBEDDING_N_PARALLEL
    global RAG_TOP_K_PER_FILE, RAG_RERANK_POOL, RAG_FINAL_TOP_N
    global USE_RERANKER, RAG_QUERY_EXPANSION, RERANK_SCORE_THRESHOLD, GGUF_SEARCH_DIRS
    global SURYA_MODE
    EMBEDDING_MODEL_NAME = rag.embedding_model
    RERANKER_MODEL_NAME = rag.reranker_model
    EMBEDDING_N_PARALLEL = rag.embedding_n_parallel
    RAG_TOP_K_PER_FILE = rag.top_k_per_file
    RAG_RERANK_POOL = rag.rerank_pool
    RAG_FINAL_TOP_N = rag.final_top_n
    USE_RERANKER = rag.use_reranker
    RAG_QUERY_EXPANSION = rag.query_expansion
    RERANK_SCORE_THRESHOLD = rag.rerank_score_threshold
    GGUF_SEARCH_DIRS = rag.gguf_search_dirs
    SURYA_MODE = rag.surya_mode


def _collect_rag_config() -> dict:
    return collect_rag_config()


RAG_CONFIG_FILE = os.path.join(BASE_DIR, "rag_config.json")


# Sync загрузка при старте (до async event loop)
def _load_config_sync():
    try:
        if os.path.exists(RAG_CONFIG_FILE):
            with open(RAG_CONFIG_FILE, "rb") as f:
                data = orjson.loads(f.read())
            _apply_rag_config(data)
    except Exception as e:
        logger.warning(f"Не удалось загрузить RAG config: {e}")


_load_config_sync()


# TTL-кэш для resolve_model_path: решает проблему устаревших путей при переименовании моделей.
_resolve_cache: dict[str, tuple[str, float]] = {}
_resolve_cache_lock = threading.Lock()
_RESOLVE_MODEL_TTL = 300.0


def invalidate_model_cache(path_or_filename: str = None) -> None:
    with _resolve_cache_lock:
        if path_or_filename:
            _resolve_cache.pop(path_or_filename, None)
        else:
            _resolve_cache.clear()


def resolve_model_path(path_or_filename: str) -> str:
    if not path_or_filename:
        return ""

    now = time.time()
    with _resolve_cache_lock:
        cached = _resolve_cache.get(path_or_filename)
        if cached and (now - cached[1]) < _RESOLVE_MODEL_TTL:
            return cached[0]

    result = _resolve_model_path_uncached(path_or_filename)
    with _resolve_cache_lock:
        _resolve_cache[path_or_filename] = (result, now)
    return result


def _resolve_model_path_uncached(path_or_filename: str) -> str:
    if os.path.isabs(path_or_filename) and os.path.exists(path_or_filename):
        return os.path.normpath(path_or_filename).lower()

    try:
        from src.gguf.scanner import find_gguf_by_name_sync

        hit = find_gguf_by_name_sync(path_or_filename)
        if hit:
            logger.info(f"[CONFIG] Модель найдена: {hit}")
            return os.path.normpath(hit).lower()
    except Exception as e:
        logger.debug(f"[CONFIG] gguf_manager.find_gguf_by_name failed: {e}")

    search_dirs = [d.strip() for d in rag.gguf_search_dirs.split(";") if d.strip()]
    filename = os.path.basename(path_or_filename)

    for base_dir in search_dirs:
        for dirpath, _dirnames, filenames in os.walk(base_dir):
            if filename in filenames:
                full_path = os.path.join(dirpath, filename)
                logger.info(f"[CONFIG] Модель найдена (fallback walk): {full_path}")
                return os.path.normpath(full_path).lower()

    return path_or_filename


def validate_gguf_path(name: str) -> bool:
    return bool(name.lower().endswith(".gguf") or (os.path.isabs(name) and os.path.exists(name)))
