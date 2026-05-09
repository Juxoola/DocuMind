import os
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NOTEBOOKS_DIR = os.path.join(BASE_DIR, "notebooks")

os.makedirs(NOTEBOOKS_DIR, exist_ok=True)

def get_notebook_paths(notebook_id: str):
    nb_path = os.path.join(NOTEBOOKS_DIR, notebook_id)
    return {
        "base": nb_path,
        "data": os.path.join(nb_path, "data"),
        "chroma_db": os.path.join(nb_path, "chroma_db"),
        "images": os.path.join(nb_path, "images")
    }

# Настройки сервера
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))

# Настройки LM Studio
LM_STUDIO_URL = os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1")

# Настройки эмбеддингов
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "Qwen/Qwen3-Embedding-0.6B")
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "Qwen/Qwen3-Reranker-0.6B")
# Квантование: 'fp16', 'int8' или '4bit'
QUANTIZATION = os.getenv("QUANTIZATION", "4bit")

# Настройки воронки RAG
RAG_TOP_K_PER_FILE = int(os.getenv("RAG_TOP_K_PER_FILE", 5))
RAG_RERANK_POOL = int(os.getenv("RAG_RERANK_POOL", 30))
RAG_FINAL_TOP_N = int(os.getenv("RAG_FINAL_TOP_N", 15))
USE_RERANKER = os.getenv("USE_RERANKER", "true").lower() == "true"

# ── Настройки локальных GGUF моделей ──

# Базовая директория для поиска GGUF моделей (можно указать несколько через ;)
# Например: "F:/llm/mradermacher;D:/models"
GGUF_SEARCH_DIRS = os.getenv("GGUF_SEARCH_DIRS", "F:/llm")

# Дефолтный порт для llama-server.exe (если не выбран свободный автоматически)
GGUF_SERVER_PORT = int(os.getenv("GGUF_SERVER_PORT", 8081))
GGUF_SERVER_HOST = os.getenv("GGUF_SERVER_HOST", "127.0.0.1")

# Количество потоков для инференса (0 = авто)
GGUF_THREADS = int(os.getenv("GGUF_THREADS", 0))

# Контекст (токенов)
GGUF_CTX_SIZE = int(os.getenv("GGUF_CTX_SIZE", 32768))

# GPU слоёв (-1 = все на GPU, 0 = только CPU)
GGUF_GPU_LAYERS = int(os.getenv("GGUF_GPU_LAYERS", -1))

# Настройки генерации Vision (строгие)
VISION_TEMPERATURE = float(os.getenv("VISION_TEMPERATURE", 0.1))
VISION_REPEAT_PENALTY = float(os.getenv("VISION_REPEAT_PENALTY", 1.3))
VISION_TOP_P = float(os.getenv("VISION_TOP_P", 0.9))
VISION_MIN_P = float(os.getenv("VISION_MIN_P", 0.05))
VISION_CONCURRENCY = int(os.getenv("VISION_CONCURRENCY", 4)) # Количество параллельных потоков анализа

# Настройки генерации Chat (творческие)
CHAT_TEMPERATURE = float(os.getenv("CHAT_TEMPERATURE", 0.7))
CHAT_REPEAT_PENALTY = float(os.getenv("CHAT_REPEAT_PENALTY", 1.1))
CHAT_TOP_P = float(os.getenv("CHAT_TOP_P", 0.95))
CHAT_MIN_P = float(os.getenv("CHAT_MIN_P", 0.05))

# ── Сохранение настроек ──

LAST_MODELS_FILE = os.path.join(BASE_DIR, "last_models.json")
RAG_CONFIG_FILE = os.path.join(BASE_DIR, "rag_config.json")

def save_last_model(gguf_path, mmproj_path):
    try:
        with open(LAST_MODELS_FILE, "w", encoding="utf-8") as f:
            json.dump({"gguf": gguf_path, "mmproj": mmproj_path}, f)
    except: pass

def load_last_model():
    try:
        if os.path.exists(LAST_MODELS_FILE):
            with open(LAST_MODELS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except: pass
    return {"gguf": None, "mmproj": None}

def save_rag_config():
    config_data = {
        "embedding_model": EMBEDDING_MODEL_NAME,
        "reranker_model": RERANKER_MODEL_NAME,
        "quantization": QUANTIZATION,
        "top_k_per_file": RAG_TOP_K_PER_FILE,
        "rerank_pool": RAG_RERANK_POOL,
        "final_top_n": RAG_FINAL_TOP_N,
        "use_reranker": USE_RERANKER,
        "gguf_search_dirs": GGUF_SEARCH_DIRS
    }
    try:
        with open(RAG_CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
    except: pass

def load_rag_config():
    global EMBEDDING_MODEL_NAME, RERANKER_MODEL_NAME, QUANTIZATION, RAG_TOP_K_PER_FILE, RAG_RERANK_POOL, RAG_FINAL_TOP_N, USE_RERANKER, GGUF_SEARCH_DIRS
    try:
        if os.path.exists(RAG_CONFIG_FILE):
            with open(RAG_CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                EMBEDDING_MODEL_NAME = data.get("embedding_model", EMBEDDING_MODEL_NAME)
                RERANKER_MODEL_NAME = data.get("reranker_model", RERANKER_MODEL_NAME)
                QUANTIZATION = data.get("quantization", QUANTIZATION)
                RAG_TOP_K_PER_FILE = data.get("top_k_per_file", RAG_TOP_K_PER_FILE)
                RAG_RERANK_POOL = data.get("rerank_pool", RAG_RERANK_POOL)
                RAG_FINAL_TOP_N = data.get("final_top_n", RAG_FINAL_TOP_N)
                USE_RERANKER = data.get("use_reranker", USE_RERANKER)
                GGUF_SEARCH_DIRS = data.get("gguf_search_dirs", GGUF_SEARCH_DIRS)
    except: pass

# Загружаем настройки при старте
load_rag_config()

def resolve_model_path(path_or_filename: str) -> str:
    """
    Если путь абсолютный и существует — возвращает его.
    Иначе ищет файл во всех директориях из GGUF_SEARCH_DIRS.
    """
    if not path_or_filename:
        return ""
    
    # Если это уже существующий абсолютный путь
    if os.path.isabs(path_or_filename) and os.path.exists(path_or_filename):
        return os.path.normpath(path_or_filename).lower()
    
    # Иначе ищем в GGUF_SEARCH_DIRS
    search_dirs = [d.strip() for d in GGUF_SEARCH_DIRS.split(";") if d.strip()]
    filename = os.path.basename(path_or_filename)
    
    for base_dir in search_dirs:
        # Рекурсивный поиск файла
        for dirpath, dirnames, filenames in os.walk(base_dir):
            if filename in filenames:
                full_path = os.path.join(dirpath, filename)
                print(f"[CONFIG] Модель найдена: {full_path}")
                return os.path.normpath(full_path).lower()
    
    # Если не нашли — возвращаем как есть (может упасть позже, но это честно)
    return path_or_filename
