import os

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
QUANTIZATION = os.getenv("QUANTIZATION", "int8")

# ── Настройки локальных GGUF моделей ──

# Базовая директория для поиска GGUF моделей (можно указать несколько через ;)
# Например: "F:/llm/mradermacher;D:/models"
GGUF_SEARCH_DIRS = os.getenv("GGUF_SEARCH_DIRS", "F:/llm")

# Порт для llama-cpp-python сервера (запускается локально)
GGUF_SERVER_PORT = int(os.getenv("GGUF_SERVER_PORT", 8081))
GGUF_SERVER_HOST = os.getenv("GGUF_SERVER_HOST", "127.0.0.1")

# Количество потоков для инференса (0 = авто)
GGUF_THREADS = int(os.getenv("GGUF_THREADS", 0))

# Контекст (токенов)
GGUF_CTX_SIZE = int(os.getenv("GGUF_CTX_SIZE", 4096))

# GPU слоёв (-1 = все на GPU, 0 = только CPU)
GGUF_GPU_LAYERS = int(os.getenv("GGUF_GPU_LAYERS", -1))

def resolve_model_path(path_or_filename: str) -> str:
    """
    Если путь абсолютный и существует — возвращает его.
    Иначе ищет файл во всех директориях из GGUF_SEARCH_DIRS.
    """
    if not path_or_filename:
        return ""
    
    # Если это уже существующий абсолютный путь
    if os.path.isabs(path_or_filename) and os.path.exists(path_or_filename):
        return os.path.normpath(path_or_filename)
    
    # Иначе ищем в GGUF_SEARCH_DIRS
    search_dirs = [d.strip() for d in GGUF_SEARCH_DIRS.split(";") if d.strip()]
    filename = os.path.basename(path_or_filename)
    
    for base_dir in search_dirs:
        # Рекурсивный поиск файла
        for dirpath, dirnames, filenames in os.walk(base_dir):
            if filename in filenames:
                full_path = os.path.join(dirpath, filename)
                print(f"[CONFIG] Модель найдена: {full_path}")
                return os.path.normpath(full_path)
    
    # Если не нашли — возвращаем как есть (может упасть позже, но это честно)
    return path_or_filename
