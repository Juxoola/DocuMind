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
