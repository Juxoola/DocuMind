import os
import json
import logging
from functools import lru_cache

# Отключаем онлайн-проверки Hugging Face (используем только локальный кэш)
os.environ["HF_HUB_OFFLINE"] = os.getenv("HF_HUB_OFFLINE", "1")

logger = logging.getLogger(__name__)

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
# F-fix #19: reload=True удобен в dev (авто-restart на save), но в prod он
# жрёт +500 MB RAM и по 5-10 сек на каждый restart. По умолчанию выключен.
RELOAD = os.getenv("RELOAD", "false").lower() in ("1", "true", "yes")
# F-fix #20: CORS origins через переменную окружения (для деплоя не на localhost).
# Дефолт — dev-окружение (Vite на 5173, uvicorn на 8000).
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173,http://localhost:8000,http://127.0.0.1:8000").split(",") if o.strip()]

# ── Таймауты и лимиты (F-fix #24: магические числа вынесены в config) ──
# Таймаут ожидания готовности GGUF-сервера при старте (секунд)
GGUF_SERVER_STARTUP_TIMEOUT = int(os.getenv("GGUF_SERVER_STARTUP_TIMEOUT", 30))
# Таймаут одного health-check (секунд) — F-fix #21
GGUF_HEALTH_CHECK_TIMEOUT = int(os.getenv("GGUF_HEALTH_CHECK_TIMEOUT", 2))
# Таймаут terminate() перед kill() для GGUF-сервера (секунд)
GGUF_SERVER_STOP_TIMEOUT = int(os.getenv("GGUF_SERVER_STOP_TIMEOUT", 5))
# HTTP-таймаут по умолчанию для LM Studio запросов (секунд)
LM_STUDIO_HTTP_TIMEOUT = int(os.getenv("LM_STUDIO_HTTP_TIMEOUT", 60))
# Размер HTTP connection pool для сессий requests
HTTP_POOL_SIZE_MAIN = int(os.getenv("HTTP_POOL_SIZE_MAIN", 10))
HTTP_POOL_SIZE_INGEST = int(os.getenv("HTTP_POOL_SIZE_INGEST", 10))
HTTP_POOL_SIZE_RERANK = int(os.getenv("HTTP_POOL_SIZE_RERANK", 4))

# Настройки LM Studio
LM_STUDIO_URL = os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1")

# F-fix #placeholder: дефолтные значения для OpenAI-совместимого клиента, когда
# пользователь не передал свои. Это НЕ реальные секреты — LM Studio и любой
# локальный llama-server игнорируют api_key. Но в OpenAPI-схеме/логах видно,
# что эти строки — placeholder, а не production-ключ.
LLM_DEFAULT_API_KEY = os.getenv("LLM_DEFAULT_API_KEY", "lm-studio")
LLM_DEFAULT_MODEL = os.getenv("LLM_DEFAULT_MODEL", "local-model")
EMBEDDING_DEFAULT_API_KEY = os.getenv("EMBEDDING_DEFAULT_API_KEY", "lm-studio")
# F-fix #embedding-model: имя модели ВАЛИДИРУЕТСЯ llama_index'ом через
# OpenAIEmbeddingModelType enum. Нужно реальное имя из списка:
#   text-embedding-3-small / text-embedding-3-large / text-embedding-ada-002
# llama-server / LM Studio игнорируют это поле, но enum-check падает
# на любую отсебятину ('text-embedding', 'local-model' и т.п.).
# Возвращаем валидный text-embedding-ada-002 — он совместим со всеми
# локальными серверами и не блокирует startup.
EMBEDDING_DEFAULT_MODEL = os.getenv("EMBEDDING_DEFAULT_MODEL", "text-embedding-ada-002")

# F-fix #upload-limit: максимальный размер одного загружаемого файла в МБ.
# Раньше лимита не было — пользователь мог загрузить 100 ГБ файл, и uvicorn
# сначала съел бы RAM под буфер, а потом диск под финальный .pdf. По умолчанию
# 500 МБ — комфортно для часового видео, но отсекает абьюз.
UPLOAD_MAX_SIZE_MB = int(os.getenv("UPLOAD_MAX_SIZE_MB", "500"))
UPLOAD_MAX_SIZE_BYTES = UPLOAD_MAX_SIZE_MB * 1024 * 1024

# F-fix #mime-validation: разрешённые расширения файлов для загрузки.
# Если прилетает файл с другим расширением — отдаём 415 Unsupported Media Type
# сразу на upload, до того как мы сожрём диск и время на ингест.
# Список синхронизирован с тем, что умеет src/ingestion.ingest_file().
ALLOWED_UPLOAD_EXTENSIONS = frozenset({
    ".pdf", ".pptx", ".docx",
    ".mp4", ".avi", ".mkv", ".mov",
    ".mp3", ".wav", ".m4a",
    ".txt", ".md", ".png", ".jpg", ".jpeg", ".webp",
})

# Настройки эмбеддингов
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "Qwen3-Embedding-0.6B-v2.Q8_0.gguf")
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "Qwen3-Reranker-0.6B-v2.Q8_0.gguf")
# Квантование: 'fp16', 'int8' или '4bit'
QUANTIZATION = os.getenv("QUANTIZATION", "4bit")

# Настройки воронки RAG
RAG_TOP_K_PER_FILE = int(os.getenv("RAG_TOP_K_PER_FILE", 5))
RAG_RERANK_POOL = int(os.getenv("RAG_RERANK_POOL", 30))
RAG_FINAL_TOP_N = int(os.getenv("RAG_FINAL_TOP_N", 15))
USE_RERANKER = os.getenv("USE_RERANKER", "true").lower() == "true"
# Query Expansion: LLM генерирует 2 доп. формулировки запроса для лучшего recall.
# Использует активный GGUF LLM-сервер (если запущен) или LM Studio.
# Если ни то, ни другое не доступно — QE безопасно пропускается.
RAG_QUERY_EXPANSION = os.getenv("RAG_QUERY_EXPANSION", "true").lower() == "true"
# Порог score реранкера: чанки ниже этого значения отбрасываются как нерелевантные
RERANK_SCORE_THRESHOLD = float(os.getenv("RERANK_SCORE_THRESHOLD", "0.05"))
# Гарантированный минимум чанков даже если они ниже порога
MIN_FINAL_CHUNKS = int(os.getenv("MIN_FINAL_CHUNKS", "5"))
# Top-K relevance ratio: после F6 adaptive threshold дополнительно отрезаем
# чанки с score < top_score * RAG_TOP_K_RATIO. 0.0 = отключено.
# Пример: top=0.99, ratio=0.1 → отрезаем всё с score <0.099 (типичный мусор).
RAG_TOP_K_RATIO = float(os.getenv("RAG_TOP_K_RATIO", "0.1"))

# ── Настройки локальных GGUF моделей ──

# Базовая директория для поиска GGUF моделей (можно указать несколько через ;)
# Например: "F:/llm/mradermacher;D:/models"
GGUF_SEARCH_DIRS = os.getenv("GGUF_SEARCH_DIRS", "F:/llm;C:/test/models")

# Дефолтный порт для llama-server.exe (если не выбран свободный автоматически)
GGUF_SERVER_PORT = int(os.getenv("GGUF_SERVER_PORT", 8081))
GGUF_SERVER_HOST = os.getenv("GGUF_SERVER_HOST", "127.0.0.1")

# Количество потоков для инференса (0 = авто)
GGUF_THREADS = int(os.getenv("GGUF_THREADS", 0))

# Контекст (токенов) - 16к это разумный баланс для 3080/4080
GGUF_CTX_SIZE = int(os.getenv("GGUF_CTX_SIZE", 16384))

# F3: Максимальный размер чанка (в символах) для embedding без split.
# Соответствует ~1.5x GGUF_CTX эмбеддинг-сервера в символах.
# 4096 chars ≈ 1.2-1.6K токенов для русского; -c 4096 у эмбеддинг-сервера.
# Если описание влезает — оставляем одним чанком (лучше recall на связных описаниях).
GGUF_CTX_EMBED_CHARS = int(os.getenv("GGUF_CTX_EMBED_CHARS", 4096))

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

# ── Системный промпт (модульный, единый для всех LLM) ──
#
# Структура:
#   SYSTEM_PROMPT_BASE        — роль + язык + форматирование (общая шапка)
#   SYSTEM_PROMPT_RULES       — словарь с двумя режимами ответа:
#                                "concise"  — «сначала короткий ответ, потом объяснение»
#                                "detailed" — «сразу развёрнуто, без короткого блока»
#   SYSTEM_PROMPT_CITATION    — общие правила цитирования [N] и формул
#   get_system_prompt(mode)   — собирает финальный промпт под выбранный режим

SYSTEM_PROMPT_BASE = (
    "Ты — умный и точный AI-помощник. ОТВЕЧАЙ СТРОГО НА РУССКОМ ЯЗЫКЕ.\n"
    "Используй Markdown для форматирования.\n"
)

# Режим по умолчанию (используется, если клиент не передал answer_mode)
ANSWER_MODE_DEFAULT = "concise"

SYSTEM_PROMPT_RULES = {
    # ── Режим 1: сначала коротко, потом объяснение (текущее поведение) ──
    "concise": (
        "ПРАВИЛА ОТВЕТА:\n"
        "1. Всегда отвечай на вопрос СРАЗУ, в самом начале ответа.\n"
        "2. Если вопрос содержит варианты ответа (тест) — СНАЧАЛА напиши правильный вариант (букву и текст).\n"
        "3. После прямого ответа приведи подробное объяснение на основе источников.\n"
    ),
    # ── Режим 2: сразу развёрнуто, без отдельного короткого блока ──
    "detailed": (
        "ПРАВИЛА ОТВЕТА:\n"
        "1. Сразу давай развёрнутый, исчерпывающий ответ — НЕ дроби его на «короткий блок + объяснение».\n"
        "2. Если вопрос содержит варианты ответа (тест) — укажи правильный вариант прямо по ходу рассуждения и сразу поясни, почему именно он.\n"
        "3. Отвечай последовательно и связно, опираясь на источники, как в хорошей статье.\n"
    ),
}

SYSTEM_PROMPT_CITATION = (
    "ПРАВИЛО ЦИТИРОВАНИЯ (КРИТИЧЕСКИ ДЛЯ СИСТЕМЫ):\n"
    "- Каждое утверждение ДОЛЖНО завершаться ссылкой в формате [N].\n"
    "- ПРИМЕР: «Поток нельзя запустить дважды [1].»\n"
    "- НИКОГДА не пиши цифру источника без квадратных скобок.\n"
    "- Если одно утверждение основано на нескольких источниках, пиши [1, 2].\n"
    "- Все формулы пиши внутри $...$ или $$...$$.\n"
    "- Если ответа нет в источниках — скажи \"В документах этого нет\".\n"
    "- НИКОГДА не вставляй ссылки [N] внутрь блоков кода (```...```) или в комментарии к коду. Указывай источники только в тексте вокруг кода.\n"
)

# Обратная совместимость: старый код, импортирующий config.SYSTEM_PROMPT,
# продолжает получать валидный промпт (режим по умолчанию — concise).
SYSTEM_PROMPT = (
    SYSTEM_PROMPT_BASE
    + "\n"
    + SYSTEM_PROMPT_RULES[ANSWER_MODE_DEFAULT]
    + "\n"
    + SYSTEM_PROMPT_CITATION
)


def get_system_prompt(mode: str = None) -> str:
    """
    Собирает финальный системный промпт под выбранный режим ответа.
    Неизвестный/пустой mode → ANSWER_MODE_DEFAULT (concise).
    """
    if not mode or mode not in SYSTEM_PROMPT_RULES:
        mode = ANSWER_MODE_DEFAULT
    return (
        SYSTEM_PROMPT_BASE
        + "\n"
        + SYSTEM_PROMPT_RULES[mode]
        + "\n"
        + SYSTEM_PROMPT_CITATION
    )

# ── Сохранение настроек ──

LAST_MODELS_FILE = os.path.join(BASE_DIR, "last_models.json")
RAG_CONFIG_FILE = os.path.join(BASE_DIR, "rag_config.json")

def save_last_model(gguf_path, mmproj_path):
    try:
        with open(LAST_MODELS_FILE, "w", encoding="utf-8") as f:
            json.dump({"gguf": gguf_path, "mmproj": mmproj_path}, f)
    except Exception as e:
        logger.warning(f"Не удалось сохранить last_models: {e}")

def load_last_model():
    try:
        if os.path.exists(LAST_MODELS_FILE):
            with open(LAST_MODELS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Не удалось загрузить last_models: {e}")
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
        "gguf_search_dirs": GGUF_SEARCH_DIRS,
        "query_expansion": RAG_QUERY_EXPANSION,
        "rerank_score_threshold": RERANK_SCORE_THRESHOLD
    }
    try:
        with open(RAG_CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"Не удалось сохранить RAG config: {e}")

def load_rag_config():
    global EMBEDDING_MODEL_NAME, RERANKER_MODEL_NAME, QUANTIZATION, RAG_TOP_K_PER_FILE, RAG_RERANK_POOL, RAG_FINAL_TOP_N, USE_RERANKER, GGUF_SEARCH_DIRS, RAG_QUERY_EXPANSION, RERANK_SCORE_THRESHOLD
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
                RAG_QUERY_EXPANSION = data.get("query_expansion", RAG_QUERY_EXPANSION)
                RERANK_SCORE_THRESHOLD = float(data.get("rerank_score_threshold", RERANK_SCORE_THRESHOLD))
    except Exception as e:
        logger.warning(f"Не удалось загрузить RAG config: {e}")

# Загружаем настройки при старте
load_rag_config()

@lru_cache(maxsize=64)
def resolve_model_path(path_or_filename: str) -> str:
    """
    Если путь абсолютный и существует — возвращает его.
    Иначе ищет файл через gguf_manager.find_gguf_by_name (mtime-keyed cache),
    и только в крайнем случае делает свежий os.walk.
    """
    if not path_or_filename:
        return ""

    # Если это уже существующий абсолютный путь
    if os.path.isabs(path_or_filename) and os.path.exists(path_or_filename):
        return os.path.normpath(path_or_filename).lower()

    # Быстрый путь: кешированный scan через gguf_manager
    try:
        from src.gguf_manager import find_gguf_by_name
        hit = find_gguf_by_name(path_or_filename)
        if hit:
            print(f"[CONFIG] Модель найдена: {hit}")
            return os.path.normpath(hit).lower()
    except Exception as e:
        logger.debug(f"[CONFIG] gguf_manager.find_gguf_by_name failed: {e}")

    # Fallback (на случай если gguf_manager не импортируется): свежий os.walk
    search_dirs = [d.strip() for d in GGUF_SEARCH_DIRS.split(";") if d.strip()]
    filename = os.path.basename(path_or_filename)

    for base_dir in search_dirs:
        for dirpath, _dirnames, filenames in os.walk(base_dir):
            if filename in filenames:
                full_path = os.path.join(dirpath, filename)
                print(f"[CONFIG] Модель найдена (fallback walk): {full_path}")
                return os.path.normpath(full_path).lower()

    return path_or_filename
