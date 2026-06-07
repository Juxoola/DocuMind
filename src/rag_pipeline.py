import os
import logging
import chromadb
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.vector_stores.types import MetadataFilters, MetadataFilter, FilterOperator
from llama_index.core.schema import TextNode

import config
import torch
import threading
import requests
import requests.adapters
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import BitsAndBytesConfig

# F-fix #15: Session для rerank-запросов (см. также ingestion.py _http_session).
# Без Session каждый POST /v1/rerank открывает новый TCP-коннект.
_rerank_session = requests.Session()
_rerank_session.mount("http://", requests.adapters.HTTPAdapter(pool_connections=config.HTTP_POOL_SIZE_RERANK, pool_maxsize=config.HTTP_POOL_SIZE_RERANK))
_rerank_session.mount("https://", requests.adapters.HTTPAdapter(pool_connections=config.HTTP_POOL_SIZE_RERANK, pool_maxsize=config.HTTP_POOL_SIZE_RERANK))

logger = logging.getLogger(__name__)

_client_cache = {}
_model_cache = {}
# F-fix #6: init_lock предотвращает гонку при конкурентной загрузке моделей.
# Без лока два потока (build_index + retrieve_nodes) могут независимо увидеть
# "embed_model" отсутствует, создать 2 экземпляра HuggingFaceEmbedding (~1.3GB каждый)
# → OOM. Lock удерживается только на init-фазу; повторные вызовы — no-op (cache hit).
_init_lock = threading.Lock()


def unload_rag_models(hard=True):
    """Выгрузка моделей RAG. Если hard=False, модели остаются в памяти (только очистка кэша)."""
    global _model_cache
    if not _model_cache:
        return

    # F-fix #6: под локом, чтобы clear() не сломал параллельный init_settings
    # (который между cache-miss и _model_cache[k] = ... мог бы увидеть clear() и получить полу-инициализированный кеш).
    with _init_lock:
        if hard:
            print("[RAG] Выгрузка всех моделей (Embedding, Reranker)...")
            _model_cache.clear()
        else:
            print("[RAG] Мягкая очистка (Эмбеддинги и Реранкер остаются в памяти)...")

    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[RAG] Память очищена.")


def preload_all_models():
    """Предзагрузка всех тяжелых моделей для ускорения работы."""
    print("[RAG] Предзагрузка моделей...")
    init_settings()
    if config.RERANKER_MODEL_NAME:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        is_gguf = config.RERANKER_MODEL_NAME.lower().endswith('.gguf') or (os.path.isabs(config.RERANKER_MODEL_NAME) and os.path.exists(config.RERANKER_MODEL_NAME))
        
        if is_gguf:
            print(f"  [RAG] Предзагрузка GGUF реранкера: {config.RERANKER_MODEL_NAME}")
            from src.gguf_direct import get_gguf_embedding_url
            model_path = config.resolve_model_path(config.RERANKER_MODEL_NAME)
            get_gguf_embedding_url(model_path, is_reranker=True)
        else:
            if "reranker" not in _model_cache:
                print(f"  [RAG] Предзагрузка реранкера: {config.RERANKER_MODEL_NAME}")
                from sentence_transformers import CrossEncoder
                _model_cache["reranker"] = CrossEncoder(config.RERANKER_MODEL_NAME, device=device)
    
    # Загружаем GGUF LLM для зрения больше не нужно, так как он грузится динамически в ingestion.py
    # и сразу очищается, экономя VRAM.
    
    print("[RAG] Все модели загружены.")

def init_settings(max_tokens=1024):
    global _model_cache
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # F-fix #6: lock на init-фазу. После инициализации cache hit не требует лока.
    # Settings.llm пересоздаётся на каждом вызове — это идемпотентно, не под локом.
    with _init_lock:
        if "embed_model" not in _model_cache:
            model_name = config.EMBEDDING_MODEL_NAME

            if model_name.lower().endswith('.gguf') or os.path.isabs(model_name) and os.path.exists(model_name):
                print(f"Инициализация GGUF эмбеддингов: {model_name}")
                from src.gguf_direct import get_gguf_embedding_url
                from llama_index.embeddings.openai import OpenAIEmbedding

                model_path = config.resolve_model_path(model_name)
                # n_parallel пробрасывается в init_settings через init_lock-guarded вызов,
                # чтобы embed_batch_size == --parallel на embedding-сервере.
                url = get_gguf_embedding_url(model_path)
                try:
                    from src.gguf_direct import get_active_embedding_parallel
                    n_parallel = get_active_embedding_parallel(model_path)
                except Exception:
                    n_parallel = 1
                print(f"[RAG] GGUF embedding server --parallel={n_parallel} → embed_batch_size={n_parallel}")

                _model_cache["embed_model"] = OpenAIEmbedding(
                    api_base=f"{url}/v1",
                    api_key="sk-local",
                    model="text-embedding-ada-002",
                    timeout=120.0,
                    # Сервер запущен с -c 4096 (= максимальный размер 1 чанка).
                    # Каждый документ обрабатывается независимо, батч не складывает токены.
                    # embed_batch_size = n_parallel embedding-сервера: больше слотов — больше параллелизма.
                    embed_batch_size=n_parallel,
                    # Инструкция для Qwen3-Embedding, чтобы он понимал задачу поиска
                    query_header="Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "
                )
            else:
                print(f"Инициализация эмбеддингов (PyTorch): {device.upper()} (Quant: {config.QUANTIZATION})")
                model_kwargs = {"trust_remote_code": True}
                model_kwargs["attn_implementation"] = "sdpa"

                if config.QUANTIZATION == "4bit":
                    model_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_quant_type="nf4"
                    )
                elif config.QUANTIZATION == "int8":
                    model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
                else:
                    model_kwargs["torch_dtype"] = torch.bfloat16

                _model_cache["embed_model"] = HuggingFaceEmbedding(
                    model_name=model_name,
                    device=device,
                    model_kwargs=model_kwargs
                )

    Settings.embed_model = _model_cache["embed_model"]

    Settings.llm = OpenAI(
        api_base=config.LM_STUDIO_URL,
        api_key="lm-studio",
        model="gpt-4o",
        temperature=config.CHAT_TEMPERATURE,
        max_tokens=max_tokens
    )

def close_all_clients():
    """Явно закрывает все открытые клиенты ChromaDB для снятия блокировок файлов."""
    global _client_cache
    for path, client in _client_cache.items():
        try:
            client.close()
        except Exception as e:
            logger.debug(f"Ошибка закрытия ChromaDB клиента {path}: {e}")
    _client_cache.clear()

def get_vector_store(notebook_id: str):
    global _client_cache
    paths = config.get_notebook_paths(notebook_id)
    db_path = paths["chroma_db"]
    os.makedirs(db_path, exist_ok=True)
    
    if db_path not in _client_cache:
        _client_cache[db_path] = chromadb.PersistentClient(path=db_path)
    
    db = _client_cache[db_path]
    chroma_collection = db.get_or_create_collection("multimodal_rag")
    return ChromaVectorStore(chroma_collection=chroma_collection)

_QUERY_GEN_PROMPT = (
    "Ты — эксперт по поиску информации. Сформулируй ровно {num_queries} разных коротких поисковых запроса "
    "на том же языке для поиска справочной теории, правил и формул в учебных материалах на основе следующего задания/вопроса.\n"
    "[ВАЖНО: Пиши ТОЛЬКО готовые поисковые запросы, по одному на строке. НЕ пиши никаких вступлений, пояснений, номеров и тегов <think>.]\n"
    "[КРИТИЧЕСКИ ВАЖНО: Категорически запрещено писать общие мусорные фразы вроде 'решение задачи', 'пошаговое решение', 'сложная математика', 'пример по математике', 'решить тест'. "
    "Запросы должны состоять строго из конкретных научных терминов, названий теорем, правил или тем из учебника, найденных в задании (например: 'нечеткая логика функция принадлежности', 'лингвистическая переменная'). Исключай конкретные числа.]\n"
    "Задание/вопрос: {query}\n"
    "Поисковые запросы:"
)

def _get_qe_llm():
    """Returns an LLM instance for Query Expansion.
    
    Priority:
    1. Running GGUF LLM server (get_active_llm_url)
    2. LM Studio (config.LM_STUDIO_URL)
    3. None — QE will be skipped silently
    """
    from src.gguf_direct import get_active_llm_url
    url = get_active_llm_url()
    if url:
        logger.debug(f"[QE] Используем GGUF LLM для Query Expansion: {url}")
    else:
        url = config.LM_STUDIO_URL  # фоллбэк на LM Studio
        logger.debug(f"[QE] GGUF LLM не найден, пробуем LM Studio: {url}")
    try:
        import requests as _req
        # Быстрая проверка доступности сервера (без ретраев)
        _req.get(url.replace("/v1", "").rstrip("/") + "/health", timeout=1)
    except Exception:
        logger.debug("[QE] LLM-сервер недоступен, Query Expansion пропускается")
        return None
    from llama_index.llms.openai import OpenAI as _OpenAI
    return _OpenAI(
        api_base=url if url.endswith("/v1") else f"{url}/v1",
        api_key="sk-local",
        model="gpt-4o",
        temperature=0.3,
        max_tokens=512,       # Достаточно для генерации 3 простых фраз
        timeout=20.0,          # Запас таймаута
        max_retries=0,         # Без ретраев — если нет ответа, сразу фоллбэк
        additional_kwargs={
            "extra_body": {
                "thinking_budget": 0,
                "thinking_budget_tokens": 0,
                "chat_template_kwargs": {"enable_thinking": False}
            }
        }
    )

def _rebuild_bm25_bg(notebook_id: str, db_path: str):
    """Перестройка BM25-индекса в фоновом потоке. Не блокирует основной поток."""
    try:
        paths = config.get_notebook_paths(notebook_id)
        bm25_dir = os.path.join(paths["base"], "bm25")
        os.makedirs(bm25_dir, exist_ok=True)
        # Отдельный клиент для фонового потока (нельзя использовать общий _client_cache)
        import chromadb as _chromadb
        tmp_client = _chromadb.PersistentClient(path=db_path)
        collection = tmp_client.get_or_create_collection("multimodal_rag")
        result = collection.get()
        bm25_nodes = []
        for i, doc_id in enumerate(result['ids']):
            text = result['documents'][i]
            meta = result['metadatas'][i] or {}
            # F1: BM25 видит координаты чанка. Только для BM25-токенизации;
            # embedding-вектора в ChromaDB не меняются, переиндексация не нужна.
            fname = meta.get('file_name', '')
            page = meta.get('page', '')
            t = meta.get('start', meta.get('time', ''))
            coord_parts = []
            if fname: coord_parts.append(str(fname))
            if page not in ('', None): coord_parts.append(f"стр.{page}")
            elif t not in ('', None): coord_parts.append(f"@{t}")
            if coord_parts:
                text = f"[{' '.join(coord_parts)}]: {text}"
            bm25_nodes.append(TextNode(text=text, id_=doc_id, metadata=meta))
        if bm25_nodes:
            from llama_index.retrievers.bm25 import BM25Retriever
            retriever = BM25Retriever.from_defaults(
                nodes=bm25_nodes,
                similarity_top_k=config.RAG_TOP_K_PER_FILE,
                language="russian"
            )
            retriever.persist(bm25_dir)
            print(f"[RAG] ✅ BM25 обновлён в фоне: {len(bm25_nodes)} узлов.")
    except Exception as e:
        logger.warning(f"[RAG] Ошибка фоновой сборки BM25: {e}")


# ── BM25 rebuild debouncer ─────────────────────────────────────────────
# При batch-загрузке (10 PDF подряд) вызов _rebuild_bm25_bg читал ВСЮ ChromaDB
# 10 раз подряд → thrashing диска. Теперь build_index откладывает rebuild на
# _BM25_DEBOUNCE_SEC и сбрасывает таймер на каждом новом файле; main.py
# вызывает flush_bm25_rebuild() в конце batch для немедленной пересборки.
_BM25_DEBOUNCE_SEC = 30.0
_bm25_pending_timers: dict = {}      # notebook_id -> threading.Timer
_bm25_pending_dbpath: dict = {}      # notebook_id -> str (db_path для callback)
_bm25_pending_lock = threading.Lock()
_bm25_rebuilding: set = set()        # notebook_id которые сейчас строят BM25


def _schedule_bm25_rebuild(notebook_id: str, db_path: str):
    """Отложить rebuild BM25. Каждый новый вызов сбрасывает таймер на _BM25_DEBOUNCE_SEC."""
    with _bm25_pending_lock:
        old = _bm25_pending_timers.get(notebook_id)
        if old is not None:
            try: old.cancel()
            except Exception: pass
        _bm25_pending_dbpath[notebook_id] = db_path
        def _fire():
            with _bm25_pending_lock:
                _bm25_pending_timers.pop(notebook_id, None)
                path = _bm25_pending_dbpath.pop(notebook_id, None)
            if path is None:
                return
            _bm25_rebuilding.add(notebook_id)
            try:
                _rebuild_bm25_bg(notebook_id, path)
            finally:
                _bm25_rebuilding.discard(notebook_id)
        t = threading.Timer(_BM25_DEBOUNCE_SEC, _fire)
        t.daemon = True
        _bm25_pending_timers[notebook_id] = t
        t.start()
        print(f"[RAG] ⏱ BM25 rebuild запланирован через {_BM25_DEBOUNCE_SEC:.0f}с (можно сбросить через flush_bm25_rebuild)")


def flush_bm25_rebuild(notebook_id: str, db_path: str = None, wait: bool = False, timeout: float = 120.0):
    """Форсировать немедленную пересборку BM25. Используется в конце batch-upload.

    Args:
        notebook_id: ID ноутбука.
        db_path: путь к ChromaDB (если None — берётся из config).
        wait: если True — блокирует вызывающий поток до завершения rebuild
              (нужно перед первым RAG-запросом, если BM25 ещё не было).
        timeout: макс. время ожидания при wait=True.
    """
    with _bm25_pending_lock:
        timer = _bm25_pending_timers.pop(notebook_id, None)
        if timer is not None:
            try: timer.cancel()
            except Exception: pass
        path = _bm25_pending_dbpath.pop(notebook_id, None)
        if path is None and db_path is not None:
            path = db_path
        if path is None:
            paths = config.get_notebook_paths(notebook_id)
            path = paths["chroma_db"]
    if path is None:
        return
    if not wait:
        _bm25_rebuilding.add(notebook_id)
        def _bg():
            try:
                _rebuild_bm25_bg(notebook_id, path)
            finally:
                _bm25_rebuilding.discard(notebook_id)
        threading.Thread(target=_bg, daemon=True, name=f"bm25-flush-{notebook_id}").start()
        return
    # wait=True: синхронный rebuild в текущем потоке
    _bm25_rebuilding.add(notebook_id)
    try:
        _rebuild_bm25_bg(notebook_id, path)
    finally:
        _bm25_rebuilding.discard(notebook_id)


def is_bm25_ready(notebook_id: str) -> bool:
    """True если BM25-индекс существует на диске и нет pending/rebuilding."""
    paths = config.get_notebook_paths(notebook_id)
    bm25_dir = os.path.join(paths["base"], "bm25")
    exists = os.path.exists(os.path.join(bm25_dir, "bm25_retriever_params.json"))
    with _bm25_pending_lock:
        has_pending = notebook_id in _bm25_pending_timers
    is_rebuilding = notebook_id in _bm25_rebuilding
    return exists and not has_pending and not is_rebuilding

def _rrf_fuse_across_files(file_results, k: int = 60):
    """F2: Merge RRF scores across files so big files don't dominate.

    Args:
        file_results: list of (file_name, [NodeWithScore]) tuples
        k: RRF constant

    Returns:
        list[NodeWithScore] sorted by cross-file RRF score desc, deduplicated.
    """
    from llama_index.core.schema import NodeWithScore

    scores: dict = {}
    nodes_by_id: dict = {}

    for _fname, per_file_nodes in file_results:
        # Внутри файла — rank = позиция в уже RRF-слитом списке
        for rank, nws in enumerate(per_file_nodes, start=1):
            nid = nws.node.node_id
            scores[nid] = scores.get(nid, 0.0) + 1.0 / (k + rank)
            nodes_by_id[nid] = nws

    sorted_ids = sorted(scores.keys(), key=lambda i: scores[i], reverse=True)
    return [NodeWithScore(node=nodes_by_id[i].node, score=scores[i]) for i in sorted_ids]


def _rrf_fuse(vector_results, bm25_results, k: int = 60):
    """Reciprocal Rank Fusion (Cormack et al. 2009).

    RRF_score(d) = Σ 1 / (k + rank_i(d)) для каждого retriever i.

    Зачем: BM25-only чанки (semantic_score=0 но bm25_score высокий) получают
    честный шанс попасть в топ. Раньше они шли после vector-only чанков
    с residual similarity, и терялись в top-K.

    Args:
        vector_results: list[NodeWithScore] от vector_retriever
        bm25_results:   list[NodeWithScore] от BM25Retriever (или [])
        k: константа сглаживания (60 — стандарт)

    Returns:
        list[NodeWithScore], отсортированный по RRF score desc, без дубликатов.
    """
    from llama_index.core.schema import NodeWithScore

    scores: dict = {}
    nodes_by_id: dict = {}

    for rank, nws in enumerate(vector_results, start=1):
        nid = nws.node.node_id
        scores[nid] = scores.get(nid, 0.0) + 1.0 / (k + rank)
        nodes_by_id[nid] = nws

    for rank, nws in enumerate(bm25_results, start=1):
        nid = nws.node.node_id
        scores[nid] = scores.get(nid, 0.0) + 1.0 / (k + rank)
        nodes_by_id[nid] = nws

    sorted_ids = sorted(scores.keys(), key=lambda i: scores[i], reverse=True)
    return [NodeWithScore(node=nodes_by_id[i].node, score=scores[i]) for i in sorted_ids]


def build_index(nodes, notebook_id: str):
    init_settings()
    vector_store = get_vector_store(notebook_id)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(nodes, storage_context=storage_context)

    # Debounced BM25 rebuild: при batch из 10 файлов будет ОДНА пересборка
    # через 30с после последнего файла, а не 10 параллельных (F-fix #4).
    paths = config.get_notebook_paths(notebook_id)
    db_path = paths["chroma_db"]
    _schedule_bm25_rebuild(notebook_id, db_path)
    return index

def retrieve_nodes(query: str, notebook_id: str, allowed_files=None, max_tokens=1024):
    """
    Для каждого выбранного файла выполняем отдельный гибридный поиск топ-K чанков.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    init_settings(max_tokens=max_tokens)
    vector_store = get_vector_store(notebook_id)
    index = VectorStoreIndex.from_vector_store(vector_store)

    if not allowed_files:
        return []

    # Загрузка BM25 ретривера
    paths = config.get_notebook_paths(notebook_id)
    bm25_dir = os.path.join(paths["base"], "bm25")
    from llama_index.retrievers.bm25 import BM25Retriever
    from llama_index.core.retrievers import QueryFusionRetriever
    
    bm25_retriever = None
    if os.path.exists(os.path.join(bm25_dir, "bm25_retriever_params.json")):
        try:
            bm25_retriever = BM25Retriever.from_persist_dir(bm25_dir)
        except Exception as e:
            logger.warning(f"Не удалось загрузить BM25: {e}")
    else:
        # BM25-индекс ещё не собран (debounced rebuild в очереди).
        # Для одного файла — форсируем синхронную сборку, иначе vector-only поиск
        # не имеет гибридного преимущества. batch-upload сам вызывает flush_bm25_rebuild
        # в конце — там дублирующая сборка будет отменена.
        if not is_bm25_ready(notebook_id):
            print(f"  [RAG] BM25 отсутствует — форсирую синхронную пересборку для первого запроса")
            flush_bm25_rebuild(notebook_id, db_path=paths["chroma_db"], wait=True, timeout=180)
            if os.path.exists(os.path.join(bm25_dir, "bm25_retriever_params.json")):
                try:
                    bm25_retriever = BM25Retriever.from_persist_dir(bm25_dir)
                except Exception as e:
                    logger.warning(f"Не удалось загрузить BM25 после flush: {e}")

    all_nodes = []

    # Определяем, доступен ли LLM для Query Expansion
    qe_llm = _get_qe_llm() if config.RAG_QUERY_EXPANSION else None
    if config.RAG_QUERY_EXPANSION:
        if qe_llm:
            print(f"  [RAG] Query Expansion включён (num_queries=3)")
        else:
            print(f"  [RAG] Query Expansion отключён (нет доступного LLM-сервера)")

    # F2: Per-file RRF — каждый файл получает равный голос, независимо от размера.
    # Внутри файла — RRF (vector+BM25), между файлами — RRF поверх.
    # Один файл → используем один MetadataFilter (быстрее).
    if len(allowed_files) == 1:
        file_filter = MetadataFilters(
            filters=[MetadataFilter(key="file_name", value=allowed_files[0], operator=FilterOperator.EQ)]
        )
    else:
        file_filter = MetadataFilters(
            filters=[MetadataFilter(key="file_name", value=allowed_files, operator=FilterOperator.IN)]
        )

    # Сколько кандидатов брать с каждого файла (используется и per-file, и как top_k для одиночного файла)
    top_k_per_file = config.RAG_TOP_K_PER_FILE

    # Один файл — старый быстрый путь (без per-file overhead)
    vector_retriever = index.as_retriever(
        similarity_top_k=top_k_per_file,
        filters=file_filter
    )

    num_q = 3 if qe_llm else 1
    qprompt = _QUERY_GEN_PROMPT if qe_llm else None

    use_qe = (num_q > 1)  # Query Expansion активен (нужен LLM-сервер)

    try:
        if use_qe:
            # F2 + QE: создаём per-file retrievers и передаём их в QueryFusionRetriever.
            # Каждый per-file retriever получает ВСЕ варианты запросов (orig + 2 от LLM).
            # QueryFusionRetriever делает RRF-мерж, mode='reciprocal_rerank' (по умолчанию)
            # корректно мерджит per-file результаты.
            per_file_retrievers = []
            for fname in allowed_files:
                ff = MetadataFilters(
                    filters=[MetadataFilter(key="file_name", value=fname, operator=FilterOperator.EQ)]
                )
                per_file_retrievers.append(
                    index.as_retriever(similarity_top_k=top_k_per_file, filters=ff)
                )
            if bm25_retriever:
                # BM25 — общий, фильтруем post-hoc (не все версии llama-index поддерживают filter в BM25)
                per_file_retrievers.append(bm25_retriever)

            fusion_retriever = QueryFusionRetriever(
                per_file_retrievers,
                similarity_top_k=top_k_per_file * len(allowed_files),
                num_queries=num_q,
                query_gen_prompt=qprompt,
                llm=qe_llm,
                use_async=False,
                mode="reciprocal_rerank",  # RRF внутри QueryFusionRetriever
            )

            # Внедряем custom_get_queries для надежной обработки thinking-моделей и очистки от разметки/нумерации
            if num_q > 1 and qe_llm:
                def custom_get_queries(original_query: str):
                    try:
                        prompt_str = fusion_retriever.query_gen_prompt.format(
                            num_queries=fusion_retriever.num_queries - 1,
                            query=original_query,
                        )
                        response = fusion_retriever._llm.complete(prompt_str)
                        text = response.text or ""

                        # Если модель думает, отсекаем <think>...</think>
                        if "<think>" in text:
                            parts = text.split("</think>")
                            text = parts[-1]
                        text = text.replace("<think>", "").replace("</think>", "")

                        # Парсим строки и чистим от нумерации/маркеров списка
                        lines = text.strip("`").split("\n")
                        queries = []
                        import re
                        for line in lines:
                            line = line.strip()
                            if not line:
                                continue
                            line = re.sub(r'^\d+[\.\)]\s*', '', line) # Убирает "1. ", "2) "
                            line = re.sub(r'^[-\*\+]\s*', '', line)   # Убирает списка "- ", "* "
                            line = line.strip()
                            if line:
                                queries.append(line)

                        # Возвращаем QueryBundle для сгенерированных запросов
                        from llama_index.core import QueryBundle
                        return [QueryBundle(q) for q in queries[:fusion_retriever.num_queries - 1]]
                    except Exception as qe_err:
                        logger.warning(f"Ошибка генерации запросов в custom_get_queries: {qe_err}")
                        return []

                fusion_retriever._get_queries = custom_get_queries

            # Логируем сгенерированные запросы для пользователя
            if num_q > 1 and qe_llm:
                try:
                    generated_bundles = fusion_retriever._get_queries(query)
                    print(f"  [RAG] 🧠 Сгенерированные поисковые запросы (Query Expansion):")
                    for i, gq in enumerate(generated_bundles, 1):
                        print(f"    {i}. {gq.query_str}")
                except Exception as qe_err:
                    logger.warning(f"Не удалось получить сгенерированные запросы для лога: {qe_err}")

            all_nodes = fusion_retriever.retrieve(query)
            # Пост-фильтр: убеждаемся, что только разрешённые файлы (BM25 может вернуть чужие)
            all_nodes = [n for n in all_nodes if n.node.metadata.get("file_name") in allowed_files]

            if bm25_retriever:
                label = f"Per-file Гибрид+QE({num_q})"
            else:
                label = f"Per-file Вектор+QE({num_q})"
            print(f"  [RAG] 🔍 {label} по {len(allowed_files)} файлам: {len(all_nodes)} фрагм.")
        else:
            # F2: QE выключен → per-file RRF.
            # Для каждого файла: vector (top_k_per_file) + BM25 (top_k_per_file, post-filter) → RRF
            # Затем RRF merge across files.
            if len(allowed_files) == 1:
                # 1 файл — простой путь, без per-file overhead
                vec_results = vector_retriever.retrieve(query)
                bm25_results = bm25_retriever.retrieve(query) if bm25_retriever else []
                bm25_results = [n for n in bm25_results if n.node.metadata.get("file_name") == allowed_files[0]][:top_k_per_file]
                all_nodes = _rrf_fuse(vec_results, bm25_results)
                if bm25_retriever:
                    print(f"  [RAG] 🔍 Гибрид (RRF, vector+BM25) по 1 файлу: {len(all_nodes)} фрагм.")
                else:
                    print(f"  [RAG] 🔍 Вектор по 1 файлу: {len(all_nodes)} фрагм.")
            else:
                # F-fix #13: 1 IN-filter vector query вместо N per-file queries.
                # N=5 файлов × 1 запрос = 5 ChromaDB round-trips. С IN-filter — 1 round-trip.
                # Семантика per-file RRF (равный голос каждого файла) СОХРАНЯЕТСЯ:
                # берём top_k_per_file × N результатов, группируем по file_name,
                # затем берём top_k_per_file из каждой группы → RRF merge.
                fetch_k = top_k_per_file * len(allowed_files)
                vec_results_all = vector_retriever.retrieve(query)[:fetch_k]
                bm25_all = bm25_retriever.retrieve(query) if bm25_retriever else []

                # Группируем vector results по файлу
                vec_by_file: dict = {}
                for n in vec_results_all:
                    fn = n.node.metadata.get("file_name", "")
                    if fn in allowed_files:
                        vec_by_file.setdefault(fn, []).append(n)
                # Оставляем top_k_per_file из каждого файла
                vec_results = []
                for fn in allowed_files:
                    vec_results.extend(vec_by_file.get(fn, [])[:top_k_per_file])

                file_results = []
                for fname in allowed_files:
                    # BM25: берём общий результат, фильтруем по файлу, top_k_per_file
                    bm = [n for n in bm25_all if n.node.metadata.get("file_name") == fname][:top_k_per_file]
                    # Vector: уже отфильтровано и capped выше
                    vec = [n for n in vec_results if n.node.metadata.get("file_name") == fname]
                    fused = _rrf_fuse(vec, bm)
                    file_results.append((fname, fused))
                all_nodes = _rrf_fuse_across_files(file_results)
                # Cap на RAG_RERANK_POOL, чтобы не раздувать reranker budget
                if len(all_nodes) > config.RAG_RERANK_POOL:
                    all_nodes = all_nodes[:config.RAG_RERANK_POOL]
                if bm25_retriever:
                    print(f"  [RAG] 🔍 Per-file Гибрид (RRF, vector+BM25) по {len(allowed_files)} файлам: {len(all_nodes)} фрагм.")
                else:
                    print(f"  [RAG] 🔍 Per-file Вектор по {len(allowed_files)} файлам: {len(all_nodes)} фрагм.")
    except Exception as e:
        print(f"Ошибка унифицированного поиска: {e}")
        all_nodes = []

    # Переранжирование (Reranking)
    if all_nodes and config.USE_RERANKER:
        # Ограничиваем общее число чанков для реранкера, чтобы избежать OOM
        if len(all_nodes) > config.RAG_RERANK_POOL:
            all_nodes.sort(key=lambda x: x.score if hasattr(x, 'score') and x.score else 0, reverse=True)
            all_nodes = all_nodes[:config.RAG_RERANK_POOL]
            
        print(f"  [RAG] Чанков для реранкинга: {len(all_nodes)}")
        
        reranker_name = config.RERANKER_MODEL_NAME
        
        if reranker_name.lower().endswith('.gguf') or os.path.isabs(reranker_name) and os.path.exists(reranker_name):
            if "reranker" not in _model_cache:
                print(f"  [RAG] Загрузка GGUF реранкера: {reranker_name}")
                from src.gguf_direct import get_gguf_embedding_url
                model_path = config.resolve_model_path(reranker_name)
                url = get_gguf_embedding_url(model_path, is_reranker=True)
                _model_cache["reranker"] = url
            
            url = _model_cache["reranker"]
            # F5: reranker получает префикс с координатами чанка, чтобы cross-encoder
            # мог учитывать координаты при оценке (для запросов "что на стр. 5 лекции 3").
            # Префикс короткий (<50 токенов) — не раздувает input и не съедает полезный контекст.
            def _rerank_doc(nws):
                meta = nws.node.metadata or {}
                coord_parts = []
                if meta.get("file_name"): coord_parts.append(str(meta["file_name"]))
                if meta.get("page") not in (None, ""): coord_parts.append(f"стр.{meta['page']}")
                elif meta.get("time") not in (None, ""): coord_parts.append(f"@{meta['time']}")
                elif meta.get("start") not in (None, ""): coord_parts.append(f"@{meta['start']}")
                prefix = f"[{' '.join(coord_parts)}] " if coord_parts else ""
                return prefix + nws.node.get_content()
            documents = [_rerank_doc(n) for n in all_nodes]
            payload = {"model": "gguf-reranker", "query": query, "documents": documents, "top_n": len(documents)}
            
            try:
                import time as _time
                _rerank_start = _time.time()

                # F-fix #12: один запрос вместо N мини-батчей.
                # llama.cpp /v1/rerank обрабатывает ВСЕ документы в одном HTTP-вызове.
                # Каждая пара (query+doc) независима и обрабатывается последовательно внутри
                # сервера, но без HTTP-overhead между ними. При --parallel > 1 сервер сам
                # параллелит обработку внутри одного запроса.
                # Экономия: 3× HTTP roundtrip (~50-100 мс каждый) + упрощение кода.
                # 30 doc × 0.3с = 9с — вписывается в timeout=120с.
                # F-fix #15: используем _rerank_session (HTTP connection pool) вместо requests.post напрямую.
                scores = [0.0] * len(all_nodes)
                success = True
                resp = _rerank_session.post(
                    f"{url}/v1/rerank",
                    json={"model": "gguf-reranker", "query": query, "documents": documents, "top_n": len(documents)},
                    timeout=120,
                )
                resp.raise_for_status()
                results = resp.json().get("results", [])
                if not results:
                    logger.debug(f"[RAG] Реранкер вернул пустой results: {resp.text[:200]}")
                for r in results:
                    orig_idx = r.get("index", 0)
                    if orig_idx < len(scores):
                        scores[orig_idx] = r.get("relevance_score", 0.0)

                elapsed_r = _time.time() - _rerank_start
                if success:
                    print(f"  [RAG] ✅ Реранкинг: {len(documents)} doc за {elapsed_r:.2f}с")

            except Exception as e:
                err_body = ""
                if hasattr(e, 'response') and e.response is not None:
                    err_body = f" body={e.response.text[:300]}"
                print(f"[RAG] Ошибка GGUF реранкера: {e}{err_body}")
                scores = [0] * len(all_nodes)


            # ПРОВЕРКА: Если все скоры слишком маленькие (например < 1e-6), 
            # значит реранкер "ослеп" и выдает шум. В этом случае лучше сохранить 
            # оригинальный порядок от BM25/Вектора.
            if scores and max(scores) < 1e-6:
                print(f"  [RAG] ⚠️ GGUF реранкер выдал слишком низкие оценки (max: {max(scores)}). Используется оригинальный порядок поиска.")
                # Оставляем оригинальные скоры от ретривера (0.01 за каждый шаг, чтобы сохранить порядок)
                scores = [1.0 - (i * 0.01) for i in range(len(all_nodes))]
        else:
            if "reranker" not in _model_cache:
                print(f"  [RAG] Загрузка реранкера: {reranker_name} ({config.QUANTIZATION})")
                rerank_kwargs = {"trust_remote_code": True}
                rerank_kwargs["attn_implementation"] = "sdpa"
                if config.QUANTIZATION == "4bit":
                    rerank_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True, 
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_quant_type="nf4"
                    )
                elif config.QUANTIZATION == "int8":
                    rerank_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
                else:
                    rerank_kwargs["torch_dtype"] = torch.bfloat16

                from sentence_transformers import CrossEncoder
                _model_cache["reranker"] = CrossEncoder(reranker_name, device=device, model_kwargs=rerank_kwargs)
            
            model = _model_cache["reranker"]
            # F5: PyTorch reranker тоже видит координаты (аналогично GGUF-ветке выше)
            def _rerank_doc(nws):
                meta = nws.node.metadata or {}
                coord_parts = []
                if meta.get("file_name"): coord_parts.append(str(meta["file_name"]))
                if meta.get("page") not in (None, ""): coord_parts.append(f"стр.{meta['page']}")
                elif meta.get("time") not in (None, ""): coord_parts.append(f"@{meta['time']}")
                elif meta.get("start") not in (None, ""): coord_parts.append(f"@{meta['start']}")
                prefix = f"[{' '.join(coord_parts)}] " if coord_parts else ""
                return prefix + nws.node.get_content()
            pairs = [[query, _rerank_doc(n)] for n in all_nodes]
            scores = model.predict(pairs)
        
        # Присваиваем скоры и сортируем
        for node, score in zip(all_nodes, scores):
            node.score = float(score)
            
        all_nodes.sort(key=lambda x: x.score, reverse=True)
        all_nodes = all_nodes[:config.RAG_FINAL_TOP_N]

        # F6: Adaptive threshold через MAD (median absolute deviation).
        # Qwen3-Reranker выдаёт логиты в широком диапазоне, абсолютный порог 0.05
        # плохо работает: для расплывчатых вопросов "что такое X" средний score ~0.6,
        # для точных "формула 4.12" средний ~0.2. MAD-based порог адаптируется к
        # распределению скоров в конкретном query.
        import statistics as _stats
        if len(all_nodes) >= 4:
            score_vals = [n.score for n in all_nodes]
            median = _stats.median(score_vals)
            mad = _stats.median([abs(s - median) for s in score_vals]) or 0.05
            adaptive_thr = max(0.0, median - 2.0 * mad)
        else:
            adaptive_thr = config.RERANK_SCORE_THRESHOLD

        above_threshold = [n for n in all_nodes if n.score >= adaptive_thr]

        # Гарантируем минимум MIN_FINAL_CHUNKS (чтобы не терять контекст для сложных вопросов)
        min_chunks = min(config.MIN_FINAL_CHUNKS, len(all_nodes))

        if len(above_threshold) >= min_chunks:
            if len(above_threshold) < len(all_nodes):
                print(f"  [RAG] 🎯 Адаптивный порог {adaptive_thr:.3f} (median-MAD): убрано {len(all_nodes) - len(above_threshold)} чанков")
            all_nodes = above_threshold
        else:
            # Оказалось слишком мало хороших чанков — добираем до минимума из лучших ниже порога
            all_nodes = all_nodes[:min_chunks]
            print(f"  [RAG] ⚠️ Адаптивный порог {adaptive_thr:.3f} оставил <{min_chunks} чанков. Добавлено до {min_chunks} лучших (мин. score: {all_nodes[-1].score:.3f})")

        # F6+: Top-K relevance ratio. F6 median-MAD отлично работает на бимодальных
        # распределениях (cluster высоких + cluster низких), но плохо — когда median
        # и MAD оба маленькие (топ-2 сильно выше, остальные плотным комом мусора).
        # В таком случае adaptive_thr ≈ 0 и ВСЕ чанки проходят, загрязняя контекст.
        # Дополнительно режем всё, что в RAG_TOP_K_RATIO раз хуже топ-1: это
        # гарантированный мусор (reranker не ошибается на 10x ratio).
        if config.RAG_TOP_K_RATIO > 0 and all_nodes:
            top_score = all_nodes[0].score
            ratio_thr = top_score * config.RAG_TOP_K_RATIO
            above_ratio = [n for n in all_nodes if n.score >= ratio_thr]
            if len(above_ratio) >= min_chunks and len(above_ratio) < len(all_nodes):
                print(f"  [RAG] 🎯 Top-K ratio {config.RAG_TOP_K_RATIO:.2f} (порог {ratio_thr:.3f} = {top_score:.3f}*{config.RAG_TOP_K_RATIO:.2f}): убрано {len(all_nodes) - len(above_ratio)} чанков")
                all_nodes = above_ratio

        print(f"  [RAG] Итого после реранкинга: {len(all_nodes)} чанков")

    return all_nodes

def build_file_context(nodes, notebook_id: str):
    """
    Каждый чанк получает свой порядковый номер [N].
    """
    paths = config.get_notebook_paths(notebook_id)
    images_dir = paths["images"]
    
    sources = []
    context_parts = []

    for i, node in enumerate(nodes, 1):
        meta = node.node.metadata
        fname = meta.get("file_name", "Неизвестный источник")
        img_path = meta.get("image_path", None)
        img_url = (
            f"/files/{notebook_id}/images/" + os.path.basename(img_path)
            if img_path and os.path.exists(img_path) else None
        )
        text = node.node.get_content()

        sources.append({
            "id": i,
            "file_name": fname,
            "text": text,
            "image_url": img_url,
            "page": meta.get("page"),
            "time": meta.get("start") or meta.get("time")
        })
        context_parts.append(f"[{i}] Файл «{fname}»:\n{text}")

    context_str = "\n\n" + ("=" * 40 + "\n\n").join(context_parts)
    return sources, context_str

def make_messages(query: str, context_str: str, answer_mode: str = None) -> list:
    """Формирует список сообщений для Chat API."""
    return [
        {
            "role": "system",
            "content": config.get_system_prompt(answer_mode)
        },
        {
            "role": "user",
            "content": f"Доступные источники:\n{context_str}\n\nВопрос пользователя: {query}"
        }
    ]

def make_prompt(query: str, context_str: str, thinking_mode: bool = False, max_tokens: int = 1024, answer_mode: str = None) -> str:
    return (
        config.get_system_prompt(answer_mode) + "\n"
        "ОТВЕЧАЙ СТРОГО С ИСПОЛЬЗОВАНИЕМ [N] ДЛЯ ССЫЛОК.\n\n"
        f"Доступные источники:\n{context_str}\n\n"
        f"Вопрос пользователя: {query}\n\n"
        "Твой ответ (используй СТРОГО формат [N] для ссылок):"
    )
