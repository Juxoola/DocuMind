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
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import BitsAndBytesConfig

logger = logging.getLogger(__name__)

_client_cache = {}
_model_cache = {}

def unload_rag_models(hard=True):
    """Выгрузка моделей RAG. Если hard=False, модели остаются в памяти (только очистка кэша)."""
    global _model_cache
    if not _model_cache:
        return

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
    
    if "embed_model" not in _model_cache:
        model_name = config.EMBEDDING_MODEL_NAME
        
        if model_name.lower().endswith('.gguf') or os.path.isabs(model_name) and os.path.exists(model_name):
            print(f"Инициализация GGUF эмбеддингов: {model_name}")
            from src.gguf_direct import get_gguf_embedding_url
            from llama_index.embeddings.openai import OpenAIEmbedding
            
            model_path = config.resolve_model_path(model_name)
            url = get_gguf_embedding_url(model_path)
            
            _model_cache["embed_model"] = OpenAIEmbedding(
                api_base=f"{url}/v1",
                api_key="sk-local",
                model="text-embedding-ada-002",
                timeout=120.0,
                # Сервер запущен с -c 2048 (= максимальный размер 1 чанка).
                # Каждый документ обрабатывается независимо, батч не складывает токены.
                # 16 параллельных запросов — безопасно и ускоряет индексацию в ~16×.
                embed_batch_size=16,
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
    "на том же языке для поиска справочной информации на основе следующего задания/вопроса.\n"
    "[ВАЖНО: Пиши ТОЛЬКО поисковые запросы, по одному запросу на строке. НЕ пиши никаких вступлений, пояснений и тегов <think>. Каждый запрос должен быть на новой отдельной строке.]\n"
    "Каждый запрос должен содержать ключевые термины, темы или формулы (исключай специфические числа из задания).\n"
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

def build_index(nodes, notebook_id: str):
    init_settings()
    vector_store = get_vector_store(notebook_id)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(nodes, storage_context=storage_context)

    # Перестройка BM25 в фоне — не блокирует завершение загрузки файла
    paths = config.get_notebook_paths(notebook_id)
    db_path = paths["chroma_db"]
    t = threading.Thread(
        target=_rebuild_bm25_bg,
        args=(notebook_id, db_path),
        daemon=True,
        name=f"bm25-{notebook_id}"
    )
    t.start()
    print(f"[RAG] BM25 перестраивается в фоне...")
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

    all_nodes = []

    # Определяем, доступен ли LLM для Query Expansion
    qe_llm = _get_qe_llm() if config.RAG_QUERY_EXPANSION else None
    if config.RAG_QUERY_EXPANSION:
        if qe_llm:
            print(f"  [RAG] Query Expansion включён (num_queries=3)")
        else:
            print(f"  [RAG] Query Expansion отключён (нет доступного LLM-сервера)")

    # Унифицированный поиск по всем файлам сразу (в 10-20 раз быстрее чем пофайловый)
    file_filter = MetadataFilters(
        filters=[MetadataFilter(key="file_name", value=allowed_files, operator=FilterOperator.IN)]
    )
    
    # Регулируем количество результатов: чем больше файлов, тем больше кандидатов берем для реранкера
    top_k_global = min(config.RAG_TOP_K_PER_FILE * len(allowed_files), config.RAG_RERANK_POOL)
    
    vector_retriever = index.as_retriever(
        similarity_top_k=top_k_global, 
        filters=file_filter
    )
    
    num_q = 3 if qe_llm else 1
    qprompt = _QUERY_GEN_PROMPT if qe_llm else None
    
    try:
        retrievers = [vector_retriever]
        if bm25_retriever:
            retrievers.append(bm25_retriever)
            
        # Запускаем QueryFusionRetriever если включен Query Expansion или доступен гибридный поиск
        if len(retrievers) > 1 or num_q > 1:
            fusion_retriever = QueryFusionRetriever(
                retrievers,
                similarity_top_k=top_k_global,
                num_queries=num_q,
                query_gen_prompt=qprompt,
                llm=qe_llm,
                use_async=False
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
            # Фильтруем результаты, так как слияние/BM25 может не поддерживать фильтрацию на уровне запроса
            all_nodes = [n for n in all_nodes if n.node.metadata.get("file_name") in allowed_files]
            
            if len(retrievers) > 1:
                label = f"Гибрид+QE({num_q})" if num_q > 1 else "Гибрид"
            else:
                label = f"Вектор+QE({num_q})"
            print(f"  [RAG] 🔍 {label} по {len(allowed_files)} файлам: {len(all_nodes)} фрагм.")
        else:
            all_nodes = vector_retriever.retrieve(query)
            print(f"  [RAG] 🔍 Вектор по {len(allowed_files)} файлам: {len(all_nodes)} фрагм.")
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
            documents = [n.node.get_content() for n in all_nodes]
            payload = {"model": "gguf-reranker", "query": query, "documents": documents, "top_n": len(documents)}
            
            try:
                import requests
                import time as _time
                _rerank_start = _time.time()

                # Мини-батчевый реранкинг.
                # llama.cpp /v1/rerank оценивает каждую пару (query + doc) НЕЗАВИСИМО —
                # контекст нужен только для ОДНОЙ пары, не для суммы всех документов.
                # Чанк до 2048 токенов + query ~50 = ~2100 токенов → вписывается в -c 4096.
                #
                # Мини-батчи (по 10 doc) нужны не из-за контекста, а чтобы:
                # 1. Держать timeout в разумных пределах (10 doc × 0.3с = 3с на батч)
                # 2. При ошибке терять только 10 score, а не все 35
                RERANK_MINI_BATCH_SIZE = 10
                scores = [0.0] * len(all_nodes)
                success = True

                for batch_start in range(0, len(documents), RERANK_MINI_BATCH_SIZE):
                    batch_docs = documents[batch_start: batch_start + RERANK_MINI_BATCH_SIZE]
                    mini_payload = {
                        "model": "gguf-reranker",
                        "query": query,
                        "documents": batch_docs,
                        "top_n": len(batch_docs)
                    }
                    try:
                        resp = requests.post(f"{url}/v1/rerank", json=mini_payload, timeout=60)
                        resp.raise_for_status()
                        results = resp.json().get("results", [])
                        if not results:
                            logger.debug(f"[RAG] Реранкер вернул пустой results для батча {batch_start}: {resp.text[:200]}")
                        for r in results:
                            # index — позиция ВНУТРИ мини-батча, нужно сдвинуть на batch_start
                            orig_idx = batch_start + r.get("index", 0)
                            if orig_idx < len(scores):
                                scores[orig_idx] = r.get("relevance_score", 0.0)
                    except Exception as mini_err:
                        err_body = ""
                        if hasattr(mini_err, 'response') and mini_err.response is not None:
                            err_body = f" body={mini_err.response.text[:200]}"
                        logger.warning(f"[RAG] Мини-батч {batch_start} не прошёл: {mini_err}{err_body}")
                        success = False

                elapsed_r = _time.time() - _rerank_start
                batches = (len(documents) + RERANK_MINI_BATCH_SIZE - 1) // RERANK_MINI_BATCH_SIZE
                if success:
                    print(f"  [RAG] ✅ Реранкинг: {len(documents)} doc / {batches} батчей за {elapsed_r:.2f}с")
                else:
                    print(f"  [RAG] ⚠️ Реранкинг частичный ({batches} батчей, {elapsed_r:.2f}с) — часть scores = 0.0")

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
            pairs = [[query, n.node.get_content()] for n in all_nodes]
            scores = model.predict(pairs)
        
        # Присваиваем скоры и сортируем
        for node, score in zip(all_nodes, scores):
            node.score = float(score)
            
        all_nodes.sort(key=lambda x: x.score, reverse=True)
        all_nodes = all_nodes[:config.RAG_FINAL_TOP_N]

        # Пороговая фильтрация: убираем нерелевантные чанки
        above_threshold = [n for n in all_nodes if n.score >= config.RERANK_SCORE_THRESHOLD]
        
        # Гарантируем минимум MIN_FINAL_CHUNKS (чтобы не терять контекст для сложных вопросов)
        min_chunks = min(config.MIN_FINAL_CHUNKS, len(all_nodes))
        
        if len(above_threshold) >= min_chunks:
            # Срезалось достаточно мусора, но осталось нужное количество
            if len(above_threshold) < len(all_nodes):
                print(f"  [RAG] 🎯 Порог {config.RERANK_SCORE_THRESHOLD}: убрано {len(all_nodes) - len(above_threshold)} нерелевантных чанков")
            all_nodes = above_threshold
        else:
            # Оказалось слишком мало хороших чанков — добираем до минимума из лучших ниже порога
            all_nodes = all_nodes[:min_chunks]
            print(f"  [RAG] ⚠️ Порог оставил <{min_chunks} чанков. Добавлено до {min_chunks} лучших (мин. score: {all_nodes[-1].score:.3f})")


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

def make_messages(query: str, context_str: str) -> list:
    """Формирует список сообщений для Chat API."""
    return [
        {
            "role": "system",
            "content": config.SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": f"Доступные источники:\n{context_str}\n\nВопрос пользователя: {query}"
        }
    ]

def make_prompt(query: str, context_str: str, thinking_mode: bool = False, max_tokens: int = 1024) -> str:
    return (
        config.SYSTEM_PROMPT + "\n"
        "ОТВЕЧАЙ СТРОГО С ИСПОЛЬЗОВАНИЕМ [N] ДЛЯ ССЫЛОК.\n\n"
        f"Доступные источники:\n{context_str}\n\n"
        f"Вопрос пользователя: {query}\n\n"
        "Твой ответ (используй СТРОГО формат [N] для ссылок):"
    )
