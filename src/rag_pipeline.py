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

def build_index(nodes, notebook_id: str):
    init_settings()
    vector_store = get_vector_store(notebook_id)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(nodes, storage_context=storage_context)
    
    # Перестройка индекса BM25
    paths = config.get_notebook_paths(notebook_id)
    bm25_dir = os.path.join(paths["base"], "bm25")
    os.makedirs(bm25_dir, exist_ok=True)
    
    try:
        collection = vector_store._collection
        result = collection.get()
        all_nodes = []
        for i, doc_id in enumerate(result['ids']):
            text = result['documents'][i]
            meta = result['metadatas'][i] or {}
            all_nodes.append(TextNode(text=text, id_=doc_id, metadata=meta))
        
        if all_nodes:
            from llama_index.retrievers.bm25 import BM25Retriever
            bm25_retriever = BM25Retriever.from_defaults(nodes=all_nodes, similarity_top_k=config.RAG_TOP_K_PER_FILE, language="russian")
            bm25_retriever.persist(bm25_dir)
            print(f"[RAG] Обновлен BM25 индекс для {len(all_nodes)} узлов.")
    except Exception as e:
        logger.warning(f"Ошибка при сборке BM25: {e}")
        
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
    for fname in allowed_files:
        file_filter = MetadataFilters(
            filters=[MetadataFilter(key="file_name", value=fname, operator=FilterOperator.EQ)]
        )
        
        vector_retriever = index.as_retriever(similarity_top_k=config.RAG_TOP_K_PER_FILE, filters=file_filter)
        
        if bm25_retriever:
            # Обновляем фильтр для BM25 вручную, так как он не поддерживает MetadataFilters напрямую
            # Мы будем фильтровать результаты после выдачи
            fusion_retriever = QueryFusionRetriever(
                [vector_retriever, bm25_retriever],
                similarity_top_k=config.RAG_TOP_K_PER_FILE,
                num_queries=1,  # Без автоматического Query Expansion
                use_async=False
            )
            try:
                # Получаем сырые ноды и фильтруем по файлу
                nodes = fusion_retriever.retrieve(query)
                filtered_nodes = [n for n in nodes if n.node.metadata.get("file_name") == fname]
                print(f"  [RAG] 🔍 Гибридный поиск (Вектор + BM25) по файлу {fname}: найдено {len(filtered_nodes)} фрагментов")
                all_nodes.extend(filtered_nodes)
            except Exception as e:
                print(f"Ошибка при гибридном поиске в {fname}: {e}")
        else:
            try:
                nodes = vector_retriever.retrieve(query)
                all_nodes.extend(nodes)
            except Exception as e:
                print(f"Ошибка при векторном поиске в {fname}: {e}")

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
                resp = requests.post(f"{url}/v1/rerank", json=payload, timeout=60)
                resp.raise_for_status()
                results = resp.json().get("results", [])
                
                # Применяем скоры из GGUF-сервера
                scores = [0] * len(all_nodes)
                for item in results:
                    scores[item["index"]] = item["relevance_score"]
            except Exception as e:
                error_details = ""
                if hasattr(e, 'response') and e.response is not None:
                    error_details = f" Details: {e.response.text}"
                print(f"[RAG] Ошибка GGUF реранкера: {e}{error_details}")
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
        all_nodes = all_nodes[:config.RAG_FINAL_TOP_N] # Финальный топ-N для контекста
        print(f"  [RAG] После переранжирования: {len(all_nodes)}")

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
