import os
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

_client_cache = {}
_model_cache = {}

def preload_all_models():
    """Предзагрузка всех тяжелых моделей для ускорения работы."""
    print("[RAG] Предзагрузка моделей...")
    init_settings()
    # Загружаем реранкер, если он указан в конфиге
    if config.RERANKER_MODEL_NAME:
        # Эмуляция вызова для инициализации кэша
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if "reranker" not in _model_cache:
            print(f"  [RAG] Предзагрузка реранкера: {config.RERANKER_MODEL_NAME}")
            from sentence_transformers import CrossEncoder
            _model_cache["reranker"] = CrossEncoder(config.RERANKER_MODEL_NAME, device=device)
    
    # Загружаем GGUF LLM для зрения, если есть пути (из конфига или сохраненные)
    last = config.load_last_model()
    gguf_path = last.get("gguf") or os.getenv("DEFAULT_GGUF_MODEL")
    mmproj_path = last.get("mmproj") or os.getenv("DEFAULT_MMPROJ_MODEL")
    
    if gguf_path and mmproj_path:
        if "vision_llm" not in _model_cache:
            g_path = config.resolve_model_path(gguf_path)
            m_path = config.resolve_model_path(mmproj_path)
            if os.path.exists(g_path) and os.path.exists(m_path):
                print(f"  [RAG] Предзагрузка Vision LLM: {os.path.basename(g_path)}")
                from llama_cpp import Llama
                _model_cache["vision_llm"] = Llama(
                    model_path=g_path,
                    chat_format="chatml",
                    clip_model_path=m_path,
                    n_ctx=4096, n_gpu_layers=-1, verbose=False, type_k=2, type_v=2
                )
            
    print("[RAG] Все модели загружены.")

def init_settings(max_tokens=1024):
    global _model_cache
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if "embed_model" not in _model_cache:
        print(f"Инициализация эмбеддингов: {device.upper()} (Quant: {config.QUANTIZATION})")
        model_kwargs = {"trust_remote_code": True}
        
        if config.QUANTIZATION == "4bit":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, 
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4"
            )
        elif config.QUANTIZATION == "int8":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            model_kwargs["torch_dtype"] = torch.float16

        _model_cache["embed_model"] = HuggingFaceEmbedding(
            model_name=config.EMBEDDING_MODEL_NAME,
            device=device,
            model_kwargs=model_kwargs
        )
    
    Settings.embed_model = _model_cache["embed_model"]

    Settings.llm = OpenAI(
        api_base=config.LM_STUDIO_URL,
        api_key="lm-studio",
        model="gpt-4o",
        temperature=0.1,
        max_tokens=max_tokens
    )

def close_all_clients():
    """Явно закрывает все открытые клиенты ChromaDB для снятия блокировок файлов."""
    global _client_cache
    for path, client in _client_cache.items():
        try:
            client.close()
        except:
            pass
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
    return index

def retrieve_nodes(query: str, notebook_id: str, allowed_files=None, max_tokens=1024):
    """
    Для каждого выбранного файла выполняем отдельный поиск топ-3 чанков.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    init_settings(max_tokens=max_tokens)
    vector_store = get_vector_store(notebook_id)
    index = VectorStoreIndex.from_vector_store(vector_store)

    if not allowed_files:
        return []

    all_nodes = []
    for fname in allowed_files:
        file_filter = MetadataFilters(
            filters=[MetadataFilter(key="file_name", value=fname, operator=FilterOperator.EQ)]
        )
        retriever = index.as_retriever(similarity_top_k=5, filters=file_filter)
        try:
            nodes = retriever.retrieve(query)
            all_nodes.extend(nodes)
        except Exception as e:
            print(f"Ошибка при поиске в {fname}: {e}")

    # Переранжирование (Reranking) для максимальной точности
    if all_nodes:
        print(f"  [RAG] Начальное количество чанков: {len(all_nodes)}")
        
        if "reranker" not in _model_cache:
            print(f"  [RAG] Загрузка реранкера: {config.RERANKER_MODEL_NAME} ({config.QUANTIZATION})")
            rerank_kwargs = {"trust_remote_code": True}
            if config.QUANTIZATION == "4bit":
                rerank_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, 
                    bnb_4bit_compute_dtype=torch.float16
                )
            elif config.QUANTIZATION == "int8":
                rerank_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            else:
                rerank_kwargs["torch_dtype"] = torch.float16

            from sentence_transformers import CrossEncoder
            _model_cache["reranker"] = CrossEncoder(config.RERANKER_MODEL_NAME, device=device, model_kwargs=rerank_kwargs)
        
        model = _model_cache["reranker"]
        
        # Подготовка пар для реранкинга
        pairs = [[query, n.node.get_content()] for n in all_nodes]
        scores = model.predict(pairs)
        
        # Присваиваем скоры и сортируем
        for node, score in zip(all_nodes, scores):
            node.score = float(score)
            
        all_nodes.sort(key=lambda x: x.score, reverse=True)
        all_nodes = all_nodes[:15]
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

def make_prompt(query: str, context_str: str) -> str:
    return (
        "Ты — умный аналитик. ОТВЕЧАЙ СТРОГО НА РУССКОМ ЯЗЫКЕ.\n"
        "Используй Markdown для форматирования (заголовки, списки, **жирный**).\n\n"
        "ПРАВИЛА ОТВЕТА:\n"
        "1. Всегда отвечай на вопрос СРАЗУ, в самом начале ответа.\n"
        "2. Если вопрос содержит варианты ответа (тест) — СНАЧАЛА напиши правильный вариант (букву и текст).\n"
        "3. После прямого ответа приведи подробное объяснение на основе источников.\n\n"
        "ВАЖНОЕ ПРАВИЛО ЦИТИРОВАНИЯ (КРИТИЧЕСКИ ДЛЯ СИСТЕМЫ):\n"
        "- Каждое утверждение ДОЛЖНО завершаться ссылкой в формате [N], где N - номер источника.\n"
        "- ПРИМЕР: «Поток нельзя запустить дважды [1].»\n"
        "- ОШИБКА: «Поток нельзя запустить дважды 1.» (так писать ЗАПРЕЩЕНО)\n"
        "- НИКОГДА не пиши цифру источника без квадратных скобок.\n"
        "- Если одно утверждение основано на нескольких источниках, пиши [1, 2].\n"
        "- Все формулы пиши внутри $...$ или $$...$$.\n"
        "- Если ответа нет в источниках — скажи \"В документах этого нет\".\n\n"
        "ОТВЕЧАЙ СТРОГО С ИСПОЛЬЗОВАНИЕМ [N] ДЛЯ ССЫЛОК.\n\n"
        f"Доступные источники:\n{context_str}\n\n"
        f"Вопрос пользователя: {query}\n\n"
        "Твой ответ (используй СТРОГО формат [N] для ссылок):"
    )
