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

def init_settings(max_tokens=1024):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Используем устройство для эмбеддингов: {device.upper()}")

    Settings.embed_model = HuggingFaceEmbedding(
        model_name=config.EMBEDDING_MODEL_NAME,
        device=device
    )

    Settings.llm = OpenAI(
        api_base=config.LM_STUDIO_URL,
        api_key="lm-studio",
        model="gpt-4o",
        temperature=0.1,
        max_tokens=max_tokens
    )

def get_vector_store(notebook_id: str):
    paths = config.get_notebook_paths(notebook_id)
    db_path = paths["chroma_db"]
    os.makedirs(db_path, exist_ok=True)
    db = chromadb.PersistentClient(path=db_path)
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
        "ВАЖНОЕ ПРАВИЛО ЦИТИРОВАНИЯ:\n"
        "- После КАЖДОГО утверждения ставь [N] — номер файла-источника.\n"
        "- Если ответа нет ни в одном источнике — скажи \"В документах этого нет\".\n\n"
        f"Доступные источники:\n{context_str}\n\n"
        f"Вопрос пользователя: {query}\n\n"
        "Твой ответ (каждое утверждение со ссылкой [N]):"
    )
