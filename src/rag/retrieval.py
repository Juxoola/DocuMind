"""Гибридный поиск RAG: векторный + BM25 со слиянием по RRF и опциональным реранкингом."""

import asyncio
import logging
import os
import re
import time as _time

import aiofiles.os
import httpx
import numpy as np
from llama_index.core import QueryBundle, Settings, VectorStoreIndex
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores.types import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)
from llama_index.llms.openai import OpenAI
from llama_index.retrievers.bm25 import BM25Retriever

import config
from src.rag.bm25 import flush_bm25_rebuild, is_bm25_ready
from src.rag.indexing import get_vector_store
from src.rag.models import init_settings
from src.rag.rerank import _filter_chunks, _rerank_nodes
from src.rag.state import (
    _INDEX_CACHE_MAXSIZE,
    _index_cache,
    _index_cache_lock,
)

logger = logging.getLogger(__name__)

# HTTP-клиент для реранкинга, кэш здоровья LLM и предкомпилированные регулярки QE
# ── HTTP-клиент для реранкинга, кэш здоровья LLM и предкомпилированные регулярки QE ──
_async_rerank_http = httpx.AsyncClient(timeout=60)

_qe_health_cache: dict[str, tuple[bool, float]] = {}
_qe_health_cache_lock = asyncio.Lock()
_QE_HEALTH_TTL = 120.0

_QE_RE_NUM = re.compile(r"^\d+[\.\)]\s*")
_QE_RE_BULLET = re.compile(r"^[-\*\+]\s*")

# Кэш результатов QE: {hash(query): (bundles, timestamp)}
_qe_result_cache: dict[str, tuple[list, float]] = {}
_QE_RESULT_TTL = 60.0


def _empty_list():
    return []


async def _is_llm_healthy(url: str) -> bool:
    now = _time.time()
    async with _qe_health_cache_lock:
        cached = _qe_health_cache.get(url)
    if cached and (now - cached[1]) < _QE_HEALTH_TTL:
        return cached[0]
    try:
        health_url = url.replace("/v1", "").rstrip("/") + "/health"
        resp = await _async_rerank_http.get(health_url, timeout=1)
        resp.raise_for_status()
        result = True
    except Exception:
        result = False
    async with _qe_health_cache_lock:
        _qe_health_cache[url] = (result, now)
    return result


# Обёртка над BM25Retriever: фильтрация по списку разрешённых файлов перед RRF-слиянием
# ── Обёртка над BM25Retriever: фильтрация по списку разрешённых файлов ──
class _FilteredBM25:
    def __init__(self, base, allowed_files):
        self._base = base
        self._allowed = set(allowed_files)

    def retrieve(self, query, **kwargs):
        results = self._base.retrieve(query, **kwargs)
        return [r for r in results if r.node.metadata.get("file_name") in self._allowed]


# ── Инвалидация кэша векторного индекса ──
async def invalidate_index_cache(notebook_id: str = None):
    async with _index_cache_lock:
        if notebook_id:
            _index_cache.pop(notebook_id, None)
        else:
            _index_cache.clear()


def _file_filter(file_names: str | list[str]):
    if isinstance(file_names, str):
        file_names = [file_names]
    if len(file_names) == 1:
        return MetadataFilters(
            filters=[
                MetadataFilter(
                    key="file_name",
                    value=file_names[0],
                    operator=FilterOperator.EQ,
                )
            ]
        )
    return MetadataFilters(
        filters=[
            MetadataFilter(
                key="file_name",
                value=file_names,
                operator=FilterOperator.IN,
            )
        ]
    )


# Промпт Query Expansion: LLM генерирует альтернативные поисковые запросы для лучшего покрытия
# ── Промпт Query Expansion: LLM генерирует альтернативные поисковые запросы ──
_QUERY_GEN_PROMPT = (
    "Ты — эксперт по поиску информации. Сформулируй ровно {num_queries} разных коротких поисковых запроса "
    "на том же языке для поиска справочной теории, правил и формул в учебных материалах на основе следующего задания/вопроса.\n"
    "[ВАЖНО: Пиши ТОЛЬКО готовые поисковые запросы, по одному на строке. НЕ пиши никаких вступлений, пояснений, номеров и тегов <think>.]\n"
    "[КРИТИЧЕСКИ ВАЖНО: Категорически запрещено писать общие мусорные фразы вроде 'решение задачи', 'пошаговое решение', 'сложная математика', 'пример по математике', 'решить тест'. "
    "Запросы должны состоять строго из конкретных научных терминов, названий теорем, правил или тем из учебника, найденных в задании (например: 'нечеткая логика функция принадлежности', 'лингвистическая переменная'). Исключай конкретные числа.]\n"
    "Задание/вопрос: {query}\n"
    "Поисковые запросы:"
)


# ── Получение LLM для Query Expansion с проверкой здоровья ──
async def _get_qe_llm():
    from src.gguf.server import get_active_llm_url

    url = await get_active_llm_url()
    if url:
        logger.debug(f"[QE] Используем GGUF LLM для Query Expansion: {url}")
    else:
        url = config.LM_STUDIO_URL
        logger.debug(f"[QE] GGUF LLM не найден, пробуем LM Studio: {url}")
    if not await _is_llm_healthy(url):
        logger.debug("[QE] LLM-сервер недоступен, Query Expansion пропускается")
        return None

    return OpenAI(
        api_base=url if url.endswith("/v1") else f"{url}/v1",
        api_key=config.LLM_DEFAULT_API_KEY,
        model=config.LLM_DEFAULT_MODEL,
        temperature=0.3,
        max_tokens=512,
        timeout=20.0,
        max_retries=0,
        additional_kwargs={
            "extra_body": {
                "thinking_budget": 0,
                "thinking_budget_tokens": 0,
                "chat_template_kwargs": {"enable_thinking": False},
            }
        },
    )


# ── Reciprocal Rank Fusion: слияние результатов поиска ──
def _rrf_fuse(*result_lists, k: int = None):
    scores: dict = {}
    nodes_by_id: dict = {}
    k = k or config.rag.rrf_k

    for ranking in result_lists:
        for rank, nws in enumerate(ranking, start=1):
            nid = nws.node.node_id
            scores[nid] = scores.get(nid, 0.0) + 1.0 / (k + rank)
            nodes_by_id[nid] = nws

    sorted_ids = sorted(scores.keys(), key=lambda i: scores[i], reverse=True)
    return [NodeWithScore(node=nodes_by_id[i].node, score=scores[i]) for i in sorted_ids]


async def _load_bm25_retriever(notebook_id: str):
    paths = config.get_notebook_paths(notebook_id)
    bm25_dir = os.path.join(paths["base"], "bm25")

    bm25_retriever = None
    if await aiofiles.os.path.exists(os.path.join(bm25_dir, "retriever.json")):
        try:
            bm25_retriever = await asyncio.to_thread(BM25Retriever.from_persist_dir, bm25_dir)
        except Exception as e:
            logger.warning(f"Не удалось загрузить BM25: {e}")
    else:
        if not await is_bm25_ready(notebook_id):
            logger.info(
                "  [RAG] BM25 отсутствует — форсирую синхронную пересборку для первого запроса"
            )
            await flush_bm25_rebuild(
                notebook_id, db_path=paths["chroma_db"], wait=True, timeout=180
            )
            if await aiofiles.os.path.exists(os.path.join(bm25_dir, "retriever.json")):
                try:
                    bm25_retriever = await asyncio.to_thread(
                        BM25Retriever.from_persist_dir, bm25_dir
                    )
                except Exception as e:
                    logger.warning(f"Не удалось загрузить BM25 после flush: {e}")

    return bm25_retriever


# Основная логика гибридного поиска: векторный + BM25 с QE и RRF-слиянием
async def _hybrid_search(index, query: str, allowed_files, bm25_retriever, qe_llm):
    all_nodes = []

    if config.rag.query_expansion:
        if qe_llm:
            logger.info("  [RAG] Query Expansion включён (num_queries=3)")
        else:
            logger.info("  [RAG] Query Expansion отключён (нет доступного LLM-сервера)")

    if len(allowed_files) == 1:
        file_filter = _file_filter(allowed_files[0])
    else:
        file_filter = _file_filter(allowed_files)

    top_k_per_file = config.rag.top_k_per_file
    vector_retriever = index.as_retriever(similarity_top_k=top_k_per_file, filters=file_filter)

    num_q = 3 if qe_llm else 1
    qprompt = _QUERY_GEN_PROMPT if qe_llm else None
    use_qe = num_q > 1

    try:
        if use_qe:
            if len(allowed_files) > 1:
                all_filter = MetadataFilters(
                    filters=[
                        MetadataFilter(
                            key="file_name", value=allowed_files, operator=FilterOperator.IN
                        )
                    ]
                )
            else:
                all_filter = _file_filter(allowed_files[0])
            retrievers = [
                index.as_retriever(
                    similarity_top_k=top_k_per_file * len(allowed_files), filters=all_filter
                )
            ]
            if bm25_retriever:
                retrievers.append(_FilteredBM25(bm25_retriever, allowed_files))

            fusion_retriever = QueryFusionRetriever(
                retrievers,
                similarity_top_k=top_k_per_file * len(allowed_files),
                num_queries=num_q,
                query_gen_prompt=qprompt,
                llm=qe_llm,
                use_async=False,
                mode="reciprocal_rerank",
            )

            if num_q > 1 and qe_llm:

                def custom_get_queries(original_query: str):
                    try:
                        prompt_str = fusion_retriever.query_gen_prompt.format(
                            num_queries=fusion_retriever.num_queries - 1,
                            query=original_query,
                        )
                        response = fusion_retriever._llm.complete(prompt_str)
                        text = response.text or ""

                        if "<think>" in text:
                            parts = text.split("</think>")
                            text = parts[-1]
                        text = text.replace("<think>", "").replace("</think>", "")

                        lines = text.strip("`").split("\n")
                        queries = []

                        for line in lines:
                            line = line.strip()
                            if not line:
                                continue
                            line = _QE_RE_NUM.sub("", line)
                            line = _QE_RE_BULLET.sub("", line)
                            line = line.strip()
                            if line:
                                queries.append(line)

                        if queries:
                            embed = Settings.embed_model
                            all_texts = [original_query, *queries]
                            try:
                                all_embs = embed.get_text_embedding_batch(all_texts)
                            except AttributeError:
                                all_embs = [embed.get_text_embedding(t) for t in all_texts]
                            orig_emb = np.asarray(all_embs[0], dtype=np.float32)
                            orig_norm = np.linalg.norm(orig_emb)
                            valid = []
                            for i, q in enumerate(queries):
                                q_emb = np.asarray(all_embs[i + 1], dtype=np.float32)
                                norm_product = orig_norm * np.linalg.norm(q_emb)
                                sim = (
                                    float(np.dot(orig_emb, q_emb) / norm_product)
                                    if norm_product > 0
                                    else 0.0
                                )
                                if sim >= 0.6:
                                    valid.append(q)
                                else:
                                    logger.debug(
                                        f"[QE] Запрос отфильтрован по cos-sim={sim:.3f}: {q}"
                                    )
                            if valid:
                                logger.info(
                                    f"  [RAG] QE валидация: {len(queries)}→{len(valid)} запросов (порог 0.6)"
                                )
                            queries = valid

                        return [QueryBundle(q) for q in queries[: fusion_retriever.num_queries - 1]]
                    except Exception as qe_err:
                        logger.warning(f"Ошибка генерации запросов в custom_get_queries: {qe_err}")
                        return []

                fusion_retriever._get_queries = custom_get_queries

            if num_q > 1 and qe_llm:
                import hashlib as _hashlib

                _qe_cache_key = _hashlib.md5(query.encode()).hexdigest()
                _now = _time.time()
                async with _qe_health_cache_lock:
                    cached_qe = _qe_result_cache.get(_qe_cache_key)
                if cached_qe and (_now - cached_qe[1]) < _QE_RESULT_TTL:
                    generated_bundles = cached_qe[0]
                    logger.info("  [RAG] 🧠 QE из кэша (60с TTL):")
                else:
                    try:
                        generated_bundles = await asyncio.to_thread(
                            fusion_retriever._get_queries, query
                        )
                        async with _qe_health_cache_lock:
                            _qe_result_cache[_qe_cache_key] = (generated_bundles, _now)
                        logger.info(
                            "  [RAG] 🧠 Сгенерированные поисковые запросы (Query Expansion):"
                        )
                    except Exception as qe_err:
                        logger.warning(
                            f"Не удалось получить сгенерированные запросы для лога: {qe_err}"
                        )
                        generated_bundles = []
                for i, gq in enumerate(generated_bundles, 1):
                    logger.info(f"    {i}. {gq.query_str}")

            all_nodes = await asyncio.to_thread(fusion_retriever.retrieve, query)
            all_nodes = [n for n in all_nodes if n.node.metadata.get("file_name") in allowed_files]

            if bm25_retriever:
                label = f"Per-file Гибрид+QE({num_q})"
            else:
                label = f"Per-file Вектор+QE({num_q})"
            logger.info(
                f"  [RAG] 🔍 {label} по {len(allowed_files)} файлам: {len(all_nodes)} фрагм."
            )
        else:
            if len(allowed_files) == 1:
                vec_results, bm25_results_raw = await asyncio.gather(
                    asyncio.to_thread(vector_retriever.retrieve, query),
                    asyncio.to_thread(bm25_retriever.retrieve, query)
                    if bm25_retriever
                    else _empty_list(),
                )
                bm25_results = bm25_results_raw
                bm25_results = [
                    n for n in bm25_results if n.node.metadata.get("file_name") == allowed_files[0]
                ][:top_k_per_file]

                all_nodes = _rrf_fuse(vec_results, bm25_results)
                if bm25_retriever:
                    logger.info(
                        f"  [RAG] 🔍 Гибрид (RRF, vector+BM25) по 1 файлу: {len(all_nodes)} фрагм."
                    )
                else:
                    logger.info(f"  [RAG] 🔍 Вектор по 1 файлу: {len(all_nodes)} фрагм.")
            else:
                fetch_k = top_k_per_file * len(allowed_files)
                vec_results_all_raw, bm25_all_raw = await asyncio.gather(
                    asyncio.to_thread(vector_retriever.retrieve, query),
                    asyncio.to_thread(bm25_retriever.retrieve, query)
                    if bm25_retriever
                    else _empty_list(),
                )
                vec_results_all = vec_results_all_raw[:fetch_k]
                bm25_all = bm25_all_raw

                vec_by_file: dict = {}
                for n in vec_results_all:
                    fn = n.node.metadata.get("file_name", "")
                    if fn in allowed_files:
                        vec_by_file.setdefault(fn, []).append(n)
                vec_results = []
                for fn in allowed_files:
                    vec_results.extend(vec_by_file.get(fn, [])[:top_k_per_file])

                all_vec = []
                all_bm = []
                for fname in allowed_files:
                    all_vec.extend(
                        n for n in vec_results if n.node.metadata.get("file_name") == fname
                    )
                    per_bm = [n for n in bm25_all if n.node.metadata.get("file_name") == fname][
                        :top_k_per_file
                    ]
                    all_bm.extend(per_bm)
                all_nodes = _rrf_fuse(all_vec, all_bm)
                if len(all_nodes) > config.rag.rerank_pool:
                    all_nodes = all_nodes[: config.rag.rerank_pool]
                if bm25_retriever:
                    logger.info(
                        f"  [RAG] 🔍 Per-file Гибрид (RRF, vector+BM25) "
                        f"по {len(allowed_files)} файлам: {len(all_nodes)} фрагм."
                    )
                else:
                    logger.info(
                        f"  [RAG] 🔍 Per-file Вектор по {len(allowed_files)} файлам: {len(all_nodes)} фрагм."
                    )

    except Exception as e:
        logger.warning(f"Ошибка унифицированного поиска: {e}")
        all_nodes = []

    return all_nodes


# Точка входа: полный пайплайн поиска (индекс → гибридный поиск → реранкинг → фильтрация)
async def retrieve_nodes(query: str, notebook_id: str, allowed_files=None, max_tokens=1024):
    await init_settings(max_tokens=max_tokens)
    vector_store = await get_vector_store(notebook_id)

    async with _index_cache_lock:
        index = _index_cache.get(notebook_id)
        if index is not None:
            _index_cache.move_to_end(notebook_id)

    if index is None:
        new_index = await asyncio.to_thread(VectorStoreIndex.from_vector_store, vector_store)
        async with _index_cache_lock:
            index = _index_cache.get(notebook_id)
            if index is None:
                if len(_index_cache) >= _INDEX_CACHE_MAXSIZE:
                    _index_cache.popitem(last=False)
                _index_cache[notebook_id] = new_index
                _index_cache.move_to_end(notebook_id)
                index = new_index
            else:
                _index_cache.move_to_end(notebook_id)

    if not allowed_files:
        return []

    bm25_retriever = await _load_bm25_retriever(notebook_id)

    qe_llm = await _get_qe_llm() if config.rag.query_expansion else None

    all_nodes = await _hybrid_search(index, query, allowed_files, bm25_retriever, qe_llm)

    all_nodes = await _rerank_nodes(all_nodes, query)
    if all_nodes and config.rag.use_reranker:
        all_nodes = _filter_chunks(all_nodes)
        logger.info(f"  [RAG] Итого после реранкинга: {len(all_nodes)} чанков")

    return all_nodes
