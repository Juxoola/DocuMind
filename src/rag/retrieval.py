"""RAG retrieval pipeline: Query Expansion, гибридный поиск (RRF), реранкинг."""

# Основной пайплайн поиска: генерация вариантов запроса (Query Expansion),
# гибридный поиск по векторному и BM25 индексам с RRF-фузией,
# реранкинг через GGUF-модель и адаптивная фильтрация по скорам.

import logging
import os
import time as _time

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
from src.rag.state import _model_cache, _rerank_session

logger = logging.getLogger(__name__)

# Промпт для генерации альтернативных поисковых запросов (Query Expansion).
# Просит LLM составить несколько коротких, конкретных запросов из терминов задания.
_QUERY_GEN_PROMPT = (
    "Ты — эксперт по поиску информации. Сформулируй ровно {num_queries} разных коротких поисковых запроса "
    "на том же языке для поиска справочной теории, правил и формул в учебных материалах на основе следующего задания/вопроса.\n"
    "[ВАЖНО: Пиши ТОЛЬКО готовые поисковые запросы, по одному на строке. НЕ пиши никаких вступлений, пояснений, номеров и тегов <think>.]\n"
    "[КРИТИЧЕСКИ ВАЖНО: Категорически запрещено писать общие мусорные фразы вроде 'решение задачи', 'пошаговое решение', 'сложная математика', 'пример по математике', 'решить тест'. "
    "Запросы должны состоять строго из конкретных научных терминов, названий теорем, правил или тем из учебника, найденных в задании (например: 'нечеткая логика функция принадлежности', 'лингвистическая переменная'). Исключай конкретные числа.]\n"
    "Задание/вопрос: {query}\n"
    "Поисковые запросы:"
)


# Создание LLM-клиента для Query Expansion. Проверяет доступность
# GGUF-сервера (через gguf_direct) или падает на LM Studio.
# Возвращает None, если LLM-сервер недоступен — QE отключается.
def _get_qe_llm():
    from src.gguf_direct import get_active_llm_url

    url = get_active_llm_url()
    if url:
        logger.debug(f"[QE] Используем GGUF LLM для Query Expansion: {url}")
    else:
        url = config.LM_STUDIO_URL
        logger.debug(f"[QE] GGUF LLM не найден, пробуем LM Studio: {url}")
    try:
        import requests as _req

        _req.get(url.replace("/v1", "").rstrip("/") + "/health", timeout=1)
    except Exception:
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


# Reciprocal Rank Fusion: объединяет результаты векторного и BM25 поиска,
# присваивая каждому документу вес 1/(k + rank) из каждого списка.
def _rrf_fuse(vector_results, bm25_results, k: int = 60):
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


# RRF для случая нескольких файлов: сначала фузия внутри каждого файла,
# затем межфайловая фузия объединённых списков.
def _rrf_fuse_across_files(file_results, k: int = 60):
    scores: dict = {}
    nodes_by_id: dict = {}

    for _fname, per_file_nodes in file_results:
        for rank, nws in enumerate(per_file_nodes, start=1):
            nid = nws.node.node_id
            scores[nid] = scores.get(nid, 0.0) + 1.0 / (k + rank)
            nodes_by_id[nid] = nws

    sorted_ids = sorted(scores.keys(), key=lambda i: scores[i], reverse=True)
    return [NodeWithScore(node=nodes_by_id[i].node, score=scores[i]) for i in sorted_ids]


# Главная точка входа в RAG-поиск. Выполняет:
# 1. Инициализацию моделей и векторного индекса
# 2. Загрузку/форсированную сборку BM25
# 3. Query Expansion (если включён)
# 4. Гибридный поиск (вектор + BM25) с RRF
# 5. Реранкинг через GGUF-модель
# 6. Адаптивную обрезку по скорам
def retrieve_nodes(query: str, notebook_id: str, allowed_files=None, max_tokens=1024):
    init_settings(max_tokens=max_tokens)
    vector_store = get_vector_store(notebook_id)
    index = VectorStoreIndex.from_vector_store(vector_store)

    if not allowed_files:
        return []

    paths = config.get_notebook_paths(notebook_id)
    bm25_dir = os.path.join(paths["base"], "bm25")

    bm25_retriever = None
    if os.path.exists(os.path.join(bm25_dir, "bm25_retriever_params.json")):
        try:
            bm25_retriever = BM25Retriever.from_persist_dir(bm25_dir)
        except Exception as e:
            logger.warning(f"Не удалось загрузить BM25: {e}")
    else:
        if not is_bm25_ready(notebook_id):
            logger.info(
                "  [RAG] BM25 отсутствует — форсирую синхронную пересборку для первого запроса"
            )
            flush_bm25_rebuild(notebook_id, db_path=paths["chroma_db"], wait=True, timeout=180)
            if os.path.exists(os.path.join(bm25_dir, "bm25_retriever_params.json")):
                try:
                    bm25_retriever = BM25Retriever.from_persist_dir(bm25_dir)
                except Exception as e:
                    logger.warning(f"Не удалось загрузить BM25 после flush: {e}")

    all_nodes = []

    qe_llm = _get_qe_llm() if config.RAG_QUERY_EXPANSION else None
    if config.RAG_QUERY_EXPANSION:
        if qe_llm:
            logger.info("  [RAG] Query Expansion включён (num_queries=3)")
        else:
            logger.info("  [RAG] Query Expansion отключён (нет доступного LLM-сервера)")

    if len(allowed_files) == 1:
        file_filter = MetadataFilters(
            filters=[
                MetadataFilter(
                    key="file_name",
                    value=allowed_files[0],
                    operator=FilterOperator.EQ,
                )
            ]
        )
    else:
        file_filter = MetadataFilters(
            filters=[
                MetadataFilter(
                    key="file_name",
                    value=allowed_files,
                    operator=FilterOperator.IN,
                )
            ]
        )

    top_k_per_file = config.RAG_TOP_K_PER_FILE
    vector_retriever = index.as_retriever(similarity_top_k=top_k_per_file, filters=file_filter)

    num_q = 3 if qe_llm else 1
    qprompt = _QUERY_GEN_PROMPT if qe_llm else None
    use_qe = num_q > 1

    try:
        if use_qe:
            per_file_retrievers = []
            for fname in allowed_files:
                ff = MetadataFilters(
                    filters=[
                        MetadataFilter(
                            key="file_name",
                            value=fname,
                            operator=FilterOperator.EQ,
                        )
                    ]
                )
                per_file_retrievers.append(
                    index.as_retriever(similarity_top_k=top_k_per_file, filters=ff)
                )
            if bm25_retriever:
                per_file_retrievers.append(bm25_retriever)

            fusion_retriever = QueryFusionRetriever(
                per_file_retrievers,
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
                        import re

                        for line in lines:
                            line = line.strip()
                            if not line:
                                continue
                            line = re.sub(r"^\d+[\.\)]\s*", "", line)
                            line = re.sub(r"^[-\*\+]\s*", "", line)
                            line = line.strip()
                            if line:
                                queries.append(line)

                        if queries:
                            embed = Settings.embed_model
                            orig_emb = embed.get_text_embedding(original_query)
                            valid = []
                            for q in queries:
                                q_emb = embed.get_text_embedding(q)
                                dot = sum(a * b for a, b in zip(orig_emb, q_emb))
                                norm_o = sum(a * a for a in orig_emb) ** 0.5
                                norm_q = sum(b * b for b in q_emb) ** 0.5
                                sim = dot / (norm_o * norm_q) if norm_o * norm_q > 0 else 0
                                if sim >= 0.6:
                                    valid.append(q)
                                else:
                                    logger.debug(f"[QE] Запрос отфильтрован по cos-sim={sim:.3f}: {q}")
                            if valid:
                                logger.info(f"  [RAG] QE валидация: {len(queries)}→{len(valid)} запросов (порог 0.6)")
                            queries = valid

                        return [QueryBundle(q) for q in queries[: fusion_retriever.num_queries - 1]]
                    except Exception as qe_err:
                        logger.warning(f"Ошибка генерации запросов в custom_get_queries: {qe_err}")
                        return []

                fusion_retriever._get_queries = custom_get_queries

            if num_q > 1 and qe_llm:
                try:
                    generated_bundles = fusion_retriever._get_queries(query)
                    logger.info("  [RAG] 🧠 Сгенерированные поисковые запросы (Query Expansion):")
                    for i, gq in enumerate(generated_bundles, 1):
                        logger.info(f"    {i}. {gq.query_str}")
                except Exception as qe_err:
                    logger.warning(
                        f"Не удалось получить сгенерированные запросы для лога: {qe_err}"
                    )

            all_nodes = fusion_retriever.retrieve(query)
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
                vec_results = vector_retriever.retrieve(query)
                bm25_results = bm25_retriever.retrieve(query) if bm25_retriever else []
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
                vec_results_all = vector_retriever.retrieve(query)[:fetch_k]
                bm25_all = bm25_retriever.retrieve(query) if bm25_retriever else []

                vec_by_file: dict = {}
                for n in vec_results_all:
                    fn = n.node.metadata.get("file_name", "")
                    if fn in allowed_files:
                        vec_by_file.setdefault(fn, []).append(n)
                vec_results = []
                for fn in allowed_files:
                    vec_results.extend(vec_by_file.get(fn, [])[:top_k_per_file])

                file_results = []
                for fname in allowed_files:
                    bm = [n for n in bm25_all if n.node.metadata.get("file_name") == fname][
                        :top_k_per_file
                    ]
                    vec = [n for n in vec_results if n.node.metadata.get("file_name") == fname]
                    fused = _rrf_fuse(vec, bm)
                    file_results.append((fname, fused))
                all_nodes = _rrf_fuse_across_files(file_results)
                if len(all_nodes) > config.RAG_RERANK_POOL:
                    all_nodes = all_nodes[: config.RAG_RERANK_POOL]
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
        logger.info(f"Ошибка унифицированного поиска: {e}")
        all_nodes = []

    # === Реранкинг через GGUF-модель ===
    # Загружает реранкер, отправляет все найденные чанки на переранжировку,
    # обновляет node.score ответом сервера.
    if all_nodes and config.USE_RERANKER:
        if len(all_nodes) > config.RAG_RERANK_POOL:
            all_nodes.sort(
                key=lambda x: x.score if hasattr(x, "score") and x.score else 0,
                reverse=True,
            )
            all_nodes = all_nodes[: config.RAG_RERANK_POOL]

        logger.info(f"  [RAG] Чанков для реранкинга: {len(all_nodes)}")

        reranker_name = config.RERANKER_MODEL_NAME
        if not (
            reranker_name.lower().endswith(".gguf")
            or (os.path.isabs(reranker_name) and os.path.exists(reranker_name))
        ):
            raise RuntimeError(
                "Поддерживаются только GGUF-модели реранкера. "
                "Укажите путь к .gguf файлу в config.RERANKER_MODEL_NAME.\n"
                f"Текущее значение: {reranker_name}"
            )

        if "reranker" not in _model_cache:
            logger.info(f"  [RAG] Загрузка GGUF реранкера: {reranker_name}")
            from src.gguf_direct import get_gguf_embedding_url

            model_path = config.resolve_model_path(reranker_name)
            url = get_gguf_embedding_url(
                model_path, is_reranker=True, n_parallel=config.EMBEDDING_N_PARALLEL
            )
            _model_cache["reranker"] = url

        url = _model_cache["reranker"]

        def _rerank_doc(nws):
            meta = nws.node.metadata or {}
            coord_parts = []
            if meta.get("file_name"):
                coord_parts.append(str(meta["file_name"]))
            if meta.get("page") not in (None, ""):
                coord_parts.append(f"стр.{meta['page']}")
            elif meta.get("time") not in (None, ""):
                coord_parts.append(f"@{meta['time']}")
            elif meta.get("start") not in (None, ""):
                coord_parts.append(f"@{meta['start']}")
            prefix = f"[{' '.join(coord_parts)}] " if coord_parts else ""
            return prefix + nws.node.get_content()

        documents = [_rerank_doc(n) for n in all_nodes]

        try:
            _rerank_start = _time.time()
            scores = [0.0] * len(all_nodes)

            resp = _rerank_session.post(
                f"{url}/v1/rerank",
                json={
                    "model": "gguf-reranker",
                    "query": query,
                    "documents": documents,
                    "top_n": len(documents),
                },
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
            logger.info(f"  [RAG] ✅ Реранкинг: {len(documents)} doc за {elapsed_r:.2f}с")

        except Exception as e:
            err_body = ""
            if hasattr(e, "response") and e.response is not None:
                err_body = f" body={e.response.text[:300]}"
            logger.info(f"[RAG] Ошибка GGUF реранкера: {e}{err_body}")
            scores = [0] * len(all_nodes)

        if scores and max(scores) < 1e-6:
            logger.warning(
                f"  [RAG] ⚠️ GGUF реранкер выдал слишком низкие оценки "
                f"(max: {max(scores)}). Используется оригинальный порядок поиска."
            )
            scores = [1.0 - (i * 0.01) for i in range(len(all_nodes))]

        for node, score in zip(all_nodes, scores):
            node.score = float(score)

        all_nodes.sort(key=lambda x: x.score, reverse=True)
        all_nodes = all_nodes[: config.RAG_FINAL_TOP_N]

        # === Адаптивная фильтрация по скорам ===
        # Отсекает чанки со скорами значительно ниже медианы (median - 2*MAD),
        # а затем по top-k ratio от максимального скора.
        import statistics as _stats

        if len(all_nodes) >= 4:
            score_vals = [n.score for n in all_nodes]
            median = _stats.median(score_vals)
            mad = _stats.median([abs(s - median) for s in score_vals]) or 0.05
            adaptive_thr = max(0.0, median - 2.0 * mad)
        else:
            adaptive_thr = config.RERANK_SCORE_THRESHOLD

        above_threshold = [n for n in all_nodes if n.score >= adaptive_thr]
        min_chunks = min(config.MIN_FINAL_CHUNKS, len(all_nodes))

        if len(above_threshold) >= min_chunks:
            if len(above_threshold) < len(all_nodes):
                logger.info(
                    f"  [RAG] 🎯 Адаптивный порог {adaptive_thr:.3f} (median-MAD): "
                    f"убрано {len(all_nodes) - len(above_threshold)} чанков"
                )
            all_nodes = above_threshold
        else:
            all_nodes = all_nodes[:min_chunks]
            logger.warning(
                f"  [RAG] ⚠️ Адаптивный порог {adaptive_thr:.3f} оставил "
                f"<{min_chunks} чанков. Добавлено до {min_chunks} лучших "
                f"(мин. score: {all_nodes[-1].score:.3f})"
            )

        if config.RAG_TOP_K_RATIO > 0 and all_nodes:
            top_score = all_nodes[0].score
            ratio_thr = top_score * config.RAG_TOP_K_RATIO
            above_ratio = [n for n in all_nodes if n.score >= ratio_thr]
            if len(above_ratio) >= min_chunks and len(above_ratio) < len(all_nodes):
                logger.info(
                    f"  [RAG] 🎯 Top-K ratio {config.RAG_TOP_K_RATIO:.2f} "
                    f"(порог {ratio_thr:.3f} = {top_score:.3f}*{config.RAG_TOP_K_RATIO:.2f}): "
                    f"убрано {len(all_nodes) - len(above_ratio)} чанков"
                )
                all_nodes = above_ratio

        logger.info(f"  [RAG] Итого после реранкинга: {len(all_nodes)} чанков")

    return all_nodes
