"""rerank.py — Реранкинг и фильтрация чанков для RAG."""

import logging
import statistics as _stats
import time as _time

import httpx

import config
from src.rag.state import _model_cache, _model_cache_lock

logger = logging.getLogger(__name__)

# HTTP-клиент для реранкинга
_async_rerank_http = httpx.AsyncClient(timeout=60)


async def _rerank_nodes(all_nodes, query: str):
    if not all_nodes or not config.rag.use_reranker:
        return all_nodes

    _original_scores = [n.score if hasattr(n, "score") else 0.0 for n in all_nodes]

    if len(all_nodes) > config.rag.rerank_pool:
        all_nodes.sort(
            key=lambda x: x.score if hasattr(x, "score") and x.score else 0,
            reverse=True,
        )
        all_nodes = all_nodes[: config.rag.rerank_pool]

    logger.info(f"  [RAG] Чанков для реранкинга: {len(all_nodes)}")

    reranker_name = config.rag.reranker_model
    reranker_available = config.validate_gguf_path(reranker_name)
    if not reranker_available:
        logger.warning(
            f"  [RAG] ⚠ Реранкер пропущен: неверный путь ({reranker_name}). "
            "Используются оригинальные RRF scores."
        )

    if reranker_available:
        async with _model_cache_lock:
            need_load = "reranker" not in _model_cache

        if need_load:
            logger.info(f"  [RAG] Загрузка GGUF реранкера: {reranker_name}")
            from src.gguf.server import get_gguf_embedding_url

            model_path = config.resolve_model_path(reranker_name)
            url = await get_gguf_embedding_url(model_path, is_reranker=True, n_parallel=1)
            async with _model_cache_lock:
                _model_cache["reranker"] = url

        async with _model_cache_lock:
            url = _model_cache.get("reranker")

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

            resp = await _async_rerank_http.post(
                f"{url}/v1/rerank",
                json={
                    "model": "gguf-reranker",
                    "query": query,
                    "documents": documents,
                    "top_n": len(documents),
                },
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])
            if not results:
                logger.debug("[RAG] Реранкер вернул пустой results")
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
            logger.warning(f"[RAG] Ошибка GGUF реранкера: {e}{err_body}")
            scores = [0] * len(all_nodes)

        if scores and max(scores) < 1e-6:
            logger.warning(
                f"  [RAG] ⚠️ GGUF реранкер выдал слишком низкие оценки "
                f"(max: {max(scores)}). Используются оригинальные RRF scores."
            )
            scores = _original_scores

        for node, score in zip(all_nodes, scores):
            node.score = float(score)

    all_nodes.sort(key=lambda x: x.score, reverse=True)
    all_nodes = all_nodes[: config.rag.final_top_n]

    return all_nodes


def _filter_chunks(all_nodes):
    if len(all_nodes) >= 4:
        score_vals = [n.score for n in all_nodes]
        median = _stats.median(score_vals)
        mad = _stats.median([abs(s - median) for s in score_vals]) or 0.05
        adaptive_thr = max(0.0, median - 2.0 * mad)
    else:
        adaptive_thr = config.rag.rerank_score_threshold

    above_threshold = [n for n in all_nodes if n.score >= adaptive_thr]
    min_chunks = min(config.rag.min_final_chunks, len(all_nodes))

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

    if config.rag.top_k_ratio > 0 and all_nodes:
        top_score = all_nodes[0].score
        ratio_thr = top_score * config.rag.top_k_ratio
        above_ratio = [n for n in all_nodes if n.score >= ratio_thr]
        if len(above_ratio) >= min_chunks and len(above_ratio) < len(all_nodes):
            logger.info(
                f"  [RAG] 🎯 Top-K ratio {config.rag.top_k_ratio:.2f} "
                f"(порог {ratio_thr:.3f} = {top_score:.3f}*{config.rag.top_k_ratio:.2f}): "
                f"убрано {len(all_nodes) - len(above_ratio)} чанков"
            )
            all_nodes = above_ratio

    return all_nodes
