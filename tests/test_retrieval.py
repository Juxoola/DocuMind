"""Тесты src/rag/retrieval.py: _rrf_fuse, _file_filter, _filter_chunks, invalidate_index_cache, _FilteredBM25."""

import asyncio
import os
import sys
import threading
import time as _time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Fake-объекты для имитации llama_index ──


class FakeTextNode:
    def __init__(self, text="", id_=None, metadata=None):
        self.text = text
        self.node_id = id_ or hash(text)
        self.metadata = metadata or {}


class FakeNodeWithScore:
    def __init__(self, node=None, score=0.0):
        self.node = node or FakeTextNode()
        self.score = score


def _nws(text, score=0.0, id_=None):
    """Хелпер:快速创建 FakeNodeWithScore."""
    return FakeNodeWithScore(node=FakeTextNode(text=text, id_=id_), score=score)


@pytest.fixture(autouse=True)
def mock_heavy_deps():
    """Мокаем все тяжёлые зависимости llama_index, chromadb, torch, httpx."""
    mock_schema = MagicMock()
    mock_schema.TextNode = FakeTextNode
    mock_schema.NodeWithScore = FakeNodeWithScore

    mock_core = MagicMock()
    mock_core.schema = mock_schema

    mocks = {
        "llama_index": MagicMock(),
        "llama_index.core": mock_core,
        "llama_index.core.schema": mock_schema,
        "llama_index.core.settings": MagicMock(),
        "llama_index.core.storage": MagicMock(),
        "llama_index.core.storage.storage_context": MagicMock(),
        "llama_index.core.vector_stores": MagicMock(),
        "llama_index.core.vector_stores.types": MagicMock(),
        "llama_index.core.node_parser": MagicMock(),
        "llama_index.core.retrievers": MagicMock(),
        "llama_index.llms.openai": MagicMock(),
        "llama_index.embeddings.openai": MagicMock(),
        "llama_index.vector_stores.chroma": MagicMock(),
        "llama_index.retrievers.bm25": MagicMock(),
        "chromadb": MagicMock(),
        "torch": MagicMock(),
        "httpx": MagicMock(),
        "numpy": MagicMock(),
    }
    with patch.dict(sys.modules, mocks, clear=False):
        yield


# ── _rrf_fuse ──


class TestRrfFuse:
    """RRF — различные комбинации входных данных."""

    def test_empty_lists(self):
        from src.rag.retrieval import _rrf_fuse

        result = _rrf_fuse()
        assert result == []

    def test_single_list_preserves_order(self):
        from src.rag.retrieval import _rrf_fuse

        a = _nws("a", 0.9)
        b = _nws("b", 0.7)
        c = _nws("c", 0.5)
        result = _rrf_fuse([a, b, c])
        ids = [r.node.node_id for r in result]
        assert ids == [a.node.node_id, b.node.node_id, c.node.node_id]

    def test_two_lists_overlap(self):
        from src.rag.retrieval import _rrf_fuse

        a = _nws("a")
        b = _nws("b")
        c = _nws("c")
        result = _rrf_fuse([a, b, c], [a, b])
        ids = [r.node.node_id for r in result]
        assert ids[0] in (a.node.node_id, b.node.node_id)
        assert ids[-1] == c.node.node_id

    def test_disjoint_lists_interleave(self):
        from src.rag.retrieval import _rrf_fuse

        a = _nws("a")
        b = _nws("b")
        result = _rrf_fuse([a], [b])
        assert len(result) == 2
        ids = {r.node.node_id for r in result}
        assert ids == {a.node.node_id, b.node.node_id}

    def test_custom_k(self):
        from src.rag.retrieval import _rrf_fuse

        a = _nws("a")
        b = _nws("b")
        result = _rrf_fuse([a, b], k=1)
        assert result[0].score > result[1].score

    def test_score_is_sum_of_reciprocal_ranks(self):
        from src.rag.retrieval import _rrf_fuse

        a = _nws("a")
        b = _nws("b")
        result = _rrf_fuse([a, b], k=60)
        expected_a = 1.0 / 61
        expected_b = 1.0 / 62
        assert abs(result[0].score - expected_a) < 1e-9
        assert abs(result[1].score - expected_b) < 1e-9

    def test_three_lists_majority_wins(self):
        from src.rag.retrieval import _rrf_fuse

        a = _nws("a")
        b = _nws("b")
        result = _rrf_fuse([a], [a], [a, b])
        assert result[0].node.node_id == a.node.node_id
        assert result[0].score > result[1].score


# ── _file_filter ──


class TestFileFilter:
    """Генерация MetadataFilters по имени файла.

    _file_filter использует MetadataFilters/MetadataFilter/FilterOperator из llama_index,
    которые в тестах замокированы. Проверяем что конструкторы вызваны с правильными аргументами.
    """

    def test_single_string_uses_eq_operator(self):
        from llama_index.core.vector_stores.types import (
            FilterOperator,
            MetadataFilter,
            MetadataFilters,
        )

        from src.rag.retrieval import _file_filter

        _file_filter("doc.pdf")

        MetadataFilters.assert_called_once()
        MetadataFilter.assert_called_once_with(
            key="file_name", value="doc.pdf", operator=FilterOperator.EQ
        )

    def test_single_item_list(self):
        from llama_index.core.vector_stores.types import (
            FilterOperator,
            MetadataFilter,
        )

        from src.rag.retrieval import _file_filter

        _file_filter(["only.pdf"])

        MetadataFilter.assert_called_once_with(
            key="file_name", value="only.pdf", operator=FilterOperator.EQ
        )

    def test_multiple_files_uses_in_operator(self):
        from llama_index.core.vector_stores.types import (
            FilterOperator,
            MetadataFilter,
        )

        from src.rag.retrieval import _file_filter

        files = ["a.pdf", "b.pdf", "c.pdf"]
        _file_filter(files)

        MetadataFilter.assert_called_once_with(
            key="file_name", value=files, operator=FilterOperator.IN
        )


# ── _filter_chunks ──


class TestFilterChunks:
    """Адаптивный порог фильтрации чанков."""

    def test_few_nodes_uses_config_threshold(self):
        """<4 нод: порог = RERANK_SCORE_THRESHOLD, но min_chunks не даёт опуститься ниже."""
        from src.rag.retrieval import _filter_chunks

        with patch("src.rag.retrieval.config") as mock_cfg:
            mock_cfg.RERANK_SCORE_THRESHOLD = 0.1
            mock_cfg.MIN_FINAL_CHUNKS = 3
            mock_cfg.RAG_TOP_K_RATIO = 0.0
            nodes = [_nws("a", 0.5), _nws("b", 0.05), _nws("c", 0.3)]
            result = _filter_chunks(nodes)
            assert len(result) == 3  # min_chunks предотвращает потерю

    def test_many_nodes_filters_by_adaptive_threshold(self):
        """>=4 нод: адаптивный порог отсекает выбросы."""
        from src.rag.retrieval import _filter_chunks

        with patch("src.rag.retrieval.config") as mock_cfg:
            mock_cfg.RERANK_SCORE_THRESHOLD = 0.1
            mock_cfg.MIN_FINAL_CHUNKS = 2
            mock_cfg.RAG_TOP_K_RATIO = 0.0
            nodes = [
                _nws("a", 0.9),
                _nws("b", 0.85),
                _nws("c", 0.8),
                _nws("d", 0.75),
                _nws("e", 0.01),
                _nws("f", 0.005),
            ]
            result = _filter_chunks(nodes)
            assert len(result) == 4
            result_scores = [r.score for r in result]
            assert all(s > 0.5 for s in result_scores)

    def test_all_above_keeps_all(self):
        """Если adaptive порог ниже всех score — все остаются."""
        from src.rag.retrieval import _filter_chunks

        with patch("src.rag.retrieval.config") as mock_cfg:
            mock_cfg.RERANK_SCORE_THRESHOLD = 0.1
            mock_cfg.MIN_FINAL_CHUNKS = 2
            mock_cfg.RAG_TOP_K_RATIO = 0.0
            nodes = [_nws("a", 0.9), _nws("b", 0.88), _nws("c", 0.86), _nws("d", 0.84)]
            result = _filter_chunks(nodes)
            assert len(result) == 4

    def test_top_k_ratio_filters(self):
        """RAG_TOP_K_RATIO: отсекает ноды с score < top_score * ratio."""
        from src.rag.retrieval import _filter_chunks

        with patch("src.rag.retrieval.config") as mock_cfg:
            mock_cfg.RERANK_SCORE_THRESHOLD = 0.0
            mock_cfg.MIN_FINAL_CHUNKS = 2
            mock_cfg.RAG_TOP_K_RATIO = 0.5
            nodes = [
                _nws("a", 1.0),
                _nws("b", 0.9),
                _nws("c", 0.8),
                _nws("d", 0.4),
                _nws("e", 0.2),
            ]
            result = _filter_chunks(nodes)
            assert len(result) == 3
            result_scores = [r.score for r in result]
            assert all(s >= 0.5 for s in result_scores)

    def test_single_node_passthrough(self):
        from src.rag.retrieval import _filter_chunks

        with patch("src.rag.retrieval.config") as mock_cfg:
            mock_cfg.RERANK_SCORE_THRESHOLD = 0.9
            mock_cfg.MIN_FINAL_CHUNKS = 3
            mock_cfg.RAG_TOP_K_RATIO = 0.0
            result = _filter_chunks([_nws("a", 0.5)])
            assert len(result) == 1

    def test_empty_list(self):
        from src.rag.retrieval import _filter_chunks

        with patch("src.rag.retrieval.config") as mock_cfg:
            mock_cfg.RERANK_SCORE_THRESHOLD = 0.1
            mock_cfg.MIN_FINAL_CHUNKS = 3
            mock_cfg.RAG_TOP_K_RATIO = 0.0
            result = _filter_chunks([])
            assert result == []


# ── invalidate_index_cache ──


class TestInvalidateIndexCache:
    """Очистка кеша VectorStoreIndex."""

    def test_invalidate_specific_notebook(self):
        from src.rag.retrieval import _index_cache

        _index_cache.clear()
        _index_cache["nb1"] = MagicMock()
        _index_cache["nb2"] = MagicMock()

        with patch("src.rag.retrieval._index_cache_lock", threading.Lock()):
            from src.rag.retrieval import invalidate_index_cache

            invalidate_index_cache("nb1")

        assert "nb1" not in _index_cache
        assert "nb2" in _index_cache

    def test_invalidate_all(self):
        from src.rag.retrieval import _index_cache

        _index_cache.clear()
        _index_cache["nb1"] = MagicMock()
        _index_cache["nb2"] = MagicMock()

        with patch("src.rag.retrieval._index_cache_lock", threading.Lock()):
            from src.rag.retrieval import invalidate_index_cache

            invalidate_index_cache()

        assert len(_index_cache) == 0

    def test_invalidate_nonexistent_is_noop(self):
        from src.rag.retrieval import _index_cache

        _index_cache.clear()
        _index_cache["nb1"] = MagicMock()

        with patch("src.rag.retrieval._index_cache_lock", threading.Lock()):
            from src.rag.retrieval import invalidate_index_cache

            invalidate_index_cache("nonexistent")

        assert "nb1" in _index_cache


# ── _FilteredBM25 ──


class TestFilteredBM25:
    """Обёртка над BM25Retriever: фильтрация по allowed_files."""

    def test_filters_by_file_name(self):
        from src.rag.retrieval import _FilteredBM25

        mock_base = MagicMock()
        r1 = FakeNodeWithScore(FakeTextNode(text="a", metadata={"file_name": "a.pdf"}), 0.9)
        r2 = FakeNodeWithScore(FakeTextNode(text="b", metadata={"file_name": "b.pdf"}), 0.8)
        r3 = FakeNodeWithScore(FakeTextNode(text="c", metadata={"file_name": "c.pdf"}), 0.7)
        mock_base.retrieve.return_value = [r1, r2, r3]

        filtered = _FilteredBM25(mock_base, ["a.pdf", "c.pdf"])
        result = filtered.retrieve("query")

        assert len(result) == 2
        names = {r.node.metadata["file_name"] for r in result}
        assert names == {"a.pdf", "c.pdf"}

    def test_empty_allowed_filters_all(self):
        from src.rag.retrieval import _FilteredBM25

        mock_base = MagicMock()
        r1 = FakeNodeWithScore(FakeTextNode(text="a", metadata={"file_name": "a.pdf"}), 0.9)
        mock_base.retrieve.return_value = [r1]

        filtered = _FilteredBM25(mock_base, [])
        result = filtered.retrieve("query")
        assert len(result) == 0

    def test_no_metadata_file_name(self):
        from src.rag.retrieval import _FilteredBM25

        mock_base = MagicMock()
        r1 = FakeNodeWithScore(FakeTextNode(text="a", metadata={}), 0.9)
        mock_base.retrieve.return_value = [r1]

        filtered = _FilteredBM25(mock_base, ["a.pdf"])
        result = filtered.retrieve("query")
        assert len(result) == 0


# ── _is_llm_healthy ──


class TestIsLlmHealthy:
    """Проверка работоспособности LLM-сервера с кешем."""

    def test_cached_result_returns_without_network(self):
        from src.rag.retrieval import _qe_health_cache

        _qe_health_cache.clear()
        _qe_health_cache["http://test:1234/v1"] = (True, _time.time())

        from src.rag.retrieval import _is_llm_healthy

        result = asyncio.run(_is_llm_healthy("http://test:1234/v1"))
        assert result is True

    def test_cached_false_returns_without_network(self):
        from src.rag.retrieval import _qe_health_cache

        _qe_health_cache.clear()
        _qe_health_cache["http://test:1234/v1"] = (False, _time.time())

        from src.rag.retrieval import _is_llm_healthy

        result = asyncio.run(_is_llm_healthy("http://test:1234/v1"))
        assert result is False

    def test_health_check_success_caches_result(self):
        from src.rag.retrieval import _qe_health_cache

        _qe_health_cache.clear()

        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_context)
        mock_context.__aexit__ = AsyncMock(return_value=False)
        mock_context.get = AsyncMock(return_value=MagicMock())

        with patch("src.rag.retrieval.httpx") as mock_httpx:
            mock_httpx.AsyncClient.return_value = mock_context

            from src.rag.retrieval import _is_llm_healthy

            result = asyncio.run(_is_llm_healthy("http://test:1234/v1"))

        assert result is True
        cached = _qe_health_cache.get("http://test:1234/v1")
        assert cached is not None and cached[0] is True

    def test_health_check_failure_caches_result(self):
        from src.rag.retrieval import _qe_health_cache

        _qe_health_cache.clear()

        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_context)
        mock_context.__aexit__ = AsyncMock(return_value=False)
        mock_context.get = AsyncMock(side_effect=ConnectionError("refused"))

        with patch("src.rag.retrieval.httpx") as mock_httpx:
            mock_httpx.AsyncClient.return_value = mock_context

            from src.rag.retrieval import _is_llm_healthy

            result = asyncio.run(_is_llm_healthy("http://test:1234/v1"))

        assert result is False
        cached = _qe_health_cache.get("http://test:1234/v1")
        assert cached is not None and cached[0] is False
