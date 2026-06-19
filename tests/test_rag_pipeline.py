"""
Тесты src/rag_pipeline.py (чистые функции RAG).

Тестируем _rrf_fuse, _rrf_fuse_across_files и API отложенной пересборки BM25.
Внешние пакеты (llama_index, torch, chromadb) мокаются через patch.dict(sys.modules, ...)
— это единственный способ перехватить top-level импорты в src.rag.retrieval
до загрузки тестируемого модуля.
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest
import asyncio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class FakeTextNode:
    """Замена TextNode для тестов — без зависимостей от llama_index."""

    def __init__(self, text="", id_=None, metadata=None):
        self.text = text
        self.node_id = id_ or hash(text)
        self.metadata = metadata or {}


class FakeNodeWithScore:
    """Замена NodeWithScore для тестов."""

    def __init__(self, node=None, score=0.0):
        self.node = node if node else FakeTextNode()
        self.score = score


@pytest.fixture(autouse=True)
def mock_llama_index():
    """
    Мокаем llama_index + torch + chromadb через sys.modules.

    Создаём иерархию моков так, чтобы `from llama_index.core.schema import NodeWithScore`
    возвращал FakeNodeWithScore, а не MagicMock — тесты создают инстансы через конструктор.
    """
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
    }

    with patch.dict(sys.modules, mocks, clear=False):
        yield


class TestRrfFuse:
    """Reciprocal Rank Fusion — основная функция гибридного поиска."""

    def _get_funcs(self):
        """Импортируем после установки моков."""
        from src.rag.retrieval import _rrf_fuse

        return _rrf_fuse

    def test_empty_both(self, mock_llama_index):
        f = self._get_funcs()
        assert f([], []) == []

    def test_empty_vector(self, mock_llama_index):
        from llama_index.core.schema import NodeWithScore, TextNode

        from src.rag.retrieval import _rrf_fuse

        bm25 = [
            NodeWithScore(node=TextNode(text="A", id_="a"), score=0.9),
            NodeWithScore(node=TextNode(text="B", id_="b"), score=0.8),
        ]
        result = _rrf_fuse([], bm25)
        assert len(result) == 2
        assert result[0].score >= result[1].score
        assert result[0].node.node_id == "a"

    def test_empty_bm25(self, mock_llama_index):
        from llama_index.core.schema import NodeWithScore, TextNode

        from src.rag.retrieval import _rrf_fuse

        vec = [NodeWithScore(node=TextNode(text="X", id_="x"), score=0.5)]
        result = _rrf_fuse(vec, [])
        assert len(result) == 1
        assert result[0].node.node_id == "x"

    def test_deduplicates(self, mock_llama_index):
        from llama_index.core.schema import NodeWithScore, TextNode

        from src.rag.retrieval import _rrf_fuse

        common = TextNode(text="Common", id_="same_id")
        vec = [NodeWithScore(node=common, score=0.9)]
        bm25 = [NodeWithScore(node=common, score=0.8)]
        result = _rrf_fuse(vec, bm25)
        assert len(result) == 1
        expected = 1.0 / 61 + 1.0 / 61
        assert abs(result[0].score - expected) < 1e-6


class TestRrfFuseAcrossFiles:
    """RRF между файлами — каждый файл имеет равный голос."""

    def test_empty_input(self, mock_llama_index):
        from src.rag.retrieval import _rrf_fuse_across_files

        assert _rrf_fuse_across_files([]) == []

    def test_single_file(self, mock_llama_index):
        from llama_index.core.schema import NodeWithScore, TextNode

        from src.rag.retrieval import _rrf_fuse_across_files

        nodes = [
            NodeWithScore(node=TextNode(text="A", id_="a1"), score=0.9),
            NodeWithScore(node=TextNode(text="B", id_="b1"), score=0.8),
        ]
        result = _rrf_fuse_across_files([("file1.pdf", nodes)])
        assert len(result) == 2

    def test_two_files_equal_weight(self, mock_llama_index):
        from llama_index.core.schema import NodeWithScore, TextNode

        from src.rag.retrieval import _rrf_fuse_across_files

        big = [
            NodeWithScore(
                node=TextNode(text=f"Big-{i}", id_=f"big_{i}"),
                score=1.0 - i * 0.01,
            )
            for i in range(100)
        ]
        small = [
            NodeWithScore(node=TextNode(text="Important", id_="golden"), score=0.99),
            NodeWithScore(node=TextNode(text="Other", id_="silver"), score=0.5),
        ]

        result = _rrf_fuse_across_files([("big.pdf", big), ("small.pdf", small)])
        ids = [nws.node.node_id for nws in result]
        assert "golden" in ids[:3], f"golden должен быть в топ-3, got: {ids[:5]}"


class TestBm25RebuildApi:
    """API отложенной пересборки BM25 (таймеры, отмена, флаш)."""

    def test_schedule_cancel(self, mock_llama_index):
        from src.rag.bm25 import _schedule_bm25_rebuild, cancel_bm25_rebuild

        asyncio.run(_schedule_bm25_rebuild("test_nb", "/tmp/fake_db"))
        asyncio.run(cancel_bm25_rebuild("test_nb"))

    def test_cancel_nonexistent(self, mock_llama_index):
        from src.rag.bm25 import cancel_bm25_rebuild

        asyncio.run(cancel_bm25_rebuild("never_scheduled"))

    def test_flush_without_wait(self, mock_llama_index):
        from src.rag.bm25 import flush_bm25_rebuild

        with patch("src.rag.bm25._rebuild_bm25_bg"):
            asyncio.run(flush_bm25_rebuild("test_nb", db_path="/tmp/fake", wait=False))

    def test_is_bm25_ready_no_index(self, mock_llama_index):
        from src.rag.bm25 import is_bm25_ready

        assert asyncio.run(is_bm25_ready("nonexistent")) is False
