"""
Тесты src/rag_pipeline.py.

Тестируем чистые функции с подменой llama_index на sys.modules уровне.
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(autouse=True)
def mock_llama_index():
    """
    Мокаем llama_index для всех тестов в этом файле.
    _rrf_fuse и _rrf_fuse_across_files импортируют TextNode и NodeWithScore
    из llama_index.core.schema — даём им работоспособные заменители.
    """

    class FakeTextNode:
        def __init__(self, text="", id_=None, metadata=None):
            self.text = text
            self.node_id = id_ or hash(text)
            self.metadata = metadata or {}

    class FakeNodeWithScore:
        def __init__(self, node=None, score=0.0):
            self.node = node if node else FakeTextNode()
            self.score = score

    mock_schema = MagicMock()
    mock_schema.TextNode = FakeTextNode
    mock_schema.NodeWithScore = FakeNodeWithScore

    mock_core = MagicMock()
    mock_core.schema = mock_schema

    mock_llama = MagicMock()
    mock_llama.core = mock_core

    mocks = {
        "llama_index": mock_llama,
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
        from src.rag_pipeline import _rrf_fuse

        return _rrf_fuse

    def test_empty_both(self, mock_llama_index):
        f = self._get_funcs()
        assert f([], []) == []

    def test_empty_vector(self, mock_llama_index):
        from llama_index.core.schema import NodeWithScore, TextNode

        from src.rag_pipeline import _rrf_fuse

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

        from src.rag_pipeline import _rrf_fuse

        vec = [NodeWithScore(node=TextNode(text="X", id_="x"), score=0.5)]
        result = _rrf_fuse(vec, [])
        assert len(result) == 1
        assert result[0].node.node_id == "x"

    def test_deduplicates(self, mock_llama_index):
        from llama_index.core.schema import NodeWithScore, TextNode

        from src.rag_pipeline import _rrf_fuse

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
        from src.rag_pipeline import _rrf_fuse_across_files

        assert _rrf_fuse_across_files([]) == []

    def test_single_file(self, mock_llama_index):
        from llama_index.core.schema import NodeWithScore, TextNode

        from src.rag_pipeline import _rrf_fuse_across_files

        nodes = [
            NodeWithScore(node=TextNode(text="A", id_="a1"), score=0.9),
            NodeWithScore(node=TextNode(text="B", id_="b1"), score=0.8),
        ]
        result = _rrf_fuse_across_files([("file1.pdf", nodes)])
        assert len(result) == 2

    def test_two_files_equal_weight(self, mock_llama_index):
        from llama_index.core.schema import NodeWithScore, TextNode

        from src.rag_pipeline import _rrf_fuse_across_files

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
        from src.rag_pipeline import _schedule_bm25_rebuild, cancel_bm25_rebuild

        _schedule_bm25_rebuild("test_nb", "/tmp/fake_db")
        cancel_bm25_rebuild("test_nb")

    def test_cancel_nonexistent(self, mock_llama_index):
        from src.rag_pipeline import cancel_bm25_rebuild

        cancel_bm25_rebuild("never_scheduled")

    def test_flush_without_wait(self, mock_llama_index):
        from src.rag_pipeline import flush_bm25_rebuild

        with patch("src.rag_pipeline._rebuild_bm25_bg"):
            flush_bm25_rebuild("test_nb", db_path="/tmp/fake", wait=False)

    def test_is_bm25_ready_no_index(self, mock_llama_index):
        from src.rag_pipeline import is_bm25_ready

        assert is_bm25_ready("nonexistent") is False
