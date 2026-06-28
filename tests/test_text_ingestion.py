"""Тесты модуля ingestion.text: обработка PDF, отмена через cancel_check."""

import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Исключение отмены ingestion ──
class IngestionCancelled(Exception):
    pass


# ── Фикстура мокирования зависимостей ──
@pytest.fixture(autouse=True)
def mock_text_deps():
    mocks = {
        "cv2": MagicMock(),
        "cv2.typing": MagicMock(),
        "fitz": MagicMock(),
        "numpy": MagicMock(),
        "torch": MagicMock(),
        "orjson": MagicMock(),
        "httpx": MagicMock(),
        "aiofiles": MagicMock(),
        "aiofiles.os": AsyncMock(),
        "llama_index": MagicMock(),
        "llama_index.core": MagicMock(),
        "llama_index.core.schema": MagicMock(),
        "src.gguf": MagicMock(),
        "src.gguf.server": MagicMock(),
        "src.ingestion.audio_video": MagicMock(),
        "src.ingestion.media_convert": MagicMock(),
        "src.ingestion.splitter": MagicMock(),
        "src.ingestion.vision": MagicMock(),
        "src.ingestion.utils": MagicMock(),
        "routers": MagicMock(),
        "routers.shared": MagicMock(),
    }
    mocks["src.ingestion.utils"].IngestionCancelled = IngestionCancelled

    for key in list(sys.modules):
        if key.startswith("src.ingestion") and key != "src.ingestion.utils":
            sys.modules.pop(key, None)

    with patch.dict(sys.modules, mocks, clear=False):
        yield

    for key in list(sys.modules):
        if key.startswith("src.ingestion") and key not in mocks:
            sys.modules.pop(key, None)


# ── Обработка PDF-файлов ──
class TestProcessPdf:
    def test_basic_pdf_processing(self):
        from src.ingestion.text import process_pdf

        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=2)
        mock_doc.__enter__ = MagicMock(return_value=mock_doc)
        mock_doc.__exit__ = MagicMock(return_value=False)

        mock_splitter = MagicMock()
        mock_splitter.get_nodes_from_documents.return_value = [MagicMock(text="node1")]

        with (
            patch("src.ingestion.text.fitz") as mock_fitz,
            patch("config.SURYA_MODE", "disabled"),
            patch(
                "pymupdf4llm.to_markdown",
                return_value=[
                    {"metadata": {"page_number": 1}, "text": "markdown text"},
                    {"metadata": {"page_number": 2}, "text": "more text"},
                ],
            ),
            patch("src.ingestion.text._get_splitter", return_value=mock_splitter),
        ):
            mock_fitz.open.return_value = mock_doc
            with patch(
                "src.ingestion.text._analyze_and_build_page", new_callable=AsyncMock
            ) as mock_abp:
                mock_abp.return_value = (0, None)
                result = asyncio.run(process_pdf("/tmp/test.pdf", "/tmp/images", llm_settings={}))
        assert len(result) == 2

    def test_cancel_before_first_page(self):
        from src.ingestion.text import process_pdf

        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=5)
        mock_doc.__enter__ = MagicMock(return_value=mock_doc)
        mock_doc.__exit__ = MagicMock(return_value=False)

        mock_splitter = MagicMock()
        mock_splitter.get_nodes_from_documents.return_value = []

        with (
            patch("src.ingestion.text.fitz") as mock_fitz,
            patch("config.SURYA_MODE", "disabled"),
            patch("pymupdf4llm.to_markdown", return_value=[]),
            patch("src.ingestion.text._get_splitter", return_value=mock_splitter),
        ):
            mock_fitz.open.return_value = mock_doc
            with patch("src.ingestion.text._analyze_and_build_page", new_callable=AsyncMock):
                with pytest.raises(IngestionCancelled):
                    asyncio.run(
                        process_pdf(
                            "/tmp/test.pdf",
                            "/tmp/images",
                            cancel_check=lambda: True,
                        )
                    )

    def test_no_vision_when_no_frames(self):
        from src.ingestion.text import process_pdf

        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=1)
        mock_doc.__enter__ = MagicMock(return_value=mock_doc)
        mock_doc.__exit__ = MagicMock(return_value=False)

        mock_splitter = MagicMock()
        mock_splitter.get_nodes_from_documents.return_value = [MagicMock(text="text_only")]

        with (
            patch("src.ingestion.text.fitz") as mock_fitz,
            patch("config.SURYA_MODE", "disabled"),
            patch(
                "pymupdf4llm.to_markdown",
                return_value=[
                    {"metadata": {"page_number": 1}, "text": "text_only"},
                ],
            ),
            patch("src.ingestion.text._get_splitter", return_value=mock_splitter),
        ):
            mock_fitz.open.return_value = mock_doc
            with patch(
                "src.ingestion.text._analyze_and_build_page", new_callable=AsyncMock
            ) as mock_abp:
                mock_abp.return_value = (0, None)
                with patch("src.ingestion.text.get_vision_url", new_callable=AsyncMock) as mock_vu:
                    asyncio.run(process_pdf("/tmp/test.pdf", "/tmp/images", llm_settings={}))
                    mock_vu.assert_not_called()
