"""Тесты src/ingestion/text.py: _detect_has_real_graphics, process_pdf, отмена через cancel_check."""

import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class IngestionCancelled(Exception):
    pass


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


class TestDetectHasRealGraphics:
    def test_empty_images_and_drawings(self):
        from src.ingestion.text import _detect_has_real_graphics

        assert _detect_has_real_graphics([], []) is False

    def test_with_images(self):
        from src.ingestion.text import _detect_has_real_graphics

        assert _detect_has_real_graphics([b"\x89PNG"], []) is True

    def test_complex_drawings_items_gt_12(self):
        from src.ingestion.text import _detect_has_real_graphics

        drawings = [{"items": [("re",)] * 15, "rect": MagicMock(), "fill": None}]
        assert _detect_has_real_graphics([], drawings) is True

    def test_curve_items(self):
        from src.ingestion.text import _detect_has_real_graphics

        drawings = [{"items": [("c",)], "rect": MagicMock(), "fill": None}]
        assert _detect_has_real_graphics([], drawings) is True

    def test_background_rect_only(self):
        from src.ingestion.text import _detect_has_real_graphics

        rect = MagicMock()
        rect.x0, rect.x1 = 0, 500
        rect.y0, rect.y1 = 0, 700
        drawings = [{"items": [("re",)], "rect": rect, "fill": (1.0, 1.0, 1.0)}]
        assert _detect_has_real_graphics([], drawings) is False

    def test_many_small_drawings(self):
        from src.ingestion.text import _detect_has_real_graphics

        drawings = []
        for _ in range(10):
            r = MagicMock()
            r.x0, r.x1 = 10, 20
            r.y0, r.y1 = 10, 20
            drawings.append({"items": [("re",)], "rect": r, "fill": None})
        assert _detect_has_real_graphics([], drawings) is True

    def test_grid_lines(self):
        from src.ingestion.text import _detect_has_real_graphics

        drawings = []
        for _ in range(3):
            rect = MagicMock()
            rect.x0, rect.x1 = 0, 200
            rect.y0, rect.y1 = 50, 51
            drawings.append({"items": [("re",)], "rect": rect, "fill": None})
        for _ in range(2):
            rect = MagicMock()
            rect.x0, rect.x1 = 50, 51
            rect.y0, rect.y1 = 0, 200
            drawings.append({"items": [("re",)], "rect": rect, "fill": None})
        assert _detect_has_real_graphics([], drawings) is True


class TestProcessPdf:
    def test_basic_pdf_processing(self):
        from src.ingestion.text import process_pdf

        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=2)
        mock_doc.__enter__ = MagicMock(return_value=mock_doc)
        mock_doc.__exit__ = MagicMock(return_value=False)

        with patch("src.ingestion.text.fitz") as mock_fitz, \
             patch("config.SURYA_MODE", "disabled"):
            mock_fitz.open.return_value = mock_doc
            with patch("src.ingestion.text._get_splitter"):
                with patch(
                    "src.ingestion.text._analyze_and_build_page", new_callable=AsyncMock
                ) as mock_abp:
                    mock_abp.return_value = (0, [MagicMock(text="node1")], None)
                    result = asyncio.run(
                        process_pdf("/tmp/test.pdf", "/tmp/images", llm_settings={})
                    )
        assert len(result) == 2

    def test_cancel_before_first_page(self):
        """Отмена сразу — cancel_check возвращает True с первого вызова."""
        from src.ingestion.text import process_pdf

        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=5)
        mock_doc.__enter__ = MagicMock(return_value=mock_doc)
        mock_doc.__exit__ = MagicMock(return_value=False)

        with patch("src.ingestion.text.fitz") as mock_fitz, \
             patch("config.SURYA_MODE", "disabled"):
            mock_fitz.open.return_value = mock_doc
            with patch("src.ingestion.text._get_splitter"):
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

        with patch("src.ingestion.text.fitz") as mock_fitz, \
             patch("config.SURYA_MODE", "disabled"):
            mock_fitz.open.return_value = mock_doc
            with patch("src.ingestion.text._get_splitter"):
                with patch(
                    "src.ingestion.text._analyze_and_build_page", new_callable=AsyncMock
                ) as mock_abp:
                    mock_abp.return_value = (0, [MagicMock(text="text_only")], None)
                    with patch(
                        "src.ingestion.text.get_vision_url", new_callable=AsyncMock
                    ) as mock_vu:
                        asyncio.run(process_pdf("/tmp/test.pdf", "/tmp/images", llm_settings={}))
                        mock_vu.assert_not_called()
