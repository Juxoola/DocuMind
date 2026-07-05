"""Тесты модуля orchestrator: маршрутизация файлов, отмена, конвертация медиа."""

import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class IngestionCancelled(Exception):
    pass


@pytest.fixture(autouse=True)
def mock_orch_deps():
    mocks = {
        "cv2": MagicMock(),
        "cv2.typing": MagicMock(),
        "numpy": MagicMock(),
        "torch": MagicMock(),
        "orjson": MagicMock(),
        "httpx": MagicMock(),
        "llama_index": MagicMock(),
        "llama_index.core": MagicMock(),
        "llama_index.core.schema": MagicMock(),
        "src.gguf": MagicMock(),
        "src.gguf.server": MagicMock(),
        "src.ingestion.text": MagicMock(),
        "src.ingestion.office_convert": MagicMock(),
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
        if key.startswith("src.ingestion"):
            sys.modules.pop(key, None)

    aiofiles_mock = MagicMock()
    aiofiles_mock.os = AsyncMock()
    aiofiles_mock.open = MagicMock()

    with patch.dict(sys.modules, mocks, clear=False):
        with patch("src.ingestion.orchestrator.aiofiles", aiofiles_mock):
            yield

    for key in list(sys.modules):
        if key.startswith("src.ingestion"):
            sys.modules.pop(key, None)


@pytest.fixture
def fake_paths(monkeypatch):
    import tempfile

    tmp = tempfile.mkdtemp(prefix="orch_test_")
    nb_id = "test_nb"
    import config as cfg

    monkeypatch.setattr(cfg, "NOTEBOOKS_DIR", tmp)

    def fake_get_notebook_paths(notebook_id):
        return {
            "root": os.path.join(tmp, notebook_id),
            "data": os.path.join(tmp, notebook_id, "data"),
            "chroma_db": os.path.join(tmp, notebook_id, "chroma_db"),
            "images": os.path.join(tmp, notebook_id, "images"),
        }

    monkeypatch.setattr(cfg, "get_notebook_paths", fake_get_notebook_paths)
    return tmp, nb_id


# ── Маршрутизация файлов по типу расширения ──
class TestFileRouting:
    def test_pdf_routes_to_process_pdf(self, fake_paths):
        from src.ingestion.orchestrator import ingest_file
        from src.ingestion.text import process_pdf

        process_pdf.return_value = AsyncMock(return_value=[MagicMock()])()
        asyncio.run(ingest_file("/tmp/test.pdf", "test_nb", llm_settings={}))
        process_pdf.assert_called_once()

    def test_pptx_routes_to_process_pptx(self, fake_paths):
        from src.ingestion.office_convert import process_pptx
        from src.ingestion.orchestrator import ingest_file

        process_pptx.return_value = AsyncMock(return_value=[MagicMock()])()
        asyncio.run(ingest_file("/tmp/test.pptx", "test_nb", llm_settings={}))
        process_pptx.assert_called_once()

    def test_docx_routes_to_process_docx(self, fake_paths):
        from src.ingestion.office_convert import process_docx
        from src.ingestion.orchestrator import ingest_file

        process_docx.return_value = AsyncMock(return_value=[MagicMock()])()
        asyncio.run(ingest_file("/tmp/test.docx", "test_nb", llm_settings={}))
        process_docx.assert_called_once()

    def test_image_routes_to_process_image(self, fake_paths):
        from src.ingestion.orchestrator import ingest_file

        mock_nodes = [MagicMock(text="img")]
        with patch(
            "src.ingestion.orchestrator.process_image",
            new_callable=AsyncMock,
            return_value=mock_nodes,
        ) as pi:
            asyncio.run(ingest_file("/tmp/photo.png", "test_nb", llm_settings={}))
        pi.assert_called_once()

    def test_jpeg_routes_to_process_image(self, fake_paths):
        from src.ingestion.orchestrator import ingest_file

        mock_nodes = [MagicMock(text="img")]
        with patch(
            "src.ingestion.orchestrator.process_image",
            new_callable=AsyncMock,
            return_value=mock_nodes,
        ) as pi:
            asyncio.run(ingest_file("/tmp/photo.jpeg", "test_nb", llm_settings={}))
        pi.assert_called_once()

    def test_txt_routes_to_text_fallback(self, fake_paths):
        from src.ingestion.orchestrator import ingest_file

        tmp, nb_id = fake_paths
        txt_path = os.path.join(tmp, "test.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("Hello world")

        mock_file = AsyncMock()
        mock_file.__aenter__ = AsyncMock(return_value=mock_file)
        mock_file.__aexit__ = AsyncMock(return_value=False)
        mock_file.read = AsyncMock(return_value="Hello world")

        aiofiles_mock = MagicMock()
        aiofiles_mock.open = MagicMock(return_value=mock_file)
        aiofiles_mock.os = AsyncMock()

        with patch("src.ingestion.orchestrator.aiofiles", aiofiles_mock):
            with patch("src.ingestion.orchestrator._get_splitter") as mock_splitter:
                mock_splitter.return_value.get_nodes_from_documents.return_value = [
                    MagicMock(text="chunk")
                ]
                result = asyncio.run(ingest_file(txt_path, nb_id, llm_settings={}))
        assert len(result) >= 1


# ── Отмена процесса ingestion ──
class TestCancellation:
    def test_cancel_before_media_conversion(self, fake_paths):
        from src.ingestion.orchestrator import ingest_file

        with pytest.raises(IngestionCancelled, match="Cancelled before media conversion"):
            asyncio.run(ingest_file("/tmp/test.pdf", "test_nb", cancel_check=lambda: True))

    def test_cancel_after_media_conversion(self, fake_paths):
        from src.ingestion.orchestrator import ingest_file

        call_count = [0]

        def cancel_fn():
            call_count[0] += 1
            return call_count[0] >= 2

        with patch("src.ingestion.orchestrator.ensure_720p_video", new_callable=AsyncMock) as mv:
            mv.return_value = "/tmp/conv.mp4"
            with pytest.raises(IngestionCancelled, match="Cancelled after media conversion"):
                asyncio.run(ingest_file("/tmp/video.mp4", "test_nb", cancel_check=cancel_fn))

    def test_no_cancel_when_check_returns_false(self, fake_paths):
        from src.ingestion.orchestrator import ingest_file
        from src.ingestion.text import process_pdf

        process_pdf.return_value = AsyncMock(return_value=[MagicMock()])()
        asyncio.run(
            ingest_file("/tmp/test.pdf", "test_nb", llm_settings={}, cancel_check=lambda: False)
        )
        process_pdf.assert_called_once()

    def test_no_cancel_when_no_check(self, fake_paths):
        from src.ingestion.orchestrator import ingest_file
        from src.ingestion.text import process_pdf

        process_pdf.return_value = AsyncMock(return_value=[MagicMock()])()
        asyncio.run(ingest_file("/tmp/test.pdf", "test_nb", llm_settings={}))
        process_pdf.assert_called_once()


# ── Конвертация медиафайлов (видео → 720p, аудио → MP3) ──
class TestMediaConversion:
    def test_video_converts_via_720p(self, fake_paths):
        from src.ingestion.audio_video import process_audio_video
        from src.ingestion.media_convert import ensure_720p_video
        from src.ingestion.orchestrator import ingest_file

        ensure_720p_video.return_value = AsyncMock(return_value="/tmp/conv.mp4")()
        process_audio_video.return_value = AsyncMock(return_value=[MagicMock()])()
        asyncio.run(ingest_file("/tmp/video.mp4", "test_nb", llm_settings={}))
        ensure_720p_video.assert_called_once()
        process_audio_video.assert_called_once()

    def test_wav_converts_via_mp3(self, fake_paths):
        from src.ingestion.audio_video import process_audio_video
        from src.ingestion.media_convert import ensure_mp3_audio
        from src.ingestion.orchestrator import ingest_file

        ensure_mp3_audio.return_value = AsyncMock(return_value="/tmp/conv.mp3")()
        process_audio_video.return_value = AsyncMock(return_value=[MagicMock()])()
        asyncio.run(ingest_file("/tmp/audio.wav", "test_nb", llm_settings={}))
        ensure_mp3_audio.assert_called_once()
        process_audio_video.assert_called_once()


# ── Передача параметров в обработчики ──
class TestParameterPassing:
    def test_progress_cb_passed_to_pdf(self, fake_paths):
        from src.ingestion.orchestrator import ingest_file
        from src.ingestion.text import process_pdf

        process_pdf.return_value = AsyncMock(return_value=[MagicMock()])()
        cb = MagicMock()
        asyncio.run(ingest_file("/tmp/test.pdf", "test_nb", llm_settings={}, progress_cb=cb))
        _, kwargs = process_pdf.call_args
        assert kwargs.get("progress_cb") == cb

    def test_cancel_check_passed_to_pdf(self, fake_paths):
        from src.ingestion.orchestrator import ingest_file
        from src.ingestion.text import process_pdf

        process_pdf.return_value = AsyncMock(return_value=[MagicMock()])()

        def cancel_fn():
            return False

        asyncio.run(
            ingest_file("/tmp/test.pdf", "test_nb", llm_settings={}, cancel_check=cancel_fn)
        )
        _, kwargs = process_pdf.call_args
        assert kwargs.get("cancel_check") == cancel_fn

    def test_llm_settings_passed_to_pdf(self, fake_paths):
        from src.ingestion.orchestrator import ingest_file
        from src.ingestion.text import process_pdf

        process_pdf.return_value = AsyncMock(return_value=[MagicMock()])()
        settings = {"temperature": 0.7}
        asyncio.run(ingest_file("/tmp/test.pdf", "test_nb", llm_settings=settings))
        args, _ = process_pdf.call_args
        assert args[2] == settings
