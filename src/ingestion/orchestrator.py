"""Оркестрация ингеста: маршрутизация файла по типу к нужному обработчику."""

import logging
import os

from llama_index.core.schema import TextNode

import config
from src.ingestion.audio_video import process_audio_video
from src.ingestion.media_convert import ensure_720p_video, ensure_mp3_audio
from src.ingestion.splitter import _get_splitter
from src.ingestion.text import process_docx, process_pdf, process_pptx
from src.ingestion.utils import IngestionCancelled

logger = logging.getLogger(__name__)


def ingest_file(
    file_path,
    notebook_id,
    progress_cb=None,
    llm_settings=None,
    cancel_check=None,
    keep_vision_alive=False,
    keep_whisper_alive=False,
):

    def _is_cancelled():
        return bool(cancel_check and cancel_check())

    ext = os.path.splitext(file_path)[1].lower()
    paths = config.get_notebook_paths(notebook_id)
    images_dir = paths["images"]
    os.makedirs(images_dir, exist_ok=True)

    if _is_cancelled():
        raise IngestionCancelled("Cancelled before media conversion")

    if ext in [".mp4", ".avi", ".mkv", ".mov"]:
        file_path = ensure_720p_video(
            file_path, progress_cb, cancel_check=cancel_check, notebook_id=notebook_id
        )
    elif ext in [".mp3", ".wav", ".m4a"]:
        file_path = ensure_mp3_audio(file_path, progress_cb)
        ext = ".mp3"

    if _is_cancelled():
        raise IngestionCancelled("Cancelled after media conversion")

    if ext in [".mp4", ".avi", ".mkv", ".mov", ".mp3"]:
        return process_audio_video(
            file_path,
            images_dir,
            ext != ".mp3",
            progress_cb,
            llm_settings,
            cancel_check=cancel_check,
            notebook_id=notebook_id,
            keep_vision_alive=keep_vision_alive,
            keep_whisper_alive=keep_whisper_alive,
        )

    shared_llm_url = None
    if ext == ".pdf":
        nodes = process_pdf(
            file_path,
            images_dir,
            llm_settings,
            shared_llm_url,
            progress_cb=progress_cb,
            cancel_check=cancel_check,
            keep_vision_alive=keep_vision_alive,
        )
    elif ext == ".pptx":
        nodes = process_pptx(
            file_path,
            images_dir,
            llm_settings,
            shared_llm_url,
            progress_cb=progress_cb,
            cancel_check=cancel_check,
            keep_vision_alive=keep_vision_alive,
        )
    elif ext == ".docx":
        nodes = process_docx(
            file_path,
            images_dir,
            llm_settings,
            shared_llm_url,
            progress_cb=progress_cb,
            cancel_check=cancel_check,
            keep_vision_alive=keep_vision_alive,
        )
    else:
        try:
            with open(file_path, encoding="utf-8") as f:
                text = f.read()
        except UnicodeDecodeError:
            with open(file_path, encoding="cp1251") as f:
                text = f.read()
        doc = TextNode(text=text, metadata={"file_name": os.path.basename(file_path)})
        nodes = _get_splitter().get_nodes_from_documents([doc])
    return nodes
