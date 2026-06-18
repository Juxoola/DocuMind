"""Оркестрация ингеста: маршрутизация файла по типу к нужному обработчику."""

import logging
import os
import shutil
import uuid

from llama_index.core.schema import TextNode

import config
from src.ingestion.audio_video import process_audio_video
from src.ingestion.media_convert import ensure_720p_video, ensure_mp3_audio
from src.ingestion.splitter import _get_splitter
from src.ingestion.text import process_docx, process_pdf, process_pptx
from src.ingestion.utils import IngestionCancelled
from src.ingestion.vision import describe_image_with_lmstudio, get_vision_url

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
    elif ext in [".jpg", ".jpeg", ".png", ".webp"]:
        nodes = process_image(
            file_path,
            images_dir,
            llm_settings,
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


def process_image(
    file_path,
    images_dir,
    llm_settings=None,
    progress_cb=None,
    cancel_check=None,
    keep_vision_alive=False,
):
    """Анализ изображения через Vision: описание → RAG + просмотр в правой панели."""
    file_name = os.path.basename(file_path)
    logger.info(f"[IMAGE] Анализ: {file_name}")

    # Копируем в images/ для просмотра в правой панели
    dest_name = f"img_{uuid.uuid4().hex[:6]}{os.path.splitext(file_path)[1].lower()}"
    dest_path = os.path.join(images_dir, dest_name)
    shutil.copy2(file_path, dest_path)

    def _is_cancelled():
        return bool(cancel_check and cancel_check())

    nodes = []
    frame_data = []

    if _is_cancelled():
        from src.ingestion.utils import IngestionCancelled

        raise IngestionCancelled("Cancelled before image analysis")

    shared_llm_url = get_vision_url(llm_settings)
    if shared_llm_url:
        if progress_cb:
            progress_cb(30, f"Анализ изображения: {file_name}...")
        desc = describe_image_with_lmstudio(dest_path, llm_settings, shared_llm_url)
        if desc and "Изображение без описания" not in desc:
            full_text = f"Изображение {file_name}: {desc}"
            splitter = _get_splitter()
            if len(full_text) <= config.GGUF_CTX_EMBED_CHARS:
                nodes.append(
                    TextNode(
                        text=full_text,
                        metadata={
                            "file_name": file_name,
                            "image_path": dest_path,
                        },
                    )
                )
            else:
                desc_nodes = splitter.get_nodes_from_documents(
                    [
                        TextNode(
                            text=full_text,
                            metadata={
                                "file_name": file_name,
                                "image_path": dest_path,
                            },
                        )
                    ]
                )
                nodes.extend(desc_nodes)
            frame_data.append({"image_path": dest_path, "description": desc})
            logger.info(f"[IMAGE] Описание получено ({len(desc)} симв.)")
        else:
            logger.info(f"[IMAGE] Vision не вернул описание для {file_name}")

        if shared_llm_url and not keep_vision_alive:
            from src.gguf.server import unload_all_models

            unload_all_models(role="llm")
    else:
        logger.info("[IMAGE] Vision не настроен — пропуск анализа изображения")

    # Сохраняем метаданные
    if frame_data:
        metadata_json = {
            "file_name": file_name,
            "is_video": False,
            "transcript": [],
            "frames": frame_data,
        }
        with open(
            os.path.join(os.path.dirname(file_path), f"{file_name}.json"),
            "w",
            encoding="utf-8",
        ) as f:
            import orjson

            f.write(orjson.dumps(metadata_json, option=orjson.OPT_INDENT_2).decode())

    if progress_cb:
        progress_cb(60, f"Изображение: {file_name} готово")

    return nodes
