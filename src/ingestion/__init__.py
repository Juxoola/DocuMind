"""Подсистема загрузки данных: разбор файлов, обработка аудио/видео, OCR, разбиение на фрагменты."""

from src.ingestion.audio_video import (
    get_or_load_whisper,
    process_audio_video,
    save_high_res_frame,
    unload_whisper_model,
)
from src.ingestion.media_convert import ensure_720p_video, ensure_mp3_audio
from src.ingestion.office_convert import process_docx, process_pptx
from src.ingestion.orchestrator import ingest_file
from src.ingestion.text import process_pdf
from src.ingestion.utils import (
    IngestionCancelled,
    cleanup_gpu,
    format_seconds,
    kill_subprocesses,
    register_subprocess,
    unregister_subprocess,
)
from src.ingestion.vision import (
    describe_image_with_lmstudio,
    get_image_base64,
    get_vision_url,
)

__all__ = [
    "IngestionCancelled",
    "cleanup_gpu",
    "describe_image_with_lmstudio",
    "ensure_720p_video",
    "ensure_mp3_audio",
    "format_seconds",
    "get_image_base64",
    "get_or_load_whisper",
    "get_vision_url",
    "ingest_file",
    "kill_subprocesses",
    "process_audio_video",
    "process_docx",
    "process_pdf",
    "process_pptx",
    "register_subprocess",
    "save_high_res_frame",
    "unload_whisper_model",
    "unregister_subprocess",
]
