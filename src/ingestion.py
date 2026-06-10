"""Trampoline для обратной совместимости.

Весь код ингеста переехал в пакет src/ingestion/. Этот файл реэкспортирует
все публичные и приватные символы, чтобы старые импорты
  from src.ingestion import ...
продолжали работать без изменений.
"""

from src.ingestion.audio_video import (
    _whisper_model_cache,
    _whisper_lock,
    get_or_load_whisper,
    process_audio_video,
    save_high_res_frame,
    unload_whisper_model,
)
from src.ingestion.media_convert import ensure_720p_video, ensure_mp3_audio
from src.ingestion.orchestrator import ingest_file
from src.ingestion.splitter import _SEMANTIC_SPLITTER_CACHE, _get_splitter
from src.ingestion.text import (
    _analyze_page_for_vision,
    process_docx,
    process_pdf,
    process_pptx,
)
from src.ingestion.utils import (
    IngestionCancelled,
    _active_subprocesses,
    _http_session,
    _safe_print,
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
