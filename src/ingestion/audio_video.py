"""Обработка аудио/видео: транскрибация WhisperX, анализ кадров, Vision."""

import gc
import json
import logging
import os
import subprocess
import threading
import time
import uuid

import cv2
import numpy as np
import torch
from llama_index.core.schema import TextNode

import config
from src.gguf_direct import unload_all_models
from src.ingestion.splitter import _get_splitter
from src.ingestion.utils import (
    IngestionCancelled,
    format_seconds,
    register_subprocess,
    unregister_subprocess,
)
from src.ingestion.vision import describe_image_with_lmstudio, get_vision_url

logger = logging.getLogger(__name__)

# WhisperX кеш
_whisper_model_cache: dict = {}
_whisper_lock = threading.Lock()


def get_or_load_whisper(model_name: str = "medium", device: str = "cuda", compute_type: str = "int8"):
    """Возвращает кешированную WhisperX-модель или загружает её."""
    key = (model_name, device, compute_type)
    with _whisper_lock:
        if key in _whisper_model_cache:
            return _whisper_model_cache[key]
        import whisperx
        logger.info(f"[WhisperX] Загрузка модели {model_name} ({device}, {compute_type})...")
        model = whisperx.load_model(model_name, device, compute_type=compute_type)
        _whisper_model_cache[key] = model
        return model


def unload_whisper_model():
    """Выгружает все кешированные WhisperX-модели."""
    with _whisper_lock:
        if not _whisper_model_cache:
            return
        logger.info(f"[WhisperX] Выгрузка {len(_whisper_model_cache)} кешированных моделей...")
        _whisper_model_cache.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def save_high_res_frame(video_path, time_sec, output_path):
    """Сохраняет кадр из видео в высоком разрешении через ffmpeg."""
    try:
        from imageio_ffmpeg import get_ffmpeg_exe

        cmd = [get_ffmpeg_exe(), "-y", "-hwaccel", "cuda", "-ss", str(time_sec),
               "-i", video_path, "-vframes", "1", "-vf", "scale=-2:720", "-q:v", "4", output_path]
        subprocess.run(cmd, capture_output=True)
    except Exception as e:
        logger.warning(f"Ошибка FFmpeg при сохранении кадра: {e}")


def process_audio_video(file_path, images_dir, is_video=False, progress_cb=None,
                        llm_settings=None, cancel_check=None, notebook_id=None,
                        keep_vision_alive=False, keep_whisper_alive=False):
    """Обрабатывает аудио/видео: транскрибация → извлечение кадров → Vision."""
    def _is_cancelled():
        return bool(cancel_check and cancel_check())

    file_name = os.path.basename(file_path)

    def prog(pct, msg):
        try:
            logger.info(f"  [{pct}%] {msg}")
        except Exception as e:
            logger.debug(f"progress print failed (encoding issue?): {e}")
        if progress_cb:
            progress_cb(pct, msg)

    nodes = []
    transcript_data = []
    frame_data = []

    # 1. ТРАНСКРИБАЦИЯ
    prog(15, "Транскрибация речи (WhisperX)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        import whisperx
        model = get_or_load_whisper("medium", device, "int8")
        audio = whisperx.load_audio(file_path)
        result = model.transcribe(audio, batch_size=16)
        for seg in result.get("segments", []):
            transcript_data.append({"start": seg["start"], "end": seg["end"], "text": seg["text"].strip()})
    except Exception as e:
        logger.error(f"WhisperX error: {e}")
    finally:
        if not keep_whisper_alive:
            unload_whisper_model()

    if transcript_data:
        chunk_text = ""
        chunk_start = 0
        for i, seg in enumerate(transcript_data):
            if not chunk_text:
                chunk_start = seg["start"]
            chunk_text += f"[{seg['start']:.1f}s] {seg['text']} "
            if (seg["end"] - chunk_start > 60) or (i == len(transcript_data) - 1):
                nodes.append(TextNode(
                    text=f"Транскрипт {file_name}:\n{chunk_text.strip()}",
                    metadata={"file_name": file_name, "start": chunk_start},
                ))
                chunk_text = ""
    prog(60, "Транскрибация завершена")

    # 2. АНАЛИЗ ВИДЕО
    if is_video:
        prog(62, "Анализ изменений в видео...")
        cap = cv2.VideoCapture(file_path)
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration_sec = total_frames / fps if fps > 0 else 0
        finally:
            cap.release()

        from imageio_ffmpeg import get_ffmpeg_exe
        ffmpeg = get_ffmpeg_exe()

        PIXEL_THR, UPDATE_PCT, NEW_SLIDE_PCT, MOTION_PCT = 15, 0.002, 0.04, 0.002
        STABLE_WAIT_SEC, CHECK_STEP_SEC = 3.0, 1.0
        COMPARE_SIZE = (320, 180)

        cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-hwaccel", "cuda",
               "-hwaccel_output_format", "cuda", "-i", file_path,
               "-vf", f"fps=1/{CHECK_STEP_SEC},scale_cuda={COMPARE_SIZE[0]}:{COMPARE_SIZE[1]}:format=yuv420p,hwdownload,format=yuv420p,format=bgr24",
               "-f", "image2pipe", "-vcodec", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]

        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10**8)
        if notebook_id is not None:
            register_subprocess(notebook_id, process)
        frame_list = []
        try:
            prev_saved_thumb = None
            last_seen_thumb = None
            stable_since_sec = 0
            current_sec = 0
            chunk_size = COMPARE_SIZE[0] * COMPARE_SIZE[1] * 3
            cancel_check_every = 5
            frame_counter = 0
            while True:
                if frame_counter % cancel_check_every == 0 and _is_cancelled():
                    logger.info(f"[Ingestion] Отмена во время извлечения кадров видео ({format_seconds(current_sec)})")
                    try:
                        process.terminate()
                    except Exception:
                        pass  # best-effort
                    raise IngestionCancelled(f"Cancelled during frame extraction at {format_seconds(current_sec)}")
                frame_counter += 1
                raw_frame = process.stdout.read(chunk_size)
                if not raw_frame or len(raw_frame) != chunk_size:
                    break
                thumb = np.frombuffer(raw_frame, dtype="uint8").reshape((COMPARE_SIZE[1], COMPARE_SIZE[0], 3))
                if last_seen_thumb is None:
                    last_seen_thumb = thumb
                    stable_since_sec = current_sec
                    img_path = os.path.join(images_dir, f"v_{uuid.uuid4().hex[:6]}.jpg")
                    save_high_res_frame(file_path, current_sec, img_path)
                    frame_list.append((img_path, current_sec))
                    prev_saved_thumb = thumb
                else:
                    diff_motion = cv2.absdiff(thumb, last_seen_thumb)
                    motion_pct = float(np.sum(diff_motion > PIXEL_THR)) / diff_motion.size
                    if motion_pct >= MOTION_PCT:
                        stable_since_sec = current_sec
                    else:
                        if current_sec - stable_since_sec >= STABLE_WAIT_SEC:
                            if prev_saved_thumb is not None:
                                diff_saved = cv2.absdiff(thumb, prev_saved_thumb)
                                saved_pct = float(np.sum(diff_saved > PIXEL_THR)) / diff_saved.size
                                if saved_pct >= UPDATE_PCT:
                                    img_path = os.path.join(images_dir, f"v_{uuid.uuid4().hex[:6]}.jpg")
                                    save_high_res_frame(file_path, current_sec, img_path)
                                    if saved_pct >= NEW_SLIDE_PCT:
                                        frame_list.append((img_path, current_sec))
                                    else:
                                        if frame_list:
                                            try:
                                                os.remove(frame_list[-1][0])
                                            except Exception:
                                                pass  # best-effort
                                            frame_list[-1] = (img_path, current_sec)
                                    prev_saved_thumb = thumb
                            stable_since_sec = current_sec
                last_seen_thumb = thumb
                current_sec += CHECK_STEP_SEC
                if int(current_sec) % 5 == 0:
                    prog(62 + int((current_sec / duration_sec) * 3) if duration_sec > 0 else 62,
                         f"Анализ видео: {format_seconds(current_sec)} / {format_seconds(duration_sec)}")
        finally:
            process.stdout.close()
            process.terminate()
            process.wait()
            if notebook_id is not None:
                unregister_subprocess(notebook_id, process)

        # 3. ОПИСАНИЕ КАДРОВ
        n = len(frame_list)
        shared_llm_url = None
        if n > 0:
            prog(65, f"Запуск Vision-сервера для описания {n} кадров...")
            shared_llm_url = get_vision_url(llm_settings, progress_cb=prog)

        v_conc = int(llm_settings.get("vision_concurrency") or config.VISION_CONCURRENCY)
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=v_conc) as executor:
            futures = [executor.submit(describe_image_with_lmstudio, path, llm_settings, shared_llm_url)
                       for path, t in frame_list]
            splitter = _get_splitter()
            try:
                for idx, future in enumerate(futures):
                    if _is_cancelled():
                        logger.info(f"[Ingestion] Отмена во время OCR кадров ({idx}/{n})")
                        executor.shutdown(wait=False, cancel_futures=True)
                        raise IngestionCancelled(f"Cancelled at frame {idx}/{n}")
                    desc = future.result()
                    path, t = frame_list[idx]
                    full_text = f"Кадр {file_name} [{format_seconds(t)}]: {desc}"
                    if len(full_text) <= config.GGUF_CTX_EMBED_CHARS:
                        nodes.append(TextNode(text=full_text, metadata={"file_name": file_name, "image_path": path, "time": t}))
                    else:
                        desc_nodes = splitter.get_nodes_from_documents([TextNode(text=full_text, metadata={"file_name": file_name, "image_path": path, "time": t})])
                        nodes.extend(desc_nodes)
                    frame_data.append({"time": t, "image_path": path, "description": desc})
                    prog(65 + int((idx + 1) / n * 22) if n else 87, f"Описание: {idx+1}/{n}")
            except IngestionCancelled:
                raise

        if shared_llm_url and not keep_vision_alive:
            unload_all_models(role="llm")

    metadata_json = {"file_name": file_name, "is_video": is_video, "transcript": transcript_data, "frames": frame_data}
    with open(os.path.join(os.path.dirname(file_path), f"{file_name}.json"), "w", encoding="utf-8") as f:
        json.dump(metadata_json, f, ensure_ascii=False, indent=2)
    return nodes
