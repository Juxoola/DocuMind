"""Обработка аудио/видео: транскрибация WhisperX, анализ кадров, Vision."""

import asyncio
import logging
import os
import subprocess
import threading
import uuid

import aiofiles
import cv2
import orjson
import torch
from llama_index.core.schema import TextNode

import config
from routers.shared import get_async_http
from src.gguf.server import unload_all_models
from src.ingestion.splitter import _get_splitter
from src.ingestion.utils import (
    IngestionCancelled,
    cleanup_gpu,
    format_seconds,
)
from src.ingestion.vision import describe_image_with_lmstudio, get_vision_url

logger = logging.getLogger(__name__)


# Кэш моделей WhisperX и монопатчинг ffmpeg для совместимости с imageio-ffmpeg
_whisper_model_cache: dict = {}
_whisper_lock = threading.Lock()

# ── Кэш доступности CUDA в bundled ffmpeg ──────────────────────────────
_ffmpeg_cuda_available: bool | None = None


async def _probe_ffmpeg_cuda(ffmpeg: str) -> bool:
    """Один раз проверяет, поддерживает ли bundled ffmpeg CUDA hwaccel."""
    global _ffmpeg_cuda_available
    if _ffmpeg_cuda_available is not None:
        return _ffmpeg_cuda_available
    try:
        proc = await asyncio.create_subprocess_exec(
            ffmpeg,
            "-hide_banner",
            "-hwaccels",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
        _ffmpeg_cuda_available = b"cuda" in out.lower()
    except Exception:
        _ffmpeg_cuda_available = False
    logger.info(
        f"[FFmpeg] CUDA hwaccel: {'да' if _ffmpeg_cuda_available else 'нет (CPU-fallback)'}"
    )
    return _ffmpeg_cuda_available


# Замена ffmpeg в WhisperX на bundled imageio-ffmpeg
def _patch_whisperx_ffmpeg():
    try:
        import whisperx.audio as _wa
        from imageio_ffmpeg import get_ffmpeg_exe

        _orig_load = _wa.load_audio
        _ffmpeg_bin = get_ffmpeg_exe()

        def _patched_load(file, sr=_wa.SAMPLE_RATE):
            import numpy as np

            cmd = [
                _ffmpeg_bin,
                "-nostdin",
                "-threads",
                "0",
                "-i",
                file,
                "-f",
                "s16le",
                "-ac",
                "1",
                "-acodec",
                "pcm_s16le",
                "-ar",
                str(sr),
                "-",
            ]
            out = subprocess.run(cmd, capture_output=True, check=True).stdout
            return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

        _wa.load_audio = _patched_load
        logger.info("[WhisperX] ffmpeg заменён на imageio-ffmpeg")
    except Exception as e:
        logger.warning(f"[WhisperX] Не удалось патчить ffmpeg: {e}")


_patch_whisperx_ffmpeg()


# Ленивая загрузка моделей WhisperX с кэшированием по (model, device, compute_type)
async def get_or_load_whisper(
    model_name: str = "large-v2", device: str = "cuda", compute_type: str = "int8"
):

    def _load():
        key = (model_name, device, compute_type)
        with _whisper_lock:
            if key in _whisper_model_cache:
                return _whisper_model_cache[key]
        import whisperx

        logger.info(f"[WhisperX] Загрузка модели {model_name} ({device}, {compute_type})...")
        logging.disable(logging.INFO)
        try:
            model = whisperx.load_model(
                model_name,
                device,
                compute_type=compute_type,
                download_root=os.path.join(config.BASE_DIR, "models", "whisper"),
            )
        finally:
            logging.disable(logging.NOTSET)
        with _whisper_lock:
            _whisper_model_cache[key] = model
        return model

    return await asyncio.to_thread(_load)


async def unload_whisper_model():

    def _unload():
        with _whisper_lock:
            if not _whisper_model_cache:
                return
            logger.info(f"[WhisperX] Выгрузка {len(_whisper_model_cache)} кешированных моделей...")
            _whisper_model_cache.clear()
            cleanup_gpu()

    await asyncio.to_thread(_unload)


# Извлечение кадра из видео через FFmpeg (CUDA или CPU-fallback)
async def save_high_res_frame(video_path, time_sec, output_path):
    try:
        from imageio_ffmpeg import get_ffmpeg_exe

        ffmpeg = get_ffmpeg_exe()
        has_cuda = await _probe_ffmpeg_cuda(ffmpeg)

        cmd = [ffmpeg, "-y"]
        if has_cuda:
            cmd += ["-hwaccel", "cuda"]
        cmd += [
            "-ss",
            str(time_sec),
            "-i",
            video_path,
            "-vframes",
            "1",
            "-vf",
            "scale=-2:720",
            "-q:v",
            "4",
            output_path,
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL
        )
        await asyncio.wait_for(proc.wait(), timeout=30)
    except Exception as e:
        logger.warning(f"Ошибка FFmpeg при сохранении кадра: {e}")


# Основной конвейер: транскрибация WhisperX → анализ кадров видео → Vision-описание
async def process_audio_video(
    file_path,
    images_dir,
    is_video=False,
    progress_cb=None,
    llm_settings=None,
    cancel_check=None,
    notebook_id=None,
    keep_vision_alive=False,
    keep_whisper_alive=False,
):

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

    prog(15, "Загрузка модели WhisperX...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        import whisperx

        model = await get_or_load_whisper("large-v2", device, "int8")
        prog(18, "Загрузка аудио...")

        def _transcribe():
            audio = whisperx.load_audio(file_path)
            dur = len(audio) / 16000

            def _whisper_progress(pct):
                pct_int = int(pct)
                if pct_int <= 100:
                    prog(20 + int(pct_int * 0.38), f"Транскрибация: {pct_int}%")

            result = model.transcribe(audio, batch_size=16, progress_callback=_whisper_progress)
            segs = []
            for seg in result.get("segments", []):
                segs.append({"start": seg["start"], "end": seg["end"], "text": seg["text"].strip()})
            return segs, dur

        prog(20, "VAD + транскрибация ...")
        transcript_data, _duration_sec = await asyncio.to_thread(_transcribe)
    except Exception as e:
        logger.error(f"WhisperX error: {e}")
    finally:
        if not keep_whisper_alive:
            await unload_whisper_model()

    if transcript_data:
        chunk_text = ""
        chunk_start = 0
        for i, seg in enumerate(transcript_data):
            if not chunk_text:
                chunk_start = seg["start"]
            chunk_text += f"[{seg['start']:.1f}s] {seg['text']} "
            if (seg["end"] - chunk_start > 60) or (i == len(transcript_data) - 1):
                nodes.append(
                    TextNode(
                        text=f"Транскрипт {file_name}:\n{chunk_text.strip()}",
                        metadata={"file_name": file_name, "start": chunk_start},
                    )
                )
                chunk_text = ""
    prog(60, "Транскрибация завершена")

    # Детекция сцен видео: histogram comparison через OpenCV
    if is_video:
        prog(62, "Поиск сцен в видео...")

        # Histogram-based scene detection: сравниваем HSV-гистограммы
        # соседних кадров. Смена фиксируется при превышении порога
        # _bhattacharyya, с debounce min_scene_len кадров.
        def _detect_scenes_cv2():
            _HIST_THRESH = 0.40
            _MIN_SCENE_LEN = 20
            _CHECK_EVERY = 2  # проверять каждый N-й кадр для скорости
            _HIST_SIZE = [64, 64, 64]
            _H_RANGES = [0, 180]
            _SV_RANGES = [0, 256]
            _RANGES = _H_RANGES + _SV_RANGES + _SV_RANGES

            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                return []
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            prev_hist = None
            scenes = []  # [(start_sec, end_sec)]
            last_cut_frame = -_MIN_SCENE_LEN
            frame_idx = 0
            last_hist_frame = -_CHECK_EVERY

            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_idx - last_hist_frame < _CHECK_EVERY:
                    frame_idx += 1
                    continue
                last_hist_frame = frame_idx

                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                hist = cv2.calcHist([hsv], [0, 1, 2], None, _HIST_SIZE, _RANGES)
                cv2.normalize(hist, hist)

                if prev_hist is not None and (frame_idx - last_cut_frame) >= _MIN_SCENE_LEN:
                    diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
                    if diff >= _HIST_THRESH:
                        sec = frame_idx / fps
                        if scenes:
                            scenes[-1] = (scenes[-1][0], sec)
                        scenes.append((sec, total / fps))
                        last_cut_frame = frame_idx

                prev_hist = hist
                frame_idx += 1

            cap.release()
            return scenes

        try:
            scene_list = await asyncio.to_thread(_detect_scenes_cv2)
        except Exception as e:
            logger.error(f"Ошибка детекции сцен: {e}")
            scene_list = []

        frame_list = []
        n_scenes = len(scene_list)
        for i, (start_sec, _end_sec) in enumerate(scene_list):
            img_path = os.path.join(images_dir, f"v_{uuid.uuid4().hex[:6]}.jpg")
            await save_high_res_frame(file_path, start_sec, img_path)
            frame_list.append((img_path, start_sec))
            if (i + 1) % 10 == 0 or i == n_scenes - 1:
                prog(
                    62 + int(((i + 1) / max(n_scenes, 1)) * 3),
                    f"Извлечено кадров: {i + 1}/{n_scenes}",
                )

        n = len(frame_list)
        shared_llm_url = None
        if n > 0:
            prog(65, f"Запуск Vision-сервера для описания {n} кадров...")
            shared_llm_url = await get_vision_url(llm_settings, progress_cb=prog)

        v_conc = int(llm_settings.get("vision_concurrency") or config.VISION_CONCURRENCY)
        VISION_BATCH_SIZE = 100

        sem = asyncio.Semaphore(v_conc)
        splitter = _get_splitter()
        results = []

        async def _describe(idx, path, t):
            nonlocal results
            if _is_cancelled():
                raise IngestionCancelled(f"Cancelled at frame {idx}/{n}")
            async with sem:
                desc = await describe_image_with_lmstudio(
                    path, llm_settings, shared_llm_url, cancel_check=cancel_check
                )
                results.append((idx, path, t, desc))

        try:
            for batch_start in range(0, n, VISION_BATCH_SIZE):
                if _is_cancelled():
                    raise IngestionCancelled(f"Cancelled at batch {batch_start}")
                batch_end = min(batch_start + VISION_BATCH_SIZE, n)
                batch_tasks = [
                    _describe(idx, path, t)
                    for idx, (path, t) in enumerate(frame_list[batch_start:batch_end], batch_start)
                ]
                await asyncio.gather(*batch_tasks)

                if batch_end < n:
                    try:
                        http = await get_async_http()
                        resp = await http.get(f"{shared_llm_url}/health", timeout=2)
                        resp.raise_for_status()
                    except Exception:
                        logger.warning("[Vision] Сервер unhealthy, перезапуск...")
                        await unload_all_models(role="vision")
                        shared_llm_url = await get_vision_url(llm_settings)
                        if not shared_llm_url:
                            logger.warning("[Vision] Не удалось перезапустить vision-сервер")
                            break
        except IngestionCancelled:
            raise

        results.sort(key=lambda x: x[0])
        for idx, path, t, desc in results:
            full_text = f"Кадр {file_name} [{format_seconds(t)}]: {desc}"
            if len(full_text) <= config.GGUF_CTX_EMBED_CHARS:
                nodes.append(
                    TextNode(
                        text=full_text,
                        metadata={"file_name": file_name, "image_path": path, "time": t},
                    )
                )
            else:
                desc_nodes = splitter.get_nodes_from_documents(
                    [
                        TextNode(
                            text=full_text,
                            metadata={
                                "file_name": file_name,
                                "image_path": path,
                                "time": t,
                            },
                        )
                    ]
                )
                nodes.extend(desc_nodes)
            frame_data.append({"time": t, "image_path": path, "description": desc})
            prog(65 + int((idx + 1) / n * 22) if n else 87, f"Описание: {idx + 1}/{n}")

        results.clear()
        import gc

        gc.collect()

        if shared_llm_url and not keep_vision_alive:
            await unload_all_models(role="vision")

    metadata_json = {
        "file_name": file_name,
        "is_video": is_video,
        "transcript": transcript_data,
        "frames": frame_data,
    }

    metadata_path = os.path.join(os.path.dirname(file_path), f"{file_name}.json")
    async with aiofiles.open(metadata_path, "w", encoding="utf-8") as f:
        await f.write(orjson.dumps(metadata_json, option=orjson.OPT_INDENT_2).decode())
    return nodes
