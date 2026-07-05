"""Медиа-конвертация: 720p ресайз видео, конвертация аудио в mp3."""

import asyncio
import logging
import os
import shutil

import aiofiles.os

from src.ingestion.utils import (
    IngestionCancelled,
    register_subprocess,
    unregister_subprocess,
)

logger = logging.getLogger(__name__)


# ── Конвертация видео и аудио: подготовка к обработке ──


# ── Конвертация видео в 720p: GPU-режим для коротких, турбо для длинных (>120с) ──
async def ensure_720p_video(file_path, prog_cb=None, cancel_check=None, notebook_id=None):

    ext = os.path.splitext(file_path)[1].lower()
    if ext not in [".mp4", ".avi", ".mkv", ".mov"]:
        return file_path
    from imageio_ffmpeg import get_ffmpeg_exe

    ffmpeg = get_ffmpeg_exe()

    def _is_cancelled():
        return bool(cancel_check and cancel_check())

    async def get_duration(path):
        try:
            cmd = [ffmpeg, "-hide_banner", "-i", path]
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE
            )
            _stdout, stderr_bytes = await asyncio.wait_for(proc.communicate(), timeout=30)
            stderr = (stderr_bytes or b"").decode("utf-8", errors="ignore")
            for line in stderr.split("\n"):
                if "Duration" in line:
                    time_str = line.split("Duration: ")[1].split(",")[0]
                    h, m, s = time_str.split(":")
                    return float(h) * 3600 + float(m) * 60 + float(s)
        except TimeoutError:
            logger.warning(
                f"[ensure_720p_video] WARNING get_duration timeout для {os.path.basename(path)} (30с)"
            )
        except Exception:
            logger.debug(f"[ensure_720p_video] Не удалось получить длительность видео: {os.path.basename(path)}")
        return 0

    logger.info(f"[ensure_720p_video] Начало: {os.path.basename(file_path)}")
    duration = await get_duration(file_path)
    temp_final = file_path + ".720p.mp4"
    use_turbo = True if duration == 0 else duration >= 120

    if not use_turbo:
        if _is_cancelled():
            raise IngestionCancelled("Cancelled before 720p encode")
        if prog_cb:
            prog_cb(5, "Оптимизация видео (GPU)...")
        cmd = [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-hwaccel",
            "cuda",
            "-hwaccel_output_format",
            "cuda",
            "-i",
            file_path,
            "-vf",
            "scale_cuda=1280:720:format=yuv420p",
            "-c:v",
            "hevc_nvenc",
            "-preset",
            "p1",
            "-rc",
            "constqp",
            "-qp",
            "30",
            "-pix_fmt",
            "yuv420p",
            "-tag:v",
            "hvc1",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-movflags",
            "+faststart",
            temp_final,
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL
        )
        if notebook_id is not None:
            register_subprocess(notebook_id, proc)
        try:
            await asyncio.wait_for(proc.wait(), timeout=3600)
        finally:
            if notebook_id is not None:
                unregister_subprocess(notebook_id, proc)
        if _is_cancelled():
            try:
                await aiofiles.os.remove(temp_final)
            except Exception:
                logger.debug(f"[ensure_720p_video] Не удалось удалить промежуточный файл {os.path.basename(temp_final)}")
            raise IngestionCancelled("Cancelled during 720p encode")
    else:
        if _is_cancelled():
            raise IngestionCancelled("Cancelled before turbo encode")
        if prog_cb:
            prog_cb(5, "Турбо-оптимизация (Параллельный GPU)...")
        num_workers = 4
        seg_len = duration / num_workers
        temp_dir = file_path + "_parts"
        await aiofiles.os.makedirs(temp_dir, exist_ok=True)

        async def encode_seg(idx):
            if _is_cancelled():
                return None
            out_part = os.path.join(temp_dir, f"part_{idx}.mp4")
            cmd = [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-hwaccel",
                "cuda",
                "-hwaccel_output_format",
                "cuda",
                "-ss",
                str(idx * seg_len),
                "-t",
                str(seg_len),
                "-i",
                file_path,
                "-vf",
                "scale_cuda=1280:720:format=yuv420p",
                "-c:v",
                "hevc_nvenc",
                "-preset",
                "p1",
                "-rc",
                "constqp",
                "-qp",
                "30",
                "-progress",
                "pipe:1",
                "-an",
                out_part,
            ]
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL
            )
            if notebook_id is not None:
                register_subprocess(notebook_id, proc)
            try:
                last_pct = -1
                while True:
                    line_bytes = await proc.stdout.readline()
                    if not line_bytes:
                        break
                    line = line_bytes.decode("utf-8", errors="ignore").strip()
                    if _is_cancelled():
                        proc.kill()
                        return None
                    if line.startswith("out_time_ms="):
                        try:
                            time_ms = int(line.split("=", 1)[1])
                            pct = min(99, int(time_ms / 10000 / seg_len))
                            if pct > last_pct and pct % 10 == 0:
                                last_pct = pct
                                if prog_cb:
                                    overall = 5 + ((idx + pct / 100.0) / num_workers) * 3
                                    prog_cb(overall, f"Сегмент {idx + 1}/{num_workers}: {pct}%")
                        except (ValueError, ZeroDivisionError):
                            pass
                await asyncio.wait_for(proc.wait(), timeout=3600)
            finally:
                if notebook_id is not None:
                    unregister_subprocess(notebook_id, proc)
            return out_part

        tasks = [encode_seg(idx) for idx in range(num_workers)]
        raw_parts = await asyncio.gather(*tasks)
        parts = [p for p in raw_parts if p is not None]
        if not parts:
            raise IngestionCancelled("All segments cancelled")

        if _is_cancelled():
            raise IngestionCancelled("Cancelled after turbo encode")
        if prog_cb:
            prog_cb(8, "Сборка сегментов...")
        list_path = os.path.join(temp_dir, "list.txt")
        async with aiofiles.open(list_path, "w") as f:
            for p in parts:
                await f.write(f"file '{os.path.abspath(p)}'\n")
        merge_cmd = [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            list_path,
            "-i",
            file_path,
            "-map",
            "0:v",
            "-map",
            "1:a?",
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            temp_final,
        ]
        merge_proc = await asyncio.create_subprocess_exec(
            *merge_cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL
        )
        if notebook_id is not None:
            register_subprocess(notebook_id, merge_proc)
        try:
            await asyncio.wait_for(merge_proc.wait(), timeout=3600)
        finally:
            if notebook_id is not None:
                unregister_subprocess(notebook_id, merge_proc)
        if _is_cancelled():
            try:
                await aiofiles.os.remove(temp_final)
            except Exception:
                logger.debug(f"[ensure_720p_video] Не удалось удалить промежуточный файл после сборки: {os.path.basename(temp_final)}")
            raise IngestionCancelled("Cancelled after video merge")
        try:
            await asyncio.to_thread(shutil.rmtree, temp_dir)
        except Exception:
            logger.debug(f"[ensure_720p_video] Не удалось удалить временную директорию: {temp_dir}")

    if await asyncio.to_thread(os.path.exists, temp_final) and os.path.getsize(temp_final) > 1000:
        if await aiofiles.os.path.exists(file_path):
            await aiofiles.os.remove(file_path)
        new_path = os.path.splitext(file_path)[0] + ".mp4"
        if await asyncio.to_thread(os.path.exists, new_path) and new_path != temp_final:
            await aiofiles.os.remove(new_path)
        await aiofiles.os.rename(temp_final, new_path)
        file_path = new_path
        if prog_cb:
            prog_cb(9, "Видео оптимизировано (Турбо)")
    return file_path


# ── Конвертация аудио в MP3 через libmp3lame ──
async def ensure_mp3_audio(file_path, prog_cb=None):

    temp_path = file_path.rsplit(".", 1)[0] + ".mp3"
    from imageio_ffmpeg import get_ffmpeg_exe

    cmd = [
        get_ffmpeg_exe(),
        "-y",
        "-i",
        file_path,
        "-acodec",
        "libmp3lame",
        "-ab",
        "128k",
        temp_path,
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE
    )
    try:
        await asyncio.wait_for(proc.communicate(), timeout=300)
    except (asyncio.CancelledError, asyncio.TimeoutError):
        try:
            proc.kill()
        except Exception:
            logger.debug("[ensure_mp3_audio] Не удалось завершить процесс ffmpeg при отмене/таймауте")
        raise
    if proc.returncode == 0 and await asyncio.to_thread(os.path.exists, temp_path) and os.path.getsize(temp_path) > 1000:
        await aiofiles.os.remove(file_path)
        logger.info(f"[media_convert] {os.path.basename(file_path)} → mp3")
        return temp_path
    try:
        if await aiofiles.os.path.exists(temp_path):
            await aiofiles.os.remove(temp_path)
    except OSError:
        pass
    logger.warning(
        f"[media_convert] Конвертация в mp3 не удалась, остаётся оригинал: {os.path.basename(file_path)}"
    )
    return file_path
