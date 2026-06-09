"""Медиа-конвертация: 720p ресайз видео, конвертация аудио в mp3."""

import concurrent.futures
import logging
import os
import shutil
import subprocess

from src.ingestion.utils import (
    IngestionCancelled,
    _safe_print,
    format_seconds,
    register_subprocess,
    unregister_subprocess,
)

logger = logging.getLogger(__name__)


def ensure_720p_video(file_path, prog_cb=None, cancel_check=None, notebook_id=None):
    """Оптимизирует видео до 720p HEVC (NVENC), турбо для длинных видео."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in [".mp4", ".avi", ".mkv", ".mov"]:
        return file_path
    from imageio_ffmpeg import get_ffmpeg_exe

    ffmpeg = get_ffmpeg_exe()

    def _is_cancelled():
        return bool(cancel_check and cancel_check())

    def get_duration(path):
        try:
            cmd = [ffmpeg, "-hide_banner", "-i", path]
            res = subprocess.run(cmd, capture_output=True, timeout=30)
            stderr = (res.stderr or b"").decode("utf-8", errors="ignore")
            for line in stderr.split("\n"):
                if "Duration" in line:
                    time_str = line.split("Duration: ")[1].split(",")[0]
                    h, m, s = time_str.split(":")
                    return float(h) * 3600 + float(m) * 60 + float(s)
        except subprocess.TimeoutExpired:
            _safe_print(f"[ensure_720p_video] WARNING get_duration timeout для {os.path.basename(path)} (30с)")
        except Exception:
            pass  # best-effort
        return 0

    _safe_print(f"[ensure_720p_video] Начало: {os.path.basename(file_path)}")
    duration = get_duration(file_path)
    temp_final = file_path + ".720p.mp4"
    use_turbo = True if duration == 0 else duration >= 120

    if not use_turbo:
        if _is_cancelled():
            raise IngestionCancelled("Cancelled before 720p encode")
        if prog_cb:
            prog_cb(5, "Оптимизация видео (GPU)...")
        cmd = [ffmpeg, "-y", "-hide_banner", "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
               "-i", file_path, "-vf", "scale_cuda=1280:720:format=yuv420p", "-c:v", "hevc_nvenc",
               "-preset", "p1", "-rc", "constqp", "-qp", "30", "-pix_fmt", "yuv420p",
               "-tag:v", "hvc1", "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", temp_final]
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if notebook_id is not None:
            register_subprocess(notebook_id, proc)
        try:
            proc.wait()
        finally:
            if notebook_id is not None:
                unregister_subprocess(notebook_id, proc)
        if _is_cancelled():
            try:
                os.remove(temp_final)
            except Exception:
                pass  # best-effort
            raise IngestionCancelled("Cancelled during 720p encode")
    else:
        if _is_cancelled():
            raise IngestionCancelled("Cancelled before turbo encode")
        if prog_cb:
            prog_cb(5, "Турбо-оптимизация (Параллельный GPU)...")
        num_workers = 4
        seg_len = duration / num_workers
        temp_dir = file_path + "_parts"
        os.makedirs(temp_dir, exist_ok=True)

        def encode_seg(idx):
            if _is_cancelled():
                return None
            out_part = os.path.join(temp_dir, f"part_{idx}.mp4")
            cmd = [ffmpeg, "-y", "-hide_banner", "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
                   "-ss", str(idx * seg_len), "-t", str(seg_len), "-i", file_path,
                   "-vf", "scale_cuda=1280:720:format=yuv420p", "-c:v", "hevc_nvenc",
                   "-preset", "p1", "-rc", "constqp", "-qp", "30", "-progress", "pipe:1", "-an", out_part]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
            if notebook_id is not None:
                register_subprocess(notebook_id, proc)
            try:
                last_pct = -1
                for line in proc.stdout:
                    if _is_cancelled():
                        proc.kill()
                        return None
                    line = line.strip()
                    if line.startswith("out_time_ms="):
                        try:
                            time_ms = int(line.split("=", 1)[1])
                            pct = min(99, int(time_ms / 10000 / seg_len))
                            if pct > last_pct and pct % 10 == 0:
                                last_pct = pct
                                if prog_cb:
                                    overall = 5 + ((idx + pct / 100.0) / num_workers) * 3
                                    prog_cb(overall, f"Сегмент {idx+1}/{num_workers}: {pct}%")
                        except (ValueError, ZeroDivisionError):
                            pass
                proc.wait()
            finally:
                if notebook_id is not None:
                    unregister_subprocess(notebook_id, proc)
            return out_part

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as ex:
            parts = []
            for p in ex.map(encode_seg, range(num_workers)):
                if _is_cancelled():
                    _safe_print(f"[Ingestion] Отмена во время турбо-кодирования")
                    ex.shutdown(wait=False, cancel_futures=True)
                    raise IngestionCancelled("Cancelled during turbo encode")
                parts.append(p)

        if _is_cancelled():
            raise IngestionCancelled("Cancelled after turbo encode")
        if prog_cb:
            prog_cb(8, "Сборка сегментов...")
        list_path = os.path.join(temp_dir, "list.txt")
        with open(list_path, "w") as f:
            for p in parts:
                f.write(f"file '{os.path.abspath(p)}'\n")
        merge_cmd = [ffmpeg, "-y", "-hide_banner", "-f", "concat", "-safe", "0", "-i", list_path,
                     "-i", file_path, "-map", "0:v", "-map", "1:a?", "-c", "copy", "-movflags", "+faststart", temp_final]
        merge_proc = subprocess.Popen(merge_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if notebook_id is not None:
            register_subprocess(notebook_id, merge_proc)
        try:
            merge_proc.wait()
        finally:
            if notebook_id is not None:
                unregister_subprocess(notebook_id, merge_proc)
        if _is_cancelled():
            try:
                os.remove(temp_final)
            except Exception:
                pass  # best-effort
            raise IngestionCancelled("Cancelled after video merge")
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass  # best-effort

    if os.path.exists(temp_final) and os.path.getsize(temp_final) > 1000:
        if os.path.exists(file_path):
            os.remove(file_path)
        new_path = os.path.splitext(file_path)[0] + ".mp4"
        if os.path.exists(new_path) and new_path != temp_final:
            os.remove(new_path)
        os.rename(temp_final, new_path)
        file_path = new_path
        if prog_cb:
            prog_cb(9, "Видео оптимизировано (Турбо)")
    return file_path


def ensure_mp3_audio(file_path, prog_cb=None):
    """Конвертирует аудио в mp3."""
    temp_path = file_path.rsplit(".", 1)[0] + ".mp3"
    from imageio_ffmpeg import get_ffmpeg_exe

    cmd = [get_ffmpeg_exe(), "-y", "-i", file_path, "-acodec", "libmp3lame", "-ab", "128k", temp_path]
    subprocess.run(cmd, capture_output=True)
    if os.path.exists(temp_path):
        os.remove(file_path)
        return temp_path
    return file_path
