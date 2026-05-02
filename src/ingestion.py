import os
import sys
import types
import warnings
import subprocess
import shutil
import cv2
import uuid
import numpy as np
import fitz  # PyMuPDF
from pptx import Presentation
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
from llama_index.core.schema import TextNode
import base64
import requests
import json
import time
import gc
from llama_index.core.node_parser import SentenceSplitter

# Подавляем шумные предупреждения
warnings.filterwarnings("ignore", message="Module 'speechbrain")
warnings.filterwarnings("ignore", message="torchcodec is not installed")
warnings.filterwarnings("ignore", message="TensorFloat-32")
warnings.filterwarnings("ignore", message=".*speechbrain.*deprecated", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Lightning automatically upgraded.*")

import logging
logging.getLogger("lightning.pytorch.utilities.migration").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
logging.getLogger("whisperx").setLevel(logging.WARNING)

# --- Фикс: inspect.stack() ---
import inspect as _inspect_module
_orig_getmodule = _inspect_module.getmodule
def _safe_getmodule(obj, filename=None):
    try: return _orig_getmodule(obj, filename)
    except Exception: return None
_inspect_module.getmodule = _safe_getmodule

# --- Фикс для Windows DLL ---
try:
    import torch
    lib_dir = os.path.join(os.path.dirname(torch.__file__), 'lib')
    if os.path.exists(lib_dir): os.add_dll_directory(lib_dir)
except: pass

import whisperx
import config

def format_seconds(s):
    h = int(s // 3600); m = int((s % 3600) // 60); sec = int(s % 60)
    return f"{h}:{m:02d}:{sec:02d}" if h > 0 else f"{m}:{sec:02d}"

def get_image_base64(image_path):
    with open(image_path, "rb") as image_file: return base64.b64encode(image_file.read()).decode("utf-8")

def describe_image_with_lmstudio(image_path, llm_settings=None, existing_llm=None):
    prompt = "Проанализируй это изображение (кадр из видео или слайд презентации) и составь его подробное описание на русском языке для системы поиска. ТЕКСТ, ГРАФИКА, ВИЗУАЛ, СМЫСЛ."
    if llm_settings and llm_settings.get("use_gguf_direct") and existing_llm:
        try:
            res = existing_llm.create_chat_completion(
                messages=[{"role":"user","content":[{"type":"text","text":prompt},{"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{get_image_base64(image_path)}"}}]}],
                temperature=0.2, max_tokens=512
            )
            return res["choices"][0]["message"]["content"]
        except Exception as e: return f"Ошибка GGUF: {e}"
    
    api_url = (llm_settings.get("llm_url") if llm_settings else None) or config.LM_STUDIO_URL
    api_key = (llm_settings.get("llm_api_key") if llm_settings else None) or "lm-studio"
    model_name = (llm_settings.get("llm_model") if llm_settings else None) or "gpt-4o"
    try:
        r = requests.post(f"{api_url.rstrip('/')}/chat/completions", headers={"Authorization":f"Bearer {api_key}"}, 
                          json={"model":model_name, "messages":[{"role":"user","content":[{"type":"text","text":prompt},{"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{get_image_base64(image_path)}"}}]}], "temperature":0.2}, timeout=30)
        return r.json()["choices"][0]["message"]["content"]
    except: return "Изображение без описания."

def save_high_res_frame(video_path, time_sec, output_path):
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
        cmd = [get_ffmpeg_exe(), "-y", "-hwaccel", "cuda", "-ss", str(time_sec), "-i", video_path, "-vframes", "1", "-vf", "scale=-2:720", "-q:v", "2", output_path]
        import subprocess
        subprocess.run(cmd, capture_output=True)
    except: pass

def process_audio_video(file_path, images_dir, is_video=False, progress_cb=None, llm_settings=None):
    file_name = os.path.basename(file_path)
    def prog(pct, msg):
        print(f"  [{pct}%] {msg}")
        if progress_cb: progress_cb(pct, msg)
    
    nodes = []
    transcript_data = []
    frame_data = []

    # 1. ТРАНСКРИБАЦИЯ
    prog(15, "Загрузка модели транскрибации (medium)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model = whisperx.load_model("medium", device, compute_type="int8")
        prog(20, "Транскрибация речи...")
        audio = whisperx.load_audio(file_path)
        result = model.transcribe(audio, batch_size=16)
        for seg in result.get('segments', []):
            transcript_data.append({"start": seg['start'], "end": seg['end'], "text": seg['text'].strip()})
    except Exception as e: print(f"WhisperX error: {e}")
    finally:
        if 'model' in locals(): del model
        gc.collect(); torch.cuda.empty_cache()

    if transcript_data:
        chunk_text = ""; chunk_start = 0
        for i, seg in enumerate(transcript_data):
            if not chunk_text: chunk_start = seg["start"]
            chunk_text += f"[{seg['start']:.1f}s] {seg['text']} "
            if (seg["end"] - chunk_start > 30) or (i == len(transcript_data) - 1):
                nodes.append(TextNode(text=f"Транскрипт {file_name}:\n{chunk_text.strip()}", metadata={"file_name":file_name, "start":chunk_start}))
                chunk_text = ""
    prog(60, "Транскрибация завершена")

    # 2. АНАЛИЗ ВИДЕО
    if is_video:
        prog(62, "Извлечение ключевых кадров (CUDA Accelerated)...")
        cap = cv2.VideoCapture(file_path)
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration_sec = total_frames / fps if fps > 0 else 0
        finally: cap.release()

        from imageio_ffmpeg import get_ffmpeg_exe
        ffmpeg = get_ffmpeg_exe()
        PIXEL_THR = 15; UPDATE_PCT = 0.002; NEW_SLIDE_PCT = 0.04; MOTION_PCT = 0.002
        STABLE_WAIT_SEC = 3.0; CHECK_STEP_SEC = 1.0; COMPARE_SIZE = (320, 180)
        
        # Оптимальная команда: Декодирование на GPU, Ресайз на CPU (для скорости пайпа)
        cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-hwaccel", "cuda", "-i", file_path, 
               "-vf", f"fps=1/{CHECK_STEP_SEC},scale={COMPARE_SIZE[0]}:{COMPARE_SIZE[1]}", 
               "-f", "image2pipe", "-vcodec", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
        
        import subprocess
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10**8)
        frame_list = []
        try:
            prev_saved_thumb = None; last_seen_thumb = None; stable_since_sec = 0; current_sec = 0
            chunk_size = COMPARE_SIZE[0] * COMPARE_SIZE[1] * 3
            while True:
                raw_frame = process.stdout.read(chunk_size)
                if not raw_frame or len(raw_frame) != chunk_size: break
                thumb = np.frombuffer(raw_frame, dtype='uint8').reshape((COMPARE_SIZE[1], COMPARE_SIZE[0], 3))
                if int(current_sec) % 10 == 0:
                    prog(62 + int((current_sec / duration_sec) * 3) if duration_sec > 0 else 65, f"Анализ видео: {format_seconds(current_sec)} / {format_seconds(duration_sec)}")
                
                if last_seen_thumb is None:
                    last_seen_thumb = thumb; stable_since_sec = current_sec
                    img_path = os.path.join(images_dir, f"v_{uuid.uuid4().hex[:6]}.jpg")
                    save_high_res_frame(file_path, current_sec, img_path)
                    frame_list.append((img_path, current_sec)); prev_saved_thumb = thumb
                else:
                    diff_motion = cv2.absdiff(thumb, last_seen_thumb)
                    motion_pct = float(np.sum(diff_motion > PIXEL_THR)) / diff_motion.size
                    if motion_pct >= MOTION_PCT: stable_since_sec = current_sec
                    else:
                        if current_sec - stable_since_sec >= STABLE_WAIT_SEC:
                            if prev_saved_thumb is not None:
                                diff_saved = cv2.absdiff(thumb, prev_saved_thumb)
                                saved_pct = float(np.sum(diff_saved > PIXEL_THR)) / diff_saved.size
                                if saved_pct >= UPDATE_PCT:
                                    img_path = os.path.join(images_dir, f"v_{uuid.uuid4().hex[:6]}.jpg")
                                    save_high_res_frame(file_path, current_sec, img_path)
                                    if saved_pct >= NEW_SLIDE_PCT: frame_list.append((img_path, current_sec))
                                    else:
                                        if frame_list: 
                                            try: os.remove(frame_list[-1][0])
                                            except: pass
                                            frame_list[-1] = (img_path, current_sec)
                                    prev_saved_thumb = thumb
                            stable_since_sec = current_sec
                last_seen_thumb = thumb; current_sec += CHECK_STEP_SEC
        finally:
            process.stdout.close(); process.terminate(); process.wait()

        # 3. ОПИСАНИЕ КАДРОВ
        n = len(frame_list)
        prog(65, f"Описание {n} кадров...")
        
        def get_native_image_base64(path):
            try:
                # Ресайз до 448px (родное для Qwen-VL) ускоряет инференс в разы без потери качества
                img = cv2.imread(path)
                img_res = cv2.resize(img, (448, 448))
                _, buffer = cv2.imencode(".jpg", img_res, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                return base64.b64encode(buffer).decode("utf-8")
            except: return get_image_base64(path)

        # Функция-воркер для параллельного ИИ (Золотой Стандарт)
        def vision_worker(images_subset, settings, prompt, out_queue):
            try:
                from llama_cpp import Llama; from llama_cpp.llama_chat_format import Llava15ChatHandler
                g_path = config.resolve_model_path(settings["gguf_model_path"])
                m_path = config.resolve_model_path(settings["gguf_mmproj_path"])
                # 3 воркера + Flash Attention - идеальная загрузка 5080
                local_llm = Llama(model_path=g_path, chat_handler=Llava15ChatHandler(clip_model_path=m_path), 
                                  n_ctx=8192, n_gpu_layers=-1, verbose=False, n_batch=2048, n_threads=4, flash_attn=True)
                
                for img_path, t in images_subset:
                    try:
                        res = local_llm.create_chat_completion(
                            messages=[{"role":"user","content":[{"type":"text","text":prompt},{"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{get_native_image_base64(img_path)}"}}]}],
                            temperature=0.2, max_tokens=512
                        )
                        out_queue.put((img_path, t, res["choices"][0]["message"]["content"]))
                    except: out_queue.put((img_path, t, "Ошибка анализа"))
                del local_llm; gc.collect(); torch.cuda.empty_cache()
            except Exception as e:
                for img_path, t in images_subset: out_queue.put((img_path, t, f"Ошибка ИИ: {e}"))

        prompt = "Проанализируй это изображение (кадр из видео или слайд презентации) и составь его подробное описание на русском языке для системы поиска. ТЕКСТ, ГРАФИКА, ВИЗУАЛ, СМЫСЛ."
        
        if llm_settings and llm_settings.get("use_gguf_direct") and n > 5:
            import multiprocessing as mp
            ctx = mp.get_context('spawn')
            num_workers = min(3, n) # 3 воркера - доказанный идеал для скорости/качества
            chunks = [frame_list[i::num_workers] for i in range(num_workers)]
            queue = ctx.Queue()
            processes = []
            
            for i in range(num_workers):
                p = ctx.Process(target=vision_worker, args=(chunks[i], llm_settings, prompt, queue))
                p.start(); processes.append(p)
            
            done = 0
            while done < n:
                try:
                    img_path, t, desc = queue.get(timeout=600)
                    nodes.append(TextNode(text=f"Кадр {file_name} [{format_seconds(t)}]: {desc}", metadata={"file_name":file_name, "image_path":img_path, "time":t}))
                    frame_data.append({"time":t, "image_path":img_path, "description":desc})
                    done += 1; prog(65 + int(done/n*22), f"Описание: {done}/{n} (Золотой Стандарт ИИ)")
                except: break
            
            for p in processes: p.join()
        else:
            # Обычный режим (мало кадров)
            shared_llm = None
            if llm_settings and llm_settings.get("use_gguf_direct"):
                try:
                    from llama_cpp import Llama; from llama_cpp.llama_chat_format import Llava15ChatHandler
                    g_path = config.resolve_model_path(llm_settings["gguf_model_path"])
                    m_path = config.resolve_model_path(llm_settings["gguf_mmproj_path"])
                    shared_llm = Llama(model_path=g_path, chat_handler=Llava15ChatHandler(clip_model_path=m_path), 
                                       n_ctx=8192, n_gpu_layers=-1, verbose=False, n_batch=2048, n_threads=8, flash_attn=True)
                except: pass

            done = 0
            with ThreadPoolExecutor(max_workers=(1 if shared_llm else 5)) as exe:
                futs = {exe.submit(describe_image_with_lmstudio, path, llm_settings, shared_llm): (path, t) for path, t in frame_list}
                for fut in as_completed(futs):
                    path, t = futs[fut]; desc = fut.result()
                    nodes.append(TextNode(text=f"Кадр {file_name} [{format_seconds(t)}]: {desc}", metadata={"file_name":file_name, "image_path":path, "time":t}))
                    frame_data.append({"time":t, "image_path":path, "description":desc})
                    done += 1; prog(65 + int(done/n*22) if n else 87, f"Описание: {done}/{n}")

    metadata_json = {"file_name": file_name, "is_video": is_video, "transcript": transcript_data, "frames": frame_data}
    with open(os.path.join(os.path.dirname(file_path), f"{file_name}.json"), "w", encoding="utf-8") as f:
        json.dump(metadata_json, f, ensure_ascii=False, indent=2)
    return nodes

def process_pdf(file_path, images_dir, llm_settings=None):
    nodes = []; file_name = os.path.basename(file_path); doc = fitz.open(file_path)
    for page_num in range(len(doc)):
        page = doc.load_page(page_num); text = page.get_text()
        if text.strip(): nodes.append(TextNode(text=f"PDF {file_name} стр {page_num+1}:\n{text}", metadata={"file_name":file_name, "page":page_num+1}))
        if len(page.get_images()) > 0 or len(page.get_drawings()) > 5:
            image_path = os.path.join(images_dir, f"p_{page_num+1}_{uuid.uuid4().hex[:6]}.png")
            page.get_pixmap(dpi=150).save(image_path)
            desc = describe_image_with_lmstudio(image_path, llm_settings)
            nodes.append(TextNode(text=f"Изображение PDF {file_name} стр {page_num+1}: {desc}", metadata={"file_name":file_name, "image_path":image_path, "page":page_num+1}))
    return nodes

def process_pptx(file_path, images_dir, llm_settings=None):
    nodes = []; file_name = os.path.basename(file_path); pdf_path = file_path.rsplit('.', 1)[0] + ".temp.pdf"
    import win32com.client, pythoncom
    try:
        pythoncom.CoInitialize()
        app = win32com.client.Dispatch("Powerpoint.Application")
        deck = app.Presentations.Open(os.path.abspath(file_path), WithWindow=False)
        deck.SaveAs(os.path.abspath(pdf_path), 32); deck.Close(); app.Quit()
        nodes = process_pdf(pdf_path, images_dir, llm_settings)
        os.remove(pdf_path)
    except:
        prs = Presentation(file_path)
        for i, slide in enumerate(prs.slides):
            nodes.append(TextNode(text="\n".join([sh.text for sh in slide.shapes if hasattr(sh, "text")]), metadata={"file_name":file_name, "page":i+1}))
    return nodes

def process_docx(file_path, images_dir, llm_settings=None):
    nodes = []; file_name = os.path.basename(file_path); pdf_path = file_path.rsplit('.', 1)[0] + ".temp.pdf"
    import win32com.client, pythoncom
    try:
        pythoncom.CoInitialize()
        app = win32com.client.Dispatch("Word.Application")
        doc = app.Documents.Open(os.path.abspath(file_path))
        doc.SaveAs(os.path.abspath(pdf_path), 17); doc.Close(); app.Quit()
        nodes = process_pdf(pdf_path, images_dir, llm_settings)
        os.remove(pdf_path)
    except:
        import docx
        nodes.append(TextNode(text="\n".join([p.text for p in docx.Document(file_path).paragraphs]), metadata={"file_name":file_name}))
    return nodes

def ensure_720p_video(file_path, prog_cb=None):
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in ['.mp4', '.avi', '.mkv', '.mov']: return file_path
    
    from imageio_ffmpeg import get_ffmpeg_exe
    ffmpeg = get_ffmpeg_exe()
    
    # 1. Получаем длительность
    def get_duration(path):
        try:
            cmd = [ffmpeg, "-i", path]
            res = subprocess.run(cmd, capture_output=True, text=True)
            for line in res.stderr.split("\n"):
                if "Duration" in line:
                    time_str = line.split("Duration: ")[1].split(",")[0]
                    h, m, s = time_str.split(":")
                    return float(h)*3600 + float(m)*60 + float(s)
        except: pass
        return 0

    duration = get_duration(file_path)
    temp_final = file_path + ".720p.mp4"
    
    # Если видео короткое (< 2 мин), кодируем в один поток, но на GPU
    if duration < 120:
        if prog_cb: prog_cb(5, "Оптимизация видео (GPU)...")
        cmd = [
            ffmpeg, "-y", "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
            "-i", file_path, "-vf", "scale_cuda=1280:720:format=yuv420p",
            "-c:v", "hevc_nvenc", "-preset", "p1", "-rc", "constqp", "-qp", "30",
            "-pix_fmt", "yuv420p", "-tag:v", "hvc1", "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart",
            temp_final
        ]
        subprocess.run(cmd, capture_output=True)
    else:
        # ТУРБО-РЕЖИМ: Параллельное кодирование 4 сегментов
        if prog_cb: prog_cb(5, "Турбо-оптимизация (Параллельный GPU)...")
        num_workers = 4
        seg_len = duration / num_workers
        temp_dir = file_path + "_parts"
        os.makedirs(temp_dir, exist_ok=True)
        
        def encode_seg(idx):
            out_part = os.path.join(temp_dir, f"part_{idx}.mp4")
            cmd = [
                ffmpeg, "-y", "-hwaccel", "cuda", "-hwaccel_output_format", "cuda",
                "-ss", str(idx * seg_len), "-t", str(seg_len), "-i", file_path,
                "-vf", "scale_cuda=1280:720:format=yuv420p",
                "-c:v", "hevc_nvenc", "-preset", "p1", "-rc", "constqp", "-qp", "30",
                "-an", out_part # Без звука для скорости, звук возьмем в конце
            ]
            subprocess.run(cmd, capture_output=True)
            return out_part

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as ex:
            parts = list(ex.map(encode_seg, range(num_workers)))
        
        # Склеиваем видео и добавляем звук из оригинала
        list_path = os.path.join(temp_dir, "list.txt")
        with open(list_path, "w") as f:
            for p in parts: f.write(f"file '{os.path.abspath(p)}'\n")
        
        # Финальная сборка: видео из кусков + аудио из оригинала
        merge_cmd = [
            ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", list_path,
            "-i", file_path, "-map", "0:v", "-map", "1:a?", # Маппим видео из склейки и звук из оригинала
            "-c", "copy", "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", temp_final
        ]
        subprocess.run(merge_cmd, capture_output=True)
        
        # Очистка временных файлов
        try: shutil.rmtree(temp_dir)
        except: pass

    # Замена оригинала
    if os.path.exists(temp_final) and os.path.getsize(temp_final) > 1000:
        if os.path.exists(file_path):
            try: os.remove(file_path)
            except: pass
        new_path = os.path.splitext(file_path)[0] + ".mp4"
        if os.path.exists(new_path) and new_path != temp_final:
            try: os.remove(new_path)
            except: pass
        os.rename(temp_final, new_path)
        file_path = new_path
        if prog_cb: prog_cb(9, "Видео оптимизировано (Турбо)")
    
    return file_path

def ensure_mp3_audio(file_path, prog_cb=None):
    temp_path = file_path.rsplit('.', 1)[0] + ".mp3"
    from imageio_ffmpeg import get_ffmpeg_exe
    cmd = [get_ffmpeg_exe(), "-y", "-i", file_path, "-acodec", "libmp3lame", "-ab", "128k", temp_path]
    import subprocess
    subprocess.run(cmd, capture_output=True)
    if os.path.exists(temp_path):
        os.remove(file_path); return temp_path
    return file_path

def ingest_file(file_path, notebook_id, progress_cb=None, llm_settings=None):
    ext = os.path.splitext(file_path)[1].lower()
    paths = config.get_notebook_paths(notebook_id)
    images_dir = paths["images"]
    os.makedirs(images_dir, exist_ok=True)
    
    if ext in ['.mp4', '.avi', '.mkv', '.mov']: file_path = ensure_720p_video(file_path, progress_cb)
    elif ext in ['.mp3', '.wav', '.m4a']: file_path = ensure_mp3_audio(file_path, progress_cb); ext = ".mp3"
    
    if ext in ['.mp4', '.avi', '.mkv', '.mov', '.mp3']: return process_audio_video(file_path, images_dir, ext != ".mp3", progress_cb, llm_settings)
    if ext == '.pdf': return process_pdf(file_path, images_dir, llm_settings)
    if ext == '.pptx': return process_pptx(file_path, images_dir, llm_settings)
    if ext == '.docx': return process_docx(file_path, images_dir, llm_settings)
    
    # Text fallback
    try:
        with open(file_path, 'r', encoding='utf-8') as f: text = f.read()
    except:
        with open(file_path, 'r', encoding='cp1251') as f: text = f.read()
    return SentenceSplitter(chunk_size=1024).get_nodes_from_documents([TextNode(text=text, metadata={"file_name":os.path.basename(file_path)})])
