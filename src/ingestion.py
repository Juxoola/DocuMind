import os
import sys
import types
import warnings
import cv2
import uuid
import numpy as np
import fitz  # PyMuPDF
from pptx import Presentation
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
from llama_index.core.schema import TextNode
import base64
import requests
import json
import time
import gc
from llama_index.core.node_parser import SentenceSplitter

# Подавляем шумные предупреждения от speechbrain и pyannote
warnings.filterwarnings("ignore", message="Module 'speechbrain")
warnings.filterwarnings("ignore", message="torchcodec is not installed")
warnings.filterwarnings("ignore", message="TensorFloat-32")
warnings.filterwarnings("ignore", message=".*speechbrain.*deprecated", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Lightning automatically upgraded.*")

import logging
logging.getLogger("lightning.pytorch.utilities.migration").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
logging.getLogger("whisperx").setLevel(logging.WARNING)

# --- Фикс: inspect.stack() + speechbrain lazy-loader ---
import inspect as _inspect_module
_orig_getmodule = _inspect_module.getmodule

def _safe_getmodule(obj, filename=None):
    try:
        return _orig_getmodule(obj, filename)
    except Exception:
        return None

_inspect_module.getmodule = _safe_getmodule

_SPEECHBRAIN_OPTIONAL_STUBS = [
    'k2', 'flair', 'speechbrain.integrations.k2_fsa', 'speechbrain.integrations.nlp',
    'speechbrain.integrations.nlp.flair_embeddings', 'speechbrain.k2_integration',
    'speechbrain.wordemb', 'speechbrain.lobes.models.huggingface_transformers',
]
for _stub_name in _SPEECHBRAIN_OPTIONAL_STUBS:
    if _stub_name not in sys.modules:
        sys.modules[_stub_name] = types.ModuleType(_stub_name)

import config

# --- Фикс для Windows DLL ---
try:
    import torch
    lib_dir = os.path.join(os.path.dirname(torch.__file__), 'lib')
    if os.path.exists(lib_dir):
        os.add_dll_directory(lib_dir)
except Exception as e:
    print(f"Подсказка: не удалось добавить torch/lib в DLL ({e})")

# --- Фикс для ffmpeg ---
try:
    import imageio_ffmpeg
    ffmpeg_path = os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe())
    os.environ["PATH"] = ffmpeg_path + os.pathsep + os.environ.get("PATH", "")
except Exception as e:
    print(f"Подсказка: imageio_ffmpeg не инициализирован ({e})")

import whisperx

def format_seconds(s):
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = int(s % 60)
    if h > 0:
        return f"{h}:{m:02d}:{sec:02d}"
    return f"{m}:{sec:02d}"

def get_image_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def describe_image_with_lmstudio(image_path, llm_settings=None, existing_llm=None):
    """Отправляет картинку в локальный LM Studio или использует прямой GGUF для получения описания."""
    base64_img = get_image_base64(image_path)
    
    if llm_settings and llm_settings.get("use_gguf_direct"):
        try:
            from llama_cpp import Llama
            import os
            
            if existing_llm:
                llm = existing_llm
            else:
                gguf_path = llm_settings["gguf_model_path"]
                mmproj_path = llm_settings.get("gguf_mmproj_path")
                if not mmproj_path or not os.path.exists(mmproj_path):
                    return "Изображение без описания (нет mmproj)."
                
                llm = Llama(
                    model_path=os.path.normpath(gguf_path),
                    chat_format="qwen",
                    clip_model_path=os.path.normpath(mmproj_path),
                    n_ctx=2048,
                    n_gpu_layers=-1,
                    verbose=False,
                    type_k=2, type_v=2
                )
            
            image_path_norm = os.path.normpath(image_path)
            prompt = """ВНИМАТЕЛЬНО проанализируй это изображение (слайд презентации или кадр видео). 
Твоя задача — составить максимально подробное описание для поисковой системы.

1. ТЕКСТ: Выпиши ВЕСЬ текст, который видишь, включая заголовки, подписи и мелкий шрифт.
2. ГРАФИКА: Если есть схемы, диаграммы или графики — опиши их структуру, оси, легенду и основные данные/тренды.
3. ВИЗУАЛ: Опиши ключевые изображения, иконки или фотографии.
4. СМЫСЛ: Кратко сформулируй главный тезис этого кадра.

ЗАПРЕЩЕНО давать пустые или отказные ответы типа 'на слайде ничего нет' или 'уточните поиск'. Если слайд пуст, опиши хотя бы фон или логотипы. Пиши только по делу, на русском языке."""
            
            response = llm.create_chat_completion(
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"file://{image_path_norm}"}}
                    ]
                }],
                temperature=0.0,
                max_tokens=500,
            )
            result = response["choices"][0]["message"]["content"]
            if not existing_llm: del llm
            return result
        except Exception as e:
            print(f"Ошибка GGUF Direct {image_path}: {e}")
            return "Изображение без описания."
    
    api_url = (llm_settings.get("llm_url") if llm_settings else None) or config.LM_STUDIO_URL
    api_key = (llm_settings.get("llm_api_key") if llm_settings else None) or "lm-studio"
    model_name = (llm_settings.get("llm_model") if llm_settings else None) or "gpt-4o"

    url = f"{api_url.rstrip('/')}/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    payload = {
        "model": model_name,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "ВНИМАТЕЛЬНО проанализируй это изображение..."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
            ]
        }],
        "temperature": 0.0,
        "max_tokens": 500
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return "Изображение без описания."

def process_audio_video(file_path, images_dir, is_video=False, progress_cb=None, llm_settings=None):
    def prog(pct, msg):
        print(f"  [{pct}%] {msg}")
        if progress_cb: progress_cb(pct, msg)

    nodes = []
    file_name = os.path.basename(file_path)

    prog(15, "Загрузка модели транскрибации...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisperx.load_model("medium", device, compute_type="int8")
    
    prog(20, "Транскрибация речи...")
    audio = whisperx.load_audio(file_path)
    transcript_data = []
    try:
        result = model.transcribe(audio, batch_size=16)
        for seg in result.get('segments', []):
            transcript_data.append({"start": seg['start'], "end": seg['end'], "text": seg['text'].strip()})
    except Exception as e:
        print(f"WhisperX error: {e}")
    finally:
        if 'model' in locals(): del model
        gc.collect()
        if device == "cuda": torch.cuda.empty_cache()

    if transcript_data:
        chunk_text = ""
        chunk_start = 0
        for i, seg in enumerate(transcript_data):
            if not chunk_text: chunk_start = seg["start"]
            chunk_text += f"[{seg['start']:.1f}s] {seg['text']} "
            if (seg["end"] - chunk_start > 30) or (i == len(transcript_data) - 1):
                nodes.append(TextNode(
                    text=f"Транскрипт {file_name} (от {format_seconds(chunk_start)}):\n{chunk_text.strip()}",
                    metadata={"file_name": file_name, "start": chunk_start}
                ))
                chunk_text = ""
    prog(60, "Транскрибация завершена")

    if is_video:
        prog(62, "Извлечение ключевых кадров...")
        frame_data = []
        cap = cv2.VideoCapture(file_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        
        # Настройки детекции
        PIXEL_THR = 15; UPDATE_PCT = 0.002; NEW_SLIDE_PCT = 0.04; MOTION_PCT = 0.002
        STABLE_WAIT = int(fps * 3.0); CHECK_STEP = max(1, int(fps * 1.0))
        COMPARE_SIZE = (320, 180); SAVE_HEIGHT = 720
        
        def resize_to_720p(f):
            h, w = f.shape[:2]
            if h <= SAVE_HEIGHT: return f
            scale = SAVE_HEIGHT / h
            return cv2.resize(f, (int(w * scale), SAVE_HEIGHT), interpolation=cv2.INTER_AREA)
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        prev_saved_thumb = None; last_seen_thumb = None; stable_since = 0; frame_count = 0
        frame_list = []

        while frame_count < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
            ret, frame = cap.read()
            if not ret: break
            
            thumb = cv2.resize(frame, COMPARE_SIZE)
            if last_seen_thumb is None:
                last_seen_thumb = thumb; stable_since = frame_count
                img_path = os.path.join(images_dir, f"video_frame_{uuid.uuid4().hex[:8]}.jpg")
                cv2.imwrite(img_path, resize_to_720p(frame))
                frame_list.append((img_path, 0.0))
                prev_saved_thumb = thumb
            else:
                diff_motion = cv2.absdiff(thumb, last_seen_thumb)
                motion_pct = float(np.sum(diff_motion > PIXEL_THR)) / diff_motion.size
                if motion_pct >= MOTION_PCT: stable_since = frame_count
                else:
                    if frame_count - stable_since >= STABLE_WAIT:
                        if prev_saved_thumb is not None:
                            diff_saved = cv2.absdiff(thumb, prev_saved_thumb)
                            saved_pct = float(np.sum(diff_saved > PIXEL_THR)) / diff_saved.size
                            if saved_pct >= UPDATE_PCT:
                                img_path = os.path.join(images_dir, f"video_frame_{uuid.uuid4().hex[:8]}.jpg")
                                cv2.imwrite(img_path, resize_to_720p(frame))
                                if saved_pct >= NEW_SLIDE_PCT: frame_list.append((img_path, frame_count / fps))
                                else:
                                    old_path = frame_list[-1][0]
                                    if os.path.exists(old_path):
                                        try: os.remove(old_path)
                                        except: pass
                                    frame_list[-1] = (img_path, frame_count / fps)
                                prev_saved_thumb = thumb
                        stable_since = frame_count
            last_seen_thumb = thumb; frame_count += CHECK_STEP
        cap.release()

        # Описание кадров
        n = len(frame_list)
        prog(65, f"Описание {n} кадров...")
        shared_llm = None
        use_direct = llm_settings and llm_settings.get("use_gguf_direct")
        
        if use_direct:
            try:
                from llama_cpp import Llama
                shared_llm = Llama(
                    model_path=os.path.normpath(llm_settings["gguf_model_path"]),
                    chat_format="qwen",
                    clip_model_path=os.path.normpath(llm_settings["gguf_mmproj_path"]),
                    n_ctx=2048, n_gpu_layers=-1, verbose=False, type_k=2, type_v=2
                )
            except Exception as e: print(f"GGUF init error: {e}")

        def _describe(args):
            img_path, t = args
            return img_path, t, describe_image_with_lmstudio(img_path, llm_settings, shared_llm)

        done = 0
        with ThreadPoolExecutor(max_workers=(1 if use_direct else 5)) as exe:
            futs = {exe.submit(_describe, item): item for item in frame_list}
            for fut in as_completed(futs):
                img_path, t, desc = fut.result()
                nodes.append(TextNode(
                    text=f"Кадр из видео {file_name} на {format_seconds(t)}. Описание: {desc}",
                    metadata={"file_name": file_name, "image_path": img_path, "time": t, "start": t}
                ))
                frame_data.append({"time": t, "image_path": img_path, "description": desc})
                done += 1; prog(65 + int(done/n*22) if n else 87, f"Описание: {done}/{n}")
        
        if shared_llm:
            del shared_llm
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        frame_data.sort(key=lambda x: x["time"])

    metadata_json = {"file_name": file_name, "is_video": is_video, "transcript": transcript_data, "frames": (frame_data if is_video else [])}
    with open(os.path.join(os.path.dirname(file_path), f"{file_name}.json"), "w", encoding="utf-8") as f:
        json.dump(metadata_json, f, ensure_ascii=False, indent=2)

    return nodes

def process_pdf(file_path, images_dir, llm_settings=None):
    nodes = []; file_name = os.path.basename(file_path); doc = fitz.open(file_path)
    for page_num in range(len(doc)):
        page = doc.load_page(page_num); text = page.get_text()
        if text.strip():
            nodes.append(TextNode(text=f"Текст из PDF {file_name}, стр {page_num+1}:\n{text}", metadata={"file_name": file_name, "page": page_num+1}))
        if len(page.get_images()) > 0 or len(page.get_drawings()) > 5:
            pix = page.get_pixmap(dpi=150)
            image_path = os.path.join(images_dir, f"pdf_page_{page_num+1}_{uuid.uuid4().hex[:8]}.png")
            pix.save(image_path)
            desc = describe_image_with_lmstudio(image_path, llm_settings)
            nodes.append(TextNode(text=f"Изображение из PDF {file_name}, стр {page_num+1}. Описание: {desc}", metadata={"file_name": file_name, "image_path": image_path, "page": page_num+1}))
    return nodes

def convert_pptx_to_pdf(pptx_path, pdf_path):
    try:
        import win32com.client; import pythoncom
        pythoncom.CoInitialize()
        powerpoint = win32com.client.Dispatch("Powerpoint.Application")
        deck = powerpoint.Presentations.Open(os.path.abspath(pptx_path), WithWindow=False)
        deck.SaveAs(os.path.abspath(pdf_path), 32); deck.Close(); powerpoint.Quit()
        return True
    except: return False

def process_pptx(file_path, images_dir, llm_settings=None):
    nodes = []; file_name = os.path.basename(file_path)
    pdf_path = file_path.rsplit('.', 1)[0] + ".pdf"
    has_pdf = convert_pptx_to_pdf(file_path, pdf_path) if sys.platform == "win32" else False
    prs = Presentation(file_path)
    for i, slide in enumerate(prs.slides):
        title = slide.shapes.title.text if slide.shapes.title else ""
        slide_text = [shape.text for shape in slide.shapes if hasattr(shape, "text")]
        nodes.append(TextNode(text=f"Слайд {i+1}: {title}\n" + "\n".join(slide_text), metadata={"file_name": file_name, "page": i+1}))
    return nodes

def process_docx(file_path, images_dir, llm_settings=None):
    import docx; nodes = []; file_name = os.path.basename(file_path)
    doc = docx.Document(file_path)
    full_text = [p.text for p in doc.paragraphs if p.text.strip()]
    if full_text: nodes.append(TextNode(text=f"Текст из DOCX {file_name}:\n" + "\n".join(full_text), metadata={"file_name": file_name}))
    return nodes

def ingest_file(file_path, notebook_id, progress_cb=None, llm_settings=None):
    paths = config.get_notebook_paths(notebook_id)
    images_dir = paths["images"]; os.makedirs(images_dir, exist_ok=True)
    ext = os.path.splitext(file_path)[1].lower()
    nodes = []
    if ext in ['.mp4', '.avi', '.mkv']: nodes = process_audio_video(file_path, images_dir, True, progress_cb, llm_settings)
    elif ext in ['.mp3', '.wav', '.m4a']: nodes = process_audio_video(file_path, images_dir, False, progress_cb, llm_settings)
    elif ext == '.pdf': nodes = process_pdf(file_path, images_dir, llm_settings)
    elif ext == '.pptx': nodes = process_pptx(file_path, images_dir, llm_settings)
    elif ext == '.docx': nodes = process_docx(file_path, images_dir, llm_settings)
    elif ext == '.txt':
        with open(file_path, "r", encoding="utf-8") as f: nodes = [TextNode(text=f.read(), metadata={"file_name": os.path.basename(file_path)})]
    if nodes:
        splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
        from llama_index.core import Document
        nodes = splitter.get_nodes_from_documents([Document(text=n.text, metadata=n.metadata) for n in nodes])
    return nodes
