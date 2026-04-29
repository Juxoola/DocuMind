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
# Python 3's hasattr() подавляет только AttributeError, но speechbrain's LazyModule
# бросает ImportError — поэтому inspect.stack() в PyTorch Lightning падает.
# Решение: (1) патчим inspect.getmodule чтобы он был exception-safe,
#           (2) регистрируем заглушки для всех опциональных speechbrain-интеграций.

import inspect as _inspect_module

_orig_getmodule = _inspect_module.getmodule

def _safe_getmodule(obj, filename=None):
    """Защищённая версия inspect.getmodule — не падает на ленивых загрузчиках."""
    try:
        return _orig_getmodule(obj, filename)
    except Exception:
        return None

_inspect_module.getmodule = _safe_getmodule

# Заглушки для всех известных опциональных зависимостей speechbrain
_SPEECHBRAIN_OPTIONAL_STUBS = [
    'k2',
    'flair',
    'speechbrain.integrations.k2_fsa',
    'speechbrain.integrations.nlp',
    'speechbrain.integrations.nlp.flair_embeddings',
    'speechbrain.k2_integration',
    'speechbrain.wordemb',
    'speechbrain.lobes.models.huggingface_transformers',
]
for _stub_name in _SPEECHBRAIN_OPTIONAL_STUBS:
    if _stub_name not in sys.modules:
        sys.modules[_stub_name] = types.ModuleType(_stub_name)

import config

# --- Фикс для Windows DLL (cublas64_12.dll) ---
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

def get_image_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def describe_image_with_lmstudio(image_path):
    """Отправляет картинку в локальный LM Studio для получения текстового описания."""
    base64_img = get_image_base64(image_path)
    url = f"{config.LM_STUDIO_URL}/chat/completions"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "gpt-4o", # Обход проверки имени
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Опиши всё, что видишь на этом слайде/кадре: схемы, графики, таблицы, а также весь видимый текст. Если на кадре есть текстовые тезисы или определения, обязательно выпиши их."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
                ]
            }
        ],
        "temperature": 0.1,
        "max_tokens": 300
    }
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Ошибка при описании картинки {image_path}: {e}")
        return "Изображение без описания."

_whisperx_model = None
def get_whisperx_model():
    global _whisperx_model
    if _whisperx_model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Для CPU float16 не поддерживается, выбираем int8. Для GPU можно float16.
        compute_type = "float16" if device == "cuda" else "int8"
        print(f"Загрузка WhisperX на {device.upper()} с {compute_type}...")
        _whisperx_model = whisperx.load_model("large-v3", device, compute_type=compute_type)
    return _whisperx_model

def process_audio_video(file_path, images_dir, is_video=False, progress_cb=None):
    """Транскрибирует аудио/видео. Для видео извлекает кадры при смене слайда (дебаунс)."""
    def prog(pct, msg):
        print(f"  [{pct}%] {msg}")
        if progress_cb: progress_cb(pct, msg)

    nodes = []
    file_name = os.path.basename(file_path)

    prog(15, "Загрузка модели транскрибации...")
    model = get_whisperx_model()
    prog(20, "Транскрибация речи (это займёт время)...")
    audio = whisperx.load_audio(file_path)

    transcript = ""
    transcript_data = []
    try:
        result = model.transcribe(audio, batch_size=16)
        for seg in result.get('segments', []):
            transcript += f"[{seg['start']:.2f}s -> {seg['end']:.2f}s] {seg['text'].strip()}\n"
            transcript_data.append({"start": seg['start'], "end": seg['end'], "text": seg['text'].strip()})
    except Exception as e:
        print(f"WhisperX VAD упал ({e}). Резервный движок (faster-whisper)...")
        from faster_whisper import WhisperModel
        device = "cuda" if torch.cuda.is_available() else "cpu"
        fw = WhisperModel("large-v3", device=device, compute_type="float16" if device=="cuda" else "int8")
        for seg, _ in [fw.transcribe(audio, vad_filter=True)]:
            for s in seg:
                transcript += f"[{s.start:.2f}s -> {s.end:.2f}s] {s.text.strip()}\n"
                transcript_data.append({"start": s.start, "end": s.end, "text": s.text.strip()})

    if transcript.strip():
        nodes.append(TextNode(
            text=f"Транскрипт аудио/видео {file_name}:\n{transcript}",
            metadata={"file_name": file_name}
        ))
    prog(60, "Транскрибация завершена")

    if is_video:
        prog(62, "Извлечение ключевых кадров (дебаунс)...")
        frame_data = []
        cap = cv2.VideoCapture(file_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0: fps = 25

        # ── Настройки детекции ──────────────────────────────────────────────
        PIXEL_THR    = 15    # минимальный diff канала
        UPDATE_PCT   = 0.002 # 0.2% — инкрементальное изменение (дорисовка схемы) -> ПЕРЕЗАПИСЬ кадра
        NEW_SLIDE_PCT= 0.04  # 4.0% — сильное изменение -> НОВЫЙ кадр
        MOTION_PCT   = 0.002 # 0.2% — порог движения
        STABLE_WAIT  = int(fps * 3.0)  # 3 сек без движений -> фиксируем результат
        CHECK_STEP   = max(1, int(fps * 1.0))
        COMPARE_SIZE = (320, 180)

        prev_saved_thumb = None    # Эталон последнего сохраненного/обновленного кадра
        last_seen_thumb  = None    
        stable_since     = 0       
        frame_count      = 0
        frame_list       = []      # [(image_path, time_sec)]

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % CHECK_STEP == 0:
                thumb = cv2.resize(frame, COMPARE_SIZE)

                if last_seen_thumb is None:
                    last_seen_thumb = thumb
                    stable_since = frame_count
                    
                    img_name = f"video_frame_{uuid.uuid4().hex[:8]}.jpg"
                    img_path = os.path.join(images_dir, img_name)
                    cv2.imwrite(img_path, frame)
                    frame_list.append((img_path, 0.0))
                    prev_saved_thumb = thumb
                else:
                    diff_motion = cv2.absdiff(thumb, last_seen_thumb)
                    motion_pct = float(np.sum(diff_motion > PIXEL_THR)) / diff_motion.size
                    
                    if motion_pct >= MOTION_PCT:
                        # Движение продолжается
                        stable_since = frame_count
                    else:
                        if frame_count - stable_since >= STABLE_WAIT:
                            if prev_saved_thumb is not None:
                                diff_saved = cv2.absdiff(thumb, prev_saved_thumb)
                                saved_pct = float(np.sum(diff_saved > PIXEL_THR)) / diff_saved.size
                                
                                if saved_pct >= UPDATE_PCT:
                                    img_name = f"video_frame_{uuid.uuid4().hex[:8]}.jpg"
                                    img_path = os.path.join(images_dir, img_name)
                                    cv2.imwrite(img_path, frame)
                                    
                                    if saved_pct >= NEW_SLIDE_PCT:
                                        # Полностью новый слайд
                                        frame_list.append((img_path, frame_count / fps))
                                    else:
                                        # Инкрементальное добавление (дорисовали схему) -> ПЕРЕЗАПИСЫВАЕМ
                                        old_path = frame_list[-1][0]
                                        if os.path.exists(old_path):
                                            try: os.remove(old_path)
                                            except: pass
                                        frame_list[-1] = (img_path, frame_count / fps)
                                        
                                    prev_saved_thumb = thumb
                                    
                            stable_since = frame_count

                last_seen_thumb = thumb

            frame_count += 1

        cap.release()
        print(f"Извлечено кадров (итоговые состояния): {len(frame_list)}")

        # ── Параллельное описание кадров ────────────────────────────────────
        n = len(frame_list)
        prog(65, f"Описание {n} кадров (параллельно)...")

        def _describe(args):
            img_path, t = args
            return img_path, t, describe_image_with_lmstudio(img_path)

        done = 0
        with ThreadPoolExecutor(max_workers=3) as exe:
            futs = {exe.submit(_describe, item): item for item in frame_list}
            for fut in as_completed(futs):
                img_path, t, desc = fut.result()
                nodes.append(TextNode(
                    text=f"Кадр из видео {file_name} на {t:.1f} секунде. Описание кадра: {desc}",
                    metadata={"file_name": file_name, "image_path": img_path}
                ))
                frame_data.append({
                    "time": t,
                    "image_path": img_path,
                    "description": desc
                })
                done += 1
                prog(65 + int(done / n * 22) if n else 87, f"Описание кадров: {done}/{n}")
                
        # Сортируем frame_data по времени
        frame_data.sort(key=lambda x: x["time"])

    # Сохраняем метаданные для фронтенда (плеера)
    metadata_json = {
        "file_name": file_name,
        "is_video": is_video,
        "transcript": transcript_data,
        "frames": frame_data if is_video else []
    }
    data_dir = os.path.dirname(file_path)
    json_path = os.path.join(data_dir, f"{file_name}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata_json, f, ensure_ascii=False, indent=2)

    # Очистка памяти
    del audio
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return nodes

def process_pdf(file_path, images_dir):
    print(f"Обработка PDF: {file_path}")
    nodes = []
    file_name = os.path.basename(file_path)
    doc = fitz.open(file_path)
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text()
        if text.strip():
            nodes.append(TextNode(
                text=f"Текст из PDF {file_name}, страница {page_num+1}:\n{text}",
                metadata={"file_name": file_name, "page": page_num + 1}
            ))
            
        drawings = page.get_drawings()
        has_real_image = len(page.get_images()) > 0
        has_vector_diagram = False
        
        # Эвристика: отличаем сложную диаграмму (примитивы Word) от простой рамки блока кода
        if not has_real_image and len(drawings) > 0:
            complex_items = 0
            rect_items = 0
            for d in drawings:
                for item in d.get("items", []):
                    if item[0] in ("l", "c", "qu"):
                        complex_items += 1
                    elif item[0] == "re":
                        rect_items += 1
            
            # Если много линий/кривых (блок-схема) или много прямоугольников (сложная таблица/схема)
            if complex_items > 10 or rect_items > 20 or (complex_items > 2 and rect_items > 5):
                has_vector_diagram = True

        if not has_real_image and not has_vector_diagram:
            continue
            
        pix = page.get_pixmap(dpi=150)
        image_bytes = pix.tobytes("png")
        image_filename = f"pdf_page_{page_num+1}_{uuid.uuid4().hex[:8]}.png"
        image_path = os.path.join(images_dir, image_filename)
        
        with open(image_path, "wb") as f:
            f.write(image_bytes)
            
        desc = describe_image_with_lmstudio(image_path)
        
        nodes.append(TextNode(
            text=f"Изображение/Схема из PDF {file_name}, страница {page_num+1}. Описание: {desc}",
            metadata={"file_name": file_name, "image_path": image_path, "page": page_num + 1}
        ))
            
    return nodes

def process_pptx(file_path, images_dir):
    print(f"Обработка презентации: {file_path}")
    nodes = []
    file_name = os.path.basename(file_path)
    prs = Presentation(file_path)
    for i, slide in enumerate(prs.slides):
        slide_text = []
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                slide_text.append(shape.text)
                
            if hasattr(shape, "image"):
                image = shape.image
                image_bytes = image.blob
                image_filename = f"pptx_img_{uuid.uuid4().hex[:8]}.{image.ext}"
                image_path = os.path.join(images_dir, image_filename)
                with open(image_path, "wb") as f:
                    f.write(image_bytes)
                    
                desc = describe_image_with_lmstudio(image_path)
                nodes.append(TextNode(
                    text=f"Изображение со слайда {i+1} презентации {file_name}. Описание: {desc}",
                    metadata={"file_name": file_name, "image_path": image_path}
                ))
                
        if slide_text:
            text_content = "\n".join(slide_text)
            nodes.append(TextNode(
                text=f"Текст со слайда {i+1} презентации {file_name}:\n{text_content}",
                metadata={"file_name": file_name}
            ))
            
    return nodes

def process_docx(file_path, images_dir):
    print(f"Обработка DOCX: {file_path}")
    import docx
    nodes = []
    file_name = os.path.basename(file_path)
    try:
        doc = docx.Document(file_path)
        
        full_text = []
        for para in doc.paragraphs:
            if para.text.strip():
                full_text.append(para.text)
                
        if full_text:
            text_content = "\n".join(full_text)
            nodes.append(TextNode(
                text=f"Текст из документа Word {file_name}:\n{text_content}",
                metadata={"file_name": file_name}
            ))
            
        for i, rel in enumerate(doc.part.rels.values()):
            if "image" in rel.target_ref:
                try:
                    img_bytes = rel.target_part.blob
                    ext = rel.target_ref.split('.')[-1]
                    image_filename = f"docx_img_{uuid.uuid4().hex[:8]}.{ext}"
                    image_path = os.path.join(images_dir, image_filename)
                    with open(image_path, "wb") as f:
                        f.write(img_bytes)
                        
                    desc = describe_image_with_lmstudio(image_path)
                    nodes.append(TextNode(
                        text=f"Изображение из документа Word {file_name}. Описание: {desc}",
                        metadata={"file_name": file_name, "image_path": image_path}
                    ))
                except Exception as img_e:
                    print(f"Ошибка извлечения картинки: {img_e}")
    except Exception as e:
        print(f"Ошибка DOCX: {e}")
        
    return nodes

def ingest_file(file_path, notebook_id, progress_cb=None):
    paths = config.get_notebook_paths(notebook_id)
    images_dir = paths["images"]
    os.makedirs(images_dir, exist_ok=True)

    ext = os.path.splitext(file_path)[1].lower()
    file_name = os.path.basename(file_path)
    if ext in ['.mp4', '.avi', '.mkv']:
        return process_audio_video(file_path, images_dir, is_video=True, progress_cb=progress_cb)
    elif ext in ['.mp3', '.wav', '.m4a']:
        return process_audio_video(file_path, images_dir, is_video=False, progress_cb=progress_cb)
    elif ext == '.pdf':
        if progress_cb: progress_cb(50, "Обработка PDF...")
        return process_pdf(file_path, images_dir)
    elif ext == '.pptx':
        if progress_cb: progress_cb(50, "Обработка презентации...")
        return process_pptx(file_path, images_dir)
    elif ext == '.docx':
        if progress_cb: progress_cb(50, "Обработка документа Word...")
        return process_docx(file_path, images_dir)
    elif ext == '.txt':
        if progress_cb: progress_cb(50, "Чтение текстового файла...")
        with open(file_path, "r", encoding="utf-8") as f:
            return [TextNode(text=f.read(), metadata={"file_name": file_name})]
    else:
        print(f"Неподдерживаемый формат: {ext}")
        return []
