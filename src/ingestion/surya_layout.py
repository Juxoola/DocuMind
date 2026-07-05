"""Surya layout detection + OCR: определение regions и текста с форматированием."""

import logging
import os
import time
import threading

from PIL import Image

logger = logging.getLogger(__name__)

# Кэш для predictors (создаются один раз)
_layout_predictor = None
_recognition_predictor = None
_inference_manager = None
_surya_lock = threading.Lock()


def _ensure_predictor():
    global _layout_predictor, _inference_manager

    if _layout_predictor is not None:
        return _layout_predictor
    with _surya_lock:
        if _layout_predictor is not None:
            return _layout_predictor
        try:
            llama_binary = os.getenv("LLAMA_CPP_BINARY")
            if not llama_binary:
                import shutil, platform
                exe = "llama-server.exe" if platform.system() == "Windows" else "llama-server"
                local = os.path.join(config.BASE_DIR, "bin", exe)
                if os.path.isfile(local):
                    llama_binary = local
                else:
                    llama_binary = shutil.which("llama-server")
                if llama_binary:
                    os.environ["LLAMA_CPP_BINARY"] = llama_binary

            os.environ.setdefault("SURYA_GUIDED_LAYOUT", "false")
            os.environ.setdefault("SURYA_INFERENCE_PARALLEL", "2")
            os.environ.setdefault("SURYA_INFERENCE_CTX_SIZE", "16384")

            from surya.inference import SuryaInferenceManager
            from surya.layout import LayoutPredictor

            logger.info("[Surya] Инициализация inference manager (llamacpp)...")
            _inference_manager = SuryaInferenceManager(method="llamacpp")
            _layout_predictor = LayoutPredictor(_inference_manager)
            logger.info("[Surya] Layout predictor готов.")
            return _layout_predictor

        except Exception as e:
            logger.error(f"[Surya] Ошибка инициализации layout: {e}")
            return None


def _ensure_recognition():
    global _recognition_predictor, _inference_manager

    if _recognition_predictor is not None:
        return _recognition_predictor

    _ensure_predictor()
    if _inference_manager is None:
        return None

    with _surya_lock:
        if _recognition_predictor is not None:
            return _recognition_predictor
        try:
            from surya.recognition import RecognitionPredictor

            _recognition_predictor = RecognitionPredictor(_inference_manager)
            logger.info("[Surya] Recognition predictor готов.")
            return _recognition_predictor

        except Exception as e:
            logger.error(f"[Surya] Ошибка инициализации recognition: {e}")
            return None


# Типы regions которые извлекаем как изображения
IMAGE_REGIONS = frozenset({"Diagram", "Table", "Figure"})


# ── Определение layout regions на страницах ──
def detect_layout(images: list[Image.Image]) -> list:
    predictor = _ensure_predictor()
    if predictor is None:
        return []

    t0 = time.time()
    results = predictor(images)
    t1 = time.time()
    logger.info(f"[Surya] Layout: {len(images)} стр, {t1 - t0:.1f}s")
    return results


# ── Извлечение текста через surya OCR с layout-aware reading order ──
def ocr_text(images: list[Image.Image], layout_results: list) -> list[dict]:
    rec = _ensure_recognition()
    if rec is None:
        return []

    t0 = time.time()
    rec_results = rec(images, layout_results)
    t1 = time.time()
    logger.info(f"[Surya] OCR: {len(images)} стр, {t1 - t0:.1f}s")

    pages = []
    for i, result in enumerate(rec_results):
        blocks = getattr(result, "blocks", [])
        html_parts = []
        text_parts = []
        for block in blocks:
            html = getattr(block, "html", "")
            text = getattr(block, "text", "")
            if html:
                html_parts.append(html)
            if text:
                text_parts.append(text)
        pages.append(
            {
                "page": i + 1,
                "html": "\n".join(html_parts),
                "text": "\n".join(text_parts),
            }
        )

    return pages


# ── Извлечение bbox для указанных типов regions ──
def extract_regions(
    images: list[Image.Image], layout_results: list, regions: set[str] | None = None
) -> dict[int, list[dict]]:
    if regions is None:
        regions = IMAGE_REGIONS

    extracted = {}
    for page_idx, result in enumerate(layout_results):
        page_regions = []
        for bbox in result.bboxes:
            label = getattr(bbox, "label", "")
            if label in regions:
                left, top, right, bottom = [int(c) for c in bbox.bbox]
                img = images[page_idx]
                cropped = img.crop((left, top, right, bottom))
                page_regions.append(
                    {
                        "label": label,
                        "bbox": [left, top, right, bottom],
                        "image": cropped,
                        "confidence": getattr(bbox, "confidence", None),
                    }
                )
        if page_regions:
            extracted[page_idx] = page_regions

    return extracted


# ── Выгрузка surya моделей и остановка inference manager ──
def shutdown():
    global _layout_predictor, _recognition_predictor, _inference_manager

    with _surya_lock:
        if _inference_manager is not None:
            try:
                _inference_manager.stop()
            except Exception as e:
                logger.warning(f"[Surya] Ошибка остановки server: {e}")

        _layout_predictor = None
        _recognition_predictor = None
        _inference_manager = None
    logger.info("[Surya] Predictor'ы и server выгружены.")
