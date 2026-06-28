"""Surya layout detection + OCR: определение regions и текста с форматированием."""

import logging
import os
import time

from PIL import Image

logger = logging.getLogger(__name__)

# Кэш для predictors (создаются один раз)
_layout_predictor = None
_recognition_predictor = None
_inference_manager = None


def _ensure_predictor():
    """Ленивая инициализация surya inference manager + layout predictor."""
    global _layout_predictor, _inference_manager

    if _layout_predictor is not None:
        return _layout_predictor

    try:
        # Устанавливаем env для llama.cpp
        llama_binary = os.getenv("LLAMA_CPP_BINARY")
        if not llama_binary:
            for candidate in [
                r"F:\llama.cpp\build\bin\Release\llama-server.exe",
                r"F:\llama.cpp\build\bin\llama-server",
            ]:
                if os.path.exists(candidate):
                    os.environ["LLAMA_CPP_BINARY"] = candidate
                    break

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
    """Ленивая инициализация surya recognition predictor."""
    global _recognition_predictor, _inference_manager

    if _recognition_predictor is not None:
        return _recognition_predictor

    # Сначала убедимся что inference manager создан
    _ensure_predictor()
    if _inference_manager is None:
        return None

    try:
        from surya.recognition import RecognitionPredictor

        _recognition_predictor = RecognitionPredictor(_inference_manager)
        logger.info("[Surya] Recognition predictor готов.")
        return _recognition_predictor

    except Exception as e:
        logger.error(f"[Surya] Ошибка инициализации recognition: {e}")
        return None


# Типы regions которые извлекаем как изображения
IMAGE_REGIONS = {"Diagram", "Table", "Figure"}


def detect_layout(images: list[Image.Image]) -> list:
    """Определяет layout regions на страницах.

    Args:
        images: список PIL.Image для каждой страницы

    Returns:
        список layout results (по одному на страницу)
    """
    predictor = _ensure_predictor()
    if predictor is None:
        return []

    t0 = time.time()
    results = predictor(images)
    t1 = time.time()
    logger.info(f"[Surya] Layout: {len(images)} стр, {t1 - t0:.1f}s")
    return results


def ocr_text(
    images: list[Image.Image], layout_results: list
) -> list[dict]:
    """Извлекает текст через surya OCR с layout-aware reading order.

    Args:
        images: PIL.Image для каждой страницы
        layout_results: результаты detect_layout

    Returns:
        список dict: [{"page": int, "html": str, "text": str}, ...]
    """
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
        pages.append({
            "page": i + 1,
            "html": "\n".join(html_parts),
            "text": "\n".join(text_parts),
        })

    return pages


def extract_regions(
    images: list[Image.Image], layout_results: list, regions: set[str] | None = None
) -> dict[int, list[dict]]:
    """Извлекает bbox для указанных типов regions.

    Args:
        images: PIL.Image для каждой страницы
        layout_results: результаты detect_layout
        regions: типы regions (по умолчанию IMAGE_REGIONS)

    Returns:
        dict {page_index: [{"label": str, "bbox": [l,t,r,b], "image": PIL.Image}, ...]}
    """
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


def shutdown():
    """Выгрузка surya моделей + остановка llama-server."""
    global _layout_predictor, _recognition_predictor, _inference_manager

    # Останавливаем inference manager (убивает llama-server процесс)
    if _inference_manager is not None:
        try:
            _inference_manager.stop()
        except Exception as e:
            logger.warning(f"[Surya] Ошибка остановки server: {e}")

    _layout_predictor = None
    _recognition_predictor = None
    _inference_manager = None
    logger.info("[Surya] Predictor'ы и server выгружены.")
