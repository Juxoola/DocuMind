"""Surya layout detection: определение regions (Diagram, Equation, Table) с bbox."""

import logging
import os
import time

from PIL import Image

logger = logging.getLogger(__name__)

# Кэш для layout predictor (создаётся один раз)
_layout_predictor = None
_inference_manager = None


def _ensure_predictor():
    """Ленивая инициализация surya layout predictor."""
    global _layout_predictor, _inference_manager

    if _layout_predictor is not None:
        return _layout_predictor

    try:
        # Устанавливаем env для llama.cpp
        llama_binary = os.getenv("LLAMA_CPP_BINARY")
        if not llama_binary:
            # Ищем llama-server в стандартных путях
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
        logger.error(f"[Surya] Ошибка инициализации: {e}")
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
    """Выгрузка surya моделей."""
    global _layout_predictor, _inference_manager
    _layout_predictor = None
    _inference_manager = None
    logger.info("[Surya] Predictor выгружен.")
