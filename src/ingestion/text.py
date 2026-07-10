"""Обработка PDF."""

import asyncio
import logging
import os
import re
import uuid

import aiofiles
import fitz
import orjson
from llama_index.core.schema import TextNode

import config
from routers.shared import get_async_http
from src.gguf.server import unload_all_models
from src.ingestion.splitter import _get_splitter
from src.ingestion.utils import IngestionCancelled
from src.ingestion.vision import describe_image_with_lmstudio, get_vision_url

logger = logging.getLogger(__name__)

# ── Pre-compiled regex patterns for markdown cleaning ──
_RE_MD_H2STAR = re.compile(r"^(#{1,6})\s*\*\*(.+?)\*\*\s*$", re.MULTILINE)
_RE_MD_PICTURE_OMITTED = re.compile(
    r"\*{0,2}={1,2}> picture \[\d+ x \d+\] intentionally omitted <={1,2}\*{0,2}"
)
_RE_MD_PICTURE_TEXT = re.compile(
    r"----- Start of picture text -----.*?----- End of picture text -----",
    re.DOTALL,
)
_RE_MD_PAGE_REF = re.compile(r"\*\*(.+?\.{3,}\d+)\*\*")
_RE_MD_PAGE_NUM = re.compile(r"\n\s*\d{1,3}\s*\n")
_RE_MD_NEWLINES = re.compile(r"\n{3,}")
_RE_MD_PICTURE_OMITTED_2 = re.compile(
    r"=> picture \[\d+ x \d+\] intentionally omitted <=.*?(?=----- Start|$)",
    re.DOTALL,
)


# Основной конвейер PDF: извлечение текста (pymupdf4llm), изображения (surya layout), Vision-анализ
# ── Основной конвейер PDF: текст (pymupdf4llm) → изображения (surya) → Vision ──
async def process_pdf(
    file_path,
    images_dir,
    llm_settings=None,
    shared_llm_url=None,
    original_filename=None,
    progress_cb=None,
    cancel_check=None,
    keep_vision_alive=False,
):

    def _is_cancelled():
        return bool(cancel_check and cancel_check())

    nodes = []
    file_name = original_filename or os.path.basename(file_path)
    doc = await asyncio.to_thread(fitz.open, file_path)
    frame_data = []
    frame_list = []
    results = []
    splitter = _get_splitter()
    total_pages = len(doc)

    # ── Шаг 1: pymupdf4llm для текста ──
    use_surya_ocr = False
    try:
        import pymupdf4llm

        def _extract_markdown():
            return pymupdf4llm.to_markdown(file_path, page_chunks=True, write_images=False)

        if progress_cb:
            progress_cb(10, "Извлечение текста (pymupdf4llm)...")
        md_chunks = await asyncio.to_thread(_extract_markdown)

        # Удаляем плейсхолдеры картинок ДО подсчёта символов
        for chunk in md_chunks:
            t = chunk.get("text", "")
            if t:
                t = _RE_MD_PICTURE_OMITTED.sub("", t)
                t = _RE_MD_PICTURE_TEXT.sub("", t)
                t = _RE_MD_PICTURE_OMITTED_2.sub("", t)
                chunk["text"] = t

        # Оцениваем качество извлечения
        total_chars = sum(len(c.get("text", "")) for c in md_chunks)
        avg_chars = total_chars / max(total_pages, 1)
        logger.info(
            f"[Ingestion] pymupdf4llm: {len(md_chunks)} чанков, {total_chars} симв., среднее {avg_chars:.0f}/стр"
        )

        if avg_chars < 100:
            # PDF-скан, мало текста → нужен Surya OCR
            logger.info("[Ingestion] Мало текста (< 100/стр) → Surya OCR")
            use_surya_ocr = True
        else:
            # Хороший текстовый PDF

            # ── Очистка markdown от артефактов pymupdf ──
            def _clean_markdown(text):
                text = _RE_MD_H2STAR.sub(r"\1 \2", text)
                text = _RE_MD_PICTURE_OMITTED.sub("", text)
                text = _RE_MD_PICTURE_TEXT.sub("", text)
                text = _RE_MD_PAGE_REF.sub(r"\1", text)
                text = _RE_MD_PAGE_NUM.sub("\n", text)
                text = text.replace("\\n", "\n")
                text = _RE_MD_NEWLINES.sub("\n\n", text)
                text = text.strip()
                return text

            for chunk in md_chunks:
                page_num = chunk.get("metadata", {}).get("page_number", 0)
                md_text = chunk.get("text", "")
                if md_text and md_text.strip():
                    md_text = _clean_markdown(md_text)

                    # Удаляем pymupdf image placeholders — surya layout извлекает изображения
                    md_text = _RE_MD_PICTURE_OMITTED_2.sub("", md_text)
                    md_text = _RE_MD_PICTURE_TEXT.sub("", md_text)
                    md_text = _RE_MD_NEWLINES.sub("\n\n", md_text).strip()

                    if md_text and md_text.strip():
                        # Добавляем номер страницы
                        page_header = f"## Стр. {page_num}\n\n"
                        nodes.extend(
                            splitter.get_nodes_from_documents(
                                [
                                    TextNode(
                                        text=page_header + md_text,
                                        metadata={"file_name": file_name, "page": page_num},
                                    )
                                ]
                            )
                        )

            logger.info(f"[Ingestion] pymupdf4llm: {len(nodes)} узлов")

    except ImportError:
        logger.warning("[Ingestion] pymupdf4llm не установлен → Surya OCR")
        use_surya_ocr = True
    except Exception as e:
        logger.warning(f"[Ingestion] pymupdf4llm ошибка: {e} → Surya OCR")
        use_surya_ocr = True

    # Surya layout: все изображения извлекаются через surya
    try:
        if not _is_cancelled():
            surya_frame_list = await _surya_layout_pass(
                doc,
                file_name,
                images_dir,
                splitter,
                nodes,
                llm_settings,
                shared_llm_url,
                progress_cb,
                cancel_check,
                use_surya_ocr,
                frame_data,
                keep_vision_alive,
            )
            frame_list.extend(surya_frame_list)

        if frame_list:
            if shared_llm_url is None:
                shared_llm_url = await get_vision_url(llm_settings)
            if shared_llm_url:
                v_conc = int(
                    (llm_settings or {}).get("vision_concurrency") or config.VISION_CONCURRENCY
                )
                n = len(frame_list)
                if progress_cb:
                    progress_cb(
                        65,
                        f"Анализ {n} страниц PDF ({'параллельно' if v_conc > 1 else 'последовательно'})...",
                    )

                sem = asyncio.Semaphore(v_conc)
                done_count = 0
                results = []
                VISION_BATCH_SIZE = 100

                async def _describe_frame(frame_info):
                    nonlocal done_count
                    if _is_cancelled():
                        raise IngestionCancelled(f"Cancelled during OCR ({done_count}/{n})")
                    async with sem:
                        desc = await describe_image_with_lmstudio(
                            frame_info["path"],
                            llm_settings,
                            shared_llm_url,
                            cancel_check=cancel_check,
                        )
                        done_count += 1
                        results.append((frame_info, desc))
                        if progress_cb:
                            progress_cb(
                                65 + int(done_count / n * 25), f"Описание PDF: {done_count}/{n}"
                            )

                for batch_start in range(0, n, VISION_BATCH_SIZE):
                    if _is_cancelled():
                        raise IngestionCancelled(f"Cancelled at batch {batch_start}")
                    batch_end = min(batch_start + VISION_BATCH_SIZE, n)
                    batch = frame_list[batch_start:batch_end]
                    await asyncio.gather(*[_describe_frame(f) for f in batch])

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

                results.sort(key=lambda x: x[0]["page"])

                # Группируем описания по страницам
                page_descs = {}
                for frame_info, desc in results:
                    if desc and "Изображение без описания" not in desc:
                        pg = frame_info["page"]
                        if pg not in page_descs:
                            page_descs[pg] = []
                        page_descs[pg].append(
                            {
                                "text": desc,
                                "image_path": frame_info["path"],
                            }
                        )

                # Добавляем описания как ОТДЕЛЬНЫЕ ноды (не обрезаются сплиттером)
                for pg, descs in list(page_descs.items()):
                    for d in descs:
                        desc_text = f"--- Изображение (стр. {pg}) ---\n{d['text']}\n---"
                        nodes.append(
                            TextNode(
                                text=desc_text,
                                metadata={
                                    "file_name": file_name,
                                    "page": pg,
                                    "image_path": d["image_path"],
                                },
                            )
                        )
                        frame_data.append(
                            {"page": pg, "image_path": d["image_path"], "description": d["text"]}
                        )

        if frame_data:
            import gc

            gc.collect()

            if shared_llm_url and not keep_vision_alive:
                await unload_all_models(role="vision")

        if frame_data:
            frame_data.sort(key=lambda x: x["page"])
            metadata_json = {
                "file_name": file_name,
                "is_video": False,
                "transcript": [],
                "frames": frame_data,
            }

            metadata_path = os.path.join(os.path.dirname(file_path), f"{file_name}.json")
            async with aiofiles.open(metadata_path, "w", encoding="utf-8") as f:
                await f.write(orjson.dumps(metadata_json, option=orjson.OPT_INDENT_2).decode())
    finally:
        await asyncio.to_thread(doc.close)
    return nodes


# Surya layout pass: определение Diagram/Equation/Table regions + OCR
# ── Проход Surya layout: определение regions, OCR, извлечение изображений ──
async def _surya_layout_pass(
    doc,
    file_name,
    images_dir,
    splitter,
    nodes,
    llm_settings,
    shared_llm_url,
    progress_cb,
    cancel_check,
    run_ocr: bool,
    frame_data,
    keep_vision_alive,
):
    """Surya layout detection + OCR + Vision LLM для Diagram/Equation regions."""
    from src.ingestion.surya_layout import (
        detect_layout,
        extract_regions,
        ocr_text,
    )
    from src.ingestion.surya_layout import (
        shutdown as surya_shutdown,
    )

    def _is_cancelled():
        return bool(cancel_check and cancel_check())

    total_pages = len(doc)
    frame_list = []

    # Конвертируем страницы PDF в PIL images для surya
    if progress_cb:
        progress_cb(60, "Surya: определение layout...")

    def _render_pages():
        pil_images = []
        for page_num in range(total_pages):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(dpi=150)
            from PIL import Image as _Image

            img = _Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            pil_images.append(img)
        return pil_images

    pil_images = await asyncio.to_thread(_render_pages)

    # Запускаем layout detection
    try:
        layout_results = await asyncio.to_thread(detect_layout, pil_images)
    except Exception as e:
        logger.warning(f"[Surya] Ошибка layout detection: {e}")
        return frame_list

    if not layout_results:
        return frame_list

    if run_ocr:
        try:
            if progress_cb:
                progress_cb(65, "Surya: OCR текста...")
            ocr_results = await asyncio.to_thread(ocr_text, pil_images, layout_results)
            if ocr_results:
                old_nodes = nodes.copy()
                nodes.clear()
                for page_data in ocr_results:
                    html_text = page_data.get("html", "")
                    if html_text and html_text.strip():
                        nodes.extend(
                            splitter.get_nodes_from_documents(
                                [
                                    TextNode(
                                        text=html_text,
                                        metadata={
                                            "file_name": file_name,
                                            "page": page_data["page"],
                                        },
                                    )
                                ]
                            )
                        )
                logger.info(f"[Surya] OCR: заменено {len(old_nodes)} -> {len(nodes)} узлов")
        except Exception as e:
            logger.warning(f"[Surya] Ошибка OCR: {e}")

    regions = {"Diagram", "Figure", "Picture"}
    extracted = extract_regions(pil_images, layout_results, regions)

    n_regions = sum(len(r) for r in extracted.values())
    if n_regions == 0:
        logger.info("[Surya] Diagram/Equation/Table regions не найдены")
        surya_shutdown()
        return frame_list

    logger.info(f"[Surya] Найдено {n_regions} regions для описания")

    # Группируем邻近ные regions на одной странице
    for page_idx, page_regions in extracted.items():
        if _is_cancelled():
            break

        # Сортируем по y, потом x
        page_regions.sort(key=lambda r: (r["bbox"][1], r["bbox"][0]))

        # Группируем: расстояние между bbox < 200px
        groups = []
        used = [False] * len(page_regions)
        for i in range(len(page_regions)):
            if used[i]:
                continue
            group = [page_regions[i]]
            used[i] = True
            for j in range(i + 1, len(page_regions)):
                if used[j]:
                    continue
                # Сравниваем с общим bbox группы
                gx0 = min(r["bbox"][0] for r in group)
                gy0 = min(r["bbox"][1] for r in group)
                gx1 = max(r["bbox"][2] for r in group)
                gy1 = max(r["bbox"][3] for r in group)
                b2 = page_regions[j]["bbox"]
                h_dist = max(0, max(gx0, b2[0]) - min(gx1, b2[2]))
                v_dist = max(0, max(gy0, b2[1]) - min(gy1, b2[3]))
                if h_dist < 200 and v_dist < 200:
                    group.append(page_regions[j])
                    used[j] = True
            groups.append(group)

        for group_idx, group in enumerate(groups):
            if _is_cancelled():
                break
            try:
                all_bboxes = [r["bbox"] for r in group]
                x0 = min(b[0] for b in all_bboxes) - 10
                y0 = min(b[1] for b in all_bboxes) - 10
                x1 = max(b[2] for b in all_bboxes) + 10
                y1 = max(b[3] for b in all_bboxes) + 10
                img = pil_images[page_idx]
                cropped = img.crop((x0, y0, x1, y1))
                labels = "+".join(set(r["label"] for r in group))
                img_name = f"surya_{labels.lower()}_p{page_idx + 1}_g{group_idx}_{uuid.uuid4().hex[:4]}.png"
                img_path = os.path.join(images_dir, img_name)
                await asyncio.to_thread(cropped.save, img_path)
                frame_list.append({"page": page_idx + 1, "path": img_path})
            except Exception:
                logger.debug(
                    f"[surya_layout] Не удалось сохранить группу layout на странице {page_idx + 1}"
                )

    surya_shutdown()
    return frame_list
