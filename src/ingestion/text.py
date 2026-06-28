"""Обработка текстовых документов: PDF, PPTX, DOCX."""

import asyncio
import logging
import os
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


# Анализ графических элементов страницы для определения необходимости Vision
def _detect_has_real_graphics(images: list, drawings: list) -> bool:
    if images:
        return True
    graphics_weight = 0
    horizontal_lines = 0
    vertical_lines = 0
    for d in drawings:
        items = d.get("items", [])
        # Кривые (Bezier) или сложные composition — настоящая графика
        if any(i[0] in ["c", "q"] for i in items) or len(items) > 12:
            return True
        rect = d.get("rect")
        fill = d.get("fill")
        # Пропускаем фоновые прямоугольники (белый фон страницы)
        is_page_background = (
            len(items) == 1
            and items[0][0] == "re"
            and fill is not None
            and all(c >= 0.99 for c in fill)
            and rect is not None
            and (rect.x1 - rect.x0) > 200
            and (rect.y1 - rect.y0) > 200
        )
        if is_page_background:
            continue
        # Пропускаем простые рамки (code blocks, border boxes):
        # один прямоугольник без заливки, шире чем высокий (горизонтальная рамка)
        is_simple_frame = (
            len(items) == 1
            and items[0][0] == "re"
            and (fill is None or all(c >= 0.95 for c in fill))
            and rect is not None
            and (rect.x1 - rect.x0) > (rect.y1 - rect.y0) * 1.5
            and (rect.x1 - rect.x0) > 100
            and (rect.y1 - rect.y0) > 3
        )
        if is_simple_frame:
            continue
        if rect is not None:
            w, h = rect.x1 - rect.x0, rect.y1 - rect.y0
            if w > 30 and h < 3:
                horizontal_lines += 1
            elif w < 3 and h > 30:
                vertical_lines += 1
        graphics_weight += 1
    return (
        graphics_weight > 8
        or (horizontal_lines >= 3 and vertical_lines >= 1)
        or (horizontal_lines + vertical_lines >= 6)
    )


async def _analyze_page_for_vision(page):

    def _sync_analyze():
        text = page.get_text()
        images = page.get_images()
        drawings = page.get_drawings()
        return text, _detect_has_real_graphics(images, drawings)

    return await asyncio.to_thread(_sync_analyze)


async def _analyze_and_build_page(page_num, doc, images_dir, file_name, splitter):

    def _sync_build():
        page = doc.load_page(page_num)
        text = page.get_text()
        images = page.get_images()
        drawings = page.get_drawings()
        has_real_graphics = _detect_has_real_graphics(images, drawings)

        local_nodes = []
        if text and text.strip():
            local_nodes = splitter.get_nodes_from_documents(
                [TextNode(text=text, metadata={"file_name": file_name, "page": page_num + 1})]
            )

        # Извлечение встроенных изображений по одному с кропом по координатам
        image_paths = []
        if has_real_graphics and page is not None:
            seen_xrefs = set()
            for img_info in images:
                xref = img_info[0]
                if xref in seen_xrefs:
                    continue
                seen_xrefs.add(xref)
                try:
                    rects = page.get_image_rects(xref)
                    if not rects:
                        continue
                    base_image = page.parent.extract_image(xref)
                    if not base_image or not base_image.get("image"):
                        continue
                    ext = base_image.get("ext", "png")
                    if ext == "jpeg":
                        ext = "jpg"
                    for rect in rects:
                        # Пропускаем tiny изображения (иконки, маркеры)
                        w, h = rect.width, rect.height
                        if w < 50 or h < 50:
                            continue
                        try:
                            clip = fitz.Rect(rect)
                            pix = page.get_pixmap(clip=clip, dpi=150)
                            img_name = f"p_{page_num + 1}_x{xref}_{uuid.uuid4().hex[:4]}.{ext}"
                            img_path = os.path.join(images_dir, img_name)
                            pix.save(img_path)
                            image_paths.append(img_path)
                        except Exception:
                            continue
                except Exception:
                    continue

            # Fallback: кластеризация vector drawings → кроп регионов
            if not image_paths:
                drawing_rects = []
                page_rect = page.rect
                for d in drawings:
                    rect = d.get("rect")
                    if rect is None:
                        continue
                    items = d.get("items", [])
                    fill = d.get("fill")
                    # Пропускаем фоновые прямоугольники
                    is_bg = (
                        len(items) == 1
                        and items[0][0] == "re"
                        and fill is not None
                        and all(c >= 0.99 for c in fill)
                        and rect.width > 200
                        and rect.height > 200
                    )
                    if is_bg:
                        continue
                    # Пропускаем простые рамки (code blocks, border boxes)
                    is_simple_frame = (
                        len(items) == 1
                        and items[0][0] == "re"
                        and (fill is None or all(c >= 0.95 for c in fill))
                        and (rect.x1 - rect.x0) > (rect.y1 - rect.y0) * 1.5
                        and (rect.x1 - rect.x0) > 100
                        and (rect.y1 - rect.y0) > 3
                    )
                    if is_simple_frame:
                        continue
                    if rect.width < 10 or rect.height < 10:
                        continue
                    # Пропускаем элементы на границе страницы (колонтитулы)
                    if rect.y0 < 30 or rect.y1 > page_rect.height - 30:
                        continue
                    drawing_rects.append(fitz.Rect(rect))

                # Кластеризация: объединяем близкие rect'ы (порог 50px)
                clusters = []
                for r in drawing_rects:
                    merged = False
                    for ci, cluster in enumerate(clusters):
                        # expanded = cluster + 50px padding
                        expanded = fitz.Rect(
                            cluster.x0 - 50,
                            cluster.y0 - 50,
                            cluster.x1 + 50,
                            cluster.y1 + 50,
                        )
                        if expanded.intersects(r):
                            clusters[ci] = cluster | r  # объединение
                            merged = True
                            break
                    if not merged:
                        clusters.append(r)

                # Рендер каждого кластера с padding
                PAD = 20
                for i, cluster_rect in enumerate(clusters):
                    clip = fitz.Rect(
                        max(0, cluster_rect.x0 - PAD),
                        max(0, cluster_rect.y0 - PAD),
                        min(page_rect.width, cluster_rect.x1 + PAD),
                        min(page_rect.height, cluster_rect.y1 + PAD),
                    )
                    # Пропускаем слишком маленькие кластеры
                    if clip.width < 40 or clip.height < 40:
                        continue
                    try:
                        pix = page.get_pixmap(clip=clip, dpi=150)
                        img_name = f"p_{page_num + 1}_d{i}_{uuid.uuid4().hex[:4]}.png"
                        img_path = os.path.join(images_dir, img_name)
                        pix.save(img_path)
                        image_paths.append(img_path)
                    except Exception:
                        continue

                # Fallback: если кластеризация ничего не дала — полная страница
                if not image_paths:
                    img_name = f"p_{page_num + 1}_{uuid.uuid4().hex[:6]}.png"
                    img_path = os.path.join(images_dir, img_name)
                    page.get_pixmap(dpi=150).save(img_path)
                    image_paths.append(img_path)

        return page_num, local_nodes, image_paths

    return await asyncio.to_thread(_sync_build)


# Основной конвейер PDF: извлечение текста, батчевая параллельная обработка, Vision-анализ
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
    n_workers = min(8, (os.cpu_count() or 4), total_pages)

    try:
        BATCH_SIZE = 16
        if n_workers <= 1:
            for page_num in range(total_pages):
                if _is_cancelled():
                    raise IngestionCancelled(f"Cancelled at page {page_num + 1}")
                pn, local_nodes, image_paths = await _analyze_and_build_page(
                    page_num, doc, images_dir, file_name, splitter
                )
                nodes.extend(local_nodes)
                for img_p in image_paths or []:
                    frame_list.append({"page": pn + 1, "path": img_p})
        else:
            for batch_start in range(0, total_pages, BATCH_SIZE):
                if _is_cancelled():
                    raise IngestionCancelled(f"Cancelled at page {batch_start + 1}")
                batch_end = min(batch_start + BATCH_SIZE, total_pages)

                async def _process_batch_pages(batch_start, batch_end):
                    tasks = []
                    for page_num in range(batch_start, batch_end):
                        tasks.append(
                            _analyze_and_build_page(page_num, doc, images_dir, file_name, splitter)
                        )
                    return await asyncio.gather(*tasks)

                page_results = await _process_batch_pages(batch_start, batch_end)
                for page_num_result, local_nodes, image_paths in page_results:
                    nodes.extend(local_nodes)
                    for img_p in image_paths or []:
                        frame_list.append({"page": page_num_result + 1, "path": img_p})

        if frame_list:
            if shared_llm_url is None:
                shared_llm_url = await get_vision_url(llm_settings)
            if shared_llm_url:
                v_conc = int(llm_settings.get("vision_concurrency") or config.VISION_CONCURRENCY)
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
                        # Watchdog контролирует RAM — restart только при unhealthy
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
                for frame_info, desc in results:
                    if desc and "Изображение без описания" not in desc:
                        full_text = f"Изображение PDF {file_name} стр {frame_info['page']}: {desc}"
                        if len(full_text) <= config.GGUF_CTX_EMBED_CHARS:
                            nodes.append(
                                TextNode(
                                    text=full_text,
                                    metadata={
                                        "file_name": file_name,
                                        "image_path": frame_info["path"],
                                        "page": frame_info["page"],
                                    },
                                )
                            )
                        else:
                            desc_nodes = splitter.get_nodes_from_documents(
                                [
                                    TextNode(
                                        text=full_text,
                                        metadata={
                                            "file_name": file_name,
                                            "image_path": frame_info["path"],
                                            "page": frame_info["page"],
                                        },
                                    )
                                ]
                            )
                            nodes.extend(desc_nodes)
                        frame_data.append(
                            {
                                "page": frame_info["page"],
                                "image_path": frame_info["path"],
                                "description": desc,
                            }
                        )
                    else:
                        try:
                            await aiofiles.os.remove(frame_info["path"])
                        except Exception:
                            pass

            results.clear()
            import gc

            gc.collect()

            if shared_llm_url and not keep_vision_alive:
                await unload_all_models(role="vision")

        # Surya layout detection: определение Diagram/Equation regions
        surya_mode = getattr(config, "SURYA_MODE", "disabled")
        if surya_mode in ("layout_only", "full") and not _is_cancelled():
            surya_frame_list = await _surya_layout_pass(
                doc, file_name, images_dir, splitter, nodes,
                llm_settings, shared_llm_url, progress_cb, cancel_check,
                surya_mode, frame_data, keep_vision_alive,
            )
            frame_list.extend(surya_frame_list)

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
        doc.close()
    return nodes


# Surya layout pass: определение Diagram/Equation/Table regions
async def _surya_layout_pass(
    doc, file_name, images_dir, splitter, nodes,
    llm_settings, shared_llm_url, progress_cb, cancel_check,
    surya_mode, frame_data, keep_vision_alive,
):
    """Surya layout detection + Vision LLM для Diagram/Equation regions."""
    from src.ingestion.surya_layout import detect_layout, extract_regions, shutdown as surya_shutdown

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

    # Извлекаем regions для Vision LLM
    regions = {"Diagram", "Equation", "Table"}
    extracted = extract_regions(pil_images, layout_results, regions)

    n_regions = sum(len(r) for r in extracted.values())
    if n_regions == 0:
        logger.info("[Surya] Diagram/Equation/Table regions не найдены")
        return frame_list

    logger.info(f"[Surya] Найдено {n_regions} regions для описания")

    # Сохраняем изображения regions и добавляем в frame_list для Vision
    for page_idx, page_regions in extracted.items():
        for region in page_regions:
            if _is_cancelled():
                break
            label = region["label"]
            img = region["image"]
            img_name = f"surya_{label.lower()}_p{page_idx + 1}_{uuid.uuid4().hex[:4]}.png"
            img_path = os.path.join(images_dir, img_name)
            await asyncio.to_thread(img.save, img_path)
            frame_list.append({"page": page_idx + 1, "path": img_path})

    surya_shutdown()
    return frame_list


# Конвертация Office-файлов в PDF: LibreOffice (приоритет), COM (резерв), текстовый fallback
def _find_soffice():
    import shutil

    local = os.path.join(config.BASE_DIR, "libreoffice", "program", "soffice.exe")
    if os.path.isfile(local):
        return local
    found = shutil.which("soffice")
    if found:
        return found
    for pf in ["Program Files", "Program Files (x86)"]:
        p = os.path.join("C:\\", pf, "LibreOffice", "program", "soffice.exe")
        if os.path.isfile(p):
            return p
    return None


async def _convert_via_libreoffice(file_path):
    soffice = _find_soffice()
    if not soffice:
        raise FileNotFoundError("LibreOffice не найден. Установите или скачайте через setup.ps1")
    out_dir = os.path.dirname(file_path)
    cmd = [
        soffice,
        "--headless",
        "--norestore",
        "--convert-to",
        "pdf",
        "--outdir",
        out_dir,
        os.path.abspath(file_path),
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
    pdf_path = os.path.splitext(file_path)[0] + ".pdf"
    if proc.returncode == 0 and os.path.exists(pdf_path):
        logger.info(f"[DOCX] Сконвертировано в PDF через LibreOffice: {os.path.basename(pdf_path)}")
        try:
            await aiofiles.os.remove(file_path)
        except OSError:
            pass
        return pdf_path
    raise RuntimeError(
        f"LibreOffice конвертация не удалась (code={proc.returncode}): "
        f"{(stderr or b'').decode('utf-8', errors='replace')[:200]}"
    )


async def _convert_via_com(file_path, app_name, format_code):

    def _sync_com():
        import pythoncom
        import win32com.client

        pdf_path = os.path.splitext(file_path)[0] + ".pdf"
        app = None
        doc = None
        try:
            pythoncom.CoInitialize()
            app = win32com.client.Dispatch(app_name)
            if app_name == "Powerpoint.Application":
                doc = app.Presentations.Open(os.path.abspath(file_path), WithWindow=False)
            else:
                doc = app.Documents.Open(os.path.abspath(file_path))
            doc.SaveAs(os.path.abspath(pdf_path), format_code)
            if not os.path.exists(pdf_path):
                raise RuntimeError(f"COM {app_name} не создал PDF: {pdf_path}")
            if os.path.exists(file_path):
                os.remove(file_path)
            return pdf_path
        except IngestionCancelled:
            raise
        except Exception as e:
            logger.warning(f"COM-конвертация через {app_name} не удалась: {e}")
            raise
        finally:
            if doc is not None:
                try:
                    doc.Close()
                except Exception:
                    pass
            if app is not None:
                try:
                    app.Quit()
                except Exception:
                    pass
            try:
                pythoncom.CoUninitialize()
            except Exception:
                pass

    return await asyncio.to_thread(_sync_com)


async def _convert_office_to_pdf(file_path, app_name, format_code, textonly_fn, log_prefix):
    file_name = os.path.basename(file_path)
    try:
        pdf_path = await _convert_via_libreoffice(file_path)
    except FileNotFoundError:
        logger.info(f"{log_prefix} LibreOffice не найден, пробую COM-конвертацию...")
        try:
            pdf_path = await _convert_via_com(file_path, app_name, format_code)
        except IngestionCancelled:
            raise
        except Exception as e:
            logger.warning(f"{log_prefix} COM-конвертация тоже не удалась: {e}")
            return None, await textonly_fn(file_path, file_name)
    except IngestionCancelled:
        raise
    except Exception as e:
        logger.warning(f"{log_prefix} LibreOffice конвертация не удалась: {e}")
        try:
            pdf_path = await _convert_via_com(file_path, app_name, format_code)
        except IngestionCancelled:
            raise
        except Exception:
            return None, await textonly_fn(file_path, file_name)
    return os.path.basename(pdf_path), None


def _pptx_extract_textonly(file_path, file_name):
    from pptx import Presentation

    prs = Presentation(file_path)
    nodes = []
    for i, slide in enumerate(prs.slides):
        text = "\n".join([sh.text for sh in slide.shapes if hasattr(sh, "text")])
        if text.strip():
            nodes.append(TextNode(text=text, metadata={"file_name": file_name, "page": i + 1}))
    if nodes:
        logger.info(
            f"[PPTX] Резервный вариант: {len(nodes)} слайдов через python-pptx (без Vision)"
        )
    return nodes


def _docx_extract_textonly(file_path, file_name):
    import docx as _docx

    text = "\n".join([p.text for p in _docx.Document(file_path).paragraphs])
    if text.strip():
        logger.info("[DOCX] Fallback: текст извлечён через python-docx (без Vision)")
        return [TextNode(text=text, metadata={"file_name": file_name})]
    return []


async def _process_office_textonly(file_path, file_name, extract_fn, log_prefix):

    def _sync():
        try:
            return extract_fn(file_path, file_name)
        except Exception as e:
            logger.warning(f"{log_prefix} резервный вариант тоже не удался: {e}")
        return []

    return await asyncio.to_thread(_sync)


# Обработка PPTX/DOCX: конвертация в PDF + Vision или текстовый fallback
async def process_pptx(
    file_path,
    images_dir,
    llm_settings=None,
    shared_llm_url=None,
    progress_cb=None,
    cancel_check=None,
    keep_vision_alive=False,
):
    file_name = os.path.basename(file_path)
    pdf_name, fallback_nodes = await _convert_office_to_pdf(
        file_path,
        "Powerpoint.Application",
        32,
        lambda fp, fn: _process_office_textonly(fp, fn, _pptx_extract_textonly, "[PPTX]"),
        "[PPTX]",
    )
    if fallback_nodes is not None:
        return fallback_nodes

    return await process_pdf(
        os.path.join(os.path.dirname(file_path), pdf_name),
        images_dir,
        llm_settings,
        shared_llm_url,
        original_filename=pdf_name,
        progress_cb=progress_cb,
        cancel_check=cancel_check,
        keep_vision_alive=keep_vision_alive,
    )


async def process_docx(
    file_path,
    images_dir,
    llm_settings=None,
    shared_llm_url=None,
    progress_cb=None,
    cancel_check=None,
    keep_vision_alive=False,
):
    file_name = os.path.basename(file_path)
    pdf_name, fallback_nodes = await _convert_office_to_pdf(
        file_path,
        "Word.Application",
        17,
        lambda fp, fn: _process_office_textonly(fp, fn, _docx_extract_textonly, "[DOCX]"),
        "[DOCX]",
    )
    if fallback_nodes is not None:
        return fallback_nodes

    return await process_pdf(
        os.path.join(os.path.dirname(file_path), pdf_name),
        images_dir,
        llm_settings,
        shared_llm_url,
        original_filename=pdf_name,
        progress_cb=progress_cb,
        cancel_check=cancel_check,
        keep_vision_alive=keep_vision_alive,
    )
