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
from src.gguf.server import unload_all_models
from src.ingestion.splitter import _get_splitter
from src.ingestion.utils import IngestionCancelled
from src.ingestion.vision import describe_image_with_lmstudio, get_vision_url

logger = logging.getLogger(__name__)


async def _analyze_page_for_vision(page):

    def _sync_analyze():
        text = page.get_text()
        images = page.get_images()
        drawings = page.get_drawings()
        has_real_graphics = bool(len(images) > 0)
        if not has_real_graphics:
            graphics_weight = 0
            horizontal_lines = 0
            vertical_lines = 0
            for d in drawings:
                items = d.get("items", [])
                if any(i[0] in ["c", "q"] for i in items) or len(items) > 12:
                    has_real_graphics = True
                    break
                rect = d.get("rect")
                fill = d.get("fill")
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
                if rect is not None:
                    w, h = rect.x1 - rect.x0, rect.y1 - rect.y0
                    if w > 30 and h < 3:
                        horizontal_lines += 1
                    elif w < 3 and h > 30:
                        vertical_lines += 1
                graphics_weight += 1
            if not has_real_graphics:
                if (
                    graphics_weight > 8
                    or (horizontal_lines >= 3 and vertical_lines >= 1)
                    or (horizontal_lines + vertical_lines >= 6)
                ):
                    has_real_graphics = True
        return text, has_real_graphics

    return await asyncio.to_thread(_sync_analyze)


async def _analyze_and_build_page(page_num, doc, images_dir, file_name, splitter):

    def _sync_build():
        page = doc.load_page(page_num)
        text = page.get_text()
        images = page.get_images()
        drawings = page.get_drawings()
        has_real_graphics = bool(len(images) > 0)
        if not has_real_graphics:
            graphics_weight = 0
            horizontal_lines = 0
            vertical_lines = 0
            for d in drawings:
                items = d.get("items", [])
                if any(i[0] in ["c", "q"] for i in items) or len(items) > 12:
                    has_real_graphics = True
                    break
                rect = d.get("rect")
                fill = d.get("fill")
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
                if rect is not None:
                    w, h = rect.x1 - rect.x0, rect.y1 - rect.y0
                    if w > 30 and h < 3:
                        horizontal_lines += 1
                    elif w < 3 and h > 30:
                        vertical_lines += 1
                graphics_weight += 1
            if not has_real_graphics:
                if (
                    graphics_weight > 8
                    or (horizontal_lines >= 3 and vertical_lines >= 1)
                    or (horizontal_lines + vertical_lines >= 6)
                ):
                    has_real_graphics = True

        local_nodes = []
        image_path = None
        if text and text.strip():
            local_nodes = splitter.get_nodes_from_documents(
                [TextNode(text=text, metadata={"file_name": file_name, "page": page_num + 1})]
            )
        if has_real_graphics and page is not None:
            image_path = os.path.join(images_dir, f"p_{page_num + 1}_{uuid.uuid4().hex[:6]}.png")
            page.get_pixmap(dpi=150).save(image_path)
        return page_num, local_nodes, image_path

    return await asyncio.to_thread(_sync_build)


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
                pn, local_nodes, image_path = await _analyze_and_build_page(
                    page_num, doc, images_dir, file_name, splitter
                )
                nodes.extend(local_nodes)
                if image_path:
                    frame_list.append({"page": pn + 1, "path": image_path})
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
                for page_num_result, local_nodes, image_path in page_results:
                    nodes.extend(local_nodes)
                    if image_path:
                        frame_list.append({"page": page_num_result + 1, "path": image_path})

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
                VISION_BATCH_SIZE = 20

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

                # Батчи по 20 — между батчами перезапускаем vision для очистки CUDA
                for batch_start in range(0, n, VISION_BATCH_SIZE):
                    if _is_cancelled():
                        raise IngestionCancelled(f"Cancelled at batch {batch_start}")
                    batch_end = min(batch_start + VISION_BATCH_SIZE, n)
                    batch = frame_list[batch_start:batch_end]
                    await asyncio.gather(*[_describe_frame(f) for f in batch])

                    if batch_end < n:
                        logger.info(
                            f"[Vision] Батч {batch_start + 1}-{batch_end}/{n} готов, перезапуск vision..."
                        )
                        await unload_all_models(role="vision")
                        import gc

                        gc.collect()
                        await asyncio.sleep(1)
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

            # Освобождаем память после обработки всех страниц
            results.clear()
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
        doc.close()
    return nodes


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
    """Конвертация Office-файла в PDF: LibreOffice → COM → извлечение текста."""
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
    """Обёртка: извлечение текста из Office-файла без конвертации."""

    def _sync():
        try:
            return extract_fn(file_path, file_name)
        except Exception as e:
            logger.warning(f"{log_prefix} резервный вариант тоже не удался: {e}")
        return []

    return await asyncio.to_thread(_sync)


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
