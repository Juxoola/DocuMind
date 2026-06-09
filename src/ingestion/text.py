"""Обработка текстовых документов: PDF, PPTX, DOCX."""

import json
import logging
import os
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

import fitz
from llama_index.core.schema import TextNode
from pptx import Presentation

import config
from src.gguf_direct import unload_all_models
from src.ingestion.splitter import _get_splitter
from src.ingestion.utils import IngestionCancelled
from src.ingestion.vision import describe_image_with_lmstudio, get_vision_url

logger = logging.getLogger(__name__)


def _analyze_page_for_vision(page):

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
                len(items) == 1 and items[0][0] == "re"
                and fill is not None and all(c >= 0.99 for c in fill)
                and rect is not None and (rect.x1 - rect.x0) > 200 and (rect.y1 - rect.y0) > 200
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
            if graphics_weight > 8 or (horizontal_lines >= 3 and vertical_lines >= 1) or (horizontal_lines + vertical_lines >= 6):
                has_real_graphics = True
    return text, has_real_graphics


def process_pdf(file_path, images_dir, llm_settings=None, shared_llm_url=None,
                original_filename=None, progress_cb=None, cancel_check=None,
                keep_vision_alive=False):

    def _is_cancelled():
        return bool(cancel_check and cancel_check())

    nodes = []
    file_name = original_filename or os.path.basename(file_path)
    doc = fitz.open(file_path)
    frame_data = []
    frame_list = []
    splitter = _get_splitter()
    total_pages = len(doc)
    n_workers = min(8, (os.cpu_count() or 4), total_pages)
    page_results = [None] * total_pages

    # Фаза 1: параллельный разбор страниц
    if n_workers <= 1:
        for page_num in range(total_pages):
            if _is_cancelled():
                raise IngestionCancelled(f"Cancelled at page {page_num + 1}")
            page = doc.load_page(page_num)
            text, has_real_graphics = _analyze_page_for_vision(page)
            page_results[page_num] = (text, has_real_graphics, page)
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            cancel_check_every = max(1, total_pages // (n_workers * 4))
            submitted = 0
            future_to_pn = {}
            for page_num in range(total_pages):
                if submitted % cancel_check_every == 0 and _is_cancelled():
                    ex.shutdown(wait=False, cancel_futures=True)
                    raise IngestionCancelled(f"Cancelled at page {submitted}")
                page = doc.load_page(page_num)
                future_to_pn[ex.submit(_analyze_page_for_vision, page)] = page_num
                submitted += 1
            for fut in as_completed(future_to_pn):
                pn = future_to_pn[fut]
                try:
                    text, has_real_graphics = fut.result()
                    page_results[pn] = (text, has_real_graphics, doc.load_page(pn))
                except Exception as e:
                    logger.warning(f"Ошибка разбора страницы {pn + 1}: {e}")
                    page_results[pn] = ("", False, None)

    # Фаза 2: создание нод + Pixmap
    def _build_page_artifacts(page_num: int):
        result = page_results[page_num]
        text, has_real_graphics = result[0], result[1]
        page = result[2] if len(result) > 2 else doc.load_page(page_num)
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

    if total_pages <= 1 or n_workers <= 1:
        for page_num in range(total_pages):
            if _is_cancelled():
                raise IngestionCancelled(f"Cancelled at page {page_num + 1}")
            pn, local_nodes, image_path = _build_page_artifacts(page_num)
            nodes.extend(local_nodes)
            if image_path:
                frame_list.append({"page": pn + 1, "path": image_path})
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futures = [ex.submit(_build_page_artifacts, pn) for pn in range(total_pages)]
            artifacts = [None] * total_pages
            for fut in as_completed(futures):
                if _is_cancelled():
                    ex.shutdown(wait=False, cancel_futures=True)
                    raise IngestionCancelled("Cancelled during artifacts build")
                pn, local_nodes, image_path = fut.result()
                artifacts[pn] = (local_nodes, image_path)
            for pn, (local_nodes, image_path) in enumerate(artifacts):
                nodes.extend(local_nodes)
                if image_path:
                    frame_list.append({"page": pn + 1, "path": image_path})

    # Фаза 3: Vision для страниц с графикой
    if frame_list:
        if shared_llm_url is None:
            shared_llm_url = get_vision_url(llm_settings)
        if shared_llm_url:
            v_conc = int(llm_settings.get("vision_concurrency") or config.VISION_CONCURRENCY)
            n = len(frame_list)
            if progress_cb:
                progress_cb(65, f"Анализ {n} страниц PDF ({'параллельно' if v_conc > 1 else 'последовательно'})...")

            with ThreadPoolExecutor(max_workers=v_conc) as executor:
                futures = {executor.submit(describe_image_with_lmstudio, f["path"], llm_settings, shared_llm_url): f for f in frame_list}
                done_count = 0
                try:
                    for future in as_completed(futures):
                        if _is_cancelled():
                            executor.shutdown(wait=False, cancel_futures=True)
                            raise IngestionCancelled(f"Cancelled during OCR ({done_count}/{n})")
                        frame_info = futures[future]
                        desc = future.result()
                        done_count += 1
                        if desc and "Изображение без описания" not in desc:
                            full_text = f"Изображение PDF {file_name} стр {frame_info['page']}: {desc}"
                            if len(full_text) <= config.GGUF_CTX_EMBED_CHARS:
                                nodes.append(TextNode(text=full_text, metadata={"file_name": file_name, "image_path": frame_info["path"], "page": frame_info["page"]}))
                            else:
                                desc_nodes = splitter.get_nodes_from_documents([TextNode(text=full_text, metadata={"file_name": file_name, "image_path": frame_info["path"], "page": frame_info["page"]})])
                                nodes.extend(desc_nodes)
                            frame_data.append({"page": frame_info["page"], "image_path": frame_info["path"], "description": desc})
                        else:
                            try:
                                os.remove(frame_info["path"])
                            except Exception:
                                pass
                        if progress_cb:
                            progress_cb(65 + int(done_count / n * 25), f"Описание PDF: {done_count}/{n}")
                except IngestionCancelled:
                    raise

        if shared_llm_url and not keep_vision_alive:
            unload_all_models(role="llm")

    if frame_data:
        frame_data.sort(key=lambda x: x["page"])
        metadata_json = {"file_name": file_name, "is_video": False, "transcript": [], "frames": frame_data}
        with open(os.path.join(os.path.dirname(file_path), f"{file_name}.json"), "w", encoding="utf-8") as f:
            json.dump(metadata_json, f, ensure_ascii=False, indent=2)
    return nodes


def process_pptx(file_path, images_dir, llm_settings=None, shared_llm_url=None,
                 progress_cb=None, cancel_check=None, keep_vision_alive=False):

    nodes = []
    file_name = os.path.basename(file_path)
    pdf_path = os.path.splitext(file_path)[0] + ".pdf"
    import pythoncom
    import win32com.client

    app = None
    deck = None
    try:
        pythoncom.CoInitialize()
        app = win32com.client.Dispatch("Powerpoint.Application")
        deck = app.Presentations.Open(os.path.abspath(file_path), WithWindow=False)
        deck.SaveAs(os.path.abspath(pdf_path), 32)
        if os.path.exists(pdf_path):
            if os.path.exists(file_path):
                os.remove(file_path)
            nodes = process_pdf(pdf_path, images_dir, llm_settings, shared_llm_url,
                                original_filename=os.path.basename(pdf_path),
                                progress_cb=progress_cb, cancel_check=cancel_check,
                                keep_vision_alive=keep_vision_alive)
        else:
            raise Exception("PDF conversion failed")
    except IngestionCancelled:
        raise
    except Exception as e:
        logger.warning(f"COM-конвертация PPTX не удалась, резерв через python-pptx: {e}")
        prs = Presentation(file_path)
        for i, slide in enumerate(prs.slides):
            nodes.append(TextNode(text="\n".join([sh.text for sh in slide.shapes if hasattr(sh, "text")]),
                                  metadata={"file_name": file_name, "page": i + 1}))
    finally:
        if deck is not None:
            try:
                deck.Close()
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
    return nodes


def process_docx(file_path, images_dir, llm_settings=None, shared_llm_url=None,
                 progress_cb=None, cancel_check=None, keep_vision_alive=False):

    nodes = []
    file_name = os.path.basename(file_path)
    pdf_path = os.path.splitext(file_path)[0] + ".pdf"
    import pythoncom
    import win32com.client

    app = None
    doc = None
    try:
        pythoncom.CoInitialize()
        app = win32com.client.Dispatch("Word.Application")
        doc = app.Documents.Open(os.path.abspath(file_path))
        doc.SaveAs(os.path.abspath(pdf_path), 17)
        if os.path.exists(pdf_path):
            if os.path.exists(file_path):
                os.remove(file_path)
            nodes = process_pdf(pdf_path, images_dir, llm_settings, shared_llm_url,
                                original_filename=os.path.basename(pdf_path),
                                progress_cb=progress_cb, cancel_check=cancel_check,
                                keep_vision_alive=keep_vision_alive)
        else:
            raise Exception("PDF conversion failed")
    except IngestionCancelled:
        raise
    except Exception as e:
        logger.warning(f"COM-конвертация DOCX не удалась, резерв через python-docx: {e}")
        import docx as _docx
        nodes.append(TextNode(text="\n".join([p.text for p in _docx.Document(file_path).paragraphs]),
                               metadata={"file_name": file_name}))
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
    return nodes
