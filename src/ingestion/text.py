"""Обработка текстовых документов: PDF, PPTX, DOCX."""

import json
import logging
import os
import subprocess
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

import fitz
from llama_index.core.schema import TextNode

import config
from src.gguf.server import unload_all_models
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


def _analyze_and_build_page(page_num, doc, images_dir, file_name, splitter):

    page = doc.load_page(page_num)
    text, has_real_graphics = _analyze_page_for_vision(page)
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


def process_pdf(
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
    doc = fitz.open(file_path)
    frame_data = []
    frame_list = []
    splitter = _get_splitter()
    total_pages = len(doc)
    n_workers = min(8, (os.cpu_count() or 4), total_pages)

    try:
        # Один проход: анализ страницы + построение узлов + рендер pixmap
        # в одном ThreadPoolExecutor — вместо двух последовательных пулов.
        # Батчами по BATCH_SIZE чтобы не держать всеPixmap в памяти.
        BATCH_SIZE = 16
        if n_workers <= 1:
            for page_num in range(total_pages):
                if _is_cancelled():
                    raise IngestionCancelled(f"Cancelled at page {page_num + 1}")
                pn, local_nodes, image_path = _analyze_and_build_page(
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
                with ThreadPoolExecutor(max_workers=n_workers) as ex:
                    futures = []
                    for page_num in range(batch_start, batch_end):
                        futures.append(
                            ex.submit(
                                _analyze_and_build_page,
                                page_num,
                                doc,
                                images_dir,
                                file_name,
                                splitter,
                            )
                        )
                    artifacts = [None] * len(futures)
                    for i, fut in enumerate(as_completed(futures)):
                        if _is_cancelled():
                            ex.shutdown(wait=False, cancel_futures=True)
                            raise IngestionCancelled("Cancelled during page processing")
                        pn, local_nodes, image_path = fut.result()
                        artifacts[i] = (pn, local_nodes, image_path)
                    for page_num_result, local_nodes, image_path in artifacts:
                        nodes.extend(local_nodes)
                        if image_path:
                            frame_list.append({"page": page_num_result + 1, "path": image_path})

        if frame_list:
            if shared_llm_url is None:
                shared_llm_url = get_vision_url(llm_settings)
            if shared_llm_url:
                v_conc = int(llm_settings.get("vision_concurrency") or config.VISION_CONCURRENCY)
                n = len(frame_list)
                if progress_cb:
                    progress_cb(
                        65,
                        f"Анализ {n} страниц PDF ({'параллельно' if v_conc > 1 else 'последовательно'})...",
                    )

                with ThreadPoolExecutor(max_workers=v_conc) as executor:
                    futures = {
                        executor.submit(
                            describe_image_with_lmstudio, f["path"], llm_settings, shared_llm_url
                        ): f
                        for f in frame_list
                    }
                    done_count = 0
                    results = []
                    try:
                        for future in as_completed(futures):
                            if _is_cancelled():
                                executor.shutdown(wait=False, cancel_futures=True)
                                raise IngestionCancelled(f"Cancelled during OCR ({done_count}/{n})")
                            frame_info = futures[future]
                            desc = future.result()
                            done_count += 1
                            results.append((frame_info, desc))
                            if progress_cb:
                                progress_cb(
                                    65 + int(done_count / n * 25), f"Описание PDF: {done_count}/{n}"
                                )
                    except IngestionCancelled:
                        raise

                    # Сортируем по номеру страницы перед добавлением в nodes
                    results.sort(key=lambda x: x[0]["page"])
                    for frame_info, desc in results:
                        if desc and "Изображение без описания" not in desc:
                            full_text = (
                                f"Изображение PDF {file_name} стр {frame_info['page']}: {desc}"
                            )
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
                                os.remove(frame_info["path"])
                            except Exception:
                                pass

            if shared_llm_url and not keep_vision_alive:
                unload_all_models(role="llm")

        if frame_data:
            frame_data.sort(key=lambda x: x["page"])
            metadata_json = {
                "file_name": file_name,
                "is_video": False,
                "transcript": [],
                "frames": frame_data,
            }
            with open(
                os.path.join(os.path.dirname(file_path), f"{file_name}.json"), "w", encoding="utf-8"
            ) as f:
                json.dump(metadata_json, f, ensure_ascii=False, indent=2)
    finally:
        doc.close()
    return nodes


def _find_soffice():
    """Находит soffice.exe LibreOffice: bin/libreoffice/ → PATH → Program Files."""
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


def _convert_via_libreoffice(file_path):
    """Конвертирует документ в PDF через LibreOffice headless.

    Возвращает путь к PDF или выбрасывает исключение.
    """
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
    result = subprocess.run(cmd, capture_output=True, timeout=120)
    pdf_path = os.path.splitext(file_path)[0] + ".pdf"
    if result.returncode == 0 and os.path.exists(pdf_path):
        logger.info(f"[DOCX] Сконвертировано в PDF через LibreOffice: {os.path.basename(pdf_path)}")
        # Удаляем оригинальный docx/pptx после успешной конвертации
        try:
            os.remove(file_path)
        except OSError:
            pass
        return pdf_path
    raise RuntimeError(
        f"LibreOffice конвертация не удалась (code={result.returncode}): "
        f"{(result.stderr or b'').decode('utf-8', errors='replace')[:200]}"
    )


def _convert_via_com(file_path, app_name, format_code):
    """Конвертирует Office-документ в PDF через COM.

    Возвращает путь к PDF или выбрасывает исключение.
    Используется как fallback для PPTX (PowerPoint) и DOCX (Word).
    """
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


def process_pptx(
    file_path,
    images_dir,
    llm_settings=None,
    shared_llm_url=None,
    progress_cb=None,
    cancel_check=None,
    keep_vision_alive=False,
):
    """PPTX → PDF (LibreOffice) → process_pdf с Vision-анализом изображений."""
    file_name = os.path.basename(file_path)

    # Всегда конвертируем pptx в PDF для анализа изображений
    try:
        pdf_path = _convert_via_libreoffice(file_path)
    except FileNotFoundError:
        logger.info("[PPTX] LibreOffice не найден, пробую COM-конвертацию...")
        try:
            pdf_path = _convert_via_com(file_path, "Powerpoint.Application", 32)
        except IngestionCancelled:
            raise
        except Exception as e:
            logger.warning(f"[PPTX] COM-конвертация тоже не удалась: {e}")
            return _process_pptx_textonly(file_path, file_name)
    except IngestionCancelled:
        raise
    except Exception as e:
        logger.warning(f"[PPTX] LibreOffice конвертация не удалась: {e}")
        try:
            pdf_path = _convert_via_com(file_path, "Powerpoint.Application", 32)
        except IngestionCancelled:
            raise
        except Exception:
            return _process_pptx_textonly(file_path, file_name)

    # После конвертации file_name должен ссылаться на PDF
    file_name = os.path.basename(pdf_path)

    return process_pdf(
        pdf_path,
        images_dir,
        llm_settings,
        shared_llm_url,
        original_filename=file_name,
        progress_cb=progress_cb,
        cancel_check=cancel_check,
        keep_vision_alive=keep_vision_alive,
    )


def _process_pptx_textonly(file_path, file_name):
    """Fallback: извлечь текст через python-pptx без анализа изображений."""
    try:
        from pptx import Presentation

        prs = Presentation(file_path)
        nodes = []
        for i, slide in enumerate(prs.slides):
            text = "\n".join([sh.text for sh in slide.shapes if hasattr(sh, "text")])
            if text.strip():
                nodes.append(TextNode(text=text, metadata={"file_name": file_name, "page": i + 1}))
        if nodes:
            logger.info(f"[PPTX] Fallback: {len(nodes)} слайдов через python-pptx (без Vision)")
            return nodes
    except Exception as e:
        logger.warning(f"[PPTX] python-pptx fallback тоже не удался: {e}")
    return []


def process_docx(
    file_path,
    images_dir,
    llm_settings=None,
    shared_llm_url=None,
    progress_cb=None,
    cancel_check=None,
    keep_vision_alive=False,
):
    """DOCX → PDF (LibreOffice) → process_pdf с Vision-анализом изображений."""
    file_name = os.path.basename(file_path)

    # Всегда конвертируем docx в PDF для корректного отображения и анализа изображений
    try:
        pdf_path = _convert_via_libreoffice(file_path)
    except FileNotFoundError:
        logger.info("[DOCX] LibreOffice не найден, пробую COM-конвертацию...")
        try:
            pdf_path = _convert_via_com(file_path, "Word.Application", 17)
        except IngestionCancelled:
            raise
        except Exception as e:
            logger.warning(f"[DOCX] COM-конвертация тоже не удалась: {e}")
            return _process_docx_textonly(file_path, file_name)
    except IngestionCancelled:
        raise
    except Exception as e:
        logger.warning(f"[DOCX] LibreOffice конвертация не удалась: {e}")
        try:
            pdf_path = _convert_via_com(file_path, "Word.Application", 17)
        except IngestionCancelled:
            raise
        except Exception:
            return _process_docx_textonly(file_path, file_name)

    # После конвертации file_name должен ссылаться на PDF
    file_name = os.path.basename(pdf_path)

    return process_pdf(
        pdf_path,
        images_dir,
        llm_settings,
        shared_llm_url,
        original_filename=file_name,
        progress_cb=progress_cb,
        cancel_check=cancel_check,
        keep_vision_alive=keep_vision_alive,
    )


def _process_docx_textonly(file_path, file_name):
    """Fallback: извлечь текст через python-docx без анализа изображений."""
    try:
        import docx as _docx

        text = "\n".join([p.text for p in _docx.Document(file_path).paragraphs])
        if text.strip():
            logger.info("[DOCX] Fallback: текст извлечён через python-docx (без Vision)")
            return [TextNode(text=text, metadata={"file_name": file_name})]
    except Exception as e:
        logger.warning(f"[DOCX] python-docx fallback тоже не удался: {e}")
    return []
