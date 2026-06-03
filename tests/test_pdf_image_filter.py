"""
Тест функции-фильтра, которая решает, какие страницы PDF отправлять в Vision.

Применяет к каждой странице ДВА варианта фильтра:
  - OLD: исходный из src/ingestion.py (graphics_weight > 25 + sum(fill) > 2.9)
  - NEW: исправленный (graphics_weight > 8 + детектор таблиц по тонким линиям)

Сравнивает, какие страницы попадают в Vision в каждом варианте.

Использование:
  python tests/test_pdf_image_filter.py <путь_к_pdf>
  python tests/test_pdf_image_filter.py notebooks/1896cc13/data/Лекция 3 РпСАПР.pdf
"""
import os
import sys
import argparse
import fitz


def analyze_page_old(page):
    """Оригинальный фильтр (до моих правок) — для сравнения."""
    images = page.get_images()
    drawings = page.get_drawings()
    if len(images) > 0:
        return True, "raster image"
    graphics_weight = 0
    for d in drawings:
        items = d.get('items', [])
        if any(i[0] in ['c', 'q'] for i in items) or len(items) > 12:
            return True, "curves/complex"
        fill = d.get('fill')
        is_white_rect = len(items) == 1 and items[0][0] == 're' and fill and (sum(fill) > 2.9)
        if is_white_rect:
            continue
        graphics_weight += 1
    if graphics_weight > 25:
        return True, f"gw={graphics_weight}>25"
    return False, f"text-only (gw={graphics_weight})"


def analyze_page_new(page):
    """Новый фильтр (A+B фикс из коммита 7795616)."""
    images = page.get_images()
    drawings = page.get_drawings()
    if len(images) > 0:
        return True, "raster image"
    has_real_graphics = False
    graphics_weight = 0
    horizontal_lines = 0
    vertical_lines = 0
    for d in drawings:
        items = d.get('items', [])
        if any(i[0] in ['c', 'q'] for i in items) or len(items) > 12:
            return True, "curves/complex"
        rect = d.get('rect')
        fill = d.get('fill')
        is_page_background = (
            len(items) == 1
            and items[0][0] == 're'
            and fill is not None
            and all(c >= 0.99 for c in fill)
            and rect is not None
            and (rect.x1 - rect.x0) > 200
            and (rect.y1 - rect.y0) > 200
        )
        if is_page_background:
            continue
        if rect is not None:
            w = rect.x1 - rect.x0
            h = rect.y1 - rect.y0
            if w > 30 and h < 3:
                horizontal_lines += 1
            elif w < 3 and h > 30:
                vertical_lines += 1
        graphics_weight += 1
    if graphics_weight > 8:
        return True, f"gw={graphics_weight}>8"
    if horizontal_lines >= 3 and vertical_lines >= 1:
        return True, f"table H={horizontal_lines},V={vertical_lines}"
    if horizontal_lines + vertical_lines >= 6:
        return True, f"lines H={horizontal_lines},V={vertical_lines}"
    return False, f"text-only (gw={graphics_weight}, H={horizontal_lines}, V={vertical_lines})"


def main():
    parser = argparse.ArgumentParser(description="Сравнение OLD vs NEW фильтра изображений в PDF")
    parser.add_argument("pdf_path", help="Путь к PDF-файлу")
    parser.add_argument("--show-text", action="store_true", help="Показывать текст каждой страницы")
    args = parser.parse_args()

    if not os.path.exists(args.pdf_path):
        print(f"Файл не найден: {args.pdf_path}")
        sys.exit(1)

    doc = fitz.open(args.pdf_path)
    name = os.path.basename(args.pdf_path)
    print(f"=== {name} ===")
    print(f"Страниц: {len(doc)}")
    print()

    header = f"{'Page':<5} {'Imgs':<5} {'Draws':<7} {'OLD verdict':<35} {'NEW verdict':<40} {'Changed?'}"
    print(header)
    print("-" * len(header))

    old_kept = 0
    new_kept = 0
    changed_pages = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        images = page.get_images()
        drawings = page.get_drawings()

        old_kept_flag, old_reason = analyze_page_old(page)
        new_kept_flag, new_reason = analyze_page_new(page)

        if old_kept_flag:
            old_kept += 1
        if new_kept_flag:
            new_kept += 1
        changed = old_kept_flag != new_kept_flag
        if changed:
            changed_pages.append(page_num + 1)

        marker = " ←" if changed else ""
        old_v = "✓ " + old_reason if old_kept_flag else "✗ " + old_reason
        new_v = "✓ " + new_reason if new_kept_flag else "✗ " + new_reason
        print(f"{page_num+1:<5} {len(images):<5} {len(drawings):<7} {old_v:<35} {new_v:<40}{marker}")

        if args.show_text and changed:
            text = page.get_text().strip()[:150].replace("\n", " ")
            print(f"      text: {text}...")

    print()
    print("=" * 70)
    print(f"OLD фильтр: {old_kept}/{len(doc)} страниц → vision ({100*old_kept/len(doc):.0f}%)")
    print(f"NEW фильтр: {new_kept}/{len(doc)} страниц → vision ({100*new_kept/len(doc):.0f}%)")
    if changed_pages:
        print(f"Разница: страницы {changed_pages} теперь попадают в vision")
    else:
        print("Разницы нет")


if __name__ == "__main__":
    main()
