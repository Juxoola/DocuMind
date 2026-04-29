# coding: utf-8
import fitz
doc = fitz.open('data/Лекция_1.pdf')
for i in range(len(doc)):
    p = doc.load_page(i)
    draws = p.get_drawings()
    imgs = p.get_images()
    if draws:
        types = set(d.get('type') for d in draws)
        print(f"Page {i+1}: {len(draws)} drawings, types={types}, {len(imgs)} images")
    else:
        print(f"Page {i+1}: no drawings, {len(imgs)} images")
