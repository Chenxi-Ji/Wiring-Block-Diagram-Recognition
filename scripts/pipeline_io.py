"""Shared input/output helpers for the formal parsing scripts."""
from __future__ import annotations

import json

import cv2
import fitz
import numpy as np
from PIL import Image
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


def render_pdf_page(page: fitz.Page, dpi: int) -> np.ndarray:
    zoom = dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    return img


def save_json(path, data) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def image_to_pdf(images, output_pdf) -> None:
    c = canvas.Canvas(str(output_pdf))
    for img in images:
        h, w = img.shape[:2]
        c.setPageSize((w, h))
        c.drawImage(ImageReader(Image.fromarray(img)), 0, 0, width=w, height=h)
        c.showPage()
    c.save()


def parse_page_range(page_str: str | None):
    if page_str is None:
        return None
    try:
        start, end = page_str.split("-")
        start, end = int(start), int(end)
        if start <= 0 or end < start:
            raise ValueError()
        return start, end
    except Exception:
        raise ValueError(f"Invalid page range: {page_str}")
