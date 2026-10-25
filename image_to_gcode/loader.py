"""Load PNG/JPG/PDF inputs as 8-bit grayscale arrays (0 = black, 255 = white)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def load_grayscale(path: Path, page: int = 1, pdf_dpi: int = 300) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        return _load_pdf(path, page, pdf_dpi)
    with Image.open(path) as img:
        img.load()
        if img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info):
            # Transparent areas become paper, not ink.
            rgba = img.convert("RGBA")
            background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            img = Image.alpha_composite(background, rgba)
        return np.asarray(img.convert("L"), dtype=np.uint8).copy()


def _load_pdf(path: Path, page: int, dpi: int) -> np.ndarray:
    import pymupdf

    with pymupdf.open(path) as doc:
        if not 1 <= page <= doc.page_count:
            raise ValueError(f"{path.name} has {doc.page_count} page(s); page {page} requested")
        pix = doc[page - 1].get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY, alpha=False)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.stride)
        return arr[:, : pix.width].copy()
