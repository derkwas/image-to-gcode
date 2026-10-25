"""Image -> ink mask -> polylines in machine millimetres."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from . import trace
from .config import Config


@dataclass
class Traced:
    paths: list[np.ndarray]  # polylines in machine mm (Y up)
    mask: np.ndarray  # the binary ink mask that was traced (True = ink)
    threshold: int
    mm_per_px: float


def image_to_paths(gray: np.ndarray, cfg: Config) -> Traced:
    ic, mc = cfg.image, cfg.machine
    gray = np.ascontiguousarray(np.rot90(gray, ic.rotate // 90 % 4))
    if ic.invert:
        gray = 255 - gray

    if ic.threshold == "auto":
        # Otsu puts its threshold in the dark class (0 for a pure black/white image),
        # so +1 to make "darker than threshold" include it.
        otsu, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        threshold = int(otsu) + 1
    else:
        threshold = int(ic.threshold)

    # Crop to the part of the image we'll draw.
    if ic.fit == "content":
        ys, xs = np.nonzero(gray < threshold)
        if len(xs) == 0:
            raise ValueError(f"no ink found at threshold {threshold}; try a different image.threshold or image.invert")
        pad = 2
        y0, y1 = max(ys.min() - pad, 0), min(ys.max() + 1 + pad, gray.shape[0])
        x0, x1 = max(xs.min() - pad, 0), min(xs.max() + 1 + pad, gray.shape[1])
        gray = gray[y0:y1, x0:x1]
    h, w = gray.shape

    # Scale to fit inside the margins, preserving aspect ratio.
    avail_w = mc.bed_width - 2 * mc.margin
    avail_h = mc.bed_height - 2 * mc.margin
    mm_per_src_px = min(avail_w / w, avail_h / h) * ic.scale
    draw_w, draw_h = w * mm_per_src_px, h * mm_per_src_px

    # Resample so one pixel is ~image.resolution mm, so all tolerances are consistent.
    new_w = max(int(round(draw_w / ic.resolution)), 1)
    new_h = max(int(round(draw_h / ic.resolution)), 1)
    interp = cv2.INTER_AREA if new_w < w else cv2.INTER_LINEAR
    gray = cv2.resize(np.ascontiguousarray(gray), (new_w, new_h), interpolation=interp)
    mm_per_px = draw_w / new_w
    if ic.blur > 0:
        gray = cv2.GaussianBlur(gray, (0, 0), ic.blur / mm_per_px)
    mask = gray < threshold

    eps = ic.simplify / mm_per_px
    paths: list[np.ndarray] = []
    for mode in (m.strip() for m in ic.mode.split("+")):
        if mode == "outline":
            paths += trace.outlines(mask, eps)
        elif mode == "centerline":
            paths += trace.centerlines(mask, eps)
        elif mode == "hatch":
            spacing = ic.hatch_spacing / mm_per_px
            paths += trace.hatch(mask, spacing, ic.hatch_angle)
            if ic.crosshatch:
                paths += trace.hatch(mask, spacing, ic.hatch_angle + 90)
        elif mode == "contour":
            paths += trace.contours(mask, ic.contour_spacing / mm_per_px, eps)

    # Pixel space (y down) -> machine mm (y up), centred in the drawable area.
    off_x = mc.origin_x + (mc.bed_width - draw_w) / 2
    off_y = mc.origin_y + (mc.bed_height - draw_h) / 2
    out = []
    for p in paths:
        mm = np.column_stack([off_x + p[:, 0] * mm_per_px, off_y + (new_h - p[:, 1]) * mm_per_px])
        if path_length(mm) >= ic.min_path_length:
            out.append(mm)
    return Traced(out, mask, threshold, mm_per_px)


def path_length(p: np.ndarray) -> float:
    return float(np.hypot(*np.diff(p, axis=0).T).sum()) if len(p) > 1 else 0.0
