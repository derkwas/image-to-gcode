"""Render the planned toolpath to a PNG so it can be checked before plotting."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from .config import Config
from .gcode import Stats, strokes_with_lifts

PAD = 20  # px around the bed
FOOTER = 24  # px for the stats line


def render(paths: list[np.ndarray], cfg: Config, stats: Stats) -> Image.Image:
    mc, oc = cfg.machine, cfg.output
    s = oc.preview_scale
    bed_w, bed_h = int(round(mc.bed_width * s)), int(round(mc.bed_height * s))
    img = Image.new("RGB", (bed_w + 2 * PAD, bed_h + 2 * PAD + FOOTER), (225, 225, 225))
    draw = ImageDraw.Draw(img)

    def px(pt) -> tuple[float, float]:
        return PAD + (pt[0] - mc.origin_x) * s, PAD + (mc.origin_y + mc.bed_height - pt[1]) * s

    draw.rectangle([PAD, PAD, PAD + bed_w, PAD + bed_h], fill="white", outline=(120, 120, 120))
    m = mc.margin * s
    _dashed_rect(draw, PAD + m, PAD + m, PAD + bed_w - m, PAD + bed_h - m, (190, 190, 210))
    # Origin marker: front-left corner of the bed (machine X/Y minimum).
    ox, oy = px((mc.origin_x, mc.origin_y))
    draw.line([ox, oy, ox + 30, oy], fill=(200, 60, 60), width=2)
    draw.line([ox, oy, ox, oy - 30], fill=(60, 160, 60), width=2)

    line_w = max(1, int(round(cfg.pen.width * s)))
    if oc.preview_travel:
        pos = (mc.park_x, mc.park_y)
        for p, lift in strokes_with_lifts(paths, cfg):
            if lift:
                draw.line([px(pos), px(p[0])], fill=(255, 150, 150), width=1)
            pos = p[-1]
        if paths:
            draw.line([px(pos), px((mc.park_x, mc.park_y))], fill=(255, 150, 150), width=1)

    pos = None
    for p, lift in strokes_with_lifts(paths, cfg):
        pts = [px(q) for q in p]
        if not lift:
            pts.insert(0, px(pos))
        draw.line(pts, fill=(20, 20, 40), width=line_w, joint="curve")
        pos = p[-1]

    text = (
        f"{mc.bed_width:g} x {mc.bed_height:g} mm   {stats.strokes} strokes   "
        f"{stats.pen_lifts} lifts   draw {stats.draw_mm / 1000:.2f} m   "
        f"travel {stats.travel_mm / 1000:.2f} m   ~{stats.minutes:.1f} min"
    )
    draw.text((PAD, PAD * 1.5 + bed_h), text, fill=(40, 40, 40))
    return img


def _dashed_rect(draw: ImageDraw.ImageDraw, x0, y0, x1, y1, color, dash=6) -> None:
    for a, b in (((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))):
        length = float(np.hypot(b[0] - a[0], b[1] - a[1]))
        for t in np.arange(0, length, 2 * dash):
            t0, t1 = t / length, min(t + dash, length) / length
            draw.line(
                [a[0] + (b[0] - a[0]) * t0, a[1] + (b[1] - a[1]) * t0,
                 a[0] + (b[0] - a[0]) * t1, a[1] + (b[1] - a[1]) * t1],
                fill=color,
            )
