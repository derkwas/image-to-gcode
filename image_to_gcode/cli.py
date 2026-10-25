"""Command-line interface: image/PDF in, preview PNG + G-code out."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

from . import gcode, preview
from .config import load_config
from .loader import load_grayscale
from .optimize import order_paths
from .pipeline import image_to_paths

DEFAULT_CONFIG = Path("config.toml")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="image-to-gcode",
        description="Convert a PNG/JPG/PDF into pen-plotter G-code, with a PNG preview of the toolpath.",
    )
    p.add_argument("input", type=Path, help="image (.png, .jpg, ...) or .pdf")
    p.add_argument("-o", "--output", type=Path, help="G-code path (default: <input>.gcode next to the input)")
    p.add_argument("-c", "--config", type=Path, help="TOML settings file (default: ./config.toml if present)")
    p.add_argument("--preview-only", action="store_true", help="write the preview PNG but no G-code")
    p.add_argument("--save-mask", action="store_true", help="also save the thresholded ink mask, for tuning")

    g = p.add_argument_group("common overrides (anything else: --set section.key=value)")
    g.add_argument("--pen-down", type=float, metavar="Z", help="pen.down_z")
    g.add_argument("--pen-up", type=float, metavar="Z", help="pen.up_z")
    g.add_argument("--bed", type=float, nargs=2, metavar=("W", "H"), help="machine.bed_width / bed_height in mm")
    g.add_argument("--margin", type=float, metavar="MM", help="machine.margin")
    g.add_argument("--mode", help="image.mode: outline, centerline, hatch, or e.g. outline+hatch")
    g.add_argument("--threshold", help='image.threshold: 0-255 or "auto"')
    g.add_argument("--page", type=int, help="image.page (PDF page, 1-based)")
    g.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE",
                   help="override any config value, e.g. --set image.hatch_spacing=1.2 (repeatable)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    overrides = []
    if args.pen_down is not None:
        overrides.append(f"pen.down_z={args.pen_down}")
    if args.pen_up is not None:
        overrides.append(f"pen.up_z={args.pen_up}")
    if args.bed:
        overrides += [f"machine.bed_width={args.bed[0]}", f"machine.bed_height={args.bed[1]}"]
    if args.margin is not None:
        overrides.append(f"machine.margin={args.margin}")
    if args.mode:
        overrides.append(f'image.mode="{args.mode}"')
    if args.threshold:
        overrides.append(f"image.threshold={args.threshold if args.threshold.isdigit() else '\"auto\"'}")
    if args.page is not None:
        overrides.append(f"image.page={args.page}")
    overrides += args.set

    config_path = args.config or (DEFAULT_CONFIG if DEFAULT_CONFIG.exists() else None)
    try:
        cfg = load_config(config_path, overrides)
        gray = load_grayscale(args.input, cfg.image.page, cfg.image.pdf_dpi)
        traced = image_to_paths(gray, cfg)
    except (KeyError, ValueError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    paths = traced.paths
    if cfg.output.optimize:
        paths = order_paths(paths, (cfg.machine.park_x, cfg.machine.park_y))
    stats = gcode.compute_stats(paths, cfg)

    out = args.output or args.input.with_suffix(".gcode")
    preview_path = out.with_name(out.stem + "_preview.png")
    preview.render(paths, cfg, stats).save(preview_path)
    print(f"config:  {config_path or 'built-in defaults'}")
    print(f"traced:  {stats.strokes} strokes at threshold {traced.threshold}, "
          f"{traced.mm_per_px:.3f} mm/px, {stats.pen_lifts} pen lifts, ~{stats.minutes:.1f} min")
    print(f"preview: {preview_path}")
    if args.save_mask:
        mask_path = out.with_name(out.stem + "_mask.png")
        Image.fromarray((~traced.mask).astype("uint8") * 255).save(mask_path)
        print(f"mask:    {mask_path}")
    if not args.preview_only:
        try:
            out.write_text(gcode.generate(paths, cfg, source=args.input.name))
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        print(f"gcode:   {out}")
    return 0
