"""Tunable settings, loaded from a TOML file and/or command-line overrides."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path


@dataclass
class MachineConfig:
    # Drawable area in mm. The pen can reach X in [origin_x, origin_x + bed_width]
    # and Y in [origin_y, origin_y + bed_height].
    bed_width: float = 220.0
    bed_height: float = 220.0
    origin_x: float = 0.0
    origin_y: float = 0.0
    # Blank border kept around the drawing, in mm.
    margin: float = 10.0
    travel_feed: float = 3000.0  # mm/min, pen up
    draw_feed: float = 1500.0  # mm/min, pen down
    # Where the pen goes when finished (machine coordinates).
    park_x: float = 0.0
    park_y: float = 0.0
    start_gcode: str = "G28 ; home all axes"
    end_gcode: str = "M84 ; disable motors"


@dataclass
class PenConfig:
    down_z: float = 0.0  # Z height while drawing
    up_z: float = 3.0  # Z height while travelling between strokes
    safe_z: float = 10.0  # Z height at start/end and when parking
    z_feed: float = 600.0  # mm/min for pen up/down moves
    down_dwell_ms: int = 0  # pause after lowering the pen
    up_dwell_ms: int = 0  # pause after lifting the pen
    # Command templates; {z} and {z_feed} are substituted. Replace these with
    # e.g. "M280 P0 S90" if your pen is on a servo instead of the Z axis.
    down_gcode: str = "G1 Z{z:.3f} F{z_feed:.0f}"
    up_gcode: str = "G1 Z{z:.3f} F{z_feed:.0f}"
    width: float = 0.4  # line width in mm, only used for the preview


@dataclass
class ImageConfig:
    # outline | centerline | hatch, combine with "+" (e.g. "outline+hatch").
    mode: str = "outline"
    # 0-255; pixels darker than this are inked. "auto" picks one (Otsu).
    threshold: int | str = "auto"
    invert: bool = False  # treat light pixels as ink
    page: int = 1  # PDF page number, 1-based
    pdf_dpi: int = 300
    rotate: int = 0  # degrees counter-clockwise: 0, 90, 180 or 270
    # "content" scales the inked region to fill the bed, "page" keeps whitespace.
    fit: str = "content"
    scale: float = 1.0  # fraction of the available area to fill
    resolution: float = 0.1  # mm per pixel used while tracing
    blur: float = 0.0  # gaussian blur radius in mm before thresholding
    simplify: float = 0.05  # max deviation in mm when simplifying paths
    min_path_length: float = 0.5  # drop strokes shorter than this, in mm
    hatch_spacing: float = 0.8  # mm between hatch lines
    hatch_angle: float = 45.0  # degrees
    crosshatch: bool = False  # add a second hatch pass at +90 degrees


@dataclass
class OutputConfig:
    optimize: bool = True  # reorder strokes to minimise pen-up travel
    join_distance: float = 0.3  # don't lift the pen for gaps smaller than this (mm)
    preview_scale: float = 4.0  # preview pixels per mm
    preview_travel: bool = True  # draw pen-up moves in the preview


@dataclass
class Config:
    machine: MachineConfig = field(default_factory=MachineConfig)
    pen: PenConfig = field(default_factory=PenConfig)
    image: ImageConfig = field(default_factory=ImageConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def set(self, section: str, key: str, value) -> None:
        sub = getattr(self, section, None)
        if sub is None or section not in {f.name for f in fields(self)}:
            raise KeyError(f"unknown config section [{section}]")
        types = {f.name: f.type for f in fields(sub)}
        if key not in types:
            raise KeyError(f"unknown config key {section}.{key}")
        if types[key] == "float" and isinstance(value, int) and not isinstance(value, bool):
            value = float(value)
        setattr(sub, key, value)

    def update(self, data: dict) -> None:
        for section, values in data.items():
            if not isinstance(values, dict):
                raise KeyError(f"expected a [section] table, got top-level key {section!r}")
            for key, value in values.items():
                self.set(section, key, value)

    def validate(self) -> None:
        m, p, i = self.machine, self.pen, self.image
        if m.bed_width <= 2 * m.margin or m.bed_height <= 2 * m.margin:
            raise ValueError("margin leaves no drawable area on the bed")
        if p.up_z <= p.down_z:
            raise ValueError(f"pen.up_z ({p.up_z}) must be above pen.down_z ({p.down_z})")
        if p.safe_z < p.up_z:
            raise ValueError(f"pen.safe_z ({p.safe_z}) must be at or above pen.up_z ({p.up_z})")
        for mode in i.mode.split("+"):
            if mode.strip() not in {"outline", "centerline", "hatch"}:
                raise ValueError(f"unknown image.mode {mode!r}")
        if i.fit not in {"content", "page"}:
            raise ValueError(f"image.fit must be 'content' or 'page', got {i.fit!r}")
        if i.rotate % 90:
            raise ValueError("image.rotate must be a multiple of 90")
        if not 0 < i.scale <= 1:
            raise ValueError("image.scale must be in (0, 1]")
        if i.threshold != "auto" and not (isinstance(i.threshold, int) and 0 <= i.threshold <= 255):
            raise ValueError("image.threshold must be 0-255 or \"auto\"")


def parse_override(text: str) -> tuple[str, str, object]:
    """Parse "section.key=value"; the value is read as TOML, falling back to a string."""
    name, sep, raw = text.partition("=")
    section, dot, key = name.strip().partition(".")
    if not sep or not dot:
        raise ValueError(f"override must look like section.key=value, got {text!r}")
    try:
        value = tomllib.loads(f"v = {raw}")["v"]
    except tomllib.TOMLDecodeError:
        value = raw
    return section, key.strip(), value


def load_config(path: Path | None = None, overrides: list[str] = ()) -> Config:
    cfg = Config()
    if path is not None:
        with open(path, "rb") as f:
            cfg.update(tomllib.load(f))
    for text in overrides:
        cfg.set(*parse_override(text))
    cfg.validate()
    return cfg
