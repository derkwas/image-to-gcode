import re

import numpy as np
import pytest
from PIL import Image, ImageDraw

from image_to_gcode import gcode
from image_to_gcode.cli import main
from image_to_gcode.config import load_config
from image_to_gcode.loader import load_grayscale
from image_to_gcode.optimize import order_paths
from image_to_gcode.pipeline import image_to_paths


def drawing() -> np.ndarray:
    img = Image.new("L", (400, 300), 255)
    d = ImageDraw.Draw(img)
    d.ellipse([20, 20, 180, 180], outline=0, width=6)
    d.rectangle([220, 40, 380, 260], fill=0)
    return np.asarray(img)


def xyz(text: str):
    """Collect every X/Y/Z value written in a G-code program."""
    return {axis: [float(v) for v in re.findall(rf"{axis}(-?\d+\.?\d*)", text)] for axis in "XYZ"}


@pytest.mark.parametrize("mode", ["outline", "centerline", "hatch", "contour", "outline+hatch"])
def test_paths_fit_inside_margins(mode):
    cfg = load_config(None, [f'image.mode="{mode}"', "machine.bed_width=150", "machine.bed_height=100",
                             "machine.origin_x=5", "machine.origin_y=7", "machine.margin=10"])
    paths = image_to_paths(drawing(), cfg).paths
    assert paths
    pts = np.vstack(paths)
    assert pts[:, 0].min() >= 5 + 10 - 0.5 and pts[:, 0].max() <= 5 + 150 - 10 + 0.5
    assert pts[:, 1].min() >= 7 + 10 - 0.5 and pts[:, 1].max() <= 7 + 100 - 10 + 0.5


def test_gcode_uses_configured_pen_heights():
    cfg = load_config(None, ["pen.down_z=1.25", "pen.up_z=4.5", "pen.safe_z=8"])
    paths = order_paths(image_to_paths(drawing(), cfg).paths, (0, 0))
    text = gcode.generate(paths, cfg)
    assert set(xyz(text)["Z"]) == {1.25, 4.5, 8.0}
    # Every drawing move happens with the pen down.
    z = None
    for line in text.splitlines():
        if m := re.match(r"G1 Z(\S+)", line):
            z = float(m.group(1))
        elif line.startswith("G1 X"):
            assert z == 1.25


def test_servo_pen_template():
    cfg = load_config(None, ['pen.down_gcode="M280 P0 S90"', 'pen.up_gcode="M280 P0 S30"'])
    text = gcode.generate(image_to_paths(drawing(), cfg).paths, cfg)
    assert "M280 P0 S90" in text and "M280 P0 S30" in text


def test_image_is_not_mirrored():
    # Ink only in the top-left of the image must land at low X, high Y on the bed.
    img = np.full((100, 100), 255, np.uint8)
    img[5:20, 5:20] = 0
    img[95:, 95:] = 0  # anchor the far corner so "content" fit keeps the whole frame
    cfg = load_config(None, ["image.fit=\"page\""])
    pts = np.vstack(image_to_paths(img, cfg).paths)
    top_left = pts[(pts[:, 0] < 110) & (pts[:, 1] > 110)]
    assert len(top_left)


def test_centerline_keeps_loops_closed():
    img = Image.new("L", (300, 300), 255)
    ImageDraw.Draw(img).rectangle([50, 50, 250, 250], outline=0, width=5)
    cfg = load_config(None, ['image.mode="centerline"'])
    paths = image_to_paths(np.asarray(img), cfg).paths
    loop = max(paths, key=len)
    assert np.allclose(loop[0], loop[-1])
    span = loop.max(axis=0) - loop.min(axis=0)
    assert span.min() > 150  # all four sides present


def test_contour_rings_shrink_inward():
    img = np.full((200, 200), 255, np.uint8)
    img[20:180, 20:180] = 0
    cfg = load_config(None, ['image.mode="contour"', "image.contour_spacing=2", "machine.margin=10"])
    paths = image_to_paths(img, cfg).paths
    widths = sorted((p[:, 0].max() - p[:, 0].min() for p in paths), reverse=True)
    # The 160 px square (plus 2 px crop padding each side) is scaled to 200 mm,
    # then filled with rings 2 mm apart: ~48 rings, each 4 mm narrower.
    square = 200 * 160 / 164
    assert 45 <= len(widths) <= 51
    assert widths[0] == pytest.approx(square - 2, abs=1.5)
    assert np.allclose(np.diff(widths[:10]), -4, atol=0.5)


def test_rejects_bad_settings():
    with pytest.raises(ValueError):
        load_config(None, ["pen.up_z=0", "pen.down_z=1"])
    with pytest.raises(KeyError):
        load_config(None, ["pen.nonsense=1"])


def test_pdf_and_cli(tmp_path):
    pymupdf = pytest.importorskip("pymupdf")
    pdf = tmp_path / "in.pdf"
    with pymupdf.open() as doc:
        page = doc.new_page(width=200, height=100)
        page.draw_rect(pymupdf.Rect(20, 20, 180, 80), color=(0, 0, 0), width=2)
        doc.save(pdf)
    assert load_grayscale(pdf).min() < 128

    out = tmp_path / "out.gcode"
    assert main([str(pdf), "-o", str(out), "--pen-down", "0.5", "--pen-up", "2", "--bed", "100", "100"]) == 0
    assert (tmp_path / "out_preview.png").exists()
    coords = xyz(out.read_text())
    assert 0.5 in coords["Z"] and 2.0 in coords["Z"]
    assert max(coords["X"]) <= 100 and max(coords["Y"]) <= 100

    assert main([str(pdf), "-o", str(tmp_path / "p.gcode"), "--preview-only"]) == 0
    assert (tmp_path / "p_preview.png").exists() and not (tmp_path / "p.gcode").exists()
