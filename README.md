# image-to-gcode

Turns PNG/JPG images and PDF pages into G-code for a 3D printer converted into a
pen plotter. Every run first writes a `*_preview.png` of the planned toolpath, so
you can check it before sending anything to the machine.

## Setup

```sh
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
```

## Usage

```sh
# Writes drawing.gcode and drawing_preview.png next to the input
.venv/bin/image-to-gcode drawing.pdf

# Preview only, to check the result before generating G-code
.venv/bin/image-to-gcode logo.png --preview-only

## Settings

All settings are in [config.toml](config.toml), with comments. These are the ones you'll most likely change:

| Setting | What it does |
| --- | --- |
| `pen.down_z` / `pen.up_z` | Z height while drawing / while moving between strokes |
| `pen.safe_z` | Z height at the start and end of a job |
| `pen.down_gcode` / `pen.up_gcode` | Command templates for moving the pen. Change these for a servo pen lift (e.g. `M280 P0 S90`) |
| `machine.bed_width` / `bed_height` | Size of the area the pen can reach, in mm |
| `machine.origin_x` / `origin_y` | Machine coordinates of that area's front-left corner. Use these to compensate for the pen's offset from the nozzle |
| `machine.margin` | Blank border around the drawing |
| `machine.start_gcode` / `end_gcode` | Commands run before and after the drawing (homing is in `start_gcode`) |
| `image.mode` | `outline`, `centerline`, `hatch`, `contour`, or a combination such as `outline+hatch` |
| `image.threshold` | 0–255. Pixels darker than this get drawn. `"auto"` picks a value for you |

### Modes

- **outline**: traces the edge of every dark region. Good for logos and solid shapes.
- **centerline**: draws a single stroke down the middle of each line. Good for line art,
  handwriting, and PDF drawings, where `outline` would draw both sides of every line.
- **hatch**: fills dark regions with parallel lines (`hatch_spacing`, `hatch_angle`,
  `crosshatch`). Adjacent lines are joined in a zig-zag so the pen lifts less often.
- **contour**: fills dark regions with rings that shrink inward from the edges, like
  the perimeters in a slicer (`contour_spacing`). Follows the shape better than hatching.

## Preview

The preview shows the whole bed. The drawing is dark blue, pen-up travel moves are
light red, the margin is the dashed rectangle, and the origin is marked in the
bottom-left corner (red is +X, green is +Y). The footer shows the number of strokes,
the number of pen lifts, the distance drawn and travelled, and an estimated run time.
If the traced shapes don't look right, use `--save-mask` to see which pixels were
treated as ink, then adjust `image.threshold`, `image.invert` or `image.blur`.

## Pipeline

1. Load the image. PDFs are rendered at `pdf_dpi`, and transparent areas count as paper.
2. Threshold the image, crop it to the inked area, and scale it to fit inside the margins.
   The aspect ratio is preserved and the drawing is centred.
3. Resample so one pixel is `image.resolution` mm, then trace (OpenCV contours, distance-transform rings,
   Zhang–Suen skeleton, or scanline hatching) and simplify the paths.
4. Order strokes nearest-neighbour first. Strokes can be reversed and loops can start
   at any point. Gaps shorter than `join_distance` are drawn without lifting the pen.
5. Write G-code: `G21`/`G90`, your start code, travel moves as `G0`, drawing moves
   as `G1`, then park and run your end code. If any point falls outside the bed, the
   tool refuses to write the file.

## Tests

```sh
.venv/bin/pytest
```
