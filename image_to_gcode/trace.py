"""Turn a binary ink mask into polylines, in pixel coordinates (x right, y down)."""

from __future__ import annotations

import math

import cv2
import numpy as np

Polyline = np.ndarray  # shape (N, 2), float


def outlines(mask: np.ndarray, epsilon_px: float) -> list[Polyline]:
    """Closed contours around every inked region (and around holes in them)."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    paths = []
    for c in contours:
        if len(c) < 2:
            continue
        c = cv2.approxPolyDP(c, epsilon_px, closed=True) if epsilon_px > 0 else c
        pts = c.reshape(-1, 2).astype(float) + 0.5
        paths.append(np.vstack([pts, pts[:1]]))
    return paths


def contours(mask: np.ndarray, spacing_px: float, epsilon_px: float) -> list[Polyline]:
    """Concentric rings shrinking inward from each edge, like slicer perimeters."""
    # Pad so the image border counts as an edge too.
    dist = cv2.distanceTransform(np.pad(mask, 1).astype(np.uint8), cv2.DIST_L2, 5)[1:-1, 1:-1]
    paths = []
    # The first ring sits half a spacing inside the edge, so the pen line ends at the edge.
    level = spacing_px / 2
    while (inner := dist > level).any():
        paths += outlines(inner, epsilon_px)
        level += spacing_px
    return paths


def centerlines(mask: np.ndarray, epsilon_px: float) -> list[Polyline]:
    """Single strokes down the middle of each line - best for line art."""
    paths = []
    for path in _trace_skeleton(_thin(mask)):
        pts = np.asarray(path, dtype=np.float32)[:, ::-1]  # (y, x) -> (x, y)
        closed = len(path) > 2 and path[0] == path[-1]
        if epsilon_px > 0 and len(pts) > 2:
            # approxPolyDP mangles open chains whose ends coincide, so treat loops as polygons.
            src = pts[:-1] if closed else pts
            pts = cv2.approxPolyDP(src.reshape(-1, 1, 2), epsilon_px, closed=closed).reshape(-1, 2)
            if closed:
                pts = np.vstack([pts, pts[:1]])
        paths.append(pts.astype(float) + 0.5)
    return paths


def hatch(mask: np.ndarray, spacing_px: float, angle_deg: float) -> list[Polyline]:
    """Parallel fill lines clipped to the inked regions."""
    h, w = mask.shape
    # Negate so the angle is counter-clockwise once Y is flipped to machine space.
    a = math.radians(-angle_deg)
    d = np.array([math.cos(a), math.sin(a)])  # along each line
    n = np.array([-d[1], d[0]])  # across lines
    corners = np.array([[0, 0], [w, 0], [0, h], [w, h]], dtype=float)
    along, across = corners @ d, corners @ n
    t = np.arange(along.min(), along.max(), 0.5)
    done: list[list[np.ndarray]] = []
    open_chains: list[list[np.ndarray]] = []  # chains that ended on the previous line
    for k, off in enumerate(np.arange(across.min() + spacing_px / 2, across.max(), spacing_px)):
        xs = off * n[0] + t * d[0]
        ys = off * n[1] + t * d[1]
        inside = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        ink = np.zeros(len(t), dtype=np.int8)
        ink[inside] = mask[ys[inside].astype(int), xs[inside].astype(int)]
        edges = np.diff(np.concatenate([[0], ink, [0]]))
        extended = []
        for s, e in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1):
            seg = [np.array([xs[s], ys[s]]), np.array([xs[e], ys[e]])]
            if k % 2:
                seg.reverse()
            # Zig-zag onto this segment from a chain on the previous line if the
            # connecting move is short and stays on ink, saving a pen lift.
            i = next((i for i, c in enumerate(open_chains) if _ink_bridge(mask, c[-1], seg[0], 2 * spacing_px)), None)
            if i is not None:
                chain = open_chains.pop(i) + seg
            else:
                chain = seg
            extended.append(chain)
        done += open_chains
        open_chains = extended
    return [np.array(c) for c in done + open_chains]


def _ink_bridge(mask: np.ndarray, a: np.ndarray, b: np.ndarray, max_len: float) -> bool:
    length = float(np.hypot(*(b - a)))
    if length > max_len:
        return False
    for f in np.linspace(0, 1, max(int(length * 2), 2)):
        x, y = (a + (b - a) * f).astype(int)
        if not (0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x]):
            return False
    return True


def _thin(mask: np.ndarray) -> np.ndarray:
    """Zhang-Suen thinning: erode strokes to a 1-pixel-wide skeleton."""
    img = np.pad(mask.astype(np.uint8), 1)
    changed = True
    while changed:
        changed = False
        for step in (0, 1):
            c = img[1:-1, 1:-1]
            p2, p3, p4 = img[:-2, 1:-1], img[:-2, 2:], img[1:-1, 2:]
            p5, p6, p7 = img[2:, 2:], img[2:, 1:-1], img[2:, :-2]
            p8, p9 = img[1:-1, :-2], img[:-2, :-2]
            ring = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
            neighbours = sum(ring[:8])
            transitions = sum((ring[i] == 0) & (ring[i + 1] == 1) for i in range(8))
            if step == 0:
                side = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                side = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            remove = (c == 1) & (neighbours >= 2) & (neighbours <= 6) & (transitions == 1) & side
            if remove.any():
                c[remove] = 0
                changed = True
    return img[1:-1, 1:-1].astype(bool)


# 4-neighbours first so walks prefer straight steps over diagonal shortcuts.
_OFFSETS = [(-1, 0), (0, 1), (1, 0), (0, -1), (-1, 1), (1, 1), (1, -1), (-1, -1)]


def _trace_skeleton(skel: np.ndarray) -> list[list[tuple[int, int]]]:
    """Walk a skeleton into pixel chains, starting from line ends, then junctions, then loops."""
    pixels = set(zip(*map(np.ndarray.tolist, np.nonzero(skel))))

    def neighbours(p):
        y, x = p
        return [q for q in ((y + dy, x + dx) for dy, dx in _OFFSETS) if q in pixels]

    degree = {p: len(neighbours(p)) for p in pixels}
    ordered = sorted(pixels)
    starts = (
        [p for p in ordered if degree[p] == 1]
        + [p for p in ordered if degree[p] >= 3]
        + ordered
    )
    visited: set[tuple[int, int]] = set()

    def walk(p):
        chain = [p]
        visited.add(p)
        while (q := next((q for q in neighbours(chain[-1]) if q not in visited), None)) is not None:
            visited.add(q)
            chain.append(q)
        return chain

    chains = []
    for p in starts:
        while True:
            if p not in visited:
                # Begin at an already-drawn neighbour so branches stay connected.
                anchor = next((q for q in neighbours(p) if q in visited), None)
                chain = ([anchor] if anchor else []) + walk(p)
            else:
                q = next((q for q in neighbours(p) if q not in visited), None)
                if q is None:
                    break
                chain = [p] + walk(q)
            if len(chain) > 8 and chain[0] in neighbours(chain[-1]):
                chain.append(chain[0])  # close loops
            if len(chain) >= 2:
                chains.append(chain)
    return chains
