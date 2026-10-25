"""Reorder strokes so the pen spends less time travelling with the pen up."""

from __future__ import annotations

import numpy as np


def order_paths(paths: list[np.ndarray], start: tuple[float, float]) -> list[np.ndarray]:
    """Greedy nearest-neighbour ordering; strokes may be reversed, loops re-started anywhere."""
    if not paths:
        return []
    heads = np.array([p[0] for p in paths])
    tails = np.array([p[-1] for p in paths])
    remaining = np.ones(len(paths), dtype=bool)
    pos = np.asarray(start, dtype=float)
    ordered = []
    for _ in range(len(paths)):
        d_head = np.where(remaining, np.hypot(*(heads - pos).T), np.inf)
        d_tail = np.where(remaining, np.hypot(*(tails - pos).T), np.inf)
        i, j = int(d_head.argmin()), int(d_tail.argmin())
        if d_head[i] <= d_tail[j]:
            path = paths[i]
        else:
            i, path = j, paths[j][::-1]
        remaining[i] = False
        if len(path) > 3 and np.allclose(path[0], path[-1]):
            # Closed loop: begin at whichever vertex is nearest.
            loop = path[:-1]
            k = int(np.hypot(*(loop - pos).T).argmin())
            path = np.vstack([loop[k:], loop[:k], loop[k : k + 1]])
        ordered.append(path)
        pos = path[-1]
    return ordered
