"""Small, deterministic readouts for an existing SNN-wave predecessor path.

The wave arrival times, predecessor chain, and STOP decision are not changed.
Coordinates use (row, column); a visible ray visits Bresenham cells, matching
the existing eight-neighbor convention (including diagonal corner cuts).
"""

import math

import numpy as np

try:
    from numba import njit
except ImportError:
    def njit(**kwargs):
        return lambda fn: fn


@njit(cache=True)
def _ray_free(free, r0, c0, r1, c1):
    x, y = c0, r0
    dx, dy = abs(c1 - c0), -abs(r1 - r0)
    sx = 1 if c0 < c1 else -1
    sy = 1 if r0 < r1 else -1
    error = dx + dy
    while True:
        if y < 0 or y >= free.shape[0] or x < 0 or x >= free.shape[1] or not free[y, x]:
            return False
        if x == c1 and y == r1:
            return True
        doubled = 2 * error
        if doubled >= dy:
            error += dy
            x += sx
        if doubled <= dx:
            error += dx
            y += sy


@njit(cache=True)
def _farthest_visible(free, path):
    r0, c0 = path[0, 0], path[0, 1]
    for idx in range(path.shape[0] - 1, 0, -1):
        if _ray_free(free, r0, c0, path[idx, 0], path[idx, 1]):
            return idx
    return 0


def select_stg(path, mode, map_resolution_cm, lookahead_cm=25.0, free=None):
    """Return (row,col), index in path, and traveled path length in cm.

    `length`: last predecessor vertex within a metric lookahead.
    `visible`: farthest predecessor vertex on a free Bresenham ray.
    """
    if mode not in ("length", "visible"):
        raise ValueError("readout mode must be length or visible")
    if not math.isfinite(map_resolution_cm) or map_resolution_cm <= 0:
        raise ValueError("map resolution must be positive")
    if not math.isfinite(lookahead_cm) or lookahead_cm <= 0:
        raise ValueError("lookahead must be positive")
    points = np.asarray(path, dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) == 0:
        raise ValueError("path must be a nonempty list of (row,column) cells")
    if mode == "length":
        idx, cumulative = 0, 0.0
        for k in range(1, len(points)):
            step = math.hypot(*(points[k] - points[k - 1])) * map_resolution_cm
            if cumulative + step > lookahead_cm + 1e-9:
                break
            cumulative += step
            idx = k
        # All supported maps use 5 cm cells and at least 20 cm lookahead.
        if idx == 0 and len(points) > 1:
            idx, cumulative = 1, math.hypot(*(points[1] - points[0])) * map_resolution_cm
    else:
        if free is None:
            raise ValueError("visible readout requires a free-cell map")
        free = np.ascontiguousarray(free, dtype=np.bool_)
        if free.ndim != 2:
            raise ValueError("free-cell map must be 2-D")
        idx = int(_farthest_visible(free, points))
        # If there is no nontrivial ray, retain the first SNN predecessor.
        if idx == 0 and len(points) > 1:
            idx = 1
        cumulative = sum(
            math.hypot(*(points[k] - points[k - 1])) * map_resolution_cm
            for k in range(1, idx + 1)
        )
    return (float(points[idx, 0]), float(points[idx, 1])), idx, cumulative


def ray_free(free, start, target):
    """Public diagnostic equivalent of the ray used in visible readout."""
    arr = np.ascontiguousarray(free, dtype=np.bool_)
    return bool(_ray_free(arr, int(start[0]), int(start[1]), int(target[0]), int(target[1])))


def warmup():
    arr = np.ones((3, 3), dtype=np.bool_)
    path = [(0, 0), (1, 1), (2, 2)]
    select_stg(path, "visible", 5, free=arr)
