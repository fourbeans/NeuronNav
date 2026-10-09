"""Bounded, optional cost-consistent MP3D readout of an existing SNN path.

The first five-cell waypoint from the old strict SNN remains the anchor.  Only
an unobstructed *later* predecessor-path vertex may replace it.  The search is
limited by path arc length, not a repeated temporal SNN rollout.
"""

import math

import numpy as np


def line_cells(start, target):
    """Bresenham cells with the same eight-neighbor corner convention as SNN."""
    r0, c0 = map(int, start)
    r1, c1 = map(int, target)
    x, y = c0, r0
    dx, dy = abs(c1 - c0), -abs(r1 - r0)
    sx, sy = (1 if c0 < c1 else -1), (1 if r0 < r1 else -1)
    error = dx + dy
    cells = []
    while True:
        cells.append((y, x))
        if x == c1 and y == r1:
            return cells
        twice = 2 * error
        if twice >= dy:
            error += dy
            x += sx
        if twice <= dx:
            error += dx
            y += sy


def _weighted_cost(cells, weights):
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) * float(weights[b])
               for a, b in zip(cells[:-1], cells[1:]))


def select_bounded_visible(path, free, weights, resolution_cm, cap_cm,
                           cost_guard=False, tolerance=0.02,
                           max_direction_delta_deg=None):
    """Return (STG, path_index, arc_cm, ray/path cost ratio, changed).

    The cost guard compares the weighted Bresenham ray to the original SNN
    path prefix ending at the same vertex.  With the old fifth-cell anchor,
    a failed visibility/cost test never changes the previous SNN waypoint.
    """
    points = [tuple(map(int, p)) for p in path]
    if not points:
        raise ValueError("path cannot be empty")
    free = np.asarray(free, dtype=bool)
    weights = np.asarray(weights)
    if free.shape != weights.shape or free.ndim != 2:
        raise ValueError("free and weights must be equal-sized 2-D arrays")
    if not math.isfinite(resolution_cm) or resolution_cm <= 0:
        raise ValueError("resolution_cm must be positive")
    if not math.isfinite(cap_cm) or cap_cm < 30:
        raise ValueError("cap_cm must be at least 30 cm")
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be nonnegative")
    if max_direction_delta_deg is not None and (not math.isfinite(max_direction_delta_deg)
                                                 or not 0 <= max_direction_delta_deg <= 180):
        raise ValueError("max_direction_delta_deg must be in [0,180]")
    anchor = min(4, len(points) - 1)
    arcs = [0.0]
    path_costs = [0.0]
    for a, b in zip(points[:-1], points[1:]):
        edge = math.hypot(b[0] - a[0], b[1] - a[1])
        arcs.append(arcs[-1] + edge * resolution_cm)
        path_costs.append(path_costs[-1] + edge * float(weights[b]))
        if len(arcs) - 1 >= anchor and arcs[-1] > cap_cm:
            break
    if anchor == len(points) - 1:
        return points[anchor], anchor, arcs[anchor], None, False

    max_index = anchor
    for idx in range(anchor + 1, len(arcs)):
        if arcs[idx] > cap_cm + 1e-9:
            break
        max_index = idx
    for idx in range(max_index, anchor, -1):
        if max_direction_delta_deg is not None:
            old_vec = np.subtract(points[anchor], points[0]).astype(float)
            new_vec = np.subtract(points[idx], points[0]).astype(float)
            similarity = float(np.dot(old_vec, new_vec) /
                               max(np.linalg.norm(old_vec) * np.linalg.norm(new_vec), 1e-12))
            delta = math.degrees(math.acos(max(-1.0, min(1.0, similarity))))
            if delta > max_direction_delta_deg + 1e-9:
                continue
        ray = line_cells(points[0], points[idx])
        if not all(0 <= r < free.shape[0] and 0 <= c < free.shape[1] and free[r, c]
                   for r, c in ray):
            continue
        ray_cost = _weighted_cost(ray, weights)
        ratio = ray_cost / max(path_costs[idx], 1e-12)
        if cost_guard and ratio > 1.0 + tolerance:
            continue
        return points[idx], idx, arcs[idx], ratio, True
    return points[anchor], anchor, arcs[anchor], None, False
