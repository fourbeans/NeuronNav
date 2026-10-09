"""Deterministic, bounded candidates. No GT and no teacher dependency."""
from dataclasses import dataclass
import numpy as np
from scipy.ndimage import maximum_filter, binary_dilation


@dataclass
class Candidates:
    rc: np.ndarray
    spatial: np.ndarray
    valid: np.ndarray
    source: np.ndarray


def build_candidates(score, allowed, explored, history=(), resolution=0.05):
    if score.ndim != 2 or not np.isfinite(score).all():
        raise ValueError("Expected a finite postprocessed 2D score map")
    h, w = score.shape
    anchor = tuple(np.unravel_index(score.argmax(), score.shape))
    points, sources, seen = [anchor], [0], {anchor}

    def add(items, quota, source):
        added = 0
        for r, c in items:
            p = int(r), int(c)
            if 0 <= p[0] < h and 0 <= p[1] < w and allowed[p] and p not in seen:
                points.append(p)
                sources.append(source)
                seen.add(p)
                added += 1
                if added >= quota:
                    break

    def ordered(mask):
        flat = np.flatnonzero(mask)
        order = np.lexsort((flat, -score.flat[flat]))
        return [(int(k // w), int(k % w)) for k in flat[order]]

    radius = max(1, round(0.5 / resolution))
    peaks = (score == maximum_filter(score, size=2 * radius + 1)) & allowed
    # Greedy NMS also separates a flat plateau deterministically.
    def separated():
        accepted = []
        for p in ordered(peaks):
            if all(
                (p[0] - r) ** 2 + (p[1] - c) ** 2 >= radius ** 2
                for r, c in [anchor] + accepted
            ):
                accepted.append(p)
                yield p
                if len(accepted) == 31:
                    return

    add(separated(), 31, 1)
    frontier = (explored < 0.5) & binary_dilation(explored >= 0.5) & allowed
    frontier_points = ordered(frontier)
    if frontier_points:
        indices = np.linspace(
            0, len(frontier_points) - 1, min(48, len(frontier_points)), dtype=int
        )
        add((frontier_points[i] for i in indices), 48, 2)
    region_points = []
    for rs in np.array_split(np.arange(h), 4):
        for cs in np.array_split(np.arange(w), 8):
            if not len(rs) or not len(cs):
                continue
            region = np.zeros_like(allowed)
            region[np.ix_(rs, cs)] = allowed[np.ix_(rs, cs)]
            ranked = ordered(region)
            if ranked:
                region_points.append(ranked[0])
    add(region_points, 32, 3)
    add(history, 16, 4)
    rc = np.zeros((128, 2), dtype=np.int32)
    spatial = np.zeros(128, dtype=np.float32)
    valid = np.zeros(128, dtype=bool)
    source = np.full(128, -1, dtype=np.int8)
    n = len(points)
    rc[:n] = points
    valid[:n] = True
    source[:n] = sources
    spatial[:n] = score[rc[:n, 0], rc[:n, 1]]
    return Candidates(rc, spatial, valid, source)


def encode_action(rc, shape, anchor_action=None, index=0):
    if index == 0 and anchor_action is not None:
        return np.asarray(anchor_action, dtype=np.float32)
    # Cell-center encoding avoids float32 division truncating to a neighbour.
    return (np.asarray(rc, dtype=np.float32) + 0.25) / np.asarray(
        shape, dtype=np.float32
    )
