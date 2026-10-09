"""Local, cost-preserving backtrace among equal-arrival SNN predecessors.

This does not rerun the wavefront.  It only reads the first-arrival time and
traversal weight of at most eight neighboring cells at each backtrace step.
The old predecessor wins every tie in smoothness, limiting perturbations.
"""

import math

import numpy as np


def _turn_score(parent, current, following):
    incoming = np.subtract(current, parent)
    outgoing = np.subtract(following, current)
    return -float(np.dot(incoming, outgoing) /
                  (np.linalg.norm(incoming) * np.linalg.norm(outgoing)))


def decode_straight_ties(wave, weights, start, target, atol=1e-8, rtol=1e-10):
    """Return (start-to-target path, changed-parent count) at original cost.

    `wave` is a forward first-arrival result.  The target's old parent is held
    fixed, then each earlier parent is selected from neighbors satisfying the
    exact Bellman arrival equation, with minimal turn into the known suffix.
    """
    times = np.asarray(wave.times)
    weights = np.asarray(weights)
    if times.ndim != 2 or times.shape != weights.shape:
        raise ValueError("time and weight maps must have equal 2-D shapes")
    if atol < 0 or rtol < 0:
        raise ValueError("arrival tolerances must be nonnegative")
    h, w = times.shape
    start, target = tuple(map(int, start)), tuple(map(int, target))
    if not np.isfinite(times[target]):
        return [], 0
    path = [target]
    following = None
    current = target
    changed = 0
    for _ in range(h * w + 1):
        if current == start:
            return path[::-1], changed
        cr, cc = current
        current_index = cr * w + cc
        old_index = int(wave.parents.flat[current_index])
        if old_index < 0:
            raise ValueError("predecessor chain ended before the seed")
        old = divmod(old_index, w)
        candidates = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                pr, pc = cr + dr, cc + dc
                if not (0 <= pr < h and 0 <= pc < w and np.isfinite(times[pr, pc])):
                    continue
                parent = (pr, pc)
                arrival = times[parent] + math.hypot(dr, dc) * float(weights[current])
                if abs(arrival - times[current]) <= atol + rtol * abs(times[current]):
                    candidates.append(parent)
        if old not in candidates:
            raise ValueError("old predecessor fails the first-arrival equation")
        if following is None:
            chosen = old
        else:
            chosen = min(candidates, key=lambda parent: (
                _turn_score(parent, current, following),
                0 if parent == old else 1,
                parent[0], parent[1],
            ))
        changed += chosen != old
        following, current = current, chosen
        path.append(current)
    raise RuntimeError("equal-arrival backtrace did not reach seed")


def count_bends(path):
    """Count direction changes in an eight-connected path."""
    directions = [tuple(np.subtract(b, a)) for a, b in zip(path[:-1], path[1:])]
    return sum(a != b for a, b in zip(directions[:-1], directions[1:]))
