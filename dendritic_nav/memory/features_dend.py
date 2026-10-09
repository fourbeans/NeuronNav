"""Schema v1: 4x32 branch features; 24 candidate features, no privileged data."""
import numpy as np
from scipy.ndimage import binary_dilation
from .history_dend import rc_to_xy


class FeatureBuilder:
    def __init__(self):
        self.previous = None
        self.origin = None

    def build(self, maps, goal_category, location, origin, resolution, history, motion):
        # maps = [obstacle, explored, current, visited, global channels..., semantic].
        obstacle, explored, semantic = maps[0], maps[1], maps[8:-1]
        target = semantic[goal_category]
        frame_changed = self.origin is not None and not np.array_equal(
            self.origin, origin
        )
        previous = self.previous
        if previous is None or frame_changed:
            previous = (semantic.copy(), explored.copy(), 0.0)
        ds = semantic - previous[0]
        frontier = (
            (explored < 0.5) & binary_dilation(explored >= 0.5) & (obstacle < 0.5)
        )
        f = np.zeros((4, 32), dtype=np.float32)
        n = min(21, len(semantic))
        # Scale small occupancy fractions to useful signed integer-event range.
        f[0, :n] = ds[:n].mean((1, 2)) * 100
        f[0, 21:28] = [
            np.maximum(ds[goal_category], 0).mean() * 100,
            np.maximum(-ds[goal_category], 0).mean() * 100,
            target.mean() * 10,
            float(target.max()),
            float((target > 0).any()),
            goal_category / 20,
            np.abs(ds).mean() * 100,
        ]
        xy = rc_to_xy(location, origin, resolution)
        delta_explored = explored - previous[1]
        current_stats = history.lookup(xy)
        f[1, :10] = [
            np.maximum(delta_explored, 0).mean() * 100,
            explored.mean(),
            frontier.mean() * 100,
            (frontier.mean() - previous[2]) * 100,
            np.log1p(current_stats[0]) / 5,
            np.log1p(current_stats[2]) / 5,
            obstacle.mean(),
            history.steps / 500,
            float(frame_changed),
            min(history.entry_count(xy), 7),
        ]
        dx, dy, turn = motion.get("delta", (0.0, 0.0, 0.0))
        action = int(motion.get("action", -1))
        f[2, :5] = [
            dx,
            dy,
            turn / np.pi,
            np.hypot(dx, dy),
            float(motion.get("collision", False)),
        ]
        if 0 <= action < 6:
            f[2, 5 + action] = 1
        distance = 0 if history.goal is None else np.linalg.norm(xy - history.goal)
        f[2, 11:14] = [
            distance / 20,
            min(history.stall, 100) / 100,
            0
            if not np.isfinite(history.best_distance)
            else (history.best_distance - distance),
        ]
        f[3, :5] = [
            history.age / 500,
            history.switches / 100,
            float(history.goal is not None),
            np.log1p(current_stats[3]) / 5,
            current_stats[5],
        ]
        if history.goal is not None:
            f[3, 5:7] = (history.goal - xy) / 20
            f[3, 7:9] = history.lookup(history.goal)[3:5] / 10
        self.previous = (semantic.copy(), explored.copy(), frontier.mean())
        self.origin = np.asarray(origin).copy()
        return (
            np.clip(f, -8, 8),
            frontier,
            bool(np.maximum(ds[goal_category], 0).sum() > 0),
        )


def candidate_features(
    candidates, maps, location, origin, resolution, history, frontier, category
):
    f = np.zeros((128, 24), dtype=np.float32)
    h, w = maps.shape[-2:]
    for i in np.flatnonzero(candidates.valid):
        r, c = candidates.rc[i]
        xy = rc_to_xy((r, c), origin, resolution)
        stats = history.lookup(xy)
        delta = (np.array([r, c]) - location) * resolution
        f[i, :11] = [
            candidates.spatial[i],
            r / h,
            c / w,
            delta[0] / 20,
            delta[1] / 20,
            np.linalg.norm(delta) / 20,
            maps[0, r, c],
            maps[1, r, c],
            float(frontier[r, c]),
            maps[8 + category, r, c],
            float(i == 0),
        ]
        f[i, 11:17] = [
            np.log1p(stats[0]) / 5,
            (history.steps - stats[1]) / 500 if stats[0] else 1,
            np.log1p(stats[2]) / 5,
            np.log1p(stats[3]) / 5,
            np.log1p(stats[4]) / 5,
            stats[5],
        ]
        f[i, 17 + int(candidates.source[i])] = 1
        if history.goal is not None:
            f[i, 22] = np.linalg.norm(xy - history.goal) / 20
            f[i, 23] = float(history.key(xy) == history.key(history.goal))
    return np.clip(f, -8, 8)
