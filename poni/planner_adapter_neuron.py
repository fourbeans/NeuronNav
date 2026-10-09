"""Dataset-specific decoding around the shared spiking wavefront core."""
from dataclasses import dataclass, field
import math
import time

import cv2
import numpy as np

from poni.fmm_planner import FMMPlanner
from poni.wavefront_neuron import propagate


class NoPathNeuron(RuntimeError):
    pass


@dataclass
class PlanNeuron:
    stg: tuple
    stop: bool
    replan: bool = False
    path: list = field(default_factory=list)
    info: dict = field(default_factory=dict)


def add_boundary(mat, value=1):
    return np.pad(mat, 1, constant_values=value)


def proximity_weights(obstacles, wscale=4.0, niters=1):
    """Match astar.cpp weights, including eight neighbors (center excluded)."""
    if not np.isfinite(wscale) or wscale < 1 or niters < 1:
        raise ValueError("Require wscale >= 1 and niters >= 1")
    obstacles = np.asarray(obstacles, dtype=bool)
    old = np.where(obstacles, wscale ** (niters + 1), 1.0).astype(np.float32)
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0
    for _ in range(niters):
        surrounding = cv2.dilate(old, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
        old = np.where(obstacles, old, np.maximum(surrounding / wscale, 1.0))
    return old


def path_cost(path, weights):
    return float(sum(math.hypot(b[0] - a[0], b[1] - a[1]) * weights[b]
                     for a, b in zip(path[:-1], path[1:])))


def validate_path(path, free, goals, start):
    if not path or tuple(path[0]) != tuple(start) or not goals[tuple(path[-1])]:
        return False
    h, w = free.shape
    for r, c in path:
        if not (0 <= r < h and 0 <= c < w and free[r, c]):
            return False
    # Match upstream eight-connectivity; corner-cutting changes are a separate ablation.
    return all(0 < max(abs(a[0] - b[0]), abs(a[1] - b[1])) <= 1
               for a, b in zip(path[:-1], path[1:]))


def gibson_plan(grid, start, goal, collision, visited, obstacle_kernel, goal_kernel,
                delay_quantum=0.0):
    tick = time.perf_counter()
    traversible = 1.0 - cv2.dilate(np.asarray(grid, dtype=np.float32), obstacle_kernel)
    traversible[collision == 1] = 0
    traversible[visited == 1] = 1
    r, c = map(int, start)
    # Intentionally preserve upstream slicing/boundary semantics.
    traversible[r - 1:r + 2, c - 1:c + 2] = 1
    traversible = add_boundary(traversible)
    goals = cv2.dilate(add_boundary(np.asarray(goal, dtype=np.float32), 0), goal_kernel) == 1
    if not goals.any():
        raise NoPathNeuron("empty_goal_mask")
    # Upstream skfmm masked-array assignment also unmasks goal cells.
    free = (traversible != 0) | goals
    padded_start = (r + 1, c + 1)
    wave = propagate(free, np.argwhere(goals), reverse=True, delay_quantum=delay_quantum)
    if not np.isfinite(wave.times[padded_start]):
        raise NoPathNeuron("disconnected_start")
    # Reuse only the upstream ring decoder, NOT skfmm.distance.
    decoder = FMMPlanner(traversible)
    finite = wave.times[np.isfinite(wave.times)]
    decoder.fmm_dist = np.where(np.isfinite(wave.times), wave.times, finite.max() + 1.0)
    sx, sy, replan, stop = decoder.get_short_term_goal(padded_start)
    path_padded = wave.path_to(padded_start)[::-1]
    if not validate_path(path_padded, free, goals, padded_start):
        raise NoPathNeuron("invalid_predecessor_path")
    return PlanNeuron((float(sx - 1), float(sy - 1)), bool(stop), bool(replan),
                      [(x - 1, y - 1) for x, y in path_padded],
                      {"spikes": wave.spikes, "events_emitted": wave.emitted,
                       "arrival_cost": float(wave.times[padded_start]),
                       "elapsed_ms": (time.perf_counter() - tick) * 1000,
                       "distance_model": "eight_neighbor_not_FMM_eikonal"})


def mp3d_plan(grid, start, goal, collision, visited, obstacle_kernel, goal_kernel,
              map_resolution=5, wscale=4.0, niters=1, downsampling=1, delay_quantum=0.0):
    tick = time.perf_counter()
    if downsampling != 1:
        raise NoPathNeuron("nonunit_downsampling_not_validated")
    traversible = 1.0 - cv2.dilate(np.asarray(grid, dtype=np.float32), obstacle_kernel)
    traversible[collision == 1] = 0
    traversible[visited == 1] = 1
    r, c = map(int, start)
    traversible[r - 1:r + 2, c - 1:c + 2] = 1
    traversible = add_boundary(traversible)
    goals = cv2.dilate(add_boundary(np.asarray(goal, dtype=np.float32), 0), goal_kernel) == 1
    free = traversible != 0
    if not (goals & free).any():
        raise NoPathNeuron("empty_or_blocked_goal_mask")
    padded_start = (r + 1, c + 1)
    weights = proximity_weights(~free, wscale, niters)
    wave = propagate(free, [padded_start], weights=weights, targets=goals,
                     delay_quantum=delay_quantum)
    if wave.reached < 0:
        raise NoPathNeuron("disconnected_goal")
    target = divmod(wave.reached, free.shape[1])
    path = wave.path_to(target)
    if not validate_path(path, free, goals, padded_start):
        raise NoPathNeuron("invalid_predecessor_path")
    step_size = 5
    stg = path[min(step_size, len(path)) - 1]
    geometric_length = sum(math.hypot(b[0] - a[0], b[1] - a[1])
                           for a, b in zip(path[:-1], path[1:])) * map_resolution / 100.0
    stop = len(path) < step_size and geometric_length <= 0.25
    return PlanNeuron((stg[0] - 1, stg[1] - 1), bool(stop), False,
                      [(x - 1, y - 1) for x, y in path],
                      {"spikes": wave.spikes, "events_emitted": wave.emitted,
                       "arrival_cost": float(wave.times[target]),
                       "path_cost": path_cost(path, weights),
                       "elapsed_ms": (time.perf_counter() - tick) * 1000,
                       "distance_model": "astar_destination_proximity_weight"})


def turn_action(stg, start, heading, turn_angle):
    angle_goal = math.degrees(math.atan2(stg[0] - start[0], stg[1] - start[1]))
    angle_agent = heading % 360.0
    if angle_agent > 180:
        angle_agent -= 360
    relative = (angle_agent - angle_goal) % 360.0
    if relative > 180:
        relative -= 360
    return 3 if relative > turn_angle / 2.0 else 2 if relative < -turn_angle / 2.0 else 1
