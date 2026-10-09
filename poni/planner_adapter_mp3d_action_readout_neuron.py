"""Independent action-aware readouts for a strict weighted SNN wavefront."""

import math
import time

import cv2
import numpy as np

from poni.path_tiebreak_neuron import count_bends, decode_straight_ties
from poni.planner_adapter_neuron import (
    PlanNeuron, add_boundary, path_cost, proximity_weights, turn_action,
    validate_path,
)
from poni.planner_adapter_strict_neuron import mp3d_plan_strict
from poni.wavefront_neuron import propagate


def mp3d_plan_action_readout(grid, start, goal, collision, visited,
                             obstacle_kernel, goal_kernel, map_resolution=5,
                             wscale=4.0, niters=1, downsampling=1,
                             delay_quantum=0.0, heading=None, turn_angle=None,
                             preserve_forward=False, forward_gain_only=False):
    """Select an equal-cost predecessor path using one action-aware rule."""
    if preserve_forward == forward_gain_only:
        raise ValueError("select exactly one action-aware readout")
    if heading is None or turn_angle is None:
        raise ValueError("action-aware readout requires heading and turn_angle")
    tick = time.perf_counter()
    if downsampling != 1:
        return mp3d_plan_strict(
            grid, start, goal, collision, visited, obstacle_kernel, goal_kernel,
            map_resolution=map_resolution, wscale=wscale, niters=niters,
            downsampling=downsampling, delay_quantum=delay_quantum,
        )
    traversible = 1.0 - cv2.dilate(np.asarray(grid, dtype=np.float32), obstacle_kernel)
    traversible[collision == 1] = 0
    traversible[visited == 1] = 1
    r, c = map(int, start)
    traversible[r - 1:r + 2, c - 1:c + 2] = 1
    free = add_boundary(traversible) != 0
    goals = cv2.dilate(add_boundary(np.asarray(goal, dtype=np.float32), 0), goal_kernel) == 1
    padded_start = (r + 1, c + 1)
    if not (goals & free).any():
        return mp3d_plan_strict(
            grid, start, goal, collision, visited, obstacle_kernel, goal_kernel,
            map_resolution=map_resolution, wscale=wscale, niters=niters,
            downsampling=downsampling, delay_quantum=delay_quantum,
        )
    weights = proximity_weights(~free, wscale, niters)
    wave = propagate(free, [padded_start], weights=weights, targets=goals,
                     delay_quantum=delay_quantum)
    if wave.reached < 0:
        return mp3d_plan_strict(
            grid, start, goal, collision, visited, obstacle_kernel, goal_kernel,
            map_resolution=map_resolution, wscale=wscale, niters=niters,
            downsampling=downsampling, delay_quantum=delay_quantum,
        )
    target = divmod(wave.reached, free.shape[1])
    old_path = wave.path_to(target)
    if not validate_path(old_path, free, goals, padded_start):
        raise RuntimeError("invalid original SNN predecessor path")
    tie_path, changed = decode_straight_ties(wave, weights, padded_start, target)
    if not validate_path(tie_path, free, goals, padded_start):
        raise RuntimeError("invalid equal-arrival SNN predecessor path")
    old_cost, tie_cost = path_cost(old_path, weights), path_cost(tie_path, weights)
    if abs(tie_cost - old_cost) > 1e-5:
        raise RuntimeError("equal-arrival backtrace changed weighted cost")
    step_size = 5
    old_stg = old_path[min(step_size, len(old_path)) - 1]
    tie_stg = tie_path[min(step_size, len(tie_path)) - 1]
    old_action = turn_action((old_stg[0] - 1, old_stg[1] - 1), start,
                             heading, turn_angle)
    tie_action = turn_action((tie_stg[0] - 1, tie_stg[1] - 1), start,
                             heading, turn_angle)
    forward_guarded = (preserve_forward and old_action == 1
                       and tie_action in (2, 3))
    gain_accepted = (forward_gain_only and old_action in (2, 3)
                     and tie_action == 1)
    select_old = forward_guarded or (forward_gain_only and not gain_accepted)
    selected_path = old_path if select_old else tie_path
    stg = old_stg if select_old else tie_stg
    old_length = sum(math.hypot(b[0] - a[0], b[1] - a[1])
                     for a, b in zip(old_path[:-1], old_path[1:])) * map_resolution / 100.0
    stop = len(old_path) < step_size and old_length <= 0.25
    return PlanNeuron(
        (float(stg[0] - 1), float(stg[1] - 1)), bool(stop), False,
        [(x - 1, y - 1) for x, y in selected_path],
        {
            "spikes": wave.spikes,
            "events_emitted": wave.emitted,
            "arrival_cost": float(wave.times[target]),
            "path_cost": old_cost if select_old else tie_cost,
            "elapsed_ms": (time.perf_counter() - tick) * 1000,
            "distance_model": "astar_destination_proximity_weight",
            "readout_mode": ("equal_arrival_forward_gain_only" if forward_gain_only
                             else "equal_arrival_forward_guard"),
            "tie_forward_guarded": bool(forward_guarded),
            "tie_forward_gain_accepted": bool(gain_accepted),
            "tie_changed_parents": int(changed),
            "tie_old_stg": [float(old_stg[0] - 1), float(old_stg[1] - 1)],
            "tie_old_bends": count_bends(old_path),
            "tie_new_bends": count_bends(tie_path),
            "tie_cost_delta": tie_cost - old_cost,
        },
    )
