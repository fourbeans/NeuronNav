"""Strict no-path semantics for the SNN wavefront without reference planners."""

import math
import time

import cv2
import numpy as np

from poni.fmm_planner import FMMPlanner
from poni.planner_adapter_neuron import (
    NoPathNeuron,
    PlanNeuron,
    add_boundary,
    mp3d_plan,
    validate_path,
)
from poni.wavefront_neuron import propagate


def _nearest_reachable_proxy(goals, base_free, padded_start, delay_quantum):
    """Project a disconnected goal onto the start-reachable free component.

    Both component discovery and projection are expressed as first-spike waves.
    The projection wave deliberately ignores obstacles: its only purpose is to
    rank safe, reachable cells by geometric proximity to the requested goal.
    """
    component_wave = propagate(
        base_free, [padded_start], delay_quantum=delay_quantum,
    )
    reachable = np.isfinite(component_wave.times) & base_free

    # The padded border is an implementation detail of the upstream planner;
    # it must never become a navigation waypoint outside the real map.
    candidates = reachable.copy()
    candidates[[0, -1], :] = False
    candidates[:, [0, -1]] = False
    if not candidates.any():
        raise NoPathNeuron("empty_start_component")

    # Prefer a waypoint at least one upstream short-term step away.  This
    # prevents a proxy next to the agent from being decoded as STOP.  Tiny
    # components fall back to any non-start cell, then finally the start.
    far_candidates = candidates & (component_wave.times >= 5.0)
    if far_candidates.any():
        candidates = far_candidates
    else:
        candidates[padded_start] = False
        if not candidates.any():
            candidates[padded_start] = True

    # Stop as soon as the first candidate spikes.  Positive unit delays and
    # heap ordering make this exactly the same argmin (including tie-breaking)
    # as filling the whole projection field, while avoiding irrelevant events.
    projection_wave = propagate(
        np.ones(goals.shape, dtype=bool),
        np.argwhere(goals),
        targets=candidates,
        reverse=True,
        delay_quantum=delay_quantum,
    )
    if projection_wave.reached < 0:
        raise NoPathNeuron("reachable_projection_failed")
    proxy = divmod(projection_wave.reached, goals.shape[1])
    return proxy, component_wave, projection_wave


def gibson_plan_strict(grid, start, goal, collision, visited, obstacle_kernel, goal_kernel,
                       delay_quantum=0.0):
    tick = time.perf_counter()
    traversible = 1.0 - cv2.dilate(np.asarray(grid, dtype=np.float32), obstacle_kernel)
    traversible[collision == 1] = 0
    traversible[visited == 1] = 1
    r, c = map(int, start)
    traversible[r - 1:r + 2, c - 1:c + 2] = 1
    traversible = add_boundary(traversible)
    goals = cv2.dilate(add_boundary(np.asarray(goal, dtype=np.float32), 0), goal_kernel) == 1
    if not goals.any():
        raise NoPathNeuron("empty_goal_mask")
    free = (traversible != 0) | goals
    padded_start = (r + 1, c + 1)
    wave = propagate(free, np.argwhere(goals), reverse=True, delay_quantum=delay_quantum)
    reachable = bool(np.isfinite(wave.times[padded_start]))
    if reachable:
        # Preserve the previously validated strict-SNN path byte-for-byte for
        # normal queries; repair work is paid only for disconnected goals.
        finite = wave.times[np.isfinite(wave.times)]
        fill = float(finite.max() + 1.0)
        decoder = FMMPlanner(traversible)
        decoder.fmm_dist = np.where(np.isfinite(wave.times), wave.times, fill)
        sx, sy, replan, stop = decoder.get_short_term_goal(padded_start)
        path_padded = wave.path_to(padded_start)[::-1]
        if not validate_path(path_padded, free, goals, padded_start):
            raise NoPathNeuron("invalid_predecessor_path")
        path = [(x - 1, y - 1) for x, y in path_padded]
        return PlanNeuron(
            (float(sx - 1), float(sy - 1)),
            bool(stop),
            bool(replan),
            path,
            {
                "spikes": wave.spikes,
                "events_emitted": wave.emitted,
                "arrival_cost": float(wave.times[padded_start]),
                "elapsed_ms": (time.perf_counter() - tick) * 1000,
                "distance_model": "eight_neighbor_not_FMM_eikonal",
                "no_path": False,
                "no_path_detected": False,
                "no_path_repaired": False,
                "no_path_policy": "none",
                "projection_used": False,
            },
        )

    # A constant-filled unreachable field has no useful local gradient.  Build
    # a temporary SNN waypoint on the safe start component instead.  This is a
    # local-planning repair, not a claim that the semantic task goal was found.
    base_free = traversible != 0
    proxy, component_wave, projection_wave = _nearest_reachable_proxy(
        goals, base_free, padded_start, delay_quantum,
    )
    repair_wave = propagate(
        base_free, [proxy], reverse=True, delay_quantum=delay_quantum,
    )
    if not np.isfinite(repair_wave.times[padded_start]):
        raise NoPathNeuron("projected_goal_disconnected")
    path_padded = repair_wave.path_to(padded_start)[::-1]
    proxy_goals = np.zeros(goals.shape, dtype=bool)
    proxy_goals[proxy] = True
    if not validate_path(path_padded, base_free, proxy_goals, padded_start):
        raise NoPathNeuron("invalid_projected_predecessor_path")

    step_size = 5
    stg = path_padded[min(step_size, len(path_padded)) - 1]
    path = [(x - 1, y - 1) for x, y in path_padded]
    total_spikes = wave.spikes + component_wave.spikes + projection_wave.spikes + repair_wave.spikes
    total_events = wave.emitted + component_wave.emitted + projection_wave.emitted + repair_wave.emitted
    return PlanNeuron(
        (float(stg[0] - 1), float(stg[1] - 1)),
        False,
        len(path_padded) <= 1,
        path,
        {
            "spikes": total_spikes,
            "events_emitted": total_events,
            "arrival_cost": float(repair_wave.times[padded_start]),
            "elapsed_ms": (time.perf_counter() - tick) * 1000,
            "distance_model": "eight_neighbor_not_FMM_eikonal",
            "no_path": False,
            "no_path_detected": True,
            "no_path_repaired": True,
            "no_path_reason": "disconnected_start",
            "no_path_policy": "reachable_component_goal_projection_snn",
            "projection_used": True,
            "projection_distance_cells": float(projection_wave.times[proxy]),
            "proxy_goal": [int(proxy[0] - 1), int(proxy[1] - 1)],
            "component_spikes": component_wave.spikes,
            "projection_spikes": projection_wave.spikes,
            "repair_spikes": repair_wave.spikes,
        },
    )


def mp3d_plan_strict(grid, start, goal, collision, visited, obstacle_kernel, goal_kernel,
                     map_resolution=5, wscale=4.0, niters=1, downsampling=1,
                     delay_quantum=0.0):
    try:
        return mp3d_plan(
            grid, start, goal, collision, visited, obstacle_kernel, goal_kernel,
            map_resolution=map_resolution, wscale=wscale, niters=niters,
            downsampling=downsampling, delay_quantum=delay_quantum,
        )
    except NoPathNeuron as exc:
        reason = str(exc)
        if reason not in ("disconnected_goal", "empty_or_blocked_goal_mask"):
            raise
        if downsampling != 1:
            raise
        step_size = 5
        random_theta = np.random.uniform(-np.pi, np.pi, (1,))[0].item()
        stg_x = int(step_size * np.cos(random_theta)) - 1
        stg_y = int(step_size * np.sin(random_theta)) - 1
        return PlanNeuron(
            (float(stg_x), float(stg_y)),
            False,
            True,
            [],
            {
                "spikes": 0,
                "events_emitted": 0,
                "arrival_cost": None,
                "elapsed_ms": 0.0,
                "distance_model": "astar_destination_proximity_weight",
                "no_path": True,
                "no_path_reason": reason,
                "no_path_policy": "upstream_random_replan_without_astar",
            },
        )
