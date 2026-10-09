"""Focused readout checks; run in the PONI environment from the repo root."""

import math

import cv2
import numpy as np

from poni.planner_adapter_opt_neuron import _free_for_ray, gibson_plan_opt, mp3d_plan_opt
from poni.planner_adapter_strict_neuron import gibson_plan_strict, mp3d_plan_strict
from poni.waypoint_opt_neuron import ray_free, select_stg


def _case(kind):
    grid = np.zeros((51, 51), np.float32)
    start, target = (25, 6), (25, 44)
    if kind == "L":
        grid[5:43, 20] = 1
        grid[35:41, 20] = 0
    elif kind == "narrow":
        grid[0:21, 19:25] = 1
        grid[29:51, 19:25] = 1
    elif kind == "diagonal":
        start, target = (6, 6), (43, 43)
        grid[20:30, 12:42] = 1
    elif kind == "disconnected":
        grid[:, 20] = 1
    elif kind == "start_goal":
        target = start
    goal = np.zeros_like(grid)
    goal[target] = 1
    collision = np.zeros_like(grid)
    visited = np.zeros_like(grid)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    goal_kernel = np.ones((3, 3), np.uint8)
    return grid, start, goal, collision, visited, kernel, goal_kernel


def run():
    checks = 0
    path = [(5, 5), (5, 6), (5, 7), (6, 8), (7, 8)]
    stg, idx, length = select_stg(path, "length", 5.0, 15.0)
    assert stg == (5.0, 7.0) and idx == 2 and length == 10.0
    free = np.ones((12, 12), bool)
    stg, idx, _ = select_stg(path, "visible", 5.0, free=free)
    assert idx == len(path) - 1 and stg == (7.0, 8.0)
    free[5, 7] = False
    assert not ray_free(free, (5, 5), (5, 9))
    free = np.ones((10, 10), bool)
    free[1, 2] = False
    # A one-edge diagonal is allowed by the upstream eight-neighbor graph.
    assert ray_free(free, (1, 1), (2, 2))
    checks += 4

    # A visible readout must not shortcut through an L-corridor wall.
    free = np.ones((12, 12), bool)
    free[:7, 8] = False
    bend = [(4, 4), (5, 5), (6, 6), (7, 7), (7, 8), (7, 9), (6, 10)]
    bend_stg, bend_idx, _ = select_stg(bend, "visible", 5.0, free=free)
    assert 0 < bend_idx < len(bend) - 1 and ray_free(free, bend[0], bend_stg)
    checks += 1

    for kind in ("open", "L", "narrow", "diagonal", "disconnected", "start_goal"):
        args = _case(kind)
        old = gibson_plan_strict(*args)
        for mode in ("length", "visible"):
            new = gibson_plan_opt(*args, mode=mode)
            assert new.path == old.path and new.stop == old.stop and new.replan == old.replan
            assert new.info["spikes"] == old.info["spikes"]
            assert math.isclose(new.info["arrival_cost"], old.info["arrival_cost"])
            assert new.info["no_path_repaired"] == old.info["no_path_repaired"]
            assert len(new.path) <= 1 or tuple(new.stg) in new.path
            if mode == "visible" and len(new.path) > 1:
                free = _free_for_ray(*args, gibson=not old.info["no_path_repaired"])
                assert ray_free(free, start=args[1], target=new.stg)
            checks += 1

        if kind == "disconnected":
            continue
        for mode in ("length", "visible"):
            old = mp3d_plan_strict(*args)
            new = mp3d_plan_opt(*args, mode=mode)
            assert new.path == old.path and new.stop == old.stop and new.replan == old.replan
            assert new.info["spikes"] == old.info["spikes"]
            assert math.isclose(new.info["arrival_cost"], old.info["arrival_cost"])
            assert len(new.path) <= 1 or tuple(new.stg) in new.path
            if mode == "visible" and len(new.path) > 1:
                assert ray_free(_free_for_ray(*args, gibson=False), args[1], new.stg)
            checks += 1
    print(f"WAVEOPT_VALIDATION_OK: {checks} focused checks")


if __name__ == "__main__":
    run()
