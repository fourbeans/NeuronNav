"""Additive SNN-wave waypoint experiments for Gibson and MP3D ObjectNav."""

import cv2
import numpy as np

from poni.planner_adapter_strict_neuron import gibson_plan_strict, mp3d_plan_strict
from poni.waypoint_opt_neuron import select_stg


def _free_for_ray(grid, start, goal, collision, visited, obstacle_kernel,
                  goal_kernel, gibson):
    traversible = 1.0 - cv2.dilate(np.asarray(grid, dtype=np.float32), obstacle_kernel)
    traversible[collision == 1] = 0
    traversible[visited == 1] = 1
    r, c = map(int, start)
    traversible[r - 1:r + 2, c - 1:c + 2] = 1
    free = traversible != 0
    if gibson:
        goals = cv2.dilate(np.asarray(goal, dtype=np.float32), goal_kernel) == 1
        free |= goals
    return free


def _replace_stg(plan, mode, map_resolution_cm, lookahead_cm, free):
    plan.info["readout_mode"] = mode
    plan.info["readout_lookahead_cm"] = lookahead_cm if mode == "length" else None
    plan.info["readout_applied"] = False
    if len(plan.path) <= 1:
        return plan
    old_stg, old_stop, old_replan = plan.stg, plan.stop, plan.replan
    stg, path_index, path_distance_cm = select_stg(
        plan.path, mode, map_resolution_cm, lookahead_cm, free,
    )
    plan.stg = stg
    plan.info["readout_applied"] = True
    plan.info["readout_old_stg"] = list(map(float, old_stg))
    plan.info["readout_path_index"] = int(path_index)
    plan.info["readout_path_distance_cm"] = float(path_distance_cm)
    plan.info["readout_path_size"] = len(plan.path)
    if plan.stop != old_stop or plan.replan != old_replan:
        raise AssertionError("Waypoint readout changed STOP or replan")
    return plan


def gibson_plan_opt(grid, start, goal, collision, visited, obstacle_kernel, goal_kernel,
                    delay_quantum=0.0, mode="length", lookahead_cm=25.0,
                    map_resolution_cm=5.0):
    plan = gibson_plan_strict(
        grid, start, goal, collision, visited, obstacle_kernel, goal_kernel,
        delay_quantum=delay_quantum,
    )
    free = None
    if mode == "visible" and len(plan.path) > 1:
        free = _free_for_ray(
            grid, start, goal, collision, visited, obstacle_kernel, goal_kernel,
            gibson=not plan.info.get("no_path_repaired", False),
        )
    return _replace_stg(plan, mode, map_resolution_cm, lookahead_cm, free)


def mp3d_plan_opt(grid, start, goal, collision, visited, obstacle_kernel, goal_kernel,
                  map_resolution=5, wscale=4.0, niters=1, downsampling=1,
                  delay_quantum=0.0, mode="length", lookahead_cm=25.0):
    plan = mp3d_plan_strict(
        grid, start, goal, collision, visited, obstacle_kernel, goal_kernel,
        map_resolution=map_resolution, wscale=wscale, niters=niters,
        downsampling=downsampling, delay_quantum=delay_quantum,
    )
    free = None
    if mode == "visible" and len(plan.path) > 1:
        free = _free_for_ray(
            grid, start, goal, collision, visited, obstacle_kernel, goal_kernel,
            gibson=False,
        )
    return _replace_stg(plan, mode, map_resolution, lookahead_cm, free)
