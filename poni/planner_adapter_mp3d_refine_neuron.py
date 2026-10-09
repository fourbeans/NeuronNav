"""Thin strict-SNN MP3D adapter for bounded visible-path readout."""

import numpy as np

from poni.planner_adapter_neuron import proximity_weights
from poni.planner_adapter_opt_neuron import _free_for_ray
from poni.planner_adapter_strict_neuron import mp3d_plan_strict
from poni.waypoint_mp3d_refine_neuron import select_bounded_visible


def _weights_for_ray(free, wscale, niters):
    """Rebuild MP3D wave weights with its original free one-cell boundary."""
    padded_free = np.pad(np.asarray(free, dtype=bool), 1, constant_values=True)
    return proximity_weights(~padded_free, wscale=wscale, niters=niters)[1:-1, 1:-1]


def mp3d_plan_refine(grid, start, goal, collision, visited, obstacle_kernel,
                     goal_kernel, map_resolution=5, wscale=4.0, niters=1,
                     downsampling=1, delay_quantum=0.0, cap_cm=50.0,
                     cost_guard=False, max_direction_delta_deg=None):
    plan = mp3d_plan_strict(
        grid, start, goal, collision, visited, obstacle_kernel, goal_kernel,
        map_resolution=map_resolution, wscale=wscale, niters=niters,
        downsampling=downsampling, delay_quantum=delay_quantum,
    )
    old_stg, old_stop, old_replan = plan.stg, plan.stop, plan.replan
    plan.info["readout_mode"] = "bounded_visible_cost" if cost_guard else "bounded_visible"
    plan.info["readout_cap_cm"] = float(cap_cm)
    plan.info["readout_max_direction_delta_deg"] = max_direction_delta_deg
    plan.info["readout_applied"] = False
    if len(plan.path) <= 1:
        return plan
    free = _free_for_ray(grid, start, goal, collision, visited,
                         obstacle_kernel, goal_kernel, gibson=False)
    weights = _weights_for_ray(free, wscale, niters)
    stg, idx, arc_cm, ratio, changed = select_bounded_visible(
        plan.path, free, weights, map_resolution, cap_cm, cost_guard=cost_guard,
        max_direction_delta_deg=max_direction_delta_deg,
    )
    plan.stg = (float(stg[0]), float(stg[1]))
    plan.info["readout_applied"] = bool(changed)
    plan.info["readout_old_stg"] = list(map(float, old_stg))
    plan.info["readout_path_index"] = int(idx)
    plan.info["readout_path_distance_cm"] = float(arc_cm)
    plan.info["readout_ray_path_cost_ratio"] = float(ratio) if ratio is not None else None
    plan.info["readout_path_size"] = len(plan.path)
    if plan.stop != old_stop or plan.replan != old_replan:
        raise AssertionError("readout changed STOP/replan")
    return plan
