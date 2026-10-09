"""Synthetic invariants for the isolated MP3D bounded-visible SNN readout."""

import math

import numpy as np

from poni.planner_adapter_mp3d_refine_neuron import _weights_for_ray, mp3d_plan_refine
from poni.planner_adapter_strict_neuron import mp3d_plan_strict
from poni.waypoint_mp3d_refine_neuron import line_cells, select_bounded_visible


def test_readout():
    assert np.all(_weights_for_ray(np.ones((5, 5), dtype=bool), 4, 1) == 1)
    edge_obstacle = np.ones((5, 5), dtype=bool)
    edge_obstacle[0, 0] = False
    edge_weights = _weights_for_ray(edge_obstacle, 4, 1)
    assert edge_weights[0, 1] == 4 and edge_weights[4, 4] == 1
    free = np.ones((15, 15), dtype=bool)
    weights = np.ones_like(free, dtype=np.float32)
    path = [(4, 2), (4, 3), (4, 4), (4, 5), (4, 6), (3, 7), (2, 8)]
    stg, idx, arc, _, changed = select_bounded_visible(path, free, weights, 5, 50)
    assert (stg, idx, changed) == (path[-1], 6, True)
    assert 30 < arc < 50
    stg, idx, _, _, changed = select_bounded_visible(path, free, weights, 5, 30)
    assert (stg, idx, changed) == (path[5], 5, True)

    ray = line_cells(path[0], path[-1])
    blocked = next(cell for cell in ray[1:-1] if cell not in path)
    free[blocked] = False
    stg, idx, _, _, changed = select_bounded_visible(path, free, weights, 5, 50)
    assert idx < 6
    assert all(free[cell] for cell in line_cells(path[0], stg))
    free[blocked] = True

    weights[blocked] = 20.0
    no_guard = select_bounded_visible(path, free, weights, 5, 50, cost_guard=False)
    guard = select_bounded_visible(path, free, weights, 5, 50, cost_guard=True)
    assert no_guard[1] == 6 and guard[1] < 6
    assert guard[1] >= 4
    angle_guard = select_bounded_visible(
        path, free, np.ones_like(weights), 5, 50,
        max_direction_delta_deg=15,
    )
    assert angle_guard[1] == 5
    assert select_bounded_visible(path[:3], free, weights, 5, 50)[0] == path[2]


def test_adapter():
    shape = (40, 40)
    grid = np.zeros(shape, dtype=np.uint8)
    goal = np.zeros(shape, dtype=np.uint8)
    goal[25, 25] = 1
    zeros = np.zeros(shape, dtype=np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    goal_kernel = np.ones((1, 1), dtype=np.uint8)
    args = (grid, (5, 5), goal, zeros, zeros, kernel, goal_kernel)
    old = mp3d_plan_strict(*args)
    new = mp3d_plan_refine(*args, cap_cm=50, cost_guard=True)
    assert old.path == new.path
    assert old.stop == new.stop and old.replan == new.replan
    assert math.isclose(old.info["arrival_cost"], new.info["arrival_cost"])
    assert math.isclose(old.info["path_cost"], new.info["path_cost"])
    assert 4 <= new.info["readout_path_index"] < len(new.path)
    assert new.info["readout_path_distance_cm"] <= 50

    blocked = np.ones(shape, dtype=np.uint8)
    blocked_goal = np.zeros(shape, dtype=np.uint8)
    blocked_goal[35, 35] = 1
    blocked_args = (blocked, (5, 5), blocked_goal, zeros, zeros, kernel, goal_kernel)
    np.random.seed(42)
    old = mp3d_plan_strict(*blocked_args)
    np.random.seed(42)
    new = mp3d_plan_refine(*blocked_args, cap_cm=50, cost_guard=True)
    assert old.path == new.path == []
    assert old.stg == new.stg and old.stop == new.stop and old.replan == new.replan
    assert new.info["readout_applied"] is False


if __name__ == "__main__":
    test_readout()
    test_adapter()
    print("MP3D_REFINE_UNIT_OK")
