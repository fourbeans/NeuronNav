"""End-to-end synthetic invariants for the separate MP3D SNN tie-breaker."""

import math

import numpy as np

from poni.planner_adapter_mp3d_tiebreak_neuron import mp3d_plan_tiebreak
from poni.planner_adapter_strict_neuron import mp3d_plan_strict


def compare(args):
    old = mp3d_plan_strict(*args)
    new = mp3d_plan_tiebreak(*args)
    assert old.stop == new.stop and old.replan == new.replan
    assert old.info["spikes"] == new.info["spikes"]
    assert old.info["events_emitted"] == new.info["events_emitted"]
    assert math.isclose(old.info["arrival_cost"], new.info["arrival_cost"], abs_tol=1e-7)
    assert math.isclose(old.info["path_cost"], new.info["path_cost"], abs_tol=1e-6)
    assert old.path[0] == new.path[0] and old.path[-1] == new.path[-1]
    assert new.info["readout_mode"] == "equal_arrival_straight_tie"
    assert abs(new.info["tie_cost_delta"]) < 1e-6


def main():
    shape = (40, 40)
    grid = np.zeros(shape, dtype=np.uint8)
    goal = np.zeros(shape, dtype=np.uint8)
    goal[25, 25] = 1
    zeros = np.zeros(shape, dtype=np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    goal_kernel = np.ones((1, 1), dtype=np.uint8)
    compare((grid, (5, 5), goal, zeros, zeros, kernel, goal_kernel))
    goal_near = np.zeros(shape, dtype=np.uint8)
    goal_near[6, 6] = 1
    compare((grid, (5, 5), goal_near, zeros, zeros, kernel, goal_kernel))
    grid[13:22, 17:21] = 1
    compare((grid, (5, 5), goal, zeros, zeros, kernel, goal_kernel))

    blocked = np.ones(shape, dtype=np.uint8)
    blocked_goal = np.zeros(shape, dtype=np.uint8)
    blocked_goal[35, 35] = 1
    args = (blocked, (5, 5), blocked_goal, zeros, zeros, kernel, goal_kernel)
    np.random.seed(42)
    old = mp3d_plan_strict(*args)
    np.random.seed(42)
    new = mp3d_plan_tiebreak(*args)
    assert old.stg == new.stg and old.path == new.path == []
    assert old.stop == new.stop and old.replan == new.replan
    print("MP3D_TIEBREAK_UNIT_OK")


if __name__ == "__main__":
    main()
