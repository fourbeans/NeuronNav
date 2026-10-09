"""Synthetic invariants for equal-arrival SNN predecessor backtrace."""

import math

import numpy as np

from poni.path_tiebreak_neuron import count_bends, decode_straight_ties
from poni.planner_adapter_neuron import path_cost
from poni.wavefront_neuron import propagate


def check(free, weights, start, target):
    goals = np.zeros_like(free, dtype=bool)
    goals[target] = True
    wave = propagate(free, [start], weights=weights, targets=goals)
    assert wave.reached >= 0
    old = wave.path_to(target)
    new, changed = decode_straight_ties(wave, weights, start, target)
    assert old[0] == new[0] == start and old[-1] == new[-1] == target
    assert all(free[cell] for cell in new)
    assert all(max(abs(a[0] - b[0]), abs(a[1] - b[1])) == 1
               for a, b in zip(new[:-1], new[1:]))
    assert math.isclose(path_cost(old, weights), path_cost(new, weights), abs_tol=1e-7)
    assert changed >= 0 and count_bends(new) >= 0


def main():
    free = np.ones((8, 8), dtype=bool)
    weights = np.ones((8, 8), dtype=np.float32)
    for target in ((5, 3), (6, 2), (3, 6)):
        check(free, weights, (1, 1), target)
    free[3:6, 3] = False
    weights[2:7, 2] = 4
    check(free, weights, (1, 1), (6, 6))
    print("PATH_TIEBREAK_UNIT_OK")


if __name__ == "__main__":
    main()
