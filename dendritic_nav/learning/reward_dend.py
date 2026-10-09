"""Privileged distance is accepted only here, never by the inference modules."""
import math


def phi(distance):
    if not math.isfinite(distance):
        raise ValueError("Non-finite training distance")
    return -min(max(distance / 20.0, 0.0), 1.0)


def reward(before, after, collision, terminal, success=0.0, spl=0.0,
           gamma=0.995, goal_switch=False):
    shaping = gamma * (0.0 if terminal else phi(after)) - phi(before)
    return (
        -0.001
        - 0.01 * int(collision)
        - 0.01 * int(goal_switch)
        + shaping
        + (5 * success + spl if terminal else 0.0)
    )
