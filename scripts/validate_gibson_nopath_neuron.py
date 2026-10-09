#!/usr/bin/env python3
"""Focused regression tests for the Gibson strict-SNN no-path repair."""

import argparse
import json

import cv2
import numpy as np

from poni.planner_adapter_strict_neuron import gibson_plan_strict


def kernels():
    return np.ones((1, 1), dtype=np.uint8), np.ones((1, 1), dtype=np.uint8)


def call(grid, start, goal):
    obstacle_kernel, goal_kernel = kernels()
    zeros = np.zeros_like(grid, dtype=np.float32)
    return gibson_plan_strict(
        grid, start, goal, zeros, zeros, obstacle_kernel, goal_kernel,
    )


def reachable_case():
    grid = np.zeros((24, 24), dtype=np.float32)
    goal = np.zeros_like(grid)
    goal[4, 18] = 1
    plan = call(grid, (18, 4), goal)
    assert not plan.info["projection_used"]
    assert not plan.info["no_path"]
    assert not plan.info["no_path_detected"]
    assert plan.path[0] == (18, 4)
    assert plan.path[-1] == (4, 18)
    return plan


def disconnected_case():
    grid = np.zeros((28, 28), dtype=np.float32)
    # Enclose the semantic target.  The padded outer border cannot reconnect it.
    cv2.rectangle(grid, (17, 7), (25, 15), 1, thickness=1)
    goal = np.zeros_like(grid)
    goal[11, 21] = 1
    start = (20, 4)
    plan = call(grid, start, goal)
    assert plan.info["projection_used"]
    assert plan.info["no_path_detected"]
    assert plan.info["no_path_repaired"]
    assert not plan.info["no_path"]
    assert not plan.stop
    assert plan.path[0] == start
    assert plan.path[-1] == tuple(plan.info["proxy_goal"])
    assert plan.stg != tuple(map(float, start))
    assert grid[tuple(plan.info["proxy_goal"])] == 0
    return plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    reachable = reachable_case()
    repaired = disconnected_case()
    report = {
        "status": "pass",
        "reachable": {
            "stg": reachable.stg,
            "path_length": len(reachable.path),
            "projection_used": reachable.info["projection_used"],
        },
        "disconnected": {
            "stg": repaired.stg,
            "path_length": len(repaired.path),
            "proxy_goal": repaired.info["proxy_goal"],
            "projection_distance_cells": repaired.info["projection_distance_cells"],
            "no_path_repaired": repaired.info["no_path_repaired"],
        },
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        with open(args.output, "w") as handle:
            handle.write(rendered + "\n")


if __name__ == "__main__":
    main()
