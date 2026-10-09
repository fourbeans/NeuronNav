#!/usr/bin/env python3
"""Prove that reachable Gibson queries retain the pre-repair behavior."""

import argparse
import ast
import hashlib
import importlib.util
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np

from poni.planner_adapter_strict_neuron import gibson_plan_strict as repaired_plan


def load_before(path):
    loader = SourceFileLoader("planner_strict_before", path)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def function_hash(path, name):
    source = Path(path).read_text()
    node = next(
        item for item in ast.parse(source).body
        if isinstance(item, ast.FunctionDef) and item.name == name
    )
    return hashlib.sha256(ast.get_source_segment(source, node).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", required=True)
    parser.add_argument("--current", required=True)
    args = parser.parse_args()
    before = load_before(args.before)
    rng = np.random.default_rng(20261003)
    checked = 0
    kernel = np.ones((1, 1), dtype=np.uint8)
    for _ in range(64):
        grid = (rng.random((48, 48)) < 0.035).astype(np.float32)
        start = tuple(map(int, rng.integers(4, 44, size=2)))
        target = tuple(map(int, rng.integers(4, 44, size=2)))
        goal = np.zeros_like(grid)
        goal[target] = 1
        zeros = np.zeros_like(grid)
        old = before.gibson_plan_strict(
            grid, start, goal, zeros, zeros, kernel, kernel,
        )
        if old.info["no_path"]:
            continue
        new = repaired_plan(grid, start, goal, zeros, zeros, kernel, kernel)
        assert new.stg == old.stg
        assert new.stop == old.stop
        assert new.replan == old.replan
        assert new.path == old.path
        assert new.info["spikes"] == old.info["spikes"]
        assert new.info["events_emitted"] == old.info["events_emitted"]
        assert new.info["arrival_cost"] == old.info["arrival_cost"]
        assert not new.info["projection_used"]
        checked += 1
    assert checked >= 50

    old_mp3d = function_hash(args.before, "mp3d_plan_strict")
    new_mp3d = function_hash(args.current, "mp3d_plan_strict")
    assert old_mp3d == new_mp3d
    print("reachable_cases_exact_match={}".format(checked))
    print("mp3d_function_sha256={}".format(new_mp3d))
    print("status=pass")


if __name__ == "__main__":
    main()
