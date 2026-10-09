"""CPU validation: independent graph/tick oracles, upstream adapters, IPC, replay.

Run from PONI root with its conda Python. No Habitat scenes are needed here.
"""
import argparse
import copy
import glob
import json
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "hlab"))
sys.path.insert(0, str(ROOT / "dependencies/astar_pycpp"))

import cv2
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from skimage.morphology import disk
import pyastar

from poni.wavefront_neuron import propagate, warmup, JIT_AVAILABLE
from poni.planner_adapter_neuron import (gibson_plan, mp3d_plan, path_cost,
                                         proximity_weights, NoPathNeuron, turn_action, PlanNeuron)
from poni.runtime_neuron import AuditNeuron
from semexp.envs.habitat.sem_exp import Sem_Exp_Env_Agent
from semexp.envs.habitat.sem_exp_neuron import SemExpEnvAgentNeuron, make_env_fn_neuron
from utils.planners import PlannerActor, PlannerActorSequential
from utils.planners_neuron import PlannerActorNeuron, PlannerActorVectorNeuron


def graph_oracle(free, weights, seeds, reverse=False, quantum=0.0):
    h, w = free.shape
    rows, cols, values = [], [], []
    for r, c in np.argwhere(free):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if (dr == dc == 0) or not (0 <= nr < h and 0 <= nc < w) or not free[nr, nc]:
                    continue
                delay = math.hypot(dr, dc) * (weights[r, c] if reverse else weights[nr, nc])
                if quantum:
                    delay = max(quantum, math.floor(delay / quantum + 0.5) * quantum)
                rows.append(r * w + c)
                cols.append(nr * w + nc)
                values.append(delay)
    graph = csr_matrix((values, (rows, cols)), shape=(h * w, h * w))
    indices = np.array([r * w + c for r, c in seeds])
    return dijkstra(graph, directed=True, indices=indices, min_only=True).reshape(h, w)


def tick_oracle(free, weights, start):
    """Independent synchronous unit-tick IF simulation; no shortest-path queue."""
    h, w = free.shape
    arrivals = np.full((h, w), np.inf)
    pending = {0: [start]}
    t = 0
    while pending:
        for r, c in pending.pop(t, []):
            if np.isfinite(arrivals[r, c]):
                continue
            arrivals[r, c] = t
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = r + dr, c + dc
                    if (dr == dc == 0) or not (0 <= nr < h and 0 <= nc < w) or not free[nr, nc]:
                        continue
                    if np.isfinite(arrivals[nr, nc]):
                        continue
                    delay = max(1, int(math.floor(math.hypot(dr, dc) * weights[nr, nc] + 0.5)))
                    pending.setdefault(t + delay, []).append((nr, nc))
        t += 1
        assert t < 10000
    return arrivals


def equal_dist(a, b):
    assert np.array_equal(np.isfinite(a), np.isfinite(b)), "Reachability mismatch"
    finite = np.isfinite(a)
    if finite.any():
        assert np.allclose(a[finite], b[finite], rtol=1e-11, atol=1e-10)


def core_checks():
    rng = np.random.RandomState(706)
    count = 0
    for i in range(60):
        free = rng.uniform(size=(19, 23)) > 0.22
        seeds = [(1, 1), (17, 21)] if i % 3 == 0 else [(1, 1)]
        for cell in seeds:
            free[cell] = True
        weights = rng.choice([1.0, 2.0, 4.0], size=free.shape)
        quantum = (0.0, 0.1, 1.0)[i % 3]
        reverse = bool(i % 2)
        wave = propagate(free, seeds, weights, reverse=reverse, delay_quantum=quantum)
        equal_dist(wave.times, graph_oracle(free, weights, seeds, reverse, quantum))
        assert wave.spikes == np.isfinite(wave.times).sum()
        assert wave.spikes <= free.sum()
        for cell in list(zip(*np.where(np.isfinite(wave.times))))[::31]:
            path = wave.path_to(cell)
            assert path[0] in seeds and path[-1] == cell
            assert len(set(path)) == len(path)
        count += 1
    free = np.ones((7, 8), dtype=bool)
    free[3, 1:7] = False
    weights = rng.choice([1.0, 3.0], size=free.shape)
    equal_dist(propagate(free, [(1, 1)], weights, delay_quantum=1).times,
               tick_oracle(free, weights, (1, 1)))
    # A new planning query must not retain stale spikes or predecessor links.
    before = propagate(np.ones((8, 8), dtype=bool), [(1, 1)])
    blocked = np.ones((8, 8), dtype=bool)
    blocked[4, :] = False
    after = propagate(blocked, [(1, 1)])
    assert np.isfinite(before.times[6, 6]) and not np.isfinite(after.times[6, 6])
    targets = np.zeros((8, 8), dtype=bool)
    targets[1, 1] = True
    already = propagate(blocked, [(1, 1)], targets=targets)
    assert already.path_to((1, 1)) == [(1, 1)] and already.spikes == 1
    budget = propagate(blocked, [(1, 1)], max_spikes=2)
    assert budget.budget_exhausted and budget.spikes == 2
    for invalid in (np.zeros((3, 3)), np.full((3, 3), np.nan)):
        try:
            propagate(np.ones((3, 3)), [(0, 0)], invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid delay accepted")
    return {"scipy_oracle_cases": count, "tick_oracle": "passed",
            "reset_disconnection_start_at_goal_budget_invalid_delays": "passed"}


def astar_checks():
    rng = np.random.RandomState(707)
    records = []
    for i in range(80):
        obstacles = rng.uniform(size=(35, 41)) < 0.23
        start = (2, 2)
        obstacles[start] = False
        goals = np.zeros(obstacles.shape, np.float32)
        goals[30, 36] = 1
        goals[8, 30] = 1
        if i % 2:
            goals = cv2.dilate(goals, disk(3).astype(np.uint8))
        goals[obstacles] = 0
        goals[30, 36] = 1
        obstacles[30, 36] = False
        if i % 10 == 0:
            obstacles[:, 15] = True
            goals[:, :16] = 0
        weights = proximity_weights(obstacles, 4.0, 1 + i % 2)
        # Verify weight construction independently of OpenCV.
        naive = np.where(obstacles, 4.0 ** (2 + i % 2), 1.0)
        for _ in range(1 + i % 2):
            nxt = naive.copy()
            for r, c in np.argwhere(~obstacles):
                nbr = [naive[r + dr, c + dc] for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                       if (dr or dc) and 0 <= r + dr < 35 and 0 <= c + dc < 41]
                nxt[r, c] = max(max(nbr) / 4.0, 1.0)
            naive = nxt
        assert np.array_equal(naive, weights)
        wave = propagate(~obstacles, [start], weights=weights, targets=goals)
        xs, ys = pyastar.multi_goal_weighted_astar_planner(obstacles.astype(np.float32),
                                                         [start[1], start[0]], goals,
                                                         True, wscale=4.0, niters=1 + i % 2)
        assert (wave.reached >= 0) == (xs is not None)
        row = {"case": i, "reachable": wave.reached >= 0}
        if wave.reached >= 0:
            snn_path = wave.path_to(divmod(wave.reached, obstacles.shape[1]))
            ref_path = list(zip(ys[::-1], xs[::-1]))
            snn_cost, ref_cost = path_cost(snn_path, weights), path_cost(ref_path, weights)
            assert snn_cost <= ref_cost + 1e-4, (i, snn_cost, ref_cost)
            row.update(neuron_cost=snn_cost, astar_cost=ref_cost, path_equal=snn_path == ref_path)
        records.append(row)
    if not records:
        raise ValueError("No map snapshots matched replay pattern: " + pattern)
    return records


def planner_cfg(size=64):
    return NS(map_size_cm=size * 5, map_resolution=5, collision_threshold=0.2,
              turn_angle=30, stg_downsampling=1, stg_disk_size=10,
              move_as_close_as_possible=True, move_close_limit=25,
              weighted_scale=4.0, weighted_niters=1, conseq_replan_thresh=5,
              n_planners=1, ACTION=NS(stop=0, move_forward=1, turn_left=2, turn_right=3))


def gibson_agent(cls, size):
    obj = cls.__new__(cls)
    obj.args = NS(map_resolution=5, turn_angle=30, collision_threshold=0.2,
                  visualize=0, print_images=0)
    obj.selem = disk(3).astype(np.uint8)
    obj.stg_selem = disk(10).astype(np.uint8)
    obj.collision_map = np.zeros((size, size))
    obj.visited = np.zeros((size, size))
    obj.curr_loc = [size * 0.025, size * 0.025, 0.0]
    obj.last_action = None
    obj.col_width = 1
    obj.episode_no = 1
    obj.timestep = 0
    obj.scene_name = "synthetic_interface"
    return obj


def interface_checks():
    rng = np.random.RandomState(709)
    os.environ["PONI_NEURON_MODE"] = "guarded"
    records = []
    for i in range(36):
        n = 64
        grid = np.zeros((n, n), dtype=np.float32)
        grid[10:50, 40] = 1
        grid[27:39, 40] = 0
        if i % 3 == 0:
            grid[15:25, 12:23] = 1
        goal = np.zeros_like(grid)
        goal[8 + i % 4 * 12, 52] = 1
        if i % 4 == 0:
            goal[50, 10] = 1
        heading = float(rng.uniform(-180, 180))
        inputs = {"map_pred": grid, "goal": goal,
                  "pose_pred": [1.6, 1.6, heading, 0, n, 0, n],
                  "found_goal": i % 2, "new_goal": True, "wait": False}
        a = gibson_agent(Sem_Exp_Env_Agent, n)
        b = gibson_agent(SemExpEnvAgentNeuron, n)
        assert a._plan(inputs) == b._plan(inputs), "Gibson controller changed in guarded mode"
        cfg = planner_cfg(n)
        a, b = PlannerActor(cfg), PlannerActorNeuron(cfg)
        a.reset()
        b.reset()
        if i % 3 == 1:
            a.is_close_to_goal = b.is_close_to_goal = True
            a.is_close_time = b.is_close_time = 24
        for step in range(3):
            state = np.random.get_state()
            ref_action, ref_flag = a.plan_and_act(inputs)
            np.random.set_state(state)
            action, flag = b.plan_and_act(inputs)
            assert np.array_equal(action, ref_action) and flag == ref_flag
            assert a.conseq_replans == b.conseq_replans
            assert a.is_close_to_goal == b.is_close_to_goal
            assert a.is_close_time == b.is_close_time
            assert np.array_equal(a.collision_map, b.collision_map)
        records.append({"case": i, "gibson_action_equal": True, "mp3d_3step_state_equal": True})
    # New worker function is exercised, not only the single-process class.
    cfg = planner_cfg()
    cfg.n_planners = 2
    vec = PlannerActorVectorNeuron(cfg)
    ref = PlannerActorSequential(cfg)
    try:
        actions, flags = vec.plan_and_act([inputs, inputs], [0, 0])
        ref_actions, ref_flags = ref.plan_and_act([inputs, inputs], [0, 0])
        assert np.array_equal(actions, ref_actions) and np.array_equal(flags, ref_flags)
    finally:
        vec.close()
    import pickle
    assert pickle.loads(pickle.dumps(make_env_fn_neuron)) is make_env_fn_neuron
    return {"cases": records, "mp3d_two_worker_ipc": "passed", "gibson_factory_pickle": "passed"}


def guard_edge_checks():
    os.environ["PONI_NEURON_MODE"] = "guarded"
    base = PlanNeuron((5, 9), False, False)
    for dataset in ("gibson", "mp3d"):
        audit = AuditNeuron(dataset)
        for candidate in (PlanNeuron((5, 9), True, False), PlanNeuron((9, 5), False, False)):
            selected = audit.choose(lambda: candidate, lambda: base, (5, 5), 0, 30)
            assert selected is base
        if dataset == "mp3d":
            candidate = PlanNeuron((5, 9), False, True)
            assert audit.choose(lambda: candidate, lambda: base, (5, 5), 0, 30) is base
        def fail():
            raise NoPathNeuron("explicit_disconnected_case")
        assert audit.choose(fail, lambda: base, (5, 5), 0, 30) is base
        same_action = PlanNeuron((5, 8), False, False)
        assert audit.choose(lambda: same_action, lambda: base, (5, 5), 0, 30) is same_action
    # The candidate reports empty goal instead of silently fabricating a path.
    grid = np.zeros((32, 32), np.float32)
    for planner in (gibson_plan, mp3d_plan):
        try:
            planner(grid, (16, 16), np.zeros_like(grid), grid, grid, disk(3).astype(np.uint8), disk(10).astype(np.uint8))
        except NoPathNeuron:
            pass
        else:
            raise AssertionError("Empty goal was accepted")
    return {"stop_action_replan_error_fallbacks": "passed", "same_action_acceptance": "passed",
            "empty_goal_detection": "passed"}


def replay_checks(pattern):
    records = []
    for path in sorted(glob.glob(pattern)):
        with np.load(path) as data:
            snapshot = {k: data[k] for k in data.files}
        grid, goal, start = snapshot["grid"], snapshot["goal"], snapshot["start"]
        n, m = grid.shape
        # Recorded Gibson maps are square; keep original global coordinate crop empty-offset.
        assert n == m
        agent = gibson_agent(Sem_Exp_Env_Agent, n)
        agent.selem = snapshot["obstacle_kernel"]
        agent.stg_selem = snapshot["goal_kernel"]
        agent.collision_map = snapshot["collision"].copy()
        agent.visited = snapshot["visited"].copy()
        tick = time.perf_counter()
        ref_stg, ref_stop = agent._get_stg(grid, start, goal.copy(), [0, n, 0, m])
        reference_ms = (time.perf_counter() - tick) * 1000
        row = {"snapshot": path, "reference_ms": reference_ms}
        try:
            candidate = gibson_plan(grid, start, goal, snapshot["collision"], snapshot["visited"],
                                    snapshot["obstacle_kernel"], snapshot["goal_kernel"])
            heading = float(snapshot["heading"])
            turn = float(snapshot["turn_angle"])
            row.update(candidate.info)
            row["stop_equal"] = bool(candidate.stop == ref_stop)
            row["stg_equal"] = bool(np.allclose(candidate.stg, ref_stg))
            row["turn_equal"] = turn_action(candidate.stg, start, heading, turn) == turn_action(ref_stg, start, heading, turn)
            row["guard_accept"] = row["stop_equal"] and row["turn_equal"]
        except NoPathNeuron as exc:
            row["error"] = str(exc)
            row["guard_accept"] = False
        records.append(row)
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--replay-glob")
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    os.environ["PONI_NEURON_LOG_DIR"] = str(root / "interface_audit")
    os.environ["PONI_NEURON_CAPTURE_LIMIT"] = "0"
    warmup()
    result = {"jit_available": JIT_AVAILABLE}
    if args.replay_glob:
        result["replay"] = replay_checks(args.replay_glob)
    else:
        result["core"] = core_checks()
        print("Core graph/tick oracle checks passed", flush=True)
        result["weighted_astar"] = astar_checks()
        print("80 weighted A* comparisons passed", flush=True)
        result["interfaces"] = interface_checks()
        result["guard_edges"] = guard_edge_checks()
        print("Gibson/MP3D controller and IPC checks passed", flush=True)
    result["status"] = "passed"
    with (root / "validation.json").open("x") as fp:
        json.dump(result, fp, indent=2)
    print(json.dumps({"status": result["status"], "output": str(root / "validation.json")}), flush=True)


if __name__ == "__main__":
    main()
