"""Check forward-guard invariants on synthetic and captured MP3D maps."""

import argparse
import math
from pathlib import Path

import numpy as np

from plot_waveopt_neuron import _args_from_snapshot, _load_audit
from poni.planner_adapter_mp3d_action_readout_neuron import mp3d_plan_action_readout
from poni.planner_adapter_mp3d_tiebreak_neuron import mp3d_plan_tiebreak
from poni.planner_adapter_neuron import turn_action
from poni.planner_adapter_strict_neuron import mp3d_plan_strict


def compare(args, heading, turn_angle, **kwargs):
    old = mp3d_plan_strict(*args, **kwargs)
    tie = mp3d_plan_tiebreak(*args, **kwargs)
    guarded = mp3d_plan_action_readout(
        *args, heading=heading, turn_angle=turn_angle,
        preserve_forward=True, **kwargs)
    gain = mp3d_plan_action_readout(
        *args, heading=heading, turn_angle=turn_angle,
        forward_gain_only=True, **kwargs)
    assert guarded.stop == old.stop == tie.stop
    assert guarded.replan == old.replan == tie.replan
    assert gain.stop == old.stop and gain.replan == old.replan
    assert guarded.info["spikes"] == old.info["spikes"] == tie.info["spikes"]
    assert guarded.info["events_emitted"] == old.info["events_emitted"] == tie.info["events_emitted"]
    assert gain.info["spikes"] == old.info["spikes"]
    if old.info["arrival_cost"] is None:
        assert guarded.path == gain.path == old.path == tie.path == []
        return None, None, False, False
    assert math.isclose(guarded.info["arrival_cost"], old.info["arrival_cost"], abs_tol=1e-6)
    assert math.isclose(guarded.info["path_cost"], old.info["path_cost"], abs_tol=1e-5)
    assert math.isclose(gain.info["path_cost"], old.info["path_cost"], abs_tol=1e-5)
    start = args[1]
    old_action = turn_action(old.stg, start, heading, turn_angle)
    tie_action = turn_action(tie.stg, start, heading, turn_angle)
    if old_action == 1 and tie_action in (2, 3):
        assert guarded.stg == old.stg and guarded.path == old.path
        assert guarded.info["tie_forward_guarded"]
    else:
        assert guarded.stg == tie.stg and guarded.path == tie.path
        assert not guarded.info["tie_forward_guarded"]
    gain_expected = old_action in (2, 3) and tie_action == 1
    assert gain.info["tie_forward_gain_accepted"] == gain_expected
    if gain_expected:
        assert gain.stg == tie.stg and gain.path == tie.path
    else:
        assert gain.stg == old.stg and gain.path == old.path
    return old_action, tie_action, guarded.info["tie_forward_guarded"], gain_expected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-root", type=Path)
    parser.add_argument("--replay-run-root", type=Path, action="append",
                        help="Compare refactored readout with archived pilot audit snapshots")
    args = parser.parse_args()
    grid = np.zeros((40, 40), dtype=np.uint8)
    goal = np.zeros_like(grid)
    goal[25, 25] = 1
    zeros = np.zeros_like(grid)
    kernel = np.ones((3, 3), dtype=np.uint8)
    goal_kernel = np.ones((1, 1), dtype=np.uint8)
    compare((grid, (5, 5), goal, zeros, zeros, kernel, goal_kernel), 0, 30)
    grid[13:22, 17:21] = 1
    compare((grid, (5, 5), goal, zeros, zeros, kernel, goal_kernel), 0, 30)
    if args.snapshot_root:
        total = guarded_count = gain_count = skipped = 0
        for path, _, _ in _load_audit(args.snapshot_root, "mp3d"):
            if not path.exists():
                continue
            with np.load(path) as data:
                planner_args = _args_from_snapshot(data)
                kwargs = {
                    "map_resolution": float(data["map_resolution"]),
                    "wscale": float(data["wscale"]),
                    "niters": int(data["niters"]),
                }
                try:
                    _, _, guarded, gained = compare(
                        planner_args, float(data["heading"]),
                        float(data["turn_angle"]), **kwargs)
                except (ValueError, IndexError):
                    skipped += 1
                    continue
            total += 1
            guarded_count += int(guarded)
            gain_count += int(gained)
        print(f"MP3D_ACTION_READOUT_SNAPSHOT_OK total={total} "
              f"guarded={guarded_count} gained={gain_count} skipped={skipped}")
    for run_root in args.replay_run_root or []:
        checked = no_path = 0
        for path, event, _ in _load_audit(run_root, "mp3d"):
            if not path.exists():
                continue
            with np.load(path) as data:
                info = event["selected"]["info"]
                mode = info.get("readout_mode")
                if mode not in ("equal_arrival_forward_guard",
                                "equal_arrival_forward_gain_only"):
                    no_path += 1
                    continue
                plan = mp3d_plan_action_readout(
                    *_args_from_snapshot(data),
                    map_resolution=float(data["map_resolution"]),
                    wscale=float(data["wscale"]), niters=int(data["niters"]),
                    heading=float(data["heading"]),
                    turn_angle=float(data["turn_angle"]),
                    preserve_forward=mode == "equal_arrival_forward_guard",
                    forward_gain_only=mode == "equal_arrival_forward_gain_only",
                )
            assert plan.stg == tuple(event["selected"]["stg"])
            assert plan.stop == event["selected"]["stop"]
            assert plan.replan == event["selected"]["replan"]
            assert plan.info["spikes"] == info["spikes"]
            assert math.isclose(plan.info["path_cost"], info["path_cost"], abs_tol=1e-6)
            assert plan.info["tie_forward_guarded"] == info["tie_forward_guarded"]
            assert plan.info["tie_forward_gain_accepted"] == info.get(
                "tie_forward_gain_accepted", False)
            checked += 1
        print(f"MP3D_ACTION_READOUT_REPLAY_OK root={run_root} "
              f"checked={checked} no_path_skipped={no_path}")
    print("MP3D_FORWARD_GUARD_UNIT_OK")


if __name__ == "__main__":
    main()
