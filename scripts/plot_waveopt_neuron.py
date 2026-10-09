"""Compact diagnostic PNGs from a wavefront-readout pilot's frozen map snapshots."""

import argparse
import glob
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from poni.planner_adapter_opt_neuron import _free_for_ray, gibson_plan_opt, mp3d_plan_opt
from poni.planner_adapter_strict_neuron import gibson_plan_strict, mp3d_plan_strict
from poni.planner_adapter_neuron import turn_action
from poni.fmm_planner import FMMPlanner


def _load_audit(root, dataset):
    pattern = "audit/*.jsonl" if dataset == "gibson" else "audit_val_part_*/*.jsonl"
    records = []
    for path in sorted(glob.glob(str(root / pattern))):
        events = [json.loads(line) for line in open(path)]
        poses = {int(e["seq"]): e for e in events if e["type"] == "pose"}
        for event in events:
            if event["type"] == "plan" and "snapshot" in event:
                records.append((Path(path).parent / event["snapshot"], event, poses.get(int(event["seq"]))))
    return records


def _args_from_snapshot(snapshot):
    keys = ("grid", "start", "goal", "collision", "visited", "obstacle_kernel", "goal_kernel")
    return [snapshot[k] for k in keys]


def _plans(dataset, data, mode, lookahead_cm):
    args = _args_from_snapshot(data)
    if dataset == "gibson":
        old = gibson_plan_strict(*args)
        new = gibson_plan_opt(*args, mode=mode, lookahead_cm=lookahead_cm,
                               map_resolution_cm=float(data["map_resolution"]))
    else:
        kwargs = dict(map_resolution=int(data["map_resolution"]),
                      wscale=float(data["wscale"]), niters=int(data["niters"]))
        old = mp3d_plan_strict(*args, **kwargs)
        new = mp3d_plan_opt(*args, mode=mode, lookahead_cm=lookahead_cm, **kwargs)
    return old, new


def _reference_stg(data, dataset):
    args = _args_from_snapshot(data)
    grid, start, goal, collision, visited, obstacle_kernel, goal_kernel = args
    traversible = _free_for_ray(*args, gibson=False).astype(np.float32)
    padded = np.pad(traversible, 1, constant_values=1)
    goals = cv2.dilate(np.pad(np.asarray(goal, np.float32), 1), goal_kernel)
    padded_start = (int(start[0]) + 1, int(start[1]) + 1)
    if dataset == "gibson":
        planner = FMMPlanner(padded)
        planner.set_multi_goal(goals)
        r, c, _, _ = planner.get_short_term_goal(padded_start)
        return float(r - 1), float(c - 1)
    import pyastar
    obstacles = 1.0 - padded
    cols, rows = pyastar.multi_goal_weighted_astar_planner(
        obstacles.astype(np.float32),
        [padded_start[1], padded_start[0]],
        goals.astype(np.float32),
        True,
        wscale=float(data["wscale"]),
        niters=int(data["niters"]),
    )
    if cols is None:
        return None
    index = -min(5, len(cols))
    return float(rows[index] - 1), float(cols[index] - 1)


def _draw_map(data, dataset, old, new, event, path):
    args = _args_from_snapshot(data)
    free = _free_for_ray(*args, gibson=(dataset == "gibson" and not old.info.get("no_path_repaired", False)))
    grid, start, goal = args[:3]
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.imshow(~free, cmap="Greys", vmin=0, vmax=1, alpha=0.75, interpolation="nearest")
    gr, gc = np.where(goal > 0)
    ax.scatter(gc, gr, s=5, c="#e49b2c", label="goal mask")
    if old.path:
        coords = np.asarray(old.path)
        ax.plot(coords[:, 1], coords[:, 0], color="#2474b5", linewidth=1.2,
                label="SNN predecessor path")
    ax.scatter([start[1]], [start[0]], marker="*", s=125, c="#185b2c", label="start")
    ax.scatter([old.stg[1]], [old.stg[0]], marker="x", s=85, c="#bb3333", label="old STG")
    ax.scatter([new.stg[1]], [new.stg[0]], marker="o", s=60, facecolors="none",
               edgecolors="#8038a1", linewidths=1.6, label="new STG")
    try:
        reference = _reference_stg(data, dataset)
    except Exception:
        reference = None
    if reference is not None:
        ax.scatter([reference[1]], [reference[0]], marker="^", s=65,
                   c="#1b9e89", label="FMM/A* STG")
    ax.plot([start[1], new.stg[1]], [start[0], new.stg[0]], color="#8038a1",
            linewidth=1.1, linestyle="--")
    ax.set(xlabel="map column (5 cm/cell)", ylabel="map row (5 cm/cell)")
    ax.set_title(f"{dataset}: old/new SNN waypoint, seq={event['seq']}")
    ax.set_xlim(max(0, start[1] - 90), min(grid.shape[1] - 1, start[1] + 90))
    ax.set_ylim(min(grid.shape[0] - 1, start[0] + 90), max(0, start[0] - 90))
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def map_examples(root, dataset, mode, lookahead_cm, output, limit):
    candidates = []
    for snapshot_path, event, pose in _load_audit(root, dataset):
        if not snapshot_path.exists():
            continue
        with np.load(snapshot_path) as data:
            try:
                old, new = _plans(dataset, data, mode, lookahead_cm)
            except Exception:
                continue
            if len(old.path) <= 1:
                continue
            heading = float(data["heading"])
            start = data["start"]
            turn = float(data["turn_angle"])
            old_action = turn_action(old.stg, start, heading, turn)
            new_action = turn_action(new.stg, start, heading, turn)
            shift = np.linalg.norm(np.asarray(new.stg) - np.asarray(old.stg))
            score = (100 if old_action != new_action else 0) + min(shift, 50)
            if old.info.get("no_path_repaired"):
                score += 10
            candidates.append((score, snapshot_path, event, pose))
    selected = sorted(candidates, key=lambda row: -row[0])[:limit]
    for idx, (_, snapshot_path, event, _) in enumerate(selected):
        with np.load(snapshot_path) as data:
            old, new = _plans(dataset, data, mode, lookahead_cm)
            _draw_map(data, dataset, old, new, event, output / f"map_{idx + 1:02d}.png")
    return len(selected)


def trajectory(root, dataset, output):
    pattern = "audit/*.jsonl" if dataset == "gibson" else "audit_val_part_*/*.jsonl"
    best = None
    for filename in glob.glob(str(root / pattern)):
        groups = {}
        for line in open(filename):
            event = json.loads(line)
            if event["type"] != "pose":
                continue
            key = event.get("episode_no", event.get("episode_index", 0))
            groups.setdefault(key, []).append(event)
        for key, events in groups.items():
            if len(events) >= 20 and (best is None or len(events) > len(best[2])):
                best = (filename, key, events)
    if best is None:
        return False
    poses = np.array([e["pose_pred"][:2] for e in best[2][:40]], float)
    fig, ax = plt.subplots(figsize=(6.2, 5.1))
    ax.plot(poses[:, 0], poses[:, 1], color="#448a9a", linewidth=1.2)
    dots = ax.scatter(poses[:, 0], poses[:, 1], c=np.arange(len(poses)), cmap="viridis", s=25)
    ax.scatter(poses[0, 0], poses[0, 1], marker="*", s=110, c="#248544")
    ax.set(xlabel="map x (m)", ylabel="map y (m)")
    ax.set_title(f"{dataset}: first {len(poses)} poses, episode {best[1]}")
    ax.axis("equal")
    fig.colorbar(dots, ax=ax, label="navigation step")
    fig.tight_layout()
    fig.savefig(output / "trajectory_40_steps.png", dpi=160)
    plt.close(fig)
    return True


def outcome_chart(root, output):
    summary_path = root / "summary_waveopt.json"
    if not summary_path.exists():
        return False
    summary = json.loads(summary_path.read_text())
    names, gains, losses = [], [], []
    for name, data in summary["baselines"].items():
        if not data["cohort_match"]:
            continue
        flips = data["difference"]["success"]
        names.append(name)
        gains.append(flips["improved"])
        losses.append(flips["worsened"])
    if not names:
        return False
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7, 2.2 + len(names)))
    ax.barh(y - 0.18, gains, height=0.33, color="#278163", label="candidate only succeeds")
    ax.barh(y + 0.18, losses, height=0.33, color="#ba5860", label="baseline only succeeds")
    ax.set_yticks(y, names)
    ax.set(xlabel="paired episodes", title=f"{summary['dataset']}: Success flips")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "success_flips.png", dpi=160)
    plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("gibson", "mp3d"), required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("length", "visible"), required=True)
    parser.add_argument("--lookahead-cm", type=float, default=25.0)
    parser.add_argument("--max-maps", type=int, default=4)
    args = parser.parse_args()
    out = args.run_root / "figures"
    out.mkdir(parents=True, exist_ok=True)
    maps = map_examples(args.run_root, args.dataset, args.mode,
                        args.lookahead_cm, out, args.max_maps)
    traced = trajectory(args.run_root, args.dataset, out)
    flips = outcome_chart(args.run_root, out)
    print(json.dumps({"map_examples": maps, "trajectory": traced,
                      "success_flips": flips, "output": str(out)}))


if __name__ == "__main__":
    main()
