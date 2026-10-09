"""Overlay candidate and replayed old-SNN navigation poses on matched episodes."""

import argparse
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _poses(root, dataset):
    pattern = "audit/*.jsonl" if dataset == "gibson" else "audit_val_part_*/*.jsonl"
    traces = {}
    for filename in sorted(glob.glob(str(root / pattern))):
        part = Path(filename).parent.name.rsplit("_", 1)[-1]
        key = None
        for line in open(filename):
            event = json.loads(line)
            if event["type"] == "episode_start":
                key = ((Path(event["scene"]).stem, int(event["episode_no"]))
                       if dataset == "gibson" else (part, int(event["episode_index"])))
                traces.setdefault(key, [])
            elif event["type"] == "pose" and key is not None:
                traces[key].append((int(event["seq"]),
                                    tuple(map(float, event["pose_pred"][:2]))))
    return {key: np.asarray([p for _, p in sorted(items)], dtype=float)
            for key, items in traces.items() if items}


def _select(summary, new, old, label, minimum):
    rows = summary["best_spl"] if label == "improved" else summary["worst_spl"]
    rows = sorted(rows, key=lambda row: row["delta_spl"],
                  reverse=(label == "improved"))
    for row in rows:
        key = tuple(row["key"])
        if (key in new and key in old and len(new[key]) >= minimum
                and len(old[key]) >= minimum and
                ((row["delta_spl"] > 0) if label == "improved" else (row["delta_spl"] < 0))):
            return key, row
    raise ValueError(f"no paired {label} episode with {minimum} steps")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("gibson", "mp3d"), required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--old-trace-root", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=40)
    args = parser.parse_args()
    if not 20 <= args.steps <= 100:
        parser.error("steps must be in [20,100]")
    summary = json.loads((args.run_root / "behavior_comparison.json").read_text())
    new = _poses(args.run_root, args.dataset)
    old = _poses(args.old_trace_root, args.dataset)
    output = args.run_root / "figures"
    output.mkdir(exist_ok=True)
    selected = {}
    for label in ("improved", "worsened"):
        key, row = _select(summary, new, old, label, 20)
        a = new[key][:args.steps]
        b = old[key][:args.steps]
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(b[:, 0], b[:, 1], color="#2474b5", linewidth=1.4,
                label=f"old SNN ({len(b)} steps)")
        ax.plot(a[:, 0], a[:, 1], color="#278163", linewidth=1.4,
                label=f"candidate ({len(a)} steps)")
        ax.scatter([a[0, 0]], [a[0, 1]], marker="*", s=125,
                   c="#d18b2a", label="shared start")
        ax.scatter([a[-1, 0]], [a[-1, 1]], marker="o", s=30, c="#278163")
        ax.scatter([b[-1, 0]], [b[-1, 1]], marker="o", s=30, c="#2474b5")
        ax.set(xlabel="map x (m)", ylabel="map y (m)",
               title=f"{args.dataset} {label}: {key}, SPL difference {row['delta_spl']:+.3f}")
        ax.axis("equal")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output / f"paired_trajectory_{label}.png", dpi=160)
        plt.close(fig)
        selected[label] = {"key": list(key), "delta_spl": row["delta_spl"],
                           "candidate_steps": len(new[key]), "old_steps": len(old[key])}
    (args.run_root / "paired_trajectory_examples.json").write_text(
        json.dumps(selected, indent=2) + "\n")
    print(json.dumps(selected))


if __name__ == "__main__":
    main()
