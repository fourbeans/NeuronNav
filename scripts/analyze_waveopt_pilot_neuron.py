"""Paired pilot outcomes and traceable action/no-path diagnostics.

The old MP3D audit has no episode delimiters and the old Gibson audit has no
poses.  Report unavailable comparisons explicitly; never label a pose-derived
forward stall as a simulator collision.
"""

import argparse
from collections import defaultdict
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from summarize_waveopt_neuron import JOINT, _gibson, _gibson_reference, _mp3d


SCENES = ("Collierville", "Corozal", "Darden", "Markleeville", "Wiconisco")


def _outcomes(root, dataset):
    if dataset == "gibson":
        current, count = _gibson(root, False)
        old = _gibson_reference(JOINT / "gibson_seed234_best32k", count)
        if not np.allclose(current["goal_distance"], old["goal_distance"], atol=1e-8, rtol=0):
            raise ValueError("Gibson cohort mismatch")
        return {
            (scene, episode + 1): {
                "success": float(current["success"][s * count + episode]),
                "old_success": float(old["success"][s * count + episode]),
                "spl": float(current["spl"][s * count + episode]),
                "old_spl": float(old["spl"][s * count + episode]),
            }
            for s, scene in enumerate(SCENES) for episode in range(count)
        }
    current, counts = _mp3d(root)
    old, _ = _mp3d(JOINT / "mp3d_seed123_best2k")
    if not set(current).issubset(old):
        raise ValueError("MP3D cohort mismatch")
    out = {}
    for part in counts:
        path = root / f"tb_seed_100_val_part_{part}/stats.json"
        ids = list(json.loads(path.read_text()))
        for episode, episode_id in enumerate(ids):
            out[(part, episode)] = {
                "episode_id": episode_id,
                "success": float(current[episode_id]["success"]),
                "old_success": float(old[episode_id]["success"]),
                "spl": float(current[episode_id]["spl"]),
                "old_spl": float(old[episode_id]["spl"]),
            }
    return out


def _audit(root, dataset, count=None):
    pattern = "audit/*.jsonl" if dataset == "gibson" else "audit_val_part_*/*.jsonl"
    records = defaultdict(lambda: {"turns": 0, "plans": 0, "no_path": 0,
                                   "actions": {}, "poses": {}})
    for filename in sorted(glob.glob(str(root / pattern))):
        part = Path(filename).parent.name.rsplit("_", 1)[-1]
        scene = None
        episode = None
        for line in open(filename):
            event = json.loads(line)
            typ = event["type"]
            if typ == "episode_start":
                if dataset == "gibson":
                    scene = Path(event["scene"]).stem
                    episode = int(event["episode_no"])
                else:
                    episode = int(event["episode_index"])
                continue
            if episode is None or (dataset == "gibson" and count and episode > count):
                continue
            key = (scene, episode) if dataset == "gibson" else (part, episode)
            row = records[key]
            if typ == "plan":
                row["plans"] += 1
                info = (event.get("selected") or event.get("candidate") or {}).get("info", {})
                row["no_path"] += bool(info.get("no_path_detected") or info.get("no_path"))
            elif typ == "action":
                action = int(event["action"])
                row["turns"] += action in (2, 3)
                row["actions"][int(event["seq"])] = action
            elif typ == "pose":
                row["poses"][int(event["seq"])] = tuple(map(float, event["pose_pred"][:2]))
    for row in records.values():
        ordered = sorted(row["poses"])
        row["forward_stalls_proxy"] = int(sum(
            row["actions"].get(a) == 1
            and np.linalg.norm(np.subtract(row["poses"][b], row["poses"][a])) < 0.03
            for a, b in zip(ordered[:-1], ordered[1:])
        )) if ordered else None
        row["has_pose_trace"] = bool(ordered)
        row["turn_fraction"] = row["turns"] / len(row["actions"]) if row["actions"] else None
        row["no_path_rate"] = row["no_path"] / row["plans"] if row["plans"] else None
        del row["actions"]
        del row["poses"]
    return dict(records)


def _mean(rows, field):
    values = [row[field] for row in rows if row and row.get(field) is not None]
    return float(np.mean(values)) if values else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("gibson", "mp3d"), required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--old-trace-root", type=Path,
                        help="Independent replay of the old strict SNN with episode/pose audit")
    args = parser.parse_args()
    outcomes = _outcomes(args.run_root, args.dataset)
    count = len(outcomes) // 5 if args.dataset == "gibson" else None
    candidate = _audit(args.run_root, args.dataset, count)
    old_audit_root = args.old_trace_root
    if old_audit_root is None and args.dataset == "gibson":
        old_audit_root = JOINT / "gibson_seed234_best32k"
    old = _audit(old_audit_root, args.dataset, count) if old_audit_root else {}
    if args.old_trace_root and any(key not in old for key in outcomes):
        raise ValueError("old trace is incomplete for candidate cohort")
    missing = [str(key) for key in outcomes if key not in candidate]
    if missing:
        raise ValueError(f"candidate audit lacks {len(missing)} episodes; first {missing[:3]}")
    rows = []
    for key, metrics in outcomes.items():
        row = {"key": list(key), **metrics,
               "delta_success": metrics["success"] - metrics["old_success"],
               "delta_spl": metrics["spl"] - metrics["old_spl"],
               "candidate_behavior": candidate[key],
               "old_behavior": old.get(key)}
        rows.append(row)
    rows.sort(key=lambda x: x["delta_spl"])
    summary = {
        "dataset": args.dataset,
        "episodes": len(rows),
        "success_candidate_only": sum(x["delta_success"] > 0 for x in rows),
        "success_old_only": sum(x["delta_success"] < 0 for x in rows),
        "spl_mean_delta": _mean(rows, "delta_spl"),
        "candidate_mean_turns": _mean((x["candidate_behavior"] for x in rows), "turns"),
        "old_mean_turns": _mean((x["old_behavior"] for x in rows), "turns"),
        "candidate_mean_no_path_calls": _mean((x["candidate_behavior"] for x in rows), "no_path"),
        "old_mean_no_path_calls": _mean((x["old_behavior"] for x in rows), "no_path"),
        "candidate_mean_turn_fraction": _mean((x["candidate_behavior"] for x in rows), "turn_fraction"),
        "old_mean_turn_fraction": _mean((x["old_behavior"] for x in rows), "turn_fraction"),
        "candidate_mean_no_path_rate": _mean((x["candidate_behavior"] for x in rows), "no_path_rate"),
        "old_mean_no_path_rate": _mean((x["old_behavior"] for x in rows), "no_path_rate"),
        "candidate_mean_forward_stalls_proxy": _mean(
            (x["candidate_behavior"] for x in rows), "forward_stalls_proxy"),
        "old_mean_forward_stalls_proxy": _mean(
            (x["old_behavior"] for x in rows), "forward_stalls_proxy"),
        "collision_measurement": "unavailable; forward stalls are only a proxy",
        "old_mp3d_episode_trace": (
            "replayed" if args.old_trace_root else "unavailable in prior audit"
        ) if args.dataset == "mp3d" else None,
    }
    result = {"summary": summary, "worst_spl": rows[:8], "best_spl": rows[-8:]}
    (args.run_root / "behavior_comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    figures = args.run_root / "figures"
    figures.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.4))
    axes[0].bar(("candidate only", "old only"),
                (summary["success_candidate_only"], summary["success_old_only"]),
                color=("#278163", "#ba5860"))
    axes[0].set_title("Success flips")
    axes[1].hist([x["delta_spl"] for x in rows], bins=np.linspace(-1, 1, 21),
                 color="#2474b5")
    axes[1].axvline(0, color="black", linewidth=0.7)
    axes[1].set(xlabel="candidate - old SPL", title="Paired SPL changes")
    fields = (("turns", "Turns per episode"),
              ("no_path", "No-path calls per episode"))
    for x, (field, label) in enumerate(fields):
        candidate_value = _mean((r["candidate_behavior"] for r in rows), field)
        old_value = _mean((r["old_behavior"] for r in rows), field)
        ax = axes[2 + x]
        ax.bar("candidate", candidate_value, color="#278163")
        if old_value is not None:
            ax.bar("old SNN", old_value, color="#2474b5")
        ax.set_title(label)
    fig.tight_layout()
    fig.savefig(figures / "paired_behavior.png", dpi=160)
    plt.close(fig)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
