"""Fixed old-SNN repair cohort comparison for Gibson waypoint pilot/full runs."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PONI_ROOT = Path(__file__).resolve().parents[1]
DAIRY = PONI_ROOT / "dairy"
SCENES = ("Collierville", "Corozal", "Darden", "Markleeville", "Wiconisco")


def _raw(path, count):
    source = json.loads(Path(path).read_text())["total_raw_metrics"]
    per_scene = len(source["success"]) // 5
    return {key: np.asarray([source[key][s * per_scene + e]
                             for s in range(5) for e in range(count)], dtype=float)
            for key in ("success", "spl", "goal_distance")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    current_path = args.run_root / "gibson_objectnav/dump/seed_100/final_eval_stats.json"
    current_full = json.loads(current_path.read_text())["total_raw_metrics"]
    count = len(current_full["success"]) // 5
    if count * 5 != len(current_full["success"]) or not 1 <= count <= 200:
        raise ValueError("invalid Gibson scene count")
    current = _raw(current_path, count)
    snn = _raw(DAIRY / "joint_sigma_d_strict_neuron_20261004/gibson_seed234_best32k/gibson_objectnav/dump/seed_100/final_eval_stats.json", count)
    fmm = _raw(DAIRY / "sigma_d_unet_20260924/navigation/gibson_seed234_best32k/gibson_objectnav/dump/seed_100/final_eval_stats.json", count)
    if not np.array_equal(current["goal_distance"], snn["goal_distance"]) or not np.array_equal(current["goal_distance"], fmm["goal_distance"]):
        raise ValueError("Gibson episode cohort/order mismatch")
    diagnosis = json.loads((DAIRY / "snn_wavefront_opt_20261004/diagnostics/gibson_nopath_repetition.json").read_text())
    repair_keys = {(Path(x["scene"]).stem, x["episode_no"]) for x in diagnosis["episodes"]}
    repaired = np.asarray([(scene, ep) in repair_keys
                           for scene in SCENES for ep in range(1, count + 1)])
    result = {"episodes": len(repaired), "per_scene": count, "groups": {}}
    for label, mask in (("old_snn_repair", repaired), ("old_snn_no_repair", ~repaired)):
        indices = np.where(mask)[0]
        result["groups"][label] = {
            "episodes": len(indices),
        }
        for name, arr in (("candidate", current), ("old_snn", snn), ("fmm", fmm)):
            result["groups"][label][name] = {
                metric: float(arr[metric][indices].mean()) if len(indices) else None
                for metric in ("success", "spl")
            }
    figures = args.run_root / "figures"
    figures.mkdir(exist_ok=True)
    (args.run_root / "subgroup_comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    labels = ("old SNN repaired", "old SNN not repaired")
    for ax, metric in zip(axes, ("success", "spl")):
        for offset, name, color in ((-0.24, "candidate", "#278163"),
                                    (0, "old_snn", "#2474b5"),
                                    (0.24, "fmm", "#b5812b")):
            values = [result["groups"][group][name][metric]
                      for group in ("old_snn_repair", "old_snn_no_repair")]
            ax.bar([offset, 1 + offset], values, width=0.23, color=color, label=name)
        ax.set_xticks((0, 1), labels)
        ax.set_ylim(0, 1)
        ax.set_title(metric.upper() if metric == "spl" else "Success")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "old_repair_cohort_comparison.png", dpi=160)
    plt.close(fig)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
