"""Quantify repeated Gibson SNN reachable-goal projections from completed audits."""

import argparse
from collections import Counter, defaultdict
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def analyze(audit_glob):
    episodes = defaultdict(list)
    total_plans = 0
    repaired = 0
    for name in sorted(glob.glob(audit_glob)):
        pending = {}
        for line in open(name):
            event = json.loads(line)
            if event["type"] == "plan":
                total_plans += 1
                info = event["candidate"]["info"]
                pending[int(event["seq"])] = info
            elif event["type"] == "action":
                info = pending.pop(int(event["seq"]), None)
                if info and info.get("no_path_repaired"):
                    repaired += 1
                    key = (str(event["scene"]), int(event["episode_no"]))
                    episodes[key].append(tuple(info["proxy_goal"]))
    rows = []
    for (scene, episode), proxies in episodes.items():
        counts = Counter(proxies)
        rows.append({
            "scene": scene, "episode_no": episode,
            "repaired_calls": len(proxies),
            "unique_proxies": len(counts),
            "most_used_proxy_calls": max(counts.values()),
            "most_used_proxy_fraction": max(counts.values()) / len(proxies),
            "adjacent_equal_fraction": sum(a == b for a, b in zip(proxies, proxies[1:]))
                                      / max(1, len(proxies) - 1),
        })
    rows.sort(key=lambda row: (-row["repaired_calls"], row["scene"], row["episode_no"]))
    return {"plans": total_plans, "repaired_calls": repaired,
            "episodes_with_repair": len(rows), "episodes": rows}


def plot(result, destination):
    rows = result["episodes"][:20]
    if not rows:
        return
    labels = [f"{r['scene'].split('/')[-1]}:{r['episode_no']}" for r in rows]
    counts = [r["repaired_calls"] for r in rows]
    dominance = [r["most_used_proxy_fraction"] for r in rows]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    ax1.bar(range(len(rows)), counts, color="#2474b5")
    ax1.set(ylabel="repaired planning calls", title="Gibson: episodes with repeated goal projection")
    ax2.bar(range(len(rows)), dominance, color="#b5812b")
    ax2.set(ylabel="top proxy share", ylim=(0, 1))
    ax2.set_xticks(range(len(rows)), labels, rotation=65, ha="right")
    fig.tight_layout()
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def paired_outcomes(result, snn_stats, reference_stats, output):
    snn = json.loads(Path(snn_stats).read_text())["total_raw_metrics"]
    reference = json.loads(Path(reference_stats).read_text())["total_raw_metrics"]
    scenes = sorted({Path(row["scene"]).stem for row in result["episodes"]})
    if len(scenes) != 5 or len(snn["success"]) != 1000 or len(reference["success"]) != 1000:
        raise ValueError("expected five Gibson scenes and 1000 episodes")
    flagged = {(Path(row["scene"]).stem, row["episode_no"]) for row in result["episodes"]}
    groups = {}
    for label, selected in (("repair", True), ("no_repair", False)):
        indices = [scene_index * 200 + episode - 1
                   for scene_index, scene in enumerate(scenes)
                   for episode in range(1, 201)
                   if ((scene, episode) in flagged) == selected]
        groups[label] = {
            "episodes": len(indices),
            "snn": {metric: sum(snn[metric][i] for i in indices) / len(indices)
                    for metric in ("success", "spl")},
            "reference": {metric: sum(reference[metric][i] for i in indices) / len(indices)
                          for metric in ("success", "spl")},
        }
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    for ax, metric in zip(axes, ("success", "spl")):
        for offset, method, color in ((-0.18, "snn", "#2474b5"),
                                      (0.18, "reference", "#b5812b")):
            values = [groups[group][method][metric] for group in ("repair", "no_repair")]
            ax.bar([0 + offset, 1 + offset], values, width=0.34, color=color,
                   label="strict SNN" if method == "snn" else "FMM")
        ax.set_xticks([0, 1], ["105 repair", "895 no repair"])
        ax.set_ylim(0, 1)
        ax.set_title(metric.upper() if metric == "spl" else "Success")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-glob", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snn-stats", type=Path)
    parser.add_argument("--reference-stats", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    result = analyze(args.audit_glob)
    if bool(args.snn_stats) != bool(args.reference_stats):
        parser.error("supply both stats paths or neither")
    if args.snn_stats:
        result["paired_outcomes"] = paired_outcomes(
            result, args.snn_stats, args.reference_stats,
            args.output / "gibson_nopath_outcomes.png",
        )
    (args.output / "gibson_nopath_repetition.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    plot(result, args.output / "gibson_nopath_repetition.png")
    print(json.dumps({k: result[k] for k in ("plans", "repaired_calls", "episodes_with_repair")}))


if __name__ == "__main__":
    main()
