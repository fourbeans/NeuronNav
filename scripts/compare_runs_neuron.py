"""Summarize instrumented ObjectNav runs with explicit fallback and pairing."""
import argparse
import glob
import json
from pathlib import Path

import numpy as np


def summarize(root):
    root = Path(root)
    for command in root.glob("**/command.json"):
        status_path = command.parent / "run_status.json"
        if not status_path.is_file() or json.loads(status_path.read_text())["returncode"] != 0:
            raise ValueError("Incomplete or failed evaluation: " + str(command.parent))
    stats_paths = sorted(root.glob("**/final_eval_stats.json"))
    if not stats_paths:
        raise ValueError("No completed Gibson stats file in " + str(root))
    parts = [json.loads(p.read_text()) for p in stats_paths]
    raw = {key: [v for p in parts for v in p["total_raw_metrics"][key]] for key in parts[0]["total_raw_metrics"]}
    stats = {"total_raw_metrics": raw, "total_metrics": {k: float(np.mean(v)) for k, v in raw.items()}}
    plans, actions, episodes = [], [], []
    for path in sorted(root.glob("**/audit/*.jsonl")):
        for line in path.read_text().splitlines():
            item = json.loads(line)
            if item["type"] == "plan":
                plans.append(item)
            elif item["type"] == "action":
                actions.append(item)
            elif item["type"] == "episode_start":
                episodes.append(item)
    reasons = {}
    for item in plans:
        if item["reason"]:
            reasons[item["reason"]] = reasons.get(item["reason"], 0) + 1
    candidates = [p for p in plans if p["candidate"] is not None]
    paired = [p for p in candidates if p["reference"] is not None]
    return {"root": str(root), "metrics": stats["total_metrics"],
            "episodes_evaluated": len(stats["total_raw_metrics"]["success"]),
            "raw": stats["total_raw_metrics"],
            "planning_calls": len(plans), "candidate_calls": len(candidates),
            "neuron_selected": sum(p["used"] == "neuron" for p in plans),
            "fallbacks": sum(p["fallback"] for p in plans), "fallback_reasons": reasons,
            "candidate_errors": sum(p["error"] is not None for p in plans),
            "paired_action_equal": sum(p["candidate"]["turn_action"] == p["reference"]["turn_action"] for p in paired),
            "paired_stop_equal": sum(p["candidate"]["stop"] == p["reference"]["stop"] for p in paired),
            "paired_calls": len(paired),
            "mean_candidate_ms": float(np.mean([p["candidate"]["info"]["elapsed_ms"] for p in candidates])) if candidates else None,
            "actions": actions, "episode_starts": episodes}


def action_key(item):
    return (item["scene"], item["episode_no"], item["step"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    parents = {Path(root).resolve().parent for root in args.runs}
    if len(parents) == 1 and not (next(iter(parents)) / "run_status.json").is_file():
        raise ValueError("Batch is not marked complete; wait for its run_status.json before comparing")
    runs = [summarize(root) for root in args.runs]
    base = runs[0]
    base_actions = {action_key(a): a["action"] for a in base["actions"]}
    def episodes_by_key(run):
        return {(e["scene"], e["episode_no"]): (e["position"], e["rotation"], e["goal_ids"])
                for e in run["episode_starts"]}
    base_episodes = episodes_by_key(base)
    for run in runs:
        actions = {action_key(a): a["action"] for a in run["actions"]}
        shared = set(actions) & set(base_actions)
        run["action_trace_comparison"] = {"same_keys": set(actions) == set(base_actions),
                                           "shared": len(shared),
                                           "mismatches": sum(actions[k] != base_actions[k] for k in shared)}
        eps = episodes_by_key(run)
        common = set(base_episodes) & set(eps)
        run["episode_start_pairing"] = {"same_keys": set(eps) == set(base_episodes),
                                        "shared": len(common),
                                        "mismatches": sum(eps[k] != base_episodes[k] for k in common)}
        run["delta_metrics"] = {k: run["metrics"][k] - base["metrics"][k]
                                for k in ("success", "spl", "dts")}
        del run["actions"], run["episode_starts"]
    with Path(args.output).open("x") as fp:
        json.dump(runs, fp, indent=2)
    print(json.dumps([{k: v for k, v in r.items() if k != "raw"} for r in runs], indent=2))


if __name__ == "__main__":
    main()
