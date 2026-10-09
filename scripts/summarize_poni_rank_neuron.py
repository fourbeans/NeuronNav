"""Only summarize completed, paired PONI rank smoke batches; no hidden fallback."""
import argparse
import json
from pathlib import Path

import numpy as np


def summarize(root):
    root = Path(root)
    if not (root / "run_status.json").is_file():
        raise ValueError("Batch not complete: " + str(root))
    statuses = json.loads((root / "run_status.json").read_text())
    if any(s["returncode"] != 0 for s in statuses):
        raise ValueError("Batch contains failures")
    results = []
    baseline_actions, baseline_episodes = {}, {}
    for mode in ("reference", "guarded", "neuron"):
        folder = root / mode
        metrics, decisions, actions, starts = {}, [], {}, {}
        for path in sorted(folder.glob("**/final_eval_stats.json")):
            data = json.loads(path.read_text())["total_raw_metrics"]
            for key, values in data.items():
                metrics.setdefault(key, []).extend(values)
        if not metrics:
            raise ValueError("Missing results for " + mode)
        for path in sorted(folder.glob("*/rank_audit/rank_*.jsonl")):
            decisions.extend(x for x in map(json.loads, path.read_text().splitlines()) if x["type"] == "decision")
        for path in sorted(folder.glob("*/rank_audit/low_level_fmm/*.jsonl")):
            for row in map(json.loads, path.read_text().splitlines()):
                if row["type"] == "action":
                    actions[(row["scene"], row["episode_no"], row["step"])] = row["action"]
                elif row["type"] == "episode_start":
                    starts[(row["scene"], row["episode_no"])] = (row["position"], row["rotation"], row["goal_ids"])
        if not decisions or not actions or not starts:
            raise ValueError("Missing decision, action or episode audit for " + mode)
        if mode == "reference":
            baseline_actions, baseline_episodes = actions, starts
        common = set(actions) & set(baseline_actions)
        common_episodes = set(starts) & set(baseline_episodes)
        reasons = {}
        for row in decisions:
            if row.get("reason"):
                reasons[row["reason"]] = reasons.get(row["reason"], 0) + 1
        results.append({"mode": mode, "raw_metrics": metrics, "metrics": {k: float(np.mean(v)) for k,v in metrics.items()},
                        "decision_calls": len(decisions), "teacher_calls": sum(d["teacher_called"] for d in decisions),
                        "student_selected": sum(d["used"] == "neuron" for d in decisions),
                        "fallbacks": sum(d["fallback"] for d in decisions), "fallback_reasons": reasons,
                        "paired_candidate_goals": sum("goal_equal" in d for d in decisions),
                        "candidate_goal_equal": sum(d.get("goal_equal", False) for d in decisions),
                        "same_action_keys": set(actions) == set(baseline_actions),
                        "shared_actions": len(common),
                        "action_mismatches": sum(actions[k] != baseline_actions[k] for k in common),
                        "same_episode_keys": set(starts) == set(baseline_episodes),
                        "episode_start_mismatches": sum(starts[k] != baseline_episodes[k] for k in common_episodes)})
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = summarize(args.run)
    with Path(args.output).open("x") as fp:
        json.dump(result, fp, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
