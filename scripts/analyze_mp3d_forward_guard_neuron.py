"""Audit strict SNN compliance and actual use of the MP3D forward guard."""

import argparse
import glob
import json
from pathlib import Path

from summarize_waveopt_neuron import JOINT, _mp3d


def actions_by_episode(root):
    actions = {}
    for part in range(11):
        files = glob.glob(str(root / f"audit_val_part_{part}/*.jsonl"))
        if len(files) != 1:
            raise ValueError(f"part {part}: expected one audit file")
        episode = None
        with open(files[0]) as stream:
            for line in stream:
                event = json.loads(line)
                if event.get("type") == "episode_start":
                    episode = int(event["episode_index"])
                    actions[(str(part), episode)] = []
                elif event.get("type") == "action" and episode is not None:
                    actions[(str(part), episode)].append(int(event["action"]))
    return actions


def analyze_divergence(new_root, old_root):
    new_actions = actions_by_episode(new_root)
    old_actions = actions_by_episode(old_root)
    new_stats, counts = _mp3d(new_root)
    old_stats, _ = _mp3d(JOINT / "mp3d_seed123_best2k")
    rows = []
    for part in counts:
        ids = list(json.loads((new_root / f"tb_seed_100_val_part_{part}/stats.json").read_text()))
        for episode, episode_id in enumerate(ids):
            key = (part, episode)
            left, right = new_actions[key], old_actions[key]
            first = next((i for i, (a, b) in enumerate(zip(left, right)) if a != b), None)
            if first is None and len(left) != len(right):
                first = min(len(left), len(right))
            rows.append({
                "part": part, "episode_index": episode, "episode_id": episode_id,
                "first_action_divergence_step": first,
                "new_actions": len(left), "old_actions": len(right),
                "new_success": float(new_stats[episode_id]["success"]),
                "old_success": float(old_stats[episode_id]["success"]),
                "new_spl": float(new_stats[episode_id]["spl"]),
                "old_spl": float(old_stats[episode_id]["spl"]),
            })
    changed = [r for r in rows if r["first_action_divergence_step"] is not None]
    return {
        "episodes": len(rows), "any_action_divergence": len(changed),
        "first_action_divergence": sum(r["first_action_divergence_step"] == 0 for r in rows),
        "divergence_within_10_steps": sum(
            r["first_action_divergence_step"] is not None
            and r["first_action_divergence_step"] < 10 for r in rows),
        "new_only_success_after_divergence": sum(
            r["new_success"] > r["old_success"] for r in changed),
        "old_only_success_after_divergence": sum(
            r["new_success"] < r["old_success"] for r in changed),
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("guard", "gain"), default="guard")
    parser.add_argument("--old-root", type=Path,
                        help="Old strict SNN pose/action trace for paired divergence")
    args = parser.parse_args()
    files = glob.glob(str(args.run_root / "audit_val_part_*/*.jsonl"))
    if len(files) != 11:
        parser.error(f"expected 11 audit files, got {len(files)}")
    result = {
        "plan_calls": 0, "plan_failures": 0, "invalid_strict_calls": 0,
        "no_path_calls": 0, "reachable_tie_calls": 0,
        "forward_guarded_calls": 0, "forward_gain_accepted_calls": 0,
        "selected_changed_stg_calls": 0,
        "max_abs_tie_cost_delta": 0.0,
    }
    for filename in files:
        with open(filename) as stream:
            for line in stream:
                event = json.loads(line)
                if event.get("type") == "plan_failure":
                    result["plan_failures"] += 1
                    continue
                if event.get("type") != "plan":
                    continue
                result["plan_calls"] += 1
                result["invalid_strict_calls"] += int(
                    event.get("used") != "neuron" or event.get("fallback")
                    or event.get("reference") is not None
                    or event.get("error") is not None)
                selected = event["selected"]
                info = selected["info"]
                result["no_path_calls"] += int(bool(info.get("no_path")))
                expected = ("equal_arrival_forward_guard" if args.mode == "guard"
                            else "equal_arrival_forward_gain_only")
                if info.get("readout_mode") != expected:
                    continue
                result["reachable_tie_calls"] += 1
                result["forward_guarded_calls"] += int(info["tie_forward_guarded"])
                result["forward_gain_accepted_calls"] += int(
                    info["tie_forward_gain_accepted"])
                result["selected_changed_stg_calls"] += int(
                    tuple(selected["stg"]) != tuple(info["tie_old_stg"]))
                result["max_abs_tie_cost_delta"] = max(
                    result["max_abs_tie_cost_delta"], abs(info["tie_cost_delta"]))
    if result["plan_failures"] or result["invalid_strict_calls"]:
        raise RuntimeError("strict SNN audit failed")
    (args.run_root / "forward_guard_usage.json").write_text(
        json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if args.old_root:
        divergence = analyze_divergence(args.run_root, args.old_root)
        (args.run_root / "action_divergence.json").write_text(
            json.dumps(divergence, indent=2) + "\n")
        print(json.dumps({k: v for k, v in divergence.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
