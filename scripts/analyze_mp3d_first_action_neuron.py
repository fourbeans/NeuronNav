"""Pair first-action changes with MP3D pilot outcome flips by episode."""

import argparse
import glob
import json
from pathlib import Path

from summarize_waveopt_neuron import JOINT, _mp3d


def first_actions(root):
    out = {}
    for part in range(11):
        files = glob.glob(str(root / f"audit_val_part_{part}/*.jsonl"))
        if len(files) != 1:
            raise ValueError(f"part {part}: expected one audit")
        episode = None
        for line in open(files[0]):
            event = json.loads(line)
            if event["type"] == "episode_start":
                episode = int(event["episode_index"])
            elif event["type"] == "plan" and episode is not None:
                out.setdefault((str(part), episode), int(event["selected"]["turn_action"]))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-root", type=Path, required=True)
    parser.add_argument("--old-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    new_actions, old_actions = first_actions(args.new_root), first_actions(args.old_root)
    current, counts = _mp3d(args.new_root)
    old, _ = _mp3d(JOINT / "mp3d_seed123_best2k")
    rows = []
    for part in counts:
        ids = list(json.loads((args.new_root / f"tb_seed_100_val_part_{part}/stats.json").read_text()))
        for episode, episode_id in enumerate(ids):
            key = (part, episode)
            if key not in new_actions or key not in old_actions:
                raise ValueError(f"missing first plan for {key}")
            rows.append({
                "key": list(key), "episode_id": episode_id,
                "new_action": new_actions[key], "old_action": old_actions[key],
                "first_action_changed": new_actions[key] != old_actions[key],
                "new_success": float(current[episode_id]["success"]),
                "old_success": float(old[episode_id]["success"]),
            })
    changes = [row for row in rows if row["first_action_changed"]]
    result = {
        "episodes": len(rows),
        "first_action_changed": len(changes),
        "first_action_changed_and_new_only_success": sum(
            x["new_success"] > x["old_success"] for x in changes),
        "first_action_changed_and_old_only_success": sum(
            x["new_success"] < x["old_success"] for x in changes),
        "old_forward_to_new_turn": sum(
            x["old_action"] == 1 and x["new_action"] in (2, 3) for x in changes),
        "old_forward_to_new_turn_new_only_success": sum(
            x["old_action"] == 1 and x["new_action"] in (2, 3)
            and x["new_success"] > x["old_success"] for x in changes),
        "old_forward_to_new_turn_old_only_success": sum(
            x["old_action"] == 1 and x["new_action"] in (2, 3)
            and x["new_success"] < x["old_success"] for x in changes),
        "changed_episodes": changes,
        "caveat": "Association only: later plans also diverge, and first-action agreement does not imply equal trajectories.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "changed_episodes"}, indent=2))


if __name__ == "__main__":
    main()
