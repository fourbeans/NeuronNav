"""Post-smoke numerical audit of actual replay and learned parameter changes."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from dendritic_nav.core.cell_dend import DendriticScorer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    args = parser.parse_args()
    root = Path(args.experiment)
    initial = torch.load(str(root / "checkpoints/initial.ckpt"), map_location="cpu")
    trained = torch.load(
        str(root / "learner/checkpoints/pilot_final.ckpt"), map_location="cpu"
    )
    changes = {}
    for key, value in trained["state_dict"].items():
        changes[key] = float((value - initial["state_dict"][key]).abs().max())
    model = DendriticScorer()
    model.load_state_dict(trained["state_dict"])
    files = sorted(root.glob("actor_*/replay/episode_*.npz"))
    total, decisions, active, nonzero_residual, differing = 0, 0, 0.0, 0, 0
    torch.set_num_threads(1)
    with torch.no_grad():
        for path in files:
            with np.load(path, allow_pickle=False) as episode:
                state = None
                for t in range(len(episode["reward"])):
                    state, events = model.cell(
                        torch.tensor(episode["features"][t : t + 1]), state
                    )
                    q, residual = model.score(
                        torch.tensor(episode["candidates"][t : t + 1]),
                        torch.tensor(episode["spatial"][t : t + 1]),
                        torch.tensor(episode["valid"][t : t + 1]),
                        state,
                        torch.tensor(episode["features"][t : t + 1]),
                    )
                    total += 1
                    active += float((events != 0).float().mean())
                    nonzero_residual += int(bool((residual != 0).any()))
                    if episode["action"][t] >= 0:
                        decisions += 1
                        differing += int(int(q.argmax()) != 0)
    result = dict(
        environment_steps=total,
        decisions=decisions,
        updates=int(trained["updates"]),
        parameter_max_abs_changes=changes,
        mean_dendrite_active_fraction=active / max(total, 1),
        steps_with_nonzero_residual=nonzero_residual,
        replay_greedy_goal_changes=differing,
        note="Training/inference connectivity audit, NOT a navigation improvement metric",
    )
    if not changes["interaction_head.weight"] or not any(
        v for k, v in changes.items() if k.startswith("cell.projections")
    ):
        raise RuntimeError("Head or dendritic input parameters did not learn")
    with open(root / "learning_audit.json", "x") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
