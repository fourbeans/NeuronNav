"""Audit final weights and representative replay without changing parameters."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from dendritic_nav.core.cell_dend import DendriticScorer, spike


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.experiment)
    torch.set_num_threads(1)
    saved = torch.load(str(root / "learner/checkpoints/pilot_final.ckpt"), map_location="cpu")
    initial = torch.load(str(root / "checkpoints/initial.ckpt"), map_location="cpu")
    target_changes = {k: float((v-initial["state_dict"][k]).abs().max())
                      for k,v in saved["target_state_dict"].items()}
    model = DendriticScorer().eval()
    model.load_state_dict(saved["state_dict"])
    values = {k:[] for k in ("residual_span", "static_span", "interaction_span", "spatial_loss",
                             "selected_distance", "anchor_distance", "candidate_zero_fraction",
                             "state_abs_max", "semantic_event_rate", "selected_anchor_separation",
                             "anchor_residual_minus_other_median")}
    counters = dict(decisions=0, changed_from_anchor=0, memory_changes_argmax=0,
                    reward_targets_bootstrap=0, reward_targets_terminal=0,
                    selection_moves_over_05m=0, role_marker_removal_restores_anchor=0)
    from dendritic_nav.learning.replay_dend import return_target
    sample_files = sorted(root.glob("actor_*/replay/episode_0000000.npz"))
    with torch.no_grad():
        for path in sample_files:
            with np.load(path,allow_pickle=False) as archive:
                episode = {key:archive[key] for key in archive.files}
                state = None
                for t in range(len(episode["reward"])):
                    state,events = model.cell(torch.tensor(episode["features"][t:t+1]),state)
                    if episode["action"][t] < 0:
                        continue
                    valid = torch.tensor(episode["valid"][t])
                    c = torch.tensor(episode["candidates"][t:t+1])
                    spatial = torch.tensor(episode["spatial"][t])
                    embed = spike(model.encoder(c))[0]
                    static = embed @ model.head.weight[0,:128]
                    interact = (embed*state.readout[0]) @ model.head.weight[0,256:]
                    # The pure-history term is identical for every candidate:
                    # it changes Q calibration but cancels from the argmax.
                    residual = static+interact
                    q = (spatial+residual).masked_fill(~valid,-float("inf"))
                    index = int(q.argmax())
                    without_interaction = int((spatial+static).masked_fill(~valid,-float("inf")).argmax())
                    counters["decisions"] += 1
                    counters["changed_from_anchor"] += int(index != 0)
                    counters["memory_changes_argmax"] += int(index != without_interaction)
                    separation = float(np.linalg.norm(episode["rc"][t,index]-episode["rc"][t,0])*.05)
                    values["selected_anchor_separation"].append(separation)
                    counters["selection_moves_over_05m"] += int(separation>.5)
                    others = residual[valid][1:]
                    values["anchor_residual_minus_other_median"].append(
                        float(residual[0]-others.median()) if len(others) else 0.)
                    without_role_markers = c.clone()
                    without_role_markers[0,0,10] = 0
                    without_role_markers[0,0,17] = 0
                    ce = spike(model.encoder(without_role_markers))[0]
                    cr = ce @ model.head.weight[0,:128] + (ce*state.readout[0]) @ model.head.weight[0,256:]
                    ci = int((spatial+cr).masked_fill(~valid,-float("inf")).argmax())
                    counters["role_marker_removal_restores_anchor"] += int(index!=0 and ci==0)
                    for key, data in (("residual_span",residual),("static_span",static),("interaction_span",interact)):
                        values[key].append(float(data[valid].max()-data[valid].min()))
                    values["spatial_loss"].append(float(spatial[0]-spatial[index]))
                    values["selected_distance"].append(float(c[0,index,5])*20)
                    values["anchor_distance"].append(float(c[0,0,5])*20)
                    values["candidate_zero_fraction"].append(float((embed[valid]==0).float().mean()))
                    values["state_abs_max"].append(max(float(x.abs().max()) for x in state))
                    values["semantic_event_rate"].append(float((events[0,0]!=0).float().mean()))
                    _,_,nxt = return_target(episode,t)
                    counters["reward_targets_terminal" if nxt is None else "reward_targets_bootstrap"] += 1
    summary = {key:dict(mean=float(np.mean(v)),p50=float(np.median(v)),p95=float(np.percentile(v,95)),max=float(np.max(v)))
               for key,v in values.items()}
    result = dict(checkpoint_updates=saved["updates"], target_max_change_from_initial=max(target_changes.values()),
                  sampled_episodes=len(sample_files),counters=counters,statistics=summary,
                  caveat="Training-replay states, not closed-loop evaluation. Decomposition is a diagnostic, not an ablation experiment.")
    output = Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("x") as stream:json.dump(result,stream,indent=2)
    print(json.dumps(result,indent=2))


if __name__ == "__main__":main()
