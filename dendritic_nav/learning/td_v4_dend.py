"""Recurrent policy-conditioned TD loss for the v4 dendritic scorer.

The spatial potential selects a policy action but is not a Bellman value.
Only the learned value plus dendritic candidate advantage is regressed to
discounted navigation return. Complete episode prefixes reconstruct state.
"""
import torch
from torch.nn import functional as F
from dendritic_nav.core.cell_dend import CellState
from .replay_dend import return_target


def training_loss(online, target, samples, device, gamma=0.995):
    batch = len(samples)
    longest = max(len(e["reward"]) for e, _, _ in samples)
    widths = [stop - start for _, start, stop in samples]
    starts = torch.tensor([start for _, start, _ in samples], device=device)

    def padded(key, dtype):
        shape = (batch, longest) + samples[0][0][key].shape[1:]
        result = torch.zeros(shape, device=device, dtype=dtype)
        for b, (episode, _, _) in enumerate(samples):
            result[b, : len(episode[key])] = torch.as_tensor(
                episode[key], device=device, dtype=dtype
            )
        return result

    features = padded("features", torch.float32)
    candidates = padded("candidates", torch.float32)
    spatial = padded("spatial", torch.float32)
    valid = padded("valid", torch.bool)
    lengths = torch.tensor([len(e["reward"]) for e, _, _ in samples], device=device)
    state = online.cell.initial(batch, device)
    target_state = target.cell.initial(batch, device)
    prefix = online.cell.initial(batch, device)
    boot_values = []
    with torch.no_grad():
        for t in range(longest):
            prefix = CellState(
                *(
                    torch.where(
                        (starts == t).view(batch, *([1] * (x.ndim - 1))), x, p
                    )
                    for x, p in zip(state, prefix)
                )
            )
            advance = t < lengths
            state, _ = online.cell(features[:, t], state, advance=advance)
            target_state, _ = target.cell(
                features[:, t], target_state, advance=advance
            )
            policy, _, _ = online.evaluate(
                candidates[:, t], spatial[:, t], valid[:, t], state,
                features[:, t]
            )
            _, _, target_q = target.evaluate(
                candidates[:, t], spatial[:, t], valid[:, t], target_state,
                features[:, t]
            )
            boot_values.append(
                target_q.gather(1, policy.argmax(-1, keepdim=True)).squeeze(1)
            )

    rows = torch.arange(batch, device=device)
    state = prefix.detach()
    losses = []
    estimates = []
    targets = []
    for offset in range(max(widths)):
        times = (starts + offset).clamp(max=longest - 1)
        active = torch.tensor([offset < width for width in widths], device=device)
        state, _ = online.cell(features[rows, times], state, advance=active)
        _, _, critic = online.evaluate(
            candidates[rows, times], spatial[rows, times],
            valid[rows, times], state, features[rows, times]
        )
        for b, (episode, start, stop) in enumerate(samples):
            t = start + offset
            if t >= stop or episode["action"][t] < 0:
                continue
            reward_sum, discount, next_t = return_target(episode, t, gamma)
            target_value = critic.new_tensor(reward_sum)
            if next_t is not None:
                target_value = target_value + discount * boot_values[next_t][b]
            estimate = critic[b, int(episode["action"][t])]
            losses.append(F.smooth_l1_loss(estimate, target_value))
            estimates.append(estimate.detach())
            targets.append(target_value)
    if not losses:
        return None, dict(decisions=0)
    loss = torch.stack(losses).mean()
    return loss, dict(
        decisions=len(losses),
        loss=float(loss.detach()),
        critic_mean=float(torch.stack(estimates).mean()),
        target_mean=float(torch.stack(targets).mean()),
    )
