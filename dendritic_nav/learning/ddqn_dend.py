"""Recurrent Double-DQN with exact current-weight episode-prefix reconstruction."""
import torch
from torch.nn import functional as F
from dendritic_nav.core.cell_dend import CellState
from .replay_dend import return_target


def training_loss(online, target, samples, device, gamma=0.995):
    batch = len(samples)
    longest = max(len(e["reward"]) for e, _, _ in samples)
    widths = [stop - start for _, start, stop in samples]
    starts = torch.tensor([start for _, start, _ in samples], device=device)
    # Episode data only: no observations, GT distances or cached hidden state.
    def padded(key, dtype):
        shape = (batch, longest) + samples[0][0][key].shape[1:]
        result = torch.zeros(shape, device=device, dtype=dtype)
        for b, (e, _, _) in enumerate(samples):
            result[b, : len(e[key])] = torch.as_tensor(
                e[key], device=device, dtype=dtype
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
            # Save pre-observation state exactly at each sampled start.
            prefix = CellState(
                *(
                    torch.where((starts == t).view(batch, *([1] * (x.ndim - 1))), x, p)
                    for x, p in zip(state, prefix)
                )
            )
            advance = t < lengths
            state, _ = online.cell(features[:, t], state, advance=advance)
            target_state, _ = target.cell(features[:, t], target_state, advance=advance)
            oq, _ = online.score(candidates[:, t], spatial[:, t], valid[:, t], state,
                                 features[:, t])
            tq, _ = target.score(
                candidates[:, t], spatial[:, t], valid[:, t], target_state,
                features[:, t]
            )
            boot_values.append(tq.gather(1, oq.argmax(-1, keepdim=True)).squeeze(1))
    rows = torch.arange(batch, device=device)
    state = prefix.detach()
    losses = []
    q_values = []
    targets = []
    for offset in range(max(widths)):
        times = (starts + offset).clamp(max=longest - 1)
        active = torch.tensor([offset < w for w in widths], device=device)
        state, _ = online.cell(features[rows, times], state, advance=active)
        q, _ = online.score(
            candidates[rows, times], spatial[rows, times], valid[rows, times], state,
            features[rows, times]
        )
        for b, (episode, start, stop) in enumerate(samples):
            t = start + offset
            if t >= stop or episode["action"][t] < 0:
                continue
            reward_sum, discount, next_t = return_target(episode, t, gamma)
            y = q.new_tensor(reward_sum)
            if next_t is not None:
                y = y + discount * boot_values[next_t][b]
            estimate = q[b, int(episode["action"][t])]
            losses.append(F.smooth_l1_loss(estimate, y))
            q_values.append(estimate.detach())
            targets.append(y)
    if not losses:
        return None, dict(decisions=0)
    loss = torch.stack(losses).mean()
    return loss, dict(
        decisions=len(losses),
        loss=float(loss.detach()),
        q_mean=float(torch.stack(q_values).mean()),
        target_mean=float(torch.stack(targets).mean()),
    )
