"""Trainable two-layer LIF candidate ranker, with analog linear rate readout.

This is an independently trained student, not a weight conversion of U-Net.
Static features encode constant current. State resets on EVERY forward call.
Surrogate derivative is only used in training; forward spikes are binary.
"""
import torch
from torch import nn
from torch.nn import functional as F


class SpikeNeuron(torch.autograd.Function):
    @staticmethod
    def forward(ctx, voltage):
        ctx.save_for_backward(voltage)
        return (voltage >= 0).to(voltage.dtype)

    @staticmethod
    def backward(ctx, gradient):
        voltage, = ctx.saved_tensors
        return gradient / (1.0 + 5.0 * voltage.abs()).pow(2)


class RankSNNNeuron(nn.Module):
    def __init__(self, input_dim, hidden=64, steps=8, beta=0.9):
        super().__init__()
        if input_dim < 1 or hidden < 1 or steps < 1 or not 0 <= beta <= 1:
            raise ValueError("Invalid ranker configuration")
        self.config = dict(input_dim=input_dim, hidden=hidden, steps=steps, beta=beta)
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.readout = nn.Linear(hidden, 1, bias=False)
        nn.init.constant_(self.fc1.bias, 0.25)
        nn.init.constant_(self.fc2.bias, 0.25)

    def forward(self, features, return_activity=False):
        current = self.fc1(features)
        v1, v2 = torch.zeros_like(current), torch.zeros_like(current)
        total = torch.zeros_like(current)
        count1 = torch.zeros((), device=current.device)
        for _ in range(self.config["steps"]):
            v1 = self.config["beta"] * v1 + current
            s1 = SpikeNeuron.apply(v1 - 1.0)
            v1 = v1 - s1.detach()
            v2 = self.config["beta"] * v2 + self.fc2(s1)
            s2 = SpikeNeuron.apply(v2 - 1.0)
            v2 = v2 - s2.detach()
            total = total + s2
            count1 = count1 + s1.detach().sum()
        score = self.readout(total / self.config["steps"]).squeeze(-1)
        if return_activity:
            slots = max(1, current.numel() * self.config["steps"])
            return score, {"layer1_rate": float(count1 / slots),
                           "layer2_rate": float(total.detach().sum() / slots)}
        return score


def distillation_loss(student, teacher, temperature=0.03):
    """Listwise soft target + meaningful-gap hard-negative ordering, no pixel MSE."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    target = torch.softmax((teacher - teacher.max()) / temperature, dim=-1)
    listwise = -(target * torch.log_softmax(student, dim=-1)).sum()
    positive = teacher.argmax()
    gaps = (teacher[positive] - teacher).clamp(min=0)
    weights = gaps / (gaps.max() + 1e-8)
    pairwise = (F.softplus(0.5 - (student[positive] - student)) * weights).sum() / (weights.sum() + 1e-8)
    return listwise + 0.5 * pairwise


@torch.no_grad()
def rank_candidates(model, bank, stride=16, refine_topk=4):
    import numpy as np
    device = next(model.parameters()).device
    coarse = bank.coarse_candidates(stride)
    if not len(coarse):
        raise ValueError("No eligible candidate")
    def score(points):
        parts = []
        for offset in range(0, len(points), 2048):
            x = torch.from_numpy(bank.features(points[offset:offset+2048])).to(device)
            parts.append(model(x).cpu().numpy())
        return np.concatenate(parts)
    scores = score(coarse)
    order = np.argsort(-scores, kind="stable")
    fine = bank.refine_candidates(coarse[order[:refine_topk]], radius=stride)
    candidates = np.unique(np.concatenate([coarse, fine]), axis=0)
    scores = score(candidates)
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite student scores")
    order = np.argsort(-scores, kind="stable")
    best = order[0]
    margin = float(scores[best] - scores[order[1]]) if len(order) > 1 else 0.0
    return tuple(map(int, candidates[best])), {"coarse_count": len(coarse), "candidate_count": len(candidates),
                                              "score_margin": margin}, candidates, scores
