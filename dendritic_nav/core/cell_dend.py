"""Four-branch, four-timescale signed-spike cell; one call per observation.

Float training reference, NOT a sparse CUDA kernel or a chip implementation.
Runtime state is explicit and never included in a model state_dict.
"""
from typing import NamedTuple
import math
import torch
from torch import nn
from torch.nn import functional as F
from .ranking_v4_dend import locality_gain

MODEL_FAMILY = "sigma_d_dendritic_v4"


def spike(x):
    q = x.round().clamp(-7, 7)
    # Unit straight-through derivative, including the zero-event region.
    return x + (q - x).detach()


class SparseLinear(nn.Module):
    """Fixed fan-in connectivity. Dense masked reference implementation."""

    def __init__(self, inputs, outputs, fan_in=16):
        super().__init__()
        fan_in = min(inputs, fan_in)
        self.weight = nn.Parameter(torch.empty(outputs, inputs))
        nn.init.uniform_(self.weight, -1 / math.sqrt(fan_in), 1 / math.sqrt(fan_in))
        mask = torch.zeros(outputs, inputs)
        for row in mask:
            row[torch.randperm(inputs)[:fan_in]] = 1
        self.register_buffer("mask", mask)

    def forward(self, x):
        return F.linear(x, self.weight * self.mask)


class CellState(NamedTuple):
    dendrite: torch.Tensor
    soma: torch.Tensor
    readout: torch.Tensor

    def detach(self):
        return CellState(*(x.detach() for x in self))


class DendriticCell(nn.Module):
    def __init__(self, branch_dim=32, width=128):
        super().__init__()
        self.width = width
        self.projections = nn.ModuleList(
            [SparseLinear(branch_dim, width) for _ in range(4)]
        )
        self.gain = nn.Parameter(torch.ones(4, 4, width))
        self.mix = nn.Parameter(torch.full((4, 4, width), 0.25))
        self.register_buffer("leaks", 1 - 2.0 ** -torch.tensor([2.0, 4.0, 6.0, 8.0]))

    def initial(self, batch, device):
        return CellState(
            torch.zeros(batch, 4, 4, self.width, device=device),
            torch.zeros(batch, self.width, device=device),
            torch.zeros(batch, self.width, device=device),
        )

    def forward(self, features, state=None, reset=None, advance=None):
        if features.shape[1:] != (4, 32):
            raise ValueError("features must have shape [B,4,32]")
        if state is None:
            state = self.initial(len(features), features.device)
        if reset is not None:
            state = CellState(
                *(x * (~reset.bool()).view(-1, *([1] * (x.ndim - 1))) for x in state)
            )
        inputs = torch.stack(
            [p(features[:, i]) for i, p in enumerate(self.projections)], 1
        )
        pre = (
            state.dendrite * self.leaks[None, None, :, None]
            + inputs[:, :, None] * self.gain
        )
        events = spike(pre)
        soma_pre = state.soma * (1 - 2 ** -5) + (events * self.mix).sum((1, 2))
        soma_events = spike(soma_pre)
        readout = state.readout * (1 - 2 ** -5) + soma_events / 32
        # Detach subtractive reset in backward so the unit STE does not
        # cancel temporal gradients through every residual state.
        new = CellState(pre - events.detach(), soma_pre - soma_events.detach(), readout)
        if advance is not None:
            new = CellState(
                *(
                    torch.where(advance.bool().view(-1, *([1] * (x.ndim - 1))), x, old)
                    for x, old in zip(new, state)
                )
            )
        return new, events


class DendriticScorer(nn.Module):
    """A local history-dependent ranking correction and Bellman critic.

    The policy uses the spatial score as a prior, but the critic never treats
    that uncalibrated potential as an action value. Both share one dendritic
    candidate advantage; only the policy applies locality attenuation.
    """

    MAX_CORRECTION = 0.03

    @staticmethod
    def evidence_gate(features):
        """Only learned temporal evidence can activate candidate reranking.

        Entry count distinguishes a revisited place from dwelling in one cell.
        The other two triggers are observed local stall and repeated collision.
        Comparisons, not nonlinear kernels, suffice for deployment.
        """
        revisit = features[:, 1, 9] >= 3
        stalled = features[:, 2, 12] >= 0.1
        collided = features[:, 1, 5] >= 0.2197
        return revisit | stalled | collided

    def __init__(self):
        super().__init__()
        self.cell = DendriticCell()
        self.encoder = SparseLinear(24, 128)
        self.interaction_head = nn.Linear(128, 1, bias=False)
        self.value_head = nn.Linear(128, 1)
        nn.init.zeros_(self.interaction_head.weight)
        nn.init.zeros_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)

    def evaluate(self, candidates, spatial, valid, state, features):
        embedding = spike(self.encoder(candidates))
        history = state.readout[:, None].expand_as(embedding)
        raw = self.interaction_head(embedding * history).squeeze(-1)
        gate = self.evidence_gate(features).to(raw.dtype)[:, None]
        advantage = raw.clamp(-1.0, 1.0) * gate
        residual = advantage * self.MAX_CORRECTION * locality_gain(candidates)
        value = self.value_head(state.readout).squeeze(-1)[:, None]
        invalid = ~valid.bool()
        policy = (spatial + residual + value).masked_fill(invalid, -float("inf"))
        critic = (value + advantage).masked_fill(invalid, -float("inf"))
        return policy, residual, critic

    def score(self, candidates, spatial, valid, state, features):
        policy, residual, _ = self.evaluate(
            candidates, spatial, valid, state, features
        )
        return policy, residual
