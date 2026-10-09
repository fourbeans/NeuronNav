"""Distance-aware gain for dendritic goal reranking.

Candidate coordinates are stored as agent-relative metres divided by 20 in
feature columns 3:5. The gain is a small set of shifts/comparisons at runtime;
it has no learned parameters and does not call another navigation policy.
"""
import torch


def locality_gain(candidates):
    """Return [B, N] gains of 1, 1/2, 1/4 or 1/16 versus the spatial anchor."""
    if candidates.ndim != 3 or candidates.shape[-1] != 24:
        raise ValueError("candidates must have shape [B,N,24]")
    displacement_metres = (
        (candidates[..., 3:5] - candidates[:, :1, 3:5]).abs().sum(-1) * 20.0
    )
    one = torch.ones_like(displacement_metres)
    return torch.where(
        displacement_metres <= 0.5,
        one,
        torch.where(
            displacement_metres <= 1.0,
            one * 0.5,
            torch.where(displacement_metres <= 2.0, one * 0.25, one * 0.0625),
        ),
    )
