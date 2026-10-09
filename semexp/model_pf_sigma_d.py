"""Online Semantic-Exploration adapter for the incremental Sigma-Delta PONI.

This module is intentionally additive: ``semexp/model_pf.py`` remains the
original stateless PONI path.  The classes here accept only checkpoints written
by ``train_sigma_d.py`` and never fall back to an original PONI network.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Union

import torch
import torch.nn as nn

from poni.model_sigma_d import PONISigmaD, PONISigmaDState
from semexp.model_pf import RL_Policy as OriginalRLPolicy


Checkpoint = Union[str, Path, Mapping[str, Any]]


def _load_checkpoint(checkpoint: Checkpoint) -> Mapping[str, Any]:
    if isinstance(checkpoint, (str, Path)):
        checkpoint = torch.load(str(checkpoint), map_location="cpu")
    if checkpoint.get("model_family") != "sigma_d":
        raise RuntimeError(
            "The Sigma-D online adapter accepts only train_sigma_d.py "
            "checkpoints (model_family='sigma_d'); original PONI fallback is "
            "deliberately disabled."
        )
    if "cfg" not in checkpoint or "state_dict" not in checkpoint:
        raise KeyError("Sigma-D checkpoint must contain cfg and state_dict")
    return checkpoint


def _build_model(cfg) -> PONISigmaD:
    sigma_cfg = cfg.MODEL.sigma_d
    return PONISigmaD(
        num_categories=int(cfg.MODEL.num_categories),
        nsf=int(cfg.MODEL.nsf),
        embedding_size=int(cfg.MODEL.embedding_size),
        enable_area_head=bool(cfg.DATASET.enable_unexp_area),
        threshold=float(sigma_cfg.threshold),
        output_threshold=float(sigma_cfg.output_threshold),
        qmax=int(sigma_cfg.qmax),
        learnable_threshold=bool(sigma_cfg.learnable_threshold),
        power_of_two_threshold=bool(sigma_cfg.power_of_two_threshold),
    )


class Potential_Function_Semantic_PolicySigmaD(nn.Module):
    """Stateful PF inference with explicit per-environment reset semantics."""

    def __init__(self, pf_model_path: Checkpoint) -> None:
        super().__init__()
        checkpoint = _load_checkpoint(pf_model_path)
        self._cfg = checkpoint["cfg"]
        self.sigma_d = _build_model(self._cfg)
        missing, unexpected = self.sigma_d.load_state_dict(
            checkpoint["state_dict"], strict=False
        )
        if missing or unexpected:
            raise RuntimeError(
                f"Sigma-D checkpoint/model mismatch; missing={missing}, "
                f"unexpected={unexpected}"
            )

        # Plain Python runtime fields are deliberately absent from state_dict.
        self._runtime_state: Optional[PONISigmaDState] = None
        self._pending_reset: Optional[torch.Tensor] = None
        self.last_event_stats = None
        self.last_reset_mask = None
        self.reset_statistics = {
            "full_requests": 0,
            "manual_rows": 0,
            "frame_jump_rows": 0,
            "episode_rows": 0,
        }
        self.eval()

    @property
    def pf_model(self):
        """Compatibility facade for the original RL_Policy properties."""

        return self

    @property
    def cfg(self):
        return self._cfg

    def get_pf_cfg(self):
        return {"dthresh": float(self._cfg.DATASET.object_pf_cutoff_dist)}

    def convert_object_pf_to_distance(
        self, pfs: torch.Tensor, min_value: float = 1e-20, max_value: float = 1.0
    ) -> torch.Tensor:
        pfs = torch.clamp(pfs, min_value, max_value)
        max_distance = float(self._cfg.DATASET.object_pf_cutoff_dist)
        return max_distance - pfs * max_distance

    def convert_distance_to_pf(self, distances: torch.Tensor) -> torch.Tensor:
        max_distance = float(self._cfg.DATASET.object_pf_cutoff_dist)
        return torch.clamp((max_distance - distances) / max_distance, 0.0, 1.0)

    def add_agent_dists_to_object_dists(
        self, pfs: torch.Tensor, agent_dists: torch.Tensor
    ) -> torch.Tensor:
        distances = self.convert_object_pf_to_distance(pfs)
        return self.convert_distance_to_pf(distances + agent_dists.unsqueeze(1))

    def reset_runtime_state(
        self, reset_mask: Optional[torch.Tensor] = None, reason: str = "manual"
    ) -> None:
        """Request a full or partial dense reset on the next policy call.

        A full reset is appropriate after moving the module/checkpoint or when
        the whole vectorized environment changes coordinate frame.  A boolean
        mask resets only selected environments, for example after local-map
        origin changes.  Episode termination is also inferred from the regular
        policy ``masks`` argument in :meth:`forward`.
        """

        if reset_mask is None:
            self._runtime_state = None
            self._pending_reset = None
            self.reset_statistics["full_requests"] += 1
            return
        requested = torch.as_tensor(reset_mask, dtype=torch.bool).flatten().cpu()
        key = "frame_jump_rows" if reason == "frame_jump" else "manual_rows"
        self.reset_statistics[key] += int(requested.sum().item())
        if self._pending_reset is None:
            self._pending_reset = requested
        else:
            if self._pending_reset.numel() != requested.numel():
                raise ValueError("Pending and requested reset masks have different sizes")
            self._pending_reset |= requested

    def _make_reset_mask(
        self, inputs: torch.Tensor, masks: Optional[torch.Tensor]
    ) -> torch.Tensor:
        batch_size = inputs.shape[0]
        reset = torch.zeros(batch_size, dtype=torch.bool, device=inputs.device)
        if masks is not None:
            episode_masks = masks.to(device=inputs.device).reshape(batch_size, -1)
            episode_reset = torch.all(episode_masks <= 0, dim=1)
            reset |= episode_reset
            self.reset_statistics["episode_rows"] += int(episode_reset.sum().item())
        if self._pending_reset is not None:
            if self._pending_reset.numel() != batch_size:
                raise ValueError("Pending reset mask does not match input batch size")
            reset |= self._pending_reset.to(device=inputs.device)
            self._pending_reset = None
        return reset

    def forward(self, inputs, rnn_hxs=None, masks=None, extras=None):
        if inputs.ndim != 4:
            raise ValueError(f"Expected BCHW semantic map, got {tuple(inputs.shape)}")
        if self._runtime_state is not None:
            previous = self._runtime_state.previous_input
            if previous.shape[0] != inputs.shape[0]:
                # Vector-env resize has no meaningful row correspondence.
                self._runtime_state = None

        reset = self._make_reset_mask(inputs, masks)
        result = self.sigma_d.forward_step(inputs, self._runtime_state, reset)
        self._runtime_state = self.sigma_d.detach_state(result.state)
        self.last_event_stats = result.event_stats
        self.last_reset_mask = reset.detach().cpu()
        return result.object_pf, result.area_pf


class RL_PolicySigmaD(OriginalRLPolicy):
    """Drop-in action-selection policy backed only by Sigma-D inference."""

    def __init__(self, args, pf_model_path: Checkpoint) -> None:
        # Do not call OriginalRLPolicy.__init__: it would instantiate and load
        # the original stateless PONI model before replacing it.
        nn.Module.__init__(self)
        self.args = args
        self.network = Potential_Function_Semantic_PolicySigmaD(pf_model_path)
        self._cached_visualizations = None
        self._previous_agent_locations: Optional[torch.Tensor] = None

    def reset_sigma_d_state(self, reset_mask: Optional[torch.Tensor] = None) -> None:
        self.network.reset_runtime_state(reset_mask)
        if reset_mask is None:
            self._previous_agent_locations = None

    def _reset_on_map_frame_jump(self, inputs, extra_maps) -> None:
        """Reset rows whose local allocentric crop moved to a new origin.

        HLab does not pass ``lmb`` to its policy, but it does pass the agent's
        local-map cell.  Smart-boundary recentering produces a large discrete
        jump in that coordinate, whereas one navigation step is small.  A
        false positive only performs an exact dense init and cannot change the
        prediction; a missed coordinate-frame change would corrupt state.
        """

        if extra_maps is None or extra_maps.get("agent_locations") is None:
            return
        locations = torch.as_tensor(
            extra_maps["agent_locations"], dtype=torch.float32
        ).reshape(inputs.shape[0], 2)
        if self._previous_agent_locations is not None:
            if self._previous_agent_locations.shape != locations.shape:
                self.network.reset_runtime_state()
            else:
                jump = (locations - self._previous_agent_locations).abs().amax(dim=1)
                threshold = max(8.0, 0.1 * float(min(inputs.shape[-2:])))
                self.network.reset_runtime_state(
                    jump > threshold, reason="frame_jump"
                )
        self._previous_agent_locations = locations.cpu()

    def act(
        self, inputs, rnn_hxs, masks, extras=None, extra_maps=None, deterministic=False
    ):
        # A pose-dependent egocentric transform changes the tensor coordinate
        # frame every call; treating that as an ordinary delta would corrupt
        # state.  Map-mode PONI evaluation uses a fixed allocentric frame.
        if self.needs_egocentric_transform:
            raise RuntimeError(
                "Sigma-D online inference requires a fixed allocentric map frame. "
                "Disable egocentric transforms or explicitly redesign the adapter "
                "to reset after every transform."
            )
        self._reset_on_map_frame_jump(inputs, extra_maps)
        return super().act(
            inputs,
            rnn_hxs,
            masks,
            extras=extras,
            extra_maps=extra_maps,
            deterministic=deterministic,
        )


__all__ = [
    "Potential_Function_Semantic_PolicySigmaD",
    "RL_PolicySigmaD",
]
