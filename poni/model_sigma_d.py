"""Incremental Sigma-Delta U-Net compatible with the original PONI model.

This module deliberately keeps *runtime* state outside ``nn.Module``.  A caller
passes a :class:`PONISigmaDState` to :meth:`PONISigmaD.forward_step` and receives
the next state in the return value.  Consequently episode state is not stored in
``state_dict()``, cannot leak between DDP replicas, and can be reset per batch
element with ``reset_mask``.

The first frame of an episode is evaluated densely and exactly.  Later frames
propagate signed activation deltas:

    u[t] = u[t-1] + W * delta_a[t]
    q[t] = round_ste((activation(u[t]) - a_hat[t-1]) / threshold)
    delta_a[t] = threshold * clip(q[t], -qmax, qmax)
    a_hat[t] = a_hat[t-1] + delta_a[t]

No artificial inner time dimension is used: one navigation map update is one
``forward_step`` call.  Pooling may currently recompute a complete pooled map,
but it emits the mathematically correct output delta.  This is a correctness
reference for a future affected-window/event implementation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn import functional as F


# -----------------------------------------------------------------------------
# Explicit runtime state and public results
# -----------------------------------------------------------------------------


@dataclass
class DeltaLayerState:
    """State of one affine layer followed by an activation/SD quantizer."""

    preactivation: Tensor
    reconstruction: Tensor


@dataclass
class PoolState:
    """Cached input/output of a stateful max-pooling operation."""

    input_reconstruction: Tensor
    output_reconstruction: Tensor


@dataclass
class DoubleConvState:
    first: DeltaLayerState
    second: DeltaLayerState


@dataclass
class DownState:
    pool: PoolState
    conv: DoubleConvState


@dataclass
class EncoderState:
    embed: DeltaLayerState
    inc: DoubleConvState
    down1: DownState
    down2: DownState
    down3: DownState
    down4: DownState


@dataclass
class UpState:
    conv: DoubleConvState


@dataclass
class DecoderState:
    up1: UpState
    up2: UpState
    up3: UpState
    up4: UpState
    outc: DeltaLayerState


@dataclass
class PONISigmaDState:
    """Complete external state for one batched PONI Sigma-Delta stream."""

    previous_input: Tensor
    encoder: EncoderState
    object_decoder: DecoderState
    area_decoder: Optional[DecoderState]


@dataclass
class LayerEventStats:
    """Tensor-valued statistics to avoid forced CPU synchronization."""

    events: Tensor
    active: Tensor
    values: int
    saturated: Tensor


@dataclass
class SigmaDStepResult:
    object_pf: Tensor
    area_pf: Optional[Tensor]
    state: PONISigmaDState
    event_stats: Dict[str, Any]


@dataclass
class OriginalPONILoadReport:
    loaded_layers: Tuple[str, ...]
    missing_layers: Tuple[str, ...]
    unused_source_keys: Tuple[str, ...]


def _tree_map(value: Any, fn) -> Any:
    if isinstance(value, Tensor):
        return fn(value)
    if is_dataclass(value):
        return type(value)(
            **{field.name: _tree_map(getattr(value, field.name), fn) for field in fields(value)}
        )
    if isinstance(value, dict):
        return {key: _tree_map(item, fn) for key, item in value.items()}
    if isinstance(value, tuple):
        return type(value)(_tree_map(item, fn) for item in value)
    if isinstance(value, list):
        return [_tree_map(item, fn) for item in value]
    if value is None:
        return None
    raise TypeError(f"Unsupported runtime-state leaf: {type(value)!r}")


def detach_sigma_d_state(state: Optional[PONISigmaDState]) -> Optional[PONISigmaDState]:
    """Detach every runtime tensor, for truncated BPTT or inference reuse."""

    if state is None:
        return None
    return _tree_map(state, Tensor.detach)


def _index_state(state: Any, indices: Tensor) -> Any:
    return _tree_map(state, lambda tensor: tensor.index_select(0, indices))


def _merge_indexed(parts: Sequence[Tuple[Tensor, Any]], batch_size: int) -> Any:
    """Merge identically structured tensor/dataclass trees along batch dim."""

    if not parts:
        raise ValueError("At least one indexed part is required")
    example = parts[0][1]
    if isinstance(example, Tensor):
        shape = (batch_size,) + tuple(example.shape[1:])
        merged = example.new_zeros(shape)
        for indices, value in parts:
            merged = merged.index_copy(0, indices, value)
        return merged
    if is_dataclass(example):
        return type(example)(
            **{
                field.name: _merge_indexed(
                    [(indices, getattr(value, field.name)) for indices, value in parts],
                    batch_size,
                )
                for field in fields(example)
            }
        )
    if example is None:
        if not all(value is None for _, value in parts):
            raise ValueError("Cannot merge a mixture of None and non-None state")
        return None
    raise TypeError(f"Unsupported merge leaf: {type(example)!r}")


# -----------------------------------------------------------------------------
# Signed straight-through Sigma-Delta quantization
# -----------------------------------------------------------------------------


class _RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value: Tensor) -> Tensor:  # type: ignore[override]
        return torch.round(value)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor]:  # type: ignore[override]
        return (grad_output,)


def round_ste(value: Tensor) -> Tensor:
    return _RoundSTE.apply(value)


class SignedSigmaDelta(nn.Module):
    """Signed error quantizer with an optionally learnable power-of-two step."""

    def __init__(
        self,
        threshold: float,
        qmax: int = 15,
        learnable_threshold: bool = False,
        power_of_two_threshold: bool = True,
    ) -> None:
        super().__init__()
        if threshold <= 0:
            raise ValueError("threshold must be positive")
        if qmax < 1:
            raise ValueError("qmax must be at least one")
        log2_threshold = torch.tensor(math.log2(float(threshold)), dtype=torch.float32)
        if learnable_threshold:
            self.log2_threshold = nn.Parameter(log2_threshold)
        else:
            self.register_buffer("log2_threshold", log2_threshold)
        self.qmax = int(qmax)
        self.power_of_two_threshold = bool(power_of_two_threshold)

    def threshold(self, reference: Tensor) -> Tensor:
        exponent = self.log2_threshold
        if self.power_of_two_threshold:
            exponent = exponent + (torch.round(exponent) - exponent).detach()
        return torch.pow(exponent.new_tensor(2.0), exponent).to(
            device=reference.device, dtype=reference.dtype
        )

    def forward(self, target: Tensor, reconstruction: Tensor) -> Tuple[Tensor, LayerEventStats]:
        theta = self.threshold(target)
        normalized_error = (target - reconstruction) / theta
        rounded = round_ste(normalized_error)
        clipped = torch.clamp(rounded, -self.qmax, self.qmax)
        delta = clipped * theta
        stats = LayerEventStats(
            events=clipped.detach().abs().sum(),
            active=clipped.detach().ne(0).sum(),
            values=clipped.numel(),
            saturated=rounded.detach().abs().gt(self.qmax).sum(),
        )
        return delta, stats


# -----------------------------------------------------------------------------
# Incremental building blocks
# -----------------------------------------------------------------------------


def _activation(name: str, value: Tensor) -> Tensor:
    if name == "relu":
        return F.relu(value)
    if name == "sigmoid":
        return torch.sigmoid(value)
    if name == "identity":
        return value
    raise ValueError(f"Unknown activation {name!r}")


def _conv_delta(conv: nn.Conv2d, delta: Tensor) -> Tensor:
    """Apply only the linear part of Conv2d; bias belongs to dense init."""

    return F.conv2d(
        delta,
        conv.weight,
        bias=None,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
    )


class DeltaConvActivation(nn.Module):
    """Conv/activation whose post-activation changes are Sigma-Delta events."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        padding: int = 0,
        activation: str = "relu",
        threshold: float = 1.0 / 128.0,
        qmax: int = 15,
        learnable_threshold: bool = False,
        power_of_two_threshold: bool = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=padding, bias=True
        )
        self.activation_name = activation
        self.sigma_delta = SignedSigmaDelta(
            threshold,
            qmax=qmax,
            learnable_threshold=learnable_threshold,
            power_of_two_threshold=power_of_two_threshold,
        )

    def initialize(self, value: Tensor) -> Tuple[Tensor, DeltaLayerState]:
        preactivation = self.conv(value)
        reconstruction = _activation(self.activation_name, preactivation)
        # Dense initialization is intentionally exact; no first-frame rounding.
        state = DeltaLayerState(preactivation, reconstruction)
        return reconstruction, state

    def update(
        self, delta: Tensor, state: DeltaLayerState
    ) -> Tuple[Tensor, Tensor, DeltaLayerState, LayerEventStats]:
        preactivation = state.preactivation + _conv_delta(self.conv, delta)
        target = _activation(self.activation_name, preactivation)
        emitted_delta, stats = self.sigma_delta(target, state.reconstruction)
        reconstruction = state.reconstruction + emitted_delta
        next_state = DeltaLayerState(preactivation, reconstruction)
        return reconstruction, emitted_delta, next_state, stats


class DeltaDoubleConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        **sigma_delta_kwargs: Any,
    ) -> None:
        super().__init__()
        self.first = DeltaConvActivation(
            in_channels, out_channels, 3, padding=1, **sigma_delta_kwargs
        )
        self.second = DeltaConvActivation(
            out_channels, out_channels, 3, padding=1, **sigma_delta_kwargs
        )

    def initialize(self, value: Tensor) -> Tuple[Tensor, DoubleConvState]:
        value, first = self.first.initialize(value)
        value, second = self.second.initialize(value)
        return value, DoubleConvState(first, second)

    def update(
        self,
        delta: Tensor,
        state: DoubleConvState,
        stats: MutableMapping[str, LayerEventStats],
        prefix: str,
    ) -> Tuple[Tensor, Tensor, DoubleConvState]:
        _, delta, first, first_stats = self.first.update(delta, state.first)
        value, delta, second, second_stats = self.second.update(delta, state.second)
        stats[f"{prefix}.first"] = first_stats
        stats[f"{prefix}.second"] = second_stats
        return value, delta, DoubleConvState(first, second)


class DeltaMaxPool2d(nn.Module):
    """Correct stateful max-pool reference; currently recomputes the full pool."""

    def __init__(self, kernel_size: int = 2, stride: int = 2) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def initialize(self, value: Tensor) -> Tuple[Tensor, PoolState]:
        pooled = F.max_pool2d(value, self.kernel_size, self.stride)
        return pooled, PoolState(value, pooled)

    def update(
        self, delta: Tensor, state: PoolState
    ) -> Tuple[Tensor, Tensor, PoolState, LayerEventStats]:
        input_reconstruction = state.input_reconstruction + delta
        pooled = F.max_pool2d(input_reconstruction, self.kernel_size, self.stride)
        pooled_delta = pooled - state.output_reconstruction
        stats = LayerEventStats(
            events=pooled_delta.detach().ne(0).sum(),
            active=pooled_delta.detach().ne(0).sum(),
            values=pooled_delta.numel(),
            saturated=pooled_delta.new_zeros((), dtype=torch.long),
        )
        return pooled, pooled_delta, PoolState(input_reconstruction, pooled), stats


class DeltaDown(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs: Any) -> None:
        super().__init__()
        self.pool = DeltaMaxPool2d(2, 2)
        self.conv = DeltaDoubleConv(in_channels, out_channels, **kwargs)

    def initialize(self, value: Tensor) -> Tuple[Tensor, DownState]:
        value, pool = self.pool.initialize(value)
        value, conv = self.conv.initialize(value)
        return value, DownState(pool, conv)

    def update(
        self,
        delta: Tensor,
        state: DownState,
        stats: MutableMapping[str, LayerEventStats],
        prefix: str,
    ) -> Tuple[Tensor, Tensor, DownState]:
        _, delta, pool, pool_stats = self.pool.update(delta, state.pool)
        stats[f"{prefix}.pool"] = pool_stats
        value, delta, conv = self.conv.update(delta, state.conv, stats, f"{prefix}.conv")
        return value, delta, DownState(pool, conv)


def _upsample_and_pad(value: Tensor, target_hw: Tuple[int, int]) -> Tensor:
    value = F.interpolate(value, scale_factor=2, mode="bilinear", align_corners=True)
    diff_y = target_hw[0] - value.shape[-2]
    diff_x = target_hw[1] - value.shape[-1]
    if diff_y < 0 or diff_x < 0:
        raise ValueError(
            f"Upsampled tensor {tuple(value.shape[-2:])} is larger than target {target_hw}"
        )
    return F.pad(
        value,
        (diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2),
    )


class DeltaUp(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs: Any) -> None:
        super().__init__()
        self.conv = DeltaDoubleConv(in_channels, out_channels, **kwargs)

    def initialize(self, deep: Tensor, skip: Tensor) -> Tuple[Tensor, UpState]:
        deep = _upsample_and_pad(deep, tuple(skip.shape[-2:]))
        value, conv = self.conv.initialize(torch.cat([skip, deep], dim=1))
        return value, UpState(conv)

    def update(
        self,
        deep_delta: Tensor,
        skip_delta: Tensor,
        state: UpState,
        stats: MutableMapping[str, LayerEventStats],
        prefix: str,
    ) -> Tuple[Tensor, Tensor, UpState]:
        deep_delta = _upsample_and_pad(deep_delta, tuple(skip_delta.shape[-2:]))
        input_delta = torch.cat([skip_delta, deep_delta], dim=1)
        value, delta, conv = self.conv.update(input_delta, state.conv, stats, f"{prefix}.conv")
        return value, delta, UpState(conv)


# -----------------------------------------------------------------------------
# Encoder, decoders and complete model
# -----------------------------------------------------------------------------


FeatureTuple = Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]


class UNetEncoderSigmaD(nn.Module):
    def __init__(
        self,
        n_channels: int,
        nsf: int = 32,
        embedding_size: int = 64,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.embed = DeltaConvActivation(
            n_channels, embedding_size, 1, activation="identity", **kwargs
        )
        self.inc = DeltaDoubleConv(embedding_size, nsf, **kwargs)
        self.down1 = DeltaDown(nsf, nsf * 2, **kwargs)
        self.down2 = DeltaDown(nsf * 2, nsf * 4, **kwargs)
        self.down3 = DeltaDown(nsf * 4, nsf * 8, **kwargs)
        self.down4 = DeltaDown(nsf * 8, nsf * 8, **kwargs)

    def initialize(self, value: Tensor) -> Tuple[FeatureTuple, EncoderState]:
        value, embed = self.embed.initialize(value)
        x1, inc = self.inc.initialize(value)
        x2, down1 = self.down1.initialize(x1)
        x3, down2 = self.down2.initialize(x2)
        x4, down3 = self.down3.initialize(x3)
        x5, down4 = self.down4.initialize(x4)
        return (x1, x2, x3, x4, x5), EncoderState(
            embed, inc, down1, down2, down3, down4
        )

    def update(
        self,
        input_delta: Tensor,
        state: EncoderState,
        stats: MutableMapping[str, LayerEventStats],
    ) -> Tuple[FeatureTuple, FeatureTuple, EncoderState]:
        _, delta, embed, layer_stats = self.embed.update(input_delta, state.embed)
        stats["encoder.embed"] = layer_stats
        x1, d1, inc = self.inc.update(delta, state.inc, stats, "encoder.inc")
        x2, d2, down1 = self.down1.update(d1, state.down1, stats, "encoder.down1")
        x3, d3, down2 = self.down2.update(d2, state.down2, stats, "encoder.down2")
        x4, d4, down3 = self.down3.update(d3, state.down3, stats, "encoder.down3")
        x5, d5, down4 = self.down4.update(d4, state.down4, stats, "encoder.down4")
        next_state = EncoderState(embed, inc, down1, down2, down3, down4)
        return (x1, x2, x3, x4, x5), (d1, d2, d3, d4, d5), next_state


class UNetDecoderSigmaD(nn.Module):
    def __init__(
        self,
        n_classes: int,
        nsf: int = 32,
        *,
        output_activation: str = "sigmoid",
        output_threshold: float = 1.0 / 255.0,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.up1 = DeltaUp(nsf * 16, nsf * 4, **kwargs)
        self.up2 = DeltaUp(nsf * 8, nsf * 2, **kwargs)
        self.up3 = DeltaUp(nsf * 4, nsf, **kwargs)
        self.up4 = DeltaUp(nsf * 2, nsf, **kwargs)
        output_kwargs = dict(kwargs)
        output_kwargs["threshold"] = output_threshold
        self.outc = DeltaConvActivation(
            nsf, n_classes, 1, activation=output_activation, **output_kwargs
        )

    def initialize(self, features: FeatureTuple) -> Tuple[Tensor, DecoderState]:
        x1, x2, x3, x4, x5 = features
        value, up1 = self.up1.initialize(x5, x4)
        value, up2 = self.up2.initialize(value, x3)
        value, up3 = self.up3.initialize(value, x2)
        value, up4 = self.up4.initialize(value, x1)
        value, outc = self.outc.initialize(value)
        return value, DecoderState(up1, up2, up3, up4, outc)

    def update(
        self,
        feature_deltas: FeatureTuple,
        state: DecoderState,
        stats: MutableMapping[str, LayerEventStats],
        prefix: str,
    ) -> Tuple[Tensor, Tensor, DecoderState]:
        d1, d2, d3, d4, d5 = feature_deltas
        _, delta, up1 = self.up1.update(d5, d4, state.up1, stats, f"{prefix}.up1")
        _, delta, up2 = self.up2.update(delta, d3, state.up2, stats, f"{prefix}.up2")
        _, delta, up3 = self.up3.update(delta, d2, state.up3, stats, f"{prefix}.up3")
        _, delta, up4 = self.up4.update(delta, d1, state.up4, stats, f"{prefix}.up4")
        value, delta, outc, layer_stats = self.outc.update(delta, state.outc)
        stats[f"{prefix}.outc"] = layer_stats
        return value, delta, DecoderState(up1, up2, up3, up4, outc)


class PONISigmaD(nn.Module):
    """PONI-compatible U-Net with explicit incremental Sigma-Delta state."""

    def __init__(
        self,
        num_categories: int,
        nsf: int = 32,
        embedding_size: int = 64,
        *,
        enable_area_head: bool = True,
        threshold: float = 1.0 / 128.0,
        output_threshold: float = 1.0 / 255.0,
        qmax: int = 15,
        learnable_threshold: bool = False,
        power_of_two_threshold: bool = True,
    ) -> None:
        super().__init__()
        common = dict(
            threshold=threshold,
            qmax=qmax,
            learnable_threshold=learnable_threshold,
            power_of_two_threshold=power_of_two_threshold,
        )
        self.num_categories = int(num_categories)
        self.encoder = UNetEncoderSigmaD(
            num_categories, nsf=nsf, embedding_size=embedding_size, **common
        )
        self.object_decoder = UNetDecoderSigmaD(
            num_categories,
            nsf=nsf,
            output_threshold=output_threshold,
            **common,
        )
        self.area_decoder = (
            UNetDecoderSigmaD(
                1, nsf=nsf, output_threshold=output_threshold, **common
            )
            if enable_area_head
            else None
        )

    def _initialize_batch(self, value: Tensor) -> SigmaDStepResult:
        features, encoder = self.encoder.initialize(value)
        object_pf, object_decoder = self.object_decoder.initialize(features)
        if self.area_decoder is not None:
            area_pf, area_decoder = self.area_decoder.initialize(features)
        else:
            area_pf, area_decoder = None, None
        # Keep an owned snapshot: online map tensors are commonly updated
        # in-place by mapping code between policy calls.
        state = PONISigmaDState(value.clone(), encoder, object_decoder, area_decoder)
        stats: Dict[str, Any] = {
            "mode": "dense_init",
            "dense_samples": value.shape[0],
            "incremental_samples": 0,
            "layers": {},
        }
        return SigmaDStepResult(object_pf, area_pf, state, stats)

    def _update_batch(self, value: Tensor, state: PONISigmaDState) -> SigmaDStepResult:
        if tuple(value.shape) != tuple(state.previous_input.shape):
            raise ValueError(
                "Input/state shape mismatch; pass reset_mask=True for streams whose "
                f"coordinate frame changed ({tuple(state.previous_input.shape)} -> "
                f"{tuple(value.shape)})"
            )
        layers: Dict[str, LayerEventStats] = {}
        input_delta = value - state.previous_input
        _, feature_deltas, encoder = self.encoder.update(input_delta, state.encoder, layers)
        object_pf, _, object_decoder = self.object_decoder.update(
            feature_deltas, state.object_decoder, layers, "object_decoder"
        )
        if self.area_decoder is not None:
            if state.area_decoder is None:
                raise ValueError("Area decoder is enabled but missing from runtime state")
            area_pf, _, area_decoder = self.area_decoder.update(
                feature_deltas, state.area_decoder, layers, "area_decoder"
            )
        else:
            area_pf, area_decoder = None, None
        next_state = PONISigmaDState(value.clone(), encoder, object_decoder, area_decoder)
        stats: Dict[str, Any] = {
            "mode": "incremental",
            "dense_samples": 0,
            "incremental_samples": value.shape[0],
            "layers": layers,
        }
        return SigmaDStepResult(object_pf, area_pf, next_state, stats)

    def forward_step(
        self,
        value: Tensor,
        state: Optional[PONISigmaDState] = None,
        reset_mask: Optional[Tensor] = None,
    ) -> SigmaDStepResult:
        """Run one map update, with independent reset decisions per batch row.

        ``reset_mask[b]`` must be true for a new episode or changed map frame.
        Reset rows run an exact dense initialization, while other rows use their
        supplied incremental state.  The two subsets are merged back into their
        original batch order.
        """

        if value.ndim != 4:
            raise ValueError(f"Expected BCHW input, got shape {tuple(value.shape)}")
        batch_size = value.shape[0]
        if reset_mask is None:
            reset_mask = torch.zeros(batch_size, dtype=torch.bool, device=value.device)
        else:
            reset_mask = reset_mask.to(device=value.device, dtype=torch.bool).flatten()
            if reset_mask.numel() != batch_size:
                raise ValueError("reset_mask must have one element per batch row")

        if state is None:
            # With no prior state every row necessarily needs dense
            # initialization, irrespective of a caller's partial mask.
            return self._initialize_batch(value)
        if state.previous_input.shape[0] != batch_size:
            raise ValueError("State and input batch sizes differ")

        reset_indices = torch.nonzero(reset_mask, as_tuple=False).flatten()
        keep_indices = torch.nonzero(~reset_mask, as_tuple=False).flatten()
        if reset_indices.numel() == 0:
            return self._update_batch(value, state)
        if keep_indices.numel() == 0:
            return self._initialize_batch(value)

        kept = self._update_batch(
            value.index_select(0, keep_indices), _index_state(state, keep_indices)
        )
        reset = self._initialize_batch(value.index_select(0, reset_indices))
        indexed_states = [(keep_indices, kept.state), (reset_indices, reset.state)]
        indexed_objects = [(keep_indices, kept.object_pf), (reset_indices, reset.object_pf)]
        merged_state = _merge_indexed(indexed_states, batch_size)
        object_pf = _merge_indexed(indexed_objects, batch_size)
        if kept.area_pf is None:
            area_pf = None
        else:
            if reset.area_pf is None:
                raise ValueError("Inconsistent area-head outputs while merging reset batch")
            area_pf = _merge_indexed(
                [(keep_indices, kept.area_pf), (reset_indices, reset.area_pf)], batch_size
            )
        stats = kept.event_stats
        stats.update(
            mode="mixed",
            dense_samples=int(reset_indices.numel()),
            incremental_samples=int(keep_indices.numel()),
        )
        return SigmaDStepResult(object_pf, area_pf, merged_state, stats)

    def forward(self, value: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        """Stateless dense interface matching original PONI ``forward``."""

        result = self._initialize_batch(value)
        return result.object_pf, result.area_pf

    @staticmethod
    def reset_state() -> None:
        """Return the canonical empty external state."""

        return None

    @staticmethod
    def detach_state(state: Optional[PONISigmaDState]) -> Optional[PONISigmaDState]:
        return detach_sigma_d_state(state)

    def load_original_poni_checkpoint(
        self,
        checkpoint: Union[str, Path, Mapping[str, Any]],
        *,
        strict: bool = True,
        batch_norm_eps: float = 1e-5,
    ) -> OriginalPONILoadReport:
        """Load and fold an original PONI U-Net checkpoint into this model.

        Original ``Conv2d -> BatchNorm2d -> ReLU`` pairs are folded into this
        model's biased Conv2d layers.  The resulting dense first-frame output is
        equivalent to the original model in eval mode (for bilinear PONI U-Net
        checkpoints with matching channel counts).
        """

        if isinstance(checkpoint, (str, Path)):
            source: Mapping[str, Any] = torch.load(str(checkpoint), map_location="cpu")
        else:
            source = checkpoint
        if "state_dict" in source:
            source = source["state_dict"]
        normalized: Dict[str, Tensor] = {}
        for key, tensor in source.items():
            if not isinstance(tensor, Tensor):
                continue
            key = key.replace(".module.", ".")
            if key.startswith("module."):
                key = key[len("module.") :]
            normalized[key] = tensor

        used = set()
        loaded = []
        missing = []

        def copy_plain(source_prefix: str, destination: nn.Conv2d) -> None:
            weight_key = f"{source_prefix}.weight"
            bias_key = f"{source_prefix}.bias"
            if weight_key not in normalized:
                missing.append(source_prefix)
                return
            weight = normalized[weight_key]
            bias = normalized.get(bias_key)
            if tuple(weight.shape) != tuple(destination.weight.shape):
                raise ValueError(
                    f"Shape mismatch for {source_prefix}: {tuple(weight.shape)} vs "
                    f"{tuple(destination.weight.shape)}"
                )
            with torch.no_grad():
                destination.weight.copy_(weight)
                if bias is None:
                    destination.bias.zero_()
                else:
                    destination.bias.copy_(bias)
            used.add(weight_key)
            if bias is not None:
                used.add(bias_key)
            loaded.append(source_prefix)

        def copy_folded(source_prefix: str, destination: nn.Conv2d) -> None:
            conv_prefix = source_prefix
            bn_prefix = source_prefix.rsplit(".", 1)[0] + "." + str(
                int(source_prefix.rsplit(".", 1)[1]) + 1
            )
            required = [
                f"{conv_prefix}.weight",
                f"{bn_prefix}.weight",
                f"{bn_prefix}.bias",
                f"{bn_prefix}.running_mean",
                f"{bn_prefix}.running_var",
            ]
            if not all(key in normalized for key in required):
                missing.append(source_prefix)
                return
            weight = normalized[f"{conv_prefix}.weight"]
            conv_bias = normalized.get(f"{conv_prefix}.bias", torch.zeros(weight.shape[0]))
            gamma = normalized[f"{bn_prefix}.weight"]
            beta = normalized[f"{bn_prefix}.bias"]
            mean = normalized[f"{bn_prefix}.running_mean"]
            variance = normalized[f"{bn_prefix}.running_var"]
            scale = gamma / torch.sqrt(variance + batch_norm_eps)
            folded_weight = weight * scale.reshape(-1, 1, 1, 1)
            folded_bias = beta + (conv_bias - mean) * scale
            if tuple(folded_weight.shape) != tuple(destination.weight.shape):
                raise ValueError(
                    f"Shape mismatch for {source_prefix}: {tuple(folded_weight.shape)} vs "
                    f"{tuple(destination.weight.shape)}"
                )
            with torch.no_grad():
                destination.weight.copy_(folded_weight)
                destination.bias.copy_(folded_bias)
            used.update(required)
            bias_key = f"{conv_prefix}.bias"
            if bias_key in normalized:
                used.add(bias_key)
            tracked_key = f"{bn_prefix}.num_batches_tracked"
            if tracked_key in normalized:
                used.add(tracked_key)
            loaded.append(source_prefix)

        copy_plain("encoder.embed.conv", self.encoder.embed.conv)

        def load_double(source_prefix: str, block: DeltaDoubleConv) -> None:
            copy_folded(f"{source_prefix}.0", block.first.conv)
            copy_folded(f"{source_prefix}.3", block.second.conv)

        load_double("encoder.inc.conv.conv", self.encoder.inc)
        load_double("encoder.down1.mpconv.1.conv", self.encoder.down1.conv)
        load_double("encoder.down2.mpconv.1.conv", self.encoder.down2.conv)
        load_double("encoder.down3.mpconv.1.conv", self.encoder.down3.conv)
        load_double("encoder.down4.mpconv.1.conv", self.encoder.down4.conv)

        def load_decoder(source_prefix: str, decoder: UNetDecoderSigmaD) -> None:
            load_double(f"{source_prefix}.up1.conv.conv", decoder.up1.conv)
            load_double(f"{source_prefix}.up2.conv.conv", decoder.up2.conv)
            load_double(f"{source_prefix}.up3.conv.conv", decoder.up3.conv)
            load_double(f"{source_prefix}.up4.conv.conv", decoder.up4.conv)
            copy_plain(f"{source_prefix}.outc.conv", decoder.outc.conv)

        load_decoder("object_decoder", self.object_decoder)
        if self.area_decoder is not None:
            load_decoder("area_decoder", self.area_decoder)

        if strict and missing:
            raise KeyError("Missing original PONI layers: " + ", ".join(sorted(missing)))
        unused = tuple(sorted(set(normalized) - used))
        return OriginalPONILoadReport(tuple(loaded), tuple(missing), unused)


__all__ = [
    "PONISigmaD",
    "PONISigmaDState",
    "SigmaDStepResult",
    "LayerEventStats",
    "OriginalPONILoadReport",
    "detach_sigma_d_state",
    "round_ste",
]
