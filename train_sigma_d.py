"""Train the incremental Sigma-Delta U-Net with direct PONI supervision.

Design constraints enforced here:

* every sequence element is a different progressive navigation map; a static
  map is never copied T times to emulate spikes;
* labels come directly from ground-truth object/area potential fields (there is
  no teacher model or distillation loss);
* only the Sigma-Delta model is constructed (there is no original-PONI
  fallback path);
* runtime state is an external dataclass, starts as ``None`` for every batch,
  and can therefore never enter a model checkpoint.
"""

import argparse
import contextlib
import hashlib
import json
import logging
import math
import os
import random
import shlex
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from default_sigma_d import get_cfg_sigma_d


LOGGER = logging.getLogger("train_sigma_d")


def _distributed_context(cfg) -> Tuple[int, int, int, torch.device]:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if not torch.cuda.is_available():
        raise RuntimeError("Sigma-Delta training requires a CUDA device")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(
            backend=cfg.RUNTIME.distributed_backend,
            init_method="env://",
            rank=rank,
            world_size=world_size,
        )
    return local_rank, rank, world_size, device


def _barrier(world_size: int) -> None:
    if world_size > 1:
        # NCCL otherwise has to guess the rank-to-device mapping during an
        # early barrier.  The guess happened to work for the original two-GPU
        # runs but can deadlock with a four-GPU CUDA_VISIBLE_DEVICES subset.
        # Every trainer rank has already called torch.cuda.set_device above,
        # so make that mapping explicit for all initialization/eval barriers.
        dist.barrier(device_ids=[torch.cuda.current_device()])


def _seed_everything(seed: int, rank: int, cfg) -> None:
    seed = int(seed) + int(rank)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(cfg.RUNTIME.deterministic)
    torch.backends.cudnn.benchmark = bool(cfg.RUNTIME.cudnn_benchmark) and not bool(
        cfg.RUNTIME.deterministic
    )


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _setup_logger(output_dir: Path, rank: int) -> None:
    LOGGER.handlers.clear()
    LOGGER.setLevel(logging.INFO if rank == 0 else logging.WARNING)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | rank=%(process)d | %(message)s"
    )
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    LOGGER.addHandler(stream)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(output_dir / "train.log")
        file_handler.setFormatter(formatter)
        LOGGER.addHandler(file_handler)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_source_manifest(output_dir: Path, filename: str) -> None:
    root = Path(__file__).resolve().parent
    relative_sources = [
        "train_sigma_d.py",
        "default_sigma_d.py",
        "poni/model_sigma_d.py",
        "poni/dataset_sigma_d.py",
        "poni/model.py",
        "poni/dataset.py",
    ]
    files = {}
    for relative in relative_sources:
        path = root / relative
        files[relative] = {
            "exists": path.is_file(),
            "sha256": _sha256(path) if path.is_file() else None,
            "size_bytes": path.stat().st_size if path.is_file() else None,
        }
    git = {"head": None, "status": None}
    try:
        git["head"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(root), text=True, timeout=10
        ).strip()
        git["status"] = subprocess.check_output(
            ["git", "status", "--short"], cwd=str(root), text=True, timeout=10
        ).splitlines()
    except (OSError, subprocess.SubprocessError):
        pass
    payload = {
        "created_unix": time.time(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "git": git,
        "files": files,
    }
    with (output_dir / filename).open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


class _PONISigmaDTrainingAdapter(torch.nn.Module):
    """Run one complete real-map sequence inside one DDP forward call.

    DDP's reducer expects one forward graph before its backward pass.  Calling
    the DDP wrapper K times and only then backpropagating through the stacked
    result is not a supported recurrent pattern.  The adapter therefore keeps
    the explicit state local to this single call while every temporal step
    still uses :meth:`PONISigmaD.forward_step` exactly once.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.sigma_d = model

    def forward(self, sequence, reset_mask):
        state = None
        object_steps = []
        area_steps = []
        event_steps = []
        for step in range(sequence.shape[1]):
            result = self.sigma_d.forward_step(
                sequence[:, step], state, reset_mask[:, step]
            )
            state = result.state
            object_steps.append(result.object_pf)
            if result.area_pf is not None:
                area_steps.append(result.area_pf)
            event_steps.append(result.event_stats)
        # State is external, sequence-local, and deliberately discarded after
        # the graph's observable predictions have been assembled.
        state = self.sigma_d.detach_state(state)
        del state
        object_preds = torch.stack(object_steps, dim=1)
        area_preds = torch.stack(area_steps, dim=1) if area_steps else None
        return object_preds, area_preds, event_steps


def _build_model(cfg) -> torch.nn.Module:
    """Construct the concrete incremental PONI implementation, and only it."""

    try:
        from poni.model_sigma_d import PONISigmaD
    except ImportError as exc:
        raise ImportError(
            "Missing poni/model_sigma_d.py; expected class PONISigmaD"
        ) from exc
    sigma_cfg = cfg.MODEL.sigma_d
    model = PONISigmaD(
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
    if cfg.MODEL.pretrained_path:
        report = model.load_original_poni_checkpoint(
            cfg.MODEL.pretrained_path, strict=True
        )
        # Plain metadata only; it never enters state_dict/checkpoints.
        model._sigma_d_initialization_report = report
    return _PONISigmaDTrainingAdapter(model)


def _build_dataset(cfg, split: str):
    """Build the concrete on-the-fly progressive-map dataset."""

    try:
        from poni.dataset_sigma_d import SigmaDTemporalSemanticMapDataset
    except ImportError as exc:
        raise ImportError(
            "Missing poni/dataset_sigma_d.py; expected "
            "SigmaDTemporalSemanticMapDataset"
        ) from exc
    samples_per_epoch = (
        cfg.DATASET.train_samples_per_epoch
        if split == "train"
        else cfg.DATASET.val_samples_per_epoch
    )
    return SigmaDTemporalSemanticMapDataset(
        cfg.DATASET,
        split=split,
        sequence_length=int(cfg.DATASET.sequence_length),
        samples_per_epoch=int(samples_per_epoch),
        seed=int(cfg.DATASET.seed),
        max_sampling_attempts=int(cfg.DATASET.max_sampling_attempts),
        require_strict_growth=bool(cfg.DATASET.require_progressive_maps),
    )


def _make_loader(cfg, split: str, rank: int, world_size: int):
    dataset = _build_dataset(cfg, split)
    is_train = split == "train"
    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=is_train,
            drop_last=is_train,
        )
    generator = torch.Generator()
    generator.manual_seed(int(cfg.SEED) + (0 if is_train else 100000) + rank)
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.OPTIM.batch_size),
        shuffle=is_train and sampler is None,
        sampler=sampler,
        num_workers=int(cfg.OPTIM.num_workers),
        pin_memory=True,
        drop_last=is_train,
        worker_init_fn=_seed_worker,
        generator=generator,
        persistent_workers=(
            int(cfg.OPTIM.num_workers) > 0 and bool(cfg.DATASET.persistent_workers)
        ),
    )
    if len(loader) == 0:
        raise RuntimeError(f"The {split} sequence loader is empty")
    return dataset, sampler, loader


def _as_float_label(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.is_floating_point():
        return tensor.float()
    # Matches the original precomputed PONI representation (0..1000).
    return tensor.float().div_(1000.0)


def _normalise_batch(batch, cfg, device):
    required = ("frames", "object_pfs", "loss_masks", "valid_steps", "reset_mask")
    missing = [key for key in required if key not in batch]
    if missing:
        raise KeyError(f"Temporal dataset batch is missing: {missing}")
    sequence = batch["frames"].to(device, non_blocking=True).float()
    if sequence.ndim != 5:
        raise ValueError(
            f"frames must be [B,T,C,H,W], got {tuple(sequence.shape)}"
        )
    batch_size, steps = sequence.shape[:2]
    valid_mask = batch["valid_steps"].to(device, non_blocking=True).bool()
    reset_mask = batch["reset_mask"].to(device, non_blocking=True).bool()
    reset_mask = reset_mask.clone()
    reset_mask[:, 0] = True
    if valid_mask.shape != (batch_size, steps) or reset_mask.shape != (
        batch_size,
        steps,
    ):
        raise ValueError("valid_mask and reset_mask must have shape [B,T]")

    if cfg.DATASET.require_progressive_maps and steps > 1:
        with torch.no_grad():
            pair_valid = valid_mask[:, 1:] & valid_mask[:, :-1] & ~reset_mask[:, 1:]
            changed = (sequence[:, 1:] != sequence[:, :-1]).flatten(2).any(dim=2)
            if torch.any(pair_valid & ~changed):
                raise ValueError(
                    "A valid sequence repeats an unchanged map; artificial T-step "
                    "map repetition is forbidden"
                )
            explored = sequence.bool().any(dim=2)
            lost = explored[:, :-1] & ~explored[:, 1:]
            lost = lost.flatten(2).any(dim=2)
            if torch.any(pair_valid & lost):
                raise ValueError(
                    "Progressive training maps lost explored cells without a reset"
                )

    labels = {
        "object_pfs": _as_float_label(
            batch["object_pfs"].to(device, non_blocking=True)
        ),
        "loss_masks": batch["loss_masks"].to(device, non_blocking=True).float(),
        "area_pfs": None,
    }
    if cfg.DATASET.enable_unexp_area:
        if "area_pfs" not in batch:
            raise KeyError("enable_unexp_area=True requires batch['area_pfs']")
        labels["area_pfs"] = _as_float_label(
            batch["area_pfs"].to(device, non_blocking=True)
        )
    for key in ("object_pfs", "loss_masks"):
        if labels[key].shape[:2] != (batch_size, steps):
            raise ValueError(f"{key} must start with [B,T]")
    return sequence, reset_mask, valid_mask, labels


def _zero(reference: torch.Tensor) -> torch.Tensor:
    return reference.sum() * 0.0


def _compute_losses(outputs: Mapping[str, Any], labels, valid_mask, reset_mask, cfg):
    if not isinstance(outputs, Mapping) or "object_preds" not in outputs:
        raise TypeError("Internal step stack must contain object_preds")
    object_preds = outputs["object_preds"]
    object_targets = labels["object_pfs"]
    masks = labels["loss_masks"]
    if object_preds.shape != object_targets.shape or masks.shape != object_targets.shape:
        raise ValueError(
            "object_preds, object_pfs, and loss_masks must have identical shapes; "
            f"got {object_preds.shape}, {object_targets.shape}, {masks.shape}"
        )
    if object_preds.ndim != 5 or object_preds.shape[2] <= 2:
        raise ValueError("object predictions must be [B,T,C>2,H,W]")

    pred = object_preds[:, :, 2:]
    target = object_targets[:, :, 2:]
    mask = masks[:, :, 2:]
    eps = float(cfg.LOSS.eps)
    per_category = ((pred - target).square() * mask).sum(dim=(-1, -2))
    per_category = per_category / (mask.sum(dim=(-1, -2)) + eps)
    valid = valid_mask.float()
    object_loss = (per_category * valid.unsqueeze(-1)).sum()
    object_loss = object_loss / (
        valid.sum() * per_category.shape[-1] + eps
    )

    area_preds = outputs.get("area_preds")
    area_targets = labels.get("area_pfs")
    if area_preds is None and area_targets is None:
        area_loss = _zero(object_preds)
    elif area_preds is None or area_targets is None:
        raise ValueError("area_preds and area_pfs must either both exist or both be absent")
    else:
        if area_preds.shape != area_targets.shape:
            raise ValueError(
                f"area prediction/target shape mismatch: {area_preds.shape} vs "
                f"{area_targets.shape}"
            )
        per_step_area = (area_preds - area_targets).square().flatten(2).mean(dim=2)
        area_loss = (per_step_area * valid).sum() / (valid.sum() + eps)

    # Direct temporal supervision: match changes in GT potential fields. This
    # uses no teacher network and is disabled by default.
    if object_preds.shape[1] > 1:
        pair_valid = valid_mask[:, 1:] & valid_mask[:, :-1] & ~reset_mask[:, 1:]
        pred_delta = pred[:, 1:] - pred[:, :-1]
        target_delta = target[:, 1:] - target[:, :-1]
        pair_mask = torch.maximum(mask[:, 1:], mask[:, :-1])
        per_category_delta = (
            (pred_delta - target_delta).square() * pair_mask
        ).sum(dim=(-1, -2)) / (pair_mask.sum(dim=(-1, -2)) + eps)
        temporal_loss = (
            per_category_delta * pair_valid.float().unsqueeze(-1)
        ).sum() / (
            pair_valid.float().sum() * per_category_delta.shape[-1] + eps
        )
    else:
        temporal_loss = _zero(object_preds)

    # Optional full-map GT reconstruction term. Unlike event statistics this is
    # differentiable, and it still uses no teacher model. It is off by default.
    per_step_reconstruction = (pred - target).square().flatten(2).mean(dim=2)
    reconstruction_loss = (per_step_reconstruction * valid).sum() / (
        valid.sum() + eps
    )

    task_loss = (
        float(cfg.LOSS.object_weight) * object_loss
        + float(cfg.LOSS.area_weight) * area_loss
    )
    total_loss = (
        task_loss
        + float(cfg.LOSS.temporal_weight) * temporal_loss
        + float(cfg.LOSS.reconstruction_weight) * reconstruction_loss
    )
    losses = {
        "loss": total_loss,
        "task_loss": task_loss,
        "object_loss": object_loss,
        "area_loss": area_loss,
        "temporal_loss": temporal_loss,
        "reconstruction_loss": reconstruction_loss,
    }
    return losses


def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


def _sigma_model(model: torch.nn.Module) -> torch.nn.Module:
    return _unwrap(model).sigma_d


def _checkpoint_model_state(model: torch.nn.Module) -> MutableMapping[str, Any]:
    # PONISigmaD keeps all temporal state in the external PONISigmaDState
    # dataclass, so its module state_dict contains parameters/buffers only.
    return _sigma_model(model).state_dict()


def _move_optimizer_state(optimizer, device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


class MetricAverager:
    def __init__(self):
        self.sums = defaultdict(float)
        self.weights = defaultdict(float)

    def add(self, values: Mapping[str, Any], weight: float = 1.0) -> None:
        for key, value in values.items():
            if torch.is_tensor(value):
                if value.numel() != 1:
                    continue
                value = value.detach().float().item()
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                self.sums[key] += float(value) * float(weight)
                self.weights[key] += float(weight)

    def distributed_mean(self, device, world_size: int) -> Dict[str, float]:
        result = {}
        for key in sorted(self.sums):
            pair = torch.tensor(
                [self.sums[key], self.weights[key]], dtype=torch.float64, device=device
            )
            if world_size > 1:
                dist.all_reduce(pair, op=dist.ReduceOp.SUM)
            result[key] = float(pair[0].item() / max(pair[1].item(), 1e-12))
        return result

    def clear(self) -> None:
        self.sums.clear()
        self.weights.clear()


def _json_scalar(value: Any) -> Any:
    if torch.is_tensor(value) and value.numel() == 1:
        return value.detach().cpu().item()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _summarize_event_stats(step_stats: Iterable[Mapping[str, Any]], reference):
    """Reduce detached model counters; these are diagnostics, never losses."""

    events = reference.new_zeros(())
    active = reference.new_zeros(())
    saturated = reference.new_zeros(())
    values = reference.new_zeros(())
    dense_samples = 0
    incremental_samples = 0
    for stats in step_stats:
        dense_samples += int(stats.get("dense_samples", 0))
        incremental_samples += int(stats.get("incremental_samples", 0))
        for layer_stats in stats.get("layers", {}).values():
            events = events + layer_stats.events.to(reference.device).float()
            active = active + layer_stats.active.to(reference.device).float()
            saturated = saturated + layer_stats.saturated.to(reference.device).float()
            values = values + float(layer_stats.values)
    denominator = torch.clamp(values, min=1.0)
    return {
        "event_count": events.detach(),
        "active_event_count": active.detach(),
        "saturated_count": saturated.detach(),
        "event_value_count": values.detach(),
        "event_magnitude_per_value": (events / denominator).detach(),
        "active_event_rate": (active / denominator).detach(),
        "saturation_rate": (saturated / denominator).detach(),
        "dense_samples": float(dense_samples),
        "incremental_samples": float(incremental_samples),
    }


class SigmaDeltaTrainer:
    def __init__(self, cfg, explicit_resume: str = ""):
        self.cfg = cfg
        self.local_rank, self.rank, self.world_size, self.device = _distributed_context(
            cfg
        )
        self.output_dir = Path(cfg.LOGGING.output_dir).resolve()
        _setup_logger(self.output_dir, self.rank)
        _seed_everything(cfg.SEED, self.rank, cfg)
        _barrier(self.world_size)

        if self.rank == 0:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            (self.output_dir / "checkpoints").mkdir(exist_ok=True)
            (self.output_dir / "tb").mkdir(exist_ok=True)
            (self.output_dir / cfg.LOGGING.config_file).write_text(cfg.dump())
            arguments = " ".join(shlex.quote(part) for part in sys.argv[1:])
            visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            command_lines = [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                "source /mnt/jfzn/miniconda3/etc/profile.d/conda.sh",
                "conda activate poni",
                "cd " + shlex.quote(str(Path(__file__).resolve().parent)),
            ]
            if visible:
                command_lines.append(
                    "export CUDA_VISIBLE_DEVICES=" + shlex.quote(visible)
                )
            if self.world_size > 1:
                command_lines.append(
                    "torchrun --standalone --nproc_per_node="
                    + str(self.world_size)
                    + " "
                    + shlex.quote(str(Path(__file__).name))
                    + (" " + arguments if arguments else "")
                )
            else:
                command_lines.append(
                    "python "
                    + shlex.quote(str(Path(__file__).name))
                    + (" " + arguments if arguments else "")
                )
            (self.output_dir / "command.sh").write_text(
                "\n".join(command_lines) + "\n"
            )
            _write_source_manifest(self.output_dir, cfg.LOGGING.manifest_file)
        _barrier(self.world_size)

        self.model = _build_model(cfg).to(self.device)
        if self.world_size > 1 and cfg.RUNTIME.sync_batchnorm:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        if self.world_size > 1:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                broadcast_buffers=False,
                find_unused_parameters=bool(cfg.RUNTIME.find_unused_parameters),
            )

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(cfg.OPTIM.lr),
            weight_decay=float(cfg.OPTIM.weight_decay),
        )
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=list(cfg.OPTIM.lr_sched_milestones),
            gamma=float(cfg.OPTIM.lr_sched_gamma),
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.OPTIM.amp))

        self.train_dataset, self.train_sampler, self.train_loader = _make_loader(
            cfg, "train", self.rank, self.world_size
        )
        self.val_dataset, self.val_sampler, self.val_loader = _make_loader(
            cfg, "val", self.rank, self.world_size
        )

        self.global_update = 0
        self.epoch = 0
        self.next_batch_in_epoch = 0
        self.best_value = math.inf if cfg.LOGGING.best_mode == "min" else -math.inf
        self.metrics_path = self.output_dir / cfg.LOGGING.metrics_file
        self.writer = (
            SummaryWriter(str(self.output_dir / "tb")) if self.rank == 0 else None
        )

        auto_path = self.output_dir / "checkpoints" / "last.ckpt"
        resume_path = Path(explicit_resume) if explicit_resume else auto_path
        if explicit_resume or (cfg.LOGGING.auto_resume and resume_path.is_file()):
            self._resume(resume_path)

        LOGGER.info(
            "world_size=%d per_rank_batch=%d grad_accum=%d effective_batch=%d",
            self.world_size,
            cfg.OPTIM.batch_size,
            cfg.OPTIM.grad_accum_steps,
            cfg.OPTIM.batch_size * self.world_size * cfg.OPTIM.grad_accum_steps,
        )
        LOGGER.info(
            "train_sequences=%d val_sequences=%d sequence_length=%d",
            len(self.train_dataset),
            len(self.val_dataset),
            cfg.DATASET.sequence_length,
        )
        init_report = getattr(
            _sigma_model(self.model), "_sigma_d_initialization_report", None
        )
        if init_report is None:
            LOGGER.info("initialization=random (no teacher and no distillation)")
        else:
            LOGGER.info(
                "initialization=original_poni_warm_start loaded_layers=%d "
                "unused_source_keys=%d (supervised training remains GT-only)",
                len(init_report.loaded_layers),
                len(init_report.unused_source_keys),
            )

    def _resume(self, path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {path}")
        checkpoint = torch.load(str(path), map_location="cpu")
        if checkpoint.get("model_family") != "sigma_d":
            raise RuntimeError("Refusing to resume a non-Sigma-Delta checkpoint")
        missing, unexpected = _sigma_model(self.model).load_state_dict(
            checkpoint["state_dict"], strict=False
        )
        if missing or unexpected:
            raise RuntimeError(
                f"Checkpoint/model mismatch; missing={missing}, unexpected={unexpected}"
            )
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        _move_optimizer_state(self.optimizer, self.device)
        self.scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        if "grad_scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["grad_scaler_state_dict"])
        self.global_update = int(checkpoint["global_update"])
        self.epoch = int(checkpoint["epoch"])
        self.next_batch_in_epoch = int(checkpoint.get("next_batch_in_epoch", 0))
        self.best_value = float(checkpoint["best_value"])
        LOGGER.info("resumed %s at update=%d", path, self.global_update)

    def _checkpoint_payload(self) -> Dict[str, Any]:
        return {
            "model_family": "sigma_d",
            "state_dict": _checkpoint_model_state(self.model),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lr_scheduler_state_dict": self.scheduler.state_dict(),
            "grad_scaler_state_dict": self.scaler.state_dict(),
            "global_update": self.global_update,
            "epoch": self.epoch,
            "next_batch_in_epoch": self.next_batch_in_epoch,
            "best_metric": self.cfg.LOGGING.best_metric,
            "best_value": self.best_value,
            "cfg": self.cfg,
            "config_yaml": self.cfg.dump(),
        }

    def _save(self, name: str) -> None:
        if self.rank != 0:
            return
        destination = self.output_dir / "checkpoints" / name
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        torch.save(self._checkpoint_payload(), temporary)
        os.replace(temporary, destination)
        LOGGER.info("saved checkpoint %s", destination)

    def _append_metrics(self, phase: str, values: Mapping[str, Any]) -> None:
        if self.rank != 0:
            return
        record = {
            "time_unix": time.time(),
            "phase": phase,
            "global_update": self.global_update,
            "epoch": self.epoch,
            "lr": self.optimizer.param_groups[0]["lr"],
        }
        record.update({key: _json_scalar(value) for key, value in values.items()})
        with self.metrics_path.open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        if self.writer is not None:
            for key, value in values.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f"{phase}/{key}", value, self.global_update)

    def _forward_batch(self, batch):
        sequence, reset_mask, valid_mask, labels = _normalise_batch(
            batch, self.cfg, self.device
        )
        # One DDP forward contains K distinct map updates; the adapter owns the
        # explicit runtime state only for the duration of this call.
        object_preds, area_preds, event_steps = self.model(sequence, reset_mask)
        if self.cfg.DATASET.enable_unexp_area:
            if area_preds is None or area_preds.shape[1] != sequence.shape[1]:
                raise RuntimeError("Enabled area head did not return every time step")
        else:
            if area_preds is not None:
                raise RuntimeError("Disabled area head unexpectedly returned predictions")
        outputs = {"object_preds": object_preds, "area_preds": area_preds}
        losses = _compute_losses(outputs, labels, valid_mask, reset_mask, self.cfg)
        stats = _summarize_event_stats(event_steps, object_preds)
        return losses, stats, sequence.shape[0]

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        self.model.eval()
        if self.val_sampler is not None:
            self.val_sampler.set_epoch(0)
        if callable(getattr(self.val_dataset, "set_epoch", None)):
            # Keep the validation cohort fixed across checkpoints. Otherwise
            # an apparent loss change can be caused by a different path draw.
            self.val_dataset.set_epoch(0)
        meter = MetricAverager()
        for batch in self.val_loader:
            losses, stats, batch_size = self._forward_batch(batch)
            meter.add({key: value for key, value in losses.items() if key != "loss"}, batch_size)
            meter.add({f"stat/{key}": value for key, value in stats.items()}, batch_size)
        metrics = meter.distributed_mean(self.device, self.world_size)
        metrics["total_loss"] = float(
            self.cfg.LOSS.object_weight * metrics["object_loss"]
            + self.cfg.LOSS.area_weight * metrics["area_loss"]
            + self.cfg.LOSS.temporal_weight * metrics["temporal_loss"]
            + self.cfg.LOSS.reconstruction_weight * metrics["reconstruction_loss"]
        )
        self.model.train()
        return metrics

    def _is_better(self, value: float) -> bool:
        if self.cfg.LOGGING.best_mode == "min":
            return value < self.best_value
        return value > self.best_value

    def _best_metric_value(self, val_metrics: Mapping[str, float]) -> float:
        configured = self.cfg.LOGGING.best_metric
        key = configured.split("/", 1)[1] if configured.startswith("val/") else configured
        if key not in val_metrics:
            raise KeyError(
                f"LOGGING.best_metric={configured!r} not found in validation metrics"
            )
        return float(val_metrics[key])

    def _optimizer_step(self, accumulated_micro_batches: int) -> float:
        grad_accum = int(self.cfg.OPTIM.grad_accum_steps)
        if accumulated_micro_batches < grad_accum:
            # Each micro loss was divided by grad_accum. Correct the last,
            # shorter accumulation group at an epoch boundary.
            factor = grad_accum / float(accumulated_micro_batches)
            for parameter in self.model.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(factor)
        self.scaler.unscale_(self.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), float(self.cfg.OPTIM.grad_clip_norm)
        )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        if self.cfg.OPTIM.scheduler_interval == "update":
            self.scheduler.step()
        return float(grad_norm.detach().item() if torch.is_tensor(grad_norm) else grad_norm)

    def train(self) -> None:
        max_updates = int(self.cfg.OPTIM.num_total_updates)
        grad_accum = int(self.cfg.OPTIM.grad_accum_steps)
        train_meter = MetricAverager()
        self.optimizer.zero_grad(set_to_none=True)
        self.model.train()

        while self.global_update < max_updates:
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(self.epoch)
            if callable(getattr(self.train_dataset, "set_epoch", None)):
                sampled_epoch = (
                    0 if self.cfg.DATASET.freeze_train_samples else self.epoch
                )
                self.train_dataset.set_epoch(sampled_epoch)

            accumulated = 0
            epoch_start = time.time()
            for batch_index, batch in enumerate(self.train_loader):
                if batch_index < self.next_batch_in_epoch:
                    continue
                is_last_batch = batch_index + 1 == len(self.train_loader)
                will_step = accumulated + 1 == grad_accum or is_last_batch
                sync_context = (
                    contextlib.nullcontext()
                    if will_step or not isinstance(self.model, DDP)
                    else self.model.no_sync()
                )
                with sync_context:
                    with torch.cuda.amp.autocast(enabled=bool(self.cfg.OPTIM.amp)):
                        losses, stats, batch_size = self._forward_batch(batch)
                        scaled_loss = losses["loss"] / float(grad_accum)
                    self.scaler.scale(scaled_loss).backward()
                accumulated += 1
                train_meter.add(
                    {key: value for key, value in losses.items() if key != "loss"},
                    batch_size,
                )
                train_meter.add(
                    {f"stat/{key}": value for key, value in stats.items()}, batch_size
                )
                self.next_batch_in_epoch = batch_index + 1

                if not will_step:
                    continue

                grad_norm = self._optimizer_step(accumulated)
                accumulated = 0
                self.global_update += 1

                if self.global_update % int(self.cfg.LOGGING.log_interval) == 0:
                    train_metrics = train_meter.distributed_mean(
                        self.device, self.world_size
                    )
                    train_metrics["grad_norm"] = grad_norm
                    self._append_metrics("train", train_metrics)
                    LOGGER.info(
                        "update=%d/%d epoch=%d task=%.6f object=%.6f area=%.6f",
                        self.global_update,
                        max_updates,
                        self.epoch,
                        train_metrics.get("task_loss", float("nan")),
                        train_metrics.get("object_loss", float("nan")),
                        train_metrics.get("area_loss", float("nan")),
                    )
                    train_meter.clear()

                if self.global_update % int(self.cfg.LOGGING.eval_interval) == 0:
                    _barrier(self.world_size)
                    val_metrics = self.validate()
                    self._append_metrics("val", val_metrics)
                    selected = self._best_metric_value(val_metrics)
                    if self._is_better(selected):
                        self.best_value = selected
                        self._save("best.ckpt")
                    LOGGER.info(
                        "validation update=%d task=%.6f total=%.6f best=%.6f",
                        self.global_update,
                        val_metrics["task_loss"],
                        val_metrics["total_loss"],
                        self.best_value,
                    )
                    _barrier(self.world_size)

                if self.global_update % int(self.cfg.LOGGING.ckpt_interval) == 0:
                    self._save("last.ckpt")

                if self.global_update >= max_updates:
                    break

            if self.global_update >= max_updates:
                break
            self.next_batch_in_epoch = 0
            self.epoch += 1
            if self.cfg.OPTIM.scheduler_interval == "epoch":
                self.scheduler.step()
            LOGGER.info("finished epoch=%d seconds=%.1f", self.epoch, time.time() - epoch_start)

        _barrier(self.world_size)
        self._save("last.ckpt")
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
        _barrier(self.world_size)

    def close(self) -> None:
        if dist.is_initialized():
            dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-config", type=str, default="")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cfg = get_cfg_sigma_d(args.exp_config, args.opts)
    trainer = SigmaDeltaTrainer(cfg, explicit_resume=args.resume)
    try:
        trainer.train()
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
