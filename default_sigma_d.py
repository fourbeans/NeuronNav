"""Configuration for the incremental Sigma-Delta PONI experiment line.

This module intentionally extends, rather than edits, :mod:`poni.default`.
The resulting config remains compatible with the original PONI dataset and
loss settings while adding explicit sequence, Sigma-Delta, and run-management
options.
"""

from typing import List, Optional, Union

from poni.default import CONFIG_FILE_SEPARATOR, Config, get_cfg as get_poni_cfg


CN = Config


def _add_sigma_d_defaults(cfg: CN) -> CN:
    cfg.defrost()

    # Model identity is saved in every checkpoint. Evaluation code must reject
    # any checkpoint whose family is not ``sigma_d``; there is no PONI fallback.
    cfg.MODEL.family = "sigma_d"
    cfg.MODEL.sigma_d = CN()
    cfg.MODEL.sigma_d.threshold = 1.0 / 128.0
    cfg.MODEL.sigma_d.output_threshold = 1.0 / 255.0
    cfg.MODEL.sigma_d.qmax = 15
    cfg.MODEL.sigma_d.learnable_threshold = False
    cfg.MODEL.sigma_d.power_of_two_threshold = True

    # A sequence contains K *different, progressively explored* maps. It must
    # never be implemented by repeating one static map K times.
    cfg.DATASET.sequence_length = 4
    cfg.DATASET.train_samples_per_epoch = 20000
    cfg.DATASET.val_samples_per_epoch = 1000
    cfg.DATASET.require_progressive_maps = True
    cfg.DATASET.max_sampling_attempts = 32
    # True only for the explicit small-set memorisation diagnostic. Normal
    # training changes the deterministic on-the-fly sample stream each epoch.
    cfg.DATASET.freeze_train_samples = False
    # Workers are recreated each epoch so dataset.set_epoch() reaches their
    # copies and on-the-fly samples remain deterministic but non-repeating.
    cfg.DATASET.persistent_workers = False

    # Direct GT supervision only. Temporal and full-map reconstruction terms
    # are computed against GT labels, never a teacher model. Event/saturation
    # counters are detached diagnostics and therefore intentionally have no
    # optimization weights.
    cfg.LOSS = CN()
    cfg.LOSS.object_weight = 1.0
    cfg.LOSS.area_weight = 1.0
    cfg.LOSS.temporal_weight = 0.0
    cfg.LOSS.reconstruction_weight = 0.0
    cfg.LOSS.eps = 1e-16

    cfg.OPTIM.grad_accum_steps = 1
    cfg.OPTIM.weight_decay = 0.0
    cfg.OPTIM.grad_clip_norm = 5.0
    cfg.OPTIM.amp = False
    # Preserve the original PONI interpretation of LR milestones by default.
    cfg.OPTIM.scheduler_interval = "epoch"  # options: epoch, update

    cfg.RUNTIME = CN()
    cfg.RUNTIME.sync_batchnorm = True
    cfg.RUNTIME.find_unused_parameters = False
    cfg.RUNTIME.deterministic = False
    cfg.RUNTIME.cudnn_benchmark = True
    cfg.RUNTIME.distributed_backend = "nccl"

    # output_dir is the sole run root. Subdirectories are derived by the
    # trainer, preventing checkpoints, TensorBoard data, and logs from mixing.
    cfg.LOGGING.output_dir = "dairy/sigma_d_unet"
    cfg.LOGGING.metrics_file = "metrics.jsonl"
    cfg.LOGGING.manifest_file = "source_manifest.json"
    cfg.LOGGING.config_file = "config.yaml"
    cfg.LOGGING.best_metric = "val/task_loss"
    cfg.LOGGING.best_mode = "min"
    cfg.LOGGING.auto_resume = True

    cfg.freeze()
    return cfg


def get_cfg_sigma_d(
    config_paths: Optional[Union[List[str], str]] = None,
    opts: Optional[list] = None,
) -> CN:
    """Return the original PONI config plus Sigma-Delta-only extensions."""

    cfg = _add_sigma_d_defaults(get_poni_cfg())
    cfg.defrost()

    if config_paths:
        if isinstance(config_paths, str):
            config_paths = (
                config_paths.split(CONFIG_FILE_SEPARATOR)
                if CONFIG_FILE_SEPARATOR in config_paths
                else [config_paths]
            )
        for config_path in config_paths:
            cfg.merge_from_file(config_path)

    if opts:
        cfg.merge_from_list(opts)

    if cfg.MODEL.family != "sigma_d":
        raise ValueError("train_sigma_d.py only accepts MODEL.family='sigma_d'")
    if cfg.DATASET.sequence_length < 2:
        raise ValueError("DATASET.sequence_length must be at least two")
    if cfg.OPTIM.grad_accum_steps < 1:
        raise ValueError("OPTIM.grad_accum_steps must be positive")
    if cfg.OPTIM.scheduler_interval not in ("epoch", "update"):
        raise ValueError("OPTIM.scheduler_interval must be 'epoch' or 'update'")
    if cfg.LOGGING.best_mode not in ("min", "max"):
        raise ValueError("LOGGING.best_mode must be 'min' or 'max'")

    cfg.freeze()
    return cfg


# Keep a familiar name for experiment scripts without shadowing poni.default.
get_cfg = get_cfg_sigma_d
