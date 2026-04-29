"""Shared utilities: config, seeding, device, I/O, metrics, logging."""

from src.common.config import (
    AttractorConfig,
    ExperimentConfig,
    PRESET_FAMILIES,
    SubjectConfig,
    SupportConfig,
    TrainingConfig,
    VisualEncoderConfig,
    load_config,
)
from src.common.device import get_device
from src.common.io import (
    best_checkpoint_filename,
    ensure_dir,
    infer_mode,
    infer_preset,
    load_checkpoint,
    save_checkpoint,
)
from src.common.logging import get_logger
from src.common.metrics import (
    METRIC_NAMES,
    accuracy,
    binary_cross_entropy,
    brier_score,
    calibration_bins,
    mae,
    multiclass_accuracy,
    multiclass_nll,
    rmse,
)
from src.common.seed import set_seed

__all__ = [
    # config
    "SupportConfig",
    "AttractorConfig",
    "SubjectConfig",
    "VisualEncoderConfig",
    "TrainingConfig",
    "ExperimentConfig",
    "PRESET_FAMILIES",
    "load_config",
    # device
    "get_device",
    # seed
    "set_seed",
    # io
    "save_checkpoint",
    "load_checkpoint",
    "infer_preset",
    "infer_mode",
    "ensure_dir",
    "best_checkpoint_filename",
    # logging
    "get_logger",
    # metrics
    "METRIC_NAMES",
    "accuracy",
    "multiclass_accuracy",
    "binary_cross_entropy",
    "multiclass_nll",
    "rmse",
    "mae",
    "brier_score",
    "calibration_bins",
]
