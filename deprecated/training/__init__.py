"""Training engine, losses, and evaluation."""

from src.training.losses import multitask_loss, total_loss
from src.training.engine import (
    MetricAccumulator,
    Trainer,
    build_modules,
    build_task_heads,
    collect_parameters,
    load_task_data,
    make_parser,
    run_training,
)
from src.training.evaluators import (
    compute_kstar_summary,
    evaluate_tasks,
    export_subject_params,
)

__all__ = [
    # losses
    "multitask_loss",
    "total_loss",
    # engine
    "build_modules",
    "build_task_heads",
    "collect_parameters",
    "load_task_data",
    "MetricAccumulator",
    "Trainer",
    "make_parser",
    "run_training",
    # evaluators
    "evaluate_tasks",
    "export_subject_params",
    "compute_kstar_summary",
]
