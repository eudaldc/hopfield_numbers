"""Transfer evaluation: apply a Cohort-A-trained model to Cohort B / Panamath.

Zero-shot approach: the core geometry (rho, sigma0, …) is frozen from the
checkpoint; new subjects get default subject parameters.

Usage::

    python scripts/eval_transfer.py --checkpoint checkpoints/best_default_dynamic.pt \\
        --data-dir data/processed_cohort_b --tasks panamath symbolic_comparison
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.common.device import get_device
from src.common.io import infer_preset
from src.common.seed import set_seed
from src.common.config import ExperimentConfig, SubjectConfig
from src.data.collate import multitask_collate
from src.data.dataset_builders import (
    SubjectIndexMap,
    SymbolicComparisonDataset,
    MatchToSampleDataset,
    DotVerificationDataset,
    PanamathDataset,
    MultitaskDataset,
)
from src.models.subject_params import SubjectParams
from src.training.engine import (
    build_modules,
    build_task_heads,
    _make_eval_loader,
    _CheckpointWrapper,
)
from src.training.evaluators import evaluate_tasks

CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"


def _latest_checkpoint(directory: Path) -> str | None:
    """Return the most-recently modified .pt file in *directory*, or None."""
    pts = sorted(directory.glob("*.pt"), key=lambda p: p.stat().st_mtime)
    return str(pts[-1]) if pts else None

_PARQUETS = {
    "symbolic_comparison": "symbolic_comparison.parquet",
    "match_to_sample": "match_to_sample.parquet",
    "dot_verification": "dot_verification.parquet",
    "panamath": "panamath.parquet",
}

_DS_CLS = {
    "symbolic_comparison": SymbolicComparisonDataset,
    "match_to_sample": MatchToSampleDataset,
    "dot_verification": DotVerificationDataset,
    "panamath": PanamathDataset,
}


def main():
    parser = argparse.ArgumentParser(description="Zero-shot transfer evaluation.")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to .pt checkpoint. Defaults to the latest in checkpoints/.")
    parser.add_argument("--data-dir", required=True, help="Cohort-B data directory")
    parser.add_argument("--tasks", nargs="+", default=["panamath", "symbolic_comparison"])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()

    ckpt_path = args.checkpoint or _latest_checkpoint(CHECKPOINT_DIR)
    if ckpt_path is None:
        parser.error("No checkpoint given and none found in checkpoints/.")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    extra = ckpt.get("extra", {})
    preset = infer_preset(ckpt["model_state_dict"], extra)
    print(f"=== Transfer eval: preset={preset} ===")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Data dir: {args.data_dir}")

    data_dir = Path(args.data_dir)
    task_dfs: dict[str, pd.DataFrame] = {}
    for task in args.tasks:
        p = data_dir / _PARQUETS[task]
        if p.exists():
            task_dfs[task] = pd.read_parquet(p)
    if not task_dfs:
        print("No task data found — aborting.")
        return

    all_ids: set[str] = set()
    for df in task_dfs.values():
        all_ids.update(df["subject_id"].unique())
    subject_map = SubjectIndexMap(sorted(all_ids))
    n_subj = subject_map.n_subjects
    print(f"Transfer subjects: {n_subj}")

    cfg = ExperimentConfig(preset=preset)
    cfg.subject.n_subjects = n_subj
    modules = build_modules(cfg, n_subj, device)
    available = list(task_dfs.keys())
    heads = build_task_heads(cfg, n_subj, available, device)

    # Load only core (non-subject) parameters from checkpoint.
    saved_state = ckpt["model_state_dict"]
    new_state = _CheckpointWrapper(modules, heads).state_dict()
    loadable = {}
    for k, v in saved_state.items():
        if k in new_state and new_state[k].shape == v.shape:
            loadable[k] = v
    _CheckpointWrapper(modules, heads).load_state_dict(loadable, strict=False)

    feat_path = data_dir / "stimulus_features.parquet"
    feature_df = pd.read_parquet(feat_path) if feat_path.exists() else None
    if feature_df is not None and "image_path" in feature_df.columns:
        feature_df = feature_df.set_index("image_path")

    datasets: dict = {}
    for task, df in task_dfs.items():
        kw: dict = {"df": df, "subject_map": subject_map}
        if task != "symbolic_comparison":
            kw["feature_df"] = feature_df
        datasets[task] = _DS_CLS[task](**kw)
        print(f"  {task}: {len(datasets[task])} trials")

    loader = _make_eval_loader(datasets, args.batch_size)
    results = evaluate_tasks(modules, heads, loader, device)

    print(f"\n=== Transfer metrics ===")
    for k, v in sorted(results.items()):
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
