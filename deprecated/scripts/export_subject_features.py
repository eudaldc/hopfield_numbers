"""Export learned subject-level parameters from a trained model.

Produces a parquet / CSV with per-subject: temperature, width_scale,
tau_sym, count_bias, and K*.  Useful for downstream analyses that
predict arithmetic or number-line performance.

Usage::

    python scripts/export_subject_features.py \\
        --checkpoint checkpoints/best_default_dynamic.pt \\
        --output subject_features.parquet

    python scripts/export_subject_features.py \\
        --checkpoint checkpoints/best_default_dynamic.pt --q 0.80 --format csv
"""

import argparse
from pathlib import Path

import torch

from src.common.device import get_device
from src.common.seed import set_seed
from src.common.config import ExperimentConfig
from src.training.engine import (
    build_modules,
    build_task_heads,
    load_task_data,
    _CheckpointWrapper,
    DEFAULT_CORE_TASKS,
)
from src.training.evaluators import export_subject_params, compute_kstar_summary

CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"


def _latest_checkpoint(directory: Path) -> str | None:
    """Return the most-recently modified .pt file in *directory*, or None."""
    pts = sorted(directory.glob("*.pt"), key=lambda p: p.stat().st_mtime)
    return str(pts[-1]) if pts else None


def main():
    parser = argparse.ArgumentParser(
        description="Export subject-level learned parameters."
    )
    parser.add_argument("--checkpoint", default=None,
                        help="Path to .pt checkpoint. Defaults to the latest in checkpoints/.")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--output", default="subject_features.parquet")
    parser.add_argument("--q", type=float, default=0.75, help="K* probability threshold")
    parser.add_argument("--format", default="parquet", choices=["parquet", "csv"])
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_CORE_TASKS)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()

    ckpt_path = args.checkpoint or _latest_checkpoint(CHECKPOINT_DIR)
    if ckpt_path is None:
        parser.error("No checkpoint given and none found in checkpoints/.")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    preset = ckpt.get("extra", {}).get("preset", "default")
    print(f"=== Exporting subject features (preset={preset}) ===")
    print(f"Checkpoint: {ckpt_path}")

    _train, _val, _test, subject_map = load_task_data(
        args.data_dir, args.tasks, seed=args.seed,
    )
    n_subj = subject_map.n_subjects
    print(f"Subjects: {n_subj}")

    cfg = ExperimentConfig(preset=preset)
    cfg.subject.n_subjects = n_subj
    modules = build_modules(cfg, n_subj, device)
    available = list(_train.keys())
    heads = build_task_heads(cfg, n_subj, available, device)

    wrapper = _CheckpointWrapper(modules, heads)
    wrapper.load_state_dict(ckpt["model_state_dict"])

    core = modules["attractor_core"]
    sp = modules["subject_params"]

    df = export_subject_params(core, sp, n_subj, q=args.q)

    # Add subject_id mapping
    df["subject_id"] = [subject_map.to_id(i) for i in range(n_subj)]

    out_path = Path(args.output)
    if args.format == "csv":
        out_path = out_path.with_suffix(".csv")
        df.to_csv(out_path, index=False)
    else:
        out_path = out_path.with_suffix(".parquet")
        df.to_parquet(out_path, index=False)

    print(f"Saved {len(df)} subjects to {out_path}")

    kstar = compute_kstar_summary(core, n_subj, q=args.q)
    print(f"\nK* summary (q={args.q}):")
    for k, v in kstar.items():
        print(f"  {k}: {v:.2f}")

    print(f"\nParameter ranges:")
    for col in ["temperature", "width_scale", "tau_sym", "count_bias", "kstar"]:
        print(f"  {col}: [{df[col].min():.3f}, {df[col].max():.3f}] (mean={df[col].mean():.3f})")


if __name__ == "__main__":
    main()
