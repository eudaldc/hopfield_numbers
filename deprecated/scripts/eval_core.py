"""Compare trained presets on held-out data with figures and tables.

Loads one checkpoint per preset+mode combination (e.g. default/dynamic,
weber/static), evaluates each on the same data split, and produces
comparison figures plus summary tables.

Usage::

    python scripts/eval_core.py
    python scripts/eval_core.py --mode dynamic
    python scripts/eval_core.py --presets default weber --mode static dynamic
    python scripts/eval_core.py --split val --output-dir output/comparison
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from torch.utils.data import DataLoader

from src.common.config import ExperimentConfig, PRESET_FAMILIES
from src.common.device import get_device
from src.common.io import best_checkpoint_filename
from src.common.seed import set_seed
from src.training.engine import (
    DEFAULT_CORE_TASKS,
    _CheckpointWrapper,
    _make_eval_loader,
    build_modules,
    build_task_heads,
    load_task_data,
)
from src.training.evaluators import compute_kstar_summary, evaluate_tasks
from src.viz import PALETTE, save_figure

CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"
ALL_PRESETS = list(PRESET_FAMILIES.keys())
ALL_MODES = ["dynamic", "static"]

PRESET_LABELS = {
    "default": "Attractor (box-cox)",
    "weber": "Attractor (Weber)",
}
TASK_LABELS = {
    "symbolic_comparison": "Symbolic\nComparison",
    "match_to_sample": "Match-to-\nSample",
    "dot_verification": "Dot\nVerification",
    "panamath": "Panamath",
}
_CORE_PARAM_ORDER = ("rho", "lam", "sigma0", "nu")
_CORE_PARAM_LABELS = {
    "rho": "ρ (rho)",
    "lam": "λ (lambda)",
    "sigma0": "σ₀ (sigma0)",
    "nu": "ν (nu)",
}


# ======================================================================
# Data collection
# ======================================================================


def discover_checkpoints(
    presets: list[str] | None = None,
    modes: list[str] | None = None,
) -> dict[str, Path]:
    """Find available ``best_{preset}_{mode}.pt`` checkpoints."""
    found = {}
    for preset in presets or ALL_PRESETS:
        for mode in modes or ALL_MODES:
            path = CHECKPOINT_DIR / best_checkpoint_filename(preset, mode)
            if path.exists():
                found[f"{preset}/{mode}"] = path
    return found


def evaluate_preset(
    preset: str,
    ckpt_path: Path,
    available_tasks: list[str],
    loader: DataLoader,
    n_subjects: int,
    device: torch.device,
) -> dict:
    """Load a checkpoint and evaluate it on *loader*.

    Returns a dict with keys ``metrics``, ``kstar``, ``core_params``,
    ``available_tasks``, ``n_subjects``, ``epoch``.
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    cfg = ExperimentConfig(preset=preset)
    cfg.subject.n_subjects = n_subjects
    cfg.attractor_mode = ckpt.get("extra", {}).get("attractor_mode", "dynamic")
    modules = build_modules(cfg, n_subjects, device)
    heads = build_task_heads(cfg, n_subjects, available_tasks, device)

    wrapper = _CheckpointWrapper(modules, heads)
    wrapper.load_state_dict(ckpt["model_state_dict"])

    metrics = evaluate_tasks(modules, heads, loader, device)

    kstar: dict[str, float] = {}
    core = modules["attractor_core"]
    if hasattr(core, "estimate_kstar"):
        try:
            kstar = compute_kstar_summary(core, n_subjects)
        except Exception:
            pass

    core_params: dict[str, float] = {}
    for attr in _CORE_PARAM_ORDER:
        if hasattr(core, attr):
            val = getattr(core, attr)
            core_params[attr] = (
                val.item() if isinstance(val, torch.Tensor) else float(val)
            )

    return {
        "metrics": metrics,
        "kstar": kstar,
        "core_params": core_params,
        "available_tasks": available_tasks,
        "n_subjects": n_subjects,
        "epoch": ckpt.get("epoch", "?"),
    }


# ======================================================================
# Plotting helpers
# ======================================================================


def _grouped_bars(
    ax: plt.Axes,
    presets: list[str],
    tasks: list[str],
    values: dict[str, list[float]],
    *,
    ylabel: str,
    title: str,
    ylim: tuple | None = None,
    value_fmt: str = ".3f",
) -> None:
    """Draw a grouped bar chart on *ax*."""
    n_tasks = len(tasks)
    n_presets = len(presets)
    bar_w = 0.8 / n_presets
    x = np.arange(n_tasks)

    for i, p in enumerate(presets):
        offset = (i - n_presets / 2 + 0.5) * bar_w
        bars = ax.bar(
            x + offset,
            values[p],
            bar_w,
            label=p,
            color=PALETTE[i % len(PALETTE)],
            edgecolor="white",
            linewidth=0.8,
        )
        for bar, val in zip(bars, values[p]):
            if not np.isnan(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{val:{value_fmt}}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

    ax.set_xticks(x)
    ax.set_xticklabels([TASK_LABELS.get(t, t) for t in tasks])
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="best", framealpha=0.9)
    if ylim is not None:
        ax.set_ylim(ylim)


def _gather_metric(
    results: dict[str, dict],
    tasks: list[str],
    suffix: str,
) -> dict[str, list[float]]:
    """Extract ``results[preset]["metrics"][task/suffix]`` for every preset x task."""
    return {
        p: [
            results[p]["metrics"].get(f"{t}/{suffix}", float("nan"))
            for t in tasks
        ]
        for p in results
    }


def _style_table_header(table, presets: list[str]) -> None:
    """Apply preset-coloured header to a matplotlib table."""
    for j, p in enumerate(presets):
        cell = table[0, j]
        cell.set_facecolor(
            to_rgba(PALETTE[j % len(PALETTE)], 0.25)
        )
        cell.set_text_props(fontweight="bold")


def _highlight_best(
    table,
    rows: list[list[str]],
    row_labels: list[str],
) -> None:
    """Bold + green tint on the best value in each row."""
    for i, row in enumerate(rows):
        nums: list[float | None] = []
        for v in row:
            try:
                nums.append(float(v))
            except ValueError:
                nums.append(None)
        valid = [(idx, v) for idx, v in enumerate(nums) if v is not None]
        if not valid:
            continue
        lbl = row_labels[i].lower()
        higher_better = "accuracy" in lbl or "k*" in lbl
        best_idx = (max if higher_better else min)(valid, key=lambda x: x[1])[0]
        table[i + 1, best_idx].set_text_props(fontweight="bold")
        table[i + 1, best_idx].set_facecolor(to_rgba(PALETTE[2], 0.15))


# ======================================================================
# Figures
# ======================================================================


def plot_accuracy(
    results: dict[str, dict],
    tasks: list[str],
    save_dir: Path,
) -> None:
    """Grouped bar chart of per-task accuracy."""
    presets = list(results.keys())
    fig, ax = plt.subplots(figsize=(max(7, 2.5 * len(tasks)), 5))
    _grouped_bars(
        ax,
        presets,
        tasks,
        _gather_metric(results, tasks, "acc"),
        ylabel="Accuracy",
        title="Per-Task Accuracy by Preset",
        ylim=(0, 1.05),
    )
    fig.tight_layout()
    save_figure(fig, "compare_accuracy", save_dir=save_dir)


def plot_loss(
    results: dict[str, dict],
    tasks: list[str],
    save_dir: Path,
) -> None:
    """Grouped bar chart of per-task loss."""
    presets = list(results.keys())
    fig, ax = plt.subplots(figsize=(max(7, 2.5 * len(tasks)), 5))
    _grouped_bars(
        ax,
        presets,
        tasks,
        _gather_metric(results, tasks, "loss"),
        ylabel="Loss (NLL)",
        title="Per-Task Loss by Preset",
        value_fmt=".4f",
    )
    fig.tight_layout()
    save_figure(fig, "compare_loss", save_dir=save_dir)


def plot_calibration(
    results: dict[str, dict],
    tasks: list[str],
    save_dir: Path,
) -> None:
    """Two-panel figure: Brier score (top) and ECE (bottom)."""
    presets = list(results.keys())
    fig, (ax_brier, ax_ece) = plt.subplots(
        2, 1, figsize=(max(7, 2.5 * len(tasks)), 9),
    )
    _grouped_bars(
        ax_brier,
        presets,
        tasks,
        _gather_metric(results, tasks, "brier"),
        ylabel="Brier Score",
        title="Brier Score (lower is better)",
    )
    _grouped_bars(
        ax_ece,
        presets,
        tasks,
        _gather_metric(results, tasks, "ece"),
        ylabel="ECE",
        title="Expected Calibration Error (lower is better)",
    )
    fig.tight_layout()
    save_figure(fig, "compare_calibration", save_dir=save_dir)


def plot_radar(
    results: dict[str, dict],
    tasks: list[str],
    save_dir: Path,
) -> bool:
    """Radar chart of per-task accuracy.  Returns False if < 3 tasks."""
    presets = list(results.keys())
    n = len(tasks)
    if n < 3:
        return False

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
    for i, p in enumerate(presets):
        vals = [
            results[p]["metrics"].get(f"{t}/acc", 0.0) for t in tasks
        ]
        vals += vals[:1]
        color = PALETTE[i % len(PALETTE)]
        ax.plot(
            angles, vals, "o-", linewidth=2,
            label=PRESET_LABELS.get(p, p), color=color,
        )
        ax.fill(angles, vals, alpha=0.08, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(
        [TASK_LABELS.get(t, t).replace("\n", " ") for t in tasks],
    )
    ax.set_ylim(0, 1)
    ax.set_title("Accuracy Profile", fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), framealpha=0.9)
    fig.tight_layout()
    save_figure(fig, "compare_radar", save_dir=save_dir)
    return True


# ======================================================================
# Tables
# ======================================================================


def plot_summary_table(
    results: dict[str, dict],
    tasks: list[str],
    split: str,
    save_dir: Path,
) -> None:
    """Full metrics table rendered as a figure."""
    presets = list(results.keys())
    rows: list[list[str]] = []
    row_labels: list[str] = []

    for task in tasks:
        label = TASK_LABELS.get(task, task).replace("\n", " ")
        for suffix, metric in [
            ("acc", "Accuracy"),
            ("loss", "Loss"),
            ("brier", "Brier"),
            ("ece", "ECE"),
        ]:
            row_labels.append(f"{label} — {metric}")
            row = []
            for p in presets:
                v = results[p]["metrics"].get(f"{task}/{suffix}", float("nan"))
                row.append(f"{v:.4f}" if not np.isnan(v) else "—")
            rows.append(row)

    row_labels.append("Total loss")
    rows.append([
        f"{results[p]['metrics'].get('loss_total', float('nan')):.4f}"
        for p in presets
    ])

    if any(results[p]["kstar"] for p in presets):
        for key, label in [
            ("analysis/kstar_mean", "K* mean"),
            ("analysis/kstar_median", "K* median"),
        ]:
            row_labels.append(label)
            rows.append([
                f"{results[p]['kstar'][key]:.2f}"
                if key in results[p]["kstar"]
                else "—"
                for p in presets
            ])

    col_labels = [PRESET_LABELS.get(p, p) for p in presets]
    fig, ax = plt.subplots(
        figsize=(3.5 + 2.2 * len(presets), 1.5 + 0.38 * len(rows)),
    )
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.4)

    _style_table_header(table, presets)
    _highlight_best(table, rows, row_labels)

    ax.set_title(
        f"Model Comparison — {split} split",
        fontweight="bold", pad=20, fontsize=13,
    )
    fig.tight_layout()
    save_figure(fig, "compare_summary_table", save_dir=save_dir)


def plot_params_table(
    results: dict[str, dict],
    save_dir: Path,
) -> None:
    """Learned core parameters rendered as a table figure."""
    presets = list(results.keys())
    all_params: set[str] = set()
    for p in presets:
        all_params.update(results[p]["core_params"].keys())
    params = [p for p in _CORE_PARAM_ORDER if p in all_params]

    rows: list[list[str]] = []
    row_labels: list[str] = []
    for param in params:
        row_labels.append(_CORE_PARAM_LABELS.get(param, param))
        rows.append([
            f"{results[p]['core_params'][param]:.4f}"
            if param in results[p]["core_params"]
            else "—"
            for p in presets
        ])

    row_labels.append("Training epoch")
    rows.append([str(results[p]["epoch"]) for p in presets])
    row_labels.append("N subjects")
    rows.append([str(results[p]["n_subjects"]) for p in presets])
    row_labels.append("Family")
    rows.append([PRESET_FAMILIES.get(p, "?") for p in presets])

    col_labels = [PRESET_LABELS.get(p, p) for p in presets]
    fig, ax = plt.subplots(
        figsize=(3.5 + 2.2 * len(presets), 1.2 + 0.38 * len(rows)),
    )
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        rowLabels=row_labels,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    _style_table_header(table, presets)

    ax.set_title(
        "Learned Core Parameters",
        fontweight="bold", pad=20, fontsize=13,
    )
    fig.tight_layout()
    save_figure(fig, "compare_params_table", save_dir=save_dir)


# ======================================================================
# CLI
# ======================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare trained presets with figures and tables.",
    )
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument(
        "--split", default="test", choices=["train", "val", "test"],
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output-dir", default="output/comparison")
    parser.add_argument(
        "--presets", nargs="+", default=None, choices=ALL_PRESETS,
        help="Presets to compare (default: all with a checkpoint).",
    )
    parser.add_argument(
        "--mode", nargs="+", default=None, choices=ALL_MODES,
        help="Attractor modes to include (default: all with a checkpoint).",
    )
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_CORE_TASKS)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = discover_checkpoints(args.presets, args.mode)
    if not checkpoints:
        parser.error(
            f"No checkpoints found in {CHECKPOINT_DIR}. "
            "Expected files: best_default_dynamic.pt, best_weber_static.pt, etc."
        )

    print(f"=== Preset comparison ({args.split} split) ===")
    print(f"Checkpoints: {list(checkpoints.keys())}")
    print(f"Output dir:  {output_dir}\n")

    train_ds, val_ds, test_ds, subject_map = load_task_data(
        args.data_dir, args.tasks, seed=args.seed,
    )
    ds = {"train": train_ds, "val": val_ds, "test": test_ds}[args.split]
    available = list(ds.keys())
    loader = _make_eval_loader(ds, args.batch_size)

    print(f"Tasks: {available}")
    for t, d in ds.items():
        print(f"  {t}: {len(d)} trials")
    print()

    results: dict[str, dict] = {}
    for combo, ckpt_path in checkpoints.items():
        preset = combo.split("/")[0]
        label = f"{PRESET_LABELS.get(preset, preset)} ({combo.split('/')[1]})"
        print(f"Evaluating {label} …")
        results[combo] = evaluate_preset(
            preset, ckpt_path, available, loader,
            subject_map.n_subjects, device,
        )
        m = results[combo]["metrics"]
        line = f"  epoch={results[combo]['epoch']}"
        line += f"  loss={m.get('loss_total', float('nan')):.4f}"
        for t in available:
            line += f"  {t}/acc={m.get(f'{t}/acc', float('nan')):.3f}"
        print(line)
    print()

    # ── Generate figures ──────────────────────────────────────────────
    print("Generating figures …")
    generated: list[str] = []

    plot_accuracy(results, available, output_dir)
    generated.append("compare_accuracy.png")

    plot_loss(results, available, output_dir)
    generated.append("compare_loss.png")

    plot_calibration(results, available, output_dir)
    generated.append("compare_calibration.png")

    if plot_radar(results, available, output_dir):
        generated.append("compare_radar.png")

    plot_summary_table(results, available, args.split, output_dir)
    generated.append("compare_summary_table.png")

    plot_params_table(results, output_dir)
    generated.append("compare_params_table.png")

    print(f"\nSaved {len(generated)} figures to {output_dir}/")
    for f in generated:
        print(f"  {f}")


if __name__ == "__main__":
    main()
