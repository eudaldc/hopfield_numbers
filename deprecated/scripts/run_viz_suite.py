#!/usr/bin/env python
"""Run the full visualization suite for the Numerical Attractors PoC.

Generates composite panel PNGs — one per plot group (data, stim, model,
subjects) — from data on disk and a trained model checkpoint.  Panels
are saved to ``--plot-dir`` and optionally logged to W&B.

For the model and subject panels the preset name is included in the
panel title and appended to the filename (e.g. ``panel_model_default.png``).

Usage examples::

    # default: model + subject panels only (preset from checkpoint)
    python scripts/run_viz_suite.py

    # explicit preset
    python scripts/run_viz_suite.py --preset weber

    # all panels (data, stim, model, subjects)
    python scripts/run_viz_suite.py --preset default --full

    # specific groups
    python scripts/run_viz_suite.py --groups data stim
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.common.config import ExperimentConfig
from src.common.io import (
    best_checkpoint_filename,
    infer_mode,
    infer_preset,
    sanitize_checkpoint_token,
)
from src.common.device import get_device
from src.common.seed import set_seed
from src.data.schemas import TASK_NAMES

CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"


def _find_checkpoint(
    directory: Path,
    preset: str | None = None,
    mode: str | None = None,
) -> str | None:
    """Find the best checkpoint, falling back to most recent.

    Resolution order when *preset* (and optionally *mode*) is given:
      1. Canonical name ``best_{preset}_{mode}.pt``
      2. Any ``.pt`` whose stem contains *preset* (newest first)
      3. Most-recently modified ``.pt`` in *directory* (fallback)
    """
    if preset:
        canonical = directory / best_checkpoint_filename(preset, mode or "dynamic")
        if canonical.exists():
            return str(canonical)
        candidates = sorted(
            (p for p in directory.glob("*.pt") if preset in p.stem),
            key=lambda p: p.stat().st_mtime,
        )
        if candidates:
            return str(candidates[-1])
    pts = sorted(directory.glob("*.pt"), key=lambda p: p.stat().st_mtime)
    return str(pts[-1]) if pts else None


# ── CLI ──────────────────────────────────────────────────────────────

def make_parser() -> argparse.ArgumentParser:
    from src.common.config import PRESET_FAMILIES

    p = argparse.ArgumentParser(description="Run visualization suite.")
    p.add_argument("--data-dir", default="data/processed",
                   help="Directory with task parquets and subjects.parquet")
    p.add_argument("--checkpoint", default=None,
                   help="Path to a model checkpoint (.pt). "
                        "Defaults to the latest .pt in checkpoints/.")
    p.add_argument("--preset", default=None,
                   choices=list(PRESET_FAMILIES),
                   help="Model preset. If omitted, read from the checkpoint metadata.")
    p.add_argument("--mode", default=None, choices=["dynamic", "static"],
                   help="Attractor mode for checkpoint lookup.")
    p.add_argument("--plot-dir", default="output",
                   help="Root output directory for plots")
    p.add_argument("--groups", nargs="*", default=None,
                   help="Plot groups to run (data, stim, model, tasks, ppc, compare, "
                        "subjects, dynamic). Default: all available.")
    p.add_argument("--full", action="store_true", default=False,
                   help="Run all plot groups including dynamic attractor diagnostics.")
    p.add_argument("--wandb", action="store_true", default=False)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--dynamic-n-samples", type=int, default=5_000,
                   help="Samples for dynamic attractor comparison plots (default: 5000)")
    return p


# ── data loaders ─────────────────────────────────────────────────────

def _load_parquet_safe(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_parquet(path)
    return None


def load_all_data(data_dir: Path) -> tuple[
    pd.DataFrame | None,
    dict[str, pd.DataFrame],
    pd.DataFrame | None,
]:
    subjects = _load_parquet_safe(data_dir / "subjects.parquet")

    _files = {
        "symbolic_comparison": "symbolic_comparison.parquet",
        "match_to_sample": "match_to_sample.parquet",
        "dot_verification": "dot_verification.parquet",
        "panamath": "panamath.parquet",
    }
    task_dfs: dict[str, pd.DataFrame] = {}
    for task, fname in _files.items():
        df = _load_parquet_safe(data_dir / fname)
        if df is not None:
            task_dfs[task] = df

    features = _load_parquet_safe(data_dir / "stimulus_features.parquet")
    return subjects, task_dfs, features


# ── model loader ─────────────────────────────────────────────────────

def load_model(
    ckpt_path: str,
    preset: str,
    device: torch.device,
):
    """Load modules and heads from a checkpoint.

    Returns ``(modules, heads, cfg, ckpt_preset)`` where *ckpt_preset*
    is the preset string stored in the checkpoint metadata (or ``None``).
    """
    from src.training.engine import build_modules, build_task_heads

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    extra = ckpt.get("extra", {})
    preset = infer_preset(ckpt["model_state_dict"], extra)

    cfg = ExperimentConfig(preset=preset)

    state = ckpt["model_state_dict"]
    sp_keys = [k for k in state if k.startswith("core.subject_params.")]
    n_subjects = 1
    for k in sp_keys:
        if "weight" in k:
            n_subjects = state[k].shape[0]
            break

    cfg.subject.n_subjects = n_subjects

    tasks_in_ckpt = [
        t for t in TASK_NAMES
        if any(k.startswith(f"heads.{t}.") for k in state)
    ]

    modules = build_modules(cfg, n_subjects, device)
    heads = build_task_heads(cfg, n_subjects, tasks_in_ckpt, device)

    from src.training.engine import _CheckpointWrapper
    wrapper = _CheckpointWrapper(modules, heads)
    wrapper.load_state_dict(state)

    ckpt_preset = extra.get("preset")
    return modules, heads, cfg, ckpt_preset


# ── W&B helper ───────────────────────────────────────────────────────

def maybe_init_wandb(
    use_wandb: bool,
    plot_dir: str,
    preset: str | None = None,
):
    if not use_wandb:
        return None
    try:
        import wandb

        tags = ["viz"]
        cfg: dict = {"plot_dir": plot_dir}
        if preset:
            tags.append(preset)
            cfg["preset"] = preset
        run = wandb.init(
            project="numeric-attractors-poc",
            job_type="viz",
            tags=tags,
            config=cfg,
            reinit=True,
        )
        return run
    except Exception:
        return None


# ── group runners ────────────────────────────────────────────────────

def run_data_plots(subjects, task_dfs, save_dir, wandb_run):
    from src.viz.plots_data import (
        plot_subject_counts,
        plot_missingness_heatmap,
        plot_trials_per_child,
        plot_rt_distributions,
        plot_accuracy_by_task_grade,
    )
    from src.viz import compose_panel, save_figure

    figs = []
    if subjects is not None and task_dfs:
        print("  [data] subject counts")
        figs.append(plot_subject_counts(subjects, task_dfs))
    if subjects is not None:
        print("  [data] missingness heatmap")
        figs.append(plot_missingness_heatmap(subjects))
    if task_dfs:
        print("  [data] trials per child")
        figs.append(plot_trials_per_child(task_dfs))
        print("  [data] RT distributions")
        figs.append(plot_rt_distributions(task_dfs))
        print("  [data] accuracy by task × grade")
        figs.append(plot_accuracy_by_task_grade(task_dfs))

    if figs:
        panel = compose_panel(figs, suptitle="Data QC")
        save_figure(panel, "panel_data", save_dir, wandb_run)


def run_stim_plots(feature_df, save_dir, wandb_run):
    from src.viz.plots_model import (
        plot_feature_correlations,
        plot_feature_distributions,
    )
    from src.viz import compose_panel, save_figure

    if feature_df is None:
        print("  [stim] no stimulus features — skipping")
        return

    figs = []
    print("  [stim] feature correlations")
    figs.append(plot_feature_correlations(feature_df))
    print("  [stim] feature distributions")
    figs.append(plot_feature_distributions(feature_df))

    panel = compose_panel(figs, suptitle="Stimulus diagnostics", max_cols=2)
    save_figure(panel, "panel_stim", save_dir, wandb_run)


def run_model_plots(
    modules,
    n_subjects,
    save_dir,
    wandb_run,
    run_label: str | None = None,
    wandb_prefix: str | None = None,
    file_suffix: str | None = None,
):
    from src.viz.plots_model import (
        plot_centers,
        plot_widths,
        plot_separation,
        plot_precision,
        plot_posterior_heatmap,
        plot_energy_landscape,
        plot_kstar_distribution,
        plot_wells_gaussian,
    )
    from src.viz import compose_panel, save_figure

    core = modules["attractor_core"]
    figs = []

    print("  [model] Gaussian wells")
    figs.append(plot_wells_gaussian(core))
    print("  [model] centres")
    figs.append(plot_centers(core))
    print("  [model] widths")
    figs.append(plot_widths(core))
    print("  [model] precision")
    figs.append(plot_precision(core))
    print("  [model] separation")
    figs.append(plot_separation(core))
    print("  [model] posterior heatmap")
    figs.append(plot_posterior_heatmap(core))
    print("  [model] energy landscape")
    figs.append(plot_energy_landscape(core))
    print("  [model] K* distribution")
    figs.append(plot_kstar_distribution(core, n_subjects))

    title = "Model geometry"
    if run_label:
        title += f" — {run_label}"
    panel = compose_panel(figs, suptitle=title)
    save_figure(
        panel, "panel_model", save_dir, wandb_run,
        wandb_prefix=wandb_prefix,
        file_suffix=file_suffix,
    )


def run_subject_plots(
    modules,
    n_subjects,
    subjects,
    save_dir,
    wandb_run,
    run_label: str | None = None,
    wandb_prefix: str | None = None,
    file_suffix: str | None = None,
    data_dir: Path | None = None,
    seed: int = 17,
):
    from src.training.evaluators import export_subject_params
    from src.viz.plots_subjects import (
        plot_subject_param_histograms,
        plot_subject_param_scattermatrix,
        plot_all_param_vs_downstream,
        plot_kstar_vs_grade,
    )
    from src.viz.subject_demographics import attach_demographics
    from src.viz import compose_panel, save_figure

    core = modules["attractor_core"]
    sp = modules["subject_params"]
    param_df = export_subject_params(core, sp, n_subjects)

    if data_dir is not None:
        param_df, demo_msgs = attach_demographics(
            param_df, n_subjects, subjects, data_dir, seed=seed
        )
        for line in demo_msgs:
            print(f"    {line}")
    elif subjects is not None:
        print(
            "    (No data_dir for subject alignment; grade/downstream plots may be empty.)"
        )

    figs = []

    print("  [subjects] parameter histograms")
    figs.append(plot_subject_param_histograms(param_df))
    print("  [subjects] scatterplot matrix")
    figs.append(plot_subject_param_scattermatrix(param_df))

    _downstream_scores = (
        "arithmetic_score",
        "number_line_score",
        "raven_score",
        "wm_score",
    )
    has_downstream = any(
        c in param_df.columns and param_df[c].notna().any()
        for c in _downstream_scores
    )
    if has_downstream:
        print("  [subjects] params vs downstream")
        figs.append(plot_all_param_vs_downstream(param_df))
    else:
        print(
            "  [subjects] params vs downstream — skipped "
            "(need cognitive columns merged from subjects.parquet)"
        )

    if (
        "kstar" in param_df.columns
        and "grade" in param_df.columns
        and param_df["grade"].notna().any()
    ):
        print("  [subjects] K* vs grade")
        figs.append(plot_kstar_vs_grade(param_df))
    else:
        print(
            "  [subjects] K* vs grade — skipped "
            "(need grade merged from subjects.parquet via task subject index)"
        )

    if figs:
        title = "Subject analysis"
        if run_label:
            title += f" — {run_label}"
        panel = compose_panel(figs, suptitle=title, max_cols=2)
        save_figure(
            panel, "panel_subj", save_dir, wandb_run,
            wandb_prefix=wandb_prefix,
            file_suffix=file_suffix,
        )


# ── dynamic attractor group (WP-6) ───────────────────────────────────

def run_dynamic_plots(
    save_dir: str,
    wandb_run,
    n_samples: int = 5_000,
    seed: int = 17,
) -> None:
    """Generate WP-6 dynamic attractor diagnostic plots.

    Does not require a trained checkpoint — runs with default model parameters.
    Produces: cleanup map, basin map, trajectories, RT curves (dynamic vs static),
    and a comparison panel.
    """
    import torch
    from src.models.attractor_dynamics import DynamicAttractorCore
    from src.models.rt_proxy import RTProxy
    from src.viz.plots_attractor import (
        plot_cleanup_map,
        plot_basin_map,
        plot_trajectories,
        plot_rt_curves,
        plot_comparison_panel,
        evaluate_comparison,
    )
    from src.viz import compose_panel, save_figure

    device = torch.device("cpu")

    core_static = DynamicAttractorCore(cleanup_mode="none").to(device).eval()
    core_dynamic = DynamicAttractorCore(cleanup_mode="gradient").to(device).eval()
    rt_proxy = RTProxy(mode="steps_plus_margin").to(device).eval()

    figs = []

    print("  [dynamic] cleanup map")
    figs.append(
        plot_cleanup_map(
            core_dynamic, save_dir=save_dir, wandb_run=wandb_run,
            wandb_prefix="attractor_core",
        )
    )

    print("  [dynamic] basin map")
    figs.append(
        plot_basin_map(
            core_dynamic, save_dir=save_dir, wandb_run=wandb_run,
            wandb_prefix="attractor_core",
        )
    )

    print("  [dynamic] trajectories")
    figs.append(
        plot_trajectories(
            core_dynamic, save_dir=save_dir, wandb_run=wandb_run,
            wandb_prefix="attractor_core",
        )
    )

    modes = {
        "static (ablation)": (core_static, rt_proxy),
        "dynamic": (core_dynamic, rt_proxy),
    }

    print("  [dynamic] RT curves")
    results = evaluate_comparison(
        modes, n_samples=n_samples, seed=seed, device=device,
    )
    figs.append(
        plot_rt_curves(
            results, save_dir=save_dir, wandb_run=wandb_run,
            wandb_prefix="synthetic_regimes",
        )
    )

    print("  [dynamic] comparison panel")
    cores_for_maps = {
        "static (ablation)": core_static,
        "dynamic": core_dynamic,
    }
    figs.append(
        plot_comparison_panel(
            results, cores=cores_for_maps,
            save_dir=save_dir, wandb_run=wandb_run,
            wandb_prefix="attractor_core",
        )
    )

    import matplotlib.pyplot as plt
    panel = compose_panel(
        [f for f in figs if f is not None],
        suptitle="Dynamic attractor diagnostics",
    )
    save_figure(panel, "panel_dynamic", save_dir, wandb_run)
    for f in figs:
        if f is not None:
            plt.close(f)


# ── main ─────────────────────────────────────────────────────────────

def main():
    parser = make_parser()
    args = parser.parse_args()
    set_seed(args.seed)
    device = get_device()

    plot_dir = Path(args.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    if args.groups:
        groups = set(args.groups)
    elif args.full:
        groups = {"data", "stim", "model", "tasks", "ppc", "compare", "subjects", "dynamic"}
    else:
        groups = {"model", "subjects"}

    data_dir = Path(args.data_dir)
    subjects, task_dfs, feature_df = load_all_data(data_dir)

    _GROUPS_NEEDING_CHECKPOINT = {"model", "subjects", "tasks", "ppc", "compare"}
    needs_checkpoint = bool(groups & _GROUPS_NEEDING_CHECKPOINT)

    ckpt_path = None
    if needs_checkpoint:
        ckpt_path = args.checkpoint or _find_checkpoint(
            CHECKPOINT_DIR, args.preset, getattr(args, "mode", None),
        )

    modules = heads = cfg = None
    n_subjects = 1
    ckpt_preset: str | None = None
    if ckpt_path:
        print(f"Loading checkpoint: {ckpt_path}")
        modules, heads, cfg, ckpt_preset = load_model(
            ckpt_path, args.preset or "default", device
        )
        n_subjects = cfg.subject.n_subjects
        if ckpt_preset:
            print(f"Checkpoint preset (metadata): {ckpt_preset}")

    needs_model = (
        ("model" in groups or "subjects" in groups) and modules is not None
    )
    preset = args.preset or ckpt_preset
    if needs_model and not preset:
        parser.error(
            "When plotting model/subjects, pass --preset NAME "
            "(e.g. default, weber), or use a checkpoint "
            "that stores the preset in its metadata."
        )
    if args.preset and ckpt_preset and args.preset != ckpt_preset:
        print(
            f"Note: --preset {args.preset!r} overrides "
            f"checkpoint metadata {ckpt_preset!r} for plot labels and filenames."
        )

    file_suffix = (
        sanitize_checkpoint_token(preset) if needs_model and preset else None
    )
    plot_run_label = f"preset: {preset}" if needs_model and preset else None
    plot_wandb_prefix = file_suffix

    wandb_run = maybe_init_wandb(
        args.wandb,
        str(plot_dir),
        preset=preset if needs_model else None,
    )

    # -- H1: Data QC ------------------------------------------------
    if "data" in groups:
        print("\n=== H1  Data QC ===")
        run_data_plots(subjects, task_dfs, str(plot_dir), wandb_run)

    # -- H2: Stimulus diagnostics -----------------------------------
    if "stim" in groups:
        print("\n=== H2  Stimulus diagnostics ===")
        run_stim_plots(feature_df, str(plot_dir), wandb_run)

    # -- H3: Model geometry -----------------------------------------
    if "model" in groups and modules is not None:
        print("\n=== H3  Model geometry ===")
        run_model_plots(
            modules,
            n_subjects,
            str(plot_dir),
            wandb_run,
            run_label=plot_run_label,
            wandb_prefix=plot_wandb_prefix,
            file_suffix=file_suffix,
        )

    # -- H7: Subject analysis (before task/ppc because it exports params)
    if "subjects" in groups and modules is not None:
        print("\n=== H7  Subject-level analysis ===")
        run_subject_plots(
            modules,
            n_subjects,
            subjects,
            str(plot_dir),
            wandb_run,
            run_label=plot_run_label,
            wandb_prefix=plot_wandb_prefix,
            file_suffix=file_suffix,
            data_dir=data_dir,
            seed=args.seed,
        )

    # -- H4/H5/H6: task fits, PPC, model comparison are meaningful
    #    only when predictions are produced.  Print a note for now.
    for grp in ("tasks", "ppc", "compare"):
        if grp in groups and modules is None:
            print(f"\n=== {grp.upper()} — skipped (no checkpoint loaded) ===")
        elif grp in groups:
            print(f"\n=== {grp.upper()} — run eval scripts first, "
                  "then pass prediction CSVs (future extension) ===")

    # -- Dynamic attractor diagnostics (WP-6) -----------------------
    if "dynamic" in groups:
        print("\n=== Dynamic attractor diagnostics ===")
        run_dynamic_plots(
            str(plot_dir),
            wandb_run,
            n_samples=args.dynamic_n_samples,
            seed=args.seed,
        )

    # -- finalise W&B -----------------------------------------------
    if wandb_run is not None:
        try:
            import wandb
            wandb.finish()
        except Exception:
            pass

    print(f"\nPanels saved under {plot_dir.resolve()}")
    if file_suffix:
        print(f"  Model / subject panels use suffix '_{file_suffix}.png'.")


if __name__ == "__main__":
    main()
