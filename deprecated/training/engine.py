"""Training engine: model construction, training loop, W&B integration.

Central module for WP-G.  Provides:

- ``build_modules`` / ``build_task_heads``  — model factory per preset
- ``load_task_data``  — parquet → train / val / test dataset dicts
- ``Trainer``  — multitask training with early stopping, checkpointing, W&B
- ``make_parser`` / ``run_training``  — CLI and entry point for ``scripts/train.py``
"""

import argparse
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.common.config import (
    DEFAULT_CONFIG_PATH,
    ExperimentConfig,
    SubjectConfig,
    TrainingConfig,
    load_config,
)
from src.common.device import get_device
from src.common.io import best_checkpoint_filename, ensure_dir, save_checkpoint
from src.common.logging import get_logger
from src.common.seed import set_seed
from src.data.collate import multitask_collate
from src.data.dataset_builders import (
    SubjectIndexMap,
    SymbolicComparisonDataset,
    MatchToSampleDataset,
    DotVerificationDataset,
    PanamathDataset,
    MultitaskDataset,
    build_dataloader,
)
from src.data.splits import subject_split, filter_by_subjects
from src.models.attractor_core import AttractorCore
from src.models.attractor_dynamics import DynamicAttractorCore
from src.models.subject_params import SubjectParams
from src.models.symbolic_head import SymbolicHead
from src.models.task_heads import (
    DotVerificationHead,
    MatchToSampleHead,
    PanamathHead,
    SymbolicComparisonHead,
)
from src.models.visual_encoder import build_visual_encoder
from src.training.losses import total_loss

log = get_logger(__name__)

DEFAULT_CORE_TASKS = TrainingConfig().tasks

_TASK_PARQUETS = {
    "symbolic_comparison": "symbolic_comparison.parquet",
    "match_to_sample": "match_to_sample.parquet",
    "dot_verification": "dot_verification.parquet",
    "panamath": "panamath.parquet",
}

_DATASET_CLASSES = {
    "symbolic_comparison": SymbolicComparisonDataset,
    "match_to_sample": MatchToSampleDataset,
    "dot_verification": DotVerificationDataset,
    "panamath": PanamathDataset,
}


# ======================================================================
# Helpers
# ======================================================================


def _to_device(x, device: torch.device, non_blocking: bool = False):
    """Recursively move tensors in nested dicts to *device*."""
    if isinstance(x, torch.Tensor):
        return x.to(device, non_blocking=non_blocking)
    if isinstance(x, dict):
        return {k: _to_device(v, device, non_blocking) for k, v in x.items()}
    return x


def collect_parameters(
    modules: dict[str, nn.Module],
    heads: dict[str, nn.Module],
) -> list[nn.Parameter]:
    """Return a deduplicated list of all trainable parameters."""
    seen: set[int] = set()
    params: list[nn.Parameter] = []
    for mod in list(modules.values()) + list(heads.values()):
        for p in mod.parameters():
            pid = id(p)
            if pid not in seen:
                seen.add(pid)
                params.append(p)
    return params


class _CheckpointWrapper(nn.Module):
    """Thin ``nn.Module`` wrapper for clean checkpoint serialisation.

    Only the visual encoder, core (which owns subject-params), and task
    heads are stored — ``SymbolicHead`` has no own parameters.
    The optional ``dynamic_core`` is included when attractor_mode="dynamic".
    """

    def __init__(
        self,
        modules: dict[str, nn.Module],
        heads: dict[str, nn.Module],
    ):
        super().__init__()
        self.ve = modules["visual_encoder"]
        self.core = modules["attractor_core"]
        self.heads = nn.ModuleDict(heads)
        optional: dict[str, nn.Module] = {}
        if "dynamic_core" in modules:
            optional["dynamic_core"] = modules["dynamic_core"]
        self.optional = nn.ModuleDict(optional)


class MetricAccumulator:
    """Running weighted-average tracker for training / validation metrics."""

    def __init__(self):
        self._totals: dict[str, float] = {}
        self._counts: dict[str, int] = {}

    def update(self, metrics: dict, weight: int = 1):
        for k, v in metrics.items():
            val = v.item() if isinstance(v, torch.Tensor) else float(v)
            self._totals[k] = self._totals.get(k, 0.0) + val * weight
            self._counts[k] = self._counts.get(k, 0) + weight

    def compute(self) -> dict[str, float]:
        return {
            k: self._totals[k] / self._counts[k]
            for k in self._totals
            if self._counts[k] > 0
        }


# ======================================================================
# Model construction
# ======================================================================


def build_modules(
    cfg: ExperimentConfig,
    n_subjects: int,
    device: torch.device,
) -> dict[str, nn.Module]:
    """Instantiate visual encoder + core + symbolic head for *cfg.preset*."""
    from src.common.config import PRESET_FAMILIES

    sub_cfg = SubjectConfig(
        n_subjects=n_subjects,
        temp_init_mean=cfg.subject.temp_init_mean,
        width_init_std=cfg.subject.width_init_std,
        tau_sym_init=cfg.subject.tau_sym_init,
        l2_penalty=cfg.subject.l2_penalty,
    )
    sp = SubjectParams(sub_cfg)
    ve = build_visual_encoder(cfg.visual)

    fam = PRESET_FAMILIES.get(cfg.preset)
    if fam != "attractor":
        raise ValueError(f"Unknown preset: {cfg.preset!r}")
    core = AttractorCore(cfg.attractor, sp)

    sh = SymbolicHead(core)

    modules: dict[str, nn.Module] = {
        "visual_encoder": ve,
        "attractor_core": core,
        "symbolic_head": sh,
        "subject_params": sp,
    }

    if getattr(cfg, "attractor_mode", "dynamic") == "dynamic":
        dyn_cfg = cfg.dynamic_attractor
        dynamic_core = DynamicAttractorCore(
            support_min=dyn_cfg.support_min,
            support_max=dyn_cfg.support_max,
            rho_init=dyn_cfg.rho_init,
            lambda_init=dyn_cfg.lambda_init,
            sigma0_init=dyn_cfg.sigma0_init,
            nu_init=dyn_cfg.nu_init,
            tau_in_init=dyn_cfg.tau_in_init,
            kappa_init=dyn_cfg.kappa_init,
            eta_init=dyn_cfg.eta_init,
            cleanup_mode=dyn_cfg.cleanup_mode,
            max_steps=dyn_cfg.max_steps,
            tol=dyn_cfg.tol,
        )
        modules["dynamic_core"] = dynamic_core

    for m in modules.values():
        m.to(device)
    return modules


def build_task_heads(
    cfg: ExperimentConfig,
    n_subjects: int,
    tasks: list[str],
    device: torch.device,
) -> dict[str, nn.Module]:
    """Instantiate task heads for the requested tasks."""
    heads: dict[str, nn.Module] = {}
    if "symbolic_comparison" in tasks:
        heads["symbolic_comparison"] = SymbolicComparisonHead()
    if "match_to_sample" in tasks:
        heads["match_to_sample"] = MatchToSampleHead()
    if "dot_verification" in tasks:
        heads["dot_verification"] = DotVerificationHead(
            cfg.dot_verification, n_subjects
        )
    if "panamath" in tasks:
        heads["panamath"] = PanamathHead()
    for h in heads.values():
        h.to(device)
    return heads


# ======================================================================
# Data loading
# ======================================================================


def load_task_data(
    data_dir: str | Path,
    tasks: list[str],
    seed: int = 17,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> tuple[dict, dict, dict, SubjectIndexMap]:
    """Load parquets and build train / val / test dataset dicts.

    Returns
    -------
    train_datasets, val_datasets, test_datasets : {task_name: Dataset}
    subject_map : SubjectIndexMap
    """
    data_dir = Path(data_dir)

    task_dfs: dict[str, pd.DataFrame] = {}
    for task in tasks:
        path = data_dir / _TASK_PARQUETS[task]
        if path.exists():
            task_dfs[task] = pd.read_parquet(path)

    if not task_dfs:
        raise FileNotFoundError(
            f"No task parquet files found in {data_dir} for: {tasks}"
        )

    all_ids: set[str] = set()
    for df in task_dfs.values():
        all_ids.update(df["subject_id"].unique())
    subject_map = SubjectIndexMap(sorted(all_ids))

    split = subject_split(
        np.array(subject_map.all_ids),
        train_frac=train_frac,
        val_frac=val_frac,
        seed=seed,
    )

    feat_path = data_dir / "stimulus_features.parquet"
    feature_df: pd.DataFrame | None = None
    if feat_path.exists():
        feature_df = pd.read_parquet(feat_path)
        if "image_path" in feature_df.columns:
            feature_df = feature_df.set_index("image_path")

    def _build(subject_ids):
        datasets: dict = {}
        for task, df in task_dfs.items():
            filt = filter_by_subjects(df, subject_ids)
            if len(filt) == 0:
                continue
            kw: dict = {"df": filt, "subject_map": subject_map}
            if task != "symbolic_comparison":
                kw["feature_df"] = feature_df
            datasets[task] = _DATASET_CLASSES[task](**kw)
        return datasets

    return (
        _build(split["train"]),
        _build(split["val"]),
        _build(split["test"]),
        subject_map,
    )


def _make_eval_loader(
    datasets: dict,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """Deterministic DataLoader (no shuffle) for evaluation."""
    mt = MultitaskDataset(datasets)
    persistent = num_workers > 0
    return DataLoader(
        mt,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
        prefetch_factor=2 if num_workers > 0 else None,
        collate_fn=multitask_collate,
    )


# ======================================================================
# Trainer
# ======================================================================


class Trainer:
    """Multitask training with early stopping, checkpointing, and W&B."""

    def __init__(
        self,
        cfg: ExperimentConfig,
        modules: dict[str, nn.Module],
        heads: dict[str, nn.Module],
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        use_wandb: bool = True,
    ):
        self.cfg = cfg
        self.modules = modules
        self.heads = heads
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.use_wandb = use_wandb

        self.optimizer = torch.optim.AdamW(
            collect_parameters(modules, heads),
            lr=cfg.training.lr,
            weight_decay=cfg.training.weight_decay,
        )

        self.ckpt_dir = ensure_dir(cfg.training.checkpoint_dir)
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.epoch = 0
        self._interrupted = False

    # ------------------------------------------------------------------
    # mode switches
    # ------------------------------------------------------------------

    def _set_train(self):
        for m in list(self.modules.values()) + list(self.heads.values()):
            m.train()

    def _set_eval(self):
        for m in list(self.modules.values()) + list(self.heads.values()):
            m.eval()

    # ------------------------------------------------------------------
    # one epoch
    # ------------------------------------------------------------------

    def train_epoch(self) -> dict[str, float]:
        self._set_train()
        self._interrupted = False
        use_pinned = getattr(self.train_loader, "pin_memory", False)
        acc = MetricAccumulator()
        pbar = tqdm(self.train_loader, desc="  train", leave=False)

        try:
            for batch in pbar:
                batch = _to_device(batch, self.device, non_blocking=use_pinned)
                task_outputs: dict[str, dict] = {}
                n_samples = 0

                for task_name, task_batch in batch.items():
                    if task_name not in self.heads:
                        continue
                    out = self.heads[task_name](task_batch, self.modules)
                    task_outputs[task_name] = out
                    bs = task_batch["label"].shape[0] if "label" in task_batch else 1
                    n_samples += bs
                    acc.update({f"train/{task_name}/loss": out["loss"]}, weight=bs)
                    for mk, mv in out["metrics"].items():
                        acc.update({f"train/{task_name}/{mk}": mv}, weight=bs)

                if not task_outputs:
                    continue

                loss, _per_task, penalty = total_loss(
                    task_outputs,
                    self.cfg.training.task_weights,
                    self.modules["subject_params"],
                )

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                acc.update(
                    {"train/loss_total": loss, "train/l2_penalty": penalty},
                    weight=n_samples,
                )
                pbar.set_postfix(loss=f"{loss.item():.4f}")
        except KeyboardInterrupt:
            pbar.close()
            self._interrupted = True
            log.warning(
                "Ctrl+C caught — finishing epoch early, " "running validation & saving…"
            )

        return acc.compute()

    # ------------------------------------------------------------------
    # evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(
        self,
        loader: DataLoader,
        prefix: str = "val",
    ) -> dict[str, float]:
        self._set_eval()
        use_pinned = getattr(loader, "pin_memory", False)
        acc = MetricAccumulator()

        for batch in tqdm(loader, desc=f"  {prefix}", leave=False):
            batch = _to_device(batch, self.device, non_blocking=use_pinned)
            task_outputs: dict[str, dict] = {}
            n_samples = 0

            for task_name, task_batch in batch.items():
                if task_name not in self.heads:
                    continue
                out = self.heads[task_name](task_batch, self.modules)
                task_outputs[task_name] = out
                bs = task_batch["label"].shape[0] if "label" in task_batch else 1
                n_samples += bs
                acc.update({f"{prefix}/{task_name}/loss": out["loss"]}, weight=bs)
                for mk, mv in out["metrics"].items():
                    acc.update({f"{prefix}/{task_name}/{mk}": mv}, weight=bs)

            if task_outputs:
                loss, _, _ = total_loss(
                    task_outputs,
                    self.cfg.training.task_weights,
                    self.modules["subject_params"],
                )
                acc.update({f"{prefix}/loss_total": loss}, weight=n_samples)

        return acc.compute()

    # ------------------------------------------------------------------
    # checkpointing
    # ------------------------------------------------------------------

    def _save_best(self, metrics: dict):
        wrapper = _CheckpointWrapper(self.modules, self.heads)
        mode = getattr(self.cfg, "attractor_mode", "dynamic")
        ckpt_name = best_checkpoint_filename(self.cfg.preset, mode)
        save_checkpoint(
            self.ckpt_dir / ckpt_name,
            wrapper,
            self.optimizer,
            epoch=self.epoch,
            metrics=metrics,
            extra={
                "preset": self.cfg.preset,
                "attractor_mode": mode,
            },
        )

    def load_checkpoint(self, path: str | Path):
        """Restore model + optimizer from a checkpoint file."""
        wrapper = _CheckpointWrapper(self.modules, self.heads)
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        wrapper.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.epoch = ckpt.get("epoch", 0)

    # ------------------------------------------------------------------
    # W&B helpers
    # ------------------------------------------------------------------

    def _init_wandb(self):
        if not self.use_wandb:
            return
        try:
            import wandb

            wandb.init(
                project=self.cfg.training.wandb_project,
                config=asdict(self.cfg),
                group=self.cfg.preset,
                tags=[
                    self.cfg.preset,
                    self.cfg.family,
                    f"cohort_{self.cfg.cohort}",
                ],
                reinit=True,
            )
            wandb.define_metric("epoch")
            wandb.define_metric("train/*", step_metric="epoch")
            wandb.define_metric("val/*", step_metric="epoch")
            wandb.define_metric("analysis/*", step_metric="epoch")
            wandb.define_metric("lr", step_metric="epoch")
        except Exception as e:
            log.error("W&B init failed: %s", e)
            self.use_wandb = False

    def _log_wandb(self, metrics: dict):
        if not self.use_wandb:
            return
        try:
            import wandb

            wandb.log(metrics, step=self.epoch)
        except Exception as e:
            log.error("W&B log failed: %s", e)

    def _finish_wandb(self, best_metrics: dict | None = None):
        if not self.use_wandb:
            return
        try:
            import wandb

            if best_metrics:
                for k, v in best_metrics.items():
                    wandb.run.summary[f"best/{k}"] = v
            wandb.finish()
        except Exception as e:
            log.error("W&B finish failed: %s", e)

    def _core_param_metrics(self) -> dict[str, float]:
        """Snapshot of key core-model parameters for logging."""
        core = self.modules["attractor_core"]
        m: dict[str, float] = {}
        for attr, key in [
            ("rho", "analysis/rho"),
            ("lam", "analysis/lambda"),
            ("sigma0", "analysis/sigma0"),
            ("nu", "analysis/nu"),
        ]:
            if hasattr(core, attr):
                val = getattr(core, attr)
                m[key] = val.item() if isinstance(val, torch.Tensor) else float(val)

        if "dynamic_core" in self.modules:
            dyn = self.modules["dynamic_core"]
            for attr, key in [
                ("tau_in", "analysis/dynamic/tau_in"),
                ("kappa", "analysis/dynamic/kappa"),
                ("eta", "analysis/dynamic/eta"),
                ("sigma0", "analysis/dynamic/sigma0"),
            ]:
                if hasattr(dyn, attr):
                    val = getattr(dyn, attr)
                    m[key] = val.item() if isinstance(val, torch.Tensor) else float(val)

        return m

    # ------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------

    def fit(self) -> dict[str, float]:
        """Run training with early stopping.  Returns best-epoch metrics.

        Graceful interruption: first Ctrl+C during training cuts the
        epoch short and jumps to validation + checkpoint.  Second Ctrl+C
        (e.g. during validation) exits immediately.
        """
        self._init_wandb()
        best_metrics: dict[str, float] = {}

        epoch_bar = tqdm(range(self.cfg.training.n_epochs), desc="epochs", unit="ep")
        for epoch in epoch_bar:
            self.epoch = epoch
            sampler = getattr(self.train_loader, "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
            t0 = time.time()

            train_m = self.train_epoch()
            val_m = self.evaluate(self.val_loader, prefix="val")
            param_m = self._core_param_metrics()
            elapsed = time.time() - t0

            current_lr = self.optimizer.param_groups[0]["lr"]
            all_m = {
                **train_m,
                **val_m,
                **param_m,
                "epoch": epoch,
                "epoch_time_s": elapsed,
                "lr": current_lr,
            }
            self._log_wandb(all_m)

            val_loss = val_m.get("val/loss_total", float("inf"))
            train_loss = train_m.get("train/loss_total", float("nan"))

            if val_loss < self.best_val_loss - self.cfg.training.min_delta:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                best_metrics = all_m.copy()
                self._save_best(val_m)
                log.info(
                    "Epoch %3d | new best val_loss=%.4f, saved checkpoint",
                    epoch,
                    val_loss,
                )
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.cfg.training.patience:
                    log.info(
                        "Early stopping after %d epochs (patience=%d)",
                        epoch + 1,
                        self.cfg.training.patience,
                    )
                    break

            epoch_bar.set_postfix(
                train=f"{train_loss:.4f}",
                val=f"{val_loss:.4f}",
                best=f"{self.best_val_loss:.4f}",
            )

            if self._interrupted:
                log.warning(
                    "Stopping after interrupted epoch %d (val_loss=%.4f)",
                    epoch,
                    val_loss,
                )
                break

        self._finish_wandb(best_metrics)
        return best_metrics


# ======================================================================
# CLI helpers shared by the three training scripts
# ======================================================================

_CLI_TO_CONFIG: dict[str, str] = {
    "data_dir": "data_dir",
    "seed": "seed",
    "lr": "lr",
    "weight_decay": "weight_decay",
    "batch_size": "batch_size",
    "epochs": "n_epochs",
    "patience": "patience",
    "min_delta": "min_delta",
    "checkpoint_dir": "checkpoint_dir",
    "plot_dir": "plot_dir",
    "tasks": "tasks",
    "wandb": "enable_wandb",
    "resume": "resume",
    "num_workers": "num_workers",
}
"""Map from CLI dest name → TrainingConfig field name."""


def make_parser() -> argparse.ArgumentParser:
    """Build an ArgumentParser for the training script.

    All overridable arguments default to ``None`` so that
    ``resolve_config`` can detect which ones the user set explicitly
    versus which ones come from the YAML config file.
    """
    from src.common.config import PRESET_FAMILIES

    parser = argparse.ArgumentParser(
        description="Train a numerical-attractor model."
    )
    parser.add_argument(
        "--preset",
        default=None,
        choices=list(PRESET_FAMILIES),
        help="Model preset (overrides YAML). Choices: default, weber.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help=f"Path to YAML config (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--min-delta", type=float, default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--plot-dir", default=None)
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--wandb", action="store_true", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    return parser


def resolve_config(
    args: argparse.Namespace,
) -> ExperimentConfig:
    """Load the YAML config and merge CLI overrides into it.

    For every CLI argument that was explicitly provided (not ``None``)
    and differs from the YAML value, a warning is emitted.  Unset
    arguments fall back to the YAML value.
    """
    cfg = load_config(args.config)
    if getattr(args, "preset", None) is not None and args.preset != cfg.preset:
        from src.common.config import ATTRACTOR_PRESETS, PRESET_FAMILIES

        log.warning(
            "CLI --preset=%r overrides config preset=%r",
            args.preset, cfg.preset,
        )
        cfg.preset = args.preset
        if cfg.preset in ATTRACTOR_PRESETS:
            for k, v in ATTRACTOR_PRESETS[cfg.preset].items():
                if not hasattr(cfg.attractor, k):
                    continue
                setattr(cfg.attractor, k, v)

    tcfg = cfg.training

    for cli_dest, cfg_field in _CLI_TO_CONFIG.items():
        cli_val = getattr(args, cli_dest, None)
        yaml_val = getattr(tcfg, cfg_field)

        if cli_val is None:
            setattr(args, cli_dest, yaml_val)
        elif cli_val != yaml_val:
            log.warning(
                "CLI --%s=%r overrides config %s=%r",
                cli_dest.replace("_", "-"),
                cli_val,
                cfg_field,
                yaml_val,
            )

    cfg.training = TrainingConfig(
        seed=args.seed,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        n_epochs=args.epochs,
        patience=args.patience,
        min_delta=args.min_delta,
        checkpoint_dir=args.checkpoint_dir,
        plot_dir=args.plot_dir,
        data_dir=args.data_dir,
        tasks=args.tasks,
        num_workers=args.num_workers,
        enable_wandb=args.wandb,
        resume=args.resume,
    )
    return cfg


def run_training(args: argparse.Namespace) -> dict[str, float]:
    """Shared entry point called by ``scripts/train.py``."""
    cfg = resolve_config(args)
    tcfg = cfg.training

    set_seed(tcfg.seed)
    device = get_device()

    log.info(f"Training {cfg.preset} ({cfg.family})")
    log.info(f"Device: {device}")
    log.info(f"Tasks: {tcfg.tasks}")
    log.info(f"Data: {tcfg.data_dir}")

    train_ds, val_ds, test_ds, subject_map = load_task_data(
        tcfg.data_dir,
        tcfg.tasks,
        seed=tcfg.seed,
    )
    cfg.subject.n_subjects = subject_map.n_subjects
    available_tasks = list(train_ds.keys())
    log.info(f"Subjects: {subject_map.n_subjects}")
    log.info(f"Available tasks: {available_tasks}")
    for t, ds in train_ds.items():
        log.info(f"{t}: {len(ds)} train trials")

    use_cuda = device.type == "cuda"
    train_loader = build_dataloader(
        train_ds,
        batch_size=tcfg.batch_size,
        seed=tcfg.seed,
        task_weights=tcfg.task_weights,
        num_workers=tcfg.num_workers,
        pin_memory=use_cuda,
    )
    val_loader = _make_eval_loader(
        val_ds, tcfg.batch_size,
        num_workers=tcfg.num_workers, pin_memory=use_cuda,
    )

    modules = build_modules(cfg, subject_map.n_subjects, device)
    heads = build_task_heads(cfg, subject_map.n_subjects, available_tasks, device)

    trainer = Trainer(
        cfg,
        modules,
        heads,
        train_loader,
        val_loader,
        device,
        use_wandb=tcfg.enable_wandb,
    )

    if tcfg.resume:
        log.info(f"Resuming from {tcfg.resume}")
        trainer.load_checkpoint(tcfg.resume)

    best_metrics = trainer.fit()

    if test_ds:
        test_loader = _make_eval_loader(
            test_ds, tcfg.batch_size,
            num_workers=tcfg.num_workers, pin_memory=use_cuda,
        )
        test_metrics = trainer.evaluate(test_loader, prefix="test")
        log.info("Test set metrics:")
        for k, v in sorted(test_metrics.items()):
            log.info(f"{k}: {v:.4f}")

    return best_metrics
