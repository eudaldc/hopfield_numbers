"""Checkpoint save/load and simple I/O helpers."""

import re
from pathlib import Path
from typing import Any, Optional

import torch

_CHECKPOINT_TOKEN_UNSAFE = re.compile(r"[^0-9A-Za-z_-]+")


def sanitize_checkpoint_token(s: str) -> str:
    """Map a config string to a safe single path component."""
    t = _CHECKPOINT_TOKEN_UNSAFE.sub("_", (s or "").strip())
    return t if t else "unknown"


def best_checkpoint_filename(preset: str, mode: str = "dynamic") -> str:
    """Default best-model filename encoding both preset and attractor mode."""
    p = sanitize_checkpoint_token(preset)
    m = sanitize_checkpoint_token(mode)
    return f"best_{p}_{m}.pt"


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: int = 0,
    metrics: Optional[dict[str, float]] = None,
    extra: Optional[dict[str, Any]] = None,
) -> Path:
    """Save a training checkpoint to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if metrics is not None:
        payload["metrics"] = metrics
    if extra is not None:
        payload["extra"] = extra

    torch.save(payload, path)
    return path


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> dict[str, Any]:
    """Load a checkpoint and restore model (and optionally optimizer) state.

    Returns the full checkpoint dict so callers can read epoch/metrics/extra.
    """
    map_location = device if device is not None else "cpu"
    ckpt = torch.load(path, map_location=map_location, weights_only=False)

    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    return ckpt


def infer_preset(state_dict: dict, extra: dict | None = None) -> str:
    """Return the preset from checkpoint *extra* metadata, falling back to
    ``"default"`` when metadata is absent."""
    if extra:
        preset = extra.get("preset")
        if preset:
            return preset
    return "default"


def infer_mode(extra: dict | None = None) -> str:
    """Return attractor_mode from checkpoint *extra*, defaulting to ``"dynamic"``."""
    if extra:
        mode = extra.get("attractor_mode")
        if mode:
            return mode
    return "dynamic"


def ensure_dir(path: str | Path) -> Path:
    """Create directory (and parents) if it doesn't exist. Returns the Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
