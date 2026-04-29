"""Synthetic benchmark datasets generated from the attractor theory.

Three tasks:
- Task A — Absolute estimation / identification: y0 → n_hat
- Task B — Pairwise comparison: (y0_left, y0_right) → which is larger?
- Task C — Same/different verification: does y0 correspond to target m?

The evidence channel y0 ~ Normal(m0(N), s0(N)) produces a neutral noisy
starting point for attractor cleanup.  This is NOT the theory under test;
it is only the initial condition.

Each generator returns a plain dict of tensors and is reproducible via seed.

Dataset dict schemas
--------------------
Task A / C:
    {
        "n_true"  : LongTensor[B],
        "y_noisy" : FloatTensor[B],
    }
Task C additionally:
    {
        "target"  : LongTensor[B],
        "label"   : LongTensor[B],   # 1 if n_true == target else 0
    }
Task B:
    {
        "n_left"  : LongTensor[B],
        "n_right" : LongTensor[B],
        "y_left"  : FloatTensor[B],
        "y_right" : FloatTensor[B],
        "label"   : LongTensor[B],   # 1 if n_right > n_left else 0
    }
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


# ── Box-Cox (plain float version, matching the evidence channel spec) ──────


def _boxcox(z: torch.Tensor, rho: float) -> torch.Tensor:
    if abs(rho) < 1e-6:
        return torch.log(z)
    return (z.pow(rho) - 1.0) / rho


# ── Neutral evidence channel ──────────────────────────────────────────────


@dataclass
class NeutralEvidenceChannel:
    """Neutral noisy evidence channel: y0 ~ Normal(m0(N), s0(N)).

    m0(N) = BoxCox(N + nu0, rho0)
    s0(N) = s00 + s0 * (N + nu0)^lambda0

    This is the initial condition for attractor cleanup, not the theory
    under test.
    """

    rho0: float = 0.5
    nu0: float = 1.0
    s00: float = 0.1
    s0: float = 0.3
    lambda0: float = 0.5

    def mean(self, n_true: torch.Tensor) -> torch.Tensor:
        z = n_true.float() + self.nu0
        return _boxcox(z, self.rho0)

    def std(self, n_true: torch.Tensor) -> torch.Tensor:
        z = n_true.float() + self.nu0
        return self.s00 + self.s0 * z.pow(self.lambda0)

    def sample(
        self,
        n_true: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample noisy initial evidence y0 ~ Normal(m0(N), s0(N))."""
        mu = self.mean(n_true)
        sigma = self.std(n_true)
        eps = torch.empty_like(mu).normal_(generator=generator)
        return mu + sigma * eps


# ── helpers ────────────────────────────────────────────────────────────────


def _make_generator(seed: int | None) -> torch.Generator | None:
    if seed is None:
        return None
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _default_channel() -> NeutralEvidenceChannel:
    return NeutralEvidenceChannel()


# ── Task A — Absolute estimation ───────────────────────────────────────────


def generate_task_a(
    n_min: int = 1,
    n_max: int = 9,
    n_samples: int = 10_000,
    seed: int | None = None,
    channel: NeutralEvidenceChannel | None = None,
) -> dict[str, torch.Tensor]:
    """Generate (N, y0) pairs for absolute numerosity estimation.

    Parameters
    ----------
    n_min, n_max : inclusive range of true numerosities
    n_samples    : number of samples
    seed         : for reproducibility
    channel      : evidence channel; uses default if None

    Returns
    -------
    dict with n_true, y_noisy
    """
    if channel is None:
        channel = _default_channel()
    g = _make_generator(seed)
    n_true = torch.randint(n_min, n_max + 1, (n_samples,), generator=g)
    y_noisy = channel.sample(n_true, generator=g)
    return {
        "n_true": n_true,
        "y_noisy": y_noisy,
    }


# ── Task B — Pairwise comparison ───────────────────────────────────────────


def generate_task_b(
    n_min: int = 1,
    n_max: int = 9,
    n_samples: int = 10_000,
    seed: int | None = None,
    channel: NeutralEvidenceChannel | None = None,
) -> dict[str, torch.Tensor]:
    """Generate (N1, N2, y1, y2) pairs for pairwise numerosity comparison.

    Returns
    -------
    dict with n_left, n_right, y_left, y_right, label
    (label = 1 if n_right > n_left, else 0; equal pairs are kept)
    """
    if channel is None:
        channel = _default_channel()
    g = _make_generator(seed)
    n_left = torch.randint(n_min, n_max + 1, (n_samples,), generator=g)
    n_right = torch.randint(n_min, n_max + 1, (n_samples,), generator=g)
    y_left = channel.sample(n_left, generator=g)
    y_right = channel.sample(n_right, generator=g)
    label = (n_right > n_left).long()
    return {
        "n_left": n_left,
        "n_right": n_right,
        "y_left": y_left,
        "y_right": y_right,
        "label": label,
    }


# ── Task C — Same/different verification ───────────────────────────────────


def generate_task_c(
    n_min: int = 1,
    n_max: int = 9,
    n_samples: int = 10_000,
    seed: int | None = None,
    channel: NeutralEvidenceChannel | None = None,
) -> dict[str, torch.Tensor]:
    """Generate (N, target, y0) triples for same/different verification.

    Returns
    -------
    dict with n_true, target, y_noisy, label
    (label = 1 if n_true == target, else 0)
    """
    if channel is None:
        channel = _default_channel()
    g = _make_generator(seed)
    n_true = torch.randint(n_min, n_max + 1, (n_samples,), generator=g)
    target = torch.randint(n_min, n_max + 1, (n_samples,), generator=g)
    y_noisy = channel.sample(n_true, generator=g)
    label = (n_true == target).long()
    return {
        "n_true": n_true,
        "target": target,
        "y_noisy": y_noisy,
        "label": label,
    }


# ── convenience builder ────────────────────────────────────────────────────


def build_benchmark_datasets(
    n_min: int = 1,
    n_max: int = 9,
    n_samples_train: int = 50_000,
    n_samples_val: int = 10_000,
    n_samples_test: int = 10_000,
    seed: int = 7,
    channel: NeutralEvidenceChannel | None = None,
) -> dict[str, dict[str, dict[str, torch.Tensor]]]:
    """Build all three tasks for train / val / test splits.

    Returns
    -------
    Nested dict: {split: {task_a/b/c: data_dict}}
    """
    if channel is None:
        channel = _default_channel()
    split_sizes = {
        "train": n_samples_train,
        "val": n_samples_val,
        "test": n_samples_test,
    }
    seed_offsets = {"train": 0, "val": 1_000, "test": 2_000}
    result: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
    for split, n in split_sizes.items():
        base_seed = seed + seed_offsets[split]
        result[split] = {
            "task_a": generate_task_a(
                n_min, n_max, n, seed=base_seed, channel=channel,
            ),
            "task_b": generate_task_b(
                n_min, n_max, n, seed=base_seed + 1, channel=channel,
            ),
            "task_c": generate_task_c(
                n_min, n_max, n, seed=base_seed + 2, channel=channel,
            ),
        }
    return result
