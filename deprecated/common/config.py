"""Lightweight config dataclasses for the PoC.

Dataclasses carry hard-coded fallback defaults.  The primary source of
truth is ``configs/config.yaml`` which is loaded via ``load_config()``.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "configs" / "config.yaml"


@dataclass
class SupportConfig:
    """Integer support range for the attractor model."""

    n_min: int = 0
    n_max: int = 20

    @property
    def size(self) -> int:
        return self.n_max - self.n_min + 1

    @property
    def integers(self) -> list[int]:
        return list(range(self.n_min, self.n_max + 1))


ATTRACTOR_PRESETS: dict[str, dict[str, float]] = {
    "default": {
        "rho_init": 0.5,
        "nu_init": 1.0,
        "sigma0_init": 1.0,
        "lam_init": 0.5,
    },
    "weber": {
        "rho_init": 1.0,
        "nu_init": 1.0,
        "sigma0_init": 0.4,
        "lam_init": 1.0,
    },
}

PRESET_FAMILIES: dict[str, str] = {
    "default": "attractor",
    "weber": "attractor",
}


@dataclass
class AttractorConfig:
    """Parameters for the attractor core."""

    rho_init: float = 0.5
    nu_init: float = 1.0
    sigma0_init: float = 1.0
    lam_init: float = 0.5
    support: SupportConfig = field(default_factory=SupportConfig)


@dataclass
class SubjectConfig:
    """Parameters for subject-level embeddings."""

    n_subjects: int = 1
    temp_init_mean: float = 1.0
    width_init_std: float = 0.01
    tau_sym_init: float = 1.0
    l2_penalty: float = 1e-3


@dataclass
class VisualEncoderConfig:
    """Parameters for the visual encoder."""

    mode: str = "features"  # "features" or "image"
    n_features: int = 6
    hidden_dim: Optional[int] = 32
    # tiny image encoder settings
    image_channels: int = 1
    image_size: int = 64


@dataclass
class TrainingConfig:
    """Training hyperparameters and operational defaults.

    CLI arguments pull their defaults from here.  When a CLI flag
    explicitly overrides a value, a warning is emitted.
    """

    seed: int = 17
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 1024
    n_epochs: int = 100
    patience: int = 10
    min_delta: float = 1e-4
    task_weights: dict[str, float] = field(
        default_factory=lambda: {
            "symbolic_comparison": 1.0,
            "match_to_sample": 1.0,
            "dot_verification": 1.0,
            "panamath": 1.0,
        }
    )
    wandb_project: str = "numeric-attractors-poc"
    checkpoint_dir: str = "checkpoints"
    plot_dir: str = "output"
    data_dir: str = "data/processed"
    tasks: list[str] = field(
        default_factory=lambda: [
            "symbolic_comparison",
            "match_to_sample",
            "dot_verification",
        ]
    )
    num_workers: int = 4
    enable_wandb: bool = False
    resume: Optional[str] = None


@dataclass
class DotVerificationConfig:
    """Hyperparameters for the dot-verification counting-route mixture."""

    eps_count: float = 0.05
    model_rt: bool = False
    rt_loss_weight: float = 0.1


@dataclass
class DynamicAttractorConfig:
    """Parameters for the dynamic attractor cleanup core."""

    support_min: int = 1
    support_max: int = 9
    rho_init: float = 0.1
    lambda_init: float = 0.5
    sigma0_init: float = 1.0
    nu_init: float = 1.0
    tau_in_init: float = 1.0
    kappa_init: float = 1.0
    eta_init: float = 0.2
    cleanup_mode: str = "gradient"   # "gradient" | "none"
    max_steps: int = 16
    tol: float = 1e-3


@dataclass
class EvidenceChannelConfig:
    """Parameters for the neutral evidence channel y0 ~ Normal(m0(N), s0(N))."""

    rho0: float = 0.5
    nu0: float = 1.0
    s00: float = 0.1
    s0: float = 0.3
    lambda0: float = 0.5


@dataclass
class SyntheticConfig:
    """Parameters for synthetic benchmark data generation."""

    enabled: bool = False
    n_min: int = 1
    n_max: int = 9
    n_samples_train: int = 50_000
    n_samples_val: int = 10_000
    n_samples_test: int = 10_000
    seed: int = 7
    evidence: EvidenceChannelConfig = field(default_factory=EvidenceChannelConfig)


@dataclass
class RTProxyConfig:
    """Parameters for the reaction-time proxy."""

    enabled: bool = True
    mode: str = "steps"              # "steps" | "steps_plus_margin"
    t0_init: float = 1.0
    c_steps_init: float = 1.0
    c_margin_init: float = 1.0


@dataclass
class ExperimentConfig:
    """Top-level config combining all components."""

    support: SupportConfig = field(default_factory=SupportConfig)
    attractor: AttractorConfig = field(default_factory=AttractorConfig)
    dynamic_attractor: DynamicAttractorConfig = field(default_factory=DynamicAttractorConfig)
    subject: SubjectConfig = field(default_factory=SubjectConfig)
    visual: VisualEncoderConfig = field(default_factory=VisualEncoderConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    dot_verification: DotVerificationConfig = field(default_factory=DotVerificationConfig)
    synthetic: SyntheticConfig = field(default_factory=SyntheticConfig)
    rt_proxy: RTProxyConfig = field(default_factory=RTProxyConfig)
    preset: str = "default"  # "default" | "weber"
    cohort: str = "A"
    attractor_mode: str = "dynamic"  # "dynamic" | "static"

    @property
    def family(self) -> str:
        """Model family derived from the preset."""
        return PRESET_FAMILIES[self.preset]


# ======================================================================
# YAML loading
# ======================================================================

_SECTION_TO_DATACLASS = {
    "support": SupportConfig,
    "attractor": AttractorConfig,
    "dynamic_attractor": DynamicAttractorConfig,
    "subject": SubjectConfig,
    "visual": VisualEncoderConfig,
    "training": TrainingConfig,
    "dot_verification": DotVerificationConfig,
    "synthetic": SyntheticConfig,
    "rt_proxy": RTProxyConfig,
    "evidence": EvidenceChannelConfig,
}


def _build_section(cls, data: dict):
    """Instantiate a dataclass, ignoring unknown keys and injecting
    a shared ``SupportConfig`` where the section accepts one.
    """
    if data is None:
        return cls()

    known = {f.name for f in cls.__dataclass_fields__.values()}
    filtered = {k: v for k, v in data.items() if k in known}
    return cls(**filtered)


def load_config(path: str | Path | None = None) -> ExperimentConfig:
    """Load an ``ExperimentConfig`` from a YAML file.

    Parameters
    ----------
    path : str | Path | None
        Path to the YAML config.  Defaults to ``configs/config.yaml``
        at the project root.

    Returns
    -------
    ExperimentConfig
    """
    path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with open(path) as fh:
        raw: dict = yaml.safe_load(fh) or {}

    preset = raw.get("preset", "default")
    if preset not in PRESET_FAMILIES:
        raise ValueError(
            f"Unknown preset {preset!r}. "
            f"Choose from: {list(PRESET_FAMILIES)}"
        )

    # For attractor-based presets, merge preset defaults before YAML overrides
    attractor_raw = raw.get("attractor", {}) or {}
    if preset in ATTRACTOR_PRESETS:
        attractor_raw = {**ATTRACTOR_PRESETS[preset], **attractor_raw}

    sections: dict = {}
    for section_name, cls in _SECTION_TO_DATACLASS.items():
        if section_name == "attractor":
            section_data = attractor_raw
        elif section_name == "evidence":
            section_data = raw.get("synthetic", {})
            if section_data:
                section_data = section_data.get("evidence", {})
        else:
            section_data = raw.get(section_name, {})
        obj = _build_section(cls, section_data)
        if hasattr(obj, "support") and "support" not in (section_data or {}):
            obj.support = sections.get("support", SupportConfig())
        sections[section_name] = obj

    synth = sections["synthetic"]
    synth.evidence = sections["evidence"]

    return ExperimentConfig(
        support=sections["support"],
        attractor=sections["attractor"],
        dynamic_attractor=sections["dynamic_attractor"],
        subject=sections["subject"],
        visual=sections["visual"],
        training=sections["training"],
        dot_verification=sections["dot_verification"],
        synthetic=synth,
        rt_proxy=sections["rt_proxy"],
        preset=preset,
        cohort=raw.get("cohort", "A"),
        attractor_mode=raw.get("attractor_mode", "dynamic"),
    )
