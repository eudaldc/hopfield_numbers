"""Visual encoder: maps stimulus features or images to scalar latent evidence y.

Two concrete encoders, both nn.Module with the same contract::

    y = encoder(batch)   # y shape [B]

  ``FeatureEncoder``
      A small MLP (or linear map) from precomputed interpretable features
      to a single scalar.  Preferred mode for v0.1.
      Expects ``batch["features"]``  shape [B, F].

  ``TinyImageEncoder``
      A deliberately tiny conv stack with global average pooling and a
      linear readout to scalar y.  Kept small and weak on purpose.
      Expects ``batch["image"]``  shape [B, C, H, W].

Use ``build_visual_encoder(cfg)`` to instantiate the right one from config.
"""

import torch
import torch.nn as nn

from src.common.config import VisualEncoderConfig


class FeatureEncoder(nn.Module):
    """features ∈ R^F  →  scalar y ∈ R.

    Architecture: input → (optional hidden + ReLU) → linear → scalar.
    """

    BATCH_KEY = "features"

    def __init__(self, n_features: int, hidden_dim: int | None = None):
        super().__init__()
        layers: list[nn.Module] = []
        if hidden_dim is not None and hidden_dim > 0:
            layers.append(nn.Linear(n_features, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dim, 1))
        else:
            layers.append(nn.Linear(n_features, 1))
        self.net = nn.Sequential(*layers)

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        """[B, F] → [B]"""
        return self.net(features).squeeze(-1)

    def forward(self, batch: dict) -> torch.Tensor:
        """[B, F] → [B]"""
        return self.encode(batch[self.BATCH_KEY])


class TinyImageEncoder(nn.Module):
    """Image tensor → scalar y, via a minimal conv stack.

    Architecture:
        Conv2d(in_ch, 16, 5, 2, 2) → ReLU
        Conv2d(16, 32, 3, 2, 1)     → ReLU
        Conv2d(32, 32, 3, 2, 1)     → ReLU
        AdaptiveAvgPool2d(1)         → flatten → Linear(32, 1)

    Receptive field is small, capacity is deliberately limited.
    """

    BATCH_KEY = "image"

    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(32, 1)

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        """[B, C, H, W] → [B]"""
        x = self.conv(image)
        x = self.pool(x).flatten(1)
        return self.head(x).squeeze(-1)

    def forward(self, batch: dict) -> torch.Tensor:
        """[B, C, H, W] → [B]"""
        return self.encode(batch[self.BATCH_KEY])


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, type[nn.Module]] = {
    "features": FeatureEncoder,
    "image": TinyImageEncoder,
}


def build_visual_encoder(cfg: VisualEncoderConfig | None = None) -> nn.Module:
    """Instantiate the right encoder from config."""
    if cfg is None:
        cfg = VisualEncoderConfig()

    if cfg.mode == "features":
        return FeatureEncoder(cfg.n_features, cfg.hidden_dim)
    if cfg.mode == "image":
        return TinyImageEncoder(cfg.image_channels)

    raise ValueError(
        f"Unknown visual encoder mode {cfg.mode!r}. "
        f"Available: {list(_REGISTRY)}"
    )
