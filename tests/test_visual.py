"""Tests for WP-C: stimulus feature extraction and visual encoder."""

import tempfile
from pathlib import Path

import numpy as np
import torch

from src.common.config import VisualEncoderConfig
from src.data.schemas import FEATURE_COLUMNS
from src.data.stimulus_features import (
    extract_features,
    extract_features_from_path,
    build_stimulus_feature_table,
    compute_total_area,
    compute_mean_dot_size,
    compute_convex_hull_area,
    compute_spacing,
    compute_fourier_power,
    _binarize,
    _label_components,
)
from src.models.visual_encoder import (
    FeatureEncoder,
    TinyImageEncoder,
    build_visual_encoder,
)


# ---------------------------------------------------------------------------
# Helpers — synthetic dot images
# ---------------------------------------------------------------------------

def _make_dot_image(
    n_dots: int = 5,
    img_size: int = 64,
    dot_radius: int = 3,
    seed: int = 13,
) -> np.ndarray:
    """Create a synthetic grayscale image with n_dots white circles on black."""
    rng = np.random.RandomState(seed)
    img = np.zeros((img_size, img_size), dtype=np.uint8)
    margin = dot_radius + 1
    for _ in range(n_dots):
        cy = rng.randint(margin, img_size - margin)
        cx = rng.randint(margin, img_size - margin)
        yy, xx = np.ogrid[:img_size, :img_size]
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= dot_radius ** 2
        img[mask] = 255
    return img


def _save_dot_image(path: Path, n_dots: int = 5, seed: int = 13) -> None:
    from PIL import Image

    arr = _make_dot_image(n_dots=n_dots, seed=seed)
    Image.fromarray(arr, mode="L").save(path)


# ---------------------------------------------------------------------------
# stimulus_features — binarization & labeling
# ---------------------------------------------------------------------------

def test_binarize_uint8():
    img = np.zeros((10, 10), dtype=np.uint8)
    img[3:6, 3:6] = 255
    mask = _binarize(img, threshold=0.5)
    assert mask.dtype == bool
    assert mask.sum() == 9


def test_binarize_float():
    img = np.zeros((10, 10), dtype=np.float32)
    img[0:2, 0:2] = 0.9
    mask = _binarize(img, threshold=0.5)
    assert mask.sum() == 4


def test_label_components_counts():
    mask = np.zeros((20, 20), dtype=bool)
    mask[2:5, 2:5] = True
    mask[10:13, 10:13] = True
    _, n = _label_components(mask)
    assert n == 2


# ---------------------------------------------------------------------------
# stimulus_features — individual feature functions
# ---------------------------------------------------------------------------

def test_total_area():
    mask = np.zeros((10, 10), dtype=bool)
    mask[0:3, 0:3] = True
    assert compute_total_area(mask) == 9.0


def test_mean_dot_size():
    sizes = np.array([10.0, 20.0, 30.0])
    assert compute_mean_dot_size(sizes) == 20.0
    assert compute_mean_dot_size(np.array([])) == 0.0


def test_convex_hull_needs_3_points():
    pts = np.array([[0.0, 0.0], [1.0, 1.0]])
    assert compute_convex_hull_area(pts) == 0.0


def test_convex_hull_triangle():
    pts = np.array([[0.0, 0.0], [4.0, 0.0], [0.0, 4.0]])
    area = compute_convex_hull_area(pts)
    assert abs(area - 8.0) < 0.01


def test_spacing_two_points():
    pts = np.array([[0.0, 0.0], [3.0, 4.0]])
    assert abs(compute_spacing(pts) - 5.0) < 0.01


def test_spacing_single_point():
    assert compute_spacing(np.array([[1.0, 2.0]])) == 0.0


def test_fourier_power_positive():
    img = _make_dot_image(n_dots=8, seed=77)
    fp = compute_fourier_power(img)
    assert fp > 0.0


# ---------------------------------------------------------------------------
# stimulus_features — full extraction
# ---------------------------------------------------------------------------

def test_extract_features_returns_all_keys():
    img = _make_dot_image(n_dots=6, seed=3)
    feats = extract_features(img)
    for col in FEATURE_COLUMNS:
        assert col in feats, f"Missing key: {col}"
        assert isinstance(feats[col], float)


def test_extract_features_empty_image():
    img = np.zeros((32, 32), dtype=np.uint8)
    feats = extract_features(img)
    for col in FEATURE_COLUMNS:
        assert feats[col] == 0.0


def test_extract_features_from_path(tmp_path):
    p = tmp_path / "dots.png"
    _save_dot_image(p, n_dots=4, seed=19)
    feats = extract_features_from_path(p)
    assert feats["total_area"] > 0
    assert feats["mean_dot_size"] > 0


def test_build_stimulus_feature_table(tmp_path):
    paths = []
    for i in range(3):
        p = tmp_path / f"img_{i}.png"
        _save_dot_image(p, n_dots=i + 2, seed=i)
        paths.append(str(p))
    true_ns = [2, 3, 4]
    df = build_stimulus_feature_table(paths, true_ns)
    assert len(df) == 3
    assert "true_n" in df.columns
    for col in FEATURE_COLUMNS:
        assert col in df.columns


# ---------------------------------------------------------------------------
# visual_encoder — FeatureEncoder
# ---------------------------------------------------------------------------

def test_feature_encoder_output_shape():
    enc = FeatureEncoder(n_features=6, hidden_dim=16)
    batch = {"features": torch.randn(8, 6)}
    y = enc(batch)
    assert y.shape == (8,)


def test_feature_encoder_no_hidden():
    enc = FeatureEncoder(n_features=6, hidden_dim=None)
    batch = {"features": torch.randn(4, 6)}
    y = enc(batch)
    assert y.shape == (4,)


def test_feature_encoder_encode_direct():
    enc = FeatureEncoder(n_features=6, hidden_dim=16)
    x = torch.randn(4, 6, requires_grad=True)
    y = enc.encode(x)
    y.sum().backward()
    assert y.shape == (4,)
    assert x.grad is not None


# ---------------------------------------------------------------------------
# visual_encoder — TinyImageEncoder
# ---------------------------------------------------------------------------

def test_tiny_image_encoder_output_shape():
    enc = TinyImageEncoder(in_channels=1)
    batch = {"image": torch.randn(4, 1, 64, 64)}
    y = enc(batch)
    assert y.shape == (4,)


def test_tiny_image_encoder_different_sizes():
    enc = TinyImageEncoder(in_channels=1)
    for sz in [32, 48, 64, 128]:
        y = enc.encode(torch.randn(2, 1, sz, sz))
        assert y.shape == (2,), f"Failed for size {sz}"


def test_tiny_image_encoder_rgb():
    enc = TinyImageEncoder(in_channels=3)
    batch = {"image": torch.randn(2, 3, 64, 64)}
    y = enc(batch)
    assert y.shape == (2,)


# ---------------------------------------------------------------------------
# build_visual_encoder dispatcher
# ---------------------------------------------------------------------------

def test_build_features_mode():
    cfg = VisualEncoderConfig(mode="features", n_features=6, hidden_dim=16)
    enc = build_visual_encoder(cfg)
    assert isinstance(enc, FeatureEncoder)
    y = enc({"features": torch.randn(8, 6)})
    assert y.shape == (8,)


def test_build_image_mode():
    cfg = VisualEncoderConfig(mode="image", image_channels=1)
    enc = build_visual_encoder(cfg)
    assert isinstance(enc, TinyImageEncoder)
    y = enc({"image": torch.randn(4, 1, 64, 64)})
    assert y.shape == (4,)


def test_build_default_config():
    enc = build_visual_encoder()
    assert isinstance(enc, FeatureEncoder)


def test_build_invalid_mode():
    cfg = VisualEncoderConfig(mode="transformer")
    try:
        build_visual_encoder(cfg)
        assert False, "Should have raised"
    except ValueError:
        pass
