"""Extract interpretable visual features from dot-display images.

Features extracted per image:
  - total_area     : total number of foreground pixels (proxy for cumulative dot area)
  - mean_dot_size  : mean area of individual connected components (dots)
  - convex_hull    : area of convex hull enclosing dot centroids
  - density        : n_dots / convex_hull_area  (0 when hull is degenerate)
  - spacing        : mean nearest-neighbour distance between dot centroids
  - fourier_power  : mean log-power in an annular band of the 2-D DFT
                     (captures spatial frequency structure of the dot pattern)

All functions work on single images (numpy arrays, H×W, uint8 or float [0,1]).
`extract_features_from_image` returns a dict whose keys match FEATURE_COLUMNS.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.data.schemas import FEATURE_COLUMNS


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _binarize(img: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Return a boolean mask of foreground pixels."""
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    return img > threshold


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Label connected components via scipy.ndimage.

    Returns (label_array, n_components).
    """
    from scipy.ndimage import label as nd_label

    labeled, n = nd_label(mask)
    return labeled, int(n)


def _centroids_and_sizes(
    labeled: np.ndarray, n_components: int
) -> tuple[np.ndarray, np.ndarray]:
    """Compute centroid (row, col) and pixel-count for each component.

    Returns
    -------
    centroids : (n_components, 2) float64
    sizes     : (n_components,) float64
    """
    from scipy.ndimage import center_of_mass, sum as nd_sum

    indices = list(range(1, n_components + 1))
    centroids = np.array(
        center_of_mass(labeled, labeled, indices), dtype=np.float64
    )
    sizes = np.array(
        nd_sum(np.ones_like(labeled), labeled, indices), dtype=np.float64
    )
    return centroids, sizes


# ---------------------------------------------------------------------------
# Individual feature functions
# ---------------------------------------------------------------------------

def compute_total_area(mask: np.ndarray) -> float:
    return float(mask.sum())


def compute_mean_dot_size(sizes: np.ndarray) -> float:
    if len(sizes) == 0:
        return 0.0
    return float(sizes.mean())


def compute_convex_hull_area(centroids: np.ndarray) -> float:
    """Convex hull area of 2-D centroids. Returns 0 for < 3 points."""
    if centroids.shape[0] < 3:
        return 0.0
    try:
        from scipy.spatial import ConvexHull

        hull = ConvexHull(centroids)
        return float(hull.volume)  # .volume is area for 2-D
    except Exception:
        return 0.0


def compute_spacing(centroids: np.ndarray) -> float:
    """Mean nearest-neighbour distance between centroids."""
    n = centroids.shape[0]
    if n < 2:
        return 0.0
    from scipy.spatial.distance import cdist

    dists = cdist(centroids, centroids)
    np.fill_diagonal(dists, np.inf)
    nn_dists = dists.min(axis=1)
    return float(nn_dists.mean())


def compute_fourier_power(
    img: np.ndarray,
    low_frac: float = 0.05,
    high_frac: float = 0.40,
) -> float:
    """Mean log-power in an annular frequency band of the 2-D DFT.

    The band is defined as radii between `low_frac` and `high_frac`
    of the Nyquist frequency.  This captures the dominant spatial-
    frequency content without being dominated by the DC component.
    """
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0

    fft2 = np.fft.fft2(img)
    power = np.abs(np.fft.fftshift(fft2)) ** 2

    h, w = img.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

    max_r = min(cy, cx)
    annulus = (radius >= low_frac * max_r) & (radius <= high_frac * max_r)

    if annulus.sum() == 0:
        return 0.0
    return float(np.log1p(power[annulus]).mean())


# ---------------------------------------------------------------------------
# Main entry point for a single image
# ---------------------------------------------------------------------------

def extract_features(img: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """Extract all stimulus features from a single H×W grayscale image.

    Parameters
    ----------
    img : 2-D array, uint8 or float32 in [0, 1]
    threshold : binarization threshold (on [0,1] scale)

    Returns
    -------
    dict with keys matching ``FEATURE_COLUMNS``.
    """
    mask = _binarize(img, threshold)
    labeled, n_comp = _label_components(mask)

    if n_comp == 0:
        return {c: 0.0 for c in FEATURE_COLUMNS}

    centroids, sizes = _centroids_and_sizes(labeled, n_comp)
    hull_area = compute_convex_hull_area(centroids)
    density = float(n_comp) / hull_area if hull_area > 0 else 0.0

    return {
        "total_area": compute_total_area(mask),
        "mean_dot_size": compute_mean_dot_size(sizes),
        "convex_hull": hull_area,
        "density": density,
        "spacing": compute_spacing(centroids),
        "fourier_power": compute_fourier_power(img),
    }


def extract_features_from_path(
    image_path: str | Path,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Load an image from disk and extract features."""
    from PIL import Image

    img = np.array(Image.open(image_path).convert("L"))
    return extract_features(img, threshold)


# ---------------------------------------------------------------------------
# Batch extraction → DataFrame
# ---------------------------------------------------------------------------

def build_stimulus_feature_table(
    image_paths: list[str] | pd.Series,
    true_ns: Optional[list[int] | pd.Series] = None,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Extract features for many images and return a DataFrame.

    The returned DataFrame has columns:
        image_path, true_n (if provided), + FEATURE_COLUMNS

    and is indexed by ``image_path`` for fast lookup.
    """
    records = []
    for i, path in enumerate(image_paths):
        feats = extract_features_from_path(path, threshold)
        feats["image_path"] = str(path)
        if true_ns is not None:
            feats["true_n"] = int(true_ns[i] if hasattr(true_ns, "__getitem__") else true_ns)
        records.append(feats)

    df = pd.DataFrame(records)
    df = df.set_index("image_path")
    return df
