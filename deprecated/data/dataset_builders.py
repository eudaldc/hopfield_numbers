"""PyTorch Dataset classes for each task.

Every dataset pre-tensorizes DataFrame columns in __init__ so that
__getitem__ is a pure tensor-indexing operation (no pandas overhead).
This is critical for multi-worker DataLoader performance.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader

from src.data.schemas import FEATURE_COLUMNS


# ---------------------------------------------------------------------------
# Subject ID -> integer index mapping
# ---------------------------------------------------------------------------

class SubjectIndexMap:
    """Bidirectional map from subject_id strings to contiguous integer indices."""

    def __init__(self, subject_ids: list[str] | np.ndarray | pd.Series):
        unique = sorted(set(np.asarray(subject_ids).tolist()))
        self._id_to_idx = {sid: i for i, sid in enumerate(unique)}
        self._idx_to_id = {i: sid for sid, i in self._id_to_idx.items()}

    def __len__(self) -> int:
        return len(self._id_to_idx)

    def __getitem__(self, subject_id: str) -> int:
        return self._id_to_idx[subject_id]

    def to_idx(self, subject_id: str) -> int:
        return self._id_to_idx[subject_id]

    def to_id(self, idx: int) -> str:
        return self._idx_to_id[idx]

    @property
    def n_subjects(self) -> int:
        return len(self._id_to_idx)

    @property
    def all_ids(self) -> list[str]:
        return list(self._id_to_idx.keys())


# ---------------------------------------------------------------------------
# Optional helpers
# ---------------------------------------------------------------------------

def _load_image(path: str | Path) -> torch.Tensor:
    """Load a grayscale image as a float32 tensor [1, H, W] in [0, 1]."""
    from PIL import Image

    img = Image.open(path).convert("L")
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def _precompute_features(
    df: pd.DataFrame,
    path_col: str,
    feature_df: pd.DataFrame,
    feature_cols: list[str] = FEATURE_COLUMNS,
) -> tuple[torch.Tensor, np.ndarray]:
    """Vectorised feature lookup for all rows.

    Returns (features [N, F], valid_mask [N] bool array).
    """
    n = len(df)
    n_feat = len(feature_cols)
    features = torch.zeros(n, n_feat, dtype=torch.float32)
    mask = np.zeros(n, dtype=bool)

    if path_col not in df.columns:
        return features, mask

    has_path = df[path_col].notna().values
    if not has_path.any():
        return features, mask

    valid_paths = df.loc[has_path, path_col].values
    feat_rows = feature_df.reindex(valid_paths)[feature_cols]
    has_feat = ~feat_rows.isna().any(axis=1).values

    if has_feat.any():
        good_idx = np.where(has_path)[0][has_feat]
        features[good_idx] = torch.from_numpy(
            feat_rows.values[has_feat].astype(np.float32)
        )
        mask[good_idx] = True

    return features, mask


def _subject_idx_tensor(
    df: pd.DataFrame,
    subject_map: SubjectIndexMap,
) -> torch.Tensor:
    """Vectorised subject-id → index conversion for a whole DataFrame."""
    idx_series = df["subject_id"].map(subject_map._id_to_idx)
    return torch.from_numpy(idx_series.values.astype(np.int64))


# ---------------------------------------------------------------------------
# Task datasets — all pre-tensorise in __init__
# ---------------------------------------------------------------------------

class SymbolicComparisonDataset(Dataset):
    """Dataset for the symbolic number comparison task."""

    def __init__(
        self,
        df: pd.DataFrame,
        subject_map: SubjectIndexMap,
    ):
        df = df.reset_index(drop=True)
        self._n = len(df)

        self._subject_idx = _subject_idx_tensor(df, subject_map)
        self._left_digit = torch.from_numpy(
            df["left_digit"].values.astype(np.int64)
        )
        self._right_digit = torch.from_numpy(
            df["right_digit"].values.astype(np.int64)
        )
        self._label = torch.from_numpy(
            (df["response_side"].values == "right").astype(np.float32)
        )
        self._correct = torch.from_numpy(
            df["correct"].values.astype(np.int64)
        )
        rt = df["rt_ms"].fillna(-1.0).values.astype(np.float32)
        self._rt_ms = torch.from_numpy(rt)
        self._grade = torch.from_numpy(df["grade"].values.astype(np.int64))
        self._trial_id = df["trial_id"].values.astype(str)
        self._cohort = df["cohort"].values.astype(str)

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> dict:
        return {
            "task": "symbolic_comparison",
            "subject_idx": self._subject_idx[idx],
            "trial_id": self._trial_id[idx],
            "left_digit": self._left_digit[idx],
            "right_digit": self._right_digit[idx],
            "label": self._label[idx],
            "correct": self._correct[idx],
            "rt_ms": self._rt_ms[idx],
            "grade": self._grade[idx],
            "cohort": self._cohort[idx],
        }


class MatchToSampleDataset(Dataset):
    """Dataset for the match-to-sample task (dots->dots or dots->digit)."""

    def __init__(
        self,
        df: pd.DataFrame,
        subject_map: SubjectIndexMap,
        feature_df: Optional[pd.DataFrame] = None,
        load_images: bool = False,
    ):
        df = df.reset_index(drop=True)
        self._n = len(df)
        self._load_images = load_images

        self._subject_idx = _subject_idx_tensor(df, subject_map)
        self._sample_true_n = torch.from_numpy(
            df["sample_true_n"].values.astype(np.int64)
        )
        self._target_true_n = torch.from_numpy(
            df["target_true_n"].values.astype(np.int64)
        )
        self._label = torch.from_numpy(
            df["response_same"].values.astype(np.float32)
        )
        self._correct = torch.from_numpy(
            df["correct"].values.astype(np.int64)
        )
        rt = df["rt_ms"].fillna(-1.0).values.astype(np.float32)
        self._rt_ms = torch.from_numpy(rt)
        self._grade = torch.from_numpy(df["grade"].values.astype(np.int64))
        self._trial_id = df["trial_id"].values.astype(str)
        self._cohort = df["cohort"].values.astype(str)
        self._target_type = df["target_type"].values.astype(str)

        # Optional target digit (only for digit-target rows)
        if "target_digit" in df.columns:
            self._has_target_digit = df["target_digit"].notna().values
            self._target_digit = torch.from_numpy(
                df["target_digit"].fillna(0).values.astype(np.int64)
            )
        else:
            self._has_target_digit = np.zeros(self._n, dtype=bool)
            self._target_digit = torch.zeros(self._n, dtype=torch.long)

        # Pre-compute features
        self._sample_feats: torch.Tensor | None = None
        self._sample_feats_mask: np.ndarray | None = None
        self._target_feats: torch.Tensor | None = None
        self._target_feats_mask: np.ndarray | None = None
        if feature_df is not None:
            self._sample_feats, self._sample_feats_mask = _precompute_features(
                df, "sample_image_path", feature_df
            )
            self._target_feats, self._target_feats_mask = _precompute_features(
                df, "target_image_path", feature_df
            )

        # Store paths only if images are needed on-demand
        self._sample_img_path = (
            df["sample_image_path"].values if load_images else None
        )
        self._target_img_path = (
            df["target_image_path"].values if load_images else None
        )

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> dict:
        out: dict = {
            "task": "match_to_sample",
            "subject_idx": self._subject_idx[idx],
            "trial_id": self._trial_id[idx],
            "sample_true_n": self._sample_true_n[idx],
            "target_true_n": self._target_true_n[idx],
            "target_type": self._target_type[idx],
            "label": self._label[idx],
            "correct": self._correct[idx],
            "rt_ms": self._rt_ms[idx],
            "grade": self._grade[idx],
            "cohort": self._cohort[idx],
        }

        if self._has_target_digit[idx]:
            out["target_digit"] = self._target_digit[idx]

        if self._load_images and self._sample_img_path is not None:
            p = self._sample_img_path[idx]
            if pd.notna(p):
                out["sample_image"] = _load_image(p)
        if self._load_images and self._target_img_path is not None:
            p = self._target_img_path[idx]
            if pd.notna(p):
                out["target_image"] = _load_image(p)

        if self._sample_feats is not None and self._sample_feats_mask[idx]:
            out["sample_features"] = self._sample_feats[idx]
        if self._target_feats is not None and self._target_feats_mask[idx]:
            out["target_features"] = self._target_feats[idx]

        return out


class DotVerificationDataset(Dataset):
    """Dataset for the dot verification / same-different task."""

    def __init__(
        self,
        df: pd.DataFrame,
        subject_map: SubjectIndexMap,
        feature_df: Optional[pd.DataFrame] = None,
        load_images: bool = False,
    ):
        df = df.reset_index(drop=True)
        self._n = len(df)
        self._load_images = load_images

        self._subject_idx = _subject_idx_tensor(df, subject_map)
        self._true_n = torch.from_numpy(df["true_n"].values.astype(np.int64))
        self._shown_digit = torch.from_numpy(
            df["shown_digit"].values.astype(np.int64)
        )
        self._label = torch.from_numpy(
            df["response_same"].values.astype(np.float32)
        )
        self._correct = torch.from_numpy(
            df["correct"].values.astype(np.int64)
        )
        rt = df["rt_ms"].fillna(-1.0).values.astype(np.float32)
        self._rt_ms = torch.from_numpy(rt)
        self._grade = torch.from_numpy(df["grade"].values.astype(np.int64))
        self._trial_id = df["trial_id"].values.astype(str)
        self._cohort = df["cohort"].values.astype(str)

        # Pre-compute features
        self._feats: torch.Tensor | None = None
        self._feats_mask: np.ndarray | None = None
        if feature_df is not None:
            self._feats, self._feats_mask = _precompute_features(
                df, "dots_image_path", feature_df
            )

        self._img_path = (
            df["dots_image_path"].values if load_images else None
        )

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> dict:
        out: dict = {
            "task": "dot_verification",
            "subject_idx": self._subject_idx[idx],
            "trial_id": self._trial_id[idx],
            "true_n": self._true_n[idx],
            "shown_digit": self._shown_digit[idx],
            "label": self._label[idx],
            "correct": self._correct[idx],
            "rt_ms": self._rt_ms[idx],
            "grade": self._grade[idx],
            "cohort": self._cohort[idx],
        }

        if self._load_images and self._img_path is not None:
            p = self._img_path[idx]
            if pd.notna(p):
                out["image"] = _load_image(p)

        if self._feats is not None and self._feats_mask[idx]:
            out["features"] = self._feats[idx]

        return out


class PanamathDataset(Dataset):
    """Dataset for the Panamath non-symbolic comparison task."""

    def __init__(
        self,
        df: pd.DataFrame,
        subject_map: SubjectIndexMap,
        feature_df: Optional[pd.DataFrame] = None,
        load_images: bool = False,
    ):
        df = df.reset_index(drop=True)
        self._n = len(df)
        self._load_images = load_images

        self._subject_idx = _subject_idx_tensor(df, subject_map)
        self._left_true_n = torch.from_numpy(
            df["left_true_n"].values.astype(np.int64)
        )
        self._right_true_n = torch.from_numpy(
            df["right_true_n"].values.astype(np.int64)
        )
        self._size_diff = torch.from_numpy(
            df["size_diff"].values.astype(np.float32)
        )
        self._spacing_diff = torch.from_numpy(
            df["spacing_diff"].values.astype(np.float32)
        )
        self._label = torch.from_numpy(
            (df["response_side"].values == "right").astype(np.float32)
        )
        self._correct = torch.from_numpy(
            df["correct"].values.astype(np.int64)
        )
        rt = df["rt_ms"].fillna(-1.0).values.astype(np.float32)
        self._rt_ms = torch.from_numpy(rt)
        self._grade = torch.from_numpy(df["grade"].values.astype(np.int64))
        self._trial_id = df["trial_id"].values.astype(str)
        self._cohort = df["cohort"].values.astype(str)
        self._congruency = df["congruency"].values.astype(str)

        # Pre-compute features
        self._left_feats: torch.Tensor | None = None
        self._left_feats_mask: np.ndarray | None = None
        self._right_feats: torch.Tensor | None = None
        self._right_feats_mask: np.ndarray | None = None
        if feature_df is not None:
            self._left_feats, self._left_feats_mask = _precompute_features(
                df, "left_image_path", feature_df
            )
            self._right_feats, self._right_feats_mask = _precompute_features(
                df, "right_image_path", feature_df
            )

        self._left_img_path = (
            df["left_image_path"].values if load_images else None
        )
        self._right_img_path = (
            df["right_image_path"].values if load_images else None
        )

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> dict:
        out: dict = {
            "task": "panamath",
            "subject_idx": self._subject_idx[idx],
            "trial_id": self._trial_id[idx],
            "left_true_n": self._left_true_n[idx],
            "right_true_n": self._right_true_n[idx],
            "size_diff": self._size_diff[idx],
            "spacing_diff": self._spacing_diff[idx],
            "congruency": self._congruency[idx],
            "label": self._label[idx],
            "correct": self._correct[idx],
            "rt_ms": self._rt_ms[idx],
            "grade": self._grade[idx],
            "cohort": self._cohort[idx],
        }

        if self._load_images:
            if self._left_img_path is not None:
                p = self._left_img_path[idx]
                if pd.notna(p):
                    out["left_image"] = _load_image(p)
            if self._right_img_path is not None:
                p = self._right_img_path[idx]
                if pd.notna(p):
                    out["right_image"] = _load_image(p)

        if self._left_feats is not None and self._left_feats_mask[idx]:
            out["left_features"] = self._left_feats[idx]
        if self._right_feats is not None and self._right_feats_mask[idx]:
            out["right_features"] = self._right_feats[idx]

        return out


# ---------------------------------------------------------------------------
# Multitask wrapper
# ---------------------------------------------------------------------------

DATASET_REGISTRY: dict[str, type[Dataset]] = {
    "symbolic_comparison": SymbolicComparisonDataset,
    "match_to_sample": MatchToSampleDataset,
    "dot_verification": DotVerificationDataset,
    "panamath": PanamathDataset,
}


class MultitaskDataset(ConcatDataset):
    """Concatenates multiple per-task datasets into one.

    Samples are drawn uniformly across the concatenated pool.
    For weighted task sampling, use MultitaskSampler instead.
    """

    def __init__(self, datasets: dict[str, Dataset]):
        self.task_datasets = datasets
        super().__init__(list(datasets.values()))


class MultitaskSampler(torch.utils.data.Sampler):
    """Weighted random sampler that respects per-task weights.

    At each draw, first picks a task proportional to its weight,
    then picks a random trial from that task.
    """

    def __init__(
        self,
        multitask_dataset: MultitaskDataset,
        task_weights: dict[str, float],
        num_samples: Optional[int] = None,
        seed: int = 17,
    ):
        self.mt = multitask_dataset
        self.seed = seed
        self._epoch = 0

        task_names = list(self.mt.task_datasets.keys())
        sizes = [len(self.mt.task_datasets[t]) for t in task_names]
        weights = [task_weights.get(t, 1.0) for t in task_names]

        sample_weights = []
        for w, s in zip(weights, sizes):
            sample_weights.extend([w / s] * s)
        self._weights = torch.tensor(sample_weights, dtype=torch.float64)

        self._num_samples = num_samples or len(self.mt)

    def set_epoch(self, epoch: int) -> None:
        """Vary the random order each epoch."""
        self._epoch = epoch

    def __iter__(self):
        gen = torch.Generator()
        gen.manual_seed(self.seed + self._epoch)
        indices = torch.multinomial(
            self._weights, self._num_samples, replacement=True, generator=gen
        )
        yield from indices.tolist()

    def __len__(self) -> int:
        return self._num_samples


def build_dataloader(
    datasets: dict[str, Dataset],
    task_weights: Optional[dict[str, float]] = None,
    batch_size: int = 256,
    num_workers: int = 0,
    pin_memory: bool = False,
    seed: int = 17,
) -> DataLoader:
    """Build a DataLoader over one or more task datasets.

    If task_weights is provided, uses MultitaskSampler for weighted
    sampling. Otherwise, shuffles uniformly.
    """
    from src.data.collate import multitask_collate

    mt = MultitaskDataset(datasets)
    persistent = num_workers > 0

    if task_weights is not None:
        sampler = MultitaskSampler(mt, task_weights, seed=seed)
        return DataLoader(
            mt,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent,
            prefetch_factor=2 if num_workers > 0 else None,
            collate_fn=multitask_collate,
        )

    return DataLoader(
        mt,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
        prefetch_factor=2 if num_workers > 0 else None,
        collate_fn=multitask_collate,
    )
