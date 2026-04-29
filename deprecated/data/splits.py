"""Subject-based train / val / test splitting.

Splits are always by subject so no child appears in more than one fold.
"""

from typing import Optional

import numpy as np
import pandas as pd


def subject_split(
    subject_ids: np.ndarray | pd.Series,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 17,
    stratify_by: Optional[np.ndarray | pd.Series] = None,
) -> dict[str, np.ndarray]:
    """Split unique subjects into train / val / test sets.

    Args:
        subject_ids: Array of all unique subject IDs.
        train_frac: Fraction for training.
        val_frac: Fraction for validation. Test gets the remainder.
        seed: Random seed for reproducibility.
        stratify_by: Optional array (same length as subject_ids) for
            stratified splitting (e.g. grade). Simple approximate
            stratification — not exact for very small strata.

    Returns:
        Dict with keys "train", "val", "test", each mapping to an
        array of subject IDs.
    """
    ids = np.asarray(subject_ids)
    unique_ids = np.unique(ids)
    rng = np.random.RandomState(seed)

    if stratify_by is not None:
        strata = np.asarray(stratify_by)
        assert len(strata) == len(unique_ids), (
            "stratify_by must have same length as unique subject_ids"
        )
        train_ids, val_ids, test_ids = [], [], []
        for stratum in np.unique(strata):
            mask = strata == stratum
            stratum_ids = unique_ids[mask]
            rng.shuffle(stratum_ids)
            n = len(stratum_ids)
            n_train = max(1, int(round(n * train_frac)))
            n_val = max(1, int(round(n * val_frac)))
            n_val = min(n_val, n - n_train)
            train_ids.append(stratum_ids[:n_train])
            val_ids.append(stratum_ids[n_train : n_train + n_val])
            test_ids.append(stratum_ids[n_train + n_val :])
        return {
            "train": np.concatenate(train_ids),
            "val": np.concatenate(val_ids),
            "test": np.concatenate(test_ids),
        }

    perm = rng.permutation(unique_ids)
    n = len(perm)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    return {
        "train": perm[:n_train],
        "val": perm[n_train : n_train + n_val],
        "test": perm[n_train + n_val :],
    }


def filter_by_subjects(
    df: pd.DataFrame,
    subject_ids: np.ndarray,
    subject_col: str = "subject_id",
) -> pd.DataFrame:
    """Keep only rows whose subject is in the given set."""
    return df[df[subject_col].isin(set(subject_ids))].reset_index(drop=True)
