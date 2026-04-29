"""Align ``export_subject_params`` rows with ``subjects.parquet`` demographics.

Training indexes subjects by ``SubjectIndexMap`` built from task parquets, not by
row order in ``subjects.parquet``.  Viz must recover ``subject_id`` per
``subject_idx`` the same way before merging grade or cognitive scores.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from src.training.engine import DEFAULT_CORE_TASKS, load_task_data

DEMOGRAPHIC_COLUMNS: tuple[str, ...] = (
    "grade",
    "arithmetic_score",
    "number_line_score",
    "raven_score",
    "wm_score",
)


def attach_demographics(
    param_df: pd.DataFrame,
    n_subjects: int,
    subjects: pd.DataFrame | None,
    data_dir: str | Path,
    seed: int = 17,
    tasks: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Return a copy of *param_df* with ``subject_id`` and demographic columns.

    Parameters
    ----------
    param_df
        Output of ``export_subject_params`` (rows ordered by ``subject_idx``).
    n_subjects
        Number of subject embeddings in the loaded model (must match the map).
    subjects
        Subject-level table (e.g. from ``subjects.parquet``).
    data_dir
        Directory containing task parquets (same layout as training).
    seed
        Passed to ``load_task_data`` (only affects splits, not the index map).
    tasks
        Task names for building the map; default matches ``TrainingConfig().tasks``
        entries that have parquet files present.

    Returns
    -------
    enriched_df, log_messages
    """
    msgs: list[str] = []
    out = param_df.copy()
    data_dir = Path(data_dir)

    if subjects is None:
        msgs.append("No subjects table loaded; skipping demographics merge.")
        return out, msgs

    task_list = list(tasks) if tasks is not None else list(DEFAULT_CORE_TASKS)
    try:
        *_, subject_map = load_task_data(data_dir, task_list, seed=seed)
    except FileNotFoundError as e:
        msgs.append(
            f"Could not build subject index from task parquets ({e}); "
            "skipping demographics merge."
        )
        return out, msgs

    if len(subject_map) != n_subjects:
        msgs.append(
            f"SubjectIndexMap from task data has {len(subject_map)} subjects but "
            f"the model has n_subjects={n_subjects}; skipping merge "
            "(use the same data_dir as training, or a matching checkpoint)."
        )
        return out, msgs

    out["subject_id"] = [subject_map.to_id(i) for i in range(n_subjects)]
    sub = subjects.drop_duplicates(subset=["subject_id"], keep="first").set_index(
        "subject_id"
    )

    for col in DEMOGRAPHIC_COLUMNS:
        if col in sub.columns:
            out[col] = out["subject_id"].map(sub[col])

    merged = [c for c in DEMOGRAPHIC_COLUMNS if c in out.columns and out[c].notna().any()]
    if merged:
        msgs.append("Merged from subjects table: " + ", ".join(merged) + ".")
    else:
        msgs.append(
            "subject_id aligned, but demographic columns are all missing or NaN "
            f"(check subjects.parquet for {list(DEMOGRAPHIC_COLUMNS)})."
        )
    return out, msgs
