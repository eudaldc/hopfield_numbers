"""Subject index alignment for viz demographics merge."""

from pathlib import Path

import pandas as pd
import torch

from src.common.config import ExperimentConfig
from src.training.engine import build_modules
from src.training.evaluators import export_subject_params
from src.viz.subject_demographics import attach_demographics


def test_attach_demographics_merges_grade(tmp_path: Path):
    data_dir = tmp_path
    sym = pd.DataFrame(
        {
            "subject_id": ["b_sub", "a_sub"],
            "trial_id": ["t1", "t2"],
            "left_digit": [1, 2],
            "right_digit": [2, 3],
            "response_side": ["right", "right"],
            "correct": [1, 1],
            "rt_ms": [100.0, 200.0],
            "cohort": ["A", "A"],
            "grade": [1, 1],
        }
    )
    sym.to_parquet(data_dir / "symbolic_comparison.parquet")

    subjects = pd.DataFrame(
        {
            "subject_id": ["a_sub", "b_sub"],
            "grade": [2, 1],
            "arithmetic_score": [10.0, 20.0],
        }
    )

    cfg = ExperimentConfig()
    device = torch.device("cpu")
    n = 2
    modules = build_modules(cfg, n, device)
    param_df = export_subject_params(
        modules["attractor_core"], modules["subject_params"], n
    )

    out, msgs = attach_demographics(param_df, n, subjects, data_dir, seed=17)
    assert "subject_id" in out.columns
    assert out["subject_id"].tolist() == ["a_sub", "b_sub"]
    assert out["grade"].tolist() == [2, 1]
    assert out["arithmetic_score"].tolist() == [10.0, 20.0]
    assert any("Merged" in m for m in msgs)


def test_attach_demographics_n_subjects_mismatch(tmp_path: Path):
    data_dir = tmp_path
    sym = pd.DataFrame(
        {
            "subject_id": ["only_one"],
            "trial_id": ["t1"],
            "left_digit": [1],
            "right_digit": [2],
            "response_side": ["right"],
            "correct": [1],
            "rt_ms": [100.0],
            "cohort": ["A"],
            "grade": [1],
        }
    )
    sym.to_parquet(data_dir / "symbolic_comparison.parquet")
    subjects = pd.DataFrame({"subject_id": ["only_one"], "grade": [1]})

    cfg = ExperimentConfig()
    device = torch.device("cpu")
    modules = build_modules(cfg, 3, device)
    param_df = export_subject_params(
        modules["attractor_core"], modules["subject_params"], 3
    )

    out, msgs = attach_demographics(param_df, 3, subjects, data_dir, seed=17)
    assert "grade" not in out.columns or out["grade"].isna().all()
    assert any("n_subjects" in m for m in msgs)
