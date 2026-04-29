"""Tests for WP-B: data layer, splits, datasets, collation."""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.schemas import (
    TASK_NAMES,
    SCHEMA_REGISTRY,
    FEATURE_COLUMNS,
    SYMBOLIC_COMPARISON_COLS,
    DOT_VERIFICATION_COLS,
    MATCH_TO_SAMPLE_COLS,
    PANAMATH_COLS,
)
from src.data.splits import subject_split, filter_by_subjects
from src.data.dataset_builders import (
    SubjectIndexMap,
    SymbolicComparisonDataset,
    MatchToSampleDataset,
    DotVerificationDataset,
    PanamathDataset,
    MultitaskDataset,
    MultitaskSampler,
    build_dataloader,
)
from src.data.collate import multitask_collate, single_task_collate


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _make_subjects(n: int = 30) -> pd.DataFrame:
    rng = np.random.RandomState(7)
    return pd.DataFrame({
        "subject_id": [f"S{i:03d}" for i in range(n)],
        "cohort": rng.choice(["A", "B"], n),
        "grade": rng.choice([1, 2], n),
    })


def _make_symbolic(subjects: pd.DataFrame, trials_per: int = 5) -> pd.DataFrame:
    rng = np.random.RandomState(8)
    rows = []
    for _, s in subjects.iterrows():
        for t in range(trials_per):
            ld = rng.randint(0, 10)
            rd = rng.randint(0, 10)
            rows.append({
                "subject_id": s["subject_id"],
                "trial_id": f"{s['subject_id']}_sym_{t}",
                "left_digit": ld,
                "right_digit": rd,
                "response_side": rng.choice(["left", "right"]),
                "correct": int(rng.rand() > 0.3),
                "rt_ms": rng.exponential(500),
                "cohort": s["cohort"],
                "grade": s["grade"],
            })
    return pd.DataFrame(rows)


def _make_dot_verification(subjects: pd.DataFrame, trials_per: int = 5) -> pd.DataFrame:
    rng = np.random.RandomState(9)
    rows = []
    for _, s in subjects.iterrows():
        for t in range(trials_per):
            tn = rng.randint(4, 9)
            sd = rng.randint(4, 10)
            rows.append({
                "subject_id": s["subject_id"],
                "trial_id": f"{s['subject_id']}_dot_{t}",
                "dots_image_path": f"img/dots_{tn}_{t}.png",
                "true_n": tn,
                "shown_digit": sd,
                "response_same": int(tn == sd) if rng.rand() > 0.2 else int(tn != sd),
                "correct": int(tn == sd),
                "rt_ms": rng.exponential(700),
                "cohort": s["cohort"],
                "grade": s["grade"],
            })
    return pd.DataFrame(rows)


def _make_match_to_sample(subjects: pd.DataFrame, trials_per: int = 4) -> pd.DataFrame:
    rng = np.random.RandomState(10)
    rows = []
    for _, s in subjects.iterrows():
        for t in range(trials_per):
            sn = rng.randint(1, 10)
            ttype = rng.choice(["dots", "digit"])
            tn = rng.randint(1, 10)
            rows.append({
                "subject_id": s["subject_id"],
                "trial_id": f"{s['subject_id']}_mts_{t}",
                "sample_type": "dots",
                "sample_image_path": f"img/sample_{sn}_{t}.png",
                "sample_true_n": sn,
                "target_type": ttype,
                "target_image_path": f"img/target_{tn}_{t}.png" if ttype == "dots" else None,
                "target_digit": float(tn) if ttype == "digit" else float("nan"),
                "target_true_n": tn,
                "response_same": int(sn == tn) if rng.rand() > 0.2 else int(sn != tn),
                "correct": int(sn == tn),
                "rt_ms": rng.exponential(600),
                "cohort": s["cohort"],
                "grade": s["grade"],
            })
    return pd.DataFrame(rows)


def _make_panamath(subjects: pd.DataFrame, trials_per: int = 4) -> pd.DataFrame:
    rng = np.random.RandomState(11)
    rows = []
    for _, s in subjects.iterrows():
        for t in range(trials_per):
            ln = rng.randint(5, 20)
            rn = rng.randint(5, 20)
            rows.append({
                "subject_id": s["subject_id"],
                "trial_id": f"{s['subject_id']}_pan_{t}",
                "left_image_path": f"img/left_{ln}_{t}.png",
                "right_image_path": f"img/right_{rn}_{t}.png",
                "left_true_n": ln,
                "right_true_n": rn,
                "size_diff": rng.randn(),
                "spacing_diff": rng.randn(),
                "congruency": rng.choice(["congruent", "incongruent"]),
                "response_side": rng.choice(["left", "right"]),
                "correct": int(rng.rand() > 0.3),
                "rt_ms": rng.exponential(800),
                "cohort": s["cohort"],
                "grade": s["grade"],
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------

def test_schema_registry_keys():
    assert "subjects" in SCHEMA_REGISTRY
    for t in TASK_NAMES:
        assert t in SCHEMA_REGISTRY


def test_feature_columns():
    assert len(FEATURE_COLUMNS) == 6
    assert "total_area" in FEATURE_COLUMNS


# ---------------------------------------------------------------------------
# SubjectIndexMap tests
# ---------------------------------------------------------------------------

def test_subject_index_map():
    ids = ["S002", "S000", "S001", "S000"]
    m = SubjectIndexMap(ids)
    assert m.n_subjects == 3
    assert m.to_idx("S000") == 0
    assert m.to_id(0) == "S000"
    assert set(m.all_ids) == {"S000", "S001", "S002"}


# ---------------------------------------------------------------------------
# Split tests
# ---------------------------------------------------------------------------

def test_subject_split_covers_all():
    subs = _make_subjects(50)
    splits = subject_split(subs["subject_id"].unique(), seed=13)
    all_assigned = set(splits["train"]) | set(splits["val"]) | set(splits["test"])
    assert all_assigned == set(subs["subject_id"])


def test_subject_split_no_overlap():
    subs = _make_subjects(100)
    sp = subject_split(subs["subject_id"].unique(), seed=5)
    assert len(set(sp["train"]) & set(sp["val"])) == 0
    assert len(set(sp["train"]) & set(sp["test"])) == 0
    assert len(set(sp["val"]) & set(sp["test"])) == 0


def test_subject_split_reproducible():
    ids = np.array([f"S{i}" for i in range(40)])
    s1 = subject_split(ids, seed=77)
    s2 = subject_split(ids, seed=77)
    assert list(s1["train"]) == list(s2["train"])


def test_subject_split_stratified():
    ids = np.array([f"S{i}" for i in range(60)])
    grades = np.array([1] * 30 + [2] * 30)
    sp = subject_split(ids, seed=3, stratify_by=grades)
    all_assigned = set(sp["train"]) | set(sp["val"]) | set(sp["test"])
    assert all_assigned == set(ids)


def test_filter_by_subjects():
    subs = _make_subjects(20)
    df = _make_symbolic(subs)
    keep = subs["subject_id"].values[:5]
    filtered = filter_by_subjects(df, keep)
    assert set(filtered["subject_id"].unique()) == set(keep)


# ---------------------------------------------------------------------------
# Dataset tests
# ---------------------------------------------------------------------------

def test_symbolic_dataset_item():
    subs = _make_subjects(10)
    df = _make_symbolic(subs)
    m = SubjectIndexMap(subs["subject_id"])
    ds = SymbolicComparisonDataset(df, m)
    item = ds[0]
    assert item["task"] == "symbolic_comparison"
    assert "subject_idx" in item
    assert "left_digit" in item
    assert "label" in item
    assert isinstance(item["label"], (float, torch.Tensor))


def test_dot_verification_dataset_item():
    subs = _make_subjects(10)
    df = _make_dot_verification(subs)
    m = SubjectIndexMap(subs["subject_id"])
    ds = DotVerificationDataset(df, m)
    item = ds[0]
    assert item["task"] == "dot_verification"
    assert "true_n" in item
    assert "shown_digit" in item


def test_match_to_sample_dataset_item():
    subs = _make_subjects(10)
    df = _make_match_to_sample(subs)
    m = SubjectIndexMap(subs["subject_id"])
    ds = MatchToSampleDataset(df, m)
    item = ds[0]
    assert item["task"] == "match_to_sample"
    assert "sample_true_n" in item
    assert "target_type" in item


def test_panamath_dataset_item():
    subs = _make_subjects(10)
    df = _make_panamath(subs)
    m = SubjectIndexMap(subs["subject_id"])
    ds = PanamathDataset(df, m)
    item = ds[0]
    assert item["task"] == "panamath"
    assert "left_true_n" in item
    assert "size_diff" in item


# ---------------------------------------------------------------------------
# Collation tests
# ---------------------------------------------------------------------------

def test_single_task_collate():
    subs = _make_subjects(5)
    df = _make_symbolic(subs)
    m = SubjectIndexMap(subs["subject_id"])
    ds = SymbolicComparisonDataset(df, m)
    batch = [ds[i] for i in range(4)]
    collated = single_task_collate(batch)
    assert collated["subject_idx"].shape == (4,)
    assert collated["left_digit"].shape == (4,)
    assert isinstance(collated["task"], list)


def test_multitask_collate():
    subs = _make_subjects(10)
    m = SubjectIndexMap(subs["subject_id"])
    ds_sym = SymbolicComparisonDataset(_make_symbolic(subs), m)
    ds_dot = DotVerificationDataset(_make_dot_verification(subs), m)
    batch = [ds_sym[0], ds_sym[1], ds_dot[0], ds_dot[1]]
    collated = multitask_collate(batch)
    assert "symbolic_comparison" in collated
    assert "dot_verification" in collated
    assert collated["symbolic_comparison"]["subject_idx"].shape == (2,)
    assert collated["dot_verification"]["subject_idx"].shape == (2,)


# ---------------------------------------------------------------------------
# MultitaskDataset / Sampler / DataLoader tests
# ---------------------------------------------------------------------------

def test_multitask_dataset_length():
    subs = _make_subjects(10)
    m = SubjectIndexMap(subs["subject_id"])
    ds_sym = SymbolicComparisonDataset(_make_symbolic(subs, 3), m)
    ds_dot = DotVerificationDataset(_make_dot_verification(subs, 2), m)
    mt = MultitaskDataset({"symbolic_comparison": ds_sym, "dot_verification": ds_dot})
    assert len(mt) == len(ds_sym) + len(ds_dot)


def test_multitask_sampler_length():
    subs = _make_subjects(10)
    m = SubjectIndexMap(subs["subject_id"])
    ds_sym = SymbolicComparisonDataset(_make_symbolic(subs, 3), m)
    ds_dot = DotVerificationDataset(_make_dot_verification(subs, 2), m)
    mt = MultitaskDataset({"symbolic_comparison": ds_sym, "dot_verification": ds_dot})
    sampler = MultitaskSampler(mt, {"symbolic_comparison": 2.0, "dot_verification": 1.0})
    assert len(sampler) == len(mt)


def test_build_dataloader_iterates():
    subs = _make_subjects(10)
    m = SubjectIndexMap(subs["subject_id"])
    datasets = {
        "symbolic_comparison": SymbolicComparisonDataset(_make_symbolic(subs, 3), m),
        "dot_verification": DotVerificationDataset(_make_dot_verification(subs, 2), m),
    }
    loader = build_dataloader(datasets, batch_size=16, seed=7)
    batch = next(iter(loader))
    # batch is a dict keyed by task name
    assert isinstance(batch, dict)
    total_samples = sum(
        b["subject_idx"].shape[0] for b in batch.values()
    )
    assert total_samples == 16


def test_build_dataloader_with_weights():
    subs = _make_subjects(10)
    m = SubjectIndexMap(subs["subject_id"])
    datasets = {
        "symbolic_comparison": SymbolicComparisonDataset(_make_symbolic(subs, 3), m),
        "dot_verification": DotVerificationDataset(_make_dot_verification(subs, 2), m),
    }
    weights = {"symbolic_comparison": 3.0, "dot_verification": 1.0}
    loader = build_dataloader(datasets, task_weights=weights, batch_size=32, seed=7)
    batch = next(iter(loader))
    assert isinstance(batch, dict)
