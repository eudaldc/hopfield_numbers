"""ETL: convert raw CSVs into clean parquet files matching the schema.

Raw inputs (in data/):
  - cleaned_statements.csv   trial-level data for all tasks
  - cleaned_scores.csv       subject-level aggregate scores
  - dot_enumeration.csv      trial design for dot verification
  - number_knowledge.csv     trial design for symbolic comparison
  - math_to_sample.csv       trial design for match-to-sample

Outputs (in data/processed/):
  - subjects.parquet
  - symbolic_comparison.parquet
  - dot_verification.parquet
  - match_to_sample.parquet
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.schemas import FEATURE_COLUMNS


RAW_DIR = Path("data")
OUT_DIR = Path("data/processed")


def _generate_synthetic_features(
    true_n: int, rng: np.random.Generator
) -> dict[str, float]:
    """Generate plausible stimulus features for a dot display of *true_n* dots.

    These approximate the statistical regularities of controlled dot displays
    used in numerosity experiments (area, spacing, hull, etc.).
    """
    n = float(true_n)
    base_dot_size = 50.0
    mean_dot_size = base_dot_size * (1.0 + rng.normal(0, 0.1))
    total_area = n * mean_dot_size * (1.0 + rng.normal(0, 0.05))
    convex_hull = 1000.0 * n ** 0.7 * (1.0 + rng.normal(0, 0.1))
    density = n / max(convex_hull, 1.0)
    spacing = 200.0 / max(np.sqrt(n), 1.0) * (1.0 + rng.normal(0, 0.1))
    fourier_power = 5.0 * np.log1p(n) * (1.0 + rng.normal(0, 0.05))

    return {
        "total_area": max(total_area, 0.0),
        "mean_dot_size": max(mean_dot_size, 1.0),
        "convex_hull": max(convex_hull, 0.0),
        "density": max(density, 0.0),
        "spacing": max(spacing, 0.0),
        "fourier_power": max(fourier_power, 0.0),
    }


def build_stimulus_features(
    image_paths: list[str],
    true_ns: list[int],
    seed: int = 73,
) -> pd.DataFrame:
    """Build a stimulus-features table for synthetic image paths."""
    rng = np.random.default_rng(seed)
    records = []
    for path, true_n in zip(image_paths, true_ns):
        feats = _generate_synthetic_features(true_n, rng)
        feats["image_path"] = path
        feats["true_n"] = int(true_n)
        records.append(feats)
    df = pd.DataFrame(records)
    return df[["image_path", "true_n"] + FEATURE_COLUMNS]


def build_subjects(statements: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    """Build subject table from aggregate scores."""
    pivot = scores.pivot_table(
        index=["student_uuid", "grade"],
        columns="cognitive_factor",
        values="score",
    ).reset_index()

    grade_map = {"grade 1": 1, "grade 2": 2}
    pivot["grade"] = pivot["grade"].map(grade_map)

    col_map = {
        "student_uuid": "subject_id",
        "grade": "grade",
        "arithmetic_fluency": "arithmetic_score",
        "numerical_line": "number_line_score",
        "reasoning": "raven_score",
        "working_memory": "wm_score",
        "dot_enumeration": "dot_enumeration_score",
        "magnitude_comparison": "magnitude_comparison_score",
        "number_knowledge": "number_knowledge_score",
    }
    keep = [c for c in col_map if c in pivot.columns]
    subjects = pivot[keep].rename(columns=col_map)
    subjects["cohort"] = "A"

    return subjects


def build_symbolic_comparison(
    statements: pd.DataFrame, design: pd.DataFrame
) -> pd.DataFrame:
    """Build symbolic comparison (number_knowledge) dataset."""
    nk = statements[statements["cognitive_factor"] == "number_knowledge"].copy()
    nk = nk.drop(columns=["cognitive_factor"])

    positions = sorted(nk["statement_position"].unique())
    rank_map = {p: i + 1 for i, p in enumerate(positions)}
    nk["statement_position"] = nk["statement_position"].map(rank_map)

    df = nk.merge(design, on="statement_position", how="inner")

    def derive_response_side(row):
        if row["score"] == 1:
            return row["Bigger"].lower()
        return "right" if row["Bigger"].lower() == "left" else "left"

    df["response_side"] = df.apply(derive_response_side, axis=1)

    grade_map = {"grade 1": 1, "grade 2": 2}

    result = pd.DataFrame({
        "subject_id": df["student_uuid"],
        "trial_id": df["statement_uuid"],
        "left_digit": df["Left"].astype(int),
        "right_digit": df["Right"].astype(int),
        "response_side": df["response_side"],
        "correct": df["score"].astype(int),
        "rt_ms": df["statement_time_seconds"] * 1000,
        "cohort": "A",
        "grade": df["grade"].map(grade_map),
    })
    return result


def build_dot_verification(
    statements: pd.DataFrame, design: pd.DataFrame
) -> pd.DataFrame:
    """Build dot verification (dot_enumeration) dataset."""
    de = statements[statements["cognitive_factor"] == "dot_enumeration"].copy()
    de = de.drop(columns=["cognitive_factor"])
    de["statement_position"] = de["statement_position"].astype(int) - 5

    design_test = design[design["Type"] == "Test"][
        ["statement_position", "ratio", "dots", "number", "equal", "dif"]
    ].copy()
    design_test["ratio"] = pd.to_numeric(
        design_test["ratio"].astype(str).str.replace(",", "."), errors="coerce"
    )

    df = de.merge(design_test, on="statement_position", how="inner")
    df = df[df["statement_time_seconds"] < 30].copy()

    equal_bool = df["equal"].astype(str).str.strip().str.lower().isin(["true", "1"])
    score_bool = df["score"].astype(float) == 1.0
    df["response_same"] = (equal_bool == score_bool).astype(int)

    grade_map = {"grade 1": 1, "grade 2": 2}

    n_rows = len(df)
    image_paths = [f"synthetic/dot_verification/{i:06d}.png" for i in range(n_rows)]

    result = pd.DataFrame({
        "subject_id": df["student_uuid"],
        "trial_id": df["statement_uuid"],
        "dots_image_path": image_paths,
        "true_n": df["dots"].astype(int),
        "shown_digit": df["number"].astype(int),
        "response_same": df["response_same"],
        "correct": df["score"].astype(int),
        "rt_ms": df["statement_time_seconds"] * 1000,
        "cohort": "A",
        "grade": df["grade"].map(grade_map),
    })
    return result


def build_match_to_sample(
    statements: pd.DataFrame, design: pd.DataFrame
) -> pd.DataFrame:
    """Build match-to-sample (magnitude_comparison) dataset."""
    mc = statements[statements["cognitive_factor"] == "magnitude_comparison"].copy()
    mc = mc.drop(columns=["cognitive_factor"])
    mc = mc[mc["statement_position"] > 14].copy()

    positions = sorted(mc["statement_position"].unique())
    rank_map = {p: i + 1 for i, p in enumerate(positions)}
    mc["num"] = mc["statement_position"].map(rank_map)

    design_test = design[design["Type"] == "Test"][
        ["num", "ratio", "sample", "match", "correct", "equalized"]
    ].copy()
    design_test["ratio"] = pd.to_numeric(
        design_test["ratio"].astype(str).str.replace(",", "."), errors="coerce"
    )
    design_test["sample"] = design_test["sample"].astype(int)
    design_test["match"] = design_test["match"].astype(int)

    df = mc.merge(design_test, on="num", how="inner")

    correct_bool = df["correct"].astype(str).str.strip().str.upper().isin(["TRUE", "1"])
    score_bool = df["score"].astype(float) == 1.0
    df["response_same"] = (correct_bool == score_bool).astype(int)

    grade_map = {"grade 1": 1, "grade 2": 2}

    n_rows = len(df)
    sample_paths = [f"synthetic/mts_sample/{i:06d}.png" for i in range(n_rows)]
    target_paths = [f"synthetic/mts_target/{i:06d}.png" for i in range(n_rows)]

    result = pd.DataFrame({
        "subject_id": df["student_uuid"],
        "trial_id": df["statement_uuid"],
        "sample_type": "dots",
        "sample_image_path": sample_paths,
        "sample_true_n": df["sample"],
        "target_type": "dots",
        "target_image_path": target_paths,
        "target_digit": np.nan,
        "target_true_n": df["match"],
        "response_same": df["response_same"],
        "correct": df["score"].astype(int),
        "rt_ms": df["statement_time_seconds"] * 1000,
        "cohort": "A",
        "grade": df["grade"].map(grade_map),
    })
    return result


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading raw data...")
    statements = pd.read_csv(RAW_DIR / "cleaned_statements.csv")
    scores = pd.read_csv(RAW_DIR / "cleaned_scores.csv")
    de_design = pd.read_csv(RAW_DIR / "dot_enumeration.csv", decimal=",")
    nk_design = pd.read_csv(RAW_DIR / "number_knowledge.csv")
    mts_design = pd.read_csv(RAW_DIR / "math_to_sample.csv")

    print("Building subjects...")
    subjects = build_subjects(statements, scores)
    subjects.to_parquet(OUT_DIR / "subjects.parquet", index=False)
    print(f"  -> {len(subjects)} subjects")

    print("Building symbolic_comparison...")
    sym = build_symbolic_comparison(statements, nk_design)
    sym.to_parquet(OUT_DIR / "symbolic_comparison.parquet", index=False)
    print(f"  -> {len(sym)} trials, {sym['subject_id'].nunique()} subjects")

    print("Building dot_verification...")
    dot = build_dot_verification(statements, de_design)
    dot.to_parquet(OUT_DIR / "dot_verification.parquet", index=False)
    print(f"  -> {len(dot)} trials, {dot['subject_id'].nunique()} subjects")

    print("Building match_to_sample...")
    mts = build_match_to_sample(statements, mts_design)
    mts.to_parquet(OUT_DIR / "match_to_sample.parquet", index=False)
    print(f"  -> {len(mts)} trials, {mts['subject_id'].nunique()} subjects")

    print("Building stimulus_features...")
    all_paths: list[str] = []
    all_ns: list[int] = []

    all_paths.extend(dot["dots_image_path"].tolist())
    all_ns.extend(dot["true_n"].tolist())

    all_paths.extend(mts["sample_image_path"].tolist())
    all_ns.extend(mts["sample_true_n"].tolist())
    all_paths.extend(mts["target_image_path"].tolist())
    all_ns.extend(mts["target_true_n"].tolist())

    stim_feat = build_stimulus_features(all_paths, all_ns)
    stim_feat.to_parquet(OUT_DIR / "stimulus_features.parquet", index=False)
    print(f"  -> {len(stim_feat)} entries")

    # Quick sanity checks
    print("\n=== Sanity checks ===")
    print(f"Symbolic comparison accuracy: {sym['correct'].mean():.3f}")
    print(f"Dot verification accuracy: {dot['correct'].mean():.3f}")
    print(f"Match-to-sample accuracy: {mts['correct'].mean():.3f}")

    print(f"\nDot verification true_n range: {dot['true_n'].min()}-{dot['true_n'].max()}")
    print(f"Dot verification shown_digit range: {dot['shown_digit'].min()}-{dot['shown_digit'].max()}")
    print(f"Symbolic comparison digit range: {sym['left_digit'].min()}-{sym['right_digit'].max()}")
    print(f"MTS sample range: {mts['sample_true_n'].min()}-{mts['sample_true_n'].max()}")
    print(f"MTS target range: {mts['target_true_n'].min()}-{mts['target_true_n'].max()}")

    print(f"\nMedian RT (ms):")
    print(f"  symbolic: {sym['rt_ms'].median():.0f}")
    print(f"  dot_ver:  {dot['rt_ms'].median():.0f}")
    print(f"  mts:      {mts['rt_ms'].median():.0f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
