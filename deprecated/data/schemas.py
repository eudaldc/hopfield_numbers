"""Expected column schemas for each task parquet file.

Each schema is a plain dict mapping column name -> expected dtype string.
These are used for validation when loading data, not for enforcement.
"""

SUBJECT_COLS = {
    "subject_id": "str",
    "cohort": "str",
    "grade": "int",
    "age_months": "float",  # optional
    "sex": "str",  # optional
    "raven_score": "float",
    "wm_score": "float",
    "arithmetic_score": "float",
    "number_line_score": "float",
}

SYMBOLIC_COMPARISON_COLS = {
    "subject_id": "str",
    "trial_id": "str",
    "left_digit": "int",
    "right_digit": "int",
    "response_side": "str",  # "left" or "right"
    "correct": "int",
    "rt_ms": "float",
    "cohort": "str",
    "grade": "int",
}

MATCH_TO_SAMPLE_COLS = {
    "subject_id": "str",
    "trial_id": "str",
    "sample_type": "str",  # "dots"
    "sample_image_path": "str",
    "sample_true_n": "int",
    "target_type": "str",  # "dots" or "digit"
    "target_image_path": "str",  # nullable
    "target_digit": "float",  # nullable → float to allow NaN
    "target_true_n": "int",
    "response_same": "int",
    "correct": "int",
    "rt_ms": "float",
    "cohort": "str",
    "grade": "int",
}

DOT_VERIFICATION_COLS = {
    "subject_id": "str",
    "trial_id": "str",
    "dots_image_path": "str",
    "true_n": "int",
    "shown_digit": "int",
    "response_same": "int",
    "correct": "int",
    "rt_ms": "float",
    "cohort": "str",
    "grade": "int",
}

PANAMATH_COLS = {
    "subject_id": "str",
    "trial_id": "str",
    "left_image_path": "str",
    "right_image_path": "str",
    "left_true_n": "int",
    "right_true_n": "int",
    "size_diff": "float",
    "spacing_diff": "float",
    "congruency": "str",
    "response_side": "str",
    "correct": "int",
    "rt_ms": "float",
    "cohort": "str",
    "grade": "int",
}

STIMULUS_FEATURES_COLS = {
    "image_path": "str",
    "true_n": "int",
    "total_area": "float",
    "mean_dot_size": "float",
    "convex_hull": "float",
    "density": "float",
    "spacing": "float",
    "fourier_power": "float",
}

# Ordered list of feature columns used by the visual encoder
FEATURE_COLUMNS = [
    "total_area",
    "mean_dot_size",
    "convex_hull",
    "density",
    "spacing",
    "fourier_power",
]

TASK_NAMES = [
    "symbolic_comparison",
    "match_to_sample",
    "dot_verification",
    "panamath",
]

SCHEMA_REGISTRY: dict[str, dict[str, str]] = {
    "subjects": SUBJECT_COLS,
    "symbolic_comparison": SYMBOLIC_COMPARISON_COLS,
    "match_to_sample": MATCH_TO_SAMPLE_COLS,
    "dot_verification": DOT_VERIFICATION_COLS,
    "panamath": PANAMATH_COLS,
    "stimulus_features": STIMULUS_FEATURES_COLS,
}
