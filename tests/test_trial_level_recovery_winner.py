import pandas as pd
import pytest

from scripts import trial_level_model_comparison as T
from scripts import trial_level_recovery_winner as recovery


def _joint_recovery_frame() -> pd.DataFrame:
    rows = []
    for rep in range(2):
        sigma_c = 0.018 + 0.002 * rep
        pow_ = 1.8 + 0.1 * rep
        n_width = recovery.n_width_sigma1(sigma_c, pow_)
        sigma3 = sigma_c * 3.0**pow_
        for duration_index, duration in enumerate(T.DURATIONS):
            s0 = 0.001 * (duration_index + 1) * (rep + 1)
            lambda0 = 2.0 + 0.1 * duration_index + 0.05 * rep
            lapse = 0.05 + 0.005 * duration_index + 0.002 * rep
            sensory_sd5 = float(T.evidence_sd(5.0, s0, lambda0))
            rows.append(
                {
                    "recovery_scope": "shared_joint",
                    "rep": rep,
                    "duration_ms": duration,
                    "gen_sigma_c": sigma_c,
                    "rec_sigma_c": sigma_c * 1.01,
                    "gen_pow": pow_,
                    "rec_pow": pow_ * 0.995,
                    "gen_s0": s0,
                    "rec_s0": s0 * 1.02,
                    "gen_lambda0": lambda0,
                    "rec_lambda0": lambda0 + 0.01,
                    "gen_lapse": lapse,
                    "rec_lapse": lapse * 0.99,
                    "gen_n_width": n_width,
                    "rec_n_width": n_width * 1.005,
                    "gen_sigma3": sigma3,
                    "rec_sigma3": sigma3 * 1.01,
                    "gen_sensory_sd5": sensory_sd5,
                    "rec_sensory_sd5": sensory_sd5 * 1.01,
                    "gen_logL_fit_grid": -100.0 - rep,
                    "rec_logL_fit_grid": -99.0 - rep,
                }
            )
    return pd.DataFrame(rows)


def test_joint_recovery_requires_one_landscape_per_dataset() -> None:
    frame = _joint_recovery_frame()
    recovery._validate_joint_output(frame)

    invalid = frame.copy()
    invalid.loc[0, "gen_pow"] += 0.1
    with pytest.raises(ValueError, match="does not share pow"):
        recovery._validate_joint_output(invalid)


def test_recovery_summary_does_not_duplicate_shared_parameters() -> None:
    summary = recovery.summarize(_joint_recovery_frame()).set_index("quantity")

    assert summary.loc["sigma_c", "n"] == 2
    assert summary.loc["pow", "n"] == 2
    assert summary.loc["n_width", "n"] == 2
    assert summary.loc["s0", "n"] == 10
    assert summary.loc["lambda0", "n"] == 10
    assert summary.loc["sensory_sd5", "n"] == 10
    assert summary.loc["n_width", "reported_R2"] == summary.loc["n_width", "R2_identity"]
    assert summary.loc["s0", "reported_R2"] == summary.loc["s0", "R2_within_duration"]


def test_failed_recovery_detection_is_per_joint_dataset() -> None:
    frame = _joint_recovery_frame()
    assert recovery._failed_recovery_reps(frame) == []

    frame.loc[frame["rep"] == 1, "rec_logL_fit_grid"] = -103.0
    assert recovery._failed_recovery_reps(frame) == [1]
