from __future__ import annotations

import numpy as np

from scripts import trial_level_rt_ladder as R


def synthetic_prepared() -> R.Prepared:
    participants = np.array(["a", "b", "c", "d"])
    count = np.full((4, len(R.DURS), len(R.NVALS)), 3.0)
    et = np.vstack(
        [np.linspace(2.0 + i, 100.0 + i, len(R.NVALS)) for i in range(len(R.DURS))]
    )
    eh = 0.3 + 0.012 * et + np.tile(
        np.linspace(-0.08, 0.08, len(R.NVALS)), (len(R.DURS), 1)
    )
    total_count = np.sum(count, axis=0)
    eh_resid, diagnostics = R._weighted_entropy_residuals(et, eh, total_count)
    location = R.T.T0 + 0.011 * et
    y = np.log(location)
    sum_y = count * y[None, :, :]
    sum_y2 = count * y[None, :, :] ** 2 + count * 0.2**2
    return R.Prepared(
        participant_ids=participants,
        count_by_participant=count,
        sum_by_participant=sum_y,
        sum2_by_participant=sum_y2,
        et=et,
        eh=eh,
        eh_resid=eh_resid,
        entropy_diagnostics=diagnostics,
    )


def test_requested_parameter_counts():
    expected = {"M1": 6, "M2": 7, "M3": 7, "M4": 9, "M5": 15}
    for model_id, k in expected.items():
        assert len(R.parameter_names(model_id)) + len(R.DURS) == k


def test_nesting_identities():
    a_tau = np.log(0.012)
    a_kappa = np.log(0.25)
    tau_1, kappa_1 = R.decode_link("M1", np.array([a_tau]))
    tau_3, kappa_3 = R.decode_link("M3", np.array([a_tau, 0.0]))
    np.testing.assert_allclose(tau_1, tau_3)
    np.testing.assert_allclose(kappa_1, kappa_3)

    tau_2, kappa_2 = R.decode_link("M2", np.array([a_tau, a_kappa]))
    tau_4, kappa_4 = R.decode_link(
        "M4", np.array([a_tau, 0.0, a_kappa, 0.0])
    )
    np.testing.assert_allclose(tau_2, tau_4)
    np.testing.assert_allclose(kappa_2, kappa_4)

    vector_5 = np.r_[np.log(tau_4), np.log(kappa_4)]
    tau_5, kappa_5 = R.decode_link("M5", vector_5)
    np.testing.assert_allclose(tau_4, tau_5)
    np.testing.assert_allclose(kappa_4, kappa_5)


def test_residual_entropy_is_weighted_orthogonal():
    prepared = synthetic_prepared()
    weights = np.sum(prepared.count_by_participant, axis=0)
    for di in range(len(R.DURS)):
        residual = prepared.eh_resid[di]
        steps = prepared.et[di]
        assert abs(np.average(residual, weights=weights[di])) < 1e-12
        centered_steps = steps - np.average(steps, weights=weights[di])
        assert abs(np.average(residual * centered_steps, weights=weights[di])) < 1e-12


def test_participant_folds_do_not_overlap():
    prepared = synthetic_prepared()
    folds = R.participant_folds(prepared, n_folds=2, seed=20260717)
    assert set(folds) == {0, 1}
    first = set(np.flatnonzero(folds == 0))
    second = set(np.flatnonzero(folds == 1))
    assert first.isdisjoint(second)
    assert first | second == set(range(len(prepared.participant_ids)))


def test_profile_fit_recovers_convergence_curve():
    prepared = synthetic_prepared()
    stats = R.aggregate(prepared)
    fit = R.fit_model(prepared, stats, "M3", starts=2)
    assert fit.converged
    assert fit.logl > -1e6
    assert np.all(fit.tau > 0)
    assert np.max(np.abs(fit.kappa)) == 0
    location, _, _ = R.model_location(prepared, "M3", fit.vector)
    assert np.all(location > R.T.T0)
