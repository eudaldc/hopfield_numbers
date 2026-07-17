from __future__ import annotations

import numpy as np

from driven_attractor.driven import fitting as F
from driven_attractor.driven import model as M


def test_constant_well_model_fixes_equal_linearly_spaced_wells() -> None:
    defs = F.param_defs("M2_tether_const")
    assert [d.name for d in defs] == ["sigma_c", "rho", "tau_x", "beta"]

    vec = np.log([0.08, 15.0, 0.25, 4.0])
    theta = F.vec_to_theta("M2_tether_const", vec, defs, gamma=0.0)
    sig = M.sigma_wells(theta["sigma_c"], theta["p"])

    assert theta["p"] == 0.0
    assert np.all(sig == sig[0])
    assert np.all(np.diff(M.WELLS) == 1)


def test_constant_well_model_matches_tether_at_p_zero() -> None:
    theta = {
        "sigma_c": 0.08,
        "p": 1.7,
        "rho": 15.0,
        "gamma": 0.0,
        "tau_x": 0.25,
        "beta": 4.0,
    }
    const = M.evaluate_model("M2_tether_const", theta, m_paths=8, dt=0.02)
    power = M.evaluate_model("M2_tether", dict(theta, p=0.0),
                             m_paths=8, dt=0.02)

    for field in ("p_wells", "p_guess", "e_trelax", "e_entropy",
                  "x_dec_mean"):
        assert np.array_equal(getattr(const, field), getattr(power, field),
                              equal_nan=True)
    assert const.frac_capped == power.frac_capped
