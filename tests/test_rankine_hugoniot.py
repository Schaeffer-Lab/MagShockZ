"""Tests for rankine_hugoniot.py — oblique MHD jump-condition baseline.

The oblique compression ratio (solved via the energy-flux residual in the module)
is cross-checked here against a *fully independent* solve of the five oblique MHD
jump conditions (mass, normal & tangential momentum, induction, energy) with
scipy.optimize.fsolve, plus the perpendicular / hydrodynamic / parallel limits.
"""

import importlib.util
import os

import numpy as np
import pytest
import scipy.optimize

_PATH = os.path.join(os.path.dirname(__file__), "..", "src", "rankine_hugoniot.py")
_spec = importlib.util.spec_from_file_location("rankine_hugoniot", _PATH)
rh = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rh)

GAMMA = 5.0 / 3.0


# ---------------------------------------------------------------------------
# Independent reference: solve the raw 5 oblique jump conditions directly.
# Units: rho1 = 1, u1 = 1 (normal inflow); scaled fields beta = B / sqrt(4 pi),
# so Bt^2/8pi -> beta_t^2/2, Bt^2/4pi -> beta_t^2, Bn Bt/4pi -> beta_n beta_t.
# ---------------------------------------------------------------------------

def _reference_compression(mach_s, mach_a, theta, gamma):
    inv_Ms2 = 1.0 / mach_s**2
    inv_Ma2 = 0.0 if not np.isfinite(mach_a) else 1.0 / mach_a**2
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    P1 = inv_Ms2 / gamma                    # c_s1^2 = 1/M_s^2, P1 = rho1 c_s1^2/gamma
    bn = cos_t * np.sqrt(inv_Ma2)           # beta_n  (v_A1^2 = 1/M_A^2)
    bt1 = sin_t * np.sqrt(inv_Ma2)          # beta_t1

    # Upstream flux constants.
    mass1 = 1.0                                             # rho1 u1
    mom_n1 = 1.0 + P1 + 0.5 * bt1**2                        # rho u_x^2 + P + Bt^2/8pi
    mom_t1 = -bn * bt1                                      # rho u_x u_y - bn bt
    induct1 = bt1                                           # u_x bt - u_y bn
    energy1 = 0.5 + gamma / (gamma - 1.0) * P1 + bt1**2     # incl. Poynting

    def eqs(v):
        rho2, u2x, u2y, P2, bt2 = v
        m = rho2 * u2x - mass1
        mn = rho2 * u2x**2 + P2 + 0.5 * bt2**2 - mom_n1
        mt = rho2 * u2x * u2y - bn * bt2 - mom_t1
        ind = u2x * bt2 - u2y * bn - induct1
        u2sq = u2x**2 + u2y**2
        en = u2x * (0.5 * rho2 * u2sq + gamma / (gamma - 1.0) * P2) \
            + (u2x * bt2**2 - u2y * bn * bt2) - energy1
        return [m, mn, mt, ind, en]

    guess = [3.0, 1.0 / 3.0, 0.0, P1 + 1.0, 3.0 * bt1]
    sol = scipy.optimize.fsolve(eqs, guess, full_output=True)
    x, info, ier, _ = sol
    assert ier == 1, "reference jump-condition solve did not converge"
    return x[0]                              # rho2 / rho1 = compression


# ---------------------------------------------------------------------------
# gamma helper
# ---------------------------------------------------------------------------

def test_gamma_from_dof():
    assert rh.gamma_from_dof(3) == pytest.approx(5.0 / 3.0)
    assert rh.gamma_from_dof(2) == pytest.approx(2.0)
    assert rh.gamma_from_dof(1) == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# shock_normal_angle / classification
# ---------------------------------------------------------------------------

def test_theta_bn_perpendicular():
    theta = rh.shock_normal_angle(0.0, 3.0, 4.0)
    assert theta == pytest.approx(np.pi / 2.0)
    assert rh.is_quasi_perpendicular(theta)


def test_theta_bn_parallel():
    theta = rh.shock_normal_angle(2.0, 0.0, 0.0)
    assert theta == pytest.approx(0.0)
    assert not rh.is_quasi_perpendicular(theta)


def test_theta_bn_array_inputs_are_mean_reduced():
    theta = rh.shock_normal_angle(np.zeros(10), np.ones(10), np.ones(10))
    assert theta == pytest.approx(np.pi / 2.0)


# ---------------------------------------------------------------------------
# compression_ratio — hydrodynamic / perpendicular limits
# ---------------------------------------------------------------------------

def test_hydro_strong_shock_compression_is_four():
    r = rh.compression_ratio(mach_s=1e6, mach_a=np.inf, gamma=GAMMA)
    assert r == pytest.approx(4.0, abs=1e-3)


def test_hydro_compression_matches_analytic_RH():
    # Hydrodynamic compression: r = (g+1) M^2 / ((g-1) M^2 + 2).  Independent of
    # theta because an unmagnetised shock has no field to feel the angle.
    for M in (2.0, 3.0, 5.0, 10.0):
        expected = (GAMMA + 1.0) * M**2 / ((GAMMA - 1.0) * M**2 + 2.0)
        for theta in (np.pi / 2.0, np.pi / 3.0, 0.0):
            r = rh.compression_ratio(mach_s=M, mach_a=np.inf, theta=theta, gamma=GAMMA)
            assert r == pytest.approx(expected, rel=1e-6)


def test_subcritical_flow_has_no_shock():
    assert np.isnan(rh.compression_ratio(mach_s=0.5, mach_a=np.inf, gamma=GAMMA))


def test_compression_bounded_by_strong_shock_limit():
    r = rh.compression_ratio(mach_s=50.0, mach_a=50.0, gamma=GAMMA)
    assert 1.0 < r <= (GAMMA + 1.0) / (GAMMA - 1.0) + 1e-9


def test_magnetic_field_reduces_compression():
    r_unmag = rh.compression_ratio(mach_s=5.0, mach_a=np.inf, gamma=GAMMA)
    r_mag = rh.compression_ratio(mach_s=5.0, mach_a=2.0, gamma=GAMMA)
    assert r_mag < r_unmag


def test_perp_wrapper_matches_general_at_90deg():
    r_wrap = rh.perp_compression_ratio(mach_s=5.0, mach_a=3.0, gamma=GAMMA)
    r_gen = rh.compression_ratio(mach_s=5.0, mach_a=3.0, theta=np.pi / 2.0, gamma=GAMMA)
    assert r_wrap == pytest.approx(r_gen, rel=1e-12)


# ---------------------------------------------------------------------------
# Oblique compression: cross-check vs the independent raw-jump-condition solve
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mach_s, mach_a, theta_deg", [
    (5.0, 4.0, 90.0),
    (5.0, 4.0, 80.0),
    (5.0, 4.0, 60.0),
    (6.0, 5.0, 75.0),
    (8.0, 6.0, 85.0),
    (4.0, 10.0, 70.0),
])
def test_oblique_compression_matches_raw_jump_conditions(mach_s, mach_a, theta_deg):
    theta = np.radians(theta_deg)
    r = rh.compression_ratio(mach_s, mach_a, theta=theta, gamma=GAMMA)
    r_ref = _reference_compression(mach_s, mach_a, theta, GAMMA)
    assert r == pytest.approx(r_ref, rel=1e-6)


def test_oblique_compression_matches_raw_jump_conditions_gamma2():
    # gamma = 2 (the f = 2 DOF case the user wants to verify).
    theta = np.radians(75.0)
    r = rh.compression_ratio(5.0, 4.0, theta=theta, gamma=2.0)
    r_ref = _reference_compression(5.0, 4.0, theta, 2.0)
    assert r == pytest.approx(r_ref, rel=1e-6)
    assert 1.0 < r <= (2.0 + 1.0) / (2.0 - 1.0) + 1e-9


# ---------------------------------------------------------------------------
# tangential_field_ratio — the density-vs-field reconciliation
# ---------------------------------------------------------------------------

def test_Bt_ratio_equals_r_for_perpendicular():
    r = rh.compression_ratio(5.0, 3.0, theta=np.pi / 2.0, gamma=GAMMA)
    assert rh.tangential_field_ratio(r, mach_a=3.0, theta=np.pi / 2.0) == pytest.approx(r)


def test_Bt_ratio_differs_from_r_when_oblique():
    theta = np.radians(60.0)
    r = rh.compression_ratio(5.0, 4.0, theta=theta, gamma=GAMMA)
    b = rh.tangential_field_ratio(r, mach_a=4.0, theta=theta)
    assert b != pytest.approx(r)          # only equal at perpendicular
    assert b > 1.0


def test_Bt_ratio_consistent_with_raw_jump_solve():
    # The module Bt2/Bt1 must match the field compression from the raw solve.
    theta = np.radians(70.0)
    mach_s, mach_a = 6.0, 5.0
    r = rh.compression_ratio(mach_s, mach_a, theta=theta, gamma=GAMMA)
    b_module = rh.tangential_field_ratio(r, mach_a, theta)
    # Reconstruct bt2/bt1 from the reference downstream solve.
    inv_Ma2 = 1.0 / mach_a**2
    bn = np.cos(theta) * np.sqrt(inv_Ma2)
    bt1 = np.sin(theta) * np.sqrt(inv_Ma2)
    u2x = 1.0 / r
    # induction + tangential momentum give bt2/bt1 = r(M_A^2-cos^2)/(M_A^2-r cos^2);
    # check the module value satisfies induction with u2y from tangential momentum.
    bt2 = b_module * bt1
    u2y = (bn * (bt2 - bt1))               # rho2 u2x u2y = bn(bt2-bt1), rho2 u2x = 1
    assert u2x * bt2 - u2y * bn == pytest.approx(bt1, rel=1e-6)


# ---------------------------------------------------------------------------
# solve_jump — downstream state & self-consistency
# ---------------------------------------------------------------------------

def test_solve_jump_perpendicular_self_consistent():
    n_e1, T_e1, T_i1 = 1.0, 1e-3, 1e-3
    abs_rqm_i, B2_1, v_inflow = 40.0, 0.01, 0.05

    jump = rh.solve_jump(n_e1, T_e1, T_i1, B2_1, abs_rqm_i, v_inflow, gamma=GAMMA)

    c_s = np.sqrt(GAMMA * (T_e1 + T_i1) / abs_rqm_i)
    v_A = np.sqrt(B2_1 / (abs_rqm_i * n_e1))
    assert jump["mach_s"] == pytest.approx(v_inflow / c_s)
    assert jump["mach_a"] == pytest.approx(v_inflow / v_A)

    # Perpendicular default: B compresses with density.
    assert jump["B_ratio"] == pytest.approx(jump["r"])
    assert jump["r"] > 1.0
    assert jump["p_ratio"] > 1.0
    assert jump["T_factor"] == pytest.approx(jump["p_ratio"] / jump["r"])
    assert jump["T_adiabatic"] == pytest.approx((T_e1 + T_i1) * jump["T_factor"])
    assert jump["theta"] == pytest.approx(np.pi / 2.0)


def test_solve_jump_oblique_Bratio_exceeds_r():
    # At an oblique angle the tangential field is amplified MORE than the density
    # compresses: Bt2/Bt1 = r (M_A^2 - cos^2)/(M_A^2 - r cos^2) > r for r > 1.
    jump = rh.solve_jump(1.0, 1e-3, 1e-3, 0.01, 40.0, 0.05,
                         theta=np.radians(60.0), gamma=GAMMA)
    assert jump["r"] > 1.0
    assert jump["B_ratio"] > jump["r"]


def test_solve_jump_subfast_inflow_has_no_shock():
    jump = rh.solve_jump(n_e1=1.0, T_e1=1e-3, T_i1=1e-3, B2_1=0.05,
                         abs_rqm_i=40.0, v_inflow=0.02, gamma=GAMMA)
    assert jump["mach_a"] < 1.0
    assert np.isnan(jump["r"])


def test_solve_jump_hydro_limit_when_unmagnetised():
    n_e1, T_e1, T_i1, abs_rqm_i = 1.0, 1e-3, 1e-3, 40.0
    v_inflow = 0.05
    jump = rh.solve_jump(n_e1, T_e1, T_i1, B2_1=0.0,
                         abs_rqm_i=abs_rqm_i, v_inflow=v_inflow, gamma=GAMMA)
    M = jump["mach_s"]
    expected = (GAMMA + 1.0) * M**2 / ((GAMMA - 1.0) * M**2 + 2.0)
    assert jump["r"] == pytest.approx(expected, rel=1e-6)


def test_solve_jump_gamma_changes_compression():
    # Same upstream, different gamma -> different compression (sweepable knob).
    args = dict(n_e1=1.0, T_e1=1e-3, T_i1=1e-3, B2_1=0.01, abs_rqm_i=40.0,
                v_inflow=0.05, theta=np.radians(85.0))
    r_53 = rh.solve_jump(**args, gamma=5.0 / 3.0)["r"]
    r_2 = rh.solve_jump(**args, gamma=2.0)["r"]
    assert r_53 != pytest.approx(r_2)
    # Strong-shock ceiling is lower for larger gamma.
    assert r_2 <= (2.0 + 1.0) / (2.0 - 1.0) + 1e-9


# ---------------------------------------------------------------------------
# anomalous_heating
# ---------------------------------------------------------------------------

def test_anomalous_heating_split():
    out = rh.anomalous_heating(T_measured_dn=5.0, T_upstream=1.0, T_factor=3.0)
    assert out["adiabatic"] == pytest.approx(3.0)
    assert out["anomalous"] == pytest.approx(2.0)
    assert out["total_heating"] == pytest.approx(4.0)
    assert out["anomalous_frac"] == pytest.approx(0.5)


def test_anomalous_heating_pure_adiabatic_is_zero_anomaly():
    out = rh.anomalous_heating(T_measured_dn=3.0, T_upstream=1.0, T_factor=3.0)
    assert out["anomalous"] == pytest.approx(0.0)
    assert out["anomalous_frac"] == pytest.approx(0.0)
