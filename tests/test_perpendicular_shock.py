"""Tests for src/perpendicular_shock.py.

Locks the perpendicular MHD shock to Fitzpatrick's equations and cross-checks it
against the independently-derived perpendicular path in rankine_hugoniot.py.
"""

import importlib.util
import os

import numpy as np
import pytest

_HERE = os.path.dirname(__file__)


def _load(name):
    path = os.path.join(_HERE, "..", "src", f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ps = _load("perpendicular_shock")
rh = _load("rankine_hugoniot")

GAMMA = 5.0 / 3.0


# ---------------------------------------------------------------------------
# plasma_beta
# ---------------------------------------------------------------------------

def test_plasma_beta_definition():
    # beta1 = (2/gamma) (M_A/M_s)^2.
    assert ps.plasma_beta(5.0, 4.0, GAMMA) == pytest.approx((2.0 / GAMMA) * (4.0 / 5.0) ** 2)


# ---------------------------------------------------------------------------
# shock_exists (Eq. 7.276 / 7.277)
# ---------------------------------------------------------------------------

def test_shock_exists_requires_super_fast_inflow():
    # 1/M_s^2 + 1/M_A^2 < 1  <=>  V1^2 > c_s^2 + v_A^2.
    assert ps.shock_exists(5.0, 4.0, GAMMA)
    assert not ps.shock_exists(0.9, np.inf, GAMMA)          # subsonic
    assert not ps.shock_exists(1.2, 1.2, GAMMA)             # sub-fast-magnetosonic


# ---------------------------------------------------------------------------
# compression_ratio
# ---------------------------------------------------------------------------

def test_hydro_limit_matches_gas_dynamic_shock():
    for M in (2.0, 3.0, 5.0, 10.0):
        expected = (GAMMA + 1.0) * M**2 / ((GAMMA - 1.0) * M**2 + 2.0)
        assert ps.compression_ratio(M, np.inf, GAMMA) == pytest.approx(expected, rel=1e-9)


def test_strong_shock_ceiling_is_four():
    assert ps.compression_ratio(1e6, np.inf, GAMMA) == pytest.approx(4.0, abs=1e-3)


def test_magnetic_field_reduces_compression():
    assert ps.compression_ratio(5.0, 2.0, GAMMA) < ps.compression_ratio(5.0, np.inf, GAMMA)


def test_no_shock_returns_nan():
    assert np.isnan(ps.compression_ratio(0.5, np.inf, GAMMA))


@pytest.mark.parametrize("mach_s, mach_a", [
    (4.0, 3.0), (5.0, 4.0), (8.0, 6.0), (4.13, 9.86), (4.97, 6.72),
])
def test_agrees_with_rankine_hugoniot_perp(mach_s, mach_a):
    # The two independent implementations of the perpendicular shock must agree.
    r_ps = ps.compression_ratio(mach_s, mach_a, GAMMA)
    r_rh = rh.compression_ratio(mach_s, mach_a, theta=np.pi / 2.0, gamma=GAMMA)
    assert r_ps == pytest.approx(r_rh, rel=1e-6)


def test_gamma_2_linear_branch():
    # gamma = 2 makes the quadratic's r^2 coefficient vanish (linear); still valid.
    r = ps.compression_ratio(5.0, 4.0, gamma=2.0)
    r_rh = rh.compression_ratio(5.0, 4.0, theta=np.pi / 2.0, gamma=2.0)
    assert r == pytest.approx(r_rh, rel=1e-6)
    assert 1.0 < r <= (2.0 + 1.0) / (2.0 - 1.0) + 1e-9


# ---------------------------------------------------------------------------
# pressure_ratio (Eq. 7.273)
# ---------------------------------------------------------------------------

def test_pressure_ratio_matches_eq_7273():
    mach_s, mach_a = 5.0, 4.0
    r = ps.compression_ratio(mach_s, mach_a, GAMMA)
    beta1 = ps.plasma_beta(mach_s, mach_a, GAMMA)
    expected = 1.0 + GAMMA * mach_s**2 * (1.0 - 1.0 / r) + (1.0 / beta1) * (1.0 - r**2)
    assert ps.pressure_ratio(r, mach_s, mach_a, GAMMA) == pytest.approx(expected)


def test_pressure_ratio_agrees_with_rankine_hugoniot():
    mach_s, mach_a = 6.0, 5.0
    r = ps.compression_ratio(mach_s, mach_a, GAMMA)
    # Compare the pressure-ratio formula directly to the RH perpendicular form.
    p_ps = ps.pressure_ratio(r, mach_s, mach_a, GAMMA)
    # RH perp p_ratio: 1 + g Ms^2 (1-1/r) - (g/2)(Ms^2/Ma^2)(r^2-1)
    p_rh = 1.0 + GAMMA * mach_s**2 * (1.0 - 1.0 / r) \
        - 0.5 * GAMMA * mach_s**2 / mach_a**2 * (r**2 - 1.0)
    assert p_ps == pytest.approx(p_rh, rel=1e-9)


# ---------------------------------------------------------------------------
# solve / solve_from_speeds
# ---------------------------------------------------------------------------

def test_solve_keys_and_T_ratio():
    out = ps.solve(5.0, 4.0, GAMMA)
    assert out["exists"]
    assert out["T_ratio"] == pytest.approx(out["p_ratio"] / out["r"])


def test_solve_from_speeds_is_ratio_invariant():
    # Only the ratios matter: scaling all speeds by a constant changes nothing.
    a = ps.solve_from_speeds(0.04, 0.01, 0.006, GAMMA)
    b = ps.solve_from_speeds(40.0, 10.0, 6.0, GAMMA)
    assert a["r"] == pytest.approx(b["r"], rel=1e-12)
    assert a["mach_s"] == pytest.approx(b["mach_s"])


def test_solve_from_speeds_unmagnetised():
    out = ps.solve_from_speeds(0.05, 0.01, 0.0, GAMMA)   # v_A = 0 -> hydro
    M = 0.05 / 0.01
    assert out["r"] == pytest.approx((GAMMA + 1.0) * M**2 / ((GAMMA - 1.0) * M**2 + 2.0))


# ---------------------------------------------------------------------------
# sound_speed / alfven_speed / solve_from_upstream
# ---------------------------------------------------------------------------

def test_sound_speed_general_two_temperature_form():
    # c_s^2 = (gamma_e n_e kTe + gamma_i n_ion kTi) / rho  (the ion-acoustic form
    # expressed in measured quantities).
    ne, Te, ni, Ti, rho = 2.0, 5.0, 1.0, 3.0, 7.0
    ge, gi = 1.0, 3.0
    expected = np.sqrt((ge * ne * Te + gi * ni * Ti) / rho)
    assert ps.sound_speed(ne, Te, ni, Ti, rho, gamma_e=ge, gamma_i=gi) == pytest.approx(expected)


def test_sound_speed_single_gamma_is_single_fluid_mhd():
    # gamma_e = gamma_i = gamma collapses to sqrt(gamma (P_e + P_i) / rho).
    ne, Te, ni, Ti, rho = 2.0, 5.0, 1.0, 3.0, 7.0
    P = ne * Te + ni * Ti
    mhd = np.sqrt(GAMMA * P / rho)
    assert ps.sound_speed(ne, Te, ni, Ti, rho, gamma_e=GAMMA, gamma_i=GAMMA) == pytest.approx(mhd)
    # default both indices -> same single-fluid value
    assert ps.sound_speed(ne, Te, ni, Ti, rho) == pytest.approx(mhd)


def test_alfven_speed_definition():
    B, rho = 3.0, 5.0
    assert ps.alfven_speed(B, rho) == pytest.approx(B / np.sqrt(4.0 * np.pi * rho))


def test_mass_flux_shock_speed_round_trips_known_frame():
    # Build a downstream consistent with a chosen v_sh and compression, then
    # recover v_sh: rho_up(v_up - v_sh) = rho_dn(v_dn - v_sh).
    v_sh, r, rho_up, v_up = 760.0, 3.0, 2.0, -1.0
    rho_dn = r * rho_up
    v_dn = v_sh + (rho_up * (v_up - v_sh)) / rho_dn   # mass continuity
    assert ps.mass_flux_shock_speed(rho_up, v_up, rho_dn, v_dn) == pytest.approx(v_sh)


def test_mass_flux_shock_speed_no_compression_is_nan():
    assert np.isnan(ps.mass_flux_shock_speed(2.0, 0.0, 2.0, 5.0))


def test_solve_from_upstream_matches_solve_from_speeds():
    # Build the speeds by hand, then check the field-level driver reproduces them.
    # Simple consistent numbers chosen so a compressive shock actually forms.
    ne, Te, ni, Ti, rho = 1.0, 1.0, 1.0, 1.0, 1.0
    B_perp, B_para = 1.0, 0.2
    v_shock, v_para = 10.0, 0.0
    c_s = ps.sound_speed(ne, Te, ni, Ti, rho)
    v_A = ps.alfven_speed(B_perp, rho)
    ref = ps.solve_from_speeds(abs(v_shock - v_para), c_s, v_A, GAMMA)

    out = ps.solve_from_upstream(
        ne=ne, Te=Te, n_ion=ni, Ti=Ti, B_perp=B_perp, B_para=B_para,
        rho=rho, v_shock=v_shock, v_para=v_para, gamma=GAMMA)
    assert out["r"] == pytest.approx(ref["r"])
    assert out["mach_s"] == pytest.approx(ref["mach_s"])
    assert out["mach_a"] == pytest.approx(ref["mach_a"])
    assert out["c_s"] == pytest.approx(c_s)
    assert out["v_A"] == pytest.approx(v_A)
    assert out["v_inflow"] == pytest.approx(abs(v_shock - v_para))


def test_solve_from_upstream_theta_bn():
    # Purely transverse field -> theta_bn = 90 deg; equal components -> 45 deg.
    kw = dict(ne=1.0, Te=1.0, n_ion=1.0, Ti=1.0, rho=1.0, v_shock=10.0, gamma=GAMMA)
    perp = ps.solve_from_upstream(B_perp=1.0, B_para=0.0, **kw)
    assert np.degrees(perp["theta_bn"]) == pytest.approx(90.0)
    obl = ps.solve_from_upstream(B_perp=1.0, B_para=1.0, **kw)
    assert np.degrees(obl["theta_bn"]) == pytest.approx(45.0)


# ---------------------------------------------------------------------------
# predict_downstream
# ---------------------------------------------------------------------------

def test_predict_downstream_applies_jump_ratios():
    jump = ps.solve(5.0, 4.0, GAMMA)
    r = jump["r"]
    pred = ps.predict_downstream(
        jump, rho1=2.0, B_perp1=3.0, p1=10.0, T1=4.0, v_inflow=100.0)
    assert pred["rho"] == pytest.approx(r * 2.0)          # rho2 = r rho1
    assert pred["B_perp"] == pytest.approx(r * 3.0)       # B2 = r B1 (perp)
    assert pred["p"] == pytest.approx(jump["p_ratio"] * 10.0)
    assert pred["T"] == pytest.approx(jump["T_ratio"] * 4.0)
    assert pred["v_inflow"] == pytest.approx(100.0 / r)   # v2 = V1 / r


def test_predict_downstream_only_requested_keys():
    jump = ps.solve(5.0, 4.0, GAMMA)
    pred = ps.predict_downstream(jump, rho1=1.0)
    assert "rho" in pred and "B_perp" not in pred and "p" not in pred


def test_predict_downstream_density_consistent_with_pressure_and_T():
    # p2 = (rho2/rho1) (T2/T1) p1  <=>  p_ratio = r * T_ratio.
    jump = ps.solve(6.0, 5.0, GAMMA)
    pred = ps.predict_downstream(jump, rho1=1.0, p1=1.0, T1=1.0)
    assert pred["p"] == pytest.approx(pred["rho"] * pred["T"])
