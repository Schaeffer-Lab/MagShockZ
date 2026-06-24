"""Tests for rankine_hugoniot.py — MHD jump-condition baseline."""

import importlib.util
import os

import numpy as np
import pytest

_PATH = os.path.join(os.path.dirname(__file__), "..", "src", "rankine_hugoniot.py")
_spec = importlib.util.spec_from_file_location("rankine_hugoniot", _PATH)
rh = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rh)

GAMMA = 5.0 / 3.0


# ---------------------------------------------------------------------------
# shock_normal_angle / classification
# ---------------------------------------------------------------------------

def test_theta_bn_perpendicular():
    # All field transverse to x1 -> perpendicular shock (pi/2).
    theta = rh.shock_normal_angle(0.0, 3.0, 4.0)
    assert theta == pytest.approx(np.pi / 2.0)
    assert rh.is_quasi_perpendicular(theta)


def test_theta_bn_parallel():
    # Field purely along the normal -> parallel shock (0).
    theta = rh.shock_normal_angle(2.0, 0.0, 0.0)
    assert theta == pytest.approx(0.0)
    assert not rh.is_quasi_perpendicular(theta)


def test_theta_bn_array_inputs_are_mean_reduced():
    b1 = np.zeros(10)
    b2 = np.ones(10)
    b3 = np.ones(10)
    theta = rh.shock_normal_angle(b1, b2, b3)
    assert theta == pytest.approx(np.pi / 2.0)


# ---------------------------------------------------------------------------
# perp_compression_ratio — hydrodynamic limit
# ---------------------------------------------------------------------------

def test_hydro_strong_shock_compression_is_four():
    # M_A -> inf (unmagnetised), M_s -> inf (strong) => r -> (g+1)/(g-1) = 4.
    r = rh.perp_compression_ratio(mach_s=1e6, mach_a=np.inf, gamma=GAMMA)
    assert r == pytest.approx(4.0, abs=1e-3)


def test_hydro_compression_matches_analytic_RH():
    # Hydrodynamic compression ratio: r = (g+1) M^2 / ((g-1) M^2 + 2).
    for M in (2.0, 3.0, 5.0, 10.0):
        expected = (GAMMA + 1.0) * M**2 / ((GAMMA - 1.0) * M**2 + 2.0)
        r = rh.perp_compression_ratio(mach_s=M, mach_a=np.inf, gamma=GAMMA)
        assert r == pytest.approx(expected, rel=1e-6)


def test_subcritical_flow_has_no_shock():
    # Sonic Mach < 1 and no magnetic field => no compressive solution.
    assert np.isnan(rh.perp_compression_ratio(mach_s=0.5, mach_a=np.inf, gamma=GAMMA))


def test_compression_bounded_by_strong_shock_limit():
    r = rh.perp_compression_ratio(mach_s=50.0, mach_a=50.0, gamma=GAMMA)
    assert 1.0 < r <= (GAMMA + 1.0) / (GAMMA - 1.0) + 1e-9


def test_magnetic_field_reduces_compression():
    # A strong field (low M_A) stiffens the plasma => weaker compression than
    # the unmagnetised case at the same sonic Mach number.
    r_unmag = rh.perp_compression_ratio(mach_s=5.0, mach_a=np.inf, gamma=GAMMA)
    r_mag = rh.perp_compression_ratio(mach_s=5.0, mach_a=2.0, gamma=GAMMA)
    assert r_mag < r_unmag


# ---------------------------------------------------------------------------
# solve_jump — downstream state & self-consistency
# ---------------------------------------------------------------------------

def test_solve_jump_perpendicular_self_consistent():
    # Construct an upstream state with known Mach numbers, then check the
    # returned ratios obey the perpendicular jump relations.
    # Super-fast-magnetosonic inflow (M_s ~ 5.5, M_A ~ 3.2) so a compressive
    # perpendicular shock exists.
    n_e1, T_e1, T_i1 = 1.0, 1e-3, 1e-3
    abs_rqm_i = 40.0
    B2_1 = 0.01
    v_inflow = 0.05

    jump = rh.solve_jump(n_e1, T_e1, T_i1, B2_1, abs_rqm_i, v_inflow, gamma=GAMMA)

    c_s = np.sqrt(GAMMA * (T_e1 + T_i1) / abs_rqm_i)
    v_A = np.sqrt(B2_1 / (abs_rqm_i * n_e1))
    assert jump["mach_s"] == pytest.approx(v_inflow / c_s)
    assert jump["mach_a"] == pytest.approx(v_inflow / v_A)

    # Perpendicular shock: B compresses with density.
    assert jump["B_ratio"] == pytest.approx(jump["r"])
    # Downstream is compressed and heated.
    assert jump["r"] > 1.0
    assert jump["p_ratio"] > 1.0
    assert jump["T_factor"] == pytest.approx(jump["p_ratio"] / jump["r"])
    assert jump["T_adiabatic"] == pytest.approx((T_e1 + T_i1) * jump["T_factor"])


def test_solve_jump_subfast_inflow_has_no_shock():
    # Strong field + slow inflow => sub-Alfvenic (no fast shock) => nan.
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


# ---------------------------------------------------------------------------
# anomalous_heating
# ---------------------------------------------------------------------------

def test_anomalous_heating_split():
    # Upstream T=1, adiabatic factor 3 -> predicted downstream 3.
    # Measured 5 -> total heating 4, adiabatic 2, anomalous 2 (50%).
    out = rh.anomalous_heating(T_measured_dn=5.0, T_upstream=1.0, T_factor=3.0)
    assert out["adiabatic"] == pytest.approx(3.0)
    assert out["anomalous"] == pytest.approx(2.0)
    assert out["total_heating"] == pytest.approx(4.0)
    assert out["anomalous_frac"] == pytest.approx(0.5)


def test_anomalous_heating_pure_adiabatic_is_zero_anomaly():
    out = rh.anomalous_heating(T_measured_dn=3.0, T_upstream=1.0, T_factor=3.0)
    assert out["anomalous"] == pytest.approx(0.0)
    assert out["anomalous_frac"] == pytest.approx(0.0)
