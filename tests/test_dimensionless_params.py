"""Physics-anchored tests for src/dimensionless_params.py (numpy only, CI-safe)."""

import math

import pytest

from dimensionless_params import (
    ion_skin_depth,
    compute_dimensionless,
    magnetic_reynolds,
)


# ---------------------------------------------------------------------------
# ion_skin_depth: d_i = sqrt(|rqm_i| / n_e)
# ---------------------------------------------------------------------------

def test_ion_skin_depth_reference_density():
    # At n_e = 1 (default), d_i = sqrt(|rqm_i|).
    assert ion_skin_depth(4.0) == pytest.approx(2.0)
    assert ion_skin_depth(2.0) == pytest.approx(math.sqrt(2.0))


def test_ion_skin_depth_scales_with_density():
    # d_i = sqrt(|rqm_i| / n_e): quadrupling n_e halves d_i.
    assert ion_skin_depth(4.0, n_e=4.0) == pytest.approx(1.0)


def test_ion_skin_depth_nonpositive_density_is_nan():
    assert math.isnan(ion_skin_depth(4.0, n_e=0.0))
    assert math.isnan(ion_skin_depth(4.0, n_e=-1.0))


# ---------------------------------------------------------------------------
# compute_dimensionless: hand-computed reference case
#   n_e=1, T_e=T_i=0.5, B2=2, v_shock=1, |rqm_i|=2, gamma=5/3
# ---------------------------------------------------------------------------

@pytest.fixture
def ref_prim():
    return {"n_e": 1.0, "T_e": 0.5, "T_i": 0.5, "B2": 2.0}


def test_beta_thermal_over_magnetic(ref_prim):
    # P_th = n_e (T_e+T_i) = 1.0 ; P_B = B2/2 = 1.0 ; beta = 1.0
    p = compute_dimensionless(ref_prim, v_shock=1.0, abs_rqm_i=2.0)
    assert p["beta"] == pytest.approx(1.0)


def test_sigma_and_alfven(ref_prim):
    # sigma = B2/(|rqm_i| n_e) = 2/2 = 1 ; v_A = sqrt(sigma) = 1 ; M_A = v_sh/v_A = 1
    p = compute_dimensionless(ref_prim, v_shock=1.0, abs_rqm_i=2.0)
    assert p["sigma"] == pytest.approx(1.0)
    assert p["v_A"] == pytest.approx(1.0)
    assert p["M_A"] == pytest.approx(1.0)


def test_sound_speed_and_sonic_mach(ref_prim):
    # cs^2 = gamma (T_e+T_i)/|rqm_i| = (5/3)(1)/2 = 5/6
    p = compute_dimensionless(ref_prim, v_shock=1.0, abs_rqm_i=2.0)
    assert p["c_s"] == pytest.approx(math.sqrt(5.0 / 6.0))
    assert p["M_s"] == pytest.approx(1.0 / math.sqrt(5.0 / 6.0))


def test_temperature_ratio_and_skin_depth(ref_prim):
    p = compute_dimensionless(ref_prim, v_shock=1.0, abs_rqm_i=2.0)
    assert p["T_e_Ti"] == pytest.approx(1.0)
    assert p["d_i"] == pytest.approx(math.sqrt(2.0))   # sqrt(|rqm_i|/n_e)


def test_alfven_scales_with_field(ref_prim):
    # Doubling B^2 doubles sigma, so v_A scales by sqrt(2) and M_A by 1/sqrt(2).
    base = compute_dimensionless(ref_prim, v_shock=1.0, abs_rqm_i=2.0)
    hi = compute_dimensionless({**ref_prim, "B2": 4.0}, v_shock=1.0, abs_rqm_i=2.0)
    assert hi["v_A"] == pytest.approx(base["v_A"] * math.sqrt(2.0))
    assert hi["M_A"] == pytest.approx(base["M_A"] / math.sqrt(2.0))


def test_gamma_scales_sound_speed(ref_prim):
    # c_s ~ sqrt(gamma): doubling gamma scales c_s by sqrt(2).
    p1 = compute_dimensionless(ref_prim, v_shock=1.0, abs_rqm_i=2.0, gamma=5.0 / 3.0)
    p2 = compute_dimensionless(ref_prim, v_shock=1.0, abs_rqm_i=2.0, gamma=10.0 / 3.0)
    assert p2["c_s"] == pytest.approx(p1["c_s"] * math.sqrt(2.0))


def test_zero_temperature_gives_nan_ratios():
    # T_i = 0 -> T_e/T_i nan ; T_e+T_i = 0 -> c_s = 0 -> M_s nan
    p = compute_dimensionless({"n_e": 1.0, "T_e": 0.0, "T_i": 0.0, "B2": 2.0},
                              v_shock=1.0, abs_rqm_i=2.0)
    assert p["c_s"] == pytest.approx(0.0)
    assert math.isnan(p["M_s"])
    assert math.isnan(p["T_e_Ti"])


# ---------------------------------------------------------------------------
# magnetic_reynolds: invalid-input guard returns nan BEFORE any plasmapy import,
# so this is safe to test without the heavy stack installed.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("T_e, n_e", [(-1.0, 1.0), (0.0, 1.0), (1.0, 0.0), (math.nan, 1.0)])
def test_magnetic_reynolds_invalid_inputs_are_nan(T_e, n_e):
    out = magnetic_reynolds(T_e, n_e, v_shock=0.04, L_sim=1000.0,
                            norm_density=None, d_e=None, Z_i=13)
    assert math.isnan(out)
