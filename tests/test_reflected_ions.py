"""Tests for reflected_ions.py and the masked moment in moments.py."""

import numpy as np
import pytest

from conftest import make_phase_space
import moments
import reflected_ions as ri


def _gauss(p, mu, sigma):
    return np.exp(-0.5 * ((p - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def bimodal_phase_space(p_arr, x_arr, a_in, mu_in, a_refl, mu_refl, sigma):
    """Two narrow Gaussians (incoming + reflected) summed, uniform over x."""
    f_p = a_in * _gauss(p_arr, mu_in, sigma) + a_refl * _gauss(p_arr, mu_refl, sigma)
    values = np.outer(f_p, np.ones(len(x_arr)))
    return make_phase_space(p_arr, x_arr, values)


P = np.linspace(-0.3, 0.3, 4001)
X = np.linspace(0.0, 10.0, 8)


# ---------------------------------------------------------------------------
# masked moment (moments.py)
# ---------------------------------------------------------------------------

def test_masked_zeroth_moments_sum_to_full():
    ps = bimodal_phase_space(P, X, a_in=2.0, mu_in=-0.05, a_refl=0.5, mu_refl=0.05, sigma=0.005)
    full = np.abs(moments.moment(ps, order=0, axis="p1"))
    left = P < 0.0
    right = ~left
    n_left = np.abs(moments.moment(ps, order=0, axis="p1", p_mask=left))
    n_right = np.abs(moments.moment(ps, order=0, axis="p1", p_mask=right))
    np.testing.assert_allclose(n_left + n_right, full, rtol=1e-6)


def test_masked_moment_wrong_shape_raises():
    ps = bimodal_phase_space(P, X, 1.0, -0.05, 1.0, 0.05, 0.005)
    with pytest.raises(ValueError):
        moments.moment(ps, order=0, axis="p1", p_mask=np.ones(10, dtype=bool))


# ---------------------------------------------------------------------------
# population split / reflected fraction
# ---------------------------------------------------------------------------

def test_reflected_fraction_matches_amplitudes():
    # v_shock=0, incoming_sign=-1 => incoming are p<0, reflected p>0.
    a_in, a_refl = 4.0, 1.0
    ps = bimodal_phase_space(P, X, a_in, -0.06, a_refl, 0.06, sigma=0.004)
    frac = ri.reflected_fraction(ps, v_shock=0.0, incoming_sign=-1)
    expected = a_refl / (a_in + a_refl)
    np.testing.assert_allclose(frac, expected, rtol=1e-4)


def test_incoming_and_reflected_densities_recover_amplitudes():
    a_in, a_refl = 3.0, 2.0
    ps = bimodal_phase_space(P, X, a_in, -0.06, a_refl, 0.06, sigma=0.004)
    n_inc, n_refl = ri.number_densities(ps, v_shock=0.0, incoming_sign=-1)
    np.testing.assert_allclose(n_inc, a_in, rtol=1e-4)
    np.testing.assert_allclose(n_refl, a_refl, rtol=1e-4)


def test_swapping_incoming_sign_swaps_populations():
    ps = bimodal_phase_space(P, X, 4.0, -0.06, 1.0, 0.06, sigma=0.004)
    frac_neg = ri.reflected_fraction(ps, v_shock=0.0, incoming_sign=-1)
    frac_pos = ri.reflected_fraction(ps, v_shock=0.0, incoming_sign=+1)
    np.testing.assert_allclose(frac_neg + frac_pos, 1.0, rtol=1e-4)


# ---------------------------------------------------------------------------
# reflected energy density
# ---------------------------------------------------------------------------

def test_reflected_energy_density_matches_analytic():
    # Reflected gaussian: amplitude a_refl, mean mu, var sigma^2.
    # u = 0.5 |rqm| * a_refl * (sigma^2 + (mu - v_shock)^2)
    a_refl, mu, sigma, rqm = 1.5, 0.06, 0.004, 40.0
    ps = bimodal_phase_space(P, X, a_in=3.0, mu_in=-0.06,
                            a_refl=a_refl, mu_refl=mu, sigma=sigma)
    u = ri.reflected_energy_density(ps, rqm=rqm, v_shock=0.0, incoming_sign=-1)
    expected = 0.5 * abs(rqm) * a_refl * (sigma**2 + mu**2)
    np.testing.assert_allclose(u, expected, rtol=1e-3)


def test_infer_incoming_sign():
    # Upstream bulk slower than shock => incoming shock-frame velocity negative.
    assert ri.infer_incoming_sign(bulk_velocity_upstream=0.01, v_shock=0.05) == -1
    assert ri.infer_incoming_sign(bulk_velocity_upstream=0.09, v_shock=0.05) == +1
