"""Tests for src/moments.py.

Each moment order is verified against a Gaussian distribution whose analytic
moments are known exactly:
    order 0  -> integral of f  = 1  (normalised Gaussian)
    order 1  -> mean           = mu
    order 2  -> variance       = sigma^2
"""

import numpy as np
import pytest

from conftest import FakeAxis, FakeH5Data, make_phase_space
import moments


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def gaussian_phase_space(mu: float, sigma: float, p_arr, x_arr) -> FakeH5Data:
    """1-D Gaussian in p, uniform over x, shape (p, x)."""
    f_p = np.exp(-0.5 * ((p_arr - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    values = np.outer(f_p, np.ones(len(x_arr)))
    return make_phase_space(p_arr, x_arr, values)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def p_arr():
    return np.linspace(-10.0, 10.0, 2001)


@pytest.fixture
def x_arr():
    return np.linspace(0.0, 50.0, 20)


# ---------------------------------------------------------------------------
# Order 0 — number density
# ---------------------------------------------------------------------------

class TestMomentOrder0:
    def test_unit_gaussian_integrates_to_one(self, p_arr, x_arr):
        ps = gaussian_phase_space(mu=0.0, sigma=1.0, p_arr=p_arr, x_arr=x_arr)
        result = moments.moment(ps, order=0, axis="p1")
        assert result.shape == (len(x_arr),)
        np.testing.assert_allclose(result, 1.0, atol=1e-4)

    def test_scaled_gaussian_density(self, p_arr, x_arr):
        """Density scales linearly with amplitude."""
        ps = gaussian_phase_space(mu=0.0, sigma=1.0, p_arr=p_arr, x_arr=x_arr)
        m0 = moments.moment(ps * 3.0, order=0, axis="p1")
        np.testing.assert_allclose(m0, 3.0, atol=1e-3)

    def test_zero_distribution_gives_zero_density(self, p_arr, x_arr):
        ps = make_phase_space(p_arr, x_arr, np.zeros((len(p_arr), len(x_arr))))
        result = moments.moment(ps, order=0, axis="p1")
        np.testing.assert_array_equal(result, 0.0)


# ---------------------------------------------------------------------------
# Order 1 — mean (bulk) velocity
# ---------------------------------------------------------------------------

class TestMomentOrder1:
    def test_zero_mean_gaussian(self, p_arr, x_arr):
        ps = gaussian_phase_space(mu=0.0, sigma=1.0, p_arr=p_arr, x_arr=x_arr)
        m1 = moments.moment(ps, order=1, axis="p1")
        np.testing.assert_allclose(m1, 0.0, atol=1e-4)

    def test_shifted_mean(self, p_arr, x_arr):
        ps = gaussian_phase_space(mu=2.5, sigma=1.0, p_arr=p_arr, x_arr=x_arr)
        m1 = moments.moment(ps, order=1, axis="p1")
        np.testing.assert_allclose(m1, 2.5, atol=1e-3)

    def test_negative_shift(self, p_arr, x_arr):
        ps = gaussian_phase_space(mu=-1.8, sigma=0.8, p_arr=p_arr, x_arr=x_arr)
        m1 = moments.moment(ps, order=1, axis="p1")
        np.testing.assert_allclose(m1, -1.8, atol=1e-3)

    def test_zero_distribution_returns_zero_not_nan(self, p_arr, x_arr):
        """Safe division: zero density should give zero, not NaN."""
        ps = make_phase_space(p_arr, x_arr, np.zeros((len(p_arr), len(x_arr))))
        m1 = moments.moment(ps, order=1, axis="p1")
        assert np.all(m1 == 0.0)


# ---------------------------------------------------------------------------
# Order 2 — velocity variance
# ---------------------------------------------------------------------------

class TestMomentOrder2:
    def test_unit_gaussian_variance(self, p_arr, x_arr):
        """sigma=1 Gaussian should give variance=1."""
        ps = gaussian_phase_space(mu=0.0, sigma=1.0, p_arr=p_arr, x_arr=x_arr)
        m2 = moments.moment(ps, order=2, axis="p1")
        np.testing.assert_allclose(m2, 1.0, atol=1e-3)

    def test_variance_independent_of_mean(self, p_arr, x_arr):
        """Variance is the *central* moment, so shifting the mean doesn't change it."""
        ps_centred = gaussian_phase_space(mu=0.0, sigma=2.0, p_arr=p_arr, x_arr=x_arr)
        ps_shifted = gaussian_phase_space(mu=3.0, sigma=2.0, p_arr=p_arr, x_arr=x_arr)
        m2_c = moments.moment(ps_centred, order=2, axis="p1")
        m2_s = moments.moment(ps_shifted, order=2, axis="p1")
        np.testing.assert_allclose(m2_c, m2_s, rtol=5e-3)

    def test_wider_gaussian_larger_variance(self, p_arr, x_arr):
        ps_narrow = gaussian_phase_space(mu=0.0, sigma=0.5, p_arr=p_arr, x_arr=x_arr)
        ps_wide = gaussian_phase_space(mu=0.0, sigma=2.0, p_arr=p_arr, x_arr=x_arr)
        m2_narrow = moments.moment(ps_narrow, order=2, axis="p1")
        m2_wide = moments.moment(ps_wide, order=2, axis="p1")
        assert np.all(m2_wide > m2_narrow)

    def test_zero_distribution_returns_zero_not_nan(self, p_arr, x_arr):
        ps = make_phase_space(p_arr, x_arr, np.zeros((len(p_arr), len(x_arr))))
        m2 = moments.moment(ps, order=2, axis="p1")
        assert np.all(m2 == 0.0)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestMomentErrors:
    def test_invalid_axis_raises(self, p_arr, x_arr):
        ps = gaussian_phase_space(mu=0.0, sigma=1.0, p_arr=p_arr, x_arr=x_arr)
        with pytest.raises(ValueError, match="axis"):
            moments.moment(ps, order=0, axis="bad_axis")

    def test_unsupported_order_raises(self, p_arr, x_arr):
        ps = gaussian_phase_space(mu=0.0, sigma=1.0, p_arr=p_arr, x_arr=x_arr)
        with pytest.raises(ValueError, match="order"):
            moments.moment(ps, order=3, axis="p1")
