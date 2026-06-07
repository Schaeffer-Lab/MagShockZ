"""Tests for src/temperature_anisotropy.py."""

import numpy as np
import pytest

from conftest import FakeAxis, FakeH5Data, make_phase_space


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def gaussian_phase_space(mu, sigma, p_arr, x_arr):
    f_p = np.exp(-0.5 * ((p_arr - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    values = np.outer(f_p, np.ones(len(x_arr)))
    return make_phase_space(p_arr, x_arr, values)


# ---------------------------------------------------------------------------
# safe_ratio
# ---------------------------------------------------------------------------

class TestSafeRatio:
    def setup_method(self):
        # Import inside method so sys.path manipulation is in effect
        import temperature_anisotropy as ta
        self.ta = ta

    def test_normal_division(self):
        num = np.array([2.0, 4.0, 6.0])
        den = np.array([1.0, 2.0, 3.0])
        result = self.ta.safe_ratio(num, den)
        np.testing.assert_allclose(result, [2.0, 2.0, 2.0])

    def test_zero_denominator_gives_nan(self):
        num = np.array([1.0, 2.0])
        den = np.array([0.0, 1.0])
        result = self.ta.safe_ratio(num, den)
        assert np.isnan(result[0])
        assert result[1] == pytest.approx(2.0)

    def test_below_floor_gives_nan(self):
        num = np.array([1.0])
        den = np.array([1e-15])  # below default floor of 1e-10
        result = self.ta.safe_ratio(num, den)
        assert np.isnan(result[0])

    def test_custom_floor(self):
        num = np.array([1.0])
        den = np.array([0.5])
        # With a very high floor, even 0.5 is below it → nan
        result = self.ta.safe_ratio(num, den, floor=1.0)
        assert np.isnan(result[0])

    def test_negative_denominator_below_floor(self):
        """safe_ratio checks |denominator|, so negatives near zero also → nan."""
        num = np.array([1.0])
        den = np.array([-1e-12])
        result = self.ta.safe_ratio(num, den)
        assert np.isnan(result[0])

    def test_all_finite_when_denominator_large(self):
        num = np.ones(10)
        den = np.full(10, 100.0)
        result = self.ta.safe_ratio(num, den)
        assert np.all(np.isfinite(result))


# ---------------------------------------------------------------------------
# temperature_profile
# ---------------------------------------------------------------------------

class TestTemperatureProfile:
    def setup_method(self):
        import temperature_anisotropy as ta
        self.ta = ta
        self.p_arr = np.linspace(-10.0, 10.0, 2001)
        self.x_arr = np.linspace(0.0, 50.0, 20)

    def test_electron_temperature_unit_gaussian(self):
        """Electrons: rqm=1, sigma=1 Gaussian → T = 1 (in m_e c^2 units)."""
        ps = gaussian_phase_space(0.0, 1.0, self.p_arr, self.x_arr)
        T = self.ta.temperature_profile(ps, rqm=1.0, momentum_axis="p1")
        np.testing.assert_allclose(T, 1.0, atol=1e-3)

    def test_rqm_scales_temperature(self):
        """Temperature scales with |rqm| (the species mass factor)."""
        ps = gaussian_phase_space(0.0, 1.0, self.p_arr, self.x_arr)
        T_e = self.ta.temperature_profile(ps, rqm=1.0, momentum_axis="p1")
        T_i = self.ta.temperature_profile(ps, rqm=100.0, momentum_axis="p1")
        np.testing.assert_allclose(T_i, 100.0 * T_e, rtol=1e-3)

    def test_negative_rqm_same_as_positive(self):
        """Sign of rqm (charge sign) must not affect temperature."""
        ps = gaussian_phase_space(0.0, 2.0, self.p_arr, self.x_arr)
        T_pos = self.ta.temperature_profile(ps, rqm=+50.0, momentum_axis="p1")
        T_neg = self.ta.temperature_profile(ps, rqm=-50.0, momentum_axis="p1")
        np.testing.assert_allclose(T_pos, T_neg, rtol=1e-10)

    def test_wider_distribution_higher_temperature(self):
        ps_cold = gaussian_phase_space(0.0, 0.5, self.p_arr, self.x_arr)
        ps_hot = gaussian_phase_space(0.0, 3.0, self.p_arr, self.x_arr)
        T_cold = self.ta.temperature_profile(ps_cold, rqm=1.0, momentum_axis="p1")
        T_hot = self.ta.temperature_profile(ps_hot, rqm=1.0, momentum_axis="p1")
        assert np.all(T_hot > T_cold)


# ---------------------------------------------------------------------------
# region_averages
# ---------------------------------------------------------------------------

class TestRegionAverages:
    def setup_method(self):
        import temperature_anisotropy as ta
        self.ta = ta

    def test_known_averages(self):
        x = np.linspace(0.0, 100.0, 101)
        # upstream: x > 60, downstream: 20 <= x <= 60
        arr = np.where(x > 60, 10.0, 1.0)
        up_mean, dn_mean = self.ta.region_averages(
            arr, x, x_shock=60.0, x_downstream_start=20.0
        )
        assert up_mean == pytest.approx(10.0)
        assert dn_mean == pytest.approx(1.0)

    def test_uniform_array_same_means(self):
        x = np.linspace(0.0, 100.0, 100)
        arr = np.ones_like(x) * 5.0
        up_mean, dn_mean = self.ta.region_averages(
            arr, x, x_shock=50.0, x_downstream_start=10.0
        )
        assert up_mean == pytest.approx(5.0)
        assert dn_mean == pytest.approx(5.0)
