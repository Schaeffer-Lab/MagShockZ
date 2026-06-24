"""Tests for src/energy_partition.py (OSIRIS pure-function module)."""

import numpy as np
import pytest

from conftest import make_phase_space, FakeAxis, FakeH5Data
import energy_partition as ep


def gaussian_phase_space(mu, sigma, p_arr, x_arr):
    f_p = np.exp(-0.5 * ((p_arr - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    values = np.outer(f_p, np.ones(len(x_arr)))
    return make_phase_space(p_arr, x_arr, values)


def gaussian_phase_space_axis(mu, sigma, p_arr, x_arr, axis_name):
    """Gaussian phase space whose momentum axis is named ``axis_name`` (p2/p3)."""
    f_p = np.exp(-0.5 * ((p_arr - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    values = np.outer(f_p, np.ones(len(x_arr)))
    axes = [FakeAxis(axis_name, p_arr), FakeAxis("x1", x_arr)]
    return FakeH5Data(values, axes)


# ---------------------------------------------------------------------------
# species_energy_profiles
# ---------------------------------------------------------------------------

class TestSpeciesEnergyProfiles:
    p_arr = np.linspace(-10.0, 10.0, 2001)
    x_arr = np.linspace(0.0, 50.0, 20)

    def test_output_shapes(self):
        ps = gaussian_phase_space(0.0, 1.0, self.p_arr, self.x_arr)
        u_ram, u_th = ep.species_energy_profiles(ps, rqm=1.0, v_shock=0.0)
        assert u_ram.shape == (len(self.x_arr),)
        assert u_th.shape == (len(self.x_arr),)

    def test_ram_zero_when_bulk_equals_shock_velocity(self):
        """If the bulk velocity equals v_shock the ram energy should be ~0."""
        mu = 0.3  # bulk velocity
        ps = gaussian_phase_space(mu, 0.5, self.p_arr, self.x_arr)
        u_ram, _ = ep.species_energy_profiles(ps, rqm=1.0, v_shock=mu)
        np.testing.assert_allclose(u_ram, 0.0, atol=1e-6)

    def test_thermal_energy_positive(self):
        ps = gaussian_phase_space(0.0, 1.0, self.p_arr, self.x_arr)
        _, u_th = ep.species_energy_profiles(ps, rqm=1.0, v_shock=0.0)
        assert np.all(u_th >= 0.0)

    def test_rqm_scales_both_channels(self):
        """Doubling |rqm| should double both ram and thermal energy densities."""
        ps = gaussian_phase_space(1.0, 1.0, self.p_arr, self.x_arr)
        u_ram_1, u_th_1 = ep.species_energy_profiles(ps, rqm=1.0, v_shock=0.0)
        u_ram_2, u_th_2 = ep.species_energy_profiles(ps, rqm=2.0, v_shock=0.0)
        np.testing.assert_allclose(u_ram_2, 2.0 * u_ram_1, rtol=1e-6)
        np.testing.assert_allclose(u_th_2, 2.0 * u_th_1, rtol=1e-6)

    def test_negative_rqm_same_as_positive(self):
        ps = gaussian_phase_space(0.5, 1.0, self.p_arr, self.x_arr)
        u_ram_p, u_th_p = ep.species_energy_profiles(ps, rqm=+50.0, v_shock=0.0)
        u_ram_n, u_th_n = ep.species_energy_profiles(ps, rqm=-50.0, v_shock=0.0)
        np.testing.assert_allclose(u_ram_p, u_ram_n, rtol=1e-10)
        np.testing.assert_allclose(u_th_p, u_th_n, rtol=1e-10)

    def test_thermal_carries_one_half(self):
        """u_th = 0.5 * n * |rqm| * sigma^2 (one direction); n=1, sigma=1 -> 0.5."""
        ps = gaussian_phase_space(0.0, 1.0, self.p_arr, self.x_arr)
        _, u_th = ep.species_energy_profiles(ps, rqm=1.0, v_shock=0.0)
        np.testing.assert_allclose(u_th, 0.5, rtol=1e-3)

    def test_perp_phase_space_adds_thermal(self):
        """A transverse phase space adds 0.5 * n * |rqm| * sigma_perp^2."""
        ps1 = gaussian_phase_space(0.0, 1.0, self.p_arr, self.x_arr)            # sigma^2 = 1
        ps2 = gaussian_phase_space_axis(0.0, 2.0, self.p_arr, self.x_arr, "p2")  # sigma^2 = 4
        _, u_th_1d = ep.species_energy_profiles(ps1, rqm=1.0, v_shock=0.0)
        _, u_th_2d = ep.species_energy_profiles(
            ps1, rqm=1.0, v_shock=0.0, perp_phase_spaces=[ps2]
        )
        np.testing.assert_allclose(u_th_2d - u_th_1d, 0.5 * 4.0, rtol=1e-3)

    def test_isotropic_three_directions_equals_three_halves_nT(self):
        """Three equal directions -> u_th = (3/2) n T with T = |rqm| sigma^2."""
        ps1 = gaussian_phase_space(0.0, 1.0, self.p_arr, self.x_arr)
        ps2 = gaussian_phase_space_axis(0.0, 1.0, self.p_arr, self.x_arr, "p2")
        ps3 = gaussian_phase_space_axis(0.0, 1.0, self.p_arr, self.x_arr, "p3")
        _, u_th = ep.species_energy_profiles(
            ps1, rqm=1.0, v_shock=0.0, perp_phase_spaces=[ps2, ps3]
        )
        np.testing.assert_allclose(u_th, 1.5, rtol=1e-3)

    def test_perp_variance_is_frame_independent(self):
        """Transverse bulk offset must not change the (central-moment) thermal sum."""
        ps1 = gaussian_phase_space(0.0, 1.0, self.p_arr, self.x_arr)
        ps2_centered = gaussian_phase_space_axis(0.0, 1.5, self.p_arr, self.x_arr, "p2")
        ps2_shifted = gaussian_phase_space_axis(0.4, 1.5, self.p_arr, self.x_arr, "p2")
        _, u_th_c = ep.species_energy_profiles(
            ps1, rqm=1.0, v_shock=0.0, perp_phase_spaces=[ps2_centered]
        )
        _, u_th_s = ep.species_energy_profiles(
            ps1, rqm=1.0, v_shock=0.0, perp_phase_spaces=[ps2_shifted]
        )
        np.testing.assert_allclose(u_th_c, u_th_s, rtol=1e-3)


# ---------------------------------------------------------------------------
# field_energy_profiles
# ---------------------------------------------------------------------------

class TestFieldEnergyProfiles:
    N = 100
    x = np.linspace(0.0, 100.0, N)

    def _uniform_fields(self, val=1.0):
        return (np.full(self.N, val),) * 6  # b1,b2,b3,e1,e2,e3

    def test_output_shapes(self):
        b1, b2, b3, e1, e2, e3 = self._uniform_fields()
        u_B, u_E = ep.field_energy_profiles(b1, b2, b3, e1, e2, e3, self.x, x_shock=50.0)
        assert u_B.shape == (self.N,)
        assert u_E.shape == (self.N,)

    def test_full_mode_uniform_field(self):
        """B² / 2 for uniform unit field along one axis."""
        b1 = np.ones(self.N)
        zeros = np.zeros(self.N)
        u_B, u_E = ep.field_energy_profiles(
            b1, zeros, zeros, zeros, zeros, zeros, self.x, x_shock=50.0, field_mode="full"
        )
        np.testing.assert_allclose(u_B, 0.5)
        np.testing.assert_allclose(u_E, 0.0)

    def test_electric_energy_formula(self):
        """E² / 2 summed over three components."""
        zeros = np.zeros(self.N)
        e1 = np.full(self.N, 1.0)
        e2 = np.full(self.N, 2.0)
        e3 = np.full(self.N, 3.0)
        _, u_E = ep.field_energy_profiles(
            zeros, zeros, zeros, e1, e2, e3, self.x, x_shock=50.0
        )
        expected = 0.5 * (1**2 + 2**2 + 3**2)
        np.testing.assert_allclose(u_E, expected)

    def test_delta_mode_subtracts_upstream_mean(self):
        """In delta mode, upstream region B energy should be ~0 for uniform upstream."""
        b2 = np.where(self.x > 50.0, 3.0, 6.0)  # uniform in upstream half
        zeros = np.zeros(self.N)
        u_B, _ = ep.field_energy_profiles(
            zeros, b2, zeros, zeros, zeros, zeros, self.x, x_shock=50.0,
            field_mode="delta"
        )
        upstream_mask = self.x > 50.0
        np.testing.assert_allclose(u_B[upstream_mask], 0.0, atol=1e-10)

    def test_full_vs_delta_differ_when_background_nonzero(self):
        b2 = np.ones(self.N) * 2.0
        zeros = np.zeros(self.N)
        u_B_full, _ = ep.field_energy_profiles(
            zeros, b2, zeros, zeros, zeros, zeros, self.x, x_shock=50.0, field_mode="full"
        )
        u_B_delta, _ = ep.field_energy_profiles(
            zeros, b2, zeros, zeros, zeros, zeros, self.x, x_shock=50.0, field_mode="delta"
        )
        # delta mode removes background → upstream energy goes to 0
        assert np.all(u_B_full >= u_B_delta - 1e-12)


# ---------------------------------------------------------------------------
# partition_by_region
# ---------------------------------------------------------------------------

class TestPartitionByRegion:
    x = np.linspace(0.0, 100.0, 101)
    x_shock = 60.0
    x_ds_start = 20.0

    def _flat_arrays(self, up_val, dn_val):
        """Arrays that equal up_val upstream and dn_val downstream."""
        arr = np.where(self.x > self.x_shock, up_val, dn_val)
        return arr

    def test_returns_upstream_and_downstream_keys(self):
        arr = self._flat_arrays(1.0, 2.0)
        result = ep.partition_by_region(arr, arr, arr, arr, self.x, self.x_shock, self.x_ds_start)
        assert "upstream" in result
        assert "downstream" in result

    def test_upstream_mean_values(self):
        u_ram = self._flat_arrays(10.0, 0.0)
        zeros = np.zeros_like(self.x)
        result = ep.partition_by_region(u_ram, zeros, zeros, zeros, self.x, self.x_shock, self.x_ds_start)
        assert result["upstream"]["ram"] == pytest.approx(10.0)
        assert result["upstream"]["thermal"] == pytest.approx(0.0)

    def test_downstream_mean_values(self):
        u_th = np.where(
            (self.x >= self.x_ds_start) & (self.x <= self.x_shock), 5.0, 0.0
        )
        zeros = np.zeros_like(self.x)
        result = ep.partition_by_region(zeros, u_th, zeros, zeros, self.x, self.x_shock, self.x_ds_start)
        assert result["downstream"]["thermal"] == pytest.approx(5.0)

    def test_all_channels_present(self):
        arr = np.ones_like(self.x)
        result = ep.partition_by_region(arr, arr, arr, arr, self.x, self.x_shock, self.x_ds_start)
        for side in ("upstream", "downstream"):
            for key in ("ram", "thermal", "B_field", "E_field"):
                assert key in result[side]

    def test_raises_on_empty_upstream(self):
        x = np.linspace(0.0, 50.0, 51)
        # x_shock beyond the grid → no upstream points
        arr = np.ones_like(x)
        with pytest.raises(ValueError, match="Empty region"):
            ep.partition_by_region(arr, arr, arr, arr, x, x_shock=100.0, x_downstream_start=0.0)

    def test_raises_on_empty_downstream(self):
        x = np.linspace(0.0, 100.0, 101)
        arr = np.ones_like(x)
        with pytest.raises(ValueError, match="Empty region"):
            ep.partition_by_region(arr, arr, arr, arr, x, x_shock=10.0, x_downstream_start=20.0)
