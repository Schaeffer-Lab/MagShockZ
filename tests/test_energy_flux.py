"""Tests for src/energy_flux.py (shock-frame energy-flux pure functions)."""

import numpy as np
import pytest

from conftest import make_phase_space, FakeAxis, FakeH5Data
import energy_flux as ef


def gaussian_phase_space(mu, sigma, p_arr, x_arr, axis_name="p1"):
    f_p = np.exp(-0.5 * ((p_arr - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    values = np.outer(f_p, np.ones(len(x_arr)))
    if axis_name == "p1":
        return make_phase_space(p_arr, x_arr, values)
    return FakeH5Data(values, [FakeAxis(axis_name, p_arr), FakeAxis("x1", x_arr)])


# ---------------------------------------------------------------------------
# poynting_flux
# ---------------------------------------------------------------------------

class TestPoyntingFlux:
    def test_formula(self):
        e2, e3, b2, b3 = 1.0, 0.0, 0.0, 2.0
        assert ef.poynting_flux(e2, e3, b2, b3) == pytest.approx(2.0)

    def test_antisymmetric_term(self):
        # E3, B2 term enters with a minus sign
        assert ef.poynting_flux(0.0, 1.0, 3.0, 0.0) == pytest.approx(-3.0)

    def test_array_elementwise(self):
        e2 = np.array([1.0, 2.0]); b3 = np.array([1.0, 1.0])
        zeros = np.zeros(2)
        np.testing.assert_allclose(ef.poynting_flux(e2, zeros, zeros, b3), [1.0, 2.0])

    def test_v_shock_default_is_lab_frame(self):
        # Omitting v_shock (default 0) reproduces the original lab-frame flux.
        e2, e3, b2, b3 = 0.7, -0.3, 1.1, 2.0
        assert ef.poynting_flux(e2, e3, b2, b3) == pytest.approx(e2 * b3 - e3 * b2)
        assert ef.poynting_flux(e2, e3, b2, b3, v_shock=0.0) == pytest.approx(e2 * b3 - e3 * b2)

    def test_shock_frame_subtracts_Bperp_advection(self):
        # With zero lab E, the shock-frame flux is the magnetic-enthalpy advection
        # −(v_shock/c)·(B2²+B3²).
        b2, b3, v = 1.5, 2.0, 0.04
        assert ef.poynting_flux(0.0, 0.0, b2, b3, v_shock=v) == pytest.approx(
            -v * (b2**2 + b3**2))

    def test_frozen_in_upstream_gives_shock_frame_enthalpy_flux(self):
        # Ideal-MHD frozen-in field for plasma flowing at v_u (normal) in the lab:
        # E2 = v_u·B3, E3 = −v_u·B2 (Gaussian, c=1).  The shock-frame Poynting
        # flux must equal U·B_perp² with U = v_u − v_shock.
        b2, b3 = 1.3, -0.8
        v_u, v_shock = 0.05, 0.04
        e2, e3 = v_u * b3, -v_u * b2
        U = v_u - v_shock
        assert ef.poynting_flux(e2, e3, b2, b3, v_shock=v_shock) == pytest.approx(
            U * (b2**2 + b3**2))


# ---------------------------------------------------------------------------
# species_energy_flux
# ---------------------------------------------------------------------------

class TestSpeciesEnergyFlux:
    p = np.linspace(-10.0, 10.0, 2001)
    x = np.linspace(0.0, 50.0, 10)

    def test_zero_flux_when_bulk_equals_shock(self):
        """All kinetic flux channels vanish when U = ⟨u⟩ − v_shock = 0."""
        ps = gaussian_phase_space(0.3, 0.5, self.p, self.x)
        Fb, Fi, Fp = ef.species_energy_flux(ps, rqm=1.0, v_shock=0.3)
        for F in (Fb, Fi, Fp):
            np.testing.assert_allclose(F, 0.0, atol=1e-6)

    def test_cold_beam_bulk_flux(self):
        """A cold (σ→0) beam of density 1 at U carries F_bulk ≈ ½ n |rqm| U³."""
        U = 0.4
        ps = gaussian_phase_space(U, 0.02, self.p, self.x)
        Fb, Fi, Fp = ef.species_energy_flux(ps, rqm=1.0, v_shock=0.0)
        np.testing.assert_allclose(Fb, 0.5 * U ** 3, rtol=1e-2)
        np.testing.assert_allclose(Fi, 0.0, atol=1e-3)
        np.testing.assert_allclose(Fp, 0.0, atol=1e-3)

    def test_pressure_and_internal_signs_follow_U(self):
        """For U<0 (upstream inflow) the advective channels are negative."""
        ps = gaussian_phase_space(0.0, 1.0, self.p, self.x)  # ⟨u⟩=0, U=-v_shock<0
        Fb, Fi, Fp = ef.species_energy_flux(ps, rqm=1.0, v_shock=0.05)
        assert np.all(Fi < 0) and np.all(Fp < 0) and np.all(Fb < 0)

    def test_pressure_is_half_internal_when_isotropic_1d(self):
        """With only p1 loaded, ε = ½n m σ² and P_xx = n m σ² ⇒ F_pressure = 2·F_internal."""
        ps = gaussian_phase_space(0.0, 1.0, self.p, self.x)
        _, Fi, Fp = ef.species_energy_flux(ps, rqm=1.0, v_shock=0.1)
        np.testing.assert_allclose(Fp, 2.0 * Fi, rtol=1e-3)

    def test_rqm_scales_all_channels(self):
        ps = gaussian_phase_space(0.2, 1.0, self.p, self.x)
        f1 = ef.species_energy_flux(ps, rqm=1.0, v_shock=0.0)
        f2 = ef.species_energy_flux(ps, rqm=2.0, v_shock=0.0)
        for a, b in zip(f1, f2):
            np.testing.assert_allclose(b, 2.0 * a, rtol=1e-6)

    def test_perp_phase_spaces_add_internal_energy(self):
        """A transverse phase space adds its variance to ε (hence to F_internal)."""
        ps1 = gaussian_phase_space(0.0, 1.0, self.p, self.x)               # σ²=1
        ps2 = gaussian_phase_space(0.0, 2.0, self.p, self.x, "p2")          # σ²=4
        _, Fi_1d, _ = ef.species_energy_flux(ps1, rqm=1.0, v_shock=0.1)
        _, Fi_2d, _ = ef.species_energy_flux(ps1, rqm=1.0, v_shock=0.1,
                                             perp_phase_spaces=[ps2])
        U = -0.1
        np.testing.assert_allclose(Fi_2d - Fi_1d, U * 0.5 * 4.0, rtol=1e-2)

    def test_perp_bulk_adds_to_bulk_ke(self):
        """A drifting transverse population increases |U|² in the bulk KE flux."""
        ps1 = gaussian_phase_space(0.0, 0.05, self.p, self.x)
        ps2_drift = gaussian_phase_space(0.3, 0.05, self.p, self.x, "p2")
        Fb_nodrift, _, _ = ef.species_energy_flux(ps1, rqm=1.0, v_shock=0.1)
        Fb_drift, _, _ = ef.species_energy_flux(ps1, rqm=1.0, v_shock=0.1,
                                                perp_phase_spaces=[ps2_drift])
        # U<0, extra perp KE makes the (negative) flux more negative
        assert np.all(Fb_drift < Fb_nodrift)
