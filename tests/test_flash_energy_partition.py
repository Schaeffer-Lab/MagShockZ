"""Tests for src/flash_energy_partition.py.

All tests use unyt arrays to mirror real FLASH lineout data, verifying
dimensional correctness and numerical values against known analytic results.
"""

import numpy as np
import pytest
import unyt

from conftest import make_phase_space  # noqa: F401 — triggers path setup
import flash_energy_partition as fep


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_inputs(N=50, v_shock_cgs=0.0):
    """Return a dict of uniform synthetic FLASH lineout inputs."""
    x = np.linspace(0.0, 1e10, N) * unyt.cm

    ne    = np.full(N, 1e8)   * unyt.cm**-3
    Te    = np.full(N, 1.0)   * unyt.eV
    Ti    = np.full(N, 2.0)   * unyt.eV
    n_ion = np.full(N, 1e8)   * unyt.cm**-3
    rho   = np.full(N, 1.67e-15) * unyt.g / unyt.cm**3   # ~1e8 protons/cm³
    v_para = np.full(N, 1e7)  * unyt.cm / unyt.s
    B_mag = np.full(N, 1e-3)  * unyt.gauss

    return dict(ne=ne, Te=Te, Ti=Ti, n_ion=n_ion, rho=rho,
                v_para=v_para, v_shock=v_shock_cgs, B_mag=B_mag, x=x)


# ---------------------------------------------------------------------------
# energy_densities
# ---------------------------------------------------------------------------

class TestEnergyDensities:
    def test_output_keys(self):
        inp = make_inputs()
        result = fep.energy_densities(
            inp["ne"], inp["Te"], inp["Ti"], inp["n_ion"],
            inp["rho"], inp["v_para"], inp["v_shock"], inp["B_mag"],
        )
        assert set(result) == {"u_kinetic", "u_th_e", "u_th_i", "u_mag"}

    def test_output_units_are_erg_per_cm3(self):
        inp = make_inputs()
        result = fep.energy_densities(
            inp["ne"], inp["Te"], inp["Ti"], inp["n_ion"],
            inp["rho"], inp["v_para"], inp["v_shock"], inp["B_mag"],
        )
        for key, arr in result.items():
            assert arr.units == unyt.erg / unyt.cm**3, f"{key} has wrong units: {arr.units}"

    def test_kinetic_zero_when_v_equals_v_shock(self):
        """When v_para == v_shock, ram energy must be zero."""
        inp = make_inputs()
        v = 3e7  # cm/s
        inp["v_para"] = np.full(50, v) * unyt.cm / unyt.s
        result = fep.energy_densities(
            inp["ne"], inp["Te"], inp["Ti"], inp["n_ion"],
            inp["rho"], inp["v_para"], v_shock=v, B_mag=inp["B_mag"],
        )
        np.testing.assert_allclose(result["u_kinetic"].value, 0.0, atol=1e-30)

    def test_kinetic_energy_formula(self):
        """u_kinetic = ½ ρ dv²; verify against hand calculation."""
        N = 10
        rho = np.full(N, 2.0) * unyt.g / unyt.cm**3
        dv  = 3.0  # cm/s — v_para - v_shock
        v_para = np.full(N, dv) * unyt.cm / unyt.s
        zeros = np.zeros(N)
        ne = np.full(N, 1.0) * unyt.cm**-3
        Ti = Te = np.full(N, 1.0) * unyt.eV
        n_ion = ne
        B_mag = np.full(N, 1e-6) * unyt.gauss

        result = fep.energy_densities(ne, Te, Ti, n_ion, rho, v_para, 0.0, B_mag)
        expected = 0.5 * 2.0 * dv**2  # erg/cm³
        np.testing.assert_allclose(
            result["u_kinetic"].to("erg/cm**3").value, expected, rtol=1e-10
        )

    def test_thermal_electron_formula(self):
        """u_th_e = 1.5 * ne * kTe; verify numerically."""
        N = 5
        ne_val = 1e10  # cm^-3
        Te_val = 1.0   # eV
        ne = np.full(N, ne_val) * unyt.cm**-3
        Te = np.full(N, Te_val) * unyt.eV
        Ti = Te.copy()
        n_ion = ne.copy()
        rho = np.full(N, 1e-15) * unyt.g / unyt.cm**3
        v_para = np.zeros(N) * unyt.cm / unyt.s
        B_mag = np.full(N, 1e-6) * unyt.gauss

        result = fep.energy_densities(ne, Te, Ti, n_ion, rho, v_para, 0.0, B_mag)
        expected = (1.5 * (ne_val * unyt.cm**-3) * (Te_val * unyt.eV)).to("erg/cm**3").value
        np.testing.assert_allclose(
            result["u_th_e"].to("erg/cm**3").value, expected, rtol=1e-6
        )

    def test_magnetic_energy_formula(self):
        """u_mag = B²/(8π) in Gaussian CGS."""
        N = 5
        B_val = 1e-3  # Gauss
        ne = np.full(N, 1e8) * unyt.cm**-3
        Te = Ti = np.full(N, 1.0) * unyt.eV
        n_ion = ne.copy()
        rho = np.full(N, 1e-15) * unyt.g / unyt.cm**3
        v_para = np.zeros(N) * unyt.cm / unyt.s
        B_mag = np.full(N, B_val) * unyt.gauss

        result = fep.energy_densities(ne, Te, Ti, n_ion, rho, v_para, 0.0, B_mag)
        expected = B_val**2 / (8.0 * np.pi)  # erg/cm³
        np.testing.assert_allclose(
            result["u_mag"].to("erg/cm**3").value, expected, rtol=1e-6
        )

    def test_bare_float_v_shock_accepted(self):
        """v_shock may be passed as a plain float (cm/s)."""
        inp = make_inputs()
        result = fep.energy_densities(
            inp["ne"], inp["Te"], inp["Ti"], inp["n_ion"],
            inp["rho"], inp["v_para"], v_shock=1e7, B_mag=inp["B_mag"],
        )
        assert "u_kinetic" in result

    def test_all_channels_non_negative(self):
        inp = make_inputs()
        result = fep.energy_densities(
            inp["ne"], inp["Te"], inp["Ti"], inp["n_ion"],
            inp["rho"], inp["v_para"], inp["v_shock"], inp["B_mag"],
        )
        for key, arr in result.items():
            assert np.all(arr.value >= 0.0), f"{key} has negative values"


# ---------------------------------------------------------------------------
# momentum_fluxes  (the conserved total-pressure channels)
# ---------------------------------------------------------------------------

class TestMomentumFluxes:
    def test_output_keys_and_units(self):
        inp = make_inputs()
        result = fep.momentum_fluxes(
            inp["ne"], inp["Te"], inp["Ti"], inp["n_ion"],
            inp["rho"], inp["v_para"], inp["v_shock"], inp["B_mag"],
        )
        assert set(result) == {"p_ram", "p_th_e", "p_th_i", "p_mag"}
        for key, arr in result.items():
            assert arr.units == unyt.dyn / unyt.cm**2, f"{key} wrong units: {arr.units}"

    def test_ram_pressure_formula(self):
        """p_ram = ρ dv² — note NO ½ (twice the ram energy density)."""
        N = 10
        rho = np.full(N, 2.0) * unyt.g / unyt.cm**3
        dv = 3.0
        v_para = np.full(N, dv) * unyt.cm / unyt.s
        ne = np.full(N, 1.0) * unyt.cm**-3
        Te = Ti = np.full(N, 1.0) * unyt.eV
        B_mag = np.full(N, 1e-6) * unyt.gauss
        result = fep.momentum_fluxes(ne, Te, Ti, ne, rho, v_para, 0.0, B_mag)
        np.testing.assert_allclose(
            result["p_ram"].to("dyn/cm**2").value, 2.0 * dv**2, rtol=1e-10
        )

    def test_thermal_pressure_is_nkT(self):
        """p_th_e = nₑ kTₑ — no 3/2 (two-thirds the thermal energy density)."""
        N = 5
        ne_val, Te_val = 1e10, 1.0
        ne = np.full(N, ne_val) * unyt.cm**-3
        Te = np.full(N, Te_val) * unyt.eV
        rho = np.full(N, 1e-15) * unyt.g / unyt.cm**3
        v_para = np.zeros(N) * unyt.cm / unyt.s
        B_mag = np.full(N, 1e-6) * unyt.gauss
        result = fep.momentum_fluxes(ne, Te, Te, ne, rho, v_para, 0.0, B_mag)
        expected = ((ne_val * unyt.cm**-3) * (Te_val * unyt.eV)).to("dyn/cm**2").value
        np.testing.assert_allclose(
            result["p_th_e"].to("dyn/cm**2").value, expected, rtol=1e-6
        )

    def test_prefactors_relative_to_energy_densities(self):
        """The documented prefactors: p_ram=2·u_kinetic, p_th=2/3·u_th, p_mag=u_mag."""
        inp = make_inputs(v_shock_cgs=2e6)
        e = fep.energy_densities(
            inp["ne"], inp["Te"], inp["Ti"], inp["n_ion"],
            inp["rho"], inp["v_para"], inp["v_shock"], inp["B_mag"],
        )
        p = fep.momentum_fluxes(
            inp["ne"], inp["Te"], inp["Ti"], inp["n_ion"],
            inp["rho"], inp["v_para"], inp["v_shock"], inp["B_mag"],
        )
        cu = lambda a: a.to("erg/cm**3").value
        np.testing.assert_allclose(cu(p["p_ram"]),  2.0 * cu(e["u_kinetic"]), rtol=1e-10)
        np.testing.assert_allclose(cu(p["p_th_e"]), (2.0 / 3.0) * cu(e["u_th_e"]), rtol=1e-10)
        np.testing.assert_allclose(cu(p["p_th_i"]), (2.0 / 3.0) * cu(e["u_th_i"]), rtol=1e-10)
        np.testing.assert_allclose(cu(p["p_mag"]),  cu(e["u_mag"]), rtol=1e-10)

    def test_transverse_field_excludes_normal_component(self):
        """With B_para given, p_mag uses B_t² = B_mag² − B_para²."""
        N = 5
        ne = np.full(N, 1e8) * unyt.cm**-3
        Te = Ti = np.full(N, 1.0) * unyt.eV
        rho = np.full(N, 1e-15) * unyt.g / unyt.cm**3
        v_para = np.zeros(N) * unyt.cm / unyt.s
        B_mag = np.full(N, 100.0) * unyt.gauss
        B_para = np.full(N, 60.0) * unyt.gauss          # B_perp = 80
        result = fep.momentum_fluxes(ne, Te, Ti, ne, rho, v_para, 0.0,
                                     B_mag, B_para=B_para)
        expected = 80.0**2 / (8.0 * np.pi)               # Gaussian erg/cm³ = dyn/cm²
        np.testing.assert_allclose(
            result["p_mag"].to("dyn/cm**2").value, expected, rtol=1e-6
        )


# ---------------------------------------------------------------------------
# continuity_check  (the conservation test on the partition)
# ---------------------------------------------------------------------------

class TestContinuityCheck:
    N = 100
    x = np.linspace(0.0, 1e10, N)
    x_shock = 6e9
    x_ds = 2e9

    def _partition(self, up_total, dn_total):
        """Each of 4 channels carries 1/4 of the requested per-side total."""
        up_each, dn_each = up_total / 4.0, dn_total / 4.0
        arr = np.where(self.x > self.x_shock, up_each, dn_each) * unyt.dyn / unyt.cm**2
        energy = {k: arr for k in ("p_ram", "p_th_e", "p_th_i", "p_mag")}
        return fep.partition_by_region(energy, self.x, self.x_shock, self.x_ds)

    def test_conserved_flux_ratio_is_one(self):
        chk = fep.continuity_check(self._partition(8.0, 8.0))
        assert chk["ratio"] == pytest.approx(1.0, rel=1e-9)
        assert chk["rel_imbalance"] == pytest.approx(0.0, abs=1e-9)

    def test_ratio_reports_imbalance(self):
        chk = fep.continuity_check(self._partition(up_total=4.0, dn_total=5.0))
        assert chk["ratio"] == pytest.approx(1.25, rel=1e-9)
        assert chk["rel_imbalance"] == pytest.approx(0.25, rel=1e-9)
        for ch_ratio in chk["channels"].values():
            assert ch_ratio == pytest.approx(1.25, rel=1e-9)

    def test_summary_is_string(self):
        s = fep.continuity_summary(fep.continuity_check(self._partition(8.0, 8.0)))
        assert isinstance(s, str) and "dn/up" in s


# ---------------------------------------------------------------------------
# partition_by_region
# ---------------------------------------------------------------------------

class TestFlashPartitionByRegion:
    N = 100
    x = np.linspace(0.0, 1e10, N)  # plain float cm array
    x_shock = 6e9
    x_ds = 2e9

    def _flat_energy(self, up_val, dn_val):
        """Uniform energy dict: up_val upstream, dn_val downstream."""
        arr_np = np.where(self.x > self.x_shock, up_val, dn_val)
        # Wrap in unyt so the function's .to("erg/cm**3") call succeeds
        arr = arr_np * unyt.erg / unyt.cm**3
        return {k: arr for k in ("u_kinetic", "u_th_e", "u_th_i", "u_mag")}

    def test_keys_present(self):
        energy = self._flat_energy(1.0, 2.0)
        result = fep.partition_by_region(energy, self.x, self.x_shock, self.x_ds)
        assert "upstream" in result and "downstream" in result
        for side in ("upstream", "downstream"):
            assert "means" in result[side]
            assert "fractions" in result[side]
            assert "total" in result[side]

    def test_upstream_mean_correct(self):
        energy = self._flat_energy(up_val=10.0, dn_val=1.0)
        result = fep.partition_by_region(energy, self.x, self.x_shock, self.x_ds)
        for ch in ("u_kinetic", "u_th_e", "u_th_i", "u_mag"):
            assert result["upstream"]["means"][ch] == pytest.approx(10.0, rel=1e-6)

    def test_downstream_mean_correct(self):
        energy = self._flat_energy(up_val=1.0, dn_val=5.0)
        result = fep.partition_by_region(energy, self.x, self.x_shock, self.x_ds)
        for ch in ("u_kinetic", "u_th_e", "u_th_i", "u_mag"):
            assert result["downstream"]["means"][ch] == pytest.approx(5.0, rel=1e-6)

    def test_fractions_sum_to_one(self):
        energy = self._flat_energy(1.0, 2.0)
        result = fep.partition_by_region(energy, self.x, self.x_shock, self.x_ds)
        for side in ("upstream", "downstream"):
            total_frac = sum(result[side]["fractions"].values())
            assert total_frac == pytest.approx(1.0, rel=1e-6)

    def test_total_equals_sum_of_means(self):
        energy = self._flat_energy(3.0, 7.0)
        result = fep.partition_by_region(energy, self.x, self.x_shock, self.x_ds)
        for side in ("upstream", "downstream"):
            s = sum(result[side]["means"].values())
            assert result[side]["total"] == pytest.approx(s, rel=1e-10)

    def test_raises_on_empty_upstream(self):
        energy = self._flat_energy(1.0, 1.0)
        with pytest.raises(ValueError, match="Empty region"):
            fep.partition_by_region(energy, self.x, x_shock=2e10, x_downstream_start=0.0)

    def test_raises_on_empty_downstream(self):
        energy = self._flat_energy(1.0, 1.0)
        with pytest.raises(ValueError, match="Empty region"):
            fep.partition_by_region(energy, self.x, x_shock=1e9, x_downstream_start=5e9)


# ---------------------------------------------------------------------------
# compression_check — FLASH compression vs Rankine--Hugoniot
# ---------------------------------------------------------------------------

class TestCompressionCheck:
    # A super-fast upstream state (CGS); proton plasma.
    def _upstream(self, B_para=0.0, B_mag=100.0):
        return dict(ne=1e16, n_ion=1e16, Te=10.0, Ti=10.0,
                    rho=1e16 * 1.6726e-24, B_mag=B_mag, B_para=B_para)

    def test_measured_density_ratio_is_exact(self):
        up = self._upstream()
        dn = dict(up, rho=up["rho"] * 3.0, B_mag=300.0)
        chk = fep.compression_check(up, dn, v_inflow=5e7)
        assert chk["r_measured"] == pytest.approx(3.0)

    def test_perpendicular_identity_Bt_equals_r(self):
        # B purely transverse (B_para = 0) => perpendicular shock: Bt2/Bt1 = r.
        up = self._upstream(B_para=0.0)
        chk = fep.compression_check(up, dict(up), v_inflow=5e7)
        assert chk["theta_bn"] == pytest.approx(np.pi / 2.0)
        assert chk["b_t_RH"] == pytest.approx(chk["r_RH"])
        assert 1.0 < chk["r_RH"] <= 4.0 + 1e-9

    def test_theta_from_field_components(self):
        up = self._upstream(B_para=60.0, B_mag=100.0)   # B_perp = 80
        chk = fep.compression_check(up, dict(up), v_inflow=5e7)
        assert chk["theta_bn"] == pytest.approx(np.arctan2(80.0, 60.0))

    def test_perpendicular_roundtrip(self):
        # Build a downstream consistent with the perpendicular prediction
        # (rho and transverse B both scale by r); measured must match theory.
        up = self._upstream(B_para=0.0, B_mag=100.0)
        r = fep.compression_check(up, dict(up), v_inflow=5e7)["r_RH"]
        dn = dict(up, rho=up["rho"] * r, B_mag=100.0 * r, B_para=0.0)
        chk = fep.compression_check(up, dn, v_inflow=5e7)
        assert chk["r_measured"] == pytest.approx(r, rel=1e-9)
        assert chk["b_t_measured"] == pytest.approx(r, rel=1e-9)
        assert chk["b_t_measured"] == pytest.approx(chk["b_t_RH"], rel=1e-9)

    def test_gamma_is_threaded_through(self):
        up = self._upstream()
        r53 = fep.compression_check(up, dict(up), v_inflow=5e7, gamma=5.0 / 3.0)["r_RH"]
        r2 = fep.compression_check(up, dict(up), v_inflow=5e7, gamma=2.0)["r_RH"]
        assert r53 != pytest.approx(r2)

    def test_summary_is_string_with_fields(self):
        up = self._upstream()
        chk = fep.compression_check(up, dict(up, rho=up["rho"] * 3.0, B_mag=300.0),
                                    v_inflow=5e7)
        s = fep.compression_summary(chk)
        assert isinstance(s, str)
        assert "theta_Bn" in s and "r (density)" in s and "Bt2/Bt1" in s


# ---------------------------------------------------------------------------
# partition_summary
# ---------------------------------------------------------------------------

class TestPartitionSummary:
    def _make_result(self):
        x = np.linspace(0.0, 1e10, 100)
        x_shock = 6e9
        x_ds = 2e9
        arr = np.where(x > x_shock, 1.0, 2.0) * unyt.erg / unyt.cm**3
        energy = {k: arr for k in ("u_kinetic", "u_th_e", "u_th_i", "u_mag")}
        return fep.partition_by_region(energy, x, x_shock, x_ds)

    def test_returns_string(self):
        result = self._make_result()
        summary = fep.partition_summary(result)
        assert isinstance(summary, str)

    def test_contains_channel_labels(self):
        result = self._make_result()
        summary = fep.partition_summary(result)
        for label in ("Kinetic", "Thermal", "Magnetic"):
            assert label in summary

    def test_contains_upstream_and_downstream(self):
        result = self._make_result()
        summary = fep.partition_summary(result)
        assert "Upstream" in summary or "upstream" in summary.lower()
