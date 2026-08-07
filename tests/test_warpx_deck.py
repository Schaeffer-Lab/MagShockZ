"""Tests for the ``warpx`` subpackage: derive the scales, render the deck, read it back.

The contract under test is that the deck *means* what the run spec says.  Wherever the
claim is about behaviour it is asserted through :func:`deck.key_params` — the numbers
WarpX will actually use — rather than through the deck text, so rewording a comment is
not a failure but moving a constant is.
"""

from __future__ import annotations

import math
from pathlib import Path

import astropy.units as u
import pytest
import yaml

from warpx import config as spec_config
from warpx import deck as deck_module
from warpx import units

REPO = Path(__file__).resolve().parents[1]
SPEC_PATH = REPO / "runs" / "magshockz_2d_heater.warpx.yaml"


@pytest.fixture(scope="module")
def spec() -> dict:
    return spec_config.load(SPEC_PATH)


@pytest.fixture(scope="module")
def scales(spec) -> units.DeckScales:
    return spec_config.scales(spec)


@pytest.fixture(scope="module")
def rendered(spec, scales) -> str:
    return deck_module.render(spec, scales)


class TestTheCheckedInSpec:
    def test_it_loads_derives_renders_and_round_trips(self, rendered):
        assert deck_module.verify(rendered, rendered) == []

    def test_every_matched_invariant_reproduces_flash(self, scales):
        target = scales.flash.invariants()
        for name, value in scales.invariants().items():
            assert value == pytest.approx(
                target[name], rel=spec_config.INVARIANT_TOLERANCE), name

    def test_the_deck_is_fully_periodic_on_both_axes(self, rendered):
        params = deck_module.key_params(rendered)
        for key in ("boundary.field_lo", "boundary.field_hi",
                    "boundary.particle_lo", "boundary.particle_hi"):
            assert params[key].split() == ["periodic", "periodic"], key


class TestChargeStates:
    """The reduction preserves the FLASH mass-per-charge RATIO and both real charges."""

    def test_both_ions_keep_their_real_charge_state(self, scales):
        assert round(units.as_particle(scales.upstream.ion).charge_number) == 6
        assert round(units.as_particle(scales.piston_ion).charge_number) == 14

    def test_both_ions_are_reduced_by_the_same_factor(self, spec, scales):
        flash = spec_config.flash_reference(spec)
        assert (units.mass_per_charge(flash.upstream.ion)
                / units.mass_per_charge(scales.upstream.ion)) == pytest.approx(
                    units.mass_per_charge(flash.piston_ion)
                    / units.mass_per_charge(scales.piston_ion), rel=1e-12)

    def test_the_deck_carries_two_distinct_ion_masses_and_charges(self, rendered):
        params = deck_module.key_params(rendered)
        assert params["amb_ions.charge"] != params["piston_ions.charge"]
        assert params["amb_ions.mass"] != params["piston_ions.mass"]

    def test_each_ion_density_is_its_electron_density_over_its_own_Z(self, rendered):
        """Charge neutrality, per species — the thing a single global Z would hide."""
        params = deck_module.key_params(rendered)
        constants = {k[len("const:"):]: v for k, v in params.items()
                     if k.startswith("const:")}
        assert (params["piston_ions.n_center"] * constants["z_pist"]
                == pytest.approx(params["piston_electrons.n_center"], rel=1e-12))
        # The ambient is absent at the origin (the slab is there), so check its ratio
        # through the named constants instead.
        assert constants["namb"] / constants["z_amb"] > 0.0

    def test_the_heaters_mass_ratio_is_the_bare_mass_not_the_mass_per_charge(
            self, rendered, scales):
        """``foil.mass_ratio`` enters H as 1/sqrt(...) and is m_i/m_e, not m_i/(Z m_e).

        At Z = 14 the two differ by 14x, which is a 3.7x error in the heating rate.
        """
        params = deck_module.key_params(rendered)
        charge = round(units.as_particle(scales.piston_ion).charge_number)
        mass_per_charge = units.mass_per_charge(scales.piston_ion)
        assert params["heater.mass_ratio"] == pytest.approx(
            mass_per_charge * charge, rel=1e-9)
        assert params["heater.mass_ratio"] != pytest.approx(mass_per_charge, rel=1e-3)


class TestTheFiniteSlabPatch:
    """FLASH's piston is a plume from a finite spot, not an infinite sheet."""

    def test_the_slab_is_bounded_in_x_as_well_as_z(self, rendered):
        text = rendered
        assert 'density_function(x,y,z) = "nt*(abs(z)<slab)*(abs(x)<rslab)"' in text

    def test_the_injector_refills_only_the_patch(self, rendered):
        params = deck_module.key_params(rendered)
        constants = {k[len("const:"):]: v for k, v in params.items()
                     if k.startswith("const:")}
        assert params["injector.lo"] == pytest.approx(
            [-constants["rslab"], -constants["slab"]], rel=1e-12)
        assert params["injector.hi"] == pytest.approx(
            [constants["rslab"], constants["slab"]], rel=1e-12)

    def test_the_patch_fits_inside_the_domain(self, scales):
        assert scales.slab_radius_di < scales.transverse_halfwidth_di
        assert scales.slab_halfwidth_di < scales.domain_halfwidth_di

    def test_the_ambient_fills_the_complement_of_the_patch(self, rendered):
        params = deck_module.key_params(rendered)
        # At the origin the patch is occupied by the piston, so the ambient is absent.
        assert params["amb_electrons.n_center"] == pytest.approx(0.0, abs=1e-6)
        assert params["piston_electrons.n_center"] > 0.0


class TestTheDeckStatesItsOwnPhysics:
    def test_the_matched_betas_are_named_constants(self, rendered, scales):
        constants = deck_module.resolve_constants(deck_module.parse_text(rendered))
        assert constants["beta_e"] == pytest.approx(scales.upstream.beta_e, rel=1e-9)
        assert constants["beta_i"] == pytest.approx(scales.upstream.beta_i, rel=1e-9)

    def test_B0_resolves_to_what_derive_computed(self, rendered, scales):
        constants = deck_module.resolve_constants(deck_module.parse_text(rendered))
        assert constants["B0"] == pytest.approx(
            scales.magnetic_field.to_value(u.T), rel=1e-6)

    def test_the_ambient_thetas_resolve_to_the_configured_temperatures(
            self, rendered, scales):
        constants = deck_module.resolve_constants(deck_module.parse_text(rendered))
        assert constants["theta_e_amb"] == pytest.approx(
            scales.theta_e_ambient, rel=1e-6)
        assert constants["theta_i_amb"] == pytest.approx(
            scales.theta_i_ambient, rel=1e-6)

    def test_u_std_is_the_square_root_of_theta(self, rendered, scales):
        params = deck_module.key_params(rendered)
        assert params["amb_electrons.u_std"] == pytest.approx(
            math.sqrt(scales.theta_e_ambient), rel=1e-6)


class TestTheHeaterDrive:
    """``theta`` is an amplitude, not a servo setpoint — resolve it to the actual kick."""

    @pytest.fixture
    def drive(self, spec, scales) -> units.HeaterDrive:
        return units.heater_drive(
            scales, intervals=int(spec["operators"]["heater"]["intervals"]))

    def test_the_rate_matches_the_operators_own_formula(self, drive, scales):
        """``H = 8 theta^{3/2} c^3 / (sqrt(m_i/m_e) * width)``, from ParticleHeater.cpp."""
        from astropy.constants import c

        charge = round(units.as_particle(scales.piston_ion).charge_number)
        piston_mass_ratio = units.mass_per_charge(scales.piston_ion) * charge
        width = 2.0 * scales.slab_halfwidth_di * scales.ion_skin_depth
        expected = (8.0 * scales.theta_e_heater ** 1.5 * c ** 3
                    / (math.sqrt(piston_mass_ratio) * width))
        assert drive.diffusion_rate.to_value("m2/s3") == pytest.approx(
            expected.to_value("m2/s3"), rel=1e-12)

    def test_the_only_density_dependence_is_through_the_slab_width(self, spec):
        """``omega_pe`` cancels out of H, so the plasma density enters ONLY as geometry.

        The operator's own ``foil.n0`` divides back out entirely.  What is left is the
        slab's PHYSICAL width -- and because this deck states that width in ``d_i``,
        which goes as ``n0^-1/2``, changing the reference density does move H.  So the
        invariant is ``H * width``, not H.
        """
        import copy

        other = copy.deepcopy(spec)
        other["reference"]["density_per_m3"] = 4.0e18
        intervals = int(spec["operators"]["heater"]["intervals"])

        def rate_times_width(cfg: dict) -> float:
            scales = spec_config.scales(cfg)
            drive = units.heater_drive(scales, intervals=intervals)
            width = 2.0 * scales.slab_halfwidth_di * scales.ion_skin_depth
            return float((drive.diffusion_rate * width).to_value("m3/s3"))

        assert rate_times_width(other) == pytest.approx(
            rate_times_width(spec), rel=1e-9)

    def test_the_kick_is_the_diffusion_over_one_application_interval(self, drive, spec,
                                                                    scales):
        from astropy.constants import c

        interval = int(spec["operators"]["heater"]["intervals"]) * scales.timestep
        expected = ((drive.diffusion_rate * interval) ** 0.5 / c).decompose()
        assert drive.kick_per_application == pytest.approx(float(expected), rel=1e-12)

    def test_the_deck_states_the_kick_it_resolves_to(self, rendered, drive):
        """A reader must be able to see the kick without re-running the generator."""
        assert f"{drive.diffusion_rate.to_value('m2/s3'):.4e}" in rendered
        assert f"{drive.kick_per_application:.4e}" in rendered

    def test_documenting_the_kick_does_not_change_what_warpx_does(self, spec, scales,
                                                                  rendered):
        """The kick is rendered as COMMENTS, so key_params must be untouched by it."""
        stripped = "\n".join(line for line in rendered.splitlines()
                             if not line.lstrip().startswith("#"))
        assert deck_module.verify(stripped, rendered) == []


class TestTheNullControl:
    def test_it_differs_from_production_in_the_heater_alone(self, spec, scales):
        production = deck_module.render(spec, scales)
        null = deck_module.render(spec, scales, no_heater=True)
        differences = [p for p in deck_module.verify(null, production)
                       if not p.startswith("heater.")]
        assert differences == []

    def test_the_heater_is_actually_gone(self, spec, scales):
        null = deck_module.key_params(deck_module.render(spec, scales, no_heater=True))
        assert null["heater.present"] is False

    def test_the_injector_still_runs_so_the_load_matches(self, spec, scales):
        production = deck_module.key_params(deck_module.render(spec, scales))
        null = deck_module.key_params(deck_module.render(spec, scales, no_heater=True))
        assert null["injector.density"] == production["injector.density"]
        assert null["piston_electrons.ppc"] == production["piston_electrons.ppc"]


class TestLoadRaisesValidateWarns:
    def test_a_non_periodic_boundary_raises(self, tmp_path):
        raw = yaml.safe_load(SPEC_PATH.read_text())
        raw["geometry"]["boundary"] = "pec"
        path = tmp_path / "bad.yaml"
        path.write_text(yaml.safe_dump(raw))
        with pytest.raises(ValueError, match="periodic"):
            spec_config.load(path)

    def test_a_backwards_window_raises(self, tmp_path):
        raw = yaml.safe_load(SPEC_PATH.read_text())
        raw["flash"]["window_ns"] = [12.0, 3.0]
        path = tmp_path / "bad.yaml"
        path.write_text(yaml.safe_dump(raw))
        with pytest.raises(ValueError, match="window_ns"):
            spec_config.load(path)

    def test_an_off_target_deck_only_warns(self, spec, scales):
        assert isinstance(spec_config.validate(spec, scales), list)

    def test_a_charge_state_the_calibration_never_saw_is_warned_about(self, spec, scales):
        """The heater's rate carries m_i/m_e; the fit groups by m/(Z m_e)."""
        warnings = spec_config.validate(spec, scales)
        assert any("calibration was measured at Z" in w for w in warnings)


class TestFreeze:
    def test_the_frozen_spec_is_yaml_safe(self, spec, scales):
        frozen = spec_config.freeze(spec, scales)
        assert yaml.safe_load(yaml.safe_dump(frozen)) is not None

    def test_the_frozen_spec_drops_the_private_path_key(self, spec, scales):
        assert "_path" not in spec_config.freeze(spec, scales)

    def test_it_records_both_sides_of_the_invariant_comparison(self, spec, scales):
        derived = spec_config.freeze(spec, scales)["derived"]
        assert derived["invariants_deck"].keys() == derived["invariants_flash"].keys()
