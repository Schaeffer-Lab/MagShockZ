"""Tests for heater_deck: rendering the WarpX heater deck and proving it round-trips.

Numpy + PyYAML only, so these run in CI without yt / WarpX / astropy.  The contract
under test is that the deck *means* what the run spec says — asserted through
``key_params`` (the numbers WarpX will actually use) rather than through the deck text,
so a reworded comment is not a test failure but a moved constant is.
"""

import os

import pytest

import heater_deck
import heater_piston_scaling as hps
import heater_spec

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEC_PATH = os.path.join(_REPO, "runs", "magshockz_2d_heater.warpx.yaml")


def make_spec(**overrides) -> dict:
    """A minimal valid heater_pic_2d spec, inline so no fixture file is needed."""
    spec = {
        "schema": heater_deck.SCHEMA,
        "meta": {"run_id": "unit_test", "deck": "inputs_2d_heater"},
        "flash_target": {
            "n_amb_per_m3": 3.0e23,
            "b_amb_tesla": 7.0,
            "te_amb_ev": 120.0,
            "ti_amb_ev": 80.0,
            "n_piston_drive_per_m3": 6.0e24,
            "v_front_ms": 9.0e5,
            "l_piston_m": 250.0e-6,
            "r_spot_m": 500.0e-6,
            "t_window_ns": [0.0, 9.0],
        },
        "scaling": {
            "n0_per_m3": 1.0e18,
            "mass_ratio": 100.0,
            "v_piston_c": 0.03,
            "kappa_expansion": 1.01,
            "theta_e_heater": None,
            "theta_e_cold": 1.0e-3,
            # Coarse on purpose: these tests care about deck STRUCTURE, and a
            # production-resolution grid makes derive() do needless work.
            "cell_size_de": 0.5,
            "slab_halfwidth_di": 2.0,
            "domain_x_halfwidth_de": 40.0,
            "domain_z_halfwidth_de": 200.0,
            "ppc_each_dim": [5, 5],
            "run_window_gyro": 0.05,
        },
        "flow": {"v_flow_kms": 0.0, "impose_motional_e": True},
        "operators": {
            "drive_stop_t_ci": None,
            "heater": {"intervals": 20, "k": 2.0},
            "injector": {"intervals": 20, "tau_over_wpe_inv": 10.0,
                         "ppc_reference": 100},
        },
        "numerics": {"particle_shape": 3, "cfl": 0.75, "max_grid_size": 256,
                     "blocking_factor": 8},
        "diagnostics": {"plotfile_intervals": 1400, "reduced_intervals": 20,
                        "phase_space_intervals": 11000, "phase_space_fraction": 0.02,
                        "checkpoint_intervals": 14000},
        "smoke": {"domain_scale": 0.25, "ppc_each_dim": [2, 2], "max_step": 200,
                  "plotfile_intervals": 50, "reduced_intervals": 10},
    }
    for block, values in overrides.items():
        if isinstance(values, dict) and isinstance(spec.get(block), dict):
            spec[block] = {**spec[block], **values}
        else:
            spec[block] = values
    return spec


def make_scaling(spec: dict | None = None, **kwargs) -> hps.ReducedScaling:
    return heater_spec.scaling(spec or make_spec(), **kwargs)


def render(spec: dict | None = None, **kwargs) -> str:
    spec = spec or make_spec()
    return heater_deck.render_deck(spec, make_scaling(spec), **kwargs)


def params(spec: dict | None = None, **kwargs) -> dict:
    return heater_deck.key_params_from_text(render(spec, **kwargs))


class TestMyConstants:
    def test_every_constant_resolves(self):
        """A constant referring to an undefined symbol would raise here, not at runtime."""
        entries = heater_deck._parse_inputs_text(render())
        constants = heater_deck.resolve_constants(entries)
        assert {"n0", "mass_ratio", "de", "di", "nt", "namb", "vA", "B0",
                "slab", "rspot", "xhalf", "zhalf"} <= set(constants)

    def test_slab_is_a_named_constant_in_ion_units(self):
        deck = render()
        assert "my_constants.slab  = 2.000000*di" in deck
        # ...and is referenced, never re-inlined, so the width has one definition.
        assert "*di)" not in deck.split("my_constants.slab")[1]

    def test_spot_radius_is_written_in_ion_units(self):
        """r_spot/d_i is a MATCHED invariant, so the deck should state it in d_i."""
        deck = render()
        assert "my_constants.rspot = " in deck and "*di" in deck.split("rspot = ")[1][:20]
        assert "particle_heater.foil.spot_radius = rspot" in deck

    def test_u_std_is_the_square_root_of_a_named_theta(self):
        """No spliced thermal decimals: u_std IS sqrt(theta), and should read that way."""
        deck = render()
        for species, theta in (("piston_electrons", "theta_e_cold"),
                               ("piston_ions", "theta_i_cold"),
                               ("amb_electrons", "theta_e_amb"),
                               ("amb_ions", "theta_i_amb")):
            assert f"{species}.ux_std = sqrt({theta})" in deck

    def test_mass_ratio_appears_once_as_a_named_constant(self):
        deck = render()
        assert "my_constants.mass_ratio = 100" in deck
        assert "my_constants.mi    = mass_ratio*m_e" in deck
        assert "particle_heater.foil.mass_ratio = mass_ratio" in deck

    def test_ambient_thetas_resolve_to_the_target_betas(self):
        """The deck's beta_e*B0^2/(...) form must reproduce derive()'s temperatures."""
        scaling = make_scaling()
        got = params()
        assert got["u_std:amb_electrons"] == pytest.approx(
            scaling.theta_e_amb ** 0.5, rel=1e-9)
        assert got["u_std:amb_ions"] == pytest.approx(
            scaling.theta_i_amb ** 0.5, rel=1e-9)

    def test_cold_ions_share_the_electrons_physical_temperature(self):
        assert "my_constants.theta_i_cold = theta_e_cold/mass_ratio" in render()

    def test_vflow_constant_is_absent_when_there_is_no_drift(self):
        """AMReX reports unused ParmParse entries; a permanent one trains you to ignore it."""
        assert "vflow" not in render()
        drifting = make_spec(flow={"v_flow_kms": 50.0, "impose_motional_e": True})
        assert "my_constants.vflow" in render(drifting)
        assert 'Ez_external_grid_function(x,y,z) = "-vflow*B0"' in render(drifting)


class TestRoundTrip:
    def test_a_freshly_rendered_deck_verifies_against_itself(self, tmp_path):
        spec = make_spec()
        path = tmp_path / "inputs"
        path.write_text(render(spec))
        assert heater_deck.verify(spec, make_scaling(spec), path) == []

    def test_resolved_b0_equals_the_scalings_b0(self):
        """Deck-vs-scaling drift, which verify() (deck-vs-deck) cannot see."""
        scaling = make_scaling()
        assert params()["By"] == pytest.approx(scaling.b0_tesla, rel=1e-9)

    def test_resolved_slab_equals_two_ion_inertial_lengths(self):
        scaling = make_scaling()
        assert params()["heater.foil_hi"] == pytest.approx(
            2.0 * scaling.d_i_m, rel=1e-9)

    def test_density_function_gives_nt_inside_the_slab_and_namb_outside(self):
        """The one typo no scalar comparison would catch: a swapped < / >=."""
        scaling = make_scaling()
        got = params()
        for piston in ("piston_electrons", "piston_ions"):
            assert got[f"n_slab:{piston}"] == pytest.approx(
                scaling.n_target_per_m3, rel=1e-9)
            assert got[f"n_amb:{piston}"] == 0.0
        for ambient in ("amb_electrons", "amb_ions"):
            assert got[f"n_slab:{ambient}"] == 0.0
            assert got[f"n_amb:{ambient}"] == pytest.approx(
                scaling.n_amb_per_m3, rel=1e-9)

    def test_verify_flags_a_hand_edited_max_step(self, tmp_path):
        spec = make_spec()
        path = tmp_path / "inputs"
        path.write_text(render(spec).replace(
            f"max_step      = {make_scaling(spec).max_step}", "max_step      = 17"))
        drift = heater_deck.verify(spec, make_scaling(spec), path)
        assert any(w.startswith("max_step:") for w in drift)
        # ...and names the benign cause, so a shortened run is not read as deck rot.
        assert any("HEATER_EXTRA_ARGS" in w for w in drift)

    def test_verify_flags_a_hand_edited_slab_width(self, tmp_path):
        spec = make_spec()
        path = tmp_path / "inputs"
        path.write_text(render(spec).replace("2.000000*di", "3.000000*di"))
        drift = heater_deck.verify(spec, make_scaling(spec), path)
        assert any("heater.foil_hi" in w for w in drift)

    def test_verify_tolerates_the_restart_line_the_sbatch_appends(self, tmp_path):
        spec = make_spec()
        path = tmp_path / "inputs"
        path.write_text(render(spec) + "\namr.restart = diags/chk014000\n")
        assert heater_deck.verify(spec, make_scaling(spec), path) == []

    def test_verify_tolerates_warpxs_own_constants(self, tmp_path):
        """WarpX resolves lengths with CODATA-2022; ours are CODATA-2018 (~1e-9 apart)."""
        spec = make_spec()
        path = tmp_path / "inputs"
        deck = render(spec)
        scaling = make_scaling(spec)
        nudged = deck.replace(
            f"my_constants.n0    = {scaling.n0_per_m3:.6e}",
            f"my_constants.n0    = {scaling.n0_per_m3 * (1 + 1e-9):.10e}")
        path.write_text(nudged)
        assert heater_deck.verify(spec, scaling, path) == []


class TestIntervals:
    @pytest.mark.parametrize("period,stop", [(20, None), (20, 23260), (7, 1000)])
    def test_parse_intervals_inverts_intervals(self, period, stop):
        text = heater_deck._intervals(period, stop)
        got_period, got_stop = heater_deck._parse_intervals(text)
        assert got_period == period
        assert got_stop == (None if stop is None else stop - stop % period)

    def test_no_stop_time_emits_a_bare_period(self):
        assert heater_deck._intervals(20, None) == "20"

    def test_the_stop_step_is_rounded_down_to_the_period(self):
        """So the last application lands exactly on the boundary, never past it."""
        assert heater_deck._intervals(20, 1007) == "0:1000:20"


class TestDriveWindow:
    def test_an_unset_window_drives_the_whole_run(self):
        assert heater_deck.drive_stop_step(make_spec(), make_scaling()) is None
        assert params()["heater.stop_step"] is None

    def test_a_window_gates_both_operators(self):
        spec = make_spec(operators={"drive_stop_t_ci": 0.01})
        got = params(spec)
        assert got["heater.stop_step"] is not None
        assert got["injector.stop_step"] == got["heater.stop_step"]

    def test_the_window_follows_t_ci_not_steps(self):
        """Halving dx halves dt, so the same T_ci must buy about twice the steps."""
        spec = make_spec(operators={"drive_stop_t_ci": 0.01})
        coarse = params(spec)["heater.stop_step"]
        fine = params(make_spec(**{**spec, "scaling": {**spec["scaling"],
                                                       "cell_size_de": 0.25}}))
        assert fine["heater.stop_step"] == pytest.approx(2 * coarse, rel=0.01)


class TestNullControl:
    def test_the_no_heater_deck_differs_from_production_only_in_the_heater(self):
        """The null control's ONLY difference must be the energy source.

        Asserted on key_params, not on the text: the claim is about what WarpX will do,
        and a text diff would also flag the deck's own explanatory comment block.
        """
        spec = make_spec()
        production = params(spec)
        null = params(spec, no_heater=True)
        changed = {k for k in set(production) | set(null)
                   if production.get(k) != null.get(k)}
        assert all(k.startswith("heater.") for k in changed), sorted(changed)
        assert production["heater.present"] and not null["heater.present"]

    def test_the_no_heater_deck_still_declares_the_injector(self):
        """The macroparticle count and load must match, or it is not a control."""
        spec = make_spec()
        assert (params(spec, no_heater=True)["injector.ppc_reference"]
                == params(spec)["injector.ppc_reference"])

    def test_the_sbatch_heater_grep_finds_nothing_in_the_null_deck(self):
        """run_heater_2d.sbatch keys its physics-free warning off this exact string."""
        assert "particle_heater.species" not in render(no_heater=True)
        assert "particle_heater.species" in render()


class TestSmokeVariant:
    def test_the_smoke_deck_is_smaller_on_every_axis(self):
        spec = make_spec()
        full = params(spec)
        smoke = heater_deck.key_params_from_text(
            heater_deck.render_deck(spec, heater_spec.scaling(spec, smoke=True),
                                    smoke=True))
        assert smoke["max_step"] < full["max_step"]
        assert smoke["ppc:amb_ions"] < full["ppc:amb_ions"]
        assert all(s < f for s, f in zip(smoke["n_cell"], full["n_cell"]))

    def test_the_smoke_deck_writes_no_checkpoints_and_no_phase_space(self):
        deck = heater_deck.render_deck(
            make_spec(), heater_spec.scaling(make_spec(), smoke=True), smoke=True)
        assert "chk.intervals" not in deck
        assert "phase.intervals" not in deck
        assert "diagnostics.diags_names = diag1\n" in deck


class TestReport:
    def test_the_report_step_count_matches_the_decks_max_step(self):
        spec = make_spec()
        scaling = make_scaling(spec)
        report = heater_deck.scaling_report(spec, scaling, ppc_each_dim=(5, 5),
                                            max_step=scaling.max_step)
        assert f"{scaling.max_step} steps" in report

    def test_the_report_carries_the_invariance_table(self):
        spec = make_spec()
        scaling = make_scaling(spec)
        report = heater_deck.scaling_report(spec, scaling, ppc_each_dim=(5, 5),
                                            max_step=scaling.max_step)
        for name in scaling.invariants():
            assert name in report

    def test_the_report_names_the_delivered_cell_size(self):
        """Not the requested one: the rounded cell count is what sets WarpX's dt."""
        spec = make_spec()
        scaling = make_scaling(spec)
        report = heater_deck.scaling_report(spec, scaling, ppc_each_dim=(5, 5),
                                            max_step=scaling.max_step)
        assert f"{scaling.cell_size_de_actual:.6g} d_e" in report


class TestTheCheckedInSpec:
    """The real spec, which the unit specs above deliberately simplify away from."""

    def test_it_loads_derives_renders_and_verifies(self, tmp_path):
        spec = heater_spec.load(SPEC_PATH)
        scaling = heater_spec.scaling(spec)
        path = tmp_path / "inputs_2d_heater"
        path.write_text(heater_deck.render_deck(spec, scaling))
        assert heater_deck.verify(spec, scaling, path) == []

    def test_the_deck_is_fully_periodic_on_both_axes(self):
        got = heater_deck.key_params_from_text(
            heater_deck.render_deck(heater_spec.load(SPEC_PATH),
                                    heater_spec.scaling(heater_spec.load(SPEC_PATH))))
        for key in ("boundary.field_lo", "boundary.field_hi",
                    "boundary.particle_lo", "boundary.particle_hi"):
            assert got[key] == "periodic periodic"

    def test_it_still_means_what_the_running_jobs_deck_means(self):
        """Commit-A gate: the restructure must not move a number the live run needs.

        Skipped when input_files/ has been cleaned (it is gitignored and regenerable);
        when present it is the strongest available check that the move preserved meaning.
        """
        on_disk = os.path.join(_REPO, "input_files", "warpx", "magshockz_2d_heater",
                               "inputs_2d_heater")
        if not os.path.isfile(on_disk):
            pytest.skip("input_files/warpx/magshockz_2d_heater is not populated")
        spec = heater_spec.load(SPEC_PATH)
        rendered = heater_deck.key_params_from_text(
            heater_deck.render_deck(spec, heater_spec.scaling(spec)))
        existing = heater_deck.key_params(on_disk)
        # A checkpoint is only resumable by a deck with the same grid AND the same
        # domain, and run_heater_2d.sbatch only checks the former -- so these four are
        # asserted EXACTLY, not within a tolerance.
        for key in ("n_cell", "prob_lo", "prob_hi", "max_step"):
            assert rendered[key] == existing[key], key
        assert heater_deck.verify(spec, heater_spec.scaling(spec), on_disk) == []
