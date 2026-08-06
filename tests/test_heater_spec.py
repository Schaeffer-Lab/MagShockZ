"""Tests for heater_spec: loading, validating and freezing a heater_pic_2d run spec.

Numpy + PyYAML only, so these run in CI without yt / WarpX / astropy.  The contract
under test is the split between the two failure modes: :func:`load` **raises** on
anything the generator cannot render, and :func:`validate` **warns** on everything else,
because a deliberately off-target deck (the null control, a resolution probe, a
frame-consistency run) is a legitimate thing to want.
"""

import os

import pytest
import yaml

import heater_piston_scaling as hps
import heater_spec
from test_config_yaml_scalars import _numeric_strings
from test_heater_deck import SPEC_PATH, make_spec

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def write_spec(tmp_path, spec: dict, name: str = "spec.warpx.yaml"):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(spec, sort_keys=False))
    return path


class TestLoad:
    def test_the_checked_in_spec_loads_and_derives(self):
        spec = heater_spec.load(SPEC_PATH)
        scaling = heater_spec.scaling(spec)
        assert spec["_spec_path"] == os.path.abspath(SPEC_PATH)
        assert scaling.max_step > 0

    def test_a_flash2warpx_spec_is_rejected_by_schema(self, tmp_path):
        """The real dispatch failure: the other runs/*.warpx.yaml are a different animal."""
        path = write_spec(tmp_path, {"extract": {}, "run": {"ppc": 16}})
        with pytest.raises(ValueError, match="heater_pic_2d"):
            heater_spec.load(path)

    @pytest.mark.parametrize("block", heater_spec.REQUIRED_BLOCKS)
    def test_a_missing_block_raises_naming_the_block(self, tmp_path, block):
        spec = make_spec()
        del spec[block]
        with pytest.raises(ValueError, match=block):
            heater_spec.load(write_spec(tmp_path, spec))

    def test_a_non_periodic_boundary_raises(self, tmp_path):
        """A uniform applied E/B requires periodic boundaries; this is not a preference."""
        spec = make_spec(geometry={"boundary": {"lo": "reflecting", "hi": "open"}})
        with pytest.raises(ValueError, match="parameters.rst"):
            heater_spec.load(write_spec(tmp_path, spec))

    def test_a_backwards_time_window_raises(self, tmp_path):
        spec = make_spec(flash_target={"t_window_ns": [9.0, 3.0]})
        with pytest.raises(ValueError, match="increasing"):
            heater_spec.load(write_spec(tmp_path, spec))

    def test_a_one_dimensional_ppc_raises(self, tmp_path):
        spec = make_spec(scaling={"ppc_each_dim": [5]})
        with pytest.raises(ValueError, match="two positive ints"):
            heater_spec.load(write_spec(tmp_path, spec))


class TestValidate:
    def test_a_coarse_grid_warns_about_the_debye_length(self):
        spec = make_spec(scaling={"cell_size_de": 2.0})
        assert any("lambda_De/dx" in w for w in heater_spec.validate(spec))

    def test_refining_the_grid_clears_the_debye_warning(self):
        """The dx derive()'s warning suggests has to actually work."""
        coarse = make_spec(scaling={"cell_size_de": 2.0})
        warning = next(w for w in heater_spec.validate(coarse) if "lambda_De/dx" in w)
        suggested = float(warning.split("(to ")[1].split(" ")[0])
        refined = make_spec(scaling={"cell_size_de": suggested})
        assert not any("lambda_De/dx" in w for w in heater_spec.validate(refined))

    def test_the_debye_warning_is_not_reported_twice(self):
        """derive() owns the default threshold; validate must not restate it."""
        warnings = heater_spec.validate(make_spec(scaling={"cell_size_de": 2.0}))
        assert len([w for w in warnings if "lambda_De/dx" in w]) == 1

    def test_a_stricter_spec_threshold_is_enforced_on_top_of_the_default(self):
        spec = make_spec(scaling={"cell_size_de": 0.12},
                         targets={"acceptance": {"debye_per_cell_min": 0.5}})
        assert any("targets.acceptance.debye_per_cell_min" in w
                   for w in heater_spec.validate(spec))

    def test_a_short_z_domain_warns_that_the_front_wraps(self):
        spec = make_spec(scaling={"domain_z_halfwidth_de": 5.0})
        assert any("wrap" in w for w in heater_spec.validate(spec))

    def test_a_narrow_x_domain_warns_about_overlapping_spot_images(self):
        spec = make_spec(scaling={"domain_x_halfwidth_de": 4.0})
        assert any("spot" in w for w in heater_spec.validate(spec))

    def test_too_few_plotfiles_warns(self):
        spec = make_spec(diagnostics={"plotfile_intervals": 10**9})
        assert any("plotfile_intervals" in w for w in heater_spec.validate(spec))

    def test_too_many_plotfiles_warns(self):
        spec = make_spec(diagnostics={"plotfile_intervals": 1})
        assert any("plotfile_intervals" in w for w in heater_spec.validate(spec))

    def test_a_sensible_cadence_does_not_warn(self):
        scaling = heater_spec.scaling(make_spec())
        spec = make_spec(diagnostics={"plotfile_intervals": scaling.max_step // 50})
        assert not any("plotfile_intervals" in w for w in heater_spec.validate(spec))

    def test_missing_checkpoints_warn(self):
        spec = make_spec(diagnostics={"checkpoint_intervals": 0})
        assert any("resume" in w for w in heater_spec.validate(spec))

    def test_no_drive_window_warns_that_flashs_laser_is_finite(self):
        assert any("drive_stop_t_ci" in w for w in heater_spec.validate(make_spec()))

    def test_a_drive_window_longer_than_the_run_warns(self):
        spec = make_spec(operators={"drive_stop_t_ci": 99.0})
        assert any("never fires" in w for w in heater_spec.validate(spec))

    def test_a_drive_window_shorter_than_one_heater_period_warns(self):
        spec = make_spec(operators={"drive_stop_t_ci": 1.0e-9})
        assert any("never be driven" in w for w in heater_spec.validate(spec))

    def test_a_good_drive_window_does_not_warn(self):
        spec = make_spec(operators={"drive_stop_t_ci": 0.01})
        assert not any("drive_stop_t_ci" in w for w in heater_spec.validate(spec))

    def test_the_invariants_block_is_checked_against_derive(self):
        """Matched by construction, so a mismatch means derive() regressed."""
        spec = make_spec(targets={"invariants": {"M_A": 1.0}})
        assert any("targets.invariants.M_A" in w for w in heater_spec.validate(spec))

    def test_the_checked_in_specs_invariants_still_hold(self):
        spec = heater_spec.load(SPEC_PATH)
        warnings = heater_spec.validate(spec)
        assert not [w for w in warnings if "targets.invariants" in w]

    def test_a_stale_checkpoint_grid_warns_before_the_job_is_queued(self, tmp_path):
        """What run_heater_2d.sbatch only discovers after the queue wait is spent."""
        checkpoint = tmp_path / "diags" / "chk000100"
        checkpoint.mkdir(parents=True)
        (checkpoint / "WarpXHeader").write_text("")
        (checkpoint / "warpx_job_info").write_text("amr.n_cell = 568 2832\n")
        warnings = heater_spec.validate(make_spec(), out_dir=tmp_path)
        assert any("568x2832" in w and "SKIP" in w for w in warnings)

    def test_a_matching_checkpoint_grid_does_not_warn(self, tmp_path):
        scaling = heater_spec.scaling(make_spec())
        checkpoint = tmp_path / "diags" / "chk000100"
        checkpoint.mkdir(parents=True)
        (checkpoint / "WarpXHeader").write_text("")
        (checkpoint / "warpx_job_info").write_text(
            f"amr.n_cell = {scaling.n_cells_x} {scaling.n_cells_z}\n")
        assert not any("SKIP" in w
                       for w in heater_spec.validate(make_spec(), out_dir=tmp_path))

    def test_a_bulk_flow_warns_that_the_injector_has_no_drift_knob(self):
        spec = make_spec(flow={"v_flow_kms": 50.0})
        assert any("TargetInjector" in w for w in heater_spec.validate(spec))

    @pytest.mark.parametrize("overrides", [
        {"scaling": {"cell_size_de": 5.0}},
        {"scaling": {"domain_z_halfwidth_de": 1.0}},
        {"scaling": {"domain_x_halfwidth_de": 1.0}},
        {"diagnostics": {"plotfile_intervals": 1}},
        {"diagnostics": {"checkpoint_intervals": 0}},
        {"operators": {"drive_stop_t_ci": 99.0}},
        {"flow": {"v_flow_kms": 250.0}},
        {"targets": {"invariants": {"M_A": 1.0}, "acceptance": {"dt_omega_pe_max": 1e-9}}},
    ])
    def test_validate_never_raises_on_bad_physics(self, overrides):
        """An off-target deck is a legitimate thing to want; refusing it would be wrong."""
        warnings = heater_spec.validate(make_spec(**overrides))
        assert isinstance(warnings, list) and warnings

    def test_warnings_are_deduplicated(self):
        warnings = heater_spec.validate(make_spec(scaling={"cell_size_de": 5.0}))
        assert len(warnings) == len(set(warnings))


class TestFreeze:
    def test_the_frozen_spec_records_the_derived_invariants(self):
        spec = heater_spec.load(SPEC_PATH)
        scaling = heater_spec.scaling(spec)
        frozen = heater_spec.freeze(spec, scaling)
        assert frozen["derived"]["invariants_deck"] == scaling.invariants()
        assert frozen["derived"]["invariants_flash"] == scaling.targets.invariants()

    def test_the_frozen_spec_drops_the_private_path_key(self):
        spec = heater_spec.load(SPEC_PATH)
        assert "_spec_path" not in heater_spec.freeze(spec, heater_spec.scaling(spec))

    def test_the_frozen_spec_carries_provenance_when_given(self):
        spec = make_spec()
        frozen = heater_spec.freeze(spec, heater_spec.scaling(spec),
                                    provenance={"git": "abc123"})
        assert frozen["provenance"]["git"] == "abc123"

    def test_the_frozen_spec_is_yaml_1_1_float_safe(self):
        """The repo's YAML-1.1 rule applies to GENERATED yaml too, which nothing else checks.

        PyYAML resolves ``1e18`` as a *string*, so a frozen spec that round-trips a
        numeric-looking string would hand the next reader a str where it expects a float.
        """
        spec = heater_spec.load(SPEC_PATH)
        frozen = heater_spec.freeze(spec, heater_spec.scaling(spec))
        reloaded = yaml.safe_load(yaml.safe_dump(frozen, sort_keys=False))
        assert _numeric_strings(reloaded) == []


class TestScalingBridge:
    def test_the_smoke_domain_is_the_configured_fraction_of_the_physical_one(self):
        spec = make_spec(scaling={"domain_x_halfwidth_de": None,
                                  "domain_z_halfwidth_de": None})
        full = heater_spec.scaling(spec)
        smoke = heater_spec.scaling(spec, smoke=True)
        assert smoke.domain_z_halfwidth_de == pytest.approx(
            0.25 * full.domain_z_halfwidth_de, rel=1e-12)

    def test_the_blocking_factor_comes_from_the_spec(self):
        spec = make_spec(numerics={"blocking_factor": 64})
        scaling = heater_spec.scaling(spec)
        assert scaling.n_cells_x % 64 == 0 and scaling.n_cells_z % 64 == 0

    def test_targets_carry_the_specs_composition(self):
        targets = heater_spec.targets(heater_spec.load(SPEC_PATH))
        assert targets.z_amb == pytest.approx(3.66402)
        assert targets.a_piston == pytest.approx(28.0855)

    def test_steps_per_gyroperiod_matches_dt_and_the_gyroperiod(self):
        scaling = heater_spec.scaling(make_spec())
        assert scaling.steps_per_gyroperiod == pytest.approx(
            scaling.gyroperiod_s / scaling.dt_s, rel=1e-12)
        assert scaling.max_step == pytest.approx(
            scaling.t_run_gyro * scaling.steps_per_gyroperiod, rel=1e-4)

    def test_the_delivered_cell_size_is_reported_not_assumed(self):
        """n_cells is rounded UP to the blocking factor while the halfwidth is not, so
        the delivered dx is smaller than the request -- and it is the delivered one that
        sets WarpX's dt.  Commit B removes the gap; until then it must be visible."""
        scaling = heater_spec.scaling(heater_spec.load(SPEC_PATH))
        assert scaling.cell_size_de_actual <= scaling.cell_size_de
        assert scaling.cell_size_de_actual == pytest.approx(
            2.0 * scaling.domain_x_halfwidth_de / scaling.n_cells_x, rel=1e-12)


def test_plotfile_count_range_is_the_module_default():
    assert hps.PLOTFILE_COUNT_RANGE == (10, 200)
