"""Tests for the WarpX heater-run measurement layer.

Two modules, one contract each: ``metrics`` turns a dump series into the invariant
scorecard, and ``analysis.warpx.flash`` reports what FLASH measured at its own EOS
ionization state -- deliberately not the integer charge state the deck imposes.
"""

from __future__ import annotations

import math
from pathlib import Path

import astropy.units as u
import numpy as np
import pytest
from astropy.constants import c

from magshockz.analysis.warpx import flash as warpx_flash
from magshockz.analysis.warpx import metrics
from magshockz.init.warpx import config as spec_config
from magshockz.init.warpx import units

REPO = Path(__file__).resolve().parents[1]
SPEC_PATH = REPO / "runs" / "magshockz_2d_heater.warpx.yaml"


@pytest.fixture(scope="module")
def scales() -> units.DeckScales:
    return spec_config.scales(spec_config.load(SPEC_PATH))


class TestFrontSpeed:
    def test_a_straight_front_recovers_its_slope(self):
        times = np.linspace(0.0, 1000.0, 11)
        # d_e per 1/omega_pe is exactly c, so the slope IS v/c.
        speed = metrics.front_speed_over_c(times, 3.0 + 0.02 * times, di_over_de=10.0)
        assert speed == pytest.approx(0.02, rel=1e-9)

    def test_a_front_that_has_barely_moved_is_refused(self):
        """Below one d_i the fitted slope is line-out quantisation, not a speed."""
        times = np.linspace(0.0, 1000.0, 11)
        travel_di = 0.5 * metrics.MIN_TRAVEL_DI
        fronts = np.linspace(0.0, travel_di * 10.0, times.size)
        assert math.isnan(metrics.front_speed_over_c(times, fronts, di_over_de=10.0))

    def test_a_single_tracked_dump_gives_no_speed(self):
        assert math.isnan(metrics.front_speed_over_c(
            np.array([0.0, 1.0]), np.array([1.0, np.nan]), di_over_de=1.0))

    def test_nan_dumps_are_dropped_rather_than_poisoning_the_fit(self):
        times = np.linspace(0.0, 1000.0, 11)
        fronts = 0.02 * times
        fronts[3] = np.nan
        assert metrics.front_speed_over_c(
            times, fronts, di_over_de=10.0) == pytest.approx(0.02, rel=1e-9)


class TestScorecard:
    def test_a_run_that_hits_its_deck_aim_scores_the_deck_column(self, scales):
        rows = metrics.scorecard(scales,
                                 measured_speed_over_c=scales.piston_speed_over_c,
                                 measured_contrast=scales.contrast)
        for row in rows:
            assert row.warpx == pytest.approx(row.deck, rel=1e-9)

    def test_the_flash_column_is_the_flash_reference_not_the_deck(self, scales):
        rows = {row.quantity: row for row in
                metrics.scorecard(scales, measured_speed_over_c=0.02,
                                  measured_contrast=4.0)}
        assert rows["M_A"].flash == pytest.approx(scales.flash.mach_alfven)
        assert rows["n_piston / n_amb"].flash == pytest.approx(scales.flash.contrast)

    def test_the_matched_rows_agree_between_codes_and_v_over_c_does_not(self, scales):
        """The whole point of the reduced-mass map: M_A matches, v/c is 10x off."""
        rows = {row.quantity: row for row in
                metrics.scorecard(scales, measured_speed_over_c=0.02,
                                  measured_contrast=4.0)}
        assert rows["M_A"].flash == pytest.approx(rows["M_A"].deck, rel=1e-6)
        assert rows["M_A"].is_matched

        speed = rows["v_piston / c"]
        assert not speed.is_matched
        assert speed.deck / speed.flash > 5.0
        assert "NOT matched" in speed.label

    def test_the_front_row_is_matched_in_di_per_gyroperiod(self, scales):
        """Same speed, different absolute units -- the matched form must agree."""
        rows = {row.quantity: row for row in
                metrics.scorecard(scales, measured_speed_over_c=0.02,
                                  measured_contrast=4.0)}
        front = rows["front [d_i/T_ci]"]
        assert front.flash == pytest.approx(front.deck, rel=1e-2)

    def test_a_speedless_run_leaves_the_speed_rows_nan_and_keeps_the_contrast(self, scales):
        rows = {row.quantity: row for row in
                metrics.scorecard(scales, measured_speed_over_c=float("nan"),
                                  measured_contrast=4.0)}
        assert math.isnan(rows["M_A"].warpx)
        assert rows["n_piston / n_amb"].warpx == 4.0

    def test_it_refuses_a_deck_with_no_flash_reference(self, scales):
        import dataclasses
        with pytest.raises(ValueError, match="FLASH"):
            metrics.scorecard(dataclasses.replace(scales, flash=None),
                              measured_speed_over_c=0.02, measured_contrast=4.0)

    def test_the_table_carries_every_row_and_the_three_columns(self, scales):
        rows = metrics.scorecard(scales, measured_speed_over_c=0.02,
                                 measured_contrast=4.0)
        text = metrics.scorecard_text(rows).splitlines()
        assert all(column in text[0] for column in ("FLASH", "deck aim", "WarpX"))
        assert len(text) == len(rows) + 1


class TestInvariantTable:
    def test_it_reports_both_the_matched_and_the_broken_scales(self, scales):
        text = units.invariant_table(scales)
        for name in scales.invariants():
            assert name in text
        assert "m_i/(Z m_e)" in text and "deliberately broken" in text

    def test_it_refuses_a_deck_with_no_flash_reference(self, scales):
        import dataclasses
        with pytest.raises(ValueError, match="FLASH"):
            units.invariant_table(dataclasses.replace(scales, flash=None))


class TestMeasuredPiston:
    """The FLASH side, at the EOS ionization state rather than the deck's integer Z."""

    @pytest.fixture
    def measured(self) -> warpx_flash.MeasuredPiston:
        return warpx_flash.MeasuredPiston(
            upstream=units.Upstream(
                ion=warpx_flash.eos_ion(mass_number=26.98, charge_state=3.66),
                electron_density=3.04e24 * u.m**-3,
                magnetic_field=7.0 * u.T,
                electron_temperature=9.83 * u.eV,
                ion_temperature=9.83 * u.eV),
            piston_electron_density=2.08e25 * u.m**-3,
            front_speed=768.5 * u.km / u.s,
            piston_scale_length=300.0 * u.um,
            spot_radius=500.0 * u.um,
            window=8.75 * u.ns)

    def test_a_fractional_zbar_is_a_usable_ion(self):
        """``Particle('Al 3.66+')`` raises, so the EOS state needs a CustomParticle."""
        ion = warpx_flash.eos_ion(mass_number=26.98, charge_state=3.66)
        assert ion.charge_number == pytest.approx(3.66)
        assert ion.mass.to_value(u.u) == pytest.approx(26.98)

    def test_it_reproduces_the_datasets_published_numbers(self, measured):
        assert measured.mach_alfven == pytest.approx(23.7, rel=0.02)
        assert measured.mach_magnetosonic == pytest.approx(21.9, rel=0.02)
        assert measured.upstream.beta_e == pytest.approx(0.245, rel=0.02)
        assert measured.upstream.ion_skin_depth.to_value(u.um) == pytest.approx(
            354.0, rel=0.02)
        assert measured.upstream.gyroperiod.to_value(u.ns) == pytest.approx(69.0, rel=0.02)

    def test_the_shock_is_supercritical_well_above_the_reflection_threshold(self, measured):
        assert measured.is_supercritical
        assert measured.mach_magnetosonic > 5 * warpx_flash.ION_REFLECTION_MACH

    def test_contrast_is_the_electron_density_ratio(self, measured):
        assert measured.contrast == pytest.approx(2.08e25 / 3.04e24)

    def test_the_window_is_a_small_fraction_of_a_gyroperiod(self, measured):
        assert measured.window_gyroperiods == pytest.approx(8.75 / 69.0, rel=0.02)

    def test_every_matched_invariant_is_reported(self, measured):
        invariants = measured.invariants()
        assert set(invariants) == {"M_A", "M_ms", "beta_e", "beta_i", "contrast",
                                   "r_spot/d_i", "t_window/T_ci"}
        assert all(np.isfinite(list(invariants.values())))

    def test_the_deck_reference_and_the_measurement_disagree_on_purpose(self, measured, scales):
        """``Al 6+`` is imposed by the spec; FLASH's EOS ran at Zbar 3.66.

        Same element, same mass density -- so the deck's ambient ``n_e`` is higher by the
        charge-state ratio, and that is a choice, not a discrepancy to reconcile.
        """
        deck_ion = scales.flash.upstream.ion
        assert deck_ion.charge_number > measured.upstream.ion.charge_number
