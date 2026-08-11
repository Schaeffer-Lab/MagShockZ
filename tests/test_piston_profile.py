"""Tests for piston_profile: front location, trajectory, e-folding length, collapse.

Numpy-only.  The profiles here are synthetic, with the answer known analytically, so a
failure points at the estimator rather than at a FLASH dump.
"""

import numpy as np
import pytest

from magshockz.common import piston_profile as pp

AMU_G = 1.66053906660e-24


def exponential_piston(positions: np.ndarray, x_front: float, scale_length: float,
                       peak: float = 1.0, floor: float = 0.0) -> np.ndarray:
    """Density that is flat at ``peak`` inside the piston and decays outward from
    ``x_front - scale_length`` with e-folding length ``scale_length``."""
    knee = x_front - scale_length
    decaying = peak * np.exp(-(positions - knee) / scale_length)
    return np.where(positions <= knee, peak, decaying) + floor


class TestPistonIonDensity:
    def test_exact_formula(self):
        rho, fraction, mass_number = 1.0e-3, 0.5, 28.0855
        expected = rho * fraction / (mass_number * AMU_G)
        assert pp.piston_ion_density(rho, fraction, mass_number,
                                     AMU_G) == pytest.approx(expected, rel=1e-12)

    def test_is_elementwise_and_linear(self):
        rho = np.array([1.0, 2.0, 4.0]) * 1e-3
        fraction = np.array([1.0, 1.0, 0.5])
        out = pp.piston_ion_density(rho, fraction, 28.0855, AMU_G)
        assert out.shape == rho.shape
        assert out[1] == pytest.approx(2.0 * out[0], rel=1e-12)
        assert out[2] == pytest.approx(2.0 * out[0], rel=1e-12)

    def test_zero_material_fraction_gives_zero(self):
        out = pp.piston_ion_density(np.array([1e-3, 1e-3]), np.array([0.0, 1.0]),
                                    28.0855, AMU_G)
        assert out[0] == 0.0 and out[1] > 0.0


class TestFrontPosition:
    def test_locates_a_step_at_the_threshold(self):
        positions = np.linspace(0.0, 10.0, 1001)
        # Flat 1.0 up to 5, then a clean linear ramp down to 0 at 6.
        density = np.clip(np.where(positions <= 5.0, 1.0, 6.0 - positions), 0.0, 1.0)
        assert pp.front_position(positions, density, 0.5) == pytest.approx(5.5, abs=1e-2)
        assert pp.front_position(positions, density, 0.1) == pytest.approx(5.9, abs=1e-2)

    def test_interpolates_between_samples(self):
        # Coarse grid: the answer must not be quantised to it.
        positions = np.array([0.0, 1.0, 2.0])
        density = np.array([1.0, 1.0, 0.0])
        assert pp.front_position(positions, density, 0.5) == pytest.approx(1.5)

    def test_takes_the_outermost_crossing_of_a_double_peaked_piston(self):
        positions = np.linspace(0.0, 10.0, 1001)
        density = (np.exp(-((positions - 2.0) / 0.3) ** 2)
                   + 0.8 * np.exp(-((positions - 6.0) / 0.3) ** 2))
        front = pp.front_position(positions, density, 0.1)
        assert front > 6.0        # tracks the leading blob, not the denser inner one

    def test_reports_the_edge_when_the_piston_fills_the_window(self):
        positions = np.linspace(0.0, 10.0, 101)
        assert pp.front_position(positions, np.ones_like(positions),
                                 0.1) == pytest.approx(10.0)

    @pytest.mark.parametrize("density", [
        np.zeros(50),
        np.full(50, np.nan),
        -np.ones(50),
    ])
    def test_degenerate_density_is_nan(self, density):
        assert np.isnan(pp.front_position(np.linspace(0, 10, 50), density))

    def test_mismatched_or_tiny_input_is_nan(self):
        assert np.isnan(pp.front_position(np.array([0.0, 1.0]), np.array([1.0])))
        assert np.isnan(pp.front_position(np.array([0.0]), np.array([1.0])))


class TestAmbientReferenceLevel:
    def test_median_of_the_outer_band(self):
        positions = np.linspace(0.0, 10.0, 1001)
        density = np.where(positions < 5.0, 100.0, 2.0)
        assert pp.ambient_reference_level(positions, density,
                                         0.2) == pytest.approx(2.0)

    def test_is_robust_to_a_far_field_spike(self):
        positions = np.linspace(0.0, 10.0, 101)
        density = np.full(101, 2.0)
        density[-3] = 1.0e6
        assert pp.ambient_reference_level(positions, density,
                                         0.3) == pytest.approx(2.0)

    def test_degenerate_input_is_nan(self):
        assert np.isnan(pp.ambient_reference_level(np.array([0.0]), np.array([1.0])))
        assert np.isnan(pp.ambient_reference_level(np.linspace(0, 1, 10),
                                                  np.full(10, np.nan)))
        assert np.isnan(pp.ambient_reference_level(np.linspace(0, 1, 10),
                                                  np.ones(10), outer_fraction=0.0))


class TestAbsoluteFrontLevel:
    def test_absolute_level_is_used_when_given(self):
        positions = np.linspace(0.0, 10.0, 1001)
        density = np.clip(np.where(positions <= 5.0, 1.0, 6.0 - positions), 0.0, 1.0)
        assert pp.front_position(positions, density,
                                level=0.5) == pytest.approx(5.5, abs=1e-2)

    def test_absolute_level_survives_a_migrating_peak(self):
        # The regression this exists for: a fast tenuous leading edge plus a dense inner
        # plume that overtakes it in amplitude. A peak-relative threshold jumps backwards
        # to the plume; an absolute one keeps tracking the leading edge.
        positions = np.linspace(0.0, 10.0, 2001)
        ambient = 1.0
        leading_edge = 20.0 * np.exp(-positions / 1.0)
        early = leading_edge + 5.0 * np.exp(-((positions - 1.0) / 0.5) ** 2)
        late = leading_edge + 5000.0 * np.exp(-((positions - 1.0) / 0.5) ** 2)

        relative_early = pp.front_position(positions, early, 0.1)
        relative_late = pp.front_position(positions, late, 0.1)
        assert relative_late < relative_early        # the bug: front moves backwards

        # The inner hump's own tail still reaches the front, so the absolute front moves
        # by ~0.02% rather than not at all; the point is that it does not JUMP BACKWARDS
        # by a factor of a few, as the peak-relative one does.
        absolute_early = pp.front_position(positions, early, level=ambient)
        absolute_late = pp.front_position(positions, late, level=ambient)
        assert absolute_late == pytest.approx(absolute_early, rel=1e-3)
        assert absolute_late >= absolute_early

    @pytest.mark.parametrize("level", [0.0, -1.0, np.nan, np.inf])
    def test_nonpositive_or_nonfinite_level_is_nan(self, level):
        positions = np.linspace(0.0, 10.0, 101)
        assert np.isnan(pp.front_position(positions, np.exp(-positions), level=level))


class TestUnperturbedAverage:
    def test_masks_out_contaminated_cells(self):
        values = np.array([1.0, 1.0, 999.0, 999.0])
        contaminant = np.array([0.0, 0.0, 1.0, 0.5])
        assert pp.unperturbed_average(values, contaminant) == pytest.approx(1.0)

    def test_uniform_pristine_background_returns_that_value(self):
        # The t=0 FLASH dump case: uniform chamber fill, zero target fraction.
        values = np.full(200, 3.04e18)
        assert pp.unperturbed_average(values, np.zeros(200)) == pytest.approx(3.04e18)

    def test_threshold_is_honoured(self):
        values = np.array([1.0, 5.0])
        contaminant = np.array([0.0, 1.0e-8])
        assert pp.unperturbed_average(values, contaminant,
                                      max_contaminant=1e-6) == pytest.approx(3.0)
        assert pp.unperturbed_average(values, contaminant,
                                      max_contaminant=1e-12) == pytest.approx(1.0)

    def test_all_contaminated_is_nan(self):
        assert np.isnan(pp.unperturbed_average(np.ones(10), np.ones(10)))

    def test_ignores_nan_values(self):
        values = np.array([1.0, np.nan, 3.0])
        assert pp.unperturbed_average(values, np.zeros(3)) == pytest.approx(2.0)


class TestUpstreamIsPristine:
    def test_true_when_the_band_is_ambient(self):
        positions = np.linspace(0.0, 10.0, 1001)
        piston = np.where(positions < 5.0, 100.0, 1e-3)
        total = np.where(positions < 5.0, 100.0, 1.0)
        assert pp.upstream_is_pristine(positions, piston, total, 4.0,
                                       offset=2.0, width=1.0)

    def test_false_when_the_cavity_has_swallowed_the_window(self):
        # Piston material everywhere: exactly the late-time MagShockZ line-out that
        # produced beta_e = 163 before this guard existed.
        positions = np.linspace(0.0, 10.0, 1001)
        density = np.full_like(positions, 50.0)
        assert not pp.upstream_is_pristine(positions, density, density, 1.0,
                                           offset=2.0, width=1.0)

    def test_false_when_the_band_is_outside_the_lineout(self):
        positions = np.linspace(0.0, 10.0, 101)
        ones = np.ones_like(positions)
        assert not pp.upstream_is_pristine(positions, ones * 1e-6, ones, 9.9,
                                           offset=5.0, width=1.0)

    def test_false_for_a_nan_front(self):
        positions = np.linspace(0.0, 10.0, 101)
        ones = np.ones_like(positions)
        assert not pp.upstream_is_pristine(positions, ones * 1e-6, ones, np.nan,
                                           offset=1.0, width=1.0)

    def test_threshold_is_honoured(self):
        positions = np.linspace(0.0, 10.0, 1001)
        total = np.ones_like(positions)
        piston = np.full_like(positions, 0.2)
        assert not pp.upstream_is_pristine(positions, piston, total, 1.0, 2.0, 1.0,
                                           contamination_max=0.1)
        assert pp.upstream_is_pristine(positions, piston, total, 1.0, 2.0, 1.0,
                                       contamination_max=0.5)


class TestEfoldingLength:
    @pytest.mark.parametrize("scale_length", [0.2, 0.5, 1.0, 2.0])
    def test_recovers_a_known_exponential(self, scale_length):
        positions = np.linspace(0.0, 20.0, 4001)
        x_front = 10.0
        density = exponential_piston(positions, x_front, scale_length)
        # Fit only inside the decaying part, i.e. inward from the front but outward of
        # the flat top at x_front - scale_length.
        got = pp.efolding_length(positions, density, x_front,
                                 window=0.9 * scale_length)
        assert got == pytest.approx(scale_length, rel=1e-6)

    def test_is_independent_of_the_peak_amplitude(self):
        positions = np.linspace(0.0, 20.0, 4001)
        a = pp.efolding_length(positions, exponential_piston(positions, 10.0, 1.0,
                                                            peak=1.0), 10.0, 0.9)
        b = pp.efolding_length(positions, exponential_piston(positions, 10.0, 1.0,
                                                            peak=1e5), 10.0, 0.9)
        assert a == pytest.approx(b, rel=1e-9)

    def test_rising_profile_refuses_to_name_a_length(self):
        positions = np.linspace(0.0, 10.0, 201)
        assert np.isnan(pp.efolding_length(positions, np.exp(positions), 10.0, 5.0))

    def test_too_few_usable_points_is_nan(self):
        positions = np.linspace(0.0, 10.0, 201)
        density = np.exp(-positions)
        assert np.isnan(pp.efolding_length(positions, density, 10.0, window=0.01))
        # Non-positive densities are unusable for a log fit.
        assert np.isnan(pp.efolding_length(positions, np.zeros_like(positions),
                                           10.0, 5.0))

    def test_bad_front_or_window_is_nan(self):
        positions = np.linspace(0.0, 10.0, 201)
        density = np.exp(-positions)
        assert np.isnan(pp.efolding_length(positions, density, np.nan, 5.0))
        assert np.isnan(pp.efolding_length(positions, density, 5.0, 0.0))


class TestFrontTrajectory:
    def test_recovers_a_known_line(self):
        times = np.linspace(3e-9, 12e-9, 10)
        speed, x_start = 9.0e5 * 100.0, 0.2      # cm/s, cm
        positions = x_start + speed * (times - times[0])
        fit = pp.fit_front_trajectory(times, positions)
        assert fit.speed == pytest.approx(speed, rel=1e-9)
        assert fit.x0 == pytest.approx(x_start, rel=1e-9)
        assert fit.t0 == pytest.approx(times[0])
        assert fit.residual_rms == pytest.approx(0.0, abs=1e-12)
        assert fit.n_points == 10

    def test_anchors_x0_at_the_window_start_not_at_zero_time(self):
        # A piston that forms mid-run must be fitted from its formation time; a
        # back-extrapolated x0 would be meaningless.
        times = np.array([5e-9, 6e-9, 7e-9])
        positions = np.array([0.1, 0.2, 0.3])
        fit = pp.fit_front_trajectory(times, positions)
        assert fit.t0 == pytest.approx(5e-9)
        assert fit.x0 == pytest.approx(0.1, rel=1e-9)
        assert fit.at(5e-9) == pytest.approx(0.1, rel=1e-9)
        assert fit.at(7e-9) == pytest.approx(0.3, rel=1e-9)

    def test_ignores_nan_samples(self):
        times = np.array([1.0, 2.0, 3.0, 4.0])
        positions = np.array([1.0, np.nan, 3.0, 4.0])
        fit = pp.fit_front_trajectory(times, positions)
        assert fit.n_points == 3
        assert fit.speed == pytest.approx(1.0, rel=1e-9)

    def test_reports_scatter_in_the_residual(self):
        times = np.linspace(0.0, 10.0, 11)
        positions = times.copy()
        positions[5] += 1.0
        fit = pp.fit_front_trajectory(times, positions)
        assert fit.residual_rms > 0.1

    def test_at_is_vectorised(self):
        fit = pp.fit_front_trajectory(np.array([0.0, 1.0]), np.array([0.0, 2.0]))
        got = fit.at(np.array([0.0, 0.5, 1.0]))
        assert got.shape == (3,)
        assert got[1] == pytest.approx(1.0)

    @pytest.mark.parametrize("times,positions", [
        (np.array([1.0]), np.array([1.0])),
        (np.array([1.0, 2.0]), np.array([np.nan, np.nan])),
    ])
    def test_too_few_points_raises(self, times, positions):
        with pytest.raises(ValueError, match="at least 2"):
            pp.fit_front_trajectory(times, positions)


class TestAheadOfFrontAverage:
    def test_averages_only_inside_the_band(self):
        positions = np.linspace(0.0, 10.0, 1001)
        values = np.where(positions < 6.0, 100.0, 1.0)
        # Band [6.5, 7.5] is entirely in the ambient.
        assert pp.ahead_of_front_average(positions, values, 5.0, offset=1.5,
                                        width=1.0) == pytest.approx(1.0)

    def test_offset_skips_the_pile_up_next_to_the_front(self):
        positions = np.linspace(0.0, 10.0, 1001)
        # Compressed shell just ahead of the front at 5, ambient beyond 5.5.
        values = np.where((positions >= 5.0) & (positions < 5.5), 10.0, 1.0)
        no_offset = pp.ahead_of_front_average(positions, values, 5.0, 0.0, 1.0)
        offset = pp.ahead_of_front_average(positions, values, 5.0, 1.0, 1.0)
        assert no_offset > 2.0            # contaminated by the shell
        assert offset == pytest.approx(1.0)

    def test_band_outside_the_lineout_is_nan(self):
        positions = np.linspace(0.0, 10.0, 101)
        values = np.ones_like(positions)
        assert np.isnan(pp.ahead_of_front_average(positions, values, 9.9,
                                                 offset=5.0, width=1.0))

    def test_nan_front_is_nan(self):
        positions = np.linspace(0.0, 10.0, 101)
        assert np.isnan(pp.ahead_of_front_average(positions, np.ones(101), np.nan,
                                                 1.0, 1.0))

    def test_ignores_nan_values_inside_the_band(self):
        positions = np.linspace(0.0, 10.0, 11)
        values = np.ones(11)
        values[7] = np.nan
        assert pp.ahead_of_front_average(positions, values, 5.0, 1.0,
                                        3.0) == pytest.approx(1.0)


class TestCollapseProfile:
    def test_two_dumps_of_a_self_similar_piston_collapse(self):
        positions = np.linspace(0.0, 40.0, 4001)
        early = exponential_piston(positions, 10.0, 1.0, peak=1.0)
        late = exponential_piston(positions, 20.0, 2.0, peak=0.25)
        xi_a, n_a = pp.collapse_profile(positions, early, 10.0, 1.0)
        xi_b, n_b = pp.collapse_profile(positions, late, 20.0, 2.0)
        # Sample both collapsed curves on a common xi and compare.
        xi_probe = np.linspace(-2.0, 0.0, 21)
        a = np.interp(xi_probe, xi_a, n_a)
        b = np.interp(xi_probe, xi_b, n_b)
        assert np.allclose(a, b, rtol=1e-6, atol=1e-6)

    def test_front_maps_to_xi_minus_one_for_this_shape(self):
        positions = np.linspace(0.0, 20.0, 2001)
        density = exponential_piston(positions, 10.0, 1.0)
        xi, normalised = pp.collapse_profile(positions, density, 10.0, 1.0)
        assert xi[np.argmin(np.abs(positions - 10.0))] == pytest.approx(0.0, abs=1e-3)
        assert np.nanmax(normalised) == pytest.approx(1.0, rel=1e-9)

    @pytest.mark.parametrize("x_front,scale_length", [
        (np.nan, 1.0), (10.0, np.nan), (10.0, 0.0), (10.0, -1.0),
    ])
    def test_degenerate_inputs_give_all_nan(self, x_front, scale_length):
        positions = np.linspace(0.0, 20.0, 101)
        density = np.exp(-positions)
        xi, normalised = pp.collapse_profile(positions, density, x_front, scale_length)
        assert np.all(np.isnan(xi)) and np.all(np.isnan(normalised))
        assert xi.shape == positions.shape

    def test_zero_density_gives_all_nan(self):
        positions = np.linspace(0.0, 20.0, 101)
        xi, normalised = pp.collapse_profile(positions, np.zeros_like(positions),
                                             10.0, 1.0)
        assert np.all(np.isnan(xi)) and np.all(np.isnan(normalised))


class TestWeightedBinAverage:
    """The particle route to a bulk-velocity profile: weights must be honoured."""

    def test_uniform_weights_reduce_to_the_plain_mean(self):
        positions = np.array([0.5, 0.6, 1.5, 1.6])
        values = np.array([10.0, 20.0, 30.0, 50.0])
        out = pp.weighted_bin_average(positions, values, np.ones(4),
                                      np.array([0.0, 1.0, 2.0]))
        assert out == pytest.approx([15.0, 40.0])

    def test_a_heavy_macroparticle_dominates_its_bin(self):
        positions = np.array([0.2, 0.8])
        velocities = np.array([0.0, 100.0])
        out = pp.weighted_bin_average(positions, velocities, np.array([1.0, 9.0]),
                                      np.array([0.0, 1.0]))
        assert out[0] == pytest.approx(90.0)

    def test_empty_bins_are_nan_not_zero(self):
        # A zero would read as "plasma at rest here", which is the opposite claim.
        out = pp.weighted_bin_average(np.array([0.5]), np.array([7.0]), np.array([1.0]),
                                      np.array([0.0, 1.0, 2.0]))
        assert out[0] == pytest.approx(7.0)
        assert np.isnan(out[1])

    def test_non_finite_samples_are_dropped(self):
        positions = np.array([0.5, 0.5, 0.5])
        values = np.array([10.0, np.nan, 30.0])
        out = pp.weighted_bin_average(positions, values, np.ones(3),
                                      np.array([0.0, 1.0]))
        assert out[0] == pytest.approx(20.0)


class TestWeightedBinDensity:
    def test_weights_over_cell_volume(self):
        out = pp.weighted_bin_density(np.array([0.5, 0.5, 1.5]),
                                      np.array([2.0, 3.0, 4.0]),
                                      np.array([0.0, 1.0, 2.0]), cell_volume=5.0)
        assert out == pytest.approx([1.0, 0.8])

    def test_empty_bins_are_zero(self):
        out = pp.weighted_bin_density(np.array([0.5]), np.array([1.0]),
                                      np.array([0.0, 1.0, 2.0]), cell_volume=1.0)
        assert out[1] == pytest.approx(0.0)

    @pytest.mark.parametrize("cell_volume", [0.0, -1.0, np.nan])
    def test_bad_cell_volume_raises(self, cell_volume):
        with pytest.raises(ValueError):
            pp.weighted_bin_density(np.array([0.5]), np.array([1.0]),
                                    np.array([0.0, 1.0]), cell_volume=cell_volume)


class TestProfileMismatch:
    def test_identical_profiles_match_exactly(self):
        x = np.linspace(0.0, 10.0, 51)
        y = np.exp(-x)
        assert pp.profile_mismatch(x, y, x, y) == pytest.approx(0.0, abs=1e-12)

    def test_a_scaled_profile_does_not_match(self):
        # The metric compares shape at a common normalisation; it is NOT scale-invariant,
        # so the caller must normalise first (both sides divide by their own n_drive).
        x = np.linspace(0.0, 10.0, 51)
        y = np.exp(-x)
        assert pp.profile_mismatch(x, y, x, 2.0 * y) == pytest.approx(1.0, rel=1e-6)

    def test_resamples_onto_a_different_grid(self):
        reference_x = np.linspace(0.0, 10.0, 51)
        x = np.linspace(0.0, 10.0, 37)
        assert pp.profile_mismatch(reference_x, 3.0 * reference_x,
                                   x, 3.0 * x) == pytest.approx(0.0, abs=1e-12)

    def test_too_little_overlap_is_nan(self):
        reference_x = np.linspace(0.0, 10.0, 51)
        y = np.ones(51)
        assert np.isnan(pp.profile_mismatch(reference_x, y,
                                            np.array([9.9, 10.0]), np.array([1.0, 1.0])))

    def test_unsorted_candidate_is_handled(self):
        x = np.linspace(0.0, 10.0, 51)
        order = np.argsort(np.sin(x))
        assert pp.profile_mismatch(x, 2.0 * x, x[order],
                                   2.0 * x[order]) == pytest.approx(0.0, abs=1e-12)


class TestBestShapeMatch:
    def test_picks_the_matching_candidate(self):
        x = np.linspace(0.0, 10.0, 51)
        reference = np.exp(-x)
        candidates = [(x, np.exp(-x / 3.0)), (x, reference), (x, np.exp(-3.0 * x))]
        index, mismatch = pp.best_shape_match(x, reference, candidates)
        assert index == 1
        assert mismatch == pytest.approx(0.0, abs=1e-12)

    def test_no_scorable_candidate_returns_minus_one(self):
        x = np.linspace(0.0, 10.0, 51)
        far = (np.array([100.0, 101.0]), np.array([1.0, 1.0]))
        index, mismatch = pp.best_shape_match(x, np.ones(51), [far])
        assert index == -1 and np.isnan(mismatch)
