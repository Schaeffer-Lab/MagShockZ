"""Tests for shock.py — shared front detection and trajectory fitting."""


import numpy as np
import pytest

from magshockz.analysis.flash import shock


# ---------------------------------------------------------------------------
# robust_linfit
# ---------------------------------------------------------------------------

def test_robust_linfit_recovers_clean_line():
    t = np.linspace(0, 10, 50)
    slope, intercept = shock.robust_linfit(t, 2.0 * t + 3.0)
    assert slope == pytest.approx(2.0, abs=1e-9)
    assert intercept == pytest.approx(3.0, abs=1e-9)


def test_robust_linfit_rejects_outliers():
    t = np.linspace(0, 10, 50)
    x = 2.0 * t + 3.0
    x[10] += 500.0   # gross outliers that plain polyfit would chase
    x[30] -= 400.0
    slope, intercept = shock.robust_linfit(t, x)
    assert slope == pytest.approx(2.0, abs=1e-6)
    assert intercept == pytest.approx(3.0, abs=1e-6)


def test_robust_linfit_keeps_at_least_three_points():
    # Pathological data: clipping must not collapse below 3 points / crash.
    t = np.array([0.0, 1.0, 2.0, 3.0])
    x = np.array([0.0, 100.0, 0.0, 100.0])
    slope, intercept = shock.robust_linfit(t, x)
    assert np.isfinite(slope) and np.isfinite(intercept)


# ---------------------------------------------------------------------------
# robust_polyfit / trajectory_at
# ---------------------------------------------------------------------------

def test_robust_polyfit_recovers_quadratic():
    t = np.linspace(0, 10, 50)
    coeffs = shock.robust_polyfit(t, 0.5 * t**2 + 2.0 * t + 3.0, deg=2)
    np.testing.assert_allclose(coeffs, [0.5, 2.0, 3.0], atol=1e-8)


def test_robust_polyfit_rejects_outliers():
    t = np.linspace(0, 10, 50)
    x = 0.5 * t**2 + 2.0 * t + 3.0
    x[10] += 500.0
    x[30] -= 400.0
    coeffs = shock.robust_polyfit(t, x, deg=2)
    np.testing.assert_allclose(coeffs, [0.5, 2.0, 3.0], atol=1e-5)


def test_trajectory_at_position_and_velocity():
    # x = 0.5 a t^2 + v0 t + x0  ->  v(t) = a t + v0
    a, v0, x0 = 0.5, 2.0, 3.0
    t = np.linspace(0, 10, 40)
    coeffs = shock.robust_polyfit(t, 0.5 * a * t**2 + v0 * t + x0, deg=2)
    pos, vel = shock.trajectory_at(coeffs, 4.0)
    assert pos == pytest.approx(0.5 * a * 16 + v0 * 4 + x0, abs=1e-6)
    assert vel == pytest.approx(a * 4 + v0, abs=1e-6)


def test_trajectory_at_linear_is_constant_velocity():
    t = np.linspace(0, 10, 30)
    coeffs = shock.robust_polyfit(t, 2.0 * t + 3.0, deg=1)
    _, vel = shock.trajectory_at(coeffs, np.array([1.0, 9.0]))
    np.testing.assert_allclose(vel, 2.0, atol=1e-9)


# ---------------------------------------------------------------------------
# front_line  (FLASH: the hand-placed straight trajectory)
# ---------------------------------------------------------------------------

def test_front_line_passes_through_its_anchor():
    assert shock.front_line(0.185, 9.7e7, 2.25e-9, t0=2.25e-9) == pytest.approx(0.185)


def test_front_line_default_anchor_is_zero():
    np.testing.assert_allclose(shock.front_line(3.0, 2.0, np.array([0.0, 1.0, 2.0])),
                               [3.0, 5.0, 7.0])


def test_front_line_anchor_shifts_the_line_not_its_slope():
    t = np.linspace(0.0, 5.0, 11)
    early = shock.front_line(10.0, 2.0, t, t0=1.0)
    late = shock.front_line(10.0, 2.0, t, t0=3.0)
    # same slope, offset by v·Δt₀ — sliding the anchor translates the line in x
    np.testing.assert_allclose(np.diff(early), np.diff(late))
    np.testing.assert_allclose(early - late, 2.0 * (3.0 - 1.0))


def test_front_line_accepts_scalar_and_array_times():
    assert np.isscalar(shock.front_line(1.0, 1.0, 2.0)) or \
        shock.front_line(1.0, 1.0, 2.0).ndim == 0
    assert shock.front_line(1.0, 1.0, np.zeros(4)).shape == (4,)


# ---------------------------------------------------------------------------
# detect_front_edge  (OSIRIS: leading edge of compression)
# ---------------------------------------------------------------------------

def test_detect_front_edge_finds_leading_edge():
    x = np.linspace(0, 100, 1001)
    # compressed (high) below x=50, ambient (low) above -> leading edge at ~50
    profile = np.where(x < 50, 10.0, 1.0)
    xf = shock.detect_front_edge(x, profile, x_pred=50, half_window=20)
    assert xf == pytest.approx(50.0, abs=0.2)


def test_detect_front_edge_nan_without_compression():
    x = np.linspace(0, 100, 1001)
    profile = np.ones_like(x)            # flat -> no front
    assert np.isnan(shock.detect_front_edge(x, profile, 50, 20))


def test_detect_front_edge_nan_empty_window():
    x = np.linspace(0, 100, 101)
    profile = np.where(x < 50, 10.0, 1.0)
    assert np.isnan(shock.detect_front_edge(x, profile, x_pred=500, half_window=5))


# ---------------------------------------------------------------------------
# detect_front_gradient  (FLASH: steepest density drop)
# ---------------------------------------------------------------------------

def test_detect_front_gradient_finds_jump():
    x = np.linspace(0, 100, 1001)
    # smooth drop centred at 60 via a tanh ramp; steepest gradient at the centre
    ne = 5.0 - 4.0 * np.tanh((x - 60) / 1.0)
    xf = shock.detect_front_gradient(x, ne, x_pred=60, half_window=20)
    assert xf == pytest.approx(60.0, abs=1.0)


def test_detect_front_gradient_nan_without_compression():
    x = np.linspace(0, 100, 1001)
    ne = np.ones_like(x)
    assert np.isnan(shock.detect_front_gradient(x, ne, 50, 20))


def test_detect_front_gradient_nan_too_few_points():
    x = np.linspace(0, 100, 1001)
    ne = 5.0 - 4.0 * np.tanh((x - 60) / 1.0)
    # window narrower than 5 cells -> nan
    assert np.isnan(shock.detect_front_gradient(x, ne, x_pred=60, half_window=0.05))


# ---------------------------------------------------------------------------
# overview_row — locating a dump in a flash_overview archive
# ---------------------------------------------------------------------------
# Row position and plot-file index diverge under --stride/--t-start. The old
# positional lookup (idx % n_rows) silently returned another dump's shock position,
# which is a wrong answer rather than an error, so this is guarded.

def test_overview_row_finds_the_dump_by_index():
    indices = np.array([0, 4, 8, 12, 16])
    assert shock.overview_row(8, indices, len(indices)) == 2
    assert shock.overview_row(0, indices, len(indices)) == 0
    assert shock.overview_row(16, indices, len(indices)) == 4


def test_overview_row_is_the_identity_for_a_full_stride_one_archive():
    indices = np.arange(62)
    assert shock.overview_row(36, indices, 62) == 36


def test_overview_row_rejects_a_dump_the_archive_does_not_cover():
    """The exact failure the positional lookup hid: a stride-2 archive asked for
    dump 36 would have returned row 36 % 31 = 5."""
    indices = np.arange(0, 62, 2)
    with pytest.raises(ValueError, match="not 37"):
        shock.overview_row(37, indices, len(indices))


def test_overview_row_falls_back_positionally_for_an_old_archive():
    assert shock.overview_row(36, None, 62) == 36


def test_overview_row_refuses_an_old_archive_that_cannot_hold_the_dump():
    with pytest.raises(ValueError, match="predates"):
        shock.overview_row(36, None, 31)


# ---------------------------------------------------------------------------
# detect_front_outermost / local_front_fit
# ---------------------------------------------------------------------------

def test_outermost_detector_finds_the_front_not_the_steepest_drop():
    """A piston cavity puts the steepest density drop near the origin, not the front.

    The gradient detectors lock onto that inner wall and report a front that marches
    backwards; marching in from the far field is monotone in how far the disturbance
    reached, so it stays on the real front.
    """
    x = np.linspace(0.0, 1.0, 1001)
    rho = np.ones_like(x)
    rho[x < 0.60] = 4.0        # shocked shell, outer edge at 0.60
    rho[x < 0.30] = 0.05       # evacuated piston cavity: the steepest drop is here

    assert shock.detect_front_outermost(x, rho) == pytest.approx(0.60, abs=2e-3)


def test_outermost_detector_is_nan_when_nothing_is_disturbed():
    x = np.linspace(0.0, 1.0, 500)
    assert np.isnan(shock.detect_front_outermost(x, np.ones_like(x)))


def test_linear_and_quadratic_slopes_agree_on_a_symmetric_window():
    """The headline claim: curvature cannot bias a centred local slope.

    Over a window symmetric about the target and evenly sampled, the quadratic term
    is even about the centre while the linear basis is odd, so they are orthogonal
    and the linear fit recovers the instantaneous slope exactly.
    """
    t = np.arange(0.0, 20.0, 0.25)
    x = 500.0 + 900.0 * t - 25.0 * t**2          # decelerating blast, curvature real
    fit = shock.local_front_fit(t, x, target=9.0, half_width=1.0)

    truth = 900.0 - 2.0 * 25.0 * 9.0
    assert fit.v_quadratic == pytest.approx(truth)
    assert fit.v_linear == pytest.approx(truth)       # unbiased despite the curvature
    assert fit.acceleration == pytest.approx(-50.0)
    assert fit.rms_quadratic == pytest.approx(0.0, abs=1e-6)
    assert fit.rms_linear > fit.rms_quadratic         # the line still misses position
    assert fit.n_points == 9


def test_quadratic_residual_separates_curvature_from_mis_tracking():
    """A merely-curved track fits cleanly; a noisy one keeps a large residual.

    This is what lets the residual distinguish los15's tracking failure from the
    honest curvature every ray has.
    """
    t = np.arange(0.0, 20.0, 0.25)
    curved = 500.0 + 900.0 * t - 25.0 * t**2
    rng = np.random.default_rng(0)
    noisy = curved + rng.normal(0.0, 80.0, t.size)

    assert shock.local_front_fit(t, curved, 9.0).rms_quadratic < 1e-6
    assert shock.local_front_fit(t, noisy, 9.0).rms_quadratic > 20.0


def test_local_front_fit_degrades_gracefully_on_a_thin_window():
    t = np.array([8.0, 9.0])
    x = np.array([100.0, 200.0])
    fit = shock.local_front_fit(t, x, target=8.5, half_width=1.0)
    assert fit.v_linear == pytest.approx(100.0)
    assert np.isnan(fit.v_quadratic) and np.isnan(fit.acceleration)

    empty = shock.local_front_fit(t, x, target=50.0, half_width=1.0)
    assert np.isnan(empty.v_linear) and empty.n_points == 0


def test_local_front_fit_ignores_nans_in_the_track():
    t = np.arange(0.0, 20.0, 0.25)
    x = 500.0 + 400.0 * t
    x[abs(t - 9.0) < 0.3] = np.nan          # a few dumps where tracking failed
    fit = shock.local_front_fit(t, x, target=9.0, half_width=1.0)
    assert fit.v_linear == pytest.approx(400.0)
    assert fit.n_points == 9 - 3


def test_contact_position_finds_the_outermost_piston_material():
    """The contact is the OUTERMOST piston material, not the first or the bulk edge.

    FLASH's advected fractions leave a thin tail of piston material ahead of the bulk
    contact; the downstream band has to clear all of it, so the detector takes the
    last sample above threshold rather than where the fraction becomes large.
    """
    x = np.linspace(0.0, 1.0, 1001)
    targ = np.zeros_like(x)
    targ[x < 0.40] = 1.0            # bulk piston
    targ[(x >= 0.40) & (x < 0.45)] = 5.0e-3   # thin advected tail

    assert shock.contact_position(x, targ) == pytest.approx(0.45, abs=2e-3)


def test_contact_position_ignores_material_below_threshold():
    x = np.linspace(0.0, 1.0, 501)
    targ = np.full_like(x, 1.0e-9)
    targ[x < 0.30] = 1.0
    assert shock.contact_position(x, targ) == pytest.approx(0.30, abs=3e-3)


def test_contact_position_is_nan_when_no_piston_reaches_the_ray():
    x = np.linspace(0.0, 1.0, 100)
    assert np.isnan(shock.contact_position(x, np.zeros_like(x)))


# ---------------------------------------------------------------------------
# resolve_bands / ShockBands
# ---------------------------------------------------------------------------

def _ray(n=2001, x_hi=1.0, contact_at=0.30):
    x = np.linspace(0.0, x_hi, n)
    piston = np.where(x < contact_at, 1.0, 0.0)
    return x, piston


def test_resolve_bands_puts_the_downstream_edge_at_the_contact():
    x, piston = _ray(contact_at=0.30)
    b = shock.resolve_bands(x, piston, x_shock=0.60, upstream_gap=0.02,
                            upstream_width=0.06, contact_gap=0.005,
                            x_downstream_config=0.58)
    assert b.x_downstream == pytest.approx(0.305, abs=1e-3)   # contact + gap
    assert b.x_upstream_lo == pytest.approx(0.62)
    assert b.x_upstream_hi == pytest.approx(0.68)
    assert b.note == ""


def test_resolve_bands_honours_the_config_edge_when_asked():
    x, piston = _ray()
    b = shock.resolve_bands(x, piston, x_shock=0.60, upstream_gap=0.02,
                            upstream_width=0.06, contact_gap=0.005,
                            x_downstream_config=0.58, edge="config")
    assert b.x_downstream == pytest.approx(0.58)


def test_resolve_bands_falls_back_and_says_so_when_no_piston_is_present():
    x = np.linspace(0.0, 1.0, 501)
    b = shock.resolve_bands(x, np.zeros_like(x), x_shock=0.60, upstream_gap=0.02,
                            upstream_width=0.06, contact_gap=0.005,
                            x_downstream_config=0.55)
    assert b.x_downstream == pytest.approx(0.55)
    assert "no piston material" in b.note


def test_resolve_bands_falls_back_when_the_contact_has_caught_the_front():
    x, piston = _ray(contact_at=0.62)
    b = shock.resolve_bands(x, piston, x_shock=0.60, upstream_gap=0.02,
                            upstream_width=0.06, contact_gap=0.005,
                            x_downstream_config=0.55)
    assert b.x_downstream == pytest.approx(0.55)
    assert "reached the front" in b.note


def test_zero_upstream_width_restores_the_to_the_end_of_ray_behaviour():
    x, piston = _ray(x_hi=1.0)
    b = shock.resolve_bands(x, piston, x_shock=0.60, upstream_gap=0.02,
                            upstream_width=0.0, contact_gap=0.005,
                            x_downstream_config=0.55)
    assert b.x_upstream_hi == pytest.approx(1.0)


def test_upstream_window_is_clipped_to_the_end_of_the_ray():
    x, piston = _ray(x_hi=1.0)
    b = shock.resolve_bands(x, piston, x_shock=0.95, upstream_gap=0.02,
                            upstream_width=0.50, contact_gap=0.005,
                            x_downstream_config=0.90)
    assert b.x_upstream_hi == pytest.approx(1.0)


def test_resolve_bands_refuses_an_upstream_window_off_the_end_of_the_ray():
    x, piston = _ray(x_hi=1.0)
    with pytest.raises(ValueError, match="past the end of the ray"):
        shock.resolve_bands(x, piston, x_shock=0.99, upstream_gap=0.05,
                            upstream_width=0.06, contact_gap=0.005,
                            x_downstream_config=0.90)


def test_resolve_bands_refuses_an_empty_downstream_band():
    x, piston = _ray(contact_at=0.62)
    with pytest.raises(ValueError, match="at or past the front"):
        shock.resolve_bands(x, piston, x_shock=0.60, upstream_gap=0.02,
                            upstream_width=0.06, contact_gap=0.005,
                            x_downstream_config=0.70)


def test_shockbands_masks_select_the_declared_windows():
    x, piston = _ray()
    b = shock.resolve_bands(x, piston, x_shock=0.60, upstream_gap=0.02,
                            upstream_width=0.06, contact_gap=0.005,
                            x_downstream_config=0.58)
    up, dn = b.upstream_mask(x), b.downstream_mask(x)
    assert up.any() and dn.any()
    assert not (up & dn).any()                       # the bands never overlap
    assert x[up].min() >= b.x_upstream_lo and x[up].max() <= b.x_upstream_hi
    assert x[dn].min() >= b.x_downstream and x[dn].max() <= b.x_shock


def test_jump_band_is_thin_and_sits_against_the_front():
    """The RH conditions are local, so their band is a thin slice at the front.

    Measured on this data, momentum-flux continuity runs 1.00 over 50 um and 0.51
    over 940 um, so testing the jump condition needs its own band rather than the
    full shocked layer.
    """
    x, piston = _ray(contact_at=0.30)
    b = shock.resolve_bands(x, piston, x_shock=0.60, upstream_gap=0.02,
                            upstream_width=0.06, contact_gap=0.005,
                            jump_width=0.01, x_downstream_config=0.58)
    assert b.x_jump == pytest.approx(0.59)
    assert b.x_downstream == pytest.approx(0.305, abs=1e-3)

    jump, layer = b.jump_mask(x), b.downstream_mask(x)
    assert jump.sum() < layer.sum()
    assert not (jump & ~layer).any()      # the jump band lies inside the layer band


def test_jump_band_never_reaches_past_the_layer_band():
    """A jump width wider than the shocked layer must clip, not run into the piston."""
    x, piston = _ray(contact_at=0.55)
    b = shock.resolve_bands(x, piston, x_shock=0.60, upstream_gap=0.02,
                            upstream_width=0.06, contact_gap=0.005,
                            jump_width=0.50, x_downstream_config=0.58)
    assert b.x_jump == pytest.approx(b.x_downstream)


def test_zero_jump_width_collapses_the_jump_band_onto_the_layer_band():
    x, piston = _ray(contact_at=0.30)
    b = shock.resolve_bands(x, piston, x_shock=0.60, upstream_gap=0.02,
                            upstream_width=0.06, contact_gap=0.005,
                            x_downstream_config=0.58)
    assert b.x_jump == pytest.approx(b.x_downstream)
    assert (b.jump_mask(x) == b.downstream_mask(x)).all()


# ---------------------------------------------------------------------------
# snap_front_to_jump
# ---------------------------------------------------------------------------

def test_snap_moves_a_front_placed_outside_the_jump_onto_it():
    """A front 20 um outside the jump fills a fifth of a 100 um band with upstream."""
    x = np.linspace(0.0, 1000.0, 1001)          # 1 um grid
    rho = np.where(x < 600.0, 4.0, 1.0)         # jump at 600, downstream at smaller x
    assert shock.snap_front_to_jump(x, rho, x_front=620.0, search=60.0) == \
        pytest.approx(600.0, abs=2.0)


def test_snap_leaves_a_correctly_placed_front_alone():
    x = np.linspace(0.0, 1000.0, 1001)
    rho = np.where(x < 600.0, 4.0, 1.0)
    assert shock.snap_front_to_jump(x, rho, x_front=600.0, search=60.0) == \
        pytest.approx(600.0, abs=2.0)


def test_snap_cannot_relocate_beyond_its_search_window():
    """It corrects a placement; it never goes hunting for a different feature."""
    x = np.linspace(0.0, 1000.0, 1001)
    rho = np.where(x < 200.0, 8.0, 1.0)          # the real jump is far away, at 200
    snapped = shock.snap_front_to_jump(x, rho, x_front=600.0, search=30.0)
    assert abs(snapped - 600.0) <= 30.0


def test_snap_picks_the_drop_not_the_rise():
    """Density rises into the shocked layer and falls at the front; only one is it."""
    x = np.linspace(0.0, 1000.0, 1001)
    rho = np.ones_like(x)
    rho[(x >= 500.0) & (x < 600.0)] = 4.0        # rise at 500, drop at 600
    assert shock.snap_front_to_jump(x, rho, x_front=590.0, search=120.0) == \
        pytest.approx(600.0, abs=2.0)


def test_snap_is_a_no_op_on_a_window_with_too_few_samples():
    x = np.array([0.0, 500.0, 1000.0])
    rho = np.array([4.0, 4.0, 1.0])
    assert shock.snap_front_to_jump(x, rho, x_front=500.0, search=1.0) == 500.0
