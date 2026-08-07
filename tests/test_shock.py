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
