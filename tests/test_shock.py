"""Tests for shock.py — shared front detection and trajectory fitting."""

import importlib.util
import os

import numpy as np
import pytest

_PATH = os.path.join(os.path.dirname(__file__), "..", "src", "shock.py")
_spec = importlib.util.spec_from_file_location("shock", _PATH)
shock = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(shock)


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
