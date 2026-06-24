"""Tests for synthetic_diagnostics.py."""

import importlib.util
import os

import numpy as np
import pytest

_PATH = os.path.join(os.path.dirname(__file__), "..", "src", "synthetic_diagnostics.py")
_spec = importlib.util.spec_from_file_location("synthetic_diagnostics", _PATH)
sd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sd)


# ---------------------------------------------------------------------------
# bremsstrahlung_emissivity
# ---------------------------------------------------------------------------

def test_emissivity_frequency_integrated_scaling():
    n_e = np.array([1.0, 2.0])
    T_e = np.array([4.0, 4.0])
    eps = sd.bremsstrahlung_emissivity(n_e, T_e, species=[(n_e, 13)], gaunt=1.0)
    # eps = n_e * (n_i Z^2) * sqrt(T_e); with n_i = n_e, Z=13, T_e=4 -> sqrt=2
    np.testing.assert_allclose(eps, n_e * (n_e * 169.0) * 2.0)


def test_emissivity_Z_squared_weighting():
    n_e = np.array([1.0])
    T_e = np.array([1.0])
    al = sd.bremsstrahlung_emissivity(n_e, T_e, species=[(n_e, 13)], gaunt=1.0)
    si = sd.bremsstrahlung_emissivity(n_e, T_e, species=[(n_e, 14)], gaunt=1.0)
    assert si / al == pytest.approx((14.0 / 13.0) ** 2)


def test_emissivity_multispecies_adds():
    n_e = np.array([1.0])
    T_e = np.array([1.0])
    both = sd.bremsstrahlung_emissivity(n_e, T_e, species=[(n_e, 13), (0.5 * n_e, 14)], gaunt=1.0)
    expect = n_e * (1.0 * 169.0 + 0.5 * 196.0) * 1.0
    np.testing.assert_allclose(both, expect)


def test_emissivity_spectral_exponential_falloff():
    n_e = np.array([1.0])
    T_e = np.array([2.0])
    hnu = np.array([0.0, 2.0, 4.0])
    eps = sd.bremsstrahlung_emissivity(n_e, T_e, species=[(n_e, 1)], hnu=hnu, gaunt=1.0)
    # ratio between successive hnu separated by T_e should be exp(-1).
    ratios = eps[0, 1:] / eps[0, :-1]
    np.testing.assert_allclose(ratios, np.exp(-1.0), rtol=1e-6)


# ---------------------------------------------------------------------------
# line_of_sight_integral
# ---------------------------------------------------------------------------

def test_los_integral_uniform_slab():
    x = np.linspace(0.0, 10.0, 101)
    col = sd.line_of_sight_integral(np.full_like(x, 3.0), x)
    assert col == pytest.approx(3.0 * 10.0)


def test_los_integral_2d_image():
    x = np.linspace(0.0, 4.0, 41)
    field = np.outer(np.array([1.0, 2.0, 3.0]), np.ones_like(x))  # 3 rows
    image = sd.line_of_sight_integral(field, x, axis=1)
    np.testing.assert_allclose(image, np.array([1.0, 2.0, 3.0]) * 4.0)


# ---------------------------------------------------------------------------
# apply_resolution / probe_signal
# ---------------------------------------------------------------------------

def test_resolution_preserves_area_and_lowers_peak():
    x = np.linspace(-10.0, 10.0, 2001)
    signal = np.zeros_like(x)
    signal[np.argmin(np.abs(x))] = 100.0  # spike
    blurred = sd.apply_resolution(signal, x, fwhm=2.0)
    dx = x[1] - x[0]
    assert np.sum(blurred) * dx == pytest.approx(np.sum(signal) * dx, rel=1e-3)
    assert blurred.max() < signal.max()


def test_resolution_zero_fwhm_is_identity():
    x = np.linspace(0.0, 5.0, 50)
    s = np.sin(x)
    np.testing.assert_array_equal(sd.apply_resolution(s, x, fwhm=0.0), s)


def test_probe_signal_samples_blurred_field():
    x = np.linspace(0.0, 10.0, 1001)
    field = 2.0 * x + 1.0
    vals = sd.probe_signal(field, x, probe_coords=[2.5, 7.5], fwhm=0.0)
    np.testing.assert_allclose(vals, [2.0 * 2.5 + 1.0, 2.0 * 7.5 + 1.0], rtol=1e-6)
