"""Tests for src/spitzer_resistivity.py.

``magnetic_diffusivity`` and ``warpx_electron_temperature`` are exact-checked; the
plasmapy wrapper ``spitzer_resistivity`` is checked against known Spitzer regimes.
"""

import math

import numpy as np
import pytest

from spitzer_resistivity import (
    MU_0,
    magnetic_diffusivity,
    warpx_electron_temperature,
    spitzer_resistivity,
)


# ---------------------------------------------------------------------------
# magnetic_diffusivity: D_m = eta / mu0
# ---------------------------------------------------------------------------

def test_magnetic_diffusivity_scalar():
    assert magnetic_diffusivity(MU_0) == pytest.approx(1.0)
    assert magnetic_diffusivity(1e-6) == pytest.approx(1e-6 / MU_0)


def test_magnetic_diffusivity_array_preserves_shape():
    eta = np.array([1e-6, 2e-6, 4e-6])
    out = magnetic_diffusivity(eta)
    assert out.shape == eta.shape
    np.testing.assert_allclose(out, eta / MU_0)


# ---------------------------------------------------------------------------
# warpx_electron_temperature: Te = Te0 (n/n0)^(gamma-1)
# ---------------------------------------------------------------------------

def test_warpx_Te_at_reference_density_is_Te0():
    assert warpx_electron_temperature(5e24, 5e24, 300.0, 5.0 / 3.0) == pytest.approx(300.0)


def test_warpx_Te_isothermal_gamma_one():
    # gamma == 1 -> exponent 0 -> Te0 everywhere, independent of density.
    for n in (1e23, 5e24, 9e25):
        assert warpx_electron_temperature(n, 5e24, 300.0, 1.0) == pytest.approx(300.0)


def test_warpx_Te_adiabatic_scaling():
    # gamma=5/3 -> Te = Te0 (n/n0)^(2/3). At n = n0/8, (1/8)^(2/3) = 1/4.
    assert warpx_electron_temperature(5e24 / 8, 5e24, 320.0, 5.0 / 3.0) == pytest.approx(80.0)


def test_warpx_Te_zero_density_returns_zero():
    out = warpx_electron_temperature(np.array([0.0, 5e24]), 5e24, 300.0, 5.0 / 3.0)
    assert out[0] == 0.0
    assert out[1] == pytest.approx(300.0)


# ---------------------------------------------------------------------------
# spitzer_resistivity: plasmapy wrapper
# ---------------------------------------------------------------------------

def test_spitzer_scalar_order_of_magnitude():
    # ~40 eV, Z~10, HED density -> ~1e-6 Ohm*m (few micro-ohm-metre), a known regime.
    eta = spitzer_resistivity(40.0, 7.5e24, 10.0)
    assert np.isscalar(eta) or np.ndim(eta) == 0
    assert 1e-7 < eta < 1e-5


def test_spitzer_decreases_with_temperature():
    # Spitzer eta ~ T^-3/2, so hotter is less resistive.
    eta = spitzer_resistivity(np.array([10.0, 100.0, 1000.0]), np.full(3, 7.5e24), np.full(3, 10.0))
    assert eta.shape == (3,)
    assert eta[0] > eta[1] > eta[2]


def test_spitzer_array_broadcast_and_grouping():
    # Mixed charge states (incl. fully-stripped Si Z~14) map via integer-charge grouping.
    Te = np.array([15.0, 40.0, 300.0, 1000.0])
    ne = np.full(4, 7.5e24)
    Z = np.array([4.2, 10.0, 13.9, 14.0])
    eta = spitzer_resistivity(Te, ne, Z)
    assert eta.shape == (4,)
    assert np.all(np.isfinite(eta))


def test_spitzer_invalid_inputs_are_nan():
    Te = np.array([0.0, -1.0, 40.0, math.nan])
    ne = np.array([7.5e24, 7.5e24, 0.0, 7.5e24])
    Z = np.array([10.0, 10.0, 10.0, 10.0])
    eta = spitzer_resistivity(Te, ne, Z)
    assert np.isnan(eta[:3]).all()  # bad Te, bad Te, bad ne
    assert math.isnan(eta[3])       # nan Te


def test_spitzer_ion_identity_negligible():
    # Al vs Si agree to well within 1% for e-i resistivity (ion mass barely enters).
    eta_al = spitzer_resistivity(40.0, 7.5e24, 13.0, ion="Al")
    eta_si = spitzer_resistivity(40.0, 7.5e24, 13.0, ion="Si")
    assert eta_al == pytest.approx(eta_si, rel=1e-2)
