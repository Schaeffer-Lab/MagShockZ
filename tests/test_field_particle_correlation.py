"""Tests for field_particle_correlation.py (Klein-Howes FPC)."""

import importlib.util
import os

import numpy as np
import pytest

_PATH = os.path.join(os.path.dirname(__file__), "..", "src", "field_particle_correlation.py")
_spec = importlib.util.spec_from_file_location("field_particle_correlation", _PATH)
fpc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fpc)


def gaussian_f(u, x, amp, mu, sigma):
    """f(u, x) = amp * Gaussian(u; mu, sigma), uniform over x."""
    f_u = amp * np.exp(-0.5 * ((u - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    return np.outer(f_u, np.ones(len(x)))


U = np.linspace(-2.0, 2.0, 4001)
X = np.linspace(0.0, 5.0, 12)


def test_net_rate_matches_jdotE_identity():
    # ∫ C_E du = sign(rqm) * E * ∫ u f du = sign(rqm) * E * (amp * mu) = j.E
    amp, mu, sigma = 2.0, 0.1, 0.3
    E0, rqm = 0.5, -1.0  # electrons
    f = gaussian_f(U, X, amp, mu, sigma)
    e1 = np.full(len(X), E0)
    net = fpc.velocity_integrated_rate(fpc.energy_transfer_rate(f, U, e1, rqm), U)
    expected = np.sign(rqm) * E0 * amp * mu
    np.testing.assert_allclose(net, expected, rtol=1e-3)


def test_maxwellian_at_rest_has_zero_net_but_bipolar_signature():
    f = gaussian_f(U, X, amp=1.0, mu=0.0, sigma=0.3)
    e1 = np.full(len(X), 0.7)
    c_e = fpc.energy_transfer_rate(f, U, e1, rqm=-1.0)
    net = fpc.velocity_integrated_rate(c_e, U)
    np.testing.assert_allclose(net, 0.0, atol=1e-6)
    # Bipolar in velocity: opposite signs on the two sides of u=0.
    col = c_e[:, 0]
    left = col[U < -0.1]
    right = col[U > 0.1]
    assert np.nanmax(np.abs(left)) > 0 and np.nanmax(np.abs(right)) > 0
    assert np.sign(left[np.argmax(np.abs(left))]) != np.sign(right[np.argmax(np.abs(right))])


def test_charge_sign_flips_correlation():
    f = gaussian_f(U, X, amp=2.0, mu=0.1, sigma=0.3)
    e1 = np.full(len(X), 0.5)
    c_e_neg = fpc.energy_transfer_rate(f, U, e1, rqm=-1.0)
    c_e_pos = fpc.energy_transfer_rate(f, U, e1, rqm=+40.0)
    np.testing.assert_allclose(c_e_neg, -c_e_pos, rtol=1e-9)


def test_advective_zero_when_uniform_in_x():
    f = gaussian_f(U, X, amp=1.0, mu=0.2, sigma=0.3)
    adv = fpc.advective_flux(f, U, X)
    np.testing.assert_allclose(adv, 0.0, atol=1e-9)


def test_shape_validation():
    f = gaussian_f(U, X, 1.0, 0.0, 0.3)
    with pytest.raises(ValueError):
        fpc.energy_transfer_rate(f, U, np.ones(len(X) + 3), rqm=-1.0)
    with pytest.raises(ValueError):
        fpc.advective_flux(f, U[:-2], X)
