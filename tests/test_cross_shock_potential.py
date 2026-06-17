"""Tests for cross_shock_potential.py."""

import importlib.util
import os

import numpy as np
import pytest

_PATH = os.path.join(os.path.dirname(__file__), "..", "src", "cross_shock_potential.py")
_spec = importlib.util.spec_from_file_location("cross_shock_potential", _PATH)
csp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(csp)


def test_constant_field_gives_linear_potential():
    # E1 = const a => e*phi = -a (x - x0), zero at x[0].
    x = np.linspace(0.0, 10.0, 101)
    a = 2.0
    e_phi = csp.potential_profile(np.full_like(x, a), x)
    np.testing.assert_allclose(e_phi, -a * (x - x[0]), atol=1e-9)
    assert e_phi[0] == pytest.approx(0.0)


def test_potential_jump_reference_free():
    # Linear potential; jump = downstream_mean - upstream_mean.
    x = np.linspace(0.0, 10.0, 1001)
    a = 3.0
    e1 = np.full_like(x, a)
    upstream = x > 9.0      # mean e*phi ~ -a*9.5
    downstream = x < 1.0    # mean e*phi ~ -a*0.5
    jump = csp.potential_jump(e1, x, upstream, downstream)
    # downstream_mean - upstream_mean = -a*0.5 - (-a*9.5) = a*9.0
    assert jump == pytest.approx(a * 9.0, rel=2e-3)


def test_zero_field_zero_jump():
    x = np.linspace(0.0, 10.0, 50)
    e1 = np.zeros_like(x)
    jump = csp.potential_jump(e1, x, x > 8.0, x < 2.0)
    assert jump == pytest.approx(0.0)


def test_reflection_parameter():
    # e*Δphi exactly equal to the ram energy => parameter = 1.
    abs_rqm_i, v_shock = 40.0, 0.05
    ram = 0.5 * abs_rqm_i * v_shock**2
    assert csp.reflection_parameter(ram, abs_rqm_i, v_shock) == pytest.approx(1.0)
    assert csp.reflection_parameter(0.5 * ram, abs_rqm_i, v_shock) == pytest.approx(0.5)


def test_reflection_parameter_zero_velocity_is_nan():
    assert np.isnan(csp.reflection_parameter(1.0, 40.0, 0.0))
