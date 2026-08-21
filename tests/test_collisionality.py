"""Tests for magshockz/common/collisionality.py.

The Chandrasekhar function and the slowing-down rate are checked against plasmapy's own
``SingleParticleCollisionFrequencies`` (the quadrature-based reference implementation this
module replaces with a vectorized closed form), and the two derived lengths are checked
against their analytic limits.
"""

import numpy as np
import pytest

from magshockz.common.collisionality import (
    MIN_COULOMB_LOG,
    Slowing,
    chandrasekhar_psi,
    coulomb_log,
    interpenetration,
    mass_number,
    shock_collisionality,
    shock_scales,
    slowing_down,
)

pytest.importorskip("plasmapy")


# ---------------------------------------------------------------------------
# chandrasekhar_psi: the closed form must match plasmapy's quadrature
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("v_drift", [1e4, 1e5, 1e6, 1e7])
def test_psi_matches_plasmapy_quadrature(v_drift):
    import astropy.units as u
    from plasmapy.formulary.collisions import SingleParticleCollisionFrequencies

    ref = SingleParticleCollisionFrequencies(
        "p+", "p+", v_drift=v_drift * u.m / u.s, T_b=1e3 * u.eV,
        n_b=1e20 * u.m**-3, Coulomb_log=10 * u.dimensionless_unscaled,
    )
    # 1e-5, not machine precision: plasmapy evaluates psi with scipy.integrate.quad at its
    # default *absolute* tolerance (1.5e-8), so for small psi the reference is the looser
    # of the two.  The closed form / series here is exact.
    assert chandrasekhar_psi(float(ref.x)) == pytest.approx(float(ref.phi), rel=1e-5)


def test_psi_beam_limit_approaches_one():
    assert chandrasekhar_psi(1e4) == pytest.approx(1.0, rel=1e-12)


def test_psi_thermal_limit_matches_small_x_series():
    # psi(x) -> 4 x^(3/2) / (3 sqrt(pi)) as x -> 0.
    x = 1e-6
    assert chandrasekhar_psi(x) == pytest.approx(4.0 * x**1.5 / (3.0 * np.sqrt(np.pi)), rel=1e-5)


def test_psi_is_monotonic_and_bounded():
    x = np.logspace(-8, 4, 200)
    psi = chandrasekhar_psi(x)
    assert np.all((psi >= 0) & (psi <= 1.0))
    assert np.all(np.diff(psi) >= 0)              # saturates at exactly 1, so ties allowed
    rising = x < 10.0                             # strictly increasing before saturation
    assert np.all(np.diff(psi[rising]) > 0)


def test_psi_preserves_shape_and_scalar():
    assert np.ndim(chandrasekhar_psi(1.0)) == 0
    assert chandrasekhar_psi(np.zeros((3, 4))).shape == (3, 4)


# ---------------------------------------------------------------------------
# slowing_down: nu_s against plasmapy, and the two lengths against their limits
# ---------------------------------------------------------------------------

def test_nu_s_matches_plasmapy_momentum_loss():
    """The SI nu_0 conversion and the (1 + m_a/m_b) psi factor together reproduce plasmapy."""
    import astropy.units as u
    from plasmapy.formulary.collisions import SingleParticleCollisionFrequencies

    v, n_b, T_b = 3e5, 1e24, 40.0
    s = slowing_down(v=v, n_field=n_b, T_field=T_b, z_test=5, z_field=5)
    ref = SingleParticleCollisionFrequencies(
        "Al 5+", "Al 5+", v_drift=v * u.m / u.s, T_b=T_b * u.eV, n_b=n_b * u.m**-3,
        Coulomb_log=s.coulomb_log * u.dimensionless_unscaled,   # same lnLambda both sides
    )
    # 1e-3 rather than machine precision: plasmapy's "Al 5+" is the atomic mass less five
    # electrons, this module uses the neutral atomic mass, and nu_0 goes as 1/m_a^2.
    assert s.nu_0 == pytest.approx(float(ref.Lorentz_collision_frequency.value), rel=1e-3)
    assert s.nu_s == pytest.approx(float(ref.momentum_loss.value), rel=1e-3)


def test_lorentz_mfp_is_the_mass_factor_times_the_slowing_mfp():
    """collisionality.md's convention omits (1 + m_a/m_b) -- exactly 2 for like ions."""
    s = slowing_down(v=1e6, n_field=1e24, T_field=40.0, z_test=5, z_field=5)
    assert s.mfp_lorentz / s.mfp == pytest.approx(2.0, rel=1e-12)


def test_lorentz_mass_factor_for_unlike_ions():
    s = slowing_down(v=1e6, n_field=1e24, T_field=40.0, z_test=10, z_field=5,
                     test="Si", field="Al")
    expected = 1.0 + mass_number("Si") / mass_number("Al")
    assert s.mfp_lorentz / s.mfp == pytest.approx(expected, rel=1e-12)


def test_stopping_range_is_a_quarter_of_the_mfp_in_the_beam_limit():
    """nu_s ~ v^-3 with psi -> 1, so int dv/nu_s = v^4/4 / (v^3 nu_s) = mfp/4."""
    s = slowing_down(v=1e7, n_field=1e24, T_field=50.0, z_test=5, z_field=5)
    assert s.x > 1e4                                    # firmly in the beam limit
    assert s.stopping_range / s.mfp == pytest.approx(0.25, rel=1e-3)


def test_stopping_range_is_always_shorter_than_the_mfp():
    """The mfp collapses as v^4 while the ion slows, so the true range is always less."""
    s = slowing_down(v=np.logspace(4, 7, 40), n_field=1e24, T_field=50.0,
                     z_test=5, z_field=5)
    assert np.all(s.stopping_range < s.mfp)


def test_mfp_scales_as_v_to_the_fourth_in_the_beam_limit():
    v = np.array([1e6, 2e6])
    s = slowing_down(v=v, n_field=1e24, T_field=10.0, z_test=5, z_field=5)
    # lnLambda drifts with v, so divide it out before checking the power law.
    scaled = s.mfp * s.coulomb_log
    assert scaled[1] / scaled[0] == pytest.approx(2.0**4, rel=1e-3)


def test_mfp_scales_inversely_with_density():
    s = slowing_down(v=1e6, n_field=np.array([1e24, 2e24]), T_field=40.0,
                     z_test=5, z_field=5, n_e=5e24, T_e=40.0)
    assert s.mfp[0] / s.mfp[1] == pytest.approx(2.0, rel=1e-12)


# ---------------------------------------------------------------------------
# shapes, broadcasting and invalid-input guards
# ---------------------------------------------------------------------------

def test_scalar_in_scalar_out():
    s = slowing_down(v=1e6, n_field=1e24, T_field=40.0, z_test=5, z_field=5)
    assert isinstance(s, Slowing)
    assert all(np.ndim(getattr(s, f)) == 0
               for f in ("v", "nu_s", "mfp", "mfp_lorentz", "stopping_range", "x"))


def test_arrays_broadcast_against_scalars():
    s = slowing_down(v=np.full((2, 3), 1e6), n_field=1e24, T_field=40.0,
                     z_test=5, z_field=5)
    assert s.mfp.shape == (2, 3)
    assert np.allclose(s.mfp, s.mfp.ravel()[0])


@pytest.mark.parametrize("bad", [
    {"v": -1.0}, {"v": 0.0}, {"v": float("nan")},
    {"n_field": 0.0}, {"n_field": -1e24}, {"T_field": 0.0}, {"z_test": 0.0},
])
def test_invalid_inputs_are_nan(bad):
    kwargs = dict(v=1e6, n_field=1e24, T_field=40.0, z_test=5, z_field=5)
    kwargs.update(bad)
    s = slowing_down(**kwargs)
    assert np.isnan(s.mfp) and np.isnan(s.stopping_range) and np.isnan(s.nu_s)


def test_invalid_cells_do_not_poison_valid_ones():
    s = slowing_down(v=np.array([1e6, -1.0, 2e6]), n_field=1e24, T_field=40.0,
                     z_test=5, z_field=5)
    assert np.isnan(s.mfp[1])
    assert np.all(np.isfinite(s.mfp[[0, 2]]))


# ---------------------------------------------------------------------------
# coulomb_log
# ---------------------------------------------------------------------------

def test_coulomb_log_is_clamped_and_counted():
    """The strongly-coupled ambient drives the classical lnLambda below the validity floor."""
    ln, n_clamped = coulomb_log(1e4, 1e27, 1.0, 5.0, 5.0, "Al", "Al")
    assert np.all(ln >= MIN_COULOMB_LOG)
    if ln[0] == MIN_COULOMB_LOG:
        assert n_clamped == 1


def test_coulomb_log_grows_with_velocity():
    ln, _ = coulomb_log(np.array([1e5, 1e6, 1e7]), 1e24, 40.0, 5.0, 5.0, "Al", "Al")
    assert np.all(np.diff(ln) > 0)


def test_coulomb_log_charge_state_is_clamped_to_the_atomic_number():
    """Zbar can exceed Z_atomic by rounding; "Al 14+" is not a valid plasmapy particle."""
    ln, _ = coulomb_log(1e6, 1e24, 40.0, 13.6, 13.6, "Al", "Al")
    assert np.all(np.isfinite(ln))


# ---------------------------------------------------------------------------
# shock_collisionality: the headline application
# ---------------------------------------------------------------------------

def test_shock_collisionality_ratios_are_consistent_with_their_parts():
    r = shock_collisionality(v_inflow=3e5, n_i=1e24, T_i=30.0, T_e=30.0, z=5.0, b=20.0)
    assert r.knudsen == pytest.approx(r.ion.stopping_range / r.d_i)
    assert r.knudsen_mfp == pytest.approx(r.ion.mfp / r.d_i)
    assert r.gyro_ratio == pytest.approx(r.ion.stopping_range / r.rho_i)


def test_shock_knudsen_is_a_quarter_of_the_literature_ratio_in_the_beam_limit():
    """range = mfp/4, so the honest collisionless margin is 4x smaller than lambda/d_i."""
    r = shock_collisionality(v_inflow=1e6, n_i=1e24, T_i=30.0, T_e=30.0, z=5.0)
    assert r.knudsen / r.knudsen_mfp == pytest.approx(0.25, rel=1e-2)


def test_electrons_are_far_more_collisional_than_ions():
    """The claim is about ions; lambda_ei must stay visibly sub-micron next to lambda_ii."""
    r = shock_collisionality(v_inflow=3e5, n_i=1e24, T_i=30.0, T_e=30.0, z=5.0)
    assert r.electron.mfp < r.ion.mfp
    assert r.electron.mfp < 1e-4          # < 100 um


def test_d_i_matches_plasmapy_inertial_length():
    import astropy.units as u
    from plasmapy.formulary import inertial_length
    from plasmapy.particles import CustomParticle
    from scipy import constants as sc

    n_i, z = 1e24, 5.0
    r = shock_collisionality(v_inflow=3e5, n_i=n_i, T_i=30.0, T_e=30.0, z=z)
    particle = CustomParticle(mass=mass_number("Al") * sc.u * u.kg, charge=z * sc.e * u.C)
    expected = float(inertial_length(n_i / u.m**3, particle).to(u.m).value)
    assert r.d_i == pytest.approx(expected, rel=1e-9)


def test_shock_collisionality_is_vectorized_over_a_profile():
    n = np.logspace(23, 25, 7)
    r = shock_collisionality(v_inflow=3e5, n_i=n, T_i=30.0, T_e=30.0,
                             z=np.full(7, 5.0), b=20.0)
    assert r.knudsen.shape == (7,)
    assert np.all(np.diff(r.knudsen) < 0)      # denser plasma is more collisional


# ---------------------------------------------------------------------------
# interpenetration: the secondary application
# ---------------------------------------------------------------------------

def test_interpenetration_defaults_to_si_into_al():
    p = interpenetration(v_drift=5e5, n_field=1e24, T_field=30.0, z_test=10, z_field=5)
    same = slowing_down(v=5e5, n_field=1e24, T_field=30.0, z_test=10, z_field=5,
                        test="Si", field="Al")
    assert p.mfp == pytest.approx(same.mfp, rel=1e-12)


def test_interpenetration_is_longer_for_a_faster_piston():
    slow = interpenetration(v_drift=2e5, n_field=1e24, T_field=30.0, z_test=10, z_field=5)
    fast = interpenetration(v_drift=8e5, n_field=1e24, T_field=30.0, z_test=10, z_field=5)
    assert fast.stopping_range > 100 * slow.stopping_range     # v^4


def test_mass_number_lookup():
    assert mass_number("Al") == pytest.approx(26.98, abs=0.02)
    assert mass_number("Si") == pytest.approx(28.08, abs=0.02)
    assert mass_number("e-") == pytest.approx(5.486e-4, rel=1e-3)


# ---------------------------------------------------------------------------
# stopping-range quadrature: blocking, node count, and the thermal degeneracy
# ---------------------------------------------------------------------------

def test_blocked_quadrature_matches_the_unblocked_one(monkeypatch):
    """Large grids are integrated in blocks; that must not change the answer."""
    from magshockz.common import collisionality as mod

    v = np.logspace(5, 7, 997).reshape(997, 1)
    kwargs = dict(n_field=1e24, T_field=40.0, z_test=5, z_field=5)
    unblocked = slowing_down(v=v, **kwargs).stopping_range
    monkeypatch.setattr(mod, "QUADRATURE_BLOCK", 5_000)
    blocked = mod.slowing_down(v=v, **kwargs).stopping_range
    assert blocked.shape == (997, 1)
    np.testing.assert_allclose(blocked, unblocked, rtol=1e-12)


def test_stopping_range_is_converged_at_the_default_node_count():
    coarse = slowing_down(v=1e6, n_field=1e24, T_field=40.0, z_test=5, z_field=5, nodes=64)
    fine = slowing_down(v=1e6, n_field=1e24, T_field=40.0, z_test=5, z_field=5, nodes=1024)
    assert coarse.stopping_range == pytest.approx(fine.stopping_range, rel=5e-3)


def test_a_thermal_test_particle_has_zero_range():
    """It is already at the background temperature -- there is nothing to stop it from.

    This is why the FLASH ``knudsen_thermal`` field compares the *mfp* to d_i, not a range.
    """
    from scipy import constants as sc

    T = 40.0
    v_th = np.sqrt(2.0 * T * sc.e / (mass_number("Al") * sc.u))
    s = slowing_down(v=v_th, n_field=1e24, T_field=T, z_test=5, z_field=5)
    assert s.stopping_range == 0.0
    assert s.mfp > 0.0


# ---------------------------------------------------------------------------
# shock_scales
# ---------------------------------------------------------------------------

def test_shock_scales_follow_their_scalings():
    d_i, rho_i = shock_scales(n_e=np.array([1e24, 4e24]), T_i=30.0, z=5.0, b=20.0)
    assert d_i[0] / d_i[1] == pytest.approx(2.0, rel=1e-9)      # d_i ~ n^-1/2
    assert rho_i[0] == pytest.approx(rho_i[1], rel=1e-9)        # rho_i is density-free

    _, rho = shock_scales(n_e=1e24, T_i=30.0, z=5.0, b=np.array([10.0, 20.0]))
    assert rho[0] / rho[1] == pytest.approx(2.0, rel=1e-9)      # rho_i ~ 1/B


def test_shock_scales_without_b_leaves_gyroradius_nan():
    d_i, rho_i = shock_scales(n_e=1e24, T_i=30.0, z=5.0)
    assert np.isfinite(d_i) and np.isnan(rho_i)


def test_shock_scales_handles_invalid_cells():
    d_i, _ = shock_scales(n_e=np.array([1e24, 0.0, np.nan]), T_i=30.0,
                          z=np.array([5.0, 5.0, 5.0]))
    assert np.isfinite(d_i[0]) and np.isnan(d_i[1]) and np.isnan(d_i[2])
