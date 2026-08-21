# -*- coding: utf-8 -*-
"""Coulomb slowing-down lengths, for the question "is the MagShockZ shock collisionless?"

The primary case is the **Al shock itself**: an upstream Al ion enters the front at the
shock-frame inflow speed and must be stopped over the transition for the shock to be
mediated by collisions.  The secondary case is Si piston ions interpenetrating the Al
chamber plasma.  Both are the same physics -- a test ion drifting through a Maxwellian
background -- so one function covers them, with different species arguments.

Physics
-------
Test particle ``a`` at speed ``v`` through a Maxwellian background ``b`` (NRL Formulary
p.31 / Trubnikov; this is what plasmapy's ``SingleParticleCollisionFrequencies`` implements,
and ``test_collisionality.py`` checks the two agree)::

    nu_0 = n_b q_a^2 q_b^2 lnLambda / (4 pi eps0^2 m_a^2 v^3)        [SI]
    x    = m_b v^2 / (2 k T_b)
    psi(x) = (2/sqrt(pi)) int_0^x sqrt(t) e^-t dt = erf(sqrt(x)) - (2/sqrt(pi)) sqrt(x) e^-x
    nu_s = (1 + m_a/m_b) psi(x) nu_0                                 momentum loss rate

The single expression spans both limits: ``psi -> 1`` for ``x >> 1`` (the fast/beam case --
the swept-up ion and the piston) and ``psi ~ (4/3 sqrt(pi)) x^(3/2)`` for ``x << 1``
(thermal).  Two lengths follow, and they answer different questions:

``mfp = v / nu_s``
    The instantaneous slowing-down mean free path.  This is the HEDP-literature convention
    and what ``results/FLASH_MagShockZ3D-corrected/collisionality.md`` tabulates.  It is an
    e-folding scale evaluated at the *initial* speed.

``stopping_range``
    How far the ion actually travels before thermalizing.  Since ``dv/dt = -nu_s v`` and
    ``dx = v dt``, the range is ``int dv / nu_s(v)`` from the thermal speed up to ``v``.
    Because ``nu_s ~ 1/v^3`` the integrand is ``~ v^3`` and the fast-limit range is
    ``mfp / 4`` -- the ion penetrates four times *less* than the instantaneous mfp suggests,
    because its mfp collapses as ``v^4`` while it slows.  This is the honest answer to "can
    collisions stop it over the ramp", so it is the headline number.

    It is only meaningful for a test particle that is *faster* than the background it is
    entering.  Feed it ``v = v_thermal`` and it correctly returns **zero**: a particle
    already at the background temperature has nothing to be stopped from.  The thermal
    counterpart to quote is the mean free path, not a range.

Note on ``collisionality.md``: that document used ``lambda = v / nu_0``, i.e. it omitted the
``(1 + m_a/m_b)`` factor -- a factor of **2** for like ions.  ``mfp_lorentz`` reproduces it
so the old table stays checkable; ``mfp`` is the correct slowing-down length.

Conventions
-----------
Bare floats at the numerical boundary, in SI, because attaching units per-cell over a FLASH
grid is a real cost: densities [m^-3], temperatures [eV], speeds [m/s], fields [T], lengths
[m], charges as charge state [dimensionless].  Species are plasmapy element symbols
(``"Al"``, ``"Si"``, ``"e-"``) and their masses come from plasmapy.  Scalars and
broadcastable arrays both work, and scalar-in gives scalar-out.

plasmapy is imported **lazily** (~2.5 s) and only for the Coulomb logarithm, the inertial
length and the gyroradius; the collision algebra is numpy because plasmapy's ``psi`` is a
``scipy.integrate.quad`` call and so cannot be mapped over a grid.
"""

from __future__ import annotations

import functools
import warnings
from dataclasses import dataclass

import numpy as np
from scipy import constants as sc
from scipy.special import erf

#: Number of log-spaced velocity nodes in the stopping-range quadrature.
STOPPING_RANGE_NODES = 256

#: Cap on ``cells * nodes`` held at once in the quadrature.  A FLASH chunk can be ~1e6
#: cells, which at full node count would allocate gigabytes, so it is integrated in blocks.
QUADRATURE_BLOCK = 4_000_000

#: Coulomb logarithm below this is outside the weak-coupling validity of the whole
#: formulation; it is clamped here and the clamp is reported in ``Slowing.n_clamped``.
MIN_COULOMB_LOG = 1.0

#: Below this the closed form for psi(x) loses precision to cancellation and the Taylor
#: series is used instead.  The two agree to ~1e-14 on either side of the switch.
PSI_SERIES_CUTOFF = 0.1


def chandrasekhar_psi(x: np.ndarray | float) -> np.ndarray | float:
    """Chandrasekhar function psi(x) = (2/sqrt(pi)) int_0^x sqrt(t) e^-t dt.

    Closed form ``erf(sqrt(x)) - (2/sqrt(pi)) sqrt(x) e^-x``, so it is vectorized --
    plasmapy computes the same quantity by quadrature and is therefore scalar-only.
    Asymptotes to 1 for large ``x`` (the beam limit) and to ``4 x^(3/2) / (3 sqrt(pi))``
    for small ``x`` (the thermal limit).

    The closed form subtracts two terms that agree to leading order as ``x -> 0``, losing
    all significant digits below ``x ~ 1e-4``; for small ``x`` the series
    ``(2/sqrt(pi)) x^(3/2) sum_k (-x)^k / (k! (k + 3/2))`` is used instead.
    """
    xa = np.asarray(x, dtype=float)
    root = np.sqrt(np.clip(xa, 0.0, None))
    psi = erf(root) - (2.0 / np.sqrt(np.pi)) * root * np.exp(-xa)

    small = xa < PSI_SERIES_CUTOFF
    if np.any(small):
        xs = np.where(small, xa, 0.0)
        series = np.zeros_like(xs)
        term = np.ones_like(xs)
        for k in range(16):
            series += term / (k + 1.5)
            term *= -xs / (k + 1.0)
        psi = np.where(small, (2.0 / np.sqrt(np.pi)) * np.sqrt(np.clip(xs, 0.0, None)) ** 3
                       * series, psi)
    return psi if np.ndim(x) else float(psi)


@functools.lru_cache(maxsize=None)
def mass_number(species: str) -> float:
    """Mass of a plasmapy species in atomic mass units, e.g. ``"Al" -> 26.98``.

    The **neutral** atomic mass: the mass an ion loses by being stripped is at most
    ``Z m_e / A``, i.e. 0.03% for fully ionized Al, which is far below the uncertainty in
    any input here.  Cached because the algebra needs it per call and plasmapy's lookup is
    not free.
    """
    from plasmapy.particles import Particle

    return float(Particle("e-" if species == "e-" else species).mass.value / sc.u)


@functools.lru_cache(maxsize=None)
def _atomic_number(species: str) -> int:
    from plasmapy.particles import atomic_number

    return 1 if species == "e-" else int(atomic_number(species))


def _particle_string(species: str, charge: int) -> str:
    return "e-" if species == "e-" else f"{species} {charge}+"


def coulomb_log(v: np.ndarray, n_e: np.ndarray, T_e: np.ndarray,
                z_test: np.ndarray, z_field: np.ndarray,
                test: str, field: str) -> tuple[np.ndarray, int]:
    """Classical Coulomb logarithm at relative speed ``v``, and the number of clamped cells.

    Thin wrapper over ``plasmapy.formulary.Coulomb_logarithm``.  plasmapy needs a *scalar*,
    integer charge state in a valid particle string, so a per-cell fractional ``Zbar`` is
    binned to integer states and one call is issued per state -- the pattern
    ``magshockz/analysis/warpx/spitzer_resistivity.py`` already uses.  Binning is safe here
    and only here: lnLambda depends on Z logarithmically, whereas the ``Z^4`` in ``nu_0`` is
    evaluated at the exact fractional charge.

    Parameters
    ----------
    v : relative speed [m/s]; n_e : electron density [m^-3]; T_e : electron temperature [eV].
    z_test, z_field : charge states.  test, field : plasmapy species symbols.

    Returns ``(lnLambda, n_clamped)``; lnLambda is clamped below at ``MIN_COULOMB_LOG``,
    which the strongly-coupled ambient (Zbar ~ 5, tens of eV) does reach.
    """
    import astropy.units as u
    from plasmapy.formulary import Coulomb_logarithm

    v_a, n_a, T_a, zt, zf = np.broadcast_arrays(
        *(np.atleast_1d(np.asarray(q, dtype=float)) for q in (v, n_e, T_e, z_test, z_field))
    )

    ln_lambda = np.full(v_a.shape, np.nan, dtype=float)
    valid = (np.isfinite(v_a) & (v_a > 0.0) & np.isfinite(n_a) & (n_a > 0.0)
             & np.isfinite(T_a) & (T_a > 0.0) & (zt > 0.0) & (zf > 0.0))
    if not valid.any():
        return ln_lambda, 0

    # One plasmapy call per (integer test charge, integer field charge) pair.
    key_t = np.clip(np.rint(zt), 1, _atomic_number(test)).astype(int)
    key_f = np.clip(np.rint(zf), 1, _atomic_number(field)).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # CouplingWarning is expected and handled by the clamp
        for zi, zj in {(int(i), int(j)) for i, j in zip(key_t[valid], key_f[valid])}:
            sub = valid & (key_t == zi) & (key_f == zj)
            ln_lambda[sub] = np.asarray(Coulomb_logarithm(
                T=T_a[sub] * u.eV, n_e=n_a[sub] / u.m**3,
                species=(_particle_string(test, zi), _particle_string(field, zj)),
                V=v_a[sub] * u.m / u.s, method="classical",
            ), dtype=float)

    n_clamped = int(np.count_nonzero(np.isfinite(ln_lambda) & (ln_lambda < MIN_COULOMB_LOG)))
    return np.maximum(ln_lambda, MIN_COULOMB_LOG), n_clamped


@dataclass(frozen=True)
class Slowing:
    """Coulomb slowing-down of a test particle streaming through a background.

    All lengths [m], rates [1/s].  ``x`` is the regime indicator ``m_b v^2 / 2 k T_b``:
    ``x >> 1`` is the beam limit the shock and piston live in, ``x << 1`` the thermal limit.
    """

    v: np.ndarray
    nu_s: np.ndarray
    nu_0: np.ndarray
    mfp: np.ndarray
    mfp_lorentz: np.ndarray
    stopping_range: np.ndarray
    coulomb_log: np.ndarray
    x: np.ndarray
    n_clamped: int


def slowing_down(*, v: np.ndarray | float,
                 n_field: np.ndarray | float, T_field: np.ndarray | float,
                 z_test: np.ndarray | float, z_field: np.ndarray | float,
                 test: str = "Al", field: str = "Al",
                 n_e: np.ndarray | float | None = None,
                 T_e: np.ndarray | float | None = None,
                 nodes: int = STOPPING_RANGE_NODES) -> Slowing:
    """Slowing-down rate, mean free path and stopping range of a test ion in a background.

    Parameters
    ----------
    v : test-particle speed **in the background's frame** [m/s].  For the shock this is the
        shock-frame inflow speed, not the lab-frame flow speed.
    n_field, T_field : background number density [m^-3] and temperature [eV].
    z_test, z_field : charge states of the two species (FLASH's Zbar; may be fractional).
    test, field : plasmapy species symbols, e.g. ``"Al"``, ``"Si"``, ``"e-"``.
    n_e, T_e : electron density [m^-3] / temperature [eV] for the Coulomb logarithm, which
        is set by electron screening.  Default to ``z_field * n_field`` and ``T_field``.
    nodes : quadrature nodes for the stopping range.  Lower it on very large grids, where
        the integral is memory-bound rather than accuracy-bound.

    Notes
    -----
    The stopping-range quadrature holds lnLambda fixed at its value at ``v``.  lnLambda is
    logarithmic and the ``v^3`` weighting concentrates the integral at the top of the range,
    so the error is far below the uncertainty in ``v`` itself (which enters as ``v^4``).
    """
    v_a, n_b, T_b, z_a, z_b = np.broadcast_arrays(
        *(np.atleast_1d(np.asarray(q, dtype=float))
          for q in (v, n_field, T_field, z_test, z_field))
    )
    scalar_in = all(np.ndim(q) == 0 for q in (v, n_field, T_field, z_test, z_field))

    ne = z_b * n_b if n_e is None else np.broadcast_to(
        np.atleast_1d(np.asarray(n_e, dtype=float)), v_a.shape)
    te = T_b if T_e is None else np.broadcast_to(
        np.atleast_1d(np.asarray(T_e, dtype=float)), v_a.shape)

    ln_lambda, n_clamped = coulomb_log(v_a, ne, te, z_a, z_b, test, field)

    m_a = mass_number(test) * sc.u
    m_b = mass_number(field) * sc.u
    mass_factor = 1.0 + m_a / m_b
    T_b_J = T_b * sc.e

    def rates(speed: np.ndarray, axis: int | None = None,
              index: slice | None = None) -> tuple[np.ndarray, np.ndarray]:
        """(nu_0, nu_s) at ``speed``, with lnLambda held at its value at ``v``.

        ``axis`` appends a trailing broadcast axis to the per-cell parameters so ``speed``
        may carry a quadrature dimension; ``index`` selects the flattened block of cells
        that ``speed`` covers.
        """
        def cell(q):
            if axis is None:
                return q
            return q.ravel()[index][..., None] if index is not None else q[..., None]

        nu0 = (cell(n_b) * (cell(z_a) * sc.e) ** 2 * (cell(z_b) * sc.e) ** 2 * cell(ln_lambda)
               / (4.0 * np.pi * sc.epsilon_0 ** 2 * m_a ** 2 * speed ** 3))
        x_local = m_b * speed ** 2 / (2.0 * cell(T_b_J))
        return nu0, mass_factor * chandrasekhar_psi(x_local) * nu0

    invalid = ~(np.isfinite(v_a) & (v_a > 0.0) & np.isfinite(n_b) & (n_b > 0.0)
                & np.isfinite(T_b) & (T_b > 0.0) & (z_a > 0.0) & (z_b > 0.0))

    with np.errstate(divide="ignore", invalid="ignore"):
        nu_0, nu_s = rates(v_a)
        mfp = v_a / nu_s
        mfp_lorentz = v_a / nu_0

        # Range: dx = -dv / nu_s(v), integrated from the thermalized speed up to v.
        # np.geomspace rejects a zero or negative endpoint, so invalid cells are given a
        # dummy interval here and masked back to nan below.
        v_thermal = np.sqrt(np.abs(2.0 * T_b_J / m_a))
        v_hi = np.where(invalid, 1.0, v_a)
        v_lo = np.where(invalid, 0.5, np.minimum(v_thermal, v_a))

        flat_hi, flat_lo = v_hi.ravel(), v_lo.ravel()
        stopping_flat = np.empty(flat_hi.size, dtype=float)
        block = max(1, QUADRATURE_BLOCK // nodes)
        for lo in range(0, flat_hi.size, block):
            sl = slice(lo, lo + block)
            grid = np.geomspace(flat_lo[sl], flat_hi[sl], nodes, axis=-1)
            _, nu_s_grid = rates(grid, axis=-1, index=sl)
            stopping_flat[sl] = np.trapezoid(1.0 / nu_s_grid, grid, axis=-1)
        stopping_range = stopping_flat.reshape(v_hi.shape)

        x = m_b * v_a ** 2 / (2.0 * T_b_J)

    fields = [nu_s, nu_0, mfp, mfp_lorentz, stopping_range, ln_lambda, x]
    fields = [np.where(invalid, np.nan, f) for f in fields]

    if scalar_in:
        return Slowing(float(v_a[0]), *(float(f[0]) for f in fields), n_clamped=n_clamped)
    return Slowing(v_a, *fields, n_clamped=n_clamped)


@dataclass(frozen=True)
class ShockCollisionality:
    """Is the transition collisionless?  Ion and electron stopping against the shock scales.

    ``knudsen`` is the number the claim rests on: the distance a swept-up upstream ion
    travels before Coulomb collisions thermalize it, in units of the ion inertial length
    over which a perpendicular magnetized shock actually transitions.  ``>> 1`` means
    collisions cannot have built the ramp.  ``knudsen_mfp`` is the same ratio using the
    instantaneous mfp, for comparison with the literature convention.

    ``electron`` is carried alongside deliberately: in MagShockZ the electron mean free path
    is sub-micron, so the plasma is *not* collisionless for electrons.  The claim is about
    ions, and reporting both keeps it from being overstated.
    """

    ion: Slowing
    electron: Slowing
    d_i: np.ndarray
    rho_i: np.ndarray
    knudsen: np.ndarray
    knudsen_mfp: np.ndarray
    gyro_ratio: np.ndarray


def shock_scales(*, n_e: np.ndarray | float, T_i: np.ndarray | float,
                 z: np.ndarray | float, b: np.ndarray | float | None = None,
                 ion_species: str = "Al") -> tuple[np.ndarray, np.ndarray]:
    """The two lengths a shock transition is measured against: ``(d_i, rho_i)`` [m].

    ``d_i`` is the ion inertial length -- a perpendicular magnetized shock transitions over
    a few of these -- and ``rho_i`` the thermal ion gyroradius.  ``b`` is the field magnitude
    [T]; ``None`` leaves ``rho_i`` nan.

    Unlike ``Coulomb_logarithm``, plasmapy's ``inertial_length`` and ``gyroradius`` accept a
    ``CustomParticle``, so the fractional FLASH ``Zbar`` is carried exactly.  They still take
    one particle at a time, so arrays are binned on the rounded charge state.
    """
    import astropy.units as u
    from plasmapy.formulary import gyroradius, inertial_length
    from plasmapy.particles import CustomParticle

    # Any of z, n_e, T_i and b may be the array that sets the shape, so broadcast first.
    z_arr, ne_arr, T_arr, b_arr = np.broadcast_arrays(
        *(np.atleast_1d(np.asarray(q, dtype=float))
          for q in (z, n_e, T_i, 0.0 if b is None else b))
    )
    a = mass_number(ion_species)
    d_i = np.full(z_arr.shape, np.nan)
    rho_i = np.full(z_arr.shape, np.nan)
    z_key = np.clip(np.rint(z_arr), 1, None)
    usable = np.isfinite(z_arr) & (z_arr > 0) & np.isfinite(ne_arr) & (ne_arr > 0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for zi in np.unique(z_key[usable]):
            sub = usable & (z_key == zi)
            particle = CustomParticle(mass=a * sc.u * u.kg, charge=float(zi) * sc.e * u.C)
            d_i[sub] = np.asarray(inertial_length(
                (ne_arr[sub] / float(zi)) / u.m**3, particle).to(u.m).value, dtype=float)
            if b is None:
                continue
            rho_i[sub] = np.asarray(gyroradius(
                np.abs(b_arr[sub]) * u.T, particle, T=T_arr[sub] * u.eV).to(u.m).value,
                dtype=float)

    if all(np.ndim(q) == 0 for q in (z, n_e, T_i)) and np.ndim(b if b is not None else 0) == 0:
        return float(d_i[0]), float(rho_i[0])
    return d_i, rho_i


def shock_collisionality(*, v_inflow: np.ndarray | float,
                         n_i: np.ndarray | float, T_i: np.ndarray | float,
                         T_e: np.ndarray | float, z: np.ndarray | float,
                         ion_species: str = "Al", b: np.ndarray | float | None = None,
                         n_e: np.ndarray | float | None = None) -> ShockCollisionality:
    """Collisionality of a shock whose upstream ions enter the front at ``v_inflow``.

    Like-ion (Al-on-Al by default) slowing-down at the **shock-frame** inflow speed, plus
    the electron-ion slowing-down of a thermal electron, compared against the ion inertial
    length and the ion gyroradius.

    Parameters
    ----------
    v_inflow : upstream inflow speed in the shock frame [m/s].
    n_i, T_i, T_e : ion density [m^-3], ion and electron temperature [eV].
    z : ion charge state (FLASH's Zbar; may be fractional).
    ion_species : plasmapy element symbol for the shocked ions.
    b : magnetic field magnitude [T], for the gyroradius.  ``None`` leaves ``rho_i`` nan.
    n_e : electron density [m^-3]; defaults to ``z * n_i``.
    """
    ne = np.asarray(z, dtype=float) * np.asarray(n_i, dtype=float) if n_e is None else n_e

    ion = slowing_down(v=v_inflow, n_field=n_i, T_field=T_i, z_test=z, z_field=z,
                       test=ion_species, field=ion_species, n_e=ne, T_e=T_e)

    v_te = np.sqrt(2.0 * np.asarray(T_e, dtype=float) * sc.e / sc.m_e)
    electron = slowing_down(v=v_te, n_field=n_i, T_field=T_i, z_test=1.0, z_field=z,
                            test="e-", field=ion_species, n_e=ne, T_e=T_e)

    d_i, rho_i = shock_scales(n_e=ne, T_i=T_i, z=z, b=b, ion_species=ion_species)

    if np.ndim(z) == 0 and np.ndim(n_i) == 0:
        d_i, rho_i = float(d_i), float(rho_i)

    return ShockCollisionality(
        ion=ion, electron=electron, d_i=d_i, rho_i=rho_i,
        knudsen=ion.stopping_range / d_i,
        knudsen_mfp=ion.mfp / d_i,
        gyro_ratio=ion.stopping_range / rho_i,
    )


def interpenetration(*, v_drift: np.ndarray | float,
                     n_field: np.ndarray | float, T_field: np.ndarray | float,
                     z_test: np.ndarray | float, z_field: np.ndarray | float,
                     piston: str = "Si", ambient: str = "Al",
                     n_e: np.ndarray | float | None = None,
                     T_e: np.ndarray | float | None = None) -> Slowing:
    """Interpenetration of a piston species into a background, e.g. Si into Al.

    ``slowing_down`` with the piston/chamber defaults; ``v_drift`` is the relative speed of
    the two populations.  Secondary to :func:`shock_collisionality` -- the shock's own
    Al-on-Al collisionality is the primary question -- but the same physics.
    """
    return slowing_down(v=v_drift, n_field=n_field, T_field=T_field,
                        z_test=z_test, z_field=z_field, test=piston, field=ambient,
                        n_e=n_e, T_e=T_e)
