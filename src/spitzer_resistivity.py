"""Spitzer resistivity of the MagShockZ plasma, for choosing the WarpX ``plasma_resistivity``.

WarpX's hybrid (Ohm's-law) solver takes a single scalar resistivity ``eta`` [Ohm*m]
(``run.plasma_resistivity`` in a flash2warpx config); the magnetic diffusivity it actually
diffuses B with is ``D_m = eta / mu0``.  The physically-motivated value is the **Spitzer**
resistivity of the plasma, which varies spatially because the FLASH slice spans a wide
electron temperature (~10-40 eV in the bulk, up to ~1 keV in the laser channel) and mean
ionization ``Zbar`` (~4-14).  This module maps that variation so a defensible constant can
be picked.

Pure functions.  The Spitzer wrapper imports astropy/plasmapy **lazily** (as
``dimensionless_params`` does) so the module stays importable in the numpy-only CI layer;
``magnetic_diffusivity`` and ``warpx_electron_temperature`` are numpy-only and unit-tested.

plasmapy quirks handled here (so callers can pass whole FLASH-slice arrays):
  * ``plasmapy.formulary.Spitzer_resistivity`` requires a **scalar** ``z_mean`` and an
    **integer** charge in the ion species string that must not exceed the element's atomic
    number ("Al 14+" is invalid).  An array ``Zbar`` is therefore handled by binning cells
    into integer charge states and issuing one plasmapy call per state (array ``T``/``n``).
  * The ion identity barely enters the electron-ion resistivity (Al and Si agree to ~5
    significant figures for these parameters), so a single ion string covers both FLASH
    materials.  The default ``ion="Si"`` (atomic number 14) is chosen because it spans the
    full MagShockZ charge range 1..14 with a valid species string for every state.
"""

from __future__ import annotations

import warnings

import numpy as np

MU_0 = 4.0e-7 * np.pi  # vacuum permeability [H/m]


def magnetic_diffusivity(eta_ohm_m):
    """Magnetic (resistive) diffusivity ``D_m = eta / mu0``  [m^2/s].

    This is the coefficient in the resistive term of the induction equation that WarpX's
    hybrid B-solve advances (``dB/dt = ... + D_m grad^2 B``); ``eta_ohm_m`` is the Spitzer
    (or configured) resistivity in Ohm*m.  numpy-only; scalars and arrays both work.
    """
    return np.asarray(eta_ohm_m, dtype=float) / MU_0 if np.ndim(eta_ohm_m) \
        else float(eta_ohm_m) / MU_0


def warpx_electron_temperature(n_e_m3, n0_m3, Te0_eV, gamma):
    """WarpX's electron temperature under its adiabatic pressure closure  [eV].

    The hybrid solver does not evolve a kinetic electron temperature: it closes the electron
    pressure with a polytrope ``P_e = n_e T_e`` and ``T_e = Te0 * (n_e / n0)^(gamma-1)``
    about the reference density ``n0`` (``run.{Te_eV, n0_per_m3, gamma}``).  This is the
    *only* way electron temperature enters WarpX, so it is what a WarpX-consistent Spitzer
    resistivity must use (contrast the true, spatially-resolved FLASH ``Te``).

    numpy-only; scalars and arrays both work.  ``gamma == 1`` gives the isothermal ``Te0``
    everywhere.  Cells with non-positive density return 0 (the closure's ``n->0`` limit for
    ``gamma > 1``).
    """
    n_e = np.asarray(n_e_m3, dtype=float)
    ratio = np.where(n_e > 0.0, n_e / float(n0_m3), 0.0)
    Te = float(Te0_eV) * ratio ** (float(gamma) - 1.0)
    return Te if np.ndim(n_e_m3) else float(Te)


def spitzer_resistivity(T_e_eV, n_e_m3, z_mean, ion: str = "Si", method: str = "classical"):
    """Spitzer resistivity ``eta``  [Ohm*m], vectorized over FLASH-slice arrays.

    Thin wrapper over ``plasmapy.formulary.Spitzer_resistivity`` (electron-ion collisions),
    with the array/z_mean handling described in the module docstring.  ``T_e_eV``, ``n_e_m3``
    (electron density; the ion density ``n_e / Z`` is formed internally) and ``z_mean`` may be
    scalars or broadcastable arrays; the return matches (scalar in, scalar out).

    Cells with non-finite or non-positive ``T_e``/``n_e``/``z_mean`` return ``nan``.  plasmapy's
    strong-coupling / mild-relativity warnings are expected for HED parameters (min ln(Lambda)
    can dip to ~4 in the cold dense liner, thermal speed ~6% c at 1 keV) and are suppressed here;
    the caller can report ln(Lambda) separately if needed.
    """
    import astropy.units as u
    from plasmapy.formulary import Spitzer_resistivity
    from plasmapy.particles import atomic_number

    scalar_in = np.ndim(T_e_eV) == 0 and np.ndim(n_e_m3) == 0 and np.ndim(z_mean) == 0
    T = np.atleast_1d(np.asarray(T_e_eV, dtype=float))
    n = np.atleast_1d(np.asarray(n_e_m3, dtype=float))
    z = np.atleast_1d(np.asarray(z_mean, dtype=float))
    T, n, z = np.broadcast_arrays(T, n, z)

    eta = np.full(T.shape, np.nan, dtype=float)
    z_atom = int(atomic_number(ion))
    valid = np.isfinite(T) & (T > 0.0) & np.isfinite(n) & (n > 0.0) & np.isfinite(z) & (z > 0.0)
    if not valid.any():
        return float("nan") if scalar_in else eta

    # Integer charge state per cell (clamped to a valid, non-zero charge for the element),
    # so each plasmapy call uses one scalar z_mean + one valid "ion Z+" species string.
    z_int = np.clip(np.rint(z), 1, z_atom).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for Zc in np.unique(z_int[valid]):
            sub = valid & (z_int == Zc)
            n_i = (n[sub] / float(Zc)) / u.m**3          # ion density for e-i collisions
            rho = Spitzer_resistivity(
                T=T[sub] * u.eV, n=n_i,
                species=("e", f"{ion} {int(Zc)}+"), z_mean=float(Zc), method=method,
            )
            eta[sub] = np.asarray(rho.to(u.ohm * u.m).value, dtype=float)

    return float(eta[0]) if scalar_in else eta
