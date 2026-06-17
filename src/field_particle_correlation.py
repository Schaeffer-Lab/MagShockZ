"""Field-particle correlation (Klein & Howes) for an OSIRIS p1-x1 phase space.

The FPC technique localises *where in velocity space* the electric field does
work on a species, which is the fingerprint of the heating mechanism: a bipolar
signature straddling a resonant velocity points to Landau/transit-time-type
collisionless energy transfer, while a one-signed signature at the bulk points
to coherent acceleration (e.g. the cross-shock potential acting on the inflow).

For the parallel (shock-normal) electric field E1 the rate of change of the
species' phase-space energy density due to the field is

    C_E(u, x) = -1/2 * sign(rqm) * u^2 * E1(x) * df/du

The ``sign(rqm)`` carries the charge sign: OSIRIS rqm = m/q, so electrons
(rqm < 0) accelerate opposite to E1.  The mass itself cancels (energy density
1/2 |rqm| u^2 f against the field acceleration E1/rqm), so C_E is mass-free.
Velocity-integrating recovers the work rate j_species . E1 (see
:func:`velocity_integrated_rate`).

Non-relativistic convention: the phase-space momentum axis u is treated as the
velocity (u ≈ v in units of c), matching the OSIRIS p1-x1 diagnostic for the
slow ions and sub-relativistic electrons of these runs.  All arrays are plain
numpy in OSIRIS normalised units; loading/interpolation is done by the caller.
"""

import numpy as np
import scipy.integrate


def _check_shapes(f, u, x):
    f = np.asarray(f, dtype=float)
    u = np.asarray(u, dtype=float)
    x = np.asarray(x, dtype=float)
    if f.shape != (u.size, x.size):
        raise ValueError(
            f"f shape {f.shape} must be (len(u), len(x)) = ({u.size}, {x.size})"
        )
    return f, u, x


def energy_transfer_rate(f: np.ndarray, u: np.ndarray, e1: np.ndarray,
                         rqm: float, x: np.ndarray = None) -> np.ndarray:
    """C_E(u, x) = -1/2 sign(rqm) u^2 E1 df/du  — field work rate in (u, x).

    Parameters
    ----------
    f : np.ndarray
        Phase-space distribution f(u, x), shape (len(u), len(x)).
    u : np.ndarray
        Momentum/velocity axis [c], length matching f.shape[0].
    e1 : np.ndarray
        Shock-normal electric field on the phase-space x-grid, length f.shape[1].
    rqm : float
        Signed OSIRIS rqm (m/q) of the species; only its sign is used.
    x : np.ndarray, optional
        Only used for shape validation if provided.
    """
    u = np.asarray(u, dtype=float)
    f = np.asarray(f, dtype=float)
    e1 = np.asarray(e1, dtype=float)
    if f.shape != (u.size, e1.size):
        raise ValueError(
            f"f shape {f.shape} must be (len(u), len(e1)) = ({u.size}, {e1.size})"
        )
    dfdu = np.gradient(f, u, axis=0)
    u2 = (u**2)[:, None]
    return -0.5 * np.sign(rqm) * u2 * e1[None, :] * dfdu


def advective_flux(f: np.ndarray, u: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Phase-space advective (transport) term 1/2 u^3 df/dx in (u, x).

    This is the spatial-transport companion to the field term; it carries
    energy through the layer rather than transferring it to/from the field, and
    is shown alongside C_E to separate genuine heating from transport.
    """
    f, u, x = _check_shapes(f, u, x)
    dfdx = np.gradient(f, x, axis=1)
    return 0.5 * (u**3)[:, None] * dfdx


def velocity_integrated_rate(c_e: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Net field work rate vs x: ∫ C_E du  (= j_species . E1)."""
    return scipy.integrate.simpson(np.asarray(c_e, dtype=float),
                                   np.asarray(u, dtype=float), axis=0)


def field_particle_correlation(f: np.ndarray, u: np.ndarray, x: np.ndarray,
                               e1: np.ndarray, rqm: float) -> dict:
    """Convenience bundle: C_E(u,x), advective flux, and net rate vs x."""
    c_e = energy_transfer_rate(f, u, e1, rqm)
    return {
        "C_E": c_e,
        "advective": advective_flux(f, u, x),
        "net_rate": velocity_integrated_rate(c_e, u),
    }
