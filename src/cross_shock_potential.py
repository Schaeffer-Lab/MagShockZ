"""Cross-shock electrostatic potential from the normal electric field E1.

The shock-normal electric field sets up a potential barrier across the front.
It (a) decelerates and partly reflects incoming ions and (b) accelerates and
heats electrons through the layer, so its magnitude relative to the ion ram
energy controls the ion-reflection fraction and the electron heating.

OSIRIS normalisation makes this clean: with E in units of
E_0 = m_e c omega_pe / e and x in c/omega_pe,

    e * phi(x) = - integral E1 dx      [in m_e c^2]

so the array returned by :func:`potential_profile` is already the potential
*energy* e*phi in m_e c^2 — directly comparable to the ion ram energy
1/2 |rqm_i| v_shock^2 (also m_e c^2).  All inputs/outputs are OSIRIS units.
"""

import numpy as np
import scipy.integrate


def potential_profile(e1: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Potential energy e*phi(x) [m_e c^2] = -∫ E1 dx, with e*phi(x[0]) = 0.

    The absolute zero is arbitrary; physically meaningful quantities are
    *differences* (see :func:`potential_jump`), which are reference-free.
    """
    e1 = np.asarray(e1, dtype=float)
    x = np.asarray(x, dtype=float)
    return -scipy.integrate.cumulative_trapezoid(e1, x, initial=0.0)


def potential_jump(e1: np.ndarray, x: np.ndarray,
                   upstream_mask: np.ndarray, downstream_mask: np.ndarray) -> float:
    """Cross-shock potential energy drop e*Δphi [m_e c^2] (downstream - upstream)."""
    e_phi = potential_profile(e1, x)
    return float(np.nanmean(e_phi[downstream_mask]) - np.nanmean(e_phi[upstream_mask]))


def reflection_parameter(e_delta_phi: float, abs_rqm_i: float, v_shock: float) -> float:
    """Ratio of the potential barrier to the ion ram energy.

    ``e*Δphi / (1/2 |rqm_i| v_shock^2)``.  Values approaching/exceeding 1 mean
    the cross-shock potential is strong enough to specularly reflect a
    significant fraction of the incoming ions.
    """
    ram = 0.5 * abs_rqm_i * v_shock**2
    return float(e_delta_phi / ram) if ram > 0.0 else float("nan")
