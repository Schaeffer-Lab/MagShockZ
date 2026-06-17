"""Reflected-ion diagnostics from an ion p1-x1 phase space.

Specular ion reflection off the cross-shock potential is a primary
collisionless ion-heating channel at supercritical magnetized shocks.  In the
shock frame the *incoming* ions stream toward the shock while a *reflected*
minority is turned back upstream; the reflected population gyrates in the
upstream field, gains energy, and is what shows up as downstream ion heating.

These functions split the distribution at p1 = v_shock (the shock-frame
boundary between forward- and backward-moving ions) and integrate each
sub-population with the masked :func:`moments.moment`.  ``incoming_sign`` is the
sign of the incoming ions' shock-frame velocity (= sign of the upstream bulk
velocity minus v_shock); the caller determines it from the data so the physics
here stays unambiguous.  Everything is in OSIRIS normalised units.
"""

import numpy as np

import moments


def _axis_index(phase_space, axis: str) -> int:
    return next(i for i in range(len(phase_space.axes)) if phase_space.axes[i].name == axis)


def population_masks(phase_space, v_shock: float, incoming_sign: int,
                     axis: str = "p1") -> tuple[np.ndarray, np.ndarray]:
    """Boolean masks over the momentum axis for (incoming, reflected) ions.

    Incoming ions have shock-frame velocity ``incoming_sign * (p - v_shock) > 0``
    (moving toward the shock); reflected ions have the opposite sign.
    """
    ax = _axis_index(phase_space, axis)
    p = np.linspace(phase_space.axes[ax].min, phase_space.axes[ax].max,
                    phase_space.axes[ax].size)
    shock_frame_v = incoming_sign * (p - v_shock)
    return shock_frame_v > 0.0, shock_frame_v < 0.0


def number_densities(phase_space, v_shock: float, incoming_sign: int,
                     axis: str = "p1") -> tuple[np.ndarray, np.ndarray]:
    """(n_incoming, n_reflected) density profiles [n_0] along x."""
    inc_mask, refl_mask = population_masks(phase_space, v_shock, incoming_sign, axis)
    n_inc = np.abs(moments.moment(phase_space, order=0, axis=axis, p_mask=inc_mask))
    n_refl = np.abs(moments.moment(phase_space, order=0, axis=axis, p_mask=refl_mask))
    return n_inc, n_refl


def reflected_fraction(phase_space, v_shock: float, incoming_sign: int,
                       axis: str = "p1") -> np.ndarray:
    """Reflected-ion number fraction n_reflected / n_total along x."""
    n_inc, n_refl = number_densities(phase_space, v_shock, incoming_sign, axis)
    total = n_inc + n_refl
    return np.divide(n_refl, total, out=np.zeros_like(total), where=total > 0.0)


def reflected_energy_density(phase_space, rqm: float, v_shock: float,
                             incoming_sign: int, axis: str = "p1") -> np.ndarray:
    """Shock-frame kinetic energy density of reflected ions [n_0 m_e c^2].

    u = 0.5 * |rqm| * ∫ (p - v_shock)^2 f dp  over the reflected sub-population.
    Evaluated from masked moments via
    ∫(p-v)^2 f dp = N * [ var + (mean - v)^2 ], where N, mean, var are the
    masked zeroth/first/second moments of the reflected population.
    """
    _, refl_mask = population_masks(phase_space, v_shock, incoming_sign, axis)
    N = np.abs(moments.moment(phase_space, order=0, axis=axis, p_mask=refl_mask))
    mean = moments.moment(phase_space, order=1, axis=axis, p_mask=refl_mask)
    var = np.maximum(moments.moment(phase_space, order=2, axis=axis, p_mask=refl_mask), 0.0)
    raw_second = N * (var + (mean - v_shock) ** 2)
    return 0.5 * abs(rqm) * raw_second


def infer_incoming_sign(bulk_velocity_upstream: float, v_shock: float) -> int:
    """Sign of the incoming ions' shock-frame velocity from the upstream bulk.

    Returns +1 or -1; the reflected population is the opposite sign.
    """
    return int(np.sign(bulk_velocity_upstream - v_shock)) or 1
