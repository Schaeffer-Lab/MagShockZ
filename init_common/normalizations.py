"""Canonical FLASH -> OSIRIS normalization constants.

This is the single source of truth for the plasma normalizations shared by both
initialization entrypoints:

  * init_python/FLASH_OSIRIS_define.py  (full-2D, interpolator / python-coupled decks)
  * init_nopython/fitting_functions.py  (1D / quasi-1D, math-function decks)

Historically each file recomputed omega_pe, B_norm, E_norm, v_norm and the
thermal-velocity norms independently.  The formulas were identical but had
already started to drift in *structure* (per-component vs. collective keys), so
any future fix had to be made in two places.  Keep the physics here.

All quantities are returned as ``yt`` unit-quantities (``unyt``) so that callers
can divide raw FLASH fields by them and get a dimensionless OSIRIS-normalized
result, exactly as before.
"""

from dataclasses import dataclass

import numpy as np
import yt


@dataclass
class Normalizations:
    """Scalar normalization constants for a given reference density / mass ratio.

    Attributes are ``yt`` quantities with explicit units:

      n0        reference electron density            [cm^-3]
      omega_pe  electron plasma frequency             [1/s]
      B         magnetic field normalization          [Gauss]
      E         electric field normalization          [statV/cm]
      v         fluid/ion velocity normalization       [cm/s]   (= c / sqrt(rqm_factor))
      vth_ele   electron thermal-velocity norm        [cm/s]   (= c)
      vth_ion   ion thermal-velocity norm (legacy)    [cm/s]   (= v)

    Thermal-velocity normalization (current scheme): BOTH electron and ion OSIRIS thermal
    velocities normalize to c, because the charge state and the sqrt(rqm_factor) mass
    reduction are baked into the FIELDS in src/my_plugins.py:
        uth_e = vth_osiris_ele / c = sqrt(k_B Te / (m_e c^2))
        uth_i = vth_osiris_ion / c = sqrt(rqm_factor * k_B T_i / ((m_i/Z) c^2))
    With these OSIRIS reads back T = rqm * me_c2 * uth^2 = the FLASH temperature for each
    species (Z = the charge state used to set that species' deck rqm), so T_e/T_i is conserved.

    ``vth_ion`` here (= v) is the LEGACY full-mass/physical-velocity norm, NOT used by the
    corrected writers; kept only for backward compatibility.
    """

    n0: object
    omega_pe: object
    B: object
    E: object
    v: object
    vth_ele: object
    vth_ion: object
    rqm_factor: float


def compute_norms(reference_density_cc: float, rqm_factor: float) -> Normalizations:
    """Compute the OSIRIS normalization constants.

    Parameters
    ----------
    reference_density_cc : float
        Reference electron density n0 in cm^-3.
    rqm_factor : float
        Artificial mass-ratio reduction factor (m_i/m_e is divided by this).
    """
    n0 = reference_density_cc * yt.units.cm**-3

    omega_pe = np.sqrt(
        n0 * yt.units.electron_charge_mks**2
        / (yt.units.eps_0 * yt.units.electron_mass)
    ).to('1/s')

    B_norm = (
        omega_pe * yt.units.electron_mass * yt.units.speed_of_light
        / yt.units.elementary_charge
    ).to('Gauss')

    v_norm = (yt.units.speed_of_light / np.sqrt(rqm_factor)).to('cm/s')

    E_norm = (
        omega_pe * yt.units.electron_mass * yt.units.speed_of_light
        / yt.units.elementary_charge / np.sqrt(rqm_factor)
    ).to('statV/cm')

    vth_ele_norm = yt.units.speed_of_light
    vth_ion_norm = v_norm

    return Normalizations(
        n0=n0,
        omega_pe=omega_pe,
        B=B_norm,
        E=E_norm,
        v=v_norm,
        vth_ele=vth_ele_norm,
        vth_ion=vth_ion_norm,
        rqm_factor=rqm_factor,
    )


def reference_check_lines(norms: Normalizations, B_gauss: float = 1e5):
    """Return human-readable lines verifying the B-field normalization round-trip.

    Uses the two independent OSIRIS conversion constants (5.681e-8 from omega_pe,
    3.204e-3 from sqrt(n0)) that both entrypoints previously printed inline.  They
    should agree with each other and recover ``B_gauss``.
    """
    B_test = (B_gauss * yt.units.Gauss / norms.B).to(yt.units.dimensionless)
    via_omega = 5.681e-8 * B_test * norms.omega_pe
    via_density = 3.204e-3 * B_test * np.sqrt(norms.n0.value)
    return [
        f"{B_gauss:.3e} Gauss -> {B_test:.3e} OSIRIS units",
        f"  back via omega_pe : {via_omega:.3e} Gauss",
        f"  back via sqrt(n0) : {via_density:.3e} Gauss",
    ]
