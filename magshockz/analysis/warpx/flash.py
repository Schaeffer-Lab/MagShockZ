"""The FLASH piston as measured, in the units the deck is matched against.

This is the counterpart of :class:`magshockz.init.warpx.units.FlashReference`, and the
distinction between the two is the point:

``FlashReference``   what the deck *imposes* — an integer charge state (``Al 6+``) chosen
                     when writing the spec, so its ambient ``n_e`` is 1.64x FLASH's.
``MeasuredPiston``   what FLASH *is* — the EOS ionization state it actually ran at
                     (Zbar 3.66 for this dataset), which is what a measurement of the
                     simulation has to report.

Both derive everything through :class:`~magshockz.init.warpx.units.Upstream`, so the two
answer the same questions with the same formulary and differ only in the ion they are
handed.  A fractional Zbar is not a plasmapy ``Particle``, so ``eos_ion`` builds a
``CustomParticle`` from the measured mass number and charge state.
"""

from __future__ import annotations

from dataclasses import dataclass

import astropy.units as u
from astropy.constants import e
from plasmapy.particles import CustomParticle
from plasmapy.particles.particle_class import ParticleLike

from magshockz.init.warpx import units

# Above this fast-magnetosonic Mach number a perpendicular shock reflects ions rather
# than dissipating resistively (Edmiston & Kennel 1984).
ION_REFLECTION_MACH = 2.76


def eos_ion(mass_number: float, charge_state: float) -> ParticleLike:
    """An ion at FLASH's EOS ionization state, which is generally fractional."""
    return CustomParticle(mass=mass_number * u.u, charge=charge_state * e.si)


@dataclass(frozen=True)
class MeasuredPiston:
    """A FLASH piston measured over a time window, with its ambient.

    ``piston_electron_density`` is the density just *behind* the front — what drives the
    shock — not the stagnated global peak, which never reaches the front.
    ``piston_scale_length`` is provenance only: the ideal-MHD piston edge is grid-sharp,
    so no fitted e-folding length describes the interface.
    """

    upstream: units.Upstream
    piston_electron_density: u.Quantity
    front_speed: u.Quantity
    piston_scale_length: u.Quantity
    spot_radius: u.Quantity
    window: u.Quantity
    source: str = ""

    @property
    def mach_alfven(self) -> float:
        return float((self.front_speed / self.upstream.alfven_speed).decompose())

    @property
    def mach_magnetosonic(self) -> float:
        return float((self.front_speed / self.upstream.fast_speed).decompose())

    @property
    def is_supercritical(self) -> bool:
        return self.mach_magnetosonic > ION_REFLECTION_MACH

    @property
    def contrast(self) -> float:
        """Piston electron density relative to the ambient."""
        return float((self.piston_electron_density
                      / self.upstream.electron_density).decompose())

    @property
    def spot_radius_over_di(self) -> float:
        return float((self.spot_radius / self.upstream.ion_skin_depth).decompose())

    @property
    def scale_length_over_di(self) -> float:
        return float((self.piston_scale_length / self.upstream.ion_skin_depth).decompose())

    @property
    def window_gyroperiods(self) -> float:
        return float((self.window / self.upstream.gyroperiod).decompose())

    def invariants(self) -> dict[str, float]:
        """The dimensionless state a reduced-mass deck is matched to."""
        return {
            "M_A": self.mach_alfven,
            "M_ms": self.mach_magnetosonic,
            "beta_e": float(self.upstream.beta_e),
            "beta_i": float(self.upstream.beta_i),
            "contrast": self.contrast,
            "r_spot/d_i": self.spot_radius_over_di,
            "t_window/T_ci": self.window_gyroperiods,
        }
