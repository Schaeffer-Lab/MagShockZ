"""Derived scales for a MagShockZ WarpX heater-driven piston run.

Only PRIMARY quantities live in the run's ``config.yaml``; everything else --
``omega_pe``, ``d_e``, ``d_i``, ``B0``, ``T_ci``, ``M_A``, the betas, ``dt``, the cell
counts -- is computed here, so there is one source of truth and no script can drift out
of sync with the deck.

Units are carried by astropy ``Quantity`` and the plasma formulas come from plasmapy's
``formulary``, so dimensions are checked by the library rather than asserted by a
variable name.  The ions are plasmapy particles: ``Particle("Al 6+")`` for the chamber
and ``Particle("Si 14+")`` for the ablated target.  The deck's reduced-mass counterparts
are ``CustomParticle``s carrying the same charge and a divided mass.

The deck runs at a REDUCED ion mass and an arbitrary reference density, so FLASH is
reproduced in dimensionless terms only.  Two knobs control the mapping:

``reduction_factor``
    Both ion masses are divided by it; charge states are untouched, so the
    piston/ambient mass-per-charge contrast (Si+14 is 2.24x lighter per charge than
    Al+6) survives.  Set indirectly, by naming the ambient's deck ``mass_per_charge``.

``vA_over_c``
    The system's speed relative to ``c``.  Set by CALIBRATION -- ``v_piston`` measured
    from a completed run divided by the FLASH Alfven Mach number being matched -- never
    by a model of how fast the heater ought to drive the slab.

Mass-per-charge ``m_i/(Z m_e)`` (OSIRIS's ``rqm``) is what every scale depends on:
``d_i/d_e = sqrt(m_i/(Z m_e))`` and ``v_A/c = (omega_ce/omega_pe)/sqrt(m_i/(Z m_e))``.
The bare ``m_i/m_e`` appears in neither.

Dimensionless quantities -- normalized lengths in ``d_e``/``d_i``, times in ``T_ci``,
``theta = kT/(mc^2)``, cell counts -- stay plain floats, since attaching a unit to a
pure number only obscures it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import astropy.units as u
from astropy.constants import c, e, eps0, m_e, mu0
from plasmapy.formulary import (
    Alfven_speed,
    Debye_length,
    beta,
    gyrofrequency,
    inertial_length,
    ion_sound_speed,
    plasma_frequency,
)
from plasmapy.particles import CustomParticle, Particle, ParticleLike

from .calibration import HeaterCalibration

#: Electron rest energy -- the normalization for every ``theta`` the deck writes.
ELECTRON_REST_ENERGY: u.Quantity = (m_e * c**2).to(u.eV)

#: Isothermal electrons, adiabatic ions -- the convention the FLASH and deck sound
#: speeds must share for ``M_ms`` to mean the same thing on both sides.
GAMMA_ELECTRON = 1.0
GAMMA_ION = 5.0 / 3.0

#: ``lambda_De/dx`` below which a cold upstream heats numerically (OSIRIS convergence
#: scan, CLAUDE.md).  The heater-off null control measured 3.7x ambient heating over a
#: full run at 0.0368, against 1210x with the heater on -- so this threshold holds.
DEBYE_PER_CELL_MIN = 0.03

#: ``v_th,e/c`` above which the heater's non-relativistic kicks stop imposing the
#: temperature they are handed: it kicks ``u = gamma*v`` with no gamma factor.
RELATIVISTIC_VTH_MAX = 0.3

BLOCKING_DEFAULT = 8

#: Cost reference: the 944 x 4720, 25 ppc/species, 4-species, 99326-step run measured at
#: 4 nodes x 6 h. Node-hours scale linearly in (macroparticles x steps).
_COST_REF_WORK = (944.0 * 4720.0) * 25.0 * 4.0 * 99326.0
_COST_REF_NODE_HOURS = 4.0 * 6.0

ALUMINIUM_6 = Particle("Al 6+")
SILICON_14 = Particle("Si 14+")


def as_particle(particle: ParticleLike) -> Particle | CustomParticle:
    """Normalize to a particle object.

    ``Particle()`` accepts only str/int/Particle, so the reduced-mass ions -- which are
    ``CustomParticle``s and not any real element -- have to pass through untouched.
    """
    if isinstance(particle, (Particle, CustomParticle)):
        return particle
    return Particle(particle)


def theta(temperature: u.Quantity, particle: ParticleLike) -> float:
    """``kT/(m c^2)`` for that particle -- exactly WarpX's ``u_std**2``."""
    mass = as_particle(particle).mass
    return float((temperature.to(u.J, equivalencies=u.temperature_energy())
                  / (mass * c**2)).decompose())


def mass_per_charge(particle: ParticleLike) -> float:
    """``m_i/(Z m_e)`` -- OSIRIS's ``rqm``, and what sets ``d_i`` and ``v_A``."""
    particle = as_particle(particle)
    return float((particle.mass / (particle.charge_number * m_e)).decompose())


def reduce_mass(particle: ParticleLike, reduction_factor: float) -> CustomParticle:
    """The deck's counterpart of an ion: same charge, mass divided by the factor."""
    particle = as_particle(particle)
    return CustomParticle(
        mass=particle.mass / reduction_factor,
        charge=particle.charge,
        symbol=f"{particle.symbol}/{reduction_factor:.6g}",
    )


# --------------------------------------------------------------------------- #
# a magnetized population: the derivation both sides share
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Upstream:
    """A uniform magnetized plasma -- FLASH's chamber IC or the deck's ambient.

    Both sides of the comparison instantiate this, so every dimensionless number is
    computed by ONE piece of code and a FLASH/deck disagreement can only come from the
    inputs.
    """

    ion: ParticleLike
    electron_density: u.Quantity
    magnetic_field: u.Quantity
    electron_temperature: u.Quantity
    ion_temperature: u.Quantity

    @property
    def ion_density(self) -> u.Quantity:
        """``n_i = n_e / Z``."""
        return self.electron_density / as_particle(self.ion).charge_number

    @property
    def alfven_speed(self) -> u.Quantity:
        return Alfven_speed(self.magnetic_field, self.ion_density, ion=self.ion)

    @property
    def sound_speed(self) -> u.Quantity:
        return ion_sound_speed(
            T_e=self.electron_temperature, T_i=self.ion_temperature, ion=self.ion,
            gamma_e=GAMMA_ELECTRON, gamma_i=GAMMA_ION)

    @property
    def fast_speed(self) -> u.Quantity:
        return (self.alfven_speed**2 + self.sound_speed**2) ** 0.5

    @property
    def beta_e(self) -> float:
        return float(beta(self.electron_temperature, self.electron_density,
                          self.magnetic_field))

    @property
    def beta_i(self) -> float:
        return float(beta(self.ion_temperature, self.ion_density, self.magnetic_field))

    @property
    def plasma_frequency(self) -> u.Quantity:
        return plasma_frequency(self.electron_density, "e-")

    @property
    def electron_skin_depth(self) -> u.Quantity:
        return inertial_length(self.electron_density, "e-")

    @property
    def ion_skin_depth(self) -> u.Quantity:
        return inertial_length(self.ion_density, self.ion)

    @property
    def gyrofrequency(self) -> u.Quantity:
        return gyrofrequency(self.magnetic_field, self.ion)

    @property
    def gyroperiod(self) -> u.Quantity:
        return (2.0 * math.pi * u.rad / self.gyrofrequency).to(u.s)

    @property
    def debye_length(self) -> u.Quantity:
        return Debye_length(self.electron_temperature, self.electron_density)

    @property
    def ion_gyroradius_over_skin_depth(self) -> float:
        """``rho_i/d_i = sqrt(beta_i/2)`` -- rides on a matched invariant."""
        return math.sqrt(max(self.beta_i, 0.0) / 2.0)

    @property
    def debye_over_skin_depth(self) -> float:
        """``lambda_De/d_e = v_th,e/c``."""
        return float((self.debye_length / self.electron_skin_depth).decompose())

    @property
    def omega_ce_over_omega_pe(self) -> float:
        """``= (v_A/c) sqrt(m_i/(Z m_e))``; preserved by the classic velocity-boost map."""
        return float((self.alfven_speed / c).decompose()) * math.sqrt(
            mass_per_charge(self.ion))

    @property
    def alfven_speed_over_c(self) -> float:
        return float((self.alfven_speed / c).decompose())


# --------------------------------------------------------------------------- #
# the FLASH side
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class FlashReference:
    """What the deck is tuned to match: FLASH's upstream plus its measured piston.

    ``piston_front_speed`` and ``piston_electron_density`` are MEASURED by
    ``scripts/flash_piston_profile.py``; the rest is the ``flash.par`` chamber IC.
    """

    upstream: Upstream
    piston_ion: ParticleLike
    piston_electron_density: u.Quantity
    piston_front_speed: u.Quantity
    spot_radius: u.Quantity
    window: u.Quantity
    source: str = ""

    @property
    def mach_alfven(self) -> float:
        return float((self.piston_front_speed / self.upstream.alfven_speed).decompose())

    @property
    def mach_magnetosonic(self) -> float:
        return float((self.piston_front_speed / self.upstream.fast_speed).decompose())

    @property
    def contrast(self) -> float:
        """Piston/ambient ELECTRON density ratio -- what sets the ram pressure."""
        return float(
            (self.piston_electron_density / self.upstream.electron_density).decompose())

    @property
    def spot_radius_over_di(self) -> float:
        return float((self.spot_radius / self.upstream.ion_skin_depth).decompose())

    @property
    def window_gyroperiods(self) -> float:
        return float((self.window / self.upstream.gyroperiod).decompose())

    @property
    def mass_per_charge_ratio(self) -> float:
        """Ambient/piston mass-per-charge -- the contrast the reduction preserves."""
        return mass_per_charge(self.upstream.ion) / mass_per_charge(self.piston_ion)

    def invariants(self) -> dict[str, float]:
        """The dimensionless numbers the reduced-mass deck reproduces."""
        return {
            "M_A": self.mach_alfven,
            "M_ms": self.mach_magnetosonic,
            "beta_e": self.upstream.beta_e,
            "beta_i": self.upstream.beta_i,
            "contrast": self.contrast,
            "r_spot/d_i": self.spot_radius_over_di,
            "rho_i/d_i": self.upstream.ion_gyroradius_over_skin_depth,
            "t_run/T_ci": self.window_gyroperiods,
        }


# --------------------------------------------------------------------------- #
# the deck side
# --------------------------------------------------------------------------- #

@dataclass
class DeckScales:
    """Everything the deck and the analysis need, derived from the config primaries."""

    reference_density: u.Quantity
    reduction_factor: float
    upstream: Upstream
    piston_ion: ParticleLike
    piston_electron_density: u.Quantity
    heater_temperature: u.Quantity
    piston_initial_temperature: u.Quantity

    piston_speed: u.Quantity

    cell_size_de: float
    slab_halfwidth_di: float
    slab_radius_di: float
    spot_radius_di: float
    domain_halfwidth_di: float
    transverse_halfwidth_di: float
    n_cells_x: int
    n_cells_z: int
    timestep: u.Quantity
    max_step: int
    run_gyroperiods: float

    flash: FlashReference | None = None
    warnings: list[str] = field(default_factory=list)

    # -- normalizations the deck is written in ------------------------------- #

    @property
    def electron_skin_depth(self) -> u.Quantity:
        return self.upstream.electron_skin_depth

    @property
    def ion_skin_depth(self) -> u.Quantity:
        return self.upstream.ion_skin_depth

    @property
    def gyroperiod(self) -> u.Quantity:
        return self.upstream.gyroperiod

    @property
    def steps_per_gyroperiod(self) -> float:
        return float((self.gyroperiod / self.timestep).decompose())

    @property
    def dt_omega_pe(self) -> float:
        return float((self.timestep * self.upstream.plasma_frequency
                      / u.rad).decompose())

    @property
    def di_over_de(self) -> float:
        return math.sqrt(mass_per_charge(self.upstream.ion))

    # -- the matched invariants ---------------------------------------------- #

    @property
    def piston_speed_over_c(self) -> float:
        return float((self.piston_speed / c).decompose())

    @property
    def mach_alfven(self) -> float:
        return float((self.piston_speed / self.upstream.alfven_speed).decompose())

    @property
    def mach_magnetosonic(self) -> float:
        return float((self.piston_speed / self.upstream.fast_speed).decompose())

    @property
    def contrast(self) -> float:
        return float(
            (self.piston_electron_density / self.upstream.electron_density).decompose())

    def invariants(self) -> dict[str, float]:
        return {
            "M_A": self.mach_alfven,
            "M_ms": self.mach_magnetosonic,
            "beta_e": self.upstream.beta_e,
            "beta_i": self.upstream.beta_i,
            "contrast": self.contrast,
            "r_spot/d_i": self.spot_radius_di,
            "rho_i/d_i": self.upstream.ion_gyroradius_over_skin_depth,
            "t_run/T_ci": self.run_gyroperiods,
        }

    # -- what the deck writes -------------------------------------------------- #

    @property
    def magnetic_field(self) -> u.Quantity:
        return self.upstream.magnetic_field

    @property
    def theta_e_ambient(self) -> float:
        return theta(self.upstream.electron_temperature, "e-")

    @property
    def theta_i_ambient(self) -> float:
        return theta(self.upstream.ion_temperature, self.upstream.ion)

    @property
    def theta_e_heater(self) -> float:
        """Heater setpoint, normalized to the ELECTRON rest mass -- the operator's own
        convention (``ParticleHeater.H``: ``m_foil_theta`` is ``kB T/(m_e c^2)``)."""
        return theta(self.heater_temperature, "e-")

    @property
    def theta_e_piston_initial(self) -> float:
        return theta(self.piston_initial_temperature, "e-")

    @property
    def theta_i_piston_initial(self) -> float:
        return theta(self.piston_initial_temperature, self.piston_ion)

    @property
    def heater_thermal_speed_over_c(self) -> float:
        return math.sqrt(self.theta_e_heater)

    @property
    def debye_per_cell(self) -> float:
        return self.upstream.debye_over_skin_depth / self.cell_size_de

    @property
    def cells_per_ion_skin_depth(self) -> float:
        return self.di_over_de / self.cell_size_de

    # -- bridging back to FLASH units ------------------------------------------ #

    def to_time(self, t_over_omega_pe: float) -> u.Quantity:
        """Sim time [1/omega_pe] -> the FLASH-equivalent time.

        The ion gyroperiod is the slowest MATCHED scale: one sim ``T_ci`` is one FLASH
        ``T_ci``.  Converting through ``1/omega_pe`` instead would be wrong by the
        deliberately broken mass ratio.
        """
        if self.flash is None:
            raise ValueError("to_time needs the FLASH reference")
        gyro = t_over_omega_pe * u.rad / (self.upstream.plasma_frequency * self.gyroperiod)
        return (gyro * self.flash.upstream.gyroperiod).to(u.ns)

    def to_length(self, x_over_de: float) -> u.Quantity:
        """Sim length [d_e] -> the FLASH-equivalent length, bridged through ``d_i``."""
        if self.flash is None:
            raise ValueError("to_length needs the FLASH reference")
        return (x_over_de / self.di_over_de * self.flash.upstream.ion_skin_depth).to(u.um)

    def cost(self, ppc_each_dim: tuple[int, int], n_species: int = 4) -> dict[str, float]:
        """Cell / macroparticle / step counts and a node-hour estimate.

        At fixed invariants and fixed ``lambda_De/dx``, 2D work scales as ``(v_A/c)^-4``
        and is INDEPENDENT of the mass ratio: cells per ``d_i`` and the Debye-limited
        ``dx`` both carry ``sqrt(m_i/(Z m_e))``, which cancels.
        """
        cells = float(self.n_cells_x * self.n_cells_z)
        per_species = float(ppc_each_dim[0] * ppc_each_dim[1])
        work = cells * per_species * n_species * self.max_step
        return {
            "cells": cells,
            "macroparticles": cells * per_species * n_species,
            "steps": float(self.max_step),
            "node_hours": _COST_REF_NODE_HOURS * work / _COST_REF_WORK,
        }


# --------------------------------------------------------------------------- #
# what the heating operator actually does
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class HeaterDrive:
    """The kick ``ParticleHeater`` applies, behind the ``theta`` setpoint.

    ``theta`` is NOT a target the operator servos towards -- it has no setpoint and no
    feedback.  It sets the amplitude of a momentum-space diffusion

        d<u_i^2>/dt = H,     H = 8 theta^{3/2} c^3 / (sqrt(m_i/m_e) * width)

    applied as an independent Gaussian kick of rms ``sqrt(H dt_heat)`` to each of
    ``ux, uy, uz`` every ``intervals`` steps (WarpX stores ``u = gamma v`` in m/s, so the
    kick is a velocity).  ``omega_pe`` cancels out of ``H``, which is why the deck's
    ``foil.n0`` does not affect the heating rate at all.

    Because ``<u_i^2>`` grows LINEARLY at ``H``, an isolated electron population reaches
    the setpoint in ``theta c^2 / H``.  That is a LOWER BOUND on the time to setpoint,
    not a prediction: the achieved temperature is a balance against cold injection and
    against the expansion doing work, so a run can clear this bar and still plateau
    short.  The runs on record do exactly that -- they saturate this timescale ~14x over
    and still reach only 71% and 80% of setpoint.  The empirical answer is
    ``calibration.CalibrationPoint.achieved_theta``; this is the operator-side number
    that says whether the heater had *time* to do its work.
    """

    setpoint_temperature: u.Quantity
    diffusion_rate: u.Quantity
    kick_per_application: float
    kick_speed: u.Quantity
    applications: int
    saturation_time: u.Quantity
    saturation_gyroperiods: float
    run_gyroperiods: float

    @property
    def has_time_to_reach_setpoint(self) -> bool:
        """Whether the run outlasts the heating-only climb to ``theta c^2``.

        Necessary, not sufficient -- see the class docstring.
        """
        return self.saturation_gyroperiods <= self.run_gyroperiods

    @property
    def saturation_margin(self) -> float:
        """How many times over the run covers the heating-only saturation time."""
        return self.run_gyroperiods / self.saturation_gyroperiods


def heater_drive(scales: DeckScales, *, intervals: int,
                 max_step: int | None = None) -> HeaterDrive:
    """Resolve the heater setpoint into the kick WarpX will actually apply.

    Parameters
    ----------
    intervals
        ``particle_heater.intervals``: steps between applications.
    max_step
        Run length in steps; defaults to the deck's own ``max_step``.
    """
    steps = scales.max_step if max_step is None else max_step
    piston_mass_ratio = (mass_per_charge(scales.piston_ion)
                         * round(as_particle(scales.piston_ion).charge_number))
    width = 2.0 * scales.slab_halfwidth_di * scales.ion_skin_depth
    setpoint = scales.theta_e_heater

    rate = (8.0 * setpoint**1.5 * c**3
            / (math.sqrt(piston_mass_ratio) * width)).to(u.m**2 / u.s**3)
    kick_speed = ((rate * intervals * scales.timestep) ** 0.5).to(u.m / u.s)
    saturation_time = (setpoint * c**2 / rate).to(u.s)

    return HeaterDrive(
        setpoint_temperature=scales.heater_temperature,
        diffusion_rate=rate,
        kick_per_application=float((kick_speed / c).decompose()),
        kick_speed=kick_speed,
        applications=steps // max(intervals, 1),
        saturation_time=saturation_time,
        saturation_gyroperiods=float((saturation_time / scales.gyroperiod).decompose()),
        run_gyroperiods=scales.run_gyroperiods,
    )


def _cell_count(extent_de: float, cell_size_de: float, blocking: int) -> int:
    """Cells spanning ``extent_de``, rounded up to AMReX's blocking factor."""
    n = math.ceil(extent_de / cell_size_de)
    if blocking > 1:
        n = math.ceil(n / blocking) * blocking
    return max(n, blocking)


def derive(
    flash: FlashReference,
    *,
    reference_density: u.Quantity,
    ambient_mass_per_charge: float,
    ambient_electron_temperature: u.Quantity,
    ambient_ion_temperature: u.Quantity,
    contrast: float,
    piston_initial_temperature: u.Quantity,
    calibration: HeaterCalibration,
    cell_size_de: float,
    slab_halfwidth_di: float,
    slab_radius_di: float | None = None,
    spot_radius_di: float | None = None,
    domain_halfwidth_di: float | None = None,
    transverse_halfwidth_di: float | None = None,
    run_gyroperiods: float | None = None,
    cfl: float = 0.75,
    blocking: int = BLOCKING_DEFAULT,
    z_margin: float = 1.35,
) -> DeckScales:
    """Config primaries -> every derived scale, in one visible chain.

    ``ambient_mass_per_charge`` is the ONE free physics knob: it sets the electron/ion
    scale separation ``d_i/d_e = sqrt(m/Ze)`` and, through ``v_A``, the cost.  Everything
    else is pinned by the FLASH targets and the measured heater calibration:

    1. ``reduction_factor`` = real ambient ``m/Ze`` / the requested deck value.  Both ion
       masses are divided by it; charge states are untouched, so Si+14 stays 2.24x
       lighter per charge than Al+6.
    2. ``v_A`` from matching ``beta_e`` at the ``flash.par`` electron temperature:
       ``beta_e = 2 Z kT_e / ((m_i + Z m_e) v_A^2)``.  Matching ``beta_e`` this way
       matches ``beta_i``, ``M_ms`` and ``rho_i/d_i`` simultaneously, because all three
       scale as ``1/(m/Ze)`` at fixed temperature and ``v_A``.
    3. ``B0 = v_A sqrt(mu0 rho)``.
    4. The REQUIRED piston speed is ``M_A_flash * v_A``; the heater setpoint that
       delivers it comes from the empirical :mod:`~warpx.calibration` fit, since the
       operator has no setpoint feedback and no closure predicts its response.
    5. Geometry is stated in ``d_i`` and converted with ``d_i/d_e = sqrt(m/Ze)``.
    """
    if reference_density <= 0 * reference_density.unit:
        raise ValueError(f"reference_density must be positive, got {reference_density!r}")
    if ambient_mass_per_charge <= 0.0:
        raise ValueError(
            f"ambient_mass_per_charge must be positive, got {ambient_mass_per_charge!r}")

    warnings: list[str] = []

    reduction_factor = mass_per_charge(flash.upstream.ion) / ambient_mass_per_charge
    ambient_ion = reduce_mass(flash.upstream.ion, reduction_factor)
    piston_ion = reduce_mass(flash.piston_ion, reduction_factor)

    # beta_e = n_e kT_e / (rho v_A^2 / 2) with rho = n_i (m_i + Z m_e), so holding the
    # FLASH beta_e at the flash.par temperature determines v_A outright.
    charge = ambient_ion.charge_number
    ion_density = reference_density / charge
    mass_density = ion_density * (ambient_ion.mass + charge * m_e)
    electron_energy = ambient_electron_temperature.to(
        u.J, equivalencies=u.temperature_energy())
    alfven_speed = ((2.0 * reference_density * electron_energy
                     / (flash.upstream.beta_e * mass_density)) ** 0.5).to(u.m / u.s)
    magnetic_field = (alfven_speed * (mu0 * mass_density) ** 0.5).to(u.T)

    upstream = Upstream(
        ion=ambient_ion,
        electron_density=reference_density,
        magnetic_field=magnetic_field,
        electron_temperature=ambient_electron_temperature,
        ion_temperature=ambient_ion_temperature,
    )

    piston_speed = flash.mach_alfven * alfven_speed
    piston_mass_per_charge = mass_per_charge(piston_ion)
    heater_theta = calibration.heater_theta(piston_speed, piston_mass_per_charge)
    heater_temperature = heater_theta * ELECTRON_REST_ENERGY

    extrapolation = calibration.extrapolation_warning(
        heater_theta, piston_mass_per_charge)
    if extrapolation is not None:
        warnings.append(extrapolation)

    thermal_speed_over_c = math.sqrt(heater_theta)
    if thermal_speed_over_c > RELATIVISTIC_VTH_MAX:
        warnings.append(
            f"the heater setpoint needed for v_piston = {piston_speed.to(u.km/u.s):.4g} "
            f"puts the piston electrons at v_th/c = {thermal_speed_over_c:.2f}, above "
            f"{RELATIVISTIC_VTH_MAX}: the heater kicks u = gamma*v with no gamma "
            f"factor, so it stops imposing the temperature it is given.")

    if contrast <= 1.0:
        warnings.append(
            f"contrast {contrast:.3g} <= 1 -- the piston is not denser than the ambient.")

    d_e = upstream.electron_skin_depth
    di_over_de = math.sqrt(mass_per_charge(ambient_ion))

    run_gyro = flash.window_gyroperiods if run_gyroperiods is None else float(run_gyroperiods)
    timestep = (cfl * cell_size_de * d_e / (c * math.sqrt(2.0))).to(u.s)
    max_step = math.ceil(float((run_gyro * upstream.gyroperiod / timestep).decompose()))

    travel_di = float((piston_speed * run_gyro * upstream.gyroperiod
                       / upstream.ion_skin_depth).decompose())
    needed_z = z_margin * (slab_halfwidth_di + travel_di)
    if domain_halfwidth_di is None:
        domain_halfwidth_di = needed_z
    elif domain_halfwidth_di < needed_z:
        warnings.append(
            f"domain_halfwidth_di = {domain_halfwidth_di:.1f} is below the "
            f"{needed_z:.1f} d_i the front needs over {run_gyro:.3f} T_ci -- it will "
            f"wrap through the periodic boundary.")

    if spot_radius_di is None:
        spot_radius_di = flash.spot_radius_over_di
    # The target slab is a finite PATCH, not a sheet spanning the domain: FLASH's piston
    # is a plume ablated from a 500 um spot, so a slab uniform in x would expand as a
    # quasi-planar foil and could not decompress radially.
    if slab_radius_di is None:
        slab_radius_di = spot_radius_di

    if transverse_halfwidth_di is None:
        transverse_halfwidth_di = max(4.0 * slab_radius_di, 4.0)
    elif transverse_halfwidth_di < 2.0 * slab_radius_di:
        warnings.append(
            f"transverse_halfwidth_di = {transverse_halfwidth_di:.1f} is under 2 slab "
            f"radii ({2 * slab_radius_di:.1f} d_i): the periodic images of the piston "
            f"patch overlap it, so the expansion is not radially free.")

    scales = DeckScales(
        reference_density=reference_density,
        reduction_factor=reduction_factor,
        upstream=upstream,
        piston_ion=piston_ion,
        piston_electron_density=contrast * reference_density,
        heater_temperature=heater_temperature,
        piston_initial_temperature=piston_initial_temperature,
        piston_speed=piston_speed,
        cell_size_de=cell_size_de,
        slab_halfwidth_di=slab_halfwidth_di,
        slab_radius_di=slab_radius_di,
        spot_radius_di=spot_radius_di,
        domain_halfwidth_di=domain_halfwidth_di,
        transverse_halfwidth_di=transverse_halfwidth_di,
        n_cells_x=_cell_count(2.0 * transverse_halfwidth_di * di_over_de,
                              cell_size_de, blocking),
        n_cells_z=_cell_count(2.0 * domain_halfwidth_di * di_over_de,
                              cell_size_de, blocking),
        timestep=timestep,
        max_step=max_step,
        run_gyroperiods=run_gyro,
        flash=flash,
        warnings=warnings,
    )

    if scales.debye_per_cell < DEBYE_PER_CELL_MIN:
        suggested = math.floor(1e3 * cell_size_de * scales.debye_per_cell
                               / DEBYE_PER_CELL_MIN) / 1e3
        warnings.append(
            f"lambda_De/dx = {scales.debye_per_cell:.4f} is below {DEBYE_PER_CELL_MIN} "
            f"-- the ambient (T_e = {ambient_electron_temperature.to(u.eV):.4g}) risks "
            f"numerical grid heating. Set cell_size_de to {suggested:.3f} or less.")

    return scales
