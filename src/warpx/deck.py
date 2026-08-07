"""Render a ``schema: heater_pic_2d`` ParmParse deck, and read one back.

The mapping spec -> deck is one-way.  To prove the deck on disk still means what the
spec says, :func:`parse_inputs` -> :func:`resolve_constants` -> :func:`key_params`
resolves the deck's own ``my_constants`` in a restricted ``eval`` and compares the
resulting NUMBERS, so the check is independent of formatting, comments, and whether a
length was written ``2.*di`` or ``20.*de``.

``my_constants`` is symbolic wherever a constant follows from another, so the deck states
its own physics: ``B0`` is written as the ``v_A`` it came from, the ambient temperatures
as the pressure balance that defines the betas, and every length in ``di``.  That puts
four of the matched invariants in the deck as named constants, checkable without running
Python.  The floats that remain carry 10 significant digits because those expressions
chain off them -- ``theta_e_amb`` reaches ``vA`` through ``B0`` *squared*.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import astropy.units as u
from astropy.constants import c, e, eps0, m_e, mu0

from . import units

SPECIES_NAMES = ("piston_electrons", "piston_ions", "amb_electrons", "amb_ions")

#: WarpX resolves ``my_constants`` against these; ``verify`` needs the same namespace.
#: The WarpX build ships CODATA-2022 against astropy's CODATA-2018, a ~1e-9 shift in
#: every resolved length -- hence :data:`VERIFY_RTOL` rather than exact comparison.
CONSTS: dict[str, float] = {
    "clight": float(c.si.value),
    "m_e": float(m_e.si.value),
    "q_e": float(e.si.value),
    "epsilon0": float(eps0.si.value),
    "mu0": float(mu0.si.value),
    "pi": math.pi,
}

VERIFY_RTOL = 1e-6

_MATH = {"sqrt": math.sqrt, "abs": abs, "exp": math.exp, "log": math.log,
         "sin": math.sin, "cos": math.cos, "tan": math.tan, "pow": pow}


# --------------------------------------------------------------------------- #
# formatting
# --------------------------------------------------------------------------- #

def _num(x: float) -> str:
    """Format a number the WarpX way: integers keep a trailing dot, ``1e+18`` -> ``1e18``."""
    x = float(x)
    if x == int(x) and abs(x) < 1e6:
        return f"{int(x)}."
    return repr(x).replace("e+", "e")


def _element(particle: units.ParticleLike) -> str:
    """The species name without the reduced-mass suffix ``reduce_mass`` appends."""
    return str(getattr(particle, "symbol", particle)).split("/", 1)[0]


def _intervals(period: int, stop_step: int | None) -> str:
    """A bare period, or ``0:stop:period`` when the drive is time-gated.

    ``stop_step`` rounds DOWN to a multiple of ``period`` so the last application lands
    exactly on the boundary.  Both operators parse this with the stock ablastr
    ``IntervalsParser``, so gating the drive needs no engine change.
    """
    if stop_step is None:
        return f"{period}"
    return f"0:{stop_step - stop_step % period}:{period}"


def _parse_intervals(text: str) -> tuple[int, int | None]:
    """Inverse of :func:`_intervals`."""
    text = text.strip().strip('"').strip("'")
    if ":" not in text:
        return int(float(text)), None
    parts = text.split(":")
    if len(parts) != 3:
        raise ValueError(f"unsupported intervals spec {text!r}")
    return int(float(parts[2])), (int(float(parts[1])) if parts[1].strip() else None)


def drive_stop_step(config: dict[str, Any], scales: units.DeckScales) -> int | None:
    """Step at which both operators switch off, or ``None`` to drive the whole run.

    Stated in the run's own normalized time (``operators.drive_stop_gyroperiods``) so it
    survives changes to ``cell_size_de`` and ``cfl``.
    """
    stop = (config.get("operators") or {}).get("drive_stop_gyroperiods")
    if not stop:
        return None
    return int(float(stop) * scales.steps_per_gyroperiod)


# --------------------------------------------------------------------------- #
# spec -> deck
# --------------------------------------------------------------------------- #

def _constants_block(scales: units.DeckScales) -> str:
    """The ``my_constants`` namespace: the deck's whole physical state, symbolically."""
    ambient_ion, piston_ion = scales.upstream.ion, scales.piston_ion
    # CustomParticle carries the charge as a float, so 14 comes back as 14.000000000000002
    # and the reduced-mass symbol as "Si 14+/163.928"; neither belongs in the deck.
    z_amb = round(ambient_ion.charge_number)
    z_piston = round(piston_ion.charge_number)
    return f"""\
# --- reference state: everything below is derived from these ------------------
my_constants.n0      = {scales.reference_density.to_value(u.m**-3):.6e}   # ambient ELECTRON density [m^-3]
my_constants.z_amb   = {_num(z_amb)}            # ambient charge state ({_element(ambient_ion)})
my_constants.z_pist  = {_num(z_piston)}           # piston charge state ({_element(piston_ion)})
my_constants.mpc_amb  = {units.mass_per_charge(ambient_ion):.10g}       # m_i/(Z m_e), REDUCED on purpose
my_constants.mpc_pist = {units.mass_per_charge(piston_ion):.10g}   # reduced by the same factor
my_constants.mi_amb  = mpc_amb*z_amb*m_e
my_constants.mi_pist = mpc_pist*z_pist*m_e
my_constants.wpe     = sqrt(n0*q_e^2/(epsilon0*m_e))
my_constants.de      = clight/wpe
my_constants.di      = de*sqrt(mpc_amb)   # ambient ion skin depth: d_i/d_e = sqrt(m/Ze)
my_constants.namb    = 1.*n0              # ambient electron density
my_constants.nt      = {scales.contrast:.10g}*n0   # piston electron density -> contrast, MATCHED
my_constants.vA      = {scales.upstream.alfven_speed_over_c:.10e}*clight   \
# ambient Alfven speed -> M_A = {scales.mach_alfven:.3f}
# rho = n_i (m_i + Z m_e): the electron mass is 1/(m/Ze) of the ion mass density, which
# is 2% here and NOT negligible at a reduced mass ratio.
my_constants.B0      = vA*sqrt(mu0*(namb/z_amb)*(mi_amb+z_amb*m_e))   \
# out-of-plane field ({scales.magnetic_field.to_value(u.T) * 1e4:.1f} G)

# --- geometry, in the ion units the invariants are stated in -------------------
my_constants.slab    = {scales.slab_halfwidth_di:.6f}*di    # piston slab half-thickness along z
my_constants.rslab   = {scales.slab_radius_di:.8f}*di  # slab is a finite PATCH, not a sheet
my_constants.rspot   = {scales.spot_radius_di:.8f}*di  # heating spot -> r_spot/d_i, MATCHED
my_constants.xhalf   = {scales.transverse_halfwidth_di * scales.di_over_de:.6f}*de
my_constants.zhalf   = {scales.domain_halfwidth_di * scales.di_over_de:.6f}*de

# --- temperatures as theta = kT/(m c^2), which is exactly WarpX's u_std^2 -------
# The ambient thetas are written as the pressure balance they came from, so the two
# MATCHED betas are named in the deck: beta = n kT/(B^2/2 mu0).  Note the electron beta
# carries n_e and the ion beta n_i = n_e/Z -- they differ by Z at a real charge state.
my_constants.beta_e  = {scales.upstream.beta_e:.10g}      # MATCHED invariant (FLASH)
my_constants.beta_i  = {scales.upstream.beta_i:.10g}     # MATCHED invariant (FLASH)
my_constants.theta_e_amb = beta_e*B0^2/(2*mu0*namb*m_e*clight^2)
my_constants.theta_i_amb = beta_i*B0^2/(2*mu0*(namb/z_amb)*mi_amb*clight^2)
my_constants.theta_e_heat = {scales.theta_e_heater:.10g}   \
# heater setpoint -> v_piston = {scales.piston_speed_over_c:.4g} c
my_constants.theta_e_cold = {scales.theta_e_piston_initial:.10g}     # piston load, pre-heating
my_constants.theta_i_cold = theta_e_cold*m_e/mi_pist  # same physical T as the electrons"""


def _time_and_grid_block(config: dict[str, Any], scales: units.DeckScales, *,
                         max_step: int) -> str:
    """Step count, CFL, the AMR grid and the domain extent."""
    numerics = config["numerics"]
    return f"""\
max_step      = {max_step}
warpx.cfl     = {float(numerics['cfl']):.6g}
warpx.verbose = 1

amr.n_cell        = {scales.n_cells_x} {scales.n_cells_z}
amr.max_level     = 0
amr.max_grid_size = {int(numerics['max_grid_size'])}
amr.blocking_factor = {int(numerics['blocking_factor'])}
geometry.dims     = 2
geometry.prob_lo  = -xhalf -zhalf
geometry.prob_hi  = xhalf zhalf"""


def _boundary_block() -> str:
    """Fully periodic on both axes -- not a choice this deck offers.

    A uniform applied B on the grid requires periodic boundaries ("do not use any other
    boundary condition than periodic", ``Docs/source/usage/parameters.rst``), and the
    symmetric slab, the domain sizing and the run-length budget all follow from it.
    ``config.load`` raises on any other request; a walled variant needs its own schema
    and its own renderer.
    """
    return """\
# Fully periodic -- required for the uniform applied B below, which in turn is why the
# slab is symmetric and TWO fronts propagate, one in each z direction.
boundary.field_lo    = periodic periodic
boundary.field_hi    = periodic periodic
boundary.particle_lo = periodic periodic
boundary.particle_hi = periodic periodic"""


def _field_block(config: dict[str, Any]) -> str:
    """The uniform out-of-plane B0: a perpendicular-shock geometry."""
    return f"""\
algo.particle_shape = {int(config['numerics']['particle_shape'])}

# Uniform out-of-plane magnetization, normal to the piston's expansion -- FLASH's magz.
# The plasma is loaded at rest, so there is no motional E field to impose.
warpx.B_ext_grid_init_style = parse_B_ext_grid_function
warpx.Bx_external_grid_function(x,y,z) = "0."
warpx.By_external_grid_function(x,y,z) = "B0"
warpx.Bz_external_grid_function(x,y,z) = "0.\""""


def _species_block(name: str, *, identity: list[str], density: str, theta: str,
                   ppc_each_dim: tuple[int, int]) -> str:
    """One ParmParse species: density profile + Maxwellian at rest.

    ``u_std`` is written ``sqrt(<theta>)`` rather than as a decimal because that IS
    WarpX's convention -- ``u_std^2 = kT/(m c^2)`` -- which makes the electron/ion
    asymmetry self-evident instead of two unrelated-looking numbers.
    """
    return "\n".join([
        *identity,
        f"{name}.injection_style = NUniformPerCell",
        f"{name}.num_particles_per_cell_each_dim = {ppc_each_dim[0]} {ppc_each_dim[1]}",
        f"{name}.profile = parse_density_function",
        f'{name}.density_function(x,y,z) = "{density}"',
        f"{name}.density_min = 1.",
        f"{name}.momentum_distribution_type = maxwellian",
        f"{name}.maxwellian_u_mean_distribution_type = constant",
        f"{name}.ux_mean = 0.",
        f"{name}.uy_mean = 0.",
        f"{name}.uz_mean = 0.",
        f"{name}.maxwellian_u_std_distribution_type = constant",
        *[f"{name}.{axis}_std = sqrt({theta})" for axis in ("ux", "uy", "uz")],
    ])


def _species_section(ppc_each_dim: tuple[int, int]) -> str:
    """All four species, split piston/ambient by the finite slab patch.

    The slab is a PATCH of radius ``rslab``, not a sheet spanning the domain: FLASH's
    piston is a plume ablated from a finite spot, and a slab uniform in x would expand
    as a quasi-planar foil with no way to decompress radially.
    """
    inside = "(abs(z)<slab)*(abs(x)<rslab)"
    return "\n\n".join([
        f"particles.species_names = {' '.join(SPECIES_NAMES)}",

        "# --- piston / target slab: heated and continuously replenished ---\n"
        + _species_block("piston_electrons",
                         identity=["piston_electrons.species_type = electron"],
                         density=f"nt*{inside}", theta="theta_e_cold",
                         ppc_each_dim=ppc_each_dim),
        _species_block("piston_ions",
                       identity=["piston_ions.charge = z_pist*q_e",
                                 "piston_ions.mass   = mi_pist"],
                       density=f"(nt/z_pist)*{inside}", theta="theta_i_cold",
                       ppc_each_dim=ppc_each_dim),

        "# --- ambient: fills everything outside the patch, NEITHER heated nor injected ---\n"
        + _species_block("amb_electrons",
                         identity=["amb_electrons.species_type = electron"],
                         density=f"namb*(1.-{inside})", theta="theta_e_amb",
                         ppc_each_dim=ppc_each_dim),
        _species_block("amb_ions",
                       identity=["amb_ions.charge = z_amb*q_e",
                                 "amb_ions.mass   = mi_amb"],
                       density=f"(namb/z_amb)*(1.-{inside})", theta="theta_i_amb",
                       ppc_each_dim=ppc_each_dim),
    ])


def _heater_block(config: dict[str, Any], scales: units.DeckScales, *,
                  stop_step: int | None, no_heater: bool) -> str:
    """The ``ParticleHeater`` block, or the null control's explanation of its absence."""
    if no_heater:
        # NULL CONTROL. Everything else -- domain, dt, steps, ppc, species, injector --
        # is identical to production, so the ambient <u^2> history has one possible
        # source left: the discretisation. With no energy input the slab cannot expand,
        # so there is no shock and no physical precursor to preheat the upstream.
        return """\
# --- laser-ablation surrogate: HEATER DELIBERATELY OMITTED (null control) ------
# The injector below still runs, so the macroparticle count and load match the
# production deck exactly. Only the energy source is gone. If the ambient <u^2>
# still climbs here, production's rise is numerical grid heating and dx must come
# down; if it stays flat, that rise is physical."""

    heater = config["operators"]["heater"]
    gate = "" if stop_step is None else (
        "# Drive window: FLASH's laser is a finite pulse, so heating for the whole run\n"
        "# compares a continuously-driven piston against a ballistic one. Hard on/off,\n"
        "# not FLASH's ramp: theta is a scalar fixed at init.\n")
    # foil.mass_ratio is the PISTON's bare m_i/m_e -- it enters H as 1/sqrt(mass_ratio),
    # setting the sound speed at which the heated slab expands. It is NOT m/(Z m_e):
    # at Z = 14 the two differ by a factor 14.
    piston_mass_ratio = (units.mass_per_charge(scales.piston_ion)
                         * round(units.as_particle(scales.piston_ion).charge_number))
    # theta is an AMPLITUDE, not a servo setpoint: the operator has no feedback. Spell
    # out the kick it resolves to, so the deck states what is actually done to a particle.
    drive = units.heater_drive(scales, intervals=int(heater["intervals"]))
    return f"""\
# --- laser-ablation surrogate: heat and replenish the piston slab ONLY --------
# The ambient must stay cold, so it is absent from both operator species lists.
#
# theta below is NOT a temperature the operator servos towards -- it has no setpoint and
# no feedback. It sets the amplitude of a momentum-space diffusion applied as an
# independent Gaussian kick to each of ux, uy, uz every `intervals` steps:
#
#   d<u_i^2>/dt = H = 8 theta^(3/2) c^3 / (sqrt(foil.mass_ratio) * (hi-lo))
#                   = {drive.diffusion_rate.to_value('m2/s3'):.4e} m^2/s^3
#   rms kick    = sqrt(H * intervals * dt)
#               = {drive.kick_per_application:.4e} c = \
{drive.kick_speed.to_value('km/s'):.4g} km/s per application
#   applied     {drive.applications} times over the run
#
# theta = {scales.theta_e_heater:.5g} is {scales.heater_temperature.to_value('keV'):.4g} keV.
# omega_pe cancels out of H, so foil.n0 below does NOT affect the heating rate -- the
# operator reads it, builds omega_pe from it, and divides it straight back out. The
# density reaches H only as GEOMETRY: (hi-lo) is a physical length, and this deck states
# it in d_i, so changing reference.density_per_m3 rescales d_i and moves H. H*(hi-lo) is
# what is actually density-independent.
#
# An isolated population would reach that temperature in {drive.saturation_gyroperiods:.4g} T_ci, \
{drive.saturation_margin:.3g}x inside this
# run -- necessary but NOT sufficient, since the achieved temperature is a balance
# against cold injection and against the expansion doing work. The runs on record clear
# this bar by ~14x and still plateau at 71-80% of setpoint.
{gate}particle_heater.species    = piston_electrons
particle_heater.intervals  = {_intervals(int(heater['intervals']), stop_step)}
particle_heater.profile    = foil
particle_heater.foil.normal = z
particle_heater.foil.lo    = -slab
particle_heater.foil.hi    = slab
particle_heater.foil.spot_radius = rspot
particle_heater.foil.spot_center = 0.
particle_heater.foil.K     = {float(heater['super_gaussian_order']):.6g}
particle_heater.foil.n0    = nt
particle_heater.foil.mass_ratio = {piston_mass_ratio:.10g}
particle_heater.piston_electrons.theta = theta_e_heat"""


def _injector_block(config: dict[str, Any]) -> str:
    """The ``TargetInjector`` block, gated with the heater.

    Stopping only the heater would leave the injector refilling the slab with *cold*
    material: the contrast would hold while the drive died, which is neither the FLASH
    behaviour nor a clean ballistic coast.

    ``density`` is the ELECTRON density.  The injector derives the neutralizing ion rate
    from the charge ratio ``-q_e/(Z q_e) = 1/Z`` itself, so the charge state is stated
    once, on the species.
    """
    injector = config["operators"]["injector"]
    return f"""\
target_injector.species              = piston_electrons
target_injector.neutralizing_species = piston_ions
target_injector.intervals            = {_intervals(int(injector['intervals']), None)}
target_injector.tau                  = {float(injector['tau_over_wpe_inv']):.6g}/wpe
target_injector.lo                   = -rslab -slab
target_injector.hi                   = rslab slab
target_injector.density              = nt
target_injector.reference_density    = n0
target_injector.ppc_reference        = {int(injector['ppc_reference'])}
target_injector.piston_electrons.u_std = sqrt(theta_e_cold)
target_injector.piston_ions.u_std      = sqrt(theta_i_cold)"""


def _diagnostics_block(config: dict[str, Any], *, max_step: int, smoke: bool,
                       macroparticles: float) -> str:
    """Tiered diagnostics: reduced scalars, grid fields, sparse particles, restarts."""
    diagnostics = config["diagnostics"]
    smoke_cfg = config.get("smoke") or {}
    plotfile = int(smoke_cfg["plotfile_intervals"] if smoke
                   else diagnostics["plotfile_intervals"])
    reduced = int(smoke_cfg["reduced_intervals"] if smoke
                  else diagnostics["reduced_intervals"])
    phase = 0 if smoke else int(diagnostics.get("phase_space_intervals", 0) or 0)
    fraction = float(diagnostics.get("phase_space_fraction", 0.02))
    checkpoint = 0 if smoke else int(diagnostics.get("checkpoint_intervals", 0) or 0)

    names = "diag1" + (" phase" if phase > 0 else "") + (" chk" if checkpoint > 0 else "")
    dumps = max(1, max_step // max(plotfile, 1))

    blocks = [f"""\
# --- diagnostics --------------------------------------------------------------
# EP/PN are the operator sanity check: EP rises then plateaus once the heater and
# injector balance, and PN shows the piston inventory being replenished. Watch the
# AMBIENT species' <u^2> for upstream grid heating, the main numerical risk here.
warpx.reduced_diags_names = EP PN
EP.type      = ParticleEnergy
EP.intervals = {reduced}
PN.type      = ParticleNumber
PN.intervals = {reduced}

diagnostics.diags_names = {names}
diag1.intervals      = {plotfile}
diag1.diag_type      = Full
diag1.fields_to_plot = Ex Ey Ez Bx By Bz jx jy jz rho {' '.join(
        f'rho_{name} T_{name}' for name in SPECIES_NAMES)}
# T_<species> assumes equipartition, which the heated piston electrons only marginally
# satisfy; <u^2> is the exact quantity the operator's own CI test checks, so carry both.
diag1.particle_fields_species = {' '.join(SPECIES_NAMES)}
diag1.particle_fields_to_plot = usq
diag1.particle_fields.usq(x,y,z,ux,uy,uz) = "ux*ux+uy*uy+uz*uz"
# NO RAW PARTICLES in diag1. At ~{macroparticles / 1e6:.0f}M macroparticles a Full diagnostic
# writes ~4.4 GB of particle data per dump against 0.27 GB of the grid fields the analysis
# reads -- {dumps} dumps of that is ~{4.6 * dumps / 1000:.1f} TB, and the I/O throttles the run
# harder than the physics does. The per-species rho/T/usq fields above carry every
# profile the comparison scripts need.
diag1.write_species = 0"""]

    if phase > 0:
        blocks.append("\n".join([
            "# Sparse phase space: raw particles at a low cadence, randomly subsampled",
            f"# to {100 * fraction:.0f}% -- enough for (z, u_z) and the reflected-ion beam,",
            "# without diag1's per-frame particle cost.",
            "phase.diag_type      = Full",
            f"phase.intervals      = {phase}",
            "phase.write_species  = 1",
            "phase.fields_to_plot = none",
            f"phase.species        = {' '.join(SPECIES_NAMES)}",
            *[f"phase.{name}.random_fraction = {fraction:.6g}" for name in SPECIES_NAMES],
            "phase.file_prefix    = diags/phase",
        ]))

    if checkpoint > 0:
        blocks.append(
            "# Restartable state, so a run longer than one queue slot can resume. The\n"
            "# sbatch finds the newest chk* whose grid matches this deck and passes it\n"
            "# as amr.restart -- a checkpoint is resumable only at the same grid.\n"
            f"chk.intervals = {checkpoint}\n"
            "chk.diag_type = Full\n"
            "chk.format    = checkpoint")

    return "\n\n".join(blocks)


def _header_block(config: dict[str, Any], scales: units.DeckScales, *,
                  variant: str, cost: dict[str, float], stop_step: int | None) -> str:
    """The provenance header: what this deck is, and what it is matched to."""
    meta = config["meta"]
    flash = scales.flash
    rows = []
    if flash is not None:
        target = flash.invariants()
        for name, value in scales.invariants().items():
            expected = target[name]
            off = "" if not expected or abs(value / expected - 1.0) < 3e-3 else "  <-- OFF"
            rows.append(f"#   {name:<12} {expected:>12.5g} {value:>12.5g}{off}")

    warnings = "\n".join(f"#   {w}" for w in scales.warnings) or "#   (none)"
    drive = ("the whole run" if stop_step is None
             else f"steps 0-{stop_step} ({config['operators']['drive_stop_gyroperiods']} T_ci)")
    return f"""\
# =============================================================================
# {meta['run_id']} -- {variant}
#
# GENERATED by scripts/make_warpx_deck.py from {config.get('_path', 'the run spec')}.
# Edit the spec, not this file.
#
# 2D3V full-PIC magnetized piston. The piston is not read from FLASH -- it is GROWN
# by ParticleHeater + TargetInjector (the Fox et al. 2018 laser-ablation surrogate):
# the injector tops a dense cold slab up towards nt, the heater drives momentum-space
# diffusion in its electrons, and the slab expands as a kinetically smooth piston.
# Boundaries are periodic, so the slab is symmetric and TWO fronts propagate in +-z;
# the run must stop before either wraps.
#
# Matched to FLASH in DIMENSIONLESS terms at a reduced mass ratio. Preserved:
#   {'  '.join(scales.invariants())}
# Deliberately broken: m_i/(Z m_e), omega_pe/omega_ci, and absolute ns/um.
#
#   invariant          FLASH         deck
{chr(10).join(rows) if rows else '#   (no FLASH reference)'}
#
# Reduced by {scales.reduction_factor:.4g}x: ambient m/Ze {units.mass_per_charge(scales.upstream.ion):.4g}, \
piston m/Ze {units.mass_per_charge(scales.piston_ion):.4g}
# Grid {scales.n_cells_x} x {scales.n_cells_z} at {scales.cell_size_de:.4g} d_e, \
lambda_De/dx = {scales.debye_per_cell:.4f}, dt*omega_pe = {scales.dt_omega_pe:.4f}
# {cost['macroparticles'] / 1e6:.0f}M macroparticles, {int(cost['steps'])} steps, \
~{cost['node_hours']:.0f} node-hours
# Drive window: {drive}
#
# Generator warnings:
{warnings}
# ============================================================================="""


def render(config: dict[str, Any], scales: units.DeckScales, *,
           smoke: bool = False, no_heater: bool = False) -> str:
    """Render the full ParmParse deck.

    ``scales`` is passed in rather than derived here because the smoke path derives it
    twice with different domain sizes, and hiding that inside the renderer would
    re-couple the two.
    """
    smoke_cfg = config.get("smoke") or {}
    ppc_each_dim = tuple(int(v) for v in (smoke_cfg["ppc_each_dim"] if smoke
                                          else config["numerics"]["ppc_each_dim"]))
    max_step = int(smoke_cfg["max_step"]) if smoke else scales.max_step
    stop_step = None if smoke else drive_stop_step(config, scales)

    cost = scales.cost(ppc_each_dim, n_species=len(SPECIES_NAMES))
    cost["node_hours"] *= max_step / cost["steps"]
    cost["steps"] = float(max_step)
    variant = ("SMOKE TEST -- shrunken domain, low ppc, few steps" if smoke
               else "NULL CONTROL -- production minus the heater" if no_heater
               else "production")

    return "\n\n".join([
        _header_block(config, scales, variant=variant, cost=cost, stop_step=stop_step),
        _constants_block(scales),
        _time_and_grid_block(config, scales, max_step=max_step),
        _boundary_block(),
        _field_block(config),
        _species_section(ppc_each_dim),
        _heater_block(config, scales, stop_step=stop_step, no_heater=no_heater),
        _injector_block(config),
        _diagnostics_block(config, max_step=max_step, smoke=smoke,
                           macroparticles=cost["macroparticles"]),
    ]) + "\n"


# --------------------------------------------------------------------------- #
# deck -> the numbers WarpX will use
# --------------------------------------------------------------------------- #

def parse_text(text: str) -> dict[str, str]:
    """Parse deck text into ``{key: raw value}``, dropping comments and blank lines."""
    entries: dict[str, str] = {}
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        entries[key.strip()] = value.strip().strip('"')
    return entries


def parse_inputs(path: str | Path) -> dict[str, str]:
    """Parse a WarpX inputs / ``warpx_used_inputs`` file into ``{key: raw value}``."""
    return parse_text(Path(path).read_text())


def _eval(expr: str, namespace: dict[str, float]) -> float:
    """Evaluate one ParmParse expression in a restricted namespace (``^`` is a power)."""
    expr = expr.strip().strip('"').replace("^", "**")
    return float(eval(expr, {"__builtins__": {}}, {**_MATH, **namespace}))


def resolve_constants(entries: dict[str, str]) -> dict[str, float]:
    """Resolve every ``my_constants.*`` to a number, in dependency order.

    The deck defines constants in terms of earlier ones, so this loops until nothing new
    resolves rather than assuming file order.
    """
    pending = {key.split(".", 1)[1]: value for key, value in entries.items()
               if key.startswith("my_constants.")}
    resolved = dict(CONSTS)
    while pending:
        progressed = False
        for name in list(pending):
            try:
                resolved[name] = _eval(pending[name], resolved)
            except (NameError, TypeError, ZeroDivisionError, SyntaxError):
                continue
            del pending[name]
            progressed = True
        if not progressed:
            raise ValueError(f"unresolvable my_constants: {sorted(pending)}")
    return {name: value for name, value in resolved.items() if name not in CONSTS}


def key_params(text: str) -> dict[str, Any]:
    """Everything about a deck that changes what WarpX does, as resolved numbers.

    Comments, formatting and symbolic-vs-decimal are all invisible here, which is what
    makes this the right thing to diff.
    """
    entries = parse_text(text)
    constants = resolve_constants(entries)
    ns = {**CONSTS, **constants}
    at_origin = {**ns, "x": 0.0, "y": 0.0, "z": 0.0}

    out: dict[str, Any] = {f"const:{k}": v for k, v in constants.items()}
    out["max_step"] = int(float(entries["max_step"]))
    out["cfl"] = float(entries["warpx.cfl"])
    out["dims"] = int(float(entries["geometry.dims"]))
    out["particle_shape"] = int(float(entries["algo.particle_shape"]))
    out["max_grid_size"] = int(float(entries["amr.max_grid_size"]))
    out["n_cell"] = [int(float(t)) for t in entries["amr.n_cell"].split()]
    out["prob_lo"] = [_eval(t, ns) for t in entries["geometry.prob_lo"].split()]
    out["prob_hi"] = [_eval(t, ns) for t in entries["geometry.prob_hi"].split()]
    for key in ("boundary.field_lo", "boundary.field_hi",
                "boundary.particle_lo", "boundary.particle_hi"):
        out[key] = entries[key].lower()

    out["By"] = _eval(entries["warpx.By_external_grid_function(x,y,z)"], at_origin)

    for name in SPECIES_NAMES:
        out[f"{name}.ppc"] = [int(float(t)) for t in
                              entries[f"{name}.num_particles_per_cell_each_dim"].split()]
        out[f"{name}.u_std"] = _eval(entries[f"{name}.ux_std"], ns)
        out[f"{name}.n_center"] = _eval(entries[f"{name}.density_function(x,y,z)"],
                                        at_origin)
        if f"{name}.charge" in entries:
            out[f"{name}.charge"] = _eval(entries[f"{name}.charge"], ns)
            out[f"{name}.mass"] = _eval(entries[f"{name}.mass"], ns)

    out["heater.present"] = "particle_heater.species" in entries
    if out["heater.present"]:
        period, stop = _parse_intervals(entries["particle_heater.intervals"])
        out["heater.intervals"] = period
        out["heater.stop_step"] = stop
        out["heater.species"] = entries["particle_heater.species"]
        out["heater.normal"] = entries["particle_heater.foil.normal"]
        for tag in ("lo", "hi", "spot_radius", "spot_center", "n0", "mass_ratio", "K"):
            out[f"heater.{tag}"] = _eval(entries[f"particle_heater.foil.{tag}"], ns)
        out["heater.theta"] = _eval(
            entries[f"particle_heater.{entries['particle_heater.species']}.theta"], ns)

    period, stop = _parse_intervals(entries["target_injector.intervals"])
    out["injector.intervals"] = period
    out["injector.stop_step"] = stop
    out["injector.tau"] = _eval(entries["target_injector.tau"], ns)
    out["injector.density"] = _eval(entries["target_injector.density"], ns)
    out["injector.reference_density"] = _eval(entries["target_injector.reference_density"], ns)
    out["injector.ppc_reference"] = int(float(entries["target_injector.ppc_reference"]))
    out["injector.lo"] = [_eval(t, ns) for t in entries["target_injector.lo"].split()]
    out["injector.hi"] = [_eval(t, ns) for t in entries["target_injector.hi"].split()]
    return out


#: ``amr.restart`` and ``max_step`` are appended by the sbatch, not the generator.
IGNORED_ON_VERIFY = ("amr.restart",)


def verify(deck_text: str, reference_text: str, *,
           rtol: float = VERIFY_RTOL) -> list[str]:
    """Differences between two decks, by resolved number. Empty list means agreement.

    ``rtol`` is not zero because the WarpX build ships CODATA-2022 constants against
    astropy's CODATA-2018 -- a ~1e-9 shift in every resolved length.
    """
    theirs, ours = key_params(deck_text), key_params(reference_text)
    problems = []
    for key in sorted(set(theirs) | set(ours)):
        if key in IGNORED_ON_VERIFY:
            continue
        if key not in theirs:
            problems.append(f"{key}: missing from the deck (spec has {ours[key]!r})")
        elif key not in ours:
            problems.append(f"{key}: in the deck ({theirs[key]!r}) but not the spec")
        elif not _close(theirs[key], ours[key], rtol):
            note = ("  (the sbatch appends max_step via HEATER_EXTRA_ARGS)"
                    if key == "max_step" else "")
            problems.append(f"{key}: deck {theirs[key]!r} != spec {ours[key]!r}{note}")
    return problems


def _close(theirs: Any, ours: Any, rtol: float) -> bool:
    if isinstance(theirs, (list, tuple)) and isinstance(ours, (list, tuple)):
        return len(theirs) == len(ours) and all(
            _close(a, b, rtol) for a, b in zip(theirs, ours))
    if isinstance(theirs, bool) or isinstance(ours, bool):
        return theirs == ours
    if isinstance(theirs, (int, float)) and isinstance(ours, (int, float)):
        return math.isclose(float(theirs), float(ours), rel_tol=rtol, abs_tol=0.0)
    return theirs == ours
