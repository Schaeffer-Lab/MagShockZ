"""heater_deck.py — render and verify the 2D WarpX heater-driven piston deck.

Forward: a loaded ``schema: heater_pic_2d`` run spec plus its
:class:`~heater_piston_scaling.ReducedScaling` become a ParmParse text deck (a string).
Reverse: a deck — or the ``warpx_used_inputs`` WarpX echoes after a run — becomes the
flat dict of numbers WarpX actually used, so the deck can be *proved* to still mean what
the spec says.  The mapping is one-way by design: edit the spec and regenerate.

Why a text deck and not PICMI: ``ParticleHeater`` / ``TargetInjector`` are ParmParse-only
on the lab fork's ``feature/particle-heater`` branch.  No PICMI binding exists anywhere
under ``warpx/Python/``, and every pre-built ``pywarpx`` predates the operator commits,
so a PICMI driver would silently produce a physics-free run.

``my_constants`` are written **symbolically** (``slab = 2.0*di``,
``B0 = vA*sqrt(mu0*namb*mi)``, ``theta_e_amb = beta_e*B0^2/(...)``), so the deck states
its own physics and WarpX still records the resolved values in ``warpx_used_inputs``.
Four of the seven matched invariants — ``M_A`` (through ``vA``), ``contrast`` (through
``nt``), ``beta_e`` and ``beta_i`` — appear in the deck as named constants and can be
checked without running Python.

numpy + stdlib only (via ``heater_piston_scaling``), so this is unit-tested in CI
without yt, WarpX or astropy.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import heater_piston_scaling as hps

SCHEMA: str = "heater_pic_2d"
DECK_NAME: str = "inputs_2d_heater"

#: Deck species names.  The piston/ambient split is by *material*, not by mass: the
#: comparison scripts measure target species against target species, never a summed n_e.
SPECIES_NAMES: tuple[str, ...] = (
    "piston_electrons", "piston_ions", "amb_electrons", "amb_ions")

#: WarpX's own ``my_constants`` namespace, sourced from :mod:`heater_piston_scaling` so
#: there is exactly one set of physical constants in the repo.
#:
#: NOTE the WarpX build ships CODATA-2022 (``m_e = 9.1093837139e-31``,
#: ``mu0 = 1.2566370612685e-06``) while these are CODATA-2018.  They differ at ~1e-8
#: relative, which is why :func:`verify` defaults to ``rtol = 1e-6``; tightening it to
#: 1e-9 would produce a wall of spurious warnings against a real ``warpx_used_inputs``.
CONSTS: dict[str, float] = {
    "m_e": hps.M_E_KG,
    "m_p": hps.M_P_KG,
    "q_e": hps.Q_E,
    "clight": hps.C_LIGHT_MS,
    "epsilon0": hps.EPSILON_0,
    "mu0": hps.MU_0,
    "kb": 1.380649e-23,
    "pi": math.pi,
}

_FUNCS: dict[str, Any] = {
    "sqrt": math.sqrt, "abs": abs, "exp": math.exp, "log": math.log,
    "sin": math.sin, "cos": math.cos, "tan": math.tan, "pow": pow,
}


# --------------------------------------------------------------------------- #
# formatting helpers
# --------------------------------------------------------------------------- #

def _num(x: float) -> str:
    """Format a number the WarpX way.

    Integer-valued floats get a trailing dot and scientific notation drops the
    redundant ``+`` (``1e+18`` -> ``1e18``), so the deck is internally consistent
    instead of mixing ``0`` and ``0.`` on adjacent lines.
    """
    x = float(x)
    if x == int(x) and abs(x) < 1e6:
        return f"{int(x)}."
    return repr(x).replace("e+", "e")


def _intervals(period: int, stop_step: int | None) -> str:
    """WarpX intervals string: a bare period, or ``0:stop:period`` when time-gated.

    ``stop_step`` is rounded DOWN to a multiple of ``period`` so the last application
    lands exactly on the boundary.  Both ``ParticleHeater`` and ``TargetInjector`` parse
    this with the stock ablastr ``IntervalsParser``, so gating the drive needs no engine
    change.
    """
    if stop_step is None:
        return f"{period}"
    return f"0:{stop_step - stop_step % period}:{period}"


def _parse_intervals(text: str) -> tuple[int, int | None]:
    """Inverse of :func:`_intervals`: ``"20"`` -> (20, None), ``"0:23260:20"`` -> (20, 23260)."""
    text = text.strip().strip('"').strip("'")
    if ":" not in text:
        return int(float(text)), None
    parts = text.split(":")
    if len(parts) != 3:
        raise ValueError(
            f"unsupported intervals spec {text!r} (want period or start:stop:period)")
    stop, period = parts[1].strip(), parts[2].strip()
    return int(float(period)), (int(float(stop)) if stop else None)


def drive_stop_step(spec: dict[str, Any], scaling: hps.ReducedScaling) -> int | None:
    """Step at which the heater and injector switch off, or ``None`` to drive all run.

    The spec states the cutoff in the run's own normalized time
    (``operators.drive_stop_t_ci``, in ambient ion gyroperiods) rather than in steps, so
    it survives changes to ``cell_size_de`` and ``cfl``.  Per-operator rounding to a
    multiple of that operator's own period happens in :func:`_intervals`.
    """
    stop_t_ci = (spec.get("operators") or {}).get("drive_stop_t_ci")
    if not stop_t_ci:
        return None
    return int(float(stop_t_ci) * scaling.steps_per_gyroperiod)


# --------------------------------------------------------------------------- #
# forward direction: spec -> deck
# --------------------------------------------------------------------------- #

def _constants_block(spec: dict[str, Any], scaling: hps.ReducedScaling) -> str:
    """The ``my_constants`` namespace: the deck's whole physical state, symbolically.

    Only the genuinely primary numbers are spliced as floats — ``n0`` and ``mass_ratio``
    (free choices), ``vA`` and ``theta_e_heat`` (resolved from ``M_A``, ``kappa`` and
    ``v_piston_c``, whose symbolic form would need three unused symbols to avoid one
    float), and the two matched betas.  Everything else is an expression WarpX resolves.

    Those floats carry **10 significant digits, not the more readable 6**, because the
    symbolic expressions chain off them: ``theta_e_amb`` reaches ``vA`` through
    ``B0 = vA*sqrt(...)`` *squared*, so ``vA`` at 7 digits would land the ambient
    temperature ~2e-7 from what :func:`derive` computed — inside :func:`verify`'s
    ``rtol`` but with only half a decade of margin.
    """
    flow = "" if scaling.v_flow_ms == 0.0 else (
        # my_constants.vflow only exists when it is used: AMReX reports unused ParmParse
        # entries at the end of every run and a permanently-listed one trains you to
        # ignore that report.
        f"my_constants.vflow  = {scaling.v_flow_ms / hps.C_LIGHT_MS:.6e}*clight"
        f"   # transverse bulk drift\n")

    return f"""\
# --- reference state: everything below is derived from these ------------------
my_constants.n0    = {scaling.n0_per_m3:.6e}     # ambient electron density [m^-3]
my_constants.mass_ratio = {scaling.mass_ratio:.6g}         # m_i/m_e, REDUCED on purpose
my_constants.mi    = mass_ratio*m_e
my_constants.wpe   = sqrt(n0*q_e^2/(epsilon0*m_e))
my_constants.de    = clight/wpe
my_constants.di    = de*sqrt(mass_ratio)   # Z = 1, so n_i = n_e = n0
my_constants.nt    = {scaling.contrast:.10g}*n0    # piston (target) density
my_constants.namb  = 1.*n0                 # ambient density
my_constants.vA    = {scaling.v_alfven_ms / hps.C_LIGHT_MS:.10e}*clight   \
# ambient Alfven speed -> M_A = {scaling.mach_alfven:.3f}
my_constants.B0    = vA*sqrt(mu0*namb*mi)  # out-of-plane field ({scaling.b0_tesla * 1e4:.1f} G)
{flow}
# --- geometry, in the ion units the invariants are stated in -------------------
my_constants.slab  = {scaling.slab_halfwidth_di:.6f}*di    # piston reservoir half-width
my_constants.rspot = {scaling.r_spot_di:.8f}*di  # heating spot -> r_spot/d_i, MATCHED
my_constants.xhalf = {scaling.domain_x_halfwidth_de:.6f}*de
my_constants.zhalf = {scaling.domain_z_halfwidth_de:.6f}*de

# --- temperatures as theta = kT/(m c^2), which is exactly WarpX's u_std^2 -------
# The ambient thetas are written as the pressure balance they came from, so the two
# MATCHED betas are named in the deck and readable without re-running the generator:
#   beta = n kT/(B^2/2 mu0)  =>  theta = beta*B0^2/(2 mu0 n m c^2).
my_constants.beta_e = {scaling.beta_e:.10g}      # MATCHED invariant (FLASH)
my_constants.beta_i = {scaling.beta_i:.10g}     # MATCHED invariant (FLASH)
my_constants.theta_e_amb = beta_e*B0^2/(2*mu0*namb*m_e*clight^2)
my_constants.theta_i_amb = beta_i*B0^2/(2*mu0*namb*mi*clight^2)
my_constants.theta_e_heat = {scaling.theta_e_heater:.10g}   \
# heater setpoint -> v_piston = {scaling.v_piston_c:.4g} c
my_constants.theta_e_cold = {scaling.theta_e_cold:.10g}     # piston load, pre-heating
my_constants.theta_i_cold = theta_e_cold/mass_ratio  # same physical T as the electrons"""


def _time_and_grid_block(spec: dict[str, Any], scaling: hps.ReducedScaling, *,
                         max_step: int) -> str:
    """Step count, CFL, the AMR grid and the domain extent."""
    numerics = spec["numerics"]
    return f"""\
max_step      = {max_step}
warpx.cfl     = {float(numerics['cfl']):.6g}
warpx.verbose = 1

amr.n_cell        = {scaling.n_cells_x} {scaling.n_cells_z}
amr.max_level     = 0
amr.max_grid_size = {int(numerics['max_grid_size'])}
geometry.dims     = 2
geometry.prob_lo  = -xhalf -zhalf
geometry.prob_hi  = xhalf zhalf"""


def _boundary_block() -> str:
    """Fully periodic on both axes — not a choice this deck offers.

    A uniform applied E/B on the grid requires periodic boundaries ("do not use any
    other boundary condition than periodic", ``Docs/source/usage/parameters.rst``), and
    the symmetric slab, the domain sizing and the run-length budget all follow from it.
    ``heater_spec.load`` raises on any other request; a one-sided/wall variant needs its
    own ``schema:`` and its own renderer.
    """
    return """\
# Fully periodic — required for the uniform applied E/B below, which in turn is why
# the slab is symmetric and TWO fronts propagate (see the header).
boundary.field_lo    = periodic periodic
boundary.field_hi    = periodic periodic
boundary.particle_lo = periodic periodic
boundary.particle_hi = periodic periodic"""


def _field_block(spec: dict[str, Any], scaling: hps.ReducedScaling) -> str:
    """The uniform out-of-plane B0, plus the motional Ez when there is a bulk drift."""
    magnetization = """\
algo.particle_shape = {shape}

# Uniform out-of-plane magnetization: a perpendicular shock geometry, matching
# FLASH's magz normal to the piston's radial expansion.
warpx.B_ext_grid_init_style = parse_B_ext_grid_function
warpx.Bx_external_grid_function(x,y,z) = "0."
warpx.By_external_grid_function(x,y,z) = "B0"
warpx.Bz_external_grid_function(x,y,z) = "0."
""".format(shape=int(spec["numerics"]["particle_shape"]))

    if scaling.v_flow_ms == 0.0:
        return magnetization + "\n# No bulk drift, so there is no motional E field to impose."
    if bool(spec["flow"].get("impose_motional_e", True)):
        return magnetization + """
# Motional field of the bulk drift: E = -v x B with v = vflow x^ and B = B0 y^ gives
# Ez = -vflow*B0. Without it the uniform cross-field drift is NOT an equilibrium and
# the plasma gyrates instead of drifting. Uniform applied grid fields need PERIODIC
# boundaries.
warpx.E_ext_grid_init_style = parse_E_ext_grid_function
warpx.Ex_external_grid_function(x,y,z) = "0."
warpx.Ey_external_grid_function(x,y,z) = "0."
warpx.Ez_external_grid_function(x,y,z) = "-vflow*B0\""""
    return magnetization + """
# WARNING: flow.impose_motional_e is false, so the cross-field drift is NOT an
# equilibrium — the plasma will gyrate at rho = vflow/omega_ci rather than drift.
# Deliberate choice; see the run spec."""


def _species_block(name: str, *, is_electron: bool, density_expr: str,
                   theta_name: str, u_mean: float,
                   ppc_each_dim: tuple[int, int]) -> str:
    """One ParmParse species block: density profile + drifting Maxwellian.

    ``u_std`` is written as ``sqrt(<theta>)`` rather than as a resolved decimal because
    that *is* WarpX's convention — ``u_std^2 = kT/(m c^2)`` — and writing it that way
    makes the electron/ion asymmetry self-evident instead of two unrelated-looking
    numbers whose ratio happens to be ``sqrt(mass_ratio)``.
    """
    identity = ([f"{name}.species_type = electron"] if is_electron
                else [f"{name}.charge = q_e", f"{name}.mass = mi"])
    lines = [
        *identity,
        f"{name}.injection_style = NUniformPerCell",
        f"{name}.num_particles_per_cell_each_dim = {ppc_each_dim[0]} {ppc_each_dim[1]}",
        f"{name}.profile = parse_density_function",
        f'{name}.density_function(x,y,z) = "{density_expr}"',
        f"{name}.density_min = 1.",
        f"{name}.momentum_distribution_type = maxwellian",
        f"{name}.maxwellian_u_mean_distribution_type = constant",
        f"{name}.ux_mean = {_num(u_mean)}",
        f"{name}.uy_mean = 0.",
        f"{name}.uz_mean = 0.",
        f"{name}.maxwellian_u_std_distribution_type = constant",
        *[f"{name}.{axis}_std = sqrt({theta_name})" for axis in ("ux", "uy", "uz")],
    ]
    return "\n".join(lines)


def _species_section(scaling: hps.ReducedScaling, *,
                     ppc_each_dim: tuple[int, int]) -> str:
    """All four species, split piston/ambient by the slab in the density parser."""
    piston = "nt*(abs(z)<slab)"
    ambient = "namb*(abs(z)>=slab)"
    return "\n\n".join([
        f"particles.species_names = {' '.join(SPECIES_NAMES)}",
        "# --- piston / target slab: heated and continuously replenished ---\n"
        + _species_block("piston_electrons", is_electron=True, density_expr=piston,
                         theta_name="theta_e_cold", u_mean=scaling.u_flow,
                         ppc_each_dim=ppc_each_dim),
        _species_block("piston_ions", is_electron=False, density_expr=piston,
                       theta_name="theta_i_cold", u_mean=scaling.u_flow,
                       ppc_each_dim=ppc_each_dim),
        "# --- ambient: uniform, magnetized, NEITHER heated nor injected ---\n"
        + _species_block("amb_electrons", is_electron=True, density_expr=ambient,
                         theta_name="theta_e_amb", u_mean=scaling.u_flow,
                         ppc_each_dim=ppc_each_dim),
        _species_block("amb_ions", is_electron=False, density_expr=ambient,
                       theta_name="theta_i_amb", u_mean=scaling.u_flow,
                       ppc_each_dim=ppc_each_dim),
    ])


def _heater_block(spec: dict[str, Any], *, stop_step: int | None,
                  no_heater: bool) -> str:
    """The ``ParticleHeater`` block, or the null control's explanation of its absence."""
    if no_heater:
        # NULL CONTROL. Everything else -- domain, dt, steps, ppc, species, injector --
        # is identical to the production deck, so the ambient <u^2> history has only one
        # possible source left: the discretisation. With no energy input the slab cannot
        # expand, so there is no shock and no physical precursor to preheat the upstream.
        # If <u^2> still climbs here, the production run's rise is numerical grid heating
        # and dx must come down; if it stays flat, that rise is physical.
        return """\
# --- laser-ablation surrogate: HEATER DELIBERATELY OMITTED (null control) ------
# The injector below still runs, so the macroparticle count and load match the
# production deck exactly. Only the energy source is gone.
# Regenerate the physics deck with gen_heater_deck.py (no --no-heater)."""

    heater = spec["operators"]["heater"]
    gate = "" if stop_step is None else (
        "# Drive window: FLASH's laser is a finite pulse, so heating for the whole run\n"
        "# would compare a continuously-driven piston against a ballistic one -- a\n"
        "# first-order error in the front trajectory. Hard on/off, not FLASH's ramp:\n"
        "# the foil profile has no time dependence (theta is a scalar fixed at init).\n")
    return f"""\
# --- laser-ablation surrogate: heat and replenish the piston slab ONLY --------
# The ambient must stay cold, so it is absent from both operator species lists.
{gate}particle_heater.species    = piston_electrons
particle_heater.intervals  = {_intervals(int(heater['intervals']), stop_step)}
particle_heater.profile    = foil
particle_heater.foil.normal = z
particle_heater.foil.lo    = -slab
particle_heater.foil.hi    = slab
particle_heater.foil.spot_radius = rspot
particle_heater.foil.spot_center = 0.
particle_heater.foil.K = {float(heater['k']):.6g}
particle_heater.foil.n0    = nt
particle_heater.foil.mass_ratio = mass_ratio
particle_heater.piston_electrons.theta = theta_e_heat"""


def _injector_block(spec: dict[str, Any], *, stop_step: int | None) -> str:
    """The ``TargetInjector`` block, gated with the heater.

    Stopping only the heater would leave the injector refilling the slab with *cold*
    material: the contrast would hold while the drive died, which is neither the FLASH
    behaviour nor a clean ballistic coast.
    """
    injector = spec["operators"]["injector"]
    return f"""\
target_injector.species              = piston_electrons
target_injector.neutralizing_species = piston_ions
target_injector.intervals            = {_intervals(int(injector['intervals']), stop_step)}
target_injector.tau                  = {float(injector['tau_over_wpe_inv']):.6g}/wpe
target_injector.lo                   = -xhalf -slab
target_injector.hi                   = xhalf slab
target_injector.density              = nt
target_injector.reference_density    = n0
target_injector.ppc_reference        = {int(injector['ppc_reference'])}
target_injector.piston_electrons.u_std = sqrt(theta_e_cold)
target_injector.piston_ions.u_std      = sqrt(theta_i_cold)"""


def _diagnostics_block(spec: dict[str, Any], *, max_step: int, smoke: bool,
                       macroparticles: float) -> str:
    """The tiered diagnostics: reduced scalars, grid fields, sparse particles, restarts."""
    diagnostics = spec["diagnostics"]
    smoke_cfg = spec.get("smoke") or {}
    plotfile = int(smoke_cfg["plotfile_intervals"] if smoke
                   else diagnostics["plotfile_intervals"])
    reduced = int(smoke_cfg["reduced_intervals"] if smoke
                  else diagnostics["reduced_intervals"])
    # Raw particles, but sparsely: the reflected-ion signature in (z, u_z) is the
    # defining kinetic diagnostic of a super-critical shock, so the capability has to
    # survive diag1.write_species = 0. A few percent of the particles at a few times is
    # plenty, and costs ~1% of what a Full particle dump per frame would.
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
# Per-species rho and T (eV) mean the piston density/temperature profiles and any
# upstream grid heating read straight off the plotfile, with no need to histogram
# raw particles (scripts/warpx_heater_compare.py relies on this).
diag1.fields_to_plot = Ex Ey Ez Bx By Bz jx jy jz rho {' '.join(
    f'rho_{name} T_{name}' for name in SPECIES_NAMES)}
# T_<species> assumes equipartition, which the heated piston electrons only
# marginally satisfy; <u^2> is the exact quantity and is what the operator's own
# CI test checks, so carry it too.
diag1.particle_fields_species = {' '.join(SPECIES_NAMES)}
diag1.particle_fields_to_plot = usq
diag1.particle_fields.usq(x,y,z,ux,uy,uz) = "ux*ux+uy*uy+uz*uz"
# NO RAW PARTICLES in diag1. With ~{macroparticles / 1e6:.0f}M macroparticles a Full
# diagnostic writes ~4.4 GB of particle data per dump against 0.27 GB of the grid fields
# the analysis actually reads -- {dumps} dumps of that is ~{4.6 * dumps / 1000:.1f} TB and the
# I/O throttles the run harder than the physics does. The per-species rho/T/usq fields
# above already carry every profile scripts/warpx_{{heater_compare,flash_evolution}}.py need.
diag1.write_species = 0"""]

    if phase > 0:
        blocks.append("\n".join([
            "# Sparse phase-space dump: raw particles at a low cadence, randomly",
            f"# subsampled to {100 * fraction:.0f}% -- enough for (z, u_z) and the",
            "# reflected-ion beam, without diag1's per-frame particle cost.",
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
            "# Restartable state, so a run longer than one queue slot can resume.\n"
            "# init_warpx/run_heater_2d.sbatch finds the newest chk* whose grid matches\n"
            "# this deck and passes it as amr.restart on the command line.\n"
            f"chk.intervals = {checkpoint}\n"
            "chk.diag_type = Full\n"
            "chk.format    = checkpoint")

    return "\n\n".join(blocks)


def _header_block(spec: dict[str, Any], scaling: hps.ReducedScaling, *,
                  variant: str, cost: dict[str, float], stop_step: int | None) -> str:
    """The prose banner: mechanism, geometry, FLASH provenance, cost, warnings."""
    targets = scaling.targets
    assert targets is not None                      # derive() always attaches them
    flash = spec["flash_target"]

    warn_lines = "".join(f"#   WARNING: {w}\n" for w in scaling.warnings) or "#   (none)\n"
    if variant.startswith("SMOKE") and scaling.warnings:
        warn_lines += ("#   (EXPECTED in a smoke deck: the shrunken box lets the front "
                       "wrap and\n#    flattens the heating spot. It only checks that "
                       "the deck parses and the\n#    operators fire — never read "
                       "physics off it.)\n")
    drive = ("the whole run" if stop_step is None
             else f"steps 0-{stop_step} of {cost['steps']:.0f}")

    return f"""\
# =============================================================================
# 2D3V heater-driven magnetized piston — {variant}
#
# GENERATED by init_warpx/gen_heater_deck.py from
#   {spec.get('_spec_path', '(unknown spec)')}
# DO NOT EDIT THIS FILE: edit the run spec and regenerate. The generator verifies
# that the deck resolves back to the spec, and --verify does the same against the
# post-run warpx_used_inputs, so a hand edit here will be reported as drift.
#
# Mechanism (PSC / Fox et al., Phys. Plasmas 25, 102106 (2018); magnetized
# follow-up Schaeffer et al., Phys. Plasmas 27, 042901 (2020)): TargetInjector
# keeps a dense cold slab at |z| < {scaling.slab_halfwidth_de:.0f} d_e topped up towards nt, ParticleHeater
# drives momentum-space diffusion in its electrons towards theta_e = {scaling.theta_e_heater:.4g}, and the
# heated slab expands as a piston into the cold magnetized ambient, driving a
# perpendicular collisionless shock. A transverse Gaussian heating spot of radius
# {scaling.spot_radius_de:.0f} d_e gives the piston a finite width in x — that is what makes this a 2D
# problem rather than a wider run_shock_1d, and it supplies the transverse
# gradients Biermann/Weibel field generation needs. The drive runs for {drive}.
#
# GEOMETRY. 2D XZ; the piston expands along +-z, B0 is out of plane (By), and the
# bulk drift is along x. The slab is SYMMETRIC about z = 0, so TWO fronts
# propagate in +-z and the run must stop before either wraps.
#
# WHY FULLY PERIODIC. A uniform applied E/B on the grid requires periodic
# boundaries ("do not use any other boundary condition than periodic",
# Docs/source/usage/parameters.rst), and every validated heater deck in
# heating_operator/ is fully periodic. Hence the symmetric slab above.
#
# FLASH PROVENANCE — this deck is tuned to match, in dimensionless terms only:
#   dataset  {flash.get('dataset', '?')}
#   window   {flash.get('t_window_ns', '?')} ns
#   source   {targets.source or '(unset)'}
#   M_A = {targets.mach_alfven:.3f}   M_ms = {targets.mach_magnetosonic:.3f}   \
beta_e = {targets.beta_e:.3f}   beta_i = {targets.beta_i:.3f}
#   n_piston/n_amb = {targets.contrast:.3f}   r_spot/d_i = {targets.r_spot_di:.3f}   \
t_run = {scaling.t_run_gyro:.3f} T_ci
# NOT matched (reduced-mass PIC): m_i/m_e {targets.a_amb * hps.M_P_KG / hps.M_E_KG:.4g} -> \
{scaling.mass_ratio:.4g}, Z {targets.z_amb:.0f} -> 1,
#   and hence absolute ns/um. See src/heater_piston_scaling.py.
#
# SIZE. {cost['cells']:.3g} cells, ~{cost['macroparticles']:.3g} macroparticles, \
{cost['steps']:.0f} steps
#   -> ~{cost['node_hours']:.1f} node-hours (factor-of-two estimate; see cost_report)
#
# Generator warnings:
{warn_lines}\
# ============================================================================="""


def render_deck(spec: dict[str, Any], scaling: hps.ReducedScaling, *,
                smoke: bool = False, no_heater: bool = False) -> str:
    """Render the full ParmParse deck as a string.

    ``scaling`` is passed in rather than derived here because the smoke path derives it
    twice with different domain sizes (once to size the box from physics, once after
    shrinking it), and hiding that inside the renderer would re-couple the two.
    """
    smoke_cfg = spec.get("smoke") or {}
    ppc_each_dim = tuple(int(v) for v in (smoke_cfg["ppc_each_dim"] if smoke
                                          else spec["scaling"]["ppc_each_dim"]))
    max_step = int(smoke_cfg["max_step"]) if smoke else scaling.max_step
    stop_step = None if smoke else drive_stop_step(spec, scaling)

    # cost_report() reports the scaling's own max_step; the smoke deck overrides it.
    cost = scaling.cost_report(ppc_each_dim=ppc_each_dim, n_species=len(SPECIES_NAMES))
    cost["node_hours"] *= max_step / cost["steps"]
    cost["steps"] = float(max_step)
    variant = "SMOKE TEST — quarter domain, low ppc, few steps" if smoke else "production"

    return "\n\n".join([
        _header_block(spec, scaling, variant=variant, cost=cost, stop_step=stop_step),
        _constants_block(spec, scaling),
        _time_and_grid_block(spec, scaling, max_step=max_step),
        _boundary_block(),
        _field_block(spec, scaling),
        _species_section(scaling, ppc_each_dim=ppc_each_dim),
        _heater_block(spec, stop_step=stop_step, no_heater=no_heater),
        _injector_block(spec, stop_step=stop_step),
        _diagnostics_block(spec, max_step=max_step, smoke=smoke,
                           macroparticles=cost["macroparticles"]),
    ]) + "\n"


# --------------------------------------------------------------------------- #
# reverse direction: deck -> the numbers WarpX will use
# --------------------------------------------------------------------------- #

def parse_inputs(path: str | Path) -> dict[str, str]:
    """Parse a WarpX inputs / ``warpx_used_inputs`` file into ``{key: raw value}``."""
    return _parse_inputs_text(Path(path).read_text())


def _parse_inputs_text(text: str) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        entries[key.strip()] = value.strip()
    return entries


def _eval(expr: str, namespace: dict[str, float]) -> float:
    """Evaluate a WarpX scalar expression (``^`` -> ``**``) in a restricted namespace."""
    return float(eval(expr.replace("^", "**").strip().strip('"'),
                      {"__builtins__": {}}, {**namespace, **_FUNCS}))


def resolve_constants(entries: dict[str, str]) -> dict[str, float]:
    """Numerically resolve every ``my_constants.*`` expression.

    Iterative, because the deck defines constants in terms of each other
    (``di`` needs ``de`` needs ``wpe``) and ParmParse imposes no ordering.
    """
    prefix = "my_constants."
    exprs = {k[len(prefix):]: v for k, v in entries.items() if k.startswith(prefix)}
    resolved = dict(CONSTS)
    pending = dict(exprs)
    for _ in range(len(pending) + 2):
        if not pending:
            break
        progressed = False
        for name, expr in list(pending.items()):
            try:
                resolved[name] = _eval(expr, resolved)
            except (NameError, KeyError, ValueError, TypeError, ZeroDivisionError):
                continue
            del pending[name]
            progressed = True
        if not progressed:
            break
    if pending:
        raise ValueError(f"could not resolve my_constants: {sorted(pending)}")
    return {name: resolved[name] for name in exprs}


def key_params_from_text(text: str) -> dict[str, Any]:
    """Resolve a deck's text to a flat dict of the numbers WarpX will actually use.

    Independent of formatting, comments, and whether a length was written as ``20.*de``
    or ``2.*di`` — which is what makes it a meaningful comparison between a freshly
    rendered deck and one WarpX has already echoed back.
    """
    entries = _parse_inputs_text(text)
    constants = resolve_constants(entries)
    ns = {**CONSTS, **constants}

    out: dict[str, Any] = {f"const:{k}": v for k, v in constants.items()}
    out["max_step"] = int(float(entries["max_step"]))
    out["cfl"] = float(entries["warpx.cfl"])
    out["dims"] = int(float(entries.get("geometry.dims", "1")))
    out["particle_shape"] = int(float(entries["algo.particle_shape"]))
    out["max_grid_size"] = int(float(entries["amr.max_grid_size"]))
    out["n_cell"] = [int(float(t)) for t in entries["amr.n_cell"].split()]
    out["prob_lo"] = [_eval(t, ns) for t in entries["geometry.prob_lo"].split()]
    out["prob_hi"] = [_eval(t, ns) for t in entries["geometry.prob_hi"].split()]
    for key in ("boundary.field_lo", "boundary.field_hi",
                "boundary.particle_lo", "boundary.particle_hi"):
        if key in entries:
            out[key] = entries[key].lower()

    out["By"] = _eval(entries["warpx.By_external_grid_function(x,y,z)"],
                      {**ns, "x": 0.0, "y": 0.0, "z": 0.0})
    ez = entries.get("warpx.Ez_external_grid_function(x,y,z)")
    out["Ez"] = 0.0 if ez is None else _eval(ez, {**ns, "x": 0.0, "y": 0.0, "z": 0.0})

    # The operators. `heater.present` is what makes the null control a testable claim,
    # and the intervals round-trip covers the drive window as well as the cadence.
    out["heater.present"] = "particle_heater.species" in entries
    if out["heater.present"]:
        period, stop = _parse_intervals(entries["particle_heater.intervals"])
        out["heater.intervals"] = period
        out["heater.stop_step"] = stop
        out["heater.species"] = entries["particle_heater.species"]
        out["heater.profile"] = entries["particle_heater.profile"]
        out["heater.normal"] = entries["particle_heater.foil.normal"]
        for tag, key in (("foil_lo", "lo"), ("foil_hi", "hi"),
                         ("spot_radius", "spot_radius"), ("spot_center", "spot_center"),
                         ("foil_n0", "n0"), ("mass_ratio", "mass_ratio"), ("K", "K")):
            out[f"heater.{tag}"] = _eval(entries[f"particle_heater.foil.{key}"], ns)
        out["heater.theta"] = _eval(
            entries[f"particle_heater.{entries['particle_heater.species']}.theta"], ns)

    period, stop = _parse_intervals(entries["target_injector.intervals"])
    out["injector.intervals"] = period
    out["injector.stop_step"] = stop
    out["injector.tau"] = _eval(entries["target_injector.tau"], ns)
    out["injector.density"] = _eval(entries["target_injector.density"], ns)
    out["injector.reference_density"] = _eval(
        entries["target_injector.reference_density"], ns)
    out["injector.ppc_reference"] = int(float(entries["target_injector.ppc_reference"]))
    for axis_key in ("lo", "hi"):
        out[f"injector.{axis_key}"] = [
            _eval(t, ns) for t in entries[f"target_injector.{axis_key}"].split()]
    for species in (entries["target_injector.species"],
                    entries["target_injector.neutralizing_species"]):
        out[f"injector.u_std:{species}"] = _eval(
            entries[f"target_injector.{species}.u_std"], ns)

    # Per species. The two density probes are the only check that catches a swapped
    # < / >= in a density function -- the most consequential possible typo in this deck,
    # and one no scalar comparison would see.
    zhalf = out["prob_hi"][-1]
    for species in entries.get("particles.species_names", "").split():
        out[f"ppc:{species}"] = [
            int(float(t))
            for t in entries[f"{species}.num_particles_per_cell_each_dim"].split()]
        out[f"u_std:{species}"] = _eval(entries[f"{species}.ux_std"], ns)
        out[f"u_mean:{species}"] = _eval(entries[f"{species}.ux_mean"], ns)
        density = entries[f"{species}.density_function(x,y,z)"]
        out[f"n_slab:{species}"] = _eval(density, {**ns, "x": 0.0, "y": 0.0, "z": 0.0})
        out[f"n_amb:{species}"] = _eval(
            density, {**ns, "x": 0.0, "y": 0.0, "z": 0.9 * zhalf})

    for diag in ("EP", "PN", "diag1", "phase", "chk"):
        key = f"{diag}.intervals"
        if key in entries:
            out[key] = int(float(entries[key]))
    if "diag1.write_species" in entries:
        out["diag1.write_species"] = int(float(entries["diag1.write_species"]))
    return out


def key_params(path: str | Path) -> dict[str, Any]:
    """:func:`key_params_from_text` for a deck on disk."""
    return key_params_from_text(Path(path).read_text())


#: Keys the launcher may legitimately override on the command line, so a divergence in
#: them is not deck drift.  ``init_warpx/run_heater_2d.sbatch`` appends ``amr.restart``
#: on every resume and passes ``$HEATER_EXTRA_ARGS`` straight through.
_CLI_OVERRIDABLE: dict[str, str] = {
    "max_step": "a CLI override (HEATER_EXTRA_ARGS) or a shortened run, not deck drift",
}


def verify(spec: dict[str, Any], scaling: hps.ReducedScaling, inputs_path: str | Path,
           *, rtol: float = 1e-6, smoke: bool = False,
           no_heater: bool = False) -> list[str]:
    """Confirm the deck (or ``warpx_used_inputs``) at ``inputs_path`` matches the spec.

    Renders a fresh deck for the same spec and diffs the resolved
    :func:`key_params`, so it catches drift between the run spec and what WarpX
    actually ran.  Returns warning strings; empty means the deck still means what the
    spec says.

    ``rtol`` is 1e-6 rather than exact because WarpX's own constants are CODATA-2022
    against this module's CODATA-2018 — see :data:`CONSTS`.
    """
    want = key_params_from_text(
        render_deck(spec, scaling, smoke=smoke, no_heater=no_heater))
    got = key_params(inputs_path)
    name = Path(inputs_path).name

    warnings: list[str] = []
    for key in sorted(set(want) | set(got)):
        if key not in got:
            # my_constants are an implementation detail: WarpX prunes unused ones from
            # warpx_used_inputs, and the same value may be written as 20.*de or 2.*di.
            # The scalar settings carry the strict comparison.
            if not key.startswith("const:"):
                warnings.append(f"{key}: missing from {name}")
            continue
        if key not in want:
            continue                       # extra keys (amr.restart, ...) are allowed
        theirs, ours = got[key], want[key]
        if not _close(theirs, ours, rtol):
            note = _CLI_OVERRIDABLE.get(key)
            detail = f" — {note}" if note else ""
            warnings.append(f"{key}: {name} {theirs!r} vs spec {ours!r}{detail}")
    return warnings


def _close(theirs: Any, ours: Any, rtol: float) -> bool:
    """Compare two key_params values, elementwise for the list-valued ones.

    ``n_cell`` / ``prob_lo`` / ``prob_hi`` / the injector box are lists, and comparing
    them with ``!=`` would make WarpX's CODATA-2022 constants read as drift: it resolves
    ``de = clight/wpe`` with its own ``m_e``, which moves every length by ~1e-9 relative.
    """
    if isinstance(theirs, (list, tuple)) or isinstance(ours, (list, tuple)):
        if not (isinstance(theirs, (list, tuple)) and isinstance(ours, (list, tuple))
                and len(theirs) == len(ours)):
            return False
        return all(_close(t, o, rtol) for t, o in zip(theirs, ours))
    try:
        return abs(float(theirs) - float(ours)) <= rtol * max(abs(float(ours)), 1e-30)
    except (TypeError, ValueError):
        return bool(theirs == ours)


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #

def scaling_report(spec: dict[str, Any], scaling: hps.ReducedScaling, *,
                   ppc_each_dim: tuple[int, int], max_step: int) -> str:
    """The human-readable scale / cost / invariance report kept beside the deck."""
    cost = scaling.cost_report(ppc_each_dim=ppc_each_dim, n_species=len(SPECIES_NAMES))
    targets = scaling.targets
    assert targets is not None
    debye_ok = ("ok" if scaling.debye_per_cell >= hps.DEBYE_PER_CELL_MIN
                else f"UNDER {hps.DEBYE_PER_CELL_MIN}")
    stop_step = drive_stop_step(spec, scaling)

    lines = [
        "Reduced-mass PIC deck",
        f"  n0            {scaling.n0_per_m3:.4g} m^-3   "
        f"d_e = {scaling.d_e_m * 1e6:.4g} um   d_i = {scaling.d_i_m * 1e6:.4g} um",
        f"  mass_ratio    {scaling.mass_ratio:.4g}   "
        f"omega_pe = {scaling.omega_pe_rad_s:.4g} rad/s",
        f"  heater        theta_e = {scaling.theta_e_heater:.4g}  ->  "
        f"c_s = {scaling.c_s_piston_ms / hps.C_LIGHT_MS:.4g} c, "
        f"v_piston = {scaling.v_piston_c:.4g} c",
        f"  ambient       B0 = {scaling.b0_tesla * 1e4:.4g} G   "
        f"v_A = {scaling.v_alfven_ms / hps.C_LIGHT_MS:.4g} c   "
        f"T_e = {scaling.te_amb_ev:.4g} eV   T_i = {scaling.ti_amb_ev:.4g} eV",
        f"  u_std         piston e {scaling.theta_e_cold ** 0.5:.4g}  "
        f"i {scaling.theta_i_cold ** 0.5:.4g}   |   ambient e "
        f"{scaling.theta_e_amb ** 0.5:.4g}  i {scaling.theta_i_amb ** 0.5:.4g}",
        f"  densities     n_target = {scaling.n_target_per_m3:.4g}   "
        f"n_ambient = {scaling.n_amb_per_m3:.4g}   (contrast {scaling.contrast:.4g})",
        f"  drive         {'whole run' if stop_step is None else f'steps 0-{stop_step}'}"
        f"  ({'no drive window set' if stop_step is None else f'{stop_step / scaling.steps_per_gyroperiod:.4f} T_ci'})",
        "",
        "Grid and time",
        f"  cells         {scaling.n_cells_x} x {scaling.n_cells_z}  "
        f"at dx = {scaling.cell_size_de_actual:.6g} d_e "
        f"(requested {scaling.cell_size_de:.4g})",
        f"  domain        x +-{scaling.domain_x_halfwidth_de:.0f} d_e "
        f"({scaling.domain_x_halfwidth_de * scaling.d_e_m / scaling.d_i_m:.1f} d_i)   "
        f"z +-{scaling.domain_z_halfwidth_de:.0f} d_e "
        f"({scaling.domain_z_halfwidth_de * scaling.d_e_m / scaling.d_i_m:.1f} d_i)",
        f"  slab / spot   +-{scaling.slab_halfwidth_de:.4g} d_e "
        f"({scaling.slab_halfwidth_di:.2f} d_i)   /   "
        f"r_H = {scaling.spot_radius_de:.4g} d_e ({scaling.r_spot_di:.2f} d_i)",
        f"  Debye         lambda_De = {scaling.debye_length_de:.4g} d_e   "
        f"lambda_De/dx = {scaling.debye_per_cell:.4g} ({debye_ok})",
        f"  timestep      dt*omega_pe = {scaling.dt_omega_pe:.4g}   "
        f"T_ci = {scaling.gyroperiod_s * scaling.omega_pe_rad_s:.4g}/omega_pe "
        f"= {scaling.steps_per_gyroperiod:.0f} steps",
        f"  duration      {max_step} steps = "
        f"{max_step / scaling.steps_per_gyroperiod:.3f} T_ci"
        f"  (target {scaling.t_run_gyro:.3f} T_ci = "
        f"{targets.t_window_s * 1e9:.2f} ns of FLASH)",
        "",
        "Cost (rough)",
        f"  {cost['cells']:.4g} cells x {cost['ppc_per_species']:.0f} ppc x "
        f"{len(SPECIES_NAMES)} species = {cost['macroparticles']:.4g} macroparticles",
        f"  {cost['steps']:.0f} steps  ->  ~{cost['node_hours']:.1f} node-hours "
        f"(~{cost['node_hours'] / 4:.1f} h on 4 CPU nodes)",
        "",
        hps.invariance_report(scaling),
    ]
    return "\n".join(lines)
