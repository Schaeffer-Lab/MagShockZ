"""heater_spec.py — load, validate and freeze a ``schema: heater_pic_2d`` run spec.

The run spec (``runs/magshockz_2d_heater.warpx.yaml``) is the single source of truth for
a WarpX heater-driven piston run.  This module turns it into the two objects everything
else needs — :class:`~heater_piston_scaling.PistonTargets` (what FLASH measured) and
:class:`~heater_piston_scaling.ReducedScaling` (what the deck will be built from) — and
reports what is wrong with it.

Named ``heater_spec``, not ``heater_config``: in this repo ``runs/*.yaml`` are *run
specs* (cf. :mod:`run_spec`) while ``config/*.yaml`` are FLASH *analysis* configs, and
this reads the former.

The split between raising and warning is deliberate and is the same one omegashock's
``config.py`` makes:

* :func:`load` **raises** on anything the generator cannot render at all — a wrong
  schema, a missing block, a non-periodic boundary request.  There is no defensible
  default for any of them.
* :func:`validate` **warns and never raises**.  A deliberately off-target deck is a
  legitimate thing to want: the ``--no-heater`` null control, a resolution probe, a
  frame-consistency run with a bulk drift.  Refusing to render them would be wrong.

Stdlib + PyYAML + numpy (via :mod:`heater_piston_scaling`), so this is unit-tested in CI
without yt or WarpX.
"""

from __future__ import annotations

import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import yaml

import heater_deck
import heater_piston_scaling as hps

SCHEMA: str = heater_deck.SCHEMA

#: Blocks the generator cannot proceed without.
REQUIRED_BLOCKS: tuple[str, ...] = (
    "flash_target", "scaling", "flow", "operators", "numerics", "diagnostics")


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #

def load(path: str | Path) -> dict[str, Any]:
    """Load and structurally check a heater_pic_2d run spec.

    Records the spec's own path under ``_spec_path`` so the rendered deck can name the
    file it came from.  Raises on structural problems; physics problems are
    :func:`validate`'s job.
    """
    path = Path(path).resolve()
    spec = yaml.safe_load(path.read_text())
    if not isinstance(spec, dict):
        raise ValueError(f"{path}: expected a YAML mapping")

    schema = spec.get("schema")
    if schema != SCHEMA:
        raise ValueError(
            f"{path}: schema is {schema!r}, expected {SCHEMA!r}. The other "
            f"runs/*.warpx.yaml files are flash2warpx hybrid specs and are driven by "
            f"init_warpx/run_prod.py, not this generator.")
    _require(spec, REQUIRED_BLOCKS, path=path)

    window = spec["flash_target"]["t_window_ns"]
    if float(window[1]) <= float(window[0]):
        raise ValueError(f"{path}: t_window_ns must be increasing, got {window!r}")

    ppc = spec["scaling"]["ppc_each_dim"]
    if not (isinstance(ppc, Sequence) and len(ppc) == 2
            and all(int(v) > 0 for v in ppc)):
        raise ValueError(f"{path}: scaling.ppc_each_dim must be two positive ints, "
                         f"got {ppc!r} (the run is 2D)")

    boundary = (spec.get("geometry") or {}).get("boundary")
    if boundary is not None and boundary != "periodic":
        raise ValueError(
            f"{path}: geometry.boundary is {boundary!r}, but this schema is fully "
            f"periodic and cannot be otherwise. A uniform applied E/B on the grid "
            f"requires periodic boundaries (\"do not use any other boundary condition "
            f"than periodic\", WarpX Docs/source/usage/parameters.rst), and the "
            f"symmetric slab, the domain sizing and the run-length budget all follow "
            f"from it. A one-sided/wall variant needs its own schema: and its own "
            f"renderer.")

    spec["_spec_path"] = str(path)
    return spec


def _require(spec: dict[str, Any], blocks: Sequence[str], *, path: Path) -> None:
    """Raise naming the first missing or non-mapping block."""
    for block in blocks:
        if not isinstance(spec.get(block), dict):
            raise ValueError(f"{path}: missing or non-mapping '{block}:' block")


# --------------------------------------------------------------------------- #
# resolution
# --------------------------------------------------------------------------- #

def targets(spec: dict[str, Any]) -> hps.PistonTargets:
    """Build :class:`PistonTargets` from the spec's ``flash_target:`` block."""
    block = spec["flash_target"]
    t_lo_ns, t_hi_ns = (float(v) for v in block["t_window_ns"])
    return hps.PistonTargets(
        n_amb_per_m3=float(block["n_amb_per_m3"]),
        b_amb_tesla=float(block["b_amb_tesla"]),
        te_amb_ev=float(block["te_amb_ev"]),
        ti_amb_ev=float(block["ti_amb_ev"]),
        n_piston_drive_per_m3=float(block["n_piston_drive_per_m3"]),
        v_front_ms=float(block["v_front_ms"]),
        l_piston_m=float(block["l_piston_m"]),
        r_spot_m=float(block["r_spot_m"]),
        t_window_s=(t_hi_ns - t_lo_ns) * 1e-9,
        a_amb=float(block.get("a_amb", 26.98)),
        z_amb=float(block.get("z_amb", 13.0)),
        a_piston=float(block.get("a_piston", 28.0855)),
        z_piston=float(block.get("z_piston", 14.0)),
        source=str(block.get("source", "")),
    )


def scaling(spec: dict[str, Any], *, smoke: bool = False) -> hps.ReducedScaling:
    """Run :func:`heater_piston_scaling.derive` with the spec's parameters.

    ``smoke`` shrinks the domain by ``smoke.domain_scale`` *before* deriving, so the
    domain-wrap warning correctly fires for the small box (a smoke deck is expected to
    let the front reach the boundary; it never runs that long).  That is why the smoke
    path derives twice, and why the renderer takes the scaling rather than deriving it.
    """
    block = spec["scaling"]
    flow = spec["flow"]

    domain_x = block.get("domain_x_halfwidth_de")
    domain_z = block.get("domain_z_halfwidth_de")
    if smoke:
        scale = float((spec.get("smoke") or {}).get("domain_scale", 0.25))
        # The sizing pass must be the run itself, differing only in the domain — a
        # cheaper approximation would size the box from a different run length and
        # hand the shrink factor the wrong number to scale.
        sized = _derive(spec)
        domain_x = scale * (domain_x or sized.domain_x_halfwidth_de)
        domain_z = scale * (domain_z or sized.domain_z_halfwidth_de)

    return _derive(spec, domain_x_halfwidth_de=domain_x, domain_z_halfwidth_de=domain_z)


def _derive(spec: dict[str, Any], *, domain_x_halfwidth_de: float | None = None,
            domain_z_halfwidth_de: float | None = None) -> hps.ReducedScaling:
    """One :func:`heater_piston_scaling.derive` call from the spec's parameters."""
    block = spec["scaling"]
    return hps.derive(
        targets(spec),
        n0_per_m3=float(block["n0_per_m3"]),
        mass_ratio=float(block["mass_ratio"]),
        v_piston_c=float(block["v_piston_c"]),
        kappa=float(block["kappa_expansion"]),
        theta_e_heater=_optional_float(block, "theta_e_heater"),
        cell_size_de=float(block["cell_size_de"]),
        slab_halfwidth_di=float(block["slab_halfwidth_di"]),
        domain_x_halfwidth_de=(None if domain_x_halfwidth_de is None
                               else float(domain_x_halfwidth_de)),
        domain_z_halfwidth_de=(None if domain_z_halfwidth_de is None
                               else float(domain_z_halfwidth_de)),
        cfl=float(spec["numerics"]["cfl"]),
        theta_e_cold=float(block["theta_e_cold"]),
        run_window_gyro=_optional_float(block, "run_window_gyro"),
        v_flow_ms=1.0e3 * float(spec["flow"].get("v_flow_kms", 0.0)),
        blocking=int(spec["numerics"].get("blocking_factor", hps.BLOCKING_DEFAULT)),
    )


def _optional_float(block: dict[str, Any], key: str) -> float | None:
    value = block.get(key)
    return None if value is None else float(value)


# --------------------------------------------------------------------------- #
# validation — warns, never raises
# --------------------------------------------------------------------------- #

def validate(spec: dict[str, Any], scales: hps.ReducedScaling | None = None, *,
             out_dir: str | Path | None = None) -> list[str]:
    """Physics and numerics warnings for a heater_pic_2d spec.

    Starts from ``scales.warnings`` (which :func:`derive` accumulates) and adds the
    checks that need the spec, not just the derived scales.  ``out_dir`` opts into the
    one check that touches the filesystem — stale checkpoints — and is a no-op when
    ``None``, keeping the pure path pure.
    """
    scales = scales or scaling(spec)
    warnings = list(scales.warnings)
    acceptance = ((spec.get("targets") or {}).get("acceptance") or {})

    warnings += _invariant_warnings(spec, scales)
    warnings += _resolution_warnings(spec, scales, acceptance)
    warnings += _decomposition_warnings(spec, scales)
    warnings += _cadence_warnings(spec, scales, acceptance)
    warnings += _drive_warnings(spec, scales)
    warnings += _smoke_warnings(spec)
    warnings += _checkpoint_warnings(scales, out_dir)

    seen: set[str] = set()
    return [w for w in warnings if not (w in seen or seen.add(w))]


def _invariant_warnings(spec: dict[str, Any],
                        scales: hps.ReducedScaling) -> list[str]:
    """Derived invariants against the spec's ``targets.invariants:``.

    Unlike omegashock's equivalent — where the target check is a *calibration* check
    against a hand-tuned deck — every one of these is matched by construction in
    :func:`derive`.  So this is a regression guard on ``derive`` itself, which is why
    the tolerance is tight rather than the ~15% omegashock allows.
    """
    declared = ((spec.get("targets") or {}).get("invariants") or {})
    if not declared:
        return []
    aliases = {"r_spot_di": "r_spot/d_i", "t_run_gyro": "t_run/T_ci"}
    derived = scales.invariants()
    warnings = []
    for name, expected in declared.items():
        key = aliases.get(name, name)
        if key not in derived:
            warnings.append(f"targets.invariants.{name}: not a matched invariant "
                            f"(have {sorted(derived)})")
            continue
        actual = derived[key]
        if abs(actual - float(expected)) > 1e-6 * max(abs(float(expected)), 1e-30):
            warnings.append(
                f"targets.invariants.{name}: spec says {float(expected):.6g} but "
                f"derive() gives {actual:.6g} — the FLASH measurement moved, or the "
                f"scaling bridge regressed")
    return warnings


def _resolution_warnings(spec: dict[str, Any], scales: hps.ReducedScaling,
                         acceptance: dict[str, Any]) -> list[str]:
    """Debye resolution and the plasma-period timestep.

    ``derive`` already warns against :data:`heater_piston_scaling.DEBYE_PER_CELL_MIN`, so
    this only adds a warning when the spec asks for a *stricter* threshold than the
    module default — otherwise the same problem would be reported twice in two wordings.
    """
    warnings = []
    debye_min = float(acceptance.get("debye_per_cell_min", hps.DEBYE_PER_CELL_MIN))
    if debye_min > hps.DEBYE_PER_CELL_MIN and scales.debye_per_cell < debye_min:
        suggested = scales.cell_size_de * scales.debye_per_cell / debye_min
        warnings.append(
            f"lambda_De/dx = {scales.debye_per_cell:.4f} is below this spec's "
            f"targets.acceptance.debye_per_cell_min = {debye_min} — "
            f"cell_size_de <= {suggested:.4g} would meet it")
    dt_max = float(acceptance.get("dt_omega_pe_max", 0.5))
    if scales.dt_omega_pe > dt_max:
        warnings.append(f"dt*omega_pe = {scales.dt_omega_pe:.4g} exceeds {dt_max} — the "
                        f"plasma period is under-resolved")
    return warnings


def _decomposition_warnings(spec: dict[str, Any],
                            scales: hps.ReducedScaling) -> list[str]:
    """Blocking factor and whether there are enough AMReX boxes to fill the ranks."""
    warnings = []
    blocking = int(spec["numerics"].get("blocking_factor", hps.BLOCKING_DEFAULT))
    for axis, cells in (("x", scales.n_cells_x), ("z", scales.n_cells_z)):
        if blocking > 1 and cells % blocking:
            warnings.append(f"n_cells_{axis} = {cells} is not a multiple of the "
                            f"blocking factor {blocking}")

    max_grid = int(spec["numerics"]["max_grid_size"])
    boxes = ((scales.n_cells_x + max_grid - 1) // max_grid
             * (scales.n_cells_z + max_grid - 1) // max_grid)
    ranks = int((spec.get("runtime") or {}).get("nodes", 0)) * 4
    if ranks and boxes < ranks:
        warnings.append(
            f"max_grid_size = {max_grid} gives {boxes} boxes over {ranks} ranks — some "
            f"ranks will idle; halve max_grid_size or reduce runtime.nodes")
    return warnings


def _cadence_warnings(spec: dict[str, Any], scales: hps.ReducedScaling,
                      acceptance: dict[str, Any]) -> list[str]:
    """Diagnostic cadences, which are in STEPS and so must move with the step count.

    This is the failure mode the run spec records being bitten by: going 35760 -> 99326
    steps at an unchanged interval would have written ~199 plotfiles of ~750 MB and
    throttled the run the way raw particles did.
    """
    diagnostics = spec["diagnostics"]
    max_step = scales.max_step
    warnings = []

    lo, hi = (int(v) for v in acceptance.get("plotfile_count", hps.PLOTFILE_COUNT_RANGE))
    plotfile = int(diagnostics["plotfile_intervals"])
    dumps = max_step // max(plotfile, 1)
    if not lo <= dumps <= hi:
        warnings.append(
            f"diagnostics.plotfile_intervals = {plotfile} gives {dumps} plotfiles over "
            f"{max_step} steps, outside [{lo}, {hi}] — "
            f"{max(1, max_step // max(lo if dumps > hi else hi, 1))} would land in range")

    phase = int(diagnostics.get("phase_space_intervals", 0) or 0)
    if phase > 0 and not 3 <= max_step // phase <= 30:
        warnings.append(f"diagnostics.phase_space_intervals = {phase} gives "
                        f"{max_step // phase} raw-particle dumps; 5-20 is the useful range")

    checkpoint = int(diagnostics.get("checkpoint_intervals", 0) or 0)
    if checkpoint <= 0:
        warnings.append("no checkpoints: a run longer than one queue slot cannot resume")
    elif max_step // checkpoint < 2:
        warnings.append(f"diagnostics.checkpoint_intervals = {checkpoint} writes fewer "
                        f"than 2 checkpoints in {max_step} steps")

    reduced = int(diagnostics["reduced_intervals"])
    if max_step // max(reduced, 1) > 1_000_000:
        warnings.append(f"diagnostics.reduced_intervals = {reduced} would write over a "
                        f"million rows to EP.txt/PN.txt")
    return warnings


def _drive_warnings(spec: dict[str, Any], scales: hps.ReducedScaling) -> list[str]:
    """The finite drive window, which FLASH has and a continuously-heated deck does not."""
    stop_t_ci = (spec["operators"] or {}).get("drive_stop_t_ci")
    if not stop_t_ci:
        return ["operators.drive_stop_t_ci is unset, so the heater drives for the whole "
                "run — FLASH's laser does not (flash.par ed_time_*), so the deck compares "
                "a continuously-driven piston against a ballistic one"]
    stop_t_ci = float(stop_t_ci)
    if stop_t_ci >= scales.t_run_gyro:
        return [f"operators.drive_stop_t_ci = {stop_t_ci:.4g} T_ci is not shorter than "
                f"the run ({scales.t_run_gyro:.4g} T_ci) — the gate never fires"]
    stop_step = heater_deck.drive_stop_step(spec, scales)
    period = int(spec["operators"]["heater"]["intervals"])
    if stop_step is not None and stop_step < period:
        return [f"operators.drive_stop_t_ci = {stop_t_ci:.4g} T_ci stops the drive at "
                f"step {stop_step}, before the heater's first application at step "
                f"{period} — the piston would never be driven at all"]
    return []


def _smoke_warnings(spec: dict[str, Any]) -> list[str]:
    smoke = spec.get("smoke") or {}
    if not smoke:
        return []
    max_step = int(smoke.get("max_step", 0))
    interval = int(smoke.get("plotfile_intervals", 0) or 0)
    if interval and max_step < interval:
        return [f"smoke.max_step = {max_step} is below smoke.plotfile_intervals = "
                f"{interval}, so the smoke run writes no plotfiles at all"]
    return []


def _checkpoint_warnings(scales: hps.ReducedScaling,
                         out_dir: str | Path | None) -> list[str]:
    """Warn at generation time that checkpoints on disk will be skipped at job time.

    ``init_warpx/run_heater_2d.sbatch`` compares each checkpoint's ``amr.n_cell`` against
    the deck's and skips a mismatch — but only once the job is running, after the queue
    wait has already been spent.
    """
    if out_dir is None:
        return []
    deck_grid = [scales.n_cells_x, scales.n_cells_z]
    stale: dict[str, list[int]] = {}
    for checkpoint in sorted(Path(out_dir).glob("diags/chk*")):
        info = checkpoint / "warpx_job_info"
        if not (checkpoint / "WarpXHeader").is_file() or not info.is_file():
            continue
        match = re.search(r"^amr\.n_cell\s*=\s*(\d+)\s+(\d+)", info.read_text(),
                          re.MULTILINE)
        if match and [int(match.group(1)), int(match.group(2))] != deck_grid:
            stale[checkpoint.name] = [int(match.group(1)), int(match.group(2))]
    if not stale:
        return []
    example = next(iter(stale.items()))
    return [f"{len(stale)} checkpoint(s) under {out_dir}/diags are a different grid "
            f"(e.g. {example[0]}: {example[1][0]}x{example[1][1]} vs this deck's "
            f"{deck_grid[0]}x{deck_grid[1]}) — run_heater_2d.sbatch will SKIP them and "
            f"restart from step 0. Archive them (diags -> diags_<tag>) first; plotfile "
            f"names collide too."]


# --------------------------------------------------------------------------- #
# freezing
# --------------------------------------------------------------------------- #

def freeze(spec: dict[str, Any], scales: hps.ReducedScaling, *,
           provenance: dict[str, Any] | None = None) -> dict[str, Any]:
    """The resolved spec plus its derived values, ready for ``yaml.safe_dump``.

    Mirrors the OSIRIS convention of freezing a resolved ``run.yaml`` into the run
    directory: the deck alone does not say which FLASH measurement or which derived
    invariants produced it, and the deck is regenerable while a finished run is not.

    Pure — the caller writes the file and gathers ``provenance`` (git SHAs need a
    subprocess, which has no business in this layer).
    """
    frozen = {k: v for k, v in spec.items() if not k.startswith("_")}
    derived = {k: v for k, v in asdict(scales).items()
               if k not in ("warnings", "targets")}
    frozen["derived"] = {
        **{k: float(v) if isinstance(v, (int, float)) else v
           for k, v in derived.items()},
        "invariants_flash": scales.targets.invariants() if scales.targets else {},
        "invariants_deck": scales.invariants(),
        "warnings": list(scales.warnings),
    }
    if provenance:
        frozen["provenance"] = dict(provenance)
    return frozen
