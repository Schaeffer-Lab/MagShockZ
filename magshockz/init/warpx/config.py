"""Load, validate and freeze a ``schema: heater_pic_2d`` run config.

``load`` RAISES on anything the generator cannot render -- a wrong schema, a missing
block, a non-periodic boundary.  ``validate`` WARNS and never refuses, because a
deliberately off-target deck (the heater-off null control, a resolution probe, a
mass-ratio scan point) is legitimate work.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import astropy.units as u
import yaml
from astropy.constants import c

from . import units
from .calibration import CalibrationPoint, HeaterCalibration, fit

SCHEMA = "heater_pic_2d"
REQUIRED_BLOCKS = ("flash", "reference", "plasma", "geometry", "numerics",
                   "operators", "calibration", "diagnostics")

#: Invariants are matched by construction, so this is a regression guard on ``derive``
#: rather than a physics tolerance.  It cannot be tightened past ~1e-3: the electron
#: mass is 1/(m/Ze) of the deck's ion mass density -- 1% at m/Ze = 100, negligible for a
#: real ion -- and plasmapy counts it in ``Alfven_speed`` but not ``ion_sound_speed``,
#: which leaves ``M_ms`` about 0.1% off however exactly ``v_A`` is solved.
INVARIANT_TOLERANCE = 3e-3


def _require(config: dict[str, Any], block: str, path: Path) -> Any:
    if block not in config:
        raise ValueError(f"{path}: missing required block {block!r}")
    return config[block]


def load(path: str | Path) -> dict[str, Any]:
    """Parse and structurally check a run config."""
    path = Path(path)
    with path.open() as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")
    if config.get("schema") != SCHEMA:
        raise ValueError(
            f"{path}: schema is {config.get('schema')!r}, expected {SCHEMA!r}")

    for block in REQUIRED_BLOCKS:
        value = _require(config, block, path)
        if block != "calibration" and not isinstance(value, dict):
            raise ValueError(f"{path}: block {block!r} must be a mapping")

    window = config["flash"]["window_ns"]
    if not (len(window) == 2 and window[1] > window[0]):
        raise ValueError(f"{path}: flash.window_ns must increase, got {window!r}")

    boundary = config["geometry"].get("boundary", "periodic")
    if boundary != "periodic":
        raise ValueError(
            f"{path}: geometry.boundary is {boundary!r}. A uniform applied E/B on the "
            f"grid requires periodic boundaries, and the symmetric slab, the domain "
            f"sizing and the run-length budget all follow from it. A walled variant "
            f"needs its own schema and renderer.")

    # geometry.piston_shape used to couple the target's radius to the heated spot's, so a
    # spec still carrying it means something different now: slab_radius_di: null was the
    # laser spot and is now the whole box. Raise rather than silently re-geometry a run.
    if "piston_shape" in config["geometry"]:
        raise ValueError(
            f"{path}: geometry.piston_shape no longer exists -- the target radius and the "
            f"heated spot radius are independent primaries now. Write "
            f"'slab_radius_di: <r>' for the old 'patch' (null = a target spanning the "
            f"box) and 'spot_radius_di: 0' for the old 'slab' (null = FLASH's spot).")

    ppc = config["numerics"]["ppc_each_dim"]
    if not (len(ppc) == 2 and all(isinstance(n, int) and n > 0 for n in ppc)):
        raise ValueError(f"{path}: numerics.ppc_each_dim must be two positive ints")

    if len(config["calibration"]) < 2:
        raise ValueError(
            f"{path}: calibration needs at least 2 completed runs to fit the heater "
            f"response; got {len(config['calibration'])}")

    config["_path"] = str(path)
    return config


def flash_reference(config: dict[str, Any]) -> units.FlashReference:
    """The ``flash:`` block as a :class:`~warpx.units.FlashReference`."""
    block = config["flash"]
    ambient, piston = block["ambient"], block["piston"]
    ambient_ion = units.as_particle(ambient["species"])
    piston_ion = units.as_particle(piston["species"])

    window = block["window_ns"]
    return units.FlashReference(
        upstream=units.Upstream(
            ion=ambient_ion,
            electron_density=(float(ambient["ion_density_per_m3"])
                              * ambient_ion.charge_number * u.m**-3),
            magnetic_field=float(ambient["magnetic_field_tesla"]) * u.T,
            electron_temperature=float(ambient["electron_temperature_ev"]) * u.eV,
            ion_temperature=float(ambient["ion_temperature_ev"]) * u.eV,
        ),
        piston_ion=piston_ion,
        piston_electron_density=(float(piston["ion_density_per_m3"])
                                 * piston_ion.charge_number * u.m**-3),
        piston_front_speed=float(piston["front_speed_km_s"]) * u.km / u.s,
        spot_radius=float(piston["spot_radius_um"]) * u.um,
        target_halfwidth=(None if piston.get("target_halfwidth_um") is None
                          else float(piston["target_halfwidth_um"]) * u.um),
        window=(float(window[1]) - float(window[0])) * u.ns,
        source=str(block.get("source", "")),
    )


def calibration(config: dict[str, Any]) -> HeaterCalibration:
    """The ``calibration:`` block, fitted over the runs that share this deck's Z.

    Z is a separate axis from the fit's grouping variable ``S = theta/(m/Ze)``: the
    heater's rate carries the BARE ``m_i/m_e``, which is Z times ``m/(Z m_e)``, so runs
    at different charge states do not lie on one curve.  Fitting them together is not
    conservative, it is wrong -- mixing the two Z = 1 runs with the Z = 14 ones flattened
    the exponent from 0.43 to 0.32 and moved the predicted setpoint by 40%.  All the
    points stay in the spec; only the matching ones are fitted.  If fewer than two match,
    everything is fitted and :func:`validate` says so.
    """
    points = tuple(
        CalibrationPoint(
            run_id=str(entry["run_id"]),
            heater_theta=float(entry["heater_theta"]),
            piston_mass_per_charge=float(entry["piston_mass_per_charge"]),
            piston_speed=float(entry["piston_speed_over_c"]) * c,
            achieved_theta=float(entry.get("achieved_theta", float("nan"))),
            charge_number=int(entry.get("charge_number", 1)),
        )
        for entry in config["calibration"]
    )
    charge = round(
        units.as_particle(config["flash"]["piston"]["species"]).charge_number)
    matching = tuple(p for p in points if p.charge_number == charge)
    return fit(matching if len(matching) >= 2 else points)


def scales(config: dict[str, Any], *, smoke: bool = False) -> units.DeckScales:
    """Config -> every derived scale."""
    reference = config["reference"]
    geometry = config["geometry"]
    flash = flash_reference(config)
    ambient = config["flash"]["ambient"]

    derived = units.derive(
        flash,
        reference_density=float(reference["density_per_m3"]) * u.m**-3,
        ambient_mass_per_charge=float(reference["ambient_mass_per_charge"]),
        ambient_electron_temperature=float(ambient["electron_temperature_ev"]) * u.eV,
        ambient_ion_temperature=float(ambient["ion_temperature_ev"]) * u.eV,
        contrast=flash.contrast,
        piston_initial_temperature=(
            float(config["plasma"]["piston"]["initial_temperature_ev"]) * u.eV),
        calibration=calibration(config),
        heater_theta=_optional(config["operators"]["heater"], "theta"),
        cell_size_de=float(geometry["cell_size_de"]),
        slab_halfwidth_di=float(geometry["slab_halfwidth_di"]),
        slab_radius_di=_optional(geometry, "slab_radius_di"),
        spot_radius_di=_optional(geometry, "spot_radius_di"),
        domain_halfwidth_di=_optional(geometry, "domain_halfwidth_di"),
        transverse_halfwidth_di=_optional(geometry, "transverse_halfwidth_di"),
        run_gyroperiods=_optional(geometry, "run_gyroperiods"),
        cfl=float(config["numerics"]["cfl"]),
        blocking=int(config["numerics"]["blocking_factor"]),
    )
    if not smoke:
        return derived

    # Re-derive at the shrunken box so the domain-wrap warning fires for the small one.
    scale = float((config.get("smoke") or {}).get("domain_scale", 0.25))
    return units.derive(
        flash,
        reference_density=float(reference["density_per_m3"]) * u.m**-3,
        ambient_mass_per_charge=float(reference["ambient_mass_per_charge"]),
        ambient_electron_temperature=float(ambient["electron_temperature_ev"]) * u.eV,
        ambient_ion_temperature=float(ambient["ion_temperature_ev"]) * u.eV,
        contrast=flash.contrast,
        piston_initial_temperature=(
            float(config["plasma"]["piston"]["initial_temperature_ev"]) * u.eV),
        calibration=calibration(config),
        heater_theta=_optional(config["operators"]["heater"], "theta"),
        cell_size_de=float(geometry["cell_size_de"]),
        slab_halfwidth_di=float(geometry["slab_halfwidth_di"]),
        # A target spanning the domain must keep spanning the SHRUNKEN one, so the smoke
        # deck stays the same geometry instead of becoming a patch in a smaller box.
        slab_radius_di=None if derived.slab_spans_domain else derived.slab_radius_di,
        spot_radius_di=derived.spot_radius_di,
        domain_halfwidth_di=derived.domain_halfwidth_di * scale,
        transverse_halfwidth_di=derived.transverse_halfwidth_di * scale,
        run_gyroperiods=_optional(geometry, "run_gyroperiods"),
        cfl=float(config["numerics"]["cfl"]),
        blocking=int(config["numerics"]["blocking_factor"]),
    )


def _optional(block: dict[str, Any], key: str) -> float | None:
    value = block.get(key)
    return None if value is None else float(value)


def validate(config: dict[str, Any], derived: units.DeckScales) -> list[str]:
    """Physical and practical warnings. Never raises."""
    messages = list(derived.warnings)
    flash = derived.flash
    numerics = config["numerics"]

    if flash is not None:
        target = flash.invariants()
        for name, value in derived.invariants().items():
            expected = target[name]
            if expected and abs(value / expected - 1.0) > INVARIANT_TOLERANCE:
                messages.append(
                    f"invariant {name} is {value:.6g} against FLASH's {expected:.6g} "
                    f"({abs(value / expected - 1.0):.2%} off)")

    if derived.debye_per_cell < units.DEBYE_PER_CELL_MIN:
        messages.append(
            f"lambda_De/dx = {derived.debye_per_cell:.4f} is below "
            f"{units.DEBYE_PER_CELL_MIN}")
    if derived.dt_omega_pe > 0.5:
        messages.append(f"dt*omega_pe = {derived.dt_omega_pe:.3f} exceeds 0.5")

    blocking = int(numerics["blocking_factor"])
    for axis, count in (("x", derived.n_cells_x), ("z", derived.n_cells_z)):
        if count % blocking:
            messages.append(
                f"n_cells_{axis} = {count} is not a multiple of blocking_factor "
                f"{blocking}")

    dumps = derived.max_step // max(int(config["diagnostics"]["plotfile_intervals"]), 1)
    if not 10 <= dumps <= 200:
        messages.append(
            f"plotfile_intervals gives {dumps} dumps over {derived.max_step} steps; "
            f"cadences are in STEPS and must be rescaled when max_step moves")

    fit = calibration(config)
    extrapolation = fit.extrapolation_warning(derived.theta_e_heater,
                                              units.mass_per_charge(derived.piston_ion))
    if extrapolation:
        messages.append(extrapolation)

    # The heater's rate is H ~ theta^{3/2}/sqrt(foil.mass_ratio), where foil.mass_ratio is
    # the piston's BARE m_i/m_e. Every calibration run so far had Z = 1, where that equals
    # m/(Z m_e), so the two collapse into the single group S = theta/(m/Ze) the fit uses.
    # At a real charge state they differ by Z and the fit is being extrapolated along an
    # axis it never sampled -- which is exactly what a calibration run at this Z fixes.
    charge = round(units.as_particle(derived.piston_ion).charge_number)
    stale = sorted({point.charge_number for point in fit.points} - {charge})
    if stale:
        messages.append(
            f"the heater calibration was measured at Z = {stale}, this deck has "
            f"Z = {charge}. The fit groups runs by S = theta/(m/Ze), but the heater's rate "
            f"carries the bare m_i/m_e, and the two differ by Z -- so the predicted piston "
            f"speed {derived.piston_speed_over_c:.4g} c is an extrapolation along an "
            f"unsampled axis. Measure it and add a calibration point at this Z.")

    if (config["operators"].get("drive_stop_gyroperiods") is None):
        messages.append(
            "operators.drive_stop_gyroperiods is unset, so the heater drives for the "
            "whole run -- FLASH's laser is a finite pulse")

    nodes = int((config.get("runtime") or {}).get("nodes", 0) or 0)
    if nodes:
        cost = derived.cost(tuple(numerics["ppc_each_dim"]))
        walltime = _walltime_hours(config)
        if walltime and cost["node_hours"] > nodes * walltime:
            messages.append(
                f"estimated {cost['node_hours']:.0f} node-hours exceeds the "
                f"{nodes * walltime:.0f} requested ({nodes} nodes x {walltime:.1f} h); "
                f"the run will need to chain across queue slots")

    return list(dict.fromkeys(messages))


def _walltime_hours(config: dict[str, Any]) -> float:
    text = str((config.get("runtime") or {}).get("walltime", "")).strip()
    if not text:
        return 0.0
    parts = [float(piece) for piece in text.split(":")]
    while len(parts) < 3:
        parts.append(0.0)
    return parts[0] + parts[1] / 60.0 + parts[2] / 3600.0


def _heater_drive_record(config: dict[str, Any],
                         derived: units.DeckScales) -> dict[str, float | int]:
    """The resolved heater kick, for the run directory's record.

    ``theta`` alone does not say what the operator does to a particle; these do.
    """
    drive = units.heater_drive(
        derived, intervals=int(config["operators"]["heater"]["intervals"]))
    return {
        "heater_diffusion_rate_m2_per_s3": float(
            drive.diffusion_rate.to_value(u.m**2 / u.s**3)),
        "heater_kick_per_application_over_c": drive.kick_per_application,
        "heater_applications": drive.applications,
        "heater_saturation_gyroperiods": drive.saturation_gyroperiods,
    }


def freeze(config: dict[str, Any], derived: units.DeckScales) -> dict[str, Any]:
    """The config plus everything derived, for the run directory's record."""
    frozen = {key: value for key, value in config.items() if not key.startswith("_")}
    flash = derived.flash
    frozen["derived"] = {
        "reduction_factor": derived.reduction_factor,
        "ambient_mass_per_charge": units.mass_per_charge(derived.upstream.ion),
        "piston_mass_per_charge": units.mass_per_charge(derived.piston_ion),
        "magnetic_field_tesla": float(derived.magnetic_field.to_value(u.T)),
        "alfven_speed_over_c": derived.upstream.alfven_speed_over_c,
        "piston_speed_over_c": derived.piston_speed_over_c,
        "heater_theta": derived.theta_e_heater,
        "heater_temperature_ev": float(derived.heater_temperature.to_value(u.eV)),
        **_heater_drive_record(config, derived),
        "electron_skin_depth_m": float(derived.electron_skin_depth.to_value(u.m)),
        "ion_skin_depth_m": float(derived.ion_skin_depth.to_value(u.m)),
        "gyroperiod_s": float(derived.gyroperiod.to_value(u.s)),
        "timestep_s": float(derived.timestep.to_value(u.s)),
        "slab_radius_di": derived.slab_radius_di,
        "slab_spans_domain": derived.slab_spans_domain,
        "spot_radius_di": derived.spot_radius_di,
        "target_halfwidth_over_di_flash": (
            flash.target_halfwidth_over_di if flash is not None else float("nan")),
        "n_cells": [derived.n_cells_x, derived.n_cells_z],
        "max_step": derived.max_step,
        "debye_per_cell": derived.debye_per_cell,
        "invariants_deck": derived.invariants(),
        "invariants_flash": flash.invariants() if flash is not None else {},
        "warnings": derived.warnings,
    }
    return frozen
