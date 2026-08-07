# -*- coding: utf-8 -*-
"""scripts/flash_piston_profile.py — measure the FLASH piston the WarpX heater is tuned to.

The 2D WarpX heater deck (``runs/magshockz_2d_heater.warpx.yaml``) does not read FLASH
data — it grows its own piston with WarpX's ParticleHeater + TargetInjector.  FLASH's
role is to say *what the piston should look like*, and this script is that measurement.

Over a window of FLASH dumps it extracts line-outs along the config's line of sight and
reports, per dump:

  * the piston front position, from the target-material ion density (mass-fraction
    weighted, so the exponential tail survives — see ``src/piston_profile.py``);
  * the piston peak density and its e-folding length behind the front;
  * the ambient state a gap ahead of the front (n_e, |B|, T_e, T_i), which is what
    actually sets M_A and beta — the chamber plasma rarefies substantially over the
    first few ns, so the initial condition is the wrong thing to use.

and across the window: a straight-line fit to the front trajectory, and the collapse of
the per-dump profiles onto ``n/n_peak`` vs ``(s - s_front)/L`` (a test of self-similarity
— if they collapse, three numbers per dump describe the piston completely).

Outputs, under ``src/yaml_edit.py::out_dir``:
    flash_piston_profile.png   4-panel figure: profiles, collapse, trajectory, ambient
    flash_piston_profile.npz   every array, plain CGS with the unit in each key
    flash_piston_profile.txt   the `flash_target:` YAML block to paste into the run spec

Usage
-----
    conda activate analysis
    python scripts/flash_piston_profile.py --config config/flash_3d_corrected.yaml \\
        [--t-window 3 12] [--npoints 1024] [--nprocs 8] \\
        [--front-threshold 0.1] [--ambient-offset-um 400] [--ambient-width-um 400] \\
        [--fit-window-um 300] [--output-dir ...] [--pub]

Run in the `analysis` conda env (yt + unyt).
"""

import argparse
import dataclasses
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

import analysis_utils
import flash_source
import flash_utils as fu
import heater_piston_scaling as hps
import piston_profile as pp
import plot_style
import yaml_edit

#: FLASH mass fractions and the electron bookkeeping fields (plot_var_5/6 in flash.par)
#: that ``flash_lineout`` does not name.  Zbar = ye/sumy is the local mean charge.
EXTRA_FIELDS = {
    "piston_frac": ("flash", "PISTON_MATERIAL"),   # filled in from the config
    "ye": ("flash", "ye"),
    "sumy": ("flash", "sumy"),
}

AMU_G = 1.66053906660e-24
CM_PER_UM = 1.0e-4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure the FLASH piston profile and trajectory along the LOS.")
    parser.add_argument("--config", required=True,
                        help="FLASH-side analysis config (config/flash_*.yaml)")
    parser.add_argument("--t-window", nargs=2, type=float, metavar=("LO", "HI"),
                        help="time window [ns] of dumps to use (default: all dumps)")
    parser.add_argument("--npoints", type=int, default=1024,
                        help="samples along the LOS (default: %(default)s)")
    parser.add_argument("--nprocs", type=int, default=8,
                        help="dumps loaded in parallel (default: %(default)s)")
    parser.add_argument("--front-level-frac", type=float, default=1.0,
                        help="front = where piston density crosses this multiple of the "
                             "far-field ambient n_e. An ABSOLUTE level, because the "
                             "dense inner plume overtakes the leading edge partway "
                             "through the run and a peak-relative threshold then jumps "
                             "backwards (default: %(default)s)")
    parser.add_argument("--front-threshold", type=float,
                        default=pp.FRONT_THRESHOLD_DEFAULT,
                        help="fall back to this fraction of the piston peak when the "
                             "far-field ambient reference is unusable "
                             "(default: %(default)s)")
    parser.add_argument("--upstream", choices=("initial", "measured"), default="initial",
                        help="where the upstream/ambient state comes from. 'initial' "
                             "(default) reads the INITIALLY UNPERTURBED background off the "
                             "ic_index dump, matching flash.par's chamber IC and ignoring "
                             "the laser channel (nothing has fired at t=0). 'measured' "
                             "samples ahead of the front instead, which by a few ns is "
                             "conduction/radiation preheated by >10x in T_e and is a "
                             "precursor state, not the background "
                             "(default: %(default)s)")
    parser.add_argument("--contamination-max", type=float, default=0.1,
                        help="a dump counts as having a pristine upstream only if "
                             "piston material is below this fraction of the total "
                             "density in the ambient band (default: %(default)s)")
    parser.add_argument("--fit-window-um", type=float, default=300.0,
                        help="window inward from the front for the e-folding fit "
                             "(default: %(default)s)")
    # 1500 um is where the measured ambient state stops changing with the offset: at
    # 400 um the band is still inside the compressed, conduction-preheated shell
    # (n_e 1.7x, |B| 1.9x, T_e 3.5x the converged values), while 1500 and 3000 um agree
    # to a few percent. Re-check this with an offset scan on any new dataset.
    parser.add_argument("--ambient-offset-um", type=float, default=1500.0,
                        help="gap ahead of the front before sampling the ambient, to "
                             "clear the compressed, preheated shell (default: %(default)s)")
    parser.add_argument("--ambient-width-um", type=float, default=600.0,
                        help="width of the ambient sampling band (default: %(default)s)")
    # The band just inside the front whose density drives the shock. Kept narrow and
    # close in: further inward the profile climbs towards the stagnated material next to
    # the target, which never reaches the front.
    parser.add_argument("--drive-offset-um", type=float, default=50.0,
                        help="gap inward from the front before sampling the piston drive "
                             "density (default: %(default)s)")
    parser.add_argument("--drive-width-um", type=float, default=300.0,
                        help="width of the piston drive-density band "
                             "(default: %(default)s)")
    parser.add_argument("--n-profiles", type=int, default=6,
                        help="dumps drawn in the profile panels (default: %(default)s)")
    parser.add_argument("--output-dir", help="override the results directory")
    plot_style.add_publication_arg(parser)
    return parser.parse_args()


def select_dumps(paths: list, t_window) -> tuple[list, np.ndarray]:
    """Dumps whose FLASH time falls in ``t_window`` [ns], with their times [s].

    Times are read from the HDF5 header only (``flash_time_s``), so selecting a window
    out of 60+ dumps does not open a single dataset.
    """
    times = np.array([fu.flash_time_s(p) for p in paths])
    if t_window is None:
        return list(paths), times
    lo, hi = (1e-9 * float(v) for v in t_window)
    keep = np.flatnonzero((times >= lo) & (times <= hi))
    if keep.size == 0:
        raise SystemExit(
            f"No FLASH dumps in t = [{t_window[0]}, {t_window[1]}] ns; the run spans "
            f"{times.min() * 1e9:.2f}-{times.max() * 1e9:.2f} ns.")
    return [paths[i] for i in keep], times[keep]


def measure_dump(lineout: dict, *, mass_number: float, front_level_frac: float,
                 front_threshold: float, contamination_max: float,
                 fit_window_cm: float, ambient_offset_cm: float,
                 ambient_width_cm: float, drive_offset_cm: float,
                 drive_width_cm: float) -> dict:
    """Front, piston peak/scale-length and ambient state for one dump's line-out.

    All lengths CGS.  ``lineout`` is what :func:`flash_utils.flash_lineout` returns plus
    the ``EXTRA_FIELDS`` entries.
    """
    x_cm = np.asarray(lineout["x"].to("cm").value, dtype=float)
    rho = np.asarray(lineout["rho"].to("g/cm**3").value, dtype=float)
    piston_frac = np.asarray(lineout["piston_frac"], dtype=float)
    ye = np.asarray(lineout["ye"], dtype=float)
    sumy = np.asarray(lineout["sumy"], dtype=float)
    ne_cm3 = np.asarray(lineout["ne"].to("cm**-3").value, dtype=float)

    # Everything is measured as an ION density, which is Z-free: it comes from the mass
    # density and a mass fraction alone.  The deck then applies the charge states the run
    # spec chose (Al 6+, Si 14+) to reach electron densities.  Measuring in electrons
    # instead would fold FLASH's EOS Zbar -- 3.7 in the ambient, ~11 in the piston, and
    # both varying along the line-out -- into a comparison whose deck has ONE charge state
    # per species by construction, so the two sides would no longer be like for like.
    n_piston_ion = pp.piston_ion_density(rho, piston_frac, mass_number, AMU_G)
    zbar = np.divide(ye, sumy, out=np.zeros_like(ye), where=sumy > 0.0)
    n_ion = np.divide(ne_cm3, zbar, out=np.zeros_like(ne_cm3), where=zbar > 0.0)

    # Absolute front level from the far-field ambient, so the front keeps tracking the
    # leading edge when the dense inner plume overtakes it in amplitude.
    reference = pp.ambient_reference_level(x_cm, n_ion)
    level = (front_level_frac * reference if np.isfinite(reference) and reference > 0.0
             else None)
    x_front = pp.front_position(x_cm, n_piston_ion, front_threshold, level=level)
    scale_length = pp.efolding_length(x_cm, n_piston_ion, x_front, fit_window_cm)

    def ahead(values) -> float:
        return pp.ahead_of_front_average(x_cm, values, x_front,
                                         ambient_offset_cm, ambient_width_cm)

    n_piston_drive = pp.behind_front_average(x_cm, n_piston_ion, x_front,
                                             drive_offset_cm, drive_width_cm)

    return {
        "n_piston_drive_cm3": n_piston_drive,
        "edge_resolved": pp.edge_is_resolved(x_cm, n_piston_ion, x_front,
                                             scale_length),
        "has_upstream": pp.upstream_is_pristine(
            x_cm, n_piston_ion, n_ion, x_front, ambient_offset_cm,
            ambient_width_cm, contamination_max=contamination_max),
        "ambient_reference_cm3": reference,
        "x_cm": x_cm,
        "n_piston_cm3": n_piston_ion,
        "n_piston_ion_cm3": n_piston_ion,
        "n_ion_cm3": n_ion,
        "ne_cm3": ne_cm3,
        "B_mag_gauss": np.asarray(lineout["B_mag"].to("gauss").value, dtype=float),
        "Te_eV": np.asarray(lineout["Te"].to("eV").value, dtype=float),
        "Ti_eV": np.asarray(lineout["Ti"].to("eV").value, dtype=float),
        "v_para_cms": np.asarray(lineout["v_para"].to("cm/s").value, dtype=float),
        "zbar": zbar,
        "t_s": float(lineout["t_s"]),
        "x_front_cm": x_front,
        "scale_length_cm": scale_length,
        "n_piston_peak_cm3": float(np.nanmax(n_piston_ion)),
        "amb_ni_cm3": ahead(n_ion),
        "amb_ne_cm3": ahead(np.asarray(lineout["ne"].to("cm**-3").value)),
        "amb_B_gauss": ahead(np.asarray(lineout["B_mag"].to("gauss").value)),
        "amb_Te_eV": ahead(np.asarray(lineout["Te"].to("eV").value)),
        "amb_Ti_eV": ahead(np.asarray(lineout["Ti"].to("eV").value)),
        "amb_zbar": ahead(zbar),
    }


def measure_unperturbed(lineout: dict, max_contaminant: float = 1.0e-6) -> dict:
    """The initially unperturbed background from the IC dump's line-out.

    Every cell containing target material is masked out; what remains is the chamber
    fill as ``flash.par`` set it up.  The laser needs no exclusion because at t = 0 it
    has not fired (the pulse ramps from 0.1 ns), so there is no channel yet.
    """
    piston_frac = np.asarray(lineout["piston_frac"], dtype=float)
    ye = np.asarray(lineout["ye"], dtype=float)
    sumy = np.asarray(lineout["sumy"], dtype=float)
    zbar = np.divide(ye, sumy, out=np.zeros_like(ye), where=sumy > 0.0)
    ne_cm3 = np.asarray(lineout["ne"].to("cm**-3").value, dtype=float)
    n_ion = np.divide(ne_cm3, zbar, out=np.zeros_like(ne_cm3), where=zbar > 0.0)

    def pristine(values) -> float:
        return pp.unperturbed_average(values, piston_frac, max_contaminant)

    return {
        "t_s": float(lineout["t_s"]),
        "amb_ni_cm3": pristine(n_ion),
        "amb_ne_cm3": pristine(ne_cm3),
        "amb_B_gauss": pristine(lineout["B_mag"].to("gauss").value),
        "amb_Te_eV": pristine(lineout["Te"].to("eV").value),
        "amb_Ti_eV": pristine(lineout["Ti"].to("eV").value),
        "amb_zbar": pristine(zbar),
        "amb_rho_gcm3": pristine(lineout["rho"].to("g/cm**3").value),
        "n_pristine_cells": int(np.count_nonzero(piston_frac <= max_contaminant)),
        "n_cells": int(piston_frac.size),
    }


def check_against_flash_par(unperturbed: dict, expected: dict | None) -> list[str]:
    """Compare the IC dump against the ``flash.par`` values recorded in the config.

    The dump is authoritative — it is what FLASH actually evolved, and it is the only
    place the EOS-determined ionisation state can be read (``flash.par`` gives the atomic
    number, not Zbar).  The cross-check exists to catch pointing the config at a dataset
    whose IC is not the one its comments describe.
    """
    if not expected:
        return []
    checks = (
        ("rho_g_cm3", "amb_rho_gcm3", 0.02),
        ("te_ev", "amb_Te_eV", 0.02),
        ("ti_ev", "amb_Ti_eV", 0.02),
        ("b_tesla", "amb_B_gauss", 0.02),
    )
    problems = []
    for key, measured_key, tolerance in checks:
        if key not in expected:
            continue
        want = float(expected[key])
        got = unperturbed[measured_key]
        if measured_key == "amb_B_gauss":
            got *= 1e-4
        if want != 0.0 and abs(got - want) / abs(want) > tolerance:
            problems.append(
                f"{key}: flash.par says {want:.6g}, the IC dump gives {got:.6g} "
                f"({100 * abs(got - want) / abs(want):.1f}% off)")
    return problems


def build_targets(per_dump: list, trajectory: pp.FrontTrajectory, *,
                  a_amb: float, a_piston: float, z_piston: float,
                  source: str, upstream: dict | None = None) -> hps.PistonTargets:
    """Window-averaged :class:`PistonTargets` in SI, ready for the run spec.

    The ambient state is averaged over ONLY the dumps whose upstream is still pristine.
    Once the diamagnetic cavity has swallowed the whole line-out there is no ambient left
    to measure, and including those dumps returns the state *inside* the cavity — field
    expelled, density high — which reads as beta ~ 160 and a sub-critical Mach number for
    a plainly super-critical shock.  The piston peak and the front trajectory are fine
    over the full window and use all of it.
    """
    # `upstream` given -> the unperturbed IC state; the per-dump ambient measurement is
    # then reported for comparison but not used.
    upstream_dumps = [upstream] if upstream else [d for d in per_dump
                                                 if d["has_upstream"]]
    if not upstream_dumps:
        raise SystemExit(
            "No dump in this window has a pristine upstream ahead of the front: the "
            "piston/cavity fills the whole line of sight. Use an earlier --t-window, "
            "push the config's line_of_sight end_point further out, or relax "
            "--contamination-max if you know what you are doing.")

    def window_mean(dumps: list, key: str) -> float:
        values = np.array([d[key] for d in dumps], dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            raise SystemExit(
                f"'{key}' is nan in every usable dump — the sampling band probably falls "
                f"outside the LOS. Lower --ambient-offset-um/--ambient-width-um or "
                f"extend the config's line_of_sight end_point.")
        return float(finite.mean())

    return hps.PistonTargets(
        n_amb_per_m3=1e6 * window_mean(upstream_dumps, "amb_ne_cm3"),
        b_amb_tesla=1e-4 * window_mean(upstream_dumps, "amb_B_gauss"),
        te_amb_ev=window_mean(upstream_dumps, "amb_Te_eV"),
        ti_amb_ev=window_mean(upstream_dumps, "amb_Ti_eV"),
        n_piston_drive_per_m3=1e6 * window_mean(per_dump, "n_piston_drive_cm3"),
        v_front_ms=1e-2 * abs(trajectory.speed),
        l_piston_m=1e-2 * window_mean(per_dump, "scale_length_cm"),
        r_spot_m=np.nan,          # filled in by the caller from the config
        t_window_s=float(per_dump[-1]["t_s"] - per_dump[0]["t_s"]),
        a_amb=a_amb,
        z_amb=window_mean(upstream_dumps, "amb_zbar"),
        a_piston=a_piston,
        z_piston=z_piston,
        source=source,
    )


def yaml_block(targets: hps.PistonTargets, source: flash_source.FlashSource,
               per_dump: list, trajectory: pp.FrontTrajectory) -> str:
    """The ``flash_target:`` block to paste into the heater run spec.

    Rendered with a dot in every mantissa and a sign on every exponent, because PyYAML
    is YAML 1.1 and would otherwise load these as strings (see CLAUDE.md).
    """
    t_lo_ns = per_dump[0]["t_s"] * 1e9
    t_hi_ns = per_dump[-1]["t_s"] * 1e9
    return "\n".join([
        "flash_target:",
        f"  source: {targets.source}, front fit rms "
        f"{trajectory.residual_rms / CM_PER_UM:.1f} um",
        f"  dataset: {source.flash_dir}",
        "  line_of_sight:",
        f"    start_point: {list(source.line_start)}",
        f"    end_point:   {list(source.line_end)}",
        f"  t_window_ns: [{t_lo_ns:.3f}, {t_hi_ns:.3f}]",
        "",
        f"  n_amb_per_m3: {targets.n_amb_per_m3:.4e}",
        f"  b_amb_tesla: {targets.b_amb_tesla:.6g}",
        f"  te_amb_ev: {targets.te_amb_ev:.6g}",
        f"  ti_amb_ev: {targets.ti_amb_ev:.6g}",
        f"  n_piston_drive_per_m3: {targets.n_piston_drive_per_m3:.4e}",
        f"  v_front_ms: {targets.v_front_ms:.4e}",
        f"  l_piston_m: {targets.l_piston_m:.4e}",
        f"  r_spot_m: {targets.r_spot_m:.4e}",
        "",
        f"  a_amb: {targets.a_amb:.6g}",
        f"  z_amb: {targets.z_amb:.6g}",
        f"  a_piston: {targets.a_piston:.6g}",
        f"  z_piston: {targets.z_piston:.6g}",
    ])


def flash_block(per_dump: list, trajectory: pp.FrontTrajectory,
                source: flash_source.FlashSource, *, upstream: dict | None,
                r_spot_um: float, ambient_species: str, piston_species: str) -> str:
    """The ``flash:`` block of a ``schema: heater_pic_2d`` run spec.

    Densities are ION densities, in the charge state each species is named with: the
    charge is stated once, in the species string, and ``src/warpx/config.py`` applies it.
    """
    ambient_dumps = ([upstream] if upstream
                     else [d for d in per_dump if d["has_upstream"]])

    def mean_of(dumps: list, key: str) -> float:
        values = np.array([d[key] for d in dumps], dtype=float)
        return float(np.nanmean(values))

    return "\n".join([
        "flash:",
        f"  dataset: {source.flash_dir}",
        f"  source: scripts/flash_piston_profile.py over {len(per_dump)} dumps, "
        f"front fit rms {trajectory.residual_rms / CM_PER_UM:.0f} um",
        f"  window_ns: [{per_dump[0]['t_s'] * 1e9:.3f}, "
        f"{per_dump[-1]['t_s'] * 1e9:.3f}]",
        "",
        "  ambient:",
        f"    species: {ambient_species}",
        f"    ion_density_per_m3: {1e6 * mean_of(ambient_dumps, 'amb_ni_cm3'):.4e}",
        f"    magnetic_field_tesla: {1e-4 * mean_of(ambient_dumps, 'amb_B_gauss'):.6g}",
        f"    electron_temperature_ev: {mean_of(ambient_dumps, 'amb_Te_eV'):.6g}",
        f"    ion_temperature_ev: {mean_of(ambient_dumps, 'amb_Ti_eV'):.6g}",
        "",
        "  piston:",
        f"    species: {piston_species}",
        f"    ion_density_per_m3: {1e6 * mean_of(per_dump, 'n_piston_drive_cm3'):.4e}",
        f"    front_speed_km_s: {abs(trajectory.speed) * 1e-5:.5g}",
        f"    spot_radius_um: {r_spot_um:.6g}",
    ])


def upstream_comparison(unperturbed: dict, per_dump: list) -> list[str]:
    """The unperturbed IC background against the state measured ahead of the front.

    Printed side by side because the two differ by a lot and the choice between them
    changes every dimensionless number.  The measured column is the preheated precursor;
    the IC column is the background the experiment was set up with.
    """
    upstream_dumps = [d for d in per_dump if d["has_upstream"]]
    if not upstream_dumps:
        return []

    def measured(key: str) -> float:
        values = np.array([d[key] for d in upstream_dumps], dtype=float)
        finite = values[np.isfinite(values)]
        return float(finite.mean()) if finite.size else float("nan")

    rows = [
        ("n_e [cm^-3]", unperturbed["amb_ne_cm3"], measured("amb_ne_cm3")),
        ("|B| [T]", unperturbed["amb_B_gauss"] * 1e-4, measured("amb_B_gauss") * 1e-4),
        ("T_e [eV]", unperturbed["amb_Te_eV"], measured("amb_Te_eV")),
        ("T_i [eV]", unperturbed["amb_Ti_eV"], measured("amb_Ti_eV")),
        ("Zbar", unperturbed["amb_zbar"], measured("amb_zbar")),
    ]
    lines = [
        "",
        "Upstream state: initially unperturbed background vs measured ahead of the front",
        f"  {'quantity':<14} {'IC (t=0)':>12} {'ahead of front':>15} {'ratio':>8}",
    ]
    for name, initial, ahead in rows:
        ratio = ahead / initial if initial else float("nan")
        lines.append(f"  {name:<14} {initial:>12.4g} {ahead:>15.4g} {ratio:>8.2f}")
    lines.append("  (the 'ahead of front' column is conduction/radiation preheated — a "
                 "precursor,")
    lines.append("   not the background; --upstream selects which one is used)")
    return lines


def summary(targets: hps.PistonTargets, per_dump: list,
            trajectory: pp.FrontTrajectory) -> str:
    """Per-dump table plus the derived dimensionless state."""
    n_upstream = sum(1 for d in per_dump if d["has_upstream"])
    n_resolved = sum(1 for d in per_dump if d["edge_resolved"])
    lines = [
        "Per-dump measurement (CGS; front and scale length along the LOS).",
        "'up'      pristine upstream ahead of the front — only these are averaged into",
        "          the ambient state below.",
        "'res'     the fitted L spans >=2 line-out samples. 'NO' means the piston edge is",
        "          grid-sharp and L is not a physical width (see the note below).",
        "n_drive   piston density just BEHIND the front — what drives the shock.",
        "n_peak    global profile peak: the stagnated material next to the target, which",
        "          never reaches the front. Reported for contrast only; NOT used.",
        f"  {'t [ns]':>8} {'x_front [um]':>13} {'L [um]':>9} {'res':>4} "
        f"{'n_drive':>11} {'n_peak':>11} {'n_amb':>11} {'|B| [T]':>9} "
        f"{'Te [eV]':>9} {'Ti [eV]':>9} {'Zbar':>6} {'up':>4}",
    ]
    for dump in per_dump:
        lines.append(
            f"  {dump['t_s'] * 1e9:>8.3f} {dump['x_front_cm'] / CM_PER_UM:>13.1f} "
            f"{dump['scale_length_cm'] / CM_PER_UM:>9.1f} "
            f"{'yes' if dump['edge_resolved'] else 'NO':>4} "
            f"{dump['n_piston_drive_cm3']:>11.3e} "
            f"{dump['n_piston_peak_cm3']:>11.3e} {dump['amb_ne_cm3']:>11.3e} "
            f"{dump['amb_B_gauss'] * 1e-4:>9.3f} {dump['amb_Te_eV']:>9.1f} "
            f"{dump['amb_Ti_eV']:>9.1f} {dump['amb_zbar']:>6.2f} "
            f"{'yes' if dump['has_upstream'] else '-':>4}")

    lines += [
        "",
        f"Front trajectory: v = {trajectory.speed / 1e5:.1f} km/s over "
        f"{trajectory.n_points} dumps (fit rms "
        f"{trajectory.residual_rms / CM_PER_UM:.1f} um)",
        f"Dumps with a pristine upstream: {n_upstream}/{len(per_dump)}",
        f"Dumps with a resolved piston edge: {n_resolved}/{len(per_dump)}",
        "",
        "Ambient averaged over the upstream-bearing dumps; derived dimensionless state",
        f"  n_e = {targets.n_amb_per_m3:.3e} m^-3   n_i = "
        f"{targets.n_i_amb_per_m3:.3e} m^-3   Zbar = {targets.z_amb:.2f}",
        f"  |B| = {targets.b_amb_tesla:.2f} T   T_e = {targets.te_amb_ev:.1f} eV   "
        f"T_i = {targets.ti_amb_ev:.1f} eV",
        f"  v_A = {targets.v_alfven_ms / 1e3:.1f} km/s   c_s = "
        f"{targets.c_s_ms / 1e3:.1f} km/s   v_fast = "
        f"{targets.v_fast_ms / 1e3:.1f} km/s",
        f"  d_i = {targets.d_i_m * 1e6:.1f} um   T_ci = "
        f"{targets.gyroperiod_s * 1e9:.3f} ns   window = "
        f"{targets.t_window_gyro:.3f} T_ci",
        "",
        f"  M_A  = {targets.mach_alfven:.3f}    M_ms = {targets.mach_magnetosonic:.3f}"
        f"   ({'SUPER' if targets.mach_magnetosonic > 2.76 else 'sub'}-critical; "
        f"the ion-reflection threshold is 2.76)",
        f"  beta_e = {targets.beta_e:.3f}   beta_i = {targets.beta_i:.3f}",
        f"  n_piston/n_amb = {targets.contrast:.2f}   "
        f"L_piston/d_i = {targets.l_piston_di:.3f}   "
        f"r_spot/d_i = {targets.r_spot_di:.3f}",
    ]
    if n_resolved < len(per_dump):
        lines += [
            "",
            "NOTE: the piston edge is grid-sharp in "
            f"{len(per_dump) - n_resolved}/{len(per_dump)} dumps, so l_piston_m is the "
            "gradient of the",
            "plateau behind the edge, not the width of the interface. That is expected "
            "— this",
            "FLASH run is ideal MHD (useMagneticResistivity = .false.), so nothing in "
            "the",
            "equations gives the piston/pile-up contact a finite width; see",
            "docs/piston_interface_smoothing_plan.md. It is also the reason the WarpX "
            "deck",
            "GROWS its piston with the heating operator rather than extracting this "
            "one, so",
            "l_piston_m is carried as provenance and is NOT one of the matched "
            "invariants.",
        ]
    return "\n".join(lines)


def plot(per_dump: list, targets: hps.PistonTargets, trajectory: pp.FrontTrajectory,
         n_profiles: int, out_path: str) -> None:
    """Four panels: piston profiles, their collapse, the trajectory, the ambient."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    ax_profile, ax_collapse, ax_trajectory, ax_ambient = axes.flat

    drawn = per_dump if len(per_dump) <= n_profiles else [
        per_dump[i] for i in np.linspace(0, len(per_dump) - 1, n_profiles).astype(int)]
    colors = plt.cm.viridis(np.linspace(0.0, 0.9, len(drawn)))

    for dump, color in zip(drawn, colors):
        label = f"{dump['t_s'] * 1e9:.2f} ns"
        x_um = dump["x_cm"] / CM_PER_UM
        ax_profile.semilogy(x_um, dump["n_piston_cm3"], color=color, lw=1.5, label=label)
        ax_profile.semilogy(x_um, dump["ne_cm3"], color=color, lw=0.8, alpha=0.4, ls=":")
        if np.isfinite(dump["x_front_cm"]):
            ax_profile.axvline(dump["x_front_cm"] / CM_PER_UM, color=color,
                               lw=0.8, alpha=0.6)
        # Normalise on the DRIVE density, not the global peak: the peak lives in the
        # stagnated material next to the target and swamps the front region, which left
        # the collapsed curves spread over more than a decade for no physical reason.
        drive = dump["n_piston_drive_cm3"]
        if np.isfinite(drive) and drive > 0.0:
            xi, _ = pp.collapse_profile(dump["x_cm"], dump["n_piston_cm3"],
                                        dump["x_front_cm"], dump["scale_length_cm"])
            ax_collapse.semilogy(xi, dump["n_piston_cm3"] / drive, color=color,
                                 lw=1.5, label=label)

    # FLASH's smallx floor puts the piston mass fraction at ~1e-99 outside the plume, so
    # an unclipped log axis spans 40 decades of pure floor and shows nothing.
    ax_profile.set_ylim(1e-3 * np.nanmin([d["amb_ne_cm3"] for d in per_dump]),
                        3.0 * np.nanmax([d["n_piston_peak_cm3"] for d in per_dump]))
    ax_profile.set_xlabel(r"distance along LOS [$\mu$m]")
    ax_profile.set_ylabel(r"$n_e$ [cm$^{-3}$]")
    ax_profile.set_title("piston (solid) vs total $n_e$ (dotted); vlines = front")
    ax_profile.legend(fontsize=8)
    ax_profile.grid(alpha=0.25, which="both")

    ax_collapse.axhline(1.0, color="0.5", lw=0.8, ls="--")
    ax_collapse.set_xlim(-6.0, 2.0)
    ax_collapse.set_ylim(1e-3, 1e2)
    ax_collapse.set_xlabel(r"$(s - s_\mathrm{front}) / L$")
    ax_collapse.set_ylabel(r"$n / n_\mathrm{drive}$")
    ax_collapse.set_title("collapse on the drive density (overlapping = self-similar)")
    ax_collapse.grid(alpha=0.25, which="both")

    times_ns = np.array([d["t_s"] for d in per_dump]) * 1e9
    fronts_um = np.array([d["x_front_cm"] for d in per_dump]) / CM_PER_UM
    upstream = np.array([d["has_upstream"] for d in per_dump])
    ax_trajectory.plot(times_ns[upstream], fronts_um[upstream], "o", ms=5,
                       color="#1f77b4", label="front (upstream intact)")
    if np.any(~upstream):
        ax_trajectory.plot(times_ns[~upstream], fronts_um[~upstream], "x", ms=7,
                           color="#d62728", label="front (no upstream left)")
    fit_um = trajectory.at(np.array([d["t_s"] for d in per_dump])) / CM_PER_UM
    ax_trajectory.plot(times_ns, fit_um, "-", color="k", lw=1.6,
                       label=f"fit: {trajectory.speed / 1e5:.0f} km/s")
    ax_trajectory.set_xlabel("t [ns]")
    ax_trajectory.set_ylabel(r"front position along LOS [$\mu$m]")
    ax_trajectory.set_title(
        f"front trajectory (rms {trajectory.residual_rms / CM_PER_UM:.0f} "
        r"$\mu$m)")
    ax_trajectory.legend(fontsize=8)
    ax_trajectory.grid(alpha=0.25)

    # The ambient the piston is driving into, dump by dump: this is what sets M_A and
    # beta, and it is NOT the initial condition (the chamber rarefies as the run goes).
    ax_ambient.semilogy(times_ns, [d["amb_ne_cm3"] for d in per_dump], "o-",
                        color="#d62728", label=r"ambient $n_e$ [cm$^{-3}$]")
    ax_ambient.set_xlabel("t [ns]")
    ax_ambient.set_ylabel(r"ambient $n_e$ [cm$^{-3}$]", color="#d62728")
    ax_ambient.tick_params(axis="y", labelcolor="#d62728")
    ax_ambient.grid(alpha=0.25, which="both")
    twin = ax_ambient.twinx()
    twin.plot(times_ns, [d["amb_B_gauss"] * 1e-4 for d in per_dump], "s--",
              color="#2ca02c", label="ambient |B| [T]")
    twin.set_ylabel("ambient |B| [T]", color="#2ca02c")
    twin.tick_params(axis="y", labelcolor="#2ca02c")
    ax_ambient.set_title("ambient ahead of the front (not the initial condition)")

    fig.suptitle(
        f"FLASH piston along the LOS — window-averaged "
        f"$M_A$ = {targets.mach_alfven:.2f}, "
        f"$M_{{ms}}$ = {targets.mach_magnetosonic:.2f}, "
        f"$\\beta_e$ = {targets.beta_e:.2f}, "
        f"$n_\\mathrm{{piston}}/n_\\mathrm{{amb}}$ = {targets.contrast:.1f}",
        fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    plot_style.apply(args.publication)

    cfg = analysis_utils.load_config(args.config)
    source = flash_source.resolve(cfg, args.config)
    out_dir = yaml_edit.out_dir(source.flash_dir, args.output_dir, cfg=cfg,
                                config_path=args.config)

    piston_material = str(cfg.get("piston_material", "targ"))
    extra_fields = dict(EXTRA_FIELDS)
    extra_fields["piston_frac"] = ("flash", piston_material)

    print(f"FLASH data : {source.flash_dir}   ({source.source})")
    print(f"LOS        : {source.line_start} -> {source.line_end} cm")
    print(f"piston mat : {piston_material}")

    all_paths = fu.find_plot_files(source.flash_dir)
    paths, _ = select_dumps(all_paths, args.t_window)
    print(f"dumps      : {len(paths)}"
          + (f" in t = {args.t_window} ns" if args.t_window else " (all)"))
    print(f"upstream   : {args.upstream}")

    # The IC dump is loaded separately from the window: the upstream state comes from the
    # unperturbed background, while the piston and its trajectory come from the evolved
    # dumps (there is no piston at t = 0).
    unperturbed = None
    if args.upstream == "initial":
        ic_path = all_paths[source.ic_index]
        ic_lineout = fu.flash_lineout(ic_path, source.line_start, source.line_end,
                                      npoints=args.npoints, extra_fields=extra_fields)
        unperturbed = measure_unperturbed(ic_lineout)
        print(f"IC dump    : {os.path.basename(ic_path)} at "
              f"t = {unperturbed['t_s'] * 1e9:.4f} ns, "
              f"{unperturbed['n_pristine_cells']}/{unperturbed['n_cells']} cells "
              f"free of target material")
        problems = check_against_flash_par(unperturbed,
                                           cfg.get("unperturbed_background"))
        for problem in problems:
            print(f"  WARNING: IC dump disagrees with the config's flash.par values — "
                  f"{problem}")

    lineouts = fu.load_lineouts(paths, source.line_start, source.line_end,
                                nprocs=args.nprocs, npoints=args.npoints,
                                extra_fields=extra_fields)

    a_piston = float(cfg.get("a_piston", 28.0855))
    per_dump = [
        measure_dump(lineout, mass_number=a_piston,
                     front_level_frac=args.front_level_frac,
                     front_threshold=args.front_threshold,
                     contamination_max=args.contamination_max,
                     fit_window_cm=args.fit_window_um * CM_PER_UM,
                     ambient_offset_cm=args.ambient_offset_um * CM_PER_UM,
                     ambient_width_cm=args.ambient_width_um * CM_PER_UM,
                     drive_offset_cm=args.drive_offset_um * CM_PER_UM,
                     drive_width_cm=args.drive_width_um * CM_PER_UM)
        for lineout in lineouts
    ]

    trajectory = pp.fit_front_trajectory(
        np.array([d["t_s"] for d in per_dump]),
        np.array([d["x_front_cm"] for d in per_dump]))

    targets = build_targets(
        per_dump, trajectory,
        a_amb=float(cfg.get("a_amb", 26.98)), a_piston=a_piston,
        z_piston=float(cfg.get("z_piston", 14.0)),
        source=(f"scripts/flash_piston_profile.py, piston over {len(per_dump)} dumps, "
                f"upstream = {args.upstream}"),
        upstream=unperturbed)
    # The transverse piston scale is the laser spot, which no line-out along the
    # expansion axis can see; it comes from the FLASH runtime parameters
    # (flash.par ed_gaussianRadiusMajor_1 = 500e-4 cm for this dataset).
    r_spot_um = float(cfg.get("laser_spot_radius_um", 500.0))
    targets = dataclasses.replace(targets, r_spot_m=r_spot_um * 1e-6)

    text = summary(targets, per_dump, trajectory)
    if unperturbed is not None:
        text += "\n" + "\n".join(upstream_comparison(unperturbed, per_dump))
    print()
    print(text)

    block = flash_block(
        per_dump, trajectory, source, upstream=unperturbed, r_spot_um=r_spot_um,
        ambient_species=str(cfg.get("ambient_species", "Al 6+")),
        piston_species=str(cfg.get("piston_species", "Si 14+")))
    print()
    print("Paste into runs/magshockz_2d_heater.warpx.yaml:")
    print()
    print(block)

    txt_path = os.path.join(out_dir, "flash_piston_profile.txt")
    with open(txt_path, "w") as handle:
        handle.write(text + "\n\n" + block + "\n\n"
                     + yaml_block(targets, source, per_dump, trajectory) + "\n")

    png_path = os.path.join(out_dir, "flash_piston_profile.png")
    plot(per_dump, targets, trajectory, args.n_profiles, png_path)

    npz_path = os.path.join(out_dir, "flash_piston_profile.npz")
    np.savez(
        npz_path,
        x_cm=per_dump[0]["x_cm"],
        t_s=np.array([d["t_s"] for d in per_dump]),
        n_piston_cm3=np.array([d["n_piston_cm3"] for d in per_dump]),
        ne_cm3=np.array([d["ne_cm3"] for d in per_dump]),
        B_mag_gauss=np.array([d["B_mag_gauss"] for d in per_dump]),
        Te_eV=np.array([d["Te_eV"] for d in per_dump]),
        Ti_eV=np.array([d["Ti_eV"] for d in per_dump]),
        v_para_cms=np.array([d["v_para_cms"] for d in per_dump]),
        x_front_cm=np.array([d["x_front_cm"] for d in per_dump]),
        scale_length_cm=np.array([d["scale_length_cm"] for d in per_dump]),
        n_piston_peak_cm3=np.array([d["n_piston_peak_cm3"] for d in per_dump]),
        n_piston_drive_cm3=np.array([d["n_piston_drive_cm3"] for d in per_dump]),
        edge_resolved=np.array([d["edge_resolved"] for d in per_dump]),
        amb_ne_cm3=np.array([d["amb_ne_cm3"] for d in per_dump]),
        amb_B_gauss=np.array([d["amb_B_gauss"] for d in per_dump]),
        amb_Te_eV=np.array([d["amb_Te_eV"] for d in per_dump]),
        amb_Ti_eV=np.array([d["amb_Ti_eV"] for d in per_dump]),
        has_upstream=np.array([d["has_upstream"] for d in per_dump]),
        ambient_reference_cm3=np.array([d["ambient_reference_cm3"] for d in per_dump]),
        front_speed_cms=np.asarray(trajectory.speed),
        front_x0_cm=np.asarray(trajectory.x0),
        front_t0_s=np.asarray(trajectory.t0),
        front_residual_rms_cm=np.asarray(trajectory.residual_rms),
        **{f"target_{key}": np.asarray(value)
           for key, value in dataclasses.asdict(targets).items()},
        **{f"invariant_{key.replace('/', '_over_')}": np.asarray(value)
           for key, value in targets.invariants().items()},
        config_path=np.asarray(os.path.abspath(args.config)),
    )

    for path in (png_path, npz_path, txt_path):
        print(f"Saved -> {path}")


if __name__ == "__main__":
    main()
