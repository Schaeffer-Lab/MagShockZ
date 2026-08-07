# -*- coding: utf-8 -*-
"""scripts/warpx_heater_compare.py — WarpX heater piston vs the FLASH piston it targets.

Closes the tuning loop for the 2D heater deck.  Reads the WarpX plotfile series, measures
the piston with the SAME estimators the FLASH side used (``src/piston_profile.py``), and
overlays the two in the variables that are actually matched.

Both codes are measured identically, which is the point: the comparison is only
meaningful because ``front_position`` / ``behind_front_average`` /
``ahead_of_front_average`` do not know which code produced the array they are handed.

WHAT IS COMPARABLE.  The deck runs at a reduced mass ratio and an arbitrary reference
density, so absolute ns and um are NOT comparable — only the dimensionless invariants
are (see ``magshockz/init/warpx/units.py``).  Where a physical axis is wanted,
``DeckScales.to_time`` / ``to_length`` bridge through the matched ion scales (d_i, T_ci);
every such axis is labelled "FLASH-equivalent" to keep that explicit.

Panels
------
1. piston density profile, WarpX vs FLASH, in d_i from the front
2. front trajectory in gyroperiods, both codes, with the target speed
3. the operator sanity check: ParticleEnergy / ParticleNumber histories (heater and
   injector in balance = EP plateaus, PN keeps rising) plus the ambient <u^2>, whose
   upward drift is the numerical grid heating this deck's main risk
4. the invariant scorecard: measured WarpX vs FLASH target

Usage
-----
    conda activate analysis
    python scripts/warpx_heater_compare.py --config runs/magshockz_2d_heater.warpx.yaml \\
        [--diag-dir ...] [--flash-npz ...] [--output-dir ...] [--pub]

Run in the `analysis` conda env (yt).  The 2D CPU WarpX build has openPMD off, so the
diagnostics are AMReX plotfiles and yt reads them directly.
"""

import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))

import astropy.units as u
from astropy.constants import c, e

from magshockz.analysis.warpx import metrics
from magshockz.init.warpx import config as spec_config
from magshockz.init.warpx import deck as deck_module
from magshockz.init.warpx import units
from magshockz.common import piston_profile as pp
from magshockz.common import plot_style
from magshockz.common import yaml_edit

_REPO = os.path.abspath(os.path.join(_HERE, ".."))

#: The deck species this script reads.  Checked against the generator's own list rather
#: than mirrored in a comment, so a rename there fails here loudly instead of leaving
#: the script asking a plotfile for a field that no longer exists.
PISTON_IONS = "piston_ions"
AMBIENT_IONS = "amb_ions"
AMBIENT_ELECTRONS = "amb_electrons"
assert {PISTON_IONS, AMBIENT_IONS, AMBIENT_ELECTRONS} <= set(deck_module.SPECIES_NAMES)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare the WarpX heater piston to its FLASH target.")
    parser.add_argument("--config", required=True,
                        help="heater_pic_2d run spec (runs/*.warpx.yaml)")
    parser.add_argument("--diag-dir",
                        help="plotfile directory (default: the run's diags/)")
    parser.add_argument("--flash-npz",
                        help="flash_piston_profile.npz (default: found from the spec's "
                             "flash_target.dataset)")
    parser.add_argument("--stride", type=int, default=1,
                        help="use every Nth plotfile (default: %(default)s)")
    parser.add_argument("--front-level-frac", type=float, default=1.0,
                        help="front = piston density crossing this multiple of the "
                             "ambient (default: %(default)s)")
    parser.add_argument("--output-dir", help="override the results directory")
    plot_style.add_publication_arg(parser)
    return parser.parse_args()


def load_spec_and_scales(config_path: str) -> tuple[dict, units.DeckScales]:
    """Load the run spec and re-derive its scales, so the bridge is never stale.

    Re-deriving rather than reading the frozen ``run.yaml`` means editing the spec and
    re-running this script cannot silently compare against the previous deck's scales.
    """
    spec = spec_config.load(config_path)
    return spec, spec_config.scales(spec, smoke=False)


def find_plotfiles(spec: dict, config_path: str, override: str | None) -> list[str]:
    """Sorted ``diag1*`` plotfile paths for this run."""
    if override:
        search_dirs = [override]
    else:
        run_name = os.path.basename(config_path).replace(".warpx.yaml", "")
        run_dir = os.path.join(_REPO, "input_files", "warpx", run_name)
        search_dirs = [os.path.join(run_dir, "diags"), run_dir]

    for directory in search_dirs:
        paths = sorted(glob.glob(os.path.join(directory, "diag1*")))
        paths = [p for p in paths if os.path.isdir(p)]
        if paths:
            return paths
    raise SystemExit(
        f"No diag1* plotfiles found under {search_dirs}. Run the deck first:\n"
        f"  sbatch init_warpx/run_heater_2d.sbatch")


def transverse_average(dataset, field: str, scales: units.DeckScales
                       ) -> tuple[np.ndarray, np.ndarray]:
    """z-profile of ``field``, averaged over x, from a 2D WarpX plotfile.

    Averaging over the full transverse extent (rather than a cut through the spot) is
    what makes the profile comparable to a FLASH line-out: both then represent the
    piston as a whole rather than one ray through it.  Returns ``(z in d_e, values)``.
    """
    grid = dataset.covering_grid(level=0, left_edge=dataset.domain_left_edge,
                                 dims=dataset.domain_dimensions)
    values = np.asarray(grid["boxlib", field])
    # 2D WarpX plotfiles are (nx, nz, 1); average over x and drop the dummy axis.
    profile = values.mean(axis=0).squeeze()
    z_edges = np.linspace(float(dataset.domain_left_edge[1]),
                          float(dataset.domain_right_edge[1]), profile.size + 1)
    z_centres = 0.5 * (z_edges[:-1] + z_edges[1:])
    return z_centres / scales.electron_skin_depth.to_value(u.m), profile


def measure_plotfile(path: str, scales: units.DeckScales, *,
                     front_level_frac: float) -> dict:
    """Piston front, drive density and ambient state from one WarpX plotfile.

    The slab is symmetric about z = 0 and expands both ways, so only the +z half is
    measured; ``piston_profile``'s estimators all assume the piston lies inward of the
    front, which is exactly the +z half's geometry.
    """
    import yt

    dataset = yt.load(path)
    time = float(dataset.current_time) * u.s

    z_de, piston_charge = transverse_average(dataset, f"rho_{PISTON_IONS}", scales)
    _, ambient_charge = transverse_average(dataset, f"rho_{AMBIENT_IONS}", scales)
    _, ambient_usq = transverse_average(dataset, f"usq_{AMBIENT_ELECTRONS}", scales)

    # rho is a charge density; the species carry +q_e, so n = rho/q_e.
    piston_density = piston_charge / e.si.value
    ambient_density = ambient_charge / e.si.value

    outward = z_de >= 0.0
    z_out = z_de[outward]
    piston_out = piston_density[outward]
    ambient_out = ambient_density[outward]

    level = front_level_frac * pp.ambient_reference_level(z_out, ambient_out)
    x_front = pp.front_position(z_out, piston_out, level=level if level > 0 else None)

    # Bands scaled to the run's own geometry, so they mean the same thing as the FLASH
    # side's (which used ~1 d_i inward and ~6 d_i outward, in its own d_i).
    d_i_de = scales.di_over_de
    drive = pp.behind_front_average(z_out, piston_out, x_front,
                                    offset=0.2 * d_i_de, width=1.2 * d_i_de)
    ambient = pp.ahead_of_front_average(z_out, ambient_out, x_front,
                                        offset=6.0 * d_i_de, width=2.5 * d_i_de)

    return {
        "t_omega_pe": float((time * scales.upstream.plasma_frequency / u.rad).decompose()),
        "t_gyro": float((time / scales.gyroperiod).decompose()),
        "z_de": z_out,
        "piston_density": piston_out,
        "ambient_density": ambient_out,
        "x_front_de": x_front,
        "n_drive": drive,
        "n_ambient": ambient,
        "contrast": drive / ambient if ambient and np.isfinite(ambient) else np.nan,
        "ambient_usq": float(np.nanmean(ambient_usq[outward])),
    }


def load_reduced_diags(plot_paths: list[str]) -> dict[str, np.ndarray]:
    """ParticleEnergy / ParticleNumber histories from ``diags/reducedfiles/``.

    Optional: a run configured without them still produces every other panel.
    """
    reduced_dir = os.path.join(os.path.dirname(plot_paths[0]), "reducedfiles")
    out: dict[str, np.ndarray] = {}
    for name in ("EP", "PN"):
        path = os.path.join(reduced_dir, f"{name}.txt")
        if not os.path.isfile(path):
            continue
        try:
            table = np.loadtxt(path, skiprows=1)
        except (ValueError, OSError):
            continue
        if table.ndim == 2 and table.shape[1] >= 3:
            out[f"{name}_t"] = table[:, 1]
            out[f"{name}_total"] = table[:, 2]
    return out


def load_flash_target(spec: dict, override: str | None) -> dict | None:
    """The FLASH measurement arrays, if ``flash_piston_profile.py`` has been run."""
    if override:
        path = override
    else:
        dataset = spec["flash"].get("dataset", "")
        stem = os.path.basename(str(dataset).rstrip("/"))
        candidates = sorted(glob.glob(os.path.join(
            _REPO, "results", stem, "**", "flash_piston_profile.npz"), recursive=True))
        if not candidates:
            return None
        path = candidates[0]
    if not os.path.isfile(path):
        return None
    with np.load(path, allow_pickle=True) as handle:
        return {key: handle[key] for key in handle.files}


def measure_run(per_dump: list, scales: units.DeckScales) -> list[metrics.ScoreRow]:
    """Fit the front across the dump series and score it against FLASH and the deck."""
    tracked = [d for d in per_dump if np.isfinite(d["x_front_de"])]
    speed_over_c = metrics.front_speed_over_c(
        np.array([d["t_omega_pe"] for d in tracked]),
        np.array([d["x_front_de"] for d in tracked]),
        di_over_de=scales.di_over_de)
    if len(tracked) >= 2 and not np.isfinite(speed_over_c):
        travel_di = (abs(tracked[-1]["x_front_de"] - tracked[0]["x_front_de"])
                     / scales.di_over_de)
        print(f"NOTE: the front has moved only {travel_di:.3f} d_i over "
              f"{tracked[-1]['t_gyro'] - tracked[0]['t_gyro']:.4f} T_ci — too little to "
              f"fit a speed (need {metrics.MIN_TRAVEL_DI} d_i). Speed/M_A left as nan; "
              f"run the full deck, not the smoke one.")

    contrast = (float(np.nanmean([d["contrast"] for d in tracked]))
                if tracked else float("nan"))
    return metrics.scorecard(scales, measured_speed_over_c=speed_over_c,
                             measured_contrast=contrast)


def plot(per_dump: list, reduced: dict, flash: dict | None,
         scales: units.DeckScales, rows: list[metrics.ScoreRow],
         out_path: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    ax_profile, ax_trajectory, ax_operator, ax_score = axes.flat
    d_i_de = scales.di_over_de
    flash_di = scales.flash.upstream.ion_skin_depth
    flash_gyroperiod = scales.flash.upstream.gyroperiod

    drawn = [per_dump[i] for i in
             np.linspace(0, len(per_dump) - 1, min(5, len(per_dump))).astype(int)]
    colors = plt.cm.plasma(np.linspace(0.0, 0.85, len(drawn)))
    for dump, color in zip(drawn, colors):
        if not (np.isfinite(dump["x_front_de"]) and np.isfinite(dump["n_drive"])
                and dump["n_drive"] > 0.0):
            continue
        xi = (dump["z_de"] - dump["x_front_de"]) / d_i_de
        ax_profile.semilogy(xi, dump["piston_density"] / dump["n_drive"], color=color,
                            lw=1.6, label=f"WarpX {dump['t_gyro']:.3f} $T_{{ci}}$")

    if flash is not None:
        flash_x = flash["x_cm"]
        for index in np.linspace(0, len(flash["t_s"]) - 1, 3).astype(int):
            front = flash["x_front_cm"][index]
            drive = flash["n_piston_drive_cm3"][index]
            if not (np.isfinite(front) and np.isfinite(drive) and drive > 0):
                continue
            xi = (flash_x - front) / flash_di.to_value(u.cm)
            ax_profile.semilogy(xi, flash["n_piston_cm3"][index] / drive, "--",
                                color="0.35", lw=1.2,
                                label="FLASH" if index == 0 else None)

    ax_profile.axvline(0.0, color="k", lw=0.8, alpha=0.5)
    ax_profile.set_xlim(-8.0, 3.0)
    ax_profile.set_ylim(1e-3, 1e2)
    ax_profile.set_xlabel(r"$(z - z_\mathrm{front}) / d_i$")
    ax_profile.set_ylabel(r"$n_\mathrm{piston} / n_\mathrm{drive}$")
    ax_profile.set_title("piston profile in ion units (WarpX solid, FLASH dashed)")
    ax_profile.legend(fontsize=7)
    ax_profile.grid(alpha=0.25, which="both")

    gyro = np.array([d["t_gyro"] for d in per_dump])
    fronts_di = np.array([d["x_front_de"] for d in per_dump]) / d_i_de
    ax_trajectory.plot(gyro, fronts_di, "o-", ms=4, color="#e377c2", label="WarpX front")
    finite = np.isfinite(gyro) & np.isfinite(fronts_di)
    if np.count_nonzero(finite) >= 2:
        # The intended slope in these axes: v_piston in d_i per gyroperiod.
        intended = float((scales.piston_speed * scales.gyroperiod
                          / scales.ion_skin_depth).decompose())
        ax_trajectory.plot(gyro, fronts_di[finite][0] + intended
                           * (gyro - gyro[finite][0]), "k--", lw=1.4,
                           label=f"deck target ({scales.piston_speed_over_c:.3f} c)")
    if flash is not None:
        flash_gyro = (flash["t_s"] - flash["t_s"][0]) / flash_gyroperiod.to_value(u.s)
        flash_front_di = ((flash["x_front_cm"] - flash["x_front_cm"][0])
                          / flash_di.to_value(u.cm))
        ax_trajectory.plot(flash_gyro, flash_front_di + fronts_di[finite][0]
                           if np.any(finite) else flash_front_di, "-", color="0.35",
                           lw=1.4, label="FLASH front")
    ax_trajectory.set_xlabel(r"$t / T_{ci}$")
    ax_trajectory.set_ylabel(r"front position [$d_i$]")
    ax_trajectory.set_title("front trajectory in matched units")
    ax_trajectory.legend(fontsize=8)
    ax_trajectory.grid(alpha=0.25)

    # The reduced diagnostics timestamp in seconds, as the plotfiles do.
    gyroperiod = scales.gyroperiod.to_value(u.s)
    for name, color, label in (("EP", "#ff7f0e", "ParticleEnergy / initial"),
                               ("PN", "#1f77b4", "ParticleNumber / initial")):
        if f"{name}_t" not in reduced:
            continue
        total = reduced[f"{name}_total"]
        ax_operator.plot(reduced[f"{name}_t"] / gyroperiod,
                         total / total[0] if total[0] else total,
                         color=color, label=label)
    # The ambient must stay cold: a rising <u^2> upstream is numerical grid heating, the
    # failure mode SHOCK_PLAN.md flags for long runs of a cold magnetized ambient.
    usq = np.array([d["ambient_usq"] for d in per_dump])
    if np.any(np.isfinite(usq)) and usq[0] > 0:
        ax_operator.plot(gyro, usq / usq[0], "s--", ms=4, color="#d62728",
                         label=r"ambient $\langle u^2\rangle$ / initial")
    ax_operator.set_xlabel(r"$t / T_{ci}$")
    ax_operator.set_ylabel("normalised to t = 0")
    ax_operator.set_title("operator balance + upstream grid heating")
    ax_operator.legend(fontsize=8)
    ax_operator.grid(alpha=0.25)

    ax_score.axis("off")
    text = [
        metrics.scorecard_text(rows),
        "",
        "FLASH -> deck aim is the mapping (magshockz/init/warpx/units.py);",
        "deck aim -> WarpX is whether the run realised it.",
        "",
        f"theta_e = {scales.theta_e_heater:.4g}   "
        f"B0 = {scales.magnetic_field.to_value(u.mT):.4g} mT",
        f"n_target/n0 = {scales.contrast:.4g}   slab = "
        f"{scales.slab_halfwidth_di:.2f} d_i   r_H = {scales.spot_radius_di:.2f} d_i",
        "",
        "Tuning knobs, in the order worth trying:",
        "  front too slow/fast     -> add this run to calibration: (the setpoint is",
        "                             fitted from completed runs, never modelled)",
        "  contrast off            -> flash.piston.ion_density_per_m3",
        "  piston runs out of mass -> geometry.slab_halfwidth_di,",
        "                             operators.injector.tau_over_wpe_inv",
        "  profile too narrow in x -> flash.piston.spot_radius_um",
    ]
    ax_score.text(0.0, 1.0, "\n".join(text), family="monospace", fontsize=9,
                  va="top", ha="left", transform=ax_score.transAxes)
    ax_score.set_title("invariant scorecard", loc="left")

    fig.suptitle("WarpX heater piston vs FLASH target "
                 f"(m/Ze {units.mass_per_charge(scales.upstream.ion):.0f}, "
                 f"n0 = {scales.reference_density.to_value(u.m**-3):.2g} m$^{{-3}}$)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    plot_style.apply(args.publication)

    spec, scales = load_spec_and_scales(args.config)
    if scales.flash is None:
        raise SystemExit(f"{args.config} has no flash: block — nothing to compare against")

    plot_paths = find_plotfiles(spec, args.config, args.diag_dir)[::max(args.stride, 1)]
    print(f"plotfiles : {len(plot_paths)} under "
          f"{os.path.dirname(plot_paths[0])}")

    per_dump = [measure_plotfile(path, scales,
                                 front_level_frac=args.front_level_frac)
                for path in plot_paths]
    reduced = load_reduced_diags(plot_paths)
    flash = load_flash_target(spec, args.flash_npz)
    if flash is None:
        print("NOTE: no flash_piston_profile.npz found — plotting WarpX only. Run "
              "scripts/flash_piston_profile.py to get the FLASH overlay.")

    rows = measure_run(per_dump, scales)
    lines = [
        metrics.scorecard_text(rows),
        "",
        "Per-plotfile WarpX measurement",
        f"  {'t/T_ci':>8} {'front [d_i]':>12} {'n_drive':>11} {'n_amb':>11} "
        f"{'contrast':>9} {'amb <u^2>':>11}",
    ]
    d_i_de = scales.di_over_de
    for dump in per_dump:
        lines.append(
            f"  {dump['t_gyro']:>8.4f} {dump['x_front_de'] / d_i_de:>12.3f} "
            f"{dump['n_drive']:>11.3e} {dump['n_ambient']:>11.3e} "
            f"{dump['contrast']:>9.3f} {dump['ambient_usq']:>11.3e}")
    text = "\n".join(lines)
    print()
    print(text)

    # results/warpx/<run>/, matching scripts/warpx_spitzer_resistivity.py. Passed as the
    # override because out_dir() keys its default on basename(), which would flatten
    # the warpx/ level away.
    run_name = os.path.basename(args.config).replace(".warpx.yaml", "")
    out_dir = yaml_edit.out_dir(
        run_name,
        args.output_dir or os.path.join(_REPO, "results", "warpx", run_name),
        cfg=spec, config_path=args.config)
    png_path = os.path.join(out_dir, "heater_vs_flash.png")
    plot(per_dump, reduced, flash, scales, rows, png_path)
    txt_path = os.path.join(out_dir, "heater_vs_flash.txt")
    with open(txt_path, "w") as handle:
        handle.write(text + "\n\n" + units.invariant_table(scales) + "\n")

    for path in (png_path, txt_path):
        print(f"Saved -> {path}")


if __name__ == "__main__":
    main()
