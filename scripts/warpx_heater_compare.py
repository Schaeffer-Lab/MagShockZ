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
are (see ``src/heater_piston_scaling.py``).  Where a physical axis is wanted,
``ReducedScaling.to_ns`` / ``to_um`` bridge through the matched ion scales (d_i, T_ci);
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
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

import heater_deck
import heater_piston_scaling as hps
import heater_spec
import piston_profile as pp
import plot_style
import yaml_edit

_REPO = os.path.abspath(os.path.join(_HERE, ".."))

#: The deck species this script reads.  Checked against the generator's own list rather
#: than mirrored in a comment, so a rename there fails here loudly instead of leaving
#: the script asking a plotfile for a field that no longer exists.
PISTON_IONS = "piston_ions"
AMBIENT_IONS = "amb_ions"
AMBIENT_ELECTRONS = "amb_electrons"
assert {PISTON_IONS, AMBIENT_IONS, AMBIENT_ELECTRONS} <= set(heater_deck.SPECIES_NAMES)

#: Front travel below which no speed is reported.  One ion inertial length is the
#: smallest displacement over which "the piston is expanding" is a statement about
#: physics rather than about the profile's discretisation.
MIN_TRAVEL_DI = 1.0


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


def load_spec_and_scaling(config_path: str) -> tuple[dict, hps.ReducedScaling]:
    """Load the run spec and re-derive its scaling, so the bridge is never stale.

    Re-deriving rather than reading the frozen ``run.yaml`` means editing the spec and
    re-running this script cannot silently compare against the previous deck's scales.
    """
    spec = heater_spec.load(config_path)
    return spec, heater_spec.scaling(spec, smoke=False)


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


def transverse_average(dataset, field: str, scaling: hps.ReducedScaling
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
    return z_centres / scaling.d_e_m, profile


def measure_plotfile(path: str, scaling: hps.ReducedScaling, *,
                     front_level_frac: float) -> dict:
    """Piston front, drive density and ambient state from one WarpX plotfile.

    The slab is symmetric about z = 0 and expands both ways, so only the +z half is
    measured; ``piston_profile``'s estimators all assume the piston lies inward of the
    front, which is exactly the +z half's geometry.
    """
    import yt

    dataset = yt.load(path)
    time_omega_pe = float(dataset.current_time) * scaling.omega_pe_rad_s

    z_de, piston_charge = transverse_average(dataset, f"rho_{PISTON_IONS}", scaling)
    _, ambient_charge = transverse_average(dataset, f"rho_{AMBIENT_IONS}", scaling)
    _, ambient_usq = transverse_average(dataset, f"usq_{AMBIENT_ELECTRONS}", scaling)

    # rho is a charge density; the species carry +q_e, so n = rho/q_e.
    piston_density = piston_charge / hps.Q_E
    ambient_density = ambient_charge / hps.Q_E

    outward = z_de >= 0.0
    z_out = z_de[outward]
    piston_out = piston_density[outward]
    ambient_out = ambient_density[outward]

    level = front_level_frac * pp.ambient_reference_level(z_out, ambient_out)
    x_front = pp.front_position(z_out, piston_out, level=level if level > 0 else None)

    # Bands scaled to the run's own geometry, so they mean the same thing as the FLASH
    # side's (which used ~1 d_i inward and ~6 d_i outward, in its own d_i).
    d_i_de = scaling.d_i_m / scaling.d_e_m
    drive = pp.behind_front_average(z_out, piston_out, x_front,
                                    offset=0.2 * d_i_de, width=1.2 * d_i_de)
    ambient = pp.ahead_of_front_average(z_out, ambient_out, x_front,
                                        offset=6.0 * d_i_de, width=2.5 * d_i_de)

    return {
        "t_omega_pe": time_omega_pe,
        "t_gyro": time_omega_pe / (scaling.gyroperiod_s * scaling.omega_pe_rad_s),
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
        dataset = spec["flash_target"].get("dataset", "")
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


def scorecard(per_dump: list, targets: hps.PistonTargets,
              scaling: hps.ReducedScaling) -> list[tuple[str, float, float, float]]:
    """``(name, FLASH target, WarpX measured, deck intended)`` rows.

    Three columns, not two, because there are two distinct ways to be wrong: the deck
    can be built with the wrong constants (target vs intended), or the run can fail to
    realise the constants it was built with (intended vs measured).  Collapsing them
    would hide which.
    """
    late = [d for d in per_dump if np.isfinite(d["x_front_de"])]
    d_i_de = scaling.d_i_m / scaling.d_e_m
    measured_v_c = np.nan
    if len(late) >= 2:
        travel_di = abs(late[-1]["x_front_de"] - late[0]["x_front_de"]) / d_i_de
        if travel_di < MIN_TRAVEL_DI:
            # A front that has moved a fraction of an ion inertial length has not
            # started expanding yet, and fitting a slope to it yields a number that
            # looks like a speed but is line-out quantisation. Refuse it outright: a
            # smoke run reporting 'v = 0.0027 c' invites believing the deck is 20x slow.
            print(f"NOTE: the front has moved only {travel_di:.3f} d_i over "
                  f"{late[-1]['t_gyro'] - late[0]['t_gyro']:.4f} T_ci — too little to "
                  f"fit a speed (need {MIN_TRAVEL_DI} d_i). Speed/M_A left as nan; run "
                  f"the full deck, not the smoke one.")
        else:
            fit = pp.fit_front_trajectory(
                np.array([d["t_omega_pe"] for d in late]),
                np.array([d["x_front_de"] for d in late]))
            # d_e per 1/omega_pe is exactly c, so the slope is already v/c.
            measured_v_c = fit.speed

    measured_contrast = float(np.nanmean([d["contrast"] for d in late])) if late else np.nan

    # Front speed in d_i per gyroperiod. This is the MATCHED form of the speed: the
    # absolute v/c differs by ~20x between the codes on purpose (reduced mass ratio),
    # so quoting only v/c makes a correct mapping look broken.
    def to_di_per_gyro(speed_ms: float, d_i_m: float, gyroperiod_s: float) -> float:
        return speed_ms * gyroperiod_s / d_i_m

    return [
        ("front [d_i/T_ci]",
         to_di_per_gyro(targets.v_front_ms, targets.d_i_m, targets.gyroperiod_s),
         to_di_per_gyro(measured_v_c * hps.C_LIGHT_MS, scaling.d_i_m,
                        scaling.gyroperiod_s),
         to_di_per_gyro(scaling.v_piston_ms, scaling.d_i_m, scaling.gyroperiod_s)),
        ("M_A", targets.mach_alfven,
         measured_v_c / (scaling.v_alfven_ms / hps.C_LIGHT_MS), scaling.mach_alfven),
        ("n_piston / n_amb", targets.contrast, measured_contrast, scaling.contrast),
        ("v_piston / c  (NOT matched)", targets.v_front_ms / hps.C_LIGHT_MS,
         measured_v_c, scaling.v_piston_c),
    ]


def plot(per_dump: list, reduced: dict, flash: dict | None,
         targets: hps.PistonTargets, scaling: hps.ReducedScaling,
         rows: list, out_path: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    ax_profile, ax_trajectory, ax_operator, ax_score = axes.flat
    d_i_de = scaling.d_i_m / scaling.d_e_m

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
            xi = (flash_x - front) / (targets.d_i_m * 1e2)
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
        intended = (scaling.v_piston_ms * scaling.gyroperiod_s / scaling.d_i_m)
        ax_trajectory.plot(gyro, fronts_di[finite][0] + intended
                           * (gyro - gyro[finite][0]), "k--", lw=1.4,
                           label=f"deck target ({scaling.v_piston_c:.3f} c)")
    if flash is not None:
        flash_gyro = (flash["t_s"] - flash["t_s"][0]) / targets.gyroperiod_s
        flash_di = (flash["x_front_cm"] - flash["x_front_cm"][0]) / (
            targets.d_i_m * 1e2)
        ax_trajectory.plot(flash_gyro, flash_di + fronts_di[finite][0]
                           if np.any(finite) else flash_di, "-", color="0.35", lw=1.4,
                           label="FLASH front")
    ax_trajectory.set_xlabel(r"$t / T_{ci}$")
    ax_trajectory.set_ylabel(r"front position [$d_i$]")
    ax_trajectory.set_title("front trajectory in matched units")
    ax_trajectory.legend(fontsize=8)
    ax_trajectory.grid(alpha=0.25)

    if "EP_t" in reduced:
        ax_operator.plot(reduced["EP_t"] * scaling.omega_pe_rad_s
                         / (scaling.gyroperiod_s * scaling.omega_pe_rad_s),
                         reduced["EP_total"] / reduced["EP_total"][0]
                         if reduced["EP_total"][0] else reduced["EP_total"],
                         color="#ff7f0e", label="ParticleEnergy / initial")
    if "PN_t" in reduced:
        ax_operator.plot(reduced["PN_t"] * scaling.omega_pe_rad_s
                         / (scaling.gyroperiod_s * scaling.omega_pe_rad_s),
                         reduced["PN_total"] / reduced["PN_total"][0]
                         if reduced["PN_total"][0] else reduced["PN_total"],
                         color="#1f77b4", label="ParticleNumber / initial")
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
    header = f"{'quantity':<28}{'FLASH':>11}{'deck aim':>11}{'WarpX':>11}"
    text = [header, ""]
    for name, target, measured, intended in rows:
        text.append(f"{name:<28}{target:>11.4g}{intended:>11.4g}{measured:>11.4g}")
    text += [
        "",
        "FLASH -> deck aim is the mapping (src/heater_piston_scaling.py);",
        "deck aim -> WarpX is whether the run realised it.",
        "",
        f"theta_e = {scaling.theta_e_heater:.4g}   B0 = {scaling.b0_tesla * 1e4:.4g} G",
        f"n_target/n0 = {scaling.contrast:.4g}   slab = "
        f"{scaling.slab_halfwidth_di:.2f} d_i   r_H = {scaling.r_spot_di:.2f} d_i",
        "",
        "Tuning knobs, in the order worth trying:",
        "  front too slow/fast     -> scaling.theta_e_heater",
        "  contrast off            -> flash_target.n_piston_drive_per_m3",
        "  piston runs out of mass -> scaling.slab_halfwidth_di, deck.injector_tau_wpe",
        "  profile too narrow in x -> flash_target.r_spot_m",
    ]
    ax_score.text(0.0, 1.0, "\n".join(text), family="monospace", fontsize=9,
                  va="top", ha="left", transform=ax_score.transAxes)
    ax_score.set_title("invariant scorecard", loc="left")

    fig.suptitle("WarpX heater piston vs FLASH target "
                 f"(mass ratio {scaling.mass_ratio:.0f}, "
                 f"n0 = {scaling.n0_per_m3:.2g} m$^{{-3}}$)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    plot_style.apply(args.publication)

    spec, scaling = load_spec_and_scaling(args.config)
    targets = scaling.targets
    assert targets is not None

    plot_paths = find_plotfiles(spec, args.config, args.diag_dir)[::max(args.stride, 1)]
    print(f"plotfiles : {len(plot_paths)} under "
          f"{os.path.dirname(plot_paths[0])}")

    per_dump = [measure_plotfile(path, scaling,
                                 front_level_frac=args.front_level_frac)
                for path in plot_paths]
    reduced = load_reduced_diags(plot_paths)
    flash = load_flash_target(spec, args.flash_npz)
    if flash is None:
        print("NOTE: no flash_piston_profile.npz found — plotting WarpX only. Run "
              "scripts/flash_piston_profile.py to get the FLASH overlay.")

    rows = scorecard(per_dump, targets, scaling)
    lines = [
        f"{'quantity':<28}{'FLASH':>12}{'deck aim':>12}{'WarpX':>12}",
        *[f"{name:<28}{target:>12.4g}{intended:>12.4g}{measured:>12.4g}"
          for name, target, measured, intended in rows],
        "",
        "Per-plotfile WarpX measurement",
        f"  {'t/T_ci':>8} {'front [d_i]':>12} {'n_drive':>11} {'n_amb':>11} "
        f"{'contrast':>9} {'amb <u^2>':>11}",
    ]
    d_i_de = scaling.d_i_m / scaling.d_e_m
    for dump in per_dump:
        lines.append(
            f"  {dump['t_gyro']:>8.4f} {dump['x_front_de'] / d_i_de:>12.3f} "
            f"{dump['n_drive']:>11.3e} {dump['n_ambient']:>11.3e} "
            f"{dump['contrast']:>9.3f} {dump['ambient_usq']:>11.3e}")
    text = "\n".join(lines)
    print()
    print(text)

    # results/warpx/<run>/, matching scripts/spitzer_resistivity.py. Passed as the
    # override because out_dir() keys its default on basename(), which would flatten
    # the warpx/ level away.
    run_name = os.path.basename(args.config).replace(".warpx.yaml", "")
    out_dir = yaml_edit.out_dir(
        run_name,
        args.output_dir or os.path.join(_REPO, "results", "warpx", run_name),
        cfg=spec, config_path=args.config)
    png_path = os.path.join(out_dir, "heater_vs_flash.png")
    plot(per_dump, reduced, flash, targets, scaling, rows, png_path)
    txt_path = os.path.join(out_dir, "heater_vs_flash.txt")
    with open(txt_path, "w") as handle:
        handle.write(text + "\n\n" + hps.invariance_report(scaling) + "\n")

    for path in (png_path, txt_path):
        print(f"Saved -> {path}")


if __name__ == "__main__":
    main()
