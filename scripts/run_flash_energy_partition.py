# -*- coding: utf-8 -*-
"""scripts/flash_energy_partition.py — energy partition between electrons, ions,
bulk flow, and magnetic field for a MagShockZ FLASH run.

Loads one representative FLASH dump, extracts a lineout along the same
line-of-sight used for the OSIRIS initialisation, and reports the partition
of total energy density (erg/cm³) into four channels:

    Kinetic (ram)   ½ ρ (v_LOS − v_shock)²
    Thermal e⁻      (3/2) nₑ kB Tₑ
    Thermal i⁺      (3/2) nᵢ kB Tᵢ   (Al + Si)
    Magnetic        B²/(8π)

Unlike the OSIRIS version there is no parallel/perpendicular split.

Usage
-----
    python scripts/flash_energy_partition.py \\
        --config config/perlmutter_1.3.1d.yaml \\
        [--snapshot-idx -1] \\
        [--x-shock-cm 0.045] \\
        [--x-downstream-start-cm 0.02] \\
        [--output-dir results/FLASH_3D_noshield]
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
sys.path.insert(0, os.path.join(_HERE, "..", "init_nopython"))

import analysis_utils
import flash_utils as fu
import flash_energy_partition as fep

_CM_TO_UM  = 1e4
_S_TO_NS   = 1e9
_CM_TO_KMS = 1e-5

CHANNEL_LABELS = {
    "u_kinetic": "Kinetic (ram)",
    "u_th_e":    r"Thermal $e^-$",
    "u_th_i":    r"Thermal $i^+$",
    "u_mag":     "Magnetic",
}
CHANNEL_COLORS = {
    "u_kinetic": "tab:blue",
    "u_th_e":    "tab:orange",
    "u_th_i":    "tab:red",
    "u_mag":     "tab:green",
}


def main():
    parser = argparse.ArgumentParser(
        description="FLASH shock energy partition — electron vs ion vs kinetic vs magnetic."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--snapshot-idx", type=int, default=-1, dest="snapshot_idx",
                        help="Index into the sorted plot-file list (default -1 = last dump).")
    parser.add_argument("--x-shock-cm", type=float, default=None, dest="x_shock_cm",
                        help="Shock position along LOS [cm].  Required unless a "
                             "flash_overview .npz is found for the same run.")
    parser.add_argument("--x-downstream-start-cm", type=float, default=None,
                        dest="x_downstream_start_cm",
                        help="Left edge of downstream region [cm].")
    parser.add_argument("--v-shock-cms", type=float, default=None, dest="v_shock_cms",
                        help="Shock velocity [cm/s] for the kinetic ram subtraction. "
                             "Default: read the fitted v_shock from the flash_overview "
                             ".npz (shock rest frame).  Pass 0 to force the lab frame.")
    parser.add_argument("--window-um", type=float, default=300.0, dest="window_um",
                        help="Half-width [µm] of the zoom window around the shock front "
                             "in the area plot (default 300 µm).")
    parser.add_argument("--output-dir", default=None, dest="output_dir")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Config + run parameters
    # ------------------------------------------------------------------
    cfg   = analysis_utils.load_config(args.config)
    spec  = analysis_utils.RunSpec.from_sim_dir(cfg["sim_dir"])

    data_path   = spec["data_path"]
    flash_dir   = str(os.path.dirname(data_path))
    line_start  = tuple(float(v) for v in spec["start_point"])
    line_end    = tuple(float(v) for v in spec["end_point"])

    all_files   = fu.find_plot_files(flash_dir)
    snap_file   = all_files[args.snapshot_idx % len(all_files)]

    out_dir = args.output_dir or os.path.join(
        _HERE, "..", "results", os.path.basename(flash_dir.rstrip("/"))
    )
    os.makedirs(out_dir, exist_ok=True)

    # Optionally load shock position AND fitted shock velocity from a
    # flash_overview .npz (so the area plot defaults to the shock rest frame).
    x_shock_cm  = args.x_shock_cm
    x_ds_start  = args.x_downstream_start_cm
    v_shock_npz = None
    if x_shock_cm is None:
        npz_files = sorted(
            f for f in os.listdir(out_dir)
            if f.startswith("flash_overview_") and f.endswith(".npz")
        )
        if npz_files:
            d = np.load(os.path.join(out_dir, npz_files[-1]), allow_pickle=True)
            snap_idx_mod = args.snapshot_idx % len(d["time_ns"])
            if "x_shock_det_cm" in d.files:
                x_shock_cm = float(d["x_shock_det_cm"][snap_idx_mod])
            if np.isnan(x_shock_cm) and "x_shock_0_cm" in d.files:
                # Fit stored as x_shock(t) = x_shock_0_cm + v_shock_cms * t[s].
                # time_ns is in ns, so convert to seconds once (÷ _S_TO_NS).
                t_snap_s = float(d["time_ns"][snap_idx_mod]) / _S_TO_NS
                x_shock_cm = (float(d["x_shock_0_cm"])
                              + float(d["v_shock_cms"]) * t_snap_s)
            if "v_shock_cms" in d.files:
                v_shock_npz = float(d["v_shock_cms"])
            if x_ds_start is None and np.isfinite(x_shock_cm):
                # Default: 30% of shock position upstream as a reasonable downstream window
                x_ds_start = x_shock_cm * 0.9

    if x_shock_cm is None or not np.isfinite(x_shock_cm):
        raise ValueError(
            "Shock position not available.  Run flash_overview.py first, or pass "
            "--x-shock-cm explicitly."
        )
    if x_ds_start is None:
        raise ValueError("Pass --x-downstream-start-cm explicitly.")

    # Frame for the kinetic ram term.  Explicit --v-shock-cms wins; otherwise
    # use the fitted v_shock from the overview (shock rest frame); else lab frame.
    if args.v_shock_cms is not None:
        v_shock_cms = args.v_shock_cms
    elif v_shock_npz is not None and np.isfinite(v_shock_npz):
        v_shock_cms = v_shock_npz
    else:
        v_shock_cms = 0.0

    print(f"Config         : {args.config}")
    print(f"FLASH dir      : {flash_dir}")
    print(f"Snapshot       : {os.path.basename(snap_file)}")
    print(f"x_shock        : {x_shock_cm * _CM_TO_UM:.2f} µm")
    print(f"x_downstream   : {x_ds_start * _CM_TO_UM:.2f} µm")
    print(f"v_shock        : {v_shock_cms * _CM_TO_KMS:.2f} km/s")

    # ------------------------------------------------------------------
    # Lineout
    # ------------------------------------------------------------------
    print("\nLoading lineout …", flush=True)
    lo = fu.flash_lineout(snap_file, line_start, line_end)

    # lo holds unyt arrays; strip the coordinate to plain cm for masks/positions.
    x_cm   = lo["x"].to("cm").value
    x_um   = x_cm * _CM_TO_UM
    t_ns   = lo["t_s"] * _S_TO_NS

    # ------------------------------------------------------------------
    # Energy densities  (unyt arrays in erg/cm³)
    # ------------------------------------------------------------------
    energy = fep.energy_densities(
        ne       = lo["ne"],
        Te       = lo["Te"],
        Ti       = lo["Ti"],
        n_ion    = lo["n_ion"],
        rho      = lo["rho"],
        v_para   = lo["v_para"],
        v_shock  = v_shock_cms,
        B_mag    = lo["B_mag"],
    )

    result = fep.partition_by_region(energy, x_cm, x_shock_cm, x_ds_start)

    print("\n--- Energy partition ---")
    print(fep.partition_summary(result))

    # ------------------------------------------------------------------
    # Figure 1 — stacked area chart, shock rest frame, centred on the front
    # ------------------------------------------------------------------
    # x-axis is distance from the shock front: 0 = shock, +ve = upstream
    # (ambient ahead), −ve = downstream (shocked side behind).  The kinetic
    # channel is already in the shock frame because energy_densities() was
    # called with v_shock=v_shock_cms, so this view shows the energy budget
    # immediately on either side of the front without the lab-frame bulk-flow
    # offset swamping everything.
    channels = ["u_kinetic", "u_th_e", "u_th_i", "u_mag"]

    x_rel_um = (x_cm - x_shock_cm) * _CM_TO_UM   # distance from shock [µm]
    order    = np.argsort(x_rel_um)              # fill_between needs sorted x
    x_sorted = x_rel_um[order]

    fig1, ax1 = plt.subplots(figsize=(11, 5))
    # Stack from bottom; each fill starts where the previous ended.
    stack = np.zeros_like(x_sorted)
    for ch in channels:
        arr = energy[ch][order].to("erg/cm**3").value   # plain CGS for plotting
        ax1.fill_between(x_sorted, stack, stack + arr,
                         color=CHANNEL_COLORS[ch], alpha=0.85,
                         label=CHANNEL_LABELS[ch])
        stack = stack + arr

    ax1.axvline(0.0, color="k", ls="--", lw=1.4, label="shock front")
    ax1.set_xlabel(r"distance from shock front [$\mu$m]   ($+$ upstream, $-$ downstream)")
    ax1.set_ylabel(r"energy density [erg cm$^{-3}$]")
    ax1.set_title(
        f"Energy partition (shock rest frame, $v_{{sh}}$ = {v_shock_cms * _CM_TO_KMS:.0f} km/s) "
        f"— {os.path.basename(snap_file)}  (t = {t_ns:.2f} ns)"
    )
    ax1.legend(loc="upper right", fontsize=9)

    # Zoom to ±window_um around the front; set the y-limit from what is visible
    # in that window so the immediate jump fills the axes.
    win = (x_sorted >= -args.window_um) & (x_sorted <= args.window_um)
    ax1.set_xlim(-args.window_um, args.window_um)
    if win.any():
        ax1.set_ylim(0, 1.05 * float(np.nanmax(stack[win])))
    ax1.grid(alpha=0.25)

    fig1.tight_layout()
    area_path = os.path.join(out_dir, f"flash_energy_area_{os.path.basename(snap_file)}.png")
    fig1.savefig(area_path, dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"\nSaved → {area_path}")

    # ------------------------------------------------------------------
    # Figure 2 — bar chart: upstream vs downstream fractions
    # ------------------------------------------------------------------
    fig2, (ax2, ax2b) = plt.subplots(
        1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [3, 1]}
    )

    n_ch   = len(channels)
    width  = 0.35
    x_pos  = np.arange(n_ch)

    up_fracs = [result["upstream"]["fractions"][ch]   * 100 for ch in channels]
    dn_fracs = [result["downstream"]["fractions"][ch] * 100 for ch in channels]

    bars_up = ax2.bar(x_pos - width / 2, up_fracs, width,
                      label="Upstream", color=[CHANNEL_COLORS[c] for c in channels],
                      alpha=0.6, edgecolor="k", linewidth=0.8)
    bars_dn = ax2.bar(x_pos + width / 2, dn_fracs, width,
                      label="Downstream", color=[CHANNEL_COLORS[c] for c in channels],
                      alpha=1.0, edgecolor="k", linewidth=0.8)

    # Hatch downstream bars to distinguish from upstream
    for bar in bars_dn:
        bar.set_hatch("///")

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([CHANNEL_LABELS[c] for c in channels], fontsize=10)
    ax2.set_ylabel("Fraction of total energy [%]")
    ax2.set_title("Energy partition by channel")
    ax2.legend(fontsize=9)
    ax2.set_ylim(0, 100)
    ax2.grid(axis="y", alpha=0.3)

    # Companion panel: total energy density per region (erg/cm³, log scale).
    # Fractions above are dimensionless; this shows the absolute energy each
    # region carries, which differs by orders of magnitude across the shock.
    totals = [result["upstream"]["total"], result["downstream"]["total"]]
    region_bars = ax2b.bar(
        ["Upstream", "Downstream"], totals,
        color=["0.65", "0.4"], edgecolor="k", linewidth=0.8,
    )
    region_bars[1].set_hatch("///")
    ax2b.set_yscale("log")
    ax2b.set_ylabel(r"total energy density [erg cm$^{-3}$]")
    ax2b.set_title("Total energy")
    ax2b.grid(axis="y", alpha=0.3, which="both")
    for bar, tot in zip(region_bars, totals):
        ax2b.annotate(f"{tot:.2e}", (bar.get_x() + bar.get_width() / 2,
                                     bar.get_height()),
                      ha="center", va="bottom", fontsize=8)
    # Headroom so the annotations are not clipped at the top (log axis).
    ax2b.set_ylim(top=max(totals) * 3)

    fig2.suptitle(
        f"Energy partition — upstream vs downstream\n"
        f"{os.path.basename(snap_file)}  (t = {t_ns:.2f} ns)  "
        f"v_shock = {v_shock_cms * _CM_TO_KMS:.1f} km/s",
        fontsize=12,
    )
    fig2.tight_layout()
    bar_path = os.path.join(out_dir, f"flash_energy_partition_{os.path.basename(snap_file)}.png")
    fig2.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved → {bar_path}")

    # ------------------------------------------------------------------
    # Save .npz
    # ------------------------------------------------------------------
    npz_path = os.path.join(out_dir, f"flash_energy_partition_{os.path.basename(snap_file)}.npz")
    np.savez(
        npz_path,
        x_um              = x_um,
        t_ns              = np.asarray(t_ns),
        **{k: v.to("erg/cm**3").value for k, v in energy.items()},
        x_shock_cm        = np.asarray(x_shock_cm),
        x_downstream_start_cm = np.asarray(x_ds_start),
        v_shock_cms       = np.asarray(v_shock_cms),
        # Upstream
        up_u_kinetic_frac = np.asarray(result["upstream"]["fractions"]["u_kinetic"]),
        up_u_th_e_frac    = np.asarray(result["upstream"]["fractions"]["u_th_e"]),
        up_u_th_i_frac    = np.asarray(result["upstream"]["fractions"]["u_th_i"]),
        up_u_mag_frac     = np.asarray(result["upstream"]["fractions"]["u_mag"]),
        up_total_erg_cm3  = np.asarray(result["upstream"]["total"]),
        # Downstream
        dn_u_kinetic_frac = np.asarray(result["downstream"]["fractions"]["u_kinetic"]),
        dn_u_th_e_frac    = np.asarray(result["downstream"]["fractions"]["u_th_e"]),
        dn_u_th_i_frac    = np.asarray(result["downstream"]["fractions"]["u_th_i"]),
        dn_u_mag_frac     = np.asarray(result["downstream"]["fractions"]["u_mag"]),
        dn_total_erg_cm3  = np.asarray(result["downstream"]["total"]),
        config_path       = np.asarray(os.path.abspath(args.config)),
    )
    print(f"Saved → {npz_path}")


if __name__ == "__main__":
    main()
