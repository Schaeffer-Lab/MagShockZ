# -*- coding: utf-8 -*-
"""scripts/run_flash_pressure_partition.py — momentum-flux (pressure) partition
for a MagShockZ FLASH run.

Energy density is not conserved across a shock, so this reports the **momentum
flux** (total pressure) — the quantity that is continuous across a steady front,
``[ρU² + p + B_t²/8π] = 0`` (the normal Rankine--Hugoniot jump condition).

Loads one representative FLASH dump, extracts a lineout along the same
line-of-sight used for the OSIRIS initialisation, and reports the partition of
the normal momentum flux (dyn/cm²) into four pressure channels:

    Ram pressure   ρ (v_LOS − v_shock)²
    Thermal e⁻     nₑ kTₑ                       (gas pressure, = P_xx; isotropic in MHD)
    Thermal i⁺     nᵢ kTᵢ                       (Al + Si)
    Magnetic       B_t²/8π                       (transverse field; B_para excluded)

plus the upstream-vs-downstream continuity check (total dn/up ≈ 1 if conserved)
and the compression vs. oblique Rankine--Hugoniot theory.  Every formula lives in
the pure, unit-tested module ``src/flash_energy_partition.py`` (``momentum_fluxes``,
``partition_by_region``, ``continuity_check``, ``compression_check``).

Usage
-----
    python scripts/run_flash_pressure_partition.py \\
        --config config/flash_3d_noshield.yaml \\
        [--snapshot-idx -1] \\
        [--x-shock-cm 0.185] \\
        [--x-downstream-start-cm 0.155] \\
        [--v-shock-cms 97000000] \\
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

import unyt as u

import analysis_utils
import plot_style
import flash_utils as fu
import flash_energy_partition as fep

# Momentum-flux (total-pressure) channels — the conserved RH quantity.
CHANNELS = ["p_ram", "p_th_e", "p_th_i", "p_mag"]
LABELS = {
    "p_ram":  "Ram pressure",
    "p_th_e": r"Thermal $e^-$",
    "p_th_i": r"Thermal $i^+$",
    "p_mag":  "Magnetic",
}
COLORS = {
    "p_ram":  "tab:blue",
    "p_th_e": "tab:orange",
    "p_th_i": "tab:red",
    "p_mag":  "tab:green",
}


def main():
    parser = argparse.ArgumentParser(
        description="FLASH shock momentum-flux (pressure) partition + continuity check."
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
                        help="Shock velocity [cm/s] for the ram pressure subtraction. "
                             "Default: read the fitted v_shock from the flash_overview "
                             ".npz (shock rest frame).  Pass 0 to force the lab frame.")
    parser.add_argument("--window-um", type=float, default=300.0, dest="window_um",
                        help="Half-width [µm] of the zoom window around the shock front "
                             "in the profile plot (default 300 µm).")
    parser.add_argument("--gamma", type=float, default=None,
                        help="Adiabatic index for the RH compression check (default: "
                             "config 'gamma' key, else 5/3). gamma=(f+2)/f: 5/3 (3 DOF), "
                             "2 (2 DOF), 3 (1 DOF) — sweep to read off the effective index.")
    parser.add_argument("--output-dir", default=None, dest="output_dir")
    plot_style.add_publication_arg(parser)
    args = parser.parse_args()
    plot_style.apply(args.publication)

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
    # flash_overview .npz (so the ram subtraction defaults to the shock rest frame).
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
                t_snap_s = (float(d["time_ns"][snap_idx_mod]) * u.ns).to("s").value
                x_shock_cm = (float(d["x_shock_0_cm"])
                              + float(d["v_shock_cms"]) * t_snap_s)
            if "v_shock_cms" in d.files:
                v_shock_npz = float(d["v_shock_cms"])
            if x_ds_start is None and np.isfinite(x_shock_cm):
                x_ds_start = x_shock_cm * 0.9

    if x_shock_cm is None or not np.isfinite(x_shock_cm):
        raise ValueError(
            "Shock position not available.  Run flash_overview.py first, or pass "
            "--x-shock-cm explicitly."
        )
    if x_ds_start is None:
        raise ValueError("Pass --x-downstream-start-cm explicitly.")

    # Frame for the ram term.  Explicit --v-shock-cms wins; else the fitted
    # v_shock from the overview (shock rest frame); else lab frame.
    if args.v_shock_cms is not None:
        v_shock_cms = args.v_shock_cms
    elif v_shock_npz is not None and np.isfinite(v_shock_npz):
        v_shock_cms = v_shock_npz
    else:
        v_shock_cms = 0.0

    print(f"Config         : {args.config}")
    print(f"FLASH dir      : {flash_dir}")
    print(f"Snapshot       : {os.path.basename(snap_file)}")
    print(f"x_shock        : {(x_shock_cm * u.cm).to('um').value:.2f} µm")
    print(f"x_downstream   : {(x_ds_start * u.cm).to('um').value:.2f} µm")
    print(f"v_shock        : {(v_shock_cms * u.cm / u.s).to('km/s').value:.2f} km/s")

    # ------------------------------------------------------------------
    # Lineout
    # ------------------------------------------------------------------
    print("\nLoading lineout …", flush=True)
    lo = fu.flash_lineout(snap_file, line_start, line_end)

    x_cm   = lo["x"].to("cm").value
    x_um   = lo["x"].to("um").value
    t_ns   = (lo["t_s"] * u.s).to("ns").value

    # ------------------------------------------------------------------
    # Momentum-flux (pressure) channels  (unyt arrays in dyn/cm²)
    # ------------------------------------------------------------------
    momflux = fep.momentum_fluxes(
        ne       = lo["ne"],
        Te       = lo["Te"],
        Ti       = lo["Ti"],
        n_ion    = lo["n_ion"],
        rho      = lo["rho"],
        v_para   = lo["v_para"],
        v_shock  = v_shock_cms,
        B_mag    = lo["B_mag"],
        B_para   = lo["B_para"],
    )
    result = fep.partition_by_region(momflux, x_cm, x_shock_cm, x_ds_start)
    labels = [LABELS[c] for c in CHANNELS]

    print("\n--- Momentum-flux (pressure) partition (CONSERVED: dn/up ≈ 1) ---")
    print(fep.partition_summary(result, channels=CHANNELS, labels=labels, unit="dyn/cm²"))
    cont = fep.continuity_check(result)
    print(fep.continuity_summary(cont))

    # ------------------------------------------------------------------
    # Compression vs Rankine--Hugoniot (oblique MHD theory, shared with OSIRIS)
    # ------------------------------------------------------------------
    gamma = args.gamma if args.gamma is not None else float(cfg.get("gamma", 5.0 / 3.0))
    up_mask = x_cm > x_shock_cm
    dn_mask = (x_cm >= x_ds_start) & (x_cm <= x_shock_cm)

    def _reg(field, units, mask):
        return float(np.nanmean(lo[field].to(units).value[mask]))

    def _prims(mask):
        return dict(
            rho=_reg("rho", "g/cm**3", mask), ne=_reg("ne", "cm**-3", mask),
            n_ion=_reg("n_ion", "cm**-3", mask), Te=_reg("Te", "eV", mask),
            Ti=_reg("Ti", "eV", mask), B_mag=_reg("B_mag", "gauss", mask),
            B_para=_reg("B_para", "gauss", mask),
        )

    v_para_up = _reg("v_para", "cm/s", up_mask)
    v_inflow = abs(v_shock_cms - v_para_up)        # shock-frame normal inflow [cm/s]
    check = fep.compression_check(_prims(up_mask), _prims(dn_mask), v_inflow, gamma=gamma)
    print("\n--- Compression vs Rankine--Hugoniot ---")
    print(fep.compression_summary(check))

    # ------------------------------------------------------------------
    # Figure — pressure profiles (stacked, shock rest frame) + continuity bars
    # ------------------------------------------------------------------
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(14, 5),
                                   gridspec_kw={"width_ratios": [2, 1]})

    # Panel A: stacked-area momentum flux vs distance from the front
    # (0 = shock, +ve upstream/ambient, −ve downstream/shocked).
    x_rel_um = ((x_cm - x_shock_cm) * u.cm).to("um").value
    order    = np.argsort(x_rel_um)
    x_sorted = x_rel_um[order]
    stack = np.zeros_like(x_sorted)
    for ch in CHANNELS:
        arr = momflux[ch][order].to("dyn/cm**2").value
        axA.fill_between(x_sorted, stack, stack + arr,
                         color=COLORS[ch], alpha=0.85, label=LABELS[ch])
        stack = stack + arr
    axA.axvline(0.0, color="k", ls="--", lw=1.4, label="shock front")
    axA.set_xlabel(r"distance from shock front [$\mu$m]   ($+$ upstream, $-$ downstream)")
    axA.set_ylabel(r"momentum flux [dyn cm$^{-2}$]")
    axA.set_title(
        f"Momentum-flux partition (shock rest frame, "
        f"$v_{{sh}}$ = {(v_shock_cms * u.cm / u.s).to('km/s').value:.0f} km/s)\n"
        f"{os.path.basename(snap_file)}  (t = {t_ns:.2f} ns)"
    )
    axA.legend(loc="upper right", fontsize=9)
    win = (x_sorted >= -args.window_um) & (x_sorted <= args.window_um)
    axA.set_xlim(-args.window_um, args.window_um)
    if win.any():
        axA.set_ylim(0, 1.05 * float(np.nanmax(stack[win])))
    axA.grid(alpha=0.25)

    # Panel B: continuity — up vs dn per channel + total (the conserved sum).
    bar_lbls = labels + ["Total"]
    up_vals = [result["upstream"]["means"][c] for c in CHANNELS] + [result["upstream"]["total"]]
    dn_vals = [result["downstream"]["means"][c] for c in CHANNELS] + [result["downstream"]["total"]]
    colors = [COLORS[c] for c in CHANNELS] + ["0.5"]
    xpos = np.arange(len(bar_lbls))
    w = 0.38
    axB.bar(xpos - w / 2, up_vals, w, color=colors, alpha=0.6,
            edgecolor="k", linewidth=0.8, label="Upstream")
    bars_dn = axB.bar(xpos + w / 2, dn_vals, w, color=colors, alpha=1.0,
                      edgecolor="k", linewidth=0.8, label="Downstream")
    for b in bars_dn:
        b.set_hatch("///")
    axB.axvline(len(CHANNELS) - 0.5, color="0.7", lw=1, ls=":")  # set off Total
    axB.annotate(f"dn/up = {cont['ratio']:.2f}  ({100 * cont['rel_imbalance']:+.0f}%)",
                 xy=(xpos[-1], max(up_vals[-1], dn_vals[-1])), xytext=(0, 4),
                 textcoords="offset points", ha="center", va="bottom", fontsize=10)
    axB.set_xticks(xpos)
    axB.set_xticklabels(bar_lbls, fontsize=9, rotation=20, ha="right")
    axB.set_ylabel(r"momentum flux [dyn cm$^{-2}$]")
    axB.set_title("Continuity (conserved if dn/up ≈ 1)")
    axB.legend(fontsize=9)
    axB.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig_path = os.path.join(out_dir, f"flash_pressure_partition_{os.path.basename(snap_file)}.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved → {fig_path}")

    # ------------------------------------------------------------------
    # Save .npz
    # ------------------------------------------------------------------
    npz_path = os.path.join(out_dir, f"flash_pressure_partition_{os.path.basename(snap_file)}.npz")
    np.savez(
        npz_path,
        x_um              = x_um,
        t_ns              = np.asarray(t_ns),
        **{c: momflux[c].to("dyn/cm**2").value for c in CHANNELS},
        x_shock_cm        = np.asarray(x_shock_cm),
        x_downstream_start_cm = np.asarray(x_ds_start),
        v_shock_cms       = np.asarray(v_shock_cms),
        up_total_dyn_cm2  = np.asarray(result["upstream"]["total"]),
        dn_total_dyn_cm2  = np.asarray(result["downstream"]["total"]),
        continuity_ratio  = np.asarray(cont["ratio"]),
        rel_imbalance     = np.asarray(cont["rel_imbalance"]),
        **{f"up_{c}_frac": np.asarray(result["upstream"]["fractions"][c]) for c in CHANNELS},
        **{f"dn_{c}_frac": np.asarray(result["downstream"]["fractions"][c]) for c in CHANNELS},
        # Compression vs Rankine--Hugoniot
        rh_gamma          = np.asarray(check["gamma"]),
        rh_theta_bn_rad   = np.asarray(check["theta_bn"]),
        rh_mach_s         = np.asarray(check["mach_s"]),
        rh_mach_a         = np.asarray(check["mach_a"]),
        r_measured        = np.asarray(check["r_measured"]),
        r_RH              = np.asarray(check["r_RH"]),
        b_t_measured      = np.asarray(check["b_t_measured"]),
        b_t_RH            = np.asarray(check["b_t_RH"]),
        config_path       = np.asarray(os.path.abspath(args.config)),
    )
    print(f"Saved → {npz_path}")


if __name__ == "__main__":
    main()
