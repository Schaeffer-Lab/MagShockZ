# -*- coding: utf-8 -*-
"""scripts/flash_rh_prediction.py — predict the downstream FLASH state from the
measured UPSTREAM state + shock speed, using the perpendicular (theta = 90 deg)
MHD shock theory in src/perpendicular_shock.py, and overlay the prediction on
lineouts across the shock front.

What it does
------------
1. Loads one FLASH dump's lineout along the OSIRIS line-of-sight (unyt arrays in
   physical CGS; units travel with the data and conversions are done with .to()).
2. Reads the shock position and fitted shock speed v_shock from the
   flash_overview .npz (run flash_overview.py first), splitting the lineout into
   an upstream and a downstream region.
3. Averages the UPSTREAM region to a single state (rho, n_e, n_ion, T_e, T_i,
   B_perp, v_para) and hands it to perpendicular_shock.solve_from_upstream,
   which forms the sound speed (the two-temperature / ion-acoustic form,
   c_s = sqrt((gamma_e P_e + gamma_i P_i)/rho), defaulting gamma_e=gamma_i=gamma
   so it matches the single-fluid jump), the Alfven speed, and the shock-frame
   inflow, then solves the perpendicular MHD shock for r, p2/p1, T2/T1.  No
   speeds or Mach numbers are assembled in this script.
4. Predicts the downstream value of every quantity (rho/n_e scale by r, the
   transverse field by r, total thermal pressure by p2/p1, the temperatures by
   T2/T1, the shock-frame inflow speed by 1/r).
5. Plots each quantity's lineout across the front with three reference lines:
   the upstream mean, the THEORY-predicted downstream value, and the measured
   downstream mean — so measured-vs-predicted is read off directly.

Fields are read DIRECTLY from FLASH: n_e, n_ion, T_e, T_i, |B|, B_para, v_para,
rho.  The ONE exception is the thermal pressure: this 3T FLASH dataset does not
store pressure on disk (the EOS makes it from the temperatures), so we form the
ideal-gas pressure P = n_e kT_e + n_ion kT_i.

This is the simplest (pure-MHD, isotropic-pressure) baseline.  Single-fluid MHD
predicts ONE temperature jump, applied to T_e and T_i alike; it does NOT predict
the electron/ion split — that, and any departure of the data from these lines,
is the kinetic/collisionless physics MHD omits.

Usage
-----
    python scripts/flash_rh_prediction.py \\
        --config config/flash_3d_noshield.yaml \\
        [--snapshot-idx -1] [--gamma 1.6667] \\
        [--x-shock-cm ...] [--x-downstream-start-cm ...] [--v-shock-cms ...] \\
        [--window-um 400] [--output-dir results/FLASH_3D_noshield]

Run in the `analysis` conda env (it has yt + unyt + osiris_utils).
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import unyt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
sys.path.insert(0, os.path.join(_HERE, "..", "init_nopython"))

import analysis_utils
import plot_style
import flash_utils as fu
import perpendicular_shock as ps


def _load_shock_geometry(cfg, idx, out_dir, snapshot_idx, x_shock_cm, x_ds_start):
    """Resolve (x_shock [cm], x_downstream_start [cm], v_shock [cm/s]) as bare
    floats.  Priority (first hit wins): explicit CLI value → the hand-placed
    ``flash_dump_params.<idx>`` in the config (written by tune_flash_shock.py) →
    the flash_overview .npz.  The caller attaches units.

    ``idx`` is the resolved positive plot-file index (the config key).  ``out_dir``
    is searched for ``flash_overview_*.npz`` only for whatever remains unset.
    """
    # 1. hand-placed per-dump positions in the config (cm).
    per = cfg.get("flash_dump_params", {}).get(idx, {})
    if x_shock_cm is None and "x_shock_cm" in per:
        x_shock_cm = float(per["x_shock_cm"])
    if x_ds_start is None and "x_downstream_start_cm" in per:
        x_ds_start = float(per["x_downstream_start_cm"])

    # 2. flash_overview .npz fills in whatever is still missing (shock position,
    #    downstream edge, and the fitted v_shock — always read so the caller can
    #    fall back to it for the shock speed).
    v_shock_npz = None
    npz_files = sorted(
        f for f in os.listdir(out_dir)
        if f.startswith("flash_overview_") and f.endswith(".npz")
    )
    if npz_files:
        d = np.load(os.path.join(out_dir, npz_files[-1]), allow_pickle=True)
        npz_idx = snapshot_idx % len(d["time_ns"])
        if "v_shock_cms" in d.files:
            v_shock_npz = float(d["v_shock_cms"])
        if x_shock_cm is None:
            if "x_shock_det_cm" in d.files:
                x_shock_cm = float(d["x_shock_det_cm"][npz_idx])
            if (x_shock_cm is None or np.isnan(x_shock_cm)) and "x_shock_0_cm" in d.files:
                t_snap_s = float(d["time_ns"][npz_idx]) * 1e-9      # the npz stores ns
                x_shock_cm = float(d["x_shock_0_cm"]) + float(d["v_shock_cms"]) * t_snap_s
    if x_ds_start is None and x_shock_cm is not None and np.isfinite(x_shock_cm):
        x_ds_start = x_shock_cm * 0.9
    return x_shock_cm, x_ds_start, v_shock_npz


def main():
    parser = argparse.ArgumentParser(
        description="Predict the downstream FLASH state from upstream + v_shock "
                    "(perpendicular MHD shock) and overlay it on lineouts."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--snapshot-idx", type=int, default=-1, dest="snapshot_idx",
                        help="Index into the sorted plot-file list (default -1 = last dump).")
    parser.add_argument("--x-shock-cm", type=float, default=None, dest="x_shock_cm",
                        help="Shock position along LOS [cm]. Default: read from the "
                             "flash_overview .npz.")
    parser.add_argument("--x-downstream-start-cm", type=float, default=None,
                        dest="x_downstream_start_cm",
                        help="Left edge of the downstream region [cm].")
    parser.add_argument("--v-shock-cms", type=float, default=None, dest="v_shock_cms",
                        help="Shock velocity [cm/s]. Default: fitted v_shock from the "
                             "flash_overview .npz.")
    parser.add_argument("--gamma", type=float, default=None,
                        help="Adiabatic index (default: config 'gamma' key, else 5/3). "
                             "gamma=(f+2)/f: 5/3 (3 DOF), 2 (2 DOF), 3 (1 DOF) — sweep "
                             "to read off the effective index.")
    parser.add_argument("--window-um", type=float, default=400.0, dest="window_um",
                        help="Half-width [µm] of the zoom window around the front (default 400).")
    parser.add_argument("--output-dir", default=None, dest="output_dir")
    plot_style.add_publication_arg(parser)
    args = parser.parse_args()
    plot_style.apply(args.publication)

    # ------------------------------------------------------------------
    # Config + run parameters
    # ------------------------------------------------------------------
    cfg  = analysis_utils.load_config(args.config)
    spec = analysis_utils.RunSpec.from_sim_dir(cfg["sim_dir"])

    data_path  = spec["data_path"]
    flash_dir  = str(os.path.dirname(data_path))
    line_start = tuple(float(v) for v in spec["start_point"])
    line_end   = tuple(float(v) for v in spec["end_point"])

    all_files = fu.find_plot_files(flash_dir)
    idx       = args.snapshot_idx % len(all_files)   # config key = positive plot-file index
    snap_file = all_files[idx]

    out_dir = args.output_dir or os.path.join(
        _HERE, "..", "results", os.path.basename(flash_dir.rstrip("/"))
    )
    os.makedirs(out_dir, exist_ok=True)

    x_shock_cm, x_ds_cm, v_shock_npz = _load_shock_geometry(
        cfg, idx, out_dir, args.snapshot_idx, args.x_shock_cm, args.x_downstream_start_cm)

    if x_shock_cm is None or not np.isfinite(x_shock_cm):
        raise ValueError(
            "Shock position not available. Place it with tune_flash_shock.py "
            "(--mode regions), run flash_overview.py, or pass --x-shock-cm.")
    if x_ds_cm is None:
        raise ValueError(
            "Downstream-region edge not available. Place it with tune_flash_shock.py "
            "or pass --x-downstream-start-cm.")

    # v_shock: CLI → config flash.v_shock_est_cms → fitted value from the npz.
    if args.v_shock_cms is not None:
        v_shock_cms = args.v_shock_cms
    elif cfg.get("flash", {}).get("v_shock_est_cms"):
        v_shock_cms = float(cfg["flash"]["v_shock_est_cms"])
    elif v_shock_npz is not None and np.isfinite(v_shock_npz):
        v_shock_cms = v_shock_npz
    else:
        v_shock_cms = 0.0

    gamma = args.gamma if args.gamma is not None else float(cfg.get("gamma", 5.0 / 3.0))

    # Attach units once; let unyt carry them from here on.
    x_shock = x_shock_cm * unyt.cm
    x_ds    = x_ds_cm    * unyt.cm
    v_shock = v_shock_cms * unyt.cm / unyt.s

    print(f"Config       : {args.config}")
    print(f"FLASH dir    : {flash_dir}")
    print(f"Snapshot     : {os.path.basename(snap_file)}")
    print(f"x_shock      : {x_shock.to('um'):.2f}")
    print(f"x_downstream : {x_ds.to('um'):.2f}")
    print(f"v_shock      : {v_shock.to('km/s'):.2f}")
    print(f"gamma        : {gamma:.4f}")

    # ------------------------------------------------------------------
    # Lineout + derived per-point quantities (all unyt)
    # ------------------------------------------------------------------
    print("\nLoading lineout …", flush=True)
    lo = fu.flash_lineout(snap_file, line_start, line_end)

    x = lo["x"].to("cm")
    t = (lo["t_s"] * unyt.s)

    # Transverse (shock-tangential) field: the perpendicular shock compresses
    # this with the density.  B_perp = sqrt(|B|^2 - B_para^2); the abs only
    # guards floating-point roundoff (|B_para| <= |B| exactly by construction).
    B_perp = np.sqrt(np.abs(lo["B_mag"]**2 - lo["B_para"]**2)).to("gauss")
    P_th   = (lo["ne"] * lo["Te"] + lo["n_ion"] * lo["Ti"]).to("erg/cm**3")
    v_sf   = np.abs(lo["v_para"] - v_shock).to("cm/s")   # shock-frame normal speed

    # ------------------------------------------------------------------
    # Region averages (upstream = ambient ahead, x > x_shock)
    # ------------------------------------------------------------------
    up = x > x_shock
    dn = (x >= x_ds) & (x <= x_shock)
    if not up.any() or not dn.any():
        raise ValueError("Empty upstream or downstream region — check the bounds.")

    def mean_up(arr):
        return np.nanmean(arr[up])

    def mean_dn(arr):
        return np.nanmean(arr[dn])

    rho_up   = mean_up(lo["rho"])
    ne_up    = mean_up(lo["ne"])
    ni_up    = mean_up(lo["n_ion"])
    Te_up    = mean_up(lo["Te"])
    Ti_up    = mean_up(lo["Ti"])
    Bperp_up = mean_up(B_perp)
    Bpara_up = mean_up(lo["B_para"])
    vpara_up = mean_up(lo["v_para"])
    # Upstream thermal pressure from the region-averaged partial pressures, so it
    # is consistent with the sound speed perpendicular_shock builds internally.
    P_up     = (ne_up * Te_up + ni_up * Ti_up).to("erg/cm**3")

    # ------------------------------------------------------------------
    # Perpendicular-shock solution — hand the upstream FLASH fields straight to
    # perpendicular_shock; it forms c_s (two-temperature), v_A and v_inflow and
    # solves the jump.  No speeds or Mach numbers are assembled here.
    # ------------------------------------------------------------------
    jump = ps.solve_from_upstream(
        ne=ne_up, Te=Te_up, n_ion=ni_up, Ti=Ti_up,
        B_perp=Bperp_up, B_para=Bpara_up, rho=rho_up,
        v_shock=v_shock, v_para=vpara_up, gamma=gamma)
    v_A      = jump["v_A"].to("cm/s")
    c_s      = jump["c_s"].to("cm/s")
    v_inflow = jump["v_inflow"].to("cm/s")
    theta_bn = jump["theta_bn"]
    r        = jump["r"]
    p_ratio  = jump["p_ratio"]
    T_ratio  = jump["T_ratio"]
    mach_s   = jump["mach_s"]
    mach_a   = jump["mach_a"]

    if not jump["exists"] or not np.isfinite(r):
        print("\n!! No compressive perpendicular shock for these upstream numbers "
              f"(M_s={mach_s:.2f}, M_A={mach_a:.2f}). Predictions will be NaN.")

    # Predicted downstream values — apply the jump ratios to the unyt upstream
    # state, so the predictions keep their units too.
    pred = ps.predict_downstream(
        jump, rho1=ne_up, B_perp1=Bperp_up, p1=P_up, v_inflow=v_inflow)
    ne_dn_pred    = pred["rho"]          # n scales like rho
    Bperp_dn_pred = pred["B_perp"]
    P_dn_pred     = pred["p"]
    vsf_dn_pred   = pred["v_inflow"]
    Te_dn_pred    = T_ratio * Te_up
    Ti_dn_pred    = T_ratio * Ti_up

    # Measured downstream means for the head-to-head.
    ne_dn_meas    = mean_dn(lo["ne"])
    Bperp_dn_meas = mean_dn(B_perp)
    P_dn_meas     = mean_dn(P_th)
    vsf_dn_meas   = mean_dn(v_sf)
    Te_dn_meas    = mean_dn(lo["Te"])
    Ti_dn_meas    = mean_dn(lo["Ti"])

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    print("\n--- Upstream state (region average) ---")
    print(f"  n_e = {ne_up.to('cm**-3'):.3e}   B_perp = {Bperp_up.to('gauss'):.2f}   "
          f"T_e = {Te_up.to('eV'):.1f}   T_i = {Ti_up.to('eV'):.1f}")
    print(f"  v_A = {v_A.to('km/s'):.1f}   c_s = {c_s.to('km/s'):.1f}   "
          f"v_inflow = {v_inflow.to('km/s'):.1f}")
    print(f"  theta_Bn = {np.degrees(theta_bn):.1f} deg   "
          f"M_s = {mach_s:.2f}   M_A = {mach_a:.2f}")
    print(f"\n--- Perpendicular MHD prediction (gamma = {gamma:.4f}) ---")
    print(f"  r = rho2/rho1 = {r:.3f}   p2/p1 = {p_ratio:.3f}   T2/T1 = {T_ratio:.3f}")
    sep = "-" * 64
    print(sep)
    print(f"  {'quantity':<22}{'upstream':>12}{'pred. dn':>12}{'meas. dn':>12}")
    print(sep)
    rows = [
        ("n_e [cm^-3]",        ne_up.to("cm**-3"),    ne_dn_pred.to("cm**-3"),    ne_dn_meas.to("cm**-3")),
        ("B_perp [G]",         Bperp_up.to("gauss"),  Bperp_dn_pred.to("gauss"),  Bperp_dn_meas.to("gauss")),
        ("T_e [eV]",           Te_up.to("eV"),        Te_dn_pred.to("eV"),        Te_dn_meas.to("eV")),
        ("T_i [eV]",           Ti_up.to("eV"),        Ti_dn_pred.to("eV"),        Ti_dn_meas.to("eV")),
        ("P_thermal [erg/cc]", P_up.to("erg/cm**3"),  P_dn_pred.to("erg/cm**3"),  P_dn_meas.to("erg/cm**3")),
        ("v_shockframe [km/s]",v_inflow.to("km/s"),   vsf_dn_pred.to("km/s"),     vsf_dn_meas.to("km/s")),
    ]
    for name, u, pdn, mdn in rows:
        print(f"  {name:<22}{float(u):>12.3e}{float(pdn):>12.3e}{float(mdn):>12.3e}")
    print(sep)

    # ------------------------------------------------------------------
    # Figure — lineouts across the front with the predicted downstream line
    # ------------------------------------------------------------------
    # Each panel: (display profile, upstream mean, predicted dn, measured dn,
    #              y-label, colour, log-y?).  Profiles/lines are converted to the
    #              panel's display unit with .to(...).value at draw time.
    panels = [
        (lo["ne"].to("cm**-3").value,  ne_up.to("cm**-3"),  ne_dn_pred.to("cm**-3"),  ne_dn_meas.to("cm**-3"),
         r"$n_e$ [cm$^{-3}$]",        "tab:purple", True),
        (B_perp.to("gauss").value,     Bperp_up.to("gauss"), Bperp_dn_pred.to("gauss"), Bperp_dn_meas.to("gauss"),
         r"$B_\perp$ [G]",            "tab:orange", False),
        (v_sf.to("km/s").value,        v_inflow.to("km/s"), vsf_dn_pred.to("km/s"),   vsf_dn_meas.to("km/s"),
         r"$|v - v_{\rm sh}|$ [km/s]", "tab:blue", False),
        (lo["Te"].to("eV").value,      Te_up.to("eV"),      Te_dn_pred.to("eV"),      Te_dn_meas.to("eV"),
         r"$T_e$ [eV]",               "tab:green",  True),
        (lo["Ti"].to("eV").value,      Ti_up.to("eV"),      Ti_dn_pred.to("eV"),      Ti_dn_meas.to("eV"),
         r"$T_i$ [eV]",               "tab:brown",  True),
        (P_th.to("erg/cm**3").value,   P_up.to("erg/cm**3"),P_dn_pred.to("erg/cm**3"),P_dn_meas.to("erg/cm**3"),
         r"$P_{\rm thermal}$ [erg cm$^{-3}$]", "tab:red", True),
    ]

    x_um       = x.to("um").value
    x_shock_um = float(x_shock.to("um"))
    x_ds_um    = float(x_ds.to("um"))

    fig, axes = plt.subplots(2, 3, figsize=(19, 9), sharex=True)
    for ax, (prof, u_val, pred_val, meas_val, ylabel, color, log) in zip(
            axes.flat, panels):
        ax.plot(x_um, prof, color=color, lw=1.6, label="FLASH lineout")
        # upstream mean (drawn over the upstream region, to the right of front)
        ax.hlines(float(u_val), x_shock_um, x_um.max(), color="0.45", ls="-", lw=1.4,
                  label="upstream mean")
        # theory-predicted downstream value (over the downstream region)
        ax.hlines(float(pred_val), x_ds_um, x_shock_um, color="k", ls="--", lw=2.0,
                  label="RH predicted dn")
        # measured downstream mean
        ax.hlines(float(meas_val), x_ds_um, x_shock_um, color="k", ls=":", lw=2.0,
                  label="measured dn")
        ax.axvline(x_shock_um, color="k", lw=1.0, alpha=0.6)
        ax.axvline(x_ds_um,    color="0.6", lw=1.0, alpha=0.6)
        if log:
            ax.set_yscale("log")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25, which="both")
    for ax in axes[-1]:
        ax.set_xlabel(r"distance along LOS [$\mu$m]")
    axes[0, 0].legend(loc="best", fontsize=8)

    # Zoom around the front (upstream to the right, downstream to the left).
    lo_x = max(x_um.min(), x_shock_um - args.window_um)
    hi_x = min(x_um.max(), x_shock_um + args.window_um)
    axes[0, 0].set_xlim(lo_x, hi_x)

    fig.suptitle(
        f"Perpendicular MHD prediction vs FLASH — {os.path.basename(snap_file)} "
        f"(t = {t.to('ns'):.2f})\n"
        f"$\\theta_{{Bn}}$ = {np.degrees(theta_bn):.0f}°   "
        f"$M_s$ = {mach_s:.2f}   $M_A$ = {mach_a:.2f}   "
        f"$\\gamma$ = {gamma:.3f}   →   $r$ = {r:.2f}",
        fontsize=12,
    )
    fig.tight_layout()
    fig_path = os.path.join(
        out_dir, f"flash_rh_prediction_{os.path.basename(snap_file)}.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved → {fig_path}")

    # ------------------------------------------------------------------
    # Save arrays (plain CGS magnitudes; unit named in each key)
    # ------------------------------------------------------------------
    npz_path = os.path.join(
        out_dir, f"flash_rh_prediction_{os.path.basename(snap_file)}.npz")
    np.savez(
        npz_path,
        x_um=x_um, t_ns=np.asarray(float(t.to("ns"))),
        ne=lo["ne"].to("cm**-3").value,
        B_perp=B_perp.to("gauss").value,
        Te=lo["Te"].to("eV").value,
        Ti=lo["Ti"].to("eV").value,
        P_th=P_th.to("erg/cm**3").value,
        v_sf_kms=v_sf.to("km/s").value,
        x_shock_cm=np.asarray(float(x_shock.to("cm"))),
        x_downstream_start_cm=np.asarray(float(x_ds.to("cm"))),
        v_shock_cms=np.asarray(float(v_shock.to("cm/s"))),
        gamma=np.asarray(gamma), theta_bn_rad=np.asarray(theta_bn),
        mach_s=np.asarray(mach_s), mach_a=np.asarray(mach_a),
        r=np.asarray(r), p_ratio=np.asarray(p_ratio), T_ratio=np.asarray(T_ratio),
        # upstream / predicted-dn / measured-dn for each channel
        ne_up=np.asarray(float(ne_up.to("cm**-3"))),
        ne_dn_pred=np.asarray(float(ne_dn_pred.to("cm**-3"))),
        ne_dn_meas=np.asarray(float(ne_dn_meas.to("cm**-3"))),
        Bperp_up=np.asarray(float(Bperp_up.to("gauss"))),
        Bperp_dn_pred=np.asarray(float(Bperp_dn_pred.to("gauss"))),
        Bperp_dn_meas=np.asarray(float(Bperp_dn_meas.to("gauss"))),
        Te_up=np.asarray(float(Te_up.to("eV"))),
        Te_dn_pred=np.asarray(float(Te_dn_pred.to("eV"))),
        Te_dn_meas=np.asarray(float(Te_dn_meas.to("eV"))),
        Ti_up=np.asarray(float(Ti_up.to("eV"))),
        Ti_dn_pred=np.asarray(float(Ti_dn_pred.to("eV"))),
        Ti_dn_meas=np.asarray(float(Ti_dn_meas.to("eV"))),
        P_up=np.asarray(float(P_up.to("erg/cm**3"))),
        P_dn_pred=np.asarray(float(P_dn_pred.to("erg/cm**3"))),
        P_dn_meas=np.asarray(float(P_dn_meas.to("erg/cm**3"))),
        v_inflow_cms=np.asarray(float(v_inflow.to("cm/s"))),
        vsf_dn_pred_cms=np.asarray(float(vsf_dn_pred.to("cm/s"))),
        vsf_dn_meas_cms=np.asarray(float(vsf_dn_meas.to("cm/s"))),
        config_path=np.asarray(os.path.abspath(args.config)),
    )
    print(f"Saved → {npz_path}")


if __name__ == "__main__":
    main()
