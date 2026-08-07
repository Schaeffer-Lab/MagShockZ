# -*- coding: utf-8 -*-
"""scripts/osiris_rh_prediction.py — the ideal-MHD downstream state, overlaid on the PIC
lineouts so the run's DEPARTURE from it is read off directly.

Predicts the downstream state from the measured upstream plus the shock speed, using the
perpendicular (theta = 90 deg) MHD jump in ``magshockz/common/perpendicular_shock.py``,
and plots each quantity across the front against three lines: the upstream mean, the
theory prediction, and the measured downstream mean.

The OSIRIS analog of ``scripts/flash_rh_prediction.py``, working entirely in normalized
units, so no conversions are needed (see ``docs/physics_notes.md``).

The upstream inputs to the jump::

    c_s      = sqrt(gamma (T_e + T_i) / |rqm_i|)     two-temperature sound speed
    v_A      = sqrt(B_perp^2 / (|rqm_i| n_e))        TRANSVERSE-field Alfven speed
    v_inflow = |v_shock - v_para|                    shock-frame normal inflow

``B_perp = sqrt(b2^2 + b3^2)`` is the shock-tangential field (normal = x1, so b1 is the
parallel one).  Using B_perp rather than the total |B| is what the theta = 90 deg jump
assumes; ``theta_Bn = atan2(|B_perp|, |B_para|)`` is reported so you can see how
perpendicular the run actually is.

**Why the PIC run should diverge from these lines.** Single-fluid ideal MHD assumes a
scalar pressure and ONE temperature, so it predicts a single T2/T1 applied to T_e and T_i
alike.  It has no separate electron/ion heating, no pressure anisotropy, no reflected-ion
foot, no finite-Larmor-radius overshoot — exactly the effects PIC keeps.  So the
electron/ion temperature split, the overshoot at the ramp, and the downstream departures
are the kinetic story this baseline exists to make visible.

Usage
-----
    conda activate analysis
    python scripts/osiris_rh_prediction.py --config config/perlmutter_1.3.1d.yaml \\
        [--timestep-idx -1] [--gamma 1.6667] [--window 400] \\
        [--units electron|ion] [--output-dir results/<run>]

Run in the `analysis` conda env (it has osh5io / osiris_utils).
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))

from magshockz.common import analysis_utils
from magshockz.common import plot_style
from magshockz.analysis.osiris import shock_state
from magshockz.common import dimensionless_params as dp
from magshockz.common import perpendicular_shock as ps


def _region_mean(profile, mask):
    """nanmean of a profile over a boolean mask (nan if the mask is empty)."""
    return float(np.nanmean(profile[mask])) if mask.any() else float("nan")


def main():
    parser = argparse.ArgumentParser(
        description="Predict the downstream OSIRIS state from upstream + v_shock "
                    "(perpendicular MHD shock) and overlay it on lineouts to show "
                    "where the PIC run diverges from ideal MHD.")
    parser.add_argument("--config", required=True, help="Path to analysis YAML config.")
    parser.add_argument("--timestep-idx", type=int, default=-1, dest="timestep_idx",
                        help="Index into the config times list (default -1 = last dump).")
    parser.add_argument("--gamma", type=float, default=None,
                        help="Adiabatic index (default: config 'gamma' key, else 5/3). "
                             "gamma=(f+2)/f: 5/3 (3 DOF), 2 (2 DOF), 3 (1 DOF) — sweep "
                             "to read off the effective index.")
    parser.add_argument("--window", type=float, default=None,
                        help="Half-width of the zoom window around the front, in the "
                             "display length unit (default: whole box).")
    parser.add_argument("--output-dir", default=None, dest="output_dir",
                        help="Output dir (default results/<run_name>).")
    parser.add_argument("--no-piston-mask", action="store_true", dest="no_piston_mask",
                        help="Do NOT exclude the silicon piston from the up/downstream "
                             "averages. By default cells where the silicon (piston) ion "
                             "density exceeds the aluminium density are dropped, so only "
                             "aluminium-populated cells count as upstream/downstream.")
    plot_style.add_publication_arg(parser)
    plot_style.add_units_arg(parser)
    args = parser.parse_args()
    plot_style.apply(args.publication)

    # ------------------------------------------------------------------
    # Load one dump through the shared shock-state loader (single code path).
    # ------------------------------------------------------------------
    cfg = analysis_utils.load_config(args.config)
    sim_dir = cfg["sim_dir"]
    gamma = args.gamma if args.gamma is not None else float(cfg.get("gamma", 5.0 / 3.0))
    disp = plot_style.build_units_from_args(args, cfg)

    print(f"Config  : {args.config}")
    print(f"sim_dir : {sim_dir}")
    print("Loading HDF5 files...", flush=True)
    st = shock_state.load_shock_state(cfg, args.timestep_idx)

    abs_rqm_i = st.abs_rqm_i
    ion = st.ion

    # ------------------------------------------------------------------
    # Per-cell profiles on the phase-space grid (OSIRIS normalised units).
    # Shock normal is x1, so the parallel field is b1 and the shock-tangential
    # (perpendicular) field, which the theta = 90 deg jump compresses, is
    # B_perp = sqrt(b2^2 + b3^2).
    # ------------------------------------------------------------------
    x = st.x_pha
    n_e = st.n_e
    T_e = st.T_iso["e"]
    T_i = st.T_iso[ion]
    B_para = st.fields["b1"]
    B_perp = np.sqrt(st.fields["b2"] ** 2 + st.fields["b3"] ** 2)
    P_th = n_e * (T_e + T_i)                       # thermal pressure [n_0 m_e c^2]

    # Ion bulk velocity profile [c] is not stored on ShockState as a full profile,
    # only its region means (prim_up/prim_dn["u_bulk_i"]).  Recompute it here from
    # the same p1x1 phase space the loader used, so we can plot |v_i - v_shock|.
    from magshockz.common import moments as mom_module
    u_bulk_i = mom_module.moment(st.pha_p1[ion], axis="p1", order=1)   # [c]
    v_sf = np.abs(u_bulk_i - st.v_shock)                              # shock-frame [c]

    # Piston exclusion: the shock lives in the AMBIENT (aluminium) plasma; the
    # silicon is the driving piston, not up/downstream.  Flag cells where the
    # silicon ion density exceeds the aluminium ion density as piston and drop
    # them from both region masks, so only aluminium-populated cells are averaged.
    import osh5io
    piston_mask = np.zeros_like(x, dtype=bool)
    if not args.no_piston_mask:
        try:
            pha_si = osh5io.read_h5(
                analysis_utils.diag_path(sim_dir, "p1x1", st.t_val, "si"))
            n_si = np.abs(mom_module.moment(pha_si, axis="p1", order=0))
            n_al = np.abs(mom_module.moment(st.pha_p1[ion], axis="p1", order=0))
            piston_mask = n_si > n_al        # silicon-dominated => piston
        except Exception as exc:             # si diagnostic absent -> keep full window
            print(f"  (silicon piston diagnostic unavailable: {exc}; keeping full window)")

    up = st.upstream_mask & ~piston_mask
    dn = st.downstream_mask & ~piston_mask
    n_dn_dropped = int((st.downstream_mask & piston_mask).sum())

    # ------------------------------------------------------------------
    # Upstream region averages -> perpendicular-shock inputs.
    # ------------------------------------------------------------------
    n_e_up   = _region_mean(n_e, up)
    T_e_up   = _region_mean(T_e, up)
    T_i_up   = _region_mean(T_i, up)
    Bperp_up = _region_mean(B_perp, up)
    Bpara_up = _region_mean(np.abs(B_para), up)
    vpara_up = _region_mean(u_bulk_i, up)
    P_up     = n_e_up * (T_e_up + T_i_up)

    v_inflow = abs(st.v_shock - vpara_up)          # shock-frame normal inflow [c]

    # Mach numbers with v_A built from the TRANSVERSE field (perpendicular-shock
    # convention).  Reuse the tested dimensionless_params formulas by handing them
    # a primitives dict whose B2 is the transverse field squared, and passing
    # v_inflow as the (shock-frame) speed so M_s, M_A are the jump's Mach numbers.
    prim_perp = {"n_e": n_e_up, "T_e": T_e_up, "T_i": T_i_up, "B2": Bperp_up ** 2}
    params = dp.compute_dimensionless(prim_perp, v_inflow, abs_rqm_i, gamma)
    c_s, v_A = params["c_s"], params["v_A"]
    mach_s, mach_a, beta1 = params["M_s"], params["M_A"], params["beta"]
    theta_bn = float(np.arctan2(abs(Bperp_up), abs(Bpara_up)))

    # ------------------------------------------------------------------
    # Solve the perpendicular MHD jump and predict the downstream state.
    # ------------------------------------------------------------------
    jump = ps.solve(mach_s, mach_a, gamma)
    r, p_ratio, T_ratio = jump["r"], jump["p_ratio"], jump["T_ratio"]

    if not jump["exists"] or not np.isfinite(r):
        print("\n!! No compressive perpendicular shock for these upstream numbers "
              f"(M_s={mach_s:.2f}, M_A={mach_a:.2f}). Predictions will be NaN.")

    pred = ps.predict_downstream(
        jump, rho1=n_e_up, B_perp1=Bperp_up, p1=P_up, v_inflow=v_inflow)
    ne_dn_pred    = pred.get("rho", float("nan"))       # n scales like rho
    Bperp_dn_pred = pred.get("B_perp", float("nan"))
    P_dn_pred     = pred.get("p", float("nan"))
    vsf_dn_pred   = pred.get("v_inflow", float("nan"))
    Te_dn_pred    = T_ratio * T_e_up
    Ti_dn_pred    = T_ratio * T_i_up

    # Measured downstream means (the PIC truth to compare the theory against).
    ne_dn_meas    = _region_mean(n_e, dn)
    Bperp_dn_meas = _region_mean(B_perp, dn)
    P_dn_meas     = _region_mean(P_th, dn)
    vsf_dn_meas   = _region_mean(v_sf, dn)
    Te_dn_meas    = _region_mean(T_e, dn)
    Ti_dn_meas    = _region_mean(T_i, dn)

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    print(f"\nt_sim   : {st.t_sim:.1f} [1/omega_pe]   dump t_val = {st.t_val}")
    print(f"x_shock : {st.x_shock:.1f} [c/omega_pe]   v_shock = {st.v_shock:.4f} [c]")
    if not args.no_piston_mask:
        print(f"piston  : dropped {n_dn_dropped} silicon-dominated cell(s) from the "
              f"downstream window (aluminium-only up/downstream).")
    print("\n--- Upstream state (region average, OSIRIS units) ---")
    print(f"  n_e = {n_e_up:.4g} [n_0]   B_perp = {Bperp_up:.4g} [B_0]   "
          f"T_e = {T_e_up:.4g}   T_i = {T_i_up:.4g} [m_e c^2]")
    print(f"  c_s = {c_s:.4g}   v_A = {v_A:.4g}   v_inflow = {v_inflow:.4g}   [c]")
    print(f"  theta_Bn = {np.degrees(theta_bn):.1f} deg   beta1 = {beta1:.3g}   "
          f"M_s = {mach_s:.2f}   M_A = {mach_a:.2f}")
    print(f"\n--- Perpendicular MHD prediction (gamma = {gamma:.4f}) ---")
    print(f"  r = rho2/rho1 = {r:.3f}   p2/p1 = {p_ratio:.3f}   T2/T1 = {T_ratio:.3f}")
    sep = "-" * 76
    print(sep)
    print(f"  {'quantity':<22}{'upstream':>12}{'pred. dn':>12}{'meas. dn':>12}{'meas/pred':>12}")
    print(sep)
    rows = [
        ("n_e [n_0]",          n_e_up,   ne_dn_pred,    ne_dn_meas),
        ("B_perp [B_0]",       Bperp_up, Bperp_dn_pred, Bperp_dn_meas),
        ("T_e [m_e c^2]",      T_e_up,   Te_dn_pred,    Te_dn_meas),
        ("T_i [m_e c^2]",      T_i_up,   Ti_dn_pred,    Ti_dn_meas),
        ("P_th [n0 me c^2]",   P_up,     P_dn_pred,     P_dn_meas),
        ("|v-v_sh| [c]",       v_inflow, vsf_dn_pred,   vsf_dn_meas),
    ]
    for name, u_val, pdn, mdn in rows:
        ratio = mdn / pdn if (np.isfinite(pdn) and pdn != 0.0) else float("nan")
        print(f"  {name:<22}{u_val:>12.4g}{pdn:>12.4g}{mdn:>12.4g}{ratio:>12.3f}")
    print(sep)
    print("  meas/pred != 1 is the PIC run's departure from ideal MHD.")

    # ------------------------------------------------------------------
    # Figure — lineouts across the front with the predicted downstream line
    # ------------------------------------------------------------------
    # Panels: (profile, upstream mean, predicted dn, measured dn, y-label, colour, log-y?)
    panels = [
        (n_e,    n_e_up,   ne_dn_pred,    ne_dn_meas,    r"$n_e\ [n_0]$",                 "tab:purple", False),
        (B_perp, Bperp_up, Bperp_dn_pred, Bperp_dn_meas, r"$B_\perp\ [B_0]$",             "tab:orange", False),
        (v_sf,   v_inflow, vsf_dn_pred,   vsf_dn_meas,   r"$|v_i - v_{\rm sh}|\ [c]$",     "tab:blue",   False),
        (T_e,    T_e_up,   Te_dn_pred,    Te_dn_meas,    r"$T_e\ [m_e c^2]$",             "tab:green",  True),
        (T_i,    T_i_up,   Ti_dn_pred,    Ti_dn_meas,    r"$T_i\ [m_e c^2]$",             "tab:brown",  True),
        (P_th,   P_up,     P_dn_pred,     P_dn_meas,     r"$P_{\rm th}\ [n_0 m_e c^2]$",   "tab:red",    True),
    ]

    x_disp       = disp.x(x)
    x_shock_disp = float(disp.x(st.x_shock))
    # Extent of the (piston-excluded) downstream region actually averaged.
    x_dn_edge = float(disp.x(x[dn].min())) if dn.any() else x_shock_disp

    fig, axes = plt.subplots(2, 3, figsize=(19, 9), sharex=True)
    for j, (ax, (prof, u_val, pred_val, meas_val, ylabel, color, log)) in enumerate(
            zip(axes.flat, panels)):
        first = j == 0
        span = ax.get_xaxis_transform()   # x in data coords, y in [0,1] axes coords
        ax.plot(x_disp, prof, color=color, lw=1.4, label="OSIRIS lineout")
        # upstream mean (over the upstream region, ahead of the front)
        ax.hlines(u_val, x_shock_disp, x_disp.max(), color="0.45", ls="-", lw=1.4,
                  label="upstream mean")
        # theory-predicted downstream value (dashed) vs measured mean (dotted)
        ax.hlines(pred_val, x_dn_edge, x_shock_disp, color="k", ls="--", lw=2.0,
                  label="MHD predicted dn")
        ax.hlines(meas_val, x_dn_edge, x_shock_disp, color="k", ls=":", lw=2.0,
                  label="measured dn")
        ax.axvline(x_shock_disp, color="k", lw=1.0, alpha=0.6)
        # Shade the aluminium downstream cells actually averaged, and mark the
        # silicon piston cells that were excluded (drawn per-cell via the masks).
        ax.fill_between(x_disp, 0, 1, where=dn, transform=span, color="0.80",
                        alpha=0.5, zorder=0, label="downstream (Al)" if first else None)
        if piston_mask.any():
            ax.fill_between(x_disp, 0, 1, where=piston_mask, transform=span,
                            color="tab:red", alpha=0.12, zorder=0,
                            label="piston (Si)" if first else None)
        if log:
            ax.set_yscale("log")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25, which="both")
    for ax in axes[-1]:
        ax.set_xlabel(disp.xlabel())
    axes[0, 0].legend(loc="best", fontsize=8)

    if args.window is not None:
        lo_x = max(x_disp.min(), x_shock_disp - args.window)
        hi_x = min(x_disp.max(), x_shock_disp + args.window)
        axes[0, 0].set_xlim(lo_x, hi_x)

    fig.suptitle(
        f"Perpendicular MHD prediction vs OSIRIS — {os.path.basename(sim_dir.rstrip('/'))} "
        f"(dump {st.t_val}, {disp.time_title(st.t_sim)})\n"
        rf"$\theta_{{Bn}}$ = {np.degrees(theta_bn):.0f}$\degree$   "
        rf"$\beta_1$ = {beta1:.2f}   $M_s$ = {mach_s:.2f}   $M_A$ = {mach_a:.2f}   "
        rf"$\gamma$ = {gamma:.3f}   $\rightarrow$   $r$ = {r:.2f}",
        fontsize=12,
    )
    fig.tight_layout()

    out_dir = args.output_dir or os.path.join(
        _HERE, "..", "results", os.path.basename(sim_dir.rstrip("/")))
    os.makedirs(out_dir, exist_ok=True)
    fig_path = os.path.join(out_dir, f"osiris_rh_prediction_t{st.t_val:06d}.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved → {fig_path}")

    # ------------------------------------------------------------------
    # Save arrays (OSIRIS normalised units; unit named in the docstring/keys)
    # ------------------------------------------------------------------
    npz_path = os.path.join(out_dir, f"osiris_rh_prediction_t{st.t_val:06d}.npz")
    np.savez(
        npz_path,
        x_cwpe=np.asarray(x), t_sim=np.asarray(st.t_sim), t_val=np.asarray(st.t_val),
        piston_mask=np.asarray(piston_mask), n_dn_dropped=np.asarray(n_dn_dropped),
        n_e=np.asarray(n_e), B_perp=np.asarray(B_perp), B_para=np.asarray(B_para),
        T_e=np.asarray(T_e), T_i=np.asarray(T_i), P_th=np.asarray(P_th),
        v_sf=np.asarray(v_sf),
        x_shock=np.asarray(st.x_shock), v_shock=np.asarray(st.v_shock),
        v_inflow=np.asarray(v_inflow), c_s=np.asarray(c_s), v_A=np.asarray(v_A),
        gamma=np.asarray(gamma), theta_bn_rad=np.asarray(theta_bn),
        beta1=np.asarray(beta1), mach_s=np.asarray(mach_s), mach_a=np.asarray(mach_a),
        r=np.asarray(r), p_ratio=np.asarray(p_ratio), T_ratio=np.asarray(T_ratio),
        # upstream / predicted-dn / measured-dn for each channel
        ne_up=np.asarray(n_e_up), ne_dn_pred=np.asarray(ne_dn_pred), ne_dn_meas=np.asarray(ne_dn_meas),
        Bperp_up=np.asarray(Bperp_up), Bperp_dn_pred=np.asarray(Bperp_dn_pred), Bperp_dn_meas=np.asarray(Bperp_dn_meas),
        Te_up=np.asarray(T_e_up), Te_dn_pred=np.asarray(Te_dn_pred), Te_dn_meas=np.asarray(Te_dn_meas),
        Ti_up=np.asarray(T_i_up), Ti_dn_pred=np.asarray(Ti_dn_pred), Ti_dn_meas=np.asarray(Ti_dn_meas),
        P_up=np.asarray(P_up), P_dn_pred=np.asarray(P_dn_pred), P_dn_meas=np.asarray(P_dn_meas),
        vsf_dn_pred=np.asarray(vsf_dn_pred), vsf_dn_meas=np.asarray(vsf_dn_meas),
        config_path=np.asarray(os.path.abspath(args.config)),
    )
    print(f"Saved → {npz_path}")


if __name__ == "__main__":
    main()
