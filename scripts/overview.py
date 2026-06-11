"""scripts/overview.py — general overview of a MagShockZ OSIRIS shock run.

Time-space "streak" plots (time on the horizontal axis, space on the vertical)
for the quantities that track a collisionless shock front:

    |B|   magnetic compression        [B_0]
    n_e   electron number density     [n_0]
    T_i   ion temperature (parallel)  [m_e c^2]

The config's linear shock-front fit (x_shock = x_shock_0 + v_shock * t_sim) is
overlaid on every streak.  A per-frame front is also detected from the density
compression and fit independently, so the quoted shock velocity can be checked
against the data.

A second figure collects the single-dump diagnostics shock studies usually
want at a representative time:
    - ion and electron p1-x phase spaces (the canonical reflected-ion view)
    - n_e and |B| line-outs across the shock
    - T_i and T_e (parallel) line-outs across the shock

Paradigms follow the other scripts: --config driven with $MAGSHOCKZ_SIM_DIR
override, MagShockZRun for unit context, StreakBuilder for [time, x] assembly,
results saved under results/<run_name>/ with the figures alongside.

Unlike the single-dump scripts this one sweeps many dumps, so it does NOT call
resolve_dump_params (which requires a per-dump config entry); the shock-front
overlay comes straight from cfg["shock"] = {v_shock, x_shock_0}.

Usage
-----
    python scripts/overview.py --config config/perlmutter_1.3.1d.yaml \\
        [--stride 16] [--t-start 0] [--t-stop 512] \\
        [--snapshot-idx -1] [--output-dir DIR]
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import osh5def
import osh5io
import osh5vis
from matplotlib.colors import LogNorm

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

import analysis_utils
import temperature_anisotropy as ta
from analysis_utils import (
    StreakBuilder, axis_values, field_path, density_path, phase_path,
    transverse_profile,
)
# Shock-front detection / trajectory fitting is single-sourced in src/shock.py.
from shock import detect_front_edge as detect_front
from shock import robust_linfit


# ---------------------------------------------------------------------------
# Per-dump 1D profile frames (H5Data, so StreakBuilder can stack them)
#
# Fields and density are full spatial maps (1D, or 2D for a 2D run); each is
# reduced to a 1D profile along the shock-normal axis by transverse_profile
# (a no-op in 1D).  Phase spaces already carry a single spatial axis, so the
# momentum moment collapses them straight to a 1D profile.
# ---------------------------------------------------------------------------

def bmag_frame(sim_dir: str, t: int, layout, hw: float):
    """|B| = sqrt(b1^2 + b2^2 + b3^2), reduced to a 1D profile [B_0]."""
    b = {q: osh5io.read_h5(field_path(sim_dir, q, t))
         for q in ("b1", "b2", "b3")}
    bmag = np.sqrt(b["b1"] ** 2 + b["b2"] ** 2 + b["b3"] ** 2)  # stays H5Data
    # Keep the propagated UNITS (osh5def carries the field unit m_e c ω_p/e through
    # the sqrt) — do NOT relabel as B_0: the data is in OSIRIS field-normalisation
    # units, not normalised to an upstream B_0.  LONG_NAME is bare TeX (osh5vis wraps).
    bmag.data_attrs = dict(bmag.data_attrs, NAME="|B|", LONG_NAME=r"|B|")
    return transverse_profile(bmag, layout.normal_axis, hw)


def density_frame(sim_dir: str, sp: str, t: int, layout, hw: float):
    """Number density n = |charge|, reduced to a 1D profile [n_0].

    The OSIRIS ``charge`` diagnostic is q·n in normalised charge-density units; the
    UNITS are relabelled to n_0 here because |charge| = |q|·n/n_0 equals n/n_0 only
    for a singly-charged species.  overview.py calls this only for electrons (q=−1),
    where that holds; do NOT use it for multiply-charged ions without dividing by Z.
    """
    ch = osh5io.read_h5(density_path(sim_dir, sp, t, savg=layout.density_savg))
    n = np.abs(ch)
    n.data_attrs = dict(n.data_attrs, NAME=f"n_{sp}", LONG_NAME=fr"n_\mathrm{{{sp}}}", UNITS="n_0")
    return transverse_profile(n, layout.normal_axis, hw)


def temperature_frame(sim_dir: str, sp: str, t: int, rqm: float, layout, axis: str = "p1"):
    """Parallel temperature T = |rqm| * <(p - <p>)^2> as an H5Data on the phase grid.

    The moment collapses f(p, x) -> T(x); the result is re-wrapped as an H5Data
    carrying the phase-space spatial axis and run_attrs (for the TIME StreakBuilder
    needs) so it can be stacked exactly like the field/density frames.  The
    phase-space name (p1x1 vs p1x2) comes from the run layout.
    """
    ps = osh5io.read_h5(phase_path(sim_dir, layout.pha_name(int(axis[1:])), sp, t))
    T = np.asarray(ta.temperature_profile(ps, rqm, axis))
    x_axis = next(a for a in ps.axes if a.name != axis)  # the spatial axis
    return osh5def.H5Data(
        T,
        data_attrs={"NAME": f"T_{sp}", "LONG_NAME": fr"T_{{{sp}}}", "UNITS": "m_e c^2"},
        run_attrs=ps.run_attrs,
        axes=[x_axis],
    )


# ---------------------------------------------------------------------------
# Streak assembly + shock-front detection
# ---------------------------------------------------------------------------

def assemble_streak(frames):
    """StreakBuilder a list of 1D H5Data frames -> (streak[time, x] H5Data, Z, time[], x[]).

    The H5Data ``streak`` (axes [time, x], carrying NAME/LONG_NAME/UNITS) is what
    osh5vis plots; Z/time/x are the plain-numpy views the .npz and detection use.
    """
    streak = StreakBuilder(frames).build()
    Z = np.asarray(streak)
    time = axis_values(streak, 0)
    x = axis_values(streak, 1)
    return streak, Z, time, x


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def plot_streak(ax, streak, x, *, title, cmap, log, shock_lines):
    """Render one streak (time horizontal, space vertical) with shock overlays.

    ``streak`` is the [time, x] H5Data; it is transposed to [x, time] and drawn
    with osh5vis.osimshow, which sets the time/space axis labels and the colorbar
    *unit* from the data's own ``data_attrs`` (so the displayed unit always matches
    the data, never a hardcoded guess).  ``title`` is just the quantity name; the
    percentile colour limits and shock-line overlays are layered on top.

    shock_lines : list of (time_arr, x_arr, style_kwargs, legend_label)
    """
    C = np.asarray(streak)
    finite = C[np.isfinite(C)]
    if log:
        pos = finite[finite > 0]
        vmax = np.percentile(pos, 99.5) if pos.size else 1.0
        vmin = max(np.percentile(pos, 2) if pos.size else 1e-6, vmax * 1e-4)
        kw = dict(norm=LogNorm(vmin=vmin, vmax=vmax))
    else:
        vmax = np.percentile(finite, 99.5) if finite.size else 1.0
        kw = dict(vmin=0.0, vmax=vmax)
    # osimshow draws [x, time] (time horizontal, x vertical), auto axes; colorbar
    # label is derived from the data's UNITS (no cblabel override).
    osh5vis.osimshow(streak.transpose(), ax=ax, cmap=cmap, title=title, **kw)
    for t_arr, x_arr, style, leg in shock_lines:
        ax.plot(t_arr, x_arr, label=leg, **style)
    ax.set_ylim(x.min(), x.max())
    ax.legend(fontsize=8, loc="upper left", framealpha=0.7)


def plot_phasespace(ax, ps, *, title, x_shock=None):
    """osimshow a p-x phase space (x horizontal, p vertical) on a log color scale.

    OSIRIS phase spaces are charge-weighted, so electron f is stored negative;
    take |f| so both species share one positive log color scale.  osh5io stores
    the data as [p, x] (in both 1D and the 2D run), so osh5vis.osimshow places the
    spatial axis horizontal and momentum vertical and labels both from the data's
    own metadata — no explicit normal-axis needed.
    """
    data = np.abs(ps)                                   # H5Data, metadata preserved
    arr = np.asarray(data)
    vmax = np.percentile(arr[arr > 0], 99.7) if (arr > 0).any() else 1.0
    vmin = vmax * 1e-4
    osh5vis.osimshow(data, ax=ax, cmap="inferno",
                     norm=LogNorm(vmin=vmin, vmax=vmax), title=title,
                     cblabel=r"$f$ (arb.)")
    if x_shock is not None and np.isfinite(x_shock):
        ax.axvline(x_shock, color="cyan", ls="--", lw=1.2)


def main():
    parser = argparse.ArgumentParser(description="General overview / streak plots of a shock run.")
    parser.add_argument("--config", required=True, help="Path to analysis YAML config.")
    parser.add_argument("--stride", type=int, default=4, dest="field_stride",
                        help="Dump stride for the cheap |B| and n_e streaks (even; default 4; "
                             "2 = every saved field dump, the maximum time resolution).")
    parser.add_argument("--phase-stride", type=int, default=16, dest="phase_stride",
                        help="Dump stride for the T_i streak (even; default 16). Phase-space "
                             "reads (~260 MB each) dominate runtime, so this is coarser than --stride.")
    parser.add_argument("--t-start", type=int, default=0, dest="t_start")
    parser.add_argument("--t-stop", type=int, default=None, dest="t_stop",
                        help="Last dump index (default: largest available).")
    parser.add_argument("--snapshot-idx", type=int, default=-1, dest="snapshot_idx",
                        help="Index into the field-dump list for the phase-space snapshot (default -1).")
    parser.add_argument("--search-halfwidth", type=float, default=400.0, dest="search_hw",
                        help="Half-width [c/wpe] of the shock-detection window around the config fit.")
    parser.add_argument("--transverse-halfwidth", type=float, default=5.0, dest="transverse_hw",
                        help="Half-width [c/wpe] of the central transverse band averaged when "
                             "reducing 2D field/density maps to a 1D shock-normal profile "
                             "(default 5 = 5 electron inertial lengths; ignored for 1D runs).")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    cfg = analysis_utils.load_config(args.config)
    sim_dir = cfg["sim_dir"]

    layout = analysis_utils.detect_layout(sim_dir)
    hw = args.transverse_hw

    sim = analysis_utils.run_from_config(cfg)
    rqm_i = sim.rqm_of("al")
    v_shock_cfg = float(cfg["shock"]["v_shock"])
    x_shock_0 = float(cfg["shock"]["x_shock_0"])

    # ------------------------------------------------------------------
    # Build the dump lists.  Fields/density are cheap (small 1D files), so the
    # |B| and n_e streaks run at a fine stride; T_i needs ~260 MB phase-space
    # reads, so it uses its own coarser stride.  Each streak carries its own
    # time axis, so the cadences need not match.
    # ------------------------------------------------------------------
    t_stop = args.t_stop
    if t_stop is None:
        b3_dir = f"{sim_dir}/MS/FLD/b3-savg"
        idxs = [int(f.split("-")[-1].split(".")[0]) for f in os.listdir(b3_dir) if f.endswith(".h5")]
        t_stop = max(idxs)

    def dump_list(stride, require_phase):
        stride = stride + (stride % 2)  # keep even so field savg files exist
        out = []
        for t in range(args.t_start, t_stop + 1, stride):
            if not os.path.exists(field_path(sim_dir, "b3", t)):
                continue
            if not os.path.exists(density_path(sim_dir, "e", t, savg=layout.density_savg)):
                continue
            if require_phase and not os.path.exists(phase_path(sim_dir, layout.pha_name(1), "al", t)):
                continue
            out.append(t)
        return out

    field_dumps = dump_list(args.field_stride, require_phase=False)
    phase_dumps = dump_list(args.phase_stride, require_phase=True)
    if len(field_dumps) < 2 or len(phase_dumps) < 2:
        raise RuntimeError(
            f"Need >=2 dumps per streak; got field={len(field_dumps)}, phase={len(phase_dumps)} "
            f"in [{args.t_start},{t_stop}]."
        )

    print(f"Config  : {args.config}")
    print(f"sim_dir : {sim_dir}")
    print(f"layout  : {layout}  (transverse band ±{hw:g} c/wpe about center)")
    print(f"|B|,n_e : {len(field_dumps)} frames, {field_dumps[0]}..{field_dumps[-1]} (stride {field_dumps[1]-field_dumps[0]})")
    print(f"T_i     : {len(phase_dumps)} frames, {phase_dumps[0]}..{phase_dumps[-1]} (stride {phase_dumps[1]-phase_dumps[0]})")
    print(f"|rqm_i| : {abs(rqm_i):.4f}   v_shock(cfg) : {v_shock_cfg:.5f} c   x_shock_0 : {x_shock_0:.1f}")

    # ------------------------------------------------------------------
    # Load frames and assemble streaks
    # ------------------------------------------------------------------
    # |B| (field grid) and n_e (density grid) can live on different normal-axis
    # grids — in the 2D run the fields are time-averaged (savg) but the density
    # is not, so they have different cell counts.  Each streak therefore keeps
    # its own spatial axis; combine via interpolation, never by shared indexing.
    print("Loading |B|, n_e frames...")
    B_h5,  B_streak,  time_f, x_f  = assemble_streak([bmag_frame(sim_dir, t, layout, hw) for t in field_dumps])
    ne_h5, ne_streak, _,      x_ne = assemble_streak([density_frame(sim_dir, "e", t, layout, hw) for t in field_dumps])
    print("Loading T_i frames (phase space)...")
    Ti_h5, Ti_streak, time_p, x_p = assemble_streak([temperature_frame(sim_dir, "al", t, rqm_i, layout) for t in phase_dumps])

    # ------------------------------------------------------------------
    # Per-frame shock-front detection (from the fine density streak) + fit
    # ------------------------------------------------------------------
    x_pred = x_shock_0 + v_shock_cfg * time_f
    x_det = np.array([
        detect_front(x_ne, ne_streak[i], x_pred[i], args.search_hw)
        for i in range(len(field_dumps))
    ])

    # The shock front is not magnetically organised until the upstream ions have
    # completed at least one gyro-orbit, so the trajectory fit must skip the
    # first ion gyroperiod.  In OSIRIS units the field value equals omega_ce
    # [1/wpe], hence omega_ci = |B|/|rqm_i| and T_ci = 2*pi*|rqm_i|/|B|.  Use the
    # ambient upstream |B| (median ahead of the t=0 predicted front) for |B|.
    upstream0 = x_f > x_shock_0
    B_upstream = float(np.median(B_streak[0][upstream0])) if upstream0.any() \
        else float(np.median(B_streak[0]))
    t_gyro = 2.0 * np.pi * abs(rqm_i) / B_upstream  # ion gyroperiod [1/wpe]

    # Upstream Alfven speed in OSIRIS units (v_A/c), measured from the t=0 frame.
    # In normalised units the field equals omega_ce [1/wpe], so omega_ci = |B|/|rqm_i|,
    # and the ion plasma frequency is omega_pi = sqrt(n_e/|rqm_i|) [wpe] (the upstream
    # charge density is n_e by quasineutrality).  Hence
    #     v_A/c = omega_ci/omega_pi = |B| / sqrt(|rqm_i| * n_e).
    # The config and detected shock speeds are then quoted as Alfvenic Mach numbers.
    upstream_ne = x_ne > x_shock_0
    ne_upstream = float(np.median(ne_streak[0][upstream_ne])) if upstream_ne.any() \
        else float(np.median(ne_streak[0]))
    v_A = B_upstream / np.sqrt(abs(rqm_i) * ne_upstream)  # [c]
    M_A_cfg = v_shock_cfg / v_A

    detected = np.isfinite(x_det)
    fit_mask = detected & (time_f >= t_gyro)
    if fit_mask.sum() >= 2:
        # Same outlier-rejecting fit as FLASH (shock.robust_linfit): on a clean
        # detected front it equals a plain polyfit, but stray detections are
        # sigma-clipped instead of biasing the trajectory.
        slope, intercept = robust_linfit(time_f[fit_mask], x_det[fit_mask])
        v_fit, x0_fit = float(slope), float(intercept)
    else:
        v_fit, x0_fit = float("nan"), float("nan")

    M_A_fit = v_fit / v_A

    print("\n--- Shock front ---")
    print(f"  upstream     : |B| = {B_upstream:.4g} [m_e c wpe/e],  n_e = {ne_upstream:.4g} n0,  v_A = {v_A:.5f} c")
    print(f"  config fit   : v = {v_shock_cfg:.5f} c = {M_A_cfg:.2f} v_A,  x0 = {x_shock_0:.1f} c/wpe")
    print(f"  ion gyroperiod : T_ci = {t_gyro:.1f} /wpe; fit skips t < T_ci")
    print(f"  detected fit : v = {v_fit:.5f} c = {M_A_fit:.2f} v_A,  x0 = {x0_fit:.1f} c/wpe  "
          f"({fit_mask.sum()}/{detected.sum()} fitted/detected frames)")

    cfg_line = (time_f, x_pred, dict(color="white", ls="-", lw=1.6),
                f"config fit (v={v_shock_cfg:.4f}c, $M_A$={M_A_cfg:.2f})")
    det_pts  = (time_f[detected], x_det[detected], dict(color="lime", ls="none", marker=".", ms=6), "detected front")
    fit_line = (time_f[time_f >= t_gyro], x0_fit + v_fit * time_f[time_f >= t_gyro],
                dict(color="lime", ls="--", lw=1.2),
                f"detected fit (v={v_fit:.4f}c, $M_A$={M_A_fit:.2f}, t≥T_ci)")

    out_dir = args.output_dir or os.path.join(_HERE, "..", "results", os.path.basename(sim_dir.rstrip("/")))
    os.makedirs(out_dir, exist_ok=True)
    tag = f"{field_dumps[0]:06d}-{field_dumps[-1]:06d}"

    # ------------------------------------------------------------------
    # Figure 1 — streaks
    # ------------------------------------------------------------------
    fig1, axes = plt.subplots(2, 2, figsize=(16, 10))
    plot_streak(axes[0, 0], B_h5, x_f,
                title=r"$|B|$", cmap="viridis", log=False,
                shock_lines=[cfg_line, det_pts, fit_line])
    plot_streak(axes[0, 1], ne_h5, x_ne,
                title=r"$n_e$", cmap="magma", log=True,
                shock_lines=[cfg_line, det_pts, fit_line])
    plot_streak(axes[1, 0], Ti_h5, x_p,
                title=r"$T_{i,\parallel}$", cmap="inferno", log=True,
                shock_lines=[cfg_line, det_pts, fit_line])

    axt = axes[1, 1]
    axt.plot(time_f, x_pred, color="tab:blue", lw=1.8,
             label=f"config fit (v={v_shock_cfg:.4f}c, $M_A$={M_A_cfg:.2f})")
    axt.plot(time_f[detected], x_det[detected], "o", color="tab:green", ms=5, label="detected front")
    if np.isfinite(v_fit):
        tf = time_f[time_f >= t_gyro]
        axt.plot(tf, x0_fit + v_fit * tf, "--", color="tab:green", lw=1.4,
                 label=f"detected fit (v={v_fit:.4f}c, $M_A$={M_A_fit:.2f}, t≥T_ci)")
    axt.axvline(t_gyro, color="0.4", ls=":", lw=1.2, label=fr"$T_{{ci}}$ = {t_gyro:.0f}/$\omega_{{pe}}$")
    axt.set_xlabel(r"$t$ [$\omega_{pe}^{-1}$]")
    axt.set_ylabel(r"$x_\mathrm{shock}$ [$c/\omega_{pe}$]")
    axt.set_title("Shock-front trajectory")
    axt.grid(alpha=0.3)
    axt.legend(fontsize=8)

    fig1.suptitle(f"Shock overview — {os.path.basename(sim_dir.rstrip('/'))}  (dumps {tag})", fontsize=13)
    fig1.tight_layout()
    streak_path = os.path.join(out_dir, f"overview_streaks_{tag}.png")
    fig1.savefig(streak_path, dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"\nSaved → {streak_path}")

    # ------------------------------------------------------------------
    # Figure 2 — representative single-dump diagnostics
    # ------------------------------------------------------------------
    t_snap = field_dumps[args.snapshot_idx]
    i_snap = args.snapshot_idx % len(field_dumps)
    x_shock_snap = x_det[i_snap] if np.isfinite(x_det[i_snap]) else x_pred[i_snap]

    ps_al = osh5io.read_h5(phase_path(sim_dir, layout.pha_name(1), "al", t_snap))
    ps_e  = osh5io.read_h5(phase_path(sim_dir, layout.pha_name(1), "e",  t_snap))

    fig2, ax2 = plt.subplots(2, 2, figsize=(16, 10))
    plot_phasespace(ax2[0, 0], ps_al, title=f"Ion $p_1$-$x$ phase space (t={time_f[i_snap]:.0f})",
                    x_shock=x_shock_snap)
    plot_phasespace(ax2[0, 1], ps_e,  title=f"Electron $p_1$-$x$ phase space (t={time_f[i_snap]:.0f})",
                    x_shock=x_shock_snap)

    # n_e and |B| line-outs (osplot1d draws each H5Data row on its own x-axis;
    # the twin y-axis, colours and shock marker are layered on afterward).
    axp = ax2[1, 0]
    # osplot1d sets each y-label (name + UNITS) from the data; just recolour them,
    # so |B| shows its true field unit rather than a hardcoded B_0.
    osh5vis.osplot1d(ne_h5[i_snap], ax=axp, color="tab:purple", show_time=False, title="")
    axp.yaxis.label.set_color("tab:purple")
    axp.tick_params(axis="y", labelcolor="tab:purple")
    axb = axp.twinx()
    osh5vis.osplot1d(B_h5[i_snap], ax=axb, color="tab:orange", show_time=False, title="")
    axb.yaxis.label.set_color("tab:orange")
    axb.tick_params(axis="y", labelcolor="tab:orange")
    axp.axvline(x_shock_snap, color="k", ls="--", lw=1)
    axp.set_title("Density & magnetic compression")
    axp.grid(alpha=0.3)

    # T_i, T_e (parallel) line-outs (phase grid), recomputed fresh at the snapshot
    # dump (the T_i streak may be on a coarser cadence).  Temperature is meaningless
    # in the upstream vacuum, so mask where n_e (interpolated onto the phase grid)
    # drops below 5% of its peak.  The masked profiles stay H5Data so ossemilogy can
    # label them; nan masks the vacuum.
    Ti_snap = temperature_frame(sim_dir, "al", t_snap, rqm_i, layout)
    Te_snap = temperature_frame(sim_dir, "e", t_snap, -1.0, layout)
    ne_on_p = np.interp(x_p, x_ne, ne_streak[i_snap])
    have_plasma = ne_on_p > 0.05 * np.nanmax(ne_streak[i_snap])
    Ti_plot = Ti_snap.copy(); Ti_plot[~have_plasma] = np.nan
    Te_plot = Te_snap.copy(); Te_plot[~have_plasma] = np.nan
    axT = ax2[1, 1]
    osh5vis.ossemilogy(Ti_plot, ax=axT, color="tab:red", label=r"$T_{i,\parallel}$ (Al)",
                       show_time=False, title="")
    osh5vis.ossemilogy(Te_plot, ax=axT, color="tab:blue", label=r"$T_{e,\parallel}$",
                       show_time=False, title="")
    axT.axvline(x_shock_snap, color="k", ls="--", lw=1)
    axT.set_ylabel(r"$T$ [$m_e c^2$]")
    axT.set_title("Temperature line-outs")
    axT.grid(alpha=0.3, which="both")
    axT.legend(fontsize=9)

    fig2.suptitle(f"Snapshot diagnostics — dump {t_snap} (t={time_f[i_snap]:.0f} $\\omega_{{pe}}^{{-1}}$)", fontsize=13)
    fig2.tight_layout()
    snap_path = os.path.join(out_dir, f"overview_snapshot_t{t_snap:06d}.png")
    fig2.savefig(snap_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved → {snap_path}")

    # ------------------------------------------------------------------
    # Save streak arrays + shock track
    # ------------------------------------------------------------------
    npz_path = os.path.join(out_dir, f"overview_{tag}.npz")
    np.savez(
        npz_path,
        field_dumps=np.asarray(field_dumps), phase_dumps=np.asarray(phase_dumps),
        time_field=time_f, time_phase=time_p, x_field=x_f, x_density=x_ne, x_phase=x_p,
        B_streak=B_streak, ne_streak=ne_streak, Ti_streak=Ti_streak,
        x_shock_detected=x_det,
        v_shock_cfg=np.asarray(v_shock_cfg), x_shock_0=np.asarray(x_shock_0),
        v_shock_fit=np.asarray(v_fit), x_shock_0_fit=np.asarray(x0_fit),
        v_A=np.asarray(v_A), M_A_cfg=np.asarray(M_A_cfg), M_A_fit=np.asarray(M_A_fit),
        ne_upstream=np.asarray(ne_upstream),
        t_gyro=np.asarray(t_gyro), B_upstream=np.asarray(B_upstream),
        config_path=np.asarray(os.path.abspath(args.config)),
    )
    print(f"Saved → {npz_path}")


if __name__ == "__main__":
    main()
