"""scripts/overview.py — general overview of a MagShockZ OSIRIS shock run.

Time-space "streak" plots (time on the horizontal axis, space on the vertical)
for the quantities that track a collisionless shock front:

    |B|   magnetic compression        [B_0]
    n_e   electron number density     [n_0]
    T_i   ion temperature (parallel)  [m_e c^2]

The config's linear shock-front trajectory (x_shock = x_shock_0 + v_shock * t_sim,
tuned with scripts/tune_shock.py) is overlaid on every streak so the chosen
velocity can be eyeballed against the compression ridge directly.  On top of that
line, each bespoke per-dump position from cfg["dump_params"] (the x_shock /
x_downstream_start hand-tuned in tune_shock.py regions mode) is marked at its own
simulation time, so the linear fit can be checked against the per-dump fronts.

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
import osh5io
import osh5vis
from matplotlib.colors import LogNorm

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

import analysis_utils
import plot_style
import osh5def
from analysis_utils import axis_values, diag_path
# Streak frame-builders + assembly are single-sourced in src/streak.py (shared with
# scripts/tune_shock.py), so the front overlay is tuned against identical data.
from streak import bmag_frame, density_frame, temperature_frame, assemble_streak

ME_C2_EV = 510998.95  # electron rest energy [eV]; T[eV] = T[m_e c^2] * ME_C2_EV


# ---------------------------------------------------------------------------
# Plot helpers
#
# The per-dump frame builders (bmag_frame / density_frame / temperature_frame)
# and assemble_streak now live in src/streak.py, shared with tune_shock.py.
# ---------------------------------------------------------------------------

def _to_display(data, disp, *, rescale_time=False):
    """Return an H5Data sharing ``data``'s array but with length (and optionally the
    ``time``) axis coordinates rescaled into the active display unit (plot_style.DisplayUnits).

    A no-op in electron units.  Shares the underlying ndarray (no data copy — important
    for the ~260 MB phase spaces); only the cheap DataAxis objects are rebuilt.  Axis
    *labels* are set separately by the caller, since the on-disk UNITS no longer describe
    the rescaled coordinates.
    """
    if disp.units == "electron":
        return data
    new_axes = []
    for ax in data.axes:
        if ax.units.is_length():
            new_axes.append(osh5def.DataAxis(float(disp.x(ax.min)), float(disp.x(ax.max)),
                                             ax.size, attrs=dict(ax.attrs)))
        elif rescale_time and ax.name == "time":
            new_axes.append(osh5def.DataAxis(float(disp.t(ax.min)), float(disp.t(ax.max)),
                                             ax.size, attrs=dict(ax.attrs)))
        else:
            new_axes.append(ax)
    return osh5def.H5Data(np.asarray(data), data_attrs=dict(data.data_attrs),
                          run_attrs=data.run_attrs, axes=new_axes)


def plot_streak(ax, streak, x, *, title, cmap, log, shock_lines, disp):
    """Render one streak (time horizontal, space vertical) with shock overlays.

    ``streak`` is the [time, x] H5Data; it is transposed to [x, time] and drawn
    with osh5vis.osimshow, which sets the time/space axis labels and the colorbar
    *unit* from the data's own ``data_attrs`` (so the displayed unit always matches
    the data, never a hardcoded guess).  ``title`` is just the quantity name; the
    percentile colour limits and shock-line overlays are layered on top.  ``disp``
    rescales both axes + the overlays into the active display unit (electron = identity).

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
    streak = _to_display(streak, disp, rescale_time=True)
    osh5vis.osimshow(streak.transpose(), ax=ax, cmap=cmap, title=title, **kw)
    if disp.units != "electron":   # rescaled coords no longer match the on-disk UNITS
        ax.set_xlabel(disp.tlabel()); ax.set_ylabel(disp.xlabel())
    for t_arr, x_arr, style, leg in shock_lines:
        ax.plot(disp.t(t_arr), disp.x(x_arr), label=leg, **style)
    ax.set_ylim(float(disp.x(x.min())), float(disp.x(x.max())))
    ax.legend(fontsize=8, loc="upper left", framealpha=0.7)


def plot_phasespace(ax, ps, *, title, x_shock=None, disp):
    """osimshow a p-x phase space (x horizontal, p vertical) on a log color scale.

    OSIRIS phase spaces are charge-weighted, so electron f is stored negative;
    take |f| so both species share one positive log color scale.  osh5io stores
    the data as [p, x] (in both 1D and the 2D run), so osh5vis.osimshow places the
    spatial axis horizontal and momentum vertical and labels both from the data's
    own metadata — no explicit normal-axis needed.  ``disp`` rescales the (length)
    spatial axis into the display unit; the momentum axis is left native.
    """
    data = _to_display(np.abs(ps), disp)               # H5Data, metadata preserved
    arr = np.asarray(data)
    vmax = np.percentile(arr[arr > 0], 99.7) if (arr > 0).any() else 1.0
    vmin = vmax * 1e-4
    osh5vis.osimshow(data, ax=ax, cmap="inferno",
                     norm=LogNorm(vmin=vmin, vmax=vmax), title=title,
                     cblabel=r"$f$ (arb.)")
    if disp.units != "electron":
        ax.set_xlabel(disp.xlabel())
    if x_shock is not None and np.isfinite(x_shock):
        ax.axvline(float(disp.x(x_shock)), color="cyan", ls="--", lw=1.2)


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
    parser.add_argument("--transverse-halfwidth", type=float, default=5.0, dest="transverse_hw",
                        help="Half-width [c/wpe] of the central transverse band averaged when "
                             "reducing 2D field/density maps to a 1D shock-normal profile "
                             "(default 5 = 5 electron inertial lengths; ignored for 1D runs).")
    parser.add_argument("--output-dir", default=None)
    plot_style.add_publication_arg(parser)
    plot_style.add_units_arg(parser)
    args = parser.parse_args()
    plot_style.apply(args.publication)

    cfg = analysis_utils.load_config(args.config)
    sim_dir = cfg["sim_dir"]
    disp = plot_style.build_units(args.units, cfg=cfg, config_path=os.path.abspath(args.config))

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
        b3 = layout.field_quantity("b3")
        b3_dir = f"{sim_dir}/MS/FLD/{b3}"
        idxs = [int(f.split("-")[-1].split(".")[0]) for f in os.listdir(b3_dir) if f.endswith(".h5")]
        t_stop = max(idxs)

    def dump_list(stride, require_phase):
        stride = stride + (stride % 2)  # keep even so field savg files exist
        out = []
        for t in range(args.t_start, t_stop + 1, stride):
            if not os.path.exists(diag_path(sim_dir, layout.field_quantity("b3"), t)):
                continue
            if not os.path.exists(diag_path(sim_dir, layout.charge_quantity, t, "e")):
                continue
            if require_phase and not os.path.exists(diag_path(sim_dir, layout.pha_name(1), t, "al")):
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
    # Shock-front trajectory from the tuned config (no auto-detection).  The
    # front is x_shock(t) = x_shock_0 + v_shock * t with the values set in the
    # config via scripts/tune_shock.py; that single tuned line is overlaid on the
    # streaks so it can be eyeballed against the compression ridge directly.
    # ------------------------------------------------------------------
    x_config = x_shock_0 + v_shock_cfg * time_f

    # Upstream Alfven speed v_A/c from the t=0 frame, so v_shock can be quoted as
    # an Alfvenic Mach number.  In normalised units the field equals omega_ce
    # [1/wpe], so omega_ci = |B|/|rqm_i|, omega_pi = sqrt(n_e/|rqm_i|), hence
    #     v_A/c = omega_ci/omega_pi = |B| / sqrt(|rqm_i| * n_e).
    upstream0 = x_f > x_shock_0
    B_upstream = float(np.median(B_streak[0][upstream0])) if upstream0.any() \
        else float(np.median(B_streak[0]))
    upstream_ne = x_ne > x_shock_0
    ne_upstream = float(np.median(ne_streak[0][upstream_ne])) if upstream_ne.any() \
        else float(np.median(ne_streak[0]))
    v_A = B_upstream / np.sqrt(abs(rqm_i) * ne_upstream)  # [c]
    M_A_cfg = v_shock_cfg / v_A

    print("\n--- Shock front (tuned config) ---")
    print(f"  upstream : |B| = {B_upstream:.4g} [m_e c wpe/e],  n_e = {ne_upstream:.4g} n0,  v_A = {v_A:.5f} c")
    print(f"  config   : v = {v_shock_cfg:.5f} c = {M_A_cfg:.2f} v_A,  x0 = {x_shock_0:.1f} c/wpe")

    cfg_line = (time_f, x_config, dict(color="white", ls="-", lw=1.8),
                f"config (v={v_shock_cfg:.4f}c, $M_A$={M_A_cfg:.2f})")

    # ------------------------------------------------------------------
    # Per-dump tuned markers.  cfg["dump_params"].<idx> holds the hand-fit front
    # (x_shock) and downstream-region edge (x_downstream_start) set with
    # scripts/tune_shock.py (regions mode) for specific dumps.  OSIRIS dump time
    # is linear in the dump index, so map index -> t_sim from the loaded field
    # frames and place each marker at its own simulation time (data coords, so it
    # overlays correctly on any panel regardless of that panel's grid).  Dumps
    # outside the plotted [t_start, t_stop] window are skipped.
    # ------------------------------------------------------------------
    dump_params = cfg.get("dump_params", {}) or {}
    t_per_idx = (time_f[-1] - time_f[0]) / (field_dumps[-1] - field_dumps[0])
    t0_idx = time_f[0] - t_per_idx * field_dumps[0]
    xs_t, xs_x, xd_t, xd_x = [], [], [], []
    down_ext = []   # downstream extents (x_shock - x_downstream_start) for window trimming
    for idx in sorted(dump_params):
        t_sim_idx = t0_idx + t_per_idx * int(idx)
        if not (time_f[0] <= t_sim_idx <= time_f[-1]):
            continue
        per = dump_params[idx] or {}
        if per.get("x_shock") is not None:
            xs_t.append(t_sim_idx); xs_x.append(float(per["x_shock"]))
        if per.get("x_downstream_start") is not None:
            xd_t.append(t_sim_idx); xd_x.append(float(per["x_downstream_start"]))
        if per.get("x_downstream_start") is not None:
            front = float(per["x_shock"]) if per.get("x_shock") is not None \
                else x_shock_0 + v_shock_cfg * t_sim_idx
            down_ext.append(front - float(per["x_downstream_start"]))
    # Half-width [c/wpe] of the shock window: the typical downstream extent from the
    # tuned config.  Every panel's spatial axis is trimmed to front +- this extent
    # (downstream + front + equal-width upstream) instead of the whole box.
    region_ext = float(np.median(down_ext)) if down_ext else None
    dump_lines = []
    if xs_t:
        dump_lines.append((np.asarray(xs_t), np.asarray(xs_x),
                           dict(color="orange", ls="none", marker="D", ms=8, mec="k", mew=0.6),
                           r"tuned $x_\mathrm{shock}$ (per dump)"))
    if xd_t:
        dump_lines.append((np.asarray(xd_t), np.asarray(xd_x),
                           dict(color="deepskyblue", ls="none", marker="v", ms=7, mec="k", mew=0.6),
                           r"tuned $x_\mathrm{down}$ (per dump)"))
    streak_lines = [cfg_line] + dump_lines
    if xs_t:
        print(f"  per-dump x_shock markers @ t_sim = "
              f"{', '.join(f'{t:.0f}' for t in xs_t)}")

    out_dir = args.output_dir or os.path.join(_HERE, "..", "results", os.path.basename(sim_dir.rstrip("/")))
    os.makedirs(out_dir, exist_ok=True)
    tag = f"{field_dumps[0]:06d}-{field_dumps[-1]:06d}"

    # ------------------------------------------------------------------
    # Figure 1 — streaks
    # ------------------------------------------------------------------
    fig1, axes = plt.subplots(2, 2, figsize=(16, 10))
    plot_streak(axes[0, 0], B_h5, x_f,
                title=r"$|B|$", cmap="viridis", log=False, shock_lines=streak_lines, disp=disp)
    plot_streak(axes[0, 1], ne_h5, x_ne,
                title=r"$n_e$", cmap="magma", log=True, shock_lines=streak_lines, disp=disp)
    plot_streak(axes[1, 0], Ti_h5, x_p,
                title=r"$T_{i,\parallel}$", cmap="inferno", log=True, shock_lines=streak_lines, disp=disp)

    axt = axes[1, 1]
    axt.plot(disp.t(time_f), disp.x(x_config), color="tab:blue", lw=1.8,
             label=f"config (v={v_shock_cfg:.4f}c, $M_A$={M_A_cfg:.2f})")
    if xs_t:
        axt.plot(disp.t(xs_t), disp.x(xs_x), color="orange", ls="none", marker="D", ms=8, mec="k",
                 mew=0.6, label=r"tuned $x_\mathrm{shock}$ (per dump)")
    if xd_t:
        axt.plot(disp.t(xd_t), disp.x(xd_x), color="deepskyblue", ls="none", marker="v", ms=7, mec="k",
                 mew=0.6, label=r"tuned $x_\mathrm{down}$ (per dump)")
    axt.set_xlabel(disp.tlabel())
    axt.set_ylabel(disp.xlabel(r"x_\mathrm{shock}"))
    axt.set_title("Shock-front trajectory (tuned config)")
    axt.grid(alpha=0.3)
    axt.legend(fontsize=8)

    # Trim the spatial axis of every streak (and the trajectory) to the band the
    # front sweeps through plus the tuned downstream/upstream extent, so the plots
    # zoom onto the shock region instead of the whole box.
    if region_ext is not None:
        x_all = np.concatenate([x_f, x_ne, x_p])
        ylo = max(float(x_all.min()), float(x_config.min()) - region_ext)
        yhi = min(float(x_all.max()), float(x_config.max()) + region_ext)
        for ax in (axes[0, 0], axes[0, 1], axes[1, 0], axt):
            ax.set_ylim(float(disp.x(ylo)), float(disp.x(yhi)))

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
    x_shock_snap = x_config[i_snap]

    ps_al = osh5io.read_h5(diag_path(sim_dir, layout.pha_name(1), t_snap, "al"))
    ps_e  = osh5io.read_h5(diag_path(sim_dir, layout.pha_name(1), t_snap, "e"))

    t_snap_title = disp.time_title(time_f[i_snap])
    fig2, ax2 = plt.subplots(2, 2, figsize=(16, 10))
    plot_phasespace(ax2[0, 0], ps_al, title=f"Ion $p_1$-$x$ phase space ({t_snap_title})",
                    x_shock=x_shock_snap, disp=disp)
    plot_phasespace(ax2[0, 1], ps_e,  title=f"Electron $p_1$-$x$ phase space ({t_snap_title})",
                    x_shock=x_shock_snap, disp=disp)

    # n_e and |B| line-outs, each normalised to its own upstream value so the two
    # compressions share one dimensionless axis (ratio -> 1 far upstream of the shock,
    # jumping to the compression ratio behind it).  The upstream value is the median
    # over the upstream window (x beyond the snapshot's front), matching how every
    # other upstream average in the codebase is taken.
    axp = ax2[1, 0]
    up_ne = x_ne > x_shock_snap
    ne_up = float(np.median(ne_streak[i_snap][up_ne])) if up_ne.any() \
        else float(np.median(ne_streak[i_snap]))
    up_B = x_f > x_shock_snap
    B_up = float(np.median(B_streak[i_snap][up_B])) if up_B.any() \
        else float(np.median(B_streak[i_snap]))
    ne_ratio = ne_streak[i_snap] / ne_up
    B_ratio = B_streak[i_snap] / B_up
    osh5vis.osplot1d(_to_display(ne_h5[i_snap] / ne_up, disp), ax=axp, color="tab:purple",
                     show_time=False, title="", label=r"$n_e / n_{e,\mathrm{up}}$")
    osh5vis.osplot1d(_to_display(B_h5[i_snap] / B_up, disp), ax=axp, color="tab:orange",
                     show_time=False, title="", label=r"$|B| / |B|_\mathrm{up}$")
    axp.axvline(float(disp.x(x_shock_snap)), color="k", ls="--", lw=1)
    # Mean compression ratio over the compressed downstream region
    # ([x_shock - region_ext, x_shock]), drawn as a horizontal line so the jump can
    # be read off as a single number per quantity.
    dn_lo = (x_shock_snap - region_ext) if region_ext is not None else float(min(x_ne.min(), x_f.min()))
    dn_ne_mask = (x_ne < x_shock_snap) & (x_ne >= dn_lo)
    dn_B_mask = (x_f < x_shock_snap) & (x_f >= dn_lo)
    ne_comp = float(np.mean(ne_ratio[dn_ne_mask])) if dn_ne_mask.any() else np.nan
    B_comp = float(np.mean(B_ratio[dn_B_mask])) if dn_B_mask.any() else np.nan
    if np.isfinite(ne_comp):
        axp.axhline(ne_comp, color="tab:purple", ls=":", lw=1.5,
                    label=fr"$\langle n_e/n_{{e,\mathrm{{up}}}}\rangle = {ne_comp:.2f}$")
    if np.isfinite(B_comp):
        axp.axhline(B_comp, color="tab:orange", ls=":", lw=1.5,
                    label=fr"$\langle |B|/|B|_\mathrm{{up}}\rangle = {B_comp:.2f}$")
    if disp.units != "electron":
        axp.set_xlabel(disp.xlabel())
    axp.set_ylabel("compression ratio")
    axp.set_title("Density & magnetic compression")
    axp.grid(alpha=0.3)
    axp.legend(fontsize=9)
    # Tighten ymax to the data actually shown (the compression maxes out well below
    # the default), with a little headroom.
    vis = np.concatenate([ne_ratio[(x_ne >= dn_lo)], B_ratio[(x_f >= dn_lo)]])
    vis = vis[np.isfinite(vis)]
    if vis.size:
        axp.set_ylim(0.0, 1.1 * float(vis.max()))

    # T_i, T_e (parallel) line-outs (phase grid), recomputed fresh at the snapshot
    # dump (the T_i streak may be on a coarser cadence).  Temperature is meaningless
    # in the upstream vacuum, so mask where n_e (interpolated onto the phase grid)
    # drops below 5% of its peak.  The masked profiles stay H5Data so ossemilogy can
    # label them; nan masks the vacuum.
    # Temperature is stored in m_e c^2; convert to eV (T[eV] = T[m_e c^2] * ME_C2_EV)
    # and relabel UNITS so the axis reads eV.
    Ti_snap = temperature_frame(sim_dir, "al", t_snap, rqm_i, layout) * ME_C2_EV
    Te_snap = temperature_frame(sim_dir, "e", t_snap, -1.0, layout) * ME_C2_EV
    Ti_snap.data_attrs = dict(Ti_snap.data_attrs, UNITS="eV")
    Te_snap.data_attrs = dict(Te_snap.data_attrs, UNITS="eV")
    ne_on_p = np.interp(x_p, x_ne, ne_streak[i_snap])
    have_plasma = ne_on_p > 0.05 * np.nanmax(ne_streak[i_snap])
    Ti_plot = Ti_snap.copy(); Ti_plot[~have_plasma] = np.nan
    Te_plot = Te_snap.copy(); Te_plot[~have_plasma] = np.nan
    axT = ax2[1, 1]
    osh5vis.ossemilogy(_to_display(Ti_plot, disp), ax=axT, color="tab:red", label=r"$T_{i,\parallel}$ (Al)",
                       show_time=False, title="")
    osh5vis.ossemilogy(_to_display(Te_plot, disp), ax=axT, color="tab:blue", label=r"$T_{e,\parallel}$",
                       show_time=False, title="")
    axT.axvline(float(disp.x(x_shock_snap)), color="k", ls="--", lw=1)
    if disp.units != "electron":
        axT.set_xlabel(disp.xlabel())
    axT.set_ylabel(r"$T$ [eV]")
    axT.set_title("Temperature line-outs")
    axT.grid(alpha=0.3, which="both")
    axT.legend(fontsize=9)

    # Trim every snapshot panel's x-axis to front +- the tuned downstream extent
    # (downstream + front + equal-width upstream), matching the streak trimming.
    if region_ext is not None and np.isfinite(x_shock_snap):
        x_all = np.concatenate([x_f, x_ne, x_p])
        xlo = max(float(x_all.min()), float(x_shock_snap) - region_ext)
        xhi = min(float(x_all.max()), float(x_shock_snap) + region_ext)
        for ax in (ax2[0, 0], ax2[0, 1], ax2[1, 0], axT):
            ax.set_xlim(float(disp.x(xlo)), float(disp.x(xhi)))

    fig2.suptitle(f"Snapshot diagnostics — dump {t_snap} ({disp.time_title(time_f[i_snap])})", fontsize=13)
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
        x_shock_config=x_config,
        dump_xshock_t=np.asarray(xs_t), dump_xshock=np.asarray(xs_x),
        dump_xdown_t=np.asarray(xd_t), dump_xdown=np.asarray(xd_x),
        v_shock_cfg=np.asarray(v_shock_cfg), x_shock_0=np.asarray(x_shock_0),
        v_A=np.asarray(v_A), M_A_cfg=np.asarray(M_A_cfg),
        ne_upstream=np.asarray(ne_upstream),
        B_upstream=np.asarray(B_upstream),
        config_path=np.asarray(os.path.abspath(args.config)),
    )
    print(f"Saved → {npz_path}")


if __name__ == "__main__":
    main()
