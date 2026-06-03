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
import yaml
from matplotlib.colors import LogNorm

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

import analysis_utils
import moments as mom_module
from analysis_utils import StreakBuilder

# Fields are dumped on even indices (savg cadence); phase spaces on every index.
_FIELD_TEMPLATE   = "{sd}/MS/FLD/{q}-savg/{q}-savg-{t:06d}.h5"
_DENSITY_TEMPLATE = "{sd}/MS/DENSITY/{sp}/charge-savg/charge-savg-{sp}-{t:06d}.h5"
_PHASE_TEMPLATE   = "{sd}/MS/PHA/{pha}/{sp}/{pha}-{sp}-{t:06d}.h5"


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["sim_dir"] = os.environ.get("MAGSHOCKZ_SIM_DIR", cfg["sim_dir"])
    if "times" in cfg:
        cfg["times"] = analysis_utils.parse_times(cfg["times"])
    return cfg


def axis_values(h5data, ax_idx: int) -> np.ndarray:
    ax = h5data.axes[ax_idx]
    return np.linspace(ax.min, ax.max, ax.size)


# ---------------------------------------------------------------------------
# Per-dump 1D profile frames (H5Data, so StreakBuilder can stack them)
# ---------------------------------------------------------------------------

def bmag_frame(sim_dir: str, t: int):
    """|B| = sqrt(b1^2 + b2^2 + b3^2) as an H5Data on the field grid [B_0]."""
    b = {q: osh5io.read_h5(_FIELD_TEMPLATE.format(sd=sim_dir, q=q, t=t))
         for q in ("b1", "b2", "b3")}
    bmag = np.sqrt(b["b1"] ** 2 + b["b2"] ** 2 + b["b3"] ** 2)  # stays H5Data
    bmag.data_attrs = dict(bmag.data_attrs, NAME="|B|", LONG_NAME=r"$|B|$", UNITS="B_0")
    return bmag


def density_frame(sim_dir: str, sp: str, t: int):
    """Number density n = |charge| as an H5Data on the field grid [n_0]."""
    ch = osh5io.read_h5(_DENSITY_TEMPLATE.format(sd=sim_dir, sp=sp, t=t))
    n = np.abs(ch)
    n.data_attrs = dict(n.data_attrs, NAME=f"n_{sp}", LONG_NAME=fr"$n_\mathrm{{{sp}}}$", UNITS="n_0")
    return n


def temperature_frame(sim_dir: str, sp: str, t: int, rqm: float, pha: str = "p1x1", axis: str = "p1"):
    """Parallel temperature T = |rqm| * <(p - <p>)^2> as an H5Data on the phase grid.

    The moment collapses f(p, x) -> T(x); the result is re-wrapped as an H5Data
    carrying the phase-space spatial axis and run_attrs (for the TIME StreakBuilder
    needs) so it can be stacked exactly like the field/density frames.
    """
    ps = osh5io.read_h5(_PHASE_TEMPLATE.format(sd=sim_dir, pha=pha, sp=sp, t=t))
    T = abs(rqm) * np.asarray(mom_module.moment(ps, order=2, axis=axis))
    x_axis = next(a for a in ps.axes if a.name != axis)  # the spatial axis
    return osh5def.H5Data(
        T,
        data_attrs={"NAME": f"T_{sp}", "LONG_NAME": fr"$T_{{{sp}}}$", "UNITS": "m_e c^2"},
        run_attrs=ps.run_attrs,
        axes=[x_axis],
    )


# ---------------------------------------------------------------------------
# Streak assembly + shock-front detection
# ---------------------------------------------------------------------------

def assemble_streak(frames):
    """StreakBuilder a list of 1D H5Data frames -> (Z[time, x], time[], x[])."""
    streak = StreakBuilder(frames).build()
    Z = np.asarray(streak)
    time = axis_values(streak, 0)
    x = axis_values(streak, 1)
    return Z, time, x


def detect_front(x, profile, x_pred, half_window, compression_min=1.3, edge_frac=0.5):
    """Detect the (upstream) leading edge of compression near x_pred.

    Returns the largest x within [x_pred - hw, x_pred + hw] at which `profile`
    exceeds baseline + edge_frac*(peak - baseline).  The shock moves toward +x
    with compressed plasma on the low-x side, so the leading edge is the
    upstream-most crossing.  Returns nan if the window holds no clear
    compression (peak/baseline < compression_min).
    """
    win = (x >= x_pred - half_window) & (x <= x_pred + half_window)
    if not win.any():
        return float("nan")
    xa, pa = x[win], profile[win]
    baseline = np.percentile(pa, 20)
    peak = np.percentile(pa, 99)
    if baseline <= 0 or peak / baseline < compression_min:
        return float("nan")
    thresh = baseline + edge_frac * (peak - baseline)
    above = xa[pa >= thresh]
    return float(above.max()) if above.size else float("nan")


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def plot_streak(ax, time, x, Z, *, label, cmap, log, shock_lines):
    """Render one streak (time horizontal, space vertical) with shock overlays.

    shock_lines : list of (time_arr, x_arr, style_kwargs, legend_label)
    """
    C = Z.T  # [x, time] for pcolormesh(X=time, Y=space)
    finite = C[np.isfinite(C)]
    if log:
        pos = finite[finite > 0]
        vmin = np.percentile(pos, 2) if pos.size else 1e-6
        vmax = np.percentile(pos, 99.5) if pos.size else 1.0
        norm = LogNorm(vmin=max(vmin, vmax * 1e-4), vmax=vmax)
        im = ax.pcolormesh(time, x, np.clip(C, max(vmin, vmax * 1e-4), None), cmap=cmap, norm=norm, shading="auto")
    else:
        vmax = np.percentile(finite, 99.5) if finite.size else 1.0
        im = ax.pcolormesh(time, x, C, cmap=cmap, vmin=0.0, vmax=vmax, shading="auto")
    cb = ax.figure.colorbar(im, ax=ax, pad=0.01)
    cb.set_label(label)
    for t_arr, x_arr, style, leg in shock_lines:
        ax.plot(t_arr, x_arr, label=leg, **style)
    ax.set_xlabel(r"$t$ [$\omega_{pe}^{-1}$]")
    ax.set_ylabel(r"$x$ [$c/\omega_{pe}$]")
    ax.set_ylim(x.min(), x.max())
    ax.set_title(label)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.7)


def plot_phasespace(ax, ps, *, title, x_shock=None):
    """imshow a p-x phase space (x horizontal, p vertical) on a log color scale.

    OSIRIS phase spaces are charge-weighted, so electron f is stored negative;
    take |f| so both species share one positive log color scale.
    """
    data = np.abs(np.asarray(ps))
    p_ax = next(a for a in ps.axes if a.name != "x1")
    x_ax = next(a for a in ps.axes if a.name == "x1")
    # axes order is [p, x]; data is f(p, x) so origin='lower' with extent below
    vmax = np.percentile(data[data > 0], 99.7) if (data > 0).any() else 1.0
    vmin = vmax * 1e-4
    im = ax.imshow(
        np.clip(data, vmin, None),
        origin="lower", aspect="auto",
        extent=[x_ax.min, x_ax.max, p_ax.min, p_ax.max],
        cmap="inferno", norm=LogNorm(vmin=vmin, vmax=vmax),
    )
    ax.figure.colorbar(im, ax=ax, pad=0.01, label=r"$f$ (arb.)")
    if x_shock is not None and np.isfinite(x_shock):
        ax.axvline(x_shock, color="cyan", ls="--", lw=1.2)
    ax.set_xlabel(r"$x$ [$c/\omega_{pe}$]")
    ax.set_ylabel(r"$p_1$ [$m_e c$]")
    ax.set_title(title)


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
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    sim_dir = cfg["sim_dir"]

    import astropy.units
    norm_density = float(cfg["norm_density_cm3"]) * astropy.units.cm**-3
    sim = analysis_utils.MagShockZRun(
        os.path.join(sim_dir, cfg.get("input_deck", "magshockz_gpu.1d")),
        norm_density=norm_density,
    )
    rqm_i = sim.rqm
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
            if not os.path.exists(_FIELD_TEMPLATE.format(sd=sim_dir, q="b3", t=t)):
                continue
            if not os.path.exists(_DENSITY_TEMPLATE.format(sd=sim_dir, sp="e", t=t)):
                continue
            if require_phase and not os.path.exists(_PHASE_TEMPLATE.format(sd=sim_dir, pha="p1x1", sp="al", t=t)):
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
    print(f"|B|,n_e : {len(field_dumps)} frames, {field_dumps[0]}..{field_dumps[-1]} (stride {field_dumps[1]-field_dumps[0]})")
    print(f"T_i     : {len(phase_dumps)} frames, {phase_dumps[0]}..{phase_dumps[-1]} (stride {phase_dumps[1]-phase_dumps[0]})")
    print(f"|rqm_i| : {abs(rqm_i):.4f}   v_shock(cfg) : {v_shock_cfg:.5f} c   x_shock_0 : {x_shock_0:.1f}")

    # ------------------------------------------------------------------
    # Load frames and assemble streaks
    # ------------------------------------------------------------------
    print("Loading |B|, n_e frames...")
    B_streak,  time_f, x_f = assemble_streak([bmag_frame(sim_dir, t) for t in field_dumps])
    ne_streak, _,      _   = assemble_streak([density_frame(sim_dir, "e", t) for t in field_dumps])
    print("Loading T_i frames (phase space)...")
    Ti_streak, time_p, x_p = assemble_streak([temperature_frame(sim_dir, "al", t, rqm_i) for t in phase_dumps])

    # ------------------------------------------------------------------
    # Per-frame shock-front detection (from the fine density streak) + fit
    # ------------------------------------------------------------------
    x_pred = x_shock_0 + v_shock_cfg * time_f
    x_det = np.array([
        detect_front(x_f, ne_streak[i], x_pred[i], args.search_hw)
        for i in range(len(field_dumps))
    ])
    good = np.isfinite(x_det)
    if good.sum() >= 2:
        slope, intercept = np.polyfit(time_f[good], x_det[good], 1)
        v_fit, x0_fit = float(slope), float(intercept)
    else:
        v_fit, x0_fit = float("nan"), float("nan")

    print("\n--- Shock front ---")
    print(f"  config fit   : v = {v_shock_cfg:.5f} c,  x0 = {x_shock_0:.1f} c/wpe")
    print(f"  detected fit : v = {v_fit:.5f} c,  x0 = {x0_fit:.1f} c/wpe  ({good.sum()}/{len(field_dumps)} frames)")

    cfg_line = (time_f, x_pred, dict(color="white", ls="-", lw=1.6), f"config fit (v={v_shock_cfg:.4f}c)")
    det_pts  = (time_f[good], x_det[good], dict(color="lime", ls="none", marker=".", ms=6), "detected front")
    fit_line = (time_f, x0_fit + v_fit * time_f, dict(color="lime", ls="--", lw=1.2), f"detected fit (v={v_fit:.4f}c)")

    out_dir = args.output_dir or os.path.join(_HERE, "..", "results", os.path.basename(sim_dir.rstrip("/")))
    os.makedirs(out_dir, exist_ok=True)
    tag = f"{field_dumps[0]:06d}-{field_dumps[-1]:06d}"

    # ------------------------------------------------------------------
    # Figure 1 — streaks
    # ------------------------------------------------------------------
    fig1, axes = plt.subplots(2, 2, figsize=(16, 10))
    plot_streak(axes[0, 0], time_f, x_f, B_streak,
                label=r"$|B|$ [$B_0$]", cmap="viridis", log=False,
                shock_lines=[cfg_line, det_pts, fit_line])
    plot_streak(axes[0, 1], time_f, x_f, ne_streak,
                label=r"$n_e$ [$n_0$]", cmap="magma", log=True,
                shock_lines=[cfg_line, det_pts, fit_line])
    plot_streak(axes[1, 0], time_p, x_p, Ti_streak,
                label=r"$T_{i,\parallel}$ [$m_e c^2$]", cmap="inferno", log=True,
                shock_lines=[cfg_line, det_pts, fit_line])

    axt = axes[1, 1]
    axt.plot(time_f, x_pred, color="tab:blue", lw=1.8, label=f"config fit (v={v_shock_cfg:.4f}c)")
    axt.plot(time_f[good], x_det[good], "o", color="tab:green", ms=5, label="detected front")
    if np.isfinite(v_fit):
        axt.plot(time_f, x0_fit + v_fit * time_f, "--", color="tab:green", lw=1.4,
                 label=f"detected fit (v={v_fit:.4f}c)")
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

    ps_al = osh5io.read_h5(_PHASE_TEMPLATE.format(sd=sim_dir, pha="p1x1", sp="al", t=t_snap))
    ps_e  = osh5io.read_h5(_PHASE_TEMPLATE.format(sd=sim_dir, pha="p1x1", sp="e",  t=t_snap))

    fig2, ax2 = plt.subplots(2, 2, figsize=(16, 10))
    plot_phasespace(ax2[0, 0], ps_al, title=f"Ion $p_1$-$x$ phase space (t={time_f[i_snap]:.0f})", x_shock=x_shock_snap)
    plot_phasespace(ax2[0, 1], ps_e,  title=f"Electron $p_1$-$x$ phase space (t={time_f[i_snap]:.0f})", x_shock=x_shock_snap)

    # n_e and |B| line-outs (field grid)
    axp = ax2[1, 0]
    axp.plot(x_f, ne_streak[i_snap], color="tab:purple", label=r"$n_e$ [$n_0$]")
    axp.set_xlabel(r"$x$ [$c/\omega_{pe}$]")
    axp.set_ylabel(r"$n_e$ [$n_0$]", color="tab:purple")
    axp.tick_params(axis="y", labelcolor="tab:purple")
    axb = axp.twinx()
    axb.plot(x_f, B_streak[i_snap], color="tab:orange", label=r"$|B|$ [$B_0$]")
    axb.set_ylabel(r"$|B|$ [$B_0$]", color="tab:orange")
    axb.tick_params(axis="y", labelcolor="tab:orange")
    axp.axvline(x_shock_snap, color="k", ls="--", lw=1)
    axp.set_title("Density & magnetic compression")
    axp.grid(alpha=0.3)

    # T_i, T_e (parallel) line-outs (phase grid), recomputed fresh at the snapshot
    # dump (the T_i streak may be on a coarser cadence).  Temperature is meaningless
    # in the upstream vacuum, so mask where n_e (interpolated onto the phase grid)
    # drops below 5% of its peak.
    Ti_snap = np.asarray(temperature_frame(sim_dir, "al", t_snap, rqm_i))
    Te_snap = np.asarray(temperature_frame(sim_dir, "e", t_snap, -1.0))
    ne_on_p = np.interp(x_p, x_f, ne_streak[i_snap])
    have_plasma = ne_on_p > 0.05 * np.nanmax(ne_streak[i_snap])
    Ti_plot = np.where(have_plasma, Ti_snap, np.nan)
    Te_plot = np.where(have_plasma, Te_snap, np.nan)
    axT = ax2[1, 1]
    axT.semilogy(x_p, Ti_plot, color="tab:red", label=r"$T_{i,\parallel}$ (Al)")
    axT.semilogy(x_p, Te_plot, color="tab:blue", label=r"$T_{e,\parallel}$")
    axT.axvline(x_shock_snap, color="k", ls="--", lw=1)
    axT.set_xlabel(r"$x$ [$c/\omega_{pe}$]")
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
        time_field=time_f, time_phase=time_p, x_field=x_f, x_phase=x_p,
        B_streak=B_streak, ne_streak=ne_streak, Ti_streak=Ti_streak,
        x_shock_detected=x_det,
        v_shock_cfg=np.asarray(v_shock_cfg), x_shock_0=np.asarray(x_shock_0),
        v_shock_fit=np.asarray(v_fit), x_shock_0_fit=np.asarray(x0_fit),
        config_path=np.asarray(os.path.abspath(args.config)),
    )
    print(f"Saved → {npz_path}")


if __name__ == "__main__":
    main()
