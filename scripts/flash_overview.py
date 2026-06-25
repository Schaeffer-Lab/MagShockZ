# -*- coding: utf-8 -*-
"""scripts/flash_overview.py — general overview of a MagShockZ FLASH shock run.

Produces two figures and a data archive that mirror the output of
scripts/overview.py for the OSIRIS run, but in physical units throughout:

  Figure 1 — time-space streak plots (time [ns] horizontal, distance [µm] vertical):
    nₑ     electron number density   [cm⁻³]
    |B|    magnetic-field magnitude  [Gauss]
    Tₑ     electron temperature      [eV]
    Tᵢ     ion temperature           [eV]
  The config-specified shock-front trajectory (flash: v_shock_est_cms,
  x_shock_0_cm) is overlaid on every streak.

  Figure 2 — representative snapshot at one dump:
    - yt SlicePlot of electron density (2D context for the 3D geometry)
    - nₑ + |B| line-out with shock marker (twin-y axes)
    - Tₑ and Tᵢ line-out (semilogy)
    - upstream Mach numbers (M_A, M_s) and v_shock annotated

Run parameters are read from two sources (never duplicated):
    run spec (run.yaml)  : data_path, line of sight, rqm_factor, reference_density
                           (RunSpec; falls back to run_manifest.yaml / runme*.sh)
    config YAML          : derived-from-data annotations (flash.v_shock_est, shock fit)

Usage
-----
    python scripts/flash_overview.py --config config/flash_3d_noshield.yaml \\
        [--stride 1] [--t-start 0] [--t-stop 20] \\
        [--snapshot-idx -1] [--search-halfwidth 5e-3] \\
        [--output-dir results/FLASH_3D_noshield]
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yt

yt.set_log_level(50)   # suppress yt chatter

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
sys.path.insert(0, os.path.join(_HERE, "..", "init_nopython"))

import unyt as u

import analysis_utils
import plot_style
import flash_utils as fu
import perpendicular_shock as ps


# ---------------------------------------------------------------------------
# Streak assembly
# ---------------------------------------------------------------------------

def assemble_streak(lineouts: list, field: str):
    """Stack a list of lineout dicts into a (Z[time, x], time_ns[], x_um[]) tuple.

    Handles differing spatial grid sizes by interpolating all lineouts onto a
    common grid (the first dump's grid, which covers the range of all dumps).
    """
    times = np.array([(lo["t_s"] * u.s).to("ns").value for lo in lineouts])

    # lineouts hold unyt arrays; np.asarray gives the plain CGS magnitude.
    xs = [np.asarray(lo["x"].to("cm")) for lo in lineouts]   # cm

    # Find the common spatial range and resolution
    x_min = min(x.min() for x in xs)
    x_max = max(x.max() for x in xs)
    # Use the first dump's resolution as the reference (average grid spacing)
    n_ref = len(xs[0])
    x_common = np.linspace(x_min, x_max, n_ref)  # Always monotonically increasing

    # Interpolate all lineouts onto the common grid
    Z_list = []
    for lo, x in zip(lineouts, xs):
        y_interp = np.interp(x_common, x, np.asarray(lo[field]))
        Z_list.append(y_interp)

    Z = np.stack(Z_list, axis=0)   # [n_dumps, n_x]
    x_um = (x_common * u.cm).to("um").value
    return Z, times, x_um


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def plot_streak(ax, time_ns, x_um, Z, *, label, cmap, log, shock_lines):
    """Render one streak (time ns horizontal, space µm vertical)."""
    from matplotlib.colors import LogNorm

    C = Z.T    # [x, time] for pcolormesh(time, x)
    finite = C[np.isfinite(C)]
    if log:
        pos  = finite[finite > 0]
        vmin = np.percentile(pos, 2) if pos.size else 1e-6
        vmax = np.percentile(pos, 99.5) if pos.size else 1.0
        vmin = max(vmin, vmax * 1e-4)
        norm = LogNorm(vmin=vmin, vmax=vmax)
        im   = ax.pcolormesh(time_ns, x_um, np.clip(C, vmin, None),
                             cmap=cmap, norm=norm, shading="auto")
    else:
        vmax = np.percentile(finite, 99.5) if finite.size else 1.0
        im   = ax.pcolormesh(time_ns, x_um, C, cmap=cmap,
                             vmin=0.0, vmax=vmax, shading="auto")
    cb = ax.figure.colorbar(im, ax=ax, pad=0.01)
    cb.set_label(label)
    for t_arr, x_arr, style, leg in shock_lines:
        ax.plot(t_arr, x_arr, label=leg, **style)
    ax.set_xlabel("$t$ [ns]")
    ax.set_ylabel(r"distance along LOS [$\mu$m]")
    ax.set_ylim(x_um.min(), x_um.max())
    ax.set_title(label)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.7)


def mass_continuity_vshock(lineouts, *, hand_fit=None, gap_cm=0.003, win_cm=0.012,
                           smooth=5, compression_min=1.3):
    """Per-dump shock speed from MASS-FLUX continuity (no trajectory fit).

    Region selection per dump:

    * **hand-fit (preferred)** — when ``hand_fit`` is given (a list aligned with
      ``lineouts``; each entry the dump's ``flash_dump_params`` dict or None), the
      downstream band is ``[x_downstream_start_cm, x_shock_cm]`` and the upstream
      is the ambient ahead of the front (``x > x_shock_cm``) — the SAME convention
      ``flash_rh_prediction.py`` uses.  Dumps whose hand-fit front is missing or
      degenerate (``x_shock <= x_downstream_start`` / ``x_shock <= 0``, i.e. "no
      shock formed yet") are left nan and contribute nothing.
    * **auto (fallback)** — only when ``hand_fit`` is None: locate the density
      front (steepest drop of n_e with increasing x) and average in ``win_cm``
      windows offset by ``gap_cm`` on each side to skip the transition layer.

    Region values are the MEDIAN of rho and v_para (robust to the spiky FLASH
    profiles).  Mass continuity is then solved for the lab-frame shock speed via
    :func:`perpendicular_shock.mass_flux_shock_speed`.

    Returns a dict of length-len(lineouts) arrays (nan where no clean,
    compressive front/regions are found): ``v_sh`` [cm/s], ``v_up`` / ``v_dn``
    [cm/s], ``rho_up`` / ``rho_dn`` [g/cm^3], ``x_front`` [cm].
    """
    n_d = len(lineouts)
    out = {k: np.full(n_d, np.nan) for k in
           ("v_sh", "v_up", "v_dn", "rho_up", "rho_dn", "x_front")}
    k = max(1, int(smooth))
    for i, lo in enumerate(lineouts):
        x   = np.asarray(lo["x"].to("cm"))
        ne  = np.asarray(lo["ne"].to("cm**-3"))
        rho = np.asarray(lo["rho"].to("g/cm**3"))
        v   = np.asarray(lo["v_para"].to("cm/s"))

        if hand_fit is not None:
            # Hand-fit regions only; skip dumps without a real, placed front.
            hf = hand_fit[i]
            if not hf:
                continue
            xs = float(hf.get("x_shock_cm", np.nan))
            xd = float(hf.get("x_downstream_start_cm", np.nan))
            if not (np.isfinite(xs) and np.isfinite(xd)) or xs <= 0 or xs <= xd:
                continue
            xf = xs
            up = x > xs                          # ambient ahead of the front
            dn = (x >= xd) & (x <= xs)           # hand-placed shocked band
        else:
            # Steepest density DROP with increasing x = the front (downstream is
            # the dense side at smaller x, upstream the ambient at larger x).
            # Smooth first and ignore the edge cells the kernel contaminates.
            ns = np.convolve(ne, np.ones(k) / k, mode="same")
            grad = np.gradient(ns, x)
            e = k if x.size > 2 * k else 0
            i_f = e + int(np.argmin(grad[e:x.size - e])) if e else int(np.argmin(grad))
            xf = x[i_f]
            up = (x >= xf + gap_cm) & (x <= xf + gap_cm + win_cm)   # ambient ahead
            dn = (x <= xf - gap_cm) & (x >= xf - gap_cm - win_cm)    # shocked behind

        if up.sum() < 3 or dn.sum() < 3:
            continue
        rho_u, rho_d = np.nanmedian(rho[up]), np.nanmedian(rho[dn])
        if not np.isfinite(rho_u) or rho_u <= 0 or rho_d / rho_u < compression_min:
            continue
        v_u, v_d = np.nanmedian(v[up]), np.nanmedian(v[dn])
        out["rho_up"][i], out["rho_dn"][i] = rho_u, rho_d
        out["v_up"][i], out["v_dn"][i] = v_u, v_d
        out["x_front"][i] = xf
        out["v_sh"][i] = ps.mass_flux_shock_speed(rho_u, v_u, rho_d, v_d)
    return out


def save_yt_slice(path, slice_axis, field, output_path, title, cmap):
    """Create a yt SlicePlot and save to file.  ``field`` is a (ftype, name) tuple."""
    ds = yt.load(path)
    slc = yt.SlicePlot(ds, slice_axis, field)
    slc.set_cmap(field, cmap)
    try:
        slc.annotate_timestamp(corner="upper_left")
    except TypeError:
        # Older yt versions may have different API
        pass
    slc.save(output_path)
    return output_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="General overview / streak plots of a FLASH MagShockZ run."
    )
    parser.add_argument("--config", required=True, help="Path to analysis YAML config.")
    parser.add_argument("--stride", type=int, default=1,
                        help="Dump stride for the streaks (default 1 = every available dump).")
    parser.add_argument("--t-start", type=int, default=0, dest="t_start",
                        help="First plot-file index to include (default 0).")
    parser.add_argument("--t-stop", type=int, default=None, dest="t_stop",
                        help="Last plot-file index (inclusive; default: all available).")
    parser.add_argument("--snapshot-idx", type=int, default=-1, dest="snapshot_idx",
                        help="Index into the dump list for the snapshot figure (default -1).")
    parser.add_argument("--slice-axis", default="z", dest="slice_axis",
                        choices=["x", "y", "z"],
                        help="Axis perpendicular to the 2D slice in the snapshot figure (default z).")
    parser.add_argument("--vshock-window-ns", type=float, nargs=2, default=(2.0, 5.0),
                        dest="vshock_window_ns", metavar=("LO", "HI"),
                        help="Time window [ns] over which the hand-fit shock front is "
                             "(believed) linear: used for the constant-velocity "
                             "trajectory fit and the windowed mass-continuity median "
                             "(default 2 5).")
    parser.add_argument("--output-dir", default=None, dest="output_dir")
    parser.add_argument("--nprocs", type=int, default=None,
                        help="Number of worker processes for loading dumps "
                             "(default: all cores on the node). Use 1 to load serially.")
    plot_style.add_publication_arg(parser)
    args = parser.parse_args()
    plot_style.apply(args.publication)

    # ------------------------------------------------------------------
    # Config + run parameters
    # ------------------------------------------------------------------
    cfg    = analysis_utils.load_config(args.config)
    spec   = analysis_utils.RunSpec.from_sim_dir(cfg["sim_dir"])

    data_path   = spec["data_path"]                       # e.g. .../plt_cnt_0009
    flash_dir   = str(os.path.dirname(data_path))
    file_prefix = os.path.basename(data_path)[:-4]        # strip 4-digit index
    flash_ic_index = int(os.path.basename(data_path)[-4:])

    line_start  = tuple(float(v) for v in spec["start_point"])
    line_end    = tuple(float(v) for v in spec["end_point"])
    rqm_factor  = float(spec.rqm_factor)
    ref_density = spec.reference_density                   # cm⁻³

    # Shock velocity and position estimates from config (used to seed detection)
    flash_cfg    = cfg.get("flash", {})
    v_shock_est  = float(flash_cfg.get("v_shock_est_cms", 0.0))
    x_shock_0_cm = float(flash_cfg.get("x_shock_0_cm", 0.0))

    all_files = fu.find_plot_files(flash_dir)
    # Restrict to requested range
    idx_range  = range(args.t_start, len(all_files) if args.t_stop is None
                       else min(args.t_stop + 1, len(all_files)), args.stride)
    loaded_indices = [i for i in idx_range if i < len(all_files)]   # plot-file index = config key
    dump_files = [all_files[i] for i in loaded_indices]

    if len(dump_files) < 2:
        raise RuntimeError(
            f"Need ≥2 dumps for streaks; found {len(dump_files)} "
            f"in range [{args.t_start}, {args.t_stop}] with stride {args.stride}."
        )

    out_dir = args.output_dir or os.path.join(
        _HERE, "..", "results",
        os.path.basename(flash_dir.rstrip("/"))
    )
    os.makedirs(out_dir, exist_ok=True)

    print(f"Config     : {args.config}")
    print(f"FLASH dir  : {flash_dir}")
    print(f"Line of sight: {line_start}  →  {line_end}  [cm]")
    print(f"n₀         : {ref_density:.2e} cm⁻³")
    print(f"rqm_factor : {rqm_factor}")
    print(f"v_shock_est: {(v_shock_est * u.cm / u.s).to('km/s').value:.1f} km/s   "
          f"x_shock_0: {(x_shock_0_cm * u.cm).to('um').value:.1f} µm  (from config flash:)")
    print(f"Dumps      : {len(dump_files)}  ({os.path.basename(dump_files[0])} … {os.path.basename(dump_files[-1])})")

    # ------------------------------------------------------------------
    # Load lineouts for every dump
    # ------------------------------------------------------------------
    # One worker per core by default (capped at the number of dumps).  SLURM
    # exposes the allocation via SLURM_CPUS_PER_TASK; fall back to os.cpu_count.
    if args.nprocs is not None:
        nprocs = max(1, args.nprocs)
    else:
        nprocs = int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) or os.cpu_count() or 1
    nprocs = min(nprocs, len(dump_files))

    print(f"\nLoading lineouts … ({nprocs} process{'es' if nprocs > 1 else ''})", flush=True)
    lineouts = fu.load_lineouts(dump_files, line_start, line_end, nprocs)

    # ------------------------------------------------------------------
    # Assemble streaks
    # ------------------------------------------------------------------
    ne_streak, time_ns, x_um = assemble_streak(lineouts, "ne")
    B_streak,  _,       _    = assemble_streak(lineouts, "B_mag")
    Te_streak, _,       _    = assemble_streak(lineouts, "Te")
    Ti_streak, _,       _    = assemble_streak(lineouts, "Ti")

    x_cm = (x_um * u.um).to("cm").value

    # ------------------------------------------------------------------
    # Shock-front detection + trajectory fit
    # ------------------------------------------------------------------
    t_s_arr = np.array([lo["t_s"] for lo in lineouts])

    # Read the IC dump time so the prediction is relative to it, not t=0.
    ic_file = all_files[flash_ic_index] if flash_ic_index < len(all_files) else dump_files[0]
    try:
        t_ic_s = fu.flash_time_s(ic_file)
    except Exception:
        t_ic_s = t_s_arr[0]

    if v_shock_est > 0.0 or x_shock_0_cm > 0.0:
        # x_pred = x_shock_0 + v_shock * (t - t_IC)
        x_pred_cm = x_shock_0_cm + v_shock_est * (t_s_arr - t_ic_s)
    else:
        # No estimate provided — seed from the midpoint; fit will correct it.
        x_pred_cm = np.full(len(lineouts), float(x_cm.mean()))

    # The plotted/recorded shock front is the config-specified trajectory
    # (flash: v_shock_est_cms, x_shock_0_cm), NOT an edge track.  With so few
    # dumps the steepest-gradient detector locks onto the fast leading edge,
    # which is biased high relative to the mass-flux frame; the hand-set config
    # front is the more trustworthy reference here.  x_pred_cm above is the
    # config line x_shock_0_cm + v_shock_est*(t - t_IC).
    v_shock_fit   = v_shock_est
    x_shock_0_fit = float(x_shock_0_cm - v_shock_est * t_ic_s)   # config front at t=0
    x_det_cm      = x_pred_cm.copy()      # front at each dump = the config line

    # Shock velocity in km/s for display
    v_shock_kms = (v_shock_fit * u.cm / u.s).to("km/s").value

    # ------------------------------------------------------------------
    # Mach numbers (from upstream average at the snapshot dump)
    # ------------------------------------------------------------------
    snap_idx   = args.snapshot_idx % len(lineouts)
    snap_lo    = lineouts[snap_idx]
    x_shock_snap = x_det_cm[snap_idx] if np.isfinite(x_det_cm[snap_idx]) else float(x_pred_cm[snap_idx])
    upstream   = np.asarray(snap_lo["x"]) > x_shock_snap

    def _up_mean(arr):
        return float(np.nanmean(arr[upstream])) if upstream.any() else float("nan")

    up_ne   = _up_mean(snap_lo["ne"])
    up_nion = _up_mean(snap_lo["n_ion"])
    up_Te   = _up_mean(snap_lo["Te"])
    up_Ti   = _up_mean(snap_lo["Ti"])
    up_B    = _up_mean(snap_lo["B_mag"])
    up_rho  = _up_mean(snap_lo["rho"])

    v_shock_for_mach = v_shock_fit if np.isfinite(v_shock_fit) else v_shock_est
    mach = fu.mach_numbers(up_ne, up_nion, up_Te, up_Ti, up_B, up_rho, v_shock_for_mach)

    print("\n--- Shock front (from config flash:) ---")
    print(f"  v_shock = {v_shock_kms:.2f} km/s   "
          f"x₀(t=0) = {(x_shock_0_fit * u.cm).to('um').value:.2f} µm")
    print("\n--- Upstream Mach numbers ---")
    print(f"  n₀  = {up_ne:.3e} cm⁻³   B  = {up_B:.2f} G   "
          f"Tₑ = {up_Te:.1f} eV   Tᵢ = {up_Ti:.1f} eV")
    print(f"  v_A = {mach['v_A'].to('km/s').value:.2f} km/s   "
          f"c_s = {mach['c_s'].to('km/s').value:.2f} km/s   beta = {float(mach['beta']):.2f}")
    print(f"  M_A = {float(mach['M_A']):.2f}   M_s = {float(mach['M_s']):.2f}")

    # ------------------------------------------------------------------
    # Per-dump shock speed from mass-flux continuity, using the HAND-FIT shock
    # fronts (flash_dump_params) to delimit upstream/downstream where available;
    # region values are medians.  Independent of any trajectory fit, so it reveals
    # whether the front is accelerating.
    # ------------------------------------------------------------------
    hand_fit = [cfg.get("flash_dump_params", {}).get(i) for i in loaded_indices]
    use_hand_fit = any(hf for hf in hand_fit)
    mc = mass_continuity_vshock(
        lineouts, hand_fit=hand_fit if use_hand_fit else None,
        gap_cm=0.003, win_cm=0.012)  # auto-mode windows: 30/120 µm
    mc_finite = np.isfinite(mc["v_sh"])
    src = "hand-fit regions, median" if use_hand_fit else "auto front, median"
    print(f"\n--- Mass-continuity shock speed (per dump; {src}) ---")
    if mc_finite.sum():
        vsh_kms_all = (mc["v_sh"][mc_finite] * u.cm / u.s).to("km/s").value
        print(f"  v_sh = {np.median(vsh_kms_all):.1f} km/s (median)  "
              f"(range {vsh_kms_all.min():.0f}–{vsh_kms_all.max():.0f}, "
              f"{mc_finite.sum()}/{len(lineouts)} dumps)")
    else:
        print("  no clean compressive front found in any dump")

    # Constant-velocity trajectory fit from the hand-fit fronts over the window
    # where the front is (believed) linear — an independent v_shock estimate.
    lo_ns, hi_ns = args.vshock_window_ns
    xf_hand_cm = np.array([
        float(hf["x_shock_cm"]) if (hf and np.isfinite(float(hf.get("x_shock_cm", np.nan)))
                                    and float(hf["x_shock_cm"]) > 0) else np.nan
        for hf in hand_fit
    ])
    in_win = np.isfinite(xf_hand_cm) & (time_ns >= lo_ns) & (time_ns <= hi_ns)
    v_traj_cms, x0_traj_cm = np.nan, np.nan
    if in_win.sum() >= 2:
        # x_shock(t) = v_traj * t + x0   (t in s, x in cm) -> slope is v_shock.
        slope, intercept = np.polyfit(t_s_arr[in_win], xf_hand_cm[in_win], 1)
        v_traj_cms, x0_traj_cm = float(slope), float(intercept)

    mc_in_win = mc_finite & (time_ns >= lo_ns) & (time_ns <= hi_ns)
    mc_med_win_cms = (np.nanmedian(mc["v_sh"][mc_in_win])
                      if mc_in_win.sum() else np.nan)

    print(f"\n--- Shock speed over the {lo_ns:.1f}–{hi_ns:.1f} ns window ---")
    if np.isfinite(v_traj_cms):
        print(f"  hand-fit trajectory   v_shock = {(v_traj_cms * u.cm / u.s).to('km/s').value:.1f} km/s   "
              f"x0(t=0) = {(x0_traj_cm * u.cm).to('um').value:.1f} µm   ({in_win.sum()} fronts)")
    else:
        print("  hand-fit trajectory   : <2 fronts in window, no linear fit")
    if np.isfinite(mc_med_win_cms):
        print(f"  mass-continuity median v_shock = {(mc_med_win_cms * u.cm / u.s).to('km/s').value:.1f} km/s   "
              f"({mc_in_win.sum()} dumps)")

    # Shock front line (config trajectory) in µm units for the streak plots.
    x_um_pred   = (x_pred_cm * u.cm).to("um").value
    config_line = (time_ns, x_um_pred,
                   dict(color="white", ls="-", lw=1.6),
                   f"config  v={v_shock_kms:.1f} km/s  $M_A$={float(mach['M_A']):.2f}")
    shock_lines = [config_line]

    tag = f"{os.path.basename(dump_files[0])}_{os.path.basename(dump_files[-1])}"

    # ------------------------------------------------------------------
    # Figure 1 — streaks
    # ------------------------------------------------------------------
    fig1, axes = plt.subplots(2, 2, figsize=(16, 10))
    plot_streak(axes[0, 0], time_ns, x_um, ne_streak,
                label=r"$n_e$ [cm$^{-3}$]", cmap="magma", log=True,
                shock_lines=shock_lines)
    plot_streak(axes[0, 1], time_ns, x_um, B_streak,
                label=r"$|B|$ [G]", cmap="viridis", log=False,
                shock_lines=shock_lines)
    plot_streak(axes[1, 0], time_ns, x_um, Te_streak,
                label=r"$T_e$ [eV]", cmap="inferno", log=True,
                shock_lines=shock_lines)
    plot_streak(axes[1, 1], time_ns, x_um, Ti_streak,
                label=r"$T_i$ [eV]", cmap="inferno", log=True,
                shock_lines=shock_lines)

    fig1.suptitle(
        f"FLASH shock overview — {os.path.basename(flash_dir)}  "
        f"({len(dump_files)} dumps)\n"
        f"v_shock = {v_shock_kms:.1f} km/s   M_A = {float(mach['M_A']):.2f}   M_s = {float(mach['M_s']):.2f}",
        fontsize=12,
    )
    fig1.tight_layout()
    streak_path = os.path.join(out_dir, f"flash_overview_streaks_{tag}.png")
    fig1.savefig(streak_path, dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"\nSaved → {streak_path}")

    # ------------------------------------------------------------------
    # Figure 1b — mass-continuity shock speed vs time (acceleration check)
    # ------------------------------------------------------------------
    figv, (axv, axin) = plt.subplots(1, 2, figsize=(15, 5.5))
    t_fin   = time_ns[mc_finite]
    vsh_fin = (mc["v_sh"][mc_finite] * u.cm / u.s).to("km/s").value
    v_shock_est_kms = (v_shock_est * u.cm / u.s).to("km/s").value

    axv.axvspan(lo_ns, hi_ns, color="0.88", zorder=0,
                label=f"fit window {lo_ns:.0f}–{hi_ns:.0f} ns")
    axv.plot(t_fin, vsh_fin, "o-", color="tab:red", label=r"mass-continuity $v_{sh}$")
    axv.axhline(v_shock_est_kms, color="0.5", ls="--",
                label=f"config $v_{{sh}}$ = {v_shock_est_kms:.0f} km/s")
    if np.isfinite(v_traj_cms):
        v_traj_kms = (v_traj_cms * u.cm / u.s).to("km/s").value
        axv.axhline(v_traj_kms, color="tab:blue", ls="-", lw=1.8,
                    label=f"hand-fit trajectory = {v_traj_kms:.0f} km/s")
    if np.isfinite(mc_med_win_cms):
        mc_med_win_kms = (mc_med_win_cms * u.cm / u.s).to("km/s").value
        axv.axhline(mc_med_win_kms, color="tab:red", ls=":", lw=1.8,
                    label=f"mass-cont. median = {mc_med_win_kms:.0f} km/s")
    accel_note = ""
    if mc_in_win.sum() >= 2:                     # is the front constant-velocity?
        tw, vw = time_ns[mc_in_win], (mc["v_sh"][mc_in_win] * u.cm / u.s).to("km/s").value
        b, a = np.polyfit(tw, vw, 1)             # v_sh = a + b·t  [km/s, ns]
        accel_note = f"   window slope = {b:+.1f} km/s/ns ({'accel' if b > 0 else 'decel'}.)"
    axv.set_xlabel("$t$ [ns]")
    axv.set_ylabel(r"$v_{sh}$ [km/s]")
    axv.set_title("Shock speed from mass continuity" + accel_note)
    axv.grid(alpha=0.3)
    axv.legend(fontsize=9)

    # Inputs panel: the region-averaged velocities and compression that feed v_sh.
    axin.plot(t_fin, (mc["v_up"][mc_finite] * u.cm / u.s).to("km/s").value,
              "o-", color="tab:blue",  label=r"$v_{up}$")
    axin.plot(t_fin, (mc["v_dn"][mc_finite] * u.cm / u.s).to("km/s").value,
              "o-", color="tab:green", label=r"$v_{dn}$")
    axin.set_xlabel("$t$ [ns]")
    axin.set_ylabel(r"$v$ [km/s]")
    axin.grid(alpha=0.3)
    axr = axin.twinx()
    r_arr = mc["rho_dn"][mc_finite] / mc["rho_up"][mc_finite]
    axr.plot(t_fin, r_arr, "s--", color="tab:purple", label=r"$r=\rho_{dn}/\rho_{up}$")
    axr.set_ylabel(r"compression $r$")
    h1, l1 = axin.get_legend_handles_labels()
    h2, l2 = axr.get_legend_handles_labels()
    axin.legend(h1 + h2, l1 + l2, fontsize=9, loc="best")
    axin.set_title("Mass-continuity inputs (region averages)")

    figv.suptitle(f"Mass-continuity shock speed — {os.path.basename(flash_dir)}", fontsize=12)
    figv.tight_layout()
    vsh_path = os.path.join(out_dir, f"flash_overview_vshock_masscontinuity_{tag}.png")
    figv.savefig(vsh_path, dpi=150, bbox_inches="tight")
    plt.close(figv)
    print(f"Saved → {vsh_path}")

    # ------------------------------------------------------------------
    # Figure 2 — snapshot (lineouts + yt slices saved separately)
    # ------------------------------------------------------------------
    snap_file = dump_files[snap_idx]
    snap_lo   = lineouts[snap_idx]
    snap_t_ns = float((snap_lo["t_s"] * u.s).to("ns").value)
    x_um_snap = snap_lo["x"].to("um").value
    x_shock_um = (x_shock_snap * u.cm).to("um").value

    # Save yt slice plots separately
    print(f"\nCreating yt slice plots …", flush=True)
    try:
        slice_edens_path = os.path.join(
            out_dir, f"flash_slice_edens_{os.path.basename(snap_file)}.png"
        )
        save_yt_slice(snap_file, args.slice_axis, ("gas", "El_number_density"),
                      slice_edens_path,
                      f"Electron density — {os.path.basename(snap_file)}", "viridis")
        print(f"Saved → {slice_edens_path}")
    except Exception as e:
        print(f"  Warning: could not save density slice: {e}")

    try:
        slice_tion_path = os.path.join(
            out_dir, f"flash_slice_tion_{os.path.basename(snap_file)}.png"
        )
        save_yt_slice(snap_file, args.slice_axis, ("flash", "tion"), slice_tion_path,
                      f"Ion temperature — {os.path.basename(snap_file)}", "hot")
        print(f"Saved → {slice_tion_path}")
    except Exception as e:
        print(f"  Warning: could not save Ti slice: {e}")

    # Lineout figure (2 panels)
    fig2, ax2 = plt.subplots(1, 2, figsize=(14, 5))

    # Panel [0]: nₑ + |B| line-out
    axp = ax2[0]
    axp.plot(x_um_snap, np.asarray(snap_lo["ne"]), color="tab:purple", label=r"$n_e$ [cm$^{-3}$]", lw=2)
    axp.set_ylabel(r"$n_e$ [cm$^{-3}$]", color="tab:purple", fontsize=11)
    axp.tick_params(axis="y", labelcolor="tab:purple")
    axb = axp.twinx()
    axb.plot(x_um_snap, np.asarray(snap_lo["B_mag"]), color="tab:orange", label=r"$|B|$ [G]", lw=2)
    axb.set_ylabel(r"$|B|$ [G]", color="tab:orange", fontsize=11)
    axb.tick_params(axis="y", labelcolor="tab:orange")
    axp.axvline(x_shock_um, color="k", ls="--", lw=1.2, alpha=0.7)
    axp.set_xlabel(r"distance along LOS [$\mu$m]", fontsize=11)
    axp.set_title("Density & magnetic compression", fontsize=12)
    axp.grid(alpha=0.3)

    # Panel [1]: Tₑ and Tᵢ line-outs
    axT = ax2[1]
    axT.semilogy(x_um_snap, np.asarray(snap_lo["Te"]), color="tab:blue",  label=r"$T_e$ [eV]", lw=2)
    axT.semilogy(x_um_snap, np.asarray(snap_lo["Ti"]), color="tab:red",   label=r"$T_i$ [eV]", lw=2)
    axT.axvline(x_shock_um, color="k", ls="--", lw=1.2, alpha=0.7)
    axT.set_xlabel(r"distance along LOS [$\mu$m]", fontsize=11)
    axT.set_ylabel("Temperature [eV]", fontsize=11)
    axT.set_title("Electron and ion temperatures", fontsize=12)
    axT.grid(alpha=0.3, which="both")
    axT.legend(fontsize=10, loc="best")
    # Annotate Mach numbers
    mach_txt = (
        f"$v_{{shock}}$ = {v_shock_kms:.1f} km/s\n"
        f"$M_A$ = {float(mach['M_A']):.2f}    $M_s$ = {float(mach['M_s']):.2f}\n"
        f"$\\beta$ = {float(mach['beta']):.2f}"
    )
    axT.text(0.02, 0.97, mach_txt, transform=axT.transAxes,
             va="top", fontsize=10, bbox=dict(boxstyle="round", alpha=0.3))

    fig2.suptitle(
        f"FLASH lineouts — {os.path.basename(snap_file)}  (t = {snap_t_ns:.2f} ns)",
        fontsize=13,
    )
    fig2.tight_layout()
    lineout_path = os.path.join(out_dir, f"flash_overview_lineouts_{os.path.basename(snap_file)}.png")
    fig2.savefig(lineout_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved → {lineout_path}")

    # ------------------------------------------------------------------
    # Save arrays
    # ------------------------------------------------------------------
    npz_path = os.path.join(out_dir, f"flash_overview_{tag}.npz")
    np.savez(
        npz_path,
        dump_files  = np.asarray([os.path.basename(f) for f in dump_files]),
        time_ns     = time_ns,
        x_um        = x_um,
        ne_streak   = ne_streak,
        B_streak    = B_streak,
        Te_streak   = Te_streak,
        Ti_streak   = Ti_streak,
        x_shock_det_cm  = x_det_cm,
        v_shock_cms     = np.asarray(v_shock_fit),
        v_shock_kms     = np.asarray(v_shock_kms),
        x_shock_0_cm    = np.asarray(x_shock_0_fit),
        # per-dump mass-flux-continuity shock speed and its region-average inputs
        mc_vsh_cms      = mc["v_sh"],
        mc_v_up_cms     = mc["v_up"],
        mc_v_dn_cms     = mc["v_dn"],
        mc_rho_up       = mc["rho_up"],
        mc_rho_dn       = mc["rho_dn"],
        mc_x_front_cm   = mc["x_front"],
        # hand-fit trajectory + windowed mass-continuity shock speed
        vshock_window_ns        = np.asarray([lo_ns, hi_ns]),
        v_traj_cms              = np.asarray(v_traj_cms),
        x0_traj_cm              = np.asarray(x0_traj_cm),
        mc_vsh_median_window_cms = np.asarray(mc_med_win_cms),
        up_ne       = np.asarray(up_ne),
        up_Te       = np.asarray(up_Te),
        up_Ti       = np.asarray(up_Ti),
        up_B        = np.asarray(up_B),
        up_rho      = np.asarray(up_rho),
        M_A         = np.asarray(float(mach["M_A"])),
        M_s         = np.asarray(float(mach["M_s"])),
        v_A_kms     = np.asarray(mach["v_A"].to("km/s").value),
        c_s_kms     = np.asarray(mach["c_s"].to("km/s").value),
        beta        = np.asarray(float(mach["beta"])),
        config_path = np.asarray(os.path.abspath(args.config)),
    )
    print(f"Saved → {npz_path}")


if __name__ == "__main__":
    main()
