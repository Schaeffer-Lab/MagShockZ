# -*- coding: utf-8 -*-
"""scripts/flash_overview.py — general overview of a MagShockZ FLASH shock run.

Produces two figures and a data archive that mirror the output of
scripts/overview.py for the OSIRIS run, but in physical units throughout:

  Figure 1 — time-space streak plots (time [ns] horizontal, distance [µm] vertical):
    nₑ     electron number density   [cm⁻³]
    |B|    magnetic-field magnitude  [Gauss]
    Tₑ     electron temperature      [eV]
    Tᵢ     ion temperature           [eV]
  The predicted and detected shock-front trajectories are overlaid on every streak.

  Figure 2 — representative snapshot at one dump:
    - yt SlicePlot of electron density (2D context for the 3D geometry)
    - nₑ + |B| line-out with shock marker (twin-y axes)
    - Tₑ and Tᵢ line-out (semilogy)
    - upstream Mach numbers (M_A, M_s) and v_shock annotated

Run parameters are read from two sources (never duplicated):
    runme_perlmutter.sh  : line of sight, rqm_factor, B0_Gauss, data_path
    config YAML          : norm_density_cm3, and optionally flash.v_shock_est

Usage
-----
    python scripts/flash_overview.py --config config/perlmutter_1.3.1d.yaml \\
        [--stride 1] [--t-start 0] [--t-stop 20] \\
        [--snapshot-idx -1] [--search-halfwidth 5e-3] \\
        [--output-dir results/FLASH_3D_noshield]
"""

import argparse
import functools
import os
import sys
from multiprocessing import Pool

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yt

yt.set_log_level(50)   # suppress yt chatter

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
sys.path.insert(0, os.path.join(_HERE, "..", "init_nopython"))

import analysis_utils
import flash_utils as fu

# Unit conversion factors
_CM_TO_UM  = 1e4      # cm → µm
_S_TO_NS   = 1e9      # s  → ns
_CM_TO_KMS = 1e-5     # cm/s → km/s


# ---------------------------------------------------------------------------
# Parallel lineout loading
# ---------------------------------------------------------------------------

def _load_one(path, line_start, line_end):
    """Module-level worker so multiprocessing can pickle it.  Each call loads
    one FLASH dump and extracts its lineout — fully independent of every other
    dump, so the dumps fan out across worker processes."""
    return fu.flash_lineout(path, line_start, line_end)


def _robust_linfit(t, x, n_iter=3, n_sigma=2.5):
    """Linear fit x = slope·t + intercept with iterative σ-clipping.

    A few bad per-frame detections would otherwise drag the trajectory fit (and
    hence the predicted window) off the real front, so we drop points more than
    n_sigma residual-σ from the line and refit.  At least 3 points are always
    kept; if clipping would drop below that the previous fit is retained.
    Returns (slope, intercept)."""
    slope, intercept = np.polyfit(t, x, 1)
    keep = np.ones(len(t), dtype=bool)
    for _ in range(n_iter):
        resid = x - (slope * t + intercept)
        sigma = np.std(resid[keep])
        if sigma == 0:
            break
        new_keep = np.abs(resid) <= n_sigma * sigma
        if new_keep.sum() < 3 or np.array_equal(new_keep, keep):
            break
        keep = new_keep
        slope, intercept = np.polyfit(t[keep], x[keep], 1)
    return float(slope), float(intercept)


# ---------------------------------------------------------------------------
# Streak assembly
# ---------------------------------------------------------------------------

def assemble_streak(lineouts: list, field: str):
    """Stack a list of lineout dicts into a (Z[time, x], time_ns[], x_um[]) tuple.

    Handles differing spatial grid sizes by interpolating all lineouts onto a
    common grid (the first dump's grid, which covers the range of all dumps).
    """
    times = np.array([lo["t_s"] * _S_TO_NS for lo in lineouts])

    # lineouts hold unyt arrays; np.asarray gives the plain CGS magnitude.
    xs = [np.asarray(lo["x"]) for lo in lineouts]   # cm

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
    x_um = x_common * _CM_TO_UM
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
    parser.add_argument("--search-halfwidth", type=float, default=0.005,
                        dest="search_hw",
                        help="Half-width [cm] of the shock-detection window "
                             "around the predicted front (default 0.005 cm = 50 µm).")
    parser.add_argument("--slice-axis", default="z", dest="slice_axis",
                        choices=["x", "y", "z"],
                        help="Axis perpendicular to the 2D slice in the snapshot figure (default z).")
    parser.add_argument("--output-dir", default=None, dest="output_dir")
    parser.add_argument("--nprocs", type=int, default=None,
                        help="Number of worker processes for loading dumps "
                             "(default: all cores on the node). Use 1 to load serially.")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Config + run parameters
    # ------------------------------------------------------------------
    cfg    = analysis_utils.load_config(args.config)
    runme  = analysis_utils.load_runme(cfg["sim_dir"])

    data_path   = runme["data_path"]                      # e.g. .../plt_cnt_0009
    flash_dir   = str(os.path.dirname(data_path))
    file_prefix = os.path.basename(data_path)[:-4]        # strip 4-digit index
    flash_ic_index = int(os.path.basename(data_path)[-4:])

    line_start  = tuple(float(v) for v in runme["start_point"])
    line_end    = tuple(float(v) for v in runme["end_point"])
    rqm_factor  = float(runme["rqm_factor"])
    ref_density = float(cfg["norm_density_cm3"])           # cm⁻³

    # Shock velocity and position estimates from config (used to seed detection)
    flash_cfg    = cfg.get("flash", {})
    v_shock_est  = float(flash_cfg.get("v_shock_est_cms", 0.0))
    x_shock_0_cm = float(flash_cfg.get("x_shock_0_cm", 0.0))

    all_files = fu.find_plot_files(flash_dir)
    # Restrict to requested range
    idx_range  = range(args.t_start, len(all_files) if args.t_stop is None
                       else min(args.t_stop + 1, len(all_files)), args.stride)
    dump_files = [all_files[i] for i in idx_range if i < len(all_files)]

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
    print(f"v_shock_est: {v_shock_est * _CM_TO_KMS:.1f} km/s   x_shock_0: {x_shock_0_cm * _CM_TO_UM:.1f} µm  (from config flash:)")
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
    worker = functools.partial(
        _load_one,
        line_start=line_start, line_end=line_end,
    )
    if nprocs == 1:
        lineouts = []
        for i, path in enumerate(dump_files):
            print(f"  [{i+1:3d}/{len(dump_files)}]  {os.path.basename(path)}", flush=True)
            lineouts.append(worker(path))
    else:
        # imap preserves input order, so lineouts[i] still matches dump_files[i].
        with Pool(nprocs) as pool:
            lineouts = []
            for i, lo in enumerate(pool.imap(worker, dump_files)):
                print(f"  [{i+1:3d}/{len(dump_files)}]  {os.path.basename(dump_files[i])}", flush=True)
                lineouts.append(lo)

    # ------------------------------------------------------------------
    # Assemble streaks
    # ------------------------------------------------------------------
    ne_streak, time_ns, x_um = assemble_streak(lineouts, "ne")
    B_streak,  _,       _    = assemble_streak(lineouts, "B_mag")
    Te_streak, _,       _    = assemble_streak(lineouts, "Te")
    Ti_streak, _,       _    = assemble_streak(lineouts, "Ti")

    x_cm = x_um / _CM_TO_UM

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

    # Iterative detect ↔ fit.  Start with a wide search window so a poor seed
    # still catches the front, then narrow as the fitted trajectory homes in.
    # Each pass uses an outlier-rejecting fit so a few stray detections don't
    # bias the prediction for the next pass.
    v_shock_fit   = float("nan")
    x_shock_0_fit = float("nan")
    x_det_cm = np.full(len(lineouts), float("nan"))
    detected = np.zeros(len(lineouts), dtype=bool)
    for hw in (3.0 * args.search_hw, 1.5 * args.search_hw, args.search_hw):
        x_det_cm = np.array([
            fu.detect_front(x_cm, ne_streak[i], x_pred_cm[i], hw)
            for i in range(len(lineouts))
        ])
        detected = np.isfinite(x_det_cm)
        if detected.sum() < 2:
            break
        v_shock_fit, x_shock_0_fit = _robust_linfit(
            t_s_arr[detected], x_det_cm[detected])
        x_pred_cm = x_shock_0_fit + v_shock_fit * t_s_arr  # predict for next pass

    # Shock velocity in km/s for display
    v_shock_kms = v_shock_fit * _CM_TO_KMS if np.isfinite(v_shock_fit) else float("nan")

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

    print("\n--- Shock front ---")
    print(f"  detected fit : v_shock = {v_shock_kms:.2f} km/s  "
          f"x₀ = {x_shock_0_fit * _CM_TO_UM:.2f} µm  "
          f"({detected.sum()}/{len(lineouts)} frames)")
    print("\n--- Upstream Mach numbers ---")
    print(f"  n₀  = {up_ne:.3e} cm⁻³   B  = {up_B:.2f} G   "
          f"Tₑ = {up_Te:.1f} eV   Tᵢ = {up_Ti:.1f} eV")
    print(f"  v_A = {mach['v_A'].to('km/s').value:.2f} km/s   "
          f"c_s = {mach['c_s'].to('km/s').value:.2f} km/s   beta = {float(mach['beta']):.2f}")
    print(f"  M_A = {float(mach['M_A']):.2f}   M_s = {float(mach['M_s']):.2f}")

    # Shock lines in µm units for streak plots
    t_ns_det  = time_ns[detected]
    x_um_det  = x_det_cm[detected] * _CM_TO_UM
    x_um_pred = x_pred_cm * _CM_TO_UM
    fit_line  = (time_ns, x_um_pred,
                 dict(color="white", ls="-", lw=1.6),
                 f"fit  v={v_shock_kms:.1f} km/s")
    det_pts   = (t_ns_det, x_um_det,
                 dict(color="lime", ls="none", marker=".", ms=6),
                 "detected front")
    shock_lines = [fit_line, det_pts]

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
    # Figure 2 — snapshot (lineouts + yt slices saved separately)
    # ------------------------------------------------------------------
    snap_file = dump_files[snap_idx]
    snap_lo   = lineouts[snap_idx]
    snap_t_ns = float(snap_lo["t_s"] * _S_TO_NS)
    x_um_snap = np.asarray(snap_lo["x"]) * _CM_TO_UM
    x_shock_um = x_shock_snap * _CM_TO_UM

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
