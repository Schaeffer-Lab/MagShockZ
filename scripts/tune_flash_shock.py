# -*- coding: utf-8 -*-
"""scripts/tune_flash_shock.py — interactively place a FLASH run's shock front.

The FLASH analog of ``scripts/tune_shock.py``.  The FLASH overview's automatic
steepest-gradient edge tracker is unreliable with so few dumps (it locks onto the
fast leading edge, biased high relative to the mass-flux frame), so this tool lets
you place the front by hand on the SAME physical-unit lineouts the analysis draws,
see the overlay move immediately, and (on confirm) write the value back into the
config with every comment preserved.

The downstream analysis reads the front from the config:

    flash:             v_shock_est_cms, x_shock_0_cm     (the linear front trajectory)
    flash_dump_params: <idx>: x_shock_cm, x_downstream_start_cm   (per-dump region edges)

``flash_dump_params`` is a separate top-level section (physical CGS, keyed by the
plot-file index) so its cm-unit positions never collide with the OSIRIS c/ωpe
``dump_params``.  ``scripts/flash_rh_prediction.py`` reads them straight back.

Display is a PNG you refresh in your editor (robust over SSH, no X11): each command
re-renders ``results/<run>/tune_flash_*.png`` and prints its path.  Distances are in
µm and times in ns throughout (the config stores cm / cm·s⁻¹).

Two modes
---------
trajectory (default) — tune ``flash.v_shock_est_cms`` / ``flash.x_shock_0_cm``
    against the nₑ and |B| streaks.  Commands:
        v <val>     set trial v_shock [km/s]
        x <val>     set trial x_shock_0 [µm]  (front position at the IC dump time)
        save        write v_shock_est_cms + x_shock_0_cm to the config (asks y/N)
        q           quit

regions — tune one dump's ``x_shock_cm`` / ``x_downstream_start_cm`` against its
    nₑ/|B|/Tₑ/Tᵢ line-outs AND a 2D density slice through the LOS (so the front can
    be placed against the actual shock geometry, not just the 1D trace).  The slice
    shares the LOS-distance axis with the line-outs, so the shock/downstream marker
    lines fall directly over the 2D density jump.  Pick the dump with --snapshot-idx.
    Commands:
        shock <x>   set trial shock-front position [µm]
        down <x>    set trial downstream-region left edge [µm]
        save        write flash_dump_params.<idx> to the config (asks y/N)
        q           quit

Env: analysis (yt / unyt).  Examples:
    python scripts/tune_flash_shock.py --config config/flash_3d_noshield.yaml
    python scripts/tune_flash_shock.py --config ...yaml --mode regions --snapshot-idx -1
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
from matplotlib.colors import LogNorm

yt.set_log_level(50)   # suppress yt chatter

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

import analysis_utils
import plot_style
import flash_utils as fu
import yaml_edit
# Reuse the overview's streak assembly + pickle-able dump loader so the streak the
# tuner draws is byte-for-byte the one the analysis produces.
from flash_overview import assemble_streak, _load_one

_CM_TO_UM  = 1e4      # cm → µm
_UM_TO_CM  = 1e-4     # µm → cm
_S_TO_NS   = 1e9      # s  → ns
_CM_TO_KMS = 1e-5     # cm/s → km/s
_KMS_TO_CMS = 1e5     # km/s → cm/s


# ---------------------------------------------------------------------------
# Shared helpers  (mirror scripts/tune_shock.py)
# ---------------------------------------------------------------------------

def _out_dir(cfg, override, flash_dir):
    out = override or os.path.join(_HERE, "..", "results",
                                   os.path.basename(flash_dir.rstrip("/")))
    os.makedirs(out, exist_ok=True)
    return out


def _ask_yes(prompt):
    try:
        return input(prompt).strip().lower() in ("y", "yes")
    except EOFError:
        return False


def _normalize(val):
    """Match yaml_edit's compact rendering when verifying the round-trip."""
    if isinstance(val, float) and val.is_integer():
        return int(val)
    return val


def _aligned_diff(old, new):
    n = max(len(old), len(new))
    old = old + [""] * (n - len(old))
    new = new + [""] * (n - len(new))
    return list(zip(old, new))


def _confirm_write(config_path, edits, no_write):
    """Apply each (dotted_path, value) edit to the YAML text, show the diff, ask y/N.

    A path under ``flash_dump_params`` routes to ``set_dump_param`` (custom section);
    everything else is a plain ``set_scalar``.  Each edit is verified to round-trip
    before the file is touched.  Returns True if written.
    """
    if no_write:
        print("  [--no-write] would write:")
        for path, val in edits:
            print(f"    {path} = {val}")
        return False

    with open(config_path) as f:
        text = f.read()
    new_text = text
    for path, val in edits:
        parts = path.split(".")
        if parts[0] == "flash_dump_params":
            new_text = yaml_edit.set_dump_param(
                new_text, int(parts[1]), parts[2], val, section="flash_dump_params")
        else:
            new_text = yaml_edit.set_scalar(new_text, path, val)
        yaml_edit.assert_roundtrip(new_text, path, _normalize(val))

    print("  pending edits:")
    for a, b in _aligned_diff(text.split("\n"), new_text.split("\n")):
        if a != b:
            print(f"    - {a}")
            print(f"    + {b}")
    if not _ask_yes(f"  write these to {os.path.basename(config_path)}? [y/N] "):
        print("  not written.")
        return False
    with open(config_path, "w") as f:
        f.write(new_text)
    print(f"  wrote → {config_path}")
    return True


def _run_paths(cfg):
    """Resolve (flash_dir, sorted plot files, LOS endpoints, IC index/time) from the spec."""
    spec       = analysis_utils.RunSpec.from_sim_dir(cfg["sim_dir"])
    data_path  = spec["data_path"]
    flash_dir  = str(os.path.dirname(data_path))
    ic_index   = int(os.path.basename(data_path)[-4:])
    line_start = tuple(float(v) for v in spec["start_point"])
    line_end   = tuple(float(v) for v in spec["end_point"])
    all_files  = fu.find_plot_files(flash_dir)
    try:
        t_ic_s = fu.flash_time_s(all_files[ic_index] if ic_index < len(all_files)
                                 else all_files[0])
    except Exception:
        t_ic_s = 0.0
    return flash_dir, all_files, line_start, line_end, ic_index, t_ic_s


def _load_lineouts(paths, line_start, line_end, nprocs):
    """Load a list of dumps' lineouts, in input order, optionally in parallel."""
    worker = functools.partial(_load_one, line_start=line_start, line_end=line_end)
    if nprocs <= 1:
        out = []
        for i, p in enumerate(paths):
            print(f"  [{i + 1:3d}/{len(paths)}] {os.path.basename(p)}", flush=True)
            out.append(worker(p))
        return out
    with Pool(nprocs) as pool:
        out = []
        for i, lo in enumerate(pool.imap(worker, paths)):
            print(f"  [{i + 1:3d}/{len(paths)}] {os.path.basename(paths[i])}", flush=True)
            out.append(lo)
        return out


# ---------------------------------------------------------------------------
# Trajectory mode
# ---------------------------------------------------------------------------

class TrajectoryTuner:
    """nₑ/|B| streaks with a movable straight front trajectory (config flash:)."""

    def __init__(self, cfg, args):
        self.cfg = cfg
        (self.flash_dir, all_files, line_start, line_end,
         ic_index, self.t_ic_s) = _run_paths(cfg)
        self.out_dir = _out_dir(cfg, args.output_dir, self.flash_dir)
        self.png = os.path.join(self.out_dir, "tune_flash_trajectory.png")

        idx_range = range(args.t_start,
                          len(all_files) if args.t_stop is None
                          else min(args.t_stop + 1, len(all_files)),
                          args.stride)
        self.loaded_indices = [i for i in idx_range if i < len(all_files)]
        paths = [all_files[i] for i in self.loaded_indices]
        if len(paths) < 2:
            raise RuntimeError(f"Need ≥2 dumps for a streak; got {len(paths)}.")

        nprocs = args.nprocs or int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) \
            or os.cpu_count() or 1
        nprocs = min(max(1, nprocs), len(paths))
        print(f"Loading nₑ, |B| over {len(paths)} dumps "
              f"({nprocs} process{'es' if nprocs > 1 else ''}) …", flush=True)
        lineouts = _load_lineouts(paths, line_start, line_end, nprocs)

        self.ne_streak, self.time_ns, self.x_um = assemble_streak(lineouts, "ne")
        self.B_streak,  _,            _         = assemble_streak(lineouts, "B_mag")
        self.t_s = self.time_ns / _S_TO_NS

        # Trial trajectory seeded from the config flash: block.
        flash = cfg.get("flash", {})
        self.v_cms  = float(flash.get("v_shock_est_cms", 0.0))
        self.x0_cm  = float(flash.get("x_shock_0_cm", float(self.x_um.mean()) * _UM_TO_CM))

    def _front_um(self):
        # x(t) = x_shock_0 + v_shock·(t − t_IC), exactly as flash_overview plots it.
        return (self.x0_cm + self.v_cms * (self.t_s - self.t_ic_s)) * _CM_TO_UM

    def _fit_fronts(self):
        """Per-dump hand-fit shock fronts (config flash_dump_params) mapped onto the
        streak's time axis.  Returns (t_ns, x_um) for the loaded dumps that carry an
        ``x_shock_cm``.  Entries where the front sits on (or below) the downstream
        edge are degenerate "no shock yet" placements — kept, so the gap before a
        shock forms is visible rather than silently dropped."""
        per = self.cfg.get("flash_dump_params", {})
        idx_to_t = dict(zip(self.loaded_indices, self.time_ns))
        t, x = [], []
        for idx, p in sorted(per.items()):
            if idx not in idx_to_t or "x_shock_cm" not in p:
                continue
            t.append(idx_to_t[idx])
            x.append(float(p["x_shock_cm"]) * _CM_TO_UM)
        return np.array(t), np.array(x)

    def render(self):
        fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
        x_line = self._front_um()
        t_fit, x_fit = self._fit_fronts()
        leg = (f"trial  v={self.v_cms * _CM_TO_KMS:.1f} km/s  "
               f"x₀={self.x0_cm * _CM_TO_UM:.1f} µm")
        self._panel(axes[0], self.ne_streak, self.x_um, r"$n_e$ [cm$^{-3}$]",
                    "magma", True, x_line, leg, t_fit, x_fit)
        self._panel(axes[1], self.B_streak, self.x_um, r"$|B|$ [G]",
                    "viridis", False, x_line, leg, t_fit, x_fit)
        axes[1].set_xlabel("$t$ [ns]")
        fig.suptitle(f"FLASH trajectory tuning — {os.path.basename(self.flash_dir)}\n"
                     f"trial v_shock={self.v_cms * _CM_TO_KMS:.1f} km/s, "
                     f"x_shock_0={self.x0_cm * _CM_TO_UM:.1f} µm", fontsize=12)
        fig.tight_layout()
        fig.savefig(self.png, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  v_shock = {self.v_cms * _CM_TO_KMS:.1f} km/s   "
              f"x_shock_0 = {self.x0_cm * _CM_TO_UM:.2f} µm ({self.x0_cm:.4g} cm)")
        print(f"  ↻ wrote {self.png} — refresh in your IDE")

    def _panel(self, ax, streak, x_um, label, cmap, log, x_line_um, leg,
               t_fit=None, x_fit=None):
        C = streak.T   # [x, time] for pcolormesh(time, x)
        finite = C[np.isfinite(C)]
        if log:
            pos = finite[finite > 0]
            vmax = np.percentile(pos, 99.5) if pos.size else 1.0
            vmin = max(np.percentile(pos, 2) if pos.size else 1e-6, vmax * 1e-4)
            im = ax.pcolormesh(self.time_ns, x_um, np.clip(C, vmin, None),
                               cmap=cmap, norm=LogNorm(vmin=vmin, vmax=vmax),
                               shading="auto")
        else:
            vmax = np.percentile(finite, 99.5) if finite.size else 1.0
            im = ax.pcolormesh(self.time_ns, x_um, C, cmap=cmap,
                               vmin=0.0, vmax=vmax, shading="auto")
        cb = ax.figure.colorbar(im, ax=ax, pad=0.01)
        cb.set_label(label)
        ax.plot(self.time_ns, x_line_um, color="white", ls="-", lw=2.0, label=leg)
        if t_fit is not None and t_fit.size:
            ax.scatter(t_fit, x_fit, s=55, marker="o",
                       facecolor="cyan", edgecolor="k", linewidths=1.0,
                       zorder=5, label="hand-fit front")
        ax.set_ylabel(r"distance along LOS [$\mu$m]")
        ax.set_ylim(x_um.min(), x_um.max())
        ax.set_title(label)
        ax.legend(fontsize=8, loc="upper left", framealpha=0.7)

    def loop(self, config_path, no_write):
        print("\ntrajectory mode — commands: v <km/s> | x <µm> | save | q")
        self.render()
        while True:
            try:
                raw = input("tune> ").strip()
            except EOFError:
                break
            if not raw:
                continue
            cmd, *rest = raw.split()
            cmd = cmd.lower()
            if cmd in ("q", "quit", "exit"):
                break
            elif cmd == "v" and rest:
                self.v_cms = float(rest[0]) * _KMS_TO_CMS; self.render()
            elif cmd == "x" and rest:
                self.x0_cm = float(rest[0]) * _UM_TO_CM; self.render()
            elif cmd == "save":
                edits = [("flash.v_shock_est_cms", round(self.v_cms)),
                         ("flash.x_shock_0_cm", round(self.x0_cm, 6))]
                _confirm_write(config_path, edits, no_write)
            else:
                print("  ? commands: v <km/s> | x <µm> | save | q")


# ---------------------------------------------------------------------------
# Regions mode
# ---------------------------------------------------------------------------

class RegionsTuner:
    """Single-dump line-outs with movable shock-front / downstream-edge markers."""

    def __init__(self, cfg, args):
        self.cfg = cfg
        (self.flash_dir, all_files, line_start, line_end,
         _ic_index, t_ic_s) = _run_paths(cfg)
        self.out_dir = _out_dir(cfg, args.output_dir, self.flash_dir)

        self.idx = args.snapshot_idx % len(all_files)   # positive plot-file index = config key
        snap_file = all_files[self.idx]
        self.png = os.path.join(self.out_dir, f"tune_flash_regions_idx{self.idx:04d}.png")

        print(f"Loading lineout for dump idx {self.idx} "
              f"({os.path.basename(snap_file)}) …", flush=True)
        self.lo = fu.flash_lineout(snap_file, line_start, line_end)
        self.x_um  = np.asarray(self.lo["x"].to("cm")) * _CM_TO_UM
        self.t_ns  = float(self.lo["t_s"] * _S_TO_NS)
        self.snap_name = os.path.basename(snap_file)

        # 2D density slice through the LOS (static image; only the markers move).
        # Built once here, then just re-overlaid each render.
        self.slice = None
        if not args.no_slice:
            try:
                self.slice = self._load_slice(snap_file, line_start, line_end,
                                              args.slice_axis, args.slice_halfwidth_um)
            except Exception as e:
                print(f"  Warning: could not build density slice ({e}); "
                      "falling back to line-outs only.")

        # Seed markers from the config (formula fallback from the flash: trajectory).
        per = cfg.get("flash_dump_params", {}).get(self.idx, {})
        flash = cfg.get("flash", {})
        if "x_shock_cm" in per:
            self.x_shock_um = float(per["x_shock_cm"]) * _CM_TO_UM
        else:
            v_cms = float(flash.get("v_shock_est_cms", 0.0))
            x0_cm = float(flash.get("x_shock_0_cm", 0.0))
            # x0_cm is the front at the IC dump time; project to this dump's time
            # exactly as flash_overview does (x0 + v·(t − t_IC)).
            self.x_shock_um = (x0_cm + v_cms * (self.lo["t_s"] - t_ic_s)) * _CM_TO_UM \
                if (v_cms or x0_cm) else float(self.x_um.mean())
        self.x_down_um = float(per["x_downstream_start_cm"]) * _CM_TO_UM \
            if "x_downstream_start_cm" in per else self.x_shock_um - 200.0

    def _load_slice(self, snap_file, line_start, line_end, slice_axis, halfwidth_um):
        """Build a 2D density slice through the LOS, oriented so the LOS axis is
        horizontal (= 'distance along LOS', shared with the line-out panels).

        Returns a dict: ``img`` [transverse, los] in cm⁻³, ``extent`` (LOS-distance
        µm horizontal, transverse µm vertical) for ``imshow``, and ``los_tr_um``
        (the LOS's transverse position, for the guide line).
        """
        start = np.asarray(line_start, dtype=float)   # cm
        end   = np.asarray(line_end,   dtype=float)
        los_axis = int(np.argmax(np.abs(end - start)))      # 1 (y) for this run

        ax_idx = {"x": 0, "y": 1, "z": 2}[slice_axis]
        if ax_idx == los_axis:                              # slice must contain the LOS
            ax_idx = next(a for a in (2, 1, 0) if a != los_axis)
        slice_coord = float(start[ax_idx])                  # plane passes through the LOS

        ds = yt.load(snap_file)
        ax_h = ds.coordinates.x_axis[ax_idx]   # in-plane horizontal data-axis index
        ax_v = ds.coordinates.y_axis[ax_idx]   # in-plane vertical   data-axis index
        tr_axis = ax_h if ax_v == los_axis else ax_v        # the transverse in-plane axis

        hw_cm = (halfwidth_um or 0.0) * _UM_TO_CM
        def _span(a):  # (width_cm, center_cm) for in-plane axis a
            if a == los_axis:
                return abs(end[a] - start[a]), 0.5 * (start[a] + end[a])
            if hw_cm > 0:
                return 2.0 * hw_cm, start[a]
            le = float(ds.domain_left_edge[a].to("cm")); re = float(ds.domain_right_edge[a].to("cm"))
            return re - le, 0.5 * (le + re)

        wh, ch = _span(ax_h)
        wv, cv = _span(ax_v)
        center = [slice_coord, slice_coord, slice_coord]
        center[ax_h], center[ax_v] = ch, cv
        res_h = 512 if ax_h == los_axis else 256
        res_v = 512 if ax_v == los_axis else 256

        frb = ds.slice(ax_idx, slice_coord).to_frb(
            width=((wh, "cm"), (wv, "cm")), resolution=(res_h, res_v), center=center)
        img = np.array(frb[("gas", "El_number_density")])   # shape (res_v, res_h) = [ax_v, ax_h]
        b = [float(v.to("cm")) for v in frb.bounds]         # (h_lo,h_hi,v_lo,v_hi) cm

        if ax_h == los_axis:                                # LOS already horizontal
            disp, (los_lo, los_hi), (tr_lo, tr_hi) = img, (b[0], b[1]), (b[2], b[3])
        else:                                               # LOS is vertical -> transpose
            disp, (los_lo, los_hi), (tr_lo, tr_hi) = img.T, (b[2], b[3]), (b[0], b[1])

        los0 = start[los_axis]                              # lineout x is distance from start
        extent = [(los_lo - los0) * _CM_TO_UM, (los_hi - los0) * _CM_TO_UM,
                  tr_lo * _CM_TO_UM, tr_hi * _CM_TO_UM]
        return dict(img=disp, extent=extent,
                    los_tr_um=float(start[tr_axis]) * _CM_TO_UM,
                    tr_label=f"transverse {'xyz'[tr_axis]} [$\\mu$m]")

    def render(self):
        if self.slice is not None:
            fig, (axs, axn, axT) = plt.subplots(
                3, 1, figsize=(13, 12), sharex=True,
                gridspec_kw=dict(height_ratios=[1.1, 1, 1]))
            self._slice_panel(axs)
        else:
            fig, (axn, axT) = plt.subplots(2, 1, figsize=(13, 9), sharex=True,
                                           gridspec_kw=dict(height_ratios=[1, 1]))
        ne = np.asarray(self.lo["ne"].to("cm**-3"))
        B  = np.asarray(self.lo["B_mag"].to("G"))
        Te = np.asarray(self.lo["Te"].to("eV"))
        Ti = np.asarray(self.lo["Ti"].to("eV"))

        # Top: nₑ (log, left) + |B| (right).  Upstream is the ambient side (larger x);
        # downstream is the shocked side between x_down and x_shock.
        axn.semilogy(self.x_um, ne, color="tab:purple", lw=1.8, label=r"$n_e$")
        axn.set_ylabel(r"$n_e$ [cm$^{-3}$]", color="tab:purple")
        axn.tick_params(axis="y", labelcolor="tab:purple")
        axb = axn.twinx()
        axb.plot(self.x_um, B, color="tab:orange", lw=1.6, label=r"$|B|$")
        axb.set_ylabel(r"$|B|$ [G]", color="tab:orange")
        axb.tick_params(axis="y", labelcolor="tab:orange")
        axn.set_title(f"dump idx {self.idx} ({self.snap_name}, t={self.t_ns:.2f} ns) "
                      "— density & field compression")

        # Bottom: Tₑ, Tᵢ.
        axT.semilogy(self.x_um, Te, color="tab:blue", lw=1.8, label=r"$T_e$")
        axT.semilogy(self.x_um, Ti, color="tab:red",  lw=1.8, label=r"$T_i$")
        axT.set_ylabel("Temperature [eV]")
        axT.set_xlabel(r"distance along LOS [$\mu$m]")
        axT.legend(fontsize=9, loc="best")
        axT.set_title("electron & ion temperature")

        marked = (axs, axn, axT) if self.slice is not None else (axn, axT)
        for ax in marked:
            self._marks(ax)
        for ax in (axn, axT):
            ax.grid(alpha=0.3, which="both")
        axn.legend(loc="upper left", fontsize=9, framealpha=0.7)

        fig.tight_layout()
        fig.savefig(self.png, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  shock={self.x_shock_um:.1f} µm, down={self.x_down_um:.1f} µm  "
              f"(shock={self.x_shock_um * _UM_TO_CM:.4g} cm, "
              f"down={self.x_down_um * _UM_TO_CM:.4g} cm)")
        # Region widths so the user can sanity-check the averaging windows.
        n_dn = int(((self.x_um >= self.x_down_um) & (self.x_um <= self.x_shock_um)).sum())
        n_up = int((self.x_um > self.x_shock_um).sum())
        print(f"  region cells: downstream {n_dn}, upstream {n_up}")
        print(f"  ↻ wrote {self.png} — refresh in your IDE")

    def _slice_panel(self, ax):
        """2D density slice with the LOS path overlaid (markers added by _marks)."""
        s = self.slice
        img = s["img"]
        pos = img[np.isfinite(img) & (img > 0)]
        vmax = np.percentile(pos, 99.5) if pos.size else 1.0
        vmin = max(np.percentile(pos, 2) if pos.size else 1e-6, vmax * 1e-4)
        im = ax.imshow(np.clip(img, vmin, None), origin="lower", extent=s["extent"],
                       aspect="auto", cmap="magma",
                       norm=LogNorm(vmin=vmin, vmax=vmax))
        # Inset colorbar (just outside the right edge) so the slice axes keeps the
        # SAME width as the line-out panels — markers then align across all panels.
        cax = ax.inset_axes([1.005, 0.0, 0.012, 1.0])
        cb = ax.figure.colorbar(im, cax=cax)
        cb.set_label(r"$n_e$ [cm$^{-3}$]")
        # The LOS itself (the line the 1D traces are sampled along).
        ax.axhline(s["los_tr_um"], color="cyan", ls="-", lw=1.0, alpha=0.7, label="LOS")
        ax.set_ylabel(s["tr_label"])
        ax.set_title(f"density slice through LOS — dump idx {self.idx} "
                     f"(t={self.t_ns:.2f} ns)")

    def _marks(self, ax):
        ax.axvline(self.x_shock_um, color="k", ls="--", lw=1.6,
                   label=f"shock {self.x_shock_um:.0f} µm")
        ax.axvline(self.x_down_um, color="0.5", ls=":", lw=1.6,
                   label=f"downstream start {self.x_down_um:.0f} µm")

    def loop(self, config_path, no_write):
        print(f"\nregions mode — dump idx {self.idx} ({self.snap_name}, t={self.t_ns:.2f} ns)")
        print("  commands: shock <µm> | down <µm> | save | q")
        self.render()
        while True:
            try:
                raw = input("tune> ").strip()
            except EOFError:
                break
            if not raw:
                continue
            cmd, *rest = raw.split()
            cmd = cmd.lower()
            if cmd in ("q", "quit", "exit"):
                break
            elif cmd == "shock" and rest:
                self.x_shock_um = float(rest[0]); self.render()
            elif cmd == "down" and rest:
                self.x_down_um = float(rest[0]); self.render()
            elif cmd == "save":
                edits = [
                    (f"flash_dump_params.{self.idx}.x_shock_cm",
                     round(self.x_shock_um * _UM_TO_CM, 6)),
                    (f"flash_dump_params.{self.idx}.x_downstream_start_cm",
                     round(self.x_down_um * _UM_TO_CM, 6)),
                ]
                _confirm_write(config_path, edits, no_write)
            else:
                print("  ? commands: shock <µm> | down <µm> | save | q")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Interactively place a FLASH run's shock front (config write-back).")
    p.add_argument("--config", required=True, help="Path to analysis YAML config.")
    p.add_argument("--mode", choices=("trajectory", "regions"), default="trajectory")
    p.add_argument("--snapshot-idx", type=int, default=-1, dest="snapshot_idx",
                   help="(regions) plot-file index to tune (default -1 = last dump).")
    p.add_argument("--slice-axis", default="z", choices=("x", "y", "z"),
                   dest="slice_axis",
                   help="(regions) axis ⟂ to the 2D density slice (default z; the "
                        "slice plane always contains the LOS).")
    p.add_argument("--slice-halfwidth-um", type=float, default=2000.0,
                   dest="slice_halfwidth_um",
                   help="(regions) transverse half-width of the slice [µm] around the "
                        "LOS (default 2000; 0 = full domain).")
    p.add_argument("--no-slice", action="store_true",
                   help="(regions) skip the 2D density slice panel (line-outs only).")
    p.add_argument("--stride", type=int, default=1,
                   help="(trajectory) dump stride for the streaks (default 1).")
    p.add_argument("--t-start", type=int, default=0, dest="t_start")
    p.add_argument("--t-stop", type=int, default=None, dest="t_stop",
                   help="(trajectory) last plot-file index (default: all available).")
    p.add_argument("--nprocs", type=int, default=None,
                   help="(trajectory) worker processes for loading dumps "
                        "(default: all node cores; 1 = serial).")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--no-write", action="store_true",
                   help="Render and preview edits but never modify the config.")
    plot_style.add_publication_arg(p)
    args = p.parse_args()
    plot_style.apply(args.publication)

    cfg = analysis_utils.load_config(args.config)
    config_path = os.path.abspath(args.config)
    print(f"Config  : {config_path}")
    print(f"sim_dir : {cfg['sim_dir']}")

    if args.mode == "trajectory":
        TrajectoryTuner(cfg, args).loop(config_path, args.no_write)
    else:
        RegionsTuner(cfg, args).loop(config_path, args.no_write)


if __name__ == "__main__":
    main()
