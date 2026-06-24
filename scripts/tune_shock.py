"""scripts/tune_shock.py — interactively tune a run's shock parameters.

The analysis suite (overview / energy_partition / temperature_ratios /
dimensionless_params) reads the shock kinematics from the run's analysis config:

    shock:        v_shock, x_shock_0          (the linear front trajectory)
    dump_params:  <idx>: x_shock, x_downstream_start   (per-dump region edges)

Checking whether a value is right used to mean re-running a whole analysis script
and looking at the overlaid line afterwards.  This tool renders the SAME streak /
phase-space plot the analysis draws, lets you type trial values and see the overlay
move immediately, and (on confirm) writes the value back into the YAML with every
comment preserved.

Display is a PNG you refresh in your editor (robust over SSH, no X11): each command
re-renders ``results/<run>/tune_*.png`` and prints its path.

Two modes
---------
trajectory (default) — tune ``shock.v_shock`` / ``shock.x_shock_0`` against the |B|
    and n_e streaks.  Commands:
        v <val>     set trial v_shock [c]
        x <val>     set trial x_shock_0 [c/wpe]
        save        write v_shock + x_shock_0 (+ cached t_ci) to the config (asks y/N)
        q           quit

regions — tune one dump's ``x_shock`` / ``x_downstream_start`` against its p2x1
    phase space (+ n_e/|B| line-outs).  Pick the dump with --dump.  Commands:
        shock <x>   set trial shock-front position [c/wpe]
        down <x>    set trial downstream-region left edge [c/wpe]
        up <x>      set trial upstream reference (writes upstream_window_ncells)
        save        write dump_params.<idx> to the config (asks y/N)
        q           quit

Env: analysis (osh5io / osh5vis).  Examples:
    python scripts/tune_shock.py --config config/magshockz_rqm100_dx0.1.yaml
    python scripts/tune_shock.py --config config/...yaml --mode regions --dump 400
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
from analysis_utils import axis_values, diag_path
from dimensionless_params import ion_gyroperiod
from streak import bmag_frame, density_frame, assemble_streak
import plot_style
import yaml_edit


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _out_dir(cfg, override):
    sim_dir = cfg["sim_dir"]
    out = override or os.path.join(_HERE, "..", "results",
                                   os.path.basename(sim_dir.rstrip("/")))
    os.makedirs(out, exist_ok=True)
    return out


def _ask_yes(prompt):
    try:
        return input(prompt).strip().lower() in ("y", "yes")
    except EOFError:
        return False


def _confirm_write(config_path, edits, no_write):
    """Show each (dotted_path, value) edit, apply to the YAML text, ask y/N, write.

    Each edit is applied via the comment-preserving yaml_edit functions and verified
    to round-trip before the file is touched.  Returns True if written.
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
        if parts[0] == "dump_params":
            new_text = yaml_edit.set_dump_param(new_text, int(parts[1]), parts[2], val)
        else:
            new_text = yaml_edit.set_scalar(new_text, path, val)
        yaml_edit.assert_roundtrip(new_text, path, _normalize(val))

    # Show the line-level diff for the user to confirm.
    print("  pending edits:")
    old_lines = text.split("\n")
    new_lines = new_text.split("\n")
    for i, (a, b) in enumerate(_aligned_diff(old_lines, new_lines)):
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


def _normalize(val):
    """Match yaml_edit's compact rendering when verifying the round-trip."""
    if isinstance(val, float) and val.is_integer():
        return int(val)
    return val


def _aligned_diff(old, new):
    """Pair lines for display; insertions show against an empty old line."""
    n = max(len(old), len(new))
    old = old + [""] * (n - len(old))
    new = new + [""] * (n - len(new))
    return list(zip(old, new))


# ---------------------------------------------------------------------------
# Trajectory mode
# ---------------------------------------------------------------------------

class TrajectoryTuner:
    """Caches the |B|/n_e streaks + front detection so each redraw is instant."""

    def __init__(self, cfg, args):
        self.cfg = cfg
        self.sim_dir = cfg["sim_dir"]
        self.layout = analysis_utils.detect_layout(self.sim_dir)
        self.out_dir = _out_dir(cfg, args.output_dir)
        self.png = os.path.join(self.out_dir, "tune_trajectory.png")

        sim = analysis_utils.run_from_config(cfg)
        self.rqm_i = sim.rqm_of("al")

        dumps = self._dump_list(args)
        if len(dumps) < 2:
            raise RuntimeError(f"Need >=2 field dumps for a streak; got {len(dumps)}.")
        print(f"Loading |B|, n_e over {len(dumps)} dumps "
              f"({dumps[0]}..{dumps[-1]}, stride {dumps[1] - dumps[0]})...")
        hw = args.transverse_hw
        self.B_h5, self.B_streak, self.time, self.x_f = assemble_streak(
            [bmag_frame(self.sim_dir, t, self.layout, hw) for t in dumps])
        self.ne_h5, self.ne_streak, _, self.x_ne = assemble_streak(
            [density_frame(self.sim_dir, "e", t, self.layout, hw) for t in dumps])

        # Upstream Alfvenic context, so the trial line can be quoted as M_A.
        self.v_A = self._alfven_speed()

        # Upstream ion gyroperiod T_ci, computed "for free" from the t=0 upstream |B'|
        # already loaded above (self.B_up).  Cached to the config on `save` so every
        # --units ion analysis reads one consistent value instead of re-measuring it.
        self.T_ci = ion_gyroperiod(abs(self.rqm_i), self.B_up)

        # Trial line seeded from the config.
        self.v_shock = float(cfg["shock"]["v_shock"])
        self.x_shock_0 = float(cfg["shock"]["x_shock_0"])

    def _dump_list(self, args):
        stride = args.field_stride + (args.field_stride % 2)  # even -> savg exists
        t_stop = args.t_stop
        if t_stop is None:
            b3 = self.layout.field_quantity("b3")
            d = f"{self.sim_dir}/MS/FLD/{b3}"
            t_stop = max(int(f.split("-")[-1].split(".")[0])
                         for f in os.listdir(d) if f.endswith(".h5"))
        out = []
        for t in range(args.t_start, t_stop + 1, stride):
            if not os.path.exists(diag_path(self.sim_dir, self.layout.field_quantity("b3"), t)):
                continue
            if not os.path.exists(diag_path(self.sim_dir, self.layout.charge_quantity, t, "e")):
                continue
            out.append(t)
        return out

    def _alfven_speed(self):
        x0 = float(self.cfg["shock"]["x_shock_0"])
        up_b = self.x_f > x0
        B_up = float(np.median(self.B_streak[0][up_b])) if up_b.any() \
            else float(np.median(self.B_streak[0]))
        up_n = self.x_ne > x0
        ne_up = float(np.median(self.ne_streak[0][up_n])) if up_n.any() \
            else float(np.median(self.ne_streak[0]))
        self.B_up, self.ne_up = B_up, ne_up
        return B_up / np.sqrt(abs(self.rqm_i) * ne_up)  # [c]

    def render(self):
        M_A = self.v_shock / self.v_A
        fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
        x_trial = self.x_shock_0 + self.v_shock * self.time
        overlays = [
            (self.time, x_trial, dict(color="white", ls="-", lw=2.0),
             f"trial (v={self.v_shock:.4f}c, $M_A$={M_A:.2f})"),
        ]
        self._panel(axes[0], self.B_h5, self.x_f, r"$|B|$", "viridis", False, overlays)
        self._panel(axes[1], self.ne_h5, self.x_ne, r"$n_e$", "magma", True, overlays)
        axes[1].set_xlabel(r"$t$ [$\omega_{pe}^{-1}$]")
        fig.suptitle(f"trajectory tuning — trial v_shock={self.v_shock:.4f}c, "
                     f"x_shock_0={self.x_shock_0:.1f}  ($M_A$={M_A:.2f})", fontsize=13)
        fig.tight_layout()
        fig.savefig(self.png, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  M_A = {M_A:.2f}   (v_A = {self.v_A:.5f} c)")
        print(f"  T_ci = {self.T_ci:.4g} 1/wpe   (upstream |B'| = {self.B_up:.4g}; "
              f"cached to config on save)")
        print(f"  ↻ wrote {self.png} — refresh in your IDE")

    @staticmethod
    def _panel(ax, streak, x, title, cmap, log, overlays):
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
        osh5vis.osimshow(streak.transpose(), ax=ax, cmap=cmap, title=title, **kw)
        for t_arr, x_arr, style, leg in overlays:
            ax.plot(t_arr, x_arr, label=leg, **style)
        ax.set_ylim(x.min(), x.max())
        ax.legend(fontsize=8, loc="upper left", framealpha=0.7)

    def loop(self, config_path, no_write):
        print("\ntrajectory mode — commands: v <val> | x <val> | save | q")
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
                self.v_shock = float(rest[0]); self.render()
            elif cmd == "x" and rest:
                self.x_shock_0 = float(rest[0]); self.render()
            elif cmd == "save":
                edits = [("shock.v_shock", round(self.v_shock, 6)),
                         ("shock.x_shock_0", round(self.x_shock_0, 1))]
                if np.isfinite(self.T_ci):
                    edits.append(("t_ci", round(float(self.T_ci), 2)))
                _confirm_write(config_path, edits, no_write)
            else:
                print("  ? commands: v <val> | x <val> | save | q")


# ---------------------------------------------------------------------------
# Regions mode
# ---------------------------------------------------------------------------

class RegionsTuner:
    """Single-dump phase space with movable shock/downstream/upstream markers."""

    def __init__(self, cfg, args):
        self.cfg = cfg
        self.sim_dir = cfg["sim_dir"]
        self.layout = analysis_utils.detect_layout(self.sim_dir)
        self.out_dir = _out_dir(cfg, args.output_dir)
        self.t_val = args.dump
        self.png = os.path.join(self.out_dir, f"tune_regions_t{self.t_val:06d}.png")

        pha_path = diag_path(self.sim_dir, self.layout.pha_name(2), self.t_val, "al")
        if not os.path.exists(pha_path):
            raise FileNotFoundError(
                f"No ion p2x1 phase space for dump {self.t_val}:\n  {pha_path}\n"
                f"Pick a dump that has phase-space output (--dump).")
        self.ps = np.abs(osh5io.read_h5(pha_path))               # |f|, H5Data
        self.x_pha = axis_values(self.ps, 1)
        self.dx = float(self.x_pha[1] - self.x_pha[0])
        self.t_sim = float(self.ps.run_attrs["TIME"][0])

        # n_e / |B| line-outs for context (read once).
        hw = args.transverse_hw
        self.ne = density_frame(self.sim_dir, "e", self.t_val, self.layout, hw)
        self.bmag = bmag_frame(self.sim_dir, self.t_val, self.layout, hw)

        # Seed markers from the existing config (formula fallback for x_shock).
        shock = cfg["shock"]
        per = cfg.get("dump_params", {}).get(self.t_val, {})
        self.x_shock = float(per.get("x_shock",
                                     shock["x_shock_0"] + shock["v_shock"] * self.t_sim))
        self.x_down = float(per.get("x_downstream_start", self.x_shock - 200.0))
        self.x_up = None

    def render(self):
        fig, (axp, axl) = plt.subplots(2, 1, figsize=(13, 9), sharex=True,
                                       gridspec_kw=dict(height_ratios=[2, 1]))
        arr = np.asarray(self.ps)
        vmax = np.percentile(arr[arr > 0], 99.7) if (arr > 0).any() else 1.0
        osh5vis.osimshow(self.ps, ax=axp, cmap="inferno",
                         norm=LogNorm(vmin=vmax * 1e-4, vmax=vmax),
                         title=f"ion $p_2$–$x$  (dump {self.t_val}, t={self.t_sim:.0f})",
                         cblabel=r"$f$ (arb.)")
        osh5vis.osplot1d(self.ne, ax=axl, color="tab:purple", show_time=False, title="")
        axl.set_title("electron density"); axl.grid(alpha=0.3)
        for ax in (axp, axl):
            self._marks(ax)
        axp.legend(fontsize=9, loc="upper left", framealpha=0.7)
        axl.set_xlabel(r"$x$ [$c/\omega_{pe}$]")
        fig.tight_layout()
        fig.savefig(self.png, dpi=130, bbox_inches="tight")
        plt.close(fig)
        up_txt = f", up={self.x_up:.1f}" if self.x_up is not None else ""
        print(f"  shock={self.x_shock:.1f}, down={self.x_down:.1f}{up_txt} [c/wpe]")
        print(f"  ↻ wrote {self.png} — refresh in your IDE")

    def _marks(self, ax):
        ax.axvline(self.x_shock, color="cyan", ls="--", lw=1.6, label=f"shock {self.x_shock:.0f}")
        ax.axvline(self.x_down, color="lime", ls="--", lw=1.6, label=f"downstream {self.x_down:.0f}")
        if self.x_up is not None:
            ax.axvline(self.x_up, color="white", ls=":", lw=1.4, label=f"upstream {self.x_up:.0f}")

    def loop(self, config_path, no_write):
        print(f"\nregions mode — dump {self.t_val} (t={self.t_sim:.0f}), dx={self.dx:.3f} c/wpe")
        print("  commands: shock <x> | down <x> | up <x> | save | q")
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
                self.x_shock = float(rest[0]); self.render()
            elif cmd == "down" and rest:
                self.x_down = float(rest[0]); self.render()
            elif cmd == "up" and rest:
                self.x_up = float(rest[0]); self.render()
            elif cmd == "save":
                edits = [(f"dump_params.{self.t_val}.x_shock", round(self.x_shock, 1)),
                         (f"dump_params.{self.t_val}.x_downstream_start", round(self.x_down, 1))]
                if self.x_up is not None:
                    ncells = int(round(abs(self.x_up - self.x_shock) / self.dx))
                    edits.append(("upstream_window_ncells", ncells))
                _confirm_write(config_path, edits, no_write)
            else:
                print("  ? commands: shock <x> | down <x> | up <x> | save | q")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Interactively tune a run's shock parameters.")
    p.add_argument("--config", required=True, help="Path to analysis YAML config.")
    p.add_argument("--mode", choices=("trajectory", "regions"), default="trajectory")
    p.add_argument("--dump", type=int, default=None,
                   help="(regions) dump index to tune; default: last config time.")
    p.add_argument("--stride", type=int, default=8, dest="field_stride",
                   help="(trajectory) field-dump stride for the streaks (even; default 8).")
    p.add_argument("--t-start", type=int, default=0, dest="t_start")
    p.add_argument("--t-stop", type=int, default=None, dest="t_stop",
                   help="(trajectory) last dump index (default: largest available).")
    p.add_argument("--transverse-halfwidth", type=float, default=5.0, dest="transverse_hw",
                   help="Half-width [c/wpe] of the central band averaged for 2D->1D (ignored in 1D).")
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
        if args.dump is None:
            if not cfg.get("times"):
                p.error("regions mode needs --dump (no times list in config).")
            args.dump = int(cfg["times"][-1])
            print(f"(no --dump given; using last config time: {args.dump})")
        RegionsTuner(cfg, args).loop(config_path, args.no_write)


if __name__ == "__main__":
    main()
