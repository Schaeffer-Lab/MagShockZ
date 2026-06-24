"""temperature_ratios — electron/ion temperature ratios and anisotropy at the shock.

Science
-------
For one OSIRIS dump this builds temperature profiles (OSIRIS units, m_e c^2) from
the parallel (p1, along shock normal) and perpendicular (p2, transverse) phase
spaces of electrons and Al ions, then forms the dimensionless ratios:

    T_e/T_al   (parallel and perpendicular)   — electron vs. ion heating
    T_par/T_perp   (each species)             — temperature anisotropy

Every formula is owned by the pure, unit-tested module
``src/temperature_anisotropy.py``: ``temperature_profile`` (T = |rqm| <(u-<u>)^2>),
``safe_ratio`` (guarded division), and ``region_averages`` (upstream/downstream
means).  See the OSIRIS units table in CLAUDE.md.

Validation
----------
This file is orchestration + plotting only; it contains no physics formulas.
To validate a number, read the named function in ``src/temperature_anisotropy.py``
and its test in ``tests/test_temperature_anisotropy.py``.  The computed result is
also saved to an inspectable ``.npz`` whose schema is the
``TemperatureRatiosResult`` dataclass below.

Usage
-----
    python scripts/temperature_ratios.py --config config/<run>.yaml \\
        [--timestep-idx -1] [--no-plot] [--output ...] [--output-dir ...]
"""

import argparse
import os
import sys
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import osh5io

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

import analysis_utils
import plot_style
from analysis_utils import axis_values
import temperature_anisotropy as ta

# The eight profiles whose upstream/downstream averages are stored (and bar-charted).
_AVG_KEYS = ("T_par_e", "T_par_al", "T_perp_e", "T_perp_al",
             "T_e_al_par", "T_e_al_perp", "anis_e", "anis_al")


# ---------------------------------------------------------------------------
# Output schema (this dataclass IS the .npz schema; see analysis_utils.save_result)
# ---------------------------------------------------------------------------

@dataclass
class TemperatureRatiosResult:
    # --- temperature profiles [m_e c^2] on the phase-space grid ---
    x_axis: np.ndarray            # [c/wpe]
    T_par_e: np.ndarray           # parallel (p1) electron T   [m_e c^2]
    T_par_al: np.ndarray          # parallel (p1) Al-ion T     [m_e c^2]
    T_perp_e: np.ndarray          # perpendicular (p2) electron T [m_e c^2]
    T_perp_al: np.ndarray         # perpendicular (p2) Al-ion T   [m_e c^2]
    # --- dimensionless ratio profiles ---
    T_e_al_par: np.ndarray        # T_par,e / T_par,al
    T_e_al_perp: np.ndarray       # T_perp,e / T_perp,al
    anis_e: np.ndarray            # T_par,e / T_perp,e
    anis_al: np.ndarray           # T_par,al / T_perp,al
    # --- shock kinematics / provenance ---
    t_val: int                    # dump index (file suffix)
    t_sim: float                  # simulation time [1/wpe]
    x_shock: float                # shock-front position [c/wpe]
    x_downstream_start: float     # left edge of downstream region [c/wpe]
    config_path: str              # absolute path of the analysis config used
    # --- region averages of each of the eight profiles above ---
    up: dict                      # upstream  mean, keyed by _AVG_KEYS
    dn: dict                      # downstream mean, keyed by _AVG_KEYS


# ---------------------------------------------------------------------------
# Compute (orchestration only — every formula lives in temperature_anisotropy.py)
# ---------------------------------------------------------------------------

def compute(cfg: dict, timestep_idx: int = -1, config_path: str = "") -> TemperatureRatiosResult:
    """Load one dump, build the temperature profiles/ratios, return the result."""
    sim_dir = cfg["sim_dir"]
    t_val = cfg["times"][timestep_idx]

    sim = analysis_utils.run_from_config(cfg)
    species_rqm = {sp: sim.rqm_of(sp) for sp in ["al", "e"]}  # per-species rqm from deck

    print("Loading HDF5 files...")
    pha_p1 = {sp: osh5io.read_h5(analysis_utils.diag_path(sim_dir, "p1x1", t_val, sp))
              for sp in ["al", "e"]}
    pha_p2 = {sp: osh5io.read_h5(analysis_utils.diag_path(sim_dir, "p2x1", t_val, sp))
              for sp in ["al", "e"]}

    x_axis = axis_values(pha_p1["al"], ax_idx=1)
    t_sim = float(pha_p1["al"].run_attrs["TIME"][0])
    dump = analysis_utils.resolve_dump_params(cfg, t_val, t_sim)
    x_shock = dump["x_shock"]
    x_downstream_start = dump["x_downstream_start"]

    # Temperature profiles in simulation units (m_e c^2).
    T_par = {sp: ta.temperature_profile(pha_p1[sp], species_rqm[sp], "p1") for sp in ["al", "e"]}
    T_perp = {sp: ta.temperature_profile(pha_p2[sp], species_rqm[sp], "p2") for sp in ["al", "e"]}

    # Dimensionless ratios (guarded division).
    profiles = {
        "T_par_e": T_par["e"], "T_par_al": T_par["al"],
        "T_perp_e": T_perp["e"], "T_perp_al": T_perp["al"],
        "T_e_al_par": ta.safe_ratio(T_par["e"], T_par["al"]),
        "T_e_al_perp": ta.safe_ratio(T_perp["e"], T_perp["al"]),
        "anis_e": ta.safe_ratio(T_par["e"], T_perp["e"]),
        "anis_al": ta.safe_ratio(T_par["al"], T_perp["al"]),
    }

    # Upstream/downstream averages of each profile.
    up, dn = {}, {}
    for key in _AVG_KEYS:
        up[key], dn[key] = ta.region_averages(profiles[key], x_axis, x_shock, x_downstream_start)

    return TemperatureRatiosResult(
        x_axis=x_axis, **profiles,
        t_val=int(t_val), t_sim=t_sim,
        x_shock=float(x_shock), x_downstream_start=float(x_downstream_start),
        config_path=config_path, up=up, dn=dn,
    )


def _print_summary(r: TemperatureRatiosResult) -> None:
    """Console tables of the temperature averages and ratios (no computation)."""
    print(f"t_sim   : {r.t_sim:.1f} [ωpe⁻¹]")
    print(f"x_shock : {r.x_shock:.1f} [c/ωpe]")
    print("\n--- Temperature summary [m_e c^2] ---")
    print(f"{'':12s} {'T_par up':>10} {'T_par dn':>10} {'T_perp up':>10} {'T_perp dn':>10}")
    for sp in ["e", "al"]:
        print(f"{sp:<12} {r.up[f'T_par_{sp}']:>10.2f} {r.dn[f'T_par_{sp}']:>10.2f} "
              f"{r.up[f'T_perp_{sp}']:>10.2f} {r.dn[f'T_perp_{sp}']:>10.2f}")
    print("\n--- Ratios ---")
    for key, label in [("T_e_al_par",  "T_e/T_al (par) "),
                       ("T_e_al_perp", "T_e/T_al (perp)"),
                       ("anis_e",      "T_par/T_perp e  "),
                       ("anis_al",     "T_par/T_perp al ")]:
        print(f"  {label}: upstream={r.up[key]:.3f}  downstream={r.dn[key]:.3f}")


# ---------------------------------------------------------------------------
# Plot (matplotlib only — reads the result, draws nothing new)
# ---------------------------------------------------------------------------

def _region_spans(ax, r: TemperatureRatiosResult, disp) -> None:
    ax.axvline(disp.x(r.x_downstream_start), color="gray", ls="--", lw=1)
    ax.axvline(disp.x(r.x_shock), color="k", ls="--", lw=1)
    ax.axvspan(disp.x(r.x_axis.min()), disp.x(r.x_downstream_start), color="gray", alpha=0.10)


def _xlim(r: TemperatureRatiosResult, disp):
    return disp.x(r.x_downstream_start), disp.x(r.x_shock + 150)


def _roi_mask(r: TemperatureRatiosResult):
    # native-coordinate window (indexes r.x_axis); display rescaling is for axes only.
    return (r.x_axis >= r.x_downstream_start) & (r.x_axis <= r.x_shock + 150)


def _set_semilogy_ylim(ax, arrs, mask, pad=3.0):
    vals = np.concatenate([a[mask] for a in arrs])
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size:
        ax.set_ylim(vals.min() / pad, vals.max() * pad)


def _set_linear_ylim(ax, arrs, mask, pad=0.15):
    vals = np.concatenate([a[mask] for a in arrs])
    vals = vals[np.isfinite(vals)]
    if vals.size:
        lo, hi = vals.min(), vals.max()
        margin = pad * (hi - lo) if hi > lo else abs(lo) * pad + 0.1
        ax.set_ylim(lo - margin, hi + margin)


def _plot_T_parallel(r, ax, disp):
    xd = disp.x(r.x_axis)
    ax.semilogy(xd, r.T_par_e, label="electrons", color="tab:blue")
    ax.semilogy(xd, r.T_par_al, label="Al ions", color="tab:orange")
    _region_spans(ax, r, disp)
    ax.set_xlim(*_xlim(r, disp))
    _set_semilogy_ylim(ax, [r.T_par_e, r.T_par_al], _roi_mask(r))
    ax.set_xlabel(disp.xlabel())
    ax.set_ylabel(r"T [$m_e c^2$]")
    ax.set_title(r"$T_\parallel$ (along shock normal, p1)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")


def _plot_T_perp(r, ax, disp):
    xd = disp.x(r.x_axis)
    ax.semilogy(xd, r.T_perp_e, label="electrons", color="tab:blue")
    ax.semilogy(xd, r.T_perp_al, label="Al ions", color="tab:orange")
    _region_spans(ax, r, disp)
    ax.set_xlim(*_xlim(r, disp))
    _set_semilogy_ylim(ax, [r.T_perp_e, r.T_perp_al], _roi_mask(r))
    ax.set_xlabel(disp.xlabel())
    ax.set_ylabel(r"T [$m_e c^2$]")
    ax.set_title(r"$T_\perp$ (transverse, p2)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")


def _plot_Te_Tal_ratio(r, ax, disp):
    xd = disp.x(r.x_axis)
    ax.semilogy(xd, r.T_e_al_par, label=r"$T_{\parallel,e}/T_{\parallel,\mathrm{Al}}$",
                color="tab:purple")
    ax.semilogy(xd, r.T_e_al_perp, label=r"$T_{\perp,e}/T_{\perp,\mathrm{Al}}$",
                color="tab:red", ls="--")
    ax.axhline(1, color="k", ls=":", lw=0.8)
    _region_spans(ax, r, disp)
    ax.set_xlim(*_xlim(r, disp))
    _set_semilogy_ylim(ax, [r.T_e_al_par, r.T_e_al_perp], _roi_mask(r))
    ax.set_xlabel(disp.xlabel())
    ax.set_ylabel(r"$T_e / T_\mathrm{Al}$")
    ax.set_title(r"Electron-to-ion temperature ratio")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")


def _plot_anisotropy(r, ax, arr, color, species_label, disp):
    ax.plot(disp.x(r.x_axis), arr, color=color)
    ax.axhline(1, color="k", ls=":", lw=0.8)
    _region_spans(ax, r, disp)
    ax.set_xlim(*_xlim(r, disp))
    _set_linear_ylim(ax, [arr], _roi_mask(r))
    ax.set_xlabel(disp.xlabel())
    ax.set_ylabel(r"$T_\parallel / T_\perp$")
    ax.set_title(rf"{species_label} anisotropy $T_\parallel/T_\perp$")
    ax.grid(alpha=0.3)


def _plot_summary_bars(r, ax):
    """Bar chart of region-averaged ratios for quick comparison."""
    quantities = [
        (r"$T_{\parallel,e}/T_{\parallel,\mathrm{Al}}$", "T_e_al_par"),
        (r"$T_{\perp,e}/T_{\perp,\mathrm{Al}}$",         "T_e_al_perp"),
        (r"$T_\parallel/T_\perp$ (e)",                    "anis_e"),
        (r"$T_\parallel/T_\perp$ (Al)",                   "anis_al"),
    ]
    labels = [q[0] for q in quantities]
    up_vals = [r.up[q[1]] for q in quantities]
    dn_vals = [r.dn[q[1]] for q in quantities]

    x = np.arange(len(labels))
    w = 0.38
    ax.bar(x - w / 2, up_vals, w, label="Upstream", color="tab:cyan")
    ax.bar(x + w / 2, dn_vals, w, label="Downstream", color="tab:green")
    ax.axhline(1, color="k", ls=":", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Ratio")
    ax.set_title("Region-averaged ratios")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)


def plot(r: TemperatureRatiosResult, output_dir: str, disp) -> str:
    """Render the 2×3 figure (profiles, ratios, anisotropy, bars) and save a .png."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    _plot_T_parallel(r, axes[0, 0], disp)
    _plot_T_perp(r, axes[0, 1], disp)
    _plot_Te_Tal_ratio(r, axes[0, 2], disp)
    _plot_anisotropy(r, axes[1, 0], r.anis_e, "tab:blue", "Electron", disp)
    _plot_anisotropy(r, axes[1, 1], r.anis_al, "tab:orange", "Al ion", disp)
    _plot_summary_bars(r, axes[1, 2])

    fig.suptitle(f"Temperature analysis, dump {r.t_val}  {disp.time_title(r.t_sim)}  "
                 f"(shock at {disp.x(r.x_shock):.1f} ${disp.length_label}$)",
                 fontsize=12)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"temperature_ratios_t{r.t_val:06d}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Shock temperature ratios (compute + plot).")
    parser.add_argument("--config", required=True, help="Path to analysis YAML config.")
    parser.add_argument("--timestep-idx", type=int, default=-1, dest="timestep_idx",
                        help="Index into config times list (default: -1, i.e. last dump).")
    parser.add_argument("--no-plot", action="store_true", dest="no_plot",
                        help="Compute and save the .npz only; skip the figure.")
    parser.add_argument("--output", default=None,
                        help="Output .npz path (default results/<run>/temperature_ratios_t{t:06d}.npz).")
    parser.add_argument("--output-dir", default=None, dest="output_dir",
                        help="Directory for the figure (default: alongside the .npz).")
    plot_style.add_publication_arg(parser)
    plot_style.add_units_arg(parser)
    args = parser.parse_args()
    plot_style.apply(args.publication)

    cfg = analysis_utils.load_config(args.config)
    sim_dir = cfg["sim_dir"]
    times = cfg["times"]
    t_val = times[args.timestep_idx]
    print(f"Config  : {args.config}")
    print(f"sim_dir : {sim_dir}")
    print(f"Dump    : t={t_val}  (index {args.timestep_idx} of {len(times)} dumps)")

    result = compute(cfg, args.timestep_idx, config_path=os.path.abspath(args.config))
    _print_summary(result)

    out_path = analysis_utils.default_output_path(args.output, sim_dir, "temperature_ratios", result.t_val)
    analysis_utils.save_result(result, out_path)
    print(f"\nSaved → {out_path}")

    if not args.no_plot:
        disp = plot_style.build_units(args.units, cfg=cfg, config_path=os.path.abspath(args.config))
        out_dir = args.output_dir or os.path.dirname(os.path.abspath(out_path))
        fig_path = plot(result, out_dir, disp)
        print(f"Saved → {fig_path}")


if __name__ == "__main__":
    main()
