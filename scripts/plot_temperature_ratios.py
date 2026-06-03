"""Plot temperature ratios from a precomputed .npz results file.

Produces a 2×3 figure:
  Row 0: T_par profiles  |  T_perp profiles  |  T_e/T_al ratio profiles
  Row 1: T_par/T_perp for e  |  T_par/T_perp for al  |  Region-averaged bar chart

Usage
-----
    python scripts/plot_temperature_ratios.py \\
        --results results/perlmutter_1.3.1d/temperature_ratios_t000360.npz \\
        [--output-dir figures/]
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

from analysis_utils import load_results


def _region_spans(ax, r):
    x = r["x_axis"]
    ax.axvline(r["x_downstream_start"], color="gray", ls="--", lw=1)
    ax.axvline(r["x_shock"], color="k", ls="--", lw=1)
    ax.axvspan(x.min(), r["x_downstream_start"], color="gray", alpha=0.10)


def _xlim(r):
    return r["x_downstream_start"], r["x_shock"] + 150


def _roi_mask(r):
    x = r["x_axis"]
    xlo, xhi = _xlim(r)
    return (x >= xlo) & (x <= xhi)


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


def plot_T_parallel(r, ax):
    x = r["x_axis"]
    mask = _roi_mask(r)
    ax.semilogy(x, r["T_par_e"], label="electrons", color="tab:blue")
    ax.semilogy(x, r["T_par_al"], label="Al ions", color="tab:orange")
    _region_spans(ax, r)
    ax.set_xlim(*_xlim(r))
    _set_semilogy_ylim(ax, [r["T_par_e"], r["T_par_al"]], mask)
    ax.set_xlabel("x [c/ωpe]")
    ax.set_ylabel(r"T [$m_e c^2$]")
    ax.set_title(r"$T_\parallel$ (along shock normal, p1)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")


def plot_T_perp(r, ax):
    x = r["x_axis"]
    mask = _roi_mask(r)
    ax.semilogy(x, r["T_perp_e"], label="electrons", color="tab:blue")
    ax.semilogy(x, r["T_perp_al"], label="Al ions", color="tab:orange")
    _region_spans(ax, r)
    ax.set_xlim(*_xlim(r))
    _set_semilogy_ylim(ax, [r["T_perp_e"], r["T_perp_al"]], mask)
    ax.set_xlabel("x [c/ωpe]")
    ax.set_ylabel(r"T [$m_e c^2$]")
    ax.set_title(r"$T_\perp$ (transverse, p2)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")


def plot_Te_Tal_ratio(r, ax):
    x = r["x_axis"]
    mask = _roi_mask(r)
    ax.semilogy(x, r["T_e_al_par"], label=r"$T_{\parallel,e}/T_{\parallel,\mathrm{Al}}$",
                color="tab:purple")
    ax.semilogy(x, r["T_e_al_perp"], label=r"$T_{\perp,e}/T_{\perp,\mathrm{Al}}$",
                color="tab:red", ls="--")
    ax.axhline(1, color="k", ls=":", lw=0.8)
    _region_spans(ax, r)
    ax.set_xlim(*_xlim(r))
    _set_semilogy_ylim(ax, [r["T_e_al_par"], r["T_e_al_perp"]], mask)
    ax.set_xlabel("x [c/ωpe]")
    ax.set_ylabel(r"$T_e / T_\mathrm{Al}$")
    ax.set_title(r"Electron-to-ion temperature ratio")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")


def plot_anisotropy_e(r, ax):
    x = r["x_axis"]
    mask = _roi_mask(r)
    ax.plot(x, r["anis_e"], color="tab:blue")
    ax.axhline(1, color="k", ls=":", lw=0.8)
    _region_spans(ax, r)
    ax.set_xlim(*_xlim(r))
    _set_linear_ylim(ax, [r["anis_e"]], mask)
    ax.set_xlabel("x [c/ωpe]")
    ax.set_ylabel(r"$T_\parallel / T_\perp$")
    ax.set_title(r"Electron anisotropy $T_\parallel/T_\perp$")
    ax.grid(alpha=0.3)


def plot_anisotropy_al(r, ax):
    x = r["x_axis"]
    mask = _roi_mask(r)
    ax.plot(x, r["anis_al"], color="tab:orange")
    ax.axhline(1, color="k", ls=":", lw=0.8)
    _region_spans(ax, r)
    ax.set_xlim(*_xlim(r))
    _set_linear_ylim(ax, [r["anis_al"]], mask)
    ax.set_xlabel("x [c/ωpe]")
    ax.set_ylabel(r"$T_\parallel / T_\perp$")
    ax.set_title(r"Al ion anisotropy $T_\parallel/T_\perp$")
    ax.grid(alpha=0.3)


def plot_summary_bars(r, ax):
    """Bar chart of region-averaged ratios for quick comparison."""
    quantities = [
        (r"$T_{\parallel,e}/T_{\parallel,\mathrm{Al}}$", "up_T_e_al_par",  "dn_T_e_al_par"),
        (r"$T_{\perp,e}/T_{\perp,\mathrm{Al}}$",         "up_T_e_al_perp", "dn_T_e_al_perp"),
        (r"$T_\parallel/T_\perp$ (e)",                    "up_anis_e",      "dn_anis_e"),
        (r"$T_\parallel/T_\perp$ (Al)",                   "up_anis_al",     "dn_anis_al"),
    ]
    labels = [q[0] for q in quantities]
    up_vals = [r[q[1]] for q in quantities]
    dn_vals = [r[q[2]] for q in quantities]

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


def main():
    parser = argparse.ArgumentParser(description="Plot shock temperature ratios.")
    parser.add_argument("--results", required=True, help="Path to .npz results file.")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    r = load_results(args.results)
    out_dir = args.output_dir or os.path.dirname(os.path.abspath(args.results))
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    plot_T_parallel(r, axes[0, 0])
    plot_T_perp(r, axes[0, 1])
    plot_Te_Tal_ratio(r, axes[0, 2])
    plot_anisotropy_e(r, axes[1, 0])
    plot_anisotropy_al(r, axes[1, 1])
    plot_summary_bars(r, axes[1, 2])

    fig.suptitle(f"Temperature analysis, t={r['t_val']}  (shock at x={r['x_shock']:.1f} c/ωpe)",
                 fontsize=12)
    plt.tight_layout()

    out_path = os.path.join(out_dir, f"temperature_ratios_t{r['t_val']:06d}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
