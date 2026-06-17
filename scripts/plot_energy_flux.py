"""Plot the shock-frame energy-flux conservation check from a precomputed .npz.

Two panels:
  left  — energy-flux channels vs x (the total should be flat across the front)
  right — upstream vs downstream bar chart (conserved if the Total bars match)

Usage
-----
    python scripts/plot_energy_flux.py \\
        --results results/<run>/energy_flux_t000360.npz [--output-dir figures/]
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

from analysis_utils import load_results

_CHANNELS = [
    ("bulk", "Bulk KE  U·½ρ|U|²", "C0"),
    ("internal", "Internal  U·ε", "C1"),
    ("pressure", "Pressure work  U·P_xx", "C2"),
    ("poynting", "Poynting  E×B", "C4"),
]


def plot_profiles(r: dict, ax: plt.Axes) -> None:
    x = r["x_axis"]
    xs = float(r["x_shock"])
    dx = float(r["dx"])
    x_up = xs + float(r["up_ncells"]) * dx
    x_dn = xs - float(r["dn_ncells"]) * dx

    for key, label, color in _CHANNELS:
        ax.plot(x, r[f"F_{key}"], color=color, lw=1.2, label=label)
    ax.plot(x, r["F_total"], color="k", lw=2.0, label="Total")
    ax.axhline(0.0, color="0.6", lw=0.8)
    ax.axvline(xs, color="k", ls="--", lw=1, label=f"shock {xs:.0f}")
    ax.axvspan(xs, x_up, color="C0", alpha=0.08, label="upstream window")
    ax.axvspan(x_dn, xs, color="C3", alpha=0.08, label="downstream window")

    ax.set_xlabel("x [c/ωpe]")
    ax.set_ylabel("Energy flux  F / c  [n₀ mₑ c²]")
    ax.set_title(f"Shock-frame energy flux, t={r['t_val']}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xlim(x_dn - 50, x_up + 50)
    win = (x >= x_dn - 50) & (x <= x_up + 50)
    if win.any():
        lo, hi = np.nanmin(r["F_total"][win]), np.nanmax(r["F_total"][win])
        pad = 0.5 * (hi - lo + 1e-12)
        ax.set_ylim(min(lo, np.nanmin(r["F_bulk"][win])) - pad, hi + pad)


def plot_bars(r: dict, ax: plt.Axes) -> None:
    keys = ["bulk", "internal", "pressure", "poynting", "total"]
    labels = ["Bulk", "Internal", "Pressure", "Poynting", "Total"]
    up = [float(r[f"upstream_{k}"]) for k in keys]
    dn = [float(r[f"downstream_{k}"]) for k in keys]

    xpos = np.arange(len(keys))
    w = 0.38
    ax.bar(xpos - w / 2, up, w, label="Upstream")
    ax.bar(xpos + w / 2, dn, w, label="Downstream")
    ax.axhline(0.0, color="0.6", lw=0.8)
    ax.axvline(len(keys) - 1.5, color="0.7", lw=1, ls=":")  # set off Total

    ratio = dn[-1] / up[-1] if up[-1] else float("nan")
    y = min(up[-1], dn[-1])
    ax.annotate(f"dn/up = {ratio:.2f}\n({100*(dn[-1]-up[-1])/abs(up[-1]):+.0f}%)",
                xy=(xpos[-1], y), xytext=(0, -2), textcoords="offset points",
                ha="center", va="top", fontsize=9)

    ax.set_xticks(xpos)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean energy flux  F / c  [n₀ mₑ c²]")
    src = str(r.get("v_shock_source", "fit"))
    ax.set_title(f"Flux conservation  (v_shock={float(r['v_shock']):.4f}c, {src})")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)


def main():
    parser = argparse.ArgumentParser(description="Plot shock-frame energy-flux conservation.")
    parser.add_argument("--results", required=True, help="Path to energy_flux .npz.")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    r = load_results(args.results)
    out_dir = args.output_dir or os.path.dirname(os.path.abspath(args.results))
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plot_profiles(r, axes[0])
    plot_bars(r, axes[1])
    plt.tight_layout()

    out_path = os.path.join(out_dir, f"energy_flux_t{int(r['t_val']):06d}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
