"""Plot synthetic experimental diagnostics from a precomputed .npz.

Shows the true (full-resolution) shock-normal profiles against the
instrument-resolution forward models for density, |B|, and X-ray emission —
i.e. what the experiment would actually record.

Usage
-----
    python scripts/plot_synthetic_diagnostics.py \\
        --results results/<run>/synthetic_diagnostics_t000360.npz [--output-dir figures/]
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

from analysis_utils import load_results


def main():
    parser = argparse.ArgumentParser(description="Plot synthetic diagnostics.")
    parser.add_argument("--results", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--half-window", type=float, default=300.0)
    args = parser.parse_args()

    r = load_results(args.results)
    out_dir = args.output_dir or os.path.dirname(os.path.abspath(args.results))
    os.makedirs(out_dir, exist_ok=True)

    x, xs = r["x_axis"], float(r["x_shock"])
    m = (x >= xs - args.half_window) & (x <= xs + args.half_window)
    res_um = float(r["resolution_um"])

    panels = [
        ("n_e", "n_e [n₀]", r["n_e"], r["n_e_obs"]),
        ("|B|", "|B| [B₀]", r["Bmag"], r["Bmag_obs"]),
        ("X-ray", "emissivity [rel]", r["xray_emissivity"], r["xray_obs"]),
    ]
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    for a, (name, ylab, full, obs) in zip(ax, panels):
        a.plot(x[m], full[m], color="0.6", lw=1, label="true (full res)")
        a.plot(x[m], obs[m], color="C3", lw=2, label=f"instrument ({res_um:.0f} µm)")
        a.axvline(xs, color="0.4", ls="--", lw=1)
        a.set_xlabel("x [c/ωpe]"); a.set_ylabel(ylab); a.set_title(name)
        a.legend(fontsize=8); a.grid(alpha=0.3)

    # Mark the B-probe sample points on the |B| panel.
    ax[1].scatter(r["probe_x"], r["B_probe"], color="k", zorder=5, label="B-probe")

    fig.suptitle(f"Synthetic diagnostics  t={int(r['t_val'])}  "
                 f"resolution {res_um:.0f} µm = {float(r['resolution_norm']):.1f} c/ωpe",
                 fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = os.path.join(out_dir, f"synthetic_diagnostics_t{int(r['t_val']):06d}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
