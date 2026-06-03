"""Plot energy partition from a precomputed .npz results file.

Usage
-----
    python scripts/plot_energy_partition.py \\
        --results results/perlmutter_1.3.1d/energy_partition_t000360.npz \\
        [--output-dir figures/]

The script produces one figure with two panels:
  left  — energy density profiles vs x
  right — upstream/downstream bar chart
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np


def load_results(path: str) -> dict:
    """Load .npz and unwrap 0-d arrays to plain scalars."""
    d = np.load(path, allow_pickle=True)
    return {k: (d[k].item() if d[k].ndim == 0 else d[k]) for k in d.files}


def plot_profiles(r: dict, ax: plt.Axes) -> None:
    x = r["x_axis"]
    x_shock = r["x_shock"]
    x_ds = r["x_downstream_start"]

    u_total = r["u_ram"] + r["u_th"] + r["u_B"] + r["u_E"]
    ax.semilogy(x, r["u_ram"], label="Ram")
    ax.semilogy(x, r["u_th"], label="Thermal")
    ax.semilogy(x, r["u_B"], label=f"B field ({r['field_mode']})")
    ax.semilogy(x, r["u_E"], label="E field")
    ax.semilogy(x, u_total, color="k", ls="--", lw=1.5, label="Total")
    ax.axvline(x_ds, color="gray", ls="--", lw=1, label=f"downstream start {x_ds:.0f}")
    ax.axvline(x_shock, color="k", ls="--", lw=1, label=f"shock {x_shock:.1f}")
    ax.axvspan(x.min(), x_ds, color="gray", alpha=0.12, label="piston")

    ax.set_xlabel("x [c/ωpe]")
    ax.set_ylabel("Energy density [n₀ mₑ c²]")
    ax.set_title(f"Energy density profiles, t={r['t_val']}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xlim(x_ds, x_shock + 100)

    downstream = (x >= x_ds) & (x <= x_shock)
    if downstream.any():
        ax.set_ylim(0, np.nanmax(u_total[downstream]) * 1.2)


def plot_partition_bars(r: dict, ax: plt.Axes) -> None:
    labels = ["Ram", "Thermal", "B field", "E field"]
    keys = ["ram", "thermal", "B_field", "E_field"]
    up_vals = [r[f"upstream_{k}"] for k in keys]
    dn_vals = [r[f"downstream_{k}"] for k in keys]

    x = np.arange(len(labels))
    w = 0.38
    ax.bar(x - w / 2, up_vals, w, label="Upstream")
    ax.bar(x + w / 2, dn_vals, w, label="Downstream")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean energy density [n₀ mₑ c²]")
    ax.set_title(f"Energy partition ({r['field_mode']} B), t={r['t_val']}")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)


def main():
    parser = argparse.ArgumentParser(description="Plot shock energy partition.")
    parser.add_argument("--results", required=True, help="Path to .npz results file.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save figures. Defaults to same directory as results file.",
    )
    args = parser.parse_args()

    r = load_results(args.results)
    out_dir = args.output_dir or os.path.dirname(os.path.abspath(args.results))
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plot_profiles(r, axes[0])
    plot_partition_bars(r, axes[1])
    plt.tight_layout()

    out_path = os.path.join(out_dir, f"energy_partition_t{r['t_val']:06d}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
