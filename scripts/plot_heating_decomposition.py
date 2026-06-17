"""Plot the shock heating decomposition from a precomputed .npz.

Usage
-----
    python scripts/plot_heating_decomposition.py \\
        --results results/<run>/heating_decomposition_t000360.npz [--output-dir figures/]

Six panels:
  (0,0) temperature profiles T_e, T_i near the shock, with the MHD total-T jump
  (0,1) cross-shock potential e*phi(x)
  (0,2) energy budget bars (upstream vs downstream)
  (1,0) FPC electron velocity-space signature (heating fingerprint)
  (1,1) FPC ion velocity-space signature
  (1,2) measured heating partition (ΔT_e vs ΔT_i) + anomalous-total annotation
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

from analysis_utils import load_results


def _window(x, x_shock, half):
    return (x >= x_shock - half) & (x <= x_shock + half)


def plot_temperatures(r, ax, half):
    x, xs = r["x_axis"], float(r["x_shock"])
    m = _window(x, xs, half)
    ax.semilogy(x[m], r["T_iso_e"][m], label="T_e (measured)")
    ax.semilogy(x[m], r["T_iso_i"][m], label="T_i (measured)")
    # MHD-predicted downstream total temperature (heat_tot_adiabatic = T_up*T_factor).
    ax.axhline(float(r["heat_tot_adiabatic"]),
               color="k", ls=":", lw=1.2, label="RH total T (MHD)")
    ax.axvline(xs, color="0.4", ls="--", lw=1)
    ax.set_xlabel("x [c/ωpe]"); ax.set_ylabel("T [mₑc²]")
    ax.set_title("Temperatures vs MHD baseline"); ax.legend(fontsize=8); ax.grid(alpha=0.3)


def plot_potential(r, ax, half):
    x, xs = r["x_axis"], float(r["x_shock"])
    m = _window(x, xs, half)
    ax.plot(x[m], r["e_phi_profile"][m], color="C3")
    ax.axvline(xs, color="0.4", ls="--", lw=1)
    ax.set_xlabel("x [c/ωpe]"); ax.set_ylabel("e·φ [mₑc²]")
    ax.set_title(f"Cross-shock potential  (e·Δφ/½m_iv² = {float(r['reflection_parameter']):.2f})")
    ax.grid(alpha=0.3)


def plot_budget(r, ax):
    labels = ["ram", "th_e", "th_i", "B", "E"]
    keys = ["ram", "thermal_e", "thermal_i", "B_field", "E_field"]
    up = [float(r[f"up_{k}"]) for k in keys]
    dn = [float(r[f"dn_{k}"]) for k in keys]
    xi = np.arange(len(labels)); w = 0.38
    ax.bar(xi - w / 2, up, w, label="upstream")
    ax.bar(xi + w / 2, dn, w, label="downstream")
    ax.set_xticks(xi); ax.set_xticklabels(labels)
    ax.set_ylabel("energy density [n₀mₑc²]")
    ax.set_title("Energy budget"); ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)


def plot_fpc(r, ax, species):
    u = r[f"fpc_u_{species}"]; spec = r[f"fpc_spectrum_{species}"]
    ax.plot(u, spec, color="C4")
    ax.axhline(0, color="0.6", lw=0.8); ax.axvline(0, color="0.6", lw=0.8)
    ax.set_xlabel("u [c]"); ax.set_ylabel("⟨C_E⟩ (layer)")
    name = "electron" if species == "e" else "ion"
    ax.set_title(f"FPC {name} velocity signature"); ax.grid(alpha=0.3)


def plot_partition(r, ax):
    dT_e, dT_i = float(r["dT_e"]), float(r["dT_i"])
    ax.bar(["ΔT_e", "ΔT_i"], [dT_e, dT_i], color=["C0", "C1"])
    ax.set_ylabel("downstream − upstream T [mₑc²]")
    anom = 100 * float(r["heat_tot_anomalous_frac"])
    fe, fi = 100 * float(r["frac_heat_e"]), 100 * float(r["frac_heat_i"])
    ax.set_title(f"Heating partition  e {fe:.0f}% / i {fi:.0f}%\n"
                 f"anomalous (vs MHD) {anom:+.0f}%")
    ax.grid(axis="y", alpha=0.3)


def main():
    parser = argparse.ArgumentParser(description="Plot shock heating decomposition.")
    parser.add_argument("--results", required=True, help="Path to .npz results file.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--half-window", type=float, default=250.0,
                        help="Half-width [c/ωpe] of the near-shock plotting window.")
    args = parser.parse_args()

    r = load_results(args.results)
    out_dir = args.output_dir or os.path.dirname(os.path.abspath(args.results))
    os.makedirs(out_dir, exist_ok=True)
    half = args.half_window

    fig, ax = plt.subplots(2, 3, figsize=(18, 9))
    plot_temperatures(r, ax[0, 0], half)
    plot_potential(r, ax[0, 1], half)
    plot_budget(r, ax[0, 2])
    plot_fpc(r, ax[1, 0], "e")
    plot_fpc(r, ax[1, 1], "i")
    plot_partition(r, ax[1, 2])

    theta = float(r["theta_bn_deg"])
    fig.suptitle(f"Heating decomposition  t={int(r['t_val'])}  "
                 f"θ_Bn={theta:.0f}°  M_A={float(r['mach_a']):.1f}  "
                 f"r_meas={float(r['r_measured']):.2f}/r_RH={float(r['r_RH']):.2f}",
                 fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    out_path = os.path.join(out_dir, f"heating_decomposition_t{int(r['t_val']):06d}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
