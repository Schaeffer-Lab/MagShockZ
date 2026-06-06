"""scripts/convergence_scan.py — upstream anomalous-heating convergence scan.

The science question this answers: how low can the resolution (dx) and particle
count (ppc) go before the cold UPSTREAM plasma starts to heat *anomalously* --
i.e. from the PIC finite-grid instability (under-resolved Debye length) and
particle shot noise, rather than from physics?  A well-resolved run keeps the
undisturbed inflow at its initial temperature; an under-resolved one shows the
upstream temperature climb with time.

For every run in the scan this measures the upstream electron (and ion)
temperature versus time, in a spatial window held ahead of the shock, and
produces:

  1. heating curves  T_up(t)/T_up(0)  vs t, one per run  -> when/whether it heats
  2. threshold plot  heating metric   vs dx (and vs cells-per-lambda_De)
                     -> the dx/ppc below which the upstream stays cold

It reuses the established analysis paradigm: osh5io for reads,
analysis_utils.detect_layout for dimension-agnostic layout, MagShockZRun for
unit context, temperature_anisotropy.temperature_profile (a central p-moment,
so bulk drift is removed) for T(x).  Results land under results/convergence/.

Usage
-----
    python scripts/convergence_scan.py \\
        [--glob 'input_files/magshockz_rqm100_dx*_ppc500_g20.1d'] \\
        [--stride 4] [--up-lo 0.55] [--up-hi 0.90] [--output-dir DIR]

Runs that have not started (no phase-space dumps yet) are skipped with a note,
so this can be re-run as the scan fills in.
"""

import argparse
import glob
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

import osh5io
import analysis_utils
from analysis_utils import phase_path, density_path, axis_values, detect_layout
import temperature_anisotropy as ta

ME_C2_EV = 510998.95  # electron rest energy [eV]; T[eV] = T[m_e c^2] * ME_C2_EV


# ---------------------------------------------------------------------------
# Run discovery
# ---------------------------------------------------------------------------

def discover_runs(pattern):
    """Return [{name, sim_dir, dx, ppc, node, gyrotime}], sorted coarse->fine."""
    runs = []
    for sim_dir in sorted(glob.glob(pattern)):
        if not os.path.isdir(sim_dir):
            continue
        man = os.path.join(sim_dir, "run_manifest.yaml")
        dx = ppc = node = gyro = None
        if os.path.exists(man):
            with open(man) as f:
                m = yaml.safe_load(f)
            cli = m.get("cli_command", "")
            dx = _flag(cli, "dx", float)
            ppc = _flag(cli, "ppc", int)
            node = _flag(cli, "node_number", int)
            gyro = (m.get("derived") or {}).get("gyrotime_wpe_inv")
        runs.append(dict(name=os.path.basename(sim_dir), sim_dir=sim_dir,
                         dx=dx, ppc=ppc, node=node, gyrotime=gyro))
    runs.sort(key=lambda r: (r["dx"] if r["dx"] is not None else 1e9), reverse=True)
    return runs


def _flag(cli, name, cast):
    m = re.search(rf"--{name}\s+([^\s]+)", cli)
    return cast(m.group(1)) if m else None


def phase_dumps(sim_dir, sp):
    """Sorted list of available p1x1 dump indices for a species."""
    files = glob.glob(phase_path(sim_dir, "p1x1", sp, 0).replace("000000", "*"))
    idx = sorted(int(re.search(r"(\d+)\.h5$", f).group(1)) for f in files)
    return idx


# ---------------------------------------------------------------------------
# Upstream temperature series
# ---------------------------------------------------------------------------

def total_T_profile(sim_dir, sp, t, rqm):
    """Scalar temperature T(x) = mean over the 3 momentum components [m_e c^2].

    Returns (x, T, time) or (None, None, None) if a component dump is missing.
    """
    comps = []
    x = time = None
    for c in (1, 2, 3):
        path = phase_path(sim_dir, f"p{c}x1", sp, t)
        if not os.path.exists(path):
            return None, None, None
        ps = osh5io.read_h5(path)
        comps.append(np.asarray(ta.temperature_profile(ps, rqm, f"p{c}")))
        if x is None:
            x = axis_values(ps, next(i for i, a in enumerate(ps.axes) if a.name != f"p{c}"))
            time = float(ps.run_attrs["TIME"][0])
    return x, np.mean(comps, axis=0), time


def upstream_series(sim_dir, sp, rqm, idx_list, up_lo, up_hi):
    """T_up(t) averaged over the fixed upstream window [up_lo, up_hi]*L.

    Returns (times[1/wpe], T_up[m_e c^2]) over the available dumps.
    """
    times, Tup = [], []
    for t in idx_list:
        x, T, tm = total_T_profile(sim_dir, sp, t, rqm)
        if x is None:
            continue
        L = x.max()
        win = (x >= up_lo * L) & (x <= up_hi * L)
        if not win.any():
            continue
        times.append(tm)
        Tup.append(float(np.nanmean(T[win])))
    return np.array(times), np.array(Tup)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glob", default="input_files/magshockz_rqm100_dx*_ppc500_g20.1d",
                    help="Glob (relative to project root) for the scan run dirs.")
    ap.add_argument("--stride", type=int, default=4,
                    help="Use every Nth phase dump (default 4; reads dominate runtime).")
    ap.add_argument("--up-lo", type=float, default=0.55, dest="up_lo",
                    help="Upstream window lower edge as fraction of box length.")
    ap.add_argument("--up-hi", type=float, default=0.90, dest="up_hi",
                    help="Upstream window upper edge (keep below the vpml boundary).")
    ap.add_argument("--e-species", default="e")
    ap.add_argument("--i-species", default="al", help="Ambient (upstream) ion species.")
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    proj = os.path.abspath(os.path.join(_HERE, ".."))
    out = args.output_dir or os.path.join(proj, "results", "convergence")
    os.makedirs(out, exist_ok=True)

    runs = discover_runs(os.path.join(proj, args.glob))
    if not runs:
        print(f"No runs matched {args.glob!r}")
        return

    # rqm per species from any deck that exists (intrinsic to the setup).
    rqm = {args.e_species: -1.0, args.i_species: None}
    series = []     # collect (run, tE, TE, tI, TI) first so we can match a common time
    summary = []
    fig, (axE, axI) = plt.subplots(1, 2, figsize=(13, 5))
    cmap = plt.get_cmap("viridis")

    # ---- pass 1: collect upstream T(t) series for every run with data ----
    active = [r for r in runs if r["dx"] is not None]
    for r in active:
        sim_dir = r["sim_dir"]
        idx = phase_dumps(sim_dir, args.e_species)[:: max(args.stride, 1)]
        if not idx:
            print(f"  skip {r['name']}: no phase dumps yet")
            continue
        detect_layout(sim_dir)  # validates the run is readable / dimension

        if rqm[args.i_species] is None:
            deck = os.path.join(sim_dir, r["name"])
            try:
                run = analysis_utils.MagShockZRun(deck, norm_density=None)
                rqm[args.i_species] = run.rqm_of(args.i_species)
            except Exception:
                rqm[args.i_species] = {"al": 38.0, "si": 36.0}.get(args.i_species, 1.0)

        tE, TE = upstream_series(sim_dir, args.e_species, rqm[args.e_species], idx, args.up_lo, args.up_hi)
        tI, TI = upstream_series(sim_dir, args.i_species, rqm[args.i_species], idx, args.up_lo, args.up_hi)
        if TE.size < 2:
            print(f"  skip {r['name']}: <2 usable dumps")
            continue
        series.append((r, tE, TE, tI, TI))

    if not series:
        print("No runs had usable data yet.")
        return

    # Fair comparison: evaluate the heating factor at a time reached by ALL runs.
    t_common = min(tE[-1] for (_, tE, _, _, _) in series)
    print(f"\n  common comparison time t = {t_common:.0f} 1/wpe "
          f"({t_common / (series[0][0]['gyrotime'] or 1.0):.2f} gyroperiods)")

    # ---- pass 2: plot heating curves (log y -- heating spans orders of magnitude) ----
    for k, (r, tE, TE, tI, TI) in enumerate(series):
        color = cmap(k / max(len(series) - 1, 1))
        gyro = r["gyrotime"] or 1.0
        lab = f"dx={r['dx']:.4g} ppc={r['ppc']} (N={r['node']})"
        TE0, TI0 = TE[0], (TI[0] if TI.size else np.nan)
        cells_per_lambdaDe = np.sqrt(TE0) / r["dx"]          # lambda_De/dx = v_te/dx
        heatE = float(np.interp(t_common, tE, TE)) / TE0     # at the common time
        heatI = float(np.interp(t_common, tI, TI)) / TI0 if TI.size >= 2 else np.nan

        axE.semilogy(tE / gyro, TE / TE0, "-o", ms=3, color=color, label=lab)
        if TI.size >= 2:
            axI.semilogy(tI / gyro, TI / TI0, "-o", ms=3, color=color, label=lab)

        summary.append(dict(name=r["name"], dx=r["dx"], ppc=r["ppc"], node=r["node"],
                            TE0_eV=TE0 * ME_C2_EV, heatE=heatE, heatI=heatI,
                            cells_per_lambdaDe=cells_per_lambdaDe,
                            t_common_wpe=t_common, t_final_wpe=float(tE[-1])))
        print(f"  {lab}: T_e0={TE0*ME_C2_EV:.1f} eV, lambda_De/dx={cells_per_lambdaDe:.3f} cells, "
              f"upstream T_e heated x{heatE:.3g} at t_common")

    for ax, ttl in ((axE, "electron"), (axI, "ion (ambient)")):
        ax.axhline(1.0, color="k", lw=0.8, ls=":")
        ax.axvline(t_common / (series[0][0]["gyrotime"] or 1.0), color="grey", lw=0.8, ls="--",
                   label="common compare time")
        ax.set_xlabel(r"$t$ [gyroperiods]")
        ax.set_ylabel(r"$T_\mathrm{up}(t)\,/\,T_\mathrm{up}(0)$")
        ax.set_title(f"upstream {ttl} anomalous heating")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    f1 = os.path.join(out, "upstream_heating_curves.png")
    fig.savefig(f1, dpi=150)
    plt.close(fig)

    # ---- threshold plot: heating at t_common vs dx, lambda_De/dx annotated ----
    s = sorted(summary, key=lambda d: d["dx"])
    dx = np.array([d["dx"] for d in s])
    heatE = np.array([d["heatE"] for d in s])
    fig2, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(dx, heatE, "-o", color="C3")
    ax.axhline(1.0, color="k", ls=":", lw=0.8, label="no heating")
    for d in s:
        ax.annotate(f"{d['cells_per_lambdaDe']:.2f}", (d["dx"], d["heatE"]),
                    textcoords="offset points", xytext=(0, 7), fontsize=8, ha="center")
    ax.set_xlabel(r"$\Delta x$ [$c/\omega_{pe}$]")
    ax.set_ylabel(rf"upstream $T_e$ heating factor at $t={t_common:.0f}\,\omega_{{pe}}^{{-1}}$")
    ax.set_title("anomalous-heating threshold vs resolution\n(annotated: $\\lambda_{De}/\\Delta x$ in cells)")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig2.tight_layout()
    f2 = os.path.join(out, "heating_vs_resolution.png")
    fig2.savefig(f2, dpi=150)
    plt.close(fig2)

    np.savez(os.path.join(out, "convergence_summary.npz"), summary=np.array(summary, dtype=object))
    print(f"\nWrote:\n  {f1}\n  {f2}\n  {os.path.join(out, 'convergence_summary.npz')}")


if __name__ == "__main__":
    main()
