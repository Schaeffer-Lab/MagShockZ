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
  3. T(x) profiles   two figures per run (linear + log y), ion + electron T(x)
                     at several times, upstream window shaded  -> *where* the
                     heating appears (a clean run stays flat-and-cold ahead of
                     the shock; an under-resolved one heats across the whole
                     upstream).  The log-y version spreads the small, bunched
                     upstream values so slow upstream heating is legible.

It reuses the established analysis paradigm: osh5io for reads,
analysis_utils.detect_layout for dimension-agnostic layout, MagShockZRun for
unit context, temperature_anisotropy.temperature_profile (a central p-moment,
so bulk drift is removed) for T(x), and osh5vis for the T(x) profile plots
(axis labels/units sourced from each profile's H5Data metadata).  The derived
heating-ratio and threshold-vs-dx plots stay raw matplotlib.  Results land
under results/convergence/.

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
import osh5def
import osh5vis
import analysis_utils
import plot_style
from analysis_utils import diag_path, axis_values, detect_layout
import temperature_anisotropy as ta

ME_C2_EV = 510998.95  # electron rest energy [eV]; T[eV] = T[m_e c^2] * ME_C2_EV


# ---------------------------------------------------------------------------
# Run discovery
# ---------------------------------------------------------------------------

def discover_runs(pattern):
    """Return [{name, sim_dir, deck, dx, ppc, node, gyrotime}], sorted coarse->fine."""
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
        deck = find_deck(sim_dir, os.path.basename(sim_dir))
        if dx is None and deck:                       # no manifest -> read the deck
            dd = parse_deck(deck)
            dx, ppc, node = dd.get("dx"), dd.get("ppc"), dd.get("node")
        runs.append(dict(name=os.path.basename(sim_dir), sim_dir=sim_dir, deck=deck,
                         dx=dx, ppc=ppc, node=node, gyrotime=gyro))
    runs.sort(key=lambda r: (r["dx"] if r["dx"] is not None else 1e9), reverse=True)
    return runs


def _flag(cli, name, cast):
    m = re.search(rf"--{name}\s+([^\s]+)", cli)
    return cast(m.group(1)) if m else None


def find_deck(sim_dir, name):
    """Locate the OSIRIS input deck inside a run dir (name-match, then *.Nd, then os-stdin)."""
    cands = [os.path.join(sim_dir, name)]
    for pat in ("*.1d", "*.2d", "*.3d", "os-stdin", "input.deck"):
        cands += sorted(glob.glob(os.path.join(sim_dir, pat)))
    for c in cands:
        if os.path.isfile(c):
            return c
    return None


def parse_deck(deck):
    """Pull dx, ppc, node from an OSIRIS deck when no run_manifest.yaml exists."""
    with open(deck) as f:
        txt = f.read()

    def grab(key, cast):
        m = re.search(rf"{key}\s*(?:\([^)]*\))?\s*=\s*([-\d.eE+]+)", txt)
        return cast(m.group(1)) if m else None

    nx = grab("nx_p", float)
    xmax = grab("xmax", float)
    xmin = grab("xmin", float) or 0.0
    dx = (xmax - xmin) / nx if (nx and xmax is not None) else None
    return dict(dx=dx, ppc=grab("num_par_x", int), node=grab("node_number", int))


def phase_dumps(sim_dir, sp):
    """Sorted list of available p1x1 dump indices for a species."""
    files = glob.glob(diag_path(sim_dir, "p1x1", 0, sp).replace("000000", "*"))
    idx = sorted(int(re.search(r"(\d+)\.h5$", f).group(1)) for f in files)
    return idx


# ---------------------------------------------------------------------------
# Upstream temperature series
# ---------------------------------------------------------------------------

def total_T_profile(sim_dir, sp, t, rqm):
    """Scalar temperature T(x) = mean over the available momentum components [m_e c^2].

    Averages over whichever of p1x1/p2x1/p3x1 are present (some runs only dump a
    subset), so T is the mean per-component temperature.  The mean is re-wrapped
    as an osh5def.H5Data carrying the phase-space spatial axis + run_attrs (TIME)
    and ``UNITS="m_e c^2"`` data_attrs — exactly like overview.temperature_frame —
    so osh5vis can label the profile from its own metadata.  Returns
    (x, T_h5, time), or (None, None, None) if no component dump exists.
    """
    comps = []
    x = time = x_axis = run_attrs = None
    for c in (1, 2, 3):
        path = diag_path(sim_dir, f"p{c}x1", t, sp)
        if not os.path.exists(path):
            continue
        ps = osh5io.read_h5(path)
        comps.append(np.asarray(ta.temperature_profile(ps, rqm, f"p{c}")))
        if x is None:
            sx = next(i for i, a in enumerate(ps.axes) if a.name != f"p{c}")
            x_axis, run_attrs = ps.axes[sx], ps.run_attrs
            x = axis_values(ps, sx)
            time = float(ps.run_attrs["TIME"][0])
    if not comps:
        return None, None, None
    T = osh5def.H5Data(
        np.mean(comps, axis=0),
        data_attrs={"NAME": f"T_{sp}", "LONG_NAME": fr"T_{{{sp}}}", "UNITS": "m_e c^2"},
        run_attrs=run_attrs,
        axes=[x_axis],
    )
    return x, T, time


def collect_profiles(sim_dir, sp, rqm, idx_list):
    """Read T(x) once per dump for a species.

    Returns (times[1/wpe], x[c/wpe], profiles) where profiles is a list of the
    scalar T(x) arrays [m_e c^2], one per readable dump.  Reading once here lets
    both the upstream time-series and the T(x)-vs-time plot share the same reads.
    """
    times, profs, xref = [], [], None
    for t in idx_list:
        x, T, tm = total_T_profile(sim_dir, sp, t, rqm)
        if x is None:
            continue
        times.append(tm)
        profs.append(T)
        xref = x
    return np.array(times), xref, profs


def shock_x(cfg, t_sim):
    """Shock-front position [c/wpe] from the config's linear trajectory.

    Uses the tuned ``shock.x_shock_0`` + ``shock.v_shock`` * t_sim (the same
    trajectory tune_shock.py writes), so the upstream window can follow the front
    at *every* dump without needing a per-dump ``dump_params`` entry.
    """
    sh = cfg["shock"]
    return sh["x_shock_0"] + sh["v_shock"] * t_sim


def upstream_medians(x, profs, times, *, cfg=None, up_lo=0.55, up_hi=0.90):
    """T_up(t): the *median* T over the upstream window, one value per dump.

    The median (not the mean) is used so the upstream estimate is robust to the
    occasional hot-particle spike / noisy cell in the under-resolved runs, which
    would otherwise drag a mean upward and mimic heating.

    With ``cfg`` the window *tracks the shock*: a fixed-width band of
    ``upstream_window_ncells`` cells just ahead of ``shock_x(cfg, t_sim)``
    (via :func:`analysis_utils.window_masks`).  Without a config it falls back to
    the legacy fixed fraction ``[up_lo, up_hi] * L`` of the box.
    """
    if x is None or not profs:
        return np.array([])
    if cfg is not None:
        up_ncells = int(cfg.get("upstream_window_ncells", 200))
        dx = float(x[1] - x[0])
        vals = []
        for T, t_sim in zip(profs, times):
            up, _ = analysis_utils.window_masks(x, shock_x(cfg, t_sim), dx, up_ncells, 0)
            vals.append(float(np.nanmedian(np.asarray(T)[up])) if up.any() else np.nan)
        return np.array(vals)
    L = x.max()
    win = (x >= up_lo * L) & (x <= up_hi * L)
    if not win.any():
        return np.array([])
    return np.array([float(np.nanmedian(np.asarray(T)[win])) for T in profs])


def plot_profiles_over_time(r, xE, tE, profE, xI, tI, profI, n_profiles, out,
                            *, gyro_e, gyro_i, unit_e, unit_i,
                            cfg=None, up_lo=0.55, up_hi=0.90, logy=False):
    """One figure per run: ion + electron T(x) [eV] at several times.

    Times are sampled evenly across the available dumps and colored by time, so
    the spatial signature of any upstream heating is visible directly.  Each panel
    is labelled in its own gyroperiod (``gyro_i``/``unit_i`` for the ions,
    ``gyro_e``/``unit_e`` for the electrons).

    The upstream region is drawn the way it is *measured*: with ``cfg`` the shock
    front (and the window just ahead of it) is marked per plotted time with a
    vertical line in that time's colour, so you see the window advance with the
    shock; without a config a static ``[up_lo, up_hi]*L`` band is shaded instead.

    With ``logy=True`` the temperature axis is logarithmic and the figure is
    saved as ``tprofiles_logy_<run>.png``; this spreads out the small, bunched
    upstream values so the slow upstream heating is legible (the linear version
    is dominated by the hot shock/downstream peak).
    """
    fig, (axI, axE) = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
    cmap = plt.get_cmap("plasma")

    for ax, x, t, profs, ttl, gyro, unit in (
            (axI, xI, tI, profI, "ion (ambient)", gyro_i, unit_i),
            (axE, xE, tE, profE, "electron", gyro_e, unit_e)):
        if x is None or len(profs) == 0:
            ax.set_title(f"{ttl}: no data")
            continue
        L = x.max()
        pick = np.unique(np.linspace(0, len(profs) - 1, min(n_profiles, len(profs))).astype(int))
        tmin, tmax = t[pick].min(), t[pick].max()
        span = (tmax - tmin) or 1.0
        title = f"{ttl} $T(x)$ over time"
        for j in pick:
            # Re-label a copy in eV for display; osh5vis sources the x-axis label
            # from the profile's own spatial axis and the y-axis from data_attrs.
            color = cmap((t[j] - tmin) / span)
            Tj = profs[j] * ME_C2_EV
            Tj.data_attrs = dict(Tj.data_attrs, UNITS="eV")
            osh5vis.osplot1d(Tj, ax=ax, color=color,
                             label=fr"$t={t[j] / gyro:.2f}\ {unit}$",
                             title=title, show_time=False)
            if cfg is not None:                      # shock-tracked upstream window
                ax.axvline(shock_x(cfg, t[j]), color=color, lw=0.8, ls="--")
        if cfg is not None:
            ax.axvline(np.nan, color="grey", lw=0.8, ls="--",
                       label="shock front (upstream = right of line)")
        else:
            ax.axvspan(up_lo * L, up_hi * L, color="grey", alpha=0.12, label="upstream window")
        if logy:
            ax.set_yscale("log")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=0.3)

    suffix = " (log y)" if logy else ""
    fig.suptitle(f"{r['name']}  (dx={r['dx']:.4g}, ppc={r['ppc']}, N={r['node']}){suffix}")
    fig.tight_layout()
    fpath = os.path.join(out, f"tprofiles{'_logy' if logy else ''}_{r['name']}.png")
    fig.savefig(fpath, dpi=150)
    plt.close(fig)
    return fpath


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None,
                    help="Analysis config YAML for a SINGLE run (e.g. "
                         "config/magshockz_rqm100_dx0.1.yaml).  In this mode the ion "
                         "time axis uses the config's T_ci (cached t_ci or measured) and "
                         "the upstream window tracks the shock front; --glob is ignored.")
    ap.add_argument("--glob", default="input_files/magshockz_rqm100_dx*_ppc500_g20.1d",
                    help="Glob (relative to project root) for the scan run dirs.")
    ap.add_argument("--stride", type=int, default=4,
                    help="Use every Nth phase dump (default 4; reads dominate runtime).")
    ap.add_argument("--up-lo", type=float, default=0.55, dest="up_lo",
                    help="Upstream window lower edge as fraction of box length.")
    ap.add_argument("--up-hi", type=float, default=0.90, dest="up_hi",
                    help="Upstream window upper edge (keep below the vpml boundary).")
    ap.add_argument("--n-profiles", type=int, default=6, dest="n_profiles",
                    help="Number of time snapshots in each run's T(x) profile figure.")
    ap.add_argument("--gyrotime", type=float, default=None,
                    help="Electron gyroperiod [1/wpe] for the time axis when a run "
                         "has no run_manifest.yaml (e.g. 268 for perlmutter_1.3.1d). "
                         "Falls back to 1/wpe units if unset.")
    ap.add_argument("--e-species", default="e")
    ap.add_argument("--i-species", default="al", help="Ambient (upstream) ion species.")
    ap.add_argument("--exclude", default=[], nargs="*",
                    help="Substrings of run names to drop from the dataset, e.g. nonphysical "
                         "blow-ups whose under-resolved upstream is not a convergence data point "
                         "(default excludes the dx0.3 run, which blew up nonphysically).")
    ap.add_argument("--output-dir", default=None)
    plot_style.add_publication_arg(ap)
    args = ap.parse_args()
    plot_style.apply(args.publication)

    proj = os.path.abspath(os.path.join(_HERE, ".."))

    # Config mode: a single run, with the shock-tracked upstream window and the
    # T_ci ion clock both sourced from the analysis config (the machinery built in
    # plot_style.build_units + analysis_utils.window_masks).
    cfg = T_ci = None
    if args.config:
        cfg = analysis_utils.load_config(os.path.abspath(args.config))
        disp = plot_style.build_units("ion", cfg=cfg, config_path=os.path.abspath(args.config))
        T_ci = disp.time_factor                       # upstream ion gyroperiod [1/wpe]
        runs = discover_runs(cfg["sim_dir"])
        default_out = os.path.join(proj, "results", "convergence",
                                   os.path.splitext(os.path.basename(args.config))[0])
    else:
        runs = discover_runs(os.path.join(proj, args.glob))
        default_out = os.path.join(proj, "results", "convergence")

    out = args.output_dir or default_out
    os.makedirs(out, exist_ok=True)

    if not runs:
        print(f"No runs matched {args.config or args.glob!r}")
        return
    if args.exclude:
        keep = [r for r in runs if not any(x in r["name"] for x in args.exclude)]
        for r in runs:
            if r not in keep:
                print(f"  excluding {r['name']} ({', '.join(args.exclude)})")
        runs = keep

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

        if r["gyrotime"] is None and args.gyrotime:
            r["gyrotime"] = args.gyrotime

        if rqm[args.i_species] is None:
            deck = r["deck"] or os.path.join(sim_dir, r["name"])
            try:
                run = analysis_utils.MagShockZRun(deck, norm_density=None)
                rqm[args.i_species] = run.rqm_of(args.i_species)
            except Exception:
                rqm[args.i_species] = {"al": 38.0, "si": 36.0}.get(args.i_species, 1.0)

        rqm_i = abs(rqm[args.i_species] or 1.0)
        if T_ci is not None:                          # config: T_ci ion clock
            gyro_i, gyro_e = T_ci, T_ci / rqm_i
            unit_i, unit_e = r"T_{ci}", r"T_{ce}"
        else:                                         # glob: manifest electron gyro
            gyro_e = r["gyrotime"] or 1.0
            gyro_i = gyro_e * rqm_i
            unit_i, unit_e = r"\mathrm{ion\,gyro}", r"\mathrm{e\,gyro}"
        r["_gyro_e"], r["_gyro_i"] = gyro_e, gyro_i   # reused in the heating-curve pass

        tE, xE, profE = collect_profiles(sim_dir, args.e_species, rqm[args.e_species], idx)
        tI, xI, profI = collect_profiles(sim_dir, args.i_species, rqm[args.i_species], idx)
        TE = upstream_medians(xE, profE, tE, cfg=cfg, up_lo=args.up_lo, up_hi=args.up_hi)
        TI = upstream_medians(xI, profI, tI, cfg=cfg, up_lo=args.up_lo, up_hi=args.up_hi)
        if TE.size < 2:
            print(f"  skip {r['name']}: <2 usable dumps")
            continue
        series.append((r, tE, TE, tI, TI))

        for logy in (False, True):
            fp = plot_profiles_over_time(r, xE, tE, profE, xI, tI, profI,
                                         args.n_profiles, out,
                                         gyro_e=gyro_e, gyro_i=gyro_i,
                                         unit_e=unit_e, unit_i=unit_i,
                                         cfg=cfg, up_lo=args.up_lo, up_hi=args.up_hi,
                                         logy=logy)
            print(f"  wrote {os.path.basename(fp)}")

    if not series:
        print("No runs had usable data yet.")
        return

    # The ion clock (gyro_i) is the upstream ion gyroperiod: T_ci from the config
    # in --config mode, else rqm_i * the manifest electron gyroperiod.  Both were
    # resolved per run in pass 1 and stashed on the run dict.
    gyro_e0, gyro_i0 = series[0][0]["_gyro_e"], series[0][0]["_gyro_i"]
    unit_e0, unit_i0 = (r"T_{ce}", r"T_{ci}") if T_ci is not None \
        else (r"\mathrm{e\,gyro}", r"\mathrm{ion\,gyro}")

    # Fair comparison: evaluate the heating factor at a time reached by ALL runs.
    t_common = min(tE[-1] for (_, tE, _, _, _) in series)
    print(f"\n  common comparison time t = {t_common:.0f} 1/wpe "
          f"({t_common / gyro_e0:.2f} e-gyroperiods = {t_common / gyro_i0:.2f} ion gyroperiods)")

    # ---- pass 2: plot heating curves (log y -- heating spans orders of magnitude) ----
    for k, (r, tE, TE, tI, TI) in enumerate(series):
        color = cmap(k / max(len(series) - 1, 1))
        gyro_e, gyro_i = r["_gyro_e"], r["_gyro_i"]
        lab = f"dx={r['dx']:.4g} ppc={r['ppc']} (N={r['node']})"
        TE0, TI0 = TE[0], (TI[0] if TI.size else np.nan)
        cells_per_lambdaDe = np.sqrt(TE0) / r["dx"]          # lambda_De/dx = v_te/dx
        heatE = float(np.interp(t_common, tE, TE)) / TE0     # at the common time
        heatI = float(np.interp(t_common, tI, TI)) / TI0 if TI.size >= 2 else np.nan

        axE.semilogy(tE / gyro_e, TE / TE0, "-o", ms=3, color=color, label=lab)
        if TI.size >= 2:
            axI.semilogy(tI / gyro_i, TI / TI0, "-o", ms=3, color=color, label=lab)

        summary.append(dict(name=r["name"], dx=r["dx"], ppc=r["ppc"], node=r["node"],
                            TE0_eV=TE0 * ME_C2_EV, heatE=heatE, heatI=heatI,
                            cells_per_lambdaDe=cells_per_lambdaDe,
                            t_common_wpe=t_common, t_final_wpe=float(tE[-1])))
        print(f"  {lab}: T_e0={TE0*ME_C2_EV:.1f} eV, lambda_De/dx={cells_per_lambdaDe:.3f} cells, "
              f"upstream heated T_e x{heatE:.3g}, T_i x{heatI:.3g} at t_common")

    for ax, ttl, gyro, xunit in ((axE, "electron", gyro_e0, unit_e0),
                                 (axI, "ion (ambient)", gyro_i0, unit_i0)):
        ax.axhline(1.0, color="k", lw=0.8, ls=":")
        ax.axvline(t_common / gyro, color="grey", lw=0.8, ls="--",
                   label="common compare time")
        ax.set_xlabel(rf"$t\ [{xunit}]$")
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
    print(f"\nWrote:\n  {f1}\n  {f2}\n  {os.path.join(out, 'convergence_summary.npz')}"
          f"\n  {len(series)} x tprofiles_<run>.png + tprofiles_logy_<run>.png "
          f"(ion+electron T(x) over time, linear & log y) in {out}")


if __name__ == "__main__":
    main()
