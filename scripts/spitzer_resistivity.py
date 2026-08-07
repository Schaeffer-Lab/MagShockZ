# -*- coding: utf-8 -*-
"""scripts/spitzer_resistivity.py — map the Spitzer resistivity of a flash2warpx slice.

WarpX's hybrid (Ohm's-law) solver takes a single scalar ``run.plasma_resistivity`` [Ohm*m]
(magnetic diffusivity ``D_m = eta / mu0``).  The physically-motivated value is the Spitzer
resistivity, which varies strongly across a MagShockZ slice because the FLASH electron
temperature ranges from ~10-40 eV in the bulk to ~1 keV in the laser channel and the mean
ionization ``Zbar`` from ~4 to ~14.  This tool reads a **generated flash2warpx tree**
(``meta.yaml`` + ``interp/{Te,n_e,Zbar}.npy`` — no yt needed) and reports:

  1. the 2D FLASH-derived Spitzer ``eta(x, y)`` map + whole-slice statistics,
  2. the local value at a handful of sampled points (incl. the hot laser channel),
  3. a Te x Zbar reference scan at the slice-mean density (order-of-magnitude intuition),
  4. (with ``--config``) a comparison against how WarpX actually treats resistivity: the
     single constant ``run.plasma_resistivity`` and the Spitzer ``eta`` implied by WarpX's
     *uniform-Te0, adiabatic* electron model (``run.{Te_eV, gamma, n0_per_m3}``) — so you can
     see where the constant is most wrong and read off a recommended replacement.

Runs in the ``analysis`` env (needs plasmapy for ``src/spitzer_resistivity.py``).

Usage
-----
    conda activate analysis
    python scripts/spitzer_resistivity.py input_files/warpx/magshockz_2d_prod \\
        [--config runs/magshockz_2d_production.warpx.yaml] \\
        [--density-floor-frac 1e-3] [--ion Si] [--output-dir results/warpx/<tree>] [--pub]
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))

from magshockz.common import plot_style
from magshockz.analysis.warpx.spitzer_resistivity import spitzer_resistivity, magnetic_diffusivity, warpx_electron_temperature

EV_PER_K = 8.617333262e-5  # eV per Kelvin (k_B / e)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_slice(tree: Path):
    """Load Te [eV], n_e [m^-3], Zbar and geometry from a flash2warpx output tree."""
    meta = yaml.safe_load((tree / "meta.yaml").read_text())
    interp = tree / "interp"
    Te_eV = np.load(interp / "Te.npy") * EV_PER_K   # extractor stores tele in Kelvin
    n_e = np.load(interp / "n_e.npy")
    zbar = np.load(interp / "Zbar.npy")
    return meta, Te_eV, n_e, zbar


def plasma_mask(n_e, floor_frac):
    """Boolean mask of plasma cells (n_e above ``floor_frac`` of its peak)."""
    return n_e > floor_frac * float(np.nanmax(n_e))


def weighted_mean(values, weights, mask):
    """Density-weighted mean of ``values`` over ``mask`` (ignoring non-finite values)."""
    m = mask & np.isfinite(values)
    w = np.clip(weights[m], 0.0, None)
    return float(np.sum(values[m] * w) / np.sum(w)) if np.sum(w) > 0 else float("nan")


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def _axis_coords(meta):
    """Physical cell-center coordinates (metres) along each plane axis."""
    n0, n1 = meta["shape"]
    a0 = np.linspace(*meta["axis0_bounds_m"], n0)
    a1 = np.linspace(*meta["axis1_bounds_m"], n1)
    return a0, a1


def sample_report(meta, Te_eV, n_e, zbar, eta, mask):
    """Return text lines describing eta at eta-percentile cells + slice statistics."""
    a0, a1 = _axis_coords(meta)
    ax0, ax1 = meta["plane_axes"]
    lines = []
    good = mask & np.isfinite(eta)
    ev = eta[good]
    lines.append(f"  plasma cells: {int(good.sum())} / {eta.size}")
    lines.append(f"  eta [Ohm*m]  min={ev.min():.3e}  median={np.median(ev):.3e}  "
                 f"max={ev.max():.3e}")
    lines.append(f"  D_m=eta/mu0 [m^2/s]  min={magnetic_diffusivity(ev.min()):.3e}  "
                 f"median={magnetic_diffusivity(np.median(ev)):.3e}  "
                 f"max={magnetic_diffusivity(ev.max()):.3e}")
    eta_nw = weighted_mean(eta, n_e, mask)
    lines.append(f"  density-weighted-mean eta = {eta_nw:.3e} Ohm*m  "
                 f"(D_m = {magnetic_diffusivity(eta_nw):.3e} m^2/s)")
    lines.append("")
    lines.append(f"  sampled cells ({ax0},{ax1} in mm):")
    lines.append(f"    {'label':<14}{ax0+' [mm]':>10}{ax1+' [mm]':>10}"
                 f"{'Te [eV]':>10}{'n_e [/m3]':>12}{'Zbar':>7}{'eta [Ohm*m]':>13}")
    # hottest (min-eta) laser-channel cell, coldest plasma (max-eta), and eta percentiles.
    idx_good = np.argwhere(good)
    ev_flat = eta[good]
    picks = {
        "hot-channel": idx_good[np.argmin(ev_flat)],
        "eta p05":     idx_good[np.argmin(np.abs(ev_flat - np.percentile(ev_flat, 5)))],
        "eta median":  idx_good[np.argmin(np.abs(ev_flat - np.median(ev_flat)))],
        "eta p95":     idx_good[np.argmin(np.abs(ev_flat - np.percentile(ev_flat, 95)))],
        "cold-dense":  idx_good[np.argmax(ev_flat)],
    }
    for label, (i, j) in picks.items():
        lines.append(f"    {label:<14}{a0[i]*1e3:>10.3f}{a1[j]*1e3:>10.3f}"
                     f"{Te_eV[i, j]:>10.1f}{n_e[i, j]:>12.2e}{zbar[i, j]:>7.2f}"
                     f"{eta[i, j]:>13.3e}")
    return lines, eta_nw


def scan_table(n_e_ref, ion):
    """Te x Zbar reference table of Spitzer eta at a fixed density (order-of-magnitude aid)."""
    Te_grid = np.array([10.0, 20.0, 40.0, 100.0, 300.0, 1000.0])
    Z_grid = np.array([4.0, 8.0, 10.0, 14.0])
    lines = [f"  Te x Zbar Spitzer eta [Ohm*m] at n_e = {n_e_ref:.2e} /m^3 (ion={ion}):",
             "    Te[eV] \\ Zbar" + "".join(f"{z:>12.0f}" for z in Z_grid)]
    for Te in Te_grid:
        row = spitzer_resistivity(np.full_like(Z_grid, Te), np.full_like(Z_grid, n_e_ref), Z_grid,
                                  ion=ion)
        lines.append(f"    {Te:>10.0f}  " + "".join(f"{e:>12.3e}" for e in row))
    return lines


def warpx_comparison(cfg, meta, n_e, zbar, eta_flash, mask, ion):
    """Compare the FLASH eta map against WarpX's constant + uniform-Te0 treatment."""
    run = cfg.get("run", {}) or {}
    eta_cfg = run.get("plasma_resistivity", None)
    Te0 = run.get("Te_eV") or meta["Te_mean_eV"]
    n0 = run.get("n0_per_m3") or meta["ne_mean_per_m3"]
    gamma = run.get("gamma", 5.0 / 3.0)

    # eta as WarpX would infer it: a single Te0 scaled adiabatically with the local density.
    Te_warpx = warpx_electron_temperature(n_e, n0, Te0, gamma)
    eta_warpx = spitzer_resistivity(Te_warpx, n_e, zbar, ion=ion)
    eta_flash_nw = weighted_mean(eta_flash, n_e, mask)
    eta_warpx_nw = weighted_mean(eta_warpx, n_e, mask)

    lines = ["  WarpX electron model: Te0=%.1f eV  n0=%.2e /m^3  gamma=%.4f" % (Te0, n0, gamma)]
    if isinstance(eta_cfg, (int, float)):
        lines.append(f"  configured constant plasma_resistivity = {eta_cfg:.3e} Ohm*m  "
                     f"(D_m = {magnetic_diffusivity(eta_cfg):.3e} m^2/s)")
        good = mask & np.isfinite(eta_flash)
        frac_below = float(np.mean(eta_flash[good] < eta_cfg))
        lines.append(f"    FLASH eta vs this constant: {frac_below*100:.0f}% of plasma cells "
                     f"are LESS resistive; ratio (n-wtd FLASH / constant) = "
                     f"{eta_flash_nw / eta_cfg:.2f}x")
    else:
        lines.append(f"  configured plasma_resistivity is non-scalar ({eta_cfg!r}); "
                     f"skipping constant comparison")
    lines.append(f"  density-weighted-mean eta:  FLASH-Te = {eta_flash_nw:.3e}   "
                 f"WarpX-Te0-model = {eta_warpx_nw:.3e} Ohm*m  "
                 f"(ratio {eta_flash_nw / eta_warpx_nw:.2f}x)")
    lines.append(f"  >> RECOMMENDED constant plasma_resistivity ~ {eta_flash_nw:.2e} Ohm*m "
                 f"(density-weighted Spitzer over the FLASH slice)")
    return lines, eta_warpx


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(meta, Te_eV, eta, mask, out_png, eta_cfg=None):
    """Two-panel figure: FLASH Te [eV] and Spitzer eta [Ohm*m] (log), over the slice."""
    ax0, ax1 = meta["plane_axes"]
    x0, x1 = np.array(meta["axis0_bounds_m"]) * 1e3   # mm
    y0, y1 = np.array(meta["axis1_bounds_m"]) * 1e3
    extent = [x0, x1, y0, y1]
    Te_plot = np.where(mask, Te_eV, np.nan).T
    eta_plot = np.where(mask & np.isfinite(eta), eta, np.nan).T

    aspect = (y1 - y0) / (x1 - x0)
    fig, axes = plt.subplots(1, 2, figsize=(8, max(4.0, 3.2 * min(aspect, 2.0))))
    for ax, data, title, cmap, log in (
        (axes[0], Te_plot, r"FLASH $T_e$ [eV]", "inferno", True),
        (axes[1], eta_plot, r"Spitzer $\eta$ [$\Omega\,$m]", "viridis_r", True),
    ):
        norm = matplotlib.colors.LogNorm() if log else None
        im = ax.imshow(data, origin="lower", extent=extent, aspect="auto", cmap=cmap, norm=norm)
        ax.set_xlabel(f"{ax0} [mm]")
        ax.set_ylabel(f"{ax1} [mm]")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if eta_cfg is not None and np.isfinite(eta_plot).any():
        # Contour where FLASH eta crosses the configured WarpX constant. X=axis0 (cols),
        # Y=axis1 (rows), Z transposed to (len(Y), len(X)) — matching the imshow orientation.
        a0, a1 = _axis_coords(meta)
        try:
            axes[1].contour(a0 * 1e3, a1 * 1e3, np.where(mask, eta, np.nan).T,
                            levels=[eta_cfg], colors="r", linewidths=1.0)
            axes[1].set_xlim(x0, x1)
            axes[1].set_ylim(y0, y1)
            axes[1].set_title(axes[1].get_title() + f"\n(red: config {eta_cfg:.1e})")
        except Exception:
            pass
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tree", help="A generated flash2warpx output tree (has meta.yaml + interp/).")
    parser.add_argument("--config", default=None,
                        help="flash2warpx config.yaml for the WarpX-treatment comparison.")
    parser.add_argument("--density-floor-frac", type=float, default=1e-3,
                        help="Mask cells with n_e below this fraction of its peak (default 1e-3).")
    parser.add_argument("--ion", default="Si",
                        help="Ion species for plasmapy (default Si; spans Z=1..14, "
                             "eta is ion-independent for Al/Si).")
    parser.add_argument("--output-dir", default=None,
                        help="Where to write figure/data (default results/warpx/<tree-name>).")
    plot_style.add_publication_arg(parser)
    args = parser.parse_args(argv)
    plot_style.apply(args.publication)

    tree = Path(args.tree)
    meta, Te_eV, n_e, zbar = load_slice(tree)
    mask = plasma_mask(n_e, args.density_floor_frac)

    eta = spitzer_resistivity(Te_eV, n_e, zbar, ion=args.ion)

    out_dir = Path(args.output_dir) if args.output_dir else Path(_HERE) / ".." / "results" / "warpx" / tree.name
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- text report ----
    report = [f"Spitzer resistivity — {tree}",
              f"  slice shape={meta['shape']}  plane_axes={meta['plane_axes']}  "
              f"box(mm)={np.array(meta['axis0_bounds_m'])*1e3} x "
              f"{np.array(meta['axis1_bounds_m'])*1e3}", ""]
    sample_lines, eta_nw = sample_report(meta, Te_eV, n_e, zbar, eta, mask)
    report += sample_lines + [""]
    report += scan_table(meta["ne_mean_per_m3"], args.ion) + [""]

    eta_cfg_scalar = None
    if args.config:
        cfg = yaml.safe_load(Path(args.config).read_text())
        cmp_lines, _ = warpx_comparison(cfg, meta, n_e, zbar, eta, mask, args.ion)
        report += ["  --- WarpX treatment comparison ---"] + cmp_lines + [""]
        eta_cfg = (cfg.get("run", {}) or {}).get("plasma_resistivity")
        eta_cfg_scalar = eta_cfg if isinstance(eta_cfg, (int, float)) else None

    text = "\n".join(report)
    print(text)
    (out_dir / "spitzer_summary.txt").write_text(text + "\n")

    # ---- data + figure ----
    np.save(out_dir / "spitzer_eta.npy", eta)
    make_figure(meta, Te_eV, eta, mask, out_dir / "spitzer_eta.png", eta_cfg=eta_cfg_scalar)
    print(f"\nwrote {out_dir}/spitzer_eta.png  spitzer_eta.npy  spitzer_summary.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
