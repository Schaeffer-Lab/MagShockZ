# -*- coding: utf-8 -*-
"""scripts/flash_experiment_compare.py — experimental streaked shadowgraphy vs FLASH nₑ.

The streak camera records one spatial axis (mm) against time (ns) — the same layout
as the FLASH nₑ streak that ``scripts/tune_flash_shock.py`` draws from LOS line-outs.

**The experimental image's axes are the reference frame.**  Everything is plotted in
its own ns and mm; the image is never stretched or resampled into simulation units.
The simulation, which covers only the first ~15 ns of a ~69 ns record and a 6.3 mm
stretch of a ±5 mm slit, is *translated* onto those axes, and the image is *cropped*
to the window being looked at:

  Figure 1 ``flash_experiment_side_by_side.png``
      the experimental streak above the FLASH nₑ streak, sharing both axes, with every
      straight-line feature drawn on both.
  Figure 2 ``flash_experiment_overlay.png``
      both in one panel: the experiment underneath in greyscale, FLASH nₑ on top in a
      second colormap whose opacity ramps with density, so the simulated structure
      glows over the data while the upstream stays see-through.
  Figure 3 ``flash_experiment_lineouts.png``
      spatial profiles at a few times: FLASH nₑ (log, left axis) against the
      experimental brightness at the nearest streak column (right axis), with each
      feature marked where it sits at that instant.

Features
--------
Straight lines ``x(t) = x0_mm + v_kms·(t − t0_ns)`` overlaid on every panel, listed in
``experiment.trajectories`` and hand-fitted in the units the axes are in (km/s, mm, ns).
``frame: experiment`` (the default) measures a feature *in the data* — the observed
shock front, the piston plasma — so it is independent of the registration; ``frame:
flash`` is a simulated feature, translated onto the image like the FLASH data itself.
The ``flash:`` block's shock front is added automatically as a ``flash``-frame line.

Registration
------------
The experiment's time origin (camera trigger) and mm zero have no known FLASH
counterpart, so where FLASH lands on the image's axes is **hand-tuned, not derived** —
a rigid translation, plus a direction flip when the slit's +mm runs opposite to the
line of sight::

    t_exp = t_flash + t_offset_ns          mm = ±(los_µm / 1000) + x_offset_mm

Defaults come from the config's ``experiment.registration`` block; ``--t-offset-ns`` /
``--x-offset-mm`` / ``--flip-space`` override them.  Slide them until the features line
up, then write the values into the config.

The image itself is a *decorated* matplotlib figure (axes burned in), so it is cropped
to its plot box — auto-detected unless ``experiment.crop_px`` says otherwise — and
labelled with ``experiment.axes`` (``t_ns`` / ``x_mm``), what that box spans, read off
its ticks once (``calib.csv`` calibrates the *raw* streak, not the decorated PNG).
Those are the image's axes, not a zoom: they may be *translated* (to re-zero the mm
scale, say) but their span must stay the true one.  To look at part of the record use
``experiment.view`` / ``--t-window`` / ``--x-window``, which crop pixels instead.

Caveat: shadowgraphy brightness responds to ∇²∫nₑ dl through the probe, not to a
point sample of nₑ, so **positions are comparable, amplitudes are not** — read the
overlay for where the front is, not for how bright it is.

Env: analysis (yt / unyt).  Examples:
    python scripts/flash_experiment_compare.py --config config/flash_3d_2026-07.yaml
    python scripts/flash_experiment_compare.py --config config/flash_3d_2026-07.yaml \\
        --stride 2 --t-offset-ns 3 --x-offset-mm 0.7 --flip-space
    python scripts/flash_experiment_compare.py --config ...yaml --t-window 0 20 --full-range
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import unyt as u
import yt
from matplotlib.colors import LogNorm, Normalize

yt.set_log_level(50)   # suppress yt chatter

_HERE = os.path.dirname(os.path.abspath(__file__))

from magshockz.common import analysis_utils
from magshockz.analysis.flash import experiment_image as ei
from magshockz.common import flash_source
from magshockz.common import flash_utils as fu
from magshockz.common import plot_style
from magshockz.common import yaml_edit
# Same streak assembly as the overview and the tuner, so the nₑ map compared here is
# byte-for-byte the one the rest of the FLASH analysis draws.
from flash_overview import assemble_streak

_REPO = os.path.dirname(_HERE)


def _abs(path):
    """Config paths are repo-relative unless absolute."""
    return path if os.path.isabs(path) else os.path.join(_REPO, path)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def load_experiment(cfg, args):
    """Load the experimental streak with its own axes, and build the registration.

    Returns ``(StreakImage, Registration, view)`` where ``view`` is the requested
    ``(t_ns, x_mm)`` window (either element may be None = "whatever FLASH covers").
    Every knob has a config default (the ``experiment:`` block) that the CLI may
    override.
    """
    block = cfg.get("experiment") or {}
    csv = args.exp_csv or block.get("csv")
    path = args.exp_image or block.get("image")
    if not csv and not path:
        raise KeyError(
            "no experimental streak: add an `experiment:` block to the config with "
            "either `csv` + `calib` (the raw streak, preferred) or `image` + `axes:` "
            "(a rendered figure), or pass --exp-csv / --exp-image. "
            "See config/flash_3d_2026-07.yaml.")

    if csv:
        # Raw counts + mm/px, ns/px: the pixel grid IS the measurement, so the axes
        # follow from the calibration with nothing read off a rendered figure.
        csv = _abs(csv)
        calib_path = _abs(args.exp_calib or block.get("calib")
                          or os.path.join(os.path.dirname(csv), "calib.csv"))
        streak = ei.load_streak_csv(csv, ei.load_calib(calib_path),
                                    origin=block.get("origin", "center"),
                                    t0_ns=float(block.get("t0_ns", 0.0) or 0.0),
                                    row0_is_top=bool(block.get("row0_is_top", True)))
    else:
        axes = block.get("axes") or {}
        for key in ("t_ns", "x_mm"):
            if key not in axes:
                raise KeyError(
                    f"experiment.axes is missing `{key}` — the ns/mm span of the WHOLE "
                    "image, read off its burned-in axis ticks. It calibrates the image; "
                    "it is not a zoom (use experiment.view / --t-window / --x-window).")
        streak = ei.load_streak(_abs(path),
                                t_ns=axes["t_ns"],
                                x_mm=axes["x_mm"],
                                crop_px=block.get("crop_px"),
                                invert=bool(block.get("invert", False)))

    reg = ei.from_config(block.get("registration"))
    if args.t_offset_ns is not None:
        reg.t_offset_ns = args.t_offset_ns
    if args.x_offset_mm is not None:
        reg.x_offset_mm = args.x_offset_mm
    if args.flip_space:
        reg.flip_space = True
    if args.no_flip_space:
        reg.flip_space = False

    view_cfg = block.get("view") or {}
    view = (args.t_window or view_cfg.get("t_ns"),
            args.x_window or view_cfg.get("x_mm"))
    return streak, reg, view


def load_flash(cfg, args, source):
    """FLASH nₑ streak along the LOS: ``(ne[t, x], time_ns, x_um, dump_files)``."""
    all_files = fu.find_plot_files(source.flash_dir)
    stop = len(all_files) if args.t_stop is None else min(args.t_stop + 1, len(all_files))
    idx = [i for i in range(args.t_start, stop, args.stride) if i < len(all_files)]
    paths = [all_files[i] for i in idx]
    if len(paths) < 2:
        raise RuntimeError(f"need ≥2 dumps for a streak, got {len(paths)} "
                           f"(from {source.flash_dir})")

    nprocs = args.nprocs or int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) or os.cpu_count() or 1
    nprocs = min(max(1, nprocs), len(paths))
    lineouts = fu.load_lineouts(paths, source.line_start, source.line_end, nprocs)

    ne_streak, time_ns, x_um = assemble_streak(lineouts, "ne")
    return ne_streak, time_ns, x_um, paths


def trajectories(cfg):
    """Every straight-line feature to overlay, in draw order.

    The FLASH shock front comes from the ``flash:`` block that ``tune_flash_shock.py``
    writes (cm/s, cm, s → km/s, mm, ns) and is a ``"flash"``-frame line, so it moves
    with the registration.  Everything in ``experiment.trajectories`` is hand-fitted
    against the image and defaults to the ``"experiment"`` frame, so it stays put.
    """
    out = []
    flash = cfg.get("flash") or {}
    if "v_shock_est_cms" in flash and "x_shock_0_cm" in flash:
        out.append(ei.Trajectory(
            label="FLASH shock front",
            v_kms=float((float(flash["v_shock_est_cms"]) * u.cm / u.s).to("km/s").value),
            x0_mm=float((float(flash["x_shock_0_cm"]) * u.cm).to("mm").value),
            t0_ns=float((float(flash.get("t_shock_0_s", 0.0)) * u.s).to("ns").value),
            frame="flash", color="cyan", style="--"))
    out += ei.trajectories_from_config((cfg.get("experiment") or {}).get("trajectories"))
    return out


# ---------------------------------------------------------------------------
# Shared rendering pieces
# ---------------------------------------------------------------------------

def ne_norm(ne_streak, floor_decades=4.0):
    """Log colour normalisation for an nₑ streak (the tuner's percentile recipe)."""
    finite = ne_streak[np.isfinite(ne_streak)]
    pos = finite[finite > 0]
    vmax = float(np.percentile(pos, 99.5)) if pos.size else 1.0
    vmin = float(np.percentile(pos, 2)) if pos.size else vmax * 1e-4
    vmin = max(vmin, vmax * 10.0 ** (-floor_decades))
    return LogNorm(vmin=vmin, vmax=vmax)


def exp_norm(img):
    """Linear colour normalisation for the experimental brightness."""
    finite = img[np.isfinite(img)]
    if not finite.size:
        return Normalize(0.0, 1.0)
    return Normalize(vmin=float(np.percentile(finite, 1)),
                     vmax=float(np.percentile(finite, 99.5)))


def inset_colorbar(ax, mappable, label):
    """Colorbar in an inset axes, so stacked panels keep identical widths."""
    cax = ax.inset_axes([1.005, 0.0, 0.014, 1.0])
    cb = ax.figure.colorbar(mappable, cax=cax)
    cb.set_label(label)
    return cb


def draw_experiment(ax, streak, cmap, norm):
    """The experimental streak, drawn on its own (ns, mm) axes.

    ``extent`` is the image's own calibration, so the pixels land where the burned-in
    axes say they do — the view is narrowed by cropping/limits, never by rescaling.
    The axes background is set to the colormap's low end so the parts of the view the
    camera never saw read as "no data" rather than as a bright white block (and, in
    the overlay, so the FLASH layer there is not washed out by a white backdrop).
    """
    ax.set_facecolor(plt.get_cmap(cmap)(0.0))
    return ax.imshow(streak.img, origin="lower", extent=streak.extent,
                     aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")


def draw_flash(ax, time_ns, x_um, ne_streak, reg, cmap, norm):
    """The FLASH nₑ streak translated onto the experiment's (ns, mm) axes.

    Dumps are unevenly spaced in time, so this is a pcolormesh on the true dump times
    rather than an image.
    """
    mm, C = ei.flash_on_exp_axis(x_um, ne_streak, reg)
    return ax.pcolormesh(reg.to_exp_t(time_ns), mm, np.clip(C.T, norm.vmin, None),
                         cmap=cmap, norm=norm, shading="auto")


def draw_trajectories(ax, trajs, t_window, reg, override_color=None):
    """Overlay every straight-line feature across the visible time window."""
    if not trajs:
        return
    t = np.linspace(t_window[0], t_window[1], 200)
    for tr in trajs:
        ts, mm = tr.points(t, reg)
        ax.plot(ts, mm, color=override_color or tr.color, ls=tr.style, lw=tr.width,
                label=tr.legend())
    ax.legend(fontsize=8, loc="upper left", framealpha=0.7)


def resolve_view(streak, reg, time_ns, x_um, view, full_range):
    """The ``(t_ns, x_mm)`` window to show, in the experiment's own units.

    Priority: an explicit window (config ``experiment.view`` / ``--t-window`` /
    ``--x-window``) → the whole record with ``--full-range`` → the stretch of the
    record FLASH actually covers.  Windows are clipped to the image, so the default
    view is the overlap: the image cropped to the simulated ns/mm, not squeezed into
    them.
    """
    t_win, x_win = view
    ft0, ft1, fx0, fx1 = reg.flash_extent(time_ns, x_um)
    t_auto = streak.t_ns if full_range else (ft0, ft1)
    x_auto = streak.x_mm if full_range else (fx0, fx1)
    t = tuple(float(v) for v in t_win) if t_win else t_auto
    x = tuple(float(v) for v in x_win) if x_win else x_auto
    return (min(t), max(t)), (min(x), max(x))


def prepare(streak, reg, time_ns, x_um, view, full_range):
    """Crop the image to the view window (pixels, not pixels-per-mm) and return both.

    Cropping — rather than only setting axis limits — keeps the colour normalisation
    tied to what is actually on screen, so an image dominated by late-time signal
    still shows structure in the first few ns.
    """
    t_view, x_view = resolve_view(streak, reg, time_ns, x_um, view, full_range)
    inside = (t_view[1] > streak.t_ns[0] and t_view[0] < streak.t_ns[1] and
              x_view[1] > streak.x_mm[0] and x_view[0] < streak.x_mm[1])
    cropped = ei.crop_window(streak, t_view, x_view) if inside else streak
    return cropped, t_view, x_view, inside


# ---------------------------------------------------------------------------
# Figure 1 — side by side
# ---------------------------------------------------------------------------

def plot_side_by_side(view_streak, reg, time_ns, x_um, ne_streak, trajs, window,
                      out_dir, args, title):
    (t0, t1), (x0, x1) = window
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True, sharey=True)

    im_e = draw_experiment(axes[0], view_streak, args.exp_cmap, exp_norm(view_streak.img))
    inset_colorbar(axes[0], im_e, "shadowgraphy signal [arb.]")
    axes[0].set_title("experiment — streaked shadowgraphy")

    norm = ne_norm(ne_streak)
    im_f = draw_flash(axes[1], time_ns, x_um, ne_streak, reg, args.flash_cmap, norm)
    inset_colorbar(axes[1], im_f, r"$n_e$ [cm$^{-3}$]")
    axes[1].set_title(r"FLASH — $n_e$ along the line of sight, placed on those axes")

    for ax in axes:
        draw_trajectories(ax, trajs, (t0, t1), reg)
        ax.set_ylabel("position [mm]")
        ax.set_xlim(t0, t1)
        ax.set_ylim(x0, x1)
    axes[1].set_xlabel("$t$ [ns]")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    path = os.path.join(out_dir, "flash_experiment_side_by_side.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figure 2 — overlay
# ---------------------------------------------------------------------------

def value_alpha_rgba(values, cmap, norm, alpha_max, gamma=1.0):
    """Colour-map ``values`` with an opacity that ramps with the normalised value.

    Low (upstream) values become transparent so the image underneath shows through;
    the densest structure reaches ``alpha_max``.  ``gamma`` > 1 makes the fade-in
    sharper, i.e. only the brightest features are opaque.
    """
    cm = plt.get_cmap(cmap)
    frac = np.clip(np.asarray(norm(np.asarray(values, dtype=float))), 0.0, 1.0)
    rgba = cm(frac)
    rgba[..., 3] = alpha_max * frac ** gamma
    bad = ~np.isfinite(values)
    if bad.any():
        rgba[bad, 3] = 0.0
    return rgba


def regrid_streak(ne_streak, time_ns, x_um, reg, n_t=None):
    """Resample the (unevenly time-spaced) nₑ streak onto a regular grid in exp. units.

    ``imshow`` — needed for the RGBA overlay — assumes uniform pixels, while FLASH
    dumps are not uniformly spaced in time; interpolating in time first keeps the
    overlay honest instead of silently stretching the gaps.  Only the FLASH side is
    ever resampled; the experimental image is drawn as-is.
    """
    t_exp = reg.to_exp_t(time_ns)
    mm, Z_in = ei.flash_on_exp_axis(x_um, ne_streak, reg)
    n_t = n_t or max(len(t_exp), 512)
    t_reg = np.linspace(float(t_exp[0]), float(t_exp[-1]), n_t)
    Z = np.empty((Z_in.shape[1], n_t))
    for i in range(Z_in.shape[1]):                      # one LOS position at a time
        Z[i] = np.interp(t_reg, t_exp, Z_in[:, i])
    extent = _edges(t_reg) + _edges(mm)
    return Z, extent


def _edges(centres):
    """Outer edges of a uniformly spaced centre grid, as (lo, hi)."""
    c = np.asarray(centres, dtype=float)
    half = 0.5 * (c[-1] - c[0]) / max(len(c) - 1, 1)
    return (float(c[0] - half), float(c[-1] + half))


def plot_overlay(view_streak, reg, time_ns, x_um, ne_streak, trajs, window,
                 out_dir, args, title):
    (t0, t1), (x0, x1) = window
    fig, ax = plt.subplots(figsize=(13, 7))

    im_e = draw_experiment(ax, view_streak, args.exp_cmap, exp_norm(view_streak.img))

    norm = ne_norm(ne_streak)
    Z, extent = regrid_streak(ne_streak, time_ns, x_um, reg)
    rgba = value_alpha_rgba(Z, args.flash_cmap, norm, args.alpha, gamma=args.alpha_gamma)
    ax.imshow(rgba, origin="lower", extent=extent, aspect="auto", interpolation="bilinear")

    draw_trajectories(ax, trajs, (t0, t1), reg)
    ax.set_xlim(t0, t1)
    ax.set_ylim(x0, x1)
    ax.set_xlabel("$t$ [ns]")
    ax.set_ylabel("position [mm]")

    # Two colorbars side by side: the greyscale data underneath, the nₑ on top.
    cb_e = fig.colorbar(im_e, ax=ax, pad=0.015, fraction=0.045)
    cb_e.set_label("experiment — shadowgraphy signal [arb.]")
    sm = plt.cm.ScalarMappable(cmap=args.flash_cmap, norm=norm)
    cb_f = fig.colorbar(sm, ax=ax, pad=0.015, fraction=0.045)
    cb_f.set_label(r"FLASH — $n_e$ [cm$^{-3}$]  (opacity $\propto n_e$)")

    ax.set_title(title, fontsize=12)
    fig.tight_layout()
    path = os.path.join(out_dir, "flash_experiment_overlay.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figure 3 — profiles at a few times
# ---------------------------------------------------------------------------

def plot_lineouts(view_streak, reg, time_ns, x_um, ne_streak, trajs, window,
                  out_dir, args, title):
    """Profiles against the experiment's own mm axis, at a few FLASH times."""
    times = [t for t in args.times if time_ns.min() <= t <= time_ns.max()] or [
        float(np.median(time_ns))]
    (x0, x1) = window[1]
    mm_flash, ne_mm = ei.flash_on_exp_axis(x_um, ne_streak, reg)
    mm_exp = view_streak.x_axis()

    fig, axes = plt.subplots(1, len(times), figsize=(5.0 * len(times), 4.6),
                             squeeze=False, sharey=True)
    twins = []
    for ax, t_req in zip(axes[0], times):
        j = int(np.argmin(np.abs(time_ns - t_req)))
        ax.semilogy(mm_flash, ne_mm[j], color="crimson", lw=1.8,
                    label=fr"FLASH $n_e$  ($t$={reg.to_exp_t(time_ns[j]):.2f} ns)")
        ax.set_xlabel("position [mm]")
        ax.set_xlim(x0, x1)

        col, t_exp = view_streak.column(float(reg.to_exp_t(t_req)))
        ax2 = ax.twinx()
        twins.append(ax2)
        ax2.plot(mm_exp, col, color="0.25", lw=1.4,
                 label=f"experiment ($t$={t_exp:.2f} ns)")
        ax.set_title(f"$t_{{exp}}$ = {reg.to_exp_t(t_req):.1f} ns")

        # Where each feature sits at this instant — the check that the FLASH jump and
        # the measured one are (or are not) at the same mm.
        for tr in trajs:
            ax.axvline(tr.at(t_exp, reg), color=tr.color, ls=tr.style, lw=1.6,
                       alpha=0.9, label=tr.legend())

        lines = ax.get_lines() + ax2.get_lines()
        ax.legend(lines, [l.get_label() for l in lines], fontsize=7, loc="upper right",
                  framealpha=0.7)
    axes[0][0].set_ylabel(r"$n_e$ [cm$^{-3}$]")
    # One brightness scale across the panels (the left, nₑ axis is already shared),
    # so the panels show how the signal FALLS with time rather than each rescaling.
    hi = max(float(np.nanmax(a.get_lines()[0].get_ydata())) for a in twins) or 1.0
    for ax2 in twins[:-1]:
        ax2.set_ylim(0.0, 1.05 * hi)
        ax2.set_yticklabels([])
    twins[-1].set_ylim(0.0, 1.05 * hi)
    twins[-1].set_ylabel("shadowgraphy signal [arb.]")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    path = os.path.join(out_dir, "flash_experiment_lineouts.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figure 4 — the fit's correlation map (only with --fit)
# ---------------------------------------------------------------------------

def plot_fit(fit, out_dir, title):
    """How well-determined the shift is: r over every trialled (t, x) offset.

    A sharp peak means the data pins the registration down; a ridge means that
    direction is degenerate and the number should not be quoted as measured.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.pcolormesh(fit.t_offsets, fit.x_offsets, fit.r_map, cmap="viridis",
                       shading="auto")
    ax.plot(fit.registration.t_offset_ns, fit.registration.x_offset_mm, marker="*",
            ms=18, color="white", mec="k", label=f"best  r={fit.r:+.3f}")
    fig.colorbar(im, ax=ax, pad=0.02).set_label(f"correlation of {fit.feature}")
    ax.set_xlabel("$t$ offset [ns]")
    ax.set_ylabel("$x$ offset [mm]")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(f"registration fit (flip={fit.registration.flip_space})\n{title}",
                 fontsize=10)
    fig.tight_layout()
    path = os.path.join(out_dir, "flash_experiment_fit.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Compare an experimental streaked-shadowgraphy image to the "
                    "FLASH electron density along the line of sight.")
    p.add_argument("--config", required=True, help="Path to a FLASH analysis YAML config.")
    p.add_argument("--exp-csv", default=None, dest="exp_csv",
                   help="Raw streak CSV to compare against (default: the config's "
                        "experiment.csv). Preferred over --exp-image.")
    p.add_argument("--exp-calib", default=None, dest="exp_calib",
                   help="Calibration CSV (px_to_mm / px_to_ns) for --exp-csv "
                        "(default: experiment.calib, else calib.csv beside the data).")
    p.add_argument("--exp-image", default=None, dest="exp_image",
                   help="Rendered streak figure to use instead of the raw CSV "
                        "(default: the config's experiment.image).")
    p.add_argument("--fit", action="store_true",
                   help="Fit the registration: slide FLASH over the image (translation "
                        "only, nothing rescaled) and take the shift with the highest "
                        "normalised cross-correlation. Overrides the offsets.")
    p.add_argument("--fit-feature", default="grad", dest="fit_feature",
                   choices=("grad", "signal"),
                   help="What --fit correlates: spatial gradients (default, apt for "
                        "shadowgraphy) or the fields themselves.")
    p.add_argument("--fit-no-flip", action="store_true", dest="fit_no_flip",
                   help="Restrict --fit to the unflipped orientation.")
    p.add_argument("--t-offset-ns", type=float, default=None, dest="t_offset_ns",
                   help="Experiment time [ns] at which FLASH t=0 occurs "
                        "(default: config experiment.registration.t_offset_ns).")
    p.add_argument("--x-offset-mm", type=float, default=None, dest="x_offset_mm",
                   help="Experiment position [mm] of LOS distance 0 "
                        "(default: config experiment.registration.x_offset_mm).")
    p.add_argument("--flip-space", action="store_true", dest="flip_space",
                   help="Experiment +mm runs opposite to the FLASH line of sight.")
    p.add_argument("--no-flip-space", action="store_true", dest="no_flip_space",
                   help="Force no spatial flip, overriding the config.")
    p.add_argument("--times", type=float, nargs="+", default=[2.0, 6.0, 10.0],
                   help="FLASH times [ns] for the line-out figure (default 2 6 10).")
    p.add_argument("--t-window", type=float, nargs=2, default=None, dest="t_window",
                   metavar=("LO", "HI"),
                   help="Time window [ns, EXPERIMENT clock] to show; the image is "
                        "cropped to it (default: config experiment.view.t_ns, else "
                        "the span FLASH covers).")
    p.add_argument("--x-window", type=float, nargs=2, default=None, dest="x_window",
                   metavar=("LO", "HI"),
                   help="Spatial window [mm, EXPERIMENT axis] to show; the image is "
                        "cropped to it (default: config experiment.view.x_mm, else "
                        "the span FLASH covers).")
    p.add_argument("--full-range", action="store_true", dest="full_range",
                   help="Show the whole experimental record instead of cropping the "
                        "view to the stretch of it the FLASH run covers.")
    p.add_argument("--exp-cmap", default="gray", dest="exp_cmap")
    p.add_argument("--flash-cmap", default="magma", dest="flash_cmap")
    p.add_argument("--alpha", type=float, default=0.75,
                   help="Peak opacity of the FLASH layer in the overlay (default 0.75).")
    p.add_argument("--alpha-gamma", type=float, default=1.5, dest="alpha_gamma",
                   help="Exponent of the overlay's opacity ramp; >1 fades in only the "
                        "densest structure (default 1.5).")
    p.add_argument("--stride", type=int, default=1,
                   help="Dump stride for the FLASH streak (default 1 = every dump).")
    p.add_argument("--t-start", type=int, default=0, dest="t_start")
    p.add_argument("--t-stop", type=int, default=None, dest="t_stop")
    p.add_argument("--nprocs", type=int, default=None,
                   help="Worker processes for loading dumps (default: all cores).")
    p.add_argument("--output-dir", default=None, dest="output_dir")
    p.add_argument("--no-npz", action="store_true", dest="no_npz",
                   help="Skip writing the .npz archive.")
    plot_style.add_publication_arg(p)
    args = p.parse_args()
    plot_style.apply(args.publication)

    cfg = analysis_utils.load_config(args.config)
    config_path = os.path.abspath(args.config)
    source = flash_source.resolve(cfg, config_path)

    streak, reg, view = load_experiment(cfg, args)
    print(f"Config     : {config_path}")
    print(f"FLASH      : {source.flash_dir}   (from {source.source})")
    print(f"Experiment : {streak.path}")
    print(f"             {streak.shape[1]}×{streak.shape[0]} px spanning "
          f"{streak.t_ns[0]:g}–{streak.t_ns[1]:g} ns, "
          f"{streak.x_mm[0]:g}–{streak.x_mm[1]:g} mm "
          f"({(streak.t_ns[1] - streak.t_ns[0]) / streak.shape[1]:.4g} ns/px, "
          f"{(streak.x_mm[1] - streak.x_mm[0]) / streak.shape[0]:.4g} mm/px)")

    ne_streak, time_ns, x_um, dump_files = load_flash(cfg, args, source)
    trajs = trajectories(cfg)

    fit = None
    if args.fit:
        fit = ei.fit_shift(streak, time_ns, x_um / 1000.0, ne_streak,
                           feature=args.fit_feature,
                           flips=(False,) if args.fit_no_flip else (False, True))
        reg = fit.registration
        print(f"Fit         : feature={fit.feature}  best r={fit.r:+.3f}  "
              f"→ t_offset={reg.t_offset_ns:+.2f} ns, x_offset={reg.x_offset_mm:+.2f} mm, "
              f"flip={reg.flip_space}")
        if fit.flip_r and len(fit.flip_r) > 1:
            print("              r by orientation: " +
                  ", ".join(f"flip={k}: {v:+.3f}" for k, v in sorted(fit.flip_r.items())))
        print("              (a correlation peak is not physics — check the r map and "
              "the overlay before trusting it)")

    ft0, ft1, fx0, fx1 = reg.flash_extent(time_ns, x_um)
    t_lo, t_hi, x_lo, x_hi = ei.overlap_window(streak, reg, time_ns, x_um)
    print(f"Registration: t_offset = {reg.t_offset_ns:g} ns, "
          f"x_offset = {reg.x_offset_mm:g} mm, flip = {reg.flip_space}")
    print(f"FLASH lands : {ft0:.2f}–{ft1:.2f} ns, {fx0:.2f}–{fx1:.2f} mm "
          f"on the image's axes  ({len(dump_files)} dumps, "
          f"LOS {x_um.min():.0f}–{x_um.max():.0f} µm)")
    if t_hi <= t_lo or x_hi <= x_lo:
        print("  ⚠ that does NOT overlap the image — adjust --t-offset-ns / "
              "--x-offset-mm / --flip-space; showing the full record meanwhile.")
        args.full_range = True
    else:
        print(f"Overlap     : {t_lo:.2f}–{t_hi:.2f} ns, {x_lo:.2f}–{x_hi:.2f} mm")

    if trajs:
        print("Features    :")
        for tr in trajs:
            t_lo_tr, t_hi_tr = (float(reg.to_exp_t(time_ns.min())),
                                float(reg.to_exp_t(time_ns.max())))
            print(f"  {tr.label:<24s} {tr.v_kms:7.0f} km/s  "
                  f"x₀={tr.x0_mm:+.2f} mm @ t₀={tr.t0_ns:.2f} ns  [{tr.frame}]  "
                  f"→ {tr.at(t_lo_tr, reg):+.2f} mm at {t_lo_tr:.1f} ns, "
                  f"{tr.at(t_hi_tr, reg):+.2f} mm at {t_hi_tr:.1f} ns")

    view_streak, t_view, x_view, _ = prepare(streak, reg, time_ns, x_um, view,
                                             args.full_range)
    window = (t_view, x_view)
    print(f"View        : {t_view[0]:.2f}–{t_view[1]:.2f} ns, "
          f"{x_view[0]:.2f}–{x_view[1]:.2f} mm  → image cropped to "
          f"{view_streak.shape[1]}×{view_streak.shape[0]} px")

    out_dir = yaml_edit.out_dir(source.flash_dir, args.output_dir,
                                cfg=cfg, config_path=config_path)
    title = (f"{os.path.basename(streak.path)}  vs  {os.path.basename(source.flash_dir)}\n"
             f"experiment axes are the frame; FLASH translated by "
             f"t+{reg.t_offset_ns:g} ns, x₀={reg.x_offset_mm:g} mm, flip={reg.flip_space}")

    for path in (plot_side_by_side(view_streak, reg, time_ns, x_um, ne_streak, trajs,
                                   window, out_dir, args, title),
                 plot_overlay(view_streak, reg, time_ns, x_um, ne_streak, trajs,
                              window, out_dir, args, title),
                 plot_lineouts(view_streak, reg, time_ns, x_um, ne_streak, trajs,
                               window, out_dir, args, title)):
        print(f"Saved → {path}")
    if fit is not None:
        print(f"Saved → {plot_fit(fit, out_dir, os.path.basename(streak.path))}")

    if not args.no_npz:
        npz_path = os.path.join(out_dir, "flash_experiment_compare.npz")
        np.savez_compressed(
            npz_path,
            # FLASH side (native CGS, as assembled for every other FLASH figure)
            dump_files=np.asarray([os.path.basename(f) for f in dump_files]),
            time_ns=time_ns,
            x_um=x_um,
            ne_streak=ne_streak,
            # experiment side: the full image with its own calibration, plus the
            # cropped view that was plotted (both carry their true ns/mm edges)
            exp_image=streak.img,
            exp_t_ns=np.asarray(streak.t_ns),
            exp_x_mm=np.asarray(streak.x_mm),
            exp_path=np.asarray(streak.path),
            view_image=view_streak.img,
            view_t_ns=np.asarray(view_streak.t_ns),
            view_x_mm=np.asarray(view_streak.x_mm),
            # the hand-tuned mapping between the two
            t_offset_ns=np.asarray(reg.t_offset_ns),
            x_offset_mm=np.asarray(reg.x_offset_mm),
            flip_space=np.asarray(reg.flip_space),
            config_path=np.asarray(config_path),
        )
        print(f"Saved → {npz_path}")

    print("\nTune the alignment by re-running with, e.g.:\n"
          f"  python {os.path.relpath(__file__, _REPO)} --config {args.config} "
          f"--stride {args.stride} --t-offset-ns {reg.t_offset_ns:g} "
          f"--x-offset-mm {reg.x_offset_mm:g}"
          f"{' --flip-space' if reg.flip_space else ''}\n"
          "then store the values in the config's experiment: block.")


if __name__ == "__main__":
    main()
