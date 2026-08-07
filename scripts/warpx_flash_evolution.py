# -*- coding: utf-8 -*-
"""scripts/warpx_flash_evolution.py — FLASH vs WarpX evolution, side by side.

Two figures comparing the two codes as they evolve:

  ``evolution_lineouts.png``   1-D profiles (target-species density, ambient-species
                               density, bulk velocity, |B|, T_e) at matched times, FLASH
                               and WarpX overlaid, one column per time
  ``evolution_slices.png``     2-D target-species slices at the same times, FLASH above
                               WarpX
  ``evolution_slices.mp4``     the same slices as a movie (--movie)

WHAT "MATCHED" MEANS.  The deck runs at a reduced mass ratio and an arbitrary reference
density, so nothing absolute is comparable — see ``magshockz/init/warpx/units.py``.  Every
axis here is therefore in ion units: length in ``d_i``, time in ``T_ci``, density relative
to the ambient, |B| relative to ``B0``.  In those units one FLASH gyroperiod IS one WarpX
gyroperiod, which is what makes the panels comparable at all.

SPECIES ARE NEVER SUMMED.  The piston comparison is target species against target
species: FLASH's electron density masked by its Si mass fraction (``targ``), against
WarpX's ``rho_piston_ions``, which at the deck's Z = 1 is that species' electron
density too.  A combined n_e would fold FLASH's EOS ionization state (Zbar 3.7 in the
ambient, ~11 in the piston) into a comparison whose deck has Z = 1 by construction.

THE VELOCITY ROW COMES FROM RAW PARTICLES, ON A COARSER CLOCK.  The deck's field
diagnostic writes only the TOTAL current (``jx jy jz``), and four species with two charge
signs cannot be unpicked into one species' bulk velocity — nor can ``usq``, the second
moment, give back a direction.  So the piston-ion velocity is binned from the sparse
``phase`` dumps (2% of particles), which run every 11000 steps against the fields' 1400 and
coincide only at step 0.  Each column therefore pairs its field profiles with the *nearest*
particle dump and prints that dump's own time in the velocity panel.

TWO CAVEATS THE FIGURES CANNOT HIDE, both stated on the plots:

1. *Clock zero means different things.*  FLASH at window start (3 ns) already has a piston
   running at 769 km/s; WarpX starts from a cold slab and needs a fraction of a gyroperiod
   for the heater to establish one.  ``--align clock`` (default) lines up elapsed time from
   each window's start, so the early WarpX columns are a startup transient, not a
   disagreement.  ``--align front`` instead shifts WarpX in time so the two fronts
   coincide at the first frame, which is the fairer comparison of *shape* once both are
   running.
2. *The WarpX box is transversely narrower.*  Its x half-width is 4 heating-spot radii
   (~5.6 d_i) because the spot's periodic images must stay clear of the spot, while the
   FLASH domain spans ~24 d_i transversely.  The 2-D panels share a d_i axis so this is
   visible rather than hidden by independent scales.

Usage
-----
    conda activate analysis
    python scripts/warpx_flash_evolution.py --config runs/magshockz_2d_heater.warpx.yaml \\
        [--n-times 5] [--align clock|front] [--diag-dir ...] [--movie] [--pub]

Run in the `analysis` conda env (yt).
"""

import argparse
import glob
import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

_HERE = os.path.dirname(os.path.abspath(__file__))

from magshockz.common import analysis_utils
from magshockz.common import flash_source
import astropy.units as u
from astropy.constants import c, e, m_e

from magshockz.init.warpx import config as spec_config
from magshockz.init.warpx import deck as deck_module
from magshockz.init.warpx import units
from magshockz.common import piston_profile as pp
from magshockz.common import plot_style
from magshockz.common import yaml_edit

# flash_utils is imported LAZILY, and only after every WarpX plotfile has been read.
# Importing it calls yt.enable_plugins(), which registers the flash2osiris ("flash", ...)
# derived fields globally; yt then validates those against the next dataset it opens, and
# a WarpX plotfile has no ('flash','velz'), so the load dies. CLAUDE.md flags the same
# hazard for flash_3d_movie.py. read_warpx_frames() asserts the ordering rather than
# trusting it.

_REPO = os.path.abspath(os.path.join(_HERE, ".."))
CM_PER_UM = 1.0e-4

#: Checked against the generator's own list rather than mirrored, so a rename there
#: fails here loudly instead of leaving the script asking for a missing plotfile field.
PISTON_IONS = "piston_ions"
AMBIENT_IONS = "amb_ions"
PISTON_ELECTRONS = "piston_electrons"
assert {PISTON_IONS, AMBIENT_IONS, PISTON_ELECTRONS} <= set(deck_module.SPECIES_NAMES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare FLASH and WarpX evolution: 1D lineouts and 2D slices.")
    parser.add_argument("--config", required=True,
                        help="heater_pic_2d run spec (runs/*.warpx.yaml)")
    parser.add_argument("--flash-config", default="config/flash_3d_corrected.yaml",
                        help="FLASH-side config for the slices (default: %(default)s)")
    parser.add_argument("--diag-dir", help="WarpX plotfile dir (default: the run's diags/)")
    parser.add_argument("--n-times", type=int, default=5,
                        help="columns / matched times (default: %(default)s)")
    parser.add_argument("--align", choices=("clock", "front"), default="clock",
                        help="'clock' matches elapsed time from each window's start; "
                             "'front' shifts WarpX so the fronts coincide at the first "
                             "frame (default: %(default)s)")
    parser.add_argument("--slice-halfwidth-um", type=float, default=4000.0,
                        help="FLASH slice transverse half-width (default: %(default)s)")
    parser.add_argument("--movie", action="store_true",
                        help="also render evolution_slices.mp4 over every WarpX frame")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--output-dir")
    plot_style.add_publication_arg(parser)
    return parser.parse_args()


def load_scales(config_path: str) -> tuple[dict, units.DeckScales]:
    """Run spec + re-derived scales (never the frozen copy, so they cannot go stale)."""
    spec = spec_config.load(config_path)
    return spec, spec_config.scales(spec, smoke=False)


def warpx_plotfiles(config_path: str, override: str | None,
                    prefix: str = "diag1") -> list[str]:
    if override:
        candidates = [override]
    else:
        run_name = os.path.basename(config_path).replace(".warpx.yaml", "")
        run_dir = os.path.join(_REPO, "input_files", "warpx", run_name)
        candidates = [os.path.join(run_dir, "diags"), run_dir]
    for directory in candidates:
        paths = sorted(p for p in glob.glob(os.path.join(directory, f"{prefix}*"))
                       if os.path.isdir(p))
        if paths:
            return paths
    if prefix != "diag1":
        return []
    raise SystemExit(f"No diag1* plotfiles under {candidates}; run the deck first.")


# ---------------------------------------------------------------------------
# WarpX side
# ---------------------------------------------------------------------------

def read_warpx_frames(paths: list[str], scales: units.DeckScales) -> list[dict]:
    """Every WarpX plotfile, read before any FLASH module can enable the yt plugin."""
    if "flash_utils" in sys.modules:
        raise RuntimeError(
            "flash_utils is already imported, so yt has the flash2osiris plugin fields "
            "registered and loading a WarpX plotfile will fail on ('flash','velz'). Read "
            "the WarpX side first.")
    return [warpx_frame(path, scales) for path in paths]


def warpx_frame(path: str, scales: units.DeckScales) -> dict:
    """One WarpX plotfile: 2-D maps and x-averaged z-profiles, in ion units.

    Only the +z half is kept.  The slab is symmetric about z = 0 and expands both ways, so
    the +z lobe is the direct analogue of FLASH's outward LOS.
    """
    import yt

    dataset = yt.load(path)
    grid = dataset.covering_grid(level=0, left_edge=dataset.domain_left_edge,
                                dims=dataset.domain_dimensions)

    def field(name: str) -> np.ndarray:
        return np.asarray(grid["boxlib", name]).squeeze()

    d_i = scales.ion_skin_depth.to_value(u.m)
    x_edges = np.linspace(float(dataset.domain_left_edge[0]),
                          float(dataset.domain_right_edge[0]),
                          field(f"rho_{PISTON_IONS}").shape[0] + 1)
    z_edges = np.linspace(float(dataset.domain_left_edge[1]),
                          float(dataset.domain_right_edge[1]),
                          field(f"rho_{PISTON_IONS}").shape[1] + 1)
    x_di = 0.5 * (x_edges[:-1] + x_edges[1:]) / d_i
    z_di = 0.5 * (z_edges[:-1] + z_edges[1:]) / d_i
    outward = z_di >= 0.0

    ambient_density = scales.upstream.electron_density.to_value(u.m**-3)
    piston = field(f"rho_{PISTON_IONS}") / e.si.value / ambient_density
    ambient = field(f"rho_{AMBIENT_IONS}") / e.si.value / ambient_density
    b_mag = (np.hypot(np.hypot(field("Bx"), field("By")), field("Bz"))
             / scales.magnetic_field.to_value(u.T))
    te_ev = field(f"T_{PISTON_ELECTRONS}")

    time_gyro = float(dataset.current_time) / scales.gyroperiod.to_value(u.s)
    # Every map and profile below assumes [x, z] indexing -- the x-average that builds the
    # profiles and the un-transposed imshow in plot_slices both depend on it.
    if piston.shape != (x_di.size, z_di.size):
        raise RuntimeError(
            f"expected a [x, z] = {(x_di.size, z_di.size)} grid, got {piston.shape}")
    front_di = pp.front_position(z_di[outward], piston[:, outward].mean(axis=0), level=1.0)

    return {
        "t_gyro": time_gyro,
        "x_di": x_di,
        "z_di": z_di[outward],
        "density_map": piston[:, outward],
        "ambient_map": ambient[:, outward],
        "b_map": b_mag[:, outward],
        "piston_profile": piston[:, outward].mean(axis=0),
        "ambient_profile": ambient[:, outward].mean(axis=0),
        "b_profile": b_mag[:, outward].mean(axis=0),
        "te_profile": (te_ev[:, outward].mean(axis=0)
                       / scales.upstream.electron_temperature.to_value(u.eV)),
        "front_di": front_di,
    }


def warpx_velocity_profile(path: str, scales: units.DeckScales,
                           z_di: np.ndarray) -> np.ndarray:
    """Piston-ion bulk velocity along z, in units of the deck's target piston speed.

    This has to come from the raw particles in a ``phase`` dump: the field diagnostic
    writes only the TOTAL current (``jx jy jz``), and a sum over four species with two
    charge signs cannot be unpicked into one species' bulk velocity.  ``usq`` is no help
    either -- it is the second moment, ``<u>^2 + thermal``, with the bulk direction and
    sign already lost.

    Sampled onto the caller's ``z_di`` bin centres so the row shares the field rows' axis,
    and returned normalised the same way every other row is: by the deck's *intended*
    reference (``DeckScales.piston_speed``), not by what the run achieved.  A piston comoving with a
    front running 9% fast therefore plateaus at 1.09, which is the honest reading.
    """
    import yt

    dataset = yt.load(path)
    left = np.asarray(dataset.domain_left_edge, dtype=float)
    right = np.asarray(dataset.domain_right_edge, dtype=float)

    data = dataset.all_data()
    z = np.asarray(data[(PISTON_IONS, "particle_position_y")], dtype=float)
    momentum = np.asarray(data[(PISTON_IONS, "particle_momentum_z")], dtype=float)
    weight = np.asarray(data[(PISTON_IONS, "particle_weight")], dtype=float)
    if z.size == 0:
        return np.full(z_di.shape, np.nan)
    # In a 2D WarpX plotfile yt still calls the second stored coordinate "y"; here that is
    # the long (z) axis. Assert it rather than assume -- reading the transverse axis by
    # mistake would give a piston that never moves, which looks like a physics result.
    assert z.min() >= left[1] - 1e-9 and z.max() <= right[1] + 1e-9, (
        f"{os.path.basename(path)}: particle_position_y spans "
        f"[{z.min():.4g}, {z.max():.4g}] m, outside the z axis "
        f"[{left[1]:.4g}, {right[1]:.4g}] -- wrong axis?")

    proper_speed = momentum / scales.piston_ion.mass.to_value(u.kg)   # p/m = gamma*v
    velocity = proper_speed / np.sqrt(1.0 + (proper_speed / c.si.value) ** 2)

    spacing = float(z_di[1] - z_di[0])
    edges = np.concatenate([z_di - 0.5 * spacing, [z_di[-1] + 0.5 * spacing]])
    edges = edges * scales.ion_skin_depth.to_value(u.m)
    outward = z >= 0.0
    bulk = pp.weighted_bin_average(z[outward], velocity[outward], weight[outward], edges)
    return bulk / scales.piston_speed.to_value(u.m / u.s)


def read_warpx_velocity(phase_paths: list[str], scales: units.DeckScales,
                        z_di: np.ndarray) -> list[dict]:
    """Every ``phase`` dump's velocity profile, read alongside the field dumps.

    Must run BEFORE ``flash_utils`` is imported, for the same reason
    :func:`read_warpx_frames` must: the FLASH yt plugin registers derived ``('flash', ...)``
    fields, and yt then evaluates them against every subsequently loaded dataset — so a
    WarpX plotfile opened afterwards dies on ``('flash', 'velz')``.  Reading all nine dumps
    up front costs little and keeps the pairing (which needs the FLASH-dependent alignment)
    as pure arithmetic in :func:`attach_warpx_velocity`.
    """
    import yt

    out = []
    for path in phase_paths:
        out.append({
            "t_gyro": (float(yt.load(path).current_time)
                       / scales.gyroperiod.to_value(u.s)),
            "velocity_profile": warpx_velocity_profile(path, scales, z_di),
        })
    return out


def attach_warpx_velocity(frames: list[dict], velocity_frames: list[dict]) -> None:
    """Give each frame the velocity profile from its nearest ``phase`` dump.

    The two diagnostics run on different cadences (1400 vs 11000 steps, coincident only at
    step 0), so a field frame is paired with the closest raw-particle dump and the time
    slip is recorded in ``velocity_t_gyro`` for the plot to state.  Without any phase dumps
    the row is left as nan and the figure simply shows no WarpX velocity.
    """
    if not velocity_frames:
        for frame in frames:
            frame["velocity_profile"] = np.full(frame["z_di"].shape, np.nan)
            frame["velocity_t_gyro"] = float("nan")
        return

    times = np.array([entry["t_gyro"] for entry in velocity_frames])
    for frame in frames:
        nearest = velocity_frames[int(np.argmin(np.abs(times - frame["t_gyro"])))]
        frame["velocity_profile"] = nearest["velocity_profile"]
        frame["velocity_t_gyro"] = nearest["t_gyro"]


# ---------------------------------------------------------------------------
# FLASH side
# ---------------------------------------------------------------------------

def flash_frame(path: str, source, targets: units.FlashReference, *,
                piston_material: str, ambient_material: str, halfwidth_um: float,
                t_start_s: float, npoints: int = 1024) -> dict:
    """One FLASH dump: LOS line-outs and a 2-D slice, in the same ion units.

    Densities are split by FLASH material rather than summed, so the piston row of the
    comparison is target species against target species.
    """
    from magshockz.common import flash_utils as fu

    extra = {"piston_frac": ("flash", piston_material)}
    lineout = fu.flash_lineout(path, source.line_start, source.line_end,
                               npoints=npoints, extra_fields=extra)
    # Both 2-D maps are the TARGET species only, so the slices compare the same thing the
    # line-outs and the front metric do.
    sliced = fu.flash_slice(path, source.line_start, source.line_end,
                            halfwidth_um=halfwidth_um,
                            mask_field=("flash", piston_material))
    ambient_slice = fu.flash_slice(path, source.line_start, source.line_end,
                                   halfwidth_um=halfwidth_um,
                                   mask_field=("flash", ambient_material))

    upstream = targets.upstream
    d_i_um = upstream.ion_skin_depth.to_value(u.um)
    x_um = lineout["x"].to("um").value
    piston_frac = np.asarray(lineout["piston_frac"], dtype=float)
    ne_cm3 = lineout["ne"].to("cm**-3").value

    # FLASH's target mass fraction masking the electron density -- see measure_dump() in
    # flash_piston_profile.py for why this and not Zbar * rho*X/(A m_u).
    n_piston = piston_frac * ne_cm3
    n_amb_cm3 = upstream.electron_density.to_value(u.cm**-3)

    los_lo, los_hi, tr_lo, tr_hi = sliced["extent"]
    return {
        "t_gyro": (lineout["t_s"] - t_start_s) / upstream.gyroperiod.to_value(u.s),
        "z_di": x_um / d_i_um,
        "piston_profile": n_piston / n_amb_cm3,
        "ambient_profile": (1.0 - piston_frac) * ne_cm3 / n_amb_cm3,
        "b_profile": (lineout["B_mag"].to("tesla").value
                      / upstream.magnetic_field.to_value(u.T)),
        "te_profile": (lineout["Te"].to("eV").value
                       / upstream.electron_temperature.to_value(u.eV)),
        # v.n_hat along the LOS, so no projection is needed; normalised by the fitted
        # front speed that the deck's piston_speed is the bridge image of, which is what
        # makes this row comparable to the WarpX one.
        "velocity_profile": (lineout["v_para"].to("m/s").value
                             / targets.piston_front_speed.to_value(u.m / u.s)),
        "density_map": sliced["img"] / n_amb_cm3,
        "ambient_map": ambient_slice["img"] / n_amb_cm3,
        "extent_di": (los_lo / d_i_um, los_hi / d_i_um,
                      (tr_lo - sliced["los_transverse_um"]) / d_i_um,
                      (tr_hi - sliced["los_transverse_um"]) / d_i_um),
        "front_di": pp.front_position(x_um / d_i_um, n_piston / n_amb_cm3, level=1.0),
    }


# ---------------------------------------------------------------------------
# Time matching
# ---------------------------------------------------------------------------

def pick_matched_times(flash_times_gyro: np.ndarray, warpx_times_gyro: np.ndarray,
                       n_times: int, align: str,
                       flash_fronts=None, warpx_fronts=None) -> tuple:
    """Index pairs into the two series at ``n_times`` matched instants, plus the offset.

    ``clock`` matches elapsed gyroperiods directly.  ``front`` first finds the constant
    WarpX time shift that makes its front position agree with FLASH's, which removes the
    heater's startup transient from the comparison of profile *shape*.
    """
    offset = 0.0
    if align == "front" and flash_fronts is not None and warpx_fronts is not None:
        usable_w = np.isfinite(warpx_fronts) & (warpx_times_gyro > 0)
        usable_f = np.isfinite(flash_fronts)
        # ONE finite FLASH front is enough -- only its first is used as the target, and
        # main() deliberately loads just that dump. Demanding two here made --align front
        # a silent no-op that still reported "shifted by 0.0000 T_ci".
        if np.count_nonzero(usable_w) >= 2 and np.count_nonzero(usable_f) >= 1:
            target = float(flash_fronts[usable_f][0])
            # First WarpX time whose front has reached FLASH's starting front position.
            reached = warpx_times_gyro[usable_w][warpx_fronts[usable_w] >= target]
            if reached.size:
                offset = float(reached[0])

    span = min(flash_times_gyro.max(), warpx_times_gyro.max() - offset)
    probes = np.linspace(0.0, span, n_times)
    flash_idx = [int(np.argmin(np.abs(flash_times_gyro - t))) for t in probes]
    warpx_idx = [int(np.argmin(np.abs(warpx_times_gyro - (t + offset)))) for t in probes]
    return flash_idx, warpx_idx, offset


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

CAVEAT_CLOCK = ("t = 0 is each window's start: FLASH already has a 769 km/s piston, "
                "WarpX starts from a cold slab (early columns = heater startup)")
CAVEAT_FRONT = ("WarpX shifted by {:.4f} $T_{{ci}}$ so the fronts coincide at t = 0 "
                "(removes the heater startup transient)")


def plot_lineouts(flash_frames: list, warpx_frames: list, offset: float, align: str,
                  out_path: str) -> None:
    """Rows = quantity, columns = time; FLASH and WarpX overlaid in ion units."""
    # Explicit y-limits per row, because both codes put unplottable values in the
    # empty regions: FLASH's smallx floor makes the piston fraction ~1e-99 outside the
    # plume, and WarpX's per-species T is exactly 0 in cells holding no particles of that
    # species. Left to autoscale, a log axis then spans 40+ decades of nothing.
    # Species are kept separate rather than summed: the piston row is target species on
    # both sides (FLASH Si, WarpX piston_ions) and the ambient row is the material each
    # code's piston is driving into. A combined n_e row would hide which species moved.
    # The velocity row is LINEAR: it legitimately passes through zero and goes negative,
    # which a log axis cannot show, and its interesting structure (does the piston comove
    # with its front?) sits within a factor of two of 1.
    rows = [
        ("piston_profile", r"$n_\mathrm{target} / n_\mathrm{amb}$", (1e-3, 1e3), "log"),
        ("ambient_profile", r"$n_\mathrm{ambient} / n_\mathrm{amb}$", (3e-1, 3e1), "log"),
        ("velocity_profile", r"$v_\parallel / v_\mathrm{front}$", (-0.4, 2.0), "linear"),
        ("b_profile", r"$|B| / B_0$", (1e-3, 3e1), "log"),
        ("te_profile", r"$T_e / T_{e,\mathrm{amb}}$", (1e-2, 1e4), "log"),
    ]
    n_cols = len(flash_frames)
    fig, axes = plt.subplots(len(rows), n_cols, figsize=(3.5 * n_cols, 2.7 * len(rows)),
                             sharex=True, squeeze=False, layout="constrained")

    def plottable(values: np.ndarray, floor: float, scale: str) -> np.ndarray:
        """Blank out values at/below a log axis's floor so the line breaks, not dives."""
        values = np.asarray(values, dtype=float)
        if scale != "log":
            return values
        return np.where(values > floor, values, np.nan)

    for col, (flash, warpx) in enumerate(zip(flash_frames, warpx_frames)):
        for row, (key, label, ylim, scale) in enumerate(rows):
            ax = axes[row][col]
            ax.plot(flash["z_di"], plottable(flash[key], ylim[0], scale),
                    color="#1f77b4", lw=1.6, label="FLASH")
            ax.plot(warpx["z_di"], plottable(warpx[key], ylim[0], scale),
                    color="#d62728", lw=1.4, ls="--", label="WarpX")
            for frame, color in ((flash, "#1f77b4"), (warpx, "#d62728")):
                if np.isfinite(frame["front_di"]):
                    ax.axvline(frame["front_di"], color=color, lw=0.9, alpha=0.5)
            if scale == "linear":
                # 1 = moving with the front, 0 = at rest: the two references that make the
                # row readable without hunting along the axis.
                ax.axhline(1.0, color="0.4", lw=0.8, ls="-.")
                ax.axhline(0.0, color="0.7", lw=0.8)
            ax.set_yscale(scale)
            ax.set_ylim(*ylim)
            ax.grid(alpha=0.25, which="both")
            if col == 0:
                ax.set_ylabel(label)
            if row == 0:
                ax.set_title(f"FLASH {flash['t_gyro']:.4f} / WarpX "
                             f"{warpx['t_gyro']:.4f} $T_{{ci}}$", fontsize=9)
            # The velocity row's WarpX curve comes from the nearest phase dump, which is on
            # a coarser cadence than the field rows -- say so rather than implying the
            # whole column is one instant.
            if key == "velocity_profile" and np.isfinite(warpx.get("velocity_t_gyro",
                                                                   float("nan"))):
                ax.text(0.02, 0.94,
                        f"WarpX particles @ {warpx['velocity_t_gyro']:.4f} $T_{{ci}}$",
                        transform=ax.transAxes, fontsize=7, va="top", color="#d62728")
            if row == len(rows) - 1:
                ax.set_xlabel(r"distance from target [$d_i$]")
    axes[0][0].legend(fontsize=8, loc="best")

    caveat = (CAVEAT_FRONT.format(offset) if align == "front" else CAVEAT_CLOCK)
    fig.suptitle("FLASH vs WarpX heater piston — 1D line-outs in ion units, species kept "
        "separate (piston row = FLASH Si vs WarpX piston_ions)\n" + caveat,
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_slices(flash_frames: list, warpx_frames: list, offset: float, align: str,
                out_path: str, vmin: float = 1e-2, vmax: float = 3e2) -> None:
    """FLASH slices above WarpX slices, sharing a d_i axis so the box sizes compare.

    Both rows show the TARGET species alone -- FLASH n_e masked by its Si mass
    fraction, WarpX rho_piston_ions -- so the panels compare the same material.
    """
    n_cols = len(flash_frames)
    fig, axes = plt.subplots(2, n_cols, figsize=(3.5 * n_cols, 6.6), squeeze=False,
                             layout="constrained")
    norm = LogNorm(vmin=vmin, vmax=vmax)

    # Crop BOTH codes to the WarpX box rather than padding WarpX out to FLASH's extent:
    # the point is to compare the same region of space, and the deck's transverse domain
    # (4 heating-spot radii, set by the periodic-image constraint) is the smaller one.
    # What FLASH does beyond that box is real but is not something a box this wide could
    # represent, so it is called out in the title instead of shown at 1/4 scale.
    x_max = float(np.abs(warpx_frames[0]["x_di"]).max())
    z_max = float(warpx_frames[0]["z_di"].max())
    flash_x_max = max(abs(f["extent_di"][2]) for f in flash_frames)
    flash_z_max = max(f["extent_di"][1] for f in flash_frames)

    for col, (flash, warpx) in enumerate(zip(flash_frames, warpx_frames)):
        image = axes[0][col].imshow(
            np.clip(flash["density_map"], vmin, None), origin="lower",
            extent=flash["extent_di"], aspect="auto", norm=norm, cmap="inferno")
        axes[0][col].set_title(
            f"FLASH  {flash['t_gyro']:.4f} $T_{{ci}}$", fontsize=9)

        # No transpose: imshow wants [row, col] = [transverse, along-axis], and a WarpX 2D
        # covering_grid is already indexed [x, z]. Transposing drew the z=0 slab along the
        # vertical axis, which reads as a piston expanding transversely.
        axes[1][col].imshow(
            np.clip(warpx["density_map"], vmin, None), origin="lower",
            extent=(warpx["z_di"].min(), warpx["z_di"].max(),
                    warpx["x_di"].min(), warpx["x_di"].max()),
            aspect="auto", norm=norm, cmap="inferno")
        axes[1][col].set_title(
            f"WarpX  {warpx['t_gyro']:.4f} $T_{{ci}}$", fontsize=9)

        for row in (0, 1):
            axes[row][col].set_xlim(0.0, z_max)
            axes[row][col].set_ylim(-x_max, x_max)
            axes[row][col].set_xlabel(r"distance from target [$d_i$]")
            if col > 0:
                axes[row][col].set_yticklabels([])
            else:
                axes[row][col].set_ylabel(r"transverse [$d_i$]")
        for frame, ax in ((flash, axes[0][col]), (warpx, axes[1][col])):
            if np.isfinite(frame["front_di"]):
                ax.axvline(frame["front_di"], color="cyan", lw=1.0, ls="--", alpha=0.8)

    fig.colorbar(image, ax=axes.ravel().tolist(),
                 label=r"$n_\mathrm{target} / n_\mathrm{amb}$",
                 fraction=0.02, pad=0.01)
    caveat = (CAVEAT_FRONT.format(offset) if align == "front" else CAVEAT_CLOCK)
    fig.suptitle(
        "FLASH vs WarpX heater piston — target species only, 2D slices on identical "
        "$d_i$ axes\n"
        f"both CROPPED to the WarpX box; FLASH itself extends to "
        f"{flash_z_max:.0f} $d_i$ along the axis and $\\pm${flash_x_max:.0f} $d_i$ "
        f"transversely.  cyan dashed = measured front\n" + caveat, fontsize=10)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_movie(flash_frames: list, warpx_frames: list, out_path: str,
                 fps: int) -> None:
    """One frame per matched pair, stitched with ffmpeg."""
    frame_dir = out_path + "_frames"
    os.makedirs(frame_dir, exist_ok=True)
    for index, (flash, warpx) in enumerate(zip(flash_frames, warpx_frames)):
        plot_slices([flash], [warpx], 0.0, "clock",
                    os.path.join(frame_dir, f"f{index:04d}.png"))
    command = ["ffmpeg", "-y", "-framerate", str(fps),
               "-i", os.path.join(frame_dir, "f%04d.png"),
               "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-pix_fmt", "yuv420p", out_path]
    try:
        subprocess.run(command, check=True, capture_output=True)
        print(f"Saved -> {out_path}")
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"NOTE: movie not rendered ({exc}); frames kept in {frame_dir}")


def main() -> None:
    args = parse_args()
    plot_style.apply(args.publication)

    spec, scales = load_scales(args.config)
    targets = scales.flash
    if targets is None:
        raise SystemExit(f"{args.config} has no flash: block — nothing to compare against")

    flash_cfg_path = args.flash_config
    if not os.path.isabs(flash_cfg_path):
        flash_cfg_path = os.path.join(_REPO, flash_cfg_path)
    flash_cfg = analysis_utils.load_config(flash_cfg_path)
    source = flash_source.resolve(flash_cfg, flash_cfg_path)
    piston_material = str(flash_cfg.get("piston_material", "targ"))
    ambient_material = str(flash_cfg.get("ambient_material", "cham"))

    # WarpX FIRST, always: reading FLASH enables the yt plugin and poisons later WarpX
    # loads (see the import note at the top). A partially-finished run still produces a
    # figure over the time it did cover.
    warpx_paths = warpx_plotfiles(args.config, args.diag_dir)
    print(f"WarpX  : {len(warpx_paths)} plotfiles")
    warpx_frames_all = read_warpx_frames(warpx_paths, scales)
    warpx_gyro = np.array([f["t_gyro"] for f in warpx_frames_all])

    # Velocity needs raw particles, and every phase dump has to be read here -- before the
    # FLASH import below -- even though only a few columns are plotted. The grid is fixed
    # (no refinement), so one z axis serves them all.
    phase_paths = warpx_plotfiles(args.config, args.diag_dir, prefix="phase")
    if phase_paths:
        print(f"         {len(phase_paths)} phase dumps for the velocity row")
        velocity_frames = read_warpx_velocity(phase_paths, scales,
                                              warpx_frames_all[0]["z_di"])
    else:
        print("         WARNING: no phase* dumps found -- the velocity row will be empty. "
              "The field diagnostic carries no per-species current, so bulk velocity "
              "cannot be recovered from diag1.")
        velocity_frames = []

    from magshockz.common import flash_utils as fu

    t_lo_ns, t_hi_ns = (float(v) for v in spec["flash_target"]["t_window_ns"])
    all_flash = fu.find_plot_files(source.flash_dir)
    flash_times = np.array([fu.flash_time_s(p) for p in all_flash])
    in_window = np.flatnonzero((flash_times >= t_lo_ns * 1e-9)
                               & (flash_times <= t_hi_ns * 1e-9))
    flash_paths = [all_flash[i] for i in in_window]
    flash_gyro = ((flash_times[in_window] - t_lo_ns * 1e-9)
                  / targets.upstream.gyroperiod.to_value(u.s))
    print(f"FLASH  : {len(flash_paths)} dumps in {t_lo_ns}-{t_hi_ns} ns "
          f"(0-{flash_gyro.max():.4f} T_ci)")
    print(f"         WarpX covers 0-{warpx_gyro.max():.4f} T_ci of the "
          f"{scales.run_gyroperiods:.4f} T_ci target")

    warpx_fronts = np.array([f["front_di"] for f in warpx_frames_all])

    # --align front needs FLASH's starting front position, which is only known after a
    # FLASH frame is loaded -- so load the first one up front rather than passing None and
    # having the alignment silently do nothing.
    flash_fronts = None
    if args.align == "front":
        first = flash_frame(flash_paths[0], source, targets,
                            piston_material=piston_material,
                            ambient_material=ambient_material,
                            halfwidth_um=args.slice_halfwidth_um,
                            t_start_s=t_lo_ns * 1e-9)
        flash_fronts = np.array([first["front_di"]])
        print(f"         FLASH front starts at {first['front_di']:.2f} d_i")

    flash_idx, warpx_idx, offset = pick_matched_times(
        flash_gyro, warpx_gyro, args.n_times, args.align,
        flash_fronts=flash_fronts, warpx_fronts=warpx_fronts)
    if args.align == "front":
        if offset == 0.0:
            print("         WARNING: WarpX never reaches FLASH's starting front "
                  "position, so no shift was applied; falling back to clock alignment.")
        else:
            print(f"         front alignment shifts WarpX by {offset:.4f} T_ci")

    flash_frames = [
        flash_frame(flash_paths[i], source, targets, piston_material=piston_material,
                    ambient_material=ambient_material,
                    halfwidth_um=args.slice_halfwidth_um, t_start_s=t_lo_ns * 1e-9)
        for i in flash_idx]
    warpx_frames = [warpx_frames_all[i] for i in warpx_idx]
    attach_warpx_velocity(warpx_frames, velocity_frames)

    out_dir = yaml_edit.out_dir(
        os.path.basename(args.config).replace(".warpx.yaml", ""),
        args.output_dir or os.path.join(
            _REPO, "results", "warpx",
            os.path.basename(args.config).replace(".warpx.yaml", "")),
        cfg=spec, config_path=args.config)

    lineout_path = os.path.join(out_dir, "evolution_lineouts.png")
    plot_lineouts(flash_frames, warpx_frames, offset, args.align, lineout_path)
    print(f"Saved -> {lineout_path}")

    slice_path = os.path.join(out_dir, "evolution_slices.png")
    plot_slices(flash_frames, warpx_frames, offset, args.align, slice_path)
    print(f"Saved -> {slice_path}")

    if args.movie:
        movie_flash_idx, movie_warpx_idx, _ = pick_matched_times(
            flash_gyro, warpx_gyro, min(len(warpx_paths), 60), args.align,
            flash_fronts=None,
            warpx_fronts=np.array([f["front_di"] for f in warpx_frames_all]))
        movie_flash = [
            flash_frame(flash_paths[i], source, targets,
                        piston_material=piston_material,
                        ambient_material=ambient_material,
                        halfwidth_um=args.slice_halfwidth_um,
                        t_start_s=t_lo_ns * 1e-9) for i in movie_flash_idx]
        render_movie(movie_flash, [warpx_frames_all[i] for i in movie_warpx_idx],
                     os.path.join(out_dir, "evolution_slices.mp4"), args.fps)


if __name__ == "__main__":
    main()
