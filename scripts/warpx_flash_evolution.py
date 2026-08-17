# -*- coding: utf-8 -*-
"""scripts/warpx_flash_evolution.py — FLASH vs WarpX evolution, side by side.

Figures, selected with ``--figures`` (default: all):

  ``evolution``  ``evolution_lineouts.png``  1-D profiles (target-species density,
                     ambient-species density, bulk velocity, |B|, T_e) at matched times,
                     FLASH and WarpX overlaid, one column per time
                 ``evolution_slices.png``    2-D target-species slices at the same times,
                     FLASH above WarpX
  ``compare``    ``piston_comparison.png``   the same two codes' pistons ROTATED so they
                     expand up the page, at TRUE ASPECT RATIO, three times; both species
                     overlaid in one panel under their own colormaps, with |B| over the top

``--movie`` animates one of those two over EVERY WarpX dump rather than the handful of
matched times the stills use: ``--movie`` (or ``--movie compare``) writes
``piston_comparison.mp4``, the same panels with FLASH beside WarpX instead of above it;
``--movie slices`` writes ``evolution_slices.mp4``, the stacked target-species slices.
Each WarpX dump is paired with the FLASH dump nearest it on the aligned clock, and FLASH
has roughly half as many over the window, so its panel holds for two frames at a time --
which is why every panel carries its own time.
  ``shock``      ``shock_rh_prediction.png`` WarpX only: line-outs along the shock normal
                     showing the ambient piled up AHEAD of the piston, with the
                     perpendicular-MHD jump predicted from the measured upstream
  ``streaks``    ``shock_streaks.png``       WarpX only: position-time streaks of both
                     species, |B| and T_i over every dump
  ``profiles``   ``flash_vs_warpx_profiles.png``  the poster figure: ONE matched time,
                     three stacked panels (density, ambient velocity, ambient
                     temperatures) with both codes overlaid, and a scorecard of the
                     shocked layer's jumps beside them.  Always drawn at publication
                     font sizes -- it exists to be read from a metre away.

WHERE THE SHOCK FIGURES GET THEIR NUMBERS.  Nothing is taken from the deck's initial
condition.  Each dump's line-out is split by ``analysis.warpx.metrics.locate_shock_regions``
into piston, the shocked-ambient layer between the contact discontinuity and the shock,
and pristine upstream; the upstream band gives n, |B|, T_e, T_i, and the shock speed is a
straight-line fit to the pile-up front over the whole series.  The one input that cannot
be measured is the ion mass -- the deck runs at a reduced ``m_i/(Z m_e)``, so ``rho`` (and
hence ``v_A``, ``c_s`` and both Mach numbers) is built from the deck's own ion, which is
the ion the run actually integrated.

WHAT "MATCHED" MEANS.  The deck runs at a reduced mass ratio and an arbitrary reference
density, so nothing absolute is comparable — see ``magshockz/init/warpx/units.py``.  Every
axis here is therefore in ion units: length in ``d_i``, time in ``omega_ci^-1``, density
relative to the ambient, |B| relative to ``B0``.  In those units one FLASH gyroperiod IS
one WarpX gyroperiod, which is what makes the panels comparable at all.  Internally times
are carried as ``t/T_ci`` and converted for display only -- see :func:`wci`.

BOTH SIDES ARE REDUCED ON AXIS.  FLASH's profiles are a ray along the shock normal; the
WarpX ones average the band ``|x| < AXIS_BAND_DI``.  A full-box transverse average would
not be the same reduction — the piston is a patch of radius ``r_spot`` inside a box many
times wider, so averaging dilutes it by the fill fraction and erases the diamagnetic cavity
along with the contrast.  The 2-D maps and the front estimator keep the whole box.

SPECIES ARE NEVER SUMMED.  The piston comparison is target species against target
species: FLASH's electron density masked by its Si mass fraction (``targ``), against
WarpX's ``rho_piston_ions`` divided by the elementary charge -- which is Z_pist times the
ion density, i.e. that species' own contribution to n_e, the same quantity FLASH's mask
gives.  Both are then normalized by the ambient ELECTRON density, so the piston panel
peaks at the matched contrast.  A combined n_e would fold FLASH's EOS ionization state
(Zbar 3.7 in the ambient, ~14 in the piston) into a comparison whose deck imposes ONE
charge state per species by construction.

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
2. *There are no in-plane magnetic field lines to draw.*  Both runs are perpendicular
   shocks: FLASH applies 7 T along z while ``flash_slice`` cuts the x-y plane, and the
   deck sets ``By = B0`` with its 2-D plane in x-z.  The field pierces the page in both,
   so ``piston_comparison.png`` shows |B| as iso-contours (equivalently, the end-on
   field-line density) rather than as streamlines, which would trace only PIC noise.
3. *The WarpX box is transversely narrower.*  Its x half-width is ``geometry
   .transverse_halfwidth_di`` -- a few heating-spot radii, kept small enough that the
   spot's periodic images stay clear of the spot -- while the FLASH domain spans ~24 d_i
   transversely.  The 2-D panels share a d_i axis so this is visible rather than hidden
   by independent scales.

Usage
-----
    conda activate analysis
    python scripts/warpx_flash_evolution.py --config runs/magshockz_2d_heater.warpx.yaml \\
        [--figures compare shock streaks profiles] [--compare-times 3] [--n-times 5] \\
        [--profile-time 1.27] [--align clock|front] [--diag-dir ...] [--cache] \\
        [--movie [compare|slices]] [--fps 10] [--jobs 32] [--pub]

Run in the `analysis` conda env (yt).
"""

import argparse
import glob
import os
import pickle
import subprocess
import sys
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Colormap, LinearSegmentedColormap, LogNorm, Normalize

_HERE = os.path.dirname(os.path.abspath(__file__))

from magshockz.common import analysis_utils
from magshockz.common import flash_source
import astropy.units as u
from astropy.constants import c, e, m_e

from magshockz.analysis.warpx import metrics as wx_metrics
from magshockz.init.warpx import config as spec_config
from magshockz.init.warpx import deck as deck_module
from magshockz.init.warpx import units
from magshockz.common import perpendicular_shock as ps
from magshockz.common import piston_profile as pp
from magshockz.common import plot_style
from magshockz.common import yaml_edit

# Both codes' dumps are read in one process, which used to require care: importing
# flash_utils called yt.enable_plugins(), registering the flash2osiris ("flash", ...)
# fields GLOBALLY, and yt then validated them against every dataset opened afterwards --
# so a WarpX plotfile died on ('flash', 'velz'). flash_utils no longer enables the plugin
# at import (it needs no plugin field), so there is nothing left to order or suspend.

_REPO = os.path.abspath(os.path.join(_HERE, ".."))
CM_PER_UM = 1.0e-4

#: Checked against the generator's own list rather than mirrored, so a rename there
#: fails here loudly instead of leaving the script asking for a missing plotfile field.
PISTON_IONS = "piston_ions"
AMBIENT_IONS = "amb_ions"
PISTON_ELECTRONS = "piston_electrons"
assert {PISTON_IONS, AMBIENT_IONS, PISTON_ELECTRONS} <= set(deck_module.SPECIES_NAMES)


#: The figures this script can make, in the order main() writes them.  ``evolution`` is
#: the original pair (evolution_lineouts / evolution_slices); the rest were added for the
#: shock question and each stands alone, so --figures can ask for just one.
FIGURES = ("evolution", "compare", "shock", "streaks", "profiles")

#: Which figure ``--movie`` animates.  Both run over every WarpX dump -- a movie is the
#: one output with no reason to subsample the series.
MOVIE_FIGURES = ("compare", "slices")

#: What each of those writes, beside its own ``<name>_frames/`` directory of PNGs.
MOVIE_FILES = {"compare": "piston_comparison.mp4", "slices": "evolution_slices.mp4"}


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
    parser.add_argument("--figures", nargs="+", default=list(FIGURES),
                        choices=list(FIGURES),
                        help="which figures to make (default: all of them)")
    parser.add_argument("--compare-times", type=int, default=3,
                        help="columns in piston_comparison.png and shock_profile.png "
                             "(default: %(default)s)")
    parser.add_argument("--profile-time", type=float,
                        help="the one matched time, in omega_ci^-1, that "
                             "flash_vs_warpx_profiles.png shows (default: the last time "
                             "both codes cover)")
    parser.add_argument("--slice-halfwidth-um", type=float,
                        help="FLASH slice transverse half-width; default matches the "
                             "WarpX transverse box, so both codes' panels cover the "
                             "same region")
    parser.add_argument("--movie", nargs="?", const="compare", choices=MOVIE_FIGURES,
                        default=None,
                        help="render a movie over EVERY WarpX dump: 'compare' (the "
                             "default when the flag is bare) animates the piston "
                             "comparison with the two codes SIDE BY SIDE, 'slices' "
                             "animates the stacked target-species slices")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--jobs", type=int, default=1,
                        help="movie frames to render in parallel. Each worker re-reads "
                             "the dumps its own frames need, so this is for a compute "
                             "node (salloc), not the login node.")
    parser.add_argument("--cache", metavar="PATH", nargs="?", const="",
                        help="reuse the measured frames from PATH if it exists, else "
                             "write them there (default path: <output-dir>/frames.pkl). "
                             "Reading the dumps dominates the runtime, so this turns a "
                             "re-plot into seconds -- use it when iterating on the "
                             "FIGURE, not when the data has changed.")
    parser.add_argument("--output-dir")

    overlay = parser.add_argument_group(
        "piston_comparison.png", "the two-species overlay and its magnetic overlay")
    overlay.add_argument("--piston-cmap", default="default",
                         help="colormap for the piston species; 'default' is the "
                              "built-in warm ramp (default: %(default)s)")
    overlay.add_argument("--ambient-cmap", default="default",
                         help="colormap for the ambient species; 'default' is the "
                              "built-in cool ramp (default: %(default)s)")
    overlay.add_argument("--piston-range", type=float, nargs=2, default=(1e-2, 3e2),
                         metavar=("VMIN", "VMAX"),
                         help="piston colour scale, n/n_amb (default: %(default)s)")
    overlay.add_argument("--ambient-range", type=float, nargs=2, default=(1e-1, 1e1),
                         metavar=("VMIN", "VMAX"),
                         help="ambient colour scale, n/n_amb (default: %(default)s)")
    overlay.add_argument("--b-overlay", choices=("contour", "stipple", "none"),
                         default="contour",
                         help="how |B| is drawn; the field is PERPENDICULAR to the "
                              "plane, so there are no in-plane field lines to draw "
                              "(default: %(default)s)")
    # No level at 1.0: that is the undisturbed upstream itself, so a contour there is
    # traced by PIC noise across the whole un-shocked box and buries the real structure.
    overlay.add_argument("--b-levels", type=float, nargs="+",
                         default=(0.25, 0.5, 1.5, 3.0, 6.0),
                         help="iso-|B|/B0 contour levels (default: %(default)s)")
    overlay.add_argument("--b-smooth", type=float, default=6.0,
                         help="Gaussian smoothing, in map cells, before contouring the "
                              "PIC-noisy |B| (default: %(default)s)")
    overlay.add_argument("--alpha-gamma", type=float, default=0.7,
                         help="exponent on each species' alpha ramp; <1 makes faint "
                              "material more visible (default: %(default)s)")

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

class FrameCache:
    """Path-keyed store of already-measured frames, so a re-plot skips the dumps.

    Reading the dumps is ~99% of this script's runtime, and the measurement depends only
    on the dump and the deck scales -- so keying on the path is enough, PROVIDED the run
    has not been rewritten underneath. It is opt-in (``--cache``) for exactly that reason.
    """

    def __init__(self, path: str | None) -> None:
        self.path = path
        self.frames: dict[str, dict] = {}
        self.dirty = False
        if path and os.path.isfile(path):
            with open(path, "rb") as handle:
                self.frames = pickle.load(handle)
            print(f"cache  : {len(self.frames)} frames from {path}")

    def get(self, key: str, read: Callable[[], dict]) -> dict:
        if key not in self.frames:
            self.frames[key] = read()
            self.dirty = True
        return self.frames[key]

    def save(self) -> None:
        if not (self.path and self.dirty):
            return
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "wb") as handle:
            pickle.dump(self.frames, handle)
        print(f"cache  : wrote {len(self.frames)} frames to {self.path}")


def read_warpx_frames(paths: list[str], scales: units.DeckScales,
                      cache: FrameCache | None = None) -> list[dict]:
    """Every plotfile's line-outs.

    Line-outs ONLY: the 2-D maps come from :func:`read_warpx_maps` for the handful of
    frames actually plotted.  A full-box field costs ~2.5 s against ~0.1 s for the
    on-axis band, so reading maps for all ~70 dumps to plot three of them would dominate
    the runtime -- and the streak figure needs every dump's line-out, not its map.
    """
    if cache is None:
        return [warpx_frame(path, scales) for path in paths]
    return [cache.get(f"warpx:v{FRAME_SCHEMA}:{p}", lambda p=p: warpx_frame(p, scales))
            for p in paths]


def read_warpx_maps(paths: list[str], indices: list[int], frames: list[dict],
                    scales: units.DeckScales,
                    cache: FrameCache | None = None) -> list[dict]:
    """COPIES of the selected frames with their 2-D maps attached.

    Copies, not in-place updates, because ``frames`` are the cache's own dicts: merging
    ~50 MB of maps into them would put the maps back into ``frames.pkl`` on the next
    save, under the key that is meant to hold line-outs only.
    """
    out = []
    for index in indices:
        path = paths[index]
        read = lambda p=path: warpx_maps(p, scales)
        maps = (cache.get(f"warpxmap:v{FRAME_SCHEMA}:{path}", read)
                if cache is not None else read())
        out.append({**frames[index], **maps})
    return out


#: Longest side a stored 2-D map may have. A slice panel is ~500 px at the figure's own
#: dpi, so anything beyond this is thrown away by the renderer anyway.
MAP_MAX_SIDE = 800

#: Times are DISPLAYED in inverse ion gyrofrequencies.  Frames store t/T_ci and the
#: gyroperiod is 2*pi/omega_ci, so the displayed number is 2*pi times the stored one.
#: Only the presentation changes -- the trajectory fit, the time matching and the cache
#: all stay in T_ci, which is the unit in which one FLASH gyroperiod IS one WarpX one.
INVERSE_OMEGA_PER_GYROPERIOD = 2.0 * np.pi

#: LaTeX for that unit, so every title and axis label spells it the same way.
TIME_UNIT = r"\omega_{ci}^{-1}"


def wci(t_gyro):
    """A time stored in gyroperiods, expressed in inverse gyrofrequencies."""
    return np.asarray(t_gyro, dtype=float) * INVERSE_OMEGA_PER_GYROPERIOD


#: Half-width, in d_i, of the band the WarpX z-profiles are taken over. FLASH's profiles
#: are an on-axis ray, so a full-box transverse average is not the same reduction: the
#: plume is ablated from a spot of radius r_spot inside a +-35 d_i box, and averaging
#: dilutes it by the fill fraction (contrast 4.20 on axis vs 0.28 averaged, |B| cavity
#: 0.37 vs 0.99). With a target that spans the box the full-box average is worse still --
#: it is then dominated by un-ablated target that never moves.
AXIS_BAND_DI = 1.0

#: Bumped whenever warpx_frame's output changes, so a stale --cache is not reused.
FRAME_SCHEMA = 4

#: The same, for flash_frame. Separate from FRAME_SCHEMA so adding a FLASH field does not
#: throw away the WarpX frames, which are far more expensive to re-measure.
FLASH_SCHEMA = 4

#: Ambient density, relative to each code's own reference, below which that code's
#: ambient temperature and velocity are not drawn.  ONE rule for both codes: below this
#: there is no ambient population to have a temperature, and what remains is PIC noise on
#: the WarpX side and a mixture dominated by piston material on the single-fluid FLASH
#: side.  Display only -- the scorecard's bands sit in ambient-rich gas by construction.
MIN_AMBIENT_FRACTION = 0.1

#: Band-mean density, relative to the ambient reference, below which a species' own
#: temperature and bulk velocity are reported as nan rather than as a number.
MIN_SPECIES_FRACTION = 1.0e-2

#: Floor on the gap between the shock and the upstream band, in d_i.  The magnetic foot
#: and electron preheat run ahead of the density jump by a few ion inertial lengths, and
#: a band inside them measures the precursor rather than the upstream -- see
#: :func:`~magshockz.analysis.warpx.metrics.locate_shock_regions`.
UPSTREAM_GAP_DI = 3.0


def _map_stride(n_x: int, n_z: int) -> int:
    """Decimation that keeps both map axes under :data:`MAP_MAX_SIDE`."""
    return max(1, -(-max(n_x, n_z) // MAP_MAX_SIDE))


class _Plotfile:
    """One WarpX plotfile's geometry in ion units, plus band- and full-grid readers.

    Both readers hand back arrays indexed ``[x, z]`` over the ``+z`` half.  Only that
    half is kept: the slab is symmetric about z = 0 and expands both ways, so the +z lobe
    is the direct analogue of FLASH's outward LOS.
    """

    def __init__(self, path: str, scales: units.DeckScales) -> None:
        import yt

        self.dataset = yt.load(path)
        self.scales = scales
        d_i = scales.ion_skin_depth.to_value(u.m)
        left = np.asarray(self.dataset.domain_left_edge, dtype=float)
        right = np.asarray(self.dataset.domain_right_edge, dtype=float)
        n_x, n_z = (int(v) for v in self.dataset.domain_dimensions[:2])

        self.x_edges = np.linspace(left[0], right[0], n_x + 1)
        z_edges = np.linspace(left[1], right[1], n_z + 1)
        self.x_di = 0.5 * (self.x_edges[:-1] + self.x_edges[1:]) / d_i
        self.z_di = 0.5 * (z_edges[:-1] + z_edges[1:]) / d_i
        self.outward = self.z_di >= 0.0
        self.t_gyro = float(self.dataset.current_time) / scales.gyroperiod.to_value(u.s)

        # Band bounds taken from the CELL EDGES, so the band grid holds exactly the cells
        # |x_centre| < AXIS_BAND_DI would select on the full grid -- not one more or less.
        self._band = np.flatnonzero(np.abs(self.x_di) < AXIS_BAND_DI)

    def band(self, name: str) -> np.ndarray:
        """One field over the on-axis band only, ``[x, z]`` -- ~30x cheaper than full()."""
        lo, hi = int(self._band[0]), int(self._band[-1]) + 1
        left = np.asarray(self.dataset.domain_left_edge, dtype=float).copy()
        left[0] = self.x_edges[lo]
        grid = self.dataset.covering_grid(
            level=0, left_edge=left,
            dims=[hi - lo, int(self.dataset.domain_dimensions[1]), 1])
        return np.asarray(grid["boxlib", name]).squeeze()[:, self.outward]

    def full(self, name: str) -> np.ndarray:
        """One field over the whole box, ``[x, z]``."""
        if not hasattr(self, "_grid"):
            self._grid = self.dataset.covering_grid(
                level=0, left_edge=self.dataset.domain_left_edge,
                dims=self.dataset.domain_dimensions)
        values = np.asarray(self._grid["boxlib", name]).squeeze()
        # Every map and profile assumes [x, z] indexing -- the un-transposed imshow in
        # plot_slices and the transposed one in plot_piston_comparison both depend on it.
        if values.shape != (self.x_di.size, self.z_di.size):
            raise RuntimeError(f"expected a [x, z] = {(self.x_di.size, self.z_di.size)} "
                               f"grid for {name}, got {values.shape}")
        return values[:, self.outward]


def _proper_velocity_to_speed(u_over_c: np.ndarray) -> np.ndarray:
    """WarpX's per-species ``u`` (= gamma v / c) as a speed in m/s.

    ``particle_fields_to_plot`` deposits the number-weighted mean of ``uz`` per cell, so
    strictly this converts the mean proper velocity rather than averaging the velocity.
    At the |u| ~ 0.03 these runs reach, gamma differs from 1 by 5e-4 and the distinction
    is far below the PIC noise on the same profile.
    """
    return c.si.value * u_over_c / np.sqrt(1.0 + u_over_c**2)


def warpx_frame(path: str, scales: units.DeckScales) -> dict:
    """One WarpX plotfile's on-axis line-outs, in ion units (no 2-D maps).

    Densities are normalized to the ambient electron density and |B| to B0; the ambient
    temperatures and bulk velocity are kept ABSOLUTE (eV, m/s) because the
    Rankine-Hugoniot solve needs a real plasma state, not a ratio.

    Temperature and velocity are DENSITY-WEIGHTED across the band, unlike the densities
    and |B|.  WarpX writes exactly 0 for a per-species T or u in any cell holding none of
    that species, so a plain transverse mean would drag the ambient's temperature towards
    zero wherever the diamagnetic cavity has swept the band clear of ambient ions.
    """
    plotfile = _Plotfile(path, scales)
    z_di = plotfile.z_di[plotfile.outward]

    reference_density = scales.upstream.electron_density.to_value(u.m**-3)
    piston = plotfile.band(f"rho_{PISTON_IONS}") / e.si.value / reference_density
    ambient = plotfile.band(f"rho_{AMBIENT_IONS}") / e.si.value / reference_density
    b_mag = (np.hypot(np.hypot(plotfile.band("Bx"), plotfile.band("By")),
                      plotfile.band("Bz")) / scales.magnetic_field.to_value(u.T))

    mean = lambda m: m.mean(axis=0)

    def weighted(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Density-weighted transverse average, nan where the species is absent.

        The floor is not cosmetic.  Inside the swept cavity the ambient density is ~0 but
        not exactly 0, and dividing by it turns per-cell noise in a field that is itself
        ~0 into averages spanning 30 decades -- which a log axis then renders as a
        full-height spike.  Below :data:`MIN_SPECIES_FRACTION` of the reference density
        there is no population to take a temperature of, so we say so.
        """
        total = weights.sum(axis=0)
        usable = total > MIN_SPECIES_FRACTION * weights.shape[0]
        return np.divide((values * weights).sum(axis=0), total,
                         out=np.full(total.shape, np.nan), where=usable)

    ambient_profile = mean(ambient)
    frame = {
        "t_gyro": plotfile.t_gyro,
        "z_di": z_di,
        "piston_profile": mean(piston),
        "ambient_profile": ambient_profile,
        "b_profile": mean(b_mag),
        "te_profile": (weighted(plotfile.band(f"T_{PISTON_ELECTRONS}"), piston)
                       / scales.upstream.electron_temperature.to_value(u.eV)),
        "amb_te_ev": weighted(plotfile.band("T_amb_electrons"), ambient),
        "amb_ti_ev": weighted(plotfile.band(f"T_{AMBIENT_IONS}"), ambient),
        "amb_vz": weighted(_proper_velocity_to_speed(
            plotfile.band(f"uz_{AMBIENT_IONS}")), ambient),
        "piston_vz": weighted(_proper_velocity_to_speed(
            plotfile.band(f"uz_{PISTON_IONS}")), piston),
    }

    # The piston front is measured on the SAME on-axis band as everything else here. The
    # full-box average this used to take never reaches n_amb for a patch inside a +-35 d_i
    # box, so it returned nan for every frame and only the FLASH front was ever drawn.
    frame["front_di"] = pp.front_position(z_di, frame["piston_profile"], level=1.0)
    regions = wx_metrics.locate_shock_regions(z_di, ambient_profile,
                                              frame["piston_profile"],
                                              min_upstream_gap=UPSTREAM_GAP_DI)
    frame["shock_di"] = regions.shock
    frame["contact_di"] = regions.contact
    return frame


def warpx_maps(path: str, scales: units.DeckScales) -> dict:
    """One plotfile's 2-D maps, coarsened for the figure.

    The line-outs are taken at FULL resolution in :func:`warpx_frame`; only the stored
    maps are decimated, and only for memory.  At 4128 x 5232 the un-coarsened maps of a
    whole series come to ~19 GB and the login node's OOM killer takes the process with no
    traceback.  A panel is ~500 px wide, so the full grid is ~8x oversampled.
    """
    plotfile = _Plotfile(path, scales)
    reference_density = scales.upstream.electron_density.to_value(u.m**-3)
    stride = _map_stride(plotfile.x_di.size, int(plotfile.outward.sum()))
    coarse = lambda m: m[::stride, ::stride]

    return {
        "x_di": plotfile.x_di[::stride],
        "density_map": coarse(plotfile.full(f"rho_{PISTON_IONS}")
                              / e.si.value / reference_density),
        "ambient_map": coarse(plotfile.full(f"rho_{AMBIENT_IONS}")
                              / e.si.value / reference_density),
        "b_map": coarse(np.hypot(np.hypot(plotfile.full("Bx"), plotfile.full("By")),
                                 plotfile.full("Bz"))
                        / scales.magnetic_field.to_value(u.T)),
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
    x = np.asarray(data[(PISTON_IONS, "particle_position_x")], dtype=float)
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
    # Same on-axis band as the field rows, so the whole column is one reduction.
    keep = (z >= 0.0) & (np.abs(x)
                         < AXIS_BAND_DI * scales.ion_skin_depth.to_value(u.m))
    bulk = pp.weighted_bin_average(z[keep], velocity[keep], weight[keep], edges)
    return bulk / scales.piston_speed.to_value(u.m / u.s)


def read_warpx_velocity(phase_paths: list[str], scales: units.DeckScales,
                        z_di: np.ndarray) -> list[dict]:
    """Every ``phase`` dump's velocity profile.

    Reading all of them at once costs little and keeps the pairing -- which needs the
    FLASH-dependent alignment -- as pure arithmetic in :func:`attach_warpx_velocity`.
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
    # |B|, not a component: the slice plane is perpendicular to the applied field (see
    # plot_piston_comparison), so there is no in-plane field to resolve into components
    # and the total is the only thing the plane can show.
    b_slice = fu.flash_slice(path, source.line_start, source.line_end,
                             field=("gas", "magnetic_field_magnitude"),
                             halfwidth_um=halfwidth_um, unit="T")

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
        # Named to match the WarpX frame's per-species fields, but FLASH is SINGLE-FLUID:
        # one Te, one Ti and one velocity per cell, shared by whatever materials that cell
        # holds.  Where the ambient is what the cell is made of -- the shocked layer and
        # everything ahead of it, which is all the comparison averages over -- these ARE
        # the ambient's.  Deeper in the plume they are the mixture's, which is why
        # MIN_AMBIENT_FRACTION blanks them for display.
        "amb_te_ev": lineout["Te"].to("eV").value,
        "amb_ti_ev": lineout["Ti"].to("eV").value,
        "amb_vz": lineout["v_para"].to("m/s").value,
        "density_map": sliced["img"] / n_amb_cm3,
        "ambient_map": ambient_slice["img"] / n_amb_cm3,
        "b_map": b_slice["img"] / upstream.magnetic_field.to_value(u.T),
        "extent_di": (los_lo / d_i_um, los_hi / d_i_um,
                      (tr_lo - sliced["los_transverse_um"]) / d_i_um,
                      (tr_hi - sliced["los_transverse_um"]) / d_i_um),
        "front_di": pp.front_position(x_um / d_i_um, n_piston / n_amb_cm3, level=1.0),
    }


# ---------------------------------------------------------------------------
# The WarpX shock: where it is, how fast it runs, what MHD says it should do
# ---------------------------------------------------------------------------

def fit_shock_trajectory(frames: list[dict],
                         scales: units.DeckScales) -> pp.FrontTrajectory | None:
    """Straight-line fit to the ambient pile-up front, in ``d_i`` per ``T_ci``.

    Left in ``T_ci`` because the frames' own times are; the streak figure divides by
    ``2*pi`` to quote it as ``d_i*omega_ci``.

    The shock speed the Rankine-Hugoniot solve needs is a LAB-frame speed, and the only
    honest way to get it from a run whose shock is still forming is to watch the front
    move.  Frames with no resolvable pile-up (early, before the ambient has been swept
    up) drop out on their own -- ``shock_di`` is nan there.
    """
    times = np.array([f["t_gyro"] for f in frames])
    fronts = np.array([f["shock_di"] for f in frames])
    if np.count_nonzero(np.isfinite(fronts)) < 2:
        return None
    return pp.fit_front_trajectory(times, fronts)


def shock_speed(trajectory: pp.FrontTrajectory, scales: units.DeckScales) -> u.Quantity:
    """``d_i`` per ``T_ci`` from the fit, restored to m/s."""
    return (trajectory.speed * scales.ion_skin_depth / scales.gyroperiod).to(u.m / u.s)


#: Quantities the RH solve predicts, as (frame key, how the jump scales it).  Densities
#: and the transverse field go as r, temperatures as T2/T1, pressure as p2/p1; the
#: lab-frame velocity is handled separately because it is the only one that is not a
#: multiple of its upstream value.
_JUMP_KEYS = (("ambient_profile", "r"), ("b_profile", "r"),
              ("amb_te_ev", "T_ratio"), ("amb_ti_ev", "T_ratio"),
              ("amb_pressure", "p_ratio"))


def with_ambient_pressure(frame: dict, scales: units.DeckScales) -> dict:
    """Copy of ``frame`` carrying the ambient thermal pressure ``n_e T_e + n_i T_i``.

    In eV m^-3.  This 2-species deck stores no pressure, and the ambient's own partial
    pressure -- not the total, which the piston also feeds -- is what the jump's
    ``p2/p1`` applies to.
    """
    charge_number = float(units.as_particle(scales.upstream.ion).charge_number)
    # ambient_profile is a normalized copy of the AMBIENT population's n_e (Z n_i).
    electron_density = (frame["ambient_profile"]
                        * scales.upstream.electron_density.to_value(u.m**-3))
    return {**frame,
            "amb_pressure": (electron_density * frame["amb_te_ev"]
                             + electron_density / charge_number * frame["amb_ti_ev"])}


def warpx_shock_state(frame: dict, scales: units.DeckScales, v_shock: u.Quantity,
                      gamma: float = ps.GAMMA_DEFAULT) -> dict:
    """Measure one frame's upstream state and solve the perpendicular MHD jump.

    Everything on the upstream side is MEASURED from this frame's own line-out, over the
    band :func:`~magshockz.analysis.warpx.metrics.locate_shock_regions` places ahead of
    the shock -- not taken from the deck's initial condition, so any drift or preheat the
    run developed upstream is in the Mach numbers rather than hidden by them.

    The ONE thing that cannot be measured is the ion mass: the deck runs at a reduced
    ``m_i/(Z m_e)``, so ``rho`` is built from the deck's own ion.  That is the same ion
    the run integrated, so ``v_A`` and ``c_s`` are the run's real speeds.

    Returns the ``jump`` dict, the upstream/downstream band averages, the predicted
    downstream values, and the regions, all keyed for :func:`plot_shock_profile`.
    """
    z_di = frame["z_di"]
    regions = wx_metrics.locate_shock_regions(z_di, frame["ambient_profile"],
                                              frame["piston_profile"],
                                              min_upstream_gap=UPSTREAM_GAP_DI)

    ion = units.as_particle(scales.upstream.ion)
    charge_number = float(ion.charge_number)
    reference_density = scales.upstream.electron_density.to_value(u.m**-3)

    def average(key: str, band: tuple[float, float]) -> float:
        return wx_metrics.band_average(z_di, frame[key], band)

    keys = ("ambient_profile", "b_profile", "amb_te_ev", "amb_ti_ev", "amb_vz",
            "amb_pressure")
    upstream = {k: average(k, regions.upstream) for k in keys}
    downstream = {k: average(k, regions.downstream) for k in keys}

    magnetic_field = upstream["b_profile"] * scales.magnetic_field
    density = upstream["ambient_profile"] * reference_density * u.m**-3
    state = units.Upstream(
        ion=ion, electron_density=density, magnetic_field=magnetic_field,
        electron_temperature=upstream["amb_te_ev"] * u.eV,
        ion_temperature=upstream["amb_ti_ev"] * u.eV)

    # c_s from the same two-temperature form the FLASH RH figure uses (gamma_e = gamma_i
    # = gamma), so M_s is the one the single-fluid jump assumes; v_A from plasmapy on the
    # deck's own reduced ion.
    sound = ps.sound_speed(density, upstream["amb_te_ev"] * u.eV,
                           density / charge_number, upstream["amb_ti_ev"] * u.eV,
                           (density / charge_number * ion.mass).to(u.kg / u.m**3),
                           gamma_e=gamma, gamma_i=gamma).to(u.m / u.s)
    inflow = abs(v_shock - upstream["amb_vz"] * u.m / u.s)
    jump = ps.solve_from_speeds(float(inflow.to_value(u.m / u.s)),
                                float(sound.to_value(u.m / u.s)),
                                float(state.alfven_speed.to_value(u.m / u.s)), gamma)

    predicted = {key: upstream[key] * jump[ratio] for key, ratio in _JUMP_KEYS}
    # Lab frame: the shock-frame inflow is slowed by r, so the downstream material is
    # dragged forward at v_sh - V1/r rather than scaled from an upstream that is at rest.
    predicted["amb_vz"] = float((v_shock - inflow / jump["r"]).to_value(u.m / u.s))

    return {"regions": regions, "jump": jump, "upstream": upstream,
            "downstream": downstream, "predicted": predicted,
            "v_shock": v_shock, "state": state,
            "upstream_is_pristine": upstream_is_pristine(upstream, scales)}


#: Factor by which the measured upstream may exceed the deck's initial condition before
#: the band is called contaminated.  Generous, because it is meant to catch a precursor
#: that has swallowed the far field, not to police PIC noise on a few hundred cells.
PRISTINE_TOLERANCE = 3.0


def upstream_is_pristine(upstream: dict, scales: units.DeckScales) -> bool:
    """Whether the upstream band still looks like the run's initial condition.

    The band is the outermost slice of the box, so it is upstream only for as long as the
    box has undisturbed gas in it.  These runs are PERIODIC in z and the slab expands both
    ways, so the two lobes' precursors eventually meet through the far boundary and the
    "upstream" quietly becomes preheated gas -- at which point c_s is too large, M_s too
    small, and the predicted jump too weak.  Better to say so on the figure than to
    publish a Mach number built on it.
    """
    initial = scales.upstream
    return bool(
        upstream["amb_te_ev"] < PRISTINE_TOLERANCE * initial.electron_temperature.to_value(u.eV)
        and upstream["amb_ti_ev"] < PRISTINE_TOLERANCE * initial.ion_temperature.to_value(u.eV)
        and upstream["b_profile"] < PRISTINE_TOLERANCE)


# ---------------------------------------------------------------------------
# The two codes side by side: one time, three profiles, one scorecard
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CodeProfiles:
    """One code's line-outs at one time, with everything needed to normalize them.

    ``reference`` is that code's own unperturbed ambient -- FLASH's chamber IC or the
    deck's -- which is what every normalized row in ``frame`` is already quoted against,
    and what restores a measured band to a real plasma state.
    """

    name: str
    color: str
    frame: dict
    reference: units.Upstream
    layer: wx_metrics.ShockedLayer
    alfven_speed: u.Quantity


def upstream_alfven_speed(frame: dict, band: tuple[float, float],
                          reference: units.Upstream) -> u.Quantity:
    """Alfven speed of the upstream this frame HAS, not the one the run started with.

    ``ambient_profile`` and ``b_profile`` are normalized to the code's own ambient density
    and B0, so multiplying them back through ``reference`` gives a real plasma state for
    whichever code produced the frame -- FLASH's Al 6+ chamber, or the deck's reduced ion.
    Both upstreams move: FLASH's chamber has rarefied to ~0.35 of its initial density by
    the end of the window, and quoting a shocked-layer speed against the initial v_A would
    overstate the Mach number by ~1.7x.
    """
    density = wx_metrics.band_average(frame["z_di"], frame["ambient_profile"], band)
    field = wx_metrics.band_average(frame["z_di"], frame["b_profile"], band)
    state = units.Upstream(ion=reference.ion,
                           electron_density=density * reference.electron_density,
                           magnetic_field=field * reference.magnetic_field,
                           electron_temperature=reference.electron_temperature,
                           ion_temperature=reference.ion_temperature)
    return state.alfven_speed


#: How far the upstream band's density may drift from the run's initial ambient before
#: the frame is no longer a fair comparison point -- see :func:`has_usable_upstream`.
UPSTREAM_DRIFT_TOLERANCE = 0.25


def has_usable_upstream(layer: wx_metrics.ShockedLayer) -> bool:
    """Whether this frame still has undisturbed gas ahead of its shock.

    Two conditions, one per way the band goes bad.  The density must still be the initial
    ambient's, because ``v_A`` and the compression ratio are read off it.  And the
    upstream must be COLDER than the layer it is being compared with -- an upstream hotter
    than the shocked gas is not an upstream, it is the far lobe's precursor arrived
    through the periodic boundary.
    """
    return bool(abs(layer.upstream_density - 1.0) < UPSTREAM_DRIFT_TOLERANCE
                and layer.te_upstream < layer.te_downstream
                and layer.ti_upstream < layer.ti_downstream)


def last_usable_pair(frames: list[dict], warpx_idx: list[int],
                     reference: units.Upstream) -> int:
    """Index into the matched pairs of the LAST time WarpX still has an upstream.

    The deck is periodic in z and the slab expands both ways, so the two lobes' precursors
    eventually meet through the far boundary and the "upstream" band quietly becomes
    preheated gas -- by the end of this run it reads 7.5 keV against a 9.8 eV initial
    condition, hotter than the shocked layer itself.  A figure whose whole content is
    ratios against that band must not default to that time.

    Falls back to the last pair, with a warning, if no time qualifies.
    """
    for pair in reversed(range(len(warpx_idx))):
        layer = code_profiles("WarpX", WARPX_COLOR, frames[warpx_idx[pair]],
                              reference).layer
        if has_usable_upstream(layer):
            return pair

    print("         WARNING: no matched time has an undisturbed WarpX upstream; the "
          "profile figure falls back to the last one, whose ratios are quoted against "
          "preheated gas.")
    return len(warpx_idx) - 1


def code_profiles(name: str, color: str, frame: dict,
                  reference: units.Upstream) -> CodeProfiles:
    """Measure one code's frame with the estimator the other code is measured by.

    The regions are located twice -- once here to find the band v_A is measured over, and
    once inside :func:`~magshockz.analysis.warpx.metrics.measure_shocked_layer` -- because
    the Alfven speed is an input to the layer and the band is an output of it.  The
    function is pure and costs microseconds, so the alternative (threading the regions
    through the API) buys nothing.
    """
    regions = wx_metrics.locate_shock_regions(frame["z_di"], frame["ambient_profile"],
                                              frame["piston_profile"],
                                              min_upstream_gap=UPSTREAM_GAP_DI)
    alfven_speed = upstream_alfven_speed(frame, regions.upstream, reference)
    layer = wx_metrics.measure_shocked_layer(
        frame["z_di"], frame["ambient_profile"], frame["piston_profile"],
        velocity=frame["amb_vz"], electron_temperature=frame["amb_te_ev"],
        ion_temperature=frame["amb_ti_ev"], magnetic_field=frame["b_profile"],
        alfven_speed=alfven_speed.to_value(u.m / u.s),
        reference_temperature=reference.electron_temperature.to_value(u.eV),
        min_upstream_gap=UPSTREAM_GAP_DI)
    return CodeProfiles(name=name, color=color, frame=frame, reference=reference,
                        layer=layer, alfven_speed=alfven_speed)


#: The poster figure's rows: (axis label, y scale, y limits).  Densities and temperatures
#: are ratios spanning decades; the velocity is linear because it passes through zero.
PROFILE_ROWS = (
    (r"$n / n_0$", "log", (3e-2, 3e2)),
    (r"$v_z\ /\ v_A$", "linear", None),
    (r"$T / T_0$", "log", (3e-1, 3e3)),
)

FLASH_COLOR = "#1f77b4"
WARPX_COLOR = "#d62728"


def _profile_rows(code: CodeProfiles) -> tuple[list, list, list]:
    """The three panels' curves for one code, as ``(values, style, label)`` triples.

    Line STYLE carries the species (or which temperature) and COLOUR carries the code, so
    a reader who has learnt "blue is FLASH" reads every panel the same way.
    """
    frame = code.frame
    reference = code.reference
    where_ambient = lambda values: np.where(
        np.asarray(frame["ambient_profile"], dtype=float) >= MIN_AMBIENT_FRACTION,
        np.asarray(values, dtype=float), np.nan)
    return (
        [(frame["ambient_profile"], "-", f"{code.name} ambient"),
         (frame["piston_profile"], "--", f"{code.name} piston")],
        [(where_ambient(frame["amb_vz"]) / code.alfven_speed.to_value(u.m / u.s),
          "-", f"{code.name}")],
        [(where_ambient(frame["amb_te_ev"])
          / reference.electron_temperature.to_value(u.eV), "-", f"{code.name} $T_e$"),
         (where_ambient(frame["amb_ti_ev"])
          / reference.ion_temperature.to_value(u.eV), "--", f"{code.name} $T_i$")],
    )


def _draw_scorecard(ax, rows: list[wx_metrics.CompareRow], subtitle: str) -> None:
    """The comparison table, drawn as text on a blank axis.

    Hand-placed rather than ``ax.table`` so the column positions, the font and the row
    striping are all set in one place at poster sizes -- the old figure's monospace block
    overflowed its panel the moment the fonts grew.
    """
    # Fixed data limits, and everything placed in DATA coordinates: axhspan's extent is
    # in data units whatever the text uses, so leaving the axis to autoscale slid the row
    # striping half a row off the numbers it was meant to be behind.
    ax.axis("off")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    columns = (0.0, 0.50, 0.70, 0.98)
    top, step = 0.94, 0.065

    ax.text(0.0, 1.0, "shocked-layer comparison", fontsize=21, fontweight="bold",
            va="bottom")
    # "ratio", not "WarpX/FLASH": the long heading runs back into the WarpX column at any
    # font size worth reading from a poster's distance.  The subtitle spells it out.
    for x, heading, align in zip(columns, ("quantity", "FLASH", "WarpX", "ratio"),
                                 ("left", "right", "right", "right")):
        ax.text(x, top, heading, fontsize=16, fontweight="bold", va="center", ha=align)

    for index, row in enumerate(rows):
        y = top - (index + 1) * step
        if index % 2 == 0:
            ax.axhspan(y - 0.5 * step, y + 0.5 * step, color="0.93", lw=0, zorder=0)
        cells = ((columns[0], row.quantity, "left", "0.1", "normal"),
                 (columns[1], f"{row.flash:.3g}", "right", FLASH_COLOR, "normal"),
                 (columns[2], f"{row.warpx:.3g}", "right", WARPX_COLOR, "normal"),
                 (columns[3], f"{row.ratio:.2f}", "right", "0.1", "bold"))
        for x, text, align, color, weight in cells:
            ax.text(x, y, text, fontsize=17, va="center", ha=align, color=color,
                    fontweight=weight)

    ax.text(0.0, top - (len(rows) + 1.5) * step, subtitle, fontsize=14, va="top",
            ha="left", color="0.25")


def plot_profile_comparison(flash: CodeProfiles, warpx: CodeProfiles,
                            rows: list[wx_metrics.CompareRow], out_path: str) -> None:
    """The poster figure: one matched time, three profiles, one scorecard.

    Always drawn at publication font sizes -- unlike every other figure here it exists
    only to be read from a distance, so it does not wait for ``--publication``.

    Both codes are on the same axis in ``d_i`` and both are normalized by their OWN
    upstream, which is the only comparison a reduced-mass deck admits: WarpX runs at
    ``m_i/(Z m_e) = 50`` against FLASH's real Al 6+, so no length, time or speed is
    comparable in absolute units and every quantity drawn here is a ratio.
    """
    with plt.rc_context(plot_style.publication_rc()):
        # Built from a gridspec rather than subplots(): the scorecard spans all three
        # rows of the right column, which a rectangular axes array cannot express.
        fig = plt.figure(figsize=(19.0, 12.0), layout="constrained")
        grid = fig.add_gridspec(len(PROFILE_ROWS), 2, width_ratios=(2.0, 1.0))
        profile_axes = [fig.add_subplot(grid[row, 0])
                        for row in range(len(PROFILE_ROWS))]
        for ax in profile_axes[:-1]:
            ax.sharex(profile_axes[-1])
            ax.tick_params(labelbottom=False)
        scorecard_ax = fig.add_subplot(grid[:, 1])

        panels = list(zip(profile_axes, PROFILE_ROWS,
                          zip(_profile_rows(flash), _profile_rows(warpx))))
        for ax, (label, scale, ylim), (flash_curves, warpx_curves) in panels:
            for code, curves in ((flash, flash_curves), (warpx, warpx_curves)):
                for values, style, name in curves:
                    ax.plot(code.frame["z_di"], values, style, color=code.color,
                            lw=3.0, label=name)
                shock = code.layer.regions.shock
                if np.isfinite(shock):
                    ax.axvline(shock, color=code.color, lw=1.8, ls=":", alpha=0.7)
                # Each code's own shocked-layer band, in its own colour: the table's
                # numbers are averages over these, so it can be read off the figure
                # rather than taken on trust.
                downstream = code.layer.regions.downstream
                if np.isfinite(downstream[0]):
                    ax.axvspan(*downstream, color=code.color, alpha=0.16, lw=0, zorder=0)
            if scale == "linear":
                ax.axhline(0.0, color="0.6", lw=1.2)
            ax.set_yscale(scale)
            if ylim is not None:
                ax.set_ylim(*ylim)
            ax.set_ylabel(label)
            ax.grid(alpha=0.25, which="both")
            ax.legend(loc="upper right", fontsize=15, ncol=2, framealpha=0.9)

        # Out to whichever line-out reaches furthest, so every number in the table is
        # visible on the plot -- FLASH's upstream band is at the far end of a ray that
        # runs well beyond the deck's periodic box, and cropping to the box would quote
        # an upstream the reader cannot see.
        profile_axes[-1].set_xlim(0.0, max(float(flash.frame["z_di"].max()),
                                           float(warpx.frame["z_di"].max())))
        profile_axes[-1].set_xlabel(
            r"distance from target along the shock normal [$d_i$]")

        _draw_scorecard(
            scorecard_ax, rows,
            "ratio = WarpX / FLASH.  Dotted vertical and shaded band:\n"
            "each code's own shock front and the shocked-ambient layer\n"
            "its column is averaged over.  The deck's periodic box ends\n"
            r"at 44 $d_i$; FLASH's ray runs on to its own far field." "\n\n"
            r"$n_0$, $T_0$ are each run's INITIAL ambient -- the state the" "\n"
            "deck was matched to.  FLASH's chamber has moved off it by\n"
            "the time the piston is running, which is what the\n"
            r"'n upstream' row and the raised FLASH $T$ curves show." "\n\n"
            r"WarpX runs at $m_i/Z m_e$ = 50 against FLASH's Al 6+, so no"
            "\nlength, time or speed compares in absolute units: every\n"
            "quantity here is a ratio within one code.")

        fig.suptitle(
            "Magnetized piston: FLASH (MHD) vs WarpX (full PIC) at "
            f"$t$ = {wci(flash.frame['t_gyro']):.2f} ${TIME_UNIT}$",
            fontsize=28)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)


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
        # a silent no-op that still reported "shifted by 0.000 w_ci^-1".
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
CAVEAT_FRONT = ("WarpX shifted by {:.3f} $\\omega_{{ci}}^{{-1}}$ so the fronts coincide at t = 0 "
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
                ax.set_title(f"FLASH {wci(flash['t_gyro']):.3f} / WarpX "
                             f"{wci(warpx['t_gyro']):.3f} ${TIME_UNIT}$", fontsize=9)
            # The velocity row's WarpX curve comes from the nearest phase dump, which is on
            # a coarser cadence than the field rows -- say so rather than implying the
            # whole column is one instant.
            if key == "velocity_profile" and np.isfinite(warpx.get("velocity_t_gyro",
                                                                   float("nan"))):
                ax.text(0.02, 0.94,
                        f"WarpX particles @ "
                        f"{wci(warpx['velocity_t_gyro']):.3f} ${TIME_UNIT}$",
                        transform=ax.transAxes, fontsize=7, va="top", color="#d62728")
    axes[0][0].legend(fontsize=8, loc="best")
    # ONE x label for the whole grid. Per-column labels collide under --publication:
    # every column repeats the same words at a width the column cannot hold.
    fig.supxlabel(r"distance from target [$d_i$]")

    caveat = (CAVEAT_FRONT.format(wci(offset)) if align == "front" else CAVEAT_CLOCK)
    fig.suptitle("FLASH vs WarpX heater piston — 1D line-outs in ion units, species kept "
        "separate (piston row = FLASH Si vs WarpX piston_ions)\n"
        f"both on axis: FLASH is a ray, WarpX averages |x| < {AXIS_BAND_DI:g} "
        r"$d_i$" "\n" + caveat, fontsize=12)
    # No tight_layout(): the figure is already layout="constrained", and calling both
    # makes matplotlib discard the constrained engine with a warning.
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_slices(flash_frames: list, warpx_frames: list, offset: float, align: str,
                out_path: str, vmin: float = 1e-2, vmax: float = 3e2,
                caveat_title: bool = True) -> None:
    """FLASH slices above WarpX slices, sharing a d_i axis so the box sizes compare.

    Both rows show the TARGET species alone -- FLASH n_e masked by its Si mass
    fraction, WarpX rho_piston_ions -- so the panels compare the same material.

    ``caveat_title`` off gives the one-line movie-frame heading: the three-line caveat
    block is taller than a single column's panels at publication font sizes, and it is
    text the reader dwells on once, not 60 times.
    """
    n_cols = len(flash_frames)
    # A floor on the width, because the title and colorbar do not shrink with the column
    # count: at one column the default 3.5" leaves the panels a narrow strip of the canvas.
    fig, axes = plt.subplots(2, n_cols, figsize=(max(3.5 * n_cols, 8.0), 6.6),
                             squeeze=False, layout="constrained")
    norm = LogNorm(vmin=vmin, vmax=vmax)

    # Crop BOTH codes to the WarpX box rather than padding WarpX out to FLASH's extent:
    # the point is to compare the same region of space, and the deck's transverse domain
    # (a few heating-spot radii, set by the periodic-image constraint) is the smaller one.
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
            f"FLASH  {wci(flash['t_gyro']):.3f} ${TIME_UNIT}$", fontsize=9)

        # No transpose: imshow wants [row, col] = [transverse, along-axis], and a WarpX 2D
        # covering_grid is already indexed [x, z]. Transposing drew the z=0 slab along the
        # vertical axis, which reads as a piston expanding transversely.
        axes[1][col].imshow(
            np.clip(warpx["density_map"], vmin, None), origin="lower",
            extent=(warpx["z_di"].min(), warpx["z_di"].max(),
                    warpx["x_di"].min(), warpx["x_di"].max()),
            aspect="auto", norm=norm, cmap="inferno")
        axes[1][col].set_title(
            f"WarpX  {wci(warpx['t_gyro']):.3f} ${TIME_UNIT}$", fontsize=9)

        for row in (0, 1):
            axes[row][col].set_xlim(0.0, z_max)
            axes[row][col].set_ylim(-x_max, x_max)
            if col > 0:
                axes[row][col].set_yticklabels([])
            else:
                axes[row][col].set_ylabel(r"transverse [$d_i$]")
        for frame, ax in ((flash, axes[0][col]), (warpx, axes[1][col])):
            if np.isfinite(frame["front_di"]):
                ax.axvline(frame["front_di"], color="cyan", lw=1.0, ls="--", alpha=0.8)

    # ONE x label for the whole grid -- see plot_lineouts.
    fig.supxlabel(r"distance from target [$d_i$]")
    fig.colorbar(image, ax=axes.ravel().tolist(),
                 label=r"$n_\mathrm{target} / n_\mathrm{amb}$",
                 fraction=min(0.06, 0.1 / n_cols), pad=0.01)
    if caveat_title:
        caveat = (CAVEAT_FRONT.format(wci(offset)) if align == "front" else CAVEAT_CLOCK)
        title = ("FLASH vs WarpX heater piston — target species only, 2D slices on "
                 "identical $d_i$ axes\n"
                 f"both CROPPED to the WarpX box; FLASH itself extends to "
                 f"{flash_z_max:.0f} $d_i$ along the axis and $\\pm${flash_x_max:.0f} "
                 f"$d_i$ transversely.  cyan dashed = measured front\n" + caveat)
    else:
        title = ("FLASH (top) vs WarpX (bottom) — target species, "
                 "cropped to the WarpX box")
    fig.suptitle(title, fontsize=10)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


#: Default two-species ramps for the overlay figure.  Deliberately hue-separated (warm
#: piston against cool ambient) and neither reaching white: two ramps that both saturate
#: pale become indistinguishable exactly in the shocked layer, which is the one region
#: the figure exists to show.  --piston-cmap / --ambient-cmap take any named colormap.
PISTON_RAMP = ("#2b0a3d", "#a11d3f", "#f0722b", "#ffd166")
AMBIENT_RAMP = ("#04263d", "#0d6a8f", "#2fb8c6", "#a8f0e0")

#: Colour behind both species, where neither has any density to draw.
EMPTY_COLOR = "#07070c"


@dataclass(frozen=True)
class OverlayStyle:
    """How :func:`plot_piston_comparison` renders one panel.

    ``piston_range`` / ``ambient_range`` are (vmin, vmax) on each species' own log
    colour scale, in units of the ambient electron density.  ``b_levels`` are contour
    levels in units of the upstream ``B0``; ``b_smooth`` is the Gaussian width, in map
    cells, applied before contouring.
    """

    piston_cmap: Colormap
    ambient_cmap: Colormap
    piston_range: tuple[float, float]
    ambient_range: tuple[float, float]
    b_mode: str
    b_levels: tuple[float, ...]
    b_smooth: float
    alpha_gamma: float


def overlay_style(args: argparse.Namespace) -> OverlayStyle:
    """The overlay CLI group as one value object, so the still and the movie agree."""
    return OverlayStyle(
        piston_cmap=species_colormap(args.piston_cmap, PISTON_RAMP),
        ambient_cmap=species_colormap(args.ambient_cmap, AMBIENT_RAMP),
        piston_range=tuple(args.piston_range),
        ambient_range=tuple(args.ambient_range),
        b_mode=args.b_overlay, b_levels=tuple(args.b_levels),
        b_smooth=args.b_smooth, alpha_gamma=args.alpha_gamma)


def species_colormap(name: str, ramp: tuple[str, ...]) -> Colormap:
    """A named matplotlib colormap, or the built-in hue-separated ramp for ``"default"``."""
    if name == "default":
        return LinearSegmentedColormap.from_list("species", list(ramp))
    return matplotlib.colormaps[name]


def _species_rgba(values: np.ndarray, norm: LogNorm, cmap: Colormap,
                  alpha_gamma: float) -> np.ndarray:
    """One species as an RGBA image whose ALPHA ramps with its own normalized density.

    Two species share one panel, so neither can be drawn opaque: the piston would hide
    the shocked ambient it is driving, which is the whole comparison.  Ramping alpha with
    the same normalized value that picks the colour makes each species fade out where it
    is absent, so the panel shows the sum without either colormap being read through the
    other's midtones.
    """
    scaled = np.clip(np.ma.filled(norm(np.asarray(values, dtype=float)), 0.0), 0.0, 1.0)
    rgba = cmap(scaled)
    rgba[..., 3] = scaled ** alpha_gamma
    return rgba


def _map_axes(extent: tuple[float, float, float, float],
              shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Coordinate vectors (transverse, along) for a stored ``[transverse, along]`` map.

    Rebuilt from the extent rather than carried alongside the map, because the WarpX maps
    are decimated for memory while their ``z_di`` is kept at full resolution -- so the
    stored axis does not index the stored map.
    """
    tr_lo, tr_hi, along_lo, along_hi = extent
    n_transverse, n_along = shape
    return (np.linspace(tr_lo, tr_hi, n_transverse),
            np.linspace(along_lo, along_hi, n_along))


def _draw_magnetic_overlay(ax, extent: tuple[float, float, float, float],
                           b_map: np.ndarray, style: OverlayStyle) -> None:
    """Draw |B| over one panel, in the only way this plane admits.

    THE FIELD IS PERPENDICULAR TO THIS PAGE.  Both runs are perpendicular-shock setups:
    FLASH applies 7 T along z and the slice is the x-y plane; the WarpX deck sets
    ``By = B0`` and its 2-D plane is x-z.  So the in-plane field is zero by construction
    -- FLASH's in-plane components are ~0.4% of ``magz``, and WarpX's are self-generated
    filamentation with no FLASH counterpart.  Streamlines of (B_transverse, B_along)
    would therefore trace noise, not field lines.

    What the plane CAN show is field-line DENSITY: lines through the page pierce it at an
    areal density exactly proportional to |B|.  ``contour`` draws iso-|B| curves, which
    bunch where the field piles up ahead of the piston and enclose the diamagnetic cavity;
    ``stipple`` draws that end-on picture literally, one dot per unit of flux.

    PIC noise is smoothed before contouring -- an un-smoothed WarpX |B| turns every
    contour into hachure.
    """
    if style.b_mode == "none":
        return
    from scipy.ndimage import gaussian_filter

    field = np.asarray(b_map, dtype=float)
    if style.b_smooth > 0:
        field = gaussian_filter(field, style.b_smooth)
    transverse, along = _map_axes(extent, field.shape)

    if style.b_mode == "contour":
        ax.contour(transverse, along, field.T, levels=list(style.b_levels),
                   colors="w", linewidths=0.8, alpha=0.8)
        return

    # Stipple: one dot per equal increment of flux, so the DOT DENSITY is |B| itself.
    # Deterministic seed -- the figure has to be reproducible across re-plots.
    cell_area = (abs(transverse[1] - transverse[0]) * abs(along[1] - along[0]))
    weight = np.clip(field, 0.0, None) * cell_area
    total = weight.sum()
    if total <= 0.0:
        return
    n_dots = 2000
    rng = np.random.default_rng(0)
    picked = rng.choice(weight.size, size=n_dots, p=weight.ravel() / total)
    rows, cols = np.unravel_index(picked, weight.shape)
    ax.plot(transverse[rows], along[cols], ls="none", marker=".", ms=1.4,
            color="w", alpha=0.55)


def _piston_panels(flash: dict, warpx: dict, along: float) -> tuple:
    """The ``(frame, extent, title)`` triple for each code at one matched instant.

    ``extent`` is ``(transverse_lo, transverse_hi, along_lo, along_hi)`` -- the along-axis
    runs up the page here, which is the rotation that separates this figure from
    :func:`plot_slices`.
    """
    los_lo, los_hi, tr_lo, tr_hi = flash["extent_di"]
    return (
        (flash, (tr_lo, tr_hi, los_lo, los_hi),
         f"FLASH  {wci(flash['t_gyro']):.2f} ${TIME_UNIT}$"),
        (warpx, (float(warpx["x_di"].min()), float(warpx["x_di"].max()), 0.0, along),
         f"WarpX  {wci(warpx['t_gyro']):.2f} ${TIME_UNIT}$"),
    )


def _draw_piston_panel(ax, frame: dict, extent: tuple[float, float, float, float],
                       title: str, style: OverlayStyle, piston_norm: LogNorm,
                       ambient_norm: LogNorm, transverse: float, along: float) -> None:
    """One code at one instant: ambient under piston under |B|, at true aspect ratio."""
    ax.set_facecolor(EMPTY_COLOR)
    # .T on every map: the stored arrays are [transverse, along] and the vertical
    # axis is the along-axis. Ambient first, piston over it -- the piston is what the
    # eye should land on, and it fades to transparent where it is absent.
    for key, norm, cmap in (("ambient_map", ambient_norm, style.ambient_cmap),
                            ("density_map", piston_norm, style.piston_cmap)):
        ax.imshow(_species_rgba(np.asarray(frame[key]).T, norm, cmap, style.alpha_gamma),
                  origin="lower", extent=extent, aspect="equal",
                  interpolation="nearest")
    _draw_magnetic_overlay(ax, extent, frame["b_map"], style)
    ax.set_title(title, fontsize=10)
    ax.set_xlim(-transverse, transverse)
    ax.set_ylim(0.0, along)


def _piston_colorbars(fig, axes: list, style: OverlayStyle, piston_norm: LogNorm,
                      ambient_norm: LogNorm, fraction: float) -> None:
    """One colorbar per species -- each carries its own scale, so neither can be shared."""
    for norm, cmap, label in (
            (piston_norm, style.piston_cmap, r"$n_\mathrm{piston} / n_\mathrm{amb}$"),
            (ambient_norm, style.ambient_cmap, r"$n_\mathrm{ambient} / n_\mathrm{amb}$")):
        fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axes, label=label,
                     fraction=fraction, pad=0.01)


def _field_note(style: OverlayStyle) -> str:
    """What the white overlay means, read back from the style so it cannot disagree."""
    return {
        "contour": (r"$\theta_{Bn} = 90^\circ$, white = iso $|B|/B_0$ at "
                    + ", ".join(f"{v:g}" for v in style.b_levels)),
        "stipple": (r"$\theta_{Bn} = 90^\circ$, white dots = field lines end-on, "
                    "one per unit flux"),
        "none": r"$\theta_{Bn} = 90^\circ$",
    }[style.b_mode]


def plot_piston_comparison(flash_frames: list, warpx_frames: list, out_path: str,
                           style: OverlayStyle) -> None:
    """FLASH above WarpX, both expanding BOTTOM TO TOP, at true aspect ratio.

    Rotated relative to :func:`plot_slices` so the shock normal runs up the page, which
    is how the experiment is drawn.  Every panel is ``aspect="equal"`` on two axes that
    are both in ``d_i``, so a feature that is round in the simulation is round here --
    the un-rotated figure stretches each panel to fill its cell and a plume 44 d_i long
    and 70 d_i wide comes out looking flat.

    BOTH SPECIES, ONE PANEL.  The piston and the ambient it is driving into carry their
    own colormaps and their own colour scales, layered by :func:`_species_rgba`, so the
    shocked ambient ahead of the contact is visible in the same panel as the piston
    behind it.  |B| goes over the top -- see :func:`_draw_magnetic_overlay` for why it is
    contours rather than field lines.
    """
    n_cols = len(flash_frames)
    transverse = float(np.abs(warpx_frames[0]["x_di"]).max())
    along = float(warpx_frames[0]["z_di"].max())

    piston_norm = LogNorm(*style.piston_range)
    ambient_norm = LogNorm(*style.ambient_range)

    # Panel aspect is fixed by the domain, so size the canvas from it rather than letting
    # constrained_layout shrink equal-aspect axes into a strip of whitespace. The panel
    # spans the FULL transverse width, 2 x the half-width the domain is quoted by.
    panel_width = 3.4
    fig, axes = plt.subplots(
        2, n_cols, squeeze=False, layout="constrained",
        figsize=(panel_width * n_cols,
                 2 * panel_width * along / (2 * transverse) + 2.2))

    for col, (flash, warpx) in enumerate(zip(flash_frames, warpx_frames)):
        for row, (frame, extent, title) in enumerate(
                _piston_panels(flash, warpx, along)):
            ax = axes[row][col]
            _draw_piston_panel(ax, frame, extent, title, style, piston_norm,
                               ambient_norm, transverse, along)
            if col != 0:
                ax.set_yticklabels([])

    # One figure-level label, not one per row: the two rows share the axis, and at
    # --pub sizes two copies of the same string collide with each other and with the
    # field note down the margin.
    fig.supylabel(r"distance from target [$d_i$]")
    fig.supxlabel(r"transverse [$d_i$]")
    _piston_colorbars(fig, axes.ravel().tolist(), style, piston_norm, ambient_norm,
                      min(0.05, 0.09 / n_cols))

    # Down the left margin, mirroring the colorbar labels on the right rather than
    # spending title lines on it -- this figure goes on a poster.
    field_note = _field_note(style)
    # Negative x puts it OUTSIDE the axes, clear of the y labels; bbox_inches="tight"
    # then grows the saved canvas to include it (constrained_layout will not reserve
    # space for a figure-level text, so placing it at x=0 lands it on the labels).
    # The offset scales with the font size because --pub roughly doubles it, and a
    # margin tuned at the default size lands the note back on top of supylabel.
    note_size = 0.75 * plt.rcParams["axes.labelsize"] if isinstance(
        plt.rcParams["axes.labelsize"], (int, float)) else 9
    fig.text(-0.03 * max(1.0, note_size / 9.0), 0.5, field_note,
             rotation=90, va="center", ha="left", fontsize=note_size)

    fig.suptitle("FLASH (top) vs WarpX (bottom) piston formation and expansion",
                 fontsize=12)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


#: One row of the shock figure: frame key, axis label, scale, and the per-frame factor
#: that turns the stored quantity into the plotted one.
SHOCK_ROWS = (
    ("ambient_profile", r"$n_\mathrm{ambient} / n_\mathrm{amb}$", "log", 1.0),
    ("b_profile", r"$|B| / B_0$", "linear", 1.0),
    ("amb_vz", r"$v_{z,\mathrm{ambient}}$ [km/s]", "linear", 1e-3),
    ("amb_te_ev", r"$T_e$ [eV]", "log", 1.0),
    ("amb_ti_ev", r"$T_i$ [eV]", "log", 1.0),
    ("amb_pressure", r"$P_\mathrm{thermal}$ [eV m$^{-3}$]", "log", 1.0),
)


def _row_limits(frames: list, key: str, scale: str, factor: float) -> tuple | None:
    """Common y-limits for one row, from the data the row actually draws.

    Shared across the columns so a jump is read by eye rather than by re-reading three
    different axes, and taken from percentiles rather than min/max so the residual PIC
    spikes at the edge of a species' support do not set the range.
    """
    values = np.concatenate([np.asarray(f[key], dtype=float).ravel() for f in frames])
    values = values[np.isfinite(values)] * factor
    if scale == "log":
        values = values[values > 0.0]
    if values.size == 0:
        return None
    lo, hi = np.percentile(values, [0.5, 99.9])
    if scale == "log":
        return max(lo, hi * 1e-4), hi * 1.6
    pad = 0.08 * (hi - lo) or 1.0
    return lo - pad, hi + pad


def plot_shock_profile(frames: list, states: list, out_path: str) -> None:
    """Line-outs along the shock normal with the perpendicular-MHD jump overlaid.

    The WarpX counterpart of ``flash_rh_prediction.png``, and read the same way: shaded
    bands are the regions averaged, the solid grey line is the measured upstream, the
    dashed black line is what the jump predicts downstream from that upstream, and the
    dotted black line is what the run actually has there.  Prediction against dotted line
    is the MHD-vs-kinetic comparison; everything else is context.

    The density row carries BOTH species, because the question the figure exists to
    answer is whether ambient piles up *ahead* of the piston: the ambient curve rising
    while the piston curve has already fallen away is the shock separating from the
    contact.
    """
    n_cols = len(frames)
    fig, axes = plt.subplots(len(SHOCK_ROWS), n_cols, squeeze=False, sharex=True,
                             figsize=(4.2 * n_cols, 2.3 * len(SHOCK_ROWS)),
                             layout="constrained")

    limits = [_row_limits(frames, key, scale, factor)
              for key, _, scale, factor in SHOCK_ROWS]

    for col, (frame, state) in enumerate(zip(frames, states)):
        regions, jump = state["regions"], state["jump"]
        for row, (key, label, scale, factor) in enumerate(SHOCK_ROWS):
            ax = axes[row][col]
            ax.plot(frame["z_di"], np.asarray(frame[key], dtype=float) * factor,
                    color="#1f77b4", lw=1.4, label="WarpX ambient")
            if key == "ambient_profile":
                ax.plot(frame["z_di"], frame["piston_profile"], color="#d62728",
                        lw=1.2, ls="--", label="WarpX piston")

            ax.axvspan(*regions.downstream, color="#aecde8", alpha=0.45, lw=0,
                       label="downstream band")
            ax.axvspan(*regions.upstream, color="0.85", alpha=0.7, lw=0,
                       label="upstream band")
            ax.axvline(regions.contact, color="#d62728", lw=1.0, ls="-.",
                       label="piston contact")
            ax.axvline(regions.shock, color="0.2", lw=1.0, label="shock front")
            for value, style, name in (
                    (state["upstream"][key], dict(color="0.35", lw=1.2),
                     "upstream mean"),
                    (state["predicted"][key], dict(color="k", lw=1.5, ls="--"),
                     "RH predicted dn"),
                    (state["downstream"][key], dict(color="k", lw=1.2, ls=":"),
                     "measured dn")):
                if np.isfinite(value):
                    ax.axhline(value * factor, label=name, **style)

            ax.set_yscale(scale)
            if limits[row] is not None:
                ax.set_ylim(*limits[row])
            ax.grid(alpha=0.25, which="both")
            if col == 0:
                ax.set_ylabel(label)
            if row == 0:
                warning = "" if state["upstream_is_pristine"] else "\nUPSTREAM NOT PRISTINE"
                ax.set_title(f"{wci(frame['t_gyro']):.2f} ${TIME_UNIT}$    "
                             f"$M_A$ = {jump['mach_a']:.1f}   "
                             f"$M_s$ = {jump['mach_s']:.1f}   "
                             f"$M_{{ms}}$ = {jump['mach_ms']:.1f}   "
                             f"$r$ = {jump['r']:.2f}{warning}", fontsize=9,
                             color="k" if state["upstream_is_pristine"] else "#b22222")
    # Above the grid, not inside a panel: at eight entries it covers the piston curve in
    # whichever corner it lands, and the bottom slot belongs to the shared x label.
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=4, fontsize=8.5)

    fig.supxlabel(r"distance from target along the shock normal [$d_i$]")
    v_shock = states[0]["v_shock"].to_value(u.km / u.s)
    fig.suptitle(
        "WarpX heater run — the shock driven ahead of the piston, against the "
        r"perpendicular ($\theta_{Bn} = 90^\circ$) MHD jump" "\n"
        f"upstream MEASURED in the shaded band ahead of each front; "
        f"$v_\\mathrm{{shock}}$ = {v_shock:.0f} km/s fitted over the whole series "
        f"(both in the deck's reduced-mass units)", fontsize=11)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


#: Streak rows: frame key, title, colour map, log scale, and per-frame factor.
STREAK_PANELS = (
    ("piston_profile", r"$n_\mathrm{piston} / n_\mathrm{amb}$", "magma", True, 1.0),
    # Linear, unlike the piston: the ambient spans 0 to ~4, and the quantity being
    # looked for is the COMPRESSION RATIO. A log scale spends its range on the swept
    # cavity and renders the whole undisturbed upstream as one flat colour.
    ("ambient_profile", r"$n_\mathrm{ambient} / n_\mathrm{amb}$", "viridis", False, 1.0),
    ("b_profile", r"$|B| / B_0$", "inferno", False, 1.0),
    ("amb_ti_ev", r"$T_{i,\mathrm{ambient}}$ [eV]", "inferno", True, 1.0),
)


def plot_streaks(frames: list, trajectory: pp.FrontTrajectory | None,
                 out_path: str) -> None:
    """Position-time maps of the on-axis line-outs -- the WarpX ``flash_overview`` streaks.

    Every dump contributes one column, so the shock shows up as the outer ridge and the
    piston contact as the inner one; the fitted trajectory is drawn over the top so a
    front that is accelerating is visible as curvature away from the straight line.
    """
    z_di = frames[0]["z_di"]
    if any(f["z_di"].shape != z_di.shape for f in frames):
        raise RuntimeError("the WarpX grid changed between dumps; a streak needs one axis")
    # `times` stays in T_ci because trajectory.at() was fitted in it; `axis_times` is the
    # same instants in the unit the figure is labelled with.
    times = np.array([f["t_gyro"] for f in frames])
    axis_times = wci(times)

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.0), layout="constrained",
                             sharex=True, sharey=True)
    extent = (axis_times.min(), axis_times.max(), z_di.min(), z_di.max())

    for ax, (key, title, cmap, log, factor) in zip(axes.ravel(), STREAK_PANELS):
        streak = np.column_stack([np.asarray(f[key], dtype=float) * factor
                                  for f in frames])
        finite = streak[np.isfinite(streak)]
        norm = None
        if log and (finite > 0).any():
            # Floor at 1e-3 of the peak: both densities go to exactly zero in the swept
            # cavity, and a log norm reaching for the true minimum spends every colour on
            # emptiness.
            peak = finite.max()
            norm = LogNorm(vmin=max(peak * 1e-3, finite[finite > 0].min()), vmax=peak)
        elif finite.size:
            norm = Normalize(vmin=max(finite.min(), 0.0),
                             vmax=float(np.percentile(finite, 99.5)))
        image = ax.imshow(streak, origin="lower", extent=extent, aspect="auto",
                          cmap=cmap, norm=norm)
        ax.plot(axis_times, [f["shock_di"] for f in frames], color="w", lw=1.0,
                ls="none", marker=".", ms=2.5, label="measured shock")
        ax.plot(axis_times, [f["contact_di"] for f in frames], color="c", lw=1.0,
                ls="none", marker=".", ms=2.5, label="piston contact")
        if trajectory is not None:
            ax.plot(axis_times, trajectory.at(times), color="w", lw=1.4,
                    label=f"fit "
                          f"{trajectory.speed / INVERSE_OMEGA_PER_GYROPERIOD:.2f} "
                          r"$d_i\,\omega_{ci}$")
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.15)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
    axes[0][0].legend(fontsize=8, loc="upper left")

    fig.supxlabel(rf"$t$ [${TIME_UNIT}$]")
    fig.supylabel(r"distance from target along the shock normal [$d_i$]")
    fig.suptitle(f"WarpX heater run — on-axis streaks over {len(frames)} dumps "
                 f"(band $|x| < {AXIS_BAND_DI:g}\\,d_i$)", fontsize=12)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


#: Width of ONE panel of a piston movie frame, in inches; its height follows from the
#: domain, because the panels are equal-aspect.  The margins are what the canvas needs
#: AROUND the two panels -- two colorbars and the y labels across, two title lines and
#: the x label down.  They have to be right: constrained_layout gives the panels
#: whatever is left, so a margin that is too generous shows up as a band of empty
#: canvas rather than as a bigger picture.
MOVIE_PANEL_WIDTH = 3.4
MOVIE_MARGIN_WIDTH = 2.4
MOVIE_MARGIN_HEIGHT = 1.5

#: Fraction of the canvas width held back for the rotated y label -- see the comment in
#: :func:`plot_piston_frame` for why it has to be reserved by hand.
MOVIE_LABEL_STRIP = 0.03


def piston_frame_figsize(transverse: float, along: float) -> tuple[float, float]:
    """Canvas for :func:`plot_piston_frame` from the domain half-width and length, in d_i.

    Computed once by the caller and reused for every frame: the panels are
    ``aspect="equal"``, so letting each frame size its own canvas would make the movie
    breathe as the titles change width.
    """
    return (2 * MOVIE_PANEL_WIDTH + MOVIE_MARGIN_WIDTH,
            MOVIE_PANEL_WIDTH * along / (2 * transverse) + MOVIE_MARGIN_HEIGHT)


def plot_piston_frame(flash: dict, warpx: dict, out_path: str, style: OverlayStyle,
                      figsize: tuple[float, float]) -> None:
    """One movie frame of the piston comparison: FLASH beside WarpX at one instant.

    Side by side rather than stacked as on the poster, because a movie is watched on a
    landscape screen; everything else -- the two species' colormaps and log ranges, the
    |B| overlay, equal aspect on two ``d_i`` axes -- is
    :func:`plot_piston_comparison`'s, so a frame grabbed out of the movie reads as one
    column of that figure.

    Saved WITHOUT ``bbox_inches="tight"``: every frame has to come out the same pixel
    size or ffmpeg refuses the sequence, and a tight box shrinks to whatever that frame's
    own labels need.
    """
    transverse = float(np.abs(warpx["x_di"]).max())
    along = float(warpx["z_di"].max())
    piston_norm = LogNorm(*style.piston_range)
    ambient_norm = LogNorm(*style.ambient_range)

    fig, axes = plt.subplots(1, 2, squeeze=False, layout="constrained", figsize=figsize)
    for ax, (frame, extent, title) in zip(axes[0], _piston_panels(flash, warpx, along)):
        _draw_piston_panel(ax, frame, extent, title, style, piston_norm, ambient_norm,
                           transverse, along)
    axes[0][1].set_yticklabels([])

    # Reserve the left strip EXPLICITLY. With equal-aspect axes carrying colorbars
    # attached to the pair, constrained_layout (matplotlib 3.11) sets no room aside for
    # the y label at all: it lands on the tick numbers, and once savefig re-runs the
    # layout at its own dpi it falls off the canvas entirely. The poster figure escapes
    # this only because bbox_inches="tight" grows the canvas afterwards, which a movie
    # frame may not do.
    fig.get_layout_engine().set(
        rect=(MOVIE_LABEL_STRIP, 0.0, 1.0 - MOVIE_LABEL_STRIP, 1.0))
    fig.supylabel(r"distance from target [$d_i$]")
    fig.supxlabel(r"transverse [$d_i$]")
    _piston_colorbars(fig, axes.ravel().tolist(), style, piston_norm, ambient_norm, 0.045)
    # The field note goes under the title, not down the left margin as on the poster: a
    # rotated figure-level text only fits there because bbox_inches="tight" grows the
    # canvas around it, and a movie frame's canvas may not grow.
    fig.suptitle("FLASH vs WarpX piston formation and expansion\n" + _field_note(style),
                 fontsize=11)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


@dataclass
class MovieWork:
    """Everything one movie frame needs: read its pair, draw it, name the file.

    ``pair(index)`` returns the ``(flash, warpx)`` dicts for that frame, read on demand
    -- a movie over every dump would otherwise hold ~70 full-box WarpX maps, ~7 MB each,
    at once.
    """

    n_frames: int
    pair: Callable[[int], tuple[dict, dict]]
    draw: Callable[[dict, dict, str], None]
    frame_dir: str

    def render(self, index: int) -> None:
        flash, warpx = self.pair(index)
        self.draw(flash, warpx, os.path.join(self.frame_dir, f"f{index:04d}.png"))


#: Parked here by :func:`render_movie` before it forks, because a pool worker must be a
#: top-level function whose arguments PICKLE -- and neither the frame readers nor the
#: draw callback do.  Under the fork start method the children inherit this module's
#: globals, so each is handed only a frame index.
_MOVIE_WORK: MovieWork | None = None


def _render_movie_frame(index: int) -> int:
    assert _MOVIE_WORK is not None
    _MOVIE_WORK.render(index)
    return index


def render_movie(work: MovieWork, out_path: str, fps: int, jobs: int = 1) -> None:
    """Draw every frame, then stitch them with ffmpeg.

    ``jobs`` > 1 forks a pool over the frame INDICES, which is what turns a
    seventy-frame movie from half an hour into a few minutes on a compute node.  Each
    worker re-reads whatever dumps its own frames need, so the pool trades repeated IO
    for wall clock -- worth it because a FLASH slice costs ~40 s and there are more
    cores than frames.
    """
    global _MOVIE_WORK

    os.makedirs(work.frame_dir, exist_ok=True)
    _MOVIE_WORK = work
    indices = range(work.n_frames)

    if jobs > 1:
        import multiprocessing

        with multiprocessing.get_context("fork").Pool(jobs) as pool:
            for done, index in enumerate(
                    pool.imap_unordered(_render_movie_frame, indices), start=1):
                print(f"         frame {done}/{work.n_frames}", flush=True)
    else:
        for index in indices:
            work.render(index)
            print(f"         frame {index + 1}/{work.n_frames}", flush=True)

    command = ["ffmpeg", "-y", "-framerate", str(fps),
               "-i", os.path.join(work.frame_dir, "f%04d.png"),
               "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-pix_fmt", "yuv420p", out_path]
    try:
        subprocess.run(command, check=True, capture_output=True)
        print(f"Saved -> {out_path}")
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"NOTE: movie not rendered ({exc}); frames kept in {work.frame_dir}")


def main() -> None:
    args = parse_args()
    plot_style.apply(args.publication)

    run_name = os.path.basename(args.config).replace(".warpx.yaml", "")
    cache = FrameCache(
        None if args.cache is None
        else (args.cache or os.path.join(
            args.output_dir or os.path.join(_REPO, "results", "warpx", run_name),
            "frames.pkl")))

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

    # Cover the same region of space on both sides by default: the FLASH slice is cut to
    # the WarpX transverse box rather than to a fixed 4000 um, so the comparison panels
    # have no dead band where one code simply was not sampled.
    halfwidth_um = args.slice_halfwidth_um
    if halfwidth_um is None:
        halfwidth_um = float(scales.transverse_halfwidth_di
                             * targets.upstream.ion_skin_depth.to_value(u.um))

    # A partially-finished run still produces a figure over the time it did cover.
    warpx_paths = warpx_plotfiles(args.config, args.diag_dir)
    print(f"WarpX  : {len(warpx_paths)} plotfiles")
    warpx_frames_all = read_warpx_frames(warpx_paths, scales, cache)
    # Persist as soon as the expensive reads are done, not at the end: saving only after
    # the FLASH side meant one failure there threw away every WarpX frame already measured.
    cache.save()
    warpx_gyro = np.array([f["t_gyro"] for f in warpx_frames_all])

    trajectory = fit_shock_trajectory(warpx_frames_all, scales)
    if trajectory is None:
        print("         WARNING: no dump has a resolvable ambient pile-up, so there is "
              "no shock to fit; the RH figure will be skipped.")
    else:
        print(f"         shock front "
              f"{trajectory.speed / INVERSE_OMEGA_PER_GYROPERIOD:.3f} d_i*w_ci = "
              f"{shock_speed(trajectory, scales).to_value(u.km / u.s):.0f} km/s "
              f"(rms {trajectory.residual_rms:.2f} d_i over "
              f"{trajectory.n_points} dumps)")

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

    def read_flash(path: str) -> dict:
        return cache.get(f"flash:v{FLASH_SCHEMA}:{halfwidth_um:.0f}:{path}", lambda: flash_frame(
            path, source, targets, piston_material=piston_material,
            ambient_material=ambient_material, halfwidth_um=halfwidth_um,
            t_start_s=t_lo_ns * 1e-9))

    t_lo_ns, t_hi_ns = (float(v) for v in spec["flash"]["window_ns"])
    all_flash = fu.find_plot_files(source.flash_dir)
    flash_times = np.array([fu.flash_time_s(p) for p in all_flash])
    in_window = np.flatnonzero((flash_times >= t_lo_ns * 1e-9)
                               & (flash_times <= t_hi_ns * 1e-9))
    flash_paths = [all_flash[i] for i in in_window]
    flash_gyro = ((flash_times[in_window] - t_lo_ns * 1e-9)
                  / targets.upstream.gyroperiod.to_value(u.s))
    print(f"FLASH  : {len(flash_paths)} dumps in {t_lo_ns}-{t_hi_ns} ns "
          f"(0-{wci(flash_gyro.max()):.3f} w_ci^-1)")
    print(f"         WarpX covers 0-{wci(warpx_gyro.max()):.3f} w_ci^-1 of the "
          f"{wci(scales.run_gyroperiods):.3f} w_ci^-1 target")

    warpx_fronts = np.array([f["front_di"] for f in warpx_frames_all])

    # --align front needs FLASH's starting front position, which is only known after a
    # FLASH frame is loaded -- so load the first one up front rather than passing None and
    # having the alignment silently do nothing.
    flash_fronts = None
    if args.align == "front":
        first = read_flash(flash_paths[0])
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
            print(f"         front alignment shifts WarpX by "
                  f"{wci(offset):.3f} w_ci^-1")

    out_dir = yaml_edit.out_dir(
        os.path.basename(args.config).replace(".warpx.yaml", ""),
        args.output_dir or os.path.join(
            _REPO, "results", "warpx",
            os.path.basename(args.config).replace(".warpx.yaml", "")),
        cfg=spec, config_path=args.config)

    def save(path: str) -> str:
        print(f"Saved -> {path}")
        return path

    if "evolution" in args.figures:
        flash_frames = [read_flash(flash_paths[i]) for i in flash_idx]
        warpx_frames = read_warpx_maps(warpx_paths, warpx_idx, warpx_frames_all,
                                       scales, cache)
        attach_warpx_velocity(warpx_frames, velocity_frames)
        cache.save()
        plot_lineouts(flash_frames, warpx_frames, offset, args.align,
                      save(os.path.join(out_dir, "evolution_lineouts.png")))
        plot_slices(flash_frames, warpx_frames, offset, args.align,
                    save(os.path.join(out_dir, "evolution_slices.png")))

    if "compare" in args.figures:
        # Its own, coarser set of matched times: three panels at true aspect ratio are
        # the whole point, and five would each be a sliver.
        compare_flash_idx, compare_warpx_idx, _ = pick_matched_times(
            flash_gyro, warpx_gyro, args.compare_times, args.align,
            flash_fronts=flash_fronts, warpx_fronts=warpx_fronts)
        plot_piston_comparison(
            [read_flash(flash_paths[i]) for i in compare_flash_idx],
            read_warpx_maps(warpx_paths, compare_warpx_idx, warpx_frames_all,
                            scales, cache),
            save(os.path.join(out_dir, "piston_comparison.png")),
            overlay_style(args))
        cache.save()

    if "shock" in args.figures and trajectory is not None:
        # The same instants as the piston comparison, so the two figures are read
        # together -- but WarpX only, since the shock is measured in the deck's units.
        probes = np.linspace(0.0, warpx_gyro.max(), args.compare_times + 1)[1:]
        shock_idx = [int(np.argmin(np.abs(warpx_gyro - t))) for t in probes]
        shock_frames = [with_ambient_pressure(warpx_frames_all[i], scales)
                        for i in shock_idx]
        states = [warpx_shock_state(f, scales, shock_speed(trajectory, scales))
                  for f in shock_frames]
        for frame, state in zip(shock_frames, states):
            jump = state["jump"]
            print(f"         t = {wci(frame['t_gyro']):.3f} w_ci^-1: shock at "
                  f"{state['regions'].shock:.1f} d_i, contact at "
                  f"{state['regions'].contact:.1f} d_i, M_A = {jump['mach_a']:.1f}, "
                  f"M_ms = {jump['mach_ms']:.1f}, r_RH = {jump['r']:.2f}, r_measured = "
                  f"{state['downstream']['ambient_profile'] / state['upstream']['ambient_profile']:.2f}")
        plot_shock_profile(shock_frames, states,
                           save(os.path.join(out_dir, "shock_rh_prediction.png")))

    if "streaks" in args.figures:
        plot_streaks(warpx_frames_all, trajectory,
                     save(os.path.join(out_dir, "shock_streaks.png")))

    if "profiles" in args.figures:
        # One column of the SAME matched pairs the other figures use, so the poster
        # figure cannot end up showing an instant no other figure agrees with.
        if args.profile_time is not None:
            probe = args.profile_time / INVERSE_OMEGA_PER_GYROPERIOD
            pair = int(np.argmin(np.abs(warpx_gyro[warpx_idx] - probe)))
        else:
            pair = last_usable_pair(warpx_frames_all, warpx_idx, scales.upstream)
            print(f"         profile figure at pair {pair} "
                  f"({wci(warpx_gyro[warpx_idx[pair]]):.3f} w_ci^-1): the last matched "
                  f"time whose WarpX upstream is still undisturbed")
        flash_side = code_profiles("FLASH", FLASH_COLOR,
                                   read_flash(flash_paths[flash_idx[pair]]),
                                   targets.upstream)
        warpx_side = code_profiles("WarpX", WARPX_COLOR,
                                   warpx_frames_all[warpx_idx[pair]], scales.upstream)
        cache.save()

        rows = wx_metrics.compare_layers(flash_side.layer, warpx_side.layer)
        header = (f"FLASH {wci(flash_side.frame['t_gyro']):.3f} / WarpX "
                  f"{wci(warpx_side.frame['t_gyro']):.3f} w_ci^-1     "
                  f"v_A: FLASH {flash_side.alfven_speed.to_value(u.km / u.s):.0f} / "
                  f"WarpX {warpx_side.alfven_speed.to_value(u.km / u.s):.0f} km/s "
                  f"(each measured upstream, NOT comparable across codes)")
        table = "\n".join((header, "", wx_metrics.compare_text(rows)))
        print()
        print(table)
        with open(save(os.path.join(out_dir, "flash_vs_warpx_profiles.txt")), "w") as fh:
            fh.write(table + "\n")
        plot_profile_comparison(
            flash_side, warpx_side, rows,
            save(os.path.join(out_dir, "flash_vs_warpx_profiles.png")))

    if args.movie:
        # EVERY WarpX dump gets a frame -- the movie is the one output with no reason to
        # subsample -- each paired with the FLASH dump nearest it on the aligned clock.
        # FLASH has half as many dumps over the window and stops 0.04 w_ci^-1 short of
        # WarpX, so its panel holds while WarpX advances; both panels carry their own
        # time, so the figure says which frames those are.
        movie_flash_idx = [int(np.argmin(np.abs(flash_gyro - (t - offset))))
                           for t in warpx_gyro]
        print(f"movie  : {len(warpx_paths)} frames over "
              f"{len(set(movie_flash_idx))} distinct FLASH dumps")

        def movie_pair(index: int) -> tuple[dict, dict]:
            """The (FLASH, WarpX) pair for one frame, its maps read on demand.

            The maps deliberately bypass ``cache``: 70 of them is ~500 MB, which does not
            belong in a ``frames.pkl`` whose other entries are line-outs.
            """
            return (read_flash(flash_paths[movie_flash_idx[index]]),
                    {**warpx_frames_all[index],
                     **warpx_maps(warpx_paths[index], scales)})

        if args.movie == "compare":
            style = overlay_style(args)
            figsize = piston_frame_figsize(
                float(scales.transverse_halfwidth_di),
                float(warpx_frames_all[0]["z_di"].max()))
            draw = lambda flash, warpx, path: plot_piston_frame(
                flash, warpx, path, style, figsize)
        else:
            draw = lambda flash, warpx, path: plot_slices(
                [flash], [warpx], 0.0, "clock", path, caveat_title=False)

        movie_path = os.path.join(out_dir, MOVIE_FILES[args.movie])
        render_movie(MovieWork(n_frames=len(warpx_paths), pair=movie_pair, draw=draw,
                               frame_dir=movie_path + "_frames"),
                     movie_path, args.fps, args.jobs)
        cache.save()


if __name__ == "__main__":
    main()
