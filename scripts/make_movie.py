"""scripts/make_movie.py — quick MP4 movies of OSIRIS diagnostics.

A fast "did the run break?" visualiser: point it at an OSIRIS ``MS`` tree (or a
single diagnostic directory), render every ``*.h5`` dump to a PNG in parallel,
and stitch them into an MP4 with ffmpeg.  This is plotting/IO only — the only
physics (the ion normalisation constants) comes from tested ``src/`` functions.

Two modes, one code path:

  - **Interactive** (default): lists the diagnostics under ``MS`` and prompts you
    to pick one; per movie you may override the colour/axis limits.  This is the
    workflow to keep around for browsing a run.
  - **Non-interactive** (``--no-interactive``, or ``-d`` pointing straight at a
    directory of ``*.h5`` files): renders from argv with no prompts.  Every
    interactive knob has a matching flag, so this is what an sbatch wrapper calls
    (see ``scripts/make_movie.sbatch``) to batch many diagnostics.

Units (``--units electron|ion``) set the spatial + time normalisation, read
automatically from the run directory (no hand-entered gyrotime):

  - ``electron`` (default): native OSIRIS units — lengths in c/wpe, the frame
    title shows t [1/wpe].
  - ``ion``: spatial (length) axes are rescaled to the ion inertial length
    d_i = sqrt(|rqm_i|); the title shows t / T_ci (upstream ion gyroperiod,
    T_ci = 2*pi*|rqm_i|/|B'| from the first field dump).

Crop bounds are given by **physical axis value** in the active length unit, not
array index:

    --xlim LO HI   horizontal spatial axis (1D: the spatial axis; 2D: axes[1])
    --ylim LO HI   2D vertical spatial axis (axes[0])
    --log          log scale: LogNorm colour (2D) or log y-axis (1D)
    --vmin/--vmax  data scale: colourbar range (2D) or line y-range (1D)

Only **length** axes are normalised: for phase spaces (e.g. p1x1) the momentum
axis is left in m_e c, detected via pyVisOS ``OSUnits.is_length()``.

Environment: ``analysis`` (needs osh5io/osh5vis + ffmpeg on PATH).

Usage
-----
    conda activate analysis

    # interactive: browse a whole run in ion units
    python scripts/make_movie.py -d /path/to/run/MS --units ion

    # non-interactive: one diagnostic, cropped, for sbatch
    python scripts/make_movie.py -d /path/to/run/MS/FLD/b2-savg --no-interactive \\
        --units ion --xlim 80 120 --vmin -0.1 --vmax 0.1 -s 4 -o b2_crop
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from multiprocessing import Pool

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — CRITICAL for parallel rendering
import numpy as np
import osh5def
import osh5io
import osh5vis

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

import plot_style
from analysis_utils import axis_values, load_config


# ---------------------------------------------------------------------------
# Normalisation — the only physics — lives in src/plot_style.py (DisplayUnits),
# shared with the analysis scripts and the --units flag.  build_units() resolves
# d_i / T_ci from the run dir (honouring a cached t_ci), so make_movie just maps
# its movie run dir + optional --config onto that helper below.
# ---------------------------------------------------------------------------

def _movie_units(units, sim_dir, config_path):
    """DisplayUnits for the movie: like plot_style.build_units, but the --config is
    only borrowed for its tuned upstream region/cached t_ci while the *fields* read
    come from this movie's own run dir (a movie may target a different run)."""
    units = plot_style.resolve_units(units)  # auto -> ion in publication mode, else electron
    if units == "electron" or config_path is None:
        return plot_style.build_units(units, sim_dir=sim_dir)
    cfg = load_config(config_path)
    cfg_run = os.path.basename(cfg["sim_dir"].rstrip("/"))
    movie_run = os.path.basename(sim_dir.rstrip("/"))
    if cfg_run != movie_run:
        print(f"  ! --config is for run {cfg_run!r} but this movie is run "
              f"{movie_run!r}; using the movie run's fields with the config's "
              "shock params.", flush=True)
    cfg["sim_dir"] = sim_dir  # read this run's fields, but use the config's regions / t_ci
    return plot_style.build_units(units, cfg=cfg)


def _value_to_index(axis_vals, lo, hi):
    """Map a physical [lo, hi] window to nearest array indices on ``axis_vals``.

    ``axis_vals`` is the 1D coordinate array for the axis (from
    ``analysis_utils.axis_values``).  ``None`` for ``lo``/``hi`` means "use the
    array edge".  Returns ``(i_lo, i_hi)`` suitable for slicing ``arr[i_lo:i_hi]``
    (``i_hi`` is exclusive and clamped to ``len`` so the endpoint is included).
    """
    n = len(axis_vals)
    i_lo = 0 if lo is None else int(np.argmin(np.abs(axis_vals - lo)))
    i_hi = n if hi is None else int(np.argmin(np.abs(axis_vals - hi))) + 1
    i_lo = max(0, min(i_lo, n - 1))
    i_hi = max(i_lo + 1, min(i_hi, n))
    return i_lo, i_hi


class MovieMaker:
    """Render a directory of OSIRIS ``*.h5`` dumps into an MP4.

    Attributes
    ----------
    root_dir : Path
        Either an ``MS`` tree (interactive: scanned for diagnostics) or a single
        diagnostic directory of ``*.h5`` files (non-interactive).
    norm : Normalization
        Display units (length-axis rescaling + title time unit).
    dpi : int
        Output image resolution.
    cmap : str
        Colormap for 2D plots.
    xlim, ylim : (float, float) or None
        Physical crop windows in the active length unit; ylim applies to 2D only.
    vmin, vmax : float or None
        Data scale (2D colourbar / 1D line y-range); auto from the middle frame
        when unset.
    title, grid, reference_density : misc display options.
    """

    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.norm = plot_style.electron_units()
        self.dpi = 100
        self.cmap = "inferno"
        self.log = False
        self.xlim = None
        self.ylim = None
        self.vmin = None
        self.vmax = None
        self.title = None
        self.grid = False
        self.reference_density = None

    # -- discovery ---------------------------------------------------------

    def _look_through_root(self):
        """Return the sorted list of diagnostic subdirs (relative to root)."""
        diagnostics = []
        for item in self.root_dir.rglob("*.h5"):
            parent = item.parent.relative_to(self.root_dir)
            if parent not in diagnostics:
                diagnostics.append(parent)
        return sorted(diagnostics, key=str)

    @staticmethod
    def _is_diagnostic_dir(path):
        """True if ``path`` directly contains ``*.h5`` frames."""
        return any(Path(path).glob("*.h5"))

    @staticmethod
    def _maybe_flip(arr):
        """Return ``-arr`` if every element is ≤ 0 (e.g. electron phase-space p1x1).

        This keeps log-scale plotting viable for diagnostics whose values are
        conventionally negative (flipped sign is physically equivalent for the
        visualisation, and the title will still show the correct field name).
        """
        if np.all(arr <= 0):
            return -arr
        return arr

    def _auto_vminmax(self, frames):
        """Set vmin/vmax from the middle frame unless already overridden.

        With ``log`` the scale is taken from the *positive* data only, so a
        LogNorm (2D) or a log y-axis (1D) gets a usable lower bound.
        """
        if self.vmin is not None and self.vmax is not None:
            return
        arr = self._maybe_flip(
            np.asarray(osh5io.read_h5(str(frames[len(frames) // 2])), dtype=float)
        )
        if self.log:
            pos = arr[arr > 0]
            lo = float(pos.min()) if pos.size else 1e-12
            hi = float(pos.max()) if pos.size else 1.0
        else:
            lo, hi = float(np.min(arr)), float(np.max(arr))
        if self.vmin is None:
            self.vmin = lo
        if self.vmax is None:
            self.vmax = hi

    # -- per-axis units handling ------------------------------------------

    def _crop_and_normalize(self, data):
        """Crop ``data`` to the requested window and rescale its length axes.

        Crop bounds are per *plotted* axis (1D: --xlim on the spatial axis; 2D:
        --xlim → axes[1] horizontal, --ylim → axes[0] vertical).  A bound on a
        length axis is given in the active display unit, so it is multiplied by
        ``length_factor`` to reach the on-disk c/wpe coordinate before indexing;
        a bound on a non-length axis (momentum) is used as-is.  In ion units the
        length axes are then rescaled (coords / d_i); momentum axes are never
        touched.  Axis *labels* are set separately in :meth:`_label_kwargs`.
        """
        targets = {0: self.xlim} if data.ndim == 1 else {1: self.xlim, 0: self.ylim}
        L = self.norm.length_factor

        slices = [slice(None)] * data.ndim
        for ax_idx, bound in targets.items():
            if bound is None:
                continue
            lo, hi = bound
            if data.axes[ax_idx].units.is_length():
                lo, hi = lo * L, hi * L          # display unit -> on-disk c/wpe
            i0, i1 = _value_to_index(axis_values(data, ax_idx), lo, hi)
            slices[ax_idx] = slice(i0, i1)
        data = data[tuple(slices)]

        if self.norm.units != "electron":
            for i, ax in enumerate(data.axes):
                if ax.units.is_length():  # rescale coords; attrs (incl. UNITS) kept
                    data.axes[i] = osh5def.DataAxis(
                        ax.min / L, ax.max / L, ax.size, attrs=dict(ax.attrs)
                    )
        return data

    def _label_kwargs(self, data):
        """Explicit osplot axis labels for length axes in non-electron units.

        Passing the label directly avoids re-deriving it from the (still c/wpe)
        axis units after we rescale the coordinates; momentum/data axes are left
        to osplot's default labelling.
        """
        if self.norm.units == "electron":
            return {}

        def length_label(ax):
            base = ax.attrs.get("LONG_NAME", ax.name)
            return f"${base}\\ [{self.norm.length_label}]$"

        kw = {}
        if data.ndim == 1:
            if data.axes[0].units.is_length():
                kw["xlabel"] = length_label(data.axes[0])
        else:
            if data.axes[1].units.is_length():
                kw["xlabel"] = length_label(data.axes[1])
            if data.axes[0].units.is_length():
                kw["ylabel"] = length_label(data.axes[0])
        return kw

    def _title_for(self, data):
        if self.title is not None:
            return str(self.title)
        t = np.round(data.run_attrs["TIME"][0] / self.norm.time_factor, 2)
        return f"{data.run_attrs['NAME']}  t = {t} ${self.norm.time_label}$"

    # -- rendering ---------------------------------------------------------

    def _frame_generator(self, path_to_frame):
        """Render one ``*.h5`` dump to ``tmp/<frame>.png``. Runs in a worker."""
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        plt.close("all")

        try:
            data = self._crop_and_normalize(osh5io.read_h5(path_to_frame))
            if np.all(np.asarray(data, dtype=float) <= 0):
                data = -data
            label_kw = self._label_kwargs(data)

            if data.ndim == 1:
                fig = plt.figure(figsize=(10, 6), dpi=self.dpi)
                # On a log y-axis a non-positive lower bound is invalid -> autoscale it.
                ylo = self.vmin if not (self.log and (self.vmin or 0) <= 0) else None
                osh5vis.osplot(data, ylim=[ylo, self.vmax], **label_kw)
                if self.log:
                    plt.yscale("log")
            else:
                fig = plt.figure(figsize=(10, 10), dpi=self.dpi)
                if self.log:
                    from matplotlib.colors import LogNorm
                    # LogNorm needs a positive vmin; fall back to autoscale otherwise.
                    vlo = self.vmin if (self.vmin or 0) > 0 else None
                    osh5vis.osplot(data, norm=LogNorm(vmin=vlo, vmax=self.vmax),
                                   cmap=self.cmap, **label_kw)
                else:
                    osh5vis.osplot(data, vmin=self.vmin, vmax=self.vmax,
                                   cmap=self.cmap, **label_kw)

            plt.title(self._title_for(data))
            if self.grid:
                plt.grid(True, which="both", linestyle="--", linewidth=0.5)
            if self.reference_density is not None:
                plt.xlabel(f"Reference density: {self.reference_density}")

            output_path = f"tmp/{path_to_frame.split('-')[-1]}.png"
            plt.savefig(output_path, dpi=self.dpi, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            del data
            return output_path
        except Exception as exc:  # one bad dump must not kill the whole movie
            plt.close("all")
            print(f"  ! failed on {path_to_frame}: {exc}", flush=True)
            return None

    def _save_movie(self, path_to_tmp, fps=40, width=1920, output_name="output"):
        path_to_tmp = Path(path_to_tmp).resolve()
        png_pattern = str(path_to_tmp / "*.h5.png")
        output_file = str(path_to_tmp / f"../{output_name}.mp4")
        # scale=width:-2 preserves aspect ratio (height divisible by 2 for h264).
        subprocess.call([
            "ffmpeg", "-y",
            "-framerate", str(fps), "-pattern_type", "glob", "-i", png_pattern,
            "-c:v", "libx264", "-vf", f"scale={width}:-2,format=yuv420p", output_file,
        ])

    def render(self, diag_dir, output_name, skip_frames=1, n_jobs=16, fps=40):
        """Render every ``*.h5`` in ``diag_dir`` (after skipping) into an MP4."""
        diag_dir = Path(diag_dir).resolve()
        frames = sorted(diag_dir.glob("*.h5"))[::skip_frames]
        if not frames:
            raise ValueError(f"No .h5 files found in {diag_dir}")

        self._auto_vminmax(frames)

        tmp = Path("tmp")
        tmp.mkdir(exist_ok=True)
        if any(tmp.iterdir()):
            print("Non-empty 'tmp/' directory; refusing to overwrite. Clean it first.")
            return

        n_jobs = max(1, min(n_jobs, len(frames)))
        chunksize = max(1, len(frames) // (n_jobs * 4))
        print(f"Rendering {len(frames)} frames with {n_jobs} workers "
              f"(chunksize={chunksize}, vmin={self.vmin:.3g}, vmax={self.vmax:.3g})...",
              flush=True)

        # imap_unordered + maxtasksperchild: bounds matplotlib memory growth over
        # long runs; frame order is encoded in the PNG name so ffmpeg still globs
        # them in sequence.
        done = 0
        with Pool(n_jobs, maxtasksperchild=50) as pool:
            for _ in pool.imap_unordered(
                self._frame_generator, [str(f) for f in frames], chunksize=chunksize
            ):
                done += 1
                if done % 50 == 0 or done == len(frames):
                    print(f"  {done}/{len(frames)} frames", flush=True)

        self._save_movie("tmp", fps=fps, output_name=output_name)

        for f in tmp.iterdir():
            try:
                f.unlink()
            except Exception as exc:
                print(f"Failed to delete {f}: {exc}")
        tmp.rmdir()
        print(f"Movie written: {output_name}.mp4", flush=True)


# ---------------------------------------------------------------------------
# Interactive override prompts (physical values, in the active length unit)
# ---------------------------------------------------------------------------

def _prompt_pair(label):
    """Prompt for an optional 'LO HI' pair of floats; Enter ⇒ None."""
    raw = input(f"  {label} as 'lo hi' (Enter to skip): ").strip()
    if not raw:
        return None
    try:
        lo, hi = (float(v) for v in raw.split())
        return (lo, hi)
    except ValueError:
        print("    invalid; skipping.")
        return None


def _interactive_setup(maker, frames):
    """Print frame info, then prompt for log scale + optional overrides.

    Mutates ``maker`` in place.  The middle-frame data range is shown to inform
    the vmin/vmax choice; if log scale is selected the log10 range of the
    positive data is shown as well.
    """
    arr = np.asarray(osh5io.read_h5(str(frames[len(frames) // 2])), dtype=float)
    print(f"  frames:     {len(frames)}")
    print(f"  shape:      {tuple(arr.shape)}")
    print(f"  data range: {float(arr.min()):.4g} .. {float(arr.max()):.4g}  (middle frame)")

    maker.log = input("  log scale? [y/N]: ").strip().lower().startswith("y")
    if maker.log:
        pos = arr[arr > 0]
        if pos.size:
            print(f"  log10 range (positive data): "
                  f"{np.log10(pos.min()):.3g} .. {np.log10(pos.max()):.3g}")
        else:
            print("  WARNING: no positive data; a log scale will mask everything.")

    if input("  override limits? [y/N]: ").strip().lower().startswith("y"):
        unit = "d_i" if maker.norm.units == "ion" else "c/wpe"
        maker.xlim = _prompt_pair(f"xlim (spatial, {unit})")
        maker.ylim = _prompt_pair(f"ylim (2D vertical axis; spatial in {unit}, else native)")
        vlim = _prompt_pair("vmin vmax (data/colour scale)")
        if vlim is not None:
            maker.vmin, maker.vmax = vlim
        dpi = input("  dpi (Enter for 100): ").strip()
        if dpi:
            try:
                maker.dpi = int(dpi)
            except ValueError:
                print("    invalid; keeping current dpi.")


def _interactive_loop(maker, skip_frames, n_jobs, fps):
    """The browse-a-run menu."""
    while True:
        subdirs = maker._look_through_root()
        if not subdirs:
            print("No .h5 diagnostics found under", maker.root_dir)
            return
        os.system("clear" if os.name == "posix" else "cls")
        print(maker.root_dir.resolve(), f"  [units: {maker.norm.units}]")
        for i, sd in enumerate(subdirs):
            print(f"{i:02d}:    {sd}")

        choice = input("\nDiagnostic number to animate ('exit' to quit): ").strip()
        if choice.lower() == "exit":
            return
        try:
            idx = int(choice)
        except ValueError:
            print("Enter a number or 'exit'.")
            continue
        if not (0 <= idx < len(subdirs)):
            print(f"Out of range (0-{len(subdirs) - 1}).")
            continue

        diag_dir = (maker.root_dir / subdirs[idx]).resolve()
        frames = sorted(diag_dir.glob("*.h5"))[::skip_frames]
        if not frames:
            print("  no .h5 frames here.")
            continue
        # Reset per-movie overrides so each pick starts from auto-scaling.
        maker.log = False
        maker.xlim = maker.ylim = maker.vmin = maker.vmax = None
        print(f"\n{subdirs[idx]}")
        _interactive_setup(maker, frames)
        output_name = str(subdirs[idx]).replace("/", "_")
        maker.render(diag_dir, output_name, skip_frames=skip_frames,
                     n_jobs=n_jobs, fps=fps)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _default_n_jobs():
    try:
        return min(16, len(os.sched_getaffinity(0)))
    except AttributeError:
        return min(16, os.cpu_count() or 1)


def _pair(values):
    return None if values is None else (values[0], values[1])


def _resolve_sim_dir(path):
    """Run directory for ``path`` = the ancestor just above the ``MS`` tree."""
    p = Path(path).resolve()
    parts = p.parts
    if "MS" in parts:
        return str(Path(*parts[: parts.index("MS")]))
    return str(p)  # path is (or is above) the run dir; RunSpec validates downstream


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-d", "--data", default=None,
                   help="MS tree (interactive) or a single diagnostic dir (non-interactive)")
    p.add_argument("-o", "--output", default=None,
                   help="Output movie name without .mp4 (non-interactive)")
    plot_style.add_units_arg(p)
    p.add_argument("--config", default=None,
                   help="Analysis YAML config; with --units ion, T_ci comes from the "
                        "config's cached t_ci (or is measured over its tuned upstream "
                        "region) instead of the whole box.")
    p.add_argument("-f", "--fps", type=int, default=40, help="Frames per second")
    p.add_argument("-s", "--skip-frames", type=int, default=1,
                   help="Render every Nth frame")
    p.add_argument("-n", "--n-jobs", type=int, default=None,
                   help="Parallel workers (default: auto, capped at 16)")
    p.add_argument("--dpi", type=int, default=100, help="Output image DPI")
    p.add_argument("--xlim", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                   help="Horizontal spatial crop, in the active length unit")
    p.add_argument("--ylim", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                   help="2D vertical-axis crop (spatial axes in the length unit)")
    p.add_argument("--log", action="store_true",
                   help="Log scale: LogNorm colour (2D) or log y-axis (1D)")
    p.add_argument("--vmin", type=float, default=None, help="Data/colour scale min")
    p.add_argument("--vmax", type=float, default=None, help="Data/colour scale max")
    p.add_argument("--cmap", default="inferno", help="Colormap for 2D plots")
    p.add_argument("--title", default=None, help="Static title (overrides the time title)")
    p.add_argument("--grid", action="store_true", help="Draw grid lines")
    p.add_argument("--reference-density", type=float, default=None,
                   help="Annotate the reference density on the x-label")
    p.add_argument("--no-interactive", action="store_true",
                   help="Force non-interactive rendering from flags only")
    plot_style.add_publication_arg(p)
    args = p.parse_args()
    plot_style.apply(args.publication)  # before the render Pool forks, so workers inherit

    data_path = args.data or input("Path to OSIRIS MS directory (or diagnostic dir): ").strip()
    n_jobs = args.n_jobs or _default_n_jobs()

    maker = MovieMaker(data_path)
    maker.norm = _movie_units(args.units, _resolve_sim_dir(data_path), args.config)
    maker.dpi = args.dpi
    maker.cmap = args.cmap
    maker.title = args.title
    maker.grid = args.grid
    maker.reference_density = args.reference_density

    non_interactive = args.no_interactive or MovieMaker._is_diagnostic_dir(data_path)
    if non_interactive:
        maker.log = args.log
        maker.xlim = _pair(args.xlim)
        maker.ylim = _pair(args.ylim)
        maker.vmin = args.vmin
        maker.vmax = args.vmax
        output_name = args.output or Path(data_path).resolve().name
        maker.render(data_path, output_name, skip_frames=args.skip_frames,
                     n_jobs=n_jobs, fps=args.fps)
    else:
        _interactive_loop(maker, args.skip_frames, n_jobs, args.fps)


if __name__ == "__main__":
    main()
