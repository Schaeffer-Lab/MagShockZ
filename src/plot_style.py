"""Shared presentation helpers for the analysis plotting scripts.

Two orthogonal display knobs live here, each a single flag wired into every script:

  - ``--publication`` (:func:`add_publication_arg` / :func:`apply`) re-renders a
    script's saved figures with large, paper / slide-sized text.  The default
    ("screen") styling is matplotlib's own, so figures are byte-for-byte unchanged
    unless the flag is passed.  This is rcParams-only.

  - ``--units electron|ion`` (:func:`add_units_arg` / :func:`build_units`) chooses the
    *display* normalisation of length and time axes: native OSIRIS ``c/ωpe`` & ``1/ωpe``
    (``electron``), or the ion inertial length ``d_i`` & upstream ion gyroperiod ``T_ci``
    (``ion``).  :func:`build_units` returns a :class:`DisplayUnits` that scripts use to
    rescale plotted coordinates and label axes; it does NOT touch the saved analysis data.

This lives in ``src/`` as the single source of the look + unit mapping so every script
shares one definition, but it is *not* part of the dependency-light, CI-tested layer for
the IO it can reach: :func:`apply` imports matplotlib (deferred) and :func:`build_units`'
ion path imports ``analysis_utils`` (deferred).  The :class:`DisplayUnits` dataclass core
(numpy only) is pure and unit-tested; merely importing this module stays cheap and safe.

Usage in a script::

    import plot_style
    ...
    plot_style.add_publication_arg(parser)
    plot_style.add_units_arg(parser)
    args = parser.parse_args()
    plot_style.apply(args.publication)        # no-op unless --publication given
    disp = plot_style.build_units(args.units, cfg=cfg, config_path=args.config)
"""

from dataclasses import dataclass

import numpy as np

# Font sizes (points) and weights applied in publication mode.  The hierarchy
# (figure title > axis title > axis labels > ticks/legend) mirrors the screen
# default but ~2x larger, so figures stay legible shrunk into a paper column or
# projected on a slide.  rcParams are global, so setting them before any figure is
# created restyles every subsequent plot the script draws (including subprocess
# workers forked after this call, e.g. make_movie's render pool).
_PUB_RC = {
    "font.size": 20,
    "axes.titlesize": 24,
    "axes.labelsize": 22,
    "axes.titleweight": "bold",
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
    "legend.title_fontsize": 18,
    "figure.titlesize": 26,
    "figure.titleweight": "bold",
    "axes.linewidth": 1.6,
    "lines.linewidth": 2.4,
    "lines.markersize": 9,
    "xtick.major.width": 1.4,
    "ytick.major.width": 1.4,
    "xtick.major.size": 7,
    "ytick.major.size": 7,
    "savefig.dpi": 200,
}


def add_publication_arg(parser):
    """Add the shared ``--publication`` flag to an argparse parser; returns it."""
    parser.add_argument(
        "--publication", "--pub", action="store_true", dest="publication",
        help="Render figures with large, paper/slide-sized text (default: screen).",
    )
    return parser


# Set by :func:`apply`; read by :func:`build_units` so that ``--units`` left at its
# "auto" default follows the publication knob (publication figures are for papers /
# slides, where ion units — d_i, T_ci — are the natural axes).
_PUBLICATION = False


def apply(publication=False):
    """Apply the publication rcParams when ``publication`` is truthy, else no-op.

    Records the publication state regardless, so :func:`build_units` can default the
    display units to ion when figures are rendered for publication.
    """
    global _PUBLICATION
    _PUBLICATION = bool(publication)
    if not publication:
        return
    import matplotlib
    matplotlib.rcParams.update(_PUB_RC)


# ---------------------------------------------------------------------------
# Display units — map on-disk OSIRIS units to the chosen display normalisation
# ---------------------------------------------------------------------------

@dataclass
class DisplayUnits:
    """How to map on-disk OSIRIS units (c/ωpe, 1/ωpe) to the display units.

    ``length_factor`` divides every *length* coordinate for display (1.0 for
    electron / c/ωpe; d_i for ion).  ``time_factor`` divides a time for display
    (1.0 for electron / 1-over-ωpe; T_ci for ion).  The labels are TeX fragments
    used on axes / in titles.  Only length and time are rescaled — momentum /
    velocity axes (``u [c]``) are never touched.
    """

    units: str
    length_factor: float
    time_factor: float
    length_label: str   # display length unit, e.g. r"c/\omega_{pe}" or "d_i"
    time_label: str     # display time unit,  e.g. r"\omega_{pe}^{-1}" or "T_{ci}"

    # -- value rescaling (length / time coordinates -> display unit) ----------

    def x(self, value):
        """Rescale a length value or array [c/ωpe] into the display length unit."""
        return np.asarray(value, dtype=float) / self.length_factor

    def t(self, value):
        """Rescale a time value or array [1/ωpe] into the display time unit."""
        return np.asarray(value, dtype=float) / self.time_factor

    # -- ready-made axis labels / titles --------------------------------------

    def xlabel(self, symbol="x"):
        """Axis label for a length axis, e.g. ``$x\\ [c/\\omega_{pe}]$``."""
        return rf"${symbol}\ [{self.length_label}]$"

    def tlabel(self, symbol="t"):
        """Axis label for a time axis, e.g. ``$t\\ [\\omega_{pe}^{-1}]$``."""
        return rf"${symbol}\ [{self.time_label}]$"

    def time_title(self, t_sim):
        """Title fragment for a simulation time, e.g. ``$t = 3.21\\ T_{ci}$``."""
        return rf"$t = {float(self.t(t_sim)):.2f}\ {self.time_label}$"

    # -- osh5 H5Data frames (streaks / movie frames) --------------------------

    def rescale_length_axes(self, h5data):
        """Return ``h5data`` with every *length* DataAxis rescaled into display units.

        The coordinates of each length axis are divided by ``length_factor`` (axis
        attrs, including the on-disk UNITS, are preserved); non-length axes are left
        untouched.  A no-op in electron units.  Axis *labels* are not changed here —
        use :meth:`length_axis_label` when drawing, since the on-disk UNITS no longer
        describe the rescaled coordinates.
        """
        if self.units == "electron":
            return h5data
        import osh5def
        out = h5data
        for i, ax in enumerate(out.axes):
            if ax.units.is_length():
                out.axes[i] = osh5def.DataAxis(
                    ax.min / self.length_factor, ax.max / self.length_factor,
                    ax.size, attrs=dict(ax.attrs),
                )
        return out

    def length_axis_label(self, ax):
        """Display label for an osh5 length ``DataAxis`` (LONG_NAME + display unit)."""
        base = ax.attrs.get("LONG_NAME", ax.name)
        return rf"${base}\ [{self.length_label}]$"


def electron_units() -> DisplayUnits:
    """The native OSIRIS display units (c/ωpe, 1/ωpe); identity rescaling."""
    return DisplayUnits("electron", 1.0, 1.0, r"c/\omega_{pe}", r"\omega_{pe}^{-1}")


def add_units_arg(parser):
    """Add the shared ``--units {electron,ion}`` flag to an argparse parser; returns it.

    Default is ``auto``: ion units in publication mode, electron otherwise (resolved in
    :func:`build_units`).  Pass ``--units electron|ion`` to force one regardless.
    """
    parser.add_argument(
        "--units", choices=("auto", "electron", "ion"), default="auto",
        help="Display normalisation: electron (c/ωpe, 1/ωpe), ion (d_i, T_ci), or "
             "auto (ion when --publication, else electron).",
    )
    return parser


def resolve_units(units):
    """Resolve the ``auto`` units choice against the recorded publication state.

    ``auto`` -> ``ion`` when figures are rendered for publication (see :func:`apply`),
    else ``electron``.  An explicit ``electron``/``ion`` is returned unchanged.
    """
    if units == "auto":
        return "ion" if _PUBLICATION else "electron"
    return units


def build_units(units, *, cfg=None, sim_dir=None, config_path=None) -> DisplayUnits:
    """Resolve a :class:`DisplayUnits` for ``units`` from the run directory / config.

    ``electron`` is the identity mapping.  ``ion`` needs the ion mass-per-charge
    ``|rqm_i|`` (from the run's deck) for ``d_i = sqrt(|rqm_i|)`` and the upstream ion
    gyroperiod ``T_ci``.  ``T_ci`` is taken from the config's cached ``t_ci`` key when
    present (written for free by ``scripts/tune_shock.py`` at t=0); otherwise it is
    measured from the field — over the config's tuned upstream region when a config is
    given (``analysis_utils.upstream_field_magnitude``), else the whole box at t=0.

    Pass either ``cfg`` (a parsed analysis config, used for the cached ``t_ci`` and the
    upstream region) or ``sim_dir`` (a run directory).  ``analysis_utils`` and the
    ion-physics helpers are imported lazily so the electron path stays dependency-light.

    ``units='auto'`` (the flag default) resolves to ion in publication mode, else electron.
    """
    units = resolve_units(units)
    if units == "electron":
        return electron_units()

    import analysis_utils
    from dimensionless_params import ion_gyroperiod, ion_skin_depth

    run_dir = sim_dir if sim_dir is not None else (cfg["sim_dir"] if cfg else None)
    if run_dir is None:
        raise ValueError("build_units(units='ion') needs cfg or sim_dir.")
    abs_rqm_i = abs(analysis_utils.run_from_config({"sim_dir": run_dir}).rqm)
    d_i = ion_skin_depth(abs_rqm_i)

    cached = cfg.get("t_ci") if cfg else None
    if cached is not None:
        T_ci = float(cached)
        provenance = "cached t_ci (config)"
    else:
        if cfg is not None:
            B_up = analysis_utils.upstream_field_magnitude(cfg)
            region = "config upstream window"
        else:
            B_up = _wholebox_field_magnitude(run_dir)
            region = "whole box at t=0 (no --config)"
        T_ci = ion_gyroperiod(abs_rqm_i, B_up)
        provenance = f"measured |B'|={B_up:.4g} [{region}]"

    if not (np.isfinite(T_ci) and np.isfinite(d_i)):
        raise ValueError(
            f"--units ion could not be resolved (|rqm_i|={abs_rqm_i}, T_ci={T_ci}); "
            "check the run has a parseable deck and a non-zero upstream field / cached t_ci."
        )
    print(f"ion units: |rqm_i|={abs_rqm_i:.4g}  d_i={d_i:.4g} c/wpe  "
          f"T_ci={T_ci:.4g} 1/wpe  [{provenance}]", flush=True)
    return DisplayUnits("ion", d_i, T_ci, r"d_i", r"T_{ci}")


def build_units_from_args(args, cfg) -> DisplayUnits:
    """:func:`build_units` for a script's parsed ``args`` + its ``cfg``.

    Wraps the one idiom every ``--config``-driven analysis script repeats —
    ``build_units(args.units, cfg=cfg, config_path=os.path.abspath(args.config))`` —
    so call sites drop the repeated ``os.path.abspath`` boilerplate.  ``args`` must
    carry ``.units`` (from :func:`add_units_arg`) and ``.config`` (the config path).
    """
    import os
    return build_units(args.units, cfg=cfg, config_path=os.path.abspath(args.config))


def _wholebox_field_magnitude(sim_dir) -> float:
    """Median |B'| over the earliest available field dump [OSIRIS B_0 units].

    The whole-frame fallback used by :func:`build_units` when no config is given: the
    first dump is the least-shocked state, so its box-wide median approximates the
    upstream field (diluted by the driver/downstream side, which is why a config — and
    thus the tuned upstream window — is preferred).  Reads b1/b2/b3 via the run's
    layout-aware field names.
    """
    import glob
    import os
    import osh5io
    from analysis_utils import detect_layout, diag_path

    layout = detect_layout(sim_dir)
    b2_sum = None
    for name in ("b1", "b2", "b3"):
        diag_dir = os.path.dirname(diag_path(sim_dir, layout.field_quantity(name), 0))
        files = sorted(glob.glob(os.path.join(diag_dir, "*.h5")))
        if not files:
            raise ValueError(
                f"--units ion needs the B field, but no {name} dumps were found "
                f"under {diag_dir}.")
        arr = np.asarray(osh5io.read_h5(files[0]), dtype=float)
        b2_sum = arr ** 2 if b2_sum is None else b2_sum + arr ** 2
    return float(np.median(np.sqrt(b2_sum)))
