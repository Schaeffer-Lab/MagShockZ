"""analysis_utils.py — lightweight OSIRIS analysis helpers.

Two public objects:

  MagShockZRun
      Thin wrapper around an osiris_utils.Simulation that exposes field access
      and physical-unit conversions (frequencies, lengths, speeds, B-field
      normalisation).

  StreakBuilder
      Standalone helper that stacks a sequence of 1D spatial profiles into an
      osh5def.H5Data object with explicit [time, space] axes so the result can
      be passed directly to osh5vis.osplot().
"""

import dataclasses
import glob
import os
import re
from typing import List, Optional

import astropy
import astropy.constants
import astropy.units
import numpy as np
import osiris_utils
# osiris_utils' own quantity registries — reused by diag_path so our on-disk path
# routing stays in sync with what osiris_utils.Simulation knows how to load.
from osiris_utils.data.diagnostic import (
    OSIRIS_FLD, OSIRIS_PHA, OSIRIS_SPECIE_REPORTS, OSIRIS_SPECIE_REP_UDIST,
)
import osh5def
import osh5io
import plasmapy
import plasmapy.formulary
import plasmapy.particles.particle_class
import yaml

# RunSpec — the single source of truth for a run's parameters — lives in its own
# dependency-light module; re-exported here so analysis_utils.RunSpec keeps working.
from run_spec import RunSpec, _parse_cli_flags  # noqa: F401

# Fallback input-deck filename when neither the config nor the run spec names one.
DEFAULT_INPUT_DECK = "magshockz_gpu.1d"


# ---------------------------------------------------------------------------
# MagShockZRun
# ---------------------------------------------------------------------------

class MagShockZRun:
    """Lightweight OSIRIS run context for field access and unit conversions.

    Parameters
    ----------
    input_deck : str or Path
        Path to the OSIRIS input deck file.
    norm_density : astropy Quantity
        Normalisation density (e.g. ``5e18 * astropy.units.cm**-3``).
    B0 : astropy Quantity, optional
        Background magnetic field (Gauss).  Required for cyclotron-frequency
        and gyrotime conversions.
    Z : int, optional
        Ion charge number.  Required for ion-frequency / length methods.
    m_i : astropy Quantity, optional
        Ion mass.  Required for ion-frequency / length methods.
    """

    def __init__(
        self,
        input_deck: str,
        norm_density,
        B0=None,
        Z: int = None,
        m_i=None,
    ):
        self.sim = osiris_utils.Simulation(input_deck_path=input_deck)
        self.deck = self.sim._input_deck
        self.norm_density = norm_density
        self.B0 = B0
        self.Z = Z
        self.m_i = m_i

    # ------------------------------------------------------------------
    # Field access
    # ------------------------------------------------------------------

    def field(self, name: str):
        """Return the osiris_utils diagnostic field object for ``name``.

        Accepts both ``"quantity"`` and ``"top/sub"`` path forms.
        """
        for sep in ("/", "."):
            if sep in name:
                left, right = name.split(sep, 1)
                for a, b in [(left, right), (right, left)]:
                    try:
                        return self.sim[a][b]
                    except Exception:
                        pass
                raise KeyError(f"Could not resolve field '{name}'")
        return self.sim[name]

    # ------------------------------------------------------------------
    # Input-deck parameters
    # ------------------------------------------------------------------

    # Common short-name -> deck-name aliases for species lookup.
    _SPECIES_ALIASES = {
        "al": ("al", "aluminum", "aluminium"),
        "si": ("si", "silicon"),
        "e":  ("e", "electron", "electrons"),
    }

    def rqm_of(self, species_name: str) -> float:
        """Return the OSIRIS rqm = m/q for ``species_name`` straight from the deck.

        Each species carries its own rqm in the deck (e.g. Al=38, Si=39, e=-1),
        so this must be used per species rather than assuming one shared ion rqm.
        Accepts both the deck name and common aliases ("aluminum" -> "al", ...).
        """
        for candidate in self._SPECIES_ALIASES.get(species_name, (species_name,)):
            if candidate in self.deck.species:
                return self.deck.species[candidate].rqm
        raise KeyError(
            f"Species '{species_name}' not found in input deck "
            f"(available: {list(self.deck.species)})."
        )

    @property
    def rqm(self):
        """Ion rqm = m/q (mass-per-charge) relative to the electron (from input deck).

        Returns the first ion species found; use :meth:`rqm_of` to select a
        specific species when more than one ion is present.
        """
        for species_name in ("al", "aluminum", "si", "silicon", "ion"):
            if species_name in self.deck.species:
                return self.deck.species[species_name].rqm
        raise KeyError("No known ion species found in input deck.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _p(self, B_real=None, Z=None, m_i=None) -> dict:
        """Resolve optional parameters, falling back to instance defaults."""
        return {
            "B_real": B_real if B_real is not None else self.B0,
            "Z": Z if Z is not None else self.Z,
            "m_i": m_i if m_i is not None else self.m_i,
        }

    def _ion(self, Z, m_i):
        return plasmapy.particles.particle_class.CustomParticle(mass=m_i, Z=Z)

    # ------------------------------------------------------------------
    # Frequencies
    # ------------------------------------------------------------------

    @property
    def omega_pe(self):
        """Electron plasma frequency [rad/s]."""
        return plasmapy.formulary.plasma_frequency(self.norm_density, particle="e-").to("rad/s")

    def omega_pi(self, Z=None, m_i=None):
        """Ion plasma frequency [rad/s]."""
        p = self._p(Z=Z, m_i=m_i)
        return plasmapy.formulary.plasma_frequency(
            self.norm_density / p["Z"],
            particle=self._ion(p["Z"], p["m_i"]),
        ).to("rad/s")

    def omega_ce(self, B_real=None):
        """Electron cyclotron frequency [rad/s]."""
        return plasmapy.formulary.gyrofrequency(
            self._p(B_real=B_real)["B_real"], particle="e-"
        ).to("rad/s")

    def omega_ci(self, B_real=None, Z=None, m_i=None):
        """Ion cyclotron frequency [rad/s]."""
        p = self._p(B_real=B_real, Z=Z, m_i=m_i)
        return plasmapy.formulary.gyrofrequency(
            p["B_real"], particle=self._ion(p["Z"], p["m_i"])
        ).to("rad/s")

    def omega_ci_norm(self, B_real=None):
        """Ion cyclotron frequency in OSIRIS normalised units (omega_ci / omega_pe)."""
        return float(self.B_norm(B_real)) / abs(self.rqm)

    # ------------------------------------------------------------------
    # Lengths
    # ------------------------------------------------------------------

    def d_e(self):
        """Electron inertial length c/omega_pe [cm]."""
        return (astropy.constants.c.si / (self.omega_pe / astropy.units.rad)).to("cm")

    def d_i(self, Z=None, m_i=None):
        """Ion inertial length c/omega_pi [cm]."""
        return (astropy.constants.c.si / (self.omega_pi(Z, m_i) / astropy.units.rad)).to("cm")

    def lambda_D(self, T_e):
        """Debye length [cm] for electron temperature T_e."""
        return plasmapy.formulary.Debye_length(n_e=self.norm_density, T_e=T_e).to("cm")

    def lambda_D_norm(self, T_e):
        """Debye length in units of c/omega_pe."""
        return (self.lambda_D(T_e) / self.d_e()).to(astropy.units.dimensionless_unscaled)

    # ------------------------------------------------------------------
    # Speeds
    # ------------------------------------------------------------------

    def vA(self, B_real=None, Z=None, m_i=None):
        """Alfven speed [cm/s]."""
        p = self._p(B_real=B_real, Z=Z, m_i=m_i)
        return (
            p["B_real"] / np.sqrt(astropy.constants.mu0 * self.norm_density / p["Z"] * p["m_i"])
        ).to("cm/s")

    def cs(self, T_e, adiabatic_index: float = 5 / 3, Z=None, m_i=None):
        """Ion sound speed [cm/s]."""
        p = self._p(Z=Z, m_i=m_i)
        return (np.sqrt(adiabatic_index * p["Z"] * T_e / p["m_i"])).to("cm/s")

    # ------------------------------------------------------------------
    # OSIRIS normalisation
    # ------------------------------------------------------------------

    def B_norm(self, B_real=None):
        """Convert a real magnetic field to OSIRIS normalised units."""
        B = self._p(B_real=B_real)["B_real"]
        B_unit = (
            astropy.constants.m_e * self.omega_pe / astropy.units.rad / astropy.constants.e.si
        ).to(astropy.units.Gauss)
        return (B / B_unit).to(astropy.units.dimensionless_unscaled)


# ---------------------------------------------------------------------------
# HDF5 path helpers — the on-disk OSIRIS diagnostic layout in one place
# ---------------------------------------------------------------------------

def diag_path(sim_dir: str, quantity: str, t: int, species: Optional[str] = None) -> str:
    """On-disk path to a single OSIRIS HDF5 dump, mirroring osiris_utils' MS/ layout.

    One helper for every diagnostic family — fields, phase spaces, species charge
    densities and UDIST moments — keyed on the OSIRIS ``quantity`` string.  The
    family (and thus the ``MS/<DIR>/...`` subtree) is derived from osiris_utils'
    own quantity registries, so this stays in sync with what
    ``osiris_utils.Simulation`` can load.

    ``quantity`` carries its own averaging suffix — pass ``"b3"`` for the raw
    field or ``"b3-savg"`` for the time-averaged one, ``"charge"`` vs
    ``"charge-savg"`` for density.  Averaging is never assumed (the old
    ``field_path`` hard-coded ``-savg``, which this replaces).

    ``species`` is required for everything except fields::

        diag_path(d, "b3-savg", t)            -> MS/FLD/b3-savg/b3-savg-000360.h5
        diag_path(d, "p1x1", t, "al")         -> MS/PHA/p1x1/al/p1x1-al-000360.h5
        diag_path(d, "charge-savg", t, "e")   -> MS/DENSITY/e/charge-savg/charge-savg-e-000360.h5
        diag_path(d, "T11", t, "al")          -> MS/UDIST/al/T11/T11-al-000360.h5
    """
    if quantity in OSIRIS_FLD:
        return f"{sim_dir}/MS/FLD/{quantity}/{quantity}-{t:06d}.h5"

    # The remaining families are per-species; the filename is the same
    # (quantity-species-iter) but the directory nesting differs: phase spaces nest
    # quantity/species, while densities and moments nest species/quantity.
    if quantity in OSIRIS_PHA:
        subtree, inner = "PHA", f"{quantity}/{species}"
    elif quantity in OSIRIS_SPECIE_REPORTS:
        subtree, inner = "DENSITY", f"{species}/{quantity}"
    elif quantity in OSIRIS_SPECIE_REP_UDIST:
        subtree, inner = "UDIST", f"{species}/{quantity}"
    else:
        raise ValueError(
            f"Unknown OSIRIS quantity {quantity!r}; not in osiris_utils' FLD / PHA / "
            f"DENSITY / UDIST registries. Use osiris_utils.which_quantities() to list them."
        )

    if species is None:
        raise ValueError(f"quantity {quantity!r} ({subtree}) requires a species.")
    return f"{sim_dir}/MS/{subtree}/{inner}/{quantity}-{species}-{t:06d}.h5"


def axis_values(h5data, ax_idx: int) -> np.ndarray:
    """Return the coordinate values of axis ``ax_idx`` as a 1D linspace."""
    ax = h5data.axes[ax_idx]
    return np.linspace(ax.min, ax.max, ax.size)


# ---------------------------------------------------------------------------
# Run layout — bridge the on-disk differences between 1D and 2D runs
# ---------------------------------------------------------------------------

class RunLayout:
    """Per-run layout: dimensionality, shock-normal axis, diagnostic naming.

    The 1D and 2D OSIRIS runs differ in three ways that the analysis scripts
    would otherwise have to hardcode:

      - the shock-normal spatial axis is ``x1`` in 1D but ``x2`` in the 2D run
        (the phase spaces are binned against, and the shock propagates along,
        that axis);
      - phase spaces are named ``p{c}x1`` (1D) vs ``p{c}x2`` (2D);
      - species charge density may be the time-averaged diagnostic
        (``charge-savg``) or the raw one (``charge``).  This is a per-run
        diagnostic choice, NOT a 1D/2D distinction — every production run uses
        ``charge-savg``; only the early 2D test run dumped raw ``charge``.
        :func:`detect_layout` decides by inspecting which directory the run
        actually wrote, so new savg-using 2D runs resolve correctly.

    Build with :func:`detect_layout`; pass the instance to the path helpers and
    to :func:`transverse_profile` so a script stays dimension-agnostic.
    """

    def __init__(self, ndim: int, normal_axis: str, momenta: List[int],
                 density_savg: bool, field_savg: bool):
        self.ndim = ndim
        self.normal_axis = normal_axis          # "x1" or "x2"
        self.momenta = momenta                  # available momentum components, e.g. [1, 2, 3]
        self.density_savg = density_savg
        self.field_savg = field_savg

    def pha_name(self, component: int) -> str:
        """OSIRIS phase-space name for momentum ``component`` vs the normal axis.

        e.g. ``pha_name(1)`` -> ``"p1x1"`` (1D) or ``"p1x2"`` (2D).
        """
        if component not in self.momenta:
            raise KeyError(
                f"Momentum component p{component} not in this run "
                f"(available: {self.momenta})."
            )
        return f"p{component}{self.normal_axis}"

    @property
    def charge_quantity(self) -> str:
        """OSIRIS charge-density quantity name for this run: ``charge-savg`` if the
        run dumped the time-averaged density diagnostic, else raw ``charge``.  The
        choice is taken from what the run actually wrote (see :func:`detect_layout`),
        not from its dimensionality.  Pass straight to :func:`diag_path`."""
        return "charge-savg" if self.density_savg else "charge"

    def field_quantity(self, name: str) -> str:
        """OSIRIS field quantity name for this run: ``<name>-savg`` if the run dumped
        time-averaged fields, else raw ``<name>`` (e.g. ``field_quantity("b3")`` ->
        ``"b3-savg"`` or ``"b3"``).  The field analogue of :attr:`charge_quantity`;
        like it, the choice comes from what the run wrote (see :func:`detect_layout`),
        not its dimensionality.  Pass straight to :func:`diag_path`."""
        return f"{name}-savg" if self.field_savg else name

    def __repr__(self):
        return (f"RunLayout(ndim={self.ndim}, normal_axis={self.normal_axis!r}, "
                f"momenta={self.momenta}, density_savg={self.density_savg}, "
                f"field_savg={self.field_savg})")


def detect_layout(sim_dir: str) -> RunLayout:
    """Inspect a run's MS/ tree and return its :class:`RunLayout`.

    Detection is data-driven (no config needed):
      - shock-normal axis + available momenta from the ``MS/PHA/p?x?`` dir names;
      - ``density_savg`` from whether ``MS/DENSITY/<sp>/charge-savg`` exists;
      - ``field_savg`` from whether any ``MS/FLD/*-savg`` dir exists;
      - ``ndim`` from the number of spatial axes in a sample field dump.
    """
    pha_root = os.path.join(sim_dir, "MS", "PHA")
    names = [d for d in os.listdir(pha_root)
             if os.path.isdir(os.path.join(pha_root, d))]
    parsed = [re.match(r"p(\d)x(\d)$", n) for n in names]
    parsed = [m for m in parsed if m]
    if not parsed:
        raise RuntimeError(f"No 'p<c>x<d>' phase-space dirs found under {pha_root}.")
    normal_digits = {m.group(2) for m in parsed}
    if len(normal_digits) != 1:
        raise RuntimeError(
            f"Phase spaces bin against more than one spatial axis "
            f"({sorted('x'+d for d in normal_digits)}); cannot pick a single "
            f"shock-normal axis automatically."
        )
    normal_axis = "x" + normal_digits.pop()
    momenta = sorted({int(m.group(1)) for m in parsed})

    dens_root = os.path.join(sim_dir, "MS", "DENSITY")
    sp0 = next(d for d in sorted(os.listdir(dens_root))
               if os.path.isdir(os.path.join(dens_root, d)))
    density_savg = os.path.isdir(os.path.join(dens_root, sp0, "charge-savg"))

    fld_root = os.path.join(sim_dir, "MS", "FLD")
    savg_dirs = sorted(glob.glob(os.path.join(fld_root, "*-savg")))
    field_savg = bool(savg_dirs)

    # ndim from a sample field dump — prefer a savg dir when present, else any field
    # dir, so this also works for a run that only dumped raw (non-savg) fields.
    sample_dirs = savg_dirs or sorted(
        d for d in glob.glob(os.path.join(fld_root, "*")) if os.path.isdir(d))
    fld_files = [f for d in sample_dirs
                 for f in sorted(glob.glob(os.path.join(d, "*.h5")))]
    if not fld_files:
        raise RuntimeError(f"No field dumps under {fld_root}/*/.")
    sample = osh5io.read_h5(fld_files[0])
    ndim = sum(1 for a in sample.axes if a.name.startswith("x"))

    return RunLayout(ndim=ndim, normal_axis=normal_axis, momenta=momenta,
                     density_savg=density_savg, field_savg=field_savg)


def transverse_profile(frame, normal_axis: str, half_width: float = 5.0,
                       center: Optional[float] = None):
    """Collapse a 2D map to a 1D profile along ``normal_axis``.

    Averages the transverse spatial axis over a band of ``±half_width`` about
    ``center`` (default: the transverse axis' geometric center).  ``half_width``
    is in the data's spatial units — in OSIRIS normalised units c/ωpe, so the
    default 5.0 is five electron inertial lengths either side of center.

    A frame with a single spatial axis (any 1D profile, or a phase space already
    reduced over momentum) is returned unchanged, so callers can apply this
    uniformly to 1D and 2D runs.

    Returns an ``osh5def.H5Data`` carrying only the normal axis, with the input's
    ``data_attrs`` and ``run_attrs`` preserved (so it stacks in StreakBuilder).
    """
    spatial = [a for a in frame.axes if a.name.startswith("x")]
    if len(spatial) <= 1:
        return frame
    normal = next(a for a in frame.axes if a.name == normal_axis)
    trans = next(a for a in spatial if a.name != normal_axis)
    ti = next(i for i, a in enumerate(frame.axes) if a.name == trans.name)
    coords = np.linspace(trans.min, trans.max, trans.size)
    c = center if center is not None else 0.5 * (trans.min + trans.max)
    band = (coords >= c - half_width) & (coords <= c + half_width)
    if not band.any():  # window narrower than a cell -> nearest single cell
        band = np.zeros(trans.size, dtype=bool)
        band[int(np.argmin(np.abs(coords - c)))] = True
    reduced = np.asarray(frame).take(np.nonzero(band)[0], axis=ti).mean(axis=ti)
    return osh5def.H5Data(
        reduced,
        data_attrs=dict(frame.data_attrs),
        run_attrs=frame.run_attrs,
        axes=[normal],
    )


# ---------------------------------------------------------------------------
# Upstream / downstream region masks — two conventions, named explicitly
# ---------------------------------------------------------------------------

def region_masks(x, x_shock: float, x_downstream_start: float):
    """Boolean (upstream, downstream) masks bounded by a downstream start.

      upstream   : x > x_shock                            (undisturbed plasma)
      downstream : x_downstream_start <= x <= x_shock     (shocked plasma)

    Used by the temperature and energy-partition analyses.
    """
    x = np.asarray(x)
    upstream = x > x_shock
    downstream = (x >= x_downstream_start) & (x <= x_shock)
    return upstream, downstream


def window_masks(x, x_shock: float, dx: float, up_ncells: int, dn_ncells: int):
    """Boolean (upstream, downstream) masks as fixed-width windows about x_shock.

      upstream   : (x_shock,              x_shock + up_ncells * dx]
      downstream : [x_shock - dn_ncells * dx,  x_shock)

    Used by the dimensionless-parameter analysis, where the averaging region is
    a fixed number of grid cells either side of the shock rather than a manually
    inspected downstream boundary.
    """
    x = np.asarray(x)
    upstream = (x > x_shock) & (x <= x_shock + up_ncells * dx)
    downstream = (x >= x_shock - dn_ncells * dx) & (x < x_shock)
    return upstream, downstream


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    """Load an analysis YAML config.

    Applies the ``$MAGSHOCKZ_SIM_DIR`` override to ``sim_dir`` and normalises
    the ``times`` entry to a list of int dump indices (see :func:`parse_times`).
    """
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["sim_dir"] = os.environ.get("MAGSHOCKZ_SIM_DIR", cfg["sim_dir"])
    if "times" in cfg:
        cfg["times"] = parse_times(cfg["times"])
    return cfg


def load_results(path: str) -> dict:
    """Load a results .npz, unwrapping 0-d arrays to plain Python scalars."""
    d = np.load(path, allow_pickle=True)
    return {k: (d[k].item() if d[k].ndim == 0 else d[k]) for k in d.files}


def save_result(result, path: str) -> str:
    """Serialise a dataclass ``result`` to ``path`` (.npz) and return the path.

    The dataclass *is* the output schema, so this avoids hand-listing keys in every
    analysis script.  Each field becomes one flat, inspectable .npz entry:

      - array fields are stored as-is;
      - scalar/string fields are wrapped as 0-d arrays;
      - ``dict`` fields (e.g. per-region averages ``{"ram": ..., "thermal": ...}``)
        are flattened to ``<field>_<key>`` entries (so ``upstream={"ram":x}`` is
        stored as ``upstream_ram``), keeping the .npz a flat key/value store you
        can ``np.load`` and read directly.
    """
    flat = {}
    for f in dataclasses.fields(result):
        val = getattr(result, f.name)
        if isinstance(val, dict):
            for k, v in val.items():
                flat[f"{f.name}_{k}"] = np.asarray(v)
        else:
            flat[f.name] = np.asarray(val)
    np.savez(path, **flat)
    return path


def load_result(cls, path: str):
    """Inverse of :func:`save_result`: rebuild a ``cls`` dataclass from its .npz.

    Fields annotated ``dict`` are reassembled from their flattened
    ``<field>_<key>`` entries; all other fields are read by name (0-d arrays
    unwrapped to Python scalars).
    """
    d = np.load(path, allow_pickle=True)
    raw = {k: (d[k].item() if d[k].ndim == 0 else d[k]) for k in d.files}
    kwargs = {}
    for f in dataclasses.fields(cls):
        is_dict = f.type is dict or f.type == "dict"
        if is_dict:
            prefix = f.name + "_"
            kwargs[f.name] = {k[len(prefix):]: raw[k]
                              for k in raw if k.startswith(prefix)}
        else:
            kwargs[f.name] = raw[f.name]
    return cls(**kwargs)


def run_from_config(cfg: dict, *, B0=None, Z=None, m_i=None) -> "MagShockZRun":
    """Build a :class:`MagShockZRun` from a parsed analysis config.

    Norm density and the input-deck name come from the run's :class:`RunSpec`
    (single source of truth), not the analysis config.  An explicit
    ``input_deck`` in the config still overrides the deck name (escape hatch for a
    run whose generated deck name is unparseable); :data:`DEFAULT_INPUT_DECK` is
    the last resort.
    """
    spec = RunSpec.from_sim_dir(cfg["sim_dir"])
    deck_name = cfg.get("input_deck") or spec.deck_name or DEFAULT_INPUT_DECK
    return MagShockZRun(os.path.join(cfg["sim_dir"], deck_name),
                        norm_density=spec.norm_density, B0=B0, Z=Z, m_i=m_i)


def default_output_path(output: Optional[str], sim_dir: str, stem: str, t_val: int) -> str:
    """Resolve an output .npz path, creating the parent directory.

    If ``output`` is given it is used as-is (its directory is created);
    otherwise the path defaults to ``results/<run_name>/<stem>_t{t:06d}.npz``
    relative to the repository root.
    """
    if output is not None:
        out_dir = os.path.dirname(output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        return output
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run_name = os.path.basename(sim_dir.rstrip("/"))
    out_dir = os.path.join(repo_root, "results", run_name)
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"{stem}_t{t_val:06d}.npz")


def parse_times(times) -> List[int]:
    """Normalise the config ``times`` entry to a list of int dump indices.

    Accepted forms (both native YAML, no string parsing):

      - list                  ``[0, 20, 40, ...]``
      - start/stop/step map   ``{start: 0, stop: 360, step: 20}``  (stop inclusive)
    """
    if isinstance(times, dict):
        start, stop = int(times["start"]), int(times["stop"])
        step = int(times.get("step", 1))
        if step == 0:
            raise ValueError("times step must be non-zero.")
        return list(range(start, stop + (1 if step > 0 else -1), step))
    return [int(t) for t in times]


def resolve_dump_params(cfg: dict, t_val: int, t_sim: float) -> dict:
    """Return the resolved shock parameters for a specific dump.

    Global values from cfg["shock"] are used as the base; per-dump entries in
    cfg["dump_params"][t_val] are merged on top.  x_shock defaults to the
    formula x_shock_0 + v_shock * t_sim unless overridden in dump_params.
    x_downstream_start has no global default and must be present in dump_params.

    Parameters
    ----------
    cfg : dict
        Parsed YAML config.
    t_val : int
        Dump file suffix (key into dump_params).
    t_sim : float
        Simulation time read from HDF5 run_attrs, used for the x_shock formula.

    Raises
    ------
    KeyError
        If dump_params has no entry for t_val, or if x_downstream_start is
        missing from that entry.
    """
    shock = cfg["shock"]
    per_dump = cfg.get("dump_params", {}).get(t_val)
    if per_dump is None:
        raise KeyError(
            f"No dump_params entry for t_val={t_val}. "
            f"Add it to the config with at least x_downstream_start."
        )
    if "x_downstream_start" not in per_dump:
        raise KeyError(
            f"dump_params[{t_val}] is missing required key 'x_downstream_start'."
        )
    params = {
        "v_shock": shock["v_shock"],
        "x_shock_0": shock["x_shock_0"],
        "x_shock": shock["x_shock_0"] + shock["v_shock"] * t_sim,
    }
    params.update(per_dump)
    return params


def fit_shock_trajectory(cfg: dict, deg: int = 2, species: str = "e",
                         search_halfwidth: float = 400.0,
                         transverse_hw: float = 5.0) -> dict:
    """Detect the shock front across every configured dump and fit ``x_shock(t)``.

    Reuses the same density leading-edge detection as ``scripts/overview.py``
    (``shock.detect_front_edge``), seeded per dump by the config shock fit
    (``shock.v_shock`` / ``shock.x_shock_0``), then fits the trajectory with a
    degree-``deg`` polynomial (:func:`shock.robust_polyfit`) so the velocity can
    be taken as its analytic time-derivative.

    Like the overview, **frames before one ion gyroperiod are excluded from the
    fit**: the front is not magnetically organised until the upstream ions have
    gyrated once, and those early detections (piston/driver edge) otherwise
    corrupt the trajectory — especially its derivative.  ``T_ci`` is computed
    from the t=0 upstream |B| (in OSIRIS units the field equals ω_ce, so
    ω_ci = |B|/|rqm_i| and T_ci = 2π|rqm_i|/|B|); override with
    ``shock.fit_t_min`` in the config.  ``deg`` is reduced automatically if too
    few post-gyro frames are detected.

    Returns a dict with the polynomial ``coeffs`` (np.polyfit convention), the
    per-dump ``t_sim`` / ``x_det`` arrays, the boolean ``fit_mask`` of frames
    actually fit, ``t_min`` used, and the ``deg`` actually used.
    """
    import osh5io
    from shock import detect_front_edge, robust_polyfit

    sim_dir = cfg["sim_dir"]
    layout = detect_layout(sim_dir)
    v_cfg = float(cfg["shock"]["v_shock"])
    x0 = float(cfg["shock"]["x_shock_0"])

    t_sim, x_det = [], []
    for t in cfg["times"]:
        ne_h5 = osh5io.read_h5(diag_path(sim_dir, layout.charge_quantity, t, species))
        prof = transverse_profile(np.abs(ne_h5), layout.normal_axis, transverse_hw)
        x = axis_values(prof, ax_idx=0)
        ts = float(ne_h5.run_attrs["TIME"][0])
        x_pred = x0 + v_cfg * ts
        t_sim.append(ts)
        x_det.append(detect_front_edge(x, np.asarray(prof), x_pred, search_halfwidth))

    t_sim = np.asarray(t_sim, dtype=float)
    x_det = np.asarray(x_det, dtype=float)

    # Skip the first ion gyroperiod (front not yet magnetically organised).
    t_min = cfg.get("shock", {}).get("fit_t_min")
    if t_min is None:
        t_min = _ion_gyroperiod(cfg, layout, x0)
    t_min = float(t_min)

    fit_mask = np.isfinite(x_det) & (t_sim >= t_min)
    if fit_mask.sum() < 2:  # too few post-gyro detections -> fall back to all detected
        fit_mask = np.isfinite(x_det)
        t_min = 0.0
    if fit_mask.sum() < 2:
        raise ValueError(
            f"shock-trajectory fit needs >=2 detected frames, got {int(fit_mask.sum())}"
        )
    deg_used = int(min(deg, fit_mask.sum() - 1))  # need deg+1 points for a deg-poly
    coeffs = robust_polyfit(t_sim[fit_mask], x_det[fit_mask], deg=deg_used)
    return {"coeffs": coeffs, "t_sim": t_sim, "x_det": x_det,
            "fit_mask": fit_mask, "t_min": t_min, "deg": deg_used}


def _ion_gyroperiod(cfg: dict, layout, x_shock_0: float) -> float:
    """Ion gyroperiod T_ci = 2π|rqm_i|/|B| [1/ωpe] from the t=0 upstream field.

    Mirrors ``scripts/overview.py``: |B| is the median of the first frame ahead
    of the t=0 predicted front; |rqm_i| comes from the run's input deck.
    """
    import osh5io
    sim_dir = cfg["sim_dir"]
    t0 = cfg["times"][0]
    b = {n: osh5io.read_h5(diag_path(sim_dir, layout.field_quantity(n), t0))
         for n in ("b1", "b2", "b3")}
    x = axis_values(b["b1"], ax_idx=0)
    Bmag = np.sqrt(sum(np.asarray(b[n]) ** 2 for n in ("b1", "b2", "b3")))
    upstream = x > x_shock_0
    B_up = float(np.median(Bmag[upstream])) if upstream.any() else float(np.median(Bmag))
    rqm_i = abs(run_from_config(cfg).rqm)
    return 2.0 * np.pi * rqm_i / B_up


def resolve_shock_velocity(cfg: dict, t_sim: float, deg: int = 2,
                           species: str = "e", **fit_kw) -> dict:
    """Instantaneous shock velocity = d/dt of the fitted front trajectory at ``t_sim``.

    For the energy-partition / heating analyses the relevant frame boost is the
    *instantaneous* shock speed at the analyzed dump, which a curved fit captures
    (the front decelerates).  The single ``shock.v_shock`` in the analysis YAML is
    kept only as the detection seed and as the fallback if the fit fails; the
    degree can be set with ``shock.fit_degree`` in the config (default 2).

    Returns a dict: ``v_shock`` [c], ``x_shock_fit`` [c/ωpe] at ``t_sim``,
    ``source`` ("fit" or "config"), ``deg``, ``n_det``, plus the config value as
    ``v_shock_cfg``.
    """
    from shock import trajectory_at

    v_cfg = float(cfg["shock"]["v_shock"])
    deg = int(cfg.get("shock", {}).get("fit_degree", deg))
    try:
        traj = fit_shock_trajectory(cfg, deg=deg, species=species, **fit_kw)
        x_fit, v_fit = trajectory_at(traj["coeffs"], t_sim)
        return {"v_shock": float(v_fit), "x_shock_fit": float(x_fit),
                "source": "fit", "deg": traj["deg"],
                "n_det": int(traj["fit_mask"].sum()), "v_shock_cfg": v_cfg}
    except Exception as exc:  # detection/fit failed -> fall back to config seed
        return {"v_shock": v_cfg, "x_shock_fit": float("nan"),
                "source": f"config ({type(exc).__name__})", "deg": 0,
                "n_det": 0, "v_shock_cfg": v_cfg}


# ---------------------------------------------------------------------------
# StreakBuilder
# ---------------------------------------------------------------------------

class StreakBuilder:
    """Stack a list of osh5def.H5Data frames into a single H5Data with [time, x] axes.

    Time values, units, spatial axis, data name, and data units are all derived
    directly from the input frames — no extra arguments needed.

    The result can be passed straight to ``osh5vis.osplot()``.

    Parameters
    ----------
    frames : list of osh5def.H5Data
        One frame per time step, as returned by ``osh5io.read_h5()``.
        You can freely manipulate (slice, scale, mask…) each frame before
        passing it in; the spatial axis from the first frame is used as-is.

    Example
    -------
    ::

        frames = [osh5io.read_h5(f"{sim_dir}/MS/FLD/b3-savg/b3-savg-{t:06d}.h5")
                    for t in timestep_list]
        streak = StreakBuilder(frames).build()
        osh5vis.osplot(streak.T, cmap="RdBu_r")
    """

    def __init__(self, frames: List[osh5def.H5Data]):
        if not frames:
            raise ValueError("frames must be a non-empty list")
        self.frames = frames

    def build(self) -> osh5def.H5Data:
        """Return the assembled H5Data with axes [time, x], ready for osh5vis.osplot()."""
        f0 = self.frames[0]

        # Time axis — value and units come directly from each frame's run_attrs
        time_vals = [float(f.run_attrs["TIME"][0]) for f in self.frames]
        time_units = f0.run_attrs["TIME UNITS"]

        time_axis = osh5def.DataAxis(
            time_vals[0],
            time_vals[-1],
            len(time_vals),
            attrs={"NAME": "time", "LONG_NAME": "time", "UNITS": time_units},
        )

        # Spatial axis — reuse the axis object from the first frame
        space_axis = f0.axes[-1]

        data = np.stack([np.asarray(f) for f in self.frames], axis=0)

        return osh5def.H5Data(
            data,
            data_attrs=f0.data_attrs,
            run_attrs=f0.run_attrs,
            axes=[time_axis, space_axis],
        )
