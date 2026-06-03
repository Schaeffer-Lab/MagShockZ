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

import glob
import os
import re
import shlex
from typing import List, Optional

import astropy
import astropy.constants
import astropy.units
import numpy as np
import osiris_utils
import osh5def
import plasmapy
import plasmapy.formulary
import plasmapy.particles.particle_class
import yaml

# Fallback input-deck filename when neither the config nor the runme names one.
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

    # Backward-compatible alias used by existing notebooks.
    def _get_field(self, name: str):
        return self.field(name)

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

def phase_path(sim_dir: str, pha: str, sp: str, t: int) -> str:
    """Path to a phase-space dump, e.g. MS/PHA/p1x1/al/p1x1-al-000360.h5."""
    return f"{sim_dir}/MS/PHA/{pha}/{sp}/{pha}-{sp}-{t:06d}.h5"


def field_path(sim_dir: str, q: str, t: int) -> str:
    """Path to a time-averaged (savg) field dump, e.g. MS/FLD/b3-savg/b3-savg-000360.h5."""
    return f"{sim_dir}/MS/FLD/{q}-savg/{q}-savg-{t:06d}.h5"


def density_path(sim_dir: str, sp: str, t: int) -> str:
    """Path to a species charge-density dump (savg)."""
    return f"{sim_dir}/MS/DENSITY/{sp}/charge-savg/charge-savg-{sp}-{t:06d}.h5"


def axis_values(h5data, ax_idx: int) -> np.ndarray:
    """Return the coordinate values of axis ``ax_idx`` as a 1D linspace."""
    ax = h5data.axes[ax_idx]
    return np.linspace(ax.min, ax.max, ax.size)


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


def input_deck_path(cfg: dict) -> str:
    """Absolute path to the OSIRIS input deck for this run.

    The deck filename is intrinsic to the run, so it is taken from the runme's
    ``--inputfile_name`` rather than the analysis config.  An explicit
    ``input_deck`` in the config still overrides (handy for ad-hoc reruns), and
    :data:`DEFAULT_INPUT_DECK` is the last-resort fallback.
    """
    sim_dir = cfg["sim_dir"]
    name = cfg.get("input_deck")
    if name is None:
        try:
            name = load_runme(sim_dir).get("inputfile_name")
        except FileNotFoundError:
            name = None
    return os.path.join(sim_dir, name or DEFAULT_INPUT_DECK)


def run_from_config(cfg: dict, *, B0=None, Z=None, m_i=None) -> "MagShockZRun":
    """Build a :class:`MagShockZRun` from a parsed config.

    Centralises the norm-density unit attach and input-deck path resolution
    that every analysis script otherwise repeats.
    """
    norm_density = float(cfg["norm_density_cm3"]) * astropy.units.cm**-3
    return MagShockZRun(input_deck_path(cfg), norm_density=norm_density, B0=B0, Z=Z, m_i=m_i)


def find_runme(sim_dir: str) -> str:
    """Return the path to the run's ``runme*.sh`` script inside ``sim_dir``."""
    matches = sorted(glob.glob(os.path.join(sim_dir, "runme*.sh")))
    if not matches:
        raise FileNotFoundError(f"No runme*.sh found in {sim_dir}")
    return matches[0]


def parse_runme(path: str) -> dict:
    """Extract ``--key value`` pairs from a python-invocation shell script.

    Handles backslash line continuations and strips comments. Multi-value flags
    (e.g. ``--start_point 0 0.07 0``) are returned as lists; if a flag appears
    more than once the last occurrence wins.
    """
    with open(path) as f:
        text = f.read()
    text = re.sub(r"#[^\n]*", "", text)   # strip comments
    text = re.sub(r"\\\s*\n", " ", text)  # join continuation lines
    text = text.rstrip().rstrip("\\")     # drop a dangling trailing backslash
    tokens = shlex.split(text)

    # Advance past the 'python' call and script path to the first flag.
    i = 0
    while i < len(tokens) and not tokens[i].startswith("--"):
        i += 1

    args: dict = {}
    while i < len(tokens):
        tok = tokens[i]
        if tok.startswith("--"):
            key = tok[2:]
            vals = []
            j = i + 1
            while j < len(tokens) and not tokens[j].startswith("--"):
                vals.append(tokens[j])
                j += 1
            args[key] = vals[0] if len(vals) == 1 else vals
            i = j
        else:
            i += 1
    return args


def load_runme(sim_dir: str) -> dict:
    """Find and parse the run's ``runme*.sh`` into a ``--key -> value`` dict."""
    return parse_runme(find_runme(sim_dir))


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
