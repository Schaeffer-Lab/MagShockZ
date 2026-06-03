"""analysis_utils.py — lightweight OSIRIS analysis helpers.

Two public objects:

  MagShockZRun
      Thin wrapper around an osiris_utils.Simulation that exposes field access
      and physical-unit conversions (frequencies, lengths, speeds, B-field
      normalisation).  No plotting, no indexing, no moment machinery.

  StreakBuilder
      Standalone helper that stacks a sequence of 1D spatial profiles into an
      osh5def.H5Data object with explicit [time, space] axes so the result can
      be passed directly to osh5vis.osplot().
"""

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

    @property
    def rqm(self):
        """Ion rqm = m/q (mass-per-charge) relative to the electron (from input deck)."""
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
# Config helpers
# ---------------------------------------------------------------------------

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
