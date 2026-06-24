"""Per-dump 1D profile frames + streak assembly for OSIRIS shock runs.

These builders turn a single OSIRIS dump into a 1D shock-normal profile (an
``H5Data`` carrying its spatial axis + run_attrs) for |B|, electron density and a
species temperature, and stack a list of such frames into a ``[time, x]`` streak.
They are I/O orchestration (they pull in ``osh5io`` / ``osh5def`` /
``analysis_utils``), so this module is NOT part of the dependency-light, CI-tested
pure-function layer — it only composes the loaders in ``analysis_utils`` with the
pure ``temperature_anisotropy`` moment.  Shared single source of truth for both
``scripts/overview.py`` (streak figures) and ``scripts/tune_shock.py`` (the
interactive trajectory tuner) so the front overlay is drawn against identical data.
"""

import numpy as np
import osh5def
import osh5io

import temperature_anisotropy as ta
from analysis_utils import StreakBuilder, axis_values, diag_path, transverse_profile


# ---------------------------------------------------------------------------
# Per-dump 1D profile frames (H5Data, so StreakBuilder can stack them)
#
# Fields and density are full spatial maps (1D, or 2D for a 2D run); each is
# reduced to a 1D profile along the shock-normal axis by transverse_profile
# (a no-op in 1D).  Phase spaces already carry a single spatial axis, so the
# momentum moment collapses them straight to a 1D profile.
# ---------------------------------------------------------------------------

def bmag_frame(sim_dir: str, t: int, layout, hw: float):
    """|B| = sqrt(b1^2 + b2^2 + b3^2), reduced to a 1D profile [B_0]."""
    b = {q: osh5io.read_h5(diag_path(sim_dir, layout.field_quantity(q), t))
         for q in ("b1", "b2", "b3")}
    bmag = np.sqrt(b["b1"] ** 2 + b["b2"] ** 2 + b["b3"] ** 2)  # stays H5Data
    # Keep the propagated UNITS (osh5def carries the field unit m_e c ω_p/e through
    # the sqrt) — do NOT relabel as B_0: the data is in OSIRIS field-normalisation
    # units, not normalised to an upstream B_0.  LONG_NAME is bare TeX (osh5vis wraps).
    bmag.data_attrs = dict(bmag.data_attrs, NAME="|B|", LONG_NAME=r"|B|")
    return transverse_profile(bmag, layout.normal_axis, hw)


def density_frame(sim_dir: str, sp: str, t: int, layout, hw: float):
    """Number density n = |charge|, reduced to a 1D profile [n_0].

    The OSIRIS ``charge`` diagnostic is q·n in normalised charge-density units; the
    UNITS are relabelled to n_0 here because |charge| = |q|·n/n_0 equals n/n_0 only
    for a singly-charged species.  overview.py calls this only for electrons (q=−1),
    where that holds; do NOT use it for multiply-charged ions without dividing by Z.
    """
    ch = osh5io.read_h5(diag_path(sim_dir, layout.charge_quantity, t, sp))
    n = np.abs(ch)
    n.data_attrs = dict(n.data_attrs, NAME=f"n_{sp}", LONG_NAME=fr"n_\mathrm{{{sp}}}", UNITS="n_0")
    return transverse_profile(n, layout.normal_axis, hw)


def temperature_frame(sim_dir: str, sp: str, t: int, rqm: float, layout, axis: str = "p1"):
    """Parallel temperature T = |rqm| * <(p - <p>)^2> as an H5Data on the phase grid.

    The moment collapses f(p, x) -> T(x); the result is re-wrapped as an H5Data
    carrying the phase-space spatial axis and run_attrs (for the TIME StreakBuilder
    needs) so it can be stacked exactly like the field/density frames.  The
    phase-space name (p1x1 vs p1x2) comes from the run layout.
    """
    ps = osh5io.read_h5(diag_path(sim_dir, layout.pha_name(int(axis[1:])), t, sp))
    T = np.asarray(ta.temperature_profile(ps, rqm, axis))
    x_axis = next(a for a in ps.axes if a.name != axis)  # the spatial axis
    return osh5def.H5Data(
        T,
        data_attrs={"NAME": f"T_{sp}", "LONG_NAME": fr"T_{{{sp}}}", "UNITS": "m_e c^2"},
        run_attrs=ps.run_attrs,
        axes=[x_axis],
    )


# ---------------------------------------------------------------------------
# Streak assembly
# ---------------------------------------------------------------------------

def assemble_streak(frames):
    """StreakBuilder a list of 1D H5Data frames -> (streak[time, x] H5Data, Z, time[], x[]).

    The H5Data ``streak`` (axes [time, x], carrying NAME/LONG_NAME/UNITS) is what
    osh5vis plots; Z/time/x are the plain-numpy views the .npz and detection use.
    """
    streak = StreakBuilder(frames).build()
    Z = np.asarray(streak)
    time = axis_values(streak, 0)
    x = axis_values(streak, 1)
    return streak, Z, time, x
