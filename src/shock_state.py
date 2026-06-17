"""Shared per-dump loader for OSIRIS shock analysis (single code path).

Several scripts (``compute_dimensionless_params.py``,
``compute_heating_decomposition.py``) need the same thing from one dump: the
electron/ion temperature profiles, electron density, the EM fields on a common
grid, the shock kinematics, and upstream/downstream region averages — all in
OSIRIS normalised units.  This module loads that once into a :class:`ShockState`
so the parameter conventions live in exactly one place.

This is I/O orchestration (it pulls in ``osh5io`` / ``analysis_utils``), so it
is *not* part of the dependency-light, CI-tested pure-function layer; it only
composes the pure functions in ``moments`` / ``temperature_anisotropy`` with
the loaders in ``analysis_utils``.  Keep new *physics* in those tested modules.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Sequence

import numpy as np
import osh5io

import analysis_utils
from analysis_utils import axis_values
import moments as mom_module
import temperature_anisotropy as ta

EMF_FIELDS = ("b1", "b2", "b3", "e1", "e2", "e3")


@dataclass
class ShockState:
    """All per-dump quantities on the phase-space spatial grid (OSIRIS units)."""

    sim_dir: str
    t_val: int
    t_sim: float
    x_pha: np.ndarray
    x_fld: np.ndarray
    dx: float

    pha_p1: Dict[str, object]
    pha_p2: Dict[str, object]
    pha_perp: Dict[str, list]             # per species: [p2x1, (p3x1)] when output

    n_e: np.ndarray                       # [n_0] on the phase grid
    T_par: Dict[str, np.ndarray]          # per species [m_e c^2]
    T_perp: Dict[str, np.ndarray]
    T_iso: Dict[str, np.ndarray]
    fields: Dict[str, np.ndarray]         # b1..e3 interpolated to the phase grid
    B2: np.ndarray                        # b1^2+b2^2+b3^2 on the phase grid

    species_rqm: Dict[str, float]
    abs_rqm_i: float
    Z_i: int
    ion: str

    v_shock: float
    x_shock: float
    up_ncells: int
    dn_ncells: int
    upstream_mask: np.ndarray
    downstream_mask: np.ndarray
    prim_up: Dict[str, float]
    prim_dn: Dict[str, float]

    sim: object = None
    spec: object = None
    L_box: float = field(default=float("nan"))
    v_shock_cfg: float = field(default=float("nan"))   # the single YAML shock.v_shock
    v_shock_source: str = "config"                     # "fit" or "config"


def _region_primitives(state_arrays: dict, mask: np.ndarray) -> dict:
    """Region-averaged primitives over a boolean spatial mask."""
    def m(arr):
        return float(np.nanmean(arr[mask])) if mask.any() else float("nan")
    f = state_arrays
    return {
        "n_e":      m(f["n_e"]),
        "T_e":      m(f["T_iso_e"]),
        "T_e_par":  m(f["T_par_e"]),
        "T_e_perp": m(f["T_perp_e"]),
        "T_i":      m(f["T_iso_i"]),
        "T_i_par":  m(f["T_par_i"]),
        "T_i_perp": m(f["T_perp_i"]),
        "B2":       m(f["B2"]),
        "b1":       m(f["b1"]),
        "b2":       m(f["b2"]),
        "b3":       m(f["b3"]),
        "u_bulk_i": m(f["u_bulk_i"]),
    }


def load_shock_state(cfg: dict, timestep_idx: int = -1,
                     species: Sequence[str] = ("al", "e"), ion: str = "al",
                     load_fields: Sequence[str] = EMF_FIELDS,
                     v_shock_from_fit: bool = True) -> ShockState:
    """Load one dump into a :class:`ShockState`.

    Parameters mirror ``compute_dimensionless_params.py``: averaging windows
    come from ``upstream_window_ncells`` / ``downstream_window_ncells`` in the
    config (default 200 cells each, measured from ``x_shock``).

    With ``v_shock_from_fit=True`` (default) the shock-frame boost velocity is
    the *instantaneous* derivative of the fitted front trajectory at this dump
    (``analysis_utils.resolve_shock_velocity``); the single config
    ``shock.v_shock`` is kept as the detection seed / fallback.  Pass
    ``v_shock_from_fit=False`` to use the config value directly (e.g. so
    ``compute_dimensionless_params`` reports M_A from the one YAML Mach number).
    """
    sim_dir = cfg["sim_dir"]
    t_val = cfg["times"][timestep_idx]
    sim = analysis_utils.run_from_config(cfg)
    spec = analysis_utils.RunSpec.from_sim_dir(sim_dir)
    layout = analysis_utils.detect_layout(sim_dir)  # for the savg field-name suffix

    Z_i = spec.charge_state(ion)
    abs_rqm_i = abs(sim.rqm)
    species_rqm = {sp: sim.rqm_of(sp) for sp in species}

    up_ncells = int(cfg.get("upstream_window_ncells", 200))
    dn_ncells = int(cfg.get("downstream_window_ncells", 200))

    pha_p1 = {sp: osh5io.read_h5(analysis_utils.diag_path(sim_dir, "p1x1", t_val, sp))
              for sp in species}
    pha_p2 = {sp: osh5io.read_h5(analysis_utils.diag_path(sim_dir, "p2x1", t_val, sp))
              for sp in species}
    # Transverse phase spaces for the full (multi-direction) thermal energy:
    # p2x1 is always present (it sets T_perp); p3x1 is added when it was output.
    pha_perp = {sp: [pha_p2[sp]] for sp in species}
    for sp in species:
        p3_path = analysis_utils.diag_path(sim_dir, "p3x1", t_val, sp)
        if os.path.exists(p3_path):
            pha_perp[sp].append(osh5io.read_h5(p3_path))
    fld = {name: osh5io.read_h5(analysis_utils.diag_path(sim_dir, layout.field_quantity(name), t_val))
           for name in load_fields}

    x_pha = axis_values(pha_p1[ion], ax_idx=1)
    x_fld = axis_values(fld[load_fields[0]], ax_idx=0)
    t_sim = float(pha_p1[ion].run_attrs["TIME"][0])

    dump = analysis_utils.resolve_dump_params(cfg, t_val, t_sim)
    v_shock = dump["v_shock"]
    x_shock = dump["x_shock"]
    v_shock_cfg = v_shock
    v_shock_source = "config"
    if v_shock_from_fit:
        vfit = analysis_utils.resolve_shock_velocity(cfg, t_sim)
        v_shock = vfit["v_shock"]
        v_shock_source = vfit["source"]

    T_par = {sp: ta.temperature_profile(pha_p1[sp], species_rqm[sp], "p1") for sp in species}
    T_perp = {sp: ta.temperature_profile(pha_p2[sp], species_rqm[sp], "p2") for sp in species}
    T_iso = {sp: (T_par[sp] + 2.0 * T_perp[sp]) / 3.0 for sp in species}

    n_e = np.abs(mom_module.moment(pha_p1["e"], axis="p1", order=0))
    u_bulk_i = mom_module.moment(pha_p1[ion], axis="p1", order=1)  # ion bulk velocity [c]

    # Interpolate all requested EM fields onto the phase-space grid.
    fields = {name: np.interp(x_pha, x_fld, np.asarray(fld[name])) for name in load_fields}
    B2 = fields["b1"] ** 2 + fields["b2"] ** 2 + fields["b3"] ** 2

    dx = float(x_pha[1] - x_pha[0])
    upstream_mask, downstream_mask = analysis_utils.window_masks(
        x_pha, x_shock, dx, up_ncells, dn_ncells
    )

    arrays = {
        "n_e": n_e, "B2": B2, "u_bulk_i": u_bulk_i,
        "T_iso_e": T_iso["e"], "T_par_e": T_par["e"], "T_perp_e": T_perp["e"],
        "T_iso_i": T_iso[ion], "T_par_i": T_par[ion], "T_perp_i": T_perp[ion],
        "b1": fields["b1"], "b2": fields["b2"], "b3": fields["b3"],
    }
    prim_up = _region_primitives(arrays, upstream_mask)
    prim_dn = _region_primitives(arrays, downstream_mask)

    L_box = float(fld[load_fields[0]].axes[0].max - fld[load_fields[0]].axes[0].min)

    return ShockState(
        sim_dir=sim_dir, t_val=t_val, t_sim=t_sim,
        x_pha=x_pha, x_fld=x_fld, dx=dx,
        pha_p1=pha_p1, pha_p2=pha_p2, pha_perp=pha_perp,
        n_e=n_e, T_par=T_par, T_perp=T_perp, T_iso=T_iso,
        fields=fields, B2=B2,
        species_rqm=species_rqm, abs_rqm_i=abs_rqm_i, Z_i=Z_i, ion=ion,
        v_shock=v_shock, x_shock=x_shock,
        up_ncells=up_ncells, dn_ncells=dn_ncells,
        upstream_mask=upstream_mask, downstream_mask=downstream_mask,
        prim_up=prim_up, prim_dn=prim_dn,
        sim=sim, spec=spec, L_box=L_box,
        v_shock_cfg=v_shock_cfg, v_shock_source=v_shock_source,
    )
