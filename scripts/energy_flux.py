"""energy_flux — shock-frame energy-flux conservation check for an OSIRIS dump.

Science
-------
Across a steady shock the conserved quantity is the energy *flux*, not the energy
*density* (which jumps because the flow compresses and slows).  This is the
rigorous companion to ``energy_partition.py``: it computes the x-directed energy
flux in the shock rest frame and compares it upstream vs. downstream — a
quasi-steady shock gives F_up ≈ F_dn.

Channels (all in n_0 m_e c², i.e. flux/c; OSIRIS normalised units):

    bulk      U·½ρ|U|²       advected bulk kinetic energy   (per species, summed)
    internal  U·ε            advected internal/thermal energy
    pressure  U·P_xx         shock-normal pressure work
    poynting  E2·B3 − E3·B2  electromagnetic energy flux

U = v − v_shock uses the fitted instantaneous shock velocity (from shock_state).
All four kinetic+EM formulas are owned by the pure, unit-tested module
``src/energy_flux.py`` (``species_energy_flux`` and ``poynting_flux``).  The
collisionless heat flux q_x (a 3rd-order moment unavailable from the marginal
pᵢx₁ phase spaces) is neglected — documented in src/energy_flux.py.

Validation
----------
This file is orchestration + plotting only; it contains no physics formulas.
To validate a number, read the named function in ``src/energy_flux.py`` and its
test in ``tests/test_energy_flux.py``.  The computed result is also saved to an
inspectable ``.npz`` whose schema is the ``EnergyFluxResult`` dataclass below.

Usage
-----
    python scripts/energy_flux.py --config config/<run>.yaml \\
        [--timestep-idx -1] [--no-plot] [--output ...] [--output-dir ...]
"""

import argparse
import os
import sys
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

import analysis_utils
import plot_style
import shock_state
import energy_flux as ef

# Flux channels, in plot order.
_CHANNELS = ("bulk", "internal", "pressure", "poynting", "total")


# ---------------------------------------------------------------------------
# Output schema (this dataclass IS the .npz schema; see analysis_utils.save_result)
# ---------------------------------------------------------------------------

@dataclass
class EnergyFluxResult:
    # --- shock-frame energy-flux profiles on the phase-space grid [n_0 m_e c^2] ---
    x_axis: np.ndarray            # [c/wpe]
    F_bulk: np.ndarray            # advected bulk KE      U·½ρ|U|²
    F_internal: np.ndarray        # advected internal energy  U·ε
    F_pressure: np.ndarray        # shock-normal pressure work U·P_xx
    F_poynting: np.ndarray        # electromagnetic (Poynting) flux
    F_total: np.ndarray           # sum of the four channels
    # --- shock kinematics / window geometry / provenance ---
    t_val: int                    # dump index (file suffix)
    t_sim: float                  # simulation time [1/wpe]
    x_shock: float                # shock-front position [c/wpe]
    v_shock: float                # shock-frame boost velocity [c]
    v_shock_cfg: float            # config detection seed [c]
    v_shock_source: str           # "fit" or "config"
    up_ncells: int                # upstream window width [cells from x_shock]
    dn_ncells: int                # downstream window width [cells from x_shock]
    dx: float                     # cell size [c/wpe]
    config_path: str              # absolute path of the analysis config used
    # --- region averages: mean flux per channel [n_0 m_e c^2] ---
    upstream: dict                # {bulk, internal, pressure, poynting, total}
    downstream: dict


# ---------------------------------------------------------------------------
# Compute (orchestration only — every formula lives in src/energy_flux.py)
# ---------------------------------------------------------------------------

def compute(cfg: dict, timestep_idx: int = -1, config_path: str = "") -> EnergyFluxResult:
    """Load one dump via shock_state, sum the flux channels, return the result."""
    print("Loading HDF5 files...")
    st = shock_state.load_shock_state(cfg, timestep_idx)
    ion = st.ion

    # Kinetic energy-flux channels, summed over species (electrons + ion).
    F_bulk = np.zeros(st.x_pha.size)
    F_internal = np.zeros(st.x_pha.size)
    F_pressure = np.zeros(st.x_pha.size)
    for sp in ("e", ion):
        fb, fi, fp = ef.species_energy_flux(
            st.pha_p1[sp], st.species_rqm[sp], st.v_shock,
            perp_phase_spaces=st.pha_perp[sp])
        F_bulk += fb
        F_internal += fi
        F_pressure += fp

    # Electromagnetic (Poynting) flux on the phase grid, boosted to the shock
    # frame so it shares the frame of the kinetic channels (fields already there).
    F_poynting = ef.poynting_flux(
        st.fields["e2"], st.fields["e3"], st.fields["b2"], st.fields["b3"],
        v_shock=st.v_shock)
    F_total = F_bulk + F_internal + F_pressure + F_poynting

    # Upstream vs. downstream window averages (masks from shock_state).
    channels = {"bulk": F_bulk, "internal": F_internal, "pressure": F_pressure,
                "poynting": F_poynting, "total": F_total}
    region_mean = lambda arr, mask: float(np.nanmean(arr[mask])) if mask.any() else float("nan")
    up = {k: region_mean(v, st.upstream_mask) for k, v in channels.items()}
    dn = {k: region_mean(v, st.downstream_mask) for k, v in channels.items()}

    return EnergyFluxResult(
        x_axis=st.x_pha,
        F_bulk=F_bulk, F_internal=F_internal, F_pressure=F_pressure,
        F_poynting=F_poynting, F_total=F_total,
        t_val=int(st.t_val), t_sim=float(st.t_sim),
        x_shock=float(st.x_shock), v_shock=float(st.v_shock),
        v_shock_cfg=float(st.v_shock_cfg), v_shock_source=str(st.v_shock_source),
        up_ncells=int(st.up_ncells), dn_ncells=int(st.dn_ncells), dx=float(st.dx),
        config_path=config_path, upstream=up, downstream=dn,
    )


def _print_summary(r: EnergyFluxResult) -> None:
    """Console table of the flux conservation check (no computation)."""
    print(f"t_sim   : {r.t_sim:.1f} [ωpe⁻¹]   x_shock : {r.x_shock:.1f} [c/ωpe]   "
          f"v_shock : {r.v_shock:.4f} [c]  ({r.v_shock_source}; YAML seed {r.v_shock_cfg:.4f} c)")
    sep = "-" * 64
    print(f"\n{sep}")
    print("  Shock-frame energy flux  [n₀ mₑ c²]   (conserved if F_up ≈ F_dn)")
    print(sep)
    print(f"  {'channel':<10} {'upstream':>13} {'downstream':>13} {'dn−up':>13}")
    for k in _CHANNELS:
        print(f"  {k:<10} {r.upstream[k]:>13.3e} {r.downstream[k]:>13.3e} "
              f"{r.downstream[k]-r.upstream[k]:>13.3e}")
    print(sep)
    up_tot, dn_tot = r.upstream["total"], r.downstream["total"]
    ratio = dn_tot / up_tot if up_tot else float("nan")
    noncons = (dn_tot - up_tot) / abs(up_tot) if up_tot else float("nan")
    print(f"  total flux  dn/up = {ratio:.3f}   "
          f"(non-conservation {100*noncons:+.1f}% of |F_up|)")


# ---------------------------------------------------------------------------
# Plot (matplotlib only — reads the result, draws nothing new)
# ---------------------------------------------------------------------------

_PLOT_CHANNELS = [
    ("F_bulk", "Bulk KE  U·½ρ|U|²", "C0"),
    ("F_internal", "Internal  U·ε", "C1"),
    ("F_pressure", "Pressure work  U·P_xx", "C2"),
    ("F_poynting", "Poynting  E×B", "C4"),
]


def _plot_profiles(r: EnergyFluxResult, ax: plt.Axes, disp) -> None:
    x = r.x_axis
    xd = disp.x(x)
    xs = r.x_shock
    x_up = xs + r.up_ncells * r.dx
    x_dn = xs - r.dn_ncells * r.dx

    for attr, label, color in _PLOT_CHANNELS:
        ax.plot(xd, getattr(r, attr), color=color, lw=1.2, label=label)
    ax.plot(xd, r.F_total, color="k", lw=2.0, label="Total")
    ax.axhline(0.0, color="0.6", lw=0.8)
    ax.axvline(disp.x(xs), color="k", ls="--", lw=1, label=f"shock {xs:.0f}")
    ax.axvspan(disp.x(xs), disp.x(x_up), color="C0", alpha=0.08, label="upstream window")
    ax.axvspan(disp.x(x_dn), disp.x(xs), color="C3", alpha=0.08, label="downstream window")

    ax.set_xlabel(disp.xlabel())
    ax.set_ylabel("Energy flux  F / c  [n₀ mₑ c²]")
    ax.set_title(f"Shock-frame energy flux, dump {r.t_val}  {disp.time_title(r.t_sim)}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xlim(disp.x(x_dn - 50), disp.x(x_up + 50))
    win = (x >= x_dn - 50) & (x <= x_up + 50)
    if win.any():
        lo, hi = np.nanmin(r.F_total[win]), np.nanmax(r.F_total[win])
        pad = 0.5 * (hi - lo + 1e-12)
        ax.set_ylim(min(lo, np.nanmin(r.F_bulk[win])) - pad, hi + pad)


def _plot_bars(r: EnergyFluxResult, ax: plt.Axes) -> None:
    labels = ["Bulk", "Internal", "Pressure", "Poynting", "Total"]
    up = [r.upstream[k] for k in _CHANNELS]
    dn = [r.downstream[k] for k in _CHANNELS]

    xpos = np.arange(len(_CHANNELS))
    w = 0.38
    ax.bar(xpos - w / 2, up, w, label="Upstream")
    ax.bar(xpos + w / 2, dn, w, label="Downstream")
    ax.axhline(0.0, color="0.6", lw=0.8)
    ax.axvline(len(_CHANNELS) - 1.5, color="0.7", lw=1, ls=":")  # set off Total

    ratio = dn[-1] / up[-1] if up[-1] else float("nan")
    y = min(up[-1], dn[-1])
    ax.annotate(f"dn/up = {ratio:.2f}\n({100*(dn[-1]-up[-1])/abs(up[-1]):+.0f}%)",
                xy=(xpos[-1], y), xytext=(0, -2), textcoords="offset points",
                ha="center", va="top", fontsize=9)

    ax.set_xticks(xpos)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean energy flux  F / c  [n₀ mₑ c²]")
    ax.set_title(f"Flux conservation  (v_shock={r.v_shock:.4f}c, {r.v_shock_source})")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)


def plot(r: EnergyFluxResult, output_dir: str, disp) -> str:
    """Render the two-panel figure (flux profiles + conservation bars), save a .png."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    _plot_profiles(r, axes[0], disp)
    _plot_bars(r, axes[1])
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"energy_flux_t{r.t_val:06d}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Shock-frame energy-flux check (compute + plot).")
    parser.add_argument("--config", required=True, help="Path to analysis YAML config.")
    parser.add_argument("--timestep-idx", type=int, default=-1, dest="timestep_idx",
                        help="Index into config times list (default: -1, last dump).")
    parser.add_argument("--no-plot", action="store_true", dest="no_plot",
                        help="Compute and save the .npz only; skip the figure.")
    parser.add_argument("--output", default=None,
                        help="Output .npz path (default results/<run>/energy_flux_t{t:06d}.npz).")
    parser.add_argument("--output-dir", default=None, dest="output_dir",
                        help="Directory for the figure (default: alongside the .npz).")
    plot_style.add_publication_arg(parser)
    plot_style.add_units_arg(parser)
    args = parser.parse_args()
    plot_style.apply(args.publication)

    cfg = analysis_utils.load_config(args.config)
    print(f"Config  : {args.config}")
    print(f"sim_dir : {cfg['sim_dir']}")

    result = compute(cfg, args.timestep_idx, config_path=os.path.abspath(args.config))
    _print_summary(result)

    out_path = analysis_utils.default_output_path(args.output, cfg["sim_dir"], "energy_flux", result.t_val)
    analysis_utils.save_result(result, out_path)
    print(f"\nSaved → {out_path}")

    if not args.no_plot:
        disp = plot_style.build_units_from_args(args, cfg)
        out_dir = args.output_dir or os.path.dirname(os.path.abspath(out_path))
        fig_path = plot(result, out_dir, disp)
        print(f"Saved → {fig_path}")


if __name__ == "__main__":
    main()
