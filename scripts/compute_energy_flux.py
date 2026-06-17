"""Shock-frame energy-FLUX conservation check for an OSIRIS dump.

Across a steady shock the conserved quantity is the energy *flux*, not the
energy *density* (which jumps because the flow compresses and slows).  This is
the rigorous companion to ``compute_energy_partition.py``: it computes the
x-directed energy flux in the shock rest frame and compares it upstream vs.
downstream — a quasi-steady shock should give F_up ≈ F_dn.

Channels (all in n_0 m_e c², i.e. flux/c; OSIRIS normalised units):
  bulk      U·½ρ|U|²    advected bulk kinetic energy   (per species, summed)
  internal  U·ε         advected internal/thermal energy
  pressure  U·P_xx      shock-normal pressure work
  poynting  E2·B3−E3·B2 electromagnetic energy flux

U = v − v_shock uses the fitted instantaneous shock velocity (shock_state).
The collisionless heat flux q_x (a 3rd-order / cross moment unavailable from the
marginal pᵢx₁ phase spaces) is neglected — see src/energy_flux.py.

Usage
-----
    conda activate analysis
    python scripts/compute_energy_flux.py \\
        --config config/perlmutter_1.3.1d.yaml [--timestep-idx -1] [--output ...]
"""

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

import analysis_utils
import shock_state
import energy_flux as ef


def main():
    parser = argparse.ArgumentParser(description="Shock-frame energy-flux conservation check.")
    parser.add_argument("--config", required=True, help="Path to analysis YAML config.")
    parser.add_argument("--timestep-idx", type=int, default=-1, dest="timestep_idx",
                        help="Index into config times list (default: -1, last dump).")
    parser.add_argument("--output", default=None,
                        help="Output .npz (default results/<run>/energy_flux_t{t:06d}.npz).")
    args = parser.parse_args()

    cfg = analysis_utils.load_config(args.config)
    print(f"Config  : {args.config}")
    print(f"sim_dir : {cfg['sim_dir']}")
    print("Loading HDF5 files...")
    st = shock_state.load_shock_state(cfg, args.timestep_idx)
    ion = st.ion

    print(f"t_sim   : {st.t_sim:.1f} [ωpe⁻¹]   x_shock : {st.x_shock:.1f} [c/ωpe]   "
          f"v_shock : {st.v_shock:.4f} [c]  ({st.v_shock_source}; YAML seed {st.v_shock_cfg:.4f} c)")

    # ------------------------------------------------------------------
    # Kinetic energy-flux channels, summed over species (electrons + ion).
    # ------------------------------------------------------------------
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

    # Electromagnetic (Poynting) flux on the phase grid (fields already there).
    F_poynting = ef.poynting_flux(
        st.fields["e2"], st.fields["e3"], st.fields["b2"], st.fields["b3"])

    F_total = F_bulk + F_internal + F_pressure + F_poynting

    # ------------------------------------------------------------------
    # Upstream vs downstream conservation check (window averages).
    # ------------------------------------------------------------------
    def region_mean(arr, mask):
        return float(np.nanmean(arr[mask])) if mask.any() else float("nan")

    channels = {"bulk": F_bulk, "internal": F_internal,
                "pressure": F_pressure, "poynting": F_poynting, "total": F_total}
    up = {k: region_mean(v, st.upstream_mask) for k, v in channels.items()}
    dn = {k: region_mean(v, st.downstream_mask) for k, v in channels.items()}

    sep = "-" * 64
    print(f"\n{sep}")
    print("  Shock-frame energy flux  [n₀ mₑ c²]   (conserved if F_up ≈ F_dn)")
    print(sep)
    print(f"  {'channel':<10} {'upstream':>13} {'downstream':>13} {'dn−up':>13}")
    for k in ("bulk", "internal", "pressure", "poynting", "total"):
        print(f"  {k:<10} {up[k]:>13.3e} {dn[k]:>13.3e} {dn[k]-up[k]:>13.3e}")
    ratio = dn["total"] / up["total"] if up["total"] else float("nan")
    nonconservation = (dn["total"] - up["total"]) / abs(up["total"]) if up["total"] else float("nan")
    print(sep)
    print(f"  total flux  dn/up = {ratio:.3f}   "
          f"(non-conservation {100*nonconservation:+.1f}% of |F_up|)")

    out_path = analysis_utils.default_output_path(
        args.output, st.sim_dir, "energy_flux", st.t_val)
    np.savez(
        out_path,
        x_axis=st.x_pha,
        F_bulk=F_bulk, F_internal=F_internal, F_pressure=F_pressure,
        F_poynting=F_poynting, F_total=F_total,
        **{f"upstream_{k}": np.asarray(up[k]) for k in channels},
        **{f"downstream_{k}": np.asarray(dn[k]) for k in channels},
        x_shock=np.asarray(st.x_shock),
        v_shock=np.asarray(st.v_shock),
        v_shock_cfg=np.asarray(st.v_shock_cfg),
        v_shock_source=np.asarray(st.v_shock_source),
        up_ncells=np.asarray(st.up_ncells), dn_ncells=np.asarray(st.dn_ncells),
        dx=np.asarray(st.dx),
        t_val=np.asarray(st.t_val), t_sim=np.asarray(st.t_sim),
        config_path=np.asarray(os.path.abspath(args.config)),
    )
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
