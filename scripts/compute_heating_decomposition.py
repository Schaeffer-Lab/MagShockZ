"""Decompose shock heating into adiabatic vs. anomalous, and attribute it.

This is the project's headline physics deliverable: for one OSIRIS dump it
assembles the full picture of *how* the shock heats —

  1. Rankine--Hugoniot baseline  (src/rankine_hugoniot.py)
       theta_Bn, the perpendicular-MHD compression ratio, and the adiabatic
       downstream temperature.  The excess of the measured downstream T over
       that is the **anomalous (collisionless) heating**, split per species.
  2. Cross-shock potential        (src/cross_shock_potential.py)
       e*Δphi across the front and its size vs. the ion ram energy
       (the electron-heating / ion-reflection driver).
  3. Reflected ions               (src/reflected_ions.py)
       reflected fraction and reflected-ion energy in the foot.
  4. Field-particle correlation   (src/field_particle_correlation.py)
       velocity-space signature + net field work rate per species.
  5. Energy budget                (src/energy_partition.py)
       ram / thermal_e / thermal_i / B / E region averages.

Everything is loaded once through src/shock_state.py and saved to one .npz for
plot_heating_decomposition.py.  OSIRIS normalised units throughout.

Usage
-----
    conda activate analysis
    python scripts/compute_heating_decomposition.py \\
        --config config/perlmutter_1.3.1d.yaml [--timestep-idx -1] [--output ...]
"""

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

import analysis_utils
from analysis_utils import axis_values
import shock_state
import rankine_hugoniot as rh
import cross_shock_potential as csp
import reflected_ions as ri
import field_particle_correlation as fpc
import energy_partition as ep


def _phase_f_u(pha):
    """Return (f, u) for a p1x1 phase space: f has shape (len(u), len(x))."""
    return np.asarray(pha, dtype=float), axis_values(pha, ax_idx=0)


def _velocity_signature(pha, e1_on_x, rqm, layer_mask):
    """FPC velocity-space spectrum (averaged over the shock layer) and net rate.

    Returns (u, C_E_spectrum(u), net_rate(x), total_rate_in_layer).
    """
    f, u = _phase_f_u(pha)
    c_e = fpc.energy_transfer_rate(f, u, e1_on_x, rqm)
    net_rate = fpc.velocity_integrated_rate(c_e, u)           # vs x
    # Velocity spectrum: average C_E over the spatial shock layer.
    spectrum = np.nanmean(c_e[:, layer_mask], axis=1) if layer_mask.any() \
        else np.full(u.size, np.nan)
    total_in_layer = float(np.nansum(net_rate[layer_mask]))
    return u, spectrum, net_rate, total_in_layer


def main():
    parser = argparse.ArgumentParser(description="Shock heating decomposition + attribution.")
    parser.add_argument("--config", required=True, help="Path to analysis YAML config.")
    parser.add_argument("--timestep-idx", type=int, default=-1, dest="timestep_idx",
                        help="Index into config times list (default: -1, last dump).")
    parser.add_argument("--output", default=None,
                        help="Output .npz (default results/<run>/heating_decomposition_t{t:06d}.npz).")
    args = parser.parse_args()

    cfg = analysis_utils.load_config(args.config)
    print(f"Config  : {args.config}")
    print(f"sim_dir : {cfg['sim_dir']}")
    print("Loading HDF5 files...")
    st = shock_state.load_shock_state(cfg, args.timestep_idx)
    ion = st.ion

    print(f"t_sim   : {st.t_sim:.1f} [ωpe⁻¹]   x_shock : {st.x_shock:.1f} [c/ωpe]   "
          f"v_shock : {st.v_shock:.4f} [c]  ({st.v_shock_source}; YAML seed {st.v_shock_cfg:.4f} c)")

    up, dn = st.prim_up, st.prim_dn

    # ------------------------------------------------------------------
    # 1. Rankine--Hugoniot baseline
    # ------------------------------------------------------------------
    theta_bn = rh.shock_normal_angle(up["b1"], up["b2"], up["b3"])
    v_inflow = st.v_shock - up["u_bulk_i"]          # shock-frame inflow speed [c]
    jump = rh.solve_jump(up["n_e"], up["T_e"], up["T_i"], up["B2"],
                         st.abs_rqm_i, v_inflow)
    T_factor = jump["T_factor"]
    r_measured = dn["n_e"] / up["n_e"] if up["n_e"] > 0 else float("nan")

    # MHD is single-fluid: the jump predicts ONE total post-shock temperature.
    # "Anomalous" heating is the measured TOTAL T over the RH total prediction;
    # how that heat splits between electrons and ions is the collisionless
    # partition we *measure* (it is not something the single-fluid jump knows).
    T_tot_up = up["T_e"] + up["T_i"]
    T_tot_dn = dn["T_e"] + dn["T_i"]
    heat_tot = rh.anomalous_heating(T_tot_dn, T_tot_up, T_factor)

    dT_e = dn["T_e"] - up["T_e"]
    dT_i = dn["T_i"] - up["T_i"]
    dT_tot = dT_e + dT_i
    frac_heat_e = dT_e / dT_tot if dT_tot != 0 else float("nan")
    frac_heat_i = dT_i / dT_tot if dT_tot != 0 else float("nan")

    # ------------------------------------------------------------------
    # 2. Cross-shock potential (from the normal field e1 on the phase grid)
    # ------------------------------------------------------------------
    e1 = st.fields["e1"]
    e_dphi = csp.potential_jump(e1, st.x_pha, st.upstream_mask, st.downstream_mask)
    refl_param = csp.reflection_parameter(e_dphi, st.abs_rqm_i, st.v_shock)
    e_phi_profile = csp.potential_profile(e1, st.x_pha)

    # ------------------------------------------------------------------
    # 3. Reflected ions
    # ------------------------------------------------------------------
    incoming_sign = ri.infer_incoming_sign(up["u_bulk_i"], st.v_shock)
    refl_frac = ri.reflected_fraction(st.pha_p1[ion], st.v_shock, incoming_sign)
    refl_energy = ri.reflected_energy_density(
        st.pha_p1[ion], st.species_rqm[ion], st.v_shock, incoming_sign)
    # Foot: a narrow window just upstream of the shock where reflected ions live.
    foot_ncells = min(st.up_ncells, 60)
    foot_mask = (st.x_pha > st.x_shock) & (st.x_pha <= st.x_shock + foot_ncells * st.dx)
    refl_frac_foot = float(np.nanmean(refl_frac[foot_mask])) if foot_mask.any() else float("nan")
    refl_frac_peak = float(np.nanmax(refl_frac[st.upstream_mask])) if st.upstream_mask.any() else float("nan")

    # ------------------------------------------------------------------
    # 4. Field-particle correlation (velocity-space heating signature)
    # ------------------------------------------------------------------
    layer_mask = st.downstream_mask | foot_mask     # ramp + foot + near downstream
    u_e, fpc_spec_e, fpc_net_e, fpc_tot_e = _velocity_signature(
        st.pha_p1["e"], e1, st.species_rqm["e"], layer_mask)
    u_i, fpc_spec_i, fpc_net_i, fpc_tot_i = _velocity_signature(
        st.pha_p1[ion], e1, st.species_rqm[ion], layer_mask)

    # ------------------------------------------------------------------
    # 5. Energy budget (ram / thermal_e / thermal_i / B / E)
    # ------------------------------------------------------------------
    u_ram = {}
    u_th = {}
    for sp in ("e", ion):
        u_ram[sp], u_th[sp] = ep.species_energy_profiles(
            st.pha_p1[sp], st.species_rqm[sp], st.v_shock,
            perp_phase_spaces=st.pha_perp[sp])
    u_ram_tot = u_ram["e"] + u_ram[ion]
    u_B = 0.5 * st.B2
    u_E = 0.5 * (st.fields["e1"]**2 + st.fields["e2"]**2 + st.fields["e3"]**2)

    def region_mean(arr, mask):
        return float(np.nanmean(arr[mask])) if mask.any() else float("nan")

    budget = {}
    for side, mask in (("upstream", st.upstream_mask), ("downstream", st.downstream_mask)):
        budget[side] = {
            "ram":       region_mean(u_ram_tot, mask),
            "thermal_e": region_mean(u_th["e"], mask),
            "thermal_i": region_mean(u_th[ion], mask),
            "B_field":   region_mean(u_B, mask),
            "E_field":   region_mean(u_E, mask),
        }
    ram_in = budget["upstream"]["ram"]

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    sep = "-" * 72
    print("\n" + "=" * 72)
    print("  Shock heating decomposition")
    print("=" * 72)
    print(f"  theta_Bn        = {np.degrees(theta_bn):6.1f} deg  "
          f"({'quasi-perp' if rh.is_quasi_perpendicular(theta_bn) else 'quasi-par'})")
    print(f"  compression r   = {r_measured:6.3f} (measured)   {jump['r']:6.3f} (RH)")
    print(f"  M_A, M_s        = {jump['mach_a']:6.2f}, {jump['mach_s']:6.2f}")
    print(f"  RH T-jump       = {T_factor:6.3f}  (predicted total T2/T1)")

    print(f"\n{sep}")
    print(f"  Total heating vs MHD baseline   (T = T_e + T_i)")
    print(sep)
    print(f"  {'T_tot_up':>11} {'T_tot_dn':>11} {'T_RH':>11} {'T_anom':>11} {'anom %':>8}")
    print(f"  {T_tot_up:>11.3e} {T_tot_dn:>11.3e} {heat_tot['adiabatic']:>11.3e} "
          f"{heat_tot['anomalous']:>11.3e} {100 * heat_tot['anomalous_frac']:>7.1f}%")
    print(f"\n  Heating partition (measured): "
          f"electrons {100 * frac_heat_e:5.1f}%   ions {100 * frac_heat_i:5.1f}%")
    print(f"    ΔT_e = {dT_e:.3e}   ΔT_i = {dT_i:.3e}   "
          f"T_e/T_i downstream = {dn['T_e'] / dn['T_i']:.2f}")

    print(f"\n  Cross-shock potential  e*Δphi = {e_dphi:.3e} [m_e c²]   "
          f"e*Δphi / (½ m_i v_sh²) = {refl_param:.3f}")
    print(f"  Reflected ions         foot fraction = {refl_frac_foot:.3f}   "
          f"peak = {refl_frac_peak:.3f}")
    print(f"  FPC net field work in layer   electrons = {fpc_tot_e:+.3e}   "
          f"ions = {fpc_tot_i:+.3e}")

    print(f"\n{sep}")
    print(f"  Energy budget (downstream channel / upstream ram-in)")
    print(sep)
    for k in ("ram", "thermal_e", "thermal_i", "B_field", "E_field"):
        frac = budget["downstream"][k] / ram_in if ram_in else float("nan")
        print(f"  {k:<12} up={budget['upstream'][k]:>11.3e}  "
              f"dn={budget['downstream'][k]:>11.3e}  dn/ram_in={100*frac:>6.1f}%")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    out_path = analysis_utils.default_output_path(
        args.output, st.sim_dir, "heating_decomposition", st.t_val)

    save = {
        # profiles on phase-space grid
        "x_axis": st.x_pha,
        "n_e": st.n_e, "B2": st.B2,
        "T_iso_e": st.T_iso["e"], "T_iso_i": st.T_iso[ion],
        "e_phi_profile": e_phi_profile,
        "reflected_fraction": refl_frac,
        "reflected_energy_density": refl_energy,
        "u_ram_total": u_ram_tot, "u_th_e": u_th["e"], "u_th_i": u_th[ion],
        "u_B": u_B, "u_E": u_E,
        # FPC velocity-space signatures
        "fpc_u_e": u_e, "fpc_spectrum_e": fpc_spec_e, "fpc_net_rate_e": fpc_net_e,
        "fpc_u_i": u_i, "fpc_spectrum_i": fpc_spec_i, "fpc_net_rate_i": fpc_net_i,
        # RH / heating scalars
        "theta_bn_deg": np.asarray(np.degrees(theta_bn)),
        "r_measured": np.asarray(r_measured), "r_RH": np.asarray(jump["r"]),
        "mach_a": np.asarray(jump["mach_a"]), "mach_s": np.asarray(jump["mach_s"]),
        "T_factor": np.asarray(T_factor),
        "T_tot_up": np.asarray(T_tot_up), "T_tot_dn": np.asarray(T_tot_dn),
        **{f"heat_tot_{k}": np.asarray(v) for k, v in heat_tot.items()},
        "dT_e": np.asarray(dT_e), "dT_i": np.asarray(dT_i),
        "frac_heat_e": np.asarray(frac_heat_e), "frac_heat_i": np.asarray(frac_heat_i),
        # potential / reflection / FPC scalars
        "e_delta_phi": np.asarray(e_dphi), "reflection_parameter": np.asarray(refl_param),
        "reflected_frac_foot": np.asarray(refl_frac_foot),
        "reflected_frac_peak": np.asarray(refl_frac_peak),
        "fpc_total_e": np.asarray(fpc_tot_e), "fpc_total_i": np.asarray(fpc_tot_i),
        # energy budget
        **{f"up_{k}": np.asarray(v) for k, v in budget["upstream"].items()},
        **{f"dn_{k}": np.asarray(v) for k, v in budget["downstream"].items()},
        "ram_in": np.asarray(ram_in),
        # metadata
        "t_val": np.asarray(st.t_val), "t_sim": np.asarray(st.t_sim),
        "x_shock": np.asarray(st.x_shock), "v_shock": np.asarray(st.v_shock),
        "abs_rqm_i": np.asarray(st.abs_rqm_i), "Z_i": np.asarray(st.Z_i),
        "config_path": np.asarray(os.path.abspath(args.config)),
    }
    np.savez(out_path, **save)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
