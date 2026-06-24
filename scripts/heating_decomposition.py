"""heating_decomposition — decompose shock heating into adiabatic vs. anomalous.

Science
-------
The project's headline physics deliverable: for one OSIRIS dump it assembles the
full picture of *how* the shock heats, composing five pure modules (OSIRIS
normalised units throughout):

  1. Rankine--Hugoniot baseline   (src/rankine_hugoniot.py)
       θ_Bn, the perpendicular-MHD compression ratio, and the adiabatic
       downstream temperature.  The excess of the measured downstream T over that
       is the **anomalous (collisionless) heating**; how the measured heat splits
       between electrons and ions is the collisionless partition.
  2. Cross-shock potential         (src/cross_shock_potential.py)
       e·Δφ across the front and its size vs. the ion ram energy.
  3. Reflected ions                (src/reflected_ions.py)
       reflected fraction and reflected-ion energy in the foot.
  4. Field-particle correlation    (src/field_particle_correlation.py)
       velocity-space heating signature + net field-work rate per species.
  5. Energy budget                 (src/energy_partition.py)
       ram / thermal_e / thermal_i / B / E region averages.

Validation
----------
This file is orchestration + plotting only; it contains no physics formulas — each
quantity is produced by the named pure module above (tested in the matching
tests/test_*.py).  The one dump is loaded once through src/shock_state.py.  The
result is saved to an inspectable ``.npz`` whose schema is the
``HeatingDecompositionResult`` dataclass below.

Usage
-----
    python scripts/heating_decomposition.py --config config/<run>.yaml \\
        [--timestep-idx -1] [--no-plot] [--half-window 250] [--output ...] [--output-dir ...]
"""

import argparse
import os
import sys
from dataclasses import dataclass

import matplotlib.pyplot as plt
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

_BUDGET_KEYS = ("ram", "thermal_e", "thermal_i", "B_field", "E_field")


# ---------------------------------------------------------------------------
# Output schema (this dataclass IS the .npz schema; see analysis_utils.save_result)
# ---------------------------------------------------------------------------

@dataclass
class HeatingDecompositionResult:
    # --- profiles on the phase-space grid (OSIRIS units) ---
    x_axis: np.ndarray                 # [c/wpe]
    n_e: np.ndarray                    # electron density [n_0]
    B2: np.ndarray                     # |B|^2 [B_0^2]
    T_iso_e: np.ndarray                # isotropic electron T [m_e c^2]
    T_iso_i: np.ndarray                # isotropic ion T [m_e c^2]
    e_phi_profile: np.ndarray          # cross-shock potential energy e·φ(x) [m_e c^2]
    reflected_fraction: np.ndarray     # reflected-ion number fraction
    reflected_energy_density: np.ndarray  # reflected-ion energy density [n_0 m_e c^2]
    u_ram_total: np.ndarray            # bulk-flow KE density (e+i) [n_0 m_e c^2]
    u_th_e: np.ndarray                 # electron thermal energy density [n_0 m_e c^2]
    u_th_i: np.ndarray                 # ion thermal energy density [n_0 m_e c^2]
    u_B: np.ndarray                    # magnetic energy density [n_0 m_e c^2]
    u_E: np.ndarray                    # electric energy density [n_0 m_e c^2]
    # --- FPC velocity-space signatures (u-grid + layer-averaged spectrum + net rate vs x) ---
    fpc_u_e: np.ndarray                # electron velocity grid [c]
    fpc_spectrum_e: np.ndarray         # ⟨C_E⟩ over the shock layer (electrons)
    fpc_net_rate_e: np.ndarray         # ∫C_E du vs x (electrons)
    fpc_u_i: np.ndarray                # ion velocity grid [c]
    fpc_spectrum_i: np.ndarray         # ⟨C_E⟩ over the shock layer (ions)
    fpc_net_rate_i: np.ndarray         # ∫C_E du vs x (ions)
    # --- RH / heating scalars ---
    theta_bn_deg: float                # shock-normal angle [deg]
    r_measured: float                  # measured compression n_dn/n_up
    r_RH: float                        # RH-predicted compression
    mach_a: float                      # Alfvénic Mach number
    mach_s: float                      # sonic Mach number
    T_factor: float                    # RH predicted total T2/T1
    T_tot_up: float                    # upstream T_e+T_i [m_e c^2]
    T_tot_dn: float                    # downstream T_e+T_i [m_e c^2]
    dT_e: float                        # downstream-upstream electron T [m_e c^2]
    dT_i: float                        # downstream-upstream ion T [m_e c^2]
    frac_heat_e: float                 # electron share of measured heating
    frac_heat_i: float                 # ion share of measured heating
    # --- potential / reflection / FPC scalars ---
    e_delta_phi: float                 # cross-shock potential jump e·Δφ [m_e c^2]
    reflection_parameter: float        # e·Δφ / (½ m_i v_sh^2)
    reflected_frac_foot: float         # mean reflected fraction in the foot
    reflected_frac_peak: float         # peak reflected fraction upstream
    fpc_total_e: float                 # net electron field work in the layer
    fpc_total_i: float                 # net ion field work in the layer
    ram_in: float                      # upstream ram energy density [n_0 m_e c^2]
    # --- metadata ---
    t_val: int                         # dump index
    t_sim: float                       # simulation time [1/wpe]
    x_shock: float                     # shock position [c/wpe]
    v_shock: float                     # shock velocity [c]
    abs_rqm_i: float                   # |rqm| of the ion
    Z_i: int                           # ion charge state
    config_path: str                   # absolute path of the analysis config used
    # --- dict groups (flattened to <field>_<key> in the .npz) ---
    heat_tot: dict                     # {adiabatic, anomalous, total_heating, anomalous_frac}
    up: dict                           # upstream  energy budget, keyed by _BUDGET_KEYS
    dn: dict                           # downstream energy budget, keyed by _BUDGET_KEYS


# ---------------------------------------------------------------------------
# Compute helpers (orchestration of src/field_particle_correlation.py)
# ---------------------------------------------------------------------------

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
    spectrum = np.nanmean(c_e[:, layer_mask], axis=1) if layer_mask.any() \
        else np.full(u.size, np.nan)
    total_in_layer = float(np.nansum(net_rate[layer_mask]))
    return u, spectrum, net_rate, total_in_layer


# ---------------------------------------------------------------------------
# Compute (orchestration only — every formula lives in the five src/ modules)
# ---------------------------------------------------------------------------

def compute(cfg: dict, timestep_idx: int = -1, config_path: str = "") -> HeatingDecompositionResult:
    """Load one dump via shock_state, run the five decompositions, return the result."""
    print("Loading HDF5 files...")
    st = shock_state.load_shock_state(cfg, timestep_idx)
    ion = st.ion
    up_p, dn_p = st.prim_up, st.prim_dn

    # 1. Rankine--Hugoniot baseline.
    theta_bn = rh.shock_normal_angle(up_p["b1"], up_p["b2"], up_p["b3"])
    v_inflow = st.v_shock - up_p["u_bulk_i"]          # shock-frame inflow speed [c]
    jump = rh.solve_jump(up_p["n_e"], up_p["T_e"], up_p["T_i"], up_p["B2"],
                         st.abs_rqm_i, v_inflow)
    T_factor = jump["T_factor"]
    r_measured = dn_p["n_e"] / up_p["n_e"] if up_p["n_e"] > 0 else float("nan")

    # MHD predicts one total post-shock T; the measured e/i split is collisionless.
    T_tot_up = up_p["T_e"] + up_p["T_i"]
    T_tot_dn = dn_p["T_e"] + dn_p["T_i"]
    heat_tot = rh.anomalous_heating(T_tot_dn, T_tot_up, T_factor)
    dT_e = dn_p["T_e"] - up_p["T_e"]
    dT_i = dn_p["T_i"] - up_p["T_i"]
    dT_tot = dT_e + dT_i
    frac_heat_e = dT_e / dT_tot if dT_tot != 0 else float("nan")
    frac_heat_i = dT_i / dT_tot if dT_tot != 0 else float("nan")

    # 2. Cross-shock potential (from the normal field e1 on the phase grid).
    e1 = st.fields["e1"]
    e_dphi = csp.potential_jump(e1, st.x_pha, st.upstream_mask, st.downstream_mask)
    refl_param = csp.reflection_parameter(e_dphi, st.abs_rqm_i, st.v_shock)
    e_phi_profile = csp.potential_profile(e1, st.x_pha)

    # 3. Reflected ions.
    incoming_sign = ri.infer_incoming_sign(up_p["u_bulk_i"], st.v_shock)
    refl_frac = ri.reflected_fraction(st.pha_p1[ion], st.v_shock, incoming_sign)
    refl_energy = ri.reflected_energy_density(
        st.pha_p1[ion], st.species_rqm[ion], st.v_shock, incoming_sign)
    foot_ncells = min(st.up_ncells, 60)
    foot_mask = (st.x_pha > st.x_shock) & (st.x_pha <= st.x_shock + foot_ncells * st.dx)
    refl_frac_foot = float(np.nanmean(refl_frac[foot_mask])) if foot_mask.any() else float("nan")
    refl_frac_peak = float(np.nanmax(refl_frac[st.upstream_mask])) if st.upstream_mask.any() else float("nan")

    # 4. Field-particle correlation (velocity-space heating signature).
    layer_mask = st.downstream_mask | foot_mask     # ramp + foot + near downstream
    u_e, fpc_spec_e, fpc_net_e, fpc_tot_e = _velocity_signature(
        st.pha_p1["e"], e1, st.species_rqm["e"], layer_mask)
    u_i, fpc_spec_i, fpc_net_i, fpc_tot_i = _velocity_signature(
        st.pha_p1[ion], e1, st.species_rqm[ion], layer_mask)

    # 5. Energy budget (ram / thermal_e / thermal_i / B / E).
    u_ram, u_th = {}, {}
    for sp in ("e", ion):
        u_ram[sp], u_th[sp] = ep.species_energy_profiles(
            st.pha_p1[sp], st.species_rqm[sp], st.v_shock,
            perp_phase_spaces=st.pha_perp[sp])
    u_ram_tot = u_ram["e"] + u_ram[ion]
    # Field energy densities (B'^2/2, E'^2/2) — physics owned by energy_partition.
    # Fields are already on the phase grid, so x_shock is irrelevant in "full" mode.
    u_B, u_E = ep.field_energy_profiles(
        st.fields["b1"], st.fields["b2"], st.fields["b3"],
        st.fields["e1"], st.fields["e2"], st.fields["e3"],
        st.x_pha, st.x_shock, field_mode="full")

    region_mean = lambda arr, mask: float(np.nanmean(arr[mask])) if mask.any() else float("nan")
    channels = {"ram": u_ram_tot, "thermal_e": u_th["e"], "thermal_i": u_th[ion],
                "B_field": u_B, "E_field": u_E}
    up = {k: region_mean(v, st.upstream_mask) for k, v in channels.items()}
    dn = {k: region_mean(v, st.downstream_mask) for k, v in channels.items()}
    ram_in = up["ram"]

    return HeatingDecompositionResult(
        x_axis=st.x_pha, n_e=st.n_e, B2=st.B2,
        T_iso_e=st.T_iso["e"], T_iso_i=st.T_iso[ion],
        e_phi_profile=e_phi_profile, reflected_fraction=refl_frac,
        reflected_energy_density=refl_energy,
        u_ram_total=u_ram_tot, u_th_e=u_th["e"], u_th_i=u_th[ion], u_B=u_B, u_E=u_E,
        fpc_u_e=u_e, fpc_spectrum_e=fpc_spec_e, fpc_net_rate_e=fpc_net_e,
        fpc_u_i=u_i, fpc_spectrum_i=fpc_spec_i, fpc_net_rate_i=fpc_net_i,
        theta_bn_deg=float(np.degrees(theta_bn)),
        r_measured=float(r_measured), r_RH=float(jump["r"]),
        mach_a=float(jump["mach_a"]), mach_s=float(jump["mach_s"]),
        T_factor=float(T_factor), T_tot_up=float(T_tot_up), T_tot_dn=float(T_tot_dn),
        dT_e=float(dT_e), dT_i=float(dT_i),
        frac_heat_e=float(frac_heat_e), frac_heat_i=float(frac_heat_i),
        e_delta_phi=float(e_dphi), reflection_parameter=float(refl_param),
        reflected_frac_foot=float(refl_frac_foot), reflected_frac_peak=float(refl_frac_peak),
        fpc_total_e=float(fpc_tot_e), fpc_total_i=float(fpc_tot_i), ram_in=float(ram_in),
        t_val=int(st.t_val), t_sim=float(st.t_sim),
        x_shock=float(st.x_shock), v_shock=float(st.v_shock),
        abs_rqm_i=float(st.abs_rqm_i), Z_i=int(st.Z_i), config_path=config_path,
        heat_tot=dict(heat_tot), up=up, dn=dn,
    )


def _print_summary(r: HeatingDecompositionResult) -> None:
    """Console summary of the heating decomposition (no computation)."""
    print(f"t_sim   : {r.t_sim:.1f} [ωpe⁻¹]   x_shock : {r.x_shock:.1f} [c/ωpe]   "
          f"v_shock : {r.v_shock:.4f} [c]")
    sep = "-" * 72
    print("\n" + "=" * 72)
    print("  Shock heating decomposition")
    print("=" * 72)
    quasi = "quasi-perp" if rh.is_quasi_perpendicular(np.radians(r.theta_bn_deg)) else "quasi-par"
    print(f"  theta_Bn        = {r.theta_bn_deg:6.1f} deg  ({quasi})")
    print(f"  compression r   = {r.r_measured:6.3f} (measured)   {r.r_RH:6.3f} (RH)")
    print(f"  M_A, M_s        = {r.mach_a:6.2f}, {r.mach_s:6.2f}")
    print(f"  RH T-jump       = {r.T_factor:6.3f}  (predicted total T2/T1)")
    print(f"\n{sep}")
    print("  Total heating vs MHD baseline   (T = T_e + T_i)")
    print(sep)
    print(f"  {'T_tot_up':>11} {'T_tot_dn':>11} {'T_RH':>11} {'T_anom':>11} {'anom %':>8}")
    print(f"  {r.T_tot_up:>11.3e} {r.T_tot_dn:>11.3e} {r.heat_tot['adiabatic']:>11.3e} "
          f"{r.heat_tot['anomalous']:>11.3e} {100*r.heat_tot['anomalous_frac']:>7.1f}%")
    print(f"\n  Heating partition (measured): "
          f"electrons {100*r.frac_heat_e:5.1f}%   ions {100*r.frac_heat_i:5.1f}%")
    print(f"    ΔT_e = {r.dT_e:.3e}   ΔT_i = {r.dT_i:.3e}")
    print(f"\n  Cross-shock potential  e*Δphi = {r.e_delta_phi:.3e} [m_e c²]   "
          f"e*Δphi / (½ m_i v_sh²) = {r.reflection_parameter:.3f}")
    print(f"  Reflected ions         foot fraction = {r.reflected_frac_foot:.3f}   "
          f"peak = {r.reflected_frac_peak:.3f}")
    print(f"  FPC net field work in layer   electrons = {r.fpc_total_e:+.3e}   "
          f"ions = {r.fpc_total_i:+.3e}")
    print(f"\n{sep}")
    print("  Energy budget (downstream channel / upstream ram-in)")
    print(sep)
    for k in _BUDGET_KEYS:
        frac = r.dn[k] / r.ram_in if r.ram_in else float("nan")
        print(f"  {k:<12} up={r.up[k]:>11.3e}  dn={r.dn[k]:>11.3e}  dn/ram_in={100*frac:>6.1f}%")


# ---------------------------------------------------------------------------
# Plot (matplotlib only — reads the result, draws nothing new)
# ---------------------------------------------------------------------------

def _window(x, x_shock, half):
    return (x >= x_shock - half) & (x <= x_shock + half)


def _plot_temperatures(r, ax, half):
    m = _window(r.x_axis, r.x_shock, half)
    ax.semilogy(r.x_axis[m], r.T_iso_e[m], label="T_e (measured)")
    ax.semilogy(r.x_axis[m], r.T_iso_i[m], label="T_i (measured)")
    ax.axhline(r.heat_tot["adiabatic"], color="k", ls=":", lw=1.2, label="RH total T (MHD)")
    ax.axvline(r.x_shock, color="0.4", ls="--", lw=1)
    ax.set_xlabel("x [c/ωpe]"); ax.set_ylabel("T [mₑc²]")
    ax.set_title("Temperatures vs MHD baseline"); ax.legend(fontsize=8); ax.grid(alpha=0.3)


def _plot_potential(r, ax, half):
    m = _window(r.x_axis, r.x_shock, half)
    ax.plot(r.x_axis[m], r.e_phi_profile[m], color="C3")
    ax.axvline(r.x_shock, color="0.4", ls="--", lw=1)
    ax.set_xlabel("x [c/ωpe]"); ax.set_ylabel("e·φ [mₑc²]")
    ax.set_title(f"Cross-shock potential  (e·Δφ/½m_iv² = {r.reflection_parameter:.2f})")
    ax.grid(alpha=0.3)


def _plot_budget(r, ax):
    labels = ["ram", "th_e", "th_i", "B", "E"]
    up = [r.up[k] for k in _BUDGET_KEYS]
    dn = [r.dn[k] for k in _BUDGET_KEYS]
    xi = np.arange(len(labels)); w = 0.38
    ax.bar(xi - w / 2, up, w, label="upstream")
    ax.bar(xi + w / 2, dn, w, label="downstream")
    ax.set_xticks(xi); ax.set_xticklabels(labels)
    ax.set_ylabel("energy density [n₀mₑc²]")
    ax.set_title("Energy budget"); ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)


def _plot_fpc(r, ax, species):
    u = getattr(r, f"fpc_u_{species}"); spec = getattr(r, f"fpc_spectrum_{species}")
    ax.plot(u, spec, color="C4")
    ax.axhline(0, color="0.6", lw=0.8); ax.axvline(0, color="0.6", lw=0.8)
    ax.set_xlabel("u [c]"); ax.set_ylabel("⟨C_E⟩ (layer)")
    name = "electron" if species == "e" else "ion"
    ax.set_title(f"FPC {name} velocity signature"); ax.grid(alpha=0.3)


def _plot_partition(r, ax):
    ax.bar(["ΔT_e", "ΔT_i"], [r.dT_e, r.dT_i], color=["C0", "C1"])
    ax.set_ylabel("downstream − upstream T [mₑc²]")
    anom = 100 * r.heat_tot["anomalous_frac"]
    ax.set_title(f"Heating partition  e {100*r.frac_heat_e:.0f}% / i {100*r.frac_heat_i:.0f}%\n"
                 f"anomalous (vs MHD) {anom:+.0f}%")
    ax.grid(axis="y", alpha=0.3)


def plot(r: HeatingDecompositionResult, output_dir: str, half_window: float = 250.0) -> str:
    """Render the 2×3 decomposition figure and save a .png."""
    fig, ax = plt.subplots(2, 3, figsize=(18, 9))
    _plot_temperatures(r, ax[0, 0], half_window)
    _plot_potential(r, ax[0, 1], half_window)
    _plot_budget(r, ax[0, 2])
    _plot_fpc(r, ax[1, 0], "e")
    _plot_fpc(r, ax[1, 1], "i")
    _plot_partition(r, ax[1, 2])

    fig.suptitle(f"Heating decomposition  t={r.t_val}  θ_Bn={r.theta_bn_deg:.0f}°  "
                 f"M_A={r.mach_a:.1f}  r_meas={r.r_measured:.2f}/r_RH={r.r_RH:.2f}",
                 fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"heating_decomposition_t{r.t_val:06d}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Shock heating decomposition (compute + plot).")
    parser.add_argument("--config", required=True, help="Path to analysis YAML config.")
    parser.add_argument("--timestep-idx", type=int, default=-1, dest="timestep_idx",
                        help="Index into config times list (default: -1, last dump).")
    parser.add_argument("--no-plot", action="store_true", dest="no_plot",
                        help="Compute and save the .npz only; skip the figure.")
    parser.add_argument("--half-window", type=float, default=250.0, dest="half_window",
                        help="Half-width [c/ωpe] of the near-shock plotting window.")
    parser.add_argument("--output", default=None,
                        help="Output .npz path (default results/<run>/heating_decomposition_t{t:06d}.npz).")
    parser.add_argument("--output-dir", default=None, dest="output_dir",
                        help="Directory for the figure (default: alongside the .npz).")
    args = parser.parse_args()

    cfg = analysis_utils.load_config(args.config)
    print(f"Config  : {args.config}")
    print(f"sim_dir : {cfg['sim_dir']}")

    result = compute(cfg, args.timestep_idx, config_path=os.path.abspath(args.config))
    _print_summary(result)

    out_path = analysis_utils.default_output_path(args.output, cfg["sim_dir"],
                                                  "heating_decomposition", result.t_val)
    analysis_utils.save_result(result, out_path)
    print(f"\nSaved → {out_path}")

    if not args.no_plot:
        out_dir = args.output_dir or os.path.dirname(os.path.abspath(out_path))
        fig_path = plot(result, out_dir, args.half_window)
        print(f"Saved → {fig_path}")


if __name__ == "__main__":
    main()
