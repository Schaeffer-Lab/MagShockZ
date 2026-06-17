"""Forward-model the experiment's observables from an OSIRIS shock dump.

The MagShockZ experiment measures the shock through **density imaging**,
**magnetic-field probes/radiography**, and **X-ray emission** (there is no
Thomson scattering, so T_e is only constrained through the T_e-sensitive X-ray
signal).  This script projects a simulation dump onto those signals at the
instrument's spatial resolution, so the run can be compared with the data.

Pure forward models live in src/synthetic_diagnostics.py; this script just loads
the dump (src/shock_state.py), computes per-species densities for the
bremsstrahlung weighting, and degrades the shock-normal profiles to the quoted
instrument resolution.  In 1D the line of sight is the shock normal, so the
products are resolution-degraded profiles plus the interferometry column
density; true 2D images (LOS across the shock normal) come with the 2D port.
The same src functions apply to a FLASH lineout (flash_utils.flash_lineout),
which already carries physical Te/ne/ni.

Usage
-----
    conda activate analysis
    python scripts/compute_synthetic_diagnostics.py \\
        --config config/perlmutter_1.3.1d.yaml [--resolution-um 50] [--timestep-idx -1]
"""

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

import analysis_utils
import shock_state
import moments as mom
import synthetic_diagnostics as sd


def main():
    parser = argparse.ArgumentParser(description="Synthetic experimental diagnostics (OSIRIS).")
    parser.add_argument("--config", required=True)
    parser.add_argument("--timestep-idx", type=int, default=-1, dest="timestep_idx")
    parser.add_argument("--resolution-um", type=float, default=50.0,
                        help="Instrument spatial resolution (FWHM) in microns.")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cfg = analysis_utils.load_config(args.config)
    print(f"Config  : {args.config}\nsim_dir : {cfg['sim_dir']}\nLoading HDF5 files...")
    st = shock_state.load_shock_state(cfg, args.timestep_idx, species=("al", "si", "e"))

    # Per-species ion densities (sim ions carry unit charge => n_s in n_0);
    # physical charge states weight the bremsstrahlung emissivity as n_s Z_s^2.
    Z = {sp: st.spec.charge_state(sp) for sp in ("al", "si")}
    n = {sp: np.abs(mom.moment(st.pha_p1[sp], axis="p1", order=0)) for sp in ("al", "si")}
    n_e = st.n_e
    T_e = st.T_iso["e"]
    Bmag = np.sqrt(st.B2)
    x = st.x_pha

    # Instrument resolution in normalised units: res[c/wpe] = res[cm] / d_e[cm].
    d_e_cm = float(st.sim.d_e().to("cm").value)
    res_norm = (args.resolution_um * 1e-4) / d_e_cm

    # Forward models (relative units).
    emiss = sd.bremsstrahlung_emissivity(n_e, T_e, species=[(n["al"], Z["al"]), (n["si"], Z["si"])])

    ne_obs = sd.apply_resolution(n_e, x, res_norm)
    B_obs = sd.apply_resolution(Bmag, x, res_norm)
    xray_obs = sd.apply_resolution(emiss, x, res_norm)

    # Interferometry column density (full LOS along the shock normal).
    column_density = float(sd.line_of_sight_integral(n_e, x))

    # B-probe samples at upstream / shock / downstream.
    probe_x = np.array([st.x_shock + 100.0, st.x_shock, st.x_shock - 100.0])
    B_probe = sd.probe_signal(Bmag, x, probe_x, fwhm=res_norm)

    # How much does the instrument smear the shock?
    def peak_near_shock(arr, half=300.0):
        m = (x > st.x_shock - half) & (x < st.x_shock + half)
        return float(np.nanmax(arr[m])) if m.any() else float("nan")

    print(f"\nInstrument resolution: {args.resolution_um:.0f} µm = {res_norm:.1f} c/ωpe "
          f"(d_e = {d_e_cm*1e4:.2f} µm)")
    print(f"Interferometry column density (∫n_e dl) = {column_density:.3e} [n_0 · c/ωpe]")
    print(f"\n{'quantity':<14} {'peak (full)':>14} {'peak (instr.)':>14} {'smearing':>10}")
    for label, full, obs in (("n_e", n_e, ne_obs), ("|B|", Bmag, B_obs), ("X-ray", emiss, xray_obs)):
        pf, po = peak_near_shock(full), peak_near_shock(obs)
        print(f"  {label:<12} {pf:>14.3e} {po:>14.3e} {100*(1-po/pf):>9.1f}%")
    print(f"\nB-probe |B| at [up, shock, dn] = "
          f"[{B_probe[0]:.4f}, {B_probe[1]:.4f}, {B_probe[2]:.4f}] [B_0]")

    out_path = analysis_utils.default_output_path(
        args.output, st.sim_dir, "synthetic_diagnostics", st.t_val)
    np.savez(
        out_path,
        x_axis=x, x_shock=np.asarray(st.x_shock),
        n_e=n_e, n_e_obs=ne_obs,
        Bmag=Bmag, Bmag_obs=B_obs,
        xray_emissivity=emiss, xray_obs=xray_obs,
        T_e=T_e,
        column_density=np.asarray(column_density),
        probe_x=probe_x, B_probe=B_probe,
        resolution_um=np.asarray(args.resolution_um),
        resolution_norm=np.asarray(res_norm),
        t_val=np.asarray(st.t_val), t_sim=np.asarray(st.t_sim),
        config_path=np.asarray(os.path.abspath(args.config)),
    )
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
