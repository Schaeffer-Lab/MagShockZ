"""Compute electron/ion temperature ratios and anisotropy across the shock.

Loads p1x1 (parallel, along shock normal) and p2x1 (perpendicular) phase
spaces for electrons and Al ions. Computes:
    - T_e / T_al  in the parallel and perpendicular directions
    - T_parallel / T_perp  for each species

Usage
-----
    python scripts/compute_temperature_ratios.py \\
        --config config/perlmutter_1.3.1d.yaml \\
        [--timestep-idx -1] \\
        [--output results/perlmutter_1.3.1d/temperature_ratios_t000360.npz]
"""

import argparse
import os
import sys

import numpy as np
import osh5io

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

import analysis_utils
from analysis_utils import axis_values
import temperature_anisotropy as ta


def main():
    parser = argparse.ArgumentParser(description="Compute shock temperature ratios.")
    parser.add_argument("--config", required=True, help="Path to analysis YAML config.")
    parser.add_argument(
        "--timestep-idx",
        type=int,
        default=-1,
        dest="timestep_idx",
        help="Index into config times list (default: -1, i.e. last dump).",
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cfg = analysis_utils.load_config(args.config)
    sim_dir = cfg["sim_dir"]
    t_val = cfg["times"][args.timestep_idx]

    print(f"Config  : {args.config}")
    print(f"sim_dir : {sim_dir}")
    print(f"Dump    : t={t_val}  (index {args.timestep_idx} of {len(cfg['times'])} dumps)")

    sim = analysis_utils.run_from_config(cfg)
    species_rqm = {sp: sim.rqm_of(sp) for sp in ["al", "e"]}  # per-species rqm from deck

    print("Loading HDF5 files...")
    pha_p1 = {
        sp: osh5io.read_h5(analysis_utils.phase_path(sim_dir, "p1x1", sp, t_val))
        for sp in ["al", "e"]
    }
    pha_p2 = {
        sp: osh5io.read_h5(analysis_utils.phase_path(sim_dir, "p2x1", sp, t_val))
        for sp in ["al", "e"]
    }

    x_axis = axis_values(pha_p1["al"], ax_idx=1)
    t_sim = float(pha_p1["al"].run_attrs["TIME"][0])
    dump = analysis_utils.resolve_dump_params(cfg, t_val, t_sim)
    x_shock = dump["x_shock"]
    x_downstream_start = dump["x_downstream_start"]

    print(f"t_sim   : {t_sim:.1f} [ωpe⁻¹]")
    print(f"x_shock : {x_shock:.1f} [c/ωpe]")

    # Temperature profiles in simulation units (m_e c^2)
    T_par = {sp: ta.temperature_profile(pha_p1[sp], species_rqm[sp], "p1") for sp in ["al", "e"]}
    T_perp = {sp: ta.temperature_profile(pha_p2[sp], species_rqm[sp], "p2") for sp in ["al", "e"]}

    # Ratios
    T_e_al_par = ta.safe_ratio(T_par["e"], T_par["al"])
    T_e_al_perp = ta.safe_ratio(T_perp["e"], T_perp["al"])
    anis_e = ta.safe_ratio(T_par["e"], T_perp["e"])
    anis_al = ta.safe_ratio(T_par["al"], T_perp["al"])

    # Region averages — compute once, store cleanly
    avgs = {
        key: ta.region_averages(arr, x_axis, x_shock, x_downstream_start)
        for key, arr in [
            ("T_par_e",      T_par["e"]),
            ("T_par_al",     T_par["al"]),
            ("T_perp_e",     T_perp["e"]),
            ("T_perp_al",    T_perp["al"]),
            ("T_e_al_par",   T_e_al_par),
            ("T_e_al_perp",  T_e_al_perp),
            ("anis_e",       anis_e),
            ("anis_al",      anis_al),
        ]
    }

    # Print summary table
    print("\n--- Temperature summary [m_e c^2] ---")
    header = f"{'':12s} {'T_par up':>10} {'T_par dn':>10} {'T_perp up':>10} {'T_perp dn':>10}"
    print(header)
    for sp in ["e", "al"]:
        up_par, dn_par = avgs[f"T_par_{sp}"]
        up_perp, dn_perp = avgs[f"T_perp_{sp}"]
        print(f"{sp:<12} {up_par:>10.2f} {dn_par:>10.2f} {up_perp:>10.2f} {dn_perp:>10.2f}")

    print("\n--- Ratios ---")
    for key, label in [
        ("T_e_al_par",  "T_e/T_al (par) "),
        ("T_e_al_perp", "T_e/T_al (perp)"),
        ("anis_e",      "T_par/T_perp e  "),
        ("anis_al",     "T_par/T_perp al "),
    ]:
        up, dn = avgs[key]
        print(f"  {label}: upstream={up:.3f}  downstream={dn:.3f}")

    # Output path
    out_path = analysis_utils.default_output_path(
        args.output, sim_dir, "temperature_ratios", t_val
    )

    # Flatten region averages for savez (one scalar per key+region)
    save_dict = {
        "x_axis": x_axis,
        "T_par_e": T_par["e"],
        "T_par_al": T_par["al"],
        "T_perp_e": T_perp["e"],
        "T_perp_al": T_perp["al"],
        "T_e_al_par": T_e_al_par,
        "T_e_al_perp": T_e_al_perp,
        "anis_e": anis_e,
        "anis_al": anis_al,
        "t_val": np.asarray(t_val),
        "t_sim": np.asarray(t_sim),
        "x_shock": np.asarray(x_shock),
        "x_downstream_start": np.asarray(x_downstream_start),
        "config_path": np.asarray(os.path.abspath(args.config)),
    }
    for key, (up, dn) in avgs.items():
        save_dict[f"up_{key}"] = np.asarray(up)
        save_dict[f"dn_{key}"] = np.asarray(dn)

    np.savez(out_path, **save_dict)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
