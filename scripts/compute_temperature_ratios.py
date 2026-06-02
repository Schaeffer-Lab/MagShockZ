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

import astropy.constants
import astropy.units
import numpy as np
import osh5io
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

import analysis_utils
import temperature_anisotropy as ta


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["sim_dir"] = os.environ.get("MAGSHOCKZ_SIM_DIR", cfg["sim_dir"])
    return cfg


def spatial_axis(h5data, ax_idx: int) -> np.ndarray:
    ax = h5data.axes[ax_idx]
    return np.linspace(ax.min, ax.max, ax.size)


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

    cfg = load_config(args.config)
    sim_dir = cfg["sim_dir"]
    t_val = cfg["times"][args.timestep_idx]
    v_shock = cfg["shock"]["v_shock"]
    x_shock_0 = cfg["shock"]["x_shock_0"]
    x_downstream_start = cfg["shock"]["x_downstream_start"]

    print(f"Config  : {args.config}")
    print(f"sim_dir : {sim_dir}")
    print(f"Dump    : t={t_val}  (index {args.timestep_idx} of {len(cfg['times'])} dumps)")

    norm_density = float(cfg["norm_density_cm3"]) * astropy.units.cm**-3
    sim = analysis_utils.MagShockZRun(
        os.path.join(sim_dir, cfg.get("input_deck", "magshockz_gpu.1d")),
        norm_density=norm_density,
    )
    m_e = astropy.constants.m_e
    m_ion = abs(sim.rqm) * m_e
    masses = {"al": m_ion, "e": m_e}

    print("Loading HDF5 files...")
    pha_p1 = {
        sp: osh5io.read_h5(f"{sim_dir}/MS/PHA/p1x1/{sp}/p1x1-{sp}-{t_val:06d}.h5")
        for sp in ["al", "e"]
    }
    pha_p2 = {
        sp: osh5io.read_h5(f"{sim_dir}/MS/PHA/p2x1/{sp}/p2x1-{sp}-{t_val:06d}.h5")
        for sp in ["al", "e"]
    }

    x_axis = spatial_axis(pha_p1["al"], ax_idx=1)
    t_sim = float(pha_p1["al"].run_attrs["TIME"][0])
    x_shock = x_shock_0 + v_shock * t_sim

    print(f"t_sim   : {t_sim:.1f} [ωpe⁻¹]")
    print(f"x_shock : {x_shock:.1f} [c/ωpe]")

    # Temperature profiles in eV
    T_par = {sp: ta.temperature_profile(pha_p1[sp], masses[sp], "p1") for sp in ["al", "e"]}
    T_perp = {sp: ta.temperature_profile(pha_p2[sp], masses[sp], "p2") for sp in ["al", "e"]}

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
    print("\n--- Temperature summary [eV] ---")
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
    if args.output is None:
        run_name = os.path.basename(sim_dir.rstrip("/"))
        out_dir = os.path.join(_HERE, "..", "results", run_name)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"temperature_ratios_t{t_val:06d}.npz")
    else:
        out_dir = os.path.dirname(args.output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        out_path = args.output

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
