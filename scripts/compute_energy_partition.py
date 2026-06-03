"""Compute shock energy partition from an OSIRIS simulation and save to .npz.

Usage
-----
    python scripts/compute_energy_partition.py \\
        --config config/perlmutter_1.3.1d.yaml \\
        [--timestep-idx -1] \\
        [--output results/perlmutter_1.3.1d/energy_partition_t000360.npz]

The output .npz contains all energy density profiles and region-averaged
partition values, plus the config metadata needed to reproduce the run.
Pass the .npz directly to plot_energy_partition.py.
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
import energy_partition as ep


def main():
    parser = argparse.ArgumentParser(description="Compute shock energy partition.")
    parser.add_argument("--config", required=True, help="Path to analysis YAML config.")
    parser.add_argument(
        "--timestep-idx",
        type=int,
        default=-1,
        dest="timestep_idx",
        help="Index into config times list (default: -1, i.e. last dump).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output .npz path. Defaults to "
            "results/<run_name>/energy_partition_t{t:06d}.npz"
        ),
    )
    args = parser.parse_args()

    cfg = analysis_utils.load_config(args.config)
    sim_dir = cfg["sim_dir"]
    times = cfg["times"]
    t_idx = args.timestep_idx
    t_val = times[t_idx]

    field_mode = cfg.get("field_mode", "full")

    print(f"Config  : {args.config}")
    print(f"sim_dir : {sim_dir}")
    print(f"Dump    : t={t_val}  (index {t_idx} of {len(times)} dumps)")

    sim = analysis_utils.run_from_config(cfg)
    species_list = list(sim.deck.species)  # species defined in the input deck

    # Load phase spaces and fields for this single dump
    print("Loading HDF5 files...")
    pha = {
        sp: osh5io.read_h5(analysis_utils.phase_path(sim_dir, "p1x1", sp, t_val))
        for sp in species_list
    }
    fld = {
        name: osh5io.read_h5(analysis_utils.field_path(sim_dir, name, t_val))
        for name in ["b1", "b2", "b3", "e1", "e2", "e3"]
    }

    # Spatial grids: phase space axes are [p1, x1]; field axes are [x1]
    x_pha = axis_values(pha[species_list[0]], ax_idx=1)
    x_field = axis_values(fld["b1"], ax_idx=0)

    t_sim = float(pha[species_list[0]].run_attrs["TIME"][0])
    dump = analysis_utils.resolve_dump_params(cfg, t_val, t_sim)
    v_shock = dump["v_shock"]
    x_shock = dump["x_shock"]
    x_downstream_start = dump["x_downstream_start"]

    print(f"t_sim   : {t_sim:.1f} [ωpe⁻¹]")
    print(f"x_shock : {x_shock:.1f} [c/ωpe]")

    species_rqm = {sp: sim.rqm_of(sp) for sp in species_list}  # per-species rqm from deck

    # Sum energy profiles over all species
    u_ram_total = np.zeros(x_pha.size)
    u_th_total = np.zeros(x_pha.size)
    for sp in species_list:
        u_ram_sp, u_th_sp = ep.species_energy_profiles(
            pha[sp], species_rqm[sp], v_shock
        )
        u_ram_total += u_ram_sp
        u_th_total += u_th_sp

    b_arrs = [np.asarray(fld[f"b{i}"]) for i in range(1, 4)]
    e_arrs = [np.asarray(fld[f"e{i}"]) for i in range(1, 4)]
    u_B_fld, u_E_fld = ep.field_energy_profiles(
        *b_arrs, *e_arrs, x_field, x_shock, field_mode=field_mode
    )

    # Interpolate field profiles onto phase-space grid for consistent masks
    u_B = np.interp(x_pha, x_field, u_B_fld)
    u_E = np.interp(x_pha, x_field, u_E_fld)

    partition = ep.partition_by_region(
        u_ram_total, u_th_total, u_B, u_E,
        x_pha, x_shock, x_downstream_start,
    )
    up = partition["upstream"]
    dn = partition["downstream"]

    total_up = sum(up.values())
    total_dn = sum(dn.values())
    print("\n--- Energy partition ---")
    print(f"{'Channel':<12} {'Upstream [sim]':>18} {'(%)':>7}  "
          f"{'Downstream [sim]':>18} {'(%)':>7}")
    for k in ("ram", "thermal", "B_field", "E_field"):
        label = k.replace("_", " ").capitalize()
        print(f"{label:<12} {up[k]:>18.3e} {100*up[k]/total_up:>6.1f}%  "
              f"{dn[k]:>18.3e} {100*dn[k]/total_dn:>6.1f}%")

    # Output path
    out_path = analysis_utils.default_output_path(
        args.output, sim_dir, "energy_partition", t_val
    )

    np.savez(
        out_path,
        # Profiles (on phase-space grid)
        x_axis=x_pha,
        u_ram=u_ram_total,
        u_th=u_th_total,
        u_B=u_B,
        u_E=u_E,
        # Scalars
        t_val=np.asarray(t_val),
        t_sim=np.asarray(t_sim),
        x_shock=np.asarray(x_shock),
        x_downstream_start=np.asarray(x_downstream_start),
        # Region averages
        upstream_ram=np.asarray(up["ram"]),
        upstream_thermal=np.asarray(up["thermal"]),
        upstream_B_field=np.asarray(up["B_field"]),
        upstream_E_field=np.asarray(up["E_field"]),
        downstream_ram=np.asarray(dn["ram"]),
        downstream_thermal=np.asarray(dn["thermal"]),
        downstream_B_field=np.asarray(dn["B_field"]),
        downstream_E_field=np.asarray(dn["E_field"]),
        # Metadata for reproducibility
        field_mode=np.asarray(field_mode),
        norm_density_cm3=np.asarray(cfg["norm_density_cm3"]),
        config_path=np.asarray(os.path.abspath(args.config)),
    )
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
