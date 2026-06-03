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
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

import analysis_utils
import energy_partition as ep


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["sim_dir"] = os.environ.get("MAGSHOCKZ_SIM_DIR", cfg["sim_dir"])
    return cfg


def spatial_axis(h5data, ax_idx: int) -> np.ndarray:
    ax = h5data.axes[ax_idx]
    return np.linspace(ax.min, ax.max, ax.size)


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

    cfg = load_config(args.config)
    sim_dir = cfg["sim_dir"]
    times = cfg["times"]
    t_idx = args.timestep_idx
    t_val = times[t_idx]

    field_mode = cfg.get("field_mode", "full")
    species_list = cfg.get("species", ["al", "e", "si"])

    print(f"Config  : {args.config}")
    print(f"sim_dir : {sim_dir}")
    print(f"Dump    : t={t_val}  (index {t_idx} of {len(times)} dumps)")

    import astropy.units
    norm_density = float(cfg["norm_density_cm3"]) * astropy.units.cm**-3
    sim = analysis_utils.MagShockZRun(
        os.path.join(sim_dir, cfg.get("input_deck", "magshockz_gpu.1d")),
        norm_density=norm_density,
    )

    # Load phase spaces and fields for this single dump
    print("Loading HDF5 files...")
    pha = {
        sp: osh5io.read_h5(
            f"{sim_dir}/MS/PHA/p1x1/{sp}/p1x1-{sp}-{t_val:06d}.h5"
        )
        for sp in species_list
    }
    fld = {
        name: osh5io.read_h5(
            f"{sim_dir}/MS/FLD/{name}-savg/{name}-savg-{t_val:06d}.h5"
        )
        for name in ["b1", "b2", "b3", "e1", "e2", "e3"]
    }

    # Spatial grids: phase space axes are [p1, x1]; field axes are [x1]
    x_pha = spatial_axis(pha[species_list[0]], ax_idx=1)
    x_field = spatial_axis(fld["b1"], ax_idx=0)

    t_sim = float(pha[species_list[0]].run_attrs["TIME"][0])
    dump = analysis_utils.resolve_dump_params(cfg, t_val, t_sim)
    v_shock = dump["v_shock"]
    x_shock = dump["x_shock"]
    x_downstream_start = dump["x_downstream_start"]

    print(f"t_sim   : {t_sim:.1f} [ωpe⁻¹]")
    print(f"x_shock : {x_shock:.1f} [c/ωpe]")

    ion_rqm = sim.rqm
    species_rqm = {sp: (-1.0 if sp == "e" else ion_rqm) for sp in species_list}

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
    if args.output is None:
        run_name = os.path.basename(sim_dir.rstrip("/"))
        out_dir = os.path.join(_HERE, "..", "results", run_name)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"energy_partition_t{t_val:06d}.npz")
    else:
        out_dir = os.path.dirname(args.output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        out_path = args.output

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
