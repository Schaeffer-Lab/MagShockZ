"""energy_partition — how a collisionless shock partitions its energy.

Science
-------
For one OSIRIS dump this splits the energy density (OSIRIS units, n_0 m_e c^2)
into four channels, upstream vs. downstream of the front:

    u_ram   bulk-flow kinetic energy   1/2 n |rqm| (<u> - v_shock)^2
    u_th    thermal energy             1/2 n |rqm| Sum_d sigma_d^2
    u_B     magnetic energy            B'^2 / 2
    u_E     electric energy            E'^2 / 2

Every formula above is owned by the pure, unit-tested module
``src/energy_partition.py`` (``species_energy_profiles`` for u_ram/u_th,
``field_energy_profiles`` for u_B/u_E, ``partition_by_region`` for the
upstream/downstream averages).  See the OSIRIS units table in CLAUDE.md.

Validation
----------
This file is orchestration + plotting only; it contains no physics formulas.
To validate a number, read the named function in ``src/energy_partition.py`` and
its test in ``tests/test_energy_partition.py``.  The computed result is also
saved to an inspectable ``.npz`` whose schema is the ``EnergyPartitionResult``
dataclass below.

Usage
-----
    python scripts/energy_partition.py --config config/<run>.yaml \\
        [--timestep-idx -1] [--no-plot] [--output ...] [--output-dir ...]
"""

import argparse
import os
import sys
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import osh5io

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

import analysis_utils
import plot_style
from analysis_utils import axis_values
import energy_partition as ep


# ---------------------------------------------------------------------------
# Output schema (this dataclass IS the .npz schema; see analysis_utils.save_result)
# ---------------------------------------------------------------------------

@dataclass
class EnergyPartitionResult:
    # --- energy-density profiles on the phase-space grid ---
    x_axis: np.ndarray            # [c/wpe]
    u_ram: np.ndarray             # bulk-flow KE density   [n_0 m_e c^2]
    u_th: np.ndarray              # thermal energy density [n_0 m_e c^2]
    u_B: np.ndarray               # magnetic energy density [n_0 m_e c^2]
    u_E: np.ndarray               # electric energy density [n_0 m_e c^2]
    # --- shock kinematics / provenance ---
    t_val: int                    # dump index (file suffix)
    t_sim: float                  # simulation time [1/wpe]
    x_shock: float                # shock-front position [c/wpe]
    x_downstream_start: float     # left edge of downstream region [c/wpe]
    v_shock: float                # shock-frame boost velocity [c] (tuned config value)
    field_mode: str               # "full" or "delta"
    norm_density_cm3: float       # reference density n_0 [cm^-3]
    config_path: str              # absolute path of the analysis config used
    # --- region averages: mean energy density per channel [n_0 m_e c^2] ---
    upstream: dict                # {"ram","thermal","B_field","E_field"}
    downstream: dict


# ---------------------------------------------------------------------------
# Compute (orchestration only — every formula lives in src/energy_partition.py)
# ---------------------------------------------------------------------------

def compute(cfg: dict, timestep_idx: int = -1, config_path: str = "") -> EnergyPartitionResult:
    """Load one dump, call the pure energy-partition functions, return the result."""
    sim_dir = cfg["sim_dir"]
    t_val = cfg["times"][timestep_idx]
    field_mode = cfg.get("field_mode", "full")

    sim = analysis_utils.run_from_config(cfg)
    spec = analysis_utils.RunSpec.from_sim_dir(sim_dir)
    layout = analysis_utils.detect_layout(sim_dir)   # for the savg field-name suffix
    species_list = list(sim.deck.species)            # species defined in the input deck

    # Load phase spaces and fields for this single dump.
    print("Loading HDF5 files...")
    pha = {
        sp: osh5io.read_h5(analysis_utils.diag_path(sim_dir, "p1x1", t_val, sp))
        for sp in species_list
    }
    # Transverse phase spaces (p2x1, p3x1) when output, so the thermal energy
    # carries every available momentum direction.
    pha_perp = {sp: [] for sp in species_list}
    for pdir in ("p2x1", "p3x1"):
        for sp in species_list:
            path = analysis_utils.diag_path(sim_dir, pdir, t_val, sp)
            if os.path.exists(path):
                pha_perp[sp].append(osh5io.read_h5(path))
    n_perp = max((len(v) for v in pha_perp.values()), default=0)
    print(f"Thermal directions: {', '.join(['p1', 'p2', 'p3'][:1 + n_perp])}")
    fld = {
        name: osh5io.read_h5(analysis_utils.diag_path(sim_dir, layout.field_quantity(name), t_val))
        for name in ["b1", "b2", "b3", "e1", "e2", "e3"]
    }

    # Spatial grids: phase-space axes are [p1, x1]; field axes are [x1].
    x_pha = axis_values(pha[species_list[0]], ax_idx=1)
    x_field = axis_values(fld["b1"], ax_idx=0)
    t_sim = float(pha[species_list[0]].run_attrs["TIME"][0])

    dump = analysis_utils.resolve_dump_params(cfg, t_val, t_sim)
    x_shock = dump["x_shock"]
    x_downstream_start = dump["x_downstream_start"]

    # Shock-frame boost velocity is the tuned config shock.v_shock (set with
    # scripts/tune_shock.py), used directly — no auto-fit of the front trajectory.
    v_shock = float(cfg["shock"]["v_shock"])

    species_rqm = {sp: sim.rqm_of(sp) for sp in species_list}  # per-species rqm from deck

    # Sum the per-species kinetic energy profiles (physics: species_energy_profiles).
    u_ram_total = np.zeros(x_pha.size)
    u_th_total = np.zeros(x_pha.size)
    for sp in species_list:
        u_ram_sp, u_th_sp = ep.species_energy_profiles(
            pha[sp], species_rqm[sp], v_shock, perp_phase_spaces=pha_perp[sp]
        )
        u_ram_total += u_ram_sp
        u_th_total += u_th_sp

    # Field energy densities, then interpolate onto the phase-space grid so all
    # channels share the same masks (physics: field_energy_profiles).
    b_arrs = [np.asarray(fld[f"b{i}"]) for i in range(1, 4)]
    e_arrs = [np.asarray(fld[f"e{i}"]) for i in range(1, 4)]
    u_B_fld, u_E_fld = ep.field_energy_profiles(
        *b_arrs, *e_arrs, x_field, x_shock, field_mode=field_mode
    )
    u_B = np.interp(x_pha, x_field, u_B_fld)
    u_E = np.interp(x_pha, x_field, u_E_fld)

    partition = ep.partition_by_region(
        u_ram_total, u_th_total, u_B, u_E, x_pha, x_shock, x_downstream_start,
    )

    return EnergyPartitionResult(
        x_axis=x_pha, u_ram=u_ram_total, u_th=u_th_total, u_B=u_B, u_E=u_E,
        t_val=int(t_val), t_sim=t_sim,
        x_shock=float(x_shock), x_downstream_start=float(x_downstream_start),
        v_shock=float(v_shock),
        field_mode=str(field_mode), norm_density_cm3=float(spec.reference_density),
        config_path=config_path,
        upstream=partition["upstream"], downstream=partition["downstream"],
    )


def _print_summary(r: EnergyPartitionResult) -> None:
    """Console table of the energy partition (no computation)."""
    print(f"t_sim   : {r.t_sim:.1f} [ωpe⁻¹]")
    print(f"x_shock : {r.x_shock:.1f} [c/ωpe]")
    print(f"v_shock : {r.v_shock:.5f} c  (tuned config value)")
    total_up = sum(r.upstream.values())
    total_dn = sum(r.downstream.values())
    print("\n--- Energy partition ---")
    print(f"{'Channel':<12} {'Upstream [sim]':>18} {'(%)':>7}  "
          f"{'Downstream [sim]':>18} {'(%)':>7}")
    for k in ("ram", "thermal", "B_field", "E_field"):
        label = k.replace("_", " ").capitalize()
        print(f"{label:<12} {r.upstream[k]:>18.3e} {100*r.upstream[k]/total_up:>6.1f}%  "
              f"{r.downstream[k]:>18.3e} {100*r.downstream[k]/total_dn:>6.1f}%")


# ---------------------------------------------------------------------------
# Plot (matplotlib only — reads the result, draws nothing new)
# ---------------------------------------------------------------------------

def _plot_profiles(r: EnergyPartitionResult, ax: plt.Axes, disp) -> None:
    x, x_shock, x_ds = r.x_axis, r.x_shock, r.x_downstream_start
    xd = disp.x(x)                       # display coordinates (c/ωpe or d_i)
    u_total = r.u_ram + r.u_th + r.u_B + r.u_E
    ax.semilogy(xd, r.u_ram, label="Ram")
    ax.semilogy(xd, r.u_th, label="Thermal")
    ax.semilogy(xd, r.u_B, label=f"B field ({r.field_mode})")
    ax.semilogy(xd, r.u_E, label="E field")
    ax.semilogy(xd, u_total, color="k", ls="--", lw=1.5, label="Total")
    ax.axvline(disp.x(x_ds), color="gray", ls="--", lw=1, label=f"downstream start {x_ds:.0f}")
    ax.axvline(disp.x(x_shock), color="k", ls="--", lw=1, label=f"shock {x_shock:.1f}")
    ax.axvspan(disp.x(x.min()), disp.x(x_ds), color="gray", alpha=0.12, label="piston")

    ax.set_xlabel(disp.xlabel())
    ax.set_ylabel("Energy density [n₀ mₑ c²]")
    ax.set_title(f"Energy density profiles, dump {r.t_val}  {disp.time_title(r.t_sim)}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xlim(disp.x(x_ds), disp.x(x_shock + 100))

    downstream = (x >= x_ds) & (x <= x_shock)
    if downstream.any():
        ax.set_ylim(0, np.nanmax(u_total[downstream]) * 1.2)


def _plot_partition_bars(r: EnergyPartitionResult, ax: plt.Axes) -> None:
    keys = ["ram", "thermal", "B_field", "E_field"]
    up_vals = [float(r.upstream[k]) for k in keys]
    dn_vals = [float(r.downstream[k]) for k in keys]
    # Fifth group: total energy density, for a direct (linear) conservation check.
    # NB this compares energy *density*, not the shock-frame energy *flux* that is
    # actually conserved across the front (see scripts/energy_flux.py).
    up_vals.append(sum(up_vals))
    dn_vals.append(sum(dn_vals))
    labels = ["Ram", "Thermal", "B field", "E field", "Total"]

    x = np.arange(len(labels))
    w = 0.38
    ax.bar(x - w / 2, up_vals, w, label="Upstream")
    ax.bar(x + w / 2, dn_vals, w, label="Downstream")
    ratio = dn_vals[-1] / up_vals[-1] if up_vals[-1] else float("nan")
    ax.text(x[-1], max(up_vals[-1], dn_vals[-1]) * 1.02,
            f"dn/up = {ratio:.2f}", ha="center", va="bottom", fontsize=9)
    ax.axvline(len(keys) - 0.5, color="0.7", lw=1, ls=":")  # set off the Total group
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean energy density [n₀ mₑ c²]")
    ax.set_title(f"Energy partition ({r.field_mode} B), t={r.t_val}")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)


def plot(r: EnergyPartitionResult, output_dir: str, disp) -> str:
    """Render the two-panel figure (profiles + partition bars) and save a .png."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    _plot_profiles(r, axes[0], disp)
    _plot_partition_bars(r, axes[1])
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"energy_partition_t{r.t_val:06d}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Shock energy partition (compute + plot).")
    parser.add_argument("--config", required=True, help="Path to analysis YAML config.")
    parser.add_argument("--timestep-idx", type=int, default=-1, dest="timestep_idx",
                        help="Index into config times list (default: -1, i.e. last dump).")
    parser.add_argument("--no-plot", action="store_true", dest="no_plot",
                        help="Compute and save the .npz only; skip the figure.")
    parser.add_argument("--output", default=None,
                        help="Output .npz path (default results/<run>/energy_partition_t{t:06d}.npz).")
    parser.add_argument("--output-dir", default=None, dest="output_dir",
                        help="Directory for the figure (default: alongside the .npz).")
    plot_style.add_publication_arg(parser)
    plot_style.add_units_arg(parser)
    args = parser.parse_args()
    plot_style.apply(args.publication)

    cfg = analysis_utils.load_config(args.config)
    sim_dir = cfg["sim_dir"]
    times = cfg["times"]
    t_val = times[args.timestep_idx]
    print(f"Config  : {args.config}")
    print(f"sim_dir : {sim_dir}")
    print(f"Dump    : t={t_val}  (index {args.timestep_idx} of {len(times)} dumps)")

    result = compute(cfg, args.timestep_idx, config_path=os.path.abspath(args.config))
    _print_summary(result)

    out_path = analysis_utils.default_output_path(args.output, sim_dir, "energy_partition", result.t_val)
    analysis_utils.save_result(result, out_path)
    print(f"\nSaved → {out_path}")

    if not args.no_plot:
        disp = plot_style.build_units(args.units, cfg=cfg, config_path=os.path.abspath(args.config))
        out_dir = args.output_dir or os.path.dirname(os.path.abspath(out_path))
        fig_path = plot(result, out_dir, disp)
        print(f"Saved → {fig_path}")


if __name__ == "__main__":
    main()
