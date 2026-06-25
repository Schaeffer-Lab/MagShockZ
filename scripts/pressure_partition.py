"""energy_partition — how a collisionless shock partitions the (conserved) momentum flux.

Science
-------
Energy *density* is not conserved across a shock (it jumps because the flow
compresses and slows), so this script reports the **momentum flux** (total
pressure) instead — the quantity that IS continuous across a steady front,
``[ρU² + P_xx + B_t²/8π] = 0`` (the normal Rankine--Hugoniot jump condition).
For one OSIRIS dump it splits the shock-frame normal momentum flux (OSIRIS units,
n_0 m_e c^2) into four pressure channels, upstream vs. downstream of the front:

    p_ram    ram (dynamic) pressure        n |rqm| (<u> - v_shock)^2   (summed over species)
    p_th_e   electron shock-normal pressure P_xx = n_e |rqm_e| sigma_p1^2
    p_th_i   ion shock-normal pressure      P_xx = n_i |rqm_i| sigma_p1^2
    p_mag    transverse magnetic pressure   (b2^2 + b3^2) / 2

The thermal channels use the shock-NORMAL pressure P_xx (the xx component of the
pressure tensor), not the isotropic/trace pressure: only P_xx enters the normal
momentum balance (it reduces to the scalar p when the plasma is isotropic).  The
magnetic channel uses the transverse field only (the normal field b1 is
continuous and carries tension, not normal momentum flux).

Every formula above is owned by the pure, unit-tested module
``src/energy_partition.py`` (``species_momentum_fluxes`` for p_ram/p_th,
``transverse_magnetic_pressure`` for p_mag, ``momentum_partition_by_region`` for
the region averages, ``continuity_check`` for dn/up).  See CLAUDE.md's OSIRIS
units table.

Validation
----------
This file is orchestration + plotting only; it contains no physics formulas.
To validate a number, read the named function in ``src/energy_partition.py`` and
its test in ``tests/test_energy_partition.py``.  The computed result is also
saved to an inspectable ``.npz`` whose schema is the ``PressurePartitionResult``
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import osh5io

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

import analysis_utils
import plot_style
from analysis_utils import axis_values
import energy_partition as ep

_CHANNELS = ("p_ram", "p_th_e", "p_th_i", "p_mag")
_LABELS = ("Ram", "Thermal e⁻", "Thermal i⁺", "Magnetic")


# ---------------------------------------------------------------------------
# Output schema (this dataclass IS the .npz schema; see analysis_utils.save_result)
# ---------------------------------------------------------------------------

@dataclass
class PressurePartitionResult:
    # --- momentum-flux (pressure) profiles on the phase-space grid [n_0 m_e c^2] ---
    x_axis: np.ndarray            # [c/wpe]
    p_ram: np.ndarray             # ram pressure  n|rqm|U²
    p_th_e: np.ndarray            # electron shock-normal pressure P_xx
    p_th_i: np.ndarray            # ion shock-normal pressure P_xx
    p_mag: np.ndarray             # transverse magnetic pressure ½(b2²+b3²)
    # --- shock kinematics / provenance ---
    t_val: int                    # dump index (file suffix)
    t_sim: float                  # simulation time [1/wpe]
    x_shock: float                # shock-front position [c/wpe]
    x_downstream_start: float     # left edge of downstream region [c/wpe]
    v_shock: float                # shock-frame boost velocity [c] (tuned config value)
    norm_density_cm3: float       # reference density n_0 [cm^-3]
    config_path: str              # absolute path of the analysis config used
    # --- region averages: {means, fractions, total} per side [n_0 m_e c^2] ---
    upstream: dict
    downstream: dict
    continuity_ratio: float       # total dn/up (≈1 if conserved)
    rel_imbalance: float          # (dn−up)/up


# ---------------------------------------------------------------------------
# Compute (orchestration only — every formula lives in src/energy_partition.py)
# ---------------------------------------------------------------------------

def compute(cfg: dict, timestep_idx: int = -1, config_path: str = "") -> PressurePartitionResult:
    """Load one dump, call the pure momentum-flux functions, return the result."""
    sim_dir = cfg["sim_dir"]
    t_val = cfg["times"][timestep_idx]

    sim = analysis_utils.run_from_config(cfg)
    spec = analysis_utils.RunSpec.from_sim_dir(sim_dir)
    layout = analysis_utils.detect_layout(sim_dir)   # for the savg field-name suffix
    species_list = list(sim.deck.species)            # species defined in the input deck

    # Load the shock-normal phase spaces (p1x1) per species and the transverse
    # field (b2, b3).  P_xx needs only p1, so no p2x1/p3x1 are required.
    print("Loading HDF5 files...")
    pha = {
        sp: osh5io.read_h5(analysis_utils.diag_path(sim_dir, "p1x1", t_val, sp))
        for sp in species_list
    }
    fld = {
        name: osh5io.read_h5(analysis_utils.diag_path(sim_dir, layout.field_quantity(name), t_val))
        for name in ("b2", "b3")
    }

    # Spatial grids: phase-space axes are [p1, x1]; field axes are [x1].
    x_pha = axis_values(pha[species_list[0]], ax_idx=1)
    x_field = axis_values(fld["b2"], ax_idx=0)
    t_sim = float(pha[species_list[0]].run_attrs["TIME"][0])

    dump = analysis_utils.resolve_dump_params(cfg, t_val, t_sim)
    x_shock = dump["x_shock"]
    x_downstream_start = dump["x_downstream_start"]

    # Shock-frame boost velocity is the tuned config shock.v_shock (set with
    # scripts/tune_shock.py), used directly — no auto-fit of the front trajectory.
    v_shock = float(cfg["shock"]["v_shock"])

    species_rqm = {sp: sim.rqm_of(sp) for sp in species_list}  # per-species rqm from deck

    # Ram pressure summed over species; thermal split electron vs ion (the
    # shock-normal pressure P_xx).  physics: species_momentum_fluxes.
    p_ram = np.zeros(x_pha.size)
    p_th_e = np.zeros(x_pha.size)
    p_th_i = np.zeros(x_pha.size)
    for sp in species_list:
        p_ram_sp, p_th_sp = ep.species_momentum_fluxes(pha[sp], species_rqm[sp], v_shock)
        p_ram += p_ram_sp
        # Electron has |rqm|=1; every ion has |rqm|≫1, so a generous threshold
        # cleanly separates the two thermal channels.
        if abs(species_rqm[sp]) < 10.0:
            p_th_e += p_th_sp
        else:
            p_th_i += p_th_sp

    # Transverse magnetic pressure, interpolated onto the phase-space grid so all
    # channels share the same masks.  physics: transverse_magnetic_pressure.
    p_mag_fld = ep.transverse_magnetic_pressure(np.asarray(fld["b2"]), np.asarray(fld["b3"]))
    p_mag = np.interp(x_pha, x_field, p_mag_fld)

    part = ep.momentum_partition_by_region(
        {"p_ram": p_ram, "p_th_e": p_th_e, "p_th_i": p_th_i, "p_mag": p_mag},
        x_pha, x_shock, x_downstream_start,
    )
    cont = ep.continuity_check(part)

    return PressurePartitionResult(
        x_axis=x_pha, p_ram=p_ram, p_th_e=p_th_e, p_th_i=p_th_i, p_mag=p_mag,
        t_val=int(t_val), t_sim=t_sim,
        x_shock=float(x_shock), x_downstream_start=float(x_downstream_start),
        v_shock=float(v_shock), norm_density_cm3=float(spec.reference_density),
        config_path=config_path,
        upstream=part["upstream"], downstream=part["downstream"],
        continuity_ratio=float(cont["ratio"]), rel_imbalance=float(cont["rel_imbalance"]),
    )


def _print_summary(r: PressurePartitionResult) -> None:
    """Console table of the momentum-flux (pressure) partition (no computation)."""
    print(f"t_sim   : {r.t_sim:.1f} [ωpe⁻¹]")
    print(f"x_shock : {r.x_shock:.1f} [c/ωpe]")
    print(f"v_shock : {r.v_shock:.5f} c  (tuned config value)")
    mu, md = r.upstream, r.downstream
    print("\n--- Momentum-flux (pressure) partition  (CONSERVED: dn/up should be ≈ 1) ---")
    print(f"{'Channel':<12} {'Upstream [sim]':>18} {'(%)':>7}  "
          f"{'Downstream [sim]':>18} {'(%)':>7}")
    for k, lbl in zip(_CHANNELS, _LABELS):
        print(f"{lbl:<12} {mu['means'][k]:>18.3e} {100*mu['fractions'][k]:>6.1f}%  "
              f"{md['means'][k]:>18.3e} {100*md['fractions'][k]:>6.1f}%")
    print(f"{'Total':<12} {mu['total']:>18.3e} {'':>7}  {md['total']:>18.3e}")
    print(f"  dn/up = {r.continuity_ratio:.3f}   ({100*r.rel_imbalance:+.1f}%)")


# ---------------------------------------------------------------------------
# Plot (matplotlib only — reads the result, draws nothing new)
# ---------------------------------------------------------------------------

def _plot_profiles(r: PressurePartitionResult, ax: plt.Axes, disp) -> None:
    x, x_shock, x_ds = r.x_axis, r.x_shock, r.x_downstream_start
    xd = disp.x(x)                       # display coordinates (c/ωpe or d_i)
    p_total = r.p_ram + r.p_th_e + r.p_th_i + r.p_mag
    ax.semilogy(xd, r.p_ram, label="Ram")
    ax.semilogy(xd, r.p_th_e, label="Thermal e⁻")
    ax.semilogy(xd, r.p_th_i, label="Thermal i⁺")
    ax.semilogy(xd, r.p_mag, label="Magnetic")
    ax.semilogy(xd, p_total, color="k", ls="--", lw=1.5, label="Total")
    ax.axvline(disp.x(x_ds), color="gray", ls="--", lw=1, label=f"downstream start {x_ds:.0f}")
    ax.axvline(disp.x(x_shock), color="k", ls="--", lw=1, label=f"shock {x_shock:.1f}")
    ax.axvspan(disp.x(x.min()), disp.x(x_ds), color="gray", alpha=0.12, label="piston")

    ax.set_xlabel(disp.xlabel())
    ax.set_ylabel("Momentum flux [n₀ mₑ c²]")
    ax.set_title(f"Momentum-flux profiles, dump {r.t_val}  {disp.time_title(r.t_sim)}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xlim(disp.x(x_ds), disp.x(x_shock + 100))

    downstream = (x >= x_ds) & (x <= x_shock)
    if downstream.any():
        top = np.nanmax(p_total[downstream])
        if top > 0:
            ax.set_ylim(top * 1e-4, top * 1.5)


def _plot_bars(r: PressurePartitionResult, ax: plt.Axes) -> None:
    """Momentum-flux continuity: up vs dn per channel + total (the conserved sum)."""
    labels = list(_LABELS) + ["Total"]
    up_vals = [float(r.upstream["means"][k]) for k in _CHANNELS] + [float(r.upstream["total"])]
    dn_vals = [float(r.downstream["means"][k]) for k in _CHANNELS] + [float(r.downstream["total"])]

    x = np.arange(len(labels))
    w = 0.38
    ax.bar(x - w / 2, up_vals, w, label="Upstream")
    ax.bar(x + w / 2, dn_vals, w, label="Downstream")
    ax.text(x[-1], max(up_vals[-1], dn_vals[-1]) * 1.02,
            f"dn/up = {r.continuity_ratio:.2f}\n({100*r.rel_imbalance:+.0f}%)",
            ha="center", va="bottom", fontsize=9)
    ax.axvline(len(_CHANNELS) - 0.5, color="0.7", lw=1, ls=":")  # set off the Total group
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean momentum flux [n₀ mₑ c²]")
    ax.set_title(f"Momentum-flux continuity (conserved if dn/up≈1), t={r.t_val}")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)


def plot(r: PressurePartitionResult, output_dir: str, disp) -> str:
    """Render the two-panel figure (pressure profiles + continuity bars), save a .png."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    _plot_profiles(r, axes[0], disp)
    _plot_bars(r, axes[1])
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"pressure_partition_t{r.t_val:06d}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Shock momentum-flux (pressure) partition + continuity check.")
    parser.add_argument("--config", required=True, help="Path to analysis YAML config.")
    parser.add_argument("--timestep-idx", type=int, default=-1, dest="timestep_idx",
                        help="Index into config times list (default: -1, i.e. last dump).")
    parser.add_argument("--no-plot", action="store_true", dest="no_plot",
                        help="Compute and save the .npz only; skip the figure.")
    parser.add_argument("--output", default=None,
                        help="Output .npz path (default results/<run>/pressure_partition_t{t:06d}.npz).")
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

    out_path = analysis_utils.default_output_path(args.output, sim_dir, "pressure_partition", result.t_val)
    analysis_utils.save_result(result, out_path)
    print(f"\nSaved → {out_path}")

    if not args.no_plot:
        disp = plot_style.build_units_from_args(args, cfg)
        out_dir = args.output_dir or os.path.dirname(os.path.abspath(out_path))
        fig_path = plot(result, out_dir, disp)
        print(f"Saved → {fig_path}")


if __name__ == "__main__":
    main()
