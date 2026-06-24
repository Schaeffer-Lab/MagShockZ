"""synthetic_diagnostics — forward-model the experiment's observables from a dump.

Science
-------
The MagShockZ experiment measures the shock through **density imaging**,
**magnetic-field probes/radiography**, and **X-ray emission** (there is no Thomson
scattering, so T_e is constrained only through the T_e-sensitive X-ray signal).
This projects one OSIRIS dump onto those signals at the instrument's spatial
resolution, so the run can be compared with data:

    X-ray emissivity   bremsstrahlung, weighted Σ n_s Z_s² √T_e
    density / |B|       degraded to the instrument FWHM
    column density      ∫ n_e dl along the shock normal (interferometry)
    B-probe             |B| sampled at upstream / shock / downstream points

All forward models are owned by the pure, unit-tested module
``src/synthetic_diagnostics.py`` (``bremsstrahlung_emissivity``,
``apply_resolution``, ``line_of_sight_integral``, ``probe_signal``); per-species
densities come from ``src/moments.py``.  In 1D the line of sight is the shock
normal, so the products are resolution-degraded profiles plus the column density;
true 2D images come with the 2D port.

Validation
----------
This file is orchestration + plotting only; it contains no physics formulas.  To
validate a number, read the named function in ``src/synthetic_diagnostics.py`` and
its test in ``tests/test_synthetic_diagnostics.py``.  The computed result is saved
to an inspectable ``.npz`` whose schema is the ``SyntheticDiagnosticsResult``
dataclass below.

Usage
-----
    python scripts/synthetic_diagnostics.py --config config/<run>.yaml \\
        [--timestep-idx -1] [--resolution-um 50] [--no-plot] \\
        [--half-window 300] [--output ...] [--output-dir ...]
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
import plot_style
import shock_state
import moments as mom
import synthetic_diagnostics as sd


# ---------------------------------------------------------------------------
# Output schema (this dataclass IS the .npz schema; see analysis_utils.save_result)
# ---------------------------------------------------------------------------

@dataclass
class SyntheticDiagnosticsResult:
    # --- shock-normal profiles: true (full-res) and instrument-degraded ---
    x_axis: np.ndarray            # [c/wpe]
    n_e: np.ndarray               # electron density, true [n_0]
    n_e_obs: np.ndarray           # electron density at instrument resolution [n_0]
    Bmag: np.ndarray              # |B|, true [B_0]
    Bmag_obs: np.ndarray          # |B| at instrument resolution [B_0]
    xray_emissivity: np.ndarray   # bremsstrahlung emissivity, true [rel]
    xray_obs: np.ndarray          # emissivity at instrument resolution [rel]
    T_e: np.ndarray               # electron temperature [m_e c^2]
    # --- B-probe samples ---
    probe_x: np.ndarray           # probe locations [up, shock, dn] [c/wpe]
    B_probe: np.ndarray           # |B| sampled at probe_x (at resolution) [B_0]
    # --- scalars / provenance ---
    x_shock: float                # shock position [c/wpe]
    column_density: float         # interferometry ∫ n_e dl [n_0 · c/wpe]
    resolution_um: float          # instrument FWHM [µm]
    resolution_norm: float        # instrument FWHM [c/wpe]
    t_val: int                    # dump index
    t_sim: float                  # simulation time [1/wpe]
    config_path: str              # absolute path of the analysis config used


# ---------------------------------------------------------------------------
# Compute (orchestration only — every formula lives in src/synthetic_diagnostics.py)
# ---------------------------------------------------------------------------

def compute(cfg: dict, timestep_idx: int = -1, resolution_um: float = 50.0,
            config_path: str = "") -> SyntheticDiagnosticsResult:
    """Load one dump, run the forward models at ``resolution_um``, return the result."""
    print("Loading HDF5 files...")
    st = shock_state.load_shock_state(cfg, timestep_idx, species=("al", "si", "e"))

    # Per-species ion densities (sim ions carry unit charge => n_s in n_0); the
    # physical charge states weight the bremsstrahlung emissivity as n_s Z_s^2.
    Z = {sp: st.spec.charge_state(sp) for sp in ("al", "si")}
    n = {sp: np.abs(mom.moment(st.pha_p1[sp], axis="p1", order=0)) for sp in ("al", "si")}
    n_e = st.n_e
    T_e = st.T_iso["e"]
    Bmag = np.sqrt(st.B2)
    x = st.x_pha

    # Instrument resolution in normalised units: res[c/wpe] = res[cm] / d_e[cm].
    d_e_cm = float(st.sim.d_e().to("cm").value)
    res_norm = (resolution_um * 1e-4) / d_e_cm

    # Forward models (relative units).
    emiss = sd.bremsstrahlung_emissivity(
        n_e, T_e, species=[(n["al"], Z["al"]), (n["si"], Z["si"])])
    ne_obs = sd.apply_resolution(n_e, x, res_norm)
    B_obs = sd.apply_resolution(Bmag, x, res_norm)
    xray_obs = sd.apply_resolution(emiss, x, res_norm)

    # Interferometry column density (full LOS along the shock normal).
    column_density = float(sd.line_of_sight_integral(n_e, x))

    # B-probe samples at upstream / shock / downstream.
    probe_x = np.array([st.x_shock + 100.0, st.x_shock, st.x_shock - 100.0])
    B_probe = sd.probe_signal(Bmag, x, probe_x, fwhm=res_norm)

    return SyntheticDiagnosticsResult(
        x_axis=x, n_e=n_e, n_e_obs=ne_obs, Bmag=Bmag, Bmag_obs=B_obs,
        xray_emissivity=emiss, xray_obs=xray_obs, T_e=T_e,
        probe_x=probe_x, B_probe=B_probe,
        x_shock=float(st.x_shock), column_density=column_density,
        resolution_um=float(resolution_um), resolution_norm=float(res_norm),
        t_val=int(st.t_val), t_sim=float(st.t_sim), config_path=config_path,
    )


def _print_summary(r: SyntheticDiagnosticsResult) -> None:
    """Console table of instrument smearing near the shock (no computation)."""
    d_e_um = r.resolution_um / r.resolution_norm if r.resolution_norm else float("nan")
    print(f"\nInstrument resolution: {r.resolution_um:.0f} µm = {r.resolution_norm:.1f} c/ωpe "
          f"(d_e = {d_e_um:.2f} µm)")
    print(f"Interferometry column density (∫n_e dl) = {r.column_density:.3e} [n_0 · c/ωpe]")

    def peak_near_shock(arr, half=300.0):
        m = (r.x_axis > r.x_shock - half) & (r.x_axis < r.x_shock + half)
        return float(np.nanmax(arr[m])) if m.any() else float("nan")

    print(f"\n{'quantity':<14} {'peak (full)':>14} {'peak (instr.)':>14} {'smearing':>10}")
    for label, full, obs in (("n_e", r.n_e, r.n_e_obs), ("|B|", r.Bmag, r.Bmag_obs),
                             ("X-ray", r.xray_emissivity, r.xray_obs)):
        pf, po = peak_near_shock(full), peak_near_shock(obs)
        print(f"  {label:<12} {pf:>14.3e} {po:>14.3e} {100*(1-po/pf):>9.1f}%")
    print(f"\nB-probe |B| at [up, shock, dn] = "
          f"[{r.B_probe[0]:.4f}, {r.B_probe[1]:.4f}, {r.B_probe[2]:.4f}] [B_0]")


# ---------------------------------------------------------------------------
# Plot (matplotlib only — reads the result, draws nothing new)
# ---------------------------------------------------------------------------

def plot(r: SyntheticDiagnosticsResult, output_dir: str, disp, half_window: float = 300.0) -> str:
    """Render the true-vs-instrument profiles (n_e, |B|, X-ray) and save a .png."""
    x, xs = r.x_axis, r.x_shock
    m = (x >= xs - half_window) & (x <= xs + half_window)
    xd = disp.x(x)

    panels = [
        ("n_e", "n_e [n₀]", r.n_e, r.n_e_obs),
        ("|B|", "|B| [B₀]", r.Bmag, r.Bmag_obs),
        ("X-ray", "emissivity [rel]", r.xray_emissivity, r.xray_obs),
    ]
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    for a, (name, ylab, full, obs) in zip(ax, panels):
        a.plot(xd[m], full[m], color="0.6", lw=1, label="true (full res)")
        a.plot(xd[m], obs[m], color="C3", lw=2, label=f"instrument ({r.resolution_um:.0f} µm)")
        a.axvline(disp.x(xs), color="0.4", ls="--", lw=1)
        a.set_xlabel(disp.xlabel()); a.set_ylabel(ylab); a.set_title(name)
        a.legend(fontsize=8); a.grid(alpha=0.3)

    # Mark the B-probe sample points on the |B| panel.
    ax[1].scatter(disp.x(r.probe_x), r.B_probe, color="k", zorder=5, label="B-probe")

    fig.suptitle(f"Synthetic diagnostics  dump {r.t_val}  {disp.time_title(r.t_sim)}  "
                 f"resolution {r.resolution_um:.0f} µm = {r.resolution_norm:.1f} c/ωpe",
                 fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"synthetic_diagnostics_t{r.t_val:06d}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Synthetic experimental diagnostics (compute + plot).")
    parser.add_argument("--config", required=True, help="Path to analysis YAML config.")
    parser.add_argument("--timestep-idx", type=int, default=-1, dest="timestep_idx",
                        help="Index into config times list (default: -1, last dump).")
    parser.add_argument("--resolution-um", type=float, default=50.0, dest="resolution_um",
                        help="Instrument spatial resolution (FWHM) in microns.")
    parser.add_argument("--no-plot", action="store_true", dest="no_plot",
                        help="Compute and save the .npz only; skip the figure.")
    parser.add_argument("--half-window", type=float, default=300.0, dest="half_window",
                        help="Half-width [c/ωpe] of the near-shock plotting window.")
    parser.add_argument("--output", default=None,
                        help="Output .npz path (default results/<run>/synthetic_diagnostics_t{t:06d}.npz).")
    parser.add_argument("--output-dir", default=None, dest="output_dir",
                        help="Directory for the figure (default: alongside the .npz).")
    plot_style.add_publication_arg(parser)
    plot_style.add_units_arg(parser)
    args = parser.parse_args()
    plot_style.apply(args.publication)

    cfg = analysis_utils.load_config(args.config)
    print(f"Config  : {args.config}\nsim_dir : {cfg['sim_dir']}")

    result = compute(cfg, args.timestep_idx, args.resolution_um,
                     config_path=os.path.abspath(args.config))
    _print_summary(result)

    out_path = analysis_utils.default_output_path(args.output, cfg["sim_dir"],
                                                  "synthetic_diagnostics", result.t_val)
    analysis_utils.save_result(result, out_path)
    print(f"\nSaved → {out_path}")

    if not args.no_plot:
        disp = plot_style.build_units(args.units, cfg=cfg, config_path=os.path.abspath(args.config))
        out_dir = args.output_dir or os.path.dirname(os.path.abspath(out_path))
        fig_path = plot(result, out_dir, disp, args.half_window)
        print(f"Saved → {fig_path}")


if __name__ == "__main__":
    main()
