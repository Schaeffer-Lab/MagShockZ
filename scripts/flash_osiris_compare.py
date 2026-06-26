"""scripts/flash_osiris_compare.py — FLASH (MHD) vs OSIRIS (PIC) direct comparison.

Compares the FLASH 3D MHD run that seeded an OSIRIS PIC run against that OSIRIS
run itself, along the same line of sight the OSIRIS deck was initialised from.
The OSIRIS run's ``run.yaml`` (read via ``RunSpec``) is the single source of truth
for *which* FLASH dump, line of sight, charge states and reference density were
used, so nothing about the FLASH side is duplicated in the analysis config — pass
the OSIRIS analysis config and the FLASH side is discovered from it.

Two comparisons are produced for one matched pair of dumps:

  1. State profiles (init / state fidelity): n_e, n_i, T_e, T_i and |B| line-outs,
     FLASH vs OSIRIS, in OSIRIS-normalised units along the LOS.
  2. Energy partition between electrons and ions (the headline): the thermal
     energy density u_th = (3/2) n kT split into electrons vs ions, as absolute
     erg/cm³ profiles, the electron thermal fraction f_e(x), and the LOS-integrated
     electron/ion share for each code side by side.

Physics lives in the pure src modules:
  - OSIRIS energy density : ``energy_partition.species_energy_profiles`` (phase-space
    moments).  The OSIRIS phase-space / charge diagnostics deposit *charge* density
    (Z·n_i for ions), so the ion channels are divided by the charge state here to get
    the number-density-weighted thermal energy that matches FLASH's (3/2) n kT.
  - FLASH energy density  : ``flash_energy_partition.energy_densities``.
  - temperatures          : ``temperature_anisotropy.temperature_profile`` (OSIRIS).

Time alignment is by elapsed **ion gyroperiods** (the physically invariant clock),
not by real seconds × ω_pe: the reduced ion mass (``rqm_factor``) runs OSIRIS ion
dynamics ``rqm_factor``× faster relative to ω_pe than reality, so the matched OSIRIS
time is ``elapsed_s · ω_pe / rqm_factor`` (≡ matching N = t/T_ci on both sides).
OSIRIS t=0 is the FLASH IC dump named by ``data_path``; the nearest OSIRIS dump with
every needed diagnostic is used.  Space alignment: the FLASH LOS distance [cm] is
divided by the electron skin depth c/ω_pe, which is exactly how the deck's box length
was set, so FLASH x=0 maps onto OSIRIS x=0.

Usage
-----
    conda activate analysis
    python scripts/flash_osiris_compare.py --config config/magshockz_rqm100_dx0.1.yaml \\
        [--flash-idx 10] [--no-crop] [--output-dir DIR]

``--flash-idx`` defaults to the IC dump (OSIRIS t=0, the initialisation check).
For the dx=0.1 run the OSIRIS run spans ~0.32 ns, so FLASH dumps IC and IC+1 have
genuine time-matched OSIRIS dumps; later FLASH dumps fall past the OSIRIS window.
"""

import argparse
import glob
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import h5py
import osh5io
import astropy.units as u_ap

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))

import analysis_utils
import plot_style
import moments
import temperature_anisotropy as ta
import energy_partition as ep
import flash_utils as fu
import flash_energy_partition as fep
from analysis_utils import axis_values, diag_path

# m_e c^2 in eV and erg — the OSIRIS energy normalisation n_0 m_e c^2 -> erg/cm^3.
ME_C2_EV = 510998.95
EV_ERG = 1.602176634e-12
ME_C2_ERG = ME_C2_EV * EV_ERG

SPECIES_IONS = ("al", "si")
# rqm per species: electrons are -1; ions read from the deck.
_E_RQM = -1.0

# numpy 2.x renamed trapz -> trapezoid; support both.
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _x_axis(d) -> np.ndarray:
    """Coordinate values of the (single) spatial axis of an H5Data."""
    i = next(i for i, a in enumerate(d.axes) if a.name.startswith("x"))
    return axis_values(d, i)


def _mom_axis(pha) -> str:
    """Name of the momentum axis of a p?x? phase space."""
    return next(a.name for a in pha.axes if a.name.startswith("p"))


def _flash_path(flash_dir: str, idx: int) -> str:
    return os.path.join(flash_dir, f"MagShockZ_hdf5_plt_cnt_{idx:04d}")


def _dump_time_wpe(path: str) -> float:
    with h5py.File(path, "r") as f:
        return float(f.attrs["TIME"][0])


# ---------------------------------------------------------------------------
# OSIRIS dump selection — nearest dump that has every diagnostic we read
# ---------------------------------------------------------------------------

def _osiris_indices(sim_dir: str, layout) -> list:
    """Dump indices that have p1/p2/p3 phase spaces, charge density and savg fields
    for every species we read (so a matched dump is fully loadable)."""
    pdir = os.path.dirname(diag_path(sim_dir, layout.pha_name(1), 0, "e"))
    idxs = sorted(int(re.search(r"-(\d{6})\.h5$", f).group(1))
                  for f in glob.glob(os.path.join(pdir, "*.h5")))

    def complete(t):
        need = [diag_path(sim_dir, layout.pha_name(c), t, sp)
                for c in layout.momenta for sp in ("e",) + SPECIES_IONS]
        need += [diag_path(sim_dir, layout.field_quantity(b), t) for b in ("b1", "b2", "b3")]
        need += [diag_path(sim_dir, layout.charge_quantity, t, sp)
                 for sp in ("e",) + SPECIES_IONS]
        return all(os.path.exists(p) for p in need)

    return [t for t in idxs if complete(t)]


def nearest_osiris_dump(sim_dir: str, layout, target_wpe: float):
    """Return (index, time_wpe) of the complete OSIRIS dump closest to ``target_wpe``."""
    idxs = _osiris_indices(sim_dir, layout)
    if not idxs:
        raise FileNotFoundError(f"No fully-diagnosed OSIRIS dumps found under {sim_dir}.")
    times = np.array([_dump_time_wpe(diag_path(sim_dir, layout.charge_quantity, t, "e"))
                      for t in idxs])
    i = int(np.argmin(np.abs(times - target_wpe)))
    return idxs[i], float(times[i])


# ---------------------------------------------------------------------------
# FLASH side — physical CGS lineout + energy partition
# ---------------------------------------------------------------------------

def flash_state(flash_dir: str, idx: int, line_start, line_end, d_e_cm: float,
                B_to_osiris: float, n0_cm3: float) -> dict:
    """FLASH lineout reduced to the comparison quantities, x in c/ω_pe."""
    lo = fu.flash_lineout(_flash_path(flash_dir, idx), line_start, line_end)
    x_cwpe = lo["x"].to("cm").value / d_e_cm

    energy = fep.energy_densities(
        ne=lo["ne"], Te=lo["Te"], Ti=lo["Ti"], n_ion=lo["n_ion"],
        rho=lo["rho"], v_para=lo["v_para"], v_shock=0.0, B_mag=lo["B_mag"],
    )
    return {
        "x": x_cwpe,
        "t_s": lo["t_s"],
        "n_e": lo["ne"].to("cm**-3").value / n0_cm3,
        "n_i": lo["n_ion"].to("cm**-3").value / n0_cm3,
        "T_e": lo["Te"].to("eV").value,
        "T_i": lo["Ti"].to("eV").value,
        "B": lo["B_mag"].to("G").value * B_to_osiris,
        "u_th_e": energy["u_th_e"].to("erg/cm**3").value,
        "u_th_i": energy["u_th_i"].to("erg/cm**3").value,
    }


# ---------------------------------------------------------------------------
# OSIRIS side — densities (full grid), |B| (field grid), T & u_th (phase grid)
# ---------------------------------------------------------------------------

def _species_phase(sim_dir, layout, sp, t):
    """Load (p1, [perp...]) phase spaces for one species."""
    p1 = osh5io.read_h5(diag_path(sim_dir, layout.pha_name(1), t, sp))
    perp = [osh5io.read_h5(diag_path(sim_dir, layout.pha_name(c), t, sp))
            for c in layout.momenta if c != 1]
    return p1, perp


def _species_T_eV(p1, perp, rqm) -> np.ndarray:
    """Isotropic temperature T = mean_d |rqm|<(p_d-<p_d>)^2> in eV, on the phase grid.

    Averages the per-direction temperature from ``temperature_anisotropy.temperature_profile``
    over the available momentum components (p1 + any perp), so T is the eV temperature
    independent of the species number density.
    """
    T_sum = ta.temperature_profile(p1, rqm, "p1")
    dirs = 1
    for pp in perp:
        T_sum = T_sum + ta.temperature_profile(pp, rqm, _mom_axis(pp))
        dirs += 1
    return np.asarray(T_sum / dirs) * ME_C2_EV


def osiris_state(sim_dir, layout, t, ion_rqm, charge_states, n0_cm3) -> dict:
    """All OSIRIS comparison quantities at dump ``t`` (each on its native grid)."""
    # --- densities (full simulation grid); ion diagnostics are charge density -> /Z
    ch = lambda sp: np.abs(np.asarray(osh5io.read_h5(
        diag_path(sim_dir, layout.charge_quantity, t, sp))))
    ne_full = osh5io.read_h5(diag_path(sim_dir, layout.charge_quantity, t, "e"))
    x_dens = _x_axis(ne_full)
    n_e = np.abs(np.asarray(ne_full))
    n_i = sum(ch(sp) / charge_states[sp] for sp in SPECIES_IONS)

    # --- |B| (field grid)
    b = {q: np.asarray(osh5io.read_h5(diag_path(sim_dir, layout.field_quantity(q), t)))
         for q in ("b1", "b2", "b3")}
    bfld = osh5io.read_h5(diag_path(sim_dir, layout.field_quantity("b1"), t))
    x_fld = _x_axis(bfld)
    B = np.sqrt(b["b1"] ** 2 + b["b2"] ** 2 + b["b3"] ** 2)

    # --- temperatures + thermal energy density (phase grid). The phase-space 0th
    # moment is charge density, so ion u_th from species_energy_profiles is divided
    # by Z to give the number-weighted (3/2) n kT that matches FLASH.
    e_p1, e_perp = _species_phase(sim_dir, layout, "e", t)
    x_pha = _x_axis(e_p1)
    T_e = _species_T_eV(e_p1, e_perp, _E_RQM)
    _, u_th_e_norm = ep.species_energy_profiles(e_p1, _E_RQM, 0.0, perp_phase_spaces=e_perp)

    nT_i = np.zeros_like(x_pha)  # number-weighted ion temperature numerator
    n_i_pha = np.zeros_like(x_pha)
    u_th_i_norm = np.zeros_like(x_pha)
    for sp in SPECIES_IONS:
        p1, perp = _species_phase(sim_dir, layout, sp, t)
        Z = charge_states[sp]
        n_sp = np.abs(moments.moment(p1, axis="p1", order=0)) / Z  # number density [n_0]
        T_sp = _species_T_eV(p1, perp, ion_rqm[sp])
        nT_i += n_sp * T_sp
        n_i_pha += n_sp
        _, u_th_sp = ep.species_energy_profiles(p1, ion_rqm[sp], 0.0, perp_phase_spaces=perp)
        u_th_i_norm += u_th_sp / Z
    with np.errstate(divide="ignore", invalid="ignore"):
        T_i = np.where(n_i_pha > 0, nT_i / n_i_pha, np.nan)

    erg = n0_cm3 * ME_C2_ERG  # n_0 m_e c^2 -> erg/cm^3
    return {
        "t_wpe": float(ne_full.run_attrs["TIME"][0]),
        "x_dens": x_dens, "n_e": n_e, "n_i": n_i,
        "x_fld": x_fld, "B": B,
        "x_pha": x_pha, "T_e": T_e, "T_i": T_i,
        "u_th_e": u_th_e_norm * erg, "u_th_i": u_th_i_norm * erg,
    }


# ---------------------------------------------------------------------------
# Partition reduction
# ---------------------------------------------------------------------------

def integrated_partition(u_e, x_e, u_i, x_i) -> dict:
    """LOS-integrated electron / ion thermal energy and the electron fraction."""
    E_e = float(_trapz(np.nan_to_num(u_e), x_e))
    E_i = float(_trapz(np.nan_to_num(u_i), x_i))
    tot = E_e + E_i
    return {"E_e": E_e, "E_i": E_i,
            "frac_e": E_e / tot if tot else float("nan"),
            "frac_i": E_i / tot if tot else float("nan")}


def _fraction_profile(u_e, u_i):
    with np.errstate(divide="ignore", invalid="ignore"):
        tot = u_e + u_i
        return np.where(tot > 0, u_e / tot, np.nan)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _xlim_from_plasma(osi, disp, pad_frac=0.05):
    """Crop window [c/ωpe -> display] around where OSIRIS has plasma (n_e > 5% max)."""
    m = osi["n_e"] > 0.05 * np.nanmax(osi["n_e"])
    if not m.any():
        return None
    x = osi["x_dens"][m]
    lo, hi = float(x.min()), float(x.max())
    pad = pad_frac * (hi - lo)
    return float(disp.x(lo - pad)), float(disp.x(hi + pad))


def plot_profiles(flash, osi, disp, title, out_path, xlim):
    """State-fidelity line-outs: n_e, n_i, T_e, T_i, |B| (FLASH dashed, OSIRIS solid)."""
    fig, ax = plt.subplots(5, 1, figsize=(10, 16), sharex=True)
    fx = disp.x(flash["x"])

    ax[0].plot(fx, flash["n_e"], "k--", label="FLASH")
    ax[0].plot(disp.x(osi["x_dens"]), osi["n_e"], "C0-", label="OSIRIS")
    ax[0].set_ylabel(r"$n_e\ /\ n_0$"); ax[0].legend()

    ax[1].plot(fx, flash["n_i"], "k--")
    ax[1].plot(disp.x(osi["x_dens"]), osi["n_i"], "C1-")
    ax[1].set_ylabel(r"$n_i\ /\ n_0$")

    ax[2].semilogy(fx, flash["T_e"], "k--")
    ax[2].semilogy(disp.x(osi["x_pha"]), osi["T_e"], "C3-")
    ax[2].set_ylabel(r"$T_e$ [eV]")

    ax[3].semilogy(fx, flash["T_i"], "k--")
    ax[3].semilogy(disp.x(osi["x_pha"]), osi["T_i"], "C2-")
    ax[3].set_ylabel(r"$T_i$ [eV]")

    ax[4].plot(fx, flash["B"], "k--")
    ax[4].plot(disp.x(osi["x_fld"]), osi["B"], "C4-")
    ax[4].set_ylabel(r"$|B|$ [OSIRIS units]")
    ax[4].set_xlabel(disp.xlabel())

    for a in ax:
        a.grid(alpha=0.3)
        if xlim:
            a.set_xlim(*xlim)
    fig.suptitle(title + "\n(dashed = FLASH,  solid = OSIRIS)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_energy_partition(flash, osi, part_f, part_o, disp, title, out_path, xlim):
    """Electron-vs-ion thermal energy: profiles, fraction f_e(x), and integrated share."""
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1], sharex=axA)
    axC = fig.add_subplot(gs[1, 0])
    axD = fig.add_subplot(gs[1, 1])

    fx = disp.x(flash["x"])
    ox = disp.x(osi["x_pha"])

    # A — absolute thermal energy density, electrons vs ions, both codes
    axA.semilogy(fx, flash["u_th_e"], "C0--", label=r"FLASH $u_{th,e}$")
    axA.semilogy(fx, flash["u_th_i"], "C3--", label=r"FLASH $u_{th,i}$")
    axA.semilogy(ox, osi["u_th_e"], "C0-", label=r"OSIRIS $u_{th,e}$")
    axA.semilogy(ox, osi["u_th_i"], "C3-", label=r"OSIRIS $u_{th,i}$")
    axA.set_ylabel(r"$u_\mathrm{th}$ [erg cm$^{-3}$]")
    axA.set_xlabel(disp.xlabel()); axA.set_title("Thermal energy density")
    axA.legend(fontsize=8); axA.grid(alpha=0.3, which="both")

    # B — electron thermal fraction f_e(x)
    axB.plot(fx, _fraction_profile(flash["u_th_e"], flash["u_th_i"]), "k--", label="FLASH")
    axB.plot(ox, _fraction_profile(osi["u_th_e"], osi["u_th_i"]), "C0-", label="OSIRIS")
    axB.axhline(0.5, color="0.6", ls=":", lw=1)
    axB.set_ylabel(r"$f_e = u_{th,e}/(u_{th,e}+u_{th,i})$")
    axB.set_xlabel(disp.xlabel()); axB.set_title("Electron thermal fraction")
    axB.set_ylim(0, 1); axB.legend(fontsize=9); axB.grid(alpha=0.3)

    if xlim:
        axA.set_xlim(*xlim); axB.set_xlim(*xlim)

    # C — LOS-integrated electron/ion thermal share, FLASH vs OSIRIS
    codes = ["FLASH", "OSIRIS"]
    fe = [part_f["frac_e"], part_o["frac_e"]]
    fi = [part_f["frac_i"], part_o["frac_i"]]
    xpos = np.arange(len(codes)); w = 0.55
    axC.bar(xpos, fe, w, color="C0", label="electrons")
    axC.bar(xpos, fi, w, bottom=fe, color="C3", label="ions")
    for i, (e, ion) in enumerate(zip(fe, fi)):
        axC.text(i, e / 2, f"{100*e:.0f}%", ha="center", va="center", color="w", fontsize=11)
        axC.text(i, e + ion / 2, f"{100*ion:.0f}%", ha="center", va="center", color="w", fontsize=11)
    axC.set_xticks(xpos); axC.set_xticklabels(codes)
    axC.set_ylabel("LOS-integrated thermal energy share")
    axC.set_title("Energy partition (e vs i)"); axC.legend(fontsize=9)
    axC.set_ylim(0, 1)

    # D — text summary
    axD.axis("off")
    lines = [
        "LOS-integrated thermal energy",
        f"  FLASH :  e {100*part_f['frac_e']:5.1f}%   i {100*part_f['frac_i']:5.1f}%",
        f"  OSIRIS:  e {100*part_o['frac_e']:5.1f}%   i {100*part_o['frac_i']:5.1f}%",
        "",
        f"  E_th,e  FLASH {part_f['E_e']:.3e}   OSIRIS {part_o['E_e']:.3e}",
        f"  E_th,i  FLASH {part_f['E_i']:.3e}   OSIRIS {part_o['E_i']:.3e}",
        "  (∫u_th dx, erg cm⁻³ · c/ω_pe)",
    ]
    axD.text(0.02, 0.95, "\n".join(lines), va="top", ha="left",
             family="monospace", fontsize=11, transform=axD.transAxes)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Direct FLASH (MHD) vs OSIRIS (PIC) comparison along the LOS.")
    parser.add_argument("--config", required=True,
                        help="OSIRIS analysis YAML config (the FLASH side is read from "
                             "the run's run.yaml via RunSpec).")
    parser.add_argument("--flash-idx", type=int, default=None, dest="flash_idx",
                        help="FLASH plot-file index to compare (default: the IC dump "
                             "named by data_path, i.e. OSIRIS t=0).")
    parser.add_argument("--no-crop", action="store_true", dest="no_crop",
                        help="Plot the full LOS instead of cropping to the plasma region.")
    parser.add_argument("--output-dir", default=None, dest="output_dir")
    plot_style.add_publication_arg(parser)
    plot_style.add_units_arg(parser)
    args = parser.parse_args()
    plot_style.apply(args.publication)

    cfg = analysis_utils.load_config(args.config)
    sim_dir = cfg["sim_dir"]
    spec = analysis_utils.RunSpec.from_sim_dir(sim_dir)
    layout = analysis_utils.detect_layout(sim_dir)
    disp = plot_style.build_units_from_args(args, cfg)

    # FLASH side, all from the run spec (single source of truth)
    data_path = spec["data_path"]
    flash_dir = os.path.dirname(data_path)
    ic_idx = int(os.path.basename(data_path)[-4:])
    flash_idx = args.flash_idx if args.flash_idx is not None else ic_idx
    line_start = tuple(float(v) for v in spec["start_point"])
    line_end = tuple(float(v) for v in spec["end_point"])
    charge_states = {sp: spec.charge_state(sp) for sp in SPECIES_IONS}

    # Unit conversions from the run context
    run = analysis_utils.run_from_config(cfg)
    wpe_hz = float(run.omega_pe.to_value(u_ap.rad / u_ap.s))
    d_e_cm = float(run.d_e().to_value(u_ap.cm))
    B_to_osiris = float(run.B_norm(1.0 * u_ap.Gauss))
    n0_cm3 = spec.reference_density
    ion_rqm = {sp: run.rqm_of(sp) for sp in SPECIES_IONS}

    # Time alignment by ELAPSED ION GYROPERIODS — the physically invariant clock.
    # The reduced ion mass (rqm_factor) runs OSIRIS ion dynamics rqm_factor× faster
    # relative to ω_pe than reality, so naively mapping real seconds via ω_pe over- or
    # under-shoots the shock evolution by rqm_factor.  Instead match the number of ion
    # gyroperiods elapsed: N = t / T_ci.  Because both sims share B' and deck structure,
    # the OSIRIS/real gyroperiod ratio is exactly rqm_factor, so the matched OSIRIS time
    # reduces to target_wpe = elapsed_s · ω_pe / rqm_factor (T_ci cancels); T_ci is still
    # used to report N_gyro for context (cached config t_ci, else measured at t=0).
    rqm_factor = spec.rqm_factor or 1.0
    t_ci_wpe = cfg.get("t_ci")
    if t_ci_wpe is None:
        from dimensionless_params import ion_gyroperiod
        t_ci_wpe = ion_gyroperiod(abs(run.rqm), analysis_utils.upstream_field_magnitude(cfg))
    t_ci_wpe = float(t_ci_wpe)
    T_ci_real_s = rqm_factor * t_ci_wpe / wpe_hz   # physical ion gyroperiod [s]

    t_ic_s = fu.flash_time_s(_flash_path(flash_dir, ic_idx))
    t_flash_s = fu.flash_time_s(_flash_path(flash_dir, flash_idx))
    elapsed_s = t_flash_s - t_ic_s
    n_gyro = elapsed_s / T_ci_real_s                # ion gyroperiods elapsed since IC
    target_wpe = n_gyro * t_ci_wpe                  # = elapsed_s · ω_pe / rqm_factor
    t_osiris, t_osiris_wpe = nearest_osiris_dump(sim_dir, layout, target_wpe)
    n_gyro_osiris = t_osiris_wpe / t_ci_wpe

    flash_name = os.path.basename(flash_dir.rstrip("/"))
    osiris_name = os.path.basename(sim_dir.rstrip("/"))
    out_dir = args.output_dir or os.path.join(
        _HERE, "..", "results", f"compare_{flash_name}__vs__{osiris_name}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Config        : {args.config}")
    print(f"OSIRIS run    : {osiris_name}")
    print(f"FLASH run     : {flash_name}  (IC dump {ic_idx:04d})")
    print(f"FLASH dump    : {flash_idx:04d}  (t = {t_flash_s*1e9:.3f} ns, "
          f"{elapsed_s*1e9:+.3f} ns = {n_gyro:+.4f} T_ci since IC)")
    print(f"gyro clock    : T_ci = {t_ci_wpe:.1f}/wpe (OSIRIS) = {T_ci_real_s*1e9:.3f} ns "
          f"(real, ×rqm_factor {rqm_factor:g})")
    print(f"target OSIRIS : {target_wpe:.1f} 1/wpe  ({n_gyro:.4f} T_ci)")
    print(f"nearest dump  : {t_osiris:06d}  (t = {t_osiris_wpe:.1f}/wpe = "
          f"{n_gyro_osiris:.4f} T_ci, Δ = {abs(n_gyro - n_gyro_osiris):.4f} T_ci)")
    print(f"c/ω_pe        : {d_e_cm*1e4:.4f} µm    n_0 = {n0_cm3:.3e} cm⁻³")
    print(f"charge states : {charge_states}    ion rqm = "
          f"{ {k: round(v,3) for k,v in ion_rqm.items()} }")

    print("\nLoading FLASH lineout …", flush=True)
    flash = flash_state(flash_dir, flash_idx, line_start, line_end,
                        d_e_cm, B_to_osiris, n0_cm3)
    print("Loading OSIRIS dump …", flush=True)
    osi = osiris_state(sim_dir, layout, t_osiris, ion_rqm, charge_states, n0_cm3)

    part_f = integrated_partition(flash["u_th_e"], flash["x"], flash["u_th_i"], flash["x"])
    part_o = integrated_partition(osi["u_th_e"], osi["x_pha"], osi["u_th_i"], osi["x_pha"])

    print("\n--- LOS-integrated thermal energy partition (electrons / ions) ---")
    print(f"  FLASH : e {100*part_f['frac_e']:5.1f}%   i {100*part_f['frac_i']:5.1f}%")
    print(f"  OSIRIS: e {100*part_o['frac_e']:5.1f}%   i {100*part_o['frac_i']:5.1f}%")

    xlim = None if args.no_crop else _xlim_from_plasma(osi, disp)
    tag = f"flash{flash_idx:04d}_osiris{t_osiris:06d}"
    title = (f"FLASH {flash_name} #{flash_idx:04d}  vs  OSIRIS {osiris_name}\n"
             f"matched at {n_gyro:.4f} ion gyroperiods since IC  "
             f"(FLASH t={t_flash_s*1e9:.3f} ns, OSIRIS t={t_osiris_wpe:.0f}/wpe)")

    prof_path = os.path.join(out_dir, f"profiles_{tag}.png")
    plot_profiles(flash, osi, disp, title, prof_path, xlim)
    print(f"\nSaved → {prof_path}")

    part_path = os.path.join(out_dir, f"energy_partition_{tag}.png")
    plot_energy_partition(flash, osi, part_f, part_o, disp, title, part_path, xlim)
    print(f"Saved → {part_path}")

    npz_path = os.path.join(out_dir, f"compare_{tag}.npz")
    np.savez(
        npz_path,
        flash_idx=flash_idx, osiris_dump=t_osiris,
        t_flash_ns=t_flash_s * 1e9, t_osiris_wpe=t_osiris_wpe,
        n_gyro_flash=n_gyro, n_gyro_osiris=n_gyro_osiris,
        t_ci_wpe=t_ci_wpe, T_ci_real_s=T_ci_real_s, rqm_factor=rqm_factor,
        target_wpe=target_wpe,
        flash_x=flash["x"], flash_u_th_e=flash["u_th_e"], flash_u_th_i=flash["u_th_i"],
        osiris_x_pha=osi["x_pha"], osiris_u_th_e=osi["u_th_e"], osiris_u_th_i=osi["u_th_i"],
        flash_frac_e=part_f["frac_e"], flash_frac_i=part_f["frac_i"],
        osiris_frac_e=part_o["frac_e"], osiris_frac_i=part_o["frac_i"],
        config_path=os.path.abspath(args.config),
    )
    print(f"Saved → {npz_path}")


if __name__ == "__main__":
    main()
