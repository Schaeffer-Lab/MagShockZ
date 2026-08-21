# -*- coding: utf-8 -*-
"""scripts/flash_laser_audit.py — audit FLASH's laser ray-trace and energy deposition.

Answers one question: is the heating and ionization the beam produces *before* it
reaches the target real inverse-bremsstrahlung absorption, or an artifact of the ray
trace / deposition numerics?  Four independent checks, each usable on its own:

  ``energy``      FLASH's own ``<basenm>_LaserEnergyProfile.dat`` (written every step by
                  ed_printEnergyStamp, no runtime switch needed): laser energy pumped
                  into the domain vs. energy carried back out by rays that leave it.
                  The out/in ratio is the fraction the domain never absorbed.

  ``deposition``  Σ ρ·depo·dV over a **checkpoint**, split by material mass fraction
                  (targ / cham / vac) and binned by distance from the beam axis and
                  along it.  This is the measured answer to "how much landed outside
                  the target".  Checkpoints only: ``depo`` and ``lase`` live in ``unk``
                  but the usual ``plot_var`` list omits them, so plot files have no
                  deposition field at all — add ``plot_var_N = "depo"`` to the deck if
                  you want this at plot-file cadence.

  ``tau``         The analytic counterpart: FLASH's *own* κ_IB
                  (flash_utils.flash_ib_opacity, mirroring
                  ed_inverseBremsstrahlungRate.F90 + ed_CoulombFactor.F90) integrated
                  along a fan of rays across the beam footprint, through the same
                  dataset, out to the first cell with targ > 1/2.  Gaussian-weighted
                  Σ w·(1 − e^−τ) is then directly comparable to ``deposition``.
                  Agreement means the ray trace is absorbing correctly and the ambient
                  heating is physics; a gap means it is not.

  ``mesh``        Cell size, refine level and nₑ/n_c along the beam axis, with the
                  beam radius expressed in cells.  A beam narrower than a few cells,
                  or a refine_var list that ignores the beam channel, makes the
                  transverse deposition profile and the reflected-ray angles
                  grid-set even when the absorbed *total* is right.

Usage
-----
    python scripts/flash_laser_audit.py --run-dir /pscratch/sd/d/dschnei/FLASH_... \\
        [--checkpoint <path or index>] [--checks energy deposition tau mesh] \\
        [--n-fan 9] [--output-dir results/laser_audit]

``--run-dir`` is the FLASH run directory: the deck (``flash.par``) is the single
source of truth for the beam geometry, pulse and wavelength, and the run directory
also holds the energy-profile file and the checkpoints.  Nothing here reads an
analysis config — this is a FLASH-internal audit, independent of any OSIRIS line of
sight.

The deposition pass streams every leaf cell of the checkpoint (tens of GB), so run it
on a compute node.
"""

import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yt

yt.set_log_level(50)

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from magshockz.common import flash_utils as fu
from magshockz.common import plot_style

K_PER_EV = 11604.518


# ---------------------------------------------------------------------------
# Run-directory layout
# ---------------------------------------------------------------------------

def find_energy_profile(run_dir: str) -> str:
    """Locate ``<basenm>_LaserEnergyProfile.dat``, or return ''."""
    hits = sorted(glob.glob(os.path.join(run_dir, "*_LaserEnergyProfile.dat")))
    return hits[0] if hits else ""


def find_checkpoints(run_dir: str) -> list:
    return sorted(glob.glob(os.path.join(run_dir, "*_hdf5_chk_*")))


def resolve_dataset(run_dir: str, spec, need_depo: bool) -> str:
    """Resolve ``--checkpoint`` (a path, an index, or None) to a dataset file.

    Checkpoints are preferred and are the *only* option for the deposition check,
    since plot files carry no ``depo`` field unless the deck lists it in ``plot_var``.
    The tau and mesh checks read only hydro state, so they fall back to plot files.
    """
    if spec and os.path.exists(str(spec)):
        return str(spec)
    chks = find_checkpoints(run_dir)
    if not chks and need_depo:
        raise SystemExit(
            "no checkpoints in %s — the deposition check needs one, because plot "
            "files carry no 'depo' field unless the deck lists it in plot_var"
            % run_dir)
    if not chks:
        chks = fu.find_plot_files(run_dir)
        if not chks:
            raise SystemExit("no FLASH datasets in %s" % run_dir)
        print("no checkpoints; falling back to plot files (no 'depo' field)")
    return chks[-1] if spec is None else chks[int(spec)]


# ---------------------------------------------------------------------------
# Check 1 — FLASH's own laser energy bookkeeping
# ---------------------------------------------------------------------------

def check_energy(path: str) -> dict:
    """Read the energy-profile file: cumulative in/out and the unabsorbed fraction."""
    d = np.loadtxt(path, comments="#")
    out = dict(step=d[:, 0], t_s=d[:, 1], dt_s=d[:, 2],
               e_in_erg=d[:, 3], e_out_erg=d[:, 4],
               de_in_erg=d[:, 5], de_out_erg=d[:, 6])
    out["frac_out"] = np.divide(out["e_out_erg"], out["e_in_erg"],
                                out=np.zeros_like(out["e_out_erg"]),
                                where=out["e_in_erg"] > 0)
    return out


# ---------------------------------------------------------------------------
# Beam-frame derived fields
# ---------------------------------------------------------------------------

def add_beam_fields(ds, beam: fu.LaserBeam) -> None:
    """Register the beam-frame coordinates and the per-cell deposited energy.

    ``laser_energy`` is an *energy*, not a density: ``depo`` is the specific energy
    (erg/g) deposited during the step the checkpoint was written in, so ρ·depo·dV is
    the erg that landed in that cell.  Summing it and comparing against P·dt is what
    closes the budget.
    """
    lens = ds.arr(np.asarray(beam.lens, float), "cm")
    axis = ds.arr(beam.axis, "dimensionless")

    def _rel(data):
        return [data["index", ax].to("cm") - lens[i] for i, ax in enumerate("xyz")]

    def _beam_distance(field, data):
        r = _rel(data)
        return sum(r[i] * float(axis[i]) for i in range(3))

    def _beam_radius(field, data):
        r = _rel(data)
        s = sum(r[i] * float(axis[i]) for i in range(3))
        perp2 = sum((r[i] - s * float(axis[i]))**2 for i in range(3))
        return np.sqrt(perp2)

    ds.add_field(("gas", "beam_distance"), function=_beam_distance, units="cm",
                 sampling_type="cell", take_log=False, force_override=True)
    ds.add_field(("gas", "beam_radius"), function=_beam_radius, units="cm",
                 sampling_type="cell", take_log=False, force_override=True)

    if ("flash", "depo") not in ds.field_list:
        # A plot file without depo: the beam-frame coordinates are still useful for
        # the tau and mesh checks, so register those and stop.
        return

    def _energy(field, data):
        # depo carries code_length**2/code_time**2, i.e. erg/g in FLASH's CGS units.
        return (data["flash", "depo"].to("erg/g") * data["flash", "dens"]
                * data["index", "cell_volume"])

    ds.add_field(("gas", "laser_energy"), function=_energy, units="erg",
                 sampling_type="cell", take_log=False, force_override=True)

    def _in_footprint(field, data):
        # A hard mask, not a binned estimate: the radial profile's bins do not land on
        # the footprint edge, so reading the split off the profile makes the answer
        # depend on the bin count.
        return (data["gas", "laser_energy"]
                * (data["gas", "beam_radius"] <= ds.quan(beam.semi_axis, "cm")))

    ds.add_field(("gas", "laser_energy_in_footprint"), function=_in_footprint,
                 units="erg", sampling_type="cell", take_log=False,
                 force_override=True)

    for spec in ("targ", "cham", "vac"):
        if ("flash", spec) not in ds.field_list:
            continue
        ds.add_field(("gas", "laser_energy_%s" % spec),
                     function=(lambda s: lambda field, data:
                               data["gas", "laser_energy"] * data["flash", s])(spec),
                     units="erg", sampling_type="cell", take_log=False,
                     force_override=True)


# ---------------------------------------------------------------------------
# Check 2 — where the deposited energy actually landed
# ---------------------------------------------------------------------------

def check_deposition(ds, beam: fu.LaserBeam, n_rbins: int = 24,
                     n_sbins: int = 24) -> dict:
    """Split Σ ρ·depo·dV by material, by beam radius and along the beam."""
    ad = ds.all_data()
    species = [s for s in ("targ", "cham", "vac") if ("flash", s) in ds.field_list]
    fields = ([("gas", "laser_energy"), ("gas", "laser_energy_in_footprint")]
              + [("gas", "laser_energy_%s" % s) for s in species])
    totals = ad.quantities.total_quantity(fields)

    res = {"t_s": float(ds.current_time.to("s")),
           "dt_s": float(ds.parameters.get("dt", np.nan)),
           "e_total_erg": float(totals[0].to("erg")),
           "e_in_footprint_erg": float(totals[1].to("erg"))}
    for s, v in zip(species, totals[2:]):
        res["e_%s_erg" % s] = float(v.to("erg"))

    # Radial and axial distribution of the deposition.  weight_field=None makes
    # create_profile *sum* the field in each bin rather than average it, so the bins
    # add back up to the total.
    box = ds.domain_width.to("cm").max()
    r_prof = yt.create_profile(ad, ("gas", "beam_radius"), ("gas", "laser_energy"),
                               weight_field=None, n_bins=n_rbins, accumulation=False,
                               extrema={"beam_radius": (1.0e-3, float(box))},
                               logs={"beam_radius": True})
    s_prof = yt.create_profile(ad, ("gas", "beam_distance"), ("gas", "laser_energy"),
                               weight_field=None, n_bins=n_sbins, accumulation=False,
                               logs={"beam_distance": False})
    res["r_cm"] = np.asarray(r_prof.x.to("cm"))
    res["e_of_r_erg"] = np.asarray(r_prof["gas", "laser_energy"].to("erg"))
    res["s_cm"] = np.asarray(s_prof.x.to("cm"))
    res["e_of_s_erg"] = np.asarray(s_prof["gas", "laser_energy"].to("erg"))

    res["p_dt_erg"] = beam.power_erg_s(res["t_s"]) * res["dt_s"]
    return res


# ---------------------------------------------------------------------------
# Check 3 — analytic optical depth with FLASH's own kappa_IB
# ---------------------------------------------------------------------------

def sample_ray(ds, p0, p1, n_crit: float) -> dict:
    """Sample κ_IB and the target mask along one straight ray, ordered lens → target."""
    ray = ds.ray(p0, p1)
    order = np.argsort(np.asarray(ray["t"]))
    length = float(np.linalg.norm(np.asarray(p1) - np.asarray(p0)))

    dens = np.asarray(ray["flash", "dens"][order])
    tele = np.asarray(ray["flash", "tele"][order])
    ye = np.asarray(ray["flash", "ye"][order])
    sumy = np.asarray(ray["flash", "sumy"][order])
    n_ele = dens * ye * 6.02214076e23
    zbar = ye / sumy
    kappa, ln_lambda = fu.flash_ib_opacity(zbar, tele, n_ele, n_crit)

    return dict(ds_cm=np.asarray(ray["dts"][order]) * length,
                dx_cm=np.asarray(ray["dx"][order].to("cm")),
                level=np.asarray(ray["index", "grid_level"][order]),
                n_ele_cm3=n_ele, zbar=zbar, t_ele_eV=tele / K_PER_EV,
                kappa_cm=kappa, ln_lambda=ln_lambda,
                targ=np.asarray(ray["flash", "targ"][order])
                if ("flash", "targ") in ds.field_list else np.zeros_like(dens),
                cham=np.asarray(ray["flash", "cham"][order])
                if ("flash", "cham") in ds.field_list else np.ones_like(dens),
                s_cm=np.cumsum(np.asarray(ray["dts"][order])) * length)


def _to_target_surface(col: dict) -> int:
    """Index of the first cell that is mostly target material (else the ray end)."""
    hit = np.where(col["targ"] > 0.5)[0]
    return int(hit[0]) if hit.size else col["kappa_cm"].size


def check_tau(ds, beam: fu.LaserBeam, n_fan: int = 9) -> dict:
    """Gaussian-weighted absorbed fraction before the target, over the footprint."""
    n_crit = beam.critical_density_cm3
    axis = beam.axis
    # Two unit vectors spanning the beam cross section.
    tmp = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(axis, tmp); e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)

    # Clip the ray to the domain: FLASH launches rays at the boundary, not at the lens.
    lo = np.asarray(ds.domain_left_edge.to("cm"))
    hi = np.asarray(ds.domain_right_edge.to("cm"))
    lens = np.asarray(beam.lens, float)
    target = np.asarray(beam.target, float)

    def clip_entry(offset):
        """Walk from the lens toward the target until inside the domain."""
        a, b = lens + offset, target + offset
        d = b - a
        t_lo = np.where(d != 0, (lo - a) / np.where(d != 0, d, 1), -np.inf)
        t_hi = np.where(d != 0, (hi - a) / np.where(d != 0, d, 1), np.inf)
        t_enter = np.max(np.minimum(t_lo, t_hi))
        return a + d * min(max(t_enter, 0.0) + 1e-9, 1.0), b

    offs, weights, taus, cols = [], [], [], []
    grid = np.linspace(-beam.semi_axis, beam.semi_axis, n_fan)
    for u in grid:
        for v in grid:
            r = np.hypot(u, v)
            if r > beam.semi_axis:
                continue
            offset = u * e1 + v * e2
            p0, p1 = clip_entry(offset)
            col = sample_ray(ds, p0, p1, n_crit)
            isurf = _to_target_surface(col)
            tau = float((col["kappa_cm"][:isurf] * col["ds_cm"][:isurf]).sum())
            offs.append((u, v))
            weights.append(np.exp(-(r / beam.gaussian_radius)**beam.gaussian_exponent))
            taus.append(tau)
            cols.append(col)

    w = np.asarray(weights); w /= w.sum()
    taus = np.asarray(taus)
    return dict(offsets=np.asarray(offs), weights=w, tau=taus,
                tau_mean=float(np.dot(w, taus)),
                frac_absorbed=float(np.dot(w, 1.0 - np.exp(-taus))),
                columns=cols)


# ---------------------------------------------------------------------------
# Check 4 — the mesh the beam actually crosses
# ---------------------------------------------------------------------------

def check_mesh(ds, beam: fu.LaserBeam) -> dict:
    """Cell size, refine level and nₑ/n_c along the beam axis."""
    lo = np.asarray(ds.domain_left_edge.to("cm"))
    hi = np.asarray(ds.domain_right_edge.to("cm"))
    lens = np.asarray(beam.lens, float)
    target = np.asarray(beam.target, float)
    d = target - lens
    t_lo = np.where(d != 0, (lo - lens) / np.where(d != 0, d, 1), -np.inf)
    t_hi = np.where(d != 0, (hi - lens) / np.where(d != 0, d, 1), np.inf)
    p0 = lens + d * (max(np.max(np.minimum(t_lo, t_hi)), 0.0) + 1e-9)

    col = sample_ray(ds, p0, target, beam.critical_density_cm3)
    isurf = _to_target_surface(col)
    col["cells_per_beam_radius"] = beam.gaussian_radius / col["dx_cm"]
    col["i_target_surface"] = isurf
    # "Ambient" is the unshocked chamber material the beam crosses on the way in, not
    # everything upstream of the target: the ablation plume sits in between and is
    # neither, and its keV temperatures and refined cells would otherwise be reported
    # as if they described the background the beam has to traverse.
    amb = np.zeros(col["kappa_cm"].size, bool)
    amb[:isurf] = True
    amb &= col["cham"] > 0.5
    col["ambient_mask"] = amb
    # FLASH counts refinement from 1 (lrefine_min/max in the deck); yt's grid_level
    # counts from 0.  Report the deck's convention so the two can be compared.
    col["lrefine"] = col["level"] + 1
    col["lrefine_in_ambient"] = np.unique(col["lrefine"][amb]).astype(int)
    col["dx_ambient_cm"] = (float(col["dx_cm"][amb].min()),
                            float(col["dx_cm"][amb].max()))
    return col


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report(beam, par, energy, depo, tau, mesh) -> None:
    print("\nlaser deck (flash.par)")
    print("  wavelength %.3f um -> n_c = %.4g cm^-3 ; %d rays ; ed_gradOrder = %g"
          % (beam.wavelength_um, beam.critical_density_cm3, beam.n_rays,
             par.get("ed_gradorder", float("nan"))))
    print("  pulse energy %.4g erg (%.1f J) over %.3g-%.3g ns"
          % (beam.energy_erg(), beam.energy_erg() * 1e-7,
             beam.pulse_time_s[0] * 1e9, beam.pulse_time_s[-1] * 1e9))
    off = [k for k in ("ed_uselaserio", "ed_saveoutofdomainrays", "ed_printbeams",
                       "ed_printrays", "ed_celltimeenergydeposition")
           if par.get(k) is False]
    if off:
        print("  diagnostics available but switched off: %s" % ", ".join(off))

    if energy is not None:
        print("\nenergy budget (FLASH's own LaserEnergyProfile.dat)")
        print("  pumped into domain   %.4g erg" % energy["e_in_erg"][-1])
        print("  carried back out     %.4g erg  (%.2f%% never absorbed)"
              % (energy["e_out_erg"][-1], 100 * energy["frac_out"][-1]))

    if depo is not None:
        e = depo["e_total_erg"]
        print("\ndeposition at t = %.4g ns (dt = %.3g s)"
              % (depo["t_s"] * 1e9, depo["dt_s"]))
        if depo["p_dt_erg"] > 0:
            print("  Sum rho*depo*dV = %.4g erg = %.1f%% of P*dt"
                  % (e, 100 * e / depo["p_dt_erg"]))
        for s in ("targ", "cham", "vac"):
            if "e_%s_erg" % s in depo:
                print("    %-5s %.4g erg  (%.2f%% of deposited)"
                      % (s, depo["e_%s_erg" % s], 100 * depo["e_%s_erg" % s] / e))
        print("  inside the %.0f um beam footprint: %.2f%% of deposited"
              % (beam.semi_axis * 1e4, 100 * depo["e_in_footprint_erg"] / e))

    if tau is not None:
        print("\nanalytic optical depth to the target surface (FLASH's own kappa_IB)")
        print("  tau over the footprint: min %.4f  max %.4f  gaussian-weighted %.4f"
              % (tau["tau"].min(), tau["tau"].max(), tau["tau_mean"]))
        print("  => %.2f%% of the incident beam absorbed before the target"
              % (100 * tau["frac_absorbed"]))
        if depo is not None and depo["p_dt_erg"] > 0 and "e_cham_erg" in depo:
            meas = 100 * depo["e_cham_erg"] / depo["p_dt_erg"]
            pred = 100 * tau["frac_absorbed"]
            print("  measured ambient deposition %.2f%% of P*dt -> analytic/measured "
                  "= %.2f" % (meas, pred / meas if meas > 0 else float("nan")))

    if mesh is not None:
        m = mesh["ambient_mask"]
        print("\nmesh in the ambient the beam crosses (cham > 1/2, before the target)")
        print("  lrefine %s ; dx %.1f-%.1f um"
              % (mesh["lrefine_in_ambient"], mesh["dx_ambient_cm"][0] * 1e4,
                 mesh["dx_ambient_cm"][1] * 1e4))
        print("  beam radius %.0f um = %.1f-%.1f cells"
              % (beam.gaussian_radius * 1e4,
                 mesh["cells_per_beam_radius"][m].min(),
                 mesh["cells_per_beam_radius"][m].max()))
        print("  n_e/n_c %.2e - %.2e ; Zbar %.2f - %.2f ; Te %.0f - %.0f eV"
              % ((mesh["n_ele_cm3"][m] / beam.critical_density_cm3).min(),
                 (mesh["n_ele_cm3"][m] / beam.critical_density_cm3).max(),
                 mesh["zbar"][m].min(), mesh["zbar"][m].max(),
                 mesh["t_ele_eV"][m].min(), mesh["t_ele_eV"][m].max()))


def make_figure(beam, energy, depo, tau, mesh, out_png: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=plot_style.figsize(10.0, 7.0))
    ax = axes.ravel()

    if energy is not None:
        t = energy["t_s"] * 1e9
        ax[0].plot(t, energy["e_in_erg"], label="into domain")
        ax[0].plot(t, energy["e_out_erg"], label="back out")
        ax[0].set_yscale("log")
        ax[0].set_xlabel("t [ns]"); ax[0].set_ylabel("cumulative laser energy [erg]")
        twin = ax[0].twinx()
        twin.plot(t, 100 * energy["frac_out"], color="C3", lw=1.0, ls="--")
        twin.set_ylabel("out / in [%]", color="C3")
        ax[0].legend(loc="lower right", fontsize="small")
        ax[0].set_title("FLASH energy bookkeeping")
    else:
        ax[0].text(0.5, 0.5, "no LaserEnergyProfile.dat", ha="center", va="center",
                   transform=ax[0].transAxes)
        ax[0].set_xticks([]); ax[0].set_yticks([])

    if depo is not None:
        e = depo["e_of_r_erg"]
        frac = np.cumsum(e) / max(e.sum(), 1e-99)
        ax[1].step(depo["r_cm"] * 1e4, 100 * frac, where="mid")
        ax[1].axvline(beam.semi_axis * 1e4, color="C3", ls="--",
                      label="beam footprint")
        ax[1].axvline(beam.gaussian_radius * 1e4, color="C2", ls=":",
                      label="gaussian radius")
        ax[1].set_xscale("log")
        ax[1].set_xlabel(r"distance from beam axis [$\mu$m]")
        ax[1].set_ylabel("cumulative deposited energy [%]")
        ax[1].legend(fontsize="small"); ax[1].set_title("radial deposition")

        ax[2].step(depo["s_cm"], depo["e_of_s_erg"], where="mid")
        ax[2].set_yscale("log")
        ax[2].set_xlabel("distance along beam from lens [cm]")
        ax[2].set_ylabel("deposited energy per bin [erg]")
        ax[2].set_title("deposition along the beam")
    else:
        for a in (ax[1], ax[2]):
            a.text(0.5, 0.5, "deposition check not run\n(needs a checkpoint)",
                   ha="center", va="center", transform=a.transAxes)
            a.set_xticks([]); a.set_yticks([])

    if mesh is not None:
        i = mesh["i_target_surface"]
        ax[3].step(mesh["s_cm"][:i], mesh["dx_cm"][:i] * 1e4, where="mid",
                   label="cell size")
        ax[3].axhline(beam.gaussian_radius * 1e4, color="C2", ls=":",
                      label="gaussian radius")
        ax[3].set_xlabel("distance along beam from domain entry [cm]")
        ax[3].set_ylabel(r"$\Delta x$ [$\mu$m]")
        twin = ax[3].twinx()
        if tau is not None:
            twin.plot(mesh["s_cm"][:i],
                      np.cumsum(mesh["kappa_cm"][:i] * mesh["ds_cm"][:i]),
                      color="C3", lw=1.0)
            twin.set_ylabel(r"cumulative $\tau$", color="C3")
        ax[3].legend(fontsize="small"); ax[3].set_title("mesh along the beam")
    else:
        ax[3].text(0.5, 0.5, "mesh check not run", ha="center", va="center",
                   transform=ax[3].transAxes)
        ax[3].set_xticks([]); ax[3].set_yticks([])

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print("\nwrote %s" % out_png)


# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True,
                   help="FLASH run directory (holds flash.par, the energy profile "
                        "and the checkpoints)")
    p.add_argument("--par", default=None, help="override the flash.par path")
    p.add_argument("--checkpoint", default=None,
                   help="checkpoint path, or an index into the run's checkpoints "
                        "(default: the last one)")
    p.add_argument("--checks", nargs="+", default=["energy", "deposition", "tau", "mesh"],
                   choices=["energy", "deposition", "tau", "mesh"])
    p.add_argument("--beam", type=int, default=1, help="which ed_ beam (1-based)")
    p.add_argument("--n-fan", type=int, default=9,
                   help="rays per side of the square fan across the footprint")
    p.add_argument("--output-dir", default="results/laser_audit")
    plot_style.add_publication_arg(p)
    args = p.parse_args()

    plot_style.apply(getattr(args, "publication", False))
    os.makedirs(args.output_dir, exist_ok=True)

    par_path = args.par or os.path.join(args.run_dir, "flash.par")
    par = fu.parse_flash_par(par_path)
    beams = fu.laser_beams(par)
    if not beams:
        raise SystemExit("%s defines no ed_ beams" % par_path)
    beam = beams[args.beam - 1]

    energy = None
    if "energy" in args.checks:
        ep = find_energy_profile(args.run_dir)
        if ep:
            energy = check_energy(ep)
        else:
            print("no *_LaserEnergyProfile.dat in %s — FLASH always writes it, so it "
                  "was lost in a copy; without it the out-of-domain energy is "
                  "unknown" % args.run_dir)

    depo = tau = mesh = None
    if {"deposition", "tau", "mesh"} & set(args.checks):
        path = resolve_dataset(args.run_dir, args.checkpoint,
                               need_depo="deposition" in args.checks)
        print("dataset: %s" % path)
        ds = yt.load(path)
        add_beam_fields(ds, beam)
        if "deposition" in args.checks:
            depo = check_deposition(ds, beam)
        if "tau" in args.checks:
            tau = check_tau(ds, beam, n_fan=args.n_fan)
        if "mesh" in args.checks:
            mesh = check_mesh(ds, beam)

    report(beam, par, energy, depo, tau, mesh)

    stem = os.path.join(args.output_dir, "flash_laser_audit")
    make_figure(beam, energy, depo, tau, mesh, stem + ".png")
    arch = {}
    for name, d in (("energy", energy), ("depo", depo), ("mesh", mesh)):
        if d is None:
            continue
        for k, v in d.items():
            if isinstance(v, (int, float, np.ndarray)):
                arch["%s_%s" % (name, k)] = v
    if tau is not None:
        arch.update(tau_offsets=tau["offsets"], tau_weights=tau["weights"],
                    tau_tau=tau["tau"], tau_mean=tau["tau_mean"],
                    tau_frac_absorbed=tau["frac_absorbed"])
    np.savez(stem + ".npz", **arch)
    print("wrote %s.npz" % stem)


if __name__ == "__main__":
    main()
