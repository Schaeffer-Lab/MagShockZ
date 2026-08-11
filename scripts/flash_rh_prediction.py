# -*- coding: utf-8 -*-
"""scripts/flash_rh_prediction.py — predict the downstream FLASH state from the
measured UPSTREAM state + shock speed, using the perpendicular (theta = 90 deg)
MHD shock theory in src/perpendicular_shock.py, and overlay the prediction on
lineouts across the shock front.

What it does
------------
1. Loads one FLASH dump's lineout along the OSIRIS line-of-sight (unyt arrays in
   physical CGS; units travel with the data and conversions are done with .to()).
2. Reads the shock position and fitted shock speed v_shock from the
   flash_overview .npz (run flash_overview.py first), splitting the lineout into
   an upstream and a downstream region.
3. Averages the UPSTREAM region to a single state (rho, n_e, n_ion, T_e, T_i,
   B_perp, v_para) and hands it to perpendicular_shock.solve_from_upstream,
   which forms the sound speed (the two-temperature / ion-acoustic form,
   c_s = sqrt((gamma_e P_e + gamma_i P_i)/rho), defaulting gamma_e=gamma_i=gamma
   so it matches the single-fluid jump), the Alfven speed, and the shock-frame
   inflow, then solves the perpendicular MHD shock for r, p2/p1, T2/T1.  No
   speeds or Mach numbers are assembled in this script.
4. Predicts the downstream value of every quantity (rho/n_e scale by r, the
   transverse field by r, total thermal pressure by p2/p1, the temperatures by
   T2/T1, the shock-frame inflow speed by 1/r).
5. Plots each quantity's lineout across the front with three reference lines:
   the upstream mean, the THEORY-predicted downstream value, and the measured
   downstream mean — so measured-vs-predicted is read off directly.

Fields are read DIRECTLY from FLASH: n_e, n_ion, T_e, T_i, |B|, B_para, v_para,
rho.  The ONE exception is the thermal pressure: this 3T FLASH dataset does not
store pressure on disk (the EOS makes it from the temperatures), so we form the
ideal-gas pressure P = n_e kT_e + n_ion kT_i.

This is the simplest (pure-MHD, isotropic-pressure) baseline.  Single-fluid MHD
predicts ONE temperature jump, applied to T_e and T_i alike; it does NOT predict
the electron/ion split — that, and any departure of the data from these lines,
is the kinetic/collisionless physics MHD omits.

Usage
-----
    python scripts/flash_rh_prediction.py \\
        --config config/flash_3d_noshield.yaml \\
        [--snapshot-idx N] [--gamma 1.6667] \\
        [--x-shock-cm ...] [--x-downstream-start-cm ...] [--v-shock-cms ...] \\
        [--window-um 400] [--output-dir results/FLASH_3D_noshield]

--snapshot-idx defaults to the dump that seeded the OSIRIS run (RunSpec data_path), so
the Mach numbers/predictions are directly comparable to the OSIRIS side; pass an
explicit index (e.g. -1 for the last dump) to look at a different snapshot.

Run in the `analysis` conda env (it has yt + unyt + osiris_utils).
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
import unyt

_HERE = os.path.dirname(os.path.abspath(__file__))

from magshockz.common import analysis_utils
from magshockz.common import plot_style
from magshockz.common import flash_source
from magshockz.common import yaml_edit
from magshockz.common import flash_utils as fu
from magshockz.common import perpendicular_shock as ps
from magshockz.analysis.flash import shock
from magshockz.analysis.flash import flash_energy_partition as fep


def _load_shock_geometry(cfg, idx, out_dir, x_shock_cm, x_ds_start, label=""):
    """Resolve (x_shock [cm], x_downstream_start [cm], v_shock [cm/s]) as bare
    floats.  Priority (first hit wins): explicit CLI value → the hand-placed
    ``flash_dump_params.<idx>`` in the config (written by flash_tune_shock.py) →
    the flash_overview .npz.  The caller attaches units.

    ``idx`` is the resolved positive plot-file index (the config key); ``label`` names
    the line of sight when the config carries several.  ``out_dir`` is searched for
    ``flash_overview_*.npz`` only for whatever remains unset — and it is already the
    ray's own directory, so the fallback cannot pick up another ray's line-out.
    """
    # 1. hand-placed per-dump positions in the config (cm).
    per = flash_source.los_params(cfg, "flash_dump_params", label).get(idx, {})
    if x_shock_cm is None and "x_shock_cm" in per:
        x_shock_cm = float(per["x_shock_cm"])
    if x_ds_start is None and "x_downstream_start_cm" in per:
        x_ds_start = float(per["x_downstream_start_cm"])

    # 2. flash_overview .npz fills in whatever is still missing (shock position,
    #    downstream edge, and the fitted v_shock — always read so the caller can
    #    fall back to it for the shock speed).
    v_shock_npz = None
    npz_files = sorted(
        f for f in os.listdir(out_dir)
        if f.startswith("flash_overview_") and f.endswith(".npz")
    )
    if npz_files:
        d = np.load(os.path.join(out_dir, npz_files[-1]), allow_pickle=True)
        npz_idx = shock.overview_row(
            idx, d["dump_indices"] if "dump_indices" in d.files else None,
            len(d["time_ns"]))
        if "v_shock_cms" in d.files:
            v_shock_npz = float(d["v_shock_cms"])
        if x_shock_cm is None:
            if "x_shock_det_cm" in d.files:
                x_shock_cm = float(d["x_shock_det_cm"][npz_idx])
            if (x_shock_cm is None or np.isnan(x_shock_cm)) and "x_shock_0_cm" in d.files:
                t_snap_s = float(d["time_ns"][npz_idx]) * 1e-9      # the npz stores ns
                x_shock_cm = float(d["x_shock_0_cm"]) + float(d["v_shock_cms"]) * t_snap_s
    if x_ds_start is None and x_shock_cm is not None and np.isfinite(x_shock_cm):
        x_ds_start = x_shock_cm * 0.9
    return x_shock_cm, x_ds_start, v_shock_npz


def main():
    parser = argparse.ArgumentParser(
        description="Predict the downstream FLASH state from upstream + v_shock "
                    "(perpendicular MHD shock) and overlay it on lineouts."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--snapshot-idx", type=int, default=None, dest="snapshot_idx",
                        help="Index into the sorted plot-file list (default: the dump "
                             "that seeded the OSIRIS run, i.e. RunSpec data_path — NOT "
                             "the last dump, so the Mach numbers/predictions are "
                             "directly comparable to the OSIRIS side).")
    parser.add_argument("--x-shock-cm", type=float, default=None, dest="x_shock_cm",
                        help="Shock position along LOS [cm]. Default: read from the "
                             "flash_overview .npz.")
    parser.add_argument("--x-downstream-start-cm", type=float, default=None,
                        dest="x_downstream_start_cm",
                        help="Left edge of the downstream region [cm].")
    parser.add_argument("--v-shock-cms", type=float, default=None, dest="v_shock_cms",
                        help="Shock velocity [cm/s]. Default: fitted v_shock from the "
                             "flash_overview .npz.")
    parser.add_argument("--gamma", type=float, default=None,
                        help="Adiabatic index (default: config 'gamma' key, else 5/3). "
                             "gamma=(f+2)/f: 5/3 (3 DOF), 2 (2 DOF), 3 (1 DOF) — sweep "
                             "to read off the effective index.")
    parser.add_argument("--window-um", type=float, default=400.0, dest="window_um",
                        help="Half-width [µm] of the zoom window around the front (default 400).")
    parser.add_argument("--snap-front-um", type=float, default=60.0,
                        dest="snap_front_um",
                        help="Search half-width [µm] for snapping the placed "
                             "front onto the steepest density drop (default 60; "
                             "0 disables). A front placed a few cells outside the "
                             "jump fills part of the thin RH band with upstream "
                             "and drags every downstream mean toward ambient. The "
                             "shift is reported.")
    parser.add_argument("--jump-band-um", type=float, default=300.0,
                        dest="jump_band_um",
                        help="Width [µm] of the thin band just behind the front "
                             "used for the Rankine-Hugoniot checks (default 300). "
                             "RH is a LOCAL jump condition, so it must be tested "
                             "against a band much narrower than the shocked layer "
                             "-- momentum-flux continuity on this data runs 1.00 "
                             "over 50 µm and 0.51 over 940 µm. It must still be "
                             "wider than the ramp: the front marks the UPSTREAM "
                             "foot of the transition, which is ~100 µm thick here, "
                             "so a band narrower than that averages the ramp rather "
                             "than the shocked plateau. Heating and the e/i "
                             "partition still use the full layer band. 0 makes the "
                             "two bands identical.")
    parser.add_argument("--upstream-gap-um", type=float, default=200.0,
                        dest="upstream_gap_um",
                        help="Start the upstream average this far [µm] ahead of the "
                             "front, clearing its precursor/ramp (default 200).")
    parser.add_argument("--downstream-edge", choices=("contact", "config"),
                        default="contact", dest="downstream_edge",
                        help="Where the downstream band's inner edge comes from. "
                             "'contact' (default) walks in from the front to the "
                             "outermost piston material and stops --contact-gap-um "
                             "short of it, so the band is as wide as the shocked "
                             "AMBIENT actually is; 'config' uses the hand-placed "
                             "x_downstream_start_cm verbatim.")
    parser.add_argument("--contact-gap-um", type=float, default=50.0,
                        dest="contact_gap_um",
                        help="Standoff [µm] to leave between the piston contact and "
                             "the downstream band, so the mixing layer at the contact "
                             "stays out of the average (default 50).")
    parser.add_argument("--upstream-width-um", type=float, default=600.0,
                        dest="upstream_width_um",
                        help="Width [µm] of the upstream average (default 1000). The "
                             "shock runs into the gas immediately ahead of it, so the "
                             "average is a window, not the whole remaining ray: "
                             "averaging to the domain edge mixes in far-field the "
                             "shock has not reached, and on some rays a laser-channel "
                             "column. 0 means 'to the end of the ray' (the old behaviour).")
    parser.add_argument("--output-dir", default=None, dest="output_dir")
    flash_source.add_los_arg(parser)
    plot_style.add_publication_arg(parser)
    args = parser.parse_args()
    plot_style.apply(args.publication)

    # ------------------------------------------------------------------
    # Config + run parameters
    # ------------------------------------------------------------------
    cfg    = analysis_utils.load_config(args.config)
    source = flash_source.resolve(cfg, args.config, los=args.los)

    flash_dir  = source.flash_dir
    line_start = source.line_start
    line_end   = source.line_end

    all_files = fu.find_plot_files(flash_dir)
    if args.snapshot_idx is None:
        # Default: the source's IC dump, so results are directly comparable to the
        # OSIRIS side (see mach-number-self-consistent-dump memory).
        idx = source.ic_index
    else:
        idx = args.snapshot_idx % len(all_files)   # config key = positive plot-file index
    snap_file = all_files[idx]

    out_dir = yaml_edit.out_dir(flash_dir, args.output_dir,
                                cfg=cfg, config_path=args.config, subdir=source.label)

    x_shock_cm, x_ds_cm, v_shock_npz = _load_shock_geometry(
        cfg, idx, out_dir, args.x_shock_cm,
        args.x_downstream_start_cm, label=source.label)

    if x_shock_cm is None or not np.isfinite(x_shock_cm):
        raise ValueError(
            "Shock position not available. Place it with tune_flash_shock.py "
            "(--mode regions), run flash_overview.py, or pass --x-shock-cm.")
    if x_ds_cm is None:
        raise ValueError(
            "Downstream-region edge not available. Place it with tune_flash_shock.py "
            "or pass --x-downstream-start-cm.")

    # v_shock: CLI → config flash.v_shock_est_cms → fitted value from the npz.
    flash_cfg = flash_source.los_params(cfg, "flash", source.label)
    if args.v_shock_cms is not None:
        v_shock_cms = args.v_shock_cms
    elif flash_cfg.get("v_shock_est_cms"):
        v_shock_cms = float(flash_cfg["v_shock_est_cms"])
    elif v_shock_npz is not None and np.isfinite(v_shock_npz):
        v_shock_cms = v_shock_npz
    else:
        v_shock_cms = 0.0

    gamma = args.gamma if args.gamma is not None else float(cfg.get("gamma", 5.0 / 3.0))

    # Attach units once; let unyt carry them from here on.
    x_shock = x_shock_cm * unyt.cm
    x_ds    = x_ds_cm    * unyt.cm
    v_shock = v_shock_cms * unyt.cm / unyt.s

    print(f"Config       : {args.config}")
    print(f"FLASH dir    : {flash_dir}")
    print(f"Snapshot     : {os.path.basename(snap_file)}")
    print(f"x_shock      : {x_shock.to('um'):.2f}")
    print(f"x_downstream : {x_ds.to('um'):.2f}")
    print(f"v_shock      : {v_shock.to('km/s'):.2f}")
    print(f"gamma        : {gamma:.4f}")

    # ------------------------------------------------------------------
    # Lineout + derived per-point quantities (all unyt)
    # ------------------------------------------------------------------
    print("\nLoading lineout …", flush=True)
    # ye/sumy are FLASH's electron bookkeeping fields; their ratio is the local mean
    # charge Zbar, which flash.par cannot supply (it states ms_chamZ = 13, the ATOMIC
    # number, while the EOS returns ~3.7 here) and which sets n_ion, beta_e and d_i.
    piston_material = str(cfg.get("piston_material", "targ"))
    lo = fu.flash_lineout(snap_file, line_start, line_end,
                          extra_fields={"ye": ("flash", "ye"),
                                        "sumy": ("flash", "sumy"),
                                        piston_material: ("flash", piston_material)})
    zbar = np.divide(np.asarray(lo["ye"]), np.asarray(lo["sumy"]),
                     out=np.zeros_like(np.asarray(lo["ye"])),
                     where=np.asarray(lo["sumy"]) > 0.0)

    x = lo["x"].to("cm")
    t = (lo["t_s"] * unyt.s)

    # Transverse (shock-tangential) field: the perpendicular shock compresses
    # this with the density.  B_perp = sqrt(|B|^2 - B_para^2); the abs only
    # guards floating-point roundoff (|B_para| <= |B| exactly by construction).
    B_perp = np.sqrt(np.abs(lo["B_mag"]**2 - lo["B_para"]**2)).to("gauss")
    P_th   = (lo["ne"] * lo["Te"] + lo["n_ion"] * lo["Ti"]).to("erg/cm**3")
    v_sf   = np.abs(lo["v_para"] - v_shock).to("cm/s")   # shock-frame normal speed

    # ------------------------------------------------------------------
    # Region averages.  Upstream is a WINDOW starting a gap ahead of the front,
    # not everything beyond it: the state that sets the Mach numbers is the gas the
    # shock is about to hit.  Averaging to the end of the ray mixes in far-field
    # ambient the blast has not reached (it rarefies ~10% over this run) and, on a ray
    # that happens to cross one, a laser-channel column — a +40% n_e bias on los15.
    # ------------------------------------------------------------------
    # Both averaging windows are placed by ONE shared resolver, so this script and
    # flash_pressure_partition cannot end up measuring the same shock over different
    # regions (they did, and reported compressions differing by 3x).
    x_cm = np.asarray(x.to("cm"))
    UM_PER_CM = 1.0e4
    # A front placed by eye can sit a few cells outside the density jump, which fills
    # part of the thin RH band with upstream. Snap it onto the gradient and say by how
    # much, so the correction is visible rather than silent.
    if args.snap_front_um > 0.0:
        snapped = shock.snap_front_to_jump(
            x_cm, np.asarray(lo["rho"].to("g/cm**3")), float(x_shock.to("cm")),
            args.snap_front_um / 1.0e4)
        shift_um = (snapped - float(x_shock.to("cm"))) * 1.0e4
        if abs(shift_um) > 1.0:
            print(f"  front snapped {shift_um:+.0f} µm to the density jump "
                  f"({float(x_shock.to("cm")) * 1.0e4:.0f} -> {snapped * 1.0e4:.0f} µm)")
        x_shock = snapped * unyt.cm

    bands = shock.resolve_bands(
        x_cm, np.asarray(lo[piston_material]), float(x_shock.to("cm")),
        upstream_gap=args.upstream_gap_um / UM_PER_CM,
        upstream_width=args.upstream_width_um / UM_PER_CM,
        contact_gap=args.contact_gap_um / UM_PER_CM,
        jump_width=args.jump_band_um / UM_PER_CM,
        x_downstream_config=float(x_ds.to("cm")), edge=args.downstream_edge)
    if bands.note:
        print(f"  ! {bands.note}")
    if bands.x_downstream != float(x_ds.to("cm")):
        was = (bands.x_shock - float(x_ds.to("cm"))) * UM_PER_CM
        now = (bands.x_shock - bands.x_downstream) * UM_PER_CM
        print(f"  x_downstream : {float(x_ds.to('um')):.0f} (config) -> "
              f"{bands.x_downstream * UM_PER_CM:.0f} µm  "
              f"(contact {bands.x_contact * UM_PER_CM:.0f} + "
              f"{args.contact_gap_um:.0f} µm); band {was:.0f} -> {now:.0f} µm wide")
    x_ds = bands.x_downstream * unyt.cm
    x_contact = bands.x_contact * unyt.cm
    up = bands.upstream_mask(x_cm)
    dn = bands.downstream_mask(x_cm)      # full shocked layer: heating, Zbar, e/i split
    jump_band = bands.jump_mask(x_cm)     # thin slice at the front: the RH jump test

    def mean_up(arr):
        return np.nanmean(arr[up])

    def mean_jump(arr):
        return np.nanmean(arr[jump_band])

    def mean_dn(arr):
        return np.nanmean(arr[dn])

    rho_up   = mean_up(lo["rho"])
    ne_up    = mean_up(lo["ne"])
    ni_up    = mean_up(lo["n_ion"])
    Te_up    = mean_up(lo["Te"])
    Ti_up    = mean_up(lo["Ti"])
    Bperp_up = mean_up(B_perp)
    Bmag_up  = mean_up(lo["B_mag"])
    Bpara_up = mean_up(lo["B_para"])
    vpara_up = mean_up(lo["v_para"])
    zbar_up  = float(np.nanmean(zbar[up]))
    # Upstream thermal pressure from the region-averaged partial pressures, so it
    # is consistent with the sound speed perpendicular_shock builds internally.
    P_up     = (ne_up * Te_up + ni_up * Ti_up).to("erg/cm**3")

    # ------------------------------------------------------------------
    # Perpendicular-shock solution — hand the upstream FLASH fields straight to
    # perpendicular_shock; it forms c_s (two-temperature), v_A and v_inflow and
    # solves the jump.  No speeds or Mach numbers are assembled here.
    # ------------------------------------------------------------------
    jump = ps.solve_from_upstream(
        ne=ne_up, Te=Te_up, n_ion=ni_up, Ti=Ti_up,
        B_perp=Bperp_up, B_para=Bpara_up, rho=rho_up,
        v_shock=v_shock, v_para=vpara_up, gamma=gamma)
    v_A      = jump["v_A"].to("cm/s")
    c_s      = jump["c_s"].to("cm/s")
    v_ms     = jump["v_ms"].to("cm/s")
    mach_ms  = jump["mach_ms"]
    v_inflow = jump["v_inflow"].to("cm/s")
    theta_bn = jump["theta_bn"]
    r        = jump["r"]
    p_ratio  = jump["p_ratio"]
    T_ratio  = jump["T_ratio"]
    mach_s   = jump["mach_s"]
    mach_a   = jump["mach_a"]

    if not jump["exists"] or not np.isfinite(r):
        print("\n!! No compressive perpendicular shock for these upstream numbers "
              f"(M_s={mach_s:.2f}, M_A={mach_a:.2f}). Predictions will be NaN.")

    zbar_dn = float(np.nanmean(zbar[dn]))
    zbar_dn_jump = float(np.nanmean(zbar[jump_band]))

    # DIAGNOSTIC ONLY — the prediction plotted and used as the heating baseline is
    # the unmodified single-fluid T2/T1 = (p2/p1)/(rho2/rho1) above.
    #
    # That form is T proportional to p/rho, which holds only at fixed mean molecular
    # weight, and the weight is not fixed here: the shock ionizes and Zbar roughly
    # doubles across the front.  With p = (rho/m_ion)(1 + Zbar) kT the jump would
    # carry an extra (1+Zbar_1)/(1+Zbar_2) -- the same pressure shared among about
    # twice as many particles per unit mass.  It is reported rather than applied
    # because it is only half the story: r and p_ratio still come from an energy
    # equation with no ionization sink, so neither number is a consistent ionizing
    # shock solution.  The size of this factor is the useful part.
    mu_ratio = (1.0 + zbar_up) / (1.0 + zbar_dn_jump)
    T_ratio_ionized = T_ratio * mu_ratio

    # Predicted downstream values — apply the jump ratios to the unyt upstream
    # state, so the predictions keep their units too.
    pred = ps.predict_downstream(
        jump, rho1=ne_up, B_perp1=Bperp_up, p1=P_up, v_inflow=v_inflow)
    ne_dn_pred    = pred["rho"]          # n scales like rho
    Bperp_dn_pred = pred["B_perp"]
    P_dn_pred     = pred["p"]
    vsf_dn_pred   = pred["v_inflow"]
    Te_dn_pred    = T_ratio * Te_up
    Ti_dn_pred    = T_ratio * Ti_up

    # Measured mass compression, and the adiabatic index that would reproduce it.
    # Mass — not electron — density, because Zbar changes across an ionizing front and
    # n_e/n_e1 then overstates the compression the RH relations are about.
    # The compression is a JUMP condition, so it is measured against the thin band at
    # the front. Over the full layer it reads low by up to 2x, because the layer's
    # inner edge was shocked ns earlier under different upstream conditions.
    rho_dn_meas = mean_jump(lo["rho"])
    rho_dn_layer = mean_dn(lo["rho"])
    r_measured = float((rho_dn_meas / rho_up).to("dimensionless"))
    r_layer = float((rho_dn_layer / rho_up).to("dimensionless"))
    gamma_eff = ps.effective_gamma(
        r_measured,
        dict(ne=ne_up, Te=Te_up, n_ion=ni_up, Ti=Ti_up, B_perp=Bperp_up,
             B_para=Bpara_up, rho=rho_up, v_shock=v_shock, v_para=vpara_up))
    # Heating, Zbar and the e/i partition describe the whole shocked LAYER — the
    # plasma an experiment would diagnose — so they take their temperatures from the
    # same band their densities come from.
    Te_dn_layer = mean_dn(lo["Te"])
    Ti_dn_layer = mean_dn(lo["Ti"])

    # Measured downstream means for the head-to-head.
    ne_dn_meas    = mean_jump(lo["ne"])
    Bperp_dn_meas = mean_jump(B_perp)
    P_dn_meas     = mean_dn(P_th)
    vsf_dn_meas   = mean_dn(v_sf)
    Te_dn_meas    = mean_jump(lo["Te"])
    Ti_dn_meas    = mean_jump(lo["Ti"])

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    # Ion-scale lengths and times, from the upstream state the shock actually runs
    # into.  d_i and 1/omega_ci are what the PIC comparison is eventually normalised to.
    # plasmapy carries the units and the definitions; convert at its boundary and come
    # straight back to unyt, per the FLASH convention in CLAUDE.md.
    from astropy import units as apu
    from astropy.constants import e as elementary_charge
    from plasmapy.formulary import gyrofrequency, inertial_length
    from plasmapy.particles import CustomParticle

    ion_mass = (rho_up / ni_up).to("g")
    ion = CustomParticle(mass=float(ion_mass.to("g").value) * apu.g,
                         charge=zbar_up * elementary_charge.si)
    B_up_apu = float(Bmag_up.to("gauss").value) * apu.G
    ni_up_apu = float(ni_up.to("cm**-3").value) * apu.cm**-3

    # The reported ion time is the INVERSE gyrofrequency 1/omega_ci — the PIC
    # normalisation — NOT the full gyroperiod 2*pi/omega_ci, which is 2*pi larger.
    # gyrofrequency already returns the angular frequency, so no factor is applied.
    # Equivalently 1/omega_ci = d_i / v_A.
    omega_ci = float(gyrofrequency(B_up_apu, ion).to(apu.rad / apu.s).value) / unyt.s
    d_i = float(inertial_length(ni_up_apu, ion).to(apu.um).value) * unyt.um
    inv_omega_ci = (1.0 / omega_ci).to("ns")

    beta_e = float((8.0 * np.pi * ne_up * Te_up / Bmag_up**2).to("dimensionless"))
    beta_i = float((8.0 * np.pi * ni_up * Ti_up / Bmag_up**2).to("dimensionless"))

    print("\n--- Upstream state (region average) ---")
    print(f"  window = [{x[up].min().to('um'):.0f}, {x[up].max().to('um'):.0f}] "
          f"({int(up.sum())} points, {args.upstream_gap_um:.0f} µm ahead of the front)")
    # Is this window still pristine ambient?  A ray whose upstream has been processed
    # by the laser channel gives a valid shock solution against a preheated upstream —
    # which is a real result, not an error, but it must not be read as the experiment's
    # ambient.  The t=0 fill is the yardstick when the config records it.
    rho_0 = (cfg.get("unperturbed_background") or {}).get("rho_g_cm3")
    if rho_0:
        ratio = float(rho_up.to("g/cm**3").value) / float(rho_0)
        verdict = "pristine" if abs(ratio - 1.0) < 0.15 else "PROCESSED — not ambient"
        print(f"  rho/rho_0 = {ratio:.3f}  ({verdict}; t=0 fill {rho_0:.3e} g/cm^3)")
    print(f"  n_e   = {ne_up.to('cm**-3'):.3e}   n_ion = {ni_up.to('cm**-3'):.3e}   "
          f"Zbar = {zbar_up:.2f}")
    print(f"  T_e   = {Te_up.to('eV'):.1f}   T_i = {Ti_up.to('eV'):.1f}")
    print(f"  |B|   = {Bmag_up.to('gauss'):.1f} = {Bmag_up.to('T'):.2f}   "
          f"B_perp = {Bperp_up.to('gauss'):.1f}   theta_Bn = {np.degrees(theta_bn):.1f} deg")
    print(f"  beta_e = {beta_e:.3f}   beta_i = {beta_i:.3f}   "
          f"beta = {beta_e + beta_i:.3f}")
    print(f"  v_A   = {v_A.to('km/s'):.1f}   c_s = {c_s.to('km/s'):.1f}   "
          f"v_ms = {v_ms.to('km/s'):.1f}")
    print(f"  d_i   = {d_i:.1f}   1/w_ci = {inv_omega_ci:.1f}")
    print(f"\n--- Shock ---")
    print(f"  v_shock = {v_shock.to('km/s'):.1f}   v_inflow = {v_inflow.to('km/s'):.1f}")
    # M_ms is the one that decides whether a shock exists at all, and the one to
    # compare against the ~2.76 critical Mach number for ion reflection.
    print(f"  M_s = {mach_s:.2f}   M_A = {mach_a:.2f}   M_ms = {mach_ms:.2f}   "
          f"({'SUPER' if mach_ms > 2.76 else 'sub'}-critical; "
          f"ion-reflection threshold is 2.76)")
    print(f"\n--- Perpendicular MHD prediction (gamma = {gamma:.4f}) ---")
    print(f"  r = rho2/rho1 = {r:.3f}   p2/p1 = {p_ratio:.3f}   T2/T1 = {T_ratio:.3f}")
    print(f"  T2/T1 corrected for ionization = {T_ratio_ionized:.3f}   "
          f"(Zbar {zbar_up:.2f} -> {zbar_dn_jump:.2f} shares the pressure among "
          f"{1.0 / mu_ratio:.2f}x more particles per unit mass)")
    print(f"  ceiling at this gamma: r_max = {(gamma + 1) / (gamma - 1):.2f}")
    print(f"\n--- Compression measured vs predicted ---")
    print(f"  r measured (mass) = {r_measured:.3f}   vs predicted {r:.3f}"
          f"   [jump band {(bands.x_shock - bands.x_jump) * UM_PER_CM:.0f} µm]")
    print(f"  r over the full shocked layer = {r_layer:.3f}   "
          f"[{(bands.x_shock - bands.x_downstream) * UM_PER_CM:.0f} µm] — not an RH test, the layer\n"
          f"    holds material shocked earlier; quoted for the heating below")
    print(f"  Zbar {zbar_up:.2f} -> {zbar_dn:.2f} across the front")
    if np.isfinite(gamma_eff):
        print(f"  effective gamma reproducing the measurement = {gamma_eff:.3f}  "
              f"({2.0 / (gamma_eff - 1.0):.1f} DOF)")
        if gamma_eff < gamma:
            # An ionizing shock puts energy into stripping electrons instead of into
            # thermal pressure, which softens the index and raises the compression.
            print(f"  -> softer than {gamma:.3f}: energy is going somewhere other than "
                  f"thermal pressure (this front ionizes)")
    else:
        print("  effective gamma: no single-fluid index in [1.02, 1.95] reproduces "
              "this compression")
    sep = "-" * 64
    print(sep)
    print(f"  {'quantity':<22}{'upstream':>12}{'pred. dn':>12}{'meas. dn':>12}")
    print(sep)
    rows = [
        ("n_e [cm^-3]",        ne_up.to("cm**-3"),    ne_dn_pred.to("cm**-3"),    ne_dn_meas.to("cm**-3")),
        ("B_perp [G]",         Bperp_up.to("gauss"),  Bperp_dn_pred.to("gauss"),  Bperp_dn_meas.to("gauss")),
        ("T_e [eV]",           Te_up.to("eV"),        Te_dn_pred.to("eV"),        Te_dn_meas.to("eV")),
        ("T_i [eV]",           Ti_up.to("eV"),        Ti_dn_pred.to("eV"),        Ti_dn_meas.to("eV")),
        ("P_thermal [erg/cc]", P_up.to("erg/cm**3"),  P_dn_pred.to("erg/cm**3"),  P_dn_meas.to("erg/cm**3")),
        ("v_shockframe [km/s]",v_inflow.to("km/s"),   vsf_dn_pred.to("km/s"),     vsf_dn_meas.to("km/s")),
    ]
    for name, u, pdn, mdn in rows:
        print(f"  {name:<22}{float(u):>12.3e}{float(pdn):>12.3e}{float(mdn):>12.3e}")
    print(sep)

    # ------------------------------------------------------------------
    # Downstream heating and its electron/ion split
    # ------------------------------------------------------------------
    heating = fep.heating_partition(
        Te_up=Te_up.to("eV"), Ti_up=Ti_up.to("eV"),
        Te_dn=Te_dn_layer.to("eV"), Ti_dn=Ti_dn_layer.to("eV"),
        ne_up=ne_up.to("cm**-3"), ni_up=ni_up.to("cm**-3"),
        ne_dn=mean_dn(lo["ne"]).to("cm**-3"), ni_dn=mean_dn(lo["n_ion"]).to("cm**-3"),
        T_factor=T_ratio)
    print()
    print(fep.heating_summary(heating))

    # ------------------------------------------------------------------
    # Figure — lineouts across the front with the predicted downstream line
    # ------------------------------------------------------------------
    # Each panel: (display profile, upstream mean, predicted dn, measured dn,
    #              y-label, colour, log-y?).  Profiles/lines are converted to the
    #              panel's display unit with .to(...).value at draw time.
    # Row-major over the panel grid, so dropping an entry re-flows the layout.
    all_panels = [
        (lo["ne"].to("cm**-3").value,  ne_up.to("cm**-3"),  ne_dn_pred.to("cm**-3"),  ne_dn_meas.to("cm**-3"),
         r"$n_e$ [cm$^{-3}$]",        "tab:purple", True),
        (B_perp.to("gauss").value,     Bperp_up.to("gauss"), Bperp_dn_pred.to("gauss"), Bperp_dn_meas.to("gauss"),
         r"$B_\perp$ [G]",            "tab:orange", False),
        (v_sf.to("km/s").value,        v_inflow.to("km/s"), vsf_dn_pred.to("km/s"),   vsf_dn_meas.to("km/s"),
         r"$|v - v_{\rm sh}|$ [km/s]", "tab:blue", False),
        (lo["Te"].to("eV").value,      Te_up.to("eV"),      Te_dn_pred.to("eV"),      Te_dn_meas.to("eV"),
         r"$T_e$ [eV]",               "tab:green",  True),
        (lo["Ti"].to("eV").value,      Ti_up.to("eV"),      Ti_dn_pred.to("eV"),      Ti_dn_meas.to("eV"),
         r"$T_i$ [eV]",               "tab:brown",  True),
        (P_th.to("erg/cm**3").value,   P_up.to("erg/cm**3"),P_dn_pred.to("erg/cm**3"),P_dn_meas.to("erg/cm**3"),
         r"$P_{\rm thermal}$ [erg cm$^{-3}$]", "tab:red", True),
    ]

    # --publication trims to the four state variables the jump is read off. |v - v_sh| and
    # P_thermal are diagnostics rather than the result -- they duplicate what n_e and the
    # temperatures already show -- and a poster panel is better spent on fewer, larger
    # axes. Both are still solved, printed and written to the .npz either way.
    POSTER_PANELS = (0, 1, 3, 4)
    panels = ([all_panels[i] for i in POSTER_PANELS] if args.publication
              else all_panels)
    n_panel_cols = len(panels) // 2

    x_um       = x.to("um").value
    x_shock_um = float(x_shock.to("um"))
    x_ds_um    = float(x_ds.to("um"))
    x_up_lo_um = float(x[up].min().to("um"))
    x_up_hi_um = float(x[up].max().to("um"))
    x_contact_um = float(x_contact.to("um")) if np.isfinite(x_contact) else float("nan")

    # Line-outs with the jump parameters tabulated beside them, rather than carried in the
    # suptitle where a poster reader has to parse a run-on line to find r.
    fig = plt.figure(figsize=plot_style.figsize(5.0 * n_panel_cols + 4.0, 9),
                     layout="constrained")
    grid = fig.add_gridspec(2, n_panel_cols + 1,
                            width_ratios=(1.0,) * n_panel_cols + (0.5,))
    axes = np.empty((2, n_panel_cols), dtype=object)
    for row in range(2):
        for col in range(n_panel_cols):
            axes[row, col] = fig.add_subplot(
                grid[row, col],
                sharex=None if (row, col) == (0, 0) else axes[0, 0])
    table_ax = fig.add_subplot(grid[:, n_panel_cols])

    for ax, (prof, u_val, pred_val, meas_val, ylabel, color, log) in zip(
            axes.flat, panels):
        ax.plot(x_um, prof, color=color, lw=1.6, label="FLASH lineout")
        # Both averaging bands are shaded, because every number in the table is an
        # average over one of them: a band that has drifted onto the wrong plasma
        # should be visible in the figure rather than only in the printed verdict.
        ax.axvspan(x_ds_um, x_shock_um, color="tab:blue", alpha=0.12, lw=0,
                   label="downstream band")
        ax.axvspan(x_up_lo_um, x_up_hi_um, color="0.75", alpha=0.30, lw=0,
                   label="upstream band")
        ax.hlines(float(u_val), x_up_lo_um, x_up_hi_um, color="0.45", ls="-", lw=1.4,
                  label="upstream mean")
        # theory-predicted downstream value (over the downstream region)
        ax.hlines(float(pred_val), x_ds_um, x_shock_um, color="k", ls="--", lw=2.0,
                  label="RH predicted dn")
        # measured downstream mean
        ax.hlines(float(meas_val), x_ds_um, x_shock_um, color="k", ls=":", lw=2.0,
                  label="measured dn")
        ax.axvline(x_shock_um, color="k", lw=1.0, alpha=0.6)
        ax.axvline(x_ds_um,    color="0.6", lw=1.0, alpha=0.6)
        if np.isfinite(x_contact_um):
            ax.axvline(x_contact_um, color="tab:red", lw=1.2, ls="-.", alpha=0.8,
                       label="piston contact")
        if log:
            ax.set_yscale("log")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25, which="both")
    # sharex is wired up by hand here, so the inner tick labels have to be hidden by hand
    # too -- plt.subplots(sharex=True) would have done it.
    for ax in axes[:-1].flat:
        ax.tick_params(labelbottom=False)
    for ax in axes[-1]:
        ax.set_xlabel(r"distance along LOS [$\mu$m]")
    axes[0, 0].legend(loc="best", fontsize=8)

    # Zoom around the front (upstream to the right, downstream to the left).
    # Keep BOTH averaging regions in frame even when they reach past --window-um: every
    # number in the table comes from one of them, so a reader must be able to see the
    # profile each was taken over.
    lo_x = max(x_um.min(), min(x_shock_um - args.window_um, x_ds_um - 100.0))
    hi_x = min(x_um.max(), max(x_shock_um + args.window_um, x_up_hi_um + 100.0))
    axes[0, 0].set_xlim(lo_x, hi_x)
    # Late dumps put the front past 10 mm, so the default locator lays five-digit labels
    # end to end and they collide at publication font sizes. Shared axis: set once.
    axes[0, 0].xaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 5, 10]))

    # The jump parameters, in the slot the two dropped panels used to occupy. Small and
    # top-anchored so it can be cropped out of the figure and placed independently.
    table_ax.axis("off")
    jump_table = table_ax.table(
        cellText=[[r"$\theta_{Bn}$", f"{np.degrees(theta_bn):.0f}°"],
                  [r"$M_s$",         f"{mach_s:.2f}"],
                  [r"$M_A$",         f"{mach_a:.2f}"],
                  [r"$\gamma$",      f"{gamma:.3f}"],
                  [r"$r$",           f"{r:.2f}"]],
        cellLoc="center", loc="upper center", bbox=(0.08, 0.74, 0.84, 0.22))
    jump_table.auto_set_font_size(False)
    jump_table.set_fontsize(10)

    fig.suptitle(
        f"MHD Rankine-Hugoniot prediction vs FLASH "
        f"(t = {t.to('ns'):.2f}, {source.label})",
        fontsize=17,
    )
    fig_path = os.path.join(
        out_dir, f"flash_rh_prediction_{os.path.basename(snap_file)}.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved → {fig_path}")

    # ------------------------------------------------------------------
    # Save arrays (plain CGS magnitudes; unit named in each key)
    # ------------------------------------------------------------------
    npz_path = os.path.join(
        out_dir, f"flash_rh_prediction_{os.path.basename(snap_file)}.npz")
    np.savez(
        npz_path,
        x_um=x_um, t_ns=np.asarray(float(t.to("ns"))),
        ne=lo["ne"].to("cm**-3").value,
        B_perp=B_perp.to("gauss").value,
        Te=lo["Te"].to("eV").value,
        Ti=lo["Ti"].to("eV").value,
        P_th=P_th.to("erg/cm**3").value,
        v_sf_kms=v_sf.to("km/s").value,
        x_shock_cm=np.asarray(float(x_shock.to("cm"))),
        x_downstream_start_cm=np.asarray(float(x_ds.to("cm"))),
        v_shock_cms=np.asarray(float(v_shock.to("cm/s"))),
        gamma=np.asarray(gamma), theta_bn_rad=np.asarray(theta_bn),
        mach_s=np.asarray(mach_s), mach_a=np.asarray(mach_a),
        mach_ms=np.asarray(mach_ms),
        r=np.asarray(r), p_ratio=np.asarray(p_ratio), T_ratio=np.asarray(T_ratio),
        T_ratio_ionized=np.asarray(T_ratio_ionized),
        mu_ratio=np.asarray(mu_ratio), zbar_dn_jump=np.asarray(zbar_dn_jump),
        # upstream plasma state and the ion scales it sets
        los_label=np.asarray(source.label),
        dump_index=np.asarray(idx),
        x_upstream_lo_um=np.asarray(float(x[up].min().to("um"))),
        x_upstream_hi_um=np.asarray(float(x[up].max().to("um"))),
        x_contact_um=np.asarray(x_contact_um),
        downstream_edge=np.asarray(args.downstream_edge),
        rho_up_over_rho_0=np.asarray(
            float(rho_up.to("g/cm**3").value) / float(rho_0) if rho_0 else np.nan),
        zbar_up=np.asarray(zbar_up), zbar_dn=np.asarray(zbar_dn),
        r_measured=np.asarray(r_measured), gamma_eff=np.asarray(gamma_eff),
        r_layer=np.asarray(r_layer),
        x_jump_um=np.asarray(bands.x_jump * UM_PER_CM),
        rho_dn_meas=np.asarray(float(rho_dn_meas.to("g/cm**3"))),
        ni_up=np.asarray(float(ni_up.to("cm**-3"))),
        rho_up=np.asarray(float(rho_up.to("g/cm**3"))),
        Bmag_up=np.asarray(float(Bmag_up.to("gauss"))),
        beta_e_up=np.asarray(beta_e), beta_i_up=np.asarray(beta_i),
        v_A_kms=np.asarray(float(v_A.to("km/s"))),
        c_s_kms=np.asarray(float(c_s.to("km/s"))),
        v_ms_kms=np.asarray(float(v_ms.to("km/s"))),
        d_i_um=np.asarray(float(d_i)),
        inv_omega_ci_ns=np.asarray(float(inv_omega_ci)),
        # downstream heating and its electron/ion split
        heat_Te_adiabatic=np.asarray(heating["electron"]["adiabatic"]),
        heat_Te_excess=np.asarray(heating["electron"]["anomalous"]),
        heat_Te_excess_frac=np.asarray(heating["electron"]["anomalous_frac"]),
        heat_Ti_adiabatic=np.asarray(heating["ion"]["adiabatic"]),
        heat_Ti_excess=np.asarray(heating["ion"]["anomalous"]),
        heat_Ti_excess_frac=np.asarray(heating["ion"]["anomalous_frac"]),
        heat_Te_over_adiabatic=np.asarray(heating["electron"]["T_dn_over_adiabatic"]),
        heat_Ti_over_adiabatic=np.asarray(heating["ion"]["T_dn_over_adiabatic"]),
        du_th_e=np.asarray(float(heating["du_th_e"])),
        du_th_i=np.asarray(float(heating["du_th_i"])),
        f_e=np.asarray(heating["f_e"]), f_i=np.asarray(heating["f_i"]),
        # upstream / predicted-dn / measured-dn for each channel
        ne_up=np.asarray(float(ne_up.to("cm**-3"))),
        ne_dn_pred=np.asarray(float(ne_dn_pred.to("cm**-3"))),
        ne_dn_meas=np.asarray(float(ne_dn_meas.to("cm**-3"))),
        Bperp_up=np.asarray(float(Bperp_up.to("gauss"))),
        Bperp_dn_pred=np.asarray(float(Bperp_dn_pred.to("gauss"))),
        Bperp_dn_meas=np.asarray(float(Bperp_dn_meas.to("gauss"))),
        Te_up=np.asarray(float(Te_up.to("eV"))),
        Te_dn_pred=np.asarray(float(Te_dn_pred.to("eV"))),
        Te_dn_meas=np.asarray(float(Te_dn_meas.to("eV"))),
        Ti_up=np.asarray(float(Ti_up.to("eV"))),
        Ti_dn_pred=np.asarray(float(Ti_dn_pred.to("eV"))),
        Ti_dn_meas=np.asarray(float(Ti_dn_meas.to("eV"))),
        P_up=np.asarray(float(P_up.to("erg/cm**3"))),
        P_dn_pred=np.asarray(float(P_dn_pred.to("erg/cm**3"))),
        P_dn_meas=np.asarray(float(P_dn_meas.to("erg/cm**3"))),
        v_inflow_cms=np.asarray(float(v_inflow.to("cm/s"))),
        vsf_dn_pred_cms=np.asarray(float(vsf_dn_pred.to("cm/s"))),
        vsf_dn_meas_cms=np.asarray(float(vsf_dn_meas.to("cm/s"))),
        config_path=np.asarray(os.path.abspath(args.config)),
    )
    print(f"Saved → {npz_path}")


if __name__ == "__main__":
    main()
