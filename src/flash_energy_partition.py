# -*- coding: utf-8 -*-
"""flash_energy_partition.py — pure-function energy partition for FLASH shock runs.

All functions operate on physical (CGS) numpy arrays.  No I/O happens here;
the calling script handles loading via flash_utils.flash_lineout().

Energy density channels (erg/cm³):
    u_kinetic  : bulk kinetic (ram) in the shock rest frame  ½ ρ (v − v_shock)²
    u_th_e     : electron thermal  (3/2) nₑ kB Tₑ
    u_th_i     : ion thermal       (3/2) nᵢ kB Tᵢ   (nᵢ = nal + nsi)
    u_mag      : magnetic field    B²/(8π)  [Gaussian CGS]

The electron-vs-ion thermal partition is the quantity that has no OSIRIS analog
(OSIRIS uses phase-space moments; FLASH stores Tₑ and Tᵢ directly as fluid
fields).
"""

import numpy as np

import rankine_hugoniot as rh


def energy_densities(
    ne,
    Te,
    Ti,
    n_ion,
    rho,
    v_para,
    v_shock,
    B_mag,
) -> dict:
    """Compute energy-density profiles along the lineout.

    Inputs are unyt arrays (from flash_lineout), so the unit algebra is checked
    by the library: temperatures are energies (kT in eV), so nₑ·Tₑ is directly a
    pressure, and B²/(8π) reduces to erg/cm³ in Gaussian CGS.  Each channel is
    returned via ``.to("erg/cm**3")``, which raises if the dimensions are wrong.

    Parameters
    ----------
    ne          : electron number density [cm⁻³]
    Te, Ti      : electron / ion temperature as energies kT [eV]
    n_ion       : total ion number density [cm⁻³]
    rho         : mass density [g/cm³]
    v_para      : bulk velocity component along the lineout [cm/s], lab frame
    v_shock     : shock velocity [cm/s] (float or unyt), positive toward +x
    B_mag       : |B| [Gauss]

    Returns
    -------
    dict of 1-D unyt arrays, all in erg/cm³:
        u_kinetic, u_th_e, u_th_i, u_mag
    """
    import unyt

    # Give v_shock units if it came in as a bare float (interpreted as cm/s).
    if not isinstance(v_shock, (unyt.unyt_array, unyt.unyt_quantity)):
        v_shock = float(v_shock) * unyt.cm / unyt.s

    # Bulk kinetic (ram) energy in the shock rest frame
    dv = v_para - v_shock
    u_kinetic = (0.5 * rho * dv**2).to("erg/cm**3")

    # Thermal energy densities (3/2) n kT; Te, Ti are already kT in eV
    u_th_e = (1.5 * ne * Te).to("erg/cm**3")
    u_th_i = (1.5 * n_ion * Ti).to("erg/cm**3")

    # Magnetic energy density (Gaussian CGS): B²/(8π)
    u_mag = (B_mag**2 / (8.0 * np.pi)).to("erg/cm**3")

    return {
        "u_kinetic": u_kinetic,
        "u_th_e":    u_th_e,
        "u_th_i":    u_th_i,
        "u_mag":     u_mag,
    }


def momentum_fluxes(
    ne,
    Te,
    Ti,
    n_ion,
    rho,
    v_para,
    v_shock,
    B_mag,
    B_para=None,
) -> dict:
    """Normal momentum-flux (total-pressure) channels along the lineout.

    The shock-frame normal momentum flux ``ρ u_n² + p + B_t²/8π`` is the
    conserved Rankine--Hugoniot quantity: it is what is *continuous* across the
    front (``[ρ v_n² + p + B²/2μ₀] = 0``), unlike the summed energy density.
    Each channel is a **pressure** (dyn/cm² ≡ erg/cm³), so the prefactors differ
    from :func:`energy_densities`:

        p_ram   = ρ (v_para − v_shock)²    ram (dynamic) pressure  [= 2·u_kinetic]
        p_th_e  = nₑ kTₑ                    electron pressure       [= (2/3)·u_th_e]
        p_th_i  = nᵢ kTᵢ                    ion pressure            [= (2/3)·u_th_i]
        p_mag   = B_t²/8π                   magnetic pressure       [= u_mag if perp]

    Inputs and units mirror :func:`energy_densities`.  ``B_t`` is the transverse
    field ``sqrt(B_mag² − B_para²)`` when ``B_para`` is supplied — the normal
    field B_n is continuous and contributes magnetic *tension*, not normal
    momentum flux, so it is excluded; with ``B_para=None`` the full ``B_mag`` is
    used (perpendicular-shock assumption).

    Returns
    -------
    dict of 1-D unyt arrays in dyn/cm²: ``p_ram, p_th_e, p_th_i, p_mag``.
    """
    import unyt

    if not isinstance(v_shock, (unyt.unyt_array, unyt.unyt_quantity)):
        v_shock = float(v_shock) * unyt.cm / unyt.s

    dv = v_para - v_shock
    p_ram = (rho * dv**2).to("dyn/cm**2")

    p_th_e = (ne * Te).to("dyn/cm**2")
    p_th_i = (n_ion * Ti).to("dyn/cm**2")

    if B_para is not None:
        bt2 = B_mag**2 - B_para**2
        bt2 = unyt.unyt_array(np.clip(bt2.value, 0.0, None), bt2.units)
    else:
        bt2 = B_mag**2
    p_mag = (bt2 / (8.0 * np.pi)).to("dyn/cm**2")

    return {
        "p_ram":   p_ram,
        "p_th_e":  p_th_e,
        "p_th_i":  p_th_i,
        "p_mag":   p_mag,
    }


def partition_by_region(
    energy: dict,
    x:      np.ndarray,
    x_shock: float,
    x_downstream_start: float,
) -> dict:
    """Average each energy channel over upstream and downstream windows.

    Parameters
    ----------
    energy              : dict from energy_densities()
    x                   : spatial coordinate [cm], same grid as energy arrays
    x_shock             : shock position [cm]; upstream is x > x_shock
    x_downstream_start  : left edge of downstream region [cm]

    Returns
    -------
    dict with keys "upstream" and "downstream".  Each is a dict:
        channel -> mean energy density [erg/cm³]
    Plus a "fractions" sub-dict giving each channel as a fraction of the
    total for that region.
    """
    upstream   = x > x_shock
    downstream = (x >= x_downstream_start) & (x <= x_shock)

    if not upstream.any() or not downstream.any():
        raise ValueError(
            f"Empty region — check x_shock={x_shock:.4g} cm and "
            f"x_downstream_start={x_downstream_start:.4g} cm "
            f"vs x=[{x.min():.4g}, {x.max():.4g}] cm"
        )

    def _side(mask: np.ndarray) -> dict:
        # energy[k] are unyt erg/cm³ arrays; reduce to plain floats [erg/cm³].
        means = {k: float(np.nanmean(v[mask]).to("erg/cm**3").value)
                 for k, v in energy.items()}
        total = sum(means.values())
        fracs = {k: v / total if total > 0 else float("nan")
                 for k, v in means.items()}
        return {"means": means, "fractions": fracs, "total": total}

    return {
        "upstream":   _side(upstream),
        "downstream": _side(downstream),
    }


def compression_check(
    prim_up: dict,
    prim_dn: dict,
    v_inflow,
    gamma: float = rh.GAMMA_DEFAULT,
) -> dict:
    """Measure the FLASH shock compression two ways and compare to MHD theory.

    Validates the FLASH source data against the *same* oblique Rankine--Hugoniot
    theory used for the OSIRIS analysis (``src/rankine_hugoniot.py``), reused
    directly — the RH core is unit-agnostic, so no jump physics is duplicated in
    CGS here.  The upstream Mach numbers are built in Gaussian CGS with the unit
    algebra checked by ``unyt`` (mirrors ``flash_utils.mach_numbers``).

    The compression is read off two independent ways:
      * density:        r = rho_dn / rho_up
      * transverse B:   |Bt_dn| / |Bt_up|   (Bt = sqrt(B_mag^2 - B_para^2))
    and both are compared to the theory r(M_s, M_A, theta) and Bt2/Bt1.  The two
    measured readings only coincide for a perpendicular shock; the predicted
    Bt2/Bt1 reconciles them given the measured obliquity.

    Parameters
    ----------
    prim_up, prim_dn : dict
        Region-averaged CGS floats with keys ``rho`` [g/cm^3], ``ne`` [cm^-3],
        ``n_ion`` [cm^-3], ``Te`` [eV], ``Ti`` [eV], ``B_mag`` [Gauss],
        ``B_para`` [Gauss] (B component along the shock normal / LOS).
    v_inflow : float
        Shock-frame normal inflow speed [cm/s] (e.g. |v_shock - v_para_up|).
    gamma : float
        Adiabatic index for the RH baseline (default 5/3); sweep to fit the
        effective index (see ``rankine_hugoniot.gamma_from_dof``).

    Returns
    -------
    dict with keys ``theta_bn`` [rad], ``mach_s``, ``mach_a``, ``r_measured``,
    ``r_RH``, ``b_t_measured``, ``b_t_RH``, ``gamma``.
    """
    import unyt

    # --- upstream obliquity from the normal/total field ---
    B_mag_up = float(prim_up["B_mag"])
    B_para_up = float(prim_up["B_para"])
    B_perp_up = float(np.sqrt(max(B_mag_up**2 - B_para_up**2, 0.0)))
    theta_bn = float(np.arctan2(B_perp_up, abs(B_para_up)))

    # --- upstream Mach numbers in Gaussian CGS (unit algebra checked by unyt) ---
    ne = float(prim_up["ne"]) * unyt.cm**-3
    n_i = float(prim_up["n_ion"]) * unyt.cm**-3
    Te = float(prim_up["Te"]) * unyt.eV
    Ti = float(prim_up["Ti"]) * unyt.eV
    B = B_mag_up * unyt.Gauss                       # total field sets v_A
    rho = float(prim_up["rho"]) * unyt.g / unyt.cm**3
    v_in = float(v_inflow) * unyt.cm / unyt.s

    v_A = (B / np.sqrt(4.0 * np.pi * rho)).to("cm/s")
    P_th = (ne * Te + n_i * Ti).to("erg/cm**3")
    c_s = np.sqrt(gamma * P_th / rho).to("cm/s")
    mach_a = float((v_in / v_A).to("dimensionless")) if float(v_A) > 0 else float("inf")
    mach_s = float((v_in / c_s).to("dimensionless")) if float(c_s) > 0 else float("inf")

    # --- theory ---
    r_RH = rh.compression_ratio(mach_s, mach_a, theta=theta_bn, gamma=gamma)
    b_t_RH = rh.tangential_field_ratio(r_RH, mach_a, theta_bn)

    # --- measurements ---
    rho_dn = float(prim_dn["rho"])
    r_measured = rho_dn / float(prim_up["rho"]) if float(prim_up["rho"]) > 0 else float("nan")
    B_perp_dn = float(np.sqrt(max(float(prim_dn["B_mag"])**2
                                  - float(prim_dn["B_para"])**2, 0.0)))
    b_t_measured = B_perp_dn / B_perp_up if B_perp_up > 0 else float("nan")

    return {
        "theta_bn": theta_bn,
        "mach_s": mach_s, "mach_a": mach_a,
        "r_measured": r_measured, "r_RH": r_RH,
        "b_t_measured": b_t_measured, "b_t_RH": b_t_RH,
        "gamma": gamma,
    }


def compression_summary(check: dict) -> str:
    """Return a formatted table string from :func:`compression_check` output."""
    sep = "-" * 56
    deg = np.degrees(check["theta_bn"])
    return "\n".join([
        sep,
        f"  Compression vs Rankine--Hugoniot (gamma = {check['gamma']:.4f})",
        sep,
        f"  theta_Bn = {deg:6.1f} deg   M_s = {check['mach_s']:6.2f}   "
        f"M_A = {check['mach_a']:6.2f}",
        f"  {'':<14}{'measured':>12}{'RH theory':>12}",
        f"  {'r (density)':<14}{check['r_measured']:>12.3f}{check['r_RH']:>12.3f}",
        f"  {'Bt2/Bt1':<14}{check['b_t_measured']:>12.3f}{check['b_t_RH']:>12.3f}",
        sep,
    ])


def partition_summary(result: dict, channels=None, labels=None,
                      unit: str = "erg/cm³") -> str:
    """Return a formatted table string from partition_by_region output.

    Defaults to the energy-density channels; pass ``channels``/``labels`` (and
    ``unit``) to format a momentum-flux partition (see :func:`momentum_fluxes`),
    e.g. ``channels=["p_ram","p_th_e","p_th_i","p_mag"]``.
    """
    if channels is None:
        channels = ["u_kinetic", "u_th_e", "u_th_i", "u_mag"]
        labels   = ["Kinetic (ram)", "Thermal e⁻", "Thermal i⁺", "Magnetic"]
    sep = "-" * 68
    lines = [
        sep,
        f"  {'Channel':<16} {'Upstream [' + unit + ']':>20} {'(%)':>6}  "
        f"{'Downstream [' + unit + ']':>20} {'(%)':>6}",
        sep,
    ]
    for ch, lbl in zip(channels, labels):
        up = result["upstream"]
        dn = result["downstream"]
        lines.append(
            f"  {lbl:<16} {up['means'][ch]:>20.3e} {100*up['fractions'][ch]:>5.1f}%  "
            f"{dn['means'][ch]:>20.3e} {100*dn['fractions'][ch]:>5.1f}%"
        )
    lines += [
        sep,
        f"  {'Total':<16} {result['upstream']['total']:>20.3e} {'':>6}  "
        f"{result['downstream']['total']:>20.3e}",
        sep,
    ]
    return "\n".join(lines)


def continuity_check(result: dict) -> dict:
    """Downstream/upstream ratio of the total and each channel.

    For a conserved flux (the momentum flux of :func:`momentum_fluxes`) the total
    ``dn/up`` ratio is ≈ 1 across a steady shock — the quantitative continuity
    test.  (Applied to the non-conserved energy-density partition it just
    reports how much the density jumps, which is *not* expected to be 1.)

    Returns
    -------
    dict with ``total_up``, ``total_dn``, ``ratio`` (= dn/up), ``rel_imbalance``
    (= (dn−up)/up), and a ``channels`` sub-dict of per-channel dn/up ratios.
    """
    up = result["upstream"]["total"]
    dn = result["downstream"]["total"]
    ratio = dn / up if up else float("nan")
    channels = {
        ch: (result["downstream"]["means"][ch] / result["upstream"]["means"][ch]
             if result["upstream"]["means"][ch] else float("nan"))
        for ch in result["upstream"]["means"]
    }
    return {
        "total_up": up,
        "total_dn": dn,
        "ratio": ratio,
        "rel_imbalance": (dn - up) / up if up else float("nan"),
        "channels": channels,
    }


def continuity_summary(check: dict, unit: str = "dyn/cm²") -> str:
    """Return a formatted continuity-check table from :func:`continuity_check`."""
    sep = "-" * 56
    return "\n".join([
        sep,
        "  Momentum-flux continuity  (conserved if dn/up ≈ 1)",
        sep,
        f"  total upstream   = {check['total_up']:.3e} {unit}",
        f"  total downstream = {check['total_dn']:.3e} {unit}",
        f"  dn/up = {check['ratio']:.3f}   "
        f"({100 * check['rel_imbalance']:+.1f}%)",
        sep,
    ])
