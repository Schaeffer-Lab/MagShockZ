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


def partition_summary(result: dict) -> str:
    """Return a formatted table string from partition_by_region output."""
    channels = ["u_kinetic", "u_th_e", "u_th_i", "u_mag"]
    labels   = ["Kinetic (ram)", "Thermal e⁻", "Thermal i⁺", "Magnetic"]
    sep = "-" * 68
    lines = [
        sep,
        f"  {'Channel':<16} {'Upstream [erg/cm³]':>20} {'(%)':>6}  "
        f"{'Downstream [erg/cm³]':>20} {'(%)':>6}",
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
