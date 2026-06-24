# -*- coding: utf-8 -*-
"""flash_utils.py — FLASH I/O and lineout helpers for MagShockZ analysis.

Everything here is pure FLASH + yt: a plot file is opened with ``yt.load`` and
sampled with yt's builtin ``LineBuffer`` lineout, reading the native ``("gas",…)``
derived fields, which already carry physical CGS units.  No OSIRIS normalisation,
``rqm_factor`` or reference density is involved.

The lineout returns **unyt arrays** (yt.units) so the units travel with the data:
  x          : cm   (distance along the LOS, zero at start_pt)
  density    : cm⁻³
  temperature: eV   (kT, via the yt "thermal" equivalence)
  B-field    : Gauss
  velocity   : cm/s
  rho        : g/cm³
  time       : s    (scalar float, not an array)
"""

import functools
import glob
import os
from multiprocessing import Pool

import numpy as np
import yt


def find_plot_files(data_dir: str) -> list:
    """Return sorted list of FLASH plot-file paths in data_dir."""
    pattern = os.path.join(data_dir, "MagShockZ_hdf5_plt_cnt_*")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No FLASH plot files found in {data_dir}")
    return files


def flash_time_s(path: str) -> float:
    """Read simulation time [s] from a FLASH plot file without loading the full dataset."""
    import h5py
    with h5py.File(path, "r") as f:
        scalars = {k.decode().strip(): float(v) for k, v in f["real scalars"][:]}
    return scalars["time"]


def flash_lineout(
    path: str,
    start_pt: tuple,
    end_pt: tuple,
    npoints: int = 512,
) -> dict:
    """Extract a lineout from a FLASH plot file along the given ray.

    Uses ``yt.load`` + ``yt.LineBuffer`` and the native ``("gas",…)`` derived
    fields, so every returned array is a unyt array already in physical CGS.

    Parameters
    ----------
    path : str
        Path to a FLASH plot file (MagShockZ_hdf5_plt_cnt_XXXX).
    start_pt, end_pt : tuple of 3 floats
        LOS endpoints in cm (as in the runme --start_point / --end_point).
    npoints : int
        Number of points sampled uniformly along the LOS.

    Returns
    -------
    dict of 1-D unyt arrays (CGS units travel with the data):
        x       : distance along LOS from start_pt [cm], zero at start_pt
        ne      : electron number density [cm⁻³]
        n_ion   : total ion number density [cm⁻³]
        Te      : electron temperature [eV]  (kT)
        Ti      : ion temperature [eV]       (kT)
        B_para  : magnetic field component along LOS [Gauss]  (B·n̂)
        B_mag   : |B| = sqrt(Bx²+By²+Bz²) [Gauss]
        v_para  : bulk velocity along LOS [cm/s]  (v·n̂, lab frame)
        rho     : mass density [g/cm³]
        t_s     : simulation time [s]  (scalar float, not an array)
    """
    ds = yt.load(path)

    start  = np.asarray(start_pt, dtype=float)
    end    = np.asarray(end_pt,   dtype=float)
    seg    = end - start
    length = float(np.linalg.norm(seg))
    nhat   = seg / length                       # dimensionless LOS unit vector

    lb = yt.LineBuffer(ds, start, end, npoints)

    # distance along LOS [cm], zero at start_pt
    x = np.linspace(0.0, length, npoints) * yt.units.cm

    ne    = lb[("gas", "El_number_density")].to("cm**-3")
    n_ion = lb[("gas", "ion_number_density")].to("cm**-3")

    # FLASH stores tele/tion in K (1 code_temperature == 1 K); the "thermal"
    # equivalence converts T → kT, i.e. temperature expressed as energy in eV.
    Te = lb[("flash", "tele")].to("K").to("eV", "thermal")
    Ti = lb[("flash", "tion")].to("K").to("eV", "thermal")

    rho = lb[("gas", "density")].to("g/cm**3")

    vx = lb[("gas", "velocity_x")].to("cm/s")
    vy = lb[("gas", "velocity_y")].to("cm/s")
    vz = lb[("gas", "velocity_z")].to("cm/s")
    v_para = vx * nhat[0] + vy * nhat[1] + vz * nhat[2]   # along LOS [cm/s]

    Bx = lb[("gas", "magnetic_field_x")].to("G")
    By = lb[("gas", "magnetic_field_y")].to("G")
    Bz = lb[("gas", "magnetic_field_z")].to("G")
    B_mag  = np.sqrt(Bx**2 + By**2 + Bz**2)              # |B| [Gauss]
    B_para = Bx * nhat[0] + By * nhat[1] + Bz * nhat[2]  # along LOS [Gauss]

    t_s = float(ds.current_time.to("s"))

    return {
        "x":      x,
        "ne":     ne,
        "n_ion":  n_ion,
        "Te":     Te,
        "Ti":     Ti,
        "B_para": B_para,
        "B_mag":  B_mag,
        "v_para": v_para,
        "rho":    rho,
        "t_s":    t_s,
    }


# ---------------------------------------------------------------------------
# Multi-dump lineout loading
# ---------------------------------------------------------------------------

def _load_one(path, start_pt, end_pt):
    """Picklable multiprocessing worker: one independent dump → its lineout dict."""
    return flash_lineout(path, start_pt, end_pt)


def load_lineouts(paths: list, start_pt: tuple, end_pt: tuple, nprocs: int = 1) -> list:
    """Load each dump's lineout (in input order), fanning the dumps across processes.

    ``paths`` is a list of FLASH plot-file paths; ``start_pt`` / ``end_pt`` are the LOS
    endpoints in cm.  Each dump is independent, so with ``nprocs > 1`` they load in
    parallel via :class:`multiprocessing.Pool` (``imap`` preserves order, so ``out[i]``
    still matches ``paths[i]``).  Returns the list of per-dump dicts from
    :func:`flash_lineout`, printing a ``[i/N] <file>`` progress line per dump.
    """
    worker = functools.partial(_load_one, start_pt=start_pt, end_pt=end_pt)
    out = []
    if nprocs <= 1:
        for i, p in enumerate(paths):
            print(f"  [{i + 1:3d}/{len(paths)}] {os.path.basename(p)}", flush=True)
            out.append(worker(p))
        return out
    with Pool(nprocs) as pool:
        for i, lo in enumerate(pool.imap(worker, paths)):
            print(f"  [{i + 1:3d}/{len(paths)}] {os.path.basename(paths[i])}", flush=True)
            out.append(lo)
    return out


# ---------------------------------------------------------------------------
# Shock-front detection — single-sourced in src/shock.py
# ---------------------------------------------------------------------------

import shock as _shock


def detect_front(
    x: np.ndarray,
    ne: np.ndarray,
    x_pred: float,
    half_window: float,
    compression_min: float = 1.3,
    smooth: int = 3,
) -> float:
    """FLASH shock front (steepest density drop) near x_pred, in physical CGS units.

    Thin wrapper over :func:`shock.detect_front_gradient` (the shared
    implementation); kept here as ``flash_utils.detect_front`` for the existing
    ``fu.detect_front`` call sites.
    """
    return _shock.detect_front_gradient(
        x, ne, x_pred, half_window,
        compression_min=compression_min, smooth=smooth)


# ---------------------------------------------------------------------------
# Mach-number helpers (unit-checked via yt/unyt — Gaussian CGS throughout)
# ---------------------------------------------------------------------------

def mach_numbers(
    ne_cm3: float,
    n_ion_cm3: float,
    Te_eV: float,
    Ti_eV: float,
    B_gauss: float,
    rho_gcm3: float,
    v_shock_cms: float,
    gamma: float = 5.0 / 3.0,
) -> dict:
    """Compute Alfvénic and sonic Mach numbers from upstream averages.

    Every dimensionful quantity is carried as a unyt (yt.units) quantity so the
    unit algebra is checked by the library rather than tracked by hand.  In
    particular the Alfvén speed is formed in Gaussian CGS, where 1 Gauss =
    g^½ cm^-½ s^-1, so B / sqrt(4π ρ) reduces to cm/s automatically; the final
    ``.to("cm/s")`` would raise if the dimensions did not work out.

    Inputs are bare floats in the CGS unit named by each argument:
        ne_cm3, n_ion_cm3 : number densities [cm⁻³]
        Te_eV, Ti_eV      : temperatures as energies kT [eV]
        B_gauss           : magnetic-field magnitude [Gauss]
        rho_gcm3          : mass density [g/cm³]
        v_shock_cms       : shock speed [cm/s]

    Returns a dict of unyt quantities (use ``.to(...)`` / ``float(...)`` at the
    call site):
        v_A   : Alfvén speed                         [cm/s]
        c_s   : sound speed                          [cm/s]
        M_A   : v_shock / v_A                        [dimensionless]
        M_s   : v_shock / c_s                        [dimensionless]
        beta  : thermal pressure / magnetic pressure [dimensionless]
    """
    from unyt import Gauss, cm, eV, g, s

    ne   = ne_cm3      * cm**-3
    n_i  = n_ion_cm3   * cm**-3
    Te   = Te_eV       * eV
    Ti   = Ti_eV       * eV
    B    = B_gauss     * Gauss
    rho  = rho_gcm3    * g / cm**3
    v_sh = v_shock_cms * cm / s

    # ---- Alfvén speed: v_A = B / sqrt(4π ρ)  (Gaussian CGS) ----
    # Gauss·cm^(3/2)/g^(1/2) reduces to cm/s; .to() asserts that algebra.
    v_A = (B / np.sqrt(4.0 * np.pi * rho)).to("cm/s")

    # ---- Thermal pressure: P_th = nₑ kTₑ + nᵢ kTᵢ ----
    # Te, Ti are already energies (kT in eV), so nₑ·Tₑ is directly a pressure.
    P_th = (ne * Te + n_i * Ti).to("erg/cm**3")

    # ---- Sound speed: c_s = sqrt(γ P_th / ρ) ----
    c_s = np.sqrt(gamma * P_th / rho).to("cm/s")

    # ---- Magnetic pressure and plasma beta (Gaussian CGS) ----
    P_B  = (B**2 / (8.0 * np.pi)).to("erg/cm**3")
    beta = (P_th / P_B).to("dimensionless")

    M_A = (v_sh / v_A).to("dimensionless")
    M_s = (v_sh / c_s).to("dimensionless")

    return {"v_A": v_A, "c_s": c_s, "M_A": M_A, "M_s": M_s, "beta": beta}


