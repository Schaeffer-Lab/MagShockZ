"""Synthetic (forward-modelled) experimental diagnostics.

The MagShockZ campaign has no Thomson scattering, so the simulated heating /
partition prediction can only be tested through its *observable consequences*:
the T_e-sensitive **X-ray emission**, line-integrated **density** (imaging /
interferometry), and the **magnetic field** seen by probes / radiography.  These
pure functions forward-model those signals from simulation fields — degrading to
the instrument's spatial resolution and integrating along the line of sight — so
a run can be compared apples-to-apples with the measurement.

Everything is unit-agnostic and numpy/scipy-only, so the same code serves the
OSIRIS (normalised) and FLASH (CGS) outputs; the calling script supplies fields
in whatever consistent unit system it works in.
"""

import numpy as np
import scipy.integrate
import scipy.ndimage

# Conversion FWHM -> Gaussian sigma.
_FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def bremsstrahlung_emissivity(n_e, T_e, species, hnu=None, gaunt=1.2):
    """Free-free (bremsstrahlung) emissivity from electrons + ion species.

    Frequency-integrated (``hnu is None``):
        eps ∝ gaunt * n_e * Σ_s (n_s Z_s^2) * sqrt(T_e)
    Spectral at photon energy ``hnu`` (same units as ``T_e``):
        eps ∝ gaunt * n_e * Σ_s (n_s Z_s^2) / sqrt(T_e) * exp(-hnu / T_e)

    The leading physical constant is dropped, so the result is a *relative*
    emissivity — appropriate for synthetic images/spectra that are compared in
    shape and contrast, not absolute brightness.

    Parameters
    ----------
    n_e, T_e : array_like
        Electron density and temperature profiles (T_e > 0).
    species : sequence of (n_i, Z)
        Per-species ion density and charge state; contributions add as n_i Z^2.
    hnu : array_like, optional
        Photon energy/energies for a spectral emissivity (same units as T_e).
    gaunt : float
        Effective Gaunt factor (order unity).
    """
    n_e = np.asarray(n_e, dtype=float)
    T_e = np.asarray(T_e, dtype=float)
    sum_niZ2 = sum(np.asarray(n_i, dtype=float) * float(Z) ** 2 for n_i, Z in species)
    Te_safe = np.maximum(T_e, np.finfo(float).tiny)
    if hnu is None:
        return gaunt * n_e * sum_niZ2 * np.sqrt(Te_safe)
    hnu = np.asarray(hnu, dtype=float)
    # Broadcast spectral axis against the spatial profile if needed.
    if hnu.ndim and n_e.ndim:
        Te_safe = Te_safe[..., None]
        n_e = n_e[..., None]
        sum_niZ2 = sum_niZ2[..., None] if np.ndim(sum_niZ2) else sum_niZ2
    return gaunt * n_e * sum_niZ2 / np.sqrt(Te_safe) * np.exp(-hnu / Te_safe)


def line_of_sight_integral(quantity, coord, axis=-1):
    """∫ quantity d(coord) along ``axis`` (trapezoidal).

    For a 1-D profile this returns the column value; for a 2-D map it returns the
    1-D synthetic image (interferometry column density, X-ray image row, ...).
    """
    return scipy.integrate.trapezoid(np.asarray(quantity, dtype=float),
                                     np.asarray(coord, dtype=float), axis=axis)


def apply_resolution(signal, coord, fwhm):
    """Convolve a 1-D signal with a Gaussian of the given spatial FWHM.

    ``coord`` must be uniformly spaced; ``fwhm`` is in the same units as
    ``coord``.  Gaussian smoothing conserves the integral (area), so column
    densities are preserved while sharp structure is blurred to the instrument
    resolution.
    """
    signal = np.asarray(signal, dtype=float)
    coord = np.asarray(coord, dtype=float)
    if fwhm <= 0:
        return signal.copy()
    dcoord = float(np.mean(np.diff(coord)))
    sigma_pix = (fwhm * _FWHM_TO_SIGMA) / abs(dcoord)
    return scipy.ndimage.gaussian_filter1d(signal, sigma_pix, mode="nearest")


def probe_signal(field, coord, probe_coords, fwhm=0.0):
    """Sample ``field`` at ``probe_coords`` after applying instrument resolution.

    Models a point probe (B-dot, Faraday rotation spot) with finite spatial
    response ``fwhm``.
    """
    blurred = apply_resolution(field, coord, fwhm)
    return np.interp(np.asarray(probe_coords, dtype=float),
                     np.asarray(coord, dtype=float), blurred)
