"""piston_profile.py — measure an expanding piston from 1-D line-outs.

The numerical core behind ``scripts/flash_piston_profile.py``: locate the piston
front along a line of sight, fit its trajectory over a time window, fit the density
e-folding length behind it, average the ambient just ahead of it, and collapse the
per-dump profiles onto one self-similar curve.

Deliberately unit-agnostic and IO-free — plain numpy arrays in, plain numpy out, with
the caller responsible for units (the FLASH script hands it CGS; the WarpX comparison
script hands it normalized units, and the same functions measure both, which is what
makes the two codes comparable).  Positions are "distance along the line of sight",
increasing outward from the piston, so the front is the OUTERMOST crossing.

Unit-tested in CI (numpy only, no yt / WarpX).
"""

from dataclasses import dataclass

import numpy as np

#: Default fraction of the piston peak that defines the front.  0.1 sits on the steep
#: flank for a roughly exponential piston edge — high enough to clear line-out noise,
#: low enough that the front tracks the leading edge rather than the bulk.
FRONT_THRESHOLD_DEFAULT = 0.1


def _finite_max(values: np.ndarray) -> float:
    """Largest finite entry, or nan if there is none.

    ``np.nanmax`` would do this but warns on an all-nan slice, and every caller here
    reaches that case on a documented, expected nan path (an empty line-out).
    """
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return float(finite.max()) if finite.size else float("nan")


def piston_ion_density(mass_density: np.ndarray, mass_fraction: np.ndarray,
                       mass_number: float, atomic_mass_unit: float) -> np.ndarray:
    """Piston ion number density from FLASH's mass density and material fraction.

    ``n_i = rho * X / (A m_u)`` is exact given the material's mass fraction, and — the
    reason it is used in preference to the yt plugin's ``<material>dens`` — it keeps the
    piston's smooth exponential tail.  That field masks cells by *dominant* material,
    which truncates the tail at the very place :func:`efolding_length` measures it.

    Units follow the inputs: pass ``mass_density`` in g/cm^3 with
    ``atomic_mass_unit`` in g to get cm^-3, or SI to get m^-3.
    """
    return np.asarray(mass_density) * np.asarray(mass_fraction) / (
        mass_number * atomic_mass_unit)


def ambient_reference_level(positions: np.ndarray, density: np.ndarray,
                            outer_fraction: float = 0.15) -> float:
    """Median ``density`` over the outermost ``outer_fraction`` of the line-out.

    The absolute reference :func:`front_position` needs.  A median, not a mean, so a
    single spike in the far field cannot move it.
    """
    positions = np.asarray(positions, dtype=float)
    density = np.asarray(density, dtype=float)
    if positions.size < 2 or not 0.0 < outer_fraction <= 1.0:
        return float("nan")

    span = positions[-1] - positions[0]
    outer = positions >= positions[-1] - outer_fraction * span
    usable = outer & np.isfinite(density)
    if not np.any(usable):
        return float("nan")
    return float(np.median(density[usable]))


def front_position(positions: np.ndarray, density: np.ndarray,
                   threshold_frac: float = FRONT_THRESHOLD_DEFAULT,
                   level: float | None = None) -> float:
    """Outermost position where ``density`` falls through a threshold.

    With ``level`` given the threshold is ABSOLUTE; otherwise it is
    ``threshold_frac`` of the profile's peak.

    Prefer the absolute form for a piston whose profile has more than one hump.  On the
    MagShockZ line-outs the dense inner plume overtakes the leading edge partway through
    the run, and a peak-relative threshold then jumps backwards from the leading edge to
    the inner plume — a front that appears to move at -390 km/s while every individual
    position is increasing.  An absolute level tied to the ambient density (see
    :func:`ambient_reference_level`) does not care which hump is tallest.

    Linearly interpolated between the bracketing samples, so the answer is not quantised
    to the line-out spacing.  Returns nan when the profile never reaches the threshold.

    Searching outward-in from the far end, rather than taking the steepest gradient,
    makes this robust to multi-peaked pistons: what is wanted is the leading edge of
    piston material, not the strongest jump.
    """
    positions = np.asarray(positions, dtype=float)
    density = np.asarray(density, dtype=float)
    if positions.size != density.size or positions.size < 2:
        return float("nan")

    if level is None:
        peak = _finite_max(density)
        if not (peak > 0.0):
            return float("nan")
        level = threshold_frac * peak
    elif not (np.isfinite(level) and level > 0.0):
        return float("nan")

    above = np.flatnonzero(density >= level)
    if above.size == 0:
        return float("nan")
    last = int(above[-1])
    if last == density.size - 1:
        # Piston material reaches the end of the line-out: the front is outside the
        # window, so report the edge rather than extrapolating into nothing.
        return float(positions[-1])

    lo, hi = density[last], density[last + 1]
    if lo == hi:
        return float(positions[last])
    weight = (lo - level) / (lo - hi)
    return float(positions[last] + weight * (positions[last + 1] - positions[last]))


def efolding_length(positions: np.ndarray, density: np.ndarray, x_front: float,
                    window: float, min_points: int = 4) -> float:
    """Fit ``n ~ exp(-x/L)`` over ``window`` inward from ``x_front``; return ``L``.

    A straight least-squares fit of ``log(n)`` against position, which is the piston's
    density scale length — the quantity ``docs/piston_interface_smoothing_plan.md``
    argues should be set by the directed ion gyroradius rather than chosen.  Returns
    nan if fewer than ``min_points`` positive samples fall in the window.
    """
    positions = np.asarray(positions, dtype=float)
    density = np.asarray(density, dtype=float)
    if not np.isfinite(x_front) or window <= 0.0:
        return float("nan")

    in_window = (positions >= x_front - window) & (positions <= x_front)
    usable = in_window & np.isfinite(density) & (density > 0.0)
    if int(np.count_nonzero(usable)) < min_points:
        return float("nan")

    slope, _ = np.polyfit(positions[usable], np.log(density[usable]), 1)
    if slope >= 0.0:
        # Density rising outward is not a decaying edge; refuse to name a length.
        return float("nan")
    return float(-1.0 / slope)


@dataclass(frozen=True)
class FrontTrajectory:
    """Straight-line fit ``x(t) = x0 + v (t - t0)`` to a measured front."""

    speed: float
    x0: float
    t0: float
    residual_rms: float
    n_points: int

    def at(self, times: np.ndarray | float) -> np.ndarray:
        """Fitted front position at ``times``."""
        return self.x0 + self.speed * (np.asarray(times, dtype=float) - self.t0)


def fit_front_trajectory(times: np.ndarray, positions: np.ndarray) -> FrontTrajectory:
    """Least-squares straight line through the measured front positions.

    ``t0`` is the first finite sample time, so ``x0`` is the front position at the start
    of the window rather than a back-extrapolation to t = 0 — the same anchoring choice
    ``tune_flash_shock.py``'s trajectory mode makes, and for the same reason: a piston
    that forms mid-run must be fitted from its formation time.
    """
    times = np.asarray(times, dtype=float)
    positions = np.asarray(positions, dtype=float)
    usable = np.isfinite(times) & np.isfinite(positions)
    n_points = int(np.count_nonzero(usable))
    if n_points < 2:
        raise ValueError(
            f"need at least 2 finite front positions to fit a trajectory, got {n_points}")

    t_used, x_used = times[usable], positions[usable]
    t0 = float(t_used.min())
    slope, intercept = np.polyfit(t_used - t0, x_used, 1)
    residuals = x_used - (intercept + slope * (t_used - t0))
    return FrontTrajectory(
        speed=float(slope),
        x0=float(intercept),
        t0=t0,
        residual_rms=float(np.sqrt(np.mean(residuals**2))),
        n_points=n_points,
    )


def ahead_of_front_average(positions: np.ndarray, values: np.ndarray, x_front: float,
                           offset: float, width: float) -> float:
    """Mean of ``values`` over ``[x_front + offset, x_front + offset + width]``.

    The ambient state that sets M_A and beta is the one the piston is *about to* drive
    into, so it is sampled a gap ``offset`` ahead of the front — clear of the
    compressed pile-up — over a band ``width`` wide.  Returns nan if the band falls
    outside the line-out.
    """
    positions = np.asarray(positions, dtype=float)
    values = np.asarray(values, dtype=float)
    if not np.isfinite(x_front):
        return float("nan")

    band = ((positions >= x_front + offset)
            & (positions <= x_front + offset + width)
            & np.isfinite(values))
    if not np.any(band):
        return float("nan")
    return float(np.mean(values[band]))


def unperturbed_average(values: np.ndarray, contaminant_fraction: np.ndarray,
                        max_contaminant: float = 1.0e-6) -> float:
    """Mean of ``values`` over cells essentially free of ``contaminant_fraction``.

    Used on the initial-condition dump to read the *pristine background* state: mask out
    every cell containing target material and average what is left.  At t = 0 that is the
    undisturbed chamber fill, and because the laser pulse has not fired yet (it ramps from
    0.1 ns) the laser channel needs no separate exclusion — there is nothing there yet to
    exclude.

    Preferring this to the state measured ahead of the piston front matters: by a few ns
    electron conduction and radiation have preheated the material the front is running
    into by more than an order of magnitude in temperature, so an "upstream" sampled then
    describes the preheated precursor rather than the background the experiment was set up
    with.  Returns nan when every cell is contaminated.
    """
    values = np.asarray(values, dtype=float)
    contaminant_fraction = np.asarray(contaminant_fraction, dtype=float)
    pristine = (contaminant_fraction <= max_contaminant) & np.isfinite(values)
    if not np.any(pristine):
        return float("nan")
    return float(np.mean(values[pristine]))


def behind_front_average(positions: np.ndarray, values: np.ndarray, x_front: float,
                         offset: float, width: float) -> float:
    """Mean of ``values`` over ``[x_front - offset - width, x_front - offset]``.

    The mirror of :func:`ahead_of_front_average`, looking *inward*.  This is the
    quantity that sets the piston's drive: the density of piston material immediately
    behind the front, which for a laser-driven plume is far below the global peak — that
    peak sits in the dense stagnated material next to the target and has no bearing on
    the mass flux arriving at the shock.  Using the peak instead overstates the
    piston/ambient contrast by more than an order of magnitude on these line-outs.
    """
    positions = np.asarray(positions, dtype=float)
    values = np.asarray(values, dtype=float)
    if not np.isfinite(x_front):
        return float("nan")

    band = ((positions >= x_front - offset - width)
            & (positions <= x_front - offset)
            & np.isfinite(values))
    if not np.any(band):
        return float("nan")
    return float(np.mean(values[band]))


def edge_is_resolved(positions: np.ndarray, density: np.ndarray, x_front: float,
                     scale_length: float, min_samples: float = 2.0) -> bool:
    """Does the fitted ``scale_length`` span enough samples to be a real width?

    Ideal-MHD FLASH gives a piston edge whose thickness is set by the grid, not by
    physics (``docs/piston_interface_smoothing_plan.md``).  When that is the case any
    "e-folding length" fitted across it is meaningless, and a fit taken over a wider
    window silently measures the plateau behind the edge instead.  This flags the
    situation rather than letting the number through unqualified.
    """
    positions = np.asarray(positions, dtype=float)
    if positions.size < 2 or not np.isfinite(scale_length) or scale_length <= 0.0:
        return False
    spacing = abs(float(positions[1] - positions[0]))
    return bool(spacing > 0.0 and scale_length >= min_samples * spacing)


def upstream_is_pristine(positions: np.ndarray, piston_density: np.ndarray,
                         total_density: np.ndarray, x_front: float, offset: float,
                         width: float, contamination_max: float = 0.1) -> bool:
    """Is the band sampled ahead of the front actually undisturbed ambient?

    True when piston material contributes less than ``contamination_max`` of the total
    density there.  Worth checking every dump rather than assuming: on these line-outs
    the diamagnetic cavity eventually swallows the *whole* window, and averaging over
    dumps past that point silently returns the state inside the cavity — field expelled,
    density high — which shows up as an absurd beta (163 instead of ~2) and a
    sub-critical Mach number for a shock that is plainly super-critical.
    """
    piston = ahead_of_front_average(positions, piston_density, x_front, offset, width)
    total = ahead_of_front_average(positions, total_density, x_front, offset, width)
    if not (np.isfinite(piston) and np.isfinite(total)) or total <= 0.0:
        return False
    return bool(piston / total < contamination_max)


def collapse_profile(positions: np.ndarray, density: np.ndarray, x_front: float,
                     scale_length: float) -> tuple[np.ndarray, np.ndarray]:
    """Rescale one profile to ``(xi, n/n_peak)`` with ``xi = (x - x_front)/L``.

    If successive dumps collapse onto one curve the expansion is self-similar, and the
    piston is then fully described by ``(x_front(t), L(t), n_peak(t))`` — three numbers
    the heater deck can be tuned against, instead of a whole profile per dump.
    """
    positions = np.asarray(positions, dtype=float)
    density = np.asarray(density, dtype=float)
    peak = _finite_max(density)
    if not (np.isfinite(x_front) and np.isfinite(scale_length) and scale_length > 0.0
            and peak > 0.0):
        return (np.full(positions.shape, np.nan), np.full(density.shape, np.nan))
    return ((positions - x_front) / scale_length, density / peak)
