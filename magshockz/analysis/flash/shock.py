"""shock.py — shared shock-front detection and trajectory fitting.

One module owns the shock-trajectory logic used by both the OSIRIS overview
(``scripts/osiris_overview.py``) and the FLASH overview (``scripts/flash_overview.py`` via
``flash_utils``).  Two front-detection strategies are provided because the two data
models call for different markers, plus a robust linear trajectory fit.

Dependency-light (numpy only) so it is unit-testable without the OSIRIS/yt stacks.
"""

from dataclasses import dataclass

import numpy as np


def detect_front_outermost(x, rho, threshold: float = 1.5,
                           n_far: int = 150) -> float:
    """Outermost position where the density still exceeds ``threshold`` x ambient.

    The gradient detectors above find the steepest jump, which stops describing the
    blast once the piston cavity opens up behind it: the steepest drop in the profile
    then sits at the cavity wall near the origin rather than at the front, and the
    tracked position collapses inward.  Marching in from the far field instead is
    monotone in the thing we actually want -- how far the disturbance has reached --
    and so stays locked on the front for the whole run.

    Parameters
    ----------
    x :
        Position along the line of sight, increasing away from the target.
    rho :
        Mass density on the same grid, in any unit; only its ratio is used.
    threshold :
        Compression, relative to the far-field ambient, that counts as disturbed.
    n_far :
        How many trailing samples define the far-field ambient (their median).

    Returns
    -------
    float
        Position of the front in ``x``'s unit, or nan when nothing on the ray
        exceeds the threshold.
    """
    x = np.asarray(x, dtype=float)
    rho = np.asarray(rho, dtype=float)
    ambient = np.nanmedian(rho[-n_far:])
    if not np.isfinite(ambient) or ambient <= 0.0:
        return float("nan")
    disturbed = np.flatnonzero(rho > threshold * ambient)
    return float(x[disturbed.max()]) if disturbed.size else float("nan")


def contact_position(x, mass_fraction, threshold: float = 1.0e-3) -> float:
    """Outermost position at which piston material is still present.

    This is the inner edge the downstream band has to respect.  The band is meant to
    hold shocked *ambient*, and past the contact the ray is in piston material -- a
    different element with a different charge state and equation of state -- so a
    band that reaches across it averages two distinct plasmas into one "downstream"
    state and reports a compression that belongs to neither.

    Parameters
    ----------
    x :
        Position along the line of sight, increasing away from the target.
    mass_fraction :
        The piston material's mass fraction on the same grid (FLASH's ``targ``).
    threshold :
        Mass fraction above which the material counts as present.  Well below any
        real contamination, since FLASH's advected fractions have small numerical
        tails on the ambient side of the contact.

    Returns
    -------
    float
        Position of the contact in ``x``'s unit, or nan when no piston material
        reaches this ray at all.
    """
    x = np.asarray(x, dtype=float)
    fraction = np.asarray(mass_fraction, dtype=float)
    present = np.flatnonzero(np.isfinite(fraction) & (fraction > threshold))
    return float(x[present.max()]) if present.size else float("nan")


@dataclass(frozen=True)
class ShockBands:
    """The two windows a shock measurement is averaged over, and where they came from.

    Every position is in the same unit as the ``x`` handed to :func:`resolve_bands`
    (cm throughout the FLASH scripts).  This exists so that the scripts measuring the
    *same* shock cannot disagree about *where* it is: resolving the bands in one place
    is what keeps ``flash_rh_prediction`` and ``flash_pressure_partition`` reporting
    the same compression, which they did not when each did it inline.

    ``note`` is empty when both bands resolved as asked, and otherwise says which
    fallback was taken -- a band that quietly relocated is the kind of thing that
    turns into a wrong number in a table nobody re-derives.

    There are TWO downstream bands because two different questions are being asked
    of the same shock, and they do not want the same region:

    * the **jump band** (``x_jump`` to the front) is thin, because the
      Rankine--Hugoniot conditions are a *local* statement at the discontinuity.
      Tested over a band much wider than the front they fail by construction --
      measured here as momentum-flux continuity degrading from 1.00 at 50 µm to
      0.51 at 940 µm -- since the inner edge holds material shocked ns earlier,
      when the shock was faster and the upstream denser.
    * the **layer band** (``x_downstream`` to the front) is the whole shocked
      layer, because that is the plasma an experiment would diagnose, and so it
      is the right region for downstream heating, Zbar and the e/i partition.
    """

    x_shock: float
    x_downstream: float
    x_jump: float
    x_upstream_lo: float
    x_upstream_hi: float
    x_contact: float
    downstream_edge: str
    note: str = ""

    def upstream_mask(self, x) -> np.ndarray:
        """Boolean mask selecting the upstream window out of ``x``."""
        x = np.asarray(x, dtype=float)
        return (x >= self.x_upstream_lo) & (x <= self.x_upstream_hi)

    def downstream_mask(self, x) -> np.ndarray:
        """Boolean mask selecting the full shocked layer out of ``x``."""
        x = np.asarray(x, dtype=float)
        return (x >= self.x_downstream) & (x <= self.x_shock)

    def jump_mask(self, x) -> np.ndarray:
        """Boolean mask selecting the thin band just behind the front."""
        x = np.asarray(x, dtype=float)
        return (x >= self.x_jump) & (x <= self.x_shock)


def resolve_bands(x, piston_fraction, x_shock: float, *,
                  upstream_gap: float, upstream_width: float,
                  contact_gap: float, jump_width: float = 0.0,
                  x_downstream_config: float = None,
                  edge: str = "contact") -> ShockBands:
    """Place the upstream and downstream averaging windows around ``x_shock``.

    Upstream is a window starting ``upstream_gap`` ahead of the front and
    ``upstream_width`` wide, not everything beyond it: the state that sets the Mach
    numbers is the gas the shock is about to hit, and averaging to the end of the ray
    mixes in far field the blast has not reached.  ``upstream_width <= 0`` restores
    the to-the-end-of-the-ray behaviour.

    Downstream runs from the front back to ``contact_gap`` short of the outermost
    piston material (``edge="contact"``), so the band is as wide as the shocked
    ambient actually is rather than a fixed offset; ``edge="config"`` uses
    ``x_downstream_config`` verbatim.  The contact edge falls back to the configured
    value -- recording why in ``note`` -- when no piston material reaches the ray, or
    when the contact has caught up with the front.

    All lengths share ``x``'s unit.

    Raises
    ------
    ValueError
        If no downstream edge can be established, or the upstream window starts
        past the end of the ray.
    """
    x = np.asarray(x, dtype=float)
    contact = (float("nan") if piston_fraction is None
               else contact_position(x, piston_fraction))

    note = ""
    x_downstream = x_downstream_config
    if edge == "contact":
        if not np.isfinite(contact):
            note = "no piston material on this ray; kept the configured downstream edge"
        elif contact + contact_gap >= x_shock:
            note = (f"piston contact ({contact:.6g}) has reached the front "
                    f"({x_shock:.6g}); kept the configured downstream edge")
        else:
            x_downstream = contact + contact_gap

    if x_downstream is None or not np.isfinite(x_downstream):
        raise ValueError(
            "no downstream edge: the contact could not be located and no "
            "x_downstream_start was configured or passed.")
    if x_downstream >= x_shock:
        raise ValueError(
            f"downstream edge {x_downstream:.6g} is at or past the front "
            f"{x_shock:.6g}; the band would be empty.")

    x_upstream_lo = x_shock + upstream_gap
    if x_upstream_lo > x.max():
        raise ValueError(
            f"the upstream window starts at {x_upstream_lo:.6g}, past the end of the "
            f"ray ({x.max():.6g}). Shorten the gap or extend the line of sight.")
    x_upstream_hi = (x.max() if upstream_width <= 0.0
                     else min(x_upstream_lo + upstream_width, x.max()))

    # The jump band never reaches past the layer band's inner edge: outside the
    # shocked layer there is nothing for a jump condition to be measured against.
    x_jump = (max(x_shock - jump_width, x_downstream) if jump_width > 0.0
              else x_downstream)

    return ShockBands(x_shock=float(x_shock), x_downstream=float(x_downstream),
                      x_jump=float(x_jump),
                      x_upstream_lo=float(x_upstream_lo),
                      x_upstream_hi=float(x_upstream_hi),
                      x_contact=float(contact), downstream_edge=edge, note=note)


@dataclass(frozen=True)
class FrontFit:
    """A shock trajectory fitted locally around one time, linear and quadratic.

    Times are in ns, positions in µm, so a slope is µm/ns -- which is km/s exactly,
    with no conversion factor.  ``acceleration`` is therefore km/s per ns.

    ``v_linear`` and ``v_quadratic`` agree to machine precision whenever the window
    is symmetric about ``t`` and evenly sampled: the curvature term is even about
    the centre and the linear basis is odd, so the two are orthogonal over such a
    window and the quadratic term cannot bias the fitted slope.  They separate only
    at the ends of a run, where the available window goes lopsided.  The quadratic
    still earns its place through the other two fields: ``acceleration`` is the
    deceleration of the blast, and ``rms_quadratic`` is an honest noise estimate
    because it no longer counts real curvature as scatter -- so a track whose
    ``rms_quadratic`` stays large is mis-tracking, not merely curved.
    """

    t: float
    v_linear: float
    v_quadratic: float
    acceleration: float
    rms_linear: float
    rms_quadratic: float
    n_points: int


def local_front_fit(t, x, target: float, half_width: float = 1.0) -> FrontFit:
    """Fit the front track ``x(t)`` over ``target`` +/- ``half_width``.

    Parameters
    ----------
    t, x :
        The front track: times [ns] and positions [µm], any order, nans ignored.
    target :
        Time [ns] the fit is centred on and the speed is quoted at.
    half_width :
        Half-width [ns] of the fitting window.  Keep it narrow enough that the
        trajectory is locally smooth; 1 ns is 9 dumps at this run's 0.25 ns cadence.

    Returns
    -------
    FrontFit
        Quadratic fields are nan when the window holds fewer than 3 points; every
        field is nan when it holds fewer than 2.
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    window = np.isfinite(t) & np.isfinite(x) & (np.abs(t - target) <= half_width + 1e-9)
    tw, xw = t[window], x[window]

    nan = float("nan")
    if tw.size < 2:
        return FrontFit(target, nan, nan, nan, nan, nan, int(tw.size))

    def slope_and_rms(deg):
        coeffs = np.polyfit(tw, xw, deg)
        residual = xw - np.polyval(coeffs, tw)
        return (float(np.polyval(np.polyder(coeffs), target)),
                float(np.sqrt(np.mean(residual**2))),
                coeffs)

    v_linear, rms_linear, _ = slope_and_rms(1)
    if tw.size < 3:
        return FrontFit(target, v_linear, nan, nan, rms_linear, nan, int(tw.size))

    v_quadratic, rms_quadratic, quad = slope_and_rms(2)
    return FrontFit(target, v_linear, v_quadratic, float(2.0 * quad[0]),
                    rms_linear, rms_quadratic, int(tw.size))


def detect_front_edge(x, profile, x_pred, half_window,
                      compression_min: float = 1.3, edge_frac: float = 0.5) -> float:
    """Leading (upstream) edge of compression near ``x_pred``.

    Returns the largest ``x`` within ``[x_pred - hw, x_pred + hw]`` at which
    ``profile`` exceeds ``baseline + edge_frac*(peak - baseline)``.  The shock moves
    toward +x with compressed plasma on the low-x side, so the leading edge is the
    upstream-most crossing.  Returns nan if the window holds no clear compression
    (peak/baseline < ``compression_min``).  Used by the OSIRIS overview, where the
    density/|B| streaks make the leading edge the cleanest marker.
    """
    x = np.asarray(x)
    profile = np.asarray(profile)
    win = (x >= x_pred - half_window) & (x <= x_pred + half_window)
    if not win.any():
        return float("nan")
    xa, pa = x[win], profile[win]
    baseline = np.percentile(pa, 20)
    peak = np.percentile(pa, 99)
    if baseline <= 0 or peak / baseline < compression_min:
        return float("nan")
    thresh = baseline + edge_frac * (peak - baseline)
    above = xa[pa >= thresh]
    return float(above.max()) if above.size else float("nan")


def detect_front_gradient(x, ne, x_pred, half_window,
                          compression_min: float = 1.3, smooth: int = 3) -> float:
    """Steepest density drop (the jump) near ``x_pred``.

    The shock moves toward +x: shocked/driver material is dense at smaller x and the
    ambient upstream is tenuous at larger x, so the front is the steepest *drop* in
    nₑ with increasing x.  Located as the most negative density gradient inside the
    search window after light boxcar smoothing, with a minimum compression so
    flat/ambient windows return nan.  Targeting the steepest gradient places the
    marker on the actual jump (not its leading edge).  Used by the FLASH overview.

    Parameters
    ----------
    x, ne       : spatial coordinate and density (same length).
    x_pred      : predicted shock position.
    half_window : search half-width.
    compression_min : minimum (95th pct / 20th pct) ratio to accept a front.
    smooth      : symmetric boxcar width [cells] to suppress single-cell noise.
    """
    x = np.asarray(x)
    ne = np.asarray(ne)
    win = (x >= x_pred - half_window) & (x <= x_pred + half_window)
    if win.sum() < 5:
        return float("nan")
    xa, pa = x[win], ne[win]
    order = np.argsort(xa)
    xa, pa = xa[order], pa[order]

    baseline = np.percentile(pa, 20)
    peak = np.percentile(pa, 95)
    if baseline <= 0 or peak / baseline < compression_min:
        return float("nan")

    # light boxcar smoothing (symmetric, so it does not shift the front)
    if smooth > 1 and pa.size >= smooth:
        pa = np.convolve(pa, np.ones(smooth) / smooth, mode="same")

    grad = np.gradient(pa, xa)
    return float(xa[np.argmin(grad)])      # steepest drop = front


def robust_linfit(t, x, n_iter: int = 3, n_sigma: float = 2.5):
    """Linear fit ``x = slope*t + intercept`` with iterative σ-clipping.

    A few bad per-frame detections would otherwise drag the trajectory fit (and the
    predicted window) off the real front, so points more than ``n_sigma`` residual-σ
    from the line are dropped and the line refit.  At least 3 points are always kept;
    if clipping would drop below that, the previous fit is retained.  Returns
    ``(slope, intercept)``.
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    slope, intercept = np.polyfit(t, x, 1)
    keep = np.ones(len(t), dtype=bool)
    for _ in range(n_iter):
        resid = x - (slope * t + intercept)
        sigma = np.std(resid[keep])
        if sigma == 0:
            break
        new_keep = np.abs(resid) <= n_sigma * sigma
        if new_keep.sum() < 3 or np.array_equal(new_keep, keep):
            break
        keep = new_keep
        slope, intercept = np.polyfit(t[keep], x[keep], 1)
    return float(slope), float(intercept)


def robust_polyfit(t, x, deg: int = 2, n_iter: int = 3, n_sigma: float = 2.5):
    """Polynomial trajectory fit ``x(t)`` of degree ``deg`` with σ-clipping.

    Generalizes :func:`robust_linfit` to arbitrary degree so the shock front can
    be fit as a (decelerating) curve and differentiated for an *instantaneous*
    shock velocity ``v_shock = dx/dt`` (see :func:`trajectory_at`).  Same
    iterative outlier rejection as :func:`robust_linfit`; at least ``deg + 2``
    points are kept (if clipping would drop below that, the previous fit is
    retained).  Returns numpy polynomial coefficients (highest power first, the
    ``np.polyfit`` / ``np.poly1d`` convention).
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    min_pts = deg + 2
    coeffs = np.polyfit(t, x, deg)
    keep = np.ones(len(t), dtype=bool)
    for _ in range(n_iter):
        resid = x - np.polyval(coeffs, t)
        sigma = np.std(resid[keep])
        if sigma == 0:
            break
        new_keep = np.abs(resid) <= n_sigma * sigma
        if new_keep.sum() < min_pts or np.array_equal(new_keep, keep):
            break
        keep = new_keep
        coeffs = np.polyfit(t[keep], x[keep], deg)
    return coeffs


def front_line(x0, v, t, t0=0.0):
    """Straight-line shock front ``x(t) = x0 + v·(t − t0)``.

    The hand-placed FLASH trajectory (config ``flash:``) in the form every consumer
    draws it.  ``t0`` is the *anchor*: the time at which the front sat at ``x0``.  It
    is a free parameter — a shock that forms partway through the run is fitted by
    sliding the anchor to its formation time rather than by back-extrapolating a
    position it never had.  Defaults to 0 (anchor at t=0).

    Units are the caller's, and must be consistent (the FLASH scripts use cm, cm/s, s).
    """
    return x0 + v * (np.asarray(t, dtype=float) - t0)


def trajectory_at(coeffs, t):
    """Evaluate the fitted front (position, velocity) at time(s) ``t``.

    ``velocity`` is the analytic time-derivative of the polynomial trajectory —
    the instantaneous shock-front speed used to boost into the shock frame.
    ``coeffs`` is the output of :func:`robust_polyfit` (or any ``np.polyfit``).
    """
    coeffs = np.asarray(coeffs, dtype=float)
    pos = np.polyval(coeffs, t)
    vel = np.polyval(np.polyder(coeffs), t)
    return pos, vel


def overview_row(idx: int, dump_indices, n_rows: int) -> int:
    """Row of a ``flash_overview`` archive that holds plot-file index ``idx``.

    Row position and plot-file index coincide only when the overview covered every
    dump from 0; under ``--stride`` / ``--t-start`` they diverge, and reading one for
    the other silently returns a different dump's shock position and time.  Archives
    written since that was noticed record their ``dump_indices``, which is what makes
    the lookup exact.

    Parameters
    ----------
    idx :
        Plot-file index wanted (already resolved to a positive value).
    dump_indices :
        The archive's ``dump_indices`` array, or ``None`` for an older archive that
        predates the key.
    n_rows :
        Number of rows in the archive.

    Returns
    -------
    int
        Row index.

    Raises
    ------
    ValueError
        If the archive does not cover ``idx``, or is too old to prove that it does.
    """
    if dump_indices is not None:
        rows = np.flatnonzero(np.asarray(dump_indices) == idx)
        if rows.size:
            return int(rows[0])
        raise ValueError(
            f"the flash_overview archive covers dumps "
            f"{list(np.asarray(dump_indices))}, not {idx}. Re-run flash_overview.py "
            f"over this dump, or pass the shock position and speed explicitly.")

    if idx >= n_rows:
        raise ValueError(
            f"the flash_overview archive predates the 'dump_indices' key and holds "
            f"only {n_rows} rows, so dump {idx} cannot be located in it (the old "
            f"positional lookup would have silently read row {idx % n_rows}). "
            f"Re-run flash_overview.py.")
    return idx
