"""shock.py — shared shock-front detection and trajectory fitting.

One module owns the shock-trajectory logic used by both the OSIRIS overview
(``scripts/overview.py``) and the FLASH overview (``scripts/flash_overview.py`` via
``flash_utils``).  Two front-detection strategies are provided because the two data
models call for different markers, plus a robust linear trajectory fit.

Dependency-light (numpy only) so it is unit-testable without the OSIRIS/yt stacks.
"""

import numpy as np


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
