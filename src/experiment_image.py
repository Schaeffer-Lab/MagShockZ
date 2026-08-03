"""Read an experimental streak image and register it onto the FLASH LOS frame.

The MagShockZ streaked-shadowgraphy data arrives as a *streak image*: one spatial
axis (mm) against time (ns), the same layout as the FLASH nₑ streak that
``scripts/flash_overview.py`` / ``scripts/tune_flash_shock.py`` assemble from
line-outs.  Two jobs live here, both pure array/number work:

1. **Crop** — the shot-3 PNG is a *decorated* matplotlib figure (axes, ticks and
   labels burned in), so the data itself is an inset rectangle.  :func:`detect_plot_box`
   finds that rectangle by looking for the rows/columns that are mostly dark against
   the white figure background; the caller supplies what the box spans in ns and mm
   (read off its tick labels once and stored in the config).  :func:`crop_window`
   then cuts a sub-rectangle in those same ns/mm units — e.g. down to the few ns the
   simulation covers.

2. **Register** — the experiment's origin in time and space has no known FLASH
   counterpart (the streak camera's trigger and the mm scale's zero are their own),
   so :class:`Registration` carries the two offsets plus a spatial flip and places
   FLASH onto the image's axes.  It is a hand-tuned quantity: slide it until the
   features line up, then store it in the config.

**The image's axes are the truth.**  The experimental streak is never stretched,
resampled or mapped into simulation units; everything is drawn in its own ns and mm,
and a shorter simulation is handled by *cropping* the image (or just narrowing the
view) and *translating* FLASH onto it.  ``t_ns`` / ``x_mm`` say what the image spans
and may be translated (to re-zero the mm scale, say), but their **span** must stay the
image's true one — shrinking a span to "zoom in" would rescale the picture, and is
what :func:`crop_window` exists to avoid.

Only numpy is imported at module scope (``matplotlib.pyplot.imread`` is imported
lazily inside :func:`load_streak`), so this module stays in the CI-tested
dependency-light layer described in CLAUDE.md.

Convention: a loaded :class:`StreakImage` holds ``img[n_x, n_t]`` with **row 0 at the
smallest mm**, so it is drawn with ``imshow(..., origin="lower")``.
"""

import os
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

MM_PER_UM = 1.0e-3
UM_PER_MM = 1.0e3


# ---------------------------------------------------------------------------
# The image
# ---------------------------------------------------------------------------

@dataclass
class StreakImage:
    """A cropped streak image plus what its edges mean in experiment units.

    Attributes
    ----------
    img : ndarray [n_x, n_t]
        Greyscale intensity, row 0 at ``x_mm[0]`` (the smallest mm), column 0 at
        ``t_ns[0]``.
    t_ns, x_mm : (float, float)
        The *outer edges* of the cropped box — what ``imshow``'s ``extent`` wants.
    path : str
        Where it came from (for figure annotation).
    """

    img: np.ndarray
    t_ns: Tuple[float, float]
    x_mm: Tuple[float, float]
    path: str = ""

    @property
    def shape(self) -> Tuple[int, int]:
        return self.img.shape

    @property
    def extent(self) -> Tuple[float, float, float, float]:
        """``(t_lo, t_hi, x_lo, x_hi)`` in experiment units, for ``imshow``."""
        return (self.t_ns[0], self.t_ns[1], self.x_mm[0], self.x_mm[1])

    def t_axis(self) -> np.ndarray:
        """Pixel-centre times [ns], length ``n_t``."""
        return _centres(self.t_ns, self.img.shape[1])

    def x_axis(self) -> np.ndarray:
        """Pixel-centre positions [mm], length ``n_x``."""
        return _centres(self.x_mm, self.img.shape[0])

    def column(self, t_ns: float) -> Tuple[np.ndarray, float]:
        """The spatial line-out nearest ``t_ns``; returns ``(values, actual t)``.

        The experimental analogue of a single FLASH dump's line-out — used for the
        side-by-side profile comparison.
        """
        t = self.t_axis()
        j = int(np.argmin(np.abs(t - float(t_ns))))
        return self.img[:, j], float(t[j])


def _centres(edges: Sequence[float], n: int) -> np.ndarray:
    """Pixel centres for ``n`` pixels spanning the outer edges ``(lo, hi)``."""
    lo, hi = float(edges[0]), float(edges[1])
    step = (hi - lo) / n
    return lo + step * (np.arange(n) + 0.5)


# ---------------------------------------------------------------------------
# Cropping the decorated figure
# ---------------------------------------------------------------------------

def to_gray(img: np.ndarray) -> np.ndarray:
    """Any imread result → float greyscale in [0, 1].

    Accepts 2-D greyscale, RGB or RGBA, uint8 or float; the alpha channel (if any)
    is dropped and the colour channels are averaged.
    """
    a = np.asarray(img)
    if a.dtype == np.uint8:
        a = a.astype(float) / 255.0
    else:
        a = a.astype(float)
    if a.ndim == 3:
        a = a[..., :3].mean(axis=-1)
    if a.ndim != 2:
        raise ValueError(f"expected a 2-D image after greyscale conversion, got {a.shape}")
    return a


def detect_plot_box(gray: np.ndarray, *, threshold: float = 0.6,
                    frac: float = 0.5) -> Tuple[int, int, int, int]:
    """Find the data rectangle inside a decorated (axes-burned-in) figure.

    The streak data is dark and fills the axes; the surrounding figure is white.  A
    column belongs to the box when more than ``frac`` of its pixels are darker than
    ``threshold``, and likewise for rows — which ignores the sparse dark pixels of
    tick labels and axis text outside the axes.

    Returns
    -------
    (left, right, top, bottom) : int
        **Slice-ready, half-open** in image (row, column) order, i.e. the data is
        ``gray[top:bottom, left:right]``.  ``top`` is the *upper* row on screen,
        which for an un-flipped image is the LARGEST spatial coordinate.
    """
    g = np.asarray(gray, dtype=float)
    dark = g < float(threshold)
    cols = np.flatnonzero(dark.sum(axis=0) > frac * dark.shape[0])
    rows = np.flatnonzero(dark.sum(axis=1) > frac * dark.shape[1])
    if cols.size == 0 or rows.size == 0:
        raise ValueError(
            "could not find a data rectangle in the image (no rows/columns are "
            f"mostly darker than {threshold}); pass crop_px explicitly")
    return int(cols[0]), int(cols[-1]) + 1, int(rows[0]), int(rows[-1]) + 1


def crop(gray: np.ndarray, box: Optional[Sequence[int]] = None) -> np.ndarray:
    """Crop to ``box`` (``left, right, top, bottom``), auto-detecting when None."""
    left, right, top, bottom = detect_plot_box(gray) if box is None else (
        int(box[0]), int(box[1]), int(box[2]), int(box[3]))
    out = gray[top:bottom, left:right]
    if out.size == 0:
        raise ValueError(f"crop box {(left, right, top, bottom)} is empty for a "
                         f"{gray.shape} image")
    return out


def load_calib(path: str) -> dict:
    """Read a two-column ``type,value`` calibration CSV (``px_to_mm`` / ``px_to_ns``)."""
    out = {}
    with open(path) as fh:
        for line in fh:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 2 or parts[0] in ("type", ""):
                continue
            try:
                out[parts[0]] = float(parts[1])
            except ValueError:
                continue
    missing = {"px_to_mm", "px_to_ns"} - set(out)
    if missing:
        raise KeyError(f"{path} is missing {sorted(missing)}")
    return out


def load_streak_csv(path: str, calib: dict, *, origin: str = "center",
                    t0_ns: float = 0.0, row0_is_top: bool = True,
                    cache: bool = True) -> StreakImage:
    """Load the RAW streak (a CSV of camera counts) and calibrate it from mm/px, ns/px.

    This is the preferred source: the pixel grid *is* the measurement, so the axes
    follow from the calibration alone — nothing is read off a rendered figure.  Rows
    are the spatial axis, columns time; ``row0_is_top`` says the file starts at the
    largest mm (as the camera writes it), so the array is flipped to ascend in mm.

    ``origin`` places mm = 0 on the slit: ``"center"`` (the image centre, matching how
    the streak is usually plotted), ``"bottom"`` (the first row), or a number giving
    the mm coordinate of the bottom edge.  ``t0_ns`` is the camera time of the first
    column.

    Parsing a 2048² CSV takes a few seconds, so the array is cached alongside it as
    ``<name>.npy`` (delete that file to force a re-parse).
    """
    npy = os.path.splitext(path)[0] + ".npy"
    if cache and os.path.exists(npy) and os.path.getmtime(npy) >= os.path.getmtime(path):
        a = np.load(npy)
    else:
        a = np.loadtxt(path, delimiter=",")
        if cache:
            try:
                np.save(npy, a)
            except OSError:
                pass
    if a.ndim != 2:
        raise ValueError(f"{path} is not a 2-D streak (got shape {a.shape})")
    if row0_is_top:
        a = a[::-1]                       # ascend in mm, for origin="lower"

    n_x, n_t = a.shape
    height = n_x * float(calib["px_to_mm"])
    span = n_t * float(calib["px_to_ns"])
    if origin == "center":
        x_lo = -0.5 * height
    elif origin == "bottom":
        x_lo = 0.0
    else:
        x_lo = float(origin)
    return StreakImage(img=a, t_ns=(float(t0_ns), float(t0_ns) + span),
                       x_mm=(x_lo, x_lo + height), path=path)


def load_streak(path: str, *, t_ns: Sequence[float], x_mm: Sequence[float],
                crop_px: Optional[Sequence[int]] = None,
                invert: bool = False) -> StreakImage:
    """Read a streak PNG, crop it to its data box and label its axes.

    ``t_ns`` / ``x_mm`` are what the *whole data box* spans — read off the burned-in
    tick labels once and kept in the analysis config, since the pixel grid of a
    decorated figure carries no calibration of its own.  They are the image's axes,
    not a zoom: to look at part of the image use :func:`crop_window`.

    ``invert`` flips the intensity (use when the feature of interest is dark on a
    bright background) so the colour scale reads the same way as nₑ.
    """
    from matplotlib.pyplot import imread   # lazy: keeps the module CI-importable

    gray = to_gray(imread(path))
    box = crop(gray, crop_px)
    # imread gives row 0 at the TOP of the figure (largest mm); flip so the array
    # ascends in mm and every consumer can use origin="lower".
    box = box[::-1, :]
    if invert:
        box = 1.0 - box
    return StreakImage(img=box,
                       t_ns=(float(t_ns[0]), float(t_ns[1])),
                       x_mm=(float(x_mm[0]), float(x_mm[1])),
                       path=path)


# ---------------------------------------------------------------------------
# Cropping to a view window — in the image's OWN units
# ---------------------------------------------------------------------------

def crop_window(streak: StreakImage, t_ns=None, x_mm=None) -> StreakImage:
    """Cut a sub-rectangle out of ``streak``, keeping its calibration exact.

    This is a **pixel crop**, not a resampling: the returned image's ``t_ns`` /
    ``x_mm`` are the true edges of the pixels that were kept, so a feature sits at
    the same ns and mm before and after.  Passing a window wider than the image
    simply keeps everything (the image is never padded or stretched).

    ``t_ns`` / ``x_mm`` are ``(lo, hi)`` windows in the image's own units; ``None``
    leaves that axis alone.
    """
    lo_t, hi_t = _pixel_span(streak.t_ns, streak.img.shape[1], t_ns)
    lo_x, hi_x = _pixel_span(streak.x_mm, streak.img.shape[0], x_mm)
    img = streak.img[lo_x:hi_x, lo_t:hi_t]
    if img.size == 0:
        raise ValueError(f"crop window t={t_ns}, x={x_mm} keeps no pixels of "
                         f"{streak.path or 'the image'} (which spans "
                         f"{streak.t_ns} ns, {streak.x_mm} mm)")
    return StreakImage(img=img,
                       t_ns=_edges_of(streak.t_ns, streak.img.shape[1], lo_t, hi_t),
                       x_mm=_edges_of(streak.x_mm, streak.img.shape[0], lo_x, hi_x),
                       path=streak.path)


def _pixel_span(edges, n, window):
    """Half-open pixel range covering ``window`` within an axis of ``n`` pixels."""
    if window is None:
        return 0, n
    lo_v, hi_v = sorted((float(window[0]), float(window[1])))
    lo_e, hi_e = float(edges[0]), float(edges[1])
    step = (hi_e - lo_e) / n
    lo = int(np.clip(np.floor((lo_v - lo_e) / step), 0, n - 1))
    hi = int(np.clip(np.ceil((hi_v - lo_e) / step), lo + 1, n))
    return lo, hi


def _edges_of(edges, n, lo, hi):
    """The (lo, hi) coordinate edges of pixels ``lo:hi`` of an axis."""
    lo_e, hi_e = float(edges[0]), float(edges[1])
    step = (hi_e - lo_e) / n
    return (lo_e + step * lo, lo_e + step * hi)


# ---------------------------------------------------------------------------
# Putting FLASH onto the image's axes
# ---------------------------------------------------------------------------

@dataclass
class Registration:
    """Places the FLASH data onto the experiment's own (ns, mm) axes.

    The experimental image is the reference frame: its burned-in ns and mm axes are
    the truth, and it is never moved, stretched or resampled.  What moves is FLASH —
    a rigid translation (plus, optionally, a direction flip)::

        t_exp = t_flash + t_offset_ns
        mm    = ±(los_µm / 1000) + x_offset_mm

    ``t_offset_ns`` is the experiment time at which FLASH's t = 0 occurs and
    ``x_offset_mm`` the experiment mm coordinate of LOS distance 0 (the LOS start
    point); ``flip_space`` is for when the experiment's +mm runs opposite to the line
    of sight.  None of this is derivable from the data files — it is hand-tuned
    against the images and then stored in the config.
    """

    t_offset_ns: float = 0.0
    x_offset_mm: float = 0.0
    flip_space: bool = False

    @property
    def sign(self) -> float:
        return -1.0 if self.flip_space else 1.0

    # -- FLASH -> experiment (the direction everything is drawn in) ----------
    def to_exp_t(self, t_flash_ns):
        return np.asarray(t_flash_ns, dtype=float) + self.t_offset_ns

    def to_exp_mm(self, los_um):
        return self.sign * np.asarray(los_um, dtype=float) * MM_PER_UM + self.x_offset_mm

    # -- experiment -> FLASH (for reporting / picking dumps) -----------------
    def to_flash_t(self, t_ns):
        return np.asarray(t_ns, dtype=float) - self.t_offset_ns

    def to_los_um(self, x_mm):
        return self.sign * (np.asarray(x_mm, dtype=float) - self.x_offset_mm) * UM_PER_MM

    def flash_extent(self, t_flash_ns, x_flash_um) -> Tuple[float, float, float, float]:
        """Where the FLASH data lands on the image's axes: ``(t0, t1, mm0, mm1)``."""
        t = self.to_exp_t(np.asarray(t_flash_ns, dtype=float))
        x = self.to_exp_mm(np.asarray(x_flash_um, dtype=float))
        return (float(t.min()), float(t.max()), float(x.min()), float(x.max()))


def flash_on_exp_axis(x_flash_um, values, reg: Registration):
    """Translate a FLASH LOS axis onto the experiment's mm axis.

    ``values`` is ``[..., n_x]`` (e.g. an ``[n_dumps, n_x]`` streak or a single
    line-out).  A flipped registration makes the mm coordinates descend, so both the
    axis and the data's last dimension are reversed together — the values keep their
    positions, only the bookkeeping order changes.
    """
    mm = reg.to_exp_mm(x_flash_um)
    v = np.asarray(values)
    if reg.flip_space:
        return mm[::-1], v[..., ::-1]
    return mm, v


def from_config(block: Optional[dict]) -> Registration:
    """Build a :class:`Registration` from a config ``experiment.registration`` block."""
    block = block or {}
    return Registration(t_offset_ns=float(block.get("t_offset_ns", 0.0) or 0.0),
                        x_offset_mm=float(block.get("x_offset_mm", 0.0) or 0.0),
                        flip_space=bool(block.get("flip_space", False)))


# ---------------------------------------------------------------------------
# Fitting the shift
# ---------------------------------------------------------------------------

@dataclass
class FitResult:
    """Best rigid placement of the FLASH data on the image, and how well it did."""

    registration: Registration
    r: float                       # Pearson correlation at the optimum
    r_map: np.ndarray              # correlation over every trialled shift [n_t, n_x]
    t_offsets: np.ndarray          # the shifts r_map is sampled at
    x_offsets: np.ndarray
    feature: str = ""
    flip_r: Optional[dict] = None  # best r for each flip that was tried


def fit_shift(streak: StreakImage, t_flash_ns, x_flash_mm, values, *,
              feature: str = "grad", flips=(False, True), decimate: int = 4,
              smooth_px: float = 2.0):
    """Find the (t_offset, x_offset, flip) that best lines FLASH up with the image.

    The transform is a pure translation, so this resamples FLASH onto the image's own
    pixel pitch — *neither* data set is rescaled — and evaluates the normalised
    cross-correlation at every whole-pixel placement at once via FFT.  Only fully
    contained placements are scored, so the correlation is always over the same
    number of pixels and cannot be gamed by sliding mostly out of frame.

    ``feature`` is what is correlated: ``"grad"`` (the spatial gradient magnitude —
    apt for shadowgraphy, which responds to density *gradients*, and insensitive to
    the very different absolute scalings) or ``"signal"`` (the fields themselves,
    percentile-normalised).  ``values`` is the FLASH map ``[n_t, n_x]``; it is
    log-scaled when strictly positive, since nₑ spans decades.

    A high ``r`` means the shapes agree once shifted; it does **not** validate the
    physics.  Inspect ``r_map`` for degeneracy — a ridge rather than a peak means the
    data does not pin that offset down.
    """
    from scipy.ndimage import gaussian_filter
    from scipy.signal import fftconvolve

    if feature not in ("grad", "signal"):
        raise ValueError(f"feature must be 'grad' or 'signal', got {feature!r}")

    t_f = np.asarray(t_flash_ns, dtype=float)
    x_f = np.asarray(x_flash_mm, dtype=float)
    V = np.asarray(values, dtype=float)
    if V.shape != (t_f.size, x_f.size):
        raise ValueError(f"values {V.shape} does not match "
                         f"({t_f.size}, {x_f.size})")
    if np.all(V > 0):
        V = np.log10(V)

    step = max(int(decimate), 1)
    E_img = np.asarray(streak.img, dtype=float)[::step, ::step]         # [x, t]
    dx = (streak.x_mm[1] - streak.x_mm[0]) / streak.img.shape[0] * step
    dt = (streak.t_ns[1] - streak.t_ns[0]) / streak.img.shape[1] * step
    x_e = streak.x_mm[0] + dx * (np.arange(E_img.shape[0]) + 0.5)
    t_e = streak.t_ns[0] + dt * (np.arange(E_img.shape[1]) + 0.5)

    E = _feature(E_img, feature, smooth_px)
    best, per_flip = None, {}
    for flip in flips:
        xf = -x_f if flip else x_f
        order = np.argsort(xf)
        xs, F_in = xf[order], V[:, order]
        n_x = max(int(round((xs[-1] - xs[0]) / dx)) + 1, 2)
        n_t = max(int(round((t_f[-1] - t_f[0]) / dt)) + 1, 2)
        if n_x > E.shape[0] or n_t > E.shape[1]:
            continue                       # FLASH is larger than the image; can't contain
        gx = xs[0] + dx * np.arange(n_x)
        gt = t_f[0] + dt * np.arange(n_t)
        tmp = np.stack([np.interp(gx, xs, row) for row in F_in])        # [n_dumps, n_x]
        G = np.stack([np.interp(gt, t_f, tmp[:, j]) for j in range(n_x)])   # [n_x, n_t]

        r_map = _ncc_valid(E, _feature(G, feature, smooth_px))
        k = np.unravel_index(int(np.argmax(r_map)), r_map.shape)
        # placement k = the patch's corner sitting on image pixel k
        t_off = (t_e[k[1]] - dt / 2) - t_f.min()
        x_off = (x_e[k[0]] - dx / 2) - (xs[0] if not flip else -x_f.max())
        per_flip[flip] = float(r_map[k])
        cand = FitResult(Registration(t_offset_ns=float(t_off), x_offset_mm=float(x_off),
                                      flip_space=bool(flip)),
                         r=float(r_map[k]), r_map=r_map,
                         t_offsets=(t_e[:r_map.shape[1]] - dt / 2) - t_f.min(),
                         x_offsets=(x_e[:r_map.shape[0]] - dx / 2) - xs[0],
                         feature=feature)
        if best is None or cand.r > best.r:
            best = cand
    if best is None:
        raise ValueError("no trial placement fits inside the image — the FLASH data "
                         "covers more ns/mm than the streak does")
    best.flip_r = per_flip
    return best


def _feature(a, feature, smooth_px):
    from scipy.ndimage import gaussian_filter
    a = gaussian_filter(np.asarray(a, dtype=float), smooth_px)
    if feature == "grad":
        a = np.abs(np.gradient(a, axis=0))       # along the spatial axis
    lo, hi = np.percentile(a, 1), np.percentile(a, 99)
    return np.clip((a - lo) / (hi - lo + 1e-30), 0.0, 1.0)


def _ncc_valid(E, F):
    """Pearson r of ``F`` against every fully contained placement inside ``E``."""
    from scipy.signal import fftconvolve
    n = F.size
    sF, sFF = F.sum(), float((F * F).sum())
    ones = np.ones_like(F)
    sEF = fftconvolve(E, F[::-1, ::-1], mode="valid")
    sE = fftconvolve(E, ones, mode="valid")
    sEE = fftconvolve(E * E, ones, mode="valid")
    num = n * sEF - sE * sF
    den = np.sqrt(np.maximum(n * sEE - sE ** 2, 1e-30) * max(n * sFF - sF ** 2, 1e-30))
    return num / den


def overlap_window(streak: StreakImage, reg: Registration,
                   t_flash_ns, x_flash_um) -> Tuple[float, float, float, float]:
    """Where the image and the registered FLASH data both exist, in experiment units.

    Returns ``(t_lo, t_hi, x_lo, x_hi)`` in ns and mm; an interval comes back empty
    (hi <= lo) when the two do not overlap on that axis, which is the signal that the
    registration is off.
    """
    ft0, ft1, fx0, fx1 = reg.flash_extent(t_flash_ns, x_flash_um)
    return (max(streak.t_ns[0], ft0), min(streak.t_ns[1], ft1),
            max(streak.x_mm[0], fx0), min(streak.x_mm[1], fx1))
