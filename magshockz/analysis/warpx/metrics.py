"""Front tracking and the invariant scorecard for a WarpX heater-piston run.

The deck runs at a reduced mass ratio and an arbitrary reference density, so absolute ns
and um are not comparable to FLASH -- only the dimensionless invariants are.  Everything
here is therefore reported in matched units (``d_i`` per ``T_ci``, ``M_A``, density
contrast), with ``v/c`` kept as an explicitly unmatched row so a correct mapping is not
mistaken for a 20x error.

Pure numerics: arrays and a :class:`~magshockz.init.warpx.units.DeckScales` in, numbers
out.  Reading the plotfiles is the caller's job.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from astropy.constants import c

from magshockz.common import piston_profile as pp
from magshockz.init.warpx import units

#: Front travel below which no speed is reported.  One ion inertial length is the
#: smallest displacement over which "the piston is expanding" is a statement about
#: physics rather than about the line-out's discretisation -- a smoke run fitting a
#: slope to less than that reports a number that looks like a speed and is quantisation.
MIN_TRAVEL_DI = 1.0

#: Times are STORED in gyroperiods -- one FLASH ``T_ci`` is one WarpX ``T_ci``, which is
#: what makes the two codes' clocks comparable -- and DISPLAYED in inverse
#: gyrofrequencies.  ``T_ci = 2*pi/omega_ci``, so the shown number is ``2*pi`` times the
#: stored one.  Only the presentation changes: fits, time matching and caches stay in
#: ``T_ci``.
INVERSE_OMEGA_PER_GYROPERIOD = 2.0 * math.pi

#: LaTeX for that unit, so every axis label and title spells it the same way.
TIME_UNIT = r"\omega_{ci}^{-1}"


def wci(t_gyro: np.ndarray | float) -> np.ndarray:
    """A time stored in gyroperiods, expressed in inverse gyrofrequencies."""
    return np.asarray(t_gyro, dtype=float) * INVERSE_OMEGA_PER_GYROPERIOD


def gyroperiods(t_wci: np.ndarray | float) -> np.ndarray:
    """Inverse gyrofrequencies back to the gyroperiods everything is stored in."""
    return np.asarray(t_wci, dtype=float) / INVERSE_OMEGA_PER_GYROPERIOD


def front_speed_over_c(times_omega_pe: np.ndarray, fronts_de: np.ndarray, *,
                       di_over_de: float) -> float:
    """Piston front speed in units of ``c``, or nan if the front has barely moved.

    ``d_e`` per ``1/omega_pe`` is exactly ``c``, so the fitted slope is already ``v/c``.
    """
    finite = np.isfinite(times_omega_pe) & np.isfinite(fronts_de)
    if np.count_nonzero(finite) < 2:
        return float("nan")

    times, fronts = times_omega_pe[finite], fronts_de[finite]
    if abs(fronts[-1] - fronts[0]) / di_over_de < MIN_TRAVEL_DI:
        return float("nan")
    return float(pp.fit_front_trajectory(times, fronts).speed)


@dataclass(frozen=True)
class ShockRegions:
    """Where the shock, the contact and the two averaging bands sit on one line-out.

    Positions are in whatever units the caller's ``positions`` are (the WarpX comparison
    hands over ``d_i``), increasing outward from the target.  ``nan`` anywhere means that
    feature was not found and the caller must not average over it.
    """

    shock: float
    contact: float
    downstream: tuple[float, float]
    upstream: tuple[float, float]

    @property
    def layer_thickness(self) -> float:
        """Shocked-ambient layer: contact discontinuity to shock front."""
        return self.shock - self.contact


def locate_shock_regions(positions: np.ndarray, ambient_density: np.ndarray,
                         piston_density: np.ndarray, *,
                         shock_level: float = 1.5,
                         band_fraction: float = 0.6,
                         upstream_gap_frac: float = 0.25,
                         min_upstream_gap: float = 0.0,
                         upstream_outer_fraction: float = 0.15) -> ShockRegions:
    """Split a line-out into the shocked-ambient layer and the pristine upstream.

    A piston-driven shock has THREE regions along the normal, not two, and the middle one
    is the whole point: piston material, then a layer of compressed *ambient* between the
    contact discontinuity and the shock, then pristine upstream.  Averaging the
    "downstream" over everything behind the shock would fold the piston into it and
    report a compression ratio that is not the ambient's.

    ``shock`` is the outermost place ``ambient_density`` falls through
    ``shock_level`` times its own far-field level (:func:`~magshockz.common.
    piston_profile.ambient_reference_level`), so it does not care how tall the pile-up
    got.  ``contact`` is the outermost place the piston stops dominating the ambient,
    i.e. where ``piston_density/ambient_density`` falls through 1.

    The bands are then placed relative to those two: ``downstream`` is the middle
    ``band_fraction`` of the layer, clear of both discontinuities, and ``upstream`` is
    the FAR FIELD -- the outermost ``upstream_outer_fraction`` of the line-out, pushed
    further out still if that would not clear the shock by the gap.

    WHY THE FAR FIELD AND NOT A BAND JUST AHEAD OF THE FRONT.  A high-Mach shock's
    precursor reaches much further ahead than its density jump does: the magnetic foot
    and the streaming electrons are already there where ``ambient_density`` still reads
    upstream.  On these runs the preheat extends ~12 units ahead of the front, so a band
    placed just outside the ramp measures the precursor and calls it upstream -- |B|
    reads 1.4 B0 and T_e 600 eV against a 9.8 eV initial condition, inflating v_A and c_s
    and so understating both Mach numbers.  Averaging from the gap all the way out is no
    better: that mixes precursor with pristine gas and lands between the two.

    That the far field is pristine is an ASSUMPTION about the box, not a guarantee.  Once
    the precursor reaches the far boundary this function will happily average heated gas
    and call it upstream; a caller that cares should check the band against the run's
    initial condition.

    Parameters
    ----------
    positions, ambient_density, piston_density
        One line-out along the shock normal, same length, positions increasing.
    shock_level
        Multiple of the far-field ambient density that defines the front.  1.5 sits on
        the steep flank of a jump whose ceiling is 4 for gamma = 5/3.
    min_upstream_gap
        Floor on the shock-to-band gap, in the caller's position units.
    upstream_outer_fraction
        Fraction of the line-out, measured in from the far end, the band spans.
    """
    positions = np.asarray(positions, dtype=float)
    ambient_density = np.asarray(ambient_density, dtype=float)
    piston_density = np.asarray(piston_density, dtype=float)

    nan_regions = ShockRegions(float("nan"), float("nan"),
                               (float("nan"), float("nan")),
                               (float("nan"), float("nan")))
    reference = pp.ambient_reference_level(positions, ambient_density)
    if not (np.isfinite(reference) and reference > 0.0):
        return nan_regions

    shock = pp.front_position(positions, ambient_density, level=shock_level * reference)
    with np.errstate(divide="ignore", invalid="ignore"):
        piston_over_ambient = np.where(ambient_density > 0.0,
                                       piston_density / ambient_density, np.inf)
    contact = pp.front_position(positions, piston_over_ambient, level=1.0)
    if not (np.isfinite(shock) and np.isfinite(contact)) or contact >= shock:
        return nan_regions

    layer = shock - contact
    margin = 0.5 * (1.0 - band_fraction) * layer
    downstream = (contact + margin, shock - margin)

    start = max(shock + max(upstream_gap_frac * layer, min_upstream_gap),
                positions[-1] - upstream_outer_fraction * (positions[-1] - positions[0]))
    if start >= positions[-1]:
        return ShockRegions(shock, contact, downstream, (float("nan"), float("nan")))
    return ShockRegions(shock, contact, downstream, (start, float(positions[-1])))


def band_average(positions: np.ndarray, values: np.ndarray,
                 band: tuple[float, float]) -> float:
    """Mean of ``values`` over the closed interval ``band``; nan if it is empty.

    The companion to :func:`locate_shock_regions` -- that function names the intervals,
    this one reduces a profile over one.  Kept separate so a caller can average any
    number of quantities over the same, once-computed regions.
    """
    positions = np.asarray(positions, dtype=float)
    values = np.asarray(values, dtype=float)
    lo, hi = band
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return float("nan")

    inside = (positions >= lo) & (positions <= hi) & np.isfinite(values)
    return float(np.mean(values[inside])) if np.any(inside) else float("nan")


@dataclass(frozen=True)
class ShockedLayer:
    """One line-out's shocked-ambient layer, reduced to numbers two codes can share.

    Every field is dimensionless, so a run at a reduced ion mass and an arbitrary
    reference density is directly comparable to the FLASH run it targets.  Any field is
    nan when :func:`locate_shock_regions` could not place the bands.

    WHAT EACH IS NORMALIZED BY.  ``compression`` and ``b_ratio`` are jumps -- downstream
    over the SAME frame's upstream -- because that is what the Rankine-Hugoniot relations
    predict.  The densities and temperatures are instead quoted against each code's
    INITIAL ambient, and ``upstream_density`` says how far that upstream has drifted from
    it.  The two runs' upstreams are not in the same state: FLASH's chamber has rarefied
    and preheated by the time the piston is running, while the deck's is still at its
    initial condition, so a downstream quoted "per upstream" would divide the two codes by
    denominators differing 15-fold and call the result a disagreement.
    """

    regions: ShockRegions
    upstream_density: float
    compression: float
    velocity_over_alfven: float
    te_upstream: float
    ti_upstream: float
    te_downstream: float
    ti_downstream: float
    b_ratio: float

    @property
    def layer_thickness(self) -> float:
        """Contact discontinuity to shock front, in the caller's position units."""
        return self.regions.layer_thickness


def measure_shocked_layer(positions: np.ndarray, ambient_density: np.ndarray,
                          piston_density: np.ndarray, *, velocity: np.ndarray,
                          electron_temperature: np.ndarray,
                          ion_temperature: np.ndarray, magnetic_field: np.ndarray,
                          alfven_speed: float, reference_temperature: float,
                          min_upstream_gap: float = 0.0) -> ShockedLayer:
    """Reduce one line-out to the state of its shocked-ambient layer.

    The estimator does not know which code produced the arrays, which is the point: hand
    it FLASH's line-out and WarpX's in turn and the two :class:`ShockedLayer` s are
    comparable even though nothing dimensional about the two runs is.

    Parameters
    ----------
    positions, ambient_density, piston_density
        One line-out along the shock normal, positions increasing outward.  The ambient
        density must already be normalized by the run's INITIAL ambient density; the
        piston density need only share a scale with it.
    velocity
        Bulk velocity of the ambient population along the normal, in the same unit as
        ``alfven_speed`` (bare floats -- the caller strips units at this boundary).  The
        reported speed is the jump ``v_down - v_up``, so an upstream that is drifting does
        not inflate it.
    electron_temperature, ion_temperature
        In the same unit as ``reference_temperature``, which is the run's initial ambient
        temperature.
    magnetic_field
        Any unit; only its downstream/upstream ratio is reported.
    alfven_speed
        Upstream Alfven speed, MEASURED from the same frame rather than taken from the
        initial condition -- a rarefied or piled-up upstream has its own v_A.
    min_upstream_gap
        Floor on the shock-to-upstream-band gap, in the caller's position units.
    """
    regions = locate_shock_regions(positions, ambient_density, piston_density,
                                   min_upstream_gap=min_upstream_gap)

    def jump(values: np.ndarray) -> tuple[float, float]:
        return (band_average(positions, values, regions.downstream),
                band_average(positions, values, regions.upstream))

    density_down, density_up = jump(ambient_density)
    velocity_down, velocity_up = jump(velocity)
    te_down, te_up = jump(electron_temperature)
    ti_down, ti_up = jump(ion_temperature)
    field_down, field_up = jump(magnetic_field)
    return ShockedLayer(
        regions=regions,
        upstream_density=density_up,
        compression=density_down / density_up if density_up else float("nan"),
        velocity_over_alfven=(velocity_down - velocity_up) / alfven_speed
                             if alfven_speed else float("nan"),
        te_upstream=te_up / reference_temperature,
        ti_upstream=ti_up / reference_temperature,
        te_downstream=te_down / reference_temperature,
        ti_downstream=ti_down / reference_temperature,
        b_ratio=field_down / field_up if field_up else float("nan"))


@dataclass(frozen=True)
class CompareRow:
    """One dimensionless quantity as the two codes report it."""

    quantity: str
    flash: float
    warpx: float

    @property
    def ratio(self) -> float:
        """WarpX over FLASH: 1 is agreement, and the sign of the error is readable."""
        return self.warpx / self.flash if self.flash else float("nan")


def compare_layers(flash: ShockedLayer, warpx: ShockedLayer) -> list[CompareRow]:
    """The two codes' shocked layers side by side, in the order a reader wants them.

    Geometry first (where the two discontinuities are), then the jumps the shock is
    defined by.  Every row is dimensionless or in ``d_i``, so the reduced-mass deck and
    the FLASH run it targets are being compared on the only terms they share.

    The contact is taken from :func:`locate_shock_regions` -- where the piston stops
    dominating the ambient -- rather than from a piston-density front.  An absolute
    density threshold is not a like-for-like measurement once one code's piston has
    expanded below it while the other's has not.
    """
    return [
        CompareRow("contact [d_i]", flash.regions.contact, warpx.regions.contact),
        CompareRow("shock front [d_i]", flash.regions.shock, warpx.regions.shock),
        CompareRow("shocked layer [d_i]", flash.layer_thickness, warpx.layer_thickness),
        CompareRow("n upstream / n_0", flash.upstream_density, warpx.upstream_density),
        CompareRow("n down / n up", flash.compression, warpx.compression),
        CompareRow("v shocked / v_A", flash.velocity_over_alfven,
                   warpx.velocity_over_alfven),
        CompareRow("T_e shocked / T_0", flash.te_downstream, warpx.te_downstream),
        CompareRow("T_i shocked / T_0", flash.ti_downstream, warpx.ti_downstream),
        CompareRow("|B| down / up", flash.b_ratio, warpx.b_ratio),
    ]


def compare_text(rows: list[CompareRow], width: int = 22) -> str:
    """The comparison as a fixed-width table, for the terminal and the saved .txt."""
    lines = [f"{'quantity':<{width}}{'FLASH':>10}{'WarpX':>10}{'WarpX/FLASH':>13}"]
    lines += [f"{row.quantity:<{width}}{row.flash:>10.3g}{row.warpx:>10.3g}"
              f"{row.ratio:>13.2f}" for row in rows]
    return "\n".join(lines)


@dataclass(frozen=True)
class ScoreRow:
    """One quantity across the three places it can be read off.

    Three columns, not two, because there are two distinct ways to be wrong: the deck can
    be built with the wrong constants (``flash`` vs ``deck``), or the run can fail to
    realise the constants it was built with (``deck`` vs ``warpx``).  Collapsing them
    would hide which.
    """

    quantity: str
    flash: float
    deck: float
    warpx: float
    is_matched: bool = True

    @property
    def label(self) -> str:
        return self.quantity if self.is_matched else f"{self.quantity}  (NOT matched)"


def scorecard(scales: units.DeckScales, *, measured_speed_over_c: float,
              measured_contrast: float) -> list[ScoreRow]:
    """The FLASH target, the deck's aim and what the run delivered."""
    flash = scales.flash
    if flash is None:
        raise ValueError("scorecard needs the FLASH reference")

    def di_per_gyro(speed, ion_skin_depth, gyroperiod) -> float:
        return float((speed * gyroperiod / ion_skin_depth).decompose())

    measured_speed = measured_speed_over_c * c
    return [
        ScoreRow("front [d_i/T_ci]",
                 di_per_gyro(flash.piston_front_speed, flash.upstream.ion_skin_depth,
                             flash.upstream.gyroperiod),
                 di_per_gyro(scales.piston_speed, scales.ion_skin_depth,
                             scales.gyroperiod),
                 di_per_gyro(measured_speed, scales.ion_skin_depth, scales.gyroperiod)),
        ScoreRow("M_A", flash.mach_alfven, scales.mach_alfven,
                 float((measured_speed / scales.upstream.alfven_speed).decompose())),
        ScoreRow("n_piston / n_amb", flash.contrast, scales.contrast, measured_contrast),
        ScoreRow("v_piston / c",
                 float((flash.piston_front_speed / c).decompose()),
                 scales.piston_speed_over_c, measured_speed_over_c,
                 is_matched=False),
    ]


def scorecard_text(rows: list[ScoreRow], width: int = 28) -> str:
    """The scorecard as a fixed-width table, for both the figure and the .txt."""
    lines = [f"{'quantity':<{width}}{'FLASH':>12}{'deck aim':>12}{'WarpX':>12}"]
    lines += [f"{row.label:<{width}}{row.flash:>12.4g}{row.deck:>12.4g}{row.warpx:>12.4g}"
              for row in rows]
    return "\n".join(lines)
