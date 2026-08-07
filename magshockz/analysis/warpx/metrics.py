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
