"""Empirical heater calibration: what setpoint delivers what piston speed.

The heater has no setpoint feedback -- ``particle_heater.<species>.theta`` only sets the
amplitude of a momentum-space diffusion rate ``H ~ theta^{3/2}`` -- and the runs on
record reach only 71-80% of the temperature they are handed, still climbing at run end.
So the map from setpoint to piston speed is MEASURED, not modelled.

Runs are compared in the sound-speed grouping ``S = theta / (m/Ze)_piston``, which is
``(c_s/c)^2``, so a calibration taken at one mass ratio transfers to another.  The fit is

    v_piston / c = amplitude * S**exponent

and the measured exponent is ~0.71, NOT the 0.5 a ``v_piston = kappa c_s`` closure
assumes -- which is exactly why a fixed ``kappa`` kept mis-predicting the front speed
(it was read as 2.5, then re-measured as 1.01, then implied 1.20 and 1.33 by the two
runs below).  Nothing here assumes a closure; add runs and the fit improves.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import astropy.units as u
from astropy.constants import c


@dataclass(frozen=True)
class CalibrationPoint:
    """One completed run: what the heater was told, and what the piston did."""

    run_id: str
    heater_theta: float
    piston_mass_per_charge: float
    piston_speed: u.Quantity
    achieved_theta: float = math.nan
    #: The piston's charge state. At Z = 1 the heater's own ``foil.mass_ratio``
    #: (bare ``m_i/m_e``) coincides with ``m/(Z m_e)``; at any other Z they differ by Z,
    #: so a fit taken at one Z does not transfer to another. See ``config.validate``.
    charge_number: int = 1

    @property
    def drive(self) -> float:
        """``S = theta/(m/Ze)_piston = (c_s/c)^2`` -- the mass-ratio-free drive."""
        return self.heater_theta / self.piston_mass_per_charge

    @property
    def piston_speed_over_c(self) -> float:
        return float((self.piston_speed / c).decompose())

    @property
    def achieved_fraction(self) -> float:
        """Achieved / setpoint temperature, from the run's ParticleEnergy history."""
        return self.achieved_theta / self.heater_theta


#: The runs on record.  Both ran at Z = 1 and mass_ratio 100, so their piston
#: mass-per-charge is 100.  ``achieved_theta`` is (2/3)<KE>/(m_e c^2) for the piston
#: electrons at the last dump of ``diags/reducedfiles/EP.txt``.
MEASURED: tuple[CalibrationPoint, ...] = (
    CalibrationPoint("magshockz_2d_heater_v1", 0.04, 100.0, 0.0202 * c, 0.02859),
    CalibrationPoint("magshockz_2d_heater_v2", 0.088227, 100.0, 0.035284 * c, 0.07067),
)


@dataclass(frozen=True)
class HeaterCalibration:
    """Power-law fit ``v_piston/c = amplitude * (theta/(m/Ze))**exponent``."""

    amplitude: float
    exponent: float
    points: tuple[CalibrationPoint, ...]

    @property
    def drive_range(self) -> tuple[float, float]:
        drives = [point.drive for point in self.points]
        return min(drives), max(drives)

    def piston_speed(self, heater_theta: float,
                     piston_mass_per_charge: float) -> u.Quantity:
        """Predicted front speed for a setpoint at this piston mass-per-charge."""
        drive = heater_theta / piston_mass_per_charge
        return self.amplitude * drive**self.exponent * c

    def heater_theta(self, piston_speed: u.Quantity,
                     piston_mass_per_charge: float) -> float:
        """The setpoint predicted to deliver ``piston_speed`` -- the inverse fit."""
        speed_over_c = float((piston_speed / c).decompose())
        if speed_over_c <= 0.0:
            raise ValueError(f"piston_speed must be positive, got {piston_speed!r}")
        drive = (speed_over_c / self.amplitude) ** (1.0 / self.exponent)
        return drive * piston_mass_per_charge

    def extrapolation_warning(self, heater_theta: float,
                              piston_mass_per_charge: float) -> str | None:
        """Message if this setpoint sits outside the measured drives, else ``None``."""
        drive = heater_theta / piston_mass_per_charge
        lo, hi = self.drive_range
        if lo <= drive <= hi:
            return None
        factor = drive / hi if drive > hi else lo / drive
        return (
            f"heater setpoint theta = {heater_theta:.4g} at (m/Ze)_piston = "
            f"{piston_mass_per_charge:.3g} gives drive S = {drive:.3g}, a factor "
            f"{factor:.2g} outside the calibrated range [{lo:.3g}, {hi:.3g}] measured "
            f"from {len(self.points)} run(s). The predicted piston speed is an "
            f"extrapolation -- expect to re-calibrate after this run.")


def fit(points: tuple[CalibrationPoint, ...] = MEASURED) -> HeaterCalibration:
    """Least-squares power law through the measured runs.

    Two points give an exact fit; more are averaged in log-log, where the power law is
    a straight line.
    """
    if len(points) < 2:
        raise ValueError(
            f"need at least 2 calibration points to fit a power law, got {len(points)}")

    log_drive = [math.log(point.drive) for point in points]
    log_speed = [math.log(point.piston_speed_over_c) for point in points]
    n = len(points)
    mean_drive = sum(log_drive) / n
    mean_speed = sum(log_speed) / n
    covariance = sum((d - mean_drive) * (s - mean_speed)
                     for d, s in zip(log_drive, log_speed))
    variance = sum((d - mean_drive) ** 2 for d in log_drive)
    if variance == 0.0:
        raise ValueError(
            "all calibration points share one drive value, so no exponent is "
            "determined -- vary the heater setpoint between runs")

    exponent = covariance / variance
    amplitude = math.exp(mean_speed - exponent * mean_drive)
    return HeaterCalibration(amplitude=amplitude, exponent=exponent, points=points)
