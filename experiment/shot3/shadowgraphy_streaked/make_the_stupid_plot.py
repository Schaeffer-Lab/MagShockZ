"""Plot the shot-3 LIONZ diagnostics.

	python make_the_stupid_plot.py                 # both figures
	python make_the_stupid_plot.py streak          # streaked shadowgraphy only
	python make_the_stupid_plot.py interferometry  # SIMX interferometry only

The SIMX camera writes its four framing images as one 2x2 mosaic TIF; the
frames are split out and shown as a 2x2 panel.  Frame times are not recorded
in the TIF, so pass them with --times if you know them:

	python make_the_stupid_plot.py interferometry --times 3305 3310 3315 3320

The interferograms default to a flat-fielded view, which is the one worth
looking at; --enhance raw shows the frames as the camera recorded them and
--enhance bandpass keeps only the fringe carrier.

	python make_the_stupid_plot.py phase                      # phase shift [rad]
	python make_the_stupid_plot.py phase --wavelength-nm 532   # areal density

`phase` demodulates the fringes against a preshot interferogram and unwraps
the difference.  Read the per-panel `r` before believing any of it: the
interferometer's static phase drifts by ~2 rad between preshot exposures,
which is as large as the plasma signal, so a panel is only meaningful to the
extent that independent references agree on it.
"""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_fill_holes, binary_opening, gaussian_filter, label
from skimage.restoration import unwrap_phase

HERE = Path(__file__).resolve().parent
STREAK_CSV = HERE / "z4290_Laserstreak.csv"
STREAK_CALIB = HERE / "calib_updated.csv"
STREAK_PNG = HERE / "z4290_Laserstreak.png"

SIMX_DIR = HERE.parent / "LIONZ_interferometry"
SIMX_TIF = SIMX_DIR / "shot" / "z4290-simx-shot.TIF"
SIMX_PNG = SIMX_TIF.with_suffix(".png")
# Preshot interferogram with the same fringe carrier as the shot; it supplies
# the carrier and the static aberrations to divide out.
SIMX_REFERENCE = SIMX_DIR / "preshot" / "simx-nr-f-1.TIF"

# Mosaic order of the four SIMX channels, row-major from the top left.
SIMX_QUADRANTS = ("upper left", "upper right", "lower left", "lower right")

# The interferogram fringes run ~20-25 px apart, so an envelope blurred on a
# longer scale than that holds the beam profile but none of the fringes.
ENVELOPE_SIGMA = 30.0
# Radius, in FFT bins, of the DC/envelope core to ignore when hunting the
# carrier peak; the sideband sits far outside it.
CARRIER_EXCLUSION = 25
# Band-pass half-width as a fraction of the carrier wavenumber.  Wide enough to
# pass the fringe bending at the shock, narrow enough to reject the 33.9 px
# fixed-pattern artifact that survives in the single-leg-blocked frames.
BAND_FRACTION = 0.35
# Fringe visibility, relative to the frame's own bright core, below which the
# phase is meaningless and gets masked out.  Loosening this past ~0.06 buys
# coverage without measurably straining the unwrap.
VISIBILITY_FLOOR = 0.06


def stretched_gray(ax, image, gamma=1.0, percentiles=(1.0, 99.5), origin="lower"):
	"""Show `image` in gray scale, clipped to its own percentile range."""
	vmin, vmax = np.percentile(image, percentiles)
	return ax.imshow(
		image,
		origin=origin,
		cmap="gray",
		norm=plt.matplotlib.colors.PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax),
	)


def load_pixel_scales(path):
	"""Read a `type,value` calibration table into a name -> float mapping."""
	# The calibration files come out of Excel, so they carry a BOM and CRLF endings.
	with open(path, newline="", encoding="utf-8-sig") as handle:
		return {row["type"]: float(row["value"]) for row in csv.DictReader(handle)}


def plot_streak():
	data = np.loadtxt(STREAK_CSV, delimiter=",")
	scales = load_pixel_scales(STREAK_CALIB)

	missing = {"px_to_mm", "px_to_ns"} - scales.keys()
	if missing:
		raise ValueError(f"{STREAK_CALIB.name} is missing {', '.join(sorted(missing))}")

	fig, ax = plt.subplots(figsize=(10, 4))
	data_processed = np.fliplr(np.rot90(data, k=2))
	rows, cols = data_processed.shape
	image = stretched_gray(ax, data_processed, gamma=0.5, percentiles=(2, 98))
	image.set_extent([0, cols * scales["px_to_ns"], 0, rows * scales["px_to_mm"]])
	ax.set_aspect("auto")
	ax.set_xlabel("Time (ns)")
	ax.set_ylabel("Space (mm)")
	ax.set_title("LIONZ Streaked Shadowgraphy")
	fig.colorbar(image, ax=ax, label="Intensity")
	fig.tight_layout()
	fig.savefig(STREAK_PNG, dpi=150, bbox_inches="tight")
	plt.close(fig)
	print(f"Saved to {STREAK_PNG}")


def split_simx_mosaic(mosaic):
	"""Split a 2x2 SIMX mosaic into its four frames, row-major from the top left."""
	rows, cols = mosaic.shape
	half_row, half_col = rows // 2, cols // 2
	return [
		mosaic[:half_row, :half_col],
		mosaic[:half_row, half_col:],
		mosaic[half_row:, :half_col],
		mosaic[half_row:, half_col:],
	]


def flat_field(frame):
	"""Divide out the slow beam envelope, so fringe contrast is uniform across the frame.

	The SIMX beam profile falls off by more than an order of magnitude from the
	centre to the edge, which leaves the outer fringes buried under any global
	intensity scale.  Dividing by the blurred frame normalises them all to the
	local mean.
	"""
	envelope = gaussian_filter(frame, ENVELOPE_SIGMA)
	return frame / np.maximum(envelope, 1e-6)


def carrier_wavevector(spectrum):
	"""Locate the fringe-carrier sideband in a centred 2-D FFT, in (kx, ky) bins from DC.

	Refined to sub-bin precision by taking the amplitude centroid around the
	peak: an integer carrier leaves a residual tilt of several radians across
	the frame, which would swamp the plasma phase.
	"""
	rows, cols = spectrum.shape
	centre_row, centre_col = rows // 2, cols // 2
	row_index, col_index = np.mgrid[:rows, :cols]
	radius = np.hypot(row_index - centre_row, col_index - centre_col)

	power = np.abs(spectrum).copy()
	power[radius < CARRIER_EXCLUSION] = 0.0
	peak_row, peak_col = np.unravel_index(np.argmax(power), power.shape)

	window = 6
	around = (slice(peak_row - window, peak_row + window + 1), slice(peak_col - window, peak_col + window + 1))
	weight = np.abs(spectrum)[around]
	kx = (col_index[around] * weight).sum() / weight.sum() - centre_col
	ky = (row_index[around] * weight).sum() / weight.sum() - centre_row
	return kx, ky


@dataclass(frozen=True)
class Demodulation:
	"""The complex fringe envelope of one interferogram, with its carrier removed.

	`analytic` is the Takeda analytic signal: its modulus is the local fringe
	visibility and its argument is the phase relative to a uniform carrier.
	"""

	analytic: np.ndarray
	carrier: tuple[float, float]

	@property
	def visibility(self):
		return np.abs(self.analytic)


def demodulate(frame, carrier=None):
	"""Isolate the carrier sideband of `frame` and shift it to DC (Takeda's method).

	Pass `carrier` to demodulate against another frame's carrier, which is what
	makes a shot and its reference directly comparable.
	"""
	rows, cols = frame.shape
	spectrum = np.fft.fftshift(np.fft.fft2(frame - frame.mean()))
	centre_row, centre_col = rows // 2, cols // 2
	row_index, col_index = np.mgrid[:rows, :cols]

	kx, ky = carrier_wavevector(spectrum) if carrier is None else carrier
	width = BAND_FRACTION * np.hypot(kx, ky)
	band = np.exp(
		-(((col_index - centre_col - kx) ** 2 + (row_index - centre_row - ky) ** 2) / (2 * width**2))
	)
	sideband = np.fft.ifft2(np.fft.ifftshift(spectrum * band))
	carrier_ramp = np.exp(-2j * np.pi * (kx * col_index / cols + ky * row_index / rows))
	return Demodulation(analytic=sideband * carrier_ramp, carrier=(kx, ky))


def fringe_bandpass(frame):
	"""Keep only the fringe carrier and its sidelobes, dropping the envelope and the noise floor.

	A Gaussian band-pass centred on the carrier: everything the interferometer
	actually encodes lives in that band, so this is the cleanest view of the
	fringe shifts alone.
	"""
	demodulated = demodulate(frame)
	kx, ky = demodulated.carrier
	rows, cols = frame.shape
	row_index, col_index = np.mgrid[:rows, :cols]
	# Re-impose the carrier, so the result is a fringe pattern again rather than an envelope.
	return np.real(demodulated.analytic * np.exp(2j * np.pi * (kx * col_index / cols + ky * row_index / rows)))


ENHANCEMENTS = {
	"raw": (lambda frame: frame, (1.0, 99.5)),
	"flat": (flat_field, (0.5, 99.5)),
	"bandpass": (fringe_bandpass, (1.0, 99.0)),
}


@dataclass(frozen=True)
class PhaseMap:
	"""Plasma phase shift over one frame, in radians, with the pixels worth believing.

	`phase` is unwrapped and referenced to the preshot interferogram, so it is
	the shift the plasma imposed; `valid` is False wherever fringe visibility
	was too low for that to mean anything.
	"""

	phase: np.ndarray
	valid: np.ndarray
	visibility: np.ndarray

	@property
	def masked_phase(self):
		return np.ma.masked_where(~self.valid, self.phase)


def fringe_mask(visibility):
	"""Flag the pixels whose fringes are strong enough to carry phase."""
	threshold = VISIBILITY_FLOOR * np.percentile(visibility, 99)
	return binary_fill_holes(binary_opening(visibility > threshold, np.ones((9, 9))))


def largest_region(mask):
	"""Keep only the biggest connected component of `mask`.

	Phase can only be unwrapped along a connected path, so an island separated
	from the main field gets an arbitrary multiple of 2π and would read as a
	density jump that isn't there.
	"""
	labels, count = label(mask)
	if count <= 1:
		return mask

	sizes = np.bincount(labels.ravel())
	sizes[0] = 0
	return labels == sizes.argmax()


def remove_smooth_background(phase, valid, order=2):
	"""Subtract a low-order polynomial fitted over `valid`, absorbing residual tilt and defocus."""
	rows, cols = phase.shape
	row_index, col_index = np.mgrid[:rows, :cols]
	terms = [np.ones_like(col_index, float), col_index, row_index]
	if order >= 2:
		terms += [col_index**2, row_index**2, col_index * row_index]

	design = np.stack([term[valid] for term in terms], axis=1)
	coefficients, *_ = np.linalg.lstsq(design, phase[valid], rcond=None)
	return phase - sum(c * term for c, term in zip(coefficients, terms))


def plasma_phase(shot_frame, reference_frame, detrend=True):
	"""Unwrapped phase shift of `shot_frame` relative to a preshot interferogram.

	Both frames are demodulated against the *reference's* carrier, so the
	difference of their analytic signals cancels the carrier and every static
	aberration the two exposures share.  Taking the difference before
	unwrapping keeps the reference's own wrapping out of the result.
	"""
	reference = demodulate(reference_frame)
	shot = demodulate(shot_frame, carrier=reference.carrier)

	valid = largest_region(fringe_mask(shot.visibility) & fringe_mask(reference.visibility))
	wrapped = np.angle(shot.analytic * np.conj(reference.analytic))
	phase = np.asarray(unwrap_phase(np.ma.masked_array(wrapped, ~valid)))

	if detrend:
		phase = remove_smooth_background(phase, valid)
	return PhaseMap(phase=phase, valid=valid, visibility=shot.visibility)


def reference_reproducibility(shot_frame, reference_frames):
	"""Correlation between the phase maps `shot_frame` yields against different preshot references.

	The interferometer's static phase drifts by ~2 rad between preshot
	exposures, which is the same size as the plasma signal.  Recovering the
	same map from independent references is therefore the honest test of
	whether a frame's phase means anything; a low value marks a panel that
	should not be read as density.
	"""
	maps = [plasma_phase(shot_frame, reference) for reference in reference_frames]

	correlations = []
	for i in range(len(maps)):
		for j in range(i + 1, len(maps)):
			overlap = maps[i].valid & maps[j].valid
			if overlap.sum() > 100:
				correlations.append(np.corrcoef(maps[i].phase[overlap], maps[j].phase[overlap])[0, 1])

	return float(np.mean(correlations)) if correlations else float("nan")


def areal_electron_density(phase, wavelength_nm):
	"""Line-integrated electron density from a phase shift, in cm^-2.

	For an underdense plasma the probe phase shift is
	Δφ = (π / λ n_c) ∫n_e dl, with n_c the critical density at λ, so the
	line integral follows directly with no assumption about the path length.
	"""
	import astropy.units as u
	from astropy.constants import c
	from plasmapy.formulary import critical_density

	wavelength = wavelength_nm * u.nm
	omega = (2 * np.pi * u.rad * c / wavelength).to("rad/s")
	n_critical = critical_density(omega)
	return (phase * wavelength * n_critical / np.pi).to("cm**-2").value


def plot_interferometry(times=None, px_to_mm=None, enhance="flat"):
	mosaic = plt.imread(SIMX_TIF).astype(float)
	frames = split_simx_mosaic(mosaic)
	transform, percentiles = ENHANCEMENTS[enhance]

	if times is not None and len(times) != len(frames):
		raise ValueError(f"--times needs {len(frames)} values, one per SIMX frame")

	fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
	for index, (ax, frame, quadrant) in enumerate(zip(axes.ravel(), frames, SIMX_QUADRANTS)):
		image = stretched_gray(ax, transform(frame), percentiles=percentiles, origin="upper")
		if px_to_mm is not None:
			rows, cols = frame.shape
			image.set_extent([0, cols * px_to_mm, 0, rows * px_to_mm])
			ax.set_xlabel("x (mm)")
			ax.set_ylabel("y (mm)")
		else:
			ax.set_xlabel("x (px)")
			ax.set_ylabel("y (px)")
		ax.set_aspect("equal")
		ax.set_title(f"{times[index]:g} ns" if times is not None else quadrant)

	fig.suptitle(f"LIONZ Interferometry (SIMX, z4290) — {enhance}")
	fig.tight_layout()
	output = SIMX_PNG if enhance == "flat" else SIMX_PNG.with_name(f"{SIMX_PNG.stem}-{enhance}.png")
	fig.savefig(output, dpi=150, bbox_inches="tight")
	plt.close(fig)
	print(f"Saved to {output}")


def plot_phase(times=None, px_to_mm=None, reference=SIMX_REFERENCE, wavelength_nm=None, flip_sign=False, check_reproducibility=True):
	"""Phase-shift maps for the four frames, or areal electron density when given a wavelength.

	Each panel is annotated with its reference-swap correlation, which is the
	only thing separating a real density map from unwrapped noise here.
	"""
	shot_frames = split_simx_mosaic(plt.imread(SIMX_TIF).astype(float))
	reference_frames = split_simx_mosaic(plt.imread(reference).astype(float))

	alternates = sorted(p for p in (SIMX_DIR / "preshot").glob("simx-*f*.TIF") if p != reference)
	alternate_quadrants = [split_simx_mosaic(plt.imread(p).astype(float)) for p in alternates]

	fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))
	for index, (ax, frame, reference_frame, quadrant) in enumerate(
		zip(axes.ravel(), shot_frames, reference_frames, SIMX_QUADRANTS)
	):
		phase_map = plasma_phase(frame, reference_frame)
		signed = -phase_map.phase if flip_sign else phase_map.phase

		if wavelength_nm is None:
			field, label = signed, "phase shift [rad]"
		else:
			field, label = areal_electron_density(signed, wavelength_nm), r"$\int n_e\,dl$ [cm$^{-2}$]"

		scale = np.percentile(np.abs(field[phase_map.valid]), 98)
		image = ax.imshow(
			np.ma.masked_where(~phase_map.valid, field), cmap="RdBu_r", vmin=-scale, vmax=scale
		)
		if px_to_mm is not None:
			rows, cols = frame.shape
			image.set_extent([0, cols * px_to_mm, 0, rows * px_to_mm])
		title = f"{times[index]:g} ns" if times is not None else quadrant
		covered = 100 * phase_map.valid.mean()

		correlation = float("nan")
		if check_reproducibility and alternate_quadrants:
			references = [reference_frame] + [quadrants[index] for quadrants in alternate_quadrants]
			correlation = reference_reproducibility(frame, references)
			title += f"   (r={correlation:.2f}{'' if correlation > 0.5 else ', UNRELIABLE'})"

		ax.set_title(title)
		ax.axis("off")
		fig.colorbar(image, ax=ax, fraction=0.04, label=label)

		span = np.percentile(field[phase_map.valid], [1, 99])
		print(f"  {quadrant:12s} valid {covered:4.0f}%   1-99% span {span[0]:+.3g} to {span[1]:+.3g}   reference-swap r={correlation:+.2f}")

	fig.suptitle(f"LIONZ Interferometry (SIMX, z4290) — {'phase' if wavelength_nm is None else 'areal density'}")
	fig.tight_layout()
	output = SIMX_PNG.with_name(f"{SIMX_PNG.stem}-{'phase' if wavelength_nm is None else 'density'}.png")
	fig.savefig(output, dpi=150, bbox_inches="tight")
	plt.close(fig)
	print(f"Saved to {output}")


def main():
	parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument(
		"figures", nargs="?", default="both", choices=["both", "streak", "interferometry", "phase"]
	)
	parser.add_argument("--times", type=float, nargs=4, metavar="NS", help="frame times, in mosaic order")
	parser.add_argument("--px-to-mm", type=float, help="SIMX plate scale; without it the axes stay in pixels")
	parser.add_argument("--enhance", default="flat", choices=sorted(ENHANCEMENTS), help="interferogram contrast treatment")
	parser.add_argument("--reference", type=Path, default=SIMX_REFERENCE, help="preshot interferogram to reference the phase to")
	parser.add_argument("--wavelength-nm", type=float, help="probe wavelength; turns the phase map into an areal density")
	parser.add_argument("--flip-sign", action="store_true", help="flip the phase sign, if the dense region comes out negative")
	args = parser.parse_args()

	if args.figures in ("both", "streak"):
		plot_streak()
	if args.figures in ("both", "interferometry"):
		plot_interferometry(times=args.times, px_to_mm=args.px_to_mm, enhance=args.enhance)
	if args.figures == "phase":
		plot_phase(
			times=args.times,
			px_to_mm=args.px_to_mm,
			reference=args.reference,
			wavelength_nm=args.wavelength_nm,
			flip_sign=args.flip_sign,
		)


if __name__ == "__main__":
	main()
