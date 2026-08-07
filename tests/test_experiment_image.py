"""Tests for experiment_image.py — cropping a decorated streak PNG and placing FLASH
onto its axes.

The image's own (ns, mm) axes are the reference frame: it is only ever pixel-cropped,
never stretched or resampled, so the invariant most of these tests pin down is that a
feature keeps its ns/mm coordinates through every operation.

Pure numpy: the module imports matplotlib only inside ``load_streak``, so none of
these touch it (or any image file).
"""

import os

import numpy as np
import pytest

from magshockz.analysis.flash import experiment_image as ei


# ---------------------------------------------------------------------------
# Fixtures: a synthetic "decorated figure" — dark data box on a white page, with
# a few stray dark pixels standing in for tick labels and axis text.
# ---------------------------------------------------------------------------

BOX = (20, 90, 10, 60)   # left, right, top, bottom (half-open, slice-ready)


def make_figure():
    page = np.ones((80, 100), dtype=float)
    left, right, top, bottom = BOX
    page[top:bottom, left:right] = 0.2
    page[70, 25:30] = 0.0        # a tick label below the axes
    page[5:8, 3] = 0.0           # a y-axis label to the left
    return page


# ---------------------------------------------------------------------------
# to_gray
# ---------------------------------------------------------------------------

def test_to_gray_averages_rgb_and_drops_alpha():
    rgba = np.zeros((2, 3, 4), dtype=float)
    rgba[..., 0] = 1.0        # R
    rgba[..., 3] = 0.25       # alpha, must be ignored
    assert np.allclose(ei.to_gray(rgba), 1.0 / 3.0)


def test_to_gray_scales_uint8():
    assert np.allclose(ei.to_gray(np.full((2, 2), 255, dtype=np.uint8)), 1.0)


def test_to_gray_rejects_4d():
    with pytest.raises(ValueError):
        ei.to_gray(np.zeros((2, 2, 2, 2)))


# ---------------------------------------------------------------------------
# detect_plot_box / crop
# ---------------------------------------------------------------------------

def test_detect_plot_box_finds_the_data_rectangle():
    assert ei.detect_plot_box(make_figure()) == BOX


def test_detect_plot_box_ignores_sparse_text_pixels():
    page = make_figure()
    page[75, :] = 0.0            # a long but isolated dark line: a row, not a column
    left, right, top, bottom = ei.detect_plot_box(page)
    assert (left, right) == (BOX[0], BOX[1])   # columns unaffected
    assert top == BOX[2]                        # the stray row extends the box only downward


def test_detect_plot_box_raises_when_nothing_is_dark():
    with pytest.raises(ValueError):
        ei.detect_plot_box(np.ones((10, 10)))


def test_crop_is_slice_ready():
    page = make_figure()
    out = ei.crop(page)
    assert out.shape == (BOX[3] - BOX[2], BOX[1] - BOX[0])
    assert np.allclose(out, 0.2)


def test_crop_honours_an_explicit_box():
    page = make_figure()
    assert ei.crop(page, (0, 10, 0, 5)).shape == (5, 10)


def test_crop_rejects_an_empty_box():
    with pytest.raises(ValueError):
        ei.crop(make_figure(), (10, 10, 0, 5))


# ---------------------------------------------------------------------------
# StreakImage axes
# ---------------------------------------------------------------------------

def make_streak():
    # img[x, t]: 4 rows of mm, 5 columns of ns; value encodes (row, col).
    img = np.arange(20, dtype=float).reshape(4, 5)
    return ei.StreakImage(img=img, t_ns=(0.0, 10.0), x_mm=(-2.0, 2.0))


def test_axes_are_pixel_centres_spanning_the_outer_edges():
    s = make_streak()
    assert np.allclose(s.t_axis(), [1.0, 3.0, 5.0, 7.0, 9.0])
    assert np.allclose(s.x_axis(), [-1.5, -0.5, 0.5, 1.5])


def test_column_picks_the_nearest_time():
    s = make_streak()
    values, t = s.column(6.9)
    assert t == 7.0
    assert np.allclose(values, s.img[:, 3])


def test_extent_is_imshow_ordered():
    assert make_streak().extent == (0.0, 10.0, -2.0, 2.0)


# ---------------------------------------------------------------------------
# crop_window — a pixel crop that preserves the calibration
# ---------------------------------------------------------------------------

def test_crop_window_keeps_features_at_the_same_coordinates():
    s = make_streak()                       # 5 columns of 2 ns, 4 rows of 1 mm
    out = ei.crop_window(s, t_ns=(4.0, 8.0))
    # Whole pixels are kept, so the edges land on pixel boundaries, not on the request.
    assert out.t_ns == (4.0, 8.0)
    assert out.x_mm == s.x_mm
    assert np.array_equal(out.img, s.img[:, 2:4])
    # A feature's time is unchanged by the crop — the point of cropping over rescaling.
    assert np.allclose(out.t_axis(), s.t_axis()[2:4])


def test_crop_window_rounds_outward_to_whole_pixels():
    s = make_streak()
    out = ei.crop_window(s, t_ns=(4.5, 5.5))   # inside a single pixel (4-6 ns)
    assert out.t_ns == (4.0, 6.0)
    assert out.img.shape[1] == 1


def test_crop_window_crops_both_axes():
    s = make_streak()
    out = ei.crop_window(s, t_ns=(0.0, 4.0), x_mm=(-1.0, 1.0))
    assert out.img.shape == (2, 2)
    assert out.t_ns == (0.0, 4.0)
    assert out.x_mm == (-1.0, 1.0)
    assert np.array_equal(out.img, s.img[1:3, 0:2])


def test_crop_window_never_pads_beyond_the_image():
    s = make_streak()
    out = ei.crop_window(s, t_ns=(-50.0, 500.0), x_mm=(-99.0, 99.0))
    assert np.array_equal(out.img, s.img)
    assert (out.t_ns, out.x_mm) == (s.t_ns, s.x_mm)


def test_crop_window_accepts_a_reversed_window():
    s = make_streak()
    assert ei.crop_window(s, t_ns=(8.0, 4.0)).t_ns == (4.0, 8.0)


def test_crop_window_is_a_no_op_without_a_window():
    s = make_streak()
    out = ei.crop_window(s)
    assert np.array_equal(out.img, s.img)
    assert (out.t_ns, out.x_mm) == (s.t_ns, s.x_mm)


# ---------------------------------------------------------------------------
# Registration — FLASH onto the image's axes
# ---------------------------------------------------------------------------

def test_offsets_translate_flash_onto_the_experiment_axes():
    reg = ei.Registration(t_offset_ns=3.0, x_offset_mm=0.7)
    assert np.isclose(reg.to_exp_t(7.0), 10.0)          # FLASH 7 ns is 10 ns on camera
    assert np.isclose(reg.to_exp_mm(1000.0), 1.7)       # LOS 1 mm sits at 1.7 mm


def test_flip_reverses_the_spatial_direction():
    reg = ei.Registration(x_offset_mm=0.7, flip_space=True)
    assert np.isclose(reg.to_exp_mm(1000.0), -0.3)
    assert np.isclose(reg.to_exp_mm(-1000.0), 1.7)


@pytest.mark.parametrize("flip", [False, True])
def test_round_trip_between_the_frames(flip):
    reg = ei.Registration(t_offset_ns=-2.5, x_offset_mm=1.25, flip_space=flip)
    mm = np.array([-5.0, 0.0, 3.3])
    ns = np.array([0.0, 17.0, 68.5])
    assert np.allclose(reg.to_exp_mm(reg.to_los_um(mm)), mm)
    assert np.allclose(reg.to_exp_t(reg.to_flash_t(ns)), ns)


def test_flash_extent_reports_where_the_simulation_lands():
    reg = ei.Registration(t_offset_ns=2.0, x_offset_mm=7.0, flip_space=True)
    t0, t1, x0, x1 = reg.flash_extent(np.array([0.0, 15.0]), np.array([0.0, 6300.0]))
    assert (t0, t1) == (2.0, 17.0)
    assert np.allclose((x0, x1), (0.7, 7.0))


def test_flash_on_exp_axis_keeps_values_with_their_positions():
    x_um = np.array([0.0, 1000.0, 2000.0])
    streak = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])   # [n_dumps, n_x]

    mm, v = ei.flash_on_exp_axis(x_um, streak, ei.Registration(x_offset_mm=7.0))
    assert np.allclose(mm, [7.0, 8.0, 9.0])
    assert np.array_equal(v, streak)

    mm, v = ei.flash_on_exp_axis(x_um, streak,
                                 ei.Registration(x_offset_mm=7.0, flip_space=True))
    assert np.allclose(mm, [5.0, 6.0, 7.0])      # ascending, so the data reverses too
    assert np.array_equal(v, streak[:, ::-1])
    # The value that was at LOS 0 (mm 7) is still at mm 7.
    assert v[0, np.argmin(np.abs(mm - 7.0))] == streak[0, 0]


def test_flash_on_exp_axis_handles_a_single_lineout():
    mm, v = ei.flash_on_exp_axis(np.array([0.0, 1000.0]), np.array([3.0, 4.0]),
                                 ei.Registration(flip_space=True))
    assert np.allclose(mm, [-1.0, 0.0])
    assert np.array_equal(v, [4.0, 3.0])


def test_from_config_reads_the_registration_block():
    reg = ei.from_config({"t_offset_ns": 2.0, "x_offset_mm": -0.5, "flip_space": True})
    assert (reg.t_offset_ns, reg.x_offset_mm, reg.flip_space) == (2.0, -0.5, True)
    empty = ei.from_config(None)
    assert (empty.t_offset_ns, empty.x_offset_mm, empty.flip_space) == (0.0, 0.0, False)


def test_from_config_treats_null_as_zero():
    reg = ei.from_config({"t_offset_ns": None, "x_offset_mm": None})
    assert (reg.t_offset_ns, reg.x_offset_mm) == (0.0, 0.0)


# ---------------------------------------------------------------------------
# Trajectory — straight-line features overlaid on the streaks
# ---------------------------------------------------------------------------

def test_trajectory_slope_is_km_per_s_as_mm_per_ns():
    assert np.isclose(ei.Trajectory("f", v_kms=300.0, x0_mm=0.0).slope_mm_per_ns, 0.3)


def test_experiment_frame_line_ignores_the_registration():
    """A feature measured in the data must not move when FLASH is re-registered."""
    tr = ei.Trajectory("front", v_kms=300.0, x0_mm=-1.0, t0_ns=8.0)
    t = np.array([8.0, 18.0])
    for reg in (None, ei.Registration(t_offset_ns=40.0, x_offset_mm=3.0, flip_space=True)):
        ts, mm = tr.points(t, reg)
        assert np.allclose(ts, t)
        assert np.allclose(mm, [-1.0, 2.0])       # 300 km/s over 10 ns = 3 mm


def test_flash_frame_line_is_translated_by_the_registration():
    tr = ei.Trajectory("sim", v_kms=1000.0, x0_mm=0.0, t0_ns=0.0, frame="flash")
    reg = ei.Registration(t_offset_ns=8.0, x_offset_mm=-1.0)
    # experiment t=8 ns is FLASH t=0 → LOS 0 mm → experiment -1 mm
    # experiment t=13 ns is FLASH t=5 ns → LOS 5 mm → experiment +4 mm
    ts, mm = tr.points(np.array([8.0, 13.0]), reg)
    assert np.allclose(mm, [-1.0, 4.0])


def test_flash_frame_line_follows_a_flip():
    tr = ei.Trajectory("sim", v_kms=1000.0, x0_mm=0.0, frame="flash")
    reg = ei.Registration(x_offset_mm=2.0, flip_space=True)
    _, mm = tr.points(np.array([0.0, 3.0]), reg)
    assert np.allclose(mm, [2.0, -1.0])           # runs the other way


def test_flash_frame_line_needs_a_registration():
    with pytest.raises(ValueError, match="Registration"):
        ei.Trajectory("sim", v_kms=1.0, x0_mm=0.0, frame="flash").points([0.0, 1.0])


def test_trajectory_rejects_an_unknown_frame():
    with pytest.raises(ValueError, match="frame"):
        ei.Trajectory("x", v_kms=1.0, x0_mm=0.0, frame="lab")


def test_trajectory_at_gives_the_position_at_one_time():
    tr = ei.Trajectory("piston", v_kms=150.0, x0_mm=-1.0, t0_ns=8.0)
    assert np.isclose(tr.at(18.0), 0.5)


def test_trajectories_from_config_builds_the_list():
    trs = ei.trajectories_from_config([
        {"label": "shock", "v_kms": 300, "x0_mm": -1.0, "t0_ns": 8.0, "color": "cyan"},
        {"v_kms": 150, "x0_mm": -1.0},
    ])
    assert [t.label for t in trs] == ["shock", "feature 2"]
    assert trs[0].color == "cyan"
    assert (trs[1].t0_ns, trs[1].frame, trs[1].style) == (0.0, "experiment", "--")


def test_trajectories_from_config_handles_no_entries():
    assert ei.trajectories_from_config(None) == []
    assert ei.trajectories_from_config([]) == []


def test_trajectories_from_config_reports_missing_and_unknown_keys():
    with pytest.raises(KeyError, match="v_kms"):
        ei.trajectories_from_config([{"x0_mm": 0.0}])
    with pytest.raises(KeyError, match="speed_kms"):
        ei.trajectories_from_config([{"v_kms": 1.0, "x0_mm": 0.0, "speed_kms": 2.0}])


# ---------------------------------------------------------------------------
# Raw CSV + calibration
# ---------------------------------------------------------------------------

def test_load_calib_reads_the_two_factors(tmp_path):
    p = tmp_path / "calib.csv"
    p.write_text("type,value\npx_to_mm,0.005026907\npx_to_ns,0.033482716\n")
    c = ei.load_calib(str(p))
    assert c == {"px_to_mm": 0.005026907, "px_to_ns": 0.033482716}


def test_load_calib_raises_when_a_factor_is_missing(tmp_path):
    p = tmp_path / "calib.csv"
    p.write_text("type,value\npx_to_mm,0.005\n")
    with pytest.raises(KeyError):
        ei.load_calib(str(p))


def _write_csv(tmp_path, arr):
    p = tmp_path / "streak.csv"
    np.savetxt(str(p), arr, delimiter=",")
    return str(p)


CALIB = {"px_to_mm": 0.01, "px_to_ns": 0.5}


def test_load_streak_csv_calibrates_from_pixel_pitch(tmp_path):
    arr = np.arange(24, dtype=float).reshape(4, 6)      # 4 rows (mm), 6 cols (ns)
    s = ei.load_streak_csv(_write_csv(tmp_path, arr), CALIB, cache=False)
    assert s.img.shape == (4, 6)
    assert np.allclose(s.t_ns, (0.0, 3.0))             # 6 px x 0.5 ns
    assert np.allclose(s.x_mm, (-0.02, 0.02))          # 4 px x 0.01 mm, centred
    assert np.allclose(s.img, arr[::-1])               # row 0 was the top of the image


def test_load_streak_csv_origin_and_t0(tmp_path):
    arr = np.zeros((4, 6))
    p = _write_csv(tmp_path, arr)
    bottom = ei.load_streak_csv(p, CALIB, origin="bottom", t0_ns=5.0, cache=False)
    assert np.allclose(bottom.x_mm, (0.0, 0.04))
    assert np.allclose(bottom.t_ns, (5.0, 8.0))
    at = ei.load_streak_csv(p, CALIB, origin=-1.5, cache=False)
    assert np.allclose(at.x_mm, (-1.5, -1.46))


def test_load_streak_csv_can_keep_the_file_order(tmp_path):
    arr = np.arange(24, dtype=float).reshape(4, 6)
    s = ei.load_streak_csv(_write_csv(tmp_path, arr), CALIB, row0_is_top=False,
                           cache=False)
    assert np.allclose(s.img, arr)


def test_load_streak_csv_caches_an_npy_beside_the_data(tmp_path):
    arr = np.arange(24, dtype=float).reshape(4, 6)
    path = _write_csv(tmp_path, arr)
    first = ei.load_streak_csv(path, CALIB)
    assert os.path.exists(os.path.splitext(path)[0] + ".npy")
    # The cache must reproduce the parse exactly, orientation included.
    assert np.allclose(ei.load_streak_csv(path, CALIB).img, first.img)


# ---------------------------------------------------------------------------
# fit_shift — recovering a known translation
# ---------------------------------------------------------------------------

def _diagonal_scene(n_x=120, n_t=160, slope=0.6, width=4.0):
    """A stand-in streak: a moving front, a stationary band and a localised blob.

    A single straight ridge would be degenerate — sliding along it leaves the overlap
    unchanged — which is exactly the failure mode ``fit_shift``'s r_map is meant to
    expose, so the scene carries extra structure to make the placement unique.
    """
    t = np.arange(n_t)[None, :]
    x = np.arange(n_x)[:, None]
    front = np.exp(-0.5 * ((x - (20 + slope * t)) / width) ** 2)
    band = 0.6 * np.exp(-0.5 * ((x - 95.0) / 3.0) ** 2) * np.ones_like(t)
    blob = 1.5 * np.exp(-0.5 * (((x - 55.0) / 6.0) ** 2 + ((t - 95.0) / 6.0) ** 2))
    ramp = 0.8 * np.exp(-x / 40.0) * np.ones_like(t)   # breaks the up/down symmetry
    return front + band + blob + ramp


@pytest.mark.parametrize("feature", ["signal", "grad"])
def test_fit_shift_recovers_a_known_offset(feature):
    scene = _diagonal_scene()
    streak = ei.StreakImage(img=scene, t_ns=(0.0, 160.0), x_mm=(0.0, 120.0))  # 1 unit/px

    # "FLASH": a patch of the same scene, on its own axes starting at zero. Placing it
    # back at (t=40, x=20) must reproduce the picture, so that is the answer.
    patch = scene[20:80, 40:110]
    t_flash = np.arange(patch.shape[1], dtype=float)
    x_flash = np.arange(patch.shape[0], dtype=float)

    fit = ei.fit_shift(streak, t_flash, x_flash, patch.T, feature=feature,
                       decimate=1, smooth_px=1.0)
    assert fit.r > 0.45
    assert fit.registration.flip_space is False
    assert abs(fit.registration.t_offset_ns - 40.0) <= 2.0
    assert abs(fit.registration.x_offset_mm - 20.0) <= 2.0


def test_fit_shift_prefers_the_orientation_that_matches():
    scene = _diagonal_scene()
    streak = ei.StreakImage(img=scene, t_ns=(0.0, 160.0), x_mm=(0.0, 120.0))
    patch = scene[20:80, 40:110]
    t_flash = np.arange(patch.shape[1], dtype=float)
    x_flash = np.arange(patch.shape[0], dtype=float)
    fit = ei.fit_shift(streak, t_flash, x_flash, patch.T, decimate=1, smooth_px=1.0)
    assert fit.flip_r[False] > fit.flip_r[True]


def test_fit_shift_reports_the_map_over_every_trialled_shift():
    scene = _diagonal_scene()
    streak = ei.StreakImage(img=scene, t_ns=(0.0, 160.0), x_mm=(0.0, 120.0))
    patch = scene[20:80, 40:110]
    fit = ei.fit_shift(streak, np.arange(70.0), np.arange(60.0), patch.T,
                       decimate=1, smooth_px=1.0)
    assert fit.r_map.shape == (fit.x_offsets.size, fit.t_offsets.size)
    assert np.isclose(fit.r_map.max(), fit.r)


def test_fit_shift_rejects_an_unknown_feature():
    streak = ei.StreakImage(img=np.zeros((10, 10)), t_ns=(0.0, 1.0), x_mm=(0.0, 1.0))
    with pytest.raises(ValueError):
        ei.fit_shift(streak, np.arange(3.0), np.arange(3.0), np.zeros((3, 3)),
                     feature="nope")


def test_fit_shift_rejects_mismatched_values():
    streak = ei.StreakImage(img=np.zeros((10, 10)), t_ns=(0.0, 1.0), x_mm=(0.0, 1.0))
    with pytest.raises(ValueError):
        ei.fit_shift(streak, np.arange(3.0), np.arange(4.0), np.zeros((3, 3)))


def test_fit_shift_refuses_when_flash_is_bigger_than_the_image():
    streak = ei.StreakImage(img=np.zeros((10, 10)), t_ns=(0.0, 10.0), x_mm=(0.0, 10.0))
    with pytest.raises(ValueError, match="covers more"):
        ei.fit_shift(streak, np.arange(0.0, 100.0, 1.0), np.arange(0.0, 100.0, 1.0),
                     np.zeros((100, 100)), decimate=1)


# ---------------------------------------------------------------------------
# overlap_window
# ---------------------------------------------------------------------------

def test_overlap_window_is_the_intersection():
    s = make_streak()                       # image: 0-10 ns, -2..2 mm
    reg = ei.Registration()
    t_lo, t_hi, x_lo, x_hi = ei.overlap_window(
        s, reg, np.array([2.0, 15.0]), np.array([0.0, 6300.0]))
    assert (t_lo, t_hi) == (2.0, 10.0)
    assert (x_lo, x_hi) == (0.0, 2.0)


def test_overlap_window_reports_an_empty_interval_when_disjoint():
    s = make_streak()
    reg = ei.Registration(t_offset_ns=100.0)     # puts FLASH at 100..115 ns
    t_lo, t_hi, _, _ = ei.overlap_window(
        s, reg, np.array([0.0, 15.0]), np.array([0.0, 6300.0]))
    assert t_hi <= t_lo
