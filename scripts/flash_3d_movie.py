# -*- coding: utf-8 -*-
"""scripts/flash_3d_movie.py — 3D volume-rendered movies of a FLASH run.

Per dump: sample each field onto an N^3 uniform grid, crop the dense basal target slab
(which otherwise renders as a flat opaque sheet hiding the plume), volume-render from a
fixed camera, and encode each field's frames to an MP4.

Fields (--fields): ``ne te ti bx by bz bmag``. The camera and transfer functions are
FIXED across a movie so colour maps to the same physical value in every frame, and that
tuning is per-run: each run's settings are a named entry in ``PRESETS``, selected with
--preset, which must match the --config's data.

Sampling each 24-36 GB AMR dump is the bottleneck, so run this as a batch job on a COMPUTE
node (scripts/flash_3d_movie.sbatch), never on a login node. It is resumable: sampled
grids are cached under <out>/grids/ and reused, so re-tuning a transfer function
re-renders in seconds. Existing frames are skipped unless --force.

Usage
-----
    python scripts/flash_3d_movie.py --config config/flash_3d_2026-07.yaml \\
        --preset trantham2026-07 \\
        [--fields ne te ti bx by bz bmag] [--stride 1] [--t-start 0] [--t-stop 61] \\
        [--grid-res 256] [--img-res 800] [--fps 5] [--force] [--no-encode]

    # a bare FLASH dump directory, no config
    python scripts/flash_3d_movie.py --data-dir <dir> --preset trantham2026-07

    # new run: prime the grid cache (the slow half), then tune a PRESETS entry for free
    # (--width/--campos/--focus/--ycrop override the preset for a one-off test frame)
    python scripts/flash_3d_movie.py --config ...yaml --fields ne bz --grids-only
"""

import argparse
import gc
import glob
import os
import subprocess
import sys

import numpy as np
import yt

yt.set_log_level(50)
# NOTE: do NOT yt.enable_plugins() here. The flash2osiris plugin registers OSIRIS
# derived fields (Ex, Jz, …) globally; yt then tries to validate them against the
# uniform-grid dataset we build for rendering and fails on ('flash',…).  A plain
# yt.load already returns ('gas','El_number_density') and — because these runs are
# unitsystem="none" — applies the correct sqrt(4π) on B (magnetic_unit = sqrt(4π)
# G), so B in Gauss is physical without the plugin.  See CLAUDE.md.

_HERE = os.path.dirname(os.path.abspath(__file__))

from magshockz.common import analysis_utils
from magshockz.common import flash_source

# NB: intentionally NOT importing flash_utils — its module-level yt.enable_plugins()
# would register the OSIRIS derived fields globally and break the uniform-grid scene
# (see the note above). find_plot_files is inlined below instead.

K_TO_EV = 8.617333262e-5  # Kelvin -> eV (k_B/e)


def find_plot_files(data_dir):
    """Sorted FLASH plot-file paths in data_dir (mirrors flash_utils.find_plot_files)."""
    files = sorted(glob.glob(os.path.join(data_dir, "MagShockZ_hdf5_plt_cnt_*")))
    if not files:
        raise FileNotFoundError(f"No FLASH plot files found in {data_dir}")
    return files


# ---------------------------------------------------------------------------
# Default camera settings (tuned on dump 9 of the ~5 ns FLASH_3D_noshield run;
# held constant across a movie so colour == physical value in every frame).
# All four are CLI flags — a longer run needs a wider frame (--width) and a
# focus/position further out (--focus/--campos) to keep the plume in view.
# See PRESETS below for the per-run tunings.
# ---------------------------------------------------------------------------
FOCUS  = [0.0, 0.12, 0.0]     # cm — look at the plume, a bit above the base
NORTH  = [0, 1, 0]            # y is vertical (the plume rises in +y)
CAMPOS = [0.55, 0.42, 0.55]   # cm — oblique 3/4 view
IMG_RES_DEFAULT = 800


# ---------------------------------------------------------------------------
# Field registry: yt source field, the unit we sample/display in, an optional
# K->eV scale (temperatures), a plot title, and the transfer-function spec.
# Every field is sampled onto the uniform grid under its own key (in disp units)
# and cached; bounds/ranges below are all in disp units.  Bounds were tuned on
# dumps ~9-10 with headroom; because grids are cached, they can be re-tuned and
# re-rendered from cache in seconds.
# ---------------------------------------------------------------------------
def _emissive(bounds, cmap, alpha_lo, alpha_hi, nlayers=8, log=True):
    return dict(kind="emissive", bounds=bounds, log=log, cmap=cmap,
                alpha=(alpha_lo, alpha_hi), nlayers=nlayers)


# Signed B_x / B_y: symmetric red (+) / blue (-) Gaussians, zero band transparent.
def _signed_B_gaussians(scale):
    """Return (range, gaussians) for a signed B component peaking near +/- scale."""
    rng = (-1.7 * scale, 1.7 * scale)
    g = []
    for sgn, cols in ((+1, [(1.0, 0.70, 0.40), (0.95, 0.35, 0.20), (0.85, 0.10, 0.10)]),
                      (-1, [(0.55, 0.75, 1.0), (0.25, 0.45, 0.95), (0.10, 0.20, 0.90)])):
        for (frac, w, a), col in zip([(0.35, 0.18, 0.15), (0.75, 0.22, 0.40),
                                      (1.30, 0.30, 0.70)], cols):
            g.append((sgn * frac * scale, w * scale, a, col))
    return rng, g


_BX_RNG, _BX_G = _signed_B_gaussians(1.5e5)
_BY_RNG, _BY_G = _signed_B_gaussians(1.5e5)

FIELDS = {
    "ne": dict(
        yt=("gas", "El_number_density"), sample_unit="cm**-3", disp_unit="cm**-3",
        scale=1.0, title=r"Electron density  $n_e$  [cm$^{-3}$]",
        tf=_emissive((4e18, 5e20), "magma", 10 ** -1.7, 10 ** 0.1)),
    "te": dict(
        yt=("flash", "tele"), sample_unit="K", disp_unit="eV", scale=K_TO_EV,
        title=r"Electron temperature  $T_e$  [eV]",
        tf=_emissive((30.0, 3000.0), "afmhot", 10 ** -1.6, 10 ** 0.05)),
    "ti": dict(
        yt=("flash", "tion"), sample_unit="K", disp_unit="eV", scale=K_TO_EV,
        title=r"Ion temperature  $T_i$  [eV]",
        tf=_emissive((30.0, 5000.0), "plasma", 10 ** -1.6, 10 ** 0.05)),
    "bmag": dict(
        yt=("gas", "magnetic_field_magnitude"), sample_unit="G", disp_unit="G",
        scale=1.0, title=r"Magnetic field  $|B|$  [G]",
        tf=_emissive((2.7e5, 1.2e6), "viridis", 10 ** -1.6, 10 ** 0.05)),
    "bx": dict(
        yt=("gas", "magnetic_field_x"), sample_unit="G", disp_unit="G", scale=1.0,
        title=r"Magnetic field  $B_x$  [G]",
        tf=dict(kind="gaussians", range=_BX_RNG, gaussians=_BX_G)),
    "by": dict(
        yt=("gas", "magnetic_field_y"), sample_unit="G", disp_unit="G", scale=1.0,
        title=r"Magnetic field  $B_y$  [G]",
        tf=dict(kind="gaussians", range=_BY_RNG, gaussians=_BY_G)),
    "bz": dict(
        yt=("gas", "magnetic_field_z"), sample_unit="G", disp_unit="G", scale=1.0,
        title=r"Magnetic field  $B_z$  [G]",
        tf=dict(kind="gaussians", range=(-4e5, 1.3e6), gaussians=[
            # compression shell (red): main band ~3.5e5, strong peaks up to ~1e6 G
            (3.5e5, 5e4, 0.10, (1.0, 0.55, 0.30)),
            (5.5e5, 8e4, 0.28, (0.95, 0.25, 0.15)),
            (9.0e5, 1.5e5, 0.60, (0.80, 0.05, 0.05)),
            # field-expelled cavity (blue): Bz dips below ambient toward 0/negative
            (0.8e5, 5e4, 0.12, (0.55, 0.75, 1.0)),
            (-1.0e5, 8e4, 0.32, (0.25, 0.45, 0.95)),
            (-3.0e5, 8e4, 0.60, (0.10, 0.20, 0.90))])),
}
ALL_FIELDS = list(FIELDS.keys())


# ---------------------------------------------------------------------------
# Per-run presets (--preset). A volume rendering's camera and transfer functions
# are only meaningful for the run they were tuned on: the frame has to contain
# that run's plume at its *latest* time, and colour has to map to that run's
# ambient/peak values. A preset therefore bundles the camera + ycrop + the TF
# overrides for one run; anything it does not override falls back to FIELDS /
# the module camera constants above. Explicit CLI flags beat the preset.
# ---------------------------------------------------------------------------
_T2607_RNG, _T2607_G = _signed_B_gaussians(1.2e5)

PRESETS = {
    # FLASH_3D_noshield (Trantham 2026-03), ~5 ns, ambient |B| ~ 27 T.
    # The original tuning: all defaults, nothing overridden.
    "noshield": dict(camera=dict(width=0.55, campos=CAMPOS, focus=FOCUS, ycrop=0.005),
                     tf={}),

    # FLASH_MagShockZ3D-Trantham_2026-07: same 1.7x1.7x1.7 cm box but run 3x
    # longer (0-15.25 ns) and with a *weaker* ambient field (B_z = 7e4 G = 7 T
    # vs ~27 T), so both the framing and every B bound had to be re-tuned.
    #   camera : the blast fills the box and a jet reaches the y=1.6 cm top
    #            boundary by ~5 ns, so the frame is the whole domain.
    #   ycrop  : the solid target slab sits at y < 0.007 cm and bleeds into the
    #            first sampled cell above it, so crop deeper than the old 0.005.
    #   ne     : ambient 3.9e18, plume shell few e19, target plume core ~1e21.
    #   te/ti  : ambient ~11 eV, shocked/jet-head gas 1e2-2e3 eV.
    #   b*     : ambient B_z 7e4 G; compression to ~4e5 G, cavity down to ~0.
    "trantham2026-07": dict(
        camera=dict(width=2.0, campos=[1.7, 1.35, 1.7], focus=[0.0, 0.6, 0.0],
                    ycrop=0.02),
        tf=dict(
            ne=_emissive((6e18, 5e20), "magma", 10 ** -1.45, 10 ** 0.05),
            te=_emissive((25.0, 1500.0), "afmhot", 10 ** -1.6, 10 ** 0.05),
            ti=_emissive((25.0, 1200.0), "plasma", 10 ** -1.6, 10 ** 0.05),
            bmag=_emissive((9e4, 5e5), "viridis", 10 ** -1.6, 10 ** 0.05),
            bx=dict(kind="gaussians", range=_T2607_RNG, gaussians=_T2607_G),
            by=dict(kind="gaussians", range=_T2607_RNG, gaussians=_T2607_G),
            bz=dict(kind="gaussians", range=(-1.0e5, 6.0e5), gaussians=[
                # compression shell (red), above the 7e4 G ambient
                (1.5e5, 3.0e4, 0.16, (1.0, 0.55, 0.30)),
                (2.6e5, 5.0e4, 0.40, (0.95, 0.25, 0.15)),
                (4.0e5, 8.0e4, 0.75, (0.80, 0.05, 0.05)),
                # field-expelled cavity (blue): B_z falls from ambient to ~0
                (4.0e4, 1.5e4, 0.14, (0.55, 0.75, 1.0)),
                (1.5e4, 2.0e4, 0.32, (0.25, 0.45, 0.95)),
                (-2.0e4, 2.0e4, 0.60, (0.10, 0.20, 0.90))]),
        )),
}


def resolve_fields(preset):
    """FIELDS with the preset's transfer-function overrides applied (a copy)."""
    tf = PRESETS[preset]["tf"]
    return {k: (dict(v, tf=tf[k]) if k in tf else v) for k, v in FIELDS.items()}


# ---------------------------------------------------------------------------
# Extraction: AMR -> uniform grid (one array per field, in disp units)
# ---------------------------------------------------------------------------
def extract(path, res, fields):
    """Sample each requested field onto a res^3 uniform grid over the full box.

    Returns (data, bbox, t_ns) where data maps field key -> float32 array in the
    field's disp unit.

    The dataset is torn down before returning. This matters: a movie samples
    dozens of 20-40 GB dumps in one process, and without the teardown each dump
    leaves its AMR field data behind (measured: ~3 GB per dump, i.e. an OOM kill
    around dump ~30 of this run). Only the res^3 float32 arrays are kept.
    """
    ds = yt.load(path)
    ag = None
    try:
        le, re_ = ds.domain_left_edge, ds.domain_right_edge
        ag = ds.arbitrary_grid(le, re_, dims=[res, res, res])
        data = {}
        for f in fields:
            spec = FIELDS[f]
            arr = np.asarray(ag[spec["yt"]].to(spec["sample_unit"]), dtype=np.float32)
            if spec["scale"] != 1.0:
                arr = arr * np.float32(spec["scale"])
            data[f] = arr
            ag.clear_data()  # drop this field's AMR-sampled buffer before the next
        bbox = np.array([[float(le[i].to("cm")), float(re_[i].to("cm"))]
                         for i in range(3)])
        t_ns = float(ds.current_time.to("ns"))
    finally:
        if ag is not None:
            ag.clear_data()
        ds.index.clear_all_data()
        del ag, ds
        gc.collect()
    return data, bbox, t_ns


def load_grid(grid_path, need, res):
    """Load cached per-field arrays (bbox, t_ns) from grid_path for the fields in
    `need`; returns (data, bbox, t_ns, missing) where `missing` are fields absent
    from the cache (or None if the file/shape doesn't exist / match)."""
    if not os.path.exists(grid_path):
        return {}, None, None, list(need)
    g = np.load(grid_path)
    if "bbox" not in g:
        return {}, None, None, list(need)
    data = {f: g[f] for f in need if f in g}
    # guard against a res change: a stale cache at a different N must be re-sampled
    for f, a in list(data.items()):
        if a.shape != (res, res, res):
            del data[f]
    missing = [f for f in need if f not in data]
    return data, g["bbox"], float(g["t_ns"]), missing


def save_grid(grid_path, data, bbox, t_ns):
    """Merge `data` into any existing cache at grid_path and re-save (all fields)."""
    merged = {}
    if os.path.exists(grid_path):
        old = np.load(grid_path)
        for k in old.files:
            if k not in ("bbox", "t_ns"):
                merged[k] = old[k]
    merged.update(data)
    np.savez(grid_path, bbox=bbox, t_ns=t_ns, **merged)


def load_uds(data, bbox, ycrop):
    """Crop the basal slab (y < ycrop) and wrap the field arrays as a uniform grid."""
    any_arr = next(iter(data.values()))
    ny = any_arr.shape[1]
    y = np.linspace(bbox[1, 0], bbox[1, 1], ny)
    keep = y >= ycrop
    fdict = {}
    for f, arr in data.items():
        fdict[("gas", f)] = (np.ascontiguousarray(arr[:, keep, :]),
                             FIELDS[f]["disp_unit"])
    bb = bbox.copy()
    bb[1, 0], bb[1, 1] = y[keep][0], y[keep][-1]
    shape = next(iter(fdict.values()))[0].shape
    return yt.load_uniform_grid(fdict, shape, length_unit="cm", bbox=bb)


# ---------------------------------------------------------------------------
# Scene builder (fixed camera + per-field TF from the registry)
# ---------------------------------------------------------------------------
def _cam(sc, uds, view):
    """Point the scene's camera per `view` = (img_res, width, campos, focus)."""
    img_res, width, campos, focus = view
    cam = sc.camera
    cam.resolution = (img_res, img_res)
    cam.set_position(uds.arr(list(campos), "cm"), north_vector=NORTH)
    cam.focus = uds.arr(list(focus), "cm")
    cam.set_width(uds.quan(width, "cm"))


def build_scene(uds, field, view, spec=None):
    """Create a fixed-camera volume-render scene for `field` using its TF spec.

    `spec` defaults to the FIELDS entry; pass a resolve_fields() entry to render
    with a preset's transfer function instead.
    """
    spec = FIELDS[field] if spec is None else spec
    tfspec = spec["tf"]
    sc = yt.create_scene(uds, field=("gas", field))
    src = sc[0]
    if tfspec["kind"] == "emissive":
        lo, hi = tfspec["bounds"]
        src.set_log(tfspec["log"])
        dom = (np.log10(lo), np.log10(hi)) if tfspec["log"] else (lo, hi)
        tf = yt.ColorTransferFunction(dom)
        alpha = np.logspace(np.log10(tfspec["alpha"][0]),
                            np.log10(tfspec["alpha"][1]), tfspec["nlayers"])
        tf.add_layers(tfspec["nlayers"], w=0.01, colormap=tfspec["cmap"], alpha=alpha)
        src.tfh.tf = tf
        src.tfh.bounds = tfspec["bounds"]
    else:  # "gaussians"
        src.set_log(False)
        tf = yt.ColorTransferFunction(tfspec["range"])
        for c, w, a, col in tfspec["gaussians"]:
            tf.add_gaussian(c, w, list(col) + [a])
        src.tfh.tf = tf
        src.tfh.bounds = tfspec["range"]
    src.tfh.grey_opacity = False
    _cam(sc, uds, view)
    return sc


def save_frame(sc, fname, title, t_ns):
    txt = [[(0.03, 0.95), title, dict(color="white", fontsize=15)],
           [(0.03, 0.90), "t = %5.2f ns" % t_ns, dict(color="white", fontsize=13)]]
    sc.save_annotated(fname, sigma_clip=3.0, text_annotate=txt)


def encode(frame_dir, out_mp4, fps):
    """Assemble frame_*.png in frame_dir into out_mp4 at fps (H.264, yuv420p)."""
    cmd = ["ffmpeg", "-y", "-framerate", str(fps),
           "-pattern_type", "glob", "-i", os.path.join(frame_dir, "frame_*.png"),
           "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", out_mp4]
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--config", help="Path to FLASH analysis YAML config; the FLASH "
                     "directory comes from its flash_data_dir, or from the OSIRIS run "
                     "spec it names (sim_dir).")
    src.add_argument("--data-dir", dest="data_dir",
                     help="Directory of FLASH plot files, bypassing the config "
                          "(mutually exclusive with --config).")
    p.add_argument("--fields", nargs="+", default=["ne", "bz"], choices=ALL_FIELDS,
                   help="Which fields to render (default: ne bz). See module docstring.")
    p.add_argument("--stride", type=int, default=1, help="Dump stride (default 1).")
    p.add_argument("--t-start", type=int, default=0, dest="t_start")
    p.add_argument("--t-stop", type=int, default=None, dest="t_stop",
                   help="Last plot-file index inclusive (default: all).")
    p.add_argument("--grid-res", type=int, default=256, dest="grid_res",
                   help="Uniform-grid sampling resolution N (N^3 cells; default 256).")
    p.add_argument("--img-res", type=int, default=IMG_RES_DEFAULT, dest="img_res",
                   help="Rendered image size in pixels (default 800).")
    p.add_argument("--preset", default="noshield", choices=sorted(PRESETS),
                   help="Per-run camera + transfer-function tuning (default noshield, "
                        "the original FLASH_3D_noshield settings). See PRESETS.")
    p.add_argument("--ycrop", type=float, default=None,
                   help="Crop the basal slab below this y [cm] (default: the preset's).")
    p.add_argument("--width", type=float, default=None,
                   help="Camera frame width [cm] (default: the preset's).")
    p.add_argument("--campos", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
                   help="Camera position [cm] (default: the preset's).")
    p.add_argument("--focus", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
                   help="Camera focus point [cm] (default: the preset's).")
    p.add_argument("--fps", type=int, default=5, help="Movie frame rate (default 5).")
    p.add_argument("--output-dir", default=None, dest="output_dir")
    p.add_argument("--force", action="store_true",
                   help="Re-render frames even if the PNG already exists.")
    p.add_argument("--no-encode", action="store_true", dest="no_encode",
                   help="Render frames only; skip ffmpeg MP4 assembly.")
    p.add_argument("--grids-only", action="store_true", dest="grids_only",
                   help="Only sample+cache the uniform grids (the slow, AMR-reading "
                        "half); skip rendering and encoding. Use this to prime the "
                        "cache for a new run, then re-run to render once the camera "
                        "and transfer functions are tuned.")
    args = p.parse_args()

    # camera: explicit CLI flag wins, else the preset's tuned value
    cam = PRESETS[args.preset]["camera"]
    specs = resolve_fields(args.preset)
    for k in ("width", "campos", "focus", "ycrop"):
        if getattr(args, k) is None:
            setattr(args, k, cam[k])

    if args.data_dir:
        flash_dir = os.path.abspath(os.path.expanduser(args.data_dir))
    else:
        cfg = analysis_utils.load_config(args.config)
        flash_dir = flash_source.resolve(cfg, args.config).flash_dir

    all_files = find_plot_files(flash_dir)
    stop = len(all_files) if args.t_stop is None else min(args.t_stop + 1, len(all_files))
    indices = [i for i in range(args.t_start, stop, args.stride) if i < len(all_files)]
    if not indices:
        raise RuntimeError("No dumps selected from range/stride.")

    out_dir = args.output_dir or os.path.join(
        _HERE, "..", "results", os.path.basename(flash_dir.rstrip("/")), "movie3d")
    grids_dir = os.path.join(out_dir, "grids")
    os.makedirs(grids_dir, exist_ok=True)
    frame_dirs = {f: os.path.join(out_dir, f) for f in args.fields}
    for d in frame_dirs.values():
        os.makedirs(d, exist_ok=True)

    print(f"FLASH dir : {flash_dir}")
    print(f"Fields    : {args.fields}")
    print(f"Dumps     : {len(indices)}  (indices {indices[0]}..{indices[-1]} stride {args.stride})")
    print(f"Grid res  : {args.grid_res}^3   Image: {args.img_res}px   ycrop: {args.ycrop} cm")
    print(f"Preset    : {args.preset}")
    print(f"Camera    : pos {args.campos} focus {args.focus} width {args.width} cm")
    print(f"Output    : {out_dir}", flush=True)

    view = (args.img_res, args.width, args.campos, args.focus)

    for n, i in enumerate(indices):
        base = os.path.basename(all_files[i])
        frame_paths = {f: os.path.join(frame_dirs[f], f"frame_{i:04d}.png")
                       for f in args.fields}
        if args.grids_only:
            need = list(args.fields)
        else:
            need = [f for f in args.fields
                    if args.force or not os.path.exists(frame_paths[f])]
        if not need:
            print(f"[{n + 1:3d}/{len(indices)}] {base}  frames exist — skip", flush=True)
            continue

        grid_path = os.path.join(grids_dir, f"grid_{i:04d}.npz")
        data, bbox, t_ns, missing = load_grid(grid_path, need, args.grid_res)
        if missing:
            new, bbox, t_ns = extract(all_files[i], args.grid_res, missing)
            data.update(new)
            save_grid(grid_path, new, bbox, t_ns)
            src_msg = f"sampled {missing}" + (f" + cached {[f for f in need if f not in missing]}"
                                              if len(missing) < len(need) else "")
        else:
            src_msg = "cached grid"
        print(f"[{n + 1:3d}/{len(indices)}] {base}  ({src_msg}, t={t_ns:.2f} ns) "
              f"-> {'cache only' if args.grids_only else 'render ' + str(need)}", flush=True)
        if args.grids_only:
            continue

        uds = load_uds(data, bbox, args.ycrop)
        for f in need:
            save_frame(build_scene(uds, f, view, specs[f]),
                       frame_paths[f], specs[f]["title"], t_ns)

    if args.grids_only:
        print(f"\n--grids-only: grids cached in {grids_dir}; nothing rendered.")
        return
    if args.no_encode:
        print("\n--no-encode: frames only.")
        return

    for f in args.fields:
        mp4 = os.path.join(os.path.dirname(out_dir), f"flash3d_{f}.mp4")
        try:
            encode(frame_dirs[f], mp4, args.fps)
            print(f"Saved → {mp4}")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"  Warning: ffmpeg failed for {f} ({e}); frames are in {frame_dirs[f]}")


if __name__ == "__main__":
    main()
