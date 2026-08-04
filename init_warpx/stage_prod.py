"""PRODUCTION stage step (PHASE 1 of 2): extract the large-domain slice + deflate sqrt(4*pi) B.

Runs in a yt-capable env (NOT the WarpX venv), on a login node -- it is serial CPU work:

    conda run -n analysis python init_warpx/stage_prod.py

Reads the `extract:` block of runs/magshockz_2d_production.warpx.yaml, slices the FLASH dump
into input_files/warpx/magshockz_2d_prod/{interp,meta.yaml}, then divides interp/B{x,y,z}.npy by
sqrt(4*pi) in place (this dump stores B inflated by that factor). Pristine copies are kept in
interp/raw_B_inflated/ and a guard file prevents double-deflation, so re-running is safe.

After this succeeds, submit PHASE 2:  sbatch init_warpx/run_production.sbatch
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import yaml

from flash_warpx.extractor import extract_slice

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

REPO = Path("/pscratch/sd/d/dschnei/MagShockZ")
CONFIG = REPO / "runs" / "magshockz_2d_production.warpx.yaml"
OUTPUT_DIR = REPO / "input_files" / "warpx" / "magshockz_2d_prod"
FLASH_DUMP = "/pscratch/sd/d/dschnei/FLASH_3D_noshield/MagShockZ_hdf5_plt_cnt_0011"

SQRT4PI = float(np.sqrt(4.0 * np.pi))
B_COMPONENTS = ("Bx", "By", "Bz")


def deflate_bfield(interp: Path) -> None:
    """Divide interp/B{x,y,z}.npy by sqrt(4*pi) in place, once, with backup + guard."""
    backup = interp / "raw_B_inflated"
    guard = backup / "DEFLATED_BY_SQRT4PI.txt"
    if guard.exists():
        logging.info("B already deflated (guard %s present) -- skipping.", guard)
        return
    backup.mkdir(parents=True, exist_ok=True)
    for comp in B_COMPONENTS:
        f = interp / f"{comp}.npy"
        arr = np.load(f)
        np.save(backup / f"{comp}.npy", arr)          # pristine inflated copy
        np.save(f, arr / SQRT4PI)                       # deflated in place
        logging.info("deflated %s: |max| %.3f -> %.3f T", comp,
                     float(np.nanmax(np.abs(arr))), float(np.nanmax(np.abs(arr)) / SQRT4PI))
    guard.write_text(f"B{{x,y,z}}.npy divided by sqrt(4*pi)={SQRT4PI:.6f} once.\n")
    logging.info("wrote guard %s", guard)


def main() -> int:
    cfg = yaml.safe_load(CONFIG.read_text())
    ex = cfg["extract"]
    logging.info("=== PROD stage: extract box=%s level=%s -> %s ===",
                 ex["box_bounds_m"], ex["level"], OUTPUT_DIR)
    meta = extract_slice(
        flash_dump=FLASH_DUMP,
        output_dir=str(OUTPUT_DIR),
        normal_axis=ex["normal_axis"],
        slice_coord=ex["slice_coord_m"],
        level=int(ex["level"]),
        box_bounds=ex["box_bounds_m"],
        allow_extrapolation=bool(ex["allow_extrapolation"]),
        species_names=dict(ex["species_names"]),
    )
    logging.info("extracted shape=%s  d=(%.3e, %.3e) m  ne_mean=%.3e /m^3  Te_mean=%.1f eV",
                 meta["shape"], meta["d0_m"], meta["d1_m"],
                 meta["ne_mean_per_m3"], meta["Te_mean_eV"])
    deflate_bfield(OUTPUT_DIR / "interp")
    logging.info("=== PROD stage COMPLETE -> now: sbatch init_warpx/run_production.sbatch ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
