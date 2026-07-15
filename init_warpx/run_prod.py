"""PRODUCTION run driver (PHASE 2 of 2): step the WarpX hybrid-PIC sim over the staged slice.

Runs in the WarpX GPU venv under srun (launched by init_warpx/run_production.sbatch). Reads the
`run:` block of runs/magshockz_2d_production.warpx.yaml into a HybridRunConfig and calls
build_and_run over the PRE-STAGED input_files/warpx/magshockz_2d_prod/{interp,meta.yaml}. It never
re-extracts; run stage_prod.py first (PHASE 1) if interp/ is missing.
"""
from __future__ import annotations

import dataclasses
import logging
import os
import sys
from pathlib import Path

import yaml

from flash_warpx.run import HybridRunConfig, build_and_run

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

REPO = Path("/pscratch/sd/d/dschnei/MagShockZ")
# Config is selectable so one driver serves both the 2-node production and the 24-node
# deep-time YAMLs (they share the same staged box). Priority: argv[1] > $F2WX_CONFIG > default.
CONFIG = Path(
    sys.argv[1] if len(sys.argv) > 1
    else os.environ.get("F2WX_CONFIG", REPO / "runs" / "magshockz_2d_production.warpx.yaml")
)
OUTPUT_DIR = REPO / "input_files" / "warpx" / "magshockz_2d_prod"


def main() -> int:
    interp = OUTPUT_DIR / "interp"
    if not (interp / "By.npy").exists():
        raise SystemExit(
            f"staged data missing under {interp}. Run PHASE 1 first:\n"
            f"    conda run -n analysis python init_warpx/stage_prod.py"
        )

    run_block = dict(yaml.safe_load(CONFIG.read_text())["run"])
    valid = {f.name for f in dataclasses.fields(HybridRunConfig)}
    unknown = set(run_block) - valid
    if unknown:
        logging.warning("ignoring unknown run keys not on HybridRunConfig: %s", sorted(unknown))
    kwargs = {k: v for k, v in run_block.items() if k in valid}

    # diag_dir in the YAML is relative to the run output dir; make it absolute.
    diag_dir = kwargs.get("diag_dir", "diags")
    if not Path(diag_dir).is_absolute():
        kwargs["diag_dir"] = str(OUTPUT_DIR / diag_dir)

    # Auto-resume: the sbatch finds the latest checkpoint and exports F2WX_RESTART so a
    # re-submitted job continues instead of restarting from t=0. Env wins over the YAML.
    restart = os.environ.get("F2WX_RESTART")
    if restart:
        kwargs["restart_from"] = restart
        logging.info("RESTART: resuming from checkpoint %s", restart)

    cfg = HybridRunConfig(**kwargs)
    logging.info("=== PROD run: n_steps=%d sim_dx_m=%s substeps=%d eta=%s ppc=%d -> %s ===",
                 cfg.n_steps, cfg.sim_dx_m, cfg.substeps, cfg.plasma_resistivity, cfg.ppc,
                 kwargs["diag_dir"])
    build_and_run(str(OUTPUT_DIR), cfg)
    logging.info("=== PROD run COMPLETE: survived %d steps ===", cfg.n_steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
