#!/bin/bash
# Python-init 1D OSIRIS setup -- first point of the parameter scan.
# All run parameters live in the structured, version-controlled run spec
# ../runs/perlmutter_1d.run.yaml (the single source of truth). This wrapper just
# runs the generator against it; the resolved spec is frozen to <run_dir>/run.yaml.

conda activate osiris2

python FLASH_OSIRIS_define.py --config ../runs/perlmutter_1d.run.yaml
