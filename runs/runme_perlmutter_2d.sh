#!/bin/bash
# Python-init 2D OSIRIS setup.
# All run parameters live in the structured, version-controlled run spec
# runs/perlmutter_2d.run.yaml (the single source of truth). The FLASH->OSIRIS
# generator now lives in the standalone flash2osiris package (pip-installed into the
# osiris2 env); this wrapper just runs it against the spec. Outputs (deck + frozen
# run.yaml) land under input_files/ at the repo root.
set -euo pipefail

PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# osiris2 env python directly -- no `conda activate` needed (the yt plugin loads from
# ~/.config/yt/my_plugins.py via yt.enable_plugins()).
PY=/global/homes/d/dschnei/.conda/envs/osiris2/bin/python

cd "$PROJ"
$PY -m flash_osiris.generator --config runs/perlmutter_2d.run.yaml
