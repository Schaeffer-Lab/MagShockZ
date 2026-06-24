#!/bin/bash
# Convergence run: identical to runme_perlmutter_1d.sh but with rqm_factor = 1
# (real ion mass ratio, no reduction). At rqm_factor = 1 the magnetization sigma
# converges to the physical value; M_A, beta_e and T_e/T_i remain conserved.
# All run parameters live in runs/perlmutter_1d_rqm1.run.yaml (single source of truth).
# The FLASH->OSIRIS generator now lives in the standalone flash2osiris package
# (pip-installed into the osiris2 env); this wrapper just runs it against the spec.
set -euo pipefail

PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# osiris2 env python directly -- no `conda activate` needed (the yt plugin loads from
# ~/.config/yt/my_plugins.py via yt.enable_plugins()).
PY=/global/homes/d/dschnei/.conda/envs/osiris2/bin/python

cd "$PROJ"
$PY -m flash_osiris.generator --config runs/perlmutter_1d_rqm1.run.yaml
