#!/bin/bash
# Convergence run: identical to runme_perlmutter_1d.sh but with rqm_factor = 1
# (real ion mass ratio, no reduction). At rqm_factor = 1 the magnetization sigma
# converges to the physical value; M_A, beta_e and T_e/T_i remain conserved.
# All run parameters live in ../runs/perlmutter_1d_rqm1.run.yaml (single source of
# truth); this wrapper just runs the generator against it.

conda activate osiris2

python FLASH_OSIRIS_define.py --config ../runs/perlmutter_1d_rqm1.run.yaml
