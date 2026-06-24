#!/bin/bash
# Batch-render OSIRIS movies non-interactively on Perlmutter.
#
# Batching lives HERE (shell loop over diagnostics), not in Python:
# scripts/make_movie.py stays single-purpose and is called once per diagnostic
# with --no-interactive so every knob comes from flags.
#
# Edit RUN and DIAGS below, then:  sbatch scripts/make_movie.sbatch
#
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -c 32
#SBATCH -J make_movie
#SBATCH -A m5032
#SBATCH -t 02:00:00
#SBATCH --mail-user=dschneidinger@g.ucla.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate analysis

REPO=/pscratch/sd/d/dschnei/MagShockZ
cd "$REPO"

# --- edit me -----------------------------------------------------------------
RUN=/pscratch/sd/d/dschnei/MagShockZ/input_files/magshockz_rqm100_dx0.1_ppc500_g20.1d
UNITS=ion              # electron (c/wpe, 1/wpe) or ion (d_i, T_ci); read from $RUN
SKIP=4                 # render every Nth dump

# One diagnostic directory per line (relative to $RUN).
DIAGS=(
  MS/FLD/b2-savg
  MS/DENSITY/e/charge-savg
)
# -----------------------------------------------------------------------------

for diag in "${DIAGS[@]}"; do
  name="${diag//\//_}"
  echo "=== ${diag} -> ${name}.mp4 ==="
  python scripts/make_movie.py \
    -d "${RUN}/${diag}" --no-interactive \
    --units "${UNITS}" -s "${SKIP}" -n 32 \
    -o "${name}"
done
