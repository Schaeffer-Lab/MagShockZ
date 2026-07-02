#!/bin/bash
# Batch-render the FLASH 3D volume-rendering movies on a Perlmutter compute node.
#
# Sampling each 24-36 GB FLASH AMR dump onto a uniform grid is CPU/IO heavy
# (~4 min/dump), so this belongs on a compute node, NOT a shared login node.
# The Python script loops over dumps + fields and encodes the MP4s internally;
# it is resumable (cached grids + existing frames are reused), so re-submitting
# after a timeout continues where it left off.
#
# Edit CONFIG below if needed, then:  sbatch scripts/flash_3d_movie.sbatch
#
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH -c 64
#SBATCH -J flash_3d_movie
#SBATCH -A m5032
#SBATCH -t 08:00:00
#SBATCH --mail-user=dschneidinger@g.ucla.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate analysis

REPO=/pscratch/sd/d/dschnei/MagShockZ
cd "$REPO"

CONFIG=config/flash_3d_noshield.yaml

# All seven fields: electron density, e/i temperature, the three B components,
# and |B|.  ne/bz grids are already cached from the first run; the five new
# fields (te ti bx by bmag) are sampled and merged into each dump's cached grid,
# so only those are re-extracted.  Resumable — cached grids + existing frames
# are reused, so re-submit after a timeout to continue.
python scripts/flash_3d_movie.py --config "${CONFIG}" \
  --fields ne te ti bx by bz bmag --fps 5 --grid-res 256 --img-res 800
