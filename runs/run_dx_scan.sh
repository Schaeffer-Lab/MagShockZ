#!/bin/bash
# =====================================================================
# dx parameter scan (1D), rqm_factor=100, ppc=500, 20 gyroperiods.
# Scans dx from 0.3 c/wpe down to the ion Debye length (~0.004 c/wpe,
# min along the lineout), 5 points, ~3x refinement per step.
#
# GPU count is MEMORY-bounded, not cell-bounded: ~11-13k cells/GPU (vs the
# old 2209). The particle COUNT is ~uniform along the box (uniform ppc, full
# occupancy), so the load is well balanced -- only the charge WEIGHT is
# concentrated at the dense target, and weight doesn't drive flops/memory.
# With balanced load, wall ~ 1/GPUs at ~constant node-hours, so using fewer
# GPUs just trades wall time for far smaller, more schedulable jobs. Node
# counts are the fewest that (a) fit ~11k cells/GPU in 40GB and (b) divide nx
# exactly. NOTE: 1D cost ~ 1/dx^2, so the fine runs are long even so -- expect
# multi-day wall at dx<=0.03, spanning several 24h jobs via restart
# (restart{} writes hourly checkpoints; resubmit with if_restart=.true.).
# node counts divide nx exactly for dx>=0.03; for dx 0.01/0.004 nx is not
# cleanly divisible (it wasn't for the old 120/300 either), so OSIRIS just
# spreads the <1-cell-per-rank remainder -- harmless. The lineout length isn't
# round so the actual dx differs from the nominal label in the 4th-5th sig
# fig. cells/GPU per point: 8836 / 13254 / 11045 / 11045 / 11045.
#
# Usage:  bash run_dx_scan.sh           # generate ALL 5 decks + sbatch
#         bash run_dx_scan.sh 0.01      # generate only the dx=0.01 point
# Run this on a node with enough memory for the FLASH covering grid
# (the generator builds a level-3 covering grid -- heavy). Then submit
# each run with:  sbatch input_files/<run>/run.sh
# =====================================================================
set -euo pipefail

# Call the osiris2 env python directly -- no `conda activate` needed (the yt plugin
# loads from ~/.config/yt/my_plugins.py via yt.enable_plugins()).
PY=/global/homes/d/dschnei/.conda/envs/osiris2/bin/python
# Absolute path to this runme, captured before the cd so we can copy it into
# each generated run folder as a provenance record of how the deck was made.
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
PROJ=/pscratch/sd/d/dschnei/MagShockZ
# Generate from the repo root so the package's outputs land in $PROJ/input_files/.
cd "$PROJ"

# n_ave grows gently with refinement (~doubles per decade in dx) so finer runs get a
# bit more diagnostic averaging without the window collapsing or n_ave blowing up.
# nominal_dx  exact_dx     node_number(=GPUs)   n_ave   walltime
CONFIGS=(
  "0.3    0.3         1     6     12:00:00"
  "0.1    0.1000033   2     9     24:00:00"
  "0.03   0.0300012   8     12    24:00:00"
  "0.01   0.0100004   24    18    24:00:00"
  "0.004  0.0040002   60    24    24:00:00"
)

ONLY="${1:-}"   # optional single nominal dx to (re)generate

# ---- arguments shared by every point of the scan ----
# These now come from the version-controlled run spec (single source of truth); the
# scan only overrides the per-point knobs (inputfile_name, dx, node_number, n_ave).
# species_names:{cham:al,targ:si} in the spec keeps the OSIRIS species named al/si.
BASE_CONFIG="$PROJ/runs/perlmutter_1d.run.yaml"

# ---- write the sbatch script into a generated run folder ----
write_sbatch() {
  local name="$1" node="$2" wall="$3"
  local nodes=$(( (node + 3) / 4 ))          # 4 GPUs per Perlmutter node
  local dir="${PROJ}/input_files/${name}.1d"
  cat > "${dir}/run.sh" <<EOF
#!/bin/bash
#SBATCH -N ${nodes}
#SBATCH -C gpu&hbm40g
#SBATCH -G ${node}
#SBATCH -q regular
#SBATCH -J ${name}.1d
#SBATCH --mail-user=dschneidinger@g.ucla.edu
#SBATCH --mail-type=ALL
#SBATCH -A m5032
#SBATCH -t ${wall}

# Disable GPU-direct RDMA (fixes cxil_map write errors)
export MPICH_GPU_SUPPORT_ENABLED=0
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_MR_CACHE_MAX_COUNT=0

export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread
ulimit -l unlimited
ulimit -s unlimited

cd ${dir}

# OSIRIS's embedded Python imports py-script-1d.py (init_type="python") -- it must be on
# PYTHONPATH; srun exports this to all ranks. The deck + binary are sbcast to /tmp, but
# the py-script stays here on shared scratch.
export PYTHONPATH=${dir}:\${PYTHONPATH:-}

echo "Before sbcast"
sbcast -f /global/common/software/m5032/osiris/bin/osiris-1D-dev.e /tmp/osiris-1D-dev.e
sbcast -f ${name}.1d /tmp/os-stdin.1d

echo "Before srun"
srun -n ${node} -c 32 --cpu_bind=cores -G ${node} --gpu-bind=single:1 /tmp/osiris-1D-dev.e /tmp/os-stdin.1d
EOF
  chmod +x "${dir}/run.sh"
  echo "  wrote ${dir}/run.sh  (-N ${nodes}, -G ${node}, srun -n ${node})"
}

# Each config is a fully independent process (separate output dir + interp/ cache),
# so deck generation is embarrassingly parallel. The covering grid is built at a
# fixed level (dx-independent) and its field cache is dropped between fields, so peak
# memory is ~the same few GB per process regardless of dx -- launching all configs at
# once fits comfortably in a compute node's RAM. Set JOBS=N to cap concurrency
# (JOBS=1 reproduces the old serial behavior); default 0 = no cap (all at once).
JOBS="${JOBS:-0}"

# Generate one deck end-to-end (python -> sbatch -> runme copy). Runs in the
# background; all output goes to a per-run log so parallel jobs don't interleave.
gen_one() {
  local nom="$1" dx="$2" node="$3" nave="$4" wall="$5"
  local name="magshockz_rqm100_dx${nom}_ppc500_g20"
  local dir="${PROJ}/input_files/${name}.1d"
  local log="/tmp/gen_${name}.log"
  echo "  START dx=${nom} (exact ${dx}), node_number=${node}, n_ave=${nave}, wall=${wall}  -> ${log}"
  {
    echo "=== dx=${nom} (exact ${dx}), node_number=${node}, n_ave=${nave}, wall=${wall} ==="
    $PY -m flash_osiris.generator \
      --config "${BASE_CONFIG}" \
      --inputfile_name "${name}" \
      --dx "${dx}" \
      --node_number "${node}" \
      --n_ave "${nave}"
    write_sbatch "${name}" "${node}" "${wall}"
    # Drop a copy of this runme next to the deck/manifest/run.sh so the directory
    # is a self-contained record of exactly how the deck was generated.
    cp "${SELF}" "${dir}/$(basename "${SELF}")"
    echo "  copied $(basename "${SELF}") into ${dir}/"
  } > "${log}" 2>&1
}

pids=()
for cfg in "${CONFIGS[@]}"; do
  read -r nom dx node nave wall <<<"$cfg"
  [[ -n "$ONLY" && "$ONLY" != "$nom" ]] && continue
  gen_one "$nom" "$dx" "$node" "$nave" "$wall" &
  pids+=("$!:magshockz_rqm100_dx${nom}_ppc500_g20")
  # Throttle to JOBS concurrent generations when a cap is set.
  if (( JOBS > 0 )); then
    while (( $(jobs -rp | wc -l) >= JOBS )); do wait -n; done
  fi
done

# Wait for every generation and surface any failures (with its log path).
fail=0
for entry in "${pids[@]}"; do
  pid="${entry%%:*}"; name="${entry#*:}"
  if wait "$pid"; then
    echo "  DONE  ${name}"
  else
    echo "  FAILED ${name}  -- see /tmp/gen_${name}.log"
    fail=1
  fi
done

echo
if (( fail )); then
  echo "One or more deck generations FAILED -- check the /tmp/gen_*.log files above."
  exit 1
fi
echo "Done. Submit each run with:  sbatch ${PROJ}/input_files/<run>.1d/run.sh"