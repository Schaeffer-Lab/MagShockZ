#!/bin/bash
# =============================================================================
#  2D heater-driven magnetized piston -- FULL electromagnetic PIC on Perlmutter GPU.
#  Deck: input_files/warpx/magshockz_2d_heater/inputs_2d_heater
#  Spec: runs/magshockz_2d_heater.warpx.yaml   (schema: heater_pic_2d)
#
#  NOT the flash2warpx hybrid pipeline. This job runs the compiled WarpX app with a
#  ParmParse text deck; there is no python driver and nothing is staged from FLASH.
#
#  PREREQUISITES
#   1. Build the 2D CUDA app ONCE (~20 min, login node is fine). Every pre-existing GPU
#      build in the warpx tree predates the ParticleHeater/TargetInjector commits, so
#      this is not optional:
#          bash init_warpx/build_warpx_gpu_2d.sh
#   2. Generate the deck (seconds, `analysis` env):
#          conda run -n analysis python init_warpx/gen_heater_deck.py \
#              --config runs/magshockz_2d_heater.warpx.yaml --smoke
#  This job aborts if either is missing.
#
#  SIZING (measured, not extrapolated). The FIRST production deck -- 568x2832 cells,
#  1.6e8 macroparticles, 35760 steps -- ran at 0.058 s/step on 2 nodes / 8 GPUs, i.e.
#  ~35 min end to end.
#
#  The CURRENT deck is 944x4720 at 4.5e8 macroparticles and 99326 steps, because kappa
#  was corrected to its measured 1.01 and v_piston_c dropped to 0.03 to keep the piston
#  electrons sub-relativistic (see the run spec). That is ~7.7x the work: ~4.4 h on
#  2 nodes, ~2.2 h on 4 -- hence -N 4 below, which also keeps memory per GPU sane at
#  5.6e7 macroparticles.
#
#  The `node_hours` figure in scaling_report.txt is CPU-calibrated (that is the only
#  timed reference point the cost model has) and overstates a GPU run by roughly an
#  order of magnitude. Trust the paragraph above for GPU planning.
#
#  Scale -N by cells, not by particles: 4.5M cells over 16 GPUs is ~280k cells/GPU, a
#  sensible working point. Going much finer per GPU makes the field solve comms-bound --
#  the lesson from the hybrid deep-time run at ~15k cells/GPU.
#
#  NULL CONTROL. To run the heater-off deck (identical in every other respect -- the
#  controlled test for whether the ambient <u^2> rise is numerical grid heating):
#      HEATER_DECK=$RUN/inputs_2d_heater_noheater sbatch init_warpx/run_heater_2d.sbatch
#  Give it its own directory too, or it will overwrite the physics run's diags:
#      HEATER_RUNDIR=<somewhere else>
#
#  CPU FALLBACK. The prebuilt CPU app also carries the operators; to use it instead:
#      HEATER_EXE=/pscratch/sd/d/dschnei/warpx/build_pm_cpu_py_2d/bin/warpx.2d \
#      sbatch -C cpu -A m5032 --ntasks-per-node=16 -c 16 init_warpx/run_heater_2d.sbatch
#  (that path needs MPICH_GPU_SUPPORT_ENABLED=0 -- see the env block below).
# =============================================================================
#SBATCH -A m5032_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 06:00:00
#SBATCH -N 4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH -c 32
#SBATCH --exclusive
# gpu-bind=none, with CUDA_VISIBLE_DEVICES set per rank inside srun below: ideally this
# would be --gpu-bind=single:1, but that hits a NERSC cgroups issue. Straight from
# warpx/Tools/machines/perlmutter-nersc/perlmutter_gpu.sbatch.
#SBATCH --gpu-bind=none
#SBATCH -J warpx_heater_2d
#SBATCH -o heater2d_slurm-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL

set -uo pipefail

REPO=/pscratch/sd/d/dschnei/MagShockZ
# A deck's own directory IS its run directory. Deriving RUN from HEATER_DECK rather than
# taking it as a second, independent variable is what stops an alternate deck (the
# heater-off null control) from writing its diag1*/chk* into the production run's tree at
# the same cadence and corrupting both -- a mistake that needs no env var to go wrong.
if [[ -n "${HEATER_DECK:-}" ]]; then
  DECK=$HEATER_DECK
  RUN=${HEATER_RUNDIR:-$(cd "$(dirname "$HEATER_DECK")" && pwd)}
else
  RUN=${HEATER_RUNDIR:-$REPO/input_files/warpx/magshockz_2d_heater}
  DECK=$RUN/inputs_2d_heater
fi
# The run spec's runtime: block, rendered to shell exports by gen_heater_deck.py. Sourced
# BEFORE the ${VAR:-default}s below so the spec chooses the binary, while this file keeps
# sole ownership of the NERSC launch logic -- the alternative, generating a run.sbatch per
# run directory, would fork that logic and leave stale copies behind.
[[ -f "$RUN/run_env.sh" ]] && source "$RUN/run_env.sh"
EXE=${HEATER_EXE:-/pscratch/sd/d/dschnei/warpx/build_pm_gpu_2d/bin/warpx.2d}
# Extra ParmParse overrides appended after the deck, e.g. HEATER_EXTRA_ARGS="max_step=20000".
# Word-split on purpose so several can be given.
read -r -a extra_args <<< "${HEATER_EXTRA_ARGS:-}"

# One GPU per rank, with GPU-aware MPI. The CPU build needs the opposite settings, so
# key off the binary rather than hardcoding -- picking the wrong pair either hangs in
# MPICH or silently disables device transport.
if [[ "$EXE" == *gpu* ]]; then
  source ~/perlmutter_gpu_warpx.profile
  export MPICH_GPU_SUPPORT_ENABLED=1
  export MPICH_OFI_NIC_POLICY=GPU
  export AMREX_DEFAULT_INIT="amrex.use_gpu_aware_mpi=1"
  export OMP_NUM_THREADS=16          # 32 virtual / 16 physical cores per rank
  use_gpu=1
else
  source ~/perlmutter_cpu_warpx.profile
  # Cray MPICH on CPU nodes: GPU transport off, and the RDMA host register / MR cache
  # tripped this build in the validated L0 launch.
  export MPICH_GPU_SUPPORT_ENABLED=0
  export FI_CXI_DISABLE_HOST_REGISTER=1
  export FI_MR_CACHE_MAX_COUNT=0
  export OMP_NUM_THREADS=8
  export OMP_PLACES=threads
  export OMP_PROC_BIND=spread
  use_gpu=0
fi
ulimit -s unlimited

if [[ ! -f "$DECK" ]]; then
  echo "ERROR: deck not found at $DECK"
  echo "       generate it first:  conda run -n analysis python \\"
  echo "         init_warpx/gen_heater_deck.py --config runs/magshockz_2d_heater.warpx.yaml"
  exit 1
fi
if [[ ! -x "$EXE" ]]; then
  echo "ERROR: WarpX 2D app not found/executable at $EXE"
  echo "       build it first:  bash init_warpx/build_warpx_gpu_2d.sh"
  exit 1
fi
# The heater and injector are ParmParse-only and silently no-op if the binary predates
# them, which would look like a successful but physics-free run.
#
# grep -c, not grep -q: `set -o pipefail` above turns a short-circuiting `grep -q` into a
# pipeline failure, because it SIGPIPEs `strings` partway through the 594 MB CUDA binary.
# That made a correct build look like it lacked the operator.
heater_hits=$(strings "$EXE" | grep -c particle_heater || true)
if [[ "${heater_hits:-0}" -eq 0 ]]; then
  echo "ERROR: $EXE contains no 'particle_heater' -- it predates the operator."
  echo "       Rebuild:  bash init_warpx/build_warpx_gpu_2d.sh"
  exit 1
fi

cd "$RUN"
mkdir -p logs
rm -f Backtrace.* 2>/dev/null
log="logs/heater2d_${SLURM_JOB_ID}.log"

# Auto-resume: chaining jobs (just re-submit) completes a longer run with no lost compute.
# The checkpoint diagnostic writes under diags/ like every other diag (chk.file_prefix =
# diags/chk), and its completion marker is WarpXHeader -- NOT the bare `Header` an AMReX
# plotfile carries. Requiring the wrong name silently rejected every valid checkpoint and
# restarted from step 0. sort -V picks the highest step rather than the newest mtime.
#
# A checkpoint is only resumable by a deck with the SAME grid. Every checkpoint records
# its `amr.n_cell` in warpx_job_info, so require it to match rather than trusting the
# directory to be from this deck -- changing v_piston_c or cell_size_de resizes the grid,
# and silently restarting a 944x4720 deck from a 568x2832 checkpoint is the failure this
# prevents.
deck_ncell=$(grep -oE '^amr\.n_cell *= *[0-9]+ +[0-9]+' "$DECK" \
  | grep -oE '[0-9]+ +[0-9]+' | tr -s ' ')
restart_args=()
latest_chk=""
while read -r d; do
  [[ -f "$d/WarpXHeader" ]] || continue
  chk_ncell=$(grep -oE '^amr\.n_cell *= *[0-9]+ +[0-9]+' "$d/warpx_job_info" 2>/dev/null \
    | grep -oE '[0-9]+ +[0-9]+' | tr -s ' ')
  if [[ "$chk_ncell" == "$deck_ncell" ]]; then
    latest_chk="$d"
  else
    echo "SKIPPING checkpoint $(basename "$d"): grid [$chk_ncell] != deck [$deck_ncell]"
  fi
done < <(ls -d "$RUN"/diags/chk* 2>/dev/null | sort -V)
if [[ -n "$latest_chk" ]]; then
  restart_args=(amr.restart="$latest_chk")
  echo "=== RESUMING from checkpoint $latest_chk ==="
else
  echo "=== FRESH start (no grid-compatible checkpoint under $RUN) ==="
fi

ranks=$((SLURM_NNODES * SLURM_NTASKS_PER_NODE))

# Chain-of-custody for the data this segment is about to write, stamped BEFORE srun so it
# survives a crash. PER JOB ID, not one provenance.txt: this run RESUMES across queue
# slots, and a single file would let the later segment overwrite the record of the binary
# that produced the earlier half of the data. `ls provenance_*.txt` reads as the history.
prov="provenance_${SLURM_JOB_ID}.txt"
warpx_src=$(cd "$(dirname "$EXE")/../.." 2>/dev/null && pwd || echo n/a)
{
  echo "job_id:         $SLURM_JOB_ID"
  echo "started:        $(date -Is)"
  echo "host:           $(hostname)"
  echo "nodes/ranks:    $SLURM_NNODES / $ranks"
  echo "exe:            $EXE"
  echo "exe_sha256:     $(sha256sum "$(readlink -f "$EXE")" | cut -d' ' -f1)"
  echo "heater_symbols: $heater_hits"
  echo "warpx_src:      $warpx_src"
  echo "warpx_git:      $(git -C "$warpx_src" rev-parse HEAD 2>/dev/null || echo n/a)"
  echo "warpx_dirty:    $(git -C "$warpx_src" status --porcelain 2>/dev/null | wc -l) file(s)"
  echo "project_git:    $(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo n/a)"
  echo "project_dirty:  $(git -C "$REPO" status --porcelain 2>/dev/null | wc -l) file(s)"
  echo "deck:           $DECK"
  echo "deck_sha256:    $(sha256sum "$DECK" | cut -d' ' -f1)"
  echo "spec_sha256:    $(sha256sum "$RUN/run.yaml" 2>/dev/null | cut -d' ' -f1 || echo n/a)"
  echo "grid:           $deck_ncell   max_step $(grep -oE '^max_step *= *[0-9]+' "$DECK" | grep -oE '[0-9]+')"
  echo "restart:        ${latest_chk:-(fresh)}"
  echo "extra_args:     ${HEATER_EXTRA_ARGS:-(none)}"
  nvidia-smi --query-gpu=index,name,memory.total,driver_version \
             --format=csv,noheader 2>/dev/null | sed 's/^/gpu:            /'
} > "$prov"

echo "=== heater2d start $(date) : N=$SLURM_NNODES ranks=$ranks exe=$(basename "$EXE") ==="
if [[ "$use_gpu" -eq 1 ]]; then
  # CUDA_VISIBLE_DEVICES must be set INSIDE srun -- SLURM_LOCALID only exists per task,
  # so setting it in the batch script would just abort under `set -u`. And the devices are
  # ordered INVERSE to local task IDs on Perlmutter (nvidia-smi topo -m), hence 3-LOCALID.
  # Both details are lifted from warpx/Tools/machines/perlmutter-nersc/perlmutter_gpu.sbatch.
  srun -n "$ranks" -c "$SLURM_CPUS_PER_TASK" --cpu-bind=cores bash -c \
    "export CUDA_VISIBLE_DEVICES=\$((3-SLURM_LOCALID)); exec \"\$@\"" \
    _ "$EXE" "$DECK" "${restart_args[@]}" "${extra_args[@]}" > "$log" 2>&1
else
  srun -n "$ranks" -c "$SLURM_CPUS_PER_TASK" --cpu-bind=cores \
       "$EXE" "$DECK" "${restart_args[@]}" "${extra_args[@]}" > "$log" 2>&1
fi
rc=$?

# One machine-readable summary line, as the flash2warpx sbatches emit.
last=$(grep -oE "STEP [0-9]+ ends" "$log" | grep -oE "[0-9]+" | tail -1)
nf=$(grep -ci "nan\|non-finite" "$log")
npf=$(ls -d "$RUN"/diags/diag1* 2>/dev/null | wc -l)
nchk=$(ls -d "$RUN"/diags/chk* 2>/dev/null | wc -l)
newest_chk=$(ls -d "$RUN"/diags/chk* 2>/dev/null | sort -V | tail -1)
heater_on=$(grep -c "particle_heater" "$log")
injector_on=$(grep -c "target_injector" "$log")
target_steps=$(grep -oE '^max_step *= *[0-9]+' "$DECK" | grep -oE '[0-9]+')
echo "RESULT heater2d: exit=${rc} last_step=${last:-0}/${target_steps} nonfinite=${nf}" \
     "plotfiles=${npf} checkpoints=${nchk} heater_lines=${heater_on}" \
     "injector_lines=${injector_on} newest_chk=${newest_chk:-none}"

{
  echo "finished:       $(date -Is)"
  echo "exit_code:      $rc"
  echo "last_step:      ${last:-0}/${target_steps}"
  echo "nonfinite:      ${nf}"
} >> "$prov"

# The operators must announce themselves; a run missing one it declared is physics-free.
# The null-control deck omits the heater ON PURPOSE, so key the expectation off the deck
# rather than warning unconditionally.
deck_wants_heater=$(grep -c '^particle_heater.species' "$DECK" || true)
if [[ "${deck_wants_heater:-0}" -gt 0 && "$heater_on" -eq 0 ]]; then
  echo "WARNING: the deck declares particle_heater.species but the heater never" \
       "reported -- the run is physics-free."
elif [[ "${deck_wants_heater:-0}" -eq 0 ]]; then
  echo "NOTE: heater-off NULL CONTROL run (deck declares no particle_heater.species)."
fi
if [[ "$injector_on" -eq 0 ]]; then
  echo "WARNING: injector never reported -- check target_injector.species."
fi
if [[ -n "$newest_chk" && "${last:-0}" -lt "${target_steps:-0}" ]]; then
  echo "NOTE: run did not reach max_step; re-submit this script to auto-resume."
fi
grep -E "particle_heater:|target_injector: step 0|Level 0: dt" "$log" | head -6
echo "=== heater2d end $(date) ==="
