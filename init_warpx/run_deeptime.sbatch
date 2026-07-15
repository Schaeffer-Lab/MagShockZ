#!/bin/bash
# =============================================================================
#  DEEP-TIME PRODUCTION run -- MagShockZ 2D hybrid-PIC on 24 GPU nodes (96 A100).
#  Config: runs/magshockz_2d_deeptime.warpx.yaml  (ppc=48, n_steps=60000 ~ 6 gyroperiods)
#
#  PREREQUISITE (run once on a login node):
#      conda run -n analysis python init_warpx/stage_prod.py
#  (stages the large box + deflates sqrt(4*pi) B into input_files/warpx/magshockz_2d_prod/;
#   shared with the 2-node production run -- stage only once). This job aborts if missing.
#
#  Efficiency note: at 1.45M cells / 96 GPUs (~15k cells/GPU) the field-substep loop is
#  comms-bound; the high ppc=48 (~10M particles/GPU) is what justifies 24 nodes. If a quick
#  scaling test shows poor efficiency, drop -N (and n_steps) accordingly.
# =============================================================================
#SBATCH -A m5032_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 06:00:00
#SBATCH -N 24
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH -c 32
#SBATCH -J f2wx_magshockz_deeptime
#SBATCH -o deeptime_slurm-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL

set -uo pipefail
source ~/perlmutter_gpu_warpx.profile
export MPICH_GPU_SUPPORT_ENABLED=1
export PYTHONPATH=/pscratch/sd/d/dschnei/flash2warpx:${PYTHONPATH:-}

REPO=/pscratch/sd/d/dschnei/MagShockZ
PROD=$REPO/input_files/warpx/magshockz_2d_prod
CONFIG=$REPO/runs/magshockz_2d_deeptime.warpx.yaml
cd "$REPO/init_warpx"

# --- PHASE-1 guard: refuse to burn 24-node hours if the slice was never staged/deflated ---
if [[ ! -f "$PROD/interp/By.npy" || ! -f "$PROD/interp/raw_B_inflated/DEFLATED_BY_SQRT4PI.txt" ]]; then
  echo "ERROR: staged+deflated data not found under $PROD/interp"
  echo "       run PHASE 1 first:  conda run -n analysis python init_warpx/stage_prod.py"
  exit 1
fi

mkdir -p logs
rm -f Backtrace.* 2>/dev/null
log="logs/deeptime_${SLURM_JOB_ID}.log"

# --- Auto-resume: if a checkpoint exists, restart from the newest one. Chaining 6 h jobs
# (re-submit this script) then completes the ~10 h run with no lost compute. A checkpoint is
# only "complete" once its Header is written, so require Header to avoid resuming a half-flushed
# dir from a killed job. Picks the highest step number, not mtime, to be robust.
CHKDIR="$PROD/diags_deeptime"
latest_chk=$(ls -d "$CHKDIR"/chk* 2>/dev/null | while read d; do [[ -f "$d/Header" ]] && echo "$d"; done | sort -V | tail -1)
if [[ -n "$latest_chk" ]]; then
  export F2WX_RESTART="$latest_chk"
  echo "=== RESUMING from checkpoint $latest_chk ==="
else
  echo "=== FRESH start (no valid checkpoint under $CHKDIR) ==="
fi

echo "=== DEEP-TIME run start $(date) : N=$SLURM_NNODES ranks=$((SLURM_NNODES*4)) ==="
srun -N24 -n96 -c32 -G96 --cpu-bind=cores python run_prod.py "$CONFIG" > "$log" 2>&1
rc=$?

last=$(grep -oE "STEP [0-9]+ starts" "$log" | grep -oE "[0-9]+" | tail -1)
surv=$(grep -c "PROD run COMPLETE" "$log")
nf=$(grep -c "Non-finite" "$log")
npf=$(ls -d "$PROD"/diags_deeptime/diag* 2>/dev/null | wc -l)
nchk=$(ls -d "$PROD"/diags_deeptime/chk* 2>/dev/null | wc -l)
newest_chk=$(ls -d "$PROD"/diags_deeptime/chk* 2>/dev/null | sort -V | tail -1)
echo "RESULT deeptime: exit=${rc} last_step=${last:-0} completed=${surv} nonfinite=${nf} plotfiles=${npf} checkpoints=${nchk} newest_chk=${newest_chk:-none}"
if [[ "$surv" -eq 0 && -n "$newest_chk" ]]; then
  echo "NOTE: run did not finish; re-submit this script to auto-resume from $newest_chk"
fi
grep -E "Auto dt|BCs:|PROD run:|thermal u_th" "$log" | head -5
echo "=== DEEP-TIME run end $(date) ==="
