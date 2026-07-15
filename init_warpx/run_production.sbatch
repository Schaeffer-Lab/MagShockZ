#!/bin/bash
# =============================================================================
#  PRODUCTION run (PHASE 2) -- MagShockZ 2D hybrid-PIC, large domain + long time.
#  NOT the example/CI job. Config: runs/magshockz_2d_production.warpx.yaml
#
#  PREREQUISITE (run once on a login node before submitting):
#      conda run -n analysis python init_warpx/stage_prod.py
#  That extracts the large box + deflates the sqrt(4*pi) B into
#  input_files/warpx/magshockz_2d_prod/. This job aborts if that staging is missing.
#
#  Resources: 2 GPU nodes x 4 A100 = 8 MPI ranks. ~1.45M PIC cells, ppc=16, 5000 steps.
#  Rough cost ~1-2 h of compute; walltime is set generously for an overnight submit.
#  To go faster / longer, scale -N (and n_steps in the YAML) together.
# =============================================================================
#SBATCH -A m5032_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 06:00:00
#SBATCH -N 2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH -c 32
#SBATCH -J f2wx_magshockz_prod
#SBATCH -o prod_slurm-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL

set -uo pipefail
source ~/perlmutter_gpu_warpx.profile
export MPICH_GPU_SUPPORT_ENABLED=1
export PYTHONPATH=/pscratch/sd/d/dschnei/flash2warpx:${PYTHONPATH:-}

REPO=/pscratch/sd/d/dschnei/MagShockZ
PROD=$REPO/input_files/warpx/magshockz_2d_prod
cd "$REPO/init_warpx"

# --- PHASE-1 guard: refuse to burn GPU hours if the slice was never staged/deflated ---
if [[ ! -f "$PROD/interp/By.npy" || ! -f "$PROD/interp/raw_B_inflated/DEFLATED_BY_SQRT4PI.txt" ]]; then
  echo "ERROR: staged+deflated data not found under $PROD/interp"
  echo "       run PHASE 1 first:  conda run -n analysis python init_warpx/stage_prod.py"
  exit 1
fi

mkdir -p logs
rm -f Backtrace.* 2>/dev/null
log="logs/prod_${SLURM_JOB_ID}.log"

echo "=== PROD run start $(date) : N=$SLURM_NNODES ranks=$((SLURM_NNODES*4)) ==="
srun -N2 -n8 -c32 -G8 --cpu-bind=cores python run_prod.py > "$log" 2>&1
rc=$?

last=$(grep -oE "STEP [0-9]+ starts" "$log" | grep -oE "[0-9]+" | tail -1)
surv=$(grep -c "PROD run COMPLETE" "$log")
nf=$(grep -c "Non-finite" "$log")
npf=$(ls -d "$PROD"/diags/diag* 2>/dev/null | wc -l)
echo "RESULT prod: exit=${rc} last_step=${last:-0} completed=${surv} nonfinite=${nf} plotfiles=${npf}"
grep -E "Auto dt|BCs:|PROD run:|thermal u_th" "$log" | head -5
echo "=== PROD run end $(date) ==="
