#!/bin/bash
# =====================================================================
# dx parameter scan (1D), rqm_factor=100, ppc=500, 20 gyroperiods.
# Scans dx from 0.3 c/wpe down to the ion Debye length (~0.004 c/wpe,
# min along the lineout), 5 points, ~3x refinement per step.
#
# GPUs are scaled in proportion to the cell count (constant 2209
# cells/GPU, matching the dx=0.3 run), so per-GPU memory and per-step
# cost stay fixed. NOTE: 1D cost ~ 1/dx^2, so wall time still grows ~1/dx
# even with this scaling -- the finest run (dx=0.004) needs 75 nodes and
# several hours. dx is pinned so nx = node_number * 2209 exactly (clean
# GPU decomposition); the lineout length isn't round so the actual dx
# differs from the nominal label in the 4th-5th sig fig.
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
cd "$(dirname "$0")"
PROJ=/pscratch/sd/d/dschnei/MagShockZ

# n_ave grows gently with refinement (~doubles per decade in dx) so finer runs get a
# bit more diagnostic averaging without the window collapsing or n_ave blowing up.
# nominal_dx  exact_dx     node_number(=GPUs)   n_ave   walltime
CONFIGS=(
  "0.3    0.3         4     6     00:30:00"
  "0.1    0.1000033   12    9     00:30:00"
  "0.03   0.0300012   40    12    01:00:00"
  "0.01   0.0100004   120   18    03:00:00"
  "0.004  0.0040002   300   24    08:00:00"
)

ONLY="${1:-}"   # optional single nominal dx to (re)generate

# ---- arguments shared by every point of the scan ----
common_args() {
  cat <<EOF
--data_path /pscratch/sd/d/dschnei/FLASH_3D_noshield/MagShockZ_hdf5_plt_cnt_0009
--dim 1
--reference_density 5e18
--rqm_factor 100
--ppc 500
--tmax_gyroperiods 20
--algorithm cuda
--start_point 0 0.07 0
--end_point 0 0.70 0
--num_threads 1
--n_dump_total 512
--restart false
--vpml_bnd_size 100
--emf_boundary pmc vpml
--part_boundary thermal thermal
--interpolation cubic
--smooth_type binomial
--smooth_order 2
--emf_reports b1 b2 b3 e1 e2 e3
--reports charge j1 j2 j3 q1 q2 q3
--rep_udist uth1 uth2 ufl1 ufl2
--phasespaces p1x1 p2x1 p3x1 gx1
--e_ps_pmin -1 -1 -0.5
--e_ps_pmax 1 1 0.5
--i_ps_pmin -0.1 -0.1 -0.05
--i_ps_pmax 0.1 0.1 0.05
--ps_np 4096 4096 4096
--ps_ngamma 128
--ps_gammamax 3.0
--ps_nx 1024
--ps_ny 512
EOF
}

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

echo "Before sbcast"
sbcast -f /global/common/software/m5032/osiris/bin/osiris-1D-dev.e /tmp/osiris-1D-dev.e
sbcast -f ${name}.1d /tmp/os-stdin.1d

echo "Before srun"
srun -n ${node} -c 32 --cpu_bind=cores -G ${node} --gpu-bind=single:1 /tmp/osiris-1D-dev.e /tmp/os-stdin.1d
EOF
  chmod +x "${dir}/run.sh"
  echo "  wrote ${dir}/run.sh  (-N ${nodes}, -G ${node}, srun -n ${node})"
}

for cfg in "${CONFIGS[@]}"; do
  read -r nom dx node nave wall <<<"$cfg"
  [[ -n "$ONLY" && "$ONLY" != "$nom" ]] && continue
  name="magshockz_rqm100_dx${nom}_ppc500_g20"
  echo "=== dx=${nom} (exact ${dx}), node_number=${node}, n_ave=${nave}, wall=${wall} ==="
  $PY FLASH_OSIRIS_define.py \
    --inputfile_name "${name}" \
    --dx "${dx}" \
    --node_number "${node}" \
    --n_ave "${nave}" \
    $(common_args)
  write_sbatch "${name}" "${node}" "${wall}"
done

echo
echo "Done. Submit each run with:  sbatch ${PROJ}/input_files/<run>.1d/run.sh"