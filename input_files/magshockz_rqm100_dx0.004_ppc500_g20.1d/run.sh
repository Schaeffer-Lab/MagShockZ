#!/bin/bash
#SBATCH -N 75
#SBATCH -C gpu&hbm40g
#SBATCH -G 300
#SBATCH -q regular
#SBATCH -J magshockz_rqm100_dx0.004_ppc500_g20.1d
#SBATCH --mail-user=dschneidinger@g.ucla.edu
#SBATCH --mail-type=ALL
#SBATCH -A m5032
#SBATCH -t 08:00:00

# Disable GPU-direct RDMA (fixes cxil_map write errors)
export MPICH_GPU_SUPPORT_ENABLED=0
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_MR_CACHE_MAX_COUNT=0

export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread
ulimit -l unlimited
ulimit -s unlimited

cd /pscratch/sd/d/dschnei/MagShockZ/input_files/magshockz_rqm100_dx0.004_ppc500_g20.1d

echo "Before sbcast"
sbcast -f /global/common/software/m5032/osiris/bin/osiris-1D-dev.e /tmp/osiris-1D-dev.e
sbcast -f magshockz_rqm100_dx0.004_ppc500_g20.1d /tmp/os-stdin.1d

echo "Before srun"
srun -n 300 -c 32 --cpu_bind=cores -G 300 --gpu-bind=single:1 /tmp/osiris-1D-dev.e /tmp/os-stdin.1d
