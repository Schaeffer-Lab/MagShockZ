#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu&hbm40g
#SBATCH -G 4
#SBATCH -q regular
#SBATCH -J perlmutter_magshockz_rqm100_dx0.3_ppc500_g20.1d
#SBATCH --mail-user=dschneidinger@g.ucla.edu
#SBATCH --mail-type=ALL
#SBATCH -A m5032
#SBATCH -t 01:00:00


# Disable GPU-direct RDMA (fixes cxil_map write errors)
export MPICH_GPU_SUPPORT_ENABLED=0

# Alternative CXI settings if above doesn't work
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_MR_CACHE_MAX_COUNT=0


# OpenMP settings
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

# Increase stack size
ulimit -l unlimited
ulimit -s unlimited


# Run
cd ${SCRATCH}/MagShockZ/input_files/magshockz_rqm100_dx0.3_ppc500_g20.1d
conda activate osiris2

#run the application:
echo "Before sbcast"
sbcast -f /global/common/software/m5032/osiris/bin/osiris-1D-dev.e /tmp/osiris-1D-dev.e
sbcast -f magshockz_rqm100_dx0.3_ppc500_g20.1d /tmp/os-stdin.1d

echo "Before srun"
srun -n 4 -c 32 --cpu_bind=cores -G 4 --gpu-bind=single:1 /tmp/osiris-1D-dev.e /tmp/os-stdin.1d