#! /bin/bash
# arg 1 is the absolute path to simulation directory
# hard code in where your project is and where osiris is on your hard drive

set -e  # Exit immediately if a command exits with a non-zero status
set -x  # Print commands and their arguments as they are executed

SIMULATIONDIR="$1"
INPUTFILENAME="input_file.txt"
OSIRISPATH="/home/${USER}/osiris"
NUM_NODES=16

# Activate the conda environment and set PYTHONPATH
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    CURRENT_ENV=$(conda info --envs | grep '*' | awk '{print $1}')
    if [ "$CURRENT_ENV" != "osiris" ]; then
        conda activate osiris || { echo "Failed to activate conda environment"; exit 1; }
    fi
else
    echo "Conda is not installed or not found in PATH"; exit 1;
fi
export PYTHONPATH=.


cd "${SIMULATIONDIR}" || { echo "Failed to change directory to OUTPUTDIR"; exit 1; }
nohup mpirun -n ${NUM_NODES} ${OSIRISPATH}/bin/osiris-2D.e -r ${INPUTFILENAME} > osiris_output_restart.log 2>&1 || { echo "Osiris simulation failed"; popd; exit 1; } & exit

