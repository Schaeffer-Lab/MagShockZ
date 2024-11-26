#!/bin/bash
# arg 1 is the absolute path to simulation directory
# hard code in where your project is and where osiris is on your hard drive

set -e  # Exit immediately if a command exits with a non-zero status

alias rm="rm -i"
OSIRISPATH="/home/${USER}/osiris"
NUM_NODES=16
restart='false'
INPUTFILENAME=''
interactive='false'
PATHTOINPUTFILE=''
PATHTOPROJECT="/home/${USER}/MagShockZ"
verbose='false'

print_usage() {
  printf "Usage: %s [-r] [-i] [-n NUM_NODES] [-f INPUTFILENAME] [-v]\n" "$(basename "$0")"
  printf "  -r  Restart the simulation. Make sure that you call this script from the directory containing the restart files\n"
  printf "  -i  Run in interactive mode\n"
  printf "  -f  Specify the input file name\n"
  printf "  -n  Specify the number of nodes\n"
  printf "  -v  Enable verbose mode\n"
}

while getopts 'rin:f:v' flag; do
    case "${flag}" in
        r) restart='true';;
        i) interactive='true' ;;
        n) NUM_NODES="${OPTARG}" ;;
        f) INPUTFILENAME=$(basename "${OPTARG}") PATHTOINPUTFILE="${OPTARG}" ;;
        v) verbose='true' ;;
        *) print_usage
             exit 1 ;;
    esac
done

if [ -z "$INPUTFILENAME" ]; then
    echo "Error: Input file name must be specified with -f option."
    print_usage
    exit 1
fi

OUTPUTDIR="${PATHTOPROJECT}/simulations/raw_data/${INPUTFILENAME}"

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

if [ "$restart" = 'true' ]; then
    if [ "$interactive" = 'true' ]; then
        echo "Restarting simulation"
        mpirun -n ${NUM_NODES} ${OSIRISPATH}/bin/osiris-2D.e -r input_file.txt > osiris_output_restart.log 2>&1 || { echo "Osiris simulation failed"; exit 1; }
    else
        echo "Restarting simulation"
        nohup mpirun -n ${NUM_NODES} ${OSIRISPATH}/bin/osiris-2D.e -r input_file.txt > osiris_output_restart.log 2>&1 || { echo "Osiris simulation failed"; exit 1; } &
    fi
else
    if [ -d "${OUTPUTDIR}" ]; then
        rm -rf "${OUTPUTDIR}"/* || { echo "Failed to empty existing output directory"; exit 1; }
    else
        mkdir -p "${OUTPUTDIR}" || { echo "Failed to create output directory"; exit 1; }
    fi

    cp py-script.py "${OUTPUTDIR}/py-script.py" || { echo "Failed to copy python script"; exit 1; }
    cp -r interp/ "${OUTPUTDIR}/interp/" || { echo "Failed to copy interp directory"; exit 1; }

    # Copy input file
    cp "$PATHTOINPUTFILE" "${OUTPUTDIR}/input_file.txt" || { echo "Failed to copy input file"; exit 1; }
    echo "Copying input file ${INPUTFILENAME}"

    cd "${OUTPUTDIR}" || { echo "Failed to change directory to OUTPUTDIR"; exit 1; }
    if [ "$interactive" = 'true' ]; then
        echo "Running simulation"
        mpirun -n ${NUM_NODES} ${OSIRISPATH}/bin/osiris-2D.e input_file.txt || { echo "Osiris simulation failed"; exit 1; }
    else
        echo "Running simulation"
        nohup mpirun -n ${NUM_NODES} ${OSIRISPATH}/bin/osiris-2D.e input_file.txt > osiris_output.log 2>&1 || { echo "Osiris simulation failed"; exit 1; } &
    fi
fi

