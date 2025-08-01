#!/bin/bash
# hard code in where your project is and where osiris is on your hard drive

set -e  # Exit immediately if a command exits with a non-zero status

# Use rm -i directly in the script to avoid unexpected behavior for users

NUM_NODES=2 # Default value for GPU 
DIMENSIONS=2
restart='false'
INPUTFILENAME=''
interactive='false'
PATHTOINPUTFILE=''
verbose='false'

# EDIT THESE PATHS AS NEEDED
OSIRISPATH="/home/${USER}/osiris"
PATHTOPROJECT="/home/${USER}/MagShockZ"

print_usage() {
  printf "Usage: %s [-r] [-i] [-n NUM_NODES] [-f INPUTFILENAME] [-d DIMENSIONS] [-v]\n" "$(basename "$0")"
  printf "  -r  Restart the simulation. Make sure that you call this script from the directory containing the restart files\n"
  printf "  -i  Run in interactive mode\n"
  printf "  -f  Specify the input file name\n"
  printf "  -n  Specify the number of nodes\n"
  printf "  -v  Enable verbose mode\n"
  printf "  -d  Specify the dimensions (1, 2, or 3)\n"
}

while getopts 'rin:f:vd:' flag; do
    case "${flag}" in
        r) restart='true';;
        i) interactive='true' ;;
        n) NUM_NODES="${OPTARG}" ;;
        f) 
            INPUTFILENAME=$(basename "${OPTARG}")
            PATHTOINPUTFILE="${OPTARG}"
            ;;
        v) verbose='true' ;;
        d) 
            if [[ "${OPTARG}" =~ ^[123]$ ]]; then
                DIMENSIONS="${OPTARG}"
            else
                echo "Error: DIMENSIONS must be 1, 2, or 3."
                exit 1
            fi
            ;;
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

if [ "$restart" = 'true' ]; then
    if [ "$interactive" = 'true' ]; then
        echo "Restarting simulation"
        mpirun -n ${NUM_NODES} ${OSIRISPATH}/bin/osiris-${DIMENSIONS}D.e -r input_file.txt > osiris_output_restart.log 2>&1 || { echo "Osiris simulation failed"; exit 1; }
    else
        echo "Restarting simulation"
        nohup mpirun -n ${NUM_NODES} ${OSIRISPATH}/bin/osiris-${DIMENSIONS}D.e -r input_file.txt > osiris_output_restart.log 2>&1 || { echo "Osiris simulation failed"; exit 1; } &
    fi
else
    if [ -d "${OUTPUTDIR}" ]; then
        rm -i -rf "${OUTPUTDIR}"/* || { echo "Failed to empty existing output directory"; exit 1; }
    else
        mkdir -p "${OUTPUTDIR}" || { echo "Failed to create output directory"; exit 1; }
    fi
    copy_files() {
        cp "${PATHTOINPUTFILE}" "${OUTPUTDIR}/input_file.txt" || { echo "Failed to copy input file"; exit 1; }
    }

    copy_files
    echo "Copying input file ${INPUTFILENAME}"

    cd "${OUTPUTDIR}" || { echo "Failed to change directory to OUTPUTDIR"; exit 1; }
    if [ "$interactive" = 'true' ]; then
        echo "Running simulation"
        mpirun -n ${NUM_NODES} ${OSIRISPATH}/bin/osiris-${DIMENSIONS}D.e input_file.txt || { echo "Osiris simulation failed"; exit 1; }
    else
        echo "Running simulation"
        nohup mpirun -n ${NUM_NODES} ${OSIRISPATH}/bin/osiris-${DIMENSIONS}D.e input_file.txt > osiris_output.log 2>&1 || { echo "Osiris simulation failed"; exit 1; } &
    fi
fi