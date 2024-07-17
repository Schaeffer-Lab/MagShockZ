#! /bin/bash
# arg 1 is the absolute path to your input file
# hard code in where your project is and where osiris is on your hard drive

PATHTOINPUTFILE="$1"
INPUTFILENAME=$(basename "$PATHTOINPUTFILE")
OSIRISPATH="/home/${USER}/osiris/osiris-1.0.0/"
PATHTOPROJECT="/home/${USER}/MagShockZ/"
OUTPUTDIR="${PATHTOPROJECT}/simulations/raw_data/${INPUTFILENAME}/"
NUM_NODES=1

# Validate input file
if [ ! -f "$PATHTOINPUTFILE" ]; then
    echo "Error: Input file does not exist or is not readable."
    exit 1
fi


mkdir -p "${OUTPUTDIR}" || { echo "Failed to create output directory"; exit 1; }

# Copy input file
cp "$PATHTOINPUTFILE" "${OSIRISPATH}/input_file.txt" || { echo "Failed to copy input file"; exit 1; }
echo "Copying input file ${INPUTFILENAME}"

pushd "${OSIRISPATH}" || { echo "Failed to change directory to OSIRISPATH"; exit 1; }
./config/docker/osiris mpirun -n ${NUM_NODES} bin/osiris-1D.e input_file.txt || { echo "Osiris simulation failed"; popd; exit 1; }


mv -f HIST/ MS/ TIMINGS/ run-info "${OUTPUTDIR}" || { echo "Failed to move simulation results"; popd; exit 1; }
rm input_file.txt || { echo "Failed to remove input file"; popd; exit 1; }


popd
echo "Done"
