#! /bin/bash
# arg 1 is the absolute path to your input file
# hard code in where your project is and where osiris is on your hard drive

set -e  # Exit immediately if a command exits with a non-zero status
set -x  # Print commands and their arguments as they are executed

PATHTOINPUTFILE="$1"
INPUTFILENAME=$(basename "$PATHTOINPUTFILE")
OSIRISPATH="/home/${USER}/osiris"
PATHTOPROJECT="/home/${USER}/MagShockZ"
OUTPUTDIR="${PATHTOPROJECT}/simulations/raw_data/${INPUTFILENAME}"
NUM_NODES=16

# Validate input file
if [ ! -f "$PATHTOINPUTFILE" ]; then
    echo "Error: Input file does not exist or is not readable."
    exit 1
fi

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

if [ -d "${OUTPUTDIR}" ]; then
    rm -rf "${OUTPUTDIR}"/* || { echo "Failed to empty existing output directory"; exit 1; }
else
    mkdir -p "${OUTPUTDIR}" || { echo "Failed to create output directory"; exit 1; }
fi

cp py-script.py "${OUTPUTDIR}/py-script.py" || { echo "Failed to copy python script"; exit 1; }
cp -r interp/ "${OUTPUTDIR}/interp/" || { echo "Failed to copy interp directory"; exit 1; }
export PYTHONPATH=.

# Copy input file
cp "$PATHTOINPUTFILE" "${OUTPUTDIR}/input_file.txt" || { echo "Failed to copy input file"; exit 1; }
echo "Copying input file ${INPUTFILENAME}"

pushd "${OUTPUTDIR}" || { echo "Failed to change directory to OUTPUTDIR"; exit 1; }
mpirun -n ${NUM_NODES} ${OSIRISPATH}/bin/osiris-2D.e input_file.txt || { echo "Osiris simulation failed"; popd; exit 1; }

# rm input_file.txt || { echo "Failed to remove input file"; popd; exit 1; }
# rm py-script.py || { echo "Failed to remove python script"; popd; exit 1; }
# rm -rf interp/ || { echo "Failed to remove interp directory"; popd; exit 1; }

# popd
# echo "Done"
