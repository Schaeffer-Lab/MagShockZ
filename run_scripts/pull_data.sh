#!/bin/bash


SCRATCH="/gpfs/alpine2/fus158/scratch/dschneidinger/"

# Directories to move from the remote server
# List each directory as a full path
SIMULATION_DIRECTORIES=(
    "MS/"
    "HIST/"
    # Add more directories as needed
)

SIMULATION_FILES=(
    "run-info"
    # Add more files as needed
)

# Local directory to which the remote directories will be moved
LOCAL_DIR=$1

# Loop through each directory and copy it to the local destination
for dir in "${SIMULATION_OUTPUT[@]}"; do
    echo "Copying ${dir} from summit to ${LOCAL_DIR}"
    scp -r "summit:${SCRATCH}${dir}" "${LOCAL_DIR}/"
    
    # Uncomment the following line if you want to remove the directory from the remote server after copying
    # ssh summit rm -rf "${SCRATCH}${dir}"
done

# Loop through each file and copy it to the local destination
for file in "${SIMULATION_FILES[@]}"; do
    echo "Copying file ${file} from summit to ${LOCAL_DIR}"
    scp "summit:${SCRATCH}${file}" "${LOCAL_DIR}/"
done

echo "All directories and files have been moved."