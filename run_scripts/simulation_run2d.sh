#! /bin/bash
# arg 1 is the absolute path to your input file
# hard code in where your project is and where osiris is on your hard drive

PATHTOINPUTFILE=$1
INPUTFILENAME=$(basename $PATHTOINPUTFILE)
OSIRISPATH=/home/${USER}/osiris/osiris-1.0.0
PATHTOPROJECT=/home/${USER}/MagShockZ
OUTPUTDIR=${PATHTOPROJECT}/simulations/raw_data/${INPUTFILENAME}
NNODES=2


mkdir ${OUTPUTDIR}
cp $1 ${OSIRISPATH}/input_file.txt
echo "copying input file ${INPUTFILE}"

cd ${OSIRISPATH}
./config/docker/osiris mpirun -n ${NNODES} bin/osiris-2D.e input_file.txt

mv -f HIST/ MS/ TIMINGS/ run-info ${OUTPUTDIR}
rm input_file.txt

echo "Done"
