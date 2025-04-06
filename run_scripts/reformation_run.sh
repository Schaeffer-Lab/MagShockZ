#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status

# Use rm -i directly in the script to avoid unexpected deletions 

NUM_NODES=16 # Default value for chablis
DIMENSIONS=1
RQM=100
PPC=100
interactive='false'
dx=0.1
OSIRISPATH="/home/${USER}/osiris"
PATHTOPROJECT="/home/${USER}/shock_reformation"

print_usage() {
  printf "Usage: %s [-i ION_THERMAL_VELOCITY] [-e ELECTRON_THERMAL_VELOCITY] [-f UPSTREAM_MAGNETIC_FIELD] [-v UPSTREAM_FLOW_VELOCITY]\n" "$(basename "$0")"
  printf "  -i Input thermal velocity of upstream ions in OSIRIS units\n"
  printf "  -e Input thermal velocity of upstream electrons in OSIRIS units\n"
  printf "  -b Input the strength of upstream field in OSIRIS units\n"
  printf "  -v Input flow velocity of upstream plasma in OSIRIS units"
}

while getopts 'i:e:b:v:' flag; do
    case "${flag}" in
        i) v_i="${OPTARG}" ;;
        e) v_e="${OPTARG}" ;;
        b) B_perp="${OPTARG}" ;;
        v) ufl="${OPTARG}" ;;
        *) print_usage
             exit 1 ;;
    esac
done

# Check if all required arguments are provided
if [ -z "${v_i}" ] || [ -z "${v_e}" ] || [ -z "${B_perp}" ] || [ -z "${ufl}" ]; then
    echo "Error: Missing required arguments."
    print_usage
    exit 1
fi

INPUTFILENAME=B${B_perp}_vi${v_i}_ve${v_e}_ufl${ufl}.1d

# Use math to derive some variables from given inputs
ion_gyrotime=$(echo "$RQM/$B_perp" | bc)
xmax=$(echo "20*$ion_gyrotime*$ufl" | bc)
echo "xmax = ${xmax}"
echo "ion_gyrotime = ${ion_gyrotime}"
### This is where you make and edit the input file




cat > $PATHTOPROJECT/$INPUTFILENAME <<EOF
!----------global simulation parameters----------
simulation 
{
  ! cuda algorithm breaks cathode
  ! algorithm = "cuda", 
}

!--------the node configuration for this simulation--------
node_conf 
{
 node_number = 16,
 if_periodic = .false.,
 n_threads = 2,
}

!----------spatial grid----------
grid
{
  nx_p = $(echo "scale=0 ; $xmax / $dx" | bc),
  coordinates = "cartesian",
}

!----------time step and global data dump timestep number----------
time_step 
{
  dt     = $(echo $dx - 0.01 | bc),
  ndump  =  2048, 
}

!----------restart information----------
restart
{
}

!----------spatial limits of the simulations----------
space 
{
  xmin =  0.0,
  xmax =  ${xmax},
  if_move= .false.,
}

!----------time limits ----------
time 
{
  tmin = 0.0d0, tmax  = $(echo "scale=0 ; $ion_gyrotime * 10" | bc),
}

!----------field solver set up----------

el_mag_fld 
{
  smooth_type = "none",
  ext_fld = "static",
  type_ext_b(1:3) = "math func", "math func", "math func",
  ext_b_mfunc(1) = "0.00",
  ext_b_mfunc(2) = "${B_perp}",
  ext_b_mfunc(3) = "0.0",
  
  type_ext_e(1:3) = "math func", "math func", "math func",
  ext_e_mfunc(1) = "0.0",
  ext_e_mfunc(2) = "0.0",
  ext_e_mfunc(3) = "$(echo "$ufl * $B_perp" | bc)",
}


!----------boundary conditions for em-fields ----------
emf_bound 
{
  type(1:2,1) =  "reflecting", "open",
}

!----------em-field diagnostics---------------
diag_emf
  {
    ndump_fac = 1,
    ndump_fac_ene_int = 1,
    reports = "part_e1", "part_e3", "part_b2",
  }

!----------number of particle species----------
particles
{ 
  num_cathode = 2,
  num_species = 0,
  interpolation = "quadratic",
}

cathode
{
 dir = 1,  ! direction to inject particles
 wall = 2, ! wall to inject particles from
 
 ! time profile information
 t_start = $(echo "$xmax / $ufl - 2" | bc),
 t_rise = 3.0, 
 t_flat = 1.0d9, 
 t_fall = 3.0,
 
 ! transverse profile information
 density = 1.0,
}

!----------information for ELECTRONS TOP ----------
species
{
  name = "electrons",
  num_par_max = 6000000,
  rqm=-1.0,
  num_par_x = $PPC,
  free_stream = .false.,
  !q_real = -1.0,
  !if_collide = .true.,
  !if_like_collide = .true.,
}



!----------information for ELECTRONS ----------

udist
{
  uth(1:3) = $v_e, $v_e, $v_e,
  ufl(1:3) = -$ufl, 0.00, 0.00,
}


!----------boundary conditions for ELECTRONS----------

spe_bound
{
  type(1:2,1) =   "reflecting", "open",
}

diag_species
{
 ndump_fac = 1, 
 reports = "charge",
 rep_udist = "ufl1",
 ndump_fac_ene = 1,
 ndump_fac_pha = 1, 
 ndump_fac_raw = 0,

 ps_xmin(1:1) =  0,
 ps_xmax(1:1) =  1875,
 ps_nx(1:1)   =  1024,
 ps_pmin(1:3) = -.5, -.5, -.5,
 ps_pmax(1:3) = .5,  .5,  .5,
 ps_np(1:3)   = 1024,  1024,  1024,
 if_ps_p_auto(1:3) = .false., .false., .false., 

 ps_gammamin = 1.0, 
 ps_gammamax = 1.001,
 ps_ngamma = 1024,
 if_ps_gamma_auto = .true.,

 !phasespaces = "p1x1", "p2x1",

 raw_gamma_limit = 0.0,
 raw_fraction = 1.0,
 raw_math_expr = "1.0",

}

cathode
{
 dir = 1,  ! direction to inject particles
 wall = 2, ! wall to inject particles from
 
 ! time profile information
 t_start = -9370.0,
 t_rise = 3.0,
 t_flat = 1.0d9,
 t_fall = 3.0,
 
 ! transverse profile information
 density = 1.0,
}

!----------information for IONS ----------
species
{
  name = "ions",
  num_par_max = 6000000,
  rqm=$RQM,
  num_par_x = $PPC,
  free_stream = .false.,
}


!----------information for species IONS ----------

udist
{
  uth(1:3) = $v_i, $v_i, $v_i,
  ufl(1:3) = -$ufl, 0.00, 0.00,
}

!----------boundary conditions for IONS ----------

spe_bound
{
  type(1:2,1) =   "reflecting", "open",
}



!----------diagnostic for IONS ----------

diag_species
{
 ndump_fac = 1, 
 reports = "charge",
 rep_udist = "ufl1",
 ndump_fac_ene = 1,
 ndump_fac_pha = 1, 
 ndump_fac_raw = 0,

 ps_xmin(1:1) =  0,
 ps_xmax(1:1) =  1875,
 ps_nx(1:1)   =  1024,
 ps_pmin(1:3) = -.5, -.5, -.5,
 ps_pmax(1:3) = .5,  .5,  .5,
 ps_np(1:3)   = 1024,  1024,  256,
 if_ps_p_auto(1:3) = .false., .false., .false., 

 ps_gammamin = 1.0, 
 ps_gammamax = 1.001,
 ps_ngamma = 1024,
 if_ps_gamma_auto = .true.,

 !phasespaces = "p1x1", "p2x1",

 raw_gamma_limit = 0.0,
 raw_fraction = 1.0,
 raw_math_expr = "1.0",

}
EOF

# Check if the input file was created successfully
if [ ! -f $INPUTFILENAME ]; then
    echo "Error: Failed to create input file."
    exit 1
fi




OUTPUTDIR="${PATHTOPROJECT}/raw_data/${INPUTFILENAME}"

if [ -d "${OUTPUTDIR}" ]; then
        rm -i -rf "${OUTPUTDIR}"/* || { echo "Failed to empty existing output directory"; exit 1; }
    else
        mkdir -p "${OUTPUTDIR}" || { echo "Failed to create output directory"; exit 1; }
fi
copy_files() {
    mv "${PATHTOPROJECT}/${INPUTFILENAME}" "${OUTPUTDIR}/." || { echo "Failed to copy input file"; exit 1; }
}

copy_files
echo "Copying input file ${INPUTFILENAME}"

cd "${OUTPUTDIR}" || { echo "Failed to change directory to OUTPUTDIR"; exit 1; }
if [ "$interactive" = 'true' ]; then
    echo "Running simulation"
    mpirun -n ${NUM_NODES} ${OSIRISPATH}/bin/osiris-${DIMENSIONS}D.e ${INPUTFILENAME} || { echo "Osiris simulation failed"; exit 1; }
else
    echo "Running simulation"
    nohup mpirun -n ${NUM_NODES} ${OSIRISPATH}/bin/osiris-${DIMENSIONS}D.e ${INPUTFILENAME} > osiris_output.log 2>&1 || { echo "Osiris simulation failed"; exit 1; } &
fi


