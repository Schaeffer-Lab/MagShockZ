!!****if* source/Simulation/SimulationMain/MagShockZ2D/Simulation_init
!!
!! NAME
!!  Simulation_init
!!
!! SYNOPSIS
!!  Simulation_init()
!!
!! DESCRIPTION
!!  Reads the MagShockZ2D runtime parameters.
!!
!!***

subroutine Simulation_init()
  use Simulation_data
  use RuntimeParameters_interface, ONLY : RuntimeParameters_get

  implicit none

#include "constants.h"
#include "Flash.h"

  call RuntimeParameters_get('sim_MagField',           sim_MagField)

  call RuntimeParameters_get('sim_AblatorThickiness',  sim_AblatorThickiness)
  call RuntimeParameters_get('sim_PlasmaThickiness',   sim_PlasmaThickiness)

  call RuntimeParameters_get('sim_rhoTarg',  sim_rhoTarg)
  call RuntimeParameters_get('sim_teleTarg', sim_teleTarg)
  call RuntimeParameters_get('sim_tionTarg', sim_tionTarg)
  call RuntimeParameters_get('sim_tradTarg', sim_tradTarg)

  call RuntimeParameters_get('sim_rhoCham',  sim_rhoCham)
  call RuntimeParameters_get('sim_teleCham', sim_teleCham)
  call RuntimeParameters_get('sim_tionCham', sim_tionCham)
  call RuntimeParameters_get('sim_tradCham', sim_tradCham)

  call RuntimeParameters_get('sim_rhoVac',  sim_rhoVac)
  call RuntimeParameters_get('sim_teleVac', sim_teleVac)
  call RuntimeParameters_get('sim_tionVac', sim_tionVac)
  call RuntimeParameters_get('sim_tradVac', sim_tradVac)

  call RuntimeParameters_get('smallX', sim_smallX)

#ifdef FLASH_USM_MHD
  call RuntimeParameters_get('killdivb', sim_killdivb)
#endif

end subroutine Simulation_init
