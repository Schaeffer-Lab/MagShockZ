!!****if* source/Simulation/SimulationMain/MagShockZ2D/Simulation_data
!!
!! NAME
!!  Simulation_data
!!
!! SYNOPSIS
!!  use Simulation_data
!!
!! DESCRIPTION
!!  Runtime parameters for the 2D MagShockZ laser-driven magnetized shock problem.
!!
!!***
module Simulation_data

  implicit none

#include "constants.h"

  !! *** Runtime Parameters *** !!

  real, save :: sim_AblatorThickiness
  real, save :: sim_PlasmaThickiness

  real, save :: sim_MagField

  real,    save :: sim_rhoTarg
  real,    save :: sim_teleTarg
  real,    save :: sim_tionTarg
  real,    save :: sim_tradTarg

  real,    save :: sim_rhoCham
  real,    save :: sim_teleCham
  real,    save :: sim_tionCham
  real,    save :: sim_tradCham

  real,    save :: sim_rhoVac
  real,    save :: sim_teleVac
  real,    save :: sim_tionVac
  real,    save :: sim_tradVac

  logical, save :: sim_killdivb = .FALSE.
  real,    save :: sim_smallX

end module Simulation_data
