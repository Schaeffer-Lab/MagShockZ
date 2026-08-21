!!****if* source/Simulation/SimulationMain/MagShockZ2D/Simulation_initBlock
!!
!! NAME
!!  Simulation_initBlock
!!
!! SYNOPSIS
!!  call Simulation_initBlock(integer(IN) :: blockID)
!!
!! DESCRIPTION
!!  Lays down the 2D MagShockZ initial state: a Si target slab on a stainless
!!  backing, an Al-foam ambient column above it, vacuum above that, threaded by a
!!  uniform out-of-plane field.  Stratified in y only — y is the laser
!!  propagation direction, so every ray sees the same layering on the way in and
!!  the beam itself is what breaks the symmetry.
!!
!! ARGUMENTS
!!  blockID - the block to initialize
!!
!!***

subroutine Simulation_initBlock(blockId)
  use Simulation_data
  use Grid_interface, ONLY : Grid_getBlkIndexLimits, Grid_getCellCoords, &
                             Grid_putPointData, Grid_getBlkPtr, &
                             Grid_releaseBlkPtr
  use RadTrans_interface, ONLY: RadTrans_mgdEFromT

  implicit none

#include "constants.h"
#include "Flash.h"

  integer, intent(in) :: blockId

  integer :: i, j, k, n
  integer :: blkLimits(2, MDIM)
  integer :: blkLimitsGC(2, MDIM)
  integer :: axis(MDIM)
  integer :: species
  real, allocatable :: xcent(:), ycent(:), zcent(:)
  real :: rho, tele, tion, trad, tradActual
  real, pointer, dimension(:,:,:,:) :: facexData, faceyData

#ifndef CHAM_SPEC
  integer :: CHAM_SPEC = 1, TARG_SPEC = 2, VAC_SPEC = 3
#endif

  call Grid_getBlkIndexLimits(blockId, blkLimits, blkLimitsGC)

  allocate(xcent(blkLimitsGC(HIGH, IAXIS)))
  call Grid_getCellCoords(IAXIS, blockId, CENTER, .true., &
       xcent, blkLimitsGC(HIGH, IAXIS))
  allocate(ycent(blkLimitsGC(HIGH, JAXIS)))
  call Grid_getCellCoords(JAXIS, blockId, CENTER, .true., &
       ycent, blkLimitsGC(HIGH, JAXIS))
  allocate(zcent(blkLimitsGC(HIGH, KAXIS)))
  call Grid_getCellCoords(KAXIS, blockId, CENTER, .true., &
       zcent, blkLimitsGC(HIGH, KAXIS))

  !----------------------------------------------------------------------------
  ! Cell-centered state.
  !----------------------------------------------------------------------------
  do k = blkLimits(LOW,KAXIS), blkLimits(HIGH,KAXIS)
     do j = blkLimits(LOW,JAXIS), blkLimits(HIGH,JAXIS)
        do i = blkLimits(LOW,IAXIS), blkLimits(HIGH,IAXIS)

           axis(IAXIS) = i
           axis(JAXIS) = j
           axis(KAXIS) = k

           if (ycent(j) < 0.0) then
              species = VAC_SPEC
           else if (ycent(j) < sim_AblatorThickiness) then
              species = TARG_SPEC
           else if (ycent(j) < sim_PlasmaThickiness) then
              species = CHAM_SPEC
           else
              species = VAC_SPEC
           end if

           if (species == TARG_SPEC) then
              rho = sim_rhoTarg; tele = sim_teleTarg
              tion = sim_tionTarg; trad = sim_tradTarg
           else if (species == CHAM_SPEC) then
              rho = sim_rhoCham; tele = sim_teleCham
              tion = sim_tionCham; trad = sim_tradCham
           else
              rho = sim_rhoVac; tele = sim_teleVac
              tion = sim_tionVac; trad = sim_tradVac
           end if

           call Grid_putPointData(blockId, CENTER, DENS_VAR, EXTERIOR, axis, rho)
           call Grid_putPointData(blockId, CENTER, TEMP_VAR, EXTERIOR, axis, tele)

           ! The field is perpendicular to the shock normal and to the simulation
           ! plane, so it lives entirely in the cell-centered z component; the
           ! in-plane components and their face values stay zero.
           call Grid_putPointData(blockId, CENTER, MAGX_VAR, EXTERIOR, axis, 0.0)
           call Grid_putPointData(blockId, CENTER, MAGY_VAR, EXTERIOR, axis, 0.0)
           call Grid_putPointData(blockId, CENTER, MAGZ_VAR, EXTERIOR, axis, sim_MagField)
           call Grid_putPointData(blockId, CENTER, MAGP_VAR, EXTERIOR, axis, &
                                  sim_MagField**2 / 2.0)

#ifdef FLASH_3T
           call Grid_putPointData(blockId, CENTER, TION_VAR, EXTERIOR, axis, tion)
           call Grid_putPointData(blockId, CENTER, TELE_VAR, EXTERIOR, axis, tele)
           call RadTrans_mgdEFromT(blockId, axis, trad, tradActual)
           call Grid_putPointData(blockId, CENTER, TRAD_VAR, EXTERIOR, axis, tradActual)
#endif

           if (NSPECIES > 0) then
              do n = SPECIES_BEGIN, SPECIES_END
                 if (n == species) then
                    call Grid_putPointData(blockID, CENTER, n, EXTERIOR, axis, &
                                           1.0e0 - (NSPECIES-1)*sim_smallX)
                 else
                    call Grid_putPointData(blockID, CENTER, n, EXTERIOR, axis, sim_smallX)
                 end if
              end do
           end if

        end do
     end do
  end do

  !----------------------------------------------------------------------------
  ! Face-centered field, guard cells included: the staggered-mesh solver takes
  ! its divergence-free initial condition from these, not from the cell-centered
  ! copy.  Both in-plane face components are zero because the field is out of
  ! plane, which is div-free by construction.
  !----------------------------------------------------------------------------
#if NFACE_VARS > 0
  if (sim_killdivb) then
     call Grid_getBlkPtr(blockID, facexData, FACEX)
     call Grid_getBlkPtr(blockID, faceyData, FACEY)
     facexData(MAG_FACE_VAR,:,:,:) = 0.0
     faceyData(MAG_FACE_VAR,:,:,:) = 0.0
     call Grid_releaseBlkPtr(blockID, facexData, FACEX)
     call Grid_releaseBlkPtr(blockID, faceyData, FACEY)
  end if
#endif

  deallocate(xcent)
  deallocate(ycent)
  deallocate(zcent)

  return

end subroutine Simulation_initBlock
