!!****if* source/Simulation/SimulationMain/MagShockZ2D/IO_writeIntegralQuantities
!!
!! NAME
!!  IO_writeIntegralQuantities
!!
!! SYNOPSIS
!!  call IO_writeIntegralQuantities(integer(in) :: isFirst, real(in) :: simTime)
!!
!! DESCRIPTION
!!
!!  The stock FLASH version with six laser columns appended, so a run reports where
!!  the beam's energy landed without any post-processing:
!!
!!    E_laser_targ / E_laser_cham / E_laser_vac
!!        laser energy deposited in each material during this step, as
!!        sum(rho * depo * X_species * dvol).  DEPO is declared TYPE: PER_MASS by the
!!        EnergyDeposition Config, so that product is an energy, and weighting by the
!!        species mass fraction rather than thresholding it keeps the three columns
!!        adding up to the total across the mixed cells at the target surface.
!!
!!    Ecum_laser_targ / Ecum_laser_cham / Ecum_laser_vac
!!        the same, accumulated over the run.  This is the quantity to compare with
!!        "Energy in" in <basenm>_LaserEnergyProfile.dat: the difference between that
!!        column and the sum of these three is the energy the rays carried back out
!!        of the domain.
!!
!!  Two requirements for the cumulative columns to mean what they say:
!!
!!    io_integralFreq = 1        this routine must be called on every step, or the
!!                               steps it skips are simply missing from the sum.
!!    ed_depoReuseMaxSteps = -1  with deposition reuse on, DEPO holds the *previous*
!!                               step's specific energy and FLASH rescales only its
!!                               own energy counters by dt/prevDt, not the field, so
!!                               these sums would be off by that ratio.
!!
!! ARGUMENTS
!!  isFirst - 1 on the first call of a run (writes the header)
!!  simTime - simulation time
!!
!!***

subroutine IO_writeIntegralQuantities ( isFirst, simTime)

  use IO_data, ONLY : io_restart, io_statsFileName, io_globalComm
  use Grid_interface, ONLY : Grid_getListOfBlocks, &
    Grid_getBlkIndexLimits, Grid_getBlkPtr, Grid_getSingleCellVol, &
    Grid_releaseBlkPtr

  use IO_data, ONLY : io_globalMe, io_writeMscalarIntegrals

#include "Flash_mpi_implicitNone.fh"

#include "constants.h"
#include "Flash.h"

  real, intent(in) :: simTime
  integer, intent(in) :: isFirst

  integer :: lb, count

  integer :: funit = 99
  integer :: error
  integer :: nGlobalSumUsed, iSum

  integer :: blockList(MAXBLOCKS)
  integer :: blkLimits(HIGH, MDIM), blkLimitsGC(HIGH, MDIM)

#ifdef MAGP_VAR
  integer, parameter ::  nGlobalSumProp = 8
#else
  integer, parameter ::  nGlobalSumProp = 7
#endif
  ! Three per-step laser columns, then three cumulative ones.  They sit directly
  ! after the stock properties and before the mass scalars so the block that is
  ! written stays contiguous whether or not mass-scalar integrals are enabled.
  integer, parameter ::  nLaser = 6
  integer, parameter ::  iLaser = nGlobalSumProp
  integer, parameter ::  nGlobalSum = nGlobalSumProp + nLaser + NMASS_SCALARS
  real :: gsum(nGlobalSum)
  real :: lsum(nGlobalSum)

  ! Running totals of deposited energy per material, in erg.
  real, save :: cumLaser(3) = 0.0

  integer :: ivar
  integer :: i, j, k
  real :: dvol
  real, DIMENSION(:,:,:,:), POINTER :: solnData

  integer :: point(MDIM)
  integer :: ioStat

  if (io_writeMscalarIntegrals) then
     nGlobalSumUsed = nGlobalSum
  else
     nGlobalSumUsed = nGlobalSumProp + nLaser
  end if

  gsum(1:nGlobalSumUsed) = 0.
  lsum(1:nGlobalSumUsed) = 0.

  call Grid_getListOfBlocks(LEAF, blockList, count)

  do lb = 1, count
     call Grid_getBlkIndexLimits(blockList(lb), blkLimits, blkLimitsGC)
     call Grid_getBlkPtr(blockList(lb), solnData)

     do k = blkLimits(LOW,KAXIS), blkLimits(HIGH,KAXIS)
        do j = blkLimits(LOW,JAXIS), blkLimits(HIGH,JAXIS)
           do i = blkLimits(LOW,IAXIS), blkLimits(HIGH,IAXIS)

              point(IAXIS) = i
              point(JAXIS) = j
              point(KAXIS) = k

              call Grid_getSingleCellVol(blockList(lb), EXTERIOR, point, dvol)

#ifdef DENS_VAR
              lsum(1) = lsum(1) + solnData(DENS_VAR,i,j,k)*dvol
#endif

#ifdef DENS_VAR
#ifdef VELX_VAR
              lsum(2) = lsum(2) + solnData(DENS_VAR,i,j,k) * &
                   &                                solnData(VELX_VAR,i,j,k)*dvol
#endif
#ifdef VELY_VAR
              lsum(3) = lsum(3) + solnData(DENS_VAR,i,j,k) * &
                   &                                solnData(VELY_VAR,i,j,k)*dvol
#endif
#ifdef VELZ_VAR
              lsum(4) = lsum(4) + solnData(DENS_VAR,i,j,k) * &
                   &                                solnData(VELZ_VAR,i,j,k)*dvol
#endif

#ifdef ENER_VAR
              lsum(5) = lsum(5) + solnData(ENER_VAR,i,j,k) * &
                   &                                solnData(DENS_VAR,i,j,k)*dvol
#ifdef MAGP_VAR
              lsum(5) = lsum(5) + solnData(MAGP_VAR,i,j,k)*dvol
#endif
#endif

#ifdef VELX_VAR
#ifdef VELY_VAR
#ifdef VELZ_VAR
              lsum(6) = lsum(6) + 0.5*solnData(DENS_VAR,i,j,k) * &
                   &                             (solnData(VELX_VAR,i,j,k)**2+ &
                   &                              solnData(VELY_VAR,i,j,k)**2+ &
                   &                              solnData(VELZ_VAR,i,j,k)**2)*dvol
#endif
#endif
#endif

#ifdef EINT_VAR
              lsum(7) = lsum(7) + solnData(DENS_VAR,i,j,k) * &
                   &                                solnData(EINT_VAR,i,j,k)*dvol
#endif
#endif ! ifdef DENS_VAR

#ifdef MAGP_VAR
              lsum(8) = lsum(8) + solnData(MAGP_VAR,i,j,k)*dvol
#endif

              ! --- laser energy deposited this step, split by material ---
#ifdef DEPO_VAR
#ifdef DENS_VAR
#ifdef TARG_SPEC
              lsum(iLaser+1) = lsum(iLaser+1) + solnData(DENS_VAR,i,j,k) * &
                   solnData(DEPO_VAR,i,j,k) * solnData(TARG_SPEC,i,j,k) * dvol
#endif
#ifdef CHAM_SPEC
              lsum(iLaser+2) = lsum(iLaser+2) + solnData(DENS_VAR,i,j,k) * &
                   solnData(DEPO_VAR,i,j,k) * solnData(CHAM_SPEC,i,j,k) * dvol
#endif
#ifdef VAC_SPEC
              lsum(iLaser+3) = lsum(iLaser+3) + solnData(DENS_VAR,i,j,k) * &
                   solnData(DEPO_VAR,i,j,k) * solnData(VAC_SPEC,i,j,k) * dvol
#endif
#endif
#endif

#ifdef DENS_VAR
              if (io_writeMscalarIntegrals) then
                 iSum = nGlobalSumProp + nLaser
                 lsum(iSum+1:iSum+NMASS_SCALARS) = &
                      lsum(iSum+1:iSum+NMASS_SCALARS) + &
                        solnData(DENS_VAR,i,j,k) * &
                        solnData(MASS_SCALARS_BEGIN: &
                                 MASS_SCALARS_END,i,j,k)*dvol
              end if
#endif
           enddo
        enddo
     enddo
     call Grid_releaseBlkPtr(blockList(lb), solnData)

  enddo

  call MPI_Reduce (lsum, gsum, nGlobalSumUsed, FLASH_REAL, MPI_SUM, &
       &                MASTER_PE, io_globalComm, error)

  if (io_globalMe  == MASTER_PE) then

     ! The cumulative columns live only on the master, which is the only rank that
     ! sees the reduced per-step sums.
     cumLaser(1:3) = cumLaser(1:3) + gsum(iLaser+1:iLaser+3)

     ioStat = 0
     open(funit, file=trim(io_statsFileName), position='APPEND', status='OLD', iostat=ioStat)
     if(ioStat .NE. 0) then
        open(funit, file=trim(io_statsFileName), position='APPEND')
     endif

     if (isFirst .EQ. 1 .AND. (.NOT. io_restart .or. ioStat .NE. 0)) then

        ! The two branches are spelled out in full rather than switching one line
        ! with a preprocessor directive: a directive inside a free-form continued
        ! statement is not portable, since some compilers' cpp leaves a blank line
        ! behind to keep line numbers in step, which breaks the continuation.
#ifndef MAGP_VAR
        write (funit, 10)                  &
             '#time                     ', &
             'mass                      ', &
             'x-momentum                ', &
             'y-momentum                ', &
             'z-momentum                ', &
             'E_total                   ', &
             'E_kinetic                 ', &
             'E_internal                ', &
             'E_laser_targ              ', &
             'E_laser_cham              ', &
             'E_laser_vac               ', &
             'Ecum_laser_targ           ', &
             'Ecum_laser_cham           ', &
             'Ecum_laser_vac            ', &
             (msName(ivar),ivar=MASS_SCALARS_BEGIN,&
              min(MASS_SCALARS_END,&
                  MASS_SCALARS_BEGIN+nGlobalSumUsed-nGlobalSumProp-nLaser-1))
#else
        write (funit, 10)                  &
             '#time                     ', &
             'mass                      ', &
             'x-momentum                ', &
             'y-momentum                ', &
             'z-momentum                ', &
             'E_total                   ', &
             'E_kinetic                 ', &
             'E_internal                ', &
             'MagEnergy                 ', &
             'E_laser_targ              ', &
             'E_laser_cham              ', &
             'E_laser_vac               ', &
             'Ecum_laser_targ           ', &
             'Ecum_laser_cham           ', &
             'Ecum_laser_vac            ', &
             (msName(ivar),ivar=MASS_SCALARS_BEGIN,&
              min(MASS_SCALARS_END,&
                  MASS_SCALARS_BEGIN+nGlobalSumUsed-nGlobalSumProp-nLaser-1))
#endif

10         format (2x,50(a25, :, 1X))

     else if(isFirst .EQ. 1) then
        write (funit, 11)
11      format('# simulation restarted')
     endif

     ! Interleave the cumulative columns into the row in header order.
     write (funit, 12) simtime, gsum(1:nGlobalSumProp+3), cumLaser(1:3), &
          gsum(nGlobalSumProp+nLaser+1:nGlobalSumUsed)

12   format (1x, 50(es25.18, :, 1x))

     close (funit)

  endif

  call MPI_Barrier (io_globalComm, error)

  return

  contains
    character(len=25) function msName(ivar)
      integer,intent(in) :: ivar
      character(len=25) :: str
      call Simulation_mapIntToStr(ivar,str,MAPBLOCK_UNK)
      msName = str
    end function msName
end subroutine IO_writeIntegralQuantities
