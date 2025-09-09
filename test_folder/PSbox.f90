!> @file
!!    Modulefile for handling of the Parallelization of the Simulation box of the Poisson Solver
!!
!! @author
!!    G. Fisicaro, L. Genovese (September 2015)
!!    Copyright (C) 2002-2015 BigDFT group
!!    This file is distributed under the terms of the
!!    GNU General Public License, see ~/COPYING file
!!    or http://www.gnu.org/copyleft/gpl.txt .
!!    For the list of contributors, see ~/AUTHORS
module PSbox
  use wrapper_MPI
  use PStypes, only: PSolver_energies, FFT_metadata
  use PSbase
  implicit none
  private

  interface PS_reduce
     module procedure reduce_scalar,reduce_array,reduce_energies
  end interface PS_reduce


  public :: PS_reduce,PS_gather,PS_scatter

contains


  !>gather a distributed array to have a full array
  !!if only src is present this is assumed to be a full array
  !!otherwise it is assumed to be a distributed array
  subroutine PS_gather(src,grid,mpi_env,dest,nsrc)
    use dynamic_memory, only: f_memcpy
    implicit none
    !> input array. If dest is present, the values are assumed to be distributed
    !!otherwise the values are not modified and are gathered in the dest
    !!array
    real(dp), dimension(*), intent(in) :: src
    type(FFT_metadata), intent(in) :: grid
    type(mpi_environment), intent(in) :: mpi_env
    real(dp), dimension(grid%m1,grid%m3,grid%m2,*), intent(out), optional :: dest
    integer, intent(in), optional :: nsrc !< number of copies of the array src (useful for spin-polarized)
    !local variables
    integer :: ispin,nspin,isrc

    nspin=1
    if (present(nsrc)) nspin=nsrc
    if (present(dest)) then
       if (mpi_env%nproc > 1) then
          isrc=1
          do ispin=1,nspin
             call fmpi_allgather(src(isrc),recvbuf=dest(1,1,1,ispin),&
               recvcounts=grid%counts,&
               displs=grid%displs,comm=mpi_env%mpi_comm)
             isrc=isrc+grid%m1*grid%m3*grid%n3p
          end do
       else
          call f_memcpy(n=grid%m1*grid%m2*grid%m3*nspin,src=src(1),dest=dest(1,1,1,1))
       end if
    else
       if (mpi_env%nproc > 1) then
          isrc=1
          do ispin=1,nspin
             call fmpi_allgather(src(isrc),recvcounts=grid%counts,&
                  displs=grid%displs,comm=mpi_env%mpi_comm)
             isrc=isrc+grid%m1*grid%m3*grid%n3p
          end do
       end if
    end if
  end subroutine PS_gather

  !> reduce all the given information
  !! MPI_SUM is applied in the case of unspecified op
  subroutine reduce_scalar(val,mpi_env,op)
    use f_enums
    implicit none
    real(dp), intent(inout) :: val
    type(mpi_environment), intent(in) :: mpi_env
    type(f_enumerator), intent(in), optional :: op !< operation to be done
    !local variables
    type(f_enumerator) :: mpi_op

    mpi_op=FMPI_SUM
    if (present(op)) mpi_op=op

    if (mpi_env%nproc > 1) &
         call fmpi_allreduce(val,1,op=mpi_op,comm=mpi_env%mpi_comm)

  end subroutine reduce_scalar

  subroutine reduce_array(val,mpi_env,op)
    use f_enums
    implicit none
    real(dp), dimension(:), intent(inout) :: val
    type(mpi_environment), intent(in) :: mpi_env
    type(f_enumerator), intent(in), optional :: op !< operation to be done
    !integer, intent(in), optional :: op !< operation to be done
    !local variables
    type(f_enumerator) :: mpi_op

    mpi_op=FMPI_SUM
    if (present(op)) mpi_op=op

    if (mpi_env%nproc > 1) &
         call fmpi_allreduce(val(1),size(val),op=mpi_op,comm=mpi_env%mpi_comm)

  end subroutine reduce_array

  !>this is of course to do the sum
  subroutine reduce_energies(e,mpi_env)
    type(PSolver_energies), intent(inout) :: e
    type(mpi_environment), intent(in) :: mpi_env
    !local variables
    integer, parameter :: energsize=10
    real(gp), dimension(energsize) :: vals

    if (mpi_env%nproc > 1) then
       vals(1)   =e%hartree
       vals(2)   =e%elec
       vals(3)   =e%eVextra
       vals(4)   =e%cavitation
       vals(5:10)=e%strten
       call PS_reduce(vals,mpi_env)
       e%hartree   =vals(1)
       e%elec      =vals(2)
       e%eVextra   =vals(3)
       e%cavitation=vals(4)
       e%strten    =vals(5:10)
    end if

  end subroutine reduce_energies

  !>the invers of gathering. Transforms a full array in a distributed one
  subroutine PS_scatter(src,dest,grid,mpi_env)
    implicit none
    type(FFT_metadata), intent(in) :: grid
    type(mpi_environment), intent(in) :: mpi_env
    real(dp), dimension(grid%m1,grid%m3,grid%m2), intent(in) :: src
    real(dp), dimension(grid%m1,grid%m3*grid%n3p), intent(out), optional :: dest

    call mpiscatterv(sendbuf=src,sendcounts=grid%counts,&
         displs=grid%displs,recvbuf=dest,comm=mpi_env%mpi_comm)

  end subroutine PS_scatter


end module PSbox
