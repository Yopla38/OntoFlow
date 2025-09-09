"""
    ------------------------------------------
    Copyright: CEA Grenoble
    Auteur: Yoann CURE
    Entité: IRIG
    Année: 2025
    Description: Agent IA d'Intégration Continue
    ------------------------------------------
    """

def nettoyer_fortran(code: str, nom_fichier: str = "name.f90", commentaires_a_conserver=None) -> None:
    """
    Supprime les commentaires dans le code Fortran (tout ce qui suit un '!')
    sauf ceux commençant par un des préfixes spécifiés dans commentaires_a_conserver.

    :param code: Le code Fortran en tant que chaîne de caractères.
    :param nom_fichier: Le nom du fichier de sortie.
    :param commentaires_a_conserver: Liste des préfixes de commentaires à conserver.
    """
    if commentaires_a_conserver is None:
        commentaires_a_conserver = ["!$omp", "!$OMP", "!$"]

    lignes_nettoyees = []
    for ligne in code.splitlines():
        index_comment = ligne.find("!")
        if index_comment != -1:
            suffixe = ligne[index_comment:]
            if any(suffixe.startswith(prefix) for prefix in commentaires_a_conserver):
                # On garde toute la ligne (directive OpenMP)
                lignes_nettoyees.append(ligne.rstrip())
            else:
                # On coupe avant le commentaire
                ligne_sans_commentaire = ligne[:index_comment].rstrip()
                if ligne_sans_commentaire.strip():
                    lignes_nettoyees.append(ligne_sans_commentaire)
        else:
            if ligne.strip():
                lignes_nettoyees.append(ligne.rstrip())

    with open(nom_fichier, "w", encoding="utf-8") as f:
        f.write("\n".join(lignes_nettoyees) + "\n")


code = """subroutine G_PoissonSolver(iproc,nproc,planes_comm,iproc_inplane,inplane_comm,ncplx,&
     n1,n2,n3,nd1,nd2,nd3,md1,md2,md3,pot,zf,scal,mu0_square,mesh,offset,strten)
  use PSbase
  use wrapper_mpi
  use mpif_module, only: MPI_DOUBLE_PRECISION
  !use memory_profiling
  use time_profiling, only: f_timing
  use dynamic_memory
  use dictionaries ! for f_err_throw
  use box
  use f_perfs
  use f_utils, only: f_size,f_sizeof
  use at_domain, only: domain_geocode,domain_periodic_dims
  implicit none
  !to be preprocessed
  include 'perfdata.inc'
  !Arguments
  integer, intent(in) :: n1,n2,n3,nd1,nd2,md1,md2,md3
  integer, intent(inout) :: nd3
  integer, intent(in) :: ncplx,nproc,iproc
  integer, intent(in) :: planes_comm,inplane_comm,iproc_inplane
  real(gp), intent(in) :: scal,offset,mu0_square
  type(cell), intent(in) :: mesh
  real(dp), dimension(nd1,nd2,nd3/nproc), intent(in) :: pot
  real(dp), dimension(ncplx,md1,md3,md2/nproc), intent(inout) :: zf
  real(dp), dimension(6), intent(out) :: strten !< non-symmetric components of Ha stress tensor
  !Local variables
  character(len=*), parameter :: subname='G_Poisson_Solver'
  logical :: perx,pery,perz,halffty,cplx
  character(len=1) :: geocode
  !Maximum number of points for FFT (should be same number in fft3d routine)
  integer :: ncache,lzt,lot,nfft,ic1,ic2,ic3,Jp2stb,J2stb,Jp2stf,J2stf
  integer :: j2,j3,i1,i3,i,j,inzee,ierr,n1dim,n2dim,n3dim,ntrig, i_stat
  !integer :: i_all
  real(kind=8) :: twopion
  !work arrays for transpositions
  real(kind=8), dimension(:,:,:,:), allocatable :: zt
  real(kind=8), dimension(:,:,:), allocatable :: zt_t
  !work arrays for MPI
  real(kind=8), dimension(:,:,:,:,:), allocatable :: zmpi1
  real(kind=8), dimension(:,:,:,:), allocatable :: zmpi2
  !cache work array
  real(kind=8), dimension(:,:,:,:), allocatable :: zw
  !FFT work arrays
  real(kind=8), dimension(:,:), allocatable :: btrig1,btrig2,btrig3
  real(kind=8), dimension(:,:), allocatable :: ftrig1,ftrig2,ftrig3,cosinarr
  integer, dimension(7) :: after1,now1,before1
  integer, dimension(7) :: after2,now2,before2,after3,now3,before3
  real(gp), dimension(6) :: strten_omp
  !integer :: ncount0,ncount1,ncount_max,ncount_rate
  logical, dimension(3) :: peri
  integer :: maxIter, nthreadx
  integer :: n3pr1,n3pr2,j1start,n1p,n2dimp
  integer :: ithread, nthread
  integer,parameter :: max_memory_zt = 500 !< maximal memory for zt array, in megabytes
  !integer(f_long) :: readb,writeb
  real(f_double) :: gflops_fft
  type(f_perf) :: performance_info
  ! OpenMP variables
  !$ integer :: omp_get_thread_num, omp_get_max_threads !, omp_get_num_threads

  call f_routine(id='G_PoissonSolver')

  !Initialize stress tensor no matter of the BC
  !call to_zero(6,strten(1))

  strten=0.d0

  !conditions for periodicity in the three directions
  !perx=(geocode /= 'F' .and. geocode /= 'W' .and. geocode /= 'H')
  geocode=domain_geocode(mesh%dom)
  peri=domain_periodic_dims(mesh%dom)
  perx=peri(1)
  pery=peri(2)
  perz=peri(3)
!!$  perx=(geocode == 'P' .or. geocode == 'S')
!!$  pery=(geocode == 'P')
!!$  perz=(geocode /= 'F' .and. geocode /= 'H')

  cplx= (ncplx == 2)

  !also for complex input this should be eliminated
  halffty=.not. pery .and. .not. cplx

  !defining work arrays dimensions
  ncache=ncache_optimal

  !n3/2 if the dimension is real and isolated
  if (halffty) then
     n3dim=n3/2
  else
     n3dim=n3
  end if

  if (perz) then
     n2dim=n2
  else
     n2dim=n2/2
  end if

  if (perx) then
     n1dim=n1
  else
     n1dim=n1/2
  end if
  call f_timing(TCAT_PSOLV_COMPUT,'ON')
  ! check input
!!$  !these checks can be moved at the creation
!!$  if (mod(n1,2) /= 0 .and. .not. perx) stop 'Parallel convolution:ERROR:n1' !this can be avoided
!!$  if (mod(n2,2) /= 0 .and. .not. perz) stop 'Parallel convolution:ERROR:n2' !this can be avoided
!!$  if (mod(n3,2) /= 0 .and. .not. pery) stop 'Parallel convolution:ERROR:n3' !this can be avoided
!!$  if (nd1 < n1/2+1) stop 'Parallel convolution:ERROR:nd1'
!!$  if (nd2 < n2/2+1) stop 'Parallel convolution:ERROR:nd2'
!!$  if (nd3 < n3/2+1) stop 'Parallel convolution:ERROR:nd3'
  if (md1 < n1dim) stop 'Parallel convolution:ERROR:md1'
  if (md2 < n2dim) stop 'Parallel convolution:ERROR:md2'
  if (md3 < n3dim) stop 'Parallel convolution:ERROR:md3'
!!$  if (mod(nd3,nproc) /= 0) stop 'Parallel convolution:ERROR:nd3'
!!$  if (mod(md2,nproc) /= 0) stop 'Parallel convolution:ERROR:md2'

  if (ncache <= max(n1,n2,n3dim)*4) ncache=max(n1,n2,n3dim)*4

  if (timing_flag == 1 .and. iproc ==0) print *,'parallel ncache=',ncache


  ntrig=max(n3dim,n1,n2)

  n1p=n1

  if (nproc>2*(n3/2+1)-1 .and. .false.) then
    n3pr1=nproc/(n3/2+1)
    n3pr2=n3/2+1
    if (mod(n1,n3pr1) .ne. 0) n1p=((n1/n3pr1)+1)*n3pr1
  else
    n3pr1=1
    n3pr2=nproc
  endif

  if (n3pr1>1) then
   lzt=(md2/nproc)*n3pr1*n3pr2
   n2dimp=lzt
  else
   lzt=n2dim
   if (mod(n2dim,2) == 0) lzt=lzt+1
   if (mod(n2dim,4) == 0) lzt=lzt+1 !maybe this is useless
   n2dimp=n2dim
  endif

  !if (iproc==0) print*,'psolver param',md1,md2,md3,nd1,nd2,nd3,n1dim,n2dim,n3dim,n1,n2,n3

  if (lzt < n2dim) then
    if (iproc==0) print*,'PSolver error: lzt < n2dim',lzt,n2dim
    call mpi_finalize(i_stat)
    stop
  endif
  !if (iproc==0) print*,'lzt n2dim',lzt,n2dim


  !Allocations
  btrig1 = f_malloc((/ 2, ntrig /),id='btrig1')
  ftrig1 = f_malloc((/ 2, ntrig /),id='ftrig1')
  btrig2 = f_malloc((/ 2, ntrig /),id='btrig2')
  ftrig2 = f_malloc((/ 2, ntrig /),id='ftrig2')
  btrig3 = f_malloc((/ 2, ntrig /),id='btrig3')
  ftrig3 = f_malloc((/ 2, ntrig /),id='ftrig3')
  !allocate(zw(2,ncache/4,2+ndebug),stat=i_stat)
  !call memocc(i_stat,zw,'zw',subname)
  !allocate(zt(2,lzt,n1+ndebug),stat=i_stat)
  !call memocc(i_stat,zt,'zt',subname)
  zmpi2 = f_malloc((/ 2, n1, md2/nproc, nd3 /),id='zmpi2')
  !also for complex input this should be eliminated
  if (halffty) then
     cosinarr = f_malloc((/ 2, n3/2 /),id='cosinarr')
  end if
  if (nproc > 1) then
     zmpi1 = f_malloc((/ 2, n1, md2/nproc, nd3/nproc, n3pr2 /),id='zmpi1')
  end if

  !calculating the FFT work arrays (beware on the HalFFT in n3 dimension)

  !$omp parallel sections default(shared)
  !$omp section
    call ctrig_sg(n3dim,ntrig,btrig3,after3,before3,now3,1,ic3)
    do j = 1, n3dim
      ftrig3(1, j) = btrig3(1, j)
      ftrig3(2, j) = -btrig3(2, j)
    enddo
  !$omp section
    call ctrig_sg(n1,ntrig,btrig1,after1,before1,now1,1,ic1)
    do j = 1, n1
      ftrig1(1, j) = btrig1(1, j)
      ftrig1(2, j) = -btrig1(2, j)
    enddo
  !$omp section
    call ctrig_sg(n2,ntrig,btrig2,after2,before2,now2,1,ic2)
    do j = 1, n2
      ftrig2(1, j) = btrig2(1, j)
      ftrig2(2, j) = -btrig2(2, j)
    enddo
  !$omp section
    if (halffty) then
      !Calculating array of phases for HalFFT decoding
      twopion=8.d0*datan(1.d0)/real(n3,kind=8)
      do i3=1,n3/2
        cosinarr(1,i3)= dcos(twopion*real(i3-1,kind=8))
        cosinarr(2,i3)=-dsin(twopion*real(i3-1,kind=8))
      end do
    end if
  !$omp end parallel sections

  ! transform along z axis
  lot=ncache/(4*n3dim)
  if (lot < 1) then
     write(6,*) &
          'convolxc_off:ncache has to be enlarged to be able to hold at' // &
          'least one 1-d FFT of this size even though this will' // &
          'reduce the performance for shorter transform lengths'
     stop
  endif


  !put to zero the zw array
  !this should not be done at each time since the
  !array is refilled always the same way
  !zw=0.0_dp
  !call razero(4*(ncache/4),zw)
  !different loop if halfft or not (output part)

  maxIter = min(md2 /nproc, n2dim - iproc *(md2 /nproc))

  if (n3pr1 > 1) zt_t = f_malloc((/ 2, lzt/n3pr1, n1p /),id='zt_t')

  ! Manually limit the number of threads such the memory requirements remain small
  nthread = max(1,1000000*max_memory_zt/(8*2*(lzt/n3pr1)*n1p))
  nthreadx = 1
  !$ nthreadx = omp_get_max_threads()
  nthread = min(nthread,nthreadx)
  !write(*,*) 'nthread', nthread
  zw = f_malloc((/ 1.to.2,1.to.ncache/4,1.to.2,0.to.nthread-1 /),id='zw')
  zt = f_malloc((/ 1.to.2,1.to.lzt/n3pr1,1.to.n1p,0.to.nthread-1 /),id='zt')

  ithread = 0
  !$omp parallel num_threads(nthread) &
  !$omp default(shared)&
  !$omp private(nfft,inzee,Jp2stb,J2stb,Jp2stf,J2stf,i3,strten_omp)&
  !$omp private(j2,i1,i,j3,j) &
  !$omp firstprivate(lot, maxIter,ithread)
  !$ ithread = omp_get_thread_num()
!  !$omp firstprivate(before3, now3, after3)

  ! SM: IS it ok to call f_err_throw within an OpenMP environment?
  ! Anyway this condition should hopefully never be true...
  if (ithread>nthread-1) then
      call f_err_throw('wrong thread ID')
  end if
  !$omp do schedule(static)
  do j2 = 1, maxIter
     !this condition ensures that we manage only the interesting part for the FFT
     !if (iproc*(md2/nproc)+j2 <= n2dim) then

        do i1=1,n1dim,lot
           nfft = min(i1 + (lot -1), n1dim) - i1 +1

            if (halffty) then
              !inserting real data into complex array of half lenght
              call halfill_upcorn(md1,md3,lot,nfft,n3,zf(1,i1,1,j2),zw(1,1,1,ithread))
            else if (cplx) then
              !zf should have four indices
              call C_fill_upcorn(md1,md3,lot,nfft,n3,zf(1,i1,1,j2),zw(1,1,1,ithread))
            else
              call P_fill_upcorn(md1,md3,lot,nfft,n3,zf(1,i1,1,j2),zw(1,1,1,ithread))
            end if
           !performing FFT

           !input: I1,I3,J2,(Jp2)
           inzee=1
           do i=1,ic3
              call fftstp_sg(lot,nfft,n3dim,lot,n3dim,zw(1,1,inzee,ithread), &
                zw(1,1,3-inzee,ithread),ntrig,btrig3,after3(i),now3(i),before3(i),1)
              inzee=3-inzee
           enddo

           !output: I1,i3,J2,(Jp2)
           !exchanging components
           !input: I1,i3,J2,(Jp2)
           if (halffty) then
              call scramble_unpack(i1,j2,lot,nfft,n1dim,n3,md2,nproc,nd3,&
                   zw(1,1,inzee,ithread),zmpi2,cosinarr)
           else
              call scramble_P(i1,j2,lot,nfft,n1,n3,md2,nproc,nd3,zw(1,1,inzee,ithread),zmpi2)
           end if
           !output: I1,J2,i3,(Jp2)
        end do
     !end if
  end do
  !$omp end do
  ! DO NOT USE NOWAIT, removes the implicit barrier

  gflops_fft=5*2*real(n1dim*maxIter,f_double)*n3dim*log(real(n3dim,f_double))

  !$omp master
    nd3=nd3/n3pr1
    !Interprocessor data transposition
    !input: I1,J2,j3,jp3,(Jp2)
    if (nproc > 1 .and. iproc < n3pr1*n3pr2) then
       call f_timing(TCAT_PSOLV_COMPUT,'OF')
       call f_timing(TCAT_PSOLV_COMMUN,'ON')
       !communication scheduling
       call MPI_ALLTOALL(zmpi2,2*n1dim*(md2/nproc)*(nd3/n3pr2), &
            MPI_double_precision, &
            zmpi1,2*n1dim*(md2/nproc)*(nd3/n3pr2), &
            MPI_double_precision,planes_comm,ierr)
       call f_timing(TCAT_PSOLV_COMMUN,'OF')
       call f_timing(TCAT_PSOLV_COMPUT,'ON')
    endif

    !output: I1,J2,j3,Jp2,(jp3)
  !$omp end master
  !$omp barrier

  !now each process perform complete convolution of its planes

  if (n3pr1==1) then
    maxIter = min(nd3 /nproc, n3/2+1 - iproc*(nd3/nproc))
  else
    if (iproc < n3pr1*n3pr2) then
        maxIter = nd3/n3pr2
    else
        maxIter = 0
    endif
  endif

  gflops_fft=gflops_fft+5*2*maxIter*(real(n2dimp/n3pr1*n1,f_double)*log(real(n1,f_double))+&
       real(n1p/n3pr1*n2,f_double)*log(real(n2,f_double)))
  gflops_fft=gflops_fft+2*f_size(pot)

  strten_omp=0

  !$omp do schedule(static)
  do j3 = 1, maxIter
       !this condition ensures that we manage only the interesting part for the FFT
     !if (iproc*(nd3/nproc)+j3 <= n3/2+1) then
          Jp2stb=1
          J2stb=1
          Jp2stf=1
          J2stf=1
          ! transform along x axis
          lot=ncache/(4*n1)
          if (lot < 1) then
             write(6,*)&
                  'convolxc_off:ncache has to be enlarged to be able to hold at' //&
                  'least one 1-d FFT of this size even though this will' //&
                  'reduce the performance for shorter transform lengths'
             stop
          endif

          do j=1,n2dimp/n3pr1,lot
           nfft=min(j+(lot-1), n2dimp/n3pr1) -j +1

             !reverse index ordering, leaving the planes to be transformed at the end
             !input: I1,J2,j3,Jp2,(jp3)
             if (nproc > 1) then
                call G_mpiswitch_upcorn2(j3,nfft,Jp2stb,J2stb,lot,&
                     n1,n1dim,md2,nd3,n3pr1,n3pr2,zmpi1,zw(1,1,1,ithread))
             else
                call G_mpiswitch_upcorn(j3,nfft,Jp2stb,J2stb,lot,&
                     n1,n1dim,md2,nd3,nproc,zmpi2,zw(1,1,1,ithread))
             endif

             !output: J2,Jp2,I1,j3,(jp3)
             !performing FFT
             !input: I2,I1,j3,(jp3)
             inzee=1
             do i=1,ic1-1
                call fftstp_sg(lot,nfft,n1,lot,n1,zw(1,1,inzee,ithread),zw(1,1,3-inzee,ithread),&
                     ntrig,btrig1,after1(i),now1(i),before1(i),1)
                inzee=3-inzee
             enddo

             !storing the last step into zt array
             i=ic1
             call fftstp_sg(lot,nfft,n1,lzt/n3pr1,n1,zw(1,1,inzee,ithread),zt(1,j,1,ithread),&
                  ntrig,btrig1,after1(i),now1(i),before1(i),1)
             !output: I2,i1,j3,(jp3)
          end do

          !transform along y axis
          lot=ncache/(4*n2)
          if (lot < 1) then
             write(6,*)&
                  'convolxc_off:ncache has to be enlarged to be able to hold at' //&
                  'least one 1-d FFT of this size even though this will' //&
                  'reduce the performance for shorter transform lengths'
             stop
          endif

          !LG: I am not convinced that putting MPI_ALLTOALL inside a loop
          !!   is a good idea: this will break OMP parallelization, moreover
          !!   the granularity of the external loop will impose too many communications
          if (n3pr1 > 1 .and. inplane_comm/=FMPI_COMM_NULL) then
            call f_timing(TCAT_PSOLV_COMPUT,'OF')
            call f_timing(TCAT_PSOLV_COMMUN,'ON')

            call MPI_ALLTOALL(zt(1,1,1,ithread),2*(n1p/n3pr1)*(lzt/n3pr1),MPI_double_precision,zt_t,2*(n1p/n3pr1)*(lzt/n3pr1), &
                            MPI_double_precision,inplane_comm,ierr)

            call f_timing(TCAT_PSOLV_COMMUN,'OF')
            call f_timing(TCAT_PSOLV_COMPUT,'ON')
          endif

          do j=1,n1p/n3pr1,lot
           nfft=min(j+(lot-1),n1p/n3pr1)-j+1
             !reverse ordering
             !input: I2,i1,j3,(jp3)
             if (n3pr1 >1) then
               call G_switch_upcorn2(nfft,n2,n2dim,lot,n1p,lzt,zt_t(1,1,1),zw(1,1,1,ithread),n3pr1,j)
             else
               call G_switch_upcorn(nfft,n2,n2dim,lot,n1,lzt,zt(1,1,j,ithread),zw(1,1,1,ithread))
             endif
             !output: i1,I2,j3,(jp3)
             !performing FFT
             !input: i1,I2,j3,(jp3)
             inzee=1

             do i=1,ic2
                call fftstp_sg(lot,nfft,n2,lot,n2,zw(1,1,inzee,ithread),zw(1,1,3-inzee,ithread),&
                     ntrig,btrig2,after2(i),now2(i),before2(i),1)
                inzee=3-inzee
             enddo
             !output: i1,i2,j3,(jp3)
             !Multiply with kernel in fourier space
             i3=mod(iproc,n3pr2)*(nd3/n3pr2)+j3

            j1start=0
            if (n3pr1>1) j1start=(n1p/n3pr1)*iproc_inplane

            if (geocode == 'P') then
               !call P_multkernel(nd1,nd2,n1,n2,n3,lot,nfft,j+j1start,pot(1,1,j3),zw(1,1,inzee,ithread),&
               !    i3,mesh%hgrids(1),mesh%hgrids(2),mesh%hgrids(3),offset,scal,strten_omp)
              call P_multkernel_NO(nd1,nd2,n1,n2,n3,lot,nfft,j+j1start,pot(1,1,j3),zw(1,1,inzee,ithread),&
                   i3,mesh,offset,scal,mu0_square,strten_omp)
             else
                !write(*,*) 'pot(1,1,j3) = ', pot(1,1,j3)
                call multkernel(nd1,nd2,n1,n2,lot,nfft,j+j1start,pot(1,1,j3),zw(1,1,inzee,ithread))
             end if

!TRANSFORM BACK IN REAL SPACE
             !transform along y axis
             !input: i1,i2,j3,(jp3)
             do i=1,ic2
                call fftstp_sg(lot,nfft,n2,lot,n2,zw(1,1,inzee,ithread),zw(1,1,3-inzee,ithread),&
                     ntrig,ftrig2,after2(i),now2(i),before2(i),-1)
               !zw(:,:,3-inzee)=zw(:,:,inzee)
                inzee=3-inzee
             end do

             !reverse ordering
             !input: i1,I2,j3,(jp3)
           if (n3pr1 == 1) then
             call G_unswitch_downcorn(nfft,n2,n2dim,lot,n1,lzt, &
               zw(1,1,inzee,ithread),zt(1,1,j,ithread))
           else
             call G_unswitch_downcorn2(nfft,n2,n2dim,lot,n1p,lzt, &
               zw(1,1,inzee,ithread),zt_t(1,1,1),n3pr1,j)
           endif
             !output: I2,i1,j3,(jp3)
        end do
          !transform along x axis
          !input: I2,i1,j3,(jp3)

        !LG: this MPI_ALLTOALL is inside a loop. I think that it will rapidly become unoptimal
        if (n3pr1 > 1 .and. inplane_comm/=FMPI_COMM_NULL) then
            call f_timing(TCAT_PSOLV_COMPUT,'OF')
            call f_timing(TCAT_PSOLV_COMMUN,'ON')

            call MPI_ALLTOALL(zt_t,2*(n1p/n3pr1)*(lzt/n3pr1),MPI_double_precision,&
                 zt(1,1,1,ithread),2*(n1p/n3pr1)*(lzt/n3pr1), &
                            MPI_double_precision,inplane_comm,ierr)

            call f_timing(TCAT_PSOLV_COMMUN,'OF')
            call f_timing(TCAT_PSOLV_COMPUT,'ON')
          endif

          lot=ncache/(4*n1)
          do j=1,n2dimp/n3pr1,lot
           nfft=min(j+(lot-1),n2dimp/n3pr1)-j+1

            !performing FFT
             i=1
             if (n3pr1 > 1) then
               call fftstp_sg(lzt/n3pr1,nfft,n1,lot,n1,zt(1,j,1,ithread),zw(1,1,1,ithread),&
                  ntrig,ftrig1,after1(i),now1(i),before1(i),-1)
             else
               call fftstp_sg(lzt,nfft,n1,lot,n1,zt(1,j,1,ithread),zw(1,1,1,ithread),&
                  ntrig,ftrig1,after1(i),now1(i),before1(i),-1)
             endif
             inzee=1
             do i=2,ic1
                call fftstp_sg(lot,nfft,n1,lot,n1,zw(1,1,inzee,ithread),zw(1,1,3-inzee,ithread),&
                     ntrig,ftrig1,after1(i),now1(i),before1(i),-1)
                inzee=3-inzee
             enddo

             !output: I2,I1,j3,(jp3)
             !reverse ordering
             !input: J2,Jp2,I1,j3,(jp3)
             if (nproc == 1) then
                call G_unmpiswitch_downcorn(j3,nfft,Jp2stf,J2stf,lot,n1,&
                     n1dim,md2,nd3,nproc,zw(1,1,inzee,ithread),zmpi2)
             else
                call G_unmpiswitch_downcorn2(j3,nfft,Jp2stf,J2stf,lot,n1,&
                     n1dim,md2,nd3,n3pr1,n3pr2,zw(1,1,inzee,ithread),zmpi1)
             endif
             ! output: I1,J2,j3,Jp2,(jp3)
          end do
     !endif

!END OF TRANSFORM FOR X AND Z

       end do
  !$omp end do
    !$omp critical
    !do i = 1, 6
    !  strten(j) = strten(j) + strten_omp(j)
    !enddo
    strten = strten + strten_omp
    !$omp end critical
  ! DO NOT USE NOWAIT, removes the implicit barrier

  !$omp master

!TRANSFORM BACK IN Y
    !Interprocessor data transposition
    !input: I1,J2,j3,Jp2,(jp3)
    if (nproc > 1 .and. iproc < n3pr1*n3pr2) then
       call f_timing(TCAT_PSOLV_COMPUT,'OF')
       call f_timing(TCAT_PSOLV_COMMUN,'ON')
       !communication scheduling
       call MPI_ALLTOALL(zmpi1,2*n1dim*(md2/nproc)*(nd3/n3pr2), &
            MPI_double_precision, &
            zmpi2,2*n1dim*(md2/nproc)*(nd3/n3pr2), &
            MPI_double_precision,planes_comm,ierr)
       call f_timing(TCAT_PSOLV_COMMUN,'OF')
       call f_timing(TCAT_PSOLV_COMPUT,'ON')
    endif
    !output: I1,J2,j3,jp3,(Jp2)
    nd3=nd3*n3pr1
  !$omp end master
  !$omp barrier

  !transform along z axis
  !input: I1,J2,i3,(Jp2)
  lot=ncache/(4*n3dim)

  maxIter = min(md2/nproc, n2dim - iproc *(md2/nproc))

  !$omp do schedule(static)
  do j2 = 1, maxIter
     !this condition ensures that we manage only the interesting part for the FFT
        do i1=1,n1dim,lot
           nfft=min(i1+(lot-1),n1dim)-i1+1

           !reverse ordering
           !input: I1,J2,i3,(Jp2)
           if (halffty) then
              call unscramble_pack(i1,j2,lot,nfft,n1dim,n3,md2,nproc,nd3,zmpi2, &
                zw(1,1,1,ithread),cosinarr)
           else
              call unscramble_P(i1,j2,lot,nfft,n1,n3,md2,nproc,nd3,zmpi2,zw(1,1,1,ithread))
           end if
           !output: I1,i3,J2,(Jp2)

           !performing FFT
           !input: I1,i3,J2,(Jp2)
           inzee=1
           do i=1,ic3
              call fftstp_sg(lot,nfft,n3dim,lot,n3dim,zw(1,1,inzee,ithread), &
                zw(1,1,3-inzee,ithread),ntrig,ftrig3,after3(i),now3(i),before3(i),-1)
              inzee=3-inzee
           enddo
           !output: I1,I3,J2,(Jp2)

           !rebuild the output array

            if (halffty) then
              call unfill_downcorn(md1,md3,lot,nfft,n3,zw(1,1,inzee,ithread),zf(1,i1,1,j2)&
                   ,scal)!,ehartreetmp)
            else if (cplx) then
              call C_unfill_downcorn(md1,md3,lot,nfft,n3,zw(1,1,inzee,ithread),zf(1,i1,1,j2),scal)
            else
              call P_unfill_downcorn(md1,md3,lot,nfft,n3,zw(1,1,inzee,ithread),zf(1,i1,1,j2),scal)
            end if

           !integrate local pieces together
           !ehartree=ehartree+0.5d0*ehartreetmp*hx*hy*hz

        end do
  end do
  !$omp end do



  !$omp end parallel

  call f_free(zw)
  call f_free(zt)

  if (n3pr1 > 1) call f_free(zt_t)

!END OF TRANSFORM IN Y DIRECTION

  !De-allocations
  call f_free(btrig1)
  call f_free(ftrig1)
  call f_free(btrig2)
  call f_free(ftrig2)
  call f_free(btrig3)
  call f_free(ftrig3)
  call f_free(zmpi2)
  if (halffty) then
     call f_free(cosinarr)
  end if
  if (nproc > 1) then
     call f_free(zmpi1)
  end if
  call f_timing(TCAT_PSOLV_COMPUT,'OF')

  call f_perf_set_model(performance_info,F_PERF_GFLOPS,nint(gflops_fft,f_long))
  call f_perf_set_model(performance_info,F_PERF_READB,f_sizeof(zf)+f_sizeof(pot))
  call f_perf_set_model(performance_info,F_PERF_WRITEB,f_sizeof(zf))
  call f_release_routine(performance_info=performance_info)
  call f_perf_free(performance_info)
  !call system_clock(ncount1,ncount_rate,ncount_max)
  !write(*,*) 'TIMING:PS ', real(ncount1-ncount0)/real(ncount_rate)
END SUBROUTINE G_PoissonSolver

"""
nettoyer_fortran(code, 'test.f90')