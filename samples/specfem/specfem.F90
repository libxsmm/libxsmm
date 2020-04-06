!
! test program for LIBXSMM function calls
!
! uses SPECFEM3D_GLOBE routine compute_forces_crust_mantle_Dev() with dummy example
!

! we switch between vectorized and non-vectorized version by using pre-processor flag FORCE_VECTORIZATION
! and macros INDEX_IJK, DO_LOOP_IJK, ENDDO_LOOP_IJK defined in config.fh
#include "config.fh"

!-------------------------------------------------------------------
!
! modules
!
!-------------------------------------------------------------------

module my_libxsmm

  use libxsmm !,only: LIBXSMM_SMMFUNCTION,libxsmm_dispatch,libxsmm_mmcall,libxsmm_init,libxsmm_finalize

  implicit none

  ! function pointers
  ! (note: defined for single precision, thus needs CUSTOM_REAL to be SIZE_REAL)
  type(LIBXSMM_SMMFUNCTION) :: xmm1, xmm2, xmm3

  ! prefetch versions
  type(LIBXSMM_SMMFUNCTION) :: xmm1p, xmm2p, xmm3p

  logical :: USE_XSMM_FUNCTION,USE_XSMM_FUNCTION_PREFETCH

end module my_libxsmm

!
!-------------------------------------------------------------------
!

module constants

  implicit none

  integer, parameter :: SIZE_REAL = 4, SIZE_DOUBLE = 8
  integer, parameter :: CUSTOM_REAL = SIZE_REAL

  integer, parameter :: ISTANDARD_OUTPUT = 6
  integer, parameter :: IMAIN = ISTANDARD_OUTPUT

  ! number of GLL points in each direction of an element (degree plus one)
  integer, parameter :: NGLLX = 5
  integer, parameter :: NGLLY = NGLLX
  integer, parameter :: NGLLZ = NGLLX
  integer, parameter :: NGLLCUBE = NGLLX * NGLLY * NGLLZ

  ! Deville routines optimized for NGLLX = NGLLY = NGLLZ = 5
  integer, parameter :: m1 = NGLLX, m2 = NGLLX * NGLLY

  ! 3-D simulation
  integer, parameter :: NDIM = 3

  ! some useful constants
  double precision, parameter :: PI = 3.141592653589793d0

  integer, parameter :: IFLAG_IN_FICTITIOUS_CUBE = 11

end module constants

!
!-------------------------------------------------------------------
!

module specfem_par

! main parameter module for specfem simulations

  use constants
  use libxsmm,only: LIBXSMM_ALIGNMENT

  implicit none

  !------------------------------------------------

  ! number of spectral elements in x/y/z-directions
  integer,parameter :: NEX = 40
  integer,parameter :: NEY = 40
  integer,parameter :: NEZ = 25

  !------------------------------------------------

  ! MPI rank (dummy, no MPI for this test needed)
  integer :: myrank

  ! array with derivatives of Lagrange polynomials and precalculated products
  real(kind=CUSTOM_REAL), dimension(NGLLX,NGLLX) :: hprime_xx,hprimewgll_xx
  real(kind=CUSTOM_REAL), dimension(NGLLX,NGLLX) :: hprime_xxT,hprimewgll_xxT

  real(kind=CUSTOM_REAL), dimension(NGLLX,NGLLY) :: wgllwgll_xy
  real(kind=CUSTOM_REAL), dimension(NGLLX,NGLLZ) :: wgllwgll_xz
  real(kind=CUSTOM_REAL), dimension(NGLLY,NGLLZ) :: wgllwgll_yz

  ! arrays for Deville and force_vectorization
  real(kind=CUSTOM_REAL), dimension(NGLLX,NGLLY,NGLLZ) :: wgllwgll_xy_3D,wgllwgll_xz_3D,wgllwgll_yz_3D

  ! mesh parameters
  ! number of spectral elements
  integer :: NSPEC

  ! number of global nodes
  integer :: NGLOB

  ! local to global indexing
  integer, dimension(:,:,:,:),allocatable :: ibool

  ! displacement, velocity, acceleration
  real(kind=CUSTOM_REAL), dimension(:,:),allocatable :: displ,accel

  ! for verification
  real(kind=CUSTOM_REAL), dimension(:,:),allocatable :: accel_default

  !slow-down: please don't use unless you're sure... !dir$ ATTRIBUTES align:LIBXSMM_ALIGNMENT :: displ,accel,ibool,accel_default

  ! gravity
  logical,parameter :: GRAVITY_VAL = .true.

  ! optimized arrays
  integer, dimension(:,:),allocatable :: ibool_inv_tbl
  integer, dimension(:,:),allocatable :: ibool_inv_st
  integer, dimension(:,:),allocatable :: phase_iglob
  integer, dimension(2) :: num_globs

  ! work array with contributions
  real(kind=CUSTOM_REAL), dimension(:,:,:,:,:),allocatable :: sum_terms

  ! inner / outer elements crust/mantle region
  integer :: num_phase_ispec
  integer :: nspec_inner,nspec_outer
  integer, dimension(:,:), allocatable :: phase_ispec_inner
  integer :: iphase

end module specfem_par



!-------------------------------------------------------------------
!
! main program
!
!-------------------------------------------------------------------

program test

  use specfem_par
  use my_libxsmm

  implicit none

  ! timing
  double precision :: duration,duration_default
  integer(kind=8) :: start

  ! verification
  real :: diff
  integer, dimension(2) :: iloc_max
  logical, parameter :: DEBUG_VERIFICATION = .false.

  ! repetitions (time steps)
  integer :: it
  integer,parameter :: NSTEP = 20 ! should be > NSTEP_JITTER because of average timing
  integer,parameter :: NSTEP_JITTER = 5 ! skip first few steps when timing the kernels (the first steps exhibit runtime jitter)

  ! different versions
  integer,parameter :: num_versions = 5
  character(len=20) :: str_version(num_versions) = (/character(len=20) :: &
                                                     "Deville loops", &
                                                     "unrolled loops", &
                                                     "LIBXSMM dispatch" , &
                                                     "LIBXSMM prefetch", &
                                                     "LIBXSMM static" &
                                                    /)
  double precision :: avg_time(num_versions)
  integer :: iversion

  print *,'--------------------------------------'
  print *,'specfem example'
  print *,'--------------------------------------'
  print *

  ! creates test mesh
  call setup_mesh()

  ! prepares arrays for time iteration loop
  call prepare_timerun()

  ! OpenMP output info
  call prepare_openmp()

  ! prepares libxsmm functions
  call prepare_xsmm()

  ! timing averages
  avg_time(:) = 0.d0
  iphase = 2

  do it = 1,NSTEP

    print *
    print *,'step ',it

    do iversion = 1,num_versions
      ! initializes
      accel(:,:) = 0._CUSTOM_REAL

      ! timing
      start = libxsmm_timer_tick()

      ! computes forces
      select case (iversion)
      case (1)
        ! Deville loops
        call compute_forces_Dev()

      case (2)
        ! unrolled loops
        call compute_forces_noDev()

      case (3)
        ! LIBXSMM with dispatch functions
        if (USE_XSMM_FUNCTION) then
          call compute_forces_with_xsmm()
        else
          cycle
        endif

      case (4)
        ! LIBXSMM with prefetch function calls
        if (USE_XSMM_FUNCTION_PREFETCH) then
          call compute_forces_with_xsmm_prefetch()
        else
          cycle
        endif

      case (5)
        ! LIBXSMM with static function calls
        call compute_forces_with_xsmm_static()

      end select

      ! timing
      duration = libxsmm_timer_duration(start, libxsmm_timer_tick())
      if (iversion == 1) duration_default = duration

      ! average time
      if (it > NSTEP_JITTER) avg_time(iversion) = avg_time(iversion) + duration

      ! for verification
      if (iversion == 1) then
        accel_default(:,:) = accel(:,:)
      endif
      diff = maxval(abs(accel(:,:) - accel_default(:,:)))

      ! user output
      if (iversion == 1) then
        write(*,'(a30,a,f8.4,a)') 'duration with '//str_version(iversion),' = ',sngl(duration),' (s)'
      else
        write(*,'(a30,a,f8.4,a,f8.2,a,e12.4)') 'duration with '//str_version(iversion),' = ', &
                                    sngl(duration),' (s) / speedup = ', &
                                    sngl(100.0 * (duration_default-duration)/duration_default),' %  / maximum diff = ',diff
      endif

      ! check
      if (DEBUG_VERIFICATION) then
        iloc_max = maxloc(abs(accel(:,:) - accel_default(:,:)))
        print *,'verification: max diff  = ',diff
        print *,'              iglob loc = ',iloc_max(1),iloc_max(2)
        print *,'maximum difference: #current vs. #default value'
        print *,'  ',accel(1,iloc_max(2)),accel_default(1,iloc_max(2))
        print *,'  ',accel(2,iloc_max(2)),accel_default(2,iloc_max(2))
        print *,'  ',accel(3,iloc_max(2)),accel_default(3,iloc_max(2))
        print *,'min/max accel values = ',minval(accel(:,:)),maxval(accel(:,:))
        print *
      endif

    enddo ! iversion
  enddo ! it

  ! average timing (avoiding the first 5 steps which fluctuate quite a bit...)
  avg_time(:) = avg_time(:) / dble(NSTEP - NSTEP_JITTER)

  print *
  print *,'=============================================='
  print *,'average over ',NSTEP - NSTEP_JITTER,'repetitions'
  write(*,'(a30,a,f8.4)') '  timing with '//str_version(1),' = ',avg_time(1)
  do iversion = 2,num_versions
    ! skip unused tests
    if (iversion == 3 .and. .not. USE_XSMM_FUNCTION) cycle
    if (iversion == 4 .and. .not. USE_XSMM_FUNCTION_PREFETCH) cycle

    write(*,'(a30,a,f8.4,a,f8.2,a)') '  timing with '//str_version(iversion),' = ', &
                                     avg_time(iversion),' / speedup = ', &
                                     sngl(100.0 * (avg_time(1)-avg_time(iversion))/avg_time(1)),' %'
  enddo
  print *,'=============================================='
  print *

  ! frees memory
  deallocate(displ,accel)
  deallocate(accel_default,ibool)
  deallocate(sum_terms)
  deallocate(ibool_inv_st,ibool_inv_tbl,phase_iglob)
  deallocate(phase_ispec_inner)

  ! finalizes LIBXSMM
  call libxsmm_finalize()

end program test

!
!-------------------------------------------------------------------
!

  subroutine setup_mesh()

  use constants
  use specfem_par

  implicit none

  integer :: i1,i2
  integer :: ix,iy,iz
  integer :: i,j,k
  integer :: iglob,ispec,inumber

  integer, dimension(:), allocatable :: mask_ibool
  integer, dimension(:,:,:,:), allocatable :: copy_ibool_ori

  logical, dimension(:), allocatable :: mask_ibool_flag

  ! total number of elements
  NSPEC = NEX * NEY * NEZ

  ! set up local to global numbering
  allocate(ibool(NGLLX,NGLLY,NGLLZ,NSPEC))
  ibool(:,:,:,:) = 0
  ispec = 0
  iglob = 0

  ! arranges a three-dimensional block, elements are collated side-by-side. mimicks a very simple unstructured grid.
  do iz = 1,NEZ
    do iy = 1,NEY
      do ix = 1,NEX
        ispec = ispec + 1
        ! GLL point indexing
        do k = 1,NGLLZ
          do j = 1,NGLLY
            do i = 1,NGLLX
              ! set up local to global numbering
              if ((i == 1) .and. (ix > 1)) then
                ! previous element along x-direction
                ibool(i,j,k,ispec) = ibool(NGLLX,j,k,ispec - 1)
              else if ((j == 1) .and. (iy > 1)) then
                ! previous element along y-direction
                ibool(i,j,k,ispec) = ibool(i,NGLLY,k,ispec - NEX)
              else if ((k == 1) .and. (iz > 1)) then
                ! previous element along z-direction
                ibool(i,j,k,ispec) = ibool(i,j,NGLLZ,ispec - NEX * NEY)
              else
                ! new point
                iglob = iglob + 1
                ibool(i,j,k,ispec) = iglob
              endif
            enddo
          enddo
        enddo

      enddo ! NEX
    enddo ! NEY
  enddo ! NEZ

  ! sets total numbers of nodes
  NGLOB = iglob

  print *,'mesh:'
  print *,' total number of elements      = ',NSPEC
  print *,' total number of global nodes  = ',NGLOB
  !print *,' ibool min/max = ',minval(ibool),maxval(ibool)

  ! checks
  if (ispec /= NSPEC) stop 'Invalid ispec count'
  if (minval(ibool(:,:,:,:)) < 1) stop 'Invalid ibool minimum value'
  if (maxval(ibool(:,:,:,:)) > NSPEC * NGLLX * NGLLY * NGLLZ) stop 'Invalid ibool maximum value'

  ! we can create a new indirect addressing to reduce cache misses
  allocate(copy_ibool_ori(NGLLX,NGLLY,NGLLZ,NSPEC),mask_ibool(nglob))
  mask_ibool(:) = -1
  copy_ibool_ori(:,:,:,:) = ibool(:,:,:,:)
  inumber = 0
  do ispec = 1,NSPEC
    do k = 1,NGLLZ
      do j = 1,NGLLY
        do i = 1,NGLLX
          if (mask_ibool(copy_ibool_ori(i,j,k,ispec)) == -1) then
            ! creates a new point
            inumber = inumber + 1
            ibool(i,j,k,ispec) = inumber
            mask_ibool(copy_ibool_ori(i,j,k,ispec)) = inumber
          else
            ! uses an existing point created previously
            ibool(i,j,k,ispec) = mask_ibool(copy_ibool_ori(i,j,k,ispec))
          endif
        enddo
      enddo
    enddo
  enddo
  if (inumber /= NGLOB) stop 'Invalid inumber count'
  deallocate(copy_ibool_ori,mask_ibool)

  ! define polynomial derivatives & weights
  ! (dummy values)
  do i1 = 1,NGLLX
    do i2 = 1,NGLLX
      hprime_xx(i2,i1) = i1 * 0.1 + i2 * 0.2  ! original: real(lagrange_deriv_GLL(i1-1,i2-1,xigll,NGLLX), kind=CUSTOM_REAL)
      hprimewgll_xx(i2,i1) = hprime_xx(i2,i1) * (i2 * 1.0/NGLLX) ! real(lagrange_deriv_GLL(i1-1,i2-1,xigll,NGLLX)*wxgll(i2), kind=CUSTOM_REAL)
    enddo
  enddo
  do i = 1,NGLLX
    do j = 1,NGLLY
      wgllwgll_xy(i,j) = (i * 1.0/NGLLX) * (j * 1.0/NGLLY) ! original: real(wxgll(i)*wygll(j), kind=CUSTOM_REAL)
    enddo
  enddo
  do i = 1,NGLLX
    do k = 1,NGLLZ
      wgllwgll_xz(i,k) = (i * 1.0/NGLLX) * (k * 1.0/NGLLZ) ! original: real(wxgll(i)*wzgll(k), kind=CUSTOM_REAL)
    enddo
  enddo
  do j = 1,NGLLY
    do k = 1,NGLLZ
      wgllwgll_yz(j,k) = (j * 1.0/NGLLY) * (k * 1.0/NGLLZ) ! original: real(wygll(j)*wzgll(k), kind=CUSTOM_REAL)
    enddo
  enddo

  ! define a 3D extension in order to be able to force vectorization in the compute_forces_**_Dev routines
  do k = 1,NGLLZ
    do j = 1,NGLLY
      do i = 1,NGLLX
        wgllwgll_yz_3D(i,j,k) = wgllwgll_yz(j,k)
        wgllwgll_xz_3D(i,j,k) = wgllwgll_xz(i,k)
        wgllwgll_xy_3D(i,j,k) = wgllwgll_xy(i,j)
      enddo
    enddo
  enddo

  ! check that optimized routines from Deville et al. (2002) can be used
  if (NGLLX /= 5 .or. NGLLY /= 5 .or. NGLLZ /= 5) &
    stop 'Deville et al. (2002) routines can only be used if NGLLX = NGLLY = NGLLZ = 5'

  ! define transpose of derivation matrix
  do j = 1,NGLLX
    do i = 1,NGLLX
      hprime_xxT(j,i) = hprime_xx(i,j)
      hprimewgll_xxT(j,i) = hprimewgll_xx(i,j)
    enddo
  enddo

  ! displacement and acceleration (dummy fields)
  allocate(displ(NDIM,NGLOB),accel(NDIM,NGLOB))
  accel(:,:) = 0._CUSTOM_REAL

  ! sets initial dummy values (to avoid getting only zero multiplications later on)
  if (.true.) then
    ! arbitrary linear function
    do iglob = 1,NGLOB
      displ(:,iglob) = dble(iglob - 1) / dble(NGLOB - 1)
    enddo
  else
    ! arbitrary sine function
    allocate(mask_ibool_flag(NGLOB))
    mask_ibool_flag(:) = .false.
    ispec = 0
    do iz = 1,NEZ
      do iy = 1,NEY
        do ix = 1,NEX
          ispec = ispec + 1
          ! GLL point indexing
          do k = 1,NGLLZ
            do j = 1,NGLLY
              do i = 1,NGLLX
                iglob = ibool(i,j,k,ispec)
                if (.not. mask_ibool_flag(iglob)) then
                  ! only assigns global value once
                  mask_ibool_flag(iglob) = .true.
                  displ(:,iglob) = sin(PI * dble(ix - 1) / dble(NEX - 1)) &
                                 * sin(PI * dble(iy - 1) / dble(NEY - 1)) &
                                 * sin(PI * dble(iz - 1) / dble(NEZ - 1))
                endif
              enddo
            enddo
          enddo
        enddo
      enddo
    enddo
    deallocate(mask_ibool_flag)
  endif

  ! for verification
  allocate(accel_default(NDIM,NGLOB))
  accel_default(:,:) = 0._CUSTOM_REAL

  end subroutine setup_mesh

!
!-------------------------------------------------------------------
!

  subroutine prepare_timerun()

  use specfem_par

  implicit none

  ! local parameters
  integer :: num_elements,ispec,ier

  ! setup inner/outer elements (single slice only, no outer elements for halo)
  myrank = 0
  ! no MPI over-lapping communication in this example
  nspec_inner = NSPEC
  nspec_outer = 0
  num_phase_ispec = NSPEC
  allocate(phase_ispec_inner(num_phase_ispec,2),stat=ier)
  if (ier /= 0 ) call exit_mpi(myrank,'Error allocating array phase_ispec_inner_crust_mantle')
  phase_ispec_inner(:,:) = 0
  do ispec = 1,NSPEC
    phase_ispec_inner(ispec,2) = ispec
  enddo

  ! from original routine prepare_timerun_ibool_inv_tbl()

  ! note: we use allocate for sum_terms arrays rather than defining within subroutine compute_forces_**_Dev() itself
  !       as it will crash when using OpenMP and operating systems with small stack sizes
  !       e.g. see http://stackoverflow.com/questions/22649827/illegal-instruction-error-when-running-openmp-in-gfortran-mac
  allocate(sum_terms(NDIM,NGLLX,NGLLY,NGLLZ,NSPEC),stat=ier)
  if (ier /= 0) stop 'Error allocating sum_terms arrays'
  sum_terms(:,:,:,:,:) = 0._CUSTOM_REAL

  ! inverse table
  ! this helps to speedup the assembly, especially with OpenMP (or on MIC) threading
  ! allocating arrays
  allocate(ibool_inv_tbl(NGLLX*NGLLY*NGLLZ*NSPEC,2),stat=ier)
  if (ier /= 0) stop 'Error allocating ibool_inv_tbl arrays'

  allocate(ibool_inv_st(NGLOB+1,2),stat=ier)
  if (ier /= 0) stop 'Error allocating ibool_inv_st arrays'

  allocate(phase_iglob(NGLOB,2),stat=ier)
  if (ier /= 0) stop 'Error allocating phase_iglob arrays'

  ! initializing
  num_globs(:) = 0
  ibool_inv_tbl(:,:) = 0
  ibool_inv_st(:,:) = 0
  phase_iglob(:,:) = 0

  !---- make inv. table ----------------------
  ! loops over phases
  ! (1 == outer elements / 2 == inner elements)
  do iphase = 1,2
    ! crust mantle
    if (iphase == 1) then
      ! outer elements (iphase=1)
      num_elements = nspec_outer
    else
      ! inner elements (iphase=2)
      num_elements = nspec_inner
    endif
    call make_inv_table(iphase,NGLOB,NSPEC, &
                        num_elements,phase_ispec_inner, &
                        ibool,phase_iglob, &
                        ibool_inv_tbl, ibool_inv_st, &
                        num_globs)
  enddo

  ! user output
  if (myrank == 0) then
    write(IMAIN,*) " inverse table of ibool done"
    call flush_IMAIN()
  endif

  ! synchronizes processes
  call synchronize_all()

  contains

    subroutine make_inv_table(iphase,nglob,nspec, &
                              phase_nspec,phase_ispec,ibool,phase_iglob, &
                              ibool_inv_tbl,ibool_inv_st,num_globs,idoubling)

    implicit none

    ! arguments
    integer,intent(in) :: iphase
    integer,intent(in) :: nglob
    integer,intent(in) :: nspec
    integer,intent(in) :: phase_nspec
    integer, dimension(:,:),intent(in) :: phase_ispec
    integer, dimension(:,:,:,:),intent(in) :: ibool

    integer, dimension(:,:),intent(inout) :: phase_iglob
    integer, dimension(:,:),intent(inout) :: ibool_inv_tbl
    integer, dimension(:,:),intent(inout) :: ibool_inv_st
    integer, dimension(:),intent(inout) :: num_globs

    integer,dimension(:),optional :: idoubling

    ! local parameters
    integer, dimension(:),   allocatable :: ibool_inv_num
    integer, dimension(:,:), allocatable :: ibool_inv_tbl_tmp
    integer :: num_alloc_ibool_inv_tbl,num_alloc_ibool_inv_tbl_theor
    integer :: num_used_ibool_inv_tbl
    integer :: ip, iglob, ispec_p, ispec, iglob_p, ier
    integer :: inum
#ifdef FORCE_VECTORIZATION
    integer :: ijk
#else
    integer :: i,j,k
#endif
    logical :: is_inner_core

    ! tolerance number of shared degrees per node
    integer, parameter :: N_TOL = 20

    ! checks if anything to do (e.g., no outer elements for single process simulations)
    if (phase_nspec == 0) return

    ! checks if inner core region
    if (present(idoubling)) then
      is_inner_core = .true.
    else
      is_inner_core = .false.
    endif

    ! allocates temporary arrays
    allocate(ibool_inv_num(nglob),stat=ier)
    if (ier /= 0) stop 'Error allocating ibool_inv_num array'

    ! gets valence of global degrees of freedom for current phase (inner/outer) elements
    ibool_inv_num(:) = 0
    do ispec_p = 1,phase_nspec
      ispec = phase_ispec(ispec_p,iphase)

      ! exclude fictitious elements in central cube
      if (is_inner_core) then
        if (idoubling(ispec) == IFLAG_IN_FICTITIOUS_CUBE) cycle
      endif

      DO_LOOP_IJK
        iglob = ibool(INDEX_IJK,ispec)
        ! increases valence counter
        ibool_inv_num(iglob) = ibool_inv_num(iglob) + 1
      ENDDO_LOOP_IJK
    enddo

    ! gets maximum valence value
    num_alloc_ibool_inv_tbl = maxval(ibool_inv_num(:))

    ! theoretical number of maximum shared degrees per node
    num_alloc_ibool_inv_tbl_theor = N_TOL*(NGLLX*NGLLY*NGLLZ*nspec/nglob+1)

    ! checks valence
    if (num_alloc_ibool_inv_tbl < 1 .or. num_alloc_ibool_inv_tbl > num_alloc_ibool_inv_tbl_theor) then
      print *,'Error invalid maximum valence:'
      print *,'valence value = ',num_alloc_ibool_inv_tbl,' - theoretical maximum = ',num_alloc_ibool_inv_tbl_theor
      stop 'Error invalid maximum valence value'
    endif
    ! debug
    !print *,myrank,'maximum shared degrees theoretical = ',num_alloc_ibool_inv_tbl_theor ! regional_Greece_small example: 40
    !print *,myrank,'maximum shared degrees from array  = ',maxval(ibool_inv_num(:))      ! regional_Greece_small example: 8 and 16

    allocate(ibool_inv_tbl_tmp(num_alloc_ibool_inv_tbl,nglob),stat=ier)
    if (ier /= 0) stop 'Error allocating ibool_inv_tbl_tmp array'

    !---- make temporary array of inv. table : ibool_inv_tbl_tmp
    ibool_inv_tbl_tmp(:,:) = 0
    ibool_inv_num(:) = 0
    do ispec_p = 1,phase_nspec
      ispec = phase_ispec(ispec_p,iphase)

      ! exclude fictitious elements in central cube
      if (is_inner_core) then
        if (idoubling(ispec) == IFLAG_IN_FICTITIOUS_CUBE) cycle
      endif

      DO_LOOP_IJK

        iglob = ibool(INDEX_IJK,ispec)

        ! increases counter
        ibool_inv_num(iglob) = ibool_inv_num(iglob) + 1

        ! inverse table
        ! sets 1D index of local GLL point (between 1 and NGLLCUBE)
#ifdef FORCE_VECTORIZATION
        inum = ijk
#else
        inum = i + (j-1)*NGLLY + (k-1)*NGLLY*NGLLZ
#endif
        ! sets 1D index in local ibool array
        ibool_inv_tbl_tmp(ibool_inv_num(iglob),iglob) = inum + NGLLX*NGLLY*NGLLZ*(ispec-1)

      ENDDO_LOOP_IJK

    enddo

    !---- packing : ibool_inv_tbl_tmp -> ibool_inv_tbl
    ip = 0
    iglob_p = 0
    num_used_ibool_inv_tbl = 0
    do iglob = 1, nglob
      if (ibool_inv_num(iglob) /= 0) then
        iglob_p = iglob_p + 1

        phase_iglob(iglob_p,iphase) = iglob

        ! sets start index of table entry for this global node
        ibool_inv_st(iglob_p,iphase) = ip + 1

        ! sets maximum of used valence
        if (ibool_inv_num(iglob) > num_used_ibool_inv_tbl) num_used_ibool_inv_tbl = ibool_inv_num(iglob)

        ! loops over valence
        do inum = 1, ibool_inv_num(iglob)
          ! increases total counter
          ip = ip + 1
          ! maps local 1D index in ibool array
          ibool_inv_tbl(ip,iphase) = ibool_inv_tbl_tmp(inum,iglob)
        enddo
      endif
    enddo
    ! sets last entry in start index table
    ibool_inv_st(iglob_p+1,iphase) = ip + 1

    ! total number global nodes in this phase (inner/outer)
    num_globs(iphase) = iglob_p

    ! checks
    if (num_used_ibool_inv_tbl > num_alloc_ibool_inv_tbl)  then
      print *,"Error invalid inverse table setting:"
      print *,"  num_alloc_ibool_inv_tbl = ",num_alloc_ibool_inv_tbl
      print *,"  num_used_ibool_inv_tbl  = ",num_used_ibool_inv_tbl
      print *,"invalid value encountered: num_used_ibool_inv_tbl > num_alloc_ibool_inv_tbl"
      print *,"#### Program exits... ##########"
      call exit_MPI(myrank,'Error making inverse table for optimized arrays')
    endif

    ! debug
    !if (myrank == 0) then
    !  print *,'ibool_inv_tbl: '
    !  do iglob_p = 1,200
    !    print *,'  ',iglob_p,'table = ',(ibool_inv_tbl(ip,iphase), &
    !                                     ip = ibool_inv_st(iglob_p,iphase),ibool_inv_st(iglob_p+1,iphase)-1)
    !  enddo
    !endif

    ! frees memory
    deallocate(ibool_inv_num)
    deallocate(ibool_inv_tbl_tmp)

    end subroutine make_inv_table

  end subroutine prepare_timerun

!
!-------------------------------------------------------------------
!

  subroutine prepare_openmp()

! outputs OpenMP support info

#ifdef USE_OPENMP
  use specfem_par,only: myrank,IMAIN
#endif

  implicit none

#ifdef USE_OPENMP
  ! local parameters
  integer :: thread_id,num_threads
  integer :: num_procs,max_threads
  logical :: is_dynamic,is_nested
  ! OpenMP functions
  integer,external :: OMP_GET_NUM_THREADS,OMP_GET_THREAD_NUM
  integer,external :: OMP_GET_NUM_PROCS,OMP_GET_MAX_THREADS
  logical,external :: OMP_GET_DYNAMIC,OMP_GET_NESTED

  ! OpenMP only supported for Deville routine

!$OMP PARALLEL DEFAULT(NONE) &
!$OMP SHARED(myrank) &
!$OMP PRIVATE(thread_id,num_threads,num_procs,max_threads,is_dynamic,is_nested)
  ! gets thread number
  thread_id = OMP_GET_THREAD_NUM()

  ! gets total number of threads for this MPI process
  num_threads = OMP_GET_NUM_THREADS()

  ! OpenMP master thread only
  if (thread_id == 0) then
    ! gets additional environment info
    num_procs = OMP_GET_NUM_PROCS()
    max_threads = OMP_GET_MAX_THREADS()
    is_dynamic = OMP_GET_DYNAMIC()
    is_nested = OMP_GET_NESTED()

    ! user output
    if (myrank == 0) then
      write(IMAIN,*) ''
      write(IMAIN,*) 'OpenMP information:'
      write(IMAIN,*) '  number of threads = ', num_threads
      write(IMAIN,*) ''
      write(IMAIN,*) '  number of processors available      = ', num_procs
      write(IMAIN,*) '  maximum number of threads available = ', num_procs
      write(IMAIN,*) '  dynamic thread adjustement          = ', is_dynamic
      write(IMAIN,*) '  nested parallelism                  = ', is_nested
      write(IMAIN,*) ''
      call flush_IMAIN()
    endif
  endif
!$OMP END PARALLEL
#else
  ! nothing to do..
  return
#endif

  end subroutine prepare_openmp

!
!-------------------------------------------------------------------
!

  subroutine prepare_xsmm()

  use constants,only: CUSTOM_REAL,SIZE_DOUBLE,m1,m2,IMAIN

  use specfem_par,only: myrank

  use my_libxsmm,only: libxsmm_init,libxsmm_dispatch,libxsmm_available,xmm1,xmm2,xmm3,USE_XSMM_FUNCTION
  ! prefetch versions
  use my_libxsmm,only: xmm1p,xmm2p,xmm3p,LIBXSMM_PREFETCH,USE_XSMM_FUNCTION_PREFETCH

  implicit none

  ! quick check
  if (m1 /= 5) stop 'LibXSMM with invalid m1 constant (must have m1 == 5)'
  if (m2 /= 5*5) stop 'LibXSMM with invalid m2 constant (must have m2 == 5*5)'
  if (CUSTOM_REAL == SIZE_DOUBLE) stop 'LibXSMM optimization only for single precision functions'

  ! initializes LIBXSMM
  call libxsmm_init()

  ! dispatch functions for matrix multiplications
  ! (see in compute_forces_**Dev.F90 routines for actual function call)
  ! example: a(n1,n2),b(n2,n3),c(n1,n3) -> c = a * b then libxsmm_dispatch(xmm,m=n1,n=n3,k=n2,alpha=1,beta=0)

  ! with A(n1,n2) 5x5-matrix, B(n2,n3) 5x25-matrix and C(n1,n3) 5x25-matrix
  call libxsmm_dispatch(xmm1, m=5, n=25, k=5, alpha=1.0_CUSTOM_REAL, beta=0.0_CUSTOM_REAL)

  ! with A(n1,n2) 25x5-matrix, B(n2,n3) 5x5-matrix and C(n1,n3) 25x5-matrix
  call libxsmm_dispatch(xmm2, m=25, n=5, k=5, alpha=1.0_CUSTOM_REAL, beta=0.0_CUSTOM_REAL)

  ! with A(n1,n2,n4) 5x5x5-matrix, B(n2,n3) 5x5-matrix and C(n1,n3,n4) 5x5x5-matrix
  call libxsmm_dispatch(xmm3, m=5, n=5, k=5, alpha=1.0_CUSTOM_REAL, beta=0.0_CUSTOM_REAL)

  !directly: call libxsmm_smm_5_5_5(A,B,C)
  if (libxsmm_available(xmm1) .and. libxsmm_available(xmm2) .and. libxsmm_available(xmm3)) then
    USE_XSMM_FUNCTION = .true.
    ! user output
    if (myrank == 0) then
      write(IMAIN,*)
      write(IMAIN,*) "LIBXSMM dispatch functions ready for small matrix-matrix multiplications"
      call flush_IMAIN()
    endif
  else
    USE_XSMM_FUNCTION = .false.
    print *,'LIBXSMM invalid dispatch function pointers:', &
            libxsmm_available(xmm1),libxsmm_available(xmm2),libxsmm_available(xmm3)
    ! hard stop
    !call exit_MPI(myrank,'LIBXSMM functions not ready, please check configuration & compilation')
  endif

  ! synchronizes processes
  call synchronize_all()

  ! prefetch versions
  call libxsmm_dispatch(xmm1p, m=5, n=25, k=5, alpha=1.0_CUSTOM_REAL, beta=0.0_CUSTOM_REAL,prefetch=LIBXSMM_PREFETCH)
  call libxsmm_dispatch(xmm2p, m=25, n=5, k=5, alpha=1.0_CUSTOM_REAL, beta=0.0_CUSTOM_REAL,prefetch=LIBXSMM_PREFETCH)
  call libxsmm_dispatch(xmm3p, m=5, n=5, k=5, alpha=1.0_CUSTOM_REAL, beta=0.0_CUSTOM_REAL,prefetch=LIBXSMM_PREFETCH)

  if (libxsmm_available(xmm1p) .and. libxsmm_available(xmm2p) .and. libxsmm_available(xmm3p)) then
    USE_XSMM_FUNCTION_PREFETCH = .true.
    ! user output
    if (myrank == 0) then
      write(IMAIN,*) "LIBXSMM prefetch functions ready for small matrix-matrix multiplications"
      write(IMAIN,*)
      call flush_IMAIN()
    endif
  else
    USE_XSMM_FUNCTION_PREFETCH = .false.
    print *,'LIBXSMM invalid prefetch function pointers:', &
            libxsmm_available(xmm1p),libxsmm_available(xmm2p),libxsmm_available(xmm3p)
    ! hard stop
    !call exit_MPI(myrank,'LIBXSMM prefetch functions not ready, please check configuration & compilation')
  endif

  ! force no dispatch
  !USE_XSMM_FUNCTION = .false.
  !USE_XSMM_FUNCTION_PREFETCH = .false.

  end subroutine prepare_xsmm



!-------------------------------------------------------------------
!
! dummy routines
!
!-------------------------------------------------------------------



  subroutine compute_element_dummy(ispec,ibool,tempx1,tempx2,tempx3,tempy1,tempy2,tempy3,tempz1,tempz2,tempz3, &
                                   dummyx_loc,dummyy_loc,dummyz_loc,rho_s_H)

! dummy example (original: isotropic element in crust/mantle region)
!
! it is mostly used to avoid over-simplification of the compute_forces routine: if we omit it, then compilers can do
! much more aggressive optimizations and the timing results would be misleading. the original routines for computing
! stresses on elements are more expensive and complicated. the dummy here will be much faster to compute, but should
! give similar relative performance results

  use constants,only: CUSTOM_REAL,NGLLX,NGLLY,NGLLZ,NDIM
#ifdef FORCE_VECTORIZATION
  use constants,only: NGLLCUBE
#endif
  use specfem_par,only: NSPEC,GRAVITY_VAL
  implicit none

  ! element id
  integer,intent(in) :: ispec

  ! arrays with mesh parameters per slice
  integer, dimension(NGLLX,NGLLY,NGLLZ,NSPEC),intent(in) :: ibool

  ! element info
  real(kind=CUSTOM_REAL), dimension(NGLLX,NGLLY,NGLLZ),intent(inout) :: &
    tempx1,tempx2,tempx3,tempy1,tempy2,tempy3,tempz1,tempz2,tempz3

  real(kind=CUSTOM_REAL), dimension(NGLLX,NGLLY,NGLLZ),intent(in) :: dummyx_loc,dummyy_loc,dummyz_loc

  real(kind=CUSTOM_REAL), dimension(NGLLX,NGLLY,NGLLZ,NDIM),intent(out) :: rho_s_H

  ! local parameters
  real(kind=CUSTOM_REAL), dimension(NGLLX,NGLLY,NGLLZ) :: sigma_xx,sigma_yy,sigma_zz
  real(kind=CUSTOM_REAL), dimension(NGLLX,NGLLY,NGLLZ) :: sigma_xy,sigma_xz,sigma_yz,sigma_yx,sigma_zx,sigma_zy
  real(kind=CUSTOM_REAL) :: xixl,xiyl,xizl,etaxl,etayl,etazl,gammaxl,gammayl,gammazl
  real(kind=CUSTOM_REAL) :: fac,factor
  integer :: idummy

#ifdef FORCE_VECTORIZATION
! in this vectorized version we have to assume that N_SLS == 3 in order to be able to unroll and thus suppress
! an inner loop that would otherwise prevent vectorization; this is safe in practice in all cases because N_SLS == 3
! in all known applications, and in the main program we check that N_SLS == 3 if FORCE_VECTORIZATION is used and we stop
  integer :: ijk
#else
  integer :: i,j,k
#endif
! note: profiling shows that this routine takes about 60% of the total time, another 30% is spend in the tiso routine below..

  DO_LOOP_IJK
    ! compute stress sigma
    ! (dummy values)
    sigma_xx(INDEX_IJK) = 0.1 * dummyx_loc(INDEX_IJK)
    sigma_yy(INDEX_IJK) = 0.1 * dummyy_loc(INDEX_IJK)
    sigma_zz(INDEX_IJK) = 0.1 * dummyz_loc(INDEX_IJK)

    sigma_xy(INDEX_IJK) = 0.3 * sigma_xx(INDEX_IJK)
    sigma_xz(INDEX_IJK) = 0.3 * sigma_yy(INDEX_IJK)
    sigma_yz(INDEX_IJK) = 0.3 * sigma_zz(INDEX_IJK)
  ENDDO_LOOP_IJK

  ! define symmetric components of sigma (to be general in case of gravity)
  DO_LOOP_IJK
    sigma_yx(INDEX_IJK) = sigma_xy(INDEX_IJK)
    sigma_zx(INDEX_IJK) = sigma_xz(INDEX_IJK)
    sigma_zy(INDEX_IJK) = sigma_yz(INDEX_IJK)
  ENDDO_LOOP_IJK

  ! compute non-symmetric terms for gravity
  if (GRAVITY_VAL) then
    ! dummy example, originally calls more complicated subroutine compute_element_gravity(..)
    DO_LOOP_IJK
      ! compute G tensor from s . g and add to sigma (not symmetric)
      ! (dummy values)
      sigma_xx(INDEX_IJK) = sigma_xx(INDEX_IJK) + 1.1 ! real(sy_l*gyl + sz_l*gzl, kind=CUSTOM_REAL)
      sigma_yy(INDEX_IJK) = sigma_yy(INDEX_IJK) + 1.1 ! real(sx_l*gxl + sz_l*gzl, kind=CUSTOM_REAL)
      sigma_zz(INDEX_IJK) = sigma_zz(INDEX_IJK) + 1.1 ! real(sx_l*gxl + sy_l*gyl, kind=CUSTOM_REAL)

      sigma_xy(INDEX_IJK) = sigma_xy(INDEX_IJK) - 0.3 ! real(sx_l * gyl, kind=CUSTOM_REAL)
      sigma_yx(INDEX_IJK) = sigma_yx(INDEX_IJK) - 0.3 ! real(sy_l * gxl, kind=CUSTOM_REAL)

      sigma_xz(INDEX_IJK) = sigma_xz(INDEX_IJK) - 0.5 ! real(sx_l * gzl, kind=CUSTOM_REAL)
      sigma_zx(INDEX_IJK) = sigma_zx(INDEX_IJK) - 0.5 ! real(sz_l * gxl, kind=CUSTOM_REAL)

      sigma_yz(INDEX_IJK) = sigma_yz(INDEX_IJK) - 0.7 ! real(sy_l * gzl, kind=CUSTOM_REAL)
      sigma_zy(INDEX_IJK) = sigma_zy(INDEX_IJK) - 0.7 ! real(sz_l * gyl, kind=CUSTOM_REAL)

      ! precompute vector
      factor = 0.5 ! 0.5 * dummyz_loc(INDEX_IJK) ! dble(jacobianl(INDEX_IJK)) * wgll_cube(INDEX_IJK)

      rho_s_H(INDEX_IJK,1) = factor * 1.5 ! real(factor * (sx_l * Hxxl + sy_l * Hxyl + sz_l * Hxzl), kind=CUSTOM_REAL)
      rho_s_H(INDEX_IJK,2) = factor * 1.5 ! real(factor * (sx_l * Hxyl + sy_l * Hyyl + sz_l * Hyzl), kind=CUSTOM_REAL)
      rho_s_H(INDEX_IJK,3) = factor * 1.5 ! real(factor * (sx_l * Hxzl + sy_l * Hyzl + sz_l * Hzzl), kind=CUSTOM_REAL)
    ENDDO_LOOP_IJK
  endif

  ! dot product of stress tensor with test vector, non-symmetric form
  DO_LOOP_IJK
    ! reloads derivatives of ux, uy and uz with respect to x, y and z
    ! (dummy)
    xixl = 1.1
    xiyl = 1.2
    xizl = 1.3
    etaxl = 1.4
    etayl = 1.5
    etazl = 1.6
    gammaxl = 1.7
    gammayl = 1.8
    gammazl = 1.9

    ! common factor (dummy)
    fac = 0.5

    ! form dot product with test vector, non-symmetric form
    ! this goes to accel_x
    tempx1(INDEX_IJK) = fac * (sigma_xx(INDEX_IJK)*xixl + sigma_yx(INDEX_IJK)*xiyl + sigma_zx(INDEX_IJK)*xizl)
    ! this goes to accel_y
    tempy1(INDEX_IJK) = fac * (sigma_xy(INDEX_IJK)*xixl + sigma_yy(INDEX_IJK)*xiyl + sigma_zy(INDEX_IJK)*xizl)
    ! this goes to accel_z
    tempz1(INDEX_IJK) = fac * (sigma_xz(INDEX_IJK)*xixl + sigma_yz(INDEX_IJK)*xiyl + sigma_zz(INDEX_IJK)*xizl)

    ! this goes to accel_x
    tempx2(INDEX_IJK) = fac * (sigma_xx(INDEX_IJK)*etaxl + sigma_yx(INDEX_IJK)*etayl + sigma_zx(INDEX_IJK)*etazl)
    ! this goes to accel_y
    tempy2(INDEX_IJK) = fac * (sigma_xy(INDEX_IJK)*etaxl + sigma_yy(INDEX_IJK)*etayl + sigma_zy(INDEX_IJK)*etazl)
    ! this goes to accel_z
    tempz2(INDEX_IJK) = fac * (sigma_xz(INDEX_IJK)*etaxl + sigma_yz(INDEX_IJK)*etayl + sigma_zz(INDEX_IJK)*etazl)

    ! this goes to accel_x
    tempx3(INDEX_IJK) = fac * (sigma_xx(INDEX_IJK)*gammaxl + sigma_yx(INDEX_IJK)*gammayl + sigma_zx(INDEX_IJK)*gammazl)
    ! this goes to accel_y
    tempy3(INDEX_IJK) = fac * (sigma_xy(INDEX_IJK)*gammaxl + sigma_yy(INDEX_IJK)*gammayl + sigma_zy(INDEX_IJK)*gammazl)
    ! this goes to accel_z
    tempz3(INDEX_IJK) = fac * (sigma_xz(INDEX_IJK)*gammaxl + sigma_yz(INDEX_IJK)*gammayl + sigma_zz(INDEX_IJK)*gammazl)

  ENDDO_LOOP_IJK

  ! avoid compiler warning
  idummy = ispec
  idummy = ibool(1,1,1,1)

  end subroutine compute_element_dummy

!
!-------------------------------------------------------------------
!

  subroutine synchronize_all()

! dummy routine to make it easier for copy-paste from the original code

  implicit none

  continue

  end subroutine synchronize_all

!
!-------------------------------------------------------------------
!


  subroutine exit_MPI(myrank,error_msg)

! dummy routine to make it easier for copy-paste from the original code

  use constants

  implicit none

  ! identifier for error message file
  integer, parameter :: IERROR = 30

  integer :: myrank
  character(len=*) :: error_msg

  ! write error message to screen
  write(*,*) error_msg(1:len(error_msg))
  write(*,*) 'Error detected, aborting MPI... proc ',myrank

  ! or just exit with message:
  stop 'Error, program ended in exit_MPI'

  end subroutine exit_MPI

!
!-------------------------------------------------------------------
!

  subroutine flush_IMAIN()

! dummy routine to make it easier for copy-paste from the original code

  implicit none

  continue

  end subroutine flush_IMAIN

