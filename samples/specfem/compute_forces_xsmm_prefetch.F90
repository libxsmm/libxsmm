!=====================================================================
!
!          S p e c f e m 3 D  G l o b e  V e r s i o n  7 . 0
!          --------------------------------------------------
!
!     Main historical authors: Dimitri Komatitsch and Jeroen Tromp
!                        Princeton University, USA
!                and CNRS / University of Marseille, France
!                 (there are currently many more authors!)
! (c) Princeton University and CNRS / University of Marseille, April 2014
!
! This program is free software; you can redistribute it and/or modify
! it under the terms of the GNU General Public License as published by
! the Free Software Foundation; either version 2 of the License, or
! (at your option) any later version.
!
! This program is distributed in the hope that it will be useful,
! but WITHOUT ANY WARRANTY; without even the implied warranty of
! MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
! GNU General Public License for more details.
!
! You should have received a copy of the GNU General Public License along
! with this program; if not, write to the Free Software Foundation, Inc.,
! 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
!
!=====================================================================

! we switch between vectorized and non-vectorized version by using pre-processor flag FORCE_VECTORIZATION
! and macros INDEX_IJK, DO_LOOP_IJK, ENDDO_LOOP_IJK defined in config.fh
#include "config.fh"

!-------------------------------------------------------------------
!
! compute forces routine
!
!-------------------------------------------------------------------

  subroutine compute_forces_with_xsmm_prefetch()

! uses LIBXSMM dispatched functions with prefetch versions
! (based on Deville version compute_forces_Dev.F90)

  use specfem_par
  use my_libxsmm

  implicit none

  ! Deville
  ! manually inline the calls to the Deville et al. (2002) routines
  real(kind=CUSTOM_REAL), dimension(NGLLX,NGLLY,NGLLZ) :: &
    tempx1,tempx2,tempx3,tempy1,tempy2,tempy3,tempz1,tempz2,tempz3

  real(kind=CUSTOM_REAL), dimension(NGLLX,NGLLY,NGLLZ) :: &
    newtempx1,newtempx2,newtempx3,newtempy1,newtempy2,newtempy3,newtempz1,newtempz2,newtempz3

  real(kind=CUSTOM_REAL), dimension(NGLLX,NGLLY,NGLLZ) :: dummyx_loc,dummyy_loc,dummyz_loc

  real(kind=CUSTOM_REAL) :: fac1,fac2,fac3

  ! for gravity
  real(kind=CUSTOM_REAL), dimension(NGLLX,NGLLY,NGLLZ,NDIM) :: rho_s_H

  integer :: num_elements,ispec_p
  integer :: ispec,iglob
#ifdef FORCE_VECTORIZATION
  integer :: ijk_spec,ip,iglob_p,ijk
#else
  integer :: i,j,k
#endif

! ****************************************************
!   big loop over all spectral elements in the solid
! ****************************************************

!  computed_elements = 0
  if (iphase == 1) then
    ! outer elements (halo region)
    num_elements = nspec_outer
  else
    ! inner elements
    num_elements = nspec_inner
  endif

#ifdef USE_OPENMP
!$OMP PARALLEL DEFAULT(NONE) &
!$OMP SHARED( &
!$OMP num_elements,iphase,phase_ispec_inner, &
!$OMP hprime_xxT,hprime_xx,hprimewgll_xx,hprimewgll_xxT, &
!$OMP wgllwgll_xy_3D, wgllwgll_xz_3D, wgllwgll_yz_3D, &
#ifdef FORCE_VECTORIZATION
!$OMP ibool_inv_tbl, ibool_inv_st, num_globs, phase_iglob, &
#endif
!$OMP ibool, &
!$OMP displ,accel, &
!$OMP sum_terms ) &
!$OMP PRIVATE( ispec,ispec_p,iglob, &
#ifdef FORCE_VECTORIZATION
!$OMP ijk_spec,ip,iglob_p, &
!$OMP ijk, &
#else
!$OMP i,j,k, &
#endif
!$OMP fac1,fac2,fac3, &
!$OMP tempx1,tempx2,tempx3,tempy1,tempy2,tempy3,tempz1,tempz2,tempz3, &
!$OMP newtempx1,newtempx2,newtempx3,newtempy1,newtempy2,newtempy3,newtempz1,newtempz2,newtempz3, &
!$OMP dummyx_loc,dummyy_loc,dummyz_loc, &
!$OMP rho_s_H )
#endif

  ! loops over all spectral-elements
#ifdef USE_OPENMP
!$OMP DO SCHEDULE(GUIDED)
#endif
  do ispec_p = 1,num_elements

    ! only compute elements which belong to current phase (inner or outer elements)
    ispec = phase_ispec_inner(ispec_p,iphase)

    DO_LOOP_IJK
      iglob = ibool(INDEX_IJK,ispec)
      dummyx_loc(INDEX_IJK) = displ(1,iglob)
      dummyy_loc(INDEX_IJK) = displ(2,iglob)
      dummyz_loc(INDEX_IJK) = displ(3,iglob)
    ENDDO_LOOP_IJK

    ! subroutines adapted from Deville, Fischer and Mund, High-order methods
    ! for incompressible fluid flow, Cambridge University Press (2002),
    ! pages 386 and 389 and Figure 8.3.1

    ! computes 1. matrix multiplication for tempx1,..
    call mxm5_3comp_singleA(hprime_xx,m1,dummyx_loc,dummyy_loc,dummyz_loc,tempx1,tempy1,tempz1,m2)
    ! computes 2. matrix multiplication for tempx2,..
    call mxm5_3comp_3dmat_singleB(dummyx_loc,dummyy_loc,dummyz_loc,m1,hprime_xxT,m1,tempx2,tempy2,tempz2,NGLLX)
    ! computes 3. matrix multiplication for tempx3,..
    call mxm5_3comp_singleB(dummyx_loc,dummyy_loc,dummyz_loc,m2,hprime_xxT,tempx3,tempy3,tempz3,m1)

    call compute_element_dummy(ispec,ibool,tempx1,tempx2,tempx3,tempy1,tempy2,tempy3,tempz1,tempz2,tempz3, &
                               dummyx_loc,dummyy_loc,dummyz_loc,rho_s_H)

    ! subroutines adapted from Deville, Fischer and Mund, High-order methods
    ! for incompressible fluid flow, Cambridge University Press (2002),
    ! pages 386 and 389 and Figure 8.3.1

    ! computes 1. matrix multiplication for newtempx1,..
    call mxm5_3comp_singleA(hprimewgll_xxT,m1,tempx1,tempy1,tempz1,newtempx1,newtempy1,newtempz1,m2)
    ! computes 2. matrix multiplication for tempx2,..
    call mxm5_3comp_3dmat_singleB(tempx2,tempy2,tempz2,m1,hprimewgll_xx,m1,newtempx2,newtempy2,newtempz2,NGLLX)
    ! computes 3. matrix multiplication for newtempx3,..
    call mxm5_3comp_singleB(tempx3,tempy3,tempz3,m2,hprimewgll_xx,newtempx3,newtempy3,newtempz3,m1)

    ! sums contributions
    DO_LOOP_IJK
      fac1 = wgllwgll_yz_3D(INDEX_IJK)
      fac2 = wgllwgll_xz_3D(INDEX_IJK)
      fac3 = wgllwgll_xy_3D(INDEX_IJK)
      sum_terms(1,INDEX_IJK,ispec) = - (fac1*newtempx1(INDEX_IJK) + fac2*newtempx2(INDEX_IJK) + fac3*newtempx3(INDEX_IJK))
      sum_terms(2,INDEX_IJK,ispec) = - (fac1*newtempy1(INDEX_IJK) + fac2*newtempy2(INDEX_IJK) + fac3*newtempy3(INDEX_IJK))
      sum_terms(3,INDEX_IJK,ispec) = - (fac1*newtempz1(INDEX_IJK) + fac2*newtempz2(INDEX_IJK) + fac3*newtempz3(INDEX_IJK))
    ENDDO_LOOP_IJK

    ! adds gravity terms
    if (GRAVITY_VAL) then
#ifdef FORCE_VECTORIZATION
      do ijk = 1,NDIM*NGLLCUBE
        sum_terms(ijk,1,1,1,ispec) = sum_terms(ijk,1,1,1,ispec) + rho_s_H(ijk,1,1,1)
      enddo
#else
      do k = 1,NGLLZ
        do j = 1,NGLLY
          do i = 1,NGLLX
            sum_terms(1,i,j,k,ispec) = sum_terms(1,i,j,k,ispec) + rho_s_H(i,j,k,1)
            sum_terms(2,i,j,k,ispec) = sum_terms(2,i,j,k,ispec) + rho_s_H(i,j,k,2)
            sum_terms(3,i,j,k,ispec) = sum_terms(3,i,j,k,ispec) + rho_s_H(i,j,k,3)
          enddo
        enddo
      enddo
#endif
    endif

    ! updates acceleration
#ifdef FORCE_VECTORIZATION
    ! update will be done later at the very end..
#else
    ! updates for non-vectorization case

! note: Critical OpenMP here might degrade performance,
!       especially for a larger number of threads (>8).
!       Using atomic operations can partially help.
#ifndef USE_OPENMP_ATOMIC_INSTEAD_OF_CRITICAL
#ifdef USE_OPENMP
!$OMP CRITICAL
#endif
#endif
! we can force vectorization using a compiler directive here because we know that there is no dependency
! inside a given spectral element, since all the global points of a local elements are different by definition
! (only common points between different elements can be the same)
! IBM, Portland PGI, and Intel and Cray syntax (Intel and Cray are the same)
!IBM* ASSERT (NODEPS)
!pgi$ ivdep
!DIR$ IVDEP
    DO_LOOP_IJK
      iglob = ibool(INDEX_IJK,ispec)
#ifdef USE_OPENMP_ATOMIC_INSTEAD_OF_CRITICAL
#ifdef USE_OPENMP
!$OMP ATOMIC
#endif
#endif
      accel(1,iglob) = accel(1,iglob) + sum_terms(1,INDEX_IJK,ispec)
#ifdef USE_OPENMP_ATOMIC_INSTEAD_OF_CRITICAL
#ifdef USE_OPENMP
!$OMP ATOMIC
#endif
#endif
      accel(2,iglob) = accel(2,iglob) + sum_terms(2,INDEX_IJK,ispec)
#ifdef USE_OPENMP_ATOMIC_INSTEAD_OF_CRITICAL
#ifdef USE_OPENMP
!$OMP ATOMIC
#endif
#endif
      accel(3,iglob) = accel(3,iglob) + sum_terms(3,INDEX_IJK,ispec)
    ENDDO_LOOP_IJK
#ifndef USE_OPENMP_ATOMIC_INSTEAD_OF_CRITICAL
#ifdef USE_OPENMP
!$OMP END CRITICAL
#endif
#endif
#endif

  enddo ! ispec
#ifdef USE_OPENMP
!$OMP enddo
#endif

  ! updates acceleration
#ifdef FORCE_VECTORIZATION
  ! updates for vectorized case
  ! loops over all global nodes in this phase (inner/outer)
#ifdef USE_OPENMP
!$OMP DO
#endif
  do iglob_p = 1,num_globs(iphase)
    ! global node index
    iglob = phase_iglob(iglob_p,iphase)
    ! loops over valence points
    do ip = ibool_inv_st(iglob_p,iphase),ibool_inv_st(iglob_p+1,iphase)-1
      ! local 1D index from array ibool
      ijk_spec = ibool_inv_tbl(ip,iphase)

      ! do NOT use array syntax ":" for the three statements below otherwise most compilers
      ! will not be able to vectorize the outer loop
      accel(1,iglob) = accel(1,iglob) + sum_terms(1,ijk_spec,1,1,1)
      accel(2,iglob) = accel(2,iglob) + sum_terms(2,ijk_spec,1,1,1)
      accel(3,iglob) = accel(3,iglob) + sum_terms(3,ijk_spec,1,1,1)
    enddo
  enddo
#ifdef USE_OPENMP
!$OMP enddo
#endif
#endif

#ifdef USE_OPENMP
!$OMP END PARALLEL
#endif

  contains

!--------------------------------------------------------------------------------------------
!
! matrix-matrix multiplications
!
! subroutines adapted from Deville, Fischer and Mund, High-order methods
! for incompressible fluid flow, Cambridge University Press (2002),
! pages 386 and 389 and Figure 8.3.1
!
!--------------------------------------------------------------------------------------------
!
! note: the matrix-matrix multiplications are used for very small matrices ( 5 x 5 x 5 elements);
!       thus, calling external optimized libraries for these multiplications are in general slower
!
! please leave the routines here to help compilers inlining the code

  subroutine mxm5_3comp_singleA(A,n1,B1,B2,B3,C1,C2,C3,n3)

! 3 different arrays for x/y/z-components, 2-dimensional arrays (25,5)/(5,25), same B matrix for all 3 component arrays

#ifdef XSMM
  use my_libxsmm,only: USE_XSMM_FUNCTION_PREFETCH, xmm1, xmm1p, &
                       libxsmm_mmcall_abc => libxsmm_smmcall_abc, &
                       libxsmm_mmcall_prf => libxsmm_smmcall_prf
  ! debug timing
  !use my_libxsmm,only: libxsmm_timer_tick,libxsmm_timer_duration
#endif

  implicit none

  integer,intent(in) :: n1,n3

  real(kind=CUSTOM_REAL),dimension(n1,5),intent(in),target :: A
  real(kind=CUSTOM_REAL),dimension(5,n3),intent(in),target :: B1,B2,B3
  real(kind=CUSTOM_REAL),dimension(n1,n3),intent(out),target :: C1,C2,C3

  ! local parameters
  integer :: i,j

#ifdef XSMM
  ! debug timing
  !double precision :: duration
  !integer(kind=8) :: start

  ! debug timing
  !start = libxsmm_timer_tick()

  ! matrix-matrix multiplication C = alpha A * B + beta C
  ! with A(n1,n2) 5x5-matrix, B(n2,n3) 5x25-matrix and C(n1,n3) 5x25-matrix
  if (USE_XSMM_FUNCTION_PREFETCH) then
    ! prefetch version
    call libxsmm_mmcall_prf(xmm1p, a=A, b=B1, c=C1, &
                            pa=A, pb=B2, pc=C2) ! with prefetch
    call libxsmm_mmcall_prf(xmm1p, a=A, b=B2, c=C2, &
                            pa=A, pb=B3, pc=C3) ! with prefetch
    call libxsmm_mmcall_abc(xmm1, a=A, b=B3, c=C3)
    !call libxsmm_mmcall_prf(xmm1p, a=A, b=B3, c=C3, &
                            !pa=A, pb=B1, pc=C1) ! with dummy prefetch

    ! debug timing
    !duration = libxsmm_timer_duration(start, libxsmm_timer_tick())
    !print *,'duration: ',duration

    ! debug
    !do j = 1,n3
    !  do i = 1,n1
    !    print *,i,j,'debug xsmm',C1(i,j),C2(i,j),C1(i,j) - C2(i,j)
    !  enddo
    !enddo
    !stop 'test stop'

    return
  endif
#endif

  ! matrix-matrix multiplication
  do j = 1,n3
!dir$ ivdep
    do i = 1,n1
      C1(i,j) =  A(i,1) * B1(1,j) &
               + A(i,2) * B1(2,j) &
               + A(i,3) * B1(3,j) &
               + A(i,4) * B1(4,j) &
               + A(i,5) * B1(5,j)

      C2(i,j) =  A(i,1) * B2(1,j) &
               + A(i,2) * B2(2,j) &
               + A(i,3) * B2(3,j) &
               + A(i,4) * B2(4,j) &
               + A(i,5) * B2(5,j)

      C3(i,j) =  A(i,1) * B3(1,j) &
               + A(i,2) * B3(2,j) &
               + A(i,3) * B3(3,j) &
               + A(i,4) * B3(4,j) &
               + A(i,5) * B3(5,j)
    enddo
  enddo

  end subroutine mxm5_3comp_singleA


!--------------------------------------------------------------------------------------------

  subroutine mxm5_3comp_singleB(A1,A2,A3,n1,B,C1,C2,C3,n3)

! 3 different arrays for x/y/z-components, 2-dimensional arrays (25,5)/(5,25), same B matrix for all 3 component arrays

#ifdef XSMM
  use my_libxsmm,only: USE_XSMM_FUNCTION_PREFETCH, xmm2, xmm2p, &
                       libxsmm_mmcall_abc => libxsmm_smmcall_abc, &
                       libxsmm_mmcall_prf => libxsmm_smmcall_prf
#endif

  implicit none

  integer,intent(in) :: n1,n3

  real(kind=CUSTOM_REAL),dimension(n1,5),intent(in),target :: A1,A2,A3
  real(kind=CUSTOM_REAL),dimension(5,n3),intent(in),target :: B
  real(kind=CUSTOM_REAL),dimension(n1,n3),intent(out),target :: C1,C2,C3

  ! local parameters
  integer :: i,j

#ifdef XSMM
  ! matrix-matrix multiplication C = alpha A * B + beta C
  ! with A(n1,n2) 25x5-matrix, B(n2,n3) 5x5-matrix and C(n1,n3) 25x5-matrix
  if (USE_XSMM_FUNCTION_PREFETCH) then
    ! prefetch version
    call libxsmm_mmcall_prf(xmm2p, a=A1, b=B, c=C1, &
                            pa=A2, pb=B, pc=C2) ! with prefetch
    call libxsmm_mmcall_prf(xmm2p, a=A2, b=B, c=C2, &
                            pa=A3, pb=B, pc=C3) ! with prefetch
    call libxsmm_mmcall_abc(xmm2, a=A3, b=B, c=C3)
    !call libxsmm_mmcall_prf(xmm2p, a=A3, b=B, c=C3, &
                            !pa=A1, pb=B, pc=C1) ! with dummy prefetch
    return
  endif
#endif

  ! matrix-matrix multiplication
  do j = 1,n3
!dir$ ivdep
    do i = 1,n1
      C1(i,j) =  A1(i,1) * B(1,j) &
               + A1(i,2) * B(2,j) &
               + A1(i,3) * B(3,j) &
               + A1(i,4) * B(4,j) &
               + A1(i,5) * B(5,j)

      C2(i,j) =  A2(i,1) * B(1,j) &
               + A2(i,2) * B(2,j) &
               + A2(i,3) * B(3,j) &
               + A2(i,4) * B(4,j) &
               + A2(i,5) * B(5,j)

      C3(i,j) =  A3(i,1) * B(1,j) &
               + A3(i,2) * B(2,j) &
               + A3(i,3) * B(3,j) &
               + A3(i,4) * B(4,j) &
               + A3(i,5) * B(5,j)
    enddo
  enddo

  end subroutine mxm5_3comp_singleB


!--------------------------------------------------------------------------------------------

  subroutine mxm5_3comp_3dmat_singleB(A1,A2,A3,n1,B,n2,C1,C2,C3,n3)

! 3 different arrays for x/y/z-components, 3-dimensional arrays (5,5,5), same B matrix for all 3 component arrays

#ifdef XSMM
  use my_libxsmm,only: USE_XSMM_FUNCTION_PREFETCH, xmm3, xmm3p, &
                       libxsmm_mmcall_abc => libxsmm_smmcall_abc, &
                       libxsmm_mmcall_prf => libxsmm_smmcall_prf
#endif

  implicit none

  integer,intent(in) :: n1,n2,n3

  real(kind=CUSTOM_REAL),dimension(n1,5,n3),intent(in),target :: A1,A2,A3
  real(kind=CUSTOM_REAL),dimension(5,n2),intent(in),target :: B
  real(kind=CUSTOM_REAL),dimension(n1,n2,n3),intent(out),target :: C1,C2,C3

  ! local parameters
  integer :: i,j,k

#ifdef XSMM
  ! matrix-matrix multiplication C = alpha A * B + beta C
  ! with A(n1,n2,n4) 5x5x5-matrix, B(n2,n3) 5x5-matrix and C(n1,n3,n4) 5x5x5-matrix
  if (USE_XSMM_FUNCTION_PREFETCH) then
    do k = 1,5
      ! prefetch version
      call libxsmm_mmcall_prf(xmm3p, a=A1(1,1,k), b=B, c=C1(1,1,k), &
                              pa=A2(1,1,k), pb=B, pc=C2(1,1,k)) ! with prefetch
      call libxsmm_mmcall_prf(xmm3p, a=A2(1,1,k), b=B, c=C2(1,1,k), &
                              pa=A3(1,1,k), pb=B, pc=C3(1,1,k)) ! with prefetch

      !if (k == 5) then
        call libxsmm_mmcall_abc(xmm3, a=A3(1,1,k), b=B, c=C3(1,1,k))
      !else
      !  call libxsmm_mmcall_prf(xmm3p, a=A3(1,1,k), b=B, c=C3(1,1,k), &
      !                      pa=A1(1,1,k+1), pb=B, pc=C1(1,1,k+1)) ! with dummy prefetch
      !endif
    enddo
    return
  endif
#endif

  ! matrix-matrix multiplication
  do k = 1,n3
    do j = 1,n2
!dir$ ivdep
      do i = 1,n1
        C1(i,j,k) =  A1(i,1,k) * B(1,j) &
                   + A1(i,2,k) * B(2,j) &
                   + A1(i,3,k) * B(3,j) &
                   + A1(i,4,k) * B(4,j) &
                   + A1(i,5,k) * B(5,j)

        C2(i,j,k) =  A2(i,1,k) * B(1,j) &
                   + A2(i,2,k) * B(2,j) &
                   + A2(i,3,k) * B(3,j) &
                   + A2(i,4,k) * B(4,j) &
                   + A2(i,5,k) * B(5,j)

        C3(i,j,k) =  A3(i,1,k) * B(1,j) &
                   + A3(i,2,k) * B(2,j) &
                   + A3(i,3,k) * B(3,j) &
                   + A3(i,4,k) * B(4,j) &
                   + A3(i,5,k) * B(5,j)
      enddo
    enddo
  enddo

  end subroutine mxm5_3comp_3dmat_singleB

  end subroutine compute_forces_with_xsmm_prefetch

