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


!-------------------------------------------------------------------
!
! compute forces routine
!
!-------------------------------------------------------------------


  subroutine compute_forces_noDev()

! fortran-loops (without Deville routines) using unrolling of the inner-most loop (over 5)

  use specfem_par
  use my_libxsmm

  implicit none

  ! local parameters
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
  integer :: i,j,k

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
!$OMP hprime_xxT,hprimewgll_xx, &
!$OMP wgllwgll_xy_3D, wgllwgll_xz_3D, wgllwgll_yz_3D, &
!$OMP ibool, &
!$OMP displ,accel, &
!$OMP sum_terms ) &
!$OMP PRIVATE( ispec,ispec_p,iglob, &
!$OMP i,j,k, &
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

    do k = 1,NGLLZ
      do j = 1,NGLLY
        do i = 1,NGLLX
          iglob = ibool(i,j,k,ispec)
          dummyx_loc(i,j,k) = displ(1,iglob)
          dummyy_loc(i,j,k) = displ(2,iglob)
          dummyz_loc(i,j,k) = displ(3,iglob)
        enddo
      enddo
    enddo

    ! uses loop unrolling for NGLLX == NGLLY == NGLLZ == 5
    do k = 1,NGLLZ
      do j = 1,NGLLY
        do i = 1,NGLLX
          ! general NGLLX == NGLLY == NGLLZ
          !tempx1l = 0._CUSTOM_REAL
          !tempx2l = 0._CUSTOM_REAL
          !tempx3l = 0._CUSTOM_REAL
          !tempy1l = 0._CUSTOM_REAL
          !tempy2l = 0._CUSTOM_REAL
          !tempy3l = 0._CUSTOM_REAL
          !tempz1l = 0._CUSTOM_REAL
          !tempz2l = 0._CUSTOM_REAL
          !tempz3l = 0._CUSTOM_REAL
          !do l = 1,NGLLX
          !  tempx1l = tempx1l + dummyx_loc(l,j,k)*hprime_xx(i,l)
          !  tempy1l = tempy1l + dummyy_loc(l,j,k)*hprime_xx(i,l)
          !  tempz1l = tempz1l + dummyz_loc(l,j,k)*hprime_xx(i,l)
          !  !!! can merge these loops because NGLLX = NGLLY = NGLLZ
          !  tempx2l = tempx2l + dummyx_loc(i,l,k)*hprime_yy(j,l)
          !  tempy2l = tempy2l + dummyy_loc(i,l,k)*hprime_yy(j,l)
          !  tempz2l = tempz2l + dummyz_loc(i,l,k)*hprime_yy(j,l)
          !  !!! can merge these loops because NGLLX = NGLLY = NGLLZ
          !  tempx3l = tempx3l + dummyx_loc(i,j,l)*hprime_zz(k,l)
          !  tempy3l = tempy3l + dummyy_loc(i,j,l)*hprime_zz(k,l)
          !  tempz3l = tempz3l + dummyz_loc(i,j,l)*hprime_zz(k,l)
          !enddo

          ! unrolled
          tempx1(i,j,k) = dummyx_loc(1,j,k)*hprime_xxT(1,i) &
                        + dummyx_loc(2,j,k)*hprime_xxT(2,i) &
                        + dummyx_loc(3,j,k)*hprime_xxT(3,i) &
                        + dummyx_loc(4,j,k)*hprime_xxT(4,i) &
                        + dummyx_loc(5,j,k)*hprime_xxT(5,i)
          tempy1(i,j,k) = dummyy_loc(1,j,k)*hprime_xxT(1,i) &
                        + dummyy_loc(2,j,k)*hprime_xxT(2,i) &
                        + dummyy_loc(3,j,k)*hprime_xxT(3,i) &
                        + dummyy_loc(4,j,k)*hprime_xxT(4,i) &
                        + dummyy_loc(5,j,k)*hprime_xxT(5,i)
          tempz1(i,j,k) = dummyz_loc(1,j,k)*hprime_xxT(1,i) &
                        + dummyz_loc(2,j,k)*hprime_xxT(2,i) &
                        + dummyz_loc(3,j,k)*hprime_xxT(3,i) &
                        + dummyz_loc(4,j,k)*hprime_xxT(4,i) &
                        + dummyz_loc(5,j,k)*hprime_xxT(5,i)
          !!! can merge these loops because NGLLX = NGLLY = NGLLZ
          tempx2(i,j,k) = dummyx_loc(i,1,k)*hprime_xxT(1,j) &
                        + dummyx_loc(i,2,k)*hprime_xxT(2,j) &
                        + dummyx_loc(i,3,k)*hprime_xxT(3,j) &
                        + dummyx_loc(i,4,k)*hprime_xxT(4,j) &
                        + dummyx_loc(i,5,k)*hprime_xxT(5,j)
          tempy2(i,j,k) = dummyy_loc(i,1,k)*hprime_xxT(1,j) &
                        + dummyy_loc(i,2,k)*hprime_xxT(2,j) &
                        + dummyy_loc(i,3,k)*hprime_xxT(3,j) &
                        + dummyy_loc(i,4,k)*hprime_xxT(4,j) &
                        + dummyy_loc(i,5,k)*hprime_xxT(5,j)
          tempz2(i,j,k) = dummyz_loc(i,1,k)*hprime_xxT(1,j) &
                        + dummyz_loc(i,2,k)*hprime_xxT(2,j) &
                        + dummyz_loc(i,3,k)*hprime_xxT(3,j) &
                        + dummyz_loc(i,4,k)*hprime_xxT(4,j) &
                        + dummyz_loc(i,5,k)*hprime_xxT(5,j)
          !!! can merge these loops because NGLLX = NGLLY = NGLLZ
          tempx3(i,j,k) = dummyx_loc(i,j,1)*hprime_xxT(1,k) &
                        + dummyx_loc(i,j,2)*hprime_xxT(2,k) &
                        + dummyx_loc(i,j,3)*hprime_xxT(3,k) &
                        + dummyx_loc(i,j,4)*hprime_xxT(4,k) &
                        + dummyx_loc(i,j,5)*hprime_xxT(5,k)
          tempy3(i,j,k) = dummyy_loc(i,j,1)*hprime_xxT(1,k) &
                        + dummyy_loc(i,j,2)*hprime_xxT(2,k) &
                        + dummyy_loc(i,j,3)*hprime_xxT(3,k) &
                        + dummyy_loc(i,j,4)*hprime_xxT(4,k) &
                        + dummyy_loc(i,j,5)*hprime_xxT(5,k)
          tempz3(i,j,k) = dummyz_loc(i,j,1)*hprime_xxT(1,k) &
                        + dummyz_loc(i,j,2)*hprime_xxT(2,k) &
                        + dummyz_loc(i,j,3)*hprime_xxT(3,k) &
                        + dummyz_loc(i,j,4)*hprime_xxT(4,k) &
                        + dummyz_loc(i,j,5)*hprime_xxT(5,k)
        enddo
      enddo
    enddo

    call compute_element_dummy(ispec,ibool,tempx1,tempx2,tempx3,tempy1,tempy2,tempy3,tempz1,tempz2,tempz3, &
                               dummyx_loc,dummyy_loc,dummyz_loc,rho_s_H)

    ! uses loop unrolling for NGLLX == NGLLY == NGLLZ == 5
    do k = 1,NGLLZ
      do j = 1,NGLLY
        do i = 1,NGLLX
          ! general NGLLX == NGLLY == NGLLZ
          !tempx1l = 0._CUSTOM_REAL
          !tempx2l = 0._CUSTOM_REAL
          !tempx3l = 0._CUSTOM_REAL
          !tempy1l = 0._CUSTOM_REAL
          !tempy2l = 0._CUSTOM_REAL
          !tempy3l = 0._CUSTOM_REAL
          !tempz1l = 0._CUSTOM_REAL
          !tempz2l = 0._CUSTOM_REAL
          !tempz3l = 0._CUSTOM_REAL
          !do l = 1,NGLLX
          !  fac1 = hprimewgll_xx(l,i)
          !  tempx1l = tempx1l + tempx1(l,j,k)*fac1
          !  tempy1l = tempy1l + tempy1(l,j,k)*fac1
          !  tempz1l = tempz1l + tempz1(l,j,k)*fac1
          !  !!! can merge these loops because NGLLX = NGLLY = NGLLZ
          !  fac2 = hprimewgll_yy(l,j)
          !  tempx2l = tempx2l + tempx2(i,l,k)*fac2
          !  tempy2l = tempy2l + tempy2(i,l,k)*fac2
          !  tempz2l = tempz2l + tempz2(i,l,k)*fac2
          !  !!! can merge these loops because NGLLX = NGLLY = NGLLZ
          !  fac3 = hprimewgll_zz(l,k)
          !  tempx3l = tempx3l + tempx3(i,j,l)*fac3
          !  tempy3l = tempy3l + tempy3(i,j,l)*fac3
          !  tempz3l = tempz3l + tempz3(i,j,l)*fac3
          !enddo

          ! unrolled
          newtempx1(i,j,k) = tempx1(1,j,k)*hprimewgll_xx(1,i) &
                           + tempx1(2,j,k)*hprimewgll_xx(2,i) &
                           + tempx1(3,j,k)*hprimewgll_xx(3,i) &
                           + tempx1(4,j,k)*hprimewgll_xx(4,i) &
                           + tempx1(5,j,k)*hprimewgll_xx(5,i)
          newtempy1(i,j,k) = tempy1(1,j,k)*hprimewgll_xx(1,i) &
                           + tempy1(2,j,k)*hprimewgll_xx(2,i) &
                           + tempy1(3,j,k)*hprimewgll_xx(3,i) &
                           + tempy1(4,j,k)*hprimewgll_xx(4,i) &
                           + tempy1(5,j,k)*hprimewgll_xx(5,i)
          newtempz1(i,j,k) = tempz1(1,j,k)*hprimewgll_xx(1,i) &
                           + tempz1(2,j,k)*hprimewgll_xx(2,i) &
                           + tempz1(3,j,k)*hprimewgll_xx(3,i) &
                           + tempz1(4,j,k)*hprimewgll_xx(4,i) &
                           + tempz1(5,j,k)*hprimewgll_xx(5,i)
          !!! can merge these loops because NGLLX = NGLLY = NGLLZ
          newtempx2(i,j,k) = tempx2(i,1,k)*hprimewgll_xx(1,j) &
                           + tempx2(i,2,k)*hprimewgll_xx(2,j) &
                           + tempx2(i,3,k)*hprimewgll_xx(3,j) &
                           + tempx2(i,4,k)*hprimewgll_xx(4,j) &
                           + tempx2(i,5,k)*hprimewgll_xx(5,j)
          newtempy2(i,j,k) = tempy2(i,1,k)*hprimewgll_xx(1,j) &
                           + tempy2(i,2,k)*hprimewgll_xx(2,j) &
                           + tempy2(i,3,k)*hprimewgll_xx(3,j) &
                           + tempy2(i,4,k)*hprimewgll_xx(4,j) &
                           + tempy2(i,5,k)*hprimewgll_xx(5,j)
          newtempz2(i,j,k) = tempz2(i,1,k)*hprimewgll_xx(1,j) &
                           + tempz2(i,2,k)*hprimewgll_xx(2,j) &
                           + tempz2(i,3,k)*hprimewgll_xx(3,j) &
                           + tempz2(i,4,k)*hprimewgll_xx(4,j) &
                           + tempz2(i,5,k)*hprimewgll_xx(5,j)
          !!! can merge these loops because NGLLX = NGLLY = NGLLZ
          newtempx3(i,j,k) = tempx3(i,j,1)*hprimewgll_xx(1,k) &
                           + tempx3(i,j,2)*hprimewgll_xx(2,k) &
                           + tempx3(i,j,3)*hprimewgll_xx(3,k) &
                           + tempx3(i,j,4)*hprimewgll_xx(4,k) &
                           + tempx3(i,j,5)*hprimewgll_xx(5,k)
          newtempy3(i,j,k) = tempy3(i,j,1)*hprimewgll_xx(1,k) &
                           + tempy3(i,j,2)*hprimewgll_xx(2,k) &
                           + tempy3(i,j,3)*hprimewgll_xx(3,k) &
                           + tempy3(i,j,4)*hprimewgll_xx(4,k) &
                           + tempy3(i,j,5)*hprimewgll_xx(5,k)
          newtempz3(i,j,k) = tempz3(i,j,1)*hprimewgll_xx(1,k) &
                           + tempz3(i,j,2)*hprimewgll_xx(2,k) &
                           + tempz3(i,j,3)*hprimewgll_xx(3,k) &
                           + tempz3(i,j,4)*hprimewgll_xx(4,k) &
                           + tempz3(i,j,5)*hprimewgll_xx(5,k)
        enddo
      enddo
    enddo

    ! sums contributions
    do k = 1,NGLLZ
      do j = 1,NGLLY
        do i = 1,NGLLX
          fac1 = wgllwgll_yz_3D(i,j,k)
          fac2 = wgllwgll_xz_3D(i,j,k)
          fac3 = wgllwgll_xy_3D(i,j,k)
          sum_terms(1,i,j,k,ispec) = - (fac1*newtempx1(i,j,k) + fac2*newtempx2(i,j,k) + fac3*newtempx3(i,j,k))
          sum_terms(2,i,j,k,ispec) = - (fac1*newtempy1(i,j,k) + fac2*newtempy2(i,j,k) + fac3*newtempy3(i,j,k))
          sum_terms(3,i,j,k,ispec) = - (fac1*newtempz1(i,j,k) + fac2*newtempz2(i,j,k) + fac3*newtempz3(i,j,k))
        enddo
      enddo
    enddo

    ! adds gravity terms
    if (GRAVITY_VAL) then
      do k = 1,NGLLZ
        do j = 1,NGLLY
          do i = 1,NGLLX
            sum_terms(1,i,j,k,ispec) = sum_terms(1,i,j,k,ispec) + rho_s_H(i,j,k,1)
            sum_terms(2,i,j,k,ispec) = sum_terms(2,i,j,k,ispec) + rho_s_H(i,j,k,2)
            sum_terms(3,i,j,k,ispec) = sum_terms(3,i,j,k,ispec) + rho_s_H(i,j,k,3)
          enddo
        enddo
      enddo
    endif

    ! updates acceleration
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
    do k = 1,NGLLZ
      do j = 1,NGLLY
        do i = 1,NGLLX
          iglob = ibool(i,j,k,ispec)
#ifdef USE_OPENMP_ATOMIC_INSTEAD_OF_CRITICAL
#ifdef USE_OPENMP
!$OMP ATOMIC
#endif
#endif
          accel(1,iglob) = accel(1,iglob) + sum_terms(1,i,j,k,ispec)
#ifdef USE_OPENMP_ATOMIC_INSTEAD_OF_CRITICAL
#ifdef USE_OPENMP
!$OMP ATOMIC
#endif
#endif
          accel(2,iglob) = accel(2,iglob) + sum_terms(2,i,j,k,ispec)
#ifdef USE_OPENMP_ATOMIC_INSTEAD_OF_CRITICAL
#ifdef USE_OPENMP
!$OMP ATOMIC
#endif
#endif
          accel(3,iglob) = accel(3,iglob) + sum_terms(3,i,j,k,ispec)
        enddo
      enddo
    enddo
#ifndef USE_OPENMP_ATOMIC_INSTEAD_OF_CRITICAL
#ifdef USE_OPENMP
!$OMP END CRITICAL
#endif
#endif

  enddo ! ispec
#ifdef USE_OPENMP
!$OMP enddo

!$OMP END PARALLEL
#endif

  end subroutine compute_forces_noDev

