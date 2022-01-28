!=======================================================================!
! Copyright (c) Intel Corporation - All rights reserved.                !
! This file is part of the LIBXSMM library.                             !
!                                                                       !
! For information on the license, see the LICENSE file.                 !
! Further information: https://github.com/libxsmm/libxsmm/              !
! SPDX-License-Identifier: BSD-3-Clause                                 !
!=======================================================================!
! Hans Pabst (Intel Corp.)
!=======================================================================!
      PROGRAM hello
        USE :: LIBXSMM, ONLY: LIBXSMM_BLASINT_KIND,                     &
     &                        LIBXSMM_MMFUNCTION => LIBXSMM_DMMFUNCTION,&
     &                        libxsmm_mmdispatch => libxsmm_dmmdispatch,&
     &                        libxsmm_mmcall => libxsmm_dmmcall
        IMPLICIT NONE
        INTEGER, PARAMETER :: T = KIND(0D0)
        INTEGER :: batchsize = 1000, i
        INTEGER(LIBXSMM_BLASINT_KIND) :: j, ki
        INTEGER(LIBXSMM_BLASINT_KIND) :: m = 13, n = 5, k = 7
        REAL(T), ALLOCATABLE :: a(:,:,:), b(:,:,:), c(:,:)
        TYPE(LIBXSMM_MMFUNCTION) :: xmm

        ALLOCATE(a(m,k,batchsize), b(k,n,batchsize), c(m,n))
        ! initialize input
        DO i = 1, batchsize
          DO ki = 1, k
            DO j = 1, m
              a(j,ki,i) = REAL(1, T) / REAL(MOD(i+j+ki, 25), T)
            END DO
            DO j = 1, n
              b(ki,j,i) = REAL(7, T) / REAL(MOD(i+j+ki, 75), T)
            END DO
          END DO
        END DO
        c(:,:) = REAL(0, T)
        ! generates and dispatches a matrix multiplication kernel
        CALL libxsmm_mmdispatch(xmm, m, n, k,                           &
     &    alpha=REAL(1, T), beta=REAL(1, T))
        ! kernel multiplies and accumulates matrices: C += Ai * Bi
        DO i = 1, batchsize
          CALL libxsmm_mmcall(xmm, a(:,:,i), b(:,:,i), c)
        END DO
        DEALLOCATE(a, b, c)
      END PROGRAM
