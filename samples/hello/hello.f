!=======================================================================!
! Copyright (c) Intel Corporation - All rights reserved.                !
! This file is part of the LIBXSMM library.                             !
!                                                                       !
! For information on the license, see the LICENSE file.                 !
! Further information: https://github.com/hfp/libxsmm/                  !
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
        INTEGER(LIBXSMM_BLASINT_KIND) :: mi, ki, ni
        INTEGER(LIBXSMM_BLASINT_KIND) :: m = 13, n = 5, k = 7
        REAL(T), ALLOCATABLE :: a(:,:,:), b(:,:,:), c(:,:)
        TYPE(LIBXSMM_MMFUNCTION) :: xmm

        ALLOCATE(a(m,k,batchsize), b(k,n,batchsize), c(m,n))
        ! generates and dispatches a matrix multiplication kernel
        CALL libxsmm_mmdispatch(xmm, m, n, k,                           &
     &    alpha=REAL(1, T), beta=REAL(1, T))
        ! initialize input
        DO i = 1, batchsize
          DO ki = 1, k
            DO mi = 1, m
              a(mi,ki,i) = REAL(1, T) / REAL(MOD(i+mi+ki, 25), T)
            END DO
            DO ni = 1, n
              b(ki,ni,i) = REAL(7, T) / REAL(MOD(i+ki+ni, 75), T)
            END DO
          END DO
        END DO
        c(:,:) = 0
        ! kernel multiplies and accumulates matrices: C += Ai * Bi
        DO i = 1, batchsize
          CALL libxsmm_mmcall(xmm, a(:,:,i), b(:,:,i), c)
        END DO
        DEALLOCATE(a, b, c)
      END PROGRAM
