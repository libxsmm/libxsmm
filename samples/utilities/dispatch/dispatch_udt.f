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
      PROGRAM dispatch_udt
        USE, INTRINSIC :: ISO_C_BINDING,  ONLY: C_PTR, C_LOC,           &
     &                                          C_ASSOCIATED,           &
     &                                          C_F_POINTER
        USE :: LIBXSMM, ONLY: LIBXSMM_BLASINT_KIND,                     &
     &                        LIBXSMM_MMFUNCTION => LIBXSMM_DMMFUNCTION,&
     &                        libxsmm_mmdispatch => libxsmm_dmmdispatch,&
     &                        libxsmm_mmcall => libxsmm_dmmcall,        &
     &                        libxsmm_xregister, libxsmm_xdispatch
        IMPLICIT NONE
        INTEGER, PARAMETER :: T = KIND(0D0)
        INTEGER :: batchsize = 1000, i
        INTEGER(LIBXSMM_BLASINT_KIND) :: j, ki, nrepeat = 100
        INTEGER(LIBXSMM_BLASINT_KIND) :: m = 13, n = 5, k = 7
        REAL(T), ALLOCATABLE :: a(:,:,:), b(:,:,:), c(:,:)
        TYPE(LIBXSMM_MMFUNCTION), TARGET  :: xmm(2) ! array of kernels
        TYPE(LIBXSMM_MMFUNCTION), POINTER :: udt(:)
        INTEGER(LIBXSMM_BLASINT_KIND), TARGET :: key(3)
        TYPE(C_PTR) :: ptr

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

        ! repeat inner part to exercise libxsmm_xdispatch
        DO j = 1, nrepeat
          key = (/m, n, k/) ! setup key
          ! query associated value using key
          ptr = libxsmm_xdispatch(                                      &
     &      C_LOC(key), SIZE(key) * LIBXSMM_BLASINT_KIND)

          IF (C_ASSOCIATED(ptr)) THEN ! value was already registered
            ! convert C-ptr to Fortran POINTER
            CALL C_F_POINTER(ptr, udt, (/SIZE(xmm)/))
          ELSE ! no value registered yet
            ! generate and dispatch a series of kernels
            CALL libxsmm_mmdispatch(xmm(1), m, n, k,                    &
     &        alpha=REAL(1, T), beta=REAL(1, T))
            CALL libxsmm_mmdispatch(xmm(2), m, n, k + 2,                &
     &        alpha=REAL(1, T), beta=REAL(1, T))
            ! register an entry that contains all kernels from above
            ptr = libxsmm_xregister(                                    &
     &        C_LOC(key), SIZE(key) * LIBXSMM_BLASINT_KIND,             &
     &        SIZE(xmm) * 8, C_LOC(xmm))
            ! point udt to xmm (below code uses udt to refer to kernels
            udt => xmm ! alternatively, use C_F_POINTER
          END IF

          ! here we executed libxsmm_xdispatch one time (for this round)
          ! all kernels have been dispatched at once (udt)
          DO i = 1, batchsize
            CALL libxsmm_mmcall(udt(1), a(:,:,i), b(:,:,i), c)
          END DO
        END DO
        DEALLOCATE(a, b, c)
      END PROGRAM
