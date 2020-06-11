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
        INTEGER(LIBXSMM_BLASINT_KIND) :: mi, ki, ni
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
            DO mi = 1, m
              a(mi,ki,i) = REAL(1, T) / REAL(MOD(i+mi+ki, 25), T)
            END DO
            DO ni = 1, n
              b(ki,ni,i) = REAL(7, T) / REAL(MOD(i+ki+ni, 75), T)
            END DO
          END DO
        END DO
        c(:,:) = 0

        key = (/m, n, k/) ! setup key to query associated value
        ptr = libxsmm_xdispatch(                                        &
     &    C_LOC(key), SIZE(key) * LIBXSMM_BLASINT_KIND)

        IF (C_ASSOCIATED(ptr)) THEN
          CALL C_F_POINTER(ptr, udt, (/SIZE(xmm)/))
        ELSE ! no value registered
          ! generate and dispatch a series of kernels
          CALL libxsmm_mmdispatch(xmm(1), m, n, k,                      &
     &      alpha=REAL(1, T), beta=REAL(1, T))
          CALL libxsmm_mmdispatch(xmm(2), m, n, k + 2,                  &
     &      alpha=REAL(1, T), beta=REAL(1, T))
          ! register content of xmm-array
          ptr = libxsmm_xregister(                                      &
     &      C_LOC(key), SIZE(key) * LIBXSMM_BLASINT_KIND,               &
     &      SIZE(xmm) * 8, C_LOC(xmm))
          ! point udt to xmm (below code uses udt to refer to kernels
          udt => xmm
        END IF

        ! kernel multiplies and accumulates matrices: C += Ai * Bi
        DO i = 1, batchsize
          ! call one of the kernels (real code may call all kernels)
          CALL libxsmm_mmcall(udt(1), a(:,:,i), b(:,:,i), c)
        END DO
        DEALLOCATE(a, b, c)
      END PROGRAM
