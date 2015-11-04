!*****************************************************************************!
!* Copyright (c) 2013-2015, Intel Corporation                                *!
!* All rights reserved.                                                      *!
!*                                                                           *!
!* Redistribution and use in source and binary forms, with or without        *!
!* modification, are permitted provided that the following conditions        *!
!* are met:                                                                  *!
!* 1. Redistributions of source code must retain the above copyright         *!
!*    notice, this list of conditions and the following disclaimer.          *!
!* 2. Redistributions in binary form must reproduce the above copyright      *!
!*    notice, this list of conditions and the following disclaimer in the    *!
!*    documentation and/or other materials provided with the distribution.   *!
!* 3. Neither the name of the copyright holder nor the names of its          *!
!*    contributors may be used to endorse or promote products derived        *!
!*    from this software without specific prior written permission.          *!
!*                                                                           *!
!* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS       *!
!* "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT         *!
!* LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR     *!
!* A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT      *!
!* HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,    *!
!* SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED  *!
!* TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR    *!
!* PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF    *!
!* LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING      *!
!* NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS        *!
!* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.              *!
!*****************************************************************************!
!* Hans Pabst (Intel Corp.), Alexander Heinecke (Intel Corp.)                *!
!*****************************************************************************!

MODULE LIBXSMM
  USE, INTRINSIC :: ISO_C_BINDING
  IMPLICIT NONE

  ! Kind of types used to parameterize the implementation.
  INTEGER, PARAMETER :: LIBXSMM_SINGLE_PRECISION  = 4 !KIND(1.0)
  INTEGER, PARAMETER :: LIBXSMM_DOUBLE_PRECISION  = 8 !KIND(1D0)
  INTEGER, PARAMETER :: LIBXSMM_INTEGER_TYPE      = KIND(1)

  ! Parameters the library was built for.
  INTEGER(LIBXSMM_INTEGER_TYPE), PARAMETER :: LIBXSMM_ALIGNMENT       = $ALIGNMENT
  INTEGER(LIBXSMM_INTEGER_TYPE), PARAMETER :: LIBXSMM_ALIGNED_STORES  = $ALIGNED_STORES
  INTEGER(LIBXSMM_INTEGER_TYPE), PARAMETER :: LIBXSMM_ALIGNED_LOADS   = $ALIGNED_LOADS
  INTEGER(LIBXSMM_INTEGER_TYPE), PARAMETER :: LIBXSMM_ALIGNED_MAX     = $ALIGNED_MAX
  INTEGER(LIBXSMM_INTEGER_TYPE), PARAMETER :: LIBXSMM_PREFETCH        = $PREFETCH
  INTEGER(LIBXSMM_INTEGER_TYPE), PARAMETER :: LIBXSMM_ROW_MAJOR       = $ROW_MAJOR
  INTEGER(LIBXSMM_INTEGER_TYPE), PARAMETER :: LIBXSMM_COL_MAJOR       = $COL_MAJOR
  INTEGER(LIBXSMM_INTEGER_TYPE), PARAMETER :: LIBXSMM_MAX_MNK         = $MAX_MNK
  INTEGER(LIBXSMM_INTEGER_TYPE), PARAMETER :: LIBXSMM_MAX_M           = $MAX_M
  INTEGER(LIBXSMM_INTEGER_TYPE), PARAMETER :: LIBXSMM_MAX_N           = $MAX_N
  INTEGER(LIBXSMM_INTEGER_TYPE), PARAMETER :: LIBXSMM_MAX_K           = $MAX_K
  INTEGER(LIBXSMM_INTEGER_TYPE), PARAMETER :: LIBXSMM_AVG_M           = $AVG_M
  INTEGER(LIBXSMM_INTEGER_TYPE), PARAMETER :: LIBXSMM_AVG_N           = $AVG_N
  INTEGER(LIBXSMM_INTEGER_TYPE), PARAMETER :: LIBXSMM_AVG_K           = $AVG_K
  INTEGER(LIBXSMM_INTEGER_TYPE), PARAMETER :: LIBXSMM_JIT             = $JIT

  ! Flag enumeration which can be IORed.
  INTEGER(LIBXSMM_INTEGER_TYPE), PARAMETER :: LIBXSMM_GEMM_FLAG_TRANS_A = 1, &
                                              LIBXSMM_GEMM_FLAG_TRANS_B = 2, &
                                              LIBXSMM_GEMM_FLAG_ALIGN_A = 4, &
                                              LIBXSMM_GEMM_FLAG_ALIGN_C = 8

  ! Flag representing the GEMM performed by the simplified interface.
  INTEGER(LIBXSMM_INTEGER_TYPE), PARAMETER :: LIBXSMM_GEMM_FLAG_DEFAULT = IOR( &
                                                MERGE(LIBXSMM_GEMM_FLAG_ALIGN_A, 0, 1.LT.LIBXSMM_ALIGNED_LOADS), &
                                                MERGE(LIBXSMM_GEMM_FLAG_ALIGN_C, 0, 1.LT.LIBXSMM_ALIGNED_STORES))

  ! Parameters representing the GEMM performed by the simplified interface.
  INTEGER(LIBXSMM_DOUBLE_PRECISION), PARAMETER :: LIBXSMM_ALPHA = $ALPHA
  INTEGER(LIBXSMM_DOUBLE_PRECISION), PARAMETER :: LIBXSMM_BETA  = $BETA

  ! Enumeration of the available prefetch strategies which can be IORed.
  !   LIBXSMM_PREFETCH_NONE:      No prefetching and no prefetch fn. signature.
  !   LIBXSMM_PREFETCH_SIGNATURE: Only function prefetch signature.
  !   LIBXSMM_PREFETCH_AL2:       Prefetch PA using accesses to A.
  !   LIBXSMM_PREFETCH_AL2_JPST:  Prefetch PA (aggressive).
  !   LIBXSMM_PREFETCH_BL2_VIA_C: Prefetch PB using accesses to C.
  !   LIBXSMM_PREFETCH_AL2_AHEAD: Prefetch A ahead.
  INTEGER(LIBXSMM_INTEGER_TYPE), PARAMETER :: LIBXSMM_PREFETCH_NONE       = 0, &
                                              LIBXSMM_PREFETCH_SIGNATURE  = 1, &
                                              LIBXSMM_PREFETCH_AL2        = 2, &
                                              LIBXSMM_PREFETCH_AL2_JPST   = 4, &
                                              LIBXSMM_PREFETCH_BL2_VIA_C  = 8, &
                                              LIBXSMM_PREFETCH_AL2_AHEAD  = 16, &
                                              LIBXSMM_PREFETCH_AL2BL2_VIA_C       = IOR(LIBXSMM_PREFETCH_BL2_VIA_C, &
                                                                                        LIBXSMM_PREFETCH_AL2), &
                                              LIBXSMM_PREFETCH_AL2BL2_VIA_C_JPST  = IOR(LIBXSMM_PREFETCH_BL2_VIA_C, &
                                                                                        LIBXSMM_PREFETCH_AL2_JPST), &
                                              LIBXSMM_PREFETCH_AL2BL2_VIA_C_AHEAD = IOR(LIBXSMM_PREFETCH_BL2_VIA_C, &
                                                                                        LIBXSMM_PREFETCH_AL2_AHEAD)

  ! Overloaded dispatch/JIT routines (single/double precision).
  INTERFACE libxsmm_mm_dispatch
    MODULE PROCEDURE smm_dispatch, dmm_dispatch
  END INTERFACE

  ! Overloaded BLAS routines (single/double precision).
  INTERFACE libxsmm_blasmm
    MODULE PROCEDURE libxsmm_sblasmm, libxsmm_dblasmm
  END INTERFACE

  ! Overloaded optimized routines (single/double precision).
  INTERFACE libxsmm_imm
    MODULE PROCEDURE libxsmm_simm, libxsmm_dimm
  END INTERFACE

  ! Overloaded auto-dispatch routines (single/double precision).
  INTERFACE libxsmm_mm
    MODULE PROCEDURE libxsmm_smm, libxsmm_dmm
  END INTERFACE

  ! Type of a function generated for a specific Alpha, Beta, and M, N, and K.
  ABSTRACT INTERFACE
    PURE SUBROUTINE LIBXSMM_SMM_FUNCTION(alpha, beta, a, b, c) BIND(C)
      IMPORT :: C_FLOAT
      REAL(C_FLOAT), VALUE, INTENT(IN) :: alpha, beta
      REAL(C_FLOAT), INTENT(IN) :: a(*), b(*)
      REAL(C_FLOAT), INTENT(INOUT) :: c(*)
    END SUBROUTINE
    PURE SUBROUTINE LIBXSMM_DMM_FUNCTION(alpha, beta, a, b, c) BIND(C)
      IMPORT :: C_DOUBLE
      REAL(C_DOUBLE), VALUE, INTENT(IN) :: alpha, beta
      REAL(C_DOUBLE), INTENT(IN) :: a(*), b(*)
      REAL(C_DOUBLE), INTENT(INOUT) :: c(*)
    END SUBROUTINE
  END INTERFACE

  !DIR$ ATTRIBUTES OFFLOAD:MIC :: sgemm, dgemm
  !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_init, libxsmm_smm_dispatch, libxsmm_dmm_dispatch
  !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_timer_tick, libxsmm_timer_duration
  INTERFACE
    SUBROUTINE sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
      IMPORT LIBXSMM_INTEGER_TYPE, LIBXSMM_SINGLE_PRECISION
      CHARACTER(1), INTENT(IN) :: transa, transb
      INTEGER(LIBXSMM_INTEGER_TYPE), INTENT(IN) :: m, n, k, lda, ldb, ldc
      REAL(LIBXSMM_SINGLE_PRECISION), INTENT(IN) :: a(lda,*), b(ldb,*), alpha, beta
      REAL(LIBXSMM_SINGLE_PRECISION), INTENT(INOUT) :: c(ldc,*)
    END SUBROUTINE
    SUBROUTINE dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
      IMPORT LIBXSMM_INTEGER_TYPE, LIBXSMM_DOUBLE_PRECISION
      CHARACTER(1), INTENT(IN) :: transa, transb
      INTEGER(LIBXSMM_INTEGER_TYPE), INTENT(IN) :: m, n, k, lda, ldb, ldc
      REAL(LIBXSMM_DOUBLE_PRECISION), INTENT(IN) :: a(lda,*), b(ldb,*), alpha, beta
      REAL(LIBXSMM_DOUBLE_PRECISION), INTENT(INOUT) :: c(ldc,*)
    END SUBROUTINE

    ! Initialize the library; pay for setup cost at a specific point.
    SUBROUTINE libxsmm_init() BIND(C)
    END SUBROUTINE

    ! Query or JIT-generate a function; return zero if it does not exist or if JIT is not supported (single-precision).
    TYPE(C_FUNPTR) PURE FUNCTION libxsmm_smm_dispatch(alpha, beta, m, n, k, lda, ldb, ldc, flags, prefetch) BIND(C)
      IMPORT :: C_FUNPTR, C_FLOAT, C_INT
      REAL(C_FLOAT), VALUE, INTENT(IN) :: alpha, beta
      INTEGER(C_INT), VALUE, INTENT(IN) :: m, n, k, lda, ldb, ldc, flags, prefetch
    END FUNCTION
    ! Query or JIT-generate a function; return zero if it does not exist or if JIT is not supported (double-precision).
    TYPE(C_FUNPTR) PURE FUNCTION libxsmm_dmm_dispatch(alpha, beta, m, n, k, lda, ldb, ldc, flags, prefetch) BIND(C)
      IMPORT :: C_FUNPTR, C_DOUBLE, C_INT
      REAL(C_DOUBLE), VALUE, INTENT(IN) :: alpha, beta
      INTEGER(C_INT), VALUE, INTENT(IN) :: m, n, k, lda, ldb, ldc, flags, prefetch
    END FUNCTION

    ! Non-pure function returning the current clock tick using a platform-specific resolution.
    INTEGER(C_LONG_LONG) FUNCTION libxsmm_timer_tick() BIND(C)
      IMPORT :: C_LONG_LONG
    END FUNCTION
    ! Non-pure function (timer freq. may vary) returning the duration between two ticks (seconds).
    REAL(C_DOUBLE) FUNCTION libxsmm_timer_duration(tick0, tick1) BIND(C)
      IMPORT :: C_LONG_LONG, C_DOUBLE
      INTEGER(C_LONG_LONG), INTENT(IN), VALUE :: tick0, tick1
    END FUNCTION
  END INTERFACE$MNK_INTERFACE_LIST

CONTAINS
  !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_up
  !DIR$ ATTRIBUTES INLINE :: libxsmm_up
  PURE FUNCTION libxsmm_up(n, up) RESULT(nup)
    INTEGER(LIBXSMM_INTEGER_TYPE), INTENT(IN) :: n, up
    INTEGER(LIBXSMM_INTEGER_TYPE) :: nup
    nup = ((n + up - 1) / up) * up
  END FUNCTION

  !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_align_value
  !DIR$ ATTRIBUTES INLINE :: libxsmm_align_value
  PURE FUNCTION libxsmm_align_value(n, typesize, alignment) RESULT(na)
    INTEGER(LIBXSMM_INTEGER_TYPE), INTENT(IN) :: n, typesize, alignment
    INTEGER(LIBXSMM_INTEGER_TYPE) :: na
    na = libxsmm_up(n * typesize, alignment) / typesize
  END FUNCTION

  !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_ld
  !DIR$ ATTRIBUTES INLINE :: libxsmm_ld
  PURE FUNCTION libxsmm_ld(m, n) RESULT(ld)
    INTEGER(LIBXSMM_INTEGER_TYPE), INTENT(IN) :: m, n
    INTEGER(LIBXSMM_INTEGER_TYPE) :: ld
    ld = MERGE(m, n, 0.NE.LIBXSMM_COL_MAJOR)
  END FUNCTION

  ! Non-dispatched matrix-matrix multiplication using BLAS (single-precision).
  !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_sblasmm
  !DIR$ ATTRIBUTES INLINE :: libxsmm_sblasmm
  SUBROUTINE libxsmm_sblasmm(alpha, beta, m, n, k, a, b, c)
    INTEGER(LIBXSMM_INTEGER_TYPE), PARAMETER :: T = LIBXSMM_SINGLE_PRECISION
    INTEGER(LIBXSMM_INTEGER_TYPE), INTENT(IN) :: m, n, k
    REAL(T), INTENT(IN) :: alpha, beta, a(:,:), b(:,:)
    REAL(T), INTENT(INOUT) :: c(:,:)
    IF (0.NE.LIBXSMM_COL_MAJOR) THEN
      CALL sgemm('N', 'N', m, n, k, alpha, a, m, b, k, beta, c, SIZE(c, 1))
    ELSE
      CALL sgemm('N', 'N', n, m, k, alpha, b, n, a, k, beta, c, SIZE(c, 1))
    ENDIF
  END SUBROUTINE

  ! Non-dispatched matrix-matrix multiplication using BLAS (double-precision).
  !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_dblasmm
  !DIR$ ATTRIBUTES INLINE :: libxsmm_dblasmm
  SUBROUTINE libxsmm_dblasmm(alpha, beta, m, n, k, a, b, c)
    INTEGER(LIBXSMM_INTEGER_TYPE), PARAMETER :: T = LIBXSMM_DOUBLE_PRECISION
    INTEGER(LIBXSMM_INTEGER_TYPE), INTENT(IN) :: m, n, k
    REAL(T), INTENT(IN) :: alpha, beta, a(:,:), b(:,:)
    REAL(T), INTENT(INOUT) :: c(:,:)
    IF (0.NE.LIBXSMM_COL_MAJOR) THEN
      CALL dgemm('N', 'N', m, n, k, alpha, a, m, b, k, beta, c, SIZE(c, 1))
    ELSE
      CALL dgemm('N', 'N', n, m, k, alpha, b, n, a, k, beta, c, SIZE(c, 1))
    ENDIF
  END SUBROUTINE

  ! Non-dispatched matrix-matrix multiplication using optimized code (single-precision).
  !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_simm
  !DIR$ ATTRIBUTES INLINE :: libxsmm_simm
  SUBROUTINE libxsmm_simm(alpha, beta, m, n, k, a, b, c)
    INTEGER(LIBXSMM_INTEGER_TYPE), PARAMETER :: T = LIBXSMM_SINGLE_PRECISION
    INTEGER(LIBXSMM_INTEGER_TYPE), INTENT(IN) :: m, n, k
    INTEGER(LIBXSMM_INTEGER_TYPE) :: i, j
    REAL(T), INTENT(IN) :: alpha, beta
    REAL(T), INTENT(IN), TARGET, CONTIGUOUS :: a(:,:), b(:,:)
    REAL(T), INTENT(INOUT) :: c($SHAPE_C1,$SHAPE_C2)
    REAL(T), POINTER :: x(:,:), y(:,:)
    REAL(T), PARAMETER :: Zero = 0, One = 1
    REAL(T) :: xalpha, xbeta
    xalpha = MERGE(alpha, One, 1.NE.LIBXSMM_COL_MAJOR)
    xbeta = MERGE(beta, Zero, 0.NE.LIBXSMM_COL_MAJOR)
    IF (0.NE.LIBXSMM_COL_MAJOR) THEN
      !DIR$ OMP SIMD COLLAPSE(2)
      DO j = LBOUND(b, 2), LBOUND(b, 2) + n - 1
        !DIR$ LOOP COUNT(1, LIBXSMM_MAX_M, LIBXSMM_AVG_M)
        DO i = LBOUND(a, 1), LBOUND(a, 1) + m - 1
          c(i,j) = xbeta * c(i,j) + xalpha * DOT_PRODUCT(a(i,:), b(:,j))
        END DO
      END DO
    ELSE
      x(1:$SHAPE_AT1,1:$SHAPE_AT2) => b(:,:)
      y(1:$SHAPE_BT1,1:$SHAPE_BT2) => a(:,:)
      !DIR$ OMP SIMD COLLAPSE(2)
      DO j = 1, m
        !DIR$ LOOP COUNT(1, LIBXSMM_MAX_N, LIBXSMM_AVG_N)
        DO i = 1, n
          c(i,j) = xbeta * c(i,j) + xalpha * DOT_PRODUCT(x(i,:), y(:,j))
        END DO
      END DO
    ENDIF
  END SUBROUTINE

  ! Non-dispatched matrix-matrix multiplication using optimized code (double-precision).
  !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_dimm
  !DIR$ ATTRIBUTES INLINE :: libxsmm_dimm
  SUBROUTINE libxsmm_dimm(alpha, beta, m, n, k, a, b, c)
    INTEGER(LIBXSMM_INTEGER_TYPE), PARAMETER :: T = LIBXSMM_DOUBLE_PRECISION
    INTEGER(LIBXSMM_INTEGER_TYPE), INTENT(IN) :: m, n, k
    INTEGER(LIBXSMM_INTEGER_TYPE) :: i, j
    REAL(T), INTENT(IN) :: alpha, beta
    REAL(T), INTENT(IN), TARGET, CONTIGUOUS :: a(:,:), b(:,:)
    REAL(T), INTENT(INOUT) :: c($SHAPE_C1,$SHAPE_C2)
    REAL(T), POINTER :: x(:,:), y(:,:)
    REAL(T), PARAMETER :: Zero = 0, One = 1
    REAL(T) :: xalpha, xbeta
    xalpha = MERGE(alpha, One, 1.NE.LIBXSMM_COL_MAJOR)
    xbeta = MERGE(beta, Zero, 0.NE.LIBXSMM_COL_MAJOR)
    IF (0.NE.LIBXSMM_COL_MAJOR) THEN
      !DIR$ OMP SIMD COLLAPSE(2)
      DO j = LBOUND(b, 2), LBOUND(b, 2) + n - 1
        !DIR$ LOOP COUNT(1, LIBXSMM_MAX_M, LIBXSMM_AVG_M)
        DO i = LBOUND(a, 1), LBOUND(a, 1) + m - 1
          c(i,j) = xbeta * c(i,j) + xalpha * DOT_PRODUCT(a(i,:), b(:,j))
        END DO
      END DO
    ELSE
      x(1:$SHAPE_AT1,1:$SHAPE_AT2) => b(:,:)
      y(1:$SHAPE_BT1,1:$SHAPE_BT2) => a(:,:)
      !DIR$ OMP SIMD COLLAPSE(2)
      DO j = 1, m
        !DIR$ LOOP COUNT(1, LIBXSMM_MAX_N, LIBXSMM_AVG_N)
        DO i = 1, n
          c(i,j) = xbeta * c(i,j) + xalpha * DOT_PRODUCT(x(i,:), y(:,j))
        END DO
      END DO
    ENDIF
  END SUBROUTINE

  ! Dispatched matrix-matrix multiplication (single-precision).
  !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_smm
  !DIR$ ATTRIBUTES INLINE :: libxsmm_smm
  SUBROUTINE libxsmm_smm(alpha, beta, m, n, k, a, b, c)
    INTEGER(LIBXSMM_INTEGER_TYPE), PARAMETER :: T = LIBXSMM_SINGLE_PRECISION
    INTEGER(LIBXSMM_INTEGER_TYPE), INTENT(IN) :: m, n, k
    REAL(T), INTENT(IN) :: alpha, beta, a(:,:), b(:,:)
    REAL(T), INTENT(INOUT) :: c(:,:)
    !DIR$ ATTRIBUTES OFFLOAD:MIC :: smm
    PROCEDURE(LIBXSMM_SMM_FUNCTION), POINTER :: smm
    TYPE(C_FUNPTR) :: f
    IF (LIBXSMM_MAX_MNK.GE.(m * n * k)) THEN
      f = libxsmm_smm_dispatch(alpha, beta, m, n, k, libxsmm_ld(m, n), k, &
            libxsmm_align_value(libxsmm_ld(m, n), T, LIBXSMM_ALIGNED_STORES), &
            LIBXSMM_GEMM_FLAG_DEFAULT, LIBXSMM_PREFETCH)
      IF (C_ASSOCIATED(f)) THEN
        CALL C_F_PROCPOINTER(f, smm)
        CALL smm(alpha, beta, a, b, c)
      ELSE
        CALL libxsmm_simm(alpha, beta, m, n, k, a, b, c)
      ENDIF
    ELSE
      CALL libxsmm_sblasmm(alpha, beta, m, n, k, a, b, c)
    ENDIF
  END SUBROUTINE

  ! Dispatched matrix-matrix multiplication (double-precision).
  !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_dmm
  !DIR$ ATTRIBUTES INLINE :: libxsmm_dmm
  SUBROUTINE libxsmm_dmm(alpha, beta, m, n, k, a, b, c)
    INTEGER(LIBXSMM_INTEGER_TYPE), PARAMETER :: T = LIBXSMM_DOUBLE_PRECISION
    INTEGER(LIBXSMM_INTEGER_TYPE), INTENT(IN) :: m, n, k
    REAL(T), INTENT(IN) :: alpha, beta, a(:,:), b(:,:)
    REAL(T), INTENT(INOUT) :: c(:,:)
    !DIR$ ATTRIBUTES OFFLOAD:MIC :: dmm
    PROCEDURE(LIBXSMM_DMM_FUNCTION), POINTER :: dmm
    TYPE(C_FUNPTR) :: f
    IF (LIBXSMM_MAX_MNK.GE.(m * n * k)) THEN
      f = libxsmm_dmm_dispatch(alpha, beta, m, n, k, libxsmm_ld(m, n), k, &
            libxsmm_align_value(libxsmm_ld(m, n), T, LIBXSMM_ALIGNED_STORES), &
            LIBXSMM_GEMM_FLAG_DEFAULT, LIBXSMM_PREFETCH)
      IF (C_ASSOCIATED(f)) THEN
        CALL C_F_PROCPOINTER(f, dmm)
        CALL dmm(alpha, beta, a, b, c)
      ELSE
        CALL libxsmm_dimm(alpha, beta, m, n, k, a, b, c)
      ENDIF
    ELSE
      CALL libxsmm_dblasmm(alpha, beta, m, n, k, a, b, c)
    ENDIF
  END SUBROUTINE

  !DIR$ ATTRIBUTES OFFLOAD:MIC :: smm_dispatch
  !DIR$ ATTRIBUTES INLINE :: smm_dispatch
  TYPE(C_FUNPTR) PURE FUNCTION smm_dispatch(alpha, beta, m, n, k, lda, ldb, ldc, flags, prefetch)
    INTEGER(LIBXSMM_INTEGER_TYPE), PARAMETER :: T = LIBXSMM_SINGLE_PRECISION, Zero = 0
    INTEGER(LIBXSMM_INTEGER_TYPE), OPTIONAL, INTENT(IN) :: lda, ldb, ldc, flags, prefetch
    INTEGER(LIBXSMM_INTEGER_TYPE), INTENT(IN) :: m, n, k
    REAL(T), INTENT(IN) :: alpha, beta
    TYPE(C_FUNPTR) :: f
    f = libxsmm_smm_dispatch(alpha, beta, m, n, k, &
          MERGE(lda, Zero, PRESENT(lda)), MERGE(ldb, Zero, PRESENT(ldb)), MERGE(ldc, Zero, PRESENT(ldc)), &
          MERGE(flags, LIBXSMM_GEMM_FLAG_DEFAULT, PRESENT(flags)), &
          MERGE(prefetch, LIBXSMM_PREFETCH, PRESENT(prefetch)))
  END FUNCTION

  !DIR$ ATTRIBUTES OFFLOAD:MIC :: dmm_dispatch
  !DIR$ ATTRIBUTES INLINE :: dmm_dispatch
  TYPE(C_FUNPTR) PURE FUNCTION dmm_dispatch(alpha, beta, m, n, k, lda, ldb, ldc, flags, prefetch)
    INTEGER(LIBXSMM_INTEGER_TYPE), PARAMETER :: T = LIBXSMM_DOUBLE_PRECISION, Zero = 0
    INTEGER(LIBXSMM_INTEGER_TYPE), OPTIONAL, INTENT(IN) :: lda, ldb, ldc, flags, prefetch
    INTEGER(LIBXSMM_INTEGER_TYPE), INTENT(IN) :: m, n, k
    REAL(T), INTENT(IN) :: alpha, beta
    TYPE(C_FUNPTR) :: f
    f = libxsmm_dmm_dispatch(alpha, beta, m, n, k, &
          MERGE(lda, Zero, PRESENT(lda)), MERGE(ldb, Zero, PRESENT(ldb)), MERGE(ldc, Zero, PRESENT(ldc)), &
          MERGE(flags, LIBXSMM_GEMM_FLAG_DEFAULT, PRESENT(flags)), &
          MERGE(prefetch, LIBXSMM_PREFETCH, PRESENT(prefetch)))
  END FUNCTION
END MODULE
