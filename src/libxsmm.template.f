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
        USE, INTRINSIC :: ISO_C_BINDING, ONLY:                          &
     &                                    C_FUNPTR, C_F_PROCPOINTER,    &
     &                                    C_PTR, C_NULL_PTR, C_LOC,     &
     &                                    C_INT, C_FLOAT, C_DOUBLE,     &
     &                                    C_LONG_LONG, C_CHAR
        IMPLICIT NONE
        PRIVATE :: libxsmm_srealptr, libxsmm_drealptr

        CHARACTER(*), PARAMETER :: LIBXSMM_VERSION = "$VERSION"
        CHARACTER(*), PARAMETER :: LIBXSMM_BRANCH  = "$BRANCH"
        INTEGER(C_INT), PARAMETER :: LIBXSMM_VERSION_MAJOR  = $MAJOR
        INTEGER(C_INT), PARAMETER :: LIBXSMM_VERSION_MINOR  = $MINOR
        INTEGER(C_INT), PARAMETER :: LIBXSMM_VERSION_UPDATE = $UPDATE
        INTEGER(C_INT), PARAMETER :: LIBXSMM_VERSION_PATCH  = $PATCH

        ! Parameters the library and static kernels were built for.
        INTEGER(C_INT), PARAMETER :: LIBXSMM_ALIGNMENT = $ALIGNMENT
        INTEGER(C_INT), PARAMETER :: LIBXSMM_ROW_MAJOR = $ROW_MAJOR
        INTEGER(C_INT), PARAMETER :: LIBXSMM_COL_MAJOR = $COL_MAJOR
        INTEGER(C_INT), PARAMETER :: LIBXSMM_PREFETCH = $PREFETCH
        INTEGER(C_INT), PARAMETER :: LIBXSMM_MAX_MNK = $MAX_MNK
        INTEGER(C_INT), PARAMETER :: LIBXSMM_MAX_M = $MAX_M
        INTEGER(C_INT), PARAMETER :: LIBXSMM_MAX_N = $MAX_N
        INTEGER(C_INT), PARAMETER :: LIBXSMM_MAX_K = $MAX_K
        INTEGER(C_INT), PARAMETER :: LIBXSMM_AVG_M = $AVG_M
        INTEGER(C_INT), PARAMETER :: LIBXSMM_AVG_N = $AVG_N
        INTEGER(C_INT), PARAMETER :: LIBXSMM_AVG_K = $AVG_K
        INTEGER(C_INT), PARAMETER :: LIBXSMM_FLAGS = $FLAGS
        INTEGER(C_INT), PARAMETER :: LIBXSMM_ILP64 = $ILP64
        INTEGER(C_INT), PARAMETER :: LIBXSMM_JIT = $JIT

        ! Integer type impacting the BLAS interface (LP64: 32-bit, and ILP64: 64-bit).
        INTEGER(C_INT), PARAMETER :: LIBXSMM_BLASINT_KIND =             &
     &    MERGE(C_INT, C_LONG_LONG, 0.EQ.LIBXSMM_ILP64)

        ! Parameters representing the GEMM performed by the simplified interface.
        REAL(C_DOUBLE), PARAMETER :: LIBXSMM_ALPHA = $ALPHA
        REAL(C_DOUBLE), PARAMETER :: LIBXSMM_BETA = $BETA

        ! Flag enumeration which can be IORed.
        INTEGER(C_INT), PARAMETER ::                                    &
     &    LIBXSMM_GEMM_FLAG_TRANS_A = 1,                                &
     &    LIBXSMM_GEMM_FLAG_TRANS_B = 2,                                &
     &    LIBXSMM_GEMM_FLAG_ALIGN_A = 4,                                &
     &    LIBXSMM_GEMM_FLAG_ALIGN_C = 8

        ! Enumeration of the available prefetch strategies which can be IORed.
        INTEGER(C_INT), PARAMETER ::                                    &
          ! No prefetching and no prefetch function signature.
     &    LIBXSMM_PREFETCH_NONE       = 0,                              &
          ! Only function prefetch signature.
     &    LIBXSMM_PREFETCH_SIGNATURE  = 1,                              &
          ! Prefetch PA using accesses to A.
     &    LIBXSMM_PREFETCH_AL2        = 2,                              &
          ! Prefetch PA (aggressive).
     &    LIBXSMM_PREFETCH_AL2_JPST   = 4,                              &
          ! Prefetch PB using accesses to C.
     &    LIBXSMM_PREFETCH_BL2_VIA_C  = 8,                              &
          ! Prefetch A ahead.
     &    LIBXSMM_PREFETCH_AL2_AHEAD  = 16,                             &
          ! Composed prefetch strategies.
     &    LIBXSMM_PREFETCH_AL2BL2_VIA_C = IOR(                          &
     &        LIBXSMM_PREFETCH_BL2_VIA_C, LIBXSMM_PREFETCH_AL2),        &
     &    LIBXSMM_PREFETCH_AL2BL2_VIA_C_JPST = IOR(                     &
     &        LIBXSMM_PREFETCH_BL2_VIA_C, LIBXSMM_PREFETCH_AL2_JPST),   &
     &    LIBXSMM_PREFETCH_AL2BL2_VIA_C_AHEAD = IOR(                    &
     &        LIBXSMM_PREFETCH_BL2_VIA_C, LIBXSMM_PREFETCH_AL2_AHEAD)

        ! Type of a function specialized for a given parameter set.
        ABSTRACT INTERFACE
          ! Specialized function with fused alpha and beta arguments.
          PURE SUBROUTINE LIBXSMM_FUNCTION0(a, b, c) BIND(C)
            IMPORT :: C_PTR
            TYPE(C_PTR), INTENT(IN), VALUE :: a, b, c
          END SUBROUTINE

          ! Specialized function with alpha, beta, and prefetch arguments.
          PURE SUBROUTINE LIBXSMM_FUNCTION1(a, b, c,                    &
     &    pa, pb, pc) BIND(C)
            IMPORT :: C_PTR
            TYPE(C_PTR), INTENT(IN), VALUE :: a, b, c
            TYPE(C_PTR), INTENT(IN), VALUE :: pa, pb, pc
          END SUBROUTINE
        END INTERFACE

        ! Generic function type constructing a procedure pointer
        ! associated with a backend function (single-precision).
        TYPE :: LIBXSMM_SMM_FUNCTION
          PRIVATE
            PROCEDURE(LIBXSMM_FUNCTION0), NOPASS, POINTER :: fn0
            PROCEDURE(LIBXSMM_FUNCTION1), NOPASS, POINTER :: fn1
        END TYPE

        ! Generic function type constructing a procedure pointer
        ! associated with a backend function (double-precision).
        TYPE :: LIBXSMM_DMM_FUNCTION
          PRIVATE
            PROCEDURE(LIBXSMM_FUNCTION0), NOPASS, POINTER :: fn0
            PROCEDURE(LIBXSMM_FUNCTION1), NOPASS, POINTER :: fn1
        END TYPE

        ! Construct procedure pointer depending on given argument set.
        INTERFACE libxsmm_dispatch
          MODULE PROCEDURE libxsmm_sdispatch, libxsmm_ddispatch
        END INTERFACE

        ! Check if a function (LIBXSMM_?MM_FUNCTION_TYPE) is available.
        INTERFACE libxsmm_available
          MODULE PROCEDURE libxsmm_savailable, libxsmm_davailable
        END INTERFACE

        ! Call a specialized function.
        INTERFACE libxsmm_call
          MODULE PROCEDURE libxsmm_scall_abx, libxsmm_scall_prx
          MODULE PROCEDURE libxsmm_scall_abc, libxsmm_scall_prf
          MODULE PROCEDURE libxsmm_dcall_abx, libxsmm_dcall_prx
          MODULE PROCEDURE libxsmm_dcall_abc, libxsmm_dcall_prf
        END INTERFACE

        ! Overloaded GEMM routines (single/double precision).
        INTERFACE libxsmm_gemm
          MODULE PROCEDURE libxsmm_sgemm, libxsmm_dgemm
        END INTERFACE

        ! Overloaded BLAS GEMM routines (single/double precision).
        INTERFACE libxsmm_blas_gemm
          MODULE PROCEDURE libxsmm_blas_sgemm, libxsmm_blas_dgemm
        END INTERFACE

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_init, libxsmm_finalize
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_timer_tick
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_timer_duration
        INTERFACE
          ! Initialize the library; pay for setup cost at a specific point.
          SUBROUTINE libxsmm_init() BIND(C)
          END SUBROUTINE

          SUBROUTINE libxsmm_finalize() BIND(C)
          END SUBROUTINE

          ! Non-pure function returning the current clock tick
          ! using a platform-specific resolution.
          INTEGER(C_LONG_LONG) FUNCTION libxsmm_timer_tick() BIND(C)
            IMPORT :: C_LONG_LONG
          END FUNCTION

          ! Non-pure function (timer freq. may vary) returning
          ! the duration between two ticks (seconds).
          REAL(C_DOUBLE) FUNCTION libxsmm_timer_duration(               &
     &    tick0, tick1) BIND(C)
            IMPORT :: C_LONG_LONG, C_DOUBLE
            INTEGER(C_LONG_LONG), INTENT(IN), VALUE :: tick0, tick1
          END FUNCTION
        END INTERFACE$MNK_INTERFACE_LIST

      CONTAINS
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_srealptr
        FUNCTION libxsmm_srealptr(a) RESULT(p)
          REAL(C_FLOAT), INTENT(IN), TARGET :: a(:,:)
          REAL(C_FLOAT), POINTER :: p
          p => a(LBOUND(a,1),LBOUND(a,2))
        END FUNCTION

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_drealptr
        FUNCTION libxsmm_drealptr(a) RESULT(p)
          REAL(C_DOUBLE), INTENT(IN), TARGET :: a(:,:)
          REAL(C_DOUBLE), POINTER :: p
          p => a(LBOUND(a,1),LBOUND(a,2))
        END FUNCTION

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_ld
        INTEGER(C_INT) PURE FUNCTION libxsmm_ld(m, n)
          INTEGER(C_INT), INTENT(IN) :: m, n
          libxsmm_ld = MERGE(m, n, 0.NE.LIBXSMM_COL_MAJOR)
        END FUNCTION

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_align_value
        INTEGER(C_INT) PURE FUNCTION libxsmm_align_value(               &
     &    n, typesize, alignment)
          INTEGER(C_INT), INTENT(IN) :: n, typesize
          INTEGER(C_INT), INTENT(IN) :: alignment
          libxsmm_align_value = (((n * typesize + alignment - 1) /      &
     &      alignment) * alignment) / typesize
        END FUNCTION

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_sfunction
        TYPE(LIBXSMM_SMM_FUNCTION) FUNCTION libxsmm_sfunction(          &
     &  m, n, k, lda, ldb, ldc, alpha, beta, flags, prefetch)
          INTEGER(C_INT), INTENT(IN) :: m, n, k
          INTEGER(C_INT), INTENT(IN), OPTIONAL :: lda, ldb, ldc
          REAL(C_FLOAT), INTENT(IN), OPTIONAL :: alpha, beta
          INTEGER(C_INT), INTENT(IN), OPTIONAL :: flags, prefetch
          PROCEDURE(LIBXSMM_FUNCTION0), POINTER :: fn0
          PROCEDURE(LIBXSMM_FUNCTION1), POINTER :: fn1
          !DIR$ ATTRIBUTES OFFLOAD:MIC :: sdispatch
          INTERFACE
            TYPE(C_FUNPTR) PURE FUNCTION sdispatch(                     &
     &      m, n, k, lda, ldb, ldc, alpha, beta, flags, prefetch)       &
     &      BIND(C, NAME="libxsmm_sdispatch")
              IMPORT :: C_FUNPTR, C_INT, C_FLOAT
              INTEGER(C_INT), INTENT(IN), VALUE :: m, n, k
              INTEGER(C_INT), INTENT(IN) :: lda, ldb, ldc
              REAL(C_FLOAT), INTENT(IN) :: alpha, beta
              INTEGER(C_INT), INTENT(IN) :: flags, prefetch
            END FUNCTION
          END INTERFACE
          IF (.NOT.PRESENT(prefetch)) THEN
            CALL C_F_PROCPOINTER(sdispatch(m, n, k,                     &
     &        lda, ldb, ldc, alpha, beta, flags, prefetch),             &
     &        fn0)
            libxsmm_sfunction%fn0 => fn0
            libxsmm_sfunction%fn1 => NULL()
          ELSE
            CALL C_F_PROCPOINTER(sdispatch(m, n, k,                     &
     &        lda, ldb, ldc, alpha, beta, flags, prefetch),             &
     &        fn1)
            libxsmm_sfunction%fn0 => NULL()
            libxsmm_sfunction%fn1 => fn1
          END IF
        END FUNCTION

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_dfunction
        TYPE(LIBXSMM_DMM_FUNCTION) FUNCTION libxsmm_dfunction(          &
     &  m, n, k, lda, ldb, ldc, alpha, beta, flags, prefetch)
          INTEGER(C_INT), INTENT(IN) :: m, n, k
          INTEGER(C_INT), INTENT(IN), OPTIONAL :: lda, ldb, ldc
          REAL(C_DOUBLE), INTENT(IN), OPTIONAL :: alpha, beta
          INTEGER(C_INT), INTENT(IN), OPTIONAL :: flags, prefetch
          PROCEDURE(LIBXSMM_FUNCTION0), POINTER :: fn0
          PROCEDURE(LIBXSMM_FUNCTION1), POINTER :: fn1
          !DIR$ ATTRIBUTES OFFLOAD:MIC :: ddispatch
          INTERFACE
            TYPE(C_FUNPTR) PURE FUNCTION ddispatch(                     &
     &      m, n, k, lda, ldb, ldc, alpha, beta, flags, prefetch)       &
     &      BIND(C, NAME="libxsmm_ddispatch")
              IMPORT :: C_FUNPTR, C_INT, C_DOUBLE
              INTEGER(C_INT), INTENT(IN), VALUE :: m, n, k
              INTEGER(C_INT), INTENT(IN) :: lda, ldb, ldc
              REAL(C_DOUBLE), INTENT(IN) :: alpha, beta
              INTEGER(C_INT), INTENT(IN) :: flags, prefetch
            END FUNCTION
          END INTERFACE
          IF (.NOT.PRESENT(prefetch)) THEN
            CALL C_F_PROCPOINTER(ddispatch(m, n, k,                     &
     &        lda, ldb, ldc, alpha, beta, flags, prefetch),             &
     &        fn0)
            libxsmm_dfunction%fn0 => fn0
            libxsmm_dfunction%fn1 => NULL()
          ELSE
            CALL C_F_PROCPOINTER(ddispatch(m, n, k,                     &
     &        lda, ldb, ldc, alpha, beta, flags, prefetch),             &
     &        fn1)
            libxsmm_dfunction%fn0 => NULL()
            libxsmm_dfunction%fn1 => fn1
          END IF
        END FUNCTION

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_sdispatch
        SUBROUTINE libxsmm_sdispatch(fn,                                &
     &  m, n, k, lda, ldb, ldc, alpha, beta, flags, prefetch)
          TYPE(LIBXSMM_SMM_FUNCTION), INTENT(OUT) :: fn
          INTEGER(C_INT), INTENT(IN) :: m, n, k
          INTEGER(C_INT), INTENT(IN), OPTIONAL :: lda, ldb, ldc
          REAL(C_FLOAT), INTENT(IN), OPTIONAL :: alpha, beta
          INTEGER(C_INT), INTENT(IN), OPTIONAL :: flags, prefetch
          fn = libxsmm_sfunction(                                       &
     &      m, n, k, lda, ldb, ldc, alpha, beta, flags, prefetch)
        END SUBROUTINE

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_ddispatch
        SUBROUTINE libxsmm_ddispatch(fn,                                &
     &  m, n, k, lda, ldb, ldc, alpha, beta, flags, prefetch)
          TYPE(LIBXSMM_DMM_FUNCTION), INTENT(OUT) :: fn
          INTEGER(C_INT), INTENT(IN) :: m, n, k
          INTEGER(C_INT), INTENT(IN), OPTIONAL :: lda, ldb, ldc
          REAL(C_DOUBLE), INTENT(IN), OPTIONAL :: alpha, beta
          INTEGER(C_INT), INTENT(IN), OPTIONAL :: flags, prefetch
          fn = libxsmm_dfunction(                                       &
     &      m, n, k, lda, ldb, ldc, alpha, beta, flags, prefetch)
        END SUBROUTINE

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_savailable
        LOGICAL PURE FUNCTION libxsmm_savailable(fn)
          TYPE(LIBXSMM_SMM_FUNCTION), INTENT(IN) :: fn
          libxsmm_savailable = ASSOCIATED(fn%fn0).OR.ASSOCIATED(fn%fn1)
        END FUNCTION

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_davailable
        LOGICAL PURE FUNCTION libxsmm_davailable(fn)
          TYPE(LIBXSMM_DMM_FUNCTION), INTENT(IN) :: fn
          libxsmm_davailable = ASSOCIATED(fn%fn0).OR.ASSOCIATED(fn%fn1)
        END FUNCTION

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_scall_abx
        PURE SUBROUTINE libxsmm_scall_abx(fn, a, b, c)
          TYPE(LIBXSMM_SMM_FUNCTION), INTENT(IN) :: fn
          TYPE(C_PTR), INTENT(IN), VALUE :: a, b, c
          CALL fn%fn0(a, b, c)
        END SUBROUTINE

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_dcall_abx
        PURE SUBROUTINE libxsmm_dcall_abx(fn, a, b, c)
          TYPE(LIBXSMM_DMM_FUNCTION), INTENT(IN) :: fn
          TYPE(C_PTR), INTENT(IN), VALUE :: a, b, c
          CALL fn%fn0(a, b, c)
        END SUBROUTINE

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_scall_prx
        PURE SUBROUTINE libxsmm_scall_prx(fn, a, b, c, pa, pb, pc)
          TYPE(LIBXSMM_SMM_FUNCTION), INTENT(IN) :: fn
          TYPE(C_PTR), INTENT(IN), VALUE :: a, b, c, pa, pb, pc
          CALL fn%fn1(a, b, c, pa, pb, pc)
        END SUBROUTINE

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_dcall_prx
        PURE SUBROUTINE libxsmm_dcall_prx(fn, a, b, c, pa, pb, pc)
          TYPE(LIBXSMM_DMM_FUNCTION), INTENT(IN) :: fn
          TYPE(C_PTR), INTENT(IN), VALUE :: a, b, c, pa, pb, pc
          CALL fn%fn1(a, b, c, pa, pb, pc)
        END SUBROUTINE

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_scall_abc
        SUBROUTINE libxsmm_scall_abc(fn, a, b, c)
          TYPE(LIBXSMM_SMM_FUNCTION), INTENT(IN) :: fn
          REAL(C_FLOAT), INTENT(IN), TARGET :: a(:,:), b(:,:)
          REAL(C_FLOAT), INTENT(INOUT), TARGET :: c(:,:)
          CALL libxsmm_scall_abx(fn,                                    &
     &      C_LOC(libxsmm_srealptr(a)),                                 &
     &      C_LOC(libxsmm_srealptr(b)),                                 &
     &      C_LOC(libxsmm_srealptr(c)))
        END SUBROUTINE

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_dcall_abc
        SUBROUTINE libxsmm_dcall_abc(fn, a, b, c)
          TYPE(LIBXSMM_DMM_FUNCTION), INTENT(IN) :: fn
          REAL(C_DOUBLE), INTENT(IN), TARGET :: a(:,:), b(:,:)
          REAL(C_DOUBLE), INTENT(INOUT), TARGET :: c(:,:)
          CALL libxsmm_dcall_abx(fn,                                    &
     &      C_LOC(libxsmm_drealptr(a)),                                 &
     &      C_LOC(libxsmm_drealptr(b)),                                 &
     &      C_LOC(libxsmm_drealptr(c)))
        END SUBROUTINE

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_scall_prf
        SUBROUTINE libxsmm_scall_prf(fn, a, b, c, pa, pb, pc)
          TYPE(LIBXSMM_SMM_FUNCTION), INTENT(IN) :: fn
          REAL(C_FLOAT), INTENT(IN), TARGET :: a(:,:), b(:,:)
          REAL(C_FLOAT), INTENT(INOUT), TARGET :: c(:,:)
          REAL(C_FLOAT), INTENT(IN), TARGET :: pa(*), pb(*), pc(*)
          CALL libxsmm_scall_prx(fn,                                    &
     &      C_LOC(libxsmm_srealptr(a)),                                 &
     &      C_LOC(libxsmm_srealptr(b)),                                 &
     &      C_LOC(libxsmm_srealptr(c)),                                 &
     &      C_LOC(pa), C_LOC(pb), C_LOC(pc))
        END SUBROUTINE

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_dcall_prf
        SUBROUTINE libxsmm_dcall_prf(fn, a, b, c, pa, pb, pc)
          TYPE(LIBXSMM_DMM_FUNCTION), INTENT(IN) :: fn
          REAL(C_DOUBLE), INTENT(IN), TARGET :: a(:,:), b(:,:)
          REAL(C_DOUBLE), INTENT(INOUT), TARGET :: c(:,:)
          REAL(C_DOUBLE), INTENT(IN), TARGET :: pa(*), pb(*), pc(*)
          CALL libxsmm_dcall_prx(fn,                                    &
     &      C_LOC(libxsmm_drealptr(a)),                                 &
     &      C_LOC(libxsmm_drealptr(b)),                                 &
     &      C_LOC(libxsmm_drealptr(c)),                                 &
     &      C_LOC(pa), C_LOC(pb), C_LOC(pc))
        END SUBROUTINE

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_scall
        SUBROUTINE libxsmm_scall(fn, a, b, c, pa, pb, pc)
          TYPE(LIBXSMM_SMM_FUNCTION), INTENT(IN) :: fn
          REAL(C_FLOAT), INTENT(IN), TARGET :: a(*), b(*)
          REAL(C_FLOAT), INTENT(INOUT), TARGET :: c(*)
          REAL(C_FLOAT), INTENT(IN), TARGET, OPTIONAL :: pa(*)
          REAL(C_FLOAT), INTENT(IN), TARGET, OPTIONAL :: pb(*)
          REAL(C_FLOAT), INTENT(IN), TARGET, OPTIONAL :: pc(*)
          IF (PRESENT(pa).OR.PRESENT(pb).OR.PRESENT(pc)) THEN
            CALL libxsmm_scall_prx(fn, C_LOC(a), C_LOC(b), C_LOC(c),    &
     &        MERGE(C_NULL_PTR, C_LOC(pa), .NOT.PRESENT(pa)),           &
     &        MERGE(C_NULL_PTR, C_LOC(pb), .NOT.PRESENT(pb)),           &
     &        MERGE(C_NULL_PTR, C_LOC(pc), .NOT.PRESENT(pc)))
          ELSE
            CALL libxsmm_scall_abx(fn, C_LOC(a), C_LOC(b), C_LOC(c))
          END IF
        END SUBROUTINE

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_dcall
        SUBROUTINE libxsmm_dcall(fn, a, b, c, pa, pb, pc)
          TYPE(LIBXSMM_DMM_FUNCTION), INTENT(IN) :: fn
          REAL(C_DOUBLE), INTENT(IN), TARGET :: a(*), b(*)
          REAL(C_DOUBLE), INTENT(INOUT), TARGET :: c(*)
          REAL(C_DOUBLE), INTENT(IN), TARGET, OPTIONAL :: pa(*)
          REAL(C_DOUBLE), INTENT(IN), TARGET, OPTIONAL :: pb(*)
          REAL(C_DOUBLE), INTENT(IN), TARGET, OPTIONAL :: pc(*)
          IF (PRESENT(pa).OR.PRESENT(pb).OR.PRESENT(pc)) THEN
            CALL libxsmm_dcall_prx(fn, C_LOC(a), C_LOC(b), C_LOC(c),    &
     &        MERGE(C_NULL_PTR, C_LOC(pa), .NOT.PRESENT(pa)),           &
     &        MERGE(C_NULL_PTR, C_LOC(pb), .NOT.PRESENT(pb)),           &
     &        MERGE(C_NULL_PTR, C_LOC(pc), .NOT.PRESENT(pc)))
          ELSE
            CALL libxsmm_dcall_abx(fn, C_LOC(a), C_LOC(b), C_LOC(c))
          END IF
        END SUBROUTINE

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_sgemm
        SUBROUTINE libxsmm_sgemm(transa, transb, m, n, k,               &
     &  alpha, a, lda, b, ldb, beta, c, ldc)
          CHARACTER(1), INTENT(IN), OPTIONAL :: transa, transb
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: m, n, k
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN), OPTIONAL :: lda
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN), OPTIONAL :: ldb
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN), OPTIONAL :: ldc
          REAL(C_FLOAT), INTENT(IN), OPTIONAL :: alpha, beta
          REAL(C_FLOAT), INTENT(IN) :: a(:,:), b(:,:)
          REAL(C_FLOAT), INTENT(INOUT) :: c(:,:)
          !DIR$ ATTRIBUTES OFFLOAD:MIC :: internal_gemm
          INTERFACE
            SUBROUTINE internal_gemm(transa, transb, m, n, k,           &
     &      alpha, a, lda, b, ldb, beta, c, ldc)                        &
     &      BIND(C, NAME="libxsmm_sgemm")
              IMPORT LIBXSMM_BLASINT_KIND, C_CHAR, C_FLOAT
              CHARACTER(C_CHAR), INTENT(IN) :: transa, transb
              INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: m, n, k
              INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: lda, ldb, ldc
              REAL(C_FLOAT), INTENT(IN) :: alpha, beta
              REAL(C_FLOAT), INTENT(IN) :: a(lda,*), b(ldb,*)
              REAL(C_FLOAT), INTENT(INOUT) :: c(ldc,*)
            END SUBROUTINE
          END INTERFACE
          CALL internal_gemm(transa, transb, m, n, k,                   &
     &      alpha, a, lda, b, ldb, beta, c, ldc)
        END SUBROUTINE

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_dgemm
        SUBROUTINE libxsmm_dgemm(transa, transb, m, n, k,               &
     &  alpha, a, lda, b, ldb, beta, c, ldc)
          CHARACTER(1), INTENT(IN), OPTIONAL :: transa, transb
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: m, n, k
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN), OPTIONAL :: lda
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN), OPTIONAL :: ldb
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN), OPTIONAL :: ldc
          REAL(C_DOUBLE), INTENT(IN), OPTIONAL :: alpha, beta
          REAL(C_DOUBLE), INTENT(IN) :: a(:,:), b(:,:)
          REAL(C_DOUBLE), INTENT(INOUT) :: c(:,:)
          !DIR$ ATTRIBUTES OFFLOAD:MIC :: internal_gemm
          INTERFACE
            SUBROUTINE internal_gemm(transa, transb, m, n, k,           &
     &      alpha, a, lda, b, ldb, beta, c, ldc)                        &
     &      BIND(C, NAME="libxsmm_dgemm")
              IMPORT LIBXSMM_BLASINT_KIND, C_CHAR, C_DOUBLE
              CHARACTER(C_CHAR), INTENT(IN) :: transa, transb
              INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: m, n, k
              INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: lda, ldb, ldc
              REAL(C_DOUBLE), INTENT(IN) :: alpha, beta
              REAL(C_DOUBLE), INTENT(IN) :: a(lda,*), b(ldb,*)
              REAL(C_DOUBLE), INTENT(INOUT) :: c(ldc,*)
            END SUBROUTINE
          END INTERFACE
          CALL internal_gemm(transa, transb, m, n, k,                   &
     &      alpha, a, lda, b, ldb, beta, c, ldc)
        END SUBROUTINE

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_blas_sgemm
        SUBROUTINE libxsmm_blas_sgemm(transa, transb, m, n, k,          &
     &  alpha, a, lda, b, ldb, beta, c, ldc)
          CHARACTER(1), INTENT(IN), OPTIONAL :: transa, transb
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: m, n, k
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN), OPTIONAL :: lda
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN), OPTIONAL :: ldb
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN), OPTIONAL :: ldc
          REAL(C_FLOAT), INTENT(IN), OPTIONAL :: alpha, beta
          REAL(C_FLOAT), INTENT(IN) :: a(:,:), b(:,:)
          REAL(C_FLOAT), INTENT(INOUT) :: c(:,:)
          !DIR$ ATTRIBUTES OFFLOAD:MIC :: internal_gemm
          INTERFACE
            SUBROUTINE internal_gemm(transa, transb, m, n, k,           &
     &      alpha, a, lda, b, ldb, beta, c, ldc)                        &
     &      BIND(C, NAME="libxsmm_blas_sgemm")
              IMPORT LIBXSMM_BLASINT_KIND, C_CHAR, C_FLOAT
              CHARACTER(C_CHAR), INTENT(IN) :: transa, transb
              INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: m, n, k
              INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: lda, ldb, ldc
              REAL(C_FLOAT), INTENT(IN) :: alpha, beta
              REAL(C_FLOAT), INTENT(IN) :: a(lda,*), b(ldb,*)
              REAL(C_FLOAT), INTENT(INOUT) :: c(ldc,*)
            END SUBROUTINE
          END INTERFACE
          CALL internal_gemm(transa, transb, m, n, k,                   &
     &      alpha, a, lda, b, ldb, beta, c, ldc)
        END SUBROUTINE

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_blas_dgemm
        SUBROUTINE libxsmm_blas_dgemm(transa, transb, m, n, k,          &
     &  alpha, a, lda, b, ldb, beta, c, ldc)
          CHARACTER(1), INTENT(IN), OPTIONAL :: transa, transb
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: m, n, k
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN), OPTIONAL :: lda
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN), OPTIONAL :: ldb
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN), OPTIONAL :: ldc
          REAL(C_DOUBLE), INTENT(IN), OPTIONAL :: alpha, beta
          REAL(C_DOUBLE), INTENT(IN) :: a(:,:), b(:,:)
          REAL(C_DOUBLE), INTENT(INOUT) :: c(:,:)
          !DIR$ ATTRIBUTES OFFLOAD:MIC :: internal_gemm
          INTERFACE
            SUBROUTINE internal_gemm(transa, transb, m, n, k,           &
     &      alpha, a, lda, b, ldb, beta, c, ldc)                        &
     &      BIND(C, NAME="libxsmm_blas_dgemm")
              IMPORT LIBXSMM_BLASINT_KIND, C_CHAR, C_DOUBLE
              CHARACTER(C_CHAR), INTENT(IN) :: transa, transb
              INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: m, n, k
              INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: lda, ldb, ldc
              REAL(C_DOUBLE), INTENT(IN) :: alpha, beta
              REAL(C_DOUBLE), INTENT(IN) :: a(lda,*), b(ldb,*)
              REAL(C_DOUBLE), INTENT(INOUT) :: c(ldc,*)
            END SUBROUTINE
          END INTERFACE
          CALL internal_gemm(transa, transb, m, n, k,                   &
     &      alpha, a, lda, b, ldb, beta, c, ldc)
        END SUBROUTINE
      END MODULE
