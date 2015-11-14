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
        INTEGER, PARAMETER ::                                           &
        ! Single-precision (FLS_KIND), and double-precision (FLD_KIND)
     &    LIBXSMM_FLS_KIND = SELECTED_REAL_KIND( 6,  30),               &
     &    LIBXSMM_FLD_KIND = SELECTED_REAL_KIND(14, 200),               &
        ! LP64: 32-bit with integers, and ILP64: 64-bit integers
     &    LIBXSMM_INT_KIND = $INTEGER_TYPE

        ! Parameters the library and static kernels were built for.
        INTEGER(LIBXSMM_INT_KIND), PARAMETER :: LIBXSMM_ALIGNMENT = $ALIGNMENT
        INTEGER(LIBXSMM_INT_KIND), PARAMETER :: LIBXSMM_ROW_MAJOR = $ROW_MAJOR
        INTEGER(LIBXSMM_INT_KIND), PARAMETER :: LIBXSMM_COL_MAJOR = $COL_MAJOR
        INTEGER(LIBXSMM_INT_KIND), PARAMETER :: LIBXSMM_PREFETCH = $PREFETCH
        INTEGER(LIBXSMM_INT_KIND), PARAMETER :: LIBXSMM_MAX_MNK = $MAX_MNK
        INTEGER(LIBXSMM_INT_KIND), PARAMETER :: LIBXSMM_MAX_M = $MAX_M
        INTEGER(LIBXSMM_INT_KIND), PARAMETER :: LIBXSMM_MAX_N = $MAX_N
        INTEGER(LIBXSMM_INT_KIND), PARAMETER :: LIBXSMM_MAX_K = $MAX_K
        INTEGER(LIBXSMM_INT_KIND), PARAMETER :: LIBXSMM_AVG_M = $AVG_M
        INTEGER(LIBXSMM_INT_KIND), PARAMETER :: LIBXSMM_AVG_N = $AVG_N
        INTEGER(LIBXSMM_INT_KIND), PARAMETER :: LIBXSMM_AVG_K = $AVG_K
        INTEGER(LIBXSMM_INT_KIND), PARAMETER :: LIBXSMM_FLAGS = $FLAGS
        INTEGER(LIBXSMM_INT_KIND), PARAMETER :: LIBXSMM_JIT = $JIT

        ! Parameters representing the GEMM performed by the simplified interface.
        REAL(LIBXSMM_FLD_KIND), PARAMETER ::                            &
     &    LIBXSMM_ALPHA = $ALPHA, LIBXSMM_BETA = $BETA

        ! Flag enumeration which can be IORed.
        INTEGER(LIBXSMM_INT_KIND), PARAMETER ::                         &
     &    LIBXSMM_GEMM_FLAG_TRANS_A = 1,                                &
     &    LIBXSMM_GEMM_FLAG_TRANS_B = 2,                                &
     &    LIBXSMM_GEMM_FLAG_ALIGN_A = 4,                                &
     &    LIBXSMM_GEMM_FLAG_ALIGN_C = 8

        ! Enumeration of the available prefetch strategies which can be IORed.
        !   LIBXSMM_PREFETCH_NONE:      No prefetching and no prefetch fn. signature.
        !   LIBXSMM_PREFETCH_SIGNATURE: Only function prefetch signature.
        !   LIBXSMM_PREFETCH_AL2:       Prefetch PA using accesses to A.
        !   LIBXSMM_PREFETCH_AL2_JPST:  Prefetch PA (aggressive).
        !   LIBXSMM_PREFETCH_BL2_VIA_C: Prefetch PB using accesses to C.
        !   LIBXSMM_PREFETCH_AL2_AHEAD: Prefetch A ahead.
        INTEGER(LIBXSMM_INT_KIND), PARAMETER ::                         &
     &    LIBXSMM_PREFETCH_NONE       = 0,                              &
     &    LIBXSMM_PREFETCH_SIGNATURE  = 1,                              &
     &    LIBXSMM_PREFETCH_AL2        = 2,                              &
     &    LIBXSMM_PREFETCH_AL2_JPST   = 4,                              &
     &    LIBXSMM_PREFETCH_BL2_VIA_C  = 8,                              &
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
          ! Specialized function with fused alpha and beta arguments (single-precision).
          PURE SUBROUTINE LIBXSMM_SFUNCTION(a, b, c) BIND(C)
            IMPORT :: C_FLOAT
            REAL(C_FLOAT), INTENT(IN) :: a(*), b(*)
            REAL(C_FLOAT), INTENT(INOUT) :: c(*)
          END SUBROUTINE

          ! Specialized function with fused alpha and beta arguments (double-precision).
          PURE SUBROUTINE LIBXSMM_DFUNCTION(a, b, c) BIND(C)
            IMPORT :: C_DOUBLE
            REAL(C_DOUBLE), INTENT(IN) :: a(*), b(*)
            REAL(C_DOUBLE), INTENT(INOUT) :: c(*)
          END SUBROUTINE

          ! Specialized function with alpha, beta, and prefetch arguments (single-precision).
          PURE SUBROUTINE LIBXSMM_SXFUNCTION(a, b, c,                   &
     &    pa, pb, pc) BIND(C)
            IMPORT :: C_FLOAT, C_PTR
            REAL(C_FLOAT), INTENT(IN) :: a(*), b(*)
            REAL(C_FLOAT), INTENT(INOUT) :: c(*)
            TYPE(C_PTR), INTENT(IN), VALUE :: pa, pb, pc
          END SUBROUTINE

          ! Specialized function with alpha, beta, and prefetch arguments (double-precision).
          PURE SUBROUTINE LIBXSMM_DXFUNCTION(a, b, c,                   &
     &    pa, pb, pc) BIND(C)
            IMPORT :: C_DOUBLE, C_PTR
            REAL(C_DOUBLE), INTENT(IN) :: a(*), b(*)
            REAL(C_DOUBLE), INTENT(INOUT) :: c(*)
            TYPE(C_PTR), INTENT(IN), VALUE :: pa, pb, pc
          END SUBROUTINE
        END INTERFACE

        ! Generic function type constructing a procedure pointer
        ! associated with a backend function.
        TYPE :: LIBXSMM_SMM_FUNCTION
          PROCEDURE(LIBXSMM_SFUNCTION), NOPASS, POINTER :: fn0
          PROCEDURE(LIBXSMM_SXFUNCTION), NOPASS, POINTER :: fn1
        END TYPE

        ! Generic function type constructing a procedure pointer
        ! associated with a backend function.
        TYPE :: LIBXSMM_DMM_FUNCTION
          PROCEDURE(LIBXSMM_DFUNCTION), NOPASS, POINTER :: fn0
          PROCEDURE(LIBXSMM_DXFUNCTION), NOPASS, POINTER :: fn1
        END TYPE

        ! Construct procedure pointer depending on given argument set.
        INTERFACE libxsmm_sdispatch
          MODULE PROCEDURE                                              &
     &      libxsmm_sfunction_mnk, libxsmm_sfunction_prf,               &
     &      libxsmm_sfunction_ldp, libxsmm_sfunction_abf,               &
     &      libxsmm_sfunction_all
        END INTERFACE

        ! Construct procedure pointer depending on given argument set.
        INTERFACE libxsmm_ddispatch
          MODULE PROCEDURE                                              &
     &      libxsmm_dfunction_mnk, libxsmm_dfunction_prf,               &
     &      libxsmm_dfunction_ldp, libxsmm_dfunction_abf,               &
     &      libxsmm_dfunction_all
        END INTERFACE

        ! Check if a function (LIBXSMM_?MM_FUNCTION_TYPE) is available.
        INTERFACE libxsmm_available
          MODULE PROCEDURE libxsmm_savailable, libxsmm_davailable
        END INTERFACE

        ! Call a specialized function.
        INTERFACE libxsmm_call
          MODULE PROCEDURE                                              &
     &      libxsmm_scall_abc, libxsmm_scall_prf,                       &
     &      libxsmm_dcall_abc, libxsmm_dcall_prf
        END INTERFACE

        ! Overloaded auto-dispatch routines (single/double precision).
        INTERFACE libxsmm_mm
          MODULE PROCEDURE libxsmm_smm, libxsmm_dmm
        END INTERFACE

        ! Overloaded BLAS routines (single/double precision).
        INTERFACE libxsmm_blasmm
          MODULE PROCEDURE                                              &
     &      libxsmm_sblasmm_ab, libxsmm_dblasmm_ab,                     &
     &      libxsmm_sblasmm, libxsmm_dblasmm
        END INTERFACE

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_init
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_sdispatch0, libxsmm_ddispatch0
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_sdispatch1, libxsmm_ddispatch1
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_timer_tick, libxsmm_timer_duration
        INTERFACE
          ! Initialize the library; pay for setup cost at a specific point.
          SUBROUTINE libxsmm_init() BIND(C)
          END SUBROUTINE

          SUBROUTINE libxsmm_finalize() BIND(C)
          END SUBROUTINE

          ! Query or JIT-generate a function; return zero if it does not exist,
          ! or if JIT is not supported (single-precision).
          TYPE(C_FUNPTR) PURE FUNCTION libxsmm_sdispatch0(              &
     &    flags, m, n, k, lda, ldb, ldc, alpha, beta)                   &
     &    BIND(C, NAME="libxsmm_sdispatch")
            IMPORT :: C_FUNPTR, C_INT, C_FLOAT
            INTEGER(C_INT), INTENT(IN), VALUE :: flags, m, n, k
            INTEGER(C_INT), INTENT(IN), VALUE :: lda, ldb, ldc
            REAL(C_FLOAT), INTENT(IN) :: alpha, beta
          END FUNCTION

          ! Query or JIT-generate a function; return zero if it does not exist,
          ! or if JIT is not supported (double-precision).
          TYPE(C_FUNPTR) PURE FUNCTION libxsmm_ddispatch0(              &
     &    flags, m, n, k, lda, ldb, ldc, alpha, beta)                   &
     &    BIND(C, NAME="libxsmm_ddispatch")
            IMPORT :: C_FUNPTR, C_INT, C_DOUBLE
            INTEGER(C_INT), INTENT(IN), VALUE :: flags, m, n, k
            INTEGER(C_INT), INTENT(IN), VALUE :: lda, ldb, ldc
            REAL(C_DOUBLE), INTENT(IN) :: alpha, beta
          END FUNCTION

          ! Query or JIT-generate a function; return zero if it does not exist,
          ! or if JIT is not supported (single-precision).
          TYPE(C_FUNPTR) PURE FUNCTION libxsmm_sdispatch1(              &
     &    flags, m, n, k, lda, ldb, ldc, alpha, beta, prefetch)         &
     &    BIND(C, NAME="libxsmm_sxdispatch")
            IMPORT :: C_FUNPTR, C_INT, C_FLOAT
            INTEGER(C_INT), INTENT(IN), VALUE :: flags, m, n, k
            INTEGER(C_INT), INTENT(IN), VALUE :: lda, ldb, ldc
            REAL(C_FLOAT), INTENT(IN) :: alpha, beta
            INTEGER(C_INT), INTENT(IN), VALUE :: prefetch
          END FUNCTION

          ! Query or JIT-generate a function; return zero if it does not exist,
          ! or if JIT is not supported (double-precision).
          TYPE(C_FUNPTR) PURE FUNCTION libxsmm_ddispatch1(              &
     &    flags, m, n, k, lda, ldb, ldc, alpha, beta, prefetch)         &
     &    BIND(C, NAME="libxsmm_dxdispatch")
            IMPORT :: C_FUNPTR, C_INT, C_DOUBLE
            INTEGER(C_INT), INTENT(IN), VALUE :: flags, m, n, k
            INTEGER(C_INT), INTENT(IN), VALUE :: lda, ldb, ldc
            REAL(C_DOUBLE), INTENT(IN) :: alpha, beta
            INTEGER(C_INT), INTENT(IN), VALUE :: prefetch
          END FUNCTION

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
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_sfunction_mnk
        TYPE(LIBXSMM_SMM_FUNCTION) FUNCTION libxsmm_sfunction_mnk(      &
     &  m, n, k, flags)
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN) :: m, n, k
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN), OPTIONAL :: flags
          REAL(LIBXSMM_FLS_KIND), POINTER :: rnull => NULL()
          PROCEDURE(LIBXSMM_SFUNCTION), POINTER :: fn0
          TYPE(C_FUNPTR) :: fn
          fn = libxsmm_sdispatch0(                                      &
     &      MERGE(LIBXSMM_FLAGS, flags, .NOT.PRESENT(flags)),           &
     &      m, n, k, 0, 0, 0, rnull, rnull)
          CALL C_F_PROCPOINTER(fn, fn0)
          libxsmm_sfunction_mnk%fn0 => fn0
        END FUNCTION

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_dfunction_mnk
        TYPE(LIBXSMM_DMM_FUNCTION) FUNCTION libxsmm_dfunction_mnk(      &
     &  m, n, k, flags)
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN) :: m, n, k
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN), OPTIONAL :: flags
          REAL(LIBXSMM_FLD_KIND), POINTER :: rnull => NULL()
          PROCEDURE(LIBXSMM_DFUNCTION), POINTER :: fn0
          TYPE(C_FUNPTR) :: fn
          fn = libxsmm_ddispatch0(                                      &
     &      MERGE(LIBXSMM_FLAGS, flags, .NOT.PRESENT(flags)),           &
     &      m, n, k, 0, 0, 0, rnull, rnull)
          CALL C_F_PROCPOINTER(fn, fn0)
          libxsmm_dfunction_mnk%fn0 => fn0
        END FUNCTION

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_sfunction_ldx
        TYPE(LIBXSMM_SMM_FUNCTION) FUNCTION libxsmm_sfunction_ldx(      &
     &  m, n, k, lda, ldb, ldc, flags)
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN) :: m, n, k
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN) :: lda, ldb, ldc
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN), OPTIONAL :: flags
          REAL(LIBXSMM_FLS_KIND), POINTER :: rnull => NULL()
          PROCEDURE(LIBXSMM_SFUNCTION), POINTER :: fn0
          INTEGER(LIBXSMM_INT_KIND) :: f
          TYPE(C_FUNPTR) :: fn
          f = MERGE(LIBXSMM_FLAGS, flags, .NOT.PRESENT(flags))
          fn = libxsmm_sdispatch0(f, m, n, k, 0, 0, 0, rnull, rnull)
          CALL C_F_PROCPOINTER(fn, fn0)
          libxsmm_sfunction_ldx%fn0 => fn0
        END FUNCTION

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_dfunction_ldx
        TYPE(LIBXSMM_DMM_FUNCTION) FUNCTION libxsmm_dfunction_ldx(      &
     &  m, n, k, lda, ldb, ldc, flags)
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN) :: m, n, k
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN) :: lda, ldb, ldc
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN), OPTIONAL :: flags
          REAL(LIBXSMM_FLD_KIND), POINTER :: rnull => NULL()
          PROCEDURE(LIBXSMM_DFUNCTION), POINTER :: fn0
          INTEGER(LIBXSMM_INT_KIND) :: f
          TYPE(C_FUNPTR) :: fn
          f = MERGE(LIBXSMM_FLAGS, flags, .NOT.PRESENT(flags))
          fn = libxsmm_ddispatch0(f, m, n, k, 0, 0, 0, rnull, rnull)
          CALL C_F_PROCPOINTER(fn, fn0)
          libxsmm_dfunction_ldx%fn0 => fn0
        END FUNCTION

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_sfunction_prf
        TYPE(LIBXSMM_SMM_FUNCTION) FUNCTION libxsmm_sfunction_prf(      &
     &  m, n, k, prefetch, flags)
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN) :: m, n, k
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN) :: prefetch
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN) :: flags
          REAL(LIBXSMM_FLS_KIND), POINTER :: rnull => NULL()
          PROCEDURE(LIBXSMM_SXFUNCTION), POINTER :: fn1
          PROCEDURE(LIBXSMM_SFUNCTION), POINTER :: fn0
          TYPE(C_FUNPTR) :: fn
          IF (LIBXSMM_PREFETCH_NONE.NE.prefetch) THEN
            fn = libxsmm_sdispatch1(flags, m, n, k, 0, 0, 0,            &
     &        rnull, rnull, prefetch)
            CALL C_F_PROCPOINTER(fn, fn1)
            libxsmm_sfunction_prf%fn1 => fn1
          ELSE
            fn = libxsmm_sdispatch0(flags, m, n, k, 0, 0, 0,            &
     &        rnull, rnull)
            CALL C_F_PROCPOINTER(fn, fn0)
            libxsmm_sfunction_prf%fn0 => fn0
          ENDIF
        END FUNCTION

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_dfunction_prf
        TYPE(LIBXSMM_DMM_FUNCTION) FUNCTION libxsmm_dfunction_prf(      &
     &  m, n, k, prefetch, flags)
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN) :: m, n, k
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN) :: prefetch
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN) :: flags
          REAL(LIBXSMM_FLD_KIND), POINTER :: rnull => NULL()
          PROCEDURE(LIBXSMM_DXFUNCTION), POINTER :: fn1
          PROCEDURE(LIBXSMM_DFUNCTION), POINTER :: fn0
          TYPE(C_FUNPTR) :: fn
          IF (LIBXSMM_PREFETCH_NONE.NE.prefetch) THEN
            fn = libxsmm_ddispatch1(flags, m, n, k, 0, 0, 0,            &
     &        rnull, rnull, prefetch)
            CALL C_F_PROCPOINTER(fn, fn1)
            libxsmm_dfunction_prf%fn1 => fn1
          ELSE
            fn = libxsmm_ddispatch0(flags, m, n, k, 0, 0, 0,            &
     &        rnull, rnull)
            CALL C_F_PROCPOINTER(fn, fn0)
            libxsmm_dfunction_prf%fn0 => fn0
          ENDIF
        END FUNCTION

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_sfunction_ldp
        TYPE(LIBXSMM_SMM_FUNCTION) FUNCTION libxsmm_sfunction_ldp(      &
     &  m, n, k, lda, ldb, ldc, prefetch, flags)
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN) :: m, n, k
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN) :: lda, ldb, ldc
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN), OPTIONAL :: prefetch
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN), OPTIONAL :: flags
          REAL(LIBXSMM_FLS_KIND), POINTER :: rnull => NULL()
          PROCEDURE(LIBXSMM_SXFUNCTION), POINTER :: fn1
          PROCEDURE(LIBXSMM_SFUNCTION), POINTER :: fn0
          INTEGER(LIBXSMM_INT_KIND) :: f
          TYPE(C_FUNPTR) :: fn
          f = MERGE(LIBXSMM_FLAGS, flags, .NOT.PRESENT(flags))
          IF (PRESENT(prefetch).AND.                                    &
     &    LIBXSMM_PREFETCH_NONE.NE.prefetch) THEN
            fn = libxsmm_sdispatch1(f, m, n, k, lda, ldb, ldc,          &
     &        rnull, rnull, prefetch)
            CALL C_F_PROCPOINTER(fn, fn1)
            libxsmm_sfunction_ldp%fn1 => fn1
          ELSE
            fn = libxsmm_sdispatch0(f, m, n, k, lda, ldb, ldc,          &
     &        rnull, rnull)
            CALL C_F_PROCPOINTER(fn, fn0)
            libxsmm_sfunction_ldp%fn0 => fn0
          ENDIF
        END FUNCTION

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_dfunction_ldp
        TYPE(LIBXSMM_DMM_FUNCTION) FUNCTION libxsmm_dfunction_ldp(      &
     &  m, n, k, lda, ldb, ldc, prefetch, flags)
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN) :: m, n, k
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN) :: lda, ldb, ldc
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN), OPTIONAL :: prefetch
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN), OPTIONAL :: flags
          REAL(LIBXSMM_FLD_KIND), POINTER :: rnull => NULL()
          PROCEDURE(LIBXSMM_DXFUNCTION), POINTER :: fn1
          PROCEDURE(LIBXSMM_DFUNCTION), POINTER :: fn0
          INTEGER(LIBXSMM_INT_KIND) :: f
          TYPE(C_FUNPTR) :: fn
          f = MERGE(LIBXSMM_FLAGS, flags, .NOT.PRESENT(flags))
          IF (PRESENT(prefetch).AND.                                    &
     &    LIBXSMM_PREFETCH_NONE.NE.prefetch) THEN
            fn = libxsmm_ddispatch1(f, m, n, k, lda, ldb, ldc,          &
     &        rnull, rnull, prefetch)
            CALL C_F_PROCPOINTER(fn, fn1)
            libxsmm_dfunction_ldp%fn1 => fn1
          ELSE
            fn = libxsmm_ddispatch0(f, m, n, k, lda, ldb, ldc,          &
     &        rnull, rnull)
            CALL C_F_PROCPOINTER(fn, fn0)
            libxsmm_dfunction_ldp%fn0 => fn0
          ENDIF
        END FUNCTION

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_sfunction_abf
        TYPE(LIBXSMM_SMM_FUNCTION) FUNCTION libxsmm_sfunction_abf(      &
     &  m, n, k, alpha, beta, flags, prefetch)
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN) :: m, n, k
          REAL(LIBXSMM_FLS_KIND), INTENT(IN) :: alpha, beta
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN), OPTIONAL :: flags
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN), OPTIONAL :: prefetch
          PROCEDURE(LIBXSMM_SXFUNCTION), POINTER :: fn1
          PROCEDURE(LIBXSMM_SFUNCTION), POINTER :: fn0
          INTEGER(LIBXSMM_INT_KIND) :: f
          TYPE(C_FUNPTR) :: fn
          f = MERGE(LIBXSMM_FLAGS, flags, .NOT.PRESENT(flags))
          IF (PRESENT(prefetch).AND.                                    &
     &    LIBXSMM_PREFETCH_NONE.NE.prefetch) THEN
            fn = libxsmm_sdispatch1(                                    &
     &        f, m, n, k, 0, 0, 0, alpha, beta, prefetch)
            CALL C_F_PROCPOINTER(fn, fn1)
            libxsmm_sfunction_abf%fn1 => fn1
          ELSE
            fn = libxsmm_sdispatch0(f, m, n, k, 0, 0, 0, alpha, beta)
            CALL C_F_PROCPOINTER(fn, fn0)
            libxsmm_sfunction_abf%fn0 => fn0
          ENDIF
        END FUNCTION

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_dfunction_abf
        TYPE(LIBXSMM_DMM_FUNCTION) FUNCTION libxsmm_dfunction_abf(      &
     &  m, n, k, alpha, beta, flags, prefetch)
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN) :: m, n, k
          REAL(LIBXSMM_FLD_KIND), INTENT(IN) :: alpha, beta
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN), OPTIONAL :: flags
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN), OPTIONAL :: prefetch
          PROCEDURE(LIBXSMM_DXFUNCTION), POINTER :: fn1
          PROCEDURE(LIBXSMM_DFUNCTION), POINTER :: fn0
          INTEGER(LIBXSMM_INT_KIND) :: f
          TYPE(C_FUNPTR) :: fn
          f = MERGE(LIBXSMM_FLAGS, flags, .NOT.PRESENT(flags))
          IF (PRESENT(prefetch).AND.                                    &
     &    LIBXSMM_PREFETCH_NONE.NE.prefetch) THEN
            fn = libxsmm_ddispatch1(                                    &
     &        f, m, n, k, 0, 0, 0, alpha, beta, prefetch)
            CALL C_F_PROCPOINTER(fn, fn1)
            libxsmm_dfunction_abf%fn1 => fn1
          ELSE
            fn = libxsmm_ddispatch0(f, m, n, k, 0, 0, 0, alpha, beta)
            CALL C_F_PROCPOINTER(fn, fn0)
            libxsmm_dfunction_abf%fn0 => fn0
          ENDIF
        END FUNCTION

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_sfunction_all
        TYPE(LIBXSMM_SMM_FUNCTION) FUNCTION libxsmm_sfunction_all(      &
     &  m, n, k, lda, ldb, ldc, alpha, beta, flags, prefetch)
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN) :: m, n, k
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN) :: lda, ldb, ldc
          REAL(LIBXSMM_FLS_KIND), INTENT(IN) :: alpha, beta
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN), OPTIONAL :: flags
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN), OPTIONAL :: prefetch
          PROCEDURE(LIBXSMM_SXFUNCTION), POINTER :: fn1
          PROCEDURE(LIBXSMM_SFUNCTION), POINTER :: fn0
          INTEGER(LIBXSMM_INT_KIND) :: f
          TYPE(C_FUNPTR) :: fn
          f = MERGE(LIBXSMM_FLAGS, flags, .NOT.PRESENT(flags))
          IF (PRESENT(prefetch).AND.                                    &
     &    LIBXSMM_PREFETCH_NONE.NE.prefetch) THEN
            fn = libxsmm_sdispatch1(                                    &
     &        f, m, n, k, lda, ldb, ldc, alpha, beta, prefetch)
            CALL C_F_PROCPOINTER(fn, fn1)
            libxsmm_sfunction_all%fn1 => fn1
          ELSE
            fn = libxsmm_sdispatch0(                                    &
     &        f, m, n, k, lda, ldb, ldc, alpha, beta)
            CALL C_F_PROCPOINTER(fn, fn0)
            libxsmm_sfunction_all%fn0 => fn0
          ENDIF
        END FUNCTION

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_dfunction_all
        TYPE(LIBXSMM_DMM_FUNCTION) FUNCTION libxsmm_dfunction_all(      &
     &  m, n, k, lda, ldb, ldc, alpha, beta, flags, prefetch)
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN) :: m, n, k
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN) :: lda, ldb, ldc
          REAL(LIBXSMM_FLD_KIND), INTENT(IN) :: alpha, beta
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN), OPTIONAL :: flags
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN), OPTIONAL :: prefetch
          PROCEDURE(LIBXSMM_DXFUNCTION), POINTER :: fn1
          PROCEDURE(LIBXSMM_DFUNCTION), POINTER :: fn0
          INTEGER(LIBXSMM_INT_KIND) :: f
          TYPE(C_FUNPTR) :: fn
          f = MERGE(LIBXSMM_FLAGS, flags, .NOT.PRESENT(flags))
          IF (PRESENT(prefetch).AND.                                    &
     &    LIBXSMM_PREFETCH_NONE.NE.prefetch) THEN
            fn = libxsmm_ddispatch1(                                    &
     &        f, m, n, k, lda, ldb, ldc, alpha, beta, prefetch)
            CALL C_F_PROCPOINTER(fn, fn1)
            libxsmm_dfunction_all%fn1 => fn1
          ELSE
            fn = libxsmm_ddispatch0(                                    &
     &        f, m, n, k, lda, ldb, ldc, alpha, beta)
            CALL C_F_PROCPOINTER(fn, fn0)
            libxsmm_dfunction_all%fn0 => fn0
          ENDIF
        END FUNCTION

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_savailable
        LOGICAL PURE FUNCTION libxsmm_savailable(fn)
          TYPE(LIBXSMM_SMM_FUNCTION), INTENT(IN) :: fn
          libxsmm_savailable =                                          &
     &      ASSOCIATED(fn%fn0).OR.ASSOCIATED(fn%fn1)
        END FUNCTION

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_davailable
        LOGICAL PURE FUNCTION libxsmm_davailable(fn)
          TYPE(LIBXSMM_DMM_FUNCTION), INTENT(IN) :: fn
          libxsmm_davailable =                                          &
     &      ASSOCIATED(fn%fn0).OR.ASSOCIATED(fn%fn1)
        END FUNCTION

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_scall_abc
        PURE SUBROUTINE libxsmm_scall_abc(fn, a, b, c)
          TYPE(LIBXSMM_SMM_FUNCTION), INTENT(IN) :: fn
          REAL(LIBXSMM_FLS_KIND), INTENT(IN) :: a(:,:), b(:,:)
          REAL(LIBXSMM_FLS_KIND), INTENT(INOUT) :: c(:,:)
          CALL fn%fn0(a, b, c)
        END SUBROUTINE

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_dcall_abc
        PURE SUBROUTINE libxsmm_dcall_abc(fn, a, b, c)
          TYPE(LIBXSMM_DMM_FUNCTION), INTENT(IN) :: fn
          REAL(LIBXSMM_FLD_KIND), INTENT(IN) :: a(:,:), b(:,:)
          REAL(LIBXSMM_FLD_KIND), INTENT(INOUT) :: c(:,:)
          CALL fn%fn0(a, b, c)
        END SUBROUTINE

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_scall_prf
        PURE SUBROUTINE libxsmm_scall_prf(fn, a, b, c, pa, pb, pc)
          TYPE(LIBXSMM_SMM_FUNCTION), INTENT(IN) :: fn
          REAL(LIBXSMM_FLS_KIND), INTENT(IN) :: a(:,:), b(:,:)
          REAL(LIBXSMM_FLS_KIND), INTENT(INOUT) :: c(:,:)
          TYPE(C_PTR), INTENT(IN), VALUE :: pa, pb, pc
          CALL fn%fn1(a, b, c, pa, pb, pc)
        END SUBROUTINE

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_dcall_prf
        PURE SUBROUTINE libxsmm_dcall_prf(fn, a, b, c, pa, pb, pc)
          TYPE(LIBXSMM_DMM_FUNCTION), INTENT(IN) :: fn
          REAL(LIBXSMM_FLD_KIND), INTENT(IN) :: a(:,:), b(:,:)
          REAL(LIBXSMM_FLD_KIND), INTENT(INOUT) :: c(:,:)
          TYPE(C_PTR), INTENT(IN), VALUE :: pa, pb, pc
          CALL fn%fn1(a, b, c, pa, pb, pc)
        END SUBROUTINE

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_align_value
        INTEGER(LIBXSMM_INT_KIND) PURE FUNCTION libxsmm_align_value(    &
     &    n, typesize, alignment)
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN) :: n, typesize
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN) :: alignment
          libxsmm_align_value = (((n * typesize + alignment - 1) /      &
     &      alignment) * alignment) / typesize
        END FUNCTION

        ! Non-dispatched matrix multiplication using BLAS (single-precision).
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_sblasmm
        SUBROUTINE libxsmm_sblasmm(m, n, k, a, b, c, flags, alpha, beta)
          INTEGER(LIBXSMM_INT_KIND), PARAMETER :: T = LIBXSMM_FLS_KIND
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN) :: m, n, k
          REAL(T), INTENT(IN) :: a(:,:), b(:,:)
          REAL(T), INTENT(INOUT) :: c(:,:)
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN), OPTIONAL :: flags
          REAL(T), INTENT(IN), OPTIONAL :: alpha, beta
          INTEGER(LIBXSMM_INT_KIND) :: f
          !DIR$ ATTRIBUTES OFFLOAD:MIC :: sgemm
          INTERFACE
            SUBROUTINE sgemm(transa, transb, m, n, k,                   &
     &      alpha, a, lda, b, ldb, beta, c, ldc)
              IMPORT LIBXSMM_INT_KIND, LIBXSMM_FLS_KIND
              CHARACTER(1), INTENT(IN) :: transa, transb
              INTEGER(LIBXSMM_INT_KIND), INTENT(IN) :: m, n, k
              INTEGER(LIBXSMM_INT_KIND), INTENT(IN) :: lda, ldb, ldc
              REAL(LIBXSMM_FLS_KIND), INTENT(IN) :: alpha, beta
              REAL(LIBXSMM_FLS_KIND), INTENT(IN) :: a(lda,*), b(ldb,*)
              REAL(LIBXSMM_FLS_KIND), INTENT(INOUT) :: c(ldc,*)
            END SUBROUTINE
          END INTERFACE
          f = MERGE(LIBXSMM_FLAGS, flags, .NOT.PRESENT(flags))
          IF (0.NE.LIBXSMM_COL_MAJOR) THEN
            CALL sgemm(                                                 &
     &        MERGE('N', 'T',                                           &
     &            0.EQ.IAND(LIBXSMM_GEMM_FLAG_TRANS_A, f)),             &
     &        MERGE('N', 'T',                                           &
     &            0.EQ.IAND(LIBXSMM_GEMM_FLAG_TRANS_B, f)),             &
     &        m, n, k,                                                  &
     &        MERGE(REAL(LIBXSMM_ALPHA, T), alpha, .NOT.PRESENT(alpha)),&
     &        a, MAX(SIZE(a, 1), m), b, MAX(SIZE(b, 1), k),             &
     &        MERGE(REAL(LIBXSMM_BETA, T), beta, .NOT.PRESENT(beta)),   &
     &        c, MAX(SIZE(c, 1), m))
          ELSE
            CALL sgemm(                                                 &
     &        MERGE('N', 'T',                                           &
     &            0.EQ.IAND(LIBXSMM_GEMM_FLAG_TRANS_A, f)),             &
     &        MERGE('N', 'T',                                           &
     &            0.EQ.IAND(LIBXSMM_GEMM_FLAG_TRANS_B, f)),             &
     &        n, m, k,                                                  &
     &        MERGE(REAL(LIBXSMM_ALPHA, T), alpha, .NOT.PRESENT(alpha)),&
     &        b, MAX(SIZE(b, 2), n), a, MAX(SIZE(a, 2), k),             &
     &        MERGE(REAL(LIBXSMM_BETA, T), beta, .NOT.PRESENT(beta)),   &
     &        c, MAX(SIZE(c, 1), n))
          ENDIF
        END SUBROUTINE

        ! Non-dispatched matrix multiplication using BLAS (double-precision).
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_dblasmm
        SUBROUTINE libxsmm_dblasmm(m, n, k, a, b, c, flags, alpha, beta)
          INTEGER(LIBXSMM_INT_KIND), PARAMETER :: T = LIBXSMM_FLD_KIND
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN) :: m, n, k
          REAL(T), INTENT(IN) :: a(:,:), b(:,:)
          REAL(T), INTENT(INOUT) :: c(:,:)
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN), OPTIONAL :: flags
          REAL(T), INTENT(IN), OPTIONAL :: alpha, beta
          INTEGER(LIBXSMM_INT_KIND) :: f
          !DIR$ ATTRIBUTES OFFLOAD:MIC :: dgemm
          INTERFACE
            SUBROUTINE dgemm(transa, transb, m, n, k,                   &
     &      alpha, a, lda, b, ldb, beta, c, ldc)
              IMPORT LIBXSMM_INT_KIND, LIBXSMM_FLD_KIND
              CHARACTER(1), INTENT(IN) :: transa, transb
              INTEGER(LIBXSMM_INT_KIND), INTENT(IN) :: m, n, k
              INTEGER(LIBXSMM_INT_KIND), INTENT(IN) :: lda, ldb, ldc
              REAL(LIBXSMM_FLD_KIND), INTENT(IN) :: alpha, beta
              REAL(LIBXSMM_FLD_KIND), INTENT(IN) :: a(lda,*), b(ldb,*)
              REAL(LIBXSMM_FLD_KIND), INTENT(INOUT) :: c(ldc,*)
            END SUBROUTINE
          END INTERFACE
          f = MERGE(LIBXSMM_FLAGS, flags, .NOT.PRESENT(flags))
          IF (0.NE.LIBXSMM_COL_MAJOR) THEN
            CALL dgemm(                                                 &
     &        MERGE('N', 'T',                                           &
     &            0.EQ.IAND(LIBXSMM_GEMM_FLAG_TRANS_A, f)),             &
     &        MERGE('N', 'T',                                           &
     &            0.EQ.IAND(LIBXSMM_GEMM_FLAG_TRANS_B, f)),             &
     &        m, n, k,                                                  &
     &        MERGE(REAL(LIBXSMM_ALPHA, T), alpha, .NOT.PRESENT(alpha)),&
     &        a, MAX(SIZE(a, 1), m), b, MAX(SIZE(b, 1), k),             &
     &        MERGE(REAL(LIBXSMM_BETA, T), beta, .NOT.PRESENT(beta)),   &
     &        c, MAX(SIZE(c, 1), m))
          ELSE
            CALL dgemm(                                                 &
     &        MERGE('N', 'T',                                           &
     &            0.EQ.IAND(LIBXSMM_GEMM_FLAG_TRANS_A, f)),             &
     &        MERGE('N', 'T',                                           &
     &            0.EQ.IAND(LIBXSMM_GEMM_FLAG_TRANS_B, f)),             &
     &        n, m, k,                                                  &
     &        MERGE(REAL(LIBXSMM_ALPHA, T), alpha, .NOT.PRESENT(alpha)),&
     &        b, MAX(SIZE(b, 2), n), a, MAX(SIZE(a, 2), k),             &
     &        MERGE(REAL(LIBXSMM_BETA, T), beta, .NOT.PRESENT(beta)),   &
     &        c, MAX(SIZE(c, 1), n))
          ENDIF
        END SUBROUTINE

        ! Non-dispatched matrix multiplication using BLAS (single-precision).
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_sblasmm_ab
        SUBROUTINE libxsmm_sblasmm_ab(                                  &
     &  m, n, k, a, b, c, salpha, sbeta, flags)
          INTEGER(LIBXSMM_INT_KIND), PARAMETER :: T = LIBXSMM_FLS_KIND
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN) :: m, n, k
          REAL(T), INTENT(IN) :: a(:,:), b(:,:)
          REAL(T), INTENT(INOUT) :: c(:,:)
          REAL(T), INTENT(IN) :: salpha, sbeta
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN), OPTIONAL :: flags
          CALL libxsmm_sblasmm(m, n, k, a, b, c,                        &
     &      MERGE(LIBXSMM_FLAGS, flags, .NOT.PRESENT(flags)),           &
     &      salpha, sbeta)
        END SUBROUTINE

        ! Non-dispatched matrix multiplication using BLAS (single-precision).
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_dblasmm_ab
        SUBROUTINE libxsmm_dblasmm_ab(                                  &
     &  m, n, k, a, b, c, dalpha, dbeta, flags)
          INTEGER(LIBXSMM_INT_KIND), PARAMETER :: T = LIBXSMM_FLD_KIND
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN) :: m, n, k
          REAL(T), INTENT(IN) :: a(:,:), b(:,:)
          REAL(T), INTENT(INOUT) :: c(:,:)
          REAL(T), INTENT(IN) :: dalpha, dbeta
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN), OPTIONAL :: flags
          CALL libxsmm_dblasmm(m, n, k, a, b, c,                        &
     &      MERGE(LIBXSMM_FLAGS, flags, .NOT.PRESENT(flags)),           &
     &      dalpha, dbeta)
        END SUBROUTINE

        ! Dispatched matrix multiplication (single-precision).
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_smm
        SUBROUTINE libxsmm_smm(                                         &
     &  m, n, k, a, b, c, pa, pb, pc, flags, alpha, beta)
          INTEGER(LIBXSMM_INT_KIND), PARAMETER :: T = LIBXSMM_FLS_KIND
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN) :: m, n, k
          REAL(T), INTENT(IN) :: a(:,:), b(:,:)
          REAL(T), INTENT(INOUT) :: c(:,:)
          TYPE(C_PTR), INTENT(IN), VALUE, OPTIONAL :: pa, pb, pc
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN), OPTIONAL :: flags
          REAL(T), INTENT(IN), OPTIONAL :: alpha, beta
          TYPE(LIBXSMM_SMM_FUNCTION) :: xmm
          INTEGER(LIBXSMM_INT_KIND) :: f
          f = MERGE(LIBXSMM_FLAGS, flags, .NOT.PRESENT(flags))
          IF (LIBXSMM_MAX_MNK.GE.(m * n * k)) THEN
            IF (PRESENT(pa).OR.PRESENT(pb).OR.PRESENT(pc)) THEN
              xmm = libxsmm_sfunction_abf(m, n, k, alpha, beta, f,      &
     &          MERGE(LIBXSMM_PREFETCH, LIBXSMM_PREFETCH_SIGNATURE,     &
     &              LIBXSMM_PREFETCH_NONE.NE.LIBXSMM_PREFETCH))
              IF (libxsmm_savailable(xmm)) THEN
                CALL libxsmm_scall_prf(xmm, a, b, c, pa, pb, pc)
              ELSE
                CALL libxsmm_sblasmm(m, n, k, a, b, c, f, alpha, beta)
              ENDIF
            ELSE
              xmm = libxsmm_sfunction_abf(m, n, k, alpha, beta, f)
              IF (libxsmm_savailable(xmm)) THEN
                CALL libxsmm_scall_abc(xmm, a, b, c)
              ELSE
                CALL libxsmm_sblasmm(m, n, k, a, b, c, f, alpha, beta)
              ENDIF
            ENDIF
          ELSE
            CALL libxsmm_sblasmm(m, n, k, a, b, c, f, alpha, beta)
          ENDIF
        END SUBROUTINE

        ! Dispatched matrix multiplication (double-precision).
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_dmm
        SUBROUTINE libxsmm_dmm(                                         &
     &  m, n, k, a, b, c, pa, pb, pc, flags, alpha, beta)
          INTEGER(LIBXSMM_INT_KIND), PARAMETER :: T = LIBXSMM_FLD_KIND
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN) :: m, n, k
          REAL(T), INTENT(IN) :: a(:,:), b(:,:)
          REAL(T), INTENT(INOUT) :: c(:,:)
          TYPE(C_PTR), INTENT(IN), VALUE, OPTIONAL :: pa, pb, pc
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN), OPTIONAL :: flags
          REAL(T), INTENT(IN), OPTIONAL :: alpha, beta
          TYPE(LIBXSMM_DMM_FUNCTION) :: xmm
          INTEGER(LIBXSMM_INT_KIND) :: f
          f = MERGE(LIBXSMM_FLAGS, flags, .NOT.PRESENT(flags))
          IF (LIBXSMM_MAX_MNK.GE.(m * n * k)) THEN
            IF (PRESENT(pa).OR.PRESENT(pb).OR.PRESENT(pc)) THEN
              xmm = libxsmm_dfunction_abf(m, n, k, alpha, beta, f,      &
     &          MERGE(LIBXSMM_PREFETCH, LIBXSMM_PREFETCH_SIGNATURE,     &
     &              LIBXSMM_PREFETCH_NONE.NE.LIBXSMM_PREFETCH))
              IF (libxsmm_davailable(xmm)) THEN
                CALL libxsmm_dcall_prf(xmm, a, b, c, pa, pb, pc)
              ELSE
                CALL libxsmm_dblasmm(m, n, k, a, b, c, f, alpha, beta)
              ENDIF
            ELSE
              xmm = libxsmm_dfunction_abf(m, n, k, alpha, beta, f)
              IF (libxsmm_davailable(xmm)) THEN
                CALL libxsmm_dcall_abc(xmm, a, b, c)
              ELSE
                CALL libxsmm_dblasmm(m, n, k, a, b, c, f, alpha, beta)
              ENDIF
            ENDIF
          ELSE
            CALL libxsmm_dblasmm(m, n, k, a, b, c, f, alpha, beta)
          ENDIF
        END SUBROUTINE
      END MODULE
