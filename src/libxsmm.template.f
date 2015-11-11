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

        ! Parameters the library was built for.
        INTEGER(LIBXSMM_INT_KIND), PARAMETER ::                         &
     &    LIBXSMM_ALIGNMENT = $ALIGNMENT
        INTEGER(LIBXSMM_INT_KIND), PARAMETER ::                         &
     &    LIBXSMM_PREFETCH = $PREFETCH
        INTEGER(LIBXSMM_INT_KIND), PARAMETER ::                         &
     &    LIBXSMM_ROW_MAJOR = $ROW_MAJOR
        INTEGER(LIBXSMM_INT_KIND), PARAMETER ::                         &
     &    LIBXSMM_COL_MAJOR = $COL_MAJOR
        INTEGER(LIBXSMM_INT_KIND), PARAMETER ::                         &
     &    LIBXSMM_MAX_MNK = $MAX_MNK
        INTEGER(LIBXSMM_INT_KIND), PARAMETER ::                         &
     &    LIBXSMM_MAX_M = $MAX_M
        INTEGER(LIBXSMM_INT_KIND), PARAMETER ::                         &
     &    LIBXSMM_MAX_N = $MAX_N
        INTEGER(LIBXSMM_INT_KIND), PARAMETER ::                         &
     &    LIBXSMM_MAX_K = $MAX_K
        INTEGER(LIBXSMM_INT_KIND), PARAMETER ::                         &
     &    LIBXSMM_AVG_M = $AVG_M
        INTEGER(LIBXSMM_INT_KIND), PARAMETER ::                         &
     &    LIBXSMM_AVG_N = $AVG_N
        INTEGER(LIBXSMM_INT_KIND), PARAMETER ::                         &
     &    LIBXSMM_AVG_K = $AVG_K
        INTEGER(LIBXSMM_INT_KIND), PARAMETER ::                         &
     &    LIBXSMM_JIT = $JIT

        ! Parameters representing the GEMM performed by the simplified interface.
        REAL(LIBXSMM_FLD_KIND), PARAMETER ::                            &
     &    LIBXSMM_ALPHA = $ALPHA
        REAL(LIBXSMM_FLD_KIND), PARAMETER ::                            &
     &    LIBXSMM_BETA = $BETA

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

        ! Type of a function generated for a specific M, N, K,
        ! and for a specific LDA, LDB, LDC, Alpha, and Beta.
        ABSTRACT INTERFACE
          PURE SUBROUTINE LIBXSMM_SMM_FUNCTION0(a, b, c) BIND(C)
            IMPORT :: C_FLOAT
            REAL(C_FLOAT), INTENT(IN) :: a(*), b(*)
            REAL(C_FLOAT), INTENT(INOUT) :: c(*)
          END SUBROUTINE
          PURE SUBROUTINE LIBXSMM_DMM_FUNCTION0(a, b, c) BIND(C)
            IMPORT :: C_DOUBLE
            REAL(C_DOUBLE), INTENT(IN) :: a(*), b(*)
            REAL(C_DOUBLE), INTENT(INOUT) :: c(*)
          END SUBROUTINE
          PURE SUBROUTINE LIBXSMM_SMM_FUNCTION1(a, b, c,                &
     &    pa, pb, pc) BIND(C)
            IMPORT :: C_FLOAT
            REAL(C_FLOAT), INTENT(IN) :: a(*), b(*)
            REAL(C_FLOAT), INTENT(INOUT) :: c(*)
            REAL(C_FLOAT), INTENT(IN) :: pa(*), pb(*), pc(*)
          END SUBROUTINE
          PURE SUBROUTINE LIBXSMM_DMM_FUNCTION1(a, b, c,                &
     &    pa, pb, pc) BIND(C)
            IMPORT :: C_DOUBLE
            REAL(C_DOUBLE), INTENT(IN) :: a(*), b(*)
            REAL(C_DOUBLE), INTENT(INOUT) :: c(*)
            REAL(C_DOUBLE), INTENT(IN) :: pa(*), pb(*), pc(*)
          END SUBROUTINE
          PURE SUBROUTINE LIBXSMM_SMM_FUNCTION2(a, b, c,                &
     &    pa, pb, pc, alpha, beta) BIND(C)
            IMPORT :: C_FLOAT
            REAL(C_FLOAT), INTENT(IN) :: a(*), b(*)
            REAL(C_FLOAT), INTENT(INOUT) :: c(*)
            REAL(C_FLOAT), INTENT(IN) :: pa(*), pb(*), pc(*)
            REAL(C_FLOAT), INTENT(IN), VALUE :: alpha, beta
          END SUBROUTINE
          PURE SUBROUTINE LIBXSMM_DMM_FUNCTION2(a, b, c,                &
     &    pa, pb, pc, alpha, beta) BIND(C)
            IMPORT :: C_DOUBLE
            REAL(C_DOUBLE), INTENT(IN) :: a(*), b(*)
            REAL(C_DOUBLE), INTENT(INOUT) :: c(*)
            REAL(C_DOUBLE), INTENT(IN) :: pa(*), pb(*), pc(*)
            REAL(C_DOUBLE), INTENT(IN), VALUE :: alpha, beta
          END SUBROUTINE
        END INTERFACE

        ! Generic function type constructing a procedure pointer
        ! associated with a backend function.
        TYPE :: LIBXSMM_SMM_FUNCTION
          PROCEDURE(LIBXSMM_SMM_FUNCTION0), NOPASS, POINTER :: smm0
          PROCEDURE(LIBXSMM_SMM_FUNCTION1), NOPASS, POINTER :: smm1
          PROCEDURE(LIBXSMM_SMM_FUNCTION2), NOPASS, POINTER :: smm2
        END TYPE

        ! Generic function type constructing a procedure pointer
        ! associated with a backend function.
        TYPE :: LIBXSMM_DMM_FUNCTION
          PROCEDURE(LIBXSMM_DMM_FUNCTION0), NOPASS, POINTER :: dmm0
          PROCEDURE(LIBXSMM_DMM_FUNCTION1), NOPASS, POINTER :: dmm1
          PROCEDURE(LIBXSMM_DMM_FUNCTION2), NOPASS, POINTER :: dmm2
        END TYPE

        ! Construct procedure pointer depending on given argument set.
        INTERFACE libxsmm_sdispatch
          MODULE PROCEDURE                                              &
     &      libxsmm_smm_function_mnk, libxsmm_smm_function_ldx,         &
     &      libxsmm_smm_function_prf, libxsmm_smm_function_ldf,         &
     &      libxsmm_smm_function_abf, libxsmm_smm_function_all
        END INTERFACE

        ! Construct procedure pointer depending on given argument set.
        INTERFACE libxsmm_ddispatch
          MODULE PROCEDURE                                              &
     &      libxsmm_dmm_function_mnk, libxsmm_dmm_function_ldx,         &
     &      libxsmm_dmm_function_prf, libxsmm_dmm_function_ldf,         &
     &      libxsmm_dmm_function_abf, libxsmm_dmm_function_all
        END INTERFACE

        ! Check if a function (LIBXSMM_?MM_FUNCTION_TYPE) is available.
        INTERFACE libxsmm_available
          MODULE PROCEDURE libxsmm_savailable, libxsmm_davailable
        END INTERFACE

        ! Overloaded auto-dispatch routines (single/double precision).
        !INTERFACE libxsmm_mm
          !MODULE PROCEDURE libxsmm_smm, libxsmm_dmm
        !END INTERFACE

        ! Overloaded BLAS routines (single/double precision).
        INTERFACE libxsmm_blasmm
          MODULE PROCEDURE libxsmm_sblasmm, libxsmm_dblasmm
        END INTERFACE

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: sgemm, dgemm
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_init
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_sdispatch0, libxsmm_ddispatch0
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_sdispatch1, libxsmm_ddispatch1
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_sdispatch2, libxsmm_ddispatch2
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_timer_tick, libxsmm_timer_duration
        INTERFACE
          SUBROUTINE sgemm(                                             &
     &    transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
            IMPORT LIBXSMM_INT_KIND, LIBXSMM_FLS_KIND
            CHARACTER(1), INTENT(IN) :: transa, transb
            INTEGER(LIBXSMM_INT_KIND), INTENT(IN) :: m, n, k
            INTEGER(LIBXSMM_INT_KIND), INTENT(IN) :: lda, ldb, ldc
            REAL(LIBXSMM_FLS_KIND), INTENT(IN) :: alpha, beta
            REAL(LIBXSMM_FLS_KIND), INTENT(IN) :: a(lda,*), b(ldb,*)
            REAL(LIBXSMM_FLS_KIND), INTENT(INOUT) :: c(ldc,*)
          END SUBROUTINE
          SUBROUTINE dgemm(                                             &
     &    transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
            IMPORT LIBXSMM_INT_KIND, LIBXSMM_FLD_KIND
            CHARACTER(1), INTENT(IN) :: transa, transb
            INTEGER(LIBXSMM_INT_KIND), INTENT(IN) :: m, n, k
            INTEGER(LIBXSMM_INT_KIND), INTENT(IN) :: lda, ldb, ldc
            REAL(LIBXSMM_FLD_KIND), INTENT(IN) :: alpha, beta
            REAL(LIBXSMM_FLD_KIND), INTENT(IN) :: a(lda,*), b(ldb,*)
            REAL(LIBXSMM_FLD_KIND), INTENT(INOUT) :: c(ldc,*)
          END SUBROUTINE

          ! Initialize the library; pay for setup cost at a specific point.
          SUBROUTINE libxsmm_init() BIND(C)
          END SUBROUTINE

          ! Query or JIT-generate a function; return zero if it does not exist,
          ! or if JIT is not supported (single-precision).
          TYPE(C_FUNPTR) PURE FUNCTION libxsmm_sdispatch0(              &
     &    m, n, k, lda, ldb, ldc, flags)                                &
     &    BIND(C, NAME="libxsmm_sdispatch")
            IMPORT :: C_FUNPTR, C_INT
            INTEGER(C_INT), INTENT(IN), VALUE :: m, n, k, lda, ldb, ldc
            INTEGER(C_INT), INTENT(IN), VALUE :: flags
          END FUNCTION
          ! Query or JIT-generate a function; return zero if it does not exist,
          ! or if JIT is not supported (double-precision).
          TYPE(C_FUNPTR) PURE FUNCTION libxsmm_ddispatch0(              &
     &    m, n, k, lda, ldb, ldc, flags)                                &
     &    BIND(C, NAME="libxsmm_ddispatch")
            IMPORT :: C_FUNPTR, C_INT
            INTEGER(C_INT), INTENT(IN), VALUE :: m, n, k, lda, ldb, ldc
            INTEGER(C_INT), INTENT(IN), VALUE :: flags
          END FUNCTION

          ! Query or JIT-generate a function; return zero if it does not exist,
          ! or if JIT is not supported (single-precision).
          TYPE(C_FUNPTR) PURE FUNCTION libxsmm_sdispatch1(              &
     &    m, n, k, lda, ldb, ldc, flags, prefetch) BIND(C)
            IMPORT :: C_FUNPTR, C_INT
            INTEGER(C_INT), INTENT(IN), VALUE :: m, n, k, lda, ldb, ldc
            INTEGER(C_INT), INTENT(IN), VALUE :: flags, prefetch
          END FUNCTION
          ! Query or JIT-generate a function; return zero if it does not exist,
          ! or if JIT is not supported (double-precision).
          TYPE(C_FUNPTR) PURE FUNCTION libxsmm_ddispatch1(              &
     &    m, n, k, lda, ldb, ldc, flags, prefetch) BIND(C)
            IMPORT :: C_FUNPTR, C_INT
            INTEGER(C_INT), INTENT(IN), VALUE :: m, n, k, lda, ldb, ldc
            INTEGER(C_INT), INTENT(IN), VALUE :: flags, prefetch
          END FUNCTION

          ! Query or JIT-generate a function; return zero if it does not exist,
          ! or if JIT is not supported (single-precision).
          TYPE(C_FUNPTR) PURE FUNCTION libxsmm_sdispatch2(              &
     &    m, n, k, lda, ldb, ldc, flags, prefetch, alpha, beta) BIND(C)
            IMPORT :: C_FUNPTR, C_FLOAT, C_INT
            INTEGER(C_INT), INTENT(IN), VALUE :: m, n, k, lda, ldb, ldc
            INTEGER(C_INT), INTENT(IN), VALUE :: flags, prefetch
            REAL(C_FLOAT),  INTENT(IN), VALUE :: alpha, beta
          END FUNCTION
          ! Query or JIT-generate a function; return zero if it does not exist,
          ! or if JIT is not supported (double-precision).
          TYPE(C_FUNPTR) PURE FUNCTION libxsmm_ddispatch2(              &
     &    m, n, k, lda, ldb, ldc, flags, prefetch, alpha, beta) BIND(C)
            IMPORT :: C_FUNPTR, C_DOUBLE, C_INT
            INTEGER(C_INT), INTENT(IN), VALUE :: m, n, k, lda, ldb, ldc
            INTEGER(C_INT), INTENT(IN), VALUE :: flags, prefetch
            REAL(C_DOUBLE), INTENT(IN), VALUE :: alpha, beta
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
        TYPE(LIBXSMM_SMM_FUNCTION)                                      &
     &  FUNCTION libxsmm_smm_function_mnk(m, n, k, flags)
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN) :: m, n, k
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN), OPTIONAL :: flags
          TYPE(C_FUNPTR) :: fn
          fn = libxsmm_sdispatch0(m, n, k, 0, 0, 0,                     &
     &      MERGE(0, flags, .NOT.PRESENT(flags)))
          CALL C_F_PROCPOINTER(fn, libxsmm_smm_function_mnk%smm0)
        END FUNCTION
        TYPE(LIBXSMM_DMM_FUNCTION)                                      &
     &  FUNCTION libxsmm_dmm_function_mnk(m, n, k, flags)
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN) :: m, n, k
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN), OPTIONAL :: flags
          TYPE(C_FUNPTR) :: fn
          fn = libxsmm_ddispatch0(m, n, k, 0, 0, 0,                     &
     &      MERGE(0, flags, .NOT.PRESENT(flags)))
          CALL C_F_PROCPOINTER(fn, libxsmm_dmm_function_mnk%dmm0)
        END FUNCTION

        TYPE(LIBXSMM_SMM_FUNCTION)                                      &
     &  FUNCTION libxsmm_smm_function_ldx(                              &
     &  m, n, k, lda, ldb, ldc, flags)
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN) :: m, n, k
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN) :: lda, ldb, ldc
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN), OPTIONAL :: flags
          TYPE(C_FUNPTR) :: fn
          fn = libxsmm_sdispatch0(m, n, k, lda, ldb, ldc,               &
     &      MERGE(0, flags, .NOT.PRESENT(flags)))
          CALL C_F_PROCPOINTER(fn, libxsmm_smm_function_ldx%smm0)
        END FUNCTION
        TYPE(LIBXSMM_DMM_FUNCTION)                                      &
     &  FUNCTION libxsmm_dmm_function_ldx(                              &
     &  m, n, k, lda, ldb, ldc, flags)
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN) :: m, n, k
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN) :: lda, ldb, ldc
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN), OPTIONAL :: flags
          TYPE(C_FUNPTR) :: fn
          fn = libxsmm_ddispatch0(m, n, k, lda, ldb, ldc,               &
     &      MERGE(0, flags, .NOT.PRESENT(flags)))
          CALL C_F_PROCPOINTER(fn, libxsmm_dmm_function_ldx%dmm0)
        END FUNCTION

        TYPE(LIBXSMM_SMM_FUNCTION)                                      &
     &  FUNCTION libxsmm_smm_function_prf(m, n, k, flags, prefetch)
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN) :: m, n, k
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN) :: flags, prefetch
          TYPE(C_FUNPTR) :: fn
          fn = libxsmm_sdispatch1(m, n, k, 0, 0, 0, flags, prefetch)
          CALL C_F_PROCPOINTER(fn, libxsmm_smm_function_prf%smm1)
        END FUNCTION
        TYPE(LIBXSMM_DMM_FUNCTION)                                      &
     &  FUNCTION libxsmm_dmm_function_prf(m, n, k, flags, prefetch)
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN) :: m, n, k
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN) :: flags, prefetch
          TYPE(C_FUNPTR) :: fn
          fn = libxsmm_ddispatch1(m, n, k, 0, 0, 0, flags, prefetch)
          CALL C_F_PROCPOINTER(fn, libxsmm_dmm_function_prf%dmm1)
        END FUNCTION

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_smm_function_ldf
        TYPE(LIBXSMM_SMM_FUNCTION)                                      &
     &  FUNCTION libxsmm_smm_function_ldf(                              &
     &  m, n, k, lda, ldb, ldc, flags, prefetch)
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN) :: m, n, k
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN) :: lda, ldb, ldc
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN) :: flags, prefetch
          TYPE(C_FUNPTR) :: fn
          fn = libxsmm_sdispatch1(m, n, k, lda, ldb, ldc,               &
     &      flags, prefetch)
          CALL C_F_PROCPOINTER(fn, libxsmm_smm_function_ldf%smm1)
        END FUNCTION
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_dmm_function_ldf
        TYPE(LIBXSMM_DMM_FUNCTION)                                      &
     &  FUNCTION libxsmm_dmm_function_ldf(                              &
     &  m, n, k, lda, ldb, ldc, flags, prefetch)
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN) :: m, n, k
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN) :: lda, ldb, ldc
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN) :: flags, prefetch
          TYPE(C_FUNPTR) :: fn
          fn = libxsmm_ddispatch1(m, n, k, lda, ldb, ldc,               &
     &      flags, prefetch)
          CALL C_F_PROCPOINTER(fn, libxsmm_dmm_function_ldf%dmm1)
        END FUNCTION

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_smm_function_abf
        TYPE(LIBXSMM_SMM_FUNCTION)                                      &
     &  FUNCTION libxsmm_smm_function_abf(                              &
     &  m, n, k, alpha, beta, flags, prefetch)
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN) :: m, n, k
          REAL(LIBXSMM_FLS_KIND), INTENT(IN) :: alpha, beta
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN), OPTIONAL :: flags
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN), OPTIONAL :: prefetch
          TYPE(C_FUNPTR) :: fn
          fn = libxsmm_sdispatch2(m, n, k, 0, 0, 0,                     &
     &      MERGE(0, flags, .NOT.PRESENT(flags)),                       &
     &      MERGE(0, prefetch, .NOT.PRESENT(prefetch)),                 &
     &      alpha, beta)
          CALL C_F_PROCPOINTER(fn, libxsmm_smm_function_abf%smm2)
        END FUNCTION
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_dmm_function_abf
        TYPE(LIBXSMM_DMM_FUNCTION)                                      &
     &  FUNCTION libxsmm_dmm_function_abf(                              &
     &  m, n, k, alpha, beta, flags, prefetch)
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN) :: m, n, k
          REAL(LIBXSMM_FLD_KIND), INTENT(IN) :: alpha, beta
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN), OPTIONAL :: flags
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN), OPTIONAL :: prefetch
          TYPE(C_FUNPTR) :: fn
          fn = libxsmm_ddispatch2(m, n, k, 0, 0, 0,                     &
     &      MERGE(0, flags, .NOT.PRESENT(flags)),                       &
     &      MERGE(0, prefetch, .NOT.PRESENT(prefetch)),                 &
     &      alpha, beta)
          CALL C_F_PROCPOINTER(fn, libxsmm_dmm_function_abf%dmm2)
        END FUNCTION

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_smm_function_all
        TYPE(LIBXSMM_SMM_FUNCTION)                                      &
     &  FUNCTION libxsmm_smm_function_all(                              &
     &  m, n, k, lda, ldb, ldc, alpha, beta, flags, prefetch)
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN) :: m, n, k
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN) :: lda, ldb, ldc
          REAL(LIBXSMM_FLS_KIND), INTENT(IN) :: alpha, beta
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN), OPTIONAL :: flags
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN), OPTIONAL :: prefetch
          TYPE(C_FUNPTR) :: fn
          fn = libxsmm_sdispatch2(m, n, k, lda, ldb, ldc,               &
     &      MERGE(0, flags, .NOT.PRESENT(flags)),                       &
     &      MERGE(0, prefetch, .NOT.PRESENT(prefetch)),                 &
     &      alpha, beta)
          CALL C_F_PROCPOINTER(fn, libxsmm_smm_function_all%smm2)
        END FUNCTION
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_dmm_function_all
        TYPE(LIBXSMM_DMM_FUNCTION)                                      &
     &  FUNCTION libxsmm_dmm_function_all(                              &
     &  m, n, k, lda, ldb, ldc, alpha, beta, flags, prefetch)
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN) :: m, n, k
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN) :: lda, ldb, ldc
          REAL(LIBXSMM_FLD_KIND), INTENT(IN) :: alpha, beta
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN), OPTIONAL :: flags
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN), OPTIONAL :: prefetch
          TYPE(C_FUNPTR) :: fn
          fn = libxsmm_ddispatch2(m, n, k, lda, ldb, ldc,               &
     &      MERGE(0, flags, .NOT.PRESENT(flags)),                       &
     &      MERGE(0, prefetch, .NOT.PRESENT(prefetch)),                 &
     &      alpha, beta)
          CALL C_F_PROCPOINTER(fn, libxsmm_dmm_function_all%dmm2)
        END FUNCTION
        
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_savailable
        LOGICAL PURE FUNCTION libxsmm_savailable(function)
          TYPE(LIBXSMM_SMM_FUNCTION), INTENT(IN) :: function
          libxsmm_savailable =                                          &
     &      ASSOCIATED(function%smm0).OR.                               &
     &      ASSOCIATED(function%smm1).OR.                               &
     &      ASSOCIATED(function%smm2)
        END FUNCTION
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_davailable
        LOGICAL PURE FUNCTION libxsmm_davailable(function)
          TYPE(LIBXSMM_DMM_FUNCTION), INTENT(IN) :: function
          libxsmm_davailable =                                          &
     &      ASSOCIATED(function%dmm0).OR.                               &
     &      ASSOCIATED(function%dmm1).OR.                               &
     &      ASSOCIATED(function%dmm2)
        END FUNCTION

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
        SUBROUTINE libxsmm_sblasmm(flags, m, n, k, a, b, c, alpha, beta)
          INTEGER(LIBXSMM_INT_KIND), PARAMETER :: T = LIBXSMM_FLS_KIND
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN) :: flags, m, n, k
          REAL(T), INTENT(IN) :: a(:,:), b(:,:)
          REAL(T), INTENT(INOUT) :: c(:,:)
          REAL(T), INTENT(IN), OPTIONAL :: alpha, beta
          IF (0.NE.LIBXSMM_COL_MAJOR) THEN
            CALL sgemm(                                                 &
     &        MERGE('N', 'T',                                           &
     &            0.EQ.IAND(LIBXSMM_GEMM_FLAG_TRANS_A, flags)),         &
     &        MERGE('N', 'T',                                           &
     &            0.EQ.IAND(LIBXSMM_GEMM_FLAG_TRANS_B, flags)),         &
     &        m, n, k,                                                  &
     &        MERGE(REAL(LIBXSMM_ALPHA, T), alpha, .NOT.PRESENT(alpha)),&
     &        a, MAX(SIZE(a, 1), m), b, MAX(SIZE(b, 1), k),             &
     &        MERGE(REAL(LIBXSMM_BETA, T), beta, .NOT.PRESENT(beta)),   &
     &        c, MAX(SIZE(c, 1), m))
          ELSE
            CALL sgemm(                                                 &
     &        MERGE('N', 'T',                                           &
     &            0.EQ.IAND(LIBXSMM_GEMM_FLAG_TRANS_A, flags)),         &
     &        MERGE('N', 'T',                                           &
     &            0.EQ.IAND(LIBXSMM_GEMM_FLAG_TRANS_B, flags)),         &
     &        n, m, k,                                                  &
     &        MERGE(REAL(LIBXSMM_ALPHA, T), alpha, .NOT.PRESENT(alpha)),&
     &        b, MAX(SIZE(b, 2), n), a, MAX(SIZE(a, 2), k),             &
     &        MERGE(REAL(LIBXSMM_BETA, T), beta, .NOT.PRESENT(beta)),   &
     &        c, MAX(SIZE(c, 1), n))
          ENDIF
        END SUBROUTINE

        ! Non-dispatched matrix multiplication using BLAS (double-precision).
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_dblasmm
        SUBROUTINE libxsmm_dblasmm(flags, m, n, k, a, b, c, alpha, beta)
          INTEGER(LIBXSMM_INT_KIND), PARAMETER :: T = LIBXSMM_FLD_KIND
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN) :: flags, m, n, k
          REAL(T), INTENT(IN) :: a(:,:), b(:,:)
          REAL(T), INTENT(INOUT) :: c(:,:)
          REAL(T), INTENT(IN), OPTIONAL :: alpha, beta
          IF (0.NE.LIBXSMM_COL_MAJOR) THEN
            CALL dgemm(                                                 &
     &        MERGE('N', 'T',                                           &
     &            0.EQ.IAND(LIBXSMM_GEMM_FLAG_TRANS_A, flags)),         &
     &        MERGE('N', 'T',                                           &
     &            0.EQ.IAND(LIBXSMM_GEMM_FLAG_TRANS_B, flags)),         &
     &        m, n, k,                                                  &
     &        MERGE(REAL(LIBXSMM_ALPHA, T), alpha, .NOT.PRESENT(alpha)),&
     &        a, MAX(SIZE(a, 1), m), b, MAX(SIZE(b, 1), k),             &
     &        MERGE(REAL(LIBXSMM_BETA, T), beta, .NOT.PRESENT(beta)),   &
     &        c, MAX(SIZE(c, 1), m))
          ELSE
            CALL dgemm(                                                 &
     &        MERGE('N', 'T',                                           &
     &            0.EQ.IAND(LIBXSMM_GEMM_FLAG_TRANS_A, flags)),         &
     &        MERGE('N', 'T',                                           &
     &            0.EQ.IAND(LIBXSMM_GEMM_FLAG_TRANS_B, flags)),         &
     &        n, m, k,                                                  &
     &        MERGE(REAL(LIBXSMM_ALPHA, T), alpha, .NOT.PRESENT(alpha)),&
     &        b, MAX(SIZE(b, 2), n), a, MAX(SIZE(a, 2), k),             &
     &        MERGE(REAL(LIBXSMM_BETA, T), beta, .NOT.PRESENT(beta)),   &
     &        c, MAX(SIZE(c, 1), n))
          ENDIF
        END SUBROUTINE

        ! Dispatched matrix multiplication (single-precision).
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_smm
        SUBROUTINE libxsmm_smm(
     &  flags, m, n, k, a, b, c, pa, pb, pc, alpha, beta)
          INTEGER(LIBXSMM_INT_KIND), PARAMETER :: T = LIBXSMM_FLS_KIND
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN) :: flags, m, n, k
          REAL(T), INTENT(IN) :: a(:,:), b(:,:)
          REAL(T), INTENT(INOUT) :: c(:,:)
          REAL(T), INTENT(IN), OPTIONAL :: pa(*), pb(*), pc(*)
          REAL(T), INTENT(IN), OPTIONAL :: alpha, beta
          TYPE(LIBXSMM_SMM_FUNCTION) :: xmm
          INTEGER(LIBXSMM_INT_KIND) :: mn
          IF (LIBXSMM_MAX_MNK.GE.(m * n * k)) THEN
            IF (.NOT.(PRESENT(pa).OR.PRESENT(pb).OR.PRESENT(pc))) THEN
              xmm = libxsmm_sdispatch(m, n, k, flags)
              IF (libxsmm_savailable(xmm)) THEN
                !CALL libxsmm_call(xmm, a, b, c)
              ELSE
                CALL libxsmm_sblasmm(                                   &
     &            flags, m, n, k, a, b, c, alpha, beta)
              ENDIF
            ELSE IF (.NOT.(PRESENT(alpha).OR.PRESENT(beta))) THEN
              xmm = libxsmm_sdispatch(m, n, k, flags,                   &
     &          LIBXSMM_PREFETCH)
              IF (libxsmm_savailable(xmm)) THEN
                !CALL libxsmm_call(xmm, a, b, c, pa, pb, pc)
              ELSE
                CALL libxsmm_sblasmm(                                   &
     &            flags, m, n, k, a, b, c, alpha, beta)
              ENDIF
            ELSE
              xmm = libxsmm_sdispatch(m, n, k, alpha, beta, flags,      &
     &          LIBXSMM_PREFETCH)
              IF (libxsmm_savailable(xmm)) THEN
                !CALL libxsmm_call(xmm, a, b, c, pa, pb, pc, alpha, beta)
              ELSE
                CALL libxsmm_sblasmm(                                   &
     &            flags, m, n, k, a, b, c, alpha, beta)
              ENDIF
            ENDIF
          ELSE
            CALL libxsmm_sblasmm(flags, m, n, k, a, b, c, alpha, beta)
          ENDIF
        END SUBROUTINE

        !!!!!!!!!!!!!!
      END MODULE
