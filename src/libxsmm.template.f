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
     &                      C_F_PROCPOINTER, C_FUNPTR, C_LOC, C_PTR,    &
     &                      C_INT, C_FLOAT, C_DOUBLE, C_LONG_LONG
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
          ! Specialized function with fused alpha and beta arguments.
          PURE SUBROUTINE LIBXSMM_FUNCTION(a, b, c) BIND(C)
            IMPORT :: C_PTR
            TYPE(C_PTR), INTENT(IN), VALUE :: a, b, c
          END SUBROUTINE

          ! Specialized function with alpha, beta, and prefetch arguments.
          PURE SUBROUTINE LIBXSMM_XFUNCTION(a, b, c,                    &
     &    pa, pb, pc) BIND(C)
            IMPORT :: C_PTR
            TYPE(C_PTR), INTENT(IN), VALUE :: a, b, c
            TYPE(C_PTR), INTENT(IN), VALUE :: pa, pb, pc
          END SUBROUTINE
        END INTERFACE

        ! Generic function type constructing a procedure pointer
        ! associated with a backend function.
        TYPE :: LIBXSMM_SMM_FUNCTION
          PROCEDURE(LIBXSMM_FUNCTION), NOPASS, POINTER ::               &
     &      fn0 => NULL()
          PROCEDURE(LIBXSMM_XFUNCTION), NOPASS, POINTER ::              &
     &      fn1 => NULL()
        END TYPE

        ! Generic function type constructing a procedure pointer
        ! associated with a backend function.
        TYPE :: LIBXSMM_DMM_FUNCTION
          PROCEDURE(LIBXSMM_FUNCTION), NOPASS, POINTER ::               &
     &      fn0 => NULL()
          PROCEDURE(LIBXSMM_XFUNCTION), NOPASS, POINTER ::              &
     &      fn1 => NULL()
        END TYPE

        ! Construct procedure pointer depending on given argument set.
        INTERFACE libxsmm_sdispatch
          MODULE PROCEDURE                                              &
     &      libxsmm_sfunction0, libxsmm_sfunction1
        END INTERFACE

        ! Construct procedure pointer depending on given argument set.
        INTERFACE libxsmm_ddispatch
          MODULE PROCEDURE                                              &
     &      libxsmm_dfunction0, libxsmm_dfunction1
        END INTERFACE

        ! Construct procedure pointer depending on given argument set.
        INTERFACE libxsmm_dispatch
          MODULE PROCEDURE                                              &
     &      libxsmm_sdispatch_mnk, libxsmm_ddispatch_mnk,               &
     &      libxsmm_sdispatch_ldx, libxsmm_ddispatch_ldx,               &
     &      libxsmm_sdispatch_abf, libxsmm_ddispatch_abf,               &
     &      libxsmm_sdispatch_all, libxsmm_ddispatch_all
        END INTERFACE

        ! Check if a function (LIBXSMM_?MM_FUNCTION_TYPE) is available.
        INTERFACE libxsmm_available
          MODULE PROCEDURE libxsmm_savailable, libxsmm_davailable
        END INTERFACE

        ! Call a specialized function (single-precision).
        INTERFACE libxsmm_scall
          MODULE PROCEDURE                                              &
     &      libxsmm_scall_abx, libxsmm_scall_abc,                       &
     &      libxsmm_scall_prx, libxsmm_scall_prf
        END INTERFACE

        ! Call a specialized function (double-precision).
        INTERFACE libxsmm_dcall
          MODULE PROCEDURE                                              &
     &      libxsmm_dcall_abx, libxsmm_dcall_abc,                       &
     &      libxsmm_dcall_prx, libxsmm_dcall_prf
        END INTERFACE

        ! Call a specialized function.
        INTERFACE libxsmm_call
          MODULE PROCEDURE                                              &
     &      libxsmm_scall_abx, libxsmm_scall_abc,                       &
     &      libxsmm_scall_prx, libxsmm_scall_prf,                       &
     &      libxsmm_dcall_abx, libxsmm_dcall_abc,                       &
     &      libxsmm_dcall_prx, libxsmm_dcall_prf
        END INTERFACE

        ! Overloaded auto-dispatch routines (single precision).
        INTERFACE libxsmm_smm
          MODULE PROCEDURE libxsmm_smm_abc, libxsmm_smm_prf
        END INTERFACE

        ! Overloaded auto-dispatch routines (double precision).
        INTERFACE libxsmm_dmm
          MODULE PROCEDURE libxsmm_dmm_abc, libxsmm_dmm_prf
        END INTERFACE

        ! Overloaded auto-dispatch routines.
        INTERFACE libxsmm_mm
          MODULE PROCEDURE                                              &
     &      libxsmm_smm_abc, libxsmm_smm_prf,                           &
     &      libxsmm_dmm_abc, libxsmm_dmm_prf
        END INTERFACE

        ! Overloaded BLAS routines (single/double precision).
        INTERFACE libxsmm_blasmm
          MODULE PROCEDURE libxsmm_sblasmm, libxsmm_dblasmm
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
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_sfunction0
        TYPE(LIBXSMM_SMM_FUNCTION) FUNCTION libxsmm_sfunction0(         &
     &  flags, m, n, k, lda, ldb, ldc, alpha, beta)
          INTEGER(LIBXSMM_INT_KIND), PARAMETER :: T = LIBXSMM_FLS_KIND
          REAL(T), PARAMETER :: default_alpha = REAL(LIBXSMM_ALPHA, T)
          REAL(T), PARAMETER :: default_beta = REAL(LIBXSMM_BETA, T)
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN) :: flags, m, n, k
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN), OPTIONAL :: lda
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN), OPTIONAL :: ldb
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN), OPTIONAL :: ldc
          REAL(T), INTENT(IN), OPTIONAL :: alpha, beta
          PROCEDURE(LIBXSMM_FUNCTION), POINTER :: function
          CALL C_F_PROCPOINTER(                                         &
     &      libxsmm_sdispatch0(flags, m, n, k,                          &
     &          MERGE(0, lda, .NOT.PRESENT(lda)),                       &
     &          MERGE(0, ldb, .NOT.PRESENT(ldb)),                       &
     &          MERGE(0, ldc, .NOT.PRESENT(ldc)),                       &
     &          MERGE(default_alpha, alpha, .NOT.PRESENT(alpha)),       &
     &          MERGE(default_beta, beta, .NOT.PRESENT(beta))),         &
     &      function)
          libxsmm_sfunction0%fn0 => function
          libxsmm_sfunction0%fn1 => NULL()
        END FUNCTION

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_dfunction0
        TYPE(LIBXSMM_DMM_FUNCTION) FUNCTION libxsmm_dfunction0(         &
     &  flags, m, n, k, lda, ldb, ldc, alpha, beta)
          INTEGER(LIBXSMM_INT_KIND), PARAMETER :: T = LIBXSMM_FLD_KIND
          REAL(T), PARAMETER :: default_alpha = REAL(LIBXSMM_ALPHA, T)
          REAL(T), PARAMETER :: default_beta = REAL(LIBXSMM_BETA, T)
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN) :: flags, m, n, k
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN), OPTIONAL :: lda
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN), OPTIONAL :: ldb
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN), OPTIONAL :: ldc
          REAL(T), INTENT(IN), OPTIONAL :: alpha, beta
          PROCEDURE(LIBXSMM_FUNCTION), POINTER :: function
          CALL C_F_PROCPOINTER(                                         &
     &      libxsmm_ddispatch0(flags, m, n, k,                          &
     &          MERGE(0, lda, .NOT.PRESENT(lda)),                       &
     &          MERGE(0, ldb, .NOT.PRESENT(ldb)),                       &
     &          MERGE(0, ldc, .NOT.PRESENT(ldc)),                       &
     &          MERGE(default_alpha, alpha, .NOT.PRESENT(alpha)),       &
     &          MERGE(default_beta, beta, .NOT.PRESENT(beta))),         &
     &      function)
          libxsmm_dfunction0%fn0 => function
          libxsmm_dfunction0%fn1 => NULL()
        END FUNCTION

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_sfunction1
        TYPE(LIBXSMM_SMM_FUNCTION) FUNCTION libxsmm_sfunction1(         &
     &  flags, m, n, k, lda, ldb, ldc, alpha, beta, prefetch)
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN) :: flags, m, n, k
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN) :: lda, ldb, ldc
          REAL(LIBXSMM_FLS_KIND), INTENT(IN) :: alpha, beta
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN) :: prefetch
          PROCEDURE(LIBXSMM_XFUNCTION), POINTER :: fn1
          PROCEDURE(LIBXSMM_FUNCTION), POINTER :: fn0
          IF (LIBXSMM_PREFETCH_NONE.NE.prefetch) THEN
            CALL C_F_PROCPOINTER(libxsmm_sdispatch1(                    &
     &        flags, m, n, k, lda, ldb, ldc, alpha, beta, prefetch),    &
     &        fn1)
            libxsmm_sfunction1%fn1 => fn1
            libxsmm_sfunction1%fn0 => NULL()
          ELSE
            CALL C_F_PROCPOINTER(libxsmm_sdispatch0(                    &
     &        flags, m, n, k, lda, ldb, ldc, alpha, beta),              &
     &        fn0)
            libxsmm_sfunction1%fn0 => fn0
            libxsmm_sfunction1%fn1 => NULL()
          END IF
        END FUNCTION

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_dfunction1
        TYPE(LIBXSMM_DMM_FUNCTION) FUNCTION libxsmm_dfunction1(         &
     &  flags, m, n, k, lda, ldb, ldc, alpha, beta, prefetch)
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN) :: flags, m, n, k
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN) :: lda, ldb, ldc
          REAL(LIBXSMM_FLD_KIND), INTENT(IN) :: alpha, beta
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN) :: prefetch
          PROCEDURE(LIBXSMM_XFUNCTION), POINTER :: fn1
          PROCEDURE(LIBXSMM_FUNCTION), POINTER :: fn0
          IF (LIBXSMM_PREFETCH_NONE.NE.prefetch) THEN
            CALL C_F_PROCPOINTER(libxsmm_ddispatch1(                    &
     &        flags, m, n, k, lda, ldb, ldc, alpha, beta, prefetch),    &
     &        fn1)
            libxsmm_dfunction1%fn1 => fn1
            libxsmm_dfunction1%fn0 => NULL()
          ELSE
            CALL C_F_PROCPOINTER(libxsmm_ddispatch0(                    &
     &        flags, m, n, k, lda, ldb, ldc, alpha, beta),              &
     &        fn0)
            libxsmm_dfunction1%fn0 => fn0
            libxsmm_dfunction1%fn1 => NULL()
          END IF
        END FUNCTION

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_sdispatch_mnk
        SUBROUTINE libxsmm_sdispatch_mnk(function,                      &
     &  m, n, k, alpha, beta, flags)
          TYPE(LIBXSMM_SMM_FUNCTION), INTENT(OUT) :: function
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN) :: m, n, k
          REAL(LIBXSMM_FLS_KIND), INTENT(IN), OPTIONAL :: alpha, beta
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN), OPTIONAL :: flags
          function = libxsmm_sfunction0(                                &
     &      MERGE(LIBXSMM_FLAGS, flags, .NOT.PRESENT(flags)),           &
     &      m, n, k, alpha = alpha, beta = beta)
        END SUBROUTINE

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_ddispatch_mnk
        SUBROUTINE libxsmm_ddispatch_mnk(function,                      &
     &  m, n, k, alpha, beta, flags)
          TYPE(LIBXSMM_DMM_FUNCTION), INTENT(OUT) :: function
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN) :: m, n, k
          REAL(LIBXSMM_FLD_KIND), INTENT(IN), OPTIONAL :: alpha, beta
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN), OPTIONAL :: flags
          function = libxsmm_dfunction0(                                &
     &      MERGE(LIBXSMM_FLAGS, flags, .NOT.PRESENT(flags)),           &
     &      m, n, k, alpha = alpha, beta = beta)
        END SUBROUTINE

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_sdispatch_ldx
        SUBROUTINE libxsmm_sdispatch_ldx(function,                      &
     &  m, n, k, lda, ldb, ldc, alpha, beta, flags)
          TYPE(LIBXSMM_SMM_FUNCTION), INTENT(OUT) :: function
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN) :: m, n, k
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN) :: lda, ldb, ldc
          REAL(LIBXSMM_FLS_KIND), INTENT(IN), OPTIONAL :: alpha, beta
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN), OPTIONAL :: flags
          function = libxsmm_sfunction0(                                &
     &      MERGE(LIBXSMM_FLAGS, flags, .NOT.PRESENT(flags)),           &
     &      m, n, k, lda, ldb, ldc, alpha, beta)
        END SUBROUTINE

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_ddispatch_ldx
        SUBROUTINE libxsmm_ddispatch_ldx(function,                      &
     &  m, n, k, lda, ldb, ldc, alpha, beta, flags)
          TYPE(LIBXSMM_DMM_FUNCTION), INTENT(OUT) :: function
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN) :: m, n, k
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN) :: lda, ldb, ldc
          REAL(LIBXSMM_FLD_KIND), INTENT(IN), OPTIONAL :: alpha, beta
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN), OPTIONAL :: flags
          function = libxsmm_dfunction0(                                &
     &      MERGE(LIBXSMM_FLAGS, flags, .NOT.PRESENT(flags)),           &
     &      m, n, k, lda, ldb, ldc, alpha, beta)
        END SUBROUTINE

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_sdispatch_abf
        SUBROUTINE libxsmm_sdispatch_abf(function,                      &
     &  flags, m, n, k, lda, ldb, ldc, ralpha, rbeta)
          TYPE(LIBXSMM_SMM_FUNCTION), INTENT(OUT) :: function
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN) :: flags, m, n, k
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN) :: lda, ldb, ldc
          REAL(LIBXSMM_FLS_KIND), INTENT(IN) :: ralpha, rbeta
          function = libxsmm_sfunction0(                                &
     &      flags, m, n, k, lda, ldb, ldc, ralpha, rbeta)
        END SUBROUTINE

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_ddispatch_abf
        SUBROUTINE libxsmm_ddispatch_abf(function,                      &
     &  flags, m, n, k, lda, ldb, ldc, ralpha, rbeta)
          TYPE(LIBXSMM_DMM_FUNCTION), INTENT(OUT) :: function
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN) :: flags, m, n, k
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN) :: lda, ldb, ldc
          REAL(LIBXSMM_FLD_KIND), INTENT(IN) :: ralpha, rbeta
          function = libxsmm_dfunction0(                                &
     &      flags, m, n, k, lda, ldb, ldc, ralpha, rbeta)
        END SUBROUTINE

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_sdispatch_all
        SUBROUTINE libxsmm_sdispatch_all(function,                      &
     &  flags, m, n, k, lda, ldb, ldc, alpha, beta, prefetch)
          TYPE(LIBXSMM_SMM_FUNCTION), INTENT(OUT) :: function
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN) :: flags, m, n, k
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN) :: lda, ldb, ldc
          REAL(LIBXSMM_FLS_KIND), INTENT(IN) :: alpha, beta
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN) :: prefetch
          function = libxsmm_sfunction1(                                &
     &      flags, m, n, k, lda, ldb, ldc, alpha, beta, prefetch)
        END SUBROUTINE

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_ddispatch_all
        SUBROUTINE libxsmm_ddispatch_all(function,                      &
     &  flags, m, n, k, lda, ldb, ldc, alpha, beta, prefetch)
          TYPE(LIBXSMM_DMM_FUNCTION), INTENT(OUT) :: function
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN) :: flags, m, n, k
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN) :: lda, ldb, ldc
          REAL(LIBXSMM_FLD_KIND), INTENT(IN) :: alpha, beta
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN) :: prefetch
          function = libxsmm_dfunction1(                                &
     &      flags, m, n, k, lda, ldb, ldc, alpha, beta, prefetch)
        END SUBROUTINE

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
          INTEGER(LIBXSMM_INT_KIND), PARAMETER :: T = LIBXSMM_FLS_KIND
          TYPE(LIBXSMM_SMM_FUNCTION), INTENT(IN) :: fn
          REAL(T), INTENT(IN), TARGET :: a(*), b(*)
          REAL(T), INTENT(INOUT), TARGET :: c(*)
          CALL libxsmm_scall_abx(fn, C_LOC(a), C_LOC(b), C_LOC(c))
        END SUBROUTINE

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_dcall_abc
        SUBROUTINE libxsmm_dcall_abc(fn, a, b, c)
          INTEGER(LIBXSMM_INT_KIND), PARAMETER :: T = LIBXSMM_FLD_KIND
          TYPE(LIBXSMM_DMM_FUNCTION), INTENT(IN) :: fn
          REAL(T), INTENT(IN), TARGET :: a(*), b(*)
          REAL(T), INTENT(INOUT), TARGET :: c(*)
          CALL libxsmm_dcall_abx(fn, C_LOC(a), C_LOC(b), C_LOC(c))
        END SUBROUTINE

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_scall_prf
        SUBROUTINE libxsmm_scall_prf(fn, a, b, c, pa, pb, pc)
          INTEGER(LIBXSMM_INT_KIND), PARAMETER :: T = LIBXSMM_FLS_KIND
          TYPE(LIBXSMM_SMM_FUNCTION), INTENT(IN) :: fn
          REAL(T), INTENT(IN), TARGET :: a(*), b(*)
          REAL(T), INTENT(INOUT), TARGET :: c(*)
          REAL(T), INTENT(IN), TARGET :: pa(*), pb(*), pc(*)
          CALL libxsmm_scall_prx(fn, C_LOC(a), C_LOC(b), C_LOC(c),      &
     &      C_LOC(pa), C_LOC(pb), C_LOC(pc))
        END SUBROUTINE

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_dcall_prf
        SUBROUTINE libxsmm_dcall_prf(fn, a, b, c, pa, pb, pc)
          INTEGER(LIBXSMM_INT_KIND), PARAMETER :: T = LIBXSMM_FLD_KIND
          TYPE(LIBXSMM_DMM_FUNCTION), INTENT(IN) :: fn
          REAL(T), INTENT(IN), TARGET :: a(*), b(*)
          REAL(T), INTENT(INOUT), TARGET :: c(*)
          REAL(T), INTENT(IN), TARGET :: pa(*), pb(*), pc(*)
          CALL libxsmm_dcall_prx(fn, C_LOC(a), C_LOC(b), C_LOC(c),      &
     &      C_LOC(pa), C_LOC(pb), C_LOC(pc))
        END SUBROUTINE

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_ld
        INTEGER(LIBXSMM_INT_KIND) PURE FUNCTION libxsmm_ld(m, n)
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN) :: m, n
          libxsmm_ld = MERGE(m, n, 0.NE.LIBXSMM_COL_MAJOR)
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
        SUBROUTINE libxsmm_sblasmm(m, n, k, a, b, c, flags, alpha, beta)
          INTEGER(LIBXSMM_INT_KIND), PARAMETER :: T = LIBXSMM_FLS_KIND
          REAL(T), PARAMETER :: default_alpha = REAL(LIBXSMM_ALPHA, T)
          REAL(T), PARAMETER :: default_beta = REAL(LIBXSMM_BETA, T)
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN) :: m, n, k
          REAL(T), INTENT(IN) :: a(:,:), b(:,:)
          REAL(T), INTENT(INOUT) :: c(:,:)
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN), OPTIONAL :: flags
          REAL(T), INTENT(IN), OPTIONAL :: alpha, beta
          INTEGER(LIBXSMM_INT_KIND) :: iflags
          INTEGER(LIBXSMM_INT_KIND), PARAMETER :: S = MERGE(8, 4,       &
     &      14.LE.PRECISION(alpha))
          !DIR$ ATTRIBUTES OFFLOAD:MIC :: sgemm
          INTERFACE
            SUBROUTINE sgemm(transa, transb, m, n, k,                   &
     &      alpha, a, lda, b, ldb, beta, c, ldc)
              IMPORT LIBXSMM_INT_KIND, T
              CHARACTER(1), INTENT(IN) :: transa, transb
              INTEGER(LIBXSMM_INT_KIND), INTENT(IN) :: m, n, k
              INTEGER(LIBXSMM_INT_KIND), INTENT(IN) :: lda, ldb, ldc
              REAL(T), INTENT(IN) :: alpha, beta
              REAL(T), INTENT(IN) :: a(lda,*), b(ldb,*)
              REAL(T), INTENT(INOUT) :: c(ldc,*)
            END SUBROUTINE
          END INTERFACE
          iflags = MERGE(LIBXSMM_FLAGS, flags, .NOT.PRESENT(flags))
          IF (0.NE.LIBXSMM_COL_MAJOR) THEN
            CALL sgemm(                                                 &
     &        MERGE('N', 'T',                                           &
     &            0.EQ.IAND(LIBXSMM_GEMM_FLAG_TRANS_A, iflags)),        &
     &        MERGE('N', 'T',                                           &
     &            0.EQ.IAND(LIBXSMM_GEMM_FLAG_TRANS_B, iflags)),        &
     &        m, n, k,                                                  &
     &        MERGE(default_alpha, alpha, .NOT.PRESENT(alpha)),         &
     &        a, MERGE(m, libxsmm_align_value(m, S, LIBXSMM_ALIGNMENT), &
     &            0.EQ.IAND(LIBXSMM_GEMM_FLAG_ALIGN_A, iflags)),        &
     &        b, k,                                                     &
     &        MERGE(default_beta, beta, .NOT.PRESENT(beta)),            &
     &        c, MERGE(m, libxsmm_align_value(m, S, LIBXSMM_ALIGNMENT), &
     &            0.EQ.IAND(LIBXSMM_GEMM_FLAG_ALIGN_C, iflags)))
          ELSE
            CALL sgemm(                                                 &
     &        MERGE('N', 'T',                                           &
     &            0.EQ.IAND(LIBXSMM_GEMM_FLAG_TRANS_A, iflags)),        &
     &        MERGE('N', 'T',                                           &
     &            0.EQ.IAND(LIBXSMM_GEMM_FLAG_TRANS_B, iflags)),        &
     &        n, m, k,                                                  &
     &        MERGE(default_alpha, alpha, .NOT.PRESENT(alpha)),         &
     &        b, MERGE(n, libxsmm_align_value(n, S, LIBXSMM_ALIGNMENT), &
     &            0.EQ.IAND(LIBXSMM_GEMM_FLAG_ALIGN_A, iflags)),        &
     &        a, k,                                                     &
     &        MERGE(default_beta, beta, .NOT.PRESENT(beta)),            &
     &        c, MERGE(n, libxsmm_align_value(n, S, LIBXSMM_ALIGNMENT), &
     &            0.EQ.IAND(LIBXSMM_GEMM_FLAG_ALIGN_C, iflags)))
          END IF
        END SUBROUTINE

        ! Non-dispatched matrix multiplication using BLAS (double-precision).
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_dblasmm
        SUBROUTINE libxsmm_dblasmm(m, n, k, a, b, c, flags, alpha, beta)
          INTEGER(LIBXSMM_INT_KIND), PARAMETER :: T = LIBXSMM_FLD_KIND
          REAL(T), PARAMETER :: default_alpha = REAL(LIBXSMM_ALPHA, T)
          REAL(T), PARAMETER :: default_beta = REAL(LIBXSMM_BETA, T)
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN) :: m, n, k
          REAL(T), INTENT(IN) :: a(:,:), b(:,:)
          REAL(T), INTENT(INOUT) :: c(:,:)
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN), OPTIONAL :: flags
          REAL(T), INTENT(IN), OPTIONAL :: alpha, beta
          INTEGER(LIBXSMM_INT_KIND) :: iflags
          INTEGER(LIBXSMM_INT_KIND), PARAMETER :: S = MERGE(8, 4,       &
     &      14.LE.PRECISION(alpha))
          !DIR$ ATTRIBUTES OFFLOAD:MIC :: dgemm
          INTERFACE
            SUBROUTINE dgemm(transa, transb, m, n, k,                   &
     &      alpha, a, lda, b, ldb, beta, c, ldc)
              IMPORT LIBXSMM_INT_KIND, T
              CHARACTER(1), INTENT(IN) :: transa, transb
              INTEGER(LIBXSMM_INT_KIND), INTENT(IN) :: m, n, k
              INTEGER(LIBXSMM_INT_KIND), INTENT(IN) :: lda, ldb, ldc
              REAL(T), INTENT(IN) :: alpha, beta
              REAL(T), INTENT(IN) :: a(lda,*), b(ldb,*)
              REAL(T), INTENT(INOUT) :: c(ldc,*)
            END SUBROUTINE
          END INTERFACE
          iflags = MERGE(LIBXSMM_FLAGS, flags, .NOT.PRESENT(flags))
          IF (0.NE.LIBXSMM_COL_MAJOR) THEN
            CALL dgemm(                                                 &
     &        MERGE('N', 'T',                                           &
     &            0.EQ.IAND(LIBXSMM_GEMM_FLAG_TRANS_A, iflags)),        &
     &        MERGE('N', 'T',                                           &
     &            0.EQ.IAND(LIBXSMM_GEMM_FLAG_TRANS_B, iflags)),        &
     &        m, n, k,                                                  &
     &        MERGE(default_alpha, alpha, .NOT.PRESENT(alpha)),         &
     &        a, MERGE(m, libxsmm_align_value(m, S, LIBXSMM_ALIGNMENT), &
     &            0.EQ.IAND(LIBXSMM_GEMM_FLAG_ALIGN_A, iflags)),        &
     &        b, k,                                                     &
     &        MERGE(default_beta, beta, .NOT.PRESENT(beta)),            &
     &        c, MERGE(m, libxsmm_align_value(m, S, LIBXSMM_ALIGNMENT), &
     &            0.EQ.IAND(LIBXSMM_GEMM_FLAG_ALIGN_C, iflags)))
          ELSE
            CALL dgemm(                                                 &
     &        MERGE('N', 'T',                                           &
     &            0.EQ.IAND(LIBXSMM_GEMM_FLAG_TRANS_A, iflags)),        &
     &        MERGE('N', 'T',                                           &
     &            0.EQ.IAND(LIBXSMM_GEMM_FLAG_TRANS_B, iflags)),        &
     &        n, m, k,                                                  &
     &        MERGE(default_alpha, alpha, .NOT.PRESENT(alpha)),         &
     &        b, MERGE(n, libxsmm_align_value(n, S, LIBXSMM_ALIGNMENT), &
     &            0.EQ.IAND(LIBXSMM_GEMM_FLAG_ALIGN_A, iflags)),        &
     &        a, k,                                                     &
     &        MERGE(default_beta, beta, .NOT.PRESENT(beta)),            &
     &        c, MERGE(n, libxsmm_align_value(n, S, LIBXSMM_ALIGNMENT), &
     &            0.EQ.IAND(LIBXSMM_GEMM_FLAG_ALIGN_C, iflags)))
          END IF
        END SUBROUTINE

        ! Dispatched matrix multiplication (single-precision).
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_smm_abc
        SUBROUTINE libxsmm_smm_abc(                                     &
     &  m, n, k, a, b, c, flags, alpha, beta)
          INTEGER(LIBXSMM_INT_KIND), PARAMETER :: T = LIBXSMM_FLS_KIND
          REAL(T), PARAMETER :: default_alpha = REAL(LIBXSMM_ALPHA, T)
          REAL(T), PARAMETER :: default_beta = REAL(LIBXSMM_BETA, T)
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN) :: m, n, k
          REAL(T), INTENT(IN) :: a(:,:), b(:,:)
          REAL(T), INTENT(INOUT) :: c(:,:)
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN), OPTIONAL :: flags
          REAL(T), INTENT(IN), OPTIONAL :: alpha, beta
          TYPE(LIBXSMM_SMM_FUNCTION) :: function
          INTEGER(LIBXSMM_INT_KIND) :: iflags
          REAL(T) :: ralpha, rbeta
          iflags = MERGE(LIBXSMM_FLAGS, flags, .NOT.PRESENT(flags))
          ralpha = MERGE(default_alpha, alpha, .NOT.PRESENT(alpha))
          rbeta = MERGE(default_beta, beta, .NOT.PRESENT(beta))
          IF (LIBXSMM_MAX_MNK.GE.(m * n * k)) THEN
            function = libxsmm_sfunction0(                              &
     &        iflags, m, n, k, 0, 0, 0, ralpha, rbeta)
            IF (ASSOCIATED(function%fn0)) THEN
              CALL libxsmm_scall_abc(function, a, b, c)
            ELSE
              CALL libxsmm_sblasmm(m, n, k, a, b, c,                    &
     &          iflags, ralpha, rbeta)
            END IF
          ELSE
            CALL libxsmm_sblasmm(m, n, k, a, b, c,                      &
     &        iflags, ralpha, rbeta)
          END IF
        END SUBROUTINE

        ! Dispatched matrix multiplication (double-precision).
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_dmm_abc
        SUBROUTINE libxsmm_dmm_abc(                                     &
     &  m, n, k, a, b, c, flags, alpha, beta)
          INTEGER(LIBXSMM_INT_KIND), PARAMETER :: T = LIBXSMM_FLD_KIND
          REAL(T), PARAMETER :: default_alpha = REAL(LIBXSMM_ALPHA, T)
          REAL(T), PARAMETER :: default_beta = REAL(LIBXSMM_BETA, T)
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN) :: m, n, k
          REAL(T), INTENT(IN) :: a(:,:), b(:,:)
          REAL(T), INTENT(INOUT) :: c(:,:)
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN), OPTIONAL :: flags
          REAL(T), INTENT(IN), OPTIONAL :: alpha, beta
          TYPE(LIBXSMM_DMM_FUNCTION) :: function
          INTEGER(LIBXSMM_INT_KIND) :: iflags
          REAL(T) :: ralpha, rbeta
          iflags = MERGE(LIBXSMM_FLAGS, flags, .NOT.PRESENT(flags))
          ralpha = MERGE(default_alpha, alpha, .NOT.PRESENT(alpha))
          rbeta = MERGE(default_beta, beta, .NOT.PRESENT(beta))
          IF (LIBXSMM_MAX_MNK.GE.(m * n * k)) THEN
            function = libxsmm_dfunction0(                              &
     &        iflags, m, n, k, 0, 0, 0, ralpha, rbeta)
            IF (ASSOCIATED(function%fn0)) THEN
              CALL libxsmm_dcall_abc(function, a, b, c)
            ELSE
              CALL libxsmm_dblasmm(m, n, k, a, b, c,                    &
     &          iflags, ralpha, rbeta)
            END IF
          ELSE
            CALL libxsmm_dblasmm(m, n, k, a, b, c,                      &
     &        iflags, ralpha, rbeta)
          END IF
        END SUBROUTINE

        ! Dispatched matrix multiplication with prefetches (single-precision).
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_smm_prf
        SUBROUTINE libxsmm_smm_prf(                                     &
     &  m, n, k, a, b, c, pa, pb, pc, flags, alpha, beta)
          INTEGER(LIBXSMM_INT_KIND), PARAMETER :: T = LIBXSMM_FLS_KIND
          REAL(T), PARAMETER :: default_alpha = REAL(LIBXSMM_ALPHA, T)
          REAL(T), PARAMETER :: default_beta = REAL(LIBXSMM_BETA, T)
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN) :: m, n, k
          REAL(T), INTENT(IN) :: a(:,:), b(:,:)
          REAL(T), INTENT(INOUT) :: c(:,:)
          REAL(T), INTENT(IN) :: pa(*), pb(*), pc(*)
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN), OPTIONAL :: flags
          REAL(T), INTENT(IN), OPTIONAL :: alpha, beta
          TYPE(LIBXSMM_SMM_FUNCTION) :: function
          INTEGER(LIBXSMM_INT_KIND) :: iflags
          REAL(T) :: ralpha, rbeta
          iflags = MERGE(LIBXSMM_FLAGS, flags, .NOT.PRESENT(flags))
          ralpha = MERGE(default_alpha, alpha, .NOT.PRESENT(alpha))
          rbeta = MERGE(default_beta, beta, .NOT.PRESENT(beta))
          IF (LIBXSMM_MAX_MNK.GE.(m * n * k)) THEN
            function = libxsmm_sfunction1(                              &
     &        iflags, m, n, k, 0, 0, 0, ralpha, rbeta,                  &
     &        MERGE(LIBXSMM_PREFETCH, LIBXSMM_PREFETCH_SIGNATURE,       &
     &            LIBXSMM_PREFETCH_NONE.NE.LIBXSMM_PREFETCH))
            IF (ASSOCIATED(function%fn1)) THEN
              CALL libxsmm_scall_prf(function, a, b, c, pa, pb, pc)
            ELSE
              CALL libxsmm_sblasmm(m, n, k, a, b, c,                    &
     &          iflags, ralpha, rbeta)
            END IF
          ELSE
            CALL libxsmm_sblasmm(m, n, k, a, b, c,                      &
     &        iflags, ralpha, rbeta)
          END IF
        END SUBROUTINE

        ! Dispatched matrix multiplication with prefetches (double-precision).
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_dmm_prf
        SUBROUTINE libxsmm_dmm_prf(                                     &
     &  m, n, k, a, b, c, pa, pb, pc, flags, alpha, beta)
          INTEGER(LIBXSMM_INT_KIND), PARAMETER :: T = LIBXSMM_FLD_KIND
          REAL(T), PARAMETER :: default_alpha = REAL(LIBXSMM_ALPHA, T)
          REAL(T), PARAMETER :: default_beta = REAL(LIBXSMM_BETA, T)
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN) :: m, n, k
          REAL(T), INTENT(IN) :: a(:,:), b(:,:)
          REAL(T), INTENT(INOUT) :: c(:,:)
          REAL(T), INTENT(IN) :: pa(*), pb(*), pc(*)
          INTEGER(LIBXSMM_INT_KIND), INTENT(IN), OPTIONAL :: flags
          REAL(T), INTENT(IN), OPTIONAL :: alpha, beta
          TYPE(LIBXSMM_DMM_FUNCTION) :: function
          INTEGER(LIBXSMM_INT_KIND) :: iflags
          REAL(T) :: ralpha, rbeta
          iflags = MERGE(LIBXSMM_FLAGS, flags, .NOT.PRESENT(flags))
          ralpha = MERGE(default_alpha, alpha, .NOT.PRESENT(alpha))
          rbeta = MERGE(default_beta, beta, .NOT.PRESENT(beta))
          IF (LIBXSMM_MAX_MNK.GE.(m * n * k)) THEN
            function = libxsmm_dfunction1(                              &
     &        iflags, m, n, k, 0, 0, 0, ralpha, rbeta,                  &
     &        MERGE(LIBXSMM_PREFETCH, LIBXSMM_PREFETCH_SIGNATURE,       &
     &            LIBXSMM_PREFETCH_NONE.NE.LIBXSMM_PREFETCH))
            IF (ASSOCIATED(function%fn1)) THEN
              CALL libxsmm_dcall_prf(function, a, b, c, pa, pb, pc)
            ELSE
              CALL libxsmm_dblasmm(m, n, k, a, b, c,                    &
     &          iflags, ralpha, rbeta)
            END IF
          ELSE
            CALL libxsmm_dblasmm(m, n, k, a, b, c,                      &
     &        iflags, ralpha, rbeta)
          END IF
        END SUBROUTINE
      END MODULE
