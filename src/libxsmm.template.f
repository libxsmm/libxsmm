!*****************************************************************************!
!* Copyright (c) 2014-2016, Intel Corporation                                *!
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

        PRIVATE ::  construct_smmfunction,                              &
     &              construct_dmmfunction,                              &
     &              srealptr, drealptr

        ! Name of the version (stringized set of version numbers).
        CHARACTER(*), PARAMETER :: LIBXSMM_VERSION = "$VERSION"
        ! Name of the branch of which the version is derived from.
        CHARACTER(*), PARAMETER :: LIBXSMM_BRANCH = "$BRANCH"
        ! Major version based on the last reachable tag under RCS.
        INTEGER(C_INT), PARAMETER :: LIBXSMM_VERSION_MAJOR = $MAJOR
        ! Minor version based on the last reachable tag of the RCS.
        INTEGER(C_INT), PARAMETER :: LIBXSMM_VERSION_MINOR = $MINOR
        ! Update number based on the last reachable tag under RCS.
        INTEGER(C_INT), PARAMETER :: LIBXSMM_VERSION_UPDATE = $UPDATE
        ! Patch number counting commits since the last version stamp.
        INTEGER(C_INT), PARAMETER :: LIBXSMM_VERSION_PATCH = $PATCH

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
        INTEGER(C_INT), PARAMETER :: LIBXSMM_SYNC = $SYNC
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

        ! Enumerates the available target architectures and instruction
        ! set extensions as returned by libxsmm_get_target_arch().
        INTEGER(C_INT), PARAMETER ::                                    &
     &    LIBXSMM_TARGET_ARCH_UNKNOWN = 0,                              &
     &    LIBXSMM_TARGET_ARCH_GENERIC = 1,                              &
     &    LIBXSMM_X86_GENERIC      = 1000,                              &
     &    LIBXSMM_X86_IMCI         = 1001,                              &
     &    LIBXSMM_X86_SSE3         = 1002,                              &
     &    LIBXSMM_X86_SSE4_1       = 1003,                              &
     &    LIBXSMM_X86_SSE4_2       = 1004,                              &
     &    LIBXSMM_X86_AVX          = 1005,                              &
     &    LIBXSMM_X86_AVX2         = 1006,                              &
     &    LIBXSMM_X86_AVX512       = 1007

        ! Type of a function specialized for a given parameter set.
        ABSTRACT INTERFACE
          ! Specialized function with fused alpha and beta arguments.
          PURE SUBROUTINE LIBXSMM_MMFUNCTION0(a, b, c) BIND(C)
            IMPORT :: C_PTR
            TYPE(C_PTR), INTENT(IN), VALUE :: a, b, c
          END SUBROUTINE

          ! Specialized function with alpha, beta, and prefetch arguments.
          PURE SUBROUTINE LIBXSMM_MMFUNCTION1(a, b, c,                  &
     &    pa, pb, pc) BIND(C)
            IMPORT :: C_PTR
            TYPE(C_PTR), INTENT(IN), VALUE :: a, b, c
            TYPE(C_PTR), INTENT(IN), VALUE :: pa, pb, pc
          END SUBROUTINE
        END INTERFACE

        ! Generic function type which is representing either one of the
        ! two wrapped backend procedure pointers (single-precision).
        TYPE :: LIBXSMM_SMMFUNCTION
          PRIVATE
            PROCEDURE(LIBXSMM_MMFUNCTION0), NOPASS, POINTER :: fn0
            PROCEDURE(LIBXSMM_MMFUNCTION1), NOPASS, POINTER :: fn1
        END TYPE

        ! Generic function type which is representing either one of the
        ! two wrapped backend procedure pointers (double-precision).
        TYPE :: LIBXSMM_DMMFUNCTION
          PRIVATE
            PROCEDURE(LIBXSMM_MMFUNCTION0), NOPASS, POINTER :: fn0
            PROCEDURE(LIBXSMM_MMFUNCTION1), NOPASS, POINTER :: fn1
        END TYPE

        ! Construct procedure pointer depending on given argument set.
        INTERFACE libxsmm_mmdispatch
          MODULE PROCEDURE libxsmm_smmdispatch, libxsmm_dmmdispatch
        END INTERFACE

        ! Construct procedure pointer depending on given argument set.
        INTERFACE libxsmm_sdispatch
          MODULE PROCEDURE libxsmm_smmdispatch
        END INTERFACE

        ! Construct procedure pointer depending on given argument set.
        INTERFACE libxsmm_ddispatch
          MODULE PROCEDURE libxsmm_dmmdispatch
        END INTERFACE

        ! Construct procedure pointer depending on given argument set.
        INTERFACE libxsmm_dispatch
          MODULE PROCEDURE libxsmm_smmdispatch, libxsmm_dmmdispatch
        END INTERFACE

        ! Check if a function is available (LIBXSMM_?MMFUNCTION).
        INTERFACE libxsmm_mmavailable
          MODULE PROCEDURE libxsmm_smmavailable, libxsmm_dmmavailable
        END INTERFACE

        ! Check if a function is available (LIBXSMM_?MMFUNCTION).
        INTERFACE libxsmm_savailable
          MODULE PROCEDURE libxsmm_smmavailable
        END INTERFACE

        ! Check if a function is available (LIBXSMM_?MMFUNCTION).
        INTERFACE libxsmm_davailable
          MODULE PROCEDURE libxsmm_dmmavailable
        END INTERFACE

        ! Check if a function is available (LIBXSMM_?MMFUNCTION).
        INTERFACE libxsmm_available
          MODULE PROCEDURE libxsmm_smmavailable, libxsmm_dmmavailable
        END INTERFACE

        ! Call a specialized function.
        INTERFACE libxsmm_mmcall
          MODULE PROCEDURE libxsmm_smmcall_abx, libxsmm_smmcall_prx
          MODULE PROCEDURE libxsmm_smmcall_abc, libxsmm_smmcall_prf
          MODULE PROCEDURE libxsmm_dmmcall_abx, libxsmm_dmmcall_prx
          MODULE PROCEDURE libxsmm_dmmcall_abc, libxsmm_dmmcall_prf
        END INTERFACE

        ! Call a specialized function.
        INTERFACE libxsmm_scall
          MODULE PROCEDURE libxsmm_smmcall_abx, libxsmm_smmcall_prx
          MODULE PROCEDURE libxsmm_smmcall_abc, libxsmm_smmcall_prf
        END INTERFACE

        ! Call a specialized function.
        INTERFACE libxsmm_dcall
          MODULE PROCEDURE libxsmm_dmmcall_abx, libxsmm_dmmcall_prx
          MODULE PROCEDURE libxsmm_dmmcall_abc, libxsmm_dmmcall_prf
        END INTERFACE

        ! Call a specialized function.
        INTERFACE libxsmm_call
          MODULE PROCEDURE libxsmm_smmcall_abx, libxsmm_smmcall_prx
          MODULE PROCEDURE libxsmm_smmcall_abc, libxsmm_smmcall_prf
          MODULE PROCEDURE libxsmm_dmmcall_abx, libxsmm_dmmcall_prx
          MODULE PROCEDURE libxsmm_dmmcall_abc, libxsmm_dmmcall_prf
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
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_get_target_arch
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_timer_duration
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_timer_tick
        INTERFACE
          ! Initialize the library; pay for setup cost at a specific point.
          SUBROUTINE libxsmm_init() BIND(C)
          END SUBROUTINE

          ! Uninitialize the library and free internal memory (optional).
          SUBROUTINE libxsmm_finalize() BIND(C)
          END SUBROUTINE

          ! Returns the architecture and instruction set extension
          ! as determined by the CPUID flags. 0 != LIBXSMM_JIT and
          ! LIBXSMM_X86_AVX <= result, then this instruction set
          ! extension is targeted by the JIT code generator.
          INTEGER(C_INT) PURE FUNCTION libxsmm_get_target_arch() BIND(C)
            IMPORT :: C_INT
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
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: srealptr
        FUNCTION srealptr(a) RESULT(p)
          REAL(C_FLOAT), INTENT(IN), TARGET :: a(:,:)
          REAL(C_FLOAT), POINTER :: p
          p => a(LBOUND(a,1),LBOUND(a,2))
        END FUNCTION

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: drealptr
        FUNCTION drealptr(a) RESULT(p)
          REAL(C_DOUBLE), INTENT(IN), TARGET :: a(:,:)
          REAL(C_DOUBLE), POINTER :: p
          p => a(LBOUND(a,1),LBOUND(a,2))
        END FUNCTION

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: construct_smmfunction
        TYPE(LIBXSMM_SMMFUNCTION) FUNCTION construct_smmfunction(       &
     &  m, n, k, lda, ldb, ldc, alpha, beta, flags, prefetch)
          INTEGER(C_INT), INTENT(IN) :: m, n, k
          INTEGER(C_INT), INTENT(IN), OPTIONAL :: lda, ldb, ldc
          REAL(C_FLOAT), INTENT(IN), OPTIONAL :: alpha, beta
          INTEGER(C_INT), INTENT(IN), OPTIONAL :: flags, prefetch
          !DIR$ ATTRIBUTES OFFLOAD:MIC :: fn0
          PROCEDURE(LIBXSMM_MMFUNCTION0), POINTER :: fn0
          !DIR$ ATTRIBUTES OFFLOAD:MIC :: fn1
          PROCEDURE(LIBXSMM_MMFUNCTION1), POINTER :: fn1
          !DIR$ ATTRIBUTES OFFLOAD:MIC :: sdispatch
          INTERFACE
            TYPE(C_FUNPTR) PURE FUNCTION sdispatch(                     &
     &      m, n, k, lda, ldb, ldc, alpha, beta, flags, prefetch)       &
     &      BIND(C, NAME="libxsmm_smmdispatch")
              IMPORT :: C_FUNPTR, C_INT, C_FLOAT
              INTEGER(C_INT), INTENT(IN), VALUE :: m, n, k
              INTEGER(C_INT), INTENT(IN) :: lda, ldb, ldc
              REAL(C_FLOAT), INTENT(IN) :: alpha, beta
              INTEGER(C_INT), INTENT(IN) :: flags, prefetch
            END FUNCTION
          END INTERFACE
          IF (.NOT.PRESENT(prefetch).OR.                                &
     &        LIBXSMM_PREFETCH_NONE.EQ.prefetch)                        &
     &    THEN
            CALL C_F_PROCPOINTER(sdispatch(m, n, k,                     &
     &        lda, ldb, ldc, alpha, beta, flags, prefetch),             &
     &        fn0)
            construct_smmfunction%fn0 => fn0
            construct_smmfunction%fn1 => NULL()
          ELSE
            CALL C_F_PROCPOINTER(sdispatch(m, n, k,                     &
     &        lda, ldb, ldc, alpha, beta, flags, prefetch),             &
     &        fn1)
            construct_smmfunction%fn0 => NULL()
            construct_smmfunction%fn1 => fn1
          END IF
        END FUNCTION

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: construct_dmmfunction
        TYPE(LIBXSMM_DMMFUNCTION) FUNCTION construct_dmmfunction(       &
     &  m, n, k, lda, ldb, ldc, alpha, beta, flags, prefetch)
          INTEGER(C_INT), INTENT(IN) :: m, n, k
          INTEGER(C_INT), INTENT(IN), OPTIONAL :: lda, ldb, ldc
          REAL(C_DOUBLE), INTENT(IN), OPTIONAL :: alpha, beta
          INTEGER(C_INT), INTENT(IN), OPTIONAL :: flags, prefetch
          !DIR$ ATTRIBUTES OFFLOAD:MIC :: fn0
          PROCEDURE(LIBXSMM_MMFUNCTION0), POINTER :: fn0
          !DIR$ ATTRIBUTES OFFLOAD:MIC :: fn1
          PROCEDURE(LIBXSMM_MMFUNCTION1), POINTER :: fn1
          !DIR$ ATTRIBUTES OFFLOAD:MIC :: ddispatch
          INTERFACE
            PURE FUNCTION ddispatch(                                    &
     &      m, n, k, lda, ldb, ldc, alpha, beta, flags, prefetch)       &
     &      RESULT(fn) BIND(C, NAME="libxsmm_dmmdispatch")
              IMPORT :: C_FUNPTR, C_INT, C_DOUBLE
              INTEGER(C_INT), INTENT(IN), VALUE :: m, n, k
              INTEGER(C_INT), INTENT(IN) :: lda, ldb, ldc
              REAL(C_DOUBLE), INTENT(IN) :: alpha, beta
              INTEGER(C_INT), INTENT(IN) :: flags, prefetch
              TYPE(C_FUNPTR) :: fn
            END FUNCTION
          END INTERFACE
          IF (.NOT.PRESENT(prefetch).OR.                                &
     &        LIBXSMM_PREFETCH_NONE.EQ.prefetch)                        &
     &    THEN
            CALL C_F_PROCPOINTER(ddispatch(m, n, k,                     &
     &        lda, ldb, ldc, alpha, beta, flags, prefetch),             &
     &        fn0)
            construct_dmmfunction%fn0 => fn0
            construct_dmmfunction%fn1 => NULL()
          ELSE
            CALL C_F_PROCPOINTER(ddispatch(m, n, k,                     &
     &        lda, ldb, ldc, alpha, beta, flags, prefetch),             &
     &        fn1)
            construct_dmmfunction%fn0 => NULL()
            construct_dmmfunction%fn1 => fn1
          END IF
        END FUNCTION

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_smmdispatch
        SUBROUTINE libxsmm_smmdispatch(fn,                              &
     &  m, n, k, lda, ldb, ldc, alpha, beta, flags, prefetch)
          TYPE(LIBXSMM_SMMFUNCTION), INTENT(OUT) :: fn
          INTEGER(C_INT), INTENT(IN) :: m, n, k
          INTEGER(C_INT), INTENT(IN), OPTIONAL :: lda, ldb, ldc
          REAL(C_FLOAT), INTENT(IN), OPTIONAL :: alpha, beta
          INTEGER(C_INT), INTENT(IN), OPTIONAL :: flags, prefetch
          fn = construct_smmfunction(                                   &
     &      m, n, k, lda, ldb, ldc, alpha, beta, flags, prefetch)
        END SUBROUTINE

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_dmmdispatch
        SUBROUTINE libxsmm_dmmdispatch(fn,                              &
     &  m, n, k, lda, ldb, ldc, alpha, beta, flags, prefetch)
          TYPE(LIBXSMM_DMMFUNCTION), INTENT(OUT) :: fn
          INTEGER(C_INT), INTENT(IN) :: m, n, k
          INTEGER(C_INT), INTENT(IN), OPTIONAL :: lda, ldb, ldc
          REAL(C_DOUBLE), INTENT(IN), OPTIONAL :: alpha, beta
          INTEGER(C_INT), INTENT(IN), OPTIONAL :: flags, prefetch
          fn = construct_dmmfunction(                                   &
     &      m, n, k, lda, ldb, ldc, alpha, beta, flags, prefetch)
        END SUBROUTINE

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_smmavailable
        LOGICAL PURE FUNCTION libxsmm_smmavailable(fn)
          TYPE(LIBXSMM_SMMFUNCTION), INTENT(IN) :: fn
          libxsmm_smmavailable =                                        &
     &      ASSOCIATED(fn%fn0).OR.ASSOCIATED(fn%fn1)
        END FUNCTION

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_dmmavailable
        LOGICAL PURE FUNCTION libxsmm_dmmavailable(fn)
          TYPE(LIBXSMM_DMMFUNCTION), INTENT(IN) :: fn
          libxsmm_dmmavailable =                                        &
     &      ASSOCIATED(fn%fn0).OR.ASSOCIATED(fn%fn1)
        END FUNCTION

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_smmcall_abx
        PURE SUBROUTINE libxsmm_smmcall_abx(fn, a, b, c)
          TYPE(LIBXSMM_SMMFUNCTION), INTENT(IN) :: fn
          TYPE(C_PTR), INTENT(IN) :: a, b, c
          IF (0.NE.LIBXSMM_COL_MAJOR) THEN
            CALL fn%fn0(a, b, c)
          ELSE
            CALL fn%fn0(b, a, c)
          END IF
        END SUBROUTINE

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_dmmcall_abx
        PURE SUBROUTINE libxsmm_dmmcall_abx(fn, a, b, c)
          TYPE(LIBXSMM_DMMFUNCTION), INTENT(IN) :: fn
          TYPE(C_PTR), INTENT(IN) :: a, b, c
          IF (0.NE.LIBXSMM_COL_MAJOR) THEN
            CALL fn%fn0(a, b, c)
          ELSE
            CALL fn%fn0(b, a, c)
          END IF
        END SUBROUTINE

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_smmcall_prx
        PURE SUBROUTINE libxsmm_smmcall_prx(fn, a, b, c, pa, pb, pc)
          TYPE(LIBXSMM_SMMFUNCTION), INTENT(IN) :: fn
          TYPE(C_PTR), INTENT(IN) :: a, b, c, pa, pb, pc
          IF (0.NE.LIBXSMM_COL_MAJOR) THEN
            CALL fn%fn1(a, b, c, pa, pb, pc)
          ELSE
            CALL fn%fn1(b, a, c, pb, pa, pc)
          END IF
        END SUBROUTINE

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_dmmcall_prx
        PURE SUBROUTINE libxsmm_dmmcall_prx(fn, a, b, c, pa, pb, pc)
          TYPE(LIBXSMM_DMMFUNCTION), INTENT(IN) :: fn
          TYPE(C_PTR), INTENT(IN) :: a, b, c, pa, pb, pc
          IF (0.NE.LIBXSMM_COL_MAJOR) THEN
            CALL fn%fn1(a, b, c, pa, pb, pc)
          ELSE
            CALL fn%fn1(b, a, c, pb, pa, pc)
          END IF
        END SUBROUTINE

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_smmcall_abc
        SUBROUTINE libxsmm_smmcall_abc(fn, a, b, c)
          TYPE(LIBXSMM_SMMFUNCTION), INTENT(IN) :: fn
          REAL(C_FLOAT), INTENT(IN), TARGET :: a(:,:), b(:,:)
          REAL(C_FLOAT), INTENT(INOUT), TARGET :: c(:,:)
          CALL libxsmm_smmcall_abx(fn,                                  &
     &      C_LOC(srealptr(a)), C_LOC(srealptr(b)), C_LOC(srealptr(c)))
        END SUBROUTINE

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_dmmcall_abc
        SUBROUTINE libxsmm_dmmcall_abc(fn, a, b, c)
          TYPE(LIBXSMM_DMMFUNCTION), INTENT(IN) :: fn
          REAL(C_DOUBLE), INTENT(IN), TARGET :: a(:,:), b(:,:)
          REAL(C_DOUBLE), INTENT(INOUT), TARGET :: c(:,:)
          CALL libxsmm_dmmcall_abx(fn,                                  &
     &      C_LOC(drealptr(a)), C_LOC(drealptr(b)), C_LOC(drealptr(c)))
        END SUBROUTINE

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_smmcall_prf
        SUBROUTINE libxsmm_smmcall_prf(fn, a, b, c, pa, pb, pc)
          TYPE(LIBXSMM_SMMFUNCTION), INTENT(IN) :: fn
          REAL(C_FLOAT), INTENT(IN), TARGET :: a(:,:), b(:,:)
          REAL(C_FLOAT), INTENT(INOUT), TARGET :: c(:,:)
          REAL(C_FLOAT), INTENT(IN), TARGET :: pa(*), pb(*), pc(*)
          CALL libxsmm_smmcall_prx(fn,                                  &
     &      C_LOC(srealptr(a)), C_LOC(srealptr(b)), C_LOC(srealptr(c)), &
     &      C_LOC(pa), C_LOC(pb), C_LOC(pc))
        END SUBROUTINE

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_dmmcall_prf
        SUBROUTINE libxsmm_dmmcall_prf(fn, a, b, c, pa, pb, pc)
          TYPE(LIBXSMM_DMMFUNCTION), INTENT(IN) :: fn
          REAL(C_DOUBLE), INTENT(IN), TARGET :: a(:,:), b(:,:)
          REAL(C_DOUBLE), INTENT(INOUT), TARGET :: c(:,:)
          REAL(C_DOUBLE), INTENT(IN), TARGET :: pa(*), pb(*), pc(*)
          CALL libxsmm_dmmcall_prx(fn,                                  &
     &      C_LOC(drealptr(a)), C_LOC(drealptr(b)), C_LOC(drealptr(c)), &
     &      C_LOC(pa), C_LOC(pb), C_LOC(pc))
        END SUBROUTINE

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_smmcall
        SUBROUTINE libxsmm_smmcall(fn, a, b, c, pa, pb, pc)
          TYPE(LIBXSMM_SMMFUNCTION), INTENT(IN) :: fn
          REAL(C_FLOAT), INTENT(IN), TARGET :: a(*), b(*)
          REAL(C_FLOAT), INTENT(INOUT), TARGET :: c(*)
          REAL(C_FLOAT), INTENT(IN), TARGET, OPTIONAL :: pa(*)
          REAL(C_FLOAT), INTENT(IN), TARGET, OPTIONAL :: pb(*)
          REAL(C_FLOAT), INTENT(IN), TARGET, OPTIONAL :: pc(*)
          TYPE(C_PTR) :: cpa, cpb, cpc
          IF (PRESENT(pa).OR.PRESENT(pb).OR.PRESENT(pc)) THEN
            IF (PRESENT(pa)) THEN
              cpa = C_LOC(pa)
            ELSE
              cpa = C_NULL_PTR
            END IF
            IF (PRESENT(pb)) THEN
              cpb = C_LOC(pb)
            ELSE
              cpb = C_NULL_PTR
            END IF
            IF (PRESENT(pc)) THEN
              cpc = C_LOC(pc)
            ELSE
              cpc = C_NULL_PTR
            END IF
            CALL libxsmm_smmcall_prx(fn,                                &
     &        C_LOC(a), C_LOC(b), C_LOC(c),                             &
     &        cpa, cpb, cpc)
          ELSE
            CALL libxsmm_smmcall_abx(fn, C_LOC(a), C_LOC(b), C_LOC(c))
          END IF
        END SUBROUTINE

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_dmmcall
        SUBROUTINE libxsmm_dmmcall(fn, a, b, c, pa, pb, pc)
          TYPE(LIBXSMM_DMMFUNCTION), INTENT(IN) :: fn
          REAL(C_DOUBLE), INTENT(IN), TARGET :: a(*), b(*)
          REAL(C_DOUBLE), INTENT(INOUT), TARGET :: c(*)
          REAL(C_DOUBLE), INTENT(IN), TARGET, OPTIONAL :: pa(*)
          REAL(C_DOUBLE), INTENT(IN), TARGET, OPTIONAL :: pb(*)
          REAL(C_DOUBLE), INTENT(IN), TARGET, OPTIONAL :: pc(*)
          TYPE(C_PTR) :: cpa, cpb, cpc
          IF (PRESENT(pa).OR.PRESENT(pb).OR.PRESENT(pc)) THEN
            IF (PRESENT(pa)) THEN
              cpa = C_LOC(pa)
            ELSE
              cpa = C_NULL_PTR
            END IF
            IF (PRESENT(pb)) THEN
              cpb = C_LOC(pb)
            ELSE
              cpb = C_NULL_PTR
            END IF
            IF (PRESENT(pc)) THEN
              cpc = C_LOC(pc)
            ELSE
              cpc = C_NULL_PTR
            END IF
            CALL libxsmm_dmmcall_prx(fn,                                &
     &        C_LOC(a), C_LOC(b), C_LOC(c),                             &
     &        cpa, cpb, cpc)
          ELSE
            CALL libxsmm_dmmcall_abx(fn, C_LOC(a), C_LOC(b), C_LOC(c))
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
