!*****************************************************************************!
!* Copyright (c) 2014-2017, Intel Corporation                                *!
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
!* Hans Pabst (Intel Corp.)                                                  *!
!*****************************************************************************!

      MODULE LIBXSMM
        USE, INTRINSIC :: ISO_C_BINDING, ONLY:                          &
     &    C_FLOAT, C_DOUBLE, C_CHAR, C_INT, C_LONG_LONG,                &
     &    C_INTPTR_T, C_F_POINTER, C_F_PROCPOINTER, C_LOC,              &
     &    C_PTR, C_NULL_PTR, C_FUNPTR
        IMPLICIT NONE

        PRIVATE :: srealptr, drealptr

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
        INTEGER(C_INT), PARAMETER :: LIBXSMM_PREFETCH = $PREFETCH
        INTEGER(C_INT), PARAMETER :: LIBXSMM_MAX_MNK = $MAX_MNK
        INTEGER(C_INT), PARAMETER :: LIBXSMM_FLAGS = $FLAGS
        INTEGER(C_INT), PARAMETER :: LIBXSMM_ILP64 = $ILP64

        ! Parameters supplied for backward compatibility (deprecated).
        INTEGER(C_INT), PARAMETER :: LIBXSMM_COL_MAJOR = 1
        INTEGER(C_INT), PARAMETER :: LIBXSMM_ROW_MAJOR = 0

        ! Integer type impacting the BLAS interface (LP64: 32-bit, and ILP64: 64-bit).
        INTEGER(C_INT), PARAMETER :: LIBXSMM_BLASINT_KIND =             &
     &    $BLASINT_KIND ! MERGE(C_INT, C_LONG_LONG, 0.EQ.LIBXSMM_ILP64)

        ! Parameters representing the GEMM performed by the simplified interface.
        REAL(C_DOUBLE), PARAMETER :: LIBXSMM_ALPHA = REAL($ALPHA, C_DOUBLE)
        REAL(C_DOUBLE), PARAMETER :: LIBXSMM_BETA = REAL($BETA, C_DOUBLE)

        ! Flag enumeration which can be IORed.
        INTEGER(C_INT), PARAMETER ::                                    &
     &    LIBXSMM_GEMM_FLAG_TRANS_A = 1,                                &
     &    LIBXSMM_GEMM_FLAG_TRANS_B = 2,                                &
     &    LIBXSMM_GEMM_FLAG_ALIGN_A = 4,                                &
     &    LIBXSMM_GEMM_FLAG_ALIGN_C = 8

        ! Flag which denotes the value type (for weak-typed interface
        ! functions such as libxsmm_xmmdispatch).
        INTEGER(C_INT), PARAMETER ::                                    &
     &    LIBXSMM_GEMM_PRECISION_F64 = 0,                               &
     &    LIBXSMM_GEMM_PRECISION_F32 = 1

        ! Enumeration of the available prefetch strategies which can be IORed.
        INTEGER(C_INT), PARAMETER ::                                    &
          ! Automatically select strategy (frontend).
     &    LIBXSMM_PREFETCH_AUTO       = -1,                             &
          ! No prefetching and no prefetch function signature.
     &    LIBXSMM_PREFETCH_NONE       = 0,                              &
          ! Only function prefetch signature.
     &    LIBXSMM_PREFETCH_SIGONLY    = 1,                              &
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
     &        LIBXSMM_PREFETCH_BL2_VIA_C, LIBXSMM_PREFETCH_AL2_AHEAD),  &
          ! Prefetch PA/PB/PC in L1 (using accesses to A, B, C)
     &    LIBXSMM_PREFETCH_AL1        = 32,                             &
     &    LIBXSMM_PREFETCH_BL1        = 64,                             &
     &    LIBXSMM_PREFETCH_CL1        = 128,                            &
     &    LIBXSMM_PREFETCH_AL1_BL1 = IOR(                               &
     &        LIBXSMM_PREFETCH_AL1, LIBXSMM_PREFETCH_BL1),              &
     &    LIBXSMM_PREFETCH_BL1_CL1 = IOR(                               &
     &        LIBXSMM_PREFETCH_BL1, LIBXSMM_PREFETCH_CL1),              &
     &    LIBXSMM_PREFETCH_AL1_CL1 = IOR(                               &
     &        LIBXSMM_PREFETCH_AL1, LIBXSMM_PREFETCH_CL1),              &
     &    LIBXSMM_PREFETCH_AL1_BL1_CL1 = IOR(                           &
     &        LIBXSMM_PREFETCH_AL1_BL1, LIBXSMM_PREFETCH_CL1)

        ! Enumerates the available target architectures and instruction
        ! set extensions as returned by libxsmm_get_target_archid().
        INTEGER(C_INT), PARAMETER ::                                    &
     &    LIBXSMM_TARGET_ARCH_UNKNOWN = 0,                              &
     &    LIBXSMM_TARGET_ARCH_GENERIC = 1,                              &
     &    LIBXSMM_X86_IMCI        = 1001,                               &
     &    LIBXSMM_X86_GENERIC     = 1002,                               &
     &    LIBXSMM_X86_SSE3        = 1003,                               &
     &    LIBXSMM_X86_SSE4        = 1004,                               &
     &    LIBXSMM_X86_AVX         = 1005,                               &
     &    LIBXSMM_X86_AVX2        = 1006,                               &
     &    LIBXSMM_X86_AVX512      = 1007,                               &
     &    LIBXSMM_X86_AVX512_MIC  = 1010,                               &
     &    LIBXSMM_X86_AVX512_KNM  = 1011,                               &
     &    LIBXSMM_X86_AVX512_CORE = 1020

        ! Generic function type (single-precision).
        TYPE :: LIBXSMM_SMMFUNCTION
          PRIVATE
            INTEGER(C_INTPTR_T) :: handle
        END TYPE

        ! Generic function type (double-precision).
        TYPE :: LIBXSMM_DMMFUNCTION
          PRIVATE
            INTEGER(C_INTPTR_T) :: handle
        END TYPE

        ! Construct JIT-code depending on given argument set.
        INTERFACE libxsmm_mmdispatch
          MODULE PROCEDURE libxsmm_smmdispatch, libxsmm_dmmdispatch
        END INTERFACE

        ! Construct JIT-code depending on given argument set.
        INTERFACE libxsmm_dispatch
          MODULE PROCEDURE libxsmm_smmdispatch, libxsmm_dmmdispatch
        END INTERFACE

        ! Check if a function is available (LIBXSMM_?MMFUNCTION).
        INTERFACE libxsmm_mmavailable
          MODULE PROCEDURE libxsmm_smmavailable, libxsmm_dmmavailable
        END INTERFACE

        ! Check if a function is available (LIBXSMM_?MMFUNCTION).
        INTERFACE libxsmm_available
          MODULE PROCEDURE libxsmm_smmavailable, libxsmm_dmmavailable
        END INTERFACE

        ! Call a specialized function.
        INTERFACE libxsmm_mmcall
          MODULE PROCEDURE libxsmm_smmcall_abc, libxsmm_smmcall_prf
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

        ! Overloaded MATMUL-style routines (single/double precision).
        INTERFACE libxsmm_matmul
          MODULE PROCEDURE libxsmm_smatmul, libxsmm_dmatmul
        END INTERFACE

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_init, libxsmm_finalize
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_get_gemm_auto_prefetch
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_set_gemm_auto_prefetch
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_get_dispatch_trylock
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_set_dispatch_trylock
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_get_target_archid
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_set_target_archid
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_set_target_arch
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_get_verbosity
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_set_verbosity
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_timer_duration
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_timer_xtick
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_timer_tick
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_xmmdispatch
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_xmmcall_abc
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_xmmcall_prf
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_sgemm_omp
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_dgemm_omp
        INTERFACE
          ! Initialize the library; pay for setup cost at a specific point.
          SUBROUTINE libxsmm_init() BIND(C)
          END SUBROUTINE

          ! De-initialize the library and free internal memory (optional).
          SUBROUTINE libxsmm_finalize() BIND(C)
          END SUBROUTINE

          ! Get the default prefetch strategy.
          PURE FUNCTION libxsmm_get_gemm_auto_prefetch() BIND(C)
            IMPORT :: C_INT
            INTEGER(C_INT) :: libxsmm_get_gemm_auto_prefetch
          END FUNCTION

          ! Set the default prefetch strategy.
          SUBROUTINE libxsmm_set_gemm_auto_prefetch(strategy) BIND(C)
            IMPORT :: C_INT
            INTEGER(C_INT), INTENT(IN), VALUE :: strategy
          END SUBROUTINE

          ! Query the try-lock property of the code registry.
          PURE FUNCTION libxsmm_get_dispatch_trylock() BIND(C)
            IMPORT :: C_INT
            INTEGER(C_INT) :: libxsmm_get_dispatch_trylock
          END FUNCTION

          ! Set the try-lock property of the code registry.
          SUBROUTINE libxsmm_set_dispatch_trylock(trylock) BIND(C)
            IMPORT :: C_INT
            INTEGER(C_INT), INTENT(IN), VALUE :: trylock
          END SUBROUTINE

          ! Returns the architecture and instruction set extension as determined
          ! by the CPUID flags, as set by the libxsmm_get_target_arch* functions,
          ! or as set by the LIBXSMM_TARGET environment variable.
          PURE FUNCTION libxsmm_get_target_archid() BIND(C)
            IMPORT :: C_INT
            INTEGER(C_INT) :: libxsmm_get_target_archid
          END FUNCTION

          ! Set target architecture (id: see PARAMETER enumeration)
          ! for subsequent code generation (JIT).
          SUBROUTINE libxsmm_set_target_archid(id) BIND(C)
            IMPORT :: C_INT
            INTEGER(C_INT), INTENT(IN), VALUE :: id
          END SUBROUTINE

          ! Set target architecture (arch="0|sse|snb|hsw|knl|skx", "0": CPUID)
          ! for subsequent code generation (JIT).
          SUBROUTINE libxsmm_set_target_arch(arch) BIND(C)
            IMPORT :: C_CHAR
            CHARACTER(C_CHAR), INTENT(IN) :: arch(*)
          END SUBROUTINE

          ! Get the level of verbosity.
          PURE FUNCTION libxsmm_get_verbosity() BIND(C)
            IMPORT :: C_INT
            INTEGER(C_INT) :: libxsmm_get_verbosity
          END FUNCTION

          ! Set the level of verbosity (0: off, positive value: verbosity level,
          ! negative value: maximum verbosity, which also dumps JIT-code).
          SUBROUTINE libxsmm_set_verbosity(level) BIND(C)
            IMPORT :: C_INT
            INTEGER(C_INT), INTENT(IN), VALUE :: level
          END SUBROUTINE

          ! Impure function which returns the current clock tick of a
          ! monotonic timer source; uses a platform-specific resolution.
          INTEGER(C_LONG_LONG) FUNCTION libxsmm_timer_tick() BIND(C)
            IMPORT :: C_LONG_LONG
          END FUNCTION

          ! Impure function which returns the current tick of a (monotonic)
          ! platform-specific counter; not necessarily CPU cycles.
          INTEGER(C_LONG_LONG) FUNCTION libxsmm_timer_xtick() BIND(C)
            IMPORT :: C_LONG_LONG
          END FUNCTION

          ! Impure function (timer freq. may vary) which returns the duration
          ! (in seconds) between two values received by libxsmm_timer_tick.
          REAL(C_DOUBLE) FUNCTION libxsmm_timer_duration(               &
     &    tick0, tick1) BIND(C)
            IMPORT :: C_LONG_LONG, C_DOUBLE
            INTEGER(C_LONG_LONG), INTENT(IN), VALUE :: tick0, tick1
          END FUNCTION

          ! Type-generic (unsafe) code dispatch (trylock: impure routine),
          ! which is also suited for FORTRAN 77 when relying on:
          ! INTEGER(4) :: precision, m, n, k, lda, ldb, ldc, flags, prefetch
          ! REAL(4|8)  :: alpha, beta
          ! INTEGER(8) :: fn
          SUBROUTINE libxsmm_xmmdispatch(fn, precision,                 &
     &    m, n, k, lda, ldb, ldc, alpha, beta, flags, prefetch)         &
     &    BIND(C, NAME="libxsmm_xmmdispatch_")
            IMPORT :: C_INTPTR_T, C_PTR, C_INT
            INTEGER(C_INTPTR_T), INTENT(OUT) :: fn
            INTEGER(C_INT), INTENT(IN) :: precision, m, n, k
            INTEGER(C_INT), INTENT(IN) :: lda, ldb, ldc
            TYPE(C_PTR), INTENT(IN), VALUE :: alpha, beta
            INTEGER(C_INT), INTENT(IN) :: flags, prefetch
          END SUBROUTINE

          ! Generic call routine (3-argument form).
          PURE SUBROUTINE libxsmm_xmmcall_abc(fn, a, b, c)              &
     &    BIND(C, NAME="libxsmm_xmmcall_abc_")
            IMPORT :: C_INTPTR_T, C_PTR
            INTEGER(C_INTPTR_T), INTENT(IN) :: fn
            TYPE(C_PTR), INTENT(IN), VALUE :: a, b, c
          END SUBROUTINE

          ! Generic call routine (6-argument form).
          PURE SUBROUTINE libxsmm_xmmcall_prf(fn, a, b, c, pa, pb, pc)  &
     &    BIND(C, NAME="libxsmm_xmmcall_prf_")
            IMPORT :: C_INTPTR_T, C_PTR
            INTEGER(C_INTPTR_T), INTENT(IN) :: fn
            TYPE(C_PTR), INTENT(IN), VALUE :: a, b, c, pa, pb, pc
          END SUBROUTINE

          SUBROUTINE libxsmm_sgemm_omp(transa, transb, m, n, k,         &
     &    alpha, a, lda, b, ldb, beta, c, ldc) BIND(C)
            IMPORT LIBXSMM_BLASINT_KIND, C_CHAR, C_FLOAT
            CHARACTER(C_CHAR), INTENT(IN) :: transa, transb
            INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: m, n, k
            INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: lda, ldb, ldc
            REAL(C_FLOAT), INTENT(IN) :: alpha, beta
            REAL(C_FLOAT), INTENT(IN) :: a(lda,*), b(ldb,*)
            REAL(C_FLOAT), INTENT(INOUT) :: c(ldc,*)
          END SUBROUTINE

          SUBROUTINE libxsmm_dgemm_omp(transa, transb, m, n, k,         &
     &    alpha, a, lda, b, ldb, beta, c, ldc) BIND(C)
            IMPORT LIBXSMM_BLASINT_KIND, C_CHAR, C_DOUBLE
            CHARACTER(C_CHAR), INTENT(IN) :: transa, transb
            INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: m, n, k
            INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: lda, ldb, ldc
            REAL(C_DOUBLE), INTENT(IN) :: alpha, beta
            REAL(C_DOUBLE), INTENT(IN) :: a(lda,*), b(ldb,*)
            REAL(C_DOUBLE), INTENT(INOUT) :: c(ldc,*)
          END SUBROUTINE
        END INTERFACE$MNK_INTERFACE_LIST

      CONTAINS
        ! Returns the name of the target architecture as determined by
        ! the CPUID flags, as set by the libxsmm_get_target_arch* functions,
        ! or as set by the LIBXSMM_TARGET environment variable.
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_get_target_arch
        FUNCTION libxsmm_get_target_arch()
          !CHARACTER(LEN=:), POINTER :: libxsmm_get_target_arch
          CHARACTER, POINTER :: libxsmm_get_target_arch(:)
          INTEGER(C_INT) :: length(1)
          TYPE(C_PTR) :: arch
          !DIR$ ATTRIBUTES OFFLOAD:MIC :: get_target_arch
          INTERFACE
            FUNCTION get_target_arch(length) BIND(C)
              IMPORT :: C_INT, C_PTR
              INTEGER(C_INT), INTENT(OUT) :: length
              TYPE(C_PTR) :: get_target_arch
            END FUNCTION
          END INTERFACE
          arch = get_target_arch(length(1))
          CALL C_F_POINTER(arch, libxsmm_get_target_arch, length)
        END FUNCTION

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: srealptr
        FUNCTION srealptr(a)
          REAL(C_FLOAT), INTENT(IN), TARGET :: a(:,:)
          REAL(C_FLOAT), POINTER :: fptr
          TYPE(C_PTR) :: srealptr
          fptr => a(LBOUND(a,1),LBOUND(a,2))
          srealptr = C_LOC(fptr)
        END FUNCTION

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: drealptr
        FUNCTION drealptr(a)
          REAL(C_DOUBLE), INTENT(IN), TARGET :: a(:,:)
          REAL(C_DOUBLE), POINTER :: fptr
          TYPE(C_PTR) :: drealptr
          fptr => a(LBOUND(a,1),LBOUND(a,2))
          drealptr = C_LOC(fptr)
        END FUNCTION

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_smmdispatch
        SUBROUTINE libxsmm_smmdispatch(fn,                              &
     &  m, n, k, lda, ldb, ldc, alpha, beta, flags, prefetch)
          TYPE(LIBXSMM_SMMFUNCTION), INTENT(OUT) :: fn
          INTEGER(C_INT), INTENT(IN), VALUE :: m, n, k
          INTEGER(C_INT), INTENT(IN), OPTIONAL :: lda, ldb, ldc
          REAL(C_FLOAT), INTENT(IN), OPTIONAL, TARGET :: alpha, beta
          INTEGER(C_INT), INTENT(IN), OPTIONAL :: flags, prefetch
          TYPE(C_PTR) :: calpha, cbeta
          IF (PRESENT(alpha)) THEN
            calpha = C_LOC(alpha)
          ELSE
            calpha = C_NULL_PTR
          END IF
          IF (PRESENT(beta)) THEN
            cbeta = C_LOC(beta)
          ELSE
            cbeta = C_NULL_PTR
          END IF
          CALL libxsmm_xmmdispatch(                                     &
     &      fn%handle, LIBXSMM_GEMM_PRECISION_F32,                      &
     &      m, n, k, lda, ldb, ldc, calpha, cbeta, flags, prefetch)
        END SUBROUTINE

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_dmmdispatch
        SUBROUTINE libxsmm_dmmdispatch(fn,                              &
     &  m, n, k, lda, ldb, ldc, alpha, beta, flags, prefetch)
          TYPE(LIBXSMM_DMMFUNCTION), INTENT(OUT) :: fn
          INTEGER(C_INT), INTENT(IN), VALUE :: m, n, k
          INTEGER(C_INT), INTENT(IN), OPTIONAL :: lda, ldb, ldc
          REAL(C_DOUBLE), INTENT(IN), OPTIONAL, TARGET :: alpha, beta
          INTEGER(C_INT), INTENT(IN), OPTIONAL :: flags, prefetch
          TYPE(C_PTR) :: calpha, cbeta
          IF (PRESENT(alpha)) THEN
            calpha = C_LOC(alpha)
          ELSE
            calpha = C_NULL_PTR
          END IF
          IF (PRESENT(beta)) THEN
            cbeta = C_LOC(beta)
          ELSE
            cbeta = C_NULL_PTR
          END IF
          CALL libxsmm_xmmdispatch(                                     &
     &      fn%handle, LIBXSMM_GEMM_PRECISION_F64,                      &
     &      m, n, k, lda, ldb, ldc, calpha, cbeta, flags, prefetch)
        END SUBROUTINE

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_smmavailable
        LOGICAL PURE FUNCTION libxsmm_smmavailable(fn)
          TYPE(LIBXSMM_SMMFUNCTION), INTENT(IN) :: fn
          libxsmm_smmavailable = 0.NE.fn%handle
        END FUNCTION

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_dmmavailable
        LOGICAL PURE FUNCTION libxsmm_dmmavailable(fn)
          TYPE(LIBXSMM_DMMFUNCTION), INTENT(IN) :: fn
          libxsmm_dmmavailable = 0.NE.fn%handle
        END FUNCTION

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_smmcall
        SUBROUTINE libxsmm_smmcall(fn, a, b, c, pa, pb, pc)
          TYPE(LIBXSMM_SMMFUNCTION), INTENT(IN) :: fn
          REAL(C_FLOAT), INTENT(IN), TARGET :: a(*), b(*)
          REAL(C_FLOAT), INTENT(INOUT), TARGET :: c(*)
          REAL(C_FLOAT), INTENT(IN), OPTIONAL, TARGET :: pa(*)
          REAL(C_FLOAT), INTENT(IN), OPTIONAL, TARGET :: pb(*)
          REAL(C_FLOAT), INTENT(IN), OPTIONAL, TARGET :: pc(*)
          IF (PRESENT(pa).AND.PRESENT(pb).AND.PRESENT(pc)) THEN
            CALL libxsmm_xmmcall_prf(fn%handle,                         &
     &        C_LOC(a), C_LOC(b), C_LOC(c),                             &
     &        C_LOC(pa), C_LOC(pb), C_LOC(pc))
          ELSE
            CALL libxsmm_xmmcall_abc(fn%handle,                         &
     &        C_LOC(a), C_LOC(b), C_LOC(c))
          END IF
        END SUBROUTINE

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_dmmcall
        SUBROUTINE libxsmm_dmmcall(fn, a, b, c, pa, pb, pc)
          TYPE(LIBXSMM_DMMFUNCTION), INTENT(IN) :: fn
          REAL(C_DOUBLE), INTENT(IN), TARGET :: a(*), b(*)
          REAL(C_DOUBLE), INTENT(INOUT), TARGET :: c(*)
          REAL(C_DOUBLE), INTENT(IN), OPTIONAL, TARGET :: pa(*)
          REAL(C_DOUBLE), INTENT(IN), OPTIONAL, TARGET :: pb(*)
          REAL(C_DOUBLE), INTENT(IN), OPTIONAL, TARGET :: pc(*)
          IF (PRESENT(pa).AND.PRESENT(pb).AND.PRESENT(pc)) THEN
            CALL libxsmm_xmmcall_prf(fn%handle,                         &
     &        C_LOC(a), C_LOC(b), C_LOC(c),                             &
     &        C_LOC(pa), C_LOC(pb), C_LOC(pc))
          ELSE
            CALL libxsmm_xmmcall_abc(fn%handle,                         &
     &        C_LOC(a), C_LOC(b), C_LOC(c))
          END IF
        END SUBROUTINE

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_smmcall_abc
        PURE SUBROUTINE libxsmm_smmcall_abc(fn, a, b, c)
          TYPE(LIBXSMM_SMMFUNCTION), INTENT(IN) :: fn
          TYPE(C_PTR), INTENT(IN) :: a, b, c
          CALL libxsmm_xmmcall_abc(fn%handle, a, b, c)
        END SUBROUTINE

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_dmmcall_abc
        PURE SUBROUTINE libxsmm_dmmcall_abc(fn, a, b, c)
          TYPE(LIBXSMM_DMMFUNCTION), INTENT(IN) :: fn
          TYPE(C_PTR), INTENT(IN) :: a, b, c
          CALL libxsmm_xmmcall_abc(fn%handle, a, b, c)
        END SUBROUTINE

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_smmcall_prf
        PURE SUBROUTINE libxsmm_smmcall_prf(fn, a, b, c, pa, pb, pc)
          TYPE(LIBXSMM_SMMFUNCTION), INTENT(IN) :: fn
          TYPE(C_PTR), INTENT(IN) :: a, b, c, pa, pb, pc
          CALL libxsmm_xmmcall_prf(fn%handle, a, b, c, pa, pb, pc)
        END SUBROUTINE

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_dmmcall_prf
        PURE SUBROUTINE libxsmm_dmmcall_prf(fn, a, b, c, pa, pb, pc)
          TYPE(LIBXSMM_DMMFUNCTION), INTENT(IN) :: fn
          TYPE(C_PTR), INTENT(IN) :: a, b, c, pa, pb, pc
          CALL libxsmm_xmmcall_prf(fn%handle, a, b, c, pa, pb, pc)
        END SUBROUTINE

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_sgemm
        SUBROUTINE libxsmm_sgemm(transa, transb, m, n, k,               &
     &  alpha, a, lda, b, ldb, beta, c, ldc)
          CHARACTER, INTENT(IN), OPTIONAL :: transa, transb
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN), VALUE :: m, n, k
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
          CHARACTER, INTENT(IN), OPTIONAL :: transa, transb
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN), VALUE :: m, n, k
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
          CHARACTER, INTENT(IN), OPTIONAL :: transa, transb
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN), VALUE :: m, n, k
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN), OPTIONAL :: lda
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN), OPTIONAL :: ldb
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN), OPTIONAL :: ldc
          REAL(C_FLOAT), INTENT(IN), OPTIONAL :: alpha, beta
          REAL(C_FLOAT), INTENT(IN) :: a(:,:), b(:,:)
          REAL(C_FLOAT), INTENT(INOUT) :: c(:,:)
          !DIR$ ATTRIBUTES OFFLOAD:MIC :: internal_blas_gemm
          INTERFACE
            SUBROUTINE internal_blas_gemm(transa, transb, m, n, k,      &
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
          CALL internal_blas_gemm(transa, transb, m, n, k,              &
     &      alpha, a, lda, b, ldb, beta, c, ldc)
        END SUBROUTINE

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_blas_dgemm
        SUBROUTINE libxsmm_blas_dgemm(transa, transb, m, n, k,          &
     &  alpha, a, lda, b, ldb, beta, c, ldc)
          CHARACTER, INTENT(IN), OPTIONAL :: transa, transb
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN), VALUE :: m, n, k
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN), OPTIONAL :: lda
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN), OPTIONAL :: ldb
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN), OPTIONAL :: ldc
          REAL(C_DOUBLE), INTENT(IN), OPTIONAL :: alpha, beta
          REAL(C_DOUBLE), INTENT(IN) :: a(:,:), b(:,:)
          REAL(C_DOUBLE), INTENT(INOUT) :: c(:,:)
          !DIR$ ATTRIBUTES OFFLOAD:MIC :: internal_blas_gemm
          INTERFACE
            SUBROUTINE internal_blas_gemm(transa, transb, m, n, k,      &
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
          CALL internal_blas_gemm(transa, transb, m, n, k,              &
     &      alpha, a, lda, b, ldb, beta, c, ldc)
        END SUBROUTINE

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_smatmul
        SUBROUTINE libxsmm_smatmul(c, a, b, alpha, beta, transa, transb)
          REAL(C_FLOAT), INTENT(INOUT) :: c(:,:)
          REAL(C_FLOAT), INTENT(IN) :: a(:,:), b(:,:)
          REAL(C_FLOAT), INTENT(IN), OPTIONAL :: alpha, beta
          CHARACTER, INTENT(IN), OPTIONAL :: transa, transb
          CHARACTER :: otransa, otransb
          INTEGER :: s(2)
          IF (.NOT.PRESENT(transa)) THEN
            otransa = 'N'
          ELSE
            otransa = transa
          END IF
          IF (.NOT.PRESENT(transb)) THEN
            otransb = 'N'
          ELSE
            otransb = transb
          END IF
          ! TODO: transpose is currently not supported by LIBXSMM
          IF (('N'.EQ.otransa).OR.('n'.EQ.otransa)) THEN
            IF (('N'.EQ.otransb).OR.('n'.EQ.otransb)) THEN
              CALL libxsmm_sgemm('N', 'N',                              &
     &          SIZE(c, 1, LIBXSMM_BLASINT_KIND),                       &
     &          SIZE(c, 2, LIBXSMM_BLASINT_KIND),                       &
     &          SIZE(a, 2, LIBXSMM_BLASINT_KIND),                       &
     &          alpha, a, SIZE(a, 1, LIBXSMM_BLASINT_KIND),             &
     &                 b, SIZE(b, 1, LIBXSMM_BLASINT_KIND),             &
     &           beta, c, SIZE(c, 1, LIBXSMM_BLASINT_KIND))
            ELSE ! A x B^T
              CALL libxsmm_sgemm('N', 'N',                              &
     &          SIZE(c, 1, LIBXSMM_BLASINT_KIND),                       &
     &          SIZE(c, 2, LIBXSMM_BLASINT_KIND),                       &
     &          SIZE(a, 2, LIBXSMM_BLASINT_KIND),                       &
     &          alpha, a, SIZE(a, 1, LIBXSMM_BLASINT_KIND),             &
     &          TRANSPOSE(b), SIZE(b, 2, LIBXSMM_BLASINT_KIND),         &
     &          beta, c, SIZE(c, 1, LIBXSMM_BLASINT_KIND))
            END IF
          ELSE ! A^T
            IF (('N'.EQ.otransb).OR.('n'.EQ.otransb)) THEN
              CALL libxsmm_sgemm('N', 'N',                              &
     &          SIZE(c, 1, LIBXSMM_BLASINT_KIND),                       &
     &          SIZE(c, 2, LIBXSMM_BLASINT_KIND),                       &
     &          SIZE(a, 1, LIBXSMM_BLASINT_KIND),                       &
     &          alpha, TRANSPOSE(a), SIZE(a, 2, LIBXSMM_BLASINT_KIND),  &
     &                 b, SIZE(b, 1, LIBXSMM_BLASINT_KIND),             &
     &           beta, c, SIZE(c, 1, LIBXSMM_BLASINT_KIND))
            ELSE ! A^T x B^T -> C = (B x A)^T
              CALL libxsmm_sgemm('N', 'N',                              &
     &          SIZE(c, 1, LIBXSMM_BLASINT_KIND),                       &
     &          SIZE(c, 2, LIBXSMM_BLASINT_KIND),                       &
     &          SIZE(a, 1, LIBXSMM_BLASINT_KIND),                       &
     &          alpha, b, SIZE(b, 1, LIBXSMM_BLASINT_KIND),             &
     &                 a, SIZE(a, 1, LIBXSMM_BLASINT_KIND),             &
     &           beta, c, SIZE(c, 1, LIBXSMM_BLASINT_KIND))
              s(1) = SIZE(c, 2); s(2) = SIZE(c, 1)
              c = TRANSPOSE(RESHAPE(c, s))
            END IF
          END IF
        END SUBROUTINE

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_dmatmul
        SUBROUTINE libxsmm_dmatmul(c, a, b, alpha, beta, transa, transb)
          REAL(C_DOUBLE), INTENT(INOUT) :: c(:,:)
          REAL(C_DOUBLE), INTENT(IN) :: a(:,:), b(:,:)
          REAL(C_DOUBLE), INTENT(IN), OPTIONAL :: alpha, beta
          CHARACTER, INTENT(IN), OPTIONAL :: transa, transb
          CHARACTER :: otransa, otransb
          INTEGER(C_INT) :: s(2)
          IF (.NOT.PRESENT(transa)) THEN
            otransa = 'N'
          ELSE
            otransa = transa
          END IF
          IF (.NOT.PRESENT(transb)) THEN
            otransb = 'N'
          ELSE
            otransb = transb
          END IF
          ! TODO: transpose is currently not supported by LIBXSMM
          IF (('N'.EQ.otransa).OR.('n'.EQ.otransa)) THEN
            IF (('N'.EQ.otransb).OR.('n'.EQ.otransb)) THEN
              CALL libxsmm_dgemm('N', 'N',                              &
     &          SIZE(c, 1, LIBXSMM_BLASINT_KIND),                       &
     &          SIZE(c, 2, LIBXSMM_BLASINT_KIND),                       &
     &          SIZE(a, 2, LIBXSMM_BLASINT_KIND),                       &
     &          alpha, a, SIZE(a, 1, LIBXSMM_BLASINT_KIND),             &
     &                 b, SIZE(b, 1, LIBXSMM_BLASINT_KIND),             &
     &           beta, c, SIZE(c, 1, LIBXSMM_BLASINT_KIND))
            ELSE ! A x B^T
              CALL libxsmm_dgemm('N', 'N',                              &
     &          SIZE(c, 1, LIBXSMM_BLASINT_KIND),                       &
     &          SIZE(c, 2, LIBXSMM_BLASINT_KIND),                       &
     &          SIZE(a, 2, LIBXSMM_BLASINT_KIND),                       &
     &          alpha, a, SIZE(a, 1, LIBXSMM_BLASINT_KIND),             &
     &          TRANSPOSE(b), SIZE(b, 2, LIBXSMM_BLASINT_KIND),         &
     &          beta, c, SIZE(c, 1, LIBXSMM_BLASINT_KIND))
            END IF
          ELSE ! A^T
            IF (('N'.EQ.otransb).OR.('n'.EQ.otransb)) THEN
              CALL libxsmm_dgemm('N', 'N',                              &
     &          SIZE(c, 1, LIBXSMM_BLASINT_KIND),                       &
     &          SIZE(c, 2, LIBXSMM_BLASINT_KIND),                       &
     &          SIZE(a, 1, LIBXSMM_BLASINT_KIND),                       &
     &          alpha, TRANSPOSE(a), SIZE(a, 2, LIBXSMM_BLASINT_KIND),  &
     &                 b, SIZE(b, 1, LIBXSMM_BLASINT_KIND),             &
     &           beta, c, SIZE(c, 1, LIBXSMM_BLASINT_KIND))
            ELSE ! A^T x B^T -> C = (B x A)^T
              CALL libxsmm_dgemm('N', 'N',                              &
     &          SIZE(c, 1, LIBXSMM_BLASINT_KIND),                       &
     &          SIZE(c, 2, LIBXSMM_BLASINT_KIND),                       &
     &          SIZE(a, 1, LIBXSMM_BLASINT_KIND),                       &
     &          alpha, b, SIZE(b, 1, LIBXSMM_BLASINT_KIND),             &
     &                 a, SIZE(a, 1, LIBXSMM_BLASINT_KIND),             &
     &           beta, c, SIZE(c, 1, LIBXSMM_BLASINT_KIND))
              s(1) = SIZE(c, 2); s(2) = SIZE(c, 1)
              c = TRANSPOSE(RESHAPE(c, s))
            END IF
          END IF
        END SUBROUTINE

        ! Matrix-copy (2-dimensional copy) routine. If the input (optional)
        ! is not present, the routine is used to zero-fill the out-matrix.
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_matcopy
        PURE SUBROUTINE libxsmm_matcopy(output, input, typesize,        &
     &  m, n, ldi, ldo, prefetch)
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: m
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN), OPTIONAL :: n
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN), OPTIONAL :: ldi
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN), OPTIONAL :: ldo
          INTEGER(C_INT), INTENT(IN), OPTIONAL :: prefetch
          INTEGER(C_INT), INTENT(IN) :: typesize
          TYPE(C_PTR), INTENT(IN), OPTIONAL :: input
          TYPE(C_PTR), INTENT(IN) :: output
          !DIR$ ATTRIBUTES OFFLOAD:MIC :: internal_matcopy
          INTERFACE
            PURE SUBROUTINE internal_matcopy(output, input, typesize,   &
     &      m, n, ldi, ldo, prefetch) BIND(C, NAME="libxsmm_matcopy_")
              IMPORT LIBXSMM_BLASINT_KIND, C_PTR, C_INT
              INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: m, n
              INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: ldi, ldo
              TYPE(C_PTR), INTENT(IN), VALUE :: output, input
              INTEGER(C_INT), INTENT(IN) :: typesize
              INTEGER(C_INT), INTENT(IN) :: prefetch
            END SUBROUTINE
          END INTERFACE
          CALL internal_matcopy(output, input, typesize,                &
     &      m, n, ldi, ldo, prefetch)
        END SUBROUTINE

        ! Transpose a matrix (out-of-place form).
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_otrans
        PURE SUBROUTINE libxsmm_otrans(output, input, typesize,         &
     &  m, n, ldi, ldo)
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: m
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN), OPTIONAL :: n
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN), OPTIONAL :: ldi
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN), OPTIONAL :: ldo
          TYPE(C_PTR), INTENT(IN) :: output, input
          INTEGER(C_INT), INTENT(IN) :: typesize
          !DIR$ ATTRIBUTES OFFLOAD:MIC :: internal_otrans
          INTERFACE
            PURE SUBROUTINE internal_otrans(output, input, typesize,    &
     &      m, n, ldi, ldo) BIND(C, NAME="libxsmm_otrans_")
              IMPORT LIBXSMM_BLASINT_KIND, C_PTR, C_INT
              INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: m, n
              INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: ldi, ldo
              TYPE(C_PTR), INTENT(IN), VALUE :: output, input
              INTEGER(C_INT), INTENT(IN) :: typesize
            END SUBROUTINE
          END INTERFACE
          CALL internal_otrans(output, input, typesize, m, n, ldi, ldo)
        END SUBROUTINE

        ! Transpose a matrix (out-of-place form, single-precision).
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_sotrans
        SUBROUTINE libxsmm_sotrans(output, input, m, n)
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN), VALUE :: m
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN), OPTIONAL :: n
          REAL(C_FLOAT), INTENT(OUT), TARGET :: output(:,:)
          REAL(C_FLOAT), INTENT(IN), TARGET :: input(:,:)
          CALL libxsmm_otrans(srealptr(input), srealptr(output),        &
     &      4, m, n, UBOUND(input,1), UBOUND(output,1))
        END SUBROUTINE

        ! Transpose a matrix (out-of-place form, double-precision).
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_dotrans
        SUBROUTINE libxsmm_dotrans(output, input, m, n)
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN), VALUE :: m
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN), OPTIONAL :: n
          REAL(C_DOUBLE), INTENT(OUT), TARGET :: output(:,:)
          REAL(C_DOUBLE), INTENT(IN), TARGET :: input(:,:)
          CALL libxsmm_otrans(drealptr(input), drealptr(output),        &
     &      8, m, n, UBOUND(input,1), UBOUND(output,1))
        END SUBROUTINE

        ! Transpose a matrix (in-place form).
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_itrans
        PURE SUBROUTINE libxsmm_itrans(matrix, typesize, m, n, ld)
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: m
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN), OPTIONAL :: n, ld
          TYPE(C_PTR), INTENT(IN) :: matrix
          INTEGER(C_INT), INTENT(IN) :: typesize
          !DIR$ ATTRIBUTES OFFLOAD:MIC :: internal_itrans
          INTERFACE
            PURE SUBROUTINE internal_itrans(matrix, typesize, m, n, ld) &
     &      BIND(C, NAME="libxsmm_itrans_")
              IMPORT LIBXSMM_BLASINT_KIND, C_PTR, C_INT
              INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: m, n, ld
              TYPE(C_PTR), INTENT(IN), VALUE :: matrix
              INTEGER(C_INT), INTENT(IN) :: typesize
            END SUBROUTINE
          END INTERFACE
          CALL internal_itrans(matrix, typesize, m, n, ld)
        END SUBROUTINE

        ! Transpose a matrix (in-place form, single-precision).
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_sitrans
        SUBROUTINE libxsmm_sitrans(matrix, m, n)
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN), VALUE :: m
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN), OPTIONAL :: n
          REAL(C_FLOAT), INTENT(INOUT), TARGET :: matrix(:,:)
          CALL libxsmm_itrans(srealptr(matrix),                         &
     &      4, m, n, UBOUND(matrix,1))
        END SUBROUTINE

        ! Transpose a matrix (in-place form, double-precision).
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_ditrans
        SUBROUTINE libxsmm_ditrans(matrix, m, n)
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN), VALUE :: m
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN), OPTIONAL :: n
          REAL(C_DOUBLE), INTENT(INOUT), TARGET :: matrix(:,:)
          CALL libxsmm_itrans(drealptr(matrix),                         &
     &      8, m, n, UBOUND(matrix,1))
        END SUBROUTINE
      END MODULE

