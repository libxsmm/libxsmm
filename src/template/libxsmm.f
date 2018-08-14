!*****************************************************************************!
!* Copyright (c) 2014-2018, Intel Corporation                                *!
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
     &    C_FLOAT, C_DOUBLE, C_CHAR, C_SHORT, C_INT, C_LONG_LONG,       &
     &    C_INTPTR_T, C_F_POINTER, C_LOC, C_PTR
        IMPLICIT NONE

        PRIVATE :: drealptr, srealptr, irealptr, wrealptr

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
        INTEGER(C_INT), PARAMETER :: LIBXSMM_CACHELINE = $CACHELINE
        INTEGER(C_INT), PARAMETER :: LIBXSMM_ALIGNMENT = $CACHELINE
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
     &    LIBXSMM_GEMM_FLAG_NONE    = 0,                                &
     &    LIBXSMM_GEMM_FLAG_TRANS_A = 1,                                &
     &    LIBXSMM_GEMM_FLAG_TRANS_B = 2

        ! Flag enumeration which can be IORed.
        INTEGER(C_INT), PARAMETER ::                                    &
          ! Handle recorded batch unsynchronized-parallel.
     &    LIBXSMM_MMBATCH_FLAG_DEFAULT      = 0,                        &
          ! Synchronize among C matrices.
     &    LIBXSMM_MMBATCH_FLAG_SYNCHRONIZED = 256,                      &
          ! Handle recorded batch sequentially.
     &    LIBXSMM_MMBATCH_FLAG_SEQUENTIAL   = 512,                      &
          ! Only record a statistic of potential SMMs.
     &    LIBXSMM_MMBATCH_FLAG_STATISTIC    = 1024

        ! Flag which denotes the value type (for weak-typed interface
        ! functions such as libxsmm_xmmdispatch).
        INTEGER(C_INT), PARAMETER ::                                    &
     &    LIBXSMM_GEMM_PRECISION_F64 = 0,                               &
     &    LIBXSMM_GEMM_PRECISION_F32 = 1,                               &
     &    LIBXSMM_GEMM_PRECISION_I32 = 2,                               &
     &    LIBXSMM_GEMM_PRECISION_I16 = 3,                               &
     &    LIBXSMM_GEMM_PRECISION_I8  = 4

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
     &    LIBXSMM_X86_AVX512_CORE = 1020,                               &
     &    LIBXSMM_X86_AVX512_ICL  = 1022

        ! Generic function type (double-precision).
        TYPE :: LIBXSMM_DMMFUNCTION
          PRIVATE
            INTEGER(C_INTPTR_T) :: handle
        END TYPE

        ! Generic function type (single-precision).
        TYPE :: LIBXSMM_SMMFUNCTION
          PRIVATE
            INTEGER(C_INTPTR_T) :: handle
        END TYPE

        ! Generic function type (single-precision).
        TYPE :: LIBXSMM_WIMMFUNCTION
          PRIVATE
            INTEGER(C_INTPTR_T) :: handle
        END TYPE

        ! Generic function type (single-precision).
        TYPE :: LIBXSMM_WSMMFUNCTION
          PRIVATE
            INTEGER(C_INTPTR_T) :: handle
        END TYPE

        ! Construct JIT-code depending on given argument set.
        INTERFACE libxsmm_mmdispatch
          MODULE PROCEDURE libxsmm_dmmdispatch, libxsmm_smmdispatch
          MODULE PROCEDURE libxsmm_wimmdispatch, libxsmm_wsmmdispatch
        END INTERFACE

        ! Construct JIT-code depending on given argument set.
        INTERFACE libxsmm_dispatch
          MODULE PROCEDURE libxsmm_dmmdispatch, libxsmm_smmdispatch
          MODULE PROCEDURE libxsmm_wimmdispatch, libxsmm_wsmmdispatch
        END INTERFACE

        ! Check if a function is available (LIBXSMM_?MMFUNCTION).
        INTERFACE libxsmm_mmavailable
          MODULE PROCEDURE libxsmm_dmmavailable, libxsmm_smmavailable
          MODULE PROCEDURE libxsmm_wimmavailable, libxsmm_wsmmavailable
        END INTERFACE

        ! Check if a function is available (LIBXSMM_?MMFUNCTION).
        INTERFACE libxsmm_available
          MODULE PROCEDURE libxsmm_smmavailable, libxsmm_dmmavailable
          MODULE PROCEDURE libxsmm_wimmavailable, libxsmm_wsmmavailable
        END INTERFACE

        ! Call a specialized function.
        INTERFACE libxsmm_mmcall
          MODULE PROCEDURE libxsmm_dmmcall_abc, libxsmm_dmmcall_prf
          MODULE PROCEDURE libxsmm_smmcall_abc, libxsmm_smmcall_prf
        END INTERFACE

        ! Overloaded GEMM routines (single/double precision).
        INTERFACE libxsmm_gemm
          MODULE PROCEDURE libxsmm_dgemm,  libxsmm_sgemm
          MODULE PROCEDURE libxsmm_wigemm, libxsmm_wsgemm
        END INTERFACE

        ! Overloaded BLAS GEMM routines (single/double precision).
        INTERFACE libxsmm_blas_gemm
          MODULE PROCEDURE libxsmm_blas_sgemm, libxsmm_blas_dgemm
        END INTERFACE

        ! Overloaded MATMUL-style routines (single/double precision).
        INTERFACE libxsmm_matmul
          MODULE PROCEDURE libxsmm_dmatmul, libxsmm_smatmul
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
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_xmmdispatch2
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_xmmdispatch
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_xmmcall_abc
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_xmmcall_prf
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_otrans_omp
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_dgemm_omp
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_sgemm_omp
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_gemm_batch
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_gemm_batch_omp
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_mmbatch
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_mmbatch_omp
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_mmbatch_begin
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_mmbatch_end
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_timer_duration
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_timer_cycles
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_timer_tick
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

          ! Set target architecture (arch="0|sse|snb|hsw|knl|knm|skx|icl", "0": CPUID)
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
          ! Implicit FORTRAN 77 interface: not available.
          INTEGER(C_LONG_LONG) FUNCTION libxsmm_timer_tick() BIND(C)
            IMPORT :: C_LONG_LONG
          END FUNCTION

          ! Returns the difference between two timer ticks (cycles).
          ! Implicit FORTRAN 77 interface: not available.
          PURE FUNCTION libxsmm_timer_cycles(tick0, tick1) BIND(C)
            IMPORT :: C_LONG_LONG
            INTEGER(C_LONG_LONG), INTENT(IN), VALUE :: tick0, tick1
            INTEGER(C_LONG_LONG) :: libxsmm_timer_cycles
          END FUNCTION

          ! Impure function (timer freq. may vary) which returns the duration
          ! (in seconds) between two values received by libxsmm_timer_tick.
          ! Implicit FORTRAN 77 interface: not available.
          FUNCTION libxsmm_timer_duration(tick0, tick1) BIND(C)
            IMPORT :: C_LONG_LONG, C_DOUBLE
            INTEGER(C_LONG_LONG), INTENT(IN), VALUE :: tick0, tick1
            REAL(C_DOUBLE) :: libxsmm_timer_duration
          END FUNCTION

          ! Type-generic (unsafe) code dispatch (trylock: impure routine).
          ! Implicit FORTRAN 77 interface:
          ! INTEGER(4)   :: prec, flags, prefetch
          ! INTEGER(4|8) :: m, n, k, lda, ldb, ldc
          ! REAL(4|8)    :: alpha, beta
          ! INTEGER(8)   :: kernel
          SUBROUTINE libxsmm_xmmdispatch(kernel, prec,                  &
     &    m, n, k, lda, ldb, ldc, alpha, beta, flags, prefetch)         &
     &    BIND(C, NAME="libxsmm_xmmdispatch_") ! FORTRAN 77 layer
            IMPORT :: C_INTPTR_T, C_PTR, C_INT, LIBXSMM_BLASINT_KIND
            INTEGER(C_INTPTR_T), INTENT(OUT) :: kernel
            INTEGER(C_INT), INTENT(IN) :: prec
            INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: m, n, k
            TYPE(C_PTR), INTENT(IN), VALUE :: lda, ldb, ldc
            TYPE(C_PTR), INTENT(IN), VALUE :: alpha, beta
            TYPE(C_PTR), INTENT(IN), VALUE :: flags, prefetch
          END SUBROUTINE

          ! Type-generic (unsafe) code dispatch (trylock: impure routine).
          ! Implicit FORTRAN 77 interface:
          ! INTEGER(4)   :: iprec, oprec, flags, prefetch
          ! INTEGER(4|8) :: m, n, k, lda, ldb, ldc
          ! REAL(4|8)    :: alpha, beta
          ! INTEGER(8)   :: kernel
          SUBROUTINE libxsmm_xmmdispatch2(kernel, iprec, oprec,         &
     &    m, n, k, lda, ldb, ldc, alpha, beta, flags, prefetch)         &
     &    BIND(C, NAME="libxsmm_xmmdispatch2_") ! FORTRAN 77 layer
            IMPORT :: C_INTPTR_T, C_PTR, C_INT, LIBXSMM_BLASINT_KIND
            INTEGER(C_INTPTR_T), INTENT(OUT) :: kernel
            INTEGER(C_INT), INTENT(IN) :: iprec, oprec
            INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: m, n, k
            TYPE(C_PTR), INTENT(IN), VALUE :: lda, ldb, ldc
            TYPE(C_PTR), INTENT(IN), VALUE :: alpha, beta
            TYPE(C_PTR), INTENT(IN), VALUE :: flags, prefetch
          END SUBROUTINE

          ! Generic call routine (3-argument form).
          ! Implicit FORTRAN 77 interface:
          ! REAL(4|8)  :: a(1), b(1), c(1)
          ! INTEGER(8) :: kernel
          PURE SUBROUTINE libxsmm_xmmcall_abc(kernel, a, b, c)            &
     &    BIND(C, NAME="libxsmm_xmmcall_abc_") ! FORTRAN 77 layer
            IMPORT :: C_INTPTR_T, C_PTR
            INTEGER(C_INTPTR_T), INTENT(IN) :: kernel
            TYPE(C_PTR), INTENT(IN), VALUE :: a, b, c
          END SUBROUTINE

          ! Generic call routine (6-argument form).
          ! Implicit FORTRAN 77 interface:
          ! REAL(4|8)  :: a(1), b(1), c(1), pa(1), pb(1), pc(1)
          ! INTEGER(8) :: kernel
          PURE SUBROUTINE libxsmm_xmmcall_prf(kernel, a,b,c, pa,pb,pc)  &
     &    BIND(C, NAME="libxsmm_xmmcall_prf_") ! FORTRAN 77 layer
            IMPORT :: C_INTPTR_T, C_PTR
            INTEGER(C_INTPTR_T), INTENT(IN) :: kernel
            TYPE(C_PTR), INTENT(IN), VALUE :: a, b, c, pa, pb, pc
          END SUBROUTINE

          ! Matrix transposition; MT via libxsmmext (out-of-place form).
          ! Implicit FORTRAN 77 interface:
          ! INTEGER(4|8) :: m, n, ldi, ldo
          ! ANY ARRAY    :: output, input
          ! INTEGER(4)   :: typesize
          PURE SUBROUTINE libxsmm_otrans_omp(output, input, typesize,   &
     &    m, n, ldi, ldo) BIND(C, NAME="libxsmm_otrans_omp_") ! FORTRAN 77 layer
            IMPORT C_PTR, C_INT, LIBXSMM_BLASINT_KIND
            INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: m, n, ldi, ldo
            TYPE(C_PTR), INTENT(IN), VALUE :: output, input
            INTEGER(C_INT), INTENT(IN) :: typesize
          END SUBROUTINE

          ! General dense matrix multiplication; MT via libxsmmext (double-precision).
          ! Implicit FORTRAN 77 interface: similar to DGEMM.
          PURE SUBROUTINE libxsmm_dgemm_omp(transa, transb, m, n, k,    &
     &    alpha, a, lda, b, ldb, beta, c, ldc)                          &
     &    BIND(C, NAME="libxsmm_dgemm_omp_") ! FORTRAN 77 layer
            IMPORT C_DOUBLE, C_CHAR, LIBXSMM_BLASINT_KIND
            CHARACTER(C_CHAR), INTENT(IN) :: transa, transb
            INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: m, n, k
            INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: lda, ldb, ldc
            REAL(C_DOUBLE), INTENT(IN) :: alpha, beta
            REAL(C_DOUBLE), INTENT(IN) :: a(lda,*), b(ldb,*)
            REAL(C_DOUBLE), INTENT(INOUT) :: c(ldc,*)
          END SUBROUTINE

          ! General dense matrix multiplication; MT via libxsmmext (single-precision).
          ! Implicit FORTRAN 77 interface: similar to SGEMM.
          PURE SUBROUTINE libxsmm_sgemm_omp(transa, transb, m, n, k,    &
     &    alpha, a, lda, b, ldb, beta, c, ldc)                          &
     &    BIND(C, NAME="libxsmm_sgemm_omp_") ! FORTRAN 77 layer
            IMPORT C_FLOAT, C_CHAR, LIBXSMM_BLASINT_KIND
            CHARACTER(C_CHAR), INTENT(IN) :: transa, transb
            INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: m, n, k
            INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: lda, ldb, ldc
            REAL(C_FLOAT), INTENT(IN) :: alpha, beta
            REAL(C_FLOAT), INTENT(IN) :: a(lda,*), b(ldb,*)
            REAL(C_FLOAT), INTENT(INOUT) :: c(ldc,*)
          END SUBROUTINE

          ! Process a series of matrix multiplications (batch); sequential.
          ! For the documentation of the call arguments have a look at libxsmm_mmbatch.
          ! Implicit FORTRAN 77 interface:
          ! INTEGER(4)   :: prec
          ! REAL(4|8)    :: alpha, beta
          ! ARRAY        :: a, b, c
          ! ARRAY/VALUE  :: stride_a, stride_b, stride_c
          ! INTEGER(4|8) :: index_base, index_stride, batchsize
          ! Otherwise arguments are similar to GEMM.
          PURE SUBROUTINE libxsmm_gemm_batch(prec, transa, transb,      &
     &    m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, index_base,     &
     &    index_stride, stride_a, stride_b, stride_c, batchsize)        &
     &    BIND(C, NAME="libxsmm_gemm_batch_") ! FORTRAN 77 layer
            IMPORT C_PTR, C_CHAR, C_INT, LIBXSMM_BLASINT_KIND
            INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: index_base
            INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: index_stride
            INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: batchsize
            INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: m, n, k
            INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: lda, ldb, ldc
            CHARACTER(C_CHAR), INTENT(IN) :: transa, transb
            TYPE(C_PTR), INTENT(IN), VALUE :: alpha, beta
            TYPE(C_PTR), INTENT(IN), VALUE :: a, b, c
            TYPE(C_PTR), INTENT(IN), VALUE :: stride_a
            TYPE(C_PTR), INTENT(IN), VALUE :: stride_b
            TYPE(C_PTR), INTENT(IN), VALUE :: stride_c
            INTEGER(C_INT), INTENT(IN) :: prec
          END SUBROUTINE

          ! Process a series of matrix multiplications (batch); MT via libxsmmext.
          ! For the documentation of the call arguments have a look at libxsmm_mmbatch.
          ! Implicit FORTRAN 77 interface:
          ! INTEGER(4)   :: prec
          ! REAL(4|8)    :: alpha, beta
          ! ARRAY        :: a, b, c
          ! ARRAY/VALUE  :: stride_a, stride_b, stride_c
          ! INTEGER(4|8) :: index_base, index_stride, batchsize
          ! Otherwise arguments are similar to GEMM.
          PURE SUBROUTINE libxsmm_gemm_batch_omp(prec, transa, transb,  &
     &    m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, index_base,     &
     &    index_stride, stride_a, stride_b, stride_c, batchsize)        &
     &    BIND(C, NAME="libxsmm_gemm_batch_omp_") ! FORTRAN 77 layer
            IMPORT C_PTR, C_CHAR, C_INT, LIBXSMM_BLASINT_KIND
            INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: index_base
            INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: index_stride
            INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: batchsize
            INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: m, n, k
            INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: lda, ldb, ldc
            CHARACTER(C_CHAR), INTENT(IN) :: transa, transb
            TYPE(C_PTR), INTENT(IN), VALUE :: alpha, beta
            TYPE(C_PTR), INTENT(IN), VALUE :: a, b, c
            TYPE(C_PTR), INTENT(IN), VALUE :: stride_a
            TYPE(C_PTR), INTENT(IN), VALUE :: stride_b
            TYPE(C_PTR), INTENT(IN), VALUE :: stride_c
            INTEGER(C_INT), INTENT(IN) :: prec
          END SUBROUTINE

          ! Process a series of matrix multiplications (batch).
          ! Implicit FORTRAN 77 interface:
          ! INTEGER(4)   :: tid, nthreads
          ! INTEGER(8)   :: kernel
          ! ARRAY        :: a, b, c
          ! ARRAY/VALUE  :: stride_a, stride_b, stride_c
          ! INTEGER(4|8) :: index_base, index_stride, batchsize
          PURE SUBROUTINE libxsmm_mmbatch(kernel, index_base,           &
     &    index_stride, stride_a, stride_b, stride_c, a, b, c,          &
     &    batchsize, tid, nthreads) BIND(C, NAME="libxsmm_mmbatch_") ! FORTRAN 77 layer
            IMPORT :: C_INTPTR_T, C_PTR, C_INT, LIBXSMM_BLASINT_KIND
            ! Determines index-base (1 for one-based indexes);
            ! uses the same unit as the strides.
            INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: index_base
            ! Stride used to walk stride_a, stride_b, and stride_c;
            ! zero turns stride_* into scalar values. The index_stride
            ! is always measured in Bytes (value of LIBXSMM_BLASINT_KIND
            ! determines a packed array of indexes).
            INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: index_stride
            ! The number of matrix multiplications. If the size is given as
            ! a negative value, then internal synchronization is omitted.
            INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: batchsize
            ! Precision, Thread-ID (TID), and the number of threads.
            INTEGER(C_INT), INTENT(IN) :: tid, nthreads
            ! Kernel (matches precision, transa, transb, beta, etc.).
            INTEGER(C_INTPTR_T), INTENT(IN) :: kernel
            ! index_stride==0: a single value (in Bytes) for stride_* is expected,
            ! index_stride!=0: stride_* are arrays of indexes (measured in elements);
            !                  array size equals batchsize, and indexes are discovered
            !                  using the index_stride (in Bytes). The typical value of
            !                  index_stride is LIBXSMM_BLASINT_KIND (packed indexes).
            ! A stride of zero (zero-index) does not advance the matrix-operand.
            ! Note: accesses to the same C-matrix are internally synchronized.
            TYPE(C_PTR), INTENT(IN), VALUE :: stride_a
            TYPE(C_PTR), INTENT(IN), VALUE :: stride_b
            TYPE(C_PTR), INTENT(IN), VALUE :: stride_c
            ! Arrays of matrix operands (a, b, c). Depending on index_stride:
            ! index_stride==0: pointers to pointers of elements.
            ! index_stride!=0: pointer to elements.
            TYPE(C_PTR), INTENT(IN), VALUE :: a, b, c
          END SUBROUTINE

          ! Process a series of matrix multiplications (batch)
          ! similar to libxsmm_mmbatch; MT via libxsmmext.
          ! Implicit FORTRAN 77 interface:
          ! INTEGER(4)   :: tid, nthreads
          ! INTEGER(8)   :: kernel
          ! ARRAY        :: a, b, c
          ! ARRAY/VALUE  :: stride_a, stride_b, stride_c
          ! INTEGER(4|8) :: index_base, index_stride, batchsize
          PURE SUBROUTINE libxsmm_mmbatch_omp(kernel, index_base,       &
     &    index_stride, stride_a, stride_b, stride_c, a, b, c,          &
     &    batchsize) BIND(C, NAME="libxsmm_mmbatch_omp_") ! FORTRAN 77 layer
            IMPORT :: C_INTPTR_T, C_PTR, C_INT, LIBXSMM_BLASINT_KIND
            INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: index_base
            INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: index_stride
            INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: batchsize
            INTEGER(C_INTPTR_T), INTENT(IN) :: kernel
            TYPE(C_PTR), INTENT(IN), VALUE :: stride_a
            TYPE(C_PTR), INTENT(IN), VALUE :: stride_b
            TYPE(C_PTR), INTENT(IN), VALUE :: stride_c
            TYPE(C_PTR), INTENT(IN), VALUE :: a, b, c
          END SUBROUTINE

          ! This function is a no-op unless LIBXSMM is built to intercept GEMM calls.
          ! Pointer arguments are used to filter intercepted GEMM calls such that
          ! non-NULL values match. Otherwise (NULL) the respective argument is
          ! considered a "free value" i.e., every value can match; libxsmmext required.
          ! Implicit FORTRAN 77 interface:
          ! INTEGER(4)   :: prec, flags
          ! INTEGER(4|8) :: m, n, k, lda, ldb, ldc
          ! REAL(4|8)    :: alpha, beta
          SUBROUTINE libxsmm_mmbatch_begin(prec, flags, m, n, k,        &
     &    lda, ldb, ldc, alpha, beta) BIND(C)
            IMPORT C_PTR, C_INT, LIBXSMM_BLASINT_KIND
            INTEGER(C_INT), INTENT(IN), VALUE :: prec
            INTEGER(C_INT), INTENT(IN) :: flags
            INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: m, n, k
            INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: lda, ldb, ldc
            TYPE(C_PTR), INTENT(IN), VALUE :: alpha, beta
          END SUBROUTINE

          ! Processes the batch of previously recorded matrix multiplications
          ! (libxsmm_mmbatch_begin); libxsmmext required.
          ! Implicit FORTRAN 77 interface: available.
          SUBROUTINE libxsmm_mmbatch_end() BIND(C)
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
          !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmmf_get_target_arch
          INTERFACE
            FUNCTION libxsmmf_get_target_arch(length) BIND(C)
              IMPORT :: C_INT, C_PTR
              INTEGER(C_INT), INTENT(OUT) :: length
              TYPE(C_PTR) :: libxsmmf_get_target_arch
            END FUNCTION
          END INTERFACE
          arch = libxsmmf_get_target_arch(length(1))
          CALL C_F_POINTER(arch, libxsmm_get_target_arch, length)
        END FUNCTION

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: drealptr
        FUNCTION drealptr(a)
          REAL(C_DOUBLE), INTENT(IN), TARGET :: a(:,:)
          REAL(C_DOUBLE), POINTER :: fptr
          TYPE(C_PTR) :: drealptr
          fptr => a(LBOUND(a,1),LBOUND(a,2))
          drealptr = C_LOC(fptr)
        END FUNCTION

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: srealptr
        FUNCTION srealptr(a)
          REAL(C_FLOAT), INTENT(IN), TARGET :: a(:,:)
          REAL(C_FLOAT), POINTER :: fptr
          TYPE(C_PTR) :: srealptr
          fptr => a(LBOUND(a,1),LBOUND(a,2))
          srealptr = C_LOC(fptr)
        END FUNCTION

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: irealptr
        FUNCTION irealptr(a)
          INTEGER(C_INT), INTENT(IN), TARGET :: a(:,:)
          INTEGER(C_INT), POINTER :: fptr
          TYPE(C_PTR) :: irealptr
          fptr => a(LBOUND(a,1),LBOUND(a,2))
          irealptr = C_LOC(fptr)
        END FUNCTION

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: wrealptr
        FUNCTION wrealptr(a)
          INTEGER(C_SHORT), INTENT(IN), TARGET :: a(:,:)
          INTEGER(C_SHORT), POINTER :: fptr
          TYPE(C_PTR) :: wrealptr
          fptr => a(LBOUND(a,1),LBOUND(a,2))
          wrealptr = C_LOC(fptr)
        END FUNCTION

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_dmmdispatch
        SUBROUTINE libxsmm_dmmdispatch(kernel,                          &
     &  m, n, k, lda, ldb, ldc, alpha, beta, flags, prefetch)
          TYPE(LIBXSMM_DMMFUNCTION), INTENT(OUT) :: kernel
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN), VALUE :: m, n, k
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN),                    &
     &                                OPTIONAL, TARGET :: lda, ldb, ldc
          REAL(C_DOUBLE), INTENT(IN), OPTIONAL, TARGET :: alpha, beta
          INTEGER(C_INT), INTENT(IN), OPTIONAL, TARGET :: flags
          INTEGER(C_INT), INTENT(IN), OPTIONAL, TARGET :: prefetch
          CALL libxsmm_xmmdispatch(                                     &
     &      kernel%handle, LIBXSMM_GEMM_PRECISION_F64,                  &
     &      m, n, k, C_LOC(lda), C_LOC(ldb), C_LOC(ldc),                &
     &      C_LOC(alpha), C_LOC(beta), C_LOC(flags), C_LOC(prefetch))
        END SUBROUTINE

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_smmdispatch
        SUBROUTINE libxsmm_smmdispatch(kernel,                          &
     &  m, n, k, lda, ldb, ldc, alpha, beta, flags, prefetch)
          TYPE(LIBXSMM_SMMFUNCTION), INTENT(OUT) :: kernel
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN), VALUE :: m, n, k
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN),                    &
     &                                OPTIONAL, TARGET :: lda, ldb, ldc
          REAL(C_FLOAT),  INTENT(IN), OPTIONAL, TARGET :: alpha, beta
          INTEGER(C_INT), INTENT(IN), OPTIONAL, TARGET :: flags
          INTEGER(C_INT), INTENT(IN), OPTIONAL, TARGET :: prefetch
          CALL libxsmm_xmmdispatch(                                     &
     &      kernel%handle, LIBXSMM_GEMM_PRECISION_F32,                  &
     &      m, n, k, C_LOC(lda), C_LOC(ldb), C_LOC(ldc),                &
     &      C_LOC(alpha), C_LOC(beta), C_LOC(flags), C_LOC(prefetch))
        END SUBROUTINE

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_wimmdispatch
        SUBROUTINE libxsmm_wimmdispatch(kernel,                         &
     &  m, n, k, lda, ldb, ldc, alpha, beta, flags, prefetch)
          TYPE(LIBXSMM_WIMMFUNCTION), INTENT(OUT) :: kernel
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN), VALUE :: m, n, k
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN),                    &
     &                                OPTIONAL, TARGET :: lda, ldb, ldc
          INTEGER(C_INT), INTENT(IN), OPTIONAL, TARGET :: alpha, beta
          INTEGER(C_INT), INTENT(IN), OPTIONAL, TARGET :: flags
          INTEGER(C_INT), INTENT(IN), OPTIONAL, TARGET :: prefetch
          CALL libxsmm_xmmdispatch2(kernel%handle,                      &
     &      LIBXSMM_GEMM_PRECISION_I16, LIBXSMM_GEMM_PRECISION_I32,     &
     &      m, n, k, C_LOC(lda), C_LOC(ldb), C_LOC(ldc),                &
     &      C_LOC(alpha), C_LOC(beta), C_LOC(flags), C_LOC(prefetch))
        END SUBROUTINE

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_wsmmdispatch
        SUBROUTINE libxsmm_wsmmdispatch(kernel,                         &
     &  m, n, k, lda, ldb, ldc, alpha, beta, flags, prefetch)
          TYPE(LIBXSMM_WSMMFUNCTION), INTENT(OUT) :: kernel
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN), VALUE :: m, n, k
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN),                    &
     &                                OPTIONAL, TARGET :: lda, ldb, ldc
          REAL(C_FLOAT),  INTENT(IN), OPTIONAL, TARGET :: alpha, beta
          INTEGER(C_INT), INTENT(IN), OPTIONAL, TARGET :: flags
          INTEGER(C_INT), INTENT(IN), OPTIONAL, TARGET :: prefetch
          CALL libxsmm_xmmdispatch2(kernel%handle,                      &
     &      LIBXSMM_GEMM_PRECISION_I16, LIBXSMM_GEMM_PRECISION_F32,     &
     &      m, n, k, C_LOC(lda), C_LOC(ldb), C_LOC(ldc),                &
     &      C_LOC(alpha), C_LOC(beta), C_LOC(flags), C_LOC(prefetch))
        END SUBROUTINE

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_dmmavailable
        LOGICAL PURE FUNCTION libxsmm_dmmavailable(kernel)
          TYPE(LIBXSMM_DMMFUNCTION), INTENT(IN) :: kernel
          libxsmm_dmmavailable = 0.NE.kernel%handle
        END FUNCTION

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_smmavailable
        LOGICAL PURE FUNCTION libxsmm_smmavailable(kernel)
          TYPE(LIBXSMM_SMMFUNCTION), INTENT(IN) :: kernel
          libxsmm_smmavailable = 0.NE.kernel%handle
        END FUNCTION

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_wimmavailable
        LOGICAL PURE FUNCTION libxsmm_wimmavailable(kernel)
          TYPE(LIBXSMM_WIMMFUNCTION), INTENT(IN) :: kernel
          libxsmm_wimmavailable = 0.NE.kernel%handle
        END FUNCTION

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_wsmmavailable
        LOGICAL PURE FUNCTION libxsmm_wsmmavailable(kernel)
          TYPE(LIBXSMM_WSMMFUNCTION), INTENT(IN) :: kernel
          libxsmm_wsmmavailable = 0.NE.kernel%handle
        END FUNCTION

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_dmmcall
        SUBROUTINE libxsmm_dmmcall(kernel, a,b,c, pa,pb,pc)
          TYPE(LIBXSMM_DMMFUNCTION), INTENT(IN) :: kernel
          REAL(C_DOUBLE), INTENT(IN), TARGET :: a(*), b(*)
          REAL(C_DOUBLE), INTENT(INOUT), TARGET :: c(*)
          REAL(C_DOUBLE), INTENT(IN), OPTIONAL, TARGET :: pa(*)
          REAL(C_DOUBLE), INTENT(IN), OPTIONAL, TARGET :: pb(*)
          REAL(C_DOUBLE), INTENT(IN), OPTIONAL, TARGET :: pc(*)
          IF (PRESENT(pa).AND.PRESENT(pb).AND.PRESENT(pc)) THEN
            CALL libxsmm_xmmcall_prf(kernel%handle,                     &
     &        C_LOC(a), C_LOC(b), C_LOC(c),                             &
     &        C_LOC(pa), C_LOC(pb), C_LOC(pc))
          ELSE
            CALL libxsmm_xmmcall_abc(kernel%handle,                     &
     &        C_LOC(a), C_LOC(b), C_LOC(c))
          END IF
        END SUBROUTINE

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_smmcall
        SUBROUTINE libxsmm_smmcall(kernel, a,b,c, pa,pb,pc)
          TYPE(LIBXSMM_SMMFUNCTION), INTENT(IN) :: kernel
          REAL(C_FLOAT), INTENT(IN), TARGET :: a(*), b(*)
          REAL(C_FLOAT), INTENT(INOUT), TARGET :: c(*)
          REAL(C_FLOAT), INTENT(IN), OPTIONAL, TARGET :: pa(*)
          REAL(C_FLOAT), INTENT(IN), OPTIONAL, TARGET :: pb(*)
          REAL(C_FLOAT), INTENT(IN), OPTIONAL, TARGET :: pc(*)
          IF (PRESENT(pa).AND.PRESENT(pb).AND.PRESENT(pc)) THEN
            CALL libxsmm_xmmcall_prf(kernel%handle,                     &
     &        C_LOC(a), C_LOC(b), C_LOC(c),                             &
     &        C_LOC(pa), C_LOC(pb), C_LOC(pc))
          ELSE
            CALL libxsmm_xmmcall_abc(kernel%handle,                     &
     &        C_LOC(a), C_LOC(b), C_LOC(c))
          END IF
        END SUBROUTINE

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_wimmcall
        SUBROUTINE libxsmm_wimmcall(kernel, a,b,c, pa,pb,pc)
          TYPE(LIBXSMM_WIMMFUNCTION), INTENT(IN) :: kernel
          INTEGER(C_SHORT), INTENT(IN),  TARGET :: a(*), b(*)
          INTEGER(C_INT), INTENT(INOUT), TARGET :: c(*)
          INTEGER(C_SHORT), INTENT(IN), OPTIONAL, TARGET :: pa(*)
          INTEGER(C_SHORT), INTENT(IN), OPTIONAL, TARGET :: pb(*)
          INTEGER(C_INT),   INTENT(IN), OPTIONAL, TARGET :: pc(*)
          IF (PRESENT(pa).AND.PRESENT(pb).AND.PRESENT(pc)) THEN
            CALL libxsmm_xmmcall_prf(kernel%handle,                     &
     &        C_LOC(a), C_LOC(b), C_LOC(c),                             &
     &        C_LOC(pa), C_LOC(pb), C_LOC(pc))
          ELSE
            CALL libxsmm_xmmcall_abc(kernel%handle,                     &
     &        C_LOC(a), C_LOC(b), C_LOC(c))
          END IF
        END SUBROUTINE

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_wsmmcall
        SUBROUTINE libxsmm_wsmmcall(kernel, a,b,c, pa,pb,pc)
          TYPE(LIBXSMM_WSMMFUNCTION), INTENT(IN) :: kernel
          INTEGER(C_SHORT), INTENT(IN), TARGET :: a(*), b(*)
          REAL(C_FLOAT), INTENT(INOUT), TARGET :: c(*)
          INTEGER(C_SHORT), INTENT(IN), OPTIONAL, TARGET :: pa(*)
          INTEGER(C_SHORT), INTENT(IN), OPTIONAL, TARGET :: pb(*)
          REAL(C_FLOAT), INTENT(IN), OPTIONAL, TARGET :: pc(*)
          IF (PRESENT(pa).AND.PRESENT(pb).AND.PRESENT(pc)) THEN
            CALL libxsmm_xmmcall_prf(kernel%handle,                     &
     &        C_LOC(a), C_LOC(b), C_LOC(c),                             &
     &        C_LOC(pa), C_LOC(pb), C_LOC(pc))
          ELSE
            CALL libxsmm_xmmcall_abc(kernel%handle,                     &
     &        C_LOC(a), C_LOC(b), C_LOC(c))
          END IF
        END SUBROUTINE

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_dmmcall_abc
        PURE SUBROUTINE libxsmm_dmmcall_abc(kernel, a, b, c)
          TYPE(LIBXSMM_DMMFUNCTION), INTENT(IN) :: kernel
          TYPE(C_PTR), INTENT(IN) :: a, b, c
          CALL libxsmm_xmmcall_abc(kernel%handle, a, b, c)
        END SUBROUTINE

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_smmcall_abc
        PURE SUBROUTINE libxsmm_smmcall_abc(kernel, a, b, c)
          TYPE(LIBXSMM_SMMFUNCTION), INTENT(IN) :: kernel
          TYPE(C_PTR), INTENT(IN) :: a, b, c
          CALL libxsmm_xmmcall_abc(kernel%handle, a, b, c)
        END SUBROUTINE

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_wimmcall_abc
        PURE SUBROUTINE libxsmm_wimmcall_abc(kernel, a, b, c)
          TYPE(LIBXSMM_WIMMFUNCTION), INTENT(IN) :: kernel
          TYPE(C_PTR), INTENT(IN) :: a, b, c
          CALL libxsmm_xmmcall_abc(kernel%handle, a, b, c)
        END SUBROUTINE

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_wsmmcall_abc
        PURE SUBROUTINE libxsmm_wsmmcall_abc(kernel, a, b, c)
          TYPE(LIBXSMM_WSMMFUNCTION), INTENT(IN) :: kernel
          TYPE(C_PTR), INTENT(IN) :: a, b, c
          CALL libxsmm_xmmcall_abc(kernel%handle, a, b, c)
        END SUBROUTINE

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_dmmcall_prf
        PURE SUBROUTINE libxsmm_dmmcall_prf(kernel, a,b,c, pa,pb,pc)
          TYPE(LIBXSMM_DMMFUNCTION), INTENT(IN) :: kernel
          TYPE(C_PTR), INTENT(IN) :: a, b, c, pa, pb, pc
          CALL libxsmm_xmmcall_prf(kernel%handle, a, b, c, pa, pb, pc)
        END SUBROUTINE

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_smmcall_prf
        PURE SUBROUTINE libxsmm_smmcall_prf(kernel, a,b,c, pa,pb,pc)
          TYPE(LIBXSMM_SMMFUNCTION), INTENT(IN) :: kernel
          TYPE(C_PTR), INTENT(IN) :: a, b, c, pa, pb, pc
          CALL libxsmm_xmmcall_prf(kernel%handle, a, b, c, pa, pb, pc)
        END SUBROUTINE

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_wimmcall_prf
        PURE SUBROUTINE libxsmm_wimmcall_prf(kernel, a,b,c, pa,pb,pc)
          TYPE(LIBXSMM_WIMMFUNCTION), INTENT(IN) :: kernel
          TYPE(C_PTR), INTENT(IN) :: a, b, c, pa, pb, pc
          CALL libxsmm_xmmcall_prf(kernel%handle, a, b, c, pa, pb, pc)
        END SUBROUTINE

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_wsmmcall_prf
        PURE SUBROUTINE libxsmm_wsmmcall_prf(kernel, a,b,c, pa,pb,pc)
          TYPE(LIBXSMM_WSMMFUNCTION), INTENT(IN) :: kernel
          TYPE(C_PTR), INTENT(IN) :: a, b, c, pa, pb, pc
          CALL libxsmm_xmmcall_prf(kernel%handle, a, b, c, pa, pb, pc)
        END SUBROUTINE

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_dgemm
        SUBROUTINE libxsmm_dgemm(transa, transb, m, n, k,               &
     &  alpha, a, lda, b, ldb, beta, c, ldc)
          CHARACTER, INTENT(IN), OPTIONAL, TARGET :: transa, transb
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN), VALUE :: m, n, k
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN),                    &
     &                                OPTIONAL, TARGET :: lda, ldb, ldc
          REAL(C_DOUBLE), INTENT(IN), OPTIONAL, TARGET :: alpha, beta
          REAL(C_DOUBLE), INTENT(IN), TARGET :: a(:,:), b(:,:)
          REAL(C_DOUBLE), INTENT(INOUT), TARGET :: c(:,:)
          !DIR$ ATTRIBUTES OFFLOAD:MIC :: internal_gemm
          INTERFACE
            SUBROUTINE internal_gemm(transa, transb, m, n, k,           &
     &      alpha, a, lda, b, ldb, beta, c, ldc)                        &
     &      BIND(C, NAME="libxsmm_dgemm")
              IMPORT C_PTR, LIBXSMM_BLASINT_KIND
              TYPE(C_PTR), INTENT(IN), VALUE :: transa, transb
              INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: m, n, k
              TYPE(C_PTR), INTENT(IN), VALUE :: lda, ldb, ldc
              TYPE(C_PTR), INTENT(IN), VALUE :: alpha, beta
              TYPE(C_PTR), INTENT(IN), VALUE :: a, b, c
            END SUBROUTINE
          END INTERFACE
          CALL internal_gemm(C_LOC(transa), C_LOC(transb), m, n, k,     &
     &      C_LOC(alpha), drealptr(a), C_LOC(lda),                      &
     &                    drealptr(b), C_LOC(ldb),                      &
     &       C_LOC(beta), drealptr(c), C_LOC(ldc))
        END SUBROUTINE

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_sgemm
        SUBROUTINE libxsmm_sgemm(transa, transb, m, n, k,               &
     &  alpha, a, lda, b, ldb, beta, c, ldc)
          CHARACTER, INTENT(IN), OPTIONAL, TARGET :: transa, transb
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN), VALUE :: m, n, k
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN),                    &
     &                               OPTIONAL, TARGET :: lda, ldb, ldc
          REAL(C_FLOAT), INTENT(IN), OPTIONAL, TARGET :: alpha, beta
          REAL(C_FLOAT), INTENT(IN), TARGET :: a(:,:), b(:,:)
          REAL(C_FLOAT), INTENT(INOUT), TARGET :: c(:,:)
          !DIR$ ATTRIBUTES OFFLOAD:MIC :: internal_gemm
          INTERFACE
            SUBROUTINE internal_gemm(transa, transb, m, n, k,           &
     &      alpha, a, lda, b, ldb, beta, c, ldc)                        &
     &      BIND(C, NAME="libxsmm_sgemm")
              IMPORT C_PTR, LIBXSMM_BLASINT_KIND
              TYPE(C_PTR), INTENT(IN), VALUE :: transa, transb
              INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: m, n, k
              TYPE(C_PTR), INTENT(IN), VALUE :: lda, ldb, ldc
              TYPE(C_PTR), INTENT(IN), VALUE :: alpha, beta
              TYPE(C_PTR), INTENT(IN), VALUE :: a, b, c
            END SUBROUTINE
          END INTERFACE
          CALL internal_gemm(C_LOC(transa), C_LOC(transb), m, n, k,     &
     &      C_LOC(alpha), srealptr(a), C_LOC(lda),                      &
     &                    srealptr(b), C_LOC(ldb),                      &
     &       C_LOC(beta), srealptr(c), C_LOC(ldc))
        END SUBROUTINE

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_wigemm
        SUBROUTINE libxsmm_wigemm(transa, transb, m, n, k,               &
     &  alpha, a, lda, b, ldb, beta, c, ldc)
          CHARACTER, INTENT(IN), OPTIONAL, TARGET :: transa, transb
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN), VALUE :: m, n, k
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN),                    &
     &                                OPTIONAL, TARGET :: lda, ldb, ldc
          INTEGER(C_INT), INTENT(IN), OPTIONAL, TARGET :: alpha, beta
          INTEGER(C_SHORT),  INTENT(IN), TARGET :: a(:,:), b(:,:)
          INTEGER(C_INT), INTENT(INOUT), TARGET :: c(:,:)
          !DIR$ ATTRIBUTES OFFLOAD:MIC :: internal_gemm
          INTERFACE
            SUBROUTINE internal_gemm(transa, transb, m, n, k,           &
     &      alpha, a, lda, b, ldb, beta, c, ldc)                        &
     &      BIND(C, NAME="libxsmm_wigemm")
              IMPORT C_PTR, LIBXSMM_BLASINT_KIND
              TYPE(C_PTR), INTENT(IN), VALUE :: transa, transb
              INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: m, n, k
              TYPE(C_PTR), INTENT(IN), VALUE :: lda, ldb, ldc
              TYPE(C_PTR), INTENT(IN), VALUE :: alpha, beta
              TYPE(C_PTR), INTENT(IN), VALUE :: a, b, c
            END SUBROUTINE
          END INTERFACE
          CALL internal_gemm(C_LOC(transa), C_LOC(transb), m, n, k,     &
     &      C_LOC(alpha), wrealptr(a), C_LOC(lda),                      &
     &                    wrealptr(b), C_LOC(ldb),                      &
     &       C_LOC(beta), irealptr(c), C_LOC(ldc))
        END SUBROUTINE

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_wsgemm
        SUBROUTINE libxsmm_wsgemm(transa, transb, m, n, k,               &
     &  alpha, a, lda, b, ldb, beta, c, ldc)
          CHARACTER, INTENT(IN), OPTIONAL, TARGET :: transa, transb
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN), VALUE :: m, n, k
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN),                    &
     &                               OPTIONAL, TARGET :: lda, ldb, ldc
          REAL(C_FLOAT), INTENT(IN), OPTIONAL, TARGET :: alpha, beta
          INTEGER(C_SHORT), INTENT(IN), TARGET :: a(:,:), b(:,:)
          REAL(C_FLOAT), INTENT(INOUT), TARGET :: c(:,:)
          !DIR$ ATTRIBUTES OFFLOAD:MIC :: internal_gemm
          INTERFACE
            SUBROUTINE internal_gemm(transa, transb, m, n, k,           &
     &      alpha, a, lda, b, ldb, beta, c, ldc)                        &
     &      BIND(C, NAME="libxsmm_wsgemm")
              IMPORT C_PTR, LIBXSMM_BLASINT_KIND
              TYPE(C_PTR), INTENT(IN), VALUE :: transa, transb
              INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: m, n, k
              TYPE(C_PTR), INTENT(IN), VALUE :: lda, ldb, ldc
              TYPE(C_PTR), INTENT(IN), VALUE :: alpha, beta
              TYPE(C_PTR), INTENT(IN), VALUE :: a, b, c
            END SUBROUTINE
          END INTERFACE
          CALL internal_gemm(C_LOC(transa), C_LOC(transb), m, n, k,     &
     &      C_LOC(alpha), wrealptr(a), C_LOC(lda),                      &
     &                    wrealptr(b), C_LOC(ldb),                      &
     &       C_LOC(beta), srealptr(c), C_LOC(ldc))
        END SUBROUTINE

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_blas_dgemm
        SUBROUTINE libxsmm_blas_dgemm(transa, transb, m, n, k,          &
     &  alpha, a, lda, b, ldb, beta, c, ldc)
          CHARACTER, INTENT(IN), OPTIONAL, TARGET :: transa, transb
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN), VALUE :: m, n, k
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN),                    &
     &                                OPTIONAL, TARGET :: lda, ldb, ldc
          REAL(C_DOUBLE), INTENT(IN), OPTIONAL, TARGET :: alpha, beta
          REAL(C_DOUBLE), INTENT(IN), TARGET :: a(:,:), b(:,:)
          REAL(C_DOUBLE), INTENT(INOUT), TARGET :: c(:,:)
          !DIR$ ATTRIBUTES OFFLOAD:MIC :: internal_gemm
          INTERFACE
            SUBROUTINE internal_gemm(transa, transb, m, n, k,           &
     &      alpha, a, lda, b, ldb, beta, c, ldc)                        &
     &      BIND(C, NAME="libxsmm_blas_dgemm_")
              IMPORT C_PTR, LIBXSMM_BLASINT_KIND
              TYPE(C_PTR), INTENT(IN), VALUE :: transa, transb
              INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: m, n, k
              TYPE(C_PTR), INTENT(IN), VALUE :: lda, ldb, ldc
              TYPE(C_PTR), INTENT(IN), VALUE :: alpha, beta
              TYPE(C_PTR), INTENT(IN), VALUE :: a, b, c
            END SUBROUTINE
          END INTERFACE
          CALL internal_gemm(C_LOC(transa), C_LOC(transb), m, n, k,     &
     &      C_LOC(alpha), drealptr(a), C_LOC(lda),                      &
     &                    drealptr(b), C_LOC(ldb),                      &
     &       C_LOC(beta), drealptr(c), C_LOC(ldc))
        END SUBROUTINE

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_blas_sgemm
        SUBROUTINE libxsmm_blas_sgemm(transa, transb, m, n, k,          &
     &  alpha, a, lda, b, ldb, beta, c, ldc)
          CHARACTER, INTENT(IN), OPTIONAL, TARGET :: transa, transb
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN), VALUE :: m, n, k
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN),                    &
     &                               OPTIONAL, TARGET :: lda, ldb, ldc
          REAL(C_FLOAT), INTENT(IN), OPTIONAL, TARGET :: alpha, beta
          REAL(C_FLOAT), INTENT(IN), TARGET :: a(:,:), b(:,:)
          REAL(C_FLOAT), INTENT(INOUT), TARGET :: c(:,:)
          !DIR$ ATTRIBUTES OFFLOAD:MIC :: internal_gemm
          INTERFACE
            SUBROUTINE internal_gemm(transa, transb, m, n, k,           &
     &      alpha, a, lda, b, ldb, beta, c, ldc)                        &
     &      BIND(C, NAME="libxsmm_blas_sgemm_")
              IMPORT C_PTR, LIBXSMM_BLASINT_KIND
              TYPE(C_PTR), INTENT(IN), VALUE :: transa, transb
              INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: m, n, k
              TYPE(C_PTR), INTENT(IN), VALUE :: lda, ldb, ldc
              TYPE(C_PTR), INTENT(IN), VALUE :: alpha, beta
              TYPE(C_PTR), INTENT(IN), VALUE :: a, b, c
            END SUBROUTINE
          END INTERFACE
          CALL internal_gemm(C_LOC(transa), C_LOC(transb), m, n, k,     &
     &      C_LOC(alpha), srealptr(a), C_LOC(lda),                      &
     &                    srealptr(b), C_LOC(ldb),                      &
     &       C_LOC(beta), srealptr(c), C_LOC(ldc))
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

        ! Matrix-copy (2-dimensional copy) routine. If the input (optional)
        ! is not present, the routine is used to zero-fill the out-matrix.
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_matcopy
        SUBROUTINE libxsmm_matcopy(output, input, typesize,             &
     &  m, n, ldi, ldo, prefetch)
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: m
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN),                    &
     &                                OPTIONAL, TARGET :: n, ldi, ldo
          INTEGER(C_INT), INTENT(IN), OPTIONAL, TARGET :: prefetch
          INTEGER(C_INT), INTENT(IN) :: typesize
          TYPE(C_PTR), INTENT(IN), OPTIONAL :: input
          TYPE(C_PTR), INTENT(IN) :: output
          !DIR$ ATTRIBUTES OFFLOAD:MIC :: internal_matcopy
          INTERFACE
            PURE SUBROUTINE internal_matcopy(output, input, typesize,   &
     &      m, n, ldi, ldo, prefetch) BIND(C, NAME="libxsmm_matcopy_") ! FORTRAN 77 layer
              IMPORT LIBXSMM_BLASINT_KIND, C_PTR, C_INT
              INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: m
              TYPE(C_PTR), INTENT(IN), VALUE :: n, ldi, ldo
              TYPE(C_PTR), INTENT(IN), VALUE :: output, input
              TYPE(C_PTR), INTENT(IN), VALUE :: prefetch
              INTEGER(C_INT), INTENT(IN) :: typesize
            END SUBROUTINE
          END INTERFACE
          CALL internal_matcopy(output, input, typesize,                &
     &      m, C_LOC(n), C_LOC(ldi), C_LOC(ldo), C_LOC(prefetch))
        END SUBROUTINE

        ! Transpose a matrix (out-of-place form).
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_otrans
        SUBROUTINE libxsmm_otrans(output, input, typesize,              &
     &  m, n, ldi, ldo)
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: m
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN),                    &
     &                                   OPTIONAL, TARGET :: n, ldi, ldo
          TYPE(C_PTR), INTENT(IN) :: output, input
          INTEGER(C_INT), INTENT(IN) :: typesize
          !DIR$ ATTRIBUTES OFFLOAD:MIC :: internal_otrans
          INTERFACE
            PURE SUBROUTINE internal_otrans(output, input, typesize,    &
     &      m, n, ldi, ldo) BIND(C, NAME="libxsmm_otrans_") ! FORTRAN 77 layer
              IMPORT LIBXSMM_BLASINT_KIND, C_PTR, C_INT
              INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: m
              TYPE(C_PTR), INTENT(IN), VALUE :: n, ldi, ldo
              TYPE(C_PTR), INTENT(IN), VALUE :: output, input
              INTEGER(C_INT), INTENT(IN) :: typesize
            END SUBROUTINE
          END INTERFACE
          CALL internal_otrans(output, input, typesize,                 &
     &      m, C_LOC(n), C_LOC(ldi), C_LOC(ldo))
        END SUBROUTINE

        ! Transpose a matrix (in-place form).
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_itrans
        SUBROUTINE libxsmm_itrans(matrix, typesize, m, n, ld)
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: m
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN),                    &
     &                                   OPTIONAL, TARGET :: n, ld
          TYPE(C_PTR), INTENT(IN) :: matrix
          INTEGER(C_INT), INTENT(IN) :: typesize
          !DIR$ ATTRIBUTES OFFLOAD:MIC :: internal_itrans
          INTERFACE
            PURE SUBROUTINE internal_itrans(matrix, typesize, m, n, ld) &
     &      BIND(C, NAME="libxsmm_itrans_") ! FORTRAN 77 layer
              IMPORT LIBXSMM_BLASINT_KIND, C_PTR, C_INT
              INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: m
              TYPE(C_PTR), INTENT(IN), VALUE :: n, ld, matrix
              INTEGER(C_INT), INTENT(IN) :: typesize
            END SUBROUTINE
          END INTERFACE
          CALL internal_itrans(matrix, typesize, m, C_LOC(n), C_LOC(ld))
        END SUBROUTINE

        ! Calculate a hash value for a given key value (array).
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_hash
        PURE FUNCTION libxsmm_hash(key, seed)
          INTEGER(C_INT), DIMENSION(:), INTENT(IN) :: key
          INTEGER(C_INT), INTENT(IN) :: seed
          INTEGER(C_INT) :: libxsmm_hash
          !DIR$ ATTRIBUTES OFFLOAD:MIC :: internal_hash
          INTERFACE
            PURE SUBROUTINE internal_hash(hash, key, keysize, seed)     &
     &      BIND(C, NAME="libxsmm_hash_") ! FORTRAN 77 layer
              IMPORT C_INT
              INTEGER(C_INT), INTENT(IN)  :: key
              INTEGER(C_INT), INTENT(IN)  :: keysize
              INTEGER(C_INT), INTENT(IN)  :: seed
              INTEGER(C_INT), INTENT(OUT) :: hash
            END SUBROUTINE
          END INTERFACE
          CALL internal_hash(libxsmm_hash,                              &
     &      key(LBOUND(key,1)), SIZE(key) * 4, seed)
        END FUNCTION
      END MODULE

