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

      MODULE LIBXSMM
        USE, INTRINSIC :: ISO_C_BINDING, ONLY:                          &
     &    C_DOUBLE, C_FLOAT, C_DOUBLE_COMPLEX, C_FLOAT_COMPLEX,         &
     &    C_LONG_LONG, C_INT, C_SHORT, C_CHAR, C_INT8_T,                &
     &    C_F_POINTER, C_ASSOCIATED, C_LOC, C_PTR, C_NULL_PTR,          &
     &    C_FUNPTR, C_NULL_FUNPTR
        IMPLICIT NONE

        !> Name of the version (stringized set of version numbers).
        CHARACTER(*), PARAMETER :: LIBXSMM_VERSION = "$VERSION"
        !> Name of the branch of which the version is derived from.
        CHARACTER(*), PARAMETER :: LIBXSMM_BRANCH = "$BRANCH"
        !> Major version based on the last reachable tag under RCS.
        INTEGER(C_INT), PARAMETER :: LIBXSMM_VERSION_MAJOR = $MAJOR
        !> Minor version based on the last reachable tag of the RCS.
        INTEGER(C_INT), PARAMETER :: LIBXSMM_VERSION_MINOR = $MINOR
        !> Update number based on the last reachable tag under RCS.
        INTEGER(C_INT), PARAMETER :: LIBXSMM_VERSION_UPDATE = $UPDATE
        !> Patch number counting commits since the last version stamp.
        INTEGER(C_INT), PARAMETER :: LIBXSMM_VERSION_PATCH = $PATCH

        !> Parameters the library and static kernels were built for.
        INTEGER(C_INT), PARAMETER :: LIBXSMM_CACHELINE = $CACHELINE
        INTEGER(C_INT), PARAMETER :: LIBXSMM_ALIGNMENT = $CACHELINE
        INTEGER(C_INT), PARAMETER :: LIBXSMM_PREFETCH = $PREFETCH
        INTEGER(C_INT), PARAMETER :: LIBXSMM_MAX_MNK = $MAX_MNK
        INTEGER(C_INT), PARAMETER :: LIBXSMM_MAX_DIM = $MAX_DIM
        INTEGER(C_INT), PARAMETER :: LIBXSMM_FLAGS = $FLAGS
        INTEGER(C_INT), PARAMETER :: LIBXSMM_ILP64 = $ILP64

        !> Parameters supplied for backward compatibility (deprecated).
        INTEGER(C_INT), PARAMETER :: LIBXSMM_COL_MAJOR = 1
        INTEGER(C_INT), PARAMETER :: LIBXSMM_ROW_MAJOR = 0

        !> LIBXSMM_BLASINT_KIND impacts BLAS interface (LP64: 32-bit, ILP64: 64-bit).
        INTEGER(C_INT), PARAMETER :: LIBXSMM_BLASINT_KIND = $BLASINT_KIND
        !> Integer kind used by timer interface.
        INTEGER(C_INT), PARAMETER :: LIBXSMM_TICKINT_KIND = C_LONG_LONG

        !> Parameters representing the GEMM performed by the simplified interface.
        REAL(C_DOUBLE), PARAMETER :: LIBXSMM_ALPHA = REAL($ALPHA, C_DOUBLE)
        REAL(C_DOUBLE), PARAMETER :: LIBXSMM_BETA = REAL($BETA, C_DOUBLE)

        !> Flag enumeration which can be IORed.
        INTEGER(C_INT), PARAMETER ::                                    &
     &    LIBXSMM_GEMM_FLAG_NONE     = 0,                               &
     &    LIBXSMM_GEMM_FLAG_TRANS_A  = 1,                               &
     &    LIBXSMM_GEMM_FLAG_TRANS_B  = 2,                               &
     &    LIBXSMM_GEMM_FLAG_TRANS_AB = IOR(                             &
     &        LIBXSMM_GEMM_FLAG_TRANS_A, LIBXSMM_GEMM_FLAG_TRANS_B),    &
     &    LIBXSMM_GEMM_FLAG_BETA_0   = 16

        !> Flag enumeration which can be IORed.
        INTEGER(C_INT), PARAMETER ::                                    &
          ! Handle recorded batch unsynchronized-parallel.
     &    LIBXSMM_MMBATCH_FLAG_DEFAULT      = 0,                        &
          ! Synchronize among C matrices.
     &    LIBXSMM_MMBATCH_FLAG_SYNCHRONIZED = 512,                      &
          ! Handle recorded batch sequentially.
     &    LIBXSMM_MMBATCH_FLAG_SEQUENTIAL   = 1024,                     &
          ! Only record a statistic of potential SMMs.
     &    LIBXSMM_MMBATCH_FLAG_STATISTIC    = 2048

        !> Enumerates element/data types.
        INTEGER(C_INT), PARAMETER ::                                    &
     &    LIBXSMM_DATATYPE_F64  = 0,                                    &
     &    LIBXSMM_DATATYPE_F32  = 1,                                    &
     &    LIBXSMM_DATATYPE_BF16 = 2,                                    &
     &    LIBXSMM_DATATYPE_I64  = 3,                                    &
     &    LIBXSMM_DATATYPE_I32  = 4,                                    &
     &    LIBXSMM_DATATYPE_I16  = 5,                                    &
     &    LIBXSMM_DATATYPE_I8   = 6,                                    &
     &    LIBXSMM_DATATYPE_UNSUPPORTED = 7

        !> Denotes the precision/data type of GEMM (for weak-typed
        !> interface functions such as libxsmm_xmmdispatch).
        INTEGER(C_INT), PARAMETER ::                                    &
     &    LIBXSMM_GEMM_PRECISION_F64  = LIBXSMM_DATATYPE_F64,           &
     &    LIBXSMM_GEMM_PRECISION_F32  = LIBXSMM_DATATYPE_F32,           &
     &    LIBXSMM_GEMM_PRECISION_BF16 = LIBXSMM_DATATYPE_BF16,          &
     &    LIBXSMM_GEMM_PRECISION_I32  = LIBXSMM_DATATYPE_I32,           &
     &    LIBXSMM_GEMM_PRECISION_I16  = LIBXSMM_DATATYPE_I16,           &
     &    LIBXSMM_GEMM_PRECISION_I8   = LIBXSMM_DATATYPE_I8

        !> Enumeration of the available prefetch strategies which can be IORed.
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

        !> Enumerates the available target architectures and instruction
        !> set extensions as returned by libxsmm_get_target_archid().
        INTEGER(C_INT), PARAMETER ::                                    &
     &    LIBXSMM_TARGET_ARCH_UNKNOWN = 0,                              &
     &    LIBXSMM_TARGET_ARCH_GENERIC = 1,                              &
     &    LIBXSMM_X86_GENERIC     = 1002,                               &
     &    LIBXSMM_X86_SSE3        = 1003,                               &
     &    LIBXSMM_X86_SSE4        = 1004,                               &
     &    LIBXSMM_X86_AVX         = 1005,                               &
     &    LIBXSMM_X86_AVX2        = 1006,                               &
     &    LIBXSMM_X86_AVX512      = 1007,                               &
     &    LIBXSMM_X86_AVX512_MIC  = 1010,                               &
     &    LIBXSMM_X86_AVX512_KNM  = 1011,                               &
     &    LIBXSMM_X86_AVX512_CORE = 1020,                               &
     &    LIBXSMM_X86_AVX512_CLX  = 1021,                               &
     &    LIBXSMM_X86_AVX512_CPX  = 1022

        !> Generic function type (double-precision).
        TYPE :: LIBXSMM_DMMFUNCTION
          TYPE(C_FUNPTR) :: handle = C_NULL_FUNPTR
        END TYPE

        !> Generic function type (single-precision).
        TYPE :: LIBXSMM_SMMFUNCTION
          TYPE(C_FUNPTR) :: handle = C_NULL_FUNPTR
        END TYPE

        !> Generic function type (low-precision)
        TYPE :: LIBXSMM_WIMMFUNCTION
          TYPE(C_FUNPTR) :: handle = C_NULL_FUNPTR
        END TYPE

        !> Generic function types with certain arity.
        ABSTRACT INTERFACE
          PURE SUBROUTINE LIBXSMM_FUNCTION3(a, b, c) BIND(C)
            IMPORT :: C_PTR
            TYPE(C_PTR), INTENT(IN), VALUE :: a, b, c
          END SUBROUTINE

          PURE SUBROUTINE LIBXSMM_FUNCTION6(a, b, c, pa, pb, pc) BIND(C)
            IMPORT :: C_PTR
            TYPE(C_PTR), INTENT(IN), VALUE :: a, b, c
            TYPE(C_PTR), INTENT(IN), VALUE :: pa, pb, pc
          END SUBROUTINE
        END INTERFACE

        !> Structure of differences with matrix norms according
        !> to http://www.netlib.org/lapack/lug/node75.html).
        TYPE, BIND(C) :: LIBXSMM_MATDIFF_INFO
          REAL(C_DOUBLE) norm1_abs, norm1_rel !! One-norm
          REAL(C_DOUBLE) normi_abs, normi_rel !! Infinity-norm
          REAL(C_DOUBLE) normf_rel            !! Froebenius-norm
          !> Maximum difference, and L2-norm (both absolute and relative).
          REAL(C_DOUBLE) linf_abs, linf_rel, l2_abs, l2_rel
          !> Statistics: sum/l1, min., max., arith. avg., and variance.
          REAL(C_DOUBLE) l1_ref, min_ref, max_ref, avg_ref, var_ref
          !> Statistics: sum/l1, min., max., arith. avg., and variance.
          REAL(C_DOUBLE) l1_tst, min_tst, max_tst, avg_tst, var_tst
          !> Location (m, n) of largest difference (linf_abs).
          INTEGER(LIBXSMM_BLASINT_KIND) m
          INTEGER(LIBXSMM_BLASINT_KIND) n
        END TYPE

        INTERFACE libxsmm_ptr0
          MODULE PROCEDURE libxsmm_ptr_z0, libxsmm_ptr_c0
          MODULE PROCEDURE libxsmm_ptr_d0, libxsmm_ptr_s0
          MODULE PROCEDURE libxsmm_ptr_i0, libxsmm_ptr_w0
          MODULE PROCEDURE libxsmm_ptr_j0 !! Byte/char
          MODULE PROCEDURE libxsmm_ptr_b0 !! Byte/char
          MODULE PROCEDURE libxsmm_ptr_l0 !! long long
        END INTERFACE

        INTERFACE libxsmm_ptr1
          MODULE PROCEDURE libxsmm_ptr_z1, libxsmm_ptr_c1
          MODULE PROCEDURE libxsmm_ptr_d1, libxsmm_ptr_s1
          MODULE PROCEDURE libxsmm_ptr_i1, libxsmm_ptr_w1
          MODULE PROCEDURE libxsmm_ptr_j1 !! Byte/char
          MODULE PROCEDURE libxsmm_ptr_b1 !! Byte/char
          MODULE PROCEDURE libxsmm_ptr_l1 !! long long
        END INTERFACE

        INTERFACE libxsmm_ptr2
          MODULE PROCEDURE libxsmm_ptr_z2, libxsmm_ptr_c2
          MODULE PROCEDURE libxsmm_ptr_d2, libxsmm_ptr_s2
          MODULE PROCEDURE libxsmm_ptr_i2, libxsmm_ptr_w2
          MODULE PROCEDURE libxsmm_ptr_j2 !! Byte/char
          MODULE PROCEDURE libxsmm_ptr_b2 !! Byte/char
          MODULE PROCEDURE libxsmm_ptr_l2 !! long long
        END INTERFACE

        !> Deallocates JIT'ted code, or unregisters/releases code from registry.
        INTERFACE libxsmm_release_mmkernel
          MODULE PROCEDURE libxsmm_release_dmmkernel
          MODULE PROCEDURE libxsmm_release_smmkernel
          MODULE PROCEDURE libxsmm_release_wimmkernel
        END INTERFACE

        !> Construct JIT-code depending on given argument set.
        INTERFACE libxsmm_mmdispatch
          MODULE PROCEDURE libxsmm_dmmdispatch, libxsmm_smmdispatch
          MODULE PROCEDURE libxsmm_wimmdispatch
        END INTERFACE

        !> Construct JIT-code depending on given argument set.
        INTERFACE libxsmm_dispatch
          MODULE PROCEDURE libxsmm_dmmdispatch, libxsmm_smmdispatch
          MODULE PROCEDURE libxsmm_wimmdispatch
        END INTERFACE

        !> Check if a function is available (LIBXSMM_?MMFUNCTION).
        INTERFACE libxsmm_mmavailable
          MODULE PROCEDURE libxsmm_dmmavailable, libxsmm_smmavailable
          MODULE PROCEDURE libxsmm_wimmavailable
        END INTERFACE

        !> Check if a function is available (LIBXSMM_?MMFUNCTION).
        INTERFACE libxsmm_available
          MODULE PROCEDURE libxsmm_smmavailable, libxsmm_dmmavailable
          MODULE PROCEDURE libxsmm_wimmavailable
        END INTERFACE

        !> Call a specialized function.
        INTERFACE libxsmm_mmcall
          MODULE PROCEDURE libxsmm_dmmcall_abc, libxsmm_dmmcall_prf
          MODULE PROCEDURE libxsmm_smmcall_abc, libxsmm_smmcall_prf
        END INTERFACE

        !> Overloaded GEMM routines (double precision).
        INTERFACE libxsmm_dgemm
          MODULE PROCEDURE libxsmm_dgemm0
          MODULE PROCEDURE libxsmm_dgemm1
          MODULE PROCEDURE libxsmm_dgemm2
          MODULE PROCEDURE libxsmm_dgemm3
        END INTERFACE

        !> Overloaded GEMM routines (single precision).
        INTERFACE libxsmm_sgemm
          MODULE PROCEDURE libxsmm_sgemm0
          MODULE PROCEDURE libxsmm_sgemm1
          MODULE PROCEDURE libxsmm_sgemm2
        END INTERFACE

        !> Overloaded GEMM routines (low precision).
        INTERFACE libxsmm_wigemm
          MODULE PROCEDURE libxsmm_wigemm0
          MODULE PROCEDURE libxsmm_wigemm1
          MODULE PROCEDURE libxsmm_wigemm2
        END INTERFACE

        !> Overloaded GEMM routines.
        INTERFACE libxsmm_gemm
          MODULE PROCEDURE libxsmm_dgemm0
          MODULE PROCEDURE libxsmm_dgemm1
          MODULE PROCEDURE libxsmm_dgemm2
          MODULE PROCEDURE libxsmm_dgemm3
          MODULE PROCEDURE libxsmm_sgemm0
          MODULE PROCEDURE libxsmm_sgemm1
          MODULE PROCEDURE libxsmm_sgemm2
          MODULE PROCEDURE libxsmm_sgemm3
          MODULE PROCEDURE libxsmm_wigemm0
          MODULE PROCEDURE libxsmm_wigemm1
          MODULE PROCEDURE libxsmm_wigemm2
          MODULE PROCEDURE libxsmm_wigemm3
        END INTERFACE

        !> Overloaded BLAS GEMM routines (double precision).
        INTERFACE libxsmm_blas_dgemm
          MODULE PROCEDURE libxsmm_blas_dgemm0
          MODULE PROCEDURE libxsmm_blas_dgemm1
          MODULE PROCEDURE libxsmm_blas_dgemm2
          MODULE PROCEDURE libxsmm_blas_dgemm3
        END INTERFACE

        !> Overloaded BLAS GEMM routines (single precision).
        INTERFACE libxsmm_blas_sgemm
          MODULE PROCEDURE libxsmm_blas_sgemm0
          MODULE PROCEDURE libxsmm_blas_sgemm1
          MODULE PROCEDURE libxsmm_blas_sgemm2
          MODULE PROCEDURE libxsmm_blas_sgemm3
        END INTERFACE

        !> Overloaded BLAS GEMM routines (single/double precision).
        INTERFACE libxsmm_blas_gemm
          MODULE PROCEDURE libxsmm_blas_dgemm0
          MODULE PROCEDURE libxsmm_blas_dgemm1
          MODULE PROCEDURE libxsmm_blas_dgemm2
          MODULE PROCEDURE libxsmm_blas_dgemm3
          MODULE PROCEDURE libxsmm_blas_sgemm0
          MODULE PROCEDURE libxsmm_blas_sgemm1
          MODULE PROCEDURE libxsmm_blas_sgemm2
          MODULE PROCEDURE libxsmm_blas_sgemm3
        END INTERFACE

        !> Calculate a hash value for a given key value (binary blob).
        !> Conceptually pure, but C_LOC may be (incorrectly) impure.
        INTERFACE libxsmm_hash
          MODULE PROCEDURE libxsmm_hash_char
          MODULE PROCEDURE libxsmm_hash_i8
          MODULE PROCEDURE libxsmm_hash_i32
          MODULE PROCEDURE libxsmm_hash_i64
        END INTERFACE

        !> Calculate whether there is a difference between two series of items.
        !> Conceptually pure, but C_LOC may be (incorrectly) impure.
        INTERFACE libxsmm_diff
          MODULE PROCEDURE libxsmm_diff_char
          MODULE PROCEDURE libxsmm_diff_i8
          MODULE PROCEDURE libxsmm_diff_i32
          MODULE PROCEDURE libxsmm_diff_i64
        END INTERFACE

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_init, libxsmm_finalize
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_get_gemm_auto_prefetch
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_set_gemm_auto_prefetch
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_get_target_archid
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_set_target_archid
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_set_target_arch
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_get_verbosity
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_set_verbosity
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_release_kernel
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_matdiff_reduce
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_matdiff_clear
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_xmmdispatch2
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_xmmdispatch
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_xmmcall_abc
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_xmmcall_prf
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_otrans_omp
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_dgemm_omp
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_sgemm_omp
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_mmbatch
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_mmbatch_begin
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_mmbatch_end
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_gemm_batch
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_gemm_batch_omp
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_timer_duration
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_timer_tick
        INTERFACE
          !> Initialize the library; pay for setup cost at a specific point.
          SUBROUTINE libxsmm_init() BIND(C)
          END SUBROUTINE

          !> De-initialize the library and free internal memory (optional).
          SUBROUTINE libxsmm_finalize() BIND(C)
          END SUBROUTINE

          !> Get the default prefetch strategy.
          PURE FUNCTION libxsmm_get_gemm_auto_prefetch() BIND(C)
            IMPORT :: C_INT
            INTEGER(C_INT) :: libxsmm_get_gemm_auto_prefetch
          END FUNCTION

          !> Set the default prefetch strategy.
          SUBROUTINE libxsmm_set_gemm_auto_prefetch(strategy) BIND(C)
            IMPORT :: C_INT
            INTEGER(C_INT), INTENT(IN), VALUE :: strategy
          END SUBROUTINE

          !> Returns the architecture and instruction set extension as determined
          !> by the CPUID flags, as set by the libxsmm_get_target_arch* functions,
          !> or as set by the LIBXSMM_TARGET environment variable.
          PURE FUNCTION libxsmm_get_target_archid() BIND(C)
            IMPORT :: C_INT
            INTEGER(C_INT) :: libxsmm_get_target_archid
          END FUNCTION

          !> Set target architecture (archid: see PARAMETER enumeration)
          !> for subsequent code generation (JIT).
          SUBROUTINE libxsmm_set_target_archid(archid) BIND(C)
            IMPORT :: C_INT
            INTEGER(C_INT), INTENT(IN), VALUE :: archid
          END SUBROUTINE

          !> Set target architecture for subsequent code generation (JIT).
          !> arch="0"|"sse"|"snb"|"hsw"|"knl"|"knm"|"skx"|"clx"|"cpx",
          !> or "0" to rely on the CPUID (default).
          !> There are some alternative target names as well:
          !> "sse", "avx", "avx2", "avx3" (incomplete list).
          SUBROUTINE libxsmm_set_target_arch(arch) BIND(C)
            IMPORT :: C_CHAR
            CHARACTER(C_CHAR), INTENT(IN) :: arch(*)
          END SUBROUTINE

          !> Get the level of verbosity.
          PURE FUNCTION libxsmm_get_verbosity() BIND(C)
            IMPORT :: C_INT
            INTEGER(C_INT) :: libxsmm_get_verbosity
          END FUNCTION

          !> Set the level of verbosity (0: off, positive value: verbosity level,
          !> negative value: maximum verbosity, which also dumps JIT-code).
          SUBROUTINE libxsmm_set_verbosity(level) BIND(C)
            IMPORT :: C_INT
            INTEGER(C_INT), INTENT(IN), VALUE :: level
          END SUBROUTINE

          !> Impure function which returns the current clock tick of a
          !> monotonic timer source; uses a platform-specific resolution.
          !> Implicit FORTRAN 77 interface: not available.
          INTEGER(LIBXSMM_TICKINT_KIND)                                 &
     &    FUNCTION libxsmm_timer_tick() BIND(C)
            IMPORT :: LIBXSMM_TICKINT_KIND
          END FUNCTION

          !> Impure function (timer freq. may vary) which returns the duration
          !> (in seconds) between two values received by libxsmm_timer_tick.
          !> Implicit FORTRAN 77 interface: not available.
          FUNCTION libxsmm_timer_duration(tick0, tick1) BIND(C)
            IMPORT :: LIBXSMM_TICKINT_KIND, C_DOUBLE
            INTEGER(LIBXSMM_TICKINT_KIND), INTENT(IN), VALUE :: tick0
            INTEGER(LIBXSMM_TICKINT_KIND), INTENT(IN), VALUE :: tick1
            REAL(C_DOUBLE) :: libxsmm_timer_duration
          END FUNCTION

          !> Deallocates the JIT'ted code, or unregisters
          !> and releases code from the registry.
          !> Implicit FORTRAN 77 interface:
          !> INTEGER(8) :: kernel
          SUBROUTINE libxsmm_release_kernel(kernel)                     &
     &    BIND(C, NAME="libxsmm_release_kernel_")
            IMPORT :: C_FUNPTR
            TYPE(C_FUNPTR), INTENT(IN) :: kernel
          END SUBROUTINE

          !> Type-generic (unsafe) code dispatch (trylock: impure routine).
          !> Implicit FORTRAN 77 interface:
          !> INTEGER(4)   :: gemm_precision, flags, prefetch
          !> INTEGER(4|8) :: m, n, k, lda, ldb, ldc
          !> REAL(4|8)    :: alpha, beta
          !> INTEGER(8)   :: kernel
          SUBROUTINE libxsmm_xmmdispatch(kernel, gemm_precision,        &
     &    m, n, k, lda, ldb, ldc, alpha, beta, flags, prefetch)         &
     &    BIND(C, NAME="libxsmm_xmmdispatch_")
            IMPORT :: C_FUNPTR, C_PTR, C_INT, LIBXSMM_BLASINT_KIND
            TYPE(C_FUNPTR), INTENT(OUT) :: kernel
            INTEGER(C_INT), INTENT(IN) :: gemm_precision
            INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: m, n, k
            TYPE(C_PTR), INTENT(IN), VALUE :: lda, ldb, ldc
            TYPE(C_PTR), INTENT(IN), VALUE :: alpha, beta
            TYPE(C_PTR), INTENT(IN), VALUE :: flags, prefetch
          END SUBROUTINE

          !> Type-generic (unsafe) code dispatch (trylock: impure routine).
          !> Implicit FORTRAN 77 interface:
          !> INTEGER(4)   :: iprec, oprec, flags, prefetch
          !> INTEGER(4|8) :: m, n, k, lda, ldb, ldc
          !> REAL(4|8)    :: alpha, beta
          !> INTEGER(8)   :: kernel
          SUBROUTINE libxsmm_xmmdispatch2(kernel, iprec, oprec,         &
     &    m, n, k, lda, ldb, ldc, alpha, beta, flags, prefetch)         &
     &    BIND(C, NAME="libxsmm_xmmdispatch2_")
            IMPORT :: C_FUNPTR, C_PTR, C_INT, LIBXSMM_BLASINT_KIND
            TYPE(C_FUNPTR), INTENT(OUT) :: kernel
            INTEGER(C_INT), INTENT(IN) :: iprec, oprec
            INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: m, n, k
            TYPE(C_PTR), INTENT(IN), VALUE :: lda, ldb, ldc
            TYPE(C_PTR), INTENT(IN), VALUE :: alpha, beta
            TYPE(C_PTR), INTENT(IN), VALUE :: flags, prefetch
          END SUBROUTINE

          !> Generic call routine (3-argument form).
          !> Implicit FORTRAN 77 interface:
          !> REAL(4|8)  :: a(1), b(1), c(1)
          !> INTEGER(8) :: kernel
          SUBROUTINE libxsmm_xmmcall_abc(kernel, a, b, c)               &
     &    BIND(C, NAME="libxsmm_xmmcall_abc_")
            IMPORT C_FUNPTR, C_PTR
            TYPE(C_FUNPTR), INTENT(IN) :: kernel
            TYPE(C_PTR), INTENT(IN), VALUE :: a, b, c
          END SUBROUTINE

          !> Generic call routine (6-argument form).
          !> Implicit FORTRAN 77 interface:
          !> REAL(4|8)  :: a(1), b(1), c(1), pa(1), pb(1), pc(1)
          !> INTEGER(8) :: kernel
          SUBROUTINE libxsmm_xmmcall_prf(kernel, a,b,c, pa,pb,pc)       &
     &    BIND(C, NAME="libxsmm_xmmcall_prf_")
            IMPORT C_FUNPTR, C_PTR
            TYPE(C_FUNPTR), INTENT(IN) :: kernel
            TYPE(C_PTR), INTENT(IN), VALUE :: a, b, c, pa, pb, pc
          END SUBROUTINE

          !> Matrix transposition; MT via libxsmmext (out-of-place form).
          !> Implicit FORTRAN 77 interface:
          !> INTEGER(4|8) :: m, n, ldi, ldo
          !> ANY ARRAY    :: output, input
          !> INTEGER(4)   :: typesize
          PURE SUBROUTINE libxsmm_otrans_omp(output, input,             &
     &    typesize, m, n, ldi, ldo) BIND(C, NAME="libxsmm_otrans_omp_")
            IMPORT C_PTR, C_INT, LIBXSMM_BLASINT_KIND
            INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: m, n, ldi, ldo
            TYPE(C_PTR), INTENT(IN), VALUE :: output, input
            INTEGER(C_INT), INTENT(IN) :: typesize
          END SUBROUTINE

          !> General dense MM; MT via libxsmmext (double-precision).
          !> Implicit FORTRAN 77 interface: similar to DGEMM.
          PURE SUBROUTINE libxsmm_dgemm_omp(transa, transb, m, n, k,    &
     &    alpha, a, lda, b, ldb, beta, c, ldc)                          &
     &    BIND(C, NAME="libxsmm_dgemm_omp_")
            IMPORT C_DOUBLE, C_CHAR, LIBXSMM_BLASINT_KIND
            CHARACTER(C_CHAR), INTENT(IN) :: transa, transb
            INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: m, n, k
            INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: lda, ldb, ldc
            REAL(C_DOUBLE), INTENT(IN) :: alpha, beta
            REAL(C_DOUBLE), INTENT(IN) :: a(lda,*), b(ldb,*)
            REAL(C_DOUBLE), INTENT(INOUT) :: c(ldc,*)
          END SUBROUTINE

          !> General dense MM; MT via libxsmmext (single-precision).
          !> Implicit FORTRAN 77 interface: similar to SGEMM.
          PURE SUBROUTINE libxsmm_sgemm_omp(transa, transb, m, n, k,    &
     &    alpha, a, lda, b, ldb, beta, c, ldc)                          &
     &    BIND(C, NAME="libxsmm_sgemm_omp_")
            IMPORT C_FLOAT, C_CHAR, LIBXSMM_BLASINT_KIND
            CHARACTER(C_CHAR), INTENT(IN) :: transa, transb
            INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: m, n, k
            INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: lda, ldb, ldc
            REAL(C_FLOAT), INTENT(IN) :: alpha, beta
            REAL(C_FLOAT), INTENT(IN) :: a(lda,*), b(ldb,*)
            REAL(C_FLOAT), INTENT(INOUT) :: c(ldc,*)
          END SUBROUTINE

          !> Process a series of MMs (batch). See also libxsmm_gemm_batch_omp.
          !> The kind of matrix operands (a, b, c) depend on index_stride:
          !> index_stride==0: pointers to pointers of elements e.g., double** for the C matrices.
          !> index_stride!=0: pointer to elements e.g., const double* for the A and B matrices.
          !> Implicit FORTRAN 77 interface:
          !> INTEGER(4)   :: iprec, oprec
          !> REAL(4|8)    :: alpha, beta
          !> ARRAY        :: a, b, c
          !> ARRAY/VALUE  :: stride_a, stride_b, stride_c
          !> INTEGER(4|8) :: index_base, index_stride, batchsize
          !> INTEGER(4)   :: tid, nthreads
          !> Otherwise arguments are similar to GEMM.
          PURE SUBROUTINE libxsmm_mmbatch(iprec, oprec, transa, transb, &
     &    m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, index_base,     &
     &    index_stride, stride_a, stride_b, stride_c, batchsize,        &
     &    tid, nthreads) BIND(C, NAME="libxsmm_mmbatch_")
            IMPORT C_PTR, C_CHAR, C_INT, LIBXSMM_BLASINT_KIND
            !> Determines index-base (usually 0, 1 for one-based indexes).
            INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: index_base
            !> Stride (measured in Bytes) used to walk stride_*. In Fortran: index_stride!=0.
            INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: index_stride
            !> Number of SMMs. If the size is given as a negative value,
            !> then internal synchronization is omitted.
            INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: batchsize
            INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: m, n, k
            INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: lda, ldb, ldc
            CHARACTER(C_CHAR), INTENT(IN) :: transa, transb
            TYPE(C_PTR), INTENT(IN), VALUE :: alpha, beta
            TYPE(C_PTR), INTENT(IN), VALUE :: a, b, c
            !> Arrays of indexes determining the position of a, b, and c operands.
            TYPE(C_PTR), INTENT(IN), VALUE :: stride_a
            TYPE(C_PTR), INTENT(IN), VALUE :: stride_b
            TYPE(C_PTR), INTENT(IN), VALUE :: stride_c
            INTEGER(C_INT), INTENT(IN) :: iprec, oprec
            !> Thread-ID (TID), and number of threads.
            INTEGER(C_INT), INTENT(IN) :: tid, nthreads
          END SUBROUTINE

          !> Process a series of SMMs (batch). See also libxsmm_mmbatch.
          !> Implicit FORTRAN 77 interface:
          !> INTEGER(4)   :: iprec, oprec
          !> REAL(4|8)    :: alpha, beta
          !> ARRAY        :: a, b, c
          !> ARRAY/VALUE  :: stride_a, stride_b, stride_c
          !> INTEGER(4|8) :: index_base, index_stride, batchsize
          !> Otherwise arguments are similar to GEMM.
          PURE SUBROUTINE libxsmm_gemm_batch(iprec, oprec,              &
     &    transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, &
     &    index_base, index_stride, stride_a, stride_b, stride_c,       &
     &    batchsize) BIND(C, NAME="libxsmm_gemm_batch_")
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
            INTEGER(C_INT), INTENT(IN) :: iprec, oprec
          END SUBROUTINE

          !> Process a series of SMMs (batch) with OpenMP (libxsmmext).
          !> Implicit FORTRAN 77 interface:
          !> INTEGER(4)   :: iprec, oprec
          !> REAL(4|8)    :: alpha, beta
          !> ARRAY        :: a, b, c
          !> ARRAY/VALUE  :: stride_a, stride_b, stride_c
          !> INTEGER(4|8) :: index_base, index_stride, batchsize
          !> Otherwise arguments are similar to GEMM.
          PURE SUBROUTINE libxsmm_gemm_batch_omp(iprec, oprec,          &
     &    transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, &
     &    index_base, index_stride, stride_a, stride_b, stride_c,       &
     &    batchsize) BIND(C, NAME="libxsmm_gemm_batch_omp_")
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
            INTEGER(C_INT), INTENT(IN) :: iprec, oprec
          END SUBROUTINE

          !> This function is a no-op unless LIBXSMM is built to intercept GEMM calls.
          !> Pointer arguments are used to filter intercepted GEMM calls such that
          !> non-NULL values match. Otherwise (NULL) the respective argument is
          !> considered a "free value" i.e., every value can match; libxsmmext required.
          !> Implicit FORTRAN 77 interface:
          !> INTEGER(4)   :: gemm_precision, flags
          !> INTEGER(4|8) :: m, n, k, lda, ldb, ldc
          !> REAL(4|8)    :: alpha, beta
          SUBROUTINE libxsmm_mmbatch_begin(gemm_precision, flags,       &
     &    m, n, k,  lda, ldb, ldc, alpha, beta) BIND(C)
            IMPORT C_PTR, C_INT, LIBXSMM_BLASINT_KIND
            INTEGER(C_INT), INTENT(IN), VALUE :: gemm_precision
            INTEGER(C_INT), INTENT(IN) :: flags
            INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: m, n, k
            INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: lda, ldb, ldc
            TYPE(C_PTR), INTENT(IN), VALUE :: alpha, beta
          END SUBROUTINE

          !> Processes the batch of previously recorded SMMs
          !> (libxsmm_mmbatch_begin); libxsmmext required.
          !> Implicit FORTRAN 77 interface: available.
          SUBROUTINE libxsmm_mmbatch_end() BIND(C)
          END SUBROUTINE

          !> Reduces input into output such that the difference is maintained
          !> or increased (max function). The very first (initial) output
          !> should be zeroed (libxsmm_matdiff_clear).
          !> Implicit FORTRAN 77 interface: available.
          PURE SUBROUTINE libxsmm_matdiff_reduce(output, input) BIND(C)
            IMPORT LIBXSMM_MATDIFF_INFO
            TYPE(LIBXSMM_MATDIFF_INFO), INTENT(INOUT) :: output
            TYPE(LIBXSMM_MATDIFF_INFO), INTENT(IN)    :: input
          END SUBROUTINE

          !> Clears the given info-structure e.g., for the initial
          !> reduction-value (libxsmm_matdiff_reduce).
          !> Implicit FORTRAN 77 interface: available.
          PURE SUBROUTINE libxsmm_matdiff_clear(info) BIND(C)
            IMPORT LIBXSMM_MATDIFF_INFO
            TYPE(LIBXSMM_MATDIFF_INFO), INTENT(OUT) :: info
          END SUBROUTINE
        END INTERFACE$MNK_INTERFACE_LIST

      CONTAINS
        !> Returns the name of the target architecture as determined by
        !> the CPUID flags, as set by the libxsmm_get_target_arch* functions,
        !> or as set by the LIBXSMM_TARGET environment variable.
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

        !> Determines the C-address of the given scalar.
        !> This overload belongs to libxsmm_ptr0.
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_ptr_z0
        FUNCTION libxsmm_ptr_z0(a)
          COMPLEX(C_DOUBLE_COMPLEX), INTENT(IN), TARGET :: a
          COMPLEX(C_DOUBLE_COMPLEX), POINTER :: fptr
          TYPE(C_PTR) :: libxsmm_ptr_z0
          fptr => a; libxsmm_ptr_z0 = C_LOC(fptr)
        END FUNCTION

        !> Determines the C-address of the given array.
        !> This overload belongs to libxsmm_ptr1.
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_ptr_z1
        FUNCTION libxsmm_ptr_z1(a)
          COMPLEX(C_DOUBLE_COMPLEX), INTENT(IN) :: a(:)
          TYPE(C_PTR) :: libxsmm_ptr_z1
          IF (0.LT.SIZE(a)) THEN
            libxsmm_ptr_z1 = libxsmm_ptr_z0(a(LBOUND(a,1)))
          ELSE
            libxsmm_ptr_z1 = C_NULL_PTR
          END IF
        END FUNCTION

        !> Determines the C-address of the given 2d-array.
        !> This overload belongs to libxsmm_ptr2.
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_ptr_z2
        FUNCTION libxsmm_ptr_z2(a)
          COMPLEX(C_DOUBLE_COMPLEX), INTENT(IN) :: a(:,:)
          TYPE(C_PTR) :: libxsmm_ptr_z2
          IF (ALL(0.LT.SHAPE(a))) THEN
            libxsmm_ptr_z2 = libxsmm_ptr_z0(a(LBOUND(a,1),LBOUND(a,2)))
          ELSE
            libxsmm_ptr_z2 = C_NULL_PTR
          END IF
        END FUNCTION

        !> Determines the C-address of the given scalar.
        !> This overload belongs to libxsmm_ptr0.
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_ptr_c0
        FUNCTION libxsmm_ptr_c0(a)
          COMPLEX(C_FLOAT_COMPLEX), INTENT(IN), TARGET :: a
          COMPLEX(C_FLOAT_COMPLEX), POINTER :: fptr
          TYPE(C_PTR) :: libxsmm_ptr_c0
          fptr => a; libxsmm_ptr_c0 = C_LOC(fptr)
        END FUNCTION

        !> Determines the C-address of the given array.
        !> This overload belongs to libxsmm_ptr1.
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_ptr_c1
        FUNCTION libxsmm_ptr_c1(a)
          COMPLEX(C_FLOAT_COMPLEX), INTENT(IN) :: a(:)
          TYPE(C_PTR) :: libxsmm_ptr_c1
          IF (0.LT.SIZE(a)) THEN
            libxsmm_ptr_c1 = libxsmm_ptr_c0(a(LBOUND(a,1)))
          ELSE
            libxsmm_ptr_c1 = C_NULL_PTR
          END IF
        END FUNCTION

        !> Determines the C-address of the given 2d-array.
        !> This overload belongs to libxsmm_ptr2.
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_ptr_c2
        FUNCTION libxsmm_ptr_c2(a)
          COMPLEX(C_FLOAT_COMPLEX), INTENT(IN) :: a(:,:)
          TYPE(C_PTR) :: libxsmm_ptr_c2
          IF (ALL(0.LT.SHAPE(a))) THEN
            libxsmm_ptr_c2 = libxsmm_ptr_c0(a(LBOUND(a,1),LBOUND(a,2)))
          ELSE
            libxsmm_ptr_c2 = C_NULL_PTR
          END IF
        END FUNCTION

        !> Determines the C-address of the given scalar.
        !> This overload belongs to libxsmm_ptr0.
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_ptr_d0
        FUNCTION libxsmm_ptr_d0(a)
          REAL(C_DOUBLE), INTENT(IN), TARGET :: a
          REAL(C_DOUBLE), POINTER :: fptr
          TYPE(C_PTR) :: libxsmm_ptr_d0
          fptr => a; libxsmm_ptr_d0 = C_LOC(fptr)
        END FUNCTION

        !> Determines the C-address of the given array.
        !> This overload belongs to libxsmm_ptr1.
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_ptr_d1
        FUNCTION libxsmm_ptr_d1(a)
          REAL(C_DOUBLE), INTENT(IN) :: a(:)
          TYPE(C_PTR) :: libxsmm_ptr_d1
          IF (0.LT.SIZE(a)) THEN
            libxsmm_ptr_d1 = libxsmm_ptr_d0(a(LBOUND(a,1)))
          ELSE
            libxsmm_ptr_d1 = C_NULL_PTR
          END IF
        END FUNCTION

        !> Determines the C-address of the given 2d-array.
        !> This overload belongs to libxsmm_ptr2.
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_ptr_d2
        FUNCTION libxsmm_ptr_d2(a)
          REAL(C_DOUBLE), INTENT(IN) :: a(:,:)
          TYPE(C_PTR) :: libxsmm_ptr_d2
          IF (ALL(0.LT.SHAPE(a))) THEN
            libxsmm_ptr_d2 = libxsmm_ptr_d0(a(LBOUND(a,1),LBOUND(a,2)))
          ELSE
            libxsmm_ptr_d2 = C_NULL_PTR
          END IF
        END FUNCTION

        !> Determines the C-address of the given scalar.
        !> This overload belongs to libxsmm_ptr0.
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_ptr_s0
        FUNCTION libxsmm_ptr_s0(a)
          REAL(C_FLOAT), INTENT(IN), TARGET :: a
          REAL(C_FLOAT), POINTER :: fptr
          TYPE(C_PTR) :: libxsmm_ptr_s0
          fptr => a; libxsmm_ptr_s0 = C_LOC(fptr)
        END FUNCTION

        !> Determines the C-address of the given array.
        !> This overload belongs to libxsmm_ptr1.
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_ptr_s1
        FUNCTION libxsmm_ptr_s1(a)
          REAL(C_FLOAT), INTENT(IN) :: a(:)
          TYPE(C_PTR) :: libxsmm_ptr_s1
          IF (0.LT.SIZE(a)) THEN
            libxsmm_ptr_s1 = libxsmm_ptr_s0(a(LBOUND(a,1)))
          ELSE
            libxsmm_ptr_s1 = C_NULL_PTR
          END IF
        END FUNCTION

        !> Determines the C-address of the given 2d-array.
        !> This overload belongs to libxsmm_ptr2.
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_ptr_s2
        FUNCTION libxsmm_ptr_s2(a)
          REAL(C_FLOAT), INTENT(IN) :: a(:,:)
          TYPE(C_PTR) :: libxsmm_ptr_s2
          IF (ALL(0.LT.SHAPE(a))) THEN
            libxsmm_ptr_s2 = libxsmm_ptr_s0(a(LBOUND(a,1),LBOUND(a,2)))
          ELSE
            libxsmm_ptr_s2 = C_NULL_PTR
          END IF
        END FUNCTION

        !> Determines the C-address of the given scalar.
        !> This overload belongs to libxsmm_ptr0.
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_ptr_i0
        FUNCTION libxsmm_ptr_i0(a)
          INTEGER(C_INT), INTENT(IN), TARGET :: a
          INTEGER(C_INT), POINTER :: fptr
          TYPE(C_PTR) :: libxsmm_ptr_i0
          fptr => a; libxsmm_ptr_i0 = C_LOC(fptr)
        END FUNCTION

        !> Determines the C-address of the given array.
        !> This overload belongs to libxsmm_ptr1.
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_ptr_i1
        FUNCTION libxsmm_ptr_i1(a)
          INTEGER(C_INT), INTENT(IN) :: a(:)
          TYPE(C_PTR) :: libxsmm_ptr_i1
          IF (0.LT.SIZE(a)) THEN
            libxsmm_ptr_i1 = libxsmm_ptr_i0(a(LBOUND(a,1)))
          ELSE
            libxsmm_ptr_i1 = C_NULL_PTR
          END IF
        END FUNCTION

        !> Determines the C-address of the given 2d-array.
        !> This overload belongs to libxsmm_ptr2.
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_ptr_i2
        FUNCTION libxsmm_ptr_i2(a)
          INTEGER(C_INT), INTENT(IN) :: a(:,:)
          TYPE(C_PTR) :: libxsmm_ptr_i2
          IF (ALL(0.LT.SHAPE(a))) THEN
            libxsmm_ptr_i2 = libxsmm_ptr_i0(a(LBOUND(a,1),LBOUND(a,2)))
          ELSE
            libxsmm_ptr_i2 = C_NULL_PTR
          END IF
        END FUNCTION

        !> Determines the C-address of the given scalar.
        !> This overload belongs to libxsmm_ptr0.
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_ptr_w0
        FUNCTION libxsmm_ptr_w0(a)
          INTEGER(C_SHORT), INTENT(IN), TARGET :: a
          INTEGER(C_SHORT), POINTER :: fptr
          TYPE(C_PTR) :: libxsmm_ptr_w0
          fptr => a; libxsmm_ptr_w0 = C_LOC(fptr)
        END FUNCTION

        !> Determines the C-address of the given array.
        !> This overload belongs to libxsmm_ptr1.
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_ptr_w1
        FUNCTION libxsmm_ptr_w1(a)
          INTEGER(C_SHORT), INTENT(IN) :: a(:)
          TYPE(C_PTR) :: libxsmm_ptr_w1
          IF (0.LT.SIZE(a)) THEN
            libxsmm_ptr_w1 = libxsmm_ptr_w0(a(LBOUND(a,1)))
          ELSE
            libxsmm_ptr_w1 = C_NULL_PTR
          END IF
        END FUNCTION

        !> Determines the C-address of the given 2d-array.
        !> This overload belongs to libxsmm_ptr2.
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_ptr_w2
        FUNCTION libxsmm_ptr_w2(a)
          INTEGER(C_SHORT), INTENT(IN) :: a(:,:)
          TYPE(C_PTR) :: libxsmm_ptr_w2
          IF (ALL(0.LT.SHAPE(a))) THEN
            libxsmm_ptr_w2 = libxsmm_ptr_w0(a(LBOUND(a,1),LBOUND(a,2)))
          ELSE
            libxsmm_ptr_w2 = C_NULL_PTR
          END IF
        END FUNCTION

        !> Determines the C-address of the given scalar.
        !> This overload belongs to libxsmm_ptr0.
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_ptr_j0
        FUNCTION libxsmm_ptr_j0(a)
          INTEGER(C_INT8_T), INTENT(IN), TARGET :: a
          INTEGER(C_INT8_T), POINTER :: fptr
          TYPE(C_PTR) :: libxsmm_ptr_j0
          fptr => a; libxsmm_ptr_j0 = C_LOC(fptr)
        END FUNCTION

        !> Determines the C-address of the given array.
        !> This overload belongs to libxsmm_ptr1.
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_ptr_j1
        FUNCTION libxsmm_ptr_j1(a)
          INTEGER(C_INT8_T), INTENT(IN) :: a(:)
          TYPE(C_PTR) :: libxsmm_ptr_j1
          IF (0.LT.SIZE(a)) THEN
            libxsmm_ptr_j1 = libxsmm_ptr_j0(a(LBOUND(a,1)))
          ELSE
            libxsmm_ptr_j1 = C_NULL_PTR
          END IF
        END FUNCTION

        !> Determines the C-address of the given 2d-array.
        !> This overload belongs to libxsmm_ptr2.
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_ptr_j2
        FUNCTION libxsmm_ptr_j2(a)
          INTEGER(C_INT8_T), INTENT(IN) :: a(:,:)
          TYPE(C_PTR) :: libxsmm_ptr_j2
          IF (ALL(0.LT.SHAPE(a))) THEN
            libxsmm_ptr_j2 = libxsmm_ptr_j0(a(LBOUND(a,1),LBOUND(a,2)))
          ELSE
            libxsmm_ptr_j2 = C_NULL_PTR
          END IF
        END FUNCTION

        !> Determines the C-address of the given scalar.
        !> This overload belongs to libxsmm_ptr0.
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_ptr_b0
        FUNCTION libxsmm_ptr_b0(a)
          CHARACTER(C_CHAR), INTENT(IN), TARGET :: a
          CHARACTER(C_CHAR), POINTER :: fptr
          TYPE(C_PTR) :: libxsmm_ptr_b0
          fptr => a; libxsmm_ptr_b0 = C_LOC(fptr)
        END FUNCTION

        !> Determines the C-address of the given array.
        !> This overload belongs to libxsmm_ptr1.
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_ptr_b1
        FUNCTION libxsmm_ptr_b1(a)
          CHARACTER(C_CHAR), INTENT(IN) :: a(:)
          TYPE(C_PTR) :: libxsmm_ptr_b1
          IF (0.LT.SIZE(a)) THEN
            libxsmm_ptr_b1 = libxsmm_ptr_b0(a(LBOUND(a,1)))
          ELSE
            libxsmm_ptr_b1 = C_NULL_PTR
          END IF
        END FUNCTION

        !> Determines the C-address of the given 2d-array.
        !> This overload belongs to libxsmm_ptr2.
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_ptr_b2
        FUNCTION libxsmm_ptr_b2(a)
          CHARACTER(C_CHAR), INTENT(IN) :: a(:,:)
          TYPE(C_PTR) :: libxsmm_ptr_b2
          IF (ALL(0.LT.SHAPE(a))) THEN
            libxsmm_ptr_b2 = libxsmm_ptr_b0(a(LBOUND(a,1),LBOUND(a,2)))
          ELSE
            libxsmm_ptr_b2 = C_NULL_PTR
          END IF
        END FUNCTION

        !> Determines the C-address of the given scalar.
        !> This overload belongs to libxsmm_ptr0.
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_ptr_l0
        FUNCTION libxsmm_ptr_l0(a)
          INTEGER(C_LONG_LONG), INTENT(IN), TARGET :: a
          INTEGER(C_LONG_LONG), POINTER :: fptr
          TYPE(C_PTR) :: libxsmm_ptr_l0
          fptr => a; libxsmm_ptr_l0 = C_LOC(fptr)
        END FUNCTION

        !> Determines the C-address of the given array.
        !> This overload belongs to libxsmm_ptr1.
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_ptr_l1
        FUNCTION libxsmm_ptr_l1(a)
          INTEGER(C_LONG_LONG), INTENT(IN) :: a(:)
          TYPE(C_PTR) :: libxsmm_ptr_l1
          IF (0.LT.SIZE(a)) THEN
            libxsmm_ptr_l1 = libxsmm_ptr_l0(a(LBOUND(a,1)))
          ELSE
            libxsmm_ptr_l1 = C_NULL_PTR
          END IF
        END FUNCTION

        !> Determines the C-address of the given 2d-array.
        !> This overload belongs to libxsmm_ptr2.
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_ptr_l2
        FUNCTION libxsmm_ptr_l2(a)
          INTEGER(C_LONG_LONG), INTENT(IN) :: a(:,:)
          TYPE(C_PTR) :: libxsmm_ptr_l2
          IF (ALL(0.LT.SHAPE(a))) THEN
            libxsmm_ptr_l2 = libxsmm_ptr_l0(a(LBOUND(a,1),LBOUND(a,2)))
          ELSE
            libxsmm_ptr_l2 = C_NULL_PTR
          END IF
        END FUNCTION

        !> Deallocate JIT'ted code created by libxsmm_create routines. To
        !> unregister code generated with libxsmm_dispatch is unnecessary.
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_release_dmmkernel
        SUBROUTINE libxsmm_release_dmmkernel(kernel)
          TYPE(LIBXSMM_DMMFUNCTION), INTENT(IN) :: kernel
          CALL libxsmm_release_kernel(kernel%handle)
        END SUBROUTINE

        !> Deallocate JIT'ted code created by libxsmm_create routines. To
        !> unregister code generated with libxsmm_dispatch is unnecessary.
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_release_smmkernel
        SUBROUTINE libxsmm_release_smmkernel(kernel)
          TYPE(LIBXSMM_SMMFUNCTION), INTENT(IN) :: kernel
          CALL libxsmm_release_kernel(kernel%handle)
        END SUBROUTINE

        !> Deallocate JIT'ted code created by libxsmm_create routines. To
        !> unregister code generated with libxsmm_dispatch is unnecessary.
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_release_wimmkernel
        SUBROUTINE libxsmm_release_wimmkernel(kernel)
          TYPE(LIBXSMM_WIMMFUNCTION), INTENT(IN) :: kernel
          CALL libxsmm_release_kernel(kernel%handle)
        END SUBROUTINE

        !> Query or JIT-generate an SMM-kernel (double-precision).
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_dmmdispatch
        SUBROUTINE libxsmm_dmmdispatch(kernel,                          &
     &  m, n, k, lda, ldb, ldc, alpha, beta, flags, prefetch)
          TYPE(LIBXSMM_DMMFUNCTION), INTENT(OUT) :: kernel
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: m, n, k
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

        !> Query or JIT-generate an SMM-kernel (single-precision).
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_smmdispatch
        SUBROUTINE libxsmm_smmdispatch(kernel,                          &
     &  m, n, k, lda, ldb, ldc, alpha, beta, flags, prefetch)
          TYPE(LIBXSMM_SMMFUNCTION), INTENT(OUT) :: kernel
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: m, n, k
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

        !> Query or JIT-generate an SMM-kernel (low-precision, int-accumulate).
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_wimmdispatch
        SUBROUTINE libxsmm_wimmdispatch(kernel,                         &
     &  m, n, k, lda, ldb, ldc, alpha, beta, flags, prefetch)
          TYPE(LIBXSMM_WIMMFUNCTION), INTENT(OUT) :: kernel
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: m, n, k
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

        !> Checks if the given kernel was generated. JIT code is guaranteed
        !> to be generated if JIT support was enabled at build-time of the
        !> library (default). This overload belongs to libxsmm_(mm)available.
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_dmmavailable
        LOGICAL FUNCTION libxsmm_dmmavailable(kernel)
          TYPE(LIBXSMM_DMMFUNCTION), INTENT(IN) :: kernel
          libxsmm_dmmavailable = C_ASSOCIATED(kernel%handle)
        END FUNCTION

        !> Checks if the given kernel was generated. JIT code is guaranteed
        !> to be generated if JIT support was enabled at build-time of the
        !> library (default). This overload belongs to libxsmm_(mm)available.
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_smmavailable
        LOGICAL FUNCTION libxsmm_smmavailable(kernel)
          TYPE(LIBXSMM_SMMFUNCTION), INTENT(IN) :: kernel
          libxsmm_smmavailable = C_ASSOCIATED(kernel%handle)
        END FUNCTION

        !> Checks if the given kernel was generated. JIT code is guaranteed
        !> to be generated if JIT support was enabled at build-time of the
        !> library (default). This overload belongs to libxsmm_(mm)available.
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_wimmavailable
        LOGICAL FUNCTION libxsmm_wimmavailable(kernel)
          TYPE(LIBXSMM_WIMMFUNCTION), INTENT(IN) :: kernel
          libxsmm_wimmavailable = C_ASSOCIATED(kernel%handle)
        END FUNCTION

        !> Calls the kernel for the given arguments. Alternatively,
        !> PROCPOINTER can be used as shown by the inner comments
        !> of this routine (LIBXSMM_FUNCTION3/6, etc.). The
        !> libxsmm_xmmcall routines can be used in FORTRAN77.
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_dmmcall
        SUBROUTINE libxsmm_dmmcall(kernel, a,b,c, pa,pb,pc)
          TYPE(LIBXSMM_DMMFUNCTION), INTENT(IN) :: kernel
          REAL(C_DOUBLE), INTENT(IN), TARGET :: a(*), b(*)
          REAL(C_DOUBLE), INTENT(INOUT), TARGET :: c(*)
          REAL(C_DOUBLE), INTENT(IN), OPTIONAL, TARGET :: pa(*)
          REAL(C_DOUBLE), INTENT(IN), OPTIONAL, TARGET :: pb(*)
          REAL(C_DOUBLE), INTENT(IN), OPTIONAL, TARGET :: pc(*)
          ! PROCEDURE(LIBXSMM_FUNCTION6), POINTER :: xmm6
          ! PROCEDURE(LIBXSMM_FUNCTION3), POINTER :: xmm3
          ! use .OR. instead of .AND. to avoid full check
          IF (PRESENT(pa).OR.PRESENT(pb).OR.PRESENT(pc)) THEN
            ! CALL C_F_PROCPOINTER(kernel%handle, xmm6)
            ! CALL xmm6(
            CALL libxsmm_xmmcall_prf(kernel%handle,                     &
     &        C_LOC(a), C_LOC(b), C_LOC(c),                             &
     &        C_LOC(pa), C_LOC(pb), C_LOC(pc))
          ELSE
            ! CALL C_F_PROCPOINTER(kernel%handle, xmm3)
            ! CALL xmm3(
            CALL libxsmm_xmmcall_abc(kernel%handle,                     &
     &        C_LOC(a), C_LOC(b), C_LOC(c))
          END IF
        END SUBROUTINE

        !> Calls the kernel for the given arguments. Alternatively,
        !> PROCPOINTER can be used as shown by the inner comments
        !> of this routine (LIBXSMM_FUNCTION3/6, etc.). The
        !> libxsmm_xmmcall routines can be used in FORTRAN77.
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_smmcall
        SUBROUTINE libxsmm_smmcall(kernel, a,b,c, pa,pb,pc)
          TYPE(LIBXSMM_SMMFUNCTION), INTENT(IN) :: kernel
          REAL(C_FLOAT), INTENT(IN), TARGET :: a(*), b(*)
          REAL(C_FLOAT), INTENT(INOUT), TARGET :: c(*)
          REAL(C_FLOAT), INTENT(IN), OPTIONAL, TARGET :: pa(*)
          REAL(C_FLOAT), INTENT(IN), OPTIONAL, TARGET :: pb(*)
          REAL(C_FLOAT), INTENT(IN), OPTIONAL, TARGET :: pc(*)
          ! PROCEDURE(LIBXSMM_FUNCTION6), POINTER :: xmm6
          ! PROCEDURE(LIBXSMM_FUNCTION3), POINTER :: xmm3
          ! use .OR. instead of .AND. to avoid full check
          IF (PRESENT(pa).OR.PRESENT(pb).OR.PRESENT(pc)) THEN
            ! CALL C_F_PROCPOINTER(kernel%handle, xmm6)
            ! CALL xmm6(
            CALL libxsmm_xmmcall_prf(kernel%handle,                     &
     &        C_LOC(a), C_LOC(b), C_LOC(c),                             &
     &        C_LOC(pa), C_LOC(pb), C_LOC(pc))
          ELSE
            ! CALL C_F_PROCPOINTER(kernel%handle, xmm3)
            ! CALL xmm3(
            CALL libxsmm_xmmcall_abc(kernel%handle,                     &
     &        C_LOC(a), C_LOC(b), C_LOC(c))
          END IF
        END SUBROUTINE

        !> Calls the kernel for the given arguments. Alternatively,
        !> PROCPOINTER can be used as shown by the inner comments
        !> of this routine (LIBXSMM_FUNCTION3/6, etc.). The
        !> libxsmm_xmmcall routines can be used in FORTRAN77.
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_wimmcall
        SUBROUTINE libxsmm_wimmcall(kernel, a,b,c, pa,pb,pc)
          TYPE(LIBXSMM_WIMMFUNCTION), INTENT(IN) :: kernel
          INTEGER(C_SHORT), INTENT(IN),  TARGET :: a(*), b(*)
          INTEGER(C_INT), INTENT(INOUT), TARGET :: c(*)
          INTEGER(C_SHORT), INTENT(IN), OPTIONAL, TARGET :: pa(*)
          INTEGER(C_SHORT), INTENT(IN), OPTIONAL, TARGET :: pb(*)
          INTEGER(C_INT),   INTENT(IN), OPTIONAL, TARGET :: pc(*)
          ! PROCEDURE(LIBXSMM_FUNCTION6), POINTER :: xmm6
          ! PROCEDURE(LIBXSMM_FUNCTION3), POINTER :: xmm3
          ! use .OR. instead of .AND. to avoid full check
          IF (PRESENT(pa).OR.PRESENT(pb).OR.PRESENT(pc)) THEN
            ! CALL C_F_PROCPOINTER(kernel%handle, xmm6)
            ! CALL xmm6(
            CALL libxsmm_xmmcall_prf(kernel%handle,                     &
     &        C_LOC(a), C_LOC(b), C_LOC(c),                             &
     &        C_LOC(pa), C_LOC(pb), C_LOC(pc))
          ELSE
            ! CALL C_F_PROCPOINTER(kernel%handle, xmm3)
            ! CALL xmm3(
            CALL libxsmm_xmmcall_abc(kernel%handle,                     &
     &        C_LOC(a), C_LOC(b), C_LOC(c))
          END IF
        END SUBROUTINE

        !> Calls the kernel for the given arguments. Alternatively,
        !> PROCPOINTER can be used as shown by the inner comments
        !> of this routine (LIBXSMM_FUNCTION3/6, etc.). The
        !> libxsmm_xmmcall routines can be used in FORTRAN77.
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_dmmcall_abc
        SUBROUTINE libxsmm_dmmcall_abc(kernel, a, b, c)
          TYPE(LIBXSMM_DMMFUNCTION), INTENT(IN) :: kernel
          TYPE(C_PTR), INTENT(IN) :: a, b, c
          ! PROCEDURE(LIBXSMM_FUNCTION3), POINTER :: xmm
          ! CALL C_F_PROCPOINTER(kernel%handle, xmm)
          ! CALL xmm(a, b, c)
          CALL libxsmm_xmmcall_abc(kernel%handle, a, b, c)
        END SUBROUTINE

        !> Calls the kernel for the given arguments. Alternatively,
        !> PROCPOINTER can be used as shown by the inner comments
        !> of this routine (LIBXSMM_FUNCTION3/6, etc.). The
        !> libxsmm_xmmcall routines can be used in FORTRAN77.
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_smmcall_abc
        SUBROUTINE libxsmm_smmcall_abc(kernel, a, b, c)
          TYPE(LIBXSMM_SMMFUNCTION), INTENT(IN) :: kernel
          TYPE(C_PTR), INTENT(IN) :: a, b, c
          ! PROCEDURE(LIBXSMM_FUNCTION3), POINTER :: xmm
          ! CALL C_F_PROCPOINTER(kernel%handle, xmm)
          ! CALL xmm(a, b, c)
          CALL libxsmm_xmmcall_abc(kernel%handle, a, b, c)
        END SUBROUTINE

        !> Calls the kernel for the given arguments. Alternatively,
        !> PROCPOINTER can be used as shown by the inner comments
        !> of this routine (LIBXSMM_FUNCTION3/6, etc.). The
        !> libxsmm_xmmcall routines can be used in FORTRAN77.
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_wimmcall_abc
        SUBROUTINE libxsmm_wimmcall_abc(kernel, a, b, c)
          TYPE(LIBXSMM_WIMMFUNCTION), INTENT(IN) :: kernel
          TYPE(C_PTR), INTENT(IN) :: a, b, c
          ! PROCEDURE(LIBXSMM_FUNCTION3), POINTER :: xmm
          ! CALL C_F_PROCPOINTER(kernel%handle, xmm)
          ! CALL xmm(a, b, c)
          CALL libxsmm_xmmcall_abc(kernel%handle, a, b, c)
        END SUBROUTINE

        !> Calls the kernel for the given arguments. Alternatively,
        !> PROCPOINTER can be used as shown by the inner comments
        !> of this routine (LIBXSMM_FUNCTION3/6, etc.). The
        !> libxsmm_xmmcall routines can be used in FORTRAN77.
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_dmmcall_prf
        SUBROUTINE libxsmm_dmmcall_prf(kernel, a,b,c, pa,pb,pc)
          TYPE(LIBXSMM_DMMFUNCTION), INTENT(IN) :: kernel
          TYPE(C_PTR), INTENT(IN) :: a, b, c, pa, pb, pc
          ! PROCEDURE(LIBXSMM_FUNCTION6), POINTER :: xmm
          ! CALL C_F_PROCPOINTER(kernel%handle, xmm)
          ! CALL xmm(a, b, c, pa, pb, pc)
          CALL libxsmm_xmmcall_prf(kernel%handle, a, b, c, pa, pb, pc)
        END SUBROUTINE

        !> Calls the kernel for the given arguments. Alternatively,
        !> PROCPOINTER can be used as shown by the inner comments
        !> of this routine (LIBXSMM_FUNCTION3/6, etc.). The
        !> libxsmm_xmmcall routines can be used in FORTRAN77.
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_smmcall_prf
        SUBROUTINE libxsmm_smmcall_prf(kernel, a,b,c, pa,pb,pc)
          TYPE(LIBXSMM_SMMFUNCTION), INTENT(IN) :: kernel
          TYPE(C_PTR), INTENT(IN) :: a, b, c, pa, pb, pc
          ! PROCEDURE(LIBXSMM_FUNCTION6), POINTER :: xmm
          ! CALL C_F_PROCPOINTER(kernel%handle, xmm)
          ! CALL xmm(a, b, c, pa, pb, pc)
          CALL libxsmm_xmmcall_prf(kernel%handle, a, b, c, pa, pb, pc)
        END SUBROUTINE

        !> Calls the kernel for the given arguments. Alternatively,
        !> PROCPOINTER can be used as shown by the inner comments
        !> of this routine (LIBXSMM_FUNCTION3/6, etc.). The
        !> libxsmm_xmmcall routines can be used in FORTRAN77.
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_wimmcall_prf
        SUBROUTINE libxsmm_wimmcall_prf(kernel, a,b,c, pa,pb,pc)
          TYPE(LIBXSMM_WIMMFUNCTION), INTENT(IN) :: kernel
          TYPE(C_PTR), INTENT(IN) :: a, b, c, pa, pb, pc
          ! PROCEDURE(LIBXSMM_FUNCTION6), POINTER :: xmm
          ! CALL C_F_PROCPOINTER(kernel%handle, xmm)
          ! CALL xmm(a, b, c, pa, pb, pc)
          CALL libxsmm_xmmcall_prf(kernel%handle, a, b, c, pa, pb, pc)
        END SUBROUTINE

        !> Auto-dispatched general dense MM (double-precision).
        !> This overload belongs to libxsmm_(d)gemm.
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_dgemm0
        PURE SUBROUTINE libxsmm_dgemm0(transa, transb, m, n, k,         &
     &  alpha, a, lda, b, ldb, beta, c, ldc)
          CHARACTER, INTENT(IN), OPTIONAL :: transa, transb
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: m, n, k
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN), OPTIONAL :: lda
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN), OPTIONAL :: ldb
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN), OPTIONAL :: ldc
          REAL(C_DOUBLE), INTENT(IN), OPTIONAL :: alpha, beta
          REAL(C_DOUBLE), INTENT(IN) :: a, b
          REAL(C_DOUBLE), INTENT(INOUT) :: c
          !DIR$ ATTRIBUTES OFFLOAD:MIC :: internal_gemm
          INTERFACE
            PURE SUBROUTINE internal_gemm(transa, transb, m, n, k,      &
     &      alpha, a, lda, b, ldb, beta, c, ldc)                        &
     &      BIND(C, NAME="libxsmm_dgemm_")
              IMPORT C_CHAR, C_DOUBLE, LIBXSMM_BLASINT_KIND
              CHARACTER(C_CHAR), INTENT(IN) :: transa, transb
              INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: m, n, k
              INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: lda
              INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: ldb
              INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: ldc
              REAL(C_DOUBLE), INTENT(IN) :: alpha, beta
              REAL(C_DOUBLE), INTENT(IN) :: a, b
              REAL(C_DOUBLE), INTENT(INOUT) :: c
            END SUBROUTINE
          END INTERFACE
          CALL internal_gemm(transa, transb, m, n, k,                   &
     &      alpha, a, lda, b, ldb, beta, c, ldc)
        END SUBROUTINE

        !> Auto-dispatched general dense MM (double-precision).
        !> This overload belongs to libxsmm_(d)gemm.
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_dgemm1
        PURE SUBROUTINE libxsmm_dgemm1(transa, transb, m, n, k,         &
     &  alpha, a, lda, b, ldb, beta, c, ldc)
          CHARACTER, INTENT(IN), OPTIONAL :: transa, transb
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: m, n, k
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN), OPTIONAL :: lda
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN), OPTIONAL :: ldb
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN), OPTIONAL :: ldc
          REAL(C_DOUBLE), INTENT(IN), OPTIONAL :: alpha, beta
          REAL(C_DOUBLE), INTENT(IN) :: a(:), b(:)
          REAL(C_DOUBLE), INTENT(INOUT) :: c(:)
          IF ((0.LT.m).AND.(0.LT.n).AND.(0.LT.k)) THEN
            CALL libxsmm_dgemm0(transa, transb, m, n, k,                &
     &        alpha, a(LBOUND(a,1)), lda,                               &
     &               b(LBOUND(b,1)), ldb,                               &
     &         beta, c(LBOUND(c,1)), ldc)
          END IF
        END SUBROUTINE

        !> Auto-dispatched general dense MM (double-precision).
        !> This overload belongs to libxsmm_(d)gemm.
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_dgemm2
        PURE SUBROUTINE libxsmm_dgemm2(transa, transb, m, n, k,         &
     &  a, b, c, alpha, beta)
          CHARACTER, INTENT(IN), OPTIONAL :: transa, transb
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: m, n, k
          REAL(C_DOUBLE), INTENT(IN), OPTIONAL :: alpha, beta
          REAL(C_DOUBLE), INTENT(IN)    :: a(m,*), b(k,*)
          REAL(C_DOUBLE), INTENT(INOUT) :: c(m,*)
          IF ((0.LT.m).AND.(0.LT.n).AND.(0.LT.k)) THEN
            CALL libxsmm_dgemm0(transa, transb, m, n, k,                &
     &        alpha, a(LBOUND(a,1),LBOUND(a,2)), m,                     &
     &               b(LBOUND(b,1),LBOUND(b,2)), k,                     &
     &         beta, c(LBOUND(c,1),LBOUND(c,2)), m)
          END IF
        END SUBROUTINE

        !> Auto-dispatched general dense MM (double-precision).
        !> This overload belongs to libxsmm_(d)gemm.
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_dgemm3
        PURE SUBROUTINE libxsmm_dgemm3(transa, transb, m, n, k,         &
     &  alpha, a, lda, b, ldb, beta, c, ldc)
          CHARACTER, INTENT(IN), OPTIONAL :: transa, transb
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: m, n, k
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: lda, ldb, ldc
          REAL(C_DOUBLE), INTENT(IN), OPTIONAL :: alpha, beta
          REAL(C_DOUBLE), INTENT(IN)    :: a(lda,*), b(ldb,*)
          REAL(C_DOUBLE), INTENT(INOUT) :: c(ldc,*)
          IF ((0.LT.m).AND.(0.LT.n).AND.(0.LT.k)) THEN
            CALL libxsmm_dgemm0(transa, transb, m, n, k,                &
     &        alpha, a(LBOUND(a,1),LBOUND(a,2)), lda,                   &
     &               b(LBOUND(b,1),LBOUND(b,2)), ldb,                   &
     &         beta, c(LBOUND(c,1),LBOUND(c,2)), ldc)
          END IF
        END SUBROUTINE

        !> Auto-dispatched general dense MM (single-precision).
        !> This overload belongs to libxsmm_(s)gemm.
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_sgemm0
        PURE SUBROUTINE libxsmm_sgemm0(transa, transb, m, n, k,         &
     &  alpha, a, lda, b, ldb, beta, c, ldc)
          CHARACTER, INTENT(IN), OPTIONAL :: transa, transb
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: m, n, k
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN), OPTIONAL :: lda
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN), OPTIONAL :: ldb
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN), OPTIONAL :: ldc
          REAL(C_FLOAT), INTENT(IN), OPTIONAL :: alpha, beta
          REAL(C_FLOAT), INTENT(IN) :: a, b
          REAL(C_FLOAT), INTENT(INOUT) :: c
          !DIR$ ATTRIBUTES OFFLOAD:MIC :: internal_gemm
          INTERFACE
            PURE SUBROUTINE internal_gemm(transa, transb, m, n, k,      &
     &      alpha, a, lda, b, ldb, beta, c, ldc)                        &
     &      BIND(C, NAME="libxsmm_sgemm_")
              IMPORT C_CHAR, C_FLOAT, LIBXSMM_BLASINT_KIND
              CHARACTER(C_CHAR), INTENT(IN) :: transa, transb
              INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: m, n, k
              INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: lda
              INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: ldb
              INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: ldc
              REAL(C_FLOAT), INTENT(IN) :: alpha, beta
              REAL(C_FLOAT), INTENT(IN) :: a, b
              REAL(C_FLOAT), INTENT(INOUT) :: c
            END SUBROUTINE
          END INTERFACE
          CALL internal_gemm(transa, transb, m, n, k,                   &
     &      alpha, a, lda, b, ldb, beta, c, ldc)
        END SUBROUTINE

        !> Auto-dispatched general dense MM (single-precision).
        !> This overload belongs to libxsmm_(s)gemm.
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_sgemm1
        PURE SUBROUTINE libxsmm_sgemm1(transa, transb, m, n, k,         &
     &  alpha, a, lda, b, ldb, beta, c, ldc)
          CHARACTER, INTENT(IN), OPTIONAL :: transa, transb
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: m, n, k
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN), OPTIONAL :: lda
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN), OPTIONAL :: ldb
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN), OPTIONAL :: ldc
          REAL(C_FLOAT), INTENT(IN), OPTIONAL :: alpha, beta
          REAL(C_FLOAT), INTENT(IN) :: a(:), b(:)
          REAL(C_FLOAT), INTENT(INOUT) :: c(:)
          IF ((0.LT.m).AND.(0.LT.n).AND.(0.LT.k)) THEN
            CALL libxsmm_sgemm0(transa, transb, m, n, k,                &
     &        alpha, a(LBOUND(a,1)), lda,                               &
     &               b(LBOUND(b,1)), ldb,                               &
     &         beta, c(LBOUND(c,1)), ldc)
          END IF
        END SUBROUTINE

        !> Auto-dispatched general dense MM (single-precision).
        !> This overload belongs to libxsmm_(s)gemm.
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_sgemm2
        PURE SUBROUTINE libxsmm_sgemm2(transa, transb, m, n, k,         &
     &  a, b, c, alpha, beta)
          CHARACTER, INTENT(IN), OPTIONAL :: transa, transb
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: m, n, k
          REAL(C_FLOAT), INTENT(IN), OPTIONAL :: alpha, beta
          REAL(C_FLOAT), INTENT(IN)    :: a(m,*), b(k,*)
          REAL(C_FLOAT), INTENT(INOUT) :: c(m,*)
          IF ((0.LT.m).AND.(0.LT.n).AND.(0.LT.k)) THEN
            CALL libxsmm_sgemm0(transa, transb, m, n, k,                &
     &        alpha, a(LBOUND(a,1),LBOUND(a,2)), m,                     &
     &               b(LBOUND(b,1),LBOUND(b,2)), k,                     &
     &         beta, c(LBOUND(c,1),LBOUND(c,2)), m)
          END IF
        END SUBROUTINE

        !> Auto-dispatched general dense MM (single-precision).
        !> This overload belongs to libxsmm_(s)gemm.
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_sgemm3
        PURE SUBROUTINE libxsmm_sgemm3(transa, transb, m, n, k,         &
     &  alpha, a, lda, b, ldb, beta, c, ldc)
          CHARACTER, INTENT(IN), OPTIONAL :: transa, transb
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: m, n, k
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: lda, ldb, ldc
          REAL(C_FLOAT), INTENT(IN), OPTIONAL :: alpha, beta
          REAL(C_FLOAT), INTENT(IN)    :: a(lda,*), b(ldb,*)
          REAL(C_FLOAT), INTENT(INOUT) :: c(ldc,*)
          IF ((0.LT.m).AND.(0.LT.n).AND.(0.LT.k)) THEN
            CALL libxsmm_sgemm0(transa, transb, m, n, k,                &
     &        alpha, a(LBOUND(a,1),LBOUND(a,2)), lda,                   &
     &               b(LBOUND(b,1),LBOUND(b,2)), ldb,                   &
     &         beta, c(LBOUND(c,1),LBOUND(c,2)), ldc)
          END IF
        END SUBROUTINE

        !> Auto-dispatched general dense MM (low-precision, int-accumulate).
        !> This overload belongs to libxsmm_(wi)gemm.
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_wigemm0
        PURE SUBROUTINE libxsmm_wigemm0(transa, transb, m, n, k,        &
     &  alpha, a, lda, b, ldb, beta, c, ldc)
          CHARACTER, INTENT(IN), OPTIONAL :: transa, transb
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: m, n, k
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN), OPTIONAL :: lda
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN), OPTIONAL :: ldb
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN), OPTIONAL :: ldc
          INTEGER(C_INT), INTENT(IN), OPTIONAL :: alpha, beta
          INTEGER(C_SHORT), INTENT(IN) :: a, b
          INTEGER(C_INT), INTENT(INOUT) :: c
          !DIR$ ATTRIBUTES OFFLOAD:MIC :: internal_gemm
          INTERFACE
            PURE SUBROUTINE internal_gemm(transa, transb, m, n, k,      &
     &      alpha, a, lda, b, ldb, beta, c, ldc)                        &
     &      BIND(C, NAME="libxsmm_wigemm_")
              IMPORT C_CHAR, C_SHORT, C_INT, LIBXSMM_BLASINT_KIND
              CHARACTER(C_CHAR), INTENT(IN) :: transa, transb
              INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: m, n, k
              INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: lda
              INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: ldb
              INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: ldc
              INTEGER(C_INT), INTENT(IN) :: alpha, beta
              INTEGER(C_SHORT), INTENT(IN) :: a, b
              INTEGER(C_INT), INTENT(INOUT) :: c
            END SUBROUTINE
          END INTERFACE
          CALL internal_gemm(transa, transb, m, n, k,                   &
     &      alpha, a, lda, b, ldb, beta, c, ldc)
        END SUBROUTINE

        !> Auto-dispatched general dense MM (low-precision, int-accumulate).
        !> This overload belongs to libxsmm_(wi)gemm.
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_wigemm1
        PURE SUBROUTINE libxsmm_wigemm1(transa, transb, m, n, k,        &
     &  alpha, a, lda, b, ldb, beta, c, ldc)
          CHARACTER, INTENT(IN), OPTIONAL :: transa, transb
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: m, n, k
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN), OPTIONAL :: lda
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN), OPTIONAL :: ldb
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN), OPTIONAL :: ldc
          INTEGER(C_INT), INTENT(IN), OPTIONAL :: alpha, beta
          INTEGER(C_SHORT), INTENT(IN) :: a(:), b(:)
          INTEGER(C_INT), INTENT(INOUT) :: c(:)
          IF ((0.LT.m).AND.(0.LT.n).AND.(0.LT.k)) THEN
            CALL libxsmm_wigemm0(transa, transb, m, n, k,               &
     &        alpha, a(LBOUND(a,1)), lda,                               &
     &               b(LBOUND(b,1)), ldb,                               &
     &         beta, c(LBOUND(c,1)), ldc)
          END IF
        END SUBROUTINE

        !> Auto-dispatched general dense MM (low-precision, int-accumulate).
        !> This overload belongs to libxsmm_(wi)gemm.
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_wigemm2
        PURE SUBROUTINE libxsmm_wigemm2(transa, transb, m, n, k,        &
     &  a, b, c, alpha, beta)
          CHARACTER, INTENT(IN), OPTIONAL :: transa, transb
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: m, n, k
          INTEGER(C_INT), INTENT(IN), OPTIONAL :: alpha, beta
          INTEGER(C_SHORT), INTENT(IN)  :: a(m,*), b(k,*)
          INTEGER(C_INT), INTENT(INOUT) :: c(m,*)
          IF ((0.LT.m).AND.(0.LT.n).AND.(0.LT.k)) THEN
            CALL libxsmm_wigemm0(transa, transb, m, n, k,               &
     &        alpha, a(LBOUND(a,1),LBOUND(a,2)), m,                     &
     &               b(LBOUND(b,1),LBOUND(b,2)), k,                     &
     &         beta, c(LBOUND(c,1),LBOUND(c,2)), m)
          END IF
        END SUBROUTINE

        !> Auto-dispatched general dense MM (low-precision, int-accumulate).
        !> This overload belongs to libxsmm_(wi)gemm.
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_wigemm3
        PURE SUBROUTINE libxsmm_wigemm3(transa, transb, m, n, k,        &
     &  alpha, a, lda, b, ldb, beta, c, ldc)
          CHARACTER, INTENT(IN), OPTIONAL :: transa, transb
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: m, n, k
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: lda, ldb, ldc
          INTEGER(C_INT), INTENT(IN), OPTIONAL :: alpha, beta
          INTEGER(C_SHORT), INTENT(IN)  :: a(lda,*), b(ldb,*)
          INTEGER(C_INT), INTENT(INOUT) :: c(ldc,*)
          IF ((0.LT.m).AND.(0.LT.n).AND.(0.LT.k)) THEN
            CALL libxsmm_wigemm0(transa, transb, m, n, k,               &
     &        alpha, a(LBOUND(a,1),LBOUND(a,2)), lda,                   &
     &               b(LBOUND(b,1),LBOUND(b,2)), ldb,                   &
     &         beta, c(LBOUND(c,1),LBOUND(c,2)), ldc)
          END IF
        END SUBROUTINE

        !> Re-exposes BLAS based GEMM routine with an interfaces similar to
        !> libxsmm_(d)gemm. This overload belongs to libxsmm_blas_(d)gemm.
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_blas_dgemm0
        PURE SUBROUTINE libxsmm_blas_dgemm0(transa, transb, m, n, k,    &
     &  alpha, a, lda, b, ldb, beta, c, ldc)
          CHARACTER, INTENT(IN), OPTIONAL :: transa, transb
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: m, n, k
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN), OPTIONAL :: lda
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN), OPTIONAL :: ldb
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN), OPTIONAL :: ldc
          REAL(C_DOUBLE), INTENT(IN), OPTIONAL :: alpha, beta
          REAL(C_DOUBLE), INTENT(IN) :: a, b
          REAL(C_DOUBLE), INTENT(INOUT) :: c
          !DIR$ ATTRIBUTES OFFLOAD:MIC :: internal_gemm
          INTERFACE
            PURE SUBROUTINE internal_gemm(transa, transb, m, n, k,      &
     &      alpha, a, lda, b, ldb, beta, c, ldc)                        &
     &      BIND(C, NAME="libxsmm_blas_dgemm_")
              IMPORT C_CHAR, C_DOUBLE, LIBXSMM_BLASINT_KIND
              CHARACTER(C_CHAR), INTENT(IN) :: transa, transb
              INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: m, n, k
              INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: lda
              INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: ldb
              INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: ldc
              REAL(C_DOUBLE), INTENT(IN) :: alpha, beta
              REAL(C_DOUBLE), INTENT(IN) :: a, b
              REAL(C_DOUBLE), INTENT(INOUT) :: c
            END SUBROUTINE
          END INTERFACE
          CALL internal_gemm(transa, transb, m, n, k,                   &
     &      alpha, a, lda, b, ldb, beta, c, ldc)
        END SUBROUTINE

        !> Re-exposes BLAS based GEMM routine with an interfaces similar to
        !> libxsmm_(d)gemm. This overload belongs to libxsmm_blas_(d)gemm.
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_blas_dgemm1
        PURE SUBROUTINE libxsmm_blas_dgemm1(transa, transb, m, n, k,    &
     &  alpha, a, lda, b, ldb, beta, c, ldc)
          CHARACTER, INTENT(IN), OPTIONAL :: transa, transb
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: m, n, k
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN), OPTIONAL :: lda
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN), OPTIONAL :: ldb
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN), OPTIONAL :: ldc
          REAL(C_DOUBLE), INTENT(IN), OPTIONAL :: alpha, beta
          REAL(C_DOUBLE), INTENT(IN) :: a(:), b(:)
          REAL(C_DOUBLE), INTENT(INOUT) :: c(:)
          IF ((0.LT.m).AND.(0.LT.n).AND.(0.LT.k)) THEN
            CALL libxsmm_blas_dgemm0(transa, transb, m, n, k,           &
     &        alpha, a(LBOUND(a,1)), lda,                               &
     &               b(LBOUND(b,1)), ldb,                               &
     &         beta, c(LBOUND(c,1)), ldc)
          END IF
        END SUBROUTINE

        !> Re-exposes BLAS based GEMM routine with an interfaces similar to
        !> libxsmm_(d)gemm. This overload belongs to libxsmm_blas_(d)gemm.
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_blas_dgemm2
        PURE SUBROUTINE libxsmm_blas_dgemm2(transa, transb, m, n, k,    &
     &  a, b, c, alpha, beta)
          CHARACTER, INTENT(IN), OPTIONAL :: transa, transb
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: m, n, k
          REAL(C_DOUBLE), INTENT(IN), OPTIONAL :: alpha, beta
          REAL(C_DOUBLE), INTENT(IN)    :: a(m,*), b(k,*)
          REAL(C_DOUBLE), INTENT(INOUT) :: c(m,*)
          IF ((0.LT.m).AND.(0.LT.n).AND.(0.LT.k)) THEN
            CALL libxsmm_blas_dgemm0(transa, transb, m, n, k,           &
     &        alpha, a(LBOUND(a,1),LBOUND(a,2)), m,                     &
     &               b(LBOUND(b,1),LBOUND(b,2)), k,                     &
     &         beta, c(LBOUND(c,1),LBOUND(c,2)), m)
          END IF
        END SUBROUTINE

        !> Re-exposes BLAS based GEMM routine with an interfaces similar to
        !> libxsmm_(d)gemm. This overload belongs to libxsmm_blas_(d)gemm.
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_blas_dgemm3
        PURE SUBROUTINE libxsmm_blas_dgemm3(transa, transb, m, n, k,    &
     &  alpha, a, lda, b, ldb, beta, c, ldc)
          CHARACTER, INTENT(IN), OPTIONAL :: transa, transb
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: m, n, k
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: lda, ldb, ldc
          REAL(C_DOUBLE), INTENT(IN), OPTIONAL :: alpha, beta
          REAL(C_DOUBLE), INTENT(IN)    :: a(lda,*), b(ldb,*)
          REAL(C_DOUBLE), INTENT(INOUT) :: c(ldc,*)
          IF ((0.LT.m).AND.(0.LT.n).AND.(0.LT.k)) THEN
            CALL libxsmm_blas_dgemm0(transa, transb, m, n, k,           &
     &        alpha, a(LBOUND(a,1),LBOUND(a,2)), lda,                   &
     &               b(LBOUND(b,1),LBOUND(b,2)), ldb,                   &
     &         beta, c(LBOUND(c,1),LBOUND(c,2)), ldc)
          END IF
        END SUBROUTINE

        !> Re-exposes BLAS based GEMM routine with an interfaces similar to
        !> libxsmm_(s)gemm. This overload belongs to libxsmm_blas_(s)gemm.
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_blas_sgemm0
        PURE SUBROUTINE libxsmm_blas_sgemm0(transa, transb, m, n, k,    &
     &  alpha, a, lda, b, ldb, beta, c, ldc)
          CHARACTER, INTENT(IN), OPTIONAL :: transa, transb
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: m, n, k
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN), OPTIONAL :: lda
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN), OPTIONAL :: ldb
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN), OPTIONAL :: ldc
          REAL(C_FLOAT), INTENT(IN), OPTIONAL :: alpha, beta
          REAL(C_FLOAT), INTENT(IN) :: a, b
          REAL(C_FLOAT), INTENT(INOUT) :: c
          !DIR$ ATTRIBUTES OFFLOAD:MIC :: internal_gemm
          INTERFACE
            PURE SUBROUTINE internal_gemm(transa, transb, m, n, k,      &
     &      alpha, a, lda, b, ldb, beta, c, ldc)                        &
     &      BIND(C, NAME="libxsmm_blas_sgemm_")
              IMPORT C_CHAR, C_FLOAT, LIBXSMM_BLASINT_KIND
              CHARACTER(C_CHAR), INTENT(IN) :: transa, transb
              INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: m, n, k
              INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: lda
              INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: ldb
              INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: ldc
              REAL(C_FLOAT), INTENT(IN) :: alpha, beta
              REAL(C_FLOAT), INTENT(IN) :: a, b
              REAL(C_FLOAT), INTENT(INOUT) :: c
            END SUBROUTINE
          END INTERFACE
          CALL internal_gemm(transa, transb, m, n, k,                   &
     &      alpha, a, lda, b, ldb, beta, c, ldc)
        END SUBROUTINE

        !> Re-exposes BLAS based GEMM routine with an interfaces similar to
        !> libxsmm_(s)gemm. This overload belongs to libxsmm_blas_(s)gemm.
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_blas_sgemm1
        PURE SUBROUTINE libxsmm_blas_sgemm1(transa, transb, m, n, k,    &
     &  alpha, a, lda, b, ldb, beta, c, ldc)
          CHARACTER, INTENT(IN), OPTIONAL :: transa, transb
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: m, n, k
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN), OPTIONAL :: lda
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN), OPTIONAL :: ldb
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN), OPTIONAL :: ldc
          REAL(C_FLOAT), INTENT(IN), OPTIONAL :: alpha, beta
          REAL(C_FLOAT), INTENT(IN) :: a(:), b(:)
          REAL(C_FLOAT), INTENT(INOUT) :: c(:)
          IF ((0.LT.m).AND.(0.LT.n).AND.(0.LT.k)) THEN
            CALL libxsmm_blas_sgemm0(transa, transb, m, n, k,           &
     &        alpha, a(LBOUND(a,1)), lda,                               &
     &               b(LBOUND(b,1)), ldb,                               &
     &         beta, c(LBOUND(c,1)), ldc)
          END IF
        END SUBROUTINE

        !> Re-exposes BLAS based GEMM routine with an interfaces similar to
        !> libxsmm_(s)gemm. This overload belongs to libxsmm_blas_(s)gemm.
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_blas_sgemm2
        PURE SUBROUTINE libxsmm_blas_sgemm2(transa, transb, m, n, k,    &
     &  a, b, c, alpha, beta)
          CHARACTER, INTENT(IN), OPTIONAL :: transa, transb
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: m, n, k
          REAL(C_FLOAT), INTENT(IN), OPTIONAL :: alpha, beta
          REAL(C_FLOAT), INTENT(IN)    :: a(m,*), b(k,*)
          REAL(C_FLOAT), INTENT(INOUT) :: c(m,*)
          IF ((0.LT.m).AND.(0.LT.n).AND.(0.LT.k)) THEN
            CALL libxsmm_blas_sgemm0(transa, transb, m, n, k,           &
     &        alpha, a(LBOUND(a,1),LBOUND(a,2)), m,                     &
     &               b(LBOUND(b,1),LBOUND(b,2)), k,                     &
     &         beta, c(LBOUND(c,1),LBOUND(c,2)), m)
          END IF
        END SUBROUTINE

        !> Re-exposes BLAS based GEMM routine with an interfaces similar to
        !> libxsmm_(s)gemm. This overload belongs to libxsmm_blas_(s)gemm.
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_blas_sgemm3
        PURE SUBROUTINE libxsmm_blas_sgemm3(transa, transb, m, n, k,    &
     &  alpha, a, lda, b, ldb, beta, c, ldc)
          CHARACTER, INTENT(IN), OPTIONAL :: transa, transb
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: m, n, k
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: lda, ldb, ldc
          REAL(C_FLOAT), INTENT(IN), OPTIONAL :: alpha, beta
          REAL(C_FLOAT), INTENT(IN)    :: a(lda,*), b(ldb,*)
          REAL(C_FLOAT), INTENT(INOUT) :: c(ldc,*)
          IF ((0.LT.m).AND.(0.LT.n).AND.(0.LT.k)) THEN
            CALL libxsmm_blas_sgemm0(transa, transb, m, n, k,           &
     &        alpha, a(LBOUND(a,1),LBOUND(a,2)), lda,                   &
     &               b(LBOUND(b,1),LBOUND(b,2)), ldb,                   &
     &         beta, c(LBOUND(c,1),LBOUND(c,2)), ldc)
          END IF
        END SUBROUTINE

        !> Matrix-copy (2-dimensional copy) routine. If the input (optional)
        !> is not present, the routine is used to zero-fill the out-matrix.
        !> Implicit FORTRAN 77 interface:
        !> ARRAY        :: input, output
        !> INTEGER(4|8) :: m, n, ldi, ldo
        !> INTEGER(4)   :: typesize, prefetch
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_matcopy
        PURE SUBROUTINE libxsmm_matcopy(output, input, typesize,        &
     &  m, n, ldi, ldo, prefetch)
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: m
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN),                    &
     &                                OPTIONAL :: n, ldi, ldo
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
              INTEGER(C_INT), INTENT(IN) :: typesize, prefetch
            END SUBROUTINE
          END INTERFACE
          CALL internal_matcopy(output, input, typesize,                &
     &      m, n, ldi, ldo, prefetch)
        END SUBROUTINE

        !> Transpose a matrix (out-of-place form).
        !> Implicit FORTRAN 77 interface:
        !> ARRAY        :: input, output
        !> INTEGER(4|8) :: m, n, ldi, ldo
        !> INTEGER(4)   :: typesize
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

        !> Transpose a matrix (in-place form).
        !> Implicit FORTRAN 77 interface:
        !> ARRAY        :: matrix
        !> INTEGER(4|8) :: m, n, ld
        !> INTEGER(4)   :: typesize
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

        !> Returns the difference between two timer ticks (cycles).
        !> Implicit FORTRAN 77 interface: subroutine available.
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_timer_ncycles
        PURE FUNCTION libxsmm_timer_ncycles(tick0, tick1)
          INTEGER(LIBXSMM_TICKINT_KIND), INTENT(IN) :: tick0, tick1
          INTEGER(LIBXSMM_TICKINT_KIND) :: libxsmm_timer_ncycles
          !DIR$ ATTRIBUTES OFFLOAD:MIC :: internal_timer_ncycles
          INTERFACE
            PURE SUBROUTINE internal_timer_ncycles(ncycles,             &
     &      tick0, tick1) BIND(C, NAME="libxsmm_timer_ncycles_")
              IMPORT LIBXSMM_TICKINT_KIND
              INTEGER(LIBXSMM_TICKINT_KIND), INTENT(IN)  :: tick0, tick1
              INTEGER(LIBXSMM_TICKINT_KIND), INTENT(OUT) :: ncycles
            END SUBROUTINE
          END INTERFACE
          CALL internal_timer_ncycles(                                  &
     &      libxsmm_timer_ncycles, tick0, tick1)
          END FUNCTION

        !> Utility function to calculate a collection of scalar differences
        !> between two matrices (libxsmm_matdiff_info). The location (m, n)
        !> of the largest difference (linf_abs) is recorded (also if NaN).
        !> In case of NaN, differences are set to infinity. If no difference
        !> is discovered, the location (m, n) is negative (OOB).
        !> Implicit FORTRAN 77 interface:
        !> TYPE         :: info
        !> INTEGER(4)   :: datatype
        !> INTEGER(4|8) :: m, n, ldref, ldtst
        !> ARRAY        :: ref, tst
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_matdiff
        PURE SUBROUTINE libxsmm_matdiff(info, datatype, m, n,           &
     &  ref, tst, ldref, ldtst)
          INTEGER(C_INT),                INTENT(IN) :: datatype
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: m
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN),                    &
     &                                     OPTIONAL :: n, ldref, ldtst
          TYPE(C_PTR), INTENT(IN),         OPTIONAL :: ref, tst
          TYPE(LIBXSMM_MATDIFF_INFO),   INTENT(OUT) :: info
          !DIR$ ATTRIBUTES OFFLOAD:MIC :: internal_matdiff
          INTERFACE
            PURE SUBROUTINE internal_matdiff(info, datatype, m, n,      &
     &      ref, tst, ldref, ldtst) BIND(C, NAME="libxsmm_matdiff_")
              IMPORT LIBXSMM_MATDIFF_INFO, LIBXSMM_BLASINT_KIND
              IMPORT C_PTR, C_INT
              INTEGER(C_INT), INTENT(IN)                :: datatype
              INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: m, n
              INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: ldref, ldtst
              TYPE(C_PTR), INTENT(IN), VALUE            :: ref, tst
              TYPE(LIBXSMM_MATDIFF_INFO),   INTENT(OUT) :: info
            END SUBROUTINE
          END INTERFACE
          CALL internal_matdiff(info, datatype, m, n,                   &
     &      ref, tst, ldref, ldtst)
        END SUBROUTINE

        !> Calculate co-prime number <= n/2 (except: libxsmm_shuffle(0|1) == 0).
        !> Implicit FORTRAN 77 interface:
        !> INTEGER(4) :: coprime (OUT)
        !> INTEGER(4) :: n
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_shuffle
        ELEMENTAL FUNCTION libxsmm_shuffle(n)
          INTEGER(C_LONG_LONG) :: libxsmm_shuffle
          INTEGER(C_INT), INTENT(IN) :: n
          !DIR$ ATTRIBUTES OFFLOAD:MIC :: internal_shuffle
          INTERFACE
            PURE SUBROUTINE internal_shuffle(coprime, n)                &
     &      BIND(C, NAME="libxsmm_shuffle_")
              IMPORT C_LONG_LONG, C_INT
              INTEGER(C_LONG_LONG), INTENT(OUT) :: coprime
              INTEGER(C_INT), INTENT(IN) :: n
            END SUBROUTINE
          END INTERFACE
          libxsmm_shuffle = 0 ! avoids warning (older CRAY FTN)
          CALL internal_shuffle(libxsmm_shuffle, n)
        END FUNCTION

        !> Implicit FORTRAN 77 interface:
        !> INTEGER(4) :: hash_seed (INOUT)
        !> CHARACTER  :: key(:)
        !> INTEGER(4) :: keysize
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_hash_char
        FUNCTION libxsmm_hash_char(key, seed)
          CHARACTER(C_CHAR), DIMENSION(:), INTENT(IN) :: key
          INTEGER(C_INT), INTENT(IN) :: seed
          INTEGER(C_INT) :: libxsmm_hash_char
          !DIR$ ATTRIBUTES OFFLOAD:MIC :: internal_hash
          INTERFACE
            PURE SUBROUTINE internal_hash(hash_seed, key, keysize)      &
     &      BIND(C, NAME="libxsmm_hash_char_")
              IMPORT C_INT, C_PTR
              INTEGER(C_INT), INTENT(INOUT)   :: hash_seed
              INTEGER(C_INT), INTENT(IN)      :: keysize
              TYPE(C_PTR), INTENT(IN), VALUE  :: key
            END SUBROUTINE
          END INTERFACE
          libxsmm_hash_char = seed
          CALL internal_hash(libxsmm_hash_char,                         &
     &      libxsmm_ptr1(key), SIZE(key))
        END FUNCTION

        !> Implicit FORTRAN 77 interface:
        !> INTEGER(4) :: hash_seed (INOUT)
        !> INTEGER(1) :: key(:)
        !> INTEGER(4) :: keysize
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_hash_i8
        FUNCTION libxsmm_hash_i8(key, seed)
          INTEGER(C_INT8_T), DIMENSION(:), INTENT(IN) :: key
          INTEGER(C_INT), INTENT(IN) :: seed
          INTEGER(C_INT) :: libxsmm_hash_i8
          !DIR$ ATTRIBUTES OFFLOAD:MIC :: internal_hash
          INTERFACE
            PURE SUBROUTINE internal_hash(hash_seed, key, keysize)      &
     &      BIND(C, NAME="libxsmm_hash_i8_")
              IMPORT C_INT, C_PTR
              INTEGER(C_INT), INTENT(INOUT)   :: hash_seed
              INTEGER(C_INT), INTENT(IN)      :: keysize
              TYPE(C_PTR), INTENT(IN), VALUE  :: key
            END SUBROUTINE
          END INTERFACE
          libxsmm_hash_i8 = seed
          CALL internal_hash(libxsmm_hash_i8,                           &
     &      libxsmm_ptr1(key), SIZE(key))
        END FUNCTION

        !> Implicit FORTRAN 77 interface:
        !> INTEGER(4) :: hash_seed (INOUT)
        !> INTEGER(4) :: key(:)
        !> INTEGER(4) :: keysize
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_hash_i32
        FUNCTION libxsmm_hash_i32(key, seed)
          INTEGER(C_INT), DIMENSION(:), INTENT(IN) :: key
          INTEGER(C_INT), INTENT(IN) :: seed
          INTEGER(C_INT) :: libxsmm_hash_i32
          !DIR$ ATTRIBUTES OFFLOAD:MIC :: internal_hash
          INTERFACE
            PURE SUBROUTINE internal_hash(hash_seed, key, keysize)      &
     &      BIND(C, NAME="libxsmm_hash_i32_")
              IMPORT C_INT, C_PTR
              INTEGER(C_INT), INTENT(INOUT)   :: hash_seed
              INTEGER(C_INT), INTENT(IN)      :: keysize
              TYPE(C_PTR), INTENT(IN), VALUE  :: key
            END SUBROUTINE
          END INTERFACE
          libxsmm_hash_i32 = seed
          CALL internal_hash(libxsmm_hash_i32,                          &
     &      libxsmm_ptr1(key), SIZE(key) * 4)
        END FUNCTION

        !> Implicit FORTRAN 77 interface:
        !> INTEGER(4) :: hash_seed (INOUT)
        !> INTEGER(8) :: key(:)
        !> INTEGER(4) :: keysize
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_hash_i64
        FUNCTION libxsmm_hash_i64(key, seed)
          INTEGER(C_LONG_LONG), DIMENSION(:), INTENT(IN) :: key
          INTEGER(C_INT), INTENT(IN) :: seed
          INTEGER(C_INT) :: libxsmm_hash_i64
          !DIR$ ATTRIBUTES OFFLOAD:MIC :: internal_hash
          INTERFACE
            PURE SUBROUTINE internal_hash(hash_seed, key, keysize)      &
     &      BIND(C, NAME="libxsmm_hash_i64_")
              IMPORT C_INT, C_PTR
              INTEGER(C_INT), INTENT(INOUT)   :: hash_seed
              INTEGER(C_INT), INTENT(IN)      :: keysize
              TYPE(C_PTR), INTENT(IN), VALUE  :: key
            END SUBROUTINE
          END INTERFACE
          libxsmm_hash_i64 = seed
          CALL internal_hash(libxsmm_hash_i64,                          &
     &      libxsmm_ptr1(key), SIZE(key) * 8)
        END FUNCTION

        !> Implicit FORTRAN 77 interface:
        !> INTEGER(4) :: memcmp (OUT)
        !> CHARACTER  :: a(:), b(:)
        !> INTEGER(8) :: nbytes
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_diff_char
        FUNCTION libxsmm_diff_char(a, b)
          CHARACTER(C_CHAR), DIMENSION(:), INTENT(IN) :: a, b
          LOGICAL :: libxsmm_diff_char
          INTEGER(C_INT) :: memcmp
          !DIR$ ATTRIBUTES OFFLOAD:MIC :: internal_diff
          INTERFACE
            PURE SUBROUTINE internal_diff(memcmp, a, b, nbytes)         &
     &      BIND(C, NAME="libxsmm_diff_char_")
              IMPORT C_LONG_LONG, C_INT, C_PTR
              TYPE(C_PTR), INTENT(IN), VALUE    :: a, b
              INTEGER(C_LONG_LONG), INTENT(IN)  :: nbytes
              INTEGER(C_INT), INTENT(OUT)       :: memcmp
            END SUBROUTINE
          END INTERFACE
          IF (SIZE(a, KIND=C_LONG_LONG) .EQ. SIZE(b, KIND=C_LONG_LONG)) &
     &    THEN
            CALL internal_diff(memcmp,                                  &
     &        libxsmm_ptr1(a), libxsmm_ptr1(b),                         &
     &        SIZE(a, KIND=C_LONG_LONG) * 4)
            libxsmm_diff_char = 0.NE.memcmp
          ELSE
            libxsmm_diff_char = .TRUE.
          END IF
        END FUNCTION

        !> Implicit FORTRAN 77 interface:
        !> INTEGER(4) :: memcmp (OUT)
        !> INTEGER(1) :: a(:), b(:)
        !> INTEGER(8) :: nbytes
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_diff_i8
        FUNCTION libxsmm_diff_i8(a, b)
          INTEGER(C_INT8_T), DIMENSION(:), INTENT(IN) :: a, b
          LOGICAL :: libxsmm_diff_i8
          INTEGER(C_INT) :: memcmp
          !DIR$ ATTRIBUTES OFFLOAD:MIC :: internal_diff
          INTERFACE
            PURE SUBROUTINE internal_diff(memcmp, a, b, nbytes)         &
     &      BIND(C, NAME="libxsmm_diff_i8_")
              IMPORT C_LONG_LONG, C_INT, C_PTR
              TYPE(C_PTR), INTENT(IN), VALUE    :: a, b
              INTEGER(C_LONG_LONG), INTENT(IN)  :: nbytes
              INTEGER(C_INT), INTENT(OUT)       :: memcmp
            END SUBROUTINE
          END INTERFACE
          IF (SIZE(a, KIND=C_LONG_LONG) .EQ. SIZE(b, KIND=C_LONG_LONG)) &
     &    THEN
            CALL internal_diff(memcmp,                                  &
     &        libxsmm_ptr1(a), libxsmm_ptr1(b),                         &
     &        SIZE(a, KIND=C_LONG_LONG))
            libxsmm_diff_i8 = 0.NE.memcmp
          ELSE
            libxsmm_diff_i8 = .TRUE.
          END IF
        END FUNCTION

        !> Implicit FORTRAN 77 interface:
        !> INTEGER(4) :: memcmp (OUT)
        !> INTEGER(4) :: a(:), b(:)
        !> INTEGER(8) :: nbytes
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_diff_i32
        FUNCTION libxsmm_diff_i32(a, b)
          INTEGER(C_INT), DIMENSION(:), INTENT(IN) :: a, b
          LOGICAL :: libxsmm_diff_i32
          INTEGER(C_INT) :: memcmp
          !DIR$ ATTRIBUTES OFFLOAD:MIC :: internal_diff
          INTERFACE
            PURE SUBROUTINE internal_diff(memcmp, a, b, nbytes)         &
     &      BIND(C, NAME="libxsmm_diff_i32_")
              IMPORT C_LONG_LONG, C_INT, C_PTR
              TYPE(C_PTR), INTENT(IN), VALUE    :: a, b
              INTEGER(C_LONG_LONG), INTENT(IN)  :: nbytes
              INTEGER(C_INT), INTENT(OUT)       :: memcmp
            END SUBROUTINE
          END INTERFACE
          IF (SIZE(a, KIND=C_LONG_LONG) .EQ. SIZE(b, KIND=C_LONG_LONG)) &
     &    THEN
            CALL internal_diff(memcmp,                                  &
     &        libxsmm_ptr1(a), libxsmm_ptr1(b),                         &
     &        SIZE(a, KIND=C_LONG_LONG) * 4)
            libxsmm_diff_i32 = 0.NE.memcmp
          ELSE
            libxsmm_diff_i32 = .TRUE.
          END IF
        END FUNCTION

        !> Implicit FORTRAN 77 interface:
        !> INTEGER(4) :: memcmp (OUT)
        !> INTEGER(8) :: a(:), b(:)
        !> INTEGER(8) :: nbytes
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_diff_i64
        FUNCTION libxsmm_diff_i64(a, b)
          INTEGER(C_LONG_LONG), DIMENSION(:), INTENT(IN) :: a, b
          LOGICAL :: libxsmm_diff_i64
          INTEGER(C_INT) :: memcmp
          !DIR$ ATTRIBUTES OFFLOAD:MIC :: internal_diff
          INTERFACE
            PURE SUBROUTINE internal_diff(memcmp, a, b, nbytes)         &
     &      BIND(C, NAME="libxsmm_diff_i64_")
              IMPORT C_LONG_LONG, C_INT, C_PTR
              TYPE(C_PTR), INTENT(IN), VALUE    :: a, b
              INTEGER(C_LONG_LONG), INTENT(IN)  :: nbytes
              INTEGER(C_INT), INTENT(OUT)       :: memcmp
            END SUBROUTINE
          END INTERFACE
          IF (SIZE(a, KIND=C_LONG_LONG) .EQ. SIZE(b, KIND=C_LONG_LONG)) &
     &    THEN
            CALL internal_diff(memcmp,                                  &
     &        libxsmm_ptr1(a), libxsmm_ptr1(b),                         &
     &        SIZE(a, KIND=C_LONG_LONG) * 8)
            libxsmm_diff_i64 = 0.NE.memcmp
          ELSE
            libxsmm_diff_i64 = .TRUE.
          END IF
        END FUNCTION
      END MODULE

