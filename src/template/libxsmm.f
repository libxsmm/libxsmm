!=======================================================================!
! Copyright (c) Intel Corporation - All rights reserved.                !
! This file is part of the LIBXSMM library.                             !
!                                                                       !
! For information on the license, see the LICENSE file.                 !
! Further information: https://github.com/libxsmm/libxsmm/              !
! SPDX-License-Identifier: BSD-3-Clause                                 !
!=======================================================================!
! Hans Pabst (Intel Corp.)
!=======================================================================!

      MODULE LIBXSMM
        USE, INTRINSIC :: ISO_C_BINDING, ONLY:                          &
     &    C_DOUBLE, C_FLOAT, C_DOUBLE_COMPLEX, C_FLOAT_COMPLEX,         &
     &    C_LONG_LONG, C_INT, C_SHORT, C_CHAR, C_INT8_T, C_BOOL,        &
     &    C_F_POINTER, C_ASSOCIATED, C_LOC, C_PTR,                      &
     &    C_FUNPTR, C_NULL_FUNPTR, C_NULL_PTR
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
     &    LIBXSMM_GEMM_FLAG_NONE      = 0,                              &
     &    LIBXSMM_GEMM_FLAG_TRANS_A   = 1,                              &
     &    LIBXSMM_GEMM_FLAG_TRANS_B   = 2,                              &
     &    LIBXSMM_GEMM_FLAG_TRANS_AB  = IOR(                            &
     &        LIBXSMM_GEMM_FLAG_TRANS_A, LIBXSMM_GEMM_FLAG_TRANS_B),    &
     &    LIBXSMM_GEMM_FLAG_BETA_0    = 4,                              &
     &    LIBXSMM_GEMM_FLAG_ALIGN_A   = 8,                              &
     &    LIBXSMM_GEMM_FLAG_ALIGN_C   =16,                              &
     &    LIBXSMM_GEMM_FLAG_ALIGN_C_NTS_HINT = IOR(1024,                &
     &        LIBXSMM_GEMM_FLAG_ALIGN_C),                               &
     &    LIBXSMM_GEMM_FLAG_ALIGN_C_NTS_HINT_BETA_0 = IOR(              &
     &        LIBXSMM_GEMM_FLAG_ALIGN_C_NTS_HINT,                       &
     &        LIBXSMM_GEMM_FLAG_BETA_0),                                &
     &    LIBXSMM_GEMM_FLAG_INVALID = 131072

        !> Flag enumeration which can be IORed.
        INTEGER(C_INT), PARAMETER ::                                    &
          ! Handle recorded batch unsynchronized-parallel.
     &    LIBXSMM_MMBATCH_FLAG_DEFAULT      = 0                         &
     &        * LIBXSMM_GEMM_FLAG_INVALID,                              &
          ! Synchronize among C matrices.
     &    LIBXSMM_MMBATCH_FLAG_SYNCHRONIZED = 1                         &
     &        * LIBXSMM_GEMM_FLAG_INVALID,                              &
          ! Handle recorded batch sequentially.
     &    LIBXSMM_MMBATCH_FLAG_SEQUENTIAL   = 2                         &
     &        * LIBXSMM_GEMM_FLAG_INVALID,                              &
          ! Only record a statistic of potential SMMs.
     &    LIBXSMM_MMBATCH_FLAG_STATISTIC    = 4                         &
     &        * LIBXSMM_GEMM_FLAG_INVALID

        !> Enumerates element/data types.
        INTEGER(C_INT), PARAMETER ::                                    &
     &    LIBXSMM_DATATYPE_F64  = 0,                                    &
     &    LIBXSMM_DATATYPE_F32  = 1,                                    &
     &    LIBXSMM_DATATYPE_BF16 = 2,                                    &
     &    LIBXSMM_DATATYPE_F16  = 3,                                    &
     &    LIBXSMM_DATATYPE_I64  = 3,                                    &
     &    LIBXSMM_DATATYPE_I32  = 4,                                    &
     &    LIBXSMM_DATATYPE_I16  = 5,                                    &
     &    LIBXSMM_DATATYPE_I8   = 6,                                    &
     &    LIBXSMM_DATATYPE_UNSUPPORTED = 7

        !> Enumeration of the available prefetch strategies which can be IORed.
        INTEGER(C_INT), PARAMETER ::                                    &
          ! Automatically select strategy (frontend).
     &    LIBXSMM_PREFETCH_AUTO                     = -1,               &
          ! No prefetching and no prefetch function signature.
     &    LIBXSMM_PREFETCH_NONE                     = 0,                &
          ! Only function prefetch signature.
     &    LIBXSMM_PREFETCH_SIGONLY                  = 1,                &
          ! Prefetch PA using accesses to A.
     &    LIBXSMM_GEMM_PREFETCH_AL2                 = 2,                &
          ! Prefetch PB using accesses to C.
     &    LIBXSMM_GEMM_PREFETCH_BL2_VIA_C           = 4,                &
          ! Prefetch A ahead.
     &    LIBXSMM_GEMM_PREFETCH_AL2_AHEAD           = 8,                &
          ! Composed prefetch strategies.
     &    LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C        = IOR(              &
     &        LIBXSMM_GEMM_PREFETCH_BL2_VIA_C,                          &
     &        LIBXSMM_GEMM_PREFETCH_AL2),                               &
     &    LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C_AHEAD  = IOR(              &
     &        LIBXSMM_GEMM_PREFETCH_BL2_VIA_C,                          &
     &        LIBXSMM_GEMM_PREFETCH_AL2_AHEAD),                         &
          ! Current B into L1.
     &    LIBXSMM_GEMM_PREFETCH_BL1                 = 16

        !> Enumerates the available target architectures and instruction
        !> set extensions as returned by libxsmm_get_target_archid().
        INTEGER(C_INT), PARAMETER ::                                    &
     &    LIBXSMM_TARGET_ARCH_UNKNOWN   = 0,                            &
     &    LIBXSMM_TARGET_ARCH_GENERIC   = 1,                            &
     &    LIBXSMM_X86_GENERIC           = 1002,                         &
     &    LIBXSMM_X86_SSE3              = 1003,                         &
     &    LIBXSMM_X86_SSE4              = 1004,                         &
     &    LIBXSMM_X86_AVX               = 1005,                         &
     &    LIBXSMM_X86_AVX2              = 1006,                         &
     &    LIBXSMM_X86_AVX512_VL256      = 1007,                         &
     &    LIBXSMM_X86_AVX512_VL256_CLX  = 1008,                         &
     &    LIBXSMM_X86_AVX512_VL256_CPX  = 1009,                         &
     &    LIBXSMM_X86_AVX512            = 1010,                         &
     &    LIBXSMM_X86_AVX512_MIC        = 1011,                         &
     &    LIBXSMM_X86_AVX512_KNM        = 1012,                         &
     &    LIBXSMM_X86_AVX512_CORE       = 1020,                         &
     &    LIBXSMM_X86_AVX512_CLX        = 1021,                         &
     &    LIBXSMM_X86_AVX512_CPX        = 1022,                         &
     &    LIBXSMM_X86_AVX512_SPR        = 1023,                         &
     &    LIBXSMM_X86_ALLFEAT           = 1999,                         &
     &    LIBXSMM_AARCH64_V81           = 2001,                         &
     &    LIBXSMM_AARCH64_V82           = 2002,                         &
     &    LIBXSMM_AARCH64_A64FX         = 2100,                         &
     &    LIBXSMM_AARCH64_APPL_M1       = 2200,                         &
     &    LIBXSMM_AARCH64_ALLFEAT       = 2999

        !> Generic function type (double-precision).
        TYPE, BIND(C) :: LIBXSMM_DMMFUNCTION
          TYPE(C_FUNPTR) :: handle = C_NULL_FUNPTR
        END TYPE

        !> Generic function type (single-precision).
        TYPE, BIND(C) :: LIBXSMM_SMMFUNCTION
          TYPE(C_FUNPTR) :: handle = C_NULL_FUNPTR
        END TYPE

        !> Generic function type (low-precision)
        TYPE, BIND(C) :: LIBXSMM_WIMMFUNCTION
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
          !> Maximum difference, L2-norm (absolute and relative), and R-squared.
          REAL(C_DOUBLE) linf_abs, linf_rel, l2_abs, l2_rel, rsq
          !> Statistics: sum/l1, min., max., arith. avg., and variance.
          REAL(C_DOUBLE) l1_ref, min_ref, max_ref, avg_ref, var_ref
          !> Statistics: sum/l1, min., max., arith. avg., and variance.
          REAL(C_DOUBLE) l1_tst, min_tst, max_tst, avg_tst, var_tst
          !> Values (v_ref, v_tst) and location (m, n) of largest linf_abs.
          REAL(C_DOUBLE) v_ref, v_tst
          !> Values (v_ref, v_tst), location (m, n), and zero-based i-th of
          !> r reductions (libxsmm_matdiff_reduce) of smallest R-squared.
          INTEGER(LIBXSMM_BLASINT_KIND) m, n, i, r
        END TYPE

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
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_xclear
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_xrelease
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_xmatcopy
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_xitrans
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_xotrans
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_matcopy_omp
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
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_xhash
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_xdiff
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
            INTEGER(C_INT), INTENT(IN)  :: gemm_precision
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
            INTEGER(C_INT), INTENT(IN)  :: iprec, oprec
            INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: m, n, k
            TYPE(C_PTR), INTENT(IN), VALUE :: lda, ldb, ldc
            TYPE(C_PTR), INTENT(IN), VALUE :: alpha, beta
            TYPE(C_PTR), INTENT(IN), VALUE :: flags, prefetch
          END SUBROUTINE

          !> Generic call routine (3-argument form).
          !> Implicit FORTRAN 77 interface:
          !> REAL(4|8)  :: a, b, c
          !> INTEGER(8) :: kernel
          PURE SUBROUTINE libxsmm_xmmcall_abc(kernel, a, b, c)          &
     &    BIND(C, NAME="libxsmm_xmmcall_abc_")
            IMPORT :: C_FUNPTR, C_PTR
            TYPE(C_FUNPTR), INTENT(IN) :: kernel
            TYPE(C_PTR), INTENT(IN), VALUE :: a, b, c
          END SUBROUTINE

          !> Generic call routine (6-argument form).
          !> Implicit FORTRAN 77 interface:
          !> REAL(4|8)  :: a, b, c, pa, pb, pc
          !> INTEGER(8) :: kernel
          PURE SUBROUTINE libxsmm_xmmcall_prf(kernel,                   &
     &    a, b, c, pa, pb, pc)                                          &
     &    BIND(C, NAME="libxsmm_xmmcall_prf_")
            IMPORT :: C_FUNPTR, C_PTR
            TYPE(C_FUNPTR), INTENT(IN) :: kernel
            TYPE(C_PTR), INTENT(IN), VALUE :: a, b, c, pa, pb, pc
          END SUBROUTINE

          !> Fill destination with zeros; treats dst in raw/binary fashion.
          SUBROUTINE libxsmm_xclear(dst, nbytes)                        &
     &    BIND(C, NAME="libxsmm_xclear_")
            IMPORT :: C_PTR, C_INT
            TYPE(C_PTR), INTENT(IN), VALUE :: dst
            INTEGER(C_INT), INTENT(IN) :: nbytes
          END SUBROUTINE

          !> Remove key-value pair from code registry and release memory.
          SUBROUTINE libxsmm_xrelease(key, keysize)                     &
     &    BIND(C, NAME="libxsmm_xrelease_")
            IMPORT :: C_PTR, C_INT
            TYPE(C_PTR), INTENT(IN), VALUE :: key
            INTEGER(C_INT), INTENT(IN) :: keysize
          END SUBROUTINE

          !> Matrix-copy (2-dimensional copy) routine.
          !> Implicit FORTRAN 77 interface:
          !> ARRAY        :: input, output
          !> INTEGER(4|8) :: m, n, ldi, ldo
          !> INTEGER(4)   :: typesize
          PURE SUBROUTINE libxsmm_xmatcopy(output, input, typesize,     &
     &    m, n, ldi, ldo) BIND(C, NAME="libxsmm_matcopy_")
            IMPORT :: LIBXSMM_BLASINT_KIND, C_PTR, C_INT
            INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: m, n, ldi, ldo
            TYPE(C_PTR), INTENT(IN), VALUE :: output, input
            INTEGER(C_INT), INTENT(IN) :: typesize
          END SUBROUTINE

          !> Transpose a matrix (in-place form).
          !> Implicit FORTRAN 77 interface:
          !> ARRAY        :: matrix
          !> INTEGER(4|8) :: m, n, ldi, ldo
          !> INTEGER(4)   :: typesize
          PURE SUBROUTINE libxsmm_xitrans(matrix, typesize,             &
     &    m, n, ldi, ldo) BIND(C, NAME="libxsmm_itrans_")
            IMPORT :: C_PTR, C_INT, LIBXSMM_BLASINT_KIND
            INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: m, n, ldi, ldo
            TYPE(C_PTR), INTENT(IN), VALUE :: matrix
            INTEGER(C_INT), INTENT(IN) :: typesize
          END SUBROUTINE

          !> Transpose a matrix (out-of-place form).
          !> Implicit FORTRAN 77 interface:
          !> ARRAY        :: input, output
          !> INTEGER(4|8) :: m, n, ldi, ldo
          !> INTEGER(4)   :: typesize
          PURE SUBROUTINE libxsmm_xotrans(output, input, typesize,      &
     &    m, n, ldi, ldo) BIND(C, NAME="libxsmm_otrans_")
            IMPORT :: C_PTR, C_INT, LIBXSMM_BLASINT_KIND
            INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: m, n, ldi, ldo
            TYPE(C_PTR), INTENT(IN), VALUE :: output, input
            INTEGER(C_INT), INTENT(IN) :: typesize
          END SUBROUTINE

          !> Matrix copy; MT via libxsmmext (out-of-place form).
          !> Implicit FORTRAN 77 interface:
          !> ARRAY        :: output, input
          !> INTEGER(4|8) :: m, n, ldi, ldo
          !> INTEGER(4)   :: typesize
          PURE SUBROUTINE libxsmm_matcopy_omp(output, input, typesize,  &
     &    m, n, ldi, ldo) BIND(C, NAME="libxsmm_matcopy_omp_")
            IMPORT :: C_PTR, C_INT, LIBXSMM_BLASINT_KIND
            INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: m, n, ldi, ldo
            TYPE(C_PTR), INTENT(IN), VALUE :: output, input
            INTEGER(C_INT), INTENT(IN) :: typesize
          END SUBROUTINE

          !> Matrix transposition; MT via libxsmmext (out-of-place form).
          !> Implicit FORTRAN 77 interface:
          !> ARRAY        :: output, input
          !> INTEGER(4|8) :: m, n, ldi, ldo
          !> INTEGER(4)   :: typesize
          PURE SUBROUTINE libxsmm_otrans_omp(output, input, typesize,   &
     &    m, n, ldi, ldo) BIND(C, NAME="libxsmm_otrans_omp_")
            IMPORT :: C_PTR, C_INT, LIBXSMM_BLASINT_KIND
            INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: m, n, ldi, ldo
            TYPE(C_PTR), INTENT(IN), VALUE :: output, input
            INTEGER(C_INT), INTENT(IN) :: typesize
          END SUBROUTINE

          !> General dense MM; MT via libxsmmext (double-precision).
          !> Implicit FORTRAN 77 interface: similar to DGEMM.
          PURE SUBROUTINE libxsmm_dgemm_omp(transa, transb, m, n, k,    &
     &    alpha, a, lda, b, ldb, beta, c, ldc)                          &
     &    BIND(C, NAME="libxsmm_dgemm_omp_")
            IMPORT :: C_DOUBLE, C_CHAR, LIBXSMM_BLASINT_KIND
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
            IMPORT :: C_FLOAT, C_CHAR, LIBXSMM_BLASINT_KIND
            CHARACTER(C_CHAR), INTENT(IN) :: transa, transb
            INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: m, n, k
            INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: lda, ldb, ldc
            REAL(C_FLOAT), INTENT(IN)    :: alpha, beta
            REAL(C_FLOAT), INTENT(IN)    :: a(lda,*), b(ldb,*)
            REAL(C_FLOAT), INTENT(INOUT) :: c(ldc,*)
          END SUBROUTINE

          !> Process a series of MMs (batch). See also libxsmm_gemm_batch_omp.
          !> The kind of matrix operands (a, b, c) depend on index_stride:
          !> index_stride==0: pointers to pointers of elements, e.g.,
          !> double** for the C matrices.
          !> index_stride!=0: pointer to elements, e.g.,
          !> const double* for the A and B matrices.
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
     &    tid, nthreads)                                                &
     &    BIND(C, NAME="libxsmm_mmbatch_")
            IMPORT :: C_PTR, C_CHAR, C_INT, LIBXSMM_BLASINT_KIND
            !> Determines index-base (usually 0, 1 for one-based indexes).
            INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: index_base
            !> Stride (measured in Bytes) used to walk stride_*.
            !> In Fortran: index_stride!=0.
            INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: index_stride
            !> Number of SMMs. If the size is given as a negative value,
            !> then internal synchronization is omitted.
            INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: batchsize
            INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: m, n, k
            INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: lda, ldb, ldc
            CHARACTER(C_CHAR),  INTENT(IN) :: transa, transb
            TYPE(C_PTR), INTENT(IN), VALUE :: alpha, beta
            TYPE(C_PTR), INTENT(IN), VALUE :: a, b, c
            !> Arrays of indexes determining the position of
            !> a, b, and c operands.
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
     &    batchsize)                                                    &
     &    BIND(C, NAME="libxsmm_gemm_batch_")
            IMPORT :: C_PTR, C_CHAR, C_INT, LIBXSMM_BLASINT_KIND
            INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: index_base
            INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: index_stride
            INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: batchsize
            INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: m, n, k
            INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: lda, ldb, ldc
            CHARACTER(C_CHAR),  INTENT(IN) :: transa, transb
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
     &    batchsize)                                                    &
     &    BIND(C, NAME="libxsmm_gemm_batch_omp_")
            IMPORT :: C_PTR, C_CHAR, C_INT, LIBXSMM_BLASINT_KIND
            INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: index_base
            INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: index_stride
            INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: batchsize
            INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: m, n, k
            INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: lda, ldb, ldc
            CHARACTER(C_CHAR),  INTENT(IN) :: transa, transb
            TYPE(C_PTR), INTENT(IN), VALUE :: alpha, beta
            TYPE(C_PTR), INTENT(IN), VALUE :: a, b, c
            TYPE(C_PTR), INTENT(IN), VALUE :: stride_a
            TYPE(C_PTR), INTENT(IN), VALUE :: stride_b
            TYPE(C_PTR), INTENT(IN), VALUE :: stride_c
            INTEGER(C_INT), INTENT(IN) :: iprec, oprec
          END SUBROUTINE

          !> This function is a no-op unless LIBXSMM is built to intercept GEMM.
          !> Pointer arguments are used to filter intercepted GEMM calls such that
          !> non-NULL values match. Otherwise (NULL) the respective argument is
          !> considered a "free value", i.e., every value can match;
          !> libxsmmext required.
          !> Implicit FORTRAN 77 interface:
          !> INTEGER(4)   :: gemm_precision, flags
          !> INTEGER(4|8) :: m, n, k, lda, ldb, ldc
          !> REAL(4|8)    :: alpha, beta
          SUBROUTINE libxsmm_mmbatch_begin(gemm_precision, flags,       &
     &    m, n, k,  lda, ldb, ldc, alpha, beta) BIND(C)
            IMPORT :: C_PTR, C_INT, LIBXSMM_BLASINT_KIND
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
            IMPORT :: LIBXSMM_MATDIFF_INFO
            TYPE(LIBXSMM_MATDIFF_INFO), INTENT(INOUT) :: output
            TYPE(LIBXSMM_MATDIFF_INFO), INTENT(IN)    :: input
          END SUBROUTINE

          !> Clears the given info-structure, e.g., for the initial
          !> reduction-value (libxsmm_matdiff_reduce).
          !> Implicit FORTRAN 77 interface: available.
          PURE SUBROUTINE libxsmm_matdiff_clear(info) BIND(C)
            IMPORT :: LIBXSMM_MATDIFF_INFO
            TYPE(LIBXSMM_MATDIFF_INFO), INTENT(OUT) :: info
          END SUBROUTINE

          !> Calculates a hash value for the given array and seed.
          !> Routine suitable for FORTRAN 77; keysize in Bytes.
          PURE SUBROUTINE libxsmm_xhash(hash_seed, key, keysize)        &
     &    BIND(C, NAME="libxsmm_xhash_")
            IMPORT :: C_INT, C_PTR
            INTEGER(C_INT), INTENT(INOUT)  :: hash_seed
            INTEGER(C_INT), INTENT(IN)     :: keysize
            TYPE(C_PTR), INTENT(IN), VALUE :: key
          END SUBROUTINE

          !> Calculates if there is a difference between two arrays.
          !> Routine suitable for FORTRAN 77; size in Bytes.
          PURE SUBROUTINE libxsmm_xdiff(diff, a, b, nbytes)             &
     &    BIND(C, NAME="libxsmm_xdiff_")
            IMPORT :: C_PTR, C_LONG_LONG, C_BOOL
            TYPE(C_PTR), INTENT(IN), VALUE   :: a, b
            INTEGER(C_LONG_LONG), INTENT(IN) :: nbytes
            LOGICAL(C_BOOL), INTENT(OUT)     :: diff
          END SUBROUTINE
        END INTERFACE$MNK_INTERFACE_LIST

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
          MODULE PROCEDURE libxsmm_ptr_dmm
          MODULE PROCEDURE libxsmm_ptr_smm
        END INTERFACE

        INTERFACE libxsmm_ptr2
          MODULE PROCEDURE libxsmm_ptr_z2, libxsmm_ptr_c2
          MODULE PROCEDURE libxsmm_ptr_d2, libxsmm_ptr_s2
          MODULE PROCEDURE libxsmm_ptr_i2, libxsmm_ptr_w2
          MODULE PROCEDURE libxsmm_ptr_j2 !! Byte/char
          MODULE PROCEDURE libxsmm_ptr_b2 !! Byte/char
          MODULE PROCEDURE libxsmm_ptr_l2 !! long long
        END INTERFACE

        INTERFACE libxsmm_ptr
          MODULE PROCEDURE libxsmm_ptr_z0, libxsmm_ptr_c0
          MODULE PROCEDURE libxsmm_ptr_d0, libxsmm_ptr_s0
          MODULE PROCEDURE libxsmm_ptr_i0, libxsmm_ptr_w0
          MODULE PROCEDURE libxsmm_ptr_j0 !! Byte/char
          MODULE PROCEDURE libxsmm_ptr_b0 !! Byte/char
          MODULE PROCEDURE libxsmm_ptr_l0 !! long long
          MODULE PROCEDURE libxsmm_ptr_z1, libxsmm_ptr_c1
          MODULE PROCEDURE libxsmm_ptr_d1, libxsmm_ptr_s1
          MODULE PROCEDURE libxsmm_ptr_i1, libxsmm_ptr_w1
          MODULE PROCEDURE libxsmm_ptr_j1 !! Byte/char
          MODULE PROCEDURE libxsmm_ptr_b1 !! Byte/char
          MODULE PROCEDURE libxsmm_ptr_l1 !! long long
          MODULE PROCEDURE libxsmm_ptr_z2, libxsmm_ptr_c2
          MODULE PROCEDURE libxsmm_ptr_d2, libxsmm_ptr_s2
          MODULE PROCEDURE libxsmm_ptr_i2, libxsmm_ptr_w2
          MODULE PROCEDURE libxsmm_ptr_j2 !! Byte/char
          MODULE PROCEDURE libxsmm_ptr_b2 !! Byte/char
          MODULE PROCEDURE libxsmm_ptr_l2 !! long long
          MODULE PROCEDURE libxsmm_ptr_dmm
          MODULE PROCEDURE libxsmm_ptr_smm
        END INTERFACE

        !> Deallocates JIT'ted code, or unregisters/releases code from registry.
        INTERFACE libxsmm_release_mmkernel
          MODULE PROCEDURE libxsmm_release_dmmkernel
          MODULE PROCEDURE libxsmm_release_smmkernel
        END INTERFACE

        !> Construct JIT-code depending on given argument set.
        INTERFACE libxsmm_mmdispatch
          MODULE PROCEDURE libxsmm_dmmdispatch, libxsmm_smmdispatch
        END INTERFACE

        !> Construct JIT-code depending on given argument set.
        INTERFACE libxsmm_dispatch
          MODULE PROCEDURE libxsmm_dmmdispatch, libxsmm_smmdispatch
        END INTERFACE

        !> Check if a function is available (LIBXSMM_?MMFUNCTION).
        INTERFACE libxsmm_mmavailable
          MODULE PROCEDURE libxsmm_dmmavailable, libxsmm_smmavailable
        END INTERFACE

        !> Check if a function is available (LIBXSMM_?MMFUNCTION).
        INTERFACE libxsmm_available
          MODULE PROCEDURE libxsmm_smmavailable, libxsmm_dmmavailable
        END INTERFACE

        !> Overloaded GEMM routines (double-precision).
        INTERFACE libxsmm_dgemm
          MODULE PROCEDURE libxsmm_dgemm0
          MODULE PROCEDURE libxsmm_dgemm1
          MODULE PROCEDURE libxsmm_dgemm2
          MODULE PROCEDURE libxsmm_dgemm3
        END INTERFACE

        !> Overloaded GEMM routines (single-precision).
        INTERFACE libxsmm_sgemm
          MODULE PROCEDURE libxsmm_sgemm0
          MODULE PROCEDURE libxsmm_sgemm1
          MODULE PROCEDURE libxsmm_sgemm2
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
        END INTERFACE

        !> Overloaded BLAS GEMM routines (double-precision).
        INTERFACE libxsmm_blas_dgemm
          MODULE PROCEDURE libxsmm_blas_dgemm0
          MODULE PROCEDURE libxsmm_blas_dgemm1
          MODULE PROCEDURE libxsmm_blas_dgemm2
          MODULE PROCEDURE libxsmm_blas_dgemm3
        END INTERFACE

        !> Overloaded BLAS GEMM routines (single-precision).
        INTERFACE libxsmm_blas_sgemm
          MODULE PROCEDURE libxsmm_blas_sgemm0
          MODULE PROCEDURE libxsmm_blas_sgemm1
          MODULE PROCEDURE libxsmm_blas_sgemm2
          MODULE PROCEDURE libxsmm_blas_sgemm3
        END INTERFACE

        !> Overloaded BLAS GEMM routines (single/double-precision).
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

        !> Overloaded MATCOPY routines (2d-copy).
        INTERFACE libxsmm_matcopy
          MODULE PROCEDURE libxsmm_matcopy_p0
          MODULE PROCEDURE libxsmm_matcopy_d1
          MODULE PROCEDURE libxsmm_matcopy_d2
          MODULE PROCEDURE libxsmm_matcopy_s1
          MODULE PROCEDURE libxsmm_matcopy_s2
        END INTERFACE

        !> Overloaded TRANSPOSE routines (in-place form).
        INTERFACE libxsmm_itrans
          MODULE PROCEDURE libxsmm_itrans_p0
          MODULE PROCEDURE libxsmm_itrans_d1
          MODULE PROCEDURE libxsmm_itrans_d2
          MODULE PROCEDURE libxsmm_itrans_s1
          MODULE PROCEDURE libxsmm_itrans_s2
        END INTERFACE

        !> Overloaded TRANSPOSE routines (out-of-place form).
        INTERFACE libxsmm_otrans
          MODULE PROCEDURE libxsmm_otrans_p0
          MODULE PROCEDURE libxsmm_otrans_d1
          MODULE PROCEDURE libxsmm_otrans_d2
          MODULE PROCEDURE libxsmm_otrans_s1
          MODULE PROCEDURE libxsmm_otrans_s2
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

        !> Returns C_NULL_PTR.
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_ptr_null
        PURE FUNCTION libxsmm_ptr_null()
          TYPE(C_PTR) :: libxsmm_ptr_null
          libxsmm_ptr_null = C_NULL_PTR
        END FUNCTION

        !> Determines the C-address of the given array.
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_ptr_z0
        FUNCTION libxsmm_ptr_z0(a)
          COMPLEX(C_DOUBLE_COMPLEX), INTENT(IN), TARGET :: a
          TYPE(C_PTR) :: libxsmm_ptr_z0
          libxsmm_ptr_z0 = C_LOC(a)
        END FUNCTION
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_ptr_z1
        FUNCTION libxsmm_ptr_z1(a)
          COMPLEX(C_DOUBLE_COMPLEX), INTENT(IN), TARGET :: a(*)
          TYPE(C_PTR) :: libxsmm_ptr_z1
          libxsmm_ptr_z1 = C_LOC(a)
        END FUNCTION
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_ptr_z2
        FUNCTION libxsmm_ptr_z2(a)
          COMPLEX(C_DOUBLE_COMPLEX), INTENT(IN) :: a(:,:)
          TYPE(C_PTR) :: libxsmm_ptr_z2
          libxsmm_ptr_z2 = libxsmm_ptr_z1(a)
        END FUNCTION

        !> Determines the C-address of the given array.
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_ptr_c0
        FUNCTION libxsmm_ptr_c0(a)
          COMPLEX(C_FLOAT_COMPLEX), INTENT(IN), TARGET :: a
          TYPE(C_PTR) :: libxsmm_ptr_c0
          libxsmm_ptr_c0 = C_LOC(a)
        END FUNCTION
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_ptr_c1
        FUNCTION libxsmm_ptr_c1(a)
          COMPLEX(C_FLOAT_COMPLEX), INTENT(IN), TARGET :: a(*)
          TYPE(C_PTR) :: libxsmm_ptr_c1
          libxsmm_ptr_c1 = C_LOC(a)
        END FUNCTION
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_ptr_c2
        FUNCTION libxsmm_ptr_c2(a)
          COMPLEX(C_FLOAT_COMPLEX), INTENT(IN) :: a(:,:)
          TYPE(C_PTR) :: libxsmm_ptr_c2
          libxsmm_ptr_c2 = libxsmm_ptr_c1(a)
        END FUNCTION

        !> Determines the C-address of the given array.
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_ptr_d0
        FUNCTION libxsmm_ptr_d0(a)
          REAL(C_DOUBLE), INTENT(IN), TARGET :: a
          TYPE(C_PTR) :: libxsmm_ptr_d0
          libxsmm_ptr_d0 = C_LOC(a)
        END FUNCTION
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_ptr_d1
        FUNCTION libxsmm_ptr_d1(a)
          REAL(C_DOUBLE), INTENT(IN), TARGET :: a(*)
          TYPE(C_PTR) :: libxsmm_ptr_d1
          libxsmm_ptr_d1 = C_LOC(a)
        END FUNCTION
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_ptr_d2
        FUNCTION libxsmm_ptr_d2(a)
          REAL(C_DOUBLE), INTENT(IN) :: a(:,:)
          TYPE(C_PTR) :: libxsmm_ptr_d2
          libxsmm_ptr_d2 = libxsmm_ptr_d1(a)
        END FUNCTION

        !> Determines the C-address of the given array.
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_ptr_s0
        FUNCTION libxsmm_ptr_s0(a)
          REAL(C_FLOAT), INTENT(IN), TARGET :: a
          TYPE(C_PTR) :: libxsmm_ptr_s0
          libxsmm_ptr_s0 = C_LOC(a)
        END FUNCTION
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_ptr_s1
        FUNCTION libxsmm_ptr_s1(a)
          REAL(C_FLOAT), INTENT(IN), TARGET :: a(*)
          TYPE(C_PTR) :: libxsmm_ptr_s1
          libxsmm_ptr_s1 = C_LOC(a)
        END FUNCTION
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_ptr_s2
        FUNCTION libxsmm_ptr_s2(a)
          REAL(C_FLOAT), INTENT(IN) :: a(:,:)
          TYPE(C_PTR) :: libxsmm_ptr_s2
          libxsmm_ptr_s2 = libxsmm_ptr_s1(a)
        END FUNCTION

        !> Determines the C-address of the given array.
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_ptr_i0
        FUNCTION libxsmm_ptr_i0(a)
          INTEGER(C_INT), INTENT(IN), TARGET :: a
          TYPE(C_PTR) :: libxsmm_ptr_i0
          libxsmm_ptr_i0 = C_LOC(a)
        END FUNCTION
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_ptr_i1
        FUNCTION libxsmm_ptr_i1(a)
          INTEGER(C_INT), INTENT(IN), TARGET :: a(*)
          TYPE(C_PTR) :: libxsmm_ptr_i1
          libxsmm_ptr_i1 = C_LOC(a)
        END FUNCTION
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_ptr_i2
        FUNCTION libxsmm_ptr_i2(a)
          INTEGER(C_INT), INTENT(IN) :: a(:,:)
          TYPE(C_PTR) :: libxsmm_ptr_i2
          libxsmm_ptr_i2 = libxsmm_ptr_i1(a)
        END FUNCTION

        !> Determines the C-address of the given array.
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_ptr_w0
        FUNCTION libxsmm_ptr_w0(a)
          INTEGER(C_SHORT), INTENT(IN), TARGET :: a
          TYPE(C_PTR) :: libxsmm_ptr_w0
          libxsmm_ptr_w0 = C_LOC(a)
        END FUNCTION
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_ptr_w1
        FUNCTION libxsmm_ptr_w1(a)
          INTEGER(C_SHORT), INTENT(IN), TARGET :: a(*)
          TYPE(C_PTR) :: libxsmm_ptr_w1
          libxsmm_ptr_w1 = C_LOC(a)
        END FUNCTION
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_ptr_w2
        FUNCTION libxsmm_ptr_w2(a)
          INTEGER(C_SHORT), INTENT(IN) :: a(:,:)
          TYPE(C_PTR) :: libxsmm_ptr_w2
          libxsmm_ptr_w2 = libxsmm_ptr_w1(a)
        END FUNCTION

        !> Determines the C-address of the given array.
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_ptr_j0
        FUNCTION libxsmm_ptr_j0(a)
          INTEGER(C_INT8_T), INTENT(IN), TARGET :: a
          TYPE(C_PTR) :: libxsmm_ptr_j0
          libxsmm_ptr_j0 = C_LOC(a)
        END FUNCTION
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_ptr_j1
        FUNCTION libxsmm_ptr_j1(a)
          INTEGER(C_INT8_T), INTENT(IN), TARGET :: a(*)
          TYPE(C_PTR) :: libxsmm_ptr_j1
          libxsmm_ptr_j1 = C_LOC(a)
        END FUNCTION
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_ptr_j2
        FUNCTION libxsmm_ptr_j2(a)
          INTEGER(C_INT8_T), INTENT(IN) :: a(:,:)
          TYPE(C_PTR) :: libxsmm_ptr_j2
          libxsmm_ptr_j2 = libxsmm_ptr_j1(a)
        END FUNCTION

        !> Determines the C-address of the given array.
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_ptr_b0
        FUNCTION libxsmm_ptr_b0(a)
          CHARACTER(C_CHAR), INTENT(IN), TARGET :: a
          TYPE(C_PTR) :: libxsmm_ptr_b0
          libxsmm_ptr_b0 = C_LOC(a)
        END FUNCTION
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_ptr_b1
        FUNCTION libxsmm_ptr_b1(a)
          CHARACTER(C_CHAR), INTENT(IN), TARGET :: a(*)
          TYPE(C_PTR) :: libxsmm_ptr_b1
          libxsmm_ptr_b1 = C_LOC(a)
        END FUNCTION
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_ptr_b2
        FUNCTION libxsmm_ptr_b2(a)
          CHARACTER(C_CHAR), INTENT(IN) :: a(:,:)
          TYPE(C_PTR) :: libxsmm_ptr_b2
          libxsmm_ptr_b2 = libxsmm_ptr_b1(a)
        END FUNCTION

        !> Determines the C-address of the given array.
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_ptr_l0
        FUNCTION libxsmm_ptr_l0(a)
          INTEGER(C_LONG_LONG), INTENT(IN), TARGET :: a
          TYPE(C_PTR) :: libxsmm_ptr_l0
          libxsmm_ptr_l0 = C_LOC(a)
        END FUNCTION
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_ptr_l1
        FUNCTION libxsmm_ptr_l1(a)
          INTEGER(C_LONG_LONG), INTENT(IN), TARGET :: a(*)
          TYPE(C_PTR) :: libxsmm_ptr_l1
          libxsmm_ptr_l1 = C_LOC(a)
        END FUNCTION
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_ptr_l2
        FUNCTION libxsmm_ptr_l2(a)
          INTEGER(C_LONG_LONG), INTENT(IN) :: a(:,:)
          TYPE(C_PTR) :: libxsmm_ptr_l2
          libxsmm_ptr_l2 = libxsmm_ptr_l1(a)
        END FUNCTION

        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_ptr_dmm
        FUNCTION libxsmm_ptr_dmm(a)
          TYPE(LIBXSMM_DMMFUNCTION), INTENT(IN), TARGET :: a(:)
          TYPE(LIBXSMM_DMMFUNCTION), POINTER :: p
          TYPE(C_PTR) :: libxsmm_ptr_dmm
          p => a(LBOUND(a,1)); libxsmm_ptr_dmm = C_LOC(p%handle)
        END FUNCTION
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_ptr_smm
        FUNCTION libxsmm_ptr_smm(a)
          TYPE(LIBXSMM_SMMFUNCTION), INTENT(IN), TARGET :: a(:)
          TYPE(LIBXSMM_SMMFUNCTION), POINTER :: p
          TYPE(C_PTR) :: libxsmm_ptr_smm
          p => a(LBOUND(a,1)); libxsmm_ptr_smm = C_LOC(p%handle)
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
     &      kernel%handle, LIBXSMM_DATATYPE_F64,                        &
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
     &      kernel%handle, LIBXSMM_DATATYPE_F32,                        &
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

        !> Calls the kernel with the given arguments. Alternatively,
        !> PROCPOINTER can be used as shown by the inner comments
        !> of this routine (LIBXSMM_FUNCTION3). The libxsmm_xmmcall
        !> routines can be used in FORTRAN77.
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_dmmcall_abc
        SUBROUTINE libxsmm_dmmcall_abc(kernel, a, b, c)
          TYPE(LIBXSMM_DMMFUNCTION), INTENT(IN) :: kernel
          REAL(C_DOUBLE), INTENT(IN),    TARGET :: a(*), b(*)
          REAL(C_DOUBLE), INTENT(INOUT), TARGET :: c(*)
          ! PROCEDURE(LIBXSMM_FUNCTION3), POINTER :: xmm
          ! CALL C_F_PROCPOINTER(kernel%handle, xmm)
          ! CALL xmm(...)
          CALL libxsmm_xmmcall_abc(kernel%handle,                       &
     &      C_LOC(a), C_LOC(b), C_LOC(c))
        END SUBROUTINE

        !> Calls the kernel with the given arguments. Alternatively,
        !> PROCPOINTER can be used as shown by the inner comments
        !> of this routine (LIBXSMM_FUNCTION6). The libxsmm_xmmcall
        !> routines can be used in FORTRAN77.
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_dmmcall_prf
        SUBROUTINE libxsmm_dmmcall_prf(kernel, a, b, c, pa, pb, pc)
          TYPE(LIBXSMM_DMMFUNCTION), INTENT(IN) :: kernel
          REAL(C_DOUBLE), INTENT(IN),    TARGET ::  a(*), b(*)
          REAL(C_DOUBLE), INTENT(INOUT), TARGET ::  c(*)
          REAL(C_DOUBLE), INTENT(IN),    TARGET :: pa(*)
          REAL(C_DOUBLE), INTENT(IN),    TARGET :: pb(*)
          REAL(C_DOUBLE), INTENT(IN),    TARGET :: pc(*)
          ! PROCEDURE(LIBXSMM_FUNCTION6), POINTER :: xmm
          ! CALL C_F_PROCPOINTER(kernel%handle, xmm)
          ! CALL xmm(...)
          CALL libxsmm_xmmcall_prf(kernel%handle,                       &
     &      C_LOC(a),  C_LOC(b),  C_LOC(c),                             &
     &      C_LOC(pa), C_LOC(pb), C_LOC(pc))
        END SUBROUTINE

        !> See also libxsmm_dmmcall_abc and libxsmm_dmmcall_prf.
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_dmmcall
        SUBROUTINE libxsmm_dmmcall(kernel, a, b, c, pa, pb, pc)
          TYPE(LIBXSMM_DMMFUNCTION),        INTENT(IN) :: kernel
          REAL(C_DOUBLE), INTENT(IN),           TARGET ::  a(*), b(*)
          REAL(C_DOUBLE), INTENT(INOUT),        TARGET ::  c(*)
          REAL(C_DOUBLE), INTENT(IN), OPTIONAL, TARGET :: pa(*)
          REAL(C_DOUBLE), INTENT(IN), OPTIONAL, TARGET :: pb(*)
          REAL(C_DOUBLE), INTENT(IN), OPTIONAL, TARGET :: pc(*)
          ! use .OR. instead of .AND. to avoid full check
          IF (PRESENT(pa).OR.PRESENT(pb).OR.PRESENT(pc)) THEN
            CALL libxsmm_xmmcall_prf(kernel%handle,                     &
     &        C_LOC(a),  C_LOC(b),  C_LOC(c),                           &
     &        C_LOC(pa), C_LOC(pb), C_LOC(pc))
          ELSE
            CALL libxsmm_xmmcall_abc(kernel%handle,                     &
     &        C_LOC(a), C_LOC(b), C_LOC(c))
          END IF
        END SUBROUTINE

        !> Calls the kernel with the given arguments. Alternatively,
        !> PROCPOINTER can be used as shown by the inner comments
        !> of this routine (LIBXSMM_FUNCTION3). The libxsmm_xmmcall
        !> routines can be used in FORTRAN77.
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_smmcall_abc
        SUBROUTINE libxsmm_smmcall_abc(kernel, a, b, c)
          TYPE(LIBXSMM_SMMFUNCTION), INTENT(IN) :: kernel
          REAL(C_FLOAT), INTENT(IN),     TARGET :: a(*), b(*)
          REAL(C_FLOAT), INTENT(INOUT),  TARGET :: c(*)
          ! PROCEDURE(LIBXSMM_FUNCTION3), POINTER :: xmm
          ! CALL C_F_PROCPOINTER(kernel%handle, xmm)
          ! CALL xmm(...)
          CALL libxsmm_xmmcall_abc(kernel%handle,                       &
     &      C_LOC(a), C_LOC(b), C_LOC(c))
        END SUBROUTINE

        !> Calls the kernel with the given arguments. Alternatively,
        !> PROCPOINTER can be used as shown by the inner comments
        !> of this routine (LIBXSMM_FUNCTION6). The libxsmm_xmmcall
        !> routines can be used in FORTRAN77.
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_smmcall_prf
        SUBROUTINE libxsmm_smmcall_prf(kernel, a, b, c, pa, pb, pc)
          TYPE(LIBXSMM_SMMFUNCTION), INTENT(IN) :: kernel
          REAL(C_FLOAT), INTENT(IN),     TARGET ::  a(*), b(*)
          REAL(C_FLOAT), INTENT(INOUT),  TARGET ::  c(*)
          REAL(C_FLOAT), INTENT(IN),     TARGET :: pa(*)
          REAL(C_FLOAT), INTENT(IN),     TARGET :: pb(*)
          REAL(C_FLOAT), INTENT(IN),     TARGET :: pc(*)
          ! PROCEDURE(LIBXSMM_FUNCTION6), POINTER :: xmm
          ! CALL C_F_PROCPOINTER(kernel%handle, xmm)
          ! CALL xmm(...)
          CALL libxsmm_xmmcall_prf(kernel%handle,                       &
     &      C_LOC(a),  C_LOC(b),  C_LOC(c),                             &
     &      C_LOC(pa), C_LOC(pb), C_LOC(pc))
        END SUBROUTINE

        !> See also libxsmm_smmcall_abc and libxsmm_smmcall_prf.
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_smmcall
        SUBROUTINE libxsmm_smmcall(kernel, a, b, c, pa, pb, pc)
          TYPE(LIBXSMM_SMMFUNCTION),       INTENT(IN) :: kernel
          REAL(C_FLOAT), INTENT(IN),           TARGET ::  a(*), b(*)
          REAL(C_FLOAT), INTENT(INOUT),        TARGET ::  c(*)
          REAL(C_FLOAT), INTENT(IN), OPTIONAL, TARGET :: pa(*)
          REAL(C_FLOAT), INTENT(IN), OPTIONAL, TARGET :: pb(*)
          REAL(C_FLOAT), INTENT(IN), OPTIONAL, TARGET :: pc(*)
          ! use .OR. instead of .AND. to avoid full check
          IF (PRESENT(pa).OR.PRESENT(pb).OR.PRESENT(pc)) THEN
            CALL libxsmm_xmmcall_prf(kernel%handle,                     &
     &        C_LOC(a),  C_LOC(b),  C_LOC(c),                           &
     &        C_LOC(pa), C_LOC(pb), C_LOC(pc))
          ELSE
            CALL libxsmm_xmmcall_abc(kernel%handle,                     &
     &        C_LOC(a), C_LOC(b), C_LOC(c))
          END IF
        END SUBROUTINE

        !> Register user-defined key-value; value can be queried (libxsmm_xdispatch).
        !> Since the key-type is unknown to LIBXSMM, the key must be binary reproducible,
        !> i.e., a structured type (can be padded) must be initialized like a binary blob
        !> (libxsmm_xclear) followed by an element-wise initialization. The size of the
        !> key is limited (see documentation). The given value is copied by LIBXSMM and
        !> can be initialized prior to registration or whenever queried. Registered data
        !> is released when the program terminates but can be also released if needed
        !> (libxsmm_xrelease), .e.g., in case of a larger value reusing the same key.
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_xregister
        FUNCTION libxsmm_xregister(key, keysize, valsize, valinit)
          TYPE(C_PTR),    INTENT(IN), VALUE     :: key
          INTEGER(C_INT), INTENT(IN)            :: keysize, valsize
          TYPE(C_PTR),    INTENT(IN),  OPTIONAL :: valinit
          TYPE(C_PTR) :: libxsmm_xregister
          !DIR$ ATTRIBUTES OFFLOAD:MIC :: internal_xregister
          INTERFACE
            SUBROUTINE internal_xregister(regval,                       &
     &      key, keysize, valsize, valinit)                             &
     &      BIND(C, NAME="libxsmm_xregister_")
              IMPORT :: C_PTR, C_INT
              TYPE(C_PTR), INTENT(OUT) :: regval
              TYPE(C_PTR), INTENT(IN), VALUE :: key, valinit
              INTEGER(C_INT), INTENT(IN)  :: keysize, valsize
            END SUBROUTINE
          END INTERFACE
          CALL internal_xregister(libxsmm_xregister,                    &
     &      key, keysize, valsize, valinit)
        END FUNCTION

        !> Query user-defined value from LIBXSMM's code registry.
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_xdispatch
        FUNCTION libxsmm_xdispatch(key, keysize)
          TYPE(C_PTR), INTENT(IN), VALUE :: key
          INTEGER(C_INT), INTENT(IN) :: keysize
          TYPE(C_PTR) :: libxsmm_xdispatch
          !DIR$ ATTRIBUTES OFFLOAD:MIC :: internal_xdispatch
          INTERFACE
            SUBROUTINE internal_xdispatch(regval, key, keysize)         &
     &      BIND(C, NAME="libxsmm_xdispatch_")
              IMPORT :: C_PTR, C_INT
              TYPE(C_PTR), INTENT(OUT) :: regval
              TYPE(C_PTR), INTENT(IN), VALUE :: key
              INTEGER(C_INT), INTENT(IN)  :: keysize
            END SUBROUTINE
          END INTERFACE
          CALL internal_xdispatch(libxsmm_xdispatch, key, keysize)
        END FUNCTION

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
              IMPORT :: C_CHAR, C_DOUBLE, LIBXSMM_BLASINT_KIND
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
          REAL(C_DOUBLE), INTENT(IN)    :: a(*), b(*)
          REAL(C_DOUBLE), INTENT(INOUT) :: c(*)
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
          REAL(C_FLOAT), INTENT(IN)    :: a, b
          REAL(C_FLOAT), INTENT(INOUT) :: c
          !DIR$ ATTRIBUTES OFFLOAD:MIC :: internal_gemm
          INTERFACE
            PURE SUBROUTINE internal_gemm(transa, transb, m, n, k,      &
     &      alpha, a, lda, b, ldb, beta, c, ldc)                        &
     &      BIND(C, NAME="libxsmm_sgemm_")
              IMPORT :: C_CHAR, C_FLOAT, LIBXSMM_BLASINT_KIND
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
          REAL(C_FLOAT), INTENT(IN)    :: a(*), b(*)
          REAL(C_FLOAT), INTENT(INOUT) :: c(*)
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
          REAL(C_DOUBLE), INTENT(IN)    :: a, b
          REAL(C_DOUBLE), INTENT(INOUT) :: c
          !DIR$ ATTRIBUTES OFFLOAD:MIC :: internal_gemm
          INTERFACE
            PURE SUBROUTINE internal_gemm(transa, transb, m, n, k,      &
     &      alpha, a, lda, b, ldb, beta, c, ldc)                        &
     &      BIND(C, NAME="libxsmm_blas_dgemm_")
              IMPORT :: C_CHAR, C_DOUBLE, LIBXSMM_BLASINT_KIND
              CHARACTER(C_CHAR), INTENT(IN) :: transa, transb
              INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: m, n, k
              INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: lda
              INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: ldb
              INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: ldc
              REAL(C_DOUBLE), INTENT(IN)    :: alpha, beta
              REAL(C_DOUBLE), INTENT(IN)    :: a, b
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
          REAL(C_DOUBLE), INTENT(IN)    :: a(*), b(*)
          REAL(C_DOUBLE), INTENT(INOUT) :: c(*)
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
          REAL(C_FLOAT), INTENT(IN)    :: a, b
          REAL(C_FLOAT), INTENT(INOUT) :: c
          !DIR$ ATTRIBUTES OFFLOAD:MIC :: internal_gemm
          INTERFACE
            PURE SUBROUTINE internal_gemm(transa, transb, m, n, k,      &
     &      alpha, a, lda, b, ldb, beta, c, ldc)                        &
     &      BIND(C, NAME="libxsmm_blas_sgemm_")
              IMPORT :: C_CHAR, C_FLOAT, LIBXSMM_BLASINT_KIND
              CHARACTER(C_CHAR), INTENT(IN) :: transa, transb
              INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: m, n, k
              INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: lda
              INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: ldb
              INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: ldc
              REAL(C_FLOAT), INTENT(IN)    :: alpha, beta
              REAL(C_FLOAT), INTENT(IN)    :: a, b
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
          REAL(C_FLOAT), INTENT(IN)    :: a(*), b(*)
          REAL(C_FLOAT), INTENT(INOUT) :: c(*)
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
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_matcopy_p0
        PURE SUBROUTINE libxsmm_matcopy_p0(output, input, typesize,     &
     &  m, n, ldi, ldo)
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: m
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN),                    &
     &                                OPTIONAL :: n, ldi, ldo
          INTEGER(C_INT), INTENT(IN) :: typesize
          TYPE(C_PTR), INTENT(IN), OPTIONAL :: input
          TYPE(C_PTR), INTENT(IN) :: output
          CALL libxsmm_xmatcopy(output, input, typesize,                &
     &      m, n, ldi, ldo)
        END SUBROUTINE

        !> Matrix-copy (2-dimensional copy) routine (DP/rank-1).
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_matcopy_d1
        SUBROUTINE libxsmm_matcopy_d1(output, input, m, n, ldi, ldo)
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: m
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN), OPTIONAL :: n
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN), OPTIONAL :: ldi
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN), OPTIONAL :: ldo
          REAL(C_DOUBLE), INTENT(OUT),          TARGET :: output(*)
          REAL(C_DOUBLE), INTENT(IN), OPTIONAL, TARGET ::  input(*)
          CALL libxsmm_xmatcopy(C_LOC(output), C_LOC(input), 8,         &
     &      m, n, ldi, ldo)
        END SUBROUTINE

        !> Matrix-copy (2-dimensional copy) routine (DP/rank-2).
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_matcopy_d2
        SUBROUTINE libxsmm_matcopy_d2(output, input, m, n, ldi, ldo)
          INTEGER(LIBXSMM_BLASINT_KIND),    INTENT(IN) :: m, n, ldi, ldo
          REAL(C_DOUBLE), INTENT(OUT),          TARGET :: output(ldo,*)
          REAL(C_DOUBLE), INTENT(IN), OPTIONAL, TARGET ::  input(ldi,*)
          CALL libxsmm_xmatcopy(C_LOC(output), C_LOC(input), 8,         &
     &      m, n, ldi, ldo)
        END SUBROUTINE

        !> Matrix-copy (2-dimensional copy) routine (SP/rank-1).
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_matcopy_s1
        SUBROUTINE libxsmm_matcopy_s1(output, input, m, n, ldi, ldo)
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: m
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN), OPTIONAL :: n
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN), OPTIONAL :: ldi
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN), OPTIONAL :: ldo
          REAL(C_FLOAT),  INTENT(OUT),          TARGET :: output(*)
          REAL(C_FLOAT),  INTENT(IN), OPTIONAL, TARGET ::  input(*)
          CALL libxsmm_xmatcopy(C_LOC(output), C_LOC(input), 4,         &
     &      m, n, ldi, ldo)
        END SUBROUTINE

        !> Matrix-copy (2-dimensional copy) routine (SP/rank-2).
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_matcopy_s2
        SUBROUTINE libxsmm_matcopy_s2(output, input, m, n, ldi, ldo)
          INTEGER(LIBXSMM_BLASINT_KIND),    INTENT(IN) :: m, n, ldi, ldo
          REAL(C_FLOAT),  INTENT(OUT),          TARGET :: output(ldo,*)
          REAL(C_FLOAT),  INTENT(IN), OPTIONAL, TARGET ::  input(ldi,*)
          CALL libxsmm_xmatcopy(C_LOC(output), C_LOC(input), 4,         &
     &      m, n, ldi, ldo)
        END SUBROUTINE

        !> Transpose a matrix (in-place form).
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_itrans_p0
        PURE SUBROUTINE libxsmm_itrans_p0(matrix, typesize,             &
     &  m, n, ldi, ldo)
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: m
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN), OPTIONAL :: n
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN), OPTIONAL :: ldi
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN), OPTIONAL :: ldo
          TYPE(C_PTR),    INTENT(IN) :: matrix
          INTEGER(C_INT), INTENT(IN) :: typesize
          CALL libxsmm_xitrans(matrix, typesize, m, n, ldi, ldo)
        END SUBROUTINE

        !> Transpose a matrix (in-place form, DP/rank-1).
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_itrans_d1
        SUBROUTINE libxsmm_itrans_d1(matrix, m, n, ldi, ldo)
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: m
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN), OPTIONAL :: n
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN), OPTIONAL :: ldi
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN), OPTIONAL :: ldo
          REAL(C_DOUBLE), INTENT(INOUT), TARGET :: matrix(*)
          CALL libxsmm_xitrans(C_LOC(matrix), 8, m, n, ldi, ldo)
        END SUBROUTINE

        !> Transpose a matrix (in-place form, DP/rank-2).
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_itrans_d2
        SUBROUTINE libxsmm_itrans_d2(matrix, m, n, ld)
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: m, n, ld
          REAL(C_DOUBLE), INTENT(INOUT), TARGET :: matrix(ld,*)
          CALL libxsmm_xitrans(C_LOC(matrix), 8, m, n, ld, ld)
        END SUBROUTINE

        !> Transpose a matrix (in-place form, SP/rank-1).
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_itrans_s1
        SUBROUTINE libxsmm_itrans_s1(matrix, m, n, ldi, ldo)
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: m
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN), OPTIONAL :: n
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN), OPTIONAL :: ldi
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN), OPTIONAL :: ldo
          REAL(C_FLOAT), INTENT(INOUT), TARGET :: matrix(*)
          CALL libxsmm_xitrans(C_LOC(matrix), 4, m, n, ldi, ldo)
        END SUBROUTINE

        !> Transpose a matrix (in-place form, SP/rank-2).
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_itrans_s2
        SUBROUTINE libxsmm_itrans_s2(matrix, m, n, ld)
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: m, n, ld
          REAL(C_FLOAT), INTENT(INOUT), TARGET :: matrix(ld,*)
          CALL libxsmm_xitrans(C_LOC(matrix), 4, m, n, ld, ld)
        END SUBROUTINE

        !> Transpose a matrix (out-of-place form).
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_otrans_p0
        PURE SUBROUTINE libxsmm_otrans_p0(output, input, typesize,      &
     &  m, n, ldi, ldo)
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: m
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN), OPTIONAL :: n
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN), OPTIONAL :: ldi
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN), OPTIONAL :: ldo
          TYPE(C_PTR),    INTENT(IN) :: output, input
          INTEGER(C_INT), INTENT(IN) :: typesize
          CALL libxsmm_xotrans(output, input, typesize, m, n, ldi, ldo)
        END SUBROUTINE

        !> Transpose a matrix (out-of-place form, DP/rank-1).
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_otrans_d1
        SUBROUTINE libxsmm_otrans_d1(output, input, m, n, ldi, ldo)
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: m
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN), OPTIONAL :: n
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN), OPTIONAL :: ldi
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN), OPTIONAL :: ldo
          REAL(C_DOUBLE), INTENT(OUT), TARGET :: output(*)
          REAL(C_DOUBLE), INTENT(IN),  TARGET ::  input(*)
          CALL libxsmm_xotrans(C_LOC(output), C_LOC(input),             &
     &      8, m, n, ldi, ldo)
        END SUBROUTINE

        !> Transpose a matrix (out-of-place form, DP/rank-2).
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_otrans_d2
        SUBROUTINE libxsmm_otrans_d2(output, input, m, n, ldi, ldo)
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: m, n, ldi, ldo
          REAL(C_DOUBLE), INTENT(OUT), TARGET :: output(ldo,*)
          REAL(C_DOUBLE), INTENT(IN),  TARGET ::  input(ldi,*)
          CALL libxsmm_xotrans(C_LOC(output), C_LOC(input),             &
     &      8, m, n, ldi, ldo)
        END SUBROUTINE

        !> Transpose a matrix (out-of-place form, SP/rank-1).
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_otrans_s1
        SUBROUTINE libxsmm_otrans_s1(output, input, m, n, ldi, ldo)
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: m
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN), OPTIONAL :: n
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN), OPTIONAL :: ldi
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN), OPTIONAL :: ldo
          REAL(C_FLOAT), INTENT(OUT), TARGET :: output(*)
          REAL(C_FLOAT), INTENT(IN),  TARGET ::  input(*)
          CALL libxsmm_xotrans(C_LOC(output), C_LOC(input),             &
     &      4, m, n, ldi, ldo)
        END SUBROUTINE

        !> Transpose a matrix (out-of-place form, SP/rank-2).
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_otrans_s2
        SUBROUTINE libxsmm_otrans_s2(output, input, m, n, ldi, ldo)
          INTEGER(LIBXSMM_BLASINT_KIND), INTENT(IN) :: m, n, ldi, ldo
          REAL(C_FLOAT), INTENT(OUT), TARGET :: output(ldo,*)
          REAL(C_FLOAT), INTENT(IN),  TARGET ::  input(ldi,*)
          CALL libxsmm_xotrans(C_LOC(output), C_LOC(input),             &
     &      4, m, n, ldi, ldo)
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
              IMPORT :: LIBXSMM_TICKINT_KIND
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
              IMPORT :: LIBXSMM_MATDIFF_INFO, LIBXSMM_BLASINT_KIND
              IMPORT :: C_PTR, C_INT
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
              IMPORT :: C_LONG_LONG, C_INT
              INTEGER(C_LONG_LONG), INTENT(OUT) :: coprime
              INTEGER(C_INT), INTENT(IN) :: n
            END SUBROUTINE
          END INTERFACE
          libxsmm_shuffle = INT(0, KIND=C_LONG_LONG) ! avoid warning (older CRAY)
          CALL internal_shuffle(libxsmm_shuffle, n)
        END FUNCTION

        !> Calculates a hash value for the given array and seed.
        !> FORTRAN 77: see libxsmm_xhash
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_hash_char
        FUNCTION libxsmm_hash_char(key, seed)
          CHARACTER(C_CHAR), INTENT(IN)$CONTIGUOUS :: key(:)
          INTEGER(C_INT), INTENT(IN) :: seed
          INTEGER(C_INT) :: libxsmm_hash_char
          libxsmm_hash_char = seed
          CALL libxsmm_xhash(libxsmm_hash_char,                         &
     &      libxsmm_ptr(key), SIZE(key))
        END FUNCTION

        !> Calculates a hash value for the given array and seed.
        !> FORTRAN 77: see libxsmm_xhash
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_hash_i8
        FUNCTION libxsmm_hash_i8(key, seed)
          INTEGER(C_INT8_T), INTENT(IN)$CONTIGUOUS :: key(:)
          INTEGER(C_INT), INTENT(IN) :: seed
          INTEGER(C_INT) :: libxsmm_hash_i8
          libxsmm_hash_i8 = seed
          CALL libxsmm_xhash(libxsmm_hash_i8,                           &
     &      libxsmm_ptr(key), SIZE(key))
        END FUNCTION

        !> Calculates a hash value for the given array and seed.
        !> FORTRAN 77: see libxsmm_xhash
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_hash_i32
        FUNCTION libxsmm_hash_i32(key, seed)
          INTEGER(C_INT), INTENT(IN)$CONTIGUOUS :: key(:)
          INTEGER(C_INT), INTENT(IN) :: seed
          INTEGER(C_INT) :: libxsmm_hash_i32
          libxsmm_hash_i32 = seed
          CALL libxsmm_xhash(libxsmm_hash_i32,                          &
     &      libxsmm_ptr(key), SIZE(key) * 4)
        END FUNCTION

        !> Calculates a hash value for the given array and seed.
        !> FORTRAN 77: see libxsmm_xhash
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_hash_i64
        FUNCTION libxsmm_hash_i64(key, seed)
          INTEGER(C_LONG_LONG), INTENT(IN)$CONTIGUOUS :: key(:)
          INTEGER(C_INT), INTENT(IN) :: seed
          INTEGER(C_INT) :: libxsmm_hash_i64
          libxsmm_hash_i64 = seed
          CALL libxsmm_xhash(libxsmm_hash_i64,                          &
     &      libxsmm_ptr(key), SIZE(key) * 8)
        END FUNCTION

        !> Calculates if there is a difference between two arrays.
        !> FORTRAN 77: see libxsmm_xdiff
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_diff_char
        FUNCTION libxsmm_diff_char(a, b)
          CHARACTER(C_CHAR), INTENT(IN)$CONTIGUOUS :: a(:), b(:)
          LOGICAL(C_BOOL) :: libxsmm_diff_char
          IF (SIZE(a, KIND=C_LONG_LONG) .EQ. SIZE(b, KIND=C_LONG_LONG)) &
     &    THEN
            CALL libxsmm_xdiff(libxsmm_diff_char,                       &
     &        libxsmm_ptr(a), libxsmm_ptr(b),                           &
     &        SIZE(a, KIND=C_LONG_LONG))
          ELSE
            libxsmm_diff_char = LOGICAL(.TRUE., KIND=C_BOOL)
          END IF
        END FUNCTION

        !> Calculates if there is a difference between two arrays.
        !> FORTRAN 77: see libxsmm_xdiff
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_diff_i8
        FUNCTION libxsmm_diff_i8(a, b)
          INTEGER(C_INT8_T), INTENT(IN)$CONTIGUOUS :: a(:), b(:)
          LOGICAL(C_BOOL) :: libxsmm_diff_i8
          IF (SIZE(a, KIND=C_LONG_LONG) .EQ. SIZE(b, KIND=C_LONG_LONG)) &
     &    THEN
            CALL libxsmm_xdiff(libxsmm_diff_i8,                         &
     &        libxsmm_ptr(a), libxsmm_ptr(b),                           &
     &        SIZE(a, KIND=C_LONG_LONG))
          ELSE
            libxsmm_diff_i8 = LOGICAL(.TRUE., KIND=C_BOOL)
          END IF
        END FUNCTION

        !> Calculates if there is a difference between two arrays.
        !> FORTRAN 77: see libxsmm_xdiff
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_diff_i32
        FUNCTION libxsmm_diff_i32(a, b)
          INTEGER(C_INT), INTENT(IN)$CONTIGUOUS :: a(:), b(:)
          LOGICAL(C_BOOL) :: libxsmm_diff_i32
          IF (SIZE(a, KIND=C_LONG_LONG) .EQ. SIZE(b, KIND=C_LONG_LONG)) &
     &    THEN
            CALL libxsmm_xdiff(libxsmm_diff_i32,                        &
     &        libxsmm_ptr(a), libxsmm_ptr(b),                           &
     &        SIZE(a, KIND=C_LONG_LONG) * INT(4, KIND=C_LONG_LONG))
          ELSE
            libxsmm_diff_i32 = LOGICAL(.TRUE., KIND=C_BOOL)
          END IF
        END FUNCTION

        !> Calculates if there is a difference between two arrays.
        !> FORTRAN 77: see libxsmm_xdiff
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_diff_i64
        FUNCTION libxsmm_diff_i64(a, b)
          INTEGER(C_LONG_LONG), INTENT(IN)$CONTIGUOUS :: a(:), b(:)
          LOGICAL(C_BOOL) :: libxsmm_diff_i64
          IF (SIZE(a, KIND=C_LONG_LONG) .EQ. SIZE(b, KIND=C_LONG_LONG)) &
     &    THEN
            CALL libxsmm_xdiff(libxsmm_diff_i64,                        &
     &        libxsmm_ptr(a), libxsmm_ptr(b),                           &
     &        SIZE(a, KIND=C_LONG_LONG) * INT(8, KIND=C_LONG_LONG))
          ELSE
            libxsmm_diff_i64 = LOGICAL(.TRUE., KIND=C_BOOL)
          END IF
        END FUNCTION

        !> Check if location is SIMD-aligned and optionally consider the next
        !> access as if reached by incrementing the location (in Bytes).
        !> Optionally calculates the alignment of the given location in Bytes.
        !DIR$ ATTRIBUTES OFFLOAD:MIC :: libxsmm_aligned
        FUNCTION libxsmm_aligned(location, increment, alignment)
          TYPE(C_PTR), INTENT(IN), VALUE :: location
          INTEGER(C_INT),  INTENT(IN), OPTIONAL :: increment
          INTEGER(C_INT), INTENT(OUT), OPTIONAL :: alignment
          LOGICAL :: libxsmm_aligned ! C_BOOL (GNU Fortran issue)
          INTEGER(C_INT) :: aligned
          !DIR$ ATTRIBUTES OFFLOAD:MIC :: internal_aligned
          INTERFACE
            SUBROUTINE internal_aligned(is_aligned, location,           &
     &      increment, alignment) BIND(C, NAME="libxsmm_aligned_")
              IMPORT :: C_PTR, C_INT, C_BOOL
              TYPE(C_PTR), VALUE, INTENT(IN) :: location
              INTEGER(C_INT),     INTENT(IN) :: increment
              INTEGER(C_INT),    INTENT(OUT) :: alignment
              INTEGER(C_INT),    INTENT(OUT) :: is_aligned ! C_BOOL
            END SUBROUTINE
          END INTERFACE
          CALL internal_aligned(aligned, location, increment, alignment)
          libxsmm_aligned = 0.NE.aligned
        END FUNCTION
      END MODULE

