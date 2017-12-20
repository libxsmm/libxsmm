/******************************************************************************
** Copyright (c) 2015-2018, Intel Corporation                                **
** All rights reserved.                                                      **
**                                                                           **
** Redistribution and use in source and binary forms, with or without        **
** modification, are permitted provided that the following conditions        **
** are met:                                                                  **
** 1. Redistributions of source code must retain the above copyright         **
**    notice, this list of conditions and the following disclaimer.          **
** 2. Redistributions in binary form must reproduce the above copyright      **
**    notice, this list of conditions and the following disclaimer in the    **
**    documentation and/or other materials provided with the distribution.   **
** 3. Neither the name of the copyright holder nor the names of its          **
**    contributors may be used to endorse or promote products derived        **
**    from this software without specific prior written permission.          **
**                                                                           **
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS       **
** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT         **
** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR     **
** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT      **
** HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,    **
** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED  **
** TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR    **
** PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF    **
** LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING      **
** NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS        **
** SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.              **
******************************************************************************/
/* Hans Pabst (Intel Corp.)
******************************************************************************/
#ifndef LIBXSMM_FRONTEND_H
#define LIBXSMM_FRONTEND_H

#include "libxsmm_typedefs.h"

/** Helper macros for eliding prefetch address calculations depending on prefetch scheme. */
#if 0 != ((LIBXSMM_PREFETCH) & 2/*AL2*/) \
 || 0 != ((LIBXSMM_PREFETCH) & 4/*AL2_JPST*/) \
 || 0 != ((LIBXSMM_PREFETCH) & 16/*AL2_AHEAD*/) \
 || 0 != ((LIBXSMM_PREFETCH) & 32/*AL1*/)
# define LIBXSMM_PREFETCH_A(EXPR) (EXPR)
#endif
#if 0 != ((LIBXSMM_PREFETCH) & 8/*BL2_VIA_C*/) \
 || 0 != ((LIBXSMM_PREFETCH) & 64/*BL1*/)
# define LIBXSMM_PREFETCH_B(EXPR) (EXPR)
#endif
#if 0 != ((LIBXSMM_PREFETCH) & 128/*CL1*/)
# define LIBXSMM_PREFETCH_C(EXPR) (EXPR)
#endif
/** Secondary helper macros derived from the above group. */
#if defined(LIBXSMM_PREFETCH_A)
# define LIBXSMM_NOPREFETCH_A(EXPR)
#else
# define LIBXSMM_NOPREFETCH_A(EXPR) EXPR
# define LIBXSMM_PREFETCH_A(EXPR) 0
#endif
#if defined(LIBXSMM_PREFETCH_B)
# define LIBXSMM_NOPREFETCH_B(EXPR)
#else
# define LIBXSMM_NOPREFETCH_B(EXPR) EXPR
# define LIBXSMM_PREFETCH_B(EXPR) 0
#endif
#if defined(LIBXSMM_PREFETCH_C)
# define LIBXSMM_NOPREFETCH_C(EXPR)
#else
# define LIBXSMM_NOPREFETCH_C(EXPR) EXPR
# define LIBXSMM_PREFETCH_C(EXPR) 0
#endif

/** Helper macro for BLAS-style prefixes. */
#define LIBXSMM_TPREFIX_NAME(TYPE) LIBXSMM_CONCATENATE(LIBXSMM_TPREFIX_, TYPE)
#define LIBXSMM_TPREFIX(TYPE, SYMBOL) LIBXSMM_CONCATENATE(LIBXSMM_TPREFIX_NAME(TYPE), SYMBOL)
#define LIBXSMM_TPREFIX_double d
#define LIBXSMM_TPREFIX_float s
#define LIBXSMM_TPREFIX_short w

/** Helper macro for type postfixes. */
#define LIBXSMM_TPOSTFIX_NAME(TYPE) LIBXSMM_CONCATENATE(LIBXSMM_TPOSTFIX_, TYPE)
#define LIBXSMM_TPOSTFIX(TYPE, SYMBOL) LIBXSMM_CONCATENATE(SYMBOL, LIBXSMM_TPOSTFIX_NAME(TYPE))
#define LIBXSMM_TPOSTFIX_double F64
#define LIBXSMM_TPOSTFIX_float F32
#define LIBXSMM_TPOSTFIX_int I32
#define LIBXSMM_TPOSTFIX_short I16

/** Helper macro for comparing types. */
#define LIBXSMM_EQUAL_CHECK(...) LIBXSMM_SELECT_HEAD(__VA_ARGS__, 0)
#define LIBXSMM_EQUAL(T1, T2) LIBXSMM_EQUAL_CHECK(LIBXSMM_CONCATENATE(LIBXSMM_CONCATENATE(LIBXSMM_EQUAL_, T1), T2))
#define LIBXSMM_EQUAL_floatfloat 1
#define LIBXSMM_EQUAL_doubledouble 1
#define LIBXSMM_EQUAL_floatdouble 0
#define LIBXSMM_EQUAL_doublefloat 0

/** Check ILP64 configuration for sanity. */
#if !defined(LIBXSMM_ILP64) || (0 == LIBXSMM_ILP64 && defined(MKL_ILP64))
# error "Inconsistent ILP64 configuration detected!"
#elif (0 != LIBXSMM_ILP64 && !defined(MKL_ILP64))
# define MKL_ILP64
#endif
#if (0 != LIBXSMM_ILP64)
# define LIBXSMM_BLASINT_NBITS 64
# define LIBXSMM_BLASINT long long
#else /* LP64 */
# define LIBXSMM_BLASINT_NBITS 32
# define LIBXSMM_BLASINT int
#endif

/** Integer type for LAPACK/BLAS (LP64: 32-bit, and ILP64: 64-bit). */
typedef LIBXSMM_BLASINT libxsmm_blasint;

/** MKL_DIRECT_CALL requires to include the MKL interface. */
#if defined(MKL_DIRECT_CALL_SEQ) || defined(MKL_DIRECT_CALL)
# if (0 != LIBXSMM_ILP64 && !defined(MKL_ILP64))
#   error "Inconsistent ILP64 configuration detected!"
# endif
# if defined(LIBXSMM_OFFLOAD_BUILD)
#   pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#   include <mkl.h>
#   pragma offload_attribute(pop)
# else
#   include <mkl.h>
# endif
#endif

/** GEMM: fall-back prototype functions served by any compliant LAPACK/BLAS. */
typedef LIBXSMM_RETARGETABLE void (*libxsmm_sgemm_function)(
  const char*, const char*, const libxsmm_blasint*, const libxsmm_blasint*, const libxsmm_blasint*,
  const float*, const float*, const libxsmm_blasint*, const float*, const libxsmm_blasint*,
  const float*, float*, const libxsmm_blasint*);
typedef LIBXSMM_RETARGETABLE void (*libxsmm_dgemm_function)(
  const char*, const char*, const libxsmm_blasint*, const libxsmm_blasint*, const libxsmm_blasint*,
  const double*, const double*, const libxsmm_blasint*, const double*, const libxsmm_blasint*,
  const double*, double*, const libxsmm_blasint*);

/** GEMV: fall-back prototype functions served by any compliant LAPACK/BLAS. */
typedef LIBXSMM_RETARGETABLE void(*libxsmm_sgemv_function)(
  const char*, const libxsmm_blasint*, const libxsmm_blasint*,
  const float*, const float*, const libxsmm_blasint*, const float*, const libxsmm_blasint*,
  const float*, float*, const libxsmm_blasint*);
typedef LIBXSMM_RETARGETABLE void(*libxsmm_dgemv_function)(
  const char*, const libxsmm_blasint*, const libxsmm_blasint*,
  const double*, const double*, const libxsmm_blasint*, const double*, const libxsmm_blasint*,
  const double*, double*, const libxsmm_blasint*);

#if defined(LIBXSMM_BUILD_EXT)
# define LIBXSMM_WEAK
# define LIBXSMM_EXT_WEAK LIBXSMM_ATTRIBUTE_WEAK
#else
# define LIBXSMM_WEAK LIBXSMM_ATTRIBUTE_WEAK
# define LIBXSMM_EXT_WEAK
#endif
#if defined(LIBXSMM_BUILD) && defined(__STATIC) /*&& defined(LIBXSMM_GEMM_WRAP)*/
# define LIBXSMM_GEMM_WEAK LIBXSMM_WEAK
# define LIBXSMM_EXT_GEMM_WEAK LIBXSMM_EXT_WEAK
#else
# define LIBXSMM_GEMM_WEAK
# define LIBXSMM_EXT_GEMM_WEAK
#endif

/** The original GEMM functions (SGEMM and DGEMM). */
LIBXSMM_API LIBXSMM_GEMM_WEAK libxsmm_sgemm_function libxsmm_original_sgemm(const char* caller);
LIBXSMM_API LIBXSMM_GEMM_WEAK libxsmm_dgemm_function libxsmm_original_dgemm(const char* caller);

/** Construct symbol name from a given real type name (float, double and short). */
#define LIBXSMM_DATATYPE(TYPE)          LIBXSMM_TPOSTFIX(TYPE, LIBXSMM_DATATYPE_)
#define LIBXSMM_GEMM_PRECISION(TYPE)    LIBXSMM_TPOSTFIX(TYPE, LIBXSMM_GEMM_PRECISION_)
#define LIBXSMM_GEMM_SYMBOL(TYPE)       LIBXSMM_FSYMBOL(LIBXSMM_TPREFIX(TYPE, gemm))
#define LIBXSMM_GEMV_SYMBOL(TYPE)       LIBXSMM_FSYMBOL(LIBXSMM_TPREFIX(TYPE, gemv))
#define LIBXSMM_GEMMFUNCTION_TYPE(TYPE) LIBXSMM_CONCATENATE(libxsmm_, LIBXSMM_TPREFIX(TYPE, gemm_function))
#define LIBXSMM_GEMVFUNCTION_TYPE(TYPE) LIBXSMM_CONCATENATE(libxsmm_, LIBXSMM_TPREFIX(TYPE, gemv_function))
#define LIBXSMM_ORIGINAL_GEMM(TYPE)     LIBXSMM_CONCATENATE(libxsmm_original_, LIBXSMM_TPREFIX(TYPE, gemm))(LIBXSMM_CALLER)
#define LIBXSMM_MMFUNCTION_TYPE(TYPE)   LIBXSMM_CONCATENATE(libxsmm_, LIBXSMM_TPREFIX(TYPE, mmfunction))
#define LIBXSMM_MMDISPATCH_SYMBOL(TYPE) LIBXSMM_CONCATENATE(libxsmm_, LIBXSMM_TPREFIX(TYPE, mmdispatch))
#define LIBXSMM_XBLAS_SYMBOL(TYPE)      LIBXSMM_CONCATENATE(libxsmm_blas_, LIBXSMM_TPREFIX(TYPE, gemm))
#define LIBXSMM_XGEMM_SYMBOL(TYPE)      LIBXSMM_CONCATENATE(libxsmm_, LIBXSMM_TPREFIX(TYPE, gemm))
#define LIBXSMM_YGEMM_SYMBOL(TYPE)      LIBXSMM_CONCATENATE(LIBXSMM_XGEMM_SYMBOL(TYPE), _omp)

#if defined(LIBXSMM_GEMM_CONST)
# undef LIBXSMM_GEMM_CONST
# define LIBXSMM_GEMM_CONST const
#elif defined(LIBXSMM_GEMM_NONCONST) || defined(__OPENBLAS)
# define LIBXSMM_GEMM_CONST
#else
# define LIBXSMM_GEMM_CONST const
#endif

#define LIBXSMM_GEMM_SYMBOL_DECL(CONST, TYPE) \
  LIBXSMM_API_EXTERN void LIBXSMM_GEMM_SYMBOL(TYPE)(CONST char*, CONST char*, \
    CONST libxsmm_blasint*, CONST libxsmm_blasint*, CONST libxsmm_blasint*, \
    CONST TYPE*, CONST TYPE*, CONST libxsmm_blasint*, \
    CONST TYPE*, CONST libxsmm_blasint*, \
    CONST TYPE*, TYPE*, CONST libxsmm_blasint*)

/** Helper macro consolidating the transpose requests into a set of flags. */
#define LIBXSMM_GEMM_FLAGS(TRANSA, TRANSB) /* check for N/n rather than T/t since C/c is also valid! */ \
   ((('N' == (TRANSA) || 'n' == (TRANSA)) ? LIBXSMM_GEMM_FLAG_NONE : LIBXSMM_GEMM_FLAG_TRANS_A) \
  | (('N' == (TRANSB) || 'n' == (TRANSB)) ? LIBXSMM_GEMM_FLAG_NONE : LIBXSMM_GEMM_FLAG_TRANS_B))

/** Helper macro allowing NULL-requests (transposes) supplied by some default. */
#define LIBXSMM_GEMM_PFLAGS(TRANSA, TRANSB, DEFAULT) LIBXSMM_GEMM_FLAGS( \
  0 != ((const void*)(TRANSA)) ? *((const char*)(TRANSA)) : (0 == (LIBXSMM_GEMM_FLAG_TRANS_A & (DEFAULT)) ? 'N' : 'T'), \
  0 != ((const void*)(TRANSB)) ? *((const char*)(TRANSB)) : (0 == (LIBXSMM_GEMM_FLAG_TRANS_B & (DEFAULT)) ? 'N' : 'T')) \
  | (~(LIBXSMM_GEMM_FLAG_TRANS_A | LIBXSMM_GEMM_FLAG_TRANS_B) & (DEFAULT))

/** BLAS-based GEMM supplied by the linked LAPACK/BLAS library (template). */
#if !defined(__BLAS) || (0 != __BLAS)
# define LIBXSMM_BLAS_XGEMM(TYPE, FLAGS, MM, NN, KK, ALPHA, A, LDA, B, LDB, BETA, C, LDC) { \
    const char libxsmm_blas_xgemm_transa_ = (char)(0 == (LIBXSMM_GEMM_FLAG_TRANS_A & (FLAGS)) ? 'N' : 'T'); \
    const char libxsmm_blas_xgemm_transb_ = (char)(0 == (LIBXSMM_GEMM_FLAG_TRANS_B & (FLAGS)) ? 'N' : 'T'); \
    const TYPE libxsmm_blas_xgemm_alpha_ = (TYPE)(ALPHA), libxsmm_blas_xgemm_beta_ = (TYPE)(BETA); \
    const libxsmm_blasint libxsmm_blas_xgemm_lda_ = (libxsmm_blasint)(LDA); \
    const libxsmm_blasint libxsmm_blas_xgemm_ldb_ = (libxsmm_blasint)(LDB); \
    const libxsmm_blasint libxsmm_blas_xgemm_ldc_ = (libxsmm_blasint)(LDC); \
    const libxsmm_blasint libxsmm_blas_xgemm_m_ = (libxsmm_blasint)(MM); \
    const libxsmm_blasint libxsmm_blas_xgemm_n_ = (libxsmm_blasint)(NN); \
    const libxsmm_blasint libxsmm_blas_xgemm_k_ = (libxsmm_blasint)(KK); \
    LIBXSMM_ASSERT(0 != ((uintptr_t)LIBXSMM_ORIGINAL_GEMM(TYPE))); \
    LIBXSMM_ORIGINAL_GEMM(TYPE)(&libxsmm_blas_xgemm_transa_, &libxsmm_blas_xgemm_transb_, \
      &libxsmm_blas_xgemm_m_, &libxsmm_blas_xgemm_n_, &libxsmm_blas_xgemm_k_, \
      &libxsmm_blas_xgemm_alpha_, (const TYPE*)(A), &libxsmm_blas_xgemm_lda_, \
                                  (const TYPE*)(B), &libxsmm_blas_xgemm_ldb_, \
       &libxsmm_blas_xgemm_beta_, (TYPE*)(C), &libxsmm_blas_xgemm_ldc_); \
  }
#else
# define LIBXSMM_BLAS_XGEMM(TYPE, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) \
    LIBXSMM_UNUSED(LDA); LIBXSMM_UNUSED(LDB); LIBXSMM_UNUSED(LDC); \
    LIBXSMM_UNUSED(M); LIBXSMM_UNUSED(N); LIBXSMM_UNUSED(K); \
    LIBXSMM_UNUSED(A); LIBXSMM_UNUSED(B); LIBXSMM_UNUSED(C); \
    LIBXSMM_UNUSED(ALPHA); LIBXSMM_UNUSED(BETA); \
    LIBXSMM_UNUSED(FLAGS)
#endif

/** BLAS-based GEMM supplied by the linked LAPACK/BLAS library (single-precision). */
#define LIBXSMM_BLAS_SGEMM(FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) \
  LIBXSMM_BLAS_XGEMM(float, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
/** BLAS-based GEMM supplied by the linked LAPACK/BLAS library (single-precision). */
#define LIBXSMM_BLAS_DGEMM(FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) \
  LIBXSMM_BLAS_XGEMM(double, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
/** BLAS-based GEMM supplied by the linked LAPACK/BLAS library. */
#define LIBXSMM_BLAS_GEMM(FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) { \
  if (sizeof(double) == sizeof(*(A)) /*always true:*/&& 0 != (LDC)) { \
    LIBXSMM_BLAS_DGEMM(FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC); \
  } \
  else { \
    LIBXSMM_BLAS_SGEMM(FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC); \
  } \
}

/** Inlinable GEMM exercising the compiler's code generation (template). */
#if defined(MKL_DIRECT_CALL_SEQ) || defined(MKL_DIRECT_CALL)
# define LIBXSMM_INLINE_XGEMM(TYPE, INT, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) \
  LIBXSMM_BLAS_XGEMM(TYPE, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
#else
# define LIBXSMM_INLINE_XGEMM(TYPE, INT, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) { \
  const TYPE libxsmm_inline_xgemm_alpha_ = (TYPE)(ALPHA), libxsmm_inline_xgemm_beta_ = (TYPE)(BETA); \
  INT libxsmm_inline_xgemm_i_, libxsmm_inline_xgemm_j_, libxsmm_inline_xgemm_k_; \
  LIBXSMM_UNUSED(FLAGS); /* TODO: remove/adjust precondition if anything other than NN is supported */ \
  LIBXSMM_ASSERT(0 == (LIBXSMM_GEMM_FLAG_TRANS_A & (FLAGS)) && 0 == (LIBXSMM_GEMM_FLAG_TRANS_B & (FLAGS))/*not supported*/); \
  LIBXSMM_ASSERT((M) <= (LDA) && (K) <= (LDB) && (M) <= (LDC)); \
  LIBXSMM_PRAGMA_SIMD \
  for (libxsmm_inline_xgemm_j_ = 0; libxsmm_inline_xgemm_j_ < ((INT)(M)); ++libxsmm_inline_xgemm_j_) { \
    LIBXSMM_PRAGMA_LOOP_COUNT(1, LIBXSMM_MAX_K, LIBXSMM_AVG_K) \
    for (libxsmm_inline_xgemm_k_ = 0; libxsmm_inline_xgemm_k_ < (K); ++libxsmm_inline_xgemm_k_) { \
      LIBXSMM_PRAGMA_UNROLL \
      for (libxsmm_inline_xgemm_i_ = 0; libxsmm_inline_xgemm_i_ < ((INT)(N)); ++libxsmm_inline_xgemm_i_) { \
        ((TYPE*)(C))[libxsmm_inline_xgemm_i_*((INT)(LDC))+libxsmm_inline_xgemm_j_] \
          = ((const TYPE*)(B))[libxsmm_inline_xgemm_i_*((INT)(LDB))+libxsmm_inline_xgemm_k_] * \
           (((const TYPE*)(A))[libxsmm_inline_xgemm_k_*((INT)(LDA))+libxsmm_inline_xgemm_j_] * libxsmm_inline_xgemm_alpha_) \
          + ((const TYPE*)(C))[libxsmm_inline_xgemm_i_*((INT)(LDC))+libxsmm_inline_xgemm_j_] * libxsmm_inline_xgemm_beta_; \
      } \
    } \
  } \
}
#endif

/** Inlinable GEMM exercising the compiler's code generation (single-precision). */
#define LIBXSMM_INLINE_SGEMM(FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) \
  LIBXSMM_INLINE_XGEMM(float, libxsmm_blasint, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
/** Inlinable GEMM exercising the compiler's code generation (double-precision). */
#define LIBXSMM_INLINE_DGEMM(FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) \
  LIBXSMM_INLINE_XGEMM(double, libxsmm_blasint, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
/** Inlinable GEMM exercising the compiler's code generation. */
#define LIBXSMM_INLINE_GEMM(FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) { \
  if (sizeof(double) == sizeof(*(A)) /*always true:*/&& 0 != (LDC)) { \
    LIBXSMM_INLINE_DGEMM(FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC); \
  } \
  else { \
    LIBXSMM_INLINE_SGEMM(FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC); \
  } \
}

/** Fall-back code paths: LIBXSMM_FALLBACK0, and LIBXSMM_FALLBACK1 (template). */
#if defined(LIBXSMM_FALLBACK_INLINE_GEMM)
# define LIBXSMM_FALLBACK0(TYPE, INT, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) \
    LIBXSMM_INLINE_XGEMM(TYPE, INT, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
#elif defined(LIBXSMM_FALLBACK_OMPS)
# define LIBXSMM_FALLBACK0(TYPE, INT, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) \
    LIBXSMM_OMPS_GEMM(TYPE, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
#else
# define LIBXSMM_FALLBACK0(TYPE, INT, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) \
    LIBXSMM_BLAS_XGEMM(TYPE, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
#endif
#if defined(LIBXSMM_FALLBACK_OMPS)
# define LIBXSMM_FALLBACK1(TYPE, INT, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) \
    LIBXSMM_OMPS_GEMM(TYPE, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
#else
# define LIBXSMM_FALLBACK1(TYPE, INT, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) \
    LIBXSMM_BLAS_XGEMM(TYPE, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
#endif

#if defined(__cplusplus) /** Fall-back from JIT inside of libxsmm_mmfunction (C++). */ \
  && (defined(LIBXSMM_FALLBACK_MMFUNCTION) || !defined(NDEBUG)/*debug code*/) \
  /* there are no individual fall-back requests for single or double-precision */ \
  && !defined(LIBXSMM_FALLBACK_SMMFUNCTION) && !defined(LIBXSMM_FALLBACK_DMMFUNCTION) \
  /* there is no request to avoid falling back */ \
  && !defined(LIBXSMM_FALLBACK_MMFUNCTION_NONE)
# define LIBXSMM_FALLBACK_SMMFUNCTION
# define LIBXSMM_FALLBACK_DMMFUNCTION
#endif

/** Helper macros for calling a dispatched function in a row/column-major aware fashion. */
#define LIBXSMM_MMCALL_ABC(FN, A, B, C) \
  LIBXSMM_ASSERT(FN); \
  FN(A, B, C)
#define LIBXSMM_MMCALL_PRF(FN, A, B, C, PA, PB, PC) { \
  LIBXSMM_NOPREFETCH_A(LIBXSMM_UNUSED(PA)); \
  LIBXSMM_NOPREFETCH_B(LIBXSMM_UNUSED(PB)); \
  LIBXSMM_NOPREFETCH_C(LIBXSMM_UNUSED(PC)); \
  LIBXSMM_ASSERT(FN); \
  FN(A, B, C, \
    LIBXSMM_PREFETCH_A(PA), \
    LIBXSMM_PREFETCH_B(PB), \
    LIBXSMM_PREFETCH_C(PC)); \
}

#if (0/*LIBXSMM_PREFETCH_NONE*/ == LIBXSMM_PREFETCH)
# define LIBXSMM_MMCALL_LDX(FN, A, B, C, M, N, K, LDA, LDB, LDC) \
  LIBXSMM_MMCALL_ABC(FN, A, B, C)
#else
# define LIBXSMM_MMCALL_LDX(FN, A, B, C, M, N, K, LDA, LDB, LDC) \
  LIBXSMM_MMCALL_PRF(FN, A, B, C, (A) + (LDA) * (K), (B) + (LDB) * (N), (C) + (LDC) * (N))
#endif
#define LIBXSMM_MMCALL(FN, A, B, C, M, N, K) LIBXSMM_MMCALL_LDX(FN, A, B, C, M, N, K, M, K, M)

/** Calculate problem size from M, N, and K using the correct integer type in order to cover the general case. */
#define LIBXSMM_MNK_SIZE(M, N, K) (((unsigned long long)(M)) * ((unsigned long long)(N)) * ((unsigned long long)(K)))

/**
 * Execute a specialized function, or use a fall-back code path depending on threshold (template).
 * LIBXSMM_FALLBACK0 or specialized function: below LIBXSMM_MAX_MNK
 * LIBXSMM_FALLBACK1: above LIBXSMM_MAX_MNK
 */
#define LIBXSMM_XGEMM(TYPE, INT, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) { \
  if (((unsigned long long)(LIBXSMM_MAX_MNK)) >= LIBXSMM_MNK_SIZE(M, N, K)) { \
    const int libxsmm_xgemm_flags_ = (int)(FLAGS); \
    const INT libxsmm_xgemm_lda_ = (INT)(LDA), libxsmm_xgemm_ldb_ = (INT)(LDB), libxsmm_xgemm_ldc_ = (INT)(LDC); \
    const TYPE libxsmm_xgemm_alpha_ = (TYPE)(ALPHA), libxsmm_xgemm_beta_ = (TYPE)(BETA); \
    const LIBXSMM_MMFUNCTION_TYPE(TYPE) libxsmm_mmfunction_ = LIBXSMM_MMDISPATCH_SYMBOL(TYPE)( \
      (INT)(M), (INT)(N), (INT)(K), &libxsmm_xgemm_lda_, &libxsmm_xgemm_ldb_, &libxsmm_xgemm_ldc_, \
      &libxsmm_xgemm_alpha_, &libxsmm_xgemm_beta_, &libxsmm_xgemm_flags_, 0); \
    if (0 != libxsmm_mmfunction_) { \
      LIBXSMM_MMCALL_LDX(libxsmm_mmfunction_, (const TYPE*)(A), (const TYPE*)(B), (TYPE*)(C), M, N, K, LDA, LDB, LDC); \
    } \
    else { \
      LIBXSMM_FALLBACK0(TYPE, INT, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC); \
    } \
  } \
  else { \
    LIBXSMM_FALLBACK1(TYPE, INT, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC); \
  } \
}

/** Dispatched general dense matrix multiplication (single-precision). */
#define LIBXSMM_SGEMM(FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) \
  LIBXSMM_XGEMM(float, libxsmm_blasint, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
/** Dispatched general dense matrix multiplication (double-precision). */
#define LIBXSMM_DGEMM(FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) \
  LIBXSMM_XGEMM(double, libxsmm_blasint, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
/** Dispatched general dense matrix multiplication. */
#define LIBXSMM_GEMM(FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) { \
  if (sizeof(double) == sizeof(*(A)) /*always true:*/&& 0 != (LDC)) { \
    LIBXSMM_DGEMM(FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC); \
  } \
  else { \
    LIBXSMM_SGEMM(FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC); \
  } \
}

/** Call libxsmm_gemm_print using LIBXSMM's GEMM-flags. */
#define LIBXSMM_GEMM_PRINT(OSTREAM, PRECISION, FLAGS, PM, PN, PK, PALPHA, A, PLDA, B, PLDB, PBETA, C, PLDC) { \
  const char libxsmm_gemm_print_transa_ = (char)(0 == (LIBXSMM_GEMM_FLAG_TRANS_A & (FLAGS)) ? 'N' : 'T'); \
  const char libxsmm_gemm_print_transb_ = (char)(0 == (LIBXSMM_GEMM_FLAG_TRANS_B & (FLAGS)) ? 'N' : 'T'); \
  libxsmm_gemm_print(OSTREAM, PRECISION, &libxsmm_gemm_print_transa_, &libxsmm_gemm_print_transb_, \
    PM, PN, PK, PALPHA, A, PLDA, B, PLDB, PBETA, C, PLDC); \
}

/**
 * Utility function, which either prints information about the GEMM call
 * or dumps (FILE/ostream=0) all input and output data into MHD files.
 * The Meta Image Format (MHD) is suitable for visual inspection using e.g.,
 * ITK-SNAP or ParaView.
 */
LIBXSMM_API void libxsmm_gemm_print(void* ostream,
  libxsmm_gemm_precision precision, const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const void* alpha, const void* a, const libxsmm_blasint* lda,
  const void* b, const libxsmm_blasint* ldb,
  const void* beta, void* c, const libxsmm_blasint* ldc);

/**
 * Structure of differences with matrix norms according
 * to http://www.netlib.org/lapack/lug/node75.html).
 */
typedef struct LIBXSMM_RETARGETABLE libxsmm_matdiff_info {
  /** One-norm */         double norm1_abs, norm1_rel;
  /** Infinity-norm */    double normi_abs, normi_rel;
  /** Froebenius-norm */  double normf_rel;
  /** L1-norm and L2-norm of differences. */
  double l2_abs, l2_rel, l1_ref, l1_tst;
  /** Maximum absolute and relative error. */
  double linf_abs, linf_rel;
  /** Location of maximum error (m, n). */
  libxsmm_blasint linf_abs_m, linf_abs_n;
} libxsmm_matdiff_info;

/** Utility function to calculate the difference between two matrices. */
LIBXSMM_API int libxsmm_matdiff(libxsmm_datatype datatype, libxsmm_blasint m, libxsmm_blasint n,
  const void* ref, const void* tst, const libxsmm_blasint* ldref, const libxsmm_blasint* ldtst,
  libxsmm_matdiff_info* info);

LIBXSMM_API_INLINE void libxsmm_matdiff_reduce(libxsmm_matdiff_info* output, const libxsmm_matdiff_info* input) {
  LIBXSMM_ASSERT(0 != output && 0 != input);
  if (output->normf_rel < input->normf_rel) {
    output->linf_abs_m = input->linf_abs_m;
    output->linf_abs_n = input->linf_abs_n;
    output->norm1_abs = input->norm1_abs;
    output->norm1_rel = input->norm1_rel;
    output->normi_abs = input->normi_abs;
    output->normi_rel = input->normi_rel;
    output->normf_rel = input->normf_rel;
    output->linf_abs = input->linf_abs;
    output->linf_rel = input->linf_rel;
    output->l2_abs = input->l2_abs;
    output->l2_rel = input->l2_rel;
    output->l1_ref = input->l1_ref;
    output->l1_tst = input->l1_tst;
  }
}

/* Implementation is taken from an anonymous GitHub Gist. */
LIBXSMM_API_INLINE unsigned int libxsmm_cbrt_u64(unsigned long long n) {
  unsigned long long b; unsigned int y = 0; int s;
  for (s = 63; s >= 0; s -= 3) {
    y += y; b = 3 * y * ((unsigned long long)y + 1) + 1;
    if (b <= (n >> s)) { n -= b << s; ++y; }
  }
  return y;
}

LIBXSMM_API_INLINE unsigned int libxsmm_cbrt_u32(unsigned int n) {
  unsigned int b; unsigned int y = 0; int s;
  for (s = 31; s >= 0; s -= 3) {
    y += y; b = 3 * y * (y + 1) + 1;
    if (b <= (n >> s)) { n -= b << s; ++y; }
  }
  return y;
}

#endif /*LIBXSMM_FRONTEND_H*/
