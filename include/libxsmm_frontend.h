/******************************************************************************
** Copyright (c) 2015-2016, Intel Corporation                                **
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
#include "libxsmm_macros.h"
#include "libxsmm.h"

#if defined(LIBXSMM_OFFLOAD_BUILD)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#include <assert.h>
#if defined(LIBXSMM_OFFLOAD_BUILD)
# pragma offload_attribute(pop)
#endif

/** Helper macro for GEMM argument permutation depending on storage scheme. */
#if (0 != LIBXSMM_COL_MAJOR)
# define LIBXSMM_LD(M, N) (M)
#else
# define LIBXSMM_LD(M, N) (N)
#endif

/** Helper macro for aligning a buffer for aligned loads/store instructions. */
#if (0 != (4/*LIBXSMM_GEMM_FLAG_ALIGN_A*/ & LIBXSMM_FLAGS) || 0 != (8/*LIBXSMM_GEMM_FLAG_ALIGN_C*/ & LIBXSMM_FLAGS))
# define LIBXSMM_ALIGN_LDST(POINTER) LIBXSMM_ALIGN2(POINTER, LIBXSMM_ALIGNMENT)
#else
# define LIBXSMM_ALIGN_LDST(POINTER) (POINTER)
#endif

/** Helper macros for eliding prefetch address calculations depending on prefetch scheme. */
#if 0 != ((LIBXSMM_PREFETCH) & 2) || 0 != ((LIBXSMM_PREFETCH) & 4)
# define LIBXSMM_PREFETCH_A(EXPR) (EXPR)
#endif
#if 0 != ((LIBXSMM_PREFETCH) & 8)
# define LIBXSMM_PREFETCH_B(EXPR) (EXPR)
#endif
#if 0/*no scheme yet using C*/
# define LIBXSMM_PREFETCH_C(EXPR) (EXPR)
#endif
#if defined(LIBXSMM_PREFETCH_A)
# define LIBXSMM_NOPREFETCH_A(EXPR)
#else
# define LIBXSMM_NOPREFETCH_A(EXPR) EXPR
# define LIBXSMM_PREFETCH_A(EXPR) NULL
#endif
#if defined(LIBXSMM_PREFETCH_B)
# define LIBXSMM_NOPREFETCH_B(EXPR)
#else
# define LIBXSMM_NOPREFETCH_B(EXPR) EXPR
# define LIBXSMM_PREFETCH_B(EXPR) NULL
#endif
#if defined(LIBXSMM_PREFETCH_C)
# define LIBXSMM_NOPREFETCH_C(EXPR)
#else
# define LIBXSMM_NOPREFETCH_C(EXPR) EXPR
# define LIBXSMM_PREFETCH_C(EXPR) NULL
#endif

/** Helper macro for GEMM function names (and similar functions). */
#define LIBXSMM_TPREFIX_AUX(REAL, FUNCTION) LIBXSMM_TPREFIX_##REAL(FUNCTION)
#define LIBXSMM_TPREFIX(REAL, FUNCTION) LIBXSMM_TPREFIX_AUX(REAL, FUNCTION)
#define LIBXSMM_TPREFIX_double(FUNCTION) d##FUNCTION
#define LIBXSMM_TPREFIX_float(FUNCTION) s##FUNCTION

/** Check ILP64 configuration for sanity. */
#if (defined(MKL_ILP64) && 0 == LIBXSMM_ILP64)
# error "Inconsistent ILP64 configuration detected!"
#endif

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
#elif (0 != LIBXSMM_ILP64)
/** Fallback prototype functions served by any compliant LAPACK/BLAS (ILP64). */
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE void LIBXSMM_FSYMBOL(dgemm)(
  const char*, const char*, const long long*, const long long*, const long long*,
  const double*, const double*, const long long*, const double*, const long long*,
  const double*, double*, const long long*);
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE void LIBXSMM_FSYMBOL(sgemm)(
  const char*, const char*, const long long*, const long long*, const long long*,
  const float*, const float*, const long long*, const float*, const long long*,
  const float*, float*, const long long*);
#else /*LP64*/
/** Fallback prototype functions served by any compliant LAPACK/BLAS (LP64). */
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE void LIBXSMM_FSYMBOL(dgemm)(
  const char*, const char*, const int*, const int*, const int*,
  const double*, const double*, const int*, const double*, const int*,
  const double*, double*, const int*);
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE void LIBXSMM_FSYMBOL(sgemm)(
  const char*, const char*, const int*, const int*, const int*,
  const float*, const float*, const int*, const float*, const int*,
  const float*, float*, const int*);
#endif

/** Construct symbol name from a given real type name (float or double). */
#define LIBXSMM_BLAS_GEMM_SYMBOL(REAL)  LIBXSMM_FSYMBOL(LIBXSMM_TPREFIX(REAL, gemm))
#define LIBXSMM_CBLAS_GEMM_SYMBOL(REAL) LIBXSMM_CONCATENATE(cblas_, LIBXSMM_TPREFIX(REAL, gemm))
#define LIBXSMM_XBLAS_GEMM_SYMBOL(REAL) LIBXSMM_CONCATENATE(libxsmm_blas_, LIBXSMM_TPREFIX(REAL, gemm))
#define LIBXSMM_MMFUNCTION_TYPE(REAL)   LIBXSMM_CONCATENATE(libxsmm_, LIBXSMM_TPREFIX(REAL, mmfunction))
#define LIBXSMM_MMDISPATCH_SYMBOL(REAL) LIBXSMM_CONCATENATE(libxsmm_, LIBXSMM_TPREFIX(REAL, mmdispatch))
#define LIBXSMM_XGEMM_SYMBOL(REAL)      LIBXSMM_CONCATENATE(libxsmm_, LIBXSMM_TPREFIX(REAL, gemm))

/** BLAS-based GEMM supplied by the linked LAPACK/BLAS library (template). */
#define LIBXSMM_BLAS_XGEMM(REAL, SYMBOL, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) { \
  const char libxsmm_blas_xgemm_transa_ = (char)(0 == (LIBXSMM_GEMM_FLAG_TRANS_A & (FLAGS)) ? 'N' : 'T'); \
  const char libxsmm_blas_xgemm_transb_ = (char)(0 == (LIBXSMM_GEMM_FLAG_TRANS_B & (FLAGS)) ? 'N' : 'T'); \
  const REAL libxsmm_blas_xgemm_alpha_ = (REAL)(ALPHA), libxsmm_blas_xgemm_beta_ = (REAL)(BETA); \
  const libxsmm_blasint libxsmm_blas_xgemm_lda_ = (libxsmm_blasint)LIBXSMM_LD(LDA, LDB); \
  const libxsmm_blasint libxsmm_blas_xgemm_ldb_ = (libxsmm_blasint)LIBXSMM_LD(LDB, LDA); \
  const libxsmm_blasint libxsmm_blas_xgemm_ldc_ = (libxsmm_blasint)(LDC); \
  const libxsmm_blasint libxsmm_blas_xgemm_m_ = (libxsmm_blasint)LIBXSMM_LD(M, N); \
  const libxsmm_blasint libxsmm_blas_xgemm_n_ = (libxsmm_blasint)LIBXSMM_LD(N, M); \
  const libxsmm_blasint libxsmm_blas_xgemm_k_ = (libxsmm_blasint)(K); \
  assert(0 != ((uintptr_t)SYMBOL)); \
  SYMBOL(&libxsmm_blas_xgemm_transa_, &libxsmm_blas_xgemm_transb_, \
    &libxsmm_blas_xgemm_m_, &libxsmm_blas_xgemm_n_, &libxsmm_blas_xgemm_k_, \
    &libxsmm_blas_xgemm_alpha_, (const REAL*)LIBXSMM_LD(A, B), &libxsmm_blas_xgemm_lda_, \
                                (const REAL*)LIBXSMM_LD(B, A), &libxsmm_blas_xgemm_ldb_, \
    &libxsmm_blas_xgemm_beta_, (REAL*)(C), &libxsmm_blas_xgemm_ldc_); \
}

/** BLAS-based GEMM supplied by the linked LAPACK/BLAS library (single-precision). */
#define LIBXSMM_BLAS_SGEMM(FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) \
  LIBXSMM_BLAS_XGEMM(float, LIBXSMM_FSYMBOL(sgemm), FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
/** BLAS-based GEMM supplied by the linked LAPACK/BLAS library (single-precision). */
#define LIBXSMM_BLAS_DGEMM(FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) \
  LIBXSMM_BLAS_XGEMM(double, LIBXSMM_FSYMBOL(dgemm), FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
/** BLAS-based GEMM supplied by the linked LAPACK/BLAS library. */
#define LIBXSMM_BLAS_GEMM(FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) { \
  if (sizeof(double) == sizeof(*(A))) { \
    LIBXSMM_BLAS_DGEMM(FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC); \
  } \
  else {\
    LIBXSMM_BLAS_SGEMM(FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC); \
  } \
}

/** Inlinable GEMM exercising the compiler's code generation (template). */
#if defined(MKL_DIRECT_CALL_SEQ) || defined(MKL_DIRECT_CALL)
# define LIBXSMM_INLINE_XGEMM(REAL, INT, SYMBOL, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) \
  assert(0 != ((uintptr_t)SYMBOL)); \
  LIBXSMM_BLAS_XGEMM(REAL, SYMBOL, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
#else
# define LIBXSMM_INLINE_XGEMM(REAL, INT, SYMBOL, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) { \
  const REAL libxsmm_inline_xgemm_alpha_ = (REAL)(1 == (ALPHA) ? 1 : (-1 == (ALPHA) ? -1 : (ALPHA))); \
  const REAL libxsmm_inline_xgemm_beta_ = (REAL)(1 == (BETA) ? 1 : (0 == (BETA) ? 0 : (BETA))); \
  INT libxsmm_inline_xgemm_i_, libxsmm_inline_xgemm_j_, libxsmm_inline_xgemm_k_; \
  assert(0 == (LIBXSMM_GEMM_FLAG_TRANS_A & (FLAGS)) && 0 == (LIBXSMM_GEMM_FLAG_TRANS_B & (FLAGS))/*not supported*/); \
  LIBXSMM_PRAGMA_SIMD \
  for (libxsmm_inline_xgemm_j_ = 0; libxsmm_inline_xgemm_j_ < ((INT)LIBXSMM_LD(M, N)); ++libxsmm_inline_xgemm_j_) { \
    LIBXSMM_PRAGMA_LOOP_COUNT(1, LIBXSMM_MAX_K, LIBXSMM_AVG_K) \
    for (libxsmm_inline_xgemm_k_ = 0; libxsmm_inline_xgemm_k_ < (K); ++libxsmm_inline_xgemm_k_) { \
      LIBXSMM_PRAGMA_UNROLL \
      for (libxsmm_inline_xgemm_i_ = 0; libxsmm_inline_xgemm_i_ < ((INT)LIBXSMM_LD(N, M)); ++libxsmm_inline_xgemm_i_) { \
        ((REAL*)(C))[libxsmm_inline_xgemm_i_*((INT)(LDC))+libxsmm_inline_xgemm_j_] \
          = ((const REAL*)LIBXSMM_LD(B, A))[libxsmm_inline_xgemm_i_*((INT)LIBXSMM_LD(LDB, LDA))+libxsmm_inline_xgemm_k_] * \
           (((const REAL*)LIBXSMM_LD(A, B))[libxsmm_inline_xgemm_k_*((INT)LIBXSMM_LD(LDA, LDB))+libxsmm_inline_xgemm_j_] * libxsmm_inline_xgemm_alpha_) \
          + ((const REAL*)(C))[libxsmm_inline_xgemm_i_*((INT)(LDC))+libxsmm_inline_xgemm_j_] * libxsmm_inline_xgemm_beta_; \
      } \
    } \
  } \
}
#endif

/** Inlinable GEMM exercising the compiler's code generation (single-precision). */
#define LIBXSMM_INLINE_SGEMM(FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) \
  LIBXSMM_INLINE_XGEMM(float, libxsmm_blasint, LIBXSMM_FSYMBOL(sgemm), FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
/** Inlinable GEMM exercising the compiler's code generation (double-precision). */
#define LIBXSMM_INLINE_DGEMM(FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) \
  LIBXSMM_INLINE_XGEMM(double, libxsmm_blasint, LIBXSMM_FSYMBOL(dgemm), FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
/** Inlinable GEMM exercising the compiler's code generation. */
#define LIBXSMM_INLINE_GEMM(FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) { \
  if (sizeof(double) == sizeof(*(A))) { \
    LIBXSMM_INLINE_DGEMM(FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC); \
  } \
  else {\
    LIBXSMM_INLINE_SGEMM(FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC); \
  } \
}

/** Fallback code paths: LIBXSMM_FALLBACK0, and LIBXSMM_FALLBACK1 (template). */
#if defined(LIBXSMM_FALLBACK_INLINE_GEMM)
# define LIBXSMM_FALLBACK0(REAL, INT, SYMBOL, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) \
    LIBXSMM_INLINE_XGEMM(REAL, INT, SYMBOL, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
#else
# define LIBXSMM_FALLBACK0(REAL, INT, SYMBOL, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) \
    LIBXSMM_BLAS_XGEMM(REAL, SYMBOL, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
#endif
#define LIBXSMM_FALLBACK1(REAL, INT, SYMBOL, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) \
  LIBXSMM_BLAS_XGEMM(REAL, SYMBOL, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)

/** Helper macros for calling a dispatched function in a row/column-major aware fashion. */
#define LIBXSMM_MMCALL_ABC(FN, A, B, C) FN(LIBXSMM_LD(A, B), LIBXSMM_LD(B, A), C)
#define LIBXSMM_MMCALL_PRF(FN, A, B, C, PA, PB, PC) { \
  LIBXSMM_NOPREFETCH_A(LIBXSMM_UNUSED(LIBXSMM_LD(PA, PB))); \
  LIBXSMM_NOPREFETCH_B(LIBXSMM_UNUSED(LIBXSMM_LD(PB, PA))); \
  LIBXSMM_NOPREFETCH_C(LIBXSMM_UNUSED(PC)); \
  FN(LIBXSMM_LD(A, B), LIBXSMM_LD(B, A), C, \
    LIBXSMM_PREFETCH_A(LIBXSMM_LD(PA, PB)), \
    LIBXSMM_PREFETCH_B(LIBXSMM_LD(PB, PA)), \
    LIBXSMM_PREFETCH_C(PC)); \
}

/**
 * Execute a specialized function, or use a fallback code path depending on threshold (template).
 * LIBXSMM_FALLBACK0 or specialized function: below LIBXSMM_MAX_MNK
 * LIBXSMM_FALLBACK1: above LIBXSMM_MAX_MNK
 */
#define LIBXSMM_XGEMM(REAL, INT, SYMBOL, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) { \
  if (((unsigned long long)(LIBXSMM_MAX_MNK)) >= \
     (((unsigned long long)(M)) * \
      ((unsigned long long)(N)) * \
      ((unsigned long long)(K)))) \
  { \
    const int libxsmm_xgemm_flags_ = (int)(FLAGS), libxsmm_xgemm_ldb_ = (int)(LDB); \
    const int libxsmm_xgemm_lda_ = (int)(0 != (LDA) ? (LDA) \
      /* if the value of LDA was zero: make LDA a multiple of LIBXSMM_ALIGNMENT */ \
      : LIBXSMM_ALIGN_VALUE(M, sizeof(REAL), LIBXSMM_ALIGNMENT)); \
    const int libxsmm_xgemm_ldc_ = (int)(0 != (LDC) ? (LDC) \
      /* if the value of LDC was zero: make LDC a multiple of LIBXSMM_ALIGNMENT */ \
      : LIBXSMM_ALIGN_VALUE(M, sizeof(REAL), LIBXSMM_ALIGNMENT)); \
    const REAL libxsmm_xgemm_alpha_ = (REAL)(ALPHA), libxsmm_xgemm_beta_ = (REAL)(BETA); \
    int libxsmm_xgemm_fallback_ = 0; \
    if (LIBXSMM_PREFETCH_NONE == LIBXSMM_PREFETCH) { \
      const LIBXSMM_MMFUNCTION_TYPE(REAL) libxsmm_mmfunction_ = \
        LIBXSMM_MMDISPATCH_SYMBOL(REAL)((int)(M), (int)(N), (int)(K), \
          &libxsmm_xgemm_lda_, &libxsmm_xgemm_ldb_, &libxsmm_xgemm_ldc_, \
          &libxsmm_xgemm_alpha_, &libxsmm_xgemm_beta_, \
          &libxsmm_xgemm_flags_, 0); \
      if (0 != libxsmm_mmfunction_) { \
        LIBXSMM_MMCALL_ABC(libxsmm_mmfunction_, (const REAL*)(A), (const REAL*)(B), (REAL*)(C)); \
      } \
      else { \
        libxsmm_xgemm_fallback_ = 1; \
      } \
    } \
    else { \
      const int libxsmm_xgemm_prefetch_ = (LIBXSMM_PREFETCH); \
      const LIBXSMM_MMFUNCTION_TYPE(REAL) libxsmm_mmfunction_ = \
        LIBXSMM_MMDISPATCH_SYMBOL(REAL)((int)(M), (int)(N), (int)(K), \
          &libxsmm_xgemm_lda_, &libxsmm_xgemm_ldb_, &libxsmm_xgemm_ldc_, \
          &libxsmm_xgemm_alpha_, &libxsmm_xgemm_beta_, \
          &libxsmm_xgemm_flags_, &libxsmm_xgemm_prefetch_); \
      if (0 != libxsmm_mmfunction_) { \
        LIBXSMM_MMCALL_PRF(libxsmm_mmfunction_, (const REAL*)(A), (const REAL*)(B), (REAL*)(C), \
          ((const REAL*)(A)) + libxsmm_xgemm_lda_ * (K), ((const REAL*)(B)) + libxsmm_xgemm_ldb_ * (N), \
          ((const REAL*)(C)) + libxsmm_xgemm_ldc_ * (N)); \
      } \
      else { \
        libxsmm_xgemm_fallback_ = 1; \
      } \
    } \
    if (0 != libxsmm_xgemm_fallback_) { \
      LIBXSMM_FALLBACK0(REAL, INT, SYMBOL, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC); \
    } \
  } \
  else { \
    LIBXSMM_FALLBACK1(REAL, INT, SYMBOL, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC); \
  } \
}

/** Dispatched general dense matrix multiplication (single-precision). */
#define LIBXSMM_SGEMM(FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) \
  LIBXSMM_XGEMM(float, libxsmm_blasint, LIBXSMM_FSYMBOL(sgemm), FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
/** Dispatched general dense matrix multiplication (double-precision). */
#define LIBXSMM_DGEMM(FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) \
  LIBXSMM_XGEMM(double, libxsmm_blasint, LIBXSMM_FSYMBOL(dgemm), FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
/** Dispatched general dense matrix multiplication. */
#define LIBXSMM_GEMM(FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) { \
  if (sizeof(double) == sizeof(*(A))) { \
    LIBXSMM_DGEMM(FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC); \
  } \
  else {\
    LIBXSMM_SGEMM(FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC); \
  } \
}

#endif /*LIBXSMM_FRONTEND_H*/
