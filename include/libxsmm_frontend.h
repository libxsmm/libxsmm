/******************************************************************************
** Copyright (c) 2013-2015, Intel Corporation                                **
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

#include "libxsmm.h"
#include <assert.h>

/** Integer type for LAPACK/BLAS (LP64: 32-bit, and ILP64: 64-bit). */
#if (0 != LIBXSMM_ILP64)
typedef long long libxsmm_blasint;
#else
typedef int libxsmm_blasint;
#endif

/** Helper macro for GEMM argument permutation depending on storage scheme. */
#if (0 != LIBXSMM_ROW_MAJOR)
# define LIBXSMM_LD(M, N) (N)
#else
# define LIBXSMM_LD(M, N) (M)
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
#if !defined(LIBXSMM_PREFETCH_A)
# define LIBXSMM_PREFETCH_A(EXPR) 0
#endif
#if !defined(LIBXSMM_PREFETCH_B)
# define LIBXSMM_PREFETCH_B(EXPR) 0
#endif
#if !defined(LIBXSMM_PREFETCH_C)
# define LIBXSMM_PREFETCH_C(EXPR) 0
#endif

/** Helper macro for GEMM function names (and similar functions). */
#define LIBXSMM_TPREFIX(REAL, FUNCTION) LIBXSMM_TPREFIX_##REAL(FUNCTION)
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
#else
/** Fallback prototype functions served by any compliant LAPACK/BLAS. */
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE void LIBXSMM_FSYMBOL(dgemm)(
  const char*, const char*, const libxsmm_blasint*, const libxsmm_blasint*, const libxsmm_blasint*,
  const double*, const double*, const libxsmm_blasint*, const double*, const libxsmm_blasint*,
  const double*, double*, const libxsmm_blasint*);
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE void LIBXSMM_FSYMBOL(sgemm)(
	const char*, const char*, const libxsmm_blasint*, const libxsmm_blasint*, const libxsmm_blasint*,
	const float*, const float*, const libxsmm_blasint*, const float*, const libxsmm_blasint*,
	const float*, float*, const libxsmm_blasint*);
#endif

/** BLAS-based GEMM supplied by the linked LAPACK/BLAS library (template). */
#define LIBXSMM_BXGEMM(REAL, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) { \
  const char libxsmm_bxgemm_transa_ = (char)(0 == (LIBXSMM_GEMM_FLAG_TRANS_A & (FLAGS)) ? 'N' : 'T'); \
  const char libxsmm_bxgemm_transb_ = (char)(0 == (LIBXSMM_GEMM_FLAG_TRANS_B & (FLAGS)) ? 'N' : 'T'); \
  const libxsmm_blasint libxsmm_bxgemm_m_ = (libxsmm_blasint)(M); \
  const libxsmm_blasint libxsmm_bxgemm_n_ = (libxsmm_blasint)(N); \
  const libxsmm_blasint libxsmm_bxgemm_k_ = (libxsmm_blasint)(K); \
  const libxsmm_blasint libxsmm_bxgemm_lda_ = (libxsmm_blasint)(0 != (LDA) ? LIBXSMM_MAX/*BLAS-conformance*/(LDA, M) \
    /* if the value of LDA was zero: make LDA a multiple of LIBXSMM_ALIGNMENT */ \
    : LIBXSMM_ALIGN_VALUE(M, sizeof(REAL), LIBXSMM_ALIGNMENT)), libxsmm_bxgemm_ldb_ = (libxsmm_blasint)(LDB); \
  const libxsmm_blasint libxsmm_bxgemm_ldc_ = (libxsmm_blasint)(0 != (LDC) ? LIBXSMM_MAX/*BLAS-conformance*/(LDC, M) \
    /* if the value of LDC was zero: make LDC a multiple of LIBXSMM_ALIGNMENT */ \
    : LIBXSMM_ALIGN_VALUE(M, sizeof(REAL), LIBXSMM_ALIGNMENT)); \
  const REAL libxsmm_bxgemm_alpha_ = (REAL)(ALPHA), libxsmm_bxgemm_beta_ = (REAL)(BETA); \
  LIBXSMM_FSYMBOL(LIBXSMM_TPREFIX(REAL, gemm))(&libxsmm_bxgemm_transa_, &libxsmm_bxgemm_transb_, \
    &libxsmm_bxgemm_m_, &libxsmm_bxgemm_n_, &libxsmm_bxgemm_k_, \
    &libxsmm_bxgemm_alpha_, A, &libxsmm_bxgemm_lda_, B, &libxsmm_bxgemm_ldb_, \
    &libxsmm_bxgemm_beta_, C, &libxsmm_bxgemm_ldc_); \
}

/** BLAS-based GEMM supplied by the linked LAPACK/BLAS library (single-precision). */
#define LIBXSMM_BSGEMM(FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) \
  LIBXSMM_BXGEMM(float, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
/** BLAS-based GEMM supplied by the linked LAPACK/BLAS library (single-precision). */
#define LIBXSMM_BDGEMM(FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) \
  LIBXSMM_BXGEMM(double, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
/** BLAS-based GEMM supplied by the linked LAPACK/BLAS library. */
#define LIBXSMM_BGEMM(FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) { \
  if (sizeof(double) == sizeof(*(A))) { \
    LIBXSMM_BDGEMM(FLAGS, M, N, K, \
      (double)(ALPHA), (const double*)(A), LDA, (const double*)(B), LDB, \
      (double) (BETA), (double*)(C), LDC); \
  } \
  else {\
    LIBXSMM_BSGEMM(FLAGS, M, N, K, \
      (float)(ALPHA), (const float*)(A), LDA, (const float*)(B), LDB, \
      (float) (BETA), (float*)(C), LDC); \
  } \
}

/** Inlinable GEMM exercising the compiler's code generation (template). */
#if defined(MKL_DIRECT_CALL_SEQ) || defined(MKL_DIRECT_CALL)
# define LIBXSMM_IXGEMM(REAL, INT, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) \
  LIBXSMM_BXGEMM(REAL, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
#else
# define LIBXSMM_IXGEMM(REAL, INT, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) { \
  const REAL *const libxsmm_ixgemm_a_ = (const REAL*)(B), *const libxsmm_ixgemm_b_ = (const REAL*)(A); \
  const INT libxsmm_ixgemm_m_ = (INT)(M), libxsmm_ixgemm_n_ = (INT)(N); \
  const INT libxsmm_ixgemm_lda_ = (INT)(0 != (LDA) ? LIBXSMM_MAX/*BLAS-conformance*/(LDA, M) \
    /* if the value of LDA was zero: make LDA a multiple of LIBXSMM_ALIGNMENT */ \
    : LIBXSMM_ALIGN_VALUE(M, sizeof(REAL), LIBXSMM_ALIGNMENT)); \
  const INT libxsmm_ixgemm_ldc_ = (INT)(0 != (LDC) ? LIBXSMM_MAX/*BLAS-conformance*/(LDC, M) \
    /* if the value of LDC was zero: make LDC a multiple of LIBXSMM_ALIGNMENT */ \
    : LIBXSMM_ALIGN_VALUE(M, sizeof(REAL), LIBXSMM_ALIGNMENT)); \
  INT libxsmm_ixgemm_i_, libxsmm_ixgemm_j_, libxsmm_ixgemm_k_; \
  REAL *const libxsmm_ixgemm_c_ = (C); \
  assert(0 == (LIBXSMM_GEMM_FLAG_TRANS_A & (FLAGS)) && 0 == (LIBXSMM_GEMM_FLAG_TRANS_B & (FLAGS))/*not supported*/); \
  LIBXSMM_PRAGMA_SIMD/*_COLLAPSE(2)*/ \
  for (libxsmm_ixgemm_j_ = 0; libxsmm_ixgemm_j_ < libxsmm_ixgemm_m_; ++libxsmm_ixgemm_j_) { \
    LIBXSMM_PRAGMA_LOOP_COUNT(1, LIBXSMM_LD(LIBXSMM_MAX_N, LIBXSMM_MAX_M), LIBXSMM_LD(LIBXSMM_AVG_N, LIBXSMM_AVG_M)) \
    for (libxsmm_ixgemm_i_ = 0; libxsmm_ixgemm_i_ < libxsmm_ixgemm_n_; ++libxsmm_ixgemm_i_) { \
      const INT libxsmm_ixgemm_index_ = libxsmm_ixgemm_i_ * libxsmm_ixgemm_ldc_ + libxsmm_ixgemm_j_; \
      REAL libxsmm_ixgemm_r_ = libxsmm_ixgemm_c_[libxsmm_ixgemm_index_] * (BETA); \
      LIBXSMM_PRAGMA_SIMD_REDUCTION(+:libxsmm_ixgemm_r_) \
      LIBXSMM_PRAGMA_UNROLL \
      for (libxsmm_ixgemm_k_ = 0; libxsmm_ixgemm_k_ < (K); ++libxsmm_ixgemm_k_) { \
        libxsmm_ixgemm_r_ += libxsmm_ixgemm_a_[libxsmm_ixgemm_i_*(LDB)+libxsmm_ixgemm_k_] * (ALPHA) \
                    * libxsmm_ixgemm_b_[libxsmm_ixgemm_k_*libxsmm_ixgemm_lda_+libxsmm_ixgemm_j_]; \
      } \
      libxsmm_ixgemm_c_[libxsmm_ixgemm_index_] = libxsmm_ixgemm_r_; \
    } \
  } \
}
#endif

/** Inlinable GEMM exercising the compiler's code generation (single-precision). */
#define LIBXSMM_ISGEMM(FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) \
  LIBXSMM_IXGEMM(float, libxsmm_blasint, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
/** Inlinable GEMM exercising the compiler's code generation (double-precision). */
#define LIBXSMM_IDGEMM(FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) \
  LIBXSMM_IXGEMM(double, libxsmm_blasint, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
/** Inlinable GEMM exercising the compiler's code generation. */
#define LIBXSMM_IGEMM(FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) { \
  if (sizeof(double) == sizeof(*(A))) { \
    LIBXSMM_IDGEMM(FLAGS, M, N, K, \
      (double)(ALPHA), (const double*)(A), LDA, (const double*)(B), LDB, \
      (double) (BETA), (double*)(C), LDC); \
  } \
  else {\
    LIBXSMM_ISGEMM(FLAGS, M, N, K, \
      (float)(ALPHA), (const float*)(A), LDA, (const float*)(B), LDB, \
      (float) (BETA), (float*)(C), LDC); \
  } \
}

/** Fallback code paths: LIBXSMM_FALLBACK0, and LIBXSMM_FALLBACK1 (template). */
#if defined(LIBXSMM_FALLBACK_IGEMM)
# define LIBXSMM_FALLBACK0(REAL, INT, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) \
    LIBXSMM_IXGEMM(REAL, INT, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
#else
# define LIBXSMM_FALLBACK0(REAL, INT, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) \
    LIBXSMM_BXGEMM(REAL, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
#endif
#define LIBXSMM_FALLBACK1(REAL, INT, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) \
  LIBXSMM_BXGEMM(REAL, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)

/**
 * Execute a specialized function, or use a fallback code path depending on threshold (template).
 * LIBXSMM_FALLBACK0 or specialized function: below LIBXSMM_MAX_MNK
 * LIBXSMM_FALLBACK1: above LIBXSMM_MAX_MNK
 */
#define LIBXSMM_GEMM(REAL, INT, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) { \
  if (((unsigned long long)(LIBXSMM_MAX_MNK)) >= \
     (((unsigned long long)(M)) * \
      ((unsigned long long)(N)) * \
      ((unsigned long long)(K)))) \
  { \
    const int libxsmm_gemm_flags_ = (int)(FLAGS), libxsmm_gemm_ldb_ = (int)(LDB); \
    const int libxsmm_gemm_lda_ = (int)(0 != (LDA) ? LIBXSMM_MAX/*BLAS-conformance*/(LDA, M) \
      /* if the value of LDA was zero: make LDA a multiple of LIBXSMM_ALIGNMENT */ \
      : LIBXSMM_ALIGN_VALUE(M, sizeof(REAL), LIBXSMM_ALIGNMENT)); \
    const int libxsmm_gemm_ldc_ = (int)(0 != (LDC) ? LIBXSMM_MAX/*BLAS-conformance*/(LDC, M) \
      /* if the value of LDC was zero: make LDC a multiple of LIBXSMM_ALIGNMENT */ \
      : LIBXSMM_ALIGN_VALUE(M, sizeof(REAL), LIBXSMM_ALIGNMENT)); \
    const REAL libxsmm_gemm_alpha_ = (REAL)(ALPHA), libxsmm_gemm_beta_ = (REAL)(BETA); \
    int libxsmm_gemm_fallback_ = 0; \
    if (LIBXSMM_PREFETCH_NONE == LIBXSMM_PREFETCH) { \
      const LIBXSMM_CONCATENATE(libxsmm_, LIBXSMM_TPREFIX(REAL, function)) libxsmm_gemm_function_ = \
        LIBXSMM_CONCATENATE(libxsmm_, LIBXSMM_TPREFIX(REAL, dispatch))((int)(M), (int)(N), (int)(K), \
          &libxsmm_gemm_lda_, &libxsmm_gemm_ldb_, &libxsmm_gemm_ldc_, \
          &libxsmm_gemm_alpha_, &libxsmm_gemm_beta_, \
          &libxsmm_gemm_flags_, 0); \
      if (0 != libxsmm_gemm_function_) { \
        libxsmm_gemm_function_(A, B, C); \
      } \
      else { \
        libxsmm_gemm_fallback_ = 1; \
      } \
    } \
    else { \
      const int libxsmm_gemm_prefetch_ = (LIBXSMM_PREFETCH); \
      const LIBXSMM_CONCATENATE(libxsmm_, LIBXSMM_TPREFIX(REAL, function)) libxsmm_gemm_function_ = \
        LIBXSMM_CONCATENATE(libxsmm_, LIBXSMM_TPREFIX(REAL, dispatch))((int)(M), (int)(N), (int)(K), \
          &libxsmm_gemm_lda_, &libxsmm_gemm_ldb_, &libxsmm_gemm_ldc_, \
          &libxsmm_gemm_alpha_, &libxsmm_gemm_beta_, \
          &libxsmm_gemm_flags_, &libxsmm_gemm_prefetch_); \
      if (0 != libxsmm_gemm_function_) { \
        libxsmm_gemm_function_(A, B, C, \
          0 != LIBXSMM_PREFETCH_A(1) ? (((const REAL*)(A)) + (libxsmm_gemm_lda_) * (K)) : ((const REAL*)(A)), \
          0 != LIBXSMM_PREFETCH_B(1) ? (((const REAL*)(B)) + (libxsmm_gemm_ldb_) * (N)) : ((const REAL*)(B)), \
          0 != LIBXSMM_PREFETCH_C(1) ? (((const REAL*)(C)) + (libxsmm_gemm_ldc_) * (N)) : ((const REAL*)(C))); \
      } \
      else { \
        libxsmm_gemm_fallback_ = 1; \
      } \
    } \
    if (0 != libxsmm_gemm_fallback_) { \
      LIBXSMM_FALLBACK0(REAL, INT, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC); \
    } \
  } \
  else { \
    LIBXSMM_FALLBACK1(REAL, INT, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC); \
  } \
}

#endif /*LIBXSMM_FRONTEND_H*/
