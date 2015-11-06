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
#ifndef LIBXSMM_FALLBACK_H
#define LIBXSMM_FALLBACK_H

#include "libxsmm.h"

#if (0 != LIBXSMM_ROW_MAJOR)
# define LIBXSMM_LD(M, N) (N)
#else
# define LIBXSMM_LD(M, N) (M)
#endif
#if (1 < LIBXSMM_ALIGNED_LOADS)
# define LIBXSMM_ASSUME_ALIGNED_LOADS(A) LIBXSMM_ASSUME_ALIGNED(A, LIBXSMM_ALIGNED_LOADS)
# define LIBXSMM_ALIGN_LOADS(N, TYPESIZE) LIBXSMM_ALIGN_VALUE(N, TYPESIZE, LIBXSMM_ALIGNED_LOADS)
#else
# define LIBXSMM_ASSUME_ALIGNED_LOADS(A)
# define LIBXSMM_ALIGN_LOADS(N, TYPESIZE) (N)
#endif
#if (1 < LIBXSMM_ALIGNED_STORES)
# define LIBXSMM_ASSUME_ALIGNED_STORES(A) LIBXSMM_ASSUME_ALIGNED(A, LIBXSMM_ALIGNED_STORES)
# define LIBXSMM_ALIGN_STORES(N, TYPESIZE) LIBXSMM_ALIGN_VALUE(N, TYPESIZE, LIBXSMM_ALIGNED_STORES)
#else
# define LIBXSMM_ASSUME_ALIGNED_STORES(A)
# define LIBXSMM_ALIGN_STORES(N, TYPESIZE) (N)
#endif

/** Flag representing the GEMM performed by the simplified interface. */
#define LIBXSMM_GEMM_FLAG_DEFAULT ((1 < LIBXSMM_ALIGNED_LOADS  ? LIBXSMM_GEMM_FLAG_ALIGN_A : 0) \
                                 | (1 < LIBXSMM_ALIGNED_STORES ? LIBXSMM_GEMM_FLAG_ALIGN_C : 0))

#if defined(MKL_DIRECT_CALL_SEQ) || defined(MKL_DIRECT_CALL)
# if defined(LIBXSMM_OFFLOAD_BUILD)
#   pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#   include <mkl.h>
#   pragma offload_attribute(pop)
# else
#   include <mkl.h>
# endif
#else
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE void LIBXSMM_FSYMBOL(dgemm)(
  const char*, const char*, const int*, const int*, const int*,
  const double*, const double*, const int*, const double*, const int*,
  const double*, double*, const int*);
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE void LIBXSMM_FSYMBOL(sgemm)(
  const char*, const char*, const int*, const int*, const int*,
  const float*, const float*, const int*, const float*, const int*,
  const float*, float*, const int*);
#endif

#define LIBXSMM_BLASMM(REAL, M, N, K, A, B, C, XARGS) { \
  int libxsmm_m_ = LIBXSMM_LD(M, N), libxsmm_n_ = LIBXSMM_LD(N, M), libxsmm_k_ = (K); \
  int libxsmm_ldc_ = LIBXSMM_ALIGN_STORES(LIBXSMM_LD(M, N), sizeof(REAL)); \
  REAL libxsmm_alpha_ = 0 == (XARGS) ? ((REAL)(LIBXSMM_ALPHA)) : (XARGS)->alpha; \
  REAL libxsmm_beta_ = 0 == (XARGS) ? ((REAL)(LIBXSMM_BETA)) : (XARGS)->beta; \
  char libxsmm_trans_ = 'N'; \
  LIBXSMM_FSYMBOL(LIBXSMM_BLASPREC(REAL, gemm))(&libxsmm_trans_, &libxsmm_trans_, \
    &libxsmm_m_, &libxsmm_n_, &libxsmm_k_, &libxsmm_alpha_, \
    (REAL*)LIBXSMM_LD(A, B), &libxsmm_m_, \
    (REAL*)LIBXSMM_LD(B, A), &libxsmm_k_, \
    &libxsmm_beta_, (C), &libxsmm_ldc_); \
}

#if defined(MKL_DIRECT_CALL_SEQ) || defined(MKL_DIRECT_CALL)
# define LIBXSMM_IMM(REAL, UINT, M, N, K, A, B, C, XARGS) LIBXSMM_BLASMM(REAL, M, N, K, A, B, C, XARGS)
#else
# define LIBXSMM_IMM(REAL, UINT, M, N, K, A, B, C, XARGS) { \
    const REAL *const libxsmm_a_ = LIBXSMM_LD(B, A), *const libxsmm_b_ = LIBXSMM_LD(A, B); \
    const REAL libxsmm_alpha_ = 0 == (XARGS) ? ((REAL)(LIBXSMM_ALPHA)) : (1 == (XARGS)->alpha ? 1 : (XARGS)->alpha); \
    const REAL libxsmm_beta_ = 0 == (XARGS) ? ((REAL)(LIBXSMM_BETA)) : (1 == (XARGS)->beta ? 1 : (0 == (XARGS)->beta ? 0 : (XARGS)->beta)); \
    const UINT libxsmm_ldc_ = LIBXSMM_ALIGN_STORES(LIBXSMM_LD(M, N), sizeof(REAL)); \
    UINT libxsmm_i_, libxsmm_j_, libxsmm_k_; \
    REAL *const libxsmm_c_ = (C); \
    LIBXSMM_ASSUME_ALIGNED_LOADS(libxsmm_a_); \
    LIBXSMM_ASSUME_ALIGNED_STORES(libxsmm_c_); \
    LIBXSMM_PRAGMA_SIMD/*_COLLAPSE(2)*/ \
    for (libxsmm_j_ = 0; libxsmm_j_ < LIBXSMM_LD(M, N); ++libxsmm_j_) { \
      LIBXSMM_PRAGMA_LOOP_COUNT(1, LIBXSMM_LD(LIBXSMM_MAX_N, LIBXSMM_MAX_M), LIBXSMM_LD(LIBXSMM_AVG_N, LIBXSMM_AVG_M)) \
      for (libxsmm_i_ = 0; libxsmm_i_ < LIBXSMM_LD(N, M); ++libxsmm_i_) { \
        const UINT libxsmm_index_ = libxsmm_i_ * libxsmm_ldc_ + libxsmm_j_; \
        REAL libxsmm_r_ = libxsmm_c_[libxsmm_index_] * libxsmm_beta_; \
        LIBXSMM_PRAGMA_SIMD_REDUCTION(+:libxsmm_r_) \
        LIBXSMM_PRAGMA_UNROLL \
        for (libxsmm_k_ = 0; libxsmm_k_ < (K); ++libxsmm_k_) { \
          libxsmm_r_ += libxsmm_a_[libxsmm_i_*(K)+libxsmm_k_] * libxsmm_alpha_ \
                      * libxsmm_b_[libxsmm_k_*LIBXSMM_LD(M,N)+libxsmm_j_]; \
        } \
        libxsmm_c_[libxsmm_index_] = libxsmm_r_; \
      } \
    } \
  }
#endif

/**
 * Execute a generated function, inlined code, or fall back to the linked LAPACK implementation.
 * If M, N, and K does not change for multiple calls, it is more efficient to query and reuse
 * the function pointer (libxsmm_?dispatch).
 */
#define LIBXSMM_MM(REAL, M, N, K, A, B, C, XARGS) \
  if ((LIBXSMM_MAX_MNK) >= ((M) * (N) * (K))) { \
    const LIBXSMM_CONCATENATE(libxsmm_, LIBXSMM_BLASPREC(REAL, function)) libxsmm_function_ = \
      LIBXSMM_CONCATENATE(libxsmm_, LIBXSMM_BLASPREC(REAL, dispatch))(M, N, K, \
      0 == (XARGS) ? ((REAL)(LIBXSMM_ALPHA)) : (XARGS)->alpha, \
      0 == (XARGS) ? ((REAL)(LIBXSMM_BETA)) : (XARGS)->beta, \
      LIBXSMM_LD(M, N), K, LIBXSMM_ALIGN_STORES(LIBXSMM_LD(M, N), sizeof(REAL)), \
      LIBXSMM_GEMM_FLAG_DEFAULT, LIBXSMM_PREFETCH); \
    if (0 != libxsmm_function_) { \
      libxsmm_function_(A, B, C, XARGS); \
    } \
    else { \
      LIBXSMM_IMM(REAL, int, M, N, K, A, B, C, XARGS); \
    } \
  } \
  else { \
    LIBXSMM_BLASMM(REAL, M, N, K, A, B, C, XARGS); \
  }

#endif /*LIBXSMM_FALLBACK_H*/
