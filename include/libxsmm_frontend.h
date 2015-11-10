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

#if (0 != LIBXSMM_ROW_MAJOR)
# define LIBXSMM_LD(M, N) (N)
#else
# define LIBXSMM_LD(M, N) (M)
#endif

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

/** BLAS based implementation with simplified interface. */
#define LIBXSMM_BLASMM(REAL, FLAGS, M, N, K, A, B, C, ALPHA, BETA) { \
  /*const*/char libxsmm_transa_ = 0 == (LIBXSMM_GEMM_FLAG_TRANS_A & (FLAGS)) ? 'N' : 'T'; \
  /*const*/char libxsmm_transb_ = 0 == (LIBXSMM_GEMM_FLAG_TRANS_B & (FLAGS)) ? 'N' : 'T'; \
  /*const*/int libxsmm_m_ = LIBXSMM_LD(M, N), libxsmm_n_ = LIBXSMM_LD(N, M), libxsmm_k_ = (K); \
  /*const*/int libxsmm_lda_ = 0 == (LIBXSMM_GEMM_FLAG_ALIGN_A & (FLAGS)) ? libxsmm_m_ : \
    LIBXSMM_ALIGN_VALUE(libxsmm_m_, sizeof(REAL), LIBXSMM_ALIGNMENT); \
  /*const*/int libxsmm_ldc_ = 0 == (LIBXSMM_GEMM_FLAG_ALIGN_C & (FLAGS)) ? libxsmm_m_ : \
    LIBXSMM_ALIGN_VALUE(libxsmm_m_, sizeof(REAL), LIBXSMM_ALIGNMENT); \
  /*const*/REAL libxsmm_alpha_ = 0 == (ALPHA) ? ((REAL)LIBXSMM_ALPHA) : *(ALPHA); \
  /*const*/REAL libxsmm_beta_  = 0 == (BETA)  ? ((REAL)LIBXSMM_BETA)  : *(BETA); \
  LIBXSMM_FSYMBOL(LIBXSMM_BLASPREC(REAL, gemm))(&libxsmm_transa_, &libxsmm_transb_, \
    &libxsmm_m_, &libxsmm_n_, &libxsmm_k_, &libxsmm_alpha_, \
    (REAL*)LIBXSMM_LD(A, B), &libxsmm_lda_, \
    (REAL*)LIBXSMM_LD(B, A), &libxsmm_k_, \
    &libxsmm_beta_, (C), &libxsmm_ldc_); \
}

/** Inlinable implementation exercising the compiler's code generation (template). */
#if defined(MKL_DIRECT_CALL_SEQ) || defined(MKL_DIRECT_CALL)
# define LIBXSMM_IMM(REAL, UINT, FLAGS, M, N, K, A, B, C, ALPHA, BETA) LIBXSMM_BLASMM(REAL, FLAGS, M, N, K, A, B, C, ALPHA, BETA)
#else
# define LIBXSMM_IMM(REAL, UINT, FLAGS, M, N, K, A, B, C, ALPHA, BETA) { \
  const REAL *const libxsmm_a_ = LIBXSMM_LD(B, A), *const libxsmm_b_ = LIBXSMM_LD(A, B); \
  const REAL libxsmm_alpha_ = 0 == (ALPHA) ? ((REAL)LIBXSMM_ALPHA) : (((REAL)1) == *(ALPHA) ? ((REAL)1) : (((REAL)-1) == *(ALPHA) ? ((REAL)-1) : *(ALPHA))); \
  const REAL libxsmm_beta_  = 0 == (BETA)  ? ((REAL)LIBXSMM_BETA)  : (((REAL)1) == *(BETA)  ? ((REAL)1) : (((REAL) 0) == *(BETA)  ? ((REAL) 0) : *(BETA))); \
  const UINT libxsmm_m_ = LIBXSMM_LD(M, N), libxsmm_n_ = LIBXSMM_LD(N, M); \
  const UINT libxsmm_lda_ = 0 == (LIBXSMM_GEMM_FLAG_ALIGN_A & (FLAGS)) ? libxsmm_m_ : \
    LIBXSMM_ALIGN_VALUE(libxsmm_m_, sizeof(REAL), LIBXSMM_ALIGNMENT); \
  const UINT libxsmm_ldc_ = 0 == (LIBXSMM_GEMM_FLAG_ALIGN_C & (FLAGS)) ? libxsmm_m_ : \
    LIBXSMM_ALIGN_VALUE(libxsmm_m_, sizeof(REAL), LIBXSMM_ALIGNMENT); \
  UINT libxsmm_i_, libxsmm_j_, libxsmm_k_; \
  REAL *const libxsmm_c_ = (C); \
  assert(0 == (LIBXSMM_GEMM_FLAG_TRANS_A & (FLAGS)) && 0 == (LIBXSMM_GEMM_FLAG_TRANS_B & (FLAGS))/*not supported*/); \
  LIBXSMM_PRAGMA_SIMD/*_COLLAPSE(2)*/ \
  for (libxsmm_j_ = 0; libxsmm_j_ < libxsmm_m_; ++libxsmm_j_) { \
    LIBXSMM_PRAGMA_LOOP_COUNT(1, LIBXSMM_LD(LIBXSMM_MAX_N, LIBXSMM_MAX_M), LIBXSMM_LD(LIBXSMM_AVG_N, LIBXSMM_AVG_M)) \
    for (libxsmm_i_ = 0; libxsmm_i_ < libxsmm_n_; ++libxsmm_i_) { \
      const UINT libxsmm_index_ = libxsmm_i_ * libxsmm_ldc_ + libxsmm_j_; \
      REAL libxsmm_r_ = libxsmm_c_[libxsmm_index_] * libxsmm_beta_; \
      LIBXSMM_PRAGMA_SIMD_REDUCTION(+:libxsmm_r_) \
      LIBXSMM_PRAGMA_UNROLL \
      for (libxsmm_k_ = 0; libxsmm_k_ < (K); ++libxsmm_k_) { \
        libxsmm_r_ += libxsmm_a_[libxsmm_i_*(K)+libxsmm_k_] * libxsmm_alpha_ \
                    * libxsmm_b_[libxsmm_k_*libxsmm_lda_+libxsmm_j_]; \
      } \
      libxsmm_c_[libxsmm_index_] = libxsmm_r_; \
    } \
  } \
}
#endif

/** Inlinable implementation exercising the compiler's code generation (single-precision). */
#define LIBXSMM_SIMM(FLAGS, M, N, K, A, B, C, ALPHA, BETA) \
  LIBXSMM_IMM(float, int, FLAGS, M, N, K, A, B, C, ALPHA, BETA)
/** Inlinable implementation exercising the compiler's code generation (double-precision). */
#define LIBXSMM_DIMM(FLAGS, M, N, K, A, B, C, ALPHA, BETA) \
  LIBXSMM_IMM(double, int, FLAGS, M, N, K, A, B, C, ALPHA, BETA)
/** Inlinable implementation exercising the compiler's code generation. */
#define LIBXSMM_XIMM(FLAGS, M, N, K, A, B, C, ALPHA, BETA) { \
  if (sizeof(double) == sizeof(*(A))) { \
    LIBXSMM_DIMM(FLAGS, M, N, K, (const double*)(A), (const double*)(B), (double*)(C), (const double*)(ALPHA), (const double*)(BETA)); \
  } \
  else {\
    LIBXSMM_SIMM(FLAGS, M, N, K, (const float*)(A), (const float*)(B), (float*)(C), (const float*)(ALPHA), (const float*)(BETA)); \
  } \
}

/** Fallback code paths: LIBXSMM_FALLBACK0, and LIBXSMM_FALLBACK1. */
#if defined(LIBXSMM_FALLBACK_IMM)
# define LIBXSMM_FALLBACK0(REAL, FLAGS, M, N, K, A, B, C, ALPHA, BETA) \
    LIBXSMM_IMM(REAL, int, FLAGS, M, N, K, A, B, C, ALPHA, BETA)
#else
# define LIBXSMM_FALLBACK0(REAL, FLAGS, M, N, K, A, B, C, ALPHA, BETA) \
    LIBXSMM_BLASMM(REAL, FLAGS, M, N, K, A, B, C, ALPHA, BETA)
#endif
#define LIBXSMM_FALLBACK1(REAL, FLAGS, M, N, K, A, B, C, ALPHA, BETA) \
  LIBXSMM_BLASMM(REAL, FLAGS, M, N, K, A, B, C, ALPHA, BETA)

/**
 * Execute a specialized function, or use a fallback code path depending on threshold.
 * LIBXSMM_FALLBACK0 or specialized function: below LIBXSMM_MAX_MNK
 * LIBXSMM_FALLBACK1: above LIBXSMM_MAX_MNK
 */
#define LIBXSMM_MM(REAL, FLAGS, M, N, K, A, B, C, PA, PB, PC, ALPHA, BETA) { \
  if (LIBXSMM_MAX_MNK >= ((M) * (N) * (K))) { \
    int libxsmm_fallback_ = 0; \
    if (0 == (ALPHA) && 0 == (BETA)) { /* function0 or function1 */ \
      if (0 != (PA) || 0 != (PB) || 0 != (PC)) { /* function1 */ \
        const LIBXSMM_CONCATENATE(libxsmm_, LIBXSMM_BLASPREC(REAL, function1)) libxsmm_function_ = \
          LIBXSMM_CONCATENATE(libxsmm_, LIBXSMM_BLASPREC(REAL, dispatch1))(FLAGS, M, N, K, \
            LIBXSMM_LD(M, N), K, LIBXSMM_LD(M, N), LIBXSMM_PREFETCH); \
        if (0 != libxsmm_function_) { \
          const REAL *const libxsmm_pa_ = ((0 == (LIBXSMM_PREFETCH & LIBXSMM_PREFETCH_AL2) && 0 == (LIBXSMM_PREFETCH & LIBXSMM_PREFETCH_AL2_JPST)) \
            || 0 == (PA)) ? (A) : (PA); \
          const REAL *const libxsmm_pb_ = ((0 == (LIBXSMM_PREFETCH & LIBXSMM_PREFETCH_BL2_VIA_C)) \
            || 0 == (PB)) ? (B) : (PB); \
          const REAL *const libxsmm_pc_ = (0 == (PC)) ? (C) : (PC); \
          libxsmm_function_(A, B, C, libxsmm_pa_, libxsmm_pb_, libxsmm_pc_); \
        } \
        else { \
          libxsmm_fallback_ = 1; \
        } \
      } \
      else { /* function0 */ \
        const LIBXSMM_CONCATENATE(libxsmm_, LIBXSMM_BLASPREC(REAL, function0)) libxsmm_function_ = \
          LIBXSMM_CONCATENATE(libxsmm_, LIBXSMM_BLASPREC(REAL, dispatch))(FLAGS, M, N, K, \
            LIBXSMM_LD(M, N), K, LIBXSMM_LD(M, N)); \
        if (0 != libxsmm_function_) { \
          libxsmm_function_(A, B, C); \
        } \
        else { \
          libxsmm_fallback_ = 1; \
        } \
      } \
    } \
    else { /* function2 */ \
      const LIBXSMM_CONCATENATE(libxsmm_, LIBXSMM_BLASPREC(REAL, function2)) libxsmm_function_ = \
        LIBXSMM_CONCATENATE(libxsmm_, LIBXSMM_BLASPREC(REAL, dispatch2))(FLAGS, M, N, K, \
          LIBXSMM_LD(M, N), K, LIBXSMM_LD(M, N), LIBXSMM_PREFETCH, *(ALPHA), *(BETA)); \
      if (0 != libxsmm_function_) { \
        const REAL *const libxsmm_pa_ = ((0 == (LIBXSMM_PREFETCH & LIBXSMM_PREFETCH_AL2) && 0 == (LIBXSMM_PREFETCH & LIBXSMM_PREFETCH_AL2_JPST)) \
          || 0 == (PA)) ? (A) : (PA); \
        const REAL *const libxsmm_pb_ = ((0 == (LIBXSMM_PREFETCH & LIBXSMM_PREFETCH_BL2_VIA_C)) \
          || 0 == (PB)) ? (B) : (PB); \
        const REAL *const libxsmm_pc_ = (0 == (PC)) ? (C) : (PC); \
        libxsmm_function_(A, B, C, libxsmm_pa_, libxsmm_pb_, libxsmm_pc_, *(ALPHA), *(BETA)); \
      } \
      else { \
        libxsmm_fallback_ = 1; \
      } \
    } \
    if (0 != libxsmm_fallback_) { \
      LIBXSMM_FALLBACK0(REAL, FLAGS, M, N, K, A, B, C, ALPHA, BETA); \
    } \
  } \
  else { \
    LIBXSMM_FALLBACK1(REAL, FLAGS, M, N, K, A, B, C, ALPHA, BETA); \
  } \
}

#endif /*LIBXSMM_FRONTEND_H*/
