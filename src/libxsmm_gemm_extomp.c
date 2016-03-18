/******************************************************************************
** Copyright (c) 2016, Intel Corporation                                     **
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
#include "libxsmm_gemm_ext.h"

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#include <math.h>
#if defined(_OPENMP)
# include <omp.h>
#endif
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

#if defined(_OPENMP)
# if !defined(LIBXSMM_GEMM_EXTOMP_TASKS) && (200805 <= _OPENMP) /*OpenMP 3.0*/
#   define LIBXSMM_GEMM_EXTOMP_TASKS
# endif
# define LIBXSMM_GEMM_EXTOMP_MIN_NTASKS(NT) (40 * omp_get_num_threads() / (NT))
# define LIBXSMM_GEMM_EXTOMP_OVERHEAD(NT) (4 * (NT))
# if defined(LIBXSMM_GEMM_EXTOMP_TASKS)
#   define LIBXSMM_GEMM_EXTOMP_START LIBXSMM_PRAGMA(omp single nowait)
#   define LIBXSMM_GEMM_EXTOMP_TASK_SYNC LIBXSMM_PRAGMA(omp taskwait)
#   define LIBXSMM_GEMM_EXTOMP_TASK(...) LIBXSMM_PRAGMA(omp task firstprivate(__VA_ARGS__))
#   define LIBXSMM_GEMM_EXTOMP_FOR(N)
# else
#   define LIBXSMM_GEMM_EXTOMP_START
#   define LIBXSMM_GEMM_EXTOMP_TASK_SYNC
#   define LIBXSMM_GEMM_EXTOMP_TASK(...)
#   define LIBXSMM_GEMM_EXTOMP_FOR(N) /*LIBXSMM_PRAGMA(omp for LIBXSMM_OPENMP_COLLAPSE(N) schedule(dynamic))*/
# endif
#else
# define LIBXSMM_GEMM_EXTOMP_MIN_NTASKS(NT) 1
# define LIBXSMM_GEMM_EXTOMP_OVERHEAD(NT) 0
# define LIBXSMM_GEMM_EXTOMP_START
# define LIBXSMM_GEMM_EXTOMP_TASK_SYNC
# define LIBXSMM_GEMM_EXTOMP_TASK(...)
# define LIBXSMM_GEMM_EXTOMP_FOR(N)
#endif

#define LIBXSMM_GEMM_EXTOMP_XGEMM_TASK(REAL, FLAGS, POS_H, POS_I, TILE_M, TILE_N, TILE_K, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) { \
  const libxsmm_blasint mm = LIBXSMM_MIN(TILE_M, (M) - (POS_H)), nn = LIBXSMM_MIN(TILE_N, (N) - (POS_I)), ic = (POS_I) * (LDC) + (POS_H); \
  libxsmm_blasint j = 0, j_next = TILE_K; \
  if (((TILE_M) == mm) && ((TILE_N) == nn)) { \
    for (; j < max_j; j = j_next) { \
      LIBXSMM_MMCALL_PRF(xmm.LIBXSMM_TPREFIX(REAL,mm), \
        (A) + (POS_H) + (LDA) * j, \
        (B) + (POS_I) * (LDB) + j, \
        (C) + ic, \
        (A) + (POS_H) + (LDA) * j_next, \
        (B) + (POS_I) * (LDB) + j_next, \
        (C) + ic); \
      j_next = j + (TILE_K); \
    } \
  } \
  for (; j < (K); j = j_next) { /* remainder */ \
    LIBXSMM_XGEMM(REAL, libxsmm_blasint, FLAGS, mm, nn, LIBXSMM_MIN(TILE_K, (K) - j), \
      ALPHA, (A) + j * (LDA) + (POS_H), LDA, (B) + (POS_I) * (LDB) + j, LDB, BETA, (C) + ic, LDC); \
    j_next = j + (TILE_K); \
  } \
}

#define LIBXSMM_GEMM_EXTOMP_XGEMM(REAL, FLAGS, NT, TILE_M, TILE_N, TILE_K, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) { \
  libxsmm_blasint tile_m = LIBXSMM_MAX(TILE_M, 2), tile_n = LIBXSMM_MAX(TILE_N, 2), tile_k = LIBXSMM_MAX(TILE_K, 2); \
  const libxsmm_blasint num_m = ((M) + tile_m - 1) / tile_m, num_n = ((N) + tile_n - 1) / tile_n, num_k = ((K) + tile_k - 1) / tile_k; \
  const signed char scalpha = (signed char)(ALPHA), scbeta = (signed char)(BETA); \
  libxsmm_xmmfunction xmm; \
  LIBXSMM_GEMM_EXTOMP_START \
  if (0 == ((FLAGS) & (LIBXSMM_GEMM_FLAG_TRANS_A | LIBXSMM_GEMM_FLAG_TRANS_B)) && 1 == scalpha && (1 == scbeta || 0 == scbeta)) { \
    const libxsmm_blasint num_t = (LIBXSMM_GEMM_EXTOMP_OVERHEAD(NT)) <= num_k ? (num_m * num_n) : (num_n <= num_m ? num_m : num_n); \
    const libxsmm_blasint min_ntasks = LIBXSMM_GEMM_EXTOMP_MIN_NTASKS(NT); \
    libxsmm_gemm_descriptor desc; \
    if (min_ntasks < num_t) { /* ensure enough parallel slack */ \
      tile_m = (M) / num_m; tile_n = (N) / num_n; \
    } \
    else if ((LIBXSMM_GEMM_EXTOMP_OVERHEAD(NT)) <= num_k) { \
      const double ratio = sqrt(((double)min_ntasks) / num_t); \
      tile_n = (int)(num_n * ratio /*+ 0.5*/); \
      tile_m = (min_ntasks + tile_n - 1) / tile_n; \
    } \
    else if (num_n <= num_m) { \
      tile_m = ((M) + min_ntasks - 1) / min_ntasks; \
    } \
    else { \
      tile_n = ((N) + min_ntasks - 1) / min_ntasks; \
    } \
    { /* adjust for non-square operand shapes */ \
      float rm = 1.f, rn = ((float)(N)) / M, rk = ((float)(K)) / M; \
      if (1.f < rn) { rm /= rn; rn = 1.f; rk /= rn; } \
      if (1.f < rk) { rm /= rk; rn /= rk; rk = 1.f; } \
      tile_m = LIBXSMM_MIN(LIBXSMM_MAX((libxsmm_blasint)(2 << LIBXSMM_LOG2(tile_m * rm /*+ 0.5*/)),  8), M); \
      tile_n = LIBXSMM_MIN(LIBXSMM_MAX((libxsmm_blasint)(2 << LIBXSMM_LOG2(tile_n * rn /*+ 0.5*/)),  8), N); \
      tile_k = LIBXSMM_MIN(LIBXSMM_MAX((libxsmm_blasint)(2 << LIBXSMM_LOG2(tile_k * rk /*+ 0.5*/)), 16), K); \
    } \
    LIBXSMM_GEMM_DESCRIPTOR(desc, LIBXSMM_ALIGNMENT, FLAGS, tile_m, tile_n, tile_k, LDA, LDB, LDC, scalpha, scbeta, LIBXSMM_PREFETCH_AL2_AHEAD); \
    xmm = libxsmm_xmmdispatch(&desc); \
  } \
  else { /* TODO: not supported (bypass) */ \
    xmm.dmm = 0; \
  } \
  if (0 != xmm.dmm) { \
    LIBXSMM_GEMM_EXTOMP_START \
    { \
      const libxsmm_blasint max_j = ((K) / tile_k) * tile_k; \
      libxsmm_blasint h = 0, i = 0; \
      if ((LIBXSMM_GEMM_EXTOMP_OVERHEAD(NT)) <= num_k) { /* amortize overhead */ \
        LIBXSMM_GEMM_EXTOMP_FOR(2) \
        for (h = 0; h < (M); h += tile_m) { \
          for (i = 0; i < (N); i += tile_n) { \
            LIBXSMM_GEMM_EXTOMP_TASK(h, i) \
            LIBXSMM_GEMM_EXTOMP_XGEMM_TASK(REAL, FLAGS, h, i, tile_m, tile_n, tile_k, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC); \
          } \
        } \
        LIBXSMM_GEMM_EXTOMP_TASK_SYNC \
      } \
      else if (num_n <= num_m) { \
        LIBXSMM_GEMM_EXTOMP_FOR(2) \
        for (h = 0; h < (M); h += tile_m) { \
          LIBXSMM_GEMM_EXTOMP_TASK(h) \
          for (i = 0; i < (N); i += tile_n) { \
            LIBXSMM_GEMM_EXTOMP_XGEMM_TASK(REAL, FLAGS, h, i, tile_m, tile_n, tile_k, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC); \
          } \
        } \
        LIBXSMM_GEMM_EXTOMP_TASK_SYNC \
      } \
      else { \
        LIBXSMM_GEMM_EXTOMP_FOR(2) \
        for (i = 0; i < (N); i += tile_n) { \
          LIBXSMM_GEMM_EXTOMP_TASK(i) \
          for (h = 0; h < (M); h += tile_m) { \
            LIBXSMM_GEMM_EXTOMP_XGEMM_TASK(REAL, FLAGS, h, i, tile_m, tile_n, tile_k, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC); \
          } \
        } \
        LIBXSMM_GEMM_EXTOMP_TASK_SYNC \
      } \
    } \
  } \
  else { /* fallback */ \
    LIBXSMM_BLAS_XGEMM(REAL, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC); \
  } \
}


LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE void libxsmm_omps_sgemm(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const float* alpha, const float* a, const libxsmm_blasint* lda,
  const float* b, const libxsmm_blasint* ldb,
  const float* beta, float* c, const libxsmm_blasint* ldc)
{
  LIBXSMM_GEMM_DECLARE_FLAGS(flags, transa, transb, m, n, k, a, b, c);
  LIBXSMM_GEMM_EXTOMP_XGEMM(float, flags | LIBXSMM_GEMM_FLAG_F32PREC, libxsmm_internal_num_nt,
    libxsmm_internal_tile_size[1/*SP*/][0/*M*/],
    libxsmm_internal_tile_size[1/*SP*/][1/*N*/],
    libxsmm_internal_tile_size[1/*SP*/][2/*K*/], *m, *n, *k,
    0 != alpha ? *alpha : ((float)LIBXSMM_ALPHA),
    a, *(lda ? lda : LIBXSMM_LD(m, k)), b, *(ldb ? ldb : LIBXSMM_LD(k, n)),
    0 != beta ? *beta : ((float)LIBXSMM_BETA),
    c, *(ldc ? ldc : LIBXSMM_LD(m, n)));
}


LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE void libxsmm_omps_dgemm(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const double* alpha, const double* a, const libxsmm_blasint* lda,
  const double* b, const libxsmm_blasint* ldb,
  const double* beta, double* c, const libxsmm_blasint* ldc)
{
  LIBXSMM_GEMM_DECLARE_FLAGS(flags, transa, transb, m, n, k, a, b, c);
  LIBXSMM_GEMM_EXTOMP_XGEMM(double, flags, libxsmm_internal_num_nt,
    libxsmm_internal_tile_size[0/*DP*/][0/*M*/],
    libxsmm_internal_tile_size[0/*DP*/][1/*N*/],
    libxsmm_internal_tile_size[0/*DP*/][2/*K*/], *m, *n, *k,
    0 != alpha ? *alpha : ((double)LIBXSMM_ALPHA),
    a, *(lda ? lda : LIBXSMM_LD(m, k)), b, *(ldb ? ldb : LIBXSMM_LD(k, n)),
    0 != beta ? *beta : ((double)LIBXSMM_BETA),
    c, *(ldc ? ldc : LIBXSMM_LD(m, n)));
}


#if defined(LIBXSMM_GEMM_EXTWRAP)

LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE void LIBXSMM_GEMM_EXTWRAP_SGEMM(
  const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const float* alpha, const float* a, const libxsmm_blasint* lda,
  const float* b, const libxsmm_blasint* ldb,
  const float* beta, float* c, const libxsmm_blasint* ldc)
{
  assert(LIBXSMM_GEMM_EXTWRAP_SGEMM != libxsmm_internal_sgemm);
  switch (libxsmm_internal_gemm) {
    case 1: {
      libxsmm_omps_sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    } break;
    case 2: {
#if defined(_OPENMP)
#     pragma omp parallel
#     pragma omp single
#endif
      libxsmm_omps_sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    } break;
    default: {
      LIBXSMM_GEMM_DECLARE_FLAGS(flags, transa, transb, m, n, k, a, b, c);
      LIBXSMM_XGEMM(float, libxsmm_blasint, flags, *m, *n, *k,
        0 != alpha ? *alpha : ((float)LIBXSMM_ALPHA),
        a, *(lda ? lda : LIBXSMM_LD(m, k)), b, *(ldb ? ldb : LIBXSMM_LD(k, n)),
        0 != beta ? *beta : ((float)LIBXSMM_BETA),
        c, *(ldc ? ldc : LIBXSMM_LD(m, n)));
    }
  }
}


LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE void LIBXSMM_GEMM_EXTWRAP_DGEMM(
  const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const double* alpha, const double* a, const libxsmm_blasint* lda,
  const double* b, const libxsmm_blasint* ldb,
  const double* beta, double* c, const libxsmm_blasint* ldc)
{
  assert(LIBXSMM_GEMM_EXTWRAP_DGEMM != libxsmm_internal_dgemm);
  switch (libxsmm_internal_gemm) {
    case 1: {
      libxsmm_omps_dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    } break;
    case 2: {
#if defined(_OPENMP)
#     pragma omp parallel
#     pragma omp single
#endif
      libxsmm_omps_dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    } break;
    default: {
      LIBXSMM_GEMM_DECLARE_FLAGS(flags, transa, transb, m, n, k, a, b, c);
      LIBXSMM_XGEMM(double, libxsmm_blasint, flags, *m, *n, *k,
        0 != alpha ? *alpha : ((double)LIBXSMM_ALPHA),
        a, *(lda ? lda : LIBXSMM_LD(m, k)), b, *(ldb ? ldb : LIBXSMM_LD(k, n)),
        0 != beta ? *beta : ((double)LIBXSMM_BETA),
        c, *(ldc ? ldc : LIBXSMM_LD(m, n)));
    }
  }
}

#endif /*defined(LIBXSMM_GEMM_EXTWRAP)*/

