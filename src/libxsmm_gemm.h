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
#ifndef LIBXSMM_GEMM_H
#define LIBXSMM_GEMM_H

#include <libxsmm.h>

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#include <math.h>
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

#if !defined(LIBXSMM_RTLD_NEXT) && defined(LIBXSMM_BUILD) && defined(__GNUC__) && \
  !defined(_WIN32) && !defined(__CYGWIN__) && \
  !(defined(__APPLE__) && defined(__MACH__) && LIBXSMM_VERSION3(6, 1, 0) >= \
    LIBXSMM_VERSION3(__clang_major__, __clang_minor__, __clang_patchlevel__))
# define LIBXSMM_RTLD_NEXT
#endif

#if defined(LIBXSMM_BUILD_EXT)
# define LIBXSMM_GEMM_WEAK LIBXSMM_ATTRIBUTE_WEAK
#else
# define LIBXSMM_GEMM_WEAK
#endif

#if !defined(LIBXSMM_GEMM_COLLAPSE)
# define LIBXSMM_GEMM_COLLAPSE 2
#endif

#define LIBXSMM_TILED_XGEMM_THRESHOLD(M, N, K) ((0 != libxsmm_mp && ((LIBXSMM_MAX_M < (M)) || (LIBXSMM_MAX_N < (N)) || (LIBXSMM_MAX_K < (K)))) ? 1 : 0)

#define LIBXSMM_TILED_XGEMM_KERNEL(KERNEL_START, REAL, FLAGS, POS_H, POS_I, TILE_M, TILE_N, TILE_K, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) { \
  const libxsmm_blasint mm = LIBXSMM_MIN(TILE_M, (M) - (POS_H)), nn = LIBXSMM_MIN(TILE_N, (N) - (POS_I)), ic = (POS_I) * (LDC) + (POS_H); \
  libxsmm_blasint j = 0, j_next = TILE_K; \
  if (((TILE_M) == mm) && ((TILE_N) == nn)) { \
    for (; j < max_j; j = j_next) { \
      LIBXSMM_MMCALL_PRF((KERNEL_START).LIBXSMM_TPREFIX(REAL,mm), \
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

#if !defined(LIBXSMM_EXT_GEMM_BLAS)
# if !defined(__BLAS) || (0 != __BLAS)
#   define LIBXSMM_EXT_GEMM_BLAS 1
# else
#   define LIBXSMM_EXT_GEMM_BLAS 0
# endif
#endif

#define LIBXSMM_TILED_XGEMM(PARALLEL, SINGLE_OUTER, SINGLE_INNER, COLLAPSE, LOOP_START, KERNEL_START, SYNC, \
  MIN_TASKS, OVERHEAD, NT, REAL, FLAGS, TILE_M, TILE_N, TILE_K, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) \
SINGLE_OUTER { \
  const signed char scalpha = (signed char)(ALPHA), scbeta = (signed char)(BETA); \
  const int sufficient_size = LIBXSMM_TILED_XGEMM_THRESHOLD(M, N, K); \
  libxsmm_blasint tile_m = 0, tile_n = 0, tile_k = 0, num_m = 0, num_n = 0, num_k = 0; \
  libxsmm_xmmfunction xmm = { 0 }; \
  SINGLE_INNER \
  if (0 != sufficient_size \
    /*TODO: not supported*/&& 0 == ((FLAGS) & (LIBXSMM_GEMM_FLAG_TRANS_A | LIBXSMM_GEMM_FLAG_TRANS_B)) \
    /*TODO: not supported*/&& 1 == scalpha && (1 == scbeta || 0 == scbeta)) \
  { \
    tile_m = LIBXSMM_MAX(TILE_M, 2); tile_n = LIBXSMM_MAX(TILE_N, 2); tile_k = LIBXSMM_MAX(TILE_K, 2); \
    num_m = ((M) + tile_m - 1) / tile_m; num_n = ((N) + tile_n - 1) / tile_n; num_k = ((K) + tile_k - 1) / tile_k; \
    { /* opening scope for additional variable declarations */ \
      const libxsmm_blasint num_t = (OVERHEAD(NT) <= num_k && 1 < (COLLAPSE)) \
        ? (num_m * num_n) : (num_n <= num_m ? num_m : num_n); \
      const libxsmm_blasint min_ntasks = MIN_TASKS(NT); \
      libxsmm_gemm_descriptor desc; \
      if (min_ntasks < num_t) { /* ensure enough parallel slack */ \
        tile_m = (M) / num_m; tile_n = (N) / num_n; \
      } \
      else if ((OVERHEAD(NT)) <= num_k) { \
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
        tile_m = LIBXSMM_MIN(LIBXSMM_MAX((libxsmm_blasint)(1 << LIBXSMM_LOG2(tile_m * rm /*+ 0.5*/)), 8), M); \
        tile_n = LIBXSMM_MIN(LIBXSMM_MAX((libxsmm_blasint)(1 << LIBXSMM_LOG2(tile_n * rn /*+ 0.5*/)), 8), N); \
        tile_k = LIBXSMM_MIN(LIBXSMM_MAX((libxsmm_blasint)(1 << LIBXSMM_LOG2(tile_k * rk /*+ 0.5*/)), 8), K); \
      } \
      LIBXSMM_GEMM_DESCRIPTOR(desc, LIBXSMM_ALIGNMENT, FLAGS, tile_m, tile_n, tile_k, \
        LDA, LDB, LDC, scalpha, scbeta, libxsmm_gemm_prefetch); \
      xmm = libxsmm_xmmdispatch(&desc); \
    } \
  } \
  if (0 != xmm.dmm) { \
    const libxsmm_blasint max_j = ((K) / tile_k) * tile_k; \
    libxsmm_blasint h = 0, i = 0; \
    if ((OVERHEAD(NT)) <= num_k) { /* amortize overhead */ \
      PARALLEL LOOP_START(COLLAPSE) \
      for (h = 0; h < (M); h += tile_m) { \
        for (i = 0; i < (N); i += tile_n) { \
          KERNEL_START(h, i) \
          LIBXSMM_TILED_XGEMM_KERNEL(xmm, REAL, FLAGS, h, i, tile_m, tile_n, tile_k, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC); \
        } \
      } \
    } \
    else if (num_n <= num_m) { \
      PARALLEL LOOP_START(COLLAPSE) \
      for (h = 0; h < (M); h += tile_m) { \
        KERNEL_START(h) \
        for (i = 0; i < (N); i += tile_n) { \
          LIBXSMM_TILED_XGEMM_KERNEL(xmm, REAL, FLAGS, h, i, tile_m, tile_n, tile_k, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC); \
        } \
      } \
    } \
    else { \
      PARALLEL LOOP_START(COLLAPSE) \
      for (i = 0; i < (N); i += tile_n) { \
        KERNEL_START(i) \
        for (h = 0; h < (M); h += tile_m) { \
          LIBXSMM_TILED_XGEMM_KERNEL(xmm, REAL, FLAGS, h, i, tile_m, tile_n, tile_k, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC); \
        } \
      } \
    } \
    SYNC \
  } \
  else if (0 != sufficient_size && 0 != LIBXSMM_EXT_GEMM_BLAS) { /* BLAS fall-back */ \
    LIBXSMM_BLAS_XGEMM(REAL, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC); \
  } \
  else { /* small problem size */ \
    LIBXSMM_XGEMM(REAL, libxsmm_blasint, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC); \
  } \
}


/**
 * INTERNAL pre-initialization step called by libxsmm_gemm_init,
 * e.g. configures the tile sizes for multi-threaded GEMM functions.
 */
LIBXSMM_API void libxsmm_gemm_configure(int archid, int prefetch);

/** Provides GEMM functions available via BLAS; NOT thread-safe. */
LIBXSMM_API int libxsmm_gemm_init(
  int archid, int prefetch/*default prefetch strategy*/);

/** Finalizes the GEMM facility; NOT thread-safe. */
LIBXSMM_API void libxsmm_gemm_finalize(void);

LIBXSMM_EXTERN LIBXSMM_RETARGETABLE LIBXSMM_GEMM_WEAK void LIBXSMM_FSYMBOL(sgemm)(
  const char*, const char*, const libxsmm_blasint*, const libxsmm_blasint*, const libxsmm_blasint*,
  const float*, const float*, const libxsmm_blasint*, const float*, const libxsmm_blasint*,
  const float*, float*, const libxsmm_blasint*);
LIBXSMM_EXTERN LIBXSMM_RETARGETABLE LIBXSMM_GEMM_WEAK void LIBXSMM_FSYMBOL(dgemm)(
  const char*, const char*, const libxsmm_blasint*, const libxsmm_blasint*, const libxsmm_blasint*,
  const double*, const double*, const libxsmm_blasint*, const double*, const libxsmm_blasint*,
  const double*, double*, const libxsmm_blasint*);

LIBXSMM_EXTERN LIBXSMM_RETARGETABLE LIBXSMM_ATTRIBUTE_WEAK void LIBXSMM_FSYMBOL(__real_sgemm)(
  const char*, const char*, const libxsmm_blasint*, const libxsmm_blasint*, const libxsmm_blasint*,
  const float*, const float*, const libxsmm_blasint*, const float* b, const libxsmm_blasint*,
  const float*, float*, const libxsmm_blasint*);
LIBXSMM_EXTERN LIBXSMM_RETARGETABLE LIBXSMM_ATTRIBUTE_WEAK void LIBXSMM_FSYMBOL(__real_dgemm)(
  const char*, const char*, const libxsmm_blasint*, const libxsmm_blasint*, const libxsmm_blasint*,
  const double*, const double*, const libxsmm_blasint*, const double* b, const libxsmm_blasint*,
  const double*, double*, const libxsmm_blasint*);

LIBXSMM_API void LIBXSMM_FSYMBOL(__wrap_sgemm)(
  const char*, const char*, const libxsmm_blasint*, const libxsmm_blasint*, const libxsmm_blasint*,
  const float*, const float*, const libxsmm_blasint*, const float* b, const libxsmm_blasint*,
  const float*, float*, const libxsmm_blasint*);
LIBXSMM_API void LIBXSMM_FSYMBOL(__wrap_dgemm)(
  const char*, const char*, const libxsmm_blasint*, const libxsmm_blasint*, const libxsmm_blasint*,
  const double*, const double*, const libxsmm_blasint*, const double* b, const libxsmm_blasint*,
  const double*, double*, const libxsmm_blasint*);

/** Configuration table containing the tile sizes separate for DP and SP. */
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE int libxsmm_gemm_tile[2/*DP/SP*/][3/*TILE_M,TILE_N,TILE_K*/];
/** Prefetch strategy. */
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE int libxsmm_gemm_prefetch /*= LIBXSMM_MAX(LIBXSMM_PREFETCH, 0)*/;

#endif /*LIBXSMM_GEMM_H*/

