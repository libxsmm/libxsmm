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
#include "libxsmm_gemm.h"

#if defined(__STATIC)
# include "libxsmm_gemm_wrap.c"
#else
# include "libxsmm_gemm_wrap.h"
#endif

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#if defined(_OPENMP)
# include <omp.h>
#endif
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

#if defined(_OPENMP)
# if !defined(LIBXSMM_GEMM_OMPS_TASKS) && (200805 <= _OPENMP) /*OpenMP 3.0*/
#   define LIBXSMM_GEMM_OMPS_TASKS
# endif
# define LIBXSMM_GEMM_OMPS_MIN_NTASKS (20 * omp_get_num_threads())
# if defined(LIBXSMM_GEMM_OMPS_TASKS)
#   define LIBXSMM_GEMM_OMPS_START LIBXSMM_PRAGMA(omp single nowait)
#   define LIBXSMM_GEMM_OMPS_TASK_SYNC LIBXSMM_PRAGMA(omp taskwait)
#   define LIBXSMM_GEMM_OMPS_TASK(...) LIBXSMM_PRAGMA(omp task firstprivate(__VA_ARGS__))
#   define LIBXSMM_GEMM_OMPS_FOR(N)
# else
#   define LIBXSMM_GEMM_OMPS_START
#   define LIBXSMM_GEMM_OMPS_TASK_SYNC
#   define LIBXSMM_GEMM_OMPS_TASK(...)
#   define LIBXSMM_GEMM_OMPS_FOR(N) /*LIBXSMM_PRAGMA(omp for LIBXSMM_OPENMP_COLLAPSE(N) schedule(dynamic))*/
# endif
#else
# define LIBXSMM_GEMM_OMPS_MIN_NTASKS 1
# define LIBXSMM_GEMM_OMPS_START
# define LIBXSMM_GEMM_OMPS_TASK_SYNC
# define LIBXSMM_GEMM_OMPS_TASK(...)
# define LIBXSMM_GEMM_OMPS_FOR(N)
#endif

#define LIBXSMM_GEMM_OMPS_XGEMM(REAL, FLAGS, TILE_M, TILE_N, TILE_K, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) { \
  libxsmm_blasint tile_m = LIBXSMM_MAX(TILE_M, 2), tile_n = LIBXSMM_MAX(TILE_N, 2), tile_k = LIBXSMM_MAX(TILE_K, 2); \
  libxsmm_xmmfunction xmm; \
  LIBXSMM_GEMM_OMPS_START \
  { \
    const libxsmm_blasint num_m = ((M) + tile_m - 1) / tile_m, num_n = ((N) + tile_n - 1) / tile_n, num_t = num_m * num_n; \
    const libxsmm_blasint min_ntasks = LIBXSMM_GEMM_OMPS_MIN_NTASKS; \
    libxsmm_gemm_descriptor desc; \
    if (num_t < min_ntasks) { /* ensure some parallel slack */ \
      const double ratio = sqrt(((double)min_ntasks) / num_t); \
      tile_n = (int)(num_n * ratio /*+ 0.5*/); tile_m = (min_ntasks + tile_n - 1) / tile_n; \
    } \
    else { \
      tile_m = (M) / num_m; tile_n = (N) / num_n; \
    } \
    { /* adjust for non-square operand shapes */ \
      float rm = 1.f, rn = ((float)(N)) / M, rk = ((float)(K)) / M; \
      if (1.f < rn) { \
        rm /= rn; rn = 1.f; rk /= rn; \
      } \
      if (1.f < rk) { \
        rm /= rk; rn /= rk; rk = 1.f; \
      } \
      tile_m = LIBXSMM_MIN(LIBXSMM_MAX((libxsmm_blasint)(1 << LIBXSMM_NBITS(tile_m * rm /*+ 0.5*/)),  8), M); \
      tile_n = LIBXSMM_MIN(LIBXSMM_MAX((libxsmm_blasint)(1 << LIBXSMM_NBITS(tile_n * rn /*+ 0.5*/)),  8), N); \
      tile_k = LIBXSMM_MIN(LIBXSMM_MAX((libxsmm_blasint)(1 << LIBXSMM_NBITS(tile_k * rk /*+ 0.5*/)), 32), K); \
    } \
    LIBXSMM_GEMM_DESCRIPTOR(desc, LIBXSMM_ALIGNMENT, FLAGS, tile_m, tile_n, tile_k, LDA, LDB, LDC, ALPHA, BETA, LIBXSMM_PREFETCH); \
    xmm = libxsmm_xmmdispatch(&desc); \
  } \
  if (0 != xmm.dmm) { \
    LIBXSMM_GEMM_OMPS_START \
    { \
      const libxsmm_blasint max_j = ((K) / tile_k) * tile_k; \
      libxsmm_blasint h, i; \
      LIBXSMM_GEMM_OMPS_FOR(2) \
      for (i = 0; i < (N); i += tile_n) { \
        for (h = 0; h < (M); h += tile_m) { \
          LIBXSMM_GEMM_OMPS_TASK(h, i) \
          { \
            const libxsmm_blasint mm = LIBXSMM_MIN(tile_m, (M) - h); \
            const libxsmm_blasint nn = LIBXSMM_MIN(tile_n, (N) - i); \
            const libxsmm_blasint ic = i * (LDC) + h; \
            libxsmm_blasint j = 0; \
            if ((tile_m == mm) && (tile_n == nn)) { \
              for (; j < max_j; j += tile_k) { \
                LIBXSMM_MMCALL(xmm.LIBXSMM_TPREFIX(REAL,mm), (A) + j * (LDA) + h, (B) + i * (LDB) + j, (C) + ic, M, N, K, LDA, LDB, LDC); \
              } \
            } \
            for (; j < (K); j += tile_k) { /* remainder */ \
              LIBXSMM_XGEMM(REAL, libxsmm_blasint, FLAGS, mm, nn, LIBXSMM_MIN(tile_k, (K) - j), \
                ALPHA, (A) + j * (LDA) + h, LDA, (B) + i * (LDB) + j, LDB, BETA, (C) + ic, LDC); \
            } \
          } \
        } \
      } \
      LIBXSMM_GEMM_OMPS_TASK_SYNC \
    } \
  } \
  else { /* fallback */ \
    LIBXSMM_BLAS_XGEMM(REAL, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC); \
  } \
}


LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE void LIBXSMM_FSYMBOL(sgemm)(
  const char*, const char*, const libxsmm_blasint*, const libxsmm_blasint*, const libxsmm_blasint*,
  const float*, const float*, const libxsmm_blasint*, const float*, const libxsmm_blasint*,
  const float*, float*, const libxsmm_blasint*);
LIBXSMM_RETARGETABLE libxsmm_sgemm_function libxsmm_internal_sgemm = LIBXSMM_FSYMBOL(sgemm);

LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE void LIBXSMM_FSYMBOL(dgemm)(
  const char*, const char*, const libxsmm_blasint*, const libxsmm_blasint*, const libxsmm_blasint*,
  const double*, const double*, const libxsmm_blasint*, const double*, const libxsmm_blasint*,
  const double*, double*, const libxsmm_blasint*);
LIBXSMM_RETARGETABLE libxsmm_dgemm_function libxsmm_internal_dgemm = LIBXSMM_FSYMBOL(dgemm);


LIBXSMM_RETARGETABLE LIBXSMM_VISIBILITY_INTERNAL int internal_gemm_tile_sizes[/*configs*/][2/*DP/SP*/][3/*TILE_M,TILE_N,TILE_K*/] = {
  { { 256, 128, 32 }, { 64, 256, 256 } }, /*generic*/
  { { 128,  48, 48 }, { 64,  48,  80 } }  /*knl*/
};
LIBXSMM_RETARGETABLE LIBXSMM_VISIBILITY_INTERNAL int internal_gemm_tile_size[/*DP/SP*/][3/*TILE_M,TILE_N,TILE_K*/] = {
  { 0, 0, 0 }, { 0, 0, 0 }
};
LIBXSMM_RETARGETABLE int libxsmm_internal_gemm = 0;


LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE void libxsmm_gemm_configure(const char* archid, int gemm_kind)
{
  const int config = (0 == archid || 'k' != archid[0] || 'n' != archid[1] || 'l' != archid[2]) ? 0 : 1;
  const char* env[3], *const env_gemm_kind = getenv("LIBXSMM_GEMM");

  /* determine what will be executed in the wrapper code (0: small gemm, 1: sequential, 2: parallelized) */
  libxsmm_internal_gemm = (env_gemm_kind ? atoi(env_gemm_kind) : gemm_kind);

  /* attempt to setup tile sizes from the environment (LIBXSMM_TILEM, LIBXSMM_TILEN, and LIBXSMM_TILEK) */
  env[0] = getenv("LIBXSMM_TILEM"); env[1] = getenv("LIBXSMM_TILEN"); env[2] = getenv("LIBXSMM_TILEK");
  internal_gemm_tile_size[0/*DP*/][0/*M*/] = (env[0] ? atoi(env[0]) : 0);
  internal_gemm_tile_size[0/*DP*/][1/*N*/] = (env[1] ? atoi(env[1]) : 0);
  internal_gemm_tile_size[0/*DP*/][2/*K*/] = (env[2] ? atoi(env[2]) : 0);
  /* environment-defined tile sizes applies for DP and SP */
  internal_gemm_tile_size[1/*SP*/][0/*M*/] = internal_gemm_tile_size[0/*DP*/][0];
  internal_gemm_tile_size[1/*SP*/][1/*N*/] = internal_gemm_tile_size[0/*DP*/][1];
  internal_gemm_tile_size[1/*SP*/][2/*K*/] = internal_gemm_tile_size[0/*DP*/][2];

  /* load predefined configuration if tile size is not setup by the environment */
  if (0 >= internal_gemm_tile_size[0/*DP*/][0/*M*/]) internal_gemm_tile_size[0][0] = internal_gemm_tile_sizes[config][0][0];
  if (0 >= internal_gemm_tile_size[0/*DP*/][1/*N*/]) internal_gemm_tile_size[0][1] = internal_gemm_tile_sizes[config][0][1];
  if (0 >= internal_gemm_tile_size[0/*DP*/][2/*K*/]) internal_gemm_tile_size[0][2] = internal_gemm_tile_sizes[config][0][2];
  if (0 >= internal_gemm_tile_size[1/*SP*/][0/*M*/]) internal_gemm_tile_size[1][0] = internal_gemm_tile_sizes[config][1][0];
  if (0 >= internal_gemm_tile_size[1/*SP*/][1/*N*/]) internal_gemm_tile_size[1][1] = internal_gemm_tile_sizes[config][1][1];
  if (0 >= internal_gemm_tile_size[1/*SP*/][2/*K*/]) internal_gemm_tile_size[1][2] = internal_gemm_tile_sizes[config][1][2];
}


LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE LIBXSMM_GEMM_WEAK int libxsmm_gemm_init(const char* archid,
  libxsmm_sgemm_function sgemm_function, libxsmm_dgemm_function dgemm_function)
{
  /* internal pre-initialization step */
  libxsmm_gemm_configure(archid, 0/*default gemm kind is small gemm*/);

  if (NULL != sgemm_function) {
    libxsmm_internal_sgemm = sgemm_function;
  }
#if defined(LIBXSMM_GEMM_WRAP) && defined(__STATIC)
  else if (NULL != LIBXSMM_FSYMBOL(__real_sgemm)) {
    libxsmm_internal_sgemm = LIBXSMM_FSYMBOL(__real_sgemm);
  }
  else if (NULL != LIBXSMM_FSYMBOL(__real_mkl_sgemm)) {
    libxsmm_internal_sgemm = LIBXSMM_FSYMBOL(__real_mkl_sgemm);
  }
#endif /*defined(LIBXSMM_GEMM_WRAP)*/

  if (NULL != dgemm_function) {
    libxsmm_internal_dgemm = dgemm_function;
  }
#if defined(LIBXSMM_GEMM_WRAP) && defined(__STATIC)
  else if (NULL != LIBXSMM_FSYMBOL(__real_dgemm)) {
    libxsmm_internal_dgemm = LIBXSMM_FSYMBOL(__real_dgemm);
  }
  else if (NULL != LIBXSMM_FSYMBOL(__real_mkl_dgemm)) {
    libxsmm_internal_dgemm = LIBXSMM_FSYMBOL(__real_mkl_dgemm);
  }
#endif /*defined(LIBXSMM_GEMM_WRAP)*/

  return (NULL != libxsmm_internal_sgemm
       && NULL != libxsmm_internal_dgemm)
    ? EXIT_SUCCESS
    : EXIT_FAILURE;
}


LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE LIBXSMM_GEMM_WEAK int libxsmm_gemm_finalize(void)
{
  return EXIT_SUCCESS;
}


LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE void libxsmm_sgemm(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const float* alpha, const float* a, const libxsmm_blasint* lda,
  const float* b, const libxsmm_blasint* ldb,
  const float* beta, float* c, const libxsmm_blasint* ldc)
{
  LIBXSMM_GEMM_DECLARE_FLAGS(flags, transa, transb, m, n, k, a, b, c);
  LIBXSMM_SGEMM(flags, *m, *n, *k,
    0 != alpha ? *alpha : ((float)LIBXSMM_ALPHA),
    a, *(lda ? lda : LIBXSMM_LD(m, k)), b, *(ldb ? ldb : LIBXSMM_LD(k, n)),
    0 != beta ? *beta : ((float)LIBXSMM_BETA),
    c, *(ldc ? ldc : LIBXSMM_LD(m, n)));
}


LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE void libxsmm_dgemm(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const double* alpha, const double* a, const libxsmm_blasint* lda,
  const double* b, const libxsmm_blasint* ldb,
  const double* beta, double* c, const libxsmm_blasint* ldc)
{
  LIBXSMM_GEMM_DECLARE_FLAGS(flags, transa, transb, m, n, k, a, b, c);
  LIBXSMM_DGEMM(flags, *m, *n, *k,
    0 != alpha ? *alpha : ((double)LIBXSMM_ALPHA),
    a, *(lda ? lda : LIBXSMM_LD(m, k)), b, *(ldb ? ldb : LIBXSMM_LD(k, n)),
    0 != beta ? *beta : ((double)LIBXSMM_BETA),
    c, *(ldc ? ldc : LIBXSMM_LD(m, n)));
}


LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE void LIBXSMM_FSYMBOL(libxsmm_sgemm)(const char*, const char*,
  const libxsmm_blasint*, const libxsmm_blasint*, const libxsmm_blasint*,
  const float*, const float*, const libxsmm_blasint*,
  const float*, const libxsmm_blasint*,
  const float*, float*, const libxsmm_blasint*);
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE void LIBXSMM_FSYMBOL(libxsmm_sgemm)(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const float* alpha, const float* a, const libxsmm_blasint* lda,
  const float* b, const libxsmm_blasint* ldb,
  const float* beta, float* c, const libxsmm_blasint* ldc)
{
  libxsmm_sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}


LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE void LIBXSMM_FSYMBOL(libxsmm_dgemm)(const char*, const char*,
  const libxsmm_blasint*, const libxsmm_blasint*, const libxsmm_blasint*,
  const double*, const double*, const libxsmm_blasint*,
  const double*, const libxsmm_blasint*,
  const double*, double*, const libxsmm_blasint*);
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE void LIBXSMM_FSYMBOL(libxsmm_dgemm)(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const double* alpha, const double* a, const libxsmm_blasint* lda,
  const double* b, const libxsmm_blasint* ldb,
  const double* beta, double* c, const libxsmm_blasint* ldc)
{
  libxsmm_dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}


LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE void libxsmm_omps_sgemm(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const float* alpha, const float* a, const libxsmm_blasint* lda,
  const float* b, const libxsmm_blasint* ldb,
  const float* beta, float* c, const libxsmm_blasint* ldc)
{
  LIBXSMM_GEMM_DECLARE_FLAGS(flags, transa, transb, m, n, k, a, b, c);
  LIBXSMM_GEMM_OMPS_XGEMM(float, flags | LIBXSMM_GEMM_FLAG_F32PREC,
    internal_gemm_tile_size[1/*SP*/][0/*M*/],
    internal_gemm_tile_size[1/*SP*/][1/*N*/],
    internal_gemm_tile_size[1/*SP*/][2/*K*/], *m, *n, *k,
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
  LIBXSMM_GEMM_OMPS_XGEMM(double, flags,
    internal_gemm_tile_size[0/*DP*/][0/*M*/],
    internal_gemm_tile_size[0/*DP*/][1/*N*/],
    internal_gemm_tile_size[0/*DP*/][2/*K*/], *m, *n, *k,
    0 != alpha ? *alpha : ((double)LIBXSMM_ALPHA),
    a, *(lda ? lda : LIBXSMM_LD(m, k)), b, *(ldb ? ldb : LIBXSMM_LD(k, n)),
    0 != beta ? *beta : ((double)LIBXSMM_BETA),
    c, *(ldc ? ldc : LIBXSMM_LD(m, n)));
}


LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE void libxsmm_blas_sgemm(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const float* alpha, const float* a, const libxsmm_blasint* lda,
  const float* b, const libxsmm_blasint* ldb,
  const float* beta, float* c, const libxsmm_blasint* ldc)
{
  LIBXSMM_GEMM_DECLARE_FLAGS(flags, transa, transb, m, n, k, a, b, c);
  LIBXSMM_BLAS_SGEMM(flags, *m, *n, *k,
    0 != alpha ? *alpha : ((float)LIBXSMM_ALPHA),
    a, *(lda ? lda : LIBXSMM_LD(m, k)), b, *(ldb ? ldb : LIBXSMM_LD(k, n)),
    0 != beta ? *beta : ((float)LIBXSMM_BETA),
    c, *(ldc ? ldc : LIBXSMM_LD(m, n)));
}


LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE void libxsmm_blas_dgemm(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const double* alpha, const double* a, const libxsmm_blasint* lda,
  const double* b, const libxsmm_blasint* ldb,
  const double* beta, double* c, const libxsmm_blasint* ldc)
{
  LIBXSMM_GEMM_DECLARE_FLAGS(flags, transa, transb, m, n, k, a, b, c);
  LIBXSMM_BLAS_DGEMM(flags, *m, *n, *k,
    0 != alpha ? *alpha : ((double)LIBXSMM_ALPHA),
    a, *(lda ? lda : LIBXSMM_LD(m, k)), b, *(ldb ? ldb : LIBXSMM_LD(k, n)),
    0 != beta ? *beta : ((double)LIBXSMM_BETA),
    c, *(ldc ? ldc : LIBXSMM_LD(m, n)));
}

