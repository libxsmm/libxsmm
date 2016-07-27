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
#if defined(LIBXSMM_GEMM_EXTWRAP) && !defined(__STATIC)
# include <stdlib.h>
# include <dlfcn.h>
#endif
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

#define LIBXSMM_GEMM_EXTOMP_SUFFICIENT_SIZE(M, N, K) (((LIBXSMM_MAX_M < (M)) || (LIBXSMM_MAX_N < (N)) || (LIBXSMM_MAX_K < (K))) ? 1 : 0)
#if defined(_OPENMP)
# if !defined(LIBXSMM_GEMM_EXTOMP_TASKS) && (200805 <= _OPENMP) /*OpenMP 3.0*/
#   define LIBXSMM_GEMM_EXTOMP_TASKS
# endif
# define LIBXSMM_GEMM_EXTOMP_MIN_NTASKS(NT) LIBXSMM_MAX(7/*arbitrary factor*/ * omp_get_num_threads() / (NT), 1)
# define LIBXSMM_GEMM_EXTOMP_OVERHEAD(NT) (/*arbitrary factor*/NT)
# define LIBXSMM_GEMM_EXTOMP_FOR_INIT
# define LIBXSMM_GEMM_EXTOMP_FOR_LOOP_BEGIN LIBXSMM_PRAGMA(omp for schedule(dynamic) LIBXSMM_OPENMP_COLLAPSE(LIBXSMM_GEMM_EXTOMP_COLLAPSE))
# define LIBXSMM_GEMM_EXTOMP_FOR_LOOP_BEGIN_PARALLEL LIBXSMM_PRAGMA(omp parallel for schedule(dynamic) LIBXSMM_OPENMP_COLLAPSE(LIBXSMM_GEMM_EXTOMP_COLLAPSE))
# define LIBXSMM_GEMM_EXTOMP_FOR_LOOP_BODY(...)
# define LIBXSMM_GEMM_EXTOMP_FOR_LOOP_END
#else
# define LIBXSMM_GEMM_EXTOMP_MIN_NTASKS(NT) 1
# define LIBXSMM_GEMM_EXTOMP_OVERHEAD(NT) 0
# define LIBXSMM_GEMM_EXTOMP_FOR_INIT
# define LIBXSMM_GEMM_EXTOMP_FOR_LOOP_BEGIN
# define LIBXSMM_GEMM_EXTOMP_FOR_LOOP_BEGIN_PARALLEL
# define LIBXSMM_GEMM_EXTOMP_FOR_LOOP_BODY(...)
# define LIBXSMM_GEMM_EXTOMP_FOR_LOOP_END
#endif

#if defined(LIBXSMM_GEMM_EXTOMP_TASKS)
# define LIBXSMM_GEMM_EXTOMP_COLLAPSE 2
# define LIBXSMM_GEMM_EXTOMP_TSK_INIT LIBXSMM_PRAGMA(omp single nowait)
# define LIBXSMM_GEMM_EXTOMP_TSK_LOOP_BEGIN
# define LIBXSMM_GEMM_EXTOMP_TSK_LOOP_BEGIN_PARALLEL LIBXSMM_PRAGMA(omp parallel) LIBXSMM_PRAGMA(omp single nowait)
# define LIBXSMM_GEMM_EXTOMP_TSK_LOOP_BODY(...) LIBXSMM_PRAGMA(omp task firstprivate(__VA_ARGS__))
# define LIBXSMM_GEMM_EXTOMP_TSK_LOOP_END LIBXSMM_PRAGMA(omp taskwait)
#else
# define LIBXSMM_GEMM_EXTOMP_COLLAPSE 2
# define LIBXSMM_GEMM_EXTOMP_TSK_INIT LIBXSMM_GEMM_EXTOMP_FOR_INIT
# define LIBXSMM_GEMM_EXTOMP_TSK_LOOP_BEGIN LIBXSMM_GEMM_EXTOMP_FOR_LOOP_BEGIN
# define LIBXSMM_GEMM_EXTOMP_TSK_LOOP_BEGIN_PARALLEL LIBXSMM_GEMM_EXTOMP_FOR_LOOP_BEGIN_PARALLEL
# define LIBXSMM_GEMM_EXTOMP_TSK_LOOP_BODY LIBXSMM_GEMM_EXTOMP_FOR_LOOP_BODY
# define LIBXSMM_GEMM_EXTOMP_TSK_LOOP_END LIBXSMM_GEMM_EXTOMP_FOR_LOOP_END
#endif

#define LIBXSMM_GEMM_EXTOMP_KERNEL(REAL, FLAGS, POS_H, POS_I, TILE_M, TILE_N, TILE_K, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) { \
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

#define LIBXSMM_GEMM_EXTOMP_XGEMM(INIT, LOOP_BEGIN, LOOP_BODY, LOOP_END, \
  REAL, FLAGS, NT, TILE_M, TILE_N, TILE_K, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) INIT \
{ \
  const signed char scalpha = (signed char)(ALPHA), scbeta = (signed char)(BETA); \
  const int sufficient_size = LIBXSMM_GEMM_EXTOMP_SUFFICIENT_SIZE(M, N, K); \
  libxsmm_blasint tile_m = 0, tile_n = 0, tile_k = 0, num_m = 0, num_n = 0, num_k = 0; \
  libxsmm_xmmfunction xmm = { 0 }; \
  if (0 != sufficient_size \
    /*TODO: not supported*/&& 0 == ((FLAGS) & (LIBXSMM_GEMM_FLAG_TRANS_A | LIBXSMM_GEMM_FLAG_TRANS_B)) \
    /*TODO: not supported*/&& 1 == scalpha && (1 == scbeta || 0 == scbeta)) \
  { \
    tile_m = LIBXSMM_MAX(TILE_M, 2); tile_n = LIBXSMM_MAX(TILE_N, 2); tile_k = LIBXSMM_MAX(TILE_K, 2); \
    num_m = ((M) + tile_m - 1) / tile_m; num_n = ((N) + tile_n - 1) / tile_n; num_k = ((K) + tile_k - 1) / tile_k; \
    { /* opening scope for additional variable declarations */ \
      const libxsmm_blasint num_t = (LIBXSMM_GEMM_EXTOMP_OVERHEAD(NT) <= num_k && 1 < LIBXSMM_GEMM_EXTOMP_COLLAPSE) \
        ? (num_m * num_n) : (num_n <= num_m ? num_m : num_n); \
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
        tile_m = LIBXSMM_MIN(LIBXSMM_MAX((libxsmm_blasint)(1 << LIBXSMM_LOG2(tile_m * rm /*+ 0.5*/)), 8), M); \
        tile_n = LIBXSMM_MIN(LIBXSMM_MAX((libxsmm_blasint)(1 << LIBXSMM_LOG2(tile_n * rn /*+ 0.5*/)), 8), N); \
        tile_k = LIBXSMM_MIN(LIBXSMM_MAX((libxsmm_blasint)(1 << LIBXSMM_LOG2(tile_k * rk /*+ 0.5*/)), 8), K); \
      } \
      LIBXSMM_GEMM_DESCRIPTOR(desc, LIBXSMM_ALIGNMENT, FLAGS, tile_m, tile_n, tile_k, \
        LDA, LDB, LDC, scalpha, scbeta, internal_gemm_prefetch); \
      xmm = libxsmm_xmmdispatch(&desc); \
    } \
  } \
  if (0 != xmm.dmm) { \
    const libxsmm_blasint max_j = ((K) / tile_k) * tile_k; \
    libxsmm_blasint h = 0, i = 0; \
    if ((LIBXSMM_GEMM_EXTOMP_OVERHEAD(NT)) <= num_k) { /* amortize overhead */ \
      LOOP_BEGIN \
      for (h = 0; h < (M); h += tile_m) { \
        for (i = 0; i < (N); i += tile_n) { \
          LOOP_BODY(h, i) \
          LIBXSMM_GEMM_EXTOMP_KERNEL(REAL, FLAGS, h, i, tile_m, tile_n, tile_k, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC); \
        } \
      } \
      LOOP_END \
    } \
    else if (num_n <= num_m) { \
      LOOP_BEGIN \
      for (h = 0; h < (M); h += tile_m) { \
        LOOP_BODY(h) \
        for (i = 0; i < (N); i += tile_n) { \
          LIBXSMM_GEMM_EXTOMP_KERNEL(REAL, FLAGS, h, i, tile_m, tile_n, tile_k, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC); \
        } \
      } \
      LOOP_END \
    } \
    else { \
      LOOP_BEGIN \
      for (i = 0; i < (N); i += tile_n) { \
        LOOP_BODY(i) \
        for (h = 0; h < (M); h += tile_m) { \
          LIBXSMM_GEMM_EXTOMP_KERNEL(REAL, FLAGS, h, i, tile_m, tile_n, tile_k, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC); \
        } \
      } \
      LOOP_END \
    } \
  } \
  else if (0 != sufficient_size) { /* BLAS fallback */ \
    LIBXSMM_BLAS_XGEMM(REAL, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC); \
  } \
  else { /* small problem size */ \
    LIBXSMM_XGEMM(REAL, libxsmm_blasint, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC); \
  } \
}


#if defined(LIBXSMM_GEMM_EXTWRAP) && !defined(__STATIC)

/* implementation variant for non-static linkage; overrides weak libxsmm_gemm_init in libxsmm_gemm.c */
LIBXSMM_API_DEFINITION int libxsmm_gemm_init(int archid, int prefetch)
{
  union { const void* pv; libxsmm_sgemm_function pf; } fn_sgemm = { NULL };
  union { const void* pv; libxsmm_dgemm_function pf; } fn_dgemm = { NULL };
  fn_sgemm.pv = dlsym(RTLD_NEXT, LIBXSMM_STRINGIFY(LIBXSMM_FSYMBOL(sgemm)));
  fn_dgemm.pv = dlsym(RTLD_NEXT, LIBXSMM_STRINGIFY(LIBXSMM_FSYMBOL(dgemm)));

  /* internal pre-initialization step */
  libxsmm_gemm_configure(archid, prefetch, fn_sgemm.pf, fn_dgemm.pf);

  { /* behaviour of libxsmm_omp_?gemm routines or LD_PRELOAD ?GEMM routines
     * 0: sequential below-threshold routine (no OpenMP); may fall-back to BLAS,
     * 1: OpenMP-parallelized but without internal parallel region,
     * 2: OpenMP-parallelized with internal parallel region" )
     */
    const char *const env = getenv("LIBXSMM_GEMM");
    if (0 != env && 0 != *env) {
      internal_gemm = atoi(env);
    }
  }

#if defined(LIBXSMM_GEMM_EXTOMP_TASKS)
  { /* consider user input about using (OpenMP-)tasks; this code must be here
    * because maybe only this translation unit is compiled with OpenMP support
    */
    const char *const env_tasks = getenv("LIBXSMM_TASKS");
    if (0 != env_tasks && 0 != *env_tasks) {
      internal_gemm_tasks = atoi(env_tasks);
    }
  }
#endif
#if !defined(__BLAS) || (0 != __BLAS)
  return (NULL != *libxsmm_original_sgemm()
       && NULL != *libxsmm_original_dgemm())
    ? EXIT_SUCCESS
    : EXIT_FAILURE;
#else
  return EXIT_SUCCESS;
#endif
}


LIBXSMM_API_DEFINITION void libxsmm_gemm_finalize(void)
{
}

#endif /*defined(LIBXSMM_GEMM_EXTWRAP) && !defined(__STATIC)*/


LIBXSMM_API_DEFINITION void libxsmm_omp_sgemm(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const float* alpha, const float* a, const libxsmm_blasint* lda,
  const float* b, const libxsmm_blasint* ldb,
  const float* beta, float* c, const libxsmm_blasint* ldc)
{
  const int tm = internal_gemm_tile[1/*SP*/][0/*M*/];
  const int tn = internal_gemm_tile[1/*SP*/][1/*N*/];
  const int tk = internal_gemm_tile[1/*SP*/][2/*K*/];
  const int nt = internal_gemm_nt;
  LIBXSMM_GEMM_DECLARE_FLAGS(flags, transa, transb, m, n, k, a, b, c);
#if !defined(_OPENMP)
  LIBXSMM_UNUSED(nt);
#endif
  if (0 == LIBXSMM_DIV2(internal_gemm, 2)) { /* enable internal parallelization */
    if (0 == internal_gemm_tasks) {
      LIBXSMM_GEMM_EXTOMP_XGEMM(LIBXSMM_GEMM_EXTOMP_FOR_INIT, LIBXSMM_GEMM_EXTOMP_FOR_LOOP_BEGIN_PARALLEL,
        LIBXSMM_GEMM_EXTOMP_FOR_LOOP_BODY, LIBXSMM_GEMM_EXTOMP_FOR_LOOP_END,
        float, flags | LIBXSMM_GEMM_FLAG_F32PREC, nt, tm, tn, tk, *m, *n, *k,
        0 != alpha ? *alpha : ((float)LIBXSMM_ALPHA),
        a, *(lda ? lda : LIBXSMM_LD(m, k)), b, *(ldb ? ldb : LIBXSMM_LD(k, n)),
        0 != beta ? *beta : ((float)LIBXSMM_BETA),
        c, *(ldc ? ldc : LIBXSMM_LD(m, n)));
    }
    else {
      LIBXSMM_GEMM_EXTOMP_XGEMM(LIBXSMM_GEMM_EXTOMP_TSK_INIT, LIBXSMM_GEMM_EXTOMP_TSK_LOOP_BEGIN_PARALLEL,
        LIBXSMM_GEMM_EXTOMP_TSK_LOOP_BODY, LIBXSMM_GEMM_EXTOMP_TSK_LOOP_END,
        float, flags | LIBXSMM_GEMM_FLAG_F32PREC, nt, tm, tn, tk, *m, *n, *k,
        0 != alpha ? *alpha : ((float)LIBXSMM_ALPHA),
        a, *(lda ? lda : LIBXSMM_LD(m, k)), b, *(ldb ? ldb : LIBXSMM_LD(k, n)),
        0 != beta ? *beta : ((float)LIBXSMM_BETA),
        c, *(ldc ? ldc : LIBXSMM_LD(m, n)));
    }
  }
  else { /* default: potentially sequential or externally parallelized */
    if (0 == internal_gemm_tasks) {
      LIBXSMM_GEMM_EXTOMP_XGEMM(LIBXSMM_GEMM_EXTOMP_FOR_INIT, LIBXSMM_GEMM_EXTOMP_FOR_LOOP_BEGIN,
        LIBXSMM_GEMM_EXTOMP_FOR_LOOP_BODY, LIBXSMM_GEMM_EXTOMP_FOR_LOOP_END,
        float, flags | LIBXSMM_GEMM_FLAG_F32PREC, nt, tm, tn, tk, *m, *n, *k,
        0 != alpha ? *alpha : ((float)LIBXSMM_ALPHA),
        a, *(lda ? lda : LIBXSMM_LD(m, k)), b, *(ldb ? ldb : LIBXSMM_LD(k, n)),
        0 != beta ? *beta : ((float)LIBXSMM_BETA),
        c, *(ldc ? ldc : LIBXSMM_LD(m, n)));
    }
    else {
      LIBXSMM_GEMM_EXTOMP_XGEMM(LIBXSMM_GEMM_EXTOMP_TSK_INIT, LIBXSMM_GEMM_EXTOMP_TSK_LOOP_BEGIN,
        LIBXSMM_GEMM_EXTOMP_TSK_LOOP_BODY, LIBXSMM_GEMM_EXTOMP_TSK_LOOP_END,
        float, flags | LIBXSMM_GEMM_FLAG_F32PREC, nt, tm, tn, tk, *m, *n, *k,
        0 != alpha ? *alpha : ((float)LIBXSMM_ALPHA),
        a, *(lda ? lda : LIBXSMM_LD(m, k)), b, *(ldb ? ldb : LIBXSMM_LD(k, n)),
        0 != beta ? *beta : ((float)LIBXSMM_BETA),
        c, *(ldc ? ldc : LIBXSMM_LD(m, n)));
    }
  }
}


LIBXSMM_API_DEFINITION void libxsmm_omp_dgemm(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const double* alpha, const double* a, const libxsmm_blasint* lda,
  const double* b, const libxsmm_blasint* ldb,
  const double* beta, double* c, const libxsmm_blasint* ldc)
{
  const int tm = internal_gemm_tile[0/*DP*/][0/*M*/];
  const int tn = internal_gemm_tile[0/*DP*/][1/*N*/];
  const int tk = internal_gemm_tile[0/*DP*/][2/*K*/];
  const int nt = internal_gemm_nt;
  LIBXSMM_GEMM_DECLARE_FLAGS(flags, transa, transb, m, n, k, a, b, c);
#if !defined(_OPENMP)
  LIBXSMM_UNUSED(nt);
#endif
  if (0 == LIBXSMM_DIV2(internal_gemm, 2)) { /* enable internal parallelization */
    if (0 == internal_gemm_tasks) {
      LIBXSMM_GEMM_EXTOMP_XGEMM(LIBXSMM_GEMM_EXTOMP_FOR_INIT, LIBXSMM_GEMM_EXTOMP_FOR_LOOP_BEGIN_PARALLEL,
        LIBXSMM_GEMM_EXTOMP_FOR_LOOP_BODY, LIBXSMM_GEMM_EXTOMP_FOR_LOOP_END,
        double, flags, nt, tm, tn, tk, *m, *n, *k,
        0 != alpha ? *alpha : ((double)LIBXSMM_ALPHA),
        a, *(lda ? lda : LIBXSMM_LD(m, k)), b, *(ldb ? ldb : LIBXSMM_LD(k, n)),
        0 != beta ? *beta : ((double)LIBXSMM_BETA),
        c, *(ldc ? ldc : LIBXSMM_LD(m, n)));
    }
    else {
      LIBXSMM_GEMM_EXTOMP_XGEMM(LIBXSMM_GEMM_EXTOMP_TSK_INIT, LIBXSMM_GEMM_EXTOMP_TSK_LOOP_BEGIN_PARALLEL,
        LIBXSMM_GEMM_EXTOMP_TSK_LOOP_BODY, LIBXSMM_GEMM_EXTOMP_TSK_LOOP_END,
        double, flags, nt, tm, tn, tk, *m, *n, *k,
        0 != alpha ? *alpha : ((double)LIBXSMM_ALPHA),
        a, *(lda ? lda : LIBXSMM_LD(m, k)), b, *(ldb ? ldb : LIBXSMM_LD(k, n)),
        0 != beta ? *beta : ((double)LIBXSMM_BETA),
        c, *(ldc ? ldc : LIBXSMM_LD(m, n)));
    }
  }
  else { /* default: potentially sequential or externally parallelized */
    if (0 == internal_gemm_tasks) {
      LIBXSMM_GEMM_EXTOMP_XGEMM(LIBXSMM_GEMM_EXTOMP_FOR_INIT, LIBXSMM_GEMM_EXTOMP_FOR_LOOP_BEGIN,
        LIBXSMM_GEMM_EXTOMP_FOR_LOOP_BODY, LIBXSMM_GEMM_EXTOMP_FOR_LOOP_END,
        double, flags, nt, tm, tn, tk, *m, *n, *k,
        0 != alpha ? *alpha : ((double)LIBXSMM_ALPHA),
        a, *(lda ? lda : LIBXSMM_LD(m, k)), b, *(ldb ? ldb : LIBXSMM_LD(k, n)),
        0 != beta ? *beta : ((double)LIBXSMM_BETA),
        c, *(ldc ? ldc : LIBXSMM_LD(m, n)));
    }
    else {
      LIBXSMM_GEMM_EXTOMP_XGEMM(LIBXSMM_GEMM_EXTOMP_TSK_INIT, LIBXSMM_GEMM_EXTOMP_TSK_LOOP_BEGIN,
        LIBXSMM_GEMM_EXTOMP_TSK_LOOP_BODY, LIBXSMM_GEMM_EXTOMP_TSK_LOOP_END,
        double, flags, nt, tm, tn, tk, *m, *n, *k,
        0 != alpha ? *alpha : ((double)LIBXSMM_ALPHA),
        a, *(lda ? lda : LIBXSMM_LD(m, k)), b, *(ldb ? ldb : LIBXSMM_LD(k, n)),
        0 != beta ? *beta : ((double)LIBXSMM_BETA),
        c, *(ldc ? ldc : LIBXSMM_LD(m, n)));
    }
  }
}


#if defined(LIBXSMM_GEMM_EXTWRAP)

LIBXSMM_EXTERN LIBXSMM_RETARGETABLE void LIBXSMM_GEMM_EXTWRAP_SGEMM(
  const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const float* alpha, const float* a, const libxsmm_blasint* lda,
  const float* b, const libxsmm_blasint* ldb,
  const float* beta, float* c, const libxsmm_blasint* ldc)
{
  assert(LIBXSMM_GEMM_EXTWRAP_SGEMM != *libxsmm_original_sgemm());
  switch (internal_gemm) {
    case 0: { /* below-THRESHOLD xGEMM */
      LIBXSMM_GEMM_DECLARE_FLAGS(flags, transa, transb, m, n, k, a, b, c);
      LIBXSMM_XGEMM(float, libxsmm_blasint, flags, *m, *n, *k,
        0 != alpha ? *alpha : ((float)LIBXSMM_ALPHA),
        a, *(lda ? lda : LIBXSMM_LD(m, k)), b, *(ldb ? ldb : LIBXSMM_LD(k, n)),
        0 != beta ? *beta : ((float)LIBXSMM_BETA),
        c, *(ldc ? ldc : LIBXSMM_LD(m, n)));
    } break;
    default: { /* tiled xGEMM */
      libxsmm_omp_sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }
  }
}


LIBXSMM_EXTERN LIBXSMM_RETARGETABLE void LIBXSMM_GEMM_EXTWRAP_DGEMM(
  const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const double* alpha, const double* a, const libxsmm_blasint* lda,
  const double* b, const libxsmm_blasint* ldb,
  const double* beta, double* c, const libxsmm_blasint* ldc)
{
  assert(LIBXSMM_GEMM_EXTWRAP_DGEMM != *libxsmm_original_dgemm());
  switch (internal_gemm) {
    case 0: { /* below-THRESHOLD xGEMM */
      LIBXSMM_GEMM_DECLARE_FLAGS(flags, transa, transb, m, n, k, a, b, c);
      LIBXSMM_XGEMM(double, libxsmm_blasint, flags, *m, *n, *k,
        0 != alpha ? *alpha : ((double)LIBXSMM_ALPHA),
        a, *(lda ? lda : LIBXSMM_LD(m, k)), b, *(ldb ? ldb : LIBXSMM_LD(k, n)),
        0 != beta ? *beta : ((double)LIBXSMM_BETA),
        c, *(ldc ? ldc : LIBXSMM_LD(m, n)));
    } break;
    default: { /* tiled xGEMM */
      libxsmm_omp_dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }
  }
}

#endif /*defined(LIBXSMM_GEMM_EXTWRAP)*/

