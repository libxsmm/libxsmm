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
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

#define LIBXSMM_OMPS_TILE_M 32
#define LIBXSMM_OMPS_TILE_N 32
#define LIBXSMM_OMPS_TILE_K 32

#if defined(_OPENMP)
# if !defined(LIBXSMM_OMPS_TASKS) && (200805 <= _OPENMP) /*OpenMP 3.0*/
#   define LIBXSMM_OMPS_TASKS
# endif
# if defined(LIBXSMM_OMPS_TASKS)
#   define LIBXSMM_OMPS_BEGIN LIBXSMM_PRAGMA(omp single nowait)
#   define LIBXSMM_OMPS_TASK LIBXSMM_PRAGMA(omp task)
#   define LIBXSMM_OMPS_FOR(N)
#   define LIBXSMM_OMPS_END LIBXSMM_PRAGMA(omp taskwait)
# else
#   define LIBXSMM_OMPS_BEGIN
#   define LIBXSMM_OMPS_TASK
#   define LIBXSMM_OMPS_FOR(N) LIBXSMM_PRAGMA(omp for schedule(static) LIBXSMM_OPENMP_COLLAPSE(N))
#   define LIBXSMM_OMPS_END
# endif
#else
# define LIBXSMM_OMPS_BEGIN
# define LIBXSMM_OMPS_TASK
# define LIBXSMM_OMPS_FOR(N)
# define LIBXSMM_OMPS_END
#endif

#define LIBXSMM_OMPS_XGEMM(REAL, SYMBOL, ARGS, FLAGS, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) { \
  const libxsmm_blasint num_m = (*m) / (LIBXSMM_OMPS_TILE_M), num_n = (*n) / (LIBXSMM_OMPS_TILE_N), num_k = (*k) / (LIBXSMM_OMPS_TILE_K); \
  LIBXSMM_GEMM_DESCRIPTOR((ARGS).desc, LIBXSMM_ALIGNMENT, FLAGS, LIBXSMM_OMPS_TILE_M, LIBXSMM_OMPS_TILE_N, LIBXSMM_OMPS_TILE_K, LDA, LDB, LDC, ALPHA, BETA, LIBXSMM_PREFETCH); \
  (ARGS).alpha.LIBXSMM_TPREFIX_NAME(REAL) = ALPHA; \
  (ARGS).beta.LIBXSMM_TPREFIX_NAME(REAL) = BETA; \
  LIBXSMM_OMPS_BEGIN { \
    libxsmm_blasint h, i; \
    libxsmm_xmmfunction xmm = libxsmm_xmmdispatch(&((ARGS).desc)); \
    if (0 == xmm.dmm) { xmm.LIBXSMM_TPREFIX(REAL,mm) = SYMBOL; /* fallback */ } \
    LIBXSMM_OMPS_FOR(2) \
    for (i = 0; i < num_n; ++i) { \
      for (h = 0; h < num_m; ++h) { \
        const libxsmm_blasint ic = i * (*ldc) * (LIBXSMM_OMPS_TILE_N) + h * (LIBXSMM_OMPS_TILE_M); \
        LIBXSMM_OMPS_TASK \
        { \
          libxsmm_blasint j; \
          for (j = 0; j < num_k; ++j) { \
            const libxsmm_blasint ia = j * (*lda) * (LIBXSMM_OMPS_TILE_K) + h * (LIBXSMM_OMPS_TILE_M); \
            const libxsmm_blasint ib = i * (*ldb) * (LIBXSMM_OMPS_TILE_N) + j * (LIBXSMM_OMPS_TILE_K); \
            xmm.LIBXSMM_TPREFIX(REAL,mm)(a + ia, b + ib, c + ic); \
          } \
        } \
      } \
    } \
    if ((*n) > num_n * (LIBXSMM_OMPS_TILE_N)) { /* remainder tiles are processed using the auto-dispatched routine */ \
      const libxsmm_blasint nn = LIBXSMM_MIN(LIBXSMM_OMPS_TILE_N, (*n) - num_n * ((LIBXSMM_OMPS_TILE_N) * (LIBXSMM_OMPS_TILE_N))); \
      LIBXSMM_OMPS_FOR(1) \
      for (h = 0; h < num_m; ++h) { \
        const libxsmm_blasint ic = num_n * (*ldc) * ((LIBXSMM_OMPS_TILE_N) * (LIBXSMM_OMPS_TILE_N)) + h * (LIBXSMM_OMPS_TILE_M); \
        const libxsmm_blasint mm = LIBXSMM_MIN(LIBXSMM_OMPS_TILE_M, (*m) - h * (LIBXSMM_OMPS_TILE_M)); \
        LIBXSMM_OMPS_TASK \
        { \
          libxsmm_blasint j; \
          for (j = 0; j < num_k; ++j) { \
            const libxsmm_blasint jk = j * (LIBXSMM_OMPS_TILE_K), kk = LIBXSMM_MIN(LIBXSMM_OMPS_TILE_K, (*k) - jk); \
            LIBXSMM_XGEMM(REAL, libxsmm_blasint, LIBXSMM_BLAS_GEMM_SYMBOL(REAL), (ARGS).desc.flags, mm, nn, kk, *alpha, \
              a + jk * (*lda) + h * (LIBXSMM_OMPS_TILE_M), *lda, \
              b + jk + (*ldb) * num_n * ((LIBXSMM_OMPS_TILE_N) * (LIBXSMM_OMPS_TILE_N)), *ldb, *beta, \
              c + ic, *ldc); \
          } \
        } \
      } \
    } \
  } LIBXSMM_OMPS_END \
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


LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE LIBXSMM_GEMM_WEAK int libxsmm_gemm_init(
  libxsmm_sgemm_function sgemm_function, libxsmm_dgemm_function dgemm_function)
{
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


struct {
  libxsmm_gemm_descriptor desc;
  union { double d; float s; } alpha;
  union { double d; float s; } beta;
} internal_omps_args;
#if defined(_OPENMP)
# pragma omp threadprivate(internal_omps_args)
#endif


LIBXSMM_INLINE LIBXSMM_RETARGETABLE void internal_omps_sblas(const float* a, const float* b, float* c, ...)
{
  LIBXSMM_BLAS_SGEMM(internal_omps_args.desc.flags, internal_omps_args.desc.m, internal_omps_args.desc.n, internal_omps_args.desc.k,
    internal_omps_args.alpha.s, a, internal_omps_args.desc.lda, b, internal_omps_args.desc.ldb,
    internal_omps_args.beta.s, c, internal_omps_args.desc.ldc);
}


LIBXSMM_INLINE LIBXSMM_RETARGETABLE void internal_omps_dblas(const double* a, const double* b, double* c, ...)
{
  LIBXSMM_BLAS_DGEMM(internal_omps_args.desc.flags, internal_omps_args.desc.m, internal_omps_args.desc.n, internal_omps_args.desc.k,
    internal_omps_args.alpha.d, a, internal_omps_args.desc.lda, b, internal_omps_args.desc.ldb,
    internal_omps_args.beta.d, c, internal_omps_args.desc.ldc);
}


LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE void libxsmm_omps_sgemm(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const float* alpha, const float* a, const libxsmm_blasint* lda,
  const float* b, const libxsmm_blasint* ldb,
  const float* beta, float* c, const libxsmm_blasint* ldc)
{
  LIBXSMM_GEMM_DECLARE_FLAGS(flags, transa, transb, m, n, k, a, b, c);
  LIBXSMM_OMPS_XGEMM(float, internal_omps_sblas, internal_omps_args, flags | LIBXSMM_GEMM_FLAG_F32PREC, *m, *n, *k,
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
  LIBXSMM_OMPS_XGEMM(double, internal_omps_dblas, internal_omps_args, flags, *m, *n, *k,
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

