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

#if defined(_OPENMP)
# if !defined(LIBXSMM_GEMM_OMPS_TASKS) && (200805 <= _OPENMP) /*OpenMP 3.0*/
#   define LIBXSMM_GEMM_OMPS_TASKS
# endif
# if defined(LIBXSMM_GEMM_OMPS_TASKS)
#   define LIBXSMM_GEMM_OMPS_TASK_START LIBXSMM_PRAGMA(omp single nowait)
#   define LIBXSMM_GEMM_OMPS_TASK_SYNC LIBXSMM_PRAGMA(omp taskwait)
#   define LIBXSMM_GEMM_OMPS_TASK(...) LIBXSMM_PRAGMA(omp task firstprivate(__VA_ARGS__))
#   define LIBXSMM_GEMM_OMPS_FOR(N)
# else
#   define LIBXSMM_GEMM_OMPS_TASK_START
#   define LIBXSMM_GEMM_OMPS_TASK_SYNC
#   define LIBXSMM_GEMM_OMPS_TASK(...)
#   define LIBXSMM_GEMM_OMPS_FOR(N) /*LIBXSMM_PRAGMA(omp for LIBXSMM_OPENMP_COLLAPSE(N) schedule(dynamic))*/
# endif
#else
# define LIBXSMM_GEMM_OMPS_TASK_START
# define LIBXSMM_GEMM_OMPS_TASK_SYNC
# define LIBXSMM_GEMM_OMPS_TASK(...)
# define LIBXSMM_GEMM_OMPS_FOR(N)
#endif

#define LIBXSMM_GEMM_OMPS_XGEMM(REAL, SYMBOL, ARGS, FLAGS, TILE_M, TILE_N, TILE_K, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC) { \
  LIBXSMM_GEMM_DESCRIPTOR((ARGS).desc, LIBXSMM_ALIGNMENT, FLAGS, \
    LIBXSMM_MIN(TILE_M, M), LIBXSMM_MIN(TILE_N, N), LIBXSMM_MIN(TILE_K, K), \
    LDA, LDB, LDC, ALPHA, BETA, LIBXSMM_PREFETCH); \
  (ARGS).alpha.LIBXSMM_TPREFIX_NAME(REAL) = ALPHA; \
  (ARGS).beta.LIBXSMM_TPREFIX_NAME(REAL) = BETA; \
  LIBXSMM_GEMM_OMPS_TASK_START \
  { \
    libxsmm_blasint h, i; \
    libxsmm_xmmfunction xmm = libxsmm_xmmdispatch(&((ARGS).desc)); \
    if (0 == xmm.dmm) { xmm.LIBXSMM_TPREFIX(REAL,mm) = SYMBOL; /* fallback */ } \
    LIBXSMM_GEMM_OMPS_FOR(2) \
    for (i = 0; i < (N); i += TILE_N) { \
      for (h = 0; h < (M); h += TILE_M) { \
        LIBXSMM_GEMM_OMPS_TASK(h, i) \
        { \
          const libxsmm_blasint mm = LIBXSMM_MIN(TILE_M, (M) - h); \
          const libxsmm_blasint nn = LIBXSMM_MIN(TILE_N, (N) - i); \
          const libxsmm_blasint ic = i * (LDC) + h; \
          libxsmm_blasint j = 0; \
          if (((TILE_M) == mm) && ((TILE_N) == nn)) { \
            for (; j < (K) - LIBXSMM_MOD2(K, TILE_K); j += TILE_K) { \
              LIBXSMM_MMCALL(xmm.LIBXSMM_TPREFIX(REAL,mm), (A) + j * (LDA) + h, (B) + i * (LDB) + j, (C) + ic, M, N, K, LDA, LDB, LDC); \
            } \
          } \
          for (; j < (K); j += TILE_K) { \
            LIBXSMM_XGEMM(REAL, libxsmm_blasint, LIBXSMM_BLAS_GEMM_SYMBOL(REAL), (ARGS).desc.flags, mm, nn, LIBXSMM_MIN(TILE_K, (K) - j), \
              ALPHA, (A) + j * (LDA) + h, LDA, (B) + i * (LDB) + j, LDB, BETA, (C) + ic, LDC); \
          } \
        } \
      } \
    } \
    LIBXSMM_GEMM_OMPS_TASK_SYNC \
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
  { {  64,  32, 16 }, { 64,  32,  16 } }  /*knl*/
};
LIBXSMM_RETARGETABLE LIBXSMM_VISIBILITY_INTERNAL int internal_gemm_tile_size[/*DP/SP*/][3/*TILE_M,TILE_N,TILE_K*/] = {
  { 0, 0, 0 }, { 0, 0, 0 }
};


LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE LIBXSMM_GEMM_WEAK void libxsmm_gemm_init_archid(const char* archid)
{
  const char *const env_tile_size[] = { getenv("LIBXSMM_TILEM"), getenv("LIBXSMM_TILEN"), getenv("LIBXSMM_TILEK") };
  const int config = (0 == archid || 'k' != archid[0] || 'n' != archid[1] || 'l' != archid[2]) ? 0 : 1;

  /* attempt to setup tile sizes from the environment */
  internal_gemm_tile_size[0/*DP*/][0/*M*/] = (env_tile_size[0] && *env_tile_size[0]) ? atoi(env_tile_size[0]) : 0;
  internal_gemm_tile_size[0/*DP*/][1/*N*/] = (env_tile_size[1] && *env_tile_size[1]) ? atoi(env_tile_size[1]) : 0;
  internal_gemm_tile_size[0/*DP*/][2/*K*/] = (env_tile_size[2] && *env_tile_size[2]) ? atoi(env_tile_size[2]) : 0;
  /* environment-defined tile sizes applies for DP and SP */
  internal_gemm_tile_size[1/*SP*/][0/*M*/] = internal_gemm_tile_size[0/*DP*/][0/*M*/];
  internal_gemm_tile_size[1/*SP*/][1/*N*/] = internal_gemm_tile_size[0/*DP*/][1/*N*/];
  internal_gemm_tile_size[1/*SP*/][2/*K*/] = internal_gemm_tile_size[0/*DP*/][2/*M*/];

  /* load predefined configuration if tile size is not setup by the environment */
  if (0 >= internal_gemm_tile_size[0/*DP*/][0/*M*/]) internal_gemm_tile_size[0/*DP*/][0/*M*/] = internal_gemm_tile_sizes[config][0/*DP*/][0/*M*/];
  if (0 >= internal_gemm_tile_size[0/*DP*/][1/*N*/]) internal_gemm_tile_size[0/*DP*/][1/*M*/] = internal_gemm_tile_sizes[config][0/*DP*/][1/*N*/];
  if (0 >= internal_gemm_tile_size[0/*DP*/][2/*K*/]) internal_gemm_tile_size[0/*DP*/][2/*M*/] = internal_gemm_tile_sizes[config][0/*DP*/][2/*K*/];
  if (0 >= internal_gemm_tile_size[1/*SP*/][0/*M*/]) internal_gemm_tile_size[1/*SP*/][0/*M*/] = internal_gemm_tile_sizes[config][1/*SP*/][0/*M*/];
  if (0 >= internal_gemm_tile_size[1/*SP*/][1/*N*/]) internal_gemm_tile_size[1/*SP*/][1/*M*/] = internal_gemm_tile_sizes[config][1/*SP*/][1/*N*/];
  if (0 >= internal_gemm_tile_size[1/*SP*/][2/*K*/]) internal_gemm_tile_size[1/*SP*/][2/*M*/] = internal_gemm_tile_sizes[config][1/*SP*/][2/*K*/];
}


LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE LIBXSMM_GEMM_WEAK int libxsmm_gemm_init(const char* archid,
  libxsmm_sgemm_function sgemm_function, libxsmm_dgemm_function dgemm_function)
{
  libxsmm_gemm_init_archid(archid);

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
  assert(0 < internal_gemm_tile_size[1/*SP*/][0/*M*/]
      && 0 < internal_gemm_tile_size[1/*SP*/][1/*N*/]
      && 0 < internal_gemm_tile_size[1/*SP*/][2/*K*/]);
  LIBXSMM_GEMM_OMPS_XGEMM(float, internal_omps_sblas, internal_omps_args, flags | LIBXSMM_GEMM_FLAG_F32PREC,
    internal_gemm_tile_size[1/*SP*/][0/*M*/], internal_gemm_tile_size[1/*SP*/][1/*N*/], internal_gemm_tile_size[1/*SP*/][2/*K*/], *m, *n, *k,
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
  assert(0 < internal_gemm_tile_size[0/*DP*/][0/*M*/]
      && 0 < internal_gemm_tile_size[0/*DP*/][1/*N*/]
      && 0 < internal_gemm_tile_size[0/*DP*/][2/*K*/]);
  LIBXSMM_GEMM_OMPS_XGEMM(double, internal_omps_dblas, internal_omps_args, flags,
    internal_gemm_tile_size[0/*DP*/][0/*M*/], internal_gemm_tile_size[0/*DP*/][1/*N*/], internal_gemm_tile_size[0/*DP*/][2/*K*/], *m, *n, *k,
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

