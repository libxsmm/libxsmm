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
#include "libxsmm_intrinsics_x86.h"
#include "libxsmm_main.h"
#include <libxsmm_sync.h>

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#include <stdlib.h>
#if defined(LIBXSMM_RTLD_NEXT)
# include <dlfcn.h>
#endif
#if !defined(NDEBUG)
# include <stdio.h>
#endif
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif


LIBXSMM_API_DEFINITION libxsmm_sgemm_function libxsmm_original_sgemm(void)
{
  static LIBXSMM_TLS libxsmm_sgemm_function original = 0;
#if defined(LIBXSMM_RTLD_NEXT) && (!defined(__BLAS) || (0 != __BLAS))

  if (0 == original) {
    union { const void* pv; libxsmm_sgemm_function pf; } gemm = { NULL };
    dlerror(); /* clear an eventual error status */
    gemm.pv = dlsym(RTLD_NEXT, LIBXSMM_STRINGIFY(LIBXSMM_FSYMBOL(sgemm)));
    original = gemm.pf;
    LIBXSMM_GEMM_WRAP(float, &original);
  }
#else
  LIBXSMM_GEMM_WRAP(float, &original);
#endif
#if !defined(NDEBUG) /* library code is expected to be mute */
  if (0 == original) {
    static LIBXSMM_TLS int error_sgemm = 0;
    if (0 == error_sgemm) {
      fprintf(stderr, "LIBXSMM: application must be linked against a LAPACK/BLAS implementation!\n");
      error_sgemm = 1;
    }
  }
#endif
  return original;
}


LIBXSMM_API_DEFINITION libxsmm_dgemm_function libxsmm_original_dgemm(void)
{
  static LIBXSMM_TLS libxsmm_dgemm_function original = 0;
#if defined(LIBXSMM_RTLD_NEXT) && (!defined(__BLAS) || (0 != __BLAS))
  if (0 == original) {
    union { const void* pv; libxsmm_dgemm_function pf; } gemm = { NULL };
    dlerror(); /* clear an eventual error status */
    gemm.pv = dlsym(RTLD_NEXT, LIBXSMM_STRINGIFY(LIBXSMM_FSYMBOL(dgemm)));
    original = gemm.pf;
    LIBXSMM_GEMM_WRAP(double, &original);
  }
#else
  LIBXSMM_GEMM_WRAP(double, &original);
#endif
#if !defined(NDEBUG) /* library code is expected to be mute */
  if (0 == original) {
    static LIBXSMM_TLS int error_dgemm = 0;
    if (0 == error_dgemm) {
      fprintf(stderr, "LIBXSMM: application must be linked against a LAPACK/BLAS implementation!\n");
      error_dgemm = 1;
    }
  }
#endif
  return original;
}


LIBXSMM_API_DEFINITION void libxsmm_gemm_init(int archid, int prefetch)
{
  int config = 0;
  libxsmm_gemm_prefetch = LIBXSMM_PREFETCH_AL2_AHEAD;
  LIBXSMM_UNUSED(prefetch);
#if defined(__MIC__) || (LIBXSMM_X86_AVX512_MIC == LIBXSMM_STATIC_TARGET_ARCH)
  LIBXSMM_UNUSED(archid);
#else
  if (LIBXSMM_X86_AVX512_MIC == archid)
#endif
  {
    config = 1;
  }
  { /* attempt to setup tile sizes from the environment (LIBXSMM_M, LIBXSMM_N, and LIBXSMM_K) */
    const LIBXSMM_GEMM_DESCRIPTOR_DIM_TYPE tile_configs[/*configs*/][2/*DP/SP*/][3/*TILE_M,TILE_N,TILE_K*/] = {
      { { 72, 32, 16 }, { 72, 32, 16 } }, /*generic*/
      { { 72, 32, 16 }, { 72, 32, 16 } }  /*knl*/
    };
    const char* env[3];
    env[0] = getenv("LIBXSMM_M"); env[1] = getenv("LIBXSMM_N"); env[2] = getenv("LIBXSMM_K");
    /* environment-defined tile sizes apply for DP and SP */
    libxsmm_gemm_tile[0/*DP*/][0/*M*/] = libxsmm_gemm_tile[1/*SP*/][0/*M*/] = (env[0] ? atoi(env[0]) : 0);
    libxsmm_gemm_tile[0/*DP*/][1/*N*/] = libxsmm_gemm_tile[1/*SP*/][1/*N*/] = (env[1] ? atoi(env[1]) : 0);
    libxsmm_gemm_tile[0/*DP*/][2/*K*/] = libxsmm_gemm_tile[1/*SP*/][2/*K*/] = (env[2] ? atoi(env[2]) : 0);
    /* load predefined configuration if tile size is not setup by the environment */
    if (0 == libxsmm_gemm_tile[0/*DP*/][0/*M*/]) libxsmm_gemm_tile[0][0] = tile_configs[config][0][0];
    if (0 == libxsmm_gemm_tile[0/*DP*/][1/*N*/]) libxsmm_gemm_tile[0][1] = tile_configs[config][0][1];
    if (0 == libxsmm_gemm_tile[0/*DP*/][2/*K*/]) libxsmm_gemm_tile[0][2] = tile_configs[config][0][2];
    if (0 == libxsmm_gemm_tile[1/*SP*/][0/*M*/]) libxsmm_gemm_tile[1][0] = tile_configs[config][1][0];
    if (0 == libxsmm_gemm_tile[1/*SP*/][1/*N*/]) libxsmm_gemm_tile[1][1] = tile_configs[config][1][1];
    if (0 == libxsmm_gemm_tile[1/*SP*/][2/*K*/]) libxsmm_gemm_tile[1][2] = tile_configs[config][1][2];
  }
}


LIBXSMM_API_DEFINITION void libxsmm_gemm_finalize(void)
{
}


#if defined(LIBXSMM_BUILD)

LIBXSMM_API_DEFINITION void libxsmm_sgemm(const char* transa, const char* transb,
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


LIBXSMM_API_DEFINITION void libxsmm_dgemm(const char* transa, const char* transb,
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

#endif /*defined(LIBXSMM_BUILD)*/


/* implementation provided for Fortran 77 compatibility */
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_sgemm)(const char*, const char*,
  const libxsmm_blasint*, const libxsmm_blasint*, const libxsmm_blasint*,
  const float*, const float*, const libxsmm_blasint*,
  const float*, const libxsmm_blasint*,
  const float*, float*, const libxsmm_blasint*);
LIBXSMM_API_DEFINITION void LIBXSMM_FSYMBOL(libxsmm_sgemm)(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const float* alpha, const float* a, const libxsmm_blasint* lda,
  const float* b, const libxsmm_blasint* ldb,
  const float* beta, float* c, const libxsmm_blasint* ldc)
{
  libxsmm_sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}


/* implementation provided for Fortran 77 compatibility */
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_dgemm)(const char*, const char*,
  const libxsmm_blasint*, const libxsmm_blasint*, const libxsmm_blasint*,
  const double*, const double*, const libxsmm_blasint*,
  const double*, const libxsmm_blasint*,
  const double*, double*, const libxsmm_blasint*);
LIBXSMM_API_DEFINITION void LIBXSMM_FSYMBOL(libxsmm_dgemm)(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const double* alpha, const double* a, const libxsmm_blasint* lda,
  const double* b, const libxsmm_blasint* ldb,
  const double* beta, double* c, const libxsmm_blasint* ldc)
{
  libxsmm_dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}


LIBXSMM_API_DEFINITION void libxsmm_blas_sgemm(const char* transa, const char* transb,
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


LIBXSMM_API_DEFINITION void libxsmm_blas_dgemm(const char* transa, const char* transb,
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


LIBXSMM_API_DEFINITION void LIBXSMM_FSYMBOL(__wrap_sgemm)(
  const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const float* alpha, const float* a, const libxsmm_blasint* lda,
  const float* b, const libxsmm_blasint* ldb,
  const float* beta, float* c, const libxsmm_blasint* ldc)
{
  const LIBXSMM_GEMM_DESCRIPTOR_DIM_TYPE tm = libxsmm_gemm_tile[1/*SP*/][0/*M*/];
  const LIBXSMM_GEMM_DESCRIPTOR_DIM_TYPE tn = libxsmm_gemm_tile[1/*SP*/][1/*N*/];
  const LIBXSMM_GEMM_DESCRIPTOR_DIM_TYPE tk = libxsmm_gemm_tile[1/*SP*/][2/*K*/];
  LIBXSMM_GEMM_DECLARE_FLAGS(flags, transa, transb, m, n, k, a, b, c);
  LIBXSMM_TILED_XGEMM(LIBXSMM_NOOP, LIBXSMM_NOOP, LIBXSMM_NOOP,
    LIBXSMM_GEMM_COLLAPSE, LIBXSMM_NOOP_ARGS, LIBXSMM_NOOP_ARGS, LIBXSMM_NOOP,
    LIBXSMM_MIN_NTASKS, LIBXSMM_OVERHEAD, libxsmm_nt,
    float, flags | LIBXSMM_GEMM_FLAG_F32PREC, tm, tn, tk, *m, *n, *k,
    0 != alpha ? *alpha : ((float)LIBXSMM_ALPHA),
    a, *(lda ? lda : LIBXSMM_LD(m, k)), b, *(ldb ? ldb : LIBXSMM_LD(k, n)),
    0 != beta ? *beta : ((float)LIBXSMM_BETA),
    c, *(ldc ? ldc : LIBXSMM_LD(m, n)));
}


LIBXSMM_API_DEFINITION void LIBXSMM_FSYMBOL(__wrap_dgemm)(
  const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const double* alpha, const double* a, const libxsmm_blasint* lda,
  const double* b, const libxsmm_blasint* ldb,
  const double* beta, double* c, const libxsmm_blasint* ldc)
{
  const LIBXSMM_GEMM_DESCRIPTOR_DIM_TYPE tm = libxsmm_gemm_tile[0/*DP*/][0/*M*/];
  const LIBXSMM_GEMM_DESCRIPTOR_DIM_TYPE tn = libxsmm_gemm_tile[0/*DP*/][1/*N*/];
  const LIBXSMM_GEMM_DESCRIPTOR_DIM_TYPE tk = libxsmm_gemm_tile[0/*DP*/][2/*K*/];
  LIBXSMM_GEMM_DECLARE_FLAGS(flags, transa, transb, m, n, k, a, b, c);
  LIBXSMM_TILED_XGEMM(LIBXSMM_NOOP, LIBXSMM_NOOP, LIBXSMM_NOOP,
    LIBXSMM_GEMM_COLLAPSE, LIBXSMM_NOOP_ARGS, LIBXSMM_NOOP_ARGS, LIBXSMM_NOOP,
    LIBXSMM_MIN_NTASKS, LIBXSMM_OVERHEAD, libxsmm_nt,
    double, flags, tm, tn, tk, *m, *n, *k,
    0 != alpha ? *alpha : ((double)LIBXSMM_ALPHA),
    a, *(lda ? lda : LIBXSMM_LD(m, k)), b, *(ldb ? ldb : LIBXSMM_LD(k, n)),
    0 != beta ? *beta : ((double)LIBXSMM_BETA),
    c, *(ldc ? ldc : LIBXSMM_LD(m, n)));
}

