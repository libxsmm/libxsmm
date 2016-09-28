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
#include "libxsmm_dump.h"

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#include <stdlib.h>
#include <stdio.h>
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif


LIBXSMM_API_DEFINITION LIBXSMM_GEMM_WEAK libxsmm_sgemm_function libxsmm_original_sgemm(const void* caller)
{
  static LIBXSMM_TLS libxsmm_sgemm_function original = 0;
  LIBXSMM_GEMM_WRAPPER(float, original, caller);
  assert(0 != original);
  return original;
}


LIBXSMM_API_DEFINITION LIBXSMM_GEMM_WEAK libxsmm_dgemm_function libxsmm_original_dgemm(const void* caller)
{
  static LIBXSMM_TLS libxsmm_dgemm_function original = 0;
  LIBXSMM_GEMM_WRAPPER(double, original, caller);
  assert(0 != original);
  return original;
}


LIBXSMM_API_DEFINITION void libxsmm_gemm_init(int archid, int prefetch)
{
  int config = 0;
  const char *const prefetch_env = getenv("LIBXSMM_GEMM_PREFETCH");
  const int uid = (0 == prefetch_env || 0 == *prefetch_env) ? 6/*LIBXSMM_PREFETCH_AL2_AHEAD*/ : atoi(prefetch_env);
  libxsmm_gemm_prefetch = 0 <= uid ? libxsmm_uid2prefetch(uid) : prefetch;

#if defined(__MIC__) || (LIBXSMM_X86_AVX512_MIC == LIBXSMM_STATIC_TARGET_ARCH)
  LIBXSMM_UNUSED(archid);
#else
  if (LIBXSMM_X86_AVX512_MIC == archid)
#endif
  { config = 1; }
#if (LIBXSMM_X86_AVX512_MIC < LIBXSMM_STATIC_TARGET_ARCH)
  LIBXSMM_UNUSED(archid);
#else
  if (LIBXSMM_X86_AVX512_MIC < archid)
#endif
  { config = 2; }

  { /* attempt to setup tile sizes from the environment (LIBXSMM_M, LIBXSMM_N, and LIBXSMM_K) */
    const LIBXSMM_GEMM_DESCRIPTOR_DIM_TYPE tile_configs[/*configs*/][2/*DP/SP*/][3/*TILE_M,TILE_N,TILE_K*/] = {
      { {  96, 32, 16 }, {  96, 32, 16 } }, /*generic*/
      { {  96, 32, 16 }, {  96, 32, 16 } }, /*knl*/
      { { 160, 32, 16 }, { 160, 32, 16 } }  /*skx*/
    };
    const char* env[3];
    env[0] = getenv("LIBXSMM_M"); env[1] = getenv("LIBXSMM_N"); env[2] = getenv("LIBXSMM_K");
    /* environment-defined tile sizes apply for DP and SP */
    libxsmm_gemm_tile[0/*DP*/][0/*M*/] = libxsmm_gemm_tile[1/*SP*/][0/*M*/] =
      (LIBXSMM_GEMM_DESCRIPTOR_DIM_TYPE)LIBXSMM_MIN(0 != env[0] ? atoi(env[0]) : 0, LIBXSMM_GEMM_DESCRIPTOR_DIM_MAX);
    libxsmm_gemm_tile[0/*DP*/][1/*N*/] = libxsmm_gemm_tile[1/*SP*/][1/*N*/] =
      (LIBXSMM_GEMM_DESCRIPTOR_DIM_TYPE)LIBXSMM_MIN(0 != env[1] ? atoi(env[1]) : 0, LIBXSMM_GEMM_DESCRIPTOR_DIM_MAX);
    libxsmm_gemm_tile[0/*DP*/][2/*K*/] = libxsmm_gemm_tile[1/*SP*/][2/*K*/] =
      (LIBXSMM_GEMM_DESCRIPTOR_DIM_TYPE)LIBXSMM_MIN(0 != env[2] ? atoi(env[2]) : 0, LIBXSMM_GEMM_DESCRIPTOR_DIM_MAX);
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


LIBXSMM_API_DEFINITION void libxsmm_gemm_dump(libxsmm_gemm_xflags precision, const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const void* alpha, const void* a, const libxsmm_blasint* lda,
  const void* b, const libxsmm_blasint* ldb,
  const void* beta, void* c, const libxsmm_blasint* ldc)
{
  const char ctransa = (char)(0 != transa ? (*transa) : (0 == (LIBXSMM_FLAGS & LIBXSMM_GEMM_FLAG_TRANS_A) ? 'N' : 'T'));
  const char ctransb = (char)(0 != transb ? (*transb) : (0 == (LIBXSMM_FLAGS & LIBXSMM_GEMM_FLAG_TRANS_B) ? 'N' : 'T'));
  char filename[256], extension_header[512];
  size_t data_size[2], size[2];

  if (0 == (LIBXSMM_GEMM_FLAG_F32PREC & precision)) {
    LIBXSMM_SNPRINTF(extension_header, sizeof(extension_header), "TRANS = %c\nALPHA = %f", ctransa, 0 != alpha ? *((const double*)alpha) : LIBXSMM_ALPHA);
  }
  else {
    LIBXSMM_SNPRINTF(extension_header, sizeof(extension_header), "TRANS = %c\nALPHA = %f", ctransa, 0 != alpha ? *((const float*)alpha) : LIBXSMM_ALPHA);
  }
  LIBXSMM_SNPRINTF(filename, sizeof(filename), "libxsmm_a_%p.mhd", a);
  data_size[0] = *(0 != lda ? lda : m); data_size[1] = *k; size[0] = *m; size[1] = *k;
  libxsmm_meta_image_write(filename, data_size, size, 2/*ndims*/, 1/*ncomponents*/, a,
    0 == (LIBXSMM_GEMM_FLAG_F32PREC & precision) ? LIBXSMM_MHD_ELEMTYPE_F64 : LIBXSMM_MHD_ELEMTYPE_F32,
    0/*spacing*/, extension_header, 0/*extension*/, 0/*extension_size*/);

  LIBXSMM_SNPRINTF(extension_header, sizeof(extension_header), "\nTRANS = %c", ctransb);
  LIBXSMM_SNPRINTF(filename, sizeof(filename), "libxsmm_b_%p.mhd", b);
  data_size[0] = *(0 != ldb ? ldb : n); data_size[1] = *n; size[0] = *k; size[1] = *n;
  libxsmm_meta_image_write(filename, data_size, size, 2/*ndims*/, 1/*ncomponents*/, b,
    0 == (LIBXSMM_GEMM_FLAG_F32PREC & precision) ? LIBXSMM_MHD_ELEMTYPE_F64 : LIBXSMM_MHD_ELEMTYPE_F32,
    0/*spacing*/, extension_header, 0/*extension*/, 0/*extension_size*/);

  if (0 == (LIBXSMM_GEMM_FLAG_F32PREC & precision)) {
    LIBXSMM_SNPRINTF(extension_header, sizeof(extension_header), "BETA = %f", 0 != beta ? *((const double*)beta): LIBXSMM_BETA);
  }
  else {
    LIBXSMM_SNPRINTF(extension_header, sizeof(extension_header), "\nBETA = %f", 0 != beta ? *((const float*)beta): LIBXSMM_BETA);
  }
  LIBXSMM_SNPRINTF(filename, sizeof(filename), "libxsmm_c_%p.mhd", c);
  data_size[0] = *(0 != ldc ? ldc : m); data_size[1] = *n; size[0] = *m; size[1] = *n;
  libxsmm_meta_image_write(filename, data_size, size, 2/*ndims*/, 1/*ncomponents*/, c,
    0 != (LIBXSMM_GEMM_FLAG_F64PREC & precision) ? LIBXSMM_MHD_ELEMTYPE_F64 : LIBXSMM_MHD_ELEMTYPE_F32,
    0/*spacing*/, extension_header, 0/*extension*/, 0/*extension_size*/);
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


#if defined(LIBXSMM_BUILD)

LIBXSMM_API_DEFINITION void libxsmm_sgemm(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const float* alpha, const float* a, const libxsmm_blasint* lda,
  const float* b, const libxsmm_blasint* ldb,
  const float* beta, float* c, const libxsmm_blasint* ldc)
{
  LIBXSMM_GEMM_DECLARE_FLAGS(flags, transa, transb, m, n, k, a, b, c);
#if defined(LIBXSMM_GEMM_TILED)
  if (0 == LIBXSMM_MOD2(libxsmm_mt, 2))
#endif
  { /* below-threshold GEMM */
    LIBXSMM_SGEMM(flags, *m, *n, *k,
      0 != alpha ? *alpha : ((float)LIBXSMM_ALPHA),
      a, *(lda ? lda : LIBXSMM_LD(m, k)), b, *(ldb ? ldb : LIBXSMM_LD(k, n)),
      0 != beta ? *beta : ((float)LIBXSMM_BETA),
      c, *(ldc ? ldc : LIBXSMM_LD(m, n)));
  }
#if defined(LIBXSMM_GEMM_TILED)
  else { /* tiled GEMM */
    LIBXSMM_GEMM_DESCRIPTOR_DIM_TYPE tm, tn, tk;
    LIBXSMM_INIT
    tm = libxsmm_gemm_tile[1/*SP*/][0/*M*/];
    tn = libxsmm_gemm_tile[1/*SP*/][1/*N*/];
    tk = libxsmm_gemm_tile[1/*SP*/][2/*K*/];
    assert(0 < tm && 0 < tn && 0 < tk && 0 < libxsmm_nt);
    LIBXSMM_TILED_XGEMM(LIBXSMM_NOOP, LIBXSMM_NOOP, LIBXSMM_NOOP,
      LIBXSMM_GEMM_COLLAPSE, LIBXSMM_NOOP_ARGS, LIBXSMM_NOOP_ARGS, LIBXSMM_NOOP,
      LIBXSMM_MIN_NTASKS, LIBXSMM_OVERHEAD, libxsmm_nt,
      float, flags | LIBXSMM_GEMM_FLAG_F32PREC, tm, tn, tk, *m, *n, *k,
      0 != alpha ? *alpha : ((float)LIBXSMM_ALPHA),
      a, *(lda ? lda : LIBXSMM_LD(m, k)), b, *(ldb ? ldb : LIBXSMM_LD(k, n)),
      0 != beta ? *beta : ((float)LIBXSMM_BETA),
      c, *(ldc ? ldc : LIBXSMM_LD(m, n)));
  }
#endif
}


LIBXSMM_API_DEFINITION void libxsmm_dgemm(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const double* alpha, const double* a, const libxsmm_blasint* lda,
  const double* b, const libxsmm_blasint* ldb,
  const double* beta, double* c, const libxsmm_blasint* ldc)
{
  LIBXSMM_GEMM_DECLARE_FLAGS(flags, transa, transb, m, n, k, a, b, c);
#if defined(LIBXSMM_GEMM_TILED)
  if (0 == LIBXSMM_MOD2(libxsmm_mt, 2))
#endif
  { /* below-threshold GEMM */
    LIBXSMM_DGEMM(flags, *m, *n, *k,
      0 != alpha ? *alpha : ((double)LIBXSMM_ALPHA),
      a, *(lda ? lda : LIBXSMM_LD(m, k)), b, *(ldb ? ldb : LIBXSMM_LD(k, n)),
      0 != beta ? *beta : ((double)LIBXSMM_BETA),
      c, *(ldc ? ldc : LIBXSMM_LD(m, n)));
  }
#if defined(LIBXSMM_GEMM_TILED)
  else { /* tiled GEMM */
    LIBXSMM_GEMM_DESCRIPTOR_DIM_TYPE tm, tn, tk;
    LIBXSMM_INIT
    tm = libxsmm_gemm_tile[0/*DP*/][0/*M*/];
    tn = libxsmm_gemm_tile[0/*DP*/][1/*N*/];
    tk = libxsmm_gemm_tile[0/*DP*/][2/*K*/];
    LIBXSMM_TILED_XGEMM(LIBXSMM_NOOP, LIBXSMM_NOOP, LIBXSMM_NOOP,
      LIBXSMM_GEMM_COLLAPSE, LIBXSMM_NOOP_ARGS, LIBXSMM_NOOP_ARGS, LIBXSMM_NOOP,
      LIBXSMM_MIN_NTASKS, LIBXSMM_OVERHEAD, libxsmm_nt,
      double, flags, tm, tn, tk, *m, *n, *k,
      0 != alpha ? *alpha : ((double)LIBXSMM_ALPHA),
      a, *(lda ? lda : LIBXSMM_LD(m, k)), b, *(ldb ? ldb : LIBXSMM_LD(k, n)),
      0 != beta ? *beta : ((double)LIBXSMM_BETA),
      c, *(ldc ? ldc : LIBXSMM_LD(m, n)));
  }
#endif
}


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


/* implementation provided for Fortran 77 compatibility */
LIBXSMM_API_DEFINITION void LIBXSMM_FSYMBOL(libxsmm_blas_sgemm)(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const float* alpha, const float* a, const libxsmm_blasint* lda,
  const float* b, const libxsmm_blasint* ldb,
  const float* beta, float* c, const libxsmm_blasint* ldc)
{
  libxsmm_blas_sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}


/* implementation provided for Fortran 77 compatibility */
LIBXSMM_API_DEFINITION void LIBXSMM_FSYMBOL(libxsmm_blas_dgemm)(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const double* alpha, const double* a, const libxsmm_blasint* lda,
  const double* b, const libxsmm_blasint* ldb,
  const double* beta, double* c, const libxsmm_blasint* ldc)
{
  libxsmm_blas_dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

#endif /*defined(LIBXSMM_BUILD)*/

