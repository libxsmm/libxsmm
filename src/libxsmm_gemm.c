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
# include "libxsmm_gemm_extwrap.c"
#else
# include "libxsmm_gemm_ext.h"
#endif

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#include <stdlib.h>
#include <stdint.h>
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif


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


LIBXSMM_RETARGETABLE LIBXSMM_VISIBILITY_INTERNAL int libxsmm_internal_tile_sizes[/*configs*/][2/*DP/SP*/][3/*TILE_M,TILE_N,TILE_K*/] = {
  { { 128, 48, 48 }, { 64, 48, 80 } }, /*generic*/
  { { 128, 48, 48 }, { 64, 48, 80 } }  /*knl*/
};
LIBXSMM_RETARGETABLE int libxsmm_internal_tile_size[/*DP/SP*/][3/*TILE_M,TILE_N,TILE_K*/] = {
  { 0, 0, 0 }, { 0, 0, 0 }
};
LIBXSMM_RETARGETABLE int libxsmm_internal_num_nt = 2;
LIBXSMM_RETARGETABLE int libxsmm_internal_gemm = 0;


LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE void libxsmm_gemm_configure(const char* archid, int gemm_kind)
{
  const int config = (0 == archid || 'k' != archid[0] || 'n' != archid[1] || 'l' != archid[2]) ? 0 : 1;
  const char* env[3], *const env_gemm_kind = getenv("LIBXSMM_GEMM");

  /* determine what will be executed in the wrapper code (0: small gemm, 1: sequential, 2: parallelized) */
  libxsmm_internal_gemm = (env_gemm_kind ? atoi(env_gemm_kind) : gemm_kind);
  libxsmm_internal_num_nt = 1 == config ? 4 : 2; /* threads per core */

  /* attempt to setup tile sizes from the environment (LIBXSMM_TILEM, LIBXSMM_TILEN, and LIBXSMM_TILEK) */
  env[0] = getenv("LIBXSMM_TILEM"); env[1] = getenv("LIBXSMM_TILEN"); env[2] = getenv("LIBXSMM_TILEK");
  libxsmm_internal_tile_size[0/*DP*/][0/*M*/] = (env[0] ? atoi(env[0]) : 0);
  libxsmm_internal_tile_size[0/*DP*/][1/*N*/] = (env[1] ? atoi(env[1]) : 0);
  libxsmm_internal_tile_size[0/*DP*/][2/*K*/] = (env[2] ? atoi(env[2]) : 0);
  /* environment-defined tile sizes applies for DP and SP */
  libxsmm_internal_tile_size[1/*SP*/][0/*M*/] = libxsmm_internal_tile_size[0/*DP*/][0];
  libxsmm_internal_tile_size[1/*SP*/][1/*N*/] = libxsmm_internal_tile_size[0/*DP*/][1];
  libxsmm_internal_tile_size[1/*SP*/][2/*K*/] = libxsmm_internal_tile_size[0/*DP*/][2];

  /* load predefined configuration if tile size is not setup by the environment */
  if (0 >= libxsmm_internal_tile_size[0/*DP*/][0/*M*/]) libxsmm_internal_tile_size[0][0] = libxsmm_internal_tile_sizes[config][0][0];
  if (0 >= libxsmm_internal_tile_size[0/*DP*/][1/*N*/]) libxsmm_internal_tile_size[0][1] = libxsmm_internal_tile_sizes[config][0][1];
  if (0 >= libxsmm_internal_tile_size[0/*DP*/][2/*K*/]) libxsmm_internal_tile_size[0][2] = libxsmm_internal_tile_sizes[config][0][2];
  if (0 >= libxsmm_internal_tile_size[1/*SP*/][0/*M*/]) libxsmm_internal_tile_size[1][0] = libxsmm_internal_tile_sizes[config][1][0];
  if (0 >= libxsmm_internal_tile_size[1/*SP*/][1/*N*/]) libxsmm_internal_tile_size[1][1] = libxsmm_internal_tile_sizes[config][1][1];
  if (0 >= libxsmm_internal_tile_size[1/*SP*/][2/*K*/]) libxsmm_internal_tile_size[1][2] = libxsmm_internal_tile_sizes[config][1][2];
}


LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE LIBXSMM_GEMM_WEAK int libxsmm_gemm_init(const char* archid,
  libxsmm_sgemm_function sgemm_function, libxsmm_dgemm_function dgemm_function)
{
  /* internal pre-initialization step */
  libxsmm_gemm_configure(archid, 0/*default gemm kind is small gemm*/);

  if (NULL != sgemm_function) {
    libxsmm_internal_sgemm = sgemm_function;
  }
#if defined(LIBXSMM_GEMM_EXTWRAP) && defined(__STATIC)
  else if (NULL != LIBXSMM_FSYMBOL(__real_sgemm)) {
    libxsmm_internal_sgemm = LIBXSMM_FSYMBOL(__real_sgemm);
  }
  else if (NULL != LIBXSMM_FSYMBOL(__real_mkl_sgemm)) {
    libxsmm_internal_sgemm = LIBXSMM_FSYMBOL(__real_mkl_sgemm);
  }
#endif /*defined(LIBXSMM_GEMM_EXTWRAP)*/

  if (NULL != dgemm_function) {
    libxsmm_internal_dgemm = dgemm_function;
  }
#if defined(LIBXSMM_GEMM_EXTWRAP) && defined(__STATIC)
  else if (NULL != LIBXSMM_FSYMBOL(__real_dgemm)) {
    libxsmm_internal_dgemm = LIBXSMM_FSYMBOL(__real_dgemm);
  }
  else if (NULL != LIBXSMM_FSYMBOL(__real_mkl_dgemm)) {
    libxsmm_internal_dgemm = LIBXSMM_FSYMBOL(__real_mkl_dgemm);
  }
#endif /*defined(LIBXSMM_GEMM_EXTWRAP)*/

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

