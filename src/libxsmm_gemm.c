/******************************************************************************
** Copyright (c) 2015-2017, Intel Corporation                                **
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
#include <libxsmm_intrinsics_x86.h>
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
  /* setup tile sizes according to CPUID or environment (LIBXSMM_GEMM_M, LIBXSMM_GEMM_N, LIBXSMM_GEMM_K) */
  const unsigned int tile_configs[/*configs*/][2/*DP/SP*/][3/*TILE_M,TILE_N,TILE_K*/][8/*size-range*/] = {
    /* generic (hsw) */
    { { {  25,  50,  69, 169, 169, 169, 169, 169 }, {  37,  98,  78,  39,  39,  39,  39,  39 }, { 100,  81,  55,  37,  37,  37,  37,  37 } },   /* DP */
      { {  43,  49, 107, 103, 103, 103, 103, 103 }, {  38,  52, 113, 141, 141, 141, 141, 141 }, { 232,  89, 100,  76,  76,  76,  76,  76 } } }, /* SP */
    /* mic (knl/knm) */
    { { { 168, 130, 131, 110, 110, 110, 110, 256 }, {  10,  28,  20,  24,  24,  24,  24,  10 }, {  39,  43,  40,  63,  63,  63,  63,  77 } },   /* DP */
      { {  69, 152, 149, 172, 172, 172, 172, 172 }, {  11,  14,  18,  28,  28,  28,  28,  28 }, { 100, 103,  61,  63,  63,  63,  63,  63 } } }, /* SP */
    /* core (skx) */
    { { {  39,  52,  57, 201, 256, 201, 201, 201 }, {  26,  86, 115,  14,  27,  14,  14,  14 }, { 256, 101, 102,  53, 114,  53,  53,  53 } },   /* DP */
      { {  41, 119, 102, 106, 106, 106, 106, 106 }, {  32,  65, 108, 130, 130, 130, 130, 130 }, {  73,  90,  86,  89,  89,  89,  89,  89 } } }  /* SP */
  };
  const char *const env_m = getenv("LIBXSMM_GEMM_M"), *const env_n = getenv("LIBXSMM_GEMM_N"), *const env_k = getenv("LIBXSMM_GEMM_K");
  const char *const env_p = getenv("LIBXSMM_GEMM_TILED_PREFETCH"), *const env_w = getenv("LIBXSMM_GEMM_WRAP");
  const int uid = ((0 == env_p || 0 == *env_p) ? 6/*LIBXSMM_PREFETCH_AL2_AHEAD*/ : atoi(env_p));
  const int gemm_m = ((0 == env_m || 0 == *env_m) ? -1 : atoi(env_m));
  const int gemm_n = ((0 == env_n || 0 == *env_n) ? -1 : atoi(env_n));
  const int gemm_k = ((0 == env_k || 0 == *env_k) ? -1 : atoi(env_k));
  int config, i;

  if (LIBXSMM_X86_AVX512_CORE <= archid) {
    config = 2;
  }
  else if (LIBXSMM_X86_AVX512_MIC <= archid && LIBXSMM_X86_AVX512_CORE > archid) {
    config = 1;
  }
  else {
    config = 0;
  }

  /* setup prefetch strategy for tiled GEMMs */
  libxsmm_gemm_tiled_prefetch = (0 <= uid ? libxsmm_gemm_uid2prefetch(uid) : prefetch);

  /* intercepted GEMMs (1: sequential and non-tiled, 2: parallelized and tiled). */
  libxsmm_gemm_wrap = ((0 == env_w || 0 == *env_w) ? (LIBXSMM_WRAP) : atoi(env_w));

  for (i = 0; i < 8; ++i) {
    /* environment-defined tile sizes apply for DP and SP */
    libxsmm_gemm_tile[0/*DP*/][0/*M*/][i] = libxsmm_gemm_tile[1/*SP*/][0/*M*/][i] = (unsigned int)LIBXSMM_MAX(gemm_m, 0);
    libxsmm_gemm_tile[0/*DP*/][1/*N*/][i] = libxsmm_gemm_tile[1/*SP*/][1/*N*/][i] = (unsigned int)LIBXSMM_MAX(gemm_n, 0);
    libxsmm_gemm_tile[0/*DP*/][2/*K*/][i] = libxsmm_gemm_tile[1/*SP*/][2/*K*/][i] = (unsigned int)LIBXSMM_MAX(gemm_k, 0);
    /* load predefined configuration if tile size is not setup by the environment */
    if (0 >= libxsmm_gemm_tile[0/*DP*/][0/*M*/][i]) libxsmm_gemm_tile[0][0][i] = tile_configs[config][0][0][i];
    if (0 >= libxsmm_gemm_tile[0/*DP*/][1/*N*/][i]) libxsmm_gemm_tile[0][1][i] = tile_configs[config][0][1][i];
    if (0 >= libxsmm_gemm_tile[0/*DP*/][2/*K*/][i]) libxsmm_gemm_tile[0][2][i] = tile_configs[config][0][2][i];
    if (0 >= libxsmm_gemm_tile[1/*SP*/][0/*M*/][i]) libxsmm_gemm_tile[1][0][i] = tile_configs[config][1][0][i];
    if (0 >= libxsmm_gemm_tile[1/*SP*/][1/*N*/][i]) libxsmm_gemm_tile[1][1][i] = tile_configs[config][1][1][i];
    if (0 >= libxsmm_gemm_tile[1/*SP*/][2/*K*/][i]) libxsmm_gemm_tile[1][2][i] = tile_configs[config][1][2][i];
  }
}


LIBXSMM_API_DEFINITION void libxsmm_gemm_finalize(void)
{
}


LIBXSMM_API_DEFINITION int libxsmm_gemm_prefetch2uid(libxsmm_gemm_prefetch_type prefetch)
{
  switch (prefetch) {
    case LIBXSMM_PREFETCH_SIGONLY:            return 2;
    case LIBXSMM_PREFETCH_BL2_VIA_C:          return 3;
    case LIBXSMM_PREFETCH_AL2_AHEAD:          return 4;
    case LIBXSMM_PREFETCH_AL2BL2_VIA_C_AHEAD: return 5;
    case LIBXSMM_PREFETCH_AL2:                return 6;
    case LIBXSMM_PREFETCH_AL2BL2_VIA_C:       return 7;
    case LIBXSMM_PREFETCH_AL2_JPST:           return 8;
    case LIBXSMM_PREFETCH_AL2BL2_VIA_C_JPST:  return 9;
    /*case LIBXSMM_PREFETCH_AL2CL2BL2_VIA_C:    return 10;*/
    case LIBXSMM_PREFETCH_AL1:                return 10;
    case LIBXSMM_PREFETCH_BL1:                return 11;
    case LIBXSMM_PREFETCH_CL1:                return 12;
    case LIBXSMM_PREFETCH_AL1_BL1:            return 13;
    case LIBXSMM_PREFETCH_BL1_CL1:            return 14;
    case LIBXSMM_PREFETCH_AL1_CL1:            return 15;
    case LIBXSMM_PREFETCH_AL1_BL1_CL1:        return 16;
    default: {
      assert(LIBXSMM_PREFETCH_NONE == prefetch);
      return 0;
    }
  }
}


LIBXSMM_API_DEFINITION libxsmm_gemm_prefetch_type libxsmm_gemm_uid2prefetch(int uid)
{
  switch (uid) {
    case  2: return LIBXSMM_PREFETCH_SIGONLY;             /* pfsigonly */
    case  3: return LIBXSMM_PREFETCH_BL2_VIA_C;           /* BL2viaC */
    case  4: return LIBXSMM_PREFETCH_AL2_AHEAD;           /* curAL2 */
    case  5: return LIBXSMM_PREFETCH_AL2BL2_VIA_C_AHEAD;  /* curAL2_BL2viaC */
    case  6: return LIBXSMM_PREFETCH_AL2;                 /* AL2 */
    case  7: return LIBXSMM_PREFETCH_AL2BL2_VIA_C;        /* AL2_BL2viaC */
    case  8: return LIBXSMM_PREFETCH_AL2_JPST;            /* AL2jpst */
    case  9: return LIBXSMM_PREFETCH_AL2BL2_VIA_C_JPST;   /* AL2jpst_BL2viaC */
    /*case 10: return LIBXSMM_PREFETCH_AL2CL2BL2_VIA_C;*/     /* AL2_BL2viaC_CL2 */
    case 10: return LIBXSMM_PREFETCH_AL1;
    case 11: return LIBXSMM_PREFETCH_BL1;
    case 12: return LIBXSMM_PREFETCH_CL1;
    case 13: return LIBXSMM_PREFETCH_AL1_BL1;
    case 14: return LIBXSMM_PREFETCH_BL1_CL1;
    case 15: return LIBXSMM_PREFETCH_AL1_CL1;
    case 16: return LIBXSMM_PREFETCH_AL1_BL1_CL1;
    default: return LIBXSMM_PREFETCH_NONE;
  }
}


LIBXSMM_API_DEFINITION void libxsmm_gemm_print(void* ostream,
  libxsmm_gemm_precision precision, const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const void* alpha, const void* a, const libxsmm_blasint* lda,
  const void* b, const libxsmm_blasint* ldb,
  const void* beta, void* c, const libxsmm_blasint* ldc)
{
  const libxsmm_blasint nn = *(n ? n : m), kk = *(k ? k : m), ilda = *(lda ? lda : m), ildb = (ldb ? *ldb : kk), ildc = *(ldc ? ldc : m);
  const char ctransa = (char)(0 != transa ? (*transa) : (0 == (LIBXSMM_FLAGS & LIBXSMM_GEMM_FLAG_TRANS_A) ? 'N' : 'T'));
  const char ctransb = (char)(0 != transb ? (*transb) : (0 == (LIBXSMM_FLAGS & LIBXSMM_GEMM_FLAG_TRANS_B) ? 'N' : 'T'));
  char string_a[128], string_b[128];

  if (LIBXSMM_GEMM_PRECISION_F64 == precision) {
    LIBXSMM_SNPRINTF(string_a, sizeof(string_a), "%g", 0 != alpha ? *((const double*)alpha) : LIBXSMM_ALPHA);
    LIBXSMM_SNPRINTF(string_b, sizeof(string_b), "%g", 0 != beta  ? *((const double*)beta)  : LIBXSMM_BETA);
  }
  else {
    LIBXSMM_SNPRINTF(string_a, sizeof(string_a), "%g", 0 != alpha ? *((const float*)alpha) : LIBXSMM_ALPHA);
    LIBXSMM_SNPRINTF(string_b, sizeof(string_b), "%g", 0 != beta  ? *((const float*)beta)  : LIBXSMM_BETA);
  }

  if (0 != ostream) { /* print information about GEMM call */
    fprintf((FILE*)ostream, "%cgemm('%c', '%c', %i/*m*/, %i/*n*/, %i/*k*/,\n"
                            "  %s/*alpha*/, %p/*a*/, %i/*lda*/,\n"
                            "              %p/*b*/, %i/*ldb*/,\n"
                            "   %s/*beta*/, %p/*c*/, %i/*ldc*/)",
      LIBXSMM_GEMM_PRECISION_F64 == precision ? 'd' : 's', ctransa, ctransa,
      *m, nn, kk, string_a, a, ilda, b, ildb, string_b, c, ildc);
  }
  else { /* dump input and output of the GEMM call into separate MHD files */
    char extension_header[256];
    size_t data_size[2], size[2];

    LIBXSMM_SNPRINTF(extension_header, sizeof(extension_header), "TRANS = %c\nALPHA = %s", ctransa, string_a);
    LIBXSMM_SNPRINTF(string_a, sizeof(string_a), "libxsmm_a_%p.mhd", a);
    data_size[0] = ilda; data_size[1] = kk; size[0] = *m; size[1] = kk;
    libxsmm_meta_image_write(string_a, data_size, size, 2/*ndims*/, 1/*ncomponents*/, a,
      LIBXSMM_GEMM_PRECISION_F64 == precision ? LIBXSMM_MHD_ELEMTYPE_F64 : LIBXSMM_MHD_ELEMTYPE_F32,
      0/*spacing*/, extension_header, 0/*extension*/, 0/*extension_size*/);

    LIBXSMM_SNPRINTF(extension_header, sizeof(extension_header), "\nTRANS = %c", ctransb);
    LIBXSMM_SNPRINTF(string_a, sizeof(string_a), "libxsmm_b_%p.mhd", b);
    data_size[0] = ildb; data_size[1] = nn; size[0] = kk; size[1] = nn;
    libxsmm_meta_image_write(string_a, data_size, size, 2/*ndims*/, 1/*ncomponents*/, b,
      LIBXSMM_GEMM_PRECISION_F64 == precision ? LIBXSMM_MHD_ELEMTYPE_F64 : LIBXSMM_MHD_ELEMTYPE_F32,
      0/*spacing*/, extension_header, 0/*extension*/, 0/*extension_size*/);

    LIBXSMM_SNPRINTF(extension_header, sizeof(extension_header), "BETA = %s", string_b);
    LIBXSMM_SNPRINTF(string_a, sizeof(string_a), "libxsmm_c_%p.mhd", c);
    data_size[0] = ildc; data_size[1] = nn; size[0] = *m; size[1] = nn;
    libxsmm_meta_image_write(string_a, data_size, size, 2/*ndims*/, 1/*ncomponents*/, c,
      LIBXSMM_GEMM_PRECISION_F64 == precision ? LIBXSMM_MHD_ELEMTYPE_F64 : LIBXSMM_MHD_ELEMTYPE_F32,
      0/*spacing*/, extension_header, 0/*extension*/, 0/*extension_size*/);
  }
}


LIBXSMM_API_DEFINITION int libxsmm_gemm_stat(libxsmm_gemm_precision precision, const void* matrix,
  libxsmm_blasint m, libxsmm_blasint n, const libxsmm_blasint* ld, libxsmm_stat_info* stat)
{
  int result = EXIT_SUCCESS;
  if (0 != matrix && 0 != stat) {
    const libxsmm_blasint ldx = (0 != ld ? *ld : m);
    double sum_error = 0; /* Kahan's compensation */
    libxsmm_blasint i, j;
    switch(precision) {
      case LIBXSMM_GEMM_PRECISION_F64: {
        const double *const values = (const double*)matrix;
        stat->sum = 0;
        for (i = 0; i < n; ++i) {
          for (j = 0; j < m; ++j) {
            const double value = values[i*ldx+j];
            const double v0 = value - sum_error;
            const double v1 = stat->sum + v0;
            sum_error = (v1 - stat->sum) - v0;
            stat->sum = v1;
          }
        }
      } break;
      case LIBXSMM_GEMM_PRECISION_F32: {
        const float *const values = (const float*)matrix;
        stat->sum = 0;
        for (i = 0; i < n; ++i) {
          for (j = 0; j < m; ++j) {
            const double value = values[i*ldx+j];
            const double v0 = value - sum_error;
            const double v1 = stat->sum + v0;
            sum_error = (v1 - stat->sum) - v0;
            stat->sum = v1;
          }
        }
      } break;
      case LIBXSMM_GEMM_PRECISION_I16: {
        result = EXIT_FAILURE;
      } break;
#if !defined(NDEBUG)
      default: {
        result = EXIT_FAILURE;
      }
#endif
    }
  }
  else {
    result = EXIT_FAILURE;
  }
  return result;
}


LIBXSMM_API_DEFINITION void libxsmm_blas_sgemm(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const float* alpha, const float* a, const libxsmm_blasint* lda,
  const float* b, const libxsmm_blasint* ldb,
  const float* beta, float* c, const libxsmm_blasint* ldc)
{
  const float ralpha = (0 != alpha ? *alpha : ((float)LIBXSMM_ALPHA));
  const float rbeta = (0 != beta ? *beta : ((float)LIBXSMM_BETA));
  const libxsmm_blasint ilda = *(lda ? lda : LIBXSMM_LD(m, k));
  const libxsmm_blasint ildb = *(ldb ? ldb : LIBXSMM_LD(k, n));
  const libxsmm_blasint ildc = *(ldc ? ldc : LIBXSMM_LD(m, n));
  LIBXSMM_GEMM_DECLARE_FLAGS(flags, transa, transb);
  LIBXSMM_BLAS_SGEMM(flags, *m, *n, *k, ralpha, a, ilda, b, ildb, rbeta, c, ildc);
}


LIBXSMM_API_DEFINITION void libxsmm_blas_dgemm(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const double* alpha, const double* a, const libxsmm_blasint* lda,
  const double* b, const libxsmm_blasint* ldb,
  const double* beta, double* c, const libxsmm_blasint* ldc)
{
  const double ralpha = (0 != alpha ? *alpha : ((double)LIBXSMM_ALPHA));
  const double rbeta = (0 != beta ? *beta : ((double)LIBXSMM_BETA));
  const libxsmm_blasint ilda = *(lda ? lda : LIBXSMM_LD(m, k));
  const libxsmm_blasint ildb = *(ldb ? ldb : LIBXSMM_LD(k, n));
  const libxsmm_blasint ildc = *(ldc ? ldc : LIBXSMM_LD(m, n));
  LIBXSMM_GEMM_DECLARE_FLAGS(flags, transa, transb);
  LIBXSMM_BLAS_DGEMM(flags, *m, *n, *k, ralpha, a, ilda, b, ildb, rbeta, c, ildc);
}


LIBXSMM_API_DEFINITION void libxsmm_sgemm(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const float* alpha, const float* a, const libxsmm_blasint* lda,
  const float* b, const libxsmm_blasint* ldb,
  const float* beta, float* c, const libxsmm_blasint* ldc)
{
  const float ralpha = (0 != alpha ? *alpha : ((float)LIBXSMM_ALPHA));
  const float rbeta = (0 != beta ? *beta : ((float)LIBXSMM_BETA));
  const libxsmm_blasint ilda = *(lda ? lda : LIBXSMM_LD(m, k));
  const libxsmm_blasint ildb = *(ldb ? ldb : LIBXSMM_LD(k, n));
  const libxsmm_blasint ildc = *(ldc ? ldc : LIBXSMM_LD(m, n));
#if !defined(NDEBUG) && (0 == LIBXSMM_NO_BLAS)
  const char *const check = getenv("LIBXSMM_CHECK");
  float* d;
#endif
  LIBXSMM_GEMM_DECLARE_FLAGS(flags, transa, transb);
#if !defined(NDEBUG) && (0 == LIBXSMM_NO_BLAS)
  d = (float*)((0 == LIBXSMM_GEMM_NO_BYPASS(flags, ralpha, rbeta)
      || 0 == check || 0 == *check || 0 == check[0]) ? 0
    : malloc((*m) * (*n) * sizeof(float)));
  if (0 != d) {
    libxsmm_blasint i, j;
    for (i = 0; i < (*n); ++i) {
      for (j = 0; j < (*m); ++j) {
        d[i*(*m)+j] = c[i*ildc+j];
      }
    }
  }
#endif
  LIBXSMM_SGEMM(flags, *m, *n, *k, ralpha, a, ilda, b, ildb, rbeta, c, ildc);
#if !defined(NDEBUG) && (0 == LIBXSMM_NO_BLAS)
  if (0 != d) {
    libxsmm_stat_info s1, s2;
    if (EXIT_SUCCESS == libxsmm_gemm_stat(LIBXSMM_GEMM_PRECISION_F32, c, *m, *n, ldc, &s1)) {
      libxsmm_blas_sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, d, m);
      if (EXIT_SUCCESS == libxsmm_gemm_stat(LIBXSMM_GEMM_PRECISION_F32, d, *m, *n, m, &s2)) {
        LIBXSMM_FLOCK(stderr);
        libxsmm_gemm_print(stderr, LIBXSMM_GEMM_PRECISION_F32, transa, transb,
          m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        fprintf(stderr, " sum1=%f sum2=%f\n", s1.sum, s2.sum);
        LIBXSMM_FUNLOCK(stderr);
      }
    }
    free(d);
  }
#endif
}


LIBXSMM_API_DEFINITION void libxsmm_dgemm(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const double* alpha, const double* a, const libxsmm_blasint* lda,
  const double* b, const libxsmm_blasint* ldb,
  const double* beta, double* c, const libxsmm_blasint* ldc)
{
  const double ralpha = (0 != alpha ? *alpha : ((double)LIBXSMM_ALPHA));
  const double rbeta = (0 != beta ? *beta : ((double)LIBXSMM_BETA));
  const libxsmm_blasint ilda = *(lda ? lda : LIBXSMM_LD(m, k));
  const libxsmm_blasint ildb = *(ldb ? ldb : LIBXSMM_LD(k, n));
  const libxsmm_blasint ildc = *(ldc ? ldc : LIBXSMM_LD(m, n));
#if !defined(NDEBUG) && (0 == LIBXSMM_NO_BLAS)
  const char *const check = getenv("LIBXSMM_CHECK");
  double* d;
#endif
  LIBXSMM_GEMM_DECLARE_FLAGS(flags, transa, transb);
#if !defined(NDEBUG) && (0 == LIBXSMM_NO_BLAS)
  d = (double*)((0 == LIBXSMM_GEMM_NO_BYPASS(flags, ralpha, rbeta)
      || 0 == check || 0 == *check || 0 == check[0]) ? 0
    : malloc((*m) * (*n) * sizeof(double)));
  if (0 != d) {
    libxsmm_blasint i, j;
    for (i = 0; i < (*n); ++i) {
      for (j = 0; j < (*m); ++j) {
        d[i*(*m)+j] = c[i*ildc+j];
      }
    }
  }
#endif
  LIBXSMM_DGEMM(flags, *m, *n, *k, ralpha, a, ilda, b, ildb, rbeta, c, ildc);
#if !defined(NDEBUG) && (0 == LIBXSMM_NO_BLAS)
  if (0 != d) {
    libxsmm_stat_info s1, s2;
    if (EXIT_SUCCESS == libxsmm_gemm_stat(LIBXSMM_GEMM_PRECISION_F64, c, *m, *n, ldc, &s1)) {
      libxsmm_blas_dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, d, m);
      if (EXIT_SUCCESS == libxsmm_gemm_stat(LIBXSMM_GEMM_PRECISION_F64, d, *m, *n, m, &s2)) {
        LIBXSMM_FLOCK(stderr);
        libxsmm_gemm_print(stderr, LIBXSMM_GEMM_PRECISION_F64, transa, transb,
          m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        fprintf(stderr, " sum1=%f sum2=%f\n", s1.sum, s2.sum);
        LIBXSMM_FUNLOCK(stderr);
      }
    }
    free(d);
  }
#endif
}


#if defined(LIBXSMM_BUILD)

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
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_blas_sgemm)(const char*, const char*,
  const libxsmm_blasint*, const libxsmm_blasint*, const libxsmm_blasint*,
  const float*, const float*, const libxsmm_blasint*,
  const float*, const libxsmm_blasint*,
  const float*, float*, const libxsmm_blasint*);
LIBXSMM_API_DEFINITION void LIBXSMM_FSYMBOL(libxsmm_blas_sgemm)(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const float* alpha, const float* a, const libxsmm_blasint* lda,
  const float* b, const libxsmm_blasint* ldb,
  const float* beta, float* c, const libxsmm_blasint* ldc)
{
  libxsmm_blas_sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}


/* implementation provided for Fortran 77 compatibility */
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_blas_dgemm)(const char*, const char*,
  const libxsmm_blasint*, const libxsmm_blasint*, const libxsmm_blasint*,
  const double*, const double*, const libxsmm_blasint*,
  const double*, const libxsmm_blasint*,
  const double*, double*, const libxsmm_blasint*);
LIBXSMM_API_DEFINITION void LIBXSMM_FSYMBOL(libxsmm_blas_dgemm)(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const double* alpha, const double* a, const libxsmm_blasint* lda,
  const double* b, const libxsmm_blasint* ldb,
  const double* beta, double* c, const libxsmm_blasint* ldc)
{
  libxsmm_blas_dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

#endif /*defined(LIBXSMM_BUILD)*/

