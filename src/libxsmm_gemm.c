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
#include <libxsmm_mhd.h>
#include "libxsmm_main.h"


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
  /* setup tile sizes according to CPUID or environment (LIBXSMM_TGEMM_M, LIBXSMM_TGEMM_N, LIBXSMM_TGEMM_K) */
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
  const char *const env_m = getenv("LIBXSMM_TGEMM_M"), *const env_n = getenv("LIBXSMM_TGEMM_N"), *const env_k = getenv("LIBXSMM_TGEMM_K");
  const char *const env_p = getenv("LIBXSMM_TGEMM_PREFETCH"), *const env_w = getenv("LIBXSMM_GEMM_WRAP");
  const char *const env_s = getenv("LIBXSMM_GEMM_BATCHSIZE"), *const env_b = getenv("LIBXSMM_GEMM_BATCH");
  const int uid = ((0 == env_p || 0 == *env_p) ? 6/*LIBXSMM_PREFETCH_AL2_AHEAD*/ : atoi(env_p));
  const int batchsize = ((0 == env_s || 0 == *env_s) ? -1 : atoi(env_s));
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

  /* intercepted GEMMs (1: sequential and non-tiled, 2: parallelized and tiled) */
  libxsmm_gemm_wrap = ((0 == env_w || 0 == *env_w) ? (LIBXSMM_WRAP) : atoi(env_w));

  if (0 != env_b && 0 != *env_b) { /* even/negative: parallelized, odd: sequential */
    libxsmm_gemm_batchmode = atoi(env_b);
  }

  if (0 != libxsmm_gemm_wrap) { /* batch-recording available */
    /* use libxsmm_malloc to draw memory from the default memory allocation domain */
    if (EXIT_SUCCESS == libxsmm_xmalloc((void**)&libxsmm_gemm_batcharray,
      sizeof(libxsmm_gemm_batchitem) * (0 > batchsize ? (LIBXSMM_GEMM_BATCHSIZE) : batchsize),
      0, LIBXSMM_MALLOC_FLAG_SCRATCH, 0/*extra*/, 0/*extra_size*/))
    {
      libxsmm_gemm_batchsize = batchsize;
    }
  }

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
  libxsmm_free(libxsmm_gemm_batcharray);
}


LIBXSMM_API_DEFINITION unsigned char libxsmm_gemm_typesize(libxsmm_gemm_precision precision)
{
  return libxsmm_typesize((libxsmm_datatype)precision);
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
    case  1: return LIBXSMM_PREFETCH_NONE;                /* nopf */
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
    default: {
      if (0 != libxsmm_verbosity) { /* library code is expected to be mute */
        static int error_once = 0;
        if (1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED)) {
          fprintf(stderr, "LIBXSMM WARNING: invalid prefetch strategy requested!\n");
        }
      }
      return LIBXSMM_PREFETCH_NONE;
    }
  }
}


LIBXSMM_API_DEFINITION int libxsmm_dgemm_descriptor_init(libxsmm_gemm_descriptor* descriptor,
  int m, int n, int k, int lda, int ldb, int ldc, double alpha, double beta, int flags, int prefetch)
{
  int result;
  if (LIBXSMM_GEMM_NO_BYPASS(flags, alpha, beta) && 0 != descriptor) {
    LIBXSMM_GEMM_DESCRIPTOR(*descriptor, LIBXSMM_GEMM_PRECISION(double), flags, m, n, k, lda, ldb, ldc, alpha, beta, prefetch);
    result = EXIT_SUCCESS;
  }
  else { /* unsupported */
    result = EXIT_FAILURE;
  }
  return result;
}


LIBXSMM_API_DEFINITION int libxsmm_sgemm_descriptor_init(libxsmm_gemm_descriptor* descriptor,
  int m, int n, int k, int lda, int ldb, int ldc, float alpha, float beta, int flags, int prefetch)
{
  int result;
  if (LIBXSMM_GEMM_NO_BYPASS(flags, alpha, beta) && 0 != descriptor) {
    LIBXSMM_GEMM_DESCRIPTOR(*descriptor, LIBXSMM_GEMM_PRECISION(float), flags, m, n, k, lda, ldb, ldc, alpha, beta, prefetch);
    result = EXIT_SUCCESS;
  }
  else { /* unsupported */
    result = EXIT_FAILURE;
  }
  return result;
}


LIBXSMM_API_DEFINITION int libxsmm_wgemm_descriptor_init(libxsmm_gemm_descriptor* descriptor,
  int m, int n, int k, int lda, int ldb, int ldc, int alpha, int beta, int flags, int prefetch)
{
  int result;
  if (LIBXSMM_GEMM_NO_BYPASS(flags, alpha, beta) && 0 != descriptor) {
    LIBXSMM_GEMM_DESCRIPTOR(*descriptor, LIBXSMM_GEMM_PRECISION(short), flags, m, n, k, lda, ldb, ldc, alpha, beta, prefetch);
    result = EXIT_SUCCESS;
  }
  else { /* unsupported */
    result = EXIT_FAILURE;
  }
  return result;
}


/*DEPRECATED*/LIBXSMM_API_DEFINITION libxsmm_gemm_descriptor* libxsmm_create_dgemm_descriptor(char transa, char transb,
  int m, int n, int k, int lda, int ldb, int ldc, double alpha, double beta,
  libxsmm_gemm_prefetch_type strategy)
{
  libxsmm_gemm_descriptor* result = (libxsmm_gemm_descriptor*)malloc(sizeof(libxsmm_gemm_descriptor));
  if (EXIT_SUCCESS != libxsmm_dgemm_descriptor_init(result, m, n, k, lda, ldb, ldc,
    alpha, beta, LIBXSMM_GEMM_FLAGS(transa, transb), strategy))
  {
    free(result);
    result = 0;
  }
  if (1 < libxsmm_verbosity || 0 > libxsmm_verbosity) { /* library code is expected to be mute */
    static int error_once = 0;
    if (1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED)) {
      fprintf(stderr, "LIBXSMM WARNING: create_dgemm_descriptor is deprecated, use libxsmm_gemm_descriptor_init instead!\n");
    }
  }
  return result;
}


/*DEPRECATED*/LIBXSMM_API_DEFINITION void libxsmm_release_gemm_descriptor(const libxsmm_gemm_descriptor* descriptor)
{
  free((void*)descriptor);
  if (1 < libxsmm_verbosity || 0 > libxsmm_verbosity) { /* library code is expected to be mute */
    static int error_once = 0;
    if (1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED)) {
      fprintf(stderr, "LIBXSMM WARNING: release_gemm_descriptor is deprecated, libxsmm_gemm_descriptor_init makes it superfluous!\n");
    }
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
  libxsmm_mhd_elemtype mhd_elemtype = LIBXSMM_MHD_ELEMTYPE_CHAR;
  char string_a[128], string_b[128], typeprefix = 0;

  switch (precision) {
    case LIBXSMM_GEMM_PRECISION_F64: {
      LIBXSMM_SNPRINTF(string_a, sizeof(string_a), "%g", 0 != alpha ? *((const double*)alpha) : LIBXSMM_ALPHA);
      LIBXSMM_SNPRINTF(string_b, sizeof(string_b), "%g", 0 != beta  ? *((const double*)beta)  : LIBXSMM_BETA);
      mhd_elemtype = LIBXSMM_MHD_ELEMTYPE_F64;
      typeprefix = 'd';
    } break;
    case LIBXSMM_GEMM_PRECISION_F32: {
      LIBXSMM_SNPRINTF(string_a, sizeof(string_a), "%g", 0 != alpha ? *((const float*)alpha) : LIBXSMM_ALPHA);
      LIBXSMM_SNPRINTF(string_b, sizeof(string_b), "%g", 0 != beta  ? *((const float*)beta)  : LIBXSMM_BETA);
      mhd_elemtype = LIBXSMM_MHD_ELEMTYPE_F32;
      typeprefix = 's';
    }
    default: /* TODO: support I16, etc. */;
  }

  if (0 != typeprefix) {
    if (0 != ostream) { /* print information about GEMM call */
      if (0 != a && 0 != b && 0 != c) {
        fprintf((FILE*)ostream, "%cgemm('%c', '%c', %lli/*m*/, %lli/*n*/, %lli/*k*/,\n"
                                "  %s/*alpha*/, %p/*a*/, %lli/*lda*/,\n"
                                "              %p/*b*/, %lli/*ldb*/,\n"
                                "   %s/*beta*/, %p/*c*/, %lli/*ldc*/)",
          typeprefix, ctransa, ctransa, (long long)*m, (long long)nn, (long long)kk,
          string_a, a, (long long)ilda, b, (long long)ildb, string_b, c, (long long)ildc);
      }
      else {
        fprintf((FILE*)ostream, "%cgemm('%c', '%c', %lli/*m*/, %lli/*n*/, %lli/*k*/, "
                                "%lli/*lda*/, %lli/*ldb*/, %lli/*ldc*/, %s/*alpha*/, %s/*beta*/)",
          typeprefix, ctransa, ctransa, (long long)*m, (long long)nn, (long long)kk,
          (long long)ilda, (long long)ildb, (long long)ildc, string_a, string_b);
      }
    }
    else { /* dump A, B, and C matrices into MHD files */
      char extension_header[256];
      size_t data_size[2], size[2];

      if (0 != a) {
        LIBXSMM_SNPRINTF(extension_header, sizeof(extension_header), "TRANS = %c\nALPHA = %s", ctransa, string_a);
        LIBXSMM_SNPRINTF(string_a, sizeof(string_a), "libxsmm_a_%p.mhd", a);
        data_size[0] = ilda; data_size[1] = kk; size[0] = *m; size[1] = kk;
        libxsmm_mhd_write(string_a, 0/*offset*/, size, data_size, 2/*ndims*/, 1/*ncomponents*/, mhd_elemtype, a,
          0/*header_size*/, extension_header, 0/*extension*/, 0/*extension_size*/);
      }
      if (0 != b) {
        LIBXSMM_SNPRINTF(extension_header, sizeof(extension_header), "\nTRANS = %c", ctransb);
        LIBXSMM_SNPRINTF(string_a, sizeof(string_a), "libxsmm_b_%p.mhd", b);
        data_size[0] = ildb; data_size[1] = nn; size[0] = kk; size[1] = nn;
        libxsmm_mhd_write(string_a, 0/*offset*/, size, data_size, 2/*ndims*/, 1/*ncomponents*/, mhd_elemtype, b,
          0/*header_size*/, extension_header, 0/*extension*/, 0/*extension_size*/);
      }
      if (0 != c) {
        LIBXSMM_SNPRINTF(extension_header, sizeof(extension_header), "BETA = %s", string_b);
        LIBXSMM_SNPRINTF(string_a, sizeof(string_a), "libxsmm_c_%p.mhd", c);
        data_size[0] = ildc; data_size[1] = nn; size[0] = *m; size[1] = nn;
        libxsmm_mhd_write(string_a, 0/*offset*/, size, data_size, 2/*ndims*/, 1/*ncomponents*/, mhd_elemtype, c,
          0/*header_size*/, extension_header, 0/*extension*/, 0/*extension_size*/);
      }
    }
  }
}


LIBXSMM_API_DEFINITION void libxsmm_blas_sgemm(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const float* alpha, const float* a, const libxsmm_blasint* lda,
  const float* b, const libxsmm_blasint* ldb,
  const float* beta, float* c, const libxsmm_blasint* ldc)
{
  const libxsmm_blasint nn = *(n ? n : m), kk = *(k ? k : m), ilda = *(lda ? lda : m), ildb = (ldb ? *ldb : kk), ildc = *(ldc ? ldc : m);
  const float ralpha = (0 != alpha ? *alpha : ((float)LIBXSMM_ALPHA));
  const float rbeta = (0 != beta ? *beta : ((float)LIBXSMM_BETA));
  const int flags = LIBXSMM_GEMM_PFLAGS(transa, transb, LIBXSMM_FLAGS);
  LIBXSMM_BLAS_SGEMM(flags, *m, nn, kk, ralpha, a, ilda, b, ildb, rbeta, c, ildc);
}


LIBXSMM_API_DEFINITION void libxsmm_blas_dgemm(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const double* alpha, const double* a, const libxsmm_blasint* lda,
  const double* b, const libxsmm_blasint* ldb,
  const double* beta, double* c, const libxsmm_blasint* ldc)
{
  const libxsmm_blasint nn = *(n ? n : m), kk = *(k ? k : m), ilda = *(lda ? lda : m), ildb = (ldb ? *ldb : kk), ildc = *(ldc ? ldc : m);
  const double ralpha = (0 != alpha ? *alpha : ((double)LIBXSMM_ALPHA));
  const double rbeta = (0 != beta ? *beta : ((double)LIBXSMM_BETA));
  const int flags = LIBXSMM_GEMM_PFLAGS(transa, transb, LIBXSMM_FLAGS);
  LIBXSMM_BLAS_DGEMM(flags, *m, nn, kk, ralpha, a, ilda, b, ildb, rbeta, c, ildc);
}


LIBXSMM_API_DEFINITION void libxsmm_sgemm(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const float* alpha, const float* a, const libxsmm_blasint* lda,
  const float* b, const libxsmm_blasint* ldb,
  const float* beta, float* c, const libxsmm_blasint* ldc)
{
  const libxsmm_blasint nn = *(n ? n : m), kk = *(k ? k : m), ilda = *(lda ? lda : m), ildb = (ldb ? *ldb : kk), ildc = *(ldc ? ldc : m);
  const float ralpha = (0 != alpha ? *alpha : ((float)LIBXSMM_ALPHA));
  const float rbeta = (0 != beta ? *beta : ((float)LIBXSMM_BETA));
  const int flags = LIBXSMM_GEMM_PFLAGS(transa, transb, LIBXSMM_FLAGS);
#if !defined(NDEBUG) && (0 == LIBXSMM_NO_BLAS)
  const char *const check = getenv("LIBXSMM_CHECK");
  float *const d = (float*)((0 == LIBXSMM_GEMM_NO_BYPASS(flags, ralpha, rbeta)
      || 0 == check || 0 == *check || 0 == check[0]) ? 0
    : libxsmm_aligned_scratch((*m) * nn * sizeof(float), 0/*auto-aligned*/));
  if (0 != d) {
    libxsmm_blasint i, j;
    for (i = 0; i < nn; ++i) {
      for (j = 0; j < (*m); ++j) {
        d[i*(*m)+j] = c[i*ildc+j];
      }
    }
  }
#endif
  LIBXSMM_SGEMM(flags, *m, nn, kk, ralpha, a, ilda, b, ildb, rbeta, c, ildc);
#if !defined(NDEBUG) && (0 == LIBXSMM_NO_BLAS)
  if (0 != d) {
    libxsmm_matdiff_info diff;
    libxsmm_blas_sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, d, m);
    if (EXIT_SUCCESS == libxsmm_matdiff(LIBXSMM_DATATYPE_F32, *m, nn, d, c, m, ldc, &diff)) {
      LIBXSMM_FLOCK(stderr);
      libxsmm_gemm_print(stderr, LIBXSMM_GEMM_PRECISION_F32, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
      fprintf(stderr, " L2abs=%f Linf=%f\n", diff.l2_abs, diff.linf_abs);
      LIBXSMM_FUNLOCK(stderr);
    }
    libxsmm_free(d);
  }
#endif
}


LIBXSMM_API_DEFINITION void libxsmm_dgemm(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const double* alpha, const double* a, const libxsmm_blasint* lda,
  const double* b, const libxsmm_blasint* ldb,
  const double* beta, double* c, const libxsmm_blasint* ldc)
{
  const libxsmm_blasint nn = *(n ? n : m), kk = *(k ? k : m), ilda = *(lda ? lda : m), ildb = (ldb ? *ldb : kk), ildc = *(ldc ? ldc : m);
  const double ralpha = (0 != alpha ? *alpha : ((double)LIBXSMM_ALPHA));
  const double rbeta = (0 != beta ? *beta : ((double)LIBXSMM_BETA));
  const int flags = LIBXSMM_GEMM_PFLAGS(transa, transb, LIBXSMM_FLAGS);
#if !defined(NDEBUG) && (0 == LIBXSMM_NO_BLAS)
  const char *const check = getenv("LIBXSMM_CHECK");
  double *const d = (double*)((0 == LIBXSMM_GEMM_NO_BYPASS(flags, ralpha, rbeta)
      || 0 == check || 0 == *check || 0 == check[0]) ? 0
    : libxsmm_aligned_scratch((*m) * nn * sizeof(double), 0/*auto-aligned*/));
  if (0 != d) {
    libxsmm_blasint i, j;
    for (i = 0; i < nn; ++i) {
      for (j = 0; j < (*m); ++j) {
        d[i*(*m)+j] = c[i*ildc+j];
      }
    }
  }
#endif
  LIBXSMM_DGEMM(flags, *m, nn, kk, ralpha, a, ilda, b, ildb, rbeta, c, ildc);
#if !defined(NDEBUG) && (0 == LIBXSMM_NO_BLAS)
  if (0 != d) {
    libxsmm_matdiff_info diff;
    libxsmm_blas_dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, d, m);
    if (EXIT_SUCCESS == libxsmm_matdiff(LIBXSMM_DATATYPE_F64, *m, nn, d, c, m, ldc, &diff)) {
      LIBXSMM_FLOCK(stderr);
      libxsmm_gemm_print(stderr, LIBXSMM_GEMM_PRECISION_F64, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
      fprintf(stderr, " L2abs=%f Linf=%f\n", diff.l2_abs, diff.linf_abs);
      LIBXSMM_FUNLOCK(stderr);
    }
    libxsmm_free(d);
  }
#endif
}


LIBXSMM_API_DEFINITION int libxsmm_mmbatch_thread(libxsmm_xmmfunction kernel, unsigned int typesize, const void* a_matrix, const void* b_matrix, void* c_matrix,
  int index_base, int index_stride, const unsigned int a_stride[], const unsigned int b_stride[], const unsigned int c_stride[], unsigned int batchsize,
  int tid, int nthreads)
{
  int result = EXIT_SUCCESS;

  if (0 != kernel.xmm && 0 != typesize && 0 != a_matrix && 0 != b_matrix && 0 != c_matrix &&
      /* use (signed) integer types, but check sanity of input */
      0 <= tid && tid < nthreads)
  {
    const unsigned int tasksize = (batchsize + nthreads - 1) / nthreads;
    const unsigned int begin = tid * tasksize;
    const unsigned int end = LIBXSMM_MIN(begin + tasksize, batchsize);

    if (begin < end) {
      const unsigned int da = (0 != a_stride ? (*a_stride - index_base) : index_base);
      const unsigned int db = (0 != b_stride ? (*b_stride - index_base) : index_base);
      const unsigned int dc = (0 != c_stride ? (*c_stride - index_base) : index_base);
      const char *a = (const char*)a_matrix, *b = (const char*)b_matrix, *ai, *bi;
      char *c = (char*)c_matrix, *ci;
      unsigned int n;

      assert(0 != dc || 1 == nthreads); /* TODO: implement synchronization */
      if (0 != index_stride) { /* stride arrays contain indexes */
        const char *const ia = (const char*)a_stride, *const ib = (const char*)b_stride, *const ic = (const char*)c_stride;
        ai = a + da * typesize; bi = b + db * typesize; ci = c + dc * typesize;
        for (n = begin; n < (end - 1); ++n) {
          const char *const an = a + (0 != ia ? (*((const unsigned int*)(ia + (n + 1) * index_stride)) - index_base) : index_base) * typesize;
          const char *const bn = b + (0 != ib ? (*((const unsigned int*)(ib + (n + 1) * index_stride)) - index_base) : index_base) * typesize;
          char *const cn = c + (0 != ic ? (*((const unsigned int*)(ic + (n + 1) * index_stride)) - index_base) : index_base) * typesize;
          kernel.xmm(ai, bi, ci, an, bn, cn); /* prefetch */
          ai = an; bi = bn; ci = cn;
        }
        /* last multiplication with pseudo-prefetch */
        kernel.xmm(ai, bi, ci, ai, bi, ci);
      }
      else if (typesize <= da && typesize <= db && typesize <= dc) { /* strides are measured in Bytes */
        ai = a; bi = b; ci = c;
        for (n = begin; n < (end - 1); ++n) {
          const char *const an = ai + da;
          const char *const bn = bi + db;
          char *const cn = ci + dc;
          kernel.xmm(ai, bi, ci, an, bn, cn); /* prefetch */
          ai = an; bi = bn; ci = cn; /* next */
        }
        /* last multiplication with pseudo-prefetch */
        kernel.xmm(ai, bi, ci, ai, bi, ci);
      }
      else { /* incorrect argument(s) */
        result = EXIT_FAILURE;
      }
    }
    else { /* nothing to do */
      result = EXIT_SUCCESS;
    }
  }
  else { /* incorrect argument(s) */
    result = EXIT_FAILURE;
  }
  return result;
}


LIBXSMM_API_DEFINITION int libxsmm_mmbatch(libxsmm_xmmfunction kernel, unsigned int typesize, const void* a_matrix, const void* b_matrix, void* c_matrix,
  int index_base, int index_stride, const unsigned int a_stride[], const unsigned int b_stride[], const unsigned int c_stride[], unsigned int batchsize)
{
  return libxsmm_mmbatch_thread(kernel, typesize, a_matrix, b_matrix, c_matrix, index_base, index_stride, a_stride, b_stride, c_stride, batchsize, 0/*tid*/, 1/*nthreads*/);
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


LIBXSMM_API_DEFINITION void LIBXSMM_FSYMBOL(libxsmm_smmbatch_thread)(libxsmm_xmmfunction kernel,
  const float a_matrix[], const float b_matrix[], float c_matrix[], const int* index_stride,
  const unsigned int a_stride[], const unsigned int b_stride[], const unsigned int c_stride[],
  const unsigned int* batchsize, const int* tid, const int* nthreads)
{
  assert(0 != a_matrix && 0 != b_matrix && 0 != c_matrix);
  assert(0 != index_stride && 0 != batchsize);
  assert(0 != tid && 0 != nthreads);
  libxsmm_mmbatch_thread(kernel, sizeof(float), a_matrix, b_matrix, c_matrix,
    1/*index_base*/, *index_stride, a_stride, b_stride, c_stride, *batchsize,
    *tid, *nthreads);
}


LIBXSMM_API_DEFINITION void LIBXSMM_FSYMBOL(libxsmm_dmmbatch_thread)(libxsmm_xmmfunction kernel,
  const float a_matrix[], const float b_matrix[], float c_matrix[], const int* index_stride,
  const unsigned int a_stride[], const unsigned int b_stride[], const unsigned int c_stride[],
  const unsigned int* batchsize, const int* tid, const int* nthreads)
{
  assert(0 != a_matrix && 0 != b_matrix && 0 != c_matrix);
  assert(0 != index_stride && 0 != batchsize);
  assert(0 != tid && 0 != nthreads);
  libxsmm_mmbatch_thread(kernel, sizeof(double), a_matrix, b_matrix, c_matrix,
    1/*index_base*/, *index_stride, a_stride, b_stride, c_stride, *batchsize,
    *tid, *nthreads);
}

#endif /*defined(LIBXSMM_BUILD)*/

