/******************************************************************************
** Copyright (c) 2016-2017, Intel Corporation                                **
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
#include "libxsmm_ext.h"


#if defined(LIBXSMM_BUILD) && defined(LIBXSMM_BUILD_EXT)

#if defined(LIBXSMM_GEMM_WRAP_STATIC) || defined(LIBXSMM_GEMM_WRAP_DYNAMIC)
LIBXSMM_API_DEFINITION LIBXSMM_GEMM_WEAK
libxsmm_sgemm_function libxsmm_original_sgemm(const void* caller)
{
  static LIBXSMM_TLS libxsmm_sgemm_function original = 0;
  LIBXSMM_GEMM_WRAPPER(float, original, caller);
  assert(0 != original);
  return original;
}


LIBXSMM_API_DEFINITION LIBXSMM_GEMM_WEAK
libxsmm_dgemm_function libxsmm_original_dgemm(const void* caller)
{
  static LIBXSMM_TLS libxsmm_dgemm_function original = 0;
  LIBXSMM_GEMM_WRAPPER(double, original, caller);
  assert(0 != original);
  return original;
}
#endif


LIBXSMM_API_DEFINITION void LIBXSMM_FSYMBOL(__wrap_sgemm)(
  const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const float* alpha, const float* a, const libxsmm_blasint* lda,
  const float* b, const libxsmm_blasint* ldb,
  const float* beta, float* c, const libxsmm_blasint* ldc)
{
  libxsmm_sgemm_omp(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}


LIBXSMM_API_DEFINITION void LIBXSMM_FSYMBOL(__wrap_dgemm)(
  const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const double* alpha, const double* a, const libxsmm_blasint* lda,
  const double* b, const libxsmm_blasint* ldb,
  const double* beta, double* c, const libxsmm_blasint* ldc)
{
  libxsmm_dgemm_omp(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

#endif /*defined(LIBXSMM_BUILD) && defined(LIBXSMM_BUILD_EXT)*/


LIBXSMM_API_DEFINITION void libxsmm_sgemm_omp(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const float* alpha, const float* a, const libxsmm_blasint* lda,
  const float* b, const libxsmm_blasint* ldb,
  const float* beta, float* c, const libxsmm_blasint* ldc)
{
  const unsigned long long size = 1ULL * (*m) * (*n) * (*k);
  LIBXSMM_INIT
  if (LIBXSMM_MAX_MNK < size) {
    const int index = LIBXSMM_MIN(libxsmm_icbrt(size) >> 10, 7);
    const unsigned int tm = LIBXSMM_MIN(libxsmm_gemm_tile[1/*SP*/][0/*M*/][index], (unsigned int)*m);
    const unsigned int tn = LIBXSMM_MIN(libxsmm_gemm_tile[1/*SP*/][1/*N*/][index], (unsigned int)*n);
    const unsigned int tk = LIBXSMM_MIN(libxsmm_gemm_tile[1/*SP*/][2/*K*/][index], (unsigned int)*k);
    const float ralpha = (0 != alpha ? *alpha : ((float)LIBXSMM_ALPHA));
    const float rbeta = (0 != beta ? *beta : ((float)LIBXSMM_BETA));
    const libxsmm_blasint ilda = *(lda ? lda : LIBXSMM_LD(m, k));
    const libxsmm_blasint ildb = *(ldb ? ldb : LIBXSMM_LD(k, n));
    const libxsmm_blasint ildc = *(ldc ? ldc : LIBXSMM_LD(m, n));
#if !defined(NDEBUG) && (0 == LIBXSMM_NO_BLAS)
    float* d = 0;
#endif
    LIBXSMM_GEMM_DECLARE_FLAGS(flags, transa, transb);
#if !defined(NDEBUG) && (0 == LIBXSMM_NO_BLAS)
    { const char *const check = getenv("LIBXSMM_CHECK");
      d = (float*)((0 == LIBXSMM_GEMM_NO_BYPASS(flags, ralpha, rbeta)
          || 0 == check || 0 == *check || 0 == check[0]) ? 0
        : malloc((*m) * (*n) * sizeof(float)));
      if (0 != d) {
        libxsmm_matcopy(d, c, sizeof(float), *m, *n, ildc, *m, 0/*prefetch*/);
      }
    }
#endif
    assert((0 < tm || 0 == *m) && (0 < tn || 0 == *n) && (0 < tk || 0 == *k) && 0 < libxsmm_nt);
#if defined(LIBXSMM_EXT_TASKS) /* implies _OPENMP */
    if (0 == omp_get_active_level())
#endif
    {
      LIBXSMM_TILED_XGEMM(
        LIBXSMM_EXT_PARALLEL, LIBXSMM_EXT_FOR_LOOP, LIBXSMM_EXT_FOR_KERNEL, LIBXSMM_NOOP,
        LIBXSMM_EXT_MIN_NTASKS, LIBXSMM_EXT_OVERHEAD, libxsmm_nt,
        float, flags, tm, tn, tk, *m, *n, *k,
        ralpha, a, ilda, b, ildb, rbeta, c, ildc);
    }
#if defined(LIBXSMM_EXT_TASKS)
    else {
      LIBXSMM_TILED_XGEMM(
        LIBXSMM_NOOP, LIBXSMM_NOOP_ARGS, LIBXSMM_EXT_TSK_KERNEL_ARGS,
        if (0 != libxsmm_sync) { LIBXSMM_EXT_TSK_SYNC } /* allow to omit synchronization */,
        LIBXSMM_EXT_MIN_NTASKS, LIBXSMM_EXT_OVERHEAD, libxsmm_nt,
        float, flags, tm, tn, tk, *m, *n, *k,
        ralpha, a, ilda, b, ildb, rbeta, c, ildc);
    }
#endif
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
  else if (0 < size) { /* small problem size */
    libxsmm_sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  }
}


LIBXSMM_API_DEFINITION void libxsmm_dgemm_omp(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const double* alpha, const double* a, const libxsmm_blasint* lda,
  const double* b, const libxsmm_blasint* ldb,
  const double* beta, double* c, const libxsmm_blasint* ldc)
{
  const unsigned long long size = 1ULL * (*m) * (*n) * (*k);
  LIBXSMM_INIT
  if (LIBXSMM_MAX_MNK < size) {
    const int index = LIBXSMM_MIN(libxsmm_icbrt(size) >> 10, 7);
    const unsigned int tm = LIBXSMM_MIN(libxsmm_gemm_tile[0/*DP*/][0/*M*/][index], (unsigned int)*m);
    const unsigned int tn = LIBXSMM_MIN(libxsmm_gemm_tile[0/*DP*/][1/*N*/][index], (unsigned int)*n);
    const unsigned int tk = LIBXSMM_MIN(libxsmm_gemm_tile[0/*DP*/][2/*K*/][index], (unsigned int)*k);
    const double ralpha = (0 != alpha ? *alpha : ((double)LIBXSMM_ALPHA));
    const double rbeta = (0 != beta ? *beta : ((double)LIBXSMM_BETA));
    const libxsmm_blasint ilda = *(lda ? lda : LIBXSMM_LD(m, k));
    const libxsmm_blasint ildb = *(ldb ? ldb : LIBXSMM_LD(k, n));
    const libxsmm_blasint ildc = *(ldc ? ldc : LIBXSMM_LD(m, n));
#if !defined(NDEBUG) && (0 == LIBXSMM_NO_BLAS)
    double* d = 0;
#endif
    LIBXSMM_GEMM_DECLARE_FLAGS(flags, transa, transb);
#if !defined(NDEBUG) && (0 == LIBXSMM_NO_BLAS)
    { const char *const check = getenv("LIBXSMM_CHECK");
      d = (double*)((0 == LIBXSMM_GEMM_NO_BYPASS(flags, ralpha, rbeta)
          || 0 == check || 0 == *check || 0 == check[0]) ? 0
        : malloc((*m) * (*n) * sizeof(double)));
      if (0 != d) {
        libxsmm_matcopy(d, c, sizeof(double), *m, *n, ildc, *m, 0/*prefetch*/);
      }
    }
#endif
    assert((0 < tm || 0 == *m) && (0 < tn || 0 == *n) && (0 < tk || 0 == *k) && 0 < libxsmm_nt);
#if defined(LIBXSMM_EXT_TASKS) /* implies _OPENMP */
    if (0 == omp_get_active_level())
#endif
    {
      LIBXSMM_TILED_XGEMM(
        LIBXSMM_EXT_PARALLEL, LIBXSMM_EXT_FOR_LOOP, LIBXSMM_EXT_FOR_KERNEL, LIBXSMM_NOOP,
        LIBXSMM_EXT_MIN_NTASKS, LIBXSMM_EXT_OVERHEAD, libxsmm_nt,
        double, flags, tm, tn, tk, *m, *n, *k,
        ralpha, a, ilda, b, ildb, rbeta, c, ildc);
    }
#if defined(LIBXSMM_EXT_TASKS)
    else {
      LIBXSMM_TILED_XGEMM(
        LIBXSMM_NOOP, LIBXSMM_NOOP_ARGS, LIBXSMM_EXT_TSK_KERNEL_ARGS,
        if (0 != libxsmm_sync) { LIBXSMM_EXT_TSK_SYNC } /* allow to omit synchronization */,
        LIBXSMM_EXT_MIN_NTASKS, LIBXSMM_EXT_OVERHEAD, libxsmm_nt,
        double, flags, tm, tn, tk, *m, *n, *k,
        ralpha, a, ilda, b, ildb, rbeta, c, ildc);
    }
#endif
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
  else if (0 < size) { /* small problem size */
    libxsmm_dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  }
}


#if defined(LIBXSMM_BUILD)

/* implementation provided for Fortran 77 compatibility */
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_sgemm_omp)(const char*, const char*,
  const libxsmm_blasint*, const libxsmm_blasint*, const libxsmm_blasint*,
  const float*, const float*, const libxsmm_blasint*,
  const float*, const libxsmm_blasint*,
  const float*, float*, const libxsmm_blasint*);
LIBXSMM_API_DEFINITION void LIBXSMM_FSYMBOL(libxsmm_sgemm_omp)(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const float* alpha, const float* a, const libxsmm_blasint* lda,
  const float* b, const libxsmm_blasint* ldb,
  const float* beta, float* c, const libxsmm_blasint* ldc)
{
  if (0 == (libxsmm_gemm_wrap % 2) || 0 > libxsmm_gemm_wrap) { /* parallelized/tiled */
    libxsmm_sgemm_omp(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  }
  else { /* small problem size */
    libxsmm_sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  }
}


/* implementation provided for Fortran 77 compatibility */
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_dgemm_omp)(const char*, const char*,
  const libxsmm_blasint*, const libxsmm_blasint*, const libxsmm_blasint*,
  const double*, const double*, const libxsmm_blasint*,
  const double*, const libxsmm_blasint*,
  const double*, double*, const libxsmm_blasint*);
LIBXSMM_API_DEFINITION void LIBXSMM_FSYMBOL(libxsmm_dgemm_omp)(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const double* alpha, const double* a, const libxsmm_blasint* lda,
  const double* b, const libxsmm_blasint* ldb,
  const double* beta, double* c, const libxsmm_blasint* ldc)
{
  if (0 == (libxsmm_gemm_wrap % 2) || 0 > libxsmm_gemm_wrap) { /* parallelized/tiled */
    libxsmm_dgemm_omp(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  }
  else { /* small problem size */
    libxsmm_dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  }
}

#endif /*defined(LIBXSMM_BUILD)*/

