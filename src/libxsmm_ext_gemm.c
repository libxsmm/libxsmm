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
#include <libxsmm.h>
#include "libxsmm_gemm.h"
#include "libxsmm_ext.h"

#if !defined(LIBXSMM_GEMM_EXT_MMBATCH) && defined(LIBXSMM_BUILD) && defined(LIBXSMM_BUILD_EXT) && \
    (defined(LIBXSMM_GEMM_WRAP_STATIC) || defined(LIBXSMM_GEMM_WRAP_DYNAMIC))
# define LIBXSMM_GEMM_EXT_MMBATCH
# include "libxsmm_gemm_diff.h"
#endif

#if !defined(LIBXSMM_GEMM_EXT_MMBATCH_PREFETCH)
# define LIBXSMM_GEMM_EXT_MMBATCH_PREFETCH LIBXSMM_PREFETCH_AUTO
#endif


#if defined(LIBXSMM_BUILD) && defined(LIBXSMM_BUILD_EXT)

LIBXSMM_API_VARIABLE libxsmm_gemm_descriptor internal_ext_gemm_batchdesc;
LIBXSMM_API_VARIABLE unsigned int internal_ext_gemm_batchsize;


#if defined(LIBXSMM_GEMM_WRAP_STATIC) || defined(LIBXSMM_GEMM_WRAP_DYNAMIC)
LIBXSMM_API_DEFINITION LIBXSMM_GEMM_WEAK
libxsmm_sgemm_function libxsmm_original_sgemm(const void* caller)
{
  static libxsmm_sgemm_function original = 0;
  LIBXSMM_GEMM_WRAPPER(float, original, caller);
  assert(0 != original);
  return original;
}


LIBXSMM_API_DEFINITION LIBXSMM_GEMM_WEAK
libxsmm_dgemm_function libxsmm_original_dgemm(const void* caller)
{
  static libxsmm_dgemm_function original = 0;
  LIBXSMM_GEMM_WRAPPER(double, original, caller);
  assert(0 != original);
  return original;
}
#endif


LIBXSMM_API_INLINE int internal_mmbatch_flush(void)
{
  int result = EXIT_SUCCESS;
  const unsigned int batchsize = internal_ext_gemm_batchsize; /* snapshot */
  if (0 != batchsize) { /* recorded/lazy multiplications */
    if (0 == (LIBXSMM_MMBATCH_FLAG_STATISTIC & internal_ext_gemm_batchdesc.flags)) {
      const unsigned int itemsize = sizeof(libxsmm_gemm_batchitem);
      result = libxsmm_mmbatch_omp(&internal_ext_gemm_batchdesc,
        &libxsmm_gemm_batcharray->value.a, &libxsmm_gemm_batcharray->value.b, &libxsmm_gemm_batcharray->value.c,
        0/*index_base*/, 0/*index_stride*/, &itemsize, &itemsize, &itemsize, batchsize);
    }
    else { /* print and clear statistic */
      unsigned int i;
      LIBXSMM_FLOCK(stdout);
      fprintf(stdout, "\nLIBXSMM STATISTIC\n");
      for (i = 0; i < batchsize; ++i) {
        const libxsmm_blasint lda = libxsmm_gemm_batcharray[i].stat.desc.lda;
        const libxsmm_blasint ldb = libxsmm_gemm_batcharray[i].stat.desc.ldb;
        const libxsmm_blasint ldc = libxsmm_gemm_batcharray[i].stat.desc.ldc;
        const libxsmm_blasint m = libxsmm_gemm_batcharray[i].stat.desc.m;
        const libxsmm_blasint n = libxsmm_gemm_batcharray[i].stat.desc.n;
        const libxsmm_blasint k = libxsmm_gemm_batcharray[i].stat.desc.k;
        const unsigned int ci = libxsmm_gemm_batcharray[i].stat.count;
        if (LIBXSMM_GEMM_PRECISION_F64 == libxsmm_gemm_batcharray[i].stat.desc.datatype) {
          const double alpha = libxsmm_gemm_batcharray[i].stat.desc.alpha;
          const double beta = libxsmm_gemm_batcharray[i].stat.desc.beta;
          LIBXSMM_GEMM_PRINT(stdout,
            LIBXSMM_GEMM_PRECISION_F64, libxsmm_gemm_batcharray[i].stat.desc.flags,
            &m, &n, &k, &alpha, 0/*a*/, &lda, 0/*b*/, &ldb, &beta, 0/*c*/, &ldc);
        }
        else if (LIBXSMM_GEMM_PRECISION_F32 == libxsmm_gemm_batcharray[i].stat.desc.datatype) {
          const float alpha = libxsmm_gemm_batcharray[i].stat.desc.alpha;
          const float beta = libxsmm_gemm_batcharray[i].stat.desc.beta;
          LIBXSMM_GEMM_PRINT(stdout,
            LIBXSMM_GEMM_PRECISION_F32, libxsmm_gemm_batcharray[i].stat.desc.flags,
            &m, &n, &k, &alpha, 0/*a*/, &lda, 0/*b*/, &ldb, &beta, 0/*c*/, &ldc);
        }
        else {
          result = EXIT_FAILURE;
        }
        fprintf(stdout, ": %.0f%%\n", 100.0 * ci / LIBXSMM_MAX(ci, internal_ext_gemm_batchsize));
      }
      LIBXSMM_FUNLOCK(stdout);
    }
    LIBXSMM_ATOMIC_STORE_ZERO(&internal_ext_gemm_batchsize, LIBXSMM_ATOMIC_RELAXED);
  }
  return result;
}


LIBXSMM_API_DEFINITION void LIBXSMM_FSYMBOL(__wrap_sgemm)(
  const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const float* alpha, const float* a, const libxsmm_blasint* lda,
  const float* b, const libxsmm_blasint* ldb,
  const float* beta, float* c, const libxsmm_blasint* ldc)
{
  assert(0 != lda && 0 != ldb && 0 != ldc && 0 != m && 0 != n && 0 != k);
  assert(0 != transa && 0 != transb && 0 != alpha && 0 != beta);
  {
#if defined(LIBXSMM_GEMM_EXT_MMBATCH)
    unsigned int i = libxsmm_gemm_batchsize + 1; /* no flush */
    int flags = -1;
    if (0 == libxsmm_gemm_batcharray
      || LIBXSMM_GEMM_PRECISION_F32 != internal_ext_gemm_batchdesc.datatype
      || ((unsigned int)*lda) != internal_ext_gemm_batchdesc.lda
      || ((unsigned int)*ldb) != internal_ext_gemm_batchdesc.ldb
      || ((unsigned int)*ldc) != internal_ext_gemm_batchdesc.ldc
      || ((unsigned int)*m) != internal_ext_gemm_batchdesc.m
      || ((unsigned int)*n) != internal_ext_gemm_batchdesc.n
      || ((unsigned int)*k) != internal_ext_gemm_batchdesc.k
      || (0 > (flags = LIBXSMM_GEMM_FLAGS(*transa, *transb))) /* false */
      || flags != (flags & internal_ext_gemm_batchdesc.flags)
      || 0 == LIBXSMM_FEQ(*alpha, internal_ext_gemm_batchdesc.alpha)
      || 0 == LIBXSMM_FEQ(*beta, internal_ext_gemm_batchdesc.beta))
#endif
    {
      if (0 == (libxsmm_gemm_wrap % 2) || 0 > libxsmm_gemm_wrap) { /* parallelized/tiled */
        libxsmm_sgemm_omp(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
      }
      else { /* small problem size */
        libxsmm_sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
      }
#if defined(LIBXSMM_GEMM_EXT_MMBATCH)
      if (0 != (LIBXSMM_MMBATCH_FLAG_STATISTIC & internal_ext_gemm_batchdesc.flags)) {
        libxsmm_gemm_descriptor descriptor;
        if (EXIT_SUCCESS == libxsmm_sgemm_descriptor_init(&descriptor,
          *m, *n, *k, *lda, *ldb, *ldc, *alpha, *beta,
          LIBXSMM_GEMM_FLAGS(*transa, *transb),
          LIBXSMM_GEMM_EXT_MMBATCH_PREFETCH))
        {
          unsigned int j = libxsmm_gemm_diffn_sw(&descriptor, libxsmm_gemm_batcharray,
            0/*begin*/, internal_ext_gemm_batchsize, sizeof(*libxsmm_gemm_batcharray));
          if (j < internal_ext_gemm_batchsize) {
            LIBXSMM_ATOMIC_ADD_FETCH(&libxsmm_gemm_batcharray[j].stat.count, 1, LIBXSMM_ATOMIC_RELAXED);
          }
          else {
            i = LIBXSMM_ATOMIC_ADD_FETCH(&internal_ext_gemm_batchsize, 1, LIBXSMM_ATOMIC_RELAXED);
            if (i <= libxsmm_gemm_batchsize) { /* bounds check */
              libxsmm_gemm_batcharray[i-1].stat.desc = descriptor;
              libxsmm_gemm_batcharray[i-1].stat.count = 1;
            }
          }
        }
      }
#endif
    }
#if defined(LIBXSMM_GEMM_EXT_MMBATCH)
    else {
      i = LIBXSMM_ATOMIC_ADD_FETCH(&internal_ext_gemm_batchsize, 1, LIBXSMM_ATOMIC_RELAXED);
      if (i <= libxsmm_gemm_batchsize) { /* bounds check */
        libxsmm_gemm_batcharray[i-1].value.a = a;
        libxsmm_gemm_batcharray[i-1].value.b = b;
        libxsmm_gemm_batcharray[i-1].value.c = c;
      }
      assert(0 <= flags);
    }
    if (i == libxsmm_gemm_batchsize) { /* flush */
      internal_mmbatch_flush();
    }
#endif
  }
}


LIBXSMM_API_DEFINITION void LIBXSMM_FSYMBOL(__wrap_dgemm)(
  const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const double* alpha, const double* a, const libxsmm_blasint* lda,
  const double* b, const libxsmm_blasint* ldb,
  const double* beta, double* c, const libxsmm_blasint* ldc)
{
  assert(0 != lda && 0 != ldb && 0 != ldc && 0 != m && 0 != n && 0 != k);
  assert(0 != transa && 0 != transb && 0 != alpha && 0 != beta);
  {
#if defined(LIBXSMM_GEMM_EXT_MMBATCH)
    unsigned int i = libxsmm_gemm_batchsize + 1; /* no flush */
    int flags = -1;
    if (0 == libxsmm_gemm_batcharray
      || LIBXSMM_GEMM_PRECISION_F64 != internal_ext_gemm_batchdesc.datatype
      || ((unsigned int)*lda) != internal_ext_gemm_batchdesc.lda
      || ((unsigned int)*ldb) != internal_ext_gemm_batchdesc.ldb
      || ((unsigned int)*ldc) != internal_ext_gemm_batchdesc.ldc
      || ((unsigned int)*m) != internal_ext_gemm_batchdesc.m
      || ((unsigned int)*n) != internal_ext_gemm_batchdesc.n
      || ((unsigned int)*k) != internal_ext_gemm_batchdesc.k
      || (0 > (flags = LIBXSMM_GEMM_FLAGS(*transa, *transb))) /* false */
      || flags != (flags & internal_ext_gemm_batchdesc.flags)
      || 0 == LIBXSMM_FEQ(*alpha, internal_ext_gemm_batchdesc.alpha)
      || 0 == LIBXSMM_FEQ(*beta, internal_ext_gemm_batchdesc.beta))
#endif
    {
      if (0 == (libxsmm_gemm_wrap % 2) || 0 > libxsmm_gemm_wrap) { /* parallelized/tiled */
        libxsmm_dgemm_omp(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
      }
      else { /* small problem size */
        libxsmm_dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
      }
#if defined(LIBXSMM_GEMM_EXT_MMBATCH)
      if (0 != (LIBXSMM_MMBATCH_FLAG_STATISTIC & internal_ext_gemm_batchdesc.flags)) {
        libxsmm_gemm_descriptor descriptor;
        if (EXIT_SUCCESS == libxsmm_dgemm_descriptor_init(&descriptor,
          *m, *n, *k, *lda, *ldb, *ldc, *alpha, *beta,
          LIBXSMM_GEMM_FLAGS(*transa, *transb),
          LIBXSMM_GEMM_EXT_MMBATCH_PREFETCH))
        {
          unsigned int j = libxsmm_gemm_diffn_sw(&descriptor, libxsmm_gemm_batcharray,
            0/*begin*/, internal_ext_gemm_batchsize, sizeof(*libxsmm_gemm_batcharray));
          if (j < internal_ext_gemm_batchsize) {
            LIBXSMM_ATOMIC_ADD_FETCH(&libxsmm_gemm_batcharray[j].stat.count, 1, LIBXSMM_ATOMIC_RELAXED);
          }
          else {
            i = LIBXSMM_ATOMIC_ADD_FETCH(&internal_ext_gemm_batchsize, 1, LIBXSMM_ATOMIC_RELAXED);
            if (i <= libxsmm_gemm_batchsize) { /* bounds check */
              libxsmm_gemm_batcharray[i-1].stat.desc = descriptor;
              libxsmm_gemm_batcharray[i-1].stat.count = 1;
            }
          }
        }
      }
#endif
    }
#if defined(LIBXSMM_GEMM_EXT_MMBATCH)
    else {
      i = LIBXSMM_ATOMIC_ADD_FETCH(&internal_ext_gemm_batchsize, 1, LIBXSMM_ATOMIC_RELAXED);
      if (i <= libxsmm_gemm_batchsize) { /* bounds check */
        libxsmm_gemm_batcharray[i-1].value.a = a;
        libxsmm_gemm_batcharray[i-1].value.b = b;
        libxsmm_gemm_batcharray[i-1].value.c = c;
      }
      assert(0 <= flags);
    }
    if (i == libxsmm_gemm_batchsize) { /* flush */
      internal_mmbatch_flush();
    }
#endif
  }
}

#endif /*defined(LIBXSMM_BUILD) && defined(LIBXSMM_BUILD_EXT)*/


LIBXSMM_API_DEFINITION void libxsmm_sgemm_omp(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const float* alpha, const float* a, const libxsmm_blasint* lda,
  const float* b, const libxsmm_blasint* ldb,
  const float* beta, float* c, const libxsmm_blasint* ldc)
{
  const libxsmm_blasint nn = *(n ? n : m), kk = *(k ? k : m);
  const unsigned long long size = 1ULL * (*m) * nn * kk;
  LIBXSMM_INIT
  if (LIBXSMM_MAX_MNK < size) {
    const int index = LIBXSMM_MIN(libxsmm_icbrt(size) >> 10, 7);
    const unsigned int tm = LIBXSMM_MIN(libxsmm_gemm_tile[1/*SP*/][0/*M*/][index], (unsigned int)*m);
    const unsigned int tn = LIBXSMM_MIN(libxsmm_gemm_tile[1/*SP*/][1/*N*/][index], (unsigned int)nn);
    const unsigned int tk = LIBXSMM_MIN(libxsmm_gemm_tile[1/*SP*/][2/*K*/][index], (unsigned int)kk);
    const float ralpha = (0 != alpha ? *alpha : ((float)LIBXSMM_ALPHA));
    const float rbeta = (0 != beta ? *beta : ((float)LIBXSMM_BETA));
    const libxsmm_blasint ilda = *(lda ? lda : m), ildb = (ldb ? *ldb : kk), ildc = *(ldc ? ldc : m);
    const int flags = LIBXSMM_GEMM_PFLAGS(transa, transb, LIBXSMM_FLAGS);
#if !defined(NDEBUG) && (0 == LIBXSMM_NO_BLAS)
    const char *const check = getenv("LIBXSMM_CHECK");
    float *const d = (float*)((0 == LIBXSMM_GEMM_NO_BYPASS(flags, ralpha, rbeta)
        || 0 == check || 0 == *check || 0 == check[0]) ? 0
      : libxsmm_aligned_scratch((*m) * nn * sizeof(float), 0/*auto-aligned*/));
    if (0 != d) {
      libxsmm_matcopy(d, c, sizeof(float), *m, nn, ildc, *m, 0/*prefetch*/);
    }
#endif
    assert((0 < tm || 0 == *m) && (0 < tn || 0 == nn) && (0 < tk || 0 == kk) && 0 < libxsmm_nt);
#if defined(LIBXSMM_EXT_TASKS) /* implies _OPENMP */
    if (0 == omp_get_active_level())
#endif
    {
      LIBXSMM_TILED_XGEMM(
        LIBXSMM_EXT_PARALLEL, LIBXSMM_EXT_FOR_LOOP, LIBXSMM_EXT_FOR_KERNEL, LIBXSMM_NOOP,
        LIBXSMM_EXT_MIN_NTASKS, LIBXSMM_EXT_OVERHEAD, libxsmm_nt,
        float, flags, tm, tn, tk, *m, nn, kk,
        ralpha, a, ilda, b, ildb, rbeta, c, ildc);
    }
#if defined(LIBXSMM_EXT_TASKS)
    else {
      LIBXSMM_TILED_XGEMM(
        LIBXSMM_NOOP, LIBXSMM_NOOP_ARGS, LIBXSMM_EXT_TSK_KERNEL_ARGS,
        if (0 != libxsmm_sync) { LIBXSMM_EXT_TSK_SYNC } /* allow to omit synchronization */,
        LIBXSMM_EXT_MIN_NTASKS, LIBXSMM_EXT_OVERHEAD, libxsmm_nt,
        float, flags, tm, tn, tk, *m, nn, kk,
        ralpha, a, ilda, b, ildb, rbeta, c, ildc);
    }
#endif
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
  const libxsmm_blasint nn = *(n ? n : m), kk = *(k ? k : m);
  const unsigned long long size = 1ULL * (*m) * nn * kk;
  LIBXSMM_INIT
  if (LIBXSMM_MAX_MNK < size) {
    const int index = LIBXSMM_MIN(libxsmm_icbrt(size) >> 10, 7);
    const unsigned int tm = LIBXSMM_MIN(libxsmm_gemm_tile[0/*DP*/][0/*M*/][index], (unsigned int)*m);
    const unsigned int tn = LIBXSMM_MIN(libxsmm_gemm_tile[0/*DP*/][1/*N*/][index], (unsigned int)nn);
    const unsigned int tk = LIBXSMM_MIN(libxsmm_gemm_tile[0/*DP*/][2/*K*/][index], (unsigned int)kk);
    const double ralpha = (0 != alpha ? *alpha : ((double)LIBXSMM_ALPHA));
    const double rbeta = (0 != beta ? *beta : ((double)LIBXSMM_BETA));
    const libxsmm_blasint ilda = *(lda ? lda : m), ildb = (ldb ? *ldb : kk), ildc = *(ldc ? ldc : m);
    const int flags = LIBXSMM_GEMM_PFLAGS(transa, transb, LIBXSMM_FLAGS);
#if !defined(NDEBUG) && (0 == LIBXSMM_NO_BLAS)
    const char *const check = getenv("LIBXSMM_CHECK");
    double *const d = (double*)((0 == LIBXSMM_GEMM_NO_BYPASS(flags, ralpha, rbeta)
        || 0 == check || 0 == *check || 0 == check[0]) ? 0
      : libxsmm_aligned_scratch((*m) * nn * sizeof(double), 0/*auto-aligned*/));
    if (0 != d) {
      libxsmm_matcopy(d, c, sizeof(double), *m, nn, ildc, *m, 0/*prefetch*/);
    }
#endif
    assert((0 < tm || 0 == *m) && (0 < tn || 0 == nn) && (0 < tk || 0 == kk) && 0 < libxsmm_nt);
#if defined(LIBXSMM_EXT_TASKS) /* implies _OPENMP */
    if (0 == omp_get_active_level())
#endif
    {
      LIBXSMM_TILED_XGEMM(
        LIBXSMM_EXT_PARALLEL, LIBXSMM_EXT_FOR_LOOP, LIBXSMM_EXT_FOR_KERNEL, LIBXSMM_NOOP,
        LIBXSMM_EXT_MIN_NTASKS, LIBXSMM_EXT_OVERHEAD, libxsmm_nt,
        double, flags, tm, tn, tk, *m, nn, kk,
        ralpha, a, ilda, b, ildb, rbeta, c, ildc);
    }
#if defined(LIBXSMM_EXT_TASKS)
    else {
      LIBXSMM_TILED_XGEMM(
        LIBXSMM_NOOP, LIBXSMM_NOOP_ARGS, LIBXSMM_EXT_TSK_KERNEL_ARGS,
        if (0 != libxsmm_sync) { LIBXSMM_EXT_TSK_SYNC } /* allow to omit synchronization */,
        LIBXSMM_EXT_MIN_NTASKS, LIBXSMM_EXT_OVERHEAD, libxsmm_nt,
        double, flags, tm, tn, tk, *m, nn, kk,
        ralpha, a, ilda, b, ildb, rbeta, c, ildc);
    }
#endif
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
  else if (0 < size) { /* small problem size */
    libxsmm_dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  }
}


LIBXSMM_API_DEFINITION int libxsmm_mmbatch_omp(const libxsmm_gemm_descriptor* descriptor, const void* a_matrix, const void* b_matrix, void* c_matrix,
  int index_base, int index_stride, const unsigned int a_stride[], const unsigned int b_stride[], const unsigned int c_stride[], unsigned int batchsize)
{
  int result;
#if defined(_OPENMP)
  if (0 != descriptor && 0 == (LIBXSMM_MMBATCH_FLAG_SEQUENTIAL & descriptor->flags)
    /* general check if parallelization should be used */
    && ((unsigned int)omp_get_max_threads()) < batchsize)
  {
    const unsigned int typesize = libxsmm_gemm_typesize((libxsmm_gemm_precision)descriptor->datatype);
    const libxsmm_xmmfunction kernel = libxsmm_xmmdispatch(descriptor);
# if defined(LIBXSMM_EXT_TASKS)
    if (0 == omp_get_active_level())
# endif
    { /* enable internal parallelization */
#     pragma omp parallel
      {
        const int tid = omp_get_thread_num(), nthreads = omp_get_num_threads();
        libxsmm_mmbatch_internal(kernel, typesize, a_matrix, b_matrix, c_matrix,
          index_base, index_stride, a_stride, b_stride, c_stride, batchsize,
          tid, nthreads, descriptor);
      } /* implicit synchronization (barrier) */
    }
# if defined(LIBXSMM_EXT_TASKS)
    else { /* assume external parallelization, and use OpenMP-tasks */
      const int ntasks = (LIBXSMM_EXT_TSK_SLACK) * omp_get_num_threads();
      int tid;
      for (tid = 0; tid < ntasks; ++tid) {
#       pragma omp task
        libxsmm_mmbatch_internal(kernel, typesize, a_matrix, b_matrix, c_matrix,
          index_base, index_stride, a_stride, b_stride, c_stride, batchsize,
          tid, ntasks, descriptor);
      }
      /* allow to omit synchronization */
      if (0 != libxsmm_sync) {
#       pragma omp taskwait
      }
    }
# endif
    result = EXIT_SUCCESS;
  }
  else
#endif /*defined(_OPENMP)*/
  { /* sequential */
    result = libxsmm_mmbatch(descriptor, a_matrix, b_matrix, c_matrix,
      index_base, index_stride, a_stride, b_stride, c_stride, batchsize);
  }
  return result;
}


LIBXSMM_API_DEFINITION void libxsmm_mmbatch_begin(libxsmm_gemm_precision precision, const int* flags,
  const int* m, const int* n, const int* k, const int* lda, const int* ldb, const int* ldc,
  const void* alpha, const void* beta)
{
#if defined(LIBXSMM_GEMM_EXT_MMBATCH)
  LIBXSMM_INIT
  if (0 != libxsmm_gemm_batcharray /* batch-recording available, but not yet running */
    /* currently, batch recording is only enabled if all values are present (no complex filtering) */
    && 0 != flags && 0 != alpha && 0 != beta
    && 0 != lda && 0 != ldb && 0 != ldc
    && 0 != m && 0 != n && 0 != k)
  {
    static int error_once = 0;
    libxsmm_gemm_descriptor descriptor;
    const int prefetch = LIBXSMM_GEMM_EXT_MMBATCH_PREFETCH;
    int result = libxsmm_gemm_descriptor_init(&descriptor,
      precision, *m, *n, *k, lda, ldb, ldc, alpha, beta, flags, &prefetch);

    if (EXIT_SUCCESS == result) {
      result = internal_mmbatch_flush();
      if (EXIT_SUCCESS == result) {
        if (0 == (LIBXSMM_MMBATCH_FLAG_STATISTIC & *flags)) {
          internal_ext_gemm_batchdesc = descriptor;
        }
        else {
          memset(&internal_ext_gemm_batchdesc, 0, sizeof(internal_ext_gemm_batchdesc));
          internal_ext_gemm_batchdesc.flags = LIBXSMM_MMBATCH_FLAG_STATISTIC;
        }
      }
    }
    else if (0 != libxsmm_verbosity /* library code is expected to be mute */
       && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
    {
      fprintf(stderr, "LIBXSMM ERROR: GEMM batch recording failed to enable!\n");
    }
  }
#else
  LIBXSMM_UNUSED(precision); LIBXSMM_UNUSED(flags);
  LIBXSMM_UNUSED(m); LIBXSMM_UNUSED(n); LIBXSMM_UNUSED(k);
  LIBXSMM_UNUSED(lda); LIBXSMM_UNUSED(ldb); LIBXSMM_UNUSED(ldc);
  LIBXSMM_UNUSED(alpha); LIBXSMM_UNUSED(beta);
#endif
}


LIBXSMM_API_DEFINITION int libxsmm_mmbatch_end(void)
{
#if defined(LIBXSMM_GEMM_EXT_MMBATCH)
  const int result = internal_mmbatch_flush();
  memset(&internal_ext_gemm_batchdesc, 0, sizeof(internal_ext_gemm_batchdesc));
#else
  const int result = EXIT_SUCCESS;
#endif
  return result;
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
  libxsmm_sgemm_omp(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
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
  libxsmm_dgemm_omp(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

#endif /*defined(LIBXSMM_BUILD)*/

