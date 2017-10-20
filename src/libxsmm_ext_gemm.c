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
    (defined(LIBXSMM_GEMM_WRAP_STATIC) || defined(LIBXSMM_GEMM_WRAP_DYNAMIC) || \
    !defined(NDEBUG) || defined(_WIN32)) /* debug purpose */
# define LIBXSMM_GEMM_EXT_MMBATCH
# include "libxsmm_gemm_diff.h"
# include "libxsmm_trace.h"
#endif

#if defined(LIBXSMM_GEMM_EXT_MMBATCH)
# if !defined(LIBXSMM_GEMM_EXT_MMBATCH_PREFETCH)
#   define LIBXSMM_GEMM_EXT_MMBATCH_PREFETCH LIBXSMM_PREFETCH_AUTO
# endif
# if !defined(LIBXSMM_GEMM_EXT_MMBATCH_MAXDEPTH)
#   define LIBXSMM_GEMM_EXT_MMBATCH_MAXDEPTH 8/*POT*/
# endif
LIBXSMM_API_VARIABLE libxsmm_gemm_descriptor internal_ext_gemm_batchdesc[LIBXSMM_GEMM_EXT_MMBATCH_MAXDEPTH];
LIBXSMM_API_VARIABLE unsigned int internal_ext_gemm_batchdepth;
LIBXSMM_API_VARIABLE unsigned int internal_ext_gemm_batchsize;
#endif


#if defined(LIBXSMM_BUILD) && defined(LIBXSMM_BUILD_EXT)

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


#if defined(LIBXSMM_GEMM_EXT_MMBATCH)
LIBXSMM_API_INLINE int internal_mmbatch_sortrev(const void* stat_a, const void* stat_b)
{
  const libxsmm_gemm_batchitem *const a = (const libxsmm_gemm_batchitem*)stat_a;
  const libxsmm_gemm_batchitem *const b = (const libxsmm_gemm_batchitem*)stat_b;
  assert(0 != stat_a && 0 != stat_b);
  return a->stat.count < b->stat.count ? 1 : (b->stat.count < a->stat.count ? -1 : 0);
}
#endif /*defined(LIBXSMM_GEMM_EXT_MMBATCH)*/


LIBXSMM_API_INLINE int internal_mmbatch_flush(const libxsmm_gemm_descriptor* batchdesc,
  unsigned int batchsize, libxsmm_gemm_batchitem* batcharray)
{
  int result = EXIT_SUCCESS;
#if defined(LIBXSMM_GEMM_EXT_MMBATCH)
  if (0 != batchsize) { /* recorded/lazy multiplications */
    const unsigned int itemsize = sizeof(libxsmm_gemm_batchitem);
    assert(0 != batchdesc);
    if (0 == (LIBXSMM_MMBATCH_FLAG_STATISTIC & batchdesc->flags)) { /* process batch */
      result = libxsmm_mmbatch_omp(batchdesc, &batcharray->value.a, &batcharray->value.b, &batcharray->value.c,
        0/*index_base*/, 0/*index_stride*/, &itemsize, &itemsize, &itemsize, batchsize);
      memset(batcharray, 0, batchsize * itemsize); /* clear */
    }
    else { /* print statistic */
      const unsigned limit = ((1 < libxsmm_verbosity || 0 > libxsmm_verbosity) ? batchsize : 7);
      unsigned int i, threshold, batchcount, count = 0;
      assert(0 != batcharray);
      qsort(batcharray, batchsize, itemsize, internal_mmbatch_sortrev);
      batchcount = batcharray[0].stat.count;
      threshold = ((1 < libxsmm_verbosity || 0 > libxsmm_verbosity || 3 >= batchsize) ? 0 : (batchcount / 2));
      for (i = 1; i < batchsize; ++i) batchcount += batcharray[i].stat.count;
      LIBXSMM_FLOCK(stdout);
      for (i = 0; i < batchsize; ++i) {
        const libxsmm_gemm_descriptor descriptor = batcharray[i].stat.desc;
        const libxsmm_blasint lda = descriptor.lda, ldb = descriptor.ldb, ldc = descriptor.ldc;
        const libxsmm_blasint m = descriptor.m, n = descriptor.n, k = descriptor.k;
        const char *const symbol = batcharray[i].stat.symbol;
        const unsigned int ci = batcharray[i].stat.count;

        memset(batcharray + i, 0, itemsize); /* clear */
        if (threshold < ci && count < limit /* limit printed statistic */
          && 0 < m && 0 < n && 0 < k && m <= lda && k <= ldb && m <= ldc)
        {
          assert(0 != ci);
          if (0 == count) {
            fprintf(stdout, "\nLIBXSMM STATISTIC: %u multiplication%c\n", batchcount, 1 < batchcount ? 's' : ' ');
          }
          if (LIBXSMM_GEMM_PRECISION_F64 == descriptor.datatype) {
            const double alpha = descriptor.alpha, beta = descriptor.beta;
            LIBXSMM_GEMM_PRINT(stdout,
              LIBXSMM_GEMM_PRECISION_F64, descriptor.flags,
              &m, &n, &k, &alpha, 0/*a*/, &lda, 0/*b*/, &ldb, &beta, 0/*c*/, &ldc);
          }
          else if (LIBXSMM_GEMM_PRECISION_F32 == descriptor.datatype) {
            const float alpha = descriptor.alpha, beta = descriptor.beta;
            LIBXSMM_GEMM_PRINT(stdout,
              LIBXSMM_GEMM_PRECISION_F32, descriptor.flags,
              &m, &n, &k, &alpha, 0/*a*/, &lda, 0/*b*/, &ldb, &beta, 0/*c*/, &ldc);
          }
          else {
            result = EXIT_FAILURE;
          }
          if (0 != symbol) {
            fprintf(stdout, ": %.0f%% [%s]\n", 100.0 * ci / batchcount, symbol);
          }
          else {
            fprintf(stdout, ": %.0f%%\n", 100.0 * ci / batchcount);
          }
          ++count;
        }
      }
      LIBXSMM_FUNLOCK(stdout);
    }
  }
#else
  LIBXSMM_UNUSED(batchdesc); LIBXSMM_UNUSED(batchsize); LIBXSMM_UNUSED(batcharray);
#endif
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
    unsigned int i = 0; /* no flush */
    int flags = -1;
# if !defined(NDEBUG)
    static int error_once = 0;
    int result = EXIT_SUCCESS;
# endif
    LIBXSMM_INIT
    if (0 == libxsmm_gemm_batcharray
      || LIBXSMM_GEMM_PRECISION_F32 != libxsmm_gemm_batchdesc.datatype
      || ((unsigned int)*lda) != libxsmm_gemm_batchdesc.lda
      || ((unsigned int)*ldb) != libxsmm_gemm_batchdesc.ldb
      || ((unsigned int)*ldc) != libxsmm_gemm_batchdesc.ldc
      || ((unsigned int)*m) != libxsmm_gemm_batchdesc.m
      || ((unsigned int)*n) != libxsmm_gemm_batchdesc.n
      || ((unsigned int)*k) != libxsmm_gemm_batchdesc.k
      || (0 > (flags = LIBXSMM_GEMM_FLAGS(*transa, *transb))) /* false */
      || flags != (flags & libxsmm_gemm_batchdesc.flags)
      || 0 == LIBXSMM_FEQ(*alpha, libxsmm_gemm_batchdesc.alpha)
      || 0 == LIBXSMM_FEQ(*beta, libxsmm_gemm_batchdesc.beta))
#endif
    {
      if (0 == (libxsmm_gemm_wrap % 2) || 0 > libxsmm_gemm_wrap) { /* parallelized/tiled */
        libxsmm_sgemm_omp(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
      }
      else { /* small problem size */
        libxsmm_sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
      }
#if defined(LIBXSMM_GEMM_EXT_MMBATCH)
      if (0 != (LIBXSMM_MMBATCH_FLAG_STATISTIC & libxsmm_gemm_batchdesc.flags)) {
        libxsmm_gemm_descriptor descriptor;

        if (EXIT_SUCCESS == libxsmm_sgemm_descriptor_init(&descriptor,
          *m, *n, *k, *lda, *ldb, *ldc, *alpha, *beta,
          LIBXSMM_GEMM_FLAGS(*transa, *transb),
          LIBXSMM_GEMM_EXT_MMBATCH_PREFETCH))
        {
          const unsigned int max_batchsize = (unsigned int)((LIBXSMM_GEMM_BATCHSCALE) * libxsmm_gemm_batchsize);
          const unsigned int max_size = (0 != internal_ext_gemm_batchsize ? (((internal_ext_gemm_batchsize - 1) % max_batchsize) + 1) : 0);
          libxsmm_gemm_batchitem* batcharray = libxsmm_gemm_batcharray;
          unsigned int size = max_size;
          if (libxsmm_gemm_batchsize < max_size) {
            size = max_size - libxsmm_gemm_batchsize;
            batcharray += libxsmm_gemm_batchsize;
          }
          i = libxsmm_gemm_diffn_sw(&descriptor, batcharray, 0/*hint*/, size, sizeof(libxsmm_gemm_batchitem));

          if (i < size) { /* update existing entry */
            LIBXSMM_ATOMIC_ADD_FETCH(&batcharray[i].stat.count, 1, LIBXSMM_ATOMIC_RELAXED);
          }
          else { /* new entry needed */
            const int maxnsyms = -1;
# if defined(NDEBUG)
            unsigned int depth = 1;
# else
            unsigned int depth = 2;
# endif
            void* extra = 0;
            i = ((LIBXSMM_ATOMIC_ADD_FETCH(&internal_ext_gemm_batchsize, 1, LIBXSMM_ATOMIC_RELAXED) - 1) % max_batchsize) + 1;
            libxsmm_gemm_batcharray[i-1].stat.desc = descriptor;
            libxsmm_gemm_batcharray[i-1].stat.count = 1;
            libxsmm_gemm_batcharray[i-1].stat.symbol = libxsmm_trace_info(&depth, 0, 0, 0, &maxnsyms);
            if (EXIT_SUCCESS == libxsmm_get_malloc_xinfo(libxsmm_gemm_batcharray, 0/*size*/, 0/*flags*/, &extra)) {
              *(libxsmm_mmbatch_flush_function*)extra = libxsmm_mmbatch_end;
            }
# if !defined(NDEBUG)
            else {
              result = EXIT_FAILURE;
            }
# endif
          }
        }
      }
#endif
    }
#if defined(LIBXSMM_GEMM_EXT_MMBATCH)
    else {
      const unsigned int max_batchsize = (unsigned int)((LIBXSMM_GEMM_BATCHSCALE) * libxsmm_gemm_batchsize);
      i = ((LIBXSMM_ATOMIC_ADD_FETCH(&internal_ext_gemm_batchsize, 1, LIBXSMM_ATOMIC_RELAXED) - 1) % max_batchsize) + 1;
      libxsmm_gemm_batcharray[i-1].value.a = a;
      libxsmm_gemm_batcharray[i-1].value.b = b;
      libxsmm_gemm_batcharray[i-1].value.c = c;
      assert(0 <= flags);
    }
    if (libxsmm_gemm_batchsize == (i - 1)) { /* condition ensure to flush once (first discovery) */
# if !defined(NDEBUG)
      result =
# endif
      internal_mmbatch_flush(&libxsmm_gemm_batchdesc, libxsmm_gemm_batchsize, libxsmm_gemm_batcharray);
    }
# if !defined(NDEBUG) /* library code is expected to be mute */
    if (EXIT_SUCCESS != result && 0 != libxsmm_verbosity &&
      1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
    {
      fprintf(stderr, "LIBXSMM ERROR: SGEMM batch recording failed!\n");
    }
# endif
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
    unsigned int i = 0; /* no flush */
    int flags = -1;
# if !defined(NDEBUG)
    static int error_once = 0;
    int result = EXIT_SUCCESS;
# endif
    LIBXSMM_INIT
    if (0 == libxsmm_gemm_batcharray
      || LIBXSMM_GEMM_PRECISION_F64 != libxsmm_gemm_batchdesc.datatype
      || ((unsigned int)*lda) != libxsmm_gemm_batchdesc.lda
      || ((unsigned int)*ldb) != libxsmm_gemm_batchdesc.ldb
      || ((unsigned int)*ldc) != libxsmm_gemm_batchdesc.ldc
      || ((unsigned int)*m) != libxsmm_gemm_batchdesc.m
      || ((unsigned int)*n) != libxsmm_gemm_batchdesc.n
      || ((unsigned int)*k) != libxsmm_gemm_batchdesc.k
      || (0 > (flags = LIBXSMM_GEMM_FLAGS(*transa, *transb))) /* false */
      || flags != (flags & libxsmm_gemm_batchdesc.flags)
      || 0 == LIBXSMM_FEQ(*alpha, libxsmm_gemm_batchdesc.alpha)
      || 0 == LIBXSMM_FEQ(*beta, libxsmm_gemm_batchdesc.beta))
#endif
    {
      if (0 == (libxsmm_gemm_wrap % 2) || 0 > libxsmm_gemm_wrap) { /* parallelized/tiled */
        libxsmm_dgemm_omp(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
      }
      else { /* small problem size */
        libxsmm_dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
      }
#if defined(LIBXSMM_GEMM_EXT_MMBATCH)
      if (0 != (LIBXSMM_MMBATCH_FLAG_STATISTIC & libxsmm_gemm_batchdesc.flags)) {
        libxsmm_gemm_descriptor descriptor;

        if (EXIT_SUCCESS == libxsmm_dgemm_descriptor_init(&descriptor,
          *m, *n, *k, *lda, *ldb, *ldc, *alpha, *beta,
          LIBXSMM_GEMM_FLAGS(*transa, *transb),
          LIBXSMM_GEMM_EXT_MMBATCH_PREFETCH))
        {
          const unsigned int max_batchsize = (unsigned int)((LIBXSMM_GEMM_BATCHSCALE) * libxsmm_gemm_batchsize);
          const unsigned int max_size = (0 != internal_ext_gemm_batchsize ? (((internal_ext_gemm_batchsize - 1) % max_batchsize) + 1) : 0);
          libxsmm_gemm_batchitem* batcharray = libxsmm_gemm_batcharray;
          unsigned int size = max_size;
          if (libxsmm_gemm_batchsize < max_size) {
            size = max_size - libxsmm_gemm_batchsize;
            batcharray += libxsmm_gemm_batchsize;
          }
          i = libxsmm_gemm_diffn_sw(&descriptor, batcharray, 0/*hint*/, size, sizeof(libxsmm_gemm_batchitem));

          if (i < size) { /* update existing entry */
            LIBXSMM_ATOMIC_ADD_FETCH(&batcharray[i].stat.count, 1, LIBXSMM_ATOMIC_RELAXED);
          }
          else { /* new entry needed */
            const int maxnsyms = -1;
# if defined(NDEBUG)
            unsigned int depth = 1;
# else
            unsigned int depth = 2;
# endif
            void* extra = 0;
            i = ((LIBXSMM_ATOMIC_ADD_FETCH(&internal_ext_gemm_batchsize, 1, LIBXSMM_ATOMIC_RELAXED) - 1) % max_batchsize) + 1;
            libxsmm_gemm_batcharray[i-1].stat.desc = descriptor;
            libxsmm_gemm_batcharray[i-1].stat.count = 1;
            libxsmm_gemm_batcharray[i-1].stat.symbol = libxsmm_trace_info(&depth, 0, 0, 0, &maxnsyms);
            if (EXIT_SUCCESS == libxsmm_get_malloc_xinfo(libxsmm_gemm_batcharray, 0/*size*/, 0/*flags*/, &extra)) {
              *(libxsmm_mmbatch_flush_function*)extra = libxsmm_mmbatch_end;
            }
# if !defined(NDEBUG)
            else {
              result = EXIT_FAILURE;
            }
# endif
          }
        }
      }
#endif
    }
#if defined(LIBXSMM_GEMM_EXT_MMBATCH)
    else {
      const unsigned int max_batchsize = (unsigned int)((LIBXSMM_GEMM_BATCHSCALE) * libxsmm_gemm_batchsize);
      i = ((LIBXSMM_ATOMIC_ADD_FETCH(&internal_ext_gemm_batchsize, 1, LIBXSMM_ATOMIC_RELAXED) - 1) % max_batchsize) + 1;
      libxsmm_gemm_batcharray[i-1].value.a = a;
      libxsmm_gemm_batcharray[i-1].value.b = b;
      libxsmm_gemm_batcharray[i-1].value.c = c;
      assert(0 <= flags);
    }
    if (libxsmm_gemm_batchsize == (i - 1)) { /* condition ensure to flush once (first discovery) */
# if !defined(NDEBUG)
      result =
# endif
      internal_mmbatch_flush(&libxsmm_gemm_batchdesc, libxsmm_gemm_batchsize, libxsmm_gemm_batcharray);
    }
# if !defined(NDEBUG) /* library code is expected to be mute */
    if (EXIT_SUCCESS != result && 0 != libxsmm_verbosity &&
      1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
    {
      fprintf(stderr, "LIBXSMM ERROR: DGEMM batch recording failed!\n");
    }
# endif
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
        libxsmm_xmmbatch(kernel, typesize, a_matrix, b_matrix, c_matrix,
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
        libxsmm_xmmbatch(kernel, typesize, a_matrix, b_matrix, c_matrix,
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
    && 0 != m && 0 != n && 0 != k
    && LIBXSMM_LOCK_ACQUIRED == LIBXSMM_LOCK_TRYLOCK(&libxsmm_gemm_batchlock))
  {
    static int error_once = 0;
    const int prefetch = LIBXSMM_GEMM_EXT_MMBATCH_PREFETCH;
    libxsmm_gemm_descriptor descriptor;
    int result = libxsmm_gemm_descriptor_init(&descriptor, precision,
      *m, *n, *k, lda, ldb, ldc, alpha, beta, flags, &prefetch);

    if (EXIT_SUCCESS == result) {
      const unsigned int max_batchsize = (unsigned int)((LIBXSMM_GEMM_BATCHSCALE) * libxsmm_gemm_batchsize);
      unsigned int i;

      /* eventually overwrite the oldest entry */
      i = LIBXSMM_MOD2(internal_ext_gemm_batchdepth, LIBXSMM_GEMM_EXT_MMBATCH_MAXDEPTH);
      internal_ext_gemm_batchdesc[i] = libxsmm_gemm_batchdesc; /* backup */
      ++internal_ext_gemm_batchdepth;

      /* ensure descriptor does not match any GEMM such that... */
      memset(&libxsmm_gemm_batchdesc, 0, sizeof(libxsmm_gemm_batchdesc));
      /* ...the batch stops and completely flushes */
      if (0 != internal_ext_gemm_batchsize) {
        result = internal_mmbatch_flush(internal_ext_gemm_batchdesc + i,
          ((internal_ext_gemm_batchsize - 1) % max_batchsize) + 1, libxsmm_gemm_batcharray);
      }

      if (EXIT_SUCCESS == result) { /* enable descriptor */
        internal_ext_gemm_batchsize = 0; /* reset */
        if (0 == (LIBXSMM_MMBATCH_FLAG_STATISTIC & *flags)) {
          libxsmm_gemm_batchdesc = descriptor;
        }
        else {
          libxsmm_gemm_batchdesc.flags = LIBXSMM_MMBATCH_FLAG_STATISTIC;
        }
      }
    }
    if (EXIT_SUCCESS != result && 0 != libxsmm_verbosity /* library code is expected to be mute */
      && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
    {
      fprintf(stderr, "LIBXSMM ERROR: GEMM batch enabling failed!\n");
    }
    LIBXSMM_LOCK_RELEASE(&libxsmm_gemm_batchlock);
  }
#else
  LIBXSMM_UNUSED(precision); LIBXSMM_UNUSED(flags);
  LIBXSMM_UNUSED(m); LIBXSMM_UNUSED(n); LIBXSMM_UNUSED(k);
  LIBXSMM_UNUSED(lda); LIBXSMM_UNUSED(ldb); LIBXSMM_UNUSED(ldc);
  LIBXSMM_UNUSED(alpha); LIBXSMM_UNUSED(beta);
#endif
}


LIBXSMM_API_DEFINITION void libxsmm_mmbatch_end(void)
{
#if defined(LIBXSMM_GEMM_EXT_MMBATCH)
  if (LIBXSMM_LOCK_ACQUIRED == LIBXSMM_LOCK_TRYLOCK(&libxsmm_gemm_batchlock)) {
    const unsigned int max_batchsize = (unsigned int)((LIBXSMM_GEMM_BATCHSCALE) * libxsmm_gemm_batchsize);
    const libxsmm_gemm_descriptor flushdesc = libxsmm_gemm_batchdesc;
    static int error_once = 0;

    /* ensure descriptor does not match any GEMM such that... */
    memset(&libxsmm_gemm_batchdesc, 0, sizeof(libxsmm_gemm_batchdesc));
    /* ...the batch stops and completely flushes */
    if (EXIT_SUCCESS == internal_mmbatch_flush(&flushdesc,
      0 != internal_ext_gemm_batchsize ? (((internal_ext_gemm_batchsize - 1) % max_batchsize) + 1) : 0,
      libxsmm_gemm_batcharray))
    {
      internal_ext_gemm_batchsize = 0; /* reset */
      --internal_ext_gemm_batchdepth; /* restore the previous descriptor */
      libxsmm_gemm_batchdesc = internal_ext_gemm_batchdesc[LIBXSMM_MOD2(internal_ext_gemm_batchdepth, LIBXSMM_GEMM_EXT_MMBATCH_MAXDEPTH)];
    }
    else if (0 != libxsmm_verbosity /* library code is expected to be mute */
      && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
    {
      fprintf(stderr, "LIBXSMM ERROR: GEMM batch processing failed!\n");
    }
    LIBXSMM_LOCK_RELEASE(&libxsmm_gemm_batchlock);
  }
#endif
}


#if defined(LIBXSMM_BUILD) && defined(LIBXSMM_BUILD_EXT)

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


/* implementation provided for Fortran 77 compatibility */
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_mmbatch_begin)(const libxsmm_gemm_precision* /*precision*/, const int* /*flags*/,
  const int* /*m*/, const int* /*n*/, const int* /*k*/, const int* /*lda*/, const int* /*ldb*/, const int* /*ldc*/,
  const void* /*alpha*/, const void* /*beta*/);
LIBXSMM_API_DEFINITION void LIBXSMM_FSYMBOL(libxsmm_mmbatch_begin)(const libxsmm_gemm_precision* precision, const int* flags,
  const int* m, const int* n, const int* k, const int* lda, const int* ldb, const int* ldc,
  const void* alpha, const void* beta)
{
  assert(0 != precision);
  libxsmm_mmbatch_begin(*precision, flags, m, n, k, lda, ldb, ldc, alpha, beta);
}


/* implementation provided for Fortran 77 compatibility */
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_mmbatch_end)(void);
LIBXSMM_API_DEFINITION void LIBXSMM_FSYMBOL(libxsmm_mmbatch_end)(void)
{
  libxsmm_mmbatch_end();
}

#endif /*defined(LIBXSMM_BUILD) && defined(LIBXSMM_BUILD_EXT)*/

