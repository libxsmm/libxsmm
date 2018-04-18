/******************************************************************************
** Copyright (c) 2016-2018, Intel Corporation                                **
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

#if defined(LIBXSMM_GEMM_MMBATCH) && defined(LIBXSMM_BUILD_EXT)
# include "libxsmm_gemm_diff.h"
# include "libxsmm_trace.h"
#endif

#if defined(LIBXSMM_GEMM_MMBATCH) && defined(LIBXSMM_BUILD_EXT)
# if !defined(LIBXSMM_GEMM_EXT_MMBATCH_PREFETCH)
#   define LIBXSMM_GEMM_EXT_MMBATCH_PREFETCH libxsmm_get_gemm_prefetch(LIBXSMM_PREFETCH_AUTO)
# endif
# if !defined(LIBXSMM_GEMM_EXT_MMBATCH_MAXDEPTH)
#   define LIBXSMM_GEMM_EXT_MMBATCH_MAXDEPTH 8/*POT*/
# endif
LIBXSMM_EXTVAR(libxsmm_gemm_descriptor internal_ext_gemm_batchdesc[LIBXSMM_GEMM_EXT_MMBATCH_MAXDEPTH]);
LIBXSMM_EXTVAR(unsigned int internal_ext_gemm_batchdepth);
LIBXSMM_EXTVAR(unsigned int internal_ext_gemm_batchsize);
#endif


#if defined(LIBXSMM_BUILD) && defined(LIBXSMM_BUILD_EXT)

#if defined(LIBXSMM_GEMM_WRAP_STATIC) || defined(LIBXSMM_GEMM_WRAP_DYNAMIC)
LIBXSMM_APIEXT LIBXSMM_GEMM_WEAK
libxsmm_dgemm_function libxsmm_original_dgemm(void)
{
  static libxsmm_dgemm_function original = 0;
  LIBXSMM_GEMM_WRAPPER(double, original);
  assert(0 != original);
  return original;
}


LIBXSMM_APIEXT LIBXSMM_GEMM_WEAK
libxsmm_sgemm_function libxsmm_original_sgemm(void)
{
  static libxsmm_sgemm_function original = 0;
  LIBXSMM_GEMM_WRAPPER(float, original);
  assert(0 != original);
  return original;
}
#endif


#if defined(LIBXSMM_GEMM_MMBATCH) && defined(LIBXSMM_BUILD_EXT)
LIBXSMM_API_INLINE int internal_mmbatch_sortrev(const void* stat_a, const void* stat_b)
{
  const libxsmm_gemm_batchitem *const a = (const libxsmm_gemm_batchitem*)stat_a;
  const libxsmm_gemm_batchitem *const b = (const libxsmm_gemm_batchitem*)stat_b;
  assert(0 != stat_a && 0 != stat_b);
  return a->stat.count < b->stat.count ? 1 : (b->stat.count < a->stat.count ? -1 : 0);
}
#endif /*defined(LIBXSMM_GEMM_MMBATCH) && defined(LIBXSMM_BUILD_EXT)*/


LIBXSMM_API_INLINE int internal_mmbatch_flush(const libxsmm_gemm_descriptor* batchdesc,
  libxsmm_blasint batchsize, libxsmm_gemm_batchitem* batcharray)
{
  int result = EXIT_SUCCESS;
#if defined(LIBXSMM_GEMM_MMBATCH) && defined(LIBXSMM_BUILD_EXT)
  if (0 != batchsize) { /* recorded/lazy multiplications */
    const libxsmm_blasint itemsize = sizeof(libxsmm_gemm_batchitem);
    assert(0 != batchdesc);
    if (0 == (LIBXSMM_MMBATCH_FLAG_STATISTIC & batchdesc->flags)) { /* process batch */
      const libxsmm_xmmfunction kernel = libxsmm_xmmdispatch(batchdesc);
      if (0 != kernel.xmm) {
        if (0 == (LIBXSMM_MMBATCH_FLAG_SEQUENTIAL & batchdesc->flags)) { /* parallelized */
          const libxsmm_blasint sync = (0 == (LIBXSMM_MMBATCH_FLAG_SYNCHRONIZED & batchdesc->flags) ? 1 : -1);
          result = libxsmm_mmbatch_omp(
            kernel, 0/*index_base*/, 0/*index_stride*/, &itemsize, &itemsize, &itemsize,
            &batcharray->value.a, &batcharray->value.b, &batcharray->value.c, batchsize * sync);
        }
        else { /* sequential */
          result = libxsmm_mmbatch(
            kernel, 0/*index_base*/, 0/*index_stride*/, &itemsize, &itemsize, &itemsize,
            &batcharray->value.a, &batcharray->value.b, &batcharray->value.c, batchsize,
            0/*tid*/, 1/*nthreads*/);
        }
      }
      else { /* may happen because of try-lock (registry) */
        const char transa = (char)(0 == (LIBXSMM_GEMM_FLAG_TRANS_A & batchdesc->flags) ? 'n' : 't');
        const char transb = (char)(0 == (LIBXSMM_GEMM_FLAG_TRANS_B & batchdesc->flags) ? 'n' : 't');
        const libxsmm_blasint lda = batchdesc->lda, ldb = batchdesc->ldb, ldc = batchdesc->ldc;
        switch (batchdesc->datatype) {
          case LIBXSMM_GEMM_PRECISION_F64: {
            const double alpha = batchdesc->alpha, beta = batchdesc->beta;
            result = libxsmm_dmmbatch_blas(&transa, &transb, batchdesc->m, batchdesc->n, batchdesc->k,
              &alpha, &batcharray->value.a, &lda, &batcharray->value.b, &ldb, &beta, &batcharray->value.c, &ldc,
              0/*index_base*/, 0/*index_stride*/, &itemsize, &itemsize, &itemsize, batchsize);
          } break;
          case LIBXSMM_GEMM_PRECISION_F32: {
            const float alpha = batchdesc->alpha, beta = batchdesc->beta;
            result = libxsmm_smmbatch_blas(&transa, &transb, batchdesc->m, batchdesc->n, batchdesc->k,
              &alpha, &batcharray->value.a, &lda, &batcharray->value.b, &ldb, &beta, &batcharray->value.c, &ldc,
              0/*index_base*/, 0/*index_stride*/, &itemsize, &itemsize, &itemsize, batchsize);
          } break;
          default: result = EXIT_FAILURE;
        }
      }
      memset(batcharray, 0, (size_t)(batchsize * itemsize)); /* clear */
    }
    else { /* print statistic */
      const libxsmm_blasint limit = (3 < libxsmm_get_verbosity() ? batchsize : 7);
      unsigned int threshold, batchcount;
      libxsmm_blasint count = 0, i;
      assert(0 != batcharray);
      qsort(batcharray, (size_t)batchsize, (size_t)itemsize, internal_mmbatch_sortrev);
      batchcount = batcharray[0].stat.count;
      threshold = ((3 < libxsmm_get_verbosity() || 3 >= batchsize) ? 0 : (batchcount / 2));
      for (i = 1; i < batchsize; ++i) batchcount += batcharray[i].stat.count;
      LIBXSMM_FLOCK(stdout);
      for (i = 0; i < batchsize; ++i) {
        const libxsmm_gemm_descriptor descriptor = batcharray[i].stat.desc;
        const libxsmm_blasint lda = descriptor.lda, ldb = descriptor.ldb, ldc = descriptor.ldc;
        const libxsmm_blasint m = descriptor.m, n = descriptor.n, k = descriptor.k;
        const char *const symbol = batcharray[i].stat.symbol;
        const unsigned int ci = batcharray[i].stat.count;

        memset(batcharray + i, 0, (size_t)itemsize); /* clear */
        if (threshold < ci && count < limit /* limit printed statistic */
          && 0 < m && 0 < n && 0 < k)
        {
          assert(0 != ci);
          if (0 == count) {
            fprintf(stdout, "\nLIBXSMM STATISTIC: %u multiplication%c\n", batchcount, 1 < batchcount ? 's' : ' ');
          }
          LIBXSMM_GEMM_PRINT2(stdout, LIBXSMM_GETENUM_INP(descriptor.datatype), LIBXSMM_GETENUM_OUT(descriptor.datatype),
            descriptor.flags, m, n, k, descriptor.alpha, NULL/*a*/, lda, NULL/*b*/, ldb, descriptor.beta, NULL/*c*/, ldc);
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


LIBXSMM_APIEXT void LIBXSMM_FSYMBOL(__wrap_dgemm)(
  const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const double* alpha, const double* a, const libxsmm_blasint* lda,
  const double* b, const libxsmm_blasint* ldb,
  const double* beta, double* c, const libxsmm_blasint* ldc)
{
  assert(0 != lda && 0 != ldb && 0 != ldc && 0 != m && 0 != n && 0 != k);
  assert(0 != transa && 0 != transb && 0 != alpha && 0 != beta);
  {
#if defined(LIBXSMM_GEMM_MMBATCH) && defined(LIBXSMM_BUILD_EXT)
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
      || LIBXSMM_NEQ(*alpha, libxsmm_gemm_batchdesc.alpha)
      || LIBXSMM_NEQ(*beta, libxsmm_gemm_batchdesc.beta))
#endif
    {
      if (0 == (libxsmm_gemm_wrap % 2) || 0 > libxsmm_gemm_wrap) { /* parallelized/tiled */
        libxsmm_dgemm_omp(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
      }
      else { /* small problem size */
        libxsmm_dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
      }
#if defined(LIBXSMM_GEMM_MMBATCH) && defined(LIBXSMM_BUILD_EXT)
      if (0 != (LIBXSMM_MMBATCH_FLAG_STATISTIC & libxsmm_gemm_batchdesc.flags)) {
        libxsmm_descriptor_blob blob;
        const libxsmm_gemm_descriptor *const descriptor = libxsmm_dgemm_descriptor_init(&blob,
          *m, *n, *k, *lda, *ldb, *ldc, *alpha, *beta, LIBXSMM_GEMM_FLAGS(*transa, *transb),
          LIBXSMM_GEMM_EXT_MMBATCH_PREFETCH);

        assert(0 != libxsmm_gemm_batchsize);
        if (0 != descriptor) {
          const unsigned int max_batchsize = (unsigned int)((LIBXSMM_GEMM_BATCHSCALE) * libxsmm_gemm_batchsize);
          const unsigned int batchsize = LIBXSMM_ATOMIC_LOAD(&internal_ext_gemm_batchsize, LIBXSMM_ATOMIC_RELAXED);
          const unsigned int max_size = (0 != batchsize ? (((batchsize - 1) % max_batchsize) + 1) : 0);
          libxsmm_gemm_batchitem* batcharray = libxsmm_gemm_batcharray;
          unsigned int size = max_size;
          if (libxsmm_gemm_batchsize < max_size) {
            size = max_size - libxsmm_gemm_batchsize;
            batcharray += libxsmm_gemm_batchsize;
          }
          i = libxsmm_gemm_diffn_sw(descriptor, batcharray, 0/*hint*/, size, sizeof(libxsmm_gemm_batchitem));

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
            libxsmm_gemm_batcharray[i-1].stat.desc = *descriptor;
            libxsmm_gemm_batcharray[i-1].stat.count = 1;
            libxsmm_gemm_batcharray[i-1].stat.symbol = libxsmm_trace_info(&depth, 0, 0, 0, &maxnsyms);
            if (EXIT_SUCCESS == libxsmm_get_malloc_xinfo(libxsmm_gemm_batcharray, NULL/*size*/, NULL/*flags*/, &extra)) {
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
#if defined(LIBXSMM_GEMM_MMBATCH) && defined(LIBXSMM_BUILD_EXT)
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
    if (EXIT_SUCCESS != result && 0 != libxsmm_get_verbosity() &&
      1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
    {
      fprintf(stderr, "LIBXSMM ERROR: DGEMM batch recording failed!\n");
    }
# endif
#endif
  }
}


LIBXSMM_APIEXT void LIBXSMM_FSYMBOL(__wrap_sgemm)(
  const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const float* alpha, const float* a, const libxsmm_blasint* lda,
  const float* b, const libxsmm_blasint* ldb,
  const float* beta, float* c, const libxsmm_blasint* ldc)
{
  assert(0 != lda && 0 != ldb && 0 != ldc && 0 != m && 0 != n && 0 != k);
  assert(0 != transa && 0 != transb && 0 != alpha && 0 != beta);
  {
#if defined(LIBXSMM_GEMM_MMBATCH) && defined(LIBXSMM_BUILD_EXT)
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
      || LIBXSMM_NEQ(*alpha, libxsmm_gemm_batchdesc.alpha)
      || LIBXSMM_NEQ(*beta, libxsmm_gemm_batchdesc.beta))
#endif
    {
      if (0 == (libxsmm_gemm_wrap % 2) || 0 > libxsmm_gemm_wrap) { /* parallelized/tiled */
        libxsmm_sgemm_omp(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
      }
      else { /* small problem size */
        libxsmm_sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
      }
#if defined(LIBXSMM_GEMM_MMBATCH) && defined(LIBXSMM_BUILD_EXT)
      if (0 != (LIBXSMM_MMBATCH_FLAG_STATISTIC & libxsmm_gemm_batchdesc.flags)) {
        libxsmm_descriptor_blob blob;
        const libxsmm_gemm_descriptor *const descriptor = libxsmm_sgemm_descriptor_init(&blob,
          *m, *n, *k, *lda, *ldb, *ldc, *alpha, *beta, LIBXSMM_GEMM_FLAGS(*transa, *transb),
          LIBXSMM_GEMM_EXT_MMBATCH_PREFETCH);

        assert(0 != libxsmm_gemm_batchsize);
        if (0 != descriptor) {
          const unsigned int max_batchsize = (unsigned int)((LIBXSMM_GEMM_BATCHSCALE) * libxsmm_gemm_batchsize);
          const unsigned int batchsize = LIBXSMM_ATOMIC_LOAD(&internal_ext_gemm_batchsize, LIBXSMM_ATOMIC_RELAXED);
          const unsigned int max_size = (0 != batchsize ? (((batchsize - 1) % max_batchsize) + 1) : 0);
          libxsmm_gemm_batchitem* batcharray = libxsmm_gemm_batcharray;
          unsigned int size = max_size;
          if (libxsmm_gemm_batchsize < max_size) {
            size = max_size - libxsmm_gemm_batchsize;
            batcharray += libxsmm_gemm_batchsize;
          }
          i = libxsmm_gemm_diffn_sw(descriptor, batcharray, 0/*hint*/, size, sizeof(libxsmm_gemm_batchitem));

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
            libxsmm_gemm_batcharray[i-1].stat.desc = *descriptor;
            libxsmm_gemm_batcharray[i-1].stat.count = 1;
            libxsmm_gemm_batcharray[i-1].stat.symbol = libxsmm_trace_info(&depth, 0, 0, 0, &maxnsyms);
            if (EXIT_SUCCESS == libxsmm_get_malloc_xinfo(libxsmm_gemm_batcharray, NULL/*size*/, NULL/*flags*/, &extra)) {
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
#if defined(LIBXSMM_GEMM_MMBATCH) && defined(LIBXSMM_BUILD_EXT)
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
    if (EXIT_SUCCESS != result && 0 != libxsmm_get_verbosity() &&
      1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
    {
      fprintf(stderr, "LIBXSMM ERROR: SGEMM batch recording failed!\n");
    }
# endif
#endif
  }
}

#endif /*defined(LIBXSMM_BUILD) && defined(LIBXSMM_BUILD_EXT)*/


LIBXSMM_APIEXT void libxsmm_sgemm_omp(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const float* alpha, const float* a, const libxsmm_blasint* lda,
  const float* b, const libxsmm_blasint* ldb,
  const float* beta, float* c, const libxsmm_blasint* ldc)
{
  const libxsmm_blasint nn = *(n ? n : m), kk = *(k ? k : m);
  const unsigned long long size = 1ULL * (*m) * nn * kk;
  LIBXSMM_INIT
  assert(0 != libxsmm_gemm_tile);
  if (LIBXSMM_MAX_MNK < size) {
    const int index = LIBXSMM_MIN(libxsmm_icbrt_u64(size) >> 10, 7);
    const unsigned int tm = LIBXSMM_MIN(libxsmm_gemm_tile[1/*SP*/][0/*M*/][index], (unsigned int)*m);
    const unsigned int tn = LIBXSMM_MIN(libxsmm_gemm_tile[1/*SP*/][1/*N*/][index], (unsigned int)nn);
    const unsigned int tk = LIBXSMM_MIN(libxsmm_gemm_tile[1/*SP*/][2/*K*/][index], (unsigned int)kk);
    const char ctransa = (char)(0 != transa ? (*transa) : (0 == (LIBXSMM_FLAGS & LIBXSMM_GEMM_FLAG_TRANS_A) ? 'n' : 't'));
    const char ctransb = (char)(0 != transb ? (*transb) : (0 == (LIBXSMM_FLAGS & LIBXSMM_GEMM_FLAG_TRANS_B) ? 'n' : 't'));
    const libxsmm_blasint ilda = (NULL != lda ? *lda : (('n' == ctransa || 'N' == ctransa) ? *m : kk));
    const libxsmm_blasint ildb = (NULL != ldb ? *ldb : (('n' == ctransb || 'N' == ctransb) ? kk : nn));
    const libxsmm_blasint ildc = *(NULL != ldc ? ldc : m);
    const float ralpha = (0 != alpha ? *alpha : ((float)LIBXSMM_ALPHA));
    const float rbeta = (0 != beta ? *beta : ((float)LIBXSMM_BETA));
    assert((0 < tm || 0 == *m) && (0 < tn || 0 == nn) && (0 < tk || 0 == kk) && 0 < libxsmm_nt);
#if defined(_OPENMP)
# if defined(LIBXSMM_EXT_TASKS)
    if (0 == omp_get_active_level())
# else
    if (0 == omp_in_parallel())
# endif
    {
      LIBXSMM_TILED_XGEMM(
        LIBXSMM_EXT_PARALLEL, LIBXSMM_EXT_FOR_DLOOP, LIBXSMM_EXT_FOR_KERNEL, LIBXSMM_NOOP,
        LIBXSMM_EXT_MIN_NTASKS, LIBXSMM_EXT_OVERHEAD, libxsmm_nt,
        float, &ctransa, &ctransb, tm, tn, tk, *m, nn, kk,
        ralpha, a, ilda, b, ildb, rbeta, c, ildc);
    }
    else
#endif /*defined(_OPENMP)*/
#if defined(LIBXSMM_EXT_TASKS) /* implies OpenMP */
    { /* assume external parallelization */
      LIBXSMM_TILED_XGEMM(
        LIBXSMM_NOOP, LIBXSMM_NOOP_ARGS, LIBXSMM_EXT_TSK_KERNEL_ARGS,
        if (0 == libxsmm_nosync) { LIBXSMM_EXT_TSK_SYNC } /* allow to omit synchronization */,
          LIBXSMM_EXT_MIN_NTASKS, LIBXSMM_EXT_OVERHEAD, libxsmm_nt,
          float, transa, transb, tm, tn, tk, *m, nn, kk,
          ralpha, a, ilda, b, ildb, rbeta, c, ildc);
    }
#else
    {
      LIBXSMM_TILED_XGEMM(
        LIBXSMM_NOOP, LIBXSMM_NOOP_ARGS, LIBXSMM_NOOP_ARGS, LIBXSMM_NOOP,
        LIBXSMM_MIN_NTASKS, LIBXSMM_OVERHEAD, libxsmm_nt,
        float, transa, transb, tm, tn, tk, *m, nn, kk,
        ralpha, a, ilda, b, ildb, rbeta, c, ildc);
    }
#endif
  }
  else if (0 < size) { /* small problem size */
    libxsmm_sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  }
}


LIBXSMM_APIEXT void libxsmm_dgemm_omp(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const double* alpha, const double* a, const libxsmm_blasint* lda,
  const double* b, const libxsmm_blasint* ldb,
  const double* beta, double* c, const libxsmm_blasint* ldc)
{
  const libxsmm_blasint nn = *(n ? n : m), kk = *(k ? k : m);
  const unsigned long long size = 1ULL * (*m) * nn * kk;
  LIBXSMM_INIT
  assert(0 != libxsmm_gemm_tile);
  if (LIBXSMM_MAX_MNK < size) {
    const int index = LIBXSMM_MIN(libxsmm_icbrt_u64(size) >> 10, 7);
    const unsigned int tm = LIBXSMM_MIN(libxsmm_gemm_tile[0/*DP*/][0/*M*/][index], (unsigned int)*m);
    const unsigned int tn = LIBXSMM_MIN(libxsmm_gemm_tile[0/*DP*/][1/*N*/][index], (unsigned int)nn);
    const unsigned int tk = LIBXSMM_MIN(libxsmm_gemm_tile[0/*DP*/][2/*K*/][index], (unsigned int)kk);
    const char ctransa = (char)(0 != transa ? (*transa) : (0 == (LIBXSMM_FLAGS & LIBXSMM_GEMM_FLAG_TRANS_A) ? 'n' : 't'));
    const char ctransb = (char)(0 != transb ? (*transb) : (0 == (LIBXSMM_FLAGS & LIBXSMM_GEMM_FLAG_TRANS_B) ? 'n' : 't'));
    const libxsmm_blasint ilda = (NULL != lda ? *lda : (('n' == ctransa || 'N' == ctransa) ? *m : kk));
    const libxsmm_blasint ildb = (NULL != ldb ? *ldb : (('n' == ctransb || 'N' == ctransb) ? kk : nn));
    const libxsmm_blasint ildc = *(NULL != ldc ? ldc : m);
    const double ralpha = (0 != alpha ? *alpha : ((double)LIBXSMM_ALPHA));
    const double rbeta = (0 != beta ? *beta : ((double)LIBXSMM_BETA));
    assert((0 < tm || 0 == *m) && (0 < tn || 0 == nn) && (0 < tk || 0 == kk) && 0 < libxsmm_nt);
#if defined(_OPENMP)
# if defined(LIBXSMM_EXT_TASKS)
    if (0 == omp_get_active_level())
# else
    if (0 == omp_in_parallel())
# endif
    {
      LIBXSMM_TILED_XGEMM(
        LIBXSMM_EXT_PARALLEL, LIBXSMM_EXT_FOR_DLOOP, LIBXSMM_EXT_FOR_KERNEL, LIBXSMM_NOOP,
        LIBXSMM_EXT_MIN_NTASKS, LIBXSMM_EXT_OVERHEAD, libxsmm_nt,
        double, &ctransa, &ctransb, tm, tn, tk, *m, nn, kk,
        ralpha, a, ilda, b, ildb, rbeta, c, ildc);
    }
    else
#endif /*defined(_OPENMP)*/
#if defined(LIBXSMM_EXT_TASKS) /* implies OpenMP */
    { /* assume external parallelization */
      LIBXSMM_TILED_XGEMM(
        LIBXSMM_NOOP, LIBXSMM_NOOP_ARGS, LIBXSMM_EXT_TSK_KERNEL_ARGS,
        if (0 == libxsmm_nosync) { LIBXSMM_EXT_TSK_SYNC } /* allow to omit synchronization */,
          LIBXSMM_EXT_MIN_NTASKS, LIBXSMM_EXT_OVERHEAD, libxsmm_nt,
          double, transa, transb, tm, tn, tk, *m, nn, kk,
          ralpha, a, ilda, b, ildb, rbeta, c, ildc);
    }
#else
    {
      LIBXSMM_TILED_XGEMM(
        LIBXSMM_NOOP, LIBXSMM_NOOP_ARGS, LIBXSMM_NOOP_ARGS, LIBXSMM_NOOP,
        LIBXSMM_MIN_NTASKS, LIBXSMM_OVERHEAD, libxsmm_nt,
        double, transa, transb, tm, tn, tk, *m, nn, kk,
        ralpha, a, ilda, b, ildb, rbeta, c, ildc);
    }
#endif
  }
  else if (0 < size) { /* small problem size */
    libxsmm_dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  }
}


LIBXSMM_APIEXT int libxsmm_mmbatch_omp(libxsmm_xmmfunction kernel, libxsmm_blasint index_base, libxsmm_blasint index_stride,
  const libxsmm_blasint stride_a[], const libxsmm_blasint stride_b[], const libxsmm_blasint stride_c[],
  const void* a, const void* b, void* c, libxsmm_blasint batchsize)
{
  int result;
  const libxsmm_kernel_info* info;
  libxsmm_code_pointer code;
  libxsmm_kernel_kind kind;
  code.xgemm = kernel;
  info = libxsmm_get_kernel_info(code, &kind, NULL/*size*/);
  if (0 != info && LIBXSMM_KERNEL_KIND_MATMUL == kind && 0 != a && 0 != b && 0 != c) {
    LIBXSMM_INIT
    {
#if defined(_OPENMP)
      const unsigned int size = info->xgemm.m * info->xgemm.n * info->xgemm.k;
      const int chunksize = (0 >= libxsmm_gemm_chunksize ? ((int)(1048576 * libxsmm_icbrt_u32(size) / size)) : libxsmm_gemm_chunksize);
      const int max_chunksize = LIBXSMM_MAX(chunksize, 1), ntasks = (int)((LIBXSMM_ABS(batchsize) + max_chunksize - 1) / max_chunksize);

      if (1 < ntasks) {
# if defined(LIBXSMM_EXT_TASKS)
        if (0 == omp_get_active_level())
# else
        if (0 == omp_in_parallel())
# endif
        { /* enable internal parallelization */
          const int max_nthreads = omp_get_max_threads();
          const int nthreads = LIBXSMM_MIN(max_nthreads, ntasks);
# if defined(LIBXSMM_EXT_TASKS)
          if (0 == libxsmm_gemm_tasks)
# endif
          {
#           pragma omp parallel num_threads(nthreads)
            libxsmm_mmbatch_internal(kernel, index_base, index_stride,
              stride_a, stride_b, stride_c, a, b, c, batchsize,
              omp_get_thread_num(), nthreads, &info->xgemm);
            /* implicit synchronization (barrier) */
          }
# if defined(LIBXSMM_EXT_TASKS)
          else { /* tasks requested */
#           pragma omp parallel num_threads(nthreads)
            {
#             pragma omp single nowait /* anyone is good */
              { /* first thread discovering work will launch all tasks */
                libxsmm_blasint tid;
                for (tid = 0; tid < ntasks; ++tid) {
#                 pragma omp task
                  libxsmm_mmbatch_internal(kernel, index_base, index_stride,
                    stride_a, stride_b, stride_c, a, b, c, batchsize,
                    tid, ntasks, &info->xgemm);
                }
              }
            } /* implicit synchronization (barrier) */
          }
# endif
          result = EXIT_SUCCESS;
        }
        else { /* assume external parallelization */
# if defined(LIBXSMM_EXT_TASKS) /* OpenMP-tasks */
          libxsmm_blasint tid;
          for (tid = 0; tid < ntasks; ++tid) {
#           pragma omp task
            libxsmm_mmbatch_internal(kernel, index_base, index_stride,
              stride_a, stride_b, stride_c, a, b, c, batchsize,
              tid, ntasks, &info->xgemm);
          }
          /* allow to omit synchronization */
          if (0 == libxsmm_nosync) {
#           pragma omp taskwait
          }
          result = EXIT_SUCCESS;
# else /* sequential */
          result = libxsmm_mmbatch_internal(kernel, index_base, index_stride,
            stride_a, stride_b, stride_c, a, b, c, batchsize,
            0/*tid*/, 1/*nthreads*/, &info->xgemm);
# endif
        }
      }
      else
#endif /*defined(_OPENMP)*/
      { /* sequential */
        result = libxsmm_mmbatch_internal(kernel, index_base, index_stride,
          stride_a, stride_b, stride_c, a, b, c, batchsize,
          0/*tid*/, 1/*nthreads*/, &info->xgemm);
      }
    }
  }
  else { /* incorrect argument(s) */
    result = EXIT_FAILURE;
  }
  return result;
}


LIBXSMM_APIEXT void libxsmm_gemm_batch2_omp(libxsmm_gemm_precision iprec, libxsmm_gemm_precision oprec,
  const char* transa, const char* transb, libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const void* alpha, const void* a, const libxsmm_blasint* lda, const void* b, const libxsmm_blasint* ldb,
  const void* beta, void* c, const libxsmm_blasint* ldc, libxsmm_blasint index_base, libxsmm_blasint index_stride,
  const libxsmm_blasint stride_a[], const libxsmm_blasint stride_b[], const libxsmm_blasint stride_c[],
  libxsmm_blasint batchsize)
{
  const int gemm_flags = LIBXSMM_GEMM_PFLAGS(transa, transb, LIBXSMM_FLAGS);
  libxsmm_descriptor_blob blob;
  const libxsmm_gemm_descriptor *const descriptor = libxsmm_gemm_descriptor_init2(&blob, iprec, oprec, m, n, k,
    NULL != lda ? *lda : (0 == (LIBXSMM_GEMM_FLAG_TRANS_A & gemm_flags) ? m : k),
    NULL != ldb ? *ldb : (0 == (LIBXSMM_GEMM_FLAG_TRANS_B & gemm_flags) ? k : n),
    0 != ldc ? *ldc : m, alpha, beta, gemm_flags, libxsmm_get_gemm_prefetch(LIBXSMM_PREFETCH_AUTO));
  const libxsmm_xmmfunction kernel = libxsmm_xmmdispatch(descriptor);
  static int error_once = 0;
  int result;

  if (0 != kernel.xmm) {
    result = libxsmm_mmbatch_omp(kernel, index_base, index_stride,
      stride_a, stride_b, stride_c, a, b, c, batchsize);
  }
  else { /* fall-back */
    switch (iprec) {
      case LIBXSMM_GEMM_PRECISION_F64: {
        result = libxsmm_dmmbatch_blas(transa, transb, m, n, k,
          (const double*)alpha, a, lda, b, ldb, (const double*)beta, c, ldc,
          index_base, index_stride, stride_a, stride_b, stride_c, batchsize);
      } break;
      case LIBXSMM_GEMM_PRECISION_F32: {
        result = libxsmm_smmbatch_blas(transa, transb, m, n, k,
          (const float*)alpha, a, lda, b, ldb, (const float*)beta, c, ldc,
          index_base, index_stride, stride_a, stride_b, stride_c, batchsize);
      } break;
      default: result = EXIT_FAILURE;
    }
  }
  if (EXIT_SUCCESS != result
    && 0 != libxsmm_get_verbosity() /* library code is expected to be mute */
    && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
  {
    fprintf(stderr, "LIBXSMM ERROR: libxsmm_gemm_batch_omp failed!\n");
  }
}


LIBXSMM_APIEXT void libxsmm_gemm_batch_omp(libxsmm_gemm_precision precision,
  const char* transa, const char* transb, libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const void* alpha, const void* a, const libxsmm_blasint* lda, const void* b, const libxsmm_blasint* ldb,
  const void* beta, void* c, const libxsmm_blasint* ldc, libxsmm_blasint index_base, libxsmm_blasint index_stride,
  const libxsmm_blasint stride_a[], const libxsmm_blasint stride_b[], const libxsmm_blasint stride_c[],
  libxsmm_blasint batchsize)
{
  libxsmm_gemm_batch2_omp(precision, precision, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
    index_base, index_stride, stride_a, stride_b, stride_c, batchsize);
}


LIBXSMM_APIEXT void libxsmm_mmbatch_begin2(libxsmm_gemm_precision iprec, libxsmm_gemm_precision oprec,
  const int* flags, const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc,
  const void* alpha, const void* beta)
{
#if defined(LIBXSMM_GEMM_MMBATCH) && defined(LIBXSMM_BUILD_EXT)
# if defined(_MSC_VER)
#   pragma warning(push)
#   pragma warning(disable: 26115) /* try-lock is treated incorrectly by static analysis */
# endif
  LIBXSMM_INIT
  if (0 != libxsmm_gemm_batcharray /* batch-recording available, but not yet running */
    /* currently, batch recording is only enabled if all values are present (no complex filtering) */
    && 0 != flags && 0 != alpha && 0 != beta
    && 0 != lda && 0 != ldb && 0 != ldc
    && 0 != m && 0 != n && 0 != k
    && LIBXSMM_LOCK_ACQUIRED(LIBXSMM_LOCK_DEFAULT) == LIBXSMM_LOCK_TRYLOCK(LIBXSMM_LOCK_DEFAULT, &libxsmm_gemm_batchlock))
  {
    libxsmm_descriptor_blob blob;
    const libxsmm_gemm_descriptor *const descriptor = libxsmm_gemm_descriptor_init2(&blob, iprec, oprec,
      *m, *n, *k, *lda, *ldb, *ldc, alpha, beta, *flags, libxsmm_get_gemm_prefetch(LIBXSMM_GEMM_EXT_MMBATCH_PREFETCH));
    static int error_once = 0;
    int result = EXIT_SUCCESS;

    if (0 != descriptor) {
      const unsigned int max_batchsize = (unsigned int)((LIBXSMM_GEMM_BATCHSCALE) * libxsmm_gemm_batchsize);
      unsigned int i;

      /* eventually overwrite the oldest entry */
      assert((LIBXSMM_GEMM_EXT_MMBATCH_MAXDEPTH) == LIBXSMM_UP2POT(LIBXSMM_GEMM_EXT_MMBATCH_MAXDEPTH)/*is pot*/);
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
          libxsmm_gemm_batchdesc = *descriptor;
        }
        else {
          libxsmm_gemm_batchdesc.flags = LIBXSMM_MMBATCH_FLAG_STATISTIC;
        }
      }
    }
    else {
      result = EXIT_FAILURE;
    }
    if (EXIT_SUCCESS != result && 0 != libxsmm_get_verbosity() /* library code is expected to be mute */
      && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
    {
      fprintf(stderr, "LIBXSMM ERROR: GEMM batch enabling failed!\n");
    }
    LIBXSMM_LOCK_RELEASE(LIBXSMM_LOCK_DEFAULT, &libxsmm_gemm_batchlock);
  }
# if defined(_MSC_VER)
#   pragma warning(pop)
# endif
#else
  LIBXSMM_UNUSED(iprec); LIBXSMM_UNUSED(oprec); LIBXSMM_UNUSED(flags);
  LIBXSMM_UNUSED(m); LIBXSMM_UNUSED(n); LIBXSMM_UNUSED(k);
  LIBXSMM_UNUSED(lda); LIBXSMM_UNUSED(ldb); LIBXSMM_UNUSED(ldc);
  LIBXSMM_UNUSED(alpha); LIBXSMM_UNUSED(beta);
#endif
}


LIBXSMM_APIEXT void libxsmm_mmbatch_begin(libxsmm_gemm_precision precision, const int* flags,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc,
  const void* alpha, const void* beta)
{
  libxsmm_mmbatch_begin2(precision, precision, flags, m, n, k, lda, ldb, ldc, alpha, beta);
}


LIBXSMM_APIEXT void libxsmm_mmbatch_end(void)
{
#if defined(LIBXSMM_GEMM_MMBATCH) && defined(LIBXSMM_BUILD_EXT)
# if defined(_MSC_VER)
#   pragma warning(push)
#   pragma warning(disable: 26115) /* try-lock is treated incorrectly by static analysis */
# endif
  if (LIBXSMM_LOCK_ACQUIRED(LIBXSMM_LOCK_DEFAULT) == LIBXSMM_LOCK_TRYLOCK(LIBXSMM_LOCK_DEFAULT, &libxsmm_gemm_batchlock)) {
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
      assert((LIBXSMM_GEMM_EXT_MMBATCH_MAXDEPTH) == LIBXSMM_UP2POT(LIBXSMM_GEMM_EXT_MMBATCH_MAXDEPTH)/*is pot*/);
      libxsmm_gemm_batchdesc = internal_ext_gemm_batchdesc[LIBXSMM_MOD2(internal_ext_gemm_batchdepth, LIBXSMM_GEMM_EXT_MMBATCH_MAXDEPTH)];
    }
    else if (0 != libxsmm_get_verbosity() /* library code is expected to be mute */
      && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
    {
      fprintf(stderr, "LIBXSMM ERROR: GEMM batch processing failed!\n");
    }
    LIBXSMM_LOCK_RELEASE(LIBXSMM_LOCK_DEFAULT, &libxsmm_gemm_batchlock);
  }
# if defined(_MSC_VER)
#   pragma warning(pop)
# endif
#endif
}


#if defined(LIBXSMM_BUILD) && defined(LIBXSMM_BUILD_EXT)

/* implementation provided for Fortran 77 compatibility */
LIBXSMM_APIEXT void LIBXSMM_FSYMBOL(libxsmm_sgemm_omp)(const char*, const char*,
  const libxsmm_blasint*, const libxsmm_blasint*, const libxsmm_blasint*,
  const float*, const float*, const libxsmm_blasint*,
  const float*, const libxsmm_blasint*,
  const float*, float*, const libxsmm_blasint*);
LIBXSMM_APIEXT void LIBXSMM_FSYMBOL(libxsmm_sgemm_omp)(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const float* alpha, const float* a, const libxsmm_blasint* lda,
  const float* b, const libxsmm_blasint* ldb,
  const float* beta, float* c, const libxsmm_blasint* ldc)
{
  libxsmm_sgemm_omp(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}


/* implementation provided for Fortran 77 compatibility */
LIBXSMM_APIEXT void LIBXSMM_FSYMBOL(libxsmm_dgemm_omp)(const char*, const char*,
  const libxsmm_blasint*, const libxsmm_blasint*, const libxsmm_blasint*,
  const double*, const double*, const libxsmm_blasint*,
  const double*, const libxsmm_blasint*,
  const double*, double*, const libxsmm_blasint*);
LIBXSMM_APIEXT void LIBXSMM_FSYMBOL(libxsmm_dgemm_omp)(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const double* alpha, const double* a, const libxsmm_blasint* lda,
  const double* b, const libxsmm_blasint* ldb,
  const double* beta, double* c, const libxsmm_blasint* ldc)
{
  libxsmm_dgemm_omp(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}


/* implementation provided for Fortran 77 compatibility */
LIBXSMM_APIEXT void LIBXSMM_FSYMBOL(libxsmm_mmbatch_omp)(libxsmm_xmmfunction kernel, const libxsmm_blasint* index_base,
  const libxsmm_blasint* index_stride, const libxsmm_blasint stride_a[], const libxsmm_blasint stride_b[], const libxsmm_blasint stride_c[],
  const void* a, const void* b, void* c, const libxsmm_blasint* batchsize);
LIBXSMM_APIEXT void LIBXSMM_FSYMBOL(libxsmm_mmbatch_omp)(libxsmm_xmmfunction kernel, const libxsmm_blasint* index_base,
  const libxsmm_blasint* index_stride, const libxsmm_blasint stride_a[], const libxsmm_blasint stride_b[], const libxsmm_blasint stride_c[],
  const void* a, const void* b, void* c, const libxsmm_blasint* batchsize)
{
  static int error_once = 0;
  assert(0 != a && 0 != b && 0 != c && 0 != index_base && 0 != index_stride && 0 != batchsize);
  if (EXIT_SUCCESS != libxsmm_mmbatch_omp(kernel, *index_base, *index_stride, stride_a, stride_b, stride_c, a, b, c, *batchsize)
    && 0 != libxsmm_get_verbosity() /* library code is expected to be mute */
    && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
  {
    fprintf(stderr, "LIBXSMM ERROR: libxsmm_mmbatch_omp failed!\n");
  }
}


/* implementation provided for Fortran 77 compatibility */
LIBXSMM_APIEXT void LIBXSMM_FSYMBOL(libxsmm_gemm_batch2_omp)(const libxsmm_gemm_precision* iprec, const libxsmm_gemm_precision* oprec,
  const char* transa, const char* transb, const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const void* alpha, const void* a, const libxsmm_blasint* lda, const void* b, const libxsmm_blasint* ldb,
  const void* beta, void* c, const libxsmm_blasint* ldc, const libxsmm_blasint* index_base, const libxsmm_blasint* index_stride,
  const libxsmm_blasint stride_a[], const libxsmm_blasint stride_b[], const libxsmm_blasint stride_c[],
  const libxsmm_blasint* batchsize);
LIBXSMM_APIEXT void LIBXSMM_FSYMBOL(libxsmm_gemm_batch2_omp)(const libxsmm_gemm_precision* iprec, const libxsmm_gemm_precision* oprec,
  const char* transa, const char* transb, const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const void* alpha, const void* a, const libxsmm_blasint* lda, const void* b, const libxsmm_blasint* ldb,
  const void* beta, void* c, const libxsmm_blasint* ldc, const libxsmm_blasint* index_base, const libxsmm_blasint* index_stride,
  const libxsmm_blasint stride_a[], const libxsmm_blasint stride_b[], const libxsmm_blasint stride_c[],
  const libxsmm_blasint* batchsize)
{
  assert(0 != iprec && 0 != oprec && 0 != m && 0 != n && 0 != k && 0 != index_base && 0 != index_stride && 0 != batchsize);
  libxsmm_gemm_batch2_omp(*iprec, *oprec, transa, transb, *m, *n, *k, alpha, a, lda, b, ldb, beta, c, ldc, *index_base,
    *index_stride, stride_a, stride_b, stride_c, *batchsize);
}


/* implementation provided for Fortran 77 compatibility */
LIBXSMM_APIEXT void LIBXSMM_FSYMBOL(libxsmm_gemm_batch_omp)(const libxsmm_gemm_precision* precision,
  const char* transa, const char* transb, const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const void* alpha, const void* a, const libxsmm_blasint* lda, const void* b, const libxsmm_blasint* ldb,
  const void* beta, void* c, const libxsmm_blasint* ldc, const libxsmm_blasint* index_base, const libxsmm_blasint* index_stride,
  const libxsmm_blasint stride_a[], const libxsmm_blasint stride_b[], const libxsmm_blasint stride_c[],
  const libxsmm_blasint* batchsize);
LIBXSMM_APIEXT void LIBXSMM_FSYMBOL(libxsmm_gemm_batch_omp)(const libxsmm_gemm_precision* precision,
  const char* transa, const char* transb, const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const void* alpha, const void* a, const libxsmm_blasint* lda, const void* b, const libxsmm_blasint* ldb,
  const void* beta, void* c, const libxsmm_blasint* ldc, const libxsmm_blasint* index_base, const libxsmm_blasint* index_stride,
  const libxsmm_blasint stride_a[], const libxsmm_blasint stride_b[], const libxsmm_blasint stride_c[],
  const libxsmm_blasint* batchsize)
{
  assert(0 != precision && 0 != m && 0 != n && 0 != k && 0 != index_base && 0 != index_stride && 0 != batchsize);
  libxsmm_gemm_batch_omp(*precision, transa, transb, *m, *n, *k, alpha, a, lda, b, ldb, beta, c, ldc, *index_base,
    *index_stride, stride_a, stride_b, stride_c, *batchsize);
}


/* implementation provided for Fortran 77 compatibility */
LIBXSMM_APIEXT void LIBXSMM_FSYMBOL(libxsmm_mmbatch_begin2)(const libxsmm_gemm_precision* iprec, const libxsmm_gemm_precision* oprec,
  const int* flags, const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc,
  const void* alpha, const void* beta);
LIBXSMM_APIEXT void LIBXSMM_FSYMBOL(libxsmm_mmbatch_begin2)(const libxsmm_gemm_precision* iprec, const libxsmm_gemm_precision* oprec,
  const int* flags, const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc,
  const void* alpha, const void* beta)
{
  assert(0 != iprec && 0 != oprec);
  libxsmm_mmbatch_begin2(*iprec, *oprec, flags, m, n, k, lda, ldb, ldc, alpha, beta);
}


/* implementation provided for Fortran 77 compatibility */
LIBXSMM_APIEXT void LIBXSMM_FSYMBOL(libxsmm_mmbatch_begin)(const libxsmm_gemm_precision* precision,
  const int* flags, const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc,
  const void* alpha, const void* beta);
LIBXSMM_APIEXT void LIBXSMM_FSYMBOL(libxsmm_mmbatch_begin)(const libxsmm_gemm_precision* precision,
  const int* flags, const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc,
  const void* alpha, const void* beta)
{
  assert(0 != precision);
  libxsmm_mmbatch_begin(*precision, flags, m, n, k, lda, ldb, ldc, alpha, beta);
}


/* implementation provided for Fortran 77 compatibility */
LIBXSMM_APIEXT void LIBXSMM_FSYMBOL(libxsmm_mmbatch_end)(void);
LIBXSMM_APIEXT void LIBXSMM_FSYMBOL(libxsmm_mmbatch_end)(void)
{
  libxsmm_mmbatch_end();
}

#endif /*defined(LIBXSMM_BUILD) && defined(LIBXSMM_BUILD_EXT)*/

