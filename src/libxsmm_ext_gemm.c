/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Hans Pabst (Intel Corp.)
******************************************************************************/
#include <libxsmm.h>
#include "libxsmm_gemm.h"
#include "libxsmm_ext.h"

#if defined(LIBXSMM_WRAP) && defined(LIBXSMM_BUILD_EXT)
# include "libxsmm_trace.h"
#endif

#if !defined(LIBXSMM_EXT_GEMM_PARGROUPS_INFO) && 0
# define LIBXSMM_EXT_GEMM_PARGROUPS_INFO
#endif

#if defined(LIBXSMM_WRAP) && defined(LIBXSMM_BUILD_EXT)
# if !defined(LIBXSMM_EXT_GEMM_MMBATCH_PREFETCH)
#   define LIBXSMM_EXT_GEMM_MMBATCH_PREFETCH libxsmm_get_gemm_prefetch(LIBXSMM_PREFETCH_AUTO)
# endif
# if !defined(LIBXSMM_EXT_GEMM_MMBATCH_MAXDEPTH)
#   define LIBXSMM_EXT_GEMM_MMBATCH_MAXDEPTH 8/*POT*/
# endif
LIBXSMM_APIVAR_DEFINE(libxsmm_gemm_descriptor internal_ext_gemm_batchdesc[LIBXSMM_EXT_GEMM_MMBATCH_MAXDEPTH]);
LIBXSMM_APIVAR_DEFINE(unsigned int internal_ext_gemm_batchdepth);
LIBXSMM_APIVAR_DEFINE(unsigned int internal_ext_gemm_batchsize);
#endif


#if defined(LIBXSMM_WRAP) && defined(LIBXSMM_BUILD_EXT)
LIBXSMM_API_INLINE int internal_mmbatch_sortrev(const void* stat_a, const void* stat_b)
{
  const libxsmm_mmbatch_item *const a = (const libxsmm_mmbatch_item*)stat_a;
  const libxsmm_mmbatch_item *const b = (const libxsmm_mmbatch_item*)stat_b;
  LIBXSMM_ASSERT(NULL != stat_a && NULL != stat_b);
  return a->stat.count < b->stat.count ? 1 : (b->stat.count < a->stat.count ? -1 : 0);
}
#endif /*defined(LIBXSMM_WRAP) && defined(LIBXSMM_BUILD_EXT)*/


LIBXSMM_API_INLINE int internal_mmbatch_flush(const libxsmm_gemm_descriptor* batchdesc,
  libxsmm_blasint batchsize, libxsmm_mmbatch_item* batcharray)
{
  int result = EXIT_SUCCESS;
#if defined(LIBXSMM_WRAP) && defined(LIBXSMM_BUILD_EXT)
  if (0 != batchsize) { /* recorded/lazy multiplications */
    const libxsmm_blasint itemsize = sizeof(libxsmm_mmbatch_item);
    LIBXSMM_ASSERT(NULL != batchdesc && 0 < batchsize);
    if (0 == (LIBXSMM_MMBATCH_FLAG_STATISTIC & batchdesc->flags)) { /* process batch */
      const libxsmm_xmmfunction kernel = libxsmm_xmmdispatch(batchdesc);
      if (NULL != kernel.xmm) {
        const unsigned char itypesize = libxsmm_typesize((libxsmm_datatype)LIBXSMM_GETENUM_INP(batchdesc->datatype));
        const unsigned char otypesize = libxsmm_typesize((libxsmm_datatype)LIBXSMM_GETENUM_OUT(batchdesc->datatype));
#if defined(_OPENMP)
        if (0 == (LIBXSMM_MMBATCH_FLAG_SEQUENTIAL & batchdesc->flags)) { /* parallelized */
          const int nchunks = (int)LIBXSMM_UPDIV(batchsize, libxsmm_gemm_taskgrain);
# if defined(LIBXSMM_EXT_TASKS)
          if (0 == omp_get_active_level()) {
            const int max_nthreads = omp_get_max_threads();
            const int nthreads = LIBXSMM_MIN(max_nthreads, nchunks);
            if (0 == libxsmm_gemm_tasks)
# else
          if (0 == omp_in_parallel()) {
            const int max_nthreads = omp_get_max_threads();
            const int nthreads = LIBXSMM_MIN(max_nthreads, nchunks);
# endif
            { /* classic internal parallelization */
#             pragma omp parallel num_threads(nthreads)
              /*check*/libxsmm_mmbatch_kernel(
                kernel, 0/*index_base*/, 0/*index_stride*/, &itemsize, &itemsize, &itemsize,
                &batcharray->value.a, &batcharray->value.b, &batcharray->value.c,
                0 == (LIBXSMM_MMBATCH_FLAG_SYNCHRONIZED & batchdesc->flags) ? batchsize : -batchsize,
                omp_get_thread_num(), nthreads, itypesize, otypesize, batchdesc->flags);
            }
# if defined(LIBXSMM_EXT_TASKS)
            else { /* internal parallelization with tasks */
#             pragma omp parallel num_threads(nthreads)
              { /* first thread discovering work will launch all tasks */
#               pragma omp single nowait /* anyone is good */
                { int tid; for (tid = 0; tid < nchunks/*ntasks*/; ++tid) {
#                 pragma omp task untied
                  /*check*/libxsmm_mmbatch_kernel(
                    kernel, 0/*index_base*/, 0/*index_stride*/, &itemsize, &itemsize, &itemsize,
                    &batcharray->value.a, &batcharray->value.b, &batcharray->value.c,
                    0 == (LIBXSMM_MMBATCH_FLAG_SYNCHRONIZED & batchdesc->flags) ? batchsize : -batchsize,
                    tid, nchunks/*ntasks*/, itypesize, otypesize, batchdesc->flags);
                  }
                }
              } /* implicit synchronization (barrier) */
            }
# endif
          }
          else { /* assume external parallelization */
            int tid; for (tid = 0; tid < nchunks/*ntasks*/; ++tid) {
# if defined(LIBXSMM_EXT_TASKS)
#             pragma omp task untied
#endif
              /*check*/libxsmm_mmbatch_kernel(
                kernel, 0/*index_base*/, 0/*index_stride*/, &itemsize, &itemsize, &itemsize,
                &batcharray->value.a, &batcharray->value.b, &batcharray->value.c,
                0 == (LIBXSMM_MMBATCH_FLAG_SYNCHRONIZED & batchdesc->flags) ? batchsize : -batchsize,
                tid, nchunks/*ntasks*/, itypesize, otypesize, batchdesc->flags);
            }
# if defined(LIBXSMM_EXT_TASKS)
            if (0 == libxsmm_nosync) { /* allow to omit synchronization */
#             pragma omp taskwait
            }
# endif
          }
        }
        else
#endif
        { /* sequential */
          result = libxsmm_mmbatch_kernel(
            kernel, 0/*index_base*/, 0/*index_stride*/, &itemsize, &itemsize, &itemsize,
            &batcharray->value.a, &batcharray->value.b, &batcharray->value.c, batchsize,
            0/*tid*/, 1/*nthreads*/, itypesize, otypesize, batchdesc->flags);
        }
      }
      else { /* no fallback */
        /* several reasons to arrive here: try-lock, unsuitable SMM, etc. */
        result = EXIT_FAILURE;
      }
      memset(batcharray, 0, (size_t)batchsize * (size_t)itemsize); /* clear */
    }
    else { /* print statistic */
      const libxsmm_blasint limit = (LIBXSMM_GEMM_MMBATCH_VERBOSITY < libxsmm_verbosity ? batchsize/*unlimited*/ : 7/*limited*/);
      unsigned int threshold, batchcount;
      libxsmm_blasint count = 0, i;
      LIBXSMM_ASSERT(NULL != batcharray);
      qsort(batcharray, (size_t)batchsize, (size_t)itemsize, internal_mmbatch_sortrev);
      batchcount = batcharray[0].stat.count;
      threshold = ((LIBXSMM_GEMM_MMBATCH_VERBOSITY < libxsmm_verbosity || 3 >= batchsize) ? 0 : (batchcount / 2));
      for (i = 1; i < batchsize; ++i) batchcount += batcharray[i].stat.count;
      LIBXSMM_STDIO_ACQUIRE();
      for (i = 0; i < batchsize; ++i) {
        const libxsmm_gemm_descriptor descriptor = batcharray[i].stat.desc;
        const libxsmm_blasint lda = descriptor.lda, ldb = descriptor.ldb, ldc = descriptor.ldc;
        const libxsmm_blasint m = descriptor.m, n = descriptor.n, k = descriptor.k;
        const char *const symbol = batcharray[i].stat.symbol;
        const unsigned int ci = batcharray[i].stat.count;
        LIBXSMM_MEMZERO127(batcharray + i); /* clear */
        if (threshold < ci && count < limit /* limit printed statistic */
          && 0 < m && 0 < n && 0 < k)
        {
          const unsigned int ciperc = (unsigned int)(100.0 * ci / batchcount + 0.5);
          if (0 != ciperc) {
            LIBXSMM_ASSERT(0 != ci);
            if (0 == count) {
              fprintf(stderr, "\nLIBXSMM STATISTIC: %u multiplication%c\n", batchcount, 1 < batchcount ? 's' : ' ');
            }
            LIBXSMM_GEMM_PRINT2(stderr,
              LIBXSMM_GETENUM_INP(descriptor.datatype), LIBXSMM_GETENUM_OUT(descriptor.datatype), descriptor.flags, m, n, k,
              1, NULL/*a*/, lda, NULL/*b*/, ldb, 0 != (LIBXSMM_GEMM_FLAG_BETA_0  & descriptor.flags) ? 0 : 1, NULL/*c*/, ldc);
            if (NULL != symbol && 0 != *symbol) {
              fprintf(stderr, ": %u%% [%s]\n", ciperc, symbol);
            }
            else {
              fprintf(stderr, ": %u%%\n", ciperc);
            }
            ++count;
          }
          else break;
        }
      }
      LIBXSMM_STDIO_RELEASE();
    }
  }
#else
  LIBXSMM_UNUSED(batchdesc); LIBXSMM_UNUSED(batchsize); LIBXSMM_UNUSED(batcharray);
#endif
  return result;
}


#if defined(LIBXSMM_BUILD) && defined(LIBXSMM_BUILD_EXT)

#if defined(LIBXSMM_BLAS_WRAP_DYNAMIC)
LIBXSMM_API libxsmm_dgemm_batch_function libxsmm_original_dgemm_batch(void)
{
# if (0 != LIBXSMM_BLAS)
  LIBXSMM_BLAS_WRAPPER(1, double, gemm_batch, libxsmm_original_dgemm_batch_function, libxsmm_original_dgemm_batch/*self*/);
  /*LIBXSMM_ASSERT(NULL != libxsmm_original_dgemm_batch_function);*/
# else
  LIBXSMM_BLAS_WRAPPER(0, double, gemm_batch, libxsmm_original_dgemm_batch_function, libxsmm_original_dgemm_batch/*self*/);
# endif
  return libxsmm_original_dgemm_batch_function;
}

LIBXSMM_API libxsmm_sgemm_batch_function libxsmm_original_sgemm_batch(void)
{
# if (0 != LIBXSMM_BLAS)
  LIBXSMM_BLAS_WRAPPER(1, float, gemm_batch, libxsmm_original_sgemm_batch_function, libxsmm_original_sgemm_batch/*self*/);
  /*LIBXSMM_ASSERT(NULL != libxsmm_original_sgemm_batch_function);*/
# else
  LIBXSMM_BLAS_WRAPPER(0, float, gemm_batch, libxsmm_original_sgemm_batch_function, libxsmm_original_sgemm_batch/*self*/);
# endif
  return libxsmm_original_sgemm_batch_function;
}

LIBXSMM_API libxsmm_dgemm_function libxsmm_original_dgemm(void)
{
# if (0 != LIBXSMM_BLAS)
  LIBXSMM_BLAS_WRAPPER(1, double, gemm, libxsmm_original_dgemm_function, libxsmm_original_dgemm/*self*/);
  LIBXSMM_ASSERT(NULL != libxsmm_original_dgemm_function);
# else
  LIBXSMM_BLAS_WRAPPER(0, double, gemm, libxsmm_original_dgemm_function, libxsmm_original_dgemm/*self*/);
# endif
  return libxsmm_original_dgemm_function;
}

LIBXSMM_API libxsmm_sgemm_function libxsmm_original_sgemm(void)
{
# if (0 != LIBXSMM_BLAS)
  LIBXSMM_BLAS_WRAPPER(1, float, gemm, libxsmm_original_sgemm_function, libxsmm_original_sgemm/*self*/);
  LIBXSMM_ASSERT(NULL != libxsmm_original_sgemm_function);
# else
  LIBXSMM_BLAS_WRAPPER(0, float, gemm, libxsmm_original_sgemm_function, libxsmm_original_sgemm/*self*/);
# endif
  return libxsmm_original_sgemm_function;
}

LIBXSMM_API libxsmm_dgemv_function libxsmm_original_dgemv(void)
{
# if (0 != LIBXSMM_BLAS)
  LIBXSMM_BLAS_WRAPPER(1, double, gemv, libxsmm_original_dgemv_function, libxsmm_original_dgemv/*self*/);
  LIBXSMM_ASSERT(NULL != libxsmm_original_dgemv_function);
# else
  LIBXSMM_BLAS_WRAPPER(0, double, gemv, libxsmm_original_dgemv_function, libxsmm_original_dgemv/*self*/);
# endif
  return libxsmm_original_dgemv_function;
}

LIBXSMM_API libxsmm_sgemv_function libxsmm_original_sgemv(void)
{
# if (0 != LIBXSMM_BLAS)
  LIBXSMM_BLAS_WRAPPER(1, float, gemv, libxsmm_original_sgemv_function, libxsmm_original_sgemv/*self*/);
  LIBXSMM_ASSERT(NULL != libxsmm_original_sgemv_function);
# else
  LIBXSMM_BLAS_WRAPPER(0, float, gemv, libxsmm_original_sgemv_function, libxsmm_original_sgemv/*self*/);
# endif
  return libxsmm_original_sgemv_function;
}
#endif /*defined(LIBXSMM_BLAS_WRAP_DYNAMIC)*/


LIBXSMM_APIEXT LIBXSMM_ATTRIBUTE_USED void LIBXSMM_FSYMBOL(__wrap_dgemm_batch)(
  const char transa_array[], const char transb_array[], const libxsmm_blasint m_array[], const libxsmm_blasint n_array[], const libxsmm_blasint k_array[],
  const double alpha_array[], const double* a_array[], const libxsmm_blasint lda_array[], const double* b_array[], const libxsmm_blasint ldb_array[],
  const double beta_array[], double* c_array[], const libxsmm_blasint ldc_array[], const libxsmm_blasint* group_count, const libxsmm_blasint group_size[])
{
  LIBXSMM_ASSERT(NULL != lda_array && NULL != ldb_array && NULL != ldc_array && NULL != m_array && NULL != n_array && NULL != k_array);
  LIBXSMM_ASSERT(NULL != transa_array && NULL != transb_array && NULL != alpha_array && NULL != beta_array);
  LIBXSMM_ASSERT(NULL != group_count && NULL != group_size);
  LIBXSMM_INIT
  if (0 != libxsmm_gemm_wrap) {
    if (0 != (libxsmm_gemm_wrap & 1)) { /* sequential */
      libxsmm_dgemm_batch(transa_array, transb_array, m_array, n_array, k_array,
        alpha_array, a_array, lda_array, b_array, ldb_array, beta_array, c_array, ldc_array,
        group_count, group_size);
    }
    else { /* parallelized */
      libxsmm_dgemm_batch_omp(transa_array, transb_array, m_array, n_array, k_array,
        alpha_array, a_array, lda_array, b_array, ldb_array, beta_array, c_array, ldc_array,
        group_count, group_size);
    }
  }
  else {
    LIBXSMM_GEMM_BATCH_SYMBOL(double)(transa_array, transb_array, m_array, n_array, k_array,
      alpha_array, a_array, lda_array, b_array, ldb_array, beta_array, c_array, ldc_array,
      group_count, group_size);
  }
}


LIBXSMM_APIEXT LIBXSMM_ATTRIBUTE_USED void LIBXSMM_FSYMBOL(__wrap_sgemm_batch)(
  const char transa_array[], const char transb_array[], const libxsmm_blasint m_array[], const libxsmm_blasint n_array[], const libxsmm_blasint k_array[],
  const float alpha_array[], const float* a_array[], const libxsmm_blasint lda_array[], const float* b_array[], const libxsmm_blasint ldb_array[],
  const float beta_array[], float* c_array[], const libxsmm_blasint ldc_array[], const libxsmm_blasint* group_count, const libxsmm_blasint group_size[])
{
  LIBXSMM_ASSERT(NULL != lda_array && NULL != ldb_array && NULL != ldc_array && NULL != m_array && NULL != n_array && NULL != k_array);
  LIBXSMM_ASSERT(NULL != transa_array && NULL != transb_array && NULL != alpha_array && NULL != beta_array);
  LIBXSMM_ASSERT(NULL != group_count && NULL != group_size);
  LIBXSMM_INIT
  if (0 != libxsmm_gemm_wrap) {
    if (0 != (libxsmm_gemm_wrap & 1)) { /* sequential */
      libxsmm_sgemm_batch(transa_array, transb_array, m_array, n_array, k_array,
        alpha_array, a_array, lda_array, b_array, ldb_array, beta_array, c_array, ldc_array,
        group_count, group_size);
    }
    else { /* parallelized */
      libxsmm_sgemm_batch_omp(transa_array, transb_array, m_array, n_array, k_array,
        alpha_array, a_array, lda_array, b_array, ldb_array, beta_array, c_array, ldc_array,
        group_count, group_size);
    }
  }
  else {
    LIBXSMM_GEMM_BATCH_SYMBOL(float)(transa_array, transb_array, m_array, n_array, k_array,
      alpha_array, a_array, lda_array, b_array, ldb_array, beta_array, c_array, ldc_array,
      group_count, group_size);
  }
}


LIBXSMM_APIEXT LIBXSMM_ATTRIBUTE_USED void LIBXSMM_FSYMBOL(__wrap_dgemm)(
  const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const double* alpha, const double* a, const libxsmm_blasint* lda,
  const double* b, const libxsmm_blasint* ldb,
  const double* beta, double* c, const libxsmm_blasint* ldc)
{
  LIBXSMM_ASSERT(NULL != lda && NULL != ldb && NULL != ldc && NULL != m && NULL != n && NULL != k);
  LIBXSMM_ASSERT(NULL != transa && NULL != transb && NULL != alpha && NULL != beta);
  {
#if defined(LIBXSMM_WRAP) && defined(LIBXSMM_BUILD_EXT)
    unsigned int i = 0; /* no flush */
    int flags = -1;
# if !defined(NDEBUG)
    static int error_once = 0;
    int result = EXIT_SUCCESS;
# endif
    LIBXSMM_INIT
    if (0 != libxsmm_gemm_wrap && (NULL == libxsmm_mmbatch_array
      || LIBXSMM_DATATYPE_F64 != libxsmm_mmbatch_desc.datatype
      || ((unsigned int)*lda) != libxsmm_mmbatch_desc.lda
      || ((unsigned int)*ldb) != libxsmm_mmbatch_desc.ldb
      || ((unsigned int)*ldc) != libxsmm_mmbatch_desc.ldc
      || ((unsigned int)*m) != libxsmm_mmbatch_desc.m
      || ((unsigned int)*n) != libxsmm_mmbatch_desc.n
      || ((unsigned int)*k) != libxsmm_mmbatch_desc.k
      || (flags = LIBXSMM_GEMM_FLAGS(*transa, *transb)) != (int)(LIBXSMM_GEMM_FLAG_TRANS_AB & libxsmm_mmbatch_desc.flags)
      || LIBXSMM_NEQ(1, *alpha) || LIBXSMM_NEQ(0 != (LIBXSMM_GEMM_FLAG_BETA_0 & libxsmm_mmbatch_desc.flags) ? 0 : 1, *beta)))
#endif
    {
#if defined(_DEBUG)
      const char *const env_check = getenv("LIBXSMM_GEMM_CHECK");
      const double check = LIBXSMM_ABS(NULL == env_check ? 0 : atof(env_check));
      void* d = NULL;
      if (LIBXSMM_NEQ(0, check)) {
        const size_t size = (size_t)(*ldc) * (size_t)(*n) * sizeof(double);
        d = libxsmm_scratch_malloc(size, 0/*auto*/, LIBXSMM_MALLOC_INTERNAL_CALLER);
        if (NULL != d && LIBXSMM_NEQ(0, *beta)) memcpy(d, c, size); /* copy destination */
      }
#endif
      if (0 != (libxsmm_gemm_wrap & 1)) { /* sequential */
        libxsmm_dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
      }
      else { /* parallelized */
        libxsmm_dgemm_omp(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
      }
#if defined(_DEBUG)
      if (NULL != d) {
        libxsmm_matdiff_info diff;
        libxsmm_blas_dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, d, ldc);
        if (EXIT_SUCCESS == libxsmm_matdiff(&diff, LIBXSMM_DATATYPE_F64, *m, *n, d, c, ldc, ldc)
          && check < 100.0 * diff.normf_rel)
        {
          LIBXSMM_STDIO_ACQUIRE();
          fprintf(stderr, "LIBXSMM: ");
          libxsmm_gemm_print(stderr, LIBXSMM_DATATYPE_F64, transa, transb,
            m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
          fprintf(stderr, " => %f%% ERROR\n", 100.0 * diff.normf_rel);
          LIBXSMM_STDIO_RELEASE();
        }
        libxsmm_free(d);
      }
#endif
#if defined(LIBXSMM_WRAP) && defined(LIBXSMM_BUILD_EXT)
      if (0 != (LIBXSMM_MMBATCH_FLAG_STATISTIC & libxsmm_mmbatch_desc.flags)) {
        libxsmm_descriptor_blob blob;
        const libxsmm_gemm_descriptor *const descriptor = libxsmm_dgemm_descriptor_init(&blob,
          *m, *n, *k, *lda, *ldb, *ldc, *alpha, *beta, LIBXSMM_GEMM_FLAGS(*transa, *transb),
          LIBXSMM_EXT_GEMM_MMBATCH_PREFETCH);

        LIBXSMM_ASSERT(0 != libxsmm_mmbatch_size);
        if (NULL != descriptor) {
          const unsigned int max_batchsize = (unsigned int)((LIBXSMM_GEMM_MMBATCH_SCALE) * libxsmm_mmbatch_size);
          const unsigned int batchsize = LIBXSMM_ATOMIC_LOAD(&internal_ext_gemm_batchsize, LIBXSMM_ATOMIC_RELAXED);
          const unsigned int max_size = (0 != batchsize ? (((batchsize - 1) % max_batchsize) + 1) : 0);
          libxsmm_mmbatch_item *const batcharray = (libxsmm_mmbatch_item*)libxsmm_mmbatch_array;
          libxsmm_mmbatch_item* batcharray_cur = batcharray;
          unsigned int size = max_size;
          if (libxsmm_mmbatch_size < max_size) {
            size = max_size - libxsmm_mmbatch_size;
            batcharray_cur += libxsmm_mmbatch_size;
          }
          i = libxsmm_diff_n(descriptor, batcharray_cur, sizeof(libxsmm_gemm_descriptor),
            sizeof(libxsmm_mmbatch_item)/*stride*/, 0/*hint*/, size);

          if (i < size) { /* update existing entry */
            LIBXSMM_ATOMIC_ADD_FETCH(&batcharray_cur[i].stat.count, 1, LIBXSMM_ATOMIC_RELAXED);
          }
          else { /* new entry needed */
            const int all = -1, shift = 0;
            void* extra = 0;
            i = ((LIBXSMM_ATOMIC_ADD_FETCH(&internal_ext_gemm_batchsize, 1, LIBXSMM_ATOMIC_RELAXED) - 1) % max_batchsize) + 1;
            batcharray[i-1].stat.desc = *descriptor;
            batcharray[i-1].stat.count = 1;
            batcharray[i-1].stat.symbol = libxsmm_trace_info(NULL/*depth*/, NULL/*tid*/, &all, LIBXSMM_FUNCNAME, &shift, &all);
            if (EXIT_SUCCESS == libxsmm_get_malloc_xinfo(libxsmm_mmbatch_array, NULL/*size*/, NULL/*flags*/, &extra)) {
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
#if defined(LIBXSMM_WRAP) && defined(LIBXSMM_BUILD_EXT)
    else {
      libxsmm_mmbatch_item *const batcharray = (libxsmm_mmbatch_item*)libxsmm_mmbatch_array;
      const unsigned int max_batchsize = (unsigned int)((LIBXSMM_GEMM_MMBATCH_SCALE) * libxsmm_mmbatch_size);
      i = ((LIBXSMM_ATOMIC_ADD_FETCH(&internal_ext_gemm_batchsize, 1, LIBXSMM_ATOMIC_RELAXED) - 1) % max_batchsize) + 1;
      batcharray[i-1].value.a = a;
      batcharray[i-1].value.b = b;
      batcharray[i-1].value.c = c;
      LIBXSMM_ASSERT(0 <= flags);
    }
    if (libxsmm_mmbatch_size == (i - 1)) { /* condition ensure to flush once (first discovery) */
# if !defined(NDEBUG)
      result =
# endif
      internal_mmbatch_flush(&libxsmm_mmbatch_desc, libxsmm_mmbatch_size, (libxsmm_mmbatch_item*)libxsmm_mmbatch_array);
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


LIBXSMM_APIEXT LIBXSMM_ATTRIBUTE_USED void LIBXSMM_FSYMBOL(__wrap_sgemm)(
  const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const float* alpha, const float* a, const libxsmm_blasint* lda,
  const float* b, const libxsmm_blasint* ldb,
  const float* beta, float* c, const libxsmm_blasint* ldc)
{
  LIBXSMM_ASSERT(NULL != lda && NULL != ldb && NULL != ldc && NULL != m && NULL != n && NULL != k);
  LIBXSMM_ASSERT(NULL != transa && NULL != transb && NULL != alpha && NULL != beta);
  {
#if defined(LIBXSMM_WRAP) && defined(LIBXSMM_BUILD_EXT)
    unsigned int i = 0; /* no flush */
    int flags = -1;
# if !defined(NDEBUG)
    static int error_once = 0;
    int result = EXIT_SUCCESS;
# endif
    LIBXSMM_INIT
    if (0 != libxsmm_gemm_wrap && (NULL == libxsmm_mmbatch_array
      || LIBXSMM_DATATYPE_F32 != libxsmm_mmbatch_desc.datatype
      || ((unsigned int)*lda) != libxsmm_mmbatch_desc.lda
      || ((unsigned int)*ldb) != libxsmm_mmbatch_desc.ldb
      || ((unsigned int)*ldc) != libxsmm_mmbatch_desc.ldc
      || ((unsigned int)*m) != libxsmm_mmbatch_desc.m
      || ((unsigned int)*n) != libxsmm_mmbatch_desc.n
      || ((unsigned int)*k) != libxsmm_mmbatch_desc.k
      || (flags = LIBXSMM_GEMM_FLAGS(*transa, *transb)) != (int)(LIBXSMM_GEMM_FLAG_TRANS_AB & libxsmm_mmbatch_desc.flags)
      || LIBXSMM_NEQ(1, *alpha) || LIBXSMM_NEQ(0 != (LIBXSMM_GEMM_FLAG_BETA_0 & libxsmm_mmbatch_desc.flags) ? 0 : 1, *beta)))
#endif
    {
#if defined(_DEBUG)
      const char *const env_check = getenv("LIBXSMM_GEMM_CHECK");
      const double check = LIBXSMM_ABS(NULL == env_check ? 0 : atof(env_check));
      void* d = NULL;
      if (LIBXSMM_NEQ(0, check)) {
        const size_t size = (size_t)(*ldc) * (size_t)(*n) * sizeof(float);
        d = libxsmm_scratch_malloc(size, 0/*auto*/, LIBXSMM_MALLOC_INTERNAL_CALLER);
        if (NULL != d && LIBXSMM_NEQ(0, *beta)) memcpy(d, c, size); /* copy destination */
      }
#endif
      if (0 != (libxsmm_gemm_wrap & 1)) { /* sequential */
        libxsmm_sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
      }
      else { /* parallelized */
        libxsmm_sgemm_omp(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
      }
#if defined(_DEBUG)
      if (NULL != d) {
        libxsmm_matdiff_info diff;
        libxsmm_blas_sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, d, ldc);
        if (EXIT_SUCCESS == libxsmm_matdiff(&diff, LIBXSMM_DATATYPE_F32, *m, *n, d, c, ldc, ldc)
          && check < 100.0 * diff.normf_rel)
        {
          LIBXSMM_STDIO_ACQUIRE();
          fprintf(stderr, "LIBXSMM: ");
          libxsmm_gemm_print(stderr, LIBXSMM_DATATYPE_F32, transa, transb,
            m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
          fprintf(stderr, " => %f%% ERROR\n", 100.0 * diff.normf_rel);
          LIBXSMM_STDIO_RELEASE();
        }
        libxsmm_free(d);
      }
#endif
#if defined(LIBXSMM_WRAP) && defined(LIBXSMM_BUILD_EXT)
      if (0 != (LIBXSMM_MMBATCH_FLAG_STATISTIC & libxsmm_mmbatch_desc.flags)) {
        libxsmm_descriptor_blob blob;
        const libxsmm_gemm_descriptor *const descriptor = libxsmm_sgemm_descriptor_init(&blob,
          *m, *n, *k, *lda, *ldb, *ldc, *alpha, *beta, LIBXSMM_GEMM_FLAGS(*transa, *transb),
          LIBXSMM_EXT_GEMM_MMBATCH_PREFETCH);

        LIBXSMM_ASSERT(0 != libxsmm_mmbatch_size);
        if (NULL != descriptor) {
          const unsigned int max_batchsize = (unsigned int)((LIBXSMM_GEMM_MMBATCH_SCALE) * libxsmm_mmbatch_size);
          const unsigned int batchsize = LIBXSMM_ATOMIC_LOAD(&internal_ext_gemm_batchsize, LIBXSMM_ATOMIC_RELAXED);
          const unsigned int max_size = (0 != batchsize ? (((batchsize - 1) % max_batchsize) + 1) : 0);
          libxsmm_mmbatch_item *const batcharray = (libxsmm_mmbatch_item*)libxsmm_mmbatch_array;
          libxsmm_mmbatch_item* batcharray_cur = batcharray;
          unsigned int size = max_size;
          if (libxsmm_mmbatch_size < max_size) {
            size = max_size - libxsmm_mmbatch_size;
            batcharray_cur += libxsmm_mmbatch_size;
          }
          i = libxsmm_diff_n(descriptor, batcharray_cur, sizeof(libxsmm_gemm_descriptor),
            sizeof(libxsmm_mmbatch_item)/*stride*/, 0/*hint*/, size);

          if (i < size) { /* update existing entry */
            LIBXSMM_ATOMIC_ADD_FETCH(&batcharray_cur[i].stat.count, 1, LIBXSMM_ATOMIC_RELAXED);
          }
          else { /* new entry needed */
            const int all = -1, shift = 0;
            void* extra = 0;
            i = ((LIBXSMM_ATOMIC_ADD_FETCH(&internal_ext_gemm_batchsize, 1, LIBXSMM_ATOMIC_RELAXED) - 1) % max_batchsize) + 1;
            batcharray[i-1].stat.desc = *descriptor;
            batcharray[i-1].stat.count = 1;
            batcharray[i-1].stat.symbol = libxsmm_trace_info(NULL/*depth*/, NULL/*tid*/, &all, LIBXSMM_FUNCNAME, &shift, &all);
            if (EXIT_SUCCESS == libxsmm_get_malloc_xinfo(libxsmm_mmbatch_array, NULL/*size*/, NULL/*flags*/, &extra)) {
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
#if defined(LIBXSMM_WRAP) && defined(LIBXSMM_BUILD_EXT)
    else {
      libxsmm_mmbatch_item *const batcharray = (libxsmm_mmbatch_item*)libxsmm_mmbatch_array;
      const unsigned int max_batchsize = (unsigned int)((LIBXSMM_GEMM_MMBATCH_SCALE) * libxsmm_mmbatch_size);
      i = ((LIBXSMM_ATOMIC_ADD_FETCH(&internal_ext_gemm_batchsize, 1, LIBXSMM_ATOMIC_RELAXED) - 1) % max_batchsize) + 1;
      batcharray[i-1].value.a = a;
      batcharray[i-1].value.b = b;
      batcharray[i-1].value.c = c;
      LIBXSMM_ASSERT(0 <= flags);
    }
    if (libxsmm_mmbatch_size == (i - 1)) { /* condition ensure to flush once (first discovery) */
# if !defined(NDEBUG)
      result =
# endif
      internal_mmbatch_flush(&libxsmm_mmbatch_desc, libxsmm_mmbatch_size, (libxsmm_mmbatch_item*)libxsmm_mmbatch_array);
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


LIBXSMM_APIEXT LIBXSMM_ATTRIBUTE_USED void LIBXSMM_FSYMBOL(__wrap_dgemv)(const char* trans, const libxsmm_blasint* m, const libxsmm_blasint* n,
  const double* alpha, const double* a, const libxsmm_blasint* lda, const double* x, const libxsmm_blasint* incx,
  const double* beta, double* y, const libxsmm_blasint* incy)
{
  LIBXSMM_ASSERT(NULL != trans && NULL != m && NULL != n && NULL != lda && NULL != incx && NULL != incy && NULL != alpha && NULL != beta);
  LIBXSMM_INIT
  if ((2 < libxsmm_gemm_wrap || 2 > libxsmm_gemm_wrap) && 1 == *incx && 1 == *incy && LIBXSMM_SMM(*m, 1, *n, 2/*RFO*/, sizeof(double))) {
    if (0 != (libxsmm_gemm_wrap & 1)) { /* sequential */
      const int flags = LIBXSMM_GEMM_FLAGS(*trans, 'N');
      const libxsmm_dmmfunction xgemv = libxsmm_dmmdispatch(*m, 1, *n, lda, n/*ldb*/, m/*ldc*/, alpha, beta, &flags, NULL);
      if (NULL != xgemv) {
        LIBXSMM_MMCALL_LDX(xgemv, a, x, y, *m, 1, *n, *lda, *n/*ldb*/, *m/*ldc*/);
      }
      else {
        LIBXSMM_GEMV_SYMBOL(double)(trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
      }
    }
    else { /* TODO: parallelized */
      LIBXSMM_GEMV_SYMBOL(double)(trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
    }
  }
  else {
    LIBXSMM_GEMV_SYMBOL(double)(trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
  }
}


LIBXSMM_APIEXT LIBXSMM_ATTRIBUTE_USED void LIBXSMM_FSYMBOL(__wrap_sgemv)(const char* trans, const libxsmm_blasint* m, const libxsmm_blasint* n,
  const float* alpha, const float* a, const libxsmm_blasint* lda, const float* x, const libxsmm_blasint* incx,
  const float* beta, float* y, const libxsmm_blasint* incy)
{
  LIBXSMM_ASSERT(NULL != trans && NULL != m && NULL != n && NULL != lda && NULL != incx && NULL != incy && NULL != alpha && NULL != beta);
  LIBXSMM_INIT
  if ((2 < libxsmm_gemm_wrap || 2 > libxsmm_gemm_wrap) && 1 == *incx && 1 == *incy && LIBXSMM_SMM(*m, 1, *n, 2/*RFO*/, sizeof(float))) {
    if (0 != (libxsmm_gemm_wrap & 1)) { /* sequential */
      const int flags = LIBXSMM_GEMM_FLAGS(*trans, 'N');
      const libxsmm_smmfunction xgemv = libxsmm_smmdispatch(*m, 1, *n, lda, n/*ldb*/, m/*ldc*/, alpha, beta, &flags, NULL);
      if (NULL != xgemv) {
        LIBXSMM_MMCALL_LDX(xgemv, a, x, y, *m, 1, *n, *lda, *n/*ldb*/, *m/*ldc*/);
      }
      else {
        LIBXSMM_GEMV_SYMBOL(float)(trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
      }
    }
    else { /* TODO: parallelized */
      LIBXSMM_GEMV_SYMBOL(float)(trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
    }
  }
  else {
    LIBXSMM_GEMV_SYMBOL(float)(trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
  }
}


LIBXSMM_APIEXT LIBXSMM_ATTRIBUTE_USED void __wrap_dgemm_batch(
  const char transa_array[], const char transb_array[], const libxsmm_blasint m_array[], const libxsmm_blasint n_array[], const libxsmm_blasint k_array[],
  const double alpha_array[], const double* a_array[], const libxsmm_blasint lda_array[], const double* b_array[], const libxsmm_blasint ldb_array[],
  const double beta_array[], double* c_array[], const libxsmm_blasint ldc_array[], const libxsmm_blasint* group_count, const libxsmm_blasint group_size[])
{
  LIBXSMM_FSYMBOL(__wrap_dgemm_batch)(transa_array, transb_array, m_array, n_array, k_array,
    alpha_array, a_array, lda_array, b_array, ldb_array, beta_array, c_array, ldc_array,
    group_count, group_size);
}


LIBXSMM_APIEXT LIBXSMM_ATTRIBUTE_USED void __wrap_sgemm_batch(
  const char transa_array[], const char transb_array[], const libxsmm_blasint m_array[], const libxsmm_blasint n_array[], const libxsmm_blasint k_array[],
  const float alpha_array[], const float* a_array[], const libxsmm_blasint lda_array[], const float* b_array[], const libxsmm_blasint ldb_array[],
  const float beta_array[], float* c_array[], const libxsmm_blasint ldc_array[], const libxsmm_blasint* group_count, const libxsmm_blasint group_size[])
{
  LIBXSMM_FSYMBOL(__wrap_sgemm_batch)(transa_array, transb_array, m_array, n_array, k_array,
    alpha_array, a_array, lda_array, b_array, ldb_array, beta_array, c_array, ldc_array,
    group_count, group_size);
}

#endif /*defined(LIBXSMM_BUILD) && defined(LIBXSMM_BUILD_EXT)*/


LIBXSMM_APIEXT void libxsmm_xgemm_omp(libxsmm_datatype iprec, libxsmm_datatype oprec,
  const char* transa, const char* transb, const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const void* alpha, const void* a, const libxsmm_blasint* lda, const void* b, const libxsmm_blasint* ldb,
  const void* beta, void* c, const libxsmm_blasint* ldc)
{
  libxsmm_gemm_blob blob;
#if defined(LIBXSMM_EXT_TASKS) /* implies _OPENMP */
  const int outerpar = omp_get_active_level(), nthreads = (0 == outerpar ? omp_get_max_threads() : omp_get_num_threads());
#elif defined(_OPENMP)
  const int outerpar = omp_in_parallel(), nthreads = (0 == outerpar ? omp_get_max_threads() : 1);
#else
  const int nthreads = 1;
#endif
  const libxsmm_gemm_handle *const handle = libxsmm_gemm_handle_init(&blob, iprec, oprec, transa, transb,
    m, n, k, lda, ldb, ldc, alpha, beta, LIBXSMM_GEMM_HANDLE_FLAG_AUTO, nthreads);
  const size_t scratch_size = libxsmm_gemm_handle_get_scratch_size(handle);
  void* scratch = NULL;
  if (NULL != handle && (0 == scratch_size ||
      NULL != (scratch = libxsmm_scratch_malloc(scratch_size, LIBXSMM_CACHELINE, LIBXSMM_MALLOC_INTERNAL_CALLER))))
  {
#if defined(_OPENMP)
    if (0 == outerpar) { /* enable internal parallelization */
# if defined(LIBXSMM_EXT_TASKS)
      if (0 == libxsmm_gemm_tasks)
# endif
      {
#       pragma omp parallel num_threads(nthreads)
        libxsmm_gemm_task(handle, scratch, a, b, c, omp_get_thread_num(), nthreads);
      }
# if defined(LIBXSMM_EXT_TASKS)
      else { /* tasks requested */
        const int ntasks = nthreads; /* TODO: apply grain-size */
#       pragma omp parallel num_threads(nthreads)
        { /* first thread discovering work will launch all tasks */
#         pragma omp single nowait /* anyone is good */
          { int tid; for (tid = 0; tid < ntasks; ++tid) {
#             pragma omp task untied
              libxsmm_gemm_task(handle, scratch, a, b, c, tid, ntasks);
            }
          }
        } /* implicit synchronization (barrier) */
      }
# endif
    }
    else { /* assume external parallelization */
# if defined(LIBXSMM_EXT_TASKS) /* implies _OPENMP */
      const int ntasks = nthreads; /* TODO: apply grain-size */
      int tid; for (tid = 0; tid < ntasks; ++tid) {
#       pragma omp task untied
        libxsmm_gemm_task(handle, scratch, a, b, c, tid, ntasks);
      }
      if (0 == libxsmm_nosync) { /* allow to omit synchronization */
#       pragma omp taskwait
      }
# else
      libxsmm_gemm_task(handle, scratch, a, b, c, 0/*tid*/, 1/*nthreads*/);
# endif
    }
    if (LIBXSMM_VERBOSITY_HIGH <= libxsmm_verbosity || 0 > libxsmm_verbosity) { /* library code is expected to be mute */
      const unsigned int ntasks = handle->mt * handle->nt * handle->kt;
      const double imbalance = 100.0 * LIBXSMM_DELTA((unsigned int)nthreads, ntasks) / nthreads;
      static double max_imbalance = 50.0;
      if (max_imbalance < imbalance) {
        fprintf(stderr, "LIBXSMM WARNING: XGEMM %.0f%% imbalance (%u of %i workers utilized)!\n",
          imbalance, ntasks, nthreads);
        max_imbalance = imbalance;
      }
    }
#else
    libxsmm_gemm_task(handle, scratch, a, b, c, 0/*tid*/, 1/*nthreads*/);
#endif /*defined(_OPENMP)*/
    libxsmm_free(scratch);
  }
  else { /* fallback or error */
    static int error_once = 0;
    if (NULL == handle) { /* fallback */
      if ((LIBXSMM_VERBOSITY_HIGH <= libxsmm_verbosity || 0 > libxsmm_verbosity) /* library code is expected to be mute */
        && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
      {
        fprintf(stderr, "LIBXSMM WARNING: XGEMM fallback code path triggered!\n");
      }
    }
    else if (0 != libxsmm_verbosity && /* library code is expected to be mute */
      1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
    {
      fprintf(stderr, "LIBXSMM ERROR: failed to allocate GEMM-scratch memory!\n");
    }
    libxsmm_blas_xgemm(iprec, oprec, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  }
}


LIBXSMM_API_INLINE void internal_gemm_batch_omp(libxsmm_datatype iprec, libxsmm_datatype oprec,
  const char transa[], const char transb[], const libxsmm_blasint m[], const libxsmm_blasint n[], const libxsmm_blasint k[],
  const void* alpha, const void* a[], const libxsmm_blasint lda[], const void* b[], const libxsmm_blasint ldb[],
  const void* beta, void* c[], const libxsmm_blasint ldc[], libxsmm_blasint index_base, libxsmm_blasint index_stride,
  const libxsmm_blasint stride_a[], const libxsmm_blasint stride_b[], const libxsmm_blasint stride_c[],
  const libxsmm_blasint batchsize[], libxsmm_blasint group_count)
{
  static int error_once = 0;
  LIBXSMM_INIT
  if ( /* check for sensible arguments */
#if defined(LIBXSMM_BATCH_CHECK)
    NULL != a && NULL != b && NULL != c && (1 == group_count || -1 == group_count ||
    (0 == index_stride && (NULL == stride_a || 0 != *stride_a) && (NULL == stride_b || 0 != *stride_b) && (NULL == stride_c || 0 != *stride_c))) &&
#endif
    0 != group_count)
  {
    int result = EXIT_SUCCESS;
    const int max_npargroups = (int)(0 < libxsmm_gemm_npargroups
      ? LIBXSMM_MIN(libxsmm_gemm_npargroups, LIBXSMM_GEMM_NPARGROUPS) : LIBXSMM_GEMM_NPARGROUPS);
    const libxsmm_gemm_prefetch_type prefetch = libxsmm_get_gemm_prefetch(LIBXSMM_PREFETCH_AUTO);
    const size_t sa = (NULL != stride_a ? (size_t)(*stride_a) : sizeof(void*));
    const size_t sb = (NULL != stride_b ? (size_t)(*stride_b) : sizeof(void*));
    const size_t sc = (NULL != stride_c ? (size_t)(*stride_c) : sizeof(void*));
    const unsigned char otypesize = libxsmm_typesize((libxsmm_datatype)oprec);
    const int ngroups = (int)LIBXSMM_ABS(group_count);
    int group = 0, group_next = LIBXSMM_GEMM_NPARGROUPS;
    libxsmm_code_pointer kernel[LIBXSMM_GEMM_NPARGROUPS];
    libxsmm_blasint base[LIBXSMM_GEMM_NPARGROUPS], i;
#if !defined(LIBXSMM_EXT_GEMM_PARGROUPS_INFO)
    int kflags[LIBXSMM_GEMM_NPARGROUPS];
#endif
    int max_nthreads = 1;
#if defined(_OPENMP)
# if defined(LIBXSMM_EXT_TASKS)
    const int outerpar = omp_get_active_level();
# else
    const int outerpar = omp_in_parallel();
# endif
    if (0 == outerpar) max_nthreads = omp_get_max_threads();
#endif
    for (i = 0; i < max_npargroups; ++i) {
#if !defined(NDEBUG)
      kernel[i].ptr = NULL;
# if !defined(LIBXSMM_EXT_GEMM_PARGROUPS_INFO)
      kflags[i] = 0;
# endif
#endif
      base[i] = 0;
    }
    for (group = 0; group < ngroups; group = group_next, group_next += max_npargroups) {
      const int npargroups = LIBXSMM_MIN(group_next, ngroups);
      libxsmm_blasint size = 0;
      int suitable = 0;
      if (0 < group) { /* base is maintained even if par-group is not suitable */
        for (i = 0; i < npargroups; ++i) {
          const libxsmm_blasint isize = batchsize[group+i-1], asize = LIBXSMM_ABS(isize);
          base[i] += asize;
        }
      }
      for (i = 0; i < npargroups; ++i) {
        const libxsmm_blasint g = group + i, im = m[g], in = n[g], ik = k[g];
        suitable = LIBXSMM_SMM_AI(im, in, ik, 2/*RFO*/, otypesize);
        if (0 != suitable) {
          const libxsmm_blasint isize = batchsize[g], asize = LIBXSMM_ABS(isize);
          const char *const ta = (NULL != transa ? (transa + g) : NULL);
          const char *const tb = (NULL != transb ? (transb + g) : NULL);
          const int flags = LIBXSMM_GEMM_PFLAGS(ta, tb, LIBXSMM_FLAGS);
          const void **const galpha = &alpha, **const gbeta = &beta;
          libxsmm_descriptor_blob blob;
          /* coverity[ptr_arith] */
          libxsmm_gemm_descriptor *const desc = libxsmm_gemm_descriptor_init2(&blob, iprec, oprec, im, in, ik,
            NULL != lda ? lda[g] : (0 == (LIBXSMM_GEMM_FLAG_TRANS_A & flags) ? im : ik),
            NULL != ldb ? ldb[g] : (0 == (LIBXSMM_GEMM_FLAG_TRANS_B & flags) ? ik : in),
            NULL != ldc ? ldc[g] : im, NULL != alpha ? galpha[g] : NULL, NULL != beta ? gbeta[g] : NULL,
            flags, prefetch);
          if (NULL != desc) {
            libxsmm_gemm_internal_set_batchflag(desc, c, index_stride, 0 < group_count ? isize : -asize, 1 != max_nthreads);
            kernel[i].xgemm = libxsmm_xmmdispatch(desc);
          }
          else kernel[i].ptr = NULL;
          if (NULL != kernel[i].ptr_const) {
            if (size < asize) size = asize;
#if !defined(LIBXSMM_EXT_GEMM_PARGROUPS_INFO)
            LIBXSMM_ASSERT(NULL != desc); /* coverity[var_deref_op] */
            kflags[i] = desc->flags;
#endif
          }
          else {
            suitable = 0;
            break;
          }
        }
        else break;
      }
      if (0 != suitable) { /* check if an SMM is suitable */
        const unsigned char itypesize = libxsmm_typesize((libxsmm_datatype)iprec);
#if defined(_OPENMP)
        const int nchunks = (int)LIBXSMM_UPDIV(size, libxsmm_gemm_taskgrain);
        const int ntasks = nchunks * npargroups, nthreads = LIBXSMM_MIN(max_nthreads, ntasks);
        if (1 < nthreads) {
          if (0 == outerpar) { /* enable internal parallelization */
# if defined(LIBXSMM_EXT_TASKS)
            if (0 == libxsmm_gemm_tasks)
# endif
            {
#             pragma omp parallel for num_threads(nthreads) private(i)
              for (i = 0; i < ntasks; ++i) {
                const libxsmm_blasint j = i * libxsmm_gemm_taskgrain, u = j / size, v = j - u * size, g = group + u;
                const libxsmm_blasint isize = batchsize[g], asize = LIBXSMM_ABS(isize);
                if (v < asize) {
#if defined(LIBXSMM_EXT_GEMM_PARGROUPS_INFO)
                  libxsmm_mmkernel_info kernel_info;
#endif
                  /*check*/libxsmm_mmbatch_kernel(kernel[g].xgemm, index_base, index_stride, stride_a, stride_b, stride_c,
                    (const char*)a + sa * base[u], (const char*)b + sb * base[u], (char*)c + sc * base[u],
                    0 < group_count ? isize : -asize, (int)i, nchunks, itypesize, otypesize,
#if defined(LIBXSMM_EXT_GEMM_PARGROUPS_INFO)
                    EXIT_SUCCESS == libxsmm_get_mmkernel_info(kernel[g].xgemm, &kernel_info) ? kernel_info.flags : 0);
#else
                    kflags[g]);
#endif
                }
              }
            }
# if defined(LIBXSMM_EXT_TASKS)
            else { /* tasks requested */
#             pragma omp parallel num_threads(nthreads) private(i)
              { /* first thread discovering work will launch all tasks */
#               pragma omp single nowait /* anyone is good */
                for (i = 0; i < ntasks; ++i) {
                  const libxsmm_blasint j = i * libxsmm_gemm_taskgrain, u = j / size, v = j - u * size, g = group + u;
                  const libxsmm_blasint isize = batchsize[g], asize = LIBXSMM_ABS(isize);
                  if (v < asize) {
#                   pragma omp task
                    {
#if defined(LIBXSMM_EXT_GEMM_PARGROUPS_INFO)
                      libxsmm_mmkernel_info kernel_info;
#endif
                      /*check*/libxsmm_mmbatch_kernel(kernel[g].xgemm, index_base, index_stride, stride_a, stride_b, stride_c,
                        (const char*)a + sa * base[u], (const char*)b + sb * base[u], (char*)c + sc * base[u],
                        0 < group_count ? isize : -asize, (int)i, nchunks, itypesize, otypesize,
#if defined(LIBXSMM_EXT_GEMM_PARGROUPS_INFO)
                        EXIT_SUCCESS == libxsmm_get_mmkernel_info(kernel[g].xgemm, &kernel_info) ? kernel_info.flags : 0);
#else
                        kflags[g]);
#endif
                    }
                  }
                }
              } /* implicit synchronization (barrier) */
            }
# endif
          }
          else { /* assume external parallelization */
            for (i = 0; i < (libxsmm_blasint)ntasks; ++i) {
              const libxsmm_blasint j = i * libxsmm_gemm_taskgrain, u = j / size, v = j - u * size, g = group + u;
              const libxsmm_blasint isize = batchsize[g], asize = LIBXSMM_ABS(isize);
              if (v < asize) {
# if defined(LIBXSMM_EXT_TASKS) /* OpenMP-tasks */
#               pragma omp task
#endif
                {
#if defined(LIBXSMM_EXT_GEMM_PARGROUPS_INFO)
                  libxsmm_mmkernel_info kernel_info;
#endif
                  /*check*/libxsmm_mmbatch_kernel(kernel[g].xgemm, index_base, index_stride, stride_a, stride_b, stride_c,
                    (const char*)a + sa * base[u], (const char*)b + sb * base[u], (char*)c + sc * base[u],
                    0 < group_count ? isize : -asize, (int)i, nchunks, itypesize, otypesize,
#if defined(LIBXSMM_EXT_GEMM_PARGROUPS_INFO)
                    EXIT_SUCCESS == libxsmm_get_mmkernel_info(kernel[g].xgemm, &kernel_info) ? kernel_info.flags : 0);
#else
                    kflags[g]);
#endif
                }
              }
            }
# if defined(LIBXSMM_EXT_TASKS) /* OpenMP-tasks */
            if (0 == libxsmm_nosync) { /* allow to omit synchronization */
#             pragma omp taskwait
            }
# endif
          }
        }
        else
#endif /*defined(_OPENMP)*/
        { /* sequential */
          for (i = 0; i < npargroups; ++i) {
            const libxsmm_blasint g = group + i;
#if defined(LIBXSMM_EXT_GEMM_PARGROUPS_INFO)
            libxsmm_mmkernel_info kernel_info;
#endif
            libxsmm_mmbatch_kernel(kernel[i].xgemm, index_base, index_stride, stride_a, stride_b, stride_c,
              (const char*)a + sa * base[i], (const char*)b + sb * base[i], (char*)c + sc * base[i], batchsize[g],
              0/*tid*/, 1/*nthreads*/, itypesize, otypesize,
#if defined(LIBXSMM_EXT_GEMM_PARGROUPS_INFO)
              EXIT_SUCCESS == libxsmm_get_mmkernel_info(kernel[i].xgemm, &kernel_info) ? kernel_info.flags : 0);
#else
              kflags[i]);
#endif
          }
        }
      }
      else { /* trigger fallback */
        result = EXIT_FAILURE;
      }
      if (EXIT_SUCCESS != result) {
        for (i = 0; i < npargroups; ++i) {
          const libxsmm_blasint g = group + i;
          const char *const ta = (NULL != transa ? (transa + g) : NULL);
          const char *const tb = (NULL != transb ? (transb + g) : NULL);
          const int flags = LIBXSMM_GEMM_PFLAGS(ta, tb, LIBXSMM_FLAGS);
          const libxsmm_blasint im = m[g], in = n[g], ik = k[g];
          const libxsmm_blasint ilda = (NULL != lda ? lda[g] : (0 == (LIBXSMM_GEMM_FLAG_TRANS_A & flags) ? im : ik));
          const libxsmm_blasint ildb = (NULL != ldb ? ldb[g] : (0 == (LIBXSMM_GEMM_FLAG_TRANS_B & flags) ? ik : in));
          const libxsmm_blasint ildc = (NULL != ldc ? ldc[g] : im);
          const void **const galpha = &alpha, **const gbeta = &beta;
          /* coverity[overrun-local] */
          const void *const ialpha = (NULL != alpha ? galpha[g] : NULL);
          /* coverity[overrun-local] */
          const void *const ibeta = (NULL != beta ? gbeta[g] : NULL);
          if (EXIT_SUCCESS == libxsmm_mmbatch_blas(iprec, oprec, ta, tb, im, in, ik, ialpha,
            (const char*)a + sa * base[i], &ilda, (const char*)b + sb * base[i], &ildb, ibeta, (char*)c + sc * base[i], &ildc,
            index_base, index_stride, stride_a, stride_b, stride_c, batchsize[g]))
          {
            if (LIBXSMM_VERBOSITY_WARN <= libxsmm_verbosity || 0 > libxsmm_verbosity) {
              const size_t threshold = LIBXSMM_MNK_SIZE(im, in, im);
              static size_t threshold_max = 0;
              if (threshold_max < threshold) {
                LIBXSMM_STDIO_ACQUIRE();
                fprintf(stderr, "LIBXSMM WARNING: ");
                libxsmm_gemm_print2(stderr, iprec, oprec, ta, tb, &im, &in, &ik,
                  ialpha, NULL/*a*/, &ilda, NULL/*b*/, &ildb, ibeta, NULL/*c*/, &ildc);
                fprintf(stderr, " => batched GEMM/omp was falling back!\n");
                LIBXSMM_STDIO_RELEASE();
                threshold_max = threshold;
              }
            }
          }
          else {
            if (0 != libxsmm_verbosity /* library code is expected to be mute */
              && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
            {
              fprintf(stderr, "LIBXSMM ERROR: libxsmm_gemm_batch_omp failed!\n");
            }
            return; /* exit routine */
          }
        }
      }
    }
  }
#if defined(LIBXSMM_BATCH_CHECK)
  else if (0 != group_count && 0 != libxsmm_verbosity /* library code is expected to be mute */
    && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
  {
    fprintf(stderr, "LIBXSMM ERROR: incorrect arguments (libxsmm_gemm_batch_omp)!\n");
  }
#endif
}


LIBXSMM_APIEXT void libxsmm_gemm_batch_omp(libxsmm_datatype iprec, libxsmm_datatype oprec,
  const char* transa, const char* transb, libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const void* alpha, const void* a, const libxsmm_blasint* lda, const void* b, const libxsmm_blasint* ldb,
  const void* beta, void* c, const libxsmm_blasint* ldc, libxsmm_blasint index_base, libxsmm_blasint index_stride,
  const libxsmm_blasint stride_a[], const libxsmm_blasint stride_b[], const libxsmm_blasint stride_c[],
  libxsmm_blasint batchsize)
{
  internal_gemm_batch_omp(iprec, oprec, transa, transb, &m, &n, &k,
    alpha, (const void**)a, lda, (const void**)b, ldb, beta, (void**)c, ldc, index_base, index_stride,
    stride_a, stride_b, stride_c, &batchsize, 1);
}


LIBXSMM_APIEXT void libxsmm_dgemm_batch_omp(
  const char transa_array[], const char transb_array[], const libxsmm_blasint m_array[], const libxsmm_blasint n_array[], const libxsmm_blasint k_array[],
  const double alpha_array[], const double* a_array[], const libxsmm_blasint lda_array[], const double* b_array[], const libxsmm_blasint ldb_array[],
  const double beta_array[], double* c_array[], const libxsmm_blasint ldc_array[], const libxsmm_blasint* group_count, const libxsmm_blasint group_size[])
{
  if (NULL != group_count) {
    const libxsmm_blasint ptrsize = sizeof(void*);
    internal_gemm_batch_omp(LIBXSMM_DATATYPE_F64, LIBXSMM_DATATYPE_F64, transa_array, transb_array, m_array, n_array, k_array,
      alpha_array, (const void**)a_array, lda_array, (const void**)b_array, ldb_array, beta_array, (void**)c_array, ldc_array,
      0/*index_base*/, 0/*index_stride*/, &ptrsize, &ptrsize, &ptrsize, group_size, *group_count);
  }
}


LIBXSMM_APIEXT void libxsmm_sgemm_batch_omp(
  const char transa_array[], const char transb_array[], const libxsmm_blasint m_array[], const libxsmm_blasint n_array[], const libxsmm_blasint k_array[],
  const float alpha_array[], const float* a_array[], const libxsmm_blasint lda_array[], const float* b_array[], const libxsmm_blasint ldb_array[],
  const float beta_array[], float* c_array[], const libxsmm_blasint ldc_array[], const libxsmm_blasint* group_count, const libxsmm_blasint group_size[])
{
  if (NULL != group_count) {
    const libxsmm_blasint ptrsize = sizeof(void*);
    internal_gemm_batch_omp(LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, transa_array, transb_array, m_array, n_array, k_array,
      alpha_array, (const void**)a_array, lda_array, (const void**)b_array, ldb_array, beta_array, (void**)c_array, ldc_array,
      0/*index_base*/, 0/*index_stride*/, &ptrsize, &ptrsize, &ptrsize, group_size, *group_count);
  }
}


LIBXSMM_APIEXT void libxsmm_mmbatch_begin(libxsmm_datatype precision,
  const int* flags, const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc,
  const void* alpha, const void* beta)
{
#if defined(LIBXSMM_WRAP) && defined(LIBXSMM_BUILD_EXT)
# if defined(_MSC_VER)
#   pragma warning(push)
#   pragma warning(disable: 26115) /* try-lock is treated incorrectly by static analysis */
# endif
  LIBXSMM_INIT
  if (NULL != libxsmm_mmbatch_array /* batch-recording available, but not yet running */
    /* currently, batch recording is only enabled if all values are present (no complex filtering) */
    && NULL != flags && NULL != alpha && NULL != beta
    && NULL != lda && NULL != ldb && NULL != ldc
    && NULL != m && NULL != n && NULL != k
    && LIBXSMM_LOCK_ACQUIRED(LIBXSMM_LOCK_DEFAULT) == LIBXSMM_LOCK_TRYLOCK(LIBXSMM_LOCK_DEFAULT, &libxsmm_mmbatch_lock))
  {
    libxsmm_descriptor_blob blob;
    const libxsmm_gemm_descriptor *const descriptor = libxsmm_gemm_descriptor_init(&blob, precision,
      *m, *n, *k, *lda, *ldb, *ldc, alpha, beta, *flags, libxsmm_get_gemm_prefetch(LIBXSMM_EXT_GEMM_MMBATCH_PREFETCH));
    static int error_once = 0;
    int result = EXIT_SUCCESS;

    if (NULL != descriptor) {
      const unsigned int max_batchsize = (unsigned int)((LIBXSMM_GEMM_MMBATCH_SCALE) * libxsmm_mmbatch_size);
      unsigned int i;
#if !defined(NDEBUG)
      const unsigned int mmbatch_maxdepth = LIBXSMM_UP2POT(LIBXSMM_EXT_GEMM_MMBATCH_MAXDEPTH);
      LIBXSMM_ASSERT((LIBXSMM_EXT_GEMM_MMBATCH_MAXDEPTH) == mmbatch_maxdepth/*is pot*/);
#endif
      /* eventually overwrite the oldest entry */
      i = LIBXSMM_MOD2(internal_ext_gemm_batchdepth, LIBXSMM_EXT_GEMM_MMBATCH_MAXDEPTH);
      internal_ext_gemm_batchdesc[i] = libxsmm_mmbatch_desc; /* backup */
      ++internal_ext_gemm_batchdepth;

      /* ensure descriptor does not match any GEMM such that... */
      LIBXSMM_MEMZERO127(&libxsmm_mmbatch_desc);
      /* ...the batch stops and completely flushes */
      if (0 != internal_ext_gemm_batchsize) {
        result = internal_mmbatch_flush(internal_ext_gemm_batchdesc + i,
          (((libxsmm_blasint)internal_ext_gemm_batchsize - 1) % max_batchsize) + 1,
          (libxsmm_mmbatch_item*)libxsmm_mmbatch_array);
      }

      if (EXIT_SUCCESS == result) { /* enable descriptor */
        internal_ext_gemm_batchsize = 0; /* reset */
        if (0 == (LIBXSMM_MMBATCH_FLAG_STATISTIC & *flags)) {
          libxsmm_mmbatch_desc = *descriptor;
        }
        else {
          libxsmm_mmbatch_desc.flags = LIBXSMM_MMBATCH_FLAG_STATISTIC;
        }
      }
    }
    else {
      result = EXIT_FAILURE;
    }
    if (EXIT_SUCCESS != result && 0 != libxsmm_verbosity /* library code is expected to be mute */
      && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
    {
      fprintf(stderr, "LIBXSMM ERROR: GEMM batch enabling failed!\n");
    }
    LIBXSMM_LOCK_RELEASE(LIBXSMM_LOCK_DEFAULT, &libxsmm_mmbatch_lock);
  }
# if defined(_MSC_VER)
#   pragma warning(pop)
# endif
#else
  LIBXSMM_UNUSED(precision); LIBXSMM_UNUSED(flags);
  LIBXSMM_UNUSED(m); LIBXSMM_UNUSED(n); LIBXSMM_UNUSED(k);
  LIBXSMM_UNUSED(lda); LIBXSMM_UNUSED(ldb); LIBXSMM_UNUSED(ldc);
  LIBXSMM_UNUSED(alpha); LIBXSMM_UNUSED(beta);
#endif
}


LIBXSMM_APIEXT void libxsmm_mmbatch_end(void)
{
#if defined(LIBXSMM_WRAP) && defined(LIBXSMM_BUILD_EXT)
# if defined(_MSC_VER)
#   pragma warning(push)
#   pragma warning(disable: 26115) /* try-lock is treated incorrectly by static analysis */
# endif
  /*const*/ int trystate = LIBXSMM_LOCK_TRYLOCK(LIBXSMM_LOCK_DEFAULT, &libxsmm_mmbatch_lock);
  if (LIBXSMM_LOCK_ACQUIRED(LIBXSMM_LOCK_DEFAULT) == trystate) {
    const unsigned int max_batchsize = (unsigned int)((LIBXSMM_GEMM_MMBATCH_SCALE) * libxsmm_mmbatch_size);
    const libxsmm_gemm_descriptor flushdesc = libxsmm_mmbatch_desc;
    static int error_once = 0;
#if !defined(NDEBUG)
    const unsigned int mmbatch_maxdepth = LIBXSMM_UP2POT(LIBXSMM_EXT_GEMM_MMBATCH_MAXDEPTH);
#endif
    /* ensure descriptor does not match any GEMM such that... */
    LIBXSMM_MEMZERO127(&libxsmm_mmbatch_desc);
    /* ...the batch stops and completely flushes */
    if (EXIT_SUCCESS == internal_mmbatch_flush(&flushdesc,
      0 != internal_ext_gemm_batchsize ? (((internal_ext_gemm_batchsize - 1) % max_batchsize) + 1) : 0,
      (libxsmm_mmbatch_item*)libxsmm_mmbatch_array))
    {
      internal_ext_gemm_batchsize = 0; /* reset */
      --internal_ext_gemm_batchdepth; /* restore the previous descriptor */
      assert((LIBXSMM_EXT_GEMM_MMBATCH_MAXDEPTH) == mmbatch_maxdepth/*is pot*/); /* no LIBXSMM_ASSERT! */
      libxsmm_mmbatch_desc = internal_ext_gemm_batchdesc[LIBXSMM_MOD2(internal_ext_gemm_batchdepth, LIBXSMM_EXT_GEMM_MMBATCH_MAXDEPTH)];
    }
    else if (0 != libxsmm_verbosity /* library code is expected to be mute */
      && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
    {
      fprintf(stderr, "LIBXSMM ERROR: GEMM batch processing failed!\n");
    }
    LIBXSMM_LOCK_RELEASE(LIBXSMM_LOCK_DEFAULT, &libxsmm_mmbatch_lock);
  }
# if defined(_MSC_VER)
#   pragma warning(pop)
# endif
#endif
}


#if defined(LIBXSMM_BUILD) && defined(LIBXSMM_BUILD_EXT) && (!defined(LIBXSMM_NOFORTRAN) || defined(__clang_analyzer__))

/* implementation provided for Fortran 77 compatibility */
LIBXSMM_APIEXT void LIBXSMM_FSYMBOL(libxsmm_xgemm_omp)(const libxsmm_datatype*, const libxsmm_datatype*,
  const char*, const char*, const libxsmm_blasint*, const libxsmm_blasint*, const libxsmm_blasint*,
  const double*, const double*, const libxsmm_blasint*, const double*, const libxsmm_blasint*,
  const double*, double*, const libxsmm_blasint*);
LIBXSMM_APIEXT void LIBXSMM_FSYMBOL(libxsmm_xgemm_omp)(const libxsmm_datatype* iprec, const libxsmm_datatype* oprec,
  const char* transa, const char* transb, const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const double* alpha, const double* a, const libxsmm_blasint* lda, const double* b, const libxsmm_blasint* ldb,
  const double* beta, double* c, const libxsmm_blasint* ldc)
{
  LIBXSMM_ASSERT(NULL != iprec && NULL != oprec);
  libxsmm_xgemm_omp(*iprec, *oprec, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
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
LIBXSMM_APIEXT void LIBXSMM_FSYMBOL(libxsmm_gemm_batch_omp)(const libxsmm_datatype*, const libxsmm_datatype*,
  const char*, const char*, const libxsmm_blasint*, const libxsmm_blasint*, const libxsmm_blasint*,
  const void*, const void*, const libxsmm_blasint*, const void*, const libxsmm_blasint*,
  const void*, void*, const libxsmm_blasint*, const libxsmm_blasint*, const libxsmm_blasint*,
  const libxsmm_blasint[], const libxsmm_blasint[], const libxsmm_blasint[],
  const libxsmm_blasint*);
LIBXSMM_APIEXT void LIBXSMM_FSYMBOL(libxsmm_gemm_batch_omp)(const libxsmm_datatype* iprec, const libxsmm_datatype* oprec,
  const char* transa, const char* transb, const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const void* alpha, const void* a, const libxsmm_blasint* lda, const void* b, const libxsmm_blasint* ldb,
  const void* beta, void* c, const libxsmm_blasint* ldc, const libxsmm_blasint* index_base, const libxsmm_blasint* index_stride,
  const libxsmm_blasint stride_a[], const libxsmm_blasint stride_b[], const libxsmm_blasint stride_c[],
  const libxsmm_blasint* batchsize)
{
  LIBXSMM_ASSERT(NULL != iprec && NULL != oprec && NULL != m && NULL != n && NULL != k);
  LIBXSMM_ASSERT(NULL != index_base && NULL != index_stride && NULL != batchsize);
  libxsmm_gemm_batch_omp(*iprec, *oprec, transa, transb, *m, *n, *k, alpha, a, lda, b, ldb, beta, c, ldc,
    *index_base, *index_stride, stride_a, stride_b, stride_c, *batchsize);
}


/* implementation provided for Fortran 77 compatibility */
LIBXSMM_APIEXT void LIBXSMM_FSYMBOL(libxsmm_mmbatch_begin)(const libxsmm_datatype*,
  const int*, const libxsmm_blasint*, const libxsmm_blasint*, const libxsmm_blasint*,
  const libxsmm_blasint*, const libxsmm_blasint*, const libxsmm_blasint*,
  const void*, const void*);
LIBXSMM_APIEXT void LIBXSMM_FSYMBOL(libxsmm_mmbatch_begin)(const libxsmm_datatype* precision,
  const int* flags, const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const libxsmm_blasint* lda, const libxsmm_blasint* ldb, const libxsmm_blasint* ldc,
  const void* alpha, const void* beta)
{
  LIBXSMM_ASSERT(NULL != precision);
  libxsmm_mmbatch_begin(*precision, flags, m, n, k, lda, ldb, ldc, alpha, beta);
}


/* implementation provided for Fortran 77 compatibility */
LIBXSMM_APIEXT void LIBXSMM_FSYMBOL(libxsmm_mmbatch_end)(void);
LIBXSMM_APIEXT void LIBXSMM_FSYMBOL(libxsmm_mmbatch_end)(void)
{
  libxsmm_mmbatch_end();
}

#endif /*defined(LIBXSMM_BUILD) && defined(LIBXSMM_BUILD_EXT) && (!defined(LIBXSMM_NOFORTRAN) || defined(__clang_analyzer__))*/

