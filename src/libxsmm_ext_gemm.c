/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
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
      libxsmm_gemm_xbatch(LIBXSMM_DATATYPE_F64, LIBXSMM_DATATYPE_F64,
        transa_array, transb_array, m_array, n_array, k_array,
        alpha_array, (const void**)a_array, lda_array, (const void**)b_array, ldb_array,
        beta_array, (void**)c_array, ldc_array,
        group_count, group_size);
    }
    else { /* parallelized */
      libxsmm_gemm_xbatch_omp(LIBXSMM_DATATYPE_F64, LIBXSMM_DATATYPE_F64,
        transa_array, transb_array, m_array, n_array, k_array,
        alpha_array, (const void**)a_array, lda_array, (const void**)b_array, ldb_array,
        beta_array, (void**)c_array, ldc_array,
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
      libxsmm_gemm_xbatch(LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32,
        transa_array, transb_array, m_array, n_array, k_array,
        alpha_array, (const void**)a_array, lda_array, (const void**)b_array, ldb_array,
        beta_array, (void**)c_array, ldc_array, group_count, group_size);
    }
    else { /* parallelized */
      libxsmm_gemm_xbatch_omp(LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32,
        transa_array, transb_array, m_array, n_array, k_array,
        alpha_array, (const void**)a_array, lda_array, (const void**)b_array, ldb_array,
        beta_array, (void**)c_array, ldc_array, group_count, group_size);
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
#if defined(LIBXSMM_WRAP) && defined(LIBXSMM_BUILD_EXT)
  LIBXSMM_ASSERT(NULL != lda && NULL != ldb && NULL != ldc && NULL != m && NULL != n && NULL != k);
  LIBXSMM_ASSERT(NULL != transa && NULL != transb && NULL != alpha && NULL != beta);
  LIBXSMM_INIT
  if (0 != libxsmm_gemm_wrap) {
    libxsmm_dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc); /* sequential */
  }
  else
#endif
  {
    LIBXSMM_GEMM_SYMBOL(double)(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  }
}


LIBXSMM_APIEXT LIBXSMM_ATTRIBUTE_USED void LIBXSMM_FSYMBOL(__wrap_sgemm)(
  const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const float* alpha, const float* a, const libxsmm_blasint* lda,
  const float* b, const libxsmm_blasint* ldb,
  const float* beta, float* c, const libxsmm_blasint* ldc)
{
#if defined(LIBXSMM_WRAP) && defined(LIBXSMM_BUILD_EXT)
  LIBXSMM_ASSERT(NULL != lda && NULL != ldb && NULL != ldc && NULL != m && NULL != n && NULL != k);
  LIBXSMM_ASSERT(NULL != transa && NULL != transb && NULL != alpha && NULL != beta);
  LIBXSMM_INIT
  if (0 != libxsmm_gemm_wrap) {
    libxsmm_sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc); /* sequential */
  }
  else
#endif
  {
    LIBXSMM_GEMM_SYMBOL(float)(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
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
    const libxsmm_bitfield prefetch = libxsmm_get_gemm_prefetch(LIBXSMM_PREFETCH_AUTO);
    const size_t sa = (NULL != stride_a ? (size_t)(*stride_a) : sizeof(void*));
    const size_t sb = (NULL != stride_b ? (size_t)(*stride_b) : sizeof(void*));
    const size_t sc = (NULL != stride_c ? (size_t)(*stride_c) : sizeof(void*));
    const unsigned char otypesize = libxsmm_typesize((libxsmm_datatype)oprec);
    const int ngroups = (int)LIBXSMM_ABS(group_count);
    int group = 0, group_next = LIBXSMM_GEMM_NPARGROUPS;
    libxsmm_xmmfunction kernel[LIBXSMM_GEMM_NPARGROUPS];
    libxsmm_blasint base[LIBXSMM_GEMM_NPARGROUPS] = { 0 }, i;
    libxsmm_bitfield kflags[LIBXSMM_GEMM_NPARGROUPS] = { 0 };
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
          const int gemm_flags = LIBXSMM_GEMM_PFLAGS(ta, tb, LIBXSMM_FLAGS);
          const libxsmm_gemm_shape shape = libxsmm_create_gemm_shape(im, in, ik,
            NULL != lda ? lda[g] : (0 == (LIBXSMM_GEMM_FLAG_TRANS_A & gemm_flags) ? im : ik),
            NULL != ldb ? ldb[g] : (0 == (LIBXSMM_GEMM_FLAG_TRANS_B & gemm_flags) ? ik : in),
            NULL != ldc ? ldc[g] : im,
            iprec, iprec, oprec, oprec);
          const libxsmm_bitfield flags = libxsmm_gemm_batch_flags(gemm_flags, &shape, c);
          kernel[i].gemm = libxsmm_dispatch_gemm_v2(shape, flags, prefetch);
          if (NULL != kernel[i].ptr_const) {
            if (size < asize) size = asize;
            kflags[i] = flags;
          }
          else {
            suitable = 0;
            break;
          }
        }
        else break;
      }
      LIBXSMM_ASSERT(0 < libxsmm_gemm_taskgrain);
      if (0 != suitable) { /* check if an SMM is suitable */
        const unsigned char itypesize = libxsmm_typesize((libxsmm_datatype)iprec);
#if defined(_OPENMP)
        const int nchunks = (int)LIBXSMM_UPDIV(size, libxsmm_gemm_taskgrain);
        const int ntasks = nchunks * npargroups, nthreads = LIBXSMM_MIN(max_nthreads, ntasks);
        if (1 < nthreads) {
          LIBXSMM_OMP_VAR(i);
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
                  /*check*/libxsmm_mmbatch_kernel(kernel[g].gemm, index_base, index_stride, stride_a, stride_b, stride_c,
                    (const char*)a + sa * base[u], (const char*)b + sb * base[u], (char*)c + sc * base[u],
                    0 < group_count ? isize : -asize, (int)i, nchunks, itypesize, otypesize,
                    kflags[g]);
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
                      /*check*/libxsmm_mmbatch_kernel(kernel[g].gemm, index_base, index_stride, stride_a, stride_b, stride_c,
                        (const char*)a + sa * base[u], (const char*)b + sb * base[u], (char*)c + sc * base[u],
                        0 < group_count ? isize : -asize, (int)i, nchunks, itypesize, otypesize,
                        kflags[g]);
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
# endif
                {
                  /*check*/libxsmm_mmbatch_kernel(kernel[g].gemm, index_base, index_stride, stride_a, stride_b, stride_c,
                    (const char*)a + sa * base[u], (const char*)b + sb * base[u], (char*)c + sc * base[u],
                    0 < group_count ? isize : -asize, (int)i, nchunks, itypesize, otypesize,
                    kflags[g]);
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
            libxsmm_mmbatch_kernel(kernel[i].gemm, index_base, index_stride, stride_a, stride_b, stride_c,
              (const char*)a + sa * base[i], (const char*)b + sb * base[i], (char*)c + sc * base[i], batchsize[g],
              0/*tid*/, 1/*nthreads*/, itypesize, otypesize,
              kflags[i]);
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


LIBXSMM_APIEXT void libxsmm_gemm_xbatch_omp(
  libxsmm_datatype iprec, libxsmm_datatype oprec, const char transa_array[], const char transb_array[],
  const libxsmm_blasint m_array[], const libxsmm_blasint n_array[], const libxsmm_blasint k_array[],
  const void* alpha_array, const void* a_array[], const libxsmm_blasint lda_array[],
                           const void* b_array[], const libxsmm_blasint ldb_array[],
  const void* beta_array,        void* c_array[], const libxsmm_blasint ldc_array[],
  const libxsmm_blasint* group_count, const libxsmm_blasint group_size[])
{
  if (NULL != group_count) {
    const libxsmm_blasint ptrsize = sizeof(void*);
    internal_gemm_batch_omp(iprec, oprec,
      transa_array, transb_array, m_array, n_array, k_array,
      alpha_array, (const void**)a_array, lda_array, (const void**)b_array, ldb_array,
      beta_array, (void**)c_array, ldc_array,
      0/*index_base*/, 0/*index_stride*/, &ptrsize, &ptrsize, &ptrsize, group_size, *group_count);
  }
}


#if defined(LIBXSMM_BUILD) && defined(LIBXSMM_BUILD_EXT) && (!defined(LIBXSMM_NOFORTRAN) || defined(__clang_analyzer__))

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

#endif /*defined(LIBXSMM_BUILD) && defined(LIBXSMM_BUILD_EXT) && (!defined(LIBXSMM_NOFORTRAN) || defined(__clang_analyzer__))*/
