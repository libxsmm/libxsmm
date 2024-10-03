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
#include "libxsmm_ext_gemm.h"
#include "libxsmm_xcopy.h"

#if defined(LIBXSMM_WRAP) && defined(LIBXSMM_BUILD_EXT)
# include "libxsmm_trace.h"
#endif


#if defined(LIBXSMM_BUILD) && defined(LIBXSMM_BUILD_EXT)

#if defined(LIBXSMM_WRAP)
LIBXSMM_APIEXT libxsmm_dgemm_batch_strided_function libxsmm_original_dgemm_batch_strided(void)
{
  LIBXSMM_BLAS_WRAPPER(double, gemm_batch_strided, libxsmm_original_dgemm_batch_strided_function);
  return libxsmm_original_dgemm_batch_strided_function;
}

LIBXSMM_APIEXT libxsmm_sgemm_batch_strided_function libxsmm_original_sgemm_batch_strided(void)
{
  LIBXSMM_BLAS_WRAPPER(float, gemm_batch_strided, libxsmm_original_sgemm_batch_strided_function);
  return libxsmm_original_sgemm_batch_strided_function;
}

LIBXSMM_APIEXT libxsmm_dgemm_batch_function libxsmm_original_dgemm_batch(void)
{
  LIBXSMM_BLAS_WRAPPER(double, gemm_batch, libxsmm_original_dgemm_batch_function);
  return libxsmm_original_dgemm_batch_function;
}

LIBXSMM_APIEXT libxsmm_sgemm_batch_function libxsmm_original_sgemm_batch(void)
{
  LIBXSMM_BLAS_WRAPPER(float, gemm_batch, libxsmm_original_sgemm_batch_function);
  return libxsmm_original_sgemm_batch_function;
}

LIBXSMM_APIEXT libxsmm_dgemm_function libxsmm_original_dgemm(void)
{
  LIBXSMM_BLAS_WRAPPER(double, gemm, libxsmm_original_dgemm_function);
  return libxsmm_original_dgemm_function;
}

LIBXSMM_APIEXT libxsmm_sgemm_function libxsmm_original_sgemm(void)
{
  LIBXSMM_BLAS_WRAPPER(float, gemm, libxsmm_original_sgemm_function);
  return libxsmm_original_sgemm_function;
}

LIBXSMM_APIEXT libxsmm_dgemv_function libxsmm_original_dgemv(void)
{
  LIBXSMM_BLAS_WRAPPER(double, gemv, libxsmm_original_dgemv_function);
  return libxsmm_original_dgemv_function;
}

LIBXSMM_APIEXT libxsmm_sgemv_function libxsmm_original_sgemv(void)
{
  LIBXSMM_BLAS_WRAPPER(float, gemv, libxsmm_original_sgemv_function);
  return libxsmm_original_sgemv_function;
}
#endif /*defined(LIBXSMM_WRAP)*/


LIBXSMM_APIEXT LIBXSMM_ATTRIBUTE_USED void LIBXSMM_BLAS_FSYMBOL_WRAP(double, gemm_batch_strided)(
  const char* transa, const char* transb, const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const double* alpha, const double* a, const libxsmm_blasint* lda, const libxsmm_blasint* stride_a,
  const double* b, const libxsmm_blasint* ldb, const libxsmm_blasint* stride_b,
  const double* beta, double* c, const libxsmm_blasint* ldc, const libxsmm_blasint* stride_c,
  const libxsmm_blasint* batchsize)
{
  LIBXSMM_ASSERT(NULL != transa && NULL != transb && NULL != m && NULL != n && NULL != k);
  LIBXSMM_ASSERT(NULL != alpha && NULL != beta && NULL != batchsize /*&& 0 <= *batchsize*/);
  LIBXSMM_ASSERT(NULL != a && NULL != lda && NULL != stride_a);
  LIBXSMM_ASSERT(NULL != b && NULL != ldb && NULL != stride_b);
  LIBXSMM_ASSERT(NULL != c && NULL != ldc && NULL != stride_c);
  LIBXSMM_INIT
#if defined(LIBXSMM_WRAP) && defined(LIBXSMM_BUILD_EXT)
  if (0 != libxsmm_gemm_wrap) {
    if ((unsigned int)*batchsize <= libxsmm_gemm_taskgrain) { /* sequential */
      libxsmm_gemm_strided(LIBXSMM_DATATYPE_F64, LIBXSMM_DATATYPE_F64, transa, transb,
        *m, *n, *k, alpha, a, lda, stride_a, b, ldb, stride_b, beta, c, ldc, stride_c,
        0/*index_base*/, *batchsize);
    }
    else { /* parallelized */
      libxsmm_gemm_strided_omp(LIBXSMM_DATATYPE_F64, LIBXSMM_DATATYPE_F64, transa, transb,
        *m, *n, *k, alpha, a, lda, stride_a, b, ldb, stride_b, beta, c, ldc, stride_c,
        0/*index_base*/, *batchsize);
    }
  }
  else
#endif
  {
    LIBXSMM_BLAS_FSYMBOL_REAL(double, gemm_batch_strided)(transa, transb, m, n, k,
      alpha, a, lda, stride_a, b, ldb, stride_b, beta, c, ldc, stride_c,
      batchsize);
  }
}


LIBXSMM_APIEXT LIBXSMM_ATTRIBUTE_USED void LIBXSMM_BLAS_FSYMBOL_WRAP(float, gemm_batch_strided)(
  const char* transa, const char* transb, const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const float* alpha, const float* a, const libxsmm_blasint* lda, const libxsmm_blasint* stride_a,
  const float* b, const libxsmm_blasint* ldb, const libxsmm_blasint* stride_b,
  const float* beta, float* c, const libxsmm_blasint* ldc, const libxsmm_blasint* stride_c,
  const libxsmm_blasint* batchsize)
{
  LIBXSMM_ASSERT(NULL != transa && NULL != transb && NULL != m && NULL != n && NULL != k);
  LIBXSMM_ASSERT(NULL != alpha && NULL != beta && NULL != batchsize /*&& 0 <= *batchsize*/);
  LIBXSMM_ASSERT(NULL != a && NULL != lda && NULL != stride_a);
  LIBXSMM_ASSERT(NULL != b && NULL != ldb && NULL != stride_b);
  LIBXSMM_ASSERT(NULL != c && NULL != ldc && NULL != stride_c);
  LIBXSMM_INIT
#if defined(LIBXSMM_WRAP) && defined(LIBXSMM_BUILD_EXT)
  if (0 != libxsmm_gemm_wrap) {
    if ((unsigned int)*batchsize <= libxsmm_gemm_taskgrain) { /* sequential */
      libxsmm_gemm_strided(LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, transa, transb,
        *m, *n, *k, alpha, a, lda, stride_a, b, ldb, stride_b, beta, c, ldc, stride_c,
        0/*index_base*/, *batchsize);
    }
    else { /* parallelized */
      libxsmm_gemm_strided_omp(LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, transa, transb,
        *m, *n, *k, alpha, a, lda, stride_a, b, ldb, stride_b, beta, c, ldc, stride_c,
        0/*index_base*/, *batchsize);
    }
  }
  else
#endif
  {
    LIBXSMM_BLAS_FSYMBOL_REAL(float, gemm_batch_strided)(transa, transb, m, n, k,
      alpha, a, lda, stride_a, b, ldb, stride_b, beta, c, ldc, stride_c,
      batchsize);
  }
}


LIBXSMM_APIEXT LIBXSMM_ATTRIBUTE_USED void LIBXSMM_BLAS_FSYMBOL_WRAP(double, gemm_batch)(
  const char transa_array[], const char transb_array[], const libxsmm_blasint m_array[], const libxsmm_blasint n_array[], const libxsmm_blasint k_array[],
  const double alpha_array[], const double* a_array[], const libxsmm_blasint lda_array[], const double* b_array[], const libxsmm_blasint ldb_array[],
  const double beta_array[], double* c_array[], const libxsmm_blasint ldc_array[], const libxsmm_blasint* group_count, const libxsmm_blasint group_size[])
{
  LIBXSMM_ASSERT(NULL != lda_array && NULL != ldb_array && NULL != ldc_array && NULL != m_array && NULL != n_array && NULL != k_array);
  LIBXSMM_ASSERT(NULL != transa_array && NULL != transb_array && NULL != alpha_array && NULL != beta_array);
  LIBXSMM_ASSERT(NULL != group_count && 0 <= *group_count && NULL != group_size);
  LIBXSMM_INIT
#if defined(LIBXSMM_WRAP) && defined(LIBXSMM_BUILD_EXT)
  if (0 != libxsmm_gemm_wrap) {
    if (1 == *group_count && (unsigned int)*group_size <= libxsmm_gemm_taskgrain) { /* sequential */
      libxsmm_gemm_groups(LIBXSMM_DATATYPE_F64, LIBXSMM_DATATYPE_F64,
        transa_array, transb_array, m_array, n_array, k_array,
        alpha_array, (const void**)a_array, lda_array, (const void**)b_array, ldb_array,
        beta_array, (void**)c_array, ldc_array, *group_count, group_size);
    }
    else { /* parallelized */
      libxsmm_gemm_groups_omp(LIBXSMM_DATATYPE_F64, LIBXSMM_DATATYPE_F64,
        transa_array, transb_array, m_array, n_array, k_array,
        alpha_array, (const void**)a_array, lda_array, (const void**)b_array, ldb_array,
        beta_array, (void**)c_array, ldc_array, *group_count, group_size);
    }
  }
  else
#endif
  {
    LIBXSMM_BLAS_FSYMBOL_REAL(double, gemm_batch)(transa_array, transb_array, m_array, n_array, k_array,
      alpha_array, a_array, lda_array, b_array, ldb_array, beta_array, c_array, ldc_array,
      group_count, group_size);
  }
}


LIBXSMM_APIEXT LIBXSMM_ATTRIBUTE_USED void LIBXSMM_BLAS_FSYMBOL_WRAP(float, gemm_batch)(
  const char transa_array[], const char transb_array[], const libxsmm_blasint m_array[], const libxsmm_blasint n_array[], const libxsmm_blasint k_array[],
  const float alpha_array[], const float* a_array[], const libxsmm_blasint lda_array[], const float* b_array[], const libxsmm_blasint ldb_array[],
  const float beta_array[], float* c_array[], const libxsmm_blasint ldc_array[], const libxsmm_blasint* group_count, const libxsmm_blasint group_size[])
{
  LIBXSMM_ASSERT(NULL != lda_array && NULL != ldb_array && NULL != ldc_array && NULL != m_array && NULL != n_array && NULL != k_array);
  LIBXSMM_ASSERT(NULL != transa_array && NULL != transb_array && NULL != alpha_array && NULL != beta_array);
  LIBXSMM_ASSERT(NULL != group_count && 0 <= *group_count && NULL != group_size);
  LIBXSMM_INIT
#if defined(LIBXSMM_WRAP) && defined(LIBXSMM_BUILD_EXT)
  if (0 != libxsmm_gemm_wrap) {
    if (1 == *group_count && (unsigned int)*group_size <= libxsmm_gemm_taskgrain) { /* sequential */
      libxsmm_gemm_groups(LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32,
        transa_array, transb_array, m_array, n_array, k_array,
        alpha_array, (const void**)a_array, lda_array, (const void**)b_array, ldb_array,
        beta_array, (void**)c_array, ldc_array, *group_count, group_size);
    }
    else { /* parallelized */
      libxsmm_gemm_groups_omp(LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32,
        transa_array, transb_array, m_array, n_array, k_array,
        alpha_array, (const void**)a_array, lda_array, (const void**)b_array, ldb_array,
        beta_array, (void**)c_array, ldc_array, *group_count, group_size);
    }
  }
  else
#endif
  {
    LIBXSMM_BLAS_FSYMBOL_REAL(float, gemm_batch)(transa_array, transb_array, m_array, n_array, k_array,
      alpha_array, a_array, lda_array, b_array, ldb_array, beta_array, c_array, ldc_array,
      group_count, group_size);
  }
}


LIBXSMM_APIEXT LIBXSMM_ATTRIBUTE_USED void LIBXSMM_BLAS_FSYMBOL_WRAP(double, gemm)(
  const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const double* alpha, const double* a, const libxsmm_blasint* lda,
  const double* b, const libxsmm_blasint* ldb,
  const double* beta, double* c, const libxsmm_blasint* ldc)
{
  LIBXSMM_ASSERT(NULL != lda && NULL != ldb && NULL != ldc && NULL != m && NULL != n && NULL != k);
  LIBXSMM_ASSERT(NULL != transa && NULL != transb && NULL != alpha && NULL != beta);
  LIBXSMM_INIT
#if defined(LIBXSMM_WRAP) && defined(LIBXSMM_BUILD_EXT)
  if (0 != libxsmm_gemm_wrap) {
    libxsmm_dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc); /* sequential */
  }
  else
#endif
  {
    LIBXSMM_BLAS_FSYMBOL_REAL(double, gemm)(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  }
}


LIBXSMM_APIEXT LIBXSMM_ATTRIBUTE_USED void LIBXSMM_BLAS_FSYMBOL_WRAP(float, gemm)(
  const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const float* alpha, const float* a, const libxsmm_blasint* lda,
  const float* b, const libxsmm_blasint* ldb,
  const float* beta, float* c, const libxsmm_blasint* ldc)
{
  LIBXSMM_ASSERT(NULL != lda && NULL != ldb && NULL != ldc && NULL != m && NULL != n && NULL != k);
  LIBXSMM_ASSERT(NULL != transa && NULL != transb && NULL != alpha && NULL != beta);
  LIBXSMM_INIT
#if defined(LIBXSMM_WRAP) && defined(LIBXSMM_BUILD_EXT)
  if (0 != libxsmm_gemm_wrap) {
    libxsmm_sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc); /* sequential */
  }
  else
#endif
  {
    LIBXSMM_BLAS_FSYMBOL_REAL(float, gemm)(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  }
}


LIBXSMM_APIEXT LIBXSMM_ATTRIBUTE_USED void LIBXSMM_BLAS_FSYMBOL_WRAP(double, gemv)(const char* trans, const libxsmm_blasint* m, const libxsmm_blasint* n,
  const double* alpha, const double* a, const libxsmm_blasint* lda, const double* x, const libxsmm_blasint* incx,
  const double* beta, double* y, const libxsmm_blasint* incy)
{
  LIBXSMM_ASSERT(NULL != trans && NULL != m && NULL != n && NULL != lda && NULL != incx && NULL != incy && NULL != alpha && NULL != beta);
  LIBXSMM_INIT
#if defined(LIBXSMM_WRAP) && defined(LIBXSMM_BUILD_EXT)
  if (2 < LIBXSMM_ABS(libxsmm_gemm_wrap) && 1 == *incx && 1 == *incy && LIBXSMM_SMM(*m, 1, *n, 2/*RFO*/, sizeof(double))) {
    const libxsmm_gemm_shape gemm_shape = libxsmm_create_gemm_shape(*m, 1/*n*/, *n/*k*/, *lda, *n/*ldb*/, *m/*ldc*/,
      LIBXSMM_DATATYPE_F64, LIBXSMM_DATATYPE_F64, LIBXSMM_DATATYPE_F64, LIBXSMM_DATATYPE_F64);
    const int flags = (('n' == *trans || *"N" == *trans) ? LIBXSMM_GEMM_FLAG_NONE : LIBXSMM_GEMM_FLAG_TRANS_A) |
      (LIBXSMM_NEQ(0, beta) ? 0 : LIBXSMM_GEMM_FLAG_BETA_0);
    if (LIBXSMM_GEMM_NO_BYPASS(gemm_shape, *alpha, *beta, flags)) { /* TODO: parallelized */
      const libxsmm_gemmfunction xgemv = libxsmm_dispatch_gemm(gemm_shape, flags, (libxsmm_bitfield)LIBXSMM_PREFETCH);
      libxsmm_gemm_param param;
      LIBXSMM_VALUE_ASSIGN(param.a.primary, a);
      LIBXSMM_VALUE_ASSIGN(param.b.primary, x);
      param.c.primary = y;
      LIBXSMM_XGEMM_PREFETCH(double, double, *m, 1/*n*/, *n/*k*/, param);
      LIBXSMM_ASSERT(NULL != xgemv);
      xgemv(&param);

    }
    else {
      LIBXSMM_BLAS_FSYMBOL_REAL(double, gemv)(trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
    }
  }
  else
#endif
  {
    LIBXSMM_BLAS_FSYMBOL_REAL(double, gemv)(trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
  }
}


LIBXSMM_APIEXT LIBXSMM_ATTRIBUTE_USED void LIBXSMM_BLAS_FSYMBOL_WRAP(float, gemv)(const char* trans, const libxsmm_blasint* m, const libxsmm_blasint* n,
  const float* alpha, const float* a, const libxsmm_blasint* lda, const float* x, const libxsmm_blasint* incx,
  const float* beta, float* y, const libxsmm_blasint* incy)
{
  LIBXSMM_ASSERT(NULL != trans && NULL != m && NULL != n && NULL != lda && NULL != incx && NULL != incy && NULL != alpha && NULL != beta);
  LIBXSMM_INIT
#if defined(LIBXSMM_WRAP) && defined(LIBXSMM_BUILD_EXT)
  if (2 < LIBXSMM_ABS(libxsmm_gemm_wrap) && 1 == *incx && 1 == *incy && LIBXSMM_SMM(*m, 1, *n, 2/*RFO*/, sizeof(float))) {
    const libxsmm_gemm_shape gemm_shape = libxsmm_create_gemm_shape(*m, 1/*n*/, *n/*k*/, *lda, *n/*ldb*/, *m/*ldc*/,
      LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32);
    const int flags = (('n' == *trans || *"N" == *trans) ? LIBXSMM_GEMM_FLAG_NONE : LIBXSMM_GEMM_FLAG_TRANS_A) |
      (LIBXSMM_NEQ(0, beta) ? 0 : LIBXSMM_GEMM_FLAG_BETA_0);
    if (LIBXSMM_GEMM_NO_BYPASS(gemm_shape, *alpha, *beta, flags)) { /* TODO: parallelized */
      const libxsmm_gemmfunction xgemv = libxsmm_dispatch_gemm(gemm_shape, flags, (libxsmm_bitfield)LIBXSMM_PREFETCH);
      libxsmm_gemm_param param;
      LIBXSMM_VALUE_ASSIGN(param.a.primary, a);
      LIBXSMM_VALUE_ASSIGN(param.b.primary, x);
      param.c.primary = y;
      LIBXSMM_XGEMM_PREFETCH(float, float, *m, 1/*n*/, *n/*k*/, param);
      LIBXSMM_ASSERT(NULL != xgemv);
      xgemv(&param);

    }
    else {
      LIBXSMM_BLAS_FSYMBOL_REAL(float, gemv)(trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
    }
  }
  else
#endif
  {
    LIBXSMM_BLAS_FSYMBOL_REAL(float, gemv)(trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
  }
}


LIBXSMM_APIEXT LIBXSMM_ATTRIBUTE_USED void __wrap_dgemm_batch_strided(
  const char* transa, const char* transb, const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const double* alpha, const double* a, const libxsmm_blasint* lda, const libxsmm_blasint* stride_a,
  const double* b, const libxsmm_blasint* ldb, const libxsmm_blasint* stride_b,
  const double* beta, double* c, const libxsmm_blasint* ldc, const libxsmm_blasint* stride_c,
  const libxsmm_blasint* batchsize)
{
  LIBXSMM_BLAS_FSYMBOL_WRAP(double, gemm_batch_strided)(transa, transb, m, n, k,
    alpha, a, lda, stride_a, b, ldb, stride_b, beta, c, ldc, stride_c,
    batchsize);
}


LIBXSMM_APIEXT LIBXSMM_ATTRIBUTE_USED void __wrap_sgemm_batch_strided(
  const char* transa, const char* transb, const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const float* alpha, const float* a, const libxsmm_blasint* lda, const libxsmm_blasint* stride_a,
  const float* b, const libxsmm_blasint* ldb, const libxsmm_blasint* stride_b,
  const float* beta, float* c, const libxsmm_blasint* ldc, const libxsmm_blasint* stride_c,
  const libxsmm_blasint* batchsize)
{
  LIBXSMM_BLAS_FSYMBOL_WRAP(float, gemm_batch_strided)(transa, transb, m, n, k,
    alpha, a, lda, stride_a, b, ldb, stride_b, beta, c, ldc, stride_c,
    batchsize);
}


LIBXSMM_APIEXT LIBXSMM_ATTRIBUTE_USED void __wrap_dgemm_batch(
  const char transa_array[], const char transb_array[], const libxsmm_blasint m_array[], const libxsmm_blasint n_array[], const libxsmm_blasint k_array[],
  const double alpha_array[], const double* a_array[], const libxsmm_blasint lda_array[], const double* b_array[], const libxsmm_blasint ldb_array[],
  const double beta_array[], double* c_array[], const libxsmm_blasint ldc_array[], const libxsmm_blasint* group_count, const libxsmm_blasint group_size[])
{
  LIBXSMM_BLAS_FSYMBOL_WRAP(double, gemm_batch)(transa_array, transb_array, m_array, n_array, k_array,
    alpha_array, a_array, lda_array, b_array, ldb_array, beta_array, c_array, ldc_array,
    group_count, group_size);
}


LIBXSMM_APIEXT LIBXSMM_ATTRIBUTE_USED void __wrap_sgemm_batch(
  const char transa_array[], const char transb_array[], const libxsmm_blasint m_array[], const libxsmm_blasint n_array[], const libxsmm_blasint k_array[],
  const float alpha_array[], const float* a_array[], const libxsmm_blasint lda_array[], const float* b_array[], const libxsmm_blasint ldb_array[],
  const float beta_array[], float* c_array[], const libxsmm_blasint ldc_array[], const libxsmm_blasint* group_count, const libxsmm_blasint group_size[])
{
  LIBXSMM_BLAS_FSYMBOL_WRAP(float, gemm_batch)(transa_array, transb_array, m_array, n_array, k_array,
    alpha_array, a_array, lda_array, b_array, ldb_array, beta_array, c_array, ldc_array,
    group_count, group_size);
}

#endif /*defined(LIBXSMM_BUILD) && defined(LIBXSMM_BUILD_EXT)*/


LIBXSMM_API_INLINE void internal_gemm_batch_omp(libxsmm_datatype iprec, libxsmm_datatype oprec,
  const char* transa, const char* transb, const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const void* alpha, const void* a, const libxsmm_blasint* lda, const libxsmm_blasint* stride_a,
  const void* b, const libxsmm_blasint* ldb, const libxsmm_blasint* stride_b,
  const void* beta, void* c, const libxsmm_blasint* ldc, const libxsmm_blasint* stride_c,
  libxsmm_blasint index_stride, libxsmm_blasint index_base,
  const libxsmm_blasint* batchsize, libxsmm_blasint group_count)
{
#if defined(LIBXSMM_BATCH_CHECK)
  static int error_once = 0;
#endif
  LIBXSMM_INIT
  if ( /* check for sensible arguments */
#if defined(LIBXSMM_BATCH_CHECK)
    NULL != a && NULL != b && NULL != c && (1 == group_count || -1 == group_count || (0 == index_stride
      && (NULL == stride_a || 0 != *stride_a)
      && (NULL == stride_b || 0 != *stride_b)
      && (NULL == stride_c || 0 != *stride_c))) &&
#endif
    0 != group_count)
  {
    const libxsmm_bitfield prefetch = libxsmm_get_gemm_prefetch(LIBXSMM_PREFETCH_AUTO);
    const size_t sa = (NULL != stride_a ? (size_t)(*stride_a) : sizeof(void*));
    const size_t sb = (NULL != stride_b ? (size_t)(*stride_b) : sizeof(void*));
    const size_t sc = (NULL != stride_c ? (size_t)(*stride_c) : sizeof(void*));
    const unsigned char otypesize = libxsmm_typesize((libxsmm_datatype)oprec);
    const libxsmm_blasint ngroups = LIBXSMM_ABS(group_count);
    libxsmm_xmmfunction kernel /*= { NULL }*/;
    libxsmm_blasint i, j = 0;
    for (i = 0; i < ngroups; ++i) {
      const libxsmm_blasint isize = batchsize[i], asize = LIBXSMM_ABS(isize);
      const void *const ialpha = (NULL != alpha ? &((const char*)alpha)[i*otypesize] : NULL);
      const void *const ibeta = (NULL != beta ? &((const char*)beta)[i*otypesize] : NULL);
      double dalpha = LIBXSMM_ALPHA, dbeta = LIBXSMM_BETA;
      if (0 < asize
        && EXIT_SUCCESS == libxsmm_dvalue(oprec, ialpha, &dalpha)
        && EXIT_SUCCESS == libxsmm_dvalue(oprec, ibeta, &dbeta))
      {
        const char *const ta = (NULL != transa ? (transa + i) : NULL);
        const char *const tb = (NULL != transb ? (transb + i) : NULL);
        const libxsmm_bitfield gemm_flags = LIBXSMM_GEMM_PFLAGS(ta, tb, LIBXSMM_FLAGS) |
          (LIBXSMM_NEQ(0, dbeta) ? 0 : LIBXSMM_GEMM_FLAG_BETA_0);
        const libxsmm_blasint im = m[i], in = n[i], ik = k[i];
        const libxsmm_gemm_shape shape = libxsmm_create_gemm_shape(im, in, ik,
          NULL != lda ? lda[i] : (0 == (LIBXSMM_GEMM_FLAG_TRANS_A & gemm_flags) ? im : ik),
          NULL != ldb ? ldb[i] : (0 == (LIBXSMM_GEMM_FLAG_TRANS_B & gemm_flags) ? ik : in),
          NULL != ldc ? ldc[i] : im, iprec, iprec, oprec, oprec);
        if  (LIBXSMM_GEMM_NO_BYPASS(shape, dalpha, dbeta, gemm_flags)
          && LIBXSMM_SMM_AI(im, in, ik, 2/*RFO*/, otypesize))
        {
          kernel.gemm = libxsmm_dispatch_gemm(shape, gemm_flags, prefetch);
        }
        else kernel.ptr_const = NULL;
        LIBXSMM_ASSERT(0 < libxsmm_gemm_taskgrain);
        if (NULL != kernel.ptr_const) { /* check if an SMM is suitable */
          const unsigned char itypesize = libxsmm_typesize((libxsmm_datatype)iprec);
#if defined(_OPENMP)
# if defined(LIBXSMM_EXT_TASKS)
          const int outerpar = omp_get_active_level();
# else
          const int outerpar = omp_in_parallel();
# endif
          const int max_nthreads = (0 == outerpar ? omp_get_max_threads() : omp_get_num_threads());
          const int ntasks = (int)LIBXSMM_UPDIV(asize, libxsmm_gemm_taskgrain);
          const int nthreads = LIBXSMM_MIN(max_nthreads, ntasks);
          if (1 < nthreads && 0 == (LIBXSMM_GEMM_FLAG_BETA_0 & gemm_flags)) {
            int tid = 0;
            LIBXSMM_OMP_VAR(tid);
            if (0 == outerpar) { /* enable internal parallelization */
# if defined(LIBXSMM_EXT_TASKS)
              if (0 == libxsmm_gemm_tasks)
# endif
              {
#               pragma omp parallel for num_threads(nthreads) private(tid)
                for (tid = 0; tid < ntasks; ++tid) {
                  LIBXSMM_EXPECT(EXIT_SUCCESS == libxsmm_gemm_batch_kernel(
                    kernel.gemm, index_base, index_stride, stride_a, stride_b, stride_c,
                    (const char*)a + j * sa, (const char*)b + j * sb, (char*)c + j * sc,
                    0 < group_count ? isize : -asize, tid, ntasks, itypesize, otypesize));
                }
              }
# if defined(LIBXSMM_EXT_TASKS)
              else { /* tasks requested */
#               pragma omp parallel num_threads(nthreads) private(tid)
                { /* first thread discovering work will launch all tasks */
#                 pragma omp single nowait /* anyone is good */
                  for (tid = 0; tid < ntasks; ++tid) {
#                   pragma omp task
                    {
                      LIBXSMM_EXPECT(EXIT_SUCCESS == libxsmm_gemm_batch_kernel(
                        kernel.gemm, index_base, index_stride, stride_a, stride_b, stride_c,
                        (const char*)a + j * sa, (const char*)b + j * sb, (char*)c + j * sc,
                        0 < group_count ? isize : -asize, tid, ntasks, itypesize, otypesize));
                    }
                  }
                } /* implicit synchronization (barrier) */
              }
# endif
            }
            else { /* assume external parallelization */
              for (tid = 0; tid < ntasks; ++tid) {
# if defined(LIBXSMM_EXT_TASKS) /* OpenMP-tasks */
#               pragma omp task
# endif
                {
                  LIBXSMM_EXPECT(EXIT_SUCCESS == libxsmm_gemm_batch_kernel(
                    kernel.gemm, index_base, index_stride, stride_a, stride_b, stride_c,
                    (const char*)a + j * sa, (const char*)b + j * sb, (char*)c + j * sc,
                    0 < group_count ? isize : -asize, tid, ntasks, itypesize, otypesize));
                }
              }
# if defined(LIBXSMM_EXT_TASKS) /* OpenMP-tasks */
              if (0 == libxsmm_nosync) { /* allow to omit synchronization */
#               pragma omp taskwait
              }
# endif
            }
          }
          else
#endif /*defined(_OPENMP)*/
          { /* sequential */
            LIBXSMM_EXPECT(EXIT_SUCCESS == libxsmm_gemm_batch_kernel(
              kernel.gemm, index_base, index_stride, stride_a, stride_b, stride_c,
              (const char*)a + j * sa, (const char*)b + j * sb, (char*)c + j * sc,
              0 < group_count ? isize : -asize, 0/*tid*/, 1/*nthreads*/, itypesize, otypesize));
          }
        }
        else { /* trigger fallback */
          libxsmm_gemm_batch_blas(iprec, oprec, ta, tb, im, in, ik,
            ialpha, (const char*)a + j * sa, &shape.lda, stride_a, (const char*)b + j * sb, &shape.ldb, stride_b,
            ibeta, (char*)c + j * sc, &shape.ldc, stride_c, index_stride, index_base, batchsize[i]);
          if (LIBXSMM_VERBOSITY_WARN <= libxsmm_verbosity || 0 > libxsmm_verbosity) {
            const size_t threshold = LIBXSMM_MNK_SIZE(im, in, im);
            static size_t threshold_max = 0;
            if (threshold_max != threshold) {
              LIBXSMM_STDIO_ACQUIRE();
              fprintf(stderr, "LIBXSMM WARNING: batched GEMM/omp was falling back!\n");
              LIBXSMM_STDIO_RELEASE();
              threshold_max = threshold;
            }
          }
        }
      }
      j += asize;
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
  const void* alpha, const void* a, const libxsmm_blasint* lda, const libxsmm_blasint stride_a[],
  const void* b, const libxsmm_blasint* ldb, const libxsmm_blasint stride_b[],
  const void* beta, void* c, const libxsmm_blasint* ldc, const libxsmm_blasint stride_c[],
  libxsmm_blasint index_stride, libxsmm_blasint index_base,
  libxsmm_blasint batchsize)
{
  internal_gemm_batch_omp(iprec, oprec, transa, transb, &m, &n, &k,
    alpha, a, lda, stride_a, b, ldb, stride_b, beta, c, ldc, stride_c,
    index_stride, index_base, &batchsize, 1/*group_count*/);
}


LIBXSMM_APIEXT void libxsmm_gemm_strided_omp(libxsmm_datatype iprec, libxsmm_datatype oprec,
  const char* transa, const char* transb, libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const void* alpha, const void* a, const libxsmm_blasint* lda, const libxsmm_blasint* stride_a,
                     const void* b, const libxsmm_blasint* ldb, const libxsmm_blasint* stride_b,
  const void* beta,        void* c, const libxsmm_blasint* ldc, const libxsmm_blasint* stride_c,
  libxsmm_blasint index_base, libxsmm_blasint batchsize)
{
  internal_gemm_batch_omp(iprec, oprec, transa, transb, &m, &n, &k,
    alpha, a, lda, stride_a, b, ldb, stride_b, beta, c, ldc, stride_c,
    -1/*index_stride*/, index_base, &batchsize, 1/*group_count*/);
}


LIBXSMM_APIEXT void libxsmm_gemm_groups_omp(
  libxsmm_datatype iprec, libxsmm_datatype oprec, const char transa_array[], const char transb_array[],
  const libxsmm_blasint m_array[], const libxsmm_blasint n_array[], const libxsmm_blasint k_array[],
  const void* alpha_array, const void* a_array[], const libxsmm_blasint lda_array[],
                           const void* b_array[], const libxsmm_blasint ldb_array[],
  const void* beta_array,        void* c_array[], const libxsmm_blasint ldc_array[],
  const libxsmm_blasint ngroups, const libxsmm_blasint batchsize[])
{
  internal_gemm_batch_omp(iprec, oprec, transa_array, transb_array, m_array, n_array, k_array,
    alpha_array, a_array, lda_array, NULL/*stride_a*/, b_array, ldb_array, NULL/*stride_b*/,
    beta_array, c_array, ldc_array, NULL/*stride_c*/, 0/*index_stride*/, 0/*index_base*/,
    batchsize, ngroups);
}


#if defined(LIBXSMM_BUILD) && defined(LIBXSMM_BUILD_EXT) && (!defined(LIBXSMM_NOFORTRAN) || defined(__clang_analyzer__))

/* implementation provided for Fortran 77 compatibility */
LIBXSMM_APIEXT void LIBXSMM_FSYMBOL(libxsmm_gemm_batch_omp)(const libxsmm_datatype* /*iprec*/, const libxsmm_datatype* /*oprec*/,
  const char* /*transa*/, const char* /*transb*/, const libxsmm_blasint* /*m*/, const libxsmm_blasint* /*n*/, const libxsmm_blasint* /*k*/,
  const void* /*alpha*/, const void* /*a*/, const libxsmm_blasint* /*lda*/, const libxsmm_blasint /*stride_a*/[],
  const void* /*b*/, const libxsmm_blasint* /*ldb*/, const libxsmm_blasint /*stride_b*/[],
  const void* /*beta*/, void* /*c*/, const libxsmm_blasint* /*ldc*/, const libxsmm_blasint /*stride_c*/[],
  const libxsmm_blasint* /*index_stride*/, const libxsmm_blasint* /*index_base*/, const libxsmm_blasint* /*batchsize*/);
LIBXSMM_APIEXT void LIBXSMM_FSYMBOL(libxsmm_gemm_batch_omp)(const libxsmm_datatype* iprec, const libxsmm_datatype* oprec,
  const char* transa, const char* transb, const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const void* alpha, const void* a, const libxsmm_blasint* lda, const libxsmm_blasint stride_a[],
  const void* b, const libxsmm_blasint* ldb, const libxsmm_blasint stride_b[],
  const void* beta, void* c, const libxsmm_blasint* ldc, const libxsmm_blasint stride_c[],
  const libxsmm_blasint* index_stride, const libxsmm_blasint* index_base, const libxsmm_blasint* batchsize)
{
  LIBXSMM_ASSERT(NULL != iprec && NULL != oprec && NULL != m && NULL != n && NULL != k);
  LIBXSMM_ASSERT(NULL != index_base && NULL != index_stride && NULL != batchsize);
  libxsmm_gemm_batch_omp(*iprec, *oprec, transa, transb, *m, *n, *k,
    alpha, a, lda, stride_a, b, ldb, stride_b, beta, c, ldc, stride_c,
    *index_stride, *index_base, *batchsize);
}

#endif /*defined(LIBXSMM_BUILD) && defined(LIBXSMM_BUILD_EXT) && (!defined(LIBXSMM_NOFORTRAN) || defined(__clang_analyzer__))*/
