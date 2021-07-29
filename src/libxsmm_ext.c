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
#include "libxsmm_ext.h"
#include "libxsmm_gemm.h"
#include <libxsmm.h>


#if defined(LIBXSMM_BUILD)
#if defined(LIBXSMM_BUILD_EXT) && !defined(__STATIC)

LIBXSMM_BLAS_SYMBOL_VISIBILITY LIBXSMM_ATTRIBUTE_WEAK
void LIBXSMM_FSYMBOL(dgemm_batch)(const char transa_array[], const char transb_array[],
  const libxsmm_blasint m_array[], const libxsmm_blasint n_array[], const libxsmm_blasint k_array[],
  const double alpha_array[], const double* a_array[], const libxsmm_blasint lda_array[],
  const double* b_array[], const libxsmm_blasint ldb_array[],
  const double beta_array[], double* c_array[], const libxsmm_blasint ldc_array[],
  const libxsmm_blasint* group_count, const libxsmm_blasint group_size[]) LIBXSMM_BLAS_NOEXCEPT(gemm_batch)
{
  if (LIBXSMM_FSYMBOL(__real_dgemm_batch) != libxsmm_original_dgemm_batch_function) {
    LIBXSMM_FSYMBOL(__wrap_dgemm_batch)(transa_array, transb_array, m_array, n_array, k_array,
      alpha_array, a_array, lda_array, b_array, ldb_array, beta_array, c_array, ldc_array,
      group_count, group_size);
  }
  else {
    libxsmm_blas_error("dgemm_batch")(transa_array, transb_array, m_array, n_array, k_array,
      alpha_array, a_array, lda_array, b_array, ldb_array, beta_array, c_array, ldc_array,
      group_count, group_size);
  }
}


LIBXSMM_BLAS_SYMBOL_VISIBILITY LIBXSMM_ATTRIBUTE_WEAK
void LIBXSMM_FSYMBOL(sgemm_batch)(const char transa_array[], const char transb_array[],
  const libxsmm_blasint m_array[], const libxsmm_blasint n_array[], const libxsmm_blasint k_array[],
  const float alpha_array[], const float* a_array[], const libxsmm_blasint lda_array[],
  const float* b_array[], const libxsmm_blasint ldb_array[],
  const float beta_array[], float* c_array[], const libxsmm_blasint ldc_array[],
  const libxsmm_blasint* group_count, const libxsmm_blasint group_size[]) LIBXSMM_BLAS_NOEXCEPT(gemm_batch)
{
  if (LIBXSMM_FSYMBOL(__real_sgemm_batch) != libxsmm_original_sgemm_batch_function) {
    LIBXSMM_FSYMBOL(__wrap_sgemm_batch)(transa_array, transb_array, m_array, n_array, k_array,
      alpha_array, a_array, lda_array, b_array, ldb_array, beta_array, c_array, ldc_array,
      group_count, group_size);
  }
  else {
    libxsmm_blas_error("sgemm_batch")(transa_array, transb_array, m_array, n_array, k_array,
      alpha_array, a_array, lda_array, b_array, ldb_array, beta_array, c_array, ldc_array,
      group_count, group_size);
  }
}


LIBXSMM_BLAS_SYMBOL_VISIBILITY LIBXSMM_ATTRIBUTE_WEAK
void LIBXSMM_FSYMBOL(dgemm)(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const double* alpha, const double* a, const libxsmm_blasint* lda,
  const double* b, const libxsmm_blasint* ldb,
  const double* beta, double* c, const libxsmm_blasint* ldc) LIBXSMM_BLAS_NOEXCEPT(gemm)
{
  if (LIBXSMM_FSYMBOL(__real_dgemm) != libxsmm_original_dgemm_function) {
    LIBXSMM_FSYMBOL(__wrap_dgemm)(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  }
  else {
    libxsmm_blas_error("dgemm")(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    LIBXSMM_INLINE_XGEMM(double, double, /* try producing a result even if LIBXSMM_INLINE_XGEMM is limited */
      transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  }
}


LIBXSMM_BLAS_SYMBOL_VISIBILITY LIBXSMM_ATTRIBUTE_WEAK
void LIBXSMM_FSYMBOL(sgemm)(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const float* alpha, const float* a, const libxsmm_blasint* lda,
  const float* b, const libxsmm_blasint* ldb,
  const float* beta, float* c, const libxsmm_blasint* ldc) LIBXSMM_BLAS_NOEXCEPT(gemm)
{
  if (LIBXSMM_FSYMBOL(__real_sgemm) != libxsmm_original_sgemm_function) {
    LIBXSMM_FSYMBOL(__wrap_sgemm)(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  }
  else {
    libxsmm_blas_error("sgemm")(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    LIBXSMM_INLINE_XGEMM(float, float, /* try producing a result even if LIBXSMM_INLINE_XGEMM is limited */
      transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  }
}


LIBXSMM_BLAS_SYMBOL_VISIBILITY LIBXSMM_ATTRIBUTE_WEAK
void LIBXSMM_FSYMBOL(dgemv)(const char* trans, const libxsmm_blasint* m, const libxsmm_blasint* n,
  const double* alpha, const double* a, const libxsmm_blasint* lda, const double* x, const libxsmm_blasint* incx,
  const double* beta, double* y, const libxsmm_blasint* incy) LIBXSMM_BLAS_NOEXCEPT(gemv)
{
  if (LIBXSMM_FSYMBOL(__real_dgemv) != libxsmm_original_dgemv_function) {
    LIBXSMM_FSYMBOL(__wrap_dgemv)(trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
  }
  else {
    libxsmm_blas_error("dgemv")(trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
  }
}


LIBXSMM_BLAS_SYMBOL_VISIBILITY LIBXSMM_ATTRIBUTE_WEAK
void LIBXSMM_FSYMBOL(sgemv)(const char* trans, const libxsmm_blasint* m, const libxsmm_blasint* n,
  const float* alpha, const float* a, const libxsmm_blasint* lda, const float* x, const libxsmm_blasint* incx,
  const float* beta, float* y, const libxsmm_blasint* incy) LIBXSMM_BLAS_NOEXCEPT(gemv)
{
  if (LIBXSMM_FSYMBOL(__real_sgemv) != libxsmm_original_sgemv_function) {
    LIBXSMM_FSYMBOL(__wrap_sgemv)(trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
  }
  else {
    libxsmm_blas_error("sgemv")(trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
  }
}


LIBXSMM_BLAS_SYMBOL_VISIBILITY LIBXSMM_ATTRIBUTE_WEAK
void dgemm_batch(const char transa_array[], const char transb_array[],
  const libxsmm_blasint m_array[], const libxsmm_blasint n_array[], const libxsmm_blasint k_array[],
  const double alpha_array[], const double* a_array[], const libxsmm_blasint lda_array[],
  const double* b_array[], const libxsmm_blasint ldb_array[],
  const double beta_array[], double* c_array[], const libxsmm_blasint ldc_array[],
  const libxsmm_blasint* group_count, const libxsmm_blasint group_size[]) LIBXSMM_BLAS_NOEXCEPT(gemm_batch)
{
  LIBXSMM_FSYMBOL(dgemm_batch)(transa_array, transb_array, m_array, n_array, k_array,
    alpha_array, a_array, lda_array, b_array, ldb_array, beta_array, c_array, ldc_array,
    group_count, group_size);
}


LIBXSMM_BLAS_SYMBOL_VISIBILITY LIBXSMM_ATTRIBUTE_WEAK
void sgemm_batch(const char transa_array[], const char transb_array[],
  const libxsmm_blasint m_array[], const libxsmm_blasint n_array[], const libxsmm_blasint k_array[],
  const float alpha_array[], const float* a_array[], const libxsmm_blasint lda_array[],
  const float* b_array[], const libxsmm_blasint ldb_array[],
  const float beta_array[], float* c_array[], const libxsmm_blasint ldc_array[],
  const libxsmm_blasint* group_count, const libxsmm_blasint group_size[]) LIBXSMM_BLAS_NOEXCEPT(gemm_batch)
{
  LIBXSMM_FSYMBOL(sgemm_batch)(transa_array, transb_array, m_array, n_array, k_array,
    alpha_array, a_array, lda_array, b_array, ldb_array, beta_array, c_array, ldc_array,
    group_count, group_size);
}

#elif (0 != LIBXSMM_NO_BLAS) /* no-BLAS library */

LIBXSMM_APIVAR_PUBLIC_DEF(LIBXSMM_ATTRIBUTE_COMMON unsigned int libxsmm_intrinsics_mm512_rng_state0[16]);
LIBXSMM_APIVAR_PUBLIC_DEF(LIBXSMM_ATTRIBUTE_COMMON unsigned int libxsmm_intrinsics_mm512_rng_state1[16]);
LIBXSMM_APIVAR_PUBLIC_DEF(LIBXSMM_ATTRIBUTE_COMMON unsigned int libxsmm_intrinsics_mm512_rng_state2[16]);
LIBXSMM_APIVAR_PUBLIC_DEF(LIBXSMM_ATTRIBUTE_COMMON unsigned int libxsmm_intrinsics_mm512_rng_state3[16]);

LIBXSMM_API_INTERN LIBXSMM_ATTRIBUTE_NO_TRACE void internal_noblas_sink(LIBXSMM_VARIADIC);
LIBXSMM_API_INTERN void internal_noblas_sink(LIBXSMM_VARIADIC)
{
  /* does nothing else but sinking given arguments */
}

LIBXSMM_API_INTERN LIBXSMM_ATTRIBUTE_NO_TRACE libxsmm_sink_function internal_noblas_error(const char* /*symbol*/);
LIBXSMM_API_INTERN libxsmm_sink_function internal_noblas_error(const char* symbol)
{
  static int internal_noblas_nerror = 0;
  LIBXSMM_BLAS_ERROR(symbol, &internal_noblas_nerror);
  return internal_noblas_sink;
}


LIBXSMM_BLAS_SYMBOL_VISIBILITY LIBXSMM_ATTRIBUTE_NO_TRACE /*LIBXSMM_ATTRIBUTE_WEAK*/
void LIBXSMM_FSYMBOL(dgemm_batch)(const char transa_array[], const char transb_array[],
  const libxsmm_blasint m_array[], const libxsmm_blasint n_array[], const libxsmm_blasint k_array[],
  const double alpha_array[], const double* a_array[], const libxsmm_blasint lda_array[],
  const double* b_array[], const libxsmm_blasint ldb_array[],
  const double beta_array[], double* c_array[], const libxsmm_blasint ldc_array[],
  const libxsmm_blasint* group_count, const libxsmm_blasint group_size[]) LIBXSMM_BLAS_NOEXCEPT(gemm_batch)
{
  internal_noblas_error("dgemm_batch")(transa_array, transb_array, m_array, n_array, k_array,
    alpha_array, a_array, lda_array, b_array, ldb_array, beta_array, c_array, ldc_array,
    group_count, group_size);
}


LIBXSMM_BLAS_SYMBOL_VISIBILITY LIBXSMM_ATTRIBUTE_NO_TRACE /*LIBXSMM_ATTRIBUTE_WEAK*/
void LIBXSMM_FSYMBOL(sgemm_batch)(const char transa_array[], const char transb_array[],
  const libxsmm_blasint m_array[], const libxsmm_blasint n_array[], const libxsmm_blasint k_array[],
  const float alpha_array[], const float* a_array[], const libxsmm_blasint lda_array[],
  const float* b_array[], const libxsmm_blasint ldb_array[],
  const float beta_array[], float* c_array[], const libxsmm_blasint ldc_array[],
  const libxsmm_blasint* group_count, const libxsmm_blasint group_size[]) LIBXSMM_BLAS_NOEXCEPT(gemm_batch)
{
  internal_noblas_error("sgemm_batch")(transa_array, transb_array, m_array, n_array, k_array,
    alpha_array, a_array, lda_array, b_array, ldb_array, beta_array, c_array, ldc_array,
    group_count, group_size);
}


LIBXSMM_BLAS_SYMBOL_VISIBILITY LIBXSMM_ATTRIBUTE_NO_TRACE /*LIBXSMM_ATTRIBUTE_WEAK*/
void LIBXSMM_FSYMBOL(dgemm)(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const double* alpha, const double* a, const libxsmm_blasint* lda,
  const double* b, const libxsmm_blasint* ldb,
  const double* beta, double* c, const libxsmm_blasint* ldc) LIBXSMM_BLAS_NOEXCEPT(gemm)
{
  internal_noblas_error("dgemm")(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}


LIBXSMM_BLAS_SYMBOL_VISIBILITY LIBXSMM_ATTRIBUTE_NO_TRACE /*LIBXSMM_ATTRIBUTE_WEAK*/
void LIBXSMM_FSYMBOL(sgemm)(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const float* alpha, const float* a, const libxsmm_blasint* lda,
  const float* b, const libxsmm_blasint* ldb,
  const float* beta, float* c, const libxsmm_blasint* ldc) LIBXSMM_BLAS_NOEXCEPT(gemm)
{
  internal_noblas_error("sgemm")(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}


LIBXSMM_BLAS_SYMBOL_VISIBILITY LIBXSMM_ATTRIBUTE_NO_TRACE /*LIBXSMM_ATTRIBUTE_WEAK*/
void LIBXSMM_FSYMBOL(dgemv)(const char* trans, const libxsmm_blasint* m, const libxsmm_blasint* n,
  const double* alpha, const double* a, const libxsmm_blasint* lda, const double* x, const libxsmm_blasint* incx,
  const double* beta, double* y, const libxsmm_blasint* incy) LIBXSMM_BLAS_NOEXCEPT(gemv)
{
  internal_noblas_error("dgemv")(trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}


LIBXSMM_BLAS_SYMBOL_VISIBILITY LIBXSMM_ATTRIBUTE_NO_TRACE /*LIBXSMM_ATTRIBUTE_WEAK*/
void LIBXSMM_FSYMBOL(sgemv)(const char* trans, const libxsmm_blasint* m, const libxsmm_blasint* n,
  const float* alpha, const float* a, const libxsmm_blasint* lda, const float* x, const libxsmm_blasint* incx,
  const float* beta, float* y, const libxsmm_blasint* incy) LIBXSMM_BLAS_NOEXCEPT(gemv)
{
  internal_noblas_error("sgemv")(trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}


LIBXSMM_BLAS_SYMBOL_VISIBILITY LIBXSMM_ATTRIBUTE_NO_TRACE
void dgemm_batch(const char transa_array[], const char transb_array[],
  const libxsmm_blasint m_array[], const libxsmm_blasint n_array[], const libxsmm_blasint k_array[],
  const double alpha_array[], const double* a_array[], const libxsmm_blasint lda_array[],
  const double* b_array[], const libxsmm_blasint ldb_array[],
  const double beta_array[], double* c_array[], const libxsmm_blasint ldc_array[],
  const libxsmm_blasint* group_count, const libxsmm_blasint group_size[]) LIBXSMM_BLAS_NOEXCEPT(gemm_batch)
{
  LIBXSMM_FSYMBOL(dgemm_batch)(transa_array, transb_array, m_array, n_array, k_array,
    alpha_array, a_array, lda_array, b_array, ldb_array, beta_array, c_array, ldc_array,
    group_count, group_size);
}


LIBXSMM_BLAS_SYMBOL_VISIBILITY LIBXSMM_ATTRIBUTE_NO_TRACE
void sgemm_batch(const char transa_array[], const char transb_array[],
  const libxsmm_blasint m_array[], const libxsmm_blasint n_array[], const libxsmm_blasint k_array[],
  const float alpha_array[], const float* a_array[], const libxsmm_blasint lda_array[],
  const float* b_array[], const libxsmm_blasint ldb_array[],
  const float beta_array[], float* c_array[], const libxsmm_blasint ldc_array[],
  const libxsmm_blasint* group_count, const libxsmm_blasint group_size[]) LIBXSMM_BLAS_NOEXCEPT(gemm_batch)
{
  LIBXSMM_FSYMBOL(sgemm_batch)(transa_array, transb_array, m_array, n_array, k_array,
    alpha_array, a_array, lda_array, b_array, ldb_array, beta_array, c_array, ldc_array,
    group_count, group_size);
}

#endif
#endif /*defined(LIBXSMM_BUILD)*/

