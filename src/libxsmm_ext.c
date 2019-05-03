/******************************************************************************
** Copyright (c) 2016-2019, Intel Corporation                                **
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
#include "libxsmm_ext.h"
#include "libxsmm_gemm.h"
#include <libxsmm.h>


#if defined(LIBXSMM_BUILD)
#if defined(LIBXSMM_BUILD_EXT) && !defined(__STATIC) /* no-BLAS library */

LIBXSMM_GEMM_SYMBOL_VISIBILITY /*LIBXSMM_ATTRIBUTE_WEAK*/ void LIBXSMM_FSYMBOL(dgemm_batch)(const char* transa_array, const char* transb_array,
  const libxsmm_blasint* m_array, const libxsmm_blasint* n_array, const libxsmm_blasint* k_array,
  const double* alpha_array, const double** a_array, const libxsmm_blasint* lda_array, const double** b_array, const libxsmm_blasint* ldb_array,
  const double* beta_array, double** c_array, const libxsmm_blasint* ldc_array,
  const libxsmm_blasint* group_count, const libxsmm_blasint* group_size)
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


LIBXSMM_GEMM_SYMBOL_VISIBILITY /*LIBXSMM_ATTRIBUTE_WEAK*/ void LIBXSMM_FSYMBOL(sgemm_batch)(const char* transa_array, const char* transb_array,
  const libxsmm_blasint* m_array, const libxsmm_blasint* n_array, const libxsmm_blasint* k_array,
  const float* alpha_array, const float** a_array, const libxsmm_blasint* lda_array, const float** b_array, const libxsmm_blasint* ldb_array,
  const float* beta_array, float** c_array, const libxsmm_blasint* ldc_array,
  const libxsmm_blasint* group_count, const libxsmm_blasint* group_size)
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


LIBXSMM_GEMM_SYMBOL_VISIBILITY /*LIBXSMM_ATTRIBUTE_WEAK*/ void LIBXSMM_FSYMBOL(dgemm)(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const double* alpha, const double* a, const libxsmm_blasint* lda,
  const double* b, const libxsmm_blasint* ldb,
  const double* beta, double* c, const libxsmm_blasint* ldc)
{
  if (LIBXSMM_FSYMBOL(__real_dgemm) != libxsmm_original_dgemm_function) {
    LIBXSMM_FSYMBOL(__wrap_dgemm)(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  }
  else {
    libxsmm_blas_error("dgemm")(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  }
}


LIBXSMM_GEMM_SYMBOL_VISIBILITY /*LIBXSMM_ATTRIBUTE_WEAK*/ void LIBXSMM_FSYMBOL(sgemm)(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const float* alpha, const float* a, const libxsmm_blasint* lda,
  const float* b, const libxsmm_blasint* ldb,
  const float* beta, float* c, const libxsmm_blasint* ldc)
{
  if (LIBXSMM_FSYMBOL(__real_sgemm) != libxsmm_original_sgemm_function) {
    LIBXSMM_FSYMBOL(__wrap_sgemm)(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  }
  else {
    libxsmm_blas_error("sgemm")(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  }
}


LIBXSMM_GEMM_SYMBOL_VISIBILITY /*LIBXSMM_ATTRIBUTE_WEAK*/ void LIBXSMM_FSYMBOL(dgemv)(const char* trans, const libxsmm_blasint* m, const libxsmm_blasint* n,
  const double* alpha, const double* a, const libxsmm_blasint* lda, const double* x, const libxsmm_blasint* incx,
  const double* beta, double* y, const libxsmm_blasint* incy)
{
  if (LIBXSMM_FSYMBOL(__real_dgemv) != libxsmm_original_dgemv_function) {
    LIBXSMM_FSYMBOL(__wrap_dgemv)(trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
  }
  else {
    libxsmm_blas_error("dgemv")(trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
  }
}


LIBXSMM_GEMM_SYMBOL_VISIBILITY /*LIBXSMM_ATTRIBUTE_WEAK*/ void LIBXSMM_FSYMBOL(sgemv)(const char* trans, const libxsmm_blasint* m, const libxsmm_blasint* n,
  const float* alpha, const float* a, const libxsmm_blasint* lda, const float* x, const libxsmm_blasint* incx,
  const float* beta, float* y, const libxsmm_blasint* incy)
{
  if (LIBXSMM_FSYMBOL(__real_sgemv) != libxsmm_original_sgemv_function) {
    LIBXSMM_FSYMBOL(__wrap_sgemv)(trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
  }
  else {
    libxsmm_blas_error("sgemv")(trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
  }
}

#elif (1 == LIBXSMM_NO_BLAS)

LIBXSMM_GEMM_SYMBOL_VISIBILITY
#if defined(__GNUC__) && !defined(__PGI)
LIBXSMM_ATTRIBUTE(no_instrument_function)
#endif
void LIBXSMM_FSYMBOL(dgemm_batch)(const char* transa_array, const char* transb_array,
  const libxsmm_blasint* m_array, const libxsmm_blasint* n_array, const libxsmm_blasint* k_array,
  const double* alpha_array, const double** a_array, const libxsmm_blasint* lda_array, const double** b_array, const libxsmm_blasint* ldb_array,
  const double* beta_array, double** c_array, const libxsmm_blasint* ldc_array,
  const libxsmm_blasint* group_count, const libxsmm_blasint* group_size)
{
  libxsmm_blas_error("dgemm_batch")(transa_array, transb_array, m_array, n_array, k_array,
    alpha_array, a_array, lda_array, b_array, ldb_array, beta_array, c_array, ldc_array,
    group_count, group_size);
}


LIBXSMM_GEMM_SYMBOL_VISIBILITY
#if defined(__GNUC__) && !defined(__PGI)
LIBXSMM_ATTRIBUTE(no_instrument_function)
#endif
void LIBXSMM_FSYMBOL(sgemm_batch)(const char* transa_array, const char* transb_array,
  const libxsmm_blasint* m_array, const libxsmm_blasint* n_array, const libxsmm_blasint* k_array,
  const float* alpha_array, const float** a_array, const libxsmm_blasint* lda_array, const float** b_array, const libxsmm_blasint* ldb_array,
  const float* beta_array, float** c_array, const libxsmm_blasint* ldc_array,
  const libxsmm_blasint* group_count, const libxsmm_blasint* group_size)
{
  libxsmm_blas_error("sgemm_batch")(transa_array, transb_array, m_array, n_array, k_array,
    alpha_array, a_array, lda_array, b_array, ldb_array, beta_array, c_array, ldc_array,
    group_count, group_size);
}


LIBXSMM_GEMM_SYMBOL_VISIBILITY
#if defined(__GNUC__) && !defined(__PGI)
LIBXSMM_ATTRIBUTE(no_instrument_function)
#endif
void LIBXSMM_FSYMBOL(dgemm)(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const double* alpha, const double* a, const libxsmm_blasint* lda,
  const double* b, const libxsmm_blasint* ldb,
  const double* beta, double* c, const libxsmm_blasint* ldc)
{
  libxsmm_blas_error("dgemm")(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}


LIBXSMM_GEMM_SYMBOL_VISIBILITY
#if defined(__GNUC__) && !defined(__PGI)
LIBXSMM_ATTRIBUTE(no_instrument_function)
#endif
void LIBXSMM_FSYMBOL(sgemm)(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const float* alpha, const float* a, const libxsmm_blasint* lda,
  const float* b, const libxsmm_blasint* ldb,
  const float* beta, float* c, const libxsmm_blasint* ldc)
{
  libxsmm_blas_error("sgemm")(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}


LIBXSMM_GEMM_SYMBOL_VISIBILITY
#if defined(__GNUC__) && !defined(__PGI)
LIBXSMM_ATTRIBUTE(no_instrument_function)
#endif
void LIBXSMM_FSYMBOL(dgemv)(const char* trans, const libxsmm_blasint* m, const libxsmm_blasint* n,
  const double* alpha, const double* a, const libxsmm_blasint* lda, const double* x, const libxsmm_blasint* incx,
  const double* beta, double* y, const libxsmm_blasint* incy)
{
  libxsmm_blas_error("dgemv")(trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}


LIBXSMM_GEMM_SYMBOL_VISIBILITY
#if defined(__GNUC__) && !defined(__PGI)
LIBXSMM_ATTRIBUTE(no_instrument_function)
#endif
void LIBXSMM_FSYMBOL(sgemv)(const char* trans, const libxsmm_blasint* m, const libxsmm_blasint* n,
  const float* alpha, const float* a, const libxsmm_blasint* lda, const float* x, const libxsmm_blasint* incx,
  const float* beta, float* y, const libxsmm_blasint* incy)
{
  libxsmm_blas_error("sgemv")(trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}
#endif
#endif /*defined(LIBXSMM_BUILD) && defined(LIBXSMM_BUILD_EXT)*/

