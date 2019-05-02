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
    LIBXSMM_UNUSED(transa); LIBXSMM_UNUSED(transb); LIBXSMM_UNUSED(m); LIBXSMM_UNUSED(n); LIBXSMM_UNUSED(k);
    LIBXSMM_UNUSED(alpha); LIBXSMM_UNUSED(a); LIBXSMM_UNUSED(lda); LIBXSMM_UNUSED(b); LIBXSMM_UNUSED(ldb);
    LIBXSMM_UNUSED(beta); LIBXSMM_UNUSED(c); LIBXSMM_UNUSED(ldc);
    libxsmm_internal_wrap_error("dgemm");
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
    LIBXSMM_UNUSED(transa); LIBXSMM_UNUSED(transb); LIBXSMM_UNUSED(m); LIBXSMM_UNUSED(n); LIBXSMM_UNUSED(k);
    LIBXSMM_UNUSED(alpha); LIBXSMM_UNUSED(a); LIBXSMM_UNUSED(lda); LIBXSMM_UNUSED(b); LIBXSMM_UNUSED(ldb);
    LIBXSMM_UNUSED(beta); LIBXSMM_UNUSED(c); LIBXSMM_UNUSED(ldc);
    libxsmm_internal_wrap_error("sgemm");
  }
}


LIBXSMM_GEMM_SYMBOL_VISIBILITY /*LIBXSMM_ATTRIBUTE_WEAK*/ void LIBXSMM_FSYMBOL(dgemv)(const char* trans, const libxsmm_blasint* m, const libxsmm_blasint* n,
  const double* alpha, const double* a, const libxsmm_blasint* lda, const double* x, const libxsmm_blasint* incx,
  const double* beta, double* y, const libxsmm_blasint* incy)
{
  if (LIBXSMM_FSYMBOL(__real_dgemv) != libxsmm_original_dgemv_function) {
    LIBXSMM_FSYMBOL(__wrap_dgemv)(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  }
  else {
    LIBXSMM_UNUSED(trans); LIBXSMM_UNUSED(m); LIBXSMM_UNUSED(n); LIBXSMM_UNUSED(alpha); LIBXSMM_UNUSED(a); LIBXSMM_UNUSED(lda);
    LIBXSMM_UNUSED(x); LIBXSMM_UNUSED(incx); LIBXSMM_UNUSED(beta); LIBXSMM_UNUSED(y); LIBXSMM_UNUSED(incy);
    libxsmm_internal_wrap_error("dgemv");
  }
}


LIBXSMM_GEMM_SYMBOL_VISIBILITY /*LIBXSMM_ATTRIBUTE_WEAK*/ void LIBXSMM_FSYMBOL(sgemv)(const char* trans, const libxsmm_blasint* m, const libxsmm_blasint* n,
  const float* alpha, const float* a, const libxsmm_blasint* lda, const float* x, const libxsmm_blasint* incx,
  const float* beta, float* y, const libxsmm_blasint* incy)
{
  if (LIBXSMM_FSYMBOL(__real_sgemv) != libxsmm_original_sgemv_function) {
    LIBXSMM_FSYMBOL(__wrap_sgemv)(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  }
  else {
    LIBXSMM_UNUSED(trans); LIBXSMM_UNUSED(m); LIBXSMM_UNUSED(n); LIBXSMM_UNUSED(alpha); LIBXSMM_UNUSED(a); LIBXSMM_UNUSED(lda);
    LIBXSMM_UNUSED(x); LIBXSMM_UNUSED(incx); LIBXSMM_UNUSED(beta); LIBXSMM_UNUSED(y); LIBXSMM_UNUSED(incy);
    libxsmm_internal_wrap_error("sgemv");
  }
}

#elif (1 == LIBXSMM_NO_BLAS)

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
  LIBXSMM_UNUSED(transa); LIBXSMM_UNUSED(transb); LIBXSMM_UNUSED(m); LIBXSMM_UNUSED(n); LIBXSMM_UNUSED(k);
  LIBXSMM_UNUSED(alpha); LIBXSMM_UNUSED(a); LIBXSMM_UNUSED(lda); LIBXSMM_UNUSED(b); LIBXSMM_UNUSED(ldb);
  LIBXSMM_UNUSED(beta); LIBXSMM_UNUSED(c); LIBXSMM_UNUSED(ldc);
  libxsmm_internal_wrap_error("dgemm");
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
  LIBXSMM_UNUSED(transa); LIBXSMM_UNUSED(transb); LIBXSMM_UNUSED(m); LIBXSMM_UNUSED(n); LIBXSMM_UNUSED(k);
  LIBXSMM_UNUSED(alpha); LIBXSMM_UNUSED(a); LIBXSMM_UNUSED(lda); LIBXSMM_UNUSED(b); LIBXSMM_UNUSED(ldb);
  LIBXSMM_UNUSED(beta); LIBXSMM_UNUSED(c); LIBXSMM_UNUSED(ldc);
  libxsmm_internal_wrap_error("sgemm");
}


LIBXSMM_GEMM_SYMBOL_VISIBILITY
#if defined(__GNUC__) && !defined(__PGI)
LIBXSMM_ATTRIBUTE(no_instrument_function)
#endif
void LIBXSMM_FSYMBOL(dgemv)(const char* trans, const libxsmm_blasint* m, const libxsmm_blasint* n,
  const double* alpha, const double* a, const libxsmm_blasint* lda, const double* x, const libxsmm_blasint* incx,
  const double* beta, double* y, const libxsmm_blasint* incy)
{
  LIBXSMM_UNUSED(trans); LIBXSMM_UNUSED(m); LIBXSMM_UNUSED(n); LIBXSMM_UNUSED(alpha); LIBXSMM_UNUSED(a); LIBXSMM_UNUSED(lda);
  LIBXSMM_UNUSED(x); LIBXSMM_UNUSED(incx); LIBXSMM_UNUSED(beta); LIBXSMM_UNUSED(y); LIBXSMM_UNUSED(incy);
  libxsmm_internal_wrap_error("dgemv");
}


LIBXSMM_GEMM_SYMBOL_VISIBILITY
#if defined(__GNUC__) && !defined(__PGI)
LIBXSMM_ATTRIBUTE(no_instrument_function)
#endif
void LIBXSMM_FSYMBOL(sgemv)(const char* trans, const libxsmm_blasint* m, const libxsmm_blasint* n,
  const float* alpha, const float* a, const libxsmm_blasint* lda, const float* x, const libxsmm_blasint* incx,
  const float* beta, float* y, const libxsmm_blasint* incy)
{
  LIBXSMM_UNUSED(trans); LIBXSMM_UNUSED(m); LIBXSMM_UNUSED(n); LIBXSMM_UNUSED(alpha); LIBXSMM_UNUSED(a); LIBXSMM_UNUSED(lda);
  LIBXSMM_UNUSED(x); LIBXSMM_UNUSED(incx); LIBXSMM_UNUSED(beta); LIBXSMM_UNUSED(y); LIBXSMM_UNUSED(incy);
  libxsmm_internal_wrap_error("sgemv");
}
#endif
#endif /*defined(LIBXSMM_BUILD) && defined(LIBXSMM_BUILD_EXT)*/

