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
#include "libxsmm_ext.h"
#include "libxsmm_gemm.h"


#if defined(LIBXSMM_BUILD)
#if (defined(LIBXSMM_BUILD_EXT) && !defined(__STATIC))

LIBXSMM_API_DEFINITION void LIBXSMM_FSYMBOL(sgemm)(LIBXSMM_GEMM_CONST char* transa, LIBXSMM_GEMM_CONST char* transb,
  LIBXSMM_GEMM_CONST libxsmm_blasint* m, LIBXSMM_GEMM_CONST libxsmm_blasint* n, LIBXSMM_GEMM_CONST libxsmm_blasint* k,
  LIBXSMM_GEMM_CONST float* alpha, LIBXSMM_GEMM_CONST float* a, LIBXSMM_GEMM_CONST libxsmm_blasint* lda,
  LIBXSMM_GEMM_CONST float* b, LIBXSMM_GEMM_CONST libxsmm_blasint* ldb,
  LIBXSMM_GEMM_CONST float* beta, float* c, LIBXSMM_GEMM_CONST libxsmm_blasint* ldc)
{
  libxsmm_sgemm_omp(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}


LIBXSMM_API_DEFINITION void LIBXSMM_FSYMBOL(dgemm)(LIBXSMM_GEMM_CONST char* transa, LIBXSMM_GEMM_CONST char* transb,
  LIBXSMM_GEMM_CONST libxsmm_blasint* m, LIBXSMM_GEMM_CONST libxsmm_blasint* n, LIBXSMM_GEMM_CONST libxsmm_blasint* k,
  LIBXSMM_GEMM_CONST double* alpha, LIBXSMM_GEMM_CONST double* a, LIBXSMM_GEMM_CONST libxsmm_blasint* lda,
  LIBXSMM_GEMM_CONST double* b, LIBXSMM_GEMM_CONST libxsmm_blasint* ldb,
  LIBXSMM_GEMM_CONST double* beta, double* c, LIBXSMM_GEMM_CONST libxsmm_blasint* ldc)
{
  libxsmm_dgemm_omp(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

#elif (1 == LIBXSMM_NO_BLAS)

#if defined(__GNUC__)
LIBXSMM_ATTRIBUTE(no_instrument_function)
#endif
LIBXSMM_API_DEFINITION void LIBXSMM_FSYMBOL(sgemm)(LIBXSMM_GEMM_CONST char* transa, LIBXSMM_GEMM_CONST char* transb,
  LIBXSMM_GEMM_CONST libxsmm_blasint* m, LIBXSMM_GEMM_CONST libxsmm_blasint* n, LIBXSMM_GEMM_CONST libxsmm_blasint* k,
  LIBXSMM_GEMM_CONST float* alpha, LIBXSMM_GEMM_CONST float* a, LIBXSMM_GEMM_CONST libxsmm_blasint* lda,
  LIBXSMM_GEMM_CONST float* b, LIBXSMM_GEMM_CONST libxsmm_blasint* ldb,
  LIBXSMM_GEMM_CONST float* beta, float* c, LIBXSMM_GEMM_CONST libxsmm_blasint* ldc)
{
#if !defined(NDEBUG) /* library code is expected to be mute */
  static int error_once = 0;
  if (1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED)) {
    fprintf(stderr, "LIBXSMM: application must be linked against a LAPACK/BLAS implementation!\n");
  }
#endif
  LIBXSMM_UNUSED(transa); LIBXSMM_UNUSED(transb); LIBXSMM_UNUSED(m); LIBXSMM_UNUSED(n); LIBXSMM_UNUSED(k);
  LIBXSMM_UNUSED(alpha); LIBXSMM_UNUSED(a); LIBXSMM_UNUSED(lda); LIBXSMM_UNUSED(b); LIBXSMM_UNUSED(ldb);
  LIBXSMM_UNUSED(beta); LIBXSMM_UNUSED(c); LIBXSMM_UNUSED(ldc);
}


#if defined(__GNUC__)
LIBXSMM_ATTRIBUTE(no_instrument_function)
#endif
LIBXSMM_API_DEFINITION void LIBXSMM_FSYMBOL(dgemm)(LIBXSMM_GEMM_CONST char* transa, LIBXSMM_GEMM_CONST char* transb,
  LIBXSMM_GEMM_CONST libxsmm_blasint* m, LIBXSMM_GEMM_CONST libxsmm_blasint* n, LIBXSMM_GEMM_CONST libxsmm_blasint* k,
  LIBXSMM_GEMM_CONST double* alpha, LIBXSMM_GEMM_CONST double* a, LIBXSMM_GEMM_CONST libxsmm_blasint* lda,
  LIBXSMM_GEMM_CONST double* b, LIBXSMM_GEMM_CONST libxsmm_blasint* ldb,
  LIBXSMM_GEMM_CONST double* beta, double* c, LIBXSMM_GEMM_CONST libxsmm_blasint* ldc)
{
#if !defined(NDEBUG) /* library code is expected to be mute */
  static int error_once = 0;
  if (1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED)) {
    fprintf(stderr, "LIBXSMM: application must be linked against a LAPACK/BLAS implementation!\n");
  }
#endif
  LIBXSMM_UNUSED(transa); LIBXSMM_UNUSED(transb); LIBXSMM_UNUSED(m); LIBXSMM_UNUSED(n); LIBXSMM_UNUSED(k);
  LIBXSMM_UNUSED(alpha); LIBXSMM_UNUSED(a); LIBXSMM_UNUSED(lda); LIBXSMM_UNUSED(b); LIBXSMM_UNUSED(ldb);
  LIBXSMM_UNUSED(beta); LIBXSMM_UNUSED(c); LIBXSMM_UNUSED(ldc);
}

#endif
#endif /*defined(LIBXSMM_BUILD)*/

