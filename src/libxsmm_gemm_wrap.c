/******************************************************************************
** Copyright (c) 2016, Intel Corporation                                     **
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
#include "libxsmm_gemm.h"
#include "libxsmm_ext_gemm.h"
#include "libxsmm_sync.h"

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#if !defined(NDEBUG)
# include <stdio.h>
#endif
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif


/* must be located in a different translation unit than libxsmm_original_sgemm */
LIBXSMM_API_DEFINITION void LIBXSMM_FSYMBOL(__real_sgemm)(
  const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const float* alpha, const float* a, const libxsmm_blasint* lda,
  const float* b, const libxsmm_blasint* ldb,
  const float* beta, float* c, const libxsmm_blasint* ldc)
{
  static const LIBXSMM_RETARGETABLE libxsmm_sgemm_function instance = LIBXSMM_FSYMBOL(sgemm);
#if !defined(NDEBUG)
  if (0 != instance)
#endif
  {
    instance(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  }
#if !defined(NDEBUG) /* library code is expected to be mute */
  else {
    static LIBXSMM_TLS int error_blas = 0;
    if (0 == error_blas) {
      fprintf(stderr, "LIBXSMM: application must be linked against a LAPACK/BLAS implementation!\n");
      error_blas = 1;
    }
  }
#endif
}


/* must be located in a different translation unit than libxsmm_original_dgemm */
LIBXSMM_API_DEFINITION void LIBXSMM_FSYMBOL(__real_dgemm)(
  const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const double* alpha, const double* a, const libxsmm_blasint* lda,
  const double* b, const libxsmm_blasint* ldb,
  const double* beta, double* c, const libxsmm_blasint* ldc)
{
  static const LIBXSMM_RETARGETABLE libxsmm_dgemm_function instance = LIBXSMM_FSYMBOL(dgemm);
#if !defined(NDEBUG)
  if (0 != instance)
#endif
  {
    instance(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
  }
#if !defined(NDEBUG) /* library code is expected to be mute */
  else {
    static LIBXSMM_TLS int error_blas = 0;
    if (0 == error_blas) {
      fprintf(stderr, "LIBXSMM: application must be linked against a LAPACK/BLAS implementation!\n");
      error_blas = 1;
    }
  }
#endif
}

