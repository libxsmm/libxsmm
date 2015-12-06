/******************************************************************************
** Copyright (c) 2013-2015, Intel Corporation                                **
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


LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE void libxsmm_sgemm(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const float* alpha, const float* a, const libxsmm_blasint* lda,
  const float* b, const libxsmm_blasint* ldb,
  const float* beta, float* c, const libxsmm_blasint* ldc)
{
  int flags = LIBXSMM_FLAGS;
  flags = (0 != transa
      ? (('N' == *transa || 'n' == *transa) ? (flags & ~LIBXSMM_GEMM_FLAG_TRANS_A)
                                            : (flags |  LIBXSMM_GEMM_FLAG_TRANS_A))
      : flags);
  flags = (0 != transb
      ? (('N' == *transb || 'n' == *transb) ? (flags & ~LIBXSMM_GEMM_FLAG_TRANS_B)
                                            : (flags |  LIBXSMM_GEMM_FLAG_TRANS_B))
      : flags);
  assert(m && n && k && a && b && c);
  LIBXSMM_SGEMM(flags, *m, *n, *k,
    0 != alpha ? *alpha : ((float)LIBXSMM_ALPHA),
    a, *(lda ? lda : m), b, *(ldb ? ldb : k),
    0 != beta ? *beta : ((float)LIBXSMM_BETA),
    c, *(ldc ? ldc : m));
}


LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE void libxsmm_dgemm(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const double* alpha, const double* a, const libxsmm_blasint* lda,
  const double* b, const libxsmm_blasint* ldb,
  const double* beta, double* c, const libxsmm_blasint* ldc)
{
  int flags = LIBXSMM_FLAGS;
  flags = (0 != transa
      ? (('N' == *transa || 'n' == *transa) ? (flags & ~LIBXSMM_GEMM_FLAG_TRANS_A)
                                            : (flags |  LIBXSMM_GEMM_FLAG_TRANS_A))
      : flags);
  flags = (0 != transb
      ? (('N' == *transb || 'n' == *transb) ? (flags & ~LIBXSMM_GEMM_FLAG_TRANS_B)
                                            : (flags |  LIBXSMM_GEMM_FLAG_TRANS_B))
      : flags);
  assert(m && n && k && a && b && c);
  LIBXSMM_DGEMM(flags, *m, *n, *k,
    0 != alpha ? *alpha : ((double)LIBXSMM_ALPHA),
    a, *(lda ? lda : m), b, *(ldb ? ldb : k),
    0 != beta ? *beta : ((double)LIBXSMM_BETA),
    c, *(ldc ? ldc : m));
}


LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE void libxsmm_blas_sgemm(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const float* alpha, const float* a, const libxsmm_blasint* lda,
  const float* b, const libxsmm_blasint* ldb,
  const float* beta, float* c, const libxsmm_blasint* ldc)
{
  int flags = LIBXSMM_FLAGS;
  flags = (0 != transa
      ? (('N' == *transa || 'n' == *transa) ? (flags & ~LIBXSMM_GEMM_FLAG_TRANS_A)
                                            : (flags |  LIBXSMM_GEMM_FLAG_TRANS_A))
      : flags);
  flags = (0 != transb
      ? (('N' == *transb || 'n' == *transb) ? (flags & ~LIBXSMM_GEMM_FLAG_TRANS_B)
                                            : (flags |  LIBXSMM_GEMM_FLAG_TRANS_B))
      : flags);
  assert(m && n && k && a && b && c);
  LIBXSMM_BLAS_SGEMM(flags, *m, *n, *k,
    0 != alpha ? *alpha : ((float)LIBXSMM_ALPHA),
    a, *(lda ? lda : m), b, *(ldb ? ldb : k),
    0 != beta ? *beta : ((float)LIBXSMM_BETA),
    c, *(ldc ? ldc : m));
}


LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE void libxsmm_blas_dgemm(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const double* alpha, const double* a, const libxsmm_blasint* lda,
  const double* b, const libxsmm_blasint* ldb,
  const double* beta, double* c, const libxsmm_blasint* ldc)
{
  int flags = LIBXSMM_FLAGS;
  flags = (0 != transa
      ? (('N' == *transa || 'n' == *transa) ? (flags & ~LIBXSMM_GEMM_FLAG_TRANS_A)
                                            : (flags |  LIBXSMM_GEMM_FLAG_TRANS_A))
      : flags);
  flags = (0 != transb
      ? (('N' == *transb || 'n' == *transb) ? (flags & ~LIBXSMM_GEMM_FLAG_TRANS_B)
                                            : (flags |  LIBXSMM_GEMM_FLAG_TRANS_B))
      : flags);
  assert(m && n && k && a && b && c);
  LIBXSMM_BLAS_DGEMM(flags, *m, *n, *k,
    0 != alpha ? *alpha : ((double)LIBXSMM_ALPHA),
    a, *(lda ? lda : m), b, *(ldb ? ldb : k),
    0 != beta ? *beta : ((double)LIBXSMM_BETA),
    c, *(ldc ? ldc : m));
}


#if defined(__STATIC) && defined(__GNUC__)
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE LIBXSMM_ATTRIBUTE(weak) void LIBXSMM_FSYMBOL(__real_sgemm)(const char*, const char*,
  const libxsmm_blasint*, const libxsmm_blasint*, const libxsmm_blasint*,
  const float*, const float*, const libxsmm_blasint*,
  const float* b, const libxsmm_blasint*,
  const float* beta, float*, const libxsmm_blasint*);
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE void LIBXSMM_FSYMBOL(__wrap_sgemm)(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const float* alpha, const float* a, const libxsmm_blasint* lda,
  const float* b, const libxsmm_blasint* ldb,
  const float* beta, float* c, const libxsmm_blasint* ldc)
{
  int flags = LIBXSMM_FLAGS;
  flags = (0 != transa
      ? (('N' == *transa || 'n' == *transa) ? (flags & ~LIBXSMM_GEMM_FLAG_TRANS_A)
                                            : (flags |  LIBXSMM_GEMM_FLAG_TRANS_A))
      : flags);
  flags = (0 != transb
      ? (('N' == *transb || 'n' == *transb) ? (flags & ~LIBXSMM_GEMM_FLAG_TRANS_B)
                                            : (flags |  LIBXSMM_GEMM_FLAG_TRANS_B))
      : flags);
  assert(m && n && k && a && b && c);
  LIBXSMM_XGEMM(float, libxsmm_blasint, LIBXSMM_FSYMBOL(__real_sgemm), flags, *m, *n, *k,
    0 != alpha ? *alpha : ((float)LIBXSMM_ALPHA),
    a, *(lda ? lda : m), b, *(ldb ? ldb : k),
    0 != beta ? *beta : ((float)LIBXSMM_BETA),
    c, *(ldc ? ldc : m));
}


LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE LIBXSMM_ATTRIBUTE(weak) void LIBXSMM_FSYMBOL(__real_dgemm)(const char*, const char*,
  const libxsmm_blasint*, const libxsmm_blasint*, const libxsmm_blasint*,
  const double*, const double*, const libxsmm_blasint*,
  const double* b, const libxsmm_blasint*,
  const double* beta, double*, const libxsmm_blasint*);
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE void LIBXSMM_FSYMBOL(__wrap_dgemm)(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const double* alpha, const double* a, const libxsmm_blasint* lda,
  const double* b, const libxsmm_blasint* ldb,
  const double* beta, double* c, const libxsmm_blasint* ldc)
{
  int flags = LIBXSMM_FLAGS;
  flags = (0 != transa
      ? (('N' == *transa || 'n' == *transa) ? (flags & ~LIBXSMM_GEMM_FLAG_TRANS_A)
                                            : (flags |  LIBXSMM_GEMM_FLAG_TRANS_A))
      : flags);
  flags = (0 != transb
      ? (('N' == *transb || 'n' == *transb) ? (flags & ~LIBXSMM_GEMM_FLAG_TRANS_B)
                                            : (flags |  LIBXSMM_GEMM_FLAG_TRANS_B))
      : flags);
  assert(m && n && k && a && b && c);
  LIBXSMM_XGEMM(double, libxsmm_blasint, LIBXSMM_FSYMBOL(__real_dgemm), flags, *m, *n, *k,
    0 != alpha ? *alpha : ((double)LIBXSMM_ALPHA),
    a, *(lda ? lda : m), b, *(ldb ? ldb : k),
    0 != beta ? *beta : ((double)LIBXSMM_BETA),
    c, *(ldc ? ldc : m));
}
#endif /*defined(__STATIC) && defined(__GNUC__)*/
