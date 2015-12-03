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
  const float* alpha, const float *LIBXSMM_RESTRICT a, const libxsmm_blasint* lda,
  const float *LIBXSMM_RESTRICT b, const libxsmm_blasint* ldb,
  const float* beta, float *LIBXSMM_RESTRICT c, const libxsmm_blasint* ldc)
{
  libxsmm_blasint ilda, ildb;
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
  ilda = *(lda ? lda : m);
  ildb = *(ldb ? ldb : n);
  LIBXSMM_GEMM(float, libxsmm_blasint, flags, *LIBXSMM_LD(m, n), *LIBXSMM_LD(n, m), *k,
    0 != alpha ? (1.f == *alpha ? 1.f : (-1.f == *alpha ? -1.f : *alpha)) : ((float)LIBXSMM_ALPHA),
    LIBXSMM_LD(a, b), LIBXSMM_LD(ilda, ildb), LIBXSMM_LD(b, a), LIBXSMM_LD(ildb, ilda),
    0 != beta  ? (1.f == *beta  ? 1.f : ( 0.f == *beta  ?  0.f : *beta))  : ((float)LIBXSMM_BETA),
    c, ldc ? *ldc : *LIBXSMM_LD(m, n));
}


LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE void libxsmm_dgemm(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const double* alpha, const double *LIBXSMM_RESTRICT a, const libxsmm_blasint* lda,
  const double *LIBXSMM_RESTRICT b, const libxsmm_blasint* ldb,
  const double* beta, double *LIBXSMM_RESTRICT c, const libxsmm_blasint* ldc)
{
  libxsmm_blasint ilda, ildb;
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
  ilda = *(lda ? lda : m);
  ildb = *(ldb ? ldb : n);
  LIBXSMM_GEMM(double, libxsmm_blasint, flags, *LIBXSMM_LD(m, n), *LIBXSMM_LD(n, m), *k,
    0 != alpha ? (1.0 == *alpha ? 1.0 : (-1.0 == *alpha ? -1.0 : *alpha)) : ((double)LIBXSMM_ALPHA),
    LIBXSMM_LD(a, b), LIBXSMM_LD(ilda, ildb), LIBXSMM_LD(b, a), LIBXSMM_LD(ildb, ilda),
    0 != beta  ? (1.0 == *beta  ? 1.0 : ( 0.0 == *beta  ?  0.0 : *beta))  : ((double)LIBXSMM_BETA),
    c, ldc ? *ldc : *LIBXSMM_LD(m, n));
}


LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE void libxsmm_blas_sgemm(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const float* alpha, const float *LIBXSMM_RESTRICT a, const libxsmm_blasint* lda,
  const float *LIBXSMM_RESTRICT b, const libxsmm_blasint* ldb,
  const float* beta, float *LIBXSMM_RESTRICT c, const libxsmm_blasint* ldc)
{
  libxsmm_blasint ilda, ildb;
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
  ilda = *(lda ? lda : m);
  ildb = *(ldb ? ldb : n);
  LIBXSMM_BSGEMM(flags, *LIBXSMM_LD(m, n), *LIBXSMM_LD(n, m), *k,
    0 != alpha ? (1.f == *alpha ? 1.f : (-1.f == *alpha ? -1.f : *alpha)) : ((float)LIBXSMM_ALPHA),
    LIBXSMM_LD(a, b), LIBXSMM_LD(ilda, ildb), LIBXSMM_LD(b, a), LIBXSMM_LD(ildb, ilda),
    0 != beta  ? (1.f == *beta  ? 1.f : ( 0.f == *beta  ?  0.f : *beta))  : ((float)LIBXSMM_BETA),
    c, ldc ? *ldc : *LIBXSMM_LD(m, n));
}


LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE void libxsmm_blas_dgemm(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const double* alpha, const double *LIBXSMM_RESTRICT a, const libxsmm_blasint* lda,
  const double *LIBXSMM_RESTRICT b, const libxsmm_blasint* ldb,
  const double* beta, double *LIBXSMM_RESTRICT c, const libxsmm_blasint* ldc)
{
  libxsmm_blasint ilda, ildb;
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
  ilda = *(lda ? lda : m);
  ildb = *(ldb ? ldb : n);
  LIBXSMM_BDGEMM(flags, *LIBXSMM_LD(m, n), *LIBXSMM_LD(n, m), *k,
    0 != alpha ? (1.0 == *alpha ? 1.0 : (-1.0 == *alpha ? -1.0 : *alpha)) : ((double)LIBXSMM_ALPHA),
    LIBXSMM_LD(a, b), LIBXSMM_LD(ilda, ildb), LIBXSMM_LD(b, a), LIBXSMM_LD(ildb, ilda),
    0 != beta  ? (1.0 == *beta  ? 1.0 : ( 0.0 == *beta  ?  0.0 : *beta))  : ((double)LIBXSMM_BETA),
    c, ldc ? *ldc : *LIBXSMM_LD(m, n));
}
