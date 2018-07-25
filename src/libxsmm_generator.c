/******************************************************************************
** Copyright (c) 2018, Intel Corporation                                     **
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
#include "libxsmm_gemm_diff.h"

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif


LIBXSMM_API libxsmm_gemm_descriptor* libxsmm_dgemm_descriptor_init(libxsmm_descriptor_blob* blob,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k, libxsmm_blasint lda, libxsmm_blasint ldb, libxsmm_blasint ldc,
  double alpha, double beta, int flags, libxsmm_gemm_prefetch_type prefetch)
{
  union {
    libxsmm_gemm_descriptor* ptr;
    libxsmm_descriptor_blob* blob;
  } result;
  if  (LIBXSMM_GEMM_NO_BYPASS(flags, alpha, beta)
    && LIBXSMM_GEMM_NO_BYPASS_DIMS(lda, ldb, ldc)
    && LIBXSMM_GEMM_NO_BYPASS_DIMS(m, n, k)
    && 0 != blob)
  {
    result.blob = blob;
    LIBXSMM_GEMM_DESCRIPTOR(*result.ptr, LIBXSMM_GEMM_PRECISION(double),
      flags, m, n, k, lda, ldb, ldc, alpha, beta, prefetch);
  }
  else { /* quiet error (unsupported) */
    result.ptr = 0;
  }
  return result.ptr;
}


LIBXSMM_API libxsmm_gemm_descriptor* libxsmm_sgemm_descriptor_init(libxsmm_descriptor_blob* blob,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k, libxsmm_blasint lda, libxsmm_blasint ldb, libxsmm_blasint ldc,
  float alpha, float beta, int flags, libxsmm_gemm_prefetch_type prefetch)
{
  union {
    libxsmm_gemm_descriptor* ptr;
    libxsmm_descriptor_blob* blob;
  } result;
  if  (LIBXSMM_GEMM_NO_BYPASS(flags, alpha, beta)
    && LIBXSMM_GEMM_NO_BYPASS_DIMS(lda, ldb, ldc)
    && LIBXSMM_GEMM_NO_BYPASS_DIMS(m, n, k)
    && 0 != blob)
  {
    result.blob = blob;
    LIBXSMM_GEMM_DESCRIPTOR(*result.ptr, LIBXSMM_GEMM_PRECISION(float),
      flags, m, n, k, lda, ldb, ldc, alpha, beta, prefetch);
  }
  else { /* unsupported */
    result.ptr = 0;
  }
  return result.ptr;
}


LIBXSMM_API libxsmm_gemm_descriptor* libxsmm_wigemm_descriptor_init(libxsmm_descriptor_blob* blob,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k, libxsmm_blasint lda, libxsmm_blasint ldb, libxsmm_blasint ldc,
  int alpha, int beta, int flags, libxsmm_gemm_prefetch_type prefetch)
{
  union {
    libxsmm_gemm_descriptor* ptr;
    libxsmm_descriptor_blob* blob;
  } result;
  if  (LIBXSMM_GEMM_NO_BYPASS(flags, alpha, beta)
    && LIBXSMM_GEMM_NO_BYPASS_DIMS(lda, ldb, ldc)
    && LIBXSMM_GEMM_NO_BYPASS_DIMS(m, n, k)
    && 0 != blob)
  {
    result.blob = blob;
    LIBXSMM_GEMM_DESCRIPTOR2(*result.ptr, LIBXSMM_GEMM_PRECISION(short), LIBXSMM_GEMM_PRECISION(int),
      flags, m, n, k, lda, ldb, ldc, alpha, beta, prefetch);
  }
  else { /* unsupported */
    result.ptr = 0;
  }
  return result.ptr;
}


LIBXSMM_API libxsmm_gemm_descriptor* libxsmm_wsgemm_descriptor_init(libxsmm_descriptor_blob* blob,
  libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k, libxsmm_blasint lda, libxsmm_blasint ldb, libxsmm_blasint ldc,
  float alpha, float beta, int flags, libxsmm_gemm_prefetch_type prefetch)
{
  union {
    libxsmm_gemm_descriptor* ptr;
    libxsmm_descriptor_blob* blob;
  } result;
  if (LIBXSMM_GEMM_NO_BYPASS(flags, alpha, beta)
    && LIBXSMM_GEMM_NO_BYPASS_DIMS(lda, ldb, ldc)
    && LIBXSMM_GEMM_NO_BYPASS_DIMS(m, n, k)
    && 0 != blob)
  {
    result.blob = blob;
    LIBXSMM_GEMM_DESCRIPTOR2(*result.ptr, LIBXSMM_GEMM_PRECISION(short), LIBXSMM_GEMM_PRECISION(float),
      flags, m, n, k, lda, ldb, ldc, alpha, beta, prefetch);
  }
  else { /* unsupported */
    result.ptr = 0;
  }
  return result.ptr;
}


LIBXSMM_API libxsmm_gemm_descriptor* libxsmm_gemm_descriptor_dinit(libxsmm_descriptor_blob* blob,
  libxsmm_gemm_precision precision, libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  libxsmm_blasint lda, libxsmm_blasint ldb, libxsmm_blasint ldc, double alpha, double beta,
  int flags, libxsmm_gemm_prefetch_type prefetch)
{
  return libxsmm_gemm_descriptor_dinit2(blob, precision, precision, m, n, k, lda, ldb, ldc, alpha, beta, flags, prefetch);
}


LIBXSMM_API libxsmm_gemm_descriptor* libxsmm_gemm_descriptor_dinit2(libxsmm_descriptor_blob* blob,
  libxsmm_gemm_precision iprec, libxsmm_gemm_precision oprec, libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  libxsmm_blasint lda, libxsmm_blasint ldb, libxsmm_blasint ldc, double alpha, double beta,
  int flags, libxsmm_gemm_prefetch_type prefetch)
{
  libxsmm_gemm_descriptor* result;
  switch (iprec) {
    case LIBXSMM_GEMM_PRECISION_F64: {
      LIBXSMM_ASSERT(iprec == oprec);
      result = libxsmm_dgemm_descriptor_init(blob, m, n, k, lda, ldb, ldc,
        alpha, beta, flags, prefetch);
    } break;
    case LIBXSMM_GEMM_PRECISION_F32: {
      LIBXSMM_ASSERT(iprec == oprec);
      result = libxsmm_sgemm_descriptor_init(blob, m, n, k, lda, ldb, ldc,
        (float)alpha, (float)beta, flags, prefetch);
    } break;
    case LIBXSMM_GEMM_PRECISION_I16: {
      if (LIBXSMM_GEMM_PRECISION_I32 == oprec) {
        result = libxsmm_wigemm_descriptor_init(blob, m, n, k, lda, ldb, ldc,
          (int)alpha, (int)beta, flags, prefetch);
      }
      else {
        LIBXSMM_ASSERT(LIBXSMM_GEMM_PRECISION_F32 == oprec);
        result = libxsmm_wsgemm_descriptor_init(blob, m, n, k, lda, ldb, ldc,
          (float)alpha, (float)beta, flags, prefetch);
      }
    } break;
    default: {
      static int error_once = 0;
      if (0 != libxsmm_verbosity /* library code is expected to be mute */
        && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
      {
        fprintf(stderr, "LIBXSMM ERROR: GEMM precision is not supported!\n");
      }
      result = 0;
    }
  }
  return result;
}


LIBXSMM_API libxsmm_gemm_descriptor* libxsmm_gemm_descriptor_init(libxsmm_descriptor_blob* blob,
  libxsmm_gemm_precision precision, libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  libxsmm_blasint lda, libxsmm_blasint ldb, libxsmm_blasint ldc, const void* alpha, const void* beta,
  int flags, libxsmm_gemm_prefetch_type prefetch)
{
  return libxsmm_gemm_descriptor_init2(blob, precision, precision, m, n, k, lda, ldb, ldc, alpha, beta, flags, prefetch);
}


LIBXSMM_API libxsmm_gemm_descriptor* libxsmm_gemm_descriptor_init2(libxsmm_descriptor_blob* blob,
  libxsmm_gemm_precision iprec, libxsmm_gemm_precision oprec, libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  libxsmm_blasint lda, libxsmm_blasint ldb, libxsmm_blasint ldc, const void* alpha, const void* beta,
  int flags, libxsmm_gemm_prefetch_type prefetch)
{
  return libxsmm_gemm_descriptor_init3(blob, iprec, oprec, m, n, k, lda, ldb, ldc, alpha, beta, flags, prefetch,
    NULL/*dalpha*/, NULL/*dbeta*/);
}


LIBXSMM_API libxsmm_gemm_descriptor* libxsmm_gemm_descriptor_init3(libxsmm_descriptor_blob* blob,
  libxsmm_gemm_precision iprec, libxsmm_gemm_precision oprec, libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  libxsmm_blasint lda, libxsmm_blasint ldb, libxsmm_blasint ldc, const void* alpha, const void* beta,
  int flags, libxsmm_gemm_prefetch_type prefetch,
  double* dalpha, double* dbeta)
{
  libxsmm_gemm_descriptor* result;
  switch (iprec) {
    case LIBXSMM_GEMM_PRECISION_F64: {
      const double aa = (NULL != alpha ? *((const double*)alpha) : (LIBXSMM_ALPHA));
      const double bb = (NULL != beta  ? *((const double*)beta)  : (LIBXSMM_BETA));
      LIBXSMM_ASSERT(LIBXSMM_GEMM_PRECISION_F64 == oprec);
      result = libxsmm_dgemm_descriptor_init(blob, m, n, k, lda, ldb, ldc, aa, bb, flags, prefetch);
      if (NULL != dalpha) *dalpha = aa;
      if (NULL != dbeta) *dbeta = bb;
    } break;
    case LIBXSMM_GEMM_PRECISION_F32: {
      const float aa = (0 != alpha ? *((const float*)alpha) : (LIBXSMM_ALPHA));
      const float bb = (0 != beta  ? *((const float*)beta)  : (LIBXSMM_BETA));
      LIBXSMM_ASSERT(LIBXSMM_GEMM_PRECISION_F32 == oprec);
      result = libxsmm_sgemm_descriptor_init(blob, m, n, k, lda, ldb, ldc, aa, bb, flags, prefetch);
      if (NULL != dalpha) *dalpha = (double)aa;
      if (NULL != dbeta) *dbeta = (double)bb;
    } break;
    case LIBXSMM_GEMM_PRECISION_I16: {
      if (LIBXSMM_GEMM_PRECISION_I32 == oprec) {
        /**
         * Take alpha and beta as short data although wgemm works on integers.
         * However, alpha and beta are only JIT-supported for certain values,
         * and the call-side may not distinct different input and output types
         * (integer/short), hence it is safer to only read short data.
         */
        const short aa = (short)(0 != alpha ? *((const short*)alpha) : (LIBXSMM_ALPHA));
        const short bb = (short)(0 != beta  ? *((const short*)beta)  : (LIBXSMM_BETA));
        result = libxsmm_wigemm_descriptor_init(blob, m, n, k, lda, ldb, ldc, aa, bb, flags, prefetch);
        if (NULL != dalpha) *dalpha = (double)aa;
        if (NULL != dbeta) *dbeta = (double)bb;
      }
      else {
        const float aa = (0 != alpha ? *((const float*)alpha) : (LIBXSMM_ALPHA));
        const float bb = (0 != beta  ? *((const float*)beta)  : (LIBXSMM_BETA));
        LIBXSMM_ASSERT(LIBXSMM_GEMM_PRECISION_F32 == oprec);
        result = libxsmm_wsgemm_descriptor_init(blob, m, n, k, lda, ldb, ldc, aa, bb, flags, prefetch);
        if (NULL != dalpha) *dalpha = (double)aa;
        if (NULL != dbeta) *dbeta = (double)bb;
      }
    } break;
    default: {
      static int error_once = 0;
      if (0 != libxsmm_verbosity /* library code is expected to be mute */
        && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
      {
        fprintf(stderr, "LIBXSMM ERROR: GEMM precision is not supported!\n");
      }
      result = 0;
    }
  }
  return result;
}


LIBXSMM_API libxsmm_trans_descriptor* libxsmm_trans_descriptor_init(libxsmm_descriptor_blob* blob,
  unsigned int typesize, unsigned int m, unsigned int n, unsigned int ldo)
{
  union {
    libxsmm_trans_descriptor* ptr;
    libxsmm_descriptor_blob* blob;
  } result;
  /* limit the amount of (unrolled) code by rejecting larger kernels */
  if (LIBXSMM_TRANS_NO_BYPASS(m, n)) {
    result.blob = blob;
    result.ptr->typesize = (unsigned char)typesize;
    result.ptr->ldo = ldo;
    result.ptr->m = m;
    result.ptr->n = n;
  }
  else {
    result.ptr = 0;
  }
  return result.ptr;
}


LIBXSMM_API libxsmm_mcopy_descriptor* libxsmm_mcopy_descriptor_init(libxsmm_descriptor_blob* blob,
  unsigned int typesize, unsigned int m, unsigned int n, unsigned int ldo,
  unsigned int ldi, int flags, int prefetch, const int* unroll)
{
  union {
    libxsmm_mcopy_descriptor* ptr;
    libxsmm_descriptor_blob* blob;
  } result;
  if (0 == LIBXSMM_MOD2(typesize, 4)) { /* TODO: more general kernel */
    const unsigned int typescale = LIBXSMM_DIV2(typesize, 4);
    result.blob = blob;
    result.ptr->unroll_level = (unsigned char)((0 == unroll || 0 >= *unroll) ? 2/*default*/ : LIBXSMM_MIN(*unroll, 64));
    result.ptr->typesize = (unsigned char)/*typesize*/4;
    result.ptr->prefetch = (unsigned char)prefetch;
    result.ptr->flags = (unsigned char)flags;
    result.ptr->ldi = ldi * typescale;
    result.ptr->ldo = ldo * typescale;
    result.ptr->m = m * typescale;
    result.ptr->n = n;
  }
  else {
    result.ptr = 0;
  }
  return result.ptr;
}


LIBXSMM_API libxsmm_trsm_descriptor* libxsmm_trsm_descriptor_init(libxsmm_descriptor_blob* blob,
  unsigned int typesize, libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint lda, libxsmm_blasint ldb,
  const void* alpha, char transa, char diag, char side, char uplo, int layout)
{
  union {
    libxsmm_trsm_descriptor* ptr;
    libxsmm_descriptor_blob* blob;
  } result;
  result.blob = blob;
  result.ptr->typesize = (unsigned char)typesize;
  result.ptr->lda = (unsigned char)lda;
  result.ptr->ldb = (unsigned char)ldb;
  result.ptr->m = (unsigned char)m;
  result.ptr->n = (unsigned char)n;
  result.ptr->transa = transa;
  result.ptr->diag = diag;
  result.ptr->side = side;
  result.ptr->uplo = uplo;
  result.ptr->layout = (unsigned char)layout;
  switch (typesize) {
  case 4: {
    result.ptr->alpha.s = (0 != alpha ? (*(const float*)alpha) : ((float)LIBXSMM_ALPHA));
  } break;
  case 8: {
    result.ptr->alpha.d = (0 != alpha ? (*(const double*)alpha) : ((double)LIBXSMM_ALPHA));
  } break;
  default: /* TODO: generate warning */
    ;
  }
  return result.ptr;
}

