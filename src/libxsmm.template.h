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
#ifndef LIBXSMM_H
#define LIBXSMM_H

/** Parameters the library was built for. */
#define LIBXSMM_ALIGNMENT $ALIGNMENT
#define LIBXSMM_ALIGNED_STORES $ALIGNED_STORES
#define LIBXSMM_ALIGNED_LOADS $ALIGNED_LOADS
#define LIBXSMM_ALIGNED_MAX $ALIGNED_MAX
#define LIBXSMM_PREFETCH $PREFETCH
#define LIBXSMM_ROW_MAJOR $ROW_MAJOR
#define LIBXSMM_COL_MAJOR $COL_MAJOR
#define LIBXSMM_MAX_MNK $MAX_MNK
#define LIBXSMM_MAX_M $MAX_M
#define LIBXSMM_MAX_N $MAX_N
#define LIBXSMM_MAX_K $MAX_K
#define LIBXSMM_AVG_M $AVG_M
#define LIBXSMM_AVG_N $AVG_N
#define LIBXSMM_AVG_K $AVG_K
#define LIBXSMM_ALPHA $ALPHA
#define LIBXSMM_BETA $BETA
#define LIBXSMM_JIT $JIT

#include "libxsmm_typedefs.h"
#include "libxsmm_prefetch.h"
#include "libxsmm_fallback.h"


/** Initialize the library; pay for setup cost at a specific point. */
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE void libxsmm_init(void);

/** Query or JIT-generate a function; return zero if it does not exist or if JIT is not supported (single-precision). */
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE libxsmm_sfunction libxsmm_sdispatch(float alpha, float beta, int m, int n, int k,
  int lda, int ldb, int ldc, int flags, int prefetch);
/** Query or JIT-generate a function; return zero if it does not exist or if JIT is not supported (double-precision). */
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE libxsmm_dfunction libxsmm_ddispatch(double alpha, double beta, int m, int n, int k,
  int lda, int ldb, int ldc, int flags, int prefetch);

/** Dispatched matrix-matrix multiplication (single-precision). */
LIBXSMM_INLINE LIBXSMM_RETARGETABLE void libxsmm_smm(float alpha, float beta, int m, int n, int k,
  const float *LIBXSMM_RESTRICT a, const float *LIBXSMM_RESTRICT b, float *LIBXSMM_RESTRICT c
  LIBXSMM_PREFETCH_DECL(const float*, pa)
  LIBXSMM_PREFETCH_DECL(const float*, pb)
  LIBXSMM_PREFETCH_DECL(const float*, pc))
{
  LIBXSMM_USE(pa); LIBXSMM_USE(pb); LIBXSMM_USE(pc);
  LIBXSMM_MM(float, alpha, beta, m, n, k, a, b, c, LIBXSMM_PREFETCH_ARG_pa, LIBXSMM_PREFETCH_ARG_pb, LIBXSMM_PREFETCH_ARG_pc);
}

/** Dispatched matrix-matrix multiplication (double-precision). */
LIBXSMM_INLINE LIBXSMM_RETARGETABLE void libxsmm_dmm(double alpha, double beta, int m, int n, int k,
  const double *LIBXSMM_RESTRICT a, const double *LIBXSMM_RESTRICT b, double *LIBXSMM_RESTRICT c
  LIBXSMM_PREFETCH_DECL(const double*, pa)
  LIBXSMM_PREFETCH_DECL(const double*, pb)
  LIBXSMM_PREFETCH_DECL(const double*, pc))
{
  LIBXSMM_USE(pa); LIBXSMM_USE(pb); LIBXSMM_USE(pc);
  LIBXSMM_MM(double, alpha, beta, m, n, k, a, b, c, LIBXSMM_PREFETCH_ARG_pa, LIBXSMM_PREFETCH_ARG_pb, LIBXSMM_PREFETCH_ARG_pc);
}

/** Non-dispatched matrix-matrix multiplication using inline code (single-precision). */
LIBXSMM_INLINE LIBXSMM_RETARGETABLE void libxsmm_simm(float alpha, float beta, int m, int n, int k,
  const float *LIBXSMM_RESTRICT a, const float *LIBXSMM_RESTRICT b, float *LIBXSMM_RESTRICT c)
{
  LIBXSMM_IMM(float, int, alpha, beta, m, n, k, a, b, c);
}

/** Non-dispatched matrix-matrix multiplication using inline code (double-precision). */
LIBXSMM_INLINE LIBXSMM_RETARGETABLE void libxsmm_dimm(double alpha, double beta, int m, int n, int k,
  const double *LIBXSMM_RESTRICT a, const double *LIBXSMM_RESTRICT b, double *LIBXSMM_RESTRICT c)
{
  LIBXSMM_IMM(double, int, alpha, beta, m, n, k, a, b, c);
}

/** Non-dispatched matrix-matrix multiplication using BLAS (single-precision). */
LIBXSMM_INLINE LIBXSMM_RETARGETABLE void libxsmm_sblasmm(float alpha, float beta, int m, int n, int k,
  const float *LIBXSMM_RESTRICT a, const float *LIBXSMM_RESTRICT b, float *LIBXSMM_RESTRICT c)
{
  LIBXSMM_BLASMM(float, alpha, beta, m, n, k, a, b, c);
}

/** Non-dispatched matrix-matrix multiplication using BLAS (double-precision). */
LIBXSMM_INLINE LIBXSMM_RETARGETABLE void libxsmm_dblasmm(double alpha, double beta, int m, int n, int k,
  const double *LIBXSMM_RESTRICT a, const double *LIBXSMM_RESTRICT b, double *LIBXSMM_RESTRICT c)
{
  LIBXSMM_BLASMM(double, alpha, beta, m, n, k, a, b, c);
}
$MNK_INTERFACE_LIST
#if defined(__cplusplus)

/** Function type depending on T. */
template<typename T> struct LIBXSMM_RETARGETABLE libxsmm_function { typedef void type; };
template<> struct LIBXSMM_RETARGETABLE libxsmm_function<float>    { typedef libxsmm_sfunction type; };
template<> struct LIBXSMM_RETARGETABLE libxsmm_function<double>   { typedef libxsmm_dfunction type; };

/** Query or JIT-generate a function; return zero if it does not exist or if JIT is not supported. */
LIBXSMM_RETARGETABLE inline libxsmm_sfunction libxsmm_dispatch(float alpha, float beta, int m, int n, int k,
  int lda = 0, int ldb = 0, int ldc = 0, int flags = LIBXSMM_GEMM_FLAG_DEFAULT, int prefetch = LIBXSMM_PREFETCH)
{
  return libxsmm_sdispatch(alpha, beta, m, n, k, lda, ldb, ldc, flags, prefetch);
}

/** Query or JIT-generate a function; return zero if it does not exist or if JIT is not supported. */
LIBXSMM_RETARGETABLE inline libxsmm_dfunction libxsmm_dispatch(double alpha, double beta, int m, int n, int k,
  int lda = 0, int ldb = 0, int ldc = 0, int flags = LIBXSMM_GEMM_FLAG_DEFAULT, int prefetch = LIBXSMM_PREFETCH)
{
  return libxsmm_ddispatch(alpha, beta, m, n, k, lda, ldb, ldc, flags, prefetch);
}

/** Dispatched matrix-matrix multiplication. */
LIBXSMM_RETARGETABLE inline void libxsmm_mm(float alpha, float beta, int m, int n, int k,
  const float *LIBXSMM_RESTRICT a, const float *LIBXSMM_RESTRICT b, float *LIBXSMM_RESTRICT c
  LIBXSMM_PREFETCH_DECL(const float*, pa)
  LIBXSMM_PREFETCH_DECL(const float*, pb)
  LIBXSMM_PREFETCH_DECL(const float*, pc))
{
  LIBXSMM_USE(pa); LIBXSMM_USE(pb); LIBXSMM_USE(pc);
  libxsmm_smm(alpha, beta, m, n, k, a, b, c LIBXSMM_PREFETCH_ARGA(pa) LIBXSMM_PREFETCH_ARGB(pb) LIBXSMM_PREFETCH_ARGC(pc));
}

/** Dispatched matrix-matrix multiplication. */
LIBXSMM_RETARGETABLE inline void libxsmm_mm(double alpha, double beta, int m, int n, int k,
  const double *LIBXSMM_RESTRICT a, const double *LIBXSMM_RESTRICT b, double *LIBXSMM_RESTRICT c
  LIBXSMM_PREFETCH_DECL(const double*, pa)
  LIBXSMM_PREFETCH_DECL(const double*, pb)
  LIBXSMM_PREFETCH_DECL(const double*, pc))
{
  LIBXSMM_USE(pa); LIBXSMM_USE(pb); LIBXSMM_USE(pc);
  libxsmm_dmm(alpha, beta, m, n, k, a, b, c LIBXSMM_PREFETCH_ARGA(pa) LIBXSMM_PREFETCH_ARGB(pb) LIBXSMM_PREFETCH_ARGC(pc));
}

/** Non-dispatched matrix-matrix multiplication using inline code. */
LIBXSMM_RETARGETABLE inline void libxsmm_imm(float alpha, float beta, int m, int n, int k,
  const float *LIBXSMM_RESTRICT a, const float *LIBXSMM_RESTRICT b, float *LIBXSMM_RESTRICT c)
{
  libxsmm_simm(alpha, beta, m, n, k, a, b, c);
}

/** Non-dispatched matrix-matrix multiplication using inline code. */
LIBXSMM_RETARGETABLE inline void libxsmm_imm(double alpha, double beta, int m, int n, int k,
  const double *LIBXSMM_RESTRICT a, const double *LIBXSMM_RESTRICT b, double *LIBXSMM_RESTRICT c)
{
  libxsmm_dimm(alpha, beta, m, n, k, a, b, c);
}

/** Non-dispatched matrix-matrix multiplication using BLAS. */
LIBXSMM_RETARGETABLE inline void libxsmm_blasmm(float alpha, float beta, int m, int n, int k,
  const float *LIBXSMM_RESTRICT a, const float *LIBXSMM_RESTRICT b, float *LIBXSMM_RESTRICT c)
{
  libxsmm_sblasmm(alpha, beta, m, n, k, a, b, c);
}

/** Non-dispatched matrix-matrix multiplication using BLAS. */
LIBXSMM_RETARGETABLE inline void libxsmm_blasmm(double alpha, double beta, int m, int n, int k,
  const double *LIBXSMM_RESTRICT a, const double *LIBXSMM_RESTRICT b, double *LIBXSMM_RESTRICT c)
{
  libxsmm_dblasmm(alpha, beta, m, n, k, a, b, c);
}

#endif /*__cplusplus*/

#endif /*LIBXSMM_H*/
