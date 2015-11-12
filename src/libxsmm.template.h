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
#include "libxsmm_frontend.h"


/** Specialized function (single-precision). */
typedef LIBXSMM_RETARGETABLE void (*libxsmm_sfunction0)(const float *LIBXSMM_RESTRICT a, const float *LIBXSMM_RESTRICT b, float *LIBXSMM_RESTRICT c);
/** Specialized function (double-precision). */
typedef LIBXSMM_RETARGETABLE void (*libxsmm_dfunction0)(const double *LIBXSMM_RESTRICT a, const double *LIBXSMM_RESTRICT b, double *LIBXSMM_RESTRICT c);

/** Specialized function with prefetches (single-precision). */
typedef LIBXSMM_RETARGETABLE void (*libxsmm_sfunction1)(const float *LIBXSMM_RESTRICT a, const float *LIBXSMM_RESTRICT b, float *LIBXSMM_RESTRICT c,
  const float* pa, const float* pb, const float* pc);
/** Specialized function with prefetches (double-precision). */
typedef LIBXSMM_RETARGETABLE void (*libxsmm_dfunction1)(const double *LIBXSMM_RESTRICT a, const double *LIBXSMM_RESTRICT b, double *LIBXSMM_RESTRICT c,
  const double* pa, const double* pb, const double* pc);

/** Specialized function with prefetches, alpha, and beta arguments (single-precision). */
typedef LIBXSMM_RETARGETABLE void (*libxsmm_sfunction2)(const float *LIBXSMM_RESTRICT a, const float *LIBXSMM_RESTRICT b, float *LIBXSMM_RESTRICT c,
  const float* pa, const float* pb, const float* pc, float alpha, float beta);
/** Specialized function with prefetches, alpha, and beta arguments (double-precision). */
typedef LIBXSMM_RETARGETABLE void (*libxsmm_dfunction2)(const double *LIBXSMM_RESTRICT a, const double *LIBXSMM_RESTRICT b, double *LIBXSMM_RESTRICT c,
  const double* pa, const double* pb, const double* pc, double alpha, double beta);

/** Initialize the library; pay for setup cost at a specific point. */
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE void libxsmm_init(void);
/** Uninitialize the library and free internal memory (optional). */
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE void libxsmm_finalize(void);

/** Query or JIT-generate a function; return zero if it does not exist or if JIT is not supported (single-precision). */
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE libxsmm_sfunction0 libxsmm_sdispatch(int m, int n, int k, int lda, int ldb, int ldc, int flags);
/** Query or JIT-generate a function; return zero if it does not exist or if JIT is not supported (double-precision). */
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE libxsmm_dfunction0 libxsmm_ddispatch(int m, int n, int k, int lda, int ldb, int ldc, int flags);

/** Query or JIT-generate a function; return zero if it does not exist or if JIT is not supported (single-precision).
 *  Signature with prefetches. */
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE libxsmm_sfunction1 libxsmm_sdispatch1(int m, int n, int k, int lda, int ldb, int ldc, int flags,
  int prefetch);
/** Query or JIT-generate a function; return zero if it does not exist or if JIT is not supported (double-precision).
 *  Signature with prefetches. */
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE libxsmm_dfunction1 libxsmm_ddispatch1(int m, int n, int k, int lda, int ldb, int ldc, int flags,
  int prefetch);

/** Query or JIT-generate a function; return zero if it does not exist or if JIT is not supported (single-precision).
 *  Signature with prefetches, alpha, and beta arguments. */
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE libxsmm_sfunction2 libxsmm_sdispatch2(int m, int n, int k, int lda, int ldb, int ldc, int flags,
  int prefetch, float alpha, float beta);
/** Query or JIT-generate a function; return zero if it does not exist or if JIT is not supported (double-precision).
 *  Signature with prefetches, alpha, and beta arguments. */
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE libxsmm_dfunction2 libxsmm_ddispatch2(int m, int n, int k, int lda, int ldb, int ldc, int flags,
  int prefetch, double alpha, double beta);

/** Dispatched matrix multiplication (single-precision). */
LIBXSMM_INLINE LIBXSMM_RETARGETABLE void libxsmm_smm(int flags, int m, int n, int k,
  const float *LIBXSMM_RESTRICT a, const float *LIBXSMM_RESTRICT b, float *LIBXSMM_RESTRICT c,
  const float* pa, const float* pb, const float* pc, const float* alpha, const float* beta)
{
  LIBXSMM_MM(float, flags, m, n, k, a, b, c, pa, pb, pc, alpha, beta);
}

/** Dispatched matrix multiplication (double-precision). */
LIBXSMM_INLINE LIBXSMM_RETARGETABLE void libxsmm_dmm(int flags, int m, int n, int k,
  const double *LIBXSMM_RESTRICT a, const double *LIBXSMM_RESTRICT b, double *LIBXSMM_RESTRICT c,
  const double* pa, const double* pb, const double* pc, const double* alpha, const double* beta)
{
  LIBXSMM_MM(double, flags, m, n, k, a, b, c, pa, pb, pc, alpha, beta);
}

/** Non-dispatched matrix multiplication using BLAS (single-precision). */
LIBXSMM_INLINE LIBXSMM_RETARGETABLE void libxsmm_sblasmm(int flags, int m, int n, int k,
  const float *LIBXSMM_RESTRICT a, const float *LIBXSMM_RESTRICT b, float *LIBXSMM_RESTRICT c,
  const float* alpha, const float* beta)
{
  LIBXSMM_BLASMM(float, flags, m, n, k, a, b, c, alpha, beta);
}

/** Non-dispatched matrix multiplication using BLAS (double-precision). */
LIBXSMM_INLINE LIBXSMM_RETARGETABLE void libxsmm_dblasmm(int flags, int m, int n, int k,
  const double *LIBXSMM_RESTRICT a, const double *LIBXSMM_RESTRICT b, double *LIBXSMM_RESTRICT c,
  const double* alpha, const double* beta)
{
  LIBXSMM_BLASMM(double, flags, m, n, k, a, b, c, alpha, beta);
}
$MNK_INTERFACE_LIST
#if defined(__cplusplus)

/** Construct and execute a specialized function. */
template<typename T> class LIBXSMM_RETARGETABLE libxsmm_function {};

/** Construct and execute a specialized function (single-precision). */
template<> class LIBXSMM_RETARGETABLE libxsmm_function<float> {
  typedef LIBXSMM_RETARGETABLE void (*type)(const float *LIBXSMM_RESTRICT a, const float *LIBXSMM_RESTRICT b, float *LIBXSMM_RESTRICT c, ...);
  mutable/*retargetable*/ type m_function;
public:
  libxsmm_function(): m_function(0) {}
  libxsmm_function(int m, int n, int k, int flags = 0)
    : m_function(reinterpret_cast<type>(libxsmm_sdispatch(m, n, k, 0, 0, 0, flags)))
  {}
  libxsmm_function(int m, int n, int k, int lda, int ldb, int ldc, int flags = 0)
    : m_function(reinterpret_cast<type>(libxsmm_sdispatch(m, n, k, lda, ldb, ldc, flags)))
  {}
  libxsmm_function(int m, int n, int k, int flags, int prefetch)
    : m_function(reinterpret_cast<type>(libxsmm_sdispatch1(m, n, k, 0, 0, 0, flags, prefetch)))
  {}
  libxsmm_function(int m, int n, int k, int lda, int ldb, int ldc, int flags, int prefetch)
    : m_function(reinterpret_cast<type>(libxsmm_sdispatch1(m, n, k, lda, ldb, ldc, flags, prefetch)))
  {}
  libxsmm_function(int m, int n, int k, float alpha, float beta, int flags = 0, int prefetch = LIBXSMM_PREFETCH)
    : m_function(reinterpret_cast<type>(libxsmm_sdispatch2(m, n, k, 0, 0, 0, flags, prefetch, alpha, beta)))
  {}
  libxsmm_function(int m, int n, int k, int lda, int ldb, int ldc, float alpha, float beta, int flags = 0, int prefetch = LIBXSMM_PREFETCH)
    : m_function(reinterpret_cast<type>(libxsmm_sdispatch2(m, n, k, lda, ldb, ldc, flags, prefetch, alpha, beta)))
  {}
public:
  operator type() const {
    return m_function;
  }
  void operator()(const float *LIBXSMM_RESTRICT a, const float *LIBXSMM_RESTRICT b, float *LIBXSMM_RESTRICT c) const {
    m_function(a, b, c);
  }
  void operator()(const float *LIBXSMM_RESTRICT a, const float *LIBXSMM_RESTRICT b, float *LIBXSMM_RESTRICT c,
    const float* pa, const float* pb, const float* pc) const
  {
    m_function(a, b, c, pa, pb, pc);
  }
  void operator()(const float *LIBXSMM_RESTRICT a, const float *LIBXSMM_RESTRICT b, float *LIBXSMM_RESTRICT c,
    const float* pa, const float* pb, const float* pc,
    float alpha, float beta) const
  {
    m_function(a, b, c, pa, pb, pc, alpha, beta);
  }
};

/** Construct and execute a specialized function (double-precision). */
template<> class LIBXSMM_RETARGETABLE libxsmm_function<double> {
  typedef LIBXSMM_RETARGETABLE void (*type)(const double *LIBXSMM_RESTRICT a, const double *LIBXSMM_RESTRICT b, double *LIBXSMM_RESTRICT c, ...);
  mutable/*retargetable*/ type m_function;
public:
  libxsmm_function(): m_function(0) {}
  libxsmm_function(int m, int n, int k, int flags = 0)
    : m_function(reinterpret_cast<type>(libxsmm_ddispatch(m, n, k, 0, 0, 0, flags)))
  {}
  libxsmm_function(int m, int n, int k, int lda, int ldb, int ldc, int flags = 0)
    : m_function(reinterpret_cast<type>(libxsmm_ddispatch(m, n, k, lda, ldb, ldc, flags)))
  {}
  libxsmm_function(int m, int n, int k, int flags, int prefetch)
    : m_function(reinterpret_cast<type>(libxsmm_ddispatch1(m, n, k, 0, 0, 0, flags, prefetch)))
  {}
  libxsmm_function(int m, int n, int k, int lda, int ldb, int ldc, int flags, int prefetch)
    : m_function(reinterpret_cast<type>(libxsmm_ddispatch1(m, n, k, lda, ldb, ldc, flags, prefetch)))
  {}
  libxsmm_function(int m, int n, int k, double alpha, double beta, int flags = 0, int prefetch = LIBXSMM_PREFETCH)
    : m_function(reinterpret_cast<type>(libxsmm_ddispatch2(m, n, k, 0, 0, 0, flags, prefetch, alpha, beta)))
  {}
  libxsmm_function(int m, int n, int k, int lda, int ldb, int ldc, double alpha, double beta, int flags = 0, int prefetch = LIBXSMM_PREFETCH)
    : m_function(reinterpret_cast<type>(libxsmm_ddispatch2(m, n, k, lda, ldb, ldc, flags, prefetch, alpha, beta)))
  {}
public:
  operator type() const {
    return m_function;
  }
  void operator()(const double *LIBXSMM_RESTRICT a, const double *LIBXSMM_RESTRICT b, double *LIBXSMM_RESTRICT c) const {
    m_function(a, b, c);
  }
  void operator()(const double *LIBXSMM_RESTRICT a, const double *LIBXSMM_RESTRICT b, double *LIBXSMM_RESTRICT c,
    const double* pa, const double* pb, const double* pc) const
  {
    m_function(a, b, c, pa, pb, pc);
  }
  void operator()(const double *LIBXSMM_RESTRICT a, const double *LIBXSMM_RESTRICT b, double *LIBXSMM_RESTRICT c,
    const double* pa, const double* pb, const double* pc,
    double alpha, double beta) const
  {
    m_function(a, b, c, pa, pb, pc, alpha, beta);
  }
};

/** Dispatched matrix multiplication (single-precision). */
LIBXSMM_RETARGETABLE inline void libxsmm_mm(int m, int n, int k,
  const float *LIBXSMM_RESTRICT a, const float *LIBXSMM_RESTRICT b, float *LIBXSMM_RESTRICT c)
{
  libxsmm_smm(0/*flags*/, m, n, k, a, b, c, 0/*pa*/, 0/*pb*/, 0/*pc*/, 0/*alpha*/, 0/*beta*/);
}

/** Dispatched matrix multiplication (double-precision). */
LIBXSMM_RETARGETABLE inline void libxsmm_mm(int m, int n, int k,
  const double *LIBXSMM_RESTRICT a, const double *LIBXSMM_RESTRICT b, double *LIBXSMM_RESTRICT c)
{
  libxsmm_dmm(0/*flags*/, m, n, k, a, b, c, 0/*pa*/, 0/*pb*/, 0/*pc*/, 0/*alpha*/, 0/*beta*/);
}

/** Dispatched matrix multiplication (single-precision). */
LIBXSMM_RETARGETABLE inline void libxsmm_mm(int flags, int m, int n, int k,
  const float *LIBXSMM_RESTRICT a, const float *LIBXSMM_RESTRICT b, float *LIBXSMM_RESTRICT c)
{
  libxsmm_smm(flags, m, n, k, a, b, c, 0/*pa*/, 0/*pb*/, 0/*pc*/, 0/*alpha*/, 0/*beta*/);
}

/** Dispatched matrix multiplication (double-precision). */
LIBXSMM_RETARGETABLE inline void libxsmm_mm(int flags, int m, int n, int k,
  const double *LIBXSMM_RESTRICT a, const double *LIBXSMM_RESTRICT b, double *LIBXSMM_RESTRICT c)
{
  libxsmm_dmm(flags, m, n, k, a, b, c, 0/*pa*/, 0/*pb*/, 0/*pc*/, 0/*alpha*/, 0/*beta*/);
}

/** Dispatched matrix multiplication (single-precision). */
LIBXSMM_RETARGETABLE inline void libxsmm_mm(int flags, int m, int n, int k,
  const float *LIBXSMM_RESTRICT a, const float *LIBXSMM_RESTRICT b, float *LIBXSMM_RESTRICT c,
  const float* pa, const float* pb, const float* pc)
{
  libxsmm_smm(flags, m, n, k, a, b, c, pa, pb, pc, 0/*alpha*/, 0/*beta*/);
}

/** Dispatched matrix multiplication (double-precision). */
LIBXSMM_RETARGETABLE inline void libxsmm_mm(int flags, int m, int n, int k,
  const double *LIBXSMM_RESTRICT a, const double *LIBXSMM_RESTRICT b, double *LIBXSMM_RESTRICT c,
  const double* pa, const double* pb, const double* pc)
{
  libxsmm_dmm(flags, m, n, k, a, b, c, pa, pb, pc, 0/*alpha*/, 0/*beta*/);
}

/** Dispatched matrix multiplication (single-precision). */
LIBXSMM_RETARGETABLE inline void libxsmm_mm(int m, int n, int k,
  const float *LIBXSMM_RESTRICT a, const float *LIBXSMM_RESTRICT b, float *LIBXSMM_RESTRICT c,
  float alpha, float beta)
{
  libxsmm_smm(0/*flags*/, m, n, k, a, b, c, 0/*pa*/, 0/*pb*/, 0/*pc*/, &alpha, &beta);
}

/** Dispatched matrix multiplication (double-precision). */
LIBXSMM_RETARGETABLE inline void libxsmm_mm(int m, int n, int k,
  const double *LIBXSMM_RESTRICT a, const double *LIBXSMM_RESTRICT b, double *LIBXSMM_RESTRICT c,
  double alpha, double beta)
{
  libxsmm_dmm(0/*flags*/, m, n, k, a, b, c, 0/*pa*/, 0/*pb*/, 0/*pc*/, &alpha, &beta);
}

/** Dispatched matrix multiplication (single-precision). */
LIBXSMM_RETARGETABLE inline void libxsmm_mm(int flags, int m, int n, int k,
  const float *LIBXSMM_RESTRICT a, const float *LIBXSMM_RESTRICT b, float *LIBXSMM_RESTRICT c,
  float alpha, float beta)
{
  libxsmm_smm(flags, m, n, k, a, b, c, 0/*pa*/, 0/*pb*/, 0/*pc*/, &alpha, &beta);
}

/** Dispatched matrix multiplication (double-precision). */
LIBXSMM_RETARGETABLE inline void libxsmm_mm(int flags, int m, int n, int k,
  const double *LIBXSMM_RESTRICT a, const double *LIBXSMM_RESTRICT b, double *LIBXSMM_RESTRICT c,
  double alpha, double beta)
{
  libxsmm_dmm(flags, m, n, k, a, b, c, 0/*pa*/, 0/*pb*/, 0/*pc*/, &alpha, &beta);
}

/** Dispatched matrix multiplication (single-precision). */
LIBXSMM_RETARGETABLE inline void libxsmm_mm(int flags, int m, int n, int k,
  const float *LIBXSMM_RESTRICT a, const float *LIBXSMM_RESTRICT b, float *LIBXSMM_RESTRICT c,
  const float* pa, const float* pb, const float* pc,
  float alpha, float beta)
{
  libxsmm_smm(flags, m, n, k, a, b, c, pa, pb, pc, &alpha, &beta);
}

/** Dispatched matrix multiplication (double-precision). */
LIBXSMM_RETARGETABLE inline void libxsmm_mm(int flags, int m, int n, int k,
  const double *LIBXSMM_RESTRICT a, const double *LIBXSMM_RESTRICT b, double *LIBXSMM_RESTRICT c,
  const double* pa, const double* pb, const double* pc,
  double alpha, double beta)
{
  libxsmm_dmm(flags, m, n, k, a, b, c, pa, pb, pc, &alpha, &beta);
}

/** Non-dispatched matrix multiplication using BLAS (single-precision). */
LIBXSMM_RETARGETABLE inline void libxsmm_blasmm(int m, int n, int k,
  const float *LIBXSMM_RESTRICT a, const float *LIBXSMM_RESTRICT b, float *LIBXSMM_RESTRICT c)
{
  libxsmm_sblasmm(0, m, n, k, a, b, c, 0, 0);
}

/** Non-dispatched matrix multiplication using BLAS (double-precision). */
LIBXSMM_RETARGETABLE inline void libxsmm_blasmm(int m, int n, int k,
  const double *LIBXSMM_RESTRICT a, const double *LIBXSMM_RESTRICT b, double *LIBXSMM_RESTRICT c)
{
  libxsmm_dblasmm(0, m, n, k, a, b, c, 0, 0);
}

/** Non-dispatched matrix multiplication using BLAS (single-precision). */
LIBXSMM_RETARGETABLE inline void libxsmm_blasmm(int m, int n, int k,
  const float *LIBXSMM_RESTRICT a, const float *LIBXSMM_RESTRICT b, float *LIBXSMM_RESTRICT c,
  float alpha, float beta)
{
  libxsmm_sblasmm(0, m, n, k, a, b, c, &alpha, &beta);
}

/** Non-dispatched matrix multiplication using BLAS (double-precision). */
LIBXSMM_RETARGETABLE inline void libxsmm_blasmm(int m, int n, int k,
  const double *LIBXSMM_RESTRICT a, const double *LIBXSMM_RESTRICT b, double *LIBXSMM_RESTRICT c,
  double alpha, double beta)
{
  libxsmm_dblasmm(0, m, n, k, a, b, c, &alpha, &beta);
}

/** Non-dispatched matrix multiplication using BLAS (single-precision). */
LIBXSMM_RETARGETABLE inline void libxsmm_blasmm(int flags, int m, int n, int k,
  const float *LIBXSMM_RESTRICT a, const float *LIBXSMM_RESTRICT b, float *LIBXSMM_RESTRICT c)
{
  libxsmm_sblasmm(flags, m, n, k, a, b, c, 0, 0);
}

/** Non-dispatched matrix multiplication using BLAS (double-precision). */
LIBXSMM_RETARGETABLE inline void libxsmm_blasmm(int flags, int m, int n, int k,
  const double *LIBXSMM_RESTRICT a, const double *LIBXSMM_RESTRICT b, double *LIBXSMM_RESTRICT c)
{
  libxsmm_dblasmm(flags, m, n, k, a, b, c, 0, 0);
}

/** Non-dispatched matrix multiplication using BLAS (single-precision). */
LIBXSMM_RETARGETABLE inline void libxsmm_blasmm(int flags, int m, int n, int k,
  const float *LIBXSMM_RESTRICT a, const float *LIBXSMM_RESTRICT b, float *LIBXSMM_RESTRICT c,
  float alpha, float beta)
{
  libxsmm_sblasmm(flags, m, n, k, a, b, c, &alpha, &beta);
}

/** Non-dispatched matrix multiplication using BLAS (double-precision). */
LIBXSMM_RETARGETABLE inline void libxsmm_blasmm(int flags, int m, int n, int k,
  const double *LIBXSMM_RESTRICT a, const double *LIBXSMM_RESTRICT b, double *LIBXSMM_RESTRICT c,
  double alpha, double beta)
{
  libxsmm_dblasmm(flags, m, n, k, a, b, c, &alpha, &beta);
}

#endif /*__cplusplus*/

#endif /*LIBXSMM_H*/
