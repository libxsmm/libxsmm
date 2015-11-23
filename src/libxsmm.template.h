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

#define LIBXSMM_VERSION "$VERSION"
#define LIBXSMM_BRANCH  "$BRANCH"
#define LIBXSMM_VERSION_MAJOR   $MAJOR
#define LIBXSMM_VERSION_MINOR   $MINOR
#define LIBXSMM_VERSION_UPDATE  $UPDATE
#define LIBXSMM_VERSION_PATCH   $PATCH

/** Parameters the library and static kernels were built for. */
#define LIBXSMM_ALIGNMENT $ALIGNMENT
#define LIBXSMM_ROW_MAJOR $ROW_MAJOR
#define LIBXSMM_COL_MAJOR $COL_MAJOR
#define LIBXSMM_PREFETCH $PREFETCH
#define LIBXSMM_MAX_MNK $MAX_MNK
#define LIBXSMM_MAX_M $MAX_M
#define LIBXSMM_MAX_N $MAX_N
#define LIBXSMM_MAX_K $MAX_K
#define LIBXSMM_AVG_M $AVG_M
#define LIBXSMM_AVG_N $AVG_N
#define LIBXSMM_AVG_K $AVG_K
#define LIBXSMM_FLAGS $FLAGS
#define LIBXSMM_ILP64 $ILP64
#define LIBXSMM_ALPHA $ALPHA
#define LIBXSMM_BETA $BETA
#define LIBXSMM_JIT $JIT

#include "libxsmm_typedefs.h"
#include "libxsmm_frontend.h"


/** Integer type impacting the BLAS interface (LP64: 32-bit, and ILP64: 64-bit). */
#if (0 != LIBXSMM_ILP64)
typedef long long libxsmm_blasint;
#else
typedef int libxsmm_blasint;
#endif

/** Specialized function with fused alpha and beta arguments (single-precision). */
typedef LIBXSMM_RETARGETABLE void (*libxsmm_sfunction)(const float *LIBXSMM_RESTRICT a, const float *LIBXSMM_RESTRICT b, float *LIBXSMM_RESTRICT c);
/** Specialized function with fused alpha and beta arguments (double-precision). */
typedef LIBXSMM_RETARGETABLE void (*libxsmm_dfunction)(const double *LIBXSMM_RESTRICT a, const double *LIBXSMM_RESTRICT b, double *LIBXSMM_RESTRICT c);

/** Specialized function with alpha, beta, and prefetch arguments (single-precision). */
typedef LIBXSMM_RETARGETABLE void (*libxsmm_sxfunction)(const float *LIBXSMM_RESTRICT a, const float *LIBXSMM_RESTRICT b, float *LIBXSMM_RESTRICT c,
  const float* pa, const float* pb, const float* pc);
/** Specialized function with alpha, beta, and prefetch arguments (double-precision). */
typedef LIBXSMM_RETARGETABLE void (*libxsmm_dxfunction)(const double *LIBXSMM_RESTRICT a, const double *LIBXSMM_RESTRICT b, double *LIBXSMM_RESTRICT c,
  const double* pa, const double* pb, const double* pc);

/** Initialize the library; pay for setup cost at a specific point. */
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE void libxsmm_init(void);
/** Uninitialize the library and free internal memory (optional). */
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE void libxsmm_finalize(void);

/** Query or JIT-generate a function; return zero if it does not exist or if JIT is not supported (single-precision). */
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE libxsmm_sfunction libxsmm_sdispatch(int flags, int m, int n, int k, int lda, int ldb, int ldc,
  const float* alpha, const float* beta);
/** Query or JIT-generate a function; return zero if it does not exist or if JIT is not supported (double-precision). */
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE libxsmm_dfunction libxsmm_ddispatch(int flags, int m, int n, int k, int lda, int ldb, int ldc,
  const double* alpha, const double* beta);

/** Query or JIT-generate a function; return zero if it does not exist or if JIT is not supported (single-precision). */
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE libxsmm_sxfunction libxsmm_sxdispatch(int flags, int m, int n, int k, int lda, int ldb, int ldc,
  const float* alpha, const float* beta, int prefetch);
/** Query or JIT-generate a function; return zero if it does not exist or if JIT is not supported (double-precision). */
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE libxsmm_dxfunction libxsmm_dxdispatch(int flags, int m, int n, int k, int lda, int ldb, int ldc,
  const double* alpha, const double* beta, int prefetch);

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
  LIBXSMM_BLASMM(float, libxsmm_blasint, flags, m, n, k, a, b, c, alpha, beta);
}

/** Non-dispatched matrix multiplication using BLAS (double-precision). */
LIBXSMM_INLINE LIBXSMM_RETARGETABLE void libxsmm_dblasmm(int flags, int m, int n, int k,
  const double *LIBXSMM_RESTRICT a, const double *LIBXSMM_RESTRICT b, double *LIBXSMM_RESTRICT c,
  const double* alpha, const double* beta)
{
  LIBXSMM_BLASMM(double, libxsmm_blasint, flags, m, n, k, a, b, c, alpha, beta);
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
  libxsmm_function(int m, int n, int k, int flags = LIBXSMM_FLAGS)
    : m_function(reinterpret_cast<type>(libxsmm_sdispatch(flags, m, n, k, 0/*lda*/, 0/*ldb*/, 0/*ldc*/, 0/*alpha*/, 0/*beta*/)))
  {}
  libxsmm_function(int m, int n, int k, int lda, int ldb, int ldc, int flags = LIBXSMM_FLAGS)
    : m_function(reinterpret_cast<type>(libxsmm_sdispatch(flags, m, n, k, lda, ldb, ldc, 0/*alpha*/, 0/*beta*/)))
  {}
  libxsmm_function(int flags, int m, int n, int k, int prefetch)
    : m_function(LIBXSMM_PREFETCH_NONE != prefetch
      ? reinterpret_cast<type>(libxsmm_sxdispatch(flags, m, n, k, 0/*lda*/, 0/*ldb*/, 0/*ldc*/, 0/*alpha*/, 0/*beta*/, prefetch))
      : reinterpret_cast<type>(libxsmm_sdispatch(flags, m, n, k, 0/*lda*/, 0/*ldb*/, 0/*ldc*/, 0/*alpha*/, 0/*beta*/)))
  {}
  libxsmm_function(int flags, int m, int n, int k, int lda, int ldb, int ldc, int prefetch = LIBXSMM_PREFETCH)
    : m_function(LIBXSMM_PREFETCH_NONE != prefetch
      ? reinterpret_cast<type>(libxsmm_sxdispatch(flags, m, n, k, lda, ldb, ldc, 0/*alpha*/, 0/*beta*/, prefetch))
      : reinterpret_cast<type>(libxsmm_sdispatch(flags, m, n, k, lda, ldb, ldc, 0/*alpha*/, 0/*beta*/)))
  {}
  libxsmm_function(int m, int n, int k, float alpha, float beta, int flags = LIBXSMM_FLAGS, int prefetch = LIBXSMM_PREFETCH)
    : m_function(LIBXSMM_PREFETCH_NONE != prefetch
      ? reinterpret_cast<type>(libxsmm_sxdispatch(flags, m, n, k, 0/*lda*/, 0/*ldb*/, 0/*ldc*/, &alpha, &beta, prefetch))
      : reinterpret_cast<type>(libxsmm_sdispatch(flags, m, n, k, 0/*lda*/, 0/*ldb*/, 0/*ldc*/, &alpha, &beta)))
  {}
  libxsmm_function(int flags, int m, int n, int k, float alpha, float beta, int prefetch = LIBXSMM_PREFETCH)
    : m_function(LIBXSMM_PREFETCH_NONE != prefetch
      ? reinterpret_cast<type>(libxsmm_sxdispatch(flags, m, n, k, 0/*lda*/, 0/*ldb*/, 0/*ldc*/, &alpha, &beta, prefetch))
      : reinterpret_cast<type>(libxsmm_sdispatch(flags, m, n, k, 0/*lda*/, 0/*ldb*/, 0/*ldc*/, &alpha, &beta)))
  {}
  libxsmm_function(int flags, int m, int n, int k, int lda, int ldb, int ldc, float alpha, float beta, int prefetch = LIBXSMM_PREFETCH)
    : m_function(LIBXSMM_PREFETCH_NONE != prefetch
      ? reinterpret_cast<type>(libxsmm_sxdispatch(flags, m, n, k, lda, ldb, ldc, &alpha, &beta, prefetch))
      : reinterpret_cast<type>(libxsmm_sdispatch(flags, m, n, k, lda, ldb, ldc, &alpha, &beta)))
  {}
  libxsmm_function(int m, int n, int k, int lda, int ldb, int ldc, float alpha, float beta, int flags = LIBXSMM_FLAGS, int prefetch = LIBXSMM_PREFETCH)
    : m_function(LIBXSMM_PREFETCH_NONE != prefetch
      ? reinterpret_cast<type>(libxsmm_sxdispatch(flags, m, n, k, lda, ldb, ldc, &alpha, &beta, prefetch))
      : reinterpret_cast<type>(libxsmm_sdispatch(flags, m, n, k, lda, ldb, ldc, &alpha, &beta)))
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
    /* TODO: transition prefetch interface to xargs */
    m_function(a, b, c, pa, pb, pc);
  }
  /* TODO: support arbitrary Alpha and Beta in the backend
  void operator()(const float *LIBXSMM_RESTRICT a, const float *LIBXSMM_RESTRICT b, float *LIBXSMM_RESTRICT c,
    const float* pa, const float* pb, const float* pc,
    float alpha, float beta) const
  {
    TODO: build xargs here
    m_function(a, b, c, xargs);
  }*/
};

/** Construct and execute a specialized function (double-precision). */
template<> class LIBXSMM_RETARGETABLE libxsmm_function<double> {
  typedef LIBXSMM_RETARGETABLE void (*type)(const double *LIBXSMM_RESTRICT a, const double *LIBXSMM_RESTRICT b, double *LIBXSMM_RESTRICT c, ...);
  mutable/*retargetable*/ type m_function;
public:
  libxsmm_function(): m_function(0) {}
  libxsmm_function(int m, int n, int k, int flags = LIBXSMM_FLAGS)
    : m_function(reinterpret_cast<type>(libxsmm_ddispatch(flags, m, n, k, 0/*lda*/, 0/*ldb*/, 0/*ldc*/, 0/*alpha*/, 0/*beta*/)))
  {}
  libxsmm_function(int m, int n, int k, int lda, int ldb, int ldc, int flags = LIBXSMM_FLAGS)
    : m_function(reinterpret_cast<type>(libxsmm_ddispatch(flags, m, n, k, lda, ldb, ldc, 0/*alpha*/, 0/*beta*/)))
  {}
  libxsmm_function(int flags, int m, int n, int k, int prefetch)
    : m_function(LIBXSMM_PREFETCH_NONE != prefetch
      ? reinterpret_cast<type>(libxsmm_dxdispatch(flags, m, n, k, 0/*lda*/, 0/*ldb*/, 0/*ldc*/, 0/*alpha*/, 0/*beta*/, prefetch))
      : reinterpret_cast<type>(libxsmm_ddispatch(flags, m, n, k, 0/*lda*/, 0/*ldb*/, 0/*ldc*/, 0/*alpha*/, 0/*beta*/)))
  {}
  libxsmm_function(int flags, int m, int n, int k, int lda, int ldb, int ldc, int prefetch = LIBXSMM_PREFETCH)
    : m_function(LIBXSMM_PREFETCH_NONE != prefetch
      ? reinterpret_cast<type>(libxsmm_dxdispatch(flags, m, n, k, lda, ldb, ldc, 0/*alpha*/, 0/*beta*/, prefetch))
      : reinterpret_cast<type>(libxsmm_ddispatch(flags, m, n, k, lda, ldb, ldc, 0/*alpha*/, 0/*beta*/)))
  {}
  libxsmm_function(int m, int n, int k, double alpha, double beta, int flags = LIBXSMM_FLAGS, int prefetch = LIBXSMM_PREFETCH)
    : m_function(LIBXSMM_PREFETCH_NONE != prefetch
      ? reinterpret_cast<type>(libxsmm_dxdispatch(flags, m, n, k, 0/*lda*/, 0/*ldb*/, 0/*ldc*/, &alpha, &beta, prefetch))
      : reinterpret_cast<type>(libxsmm_ddispatch(flags, m, n, k, 0/*lda*/, 0/*ldb*/, 0/*ldc*/, &alpha, &beta)))
  {}
  libxsmm_function(int flags, int m, int n, int k, double alpha, double beta, int prefetch = LIBXSMM_PREFETCH)
    : m_function(LIBXSMM_PREFETCH_NONE != prefetch
      ? reinterpret_cast<type>(libxsmm_dxdispatch(flags, m, n, k, 0/*lda*/, 0/*ldb*/, 0/*ldc*/, &alpha, &beta, prefetch))
      : reinterpret_cast<type>(libxsmm_ddispatch(flags, m, n, k, 0/*lda*/, 0/*ldb*/, 0/*ldc*/, &alpha, &beta)))
  {}
  libxsmm_function(int flags, int m, int n, int k, int lda, int ldb, int ldc, double alpha, double beta, int prefetch = LIBXSMM_PREFETCH)
    : m_function(LIBXSMM_PREFETCH_NONE != prefetch
      ? reinterpret_cast<type>(libxsmm_dxdispatch(flags, m, n, k, lda, ldb, ldc, &alpha, &beta, prefetch))
      : reinterpret_cast<type>(libxsmm_ddispatch(flags, m, n, k, lda, ldb, ldc, &alpha, &beta)))
  {}
  libxsmm_function(int m, int n, int k, int lda, int ldb, int ldc, double alpha, double beta, int flags = LIBXSMM_FLAGS, int prefetch = LIBXSMM_PREFETCH)
    : m_function(LIBXSMM_PREFETCH_NONE != prefetch
      ? reinterpret_cast<type>(libxsmm_dxdispatch(flags, m, n, k, lda, ldb, ldc, &alpha, &beta, prefetch))
      : reinterpret_cast<type>(libxsmm_ddispatch(flags, m, n, k, lda, ldb, ldc, &alpha, &beta)))
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
    /* TODO: transition prefetch interface to xargs */
    m_function(a, b, c, pa, pb, pc);
  }
  /* TODO: support arbitrary Alpha and Beta in the backend
  void operator()(const double *LIBXSMM_RESTRICT a, const double *LIBXSMM_RESTRICT b, double *LIBXSMM_RESTRICT c,
    const double* pa, const double* pb, const double* pc,
    double alpha, double beta) const
  {
    TODO: build xargs here
    m_function(a, b, c, pa, pb, pc, alpha, beta);
  }*/
};

/** Dispatched matrix multiplication (single-precision). */
LIBXSMM_RETARGETABLE inline void libxsmm_mm(int m, int n, int k,
  const float *LIBXSMM_RESTRICT a, const float *LIBXSMM_RESTRICT b, float *LIBXSMM_RESTRICT c,
  int flags = LIBXSMM_FLAGS)
{
  libxsmm_smm(flags, m, n, k, a, b, c, 0/*pa*/, 0/*pb*/, 0/*pc*/, 0/*alpha*/, 0/*beta*/);
}

/** Dispatched matrix multiplication (double-precision). */
LIBXSMM_RETARGETABLE inline void libxsmm_mm(int m, int n, int k,
  const double *LIBXSMM_RESTRICT a, const double *LIBXSMM_RESTRICT b, double *LIBXSMM_RESTRICT c,
  int flags = LIBXSMM_FLAGS)
{
  libxsmm_dmm(flags, m, n, k, a, b, c, 0/*pa*/, 0/*pb*/, 0/*pc*/, 0/*alpha*/, 0/*beta*/);
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
LIBXSMM_RETARGETABLE inline void libxsmm_mm(int m, int n, int k,
  const float *LIBXSMM_RESTRICT a, const float *LIBXSMM_RESTRICT b, float *LIBXSMM_RESTRICT c,
  const float* pa, const float* pb, const float* pc,
  int flags = LIBXSMM_FLAGS)
{
  libxsmm_smm(flags, m, n, k, a, b, c, pa, pb, pc, 0/*alpha*/, 0/*beta*/);
}

/** Dispatched matrix multiplication (double-precision). */
LIBXSMM_RETARGETABLE inline void libxsmm_mm(int m, int n, int k,
  const double *LIBXSMM_RESTRICT a, const double *LIBXSMM_RESTRICT b, double *LIBXSMM_RESTRICT c,
  const double* pa, const double* pb, const double* pc,
  int flags = LIBXSMM_FLAGS)
{
  libxsmm_dmm(flags, m, n, k, a, b, c, pa, pb, pc, 0/*alpha*/, 0/*beta*/);
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
  float alpha, float beta, int flags = LIBXSMM_FLAGS)
{
  libxsmm_smm(flags, m, n, k, a, b, c, 0/*pa*/, 0/*pb*/, 0/*pc*/, &alpha, &beta);
}

/** Dispatched matrix multiplication (double-precision). */
LIBXSMM_RETARGETABLE inline void libxsmm_mm(int m, int n, int k,
  const double *LIBXSMM_RESTRICT a, const double *LIBXSMM_RESTRICT b, double *LIBXSMM_RESTRICT c,
  double alpha, double beta, int flags = LIBXSMM_FLAGS)
{
  libxsmm_dmm(flags, m, n, k, a, b, c, 0/*pa*/, 0/*pb*/, 0/*pc*/, &alpha, &beta);
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
LIBXSMM_RETARGETABLE inline void libxsmm_mm(int m, int n, int k,
  const float *LIBXSMM_RESTRICT a, const float *LIBXSMM_RESTRICT b, float *LIBXSMM_RESTRICT c,
  const float* pa, const float* pb, const float* pc,
  float alpha, float beta, int flags = LIBXSMM_FLAGS)
{
  libxsmm_smm(flags, m, n, k, a, b, c, pa, pb, pc, &alpha, &beta);
}

/** Dispatched matrix multiplication (double-precision). */
LIBXSMM_RETARGETABLE inline void libxsmm_mm(int m, int n, int k,
  const double *LIBXSMM_RESTRICT a, const double *LIBXSMM_RESTRICT b, double *LIBXSMM_RESTRICT c,
  const double* pa, const double* pb, const double* pc,
  double alpha, double beta, int flags = LIBXSMM_FLAGS)
{
  libxsmm_dmm(flags, m, n, k, a, b, c, pa, pb, pc, &alpha, &beta);
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
  const float *LIBXSMM_RESTRICT a, const float *LIBXSMM_RESTRICT b, float *LIBXSMM_RESTRICT c,
  int flags = LIBXSMM_FLAGS)
{
  libxsmm_sblasmm(flags, m, n, k, a, b, c, 0/*alpha*/, 0/*beta*/);
}

/** Non-dispatched matrix multiplication using BLAS (double-precision). */
LIBXSMM_RETARGETABLE inline void libxsmm_blasmm(int m, int n, int k,
  const double *LIBXSMM_RESTRICT a, const double *LIBXSMM_RESTRICT b, double *LIBXSMM_RESTRICT c,
  int flags = LIBXSMM_FLAGS)
{
  libxsmm_dblasmm(flags, m, n, k, a, b, c, 0/*alpha*/, 0/*beta*/);
}

/** Non-dispatched matrix multiplication using BLAS (single-precision). */
LIBXSMM_RETARGETABLE inline void libxsmm_blasmm(int flags, int m, int n, int k,
  const float *LIBXSMM_RESTRICT a, const float *LIBXSMM_RESTRICT b, float *LIBXSMM_RESTRICT c)
{
  libxsmm_sblasmm(flags, m, n, k, a, b, c, 0/*alpha*/, 0/*beta*/);
}

/** Non-dispatched matrix multiplication using BLAS (double-precision). */
LIBXSMM_RETARGETABLE inline void libxsmm_blasmm(int flags, int m, int n, int k,
  const double *LIBXSMM_RESTRICT a, const double *LIBXSMM_RESTRICT b, double *LIBXSMM_RESTRICT c)
{
  libxsmm_dblasmm(flags, m, n, k, a, b, c, 0/*alpha*/, 0/*beta*/);
}

/** Non-dispatched matrix multiplication using BLAS (single-precision). */
LIBXSMM_RETARGETABLE inline void libxsmm_blasmm(int m, int n, int k,
  const float *LIBXSMM_RESTRICT a, const float *LIBXSMM_RESTRICT b, float *LIBXSMM_RESTRICT c,
  float alpha, float beta, int flags = LIBXSMM_FLAGS)
{
  libxsmm_sblasmm(flags, m, n, k, a, b, c, &alpha, &beta);
}

/** Non-dispatched matrix multiplication using BLAS (double-precision). */
LIBXSMM_RETARGETABLE inline void libxsmm_blasmm(int m, int n, int k,
  const double *LIBXSMM_RESTRICT a, const double *LIBXSMM_RESTRICT b, double *LIBXSMM_RESTRICT c,
  double alpha, double beta, int flags = LIBXSMM_FLAGS)
{
  libxsmm_dblasmm(flags, m, n, k, a, b, c, &alpha, &beta);
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
