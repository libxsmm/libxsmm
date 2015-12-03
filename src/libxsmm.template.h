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


/** Integer type for LAPACK/BLAS (LP64: 32-bit, and ILP64: 64-bit). */
#if (0 != LIBXSMM_ILP64)
typedef long long libxsmm_blasint;
#else
typedef int libxsmm_blasint;
#endif

/** Specialized function with fused alpha and beta arguments, and optional prefetch locations (single-precision). */
typedef LIBXSMM_RETARGETABLE void (*libxsmm_sfunction)(const float *LIBXSMM_RESTRICT a, const float *LIBXSMM_RESTRICT b, float *LIBXSMM_RESTRICT c, ...);
/** Specialized function with fused alpha and beta arguments, and optional prefetch locations (double-precision). */
typedef LIBXSMM_RETARGETABLE void (*libxsmm_dfunction)(const double *LIBXSMM_RESTRICT a, const double *LIBXSMM_RESTRICT b, double *LIBXSMM_RESTRICT c, ...);

/** Initialize the library; pay for setup cost at a specific point. */
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE void libxsmm_init(void);
/** Uninitialize the library and free internal memory (optional). */
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE void libxsmm_finalize(void);

/** Query or JIT-generate a function; return zero if it does not exist or if JIT is not supported (single-precision). */
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE libxsmm_sfunction libxsmm_sdispatch(int m, int n, int k,
  const int* lda, const int* ldb, const int* ldc,
  const float* alpha, const float* beta,
  const int* flags, const int* prefetch);
/** Query or JIT-generate a function; return zero if it does not exist or if JIT is not supported (double-precision). */
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE libxsmm_dfunction libxsmm_ddispatch(int m, int n, int k,
  const int* lda, const int* ldb, const int* ldc,
  const double* alpha, const double* beta,
  const int* flags, const int* prefetch);

/** Dispatched general dense matrix multiplication (single-precision). */
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE void libxsmm_sgemm(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const float* alpha, const float *LIBXSMM_RESTRICT a, const libxsmm_blasint* lda,
  const float *LIBXSMM_RESTRICT b, const libxsmm_blasint* ldb,
  const float* beta, float *LIBXSMM_RESTRICT c, const libxsmm_blasint* ldc);

/** Dispatched general dense matrix multiplication (double-precision). */
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE void libxsmm_dgemm(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const double* alpha, const double *LIBXSMM_RESTRICT a, const libxsmm_blasint* lda,
  const double *LIBXSMM_RESTRICT b, const libxsmm_blasint* ldb,
  const double* beta, double *LIBXSMM_RESTRICT c, const libxsmm_blasint* ldc);

/** General dense matrix multiplication based on LAPACK/BLAS (single-precision). */
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE void libxsmm_blas_sgemm(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const float* alpha, const float *LIBXSMM_RESTRICT a, const libxsmm_blasint* lda,
  const float *LIBXSMM_RESTRICT b, const libxsmm_blasint* ldb,
  const float* beta, float *LIBXSMM_RESTRICT c, const libxsmm_blasint* ldc);

/** General dense matrix multiplication based on LAPACK/BLAS (double-precision). */
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE void libxsmm_blas_dgemm(const char* transa, const char* transb,
  const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const double* alpha, const double *LIBXSMM_RESTRICT a, const libxsmm_blasint* lda,
  const double *LIBXSMM_RESTRICT b, const libxsmm_blasint* ldb,
  const double* beta, double *LIBXSMM_RESTRICT c, const libxsmm_blasint* ldc);
$MNK_INTERFACE_LIST
#if defined(__cplusplus)

/** Construct and execute a specialized function. */
template<typename T> class LIBXSMM_RETARGETABLE libxsmm_function {};

/** Construct and execute a specialized function (single-precision). */
template<> class LIBXSMM_RETARGETABLE libxsmm_function<float> {
  mutable/*retargetable*/ libxsmm_sfunction m_function;
public:
  libxsmm_function(): m_function(0) {}
  libxsmm_function(int m, int n, int k, int flags = LIBXSMM_FLAGS)
    : m_function(libxsmm_sdispatch(m, n, k, 0/*lda*/, 0/*ldb*/, 0/*ldc*/, 0/*alpha*/, 0/*beta*/, &flags, 0/*prefetch*/))
  {}
  libxsmm_function(int m, int n, int k, int lda, int ldb, int ldc, int flags = LIBXSMM_FLAGS)
    : m_function(libxsmm_sdispatch(m, n, k, &lda, &ldb, &ldc, 0/*alpha*/, 0/*beta*/, &flags, 0/*prefetch*/))
  {}
  libxsmm_function(int flags, int m, int n, int k, int prefetch)
    : m_function(libxsmm_sdispatch(m, n, k, 0/*lda*/, 0/*ldb*/, 0/*ldc*/, 0/*alpha*/, 0/*beta*/, &flags, &prefetch))
  {}
  libxsmm_function(int flags, int m, int n, int k, int lda, int ldb, int ldc, int prefetch = LIBXSMM_PREFETCH)
    : m_function(libxsmm_sdispatch(m, n, k, &lda, &ldb, &ldc, 0/*alpha*/, 0/*beta*/, &flags, &prefetch))
  {}
  libxsmm_function(int m, int n, int k, float alpha, float beta, int flags = LIBXSMM_FLAGS, int prefetch = LIBXSMM_PREFETCH)
    : m_function(libxsmm_sdispatch(m, n, k, 0/*lda*/, 0/*ldb*/, 0/*ldc*/, &alpha, &beta, &flags, &prefetch))
  {}
  libxsmm_function(int flags, int m, int n, int k, float alpha, float beta, int prefetch = LIBXSMM_PREFETCH)
    : m_function(libxsmm_sdispatch(m, n, k, 0/*lda*/, 0/*ldb*/, 0/*ldc*/, &alpha, &beta, &flags, &prefetch))
  {}
  libxsmm_function(int flags, int m, int n, int k, int lda, int ldb, int ldc, float alpha, float beta, int prefetch = LIBXSMM_PREFETCH)
    : m_function(libxsmm_sdispatch(m, n, k, &lda, &ldb, &ldc, &alpha, &beta, &flags, &prefetch))
  {}
  libxsmm_function(int m, int n, int k, int lda, int ldb, int ldc, float alpha, float beta, int flags = LIBXSMM_FLAGS, int prefetch = LIBXSMM_PREFETCH)
    : m_function(libxsmm_sdispatch(m, n, k, &lda, &ldb, &ldc, &alpha, &beta, &flags, &prefetch))
  {}
public:
  operator libxsmm_sfunction() const {
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
  mutable/*retargetable*/ libxsmm_dfunction m_function;
public:
  libxsmm_function(): m_function(0) {}
  libxsmm_function(int m, int n, int k, int flags = LIBXSMM_FLAGS)
    : m_function(libxsmm_ddispatch(m, n, k, 0/*lda*/, 0/*ldb*/, 0/*ldc*/, 0/*alpha*/, 0/*beta*/, &flags, 0/*prefetch*/))
  {}
  libxsmm_function(int m, int n, int k, int lda, int ldb, int ldc, int flags = LIBXSMM_FLAGS)
    : m_function(libxsmm_ddispatch(m, n, k, &lda, &ldb, &ldc, 0/*alpha*/, 0/*beta*/, &flags, 0/*prefetch*/))
  {}
  libxsmm_function(int flags, int m, int n, int k, int prefetch)
    : m_function(libxsmm_ddispatch(m, n, k, 0/*lda*/, 0/*ldb*/, 0/*ldc*/, 0/*alpha*/, 0/*beta*/, &flags, &prefetch))
  {}
  libxsmm_function(int flags, int m, int n, int k, int lda, int ldb, int ldc, int prefetch = LIBXSMM_PREFETCH)
    : m_function(libxsmm_ddispatch(m, n, k, &lda, &ldb, &ldc, 0/*alpha*/, 0/*beta*/, &flags, &prefetch))
  {}
  libxsmm_function(int m, int n, int k, double alpha, double beta, int flags = LIBXSMM_FLAGS, int prefetch = LIBXSMM_PREFETCH)
    : m_function(libxsmm_ddispatch(m, n, k, 0/*lda*/, 0/*ldb*/, 0/*ldc*/, &alpha, &beta, &flags, &prefetch))
  {}
  libxsmm_function(int flags, int m, int n, int k, double alpha, double beta, int prefetch = LIBXSMM_PREFETCH)
    : m_function(libxsmm_ddispatch(m, n, k, 0/*lda*/, 0/*ldb*/, 0/*ldc*/, &alpha, &beta, &flags, &prefetch))
  {}
  libxsmm_function(int flags, int m, int n, int k, int lda, int ldb, int ldc, double alpha, double beta, int prefetch = LIBXSMM_PREFETCH)
    : m_function(libxsmm_ddispatch(m, n, k, &lda, &ldb, &ldc, &alpha, &beta, &flags, &prefetch))
  {}
  libxsmm_function(int m, int n, int k, int lda, int ldb, int ldc, double alpha, double beta, int flags = LIBXSMM_FLAGS, int prefetch = LIBXSMM_PREFETCH)
    : m_function(libxsmm_ddispatch(m, n, k, &lda, &ldb, &ldc, &alpha, &beta, &flags, &prefetch))
  {}
public:
  operator libxsmm_dfunction() const {
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

/** Dispatched general dense matrix multiplication (single-precision). */
LIBXSMM_RETARGETABLE inline void libxsmm_gemm(const char* transa, const char* transb, const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const float* alpha, const float *LIBXSMM_RESTRICT a, const libxsmm_blasint* lda, const float *LIBXSMM_RESTRICT b, const libxsmm_blasint* ldb,
  const float* beta, float *LIBXSMM_RESTRICT c, const libxsmm_blasint* ldc)
{
  libxsmm_sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

/** Dispatched general dense matrix multiplication (double-precision). */
LIBXSMM_RETARGETABLE inline void libxsmm_gemm(const char* transa, const char* transb, const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const double* alpha, const double *LIBXSMM_RESTRICT a, const libxsmm_blasint* lda, const double *LIBXSMM_RESTRICT b, const libxsmm_blasint* ldb,
  const double* beta, double *LIBXSMM_RESTRICT c, const libxsmm_blasint* ldc)
{
  libxsmm_dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

/** Dispatched general dense matrix multiplication (single-precision). */
LIBXSMM_RETARGETABLE inline void libxsmm_sgemm(const char* transa, const char* transb, libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const float* alpha, const float *LIBXSMM_RESTRICT a, const libxsmm_blasint* lda, const float *LIBXSMM_RESTRICT b, const libxsmm_blasint* ldb,
  const float* beta, float *LIBXSMM_RESTRICT c, const libxsmm_blasint* ldc)
{
  libxsmm_sgemm(transa, transb, &m, &n, &k, alpha, a, lda, b, ldb, beta, c, ldc);
}

/** Dispatched general dense matrix multiplication (double-precision). */
LIBXSMM_RETARGETABLE inline void libxsmm_dgemm(const char* transa, const char* transb, libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const double* alpha, const double *LIBXSMM_RESTRICT a, const libxsmm_blasint* lda, const double *LIBXSMM_RESTRICT b, const libxsmm_blasint* ldb,
  const double* beta, double *LIBXSMM_RESTRICT c, const libxsmm_blasint* ldc)
{
  libxsmm_dgemm(transa, transb, &m, &n, &k, alpha, a, lda, b, ldb, beta, c, ldc);
}

/** Dispatched general dense matrix multiplication (single-precision). */
LIBXSMM_RETARGETABLE inline void libxsmm_gemm(const char* transa, const char* transb, libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const float* alpha, const float *LIBXSMM_RESTRICT a, const libxsmm_blasint* lda, const float *LIBXSMM_RESTRICT b, const libxsmm_blasint* ldb,
  const float* beta, float *LIBXSMM_RESTRICT c, const libxsmm_blasint* ldc)
{
  libxsmm_sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

/** Dispatched general dense matrix multiplication (double-precision). */
LIBXSMM_RETARGETABLE inline void libxsmm_gemm(const char* transa, const char* transb, libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const double* alpha, const double *LIBXSMM_RESTRICT a, const libxsmm_blasint* lda, const double *LIBXSMM_RESTRICT b, const libxsmm_blasint* ldb,
  const double* beta, double *LIBXSMM_RESTRICT c, const libxsmm_blasint* ldc)
{
  libxsmm_dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

/** General dense matrix multiplication based on LAPACK/BLAS (single-precision). */
LIBXSMM_RETARGETABLE inline void libxsmm_blas_gemm(const char* transa, const char* transb, const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const float* alpha, const float *LIBXSMM_RESTRICT a, const libxsmm_blasint* lda, const float *LIBXSMM_RESTRICT b, const libxsmm_blasint* ldb,
  const float* beta, float *LIBXSMM_RESTRICT c, const libxsmm_blasint* ldc)
{
  libxsmm_blas_sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

/** General dense matrix multiplication based on LAPACK/BLAS (double-precision). */
LIBXSMM_RETARGETABLE inline void libxsmm_blas_gemm(const char* transa, const char* transb, const libxsmm_blasint* m, const libxsmm_blasint* n, const libxsmm_blasint* k,
  const double* alpha, const double *LIBXSMM_RESTRICT a, const libxsmm_blasint* lda, const double *LIBXSMM_RESTRICT b, const libxsmm_blasint* ldb,
  const double* beta, double *LIBXSMM_RESTRICT c, const libxsmm_blasint* ldc)
{
  libxsmm_blas_dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

/** General dense matrix multiplication based on LAPACK/BLAS (single-precision). */
LIBXSMM_RETARGETABLE inline void libxsmm_blas_sgemm(const char* transa, const char* transb, libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const float* alpha, const float *LIBXSMM_RESTRICT a, const libxsmm_blasint* lda, const float *LIBXSMM_RESTRICT b, const libxsmm_blasint* ldb,
  const float* beta, float *LIBXSMM_RESTRICT c, const libxsmm_blasint* ldc)
{
  libxsmm_blas_sgemm(transa, transb, &m, &n, &k, alpha, a, lda, b, ldb, beta, c, ldc);
}

/** General dense matrix multiplication based on LAPACK/BLAS (double-precision). */
LIBXSMM_RETARGETABLE inline void libxsmm_blas_dgemm(const char* transa, const char* transb, libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const double* alpha, const double *LIBXSMM_RESTRICT a, const libxsmm_blasint* lda, const double *LIBXSMM_RESTRICT b, const libxsmm_blasint* ldb,
  const double* beta, double *LIBXSMM_RESTRICT c, const libxsmm_blasint* ldc)
{
  libxsmm_blas_dgemm(transa, transb, &m, &n, &k, alpha, a, lda, b, ldb, beta, c, ldc);
}

/** General dense matrix multiplication based on LAPACK/BLAS (single-precision). */
LIBXSMM_RETARGETABLE inline void libxsmm_blas_gemm(const char* transa, const char* transb, libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const float* alpha, const float *LIBXSMM_RESTRICT a, const libxsmm_blasint* lda, const float *LIBXSMM_RESTRICT b, const libxsmm_blasint* ldb,
  const float* beta, float *LIBXSMM_RESTRICT c, const libxsmm_blasint* ldc)
{
  libxsmm_blas_sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

/** General dense matrix multiplication based on LAPACK/BLAS (double-precision). */
LIBXSMM_RETARGETABLE inline void libxsmm_blas_gemm(const char* transa, const char* transb, libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint k,
  const double* alpha, const double *LIBXSMM_RESTRICT a, const libxsmm_blasint* lda, const double *LIBXSMM_RESTRICT b, const libxsmm_blasint* ldb,
  const double* beta, double *LIBXSMM_RESTRICT c, const libxsmm_blasint* ldc)
{
  libxsmm_blas_dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

#endif /*__cplusplus*/
#endif /*LIBXSMM_H*/
