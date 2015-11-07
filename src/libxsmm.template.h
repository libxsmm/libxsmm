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


/** Structure providing the actual/extended arguments of an SGEMM call. */
typedef struct libxsmm_sgemm_xargs {
  /** The Alpha and Beta arguments. */
  float alpha, beta;
  /** The prefetch arguments. */
  LIBXSMM_PREFETCH_DECL(const float* pa)
  LIBXSMM_PREFETCH_DECL(const float* pb)
  LIBXSMM_PREFETCH_DECL(const float* pc)
} libxsmm_sgemm_xargs;

/** Structure providing the actual/extended arguments of a DGEMM call. */
typedef struct libxsmm_dgemm_xargs {
  /** The Alpha and Beta arguments. */
  double alpha, beta;
  /** The prefetch arguments. */
  LIBXSMM_PREFETCH_DECL(const double* pa)
  LIBXSMM_PREFETCH_DECL(const double* pb)
  LIBXSMM_PREFETCH_DECL(const double* pc)
} libxsmm_dgemm_xargs;

/** Generic type of a function. */
typedef LIBXSMM_RETARGETABLE void (*libxsmm_sfunction)(const float *LIBXSMM_RESTRICT a, const float *LIBXSMM_RESTRICT b, float *LIBXSMM_RESTRICT c, const libxsmm_sgemm_xargs* xargs);
typedef LIBXSMM_RETARGETABLE void (*libxsmm_dfunction)(const double *LIBXSMM_RESTRICT a, const double *LIBXSMM_RESTRICT b, double *LIBXSMM_RESTRICT c, const libxsmm_dgemm_xargs* xargs);

/** Initialize the library; pay for setup cost at a specific point. */
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE void libxsmm_init(void);

/** Query or JIT-generate a function; return zero if it does not exist or if JIT is not supported (single-precision). */
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE libxsmm_sfunction libxsmm_sdispatch(int m, int n, int k, float alpha, float beta,
  int lda, int ldb, int ldc, int flags, int prefetch);
/** Query or JIT-generate a function; return zero if it does not exist or if JIT is not supported (double-precision). */
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE libxsmm_dfunction libxsmm_ddispatch(int m, int n, int k, double alpha, double beta,
  int lda, int ldb, int ldc, int flags, int prefetch);

/** Dispatched matrix-matrix multiplication (single-precision). */
LIBXSMM_INLINE LIBXSMM_RETARGETABLE void libxsmm_smm(int m, int n, int k,
  const float *LIBXSMM_RESTRICT a, const float *LIBXSMM_RESTRICT b, float *LIBXSMM_RESTRICT c,
  const libxsmm_sgemm_xargs* xargs)
{
  LIBXSMM_MM(float, m, n, k, a, b, c, xargs);
}

/** Dispatched matrix-matrix multiplication (double-precision). */
LIBXSMM_INLINE LIBXSMM_RETARGETABLE void libxsmm_dmm(int m, int n, int k,
  const double *LIBXSMM_RESTRICT a, const double *LIBXSMM_RESTRICT b, double *LIBXSMM_RESTRICT c,
  const libxsmm_dgemm_xargs* xargs)
{
  LIBXSMM_MM(double, m, n, k, a, b, c, xargs);
}

/** Non-dispatched matrix-matrix multiplication using BLAS (single-precision). */
LIBXSMM_INLINE LIBXSMM_RETARGETABLE void libxsmm_sblasmm(int m, int n, int k,
  const float *LIBXSMM_RESTRICT a, const float *LIBXSMM_RESTRICT b, float *LIBXSMM_RESTRICT c,
  const libxsmm_sgemm_xargs* xargs)
{
  LIBXSMM_BLASMM(float, m, n, k, a, b, c, xargs);
}

/** Non-dispatched matrix-matrix multiplication using BLAS (double-precision). */
LIBXSMM_INLINE LIBXSMM_RETARGETABLE void libxsmm_dblasmm(int m, int n, int k,
  const double *LIBXSMM_RESTRICT a, const double *LIBXSMM_RESTRICT b, double *LIBXSMM_RESTRICT c,
  const libxsmm_dgemm_xargs* xargs)
{
  LIBXSMM_BLASMM(double, m, n, k, a, b, c, xargs);
}
$MNK_INTERFACE_LIST
#if defined(__cplusplus)

/** Function type depending on T. */
template<typename T> struct LIBXSMM_RETARGETABLE libxsmm_function   { typedef void type; };
template<> struct LIBXSMM_RETARGETABLE libxsmm_function<float>      { typedef libxsmm_sfunction type; };
template<> struct LIBXSMM_RETARGETABLE libxsmm_function<double>     { typedef libxsmm_dfunction type; };

/** Extended argument type depending on T. */
template<typename T> struct LIBXSMM_RETARGETABLE libxsmm_gemm_xargs { typedef void type; };
template<> struct LIBXSMM_RETARGETABLE libxsmm_gemm_xargs<float>    { typedef libxsmm_sgemm_xargs type; };
template<> struct LIBXSMM_RETARGETABLE libxsmm_gemm_xargs<double>   { typedef libxsmm_dgemm_xargs type; };

/** Query or JIT-generate a function; return zero if it does not exist or if JIT is not supported. */
template<typename T> class LIBXSMM_RETARGETABLE libxsmm_dispatch {};
template<> class LIBXSMM_RETARGETABLE libxsmm_dispatch<float> {
  mutable/*retargetable*/ libxsmm_sfunction m_function;
public:
  libxsmm_dispatch(): m_function(0) {}
  libxsmm_dispatch(int m, int n, int k,
    float alpha = LIBXSMM_ALPHA, float beta = LIBXSMM_BETA,
    int lda = 0, int ldb = 0, int ldc = 0,
    int flags = LIBXSMM_GEMM_FLAG_DEFAULT,
    int prefetch = LIBXSMM_PREFETCH)  : m_function(libxsmm_sdispatch(m, n, k, alpha, beta, lda, ldb, ldc, flags, prefetch))
  {}
  operator libxsmm_sfunction() const {
    return m_function;
  }
  void operator()(const float a[], const float b[], float c[], const libxsmm_sgemm_xargs* xargs = 0) const {
    m_function(a, b, c, xargs);
  }
  void operator()(const float a[], const float b[], float c[], const libxsmm_sgemm_xargs& xargs) const {
    m_function(a, b, c, &xargs);
  }
};
template<> class LIBXSMM_RETARGETABLE libxsmm_dispatch<double> {
  mutable/*retargetable*/ libxsmm_dfunction m_function;
public:
  libxsmm_dispatch(): m_function(0) {}
  libxsmm_dispatch(int m, int n, int k,
    double alpha = LIBXSMM_ALPHA, double beta = LIBXSMM_BETA,
    int lda = 0, int ldb = 0, int ldc = 0,
    int flags = LIBXSMM_GEMM_FLAG_DEFAULT,
    int prefetch = LIBXSMM_PREFETCH)  : m_function(libxsmm_ddispatch(m, n, k, alpha, beta, lda, ldb, ldc, flags, prefetch))
  {}
  operator libxsmm_dfunction() const {
    return m_function;
  }
  void operator()(const double a[], const double b[], double c[], const libxsmm_dgemm_xargs* xargs = 0) const {
    m_function(a, b, c, xargs);
  }
  void operator()(const double a[], const double b[], double c[], const libxsmm_dgemm_xargs& xargs) const {
    m_function(a, b, c, &xargs);
  }
};

/** Dispatched matrix-matrix multiplication. */
LIBXSMM_RETARGETABLE inline void libxsmm_mm(int m, int n, int k,
  const float *LIBXSMM_RESTRICT a, const float *LIBXSMM_RESTRICT b, float *LIBXSMM_RESTRICT c,
  const libxsmm_sgemm_xargs* xargs = 0)
{
  libxsmm_smm(m, n, k, a, b, c, xargs);
}

/** Dispatched matrix-matrix multiplication. */
LIBXSMM_RETARGETABLE inline void libxsmm_mm(int m, int n, int k,
  const double *LIBXSMM_RESTRICT a, const double *LIBXSMM_RESTRICT b, double *LIBXSMM_RESTRICT c,
  const libxsmm_dgemm_xargs* xargs = 0)
{
  libxsmm_dmm(m, n, k, a, b, c, xargs);
}

/** Non-dispatched matrix-matrix multiplication using BLAS. */
LIBXSMM_RETARGETABLE inline void libxsmm_blasmm(int m, int n, int k,
  const float *LIBXSMM_RESTRICT a, const float *LIBXSMM_RESTRICT b, float *LIBXSMM_RESTRICT c,
  const libxsmm_sgemm_xargs* xargs = 0)
{
  libxsmm_sblasmm(m, n, k, a, b, c, xargs);
}

/** Non-dispatched matrix-matrix multiplication using BLAS. */
LIBXSMM_RETARGETABLE inline void libxsmm_blasmm(int m, int n, int k,
  const double *LIBXSMM_RESTRICT a, const double *LIBXSMM_RESTRICT b, double *LIBXSMM_RESTRICT c,
  const libxsmm_dgemm_xargs* xargs = 0)
{
  libxsmm_dblasmm(m, n, k, a, b, c, xargs);
}

#endif /*__cplusplus*/

#endif /*LIBXSMM_H*/
