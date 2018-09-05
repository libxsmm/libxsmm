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
#if !defined(__EIGEN) && 0
# define __EIGEN
#endif

#if defined(__EIGEN)
# if !defined(EIGEN_DONT_PARALLELIZE)
#   define EIGEN_DONT_PARALLELIZE
# endif
# include <Eigen/Dense>
# if defined(_OPENMP)
#   include <omp.h>
# else
#   include <bench/BenchTimer.h>
# endif
#endif
#include <memory>
#include <cstdlib>
#include <cstdio>

#if 1
# define STREAM_A(EXPR) (EXPR)
#else
# define STREAM_A(EXPR) 0
#endif
#if 1
# define STREAM_B(EXPR) (EXPR)
#else
# define STREAM_B(EXPR) 0
#endif
#if 1
# define STREAM_C(EXPR) (EXPR)
#else
# define STREAM_C(EXPR) 0
#endif


template<typename T> void init(int seed, T* dst, int nrows, int ncols, int ld, double scale) {
  const double seed1 = scale * seed + scale;
  for (int i = 0; i < ncols; ++i) {
    int j = 0;
    for (; j < nrows; ++j) {
      const int k = i * ld + j;
      dst[k] = static_cast<T>(seed1 / (1.0 + k));
    }
    for (; j < ld; ++j) {
      const int k = i * ld + j;
      dst[k] = static_cast<T>(seed);
    }
  }
}


#if defined(__EIGEN)
template<bool pad> struct stride_helper {
  stride_helper(int pad_a, int pad_b, int pad_c): a(pad_a, 1), b(pad_b, 1), c(pad_c, 1) {}
  Eigen::Stride<Eigen::Dynamic,Eigen::Dynamic> a, b, c;
};
template<> struct stride_helper<false> {
  stride_helper(...) {}
  Eigen::Stride<0,0> a, b, c;
};
#endif


int main(int argc, char* argv[])
{
#if defined(__EIGEN)
  typedef double T;
  typedef Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor> matrix_type;
#if 1 /* dynamic strides make things slower even if lda == m, etc. */
  const size_t alignment = 64; /* must be power of two */
#else
  const size_t alignment = 1;
#endif
  /* batch-size is used to stream matrix-operands from memory */
  const int batchsize = (1 < argc ? atoi(argv[1]) : 0/*auto*/);
  /* default: M, N, and K are 13, 5, and 7 respectively */
  const int m = (2 < argc ? atoi(argv[2]) : 13);
  const int n = (3 < argc ? atoi(argv[3]) : 5);
  const int k = (4 < argc ? atoi(argv[4]) : 7);
  /* leading dimensions are used to each pad (row-major!) */
  const int lda = ((sizeof(T) * m + alignment - 1) & ~(alignment - 1)) / sizeof(T);
  const int ldb = ((sizeof(T) * k + alignment - 1) & ~(alignment - 1)) / sizeof(T);
  const int ldc = ((sizeof(T) * m + alignment - 1) & ~(alignment - 1)) / sizeof(T);
  /* Eigen specifies leading dimensions per "outer stride" */
  stride_helper<(sizeof(T)<alignment)> stride(lda, ldb, ldc);
#if 0
  const char transa = 'n', transb = 'n';
#endif
  const T alpha = 1, beta = 1;
  /* calculate matrix sizes incl. padded elements */
  const size_t na = ((sizeof(T) * lda * k + alignment - 1) & ~(alignment - 1)) / sizeof(T);
  const size_t nb = ((sizeof(T) * ldb * n + alignment - 1) & ~(alignment - 1)) / sizeof(T);
  const size_t nc = ((sizeof(T) * ldc * n + alignment - 1) & ~(alignment - 1)) / sizeof(T);
  /* calculate default batch-size to hit work-set size of approx. 2 GB */
  const int size = (0 >= batchsize ? static_cast<int>((2ULL << 30/*2 GB*/) / (sizeof(T) * (na + nb + nc))) : batchsize);
  size_t sa = sizeof(T) * na * size + alignment - 1;
  size_t sb = sizeof(T) * nb * size + alignment - 1;
  size_t sc = sizeof(T) * nc * size + alignment - 1;
  /* allocate A, B, and C matrix buffers */
  void *const va = malloc(sa), *const vb = malloc(sb), *const vc = malloc(sc), *wa = va, *wb = vb, *wc = vc;
  /* align memory according to alignment */
  T *const pa = static_cast<T*>(std::align(alignment, sa - alignment + 1, wa, sa));
  T *const pb = static_cast<T*>(std::align(alignment, sb - alignment + 1, wb, sb));
  T *const pc = static_cast<T*>(std::align(alignment, sc - alignment + 1, wc, sc));
  const double scale = 1.0 / size;
  double duration;
#if !defined(_OPENMP)
  Eigen::BenchTimer timer;
#endif

  /* initialize data according to touch-first policy */
#if defined(_OPENMP)
# pragma omp parallel for
#endif
  for (int i = 0; i < size; ++i) {
    init(25 + i, pa + i * na, m, k, lda, scale);
    init(75 + i, pb + i * nb, k, n, ldb, scale);
    init(42 + i, pc + i * nc, m, n, ldc, scale);
  }

#if defined(_OPENMP)
# pragma omp parallel
#endif
  {
#if !defined(_OPENMP)
    timer.start();
#else /* OpenMP thread pool is already populated (parallel region) */
#   pragma omp single
    duration = omp_get_wtime();
#   pragma omp for
#endif
    for (int i = 0; i < size; ++i) {
      /* using "matrix_type" instead of "auto" induces an unnecessary copy */
      const auto a = matrix_type::Map/*Aligned*/(pa + STREAM_A(i * na), m, k, stride.a);
      const auto b = matrix_type::Map/*Aligned*/(pb + STREAM_B(i * nb), k, n, stride.b);
            auto c = matrix_type::Map/*Aligned*/(pc + STREAM_C(i * nc), m, n, stride.c);
      /**
       * Expression templates attempt to delay evaluation until the sequence point
       * is reached, or an "expression object" goes out of scope and hence must
       * materialize the effect. Ideally, a complex expression is mapped to the
       * best possible implementation e.g., c = alpha * a * b + beta * c may be
       * mapped to GEMM or definitely omits alpha*a in case of alpha=1, or similar
       * for special cases for beta=0 and beta=1. However, to not rely on an ideal
       * transformation a *manually specialized* expression is written for e.g.,
       * alpha=1 and beta=1 (c += a * b) or tweaked manually ("noalias").
       * NOTE: changing alpha or beta from above may not have an effect
       *       depending on what is selected below (expression).
       */
#if 0 /* alpha=1 anyway */
      c.noalias() = alpha * a * b + beta * c;
#elif 0
      (void)alpha; /* unused */
      c.noalias() = a * b + beta * c;
#elif 0 /* beta=0 */
      (void)alpha; /* unused */
      (void)beta; /* unused */
      c.noalias() = a * b;
#else /* beta=1 */
      (void)alpha; /* unused */
      (void)beta; /* unused */
      c.noalias() += a * b;
#endif
    }
  }
#if defined(_OPENMP)
  duration = omp_get_wtime() - duration;
#else
  timer.stop();
  duration = timer.total();
#endif
  if (0 < duration) {
    const double gflops = 2.0 * m * n * k * 1E-9;
    printf("%.1f GFLOPS/s\n", gflops / duration * size);
  }
  printf("%.1f ms\n", 1000.0 * duration);

  free(va);
  free(vb);
  free(vc);
  return EXIT_SUCCESS;
#else
  (void)argc; /* unused */
  (void)argv; /* unused */
  return EXIT_FAILURE;
#endif
}

