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
#if !defined(__BLAZE) && 0
# define __BLAZE
#endif

#if defined(__BLAZE)
# if !defined(BLAZE_USE_SHARED_MEMORY_PARALLELIZATION)
/* Example uses outer parallelism hence Blaze-internal parallelism is disabled */
#   define BLAZE_USE_SHARED_MEMORY_PARALLELIZATION 0
# endif
# define _mm512_setzero_epi16 _mm512_setzero_si512
# define _mm512_setzero_epi8  _mm512_setzero_si512
# include <blaze/Blaze.h>
#endif
#include <memory>
#include <cstdlib>
#include <cstdio>


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


/**
 * Example program that multiplies matrices independently (C += A * B).
 * A and B-matrices are accumulated into a single C matrix (beta=1).
 * Streaming A, B, C, AB, AC, BC, or ABC are other useful benchmarks
 * However, running a kernel without loading any matrix operand from
 * memory ("cache-hot loop") is not modeling typical applications
 * since no actual work is performed.
 */
int main(int argc, char* argv[])
{
#if defined(__BLAZE)
  typedef double T;
  typedef blaze::CustomMatrix<T,blaze::aligned,blaze::padded,blaze::columnMajor> matrix_type;
  const size_t alignment = 64; /* must be power of two */

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
  blaze::timing::WcTimer timer;

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
    timer.start();
#   pragma omp for
#endif
    for (int i = 0; i < size; ++i) {
      const matrix_type a(pa + i * na, m, k, lda);
      const matrix_type b(pb + i * nb, k, n, ldb);
      matrix_type c(pc + i * nc, m, n, ldc);
      /**
       * Expression templates attempt to delay evaluation until the sequence point
       * is reached, or an "expression object" goes out of scope and hence must
       * materialize the effect. Ideally, a complex expression is mapped to the
       * best possible implementation e.g., c = alpha * a * b + beta * c may be
       * mapped to GEMM or definitely omits alpha*a in case of alpha=1, or similar
       * for special cases for beta=0 and beta=1.
       * However, to not rely on an ideal transformation a *manually specialized*
       * expression is written for e.g., alpha=1 and beta=1 (c += a * b).
       * NOTE: changing alpha or beta from above may not have an effect
       *       depending on what is selected below (expression).
       */
#if 0 /* alpha=1 anyway */
      c = alpha * a * b + beta * c;
#elif 0
      (void)alpha; /* unused */
      c = a * b + beta * c;
#elif 0 /* beta=0 */
      (void)alpha; /* unused */
      (void)beta; /* unused */
      c = a * b;
#else /* beta=1 */
      (void)alpha; /* unused */
      (void)beta; /* unused */
      c += a * b;
#endif
    }
  }
  timer.end();

  if (0 < timer.total()) {
    const double gflops = 2.0 * m * n * k * 1E-9;
    printf("%.1f GFLOPS/s\n", gflops / timer.total() * size);
  }
  printf("%.1f ms\n", 1000.0 * timer.total());

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

