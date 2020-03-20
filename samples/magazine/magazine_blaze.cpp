/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Hans Pabst (Intel Corp.)
******************************************************************************/
#include "magazine.h"

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


int main(int argc, char* argv[])
{
#if defined(__BLAZE)
  typedef TYPE T;
  typedef blaze::CustomMatrix<T,blaze::unaligned,blaze::unpadded,blaze::columnMajor> matrix_type;
  /* batch-size is used to stream matrix-operands from memory */
  const int batchsize = (1 < argc ? atoi(argv[1]) : 0/*auto*/);
  /* default: M, N, and K are 13, 5, and 7 respectively */
  const int m = (2 < argc ? atoi(argv[2]) : 13);
  const int n = (3 < argc ? atoi(argv[3]) : 5);
  const int k = (4 < argc ? atoi(argv[4]) : 7);
  /* leading dimensions are used to each pad (row-major!) */
  const int lda = (5 < argc ? (m < atoi(argv[5]) ? atoi(argv[5]) : m) : static_cast<int>(((sizeof(T) * m + PAD - 1) & ~(PAD - 1)) / sizeof(T)));
  const int ldb = (6 < argc ? (k < atoi(argv[6]) ? atoi(argv[6]) : k) : static_cast<int>(((sizeof(T) * k + PAD - 1) & ~(PAD - 1)) / sizeof(T)));
  const int ldc = (7 < argc ? (m < atoi(argv[7]) ? atoi(argv[7]) : m) : static_cast<int>(((sizeof(T) * m + PAD - 1) & ~(PAD - 1)) / sizeof(T)));
#if 0
  const char transa = 'n', transb = 'n';
#endif
  const T alpha = 1, beta = 1;
  /* calculate matrix sizes incl. padded elements */
  const size_t na = ((sizeof(T) * lda * k + PAD - 1) & ~(PAD - 1)) / sizeof(T);
  const size_t nb = ((sizeof(T) * ldb * n + PAD - 1) & ~(PAD - 1)) / sizeof(T);
  const size_t nc = ((sizeof(T) * ldc * n + PAD - 1) & ~(PAD - 1)) / sizeof(T);
  /* calculate default batch-size to hit work-set size of approx. 2 GB */
  const int size = (0 >= batchsize ? static_cast<int>((2ULL << 30/*2 GB*/) / (sizeof(T) * (na + nb + nc))) : batchsize);
#if defined(SHUFFLE)
  const size_t shuffle = libxsmm_shuffle((unsigned int)size);
#endif
  size_t sa = sizeof(T) * na * size + PAD - 1;
  size_t sb = sizeof(T) * nb * size + PAD - 1;
  size_t sc = sizeof(T) * nc * size + PAD - 1;
  /* allocate A, B, and C matrix buffers */
  void *const va = malloc(sa), *const vb = malloc(sb), *const vc = malloc(sc), *wa = va, *wb = vb, *wc = vc;
  /* align memory according to PAD */
#if defined(PAD) && (1 < (PAD))
  T *const pa = static_cast<T*>(std::align(PAD, sa - PAD + 1, wa, sa));
  T *const pb = static_cast<T*>(std::align(PAD, sb - PAD + 1, wb, sb));
  T *const pc = static_cast<T*>(std::align(PAD, sc - PAD + 1, wc, sc));
#else
  T *const pa = static_cast<T*>(wa);
  T *const pb = static_cast<T*>(wb);
  T *const pc = static_cast<T*>(wc);
#endif
  const double scale = 1.0 / size;
  blaze::timing::WcTimer timer;

  /* initialize data according to touch-first policy */
#if defined(_OPENMP)
# pragma omp parallel for
#endif
  for (int i = 0; i < size; ++i) {
#if defined(SHUFFLE)
    const int j = (i * shuffle) % size;
#else
    const int j = i;
#endif
    init(25 + i, pa + j * na, m, k, lda, scale);
    init(75 + i, pb + j * nb, k, n, ldb, scale);
    if (0 != beta) { /* no need to initialize for beta=0 */
      init(42 + i, pc + j * nc, m, n, ldc, scale);
    }
  }

#if defined(_OPENMP)
# pragma omp parallel
#endif
  {
#if defined(_OPENMP)
#   pragma omp single
#endif
    timer.start();
#if defined(_OPENMP)
#   pragma omp for
#endif
    for (int i = 0; i < size; ++i) {
#if defined(SHUFFLE)
      const int j = (i * shuffle) % size;
#else
      const int j = i;
#endif
      const matrix_type a(pa + STREAM_A(j * na), m, k, lda);
      const matrix_type b(pb + STREAM_B(j * nb), k, n, ldb);
            matrix_type c(pc + STREAM_C(j * nc), m, n, ldc);
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

  { /* calculate checksum */
    double check = 0;
    for (int i = 0; i < size; ++i) {
      const double cn = norm(pc + STREAM_C(i * nc), m, n, ldc);
      if (check < cn) check = cn;
    }
    printf("\n%f (check)\n", check);
  }
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

