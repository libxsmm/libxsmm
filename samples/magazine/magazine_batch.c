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
#if !defined(SHUFFLE)
# include <libxsmm.h>
#endif

#if defined(_OPENMP)
# define USEOMP(FUNCTION) LIBXSMM_USEOMP(FUNCTION)
#else
# define USEOMP(FUNCTION) (FUNCTION)
#endif


int main(int argc, char* argv[])
{
  /* batch-size is used to stream matrix-operands from memory */
  const int batchsize = (1 < argc ? atoi(argv[1]) : 0/*auto*/);
  /* default: M, N, and K are 13, 5, and 7 respectively */
  const libxsmm_blasint m = (2 < argc ? atoi(argv[2]) : 13);
  const libxsmm_blasint n = (3 < argc ? atoi(argv[3]) : 5);
  const libxsmm_blasint k = (4 < argc ? atoi(argv[4]) : 7);
  /* leading dimensions are made multiples of the size of a cache-line */
  const libxsmm_blasint lda = (5 < argc ? LIBXSMM_MAX(atoi(argv[5]), m) : (libxsmm_blasint)(LIBXSMM_UP2(sizeof(TYPE) * m, PAD) / sizeof(TYPE)));
  const libxsmm_blasint ldb = (6 < argc ? LIBXSMM_MAX(atoi(argv[6]), k) : (libxsmm_blasint)(LIBXSMM_UP2(sizeof(TYPE) * k, PAD) / sizeof(TYPE)));
  const libxsmm_blasint ldc = (7 < argc ? LIBXSMM_MAX(atoi(argv[7]), m) : (libxsmm_blasint)(LIBXSMM_UP2(sizeof(TYPE) * m, PAD) / sizeof(TYPE)));
  /* micro-kernels are limited to certain alpha- and beta-values */
  const char transa = 'n', transb = 'n';
  const TYPE alpha = 1, beta = 1;
  /* calculate matrix sizes incl. padded elements */
  const size_t na = LIBXSMM_UP2(sizeof(TYPE) * lda * k, PAD) / sizeof(TYPE);
  const size_t nb = LIBXSMM_UP2(sizeof(TYPE) * ldb * n, PAD) / sizeof(TYPE);
  const size_t nc = LIBXSMM_UP2(sizeof(TYPE) * ldc * n, PAD) / sizeof(TYPE);
  /* calculate default batch-size to hit work-set size of approx. 2 GB */
  const int size = (0 >= batchsize ? (int)((2ULL << 30/*2 GB*/) / (sizeof(TYPE) * (na + nb + nc))) : batchsize);
#if defined(SHUFFLE)
  const size_t shuffle = libxsmm_shuffle((unsigned int)size);
#endif
  /* allocate A, B, and C matrix buffers */
  TYPE *const a = (TYPE*)libxsmm_aligned_malloc(sizeof(TYPE) * na * size, LIBXSMM_CACHELINE);
  TYPE *const b = (TYPE*)libxsmm_aligned_malloc(sizeof(TYPE) * nb * size, LIBXSMM_CACHELINE);
  TYPE *const c = (TYPE*)libxsmm_aligned_malloc(sizeof(TYPE) * nc * size, LIBXSMM_CACHELINE);
  libxsmm_blasint *const ia = (libxsmm_blasint*)libxsmm_malloc(sizeof(libxsmm_blasint) * size);
  libxsmm_blasint *const ib = (libxsmm_blasint*)libxsmm_malloc(sizeof(libxsmm_blasint) * size);
  libxsmm_blasint *const ic = (libxsmm_blasint*)libxsmm_malloc(sizeof(libxsmm_blasint) * size);
  const double scale = 1.0 / size;
  libxsmm_timer_tickint start;
  double duration;
#if 1 /* synchronize among C-locations */
  const libxsmm_blasint xsize = size;
#else /* assume no data race (C-index) */
  const libxsmm_blasint xsize = -size;
#endif
  int i;

  /* initialize data according to touch-first policy */
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < size; ++i) {
#if defined(SHUFFLE)
    const int j = (i * shuffle) % size;
#else
    const int j = i;
#endif
    init(25 + i, a + j * na, (int)m, (int)k, (int)lda, scale);
    init(75 + i, b + j * nb, (int)k, (int)n, (int)ldb, scale);
    if (LIBXSMM_NEQ(0, beta)) { /* no need to initialize for beta=0 */
      init(42 + i, c + j * nc, (int)m, (int)n, (int)ldc, scale);
    }
    ia[i] = (int)STREAM_A(j * na);
    ib[i] = (int)STREAM_B(j * nb);
    ic[i] = (int)STREAM_C(SYNC(j, nc, size));
  }

  start = libxsmm_timer_tick();
  USEOMP(libxsmm_gemm_batch)(LIBXSMM_GEMM_PRECISION(TYPE), LIBXSMM_GEMM_PRECISION(TYPE),
    &transa, &transb, m, n, k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc,
    0/*index_base*/, sizeof(int)/*index_stride*/, ia, ib, ic, xsize);
  duration = libxsmm_timer_duration(start, libxsmm_timer_tick());

  if (0 < duration) {
    const double gflops = 2.0 * m * n * k * 1E-9;
    printf("%.1f GFLOPS/s\n", gflops / duration * size);
  }
  printf("%.1f ms\n", 1000.0 * duration);

  { /* calculate checksum */
    double check = 0;
    for (i = 0; i < size; ++i) {
      const double cn = norm(c + STREAM_C(SYNC(i, nc, size)), (int)m, (int)n, (int)ldc);
      if (check < cn) check = cn;
    }
    printf("\n%f (check)\n", check);
  }
  libxsmm_free(ia);
  libxsmm_free(ib);
  libxsmm_free(ic);
  libxsmm_free(a);
  libxsmm_free(b);
  libxsmm_free(c);

  return EXIT_SUCCESS;
}

