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

#if 0 /* auto-dispatch SMM kernel */
# define AUTO
#endif
#if 0 /* disable auto-prefetch */
# define NOPREFETCH
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
  const double scale = 1.0 / size;
  libxsmm_timer_tickint start;
  double duration;
  int i, j;

  /**
   * LIBXSMM's C interface really is type-specific, and the helper macros (such as LIBXSMM_MMFUNCTION_TYPE)
   * are only for "entertainment". The C++ interface on the other hand provides overloaded functions
   * and some helpers for more type-generic programming tasks (e.g., libxsmm_mmfunction<T>).
   */
#if !defined(AUTO) /* explicitly dispatch a kernel according to parameters */
  const int flags = LIBXSMM_GEMM_FLAGS(transa, transb);
# if !defined(NOPREFETCH) && (STREAM_A(1) || STREAM_B(1) || STREAM_C(1)) /* prefetch */
  const int prefetch = LIBXSMM_PREFETCH_AUTO;
# else
  const int prefetch = LIBXSMM_PREFETCH_NONE;
# endif
  union { /* convert between fn.ptr and (data)pointer */
    LIBXSMM_MMFUNCTION_TYPE(TYPE) fun;
    const void* ptr;
  } xmm;
  xmm.fun = LIBXSMM_MMDISPATCH_SYMBOL(TYPE)(m, n, k, &lda, &ldb, &ldc, &alpha, &beta, &flags, &prefetch);
#endif

  /* initialize data according to touch-first policy */
#if defined(_OPENMP)
# pragma omp parallel for private(i, j)
#endif
  for (i = 0; i < size; ++i) {
#if defined(SHUFFLE)
    j = (i * shuffle) % size;
#else
    j = i;
#endif
    init(25 + i, a + j * na, (int)m, (int)k, (int)lda, scale);
    init(75 + i, b + j * nb, (int)k, (int)n, (int)ldb, scale);
    if (LIBXSMM_NEQ(0, beta)) { /* no need to initialize for beta=0 */
      init(42 + i, c + j * nc, (int)m, (int)n, (int)ldc, scale);
    }
  }

#if defined(_OPENMP)
# pragma omp parallel
#endif
  { /* OpenMP thread pool is already populated (parallel region) */
#if defined(_OPENMP)
#   pragma omp single
#endif
    start = libxsmm_timer_tick();
#if defined(_OPENMP)
#   pragma omp for private(i, j)
#endif
    for (i = 0; i < size - 1; ++i) {
#if defined(SHUFFLE)
# if !defined(AUTO) && !defined(NOPREFETCH) && (STREAM_A(1) || STREAM_B(1) || STREAM_C(1)) /* prefetch */
      const int p = ((i + 1) * shuffle) % size;
# endif
      j = (i * shuffle) % size;
#else
# if !defined(AUTO) && !defined(NOPREFETCH) && (STREAM_A(1) || STREAM_B(1) || STREAM_C(1)) /* prefetch */
      const int p = i + 1; /* next location */
# endif
      j = i;
#endif
#if defined(AUTO)
      libxsmm_dgemm(&transa, &transb, &m, &n, &k,
        &alpha, a + STREAM_A(j * na), &lda, b + STREAM_B(j * nb), &ldb,
         &beta, c + STREAM_C(SYNC(j, nc, size)), &ldc);
#elif !defined(NOPREFETCH) && (STREAM_A(1) || STREAM_B(1) || STREAM_C(1)) /* prefetch */
      xmm.fun(a + STREAM_A(j * na), b + STREAM_B(j * nb), c + STREAM_C(SYNC(j, nc, size)),
              a + STREAM_A(p * na), b + STREAM_B(p * nb), c + STREAM_C(SYNC(p, nc, size)));
#else
      xmm.fun(a + STREAM_A(j * na), b + STREAM_B(j * nb), c + STREAM_C(SYNC(j, nc, size)));
#endif
    }
  }
#if defined(SHUFFLE)
  j = ((size - 1) * shuffle) % size;
#else
  j = size - 1;
#endif
#if defined(AUTO)
  libxsmm_dgemm(&transa, &transb, &m, &n, &k,
    &alpha, a + STREAM_A(j * na), &lda, b + STREAM_B(j * nb), &ldb,
     &beta, c + STREAM_C(SYNC(j, nc, size)), &ldc);
#elif !defined(NOPREFETCH) && (STREAM_A(1) || STREAM_B(1) || STREAM_C(1)) /* prefetch */
  xmm.fun(a + STREAM_A(j * na), b + STREAM_B(j * nb), c + STREAM_C(SYNC(j, nc, size)),
          a + STREAM_A(j * na), b + STREAM_B(j * nb), c + STREAM_C(SYNC(j, nc, size)));
#else
  xmm.fun(a + STREAM_A(j * na), b + STREAM_B(j * nb), c + STREAM_C(SYNC(j, nc, size)));
#endif
  duration = libxsmm_timer_duration(start, libxsmm_timer_tick());

  if (0 < duration) {
    libxsmm_kernel_info info;
#if defined(AUTO) /* no explicit kernel hence no query */
    info.nflops = 2 * m * n * k;
#else
    libxsmm_get_kernel_info(xmm.ptr, &info);
#endif
    printf("%.1f GFLOPS/s\n", (1E-9 * info.nflops) / duration * size);
  }
  printf("%.1f ms\n", 1000.0 * duration);

  { /* calculate checksum */
    double check = 0;
    for (i = 0; i < size; ++i) {
      const double cn = norm(c + STREAM_C(i * nc), (int)m, (int)n, (int)ldc);
      if (check < cn) check = cn;
    }
    printf("\n%f (check)\n", check);
  }
  libxsmm_free(a);
  libxsmm_free(b);
  libxsmm_free(c);

  return EXIT_SUCCESS;
}

