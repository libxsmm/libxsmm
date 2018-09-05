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
#include "magazine.h"
#include <libxsmm.h>

#if 0 /* process batch of A, B, and C in "random" order */
# define SHUFFLE
#endif
#if 0 /* manually dispatch SMM kernel */
# define KERNEL
#endif
#if 0 /* synchronization among C matrices */
# define SYNC
#endif


int main(int argc, char* argv[])
{
  /* batch-size is used to stream matrix-operands from memory */
  const int batchsize = (1 < argc ? atoi(argv[1]) : 0/*auto*/);
  /* default: M, N, and K are 13, 5, and 7 respectively */
  const int m = (2 < argc ? atoi(argv[2]) : 13);
  const int n = (3 < argc ? atoi(argv[3]) : 5);
  const int k = (4 < argc ? atoi(argv[4]) : 7);
  /* leading dimensions are made multiples of the size of a cache-line */
  const int lda = LIBXSMM_UP2(sizeof(TYPE) * m, LIBXSMM_CACHELINE) / sizeof(TYPE);
  const int ldb = LIBXSMM_UP2(sizeof(TYPE) * k, LIBXSMM_CACHELINE) / sizeof(TYPE);
  const int ldc = LIBXSMM_UP2(sizeof(TYPE) * m, LIBXSMM_CACHELINE) / sizeof(TYPE);
  /* micro-kernels are limited to certain alpha- and beta-values */
  const char transa = 'n', transb = 'n';
  const TYPE alpha = 1, beta = 1;
  /* calculate matrix sizes incl. padded elements */
  const size_t na = LIBXSMM_UP2(sizeof(TYPE) * lda * k, LIBXSMM_CACHELINE) / sizeof(TYPE);
  const size_t nb = LIBXSMM_UP2(sizeof(TYPE) * ldb * n, LIBXSMM_CACHELINE) / sizeof(TYPE);
  const size_t nc = LIBXSMM_UP2(sizeof(TYPE) * ldc * n, LIBXSMM_CACHELINE) / sizeof(TYPE);
  /* calculate default batch-size to hit work-set size of approx. 2 GB */
  const int size = (0 >= batchsize ? (int)((2ULL << 30/*2 GB*/) / (sizeof(TYPE) * (na + nb + nc))) : batchsize);
#if defined(SHUFFLE)
  const size_t shuffle = libxsmm_shuffle((unsigned int)size);
#endif
  /* allocate A, B, and C matrix buffers */
  TYPE *const a = (TYPE*)libxsmm_aligned_malloc(sizeof(TYPE) * na * size, LIBXSMM_CACHELINE);
  TYPE *const b = (TYPE*)libxsmm_aligned_malloc(sizeof(TYPE) * nb * size, LIBXSMM_CACHELINE);
  TYPE *const c = (TYPE*)libxsmm_aligned_malloc(sizeof(TYPE) * nc * size, LIBXSMM_CACHELINE);
  int *const ia = (int*)libxsmm_malloc(sizeof(int) * size), i;
  int *const ib = (int*)libxsmm_malloc(sizeof(int) * size);
  int *const ic = (int*)libxsmm_malloc(sizeof(int) * size);
  const double scale = 1.0 / size;
  libxsmm_timer_tickint start;
  double duration;
#if defined(SYNC)
  const int xsize = size;
#else
  const int xsize = -size;
#endif

  /**
   * LIBXSMM's C interface really is type-specific, and the helper macros (such as LIBXSMM_MMFUNCTION_TYPE)
   * are only for "entertainment". The C++ interface on the other hand is provides overloaded functions
   * and some helpers for more type-generic programming tasks (e.g., libxsmm_mmfunction<T>).
   */
#if defined(KERNEL) /* explicitly dispatch a kernel according to parameters */
  libxsmm_descriptor_blob blob;
  const libxsmm_gemm_descriptor *const desc = libxsmm_gemm_descriptor_init(&blob, LIBXSMM_GEMM_PRECISION(TYPE),
    m, n, k, lda, ldb, ldc, &alpha, &beta, LIBXSMM_GEMM_FLAGS(transa, transb),
    libxsmm_get_gemm_prefetch(LIBXSMM_PREFETCH_AUTO));
  const libxsmm_xmmfunction xmm = libxsmm_xmmdispatch(desc);
#endif

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
    init(25 + i, a + j * na, m, k, lda, scale);
    init(75 + i, b + j * nb, k, n, ldb, scale);
    if (LIBXSMM_NEQ(0, beta)) { /* no need to initialize for beta=0 */
      init(42 + i, c + j * nc, m, n, ldc, scale);
    }
    ia[i] = (int)STREAM_A(j * na);
    ib[i] = (int)STREAM_B(j * nb);
    ic[i] = (int)STREAM_C(j * nc);
  }

  start = libxsmm_timer_tick();
#if defined(KERNEL) /* explicitly dispatch a kernel according to parameters */
  libxsmm_mmbatch_omp(xmm, 0/*index_base*/, sizeof(int)/*index_stride*/, ia, ib, ic, a, b, c, xsize);
#else
  libxsmm_gemm_batch_omp(LIBXSMM_GEMM_PRECISION(TYPE),
    &transa, &transb, m, n, k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc,
    0/*index_base*/, sizeof(int)/*index_stride*/, ia, ib, ic, xsize);
#endif
  duration = libxsmm_timer_duration(start, libxsmm_timer_tick());

  if (0 < duration) {
    const double gflops = 2.0 * m * n * k * 1E-9;
    printf("%.1f GFLOPS/s\n", gflops / duration * size);
  }
  printf("%.1f ms\n", 1000.0 * duration);

  libxsmm_free(ia);
  libxsmm_free(ib);
  libxsmm_free(ic);
  libxsmm_free(a);
  libxsmm_free(b);
  libxsmm_free(c);

  return EXIT_SUCCESS;
}

