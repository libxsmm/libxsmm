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

#if defined(__MKL) || defined(MKL_DIRECT_CALL_SEQ) || defined(MKL_DIRECT_CALL)
# include <mkl.h>
#define GEMM_float  sgemm
#define GEMM_double dgemm
#else /* prototypes for GEMM */
#define GEMM_float  sgemm_
#define GEMM_double dgemm_
void dgemm_(const char*, const char*, const int*, const int*, const int*,
  const double*, const double*, const int*, const double*, const int*,
  const double*, double*, const int*);
void sgemm_(const char*, const char*, const int*, const int*, const int*,
  const float*, const float*, const int*, const float*, const int*,
  const float*, float*, const int*);
#endif
#include <stdlib.h>
#include <stdint.h>

#if !defined(GEMM)
# define CONCATENATE_AUX(A, B) A##B
# define CONCATENATE(A, B) CONCATENATE_AUX(A, B)
# define GEMM CONCATENATE(GEMM_, TYPE)
#endif
#if !defined(INTEL_MKL_VERSION) || (20190003 <= INTEL_MKL_VERSION)
# define NOFALLBACK
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
  const int lda = (5 < argc ? (m < atoi(argv[5]) ? atoi(argv[5]) : m) : (int)(((sizeof(TYPE) * m + PAD - 1) & ~(PAD - 1)) / sizeof(TYPE)));
  const int ldb = (6 < argc ? (k < atoi(argv[6]) ? atoi(argv[6]) : k) : (int)(((sizeof(TYPE) * k + PAD - 1) & ~(PAD - 1)) / sizeof(TYPE)));
  const int ldc = (7 < argc ? (m < atoi(argv[7]) ? atoi(argv[7]) : m) : (int)(((sizeof(TYPE) * m + PAD - 1) & ~(PAD - 1)) / sizeof(TYPE)));
  /* micro-kernels are limited to certain alpha- and beta-values */
  const char transa = 'n', transb = 'n';
  const TYPE alpha = 1, beta = 1;
  /* calculate matrix sizes incl. padded elements */
  const size_t na = ((sizeof(TYPE) * lda * k + PAD - 1) & ~(PAD - 1)) / sizeof(TYPE);
  const size_t nb = ((sizeof(TYPE) * ldb * n + PAD - 1) & ~(PAD - 1)) / sizeof(TYPE);
  const size_t nc = ((sizeof(TYPE) * ldc * n + PAD - 1) & ~(PAD - 1)) / sizeof(TYPE);
  /* calculate default batch-size to hit work-set size of approx. 2 GB */
  const int size = (0 >= batchsize ? (int)((2ULL << 30/*2 GB*/) / (sizeof(TYPE) * (na + nb + nc))) : batchsize);
#if defined(SHUFFLE)
  const size_t shuffle = libxsmm_shuffle((unsigned int)size);
#endif
  /* allocate A, B, and C matrix buffers */
  void *const va = malloc(sizeof(TYPE) * na * size + PAD - 1);
  void *const vb = malloc(sizeof(TYPE) * nb * size + PAD - 1);
  void *const vc = malloc(sizeof(TYPE) * nc * size + PAD - 1);
  /* align memory according to PAD */
  TYPE *const a = (TYPE*)(((uintptr_t)va + PAD - 1) & ~(PAD - 1));
  TYPE *const b = (TYPE*)(((uintptr_t)vb + PAD - 1) & ~(PAD - 1));
  TYPE *const c = (TYPE*)(((uintptr_t)vc + PAD - 1) & ~(PAD - 1));
  const double scale = 1.0 / size;
  double duration = 0;
  int i;

#if defined(mkl_jit_create_sgemm) && defined(mkl_jit_create_dgemm)
  void* jitter;
  CONCATENATE(GEMM, _jit_kernel_t) kernel = NULL;
  if (MKL_JIT_SUCCESS == CONCATENATE(mkl_cblas_jit_create_, GEMM)(&jitter, MKL_COL_MAJOR,
    ('N' == transa || 'n' == transa) ? MKL_NOTRANS : MKL_TRANS,
    ('N' == transb || 'n' == transb) ? MKL_NOTRANS : MKL_TRANS,
    m, n, k, alpha, lda, ldb, beta, ldc))
  { /* explicitly dispatch a kernel according to parameters */
    kernel = CONCATENATE(CONCATENATE(mkl_jit_get_, GEMM), _ptr)(jitter);
  }
  else jitter = NULL;
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
    if (0 != beta) { /* no need to initialize for beta=0 */
      init(42 + i, c + j * nc, m, n, ldc, scale);
    }
  }

#if defined(mkl_jit_create_sgemm) && defined(mkl_jit_create_dgemm)
  if (NULL != jitter) {
# if defined(_OPENMP)
#   pragma omp parallel
# endif
    { /* OpenMP thread pool is already populated (parallel region) */
# if defined(_OPENMP)
#     pragma omp single
# endif
      duration = seconds();
# if defined(_OPENMP)
#     pragma omp for private(i)
# endif
      for (i = 0; i < size; ++i) {
#if defined(SHUFFLE)
        const int j = (i * shuffle) % size;
#else
        const int j = i;
#endif
        kernel(jitter, a + STREAM_A(j * na), b + STREAM_B(j * nb), c + STREAM_C(j * nc));
      }
    }
    duration = seconds() - duration;
  }
  else
# if defined(NOFALLBACK)
  if (0/*false*/)
# endif
#endif
  {
#if defined(_OPENMP)
#   pragma omp parallel
#endif
    { /* OpenMP thread pool is already populated (parallel region) */
#if defined(_OPENMP)
#     pragma omp single
#endif
      duration = seconds();
#if defined(_OPENMP)
#     pragma omp for private(i)
#endif
      for (i = 0; i < size; ++i) {
#if defined(SHUFFLE)
        const int j = (i * shuffle) % size;
#else
        const int j = i;
#endif
        GEMM(&transa, &transb, &m, &n, &k,
          &alpha, a + STREAM_A(j * na), &lda, b + STREAM_B(j * nb), &ldb,
           &beta, c + STREAM_C(j * nc), &ldc);
      }
    }
    duration = seconds() - duration;
  }

  if (0 < duration) {
    const double gflops = 2.0 * m * n * k * 1E-9;
    printf("%.1f GFLOPS/s\n", gflops / duration * size);
  }
  printf("%.1f ms\n", 1000.0 * duration);

  { /* calculate checksum */
    double check = 0;
    for (i = 0; i < size; ++i) {
      const double cn = norm(c + STREAM_C(i * nc), m, n, ldc);
      if (check < cn) check = cn;
    }
    printf("\n%f (check)\n", check);
  }
#if defined(mkl_jit_create_sgemm) && defined(mkl_jit_create_dgemm)
  mkl_jit_destroy(jitter);
#endif
  free(va);
  free(vb);
  free(vc);

  return EXIT_SUCCESS;
}

