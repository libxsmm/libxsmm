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
#if defined(__MKL) || defined(MKL_DIRECT_CALL_SEQ) || defined(MKL_DIRECT_CALL)
# include <mkl.h>
#else /* prototypes for GEMM */
void dgemm_(const char*, const char*, const int*, const int*, const int*,
  const double*, const double*, const int*, const double*, const int*,
  const double*, double*, const int*);
void sgemm_(const char*, const char*, const int*, const int*, const int*,
  const double*, const double*, const int*, const double*, const int*,
  const double*, double*, const int*);
#endif
#if defined(_OPENMP)
# include <omp.h>
#endif
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>

#if !defined(TYPE)
# define TYPE double
#endif


void init(int seed, TYPE* dst, int nrows, int ncols, int ld, double scale) {
  const double seed1 = scale * seed + scale;
  int i;
  for (i = 0; i < ncols; ++i) {
    int j = 0;
    for (; j < nrows; ++j) {
      const int k = i * ld + j;
      dst[k] = (TYPE)(seed1 / (1.0 + k));
    }
    for (; j < ld; ++j) {
      const int k = i * ld + j;
      dst[k] = (TYPE)(seed);
    }
  }
}


/**
 * Example program that multiplies matrices independently (C += A * B).
 * A and B-matrices are accumulated into C matrices (beta=1).
 * Streaming A, B, C, AB, AC, BC, or ABC are other useful benchmarks
 * However, running a kernel without loading any matrix operand from
 * memory ("cache-hot loop") is not modeling typical applications
 * since no actual work is performed.
 */
int main(int argc, char* argv[])
{
  const int alignment = 64; /* must be power of two */
  /* batch-size is used to stream matrix-operands from memory */
  const int batchsize = (1 < argc ? atoi(argv[1]) : 0/*auto*/);
  /* default: M, N, and K are 13, 5, and 7 respectively */
  const int m = (2 < argc ? atoi(argv[2]) : 13);
  const int n = (3 < argc ? atoi(argv[3]) : 5);
  const int k = (4 < argc ? atoi(argv[4]) : 7);
  /* leading dimensions are made multiples of the size of a cache-line */
  const int lda = ((sizeof(TYPE) * m + alignment - 1) & ~(alignment - 1)) / sizeof(TYPE);
  const int ldb = ((sizeof(TYPE) * k + alignment - 1) & ~(alignment - 1)) / sizeof(TYPE);
  const int ldc = ((sizeof(TYPE) * m + alignment - 1) & ~(alignment - 1)) / sizeof(TYPE);
  /* micro-kernels are limited to certain alpha- and beta-values */
  const char transa = 'n', transb = 'n';
  const TYPE alpha = 1, beta = 1;
  /* calculate matrix sizes incl. padded elements */
  const size_t na = ((sizeof(TYPE) * lda * k + alignment - 1) & ~(alignment - 1)) / sizeof(TYPE);
  const size_t nb = ((sizeof(TYPE) * ldb * n + alignment - 1) & ~(alignment - 1)) / sizeof(TYPE);
  const size_t nc = ((sizeof(TYPE) * ldc * n + alignment - 1) & ~(alignment - 1)) / sizeof(TYPE);
  /* calculate default batch-size to hit work-set size of approx. 2 GB */
  const int size = (0 >= batchsize ? (int)((2ULL << 30/*2 GB*/) / (sizeof(TYPE) * (na + nb + nc))) : batchsize);
  /* allocate A, B, and C matrix buffers */
  void *const va = malloc(sizeof(TYPE) * na * size + alignment - 1);
  void *const vb = malloc(sizeof(TYPE) * nb * size + alignment - 1);
  void *const vc = malloc(sizeof(TYPE) * nc * size + alignment - 1);
  /* align memory according to alignment */
  TYPE *const a = (TYPE*)(((uintptr_t)va + alignment - 1) & ~(alignment - 1));
  TYPE *const b = (TYPE*)(((uintptr_t)vb + alignment - 1) & ~(alignment - 1));
  TYPE *const c = (TYPE*)(((uintptr_t)vc + alignment - 1) & ~(alignment - 1));
  const double scale = 1.0 / size;
  double duration = 0;
  int i;

#if defined(mkl_jit_create_dgemm) /* explicitly dispatch a kernel according to parameters */
  void* jitter;
  dgemm_jit_kernel_t kernel = NULL;
  if (MKL_JIT_SUCCESS == mkl_cblas_jit_create_dgemm(&jitter, MKL_COL_MAJOR,
    ('N' == transa || 'n' == transa) ? MKL_NOTRANS : MKL_TRANS,
    ('N' == transb || 'n' == transb) ? MKL_NOTRANS : MKL_TRANS,
    m, n, k, alpha, lda, ldb, beta, ldc))
  {
    kernel = mkl_jit_get_dgemm_ptr(jitter);
  }
  else jitter = NULL;
#endif

  /* initialize data according to touch-first policy */
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < size; ++i) {
    init(25 + i, a + i * na, m, k, lda, scale);
    init(75 + i, b + i * nb, k, n, ldb, scale);
    init(42 + i, c + i * nc, m, n, ldc, scale);
  }

#if defined(_OPENMP)
# pragma omp parallel
  {
    /* OpenMP thread pool is already populated (parallel region) */
#   pragma omp single
    duration = omp_get_wtime();
#   pragma omp for private(i)
#endif
    for (i = 0; i < size; ++i) {
#if defined(mkl_jit_create_dgemm)
      kernel(jitter, a + i * na, b + i * nb, c + i * nc);
#else
      dgemm_(&transa, &transb, &m, &n, &k,
        &alpha, a + i * na, &lda, b + i * nb, &ldb,
         &beta, c + i * nc, &ldc);
#endif
    }
#if defined(_OPENMP)
  }
  duration = omp_get_wtime() - duration;
#endif
  if (0 < duration) {
    const double gflops = 2.0 * m * n * k * 1E-9;
    printf("%.1f GFLOPS/s\n", gflops / duration * size);
  }
  printf("%.1f ms\n", 1000.0 * duration);

#if defined(mkl_jit_create_dgemm)
  mkl_jit_destroy(jitter);
#endif
  free(va);
  free(vb);
  free(vc);

  return EXIT_SUCCESS;
}

