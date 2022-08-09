/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Hans Pabst (Intel Corp.)
******************************************************************************/
#include <libxsmm.h>

#if !defined(TYPE)
# define TYPE double
#endif

#if !defined(GEMM)
# if defined(__MKL) || defined(MKL_DIRECT_CALL_SEQ) || defined(MKL_DIRECT_CALL)
#   include <mkl.h>
#   if defined(LIBXSMM_MKL_VERSION3) && (LIBXSMM_VERSION3(11, 3, 0) <= LIBXSMM_MKL_VERSION3)
#     define GEMM_BATCH LIBXSMM_TPREFIX(TYPE, gemm_batch)
#   endif
# else
LIBXSMM_BLAS_SYMBOL_DECL(TYPE, gemm)
# endif
# define GEMM LIBXSMM_GEMM_SYMBOL(TYPE)
#endif

#if defined(_DEBUG)
# define FPRINTF(STREAM, ...) do { fprintf(STREAM, __VA_ARGS__); } while(0)
#else
# define FPRINTF(STREAM, ...) do {} while(0)
#endif

#if defined(_OPENMP)
# define USEOMP(FUNCTION) LIBXSMM_USEOMP(FUNCTION)
#else
# define USEOMP(FUNCTION) (FUNCTION)
#endif

#define EPSILON(T) LIBXSMM_CONCATENATE(EPSILON_, T)
#define EPSILON_double 1e-8
#define EPSILON_float 1e-4


int main(int argc, char* argv[])
{
  /* batch-size is used to stream matrix-operands from memory */
  const libxsmm_blasint batchsize = (1 < argc ? atoi(argv[1]) : 0/*auto*/);
  /* default: M, N, and K are 13, 5, and 7 respectively */
  const libxsmm_blasint m = (2 < argc ? atoi(argv[2]) : 13);
  const libxsmm_blasint n = (3 < argc ? atoi(argv[3]) : 5);
  const libxsmm_blasint k = (4 < argc ? atoi(argv[4]) : 7);
  /* leading dimensions are made multiples of the size of a cache-line */
  const libxsmm_blasint lda = (5 < argc ? LIBXSMM_MAX(atoi(argv[5]), m) : m);
  const libxsmm_blasint ldb = (6 < argc ? LIBXSMM_MAX(atoi(argv[6]), k) : k);
  const libxsmm_blasint ldc = (7 < argc ? LIBXSMM_MAX(atoi(argv[7]), m) : LIBXSMM_UP(m, 8));
  const libxsmm_blasint dup = (8 < argc ? LIBXSMM_CLMP(atoi(argv[8]), 0, 100) : 33);
  const libxsmm_datatype iprec = LIBXSMM_DATATYPE(TYPE), oprec = LIBXSMM_DATATYPE(TYPE);
  /* micro-kernels are limited to certain alpha- and beta-values */
  const char transa = 'n', transb = 'n';
  const TYPE alpha = 1, beta = 1;
  /* calculate matrix sizes incl. padded elements */
  const libxsmm_blasint na = lda * k, nb = ldb * n, nc = ldc * n;
  /* calculate default batch-size to hit work-set size of approx. 2 GB */
  const libxsmm_blasint size = (0 >= batchsize
    ? (libxsmm_blasint)((2ULL << 30/*2 GB*/) / (sizeof(TYPE) * ((size_t)na + nb + nc)))
    : batchsize);
  const size_t shuffle = libxsmm_shuffle((unsigned int)size);
  /* allocate A, B, C, and D/Gold matrix buffers */
  TYPE *const a = (TYPE*)libxsmm_aligned_malloc(sizeof(TYPE) * na * size, LIBXSMM_CACHELINE);
  TYPE *const b = (TYPE*)libxsmm_aligned_malloc(sizeof(TYPE) * nb * size, LIBXSMM_CACHELINE);
  TYPE *const c = (TYPE*)libxsmm_aligned_malloc(sizeof(TYPE) * nc * size, LIBXSMM_CACHELINE);
  TYPE *const d = (TYPE*)libxsmm_aligned_malloc(sizeof(TYPE) * nc * size, LIBXSMM_CACHELINE);
  libxsmm_blasint *const ia = (libxsmm_blasint*)libxsmm_malloc(sizeof(libxsmm_blasint) * size);
  libxsmm_blasint *const ib = (libxsmm_blasint*)libxsmm_malloc(sizeof(libxsmm_blasint) * size);
  libxsmm_blasint *const ic = (libxsmm_blasint*)libxsmm_malloc(sizeof(libxsmm_blasint) * size);
  libxsmm_blasint* const id = (libxsmm_blasint*)(0 < dup ? libxsmm_aligned_scratch(
    sizeof(libxsmm_blasint) * size, 0/*auto*/) : NULL);
  int result = EXIT_SUCCESS;

  if (NULL != a && NULL != b && NULL != c && NULL != d && NULL != ia && NULL != ib && NULL != ic) {
    const double scale = 1.0 / size;
    libxsmm_matdiff_info diff, di;
    libxsmm_blasint i;
    libxsmm_matdiff_clear(&diff);

    /* initialize data according to touch-first policy */
#if defined(_OPENMP)
#   pragma omp parallel for private(i)
#endif
    for (i = 0; i < size; ++i) {
      const libxsmm_blasint j = (i * shuffle) % size;
      const libxsmm_blasint ja = j * na, jb = j * nb, jc = j * nc;
      LIBXSMM_MATINIT(TYPE, 25 + i, a + ja, m, k, lda, scale);
      LIBXSMM_MATINIT(TYPE, 75 + i, b + jb, k, n, ldb, scale);
      if (LIBXSMM_NEQ(0, beta)) { /* no need to initialize for beta=0 */
        LIBXSMM_MATINIT(TYPE, 42 + i, c + jc, m, n, ldc, scale);
        LIBXSMM_MATINIT(TYPE, 42 + i, d + jc, m, n, ldc, scale);
      }
      ia[i] = ja; ib[i] = jb; ic[i] = jc;
    }

    if (NULL != id) { /* duplicate indexes (requested percentage) */
      memcpy(id, ic, sizeof(libxsmm_blasint) * size);
      for (i = 0; i < size; ++i) {
        const unsigned int r = libxsmm_rng_u32(100);
        if (r < (unsigned int)dup) { /* duplicate index */
          const libxsmm_blasint s = (libxsmm_blasint)libxsmm_rng_u32(size - 1);
          const libxsmm_blasint j = (s < i ? s : (s + 1));
          ic[i] = id[j];
        }
      }
    }
    else if (0 < dup) result = EXIT_FAILURE;

    if (EXIT_SUCCESS == result) {
      USEOMP(libxsmm_gemm_batch)(iprec, oprec, &transa, &transb, m, n, k,
        &alpha, a, &lda, b, &ldb, &beta, c, &ldc,
        0/*index_base*/, sizeof(libxsmm_blasint)/*index_stride*/,
        ia, ib, ic, size);
      for (i = 0; i < size; ++i) {
        GEMM(&transa, &transb, &m, &n, &k,
          &alpha, a + ia[i], &lda, b + ib[i], &ldb,
          &beta, d + ic[i], &ldc);
      }
      result = libxsmm_matdiff(&di, oprec, m, n, d, c, &ldc, &ldc);
      if (EXIT_SUCCESS == result) {
        FPRINTF(stderr, "Test error (line #%i): %f", __LINE__, di.linf_abs);
        if (EPSILON(TYPE) < libxsmm_matdiff_epsilon(&di)) {
          FPRINTF(stderr, " (%f != %f)\n", di.v_ref, di.v_tst);
          result = EXIT_FAILURE;
        }
        else FPRINTF(stderr, "\n");
      }
    }

    if (EXIT_SUCCESS == result) {
      libxsmm_gemm_batch(iprec, oprec, &transa, &transb, m, n, k,
        &alpha, a, &lda, b, &ldb, &beta, c, &ldc,
        0/*index_base*/, sizeof(libxsmm_blasint)/*index_stride*/,
        ia, ib, ic, size);
      for (i = 0; i < size; ++i) {
        GEMM(&transa, &transb, &m, &n, &k,
          &alpha, a + ia[i], &lda, b + ib[i], &ldb,
          &beta, d + ic[i], &ldc);
      }
      libxsmm_matdiff_reduce(&diff, &di);
      result = libxsmm_matdiff(&di, oprec, m, n, d, c, &ldc, &ldc);
      if (EXIT_SUCCESS == result) {
        FPRINTF(stderr, "Test error (line #%i): %f", __LINE__, di.linf_abs);
        if (EPSILON(TYPE) < libxsmm_matdiff_epsilon(&di)) {
          FPRINTF(stderr, " (%f != %f)\n", di.v_ref, di.v_tst);
          result = EXIT_FAILURE;
        }
        else FPRINTF(stderr, "\n");
      }
    }

#if defined(GEMM_BATCH)
    if (EXIT_SUCCESS == result) {
      const TYPE* *const pa = (const TYPE**)libxsmm_malloc(sizeof(TYPE*) * size);
      const TYPE* *const pb = (const TYPE**)libxsmm_malloc(sizeof(TYPE*) * size);
      TYPE* *const pc = (TYPE**)libxsmm_malloc(sizeof(TYPE*) * size);
      TYPE* *const pd = (TYPE**)libxsmm_malloc(sizeof(TYPE*) * size);
      if (NULL != pa && NULL != pb && NULL != pc) {
        const libxsmm_blasint ptrsize = sizeof(void*);
        const libxsmm_blasint group_count = 1;
        for (i = 0; i < size; ++i) { /* use pointers instead of indexes */
          pa[i] = a + ia[i]; pb[i] = b + ib[i]; pc[i] = c + ic[i]; pd[i] = d + ic[i];
        }
        USEOMP(libxsmm_gemm_batch)(iprec, oprec,
          &transa, &transb, m, n, k, &alpha, pa, &lda, pb, &ldb, &beta, pc, &ldc,
          0/*index_base*/, 0/*index_stride*/, &ptrsize, &ptrsize, &ptrsize, size);
        libxsmm_gemm_batch(iprec, oprec,
          &transa, &transb, m, n, k, &alpha, pa, &lda, pb, &ldb, &beta, pc, &ldc,
          0/*index_base*/, 0/*index_stride*/, &ptrsize, &ptrsize, &ptrsize, size);
        USEOMP(libxsmm_gemm_groups)(iprec, oprec, &transa, &transb, &m, &n, &k,
          &alpha, (const void**)pa, &lda, (const void**)pb, &ldb,
          &beta, (void**)pc, &ldc, &group_count, &size);
        libxsmm_gemm_groups(iprec, oprec, &transa, &transb, &m, &n, &k,
          &alpha, (const void**)pa, &lda, (const void**)pb, &ldb,
          &beta, (void**)pc, &ldc, &group_count, &size);
        for (i = 0; i < 4; ++i) {
          GEMM_BATCH(&transa, &transb, &m, &n, &k,
            &alpha, pa, &lda, pb, &ldb,
            &beta, pd, &ldc, &group_count, &size);
        }
        libxsmm_matdiff_reduce(&diff, &di);
        result = libxsmm_matdiff(&di, oprec, m, n, d, c, &ldc, &ldc);
        if (EXIT_SUCCESS == result) {
          FPRINTF(stderr, "Test error (line #%i): %f", __LINE__, di.linf_abs);
          if (EPSILON(TYPE) < libxsmm_matdiff_epsilon(&di)) {
            FPRINTF(stderr, " (%f != %f)\n", di.v_ref, di.v_tst);
            result = EXIT_FAILURE;
          }
          else FPRINTF(stderr, "\n");
        }
      }
      else result = EXIT_FAILURE;
      libxsmm_free(pa);
      libxsmm_free(pb);
      libxsmm_free(pc);
    }
#endif

    if (EXIT_SUCCESS == result) {
      libxsmm_matdiff_reduce(&diff, &di);
      FPRINTF(stderr, "Total error: %f", diff.linf_abs);
      if (EPSILON(TYPE) < libxsmm_matdiff_epsilon(&diff)) {
        FPRINTF(stderr, " (%f != %f)\n", diff.v_ref, diff.v_tst);
        result = EXIT_FAILURE;
      }
      else FPRINTF(stderr, "\n");
    }
  }
  else result = EXIT_FAILURE;

  libxsmm_free(ia);
  libxsmm_free(ib);
  libxsmm_free(ic);
  libxsmm_free(id);
  libxsmm_free(a);
  libxsmm_free(b);
  libxsmm_free(c);

  return result;
}
