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
#include <utils/libxsmm_utils.h>
#include <libxsmm.h>

#if !defined(TYPE)
# define TYPE double
#endif
#if !defined(ALPHA)
# define ALPHA 1
#endif
#if !defined(BETA)
# define BETA 1
#endif

#if !defined(GEMM)
# if (defined(__MKL) || defined(MKL_DIRECT_CALL_SEQ) || defined(MKL_DIRECT_CALL)) && \
     (defined(LIBXSMM_PLATFORM_X86))
#   include <mkl.h>
#   if defined(LIBXSMM_MKL_VERSION3) && (LIBXSMM_VERSION3(2020, 0, 2) <= LIBXSMM_MKL_VERSION3)
#     define GEMM_BATCH_STRIDED LIBXSMM_TPREFIX(TYPE, gemm_batch_strided)
#   endif
#   if defined(LIBXSMM_MKL_VERSION2) && (LIBXSMM_VERSION2(11, 3) <= LIBXSMM_MKL_VERSION2)
#     define GEMM_BATCH LIBXSMM_TPREFIX(TYPE, gemm_batch)
#   endif
# else
LIBXSMM_BLAS_SYMBOL_DECL(TYPE, gemm)
# endif
# define GEMM LIBXSMM_GEMM_SYMBOL(TYPE)
#endif

#if !defined(PRINT) && (defined(_DEBUG) || 0)
# define PRINT
#endif
#if defined(PRINT)
# define FPRINTF(STREAM, ...) do { fprintf(STREAM, __VA_ARGS__); } while(0)
#else
# define FPRINTF(STREAM, ...) do {} while(0)
#endif

#if defined(_OPENMP) && 1
# define USEOMP(FUNCTION) LIBXSMM_USEOMP(FUNCTION)
#else
# define USEOMP(FUNCTION) (FUNCTION)
#endif

#define MALLOC libxsmm_malloc
#define FREE libxsmm_free

#define EPSILON(T) LIBXSMM_CONCATENATE(EPSILON_, T)
#define EPSILON_double 1e-12
#define EPSILON_float 1e-6


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
  const TYPE alpha = ALPHA, beta = BETA;
  const char transa = 'n', transb = 'n';
  /* calculate matrix sizes incl. padded elements */
  const libxsmm_blasint na = lda * k, nb = ldb * n, nc = ldc * n;
  /* calculate default batch-size to hit work-set size of approx. 2 GB */
  const libxsmm_blasint size = (0 >= batchsize
    ? (libxsmm_blasint)(((0 == batchsize ? 512ULL : (unsigned long long)(-batchsize)) << 20/*512 MB*/) /
      (sizeof(TYPE) * ((size_t)na + nb + nc)))
    : batchsize);
  /* process batch of A, B, and C in "random" order */
  const size_t shuffle = (0 != (9 < argc ? atoi(argv[9]) : 1)
    ? libxsmm_coprime2(size)
    : 0);
  /* allocate A, B, C, and D/Gold matrix buffers */
  TYPE *const a = (TYPE*)MALLOC(sizeof(TYPE) * na * size);
  TYPE *const b = (TYPE*)MALLOC(sizeof(TYPE) * nb * size);
  TYPE *const c = (TYPE*)MALLOC(sizeof(TYPE) * nc * size);
  TYPE *const d = (TYPE*)MALLOC(sizeof(TYPE) * nc * size);
  libxsmm_blasint *const psize = (libxsmm_blasint*)MALLOC(sizeof(libxsmm_blasint) * size);
  libxsmm_blasint *const plda = (libxsmm_blasint*)MALLOC(sizeof(libxsmm_blasint) * size);
  libxsmm_blasint *const pldb = (libxsmm_blasint*)MALLOC(sizeof(libxsmm_blasint) * size);
  libxsmm_blasint *const pldc = (libxsmm_blasint*)MALLOC(sizeof(libxsmm_blasint) * size);
  libxsmm_blasint *const pm = (libxsmm_blasint*)MALLOC(sizeof(libxsmm_blasint) * size);
  libxsmm_blasint *const pn = (libxsmm_blasint*)MALLOC(sizeof(libxsmm_blasint) * size);
  libxsmm_blasint *const pk = (libxsmm_blasint*)MALLOC(sizeof(libxsmm_blasint) * size);
  libxsmm_blasint *const ia = (libxsmm_blasint*)MALLOC(sizeof(libxsmm_blasint) * size);
  libxsmm_blasint *const ib = (libxsmm_blasint*)MALLOC(sizeof(libxsmm_blasint) * size);
  libxsmm_blasint *const ic = (libxsmm_blasint*)MALLOC(sizeof(libxsmm_blasint) * size);
  char* const ta = (char*)MALLOC(size), * const tb = (char*)MALLOC(size);
  const TYPE** const pa = (const TYPE**)MALLOC(sizeof(TYPE*) * size);
  const TYPE** const pb = (const TYPE**)MALLOC(sizeof(TYPE*) * size);
  TYPE** const pc = (TYPE**)MALLOC(sizeof(TYPE*) * size);
  TYPE** const pd = (TYPE**)MALLOC(sizeof(TYPE*) * size);
  TYPE* const palpha = (TYPE*)MALLOC(sizeof(TYPE) * size);
  TYPE* const pbeta = (TYPE*)MALLOC(sizeof(TYPE) * size);
  int result = EXIT_SUCCESS;

  if (NULL != a && NULL != b && NULL != c && NULL != d
    && NULL != palpha && NULL != pbeta && NULL != psize
    && NULL != pa && NULL != pb && NULL != pc && NULL != pd
    && NULL != plda && NULL != pldb && NULL != pldc
    && NULL != pm && NULL != pn && NULL != pk
    && NULL != ia && NULL != ib && NULL != ic
    && NULL != ta && NULL != tb)
  {
    const double scale = 1.0 / size;
    libxsmm_matdiff_info diff, di;
    libxsmm_blasint i = 0;
#if defined(PRINT)
    libxsmm_timer_tickint start;
    double d1, d2;
#endif
    libxsmm_matdiff_clear(&diff);

    /* initialize data according to touch-first policy */
#if defined(_OPENMP)
#   pragma omp parallel for private(i)
#endif
    for (i = 0; i < size; ++i) {
      const libxsmm_blasint j = (0 != shuffle ? (libxsmm_blasint)((i * shuffle) % size) : i);
      ia[i] = j * na; ib[i] = j * nb; ic[i] = j * nc;
      LIBXSMM_MATINIT(TYPE, 25 + i, &a[i*na], m, k, lda, scale);
      LIBXSMM_MATINIT(TYPE, 75 + i, &b[i*nb], k, n, ldb, scale);
      if (LIBXSMM_NEQ(0, beta)) { /* no need to initialize for beta=0 */
        LIBXSMM_MATINIT(TYPE, 42 + i, &c[i*nc], m, n, ldc, scale);
        LIBXSMM_MATINIT(TYPE, 42 + i, &d[i*nc], m, n, ldc, scale);
      }
    }

    if (0 < dup) { /* duplicate indexes (requested percentage) */
      for (i = 0; i < size; ++i) {
        const unsigned int r = libxsmm_rng_u32(100);
        if (r < (unsigned int)dup) { /* duplicate index */
          ic[i] = ic[libxsmm_rng_u32(size - 1)];
        }
      }
    }

    if (EXIT_SUCCESS == result) {
#if defined(PRINT)
      start = libxsmm_timer_tick();
#endif
      USEOMP(libxsmm_gemm_batch)(iprec, oprec, &transa, &transb, m, n, k,
        &alpha, a, &lda, ia, b, &ldb, ib, &beta, c, &ldc, ic,
        sizeof(libxsmm_blasint)/*index_stride*/, 0/*index_base*/,
        size, 0/*batchcheck*/);
#if defined(PRINT)
      d1 = libxsmm_timer_duration(start, libxsmm_timer_tick());
      start = libxsmm_timer_tick();
#endif
      for (i = 0; i < size; ++i) {
        GEMM(&transa, &transb, &m, &n, &k,
          &alpha, a + ia[i], &lda, b + ib[i], &ldb,
          &beta, d + ic[i], &ldc);
      }
#if defined(PRINT)
      d2 = libxsmm_timer_duration(start, libxsmm_timer_tick());
#endif
      result = libxsmm_matdiff(&di, oprec, m, n, d, c, &ldc, &ldc);
      if (EXIT_SUCCESS == result) {
        FPRINTF(stderr, "Line #%04i error=%f", __LINE__, di.linf_abs);
        if (EPSILON(TYPE) >= libxsmm_matdiff_epsilon(&di)) {
          FPRINTF(stderr, " (%.0f vs %.0f ms)\n", d1 * 1E3, d2 * 1E3);
        }
        else {
          FPRINTF(stderr, " (%f != %f)\n", di.v_ref, di.v_tst);
          result = EXIT_FAILURE;
        }
      }
    }

    if (EXIT_SUCCESS == result) {
#if defined(PRINT)
      start = libxsmm_timer_tick();
#endif
      USEOMP(libxsmm_gemm_batch)(iprec, oprec, &transa, &transb, m, n, k,
        &alpha, a, &lda, ia, b, &ldb, ib, &beta, c, &ldc, ic,
        sizeof(libxsmm_blasint)/*index_stride*/, 0/*index_base*/,
        size, 1/*batchcheck*/);
#if defined(PRINT)
      d1 = libxsmm_timer_duration(start, libxsmm_timer_tick());
      start = libxsmm_timer_tick();
#endif
      for (i = 0; i < size; ++i) {
        GEMM(&transa, &transb, &m, &n, &k,
          &alpha, a + ia[i], &lda, b + ib[i], &ldb,
          &beta, d + ic[i], &ldc);
      }
#if defined(PRINT)
      d2 = libxsmm_timer_duration(start, libxsmm_timer_tick());
#endif
      result = libxsmm_matdiff(&di, oprec, m, n, d, c, &ldc, &ldc);
      if (EXIT_SUCCESS == result) {
        FPRINTF(stderr, "Line #%04i error=%f", __LINE__, di.linf_abs);
        if (EPSILON(TYPE) >= libxsmm_matdiff_epsilon(&di)) {
          FPRINTF(stderr, " (%.0f vs %.0f ms)\n", d1 * 1E3, d2 * 1E3);
        }
        else {
          FPRINTF(stderr, " (%f != %f)\n", di.v_ref, di.v_tst);
          result = EXIT_FAILURE;
        }
      }
    }

    if (EXIT_SUCCESS == result) {
#if defined(PRINT)
      start = libxsmm_timer_tick();
#endif
#if defined(_OPENMP) /* external parallelization */
#     pragma omp parallel
#     pragma omp single nowait
#endif
      USEOMP(libxsmm_gemm_batch)(iprec, oprec, &transa, &transb, m, n, k,
        &alpha, a, &lda, ia, b, &ldb, ib, &beta, c, &ldc, ic,
        sizeof(libxsmm_blasint)/*index_stride*/, 0/*index_base*/,
        size, 0/*batchcheck*/);
#if defined(PRINT)
      d1 = libxsmm_timer_duration(start, libxsmm_timer_tick());
      start = libxsmm_timer_tick();
#endif
      for (i = 0; i < size; ++i) {
        GEMM(&transa, &transb, &m, &n, &k,
          &alpha, a + ia[i], &lda, b + ib[i], &ldb,
          &beta, d + ic[i], &ldc);
      }
#if defined(PRINT)
      d2 = libxsmm_timer_duration(start, libxsmm_timer_tick());
#endif
      result = libxsmm_matdiff(&di, oprec, m, n, d, c, &ldc, &ldc);
      if (EXIT_SUCCESS == result) {
        FPRINTF(stderr, "Line #%04i error=%f", __LINE__, di.linf_abs);
        if (EPSILON(TYPE) >= libxsmm_matdiff_epsilon(&di)) {
          FPRINTF(stderr, " (%.0f vs %.0f ms)\n", d1 * 1E3, d2 * 1E3);
        }
        else {
          FPRINTF(stderr, " (%f != %f)\n", di.v_ref, di.v_tst);
          result = EXIT_FAILURE;
        }
      }
    }

    if (EXIT_SUCCESS == result) {
#if defined(PRINT)
      start = libxsmm_timer_tick();
#endif
#if defined(_OPENMP) /* external parallelization */
#     pragma omp parallel
#     pragma omp single nowait
#endif
      USEOMP(libxsmm_gemm_batch)(iprec, oprec, &transa, &transb, m, n, k,
        &alpha, a, &lda, ia, b, &ldb, ib, &beta, c, &ldc, ic,
        sizeof(libxsmm_blasint)/*index_stride*/, 0/*index_base*/,
        size, 1/*batchcheck*/);
#if defined(PRINT)
      d1 = libxsmm_timer_duration(start, libxsmm_timer_tick());
      start = libxsmm_timer_tick();
#endif
      for (i = 0; i < size; ++i) {
        GEMM(&transa, &transb, &m, &n, &k,
          &alpha, a + ia[i], &lda, b + ib[i], &ldb,
          &beta, d + ic[i], &ldc);
      }
#if defined(PRINT)
      d2 = libxsmm_timer_duration(start, libxsmm_timer_tick());
#endif
      result = libxsmm_matdiff(&di, oprec, m, n, d, c, &ldc, &ldc);
      if (EXIT_SUCCESS == result) {
        FPRINTF(stderr, "Line #%04i error=%f", __LINE__, di.linf_abs);
        if (EPSILON(TYPE) >= libxsmm_matdiff_epsilon(&di)) {
          FPRINTF(stderr, " (%.0f vs %.0f ms)\n", d1 * 1E3, d2 * 1E3);
        }
        else {
          FPRINTF(stderr, " (%f != %f)\n", di.v_ref, di.v_tst);
          result = EXIT_FAILURE;
        }
      }
    }

    if (EXIT_SUCCESS == result) { /* sequential */
#if defined(PRINT)
      start = libxsmm_timer_tick();
#endif
      libxsmm_gemm_batch(iprec, oprec, &transa, &transb, m, n, k,
        &alpha, a, &lda, ia, b, &ldb, ib, &beta, c, &ldc, ic,
        sizeof(libxsmm_blasint)/*index_stride*/, 0/*index_base*/,
        size, 0/*batchcheck*/);
#if defined(PRINT)
      d1 = libxsmm_timer_duration(start, libxsmm_timer_tick());
      start = libxsmm_timer_tick();
#endif
      for (i = 0; i < size; ++i) {
        GEMM(&transa, &transb, &m, &n, &k,
          &alpha, a + ia[i], &lda, b + ib[i], &ldb,
          &beta, d + ic[i], &ldc);
      }
#if defined(PRINT)
      d2 = libxsmm_timer_duration(start, libxsmm_timer_tick());
#endif
      libxsmm_matdiff_reduce(&diff, &di);
      result = libxsmm_matdiff(&di, oprec, m, n, d, c, &ldc, &ldc);
      if (EXIT_SUCCESS == result) {
        FPRINTF(stderr, "Line #%04i error=%f", __LINE__, di.linf_abs);
        if (EPSILON(TYPE) >= libxsmm_matdiff_epsilon(&di)) {
          FPRINTF(stderr, " (%.0f vs %.0f ms)\n", d1 * 1E3, d2 * 1E3);
        }
        else {
          FPRINTF(stderr, " (%f != %f)\n", di.v_ref, di.v_tst);
          result = EXIT_FAILURE;
        }
      }
    }

    if (EXIT_SUCCESS == result) { /* sequential */
#if defined(PRINT)
      start = libxsmm_timer_tick();
#endif
      libxsmm_gemm_batch(iprec, oprec, &transa, &transb, m, n, k,
        &alpha, a, &lda, ia, b, &ldb, ib, &beta, c, &ldc, ic,
        sizeof(libxsmm_blasint)/*index_stride*/, 0/*index_base*/,
        size, 1/*batchcheck*/);
#if defined(PRINT)
      d1 = libxsmm_timer_duration(start, libxsmm_timer_tick());
      start = libxsmm_timer_tick();
#endif
      for (i = 0; i < size; ++i) {
        GEMM(&transa, &transb, &m, &n, &k,
          &alpha, a + ia[i], &lda, b + ib[i], &ldb,
          &beta, d + ic[i], &ldc);
      }
#if defined(PRINT)
      d2 = libxsmm_timer_duration(start, libxsmm_timer_tick());
#endif
      libxsmm_matdiff_reduce(&diff, &di);
      result = libxsmm_matdiff(&di, oprec, m, n, d, c, &ldc, &ldc);
      if (EXIT_SUCCESS == result) {
        FPRINTF(stderr, "Line #%04i error=%f", __LINE__, di.linf_abs);
        if (EPSILON(TYPE) >= libxsmm_matdiff_epsilon(&di)) {
          FPRINTF(stderr, " (%.0f vs %.0f ms)\n", d1 * 1E3, d2 * 1E3);
        }
        else {
          FPRINTF(stderr, " (%f != %f)\n", di.v_ref, di.v_tst);
          result = EXIT_FAILURE;
        }
      }
    }

    if (EXIT_SUCCESS == result) {
      const libxsmm_blasint ptrsize = sizeof(void*);
      for (i = 0; i < size; ++i) { /* use pointers instead of indexes */
        pa[i] = a + ia[i]; pb[i] = b + ib[i]; pc[i] = c + ic[i]; pd[i] = d + ic[i];
      }
#if defined(PRINT)
      start = libxsmm_timer_tick();
#endif
      USEOMP(libxsmm_gemm_batch)(iprec, oprec, &transa, &transb, m, n, k,
        &alpha, pa, &lda, &ptrsize, pb, &ldb, &ptrsize,
        &beta, pc, &ldc, &ptrsize, 0/*index_stride*/, 0/*index_base*/,
        size, 0/*batchcheck*/);
#if defined(PRINT)
      d1 = libxsmm_timer_duration(start, libxsmm_timer_tick());
      start = libxsmm_timer_tick();
#endif
      for (i = 0; i < size; ++i) {
        GEMM(&transa, &transb, &m, &n, &k,
          &alpha, pa[i], &lda, pb[i], &ldb,
          &beta, pd[i], &ldc);
      }
#if defined(PRINT)
      d2 = libxsmm_timer_duration(start, libxsmm_timer_tick());
#endif
      libxsmm_matdiff_reduce(&diff, &di);
      result = libxsmm_matdiff(&di, oprec, m, n, d, c, &ldc, &ldc);
      if (EXIT_SUCCESS == result) {
        FPRINTF(stderr, "Line #%04i error=%f", __LINE__, di.linf_abs);
        if (EPSILON(TYPE) >= libxsmm_matdiff_epsilon(&di)) {
          FPRINTF(stderr, " (%.0f vs %.0f ms)\n", d1 * 1E3, d2 * 1E3);
        }
        else {
          FPRINTF(stderr, " (%f != %f)\n", di.v_ref, di.v_tst);
          result = EXIT_FAILURE;
        }
      }
    }

    if (EXIT_SUCCESS == result) {
      const libxsmm_blasint ptrsize = sizeof(void*);
#if defined(PRINT)
      start = libxsmm_timer_tick();
#endif
      USEOMP(libxsmm_gemm_batch)(iprec, oprec, &transa, &transb, m, n, k,
        &alpha, pa, &lda, &ptrsize, pb, &ldb, &ptrsize,
        &beta, pc, &ldc, &ptrsize, 0/*index_stride*/, 0/*index_base*/,
        size, 1/*batchcheck*/);
#if defined(PRINT)
      d1 = libxsmm_timer_duration(start, libxsmm_timer_tick());
      start = libxsmm_timer_tick();
#endif
      for (i = 0; i < size; ++i) {
        GEMM(&transa, &transb, &m, &n, &k,
          &alpha, pa[i], &lda, pb[i], &ldb,
          &beta, pd[i], &ldc);
      }
#if defined(PRINT)
      d2 = libxsmm_timer_duration(start, libxsmm_timer_tick());
#endif
      libxsmm_matdiff_reduce(&diff, &di);
      result = libxsmm_matdiff(&di, oprec, m, n, d, c, &ldc, &ldc);
      if (EXIT_SUCCESS == result) {
        FPRINTF(stderr, "Line #%04i error=%f", __LINE__, di.linf_abs);
        if (EPSILON(TYPE) >= libxsmm_matdiff_epsilon(&di)) {
          FPRINTF(stderr, " (%.0f vs %.0f ms)\n", d1 * 1E3, d2 * 1E3);
        }
        else {
          FPRINTF(stderr, " (%f != %f)\n", di.v_ref, di.v_tst);
          result = EXIT_FAILURE;
        }
      }
    }

    if (EXIT_SUCCESS == result) {
      const libxsmm_blasint ptrsize = sizeof(void*);
#if defined(PRINT)
      start = libxsmm_timer_tick();
#endif
#if defined(_OPENMP) /* external parallelization */
#     pragma omp parallel
#     pragma omp single nowait
#endif
      USEOMP(libxsmm_gemm_batch)(iprec, oprec, &transa, &transb, m, n, k,
        &alpha, pa, &lda, &ptrsize, pb, &ldb, &ptrsize,
        &beta, pc, &ldc, &ptrsize, 0/*index_stride*/, 0/*index_base*/,
        size, 0/*batchcheck*/);
#if defined(PRINT)
      d1 = libxsmm_timer_duration(start, libxsmm_timer_tick());
      start = libxsmm_timer_tick();
#endif
      for (i = 0; i < size; ++i) {
        GEMM(&transa, &transb, &m, &n, &k,
          &alpha, pa[i], &lda, pb[i], &ldb,
          &beta, pd[i], &ldc);
      }
#if defined(PRINT)
      d2 = libxsmm_timer_duration(start, libxsmm_timer_tick());
#endif
      libxsmm_matdiff_reduce(&diff, &di);
      result = libxsmm_matdiff(&di, oprec, m, n, d, c, &ldc, &ldc);
      if (EXIT_SUCCESS == result) {
        FPRINTF(stderr, "Line #%04i error=%f", __LINE__, di.linf_abs);
        if (EPSILON(TYPE) >= libxsmm_matdiff_epsilon(&di)) {
          FPRINTF(stderr, " (%.0f vs %.0f ms)\n", d1 * 1E3, d2 * 1E3);
        }
        else {
          FPRINTF(stderr, " (%f != %f)\n", di.v_ref, di.v_tst);
          result = EXIT_FAILURE;
        }
      }
    }

    if (EXIT_SUCCESS == result) {
      const libxsmm_blasint ptrsize = sizeof(void*);
#if defined(PRINT)
      start = libxsmm_timer_tick();
#endif
#if defined(_OPENMP) /* external parallelization */
#     pragma omp parallel
#     pragma omp single nowait
#endif
      USEOMP(libxsmm_gemm_batch)(iprec, oprec, &transa, &transb, m, n, k,
        &alpha, pa, &lda, &ptrsize, pb, &ldb, &ptrsize,
        &beta, pc, &ldc, &ptrsize, 0/*index_stride*/, 0/*index_base*/,
        size, 1/*batchcheck*/);
#if defined(PRINT)
      d1 = libxsmm_timer_duration(start, libxsmm_timer_tick());
      start = libxsmm_timer_tick();
#endif
      for (i = 0; i < size; ++i) {
        GEMM(&transa, &transb, &m, &n, &k,
          &alpha, pa[i], &lda, pb[i], &ldb,
          &beta, pd[i], &ldc);
      }
#if defined(PRINT)
      d2 = libxsmm_timer_duration(start, libxsmm_timer_tick());
#endif
      libxsmm_matdiff_reduce(&diff, &di);
      result = libxsmm_matdiff(&di, oprec, m, n, d, c, &ldc, &ldc);
      if (EXIT_SUCCESS == result) {
        FPRINTF(stderr, "Line #%04i error=%f", __LINE__, di.linf_abs);
        if (EPSILON(TYPE) >= libxsmm_matdiff_epsilon(&di)) {
          FPRINTF(stderr, " (%.0f vs %.0f ms)\n", d1 * 1E3, d2 * 1E3);
        }
        else {
          FPRINTF(stderr, " (%f != %f)\n", di.v_ref, di.v_tst);
          result = EXIT_FAILURE;
        }
      }
    }

    if (EXIT_SUCCESS == result) { /* sequential */
      const libxsmm_blasint ptrsize = sizeof(void*);
#if defined(PRINT)
      start = libxsmm_timer_tick();
#endif
      libxsmm_gemm_batch(iprec, oprec, &transa, &transb, m, n, k,
        &alpha, pa, &lda, &ptrsize, pb, &ldb, &ptrsize,
        &beta, pc, &ldc, &ptrsize, 0/*index_stride*/, 0/*index_base*/,
        size, 0/*batchcheck*/);
#if defined(PRINT)
      d1 = libxsmm_timer_duration(start, libxsmm_timer_tick());
      start = libxsmm_timer_tick();
#endif
      for (i = 0; i < size; ++i) {
        GEMM(&transa, &transb, &m, &n, &k,
          &alpha, pa[i], &lda, pb[i], &ldb,
          &beta, pd[i], &ldc);
      }
#if defined(PRINT)
      d2 = libxsmm_timer_duration(start, libxsmm_timer_tick());
#endif
      libxsmm_matdiff_reduce(&diff, &di);
      result = libxsmm_matdiff(&di, oprec, m, n, d, c, &ldc, &ldc);
      if (EXIT_SUCCESS == result) {
        FPRINTF(stderr, "Line #%04i error=%f", __LINE__, di.linf_abs);
        if (EPSILON(TYPE) >= libxsmm_matdiff_epsilon(&di)) {
          FPRINTF(stderr, " (%.0f vs %.0f ms)\n", d1 * 1E3, d2 * 1E3);
        }
        else {
          FPRINTF(stderr, " (%f != %f)\n", di.v_ref, di.v_tst);
          result = EXIT_FAILURE;
        }
      }
    }

    if (EXIT_SUCCESS == result) { /* sequential */
      const libxsmm_blasint ptrsize = sizeof(void*);
#if defined(PRINT)
      start = libxsmm_timer_tick();
#endif
      libxsmm_gemm_batch(iprec, oprec, &transa, &transb, m, n, k,
        &alpha, pa, &lda, &ptrsize, pb, &ldb, &ptrsize,
        &beta, pc, &ldc, &ptrsize, 0/*index_stride*/, 0/*index_base*/,
        size, 1/*batchcheck*/);
#if defined(PRINT)
      d1 = libxsmm_timer_duration(start, libxsmm_timer_tick());
      start = libxsmm_timer_tick();
#endif
      for (i = 0; i < size; ++i) {
        GEMM(&transa, &transb, &m, &n, &k,
          &alpha, pa[i], &lda, pb[i], &ldb,
          &beta, pd[i], &ldc);
      }
#if defined(PRINT)
      d2 = libxsmm_timer_duration(start, libxsmm_timer_tick());
#endif
      libxsmm_matdiff_reduce(&diff, &di);
      result = libxsmm_matdiff(&di, oprec, m, n, d, c, &ldc, &ldc);
      if (EXIT_SUCCESS == result) {
        FPRINTF(stderr, "Line #%04i error=%f", __LINE__, di.linf_abs);
        if (EPSILON(TYPE) >= libxsmm_matdiff_epsilon(&di)) {
          FPRINTF(stderr, " (%.0f vs %.0f ms)\n", d1 * 1E3, d2 * 1E3);
        }
        else {
          FPRINTF(stderr, " (%f != %f)\n", di.v_ref, di.v_tst);
          result = EXIT_FAILURE;
        }
      }
    }

    if (EXIT_SUCCESS == result) {
#if defined(PRINT)
      start = libxsmm_timer_tick();
#endif
      USEOMP(libxsmm_gemm_strided)(iprec, oprec, &transa, &transb, m, n, k,
        &alpha, a, &lda, &na, b, &ldb, &nb, &beta, c, &ldc, &nc,
        0/*index_base*/, size);
#if defined(PRINT)
      d1 = libxsmm_timer_duration(start, libxsmm_timer_tick());
      start = libxsmm_timer_tick();
#endif
      for (i = 0; i < size; ++i) {
        GEMM(&transa, &transb, &m, &n, &k,
          &alpha, &a[i*na], &lda, &b[i*nb], &ldb,
          &beta, &d[i*nc], &ldc);
      }
#if defined(PRINT)
      d2 = libxsmm_timer_duration(start, libxsmm_timer_tick());
#endif
      libxsmm_matdiff_reduce(&diff, &di);
      result = libxsmm_matdiff(&di, oprec, m, n, d, c, &ldc, &ldc);
      if (EXIT_SUCCESS == result) {
        FPRINTF(stderr, "Line #%04i error=%f", __LINE__, di.linf_abs);
        if (EPSILON(TYPE) >= libxsmm_matdiff_epsilon(&di)) {
          FPRINTF(stderr, " (%.0f vs %.0f ms)\n", d1 * 1E3, d2 * 1E3);
        }
        else {
          FPRINTF(stderr, " (%f != %f)\n", di.v_ref, di.v_tst);
          result = EXIT_FAILURE;
        }
      }
    }

    if (EXIT_SUCCESS == result) {
#if defined(PRINT)
      start = libxsmm_timer_tick();
#endif
#if defined(_OPENMP) /* external parallelization */
#     pragma omp parallel
#     pragma omp single nowait
#endif
      USEOMP(libxsmm_gemm_strided)(iprec, oprec, &transa, &transb, m, n, k,
        &alpha, a, &lda, &na, b, &ldb, &nb, &beta, c, &ldc, &nc,
        0/*index_base*/, size);
#if defined(PRINT)
      d1 = libxsmm_timer_duration(start, libxsmm_timer_tick());
      start = libxsmm_timer_tick();
#endif
      for (i = 0; i < size; ++i) {
        GEMM(&transa, &transb, &m, &n, &k,
          &alpha, &a[i*na], &lda, &b[i*nb], &ldb,
          &beta, &d[i*nc], &ldc);
      }
#if defined(PRINT)
      d2 = libxsmm_timer_duration(start, libxsmm_timer_tick());
#endif
      libxsmm_matdiff_reduce(&diff, &di);
      result = libxsmm_matdiff(&di, oprec, m, n, d, c, &ldc, &ldc);
      if (EXIT_SUCCESS == result) {
        FPRINTF(stderr, "Line #%04i error=%f", __LINE__, di.linf_abs);
        if (EPSILON(TYPE) >= libxsmm_matdiff_epsilon(&di)) {
          FPRINTF(stderr, " (%.0f vs %.0f ms)\n", d1 * 1E3, d2 * 1E3);
        }
        else {
          FPRINTF(stderr, " (%f != %f)\n", di.v_ref, di.v_tst);
          result = EXIT_FAILURE;
        }
      }
    }

    if (EXIT_SUCCESS == result) { /* sequential */
#if defined(PRINT)
      start = libxsmm_timer_tick();
#endif
      libxsmm_gemm_strided(iprec, oprec, &transa, &transb, m, n, k,
        &alpha, a, &lda, &na, b, &ldb, &nb, &beta, c, &ldc, &nc,
        0/*index_base*/, size);
#if defined(PRINT)
      d1 = libxsmm_timer_duration(start, libxsmm_timer_tick());
      start = libxsmm_timer_tick();
#endif
      for (i = 0; i < size; ++i) {
        GEMM(&transa, &transb, &m, &n, &k,
          &alpha, &a[i*na], &lda, &b[i*nb], &ldb,
          &beta, &d[i*nc], &ldc);
      }
#if defined(PRINT)
      d2 = libxsmm_timer_duration(start, libxsmm_timer_tick());
#endif
      libxsmm_matdiff_reduce(&diff, &di);
      result = libxsmm_matdiff(&di, oprec, m, n, d, c, &ldc, &ldc);
      if (EXIT_SUCCESS == result) {
        FPRINTF(stderr, "Line #%04i error=%f", __LINE__, di.linf_abs);
        if (EPSILON(TYPE) >= libxsmm_matdiff_epsilon(&di)) {
          FPRINTF(stderr, " (%.0f vs %.0f ms)\n", d1 * 1E3, d2 * 1E3);
        }
        else {
          FPRINTF(stderr, " (%f != %f)\n", di.v_ref, di.v_tst);
          result = EXIT_FAILURE;
        }
      }
    }

#if defined(GEMM_BATCH)
    if (EXIT_SUCCESS == result) {
      const libxsmm_blasint group_count = 1, ptrsize = sizeof(void*);
#if defined(PRINT)
      start = libxsmm_timer_tick();
#endif
      USEOMP(libxsmm_gemm_batch)(iprec, oprec, &transa, &transb, m, n, k,
        &alpha, pa, &lda, &ptrsize, pb, &ldb, &ptrsize,
        &beta, pc, &ldc, &ptrsize, 0/*index_stride*/, 0/*index_base*/,
        size, 0/*batchcheck*/);
#if defined(PRINT)
      d1 = libxsmm_timer_duration(start, libxsmm_timer_tick());
      start = libxsmm_timer_tick();
#endif
      GEMM_BATCH(&transa, &transb, &m, &n, &k,
        &alpha, pa, &lda, pb, &ldb,
        &beta, pd, &ldc, &group_count, &size);
#if defined(PRINT)
      d2 = libxsmm_timer_duration(start, libxsmm_timer_tick());
#endif
      libxsmm_matdiff_reduce(&diff, &di);
      result = libxsmm_matdiff(&di, oprec, m, n, d, c, &ldc, &ldc);
      if (EXIT_SUCCESS == result) {
        FPRINTF(stderr, "Line #%04i error=%f", __LINE__, di.linf_abs);
        if (EPSILON(TYPE) >= libxsmm_matdiff_epsilon(&di)) {
          FPRINTF(stderr, " (%.0f vs %.0f ms)\n", d1 * 1E3, d2 * 1E3);
        }
        else {
          FPRINTF(stderr, " (%f != %f)\n", di.v_ref, di.v_tst);
          result = EXIT_FAILURE;
        }
      }
    }

    if (EXIT_SUCCESS == result) {
      const libxsmm_blasint group_count = 1, ptrsize = sizeof(void*);
#if defined(PRINT)
      start = libxsmm_timer_tick();
#endif
      USEOMP(libxsmm_gemm_batch)(iprec, oprec, &transa, &transb, m, n, k,
        &alpha, pa, &lda, &ptrsize, pb, &ldb, &ptrsize,
        &beta, pc, &ldc, &ptrsize, 0/*index_stride*/, 0/*index_base*/,
        size, 1/*batchcheck*/);
#if defined(PRINT)
      d1 = libxsmm_timer_duration(start, libxsmm_timer_tick());
      start = libxsmm_timer_tick();
#endif
      GEMM_BATCH(&transa, &transb, &m, &n, &k,
        &alpha, pa, &lda, pb, &ldb,
        &beta, pd, &ldc, &group_count, &size);
#if defined(PRINT)
      d2 = libxsmm_timer_duration(start, libxsmm_timer_tick());
#endif
      libxsmm_matdiff_reduce(&diff, &di);
      result = libxsmm_matdiff(&di, oprec, m, n, d, c, &ldc, &ldc);
      if (EXIT_SUCCESS == result) {
        FPRINTF(stderr, "Line #%04i error=%f", __LINE__, di.linf_abs);
        if (EPSILON(TYPE) >= libxsmm_matdiff_epsilon(&di)) {
          FPRINTF(stderr, " (%.0f vs %.0f ms)\n", d1 * 1E3, d2 * 1E3);
        }
        else {
          FPRINTF(stderr, " (%f != %f)\n", di.v_ref, di.v_tst);
          result = EXIT_FAILURE;
        }
      }
    }

    if (EXIT_SUCCESS == result) {
      const libxsmm_blasint group_count = 1;
#if defined(PRINT)
      start = libxsmm_timer_tick();
#endif
      USEOMP(libxsmm_gemm_groups)(iprec, oprec, &transa, &transb, &m, &n, &k,
        &alpha, (const void**)pa, &lda, (const void**)pb, &ldb,
        &beta, (void**)pc, &ldc, group_count,
        &size, 0/*batchcheck*/);
#if defined(PRINT)
      d1 = libxsmm_timer_duration(start, libxsmm_timer_tick());
      start = libxsmm_timer_tick();
#endif
      GEMM_BATCH(&transa, &transb, &m, &n, &k,
        &alpha, pa, &lda, pb, &ldb,
        &beta, pd, &ldc, &group_count, &size);
#if defined(PRINT)
      d2 = libxsmm_timer_duration(start, libxsmm_timer_tick());
#endif
      libxsmm_matdiff_reduce(&diff, &di);
      result = libxsmm_matdiff(&di, oprec, m, n, d, c, &ldc, &ldc);
      if (EXIT_SUCCESS == result) {
        FPRINTF(stderr, "Line #%04i error=%f", __LINE__, di.linf_abs);
        if (EPSILON(TYPE) >= libxsmm_matdiff_epsilon(&di)) {
          FPRINTF(stderr, " (%.0f vs %.0f ms)\n", d1 * 1E3, d2 * 1E3);
        }
        else {
          FPRINTF(stderr, " (%f != %f)\n", di.v_ref, di.v_tst);
          result = EXIT_FAILURE;
        }
      }
    }

    if (EXIT_SUCCESS == result) {
      const libxsmm_blasint group_count = 1;
#if defined(PRINT)
      start = libxsmm_timer_tick();
#endif
      USEOMP(libxsmm_gemm_groups)(iprec, oprec, &transa, &transb, &m, &n, &k,
        &alpha, (const void**)pa, &lda, (const void**)pb, &ldb,
        &beta, (void**)pc, &ldc, group_count,
        &size, 1/*batchcheck*/);
#if defined(PRINT)
      d1 = libxsmm_timer_duration(start, libxsmm_timer_tick());
      start = libxsmm_timer_tick();
#endif
      GEMM_BATCH(&transa, &transb, &m, &n, &k,
        &alpha, pa, &lda, pb, &ldb,
        &beta, pd, &ldc, &group_count, &size);
#if defined(PRINT)
      d2 = libxsmm_timer_duration(start, libxsmm_timer_tick());
#endif
      libxsmm_matdiff_reduce(&diff, &di);
      result = libxsmm_matdiff(&di, oprec, m, n, d, c, &ldc, &ldc);
      if (EXIT_SUCCESS == result) {
        FPRINTF(stderr, "Line #%04i error=%f", __LINE__, di.linf_abs);
        if (EPSILON(TYPE) >= libxsmm_matdiff_epsilon(&di)) {
          FPRINTF(stderr, " (%.0f vs %.0f ms)\n", d1 * 1E3, d2 * 1E3);
        }
        else {
          FPRINTF(stderr, " (%f != %f)\n", di.v_ref, di.v_tst);
          result = EXIT_FAILURE;
        }
      }
    }

    if (EXIT_SUCCESS == result) { /* many groups */
      for (i = 0; i < size; ++i) {
        plda[i] = lda; pldb[i] = ldb; pldc[i] = ldc;
        palpha[i] = alpha; pbeta[i] = beta;
        pm[i] = m; pn[i] = n; pk[i] = k;
        ta[i] = transa; tb[i] = transb;
        psize[i] = 1;
      }
#if defined(PRINT)
      start = libxsmm_timer_tick();
#endif
      USEOMP(libxsmm_gemm_groups)(iprec, oprec, ta, tb, pm, pn, pk,
        palpha, (const void**)pa, plda, (const void**)pb, pldb,
        pbeta, (void**)pc, pldc, size,
        psize, 0/*batchcheck*/);
#if defined(PRINT)
      d1 = libxsmm_timer_duration(start, libxsmm_timer_tick());
      start = libxsmm_timer_tick();
#endif
      GEMM_BATCH(ta, tb, pm, pn, pk,
        palpha, pa, plda, pb, pldb,
        pbeta, pd, pldc, &size, psize);
#if defined(PRINT)
      d2 = libxsmm_timer_duration(start, libxsmm_timer_tick());
#endif
      libxsmm_matdiff_reduce(&diff, &di);
      result = libxsmm_matdiff(&di, oprec, m, n, d, c, &ldc, &ldc);
      if (EXIT_SUCCESS == result) {
        FPRINTF(stderr, "Line #%04i error=%f", __LINE__, di.linf_abs);
        if (EPSILON(TYPE) >= libxsmm_matdiff_epsilon(&di)) {
          FPRINTF(stderr, " (%.0f vs %.0f ms)\n", d1 * 1E3, d2 * 1E3);
        }
        else {
          FPRINTF(stderr, " (%f != %f)\n", di.v_ref, di.v_tst);
          result = EXIT_FAILURE;
        }
      }
    }

    if (EXIT_SUCCESS == result) { /* many groups */
#if defined(PRINT)
      start = libxsmm_timer_tick();
#endif
      USEOMP(libxsmm_gemm_groups)(iprec, oprec, ta, tb, pm, pn, pk,
        palpha, (const void**)pa, plda, (const void**)pb, pldb,
        pbeta, (void**)pc, pldc, size,
        psize, 1/*batchcheck*/);
#if defined(PRINT)
      d1 = libxsmm_timer_duration(start, libxsmm_timer_tick());
      start = libxsmm_timer_tick();
#endif
      GEMM_BATCH(ta, tb, pm, pn, pk,
        palpha, pa, plda, pb, pldb,
        pbeta, pd, pldc, &size, psize);
#if defined(PRINT)
      d2 = libxsmm_timer_duration(start, libxsmm_timer_tick());
#endif
      libxsmm_matdiff_reduce(&diff, &di);
      result = libxsmm_matdiff(&di, oprec, m, n, d, c, &ldc, &ldc);
      if (EXIT_SUCCESS == result) {
        FPRINTF(stderr, "Line #%04i error=%f", __LINE__, di.linf_abs);
        if (EPSILON(TYPE) >= libxsmm_matdiff_epsilon(&di)) {
          FPRINTF(stderr, " (%.0f vs %.0f ms)\n", d1 * 1E3, d2 * 1E3);
        }
        else {
          FPRINTF(stderr, " (%f != %f)\n", di.v_ref, di.v_tst);
          result = EXIT_FAILURE;
        }
      }
    }

    if (EXIT_SUCCESS == result) { /* many groups */
#if defined(PRINT)
      start = libxsmm_timer_tick();
#endif
#if defined(_OPENMP) /* external parallelization */
#     pragma omp parallel
#     pragma omp single nowait
#endif
      USEOMP(libxsmm_gemm_groups)(iprec, oprec, ta, tb, pm, pn, pk,
        palpha, (const void**)pa, plda, (const void**)pb, pldb,
        pbeta, (void**)pc, pldc, size,
        psize, 0/*batchcheck*/);
#if defined(PRINT)
      d1 = libxsmm_timer_duration(start, libxsmm_timer_tick());
      start = libxsmm_timer_tick();
#endif
      GEMM_BATCH(ta, tb, pm, pn, pk,
        palpha, pa, plda, pb, pldb,
        pbeta, pd, pldc, &size, psize);
#if defined(PRINT)
      d2 = libxsmm_timer_duration(start, libxsmm_timer_tick());
#endif
      libxsmm_matdiff_reduce(&diff, &di);
      result = libxsmm_matdiff(&di, oprec, m, n, d, c, &ldc, &ldc);
      if (EXIT_SUCCESS == result) {
        FPRINTF(stderr, "Line #%04i error=%f", __LINE__, di.linf_abs);
        if (EPSILON(TYPE) >= libxsmm_matdiff_epsilon(&di)) {
          FPRINTF(stderr, " (%.0f vs %.0f ms)\n", d1 * 1E3, d2 * 1E3);
        }
        else {
          FPRINTF(stderr, " (%f != %f)\n", di.v_ref, di.v_tst);
          result = EXIT_FAILURE;
        }
      }
    }

    if (EXIT_SUCCESS == result) { /* many groups */
#if defined(PRINT)
      start = libxsmm_timer_tick();
#endif
#if defined(_OPENMP) /* external parallelization */
#     pragma omp parallel
#     pragma omp single nowait
#endif
      USEOMP(libxsmm_gemm_groups)(iprec, oprec, ta, tb, pm, pn, pk,
        palpha, (const void**)pa, plda, (const void**)pb, pldb,
        pbeta, (void**)pc, pldc, size,
        psize, 1/*batchcheck*/);
#if defined(PRINT)
      d1 = libxsmm_timer_duration(start, libxsmm_timer_tick());
      start = libxsmm_timer_tick();
#endif
      GEMM_BATCH(ta, tb, pm, pn, pk,
        palpha, pa, plda, pb, pldb,
        pbeta, pd, pldc, &size, psize);
#if defined(PRINT)
      d2 = libxsmm_timer_duration(start, libxsmm_timer_tick());
#endif
      libxsmm_matdiff_reduce(&diff, &di);
      result = libxsmm_matdiff(&di, oprec, m, n, d, c, &ldc, &ldc);
      if (EXIT_SUCCESS == result) {
        FPRINTF(stderr, "Line #%04i error=%f", __LINE__, di.linf_abs);
        if (EPSILON(TYPE) >= libxsmm_matdiff_epsilon(&di)) {
          FPRINTF(stderr, " (%.0f vs %.0f ms)\n", d1 * 1E3, d2 * 1E3);
        }
        else {
          FPRINTF(stderr, " (%f != %f)\n", di.v_ref, di.v_tst);
          result = EXIT_FAILURE;
        }
      }
    }

    if (EXIT_SUCCESS == result) { /* many groups */
#if defined(PRINT)
      start = libxsmm_timer_tick();
#endif
      libxsmm_gemm_groups(iprec, oprec, ta, tb, pm, pn, pk,
        palpha, (const void**)pa, plda, (const void**)pb, pldb,
        pbeta, (void**)pc, pldc, size,
        psize, 0/*batchcheck*/);
#if defined(PRINT)
      d1 = libxsmm_timer_duration(start, libxsmm_timer_tick());
      start = libxsmm_timer_tick();
#endif
      GEMM_BATCH(ta, tb, pm, pn, pk,
        palpha, pa, plda, pb, pldb,
        pbeta, pd, pldc, &size, psize);
#if defined(PRINT)
      d2 = libxsmm_timer_duration(start, libxsmm_timer_tick());
#endif
      libxsmm_matdiff_reduce(&diff, &di);
      result = libxsmm_matdiff(&di, oprec, m, n, d, c, &ldc, &ldc);
      if (EXIT_SUCCESS == result) {
        FPRINTF(stderr, "Line #%04i error=%f", __LINE__, di.linf_abs);
        if (EPSILON(TYPE) >= libxsmm_matdiff_epsilon(&di)) {
          FPRINTF(stderr, " (%.0f vs %.0f ms)\n", d1 * 1E3, d2 * 1E3);
        }
        else {
          FPRINTF(stderr, " (%f != %f)\n", di.v_ref, di.v_tst);
          result = EXIT_FAILURE;
        }
      }
    }

    if (EXIT_SUCCESS == result) { /* many groups */
#if defined(PRINT)
      start = libxsmm_timer_tick();
#endif
      libxsmm_gemm_groups(iprec, oprec, ta, tb, pm, pn, pk,
        palpha, (const void**)pa, plda, (const void**)pb, pldb,
        pbeta, (void**)pc, pldc, size,
        psize, 1/*batchcheck*/);
#if defined(PRINT)
      d1 = libxsmm_timer_duration(start, libxsmm_timer_tick());
      start = libxsmm_timer_tick();
#endif
      GEMM_BATCH(ta, tb, pm, pn, pk,
        palpha, pa, plda, pb, pldb,
        pbeta, pd, pldc, &size, psize);
#if defined(PRINT)
      d2 = libxsmm_timer_duration(start, libxsmm_timer_tick());
#endif
      libxsmm_matdiff_reduce(&diff, &di);
      result = libxsmm_matdiff(&di, oprec, m, n, d, c, &ldc, &ldc);
      if (EXIT_SUCCESS == result) {
        FPRINTF(stderr, "Line #%04i error=%f", __LINE__, di.linf_abs);
        if (EPSILON(TYPE) >= libxsmm_matdiff_epsilon(&di)) {
          FPRINTF(stderr, " (%.0f vs %.0f ms)\n", d1 * 1E3, d2 * 1E3);
        }
        else {
          FPRINTF(stderr, " (%f != %f)\n", di.v_ref, di.v_tst);
          result = EXIT_FAILURE;
        }
      }
    }
#endif

#if defined(GEMM_BATCH_STRIDED)
    if (EXIT_SUCCESS == result) {
#if defined(PRINT)
      start = libxsmm_timer_tick();
#endif
      USEOMP(libxsmm_gemm_strided)(iprec, oprec, &transa, &transb, m, n, k,
        &alpha, a, &lda, &na, b, &ldb, &nb, &beta, c, &ldc, &nc,
        0/*index_base*/, size);
#if defined(PRINT)
      d1 = libxsmm_timer_duration(start, libxsmm_timer_tick());
      start = libxsmm_timer_tick();
#endif
      GEMM_BATCH_STRIDED(&transa, &transb, &m, &n, &k,
        &alpha, a, &lda, &na, b, &ldb, &nb,
        &beta, d, &ldc, &nc, &size);
#if defined(PRINT)
      d2 = libxsmm_timer_duration(start, libxsmm_timer_tick());
#endif
      libxsmm_matdiff_reduce(&diff, &di);
      result = libxsmm_matdiff(&di, oprec, m, n, d, c, &ldc, &ldc);
      if (EXIT_SUCCESS == result) {
        FPRINTF(stderr, "Line #%04i error=%f", __LINE__, di.linf_abs);
        if (EPSILON(TYPE) >= libxsmm_matdiff_epsilon(&di)) {
          FPRINTF(stderr, " (%.0f vs %.0f ms)\n", d1 * 1E3, d2 * 1E3);
        }
        else {
          FPRINTF(stderr, " (%f != %f)\n", di.v_ref, di.v_tst);
          result = EXIT_FAILURE;
        }
      }
    }
#endif

    if (EXIT_SUCCESS == result) {
      double epsilon;
      libxsmm_matdiff_reduce(&diff, &di);
      epsilon = libxsmm_matdiff_epsilon(&diff);
      FPRINTF(stderr, "Summary    error=%f (%f)", diff.linf_abs, epsilon);
      if (EPSILON(TYPE) >= epsilon) FPRINTF(stderr, "\n");
      else {
        FPRINTF(stderr, " (%f != %f)\n", di.v_ref, di.v_tst);
        result = EXIT_FAILURE;
      }
    }
  }
  else result = EXIT_FAILURE;

  FREE(pm); FREE(pn); FREE(pk); FREE(ta); FREE(tb);
  FREE(pa); FREE(pb); FREE(pc); FREE(pd);
  FREE(palpha); FREE(pbeta); FREE(psize);
  FREE(plda); FREE(pldb); FREE(pldc);
  FREE(ia); FREE(ib); FREE(ic);
  FREE(a); FREE(b); FREE(c);

  return result;
}
