#include <libxsmm_timer.h>
#include <libxsmm_malloc.h>

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#include <stdlib.h>
#include <stdio.h>
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

#if !defined(REAL_TYPE)
# define REAL_TYPE double
#endif


LIBXSMM_RETARGETABLE void init(int seed, REAL_TYPE *LIBXSMM_RESTRICT dst, double scale, libxsmm_blasint nrows, libxsmm_blasint ncols, libxsmm_blasint ld);
LIBXSMM_RETARGETABLE void init(int seed, REAL_TYPE *LIBXSMM_RESTRICT dst, double scale, libxsmm_blasint nrows, libxsmm_blasint ncols, libxsmm_blasint ld)
{
  const libxsmm_blasint minval = seed, addval = (nrows - 1) * ld + (ncols - 1);
  const libxsmm_blasint maxval = LIBXSMM_MAX(LIBXSMM_ABS(minval), addval);
  const double norm = 0 != maxval ? (scale / maxval) : scale;
  libxsmm_blasint i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < ncols; ++i) {
    libxsmm_blasint j;
    for (j = 0; j < nrows; ++j) {
      const libxsmm_blasint k = i * ld + j;
      const double value = (double)(k + minval);
      dst[k] = (REAL_TYPE)(norm * (value - 0.5 * addval));
    }
  }
}


int main(int argc, char* argv[])
{
  const libxsmm_blasint m = LIBXSMM_DEFAULT(512, 1 < argc ? atoi(argv[1]) : 0);
  const libxsmm_blasint n = LIBXSMM_DEFAULT(m, 2 < argc ? atoi(argv[2]) : 0);
  const libxsmm_blasint k = LIBXSMM_DEFAULT(m, 3 < argc ? atoi(argv[3]) : 0);
  const libxsmm_blasint lda = LIBXSMM_DEFAULT(m, 4 < argc ? atoi(argv[4]) : 0);
  const libxsmm_blasint ldb = LIBXSMM_DEFAULT(k, 5 < argc ? atoi(argv[5]) : 0);
  const libxsmm_blasint ldc = LIBXSMM_DEFAULT(m, 6 < argc ? atoi(argv[6]) : 0);
  const int nrepeat = LIBXSMM_DEFAULT(13, 7 < argc ? atoi(argv[7]) : 0);
  const double scale = 1.0 / nrepeat, gflops = 2.0 * m * n * k * 1E-9;
  const char transa = 'N', transb = 'N';
  const REAL_TYPE alpha = 1, beta = 1;

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload target(LIBXSMM_OFFLOAD_TARGET)
#endif
  {
    REAL_TYPE *const a = (REAL_TYPE*)libxsmm_malloc(lda * k * sizeof(REAL_TYPE));
    REAL_TYPE *const b = (REAL_TYPE*)libxsmm_malloc(ldb * n * sizeof(REAL_TYPE));
    REAL_TYPE *const c = (REAL_TYPE*)libxsmm_malloc(ldc * n * sizeof(REAL_TYPE));
    REAL_TYPE *const d = (REAL_TYPE*)libxsmm_malloc(ldc * n * sizeof(REAL_TYPE));

    init(42, a, scale, m, k, lda);
    init(24, b, scale, k, n, ldb);
    init(0, c, scale, m, n, ldc);
    init(0, d, scale, m, n, ldc);

    /* warmup BLAS library (populate thread pool) */
    LIBXSMM_YGEMM_SYMBOL(REAL_TYPE)(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
    LIBXSMM_XBLAS_SYMBOL(REAL_TYPE)(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, d, &ldc);

    fprintf(stdout, "%s('%c', '%c', %i/*m*/, %i/*n*/, %i/*k*/,\n"
                    "      %g/*alpha*/, %p/*a*/, %i/*lda*/,\n"
                    "                  %p/*b*/, %i/*ldb*/,\n"
                    "       %g/*beta*/, %p/*c*/, %i/*ldc*/)\n",
      LIBXSMM_STRINGIFY(LIBXSMM_TPREFIX(REAL_TYPE, gemm)),
      transa, transb, m, n, k, alpha, (const void*)a, lda,
                                      (const void*)b, ldb,
                                beta, (const void*)c, ldc);

    { /* Tiled xGEMM */
      int i; double duration;
      unsigned long long start = libxsmm_timer_tick();
      for (i = 0; i < nrepeat; ++i) {
        LIBXSMM_YGEMM_SYMBOL(REAL_TYPE)(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
      }
      duration = libxsmm_timer_duration(start, libxsmm_timer_tick());
      if (0 < duration) {
        fprintf(stdout, "\tLIBXSMM: %.1f GFLOPS/s\n", gflops * nrepeat / duration);
      }
    }

    { /* LAPACK/BLAS xGEMM */
      int i; double duration;
      unsigned long long start = libxsmm_timer_tick();
      for (i = 0; i < nrepeat; ++i) {
        LIBXSMM_XBLAS_SYMBOL(REAL_TYPE)(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, d, &ldc);
      }
      duration = libxsmm_timer_duration(start, libxsmm_timer_tick());
      if (0 < duration) {
        fprintf(stdout, "\tBLAS: %.1f GFLOPS/s\n", gflops * nrepeat / duration);
      }
    }

    { /* Validate with LAPACK/BLAS */
      libxsmm_blasint i, j; double diff = 0;
      for (i = 0; i < n; ++i) {
        for (j = 0; j < m; ++j) {
          const libxsmm_blasint h = i * ldc + j;
          const double e = c[h] - d[h];
          diff = LIBXSMM_MAX(diff, e * e);
        }
      }
      fprintf(stdout, "\tdiff=%f\n", diff);
    }

    libxsmm_free(a);
    libxsmm_free(b);
    libxsmm_free(c);
    libxsmm_free(d);
  }
  fprintf(stdout, "Finished\n");

  return EXIT_SUCCESS;
}
