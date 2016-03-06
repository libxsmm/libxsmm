#include <libxsmm.h>
#include <libxsmm_timer.h>
#include <stdlib.h>
#include <stdio.h>

#if !defined(REAL_TYPE)
# define REAL_TYPE double
#endif

#define USE_PARALLEL



void init(int seed, REAL_TYPE *LIBXSMM_RESTRICT dst, double scale, libxsmm_blasint nrows, libxsmm_blasint ncols, libxsmm_blasint ld)
{
  const libxsmm_blasint minval = seed, addval = (nrows - 1) * ld + (ncols - 1);
  const libxsmm_blasint maxval = LIBXSMM_MAX(LIBXSMM_ABS(minval), addval);
  const double norm = 0 != maxval ? (scale / maxval) : scale;
  libxsmm_blasint i, j;
#if defined(_OPENMP)
# pragma omp parallel for
#endif
  for (i = 0; i < ncols; ++i) {
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
  REAL_TYPE *const a = (REAL_TYPE*)malloc(lda * k * sizeof(REAL_TYPE));
  REAL_TYPE *const b = (REAL_TYPE*)malloc(ldb * n * sizeof(REAL_TYPE));
  REAL_TYPE *const c = (REAL_TYPE*)malloc(ldc * n * sizeof(REAL_TYPE));
  REAL_TYPE *const d = (REAL_TYPE*)malloc(ldc * n * sizeof(REAL_TYPE));
  const double scale = 1.0 / nrepeat, gflops = 2.0 * m * n * k * 1E-9;
  const char transa = 'N', transb = 'N';
  const REAL_TYPE alpha = 1, beta = 1;
  int i;

  init(42, a, scale, m, k, lda);
  init(24, b, scale, k, n, ldb);
  init(0, c, scale, m, n, ldc);
  init(0, d, scale, m, n, ldc);

  { /* Tiled xGEMM */
    double duration;
    unsigned long long start = libxsmm_timer_tick();
#if defined(_OPENMP) && defined(USE_PARALLEL)
#   pragma omp parallel
#   pragma omp single
#endif
    for (i = 0; i < nrepeat; ++i) {
      LIBXSMM_XOMPS_SYMBOL(REAL_TYPE)(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
    }
    duration = libxsmm_timer_duration(start, libxsmm_timer_tick());
    if (0 < duration) {
      fprintf(stdout, "\tLIBXSMM: %.1f GFLOPS/s\n", gflops * nrepeat / duration);
    }
  }

  { /* LAPACK/BLAS xGEMM */
    double duration;
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
        const libxsmm_blasint k = i * ldc + j;
        const double e = c[k] - d[k];
        diff = LIBXSMM_MAX(diff, e * e);
      }
    }
    fprintf(stdout, "\tdiff=%f\n", diff);
  }

  free(a); free(b); free(c); free(d);
  fprintf(stdout, "Finished\n");

  return EXIT_SUCCESS;
}
