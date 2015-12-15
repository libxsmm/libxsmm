#include <libxsmm.h>
#include <stdlib.h>
#include <stdio.h>

#if defined(__MKL)
# include <mkl.h>
#endif

/*#define USE_LIBXSMM_BLAS*/
/*#define USE_CBLAS*/
#define REAL float


int main()
{
#if 0 != LIBXSMM_ROW_MAJOR
  const libxsmm_blasint m = 64, n = 239, k = 64, lda = 64, ldb = 240, ldc = 240;
  REAL a[k*lda], b[n*ldb], c[n*ldc], d[n*ldc];
  const REAL alpha = 1, beta = 1;
  const char notrans = 'N';
  int i, j;

  for (i = 0; i < m; ++i) {
    for (j = 0; j < k; ++j) {
      const int index = i * lda + j;
      a[index] = ((REAL)1) / (index + 1);
    }
  }
  for (i = 0; i < k; ++i) {
    for (j = 0; j < n; ++j) {
      const int index = i * ldb + j;
      b[index] = ((REAL)2) / (index + 1);
    }
  }
  for (i = 0; i < m; ++i) {
    for (j = 0; j < n; ++j) {
      const int index = i * ldc + j;
      c[index] = d[index] = 1000;
    }
  }

  LIBXSMM_CONCATENATE(libxsmm_, LIBXSMM_TPREFIX(REAL, gemm))(&notrans, &notrans, &m, &n, &k,
    &alpha, a, &lda, b, &ldb, &beta, c, &ldc);

#if defined(__MKL) || defined(USE_CBLAS)
  LIBXSMM_CONCATENATE(cblas_, LIBXSMM_TPREFIX(REAL, gemm))(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k,
    alpha, a, lda, b, ldb, beta, d, ldc);
#elif defined(USE_LIBXSMM_BLAS)
  LIBXSMM_CONCATENATE(libxsmm_blas_, LIBXSMM_TPREFIX(REAL, gemm))(&notrans, &notrans, &m, &n, &k,
    &alpha, a, &lda, b, &ldb, &beta, d, &ldc);
#else
  LIBXSMM_FSYMBOL(LIBXSMM_TPREFIX(REAL, gemm))(&notrans, &notrans, &n, &m, &k,
    &alpha, b, &ldb, a, &lda, &beta, d, &ldc);
#endif

  double d2 = 0;
  for (i = 0; i < m; ++i) {
    for (j = 0; j < n; ++j) {
      const int index = i * ldc + j;
      const double d1 = c[index] - d[index];
      d2 += d1 * d1;
    }
  }

  return 0.001 > d2 ? EXIT_SUCCESS : EXIT_FAILURE;
#else
  fprintf(stderr, "Please rebuild LIBXSMM with ROW_MAJOR=1");
  return EXIT_SUCCESS;
#endif
}

