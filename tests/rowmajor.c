#include <libxsmm.h>
#include <stdlib.h>
#include <stdio.h>

#if !defined(REAL_TYPE)
# define REAL_TYPE float
#endif

/*#define USE_LIBXSMM_BLAS*/
#define M 64
#define N 239
#define K 64
#define LDA 64
#define LDB 240
#define LDC 240


int main()
{
#if 0 != LIBXSMM_ROW_MAJOR
  const libxsmm_blasint m = M, n = N, k = K, lda = LDA, ldb = LDB, ldc = LDC;
  REAL_TYPE a[K*LDA], b[N*LDB], c[N*LDC], d[N*LDC];
  const REAL_TYPE alpha = 1, beta = 1;
  const char notrans = 'N';
  double d2 = 0;
  int i, j;

  for (i = 0; i < m; ++i) {
    for (j = 0; j < k; ++j) {
      const int index = i * lda + j;
      a[index] = ((REAL_TYPE)1) / (index + 1);
    }
  }
  for (i = 0; i < k; ++i) {
    for (j = 0; j < n; ++j) {
      const int index = i * ldb + j;
      b[index] = ((REAL_TYPE)2) / (index + 1);
    }
  }
  for (i = 0; i < m; ++i) {
    for (j = 0; j < n; ++j) {
      const int index = i * ldc + j;
      c[index] = d[index] = 1000;
    }
  }

  LIBXSMM_XGEMM_SYMBOL(REAL_TYPE)(&notrans, &notrans, &m, &n, &k,
    &alpha, a, &lda, b, &ldb, &beta, c, &ldc);

#if defined(USE_LIBXSMM_BLAS)
  LIBXSMM_XBLAS_GEMM_SYMBOL(REAL_TYPE)(&notrans, &notrans, &m, &n, &k,
    &alpha, a, &lda, b, &ldb, &beta, d, &ldc);
#else
  LIBXSMM_BLAS_GEMM_SYMBOL(REAL_TYPE)(&notrans, &notrans, &n, &m, &k,
    &alpha, b, &ldb, a, &lda, &beta, d, &ldc);
#endif

  for (i = 0; i < m; ++i) {
    for (j = 0; j < n; ++j) {
      const int index = i * ldc + j;
      const double d1 = c[index] - d[index];
      d2 += d1 * d1;
    }
  }

  return 0.001 > d2 ? EXIT_SUCCESS : EXIT_FAILURE;
#else
  fprintf(stderr, "Please rebuild LIBXSMM with ROW_MAJOR=1\n");
  return EXIT_SUCCESS;
#endif
}

