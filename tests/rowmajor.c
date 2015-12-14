#include <libxsmm.h>
#include <stdlib.h>
#include <stdio.h>

#define REAL float


int main()
{
  const libxsmm_blasint m = 64, n = 239, k = 64, lda = 64, ldb = 240, ldc = 240;
  REAL a[lda*k], b[ldb*n], c[ldc*n], d[ldc*n];
  const char notrans = 'N';
  int i, j;

  for (i = 0; i < lda; ++i) {
    for (j = 0; j < k; ++j) {
      const int index = i * lda + j;
      a[index] = ((REAL)1) / ((REAL)(index + 1));
    }
  }
  for (i = 0; i < ldb; ++i) {
    for (j = 0; j < n; ++j) {
      const int index = i * ldb + j;
      b[index] = ((REAL)2) / ((REAL)(index + 1));
    }
  }
  for (i = 0; i < ldc; ++i) {
    for (j = 0; j < n; ++j) {
      const int index = i * ldb + j;
      c[index] = d[index] = 1000;
    }
  }

  LIBXSMM_CONCATENATE(libxsmm_, LIBXSMM_TPREFIX(REAL, gemm))(&notrans, &notrans, &m, &n, &k,
    NULL/*alpha*/, a, &lda, b, &ldb, NULL/*beta*/, c, &ldc);

  LIBXSMM_CONCATENATE(libxsmm_blas_, LIBXSMM_TPREFIX(REAL, gemm))(&notrans, &notrans, &m, &n, &k,
    NULL/*alpha*/, a, &lda, b, &ldb, NULL/*beta*/, d, &ldc);

  double d2 = 0;
  for (i = 0; i < m; ++i) {
    for (j = 0; j < n; ++j) {
      const double d1 = c[i*ldc+j] - d[i*ldc+j];
      d2 += d1 * d1;
    }
  }

  return 0.001 > d2 ? EXIT_SUCCESS : EXIT_FAILURE;
}
