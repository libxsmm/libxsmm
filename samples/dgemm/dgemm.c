#include <stdlib.h>
#include <stdio.h>

/** Problem size (sample) */
#define M 23
#define N M
#define K M
#define LDA M
#define LDB K
#define LDC M

/** Construct symbol name from a given real type name (float or double). */
#define GEMM(REAL) LIBXSMM_FSYMBOL(LIBXSMM_TPREFIX(REAL, gemm))

/** Function prototype for SGEMM; any kind of BLAS library should be sufficient at link-time. */
void sgemm_(const char*, const char*, const int*, const int*, const int*,
  const float*, const float*, const int*, const float*, const int*,
  const float*, float*, const int*);
/** Function prototype for DGEMM; any kind of BLAS library should be sufficient at link-time. */
void dgemm_(const char*, const char*, const int*, const int*, const int*,
  const double*, const double*, const int*, const double*, const int*,
  const double*, double*, const int*);


int main()
{
  const int m = M, n = N, k = K, lda = LDA, ldb = LDB, ldc = LDC;
  double a[LDA*K], b[LDB*N], c[LDC*N];
  const double alpha = 1, beta = 1;
  const char notrans = 'N';
  int i, j;

  for (i = 0; i < m; ++i) {
    for (j = 0; j < k; ++j) {
      const int index = i * lda + j;
      a[index] = 1.0 / (index + 1);
    }
  }
  for (i = 0; i < k; ++i) {
    for (j = 0; j < n; ++j) {
      const int index = i * ldb + j;
      b[index] = 2.0 / (index + 1);
    }
  }
  for (i = 0; i < m; ++i) {
    for (j = 0; j < n; ++j) {
      const int index = i * ldc + j;
      c[index] = -1.0;
    }
  }

  for (i = 0; i < 1000; ++i) {
    dgemm_(&notrans, &notrans, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
  }

  return EXIT_SUCCESS;
}

