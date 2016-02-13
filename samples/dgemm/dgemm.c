#include <stdlib.h>
#include <stdio.h>

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


int main(int argc, char* argv[])
{
  const int size = 1 < argc ? atoi(argv[1]) : (1 << 22);
  const int m = 23, n = m, k = m, lda = m, ldb = k, ldc = m, big = 512;
  const double alpha = 1, beta = 1;
  const char notrans = 'N';
  double *a, *b, *c;
  int i, j;

  a = (double*)malloc(big/*m*/ * big/*k*/ * sizeof(double));
  b = (double*)malloc(big/*k*/ * big/*n*/ * sizeof(double));
  c = (double*)malloc(big/*m*/ * big/*n*/ * sizeof(double));

  for (i = 0; i < big/*m*/; ++i) {
    for (j = 0; j < big/*k*/; ++j) {
      const int index = i * big/*lda*/ + j;
      a[index] = 1.0 / (index + 1);
    }
  }
  for (i = 0; i < big/*k*/; ++i) {
    for (j = 0; j < big/*n*/; ++j) {
      const int index = i * big/*ldb*/ + j;
      b[index] = 2.0 / (index + 1);
    }
  }
  for (i = 0; i < big/*m*/; ++i) {
    for (j = 0; j < big/*n*/; ++j) {
      const int index = i * big/*ldc*/ + j;
      c[index] = -1.0;
    }
  }

  /**
   * warmup BLAS, and check that a "bigger" DGEMM (above LIBXSMM's threshold)
   * does not enter a recursion if LIBXSMM is LD_PRELOADED or wrapped-in
   */
  dgemm_(&notrans, &notrans, &big, &big, &big, &alpha, a, &big, b, &big, &beta, c, &big);

#if defined(_OPENMP)
# pragma omp parallel for
#endif
  for (i = 0; i < size; ++i) {
    dgemm_(&notrans, &notrans, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
  }

  free(a);
  free(b);
  free(c);

  return EXIT_SUCCESS;
}

