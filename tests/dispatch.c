#include <libxsmm.h>
#include <stdlib.h>
#include <stdio.h>

#if !defined(REAL_TYPE)
# define REAL_TYPE float
#endif


int main()
{
#if 0 != LIBXSMM_JIT
  const int m[] = { 1, 2, 3, 4, 5, 6, 7, LIBXSMM_MAX_M - 1, LIBXSMM_MAX_M, LIBXSMM_MAX_M + 1 };
  const int n[] = { 1, 2, 3, 4, 5, 6, 7, LIBXSMM_MAX_N - 1, LIBXSMM_MAX_N, LIBXSMM_MAX_N + 1 };
  const int k[] = { 1, 2, 3, 4, 5, 6, 7, LIBXSMM_MAX_K - 1, LIBXSMM_MAX_K, LIBXSMM_MAX_K + 1 };
  const int size = sizeof(m) / sizeof(*m), flags = LIBXSMM_FLAGS, prefetch = LIBXSMM_PREFETCH;
  const REAL_TYPE alpha = LIBXSMM_ALPHA, beta = LIBXSMM_BETA;
  int i, j = 0, nerrors = 0;

  for (i = 0; i < size; ++i) {
    const int lda = m[i], ldb = k[i], ldc = m[i];
    if (0 == LIBXSMM_MMDISPATCH_SYMBOL(REAL_TYPE)(m[i], n[i], k[i], &lda, &ldb, &ldc,
      &alpha, &beta, &flags, &prefetch))
    {
      if (0 == j) { /* capture first failure*/
        j = i;
      }
      ++nerrors;
    }
  }

  if (size != nerrors) {
    return size == i ? EXIT_SUCCESS : (i + 1)/*EXIT_FAILURE*/;
  }
  else { /* potentially unsupported platforms (due to calling convention)
          * or environment variable LIBXSMM_JIT is set to zero */
    fprintf(stderr, "JIT support is potentially unavailable\n");
    return EXIT_SUCCESS;
  }
#else
  fprintf(stderr, "Please rebuild LIBXSMM with JIT=1\n");
  return EXIT_SUCCESS;
#endif
}

