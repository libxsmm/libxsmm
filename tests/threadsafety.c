#include <libxsmm.h>
#include <stdlib.h>


int main()
{
  /* we do not care about the initial values */
  /*const*/ float a[23*23], b[23*23];
  int i;

#if defined(_OPENMP)
# pragma omp parallel for default(none) private(i)
#endif
  for (i = 0; i < 1000; ++i) {
    libxsmm_init();
  }

#if defined(_OPENMP)
# pragma omp parallel for default(none) private(i) shared(a, b)
#endif
  for (i = 0; i < 1000; ++i) {
    float c[23*23];
    const libxsmm_smmfunction f = libxsmm_smmdispatch(23, 23, (i / 50) % 23 + 1,
      NULL/*lda*/, NULL/*ldb*/, NULL/*ldc*/, NULL/*alpha*/, NULL/*beta*/,
      NULL/*flags*/, NULL/*prefetch*/);
    if (NULL != f) {
      LIBXSMM_MMCALL_ABC(f, a, b, c);
    }
    else {
      const libxsmm_blasint m = 23, n = 23, k = (i / 50) % 23 + 1;
      libxsmm_sgemm(NULL/*transa*/, NULL/*transb*/, &m, &n, &k,
        NULL/*alpha*/, a, NULL/*lda*/, b, NULL/*ldb*/, 
        NULL/*beta*/, c, NULL/*ldc*/);
    }
  }

#if defined(_OPENMP)
# pragma omp parallel for default(none) private(i)
#endif
  for (i = 0; i < 1000; ++i) {
    libxsmm_finalize();
  }

  return EXIT_SUCCESS;
}
