#include <libxsmm.h>
#include <stdlib.h>


int main()
{
  /* we do not care about the initial values */
  const float a[23*23], b[23*23];
  int i;

#if defined(_OPENMP)
# pragma omp parallel for
#endif
  for (i = 0; i < 1000; ++i) {
    libxsmm_init();
  }

#if defined(_OPENMP)
# pragma omp parallel for
#endif
  for (i = 0; i < 1000; ++i) {
    LIBXSMM_ALIGNED(float c[LIBXSMM_ALIGN_VALUE(23,sizeof(float),LIBXSMM_ALIGNMENT)*23], LIBXSMM_ALIGNMENT);
    const libxsmm_sfunction f = libxsmm_sdispatch(
      LIBXSMM_FLAGS, 23, 23, 23,
      0/*lda*/, 0/*ldb*/, 0/*ldc*/,
      0/*alpha*/, 0/*beta*/);
    if (0 != f) {
      f(a, b, c);
    }
    else {
      libxsmm_smm(LIBXSMM_FLAGS,
        23, 23, 23, a, b, c,
        LIBXSMM_PREFETCH_A(a),
        LIBXSMM_PREFETCH_B(b),
        LIBXSMM_PREFETCH_C(c),
        0/*alpha*/, 0/*beta*/);
    }
  }

#if defined(_OPENMP)
# pragma omp parallel for
#endif
  for (i = 0; i < 1000; ++i) {
    libxsmm_finalize();
  }

  return EXIT_SUCCESS;
}
