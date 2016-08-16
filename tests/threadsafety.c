#include <libxsmm.h>
#include <stdlib.h>
#include <stdio.h>

#if !defined(MAX_NKERNELS)
# define MAX_NKERNELS 1000
#endif

#if !defined(USE_PARALLEL_JIT)
# define USE_PARALLEL_JIT
#endif


int main()
{
  /* we do not care about the initial values */
  /*const*/ float a[23*23], b[23*23];
  libxsmm_smmfunction f[MAX_NKERNELS];
  int result = EXIT_SUCCESS, i;

#if defined(_OPENMP)
# pragma omp parallel for default(none) private(i)
#endif
  for (i = 0; i < MAX_NKERNELS; ++i) {
    libxsmm_init();
  }

#if defined(_OPENMP) && defined(USE_PARALLEL_JIT)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < MAX_NKERNELS; ++i) {
    const libxsmm_blasint m = 23, n = 23, k = (i / 50) % 23 + 1;
    /* OpenMP: playing ping-pong with fi's cache line is not the subject */
    f[i] = libxsmm_smmdispatch(m, n, k,
      NULL/*lda*/, NULL/*ldb*/, NULL/*ldc*/, NULL/*alpha*/, NULL/*beta*/,
      NULL/*flags*/, NULL/*prefetch*/);
  }

#if defined(_OPENMP) && !defined(USE_PARALLEL_JIT)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < MAX_NKERNELS; ++i) {
    if (EXIT_SUCCESS == result) {
      const libxsmm_blasint m = 23, n = 23, k = (i / 50) % 23 + 1;
      float c[23/*m*/*23/*n*/];

      if (NULL != f[i]) {
        const libxsmm_smmfunction fi = libxsmm_smmdispatch(m, n, k,
          NULL/*lda*/, NULL/*ldb*/, NULL/*ldc*/, NULL/*alpha*/, NULL/*beta*/,
          NULL/*flags*/, NULL/*prefetch*/);

        if (fi == f[i]) {
          LIBXSMM_MMCALL(f[i], a, b, c, m, n, k);
        }
        else if (NULL != fi) {
#if defined(_DEBUG)
          fprintf(stderr, "Error: the %ix%ix%i-kernel does not match!\n", m, n, k);
#endif
#if defined(_OPENMP) && !defined(USE_PARALLEL_JIT)
# if (201107 <= _OPENMP)
#         pragma omp atomic write
# else
#         pragma omp critical
# endif
#endif
          result = EXIT_FAILURE;
        }
#if defined(_DEBUG)
        else { /* did not find previously generated and recorded kernel */
          fprintf(stderr, "Error: cannot find %ix%ix%i-kernel!\n", m, n, k);
        }
#endif
      }
      else {
        libxsmm_sgemm(NULL/*transa*/, NULL/*transb*/, &m, &n, &k,
          NULL/*alpha*/, a, NULL/*lda*/, b, NULL/*ldb*/,
          NULL/*beta*/, c, NULL/*ldc*/);
      }
    }
  }

#if defined(_OPENMP)
# pragma omp parallel for default(none) private(i)
#endif
  for (i = 0; i < MAX_NKERNELS; ++i) {
    libxsmm_finalize();
  }

  return result;
}
