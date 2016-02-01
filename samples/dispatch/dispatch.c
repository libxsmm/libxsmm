#include <libxsmm.h>
#include <libxsmm_timer.h>
#include <stdlib.h>
#include <stdio.h>


/**
 * This (micro-)benchmark optionally takes a number of dispatches to be performed.
 * The program measures the duration needed to figure out whether a requested matrix
 * multiplication is available or not. The measured duration excludes the time taken
 * to actually generate the code during the first dispatch.
 */
int main(int argc, char* argv[])
{
  const int size = LIBXSMM_DEFAULT(1 << 25, 1 < argc ? atoi(argv[1]) : 0);
  unsigned long long start;
  double duration;
  int i;

  fprintf(stdout, "Dispatching %i calls %s internal synchronization...\n", size,
#if 0 != LIBXSMM_SYNC
    "with");
#else
    "without");
#endif
#if 0 != LIBXSMM_JIT
  { const char *const jit = getenv("LIBXSMM_JIT");
    if (0 != jit && '0' == *jit) {
      fprintf(stderr, "\tWarning: JIT support has been disabled at runtime!\n");
    }
  }
#else
  fprintf(stderr, "\tWarning: JIT support has been disabled at build time!\n");
#endif

  /* first invocation may actually generate code (which is here out of interest) */
  libxsmm_dmmdispatch(LIBXSMM_AVG_M, LIBXSMM_AVG_N, LIBXSMM_AVG_K,
    NULL/*lda*/, NULL/*ldb*/, NULL/*ldc*/, NULL/*alpha*/, NULL/*beta*/,
    NULL/*flags*/, NULL/*prefetch*/);

  start = libxsmm_timer_tick();
#if defined(_OPENMP)
# pragma omp parallel for
#endif
  for (i = 0; i < size; ++i) {
    libxsmm_dmmdispatch(LIBXSMM_AVG_M, LIBXSMM_AVG_N, LIBXSMM_AVG_K,
      NULL/*lda*/, NULL/*ldb*/, NULL/*ldc*/, NULL/*alpha*/, NULL/*beta*/,
      NULL/*flags*/, NULL/*prefetch*/);
  }

  duration = libxsmm_timer_duration(start, libxsmm_timer_tick());
  if (0 < duration) {
    fprintf(stdout, "\tcalls/s: %.0f Hz\n", size / duration);
    fprintf(stdout, "\tduration: %.0f ms\n", 1000.0 * duration);
  }
  fprintf(stdout, "Finished\n");

  return EXIT_SUCCESS;
}

