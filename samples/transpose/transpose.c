#include <libxsmm_timer.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#if !defined(REAL_TYPE)
# define REAL_TYPE double
#endif


LIBXSMM_INLINE REAL_TYPE initial_value(libxsmm_blasint i, libxsmm_blasint j, libxsmm_blasint ld)
{
  return (REAL_TYPE)(j * ld + i);
}


int main(int argc, char* argv[])
{
  const libxsmm_blasint m = 1 < argc ? atoi(argv[1]) : 4096;
  const libxsmm_blasint n = 2 < argc ? atoi(argv[2]) : m;
  const libxsmm_blasint lda = LIBXSMM_MAX(3 < argc ? atoi(argv[3]) : 0, LIBXSMM_LD(n, m));
  const libxsmm_blasint ldb = LIBXSMM_MAX(4 < argc ? atoi(argv[4]) : 0, LIBXSMM_LD(m, n));

  REAL_TYPE *const a = (REAL_TYPE*)malloc(lda * LIBXSMM_LD(m, n) * sizeof(REAL_TYPE));
  REAL_TYPE *const b = (REAL_TYPE*)malloc(ldb * LIBXSMM_LD(n, m) * sizeof(REAL_TYPE));
  const unsigned int size = m * n * sizeof(REAL_TYPE);
  libxsmm_blasint i, j;

  fprintf(stdout, "m=%i n=%i lda=%i ldb=%i (%s, %s) memory=%.f MB\n", m, n, lda, ldb,
    0 != LIBXSMM_ROW_MAJOR ? "row-major" : "column-major",
    8 == sizeof(REAL_TYPE) ? "DP" : "SP",
    1.0 * size / (1 << 20));

  for (j = 0; j < LIBXSMM_LD(m, n); ++j) {
    for (i = 0; i < LIBXSMM_LD(n, m); ++i) {
      a[j*lda+i] = initial_value(i, j, lda);
    }
  }

  {
    double duration;
    const unsigned long long start = libxsmm_timer_tick();
    libxsmm_transpose_oop(b, a, sizeof(REAL_TYPE), m, n, lda, ldb);
    libxsmm_transpose_oop(a, b, sizeof(REAL_TYPE), n, m, ldb, lda);
    duration = libxsmm_timer_duration(start, libxsmm_timer_tick());
    if (0 < duration) {
      fprintf(stdout, "\tbandwidth: %.1f GB/s\n", size / (duration * (1 << 30)));
    }
    fprintf(stdout, "\tduration: %.0f ms\n", 1000.0 * duration);
  }

  for (j = 0; j < LIBXSMM_LD(m, n); ++j) {
    for (i = 0; i < LIBXSMM_LD(n, m); ++i) {
      if (0 < fabs(a[j*lda+i] - initial_value(i, j, lda))) {
        j = LIBXSMM_LD(m, n) + 1;
        break;
      }
    }
  }

  if (LIBXSMM_LD(m, n) < j) {
    fprintf(stderr, "Validation failed!\n");
  }

  free(a);
  free(b);

  return EXIT_SUCCESS;
}

