/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Hans Pabst (Intel Corp.)
******************************************************************************/
#include <utils/libxsmm_timer.h>
#include <libxsmm.h>

#if defined(_OPENMP)
# include <omp.h>
#endif
#if defined(__TBB)
# include <tbb/scalable_allocator.h>
#endif

#if defined(__TBB)
# define MALLOC(SIZE) scalable_malloc(SIZE)
# define FREE(PTR) scalable_free(PTR)
#elif defined(_OPENMP) && defined(LIBXSMM_INTEL_COMPILER) && (1901 > LIBXSMM_INTEL_COMPILER) && 0
# define MALLOC(SIZE) kmp_malloc(SIZE)
# define FREE(PTR) kmp_free(PTR)
#elif defined(LIBXSMM_PLATFORM_X86) && 0
# define MALLOC(SIZE) _mm_malloc(SIZE, LIBXSMM_ALIGNMENT)
# define FREE(PTR) _mm_free(PTR)
#elif 1
# define MALLOC(SIZE) malloc(SIZE)
# define FREE(PTR) free(PTR)
#endif

#if !defined(MAX_MALLOC_MB)
# define MAX_MALLOC_MB 100
#endif
#if !defined(MAX_MALLOC_N)
# define MAX_MALLOC_N 24
#endif


void* malloc_offsite(size_t size);


int main(int argc, char* argv[])
{
#if defined(_OPENMP)
  const int max_nthreads = omp_get_max_threads();
#else
  const int max_nthreads = 1;
#endif
  const int ncycles = LIBXSMM_MAX(1 < argc ? atoi(argv[1]) : 100, 1);
  const int max_nallocs = LIBXSMM_CLMP(2 < argc ? atoi(argv[2]) : 4, 1, MAX_MALLOC_N);
  const int nthreads = LIBXSMM_CLMP(3 < argc ? atoi(argv[3]) : 1, 1, max_nthreads);
  const char *const longlife_env = getenv("LONGLIFE"), *const env_check = getenv("CHECK");
  const int enable_longlife = ((NULL == longlife_env || 0 == *longlife_env) ? 0 : atoi(longlife_env));
  void* longlife = (0 == enable_longlife ? NULL : malloc_offsite((MAX_MALLOC_MB) << 20));
  const double check = LIBXSMM_ABS(NULL == env_check ? 0 : atof(env_check));
  unsigned int nallocs = 0, nerrors0 = 0, nerrors1 = 0;
  libxsmm_timer_tickint d0 = 0, d1 = 0;
  libxsmm_scratch_info info;
  int r[MAX_MALLOC_N], i;
  int max_size = 0;
  int scratch = 0;

  /* generate set of random numbers for parallel region */
  for (i = 0; i < (MAX_MALLOC_N); ++i) r[i] = rand();

  /* count number of calls according to randomized scheme */
  for (i = 0; i < ncycles; ++i) {
    const int count = r[i%(MAX_MALLOC_N)] % max_nallocs + 1;
    int mbytes = 0, j;
    for (j = 0; j < count; ++j) {
      const int k = (i * count + j) % (MAX_MALLOC_N);
      mbytes += (r[k] % (MAX_MALLOC_MB) + 1);
    }
    if (max_size < mbytes) max_size = mbytes;
    nallocs += count;
  }
  assert(0 != nallocs);

  fprintf(stdout, "Running %i cycles with max. %i malloc+free (%u calls) using %i thread%s...\n",
    ncycles, max_nallocs, nallocs, 1 >= nthreads ? 1 : nthreads, 1 >= nthreads ? "" : "s");

  libxsmm_init();

#if defined(_OPENMP)
# pragma omp parallel for num_threads(nthreads) private(i) reduction(+:d1,nerrors1)
#endif
  for (i = 0; i < ncycles; ++i) {
    const int count = r[i%(MAX_MALLOC_N)] % max_nallocs + 1;
    void* p[MAX_MALLOC_N];
    int j;
    assert(count <= MAX_MALLOC_N);
    for (j = 0; j < count; ++j) {
      const int k = (i * count + j) % (MAX_MALLOC_N);
      const size_t nbytes = ((size_t)r[k] % (MAX_MALLOC_MB) + 1) << 20;
      const libxsmm_timer_tickint t1 = libxsmm_timer_tick();
      p[j] = libxsmm_aligned_scratch(nbytes, 0/*auto*/);
      d1 += libxsmm_timer_ncycles(t1, libxsmm_timer_tick());
      if (NULL == p[j]) {
        ++nerrors1;
      }
      else if (0 != check) {
        memset(p[j], j, nbytes);
      }
    }
    for (j = 0; j < count; ++j) {
      libxsmm_free(p[j]);
    }
  }
  libxsmm_free(longlife);
  if (EXIT_SUCCESS == libxsmm_get_scratch_info(&info) && 0 < info.size) {
    scratch = (int)(1.0 * LIBXSMM_MAX(info.size, info.local) / (1ULL << 20) + 0.5);
    fprintf(stdout, "\nScratch: %i MB (mallocs=%lu, pools=%u)\n",
      scratch, (unsigned long int)info.nmallocs, info.npools);
    libxsmm_release_scratch(); /* suppress LIBXSMM's termination message about scratch */
  }

#if (defined(MALLOC) && defined(FREE))
  longlife = (0 == enable_longlife ? NULL : MALLOC((MAX_MALLOC_MB) << 20));
  if (NULL == longlife) max_size += MAX_MALLOC_MB;
#if defined(_OPENMP)
# pragma omp parallel for num_threads(nthreads) private(i) reduction(+:d0,nerrors0)
#endif
  for (i = 0; i < ncycles; ++i) {
    const int count = r[i % (MAX_MALLOC_N)] % max_nallocs + 1;
    void* p[MAX_MALLOC_N];
    int j;
    assert(count <= MAX_MALLOC_N);
    for (j = 0; j < count; ++j) {
      const int k = (i * count + j) % (MAX_MALLOC_N);
      const size_t nbytes = ((size_t)r[k] % (MAX_MALLOC_MB) + 1) << 20;
      const libxsmm_timer_tickint t1 = libxsmm_timer_tick();
      p[j] = MALLOC(nbytes);
      d0 += libxsmm_timer_ncycles(t1, libxsmm_timer_tick());
      if (NULL == p[j]) {
        ++nerrors0;
      }
      else if (0 != check) {
        memset(p[j], j, nbytes);
      }
    }
    for (j = 0; j < count; ++j) FREE(p[j]);
  }
  FREE(longlife);
#endif /*(defined(MALLOC) && defined(FREE))*/

  if (0 != d0 && 0 != d1 && 0 < nallocs) {
    const double dcalls = libxsmm_timer_duration(0, d0);
    const double dalloc = libxsmm_timer_duration(0, d1);
    const double scratch_freq = 1E-3 * nallocs / dalloc;
    const double malloc_freq = 1E-3 * nallocs / dcalls;
    const double speedup = scratch_freq / malloc_freq;
    fprintf(stdout, "\tlibxsmm scratch calls/s: %.1f kHz\n", scratch_freq);
    fprintf(stdout, "Malloc: %i MB\n", max_size);
    fprintf(stdout, "\tstd.malloc+free calls/s: %.1f kHz\n", malloc_freq);
    fprintf(stdout, "Fair (size vs. speed): %.1fx\n", max_size * speedup / scratch);
    fprintf(stdout, "Scratch Speedup: %.1fx\n", speedup);
  }

  if (0 != nerrors0 || 0 != nerrors1) {
    fprintf(stdout, "FAILED (errors: malloc=%u libxsmm=%u)\n", nerrors0, nerrors1);
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}


void* malloc_offsite(size_t size) { return libxsmm_aligned_scratch(size, 0/*auto*/); }
