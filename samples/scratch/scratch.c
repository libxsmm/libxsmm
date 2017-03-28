/******************************************************************************
** Copyright (c) 2017, Intel Corporation                                     **
** All rights reserved.                                                      **
**                                                                           **
** Redistribution and use in source and binary forms, with or without        **
** modification, are permitted provided that the following conditions        **
** are met:                                                                  **
** 1. Redistributions of source code must retain the above copyright         **
**    notice, this list of conditions and the following disclaimer.          **
** 2. Redistributions in binary form must reproduce the above copyright      **
**    notice, this list of conditions and the following disclaimer in the    **
**    documentation and/or other materials provided with the distribution.   **
** 3. Neither the name of the copyright holder nor the names of its          **
**    contributors may be used to endorse or promote products derived        **
**    from this software without specific prior written permission.          **
**                                                                           **
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS       **
** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT         **
** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR     **
** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT      **
** HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,    **
** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED  **
** TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR    **
** PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF    **
** LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING      **
** NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS        **
** SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.              **
******************************************************************************/
/* Hans Pabst (Intel Corp.)
******************************************************************************/
#include <libxsmm.h>

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#include <stdlib.h>
#include <stdio.h>
#if defined(_OPENMP)
# include <omp.h>
#endif
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

#if !defined(USE_SCRATCH_MALLOC)
# define USE_SCRATCH_MALLOC
#endif
#if defined(USE_SCRATCH_MALLOC)
# define MALLOC(SIZE) libxsmm_aligned_scratch(SIZE, 0/*auto*/)
# define FREE(POINTER) libxsmm_free(POINTER)
#else
# define MALLOC(SIZE) malloc(SIZE)
# define FREE(POINTER) free(POINTER)
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
  const int ncycles = LIBXSMM_DEFAULT(1000000, 1 < argc ? atoi(argv[1]) : 0);
  const int nalloc = LIBXSMM_CLMP(2 < argc ? atoi(argv[2]) : 4, 1, MAX_MALLOC_N);
  const int nthreads = LIBXSMM_DEFAULT(1, 3 < argc ? atoi(argv[3]) : 0);
  unsigned long long start;
  unsigned int ncalls = 0;
  double dcall, dalloc;
  void* p[MAX_MALLOC_N];
  int r[MAX_MALLOC_N];
  int i;

#if defined(_OPENMP)
  if (0 < nthreads) omp_set_num_threads(nthreads);
#endif

  /* generate set of random number for parallel region */
  for (i = 0; i < (MAX_MALLOC_N); ++i) r[i] = rand();

  /* count number of calls according to randomized scheme */
  for (i = 0; i < ncycles; ++i) {
    ncalls += LIBXSMM_MAX(r[i%(MAX_MALLOC_N)] % nalloc, 1);
  }
  assert(0 != ncalls);

  fprintf(stdout, "Running %i cycles with max. %i malloc+free (%u calls) using %i thread%s...\n",
    ncycles, nalloc, ncalls, 1 >= nthreads ? 1 : nthreads, 1 >= nthreads ? "" : "s");

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload target(LIBXSMM_OFFLOAD_TARGET)
#endif
  {
    const char *const longlife_env = getenv("LONGLIFE");
    const int enable_longlife = ((0 == longlife_env || 0 == *longlife_env) ? 0 : atoi(longlife_env));
    void *const longlife = (0 == enable_longlife ? 0 : malloc_offsite((MAX_MALLOC_MB) << 20));

    /* run non-inline function to measure call overhead of an "empty" function */
    start = libxsmm_timer_tick();
#if defined(_OPENMP)
#   pragma omp parallel for default(none) private(i)
#endif
    for (i = 0; i < (ncycles * (MAX_MALLOC_N)); ++i) {
      libxsmm_init(); /* subsequent calls are not doing any work */
    }
    dcall = libxsmm_timer_duration(start, libxsmm_timer_tick());

    start = libxsmm_timer_tick();
#if defined(_OPENMP)
#   pragma omp parallel for default(none) private(i)
#endif
    for (i = 0; i < ncycles; ++i) {
      const int count = LIBXSMM_MAX(r[i%(MAX_MALLOC_N)] % nalloc, 1);
      int j;
      for (j = 0; j < count; ++j) {
        const size_t nbytes = (r[j%(MAX_MALLOC_N)] % (MAX_MALLOC_MB) + 1) << 20;
        p[j] = MALLOC(nbytes);
      }
      for (j = 0; j < count; ++j) {
        FREE(p[j]);
      }
    }
    dalloc = libxsmm_timer_duration(start, libxsmm_timer_tick());
    FREE(longlife);
  }

  if (0 < dcall && 0 < dalloc && 0 < ncalls) {
    const double alloc_freq = 1E-6 * ncalls / dalloc;
    const double empty_freq = 1E-6 * (ncycles * (MAX_MALLOC_N)) / dcall;
    fprintf(stdout, "\tallocation+free calls/s: %.1f MHz\n", alloc_freq);
    fprintf(stdout, "\tempty calls/s: %.1f MHz\n", empty_freq);
    fprintf(stdout, "\toverhead: %.1fx\n", empty_freq / alloc_freq);
  }

  fprintf(stdout, "Finished\n");
  return EXIT_SUCCESS;
}


void* malloc_offsite(size_t size) { return MALLOC(size); }

