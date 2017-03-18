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

#if !defined(MAX_BATCHSIZE)
# define MAX_BATCHSIZE 10
#endif
#if !defined(MAX_ALLOC_MB)
# define MAX_ALLOC_MB 100
#endif


int main(int argc, char* argv[])
{
  const int size = LIBXSMM_DEFAULT(10, 1 < argc ? atoi(argv[1]) : 0);
  const int nthreads = LIBXSMM_DEFAULT(1, 2 < argc ? atoi(argv[2]) : 0);
  unsigned long long start;
  double dcall, dalloc;
  void* p[MAX_BATCHSIZE];
  int r[MAX_BATCHSIZE];
  unsigned int ncalls = 0;
  int i, j;

#if defined(_OPENMP)
  if (1 < nthreads) omp_set_num_threads(nthreads);
#endif

  /* generate set of random number for parallel region */
  for (i = 0; i < (MAX_BATCHSIZE); ++i) r[i] = rand();

  /* count number of calls according to randomized scheme */
  for (i = 0; i < size; ++i) {
    ncalls += (r[i % (MAX_BATCHSIZE)] % (MAX_BATCHSIZE)) + 1;
  }
  assert(0 != ncalls);

  fprintf(stdout, "Running %u allocation+free cycles using %i thread%s...\n", ncalls,
    1 >= nthreads ? 1 : nthreads,
    1 >= nthreads ? "" : "s");

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload target(LIBXSMM_OFFLOAD_TARGET)
#endif
  {
    /* run non-inline function to measure call overhead of an "empty" function */
    start = libxsmm_timer_tick();
#if defined(_OPENMP)
#   pragma omp parallel for default(none) private(i)
#endif
    for (i = 0; i < (size * (MAX_BATCHSIZE)); ++i) {
      libxsmm_init(); /* subsequent calls are not doing any work */
    }
    dcall = libxsmm_timer_duration(start, libxsmm_timer_tick());

    start = libxsmm_timer_tick();
#if defined(_OPENMP)
#   pragma omp parallel for default(none) private(i, j)
#endif
    for (i = 0; i < size; ++i) {
      const int nbatch = (r[i%(MAX_BATCHSIZE)] % (MAX_BATCHSIZE)) + 1;
      for (j = 0; j < nbatch; ++j) {
        const size_t nbytes = (r[j%(MAX_BATCHSIZE)] % (MAX_ALLOC_MB) + 1) << 20;
        p[j] = libxsmm_aligned_scratch(nbytes, 0/*auto*/);
      }
      for (j = 0; j < nbatch; ++j) {
        libxsmm_free(p[j]);
      }
    }
    dalloc = libxsmm_timer_duration(start, libxsmm_timer_tick());
  }

  if (0 < dcall && 0 < dalloc) {
    fprintf(stdout, "\tallocation+free calls/s: %.1f MHz\n", 1E-6 * ncalls / dalloc);
    fprintf(stdout, "\tempty calls/s: %.1f MHz\n", 1E-6 * (size * (MAX_BATCHSIZE)) / dcall);
    fprintf(stdout, "\toverhead: %.1fx\n", dalloc / dcall);
  }
  fprintf(stdout, "Finished\n");

  return EXIT_SUCCESS;
}

