/******************************************************************************
** Copyright (c) 2014-2019, Intel Corporation                                **
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
/* Alexander Heinecke (Intel Corp.), Hans Pabst (Intel Corp.)
******************************************************************************/
#include <libxsmm.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#if defined(_OPENMP)
# include <omp.h>
#endif


int main(int argc, char* argv[])
{
  int num_cores, threads_per_core, num_threads, num_iterations;
  libxsmm_timer_tickint start;
  libxsmm_barrier* barrier;

  if (4 < argc) {
    fprintf(stderr, "Usage:\n  %s <cores> <threads-per-core> [<iterations>]\n", argv[0]);
    return EXIT_SUCCESS;
  }

  /* parse the command line and set up the test parameters */
  num_cores = (1 < argc ? atoi(argv[1]) : 2);
  assert(num_cores >= 1);

  threads_per_core = (2 < argc ? atoi(argv[2]) : 2);
  assert(threads_per_core >= 1);

  num_iterations = (1 < argc ? atoi(argv[1]) : 50000);
  assert(num_iterations > 0);

  /* create a new barrier */
  barrier = libxsmm_barrier_create(num_cores, threads_per_core);
  assert(NULL != barrier);

  /* each thread must initialize with the barrier */
  num_threads = num_cores * threads_per_core;
#if defined(_OPENMP)
# pragma omp parallel num_threads(num_threads)
#endif
  {
#if defined(_OPENMP)
    const int tid = omp_get_thread_num();
#else
    const int tid = 0;
#endif
    libxsmm_barrier_init(barrier, tid);
  }

  start = libxsmm_timer_tick();
#if defined(_OPENMP)
# pragma omp parallel num_threads(num_threads)
#endif
  {
#if defined(_OPENMP)
    const int tid = omp_get_thread_num();
#else
    const int tid = 0;
#endif
    int i;
    for (i = 0; i < num_iterations; ++i) {
      libxsmm_barrier_wait(barrier, tid);
    }
  }

  printf("libxsmm_barrier_wait(): %llu cycles (%d threads)\n",
    libxsmm_timer_cycles(start, libxsmm_timer_tick()) / num_iterations,
    num_threads);

  libxsmm_barrier_destroy(barrier);

  return EXIT_SUCCESS;
}

