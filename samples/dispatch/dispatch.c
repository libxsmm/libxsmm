/******************************************************************************
** Copyright (c) 2016-2017, Intel Corporation                                **
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


/**
 * This (micro-)benchmark optionally takes a number of dispatches to be performed.
 * The program measures the duration needed to figure out whether a requested matrix
 * multiplication is available or not. The measured duration excludes the time taken
 * to actually generate the code during the first dispatch.
 */
int main(int argc, char* argv[])
{
  const int size = LIBXSMM_DEFAULT(10000000, 1 < argc ? atoi(argv[1]) : 0);
  const int nthreads = LIBXSMM_DEFAULT(1, 2 < argc ? atoi(argv[2]) : 0);
  unsigned long long start;
  double dcall, ddisp;
  int i;

#if defined(_OPENMP)
  if (0 < nthreads) omp_set_num_threads(nthreads);
#endif

  fprintf(stdout, "Dispatching %i calls %s internal synchronization using %i thread%s...\n", size,
#if 0 != LIBXSMM_SYNC
    "with",
#else
    "without",
#endif
    1 >= nthreads ? 1 : nthreads,
    1 >= nthreads ? "" : "s");

#if 0 != LIBXSMM_JIT
  if (LIBXSMM_X86_AVX > libxsmm_get_target_archid()) {
    fprintf(stderr, "\tWarning: JIT support is not available at runtime!\n");
  }
#else
  fprintf(stderr, "\tWarning: JIT support has been disabled at build time!\n");
#endif

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload target(LIBXSMM_OFFLOAD_TARGET)
#endif
  {
    /* first invocation may initialize some internals (libxsmm_init),
     * or actually generate code (code gen. time is out of scope)
     */
    libxsmm_dmmdispatch(23, 23, 23,
      NULL/*lda*/, NULL/*ldb*/, NULL/*ldc*/, NULL/*alpha*/, NULL/*beta*/,
      NULL/*flags*/, NULL/*prefetch*/);

    /* run non-inline function to measure call overhead of an "empty" function */
    start = libxsmm_timer_tick();
#if defined(_OPENMP)
#   pragma omp parallel for default(none) private(i)
#endif
    for (i = 0; i < size; ++i) {
      libxsmm_init(); /* subsequent calls are not doing any work */
    }
    dcall = libxsmm_timer_duration(start, libxsmm_timer_tick());

    start = libxsmm_timer_tick();
#if defined(_OPENMP)
#   pragma omp parallel for default(none) private(i)
#endif
    for (i = 0; i < size; ++i) {
      libxsmm_dmmdispatch(23, 23, 23,
        NULL/*lda*/, NULL/*ldb*/, NULL/*ldc*/, NULL/*alpha*/, NULL/*beta*/,
        NULL/*flags*/, NULL/*prefetch*/);
    }
    ddisp = libxsmm_timer_duration(start, libxsmm_timer_tick());
  }

  if (0 < dcall && 0 < ddisp) {
    fprintf(stdout, "\tdispatch calls/s: %.1f MHz\n", 1E-6 * size / ddisp);
    fprintf(stdout, "\tempty calls/s: %.1f MHz\n", 1E-6 * size / dcall);
    fprintf(stdout, "\toverhead: %.1fx\n", ddisp / dcall);
  }
  fprintf(stdout, "Finished\n");

  return EXIT_SUCCESS;
}

