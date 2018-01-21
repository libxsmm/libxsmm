/******************************************************************************
** Copyright (c) 2016-2018, Intel Corporation                                **
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
  const int size = LIBXSMM_DEFAULT(10000, 1 < argc ? atoi(argv[1]) : 0);
  const int nthreads = LIBXSMM_DEFAULT(1, 2 < argc ? atoi(argv[2]) : 0);
  libxsmm_timer_tickint tdisp = 0, tcgen = 0, tcall, start;

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
    libxsmm_blasint *const r = (libxsmm_blasint*)malloc(3/*m,n,k*/ * size * sizeof(libxsmm_blasint));
    libxsmm_registry_info reginfo;
    int i;

    assert(0 != r);
    /* generate a set of random numbers outside of any parallel region */
    for (i = 0; i < (3/*m,n,k*/ * size); ++i) r[i] = rand();

    /* run non-inline function to measure call overhead of an "empty" function */
    start = libxsmm_timer_tick();
    for (i = 0; i < size; ++i) {
      libxsmm_init(); /* subsequent calls are not doing any work */
    }
    tcall = libxsmm_timer_diff(start, libxsmm_timer_tick());

    /* first invocation may initialize some internals (libxsmm_init),
     * or actually generate code (code gen. time is out of scope)
     */
    libxsmm_dmmdispatch(23, 23, 23,
      NULL/*lda*/, NULL/*ldb*/, NULL/*ldc*/, NULL/*alpha*/, NULL/*beta*/,
      NULL/*flags*/, NULL/*prefetch*/);

#if defined(_OPENMP)
#   pragma omp parallel for default(none) private(i)
#endif
    for (i = 0; i < size; ++i) {
      const libxsmm_timer_tickint t0 = libxsmm_timer_tick();
      libxsmm_dmmdispatch(23, 23, 23,
        NULL/*lda*/, NULL/*ldb*/, NULL/*ldc*/, NULL/*alpha*/, NULL/*beta*/,
        NULL/*flags*/, NULL/*prefetch*/);
      tdisp += libxsmm_timer_diff(t0, libxsmm_timer_tick());
    }

#if defined(_OPENMP)
#   pragma omp parallel for default(none) private(i)
#endif
    for (i = 0; i < size; ++i) {
      const int j = 3 * i;
      const libxsmm_timer_tickint t0 = libxsmm_timer_tick();
      libxsmm_dmmdispatch(r[j], r[j+1], r[j+2],
        NULL/*lda*/, NULL/*ldb*/, NULL/*ldc*/, NULL/*alpha*/, NULL/*beta*/,
        NULL/*flags*/, NULL/*prefetch*/);
      tcgen += libxsmm_timer_diff(t0, libxsmm_timer_tick());
    }

    /* correct for duplicated code generation requests */
    if (EXIT_SUCCESS == libxsmm_get_registry_info(&reginfo)) {
      tcgen -= tcall * (reginfo.size - size - 1/*initial code gen.*/);
    }

    free(r);
  }

  if (0 < size) {
    const double dcall = libxsmm_timer_duration(0, tcall) / size;
    const double ddisp = libxsmm_timer_duration(0, tdisp) / size;
    const double dcgen = libxsmm_timer_duration(0, tcgen) / size;
    if (0 < tcall && 0 < tdisp && 0 < tcgen) {
      fprintf(stdout, "\tempty fn: %.0f ns (%.0f MHz)\n", 1E9 * dcall, 1E-6 / dcall);
      fprintf(stdout, "\tdispatch: %.0f ns (%.0f MHz)\n", 1E9 * ddisp, 1E-6 / ddisp);
      fprintf(stdout, "\tcode gen: %.0f us (%.0f kHz)\n", 1E6 * dcgen, 1E-3 / dcgen);
    }
  }
  fprintf(stdout, "Finished\n");

  return EXIT_SUCCESS;
}

