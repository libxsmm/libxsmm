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
#if defined(__MKL)
# include <mkl_blas.h>
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
#if defined(_OPENMP)
  const int max_nthreads = omp_get_max_threads();
#else
  const int max_nthreads = 1;
#endif
  const int default_minsize = 4, default_maxsize = 16;
  const int size = LIBXSMM_MAX(1 < argc ? atoi(argv[1]) : 10000/*default*/, 1);
  const int nthreads = LIBXSMM_CLMP(2 < argc ? atoi(argv[2]) : 1/*default*/, 1, max_nthreads);
  const libxsmm_blasint maxksize = LIBXSMM_CLMP(3 < argc ? atoi(argv[3]) : default_maxsize, 1, LIBXSMM_MAX_M);
  const libxsmm_blasint minksize = LIBXSMM_CLMP(4 < argc ? atoi(argv[4]) : default_minsize, 1, maxksize);
  const libxsmm_blasint krange = maxksize - minksize;
  double tcall, tdsp0, tdsp1, tcgen;
  libxsmm_timer_tickint start;
  int result = EXIT_SUCCESS;

  printf("Dispatching %i calls %s internal synchronization using %i thread%s...\n", size,
#if (0 != LIBXSMM_SYNC)
    "with",
#else
    "without",
#endif
    1 >= nthreads ? 1 : nthreads,
    1 >= nthreads ? "" : "s");

#if 0 != LIBXSMM_JIT
  if (LIBXSMM_X86_SSE3 > libxsmm_get_target_archid()) {
    fprintf(stderr, "\tWarning: JIT support is not available at runtime!\n");
  }
#else
  fprintf(stderr, "\tWarning: JIT support has been disabled at build time!\n");
#endif

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload target(LIBXSMM_OFFLOAD_TARGET)
#endif
  {
    libxsmm_blasint *const rm = (libxsmm_blasint*)malloc(size * sizeof(libxsmm_blasint));
    libxsmm_blasint *const rn = (libxsmm_blasint*)malloc(size * sizeof(libxsmm_blasint));
    libxsmm_blasint *const rk = (libxsmm_blasint*)malloc(size * sizeof(libxsmm_blasint));
    const libxsmm_blasint m0 = (1 < krange ? ((default_maxsize % krange) + minksize) : minksize);
    const libxsmm_blasint n0 = (1 < krange ? ((default_maxsize % krange) + minksize) : minksize);
    const libxsmm_blasint k0 = (1 < krange ? ((default_maxsize % krange) + minksize) : minksize);
    const size_t shuffle = libxsmm_shuffle(size);
    const double alpha = 1, beta = 1;
    int i;

#if defined(mkl_jit_create_dgemm)
    void* *const jitter = malloc(size * sizeof(void*));
    if (NULL == jitter) exit(EXIT_FAILURE);
#else
    const int prefetch = LIBXSMM_GEMM_PREFETCH_NONE;
    const int flags = LIBXSMM_GEMM_FLAG_NONE;
#endif
    if (NULL == rm && NULL == rn && NULL == rk) exit(EXIT_FAILURE);

    /* generate a set of random numbers outside of any parallel region */
    for (i = 0; i < size; ++i) {
      rm[i] = (1 < krange ? ((rand() % krange) + minksize) : minksize);
      rn[i] = (1 < krange ? ((rand() % krange) + minksize) : minksize);
      rk[i] = (1 < krange ? ((rand() % krange) + minksize) : minksize);
#if defined(mkl_jit_create_dgemm)
      jitter[i] = NULL;
#endif
    }

    /* first invocation may initialize some internals */
    libxsmm_init(); /* subsequent calls are not doing any work */
    start = libxsmm_timer_tick();
    for (i = 0; i < size; ++i) {
      /* measure call overhead of an "empty" function (not inlined) */
      libxsmm_init();
    }
    tcall = libxsmm_timer_duration(start, libxsmm_timer_tick());

    { /* trigger code generation to subsequently measure only dispatch time */
#if defined(mkl_jit_create_dgemm)
      LIBXSMM_EXPECT(MKL_JIT_SUCCESS, mkl_cblas_jit_create_dgemm(jitter,
        MKL_COL_MAJOR, MKL_NOTRANS/*transa*/, MKL_NOTRANS/*transb*/,
        m0, n0, k0, alpha, m0, k0, beta, m0));
      LIBXSMM_EXPECT_NOT(NULL, mkl_jit_get_dgemm_ptr(jitter[0])); /* to measure "cached" lookup time (below) */
#else
      LIBXSMM_EXPECT_NOT(NULL, libxsmm_dmmdispatch(m0, n0, k0, &m0, &k0, &m0, &alpha, &beta, &flags, &prefetch));
#endif
    }

    /* measure dispatching previously generated (likely cached) */
#if defined(_OPENMP)
#   pragma omp parallel num_threads(nthreads) private(i)
    {
#     pragma omp single
      start = libxsmm_timer_tick();
#     pragma omp for
#else
    {
      start = libxsmm_timer_tick();
#endif
      for (i = 0; i < size; ++i) {
#if defined(mkl_jit_create_dgemm)
        mkl_jit_get_dgemm_ptr(jitter[0]); /* no "dispatch" just unwrapping the jitter */
#else
        libxsmm_dmmdispatch(m0, n0, k0, &m0, &k0, &m0, &alpha, &beta, &flags, &prefetch);
#endif
      }
    }
    tdsp1 = libxsmm_timer_duration(start, libxsmm_timer_tick());
#if defined(mkl_jit_create_dgemm)
    mkl_jit_destroy(jitter[0]); jitter[0] = NULL;
#endif

    /* measure generating JIT-kernels (randomized parameterization) */
#if defined(_OPENMP)
#   pragma omp parallel num_threads(nthreads) private(i)
    {
#     pragma omp single
      start = libxsmm_timer_tick();
#     pragma omp for
#else
    {
      start = libxsmm_timer_tick();
#endif
      for (i = 1; i < size; ++i) {
#if defined(mkl_jit_create_dgemm)
        LIBXSMM_EXPECT(MKL_JIT_SUCCESS, mkl_cblas_jit_create_dgemm(jitter + i,
          MKL_COL_MAJOR, MKL_NOTRANS/*transa*/, MKL_NOTRANS/*transb*/,
          rm[i], rn[i], rk[i], alpha, rm[i], rk[i], beta, rm[i]));
        LIBXSMM_EXPECT_NOT(NULL, mkl_jit_get_dgemm_ptr(jitter[i]));
#else
        LIBXSMM_EXPECT_NOT(NULL, libxsmm_dmmdispatch(rm[i], rn[i], rk[i],
          rm + i, rk + i, rm + i, &alpha, &beta, &flags, &prefetch));
#endif
      }
    }
    tcgen = libxsmm_timer_duration(start, libxsmm_timer_tick());

    /* measure dispatching previously generated kernel (likely non-cached) */
#if defined(_OPENMP)
#   pragma omp parallel num_threads(nthreads) private(i)
    {
#     pragma omp single
      start = libxsmm_timer_tick();
#     pragma omp for
#else
    {
      start = libxsmm_timer_tick();
#endif
      for (i = 0; i < size; ++i) {
        const int j = (int)((shuffle * i) % size);
#if defined(mkl_jit_create_dgemm)
        LIBXSMM_EXPECT_NOT(NULL, mkl_jit_get_dgemm_ptr(jitter[j]));
#else
        LIBXSMM_EXPECT_NOT(NULL, libxsmm_dmmdispatch(rm[j], rn[j], rk[j],
          rm + j, rk + j, rm + j, &alpha, &beta, &flags, &prefetch));
#endif
      }
    }
    tdsp0 = libxsmm_timer_duration(start, libxsmm_timer_tick());

    /* release random numbers */
    free(rm); free(rn); free(rk);
#if defined(mkl_jit_create_dgemm) /* release dispatched code */
    for (i = 0; i < size; ++i) mkl_jit_destroy(jitter[i]);
    free(jitter); /* release array used to store dispatched code */
#endif
  }

  if (1 < size) {
    tcall /= size; tdsp0 /= size; tdsp1 /= size;
#if !defined(mkl_jit_create_dgemm)
    { /* correct for duplicated code generation requests */
      libxsmm_registry_info reginfo;
      if (EXIT_SUCCESS == libxsmm_get_registry_info(&reginfo)) {
        const int ngen = (int)reginfo.size;
        LIBXSMM_ASSERT(1 < ngen);
        tcgen -= tdsp0 * (size - ngen);
        tcgen /= ngen - 1;
      }
    }
#else
    tcgen /= size - 1;
#endif
    if (0 < tcall && 0 < tdsp0 && 0 < tdsp1 && 0 < tcgen) {
      printf("\tfunction-call (empty): %.0f ns (%.0f MHz)\n", 1E9 * tcall, 1E-6 / tcall);
      printf("\tdispatch (ro/cached): %.0f ns (%.0f MHz)\n", 1E9 * tdsp1, 1E-6 / tdsp1);
      printf("\tdispatch (ro): %.0f ns (%.0f MHz)\n", 1E9 * tdsp0, 1E-6 / tdsp0);
      if (1E-6 <= tcgen) {
        printf("\tcode-gen (rw): %.0f us (%.0f kHz)\n", 1E6 * tcgen, 1E-3 / tcgen);
      }
      else {
        printf("\tcode-gen (rw): %.0f ns (%.0f MHz)\n", 1E9 * tcgen, 1E-6 / tcgen);
      }
    }
  }
  printf("Finished\n");

  return result;
}

