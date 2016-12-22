/******************************************************************************
** Copyright (c) 2015-2017, Intel Corporation                                **
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
#include <stdlib.h>
#if defined(_DEBUG)
# include <stdio.h>
#endif

#if !defined(MAX_NKERNELS)
# define MAX_NKERNELS 1000
#endif

#if !defined(USE_PARALLEL_JIT)
# define USE_PARALLEL_JIT
#endif


int main(void)
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
