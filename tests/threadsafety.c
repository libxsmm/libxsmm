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
#include <string.h>
#include <stdio.h>

#if !defined(MAX_NKERNELS)
# if defined(LIBXSMM_REGSIZE)
#   define MAX_NKERNELS ((LIBXSMM_REGSIZE) * 2)
# else
#   define MAX_NKERNELS 100
# endif
#endif
#if !defined(USE_PARALLEL_JIT)
# define USE_PARALLEL_JIT
#endif
#if !defined(USE_VERBOSE)
# define USE_VERBOSE
#endif


int main(void)
{
  union { libxsmm_smmfunction s; void* p; } f[MAX_NKERNELS];
  const int max_shape = LIBXSMM_AVG_M;
  int result = EXIT_SUCCESS;
  int r[3*MAX_NKERNELS], i;

  for (i = 0; i < (3 * MAX_NKERNELS); i += 3) {
    r[i+0] = rand();
    r[i+1] = rand();
    r[i+2] = rand();
  }

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
    const libxsmm_blasint m = r[3*i+0] % max_shape + 1;
    const libxsmm_blasint n = r[3*i+1] % max_shape + 1;
    const libxsmm_blasint k = r[3*i+2] % max_shape + 1;
    f[i].s = libxsmm_smmdispatch(m, n, k,
      NULL/*lda*/, NULL/*ldb*/, NULL/*ldc*/, NULL/*alpha*/, NULL/*beta*/,
      NULL/*flags*/, NULL/*prefetch*/);
  }

#if defined(_OPENMP) && !defined(USE_PARALLEL_JIT)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < MAX_NKERNELS; ++i) {
    if (EXIT_SUCCESS == result) {
      const libxsmm_blasint m = r[3*i+0] % max_shape + 1;
      const libxsmm_blasint n = r[3*i+1] % max_shape + 1;
      const libxsmm_blasint k = r[3*i+2] % max_shape + 1;
      union { libxsmm_smmfunction s; void* p; } fi;
      fi.s = libxsmm_smmdispatch(m, n, k,
        NULL/*lda*/, NULL/*ldb*/, NULL/*ldc*/, NULL/*alpha*/, NULL/*beta*/,
        NULL/*flags*/, NULL/*prefetch*/);

      if (fi.p != f[i].p) {
        if (NULL != fi.p) {
          if (NULL != f[i].p) {
            const libxsmm_gemm_descriptor *const a = libxsmm_get_gemm_descriptor(fi.p);
            const libxsmm_gemm_descriptor *const b = libxsmm_get_gemm_descriptor(f[i].p);

            /* perform deeper check based on the descriptor of each of the kernels */
            if (0 != memcmp(a, b, LIBXSMM_GEMM_DESCRIPTOR_SIZE)) {
#if defined(_DEBUG) || defined(USE_VERBOSE)
              fprintf(stderr, "Error: the %ix%ix%i-kernel does not match!\n", m, n, k);
#endif
#if defined(_OPENMP) && !defined(USE_PARALLEL_JIT)
# if (201107 <= _OPENMP)
#             pragma omp atomic write
# else
#             pragma omp critical
# endif
#endif
              result = EXIT_FAILURE;
            }
#if defined(_DEBUG) || defined(USE_VERBOSE)
            else {
              fprintf(stderr, "(%ix%ix%i-kernel is duplicated)\n", m, n, k);
            }
#endif
          }
          else if (0 != LIBXSMM_JIT && 0 == libxsmm_get_dispatch_trylock()) {
#if defined(_DEBUG) || defined(USE_VERBOSE)
            fprintf(stderr, "Error: no code generated for %ix%ix%i-kernel!\n", m, n, k);
#endif
#if defined(_OPENMP) && !defined(USE_PARALLEL_JIT)
# if (201107 <= _OPENMP)
#           pragma omp atomic write
# else
#           pragma omp critical
# endif
#endif
            result = EXIT_FAILURE;
          }
        }
        else if (0 != LIBXSMM_JIT && 0 == libxsmm_get_dispatch_trylock()) {
#if defined(_DEBUG) || defined(USE_VERBOSE)
          fprintf(stderr, "Error: cannot find %ix%ix%i-kernel!\n", m, n, k);
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

