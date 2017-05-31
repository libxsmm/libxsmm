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
# define MAX_NKERNELS 1000
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
  const char *const target_arch = libxsmm_get_target_arch();
  libxsmm_generated_code generated_code;
  libxsmm_registry_info registry_info;
  const int prefetch = LIBXSMM_PREFETCH_NONE;
  const int max_shape = LIBXSMM_AVG_M;
  const int flags = LIBXSMM_FLAGS;
  int nkernels = MAX_NKERNELS;
  int result = EXIT_SUCCESS;
  int r[3*MAX_NKERNELS], i;

  memset(&generated_code, 0, sizeof(generated_code));
  generated_code.generated_code = malloc(131072);
  generated_code.buffer_size = 0 != generated_code.generated_code ? 131072 : 0;
  generated_code.code_type = 2;

  /* generate set of random number for parallel region */
  for (i = 0; i < (3 * MAX_NKERNELS); i += 3) {
    r[i+0] = rand();
    r[i+1] = rand();
    r[i+2] = rand();
  }

#if defined(_OPENMP)
# pragma omp parallel for default(none) private(i)
#endif
  for (i = 0; i < MAX_NKERNELS; ++i) {
    if (0 == (i % 2)) {
      libxsmm_init();
    }
    else {
      libxsmm_finalize();
    }
  }
  libxsmm_init();

  result = libxsmm_get_registry_info(&registry_info);
  if (EXIT_SUCCESS == result) {
    nkernels = (int)LIBXSMM_MIN((size_t)nkernels, registry_info.capacity);
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
      &flags, &prefetch);
  }

#if defined(_OPENMP) && !defined(USE_PARALLEL_JIT)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < nkernels; ++i) {
    if (EXIT_SUCCESS == result) {
      const libxsmm_blasint m = r[3*i+0] % max_shape + 1;
      const libxsmm_blasint n = r[3*i+1] % max_shape + 1;
      const libxsmm_blasint k = r[3*i+2] % max_shape + 1;
      union { libxsmm_xmmfunction x; void* p; } fi;
      LIBXSMM_GEMM_DESCRIPTOR_TYPE(descriptor, LIBXSMM_GEMM_PRECISION(float), flags,
        m, n, k, m/*lda*/, k/*ldb*/, m/*ldc*/, LIBXSMM_ALPHA, LIBXSMM_BETA, prefetch);
      fi.x = libxsmm_xmmdispatch(&descriptor);

      if (fi.p != f[i].p) {
        if (NULL != fi.p) {
          if (NULL != f[i].p) {
            generated_code.code_size = 0; /* reset size; avoid stitching code */
            libxsmm_generator_gemm_kernel(&generated_code, &descriptor, target_arch);

            if (0 == generated_code.last_error && 0 < generated_code.code_size) {
              /* perform deeper check based on another code generation (used as reference) */
              if  (0 == registry_info.nstatic &&
                  (0 != memcmp(generated_code.generated_code, fi.p, generated_code.code_size)
                || 0 != memcmp(generated_code.generated_code, f[i].p, generated_code.code_size)))
              {
#if defined(_DEBUG) || defined(USE_VERBOSE)
                fprintf(stderr, "Error: the %ix%ix%i-kernel does not match!\n", m, n, k);
#endif
#if defined(_OPENMP) && !defined(USE_PARALLEL_JIT)
# if (201107 <= _OPENMP)
#               pragma omp atomic write
# else
#               pragma omp critical
# endif
#endif
                result = EXIT_FAILURE;
              }
#if defined(_DEBUG) /* warning or an info message is not part of USE_VERBOSE */
              else if (0 != registry_info.nstatic) {
                fprintf(stderr, "Warning: the %ix%ix%i-kernel may not match!\n", m, n, k);
              }
              else {
                fprintf(stderr, "(%ix%ix%i-kernel is duplicated)\n", m, n, k);
              }
#endif
            }
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

  /* release buffer of eventually generated code (deep comparison) */
  free(generated_code.generated_code);
  libxsmm_finalize();

  return result;
}

