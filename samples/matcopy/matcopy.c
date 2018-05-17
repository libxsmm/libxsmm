/******************************************************************************
** Copyright (c) 2017-2018, Intel Corporation                                **
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
/* Alexander Heinecke, Hans Pabst (Intel Corp.)
******************************************************************************/
#include <libxsmm.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>


int main(int argc, char* argv[])
{
  const int m = (1 < argc ? atoi(argv[1]) : 16);
  const int n = (2 < argc ? atoi(argv[2]) : m);
  const int unsigned ldi = LIBXSMM_MAX(3 < argc ? atoi(argv[3]) : 0, m);
  const int unsigned ldo = LIBXSMM_MAX(4 < argc ? atoi(argv[4]) : 0, m);
  const int unroll = (5 < argc ? atoi(argv[5]) : 1);
  const int prefetch = (6 < argc ? atoi(argv[6]) : 0);
  const int flags = ((7 < argc && 0 != atoi(argv[7])) ? LIBXSMM_MATCOPY_FLAG_ZERO_SOURCE : 0);
  const int iters = (8 < argc ? atoi(argv[8]) : 1);

  /* we should modify to test all data-types */
  const libxsmm_mcopy_descriptor* desc;
  libxsmm_xmcopyfunction skernel;
  libxsmm_descriptor_blob blob;
  libxsmm_timer_tickint l_start;
  libxsmm_timer_tickint l_end;
  int error = 0, i, j;
  float *a, *b;
  double copy_time;

  printf("This is a tester for JIT matcopy kernels!\n");
  desc = libxsmm_mcopy_descriptor_init(&blob, sizeof(float),
    m, n, ldo, ldi, flags, prefetch, &unroll);

  a = (float*) malloc(n * ldi * sizeof(float));
  b = (float*) malloc(n * ldo * sizeof(float));

  for (i = 0; i < n; i++) {
    for (j = 0; j < m; j++) {
      a[j+ldi*i] = 1.f * rand();
      if (0 != (LIBXSMM_MATCOPY_FLAG_ZERO_SOURCE & flags)) {
        b[j+ldo*i] = 1.f * rand();
      }
    }
  }

  /* test dispatch call */
  skernel = libxsmm_dispatch_mcopy(desc);

  if (skernel == 0) {
    printf("JIT error -> exit!!!!\n");
    exit(EXIT_FAILURE);
  }

  /* let's call */
  skernel(a, &ldi, b, &ldo, &a[128]);

  l_start = libxsmm_timer_tick();

  for (i = 0; i<iters; i++) {
    skernel(a, &ldi, b, &ldo, &a[128]);
  }

  l_end = libxsmm_timer_tick();
  copy_time = libxsmm_timer_duration(l_start, l_end);

  for (i = 0; i < n; i++) {
    for (j = 0; j < m; j++) {
      if (0 != (LIBXSMM_MATCOPY_FLAG_ZERO_SOURCE & flags)) {
        if (LIBXSMM_NEQ(b[j+ldo*i], 0)) {
          printf("ERROR!!!\n");
          i = n;
          error = 1;
          break;
        }
      }
      else if (LIBXSMM_NEQ(a[j+ldi*i], b[j+ldo*i])) {
        printf("ERROR!!!\n");
        i = n;
        error = 1;
        break;
      }
    }
  }

  if (error == 0) {
    printf("CORRECT copy!!!!\n");
    printf("Time taken is\t%.5f seconds\n",copy_time);
  }

  return EXIT_SUCCESS;
}

