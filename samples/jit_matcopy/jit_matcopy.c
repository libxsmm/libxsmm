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
/* Alexander Heinecke (Intel Corp.)
******************************************************************************/
#include <libxsmm.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>


int main(int argc, char* argv[])
{
  libxsmm_xmatcopyfunction skernel;
  libxsmm_matcopy_descriptor desc;
  float *a, *b;
  unsigned int i, j, iters;
  unsigned int ldi, ldo;
  int error = 0;
  double copy_time;
  unsigned long long l_start, l_end;

  printf("This is a tester for JIT matcopy kernels!\n");
  desc.m = atoi(argv[1]);
  desc.n = atoi(argv[2]);
  desc.ldi = atoi(argv[3]);
  desc.ldo = atoi(argv[4]);
  desc.unroll_level = (unsigned char)atoi(argv[5]);
  desc.typesize = 4;
  desc.prefetch = (unsigned char)atoi(argv[6]);;
  desc.flags = (unsigned char)(0 != atoi(argv[7]) ? LIBXSMM_MATCOPY_FLAG_ZERO_SOURCE : 0);
  iters = atoi(argv[8]);


  a = (float *) malloc(desc.m * desc.ldi * sizeof(float));
  b = (float *) malloc(desc.m * desc.ldo * sizeof(float));


  for (i=0; i < desc.m; i++ ) {
    for (j=0; j < desc.n; j++) {
      a[j+desc.ldi*i] = 1.f * rand();
      if (0 != (LIBXSMM_MATCOPY_FLAG_ZERO_SOURCE & desc.flags)) {
        b[j+desc.ldo*i] = 1.f * rand();
      }
    }
  }

  /* test dispatch call */
  skernel = libxsmm_xmatcopydispatch(&desc);

  if (skernel == 0) {
    printf("JIT error -> exit!!!!\n");
    exit(-1);
  }


  /* let's call */
  skernel(a, &ldi, b, &ldo, &a[128]);

  l_start = libxsmm_timer_tick();

  for (i=0; i<iters; i++) {
    skernel(a, &ldi, b, &ldo, &a[128]);
  }

  l_end = libxsmm_timer_tick();
  copy_time = libxsmm_timer_duration(l_start, l_end);

  for (i=0; i < desc.m; i++ ) {
    for (j=0; j < desc.n; j++) {
      if (0 != (LIBXSMM_MATCOPY_FLAG_ZERO_SOURCE & desc.flags)) {
        if (b[j+desc.ldo*i] > 0.00000000) {
          printf("ERROR!!!\n");
          error = 1;
        }
      }
      else if ( (a[j+desc.ldi*i] - b[j+desc.ldo*i]) > 0.000000001 ) {
        printf("ERROR!!!\n");
        error = 1;
      }
    }
  }

  if (error == 0) {
    printf("CORRECT copy!!!!\n");
    printf("Time taken is\t%.5f seconds\n",copy_time);
  }

  return 0;
}

