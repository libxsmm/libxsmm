/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Heinecke, Hans Pabst (Intel Corp.)
******************************************************************************/
#include <libxsmm.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#if !defined(ELEM_TYPE)
# define ELEM_TYPE float
#endif


int main(int argc, char* argv[])
{
  const unsigned int m = (unsigned int)LIBXSMM_MAX(1 < argc ? atoi(argv[1]) : 16, 0);
  const unsigned int n = (unsigned int)LIBXSMM_MAX(2 < argc ? atoi(argv[2]) : 0, (int)m);
  const unsigned int ldi = (unsigned int)LIBXSMM_MAX(3 < argc ? atoi(argv[3]) : 0, (int)m);
  const unsigned int ldo = (unsigned int)LIBXSMM_MAX(4 < argc ? atoi(argv[4]) : 0, (int)m);
  const int unroll = (5 < argc ? atoi(argv[5]) : 1), prefetch = (6 < argc ? atoi(argv[6]) : 0);
  const int flags = ((7 < argc && 0 != atoi(argv[7])) ? LIBXSMM_MATCOPY_FLAG_ZERO_SOURCE : 0);
  const unsigned int iters = (unsigned int)LIBXSMM_MAX(8 < argc ? atoi(argv[8]) : 0, 1);

  /* we should modify to test all data-types */
  const libxsmm_mcopy_descriptor* desc;
  libxsmm_xmcopyfunction kernel;
  libxsmm_descriptor_blob blob;
  libxsmm_timer_tickint l_start;
  libxsmm_timer_tickint l_end;
  unsigned int error = 0, i, j;
  ELEM_TYPE *a, *b;
  double copy_time;

  libxsmm_init();
  printf("This is a tester for JIT matcopy kernels!\n");
  desc = libxsmm_mcopy_descriptor_init(&blob, sizeof(ELEM_TYPE),
    m, n, ldo, ldi, flags, prefetch, &unroll);

  a = (ELEM_TYPE*)((0 < n && 0 < ldi) ? malloc(sizeof(ELEM_TYPE) * n * ldi) : NULL);
  b = (ELEM_TYPE*)((0 < n && 0 < ldo) ? malloc(sizeof(ELEM_TYPE) * n * ldo) : NULL);
  if (NULL == a || NULL == b) {
    printf("buffer allocation failed!\n");
    free(a); free(b);
    exit(EXIT_FAILURE);
  }
  assert(NULL != a && NULL != b);

  for (i = 0; i < n; ++i) {
    for (j = 0; j < ldi; ++j) {
      a[j+ldi*i] = (ELEM_TYPE)rand();
      if (0 != (LIBXSMM_MATCOPY_FLAG_ZERO_SOURCE & flags) && j < m) {
        b[j+ldo*i] = (ELEM_TYPE)rand();
      }
    }
    for (j = m; j < ldo; ++j) {
      b[j+ldo*i] = (ELEM_TYPE)0xCD;
    }
  }

  /* test dispatch call */
  kernel = libxsmm_dispatch_mcopy(desc);
  if (kernel == 0) {
    printf("JIT error -> exit!!!!\n");
    exit(EXIT_FAILURE);
  }

  /* let's call */
  kernel(a, &ldi, b, &ldo, &a[128]);

  l_start = libxsmm_timer_tick();
  for (i = 0; i < iters; ++i) {
    kernel(a, &ldi, b, &ldo, &a[128]);
  }
  l_end = libxsmm_timer_tick();
  copy_time = libxsmm_timer_duration(l_start, l_end);

  for (i = 0; i < n; ++i) {
    for (j = 0; j < m; ++j) {
      if (0 != (LIBXSMM_MATCOPY_FLAG_ZERO_SOURCE & flags)) {
        if (LIBXSMM_NEQ(0, b[j+ldo*i])) {
          printf("ERROR!!!\n");
          error = 1; i = n; break;
        }
      }
      else if (LIBXSMM_NEQ(a[j+ldi*i], b[j+ldo*i])) {
        printf("ERROR!!!\n");
        error = 1; i = n; break;
      }
    }
    for (j = m; j < ldo; ++j) {
      if (LIBXSMM_NEQ((ELEM_TYPE)0xCD, b[j+ldo*i])) {
        printf("ERROR!!!\n");
        error = 1; i = n; break;
      }
    }
  }

  if (error == 0) {
    printf("CORRECT copy!!!!\n");
    printf("Time taken is\t%.5f seconds\n", copy_time);
    return EXIT_SUCCESS;
  }
  else return EXIT_FAILURE;
}

