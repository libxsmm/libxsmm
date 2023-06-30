/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Hans Pabst (Intel Corp.)
******************************************************************************/
#include <utils/libxsmm_utils.h>
#include <libxsmm.h>


LIBXSMM_INLINE
void shuffle(void* inout, size_t elemsize, size_t count) {
  if (1 < count) {
    char *const data = (char*)inout;
    size_t i = 0;
    for (; i < (count - 1); ++i) {
      const size_t j = i + rand() / (RAND_MAX / (count - i) + 1);
      if (i != j) LIBXSMM_MEMSWP127(data + i * elemsize, data + j * elemsize, elemsize);
    }
  }
}


int main(int argc, char* argv[])
{
  const int nelems = (1 < argc ? atoi(argv[1]) : 0);
  const int insize = (2 < argc ? atoi(argv[2]) : 0);
  const int repeat = (3 < argc ? atoi(argv[3]) : 1);
  const int niters = (4 < argc ? atoi(argv[4]) : 3);
  const int elsize = (0 >= insize ? 8 : insize);
  const size_t m = (0 <= repeat ? repeat : 1);
  const size_t n = (0 >= nelems
    ? (((size_t)64 << 20/*64 MB*/) / elsize)
    : ((size_t)nelems));
  const size_t nbytes = n * elsize;
  void *const data1 = malloc(nbytes);
  void *const data2 = malloc(nbytes);
  int result = EXIT_SUCCESS;

  libxsmm_init();

  if (NULL != data1 && NULL != data2) {
    double d0, d1 = 0, d2 = 0, d3 = 0;
    int i = 0;
    size_t j;

    for (; i <= niters; ++i) {
      printf("-----------------------------------------\n");
      /* initialize the data */
      for (j = 0; j < nbytes; j += sizeof(int)) {
        const int r = rand();
        LIBXSMM_MEMCPY127((char*)data1 + j, &r,
          LIBXSMM_MIN(sizeof(int), nbytes - j));
      }

      { /* benchmark RNG-based shuffle routine */
        const libxsmm_timer_tickint start = libxsmm_timer_tick();
        for (j = 0; j < m; ++j) shuffle(data1, elsize, n);
        d0 = libxsmm_timer_duration(start, libxsmm_timer_tick());
        if (0 < d0) printf("RNG-shuffle:\t%.8f s (%i MB/s)\n", d0,
          (int)LIBXSMM_ROUND((2.0 * nbytes) / ((1024.0 * 1024.0) * d0)));
        if (0 < i) d1 += d0; /* ignore first iteration */
      }

      { /* benchmark COP-based shuffle routine */
        const libxsmm_timer_tickint start = libxsmm_timer_tick();
        libxsmm_shuffle(data1, elsize, n, NULL, &m);
        d0 = libxsmm_timer_duration(start, libxsmm_timer_tick());
        if (0 < d0) printf("COP-shuffle:\t%.8f s (%i MB/s)\n", d0,
          (int)LIBXSMM_ROUND((2.0 * nbytes) / ((1024.0 * 1024.0) * d0)));
        if (0 < i) d2 += d0; /* ignore first iteration */
      }

      { /* benchmark CO2-based shuffle routine */
        const libxsmm_timer_tickint start = libxsmm_timer_tick();
        libxsmm_shuffle2(data2, data1, elsize, n, NULL, &m);
        d0 = libxsmm_timer_duration(start, libxsmm_timer_tick());
        if (0 < d0) printf("COP-shuffle:\t%.8f s (%i MB/s)\n", d0,
          (int)LIBXSMM_ROUND((2.0 * nbytes) / ((1024.0 * 1024.0) * d0)));
        if (0 < i) d3 += d0; /* ignore first iteration */
      }
    }

    if (1 < niters) {
      printf("-----------------------------------------\n");
      printf("Arithmetic average of %i iterations\n", niters);
      printf("-----------------------------------------\n");
      d1 /= niters; d2 /= niters; d3 /= niters;
      if (0 < d1) printf("RNG-shuffle:\t%.8f s (%i MB/s)\n", d1,
        (int)LIBXSMM_ROUND((2.0 * nbytes) / ((1024.0 * 1024.0) * d1)));
      if (0 < d2) printf("COP-shuffle:\t%.8f s (%i MB/s)\n", d2,
        (int)LIBXSMM_ROUND((2.0 * nbytes) / ((1024.0 * 1024.0) * d2)));
      if (0 < d3) printf("CO2-shuffle:\t%.8f s (%i MB/s)\n", d3,
        (int)LIBXSMM_ROUND((2.0 * nbytes) / ((1024.0 * 1024.0) * d3)));
    }
    if (0 < niters) {
      printf("-----------------------------------------\n");
    }
  }
  else {
    result = EXIT_FAILURE;
  }

  free(data1);
  free(data2);

  return result;
}
