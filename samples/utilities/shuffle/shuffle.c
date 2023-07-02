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

#define SHUFFLE(INOUT, ELEMSIZE, COUNT) do { \
  char *const data = (char*)(INOUT); \
  size_t i = 0; \
  for (; i < ((COUNT) - 1); ++i) { \
    const size_t j = i + rand() / (RAND_MAX / ((COUNT) - i) + 1); \
    if (i != j) LIBXSMM_MEMSWP127( \
      data + (ELEMSIZE) * i, \
      data + (ELEMSIZE) * j, \
      ELEMSIZE); \
  } \
} while(0)


LIBXSMM_INLINE
void shuffle(void* inout, size_t elemsize, size_t count) {
  if (1 < count) {
    switch (elemsize) {
      case 8:   SHUFFLE(inout, 8, count); break;
      case 4:   SHUFFLE(inout, 4, count); break;
      case 2:   SHUFFLE(inout, 2, count); break;
      case 1:   SHUFFLE(inout, 1, count); break;
      default:  SHUFFLE(inout, elemsize, count);
    }
  }
}


int main(int argc, char* argv[])
{
  const int nelems = (1 < argc ? atoi(argv[1]) : 0);
  const int insize = (2 < argc ? atoi(argv[2]) : 0);
  const int niters = (3 < argc ? atoi(argv[3]) : 1);
  const int repeat = (4 < argc ? atoi(argv[4]) : 3);
  const int elsize = (0 >= insize ? 8 : insize);
  const size_t m = (0 <= niters ? niters : 1);
  const size_t n = (0 >= nelems
    ? (((size_t)64 << 20/*64 MB*/) / elsize)
    : ((size_t)nelems));
  const size_t m1 = LIBXSMM_MAX(m, 1);
  const size_t nbytes = n * elsize;
  void *const data1 = malloc(nbytes);
  void *const data2 = malloc(nbytes);
  int result = EXIT_SUCCESS;

  libxsmm_init();

  if (NULL != data1 && NULL != data2) {
    double d0, d1 = 0, d2 = 0, d3 = 0;
    int i = 0;
    size_t j;

    /* initialize the data */
    for (j = 0; j < nbytes; j += sizeof(int)) {
      const int r = rand();
      LIBXSMM_MEMCPY127((char*)data1 + j, &r,
        LIBXSMM_MIN(sizeof(int), nbytes - j));
    }

    for (; i <= repeat; ++i) {
      printf("-----------------------------------------\n");

      { /* benchmark RNG-based shuffle routine */
        const libxsmm_timer_tickint start = libxsmm_timer_tick();
        for (j = 0; j < m; ++j) shuffle(data1, elsize, n);
        d0 = libxsmm_timer_duration(start, libxsmm_timer_tick()) / m1;
        if (0 < d0) printf("RNG-shuffle:\t%.8f s (%i MB/s)\n", d0,
          (int)LIBXSMM_ROUND((2.0 * nbytes) / ((1024.0 * 1024.0) * d0)));
        if (0 < i) d1 += d0; /* ignore first iteration */
      }

      { /* benchmark DS1-based shuffle routine */
        const libxsmm_timer_tickint start = libxsmm_timer_tick();
        libxsmm_shuffle(data1, elsize, n, NULL, &m);
        d0 = libxsmm_timer_duration(start, libxsmm_timer_tick()) / m1;
        if (0 < d0) printf("DS1-shuffle:\t%.8f s (%i MB/s)\n", d0,
          (int)LIBXSMM_ROUND((2.0 * nbytes) / ((1024.0 * 1024.0) * d0)));
        if (0 < i) d2 += d0; /* ignore first iteration */
      }

      { /* benchmark DS2-based shuffle routine */
        const libxsmm_timer_tickint start = libxsmm_timer_tick();
        libxsmm_shuffle2(data2, data1, elsize, n, NULL, &m);
        d0 = libxsmm_timer_duration(start, libxsmm_timer_tick()) / m1;
        if (0 < d0) printf("DS2-shuffle:\t%.8f s (%i MB/s)\n", d0,
          (int)LIBXSMM_ROUND((2.0 * nbytes) / ((1024.0 * 1024.0) * d0)));
        if (0 < i) d3 += d0; /* ignore first iteration */
      }
    }

    if (1 < repeat) {
      printf("-----------------------------------------\n");
      printf("Arithmetic average of %i iterations\n", repeat);
      printf("-----------------------------------------\n");
      d1 /= repeat; d2 /= repeat; d3 /= repeat;
      if (0 < d1) printf("RNG-shuffle:\t%.8f s (%i MB/s)\n", d1,
        (int)LIBXSMM_ROUND((2.0 * nbytes) / ((1024.0 * 1024.0) * d1)));
      if (0 < d2) printf("DS1-shuffle:\t%.8f s (%i MB/s)\n", d2,
        (int)LIBXSMM_ROUND((2.0 * nbytes) / ((1024.0 * 1024.0) * d2)));
      if (0 < d3) printf("DS2-shuffle:\t%.8f s (%i MB/s)\n", d3,
        (int)LIBXSMM_ROUND((2.0 * nbytes) / ((1024.0 * 1024.0) * d3)));
    }
    if (0 < repeat) {
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
