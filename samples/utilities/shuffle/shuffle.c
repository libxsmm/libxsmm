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

/* Fisher-Yates shuffle */
#define SHUFFLE(INOUT, ELEMSIZE, COUNT) do { \
  char *const data = (char*)(INOUT); \
  size_t i = 0; \
  for (; i < ((COUNT) - 1); ++i) { \
    const size_t j = i + libxsmm_rng_u32((unsigned int)((COUNT) - i)); \
    LIBXSMM_ASSERT(i <= j && j < (COUNT)); \
    if (i != j) LIBXSMM_MEMSWP127( \
      data + (ELEMSIZE) * i, \
      data + (ELEMSIZE) * j, \
      ELEMSIZE); \
  } \
} while(0)

#define REDUCE_ADD(TYPE, INPUT, COUNT, LO, HI) do { \
  const TYPE *const data = (const TYPE*)(INPUT); \
  const size_t locount = (COUNT) / 2 + ((COUNT) & 1); \
  size_t i = 0; \
  for ((LO) = 0ULL; i < locount; ++i) (LO) += data[i]; \
  for ((HI) = 0ULL; i < (COUNT); ++i) (HI) += data[i]; \
} while(0)


void shuffle(void* inout, size_t elemsize, size_t count);
size_t imbalance_uint(const void* input, size_t elemsize, size_t count, size_t split);
size_t bsort_uint_asc(void* inout, size_t elemsize, size_t count);


int main(int argc, char* argv[])
{
  const int nelems = (1 < argc ? atoi(argv[1]) : 0);
  const int insize = (2 < argc ? atoi(argv[2]) : 0);
  const int niters = (3 < argc ? atoi(argv[3]) : 1);
  const int repeat = (4 < argc ? atoi(argv[4]) : 3);
  const int elsize = (0 >= insize ? 8 : insize);
  const int random = (NULL == getenv("RANDOM")
    ? 0 : atoi(getenv("RANDOM")));
  const int split = (0 < random ? random : 1);
  const size_t m = (0 <= niters ? niters : 1);
  const size_t n = (0 >= nelems
    ? (((size_t)64 << 20/*64 MB*/) / elsize)
    : ((size_t)nelems));
  const size_t mm = LIBXSMM_MAX(m, 1);
  const size_t nbytes = n * elsize;
  void *const data1 = malloc(nbytes);
  void *const data2 = malloc(nbytes);
  int result = EXIT_SUCCESS;

  libxsmm_init();

  if (NULL != data1 && NULL != data2) {
    size_t a1 = 0, a2 = 0, a3 = 0, b1 = 0, b2 = 0, b3 = 0;
    size_t n1 = 0, n2 = 0, n3 = 0, j;
    double d0, d1 = 0, d2 = 0, d3 = 0;
    libxsmm_timer_tickint start;
    int i = 0;

    /* initialize the data */
    if (sizeof(size_t) < elsize) memset(data1, 0, nbytes);
    for (j = 0; j < n; ++j) {
      LIBXSMM_MEMCPY127((char*)data1 + elsize * j, &j,
        LIBXSMM_MIN(elsize, sizeof(size_t)));
    }

    for (; i <= repeat; ++i) {
      printf("---------------------------------------\n");

      { /* benchmark RNG-based shuffle routine */
        memcpy(data2, data1, nbytes);
        start = libxsmm_timer_tick();
        for (j = 0; j < m; ++j) shuffle(data2, elsize, n);
        d0 = libxsmm_timer_duration(start, libxsmm_timer_tick()) / mm;
        if (0 != random && i == repeat) {
          a1 = imbalance_uint(data2, elsize, n, split);
          b1 = imbalance_uint(data2, elsize, n, split * 2);
          n1 = bsort_uint_asc(data2, elsize, n);
        }
        if (0 < d0) printf("RNG-shuffle: %.8f s (%i MB/s)\n", d0,
          (int)LIBXSMM_ROUND((2.0 * nbytes) / ((1024.0 * 1024.0) * d0)));
        if (0 < i) d1 += d0; /* ignore first iteration */
      }

      { /* benchmark DS1-based shuffle routine */
        memcpy(data2, data1, nbytes);
        start = libxsmm_timer_tick();
        libxsmm_shuffle(data2, elsize, n, NULL, &m);
        d0 = libxsmm_timer_duration(start, libxsmm_timer_tick()) / mm;
        if (0 != random && i == repeat) {
          a2 = imbalance_uint(data2, elsize, n, split);
          b2 = imbalance_uint(data2, elsize, n, split * 2);
          n2 = bsort_uint_asc(data2, elsize, n);
        }
        if (0 < d0) printf("DS1-shuffle: %.8f s (%i MB/s)\n", d0,
          (int)LIBXSMM_ROUND((2.0 * nbytes) / ((1024.0 * 1024.0) * d0)));
        if (0 < i) d2 += d0; /* ignore first iteration */
      }

      { /* benchmark DS2-based shuffle routine */
        start = libxsmm_timer_tick();
        libxsmm_shuffle2(data2, data1, elsize, n, NULL, &m);
        d0 = libxsmm_timer_duration(start, libxsmm_timer_tick()) / mm;
        if (0 != random && i == repeat) {
          a3 = imbalance_uint(data2, elsize, n, split);
          b3 = imbalance_uint(data2, elsize, n, split * 2);
          n3 = bsort_uint_asc(data2, elsize, n);
        }
        if (0 < d0) printf("DS2-shuffle: %.8f s (%i MB/s)\n", d0,
          (int)LIBXSMM_ROUND((2.0 * nbytes) / ((1024.0 * 1024.0) * d0)));
        if (0 < i) d3 += d0; /* ignore first iteration */
      }
    }

    if (1 < repeat) {
      const unsigned long long nn = (n * n - n + 3) / 4;
      printf("---------------------------------------\n");
      printf("Arithmetic average of %i iterations\n", repeat);
      printf("---------------------------------------\n");
      d1 /= repeat; d2 /= repeat; d3 /= repeat;
      if (0 < d1) {
        printf("RNG-shuffle: %.8f s (%i MB/s)\n", d1,
          (int)LIBXSMM_ROUND((2.0 * nbytes) / ((1024.0 * 1024.0) * d1)));
        if (0 != random) {
          printf("             rand=%llu%% imb%i=%llu%% imb%i=%llu%%\n",
            (LIBXSMM_MIN(n1, nn) * 100 + nn - 1) / nn,
            split * 2, (b1 * 100 + n - 1) / n,
            split, (a1 * 100 + n - 1) / n);
        }
      }
      if (0 < d2) {
        printf("DS1-shuffle: %.8f s (%i MB/s)\n", d2,
          (int)LIBXSMM_ROUND((2.0 * nbytes) / ((1024.0 * 1024.0) * d2)));
        if (0 != random) {
          printf("             rand=%llu%% imb%i=%llu%% imb%i=%llu%%\n",
            (LIBXSMM_MIN(n2, nn) * 100 + nn - 1) / nn,
            split * 2, (b2 * 100 + n - 1) / n,
            split, (a2 * 100 + n - 1) / n);
        }
      }
      if (0 < d3) {
        printf("DS2-shuffle: %.8f s (%i MB/s)\n", d3,
          (int)LIBXSMM_ROUND((2.0 * nbytes) / ((1024.0 * 1024.0) * d3)));
        if (0 != random) {
          printf("             rand=%llu%% imb%i=%llu%% imb%i=%llu%%\n",
            (LIBXSMM_MIN(n3, nn) * 100 + nn - 1) / nn,
            split * 2, (b3 * 100 + n - 1) / n,
            split, (a3 * 100 + n - 1) / n);
        }
      }
    }
    if (0 < repeat) {
      printf("---------------------------------------\n");
    }
  }
  else {
    result = EXIT_FAILURE;
  }

  free(data1);
  free(data2);

  return result;
}


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


size_t imbalance_uint_aux(const void* input, size_t elemsize, size_t count, size_t split);
size_t imbalance_uint_aux(const void* input, size_t elemsize, size_t count, size_t split)
{
  const size_t s = count / 2 + (count & 1);
  unsigned long long n, lo, hi;
  switch (elemsize) {
    case 8: {
      REDUCE_ADD(unsigned long long, input, s, lo, hi);
    } break;
    case 4: {
      REDUCE_ADD(unsigned int, input, s, lo, hi);
    } break;
    case 2: {
      REDUCE_ADD(unsigned short, input, s, lo, hi);
    } break;
    default: {
      REDUCE_ADD(unsigned char, input, elemsize * s, lo, hi);
    }
  }
  if (1 < split) {
    unsigned long long a, b;
    a = imbalance_uint_aux(input, elemsize, s, split - 1);
    b = imbalance_uint_aux((const char*)input + elemsize * s, elemsize, count - s, split - 1);
    n = a + b;
  }
  else {
    n = lo + hi;
    if (0 < split && 0 != n) {
      n = (LIBXSMM_DELTA(lo, hi) * count + n - 1) / n;
    }
  }
  return (size_t)n;
}


size_t imbalance_uint(const void* input, size_t elemsize, size_t count, size_t split)
{
  const size_t result = imbalance_uint_aux(input, elemsize, count, split);
  return (result + split - 1) / split;
}


size_t bsort_uint_asc(void* inout, size_t elemsize, size_t count) {
  size_t result = 0; /* count number of swaps */
  if (0 != count) {
    unsigned char *const data = (unsigned char*)inout;
    int swap = 1;
    for (; 0 != swap; --count) {
      size_t i = 0, j, k;
      for (swap = 0; i < elemsize * (count - 1); i += elemsize) {
        for (j = i + elemsize, k = elemsize; 0 < k; --k) {
          const unsigned char x = data[i + k - 1], y = data[j + k - 1];
          if (x != y) {
            if (x > y) {
              LIBXSMM_MEMSWP127(data + i, data + j, elemsize);
              swap = 1; ++result;
            }
            break;
          }
        }
      }
    }
  }
  return result;
}
