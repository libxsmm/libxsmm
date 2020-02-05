/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Hans Pabst (Intel Corp.)
******************************************************************************/
#include <libxsmm.h>
#include <string.h>
#include <stdio.h>
#include <math.h>


int main(int argc, char* argv[])
{
  const int insize = (1 < argc ? atoi(argv[1]) : 0);
  const int incrmt = (2 < argc ? atoi(argv[2]) : 0);
  const int nelems = (3 < argc ? atoi(argv[3]) : 0);
  const int niters = (4 < argc ? atoi(argv[4]) : 1);
  const int elsize = (0 >= insize ? LIBXSMM_DESCRIPTOR_SIGSIZE : insize);
  const int stride = (0 >= incrmt ? LIBXSMM_MAX(LIBXSMM_DESCRIPTOR_MAXSIZE, elsize) : LIBXSMM_MAX(incrmt, elsize));
  const size_t n = (0 >= nelems ? (((size_t)2 << 30/*2 GB*/) / stride) : ((size_t)nelems));
  const char *const env_strided = getenv("STRIDED"), *const env_check = getenv("CHECK");
  const int strided = (NULL == env_strided || 0 == *env_strided) ? 0/*default*/ : atoi(env_strided);
  const int check = (NULL == env_check || 0 == *env_check) ? 0/*default*/ : atoi(env_check);
  int result = EXIT_SUCCESS;
  size_t nbytes, size, nrpt;
  unsigned char *a, *b;
  double duration;

  LIBXSMM_ASSERT(elsize <= stride);
  if (0 < niters) {
    size = n;
    nrpt = niters;
  }
  else {
    size = LIBXSMM_MAX(LIBXSMM_ABS(niters), 1);
    nrpt = n;
  }
  nbytes = size * stride;

  libxsmm_init();
  a = (unsigned char*)(0 != nbytes ? malloc(nbytes) : NULL);
  b = (unsigned char*)(0 != nbytes ? malloc(nbytes) : NULL);

  if (NULL != a && NULL != b) {
    /* initialize the data */
    libxsmm_rng_seq(a, (libxsmm_blasint)nbytes);
    memcpy(b, a, nbytes); /* same content */
    /* benchmark libxsmm_diff (always strided) */
    if (elsize < 256) {
      size_t diff = 0, i, j;
      const libxsmm_timer_tickint start = libxsmm_timer_tick();
      for (i = 0; i < nrpt; ++i) {
        for (j = 0; j < nbytes; j += stride) {
          const void *const aj = a + j, *const bj = b + j;
          diff += libxsmm_diff(aj, bj, (unsigned char)elsize);
        }
      }
      duration = libxsmm_timer_duration(start, libxsmm_timer_tick());
      if (0 < duration) printf("libxsmm_diff:\t\t%.8f s (%i MB/s)\n", duration,
        (int)LIBXSMM_ROUND((2.0 * nrpt * nbytes) / ((1024.0 * 1024.0) * duration)));
      result += (int)diff * ((int)stride / ((int)stride + 1)); /* ignore result */
    }

    { /* benchmark libxsmm_memcmp */
      size_t diff = 0, i, j;
      libxsmm_timer_tickint start;
      /* reinitialize the data (flush caches) */
      libxsmm_rng_seq(a, (libxsmm_blasint)nbytes);
      memcpy(b, a, nbytes); /* same content */
      start = libxsmm_timer_tick();
      for (i = 0; i < nrpt; ++i) {
        if (stride == elsize && 0 == strided) {
          diff += libxsmm_memcmp(a, b, nbytes);
        }
        else {
          for (j = 0; j < nbytes; j += stride) {
            const void *const aj = a + j, *const bj = b + j;
            diff += libxsmm_memcmp(aj, bj, elsize);
          }
        }
      }
      duration = libxsmm_timer_duration(start, libxsmm_timer_tick());
      if (0 < duration) printf("libxsmm_memcmp:\t\t%.8f s (%i MB/s)\n", duration,
        (int)LIBXSMM_ROUND((2.0 * nrpt * nbytes) / ((1024.0 * 1024.0) * duration)));
      result += (int)diff * ((int)stride / ((int)stride + 1)); /* ignore result */
    }

    { /* benchmark stdlib's memcmp */
      size_t diff = 0, i, j;
      libxsmm_timer_tickint start;
      /* reinitialize the data (flush caches) */
      libxsmm_rng_seq(a, (libxsmm_blasint)nbytes);
      memcpy(b, a, nbytes); /* same content */
      start = libxsmm_timer_tick();
      for (i = 0; i < nrpt; ++i) {
        if (stride == elsize && 0 == strided) {
          diff += (0 != memcmp(a, b, nbytes));
        }
        else {
          for (j = 0; j < nbytes; j += stride) {
            const void *const aj = a + j, *const bj = b + j;
#if defined(_MSC_VER)
#           pragma warning(push)
#           pragma warning(disable: 6385)
#endif
            diff += (0 != memcmp(aj, bj, elsize));
#if defined(_MSC_VER)
#           pragma warning(pop)
#endif
          }
        }
      }
      duration = libxsmm_timer_duration(start, libxsmm_timer_tick());
      if (0 < duration) printf("stdlib memcmp:\t\t%.8f s (%i MB/s)\n", duration,
        (int)LIBXSMM_ROUND((2.0 * nrpt * nbytes) / ((1024.0 * 1024.0) * duration)));
      result += (int)diff * ((int)stride / ((int)stride + 1)); /* ignore result */
    }

    if (0 != check) { /* validation */
      size_t diff = 0, i, j, k;
      for (i = 0; i < nrpt; ++i) {
        for (j = 0; j < nbytes; j += stride) {
          unsigned char *const aj = a + j, *const bj = b + j;
          for (k = 0; k < 2; ++k) {
            const int r = rand() % elsize;
#if defined(_MSC_VER)
#           pragma warning(push)
#           pragma warning(disable: 6385)
#endif
            if (0 != memcmp(aj, bj, elsize)) {
              if (elsize < 256 && 0 == libxsmm_diff(aj, bj, (unsigned char)elsize)) ++diff;
              if (0 == libxsmm_memcmp(aj, bj, elsize)) ++diff;
            }
            else {
              if (elsize < 256 && 0 != libxsmm_diff(aj, bj, (unsigned char)elsize)) ++diff;
              if (0 != libxsmm_memcmp(aj, bj, elsize)) ++diff;
            }
#if defined(_MSC_VER)
#           pragma warning(pop)
#endif
            /* inject difference into a or b */
            if (0 != (rand() & 1)) {
              aj[r] = (unsigned char)(rand() % 256);
            }
            else {
              bj[r] = (unsigned char)(rand() % 256);
            }
          }
        }
      }
      if (0 != diff) {
        fprintf(stderr, "ERROR: errors=%i - validation failed!", (int)diff);
        result = EXIT_FAILURE;
      }
    }
  }
  else {
    result = EXIT_FAILURE;
  }

  free(a);
  free(b);

  return result;
}

