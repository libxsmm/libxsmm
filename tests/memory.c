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
#include <libxsmm_source.h>

#if !defined(PRINT) && (defined(_DEBUG) || 0)
# define PRINT
#endif
#if defined(PRINT)
# define FPRINTF(STREAM, ...) do { fprintf(STREAM, __VA_ARGS__); } while(0)
#else
# define FPRINTF(STREAM, ...) do {} while(0)
#endif


int main(int argc, char* argv[])
{
  char item[LIBXSMM_DESCRIPTOR_MAXSIZE];
  const size_t elemsize = sizeof(item);
  const libxsmm_blasint count = 1000, ntests = 1000;
  const char init[] = "The quick brown fox jumps over the lazy dog";
  int result = EXIT_SUCCESS;
  LIBXSMM_UNUSED(argc); LIBXSMM_UNUSED(argv);

  /* check libxsmm_stristr */
  if (EXIT_SUCCESS == result && NULL != libxsmm_stristr("ends with b", "Begins with b")) result = EXIT_FAILURE;
  if (EXIT_SUCCESS == result && NULL == libxsmm_stristr("in between of", "BeTwEEn")) result = EXIT_FAILURE;
  if (EXIT_SUCCESS == result && NULL == libxsmm_stristr("spr", "SPR")) result = EXIT_FAILURE;
  if (EXIT_SUCCESS == result && NULL != libxsmm_stristr(NULL, "bb")) result = EXIT_FAILURE;
  if (EXIT_SUCCESS == result && NULL != libxsmm_stristr("aa", NULL)) result = EXIT_FAILURE;
  if (EXIT_SUCCESS == result && NULL != libxsmm_stristr(NULL, NULL)) result = EXIT_FAILURE;

  /* check libxsmm_strimatch */
  if (EXIT_SUCCESS == result && 2 != libxsmm_strimatch("Co Product A", "Corp Prod B", NULL)) result = EXIT_FAILURE;
  if (EXIT_SUCCESS == result && 2 != libxsmm_strimatch("Corp Prod B", "Co Product A", NULL)) result = EXIT_FAILURE;
  if (EXIT_SUCCESS == result && 3 != libxsmm_strimatch("Co Product A", "Corp Prod AA", NULL)) result = EXIT_FAILURE;
  if (EXIT_SUCCESS == result && 3 != libxsmm_strimatch("Corp Prod AA", "Co Product A", NULL)) result = EXIT_FAILURE;
  if (EXIT_SUCCESS == result && 3 != libxsmm_strimatch("Corp Prod AA", "Co Product A", NULL)) result = EXIT_FAILURE;
  if (EXIT_SUCCESS == result && 3 != libxsmm_strimatch("Co Product A", "Corp Prod AA", NULL)) result = EXIT_FAILURE;
  if (EXIT_SUCCESS == result && 3 != libxsmm_strimatch("Corp Prod A", "Co Product A", NULL)) result = EXIT_FAILURE;
  if (EXIT_SUCCESS == result && 3 != libxsmm_strimatch("Co Product A", "Corp Prod A", NULL)) result = EXIT_FAILURE;
  if (EXIT_SUCCESS == result && 3 != libxsmm_strimatch("C Product A", "Cor Prod AA", NULL)) result = EXIT_FAILURE;
  if (EXIT_SUCCESS == result && 3 != libxsmm_strimatch("Cor Prod AA", "C Product A", NULL)) result = EXIT_FAILURE;
  if (EXIT_SUCCESS == result && 1 != libxsmm_strimatch("aaaa", "A A A A", NULL)) result = EXIT_FAILURE;
  if (EXIT_SUCCESS == result && 1 != libxsmm_strimatch("A A A A", "aaaa", NULL)) result = EXIT_FAILURE;
  if (EXIT_SUCCESS == result) {
    const char *const sample[] = {
      "The quick red squirrel jumps over the low fence",
      "The slow green frog jumps over the lazy dog",
      "The lazy brown dog jumps over the quick fox", /* match */
      "The hazy fog crawls over the lazy crocodile"
    };
    int match = 0, i = 0;
#if defined(PRINT)
    int j = 0;
#endif
    for (; i < ((int)sizeof(sample) / (int)sizeof(*sample)); ++i) {
      const int score = libxsmm_strimatch(init, sample[i], NULL);
      if (match < score) {
        match = score;
#if defined(PRINT)
        j = i;
#endif
      }
      else if (0 > score) result = EXIT_FAILURE;
    }
    if (EXIT_SUCCESS == result) {
      int self = 0;
      if (0 < match) {
        self = libxsmm_strimatch(init, init, NULL);
        FPRINTF(stdout, "orig (%i): %s\n", self, init);
        FPRINTF(stdout, "best (%i): %s\n", match, sample[j]);
      }
      if (9 != self || 8 != match) result = EXIT_FAILURE; /* test */
    }
  }

  /* check libxsmm_offset */
  if (EXIT_SUCCESS == result) {
    const size_t shape[] = { 17, 13, 64, 4 }, ndims = sizeof(shape) / sizeof(*shape);
    size_t size1 = 0, n;
    for (n = 0; n < ndims && EXIT_SUCCESS == result; ++n) {
      if (0 != libxsmm_offset(NULL, shape, n, NULL)) result = EXIT_FAILURE;
    }
    for (n = 0; n < ndims && EXIT_SUCCESS == result; ++n) {
      const size_t offset1 = libxsmm_offset(shape, shape, n, &size1);
      if (offset1 != size1) result = EXIT_FAILURE;
    }
  }

  /* check LIBXSMM_MEMCPY127 and libxsmm_diff_n */
  if (EXIT_SUCCESS == result) {
    char *const data = (char*)malloc(elemsize * count);
    if (NULL != data) { /* check if buffer was allocated */
      libxsmm_blasint i = 0;
      libxsmm_rng_seq(data, elemsize * count);

      for (; i < ntests; ++i) {
        const size_t j = libxsmm_rng_u32((unsigned int)count);
        const size_t s = libxsmm_rng_u32((unsigned int)elemsize) + 1;
        size_t k = s;
        libxsmm_rng_seq(item, s);
        for (; k < elemsize; ++k) item[k] = 0;
        LIBXSMM_MEMCPY127(data + elemsize * j, item, elemsize);
        k = libxsmm_diff_n(item, data,
          (unsigned char)s, (unsigned char)elemsize,
          0, count);
        while (k < j) {
          k = libxsmm_diff_n(item, data,
            (unsigned char)s, (unsigned char)elemsize,
            LIBXSMM_CAST_UINT(k + 1), count);
        }
        if (k == j) continue;
        else {
          result = EXIT_FAILURE;
          break;
        }
      }
      free(data);
    }
    else result = EXIT_FAILURE;
  }

  /* check LIBXSMM_MEMSWP127 */
  if (EXIT_SUCCESS == result) {
    char a[sizeof(init)] = { 0 };
    const size_t size = sizeof(init);
    size_t i, j, k;
    memcpy(a, init, size);

    for (k = 1; k <= 8; ++k) {
      const size_t s = (size - 1) / k;
      for (j = 0; j < s; ++j) {
        for (i = 0; i < (s - 1); ++i) {
          LIBXSMM_MEMSWP127(a + k * i, a + k * i + k, k);
        }
      }
      if (0 != strcmp(a, init)) {
        FPRINTF(stderr, "LIBXSMM_MEMSWP127: incorrect result!\n");
        result = EXIT_FAILURE;
        break;
      }
    }
  }

  return result;
}
