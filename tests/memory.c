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
  const libxsmm_blasint elemsize = sizeof(item);
  const libxsmm_blasint count = 1000, ntests = 1000;
  char *const data = (char*)malloc((size_t)elemsize * count);
  const char init[] = "The quick brown fox jumps over the lazy dog";
  int result = EXIT_SUCCESS;
  LIBXSMM_UNUSED(argc); LIBXSMM_UNUSED(argv);

  /* check if buffers are allocated (prerequisite) */
  if (EXIT_SUCCESS == result && NULL == data) result = EXIT_FAILURE;

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

  /* check LIBXSMM_MEMCPY127 and libxsmm_diff_n */
  if (EXIT_SUCCESS == result) {
    libxsmm_blasint i = 0;
    libxsmm_rng_seq(data, elemsize * count);

    for (; i < ntests; ++i) {
      const libxsmm_blasint j = (libxsmm_blasint)libxsmm_rng_u32(count);
      const libxsmm_blasint s = libxsmm_rng_u32(elemsize) + 1;
      libxsmm_blasint k = s;
      libxsmm_rng_seq(item, s);
      for (; k < elemsize; ++k) item[k] = 0;
      LIBXSMM_MEMCPY127(data + (j * elemsize), item, elemsize);
      k = libxsmm_diff_n(item, data,
        (unsigned char)s, (unsigned char)elemsize,
        0, count);
      while (k < j) {
        k = libxsmm_diff_n(item, data,
          (unsigned char)s, (unsigned char)elemsize,
          k + 1, count);
      }
      if (k == j) {
        continue;
      }
      else {
        result = EXIT_FAILURE;
        break;
      }
    }
  }

  if (EXIT_SUCCESS == result) { /* check libxsmm_shuffle */
    char a[sizeof(init)], b[sizeof(init)];
    const size_t size = sizeof(init);
    size_t s = 0, i;
    for (; s < size; ++s) {
      const size_t shuffle = libxsmm_coprime2(s);
      const size_t gcd = libxsmm_gcd(shuffle, s);
      if (1 == gcd) {
        const size_t r = libxsmm_unshuffle(s, &shuffle);
        int cmp;
        memset(a, 0, size); /* clear */
        libxsmm_shuffle(a, init, 1, s, &shuffle, NULL);
        cmp = memcmp(a, init, s);
        if ((1 >= s || 0 == cmp) && (1 < s || 0 != cmp)) {
          FPRINTF(stderr, "libxsmm_shuffle: data not shuffled or copy failed!\n");
          result = EXIT_FAILURE; break;
        }
        /* shuffle restores initial input */
        for (i = 0; i < r; ++i) {
          memset(b, 0, size); /* clear */
          libxsmm_shuffle(b, a, 1, s, &shuffle, NULL);
          /* every shuffle is different from input */
          if (1 < s && 0 == memcmp(a, b, s)) {
            FPRINTF(stderr, "libxsmm_shuffle: data not shuffled!\n");
            result = EXIT_FAILURE; break;
          }
          if (0 == memcmp(b, init, s)) break; /* restored */
          else if (r == (i + 1)) {
            FPRINTF(stderr, "libxsmm_shuffle: data not restored!\n");
            result = EXIT_FAILURE;
          }
          memcpy(a, b, s);
        }
        if (EXIT_SUCCESS == result) {
          memset(a, 0, size); /* clear */
          libxsmm_shuffle(a, init, 1, s, &shuffle, NULL);
          memset(b, 0, size); /* clear */
          libxsmm_shuffle(b, a, 1, s, &shuffle, &r);
          if (0 != memcmp(b, init, s)) {
            FPRINTF(stderr, "libxsmm_shuffle: data not restored!\n");
            result = EXIT_FAILURE;
            break;
          }
        }
        else break; /* previous error */
      }
      else {
        FPRINTF(stderr, "libxsmm_shuffle: shuffle argument not coprime!\n");
        result = EXIT_FAILURE;
        break;
      }
    }
  }
  free(data);

  return result;
}
