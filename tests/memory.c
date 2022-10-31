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


int main(int argc, char* argv[])
{
  char item[LIBXSMM_DESCRIPTOR_MAXSIZE];
  const libxsmm_blasint isize = sizeof(item);
  const libxsmm_blasint size = 1000, ntests = 1000;
  char *const data = (char*)malloc((size_t)isize * size);
  char* const shuf = (char*)malloc((size_t)isize * size);
  int result = EXIT_SUCCESS;
  LIBXSMM_UNUSED(argc); LIBXSMM_UNUSED(argv);

  /* check if buffers are allocated (prerequisite) */
  if (EXIT_SUCCESS == result && NULL == data) result = EXIT_FAILURE;
  if (EXIT_SUCCESS == result && NULL == shuf) result = EXIT_FAILURE;

  /* check libxsmm_stristr */
  if (EXIT_SUCCESS == result && NULL != libxsmm_stristr("ends with b", "Begins with b")) result = EXIT_FAILURE;
  if (EXIT_SUCCESS == result && NULL == libxsmm_stristr("in between of", "BeTwEEn")) result = EXIT_FAILURE;
  if (EXIT_SUCCESS == result && NULL == libxsmm_stristr("spr", "SPR")) result = EXIT_FAILURE;
  if (EXIT_SUCCESS == result && NULL != libxsmm_stristr(NULL, "bb")) result = EXIT_FAILURE;
  if (EXIT_SUCCESS == result && NULL != libxsmm_stristr("aa", NULL)) result = EXIT_FAILURE;
  if (EXIT_SUCCESS == result && NULL != libxsmm_stristr(NULL, NULL)) result = EXIT_FAILURE;

  /* check LIBXSMM_MEMCPY127 and libxsmm_diff_n */
  if (EXIT_SUCCESS == result) {
    libxsmm_blasint i = 0;
    libxsmm_rng_seq(data, isize * size);

    for (; i < ntests; ++i) {
      const libxsmm_blasint j = (libxsmm_blasint)libxsmm_rng_u32(size);
      const libxsmm_blasint s = libxsmm_rng_u32(isize) + 1;
      libxsmm_blasint k = s;
      libxsmm_rng_seq(item, s);
      for (; k < isize; ++k) item[k] = 0;
      LIBXSMM_MEMCPY127(data + (j * isize), item, isize);
      k = libxsmm_diff_n(item, data,
        (unsigned char)s, (unsigned char)isize,
        0, size);
      while (k < j) {
        k = libxsmm_diff_n(item, data,
          (unsigned char)s, (unsigned char)isize,
          k + 1, size);
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

  /* check libxsmm_shuffle2 */
  if (EXIT_SUCCESS == result) {
    libxsmm_blasint i = 0;
    const char *const src = (const char*)data;
    libxsmm_shuffle2(shuf, src, isize, size);
    for (; i < size; ++i) {
      const size_t j = libxsmm_diff_n(&src[i*isize], shuf,
        LIBXSMM_CAST_UCHAR(isize), LIBXSMM_CAST_UCHAR(isize),
        (i + size / 2) % size, size);
      if ((size_t)size <= j) {
        result = EXIT_FAILURE;
        break;
      }
    }
  }
#if 0
  /* check libxsmm_shuffle */
  if (EXIT_SUCCESS == result) {
    libxsmm_blasint i = 0;
    const char *const src = (const char*)data, *const dst = (const char*)shuf;
    libxsmm_shuffle(shuf, isize, size);
    for (; i < size; ++i) {
      if (0 != libxsmm_diff(&src[i*isize], &dst[i*isize], LIBXSMM_CAST_UCHAR(isize))) {
        result = EXIT_FAILURE;
        break;
      }
    }
  }
#endif
  free(data);
  free(shuf);

  return result;
}

