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
  const libxsmm_blasint elemsize = sizeof(item);
  const libxsmm_blasint count = 1000, ntests = 1000;
  char *const data = (char*)malloc((size_t)elemsize * count);
  char *const shuf = (char*)malloc((size_t)elemsize * count);
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

  /* check libxsmm_shuffle2 */
  if (EXIT_SUCCESS == result) {
    libxsmm_blasint i = 0;
    libxsmm_shuffle2(shuf, data, elemsize, count);
    for (; i < count; ++i) {
      const unsigned int j = libxsmm_diff_n(&data[i*elemsize], shuf,
        LIBXSMM_CAST_UCHAR(elemsize), LIBXSMM_CAST_UCHAR(elemsize),
        (i + count / 2) % count, count);
      if ((size_t)count <= j) {
        result = EXIT_FAILURE;
        break;
      }
    }
  }

  /* check libxsmm_shuffle */
  if (EXIT_SUCCESS == result) {
    libxsmm_blasint i = 0;
    libxsmm_shuffle(shuf, elemsize, count);
    for (; i < count; ++i) {
      if (0 != libxsmm_diff(&data[i*elemsize], &shuf[i*elemsize], LIBXSMM_CAST_UCHAR(elemsize))) {
        result = EXIT_FAILURE;
        break;
      }
    }
  }

  free(data);
  free(shuf);

  return result;
}
