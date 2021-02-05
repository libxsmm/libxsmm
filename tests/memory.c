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
#include <libxsmm_source.h>


int main(/*int argc, char* argv[]*/)
{
  char item[LIBXSMM_DESCRIPTOR_MAXSIZE];
  const libxsmm_blasint isize = sizeof(item);
  const libxsmm_blasint size = 1000, ntests = 1000;
  char *const data = (char*)malloc((size_t)isize * size);
  libxsmm_blasint i, j, k, s;

  if (NULL != libxsmm_stristr("ends with b", "Begins with b")) return EXIT_FAILURE;
  if (NULL == libxsmm_stristr("in between of", "BeTwEEn")) return EXIT_FAILURE;
  if (NULL == libxsmm_stristr("spr", "SPR")) return EXIT_FAILURE;
  if (NULL != libxsmm_stristr(NULL, "bb")) return EXIT_FAILURE;
  if (NULL != libxsmm_stristr("aa", NULL)) return EXIT_FAILURE;
  if (NULL != libxsmm_stristr(NULL, NULL)) return EXIT_FAILURE;

  if (NULL == data) return EXIT_FAILURE;
  libxsmm_rng_seq(data, isize * size);

  for (i = 0; i < ntests; ++i) {
    j = (libxsmm_blasint)libxsmm_rng_u32(size);
    s = libxsmm_rng_u32(isize) + 1;
    libxsmm_rng_seq(item, s);
    for (k = s; k < isize; ++k) item[k] = 0;
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
      free(data); return EXIT_FAILURE;
    }
  }
  free(data);

  return EXIT_SUCCESS;
}

