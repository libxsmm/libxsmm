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

#if defined(_DEBUG)
# define FPRINTF(STREAM, ...) fprintf(STREAM, __VA_ARGS__)
#else
# define FPRINTF(STREAM, ...)
#endif

#if !defined(ELEM_TYPE)
# define ELEM_TYPE int
#endif


/**
 * This test case is NOT an example of how to use LIBXSMM
 * since INTERNAL functions are tested which are not part
 * of the LIBXSMM API.
 */
int main(void)
{
  const unsigned int seed = 1975, size = 2507;
  const unsigned int n512 = 512 / (8 * sizeof(ELEM_TYPE));
  unsigned int s = LIBXSMM_UP(size, n512), i, h1, h2;
  int result = EXIT_SUCCESS;
  const ELEM_TYPE* value;

  ELEM_TYPE *const data = (ELEM_TYPE*)libxsmm_malloc(sizeof(ELEM_TYPE) * s);
  if (NULL == data) s = 0;
  for (i = 0; i < s; ++i) data[i] = (ELEM_TYPE)(rand() - ((RAND_MAX) >> 1));

  h1 = libxsmm_crc32_u64(seed, data);
  h2 = libxsmm_crc32_u32(seed, data);
  h2 = libxsmm_crc32_u32(h2, (unsigned int*)data + 1);
  if (h1 != h2) {
    FPRINTF(stderr, "crc32_u32 or crc32_u64 is wrong\n");
    result = EXIT_FAILURE;
  }

  h1 = libxsmm_crc32(seed, data, sizeof(ELEM_TYPE) * s);
  h2 = seed; value = data;
  for (i = 0; i < s; i += n512) {
    h2 = libxsmm_crc32_u512(h2, value + i);
  }
  if (h1 != h2) {
    FPRINTF(stderr, "(crc32=%u) != (crc32_sw=%u)\n", h1, h2);
    result = EXIT_FAILURE;
  }

  if (seed != libxsmm_hash(NULL/*data*/, 0/*size*/, seed)) {
    result = EXIT_FAILURE;
  }

  if (0 != libxsmm_hash_string(NULL/*string*/)) {
    result = EXIT_FAILURE;
  }

  libxsmm_free(data);

  return result;
}

