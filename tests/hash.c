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

#if defined(_DEBUG)
# define FPRINTF(STREAM, ...) do { fprintf(STREAM, __VA_ARGS__); } while(0)
#else
# define FPRINTF(STREAM, ...) do {} while(0)
#endif

#if !defined(ELEMTYPE)
# define ELEMTYPE int
#endif


/**
 * This test case is NOT an example of how to use LIBXSMM
 * since INTERNAL functions are tested which are not part
 * of the LIBXSMM API.
 */
int main(void)
{
  const unsigned int seed = 1975, size = 2507;
  const unsigned int n512 = 512 / (8 * sizeof(ELEMTYPE));
  unsigned int s = LIBXSMM_UP(size, n512), i, h1, h2;
  unsigned long long a, b;
  int result = EXIT_SUCCESS;
  const ELEMTYPE* value;

  ELEMTYPE *const data = (ELEMTYPE*)libxsmm_malloc(sizeof(ELEMTYPE) * s);
  if (NULL == data) s = 0;
  for (i = 0; i < s; ++i) data[i] = (ELEMTYPE)(rand() - ((RAND_MAX) >> 1));

  h1 = libxsmm_crc32_u64(seed, data);
  h2 = libxsmm_crc32_u32(seed, data);
  h2 = libxsmm_crc32_u32(h2, (unsigned int*)data + 1);
  a = h1;
  b = h2;
  if (a != b) {
    FPRINTF(stderr, "ERROR line #%i: %llu != %llu\n", __LINE__, a, b);
    result = EXIT_FAILURE;
  }

  h1 = libxsmm_crc32(seed, data, sizeof(ELEMTYPE) * s);
  h2 = seed; value = data;
  for (i = 0; i < s; i += n512) {
    h2 = libxsmm_crc32_u512(h2, value + i);
  }
  a = h1;
  b = h2;
  if (a != b) {
    FPRINTF(stderr, "ERROR line #%i: %llu != %llu\n", __LINE__, a, b);
    result = EXIT_FAILURE;
  }

  h2 = h1 >> 16;
  a = libxsmm_crc32_u16(h2, &h1) & 0xFFFF;
  b = libxsmm_crc32_u16(h1 & 0xFFFF, &h2) & 0xFFFF;
  if (a != b) {
    FPRINTF(stderr, "ERROR line #%i: %llu != %llu\n", __LINE__, a, b);
    result = EXIT_FAILURE;
  }

  h2 = libxsmm_crc32_u16(h2, &h1) & 0xFFFF;
  a = h2;
  b = libxsmm_hash16(h1);
  if (a != b) {
    FPRINTF(stderr, "ERROR line #%i: %llu != %llu\n", __LINE__, a, b);
    result = EXIT_FAILURE;
  }

  a = seed;
  b = libxsmm_hash(NULL/*data*/, 0/*size*/, seed);
  if (a != b) {
    FPRINTF(stderr, "ERROR line #%i: %llu != %llu\n", __LINE__, a, b);
    result = EXIT_FAILURE;
  }

  a = 0;
  b = libxsmm_hash_string(NULL/*string*/);
  if (a != b) {
    FPRINTF(stderr, "ERROR line #%i: %llu != %llu\n", __LINE__, a, b);
    result = EXIT_FAILURE;
  }

  a = '1';
  b = libxsmm_hash_string("1");
  if (a != b) {
    FPRINTF(stderr, "ERROR line #%i: %llu != %llu\n", __LINE__, a, b);
    result = EXIT_FAILURE;
  }

  a = 4050765991979987505ULL;
  b = libxsmm_hash_string("12345678");
  if (a != b) {
    FPRINTF(stderr, "ERROR line #%i: %llu != %llu\n", __LINE__, a, b);
    result = EXIT_FAILURE;
  }

  a = 17777927841313886634ULL;
  b = libxsmm_hash_string("01234567890");
  if (a != b) {
    FPRINTF(stderr, "ERROR line #%i: %llu != %llu\n", __LINE__, a, b);
    result = EXIT_FAILURE;
  }

  a = 3199039660;
  b = libxsmm_hash32(b);
  if (a != b) {
    FPRINTF(stderr, "ERROR line #%i: %llu != %llu\n", __LINE__, a, b);
    result = EXIT_FAILURE;
  }

  a = 22875;
  b = libxsmm_hash16((unsigned int)b);
  if (a != b) {
    FPRINTF(stderr, "ERROR line #%i: %llu != %llu\n", __LINE__, a, b);
    result = EXIT_FAILURE;
  }

  a = 237;
  b = libxsmm_hash8((unsigned int)b);
  if (a != b) {
    FPRINTF(stderr, "ERROR line #%i: %llu != %llu\n", __LINE__, a, b);
    result = EXIT_FAILURE;
  }

  libxsmm_free(data);

  return result;
}
