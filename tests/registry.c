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


int main(/*int argc, char* argv[]*/)
{
  int result = EXIT_SUCCESS, i;
  struct { int x, y, z; } key[] = {
    { 0, 0, 0 },
    { 0, 0, 1 },
    { 0, 1, 0 },
    { 0, 1, 1 },
    { 1, 0, 0 },
    { 1, 0, 1 },
    { 1, 1, 0 },
    { 1, 1, 1 }
  };
  const size_t key_size = sizeof(*key);
  const int n = (int)sizeof(key) / (int)key_size;
  /*const*/ char* value[] = {
    "hello", "world", "libxsmm",
    "hello world", "hello libxsmm",
    "value", "next", "last"
  };

  if (EXIT_SUCCESS == result) { /* test for some expected failure */
    result = (EXIT_SUCCESS != libxsmm_xregister(key, /*too large*/LIBXSMM_DESCRIPTOR_MAXSIZE + 1,
      value[0], strlen(value[0]) + 1) ? EXIT_SUCCESS : EXIT_FAILURE);
  }
  if (EXIT_SUCCESS == result) { /* test for some expected failure */
    result = (EXIT_SUCCESS != libxsmm_xregister(NULL, 16, /* invalid combination */
      value[0], strlen(value[0]) + 1) ? EXIT_SUCCESS : EXIT_FAILURE);
  }
  if (EXIT_SUCCESS == result) { /* test for some expected failure */
    result = (EXIT_SUCCESS != libxsmm_xregister(NULL, 0, /* invalid combination */
      value[0], strlen(value[0]) + 1) ? EXIT_SUCCESS : EXIT_FAILURE);
  }
  if (EXIT_SUCCESS == result) { /* test for some expected failure */
    result = (EXIT_SUCCESS != libxsmm_xregister(key, key_size,
      NULL, 1) /* invalid combination */
      ? EXIT_SUCCESS : EXIT_FAILURE);
  }
#if (0 != LIBXSMM_JIT) /* registry service only with JIT */
# if 0 /* TODO: register with duplicate key, etc. */
  if (EXIT_SUCCESS == result) { /* test for some expected failure */
    result = libxsmm_xregister(key, key_size, NULL, 0);
  }
# endif
  for (i = 0; i < n && EXIT_SUCCESS == result; ++i) {
    result = libxsmm_xregister(key + i, key_size, value[i], strlen(value[i]) + 1);
  }
  for (i = 0; i < n && EXIT_SUCCESS == result; ++i) {
    const char *const v = (char*)libxsmm_xdispatch(key + i, key_size);
    libxsmm_kernel_info info;
    result = libxsmm_get_kernel_info(v, &info);
    if (EXIT_SUCCESS == result) {
      result = (LIBXSMM_KERNEL_KIND_USER == info.kind ? EXIT_SUCCESS : EXIT_FAILURE);
    }
    if (EXIT_SUCCESS == result) {
      result = strcmp(v, value[i]);
    }
  }
#endif
  return result;
}

