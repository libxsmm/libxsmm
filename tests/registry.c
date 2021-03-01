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
  int result = EXIT_SUCCESS;
  typedef struct key_type { int x, y, z; } key_type;
  key_type key[] = {
    { 0, 0, 0 },
    { 0, 0, 1 },
    { 0, 1, 0 },
    { 0, 1, 1 },
    { 1, 0, 0 },
    { 1, 0, 1 },
    { 1, 1, 0 },
    { 1, 1, 1 }
  };
  /*const*/ char* value[] = {
    "hello", "world", "libxsmm",
    "hello world", "hello libxsmm",
    "value", "next", "last"
  };
  const size_t key_size = sizeof(*key);
#if (0 != LIBXSMM_JIT) /* unused variable warning */
  const int n = (int)sizeof(key) / (int)key_size;
  int i;
#endif
  if (EXIT_SUCCESS == result) { /* test for some expected failure */
    result = (NULL == libxsmm_xregister(key, /*too large*/LIBXSMM_DESCRIPTOR_MAXSIZE + 1,
      strlen(value[0]) + 1, value[0], NULL) ? EXIT_SUCCESS : EXIT_FAILURE);
  }
  if (EXIT_SUCCESS == result) { /* test for some expected failure */
    result = (NULL == libxsmm_xregister(NULL, 16, /* invalid combination */
      strlen(value[0]) + 1, value[0], NULL) ? EXIT_SUCCESS : EXIT_FAILURE);
  }
  if (EXIT_SUCCESS == result) { /* test for some expected failure */
    result = (NULL == libxsmm_xregister(NULL, 0, /* invalid combination */
      strlen(value[0]) + 1, value[0], NULL) ? EXIT_SUCCESS : EXIT_FAILURE);
  }
  if (EXIT_SUCCESS == result) { /* test for some expected failure */
    result = (NULL == libxsmm_xregister(key, key_size,
      0, NULL, NULL) ? EXIT_SUCCESS : EXIT_FAILURE);
  }
#if (0 != LIBXSMM_JIT) /* registry service is only available if JIT is enabled */
  if (EXIT_SUCCESS == result) { /* same key but (larger) payload; initialized later */
    result = (NULL != libxsmm_xregister(key, key_size,
      strlen(value[0]) + 1, NULL, NULL) ? EXIT_SUCCESS : EXIT_FAILURE);
  }
  if (EXIT_SUCCESS == result) { /* re-register same key with larger payload */
    result = (NULL == libxsmm_xregister(key, key_size,
      strlen(value[3]) + 1, value[0], NULL) ? EXIT_SUCCESS : EXIT_FAILURE);
  }
  if (EXIT_SUCCESS == result) { /* release registered value */
    libxsmm_xrelease(key, key_size);
  }
  for (i = 0; i < n && EXIT_SUCCESS == result; ++i) {
    result = (NULL != libxsmm_xregister(key + i, key_size,
      strlen(value[i]) + 1, value[i], NULL) ? EXIT_SUCCESS : EXIT_FAILURE);
  }
  if (EXIT_SUCCESS == result) {
    const void* regkey = NULL;
    const void* regentry = libxsmm_get_registry_begin(LIBXSMM_KERNEL_KIND_USER, &regkey);
    for (; NULL != regentry; regentry = libxsmm_get_registry_next(regentry, &regkey)) {
      const key_type *const ikey = (const key_type*)regkey;
      const char *const ivalue = (const char*)regentry;
      result = EXIT_FAILURE;
      for (i = 0; i < n; ++i) {
        if (ikey->x == key[i].x && ikey->y == key[i].y && ikey->z == key[i].z) {
          result = (0 == strcmp(ivalue, value[i]) ? EXIT_SUCCESS : EXIT_SUCCESS);
          break;
        }
      }
      if (EXIT_SUCCESS != result) break;
    }
  }
  for (i = 0; i < n && EXIT_SUCCESS == result; ++i) {
    const char *const v = (char*)libxsmm_xdispatch(key + i, key_size, NULL);
    libxsmm_kernel_info info;
    result = libxsmm_get_kernel_info(v, &info);
    if (EXIT_SUCCESS == result) {
      result = (LIBXSMM_KERNEL_KIND_USER == info.kind ? EXIT_SUCCESS : EXIT_FAILURE);
    }
    if (EXIT_SUCCESS == result) {
      result = strcmp(v, value[i]);
    }
    libxsmm_release_kernel(v);
  }
#endif
  return result;
}

