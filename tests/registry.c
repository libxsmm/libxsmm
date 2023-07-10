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
#include <libxsmm.h>


int main(int argc, char* argv[])
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
  const char* value[] = {
    "hello", "world", "libxsmm",
    "hello world", "hello libxsmm",
    "value", "next", "last"
  };
  const size_t key_size = sizeof(*key);
#if (0 != LIBXSMM_JIT) /* unused variable warning */
  const int small_key = 0, n = (int)sizeof(key) / (int)key_size;
  const char string[] = "payload";
  int i;
#endif
  LIBXSMM_UNUSED(argc); LIBXSMM_UNUSED(argv);
  if (EXIT_SUCCESS == result) { /* test for some expected failure */
    result = (NULL == libxsmm_xregister(key, /*too large*/LIBXSMM_DESCRIPTOR_MAXSIZE + 1,
      strlen(value[0]) + 1, value[0]) ? EXIT_SUCCESS : EXIT_FAILURE);
  }
  if (EXIT_SUCCESS == result) { /* test for some expected failure */
    result = (NULL == libxsmm_xregister(NULL, 16, /* invalid combination */
      strlen(value[0]) + 1, value[0]) ? EXIT_SUCCESS : EXIT_FAILURE);
  }
  if (EXIT_SUCCESS == result) { /* test for some expected failure */
    result = (NULL == libxsmm_xregister(NULL, 0, /* invalid combination */
      strlen(value[0]) + 1, value[0]) ? EXIT_SUCCESS : EXIT_FAILURE);
  }
  if (EXIT_SUCCESS == result) { /* test for some expected failure */
    result = (NULL == libxsmm_xregister(key, key_size,
      0, NULL) ? EXIT_SUCCESS : EXIT_FAILURE);
  }
#if (0 != LIBXSMM_JIT) /* registry service is only available if JIT is enabled */
  if (EXIT_SUCCESS == result) { /* register and initialize value later */
    char *const v = (char*)libxsmm_xregister(key, key_size, strlen(value[0]) + 1, NULL);
    strcpy(v, value[0]); /* initialize value after registration */
    result = (NULL != v ? EXIT_SUCCESS : EXIT_FAILURE);
  }
  if (EXIT_SUCCESS == result) { /* retrieve previously registered value */
    const char *const v = (const char*)libxsmm_xdispatch(key, key_size);
    result = ((NULL != v && 0 == strcmp(v, value[0])) ? EXIT_SUCCESS : EXIT_FAILURE);
  }
  if (EXIT_SUCCESS == result) { /* re-register with same size of payload */
    const size_t samesize = strlen(value[0]);
    char *const v = (char*)libxsmm_xregister(key, key_size, samesize + 1, value[5]);
    if (NULL != v) {
      v[samesize] = '\0';
      result = (0 == strncmp(v, value[5], samesize) ? EXIT_SUCCESS : EXIT_FAILURE);
    }
    else result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS == result) { /* re-register with larger payload (failure) */
    result = (NULL == libxsmm_xregister(key, key_size,
      strlen(value[3]) + 1, value[3]) ? EXIT_SUCCESS : EXIT_FAILURE);
  }
  if (EXIT_SUCCESS == result) { /* release entry (enabled for user-data) */
    libxsmm_xrelease(key, key_size);
  }
  if (EXIT_SUCCESS == result) { /* re-register with larger payload */
    result = (NULL != libxsmm_xregister(key, key_size,
      strlen(value[3]) + 1, value[3]) ? EXIT_SUCCESS : EXIT_FAILURE);
  }
  if (EXIT_SUCCESS == result) { /* retrieve previously registered value */
    const char *const v = (const char*)libxsmm_xdispatch(key, key_size);
    result = ((NULL != v && 0 == strcmp(v, value[3])) ? EXIT_SUCCESS : EXIT_FAILURE);
  }
  if (EXIT_SUCCESS == result) { /* release entry (enabled for user-data) */
    libxsmm_xrelease(key, key_size);
  }
  for (i = 0; i < n && EXIT_SUCCESS == result; ++i) { /* register all entries */
    result = (NULL != libxsmm_xregister(key + i, key_size,
      strlen(value[i]) + 1, value[i]) ? EXIT_SUCCESS : EXIT_FAILURE);
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
          result = (0 == strcmp(ivalue, value[i]) ? EXIT_SUCCESS : EXIT_FAILURE);
          break;
        }
      }
      if (EXIT_SUCCESS != result) break;
    }
  }
  if (EXIT_SUCCESS == result) { /* register small key */
    result = (NULL != libxsmm_xregister(&small_key, sizeof(small_key),
      sizeof(string), string) ? EXIT_SUCCESS : EXIT_FAILURE);
  }
  for (i = 0; i < n && EXIT_SUCCESS == result; ++i) {
    const char *const v = (char*)libxsmm_xdispatch(key + i, key_size);
    if (NULL != v) {
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
    else result = EXIT_FAILURE;
  }
  if (EXIT_SUCCESS == result) {
    const void* regentry = libxsmm_get_registry_begin(LIBXSMM_KERNEL_KIND_USER, NULL);
    for (; NULL != regentry; regentry = libxsmm_get_registry_next(regentry, NULL)) {
      libxsmm_release_kernel(regentry);
    }
  }
#endif
  return result;
}
