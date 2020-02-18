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
#include <libxsmm_intrinsics_x86.h>

#if !defined(CHECK_SETUP) && 1
# define CHECK_SETUP
#endif
#if !defined(CHECK_REALLOC) && 1
# define CHECK_REALLOC
#endif


int main(void)
{
  const size_t size = 2507, alignment = (2U << 20);
  libxsmm_malloc_info malloc_info;
  int nerrors = 0;
  void* p;

#if defined(CHECK_SETUP)
  { /* check allocator setup */
    libxsmm_malloc_function malloc_fn;
    libxsmm_free_function free_fn;
    const void* context;
    malloc_fn.function = malloc; free_fn.function = free;
    libxsmm_set_default_allocator(NULL/*context*/, malloc_fn/*malloc*/, free_fn/*free*/);
    malloc_fn.function = NULL; free_fn.function = NULL;
    libxsmm_set_scratch_allocator(NULL/*context*/, malloc_fn/*NULL*/, free_fn/*NULL*/);

    /* check adoption of the default allocator */
    libxsmm_get_scratch_allocator(&context, &malloc_fn, &free_fn);
    if (NULL != context || malloc != malloc_fn.function || free != free_fn.function) {
      ++nerrors;
    }
  }
#endif

  /* allocate some amount of memory */
  p = libxsmm_malloc(size);

  /* query and check the size of the buffer */
  if (NULL != p && (EXIT_SUCCESS != libxsmm_get_malloc_info(p, &malloc_info) || malloc_info.size < size)) {
    ++nerrors;
  }

#if defined(CHECK_REALLOC)
  if (NULL != p) { /* reallocate larger amount of memory */
    const int palign = 1 << LIBXSMM_INTRINSICS_BITSCANFWD64((uintptr_t)p);
    unsigned char* c = (unsigned char*)p;
    size_t i;
    for (i = 0; i < size; ++i) c[i] = (unsigned char)LIBXSMM_MOD2(i, 256);
    p = libxsmm_realloc(size * 2, p);
    /* check that alignment is preserved */
    if (0 != (((uintptr_t)p) % palign)) {
      ++nerrors;
    }
    c = (unsigned char*)p;
    for (i = size; i < (size * 2); ++i) c[i] = (unsigned char)LIBXSMM_MOD2(i, 256);
    /* reallocate again with same size */
    p = libxsmm_realloc(size * 2, p);
    /* check that alignment is preserved */
    if (0 != (((uintptr_t)p) % palign)) {
      ++nerrors;
    }
    c = (unsigned char*)p;
    for (i = 0; i < (size * 2); ++i) { /* check that content is preserved */
      nerrors += (c[i] == (unsigned char)LIBXSMM_MOD2(i, 256) ? 0 : 1);
    }
    /* reallocate with smaller size */
    p = libxsmm_realloc(size / 2, p);
    /* check that alignment is preserved */
    if (0 != (((uintptr_t)p) % palign)) {
      ++nerrors;
    }
    c = (unsigned char*)p;
    for (i = 0; i < size / 2; ++i) { /* check that content is preserved */
      nerrors += (c[i] == (unsigned char)LIBXSMM_MOD2(i, 256) ? 0 : 1);
    }
  }
  /* query and check the size of the buffer */
  if (NULL != p && (EXIT_SUCCESS != libxsmm_get_malloc_info(p, &malloc_info) || malloc_info.size < (size / 2))) {
    ++nerrors;
  }
  libxsmm_free(p); /* release buffer */

  /* check degenerated reallocation */
  p = libxsmm_realloc(size, NULL/*allocation*/);
  /* query and check the size of the buffer */
  if (NULL != p && (EXIT_SUCCESS != libxsmm_get_malloc_info(p, &malloc_info) || malloc_info.size < size)) {
    ++nerrors;
  }
#endif

  /* check that a NULL-pointer yields no size */
  if (EXIT_SUCCESS != libxsmm_get_malloc_info(NULL, &malloc_info) || 0 != malloc_info.size) {
    ++nerrors;
  }

  /* release NULL pointer */
  libxsmm_free(NULL);

  /* release buffer */
  libxsmm_free(p);

  /* allocate memory with specific alignment */
  p = libxsmm_aligned_malloc(size, alignment);

  /* check the alignment of the allocation */
  if (0 != (((uintptr_t)p) % alignment)) {
    ++nerrors;
  }

  /* release aligned memory */
  libxsmm_free(p);

  /* check foreign memory */
  if (EXIT_SUCCESS == libxsmm_get_malloc_info(&size/*faulty pointer*/, &malloc_info)) {
    ++nerrors;
  }

  return 0 == nerrors ? EXIT_SUCCESS : EXIT_FAILURE;
}

