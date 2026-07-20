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
#include <libxsmm_intrinsics_x86.h>
#include <libxsmm.h>

#if !defined(CHECK_REALLOC) && 1
# if !defined(_WIN32)
#   define CHECK_REALLOC
# endif
#endif

#if defined(_DEBUG)
# define FPRINTF(STREAM, ...) do { fprintf(STREAM, __VA_ARGS__); } while(0)
#else
# define FPRINTF(STREAM, ...) do {} while(0)
#endif


int main(void)
{
  const size_t size_malloc = 2507, alignment = (2U << 20);
  libxsmm_malloc_info malloc_info;
  int avalue, nerrors = 0;
  void* s;

  /* allocate some amount of memory */
  s = libxsmm_malloc(size_malloc);

  /* query and check the size of the buffer */
  if (NULL != s && (EXIT_SUCCESS != libxsmm_get_malloc_info(s, &malloc_info) || malloc_info.size < size_malloc)) {
    FPRINTF(stderr, "Error: buffer info (1/4) failed!\n");
    ++nerrors;
  }

#if defined(CHECK_REALLOC)
  if (NULL != s) { /* reallocate larger amount of memory */
    unsigned char* c = (unsigned char*)s;
    size_t i;
    int n;
    avalue = 1 << LIBXSMM_INTRINSICS_BITSCANFWD64((uintptr_t)s);
    for (i = 0; i < size_malloc; ++i) c[i] = (unsigned char)LIBXSMM_MOD2(i, 256);
    s = libxsmm_realloc(size_malloc * 2, s);
    /* check that alignment is preserved */
    if (0 != (((uintptr_t)s) % avalue)) {
      FPRINTF(stderr, "Error: buffer alignment (1/3) not preserved!\n");
      ++nerrors;
    }
    c = (unsigned char*)s;
    for (i = size_malloc; i < (size_malloc * 2); ++i) c[i] = (unsigned char)LIBXSMM_MOD2(i, 256);
    /* reallocate again with same size */
    s = libxsmm_realloc(size_malloc * 2, s);
    /* check that alignment is preserved */
    if (0 != (((uintptr_t)s) % avalue)) {
      FPRINTF(stderr, "Error: buffer alignment (2/3) not preserved!\n");
      ++nerrors;
    }
    c = (unsigned char*)s;
    for (i = n = 0; i < (size_malloc * 2); ++i) { /* check that content is preserved */
      n += (c[i] == (unsigned char)LIBXSMM_MOD2(i, 256) ? 0 : 1);
    }
    if (0 < n) {
      FPRINTF(stderr, "Error: buffer content (1/2) not preserved!\n");
      nerrors += n;
    }
    /* reallocate with smaller size */
    s = libxsmm_realloc(size_malloc / 2, s);
    /* check that alignment is preserved */
    if (0 != (((uintptr_t)s) % avalue)) {
      FPRINTF(stderr, "Error: buffer alignment (3/3) not preserved!\n");
      ++nerrors;
    }
    c = (unsigned char*)s;
    for (i = n = 0; i < size_malloc / 2; ++i) { /* check that content is preserved */
      n += (c[i] == (unsigned char)LIBXSMM_MOD2(i, 256) ? 0 : 1);
    }
    if (0 < n) {
      FPRINTF(stderr, "Error: buffer content (2/2) not preserved!\n");
      nerrors += n;
    }
  }
  /* query and check the size of the buffer */
  if (NULL != s && (EXIT_SUCCESS != libxsmm_get_malloc_info(s, &malloc_info) || malloc_info.size < (size_malloc / 2))) {
    FPRINTF(stderr, "Error: buffer info (2/4) failed!\n");
    ++nerrors;
  }
  libxsmm_free(s); /* release buffer */

  /* check degenerated reallocation */
  s = libxsmm_realloc(size_malloc, NULL/*allocation*/);
  /* query and check the size of the buffer */
  if (NULL != s && (EXIT_SUCCESS != libxsmm_get_malloc_info(s, &malloc_info) || malloc_info.size < size_malloc)) {
    FPRINTF(stderr, "Error: buffer info (3/4) failed!\n");
    ++nerrors;
  }
#endif

  /* check that a NULL-pointer yields no size */
  if (EXIT_SUCCESS != libxsmm_get_malloc_info(NULL, &malloc_info) || 0 != malloc_info.size) {
    FPRINTF(stderr, "Error: buffer info (4/4) failed!\n");
    ++nerrors;
  }

  /* release NULL pointer */
  libxsmm_free(NULL);

  /* release buffer */
  libxsmm_free(s);

  /* allocate memory with specific alignment */
  s = libxsmm_aligned_malloc(size_malloc, alignment);
  /* check function that determines alignment */
  libxsmm_aligned(s, NULL/*inc*/, &avalue);

  /* check the alignment of the allocation */
  if (0 != (((uintptr_t)s) % alignment) || ((size_t)avalue) < alignment) {
    FPRINTF(stderr, "Error: buffer alignment (1/3) incorrect!\n");
    ++nerrors;
  }

  if (libxsmm_aligned(s, NULL/*inc*/, NULL/*alignment*/)) { /* pointer is SIMD-aligned */
    if (alignment < ((size_t)libxsmm_cpuid_vlen(libxsmm_get_target_archid()))) {
      FPRINTF(stderr, "Error: buffer alignment (2/3) incorrect!\n");
      ++nerrors;
    }
  }
  else { /* pointer is not SIMD-aligned */
    if (((size_t)libxsmm_cpuid_vlen(libxsmm_get_target_archid())) <= alignment) {
      FPRINTF(stderr, "Error: buffer alignment (3/3) incorrect!\n");
      ++nerrors;
    }
  }

  /* release memory */
  libxsmm_free(s);

  /* check foreign memory */
  if (EXIT_SUCCESS == libxsmm_get_malloc_info(&size_malloc/*faulty pointer*/, &malloc_info)) {
    FPRINTF(stderr, "Error: uncaught faulty pointer!\n");
    ++nerrors;
  }

  if (0 == nerrors) {
    return EXIT_SUCCESS;
  }
  else {
    FPRINTF(stderr, "Errors: %i\n", nerrors);
    return EXIT_FAILURE;
  }
}
