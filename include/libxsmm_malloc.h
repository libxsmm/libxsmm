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
#ifndef LIBXSMM_MALLOC_H
#define LIBXSMM_MALLOC_H

#include "libxsmm_macros.h"

/** Allocate memory (malloc/free interface). */
LIBXSMM_API LIBXSMM_ATTRIBUTE_MALLOC void* libxsmm_malloc(size_t size);

/** Allocate aligned memory using the default allocator. */
LIBXSMM_API LIBXSMM_ATTRIBUTE_MALLOC void* libxsmm_aligned_malloc(size_t size,
  /**
   * =0: align automatically according to the size
   * 0<: align according to the alignment value
   */
  size_t alignment);

/** Reallocate memory using the default allocator (alignment is preserved). */
LIBXSMM_API void* libxsmm_realloc(size_t size, void* ptr);

/** Deallocate memory (malloc/free interface). */
LIBXSMM_API void libxsmm_free(const void* memory);

/** Information about a buffer (default memory domain). */
LIBXSMM_EXTERN_C typedef struct libxsmm_malloc_info {
  /** Size of the buffer. */
  size_t size;
} libxsmm_malloc_info;

/** Retrieve information about a buffer (default memory domain). */
LIBXSMM_API int libxsmm_get_malloc_info(const void* memory, libxsmm_malloc_info* info);

#endif /*LIBXSMM_MALLOC_H*/
