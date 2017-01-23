/******************************************************************************
** Copyright (c) 2014-2017, Intel Corporation                                **
** All rights reserved.                                                      **
**                                                                           **
** Redistribution and use in source and binary forms, with or without        **
** modification, are permitted provided that the following conditions        **
** are met:                                                                  **
** 1. Redistributions of source code must retain the above copyright         **
**    notice, this list of conditions and the following disclaimer.          **
** 2. Redistributions in binary form must reproduce the above copyright      **
**    notice, this list of conditions and the following disclaimer in the    **
**    documentation and/or other materials provided with the distribution.   **
** 3. Neither the name of the copyright holder nor the names of its          **
**    contributors may be used to endorse or promote products derived        **
**    from this software without specific prior written permission.          **
**                                                                           **
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS       **
** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT         **
** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR     **
** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT      **
** HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,    **
** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED  **
** TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR    **
** PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF    **
** LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING      **
** NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS        **
** SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.              **
******************************************************************************/
/* Hans Pabst (Intel Corp.)
******************************************************************************/
#ifndef LIBXSMM_MALLOC_H
#define LIBXSMM_MALLOC_H

#include "libxsmm_macros.h"

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#include <stddef.h>
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif


/** Function type accepted for memory allocation (libxsmm_set_allocator). */
typedef LIBXSMM_RETARGETABLE void* (*libxsmm_malloc_function)(size_t size);
/** Function type accepted for memory release (libxsmm_set_allocator). */
typedef LIBXSMM_RETARGETABLE void (*libxsmm_free_function)(void* buffer);

/**
 * To setup the memory allocator with the first two arguments, either a default_malloc_fn
 * and corresponding default_free_fn function are given (custom default allocator), or two
 * NULL-pointers are given (reset default allocator to library's solution).
 * For the scratch allocator, a scratch_malloc_fn function different from default_malloc_fn
 * can be supplied; a scratch_free_fn is optional (this is for cases where the lifetime and
 * deallocation is controlled differently. If NULL-pointers are given for both
 * scratch_malloc_fn and scratch_free_fn, the default allocator is adopted for
 * scratch memory allocation and release.
 * It is supported to change the allocator while buffers are pending.
 */
LIBXSMM_API void libxsmm_set_allocator(/* malloc_fn/free_fn must correspond */
  libxsmm_malloc_function default_malloc_fn, libxsmm_free_function default_free_fn,
  libxsmm_malloc_function scratch_malloc_fn, libxsmm_free_function scratch_free_fn);

/** Allocate aligned default memory. */
LIBXSMM_API void* libxsmm_aligned_malloc(size_t size,
  /**
   * =0: align automatically according to the size
   * 0<: align according to the alignment value
   */
  size_t alignment);

/** Allocate aligned scratch memory. */
LIBXSMM_API void* libxsmm_aligned_scratch(size_t size,
  /**
   * =0: align automatically according to the size
   * 0<: align according to the alignment value
   */
  size_t alignment);

/** Allocate memory (malloc/free interface). */
LIBXSMM_API void* libxsmm_malloc(size_t size);

/** Deallocate memory (malloc/free interface). */
LIBXSMM_API void libxsmm_free(const void* memory);

/** Get the size of the allocated memory; zero in case of an error. */
LIBXSMM_API size_t libxsmm_malloc_size(const void* memory);

#endif /*LIBXSMM_MALLOC_H*/

