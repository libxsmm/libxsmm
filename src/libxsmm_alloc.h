/******************************************************************************
** Copyright (c) 2014-2016, Intel Corporation                                **
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
#ifndef LIBXSMM_ALLOC_H
#define LIBXSMM_ALLOC_H

#include <libxsmm.h>


typedef enum libxsmm_alloc_flags {
  LIBXSMM_ALLOC_FLAG_R = 1,
  LIBXSMM_ALLOC_FLAG_W = 2,
  LIBXSMM_ALLOC_FLAG_X = 4,
  LIBXSMM_ALLOC_FLAG_RWX = LIBXSMM_ALLOC_FLAG_R | LIBXSMM_ALLOC_FLAG_W | LIBXSMM_ALLOC_FLAG_X,
  LIBXSMM_ALLOC_FLAG_RW  = LIBXSMM_ALLOC_FLAG_R | LIBXSMM_ALLOC_FLAG_W,
  /** LIBXSMM_ALLOC_FLAG_DEFAULT is an alias for setting no flag bits. */
  LIBXSMM_ALLOC_FLAG_DEFAULT = LIBXSMM_ALLOC_FLAG_RW
} libxsmm_alloc_flags;

LIBXSMM_API size_t libxsmm_gcd(size_t a, size_t b);
LIBXSMM_API size_t libxsmm_lcm(size_t a, size_t b);
LIBXSMM_API size_t libxsmm_alignment(size_t size, size_t alignment);

/** Receive the size, the flags, or the extra attachment of the given buffer. */
LIBXSMM_API int libxsmm_alloc_info(const void* memory, size_t* size, int* flags, void** extra);

/** Allocate memory of the requested size, which is aligned according to the given alignment. */
LIBXSMM_API int libxsmm_allocate(void** memory, size_t size, size_t alignment, int flags,
  /* The extra information is stored along with the allocated chunk; can be NULL/zero. */
  const void* extra, size_t extra_size);
LIBXSMM_API int libxsmm_deallocate(const void* memory);

/** Attribute memory allocation such as to revoke protection flags. */
LIBXSMM_API int libxsmm_alloc_attribute(const void* memory, int flags, const char* name);

/** Allocate aligned memory (malloc/free interface). */
LIBXSMM_API_INLINE void* libxsmm_aligned_malloc(size_t size, size_t alignment)
#if defined(LIBXSMM_BUILD)
;
#else
{ void* result = 0;
  return 0 == libxsmm_allocate(&result, size, alignment, LIBXSMM_ALLOC_FLAG_DEFAULT,
    0/*extra*/, 0/*extra_size*/) ? result : 0;
}
#endif

/** Deallocate memory (malloc/free interface). */
LIBXSMM_API_INLINE void libxsmm_aligned_free(const void* memory)
#if defined(LIBXSMM_BUILD)
;
#else
{ libxsmm_deallocate(memory); }
#endif

/** Allocate memory (malloc/free interface). */
LIBXSMM_API_INLINE void* libxsmm_malloc(size_t size)
#if defined(LIBXSMM_BUILD)
;
#else
{ return libxsmm_aligned_malloc(size, 0/*auto*/); }
#endif

/** Deallocate memory (malloc/free interface). */
LIBXSMM_API_INLINE void libxsmm_free(const void* memory)
#if defined(LIBXSMM_BUILD)
;
#else
{ libxsmm_aligned_free(memory); }
#endif

#endif /*LIBXSMM_ALLOC_H*/

