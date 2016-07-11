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
  LIBXSMM_ALLOC_FLAG_X = 4
} libxsmm_alloc_flags;

LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE unsigned int libxsmm_gcd(unsigned int a, unsigned int b);
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE unsigned int libxsmm_lcm(unsigned int a, unsigned int b);
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE unsigned int libxsmm_alignment(unsigned int size, unsigned int alignment);

/** Receive the size or the extra attachment of the given buffer. */
int libxsmm_alloc_info(const void* memory, unsigned int* size, void** extra);

/** Allocate memory of the requested size, which is aligned according to the given alignment. */
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE int libxsmm_allocate(void** memory, unsigned int size, unsigned int alignment,
  /* The extra information is stored along with the allocated chunk; can be NULL/zero. */
  const void* extra, unsigned int extra_size);
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE int libxsmm_deallocate(const void* memory);

/** Allocate memory (malloc/free interface). */
LIBXSMM_INLINE_EXPORT LIBXSMM_RETARGETABLE void* libxsmm_malloc(unsigned int size)
#if defined(LIBXSMM_BUILD)
;
#else
{ void* result = 0; return 0 == libxsmm_allocate(&result, size, 0, 0/*extra*/, 0/*extra_size*/) ? result : 0; }
#endif

/** Deallocate memory (malloc/free interface). */
LIBXSMM_INLINE_EXPORT LIBXSMM_RETARGETABLE void libxsmm_free(const void* memory)
#if defined(LIBXSMM_BUILD)
;
#else
{ libxsmm_deallocate(memory); }
#endif

#endif /*LIBXSMM_ALLOC_H*/

