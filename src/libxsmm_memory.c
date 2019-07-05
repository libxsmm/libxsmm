/******************************************************************************
** Copyright (c) 2017-2019, Intel Corporation                                **
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
#include "libxsmm_hash.h"

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#include <string.h>
#include <stdio.h>
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif


LIBXSMM_API unsigned int libxsmm_hash(const void* data, unsigned int size, unsigned int seed)
{
  LIBXSMM_INIT
  return libxsmm_crc32(seed, data, size);
}


LIBXSMM_API unsigned long long libxsmm_hash_string(const char* string)
{
  unsigned long long result;
  const size_t length = NULL != string ? strlen(string) : 0;
  if (sizeof(result) < length) {
    const size_t length2 = length / 2;
    unsigned int seed32 = 0; /* seed=0: match else-optimization */
    LIBXSMM_INIT
    seed32 = libxsmm_crc32(seed32, string, length2);
    result = libxsmm_crc32(seed32, string + length2, length - length2);
    result = (result << 32) | seed32;
  }
  else { /* reinterpret directly as hash value */
    char *const s = (char*)&result; signed char i;
    for (i = 0; i < (signed char)length; ++i) s[i] = string[i];
    for (; i < (signed char)sizeof(result); ++i) s[i] = 0;
  }
  return result;
}


#if defined(LIBXSMM_BUILD) && !defined(LIBXSMM_NOFORTRAN)

/* implementation provided for Fortran 77 compatibility */
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_hash)(void* /*hash_seed*/, const void* /*data*/, const int* /*size*/);
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_hash)(void* hash_seed, const void* data, const int* size)
{
#if !defined(NDEBUG)
  static int error_once = 0;
  if (NULL != hash_seed && NULL != data && NULL != size && 0 <= *size)
#endif
  {
    unsigned int *const hash_seed_ui32 = (unsigned int*)hash_seed;
    *hash_seed_ui32 = (libxsmm_hash(data, (unsigned int)*size, *hash_seed_ui32) & 0x7FFFFFFF/*sign-bit*/);
  }
#if !defined(NDEBUG)
  else if (0 != libxsmm_verbosity /* library code is expected to be mute */
    && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
  {
    fprintf(stderr, "LIBXSMM ERROR: invalid arguments for libxsmm_hash specified!\n");
  }
#endif
}


/* implementation provided for Fortran 77 compatibility */
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_diff)(int* /*result*/, const void* /*a*/, const void* /*b*/, const long long* /*size*/);
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_diff)(int* result, const void* a, const void* b, const long long* size)
{
#if !defined(NDEBUG)
  static int error_once = 0;
  if (NULL != result && NULL != a && NULL != b && NULL != size && 0 <= *size)
#endif
  {
    *result = libxsmm_memcmp(a, b, (size_t)*size);
  }
#if !defined(NDEBUG)
  else if (0 != libxsmm_verbosity /* library code is expected to be mute */
    && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
  {
    fprintf(stderr, "LIBXSMM ERROR: invalid arguments for libxsmm_memcmp specified!\n");
  }
#endif
}

#endif /*defined(LIBXSMM_BUILD)*/

