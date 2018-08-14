/******************************************************************************
** Copyright (c) 2015-2018, Intel Corporation                                **
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
#ifndef LIBXSMM_HASH_H
#define LIBXSMM_HASH_H

#include <libxsmm.h>

#if !defined(LIBXSMM_HASH_SW) && 0
# define LIBXSMM_HASH_SW
#endif

#if defined(LIBXSMM_BUILD) && !defined(LIBXSMM_HASH_NOINLINE)
# define LIBXSMM_HASH_API LIBXSMM_API_INLINE
# define LIBXSMM_HASH_API_DEFINITION LIBXSMM_HASH_API LIBXSMM_ATTRIBUTE_UNUSED
#else
# define LIBXSMM_HASH_API LIBXSMM_API
# define LIBXSMM_HASH_API_DEFINITION LIBXSMM_API
#endif


/** Function type representing the CRC32 functionality (elemental form; 32-bit). */
LIBXSMM_EXTERN_C typedef LIBXSMM_RETARGETABLE unsigned int (*libxsmm_hash_u32_function)(
  unsigned int, unsigned int);
/** Function type representing the CRC32 functionality (elemental form; 64-bit). */
LIBXSMM_EXTERN_C typedef LIBXSMM_RETARGETABLE unsigned int (*libxsmm_hash_u64_function)(
  unsigned int, unsigned long long);
/** Function type representing the CRC32 functionality (taking an entire buffer). */
LIBXSMM_EXTERN_C typedef LIBXSMM_RETARGETABLE unsigned int (*libxsmm_hash_function)(
  const void*, size_t, unsigned int);

/** Initialize hash function module; not thread-safe. */
LIBXSMM_HASH_API void libxsmm_hash_init(int target_arch);
LIBXSMM_HASH_API void libxsmm_hash_finalize(void);

LIBXSMM_HASH_API unsigned int libxsmm_crc32_u32(unsigned int seed, unsigned int value);
LIBXSMM_HASH_API unsigned int libxsmm_crc32_u32_sw(unsigned int seed, unsigned int value);
LIBXSMM_HASH_API unsigned int libxsmm_crc32_u32_sse4(unsigned int seed, unsigned int value);

LIBXSMM_HASH_API unsigned int libxsmm_crc32_u64(unsigned int seed, unsigned long long value);
LIBXSMM_HASH_API unsigned int libxsmm_crc32_u64_sw(unsigned int seed, unsigned long long value);
LIBXSMM_HASH_API unsigned int libxsmm_crc32_u64_sse4(unsigned int seed, unsigned long long value);

/** Dispatched implementation which may (or may not) use a SIMD extension. */
LIBXSMM_HASH_API unsigned int libxsmm_crc32(const void* data, size_t size, unsigned int seed);
/** Calculate the CRC32 for a given quantity (size) of raw data according to the seed. */
LIBXSMM_HASH_API unsigned int libxsmm_crc32_sw(const void* data, size_t size, unsigned int seed);
/** Similar to libxsmm_crc32_sw (uses CRC32 instructions available since SSE4.2). */
LIBXSMM_HASH_API unsigned int libxsmm_crc32_sse4(const void* data, size_t size, unsigned int seed);


#if defined(LIBXSMM_BUILD) && !defined(LIBXSMM_HASH_NOINLINE)
# include "libxsmm_hash.c"
#endif

#endif /*LIBXSMM_HASH_H*/
