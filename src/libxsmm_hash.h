/******************************************************************************
** Copyright (c) 2015-2016, Intel Corporation                                **
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

#if !defined(LIBXSMM_HASH_SW)
/*# define LIBXSMM_HASH_SW*/
#endif


/** Function type representing the CRC32 functionality. */
typedef LIBXSMM_RETARGETABLE unsigned int (*libxsmm_hash_function)(const void*, unsigned int, unsigned int);

/** Initialize hash function module; not thread-safe. */
LIBXSMM_API void libxsmm_hash_init(int target_arch);
LIBXSMM_API void libxsmm_hash_finalize(void);

/** Dispatched implementation which may (or may not) use a SIMD extension. */
LIBXSMM_API unsigned int libxsmm_crc32(
  const void* data, unsigned int size, unsigned int seed);

/** Calculate the CRC32 for a given quantity (size) of raw data according to the seed. */
LIBXSMM_API unsigned int libxsmm_crc32_sw(
  const void* data, unsigned int size, unsigned int seed);

/** Similar to libxsmm_crc32_sw (uses CRC32 instructions available since SSE4.2). */
LIBXSMM_API unsigned int libxsmm_crc32_sse42(
  const void* data, unsigned int size, unsigned int seed);

/** Calculate a hash value for a given quantity (size) of raw data according to the seed. */
LIBXSMM_API unsigned int libxsmm_hash(
  const void* data, unsigned int size,
  /** Upper bound of the result. */
  unsigned int n);

/** Calculate a hash value for a given quantity (size) of raw data according to the seed. */
LIBXSMM_API unsigned int libxsmm_hash_npot(
  const void* data, unsigned int size,
  /** Upper bound of the result. */
  unsigned int npot);

#endif /*LIBXSMM_HASH_H*/
