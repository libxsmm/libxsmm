/******************************************************************************
** Copyright (c) 2015-2019, Intel Corporation                                **
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

/* Map number of Bits to corresponding routine. */
#define LIBXSMM_CRC32U(N) LIBXSMM_CONCATENATE(libxsmm_crc32_u, N)
/* Map number of Bytes to number of bits. */
#define LIBXSMM_CRC32(N) LIBXSMM_CONCATENATE(libxsmm_crc32_b, N)
#define libxsmm_crc32_b4 libxsmm_crc32_u32
#define libxsmm_crc32_b8 libxsmm_crc32_u64
#define libxsmm_crc32_b16 libxsmm_crc32_u128
#define libxsmm_crc32_b32 libxsmm_crc32_u256
#define libxsmm_crc32_b48 libxsmm_crc32_u384
#define libxsmm_crc32_b64 libxsmm_crc32_u512


/** Function type representing the CRC32 functionality. */
LIBXSMM_EXTERN_C typedef LIBXSMM_RETARGETABLE unsigned int (*libxsmm_hash_function)(
  unsigned int /*seed*/, const void* /*data*/, ... /*size*/);

/** Initialize hash function module; not thread-safe. */
LIBXSMM_API_INTERN void libxsmm_hash_init(int target_arch);
LIBXSMM_API_INTERN void libxsmm_hash_finalize(void);

LIBXSMM_API_INTERN unsigned int libxsmm_crc32_u32(unsigned int seed, const void* value, ...);
LIBXSMM_API_INTERN unsigned int libxsmm_crc32_u64(unsigned int seed, const void* value, ...);
LIBXSMM_API_INTERN unsigned int libxsmm_crc32_u128(unsigned int seed, const void* value, ...);
LIBXSMM_API_INTERN unsigned int libxsmm_crc32_u256(unsigned int seed, const void* value, ...);
LIBXSMM_API_INTERN unsigned int libxsmm_crc32_u384(unsigned int seed, const void* value, ...);
LIBXSMM_API_INTERN unsigned int libxsmm_crc32_u512(unsigned int seed, const void* value, ...);

/** Calculate the CRC32 for a given quantity (size) of raw data according to the seed. */
LIBXSMM_API_INTERN unsigned int libxsmm_crc32(unsigned int seed, const void* data, size_t size);

#endif /*LIBXSMM_HASH_H*/

