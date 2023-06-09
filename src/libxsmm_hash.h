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
#ifndef LIBXSMM_HASH_H
#define LIBXSMM_HASH_H

#include <libxsmm_macros.h>

/** Map number of Bits to corresponding routine. */
#define LIBXSMM_CRC32U(N) LIBXSMM_CONCATENATE(libxsmm_crc32_u, N)
/** Calculate CRC32-value of the given pointer. */
#define LIBXSMM_CRCPTR(SEED, PTR) LIBXSMM_CRC32U(LIBXSMM_BITS)(SEED, &(PTR))
/** Map number of Bytes to number of bits. */
#define LIBXSMM_CRC32(N) LIBXSMM_CONCATENATE(libxsmm_crc32_b, N)
#define libxsmm_crc32_b1 libxsmm_crc32_u8
#define libxsmm_crc32_b2 libxsmm_crc32_u16
#define libxsmm_crc32_b4 libxsmm_crc32_u32
#define libxsmm_crc32_b8 libxsmm_crc32_u64
#define libxsmm_crc32_b16 libxsmm_crc32_u128
#define libxsmm_crc32_b32 libxsmm_crc32_u256
#define libxsmm_crc32_b48 libxsmm_crc32_u384
#define libxsmm_crc32_b64 libxsmm_crc32_u512


/** Function type representing the CRC32 functionality. */
LIBXSMM_EXTERN_C typedef unsigned int (*libxsmm_hash_function)(
  unsigned int /*seed*/, const void* /*data*/, ... /*size*/);

/** Initialize hash function module; not thread-safe. */
LIBXSMM_API_INTERN void libxsmm_hash_init(int target_arch);
LIBXSMM_API_INTERN void libxsmm_hash_finalize(void);

LIBXSMM_API_INTERN unsigned int libxsmm_crc32_u8(unsigned int seed, const void* value, ...);
LIBXSMM_API_INTERN unsigned int libxsmm_crc32_u16(unsigned int seed, const void* value, ...);
LIBXSMM_API_INTERN unsigned int libxsmm_crc32_u32(unsigned int seed, const void* value, ...);
LIBXSMM_API_INTERN unsigned int libxsmm_crc32_u64(unsigned int seed, const void* value, ...);
LIBXSMM_API_INTERN unsigned int libxsmm_crc32_u128(unsigned int seed, const void* value, ...);
LIBXSMM_API_INTERN unsigned int libxsmm_crc32_u256(unsigned int seed, const void* value, ...);
LIBXSMM_API_INTERN unsigned int libxsmm_crc32_u384(unsigned int seed, const void* value, ...);
LIBXSMM_API_INTERN unsigned int libxsmm_crc32_u512(unsigned int seed, const void* value, ...);

/** Calculate the CRC32 for a given quantity (size) of raw data according to the seed. */
LIBXSMM_API_INTERN unsigned int libxsmm_crc32(unsigned int seed, const void* data, size_t size);

#endif /*LIBXSMM_HASH_H*/
