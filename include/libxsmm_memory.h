/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Hans Pabst (Intel Corp.)
******************************************************************************/
#ifndef LIBXSMM_MEMORY_H
#define LIBXSMM_MEMORY_H

#include "libxsmm_macros.h"

#if defined(__clang_analyzer__)
# define LIBXSMM_MEMSET127(PTRDST, VALUE, SIZE) memset((void*)(PTRDST), VALUE, SIZE)
#else
# define LIBXSMM_MEMSET127(PTRDST, VALUE, SIZE) do { \
  char *const libxsmm_memset127_dst_ = (char*)(PTRDST); \
  union { size_t size; signed char size1; } libxsmm_memset127_; \
  signed char libxsmm_memset127_i_; LIBXSMM_ASSERT((SIZE) <= 127); \
  libxsmm_memset127_.size = (SIZE); \
  LIBXSMM_PRAGMA_UNROLL \
  for (libxsmm_memset127_i_ = 0; libxsmm_memset127_i_ < libxsmm_memset127_.size1; \
    ++libxsmm_memset127_i_) \
  { \
    libxsmm_memset127_dst_[libxsmm_memset127_i_] = (char)(VALUE); \
  } \
} while(0)
#endif
#define LIBXSMM_MEMZERO127(PTRDST) LIBXSMM_MEMSET127(PTRDST, '\0', sizeof(*(PTRDST)))

#define LIBXSMM_MEMCPY127_LOOP(PTRDST, PTRSRC, SIZE, NTS) do { \
  const unsigned char *const libxsmm_memcpy127_loop_src_ = (const unsigned char*)(PTRSRC); \
  unsigned char *const libxsmm_memcpy127_loop_dst_ = (unsigned char*)(PTRDST); \
  signed char libxsmm_memcpy127_loop_i_; LIBXSMM_ASSERT((SIZE) <= 127); \
  NTS(libxsmm_memcpy127_loop_dst_) LIBXSMM_PRAGMA_UNROLL \
  for (libxsmm_memcpy127_loop_i_ = 0; libxsmm_memcpy127_loop_i_ < (signed char)(SIZE); \
    ++libxsmm_memcpy127_loop_i_) \
  { \
    libxsmm_memcpy127_loop_dst_[libxsmm_memcpy127_loop_i_] = \
    libxsmm_memcpy127_loop_src_[libxsmm_memcpy127_loop_i_]; \
  } \
} while(0)
#define LIBXSMM_MEMCPY127_NTS(...)
#define LIBXSMM_MEMCPY127(PTRDST, PTRSRC, SIZE) \
  LIBXSMM_MEMCPY127_LOOP(PTRDST, PTRSRC, SIZE, LIBXSMM_MEMCPY127_NTS)
#define LIBXSMM_ASSIGN127(PTRDST, PTRSRC) do { \
  LIBXSMM_ASSERT(sizeof(*(PTRSRC)) <= sizeof(*(PTRDST))); \
  LIBXSMM_MEMCPY127(PTRDST, PTRSRC, sizeof(*(PTRSRC))); \
} while(0)


/**
 * Calculates if there is a difference between two (short) buffers.
 * Returns zero if there is no difference; otherwise non-zero.
 */
LIBXSMM_API unsigned char libxsmm_diff(const void* a, const void* b, unsigned char size);

/**
 * Calculates if there is a difference between "a" and "n x b".
 * Returns the index of the first match (or "n" in case of no match).
 */
LIBXSMM_API unsigned int libxsmm_diff_n(const void* a, const void* bn, unsigned char size,
  unsigned char stride, unsigned int hint, unsigned int n);

/** Similar to memcmp (C standard library), but the result is conceptually only a boolean. */
LIBXSMM_API int libxsmm_memcmp(const void* a, const void* b, size_t size);

/** Calculate a hash value for the given buffer and seed; accepts NULL-buffer. */
LIBXSMM_API unsigned int libxsmm_hash(const void* data, unsigned int size, unsigned int seed);

/** Calculate a 64-bit hash for the given character string; accepts NULL-string. */
LIBXSMM_API unsigned long long libxsmm_hash_string(const char string[]);

/** Return the pointer to the 1st match of "b" in "a", or NULL (no match). */
LIBXSMM_API const char* libxsmm_stristr(const char a[], const char b[]);

/**
 * Check if pointer is SIMD-aligned and optionally consider the next access (increment in Bytes).
 * Optionally calculates the alignment of the given pointer in Bytes.
 */
LIBXSMM_API int libxsmm_aligned(const void* ptr, const size_t* inc, int* alignment);

#endif /*LIBXSMM_MEMORY_H*/

