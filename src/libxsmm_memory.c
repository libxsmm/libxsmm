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
#include <libxsmm_memory.h>
#include "libxsmm_hash.h"
#include "libxsmm_diff.h"
#include "libxsmm_main.h"

#if !defined(LIBXSMM_MEMORY_STDLIB) && 0
# define LIBXSMM_MEMORY_STDLIB
#endif
#if !defined(LIBXSMM_MEMORY_SW) && 0
# define LIBXSMM_MEMORY_SW
#endif


#if !defined(LIBXSMM_MEMORY_SW)
LIBXSMM_APIVAR_DEFINE(unsigned char (*internal_diff_function)(const void*, const void*, unsigned char));
LIBXSMM_APIVAR_DEFINE(int (*internal_memcmp_function)(const void*, const void*, size_t));
#endif


LIBXSMM_API_INLINE
unsigned char internal_diff_sw(const void* a, const void* b, unsigned char size)
{
#if defined(LIBXSMM_MEMORY_STDLIB) && defined(LIBXSMM_MEMORY_SW)
  return (unsigned char)memcmp(a, b, size);
#else
  const uint8_t *const a8 = (const uint8_t*)a, *const b8 = (const uint8_t*)b;
  unsigned char i;
  LIBXSMM_PRAGMA_UNROLL/*_N(2)*/
  for (i = 0; i < (size & 0xF0); i += 16) {
    LIBXSMM_DIFF_16_DECL(aa);
    LIBXSMM_DIFF_16_LOAD(aa, a8 + i);
    if (LIBXSMM_DIFF_16(aa, b8 + i, 0/*dummy*/)) return 1;
  }
  for (; i < size; ++i) if (a8[i] ^ b8[i]) return 1;
  return 0;
#endif
}


LIBXSMM_API_INLINE LIBXSMM_INTRINSICS(LIBXSMM_X86_SSE3)
unsigned char internal_diff_sse3(const void* a, const void* b, unsigned char size)
{
#if defined(LIBXSMM_INTRINSICS_SSE3) && !defined(LIBXSMM_MEMORY_SW)
  const uint8_t *const a8 = (const uint8_t*)a, *const b8 = (const uint8_t*)b;
  unsigned char i;
  LIBXSMM_PRAGMA_UNROLL/*_N(2)*/
  for (i = 0; i < (size & 0xF0); i += 16) {
    LIBXSMM_DIFF_SSE3_DECL(aa);
    LIBXSMM_DIFF_SSE3_LOAD(aa, a8 + i);
    if (LIBXSMM_DIFF_SSE3(aa, b8 + i, 0/*dummy*/)) return 1;
  }
  for (; i < size; ++i) if (a8[i] ^ b8[i]) return 1;
  return 0;
#else
  return internal_diff_sw(a, b, size);
#endif
}


LIBXSMM_API_INLINE LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX2)
unsigned char internal_diff_avx2(const void* a, const void* b, unsigned char size)
{
#if defined(LIBXSMM_INTRINSICS_AVX2) && !defined(LIBXSMM_MEMORY_SW)
  const uint8_t *const a8 = (const uint8_t*)a, *const b8 = (const uint8_t*)b;
  unsigned char i;
  LIBXSMM_PRAGMA_UNROLL/*_N(2)*/
  for (i = 0; i < (size & 0xE0); i += 32) {
    LIBXSMM_DIFF_AVX2_DECL(aa);
    LIBXSMM_DIFF_AVX2_LOAD(aa, a8 + i);
    if (LIBXSMM_DIFF_AVX2(aa, b8 + i, 0/*dummy*/)) return 1;
  }
  for (; i < size; ++i) if (a8[i] ^ b8[i]) return 1;
  return 0;
#else
  return internal_diff_sw(a, b, size);
#endif
}


LIBXSMM_API_INLINE LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512)
unsigned char internal_diff_avx512(const void* a, const void* b, unsigned char size)
{
#if defined(LIBXSMM_INTRINSICS_AVX512) && !defined(LIBXSMM_MEMORY_SW)
  const uint8_t *const a8 = (const uint8_t*)a, *const b8 = (const uint8_t*)b;
  unsigned char i;
  LIBXSMM_PRAGMA_UNROLL/*_N(2)*/
  for (i = 0; i < (size & 0xC0); i += 64) {
    LIBXSMM_DIFF_AVX512_DECL(aa);
    LIBXSMM_DIFF_AVX512_LOAD(aa, a8 + i);
    if (LIBXSMM_DIFF_AVX512(aa, b8 + i, 0/*dummy*/)) return 1;
  }
  for (; i < size; ++i) if (a8[i] ^ b8[i]) return 1;
  return 0;
#else
  return internal_diff_sw(a, b, size);
#endif
}


LIBXSMM_API_INLINE
int internal_memcmp_sw(const void* a, const void* b, size_t size)
{
#if defined(LIBXSMM_MEMORY_STDLIB)
  return memcmp(a, b, size);
#else
  const uint8_t *const a8 = (const uint8_t*)a, *const b8 = (const uint8_t*)b;
  size_t i;
  LIBXSMM_DIFF_16_DECL(aa);
  LIBXSMM_PRAGMA_UNROLL/*_N(2)*/
  for (i = 0; i < (size & 0xFFFFFFFFFFFFFFF0); i += 16) {
    LIBXSMM_DIFF_16_LOAD(aa, a8 + i);
    if (LIBXSMM_DIFF_16(aa, b8 + i, 0/*dummy*/)) return 1;
  }
  for (; i < size; ++i) if (a8[i] ^ b8[i]) return 1;
  return 0;
#endif
}


LIBXSMM_API_INLINE LIBXSMM_INTRINSICS(LIBXSMM_X86_SSE3)
int internal_memcmp_sse3(const void* a, const void* b, size_t size)
{
#if defined(LIBXSMM_INTRINSICS_SSE3) && !defined(LIBXSMM_MEMORY_SW)
  const uint8_t *const a8 = (const uint8_t*)a, *const b8 = (const uint8_t*)b;
  size_t i;
  LIBXSMM_DIFF_SSE3_DECL(aa);
  LIBXSMM_PRAGMA_UNROLL/*_N(2)*/
  for (i = 0; i < (size & 0xFFFFFFFFFFFFFFF0); i += 16) {
    LIBXSMM_DIFF_SSE3_LOAD(aa, a8 + i);
    if (LIBXSMM_DIFF_SSE3(aa, b8 + i, 0/*dummy*/)) return 1;
  }
  for (; i < size; ++i) if (a8[i] ^ b8[i]) return 1;
  return 0;
#else
  return internal_memcmp_sw(a, b, size);
#endif
}


LIBXSMM_API_INLINE LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX2)
int internal_memcmp_avx2(const void* a, const void* b, size_t size)
{
#if defined(LIBXSMM_INTRINSICS_AVX2) && !defined(LIBXSMM_MEMORY_SW)
  const uint8_t *const a8 = (const uint8_t*)a, *const b8 = (const uint8_t*)b;
  size_t i;
  LIBXSMM_DIFF_AVX2_DECL(aa);
  LIBXSMM_PRAGMA_UNROLL/*_N(2)*/
  for (i = 0; i < (size & 0xFFFFFFFFFFFFFFE0); i += 32) {
    LIBXSMM_DIFF_AVX2_LOAD(aa, a8 + i);
    if (LIBXSMM_DIFF_AVX2(aa, b8 + i, 0/*dummy*/)) return 1;
  }
  for (; i < size; ++i) if (a8[i] ^ b8[i]) return 1;
  return 0;
#else
  return internal_memcmp_sw(a, b, size);
#endif
}


LIBXSMM_API_INLINE LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512)
int internal_memcmp_avx512(const void* a, const void* b, size_t size)
{
#if defined(LIBXSMM_INTRINSICS_AVX512) && !defined(LIBXSMM_MEMORY_SW)
  const uint8_t *const a8 = (const uint8_t*)a, *const b8 = (const uint8_t*)b;
  size_t i;
  LIBXSMM_DIFF_AVX512_DECL(aa);
  LIBXSMM_PRAGMA_UNROLL/*_N(2)*/
  for (i = 0; i < (size & 0xFFFFFFFFFFFFFFC0); i += 64) {
    LIBXSMM_DIFF_AVX512_LOAD(aa, a8 + i);
    if (LIBXSMM_DIFF_AVX512(aa, b8 + i, 0/*dummy*/)) return 1;
  }
  for (; i < size; ++i) if (a8[i] ^ b8[i]) return 1;
  return 0;
#else
  return internal_memcmp_sw(a, b, size);
#endif
}


LIBXSMM_API_INTERN void libxsmm_memory_init(int target_arch)
{
#if defined(LIBXSMM_MEMORY_SW)
  LIBXSMM_UNUSED(target_arch);
#else
  if (LIBXSMM_X86_AVX512 <= target_arch) {
# if defined(LIBXSMM_DIFF_AVX512_ENABLED)
    internal_diff_function = internal_diff_avx512;
# else
    internal_diff_function = internal_diff_avx2;
# endif
# if defined(LIBXSMM_DIFF_AVX512_ENABLED)
    internal_memcmp_function = internal_memcmp_avx512;
# else
    internal_memcmp_function = internal_memcmp_avx2;
# endif
  }
  else if (LIBXSMM_X86_AVX2 <= target_arch) {
    internal_diff_function = internal_diff_avx2;
    internal_memcmp_function = internal_memcmp_avx2;
  }
  else if (LIBXSMM_X86_SSE3 <= target_arch) {
    internal_diff_function = internal_diff_sse3;
    internal_memcmp_function = internal_memcmp_sse3;
  }
  else {
    internal_diff_function = internal_diff_sw;
    internal_memcmp_function = internal_memcmp_sw;
  }
  LIBXSMM_ASSERT(NULL != internal_diff_function);
  LIBXSMM_ASSERT(NULL != internal_memcmp_function);
#endif
}


LIBXSMM_API_INTERN void libxsmm_memory_finalize(void)
{
#if !defined(NDEBUG) && !defined(LIBXSMM_MEMORY_SW)
  internal_diff_function = NULL;
  internal_memcmp_function = NULL;
#endif
}


LIBXSMM_API unsigned char libxsmm_diff_16(const void* a, const void* b, ...)
{
#if defined(LIBXSMM_MEMORY_SW)
  return internal_diff_sw(a, b, 16);
#else
  LIBXSMM_DIFF_16_DECL(a16);
  LIBXSMM_DIFF_16_LOAD(a16, a);
  return LIBXSMM_DIFF_16(a16, b, 0/*dummy*/);
#endif
}


LIBXSMM_API unsigned char libxsmm_diff_32(const void* a, const void* b, ...)
{
#if defined(LIBXSMM_MEMORY_SW)
  return internal_diff_sw(a, b, 32);
#else
  LIBXSMM_DIFF_32_DECL(a32);
  LIBXSMM_DIFF_32_LOAD(a32, a);
  return LIBXSMM_DIFF_32(a32, b, 0/*dummy*/);
#endif
}


LIBXSMM_API unsigned char libxsmm_diff_48(const void* a, const void* b, ...)
{
#if defined(LIBXSMM_MEMORY_SW)
  return internal_diff_sw(a, b, 48);
#else
  LIBXSMM_DIFF_48_DECL(a48);
  LIBXSMM_DIFF_48_LOAD(a48, a);
  return LIBXSMM_DIFF_48(a48, b, 0/*dummy*/);
#endif
}


LIBXSMM_API unsigned char libxsmm_diff_64(const void* a, const void* b, ...)
{
#if defined(LIBXSMM_MEMORY_SW)
  return internal_diff_sw(a, b, 64);
#else
  LIBXSMM_DIFF_64_DECL(a64);
  LIBXSMM_DIFF_64_LOAD(a64, a);
  return LIBXSMM_DIFF_64(a64, b, 0/*dummy*/);
#endif
}


LIBXSMM_API unsigned char libxsmm_diff(const void* a, const void* b, unsigned char size)
{
#if defined(LIBXSMM_MEMORY_SW) && !defined(LIBXSMM_MEMORY_STDLIB)
  return internal_diff_sw(a, b, size);
#else
# if defined(LIBXSMM_MEMORY_STDLIB)
  return 0 != memcmp(a, b, size);
# elif (LIBXSMM_X86_AVX512 <= LIBXSMM_STATIC_TARGET_ARCH) && defined(LIBXSMM_DIFF_AVX512_ENABLED)
  return internal_diff_avx512(a, b, size);
# elif (LIBXSMM_X86_AVX2 <= LIBXSMM_STATIC_TARGET_ARCH)
  return internal_diff_avx2(a, b, size);
# elif (LIBXSMM_X86_SSE3 <= LIBXSMM_STATIC_TARGET_ARCH)
# if (LIBXSMM_X86_AVX2 > LIBXSMM_MAX_STATIC_TARGET_ARCH)
  return internal_diff_sse3(a, b, size);
# else /* pointer based function call */
# if defined(LIBXSMM_INIT_COMPLETED)
  LIBXSMM_ASSERT(NULL != internal_diff_function);
  return internal_diff_function(a, b, size);
# else
  return (unsigned char)(NULL != internal_diff_function
    ? internal_diff_function(a, b, size)
    : internal_diff_sse3(a, b, size));
# endif
# endif
# else
  return internal_diff_sw(a, b, size);
# endif
#endif
}


LIBXSMM_API unsigned int libxsmm_diff_n(const void* a, const void* bn, unsigned char size,
  unsigned char stride, unsigned int hint, unsigned int n)
{
  unsigned int result;
  LIBXSMM_ASSERT(size <= stride);
#if defined(LIBXSMM_MEMORY_STDLIB) && !defined(LIBXSMM_MEMORY_SW)
  LIBXSMM_DIFF_N(unsigned int, result, memcmp, a, bn, size, stride, hint, n);
#else
# if !defined(LIBXSMM_MEMORY_SW)
  switch (size) {
    case 64: {
      LIBXSMM_DIFF_64_DECL(a64);
      LIBXSMM_DIFF_64_LOAD(a64, a);
      LIBXSMM_DIFF_N(unsigned int, result, LIBXSMM_DIFF_64, a64, bn, size, stride, hint, n);
    } break;
    case 48: {
      LIBXSMM_DIFF_48_DECL(a48);
      LIBXSMM_DIFF_48_LOAD(a48, a);
      LIBXSMM_DIFF_N(unsigned int, result, LIBXSMM_DIFF_48, a48, bn, size, stride, hint, n);
    } break;
    case 32: {
      LIBXSMM_DIFF_32_DECL(a32);
      LIBXSMM_DIFF_32_LOAD(a32, a);
      LIBXSMM_DIFF_N(unsigned int, result, LIBXSMM_DIFF_32, a32, bn, size, stride, hint, n);
    } break;
    case 16: {
      LIBXSMM_DIFF_16_DECL(a16);
      LIBXSMM_DIFF_16_LOAD(a16, a);
      LIBXSMM_DIFF_N(unsigned int, result, LIBXSMM_DIFF_16, a16, bn, size, stride, hint, n);
    } break;
    default:
# endif
    {
      LIBXSMM_DIFF_N(unsigned int, result, libxsmm_diff, a, bn, size, stride, hint, n);
    }
# if !defined(LIBXSMM_MEMORY_SW)
  }
# endif
#endif
  return result;
}


LIBXSMM_API int libxsmm_memcmp(const void* a, const void* b, size_t size)
{
#if defined(LIBXSMM_MEMORY_SW) && !defined(LIBXSMM_MEMORY_STDLIB)
  return internal_memcmp_sw(a, b, size);
#else
# if defined(LIBXSMM_MEMORY_STDLIB)
  return memcmp(a, b, size);
# elif (LIBXSMM_X86_AVX512 <= LIBXSMM_STATIC_TARGET_ARCH) && defined(LIBXSMM_DIFF_AVX512_ENABLED)
  return internal_memcmp_avx512(a, b, size);
# elif (LIBXSMM_X86_AVX2 <= LIBXSMM_STATIC_TARGET_ARCH)
  return internal_memcmp_avx2(a, b, size);
# elif (LIBXSMM_X86_SSE3 <= LIBXSMM_STATIC_TARGET_ARCH)
# if (LIBXSMM_X86_AVX2 > LIBXSMM_MAX_STATIC_TARGET_ARCH)
  return internal_memcmp_sse3(a, b, size);
# else /* pointer based function call */
# if defined(LIBXSMM_INIT_COMPLETED)
  LIBXSMM_ASSERT(NULL != internal_memcmp_function);
  return internal_memcmp_function(a, b, size);
# else
  return NULL != internal_memcmp_function
    ? internal_memcmp_function(a, b, size)
    : internal_memcmp_sse3(a, b, size);
# endif
# endif
# else
  return internal_memcmp_sw(a, b, size);
# endif
#endif
}


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


#if defined(LIBXSMM_BUILD) && (!defined(LIBXSMM_NOFORTRAN) || defined(__clang_analyzer__))

/* implementation provided for Fortran 77 compatibility */
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_xhash)(int* /*hash_seed*/, const void* /*data*/, const int* /*size*/);
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_xhash)(int* hash_seed, const void* data, const int* size)
{
#if !defined(NDEBUG)
  static int error_once = 0;
  if (NULL != hash_seed && NULL != data && NULL != size && 0 <= *size)
#endif
  {
    *hash_seed = (int)(libxsmm_hash(data, (unsigned int)*size, (unsigned int)*hash_seed) & 0x7FFFFFFF/*sign-bit*/);
  }
#if !defined(NDEBUG)
  else if (0 != libxsmm_verbosity /* library code is expected to be mute */
    && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
  {
    fprintf(stderr, "LIBXSMM ERROR: invalid arguments for libxsmm_xhash specified!\n");
  }
#endif
}


/* implementation provided for Fortran 77 compatibility */
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_xdiff)(_Bool* /*result*/, const void* /*a*/, const void* /*b*/, const long long* /*size*/);
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_xdiff)(_Bool* result, const void* a, const void* b, const long long* size)
{
#if !defined(NDEBUG)
  static int error_once = 0;
  if (NULL != result && NULL != a && NULL != b && NULL != size && 0 <= *size)
#endif
  {
    *result = (_Bool)libxsmm_memcmp(a, b, (size_t)*size);
  }
#if !defined(NDEBUG)
  else if (0 != libxsmm_verbosity /* library code is expected to be mute */
    && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
  {
    fprintf(stderr, "LIBXSMM ERROR: invalid arguments for libxsmm_xdiff specified!\n");
  }
#endif
}


/* implementation provided for Fortran 77 compatibility */
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_xclear)(void* /*dst*/, const int* /*size*/);
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_xclear)(void* dst, const int* size)
{
#if !defined(NDEBUG)
  static int error_once = 0;
  if (NULL != dst && NULL != size && 0 <= *size && 128 > *size)
#endif
  {
    LIBXSMM_MEMSET127(dst, 0, *size);
  }
#if !defined(NDEBUG)
  else if (0 != libxsmm_verbosity /* library code is expected to be mute */
    && 1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED))
  {
    fprintf(stderr, "LIBXSMM ERROR: invalid arguments for libxsmm_xclear specified!\n");
  }
#endif
}

#endif /*defined(LIBXSMM_BUILD) && (!defined(LIBXSMM_NOFORTRAN) || defined(__clang_analyzer__))*/

