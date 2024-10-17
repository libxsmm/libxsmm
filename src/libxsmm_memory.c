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
#include <libxsmm.h>
#include "libxsmm_main.h"
#include "libxsmm_hash.h"
#include "libxsmm_diff.h"
#include <ctype.h>

#if !defined(LIBXSMM_MEMORY_STDLIB) && 0
# define LIBXSMM_MEMORY_STDLIB
#endif
#if !defined(LIBXSMM_MEMORY_SW) && 0
# define LIBXSMM_MEMORY_SW
#endif

#define LIBXSMM_MEMORY_SHUFFLE_COPRIME(N) libxsmm_coprime(N, (N) / 2)
#define LIBXSMM_MEMORY_SHUFFLE(INOUT, ELEMSIZE, COUNT, SHUFFLE, NREPEAT) do { \
  unsigned char *const LIBXSMM_RESTRICT data = (unsigned char*)(INOUT); \
  const size_t c = (COUNT) - 1, c2 = ((COUNT) + 1) / 2; \
  size_t i; \
  for (i = (0 != (NREPEAT) ? 0 : (COUNT)); i < (COUNT); ++i) { \
    size_t j = i, k = 0; \
    for (; k < (NREPEAT) || j < i; ++k) j = ((SHUFFLE) * j) % (COUNT); \
    if (i < j) LIBXSMM_MEMSWP127( \
      data + (ELEMSIZE) * (c - j), \
      data + (ELEMSIZE) * (c - i), \
      ELEMSIZE); \
    if (c2 <= i) LIBXSMM_MEMSWP127( \
      data + (ELEMSIZE) * (c - i), \
      data + (ELEMSIZE) * i, \
      ELEMSIZE); \
  } \
} while(0)


#if !defined(LIBXSMM_MEMORY_SW)
LIBXSMM_APIVAR_DEFINE(unsigned char (*internal_diff_function)(const void*, const void*, unsigned char));
LIBXSMM_APIVAR_DEFINE(int (*internal_memcmp_function)(const void*, const void*, size_t));
#endif


LIBXSMM_API unsigned char libxsmm_typesize(libxsmm_datatype datatype)
{
  const unsigned char result = (unsigned char)LIBXSMM_TYPESIZE(datatype);
  if (0 != result) {
    return result;
  }
  else {
    static int error_once = 0;
    LIBXSMM_ASSERT_MSG(0, "unsupported data type");
    if (1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED)) {
      fprintf(stderr, "LIBXSMM ERROR: unsupported data type!\n");
    }
    return 1; /* avoid to return 0 to avoid div-by-zero in static analysis of depending code */
  }
}


LIBXSMM_API size_t libxsmm_offset(const size_t offset[], const size_t shape[], size_t ndims, size_t* size)
{
  size_t result = 0, size1 = 0;
  if (0 != ndims && NULL != shape) {
    size_t i;
    result = (NULL != offset ? offset[0] : 0);
    size1 = shape[0];
    for (i = 1; i < ndims; ++i) {
      result += (NULL != offset ? offset[i] : 0) * size1;
      size1 *= shape[i];
    }
  }
  if (NULL != size) *size = size1;
  return result;
}


LIBXSMM_API int libxsmm_aligned(const void* ptr, const size_t* inc, int* alignment)
{
  const int minalign = libxsmm_cpuid_vlen(libxsmm_target_archid);
  const uintptr_t address = (uintptr_t)ptr;
  int ptr_is_aligned;
  LIBXSMM_ASSERT(LIBXSMM_ISPOT(minalign));
  if (NULL == alignment) {
    ptr_is_aligned = !LIBXSMM_MOD2(address, (uintptr_t)minalign);
}
  else {
    const unsigned int nbits = LIBXSMM_INTRINSICS_BITSCANFWD64(address);
    *alignment = (32 > nbits ? (1 << nbits) : INT_MAX);
    ptr_is_aligned = (minalign <= *alignment);
  }
  return ptr_is_aligned && (NULL == inc || !LIBXSMM_MOD2(*inc, (size_t)minalign));
}


LIBXSMM_API_INLINE
unsigned char internal_diff_sw(const void* a, const void* b, unsigned char size)
{
#if defined(LIBXSMM_MEMORY_STDLIB) && defined(LIBXSMM_MEMORY_SW)
  return (unsigned char)memcmp(a, b, size);
#else
  const uint8_t *const a8 = (const uint8_t*)a, *const b8 = (const uint8_t*)b;
  unsigned char i;
  LIBXSMM_PRAGMA_UNROLL/*_N(2)*/
  for (i = 0; i < (unsigned char)(size & (unsigned char)0xF0); i += 16) {
    LIBXSMM_DIFF_16_DECL(aa);
    LIBXSMM_DIFF_16_LOAD(aa, a8 + i);
    if (LIBXSMM_DIFF_16(aa, b8 + i, 0/*dummy*/)) return 1;
  }
  for (; i < size; ++i) if (a8[i] ^ b8[i]) return 1;
  return 0;
#endif
}


LIBXSMM_API_INLINE LIBXSMM_INTRINSICS(LIBXSMM_X86_GENERIC)
unsigned char internal_diff_sse(const void* a, const void* b, unsigned char size)
{
#if defined(LIBXSMM_INTRINSICS_X86) && !defined(LIBXSMM_MEMORY_SW)
  const uint8_t *const a8 = (const uint8_t*)a, *const b8 = (const uint8_t*)b;
  unsigned char i;
  LIBXSMM_PRAGMA_UNROLL/*_N(2)*/
  for (i = 0; i < (unsigned char)(size & (unsigned char)0xF0); i += 16) {
    LIBXSMM_DIFF_SSE_DECL(aa);
    LIBXSMM_DIFF_SSE_LOAD(aa, a8 + i);
    if (LIBXSMM_DIFF_SSE(aa, b8 + i, 0/*dummy*/)) return 1;
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
  for (i = 0; i < (unsigned char)(size & (unsigned char)0xE0); i += 32) {
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


LIBXSMM_API_INLINE LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512_SKX)
unsigned char internal_diff_avx512(const void* a, const void* b, unsigned char size)
{
#if defined(LIBXSMM_INTRINSICS_AVX512_SKX) && !defined(LIBXSMM_MEMORY_SW)
  const uint8_t *const a8 = (const uint8_t*)a, *const b8 = (const uint8_t*)b;
  unsigned char i;
  LIBXSMM_PRAGMA_UNROLL/*_N(2)*/
  for (i = 0; i < (unsigned char)(size & (unsigned char)0xC0); i += 64) {
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


LIBXSMM_API_INLINE LIBXSMM_INTRINSICS(LIBXSMM_X86_GENERIC)
int internal_memcmp_sse(const void* a, const void* b, size_t size)
{
#if defined(LIBXSMM_INTRINSICS_X86) && !defined(LIBXSMM_MEMORY_SW)
  const uint8_t *const a8 = (const uint8_t*)a, *const b8 = (const uint8_t*)b;
  size_t i;
  LIBXSMM_DIFF_SSE_DECL(aa);
  LIBXSMM_PRAGMA_UNROLL/*_N(2)*/
  for (i = 0; i < (size & 0xFFFFFFFFFFFFFFF0); i += 16) {
    LIBXSMM_DIFF_SSE_LOAD(aa, a8 + i);
    if (LIBXSMM_DIFF_SSE(aa, b8 + i, 0/*dummy*/)) return 1;
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


LIBXSMM_API_INLINE LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512_SKX)
int internal_memcmp_avx512(const void* a, const void* b, size_t size)
{
#if defined(LIBXSMM_INTRINSICS_AVX512_SKX) && !defined(LIBXSMM_MEMORY_SW)
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
  if (LIBXSMM_X86_AVX512_SKX <= target_arch) {
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
  else if (LIBXSMM_X86_GENERIC <= target_arch) {
    internal_diff_function = internal_diff_sse;
    internal_memcmp_function = internal_memcmp_sse;
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
#if !defined(NDEBUG) && !defined(LIBXSMM_MEMORY_SW) && 0
  internal_diff_function = NULL;
  internal_memcmp_function = NULL;
#endif
}


LIBXSMM_API unsigned char libxsmm_diff_4(const void* a, const void* b, ...)
{
#if defined(LIBXSMM_MEMORY_SW)
  return internal_diff_sw(a, b, 4);
#else
  LIBXSMM_DIFF_4_DECL(a4);
  LIBXSMM_DIFF_4_LOAD(a4, a);
  return LIBXSMM_DIFF_4(a4, b, 0/*dummy*/);
#endif
}


LIBXSMM_API unsigned char libxsmm_diff_8(const void* a, const void* b, ...)
{
#if defined(LIBXSMM_MEMORY_SW)
  return internal_diff_sw(a, b, 8);
#else
  LIBXSMM_DIFF_8_DECL(a8);
  LIBXSMM_DIFF_8_LOAD(a8, a);
  return LIBXSMM_DIFF_8(a8, b, 0/*dummy*/);
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
# elif (LIBXSMM_X86_AVX512_SKX <= LIBXSMM_STATIC_TARGET_ARCH) && defined(LIBXSMM_DIFF_AVX512_ENABLED)
  return internal_diff_avx512(a, b, size);
# elif (LIBXSMM_X86_AVX2 <= LIBXSMM_STATIC_TARGET_ARCH)
  return internal_diff_avx2(a, b, size);
# elif (LIBXSMM_X86_SSE3 <= LIBXSMM_STATIC_TARGET_ARCH)
# if (LIBXSMM_X86_AVX2 > LIBXSMM_MAX_STATIC_TARGET_ARCH)
  return internal_diff_sse(a, b, size);
# else /* pointer based function call */
# if defined(LIBXSMM_INIT_COMPLETED)
  LIBXSMM_ASSERT(NULL != internal_diff_function);
  return internal_diff_function(a, b, size);
# else
  return (unsigned char)(NULL != internal_diff_function
    ? internal_diff_function(a, b, size)
    : internal_diff_sse(a, b, size));
# endif
# endif
# else
  return internal_diff_sw(a, b, size);
# endif
#endif
}


LIBXSMM_API unsigned int libxsmm_diff_n(const void* a, const void* bn, unsigned char elemsize,
  unsigned char stride, unsigned int hint, unsigned int count)
{
  unsigned int result;
  LIBXSMM_ASSERT(elemsize <= stride);
#if defined(LIBXSMM_MEMORY_STDLIB) && !defined(LIBXSMM_MEMORY_SW)
  LIBXSMM_DIFF_N(unsigned int, result, memcmp, a, bn, elemsize, stride, hint, count);
#else
# if !defined(LIBXSMM_MEMORY_SW)
  switch (elemsize) {
    case 64: {
      LIBXSMM_DIFF_64_DECL(a64);
      LIBXSMM_DIFF_64_LOAD(a64, a);
      LIBXSMM_DIFF_N(unsigned int, result, LIBXSMM_DIFF_64, a64, bn, 64, stride, hint, count);
    } break;
    case 48: {
      LIBXSMM_DIFF_48_DECL(a48);
      LIBXSMM_DIFF_48_LOAD(a48, a);
      LIBXSMM_DIFF_N(unsigned int, result, LIBXSMM_DIFF_48, a48, bn, 48, stride, hint, count);
    } break;
    case 32: {
      LIBXSMM_DIFF_32_DECL(a32);
      LIBXSMM_DIFF_32_LOAD(a32, a);
      LIBXSMM_DIFF_N(unsigned int, result, LIBXSMM_DIFF_32, a32, bn, 32, stride, hint, count);
    } break;
    case 16: {
      LIBXSMM_DIFF_16_DECL(a16);
      LIBXSMM_DIFF_16_LOAD(a16, a);
      LIBXSMM_DIFF_N(unsigned int, result, LIBXSMM_DIFF_16, a16, bn, 16, stride, hint, count);
    } break;
    case 8: {
      LIBXSMM_DIFF_8_DECL(a8);
      LIBXSMM_DIFF_8_LOAD(a8, a);
      LIBXSMM_DIFF_N(unsigned int, result, LIBXSMM_DIFF_8, a8, bn, 8, stride, hint, count);
    } break;
    case 4: {
      LIBXSMM_DIFF_4_DECL(a4);
      LIBXSMM_DIFF_4_LOAD(a4, a);
      LIBXSMM_DIFF_N(unsigned int, result, LIBXSMM_DIFF_4, a4, bn, 4, stride, hint, count);
    } break;
    default:
# endif
    {
      LIBXSMM_DIFF_N(unsigned int, result, libxsmm_diff, a, bn, elemsize, stride, hint, count);
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
# elif (LIBXSMM_X86_AVX512_SKX <= LIBXSMM_STATIC_TARGET_ARCH) && defined(LIBXSMM_DIFF_AVX512_ENABLED)
  return internal_memcmp_avx512(a, b, size);
# elif (LIBXSMM_X86_AVX2 <= LIBXSMM_STATIC_TARGET_ARCH)
  return internal_memcmp_avx2(a, b, size);
# elif (LIBXSMM_X86_SSE3 <= LIBXSMM_STATIC_TARGET_ARCH)
# if (LIBXSMM_X86_AVX2 > LIBXSMM_MAX_STATIC_TARGET_ARCH)
  return internal_memcmp_sse(a, b, size);
# else /* pointer based function call */
# if defined(LIBXSMM_INIT_COMPLETED)
  LIBXSMM_ASSERT(NULL != internal_memcmp_function);
  return internal_memcmp_function(a, b, size);
# else
  return NULL != internal_memcmp_function
    ? internal_memcmp_function(a, b, size)
    : internal_memcmp_sse(a, b, size);
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


LIBXSMM_API unsigned int libxsmm_hash8(unsigned int data)
{
  const unsigned int hash = libxsmm_hash16(data);
  return libxsmm_crc32_u8(hash >> 8, &hash) & 0xFF;
}


LIBXSMM_API unsigned int libxsmm_hash16(unsigned int data)
{
  return libxsmm_crc32_u16(data >> 16, &data) & 0xFFFF;
}


LIBXSMM_API unsigned int libxsmm_hash32(unsigned long long data)
{
  return libxsmm_crc32_u32(data >> 32, &data) & 0xFFFFFFFF;
}


LIBXSMM_API unsigned long long libxsmm_hash_string(const char string[])
{
  unsigned long long result;
  const size_t length = (NULL != string ? strlen(string) : 0);
  if (sizeof(result) < length) {
    const size_t length2 = LIBXSMM_MAX(length / 2, sizeof(result));
    unsigned int hash32, seed32 = 0; /* seed=0: match else-optimization */
    LIBXSMM_INIT
    seed32 = libxsmm_crc32(seed32, string, length2);
    hash32 = libxsmm_crc32(seed32, string + length2, length - length2);
    result = hash32; result = (result << 32) | seed32;
  }
  else if (sizeof(result) != length) {
    char *const s = (char*)&result; signed char i;
    for (i = 0; i < (signed char)length; ++i) s[i] = string[i];
    for (; i < (signed char)sizeof(result); ++i) s[i] = 0;
  }
  else { /* reinterpret directly as hash value */
    LIBXSMM_ASSERT(NULL != string);
    result = *(const unsigned long long*)string;
  }
  return result;
}


LIBXSMM_API const char* libxsmm_stristrn(const char a[], const char b[], size_t maxlen)
{
  const char* result = NULL;
  if (NULL != a && NULL != b && '\0' != *a && '\0' != *b && 0 != maxlen) {
    do {
      if (tolower(*a) != tolower(*b)) {
        ++a;
      }
      else {
        const char* c = b;
        size_t i = 0;
        result = a;
        while ('\0' != *++a && '\0' != c[++i] && i != maxlen) {
          if (tolower(*a) != tolower(c[i])) {
            result = NULL;
            break;
          }
        }
        if ('\0' != c[i] && '\0' != c[i+1] && c[i] != c[i+1] && i != maxlen) {
          result = NULL;
        }
        else break;
      }
    } while ('\0' != *a);
  }
  return result;
}


LIBXSMM_API const char* libxsmm_stristr(const char a[], const char b[])
{
  return libxsmm_stristrn(a, b, (size_t)-1);
}


LIBXSMM_API_INLINE int internal_isbreak(char c, const char delims[])
{
  char s[2] = { '\0' }; s[0] = c;
  return NULL != strpbrk(s, delims);
}


LIBXSMM_API int libxsmm_strimatch(const char a[], const char b[], const char delims[])
{
  int result = 0;
  if (NULL != a && NULL != b && '\0' != *a && '\0' != *b) {
    const char *const sep = ((NULL == delims || '\0' == *delims) ? " \t;,:-" : delims);
    const char *c, *tmp;
    int nwords = 0;
    size_t m, n;
    do {
      while (internal_isbreak(*b, sep)) ++b; /* left-trim */
      tmp = b;
      while ('\0' != *tmp && !internal_isbreak(*tmp, sep)) ++tmp;
      m = tmp - b;
      c = libxsmm_stristrn(a, b, LIBXSMM_MIN(1, m));
      if (NULL != c) {
        const char *d = c;
        while ('\0' != *d && !internal_isbreak(*d, sep)) ++d;
        n = d - c;
        if (1 >= n || NULL != libxsmm_stristrn(c, b, LIBXSMM_MIN(m, n))) ++result;
      }
      b = tmp;
    } while ('\0' != *b);
    do { /* count number of words */
      while (internal_isbreak(*a, sep)) ++a; /* left-trim */
      if ('\0' != *a) ++nwords;
      while ('\0' != *a && !internal_isbreak(*a, sep)) ++a;
    } while ('\0' != *a);
    if (nwords < result) result = nwords;
  } else result = -1;
  return result;
}


LIBXSMM_API int libxsmm_shuffle(void* inout, size_t elemsize, size_t count,
  const size_t* shuffle, const size_t* nrepeat)
{
  int result;
  if (NULL != inout || 0 == elemsize || 0 == count) {
    const size_t s = (NULL == shuffle ? LIBXSMM_MEMORY_SHUFFLE_COPRIME(count) : *shuffle);
    const size_t n = (NULL == nrepeat ? 1 : *nrepeat);
    switch (elemsize) {
      case 8:   LIBXSMM_MEMORY_SHUFFLE(inout, 8, count, s, n); break;
      case 4:   LIBXSMM_MEMORY_SHUFFLE(inout, 4, count, s, n); break;
      case 2:   LIBXSMM_MEMORY_SHUFFLE(inout, 2, count, s, n); break;
      case 1:   LIBXSMM_MEMORY_SHUFFLE(inout, 1, count, s, n); break;
      default:  LIBXSMM_MEMORY_SHUFFLE(inout, elemsize, count, s, n);
    }
    result = EXIT_SUCCESS;
  }
  else result = EXIT_FAILURE;
  return result;
}


LIBXSMM_API int libxsmm_shuffle2(void* dst, const void* src, size_t elemsize, size_t count,
  const size_t* shuffle, const size_t* nrepeat)
{
  const unsigned char *const LIBXSMM_RESTRICT inp = (const unsigned char*)src;
  unsigned char *const LIBXSMM_RESTRICT out = (unsigned char*)dst;
  const size_t size = elemsize * count;
  int result;
  if ((NULL != inp && NULL != out && ((out + size) <= inp || (inp + size) <= out)) || 0 == size) {
    const size_t s = (NULL == shuffle ? LIBXSMM_MEMORY_SHUFFLE_COPRIME(count) : *shuffle);
    size_t i = 0, j = 1;
    if (NULL == nrepeat || 1 == *nrepeat) {
      if (elemsize < 128) {
        switch (elemsize) {
          case 8: for (; i < size; i += 8, j += s) {
            if (count < j) j -= count;
            *(unsigned long long*)(out + i) = *(const unsigned long long*)(inp + size - 8 * j);
          } break;
          case 4: for (; i < size; i += 4, j += s) {
            if (count < j) j -= count;
            *(unsigned int*)(out + i) = *(const unsigned int*)(inp + size - 4 * j);
          } break;
          case 2: for (; i < size; i += 2, j += s) {
            if (count < j) j -= count;
            *(unsigned short*)(out + i) = *(const unsigned short*)(inp + size - 2 * j);
          } break;
          case 1: for (; i < size; ++i, j += s) {
            if (count < j) j -= count;
            out[i] = inp[size-j];
          } break;
          default: for (; i < size; i += elemsize, j += s) {
            if (count < j) j -= count;
            LIBXSMM_MEMCPY127(out + i, inp + size - elemsize * j, elemsize);
          }
        }
      }
      else { /* generic path */
        for (; i < size; i += elemsize, j += s) {
          if (count < j) j -= count;
          memcpy(out + i, inp + size - elemsize * j, elemsize);
        }
      }
    }
    else if (0 != *nrepeat) { /* generic path */
      const size_t c = count - 1;
      for (; i < count; ++i) {
        size_t k = 0;
        LIBXSMM_ASSERT(NULL != inp && NULL != out);
        for (j = i; k < *nrepeat; ++k) j = c - ((s * j) % count);
        memcpy(out + elemsize * i, inp + elemsize * j, elemsize);
      }
    }
    else { /* ordinary copy */
      memcpy(out, inp, size);
    }
    result = EXIT_SUCCESS;
  }
  else result = EXIT_FAILURE;
  return result;
}


LIBXSMM_API size_t libxsmm_unshuffle(size_t count, const size_t* shuffle)
{
  size_t result = 0;
  if (0 < count) {
    const size_t n = (NULL == shuffle ? LIBXSMM_MEMORY_SHUFFLE_COPRIME(count) : *shuffle);
    size_t c = count - 1, j = c, d = 0;
    for (; result < count; ++result, j = c - d) {
      d = (j * n) % count;
      if (0 == d) break;
    }
  }
  assert(result <= count);
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
  else if (1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED)) {
    LIBXSMM_INIT
    if (0 != libxsmm_verbosity) { /* library code is expected to be mute */
      fprintf(stderr, "LIBXSMM ERROR: invalid arguments for libxsmm_xhash specified!\n");
    }
  }
#endif
}


/* implementation provided for Fortran 77 compatibility */
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_xdiff)(int* /*result*/, const void* /*a*/, const void* /*b*/, const long long* /*size*/);
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_xdiff)(int* result, const void* a, const void* b, const long long* size)
{
#if !defined(NDEBUG)
  static int error_once = 0;
  if (NULL != result && NULL != a && NULL != b && NULL != size && 0 <= *size)
#endif
  {
    *result = libxsmm_memcmp(a, b, (size_t)*size);
  }
#if !defined(NDEBUG)
  else if (1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED)) {
    LIBXSMM_INIT
    if (0 != libxsmm_verbosity) { /* library code is expected to be mute */
      fprintf(stderr, "LIBXSMM ERROR: invalid arguments for libxsmm_xdiff specified!\n");
    }
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
  { const int s = *size;
    LIBXSMM_MEMSET127(dst, 0, s);
  }
#if !defined(NDEBUG)
  else if (1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED)) {
    LIBXSMM_INIT
    if (0 != libxsmm_verbosity) { /* library code is expected to be mute */
      fprintf(stderr, "LIBXSMM ERROR: invalid arguments for libxsmm_xclear specified!\n");
    }
  }
#endif
}


LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_aligned)(int* /*result*/, const void* /*ptr*/, const int* /*inc*/, int* /*alignment*/);
LIBXSMM_API void LIBXSMM_FSYMBOL(libxsmm_aligned)(int* result, const void* ptr, const int* inc, int* alignment)
{
#if !defined(NDEBUG)
  static int error_once = 0;
  if (NULL != result)
#endif
  {
    const size_t next = (NULL != inc ? *inc : 0);
    *result = libxsmm_aligned(ptr, &next, alignment);
  }
#if !defined(NDEBUG)
  else if (1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED)) {
    LIBXSMM_INIT
    if (0 != libxsmm_verbosity) { /* library code is expected to be mute */
      fprintf(stderr, "LIBXSMM ERROR: invalid arguments for libxsmm_aligned specified!\n");
    }
  }
#endif
}

#endif /*defined(LIBXSMM_BUILD) && (!defined(LIBXSMM_NOFORTRAN) || defined(__clang_analyzer__))*/
