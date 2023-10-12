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
#ifndef LIBXSMM_DIFF_H
#define LIBXSMM_DIFF_H

#include <libxsmm_intrinsics_x86.h>

#if !defined(LIBXSMM_DIFF_AVX512_ENABLED) && 0
# define LIBXSMM_DIFF_AVX512_ENABLED
#endif

#define LIBXSMM_DIFF_4_DECL(A) const uint32_t */*const*/ A = NULL
#define LIBXSMM_DIFF_4_ASSIGN(A, B) (A) = (B)
#define LIBXSMM_DIFF_4_LOAD(A, SRC) A = (const uint32_t*)(SRC)
#define LIBXSMM_DIFF_4(A, B, ...) ((unsigned char)(0 != (*(A) ^ (*(const uint32_t*)(B)))))

#define LIBXSMM_DIFF_8_DECL(A) const uint64_t */*const*/ A = NULL
#define LIBXSMM_DIFF_8_ASSIGN(A, B) (A) = (B)
#define LIBXSMM_DIFF_8_LOAD(A, SRC) A = (const uint64_t*)(SRC)
#define LIBXSMM_DIFF_8(A, B, ...) ((unsigned char)(0 != (*(A) ^ (*(const uint64_t*)(B)))))

#define LIBXSMM_DIFF_SSE_DECL(A) __m128i A = LIBXSMM_INTRINSICS_MM_UNDEFINED_SI128()
#define LIBXSMM_DIFF_SSE_ASSIGN(A, B) (A) = (B)
#define LIBXSMM_DIFF_SSE_LOAD(A, SRC) A = LIBXSMM_INTRINSICS_LOADU_SI128((const __m128i*)(SRC))
#define LIBXSMM_DIFF_SSE(A, B, ...) ((unsigned char)(0xFFFF != _mm_movemask_epi8(_mm_cmpeq_epi8( \
  A, LIBXSMM_INTRINSICS_LOADU_SI128((const __m128i*)(B))))))

#if (LIBXSMM_X86_GENERIC <= LIBXSMM_STATIC_TARGET_ARCH) /*|| defined(LIBXSMM_INTRINSICS_TARGET)*/
# define LIBXSMM_DIFF_16_DECL LIBXSMM_DIFF_SSE_DECL
# define LIBXSMM_DIFF_16_ASSIGN LIBXSMM_DIFF_SSE_ASSIGN
# define LIBXSMM_DIFF_16_LOAD LIBXSMM_DIFF_SSE_LOAD
# define LIBXSMM_DIFF_16 LIBXSMM_DIFF_SSE
#else
# define LIBXSMM_DIFF_16_DECL(A) const uint64_t */*const*/ A = NULL
# define LIBXSMM_DIFF_16_ASSIGN(A, B) (A) = (B)
# define LIBXSMM_DIFF_16_LOAD(A, SRC) A = (const uint64_t*)(SRC)
# define LIBXSMM_DIFF_16(A, B, ...) ((unsigned char)(0 != (((A)[0] ^ (*(const uint64_t*)(B))) | \
    ((A)[1] ^ ((const uint64_t*)(B))[1]))))
#endif

#define LIBXSMM_DIFF_AVX2_DECL(A) __m256i A = LIBXSMM_INTRINSICS_MM256_UNDEFINED_SI256()
#define LIBXSMM_DIFF_AVX2_ASSIGN(A, B) (A) = (B)
#define LIBXSMM_DIFF_AVX2_LOAD(A, SRC) A = _mm256_loadu_si256((const __m256i*)(SRC))
#define LIBXSMM_DIFF_AVX2(A, B, ...) ((unsigned char)(-1 != _mm256_movemask_epi8(_mm256_cmpeq_epi8( \
  A, _mm256_loadu_si256((const __m256i*)(B))))))

#if (LIBXSMM_X86_AVX2 <= LIBXSMM_STATIC_TARGET_ARCH)
# define LIBXSMM_DIFF_32_DECL LIBXSMM_DIFF_AVX2_DECL
# define LIBXSMM_DIFF_32_ASSIGN LIBXSMM_DIFF_AVX2_ASSIGN
# define LIBXSMM_DIFF_32_LOAD LIBXSMM_DIFF_AVX2_LOAD
# define LIBXSMM_DIFF_32 LIBXSMM_DIFF_AVX2
#else
# define LIBXSMM_DIFF_32_DECL(A) LIBXSMM_DIFF_16_DECL(A); LIBXSMM_DIFF_16_DECL(LIBXSMM_CONCATENATE3(libxsmm_diff_32_, A, _))
# define LIBXSMM_DIFF_32_ASSIGN(A, B) LIBXSMM_DIFF_16_ASSIGN(A, B); LIBXSMM_DIFF_16_ASSIGN(LIBXSMM_CONCATENATE3(libxsmm_diff_32_, A, _), LIBXSMM_CONCATENATE3(libxsmm_diff_32_, B, _))
# define LIBXSMM_DIFF_32_LOAD(A, SRC) LIBXSMM_DIFF_16_LOAD(A, SRC); LIBXSMM_DIFF_16_LOAD(LIBXSMM_CONCATENATE3(libxsmm_diff_32_, A, _), (const uint64_t*)(SRC) + 2)
# define LIBXSMM_DIFF_32(A, B, ...) ((unsigned char)(0 != LIBXSMM_DIFF_16(A, B, __VA_ARGS__) ? 1 : LIBXSMM_DIFF_16(LIBXSMM_CONCATENATE3(libxsmm_diff_32_, A, _), (const uint64_t*)(B) + 2, __VA_ARGS__)))
#endif

#define LIBXSMM_DIFF_48_DECL(A) LIBXSMM_DIFF_16_DECL(A); LIBXSMM_DIFF_32_DECL(LIBXSMM_CONCATENATE3(libxsmm_diff_48_, A, _))
#define LIBXSMM_DIFF_48_ASSIGN(A, B) LIBXSMM_DIFF_16_ASSIGN(A, B); LIBXSMM_DIFF_32_ASSIGN(LIBXSMM_CONCATENATE3(libxsmm_diff_48_, A, _), LIBXSMM_CONCATENATE3(libxsmm_diff_48_, B, _))
#define LIBXSMM_DIFF_48_LOAD(A, SRC) LIBXSMM_DIFF_16_LOAD(A, SRC); LIBXSMM_DIFF_32_LOAD(LIBXSMM_CONCATENATE3(libxsmm_diff_48_, A, _), (const uint64_t*)(SRC) + 2)
#define LIBXSMM_DIFF_48(A, B, ...) ((unsigned char)(0 != LIBXSMM_DIFF_16(A, B, __VA_ARGS__) ? 1 : LIBXSMM_DIFF_32(LIBXSMM_CONCATENATE3(libxsmm_diff_48_, A, _), (const uint64_t*)(B) + 2, __VA_ARGS__)))

#define LIBXSMM_DIFF_64SW_DECL(A) LIBXSMM_DIFF_32_DECL(A); LIBXSMM_DIFF_32_DECL(LIBXSMM_CONCATENATE3(libxsmm_diff_64_, A, _))
#define LIBXSMM_DIFF_64SW_ASSIGN(A, B) LIBXSMM_DIFF_32_ASSIGN(A, B); LIBXSMM_DIFF_32_ASSIGN(LIBXSMM_CONCATENATE3(libxsmm_diff_64_, A, _), LIBXSMM_CONCATENATE3(libxsmm_diff_64_, B, _))
#define LIBXSMM_DIFF_64SW_LOAD(A, SRC) LIBXSMM_DIFF_32_LOAD(A, SRC); LIBXSMM_DIFF_32_LOAD(LIBXSMM_CONCATENATE3(libxsmm_diff_64_, A, _), (const uint64_t*)(SRC) + 4)
#define LIBXSMM_DIFF_64SW(A, B, ...) ((unsigned char)(0 != LIBXSMM_DIFF_32(A, B, __VA_ARGS__) ? 1 : LIBXSMM_DIFF_32(LIBXSMM_CONCATENATE3(libxsmm_diff_64_, A, _), (const uint64_t*)(B) + 4, __VA_ARGS__)))

#if defined(LIBXSMM_DIFF_AVX512_ENABLED)
# define LIBXSMM_DIFF_AVX512_DECL(A) __m512i A = LIBXSMM_INTRINSICS_MM512_UNDEFINED_EPI32()
# define LIBXSMM_DIFF_AVX512_ASSIGN(A, B) (A) = (B)
# define LIBXSMM_DIFF_AVX512_LOAD(A, SRC) A = _mm512_loadu_si512((const __m512i*)(SRC))
# define LIBXSMM_DIFF_AVX512(A, B, ...) ((unsigned char)(0xFFFF != (unsigned int)/*_cvtmask16_u32*/(_mm512_cmpeq_epi32_mask( \
    A, _mm512_loadu_si512((const __m512i*)(B))))))
#else
# define LIBXSMM_DIFF_AVX512_DECL LIBXSMM_DIFF_64SW_DECL
# define LIBXSMM_DIFF_AVX512_ASSIGN LIBXSMM_DIFF_64SW_ASSIGN
# define LIBXSMM_DIFF_AVX512_LOAD LIBXSMM_DIFF_64SW_LOAD
# define LIBXSMM_DIFF_AVX512 LIBXSMM_DIFF_64SW
#endif

#if (LIBXSMM_X86_AVX512_SKX <= LIBXSMM_STATIC_TARGET_ARCH)
# define LIBXSMM_DIFF_64_DECL LIBXSMM_DIFF_AVX512_DECL
# define LIBXSMM_DIFF_64_ASSIGN LIBXSMM_DIFF_AVX512_ASSIGN
# define LIBXSMM_DIFF_64_LOAD LIBXSMM_DIFF_AVX512_LOAD
# define LIBXSMM_DIFF_64 LIBXSMM_DIFF_AVX512
#else
# define LIBXSMM_DIFF_64_DECL LIBXSMM_DIFF_64SW_DECL
# define LIBXSMM_DIFF_64_ASSIGN LIBXSMM_DIFF_64SW_ASSIGN
# define LIBXSMM_DIFF_64_LOAD LIBXSMM_DIFF_64SW_LOAD
# define LIBXSMM_DIFF_64 LIBXSMM_DIFF_64SW
#endif

#define LIBXSMM_DIFF_DECL(N, A) LIBXSMM_CONCATENATE3(LIBXSMM_DIFF_, N, _DECL)(A)
#define LIBXSMM_DIFF_LOAD(N, A, SRC) LIBXSMM_CONCATENATE3(LIBXSMM_DIFF_, N, _LOAD)(A, SRC)
#define LIBXSMM_DIFF(N) LIBXSMM_CONCATENATE(LIBXSMM_DIFF_, N)

#define LIBXSMM_DIFF_N(TYPE, RESULT, DIFF, A, BN, ELEMSIZE, STRIDE, HINT, N) do { \
  const char* libxsmm_diff_b_ = (const char*)(BN) + (size_t)(HINT) * (STRIDE); \
  for (RESULT = (HINT); (RESULT) < (N); ++(RESULT)) { \
    if (0 == DIFF(A, libxsmm_diff_b_, ELEMSIZE)) break; \
    libxsmm_diff_b_ += (STRIDE); \
  } \
  if ((N) == (RESULT)) { /* wrong hint */ \
    TYPE libxsmm_diff_r_ = 0; \
    libxsmm_diff_b_ = (const char*)(BN); /* reset */ \
    for (; libxsmm_diff_r_ < (HINT); ++libxsmm_diff_r_) { \
      if (0 == DIFF(A, libxsmm_diff_b_, ELEMSIZE)) { \
        RESULT = libxsmm_diff_r_; \
        break; \
      } \
      libxsmm_diff_b_ += (STRIDE); \
    } \
  } \
} while(0)


/** Function type representing the diff-functionality. */
LIBXSMM_EXTERN_C typedef unsigned int (*libxsmm_diff_function)(
  const void* /*a*/, const void* /*b*/, ... /*size*/);

/** Compare two data blocks of 4 Byte each. */
LIBXSMM_API unsigned char libxsmm_diff_4(const void* a, const void* b, ...);
/** Compare two data blocks of 8 Byte each. */
LIBXSMM_API unsigned char libxsmm_diff_8(const void* a, const void* b, ...);
/** Compare two data blocks of 16 Byte each. */
LIBXSMM_API unsigned char libxsmm_diff_16(const void* a, const void* b, ...);
/** Compare two data blocks of 32 Byte each. */
LIBXSMM_API unsigned char libxsmm_diff_32(const void* a, const void* b, ...);
/** Compare two data blocks of 48 Byte each. */
LIBXSMM_API unsigned char libxsmm_diff_48(const void* a, const void* b, ...);
/** Compare two data blocks of 64 Byte each. */
LIBXSMM_API unsigned char libxsmm_diff_64(const void* a, const void* b, ...);

#endif /*LIBXSMM_DIFF_H*/
