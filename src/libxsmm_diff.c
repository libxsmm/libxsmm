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
#include <libxsmm_math.h>
#include "libxsmm_diff.h"
#include "libxsmm_main.h"

#if !defined(LIBXSMM_DIFF_NAVG)
# define LIBXSMM_DIFF_NAVG LIBXSMM_CAPACITY_CACHE
#endif

#if (LIBXSMM_X86_SSE3 <= LIBXSMM_STATIC_TARGET_ARCH)
# define LIBXSMM_DIFF_16(RESULT, A, B) { \
    const __m128i libxsmm_math_diff_16_a128_ = _mm_loadu_si128((const __m128i*)(A)); \
    const __m128i libxsmm_math_diff_16_b128_ = _mm_loadu_si128((const __m128i*)(B)); \
    RESULT = (unsigned char)(0xFFFF != _mm_movemask_epi8(_mm_cmpeq_epi8( \
      libxsmm_math_diff_16_a128_, libxsmm_math_diff_16_b128_))); \
  }
#else
# define LIBXSMM_DIFF_16(RESULT, A, B) { \
    const uint64_t *const libxsmm_math_diff_16_a64_ = (const uint64_t*)(A); \
    const uint64_t *const libxsmm_math_diff_16_b64_ = (const uint64_t*)(B); \
    RESULT = (unsigned char)(0 != ((libxsmm_math_diff_16_a64_[0] ^ libxsmm_math_diff_16_b64_[0]) | \
      (libxsmm_math_diff_16_a64_[1] ^ libxsmm_math_diff_16_b64_[1]))); \
  }
#endif
#if (LIBXSMM_X86_AVX2 <= LIBXSMM_STATIC_TARGET_ARCH)
# define LIBXSMM_DIFF_32(RESULT, A, B) { \
    const __m256i libxsmm_math_diff_32_a256_ = _mm256_loadu_si256((const __m256i*)(A)); \
    const __m256i libxsmm_math_diff_32_b256_ = _mm256_loadu_si256((const __m256i*)(B)); \
    RESULT = (unsigned char)(-1 != _mm256_movemask_epi8(_mm256_cmpeq_epi8( \
      libxsmm_math_diff_32_a256_, libxsmm_math_diff_32_b256_))); \
  }
#else
# define LIBXSMM_DIFF_32(RESULT, A, B) { \
    unsigned char libxsmm_math_diff_32_r1_, libxsmm_math_diff_32_r2_; \
    LIBXSMM_DIFF_16(libxsmm_math_diff_32_r1_, A, B); \
    LIBXSMM_DIFF_16(libxsmm_math_diff_32_r2_, \
      (const uint8_t*)(A) + 16, (const uint8_t*)(B) + 16); \
    RESULT = (unsigned char)(libxsmm_math_diff_32_r1_ | libxsmm_math_diff_32_r2_); \
  }
#endif

#define LIBXSMM_MATH_DIFF(DIFF, MOD, A, BN, ELEMSIZE, STRIDE, HINT, N, NAVG) { \
  const char *const libxsmm_diff_b_ = (const char*)(BN); \
  unsigned int libxsmm_diff_i_ = HINT; \
  if (0 == (HINT)) { /* fast-path */ \
    unsigned int libxsmm_diff_j_ = 0; \
    LIBXSMM_PRAGMA_LOOP_COUNT(4, 1024, NAVG) \
    for (; libxsmm_diff_i_ < (N); ++libxsmm_diff_i_) { \
      if (0 == (DIFF)(A, libxsmm_diff_b_ + libxsmm_diff_j_, ELEMSIZE)) return libxsmm_diff_i_; \
      libxsmm_diff_j_ += STRIDE; \
    } \
  } \
  else { /* wrap around index */ \
    LIBXSMM_ASSERT(0 != (HINT)); \
    LIBXSMM_PRAGMA_LOOP_COUNT(4, 1024, NAVG) \
    for (; libxsmm_diff_i_ < ((HINT) + (N)); ++libxsmm_diff_i_) { \
      const unsigned int libxsmm_diff_j_ = MOD(libxsmm_diff_i_, N); \
      const unsigned int libxsmm_diff_k_ = libxsmm_diff_j_ * (STRIDE); \
      if (0 == (DIFF)(A, libxsmm_diff_b_ + libxsmm_diff_k_, ELEMSIZE)) return libxsmm_diff_j_; \
    } \
  } \
  return N; \
}


LIBXSMM_API unsigned char libxsmm_diff_16(const void* a, const void* b, ...)
{
  unsigned char result;
  LIBXSMM_DIFF_16(result, a, b);
  return result;
}


LIBXSMM_API unsigned char libxsmm_diff_32(const void* a, const void* b, ...)
{
  unsigned char result;
  LIBXSMM_DIFF_32(result, a, b);
  return result;
}


LIBXSMM_API unsigned char libxsmm_diff_48(const void* a, const void* b, ...)
{
  unsigned char r1, r2;
  LIBXSMM_DIFF_32(r1, a, b);
  LIBXSMM_DIFF_16(r2, (const uint8_t*)a + 32, (const uint8_t*)b + 32);
  return (unsigned char)(r1 | r2);
}


LIBXSMM_API unsigned char libxsmm_diff_64(const void* a, const void* b, ...)
{
  unsigned char r1, r2;
  LIBXSMM_DIFF_32(r1, a, b);
  LIBXSMM_DIFF_32(r2, (const uint8_t*)a + 32, (const uint8_t*)b + 32);
  return (unsigned char)(r1 | r2);
}


LIBXSMM_API unsigned char libxsmm_diff(const void* a, const void* b, unsigned char size)
{
  const uint8_t *const a8 = (const uint8_t*)a, *const b8 = (const uint8_t*)b;
  unsigned char i;
  for (i = 0; i < (size & 0xF0); i += 16) {
    unsigned char r;
    LIBXSMM_DIFF_16(r, a8 + i, b8 + i);
    if (r) return 1;
  }
  for (; i < size; ++i) if (a8[i] ^ b8[i]) return 1;
  return 0;
}


LIBXSMM_API unsigned int libxsmm_diff_n(const void* a, const void* bn, unsigned char size,
  unsigned char stride, unsigned int hint, unsigned int n)
{
  LIBXSMM_ASSERT(size <= stride);
  switch (size) {
    case 64: {
      LIBXSMM_MATH_DIFF(libxsmm_diff_64, LIBXSMM_MOD, a, bn, size, stride, hint, n, LIBXSMM_DIFF_NAVG);
    } break;
    case 32: {
      LIBXSMM_MATH_DIFF(libxsmm_diff_32, LIBXSMM_MOD, a, bn, size, stride, hint, n, LIBXSMM_DIFF_NAVG);
    } break;
    case 16: {
      LIBXSMM_MATH_DIFF(libxsmm_diff_16, LIBXSMM_MOD, a, bn, size, stride, hint, n, LIBXSMM_DIFF_NAVG);
    } break;
    default: {
      LIBXSMM_MATH_DIFF(libxsmm_diff, LIBXSMM_MOD, a, bn, size, stride, hint, n, LIBXSMM_DIFF_NAVG);
    }
  }
}


LIBXSMM_API unsigned int libxsmm_diff_npot(const void* a, const void* bn, unsigned char size,
  unsigned char stride, unsigned int hint, unsigned int n)
{
#if !defined(NDEBUG)
  const unsigned int npot = LIBXSMM_UP2POT(n);
  assert(size <= stride && n == npot); /* !LIBXSMM_ASSERT */
#endif
  switch (size) {
    case 64: {
      LIBXSMM_MATH_DIFF(libxsmm_diff_64, LIBXSMM_MOD2, a, bn, size, stride, hint, n, LIBXSMM_DIFF_NAVG);
    } break;
    case 32: {
      LIBXSMM_MATH_DIFF(libxsmm_diff_32, LIBXSMM_MOD2, a, bn, size, stride, hint, n, LIBXSMM_DIFF_NAVG);
    } break;
    case 16: {
      LIBXSMM_MATH_DIFF(libxsmm_diff_16, LIBXSMM_MOD2, a, bn, size, stride, hint, n, LIBXSMM_DIFF_NAVG);
    } break;
    default: {
      LIBXSMM_MATH_DIFF(libxsmm_diff, LIBXSMM_MOD2, a, bn, size, stride, hint, n, LIBXSMM_DIFF_NAVG);
    }
  }
}

