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
#ifndef LIBXSMM_DIFF_H
#define LIBXSMM_DIFF_H

#include <libxsmm_intrinsics_x86.h>

#if (LIBXSMM_X86_SSE3 <= LIBXSMM_STATIC_TARGET_ARCH)
# define LIBXSMM_DIFF_16_DECL(A) __m128i A
# define LIBXSMM_DIFF_16_LOAD(A, SRC) A = _mm_loadu_si128((const __m128i*)(SRC))
# define LIBXSMM_DIFF_16(A, B, ...) ((unsigned char)(0xFFFF != _mm_movemask_epi8(_mm_cmpeq_epi8( \
    A, _mm_loadu_si128((const __m128i*)(B))))))
#else
# define LIBXSMM_DIFF_16_DECL(A) const uint64_t */*const*/ A
# define LIBXSMM_DIFF_16_LOAD(A, SRC) A = (const uint64_t*)(SRC)
# define LIBXSMM_DIFF_16(A, B, ...) ((unsigned char)(0 != (((A)[0] ^ (*(const uint64_t*)(B))) | \
    ((A)[1] ^ ((const uint64_t*)(B))[1]))))
#endif
#if (LIBXSMM_X86_AVX2 <= LIBXSMM_STATIC_TARGET_ARCH)
# define LIBXSMM_DIFF_32_DECL(A) __m256i A
# define LIBXSMM_DIFF_32_LOAD(A, SRC) A = _mm256_loadu_si256((const __m256i*)(SRC))
# define LIBXSMM_DIFF_32(A, B, ...) ((unsigned char)(-1 != _mm256_movemask_epi8(_mm256_cmpeq_epi8( \
    A, _mm256_loadu_si256((const __m256i*)(B))))))
#else
# define LIBXSMM_DIFF_32_DECL(A) LIBXSMM_DIFF_16_DECL(A); LIBXSMM_DIFF_16_DECL(LIBXSMM_CONCATENATE2(libxsmm_diff_32_, A, _))
# define LIBXSMM_DIFF_32_LOAD(A, SRC) LIBXSMM_DIFF_16_LOAD(A, SRC); LIBXSMM_DIFF_16_LOAD(LIBXSMM_CONCATENATE2(libxsmm_diff_32_, A, _), (const uint64_t*)(SRC) + 2)
# define LIBXSMM_DIFF_32(A, B, ...) ((unsigned char)(0 != LIBXSMM_DIFF_16(A, B, __VA_ARGS__) ? 1 : LIBXSMM_DIFF_16(LIBXSMM_CONCATENATE2(libxsmm_diff_32_, A, _), (const uint64_t*)(B) + 2, __VA_ARGS__)))
#endif

#define LIBXSMM_DIFF_48_DECL(A) LIBXSMM_DIFF_16_DECL(A); LIBXSMM_DIFF_32_DECL(LIBXSMM_CONCATENATE2(libxsmm_diff_48_, A, _))
#define LIBXSMM_DIFF_48_LOAD(A, SRC) LIBXSMM_DIFF_16_LOAD(A, SRC); LIBXSMM_DIFF_32_LOAD(LIBXSMM_CONCATENATE2(libxsmm_diff_48_, A, _), (const uint64_t*)(SRC) + 2)
#define LIBXSMM_DIFF_48(A, B, ...) ((unsigned char)(0 != LIBXSMM_DIFF_16(A, B, __VA_ARGS__) ? 1 : LIBXSMM_DIFF_32(LIBXSMM_CONCATENATE2(libxsmm_diff_48_, A, _), (const uint64_t*)(B) + 2, __VA_ARGS__)))

#define LIBXSMM_DIFF_64_DECL(A) LIBXSMM_DIFF_32_DECL(A); LIBXSMM_DIFF_32_DECL(LIBXSMM_CONCATENATE2(libxsmm_diff_64_, A, _))
#define LIBXSMM_DIFF_64_LOAD(A, SRC) LIBXSMM_DIFF_32_LOAD(A, SRC); LIBXSMM_DIFF_32_LOAD(LIBXSMM_CONCATENATE2(libxsmm_diff_64_, A, _), (const uint64_t*)(SRC) + 4)
#define LIBXSMM_DIFF_64(A, B, ...) ((unsigned char)(0 != LIBXSMM_DIFF_32(A, B, __VA_ARGS__) ? 1 : LIBXSMM_DIFF_32(LIBXSMM_CONCATENATE2(libxsmm_diff_64_, A, _), (const uint64_t*)(B) + 4, __VA_ARGS__)))

#define LIBXSMM_DIFF_DECL(N, A) LIBXSMM_CONCATENATE2(LIBXSMM_DIFF_, N, _DECL)(A)
#define LIBXSMM_DIFF_LOAD(N, A, SRC) LIBXSMM_CONCATENATE2(LIBXSMM_DIFF_, N, _LOAD)(A, SRC)
#define LIBXSMM_DIFF(N) LIBXSMM_CONCATENATE(LIBXSMM_DIFF_, N)

#define LIBXSMM_DIFF_N(TYPE, RESULT, DIFF, A, BN, ELEMSIZE, STRIDE, HINT, N) { \
  const char* libxsmm_diff_b_ = (const char*)(BN); \
  if (0 == (HINT)) { /* fast-path */ \
    for (RESULT = 0; (RESULT) < (N); ++(RESULT)) { \
      if (0 == DIFF(A, libxsmm_diff_b_, ELEMSIZE)) break; \
      libxsmm_diff_b_ += (STRIDE); \
    } \
  } \
  else { /* wrap around index */ \
    TYPE libxsmm_diff_i_ = (HINT); \
    const size_t libxsmm_diff_end_ = (size_t)(N) * (STRIDE); \
    libxsmm_diff_b_ += (size_t)(HINT) * (STRIDE); \
    RESULT = (N); \
    for (; libxsmm_diff_i_ < ((N) + (HINT)); ++libxsmm_diff_i_) { \
      const char *const libxsmm_diff_c_ = (libxsmm_diff_i_ < (N) ? libxsmm_diff_b_ : (libxsmm_diff_b_ - libxsmm_diff_end_)); \
      if (0 == DIFF(A, libxsmm_diff_c_, ELEMSIZE)) { \
        RESULT = (libxsmm_diff_i_ < (N) ? libxsmm_diff_i_ : (libxsmm_diff_i_ - (N))); \
        break; \
      } \
      libxsmm_diff_b_ += (STRIDE); \
    } \
  } \
}


/** Function type representing the diff-functionality. */
LIBXSMM_EXTERN_C typedef LIBXSMM_RETARGETABLE unsigned int (*libxsmm_diff_function)(
  const void* /*a*/, const void* /*b*/, ... /*size*/);

/** Compare two data blocks of 16 Byte each. */
LIBXSMM_API unsigned char libxsmm_diff_16(const void* a, const void* b, ...);
/** Compare two data blocks of 32 Byte each. */
LIBXSMM_API unsigned char libxsmm_diff_32(const void* a, const void* b, ...);
/** Compare two data blocks of 48 Byte each. */
LIBXSMM_API unsigned char libxsmm_diff_48(const void* a, const void* b, ...);
/** Compare two data blocks of 64 Byte each. */
LIBXSMM_API unsigned char libxsmm_diff_64(const void* a, const void* b, ...);

#endif /*LIBXSMM_DIFF_H*/

