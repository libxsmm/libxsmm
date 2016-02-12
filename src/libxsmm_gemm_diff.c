/******************************************************************************
** Copyright (c) 2016, Intel Corporation                                     **
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
#include "libxsmm_gemm_diff.h"

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#include <stdint.h>
#include <stdio.h>
#if !defined(NDEBUG)
# include <assert.h>
#endif
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif
/* must be the last included header */
#include "libxsmm_intrinsics.h"


LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE
unsigned int libxsmm_gemm_diff(const libxsmm_gemm_descriptor* a, const libxsmm_gemm_descriptor* b)
{
  const unsigned *const ia = (const unsigned int*)a, *const ib = (const unsigned int*)b;
  unsigned int result, i;
  assert(0 == LIBXSMM_MOD2(LIBXSMM_GEMM_DESCRIPTOR_SIZE, sizeof(unsigned int)));
  assert(0 != a && 0 != b);

  result = ia[0] ^ ib[0];
  for (i = 1; i < LIBXSMM_DIV2(LIBXSMM_GEMM_DESCRIPTOR_SIZE, sizeof(unsigned int)); ++i) {
    result |= (ia[i] ^ ib[i]);
  }

  return result;
}


LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE LIBXSMM_INTRINSICS
unsigned int libxsmm_gemm_diff_sse(const libxsmm_gemm_descriptor* a, const libxsmm_gemm_descriptor* b)
{
  return libxsmm_gemm_diff(a, b); /*TODO: SSE based implementation*/
}


LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE LIBXSMM_INTRINSICS
unsigned int libxsmm_gemm_diff_avx(const libxsmm_gemm_descriptor* a, const libxsmm_gemm_descriptor* b)
{
#if defined(LIBXSMM_AVX_MAX) && (1 <= (LIBXSMM_AVX_MAX)) && !(defined(__APPLE__) && defined(__MACH__) && \
  /* prevents fatal error (error in backend) apparently caused by _mm256_testnzc_ps */ \
  LIBXSMM_VERSION3(6, 1, 0) >= LIBXSMM_VERSION3(__clang_major__, __clang_minor__, __clang_patchlevel__))
  assert(0 == LIBXSMM_MOD2(LIBXSMM_GEMM_DESCRIPTOR_SIZE, sizeof(unsigned int)));
  assert(8 >= LIBXSMM_DIV2(LIBXSMM_GEMM_DESCRIPTOR_SIZE, 4));
  assert(0 != a && 0 != b);
  {
# if (28 == LIBXSMM_GEMM_DESCRIPTOR_SIZE) /* otherwise generate a compile-time error */
    int r0, r1;
    union { __m256 s; __m256i i; } a256, b256;
#   if defined(__CYGWIN__) && !defined(NDEBUG) /* Cygwin/GCC: _mm256_set_epi32 may cause an illegal instruction */
    const union { int32_t array[8]; __m256i m256i; } mask = { /* use literal value rather than yes/no
      in order to avoid warning about "initializer element is not computable at load time" */
      { 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x0 }
    };
#   else
    const int yes = 0x80000000, no = 0x0;
    struct { __m256i m256i; } mask;
    mask.m256i = _mm256_set_epi32(no, yes, yes, yes, yes, yes, yes, yes);
#   endif
# endif
    a256.s = _mm256_maskload_ps((const float*)a, mask.m256i);
    b256.s = _mm256_maskload_ps((const float*)b, mask.m256i);
    r0 = _mm256_testnzc_si256(a256.i, b256.i);
    r1 = _mm256_testnzc_si256(b256.i, a256.i);
    return r0 | r1;
  }
#else
# if !defined(NDEBUG) /* library code is expected to be mute */
  static LIBXSMM_TLS int once = 0;
  if (0 == once) {
    fprintf(stderr, "LIBXSMM: unable to enter AVX instruction code path!\n");
    once = 1;
  }
# endif
# if !defined(__MIC__)
  LIBXSMM_MESSAGE("================================================================================");
  LIBXSMM_MESSAGE("LIBXSMM: Unable to enter the code path which is using AVX instructions!");
  LIBXSMM_MESSAGE("================================================================================");
# endif
  return libxsmm_gemm_diff(a, b);
#endif
}


LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE LIBXSMM_INTRINSICS
unsigned int libxsmm_gemm_diff_avx2(const libxsmm_gemm_descriptor* a, const libxsmm_gemm_descriptor* b)
{
#if defined(LIBXSMM_AVX_MAX) && (2 <= (LIBXSMM_AVX_MAX)) && !(defined(__APPLE__) && defined(__MACH__) && \
  /* prevents fatal error (error in backend) apparently caused by _mm256_testnzc_si256 */ \
  LIBXSMM_VERSION3(6, 1, 0) >= LIBXSMM_VERSION3(__clang_major__, __clang_minor__, __clang_patchlevel__))
  assert(0 == LIBXSMM_MOD2(LIBXSMM_GEMM_DESCRIPTOR_SIZE, sizeof(unsigned int)));
  assert(8 >= LIBXSMM_DIV2(LIBXSMM_GEMM_DESCRIPTOR_SIZE, 4));
  assert(0 != a && 0 != b);
  {
# if (28 == LIBXSMM_GEMM_DESCRIPTOR_SIZE) /* otherwise generate a compile-time error */
    const int yes = 0x80000000, no = 0x0;
    const __m256i mask = _mm256_set_epi32(no, yes, yes, yes, yes, yes, yes, yes);
    const __m256i a256 = _mm256_maskload_epi32((const void*)a, mask);
    const __m256i b256 = _mm256_maskload_epi32((const void*)b, mask);
    int r0, r1;
# endif
    r0 = _mm256_testnzc_si256(a256, b256);
    r1 = _mm256_testnzc_si256(b256, a256);
    return r0 | r1;
  }
#else
# if !defined(NDEBUG) /* library code is expected to be mute */
  static LIBXSMM_TLS int once = 0;
  if (0 == once) {
    fprintf(stderr, "LIBXSMM: unable to enter AVX2 instruction code path!\n");
    once = 1;
  }
# endif
# if !defined(__MIC__)
  LIBXSMM_MESSAGE("================================================================================");
  LIBXSMM_MESSAGE("LIBXSMM: Unable to enter the code path which is using AVX2 instructions!");
  LIBXSMM_MESSAGE("================================================================================");
# endif
  return libxsmm_gemm_diff(a, b);
#endif
}


#if defined(__MIC__)
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE
unsigned int libxsmm_gemm_diff_imci(const libxsmm_gemm_descriptor* a, const libxsmm_gemm_descriptor* b)
{
  assert(0 == LIBXSMM_MOD2(LIBXSMM_GEMM_DESCRIPTOR_SIZE, sizeof(unsigned int)));
  assert(16 >= LIBXSMM_DIV2(LIBXSMM_GEMM_DESCRIPTOR_SIZE, 4));
  assert(0 != a && 0 != b);
  {
    const __mmask16 mask = (0xFFFF >> (16 - LIBXSMM_DIV2(LIBXSMM_GEMM_DESCRIPTOR_SIZE, 4)));
    const __m512i a512 = _mm512_mask_loadunpackhi_epi32(
      _mm512_mask_loadunpacklo_epi32(_mm512_set1_epi32(0), mask, a),
      mask, ((const char*)a) + 32);
    const __m512i b512 = _mm512_mask_loadunpackhi_epi32(
      _mm512_mask_loadunpacklo_epi32(_mm512_set1_epi32(0), mask, b),
      mask, ((const char*)b) + 32);
    return _mm512_reduce_or_epi32(_mm512_xor_si512(a512, b512));
  }
}
#endif /*defined(__MIC__)*/

