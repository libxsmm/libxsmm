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


LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE
unsigned int libxsmm_gemm_diff_sse(const libxsmm_gemm_descriptor* a, const libxsmm_gemm_descriptor* b)
{
  return libxsmm_gemm_diff(a, b);
}


LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE LIBXSMM_INTRINSICS
unsigned int libxsmm_gemm_diff_avx(const libxsmm_gemm_descriptor* a, const libxsmm_gemm_descriptor* b)
{
#if defined(LIBXSMM_AVX_MAX) && (1 <= (LIBXSMM_AVX_MAX))
  __m256 ia, ib;
# if (28 == LIBXSMM_GEMM_DESCRIPTOR_SIZE) /* otherwise generate a compilation error */
#   if !defined(__CYGWIN__) && !(defined(__INTEL_COMPILER) && defined(_WIN32))
  struct { __m256i i32; } mask;
  mask.i32 = _mm256_set_epi32(0, -1, -1, -1, -1, -1, -1, -1);
#   else /* Cygwin/GCC: _mm256_set_epi32 causes an illegal instruction */
  const union { int32_t array[8]; __m256i i32; } mask = { { -1, -1, -1, -1, -1, -1, -1, 0 } };
#   endif
# endif
  assert(0 == LIBXSMM_MOD2(LIBXSMM_GEMM_DESCRIPTOR_SIZE, sizeof(unsigned int)));
  assert(8 >= LIBXSMM_DIV2(LIBXSMM_GEMM_DESCRIPTOR_SIZE, 4));
  assert(0 != a && 0 != b);

  ia = _mm256_maskload_ps((const float*)a, mask.i32);
  ib = _mm256_maskload_ps((const float*)b, mask.i32);

  return _mm256_testnzc_ps(ia, ib);
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
#if defined(LIBXSMM_AVX_MAX) && (2 <= (LIBXSMM_AVX_MAX))
  __m256i mask = _mm256_setzero_si256(), ia, ib;
  assert(0 == LIBXSMM_MOD2(LIBXSMM_GEMM_DESCRIPTOR_SIZE, sizeof(unsigned int)));
  assert(8 >= LIBXSMM_DIV2(LIBXSMM_GEMM_DESCRIPTOR_SIZE, 4));
  assert(0 != a && 0 != b);

  mask = _mm256_srai_epi32(mask, LIBXSMM_DIV2(LIBXSMM_GEMM_DESCRIPTOR_SIZE, 4));
  ia = _mm256_maskload_epi32((const void*)a, mask);
  ib = _mm256_maskload_epi32((const void*)b, mask);

  return _mm256_testnzc_si256(ia, ib);
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
  const __mmask16 mask = (0xFFFF >> (16 - LIBXSMM_DIV2(LIBXSMM_GEMM_DESCRIPTOR_SIZE, 4)));
  __m512i ia, ib; /* we do not care about the initial state */
  /* however, avoid warning about "variable is used before its value is set" */
  ia = ib = _mm512_set1_epi32(0);
  assert(0 == LIBXSMM_MOD2(LIBXSMM_GEMM_DESCRIPTOR_SIZE, sizeof(unsigned int)));
  assert(16 >= LIBXSMM_DIV2(LIBXSMM_GEMM_DESCRIPTOR_SIZE, 4));
  assert(0 != a && 0 != b);

  ia = _mm512_mask_loadunpackhi_epi32(
    _mm512_mask_loadunpacklo_epi32(ia/*some state*/, mask, a),
    mask, ((const char*)a) + 32);
  ib = _mm512_mask_loadunpackhi_epi32(
    _mm512_mask_loadunpacklo_epi32(ib/*some state*/, mask, b),
    mask, ((const char*)b) + 32);

  /* mask not required here since ia and ib are zero-initialized */
  return _mm512_reduce_or_epi32(_mm512_xor_si512(ia, ib));
}
#endif /*defined(__MIC__)*/

