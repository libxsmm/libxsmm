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
#include "libxsmm_intrinsics_x86.h"

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#include <stdint.h>
#include <string.h>
#if !defined(NDEBUG)
# include <assert.h>
# include <stdio.h>
#endif
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif


LIBXSMM_RETARGETABLE LIBXSMM_VISIBILITY_INTERNAL libxsmm_gemm_diff_function internal_gemm_diff_function = libxsmm_gemm_diff_sw;
LIBXSMM_RETARGETABLE LIBXSMM_VISIBILITY_INTERNAL libxsmm_gemm_diffn_function internal_gemm_diffn_function = libxsmm_gemm_diffn_sw;


LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE void libxsmm_gemm_diff_init(int target_arch)
{
#if defined(__MIC__)
  LIBXSMM_UNUSED(target_arch);
  internal_gemm_diffn_function = libxsmm_gemm_diffn_imci;
  internal_gemm_diff_function = libxsmm_gemm_diff_imci;
#else
  if (LIBXSMM_X86_AVX512 <= target_arch) {
    internal_gemm_diffn_function = libxsmm_gemm_diffn_avx512;
    internal_gemm_diff_function = libxsmm_gemm_diff_avx2;
  }
  else if (LIBXSMM_X86_AVX2 <= target_arch) {
    internal_gemm_diffn_function = libxsmm_gemm_diffn_avx2;
    internal_gemm_diff_function = libxsmm_gemm_diff_avx2;
  }
  else if (LIBXSMM_X86_AVX <= target_arch) {
    internal_gemm_diffn_function = libxsmm_gemm_diffn_avx;
    internal_gemm_diff_function = libxsmm_gemm_diff_avx;
  }
#endif
}


LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE void libxsmm_gemm_diff_finalize(void)
{
}


LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE
unsigned int libxsmm_gemm_diff(const libxsmm_gemm_descriptor* reference, const libxsmm_gemm_descriptor* desc)
{
  /* attempt to rely on static code path avoids to rely on capability of inlining pointer-based function call */
#if defined(LIBXSMM_GEMM_DIFF_SW) && (0 != LIBXSMM_GEMM_DIFF_SW)
  return libxsmm_gemm_diff_sw(reference, desc);
#elif defined(__MIC__)
  return libxsmm_gemm_diff_imci(reference, desc);
#elif defined(LIBXSMM_STATIC_TARGET_ARCH) && (LIBXSMM_X86_AVX2 <= LIBXSMM_STATIC_TARGET_ARCH)
  return libxsmm_gemm_diff_avx2(reference, desc);
#elif defined(LIBXSMM_STATIC_TARGET_ARCH) && (LIBXSMM_X86_AVX <= LIBXSMM_STATIC_TARGET_ARCH)
  return libxsmm_gemm_diff_avx(reference, desc);
#else /* pointer based function call */
  assert(0 != internal_gemm_diff_function);
  return (*internal_gemm_diff_function)(reference, desc);
#endif
}


LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE
unsigned int libxsmm_gemm_diff_sw(const libxsmm_gemm_descriptor* reference, const libxsmm_gemm_descriptor* desc)
{
#if defined(LIBXSMM_GEMM_DIFF_SW) && (2 == (LIBXSMM_GEMM_DIFF_SW))
  return 0 != memcmp(reference, desc, LIBXSMM_GEMM_DESCRIPTOR_SIZE);
#else
  const unsigned *const ia = (const unsigned int*)reference, *const ib = (const unsigned int*)desc;
  const unsigned int end = (LIBXSMM_GEMM_DESCRIPTOR_SIZE >> 2/*LOG2(sizeof(int))*/);
  unsigned int result, i;
  assert(0 == LIBXSMM_MOD2(LIBXSMM_GEMM_DESCRIPTOR_SIZE, sizeof(unsigned int)));
  assert(0 != reference && 0 != desc);
  result = ia[0] ^ ib[0];
  for (i = 1; i < end; ++i) {
    result |= (ia[i] ^ ib[i]);
  }
  return result;
#endif
}


LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE LIBXSMM_INTRINSICS
unsigned int libxsmm_gemm_diff_avx(const libxsmm_gemm_descriptor* reference, const libxsmm_gemm_descriptor* desc)
{
#if defined(LIBXSMM_MAX_STATIC_TARGET_ARCH) && (LIBXSMM_X86_AVX <= LIBXSMM_MAX_STATIC_TARGET_ARCH)
  assert(0 == LIBXSMM_MOD2(LIBXSMM_GEMM_DESCRIPTOR_SIZE, sizeof(unsigned int)));
  assert(8 >= LIBXSMM_DIV2(LIBXSMM_GEMM_DESCRIPTOR_SIZE, sizeof(unsigned int)));
  assert(0 != reference && 0 != desc);
  {
    int r0, r1;
    __m256i a256, b256;
# if (28 == LIBXSMM_GEMM_DESCRIPTOR_SIZE) /* otherwise generate a compile-time error */
#   if defined(__CYGWIN__) && !defined(NDEBUG) /* Cygwin/GCC: _mm256_set_epi32 may cause an illegal instruction */
    const union { int32_t array[8]; __m256i i; } m256 = { /* use literal value rather than yes/no
      in order to avoid warning about "initializer element is not computable at load time" */
      { 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x0 }
    };
#   else
    const int yes = 0x80000000, no = 0x0;
    struct { __m256i i; } m256;
    m256.i = _mm256_set_epi32(no, yes, yes, yes, yes, yes, yes, yes);
#   endif
# endif
# if defined(LIBXSMM_GEMM_DIFF_MASK_A) || !defined(LIBXSMM_GEMM_DIFF_ZERO_PADDED)
    a256 = _mm256_castps_si256(_mm256_maskload_ps((const float*)reference, m256.i));
# else
    /*a256 = _mm256_lddqu_si256((const __m256i*)reference);*/
    a256 = _mm256_loadu_si256((const __m256i*)reference);
# endif
    b256 = _mm256_castps_si256(_mm256_maskload_ps((const float*)desc, m256.i));
    r0 = _mm256_testnzc_si256(a256, b256);
    r1 = _mm256_testnzc_si256(b256, a256);
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
  return libxsmm_gemm_diff_sw(reference, desc);
#endif
}


LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE LIBXSMM_INTRINSICS
unsigned int libxsmm_gemm_diff_avx2(const libxsmm_gemm_descriptor* reference, const libxsmm_gemm_descriptor* desc)
{
#if defined(LIBXSMM_MAX_STATIC_TARGET_ARCH) && (LIBXSMM_X86_AVX2 <= LIBXSMM_MAX_STATIC_TARGET_ARCH)
  assert(0 == LIBXSMM_MOD2(LIBXSMM_GEMM_DESCRIPTOR_SIZE, sizeof(unsigned int)));
  assert(8 >= LIBXSMM_DIV2(LIBXSMM_GEMM_DESCRIPTOR_SIZE, sizeof(unsigned int)));
  assert(0 != reference && 0 != desc);
  {
# if (28 == LIBXSMM_GEMM_DESCRIPTOR_SIZE) /* otherwise generate a compile-time error */
    const int yes = 0x80000000, no = 0x0;
    const __m256i m256 = _mm256_set_epi32(no, yes, yes, yes, yes, yes, yes, yes);
# endif
# if defined(LIBXSMM_GEMM_DIFF_MASK_A) || !defined(LIBXSMM_GEMM_DIFF_ZERO_PADDED)
    const __m256i a256 = _mm256_maskload_epi32((const void*)reference, m256);
# else
    /*const __m256i a256 = _mm256_lddqu_si256((const __m256i*)reference);*/
    const __m256i a256 = _mm256_loadu_si256((const __m256i*)reference);
# endif
    const __m256i b256 = _mm256_maskload_epi32((const void*)desc, m256);
    const int r0 = _mm256_testnzc_si256(a256, b256);
    const int r1 = _mm256_testnzc_si256(b256, a256);
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
  return libxsmm_gemm_diff_avx(reference, desc);
#endif
}


LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE
unsigned int libxsmm_gemm_diff_imci(const libxsmm_gemm_descriptor* reference, const libxsmm_gemm_descriptor* desc)
{
#if defined(__MIC__)
  assert(0 ==  LIBXSMM_MOD2(LIBXSMM_GEMM_DESCRIPTOR_SIZE, sizeof(unsigned int)));
  assert(16 >= LIBXSMM_DIV2(LIBXSMM_GEMM_DESCRIPTOR_SIZE, sizeof(unsigned int)));
  assert(0 != reference && 0 != desc);
  {
    const __mmask16 mask = (0xFFFF >> (16 - (LIBXSMM_GEMM_DESCRIPTOR_SIZE >> 2/*LOG2(sizeof(int))*/)));
    const __m512i a512 = _mm512_mask_loadunpackhi_epi32(
      _mm512_mask_loadunpacklo_epi32(_mm512_set1_epi32(0), mask, reference),
      mask, ((const char*)reference) + 32);
    const __m512i b512 = _mm512_mask_loadunpackhi_epi32(
      _mm512_mask_loadunpacklo_epi32(_mm512_set1_epi32(0), mask, desc),
      mask, ((const char*)desc) + 32);
    return _mm512_reduce_or_epi32(_mm512_xor_si512(a512, b512));
  }
#else
  return libxsmm_gemm_diff_sw(reference, desc);
#endif
}


LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE
unsigned int libxsmm_gemm_diffn(const libxsmm_gemm_descriptor* reference, const libxsmm_gemm_descriptor* descs,
  unsigned int hint, unsigned int ndescs, int nbytes)
{
  /* attempt to rely on static code path avoids to rely on capability of inlining pointer-based function call */
#if defined(LIBXSMM_GEMM_DIFF_SW) && (0 != LIBXSMM_GEMM_DIFF_SW)
  return libxsmm_gemm_diffn_sw(reference, descs, hint, ndescs, nbytes);
#elif defined(LIBXSMM_STATIC_TARGET_ARCH) && (LIBXSMM_X86_AVX512 <= LIBXSMM_STATIC_TARGET_ARCH)
  return libxsmm_gemm_diffn_avx512(reference, descs, hint, ndescs, nbytes);
#elif defined(LIBXSMM_STATIC_TARGET_ARCH) && (LIBXSMM_X86_AVX2 <= LIBXSMM_STATIC_TARGET_ARCH)
  return libxsmm_gemm_diffn_avx2(reference, descs, hint, ndescs, nbytes);
#elif defined(LIBXSMM_STATIC_TARGET_ARCH) && (LIBXSMM_X86_AVX <= LIBXSMM_STATIC_TARGET_ARCH)
  return libxsmm_gemm_diffn_avx(reference, descs, hint, ndescs, nbytes);
#else /* pointer based function call */
  assert(0 != internal_gemm_diffn_function);
  return (*internal_gemm_diffn_function)(reference, descs, hint, ndescs, nbytes);
#endif
}


LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE
unsigned int libxsmm_gemm_diffn_sw(const libxsmm_gemm_descriptor* reference, const libxsmm_gemm_descriptor* descs,
  unsigned int hint, unsigned int ndescs, int nbytes)
{
  const char *const desc = (const char*)descs;
  const unsigned int end = hint + ndescs;
  unsigned int i;
#if defined(LIBXSMM_GEMM_DIFF_SW) && (2 == (LIBXSMM_GEMM_DIFF_SW)) /* most general implementation */
  for (i = hint; i < end; ++i) {
    const unsigned int j = i % ndescs; /* wrap around index */
    /* negative stride runs backwards */
    const char *const d = desc + j * nbytes;
    if (0 == libxsmm_gemm_diff(reference, (const libxsmm_gemm_descriptor*)d)) {
      return j;
    }
  }
#else /* slightly more optimized implementation */
  assert(/*is pot*/ndescs == (1 << LIBXSMM_LOG2(ndescs)));
  for (i = hint; i < end; ++i) {
    const unsigned int j = LIBXSMM_MOD2(i, ndescs); /* wrap around index */
    const char *const d = desc + j * nbytes;
    if (0 == libxsmm_gemm_diff(reference, (const libxsmm_gemm_descriptor*)d)) {
      return j;
    }
  }
#endif
  return ndescs;
}


LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE LIBXSMM_INTRINSICS
unsigned int libxsmm_gemm_diffn_avx(const libxsmm_gemm_descriptor* reference, const libxsmm_gemm_descriptor* descs,
  unsigned int hint, unsigned int ndescs, int nbytes)
{
#if defined(LIBXSMM_MAX_STATIC_TARGET_ARCH) && (LIBXSMM_X86_AVX <= LIBXSMM_MAX_STATIC_TARGET_ARCH)
  assert(/*is pot*/ndescs == (1 << LIBXSMM_LOG2(ndescs)));
  assert(32 == nbytes); /* padded descriptor array */
  {
    const unsigned int end = hint + ndescs;
    const char *const desc = (const char*)descs;
    __m256i a256;
    unsigned int i;
# if (28 == LIBXSMM_GEMM_DESCRIPTOR_SIZE) /* otherwise generate a compile-time error */
#   if defined(__CYGWIN__) && !defined(NDEBUG) /* Cygwin/GCC: _mm256_set_epi32 may cause an illegal instruction */
    const union { int32_t array[8]; __m256i i; } m256 = { /* use literal value rather than yes/no
      in order to avoid warning about "initializer element is not computable at load time" */
      { 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x0 }
    };
#   else
    const int yes = 0x80000000, no = 0x0;
    struct { __m256i i; } m256;
    m256.i = _mm256_set_epi32(no, yes, yes, yes, yes, yes, yes, yes);
#   endif
# endif
# if defined(LIBXSMM_GEMM_DIFF_MASK_A) || !defined(LIBXSMM_GEMM_DIFF_ZERO_PADDED)
    a256 = _mm256_castps_si256(_mm256_maskload_ps((const float*)reference, m256.i));
# else
    /*a256 = _mm256_lddqu_si256((const __m256i*)reference);*/
    a256 = _mm256_loadu_si256((const __m256i*)reference);
# endif
    for (i = hint; i < end; ++i) {
      const unsigned int j = LIBXSMM_MOD2(i, ndescs); /* wrap around index */
#if defined(LIBXSMM_GEMM_DIFF_ZERO_PADDED)
      const __m256i b256 = _mm256_loadu_si256((const __m256i*)(desc + j * nbytes));
#else
      const __m256i b256 = _mm256_castps_si256(_mm256_maskload_ps((const float*)(desc + j * nbytes), m256.i));
#endif
      if (0 == _mm256_testnzc_si256(a256, b256) && 0 == _mm256_testnzc_si256(b256, a256)) {
        return j;
      }
    }
  }
  return ndescs;
#else
# if !defined(NDEBUG) /* library code is expected to be mute */
  static LIBXSMM_TLS int once = 0;
  if (0 == once) {
    fprintf(stderr, "LIBXSMM: unable to enter AVX instruction code path!\n");
    once = 1;
  }
# endif
  return libxsmm_gemm_diffn_sw(reference, descs, hint, ndescs, nbytes);
#endif
}


LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE LIBXSMM_INTRINSICS
unsigned int libxsmm_gemm_diffn_avx2(const libxsmm_gemm_descriptor* reference, const libxsmm_gemm_descriptor* descs,
  unsigned int hint, unsigned int ndescs, int nbytes)
{
#if defined(LIBXSMM_MAX_STATIC_TARGET_ARCH) && (LIBXSMM_X86_AVX2 <= LIBXSMM_MAX_STATIC_TARGET_ARCH)
  assert(/*is pot*/ndescs == (1 << LIBXSMM_LOG2(ndescs)));
  assert(32 == nbytes); /* padded descriptor array */
  {
    const unsigned int end = hint + ndescs;
    const char *const desc = (const char*)descs;
    unsigned int i;
# if (28 == LIBXSMM_GEMM_DESCRIPTOR_SIZE) /* otherwise generate a compile-time error */
    const int yes = 0x80000000, no = 0x0;
    const __m256i m256 = _mm256_set_epi32(no, yes, yes, yes, yes, yes, yes, yes);
# endif
# if defined(LIBXSMM_GEMM_DIFF_MASK_A) || !defined(LIBXSMM_GEMM_DIFF_ZERO_PADDED)
    const __m256i a256 = _mm256_maskload_epi32((const void*)reference, m256);
# else
    /*const __m256i a256 = _mm256_lddqu_si256((const __m256i*)reference);*/
    const __m256i a256 = _mm256_loadu_si256((const __m256i*)reference);
#endif
    for (i = hint; i < end; ++i) {
      const unsigned int j = LIBXSMM_MOD2(i, ndescs); /* wrap around index */
#if defined(LIBXSMM_GEMM_DIFF_ZERO_PADDED)
      const __m256i b256 = _mm256_loadu_si256((const __m256i*)(desc + j * nbytes));
#else
      const __m256i b256 = _mm256_maskload_epi32((const void*)(desc + j * nbytes), m256);
#endif
      if (0 == _mm256_testnzc_si256(a256, b256) && 0 == _mm256_testnzc_si256(b256, a256)) {
        return j;
      }
    }
  }
  return ndescs;
#else
# if !defined(NDEBUG) /* library code is expected to be mute */
  static LIBXSMM_TLS int once = 0;
  if (0 == once) {
    fprintf(stderr, "LIBXSMM: unable to enter AVX2 instruction code path!\n");
    once = 1;
  }
# endif
  return libxsmm_gemm_diffn_avx(reference, descs, hint, ndescs, nbytes);
#endif
}


LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE LIBXSMM_INTRINSICS
unsigned int libxsmm_gemm_diffn_avx512(const libxsmm_gemm_descriptor* reference, const libxsmm_gemm_descriptor* descs,
  unsigned int hint, unsigned int ndescs, int nbytes)
{
#if defined(LIBXSMM_MAX_STATIC_TARGET_ARCH) && (LIBXSMM_X86_AVX512 <= LIBXSMM_MAX_STATIC_TARGET_ARCH)
  assert(/*is pot*/ndescs == (1 << LIBXSMM_LOG2(ndescs)));
  assert(32 == nbytes); /* padded descriptor array */
  {
    const unsigned int hint_even = (hint & 0xFFFFFFFE), end = hint_even + ndescs;
    const char *const desc = (const char*)descs;
    unsigned int i;
# if (28 == LIBXSMM_GEMM_DESCRIPTOR_SIZE) /* otherwise generate a compile-time error */
    const __mmask16 mask_lo = 0x7F, mask_hi = 0x7F00;
    const int yes = 0x80000000, no = 0x0;
    const __m256i m256 = _mm256_set_epi32(no, yes, yes, yes, yes, yes, yes, yes);
# endif
# if defined(LIBXSMM_GEMM_DIFF_MASK_A)
    const __m256i a256 = _mm256_maskload_epi32((const void*)reference, m256);
# else
    /* SKX: consider _mm256_maskz_expandloadu_epi32 */
    const __m256i a256 = _mm256_loadu_si256((const __m256i*)reference);
#endif
    const __m512i a512 = _mm512_broadcast_i64x4(a256);
    for (i = hint_even; i < end; i += 2) {
      const unsigned int j = LIBXSMM_MOD2(i, ndescs); /* wrap around index */
      const __m512i b512 = _mm512_loadu_si512(desc + j * nbytes);
      const __m512i c512 = _mm512_xor_si512(a512, b512);
      if (0 == _mm512_mask_reduce_or_epi32(mask_lo, c512)) {
        return j;
      }
      if (0 == _mm512_mask_reduce_or_epi32(mask_hi, c512)) {
        return j + 1;
      }
    }
  }
  return ndescs;
#else
# if !defined(NDEBUG) /* library code is expected to be mute */
  static LIBXSMM_TLS int once = 0;
  if (0 == once) {
    fprintf(stderr, "LIBXSMM: unable to enter AVX-512 instruction code path!\n");
    once = 1;
  }
# endif
  return libxsmm_gemm_diffn_avx2(reference, descs, hint, ndescs, nbytes);
#endif
}


LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE
unsigned int libxsmm_gemm_diffn_imci(const libxsmm_gemm_descriptor* reference, const libxsmm_gemm_descriptor* descs,
  unsigned int hint, unsigned int ndescs, int nbytes)
{
#if defined(__MIC__) && 0
  assert(/*is pot*/ndescs == (1 << LIBXSMM_LOG2(ndescs)));
  assert(32 == nbytes); /* padded descriptor array */
  {
    const unsigned int hint_even = (hint & 0xFFFFFFFE), end = hint_even + ndescs;
    const char *const desc = (const char*)descs;
    unsigned int i;
# if (28 == LIBXSMM_GEMM_DESCRIPTOR_SIZE) /* otherwise generate a compile-time error */
    const __mmask16 mask_lo = 0x7F, mask_hi = 0x7F00;
# endif
    const __m512i h512 = _mm512_mask_loadunpackhi_epi32(
      _mm512_mask_loadunpacklo_epi32(_mm512_set1_epi32(0), mask_lo, reference),
      mask_lo, ((const char*)reference) + 32);
    const __m512i a512 = _mm512_permute4f128_epi32(h512, 0x44);
    for (i = hint_even; i < end; i += 2) {
      const unsigned int j = LIBXSMM_MOD2(i, ndescs); /* wrap around index */
      const char *const d = desc + j * nbytes;
      const __m512i b512 = _mm512_loadunpackhi_epi32(_mm512_loadunpacklo_epi32(_mm512_set1_epi32(0), d), d + 32);
      const __m512i c512 = _mm512_xor_si512(a512, b512);
      if (0 == _mm512_mask_reduce_or_epi32(mask_lo, c512)) {
        return j;
      }
      if (0 == _mm512_mask_reduce_or_epi32(mask_hi, c512)) {
        return j + 1;
      }
    }
  }
  return ndescs;
#else
  return libxsmm_gemm_diffn_sw(reference, descs, hint, ndescs, nbytes);
#endif
}

