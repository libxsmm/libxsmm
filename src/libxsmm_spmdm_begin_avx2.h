/******************************************************************************
** Copyright (c) 2016-2017, Intel Corporation                                **
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
/* Nadathur Satish, Hans Pabst (Intel Corp.)
******************************************************************************/

#if !defined(LIBXSMM_MAX_STATIC_TARGET_ARCH)
# error "libxsmm_intrinsics_x86.h not included!"
#endif

#if (LIBXSMM_X86_AVX2 <= LIBXSMM_MAX_STATIC_TARGET_ARCH)
#define SIMD_WIDTH_FP32 (8)
#define SIMDTYPE_FP32 __m256
#define SIMDTYPE_INT32 __m256i
#define SIMDMASKTYPE_FP32 __m256
#define _MM_SETZERO_FP32 _mm256_setzero_ps
#define _MM_SETZERO_INT32 _mm256_setzero_si256
#define _MM_SET1_FP32 _mm256_set1_ps
#define _MM_SET1_INT32 _mm256_set1_epi32
#define _MM_SET1_INT16 _mm256_set1_epi16
#define _MM_SET_INT32 _mm256_set_epi32
#define _MM_LOAD_FP32 _mm256_load_ps
#define _MM_LOADU_FP32 _mm256_loadu_ps
#define _MM_LOAD_INT32 _mm256_load_si256
#define _MM_STORE_INT32 _mm256_store_si256
#define _MM_LOADU_INT32(x) _mm256_loadu_si256( (__m256i const *)(x))
#define _MM_GATHER_INT32(Addr, idx, scale) _mm256_i32gather_epi32((Addr), (idx), (scale))
#define _MM_GATHER_FP32(Addr, idx, scale) _mm256_i32gather_ps(((float const *)(Addr)), (idx), (scale))
#define _MM_CMPNEQ_FP32(v1,v2) _mm256_cmp_ps(v1,v2,12)
#define _MM_STORE_FP32 _mm256_store_ps
#define _MM_STOREU_FP32 _mm256_storeu_ps
#define _MM_ADD_FP32 _mm256_add_ps
#define _MM_FMADD_FP32 _mm256_fmadd_ps
#define _MM_MUL_FP32 _mm256_mul_ps
#define _MM_PREFETCH(x, y) _mm_prefetch(x, y)
#define TRANSPOSE_SIMD_WIDTH_KERNEL(ptr_A, ldA, ptr_B, ldB) \
  { \
  __m256 ymm9  = _mm256_loadu_ps(ptr_A); \
  __m256 ymm10 = _mm256_loadu_ps(ptr_A + ldA); \
  __m256 ymm11 = _mm256_loadu_ps(ptr_A + 2*ldA); \
  __m256 ymm12 = _mm256_loadu_ps(ptr_A + 3*ldA); \
  __m256 ymm13 = _mm256_loadu_ps(ptr_A + 4*ldA); \
  __m256 ymm14 = _mm256_loadu_ps(ptr_A + 5*ldA); \
  __m256 ymm15 = _mm256_loadu_ps(ptr_A + 6*ldA); \
  __m256 ymm2  = _mm256_loadu_ps(ptr_A + 7*ldA); \
  __m256 ymm6  = _mm256_unpacklo_ps(ymm9, ymm10);\
  __m256 ymm1  = _mm256_unpacklo_ps(ymm11, ymm12);\
  __m256 ymm8  = _mm256_unpackhi_ps(ymm9, ymm10);\
  __m256 ymm0  = _mm256_unpacklo_ps(ymm13, ymm14);\
         ymm9  = _mm256_unpacklo_ps(ymm15, ymm2);{\
  __m256 ymm3  = _mm256_shuffle_ps(ymm6, ymm1, 0x4E);\
         ymm10 = _mm256_blend_ps(ymm6, ymm3, 0xCC);\
         ymm6  = _mm256_shuffle_ps(ymm0, ymm9, 0x4E);{\
  __m256 ymm7  = _mm256_unpackhi_ps(ymm11, ymm12);\
         ymm11 = _mm256_blend_ps(ymm0, ymm6, 0xCC);\
         ymm12 = _mm256_blend_ps(ymm3, ymm1, 0xCC);\
         ymm3  = _mm256_permute2f128_ps(ymm10, ymm11, 0x20);\
         _mm256_storeu_ps(ptr_B, ymm3);{\
  __m256 ymm5  = _mm256_unpackhi_ps(ymm13, ymm14);\
         ymm13 = _mm256_blend_ps(ymm6, ymm9, 0xCC);{\
  __m256 ymm4  = _mm256_unpackhi_ps(ymm15, ymm2);\
         ymm2  = _mm256_permute2f128_ps(ymm12, ymm13, 0x20);\
         _mm256_storeu_ps(ptr_B + ldB, ymm2);\
         ymm14 = _mm256_shuffle_ps(ymm8, ymm7, 0x4E);\
         ymm15 = _mm256_blend_ps(ymm14, ymm7, 0xCC);\
         ymm7  = _mm256_shuffle_ps(ymm5, ymm4, 0x4E);\
         ymm8  = _mm256_blend_ps(ymm8, ymm14, 0xCC);\
         ymm5  = _mm256_blend_ps(ymm5, ymm7, 0xCC);\
         ymm6  = _mm256_permute2f128_ps(ymm8, ymm5, 0x20);\
         _mm256_storeu_ps(ptr_B + 2*ldB, ymm6);\
         ymm4  = _mm256_blend_ps(ymm7, ymm4, 0xCC);\
         ymm7  = _mm256_permute2f128_ps(ymm15, ymm4, 0x20);\
         _mm256_storeu_ps(ptr_B + 3*ldB, ymm7);\
         ymm1  = _mm256_permute2f128_ps(ymm10, ymm11, 0x31);\
         ymm0  = _mm256_permute2f128_ps(ymm12, ymm13, 0x31);\
         _mm256_storeu_ps(ptr_B + 4*ldB, ymm1);\
         ymm5  = _mm256_permute2f128_ps(ymm8, ymm5, 0x31);\
         ymm4  = _mm256_permute2f128_ps(ymm15, ymm4, 0x31);\
         _mm256_storeu_ps(ptr_B + 5*ldB, ymm0);\
         _mm256_storeu_ps(ptr_B + 6*ldB, ymm5);\
         _mm256_storeu_ps(ptr_B + 7*ldB, ymm4);}}}}\
  }

#define TRANSPOSE_SIMD_WIDTH_KERNEL_BFLOAT16(ptr_A, ldA, ptr_B, ldB) \
  { \
  __m256 ymm9, ymm10, ymm11, ymm12, ymm13, ymm14, ymm15, ymm2; \
  __m256i vload_1 =  _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*)(ptr_A))); \
  vload_1 =  _mm256_inserti128_si256(vload_1, _mm_loadu_si128((const __m128i*)(ptr_A + ldA)), 1); \
  EXPAND_BFLOAT16(vload_1, ymm9, ymm10);{ \
  __m256i vload_2 =  _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*)(ptr_A + 2*ldA))); \
  vload_2 =  _mm256_inserti128_si256(vload_2, _mm_loadu_si128((const __m128i*)(ptr_A + 3*ldA)), 1); \
  EXPAND_BFLOAT16(vload_2, ymm11, ymm12);{ \
  __m256i vload_3 =  _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*)(ptr_A + 4*ldA))); \
  vload_3 =  _mm256_inserti128_si256(vload_3, _mm_loadu_si128((const __m128i*)(ptr_A + 5*ldA)), 1); \
  EXPAND_BFLOAT16(vload_3, ymm13, ymm14);{ \
  __m256i vload_4 =  _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*)(ptr_A + 6*ldA))); \
  vload_4 =  _mm256_inserti128_si256(vload_4, _mm_loadu_si128((const __m128i*)(ptr_A + 7*ldA)), 1); \
  EXPAND_BFLOAT16(vload_4, ymm15, ymm2);{ \
  __m256 ymm6  = _mm256_unpacklo_ps(ymm9, ymm10);\
  __m256 ymm1  = _mm256_unpacklo_ps(ymm11, ymm12);\
  __m256 ymm8  = _mm256_unpackhi_ps(ymm9, ymm10);\
  __m256 ymm0  = _mm256_unpacklo_ps(ymm13, ymm14);\
         ymm9  = _mm256_unpacklo_ps(ymm15, ymm2);{\
  __m256 ymm3  = _mm256_shuffle_ps(ymm6, ymm1, 0x4E);\
         ymm10 = _mm256_blend_ps(ymm6, ymm3, 0xCC);\
         ymm6  = _mm256_shuffle_ps(ymm0, ymm9, 0x4E);{\
  __m256 ymm7  = _mm256_unpackhi_ps(ymm11, ymm12);\
         ymm11 = _mm256_blend_ps(ymm0, ymm6, 0xCC);\
         ymm12 = _mm256_blend_ps(ymm3, ymm1, 0xCC);\
         ymm3  = _mm256_permute2f128_ps(ymm10, ymm11, 0x20);\
         _mm256_storeu_ps(ptr_B, ymm3);{\
  __m256 ymm5  = _mm256_unpackhi_ps(ymm13, ymm14);\
         ymm13 = _mm256_blend_ps(ymm6, ymm9, 0xCC);{\
  __m256 ymm4  = _mm256_unpackhi_ps(ymm15, ymm2);\
         ymm2  = _mm256_permute2f128_ps(ymm12, ymm13, 0x20);\
         _mm256_storeu_ps(ptr_B + ldB, ymm2);\
         ymm14 = _mm256_shuffle_ps(ymm8, ymm7, 0x4E);\
         ymm15 = _mm256_blend_ps(ymm14, ymm7, 0xCC);\
         ymm7  = _mm256_shuffle_ps(ymm5, ymm4, 0x4E);\
         ymm8  = _mm256_blend_ps(ymm8, ymm14, 0xCC);\
         ymm5  = _mm256_blend_ps(ymm5, ymm7, 0xCC);\
         ymm6  = _mm256_permute2f128_ps(ymm8, ymm5, 0x20);\
         _mm256_storeu_ps(ptr_B + 2*ldB, ymm6);\
         ymm4  = _mm256_blend_ps(ymm7, ymm4, 0xCC);\
         ymm7  = _mm256_permute2f128_ps(ymm15, ymm4, 0x20);\
         _mm256_storeu_ps(ptr_B + 3*ldB, ymm7);\
         ymm1  = _mm256_permute2f128_ps(ymm10, ymm11, 0x31);\
         ymm0  = _mm256_permute2f128_ps(ymm12, ymm13, 0x31);\
         _mm256_storeu_ps(ptr_B + 4*ldB, ymm1);\
         ymm5  = _mm256_permute2f128_ps(ymm8, ymm5, 0x31);\
         ymm4  = _mm256_permute2f128_ps(ymm15, ymm4, 0x31);\
         _mm256_storeu_ps(ptr_B + 5*ldB, ymm0);\
         _mm256_storeu_ps(ptr_B + 6*ldB, ymm5);\
         _mm256_storeu_ps(ptr_B + 7*ldB, ymm4);}}}}}}}}\
  }

#define COMPRESS_FP32(v, k, m, cnt) \
  { \
  const unsigned int mask = _mm256_movemask_ps(m); \
  const SIMDTYPE_INT32 vk = _MM_SET1_INT16((short)(k)); \
  const __m256i perm_ctrl = _mm256_load_si256(&shufmasks[mask]); \
  const __m256 v_packed = _mm256_permutevar8x32_ps(v, perm_ctrl); \
  const __m256i v_shuff = _mm256_load_si256(&shufmasks2[mask]); \
  const __m256i v_idx = _mm256_add_epi32(vk, v_shuff); \
  _mm256_storeu_ps(values_ptr + (cnt), v_packed); \
  _mm256_storeu_si256((__m256i *)(colidx_ptr + (cnt)), v_idx); \
  cnt = (unsigned short)((cnt) + _mm_popcnt_u32(mask)); \
  }

#define EXPAND_BFLOAT16(v, vlo_final, vhi_final) \
  { \
  const __m256i vlo = _mm256_unpacklo_epi16(vzero, v); \
  const __m256i vhi = _mm256_unpackhi_epi16(vzero, v); \
  vlo_final = _mm256_castsi256_ps(_mm256_permute2f128_si256(vlo, vhi, 0x20)); \
  vhi_final = _mm256_castsi256_ps(_mm256_permute2f128_si256(vlo, vhi, 0x31)); \
  }

#define COMPRESS_BFLOAT16(vlo, vhi, v) \
  { \
  const __m256i vtmp1 =  _mm256_castps_si256(_mm256_permute2f128_ps(vlo, vhi, 0x20)); \
  const __m256i vtmp2 =  _mm256_castps_si256(_mm256_permute2f128_ps(vlo, vhi, 0x31)); \
  const __m256i a = _mm256_srli_epi32(vtmp1, 16), b = _mm256_srli_epi32(vtmp2, 16); \
  v = _mm256_packus_epi32(a, b); \
  }

#endif

