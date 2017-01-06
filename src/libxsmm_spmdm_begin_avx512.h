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

#if (LIBXSMM_X86_AVX512_CORE <= LIBXSMM_MAX_STATIC_TARGET_ARCH)
#define SIMD_WIDTH_FP32 (16)
#define SIMDTYPE_FP32 __m512
#define SIMDTYPE_INT32 __m512i
#define SIMDMASKTYPE_FP32 __mmask16
#define _MM_SETZERO_FP32 _mm512_setzero_ps
#define _MM_SETZERO_INT32 _mm512_setzero_epi32
#define _MM_SET1_FP32 _mm512_set1_ps
#define _MM_SET1_INT32 _mm512_set1_epi32
#define _MM_SET1_INT16 _mm512_set1_epi16
#define _MM_SET_INT32 _mm512_set_epi32
#define _MM_LOAD_FP32 _mm512_load_ps
#define _MM_LOADU_FP32 _mm512_loadu_ps
#define _MM_LOAD_INT32 _mm512_load_epi32
#define _MM_STORE_INT32 _mm512_store_epi32
#define _MM_LOADU_INT32(x) _mm512_loadu_si512( (void const *)(x))
#define _MM_GATHER_INT32(Addr, idx, scale) _mm512_i32gather_epi32((idx), (Addr), (scale))
#define _MM_GATHER_FP32(Addr, idx, scale) _mm512_i32gather_ps((idx), (Addr), (scale))
#define _MM_CMPNEQ_FP32(v1,v2) _mm512_cmp_ps_mask(v1,v2,12)
#define _MM_STORE_FP32 _mm512_store_ps
#define _MM_STOREU_FP32 _mm512_storeu_ps
#define _MM_ADD_FP32 _mm512_add_ps
#define _MM_FMADD_FP32 _mm512_fmadd_ps
#define _MM_MUL_FP32 _mm512_mul_ps
#define _MM_PREFETCH(x, y) _mm_prefetch(x, y)
#define TRANSPOSE_SIMD_WIDTH_KERNEL(ptr_A, ldA, ptr_B, ldB) \
  { \
    __m512 r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, ra, rb, rc, rd, re, rf;\
    __m512 t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, ta, tb, tc, td, te, tf;\
    r0 = _mm512_loadu_ps(ptr_A);\
    r1 = _mm512_loadu_ps(ptr_A + ldA);\
    r2 = _mm512_loadu_ps(ptr_A + 2*ldA);\
    r3 = _mm512_loadu_ps(ptr_A + 3*ldA);\
    r4 = _mm512_loadu_ps(ptr_A + 4*ldA);\
    r5 = _mm512_loadu_ps(ptr_A + 5*ldA);\
    r6 = _mm512_loadu_ps(ptr_A + 6*ldA);\
    r7 = _mm512_loadu_ps(ptr_A + 7*ldA);\
    r8 = _mm512_loadu_ps(ptr_A + 8*ldA);\
    r9 = _mm512_loadu_ps(ptr_A + 9*ldA);\
    ra = _mm512_loadu_ps(ptr_A + 10*ldA);\
    rb = _mm512_loadu_ps(ptr_A + 11*ldA);\
    rc = _mm512_loadu_ps(ptr_A + 12*ldA);\
    rd = _mm512_loadu_ps(ptr_A + 13*ldA);\
    re = _mm512_loadu_ps(ptr_A + 14*ldA);\
    rf = _mm512_loadu_ps(ptr_A + 15*ldA);\
    \
    t0 = _mm512_unpacklo_ps(r0,r1);\
    t1 = _mm512_unpackhi_ps(r0,r1);\
    t2 = _mm512_unpacklo_ps(r2,r3);\
    t3 = _mm512_unpackhi_ps(r2,r3);\
    t4 = _mm512_unpacklo_ps(r4,r5);\
    t5 = _mm512_unpackhi_ps(r4,r5);\
    t6 = _mm512_unpacklo_ps(r6,r7);\
    t7 = _mm512_unpackhi_ps(r6,r7);\
    t8 = _mm512_unpacklo_ps(r8,r9);\
    t9 = _mm512_unpackhi_ps(r8,r9);\
    ta = _mm512_unpacklo_ps(ra,rb);\
    tb = _mm512_unpackhi_ps(ra,rb);\
    tc = _mm512_unpacklo_ps(rc,rd);\
    td = _mm512_unpackhi_ps(rc,rd);\
    te = _mm512_unpacklo_ps(re,rf);\
    tf = _mm512_unpackhi_ps(re,rf);\
    \
    { const __m512d td1 = _mm512_castps_pd(t0), td2 = _mm512_castps_pd(t2);\
      r0 = _mm512_castpd_ps(_mm512_unpacklo_pd(td1, td2));\
      r1 = _mm512_castpd_ps(_mm512_unpackhi_pd(td1, td2));}\
    { const __m512d td1 = _mm512_castps_pd(t1), td2 = _mm512_castps_pd(t3);\
      r2 = _mm512_castpd_ps(_mm512_unpacklo_pd(td1, td2));\
      r3 = _mm512_castpd_ps(_mm512_unpackhi_pd(td1, td2));}\
    { const __m512d td1 = _mm512_castps_pd(t4), td2 = _mm512_castps_pd(t6);\
      r4 = _mm512_castpd_ps(_mm512_unpacklo_pd(td1, td2));\
      r5 = _mm512_castpd_ps(_mm512_unpackhi_pd(td1, td2));}\
    { const __m512d td1 = _mm512_castps_pd(t5), td2 = _mm512_castps_pd(t7);\
      r6 = _mm512_castpd_ps(_mm512_unpacklo_pd(td1, td2));\
      r7 = _mm512_castpd_ps(_mm512_unpackhi_pd(td1, td2));}\
    { const __m512d td1 = _mm512_castps_pd(t8), td2 = _mm512_castps_pd(ta);\
      r8 = _mm512_castpd_ps(_mm512_unpacklo_pd(td1, td2));\
      r9 = _mm512_castpd_ps(_mm512_unpackhi_pd(td1, td2));}\
    { const __m512d td1 = _mm512_castps_pd(t9), td2 = _mm512_castps_pd(tb);\
      ra = _mm512_castpd_ps(_mm512_unpacklo_pd(td1, td2));\
      rb = _mm512_castpd_ps(_mm512_unpackhi_pd(td1, td2));}\
    { const __m512d td1 = _mm512_castps_pd(tc), td2 = _mm512_castps_pd(te);\
      rc = _mm512_castpd_ps(_mm512_unpacklo_pd(td1, td2));\
      rd = _mm512_castpd_ps(_mm512_unpackhi_pd(td1, td2));}\
    { const __m512d td1 = _mm512_castps_pd(td), td2 = _mm512_castps_pd(tf);\
      re = _mm512_castpd_ps(_mm512_unpacklo_pd(td1, td2));\
      rf = _mm512_castpd_ps(_mm512_unpackhi_pd(td1, td2));}\
    \
    t0 = _mm512_shuffle_f32x4(r0, r4, 0x88);\
    t1 = _mm512_shuffle_f32x4(r1, r5, 0x88);\
    t2 = _mm512_shuffle_f32x4(r2, r6, 0x88);\
    t3 = _mm512_shuffle_f32x4(r3, r7, 0x88);\
    t4 = _mm512_shuffle_f32x4(r0, r4, 0xdd);\
    t5 = _mm512_shuffle_f32x4(r1, r5, 0xdd);\
    t6 = _mm512_shuffle_f32x4(r2, r6, 0xdd);\
    t7 = _mm512_shuffle_f32x4(r3, r7, 0xdd);\
    t8 = _mm512_shuffle_f32x4(r8, rc, 0x88);\
    t9 = _mm512_shuffle_f32x4(r9, rd, 0x88);\
    ta = _mm512_shuffle_f32x4(ra, re, 0x88);\
    tb = _mm512_shuffle_f32x4(rb, rf, 0x88);\
    tc = _mm512_shuffle_f32x4(r8, rc, 0xdd);\
    td = _mm512_shuffle_f32x4(r9, rd, 0xdd);\
    te = _mm512_shuffle_f32x4(ra, re, 0xdd);\
    tf = _mm512_shuffle_f32x4(rb, rf, 0xdd);\
    \
    r0 = _mm512_shuffle_f32x4(t0, t8, 0x88);\
    r1 = _mm512_shuffle_f32x4(t1, t9, 0x88);\
    r2 = _mm512_shuffle_f32x4(t2, ta, 0x88);\
    r3 = _mm512_shuffle_f32x4(t3, tb, 0x88);\
    r4 = _mm512_shuffle_f32x4(t4, tc, 0x88);\
    r5 = _mm512_shuffle_f32x4(t5, td, 0x88);\
    r6 = _mm512_shuffle_f32x4(t6, te, 0x88);\
    r7 = _mm512_shuffle_f32x4(t7, tf, 0x88);\
    r8 = _mm512_shuffle_f32x4(t0, t8, 0xdd);\
    r9 = _mm512_shuffle_f32x4(t1, t9, 0xdd);\
    ra = _mm512_shuffle_f32x4(t2, ta, 0xdd);\
    rb = _mm512_shuffle_f32x4(t3, tb, 0xdd);\
    rc = _mm512_shuffle_f32x4(t4, tc, 0xdd);\
    rd = _mm512_shuffle_f32x4(t5, td, 0xdd);\
    re = _mm512_shuffle_f32x4(t6, te, 0xdd);\
    rf = _mm512_shuffle_f32x4(t7, tf, 0xdd);\
    \
    _mm512_storeu_ps(ptr_B + 0*ldB, r0);\
    _mm512_storeu_ps(ptr_B + 1*ldB, r1);\
    _mm512_storeu_ps(ptr_B + 2*ldB, r2);\
    _mm512_storeu_ps(ptr_B + 3*ldB, r3);\
    _mm512_storeu_ps(ptr_B + 4*ldB, r4);\
    _mm512_storeu_ps(ptr_B + 5*ldB, r5);\
    _mm512_storeu_ps(ptr_B + 6*ldB, r6);\
    _mm512_storeu_ps(ptr_B + 7*ldB, r7);\
    _mm512_storeu_ps(ptr_B + 8*ldB, r8);\
    _mm512_storeu_ps(ptr_B + 9*ldB, r9);\
    _mm512_storeu_ps(ptr_B + 10*ldB, ra);\
    _mm512_storeu_ps(ptr_B + 11*ldB, rb);\
    _mm512_storeu_ps(ptr_B + 12*ldB, rc);\
    _mm512_storeu_ps(ptr_B + 13*ldB, rd);\
    _mm512_storeu_ps(ptr_B + 14*ldB, re);\
    _mm512_storeu_ps(ptr_B + 15*ldB, rf);\
  }

#define TRANSPOSE_SIMD_WIDTH_KERNEL_BFLOAT16(ptr_A, ldA, ptr_B, ldB) \
  { \
    __m512 r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, ra, rb, rc, rd, re, rf;\
    __m512 t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, ta, tb, tc, td, te, tf;\
    __m512i vload_1 =  _mm512_castsi256_si512(_mm256_loadu_si256((const __m256i*)(ptr_A))); \
    vload_1 =  _mm512_inserti32x8(vload_1, _mm256_loadu_si256((const __m256i*)(ptr_A + ldA)), 1); \
    EXPAND_BFLOAT16(vload_1, r0, r1);{ \
    __m512i vload_2 =  _mm512_castsi256_si512(_mm256_loadu_si256((const __m256i*)(ptr_A + 2*ldA))); \
    vload_2 =  _mm512_inserti32x8(vload_2, _mm256_loadu_si256((const __m256i*)(ptr_A + 3*ldA)), 1); \
    EXPAND_BFLOAT16(vload_2, r2, r3);{ \
    __m512i vload_3 =  _mm512_castsi256_si512(_mm256_loadu_si256((const __m256i*)(ptr_A + 4*ldA))); \
    vload_3 =  _mm512_inserti32x8(vload_3, _mm256_loadu_si256((const __m256i*)(ptr_A + 5*ldA)), 1); \
    EXPAND_BFLOAT16(vload_3, r4, r5);{ \
    __m512i vload_4 =  _mm512_castsi256_si512(_mm256_loadu_si256((const __m256i*)(ptr_A + 6*ldA))); \
    vload_4 =  _mm512_inserti32x8(vload_4, _mm256_loadu_si256((const __m256i*)(ptr_A + 7*ldA)), 1); \
    EXPAND_BFLOAT16(vload_4, r6, r7);{ \
    __m512i vload_5 =  _mm512_castsi256_si512(_mm256_loadu_si256((const __m256i*)(ptr_A + 8*ldA))); \
    vload_5 =  _mm512_inserti32x8(vload_5, _mm256_loadu_si256((const __m256i*)(ptr_A + 9*ldA)), 1); \
    EXPAND_BFLOAT16(vload_5, r8, r9);{ \
    __m512i vload_6 =  _mm512_castsi256_si512(_mm256_loadu_si256((const __m256i*)(ptr_A + 10*ldA))); \
    vload_6 =  _mm512_inserti32x8(vload_6, _mm256_loadu_si256((const __m256i*)(ptr_A + 11*ldA)), 1); \
    EXPAND_BFLOAT16(vload_6, ra, rb);{ \
    __m512i vload_7 =  _mm512_castsi256_si512(_mm256_loadu_si256((const __m256i*)(ptr_A + 12*ldA))); \
    vload_7 =  _mm512_inserti32x8(vload_7, _mm256_loadu_si256((const __m256i*)(ptr_A + 13*ldA)), 1); \
    EXPAND_BFLOAT16(vload_7, rc, rd);{ \
    __m512i vload_8 =  _mm512_castsi256_si512(_mm256_loadu_si256((const __m256i*)(ptr_A + 14*ldA))); \
    vload_8 =  _mm512_inserti32x8(vload_8, _mm256_loadu_si256((const __m256i*)(ptr_A + 15*ldA)), 1); \
    EXPAND_BFLOAT16(vload_8, re, rf); \
    \
    t0 = _mm512_unpacklo_ps(r0,r1);\
    t1 = _mm512_unpackhi_ps(r0,r1);\
    t2 = _mm512_unpacklo_ps(r2,r3);\
    t3 = _mm512_unpackhi_ps(r2,r3);\
    t4 = _mm512_unpacklo_ps(r4,r5);\
    t5 = _mm512_unpackhi_ps(r4,r5);\
    t6 = _mm512_unpacklo_ps(r6,r7);\
    t7 = _mm512_unpackhi_ps(r6,r7);\
    t8 = _mm512_unpacklo_ps(r8,r9);\
    t9 = _mm512_unpackhi_ps(r8,r9);\
    ta = _mm512_unpacklo_ps(ra,rb);\
    tb = _mm512_unpackhi_ps(ra,rb);\
    tc = _mm512_unpacklo_ps(rc,rd);\
    td = _mm512_unpackhi_ps(rc,rd);\
    te = _mm512_unpacklo_ps(re,rf);\
    tf = _mm512_unpackhi_ps(re,rf);\
    \
    { const __m512d td1 = _mm512_castps_pd(t0), td2 = _mm512_castps_pd(t2);\
      r0 = _mm512_castpd_ps(_mm512_unpacklo_pd(td1, td2));\
      r1 = _mm512_castpd_ps(_mm512_unpackhi_pd(td1, td2));}\
    { const __m512d td1 = _mm512_castps_pd(t1), td2 = _mm512_castps_pd(t3);\
      r2 = _mm512_castpd_ps(_mm512_unpacklo_pd(td1, td2));\
      r3 = _mm512_castpd_ps(_mm512_unpackhi_pd(td1, td2));}\
    { const __m512d td1 = _mm512_castps_pd(t4), td2 = _mm512_castps_pd(t6);\
      r4 = _mm512_castpd_ps(_mm512_unpacklo_pd(td1, td2));\
      r5 = _mm512_castpd_ps(_mm512_unpackhi_pd(td1, td2));}\
    { const __m512d td1 = _mm512_castps_pd(t5), td2 = _mm512_castps_pd(t7);\
      r6 = _mm512_castpd_ps(_mm512_unpacklo_pd(td1, td2));\
      r7 = _mm512_castpd_ps(_mm512_unpackhi_pd(td1, td2));}\
    { const __m512d td1 = _mm512_castps_pd(t8), td2 = _mm512_castps_pd(ta);\
      r8 = _mm512_castpd_ps(_mm512_unpacklo_pd(td1, td2));\
      r9 = _mm512_castpd_ps(_mm512_unpackhi_pd(td1, td2));}\
    { const __m512d td1 = _mm512_castps_pd(t9), td2 = _mm512_castps_pd(tb);\
      ra = _mm512_castpd_ps(_mm512_unpacklo_pd(td1, td2));\
      rb = _mm512_castpd_ps(_mm512_unpackhi_pd(td1, td2));}\
    { const __m512d td1 = _mm512_castps_pd(tc), td2 = _mm512_castps_pd(te);\
      rc = _mm512_castpd_ps(_mm512_unpacklo_pd(td1, td2));\
      rd = _mm512_castpd_ps(_mm512_unpackhi_pd(td1, td2));}\
    { const __m512d td1 = _mm512_castps_pd(td), td2 = _mm512_castps_pd(tf);\
      re = _mm512_castpd_ps(_mm512_unpacklo_pd(td1, td2));\
      rf = _mm512_castpd_ps(_mm512_unpackhi_pd(td1, td2));}\
    \
    t0 = _mm512_shuffle_f32x4(r0, r4, 0x88);\
    t1 = _mm512_shuffle_f32x4(r1, r5, 0x88);\
    t2 = _mm512_shuffle_f32x4(r2, r6, 0x88);\
    t3 = _mm512_shuffle_f32x4(r3, r7, 0x88);\
    t4 = _mm512_shuffle_f32x4(r0, r4, 0xdd);\
    t5 = _mm512_shuffle_f32x4(r1, r5, 0xdd);\
    t6 = _mm512_shuffle_f32x4(r2, r6, 0xdd);\
    t7 = _mm512_shuffle_f32x4(r3, r7, 0xdd);\
    t8 = _mm512_shuffle_f32x4(r8, rc, 0x88);\
    t9 = _mm512_shuffle_f32x4(r9, rd, 0x88);\
    ta = _mm512_shuffle_f32x4(ra, re, 0x88);\
    tb = _mm512_shuffle_f32x4(rb, rf, 0x88);\
    tc = _mm512_shuffle_f32x4(r8, rc, 0xdd);\
    td = _mm512_shuffle_f32x4(r9, rd, 0xdd);\
    te = _mm512_shuffle_f32x4(ra, re, 0xdd);\
    tf = _mm512_shuffle_f32x4(rb, rf, 0xdd);\
    \
    r0 = _mm512_shuffle_f32x4(t0, t8, 0x88);\
    r1 = _mm512_shuffle_f32x4(t1, t9, 0x88);\
    r2 = _mm512_shuffle_f32x4(t2, ta, 0x88);\
    r3 = _mm512_shuffle_f32x4(t3, tb, 0x88);\
    r4 = _mm512_shuffle_f32x4(t4, tc, 0x88);\
    r5 = _mm512_shuffle_f32x4(t5, td, 0x88);\
    r6 = _mm512_shuffle_f32x4(t6, te, 0x88);\
    r7 = _mm512_shuffle_f32x4(t7, tf, 0x88);\
    r8 = _mm512_shuffle_f32x4(t0, t8, 0xdd);\
    r9 = _mm512_shuffle_f32x4(t1, t9, 0xdd);\
    ra = _mm512_shuffle_f32x4(t2, ta, 0xdd);\
    rb = _mm512_shuffle_f32x4(t3, tb, 0xdd);\
    rc = _mm512_shuffle_f32x4(t4, tc, 0xdd);\
    rd = _mm512_shuffle_f32x4(t5, td, 0xdd);\
    re = _mm512_shuffle_f32x4(t6, te, 0xdd);\
    rf = _mm512_shuffle_f32x4(t7, tf, 0xdd);\
    \
    _mm512_storeu_ps(ptr_B + 0*ldB, r0);\
    _mm512_storeu_ps(ptr_B + 1*ldB, r1);\
    _mm512_storeu_ps(ptr_B + 2*ldB, r2);\
    _mm512_storeu_ps(ptr_B + 3*ldB, r3);\
    _mm512_storeu_ps(ptr_B + 4*ldB, r4);\
    _mm512_storeu_ps(ptr_B + 5*ldB, r5);\
    _mm512_storeu_ps(ptr_B + 6*ldB, r6);\
    _mm512_storeu_ps(ptr_B + 7*ldB, r7);\
    _mm512_storeu_ps(ptr_B + 8*ldB, r8);\
    _mm512_storeu_ps(ptr_B + 9*ldB, r9);\
    _mm512_storeu_ps(ptr_B + 10*ldB, ra);\
    _mm512_storeu_ps(ptr_B + 11*ldB, rb);\
    _mm512_storeu_ps(ptr_B + 12*ldB, rc);\
    _mm512_storeu_ps(ptr_B + 13*ldB, rd);\
    _mm512_storeu_ps(ptr_B + 14*ldB, re);\
    _mm512_storeu_ps(ptr_B + 15*ldB, rf);}}}}}}}\
  }

#define COMPRESS_FP32(v, k, m, cnt) \
{ \
  _mm512_mask_compressstoreu_ps(values_ptr + (cnt), m, v); \
  { \
    __m256i vk1 = _mm256_set1_epi16((short)(k)); \
    __m256i vk2 = _mm256_set1_epi16((short)((k) + 8)); \
    __m256i v_idx = _mm256_add_epi32(vk1, _mm256_load_si256(&shufmasks2[(m)&0xFF])); \
    __m256i v_idx_2 = _mm256_add_epi32(vk2, _mm256_load_si256(&shufmasks2[((m)>>8)&0xFF])); \
    _mm256_storeu_si256((__m256i *)(colidx_ptr + (cnt)), v_idx); \
    cnt = (unsigned short)((cnt) + _mm_popcnt_u32((m)&0xFF)); \
    _mm256_storeu_si256((__m256i *)(colidx_ptr + (cnt)), v_idx_2); \
    cnt = (unsigned short)((cnt) + _mm_popcnt_u32(((m)>>8)&0xFF)); \
  } \
}

#define EXPAND_BFLOAT16(v, vlo_final, vhi_final) \
  { \
  const __m512i vlo = _mm512_unpacklo_epi16(vzero, v); \
  const __m512i vhi = _mm512_unpackhi_epi16(vzero, v); \
  const __m512i permmask1 = _mm512_set_epi64(11, 10, 3, 2, 9, 8, 1, 0); \
  const __m512i permmask2 = _mm512_set_epi64(15, 14, 7, 6, 13, 12, 5, 4); \
  vlo_final = _mm512_castsi512_ps(_mm512_permutex2var_epi64(vlo, permmask1, vhi)); \
  vhi_final = _mm512_castsi512_ps(_mm512_permutex2var_epi64(vlo, permmask2, vhi)); \
  }

#define COMPRESS_BFLOAT16(vlo, vhi, v) \
  { \
  const __m512i permmask1 = _mm512_set_epi64(13, 12, 9, 8, 5, 4, 1, 0); \
  const __m512i permmask2 = _mm512_set_epi64(15, 14, 11, 10, 7, 6, 3, 2); \
  const __m512i va = _mm512_castps_si512(vlo), vb = _mm512_castps_si512(vhi); \
  const __m512i vtmp1 =  _mm512_permutex2var_epi64(va, permmask1, vb); \
  const __m512i vtmp2 =  _mm512_permutex2var_epi64(va, permmask2, vb); \
  const __m512i a = _mm512_srli_epi32(vtmp1, 16), b = _mm512_srli_epi32(vtmp2, 16); \
  v = _mm512_packus_epi32(a, b); \
  }

#endif

