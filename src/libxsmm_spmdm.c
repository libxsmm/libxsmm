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
/* Nadathur Satish (Intel Corp.)
******************************************************************************/

/* NOTE: This code currently ignores alpha input to the matrix multiply */
#include <libxsmm_spmdm.h>
#include <libxsmm_intrinsics_x86.h>
#include <libxsmm_malloc.h>

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

#if !defined(LIBXSMM_SPMDM_MALLOC_INTRINSIC) && !defined(LIBXSMM_INTRINSICS_NONE)
# define LIBXSMM_SPMDM_MALLOC_INTRINSIC
#endif
#if defined(LIBXSMM_SPMDM_MALLOC_INTRINSIC)
# define LIBXSMM_SPMDM_MALLOC(SIZE, ALIGNMENT) _mm_malloc(SIZE, ALIGNMENT)
# define LIBXSMM_SPMDM_FREE(BUFFER) _mm_free((void*)(BUFFER))
#else
# define LIBXSMM_SPMDM_MALLOC(SIZE, ALIGNMENT) libxsmm_aligned_malloc(SIZE, -(ALIGNMENT))
# define LIBXSMM_SPMDM_FREE(BUFFER) libxsmm_free(BUFFER)
#endif

#ifndef LIBXSMM_STATIC_TARGET_ARCH
#error "LIBXSMM_STATIC_TARGET_ARCH undefined"
#endif

#if (LIBXSMM_X86_AVX512_CORE <= LIBXSMM_STATIC_TARGET_ARCH)
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
    r0 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(t0),_mm512_castps_pd(t2)));\
    r1 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(t0),_mm512_castps_pd(t2)));\
    r2 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(t1),_mm512_castps_pd(t3)));\
    r3 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(t1),_mm512_castps_pd(t3)));\
    r4 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(t4),_mm512_castps_pd(t6)));\
    r5 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(t4),_mm512_castps_pd(t6)));\
    r6 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(t5),_mm512_castps_pd(t7)));\
    r7 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(t5),_mm512_castps_pd(t7)));\
    r8 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(t8),_mm512_castps_pd(ta)));\
    r9 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(t8),_mm512_castps_pd(ta)));\
    ra = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(t9),_mm512_castps_pd(tb)));\
    rb = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(t9),_mm512_castps_pd(tb)));\
    rc = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(tc),_mm512_castps_pd(te)));\
    rd = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(tc),_mm512_castps_pd(te)));\
    re = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(td),_mm512_castps_pd(tf)));\
    rf = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(td),_mm512_castps_pd(tf)));\
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
    __m512i vload_1 =  _mm512_inserti32x8(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i*)(ptr_A))), _mm256_loadu_si256((const __m256i*)(ptr_A + ldA)), 1); \
    EXPAND_BFLOAT16(vload_1, r0, r1); \
    __m512i vload_2 =  _mm512_inserti32x8(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i*)(ptr_A + 2*ldA))), _mm256_loadu_si256((const __m256i*)(ptr_A + 3*ldA)), 1); \
    EXPAND_BFLOAT16(vload_2, r2, r3); \
    __m512i vload_3 =  _mm512_inserti32x8(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i*)(ptr_A + 4*ldA))), _mm256_loadu_si256((const __m256i*)(ptr_A + 5*ldA)), 1); \
    EXPAND_BFLOAT16(vload_3, r4, r5); \
    __m512i vload_4 =  _mm512_inserti32x8(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i*)(ptr_A + 6*ldA))), _mm256_loadu_si256((const __m256i*)(ptr_A + 7*ldA)), 1); \
    EXPAND_BFLOAT16(vload_4, r6, r7); \
    __m512i vload_5 =  _mm512_inserti32x8(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i*)(ptr_A + 8*ldA))), _mm256_loadu_si256((const __m256i*)(ptr_A + 9*ldA)), 1); \
    EXPAND_BFLOAT16(vload_5, r8, r9); \
    __m512i vload_6 =  _mm512_inserti32x8(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i*)(ptr_A + 10*ldA))), _mm256_loadu_si256((const __m256i*)(ptr_A + 11*ldA)), 1); \
    EXPAND_BFLOAT16(vload_6, ra, rb); \
    __m512i vload_7 =  _mm512_inserti32x8(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i*)(ptr_A + 12*ldA))), _mm256_loadu_si256((const __m256i*)(ptr_A + 13*ldA)), 1); \
    EXPAND_BFLOAT16(vload_7, rc, rd); \
    __m512i vload_8 =  _mm512_inserti32x8(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i*)(ptr_A + 14*ldA))), _mm256_loadu_si256((const __m256i*)(ptr_A + 15*ldA)), 1); \
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
    r0 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(t0),_mm512_castps_pd(t2)));\
    r1 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(t0),_mm512_castps_pd(t2)));\
    r2 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(t1),_mm512_castps_pd(t3)));\
    r3 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(t1),_mm512_castps_pd(t3)));\
    r4 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(t4),_mm512_castps_pd(t6)));\
    r5 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(t4),_mm512_castps_pd(t6)));\
    r6 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(t5),_mm512_castps_pd(t7)));\
    r7 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(t5),_mm512_castps_pd(t7)));\
    r8 = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(t8),_mm512_castps_pd(ta)));\
    r9 = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(t8),_mm512_castps_pd(ta)));\
    ra = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(t9),_mm512_castps_pd(tb)));\
    rb = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(t9),_mm512_castps_pd(tb)));\
    rc = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(tc),_mm512_castps_pd(te)));\
    rd = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(tc),_mm512_castps_pd(te)));\
    re = _mm512_castpd_ps(_mm512_unpacklo_pd(_mm512_castps_pd(td),_mm512_castps_pd(tf)));\
    rf = _mm512_castpd_ps(_mm512_unpackhi_pd(_mm512_castps_pd(td),_mm512_castps_pd(tf)));\
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


LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE __m256i internal_spmdm_shufmasks_32[256];
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE __m256i internal_spmdm_shufmasks_16[256];

LIBXSMM_INLINE LIBXSMM_RETARGETABLE void _mm512_print(__m512 a, char * s)
{
  float *v=(float *)(&a);
  int i;
  printf("[%8s(%3lud)]: ", s, (size_t)16);
  for (i=0; i < 16; i++)
    printf("%4f ", v[i]);
  printf("\n");
}

LIBXSMM_INLINE LIBXSMM_RETARGETABLE void _mm512i_print(__m512i a, char * s)
{
  int *v=(int *)(&a);
  int i;
  printf("[%8s(%3lud)]: ", s, (size_t)16);
  for (i=0; i < 16; i++)
    printf("%4d ", v[i]);
  printf("\n");
}

LIBXSMM_INLINE LIBXSMM_RETARGETABLE void _mm512i_epi16_print(__m512i a, char * s)
{
  uint16_t *v=(uint16_t*)(&a);
  int i;
  printf("[%8s(%3lud)]: ", s, (size_t)32);
  for (i=0; i < 32; i++)
    printf("%4d ", v[i]);
  printf("\n");
}

LIBXSMM_INLINE LIBXSMM_RETARGETABLE void _mm256i_epi16_print(__m256i a, char * s)
{
  uint16_t *v=(uint16_t*)(&a);
  int i;
  printf("[%8s(%3lud)]: ", s, (size_t)16);
  for (i=0; i < 16; i++)
    printf("%4d ", v[i]);
  printf("\n");
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

#elif (LIBXSMM_X86_AVX2 <= LIBXSMM_STATIC_TARGET_ARCH)
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
         ymm9  = _mm256_unpacklo_ps(ymm15, ymm2);\
  __m256 ymm3  = _mm256_shuffle_ps(ymm6, ymm1, 0x4E);\
         ymm10 = _mm256_blend_ps(ymm6, ymm3, 0xCC);\
         ymm6  = _mm256_shuffle_ps(ymm0, ymm9, 0x4E);\
  __m256 ymm7  = _mm256_unpackhi_ps(ymm11, ymm12);\
         ymm11 = _mm256_blend_ps(ymm0, ymm6, 0xCC);\
         ymm12 = _mm256_blend_ps(ymm3, ymm1, 0xCC);\
         ymm3  = _mm256_permute2f128_ps(ymm10, ymm11, 0x20);\
         _mm256_storeu_ps(ptr_B, ymm3);\
  __m256 ymm5  = _mm256_unpackhi_ps(ymm13, ymm14);\
         ymm13 = _mm256_blend_ps(ymm6, ymm9, 0xCC);\
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
         _mm256_storeu_ps(ptr_B + 7*ldB, ymm4);\
  }

#define TRANSPOSE_SIMD_WIDTH_KERNEL_BFLOAT16(ptr_A, ldA, ptr_B, ldB) \
  { \
  __m256 ymm9, ymm10, ymm11, ymm12, ymm13, ymm14, ymm15, ymm2; \
  __m256i vload_1 =  _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_loadu_si128((const __m128i*)(ptr_A))), _mm_loadu_si128((const __m128i*)(ptr_A + ldA)), 1); \
  EXPAND_BFLOAT16(vload_1, ymm9, ymm10); \
  __m256i vload_2 =  _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_loadu_si128((const __m128i*)(ptr_A + 2*ldA))), _mm_loadu_si128((const __m128i*)(ptr_A + 3*ldA)), 1); \
  EXPAND_BFLOAT16(vload_2, ymm11, ymm12); \
  __m256i vload_3 =  _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_loadu_si128((const __m128i*)(ptr_A + 4*ldA))), _mm_loadu_si128((const __m128i*)(ptr_A + 5*ldA)), 1); \
  EXPAND_BFLOAT16(vload_3, ymm13, ymm14); \
  __m256i vload_4 =  _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_loadu_si128((const __m128i*)(ptr_A + 6*ldA))), _mm_loadu_si128((const __m128i*)(ptr_A + 7*ldA)), 1); \
  EXPAND_BFLOAT16(vload_4, ymm15, ymm2); \
  __m256 ymm6  = _mm256_unpacklo_ps(ymm9, ymm10);\
  __m256 ymm1  = _mm256_unpacklo_ps(ymm11, ymm12);\
  __m256 ymm8  = _mm256_unpackhi_ps(ymm9, ymm10);\
  __m256 ymm0  = _mm256_unpacklo_ps(ymm13, ymm14);\
         ymm9  = _mm256_unpacklo_ps(ymm15, ymm2);\
  __m256 ymm3  = _mm256_shuffle_ps(ymm6, ymm1, 0x4E);\
         ymm10 = _mm256_blend_ps(ymm6, ymm3, 0xCC);\
         ymm6  = _mm256_shuffle_ps(ymm0, ymm9, 0x4E);\
  __m256 ymm7  = _mm256_unpackhi_ps(ymm11, ymm12);\
         ymm11 = _mm256_blend_ps(ymm0, ymm6, 0xCC);\
         ymm12 = _mm256_blend_ps(ymm3, ymm1, 0xCC);\
         ymm3  = _mm256_permute2f128_ps(ymm10, ymm11, 0x20);\
         _mm256_storeu_ps(ptr_B, ymm3);\
  __m256 ymm5  = _mm256_unpackhi_ps(ymm13, ymm14);\
         ymm13 = _mm256_blend_ps(ymm6, ymm9, 0xCC);\
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
         _mm256_storeu_ps(ptr_B + 7*ldB, ymm4);\
  }


LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE __m256i internal_spmdm_shufmasks_32[256];
LIBXSMM_EXTERN_C LIBXSMM_RETARGETABLE __m256i internal_spmdm_shufmasks_16[256];

LIBXSMM_INLINE LIBXSMM_RETARGETABLE void _mm256_print(__m256 a, char * s)
{
  float *v=(float *)(&a);
  int i;
  printf("[%8s(%3lud)]: ", s, (size_t)8);
  for (i=0; i < 8; i++)
    printf("%4f ", v[i]);
  printf("\n");
}

LIBXSMM_INLINE LIBXSMM_RETARGETABLE void _mm256i_print(__m256i a, char * s)
{
  int *v=(int *)(&a);
  int i;
  printf("[%8s(%3lud)]: ", s, (size_t)8);
  for (i=0; i < 8; i++)
    printf("%4d ", v[i]);
  printf("\n");
}

LIBXSMM_INLINE LIBXSMM_RETARGETABLE void _mm256i_epi16_print(__m256i a, char * s)
{
  uint16_t *v=(uint16_t*)(&a);
  int i;
  printf("[%8s(%3lud)]: ", s, (size_t)16);
  for (i=0; i < 16; i++)
    printf("%4d ", v[i]);
  printf("\n");
}

#define COMPRESS_FP32(v, k, m, cnt) \
  { \
  unsigned int mask = _mm256_movemask_ps(m); \
  SIMDTYPE_INT32 vk = _MM_SET1_INT16((short)(k)); \
  __m256i perm_ctrl = _mm256_load_si256(&shufmasks[mask]); \
  __m256 v_packed = _mm256_permutevar8x32_ps(v, perm_ctrl); \
  __m256i v_idx = _mm256_add_epi32(vk, _mm256_load_si256(&shufmasks2[mask])); \
  _mm256_storeu_ps(values_ptr + (cnt), v_packed); \
  _mm256_storeu_si256((__m256i *)(colidx_ptr + (cnt)), v_idx); \
  cnt = (unsigned short)((cnt) + _mm_popcnt_u32(mask)); \
  }

#define EXPAND_BFLOAT16(v, vlo_final, vhi_final) \
  { \
  __m256i vlo = _mm256_unpacklo_epi16(vzero, v); \
  __m256i vhi = _mm256_unpackhi_epi16(vzero, v); \
  vlo_final = _mm256_castsi256_ps(_mm256_permute2f128_si256(vlo, vhi, 0x20)); \
  vhi_final = _mm256_castsi256_ps(_mm256_permute2f128_si256(vlo, vhi, 0x31)); \
  }

#define COMPRESS_BFLOAT16(vlo, vhi, v) \
  { \
  const __m256i vtmp1 =  _mm256_castps_si256(_mm256_permute2f128_ps(vlo, vhi, 0x20)); \
  const __m256i vtmp2 =  _mm256_castps_si256(_mm256_permute2f128_ps(vlo, vhi, 0x31)); \
  const __m256i a = _mm256_srli_epi32(vtmp1, 16), b = _mm256_srli_epi32(vtmp2,16); \
  v = _mm256_packus_epi32(a, b); \
  }

#else
#define SIMD_WIDTH_FP32 (1)
#define SIMDTYPE_FP32 float
#define SIMDTYPE_INT32 int
#define SIMDMASKTYPE_FP32 int
#define _MM_SETZERO_FP32() (0)
#define _MM_SETZERO_INT32() (0)
#define _MM_SET1_FP32(x) (x)
#define _MM_SET1_INT32(x) (x)
#define _MM_SET1_INT16 (x)
#define _MM_LOAD_FP32(x) (*(x))
#define _MM_LOADU_FP32(x) (*(x))
#define _MM_LOAD_INT32(x) (*(x))
#define _MM_STORE_INT32(x,y) ((*(x)) = (y))
#define _MM_LOADU_INT32(x) (*(x))
#define _MM_GATHER_FP32(Addr, idx, scale) (*(Addr + (idx)))
#define _MM_CMPNEQ_FP32(v1,v2) (LIBXSMM_FEQ(v1, v2) ? 0 : 1)
#define _MM_STORE_FP32(x,y) ((*(x)) = (y))
#define _MM_STOREU_FP32(x,y) ((*(x)) = (y))
#define _MM_ADD_FP32(x,y) ((x) + (y))
#define _MM_FMADD_FP32(x,y,z) (((x)*(y))+(z))
#define _MM_MUL_FP32(x,y) ((x)*(y))
#define _MM_PREFETCH(x, y)
#define TRANSPOSE_SIMD_WIDTH_KERNEL(ptr_A, ldA, ptr_B, ldB) ((*(ptr_B)) = (*(ptr_A)))
#define TRANSPOSE_SIMD_WIDTH_KERNEL_BFLOAT16(ptr_A, ldA, ptr_B, ldB) \
{\
            uint16_t restmp = (*(ptr_A));\
            union { int i; float f; } res;\
            res.i = restmp;\
            res.i <<= 16;\
            (*(ptr_B)) = res.f;\
}

#define COMPRESS_FP32(v, k, m, cnt) \
  { \
  if (m) \
  { \
    values_ptr[cnt] = v; \
    colidx_ptr[cnt] = (uint16_t)(k); \
    cnt++; \
  } \
  }

#define EXPAND_BFLOAT16(v, vlo_final, vhi_final) \
  { \
    union { int i; float f; } vlo_tmp, vhi_tmp; \
    vlo_tmp.i = (v) & 0xFFFF; vlo_tmp.i <<= 16; \
    vlo_final = vlo_tmp.f; \
    vhi_tmp.i = (v) & 0x0000FFFF; \
    vhi_final = vhi_tmp.f; \
  }

#define COMPRESS_BFLOAT16(vlo, vhi, v) \
  { \
    union { int i; float f; } vlo_tmp, vhi_tmp; \
    vlo_tmp.f = vlo; \
    v = (vlo_tmp.i >> 16); \
    vhi_tmp.f = vhi; \
    v = v | (vhi_tmp.i & 0xFFFF0000); \
  }

#endif


LIBXSMM_INLINE LIBXSMM_RETARGETABLE void internal_spmdm_init_shufmask()
{
#if SIMD_WIDTH_FP32 != 1
  unsigned int i,j, c, last_bit;
  LIBXSMM_ALIGNED(int temp_shufmasks[8], 64);
  LIBXSMM_ALIGNED(uint16_t temp_shufmasks2[16], 64);
  int cnt;
  for (i = 0; i < 256; i++) {
    cnt = 0;
    j = i;
    for (c = 0; c < 8; c++) temp_shufmasks[c] = 0;
    for (c = 0; c < 16; c++) temp_shufmasks2[c] = 0;
    while ( j ) {
      last_bit = LIBXSMM_INTRINSICS_BITSCANFWD(j);
      temp_shufmasks[cnt] = last_bit;
      temp_shufmasks2[cnt] = (uint16_t)last_bit;
      j &= (~(1<<last_bit));
      cnt++;
    }
    internal_spmdm_shufmasks_32[i] = _mm256_loadu_si256((const __m256i*)temp_shufmasks);
    internal_spmdm_shufmasks_16[i] = _mm256_loadu_si256((const __m256i*)temp_shufmasks2);
  }
#endif
}


LIBXSMM_INLINE LIBXSMM_RETARGETABLE void internal_spmdm_allocate_csr_a( libxsmm_spmdm_handle* handle, libxsmm_CSR_sparseslice ** libxsmm_output_csr)
{
  int kb, mb;
  int m_blocks = handle->mb;
  int k_blocks = handle->kb;

  size_t sz_block = ((handle->bm + 1)*sizeof(uint16_t) + (handle->bm)*(handle->bk)*sizeof(uint16_t) + (handle->bm)*(handle->bk)*sizeof(float) + sizeof(libxsmm_CSR_sparseslice));
  size_t sz_all_blocks = sz_block * handle->mb * handle->kb;

  char * memory_block = (char *)LIBXSMM_SPMDM_MALLOC( sz_all_blocks, 2097152);
  char * memory_head  = memory_block;

  libxsmm_CSR_sparseslice* libxsmm_output_csr_a = (libxsmm_CSR_sparseslice*)(memory_head);
  memory_head += handle->mb * handle->kb * sizeof(libxsmm_CSR_sparseslice);

  for ( kb = 0; kb < k_blocks; kb++ ) {
    for ( mb = 0; mb < m_blocks; mb++ ) {
      int i = kb*m_blocks + mb;
      libxsmm_output_csr_a[i].rowidx = (uint16_t *)(memory_head);
      memory_head += (handle->bm + 1)*sizeof(uint16_t);
      libxsmm_output_csr_a[i].colidx = (uint16_t *)(memory_head);
      memory_head += (handle->bm)*(handle->bk)*sizeof(uint16_t);
      libxsmm_output_csr_a[i].values = (float*)(memory_head);
      memory_head += (handle->bm)*(handle->bk)*sizeof(float);
    }
  }
  assert(memory_head == (memory_block + sz_all_blocks));
  *libxsmm_output_csr = libxsmm_output_csr_a;
  handle->base_ptr_scratch_A = memory_block;
}


LIBXSMM_INLINE LIBXSMM_RETARGETABLE void internal_spmdm_allocate_scratch( libxsmm_spmdm_handle* handle, int max_threads)
{
  size_t sz_memory_for_scratch_per_thread = ((handle->bm)*(handle->bn)*sizeof(float) + (handle->bk)*(handle->bn)*sizeof(float))*max_threads, sz_total_memory;
  sz_memory_for_scratch_per_thread = (sz_memory_for_scratch_per_thread + 4095)/4096 * 4096;
  sz_total_memory = sz_memory_for_scratch_per_thread * max_threads;

  handle->base_ptr_scratch_B_scratch_C = (char *)LIBXSMM_SPMDM_MALLOC( sz_total_memory, 2097152 );
  handle->memory_for_scratch_per_thread = (int)sz_memory_for_scratch_per_thread;
}

LIBXSMM_API_DEFINITION void libxsmm_spmdm_init(int M, int N, int K, int max_threads, libxsmm_spmdm_handle * handle, libxsmm_CSR_sparseslice ** libxsmm_output_csr)
{
  handle->m  = M;
  handle->n  = N;
  handle->k  = K;

  if (M >= 4096 || M <= 1024) 
    handle->bm = 512;
  else 
    handle->bm = 256;
#if (LIBXSMM_X86_AVX512_CORE <= LIBXSMM_STATIC_TARGET_ARCH)
  handle->bn = 96;
#elif (LIBXSMM_X86_AVX2 <= LIBXSMM_STATIC_TARGET_ARCH)
  handle->bn = 48;
#else
  handle->bn = 6;
#endif
  handle->bk = 128;
  handle->mb = (handle->m + handle->bm - 1) / handle->bm;
  handle->nb = (handle->n + handle->bn - 1) / handle->bn;
  handle->kb = (handle->k + handle->bk - 1) / handle->bk;

  /* This is temporary space needed; allocate for each different size of A */
  internal_spmdm_allocate_csr_a( handle, libxsmm_output_csr);
  internal_spmdm_allocate_scratch( handle, max_threads);

  /* Initialize shuffle masks for the computation */
  internal_spmdm_init_shufmask();
}

LIBXSMM_INLINE LIBXSMM_RETARGETABLE void internal_spmdm_deallocate_csr_a(libxsmm_spmdm_handle* handle)
{
  LIBXSMM_SPMDM_FREE(handle->base_ptr_scratch_A);
  handle->base_ptr_scratch_A= NULL;
  LIBXSMM_SPMDM_FREE(handle->base_ptr_scratch_B_scratch_C);
  handle->base_ptr_scratch_B_scratch_C = NULL;
}

LIBXSMM_API_DEFINITION void libxsmm_spmdm_destroy(libxsmm_spmdm_handle * handle)
{
  internal_spmdm_deallocate_csr_a(handle);
}

LIBXSMM_API_DEFINITION int libxsmm_spmdm_get_num_createSparseSlice_blocks(const libxsmm_spmdm_handle* handle)
{
  return (handle->mb * handle->kb);
}

LIBXSMM_API_DEFINITION int libxsmm_spmdm_get_num_compute_blocks(const libxsmm_spmdm_handle* handle)
{
  return (handle->mb * handle->nb);
}

/* This converts a dense representation of the sparse matrix to 2D array of sparse slices. */
LIBXSMM_API_DEFINITION void libxsmm_spmdm_createSparseSlice_fp32_thread(
  const libxsmm_spmdm_handle* handle,
  char transA,
  const float * A,
  libxsmm_CSR_sparseslice* libxsmm_output_csr_a,
  int block_id,
  int tid, int nthreads)
{
   int i,k;
   int mb, kb;
#if SIMD_WIDTH_FP32 == 8
   __m256i * shufmasks = internal_spmdm_shufmasks_32;
#endif

#if SIMD_WIDTH_FP32 > 1
   __m256i * shufmasks2 = internal_spmdm_shufmasks_16;
#endif
   int block_offset_base, block_offset;
   int index[16];
   SIMDTYPE_INT32 vindex;

   LIBXSMM_UNUSED(nthreads);
   LIBXSMM_UNUSED(tid);

   kb = block_id / handle->mb;
   mb = block_id % handle->mb;
   if (transA == 'Y')
   {
     int kk;
     block_offset_base = mb * handle->bm;
     block_offset = block_offset_base + kb * handle->m * handle->bk;
     for (kk = 0; kk < SIMD_WIDTH_FP32; kk++) index[kk] = kk*handle->m;
     vindex = _MM_LOADU_INT32(index);
   }
   else
   {
     block_offset_base = kb * handle->bk;
     block_offset = block_offset_base + mb * handle->k * handle->bm;
   }
   {
     libxsmm_CSR_sparseslice slice = libxsmm_output_csr_a[kb*handle->mb + mb];
     int nrows = ((mb + 1)*handle->bm > handle->m)?(handle->m - (mb)*handle->bm):handle->bm;
     int ncols = ((kb + 1)*handle->bk > handle->k)?(handle->k - (kb)*handle->bk):handle->bk;
     /*printf("nrows: %d, ncols: %d\n", nrows, ncols);*/
     int ncols_aligned = ncols / (4*SIMD_WIDTH_FP32)*(4*SIMD_WIDTH_FP32);
     int ncols_aligned_2 = ncols / (SIMD_WIDTH_FP32)*(SIMD_WIDTH_FP32);
     const float * input_ptr = A + block_offset;
     uint16_t * rowidx_ptr = slice.rowidx;
     uint16_t * colidx_ptr = slice.colidx;
     float    * values_ptr = (float *)(slice.values);
     SIMDTYPE_FP32 vzero = _MM_SET1_FP32(0.0);
     uint16_t cnt = 0;
     if (SIMD_WIDTH_FP32 == 1) { ncols_aligned = 0; ncols_aligned_2 = 0; }
     for (i = 0; i < nrows; i++) {
       rowidx_ptr[i] = cnt;
       if (transA == 'Y')
       {
         for (k = 0; k < ncols_aligned; k+= 4*SIMD_WIDTH_FP32) {
           SIMDTYPE_FP32 v1 = _MM_GATHER_FP32(input_ptr + k*handle->m + i, vindex, 4);
           SIMDTYPE_FP32 v2 = _MM_GATHER_FP32(input_ptr + (k+SIMD_WIDTH_FP32)*handle->m + i, vindex, 4);
           SIMDTYPE_FP32 v3 = _MM_GATHER_FP32(input_ptr + (k+2*SIMD_WIDTH_FP32)*handle->m + i, vindex, 4);
           SIMDTYPE_FP32 v4 = _MM_GATHER_FP32(input_ptr + (k+3*SIMD_WIDTH_FP32)*handle->m + i, vindex, 4);
           SIMDMASKTYPE_FP32 m1 = _MM_CMPNEQ_FP32(v1, vzero);
           SIMDMASKTYPE_FP32 m2 = _MM_CMPNEQ_FP32(v2, vzero);
           SIMDMASKTYPE_FP32 m3 = _MM_CMPNEQ_FP32(v3, vzero);
           SIMDMASKTYPE_FP32 m4 = _MM_CMPNEQ_FP32(v4, vzero);
           COMPRESS_FP32(v1, k, m1, cnt);
           COMPRESS_FP32(v2, k + SIMD_WIDTH_FP32, m2, cnt);
           COMPRESS_FP32(v3, k + 2*SIMD_WIDTH_FP32, m3, cnt);
           COMPRESS_FP32(v4, k + 3*SIMD_WIDTH_FP32, m4, cnt);
         }
         for (k = ncols_aligned; k < ncols_aligned_2; k+= SIMD_WIDTH_FP32) {
           SIMDTYPE_FP32 v1 = _MM_GATHER_FP32(input_ptr + k*handle->m + i, vindex, 4);
           SIMDMASKTYPE_FP32 m1 = _MM_CMPNEQ_FP32(v1, vzero);
           COMPRESS_FP32(v1, k, m1, cnt);
         }

         for (k = ncols_aligned_2; k < ncols; k++) {
           const float v1 = input_ptr[i + k*handle->m];
           const int m1 = LIBXSMM_FEQ(0, v1) ? 0 : 1;
           if (m1) { colidx_ptr[cnt] = (uint16_t)k; values_ptr[cnt] = v1; cnt++; }
         }
       }
       else
       {
         for (k = 0; k < ncols_aligned; k+= 4*SIMD_WIDTH_FP32) {
           SIMDTYPE_FP32 v1, v2, v3, v4;
           SIMDMASKTYPE_FP32 m1, m2, m3, m4;
           v1 = _MM_LOADU_FP32(input_ptr + i*handle->k + k);
           _MM_PREFETCH((char *)(input_ptr + (i+2)*handle->k + k), _MM_HINT_T0);
           v2 = _MM_LOADU_FP32(input_ptr + i*handle->k + k + SIMD_WIDTH_FP32);
           _MM_PREFETCH((char *)(input_ptr + (i+2)*handle->k + k + SIMD_WIDTH_FP32), _MM_HINT_T0);
           v3 = _MM_LOADU_FP32(input_ptr + i*handle->k + k + 2*SIMD_WIDTH_FP32);
           _MM_PREFETCH((char *)(input_ptr + (i+2)*handle->k + k + 2*SIMD_WIDTH_FP32), _MM_HINT_T0);
           v4 = _MM_LOADU_FP32(input_ptr + i*handle->k + k + 3*SIMD_WIDTH_FP32);
           _MM_PREFETCH((char *)(input_ptr + (i+2)*handle->k + k + 3*SIMD_WIDTH_FP32), _MM_HINT_T0);
           m1 = _MM_CMPNEQ_FP32(v1, vzero);
           m2 = _MM_CMPNEQ_FP32(v2, vzero);
           m3 = _MM_CMPNEQ_FP32(v3, vzero);
           m4 = _MM_CMPNEQ_FP32(v4, vzero);
           COMPRESS_FP32(v1, k, m1, cnt);
           COMPRESS_FP32(v2, k + SIMD_WIDTH_FP32, m2, cnt);
           COMPRESS_FP32(v3, k + 2*SIMD_WIDTH_FP32, m3, cnt);
           COMPRESS_FP32(v4, k + 3*SIMD_WIDTH_FP32, m4, cnt);
         }
         for (k = ncols_aligned; k < ncols_aligned_2; k+= SIMD_WIDTH_FP32) {
           SIMDTYPE_FP32 v1;
           SIMDMASKTYPE_FP32 m1;
           v1 = _MM_LOADU_FP32(input_ptr + i*handle->k + k);
           _MM_PREFETCH((char *)(input_ptr + (i+2)*handle->k + k), _MM_HINT_T0);
           m1 = _MM_CMPNEQ_FP32(v1, vzero);
           COMPRESS_FP32(v1, k, m1, cnt);
         }
         for (k = ncols_aligned_2; k < ncols; k++) {
           const float v1 = input_ptr[i*handle->k + k];
           const int m1 = LIBXSMM_FEQ(0, v1) ? 0 : 1;
           if (m1) { colidx_ptr[cnt] = (uint16_t)k; values_ptr[cnt] = v1; cnt++; }
         }
       }
     }
     rowidx_ptr[nrows] = cnt;
#if 0
     printf("cnt: %d\n", cnt);
     for (i = 0; i <= nrows; i++) {
       int j;
       for (j = slice.rowidx[i]; j < slice.rowidx[i+1]; j++) {
         printf("(%d, %d): %f ", i, colidx_ptr[j], values_ptr[j]);
       }
     }
#endif
   }
}


LIBXSMM_API_DEFINITION void libxsmm_spmdm_createSparseSlice_bfloat16_thread(
  const libxsmm_spmdm_handle* handle,
  char transA,
  const uint16_t * A,
  libxsmm_CSR_sparseslice* libxsmm_output_csr_a,
  int block_id,
  int tid, int nthreads)
{
   int i,k;
   int mb, kb;
#if SIMD_WIDTH_FP32 == 8
   __m256i * shufmasks = internal_spmdm_shufmasks_32;
#endif
#if SIMD_WIDTH_FP32 > 1
   __m256i * shufmasks2 = internal_spmdm_shufmasks_16;
#endif
   int block_offset_base, block_offset;

   LIBXSMM_UNUSED(nthreads);
   LIBXSMM_UNUSED(tid);

   kb = block_id / handle->mb;
   mb = block_id % handle->mb;

   if (transA == 'Y')
   {
     block_offset_base = mb * handle->bm;
     block_offset = block_offset_base + kb * handle->m * handle->bk;
   }
   else
   {
     block_offset_base = kb * handle->bk;
     block_offset = block_offset_base + mb * handle->k * handle->bm;
   }
   {
     libxsmm_CSR_sparseslice slice = libxsmm_output_csr_a[kb*handle->mb + mb];
     int nrows = ((mb + 1)*handle->bm > handle->m)?(handle->m - (mb)*handle->bm):handle->bm;
     int ncols = ((kb + 1)*handle->bk > handle->k)?(handle->k - (kb)*handle->bk):handle->bk;
     /*printf("nrows: %d, ncols: %d\n", nrows, ncols);*/
     int ncols_aligned = ncols / (4*SIMD_WIDTH_FP32)*(4*SIMD_WIDTH_FP32);
     const uint16_t * input_ptr = A + block_offset;
     uint16_t * rowidx_ptr = slice.rowidx;
     uint16_t * colidx_ptr = slice.colidx;
     float * values_ptr = (float *)(slice.values);
#if SIMD_WIDTH_FP32 > 1
     SIMDTYPE_INT32 vzero = _MM_SET1_INT32(0);
#endif
     SIMDTYPE_FP32 vzerof = _MM_SET1_FP32(0.0);
     uint16_t cnt = 0;
     if (SIMD_WIDTH_FP32 == 1) { ncols_aligned = 0; }
     for (i = 0; i < nrows; i++) {
       rowidx_ptr[i] = cnt;
       if (transA == 'Y')
       {
         for (k = 0; k < ncols_aligned; k+= 4*SIMD_WIDTH_FP32) {
           int vals[32];
           int kk;
           for (kk = 0; kk < 4*SIMD_WIDTH_FP32; kk+=2) { vals[kk/2] = (int)input_ptr[(k+kk)*handle->m + i]; vals[kk/2] |= ((int)(input_ptr[(k+kk+1)*handle->m + i]) << 16); }
           {
             SIMDTYPE_INT32 v1tmp = _MM_LOADU_INT32(vals);
             SIMDTYPE_INT32 v2tmp = _MM_LOADU_INT32(vals + SIMD_WIDTH_FP32);
             SIMDTYPE_FP32 v1, v2, v3, v4;
             SIMDMASKTYPE_FP32 m1, m2, m3, m4;
             EXPAND_BFLOAT16(v1tmp, v1, v2);
             EXPAND_BFLOAT16(v2tmp, v3, v4);
             m1 = _MM_CMPNEQ_FP32(v1, vzerof);
             m2 = _MM_CMPNEQ_FP32(v2, vzerof);
             m3 = _MM_CMPNEQ_FP32(v3, vzerof);
             m4 = _MM_CMPNEQ_FP32(v4, vzerof);
             COMPRESS_FP32(v1, k, m1, cnt);
             COMPRESS_FP32(v2, k + SIMD_WIDTH_FP32, m2, cnt);
             COMPRESS_FP32(v3, k + 2*SIMD_WIDTH_FP32, m3, cnt);
             COMPRESS_FP32(v4, k + 3*SIMD_WIDTH_FP32, m4, cnt);
           }
         }

         for (k = ncols_aligned; k < ncols; k++) {
           uint16_t v1tmp = input_ptr[k*handle->m + i];
           union {int i; float f; } v1tmp_int;
           v1tmp_int.i = v1tmp;
           v1tmp_int.i <<= 16;
           {
             const int m1 = LIBXSMM_FEQ(0, v1tmp_int.f) ? 0 : 1;
             if (m1) { colidx_ptr[cnt] = (uint16_t)k; values_ptr[cnt] = v1tmp_int.f; cnt++; }
           }
         }
       }
       else
       {
         for (k = 0; k < ncols_aligned; k+= 4*SIMD_WIDTH_FP32) {
           SIMDTYPE_INT32 v1tmp, v2tmp;
           SIMDTYPE_FP32 v1, v2, v3, v4;
           SIMDMASKTYPE_FP32 m1, m2, m3, m4;
           v1tmp = _MM_LOADU_INT32((const SIMDTYPE_INT32* )(input_ptr + i*handle->k + k));
           _MM_PREFETCH((char *)(input_ptr + (i+2)*handle->k + k), _MM_HINT_T0);
           v2tmp = _MM_LOADU_INT32((const SIMDTYPE_INT32*)(input_ptr + i*handle->k + k + 2*SIMD_WIDTH_FP32));
           _MM_PREFETCH((char *)(input_ptr + (i+2)*handle->k + k + SIMD_WIDTH_FP32), _MM_HINT_T0);
           EXPAND_BFLOAT16(v1tmp, v1, v2);
           EXPAND_BFLOAT16(v2tmp, v3, v4);
           m1 = _MM_CMPNEQ_FP32(v1, vzerof);
           m2 = _MM_CMPNEQ_FP32(v2, vzerof);
           m3 = _MM_CMPNEQ_FP32(v3, vzerof);
           m4 = _MM_CMPNEQ_FP32(v4, vzerof);
           COMPRESS_FP32(v1, k, m1, cnt);
           COMPRESS_FP32(v2, k + SIMD_WIDTH_FP32, m2, cnt);
           COMPRESS_FP32(v3, k + 2*SIMD_WIDTH_FP32, m3, cnt);
           COMPRESS_FP32(v4, k + 3*SIMD_WIDTH_FP32, m4, cnt);
         }
         for (k = ncols_aligned; k < ncols; k++) {
           uint16_t v1tmp = input_ptr[i*handle->k + k];
           union {int i; float f; } v1tmp_int;
           v1tmp_int.i = v1tmp;
           v1tmp_int.i <<= 16;
           {
             int m1 = LIBXSMM_FEQ(0, v1tmp_int.f) ? 0 : 1;
             if (m1) { colidx_ptr[cnt] = (uint16_t)k; values_ptr[cnt] = v1tmp_int.f; cnt++; }
           }
         }
       }
     }
     rowidx_ptr[nrows] = cnt;
#if 0
     printf("cnt: %d\n", cnt);
     for (i = 0; i <= nrows; i++) {
       for (j = slice.rowidx[i]; j < slice.rowidx[i+1]; j++) {
         printf("(%d, %d): %f ", i, colidx_ptr[j], values_ptr[j]);
       }
     }
#endif
   }
}


LIBXSMM_API_DEFINITION void libxsmm_spmdm_compute_fp32_thread(
  const libxsmm_spmdm_handle* handle,
  char transA,
  char transB,
  const float *alpha,
  libxsmm_CSR_sparseslice* A_sparse,
  const float *B,
  char transC,
  const float *beta,
  float* C,
  int block_id,
  int tid, int nthreads)
{
  const int m_blocks = handle->mb;
  /* const int n_blocks = handle->nb; */
  const int k_blocks = handle->kb;
  const int m_block_size = handle->bm;
  const int n_block_size = handle->bn;
  const int k_block_size = handle->bk;
  int mb = block_id / handle->nb;
  int nb = block_id % handle->nb;

#define num_regs (6)
  int m_overall_start = mb*m_block_size;
  int m_overall_end   = (mb + 1)*m_block_size;
  int num_m;
  int num_m_aligned;

  int n_overall_start = nb*n_block_size;
  int n_overall_end   = (nb + 1)*n_block_size;
  int num_n;
  int m, n, k, kb;
  int last_block_n, num_full_regs, last_n_start;

  int k_overall_start, k_overall_end, num_k;

  float *const scratch_C = (float *)(handle->base_ptr_scratch_B_scratch_C + tid*handle->memory_for_scratch_per_thread);
  float *const scratch_B = (float *)(handle->base_ptr_scratch_B_scratch_C + tid*handle->memory_for_scratch_per_thread + m_block_size*n_block_size*sizeof(float));
  SIMDTYPE_FP32 sum[2*num_regs];
  float* LIBXSMM_RESTRICT ptr_result;

  LIBXSMM_UNUSED(nthreads);
  LIBXSMM_UNUSED(transA);
  LIBXSMM_UNUSED(alpha);
  LIBXSMM_UNUSED(beta);
  LIBXSMM_UNUSED(tid);

  /* really is twice this */
  assert(n_block_size == num_regs*SIMD_WIDTH_FP32);

  if (m_overall_end   > handle->m) m_overall_end   = handle->m;
  num_m = (m_overall_end - m_overall_start);
  num_m_aligned = (num_m / 2) * 2;

  if (n_overall_end   > handle->n) n_overall_end   = handle->n;
  num_n = (n_overall_end - n_overall_start);
  last_block_n = (num_n != n_block_size);
  num_full_regs = (num_n / SIMD_WIDTH_FP32);
  if ((num_full_regs > 0) && (num_full_regs%2)) num_full_regs--;
  last_n_start = num_full_regs*SIMD_WIDTH_FP32;

#if 0
  printf("Block: m_overall_start: %d, m_overall_end: %d, num_m: %d, num_m_aligned: %d\n", m_overall_start, m_overall_end, num_m, num_m_aligned);
  printf("Block: n_overall_start: %d, n_overall_end: %d, num_n: %d, last_block_n: %d\n", n_overall_start, n_overall_end, num_n, last_block_n);
  printf("Block: k_blocks: %d\n", k_blocks);
#endif
  /* Copy in C matrix to buffer*/
  ptr_result = C + m_overall_start*handle->n + n_overall_start;
  if (LIBXSMM_FEQ(0.f, *beta)) {
    if (!last_block_n) {
      for (m = 0; m < num_m; m++) {
        _MM_STORE_FP32(scratch_C + m*num_regs*SIMD_WIDTH_FP32 + 0*SIMD_WIDTH_FP32, _MM_SETZERO_FP32());
        _MM_STORE_FP32(scratch_C + m*num_regs*SIMD_WIDTH_FP32 + 1*SIMD_WIDTH_FP32, _MM_SETZERO_FP32());
        _MM_STORE_FP32(scratch_C + m*num_regs*SIMD_WIDTH_FP32 + 2*SIMD_WIDTH_FP32, _MM_SETZERO_FP32());
        _MM_STORE_FP32(scratch_C + m*num_regs*SIMD_WIDTH_FP32 + 3*SIMD_WIDTH_FP32, _MM_SETZERO_FP32());
        _MM_STORE_FP32(scratch_C + m*num_regs*SIMD_WIDTH_FP32 + 4*SIMD_WIDTH_FP32, _MM_SETZERO_FP32());
        _MM_STORE_FP32(scratch_C + m*num_regs*SIMD_WIDTH_FP32 + 5*SIMD_WIDTH_FP32, _MM_SETZERO_FP32());
      }
    } else {
      for (m = 0; m < num_m; m++) {
        for (n = 0; n < num_full_regs; n+=2) {
          _MM_STORE_FP32(scratch_C + m*num_regs*SIMD_WIDTH_FP32 + n*SIMD_WIDTH_FP32, _MM_SETZERO_FP32());
          _MM_STORE_FP32(scratch_C + m*num_regs*SIMD_WIDTH_FP32 + (n+1)*SIMD_WIDTH_FP32, _MM_SETZERO_FP32());
        }
        for (n = last_n_start; n < num_n; n++) {
          scratch_C[m*num_regs*SIMD_WIDTH_FP32 + n] = 0;
        }
      }
    }
  }
  else if (LIBXSMM_FEQ(1.f, *beta)) {
    if(transC == 'Y') {
      int num_m_simd = num_m / SIMD_WIDTH_FP32 * SIMD_WIDTH_FP32;
      int num_n_simd = num_n / SIMD_WIDTH_FP32 * SIMD_WIDTH_FP32;
      int m2;

      ptr_result = C + n_overall_start*handle->m + m_overall_start;
 
      for(m = 0; m < num_m_simd; m+=SIMD_WIDTH_FP32){
        for(n = 0; n < num_n_simd; n+=SIMD_WIDTH_FP32){
          //for(int m2 = m; m2 < m + SIMD_WIDTH_FP32; m2++) for( int n2 = n; n2 < n + SIMD_WIDTH_FP32; n2++) scratch_C[m2*N + n2] = ptr_result[n2*M + m2];
          TRANSPOSE_SIMD_WIDTH_KERNEL(ptr_result + n*handle->m + m, handle->m, scratch_C + m*n_block_size + n, n_block_size);
        }
        /* Transpose a SIMD_WIDTH_FP32 * (num_n - num_n_simd) block of output space - input is of size (num_n - num_n_simd) * SIMD_WIDTH_FP32 */
        for(m2 = m; m2 < m + SIMD_WIDTH_FP32; m2++) {
          for(n = num_n_simd; n < num_n; n++){
            scratch_C[m2*n_block_size + n] = ptr_result[n*handle->m + m2];
          }
        }
      }
      /* Transpose a (num_m - num_m_simd) * num_n block of output space - input is of size num_n * (num_m - num_m_simd) */
      for(m = num_m_simd; m < num_m; m++){
        for(n = 0; n < num_n; n++){
          scratch_C[m*n_block_size + n] = ptr_result[n*handle->m + m];
        }
      }
    }
    else {
      if (!last_block_n) {
        for (m = 0; m < num_m; m++) {
          _MM_STORE_FP32(scratch_C + m*num_regs*SIMD_WIDTH_FP32 + 0*SIMD_WIDTH_FP32, _MM_LOADU_FP32(ptr_result + m*handle->n + 0*SIMD_WIDTH_FP32));
          _MM_STORE_FP32(scratch_C + m*num_regs*SIMD_WIDTH_FP32 + 1*SIMD_WIDTH_FP32, _MM_LOADU_FP32(ptr_result + m*handle->n + 1*SIMD_WIDTH_FP32));
          _MM_STORE_FP32(scratch_C + m*num_regs*SIMD_WIDTH_FP32 + 2*SIMD_WIDTH_FP32, _MM_LOADU_FP32(ptr_result + m*handle->n + 2*SIMD_WIDTH_FP32));
          _MM_STORE_FP32(scratch_C + m*num_regs*SIMD_WIDTH_FP32 + 3*SIMD_WIDTH_FP32, _MM_LOADU_FP32(ptr_result + m*handle->n + 3*SIMD_WIDTH_FP32));
          _MM_STORE_FP32(scratch_C + m*num_regs*SIMD_WIDTH_FP32 + 4*SIMD_WIDTH_FP32, _MM_LOADU_FP32(ptr_result + m*handle->n + 4*SIMD_WIDTH_FP32));
          _MM_STORE_FP32(scratch_C + m*num_regs*SIMD_WIDTH_FP32 + 5*SIMD_WIDTH_FP32, _MM_LOADU_FP32(ptr_result + m*handle->n + 5*SIMD_WIDTH_FP32));
        }
      }
      else {
        for (m = 0; m < num_m; m++) {
          for (n = 0; n < num_full_regs; n+=2) {
            _MM_STORE_FP32(scratch_C + m*num_regs*SIMD_WIDTH_FP32 + n*SIMD_WIDTH_FP32, _MM_LOADU_FP32(ptr_result + m*handle->n + n*SIMD_WIDTH_FP32));
            _MM_STORE_FP32(scratch_C + m*num_regs*SIMD_WIDTH_FP32 + (n+1)*SIMD_WIDTH_FP32, _MM_LOADU_FP32(ptr_result + m*handle->n + (n+1)*SIMD_WIDTH_FP32));
          }
          for (n = last_n_start; n < num_n; n++) {
            scratch_C[m*num_regs*SIMD_WIDTH_FP32 + n] = ptr_result[m*handle->n + n];
          }
        }
      }
    }
  }
  else {
    SIMDTYPE_FP32 beta_v = _MM_SET1_FP32(*beta);
    if(transC == 'Y') {
      int num_m_simd = num_m / SIMD_WIDTH_FP32 * SIMD_WIDTH_FP32;
      int num_n_simd = num_n / SIMD_WIDTH_FP32 * SIMD_WIDTH_FP32;
      int m2;

      ptr_result = C + n_overall_start*handle->m + m_overall_start;
 
      for(m = 0; m < num_m_simd; m+=SIMD_WIDTH_FP32){
        for(n = 0; n < num_n_simd; n+=SIMD_WIDTH_FP32){
          //for(int m2 = m; m2 < m + SIMD_WIDTH_FP32; m2++) for( int n2 = n; n2 < n + SIMD_WIDTH_FP32; n2++) scratch_C[m2*N + n2] = ptr_result[n2*M + m2];
          TRANSPOSE_SIMD_WIDTH_KERNEL(ptr_result + n*handle->m + m, handle->m, scratch_C + m*n_block_size + n, n_block_size);
          _MM_STORE_FP32(scratch_C + m*n_block_size + n, _MM_MUL_FP32(beta_v, _MM_LOADU_FP32(scratch_C + m*n_block_size + n)));
          _MM_STORE_FP32(scratch_C + m*n_block_size + n + n_block_size, _MM_MUL_FP32(beta_v, _MM_LOADU_FP32(scratch_C + m*n_block_size + n + n_block_size)));
          _MM_STORE_FP32(scratch_C + m*n_block_size + n + 2*n_block_size, _MM_MUL_FP32(beta_v, _MM_LOADU_FP32(scratch_C + m*n_block_size + n + 2*n_block_size)));
          _MM_STORE_FP32(scratch_C + m*n_block_size + n + 3*n_block_size, _MM_MUL_FP32(beta_v, _MM_LOADU_FP32(scratch_C + m*n_block_size + n + 3*n_block_size)));
          _MM_STORE_FP32(scratch_C + m*n_block_size + n + 4*n_block_size, _MM_MUL_FP32(beta_v, _MM_LOADU_FP32(scratch_C + m*n_block_size + n + 4*n_block_size)));
          _MM_STORE_FP32(scratch_C + m*n_block_size + n + 5*n_block_size, _MM_MUL_FP32(beta_v, _MM_LOADU_FP32(scratch_C + m*n_block_size + n + 5*n_block_size)));
          _MM_STORE_FP32(scratch_C + m*n_block_size + n + 6*n_block_size, _MM_MUL_FP32(beta_v, _MM_LOADU_FP32(scratch_C + m*n_block_size + n + 6*n_block_size)));
          _MM_STORE_FP32(scratch_C + m*n_block_size + n + 7*n_block_size, _MM_MUL_FP32(beta_v, _MM_LOADU_FP32(scratch_C + m*n_block_size + n + 7*n_block_size)));
        }
        /* Transpose a SIMD_WIDTH_FP32 * (num_n - num_n_simd) block of output space - input is of size (num_n - num_n_simd) * SIMD_WIDTH_FP32 */
        for(m2 = m; m2 < m + SIMD_WIDTH_FP32; m2++) {
          for(n = num_n_simd; n < num_n; n++){
            scratch_C[m2*n_block_size + n] = (*beta)*ptr_result[n*handle->m + m2];
          }
        }
      }
      /* Transpose a (num_m - num_m_simd) * num_n block of output space - input is of size num_n * (num_m - num_m_simd) */
      for(m = num_m_simd; m < num_m; m++){
        for(n = 0; n < num_n; n++){
          scratch_C[m*n_block_size + n] = (*beta)*ptr_result[n*handle->m + m];
        }
      }

    }
    else {
      if (!last_block_n) {
        for (m = 0; m < num_m; m++) {
          _MM_STORE_FP32(scratch_C + m*num_regs*SIMD_WIDTH_FP32 + 0*SIMD_WIDTH_FP32, _MM_MUL_FP32(beta_v, _MM_LOADU_FP32(ptr_result + m*handle->n + 0*SIMD_WIDTH_FP32)));
          _MM_STORE_FP32(scratch_C + m*num_regs*SIMD_WIDTH_FP32 + 1*SIMD_WIDTH_FP32, _MM_MUL_FP32(beta_v, _MM_LOADU_FP32(ptr_result + m*handle->n + 1*SIMD_WIDTH_FP32)));
          _MM_STORE_FP32(scratch_C + m*num_regs*SIMD_WIDTH_FP32 + 2*SIMD_WIDTH_FP32, _MM_MUL_FP32(beta_v, _MM_LOADU_FP32(ptr_result + m*handle->n + 2*SIMD_WIDTH_FP32)));
          _MM_STORE_FP32(scratch_C + m*num_regs*SIMD_WIDTH_FP32 + 3*SIMD_WIDTH_FP32, _MM_MUL_FP32(beta_v, _MM_LOADU_FP32(ptr_result + m*handle->n + 3*SIMD_WIDTH_FP32)));
          _MM_STORE_FP32(scratch_C + m*num_regs*SIMD_WIDTH_FP32 + 4*SIMD_WIDTH_FP32, _MM_MUL_FP32(beta_v, _MM_LOADU_FP32(ptr_result + m*handle->n + 4*SIMD_WIDTH_FP32)));
          _MM_STORE_FP32(scratch_C + m*num_regs*SIMD_WIDTH_FP32 + 5*SIMD_WIDTH_FP32, _MM_MUL_FP32(beta_v, _MM_LOADU_FP32(ptr_result + m*handle->n + 5*SIMD_WIDTH_FP32)));
        }
      }
      else {
        for (m = 0; m < num_m; m++) {
          for (n = 0; n < num_full_regs; n+=2) {
            _MM_STORE_FP32(scratch_C + m*num_regs*SIMD_WIDTH_FP32 + n*SIMD_WIDTH_FP32, _MM_MUL_FP32(beta_v, _MM_LOADU_FP32(ptr_result + m*handle->n + n*SIMD_WIDTH_FP32)));
            _MM_STORE_FP32(scratch_C + m*num_regs*SIMD_WIDTH_FP32 + (n+1)*SIMD_WIDTH_FP32, _MM_MUL_FP32(beta_v, _MM_LOADU_FP32(ptr_result + m*handle->n + (n+1)*SIMD_WIDTH_FP32)));
          }
          for (n = last_n_start; n < num_n; n++) {
            scratch_C[m*num_regs*SIMD_WIDTH_FP32 + n] = (*beta)*ptr_result[m*handle->n + n];
          }
        }
      }
    }
  }
 
  for (kb = 0; kb < k_blocks; kb++) {
    const float * LIBXSMM_RESTRICT ptr_dense;
    float * LIBXSMM_RESTRICT scratch_C_base;
    const float * LIBXSMM_RESTRICT scratch_B_base;
    int block_A = kb * m_blocks + mb;
    libxsmm_CSR_sparseslice slice = A_sparse[block_A];
    int m_local = 0;

    k_overall_start = kb*k_block_size;
    k_overall_end   = (kb+1)*k_block_size;
    num_k = (k_overall_end - k_overall_start);

    /* Copy in B matrix*/
    if (transB == 'Y')
    {
      int num_k_simd = num_k / SIMD_WIDTH_FP32 * SIMD_WIDTH_FP32;
      int num_n_simd = num_n / SIMD_WIDTH_FP32 * SIMD_WIDTH_FP32;
      int k2;

      ptr_dense = B + n_overall_start*handle->k + k_overall_start;
 
      for(k = 0; k < num_k_simd; k+=SIMD_WIDTH_FP32){
        for(n = 0; n < num_n_simd; n+=SIMD_WIDTH_FP32){
          //for(int m2 = m; m2 < m + SIMD_WIDTH_FP32; m2++) for( int n2 = n; n2 < n + SIMD_WIDTH_FP32; n2++) ptr_B[m2*N + n2] = ptr_A[n2*M + m2];
          TRANSPOSE_SIMD_WIDTH_KERNEL(ptr_dense + n*handle->k + k, handle->k, scratch_B + k*n_block_size + n, n_block_size);
        }
        /* Transpose a SIMD_WIDTH_FP32 * (num_n - num_n_simd) block of output space - input is of size (num_n - num_n_simd) * SIMD_WIDTH_FP32 */
        for(k2 = k; k2 < k + SIMD_WIDTH_FP32; k2++) {
          for(n = num_n_simd; n < num_n; n++){
            scratch_B[k2*n_block_size + n] = ptr_dense[n*handle->k + k2];
          }
        }
      }
      /* Transpose a (num_m - num_m_simd) * num_n block of output space - input is of size num_n * (num_m - num_m_simd) */
      for(k = num_k_simd; k < num_k; k++){
        for(n = 0; n < num_n; n++){
          scratch_B[k*n_block_size + n] = ptr_dense[n*handle->k + k];
        }
      }
    }
    else
    {
      ptr_dense = B + k_overall_start*handle->n + n_overall_start;
      if (!last_block_n) {
        for (k = 0; k < num_k; k++) {
          _MM_STORE_FP32(scratch_B + k*num_regs*SIMD_WIDTH_FP32 + 0*SIMD_WIDTH_FP32, _MM_LOADU_FP32(ptr_dense + k*handle->n + 0*SIMD_WIDTH_FP32));
          _MM_STORE_FP32(scratch_B + k*num_regs*SIMD_WIDTH_FP32 + 1*SIMD_WIDTH_FP32, _MM_LOADU_FP32(ptr_dense + k*handle->n + 1*SIMD_WIDTH_FP32));
          _MM_STORE_FP32(scratch_B + k*num_regs*SIMD_WIDTH_FP32 + 2*SIMD_WIDTH_FP32, _MM_LOADU_FP32(ptr_dense + k*handle->n + 2*SIMD_WIDTH_FP32));
          _MM_STORE_FP32(scratch_B + k*num_regs*SIMD_WIDTH_FP32 + 3*SIMD_WIDTH_FP32, _MM_LOADU_FP32(ptr_dense + k*handle->n + 3*SIMD_WIDTH_FP32));
          _MM_STORE_FP32(scratch_B + k*num_regs*SIMD_WIDTH_FP32 + 4*SIMD_WIDTH_FP32, _MM_LOADU_FP32(ptr_dense + k*handle->n + 4*SIMD_WIDTH_FP32));
          _MM_STORE_FP32(scratch_B + k*num_regs*SIMD_WIDTH_FP32 + 5*SIMD_WIDTH_FP32, _MM_LOADU_FP32(ptr_dense + k*handle->n + 5*SIMD_WIDTH_FP32));
        }
      } else {
        for (k = 0; k < num_k; k++) {
          for (n = 0; n < num_full_regs; n+=2) {
            _MM_STORE_FP32(scratch_B + k*num_regs*SIMD_WIDTH_FP32 + n*SIMD_WIDTH_FP32, _MM_LOADU_FP32(ptr_dense + k*handle->n + n*SIMD_WIDTH_FP32));
            _MM_STORE_FP32(scratch_B + k*num_regs*SIMD_WIDTH_FP32 + (n+1)*SIMD_WIDTH_FP32, _MM_LOADU_FP32(ptr_dense + k*handle->n + (n+1)*SIMD_WIDTH_FP32));
          }
          for (n = last_n_start; n < num_n; n++) {
            scratch_B[k*num_regs*SIMD_WIDTH_FP32 + n] = ptr_dense[k*handle->n + n];
          }
        }
      }
    }
#if 0
    printf("B_col\n");
    for (k = 0; k < num_k; k++) {
      printf(" %lf ", ptr_dense[k*handle->n]);
    }
    printf("\n");
#endif
    scratch_C_base = scratch_C - m_overall_start*num_regs*SIMD_WIDTH_FP32;
    scratch_B_base = scratch_B; /* - k_overall_start*num_regs*SIMD_WIDTH_FP32;*/

    for (m = m_overall_start; m < m_overall_start + num_m_aligned; m+=2, m_local+=2) {
      int start_j, end_j, end_j_2, num_j, num_j_2;
      const uint16_t*  LIBXSMM_RESTRICT sp_c_ptr_base;
      const uint16_t*  LIBXSMM_RESTRICT sp_c_ptr_base_2;
      const float* LIBXSMM_RESTRICT sp_v_ptr_base;
      const float* LIBXSMM_RESTRICT sp_v_ptr_base_2;
      float* LIBXSMM_RESTRICT result_m_index;
      float* LIBXSMM_RESTRICT result_m_index_2;

      if ( m_local >= m_block_size) { block_A++; slice = A_sparse[block_A]; m_local = 0; }

      start_j =  slice.rowidx[m_local];
      end_j   =  slice.rowidx[m_local + 1];
      end_j_2 =  slice.rowidx[m_local + 2];
      num_j   = (end_j - start_j);
      num_j_2   = (end_j_2 - end_j);
      sp_c_ptr_base = slice.colidx + start_j;
      sp_c_ptr_base_2 = slice.colidx + end_j;
      sp_v_ptr_base = (float *)(slice.values) + start_j;
      sp_v_ptr_base_2 = (float *)(slice.values) + end_j;
      result_m_index = scratch_C_base + (m)*num_regs*SIMD_WIDTH_FP32;
      result_m_index_2 = scratch_C_base + (m+1)*num_regs*SIMD_WIDTH_FP32;

      if (!last_block_n)
      {
        int64_t j = 0, j2 = 0;
        sum[0] = _MM_LOAD_FP32(result_m_index + 0*SIMD_WIDTH_FP32);
        sum[0+num_regs] = _MM_LOAD_FP32(result_m_index_2 + 0*SIMD_WIDTH_FP32);
        sum[1] = _MM_LOAD_FP32(result_m_index + 1*SIMD_WIDTH_FP32);
        sum[1+num_regs] = _MM_LOAD_FP32(result_m_index_2 + 1*SIMD_WIDTH_FP32);
        sum[2] = _MM_LOAD_FP32(result_m_index + 2*SIMD_WIDTH_FP32);
        sum[2+num_regs] = _MM_LOAD_FP32(result_m_index_2 + 2*SIMD_WIDTH_FP32);
        sum[3] = _MM_LOAD_FP32(result_m_index + 3*SIMD_WIDTH_FP32);
        sum[3+num_regs] = _MM_LOAD_FP32(result_m_index_2 + 3*SIMD_WIDTH_FP32);
        sum[4] = _MM_LOAD_FP32(result_m_index + 4*SIMD_WIDTH_FP32);
        sum[4+num_regs] = _MM_LOAD_FP32(result_m_index_2 + 4*SIMD_WIDTH_FP32);
        sum[5] = _MM_LOAD_FP32(result_m_index + 5*SIMD_WIDTH_FP32);
        sum[5+num_regs] = _MM_LOAD_FP32(result_m_index_2 + 5*SIMD_WIDTH_FP32);
        for (; j < num_j && j2 < num_j_2; j++, j2++) {
          const float* const LIBXSMM_RESTRICT sp_col_dense_index = scratch_B_base +  (unsigned int)sp_c_ptr_base[j]*num_regs*SIMD_WIDTH_FP32;
          const float* const LIBXSMM_RESTRICT sp_col_dense_index_2 = scratch_B_base + (unsigned int)sp_c_ptr_base_2[j2]*num_regs*SIMD_WIDTH_FP32;
          SIMDTYPE_FP32 v_v = _MM_SET1_FP32(sp_v_ptr_base[j]);
          SIMDTYPE_FP32 v_v_2 = _MM_SET1_FP32(sp_v_ptr_base_2[j2]);
          sum[0] = _MM_FMADD_FP32(v_v, _MM_LOAD_FP32(sp_col_dense_index + 0*SIMD_WIDTH_FP32), sum[0]);
          sum[0 + num_regs] = _MM_FMADD_FP32(v_v_2, _MM_LOAD_FP32(sp_col_dense_index_2 + 0*SIMD_WIDTH_FP32), sum[0+num_regs]);
          sum[1] = _MM_FMADD_FP32(v_v, _MM_LOAD_FP32(sp_col_dense_index + 1*SIMD_WIDTH_FP32), sum[1]);
          sum[1 + num_regs] = _MM_FMADD_FP32(v_v_2, _MM_LOAD_FP32(sp_col_dense_index_2 + 1*SIMD_WIDTH_FP32), sum[1+num_regs]);
          sum[2] = _MM_FMADD_FP32(v_v, _MM_LOAD_FP32(sp_col_dense_index + 2*SIMD_WIDTH_FP32), sum[2]);
          sum[2 + num_regs] = _MM_FMADD_FP32(v_v_2, _MM_LOAD_FP32(sp_col_dense_index_2 + 2*SIMD_WIDTH_FP32), sum[2+num_regs]);
          sum[3] = _MM_FMADD_FP32(v_v, _MM_LOAD_FP32(sp_col_dense_index + 3*SIMD_WIDTH_FP32), sum[3]);
          sum[3 + num_regs] = _MM_FMADD_FP32(v_v_2, _MM_LOAD_FP32(sp_col_dense_index_2 + 3*SIMD_WIDTH_FP32), sum[3+num_regs]);
          sum[4] = _MM_FMADD_FP32(v_v, _MM_LOAD_FP32(sp_col_dense_index + 4*SIMD_WIDTH_FP32), sum[4]);
          sum[4 + num_regs] = _MM_FMADD_FP32(v_v_2, _MM_LOAD_FP32(sp_col_dense_index_2 + 4*SIMD_WIDTH_FP32), sum[4+num_regs]);
          sum[5] = _MM_FMADD_FP32(v_v, _MM_LOAD_FP32(sp_col_dense_index + 5*SIMD_WIDTH_FP32), sum[5]);
          sum[5 + num_regs] = _MM_FMADD_FP32(v_v_2, _MM_LOAD_FP32(sp_col_dense_index_2 + 5*SIMD_WIDTH_FP32), sum[5+num_regs]);
        }
        for (; j < num_j; j++) {
          const float* const LIBXSMM_RESTRICT sp_col_dense_index = scratch_B_base +  (unsigned int)sp_c_ptr_base[j]*num_regs*SIMD_WIDTH_FP32;
          SIMDTYPE_FP32 v_v = _MM_SET1_FP32(sp_v_ptr_base[j]);
          sum[0] = _MM_FMADD_FP32(v_v, _MM_LOAD_FP32(sp_col_dense_index + 0*SIMD_WIDTH_FP32), sum[0]);
          sum[1] = _MM_FMADD_FP32(v_v, _MM_LOAD_FP32(sp_col_dense_index + 1*SIMD_WIDTH_FP32), sum[1]);
          sum[2] = _MM_FMADD_FP32(v_v, _MM_LOAD_FP32(sp_col_dense_index + 2*SIMD_WIDTH_FP32), sum[2]);
          sum[3] = _MM_FMADD_FP32(v_v, _MM_LOAD_FP32(sp_col_dense_index + 3*SIMD_WIDTH_FP32), sum[3]);
          sum[4] = _MM_FMADD_FP32(v_v, _MM_LOAD_FP32(sp_col_dense_index + 4*SIMD_WIDTH_FP32), sum[4]);
          sum[5] = _MM_FMADD_FP32(v_v, _MM_LOAD_FP32(sp_col_dense_index + 5*SIMD_WIDTH_FP32), sum[5]);
        }
        for (; j2 < num_j_2; j2++) {
          const float* const LIBXSMM_RESTRICT sp_col_dense_index_2 = scratch_B_base + (unsigned int)sp_c_ptr_base_2[j2]*num_regs*SIMD_WIDTH_FP32;
          SIMDTYPE_FP32 v_v_2 = _MM_SET1_FP32(sp_v_ptr_base_2[j2]);
          sum[0 + num_regs] = _MM_FMADD_FP32(v_v_2, _MM_LOAD_FP32(sp_col_dense_index_2 + 0*SIMD_WIDTH_FP32), sum[0+num_regs]);
          sum[1 + num_regs] = _MM_FMADD_FP32(v_v_2, _MM_LOAD_FP32(sp_col_dense_index_2 + 1*SIMD_WIDTH_FP32), sum[1+num_regs]);
          sum[2 + num_regs] = _MM_FMADD_FP32(v_v_2, _MM_LOAD_FP32(sp_col_dense_index_2 + 2*SIMD_WIDTH_FP32), sum[2+num_regs]);
          sum[3 + num_regs] = _MM_FMADD_FP32(v_v_2, _MM_LOAD_FP32(sp_col_dense_index_2 + 3*SIMD_WIDTH_FP32), sum[3+num_regs]);
          sum[4 + num_regs] = _MM_FMADD_FP32(v_v_2, _MM_LOAD_FP32(sp_col_dense_index_2 + 4*SIMD_WIDTH_FP32), sum[4+num_regs]);
          sum[5 + num_regs] = _MM_FMADD_FP32(v_v_2, _MM_LOAD_FP32(sp_col_dense_index_2 + 5*SIMD_WIDTH_FP32), sum[5+num_regs]);
        }
        _MM_STORE_FP32(result_m_index + 0*SIMD_WIDTH_FP32, sum[0]);
        _MM_STORE_FP32(result_m_index_2 + 0*SIMD_WIDTH_FP32, sum[0+num_regs]);
        _MM_STORE_FP32(result_m_index + 1*SIMD_WIDTH_FP32, sum[1]);
        _MM_STORE_FP32(result_m_index_2 + 1*SIMD_WIDTH_FP32, sum[1+num_regs]);
        _MM_STORE_FP32(result_m_index + 2*SIMD_WIDTH_FP32, sum[2]);
        _MM_STORE_FP32(result_m_index_2 + 2*SIMD_WIDTH_FP32, sum[2+num_regs]);
        _MM_STORE_FP32(result_m_index + 3*SIMD_WIDTH_FP32, sum[3]);
        _MM_STORE_FP32(result_m_index_2 + 3*SIMD_WIDTH_FP32, sum[3+num_regs]);
        _MM_STORE_FP32(result_m_index + 4*SIMD_WIDTH_FP32, sum[4]);
        _MM_STORE_FP32(result_m_index_2 + 4*SIMD_WIDTH_FP32, sum[4+num_regs]);
        _MM_STORE_FP32(result_m_index + 5*SIMD_WIDTH_FP32, sum[5]);
        _MM_STORE_FP32(result_m_index_2 + 5*SIMD_WIDTH_FP32, sum[5+num_regs]);
      }
      else {
        int64_t j = 0, j2 = 0;
        for (n = 0; n < num_full_regs; n+=2) {
          sum[n] = _MM_SETZERO_FP32();
          sum[n+num_regs] = _MM_SETZERO_FP32();
          sum[n+1] = _MM_SETZERO_FP32();
          sum[n+1+num_regs] = _MM_SETZERO_FP32();
        }
        for (; j < num_j && j2 < num_j_2; j++, j2++) {
          const float* const LIBXSMM_RESTRICT sp_col_dense_index = scratch_B_base +  (unsigned int)sp_c_ptr_base[j]*num_regs*SIMD_WIDTH_FP32;
          const float* const LIBXSMM_RESTRICT sp_col_dense_index_2 = scratch_B_base + (unsigned int)sp_c_ptr_base_2[j2]*num_regs*SIMD_WIDTH_FP32;
          SIMDTYPE_FP32 v_v = _MM_SET1_FP32(sp_v_ptr_base[j]);
          SIMDTYPE_FP32 v_v_2 = _MM_SET1_FP32(sp_v_ptr_base_2[j2]);
          for (n = 0; n < num_full_regs; n+=2) {
            sum[n] = _MM_FMADD_FP32(v_v, _MM_LOAD_FP32(sp_col_dense_index + n*SIMD_WIDTH_FP32), sum[n]);
            sum[n + num_regs] = _MM_FMADD_FP32(v_v_2, _MM_LOAD_FP32(sp_col_dense_index_2 + n*SIMD_WIDTH_FP32), sum[n+num_regs]);
            sum[n+1] = _MM_FMADD_FP32(v_v, _MM_LOAD_FP32(sp_col_dense_index + (n+1)*SIMD_WIDTH_FP32), sum[n+1]);
            sum[n+1 + num_regs] = _MM_FMADD_FP32(v_v_2, _MM_LOAD_FP32(sp_col_dense_index_2 + (n+1)*SIMD_WIDTH_FP32), sum[n+1+num_regs]);
          }
          {
            float v_v_f = sp_v_ptr_base[j];
            float v_v_f_2 = sp_v_ptr_base_2[j2];
            for (n = last_n_start; n < num_n; n++) {
              result_m_index[n] += sp_col_dense_index[n]*v_v_f;
              result_m_index_2[n] += sp_col_dense_index_2[n]*v_v_f_2;
            }
          }
        }
        for (; j < num_j; j++) {
          const float* const LIBXSMM_RESTRICT sp_col_dense_index = scratch_B_base +  (unsigned int)sp_c_ptr_base[j]*num_regs*SIMD_WIDTH_FP32;
          SIMDTYPE_FP32 v_v = _MM_SET1_FP32(sp_v_ptr_base[j]);
          for (n = 0; n < num_full_regs; n+=2) {
            sum[n] = _MM_FMADD_FP32(v_v, _MM_LOAD_FP32(sp_col_dense_index + n*SIMD_WIDTH_FP32), sum[n]);
            sum[n+1] = _MM_FMADD_FP32(v_v, _MM_LOAD_FP32(sp_col_dense_index + (n+1)*SIMD_WIDTH_FP32), sum[n+1]);
          }
          {
            float v_v_f = sp_v_ptr_base[j];
            for (n = last_n_start; n < num_n; n++) {
              result_m_index[n] += sp_col_dense_index[n]*v_v_f;
            }
          }
        }
        for (; j2 < num_j_2; j2++) {
          const float* const LIBXSMM_RESTRICT sp_col_dense_index_2 = scratch_B_base + (unsigned int)sp_c_ptr_base_2[j2]*num_regs*SIMD_WIDTH_FP32;
          SIMDTYPE_FP32 v_v_2 = _MM_SET1_FP32(sp_v_ptr_base_2[j2]);
          for (n = 0; n < num_full_regs; n+=2) {
            sum[n + num_regs] = _MM_FMADD_FP32(v_v_2, _MM_LOAD_FP32(sp_col_dense_index_2 + n*SIMD_WIDTH_FP32), sum[n+num_regs]);
            sum[n+1 + num_regs] = _MM_FMADD_FP32(v_v_2, _MM_LOAD_FP32(sp_col_dense_index_2 + (n+1)*SIMD_WIDTH_FP32), sum[n+1+num_regs]);
          }
          {
            float v_v_f_2 = sp_v_ptr_base_2[j2];
            for (n = last_n_start; n < num_n; n++) {
              result_m_index_2[n] += sp_col_dense_index_2[n]*v_v_f_2;
            }
          }
        }
        for (n = 0; n < num_full_regs; n+=2) {
          _MM_STORE_FP32(result_m_index + n*SIMD_WIDTH_FP32,  _MM_ADD_FP32(sum[n], _MM_LOAD_FP32(result_m_index + n*SIMD_WIDTH_FP32)));
          _MM_STORE_FP32(result_m_index_2 + n*SIMD_WIDTH_FP32,  _MM_ADD_FP32(sum[n+num_regs], _MM_LOAD_FP32(result_m_index_2 + n*SIMD_WIDTH_FP32)));
          _MM_STORE_FP32(result_m_index + (n+1)*SIMD_WIDTH_FP32,  _MM_ADD_FP32(sum[n+1], _MM_LOAD_FP32(result_m_index + (n+1)*SIMD_WIDTH_FP32)));
          _MM_STORE_FP32(result_m_index_2 + (n+1)*SIMD_WIDTH_FP32,  _MM_ADD_FP32(sum[n+1+num_regs], _MM_LOAD_FP32(result_m_index_2 + (n+1)*SIMD_WIDTH_FP32)));
        }
      }
    }
    for (m = m_overall_start + num_m_aligned; m < m_overall_end; m++, m_local++) {
      int start_j, end_j, num_j;
      const uint16_t*  LIBXSMM_RESTRICT sp_c_ptr_base;
      const float* LIBXSMM_RESTRICT sp_v_ptr_base;
      float* LIBXSMM_RESTRICT result_m_index;

      if ( m_local >= m_block_size) { block_A++; slice = A_sparse[block_A]; m_local = 0; }

      start_j =  slice.rowidx[m_local];
      end_j   =  slice.rowidx[m_local + 1];
      num_j   = (end_j - start_j);
      sp_c_ptr_base = slice.colidx + start_j;
      sp_v_ptr_base = slice.values + start_j;
      result_m_index = scratch_C_base + (m)*num_regs*SIMD_WIDTH_FP32;

      if (!last_block_n) {
        int64_t j = 0;
        sum[0] = _MM_LOAD_FP32(result_m_index + 0*SIMD_WIDTH_FP32);
        sum[1] = _MM_LOAD_FP32(result_m_index + 1*SIMD_WIDTH_FP32);
        sum[2] = _MM_LOAD_FP32(result_m_index + 2*SIMD_WIDTH_FP32);
        sum[3] = _MM_LOAD_FP32(result_m_index + 3*SIMD_WIDTH_FP32);
        sum[4] = _MM_LOAD_FP32(result_m_index + 4*SIMD_WIDTH_FP32);
        sum[5] = _MM_LOAD_FP32(result_m_index + 5*SIMD_WIDTH_FP32);
        for (; j < num_j; j++) {
          const float* const LIBXSMM_RESTRICT sp_col_dense_index = scratch_B_base +  (unsigned int)sp_c_ptr_base[j]*num_regs*SIMD_WIDTH_FP32;
          SIMDTYPE_FP32 v_v = _MM_SET1_FP32(sp_v_ptr_base[j]);
          sum[0] = _MM_FMADD_FP32(v_v, _MM_LOAD_FP32(sp_col_dense_index + 0*SIMD_WIDTH_FP32), sum[0]);
          sum[1] = _MM_FMADD_FP32(v_v, _MM_LOAD_FP32(sp_col_dense_index + 1*SIMD_WIDTH_FP32), sum[1]);
          sum[2] = _MM_FMADD_FP32(v_v, _MM_LOAD_FP32(sp_col_dense_index + 2*SIMD_WIDTH_FP32), sum[2]);
          sum[3] = _MM_FMADD_FP32(v_v, _MM_LOAD_FP32(sp_col_dense_index + 3*SIMD_WIDTH_FP32), sum[3]);
          sum[4] = _MM_FMADD_FP32(v_v, _MM_LOAD_FP32(sp_col_dense_index + 4*SIMD_WIDTH_FP32), sum[4]);
          sum[5] = _MM_FMADD_FP32(v_v, _MM_LOAD_FP32(sp_col_dense_index + 5*SIMD_WIDTH_FP32), sum[5]);
        }
        _MM_STORE_FP32(result_m_index + 0*SIMD_WIDTH_FP32, sum[0]);
        _MM_STORE_FP32(result_m_index + 1*SIMD_WIDTH_FP32, sum[1]);
        _MM_STORE_FP32(result_m_index + 2*SIMD_WIDTH_FP32, sum[2]);
        _MM_STORE_FP32(result_m_index + 3*SIMD_WIDTH_FP32, sum[3]);
        _MM_STORE_FP32(result_m_index + 4*SIMD_WIDTH_FP32, sum[4]);
        _MM_STORE_FP32(result_m_index + 5*SIMD_WIDTH_FP32, sum[5]);
      }
      else {
        int64_t j = 0;
        for (n = 0; n < num_full_regs; n+=2) {
          sum[n] = _MM_SETZERO_FP32();
          sum[n+1] = _MM_SETZERO_FP32();
        }
        for (; j < num_j; j++) {
          const float* const LIBXSMM_RESTRICT sp_col_dense_index = scratch_B_base +  (unsigned int)sp_c_ptr_base[j]*num_regs*SIMD_WIDTH_FP32;
          SIMDTYPE_FP32 v_v = _MM_SET1_FP32(sp_v_ptr_base[j]);
          for (n = 0; n < num_full_regs; n+=2) {
            sum[n] = _MM_FMADD_FP32(v_v, _MM_LOAD_FP32(sp_col_dense_index + n*SIMD_WIDTH_FP32), sum[n]);
            sum[n+1] = _MM_FMADD_FP32(v_v, _MM_LOAD_FP32(sp_col_dense_index + (n+1)*SIMD_WIDTH_FP32), sum[n+1]);
          }
          {
            float v_v_f = sp_v_ptr_base[j];
            for (n = last_n_start; n < num_n; n++) {
              result_m_index[n] += sp_col_dense_index[n]*v_v_f;
            }
          }
        }
        for (n = 0; n < num_full_regs; n+=2) {
          _MM_STORE_FP32(result_m_index + n*SIMD_WIDTH_FP32, _MM_ADD_FP32(sum[n], _MM_LOAD_FP32(result_m_index + n*SIMD_WIDTH_FP32)));
          _MM_STORE_FP32(result_m_index + (n+1)*SIMD_WIDTH_FP32, _MM_ADD_FP32(sum[n+1], _MM_LOAD_FP32(result_m_index + (n+1)*SIMD_WIDTH_FP32)));
        }
      }
    }
  } /* kb */
#if 0
  for (m = 0; m < 3; m++) {
    for (n = 0; n < num_n; n++) {
      printf("%f ", scratch_C[m*num_regs*SIMD_WIDTH_FP32 + n]);
    }
    printf("\n");
  }
#endif
  /* Copy out C matrix */
  if(transC == 'Y') {
    int num_m_simd = num_m / SIMD_WIDTH_FP32 * SIMD_WIDTH_FP32;
    int num_n_simd = num_n / SIMD_WIDTH_FP32 * SIMD_WIDTH_FP32;
    int n2;

    ptr_result = C + n_overall_start*handle->m + m_overall_start;
    for(n = 0; n < num_n_simd; n+=SIMD_WIDTH_FP32){
      for(m = 0; m < num_m_simd; m+=SIMD_WIDTH_FP32){
        TRANSPOSE_SIMD_WIDTH_KERNEL(scratch_C + m*n_block_size + n, n_block_size, ptr_result + n*handle->m + m, handle->m);
      }
      /* Transpose a SIMD_WIDTH_FP32 * (num_m - num_m_simd) block of output space - input is of size (num_m - num_m_simd) * SIMD_WIDTH_FP32 */
      for(n2 = n; n2 < n + SIMD_WIDTH_FP32; n2++) {
        for(m = num_m_simd; m < num_m; m++){
          ptr_result[n2*handle->m + m] = scratch_C[m*n_block_size + n2]; 
        }
      }
    }
    /* Transpose a (num_n - num_n_simd) * num_m block of output space - input is of size num_m * (num_n - num_n_simd) */
    for(n = num_n_simd; n < num_n; n++){
      for(m = 0; m < num_m; m++){
        ptr_result[n*handle->m + m] = scratch_C[m*n_block_size + n];
      }
    }
  }
  else {
    if (!last_block_n) {
      for (m = 0; m < num_m; m++) {
        _MM_STOREU_FP32(ptr_result + m*handle->n + 0*SIMD_WIDTH_FP32, _MM_LOAD_FP32(scratch_C + m*num_regs*SIMD_WIDTH_FP32 + 0*SIMD_WIDTH_FP32));
        _MM_STOREU_FP32(ptr_result + m*handle->n + 1*SIMD_WIDTH_FP32, _MM_LOAD_FP32(scratch_C + m*num_regs*SIMD_WIDTH_FP32 + 1*SIMD_WIDTH_FP32));
        _MM_STOREU_FP32(ptr_result + m*handle->n + 2*SIMD_WIDTH_FP32, _MM_LOAD_FP32(scratch_C + m*num_regs*SIMD_WIDTH_FP32 + 2*SIMD_WIDTH_FP32));
        _MM_STOREU_FP32(ptr_result + m*handle->n + 3*SIMD_WIDTH_FP32, _MM_LOAD_FP32(scratch_C + m*num_regs*SIMD_WIDTH_FP32 + 3*SIMD_WIDTH_FP32));
        _MM_STOREU_FP32(ptr_result + m*handle->n + 4*SIMD_WIDTH_FP32, _MM_LOAD_FP32(scratch_C + m*num_regs*SIMD_WIDTH_FP32 + 4*SIMD_WIDTH_FP32));
        _MM_STOREU_FP32(ptr_result + m*handle->n + 5*SIMD_WIDTH_FP32, _MM_LOAD_FP32(scratch_C + m*num_regs*SIMD_WIDTH_FP32 + 5*SIMD_WIDTH_FP32));
      }
    }
    else {
      for (m = 0; m < num_m; m++) {
        for (n = 0; n < num_full_regs; n+=2) {
          _MM_STOREU_FP32(ptr_result + m*handle->n + n*SIMD_WIDTH_FP32, _MM_LOAD_FP32(scratch_C + m*num_regs*SIMD_WIDTH_FP32 + n*SIMD_WIDTH_FP32));
          _MM_STOREU_FP32(ptr_result + m*handle->n + (n+1)*SIMD_WIDTH_FP32, _MM_LOAD_FP32(scratch_C + m*num_regs*SIMD_WIDTH_FP32 + (n+1)*SIMD_WIDTH_FP32));
        }
        for (n = last_n_start; n < num_n; n++) {
          ptr_result[m*handle->n + n] = scratch_C[m*num_regs*SIMD_WIDTH_FP32 + n];
        }
      }
    }
  }

}


LIBXSMM_API_DEFINITION void libxsmm_spmdm_compute_bfloat16_thread(
  const libxsmm_spmdm_handle* handle,
  char transA,
  char transB,
  const uint16_t *alpha,
  libxsmm_CSR_sparseslice* A_sparse,
  const uint16_t *B,
  char transC, 
  const uint16_t *beta,
  float* C,
  int block_id,
  int tid, int nthreads)
{
  const int m_blocks = handle->mb;
  /*const int n_blocks = handle->nb;*/
  const int k_blocks = handle->kb;
  const int m_block_size = handle->bm;
  const int n_block_size = handle->bn;
  const int k_block_size = handle->bk;
  int mb = block_id / handle->nb;
  int nb = block_id % handle->nb;


#define num_regs (6)
  int m_overall_start = mb*m_block_size;
  int m_overall_end   = (mb + 1)*m_block_size;
  int num_m;
  int num_m_aligned;

  int n_overall_start = nb*n_block_size;
  int n_overall_end   = (nb + 1)*n_block_size;
  int num_n;
  int m, n, k, kb;
  int last_block_n, num_full_regs, last_n_start;

  int k_overall_start, k_overall_end, num_k;

  float *const scratch_C = (float *)(handle->base_ptr_scratch_B_scratch_C + tid*handle->memory_for_scratch_per_thread);
  float *const scratch_B = (float *)(handle->base_ptr_scratch_B_scratch_C + tid*handle->memory_for_scratch_per_thread + m_block_size*n_block_size*sizeof(float));
  #if 0
  float *const scratch_C = (float *)(handle->spmdm_scratch_C + tid*m_block_size*n_block_size*sizeof(float));
  float *const scratch_B = (float *)(handle->spmdm_scratch_B + tid*k_block_size*n_block_size*sizeof(float));
  #endif

  SIMDTYPE_FP32 sum[2*num_regs];
  float* LIBXSMM_RESTRICT ptr_result;
#if SIMD_WIDTH_FP32 > 1
  SIMDTYPE_INT32 vzero = _MM_SETZERO_INT32();
#endif

  LIBXSMM_UNUSED(nthreads);
  LIBXSMM_UNUSED(transA);
  LIBXSMM_UNUSED(alpha);
  LIBXSMM_UNUSED(beta);
  LIBXSMM_UNUSED(tid);

  /* really is twice this */
  assert(n_block_size == num_regs*SIMD_WIDTH_FP32);

  if (m_overall_end   > handle->m) m_overall_end   = handle->m;
  num_m = (m_overall_end - m_overall_start);
  num_m_aligned = (num_m / 2) * 2;

  if (n_overall_end   > handle->n) n_overall_end   = handle->n;
  num_n = (n_overall_end - n_overall_start);
  last_block_n = (num_n != n_block_size);
  num_full_regs = (num_n / SIMD_WIDTH_FP32);
  if ((num_full_regs > 0) && (num_full_regs%2)) num_full_regs--;
  last_n_start = num_full_regs*SIMD_WIDTH_FP32;
#if 0
  printf("Block: m_overall_start: %d, m_overall_end: %d, num_m: %d, num_m_aligned: %d\n", m_overall_start, m_overall_end, num_m, num_m_aligned);
  printf("Block: n_overall_start: %d, n_overall_end: %d, num_n: %d, last_block_n: %d\n", n_overall_start, n_overall_end, num_n, last_block_n);
  printf("Block: k_blocks: %d\n", k_blocks);
#endif
  /* Copy in C matrix to buffer */
  ptr_result = C + m_overall_start*handle->n + n_overall_start;
  if (LIBXSMM_FEQ(0.f, *beta)) {
    if (!last_block_n) {
      for (m = 0; m < num_m; m++) {
        _MM_STORE_FP32(scratch_C + m*num_regs*SIMD_WIDTH_FP32 + 0*SIMD_WIDTH_FP32, _MM_SETZERO_FP32());
        _MM_STORE_FP32(scratch_C + m*num_regs*SIMD_WIDTH_FP32 + 1*SIMD_WIDTH_FP32, _MM_SETZERO_FP32());
        _MM_STORE_FP32(scratch_C + m*num_regs*SIMD_WIDTH_FP32 + 2*SIMD_WIDTH_FP32, _MM_SETZERO_FP32());
        _MM_STORE_FP32(scratch_C + m*num_regs*SIMD_WIDTH_FP32 + 3*SIMD_WIDTH_FP32, _MM_SETZERO_FP32());
        _MM_STORE_FP32(scratch_C + m*num_regs*SIMD_WIDTH_FP32 + 4*SIMD_WIDTH_FP32, _MM_SETZERO_FP32());
        _MM_STORE_FP32(scratch_C + m*num_regs*SIMD_WIDTH_FP32 + 5*SIMD_WIDTH_FP32, _MM_SETZERO_FP32());
      }
    } else {
      for (m = 0; m < num_m; m++) {
        for (n = 0; n < num_full_regs; n+=2) {
          _MM_STORE_FP32(scratch_C + m*num_regs*SIMD_WIDTH_FP32 + n*SIMD_WIDTH_FP32, _MM_SETZERO_FP32());
          _MM_STORE_FP32(scratch_C + m*num_regs*SIMD_WIDTH_FP32 + (n+1)*SIMD_WIDTH_FP32, _MM_SETZERO_FP32());
        }
        for (n = last_n_start; n < num_n; n++) {
          scratch_C[m*num_regs*SIMD_WIDTH_FP32 + n] = 0;
        }
      }
    }
  }
  else if (LIBXSMM_FEQ(1.f, *beta)) {
    if(transC == 'Y') {
      int num_m_simd = num_m / SIMD_WIDTH_FP32 * SIMD_WIDTH_FP32;
      int num_n_simd = num_n / SIMD_WIDTH_FP32 * SIMD_WIDTH_FP32;
      int m2;

      ptr_result = C + n_overall_start*handle->m + m_overall_start;
 
      for(m = 0; m < num_m_simd; m+=SIMD_WIDTH_FP32){
        for(n = 0; n < num_n_simd; n+=SIMD_WIDTH_FP32){
          //for(int m2 = m; m2 < m + SIMD_WIDTH_FP32; m2++) for( int n2 = n; n2 < n + SIMD_WIDTH_FP32; n2++) scratch_C[m2*N + n2] = ptr_result[n2*M + m2];
          TRANSPOSE_SIMD_WIDTH_KERNEL(ptr_result + n*handle->m + m, handle->m, scratch_C + m*n_block_size + n, n_block_size);
        }
        /* Transpose a SIMD_WIDTH_FP32 * (num_n - num_n_simd) block of output space - input is of size (num_n - num_n_simd) * SIMD_WIDTH_FP32 */
        for(m2 = m; m2 < m + SIMD_WIDTH_FP32; m2++) {
          for(n = num_n_simd; n < num_n; n++){
            scratch_C[m2*n_block_size + n] = ptr_result[n*handle->m + m2];
          }
        }
      }
      /* Transpose a (num_m - num_m_simd) * num_n block of output space - input is of size num_n * (num_m - num_m_simd) */
      for(m = num_m_simd; m < num_m; m++){
        for(n = 0; n < num_n; n++){
          scratch_C[m*n_block_size + n] = ptr_result[n*handle->m + m];
        }
      }
    }
    else {
      if (!last_block_n) {
        for (m = 0; m < num_m; m++) {
          _MM_STORE_FP32(scratch_C + m*num_regs*SIMD_WIDTH_FP32 + 0*SIMD_WIDTH_FP32, _MM_LOADU_FP32(ptr_result + m*handle->n + 0*SIMD_WIDTH_FP32));
          _MM_STORE_FP32(scratch_C + m*num_regs*SIMD_WIDTH_FP32 + 1*SIMD_WIDTH_FP32, _MM_LOADU_FP32(ptr_result + m*handle->n + 1*SIMD_WIDTH_FP32));
          _MM_STORE_FP32(scratch_C + m*num_regs*SIMD_WIDTH_FP32 + 2*SIMD_WIDTH_FP32, _MM_LOADU_FP32(ptr_result + m*handle->n + 2*SIMD_WIDTH_FP32));
          _MM_STORE_FP32(scratch_C + m*num_regs*SIMD_WIDTH_FP32 + 3*SIMD_WIDTH_FP32, _MM_LOADU_FP32(ptr_result + m*handle->n + 3*SIMD_WIDTH_FP32));
          _MM_STORE_FP32(scratch_C + m*num_regs*SIMD_WIDTH_FP32 + 4*SIMD_WIDTH_FP32, _MM_LOADU_FP32(ptr_result + m*handle->n + 4*SIMD_WIDTH_FP32));
          _MM_STORE_FP32(scratch_C + m*num_regs*SIMD_WIDTH_FP32 + 5*SIMD_WIDTH_FP32, _MM_LOADU_FP32(ptr_result + m*handle->n + 5*SIMD_WIDTH_FP32));
        }
      }
      else {
        for (m = 0; m < num_m; m++) {
          for (n = 0; n < num_full_regs; n+=2) {
            _MM_STORE_FP32(scratch_C + m*num_regs*SIMD_WIDTH_FP32 + n*SIMD_WIDTH_FP32, _MM_LOADU_FP32(ptr_result + m*handle->n + n*SIMD_WIDTH_FP32));
            _MM_STORE_FP32(scratch_C + m*num_regs*SIMD_WIDTH_FP32 + (n+1)*SIMD_WIDTH_FP32, _MM_LOADU_FP32(ptr_result + m*handle->n + (n+1)*SIMD_WIDTH_FP32));
          }
          for (n = last_n_start; n < num_n; n++) {
            scratch_C[m*num_regs*SIMD_WIDTH_FP32 + n] = ptr_result[m*handle->n + n];
          }
        }
      }
    }
  }
  else {
    SIMDTYPE_FP32 beta_v = _MM_SET1_FP32(*beta);
    if(transC == 'Y') {
      int num_m_simd = num_m / SIMD_WIDTH_FP32 * SIMD_WIDTH_FP32;
      int num_n_simd = num_n / SIMD_WIDTH_FP32 * SIMD_WIDTH_FP32;
      int m2;

      ptr_result = C + n_overall_start*handle->m + m_overall_start;
 
      for(m = 0; m < num_m_simd; m+=SIMD_WIDTH_FP32){
        for(n = 0; n < num_n_simd; n+=SIMD_WIDTH_FP32){
          //for(int m2 = m; m2 < m + SIMD_WIDTH_FP32; m2++) for( int n2 = n; n2 < n + SIMD_WIDTH_FP32; n2++) scratch_C[m2*N + n2] = ptr_result[n2*M + m2];
          TRANSPOSE_SIMD_WIDTH_KERNEL(ptr_result + n*handle->m + m, handle->m, scratch_C + m*n_block_size + n, n_block_size);
          _MM_STORE_FP32(scratch_C + m*n_block_size + n, _MM_MUL_FP32(beta_v, _MM_LOADU_FP32(scratch_C + m*n_block_size + n)));
          _MM_STORE_FP32(scratch_C + m*n_block_size + n + n_block_size, _MM_MUL_FP32(beta_v, _MM_LOADU_FP32(scratch_C + m*n_block_size + n + n_block_size)));
          _MM_STORE_FP32(scratch_C + m*n_block_size + n + 2*n_block_size, _MM_MUL_FP32(beta_v, _MM_LOADU_FP32(scratch_C + m*n_block_size + n + 2*n_block_size)));
          _MM_STORE_FP32(scratch_C + m*n_block_size + n + 3*n_block_size, _MM_MUL_FP32(beta_v, _MM_LOADU_FP32(scratch_C + m*n_block_size + n + 3*n_block_size)));
          _MM_STORE_FP32(scratch_C + m*n_block_size + n + 4*n_block_size, _MM_MUL_FP32(beta_v, _MM_LOADU_FP32(scratch_C + m*n_block_size + n + 4*n_block_size)));
          _MM_STORE_FP32(scratch_C + m*n_block_size + n + 5*n_block_size, _MM_MUL_FP32(beta_v, _MM_LOADU_FP32(scratch_C + m*n_block_size + n + 5*n_block_size)));
          _MM_STORE_FP32(scratch_C + m*n_block_size + n + 6*n_block_size, _MM_MUL_FP32(beta_v, _MM_LOADU_FP32(scratch_C + m*n_block_size + n + 6*n_block_size)));
          _MM_STORE_FP32(scratch_C + m*n_block_size + n + 7*n_block_size, _MM_MUL_FP32(beta_v, _MM_LOADU_FP32(scratch_C + m*n_block_size + n + 7*n_block_size)));
        }
        /* Transpose a SIMD_WIDTH_FP32 * (num_n - num_n_simd) block of output space - input is of size (num_n - num_n_simd) * SIMD_WIDTH_FP32 */
        for(m2 = m; m2 < m + SIMD_WIDTH_FP32; m2++) {
          for(n = num_n_simd; n < num_n; n++){
            scratch_C[m2*n_block_size + n] = (*beta)*ptr_result[n*handle->m + m2];
          }
        }
      }
      /* Transpose a (num_m - num_m_simd) * num_n block of output space - input is of size num_n * (num_m - num_m_simd) */
      for(m = num_m_simd; m < num_m; m++){
        for(n = 0; n < num_n; n++){
          scratch_C[m*n_block_size + n] = (*beta)*ptr_result[n*handle->m + m];
        }
      }

    }
    else {
      if (!last_block_n) {
        for (m = 0; m < num_m; m++) {
          _MM_STORE_FP32(scratch_C + m*num_regs*SIMD_WIDTH_FP32 + 0*SIMD_WIDTH_FP32, _MM_MUL_FP32(beta_v, _MM_LOADU_FP32(ptr_result + m*handle->n + 0*SIMD_WIDTH_FP32)));
          _MM_STORE_FP32(scratch_C + m*num_regs*SIMD_WIDTH_FP32 + 1*SIMD_WIDTH_FP32, _MM_MUL_FP32(beta_v, _MM_LOADU_FP32(ptr_result + m*handle->n + 1*SIMD_WIDTH_FP32)));
          _MM_STORE_FP32(scratch_C + m*num_regs*SIMD_WIDTH_FP32 + 2*SIMD_WIDTH_FP32, _MM_MUL_FP32(beta_v, _MM_LOADU_FP32(ptr_result + m*handle->n + 2*SIMD_WIDTH_FP32)));
          _MM_STORE_FP32(scratch_C + m*num_regs*SIMD_WIDTH_FP32 + 3*SIMD_WIDTH_FP32, _MM_MUL_FP32(beta_v, _MM_LOADU_FP32(ptr_result + m*handle->n + 3*SIMD_WIDTH_FP32)));
          _MM_STORE_FP32(scratch_C + m*num_regs*SIMD_WIDTH_FP32 + 4*SIMD_WIDTH_FP32, _MM_MUL_FP32(beta_v, _MM_LOADU_FP32(ptr_result + m*handle->n + 4*SIMD_WIDTH_FP32)));
          _MM_STORE_FP32(scratch_C + m*num_regs*SIMD_WIDTH_FP32 + 5*SIMD_WIDTH_FP32, _MM_MUL_FP32(beta_v, _MM_LOADU_FP32(ptr_result + m*handle->n + 5*SIMD_WIDTH_FP32)));
        }
      }
      else {
        for (m = 0; m < num_m; m++) {
          for (n = 0; n < num_full_regs; n+=2) {
            _MM_STORE_FP32(scratch_C + m*num_regs*SIMD_WIDTH_FP32 + n*SIMD_WIDTH_FP32, _MM_MUL_FP32(beta_v, _MM_LOADU_FP32(ptr_result + m*handle->n + n*SIMD_WIDTH_FP32)));
            _MM_STORE_FP32(scratch_C + m*num_regs*SIMD_WIDTH_FP32 + (n+1)*SIMD_WIDTH_FP32, _MM_MUL_FP32(beta_v, _MM_LOADU_FP32(ptr_result + m*handle->n + (n+1)*SIMD_WIDTH_FP32)));
          }
          for (n = last_n_start; n < num_n; n++) {
            scratch_C[m*num_regs*SIMD_WIDTH_FP32 + n] = (*beta)*ptr_result[m*handle->n + n];
          }
        }
      }
    }
  }
 
  for (kb = 0; kb < k_blocks; kb++) {
    const uint16_t* LIBXSMM_RESTRICT ptr_dense;
    float * LIBXSMM_RESTRICT scratch_C_base;
    const float * LIBXSMM_RESTRICT scratch_B_base;
    int block_A = kb * m_blocks + mb;
    libxsmm_CSR_sparseslice slice = A_sparse[block_A];
    int m_local = 0;

    k_overall_start = kb*k_block_size;
    k_overall_end   = (kb+1)*k_block_size;
    num_k = (k_overall_end - k_overall_start);

    /* Copy in B matrix */
    if (transB == 'Y')
    {
      int num_k_simd = num_k / SIMD_WIDTH_FP32 * SIMD_WIDTH_FP32;
      int num_n_simd = num_n / SIMD_WIDTH_FP32 * SIMD_WIDTH_FP32;
      int k2;

      ptr_dense = B + n_overall_start*handle->k + k_overall_start;
 
      for(k = 0; k < num_k_simd; k+=SIMD_WIDTH_FP32){
        for(n = 0; n < num_n_simd; n+=SIMD_WIDTH_FP32){
          //for(int m2 = m; m2 < m + SIMD_WIDTH_FP32; m2++) for( int n2 = n; n2 < n + SIMD_WIDTH_FP32; n2++) ptr_B[m2*N + n2] = ptr_A[n2*M + m2];
          TRANSPOSE_SIMD_WIDTH_KERNEL_BFLOAT16(ptr_dense + n*handle->k + k, handle->k, scratch_B + k*n_block_size + n, n_block_size);
        }
        /* Transpose a SIMD_WIDTH_FP32 * (num_n - num_n_simd) block of output space - input is of size (num_n - num_n_simd) * SIMD_WIDTH_FP32 */
        for(k2 = k; k2 < k + SIMD_WIDTH_FP32; k2++) {
          for(n = num_n_simd; n < num_n; n++){
            uint16_t restmp = ptr_dense[n*handle->k + k2];
            union { int i; float f; } res;
            res.i = restmp;
            res.i <<= 16;
            scratch_B[k2*n_block_size + n] = res.f;
          }
        }
      }
      /* Transpose a (num_m - num_m_simd) * num_n block of output space - input is of size num_n * (num_m - num_m_simd) */
      for(k = num_k_simd; k < num_k; k++){
        for(n = 0; n < num_n; n++){
          uint16_t restmp = ptr_dense[n*handle->k + k];
          union { int i; float f; } res;
          res.i = restmp;
          res.i <<= 16;
          scratch_B[k*n_block_size + n] = res.f;
        }
      }
    }
    else
    {
      ptr_dense = B + k_overall_start*handle->n + n_overall_start;
      if (!last_block_n) {
        for (k = 0; k < num_k; k++) {
          SIMDTYPE_INT32 vload_0 =  _MM_LOADU_INT32((const SIMDTYPE_INT32 *)(ptr_dense + k*handle->n + 2*0*SIMD_WIDTH_FP32));
          SIMDTYPE_INT32 vload_1, vload_2;
          SIMDTYPE_FP32 v1_0, v2_0;
          SIMDTYPE_FP32 v1_1, v2_1;
          SIMDTYPE_FP32 v1_2, v2_2;
          EXPAND_BFLOAT16(vload_0, v1_0, v2_0);
          _MM_STORE_FP32(scratch_B + k*num_regs*SIMD_WIDTH_FP32 + 2*0*SIMD_WIDTH_FP32, v1_0);
          _MM_STORE_FP32(scratch_B + k*num_regs*SIMD_WIDTH_FP32 + (2*0+1)*SIMD_WIDTH_FP32, v2_0);
          vload_1 =  _MM_LOADU_INT32((const SIMDTYPE_INT32 *)(ptr_dense + k*handle->n + 2*1*SIMD_WIDTH_FP32));
          EXPAND_BFLOAT16(vload_1, v1_1, v2_1);
          _MM_STORE_FP32(scratch_B + k*num_regs*SIMD_WIDTH_FP32 + 2*1*SIMD_WIDTH_FP32, v1_1);
          _MM_STORE_FP32(scratch_B + k*num_regs*SIMD_WIDTH_FP32 + (2*1+1)*SIMD_WIDTH_FP32, v2_1);
          vload_2 =  _MM_LOADU_INT32((const SIMDTYPE_INT32 *)(ptr_dense + k*handle->n + 2*2*SIMD_WIDTH_FP32));
          EXPAND_BFLOAT16(vload_2, v1_2, v2_2);
          _MM_STORE_FP32(scratch_B + k*num_regs*SIMD_WIDTH_FP32 + 2*2*SIMD_WIDTH_FP32, v1_2);
          _MM_STORE_FP32(scratch_B + k*num_regs*SIMD_WIDTH_FP32 + (2*2+1)*SIMD_WIDTH_FP32, v2_2);
        }
      } else {
        for (k = 0; k < num_k; k++) {
          for (n = 0; n < num_full_regs; n+=2) {
            SIMDTYPE_INT32 vload_0 =  _MM_LOADU_INT32((const SIMDTYPE_INT32 *)(ptr_dense + k*handle->n + n*SIMD_WIDTH_FP32));
            SIMDTYPE_FP32 v1_0, v2_0;
            EXPAND_BFLOAT16(vload_0, v1_0, v2_0);
            _MM_STORE_FP32(scratch_B + k*num_regs*SIMD_WIDTH_FP32 + n*SIMD_WIDTH_FP32, v1_0);
            _MM_STORE_FP32(scratch_B + k*num_regs*SIMD_WIDTH_FP32 + (n+1)*SIMD_WIDTH_FP32, v2_0);
          }
          for (n = last_n_start; n < num_n; n++) {
            uint16_t restmp = ptr_dense[k*handle->n + n];
            union { int i; float f; } res;
            res.i = restmp;
            res.i <<= 16;
            {
              scratch_B[k*num_regs*SIMD_WIDTH_FP32 + n] = res.f;
            }
          }
        }
      }
    }
#if 0
    printf("B_col\n");
    for (k = 0; k < num_k; k++) {
      printf(" %lf ", ptr_dense[k*handle->n]);
    }
    printf("\n");
#endif
    scratch_C_base = scratch_C - m_overall_start*num_regs*SIMD_WIDTH_FP32;
    scratch_B_base = scratch_B; /* - k_overall_start*num_regs*SIMD_WIDTH_FP32; */

    for (m = m_overall_start; m < m_overall_start + num_m_aligned; m+=2, m_local+=2) {
      int start_j, end_j, end_j_2, num_j, num_j_2;
      const uint16_t*  LIBXSMM_RESTRICT sp_c_ptr_base;
      const uint16_t*  LIBXSMM_RESTRICT sp_c_ptr_base_2;
      const float* LIBXSMM_RESTRICT sp_v_ptr_base;
      const float* LIBXSMM_RESTRICT sp_v_ptr_base_2;
      float* const LIBXSMM_RESTRICT result_m_index = scratch_C_base + (m)*num_regs*SIMD_WIDTH_FP32;
      float* const LIBXSMM_RESTRICT result_m_index_2 = scratch_C_base + (m+1)*num_regs*SIMD_WIDTH_FP32;

      if ( m_local >= m_block_size) { block_A++; slice = A_sparse[block_A]; m_local = 0; }

      start_j =  slice.rowidx[m_local];
      end_j   =  slice.rowidx[m_local + 1];
      end_j_2 =  slice.rowidx[m_local + 2];
      num_j   = (end_j - start_j);
      num_j_2   = (end_j_2 - end_j);
      sp_c_ptr_base = slice.colidx + start_j;
      sp_c_ptr_base_2 = slice.colidx + end_j;
      sp_v_ptr_base = (float *)(slice.values) + start_j;
      sp_v_ptr_base_2 = (float *)(slice.values) + end_j;

      if (!last_block_n)
      {
        int64_t j = 0, j2 = 0;
        sum[0] = _MM_LOAD_FP32(result_m_index + 0*SIMD_WIDTH_FP32);
        sum[0+num_regs] = _MM_LOAD_FP32(result_m_index_2 + 0*SIMD_WIDTH_FP32);
        sum[1] = _MM_LOAD_FP32(result_m_index + 1*SIMD_WIDTH_FP32);
        sum[1+num_regs] = _MM_LOAD_FP32(result_m_index_2 + 1*SIMD_WIDTH_FP32);
        sum[2] = _MM_LOAD_FP32(result_m_index + 2*SIMD_WIDTH_FP32);
        sum[2+num_regs] = _MM_LOAD_FP32(result_m_index_2 + 2*SIMD_WIDTH_FP32);
        sum[3] = _MM_LOAD_FP32(result_m_index + 3*SIMD_WIDTH_FP32);
        sum[3+num_regs] = _MM_LOAD_FP32(result_m_index_2 + 3*SIMD_WIDTH_FP32);
        sum[4] = _MM_LOAD_FP32(result_m_index + 4*SIMD_WIDTH_FP32);
        sum[4+num_regs] = _MM_LOAD_FP32(result_m_index_2 + 4*SIMD_WIDTH_FP32);
        sum[5] = _MM_LOAD_FP32(result_m_index + 5*SIMD_WIDTH_FP32);
        sum[5+num_regs] = _MM_LOAD_FP32(result_m_index_2 + 5*SIMD_WIDTH_FP32);
        for (; j < num_j && j2 < num_j_2; j++, j2++) {
          const float* const LIBXSMM_RESTRICT sp_col_dense_index = scratch_B_base +  (unsigned int)sp_c_ptr_base[j]*num_regs*SIMD_WIDTH_FP32;
          const float* const LIBXSMM_RESTRICT sp_col_dense_index_2 = scratch_B_base + (unsigned int)sp_c_ptr_base_2[j2]*num_regs*SIMD_WIDTH_FP32;
          SIMDTYPE_FP32 v_v = _MM_SET1_FP32(sp_v_ptr_base[j]);
          SIMDTYPE_FP32 v_v_2 = _MM_SET1_FP32(sp_v_ptr_base_2[j2]);
          sum[0] = _MM_FMADD_FP32(v_v, _MM_LOAD_FP32(sp_col_dense_index + 0*SIMD_WIDTH_FP32), sum[0]);
          sum[0 + num_regs] = _MM_FMADD_FP32(v_v_2, _MM_LOAD_FP32(sp_col_dense_index_2 + 0*SIMD_WIDTH_FP32), sum[0+num_regs]);
          sum[1] = _MM_FMADD_FP32(v_v, _MM_LOAD_FP32(sp_col_dense_index + 1*SIMD_WIDTH_FP32), sum[1]);
          sum[1 + num_regs] = _MM_FMADD_FP32(v_v_2, _MM_LOAD_FP32(sp_col_dense_index_2 + 1*SIMD_WIDTH_FP32), sum[1+num_regs]);
          sum[2] = _MM_FMADD_FP32(v_v, _MM_LOAD_FP32(sp_col_dense_index + 2*SIMD_WIDTH_FP32), sum[2]);
          sum[2 + num_regs] = _MM_FMADD_FP32(v_v_2, _MM_LOAD_FP32(sp_col_dense_index_2 + 2*SIMD_WIDTH_FP32), sum[2+num_regs]);
          sum[3] = _MM_FMADD_FP32(v_v, _MM_LOAD_FP32(sp_col_dense_index + 3*SIMD_WIDTH_FP32), sum[3]);
          sum[3 + num_regs] = _MM_FMADD_FP32(v_v_2, _MM_LOAD_FP32(sp_col_dense_index_2 + 3*SIMD_WIDTH_FP32), sum[3+num_regs]);
          sum[4] = _MM_FMADD_FP32(v_v, _MM_LOAD_FP32(sp_col_dense_index + 4*SIMD_WIDTH_FP32), sum[4]);
          sum[4 + num_regs] = _MM_FMADD_FP32(v_v_2, _MM_LOAD_FP32(sp_col_dense_index_2 + 4*SIMD_WIDTH_FP32), sum[4+num_regs]);
          sum[5] = _MM_FMADD_FP32(v_v, _MM_LOAD_FP32(sp_col_dense_index + 5*SIMD_WIDTH_FP32), sum[5]);
          sum[5 + num_regs] = _MM_FMADD_FP32(v_v_2, _MM_LOAD_FP32(sp_col_dense_index_2 + 5*SIMD_WIDTH_FP32), sum[5+num_regs]);
        }
        for (; j < num_j; j++) {
          const float* const LIBXSMM_RESTRICT sp_col_dense_index = scratch_B_base +  (unsigned int)sp_c_ptr_base[j]*num_regs*SIMD_WIDTH_FP32;
          SIMDTYPE_FP32 v_v = _MM_SET1_FP32(sp_v_ptr_base[j]);
          sum[0] = _MM_FMADD_FP32(v_v, _MM_LOAD_FP32(sp_col_dense_index + 0*SIMD_WIDTH_FP32), sum[0]);
          sum[1] = _MM_FMADD_FP32(v_v, _MM_LOAD_FP32(sp_col_dense_index + 1*SIMD_WIDTH_FP32), sum[1]);
          sum[2] = _MM_FMADD_FP32(v_v, _MM_LOAD_FP32(sp_col_dense_index + 2*SIMD_WIDTH_FP32), sum[2]);
          sum[3] = _MM_FMADD_FP32(v_v, _MM_LOAD_FP32(sp_col_dense_index + 3*SIMD_WIDTH_FP32), sum[3]);
          sum[4] = _MM_FMADD_FP32(v_v, _MM_LOAD_FP32(sp_col_dense_index + 4*SIMD_WIDTH_FP32), sum[4]);
          sum[5] = _MM_FMADD_FP32(v_v, _MM_LOAD_FP32(sp_col_dense_index + 5*SIMD_WIDTH_FP32), sum[5]);
        }
        for (; j2 < num_j_2; j2++) {
          const float* const LIBXSMM_RESTRICT sp_col_dense_index_2 = scratch_B_base + (unsigned int)sp_c_ptr_base_2[j2]*num_regs*SIMD_WIDTH_FP32;
          SIMDTYPE_FP32 v_v_2 = _MM_SET1_FP32(sp_v_ptr_base_2[j2]);
          sum[0 + num_regs] = _MM_FMADD_FP32(v_v_2, _MM_LOAD_FP32(sp_col_dense_index_2 + 0*SIMD_WIDTH_FP32), sum[0+num_regs]);
          sum[1 + num_regs] = _MM_FMADD_FP32(v_v_2, _MM_LOAD_FP32(sp_col_dense_index_2 + 1*SIMD_WIDTH_FP32), sum[1+num_regs]);
          sum[2 + num_regs] = _MM_FMADD_FP32(v_v_2, _MM_LOAD_FP32(sp_col_dense_index_2 + 2*SIMD_WIDTH_FP32), sum[2+num_regs]);
          sum[3 + num_regs] = _MM_FMADD_FP32(v_v_2, _MM_LOAD_FP32(sp_col_dense_index_2 + 3*SIMD_WIDTH_FP32), sum[3+num_regs]);
          sum[4 + num_regs] = _MM_FMADD_FP32(v_v_2, _MM_LOAD_FP32(sp_col_dense_index_2 + 4*SIMD_WIDTH_FP32), sum[4+num_regs]);
          sum[5 + num_regs] = _MM_FMADD_FP32(v_v_2, _MM_LOAD_FP32(sp_col_dense_index_2 + 5*SIMD_WIDTH_FP32), sum[5+num_regs]);
        }
        _MM_STORE_FP32(result_m_index + 0*SIMD_WIDTH_FP32, sum[0]);
        _MM_STORE_FP32(result_m_index_2 + 0*SIMD_WIDTH_FP32, sum[0+num_regs]);
        _MM_STORE_FP32(result_m_index + 1*SIMD_WIDTH_FP32, sum[1]);
        _MM_STORE_FP32(result_m_index_2 + 1*SIMD_WIDTH_FP32, sum[1+num_regs]);
        _MM_STORE_FP32(result_m_index + 2*SIMD_WIDTH_FP32, sum[2]);
        _MM_STORE_FP32(result_m_index_2 + 2*SIMD_WIDTH_FP32, sum[2+num_regs]);
        _MM_STORE_FP32(result_m_index + 3*SIMD_WIDTH_FP32, sum[3]);
        _MM_STORE_FP32(result_m_index_2 + 3*SIMD_WIDTH_FP32, sum[3+num_regs]);
        _MM_STORE_FP32(result_m_index + 4*SIMD_WIDTH_FP32, sum[4]);
        _MM_STORE_FP32(result_m_index_2 + 4*SIMD_WIDTH_FP32, sum[4+num_regs]);
        _MM_STORE_FP32(result_m_index + 5*SIMD_WIDTH_FP32, sum[5]);
        _MM_STORE_FP32(result_m_index_2 + 5*SIMD_WIDTH_FP32, sum[5+num_regs]);
      }
      else {
        int64_t j = 0, j2 = 0;
        for (n = 0; n < num_full_regs; n+=2) {
          sum[n] = _MM_SETZERO_FP32();
          sum[n+num_regs] = _MM_SETZERO_FP32();
          sum[n+1] = _MM_SETZERO_FP32();
          sum[n+1+num_regs] = _MM_SETZERO_FP32();
        }
        for (; j < num_j && j2 < num_j_2; j++, j2++) {
          const float* const LIBXSMM_RESTRICT sp_col_dense_index = scratch_B_base +  (unsigned int)sp_c_ptr_base[j]*num_regs*SIMD_WIDTH_FP32;
          const float* const LIBXSMM_RESTRICT sp_col_dense_index_2 = scratch_B_base + (unsigned int)sp_c_ptr_base_2[j2]*num_regs*SIMD_WIDTH_FP32;
          SIMDTYPE_FP32 v_v = _MM_SET1_FP32(sp_v_ptr_base[j]);
          SIMDTYPE_FP32 v_v_2 = _MM_SET1_FP32(sp_v_ptr_base_2[j2]);
          for (n = 0; n < num_full_regs; n+=2) {
            sum[n] = _MM_FMADD_FP32(v_v, _MM_LOAD_FP32(sp_col_dense_index + n*SIMD_WIDTH_FP32), sum[n]);
            sum[n + num_regs] = _MM_FMADD_FP32(v_v_2, _MM_LOAD_FP32(sp_col_dense_index_2 + n*SIMD_WIDTH_FP32), sum[n+num_regs]);
            sum[n+1] = _MM_FMADD_FP32(v_v, _MM_LOAD_FP32(sp_col_dense_index + (n+1)*SIMD_WIDTH_FP32), sum[n+1]);
            sum[n+1 + num_regs] = _MM_FMADD_FP32(v_v_2, _MM_LOAD_FP32(sp_col_dense_index_2 + (n+1)*SIMD_WIDTH_FP32), sum[n+1+num_regs]);
          }
          {
            float v_v_f = sp_v_ptr_base[j];
            float v_v_f_2 = sp_v_ptr_base_2[j2];
            for (n = last_n_start; n < num_n; n++) {
              result_m_index[n] += sp_col_dense_index[n]*v_v_f;
              result_m_index_2[n] += sp_col_dense_index_2[n]*v_v_f_2;
            }
          }
        }
        for (; j < num_j; j++) {
          const float* const LIBXSMM_RESTRICT sp_col_dense_index = scratch_B_base +  (unsigned int)sp_c_ptr_base[j]*num_regs*SIMD_WIDTH_FP32;
          SIMDTYPE_FP32 v_v = _MM_SET1_FP32(sp_v_ptr_base[j]);
          for (n = 0; n < num_full_regs; n+=2) {
            sum[n] = _MM_FMADD_FP32(v_v, _MM_LOAD_FP32(sp_col_dense_index + n*SIMD_WIDTH_FP32), sum[n]);
            sum[n+1] = _MM_FMADD_FP32(v_v, _MM_LOAD_FP32(sp_col_dense_index + (n+1)*SIMD_WIDTH_FP32), sum[n+1]);
          }
          {
            float v_v_f = sp_v_ptr_base[j];
            for (n = last_n_start; n < num_n; n++) {
              result_m_index[n] += sp_col_dense_index[n]*v_v_f;
            }
          }
        }
        for (; j2 < num_j_2; j2++) {
          const float* const LIBXSMM_RESTRICT sp_col_dense_index_2 = scratch_B_base + (unsigned int)sp_c_ptr_base_2[j2]*num_regs*SIMD_WIDTH_FP32;
          SIMDTYPE_FP32 v_v_2 = _MM_SET1_FP32(sp_v_ptr_base_2[j2]);
          for (n = 0; n < num_full_regs; n+=2) {
            sum[n + num_regs] = _MM_FMADD_FP32(v_v_2, _MM_LOAD_FP32(sp_col_dense_index_2 + n*SIMD_WIDTH_FP32), sum[n+num_regs]);
            sum[n+1 + num_regs] = _MM_FMADD_FP32(v_v_2, _MM_LOAD_FP32(sp_col_dense_index_2 + (n+1)*SIMD_WIDTH_FP32), sum[n+1+num_regs]);
          }
          {
            float v_v_f_2 = sp_v_ptr_base_2[j2];
            for (n = last_n_start; n < num_n; n++) {
              result_m_index_2[n] += sp_col_dense_index_2[n]*v_v_f_2;
            }
          }
        }
        for (n = 0; n < num_full_regs; n+=2) {
          _MM_STORE_FP32(result_m_index + n*SIMD_WIDTH_FP32,  _MM_ADD_FP32(sum[n], _MM_LOAD_FP32(result_m_index + n*SIMD_WIDTH_FP32)));
          _MM_STORE_FP32(result_m_index_2 + n*SIMD_WIDTH_FP32,  _MM_ADD_FP32(sum[n+num_regs], _MM_LOAD_FP32(result_m_index_2 + n*SIMD_WIDTH_FP32)));
          _MM_STORE_FP32(result_m_index + (n+1)*SIMD_WIDTH_FP32,  _MM_ADD_FP32(sum[n+1], _MM_LOAD_FP32(result_m_index + (n+1)*SIMD_WIDTH_FP32)));
          _MM_STORE_FP32(result_m_index_2 + (n+1)*SIMD_WIDTH_FP32,  _MM_ADD_FP32(sum[n+1+num_regs], _MM_LOAD_FP32(result_m_index_2 + (n+1)*SIMD_WIDTH_FP32)));
        }
      }
    }
    for (m = m_overall_start + num_m_aligned; m < m_overall_end; m++, m_local++) {
      int start_j, end_j, num_j;
      const uint16_t*  LIBXSMM_RESTRICT sp_c_ptr_base;
      const float* LIBXSMM_RESTRICT sp_v_ptr_base;
      float* LIBXSMM_RESTRICT result_m_index;

      if ( m_local >= m_block_size) { block_A++; slice = A_sparse[block_A]; m_local = 0; }

      start_j =  slice.rowidx[m_local];
      end_j   =  slice.rowidx[m_local + 1];
      num_j   = (end_j - start_j);
      sp_c_ptr_base = slice.colidx + start_j;
      sp_v_ptr_base = slice.values + start_j;
      result_m_index = scratch_C_base + (m)*num_regs*SIMD_WIDTH_FP32;

      if (!last_block_n) {
        int64_t j = 0;
        sum[0] = _MM_LOAD_FP32(result_m_index + 0*SIMD_WIDTH_FP32);
        sum[1] = _MM_LOAD_FP32(result_m_index + 1*SIMD_WIDTH_FP32);
        sum[2] = _MM_LOAD_FP32(result_m_index + 2*SIMD_WIDTH_FP32);
        sum[3] = _MM_LOAD_FP32(result_m_index + 3*SIMD_WIDTH_FP32);
        sum[4] = _MM_LOAD_FP32(result_m_index + 4*SIMD_WIDTH_FP32);
        sum[5] = _MM_LOAD_FP32(result_m_index + 5*SIMD_WIDTH_FP32);
        for (; j < num_j; j++) {
          const float* const LIBXSMM_RESTRICT sp_col_dense_index = scratch_B_base +  (unsigned int)sp_c_ptr_base[j]*num_regs*SIMD_WIDTH_FP32;
          SIMDTYPE_FP32 v_v = _MM_SET1_FP32(sp_v_ptr_base[j]);
          sum[0] = _MM_FMADD_FP32(v_v, _MM_LOAD_FP32(sp_col_dense_index + 0*SIMD_WIDTH_FP32), sum[0]);
          sum[1] = _MM_FMADD_FP32(v_v, _MM_LOAD_FP32(sp_col_dense_index + 1*SIMD_WIDTH_FP32), sum[1]);
          sum[2] = _MM_FMADD_FP32(v_v, _MM_LOAD_FP32(sp_col_dense_index + 2*SIMD_WIDTH_FP32), sum[2]);
          sum[3] = _MM_FMADD_FP32(v_v, _MM_LOAD_FP32(sp_col_dense_index + 3*SIMD_WIDTH_FP32), sum[3]);
          sum[4] = _MM_FMADD_FP32(v_v, _MM_LOAD_FP32(sp_col_dense_index + 4*SIMD_WIDTH_FP32), sum[4]);
          sum[5] = _MM_FMADD_FP32(v_v, _MM_LOAD_FP32(sp_col_dense_index + 5*SIMD_WIDTH_FP32), sum[5]);
        }
        _MM_STORE_FP32(result_m_index + 0*SIMD_WIDTH_FP32, sum[0]);
        _MM_STORE_FP32(result_m_index + 1*SIMD_WIDTH_FP32, sum[1]);
        _MM_STORE_FP32(result_m_index + 2*SIMD_WIDTH_FP32, sum[2]);
        _MM_STORE_FP32(result_m_index + 3*SIMD_WIDTH_FP32, sum[3]);
        _MM_STORE_FP32(result_m_index + 4*SIMD_WIDTH_FP32, sum[4]);
        _MM_STORE_FP32(result_m_index + 5*SIMD_WIDTH_FP32, sum[5]);
      }
      else {
        int64_t j = 0;
        for (n = 0; n < num_full_regs; n+=2) {
          sum[n] = _MM_SETZERO_FP32();
          sum[n+1] = _MM_SETZERO_FP32();
        }
        for (; j < num_j; j++) {
          const float* const LIBXSMM_RESTRICT sp_col_dense_index = scratch_B_base +  (unsigned int)sp_c_ptr_base[j]*num_regs*SIMD_WIDTH_FP32;
          SIMDTYPE_FP32 v_v = _MM_SET1_FP32(sp_v_ptr_base[j]);
          for (n = 0; n < num_full_regs; n+=2) {
            sum[n] = _MM_FMADD_FP32(v_v, _MM_LOAD_FP32(sp_col_dense_index + n*SIMD_WIDTH_FP32), sum[n]);
            sum[n+1] = _MM_FMADD_FP32(v_v, _MM_LOAD_FP32(sp_col_dense_index + (n+1)*SIMD_WIDTH_FP32), sum[n+1]);
          }
          {
            float v_v_f = sp_v_ptr_base[j];
            for (n = last_n_start; n < num_n; n++) {
              result_m_index[n] += sp_col_dense_index[n]*v_v_f;
            }
          }
        }
        for (n = 0; n < num_full_regs; n+=2) {
          _MM_STORE_FP32(result_m_index + n*SIMD_WIDTH_FP32, _MM_ADD_FP32(sum[n], _MM_LOAD_FP32(result_m_index + n*SIMD_WIDTH_FP32)));
          _MM_STORE_FP32(result_m_index + (n+1)*SIMD_WIDTH_FP32, _MM_ADD_FP32(sum[n+1], _MM_LOAD_FP32(result_m_index + (n+1)*SIMD_WIDTH_FP32)));
        }

      }
    }
  } /* kb */
#if 0
  for (m = 0; m < 3; m++) {
    for (n = 0; n < num_n; n++) {
      printf("%f ", scratch_C[m*num_regs*SIMD_WIDTH_FP32 + n]);
    }
    printf("\n");
  }
#endif
  /* Copy out C matrix */

  if(transC == 'Y') {
    int num_m_simd = num_m / SIMD_WIDTH_FP32 * SIMD_WIDTH_FP32;
    int num_n_simd = num_n / SIMD_WIDTH_FP32 * SIMD_WIDTH_FP32;
    int n2;

    ptr_result = C + n_overall_start*handle->m + m_overall_start;
    for(n = 0; n < num_n_simd; n+=SIMD_WIDTH_FP32){
      for(m = 0; m < num_m_simd; m+=SIMD_WIDTH_FP32){
        TRANSPOSE_SIMD_WIDTH_KERNEL(scratch_C + m*n_block_size + n, n_block_size, ptr_result + n*handle->m + m, handle->m);
      }
      /* Transpose a SIMD_WIDTH_FP32 * (num_m - num_m_simd) block of output space - input is of size (num_m - num_m_simd) * SIMD_WIDTH_FP32 */
      for(n2 = n; n2 < n + SIMD_WIDTH_FP32; n2++) {
        for(m = num_m_simd; m < num_m; m++){
          ptr_result[n2*handle->m + m] = scratch_C[m*n_block_size + n2]; 
        }
      }
    }
    /* Transpose a (num_n - num_n_simd) * num_m block of output space - input is of size num_m * (num_n - num_n_simd) */
    for(n = num_n_simd; n < num_n; n++){
      for(m = 0; m < num_m; m++){
        ptr_result[n*handle->m + m] = scratch_C[m*n_block_size + n];
      }
    }
  }
  else {
    if (!last_block_n) {
      for (m = 0; m < num_m; m++) {
        _MM_STOREU_FP32(ptr_result + m*handle->n + 0*SIMD_WIDTH_FP32, _MM_LOAD_FP32(scratch_C + m*num_regs*SIMD_WIDTH_FP32 + 0*SIMD_WIDTH_FP32));
        _MM_STOREU_FP32(ptr_result + m*handle->n + 1*SIMD_WIDTH_FP32, _MM_LOAD_FP32(scratch_C + m*num_regs*SIMD_WIDTH_FP32 + 1*SIMD_WIDTH_FP32));
        _MM_STOREU_FP32(ptr_result + m*handle->n + 2*SIMD_WIDTH_FP32, _MM_LOAD_FP32(scratch_C + m*num_regs*SIMD_WIDTH_FP32 + 2*SIMD_WIDTH_FP32));
        _MM_STOREU_FP32(ptr_result + m*handle->n + 3*SIMD_WIDTH_FP32, _MM_LOAD_FP32(scratch_C + m*num_regs*SIMD_WIDTH_FP32 + 3*SIMD_WIDTH_FP32));
        _MM_STOREU_FP32(ptr_result + m*handle->n + 4*SIMD_WIDTH_FP32, _MM_LOAD_FP32(scratch_C + m*num_regs*SIMD_WIDTH_FP32 + 4*SIMD_WIDTH_FP32));
        _MM_STOREU_FP32(ptr_result + m*handle->n + 5*SIMD_WIDTH_FP32, _MM_LOAD_FP32(scratch_C + m*num_regs*SIMD_WIDTH_FP32 + 5*SIMD_WIDTH_FP32));
      }
    }
    else {
      for (m = 0; m < num_m; m++) {
        for (n = 0; n < num_full_regs; n+=2) {
          _MM_STOREU_FP32(ptr_result + m*handle->n + n*SIMD_WIDTH_FP32, _MM_LOAD_FP32(scratch_C + m*num_regs*SIMD_WIDTH_FP32 + n*SIMD_WIDTH_FP32));
          _MM_STOREU_FP32(ptr_result + m*handle->n + (n+1)*SIMD_WIDTH_FP32, _MM_LOAD_FP32(scratch_C + m*num_regs*SIMD_WIDTH_FP32 + (n+1)*SIMD_WIDTH_FP32));
        }
        for (n = last_n_start; n < num_n; n++) {
          ptr_result[m*handle->n + n] = scratch_C[m*num_regs*SIMD_WIDTH_FP32 + n];
        }
      }
    }
  }
}

