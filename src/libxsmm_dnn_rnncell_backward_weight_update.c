/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Heinecke, Kunal Banerjee, Evangelos Georganas (Intel Corp.)
******************************************************************************/
#include "libxsmm_dnn_rnncell_backward_weight_update.h"
#include "libxsmm_dnn_elementwise.h"
#include "libxsmm_main.h"

LIBXSMM_API_INLINE LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512_CORE)
void trans_act(short int *in, short int *out)
{
#if defined(LIBXSMM_INTRINSICS_AVX512_CORE)
  __m512i r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, ra, rb, rc, rd, re, rf;
  __m512i t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, ta, tb, tc, td, te, tf;
  __m512i v0, v1, v2, v3, v4, v5, v6, v7;
  const __m512i idx_v  = _mm512_set_epi64(13, 12, 7, 6, 9, 8, 3, 2);
  const __mmask8 mask0 = LIBXSMM_INTRINSICS_MM512_CVTU32_MASK8(204);
  const __mmask8 mask1 = LIBXSMM_INTRINSICS_MM512_CVTU32_MASK8(51);
  const int in_width = 32, out_width = 32;

  r0 = _mm512_loadu_si512(in + 0*in_width);
  r1 = _mm512_loadu_si512(in + 1*in_width);
  t0 = _mm512_unpacklo_epi16(r0,r1);
  t1 = _mm512_unpackhi_epi16(r0,r1);
  r2 = _mm512_loadu_si512(in + 2*in_width);
  r3 = _mm512_loadu_si512(in + 3*in_width);
  t2 = _mm512_unpacklo_epi16(r2,r3);
  t3 = _mm512_unpackhi_epi16(r2,r3);
  r4 = _mm512_loadu_si512(in + 4*in_width);
  r5 = _mm512_loadu_si512(in + 5*in_width);
  t4 = _mm512_unpacklo_epi16(r4,r5);
  t5 = _mm512_unpackhi_epi16(r4,r5);
  r6 = _mm512_loadu_si512(in + 6*in_width);
  r7 = _mm512_loadu_si512(in + 7*in_width);
  t6 = _mm512_unpacklo_epi16(r6,r7);
  t7 = _mm512_unpackhi_epi16(r6,r7);
  r8 = _mm512_loadu_si512(in + 8*in_width);
  r9 = _mm512_loadu_si512(in + 9*in_width);
  t8 = _mm512_unpacklo_epi16(r8,r9);
  t9 = _mm512_unpackhi_epi16(r8,r9);
  ra = _mm512_loadu_si512(in + 10*in_width);
  rb = _mm512_loadu_si512(in + 11*in_width);
  ta = _mm512_unpacklo_epi16(ra,rb);
  tb = _mm512_unpackhi_epi16(ra,rb);
  rc = _mm512_loadu_si512(in + 12*in_width);
  rd = _mm512_loadu_si512(in + 13*in_width);
  tc = _mm512_unpacklo_epi16(rc,rd);
  td = _mm512_unpackhi_epi16(rc,rd);
  re = _mm512_loadu_si512(in + 14*in_width);
  rf = _mm512_loadu_si512(in + 15*in_width);
  te = _mm512_unpacklo_epi16(re,rf);
  tf = _mm512_unpackhi_epi16(re,rf);

  r0 = _mm512_unpacklo_epi32(t0,t2);
  r1 = _mm512_unpackhi_epi32(t0,t2);
  r2 = _mm512_unpacklo_epi32(t1,t3);
  r3 = _mm512_unpackhi_epi32(t1,t3);
  r4 = _mm512_unpacklo_epi32(t4,t6);
  r5 = _mm512_unpackhi_epi32(t4,t6);
  r6 = _mm512_unpacklo_epi32(t5,t7);
  r7 = _mm512_unpackhi_epi32(t5,t7);
  r8 = _mm512_unpacklo_epi32(t8,ta);
  r9 = _mm512_unpackhi_epi32(t8,ta);
  ra = _mm512_unpacklo_epi32(t9,tb);
  rb = _mm512_unpackhi_epi32(t9,tb);
  rc = _mm512_unpacklo_epi32(tc,te);
  rd = _mm512_unpackhi_epi32(tc,te);
  re = _mm512_unpacklo_epi32(td,tf);
  rf = _mm512_unpackhi_epi32(td,tf);

  t0 = _mm512_unpacklo_epi64(r0,r4);
  t1 = _mm512_unpackhi_epi64(r0,r4);
  t2 = _mm512_unpacklo_epi64(r1,r5);
  t3 = _mm512_unpackhi_epi64(r1,r5);
  t4 = _mm512_unpacklo_epi64(r2,r6);
  t5 = _mm512_unpackhi_epi64(r2,r6);
  t6 = _mm512_unpacklo_epi64(r3,r7);
  t7 = _mm512_unpackhi_epi64(r3,r7);
  t8 = _mm512_unpacklo_epi64(r8,rc);
  t9 = _mm512_unpackhi_epi64(r8,rc);
  ta = _mm512_unpacklo_epi64(r9,rd);
  tb = _mm512_unpackhi_epi64(r9,rd);
  tc = _mm512_unpacklo_epi64(ra,re);
  td = _mm512_unpackhi_epi64(ra,re);
  te = _mm512_unpacklo_epi64(rb,rf);
  tf = _mm512_unpackhi_epi64(rb,rf);

  r0 = _mm512_shuffle_i32x4(t0, t1, 0x88);
  r1 = _mm512_shuffle_i32x4(t2, t3, 0x88);
  r2 = _mm512_shuffle_i32x4(t4, t5, 0x88);
  r3 = _mm512_shuffle_i32x4(t6, t7, 0x88);
  r4 = _mm512_shuffle_i32x4(t0, t1, 0xdd);
  r5 = _mm512_shuffle_i32x4(t2, t3, 0xdd);
  r6 = _mm512_shuffle_i32x4(t4, t5, 0xdd);
  r7 = _mm512_shuffle_i32x4(t6, t7, 0xdd);
  r8 = _mm512_shuffle_i32x4(t8, t9, 0x88);
  r9 = _mm512_shuffle_i32x4(ta, tb, 0x88);
  ra = _mm512_shuffle_i32x4(tc, td, 0x88);
  rb = _mm512_shuffle_i32x4(te, tf, 0x88);
  rc = _mm512_shuffle_i32x4(t8, t9, 0xdd);
  rd = _mm512_shuffle_i32x4(ta, tb, 0xdd);
  re = _mm512_shuffle_i32x4(tc, td, 0xdd);
  rf = _mm512_shuffle_i32x4(te, tf, 0xdd);

  v0 = _mm512_permutex2var_epi64(r0, idx_v, r8);
  t0 = _mm512_mask_blend_epi64( mask0, r0, v0);
  _mm256_storeu_si256((__m256i*)(out + 0*out_width), _mm512_extracti64x4_epi64(t0, 0));
  _mm256_storeu_si256((__m256i*)(out + 1*out_width), _mm512_extracti64x4_epi64(t0, 1));
  t8 = _mm512_mask_blend_epi64( mask1, r8, v0);
  _mm256_storeu_si256((__m256i*)(out + 16*out_width), _mm512_extracti64x4_epi64(t8, 0));
  _mm256_storeu_si256((__m256i*)(out + 17*out_width), _mm512_extracti64x4_epi64(t8, 1));
  v1 = _mm512_permutex2var_epi64(r1, idx_v, r9);
  t1 = _mm512_mask_blend_epi64( mask0, r1, v1);
  _mm256_storeu_si256((__m256i*)(out + 2*out_width), _mm512_extracti64x4_epi64(t1, 0));
  _mm256_storeu_si256((__m256i*)(out + 3*out_width), _mm512_extracti64x4_epi64(t1, 1));
  t9 = _mm512_mask_blend_epi64( mask1, r9, v1);
  _mm256_storeu_si256((__m256i*)(out + 18*out_width), _mm512_extracti64x4_epi64(t9, 0));
  _mm256_storeu_si256((__m256i*)(out + 19*out_width), _mm512_extracti64x4_epi64(t9, 1));
  v2 = _mm512_permutex2var_epi64(r2, idx_v, ra);
  t2 = _mm512_mask_blend_epi64( mask0, r2, v2);
  _mm256_storeu_si256((__m256i*)(out + 4*out_width), _mm512_extracti64x4_epi64(t2, 0));
  _mm256_storeu_si256((__m256i*)(out + 5*out_width), _mm512_extracti64x4_epi64(t2, 1));
  ta = _mm512_mask_blend_epi64( mask1, ra, v2);
  _mm256_storeu_si256((__m256i*)(out + 20*out_width), _mm512_extracti64x4_epi64(ta, 0));
  _mm256_storeu_si256((__m256i*)(out + 21*out_width), _mm512_extracti64x4_epi64(ta, 1));
  v3 = _mm512_permutex2var_epi64(r3, idx_v, rb);
  t3 = _mm512_mask_blend_epi64( mask0, r3, v3);
  _mm256_storeu_si256((__m256i*)(out + 6*out_width), _mm512_extracti64x4_epi64(t3, 0));
  _mm256_storeu_si256((__m256i*)(out + 7*out_width), _mm512_extracti64x4_epi64(t3, 1));
  tb = _mm512_mask_blend_epi64( mask1, rb, v3);
  _mm256_storeu_si256((__m256i*)(out + 22*out_width), _mm512_extracti64x4_epi64(tb, 0));
  _mm256_storeu_si256((__m256i*)(out + 23*out_width), _mm512_extracti64x4_epi64(tb, 1));
  v4 = _mm512_permutex2var_epi64(r4, idx_v, rc);
  t4 = _mm512_mask_blend_epi64( mask0, r4, v4);
  _mm256_storeu_si256((__m256i*)(out + 8*out_width), _mm512_extracti64x4_epi64(t4, 0));
  _mm256_storeu_si256((__m256i*)(out + 9*out_width), _mm512_extracti64x4_epi64(t4, 1));
  tc = _mm512_mask_blend_epi64( mask1, rc, v4);
  _mm256_storeu_si256((__m256i*)(out + 24*out_width), _mm512_extracti64x4_epi64(tc, 0));
  _mm256_storeu_si256((__m256i*)(out + 25*out_width), _mm512_extracti64x4_epi64(tc, 1));
  v5 = _mm512_permutex2var_epi64(r5, idx_v, rd);
  t5 = _mm512_mask_blend_epi64( mask0, r5, v5);
  _mm256_storeu_si256((__m256i*)(out + 10*out_width), _mm512_extracti64x4_epi64(t5, 0));
  _mm256_storeu_si256((__m256i*)(out + 11*out_width), _mm512_extracti64x4_epi64(t5, 1));
  td = _mm512_mask_blend_epi64( mask1, rd, v5);
  _mm256_storeu_si256((__m256i*)(out + 26*out_width), _mm512_extracti64x4_epi64(td, 0));
  _mm256_storeu_si256((__m256i*)(out + 27*out_width), _mm512_extracti64x4_epi64(td, 1));
  v6 = _mm512_permutex2var_epi64(r6, idx_v, re);
  t6 = _mm512_mask_blend_epi64( mask0, r6, v6);
  _mm256_storeu_si256((__m256i*)(out + 12*out_width), _mm512_extracti64x4_epi64(t6, 0));
  _mm256_storeu_si256((__m256i*)(out + 13*out_width), _mm512_extracti64x4_epi64(t6, 1));
  te = _mm512_mask_blend_epi64( mask1, re, v6);
  _mm256_storeu_si256((__m256i*)(out + 28*out_width), _mm512_extracti64x4_epi64(te, 0));
  _mm256_storeu_si256((__m256i*)(out + 29*out_width), _mm512_extracti64x4_epi64(te, 1));
  v7 = _mm512_permutex2var_epi64(r7, idx_v, rf);
  t7 = _mm512_mask_blend_epi64( mask0, r7, v7);
  _mm256_storeu_si256((__m256i*)(out + 14*out_width), _mm512_extracti64x4_epi64(t7, 0));
  _mm256_storeu_si256((__m256i*)(out + 15*out_width), _mm512_extracti64x4_epi64(t7, 1));
  tf = _mm512_mask_blend_epi64( mask1, rf, v7);
  _mm256_storeu_si256((__m256i*)(out + 30*out_width), _mm512_extracti64x4_epi64(tf, 0));
  _mm256_storeu_si256((__m256i*)(out + 31*out_width), _mm512_extracti64x4_epi64(tf, 1));

  r0 = _mm512_loadu_si512(in + 16*32 + 0*in_width);
  r1 = _mm512_loadu_si512(in + 16*32 + 1*in_width);
  t0 = _mm512_unpacklo_epi16(r0,r1);
  t1 = _mm512_unpackhi_epi16(r0,r1);
  r2 = _mm512_loadu_si512(in + 16*32 + 2*in_width);
  r3 = _mm512_loadu_si512(in + 16*32 + 3*in_width);
  t2 = _mm512_unpacklo_epi16(r2,r3);
  t3 = _mm512_unpackhi_epi16(r2,r3);
  r4 = _mm512_loadu_si512(in + 16*32 + 4*in_width);
  r5 = _mm512_loadu_si512(in + 16*32 + 5*in_width);
  t4 = _mm512_unpacklo_epi16(r4,r5);
  t5 = _mm512_unpackhi_epi16(r4,r5);
  r6 = _mm512_loadu_si512(in + 16*32 + 6*in_width);
  r7 = _mm512_loadu_si512(in + 16*32 + 7*in_width);
  t6 = _mm512_unpacklo_epi16(r6,r7);
  t7 = _mm512_unpackhi_epi16(r6,r7);
  r8 = _mm512_loadu_si512(in + 16*32 + 8*in_width);
  r9 = _mm512_loadu_si512(in + 16*32 + 9*in_width);
  t8 = _mm512_unpacklo_epi16(r8,r9);
  t9 = _mm512_unpackhi_epi16(r8,r9);
  ra = _mm512_loadu_si512(in + 16*32 + 10*in_width);
  rb = _mm512_loadu_si512(in + 16*32 + 11*in_width);
  ta = _mm512_unpacklo_epi16(ra,rb);
  tb = _mm512_unpackhi_epi16(ra,rb);
  rc = _mm512_loadu_si512(in + 16*32 + 12*in_width);
  rd = _mm512_loadu_si512(in + 16*32 + 13*in_width);
  tc = _mm512_unpacklo_epi16(rc,rd);
  td = _mm512_unpackhi_epi16(rc,rd);
  re = _mm512_loadu_si512(in + 16*32 + 14*in_width);
  rf = _mm512_loadu_si512(in + 16*32 + 15*in_width);
  te = _mm512_unpacklo_epi16(re,rf);
  tf = _mm512_unpackhi_epi16(re,rf);

  r0 = _mm512_unpacklo_epi32(t0,t2);
  r1 = _mm512_unpackhi_epi32(t0,t2);
  r2 = _mm512_unpacklo_epi32(t1,t3);
  r3 = _mm512_unpackhi_epi32(t1,t3);
  r4 = _mm512_unpacklo_epi32(t4,t6);
  r5 = _mm512_unpackhi_epi32(t4,t6);
  r6 = _mm512_unpacklo_epi32(t5,t7);
  r7 = _mm512_unpackhi_epi32(t5,t7);
  r8 = _mm512_unpacklo_epi32(t8,ta);
  r9 = _mm512_unpackhi_epi32(t8,ta);
  ra = _mm512_unpacklo_epi32(t9,tb);
  rb = _mm512_unpackhi_epi32(t9,tb);
  rc = _mm512_unpacklo_epi32(tc,te);
  rd = _mm512_unpackhi_epi32(tc,te);
  re = _mm512_unpacklo_epi32(td,tf);
  rf = _mm512_unpackhi_epi32(td,tf);

  t0 = _mm512_unpacklo_epi64(r0,r4);
  t1 = _mm512_unpackhi_epi64(r0,r4);
  t2 = _mm512_unpacklo_epi64(r1,r5);
  t3 = _mm512_unpackhi_epi64(r1,r5);
  t4 = _mm512_unpacklo_epi64(r2,r6);
  t5 = _mm512_unpackhi_epi64(r2,r6);
  t6 = _mm512_unpacklo_epi64(r3,r7);
  t7 = _mm512_unpackhi_epi64(r3,r7);
  t8 = _mm512_unpacklo_epi64(r8,rc);
  t9 = _mm512_unpackhi_epi64(r8,rc);
  ta = _mm512_unpacklo_epi64(r9,rd);
  tb = _mm512_unpackhi_epi64(r9,rd);
  tc = _mm512_unpacklo_epi64(ra,re);
  td = _mm512_unpackhi_epi64(ra,re);
  te = _mm512_unpacklo_epi64(rb,rf);
  tf = _mm512_unpackhi_epi64(rb,rf);

  r0 = _mm512_shuffle_i32x4(t0, t1, 0x88);
  r1 = _mm512_shuffle_i32x4(t2, t3, 0x88);
  r2 = _mm512_shuffle_i32x4(t4, t5, 0x88);
  r3 = _mm512_shuffle_i32x4(t6, t7, 0x88);
  r4 = _mm512_shuffle_i32x4(t0, t1, 0xdd);
  r5 = _mm512_shuffle_i32x4(t2, t3, 0xdd);
  r6 = _mm512_shuffle_i32x4(t4, t5, 0xdd);
  r7 = _mm512_shuffle_i32x4(t6, t7, 0xdd);
  r8 = _mm512_shuffle_i32x4(t8, t9, 0x88);
  r9 = _mm512_shuffle_i32x4(ta, tb, 0x88);
  ra = _mm512_shuffle_i32x4(tc, td, 0x88);
  rb = _mm512_shuffle_i32x4(te, tf, 0x88);
  rc = _mm512_shuffle_i32x4(t8, t9, 0xdd);
  rd = _mm512_shuffle_i32x4(ta, tb, 0xdd);
  re = _mm512_shuffle_i32x4(tc, td, 0xdd);
  rf = _mm512_shuffle_i32x4(te, tf, 0xdd);

  v0 = _mm512_permutex2var_epi64(r0, idx_v, r8);
  t0 = _mm512_mask_blend_epi64( mask0, r0, v0);
  _mm256_storeu_si256((__m256i*)(out + 16 + 0*out_width), _mm512_extracti64x4_epi64(t0, 0));
  _mm256_storeu_si256((__m256i*)(out + 16 + 1*out_width), _mm512_extracti64x4_epi64(t0, 1));
  t8 = _mm512_mask_blend_epi64( mask1, r8, v0);
  _mm256_storeu_si256((__m256i*)(out + 16 + 16*out_width), _mm512_extracti64x4_epi64(t8, 0));
  _mm256_storeu_si256((__m256i*)(out + 16 + 17*out_width), _mm512_extracti64x4_epi64(t8, 1));
  v1 = _mm512_permutex2var_epi64(r1, idx_v, r9);
  t1 = _mm512_mask_blend_epi64( mask0, r1, v1);
  _mm256_storeu_si256((__m256i*)(out + 16 + 2*out_width), _mm512_extracti64x4_epi64(t1, 0));
  _mm256_storeu_si256((__m256i*)(out + 16 + 3*out_width), _mm512_extracti64x4_epi64(t1, 1));
  t9 = _mm512_mask_blend_epi64( mask1, r9, v1);
  _mm256_storeu_si256((__m256i*)(out + 16 + 18*out_width), _mm512_extracti64x4_epi64(t9, 0));
  _mm256_storeu_si256((__m256i*)(out + 16 + 19*out_width), _mm512_extracti64x4_epi64(t9, 1));
  v2 = _mm512_permutex2var_epi64(r2, idx_v, ra);
  t2 = _mm512_mask_blend_epi64( mask0, r2, v2);
  _mm256_storeu_si256((__m256i*)(out + 16 + 4*out_width), _mm512_extracti64x4_epi64(t2, 0));
  _mm256_storeu_si256((__m256i*)(out + 16 + 5*out_width), _mm512_extracti64x4_epi64(t2, 1));
  ta = _mm512_mask_blend_epi64( mask1, ra, v2);
  _mm256_storeu_si256((__m256i*)(out + 16 + 20*out_width), _mm512_extracti64x4_epi64(ta, 0));
  _mm256_storeu_si256((__m256i*)(out + 16 + 21*out_width), _mm512_extracti64x4_epi64(ta, 1));
  v3 = _mm512_permutex2var_epi64(r3, idx_v, rb);
  t3 = _mm512_mask_blend_epi64( mask0, r3, v3);
  _mm256_storeu_si256((__m256i*)(out + 16 + 6*out_width), _mm512_extracti64x4_epi64(t3, 0));
  _mm256_storeu_si256((__m256i*)(out + 16 + 7*out_width), _mm512_extracti64x4_epi64(t3, 1));
  tb = _mm512_mask_blend_epi64( mask1, rb, v3);
  _mm256_storeu_si256((__m256i*)(out + 16 + 22*out_width), _mm512_extracti64x4_epi64(tb, 0));
  _mm256_storeu_si256((__m256i*)(out + 16 + 23*out_width), _mm512_extracti64x4_epi64(tb, 1));
  v4 = _mm512_permutex2var_epi64(r4, idx_v, rc);
  t4 = _mm512_mask_blend_epi64( mask0, r4, v4);
  _mm256_storeu_si256((__m256i*)(out + 16 + 8*out_width), _mm512_extracti64x4_epi64(t4, 0));
  _mm256_storeu_si256((__m256i*)(out + 16 + 9*out_width), _mm512_extracti64x4_epi64(t4, 1));
  tc = _mm512_mask_blend_epi64( mask1, rc, v4);
  _mm256_storeu_si256((__m256i*)(out + 16 + 24*out_width), _mm512_extracti64x4_epi64(tc, 0));
  _mm256_storeu_si256((__m256i*)(out + 16 + 25*out_width), _mm512_extracti64x4_epi64(tc, 1));
  v5 = _mm512_permutex2var_epi64(r5, idx_v, rd);
  t5 = _mm512_mask_blend_epi64( mask0, r5, v5);
  _mm256_storeu_si256((__m256i*)(out + 16 + 10*out_width), _mm512_extracti64x4_epi64(t5, 0));
  _mm256_storeu_si256((__m256i*)(out + 16 + 11*out_width), _mm512_extracti64x4_epi64(t5, 1));
  td = _mm512_mask_blend_epi64( mask1, rd, v5);
  _mm256_storeu_si256((__m256i*)(out + 16 + 26*out_width), _mm512_extracti64x4_epi64(td, 0));
  _mm256_storeu_si256((__m256i*)(out + 16 + 27*out_width), _mm512_extracti64x4_epi64(td, 1));
  v6 = _mm512_permutex2var_epi64(r6, idx_v, re);
  t6 = _mm512_mask_blend_epi64( mask0, r6, v6);
  _mm256_storeu_si256((__m256i*)(out + 16 + 12*out_width), _mm512_extracti64x4_epi64(t6, 0));
  _mm256_storeu_si256((__m256i*)(out + 16 + 13*out_width), _mm512_extracti64x4_epi64(t6, 1));
  te = _mm512_mask_blend_epi64( mask1, re, v6);
  _mm256_storeu_si256((__m256i*)(out + 16 + 28*out_width), _mm512_extracti64x4_epi64(te, 0));
  _mm256_storeu_si256((__m256i*)(out + 16 + 29*out_width), _mm512_extracti64x4_epi64(te, 1));
  v7 = _mm512_permutex2var_epi64(r7, idx_v, rf);
  t7 = _mm512_mask_blend_epi64( mask0, r7, v7);
  _mm256_storeu_si256((__m256i*)(out + 16 + 14*out_width), _mm512_extracti64x4_epi64(t7, 0));
  _mm256_storeu_si256((__m256i*)(out + 16 + 15*out_width), _mm512_extracti64x4_epi64(t7, 1));
  tf = _mm512_mask_blend_epi64( mask1, rf, v7);
  _mm256_storeu_si256((__m256i*)(out + 16 + 30*out_width), _mm512_extracti64x4_epi64(tf, 0));
  _mm256_storeu_si256((__m256i*)(out + 16 + 31*out_width), _mm512_extracti64x4_epi64(tf, 1));
#else
 LIBXSMM_UNUSED(in); LIBXSMM_UNUSED(out);
#endif
}

LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_rnncell_st_bwdupd_nc_ck_f32_f32(libxsmm_dnn_rnncell* handle, libxsmm_dnn_compute_kind kind, int start_thread, int tid);
LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_rnncell_st_bwdupd_nc_ck_bf16_bf16(libxsmm_dnn_rnncell* handle, libxsmm_dnn_compute_kind kind, int start_thread, int tid);
LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_rnncell_st_bwdupd_nc_ck_bf16_bf16_emu(libxsmm_dnn_rnncell* handle, libxsmm_dnn_compute_kind kind, int start_thread, int tid);
LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_rnncell_st_bwdupd_nc_ck_bf16_bf16_amx(libxsmm_dnn_rnncell* handle, libxsmm_dnn_compute_kind kind, int start_thread, int tid);
LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_rnncell_st_bwdupd_nc_kcck_bf16_bf16(libxsmm_dnn_rnncell* handle, libxsmm_dnn_compute_kind kind, int start_thread, int tid);
LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_rnncell_st_bwdupd_nc_kcck_bf16_bf16_emu(libxsmm_dnn_rnncell* handle, libxsmm_dnn_compute_kind kind, int start_thread, int tid);
LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_rnncell_st_bwdupd_nc_kcck_bf16_bf16_amx(libxsmm_dnn_rnncell* handle, libxsmm_dnn_compute_kind kind, int start_thread, int tid);
LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_rnncell_st_bwdupd_nc_kcck_f32_f32(libxsmm_dnn_rnncell* handle, libxsmm_dnn_compute_kind kind, int start_thread, int tid);
LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_rnncell_st_bwdupd_ncnc_kcck_f32_f32(libxsmm_dnn_rnncell* handle, libxsmm_dnn_compute_kind kind, int start_thread, int tid);
LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_rnncell_st_bwdupd_ncnc_kcck_bf16_bf16_amx(libxsmm_dnn_rnncell* handle, libxsmm_dnn_compute_kind kind, int start_thread, int tid);

LIBXSMM_API_INTERN LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512)
libxsmm_dnn_err_t libxsmm_dnn_rnncell_st_bwdupd_nc_ck_f32_f32(libxsmm_dnn_rnncell* handle, libxsmm_dnn_compute_kind kind, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
#if defined(LIBXSMM_INTRINSICS_AVX512) /*__AVX512F__*/
#define LIBXSMM_RNN_CELL_AVX512
  typedef float element_input_type;
  typedef float element_output_type;
  typedef float element_filter_type;
  if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_RNN_RELU ) {
# define LIBXSMM_DNN_RNN_RELU_BWDUPD
# include "template/libxsmm_dnn_rnncell_st_rnn_bwdupd_nc_ck_generic.tpl.c"
# undef LIBXSMM_DNN_RNN_RELU_BWDUPD
  } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_RNN_SIGMOID ) {
# define LIBXSMM_DNN_RNN_SIGMOID_BWDUPD
# include "template/libxsmm_dnn_rnncell_st_rnn_bwdupd_nc_ck_generic.tpl.c"
# undef LIBXSMM_DNN_RNN_SIGMOID_BWDUPD
  } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_RNN_TANH ) {
# define LIBXSMM_DNN_RNN_TANH_BWDUPD
# include "template/libxsmm_dnn_rnncell_st_rnn_bwdupd_nc_ck_generic.tpl.c"
# undef LIBXSMM_DNN_RNN_TANH_BWDUPD
  } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_LSTM ) {
# include "template/libxsmm_dnn_rnncell_st_lstm_bwdupd_nc_ck_generic.tpl.c"
  } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_GRU ) {
# include "template/libxsmm_dnn_rnncell_st_gru_bwdupd_nc_ck_generic.tpl.c"
  } else {
    /* should not happen */
  }
#undef LIBXSMM_RNN_CELL_AVX512
#else /* should not happen */
  LIBXSMM_UNUSED(handle); LIBXSMM_UNUSED(start_thread); LIBXSMM_UNUSED(tid); LIBXSMM_UNUSED(kind);
  status = LIBXSMM_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}

LIBXSMM_API_INTERN LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512_CORE)
libxsmm_dnn_err_t libxsmm_dnn_rnncell_st_bwdupd_nc_ck_bf16_bf16_emu(libxsmm_dnn_rnncell* handle, libxsmm_dnn_compute_kind kind, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
#if defined(LIBXSMM_INTRINSICS_AVX512_CORE) /*__AVX512F__,__AVX512BW__,__AVX512DQ__*/
#define LIBXSMM_RNN_CELL_AVX512
  typedef libxsmm_bfloat16 element_input_type;
  typedef libxsmm_bfloat16 element_output_type;
  typedef libxsmm_bfloat16 element_filter_type;

  /* some portable macrros fof BF16 <-> FP32 */
# include "template/libxsmm_dnn_bf16_macros_define.tpl.c"

  if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_RNN_RELU ) {
    status = LIBXSMM_DNN_ERR_NOT_IMPLEMENTED;
  } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_RNN_SIGMOID ) {
    status = LIBXSMM_DNN_ERR_NOT_IMPLEMENTED;
  } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_RNN_TANH ) {
    status = LIBXSMM_DNN_ERR_NOT_IMPLEMENTED;
  } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_LSTM ) {
# include "template/libxsmm_dnn_rnncell_st_lstm_bwdupd_nc_ck_generic_bf16.tpl.c"
  } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_GRU ) {
    status = LIBXSMM_DNN_ERR_NOT_IMPLEMENTED;
  } else {
    /* should not happen */
  }

# include "template/libxsmm_dnn_bf16_macros_undefine.tpl.c"
#undef LIBXSMM_RNN_CELL_AVX512
#else /* should not happen */
  LIBXSMM_UNUSED(handle); LIBXSMM_UNUSED(start_thread); LIBXSMM_UNUSED(tid); LIBXSMM_UNUSED(kind);
  status = LIBXSMM_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}

#if defined(LIBXSMM_INTRINSICS_AVX512_CPX)
LIBXSMM_API_INTERN LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512_CPX)
libxsmm_dnn_err_t libxsmm_dnn_rnncell_st_bwdupd_nc_ck_bf16_bf16(libxsmm_dnn_rnncell* handle, libxsmm_dnn_compute_kind kind, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
#if defined(LIBXSMM_INTRINSICS_AVX512_CPX) /*__AVX512F__,__AVX512BW__,__AVX512DQ__,__AVX512BF16__*/
#define LIBXSMM_RNN_CELL_AVX512
  typedef libxsmm_bfloat16 element_input_type;
  typedef libxsmm_bfloat16 element_output_type;
  typedef libxsmm_bfloat16 element_filter_type;

#define LIBXSMM_DNN_BF16_USE_CPX_AVX512_NI
  /* some portable macrros fof BF16 <-> FP32 */
# include "template/libxsmm_dnn_bf16_macros_define.tpl.c"

  if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_RNN_RELU ) {
    status = LIBXSMM_DNN_ERR_NOT_IMPLEMENTED;
  } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_RNN_SIGMOID ) {
    status = LIBXSMM_DNN_ERR_NOT_IMPLEMENTED;
  } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_RNN_TANH ) {
    status = LIBXSMM_DNN_ERR_NOT_IMPLEMENTED;
  } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_LSTM ) {
# include "template/libxsmm_dnn_rnncell_st_lstm_bwdupd_nc_ck_generic_bf16.tpl.c"
  } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_GRU ) {
    status = LIBXSMM_DNN_ERR_NOT_IMPLEMENTED;
  } else {
    /* should not happen */
  }

# include "template/libxsmm_dnn_bf16_macros_undefine.tpl.c"
#undef LIBXSMM_DNN_BF16_USE_CPX_AVX512_NI
#undef LIBXSMM_RNN_CELL_AVX512
#else /* should not happen */
  LIBXSMM_UNUSED(handle); LIBXSMM_UNUSED(start_thread); LIBXSMM_UNUSED(tid); LIBXSMM_UNUSED(kind);
  status = LIBXSMM_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}
#else
LIBXSMM_API_INTERN LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512_CORE)
libxsmm_dnn_err_t libxsmm_dnn_rnncell_st_bwdupd_nc_ck_bf16_bf16(libxsmm_dnn_rnncell* handle, libxsmm_dnn_compute_kind kind, int start_thread, int tid)
{
  return libxsmm_dnn_rnncell_st_bwdupd_nc_ck_bf16_bf16_emu(handle, kind, start_thread, tid);
}
#endif

#if defined(LIBXSMM_INTRINSICS_AVX512_CPX)
LIBXSMM_API_INTERN LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512_CPX)
libxsmm_dnn_err_t libxsmm_dnn_rnncell_st_bwdupd_nc_ck_bf16_bf16_amx(libxsmm_dnn_rnncell* handle, libxsmm_dnn_compute_kind kind, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
#if defined(LIBXSMM_INTRINSICS_AVX512_CPX) /*__AVX512F__,__AVX512BW__,__AVX512DQ__,__AVX512BF16__*/
#define LIBXSMM_RNN_CELL_AVX512
#define LIBXSMM_DNN_BF16_USE_CPX_AVX512_NI
  /* some portable macrros fof BF16 <-> FP32 */
# include "template/libxsmm_dnn_bf16_macros_define.tpl.c"
  typedef libxsmm_bfloat16 element_input_type;
  typedef libxsmm_bfloat16 element_output_type;
  typedef libxsmm_bfloat16 element_filter_type;
  if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_RNN_RELU ) {
    status = LIBXSMM_DNN_ERR_NOT_IMPLEMENTED;
  } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_RNN_SIGMOID ) {
    status = LIBXSMM_DNN_ERR_NOT_IMPLEMENTED;
  } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_RNN_TANH ) {
    status = LIBXSMM_DNN_ERR_NOT_IMPLEMENTED;
  } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_LSTM ) {
# include "template/libxsmm_dnn_rnncell_st_lstm_bwdupd_nc_ck_generic_bf16_amx.tpl.c"
  } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_GRU ) {
    status = LIBXSMM_DNN_ERR_NOT_IMPLEMENTED;
  } else {
    /* should not happen */
  }
# include "template/libxsmm_dnn_bf16_macros_undefine.tpl.c"
#undef LIBXSMM_DNN_BF16_USE_CPX_AVX512_NI
#undef LIBXSMM_RNN_CELL_AVX512
#else /* should not happen */
  LIBXSMM_UNUSED(handle); LIBXSMM_UNUSED(start_thread); LIBXSMM_UNUSED(tid); LIBXSMM_UNUSED(kind);
  status = LIBXSMM_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}
#else
LIBXSMM_API_INTERN LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512_CORE)
libxsmm_dnn_err_t libxsmm_dnn_rnncell_st_bwdupd_nc_ck_bf16_bf16_amx(libxsmm_dnn_rnncell* handle, libxsmm_dnn_compute_kind kind, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
#if defined(LIBXSMM_INTRINSICS_AVX512_CORE) /*__AVX512F__,__AVX512BW__,__AVX512DQ__*/
#define LIBXSMM_RNN_CELL_AVX512
  /* some portable macrros fof BF16 <-> FP32 */
# include "template/libxsmm_dnn_bf16_macros_define.tpl.c"
  typedef libxsmm_bfloat16 element_input_type;
  typedef libxsmm_bfloat16 element_output_type;
  typedef libxsmm_bfloat16 element_filter_type;
  if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_RNN_RELU ) {
    status = LIBXSMM_DNN_ERR_NOT_IMPLEMENTED;
  } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_RNN_SIGMOID ) {
    status = LIBXSMM_DNN_ERR_NOT_IMPLEMENTED;
  } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_RNN_TANH ) {
    status = LIBXSMM_DNN_ERR_NOT_IMPLEMENTED;
  } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_LSTM ) {
# include "template/libxsmm_dnn_rnncell_st_lstm_bwdupd_nc_ck_generic_bf16_amx.tpl.c"
  } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_GRU ) {
    status = LIBXSMM_DNN_ERR_NOT_IMPLEMENTED;
  } else {
    /* should not happen */
  }
# include "template/libxsmm_dnn_bf16_macros_undefine.tpl.c"
#undef LIBXSMM_RNN_CELL_AVX512
#else /* should not happen */
  LIBXSMM_UNUSED(handle); LIBXSMM_UNUSED(start_thread); LIBXSMM_UNUSED(tid); LIBXSMM_UNUSED(kind);
  status = LIBXSMM_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}
#endif

LIBXSMM_API_INTERN LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512_CORE)
libxsmm_dnn_err_t libxsmm_dnn_rnncell_st_bwdupd_nc_kcck_bf16_bf16_emu(libxsmm_dnn_rnncell* handle, libxsmm_dnn_compute_kind kind, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
#if defined(LIBXSMM_INTRINSICS_AVX512_CORE) /*__AVX512F__,__AVX512BW__,__AVX512DQ__*/
#define LIBXSMM_RNN_CELL_AVX512
  typedef libxsmm_bfloat16 element_input_type;
  typedef libxsmm_bfloat16 element_output_type;
  typedef libxsmm_bfloat16 element_filter_type;

  /* some portable macrros fof BF16 <-> FP32 */
# include "template/libxsmm_dnn_bf16_macros_define.tpl.c"

  if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_RNN_RELU ) {
    status = LIBXSMM_DNN_ERR_NOT_IMPLEMENTED;
  } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_RNN_SIGMOID ) {
    status = LIBXSMM_DNN_ERR_NOT_IMPLEMENTED;
  } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_RNN_TANH ) {
    status = LIBXSMM_DNN_ERR_NOT_IMPLEMENTED;
  } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_LSTM ) {
# include "template/libxsmm_dnn_rnncell_st_lstm_bwdupd_nc_kcck_bf16.tpl.c"
  } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_GRU ) {
    status = LIBXSMM_DNN_ERR_NOT_IMPLEMENTED;
  } else {
    /* should not happen */
  }

# include "template/libxsmm_dnn_bf16_macros_undefine.tpl.c"

#undef LIBXSMM_RNN_CELL_AVX512
#else /* should not happen */
  LIBXSMM_UNUSED(handle); LIBXSMM_UNUSED(start_thread); LIBXSMM_UNUSED(tid); LIBXSMM_UNUSED(kind);
  status = LIBXSMM_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}

#if defined(LIBXSMM_INTRINSICS_AVX512_CPX)
LIBXSMM_API_INTERN LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512_CPX)
libxsmm_dnn_err_t libxsmm_dnn_rnncell_st_bwdupd_ncnc_kcck_bf16_bf16_amx(libxsmm_dnn_rnncell* handle, libxsmm_dnn_compute_kind kind, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
#if defined(LIBXSMM_INTRINSICS_AVX512_CPX) /*__AVX512F__,__AVX512BW__,__AVX512DQ__,__AVX512BF16__*/
#define LIBXSMM_RNN_CELL_AVX512
  typedef libxsmm_bfloat16 element_input_type;
  typedef libxsmm_bfloat16 element_output_type;
  typedef libxsmm_bfloat16 element_filter_type;

#define LIBXSMM_DNN_BF16_USE_CPX_AVX512_NI
  /* some portable macrros fof BF16 <-> FP32 */
# include "template/libxsmm_dnn_bf16_macros_define.tpl.c"

  if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_RNN_RELU ) {
    status = LIBXSMM_DNN_ERR_NOT_IMPLEMENTED;
  } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_RNN_SIGMOID ) {
    status = LIBXSMM_DNN_ERR_NOT_IMPLEMENTED;
  } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_RNN_TANH ) {
    status = LIBXSMM_DNN_ERR_NOT_IMPLEMENTED;
  } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_LSTM ) {
# include "template/libxsmm_dnn_rnncell_st_lstm_bwdupd_ncnc_kcck_bf16_amx.tpl.c"
  } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_GRU ) {
    status = LIBXSMM_DNN_ERR_NOT_IMPLEMENTED;
  } else {
    /* should not happen */
  }

# include "template/libxsmm_dnn_bf16_macros_undefine.tpl.c"

#undef LIBXSMM_DNN_BF16_USE_CPX_AVX512_NI
#undef LIBXSMM_RNN_CELL_AVX512
#else /* should not happen */
  LIBXSMM_UNUSED(handle); LIBXSMM_UNUSED(start_thread); LIBXSMM_UNUSED(tid); LIBXSMM_UNUSED(kind);
  status = LIBXSMM_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}
#else
LIBXSMM_API_INTERN LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512_CORE)
libxsmm_dnn_err_t libxsmm_dnn_rnncell_st_bwdupd_ncnc_kcck_bf16_bf16_amx(libxsmm_dnn_rnncell* handle, libxsmm_dnn_compute_kind kind, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
#if defined(LIBXSMM_INTRINSICS_AVX512_CORE) /*__AVX512F__,__AVX512BW__,__AVX512DQ__ */
#define LIBXSMM_RNN_CELL_AVX512
  typedef libxsmm_bfloat16 element_input_type;
  typedef libxsmm_bfloat16 element_output_type;
  typedef libxsmm_bfloat16 element_filter_type;

  /* some portable macrros fof BF16 <-> FP32 */
# include "template/libxsmm_dnn_bf16_macros_define.tpl.c"

  if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_RNN_RELU ) {
    status = LIBXSMM_DNN_ERR_NOT_IMPLEMENTED;
  } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_RNN_SIGMOID ) {
    status = LIBXSMM_DNN_ERR_NOT_IMPLEMENTED;
  } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_RNN_TANH ) {
    status = LIBXSMM_DNN_ERR_NOT_IMPLEMENTED;
  } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_LSTM ) {
# include "template/libxsmm_dnn_rnncell_st_lstm_bwdupd_ncnc_kcck_bf16_amx.tpl.c"
  } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_GRU ) {
    status = LIBXSMM_DNN_ERR_NOT_IMPLEMENTED;
  } else {
    /* should not happen */
  }

# include "template/libxsmm_dnn_bf16_macros_undefine.tpl.c"
#undef LIBXSMM_RNN_CELL_AVX512
#else /* should not happen */
  LIBXSMM_UNUSED(handle); LIBXSMM_UNUSED(start_thread); LIBXSMM_UNUSED(tid); LIBXSMM_UNUSED(kind);
  status = LIBXSMM_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}
#endif

#if defined(LIBXSMM_INTRINSICS_AVX512_CPX)
LIBXSMM_API_INTERN LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512_CPX)
libxsmm_dnn_err_t libxsmm_dnn_rnncell_st_bwdupd_nc_kcck_bf16_bf16(libxsmm_dnn_rnncell* handle, libxsmm_dnn_compute_kind kind, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
#if defined(LIBXSMM_INTRINSICS_AVX512_CPX) /*__AVX512F__,__AVX512BW__,__AVX512DQ__,__AVX512BF16__*/
#define LIBXSMM_RNN_CELL_AVX512
  typedef libxsmm_bfloat16 element_input_type;
  typedef libxsmm_bfloat16 element_output_type;
  typedef libxsmm_bfloat16 element_filter_type;

#define LIBXSMM_DNN_BF16_USE_CPX_AVX512_NI
  /* some portable macrros fof BF16 <-> FP32 */
# include "template/libxsmm_dnn_bf16_macros_define.tpl.c"

  if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_RNN_RELU ) {
    status = LIBXSMM_DNN_ERR_NOT_IMPLEMENTED;
  } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_RNN_SIGMOID ) {
    status = LIBXSMM_DNN_ERR_NOT_IMPLEMENTED;
  } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_RNN_TANH ) {
    status = LIBXSMM_DNN_ERR_NOT_IMPLEMENTED;
  } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_LSTM ) {
# include "template/libxsmm_dnn_rnncell_st_lstm_bwdupd_nc_kcck_bf16.tpl.c"
  } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_GRU ) {
    status = LIBXSMM_DNN_ERR_NOT_IMPLEMENTED;
  } else {
    /* should not happen */
  }

# include "template/libxsmm_dnn_bf16_macros_undefine.tpl.c"
#undef LIBXSMM_DNN_BF16_USE_CPX_AVX512_NI

#undef LIBXSMM_RNN_CELL_AVX512
#else /* should not happen */
  LIBXSMM_UNUSED(handle); LIBXSMM_UNUSED(start_thread); LIBXSMM_UNUSED(tid); LIBXSMM_UNUSED(kind);
  status = LIBXSMM_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}
#else
LIBXSMM_API_INTERN LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512_CORE)
libxsmm_dnn_err_t libxsmm_dnn_rnncell_st_bwdupd_nc_kcck_bf16_bf16(libxsmm_dnn_rnncell* handle, libxsmm_dnn_compute_kind kind, int start_thread, int tid)
{
  return libxsmm_dnn_rnncell_st_bwdupd_nc_kcck_bf16_bf16_emu(handle, kind, start_thread, tid);
}
#endif

#if defined(LIBXSMM_INTRINSICS_AVX512_CPX)
LIBXSMM_API_INTERN LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512_CPX)
libxsmm_dnn_err_t libxsmm_dnn_rnncell_st_bwdupd_nc_kcck_bf16_bf16_amx(libxsmm_dnn_rnncell* handle, libxsmm_dnn_compute_kind kind, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
#if defined(LIBXSMM_INTRINSICS_AVX512_CPX) /*__AVX512F__,__AVX512BW__,__AVX512DQ__,__AVX512BF16__*/
#define LIBXSMM_RNN_CELL_AVX512
#define LIBXSMM_DNN_BF16_USE_CPX_AVX512_NI
  /* some portable macrros fof BF16 <-> FP32 */
# include "template/libxsmm_dnn_bf16_macros_define.tpl.c"
  typedef libxsmm_bfloat16 element_input_type;
  typedef libxsmm_bfloat16 element_output_type;
  typedef libxsmm_bfloat16 element_filter_type;
  if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_RNN_RELU ) {
    status = LIBXSMM_DNN_ERR_NOT_IMPLEMENTED;
  } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_RNN_SIGMOID ) {
    status = LIBXSMM_DNN_ERR_NOT_IMPLEMENTED;
  } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_RNN_TANH ) {
    status = LIBXSMM_DNN_ERR_NOT_IMPLEMENTED;
  } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_LSTM ) {
# include "template/libxsmm_dnn_rnncell_st_lstm_bwdupd_nc_kcck_bf16_amx.tpl.c"
  } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_GRU ) {
    status = LIBXSMM_DNN_ERR_NOT_IMPLEMENTED;
  } else {
    /* should not happen */
  }
# include "template/libxsmm_dnn_bf16_macros_undefine.tpl.c"
#undef LIBXSMM_DNN_BF16_USE_CPX_AVX512_NI
#undef LIBXSMM_RNN_CELL_AVX512
#else /* should not happen */
  LIBXSMM_UNUSED(handle); LIBXSMM_UNUSED(start_thread); LIBXSMM_UNUSED(tid); LIBXSMM_UNUSED(kind);
  status = LIBXSMM_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}
#else
LIBXSMM_API_INTERN LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512_CORE)
libxsmm_dnn_err_t libxsmm_dnn_rnncell_st_bwdupd_nc_kcck_bf16_bf16_amx(libxsmm_dnn_rnncell* handle, libxsmm_dnn_compute_kind kind, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
#if defined(LIBXSMM_INTRINSICS_AVX512_CORE) /*__AVX512F__,__AVX512BW__,__AVX512DQ__*/
#define LIBXSMM_RNN_CELL_AVX512
# include "template/libxsmm_dnn_bf16_macros_define.tpl.c"
  typedef libxsmm_bfloat16 element_input_type;
  typedef libxsmm_bfloat16 element_output_type;
  typedef libxsmm_bfloat16 element_filter_type;
  if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_RNN_RELU ) {
    status = LIBXSMM_DNN_ERR_NOT_IMPLEMENTED;
  } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_RNN_SIGMOID ) {
    status = LIBXSMM_DNN_ERR_NOT_IMPLEMENTED;
  } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_RNN_TANH ) {
    status = LIBXSMM_DNN_ERR_NOT_IMPLEMENTED;
  } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_LSTM ) {
# include "template/libxsmm_dnn_rnncell_st_lstm_bwdupd_nc_kcck_bf16_amx.tpl.c"
  } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_GRU ) {
    status = LIBXSMM_DNN_ERR_NOT_IMPLEMENTED;
  } else {
    /* should not happen */
  }
# include "template/libxsmm_dnn_bf16_macros_undefine.tpl.c"
#undef LIBXSMM_RNN_CELL_AVX512
#else /* should not happen */
  LIBXSMM_UNUSED(handle); LIBXSMM_UNUSED(start_thread); LIBXSMM_UNUSED(tid); LIBXSMM_UNUSED(kind);
  status = LIBXSMM_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}
#endif

LIBXSMM_API_INTERN LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512)
libxsmm_dnn_err_t libxsmm_dnn_rnncell_st_bwdupd_nc_kcck_f32_f32(libxsmm_dnn_rnncell* handle, libxsmm_dnn_compute_kind kind, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
#if defined(LIBXSMM_INTRINSICS_AVX512) /*__AVX512F__*/
#define LIBXSMM_RNN_CELL_AVX512
  typedef float element_input_type;
  typedef float element_output_type;
  typedef float element_filter_type;
  if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_RNN_RELU ) {
# define LIBXSMM_DNN_RNN_RELU_BWDUPD
# include "template/libxsmm_dnn_rnncell_st_rnn_bwdupd_nc_kcck.tpl.c"
# undef LIBXSMM_DNN_RNN_RELU_BWDUPD
  } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_RNN_SIGMOID ) {
# define LIBXSMM_DNN_RNN_SIGMOID_BWDUPD
# include "template/libxsmm_dnn_rnncell_st_rnn_bwdupd_nc_kcck.tpl.c"
# undef LIBXSMM_DNN_RNN_SIGMOID_BWDUPD
  } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_RNN_TANH ) {
# define LIBXSMM_DNN_RNN_TANH_BWDUPD
# include "template/libxsmm_dnn_rnncell_st_rnn_bwdupd_nc_kcck.tpl.c"
# undef LIBXSMM_DNN_RNN_TANH_BWDUPD
  } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_LSTM ) {
# include "template/libxsmm_dnn_rnncell_st_lstm_bwdupd_nc_kcck.tpl.c"
  } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_GRU ) {
# include "template/libxsmm_dnn_rnncell_st_gru_bwdupd_nc_kcck.tpl.c"
  } else {
    /* should not happen */
  }
#undef LIBXSMM_RNN_CELL_AVX512
#else /* should not happen */
  LIBXSMM_UNUSED(handle); LIBXSMM_UNUSED(start_thread); LIBXSMM_UNUSED(tid); LIBXSMM_UNUSED(kind);
  status = LIBXSMM_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}


LIBXSMM_API_INTERN LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512)
libxsmm_dnn_err_t libxsmm_dnn_rnncell_st_bwdupd_ncnc_kcck_f32_f32(libxsmm_dnn_rnncell* handle, libxsmm_dnn_compute_kind kind, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
#if defined(LIBXSMM_INTRINSICS_AVX512) /*__AVX512F__*/
  LIBXSMM_UNUSED(handle); LIBXSMM_UNUSED(start_thread); LIBXSMM_UNUSED(tid);
  status = LIBXSMM_DNN_ERR_NOT_IMPLEMENTED;
#if 0
  typedef float element_input_type;
  typedef float element_output_type;
  typedef float element_filter_type;
# include "template/libxsmm_dnn_rnncell_st_rnn_bwdupd_ncnc_kcck_generic.tpl.c"
#endif
  LIBXSMM_UNUSED(handle); LIBXSMM_UNUSED(start_thread); LIBXSMM_UNUSED(tid); LIBXSMM_UNUSED(kind);
#else /* should not happen */
  LIBXSMM_UNUSED(handle); LIBXSMM_UNUSED(start_thread); LIBXSMM_UNUSED(tid); LIBXSMM_UNUSED(kind);
  status = LIBXSMM_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}


LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_rnncell_st_bwdupd_nc_ck(libxsmm_dnn_rnncell* handle, libxsmm_dnn_compute_kind kind, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  /* check if we have input, output and filter */
#if 0
  if (handle->? == 0 ) {
    status = LIBXSMM_DNN_ERR_DATA_NOT_BOUND;
    return status;
  }
#endif

  /* check if we are on AVX512 */
#if defined(LIBXSMM_INTRINSICS_AVX512) /*__AVX512F__*/
  if ( (libxsmm_target_archid >= LIBXSMM_X86_AVX512) && (libxsmm_target_archid <= LIBXSMM_X86_ALLFEAT) ) {
    if (handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_F32 && handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_F32 ) {
      status = libxsmm_dnn_rnncell_st_bwdupd_nc_ck_f32_f32( handle, kind, start_thread, tid );
    }
#if defined(LIBXSMM_INTRINSICS_AVX512_CORE) /*__AVX512F__,__AVX512BW__,__AVX512DQ__*/
    else if (handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_BF16 && handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_BF16) {
      if ( handle->desc.N % 2 != 0 ) {
        status = LIBXSMM_DNN_ERR_NOT_IMPLEMENTED;
      } else {
#if defined(LIBXSMM_INTRINSICS_AVX512_CPX) /*__AVX512F__,__AVX512BW__,__AVX512DQ__,__AVX512BF16__*/
        if ( libxsmm_target_archid >= LIBXSMM_X86_AVX512_CORE && libxsmm_target_archid < LIBXSMM_X86_AVX512_CPX ) {
          status = libxsmm_dnn_rnncell_st_bwdupd_nc_ck_bf16_bf16_emu( handle, kind, start_thread, tid );
        } else if ( libxsmm_target_archid >= LIBXSMM_X86_AVX512_CPX && libxsmm_target_archid < LIBXSMM_X86_AVX512_SPR )  {
          status = libxsmm_dnn_rnncell_st_bwdupd_nc_ck_bf16_bf16( handle, kind, start_thread, tid );
        } else if ( libxsmm_target_archid >= LIBXSMM_X86_AVX512_SPR )  {
          status = libxsmm_dnn_rnncell_st_bwdupd_nc_ck_bf16_bf16_amx( handle, kind, start_thread, tid );
        }
#else
        if ( libxsmm_target_archid >= LIBXSMM_X86_AVX512_CORE ) {
          status = libxsmm_dnn_rnncell_st_bwdupd_nc_ck_bf16_bf16_emu( handle, kind, start_thread, tid );
        }
#endif
        else {
          status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
          return status;
        }
      }
    }
#endif
    else  {
      status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  } else
#endif
  {
    if (handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_F32 && handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_F32 ) {
      typedef float element_input_type;
      typedef float element_output_type;
      typedef float element_filter_type;
      if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_RNN_RELU ) {
#define LIBXSMM_DNN_RNN_RELU_BWDUPD
# include "template/libxsmm_dnn_rnncell_st_rnn_bwdupd_nc_ck_generic.tpl.c"
#undef LIBXSMM_DNN_RNN_RELU_BWDUPD
      } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_RNN_SIGMOID ) {
#define LIBXSMM_DNN_RNN_SIGMOID_BWDUPD
# include "template/libxsmm_dnn_rnncell_st_rnn_bwdupd_nc_ck_generic.tpl.c"
#undef LIBXSMM_DNN_RNN_SIGMOID_BWDUPD
      } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_RNN_TANH ) {
#define LIBXSMM_DNN_RNN_TANH_BWDUPD
# include "template/libxsmm_dnn_rnncell_st_rnn_bwdupd_nc_ck_generic.tpl.c"
#undef LIBXSMM_DNN_RNN_TANH_BWDUPD
      } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_LSTM ) {
# include "template/libxsmm_dnn_rnncell_st_lstm_bwdupd_nc_ck_generic.tpl.c"
      } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_GRU ) {
# include "template/libxsmm_dnn_rnncell_st_gru_bwdupd_nc_ck_generic.tpl.c"
      } else {
        /* should not happen */
      }
    } else {
      status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  }

  return status;
}

LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_rnncell_st_bwdupd_nc_kcck(libxsmm_dnn_rnncell* handle, libxsmm_dnn_compute_kind kind, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  /* check if we have input, output and filter */
#if 0
  if (handle->? == 0 ) {
    status = LIBXSMM_DNN_ERR_DATA_NOT_BOUND;
    return status;
  }
#endif

  /* check if we are on AVX512 */
#if defined(LIBXSMM_INTRINSICS_AVX512) /*__AVX512F__*/
  if ( (libxsmm_target_archid >= LIBXSMM_X86_AVX512) && (libxsmm_target_archid <= LIBXSMM_X86_ALLFEAT) ) {
    if ( handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_F32 && handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_F32 ) {
      status = libxsmm_dnn_rnncell_st_bwdupd_nc_kcck_f32_f32( handle, kind, start_thread, tid );
    }
#if defined(LIBXSMM_INTRINSICS_AVX512_CORE) /*__AVX512F__,__AVX512BW__,__AVX512DQ__*/
    else if ( handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_BF16 && handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_BF16 ) {
      if ( handle->desc.N % 2 != 0 ) {
        status = LIBXSMM_DNN_ERR_NOT_IMPLEMENTED;
      } else {
#if defined(LIBXSMM_INTRINSICS_AVX512_CPX) /*__AVX512F__,__AVX512BW__,__AVX512DQ__,__AVX512BF16__*/
        if ( libxsmm_target_archid >= LIBXSMM_X86_AVX512_CORE && libxsmm_target_archid < LIBXSMM_X86_AVX512_CPX ) {
          status = libxsmm_dnn_rnncell_st_bwdupd_nc_kcck_bf16_bf16_emu( handle, kind, start_thread, tid );
        } else if ( libxsmm_target_archid >= LIBXSMM_X86_AVX512_CPX && libxsmm_target_archid < LIBXSMM_X86_AVX512_SPR ) {
          status = libxsmm_dnn_rnncell_st_bwdupd_nc_kcck_bf16_bf16( handle, kind, start_thread, tid );
        } else if ( libxsmm_target_archid >= LIBXSMM_X86_AVX512_SPR ) {
          status = libxsmm_dnn_rnncell_st_bwdupd_nc_kcck_bf16_bf16_amx( handle, kind, start_thread, tid );
        }
#else
        if ( libxsmm_target_archid >= LIBXSMM_X86_AVX512_CORE && libxsmm_target_archid < LIBXSMM_X86_AVX512_SPR) {
          status = libxsmm_dnn_rnncell_st_bwdupd_nc_kcck_bf16_bf16_emu( handle, kind, start_thread, tid );
        } else if (libxsmm_target_archid >= LIBXSMM_X86_AVX512_SPR) {
          status = libxsmm_dnn_rnncell_st_bwdupd_nc_kcck_bf16_bf16_amx( handle, kind, start_thread, tid );
        }
#endif
        else {
          status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
          return status;
        }
      }
    }
#endif
    else {
      status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  } else
#endif
  {
    if (handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_F32 && handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_F32 ) {
      typedef float element_input_type;
      typedef float element_output_type;
      typedef float element_filter_type;
      if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_RNN_RELU ) {
#define LIBXSMM_DNN_RNN_RELU_BWDUPD
# include "template/libxsmm_dnn_rnncell_st_rnn_bwdupd_nc_kcck.tpl.c"
#undef LIBXSMM_DNN_RNN_RELU_BWDUPD
      } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_RNN_SIGMOID ) {
#define LIBXSMM_DNN_RNN_SIGMOID_BWDUPD
# include "template/libxsmm_dnn_rnncell_st_rnn_bwdupd_nc_kcck.tpl.c"
#undef LIBXSMM_DNN_RNN_SIGMOID_BWDUPD
      } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_RNN_TANH ) {
#define LIBXSMM_DNN_RNN_TANH_BWDUPD
# include "template/libxsmm_dnn_rnncell_st_rnn_bwdupd_nc_kcck.tpl.c"
#undef LIBXSMM_DNN_RNN_TANH_BWDUPD
      } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_LSTM ) {
# include "template/libxsmm_dnn_rnncell_st_lstm_bwdupd_nc_kcck.tpl.c"
      } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_GRU ) {
# include "template/libxsmm_dnn_rnncell_st_gru_bwdupd_nc_kcck.tpl.c"
      } else {
        /* should not happen */
      }
    } else {
      status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  }

  return status;
}


LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_rnncell_st_bwdupd_ncnc_kcck(libxsmm_dnn_rnncell* handle, libxsmm_dnn_compute_kind kind, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  /* check if we have input, output and filter */
#if 0
  if (handle->? == 0 ) {
    status = LIBXSMM_DNN_ERR_DATA_NOT_BOUND;
    return status;
  }
#endif

  /* check if we are on AVX512 */
#if defined(LIBXSMM_INTRINSICS_AVX512_CPX) /*__AVX512F__,__AVX512BW__,__AVX512DQ__,__AVX512BF16__*/
  if ( (libxsmm_target_archid >= LIBXSMM_X86_AVX512) && (libxsmm_target_archid <= LIBXSMM_X86_ALLFEAT) ) {
    if (handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_F32 && handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_F32 ) {
      status = libxsmm_dnn_rnncell_st_bwdupd_ncnc_kcck_f32_f32( handle, kind, start_thread, tid );
    } else if ( handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_BF16 && handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_BF16 && libxsmm_target_archid >= LIBXSMM_X86_AVX512_SPR ) {
      status = libxsmm_dnn_rnncell_st_bwdupd_ncnc_kcck_bf16_bf16_amx( handle, kind, start_thread, tid);
    } else {
      status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  } else
#elif defined(LIBXSMM_INTRINSICS_AVX512) /*__AVX512F__*/
  if ( (libxsmm_target_archid >= LIBXSMM_X86_AVX512) && (libxsmm_target_archid <= LIBXSMM_X86_ALLFEAT) ) {
    if (handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_F32 && handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_F32 ) {
      status = libxsmm_dnn_rnncell_st_bwdupd_ncnc_kcck_f32_f32( handle, kind, start_thread, tid );
    } else if ( handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_BF16 && handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_BF16 && libxsmm_target_archid >= LIBXSMM_X86_AVX512_SPR ) {
      status = libxsmm_dnn_rnncell_st_bwdupd_ncnc_kcck_bf16_bf16_amx( handle, kind, start_thread, tid);
    } else {
      status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  } else
#endif
  {
    if (handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_F32 && handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_F32 ) {
      LIBXSMM_UNUSED(kind); LIBXSMM_UNUSED(start_thread); LIBXSMM_UNUSED(tid);
      status = LIBXSMM_DNN_ERR_NOT_IMPLEMENTED;
    } else {
      status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  }

  return status;
}

