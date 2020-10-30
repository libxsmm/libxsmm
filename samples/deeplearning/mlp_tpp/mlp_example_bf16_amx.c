/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Evangelos Georganas, Alexander Heinecke (Intel Corp.)
******************************************************************************/
#include <libxsmm.h>
#include <libxsmm_sync.h>

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#if defined(_OPENMP)
# include <omp.h>
#endif

/* include c-based dnn library */
#include "../common/dnn_common.h"

#define CHECK_L1

#define _mm512_loadcvt_bf16_fp32(A)   LIBXSMM_INTRINSICS_MM512_CVTPBH_PS(_mm256_loadu_si256((__m256i*)(A)))
#define LIBXSMM_INTRINSISCS_MM512_CVTNE2PS_PBH( A, B ) (__m512i)_mm512_cvtne2ps_pbh( A, B )
#define _mm512_load_fil(A)   _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepi16_epi32(_mm256_loadu_si256((__m256i*)(A))),16))
#define _mm512_store_fil(A,B)  _mm256_storeu_si256((__m256i*)(A), _mm512_cvtneps_pbh((B)))

#define LIBXSMM_DNN_CONVERT_BUFFER_BF16_F32(in, out, length) do { \
  unsigned int full_chunks = length / 16; \
  unsigned int remainder = length % 16; \
  int __i = 0; \
  if (remainder == 0) { \
    for ( __i = 0; __i < length; __i+= 16) { \
      _mm512_storeu_ps( (float*)out+__i, LIBXSMM_INTRINSICS_MM512_CVTPBH_PS( _mm256_loadu_si256((__m256i*)((const libxsmm_bfloat16*)in+__i)))); \
    } \
  } else { \
    unsigned int chunk; \
    for ( chunk = 0; chunk < full_chunks; chunk++) { \
      __i = chunk * 16; \
      _mm512_storeu_ps( (float*)out+__i, LIBXSMM_INTRINSICS_MM512_CVTPBH_PS( _mm256_loadu_si256((__m256i*)((const libxsmm_bfloat16*)in+__i)))); \
    } \
    libxsmm_convert_bf16_f32((const libxsmm_bfloat16*)in+16*full_chunks, (float*)out+16*full_chunks, remainder); \
  } \
} while(0)

#define LIBXSMM_DNN_CONVERT_BUFFER_F32_BF16(in, out, length) do { \
  unsigned int full_chunks = length / 32; \
  unsigned int remainder = length % 32; \
  int __i = 0; \
  if (remainder == 0) { \
    for ( __i = 0; __i < length; __i+= 32) { \
      _mm512_storeu_si512((libxsmm_bfloat16*)out+__i, LIBXSMM_INTRINSISCS_MM512_CVTNE2PS_PBH(LIBXSMM_INTRINSICS_MM512_LOAD_PS((const float*)in+__i+16), LIBXSMM_INTRINSICS_MM512_LOAD_PS((const float*)in+__i))); \
    } \
  } else { \
    unsigned int chunk; \
    for ( chunk = 0; chunk < full_chunks; chunk++) { \
      __i = chunk * 32; \
      _mm512_storeu_si512((libxsmm_bfloat16*)out+__i, LIBXSMM_INTRINSISCS_MM512_CVTNE2PS_PBH(LIBXSMM_INTRINSICS_MM512_LOAD_PS((const float*)in+__i+16), LIBXSMM_INTRINSICS_MM512_LOAD_PS((const float*)in+__i))); \
    } \
    libxsmm_rne_convert_fp32_bf16((const float*)in+32*full_chunks, (libxsmm_bfloat16*)out+32*full_chunks, remainder); \
  } \
} while(0)

LIBXSMM_INLINE void my_init_buf(float* buf, size_t size, int initPos, int initOne)
{
  int i;
  zero_buf(buf, size);
  for (i = 0; i < (int)size; ++i) {
    buf[i] = (float)((initOne != 0) ? 1.0 : ((initPos != 0) ? libxsmm_rng_f64() : (0.05 - libxsmm_rng_f64()/10.0)));
  }
}

LIBXSMM_INLINE void my_init_buf_bf16(libxsmm_bfloat16* buf, size_t size, int initPos, int initOne)
{
  int i;
  zero_buf_bf16(buf, size);
  for (i = 0; i < (int)size; ++i) {
    libxsmm_bfloat16_hp tmp;
    tmp.f = (float)((initOne != 0) ? 1.0 : ((initPos != 0) ? libxsmm_rng_f64() : (0.05 - libxsmm_rng_f64()/10.0)));
    buf[i] = tmp.i[1];
  }
}

void my_bf16_vnni_transpose_16x16(void* source_void, void* dest_void, int source_stride, int dest_stride)
{
  libxsmm_bfloat16 *source = (libxsmm_bfloat16*)source_void;
  libxsmm_bfloat16 *dest = (libxsmm_bfloat16*)dest_void;
  __m512i zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6, zmm7;
  __m512i tmp0, tmp1, tmp2, tmp3;
  const __m512i abcdefgh_to_abefcdgh = _mm512_set4_epi32(0x0f0e0b0a, 0x0d0c0908, 0x07060302, 0x05040100);

  zmm0 = _mm512_loadu_si512(source);
  zmm1 = _mm512_loadu_si512(source + source_stride);
  zmm2 = _mm512_loadu_si512(source + source_stride*2);
  zmm3 = _mm512_loadu_si512(source + source_stride*3);
  zmm4 = _mm512_loadu_si512(source + source_stride*4);
  zmm5 = _mm512_loadu_si512(source + source_stride*5);
  zmm6 = _mm512_loadu_si512(source + source_stride*6);
  zmm7 = _mm512_loadu_si512(source + source_stride*7);

  zmm0 = _mm512_shuffle_epi8(zmm0, abcdefgh_to_abefcdgh);
  zmm1 = _mm512_shuffle_epi8(zmm1, abcdefgh_to_abefcdgh);
  zmm2 = _mm512_shuffle_epi8(zmm2, abcdefgh_to_abefcdgh);
  zmm3 = _mm512_shuffle_epi8(zmm3, abcdefgh_to_abefcdgh);
  zmm4 = _mm512_shuffle_epi8(zmm4, abcdefgh_to_abefcdgh);
  zmm5 = _mm512_shuffle_epi8(zmm5, abcdefgh_to_abefcdgh);
  zmm6 = _mm512_shuffle_epi8(zmm6, abcdefgh_to_abefcdgh);
  zmm7 = _mm512_shuffle_epi8(zmm7, abcdefgh_to_abefcdgh);

  tmp0 = _mm512_unpacklo_epi64(zmm0, zmm1);
  tmp1 = _mm512_unpackhi_epi64(zmm0, zmm1);
  tmp2 = _mm512_unpacklo_epi64(zmm2, zmm3);
  tmp3 = _mm512_unpackhi_epi64(zmm2, zmm3);
  zmm0 = _mm512_unpacklo_epi64(zmm4, zmm5);
  zmm1 = _mm512_unpackhi_epi64(zmm4, zmm5);
  zmm2 = _mm512_unpacklo_epi64(zmm6, zmm7);
  zmm3 = _mm512_unpackhi_epi64(zmm6, zmm7);

  zmm4 = _mm512_shuffle_i32x4(tmp0, tmp2, 0x88);
  zmm6 = _mm512_shuffle_i32x4(tmp0, tmp2, 0xdd);
  zmm5 = _mm512_shuffle_i32x4(tmp1, tmp3, 0x88);
  zmm7 = _mm512_shuffle_i32x4(tmp1, tmp3, 0xdd);
  tmp0 = _mm512_shuffle_i32x4(zmm0, zmm2, 0x88);
  tmp1 = _mm512_shuffle_i32x4(zmm0, zmm2, 0xdd);
  tmp2 = _mm512_shuffle_i32x4(zmm1, zmm3, 0x88);
  tmp3 = _mm512_shuffle_i32x4(zmm1, zmm3, 0xdd);

  zmm0 = _mm512_shuffle_i32x4(zmm4, tmp0, 0x88);
  zmm1 = _mm512_shuffle_i32x4(zmm5, tmp2, 0x88);
  zmm2 = _mm512_shuffle_i32x4(zmm6, tmp1, 0x88);
  zmm3 = _mm512_shuffle_i32x4(zmm7, tmp3, 0x88);
  zmm4 = _mm512_shuffle_i32x4(zmm4, tmp0, 0xdd);
  zmm5 = _mm512_shuffle_i32x4(zmm5, tmp2, 0xdd);
  zmm6 = _mm512_shuffle_i32x4(zmm6, tmp1, 0xdd);
  zmm7 = _mm512_shuffle_i32x4(zmm7, tmp3, 0xdd);

  _mm512_storeu_si512(dest, zmm0);
  _mm512_storeu_si512(dest + dest_stride, zmm1);
  _mm512_storeu_si512(dest + dest_stride * 2, zmm2);
  _mm512_storeu_si512(dest + dest_stride * 3, zmm3);
  _mm512_storeu_si512(dest + dest_stride * 4, zmm4);
  _mm512_storeu_si512(dest + dest_stride * 5, zmm5);
  _mm512_storeu_si512(dest + dest_stride * 6, zmm6);
  _mm512_storeu_si512(dest + dest_stride * 7, zmm7);
}

void my_bf16_vnni_transpose(libxsmm_bfloat16* src, libxsmm_bfloat16* dst, int M, int N, int ld_in, int ld_out)
{
  const int _M = M/16, _N = N/16;
  int i = 0, j = 0;
  for (i = 0; i < _N; i++) {
    for (j = 0; j < _M; j++) {
      my_bf16_vnni_transpose_16x16((libxsmm_bfloat16*) src+i*16*ld_in+j*32, (libxsmm_bfloat16*) dst+j*16*ld_out+i*32, ld_in*2, ld_out*2);
    }
  }

}

void my_bf16_transpose_32x16(libxsmm_bfloat16 *in, libxsmm_bfloat16 *out, int ld_in, int ld_out)
{
  __m512i r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, ra, rb, rc, rd, re, rf;
  __m512i t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, ta, tb, tc, td, te, tf;
  const int in_width=ld_in, out_width=ld_out;
  const __m512i idx_lo         = _mm512_set_epi64(13, 12, 5, 4, 9, 8, 1, 0);
  const __m512i idx_hi         = _mm512_set_epi64(7, 6, 15, 14, 3, 2, 11, 10);

  r0 = _mm512_loadu_si512(in + 0*in_width);
  r1 = _mm512_loadu_si512(in + 1*in_width);
  r2 = _mm512_loadu_si512(in + 2*in_width);
  r3 = _mm512_loadu_si512(in + 3*in_width);
  r4 = _mm512_loadu_si512(in + 4*in_width);
  r5 = _mm512_loadu_si512(in + 5*in_width);
  r6 = _mm512_loadu_si512(in + 6*in_width);
  r7 = _mm512_loadu_si512(in + 7*in_width);
  r8 = _mm512_loadu_si512(in + 8*in_width);
  r9 = _mm512_loadu_si512(in + 9*in_width);
  ra = _mm512_loadu_si512(in + 10*in_width);
  rb = _mm512_loadu_si512(in + 11*in_width);
  rc = _mm512_loadu_si512(in + 12*in_width);
  rd = _mm512_loadu_si512(in + 13*in_width);
  re = _mm512_loadu_si512(in + 14*in_width);
  rf = _mm512_loadu_si512(in + 15*in_width);

  t0 = _mm512_unpacklo_epi16(r0,r1);
  t1 = _mm512_unpackhi_epi16(r0,r1);
  t2 = _mm512_unpacklo_epi16(r2,r3);
  t3 = _mm512_unpackhi_epi16(r2,r3);
  t4 = _mm512_unpacklo_epi16(r4,r5);
  t5 = _mm512_unpackhi_epi16(r4,r5);
  t6 = _mm512_unpacklo_epi16(r6,r7);
  t7 = _mm512_unpackhi_epi16(r6,r7);
  t8 = _mm512_unpacklo_epi16(r8,r9);
  t9 = _mm512_unpackhi_epi16(r8,r9);
  ta = _mm512_unpacklo_epi16(ra,rb);
  tb = _mm512_unpackhi_epi16(ra,rb);
  tc = _mm512_unpacklo_epi16(rc,rd);
  td = _mm512_unpackhi_epi16(rc,rd);
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

  t0 = _mm512_permutex2var_epi64(r0, idx_lo, r8);
  t1 = _mm512_permutex2var_epi64(r1, idx_lo, r9);
  t2 = _mm512_permutex2var_epi64(r2, idx_lo, ra);
  t3 = _mm512_permutex2var_epi64(r3, idx_lo, rb);
  t4 = _mm512_permutex2var_epi64(r4, idx_lo, rc);
  t5 = _mm512_permutex2var_epi64(r5, idx_lo, rd);
  t6 = _mm512_permutex2var_epi64(r6, idx_lo, re);
  t7 = _mm512_permutex2var_epi64(r7, idx_lo, rf);
  t8 = _mm512_permutex2var_epi64(r8, idx_hi, r0);
  t9 = _mm512_permutex2var_epi64(r9, idx_hi, r1);
  ta = _mm512_permutex2var_epi64(ra, idx_hi, r2);
  tb = _mm512_permutex2var_epi64(rb, idx_hi, r3);
  tc = _mm512_permutex2var_epi64(rc, idx_hi, r4);
  td = _mm512_permutex2var_epi64(rd, idx_hi, r5);
  te = _mm512_permutex2var_epi64(re, idx_hi, r6);
  tf = _mm512_permutex2var_epi64(rf, idx_hi, r7);

  LIBXSMM_INTRINSICS_MM256_STORE_EPI32(out + 0*out_width, LIBXSMM_INTRINSICS_MM512_EXTRACTI64X4_EPI64(t0, 0));
  LIBXSMM_INTRINSICS_MM256_STORE_EPI32(out + 1*out_width, LIBXSMM_INTRINSICS_MM512_EXTRACTI64X4_EPI64(t0, 1));
  LIBXSMM_INTRINSICS_MM256_STORE_EPI32(out + 2*out_width, LIBXSMM_INTRINSICS_MM512_EXTRACTI64X4_EPI64(t1, 0));
  LIBXSMM_INTRINSICS_MM256_STORE_EPI32(out + 3*out_width, LIBXSMM_INTRINSICS_MM512_EXTRACTI64X4_EPI64(t1, 1));
  LIBXSMM_INTRINSICS_MM256_STORE_EPI32(out + 4*out_width, LIBXSMM_INTRINSICS_MM512_EXTRACTI64X4_EPI64(t2, 0));
  LIBXSMM_INTRINSICS_MM256_STORE_EPI32(out + 5*out_width, LIBXSMM_INTRINSICS_MM512_EXTRACTI64X4_EPI64(t2, 1));
  LIBXSMM_INTRINSICS_MM256_STORE_EPI32(out + 6*out_width, LIBXSMM_INTRINSICS_MM512_EXTRACTI64X4_EPI64(t3, 0));
  LIBXSMM_INTRINSICS_MM256_STORE_EPI32(out + 7*out_width, LIBXSMM_INTRINSICS_MM512_EXTRACTI64X4_EPI64(t3, 1));
  LIBXSMM_INTRINSICS_MM256_STORE_EPI32(out + 8*out_width, LIBXSMM_INTRINSICS_MM512_EXTRACTI64X4_EPI64(t4, 0));
  LIBXSMM_INTRINSICS_MM256_STORE_EPI32(out + 9*out_width, LIBXSMM_INTRINSICS_MM512_EXTRACTI64X4_EPI64(t4, 1));
  LIBXSMM_INTRINSICS_MM256_STORE_EPI32(out + 10*out_width, LIBXSMM_INTRINSICS_MM512_EXTRACTI64X4_EPI64(t5, 0));
  LIBXSMM_INTRINSICS_MM256_STORE_EPI32(out + 11*out_width, LIBXSMM_INTRINSICS_MM512_EXTRACTI64X4_EPI64(t5, 1));
  LIBXSMM_INTRINSICS_MM256_STORE_EPI32(out + 12*out_width, LIBXSMM_INTRINSICS_MM512_EXTRACTI64X4_EPI64(t6, 0));
  LIBXSMM_INTRINSICS_MM256_STORE_EPI32(out + 13*out_width, LIBXSMM_INTRINSICS_MM512_EXTRACTI64X4_EPI64(t6, 1));
  LIBXSMM_INTRINSICS_MM256_STORE_EPI32(out + 14*out_width, LIBXSMM_INTRINSICS_MM512_EXTRACTI64X4_EPI64(t7, 0));
  LIBXSMM_INTRINSICS_MM256_STORE_EPI32(out + 15*out_width, LIBXSMM_INTRINSICS_MM512_EXTRACTI64X4_EPI64(t7, 1));
  LIBXSMM_INTRINSICS_MM256_STORE_EPI32(out + 16*out_width, LIBXSMM_INTRINSICS_MM512_EXTRACTI64X4_EPI64(t8, 0));
  LIBXSMM_INTRINSICS_MM256_STORE_EPI32(out + 17*out_width, LIBXSMM_INTRINSICS_MM512_EXTRACTI64X4_EPI64(t8, 1));
  LIBXSMM_INTRINSICS_MM256_STORE_EPI32(out + 18*out_width, LIBXSMM_INTRINSICS_MM512_EXTRACTI64X4_EPI64(t9, 0));
  LIBXSMM_INTRINSICS_MM256_STORE_EPI32(out + 19*out_width, LIBXSMM_INTRINSICS_MM512_EXTRACTI64X4_EPI64(t9, 1));
  LIBXSMM_INTRINSICS_MM256_STORE_EPI32(out + 20*out_width, LIBXSMM_INTRINSICS_MM512_EXTRACTI64X4_EPI64(ta, 0));
  LIBXSMM_INTRINSICS_MM256_STORE_EPI32(out + 21*out_width, LIBXSMM_INTRINSICS_MM512_EXTRACTI64X4_EPI64(ta, 1));
  LIBXSMM_INTRINSICS_MM256_STORE_EPI32(out + 22*out_width, LIBXSMM_INTRINSICS_MM512_EXTRACTI64X4_EPI64(tb, 0));
  LIBXSMM_INTRINSICS_MM256_STORE_EPI32(out + 23*out_width, LIBXSMM_INTRINSICS_MM512_EXTRACTI64X4_EPI64(tb, 1));
  LIBXSMM_INTRINSICS_MM256_STORE_EPI32(out + 24*out_width, LIBXSMM_INTRINSICS_MM512_EXTRACTI64X4_EPI64(tc, 0));
  LIBXSMM_INTRINSICS_MM256_STORE_EPI32(out + 25*out_width, LIBXSMM_INTRINSICS_MM512_EXTRACTI64X4_EPI64(tc, 1));
  LIBXSMM_INTRINSICS_MM256_STORE_EPI32(out + 26*out_width, LIBXSMM_INTRINSICS_MM512_EXTRACTI64X4_EPI64(td, 0));
  LIBXSMM_INTRINSICS_MM256_STORE_EPI32(out + 27*out_width, LIBXSMM_INTRINSICS_MM512_EXTRACTI64X4_EPI64(td, 1));
  LIBXSMM_INTRINSICS_MM256_STORE_EPI32(out + 28*out_width, LIBXSMM_INTRINSICS_MM512_EXTRACTI64X4_EPI64(te, 0));
  LIBXSMM_INTRINSICS_MM256_STORE_EPI32(out + 29*out_width, LIBXSMM_INTRINSICS_MM512_EXTRACTI64X4_EPI64(te, 1));
  LIBXSMM_INTRINSICS_MM256_STORE_EPI32(out + 30*out_width, LIBXSMM_INTRINSICS_MM512_EXTRACTI64X4_EPI64(tf, 0));
  LIBXSMM_INTRINSICS_MM256_STORE_EPI32(out + 31*out_width, LIBXSMM_INTRINSICS_MM512_EXTRACTI64X4_EPI64(tf, 1));
}

void my_bf16_transpose_32xcols(libxsmm_bfloat16 *in, libxsmm_bfloat16 *out, int col, int ld_in, int ld_out)
{
  __m512i r0 = LIBXSMM_INTRINSICS_MM512_UNDEFINED_EPI32(), r1 = LIBXSMM_INTRINSICS_MM512_UNDEFINED_EPI32(), r2 = LIBXSMM_INTRINSICS_MM512_UNDEFINED_EPI32(), r3 = LIBXSMM_INTRINSICS_MM512_UNDEFINED_EPI32(), r4 = LIBXSMM_INTRINSICS_MM512_UNDEFINED_EPI32(), r5 = LIBXSMM_INTRINSICS_MM512_UNDEFINED_EPI32(), r6 = LIBXSMM_INTRINSICS_MM512_UNDEFINED_EPI32(), r7 = LIBXSMM_INTRINSICS_MM512_UNDEFINED_EPI32(), r8 = LIBXSMM_INTRINSICS_MM512_UNDEFINED_EPI32(), r9 = LIBXSMM_INTRINSICS_MM512_UNDEFINED_EPI32(), ra = LIBXSMM_INTRINSICS_MM512_UNDEFINED_EPI32(), rb = LIBXSMM_INTRINSICS_MM512_UNDEFINED_EPI32(), rc = LIBXSMM_INTRINSICS_MM512_UNDEFINED_EPI32(), rd = LIBXSMM_INTRINSICS_MM512_UNDEFINED_EPI32(), re = LIBXSMM_INTRINSICS_MM512_UNDEFINED_EPI32(), rf = LIBXSMM_INTRINSICS_MM512_UNDEFINED_EPI32();
  __m512i t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, ta, tb, tc, td, te, tf;
  const int in_width=ld_in, out_width=ld_out;
  const __m512i idx_lo         = _mm512_set_epi64(13, 12, 5, 4, 9, 8, 1, 0);
  const __m512i idx_hi         = _mm512_set_epi64(7, 6, 15, 14, 3, 2, 11, 10);
  __mmask16 store_mask         = LIBXSMM_INTRINSICS_MM512_CVTU32_MASK16(((unsigned int)1 << col) - 1);

  if (col == 15) {
    r0 = _mm512_loadu_si512(in + 0*in_width);
    r1 = _mm512_loadu_si512(in + 1*in_width);
    r2 = _mm512_loadu_si512(in + 2*in_width);
    r3 = _mm512_loadu_si512(in + 3*in_width);
    r4 = _mm512_loadu_si512(in + 4*in_width);
    r5 = _mm512_loadu_si512(in + 5*in_width);
    r6 = _mm512_loadu_si512(in + 6*in_width);
    r7 = _mm512_loadu_si512(in + 7*in_width);
    r8 = _mm512_loadu_si512(in + 8*in_width);
    r9 = _mm512_loadu_si512(in + 9*in_width);
    ra = _mm512_loadu_si512(in + 10*in_width);
    rb = _mm512_loadu_si512(in + 11*in_width);
    rc = _mm512_loadu_si512(in + 12*in_width);
    rd = _mm512_loadu_si512(in + 13*in_width);
    re = _mm512_loadu_si512(in + 14*in_width);
  } else if (col == 14) {
    r0 = _mm512_loadu_si512(in + 0*in_width);
    r1 = _mm512_loadu_si512(in + 1*in_width);
    r2 = _mm512_loadu_si512(in + 2*in_width);
    r3 = _mm512_loadu_si512(in + 3*in_width);
    r4 = _mm512_loadu_si512(in + 4*in_width);
    r5 = _mm512_loadu_si512(in + 5*in_width);
    r6 = _mm512_loadu_si512(in + 6*in_width);
    r7 = _mm512_loadu_si512(in + 7*in_width);
    r8 = _mm512_loadu_si512(in + 8*in_width);
    r9 = _mm512_loadu_si512(in + 9*in_width);
    ra = _mm512_loadu_si512(in + 10*in_width);
    rb = _mm512_loadu_si512(in + 11*in_width);
    rc = _mm512_loadu_si512(in + 12*in_width);
    rd = _mm512_loadu_si512(in + 13*in_width);
  } else if (col == 13) {
    r0 = _mm512_loadu_si512(in + 0*in_width);
    r1 = _mm512_loadu_si512(in + 1*in_width);
    r2 = _mm512_loadu_si512(in + 2*in_width);
    r3 = _mm512_loadu_si512(in + 3*in_width);
    r4 = _mm512_loadu_si512(in + 4*in_width);
    r5 = _mm512_loadu_si512(in + 5*in_width);
    r6 = _mm512_loadu_si512(in + 6*in_width);
    r7 = _mm512_loadu_si512(in + 7*in_width);
    r8 = _mm512_loadu_si512(in + 8*in_width);
    r9 = _mm512_loadu_si512(in + 9*in_width);
    ra = _mm512_loadu_si512(in + 10*in_width);
    rb = _mm512_loadu_si512(in + 11*in_width);
    rc = _mm512_loadu_si512(in + 12*in_width);
  } else if (col == 12) {
    r0 = _mm512_loadu_si512(in + 0*in_width);
    r1 = _mm512_loadu_si512(in + 1*in_width);
    r2 = _mm512_loadu_si512(in + 2*in_width);
    r3 = _mm512_loadu_si512(in + 3*in_width);
    r4 = _mm512_loadu_si512(in + 4*in_width);
    r5 = _mm512_loadu_si512(in + 5*in_width);
    r6 = _mm512_loadu_si512(in + 6*in_width);
    r7 = _mm512_loadu_si512(in + 7*in_width);
    r8 = _mm512_loadu_si512(in + 8*in_width);
    r9 = _mm512_loadu_si512(in + 9*in_width);
    ra = _mm512_loadu_si512(in + 10*in_width);
    rb = _mm512_loadu_si512(in + 11*in_width);
  } else if (col == 11) {
    r0 = _mm512_loadu_si512(in + 0*in_width);
    r1 = _mm512_loadu_si512(in + 1*in_width);
    r2 = _mm512_loadu_si512(in + 2*in_width);
    r3 = _mm512_loadu_si512(in + 3*in_width);
    r4 = _mm512_loadu_si512(in + 4*in_width);
    r5 = _mm512_loadu_si512(in + 5*in_width);
    r6 = _mm512_loadu_si512(in + 6*in_width);
    r7 = _mm512_loadu_si512(in + 7*in_width);
    r8 = _mm512_loadu_si512(in + 8*in_width);
    r9 = _mm512_loadu_si512(in + 9*in_width);
    ra = _mm512_loadu_si512(in + 10*in_width);
  } else if (col == 10) {
    r0 = _mm512_loadu_si512(in + 0*in_width);
    r1 = _mm512_loadu_si512(in + 1*in_width);
    r2 = _mm512_loadu_si512(in + 2*in_width);
    r3 = _mm512_loadu_si512(in + 3*in_width);
    r4 = _mm512_loadu_si512(in + 4*in_width);
    r5 = _mm512_loadu_si512(in + 5*in_width);
    r6 = _mm512_loadu_si512(in + 6*in_width);
    r7 = _mm512_loadu_si512(in + 7*in_width);
    r8 = _mm512_loadu_si512(in + 8*in_width);
    r9 = _mm512_loadu_si512(in + 9*in_width);
  } else if (col == 9) {
    r0 = _mm512_loadu_si512(in + 0*in_width);
    r1 = _mm512_loadu_si512(in + 1*in_width);
    r2 = _mm512_loadu_si512(in + 2*in_width);
    r3 = _mm512_loadu_si512(in + 3*in_width);
    r4 = _mm512_loadu_si512(in + 4*in_width);
    r5 = _mm512_loadu_si512(in + 5*in_width);
    r6 = _mm512_loadu_si512(in + 6*in_width);
    r7 = _mm512_loadu_si512(in + 7*in_width);
    r8 = _mm512_loadu_si512(in + 8*in_width);
  } else if (col == 8) {
    r0 = _mm512_loadu_si512(in + 0*in_width);
    r1 = _mm512_loadu_si512(in + 1*in_width);
    r2 = _mm512_loadu_si512(in + 2*in_width);
    r3 = _mm512_loadu_si512(in + 3*in_width);
    r4 = _mm512_loadu_si512(in + 4*in_width);
    r5 = _mm512_loadu_si512(in + 5*in_width);
    r6 = _mm512_loadu_si512(in + 6*in_width);
    r7 = _mm512_loadu_si512(in + 7*in_width);
  } else if (col == 7) {
    r0 = _mm512_loadu_si512(in + 0*in_width);
    r1 = _mm512_loadu_si512(in + 1*in_width);
    r2 = _mm512_loadu_si512(in + 2*in_width);
    r3 = _mm512_loadu_si512(in + 3*in_width);
    r4 = _mm512_loadu_si512(in + 4*in_width);
    r5 = _mm512_loadu_si512(in + 5*in_width);
    r6 = _mm512_loadu_si512(in + 6*in_width);
  } else if (col == 6) {
    r0 = _mm512_loadu_si512(in + 0*in_width);
    r1 = _mm512_loadu_si512(in + 1*in_width);
    r2 = _mm512_loadu_si512(in + 2*in_width);
    r3 = _mm512_loadu_si512(in + 3*in_width);
    r4 = _mm512_loadu_si512(in + 4*in_width);
    r5 = _mm512_loadu_si512(in + 5*in_width);
  } else if (col == 5) {
    r0 = _mm512_loadu_si512(in + 0*in_width);
    r1 = _mm512_loadu_si512(in + 1*in_width);
    r2 = _mm512_loadu_si512(in + 2*in_width);
    r3 = _mm512_loadu_si512(in + 3*in_width);
    r4 = _mm512_loadu_si512(in + 4*in_width);
  } else if (col == 4) {
    r0 = _mm512_loadu_si512(in + 0*in_width);
    r1 = _mm512_loadu_si512(in + 1*in_width);
    r2 = _mm512_loadu_si512(in + 2*in_width);
    r3 = _mm512_loadu_si512(in + 3*in_width);
  } else if (col == 3) {
    r0 = _mm512_loadu_si512(in + 0*in_width);
    r1 = _mm512_loadu_si512(in + 1*in_width);
    r2 = _mm512_loadu_si512(in + 2*in_width);
  } else if (col == 2) {
    r0 = _mm512_loadu_si512(in + 0*in_width);
    r1 = _mm512_loadu_si512(in + 1*in_width);
  } else if (col == 1) {
    r0 = _mm512_loadu_si512(in + 0*in_width);
  }

  t0 = _mm512_unpacklo_epi16(r0,r1);
  t1 = _mm512_unpackhi_epi16(r0,r1);
  t2 = _mm512_unpacklo_epi16(r2,r3);
  t3 = _mm512_unpackhi_epi16(r2,r3);
  t4 = _mm512_unpacklo_epi16(r4,r5);
  t5 = _mm512_unpackhi_epi16(r4,r5);
  t6 = _mm512_unpacklo_epi16(r6,r7);
  t7 = _mm512_unpackhi_epi16(r6,r7);
  t8 = _mm512_unpacklo_epi16(r8,r9);
  t9 = _mm512_unpackhi_epi16(r8,r9);
  ta = _mm512_unpacklo_epi16(ra,rb);
  tb = _mm512_unpackhi_epi16(ra,rb);
  tc = _mm512_unpacklo_epi16(rc,rd);
  td = _mm512_unpackhi_epi16(rc,rd);
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

  t0 = _mm512_permutex2var_epi64(r0, idx_lo, r8);
  t1 = _mm512_permutex2var_epi64(r1, idx_lo, r9);
  t2 = _mm512_permutex2var_epi64(r2, idx_lo, ra);
  t3 = _mm512_permutex2var_epi64(r3, idx_lo, rb);
  t4 = _mm512_permutex2var_epi64(r4, idx_lo, rc);
  t5 = _mm512_permutex2var_epi64(r5, idx_lo, rd);
  t6 = _mm512_permutex2var_epi64(r6, idx_lo, re);
  t7 = _mm512_permutex2var_epi64(r7, idx_lo, rf);
  t8 = _mm512_permutex2var_epi64(r8, idx_hi, r0);
  t9 = _mm512_permutex2var_epi64(r9, idx_hi, r1);
  ta = _mm512_permutex2var_epi64(ra, idx_hi, r2);
  tb = _mm512_permutex2var_epi64(rb, idx_hi, r3);
  tc = _mm512_permutex2var_epi64(rc, idx_hi, r4);
  td = _mm512_permutex2var_epi64(rd, idx_hi, r5);
  te = _mm512_permutex2var_epi64(re, idx_hi, r6);
  tf = _mm512_permutex2var_epi64(rf, idx_hi, r7);

  _mm256_mask_storeu_epi16(out + 0*out_width, store_mask, LIBXSMM_INTRINSICS_MM512_EXTRACTI64X4_EPI64(t0, 0));
  _mm256_mask_storeu_epi16(out + 1*out_width, store_mask, LIBXSMM_INTRINSICS_MM512_EXTRACTI64X4_EPI64(t0, 1));
  _mm256_mask_storeu_epi16(out + 2*out_width, store_mask, LIBXSMM_INTRINSICS_MM512_EXTRACTI64X4_EPI64(t1, 0));
  _mm256_mask_storeu_epi16(out + 3*out_width, store_mask, LIBXSMM_INTRINSICS_MM512_EXTRACTI64X4_EPI64(t1, 1));
  _mm256_mask_storeu_epi16(out + 4*out_width, store_mask, LIBXSMM_INTRINSICS_MM512_EXTRACTI64X4_EPI64(t2, 0));
  _mm256_mask_storeu_epi16(out + 5*out_width, store_mask, LIBXSMM_INTRINSICS_MM512_EXTRACTI64X4_EPI64(t2, 1));
  _mm256_mask_storeu_epi16(out + 6*out_width, store_mask, LIBXSMM_INTRINSICS_MM512_EXTRACTI64X4_EPI64(t3, 0));
  _mm256_mask_storeu_epi16(out + 7*out_width, store_mask, LIBXSMM_INTRINSICS_MM512_EXTRACTI64X4_EPI64(t3, 1));
  _mm256_mask_storeu_epi16(out + 8*out_width, store_mask, LIBXSMM_INTRINSICS_MM512_EXTRACTI64X4_EPI64(t4, 0));
  _mm256_mask_storeu_epi16(out + 9*out_width, store_mask, LIBXSMM_INTRINSICS_MM512_EXTRACTI64X4_EPI64(t4, 1));
  _mm256_mask_storeu_epi16(out + 10*out_width, store_mask, LIBXSMM_INTRINSICS_MM512_EXTRACTI64X4_EPI64(t5, 0));
  _mm256_mask_storeu_epi16(out + 11*out_width, store_mask, LIBXSMM_INTRINSICS_MM512_EXTRACTI64X4_EPI64(t5, 1));
  _mm256_mask_storeu_epi16(out + 12*out_width, store_mask, LIBXSMM_INTRINSICS_MM512_EXTRACTI64X4_EPI64(t6, 0));
  _mm256_mask_storeu_epi16(out + 13*out_width, store_mask, LIBXSMM_INTRINSICS_MM512_EXTRACTI64X4_EPI64(t6, 1));
  _mm256_mask_storeu_epi16(out + 14*out_width, store_mask, LIBXSMM_INTRINSICS_MM512_EXTRACTI64X4_EPI64(t7, 0));
  _mm256_mask_storeu_epi16(out + 15*out_width, store_mask, LIBXSMM_INTRINSICS_MM512_EXTRACTI64X4_EPI64(t7, 1));
  _mm256_mask_storeu_epi16(out + 16*out_width, store_mask, LIBXSMM_INTRINSICS_MM512_EXTRACTI64X4_EPI64(t8, 0));
  _mm256_mask_storeu_epi16(out + 17*out_width, store_mask, LIBXSMM_INTRINSICS_MM512_EXTRACTI64X4_EPI64(t8, 1));
  _mm256_mask_storeu_epi16(out + 18*out_width, store_mask, LIBXSMM_INTRINSICS_MM512_EXTRACTI64X4_EPI64(t9, 0));
  _mm256_mask_storeu_epi16(out + 19*out_width, store_mask, LIBXSMM_INTRINSICS_MM512_EXTRACTI64X4_EPI64(t9, 1));
  _mm256_mask_storeu_epi16(out + 20*out_width, store_mask, LIBXSMM_INTRINSICS_MM512_EXTRACTI64X4_EPI64(ta, 0));
  _mm256_mask_storeu_epi16(out + 21*out_width, store_mask, LIBXSMM_INTRINSICS_MM512_EXTRACTI64X4_EPI64(ta, 1));
  _mm256_mask_storeu_epi16(out + 22*out_width, store_mask, LIBXSMM_INTRINSICS_MM512_EXTRACTI64X4_EPI64(tb, 0));
  _mm256_mask_storeu_epi16(out + 23*out_width, store_mask, LIBXSMM_INTRINSICS_MM512_EXTRACTI64X4_EPI64(tb, 1));
  _mm256_mask_storeu_epi16(out + 24*out_width, store_mask, LIBXSMM_INTRINSICS_MM512_EXTRACTI64X4_EPI64(tc, 0));
  _mm256_mask_storeu_epi16(out + 25*out_width, store_mask, LIBXSMM_INTRINSICS_MM512_EXTRACTI64X4_EPI64(tc, 1));
  _mm256_mask_storeu_epi16(out + 26*out_width, store_mask, LIBXSMM_INTRINSICS_MM512_EXTRACTI64X4_EPI64(td, 0));
  _mm256_mask_storeu_epi16(out + 27*out_width, store_mask, LIBXSMM_INTRINSICS_MM512_EXTRACTI64X4_EPI64(td, 1));
  _mm256_mask_storeu_epi16(out + 28*out_width, store_mask, LIBXSMM_INTRINSICS_MM512_EXTRACTI64X4_EPI64(te, 0));
  _mm256_mask_storeu_epi16(out + 29*out_width, store_mask, LIBXSMM_INTRINSICS_MM512_EXTRACTI64X4_EPI64(te, 1));
  _mm256_mask_storeu_epi16(out + 30*out_width, store_mask, LIBXSMM_INTRINSICS_MM512_EXTRACTI64X4_EPI64(tf, 0));
  _mm256_mask_storeu_epi16(out + 31*out_width, store_mask, LIBXSMM_INTRINSICS_MM512_EXTRACTI64X4_EPI64(tf, 1));
}

void my_bf16_transpose(libxsmm_bfloat16 *in, libxsmm_bfloat16 *out, int M, int N, int ld_in, int ld_out){
  int i, j;
  int full16_chunks = N/16;
  int remainder_cols = N%16;
  int _N = N - remainder_cols;

  if (full16_chunks) {
    for (i=0; i<M; i+=32) {
      for (j=0; j<_N; j+=16) {
        my_bf16_transpose_32x16((libxsmm_bfloat16*)in + i + ld_in*j, (libxsmm_bfloat16*)out + j + i*ld_out, ld_in, ld_out);
      }
    }
  }

  if (remainder_cols) {
    for (i=0; i<M; i+=32) {
      my_bf16_transpose_32xcols((libxsmm_bfloat16*)in + i + ld_in*full16_chunks*16, (libxsmm_bfloat16*)out + full16_chunks*16 + i*ld_out, remainder_cols, ld_in, ld_out);
    }
  }
}

void my_bf16_vnni_reformat(libxsmm_bfloat16 *_in, libxsmm_bfloat16 *_out, int M, int N, int ld_in, int ld_out) {
  int n_full_pairs = N/2, n_pair, m;
  int half_n_pair = N%2;
  libxsmm_bfloat16 *in = _in, *out = _out;
  const __m512i selector = LIBXSMM_INTRINSICS_MM512_SET_EPI16(32, 0, 32, 0, 32, 0, 32, 0, 32, 0, 32, 0, 32, 0, 32, 0, 32, 0, 32, 0, 32, 0, 32, 0, 32, 0, 32, 0, 32, 0, 32, 0);
  const __m512i offsets_lo = LIBXSMM_INTRINSICS_MM512_SET_EPI16(15, 15, 14, 14, 13, 13, 12, 12, 11, 11, 10, 10, 9, 9, 8, 8, 7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0);
  const __m512i offsets_hi = LIBXSMM_INTRINSICS_MM512_SET_EPI16(31, 31, 30, 30, 29, 29, 28, 28, 27, 27, 26, 26, 25, 25, 24, 24, 23, 23, 22, 22, 21, 21, 20, 20, 19, 19, 18, 18, 17, 17, 16, 16);
  const __m512i idx_lo =  _mm512_or_epi32(selector, offsets_lo);
  const __m512i idx_hi =  _mm512_or_epi32(selector, offsets_hi);
  const __m512i zero_reg = _mm512_setzero_si512();
  __m512i n0, n1, out_lo, out_hi;
  LIBXSMM_UNUSED(ld_out);
  for (n_pair = 0; n_pair < n_full_pairs; n_pair++) {
    for (m = 0; m < M; m+=32) {
      n0 = _mm512_loadu_si512((const libxsmm_bfloat16*)in+m);
      n1 = _mm512_loadu_si512((const libxsmm_bfloat16*)in+m+ld_in);
      out_lo = _mm512_permutex2var_epi16(n0, idx_lo, n1);
      out_hi = _mm512_permutex2var_epi16(n0, idx_hi, n1);
      _mm512_storeu_si512((libxsmm_bfloat16*)out+m*2, out_lo);
      _mm512_storeu_si512((libxsmm_bfloat16*)out+m*2+32, out_hi);
    }
    in += 2*ld_in;
    out += 2*ld_in;
  }
  if (half_n_pair == 1) {
    for (m = 0; m < M; m+=32) {
      n0 = _mm512_loadu_si512((const libxsmm_bfloat16*)in+m);
      n1 = zero_reg;
      out_lo = _mm512_permutex2var_epi16(n0, idx_lo, n1);
      out_hi = _mm512_permutex2var_epi16(n0, idx_lo, n1);
      _mm512_storeu_si512((libxsmm_bfloat16*)out+m*2, out_lo);
      _mm512_storeu_si512((libxsmm_bfloat16*)out+m*2+32, out_hi);
    }
  }
}

typedef enum my_eltwise_fuse {
  MY_ELTWISE_FUSE_NONE = 0,
  MY_ELTWISE_FUSE_BIAS = 1,
  MY_ELTWISE_FUSE_RELU = 2,
  MY_ELTWISE_FUSE_BIAS_RELU = MY_ELTWISE_FUSE_BIAS | MY_ELTWISE_FUSE_RELU
} my_eltwise_fuse;

typedef enum my_pass {
  MY_PASS_FWD   = 1,
  MY_PASS_BWD_D = 2,
  MY_PASS_BWD_W = 4,
  MY_PASS_BWD   = 6
} my_pass;

typedef struct my_opt_config {
  libxsmm_blasint C;
  libxsmm_blasint K;
  libxsmm_blasint bc;
  libxsmm_blasint bk;
  libxsmm_blasint threads;
  float           lr;
  size_t          scratch_size;
  libxsmm_barrier* barrier;
} my_opt_config;

typedef struct my_smax_fwd_config {
  libxsmm_blasint N;
  libxsmm_blasint C;
  libxsmm_blasint bn;
  libxsmm_blasint bc;
  libxsmm_blasint threads;
  size_t          scratch_size;
  libxsmm_barrier* barrier;
} my_smax_fwd_config;

typedef struct my_smax_bwd_config {
  libxsmm_blasint N;
  libxsmm_blasint C;
  libxsmm_blasint bn;
  libxsmm_blasint bc;
  libxsmm_blasint threads;
  size_t          scratch_size;
  float           loss_weight;
  libxsmm_barrier* barrier;
} my_smax_bwd_config;


typedef struct my_fc_fwd_config {
  libxsmm_blasint N;
  libxsmm_blasint C;
  libxsmm_blasint K;
  libxsmm_blasint bn;
  libxsmm_blasint bc;
  libxsmm_blasint bk;
  libxsmm_blasint threads;
  my_eltwise_fuse fuse_type;
  libxsmm_blasint fwd_bf;
  libxsmm_blasint fwd_2d_blocking;
  libxsmm_blasint fwd_row_teams;
  libxsmm_blasint fwd_column_teams;
  size_t          scratch_size;
  libxsmm_barrier* barrier;
  libxsmm_bsmmfunction fwd_config_kernel;
  libxsmm_bsmmfunction tilerelease_kernel;
  libxsmm_bsmmfunction_reducebatch_strd gemm_fwd;
  libxsmm_bsmmfunction_reducebatch_strd gemm_fwd2;
  libxsmm_bmmfunction_reducebatch_strd gemm_fwd3;
  libxsmm_bmmfunction_reducebatch_strd_meltwfused gemm_fwd4;
  libxsmm_bmmfunction_reducebatch_strd_meltwfused gemm_fwd5;
  libxsmm_bmmfunction_reducebatch_strd_meltwfused gemm_fwd6;
  libxsmm_bmmfunction_reducebatch_strd_meltwfused gemm_fwd7;
  libxsmm_bmmfunction_reducebatch_strd_meltwfused gemm_fwd8;
  libxsmm_meltwfunction_cvtfp32bf16     fwd_cvtfp32bf16_kernel;
  libxsmm_meltwfunction_cvtfp32bf16_act fwd_cvtfp32bf16_relu_kernel;
  libxsmm_meltwfunction_act_cvtfp32bf16 fwd_sigmoid_cvtfp32bf16_kernel;
} my_fc_fwd_config;

typedef struct my_fc_bwd_config {
  libxsmm_blasint N;
  libxsmm_blasint C;
  libxsmm_blasint K;
  libxsmm_blasint bn;
  libxsmm_blasint bc;
  libxsmm_blasint bk;
  libxsmm_blasint threads;
  my_eltwise_fuse fuse_type;
  libxsmm_blasint bwd_bf;
  libxsmm_blasint bwd_2d_blocking;
  libxsmm_blasint bwd_row_teams;
  libxsmm_blasint bwd_column_teams;
  libxsmm_blasint upd_bf;
  libxsmm_blasint upd_2d_blocking;
  libxsmm_blasint upd_row_teams;
  libxsmm_blasint upd_column_teams;
  libxsmm_blasint ifm_subtasks;
  libxsmm_blasint ofm_subtasks;
  size_t          scratch_size;
  size_t  doutput_scratch_mark;
  libxsmm_barrier* barrier;
  libxsmm_bsmmfunction bwd_config_kernel;
  libxsmm_bsmmfunction upd_config_kernel;
  libxsmm_bsmmfunction tilerelease_kernel;
  libxsmm_bsmmfunction_reducebatch_strd gemm_bwd;
  libxsmm_bsmmfunction_reducebatch_strd gemm_bwd2;
  libxsmm_bmmfunction_reducebatch_strd gemm_bwd3;
  libxsmm_bsmmfunction_reducebatch_strd gemm_upd;
  libxsmm_bsmmfunction_reducebatch_strd gemm_upd2;
  libxsmm_bmmfunction_reducebatch_strd gemm_upd3;
  libxsmm_meltwfunction_cvtfp32bf16     bwd_cvtfp32bf16_kernel;
  libxsmm_meltwfunction_relu            bwd_relu_kernel;
} my_fc_bwd_config;

my_fc_fwd_config setup_my_fc_fwd(libxsmm_blasint N, libxsmm_blasint C, libxsmm_blasint K, libxsmm_blasint bn,
                                 libxsmm_blasint bc, libxsmm_blasint bk, libxsmm_blasint threads, my_eltwise_fuse fuse_type) {
  my_fc_fwd_config res;
  libxsmm_blasint lda = bk;
  libxsmm_blasint ldb = bc;
  libxsmm_blasint ldc = bk;
  float alpha = 1.0f;
  float beta = 1.0f;
  float zerobeta = 0.0f;
  libxsmm_meltw_flags fusion_flags;
  int l_flags, l_tc_flags;
  int l_tr_flags = LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG | ( LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N') );
  libxsmm_blasint unroll_hint;

  /* setting up some handle values */
  res.N = N;
  res.C = C;
  res.K = K;
  res.bn = bn;
  res.bc = bc;
  res.bk = bk;
  res.threads = threads;
  res.fuse_type = fuse_type;

  /* setup parallelization strategy */
  if (threads == 16) {
    res.fwd_bf = 1;
    res.fwd_2d_blocking = 1;
    res.fwd_row_teams = 2;
    res.fwd_column_teams = 8;
  } else {
    res.fwd_bf = 1;
    res.fwd_2d_blocking = 0;
    res.fwd_row_teams = 1;
    res.fwd_column_teams = 1;
  }

#if 0
  res.fwd_bf = atoi(getenv("FWD_BF"));
  res.fwd_2d_blocking = atoi(getenv("FWD_2D_BLOCKING"));
  res.fwd_row_teams = atoi(getenv("FWD_ROW_TEAMS"));
  res.fwd_column_teams = atoi(getenv("FWD_COLUMN_TEAMS"));
#endif

  /* setting up the barrier */
  res.barrier = libxsmm_barrier_create(threads, 1);

  /* TPP creation */
  l_flags = ( LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N') ) | LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG | LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG;
  l_tc_flags = LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG | ( LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N') );
  unroll_hint = (res.C/res.bc)/res.fwd_bf;

  res.fwd_config_kernel = libxsmm_bsmmdispatch(res.bk, res.bn, res.bc, &lda, &ldb, &ldc, NULL, &beta, &l_tc_flags, NULL);
  if ( res.fwd_config_kernel == NULL ) {
    fprintf( stderr, "JIT for BRGEMM TPP fwd_config_kernel failed. Bailing...!\n");
    exit(-1);
  }
  res.gemm_fwd = libxsmm_bsmmdispatch_reducebatch_strd_unroll(res.bk, res.bn, res.bc, res.bk*res.bc*sizeof(libxsmm_bfloat16), res.bc*res.bn*sizeof(libxsmm_bfloat16), unroll_hint, &lda, &ldb, &ldc, &alpha, &beta, &l_flags, NULL);
  if ( res.gemm_fwd == NULL ) {
    fprintf( stderr, "JIT for BRGEMM TPP gemm_fwd failed. Bailing...!\n");
    exit(-1);
  }
  res.gemm_fwd2 = libxsmm_bsmmdispatch_reducebatch_strd_unroll(res.bk, res.bn, res.bc, res.bk*res.bc*sizeof(libxsmm_bfloat16), res.bc*res.bn*sizeof(libxsmm_bfloat16), unroll_hint, &lda, &ldb, &ldc, &alpha, &zerobeta, &l_flags, NULL);
  if ( res.gemm_fwd2 == NULL ) {
    fprintf( stderr, "JIT for BRGEMM TPP gemm_fwd2 failed. Bailing...!\n");
    exit(-1);
  }
  res.gemm_fwd3 = libxsmm_bmmdispatch_reducebatch_strd_unroll(res.bk, res.bn, res.bc, res.bk*res.bc*sizeof(libxsmm_bfloat16), res.bc*res.bn*sizeof(libxsmm_bfloat16), unroll_hint, &lda, &ldb, &ldc, &alpha, &zerobeta, &l_flags, NULL);
  if ( res.gemm_fwd3 == NULL ) {
    fprintf( stderr, "JIT for BRGEMM TPP gemm_fwd3 failed. Bailing...!\n");
    exit(-1);
  }
  fusion_flags = LIBXSMM_MELTW_FLAG_COLBIAS_OVERWRITE_C;
  res.gemm_fwd4 = libxsmm_bmmdispatch_reducebatch_strd_meltwfused_unroll(res.bk, res.bn, res.bc, res.bk*res.bc*sizeof(libxsmm_bfloat16), res.bc*res.bn*sizeof(libxsmm_bfloat16), unroll_hint, &lda, &ldb, &ldc, &alpha, &zerobeta, &l_flags, NULL, LIBXSMM_MELTW_OPERATION_COLBIAS_ACT, LIBXSMM_DATATYPE_F32, fusion_flags, 0);
  if ( res.gemm_fwd4 == NULL ) {
    fprintf( stderr, "JIT for BRGEMM TPP gemm_fwd4 failed. Bailing...!\n");
    exit(-1);
  }
  fusion_flags = LIBXSMM_MELTW_FLAG_ACT_RELU_OVERWRITE_C;
  res.gemm_fwd5 = libxsmm_bmmdispatch_reducebatch_strd_meltwfused_unroll(res.bk, res.bn, res.bc, res.bk*res.bc*sizeof(libxsmm_bfloat16), res.bc*res.bn*sizeof(libxsmm_bfloat16), unroll_hint, &lda, &ldb, &ldc, &alpha, &zerobeta, &l_flags, NULL, LIBXSMM_MELTW_OPERATION_COLBIAS_ACT, LIBXSMM_DATATYPE_F32, fusion_flags, 0);
  if ( res.gemm_fwd5 == NULL ) {
    fprintf( stderr, "JIT for BRGEMM TPP gemm_fwd5 failed. Bailing...!\n");
    exit(-1);
  }
  fusion_flags = LIBXSMM_MELTW_FLAG_ACT_SIGM_OVERWRITE_C;
  res.gemm_fwd6 = libxsmm_bmmdispatch_reducebatch_strd_meltwfused_unroll(res.bk, res.bn, res.bc, res.bk*res.bc*sizeof(libxsmm_bfloat16), res.bc*res.bn*sizeof(libxsmm_bfloat16), unroll_hint, &lda, &ldb, &ldc, &alpha, &zerobeta, &l_flags, NULL, LIBXSMM_MELTW_OPERATION_COLBIAS_ACT, LIBXSMM_DATATYPE_F32, fusion_flags, 0);
  if ( res.gemm_fwd6 == NULL ) {
    fprintf( stderr, "JIT for BRGEMM TPP gemm_fwd6 failed. Bailing...!\n");
    exit(-1);
  }
  fusion_flags = LIBXSMM_MELTW_FLAG_COLBIAS_ACT_RELU_OVERWRITE_C;
  res.gemm_fwd7 = libxsmm_bmmdispatch_reducebatch_strd_meltwfused_unroll(res.bk, res.bn, res.bc, res.bk*res.bc*sizeof(libxsmm_bfloat16), res.bc*res.bn*sizeof(libxsmm_bfloat16), unroll_hint, &lda, &ldb, &ldc, &alpha, &zerobeta, &l_flags, NULL, LIBXSMM_MELTW_OPERATION_COLBIAS_ACT, LIBXSMM_DATATYPE_F32, fusion_flags, 0);
  if ( res.gemm_fwd7 == NULL ) {
    fprintf( stderr, "JIT for BRGEMM TPP gemm_fwd7 failed. Bailing...!\n");
    exit(-1);
  }
  fusion_flags = LIBXSMM_MELTW_FLAG_COLBIAS_ACT_SIGM_OVERWRITE_C;
  res.gemm_fwd8 = libxsmm_bmmdispatch_reducebatch_strd_meltwfused_unroll(res.bk, res.bn, res.bc, res.bk*res.bc*sizeof(libxsmm_bfloat16), res.bc*res.bn*sizeof(libxsmm_bfloat16), unroll_hint, &lda, &ldb, &ldc, &alpha, &zerobeta, &l_flags, NULL, LIBXSMM_MELTW_OPERATION_COLBIAS_ACT, LIBXSMM_DATATYPE_F32, fusion_flags, 0);
  if ( res.gemm_fwd8 == NULL ) {
    fprintf( stderr, "JIT for BRGEMM TPP gemm_fwd8 failed. Bailing...!\n");
    exit(-1);
  }

  /* Also JIT eltwise TPPs... */
  res.fwd_cvtfp32bf16_kernel = libxsmm_dispatch_meltw_cvtfp32bf16(res.bk, res.bn, &ldc, &ldc, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_BF16);
  if ( res.fwd_cvtfp32bf16_kernel == NULL ) {
    fprintf( stderr, "JIT for TPP fwd_cvtfp32bf16_kernel failed. Bailing...!\n");
    exit(-1);
  }
  res.fwd_cvtfp32bf16_relu_kernel = libxsmm_dispatch_meltw_cvtfp32bf16_act(res.bk, res.bn, &ldc, &ldc, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_BF16, LIBXSMM_MELTW_FLAG_CVTA_FUSE_RELU, 0);
  if ( res.fwd_cvtfp32bf16_relu_kernel == NULL ) {
    fprintf( stderr, "JIT for TPP fwd_cvtfp32bf16_relu_kernel failed. Bailing...!\n");
    exit(-1);
  }
  res.fwd_sigmoid_cvtfp32bf16_kernel = libxsmm_dispatch_meltw_act_cvtfp32bf16(res.bk, res.bn, &ldc, &ldc, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_BF16, LIBXSMM_MELTW_FLAG_ACVT_FUSE_SIGM, 0);
  if ( res.fwd_sigmoid_cvtfp32bf16_kernel == NULL ) {
    fprintf( stderr, "JIT for TPP fwd_sigmoid_cvtfp32bf16_kernel failed. Bailing...!\n");
    exit(-1);
  }
  res.tilerelease_kernel = libxsmm_bsmmdispatch(res.bk, res.bk, res.bk, NULL, NULL, NULL, NULL, NULL, &l_tr_flags, NULL);
  if ( res.tilerelease_kernel == NULL ) {
    fprintf( stderr, "JIT for TPP tilerelease_kernel failed. Bailing...!\n");
    exit(-1);
  }
  /* init scratch */
  res.scratch_size = sizeof(float) *  LIBXSMM_MAX(res.K * res.N, res.threads * LIBXSMM_MAX(res.bk * res.bn, res.K));

  return res;
}

my_fc_bwd_config setup_my_fc_bwd(libxsmm_blasint N, libxsmm_blasint C, libxsmm_blasint K, libxsmm_blasint bn,
                                 libxsmm_blasint bc, libxsmm_blasint bk, libxsmm_blasint threads, my_eltwise_fuse fuse_type) {
  my_fc_bwd_config res;
  const libxsmm_trans_descriptor* tr_desc = 0;
  libxsmm_descriptor_blob blob;
  libxsmm_blasint lda = bk;
  libxsmm_blasint ldb = bc;
  libxsmm_blasint ldc = bk;
  float alpha = 1.0f;
  float beta = 1.0f;
  float zerobeta = 0.0f;
  libxsmm_blasint updM;
  libxsmm_blasint updN;
  libxsmm_meltw_flags fusion_flags;
  int l_flags, l_tc_flags;
  int l_tr_flags = LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG | ( LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N') );
  libxsmm_blasint unroll_hint;
  size_t size_bwd_scratch;
  size_t size_upd_scratch;

  /* setting up some handle values */
  res.N = N;
  res.C = C;
  res.K = K;
  res.bn = bn;
  res.bc = bc;
  res.bk = bk;
  res.threads = threads;
  res.fuse_type = fuse_type;

  /* setup parallelization strategy */
  if (threads == 16) {
    res.bwd_bf = 1;
    res.bwd_2d_blocking = 1;
    res.bwd_row_teams = 2;
    res.bwd_column_teams = 8;
    res.upd_bf = 1;
    res.upd_2d_blocking = 0;
    res.upd_row_teams = 1;
    res.upd_column_teams = 1;
    res.ifm_subtasks = 1;
    res.ofm_subtasks = 1;
  } else {
    res.bwd_bf = 1;
    res.bwd_2d_blocking = 0;
    res.bwd_row_teams = 1;
    res.bwd_column_teams = 1;
    res.upd_bf = 1;
    res.upd_2d_blocking = 0;
    res.upd_row_teams = 1;
    res.upd_column_teams = 1;
    res.ifm_subtasks = 1;
    res.ofm_subtasks = 1;
  }

#if 0
  res.bwd_bf = atoi(getenv("BWD_BF"));
  res.bwd_2d_blocking = atoi(getenv("BWD_2D_BLOCKING"));
  res.bwd_row_teams = atoi(getenv("BWD_ROW_TEAMS"));
  res.bwd_column_teams = atoi(getenv("BWD_COLUMN_TEAMS"));
  res.upd_bf = atoi(getenv("UPD_BF"));
  res.upd_2d_blocking = atoi(getenv("UPD_2D_BLOCKING"));
  res.upd_row_teams = atoi(getenv("UPD_ROW_TEAMS"));
  res.upd_column_teams = atoi(getenv("UPD_COLUMN_TEAMS"));
  res.ifm_subtasks = atoi(getenv("IFM_SUBTASKS"));
  res.ofm_subtasks = atoi(getenv("OFM_SUBTASKS"));
#endif

  /* setting up the barrier */
  res.barrier = libxsmm_barrier_create(threads, 1);

  /* TPP creation */
  /* BWD GEMM */
  l_flags = ( LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N') ) | LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG | LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG;
  l_tc_flags = LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG | ( LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N') );
  unroll_hint = (res.K/res.bk)/res.bwd_bf;

  res.gemm_bwd = libxsmm_bsmmdispatch_reducebatch_strd_unroll(res.bc, res.bn, res.bk, res.bk*res.bc*sizeof(libxsmm_bfloat16), res.bk*res.bn*sizeof(libxsmm_bfloat16), unroll_hint, &ldb, &lda, &ldb, &alpha, &beta, &l_flags, NULL);
  if ( res.gemm_bwd == NULL ) {
    fprintf( stderr, "JIT for BRGEMM TPP gemm_bwd failed. Bailing...!\n");
    exit(-1);
  }
  res.gemm_bwd2 = libxsmm_bsmmdispatch_reducebatch_strd_unroll(res.bc, res.bn, res.bk, res.bk*res.bc*sizeof(libxsmm_bfloat16), res.bk*res.bn*sizeof(libxsmm_bfloat16), unroll_hint, &ldb, &lda, &ldb, &alpha, &zerobeta, &l_flags, NULL);
  if ( res.gemm_bwd2 == NULL ) {
    fprintf( stderr, "JIT for BRGEMM TPP gemm_bwd2 failed. Bailing...!\n");
    exit(-1);
  }
  res.gemm_bwd3 = libxsmm_bmmdispatch_reducebatch_strd_unroll(res.bc, res.bn, res.bk, res.bk*res.bc*sizeof(libxsmm_bfloat16), res.bk*res.bn*sizeof(libxsmm_bfloat16), unroll_hint, &ldb, &lda, &ldb, &alpha, &zerobeta, &l_flags, NULL);
  if ( res.gemm_bwd3 == NULL ) {
    fprintf( stderr, "JIT for BRGEMM TPP gemm_bwd3 failed. Bailing...!\n");
    exit(-1);
  }
  res.bwd_config_kernel = libxsmm_bsmmdispatch(res.bc, res.bn, res.bk, &ldb, &lda, &ldb, NULL, &beta, &l_tc_flags, NULL);
  if ( res.bwd_config_kernel == NULL ) {
    fprintf( stderr, "JIT for BRGEMM TPP bwd_config_kernel failed. Bailing...!\n");
    exit(-1);
  }

  /* Also JIT eltwise TPPs... */
  res.bwd_cvtfp32bf16_kernel  = libxsmm_dispatch_meltw_cvtfp32bf16(res.bc, res.bn, &ldb, &ldb, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_BF16);
  if ( res.bwd_cvtfp32bf16_kernel == NULL ) {
    fprintf( stderr, "JIT for TPP bwd_cvtfp32bf16_kernel failed. Bailing...!\n");
    exit(-1);
  }

  res.bwd_relu_kernel  = libxsmm_dispatch_meltw_relu(res.bc, res.bn, &ldb, &ldb, LIBXSMM_DATATYPE_BF16, LIBXSMM_DATATYPE_BF16, LIBXSMM_MELTW_FLAG_RELU_BWD, 0);
  if ( res.bwd_relu_kernel == NULL ) {
    fprintf( stderr, "JIT for TPP bwd_relu_kernel failed. Bailing...!\n");
    exit(-1);

  }

  /* UPD GEMM */
  lda = res.bk;
  ldb = res.bn;
  ldc = res.bk;
  updM = res.bk/res.ofm_subtasks;
  updN = res.bc/res.ifm_subtasks;

  l_flags = ( LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N') ) | LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG | LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG;
  l_tc_flags = LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG | ( LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N') );
  unroll_hint = (res.N/res.bn)/res.upd_bf;
  res.gemm_upd = libxsmm_bsmmdispatch_reducebatch_strd_unroll(updM, updN, res.bn, res.bk*res.bn*sizeof(libxsmm_bfloat16), res.bc*res.bn*sizeof(libxsmm_bfloat16), unroll_hint, &lda, &ldb, &ldc, &alpha, &beta, &l_flags, NULL);
  if ( res.gemm_upd == NULL ) {
    fprintf( stderr, "JIT for BRGEMM TPP gemm_upd failed. Bailing...!\n");
    exit(-1);
  }
  res.gemm_upd2 = libxsmm_bsmmdispatch_reducebatch_strd_unroll(updM, updN, res.bn, res.bk*res.bn*sizeof(libxsmm_bfloat16), res.bc*res.bn*sizeof(libxsmm_bfloat16), unroll_hint, &lda, &ldb, &ldc, &alpha, &zerobeta, &l_flags, NULL);
  if ( res.gemm_upd2 == NULL ) {
    fprintf( stderr, "JIT for BRGEMM TPP gemm_upd2 failed. Bailing...!\n");
    exit(-1);
  }
  l_flags = l_flags | LIBXSMM_GEMM_FLAG_VNNI_C;
  res.gemm_upd3 = libxsmm_bmmdispatch_reducebatch_strd_unroll(updM, updN, res.bn, res.bk*res.bn*sizeof(libxsmm_bfloat16), res.bc*res.bn*sizeof(libxsmm_bfloat16), unroll_hint, &lda, &ldb, &ldc, &alpha, &zerobeta, &l_flags, NULL);
  if ( res.gemm_upd3 == NULL ) {
    fprintf( stderr, "JIT for BRGEMM TPP gemm_upd3 failed. Bailing...!\n");
    exit(-1);
  }
  res.upd_config_kernel = libxsmm_bsmmdispatch(updM, updN, res.bn, &lda, &ldb, &ldc, NULL, &beta, &l_tc_flags, NULL);
  if ( res.upd_config_kernel == NULL ) {
    fprintf( stderr, "JIT for BRGEMM TPP upd_config_kernel failed. Bailing...!\n");
    exit(-1);
  }

  res.tilerelease_kernel = libxsmm_bsmmdispatch(res.bk, res.bk, res.bk, NULL, NULL, NULL, NULL, NULL, &l_tr_flags, NULL);
  if ( res.tilerelease_kernel == NULL ) {
    fprintf( stderr, "JIT for TPP tilerelease_kernel failed. Bailing...!\n");
    exit(-1);
  }

  /* init scratch */
  size_bwd_scratch = sizeof(float) * LIBXSMM_MAX(res.C * res.N, res.threads * res.bc * res.bn) + sizeof(libxsmm_bfloat16) * res.C * res.K;
  size_upd_scratch = sizeof(float) * LIBXSMM_MAX(res.C * res.K, res.threads * res.bc * res.bk) + sizeof(libxsmm_bfloat16) * res.threads * res.bk * res.bc + sizeof(libxsmm_bfloat16) * (res.N * (res.C + res.K));
  res.scratch_size = LIBXSMM_MAX(size_bwd_scratch, size_upd_scratch) + 2 * sizeof(libxsmm_bfloat16) * res.N * res.K;
  res.doutput_scratch_mark = LIBXSMM_MAX(size_bwd_scratch, size_upd_scratch) ;

  return res;
}

my_opt_config setup_my_opt(libxsmm_blasint C, libxsmm_blasint K, libxsmm_blasint bc, libxsmm_blasint bk,
                           libxsmm_blasint threads, float lr) {
  my_opt_config res;

  /* setting up some handle values */
  res.C = C;
  res.K = K;
  res.bc = bc;
  res.bk = bk;
  res.threads = threads;
  res.lr = lr;

  /* setting up the barrier */
  res.barrier = libxsmm_barrier_create(threads, 1);

  /* init scratch */
  res.scratch_size = 0;

  return res;
}

my_smax_fwd_config setup_my_smax_fwd(libxsmm_blasint N, libxsmm_blasint C, libxsmm_blasint bn, libxsmm_blasint bc,
                                     libxsmm_blasint threads) {
  my_smax_fwd_config res;

  /* setting up some handle values */
  res.C = C;
  res.N = N;
  res.bc = bc;
  res.bn = bn;
  res.threads = threads;

  /* setting up the barrier */
  res.barrier = libxsmm_barrier_create(threads, 1);

  /* init scratch */
  res.scratch_size = (sizeof(float)*res.C*res.N*2);;

  return res;
}

my_smax_bwd_config setup_my_smax_bwd(libxsmm_blasint N, libxsmm_blasint C, libxsmm_blasint bn, libxsmm_blasint bc,
                                     libxsmm_blasint threads, float loss_weight) {
  my_smax_bwd_config res;

  /* setting up some handle values */
  res.C = C;
  res.N = N;
  res.bc = bc;
  res.bn = bn;
  res.threads = threads;
  res.loss_weight = loss_weight;

  /* setting up the barrier */
  res.barrier = libxsmm_barrier_create(threads, 1);

  /* init scratch */
  res.scratch_size = (sizeof(float)*res.C*res.N*2);;

  return res;
}

void my_fc_fwd_exec( my_fc_fwd_config cfg, const libxsmm_bfloat16* wt_ptr, const libxsmm_bfloat16* in_act_ptr, libxsmm_bfloat16* out_act_ptr,
                     const libxsmm_bfloat16* bias_ptr, unsigned char* relu_ptr, int start_tid, int my_tid, void* scratch ) {
  const libxsmm_blasint nBlocksIFm = cfg.C / cfg.bc;
  const libxsmm_blasint nBlocksOFm = cfg.K / cfg.bk;
  const libxsmm_blasint nBlocksMB  = cfg.N / cfg.bn;
  const libxsmm_blasint bn = cfg.bn;
  const libxsmm_blasint bk = cfg.bk;
  const libxsmm_blasint lpb = 2;
  const libxsmm_blasint bc_lp = cfg.bc/lpb;
  /* const libxsmm_blasint bc = cfg.bc;*/
  libxsmm_blasint use_2d_blocking = cfg.fwd_2d_blocking;

  /* computing first logical thread */
  const libxsmm_blasint ltid = my_tid - start_tid;
  /* number of tasks that could be run in parallel */
  const libxsmm_blasint work = nBlocksOFm * nBlocksMB;
  /* compute chunk size */
  const libxsmm_blasint chunksize = (work % cfg.threads == 0) ? (work / cfg.threads) : ((work / cfg.threads) + 1);
  /* compute thr_begin and thr_end */
  const libxsmm_blasint thr_begin = (ltid * chunksize < work) ? (ltid * chunksize) : work;
  const libxsmm_blasint thr_end = ((ltid + 1) * chunksize < work) ? ((ltid + 1) * chunksize) : work;

  /* loop variables */
  libxsmm_blasint mb1ofm1 = 0, mb1 = 0, ofm1 = 0, ifm1 = 0;
  libxsmm_blasint im_tasks_per_thread = 0, in_tasks_per_thread = 0, my_in_start = 0, my_in_end = 0, my_im_start = 0, my_im_end = 0, my_row_id = 0, my_col_id = 0, row_teams = 0, column_teams = 0;
  LIBXSMM_VLA_DECL(4, libxsmm_bfloat16,       output,  out_act_ptr, nBlocksOFm, cfg.bn, cfg.bk);
  LIBXSMM_VLA_DECL(4, const libxsmm_bfloat16,  input,   in_act_ptr,  nBlocksIFm, cfg.bn, cfg.bc);
  LIBXSMM_VLA_DECL(5, const libxsmm_bfloat16, filter,       wt_ptr, nBlocksIFm, bc_lp, cfg.bk, lpb);
  LIBXSMM_VLA_DECL(4, float, output_f32, (float*)scratch, nBlocksOFm, bn, bk);
  libxsmm_meltw_gemm_param gemm_eltwise_params;
  libxsmm_blasint mb2 = 0;
  float* fp32_bias_scratch =  ((cfg.fuse_type & MY_ELTWISE_FUSE_BIAS) == MY_ELTWISE_FUSE_BIAS) ? (float*)scratch + ltid * cfg.K : NULL;
  LIBXSMM_VLA_DECL(2, const libxsmm_bfloat16, bias, ((cfg.fuse_type & MY_ELTWISE_FUSE_BIAS) == MY_ELTWISE_FUSE_BIAS) ? (libxsmm_bfloat16*) bias_ptr : NULL, cfg.bk);
  LIBXSMM_VLA_DECL(4, __mmask32,  relubitmask, ((cfg.fuse_type & MY_ELTWISE_FUSE_RELU) == MY_ELTWISE_FUSE_RELU) ? (__mmask32*)relu_ptr : NULL, nBlocksOFm, cfg.bn, cfg.bk/32);
  libxsmm_meltwfunction_cvtfp32bf16_act eltwise_kernel_act = cfg.fwd_cvtfp32bf16_relu_kernel;
  libxsmm_meltw_cvtfp32bf16_act_param   eltwise_params_act;
  libxsmm_meltwfunction_cvtfp32bf16     eltwise_kernel = cfg.fwd_cvtfp32bf16_kernel;
  libxsmm_meltw_cvtfp32bf16_param       eltwise_params;
  libxsmm_bmmfunction_reducebatch_strd_meltwfused bf16_batchreduce_kernel_zerobeta_fused_eltwise;

  unsigned long long  blocks = nBlocksIFm;
  libxsmm_blasint CB_BLOCKS = nBlocksIFm, BF = 1;

  if (((cfg.fuse_type & MY_ELTWISE_FUSE_BIAS) == MY_ELTWISE_FUSE_BIAS) && ((cfg.fuse_type & MY_ELTWISE_FUSE_RELU) == MY_ELTWISE_FUSE_RELU )) {
    bf16_batchreduce_kernel_zerobeta_fused_eltwise = cfg.gemm_fwd7;
  } else if ((cfg.fuse_type & MY_ELTWISE_FUSE_BIAS) == MY_ELTWISE_FUSE_BIAS) {
    bf16_batchreduce_kernel_zerobeta_fused_eltwise = cfg.gemm_fwd4;
  } else if ((cfg.fuse_type & MY_ELTWISE_FUSE_RELU) == MY_ELTWISE_FUSE_RELU) {
    bf16_batchreduce_kernel_zerobeta_fused_eltwise = cfg.gemm_fwd5;
  }

  BF = cfg.fwd_bf;
  CB_BLOCKS = nBlocksIFm/BF;
  blocks = CB_BLOCKS;

  if (use_2d_blocking == 1) {
    row_teams = cfg.fwd_row_teams;
    column_teams = cfg.fwd_column_teams;
    my_col_id = ltid % column_teams;
    my_row_id = ltid / column_teams;
    im_tasks_per_thread = (nBlocksMB + row_teams-1)/row_teams;
    in_tasks_per_thread = (nBlocksOFm + column_teams-1)/column_teams;
    my_im_start = LIBXSMM_MIN( my_row_id * im_tasks_per_thread, nBlocksMB);
    my_im_end = LIBXSMM_MIN( (my_row_id+1) * im_tasks_per_thread, nBlocksMB);
    my_in_start = LIBXSMM_MIN( my_col_id * in_tasks_per_thread, nBlocksOFm);
    my_in_end = LIBXSMM_MIN( (my_col_id+1) * in_tasks_per_thread, nBlocksOFm);
  }

  /* lazy barrier init */
  libxsmm_barrier_init(cfg.barrier, ltid);

  cfg.fwd_config_kernel(NULL, NULL, NULL);

  if (use_2d_blocking == 1) {
    if (BF > 1) {
      for ( ifm1 = 0; ifm1 < BF; ++ifm1 ) {
        for (ofm1 = my_in_start; ofm1 < my_in_end; ++ofm1) {
          for (mb1 = my_im_start; mb1 < my_im_end; ++mb1) {
            if ( ifm1 == 0 ) {
              if ( (cfg.fuse_type & MY_ELTWISE_FUSE_BIAS) == MY_ELTWISE_FUSE_BIAS ) {
                for ( mb2 = 0; mb2 <cfg.bn; ++mb2 ) {
                  LIBXSMM_DNN_CONVERT_BUFFER_BF16_F32( &LIBXSMM_VLA_ACCESS(2, bias, ofm1, 0,cfg.bk), &LIBXSMM_VLA_ACCESS(4, output_f32, mb1, ofm1, mb2, 0, nBlocksOFm,cfg.bn,cfg.bk), cfg.bk );
                }
              } else {
                memset(&LIBXSMM_VLA_ACCESS(4, output_f32, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk), 0, cfg.bn*cfg.bk*sizeof(float));
              }
            }

            cfg.gemm_fwd( &LIBXSMM_VLA_ACCESS(5, filter, ofm1, ifm1*CB_BLOCKS, 0, 0, 0, nBlocksIFm, bc_lp, cfg.bk, lpb),
                &LIBXSMM_VLA_ACCESS(4, input,  mb1, ifm1*CB_BLOCKS, 0, 0, nBlocksIFm, cfg.bn, cfg.bc),
                &LIBXSMM_VLA_ACCESS(4, output_f32, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk), &blocks);

            if ( ifm1 == BF-1  ) {
              if ( (cfg.fuse_type & MY_ELTWISE_FUSE_RELU) == MY_ELTWISE_FUSE_RELU ) {
                eltwise_params_act.in_ptr = &LIBXSMM_VLA_ACCESS(4, output_f32, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk);
                eltwise_params_act.out_ptr = &LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk);
                eltwise_params_act.actstore_ptr = &LIBXSMM_VLA_ACCESS(4, relubitmask, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk/32);
                eltwise_kernel_act(&eltwise_params_act);
              } else {
                eltwise_params.in_ptr = &LIBXSMM_VLA_ACCESS(4, output_f32, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk);
                eltwise_params.out_ptr = &LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk);
                eltwise_kernel(&eltwise_params);
              }
            }
          }
        }
      }
    } else {
      if ( (cfg.fuse_type & MY_ELTWISE_FUSE_BIAS) == MY_ELTWISE_FUSE_BIAS ) {
        LIBXSMM_DNN_CONVERT_BUFFER_BF16_F32( &LIBXSMM_VLA_ACCESS(2, bias, 0, 0,cfg.bk), fp32_bias_scratch, cfg.K );
      }
      for (ofm1 = my_in_start; ofm1 < my_in_end; ++ofm1) {
        for (mb1 = my_im_start; mb1 < my_im_end; ++mb1) {
          if ( ((cfg.fuse_type & MY_ELTWISE_FUSE_BIAS) == MY_ELTWISE_FUSE_BIAS) || ((cfg.fuse_type & MY_ELTWISE_FUSE_RELU) == MY_ELTWISE_FUSE_RELU )) {
            if ((cfg.fuse_type & MY_ELTWISE_FUSE_BIAS) == MY_ELTWISE_FUSE_BIAS) {
              gemm_eltwise_params.bias_ptr  = (float*) fp32_bias_scratch + ofm1 * cfg.bk;
            }
            if ((cfg.fuse_type & MY_ELTWISE_FUSE_RELU) == MY_ELTWISE_FUSE_RELU) {
              gemm_eltwise_params.out_ptr   = &LIBXSMM_VLA_ACCESS(4, relubitmask, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk/32);
            }
            bf16_batchreduce_kernel_zerobeta_fused_eltwise( &LIBXSMM_VLA_ACCESS(5, filter, ofm1, 0, 0, 0, 0, nBlocksIFm, bc_lp, cfg.bk, lpb),
              &LIBXSMM_VLA_ACCESS(4, input,  mb1, 0,  0, 0, nBlocksIFm, cfg.bn, cfg.bc),
              &LIBXSMM_VLA_ACCESS(4, output, mb1,  ofm1, 0, 0, nBlocksOFm, bn, bk), &blocks, &gemm_eltwise_params);
          } else {
            cfg.gemm_fwd3( &LIBXSMM_VLA_ACCESS(5, filter, ofm1, 0, 0, 0, 0, nBlocksIFm, bc_lp, cfg.bk, lpb),
              &LIBXSMM_VLA_ACCESS(4, input,  mb1, 0,  0, 0, nBlocksIFm, cfg.bn, cfg.bc),
              &LIBXSMM_VLA_ACCESS(4, output, mb1,  ofm1, 0, 0, nBlocksOFm, bn, bk), &blocks);
          }
        }
      }
    }
  } else {
    if (BF > 1) {
      for ( ifm1 = 0; ifm1 < BF; ++ifm1 ) {
        for ( mb1ofm1 = thr_begin; mb1ofm1 < thr_end; ++mb1ofm1 ) {
          mb1  = mb1ofm1%nBlocksMB;
          ofm1 = mb1ofm1/nBlocksMB;
          /* Initialize libxsmm_blasintermediate f32 tensor */
          if ( ifm1 == 0 ) {
            if ( (cfg.fuse_type & MY_ELTWISE_FUSE_BIAS) == MY_ELTWISE_FUSE_BIAS ) {
              for ( mb2 = 0; mb2 <cfg.bn; ++mb2 ) {
                LIBXSMM_DNN_CONVERT_BUFFER_BF16_F32( &LIBXSMM_VLA_ACCESS(2, bias, ofm1, 0,cfg.bk), &LIBXSMM_VLA_ACCESS(4, output_f32, mb1, ofm1, mb2, 0, nBlocksOFm,cfg.bn,cfg.bk), cfg.bk );
              }
            } else {
              memset(&LIBXSMM_VLA_ACCESS(4, output_f32, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk), 0, cfg.bn*cfg.bk*sizeof(float));
            }
          }
          cfg.gemm_fwd( &LIBXSMM_VLA_ACCESS(5, filter, ofm1, ifm1*CB_BLOCKS, 0, 0, 0, nBlocksIFm, bc_lp, cfg.bk, lpb),
              &LIBXSMM_VLA_ACCESS(4, input,  mb1, ifm1*CB_BLOCKS, 0, 0, nBlocksIFm, cfg.bn, cfg.bc),
              &LIBXSMM_VLA_ACCESS(4, output_f32, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk), &blocks);

          if ( ifm1 == BF-1  ) {
            if ( (cfg.fuse_type & MY_ELTWISE_FUSE_RELU) == MY_ELTWISE_FUSE_RELU ) {
              eltwise_params_act.in_ptr = &LIBXSMM_VLA_ACCESS(4, output_f32, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk);
              eltwise_params_act.out_ptr = &LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk);
              eltwise_params_act.actstore_ptr = &LIBXSMM_VLA_ACCESS(4, relubitmask, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk/32);
              eltwise_kernel_act(&eltwise_params_act);
            } else {
              eltwise_params.in_ptr = &LIBXSMM_VLA_ACCESS(4, output_f32, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk);
              eltwise_params.out_ptr = &LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk);
              eltwise_kernel(&eltwise_params);
            }
          }
        }
      }
    } else {
      if ( (cfg.fuse_type & MY_ELTWISE_FUSE_BIAS) == MY_ELTWISE_FUSE_BIAS ) {
        LIBXSMM_DNN_CONVERT_BUFFER_BF16_F32( &LIBXSMM_VLA_ACCESS(2, bias, 0, 0,cfg.bk), fp32_bias_scratch, cfg.K );
      }
      for ( mb1ofm1 = thr_begin; mb1ofm1 < thr_end; ++mb1ofm1 ) {
        mb1  = mb1ofm1%nBlocksMB;
        ofm1 = mb1ofm1/nBlocksMB;
        if ( ((cfg.fuse_type & MY_ELTWISE_FUSE_BIAS) == MY_ELTWISE_FUSE_BIAS) || ((cfg.fuse_type & MY_ELTWISE_FUSE_RELU) == MY_ELTWISE_FUSE_RELU )) {
          if ((cfg.fuse_type & MY_ELTWISE_FUSE_BIAS) == MY_ELTWISE_FUSE_BIAS) {
            gemm_eltwise_params.bias_ptr  = (float*) fp32_bias_scratch + ofm1 * cfg.bk;
          }
          if ((cfg.fuse_type & MY_ELTWISE_FUSE_RELU) == MY_ELTWISE_FUSE_RELU) {
            gemm_eltwise_params.out_ptr   = &LIBXSMM_VLA_ACCESS(4, relubitmask, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk/32);
          }
          bf16_batchreduce_kernel_zerobeta_fused_eltwise( &LIBXSMM_VLA_ACCESS(5, filter, ofm1, 0, 0, 0, 0, nBlocksIFm, bc_lp, cfg.bk, lpb),
            &LIBXSMM_VLA_ACCESS(4, input,  mb1, 0,  0, 0, nBlocksIFm, cfg.bn, cfg.bc),
            &LIBXSMM_VLA_ACCESS(4, output, mb1,  ofm1, 0, 0, nBlocksOFm, bn, bk), &blocks, &gemm_eltwise_params);
        } else {
          cfg.gemm_fwd3( &LIBXSMM_VLA_ACCESS(5, filter, ofm1, 0, 0, 0, 0, nBlocksIFm, bc_lp, cfg.bk, lpb),
            &LIBXSMM_VLA_ACCESS(4, input,  mb1, 0,  0, 0, nBlocksIFm, cfg.bn, cfg.bc),
            &LIBXSMM_VLA_ACCESS(4, output, mb1,  ofm1, 0, 0, nBlocksOFm, bn, bk), &blocks);
        }
      }
    }
  }

  cfg.tilerelease_kernel(NULL, NULL, NULL);
  libxsmm_barrier_wait(cfg.barrier, ltid);
}

void my_fc_bwd_exec( my_fc_bwd_config cfg, const libxsmm_bfloat16* wt_ptr, libxsmm_bfloat16* din_act_ptr,
                     const libxsmm_bfloat16* dout_act_ptr, libxsmm_bfloat16* dwt_ptr, const libxsmm_bfloat16* in_act_ptr,
                     libxsmm_bfloat16* dbias_ptr, const unsigned char* relu_ptr, my_pass pass, int start_tid, int my_tid, void* scratch ) {
  /* size variables, all const */
  /* here we assume that input and output blocking is similar */
  const libxsmm_blasint bn = cfg.bn;
  const libxsmm_blasint bk = cfg.bk;
  const libxsmm_blasint bc = cfg.bc;
  libxsmm_blasint lpb = 2;
  const libxsmm_blasint bc_lp = bc/lpb;
  const libxsmm_blasint bk_lp = bk/lpb;
  const libxsmm_blasint bn_lp = bn/lpb;
  const libxsmm_blasint nBlocksIFm = cfg.C / cfg.bc;
  const libxsmm_blasint nBlocksOFm = cfg.K / cfg.bk;
  const libxsmm_blasint nBlocksMB  = cfg.N / cfg.bn;
  libxsmm_blasint mb1ofm1 = 0, mb1 = 0, ofm1 = 0, mb2 = 0, ofm2 = 0;
  libxsmm_blasint iteri = 0, iterj = 0;
  libxsmm_blasint performed_doutput_transpose = 0;

  /* computing first logical thread */
  const libxsmm_blasint ltid = my_tid - start_tid;

  /* number of tasks for transpose that could be run in parallel */
  const libxsmm_blasint eltwise_work = nBlocksOFm * nBlocksMB;
  /* compute chunk size */
  const libxsmm_blasint eltwise_chunksize = (eltwise_work % cfg.threads == 0) ? (eltwise_work / cfg.threads) : ((eltwise_work / cfg.threads) + 1);
  /* compute thr_begin and thr_end */
  const libxsmm_blasint eltwise_thr_begin = (ltid * eltwise_chunksize < eltwise_work) ? (ltid * eltwise_chunksize) : eltwise_work;
  const libxsmm_blasint eltwise_thr_end = ((ltid + 1) * eltwise_chunksize < eltwise_work) ? ((ltid + 1) * eltwise_chunksize) : eltwise_work;

  /* number of tasks for transpose that could be run in parallel */
  const libxsmm_blasint dbias_work = nBlocksOFm;
  /* compute chunk size */
  const libxsmm_blasint dbias_chunksize = (dbias_work % cfg.threads == 0) ? (dbias_work / cfg.threads) : ((dbias_work / cfg.threads) + 1);
  /* compute thr_begin and thr_end */
  const libxsmm_blasint dbias_thr_begin = (ltid * dbias_chunksize < dbias_work) ? (ltid * dbias_chunksize) : dbias_work;
  const libxsmm_blasint dbias_thr_end = ((ltid + 1) * dbias_chunksize < dbias_work) ? ((ltid + 1) * dbias_chunksize) : dbias_work;

  LIBXSMM_VLA_DECL(2, libxsmm_bfloat16, dbias, ((cfg.fuse_type & MY_ELTWISE_FUSE_BIAS) == MY_ELTWISE_FUSE_BIAS) ? (libxsmm_bfloat16*) dbias_ptr : NULL, cfg.bk);
  LIBXSMM_VLA_DECL(4,     __mmask32, relubitmask, ((cfg.fuse_type & MY_ELTWISE_FUSE_RELU) == MY_ELTWISE_FUSE_RELU) ? (__mmask32*)relu_ptr : NULL, nBlocksOFm, cfg.bn, cfg.bk/32);

  libxsmm_bfloat16 *grad_output_ptr = (((cfg.fuse_type & MY_ELTWISE_FUSE_RELU) == MY_ELTWISE_FUSE_RELU)) ? (libxsmm_bfloat16*)((char*)scratch + cfg.doutput_scratch_mark) : (libxsmm_bfloat16*)dout_act_ptr;
  libxsmm_bfloat16 *tr_doutput_ptr = (((cfg.fuse_type & MY_ELTWISE_FUSE_RELU) == MY_ELTWISE_FUSE_RELU)) ? (libxsmm_bfloat16*)grad_output_ptr + cfg.N * cfg.K : (libxsmm_bfloat16*)scratch;
  LIBXSMM_VLA_DECL(4, const libxsmm_bfloat16,   doutput_orig, (libxsmm_bfloat16*)dout_act_ptr, nBlocksOFm, bn, bk);
  libxsmm_meltw_relu_param   relu_params;
  libxsmm_meltwfunction_relu relu_kernel = cfg.bwd_relu_kernel;
  LIBXSMM_VLA_DECL(4, libxsmm_bfloat16,   doutput, grad_output_ptr, nBlocksOFm, bn, bk);
  LIBXSMM_VLA_DECL(5, libxsmm_bfloat16, doutput_tr, tr_doutput_ptr, nBlocksMB, bn_lp, bk, lpb);

  libxsmm_meltwfunction_cvtfp32bf16 eltwise_kernel = cfg.bwd_cvtfp32bf16_kernel;
  libxsmm_meltw_cvtfp32bf16_param   eltwise_params;

  /* lazy barrier init */
  libxsmm_barrier_init(cfg.barrier, ltid);
  cfg.bwd_config_kernel(NULL, NULL, NULL);

  /* Apply to doutput potential fusions */
  if (((cfg.fuse_type & MY_ELTWISE_FUSE_RELU) == MY_ELTWISE_FUSE_RELU)) {
    for ( mb1ofm1 = eltwise_thr_begin; mb1ofm1 < eltwise_thr_end; ++mb1ofm1 ) {
      mb1  = mb1ofm1/nBlocksOFm;
      ofm1 = mb1ofm1%nBlocksOFm;

      relu_params.in_ptr   = &LIBXSMM_VLA_ACCESS(4, doutput_orig, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk);
      relu_params.out_ptr  = &LIBXSMM_VLA_ACCESS(4, doutput, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk);
      relu_params.mask_ptr = &LIBXSMM_VLA_ACCESS(4, relubitmask, mb1, ofm1, 0, 0, nBlocksOFm, cfg.bn, cfg.bk/32);
      relu_kernel(&relu_params);

      /* If in UPD pass, also perform transpose of doutput  */
      if ( (pass & MY_PASS_BWD_W) == MY_PASS_BWD_W ) {
        my_bf16_vnni_reformat((libxsmm_bfloat16*)&LIBXSMM_VLA_ACCESS(4, doutput,  mb1, ofm1, 0, 0, nBlocksOFm, bn, bk), &LIBXSMM_VLA_ACCESS(5, doutput_tr, ofm1, mb1, 0, 0, 0, nBlocksMB, bn_lp, bk, lpb), bk, bn, bk, bn);
      }
    }

    if ( (pass & MY_PASS_BWD_W) == MY_PASS_BWD_W ) {
      performed_doutput_transpose = 1;
    }
    libxsmm_barrier_wait(cfg.barrier, ltid);
  }


  /* Accumulation of bias happens in f32 */
  if (((cfg.fuse_type & MY_ELTWISE_FUSE_BIAS) == MY_ELTWISE_FUSE_BIAS)) {
    float *scratch_dbias = (float*) ((libxsmm_bfloat16*)scratch + cfg.N * (cfg.K + cfg.C) + ltid * bk * 2);
    if (cfg.bk % 32 == 0) {
      for ( ofm1 = dbias_thr_begin; ofm1 < dbias_thr_end; ++ofm1 ) {
        for ( iterj = 0; iterj < cfg.bk; iterj += 32 ) {
          __m512 doutput_reg_0, doutput_reg_1, dbias_reg_0, dbias_reg_1;
          dbias_reg_0 = _mm512_setzero_ps();
          dbias_reg_1 = _mm512_setzero_ps();
          for ( mb1 = 0; mb1 < nBlocksMB; ++mb1 ) {
            for ( iteri = 0; iteri < cfg.bn; ++iteri ) {
              doutput_reg_0 = _mm512_loadcvt_bf16_fp32(&LIBXSMM_VLA_ACCESS(4,  doutput, mb1, ofm1, iteri, iterj, nBlocksOFm, cfg.bn, cfg.bk));
              doutput_reg_1 = _mm512_loadcvt_bf16_fp32(&LIBXSMM_VLA_ACCESS(4,  doutput, mb1, ofm1, iteri, iterj+16, nBlocksOFm, cfg.bn, cfg.bk));
              dbias_reg_0 = _mm512_add_ps(dbias_reg_0, doutput_reg_0);
              dbias_reg_1 = _mm512_add_ps(dbias_reg_1, doutput_reg_1);
            }
          }
          LIBXSMM_INTRINSICS_MM512_STREAM_SI512(&LIBXSMM_VLA_ACCESS( 2, dbias, ofm1, iterj, cfg.bk), LIBXSMM_INTRINSISCS_MM512_CVTNE2PS_PBH(dbias_reg_1, dbias_reg_0));
        }
      }
    } else {
      for ( ofm1 = dbias_thr_begin; ofm1 < dbias_thr_end; ++ofm1 ) {
        for ( iterj = 0; iterj < cfg.bk; ++iterj ) {
          scratch_dbias[iterj] = 0.0;
        }
        for ( mb1 = 0; mb1 < nBlocksMB; ++mb1 ) {
          for ( iteri = 0; iteri < cfg.bn; ++iteri ) {
            for ( iterj = 0; iterj < cfg.bk; ++iterj ) {
              float doutput_f32 = 0;
              libxsmm_bfloat16_hp tmp;
              tmp.i[0] = 0;
              tmp.i[1] = LIBXSMM_VLA_ACCESS(4,  doutput, mb1, ofm1, iteri, iterj, nBlocksOFm, cfg.bn, cfg.bk);
              doutput_f32 = tmp.f;
              scratch_dbias[iterj] += doutput_f32;
            }
          }
        }
        libxsmm_rne_convert_fp32_bf16(scratch_dbias, &LIBXSMM_VLA_ACCESS( 2, dbias, ofm1, 0, cfg.bk ), cfg.bk);
      }
    }

    /* wait for eltwise to finish */
    libxsmm_barrier_wait(cfg.barrier, ltid);
  }

  if ( (pass & MY_PASS_BWD_D) == MY_PASS_BWD_D ){
    libxsmm_blasint use_2d_blocking = cfg.bwd_2d_blocking;

    /* number of tasks that could be run in parallel */
    const libxsmm_blasint work = nBlocksIFm * nBlocksMB;
    /* compute chunk size */
    const libxsmm_blasint chunksize = (work % cfg.threads == 0) ? (work / cfg.threads) : ((work / cfg.threads) + 1);
    /* compute thr_begin and thr_end */
    const libxsmm_blasint thr_begin = (ltid * chunksize < work) ? (ltid * chunksize) : work;
    const libxsmm_blasint thr_end = ((ltid + 1) * chunksize < work) ? ((ltid + 1) * chunksize) : work;

    /* number of tasks for transpose that could be run in parallel */
    const libxsmm_blasint transpose_work = nBlocksIFm * nBlocksOFm;
    /* compute chunk size */
    const libxsmm_blasint transpose_chunksize = (transpose_work % cfg.threads == 0) ? (transpose_work / cfg.threads) : ((transpose_work / cfg.threads) + 1);
    /* compute thr_begin and thr_end */
    const libxsmm_blasint transpose_thr_begin = (ltid * transpose_chunksize < transpose_work) ? (ltid * transpose_chunksize) : transpose_work;
    const libxsmm_blasint transpose_thr_end = ((ltid + 1) * transpose_chunksize < transpose_work) ? ((ltid + 1) * transpose_chunksize) : transpose_work;

    /* loop variables */
    libxsmm_blasint ifm1 = 0, ifm2 = 0, ifm1ofm1 = 0, mb1ifm1 = 0;
    libxsmm_blasint im_tasks_per_thread = 0, in_tasks_per_thread = 0, my_in_start = 0, my_in_end = 0, my_im_start = 0, my_im_end = 0, my_row_id = 0, my_col_id = 0, row_teams = 0, column_teams = 0;

    LIBXSMM_VLA_DECL(5, const libxsmm_bfloat16, filter, (libxsmm_bfloat16*)wt_ptr, nBlocksIFm, bc_lp, bk, lpb);
    LIBXSMM_VLA_DECL(4,        libxsmm_bfloat16,    dinput, (libxsmm_bfloat16* )din_act_ptr, nBlocksIFm, bn, bc);
    LIBXSMM_VLA_DECL(5,       libxsmm_bfloat16, filter_tr, (libxsmm_bfloat16*)scratch, nBlocksOFm, bk_lp, bc, lpb);
    float* temp_output = (float*)scratch + (cfg.C * cfg.K)/2;
    LIBXSMM_VLA_DECL(4,        float,    dinput_f32, (float*) temp_output, nBlocksIFm, bn, bc);

    unsigned long long  blocks = nBlocksOFm;
    libxsmm_blasint KB_BLOCKS = nBlocksOFm, BF = 1;
    BF = cfg.bwd_bf;
    KB_BLOCKS = nBlocksOFm/BF;
    blocks = KB_BLOCKS;

    if (use_2d_blocking == 1) {
      row_teams = cfg.bwd_row_teams;
      column_teams = cfg.bwd_column_teams;
      my_col_id = ltid % column_teams;
      my_row_id = ltid / column_teams;
      im_tasks_per_thread = (nBlocksMB + row_teams-1)/row_teams;
      in_tasks_per_thread = (nBlocksIFm + column_teams-1)/column_teams;
      my_im_start = LIBXSMM_MIN( my_row_id * im_tasks_per_thread, nBlocksMB);
      my_im_end = LIBXSMM_MIN( (my_row_id+1) * im_tasks_per_thread, nBlocksMB);
      my_in_start = LIBXSMM_MIN( my_col_id * in_tasks_per_thread, nBlocksIFm);
      my_in_end = LIBXSMM_MIN( (my_col_id+1) * in_tasks_per_thread, nBlocksIFm);
    }

    /* transpose weight */
    if ((bk % 16 == 0) && (bc % 16 == 0)) {
      for (ifm1ofm1 = transpose_thr_begin; ifm1ofm1 < transpose_thr_end; ++ifm1ofm1) {
        ofm1 = ifm1ofm1 / nBlocksIFm;
        ifm1 = ifm1ofm1 % nBlocksIFm;
        my_bf16_vnni_transpose((libxsmm_bfloat16*)&LIBXSMM_VLA_ACCESS(5, filter,  ofm1, ifm1, 0, 0, 0, nBlocksIFm, bc_lp, bk, lpb), (libxsmm_bfloat16*)&LIBXSMM_VLA_ACCESS(5, filter_tr, ifm1, ofm1, 0, 0, 0, nBlocksOFm, bk_lp, bc, lpb), bk, bc, bk, bc);
      }
    } else {
      for (ifm1ofm1 = transpose_thr_begin; ifm1ofm1 < transpose_thr_end; ++ifm1ofm1) {
        ofm1 = ifm1ofm1 / nBlocksIFm;
        ifm1 = ifm1ofm1 % nBlocksIFm;
        for (ofm2 = 0; ofm2 < bk; ++ofm2) {
          for (ifm2 = 0; ifm2 < bc; ++ifm2) {
            LIBXSMM_VLA_ACCESS(5, filter_tr, ifm1, ofm1, ofm2/lpb, ifm2, ofm2%lpb, nBlocksOFm, bk_lp, bc, lpb) = LIBXSMM_VLA_ACCESS(5, filter,  ofm1, ifm1, ifm2/lpb, ofm2, ifm2%lpb, nBlocksIFm, bc_lp, bk, lpb);
          }
        }
      }
    }

    /* wait for transpose to finish */
    libxsmm_barrier_wait(cfg.barrier, ltid);

    if (use_2d_blocking == 1) {
      if (BF > 1) {
        for ( ofm1 = 0; ofm1 < BF; ++ofm1 ) {
          for (ifm1 = my_in_start; ifm1 < my_in_end; ++ifm1) {
            for (mb1 = my_im_start; mb1 < my_im_end; ++mb1) {
              /* Initialize libxsmm_blasintermediate f32 tensor */
              if ( ofm1 == 0 ) {
                memset(&LIBXSMM_VLA_ACCESS(4, dinput_f32, mb1, ifm1, 0, 0, nBlocksIFm, bn, bc), 0, bn*bc*sizeof(float));
              }
              cfg.gemm_bwd( &LIBXSMM_VLA_ACCESS(5, filter_tr, ifm1, ofm1*KB_BLOCKS, 0, 0, 0, nBlocksOFm, bk_lp, bc, lpb),
                  &LIBXSMM_VLA_ACCESS(4, doutput,   mb1,  ofm1*KB_BLOCKS, 0, 0, nBlocksOFm, bn, bk),
                  &LIBXSMM_VLA_ACCESS(4, dinput_f32,    mb1,  ifm1, 0, 0, nBlocksIFm, bn, bc), &blocks);
              /* downconvert libxsmm_blasintermediate f32 tensor to bf 16 and store to final C */
              if ( ofm1 == BF-1  ) {
                eltwise_params.in_ptr = &LIBXSMM_VLA_ACCESS(4, dinput_f32,    mb1,  ifm1, 0, 0, nBlocksIFm, bn, bc);
                eltwise_params.out_ptr = &LIBXSMM_VLA_ACCESS(4, dinput,    mb1,  ifm1, 0, 0, nBlocksIFm, bn, bc);
                eltwise_kernel(&eltwise_params);
              }
            }
          }
        }
      } else {
        for (ifm1 = my_in_start; ifm1 < my_in_end; ++ifm1) {
          for (mb1 = my_im_start; mb1 < my_im_end; ++mb1) {
            cfg.gemm_bwd3( &LIBXSMM_VLA_ACCESS(5, filter_tr, ifm1, 0, 0, 0, 0, nBlocksOFm, bk_lp, bc, lpb),
                &LIBXSMM_VLA_ACCESS(4, doutput,   mb1,  0, 0, 0, nBlocksOFm, bn, bk),
                &LIBXSMM_VLA_ACCESS(4, dinput,    mb1,  ifm1, 0, 0, nBlocksIFm, bn, bc), &blocks);
          }
        }
      }
    } else {
      if (BF > 1) {
        for ( ofm1 = 0; ofm1 < BF; ++ofm1 ) {
          for ( mb1ifm1 = thr_begin; mb1ifm1 < thr_end; ++mb1ifm1 ) {
            mb1  = mb1ifm1%nBlocksMB;
            ifm1 = mb1ifm1/nBlocksMB;
            /* Initialize libxsmm_blasintermediate f32 tensor */
            if ( ofm1 == 0 ) {
              memset(&LIBXSMM_VLA_ACCESS(4, dinput_f32, mb1, ifm1, 0, 0, nBlocksIFm, bn, bc), 0, bn*bc*sizeof(float));
            }
            cfg.gemm_bwd( &LIBXSMM_VLA_ACCESS(5, filter_tr, ifm1, ofm1*KB_BLOCKS, 0, 0, 0, nBlocksOFm, bk_lp, bc, lpb),
                &LIBXSMM_VLA_ACCESS(4, doutput,   mb1,  ofm1*KB_BLOCKS, 0, 0, nBlocksOFm, bn, bk),
                &LIBXSMM_VLA_ACCESS(4, dinput_f32,    mb1,  ifm1, 0, 0, nBlocksIFm, bn, bc), &blocks);
            /* downconvert libxsmm_blasintermediate f32 tensor to bf 16 and store to final C */
            if ( ofm1 == BF-1  ) {
                eltwise_params.in_ptr = &LIBXSMM_VLA_ACCESS(4, dinput_f32,    mb1,  ifm1, 0, 0, nBlocksIFm, bn, bc);
                eltwise_params.out_ptr = &LIBXSMM_VLA_ACCESS(4, dinput,    mb1,  ifm1, 0, 0, nBlocksIFm, bn, bc);
                eltwise_kernel(&eltwise_params);
            }
          }
        }
      } else {
        for ( mb1ifm1 = thr_begin; mb1ifm1 < thr_end; ++mb1ifm1 ) {
          mb1  = mb1ifm1%nBlocksMB;
          ifm1 = mb1ifm1/nBlocksMB;
          cfg.gemm_bwd3( &LIBXSMM_VLA_ACCESS(5, filter_tr, ifm1, 0, 0, 0, 0, nBlocksOFm, bk_lp, bc, lpb),
              &LIBXSMM_VLA_ACCESS(4, doutput,   mb1,  0, 0, 0, nBlocksOFm, bn, bk),
              &LIBXSMM_VLA_ACCESS(4, dinput,    mb1,  ifm1, 0, 0, nBlocksIFm, bn, bc), &blocks);
        }
      }
    }

    libxsmm_barrier_wait(cfg.barrier, ltid);
  }

  if ( (pass & MY_PASS_BWD_W) == MY_PASS_BWD_W ) {
    /* number of tasks that could be run in parallel */
    const libxsmm_blasint ofm_subtasks = (cfg.upd_2d_blocking == 1) ? 1 : cfg.ofm_subtasks;
    const libxsmm_blasint ifm_subtasks = (cfg.upd_2d_blocking == 1) ? 1 : cfg.ifm_subtasks;
    const libxsmm_blasint bbk = (cfg.upd_2d_blocking == 1) ? bk : bk/ofm_subtasks;
    const libxsmm_blasint bbc = (cfg.upd_2d_blocking == 1) ? bc : bc/ifm_subtasks;
    const libxsmm_blasint work = nBlocksIFm * ifm_subtasks * nBlocksOFm * ofm_subtasks;
    const libxsmm_blasint Cck_work = nBlocksIFm * ifm_subtasks * ofm_subtasks;
    const libxsmm_blasint Cc_work = nBlocksIFm * ifm_subtasks;

    /* 2D blocking parameters  */
    libxsmm_blasint use_2d_blocking = cfg.upd_2d_blocking;
    libxsmm_blasint im_tasks_per_thread = 0, in_tasks_per_thread = 0, my_in_start = 0, my_in_end = 0, my_im_start = 0, my_im_end = 0, my_row_id = 0, my_col_id = 0, row_teams = 0, column_teams = 0;

    /* compute chunk size */
    const libxsmm_blasint chunksize = (work % cfg.threads == 0) ? (work / cfg.threads) : ((work / cfg.threads) + 1);
    /* compute thr_begin and thr_end */
    const libxsmm_blasint thr_begin = (ltid * chunksize < work) ? (ltid * chunksize) : work;
    const libxsmm_blasint thr_end = ((ltid + 1) * chunksize < work) ? ((ltid + 1) * chunksize) : work;
    libxsmm_blasint BF = cfg.upd_bf;

    /* loop variables */
    libxsmm_blasint ifm1ofm1 = 0, ifm1 = 0, ifm2 = 0, bfn = 0, ii = 0, jj = 0, mb1ifm1 = 0, jc = 0, jk = 0;

    /* Batch reduce related variables */
    unsigned long long  blocks = nBlocksMB/BF;

    LIBXSMM_VLA_DECL(4, const libxsmm_bfloat16,  input,    (libxsmm_bfloat16* )in_act_ptr, nBlocksIFm, bn, bc);
    LIBXSMM_VLA_DECL(5,       libxsmm_bfloat16, dfilter,  (libxsmm_bfloat16*)dwt_ptr, nBlocksIFm, bc_lp, bk, lpb);

    /* Set up tensors for transposing/scratch before vnni reformatting dfilter */
    libxsmm_bfloat16  *tr_inp_ptr = (libxsmm_bfloat16*) ((libxsmm_bfloat16*)scratch + cfg.N * cfg.K);
    float               *dfilter_f32_ptr = (float*) ((libxsmm_bfloat16*)tr_inp_ptr + cfg.N * cfg.C);
    libxsmm_bfloat16 *dfilter_scratch = (libxsmm_bfloat16*) ((float*)dfilter_f32_ptr + cfg.C * cfg.K) + ltid * bc * bk;

    LIBXSMM_VLA_DECL(4, libxsmm_bfloat16,  input_tr,    (libxsmm_bfloat16*)tr_inp_ptr, nBlocksMB, bc, bn);
    LIBXSMM_VLA_DECL(4,       float, dfilter_f32,  (float*)dfilter_f32_ptr, nBlocksIFm, bc, bk);
    LIBXSMM_VLA_DECL(2, libxsmm_bfloat16, dfilter_block,  (libxsmm_bfloat16*)dfilter_scratch, bk);

    const libxsmm_blasint tr_out_work = nBlocksMB * nBlocksOFm;
    const libxsmm_blasint tr_out_chunksize = (tr_out_work % cfg.threads == 0) ? (tr_out_work / cfg.threads) : ((tr_out_work / cfg.threads) + 1);
    const libxsmm_blasint tr_out_thr_begin = (ltid * tr_out_chunksize < tr_out_work) ? (ltid * tr_out_chunksize) : tr_out_work;
    const libxsmm_blasint tr_out_thr_end = ((ltid + 1) * tr_out_chunksize < tr_out_work) ? ((ltid + 1) * tr_out_chunksize) : tr_out_work;

    const libxsmm_blasint tr_inp_work = nBlocksMB * nBlocksIFm;
    const libxsmm_blasint tr_inp_chunksize = (tr_inp_work % cfg.threads == 0) ? (tr_inp_work / cfg.threads) : ((tr_inp_work / cfg.threads) + 1);
    const libxsmm_blasint tr_inp_thr_begin = (ltid * tr_inp_chunksize < tr_inp_work) ? (ltid * tr_inp_chunksize) : tr_inp_work;
    const libxsmm_blasint tr_inp_thr_end = ((ltid + 1) * tr_inp_chunksize < tr_inp_work) ? ((ltid + 1) * tr_inp_chunksize) : tr_inp_work;

    /* These are used for the vnni reformatting of the f32 output  */
    __m512 a01, b01;
    __m512i c01 = LIBXSMM_INTRINSICS_MM512_UNDEFINED_EPI32();
    const __m512i perm_index = LIBXSMM_INTRINSICS_MM512_SET_EPI16(31, 15, 30, 14, 29, 13, 28, 12, 27, 11, 26, 10, 25, 9, 24, 8, 23, 7, 22, 6, 21, 5, 20, 4, 19, 3, 18, 2, 17, 1, 16, 0);

    if (use_2d_blocking == 1) {
      row_teams = cfg.upd_row_teams;
      column_teams = cfg.upd_column_teams;
      my_col_id = ltid % column_teams;
      my_row_id = ltid / column_teams;
      im_tasks_per_thread = (nBlocksIFm + row_teams-1)/row_teams;
      in_tasks_per_thread = (nBlocksOFm + column_teams-1)/column_teams;
      my_im_start = LIBXSMM_MIN( my_row_id * im_tasks_per_thread, nBlocksIFm);
      my_im_end = LIBXSMM_MIN( (my_row_id+1) * im_tasks_per_thread, nBlocksIFm);
      my_in_start = LIBXSMM_MIN( my_col_id * in_tasks_per_thread, nBlocksOFm);
      my_in_end = LIBXSMM_MIN( (my_col_id+1) * in_tasks_per_thread, nBlocksOFm);
    }

    /* Required upfront tranposes */
    if (bc % 32 == 0) {
      for (mb1ifm1 = tr_inp_thr_begin; mb1ifm1 < tr_inp_thr_end; mb1ifm1++) {
        mb1 = mb1ifm1%nBlocksMB;
        ifm1 = mb1ifm1/nBlocksMB;
        my_bf16_transpose((libxsmm_bfloat16*)&LIBXSMM_VLA_ACCESS(4, input, mb1, ifm1, 0, 0, nBlocksIFm, bn, bc), &LIBXSMM_VLA_ACCESS(4, input_tr, ifm1, mb1, 0, 0, nBlocksMB, bc, bn), bc, bn, bc, bn);
      }
    } else {
      for (mb1ifm1 = tr_inp_thr_begin; mb1ifm1 < tr_inp_thr_end; mb1ifm1++) {
        mb1 = mb1ifm1%nBlocksMB;
        ifm1 = mb1ifm1/nBlocksMB;
        for (mb2 = 0; mb2 < bn; mb2++) {
          for (ifm2 = 0; ifm2 < bc; ifm2++) {
            LIBXSMM_VLA_ACCESS(4, input_tr, ifm1, mb1, ifm2, mb2, nBlocksMB, bc, bn) = LIBXSMM_VLA_ACCESS(4, input, mb1, ifm1, mb2, ifm2, nBlocksIFm, bn, bc);
          }
        }
      }
    }

    if (performed_doutput_transpose == 0) {
      if (bk % 32 == 0) {
        for (mb1ofm1 = tr_out_thr_begin; mb1ofm1 < tr_out_thr_end; mb1ofm1++) {
          mb1 = mb1ofm1%nBlocksMB;
          ofm1 = mb1ofm1/nBlocksMB;
          my_bf16_vnni_reformat((libxsmm_bfloat16*)&LIBXSMM_VLA_ACCESS(4, doutput,  mb1, ofm1, 0, 0, nBlocksOFm, bn, bk), &LIBXSMM_VLA_ACCESS(5, doutput_tr, ofm1, mb1, 0, 0, 0, nBlocksMB, bn_lp, bk, lpb), bk, bn, bk, bn);
        }
      } else {
        for (mb1ofm1 = tr_out_thr_begin; mb1ofm1 < tr_out_thr_end; mb1ofm1++) {
          mb1 = mb1ofm1%nBlocksMB;
          ofm1 = mb1ofm1/nBlocksMB;
          for (mb2 = 0; mb2 < bn; mb2++) {
            for (ofm2 = 0; ofm2 < bk; ofm2++) {
              LIBXSMM_VLA_ACCESS(5, doutput_tr, ofm1, mb1, mb2/lpb, ofm2, mb2%lpb, nBlocksMB, bn_lp, bk, lpb) = LIBXSMM_VLA_ACCESS(4, doutput,  mb1, ofm1, mb2, ofm2, nBlocksOFm, bn, bk);
            }
          }
        }
      }
    }

    libxsmm_barrier_wait(cfg.barrier, ltid);

    if (use_2d_blocking == 1) {
      ifm2 = 0;
      ofm2 = 0;
      if (BF == 1) {
        for (ofm1 = my_in_start; ofm1 < my_in_end; ++ofm1) {
          for (ifm1 = my_im_start; ifm1 < my_im_end; ++ifm1) {
            cfg.gemm_upd3(&LIBXSMM_VLA_ACCESS(5, doutput_tr, ofm1, 0, 0, ofm2*bbk, 0, nBlocksMB, bn_lp, bk, lpb), &LIBXSMM_VLA_ACCESS(4, input_tr, ifm1, 0, ifm2*bbc, 0, nBlocksMB, bc, bn), &LIBXSMM_VLA_ACCESS(5, dfilter, ofm1, ifm1, 0, 0, 0, nBlocksIFm, bc_lp, bk, lpb), &blocks);
          }
        }
      } else {
        for (bfn = 0; bfn < BF; bfn++) {
          for (ofm1 = my_in_start; ofm1 < my_in_end; ++ofm1) {
            for (ifm1 = my_im_start; ifm1 < my_im_end; ++ifm1) {
              /* initialize current work task to zero */
              if (bfn == 0) {
                for (ii = 0; ii<bbc; ii++) {
                  for (jj = 0; jj<bbk; jj++) {
                    LIBXSMM_VLA_ACCESS(4, dfilter_f32, ofm1, ifm1, ifm2*bbc+ii, ofm2*bbk+jj, nBlocksIFm, bc, bk) = 0;
                  }
                }
              }
              cfg.gemm_upd(&LIBXSMM_VLA_ACCESS(5, doutput_tr, ofm1, bfn*blocks, 0, ofm2*bbk, 0, nBlocksMB, bn_lp, bk, lpb), &LIBXSMM_VLA_ACCESS(4, input_tr, ifm1, bfn*blocks, ifm2*bbc, 0, nBlocksMB, bc, bn), &LIBXSMM_VLA_ACCESS(4, dfilter_f32, ofm1, ifm1, ifm2*bbc, ofm2*bbk, nBlocksIFm, bc, bk), &blocks);
              /* Downconvert result to BF16 and vnni format */
              if (bfn == BF-1) {
                if ((bbc % 2 == 0) && (bbk % 16 == 0)) {
                  for (jc = 0; jc < bbc; jc+=2) {
                    for (jk = 0; jk < bbk; jk+=16) {
                      a01 = LIBXSMM_INTRINSICS_MM512_LOAD_PS(&LIBXSMM_VLA_ACCESS(4, dfilter_f32, ofm1, ifm1, ifm2*bbc+jc+1, ofm2*bbk+jk, nBlocksIFm, bc, bk));
                      b01 = LIBXSMM_INTRINSICS_MM512_LOAD_PS(&LIBXSMM_VLA_ACCESS(4, dfilter_f32, ofm1, ifm1, ifm2*bbc+jc, ofm2*bbk+jk, nBlocksIFm, bc, bk));
                      c01 = LIBXSMM_INTRINSISCS_MM512_CVTNE2PS_PBH(a01, b01);
                      _mm512_storeu_si512(&LIBXSMM_VLA_ACCESS(5, dfilter, ofm1, ifm1, (ifm2*bbc+jc)/lpb, ofm2*bbk+jk, 0, nBlocksIFm, bc_lp, bk, lpb), _mm512_permutexvar_epi16(perm_index, c01));
                    }
                  }
                } else {
                  for (jc = 0; jc < bbc; jc++) {
                    LIBXSMM_DNN_CONVERT_BUFFER_F32_BF16(&LIBXSMM_VLA_ACCESS(4, dfilter_f32, ofm1, ifm1, ifm2*bbc+jc, ofm2*bbk, nBlocksIFm, bc, bk), &LIBXSMM_VLA_ACCESS(2, dfilter_block, ifm2*bbc+jc, ofm2*bbk, bk), bbk);
                  }
                  for (ii = 0; ii < bbc; ii++) {
                    for (jj = 0; jj < bbk; jj++) {
                      LIBXSMM_VLA_ACCESS(5, dfilter, ofm1, ifm1, (ifm2*bbc+ii)/lpb, ofm2*bbk+jj, (ifm2*bbc+ii)%lpb, nBlocksIFm, bc_lp, bk, lpb) = LIBXSMM_VLA_ACCESS(2, dfilter_block, ifm2*bbc+ii, ofm2*bbk+jj, bk);
                    }
                  }
                }
              }
            }
          }
        }
      }
    } else {
      if (BF == 1) {
        for ( ifm1ofm1 = thr_begin; ifm1ofm1 < thr_end; ++ifm1ofm1 ) {
          ofm1 = ifm1ofm1 / Cck_work;
          ofm2 = (ifm1ofm1 % Cck_work) / Cc_work;
          ifm1 = ((ifm1ofm1 % Cck_work) % Cc_work) / ifm_subtasks;
          ifm2 = ((ifm1ofm1 % Cck_work) % Cc_work) % ifm_subtasks;
          cfg.gemm_upd3(&LIBXSMM_VLA_ACCESS(5, doutput_tr, ofm1, 0, 0, ofm2*bbk, 0, nBlocksMB, bn_lp, bk, lpb), &LIBXSMM_VLA_ACCESS(4, input_tr, ifm1, 0, ifm2*bbc, 0, nBlocksMB, bc, bn), &LIBXSMM_VLA_ACCESS(5, dfilter, ofm1, ifm1, (ifm2*bbc)/lpb, ofm2*bbk, 0, nBlocksIFm, bc_lp, bk, lpb), &blocks);
        }
      } else {
        for (bfn = 0; bfn < BF; bfn++) {
          for ( ifm1ofm1 = thr_begin; ifm1ofm1 < thr_end; ++ifm1ofm1 ) {
            ofm1 = ifm1ofm1 / Cck_work;
            ofm2 = (ifm1ofm1 % Cck_work) / Cc_work;
            ifm1 = ((ifm1ofm1 % Cck_work) % Cc_work) / ifm_subtasks;
            ifm2 = ((ifm1ofm1 % Cck_work) % Cc_work) % ifm_subtasks;
            /* initialize current work task to zero */
            if (bfn == 0) {
              for (ii = 0; ii<bbc; ii++) {
                for (jj = 0; jj<bbk; jj++) {
                  LIBXSMM_VLA_ACCESS(4, dfilter_f32, ofm1, ifm1, ifm2*bbc+ii, ofm2*bbk+jj, nBlocksIFm, bc, bk) = 0;
                }
              }
            }
            cfg.gemm_upd(&LIBXSMM_VLA_ACCESS(5, doutput_tr, ofm1, bfn*blocks, 0, ofm2*bbk, 0, nBlocksMB, bn_lp, bk, lpb), &LIBXSMM_VLA_ACCESS(4, input_tr, ifm1, bfn*blocks, ifm2*bbc, 0, nBlocksMB, bc, bn), &LIBXSMM_VLA_ACCESS(4, dfilter_f32, ofm1, ifm1, ifm2*bbc, ofm2*bbk, nBlocksIFm, bc, bk), &blocks);
            /* Downconvert result to BF16 and vnni format */
            if (bfn == BF-1) {
              if ((bbc % 2 == 0) && (bbk % 16 == 0)) {
                for (jc = 0; jc < bbc; jc+=2) {
                  for (jk = 0; jk < bbk; jk+=16) {
                    a01 = LIBXSMM_INTRINSICS_MM512_LOAD_PS(&LIBXSMM_VLA_ACCESS(4, dfilter_f32, ofm1, ifm1, ifm2*bbc+jc+1, ofm2*bbk+jk, nBlocksIFm, bc, bk));
                    b01 = LIBXSMM_INTRINSICS_MM512_LOAD_PS(&LIBXSMM_VLA_ACCESS(4, dfilter_f32, ofm1, ifm1, ifm2*bbc+jc, ofm2*bbk+jk, nBlocksIFm, bc, bk));
                    c01 = LIBXSMM_INTRINSISCS_MM512_CVTNE2PS_PBH(a01, b01);
                    _mm512_storeu_si512(&LIBXSMM_VLA_ACCESS(5, dfilter, ofm1, ifm1, (ifm2*bbc+jc)/lpb, ofm2*bbk+jk, 0, nBlocksIFm, bc_lp, bk, lpb), _mm512_permutexvar_epi16(perm_index, c01));
                  }
                }
              } else {
                for (jc = 0; jc < bbc; jc++) {
                  LIBXSMM_DNN_CONVERT_BUFFER_F32_BF16(&LIBXSMM_VLA_ACCESS(4, dfilter_f32, ofm1, ifm1, ifm2*bbc+jc, ofm2*bbk, nBlocksIFm, bc, bk), &LIBXSMM_VLA_ACCESS(2, dfilter_block, ifm2*bbc+jc, ofm2*bbk, bk), bbk);
                }
                for (ii = 0; ii < bbc; ii++) {
                  for (jj = 0; jj < bbk; jj++) {
                    LIBXSMM_VLA_ACCESS(5, dfilter, ofm1, ifm1, (ifm2*bbc+ii)/lpb, ofm2*bbk+jj, (ifm2*bbc+ii)%lpb, nBlocksIFm, bc_lp, bk, lpb) = LIBXSMM_VLA_ACCESS(2, dfilter_block, ifm2*bbc+ii, ofm2*bbk+jj, bk);
                  }
                }
              }
            }
          }
        }
      }
    }
    libxsmm_barrier_wait(cfg.barrier, ltid);
  }
  cfg.tilerelease_kernel(NULL, NULL, NULL);
}

void my_opt_exec( my_opt_config cfg, libxsmm_bfloat16* wt_ptr, float* master_wt_ptr, const libxsmm_bfloat16* delwt_ptr, int start_tid, int my_tid, void* scratch ) {
  /* loop counters */
  libxsmm_blasint i;

  /* computing first logical thread */
  const libxsmm_blasint ltid = my_tid - start_tid;

  /* number of tasks that could run in parallel for the filters */
  const libxsmm_blasint work = cfg.C * cfg.K;
  /* compute chunk size */
  const libxsmm_blasint chunksize = (work % cfg.threads == 0) ? (work / cfg.threads) : ((work / cfg.threads) + 1);
  /* compute thr_begin and thr_end */
  const libxsmm_blasint thr_begin = (ltid * chunksize < work) ? (ltid * chunksize) : work;
  const libxsmm_blasint thr_end = ((ltid + 1) * chunksize < work) ? ((ltid + 1) * chunksize) : work;

  /* lazy barrier init */
  libxsmm_barrier_init( cfg.barrier, ltid );

#if defined(__AVX512BW__)
  libxsmm_blasint iv = ( (thr_end-thr_begin)/16 ) * 16; /* compute iterations which are vectorizable */
  __m512 vlr = _mm512_set1_ps( cfg.lr );
  for ( i = thr_begin; i < thr_begin+iv; i+=16 ) {
    __m512 newfilter = _mm512_sub_ps( _mm512_loadu_ps( master_wt_ptr+i ), _mm512_mul_ps( vlr, _mm512_load_fil( delwt_ptr + i ) ) );
    _mm512_store_fil( wt_ptr+i, newfilter );
    _mm512_storeu_ps( master_wt_ptr+i, newfilter );
  }
  for ( i = thr_begin+iv; i < thr_end; ++i ) {
    libxsmm_bfloat16_hp t1, t2;
    t1.i[0] =0;
    t1.i[1] = delwt_ptr[i];
    master_wt_ptr[i] = master_wt_ptr[i] - (cfg.lr*t1.f);
    t2.f = master_wt_ptr[i];
    wt_ptr[i] = t2.i[1];
  }
#else
  for ( i = thr_begin; i < thr_end; ++i ) {
    libxsmm_bfloat16_hp t1, t2;
    t1.i[0] =0;
    t1.i[1] = delwt_ptr[i];
    master_wt_ptr[i] = master_wt_ptr[i] - (cfg.lr*t1.f);
    t2.f = master_wt_ptr[i];
    wt_ptr[i] = t2.i[1];
  }
#endif

  libxsmm_barrier_wait( cfg.barrier, ltid );
}

void my_smax_fwd_exec( my_smax_fwd_config cfg, const libxsmm_bfloat16* in_act_ptr, libxsmm_bfloat16* out_act_ptr, const int* label_ptr, float* loss, int start_tid, int my_tid, void* scratch ) {
  libxsmm_blasint bn = cfg.bn;
  libxsmm_blasint Bn = cfg.N/cfg.bn;
  libxsmm_blasint bc = cfg.bc;
  libxsmm_blasint Bc = cfg.C/cfg.bc;

  /* loop counters */
  libxsmm_blasint i = 0;
  libxsmm_blasint img1, img2, ifm1, ifm2;

  /* computing first logical thread */
  const libxsmm_blasint ltid = my_tid - start_tid;

  /* number of tasks that could run in parallel for the batch */
  const libxsmm_blasint n_work = Bn * bn;
  /* compute chunk size */
  const libxsmm_blasint n_chunksize = (n_work % cfg.threads == 0) ? (n_work / cfg.threads) : ((n_work / cfg.threads) + 1);
  /* compute thr_begin and thr_end */
  const libxsmm_blasint n_thr_begin = (ltid * n_chunksize < n_work) ? (ltid * n_chunksize) : n_work;
  const libxsmm_blasint n_thr_end = ((ltid + 1) * n_chunksize < n_work) ? ((ltid + 1) * n_chunksize) : n_work;

  /* number of tasks that could run in parallel for the batch */
  const libxsmm_blasint nc_work = Bn * bn;
  /* compute chunk size */
  const libxsmm_blasint nc_chunksize = (nc_work % cfg.threads == 0) ? (nc_work / cfg.threads) : ((nc_work / cfg.threads) + 1);
  /* compute thr_begin and thr_end */
  const libxsmm_blasint nc_thr_begin = (ltid * nc_chunksize < nc_work) ? (ltid * nc_chunksize) : nc_work;
  const libxsmm_blasint nc_thr_end = ((ltid + 1) * nc_chunksize < nc_work) ? ((ltid + 1) * nc_chunksize) : nc_work;

  libxsmm_bfloat16* poutput_bf16 = out_act_ptr;
  const libxsmm_bfloat16* pinput_bf16  = in_act_ptr;
  float*            poutput_fp32 = (float*)scratch;
  float*            pinput_fp32  = ((float*)scratch)+(cfg.N*cfg.C);
  LIBXSMM_VLA_DECL(4,       float, output, poutput_fp32, Bc, bn, bc);
  LIBXSMM_VLA_DECL(4, const float,  input, pinput_fp32,  Bc, bn, bc);
  LIBXSMM_VLA_DECL(2,   const int,  label,   label_ptr,         bn);

  /* lazy barrier init */
  libxsmm_barrier_init( cfg.barrier, ltid );

#if defined(__AVX512BW__)
  LIBXSMM_DNN_CONVERT_BUFFER_BF16_F32(pinput_bf16+nc_thr_begin, pinput_fp32+nc_thr_begin, nc_thr_end-nc_thr_begin);
#else
  for ( i = nc_thr_begin; i < nc_thr_end; ++i ) {
    libxsmm_bfloat16_hp in;
    in.i[0] = 0;
    in.i[1] = pinput_bf16[i];
    pinput_fp32[i] = in.f;
  }
#endif

  libxsmm_barrier_wait( cfg.barrier, ltid );

  for ( i = n_thr_begin; i < n_thr_end; ++i ) {
    float max =        FLT_MIN;
    float sum_of_exp = 0.0f;

    img1 = i/bn;
    img2 = i%bn;

    /* set output to input and set compute max per image */
    for ( ifm1 = 0; ifm1 < Bc; ++ifm1 ) {
      for ( ifm2 = 0; ifm2 < bc; ++ifm2 ) {
        LIBXSMM_VLA_ACCESS( 4, output, img1, ifm1, img2, ifm2, Bc, bn, bc ) = LIBXSMM_VLA_ACCESS( 4, input, img1, ifm1, img2, ifm2, Bc, bn, bc );
        if ( LIBXSMM_VLA_ACCESS( 4, input, img1, ifm1, img2, ifm2, Bc, bn, bc ) > max ) {
          max = LIBXSMM_VLA_ACCESS( 4, input, img1, ifm1, img2, ifm2, Bc, bn, bc );
        }
      }
    }

    /* sum exp over outputs */
    for ( ifm1 = 0; ifm1 < Bc; ++ifm1 ) {
      for ( ifm2 = 0; ifm2 < bc; ++ifm2 ) {
        LIBXSMM_VLA_ACCESS( 4, output, img1, ifm1, img2, ifm2, Bc, bn, bc ) = (float)exp( (double)(LIBXSMM_VLA_ACCESS( 4, output, img1, ifm1, img2, ifm2, Bc, bn, bc ) - max) );
        sum_of_exp += LIBXSMM_VLA_ACCESS( 4, output, img1, ifm1, img2, ifm2, Bc, bn, bc );
      }
    }

    /* scale output */
    sum_of_exp = 1.0f/sum_of_exp;
    for ( ifm1 = 0; ifm1 < Bc; ++ifm1 ) {
      for ( ifm2 = 0; ifm2 < bc; ++ifm2 ) {
        LIBXSMM_VLA_ACCESS( 4, output, img1, ifm1, img2, ifm2, Bc, bn, bc ) = LIBXSMM_VLA_ACCESS( 4, output, img1, ifm1, img2, ifm2, Bc, bn, bc ) * sum_of_exp;
      }
    }
  }

  libxsmm_barrier_wait( cfg.barrier, ltid );

  /* calculate loss single threaded */
  if ( ltid == 0 ) {
    (*loss) = 0.0f;
    for ( img1 = 0; img1 < Bn; ++img1 ) {
      for ( img2 = 0; img2 <bn; ++img2 ) {
        libxsmm_blasint ifm = (libxsmm_blasint)LIBXSMM_VLA_ACCESS( 2, label, img1, img2, bn );
        libxsmm_blasint ifm1b = ifm/bc;
        libxsmm_blasint ifm2b = ifm%bc;
        float val = ( LIBXSMM_VLA_ACCESS( 4, output, img1, ifm1b, img2, ifm2b, Bc, bn, bc ) > FLT_MIN ) ? LIBXSMM_VLA_ACCESS( 4, output, img1, ifm1b, img2, ifm2b, Bc, bn, bc ) : FLT_MIN;
        *loss = LIBXSMM_LOGF( val );
      }
    }
    *loss = ((-1.0f)*(*loss))/cfg.N;
  }

  libxsmm_barrier_wait( cfg.barrier, ltid );

#if defined(__AVX512BW__)
  LIBXSMM_DNN_CONVERT_BUFFER_F32_BF16(poutput_fp32+nc_thr_begin, poutput_bf16+nc_thr_begin, nc_thr_end-nc_thr_begin);
#else
  for ( i = nc_thr_begin; i < nc_thr_end; ++i ) {
    libxsmm_bfloat16_hp in;
    in.f = poutput_fp32[i];
    poutput_bf16[i] = in.i[1];
  }
#endif
  libxsmm_barrier_wait( cfg.barrier, ltid );
}

void my_smax_bwd_exec( my_smax_bwd_config cfg, libxsmm_bfloat16* delin_act_ptr, const libxsmm_bfloat16* out_act_ptr, const int* label_ptr, int start_tid, int my_tid, void* scratch ) {
  libxsmm_blasint bn = cfg.bn;
  libxsmm_blasint Bn = cfg.N/cfg.bn;
  libxsmm_blasint bc = cfg.bc;
  libxsmm_blasint Bc = cfg.C/cfg.bc;

  /* loop counters */
  libxsmm_blasint i = 0;
  libxsmm_blasint img1, img2, ifm1, ifm2;

  float rcp_N = 1.0f/cfg.N;

  /* computing first logical thread */
  const libxsmm_blasint ltid = my_tid - start_tid;
  /* number of tasks that could run in parallel for the batch */
  const libxsmm_blasint n_work = Bn * bn;
  /* compute chunk size */
  const libxsmm_blasint n_chunksize = (n_work % cfg.threads == 0) ? (n_work / cfg.threads) : ((n_work / cfg.threads) + 1);
  /* compute thr_begin and thr_end */
  const libxsmm_blasint n_thr_begin = (ltid * n_chunksize < n_work) ? (ltid * n_chunksize) : n_work;
  const libxsmm_blasint n_thr_end = ((ltid + 1) * n_chunksize < n_work) ? ((ltid + 1) * n_chunksize) : n_work;

  /* number of tasks that could run in parallel for the batch */
  const int nc_work = Bn * bn;
  /* compute chunk size */
  const int nc_chunksize = (nc_work % cfg.threads == 0) ? (nc_work / cfg.threads) : ((nc_work / cfg.threads) + 1);
  /* compute thr_begin and thr_end */
  const int nc_thr_begin = (ltid * nc_chunksize < nc_work) ? (ltid * nc_chunksize) : nc_work;
  const int nc_thr_end = ((ltid + 1) * nc_chunksize < nc_work) ? ((ltid + 1) * nc_chunksize) : nc_work;

  const libxsmm_bfloat16* poutput_bf16 = out_act_ptr;
  libxsmm_bfloat16* pdinput_bf16 = delin_act_ptr;
  float*            poutput_fp32 = (float*)scratch;
  float*            pdinput_fp32 = ((float*)scratch)+(cfg.N*cfg.C);
  LIBXSMM_VLA_DECL(4, const float, output, poutput_fp32, Bc, bn, bc);
  LIBXSMM_VLA_DECL(4,       float, dinput, pdinput_fp32, Bc, bn, bc);
  LIBXSMM_VLA_DECL(2,   const int,  label,     label_ptr,          bn);

  /* lazy barrier init */
  libxsmm_barrier_init( cfg.barrier, ltid );

#if defined(__AVX512BW__)
  LIBXSMM_DNN_CONVERT_BUFFER_BF16_F32(poutput_bf16+nc_thr_begin, poutput_fp32+nc_thr_begin, nc_thr_end-nc_thr_begin);
#else
  for ( i = nc_thr_begin; i < nc_thr_end; ++i ) {
    libxsmm_bfloat16_hp out;
    out.i[0] = 0;
    out.i[1] = poutput_bf16[i];
    poutput_fp32[i] = out.f;
  }
#endif

  libxsmm_barrier_wait( cfg.barrier, ltid );

  for ( i = n_thr_begin; i < n_thr_end; ++i ) {
    img1 = i/bn;
    img2 = i%bn;

    /* set output to input and set compute max per image */
    for ( ifm1 = 0; ifm1 < Bc; ++ifm1 ) {
      for ( ifm2 = 0; ifm2 < bc; ++ifm2 ) {
        if ( (ifm1*Bc)+ifm2 == (libxsmm_blasint)LIBXSMM_VLA_ACCESS( 2, label, img1, img2, bn ) ) {
          LIBXSMM_VLA_ACCESS( 4, dinput, img1, ifm1, img2, ifm2, Bc, bn, bc ) =
            ( LIBXSMM_VLA_ACCESS( 4, output, img1, ifm1, img2, ifm2, Bc, bn, bc ) - 1.0f ) * rcp_N * cfg.loss_weight;
        } else {
          LIBXSMM_VLA_ACCESS( 4, dinput, img1, ifm1, img2, ifm2, Bc, bn, bc ) =
            LIBXSMM_VLA_ACCESS( 4, output, img1, ifm1, img2, ifm2, Bc, bn, bc ) * rcp_N * cfg.loss_weight;
        }
      }
    }
  }

  libxsmm_barrier_wait( cfg.barrier, ltid );

#if defined(__AVX512BW__)
  LIBXSMM_DNN_CONVERT_BUFFER_F32_BF16(pdinput_fp32+nc_thr_begin, pdinput_bf16+nc_thr_begin, nc_thr_end-nc_thr_begin);
#else
  for ( i = nc_thr_begin; i < nc_thr_end; ++i ) {
    libxsmm_bfloat16_hp in;
    in.f = pdinput_fp32[i];
    pdinput_bf16[i] = in.i[1];
  }
#endif
  libxsmm_barrier_wait( cfg.barrier, ltid );
}

int main(int argc, char* argv[])
{
  libxsmm_bfloat16 **act_libxsmm, **fil_libxsmm, **delact_libxsmm, **delfil_libxsmm;
  libxsmm_bfloat16 **bias_libxsmm, **delbias_libxsmm;
  float **fil_master;
  unsigned char **relumask_libxsmm;
  int *label_libxsmm;
  my_eltwise_fuse my_fuse;
  my_fc_fwd_config* my_fc_fwd;
  my_fc_bwd_config* my_fc_bwd;
  my_opt_config* my_opt;
  my_smax_fwd_config my_smax_fwd;
  my_smax_bwd_config my_smax_bwd;
  void* scratch = NULL;
  size_t scratch_size = 0;
#ifdef CHECK_L1
  float *last_act_fwd_f32 = NULL;
  float *first_wt_bwdupd_f32 = NULL;
#endif

  /* some parameters we can overwrite via cli,
     default is some inner layer of overfeat */
  int iters = 10;       /* repetitions of benchmark */
  int MB = 32;          /* mini-batch size, "N" */
  int fuse_type = 0;    /* 0: nothing fused, 1: relu fused, 2: elementwise fused, 3: relu and elementwise fused */
  char type = 'A';      /* 'A': ALL, 'F': FP, 'B': BP */
  int bn = 64;
  int bk = 64;
  int bc = 64;
  int *C;               /* number of input feature maps, "C" */
  int num_layers = 0;

  const char *const env_check = getenv("CHECK");
  const double check = LIBXSMM_ABS(0 == env_check ? 1 : atof(env_check));

#if defined(_OPENMP)
  int nThreads = omp_get_max_threads(); /* number of threads */
#else
  int nThreads = 1; /* number of threads */
#endif

  unsigned long long l_start, l_end;
  double l_total = 0.0;
  double gflop = 0.0;
  int i, j;
  double act_size = 0.0;
  double fil_size = 0.0;
  float lr = 0.2f;
  float loss = 0;
  float loss_weight = 0.1f;

  libxsmm_matdiff_info norms_fwd, norms_bwd, norms_upd, diff;
  libxsmm_matdiff_clear(&norms_fwd);
  libxsmm_matdiff_clear(&norms_bwd);
  libxsmm_matdiff_clear(&norms_upd);
  libxsmm_matdiff_clear(&diff);

  if (argc > 1 && !strncmp(argv[1], "-h", 3)) {
    printf("Usage: %s iters MB fuse_type type bn bk bc C1 C2 ... CN\n", argv[0]);
    return 0;
  }
  libxsmm_rng_set_seed(1);

  /* reading new values from cli */
  i = 1;
  num_layers = argc - 9;
  if (argc > i) iters      = atoi(argv[i++]);
  if (argc > i) MB         = atoi(argv[i++]);
  if (argc > i) fuse_type  = atoi(argv[i++]);
  if (argc > i) type       = *(argv[i++]);
  if (argc > i) bn         = atoi(argv[i++]);
  if (argc > i) bk         = atoi(argv[i++]);
  if (argc > i) bc         = atoi(argv[i++]);
  /* allocate the number of channles buffer */
  if ( num_layers < 1 ) {
    printf("Usage: %s iters MB fuse_type type bn bk bc C1 C2 ... CN\n", argv[0]);
    return 0;
  }
  C = (int*)malloc((num_layers+2)*sizeof(int));
  for (j = 0 ; i < argc; ++i, ++j ) {
    C[j] = atoi(argv[i]);
  }
  /* handle softmax config */
  C[num_layers+1] = C[num_layers];

  if (type != 'A' && type != 'F' && type != 'B') {
    printf("type needs to be 'A' (All), 'F' (FP only), 'B' (BP only)\n");
    return -1;
  }
  if ( (fuse_type < 0) || (fuse_type > 5) ) {
    printf("fuse type needs to be 0 (None), 1 (Bias), 2 (ReLU), 3 (Sigmoid), 4 (Bias+ReLU), 5 (Bias+Sigmoid)\n");
    return -1;
  }

#if defined(__SSE3__)
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_ROUNDING_MODE(_MM_ROUND_NEAREST);
#endif

  /* print some summary */
  printf("##########################################\n");
  printf("#          Setting Up (Common)           #\n");
  printf("##########################################\n");
  printf("PARAMS: N:%d\n", MB);
  printf("PARAMS: Layers: %d\n", num_layers);
  printf("PARAMS: ITERS:%d", iters); if (LIBXSMM_FEQ(0, check)) printf("  Threads:%d\n", nThreads); else printf("\n");
  for (i = 0; i < num_layers; ++i ) {
    if (i == 0) {
      act_size += (double)(MB*C[i]*sizeof(libxsmm_bfloat16))/(1024.0*1024.0);
      printf("SIZE Activations  %i (%dx%d): %10.2f MiB\n", i, MB, C[i], (double)(MB*C[i]*sizeof(libxsmm_bfloat16))/(1024.0*1024.0) );
    }
    act_size += (double)(MB*C[i+1]*sizeof(libxsmm_bfloat16))/(1024.0*1024.0);
    fil_size += (double)(C[i]*C[i+1]*sizeof(libxsmm_bfloat16))/(1024.0*1024.0);
    printf("SIZE Filter       %i (%dx%d): %10.2f MiB\n", i, C[i], C[i+1], (double)(C[i]*C[i+1]*sizeof(libxsmm_bfloat16))/(1024.0*1024.0) );
    printf("SIZE Activations  %i (%dx%d): %10.2f MiB\n", i+1, MB, C[i+1], (double)(MB*C[i+1]*sizeof(libxsmm_bfloat16))/(1024.0*1024.0) );
  }
  act_size += (double)(MB*C[num_layers+1]*sizeof(float))/(1024.0*1024.0);
  printf("SIZE Activations softmax (%dx%d): %10.2f MiB\n", MB, C[num_layers+1], (double)(MB*C[num_layers+1]*sizeof(libxsmm_bfloat16))/(1024.0*1024.0) );
  printf("\nTOTAL SIZE Activations:            %10.2f MiB\n", act_size );
  printf("TOTAL SIZE Filter (incl. master):  %10.2f MiB\n", 3.0*fil_size );
  printf("TOTAL SIZE delActivations:         %10.2f MiB\n", act_size );
  printf("TOTAL SIZE delFilter:              %10.2f MiB\n", fil_size );
  printf("TOTAL SIZE MLP:                    %10.2f MiB\n", (4.0*fil_size) + (2.0*act_size) );

  /* allocate data */
  act_libxsmm    = (libxsmm_bfloat16**)malloc( (num_layers+2)*sizeof(libxsmm_bfloat16*) );
  delact_libxsmm = (libxsmm_bfloat16**)malloc( (num_layers+1)*sizeof(libxsmm_bfloat16*) );
  for ( i = 0 ; i < num_layers+2; ++i ) {
    act_libxsmm[i]                = (libxsmm_bfloat16*)libxsmm_aligned_malloc( MB*C[i]*sizeof(libxsmm_bfloat16), 2097152);
    /* softmax has no incoming gradients */
    if ( i < num_layers+1 ) {
      delact_libxsmm[i]             = (libxsmm_bfloat16*)libxsmm_aligned_malloc( MB*C[i]*sizeof(libxsmm_bfloat16), 2097152);
    }
  }
  fil_master     = (float**)           malloc( num_layers*sizeof(float*) );
  fil_libxsmm    = (libxsmm_bfloat16**)malloc( num_layers*sizeof(libxsmm_bfloat16*) );
  delfil_libxsmm = (libxsmm_bfloat16**)malloc( num_layers*sizeof(libxsmm_bfloat16*) );
  for ( i = 0 ; i < num_layers; ++i ) {
    fil_master[i]                 = (float*)           libxsmm_aligned_malloc( C[i]*C[i+1]*sizeof(float), 2097152);
    fil_libxsmm[i]                = (libxsmm_bfloat16*)libxsmm_aligned_malloc( C[i]*C[i+1]*sizeof(libxsmm_bfloat16), 2097152);
    delfil_libxsmm[i]             = (libxsmm_bfloat16*)libxsmm_aligned_malloc( C[i]*C[i+1]*sizeof(libxsmm_bfloat16), 2097152);
  }
  bias_libxsmm    = (libxsmm_bfloat16**)malloc( num_layers*sizeof(libxsmm_bfloat16*) );
  delbias_libxsmm = (libxsmm_bfloat16**)malloc( num_layers*sizeof(libxsmm_bfloat16*) );
  for ( i = 0 ; i < num_layers; ++i ) {
    bias_libxsmm[i]               = (libxsmm_bfloat16*)libxsmm_aligned_malloc( C[i+1]*sizeof(libxsmm_bfloat16), 2097152);
    delbias_libxsmm[i]            = (libxsmm_bfloat16*)libxsmm_aligned_malloc( C[i+1]*sizeof(libxsmm_bfloat16), 2097152);
  }
  relumask_libxsmm = (unsigned char**)malloc( num_layers*sizeof(unsigned char*) );
  for ( i = 0 ; i < num_layers; ++i ) {
    relumask_libxsmm[i]           = (unsigned char*)libxsmm_aligned_malloc( MB*C[i+1]*sizeof(unsigned char), 2097152);
  }
  label_libxsmm = (int*)libxsmm_aligned_malloc( MB*sizeof(int), 2097152);

  /* init data */
  for ( i = 0 ; i < num_layers+2; ++i ) {
    my_init_buf_bf16( act_libxsmm[i], MB*C[i], 0, 0 );
  }
  for ( i = 0 ; i < num_layers+1; ++i ) {
    my_init_buf_bf16( delact_libxsmm[i], MB*C[i], 0, 0 );
  }
  for ( i = 0 ; i < num_layers; ++i ) {
    my_init_buf( fil_master[i], C[i]*C[i+1], 0, 0 );
    libxsmm_rne_convert_fp32_bf16( fil_master[i], fil_libxsmm[i], C[i]*C[i+1] );
  }
  for ( i = 0 ; i < num_layers; ++i ) {
    my_init_buf_bf16( delfil_libxsmm[i], C[i]*C[i+1], 0, 0 );
  }
  for ( i = 0 ; i < num_layers; ++i ) {
    my_init_buf_bf16( bias_libxsmm[i], C[i+1], 0, 0 );
  }
  for ( i = 0 ; i < num_layers; ++i ) {
    my_init_buf_bf16( delbias_libxsmm[i], C[i+1], 0, 0 );
  }
  for ( i = 0 ; i < num_layers; ++i ) {
    zero_buf_uint8( relumask_libxsmm[i], MB*C[i+1] );
  }
  zero_buf_int32( label_libxsmm, MB );

  printf("\n");
  printf("##########################################\n");
  printf("#      Setting Up  (custom-Storage)      #\n");
  printf("##########################################\n");

  if ( fuse_type == 0 ) {
    my_fuse = MY_ELTWISE_FUSE_NONE;
  } else if ( fuse_type == 1 ) {
    my_fuse = MY_ELTWISE_FUSE_BIAS;
  } else if ( fuse_type == 2 ) {
    my_fuse = MY_ELTWISE_FUSE_RELU;
  } else if ( fuse_type == 4 ) {
    my_fuse = MY_ELTWISE_FUSE_BIAS_RELU;
  } else {
    /* cannot happen */
  }

  /* allocating handles */
  my_fc_fwd = (my_fc_fwd_config*) malloc( num_layers*sizeof(my_fc_fwd_config) );
  my_fc_bwd = (my_fc_bwd_config*) malloc( num_layers*sizeof(my_fc_bwd_config) );
  my_opt    = (my_opt_config*)    malloc( num_layers*sizeof(my_opt_config)    );

  /* setting up handles + scratch */
  for ( i = 0; i < num_layers; ++i ) {
    my_fc_fwd[i] = setup_my_fc_fwd(MB, C[i], C[i+1], (MB % bn == 0) ? bn : MB,
                                             (C[i  ] % bc == 0) ? bc : C[i  ],
                                             (C[i+1] % bk == 0) ? bk : C[i+1],
                                             nThreads, my_fuse);

    my_fc_bwd[i] = setup_my_fc_bwd(MB, C[i], C[i+1], (MB % bn == 0) ? bn : MB,
                                             (C[i  ] % bc == 0) ? bc : C[i  ],
                                             (C[i+1] % bk == 0) ? bk : C[i+1],
                                              nThreads, my_fuse);

    my_opt[i] = setup_my_opt( C[i], C[i+1], (C[i  ] % bc == 0) ? bc : C[i  ],
                                            (C[i+1] % bk == 0) ? bk : C[i+1],
                                            nThreads, lr );

    /* let's allocate and bind scratch */
    if ( my_fc_fwd[i].scratch_size > 0 || my_fc_bwd[i].scratch_size > 0 || my_opt[i].scratch_size > 0 ) {
      size_t alloc_size = LIBXSMM_MAX( LIBXSMM_MAX( my_fc_fwd[i].scratch_size, my_fc_bwd[i].scratch_size), my_opt[i].scratch_size );
      if ( alloc_size > scratch_size ) {
        if ( scratch != NULL ) libxsmm_free( scratch );
        scratch_size = alloc_size;
        scratch = libxsmm_aligned_scratch( scratch_size, 2097152 );
        my_init_buf( (float*)(scratch), (scratch_size)/4, 0, 0 );
      }
    }
  }

  /* softmax+loss is treated as N+! layer */
  my_smax_fwd = setup_my_smax_fwd( MB, C[num_layers+1], (MB % bn == 0) ? bn : MB,
                                       (C[num_layers+1] % bk == 0) ? bk : C[num_layers+1],
                                       nThreads );

  my_smax_bwd = setup_my_smax_bwd( MB, C[num_layers+1], (MB % bn == 0) ? bn : MB,
                                       (C[num_layers+1] % bk == 0) ? bk : C[num_layers+1],
                                       nThreads, loss_weight );

  if ( my_smax_fwd.scratch_size > 0 || my_smax_bwd.scratch_size > 0 ) {
    size_t alloc_size = LIBXSMM_MAX( my_smax_fwd.scratch_size, my_smax_bwd.scratch_size );
    if ( alloc_size > scratch_size ) {
      if ( scratch != NULL ) libxsmm_free( scratch );
      scratch_size = alloc_size;
      scratch = libxsmm_aligned_scratch( scratch_size, 2097152 );
      my_init_buf( (float*)(scratch), (scratch_size)/4, 0, 0 );
    }
  }

  if ( type == 'F') {
    printf("##########################################\n");
    printf("#   Performance - FWD (custom-Storage)   #\n");
    printf("##########################################\n");
    l_start = libxsmm_timer_tick();
#if defined(_OPENMP)
#   pragma omp parallel private(i,j)
#endif
    {
#if defined(_OPENMP)
      const int tid = omp_get_thread_num();
#else
      const int tid = 0;
#endif
      for (j = 0; j < iters; ++j) {
        for ( i = 0; i < num_layers; ++i) {
          my_fc_fwd_exec( my_fc_fwd[i], fil_libxsmm[i], act_libxsmm[i], act_libxsmm[i+1],
                          bias_libxsmm[i], relumask_libxsmm[i], 0, tid, scratch );
        }
#ifdef USE_SOFTMAX
        my_smax_fwd_exec( my_smax_fwd, act_libxsmm[num_layers], act_libxsmm[num_layers+1], label_libxsmm, &loss,
                          0, tid, scratch );
#endif
      }
    }
    l_end = libxsmm_timer_tick();
    l_total = libxsmm_timer_duration(l_start, l_end);

    gflop = 0.0;
    for ( i = 0; i < num_layers; ++i) {
      gflop += (2.0*(double)MB*(double)C[i]*(double)C[i+1]*(double)iters) / (1000.0*1000.0*1000.0);
    }
    printf("GFLOP  = %.5g\n", gflop/(double)iters);
    printf("fp time = %.5g\n", ((double)(l_total/iters)));
    printf("GFLOPS  = %.5g\n", gflop/l_total);
    printf("PERFDUMP,FP,%s,%i,%i,", LIBXSMM_VERSION, nThreads, MB );
    for ( i = 0; i < num_layers; ++i ) {
      printf("%i,", C[i] );
    }
    printf("%f,%f\n", ((double)(l_total/iters)), gflop/l_total);
  }

  if (type == 'B') {
    printf("##########################################\n");
    printf("#   Performance - BWD (custom-Storage)   #\n");
    printf("##########################################\n");
    l_start = libxsmm_timer_tick();
#if defined(_OPENMP)
#   pragma omp parallel private(i,j)
#endif
    {
#if defined(_OPENMP)
      const int tid = omp_get_thread_num();
#else
      const int tid = 0;
#endif
      for (j = 0; j < iters; ++j) {
#ifdef USE_SOFTMAX
        my_smax_bwd_exec( my_smax_bwd, delact_libxsmm[num_layers], act_libxsmm[num_layers+1], label_libxsmm,
                          0, tid, scratch );
#endif
        for ( i = num_layers-1; i > 0; --i) {
          my_fc_bwd_exec( my_fc_bwd[i], fil_libxsmm[i], delact_libxsmm[i], delact_libxsmm[i+1], delfil_libxsmm[i],
                          act_libxsmm[i], delbias_libxsmm[i], relumask_libxsmm[i], MY_PASS_BWD, 0, tid, scratch );
          my_opt_exec( my_opt[i], fil_libxsmm[i], fil_master[i], delfil_libxsmm[i], 0, tid, scratch );
        }
        my_fc_bwd_exec( my_fc_bwd[0], fil_libxsmm[0], delact_libxsmm[0], delact_libxsmm[0+1], delfil_libxsmm[0],
                        act_libxsmm[0], delbias_libxsmm[0], relumask_libxsmm[0], MY_PASS_BWD_W, 0, tid, scratch );
        my_opt_exec( my_opt[0], fil_libxsmm[0], fil_master[0], delfil_libxsmm[0], 0, tid, scratch );
      }
    }
    l_end = libxsmm_timer_tick();
    l_total = libxsmm_timer_duration(l_start, l_end);

    gflop = 0.0;
    for ( i = num_layers-1; i > 0; --i) {
      gflop += (4.0*(double)MB*(double)C[i]*(double)C[i+1]*(double)iters) / (1000.0*1000.0*1000.0);
    }
    gflop += (2.0*(double)MB*(double)C[0]*(double)C[1]*(double)iters) / (1000.0*1000.0*1000.0);
    printf("GFLOP  = %.5g\n", gflop/(double)iters);
    printf("fp time = %.5g\n", ((double)(l_total/iters)));
    printf("GFLOPS  = %.5g\n", gflop/l_total);
    printf("PERFDUMP,BP,%s,%i,%i,", LIBXSMM_VERSION, nThreads, MB );
    for ( i = 0; i < num_layers; ++i ) {
      printf("%i,", C[i] );
    }
    printf("%f,%f\n", ((double)(l_total/iters)), gflop/l_total);
  }

  if (type == 'A') {
    printf("##########################################\n");
    printf("# Performance - FWD-BWD (custom-Storage) #\n");
    printf("##########################################\n");
    l_start = libxsmm_timer_tick();
#if defined(_OPENMP)
#   pragma omp parallel private(i,j)
#endif
    {
#if defined(_OPENMP)
      const int tid = omp_get_thread_num();
#else
      const int tid = 0;
#endif
      for (j = 0; j < iters; ++j) {
        for ( i = 0; i < num_layers; ++i) {
          my_fc_fwd_exec( my_fc_fwd[i], fil_libxsmm[i], act_libxsmm[i], act_libxsmm[i+1],
                          bias_libxsmm[i], relumask_libxsmm[i], 0, tid, scratch );
        }
#ifdef USE_SOFTMAX
        my_smax_fwd_exec( my_smax_fwd, act_libxsmm[num_layers], act_libxsmm[num_layers+1], label_libxsmm, &loss,
                          0, tid, scratch );
        my_smax_bwd_exec( my_smax_bwd, delact_libxsmm[num_layers], act_libxsmm[num_layers+1], label_libxsmm,
                          0, tid, scratch );
#endif
        for ( i = num_layers-1; i > 0; --i) {
          my_fc_bwd_exec( my_fc_bwd[i], fil_libxsmm[i], delact_libxsmm[i], delact_libxsmm[i+1], delfil_libxsmm[i],
                          act_libxsmm[i], delbias_libxsmm[i], relumask_libxsmm[i], MY_PASS_BWD, 0, tid, scratch );
          my_opt_exec( my_opt[i], fil_libxsmm[i], fil_master[i], delfil_libxsmm[i], 0, tid, scratch );
        }
        my_fc_bwd_exec( my_fc_bwd[0], fil_libxsmm[0], delact_libxsmm[0], delact_libxsmm[0+1], delfil_libxsmm[0],
                        act_libxsmm[0], delbias_libxsmm[0], relumask_libxsmm[0], MY_PASS_BWD_W, 0, tid, scratch );
        my_opt_exec( my_opt[0], fil_libxsmm[0], fil_master[0], delfil_libxsmm[0], 0, tid, scratch );
      }
    }
    l_end = libxsmm_timer_tick();
    l_total = libxsmm_timer_duration(l_start, l_end);

#ifdef CHECK_L1
    /* Print some norms on last act for fwd and weights of first layer after all iterations */
    last_act_fwd_f32    = (float*) malloc(MB*C[num_layers]*sizeof(float));
    first_wt_bwdupd_f32 = (float*) malloc(C[0]*C[1]*sizeof(float));
    matrix_copy_bf16_f32(MB*C[num_layers], act_libxsmm[num_layers], last_act_fwd_f32);
    matrix_copy_bf16_f32(C[0]*C[1], fil_libxsmm[0], first_wt_bwdupd_f32);

    libxsmm_matdiff(&norms_fwd, LIBXSMM_DATATYPE_F32, MB*C[num_layers], 1, last_act_fwd_f32, last_act_fwd_f32, 0, 0);
    printf("L1 of act[num_layers]  : %.25g\n", norms_fwd.l1_ref);
    libxsmm_matdiff_reduce(&diff, &norms_fwd);
    libxsmm_matdiff(&norms_bwd, LIBXSMM_DATATYPE_F32, C[0]*C[1], 1, first_wt_bwdupd_f32, first_wt_bwdupd_f32, 0, 0);
    printf("L1 of wt[0]  : %.25g\n", norms_bwd.l1_ref);
    libxsmm_matdiff_reduce(&diff, &norms_bwd);

    free(first_wt_bwdupd_f32);
    free(last_act_fwd_f32);
#endif

    gflop = 0.0;
    for ( i = num_layers-1; i > 0; --i) {
      gflop += (6.0*(double)MB*(double)C[i]*(double)C[i+1]*(double)iters) / (1000.0*1000.0*1000.0);
    }
    gflop += (4.0*(double)MB*(double)C[0]*(double)C[1]*(double)iters) / (1000.0*1000.0*1000.0);
    printf("GFLOP  = %.5g\n", gflop/(double)iters);
    printf("fp time = %.5g\n", ((double)(l_total/iters)));
    printf("GFLOPS  = %.5g\n", gflop/l_total);
    printf("PERFDUMP,BP,%s,%i,%i,", LIBXSMM_VERSION, nThreads, MB );
    for ( i = 0; i < num_layers; ++i ) {
      printf("%i,", C[i] );
    }
    printf("%f,%f\n", ((double)(l_total/iters)), gflop/l_total);
  }

  /* deallocate data */
  if ( scratch != NULL ) {
    libxsmm_free(scratch);
  }

  for ( i = 0; i < num_layers; ++i ) {
    if ( i == 0 ) {
      libxsmm_free(act_libxsmm[i]);
      libxsmm_free(delact_libxsmm[i]);
    }
    libxsmm_free(act_libxsmm[i+1]);
    libxsmm_free(delact_libxsmm[i+1]);

    libxsmm_free(fil_libxsmm[i]);
    libxsmm_free(delfil_libxsmm[i]);
    libxsmm_free(bias_libxsmm[i]);
    libxsmm_free(delbias_libxsmm[i]);
    libxsmm_free(relumask_libxsmm[i]);
    libxsmm_free(fil_master[i]);
  }
  libxsmm_free(act_libxsmm[num_layers+1]);
  libxsmm_free(label_libxsmm);

  free( my_opt );
  free( my_fc_fwd );
  free( my_fc_bwd );

  free( act_libxsmm );
  free( delact_libxsmm );
  free( fil_master );
  free( fil_libxsmm );
  free( delfil_libxsmm );
  free( bias_libxsmm );
  free( delbias_libxsmm );
  free( relumask_libxsmm );

  free( C );

  /* some empty lines at the end */
  printf("\n\n\n");

  return 0;
}

