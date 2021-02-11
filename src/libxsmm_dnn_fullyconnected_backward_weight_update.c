/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Heinecke, Evangelos Georganas (Intel Corp.)
******************************************************************************/
#include "libxsmm_dnn_fullyconnected_backward_weight_update.h"
#include "libxsmm_main.h"

LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_fullyconnected_st_bwdupd_custom_f32_f32(libxsmm_dnn_fullyconnected* handle, libxsmm_dnn_compute_kind kind, int start_thread, int tid);
LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_fullyconnected_st_bwdupd_ncnc_kcck_f32_f32(libxsmm_dnn_fullyconnected* handle, libxsmm_dnn_compute_kind kind, int start_thread, int tid);
LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_fullyconnected_st_bwdupd_custom_bf16_f32(libxsmm_dnn_fullyconnected* handle, libxsmm_dnn_compute_kind kind, int start_thread, int tid);
LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_fullyconnected_st_bwdupd_ncnc_kcck_bf16_bf16(libxsmm_dnn_fullyconnected* handle, libxsmm_dnn_compute_kind kind, int start_thread, int tid);
LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_fullyconnected_st_bwdupd_ncnc_kcck_bf16_bf16_emu(libxsmm_dnn_fullyconnected* handle, libxsmm_dnn_compute_kind kind, int start_thread, int tid);
LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_fullyconnected_st_bwdupd_ncnc_kcck_bf16_bf16_amx(libxsmm_dnn_fullyconnected* handle, libxsmm_dnn_compute_kind kind, int start_thread, int tid);
LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_fullyconnected_st_bwdupd_ncnc_kcck_bf16_bf16_amx_emu(libxsmm_dnn_fullyconnected* handle, libxsmm_dnn_compute_kind kind, int start_thread, int tid);

#if 0
#define USE_CLDEMOTE
#endif

LIBXSMM_API_INLINE LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512_CORE)
void bf16_vnni_transpose_16x16(void* source_void, void* dest_void, int source_stride, int dest_stride)
{
#if defined(LIBXSMM_INTRINSICS_AVX512_CORE)
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
#ifdef USE_CLDEMOTE
  _mm_cldemote(dest);
  _mm_cldemote(dest + dest_stride);
  _mm_cldemote(dest + dest_stride * 2);
  _mm_cldemote(dest + dest_stride * 3);
  _mm_cldemote(dest + dest_stride * 4);
  _mm_cldemote(dest + dest_stride * 5);
  _mm_cldemote(dest + dest_stride * 6);
  _mm_cldemote(dest + dest_stride * 7);
#endif
#else
  LIBXSMM_UNUSED(source_void); LIBXSMM_UNUSED(dest_void); LIBXSMM_UNUSED(source_stride); LIBXSMM_UNUSED(dest_stride);
#endif
}

LIBXSMM_API_INLINE LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512_CORE)
void bf16_vnni_transpose(libxsmm_bfloat16* src, libxsmm_bfloat16* dst, int M, int N, int ld_in, int ld_out)
{
#if defined(LIBXSMM_INTRINSICS_AVX512_CORE)
  const int _M = M/16, _N = N/16;
  int i = 0, j = 0;
  for (i = 0; i < _N; i++) {
    for (j = 0; j < _M; j++) {
      bf16_vnni_transpose_16x16((libxsmm_bfloat16*) src+i*16*ld_in+j*32, (libxsmm_bfloat16*) dst+j*16*ld_out+i*32, ld_in*2, ld_out*2);
    }
  }
#else
  LIBXSMM_UNUSED(src); LIBXSMM_UNUSED(dst); LIBXSMM_UNUSED(M); LIBXSMM_UNUSED(N); LIBXSMM_UNUSED(ld_in); LIBXSMM_UNUSED(ld_out);
#endif
}

LIBXSMM_API_INLINE LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512_CORE)
void bf16_transpose_32x16(libxsmm_bfloat16 *in, libxsmm_bfloat16 *out, int ld_in, int ld_out)
{
#if defined(LIBXSMM_INTRINSICS_AVX512_CORE)
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
#ifdef USE_CLDEMOTE
  _mm_cldemote(out + 0*out_width);
  _mm_cldemote(out + 1*out_width);
  _mm_cldemote(out + 2*out_width);
  _mm_cldemote(out + 3*out_width);
  _mm_cldemote(out + 4*out_width);
  _mm_cldemote(out + 5*out_width);
  _mm_cldemote(out + 6*out_width);
  _mm_cldemote(out + 7*out_width);
  _mm_cldemote(out + 8*out_width);
  _mm_cldemote(out + 9*out_width);
  _mm_cldemote(out + 10*out_width);
  _mm_cldemote(out + 11*out_width);
  _mm_cldemote(out + 12*out_width);
  _mm_cldemote(out + 13*out_width);
  _mm_cldemote(out + 14*out_width);
  _mm_cldemote(out + 15*out_width);
  _mm_cldemote(out + 16*out_width);
  _mm_cldemote(out + 17*out_width);
  _mm_cldemote(out + 18*out_width);
  _mm_cldemote(out + 19*out_width);
  _mm_cldemote(out + 20*out_width);
  _mm_cldemote(out + 21*out_width);
  _mm_cldemote(out + 22*out_width);
  _mm_cldemote(out + 23*out_width);
  _mm_cldemote(out + 24*out_width);
  _mm_cldemote(out + 25*out_width);
  _mm_cldemote(out + 26*out_width);
  _mm_cldemote(out + 27*out_width);
  _mm_cldemote(out + 28*out_width);
  _mm_cldemote(out + 29*out_width);
  _mm_cldemote(out + 30*out_width);
  _mm_cldemote(out + 31*out_width);
#endif
#else
 LIBXSMM_UNUSED(in); LIBXSMM_UNUSED(out); LIBXSMM_UNUSED(ld_in); LIBXSMM_UNUSED(ld_out);
#endif
}

LIBXSMM_API_INLINE LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512_CORE)
void bf16_transpose_32xcols(libxsmm_bfloat16 *in, libxsmm_bfloat16 *out, int col, int ld_in, int ld_out)
{
#if defined(LIBXSMM_INTRINSICS_AVX512_CORE)
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
#ifdef USE_CLDEMOTE
  _mm_cldemote(out + 0*out_width);
  _mm_cldemote(out + 1*out_width);
  _mm_cldemote(out + 2*out_width);
  _mm_cldemote(out + 3*out_width);
  _mm_cldemote(out + 4*out_width);
  _mm_cldemote(out + 5*out_width);
  _mm_cldemote(out + 6*out_width);
  _mm_cldemote(out + 7*out_width);
  _mm_cldemote(out + 8*out_width);
  _mm_cldemote(out + 9*out_width);
  _mm_cldemote(out + 10*out_width);
  _mm_cldemote(out + 11*out_width);
  _mm_cldemote(out + 12*out_width);
  _mm_cldemote(out + 13*out_width);
  _mm_cldemote(out + 14*out_width);
  _mm_cldemote(out + 15*out_width);
  _mm_cldemote(out + 16*out_width);
  _mm_cldemote(out + 17*out_width);
  _mm_cldemote(out + 18*out_width);
  _mm_cldemote(out + 19*out_width);
  _mm_cldemote(out + 20*out_width);
  _mm_cldemote(out + 21*out_width);
  _mm_cldemote(out + 22*out_width);
  _mm_cldemote(out + 23*out_width);
  _mm_cldemote(out + 24*out_width);
  _mm_cldemote(out + 25*out_width);
  _mm_cldemote(out + 26*out_width);
  _mm_cldemote(out + 27*out_width);
  _mm_cldemote(out + 28*out_width);
  _mm_cldemote(out + 29*out_width);
  _mm_cldemote(out + 30*out_width);
  _mm_cldemote(out + 31*out_width);
#endif
#else
 LIBXSMM_UNUSED(in); LIBXSMM_UNUSED(out); LIBXSMM_UNUSED(ld_in); LIBXSMM_UNUSED(ld_out); LIBXSMM_UNUSED(col);
#endif
}

LIBXSMM_API_INLINE LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512_CORE)
void bf16_transpose(libxsmm_bfloat16 *in, libxsmm_bfloat16 *out, int M, int N, int ld_in, int ld_out){
#if defined(LIBXSMM_INTRINSICS_AVX512_CORE)
  int i, j;
  int full16_chunks = N/16;
  int remainder_cols = N%16;
  int _N = N - remainder_cols;

  if (full16_chunks) {
    for (i=0; i<M; i+=32) {
      for (j=0; j<_N; j+=16) {
        bf16_transpose_32x16((libxsmm_bfloat16*)in + i + ld_in*j, (libxsmm_bfloat16*)out + j + i*ld_out, ld_in, ld_out);
      }
    }
  }

  if (remainder_cols) {
    for (i=0; i<M; i+=32) {
      bf16_transpose_32xcols((libxsmm_bfloat16*)in + i + ld_in*full16_chunks*16, (libxsmm_bfloat16*)out + full16_chunks*16 + i*ld_out, remainder_cols, ld_in, ld_out);
    }
  }
#else
 LIBXSMM_UNUSED(in); LIBXSMM_UNUSED(out); LIBXSMM_UNUSED(M); LIBXSMM_UNUSED(N); LIBXSMM_UNUSED(ld_in); LIBXSMM_UNUSED(ld_out);
#endif
}

LIBXSMM_API_INLINE LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512_CORE)
void bf16_vnni_reformat(libxsmm_bfloat16 *_in, libxsmm_bfloat16 *_out, int M, int N, int ld_in, int ld_out) {
#if defined(LIBXSMM_INTRINSICS_AVX512_CORE)
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
#ifdef USE_CLDEMOTE
      _mm_cldemote((libxsmm_bfloat16*)out+m*2);
      _mm_cldemote((libxsmm_bfloat16*)out+m*2+32);
#endif
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
#ifdef USE_CLDEMOTE
      _mm_cldemote((libxsmm_bfloat16*)out+m*2);
      _mm_cldemote((libxsmm_bfloat16*)out+m*2+32);
#endif
    }
  }
#else
 LIBXSMM_UNUSED(_in); LIBXSMM_UNUSED(_out); LIBXSMM_UNUSED(M); LIBXSMM_UNUSED(N); LIBXSMM_UNUSED(ld_in); LIBXSMM_UNUSED(ld_out);
#endif
}

#undef USE_CLDEMOTE

LIBXSMM_API_INTERN LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512)
libxsmm_dnn_err_t libxsmm_dnn_fullyconnected_st_bwdupd_custom_f32_f32(libxsmm_dnn_fullyconnected* handle, libxsmm_dnn_compute_kind kind, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
#if defined(LIBXSMM_INTRINSICS_AVX512) /*__AVX512F__*/
  typedef float element_input_type;
  typedef float element_output_type;
  typedef float element_filter_type;
  libxsmm_blasint lda_bwd = (libxsmm_blasint)handle->ifmblock;
  libxsmm_blasint ldb_bwd = (libxsmm_blasint)handle->desc.K;
  libxsmm_blasint ldc_bwd = (libxsmm_blasint)handle->desc.C;
  libxsmm_blasint lda_upd = (libxsmm_blasint)handle->desc.K;
  libxsmm_blasint ldb_upd = (libxsmm_blasint)handle->desc.N;
  libxsmm_blasint ldc_upd = (libxsmm_blasint)handle->ofmblock;
  element_input_type alpha = (element_input_type)1;
  element_input_type beta = (element_input_type)0;

  if ( handle->desc.fuse_ops == LIBXSMM_DNN_FULLYCONNECTED_FUSE_NONE ) {
    typedef libxsmm_smmfunction gemm_function;
    gemm_function gemm_kernel_bwd = libxsmm_smmdispatch(handle->ifmblock, handle->desc.N, handle->desc.K, &lda_bwd, &ldb_bwd, &ldc_bwd, &alpha, &beta, NULL, NULL);
    gemm_function gemm_kernel_upd = libxsmm_smmdispatch(handle->ofmblock, handle->ifmblock, handle->desc.N, &lda_upd, &ldb_upd, &ldc_upd, &alpha, &beta, NULL, NULL);
# include "template/libxsmm_dnn_fullyconnected_st_bwdupd_custom_generic.tpl.c"
  } else {
    status = LIBXSMM_DNN_ERR_FC_UNSUPPORTED_FUSION;
  }
#else /* should not happen */
  LIBXSMM_UNUSED(handle); LIBXSMM_UNUSED(kind); LIBXSMM_UNUSED(start_thread); LIBXSMM_UNUSED(tid);
  status = LIBXSMM_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}

LIBXSMM_API_INTERN LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512)
libxsmm_dnn_err_t libxsmm_dnn_fullyconnected_st_bwdupd_custom_bf16_f32(libxsmm_dnn_fullyconnected* handle, libxsmm_dnn_compute_kind kind, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
#if defined(LIBXSMM_INTRINSICS_AVX512) /*__AVX512F__*/
  typedef libxsmm_bfloat16 element_input_type;
  typedef float element_output_type;
  typedef libxsmm_bfloat16 element_filter_type;
  typedef libxsmm_smmfunction gemm_function;
  libxsmm_blasint lda_bwd = (libxsmm_blasint)handle->ifmblock;
  libxsmm_blasint ldb_bwd = (libxsmm_blasint)handle->desc.K;
  libxsmm_blasint ldc_bwd = (libxsmm_blasint)handle->desc.C;
  libxsmm_blasint lda_upd = (libxsmm_blasint)handle->desc.K;
  libxsmm_blasint ldb_upd = (libxsmm_blasint)handle->desc.N;
  libxsmm_blasint ldc_upd = (libxsmm_blasint)handle->ofmblock;
  float alpha = (element_input_type)1;
  float beta = (element_input_type)0;

  if ( handle->desc.fuse_ops == LIBXSMM_DNN_FULLYCONNECTED_FUSE_NONE ) {
    gemm_function gemm_kernel_bwd = libxsmm_smmdispatch(handle->ifmblock, handle->desc.N, handle->desc.K, &lda_bwd, &ldb_bwd, &ldc_bwd, &alpha, &beta, NULL, NULL);
    gemm_function gemm_kernel_upd = libxsmm_smmdispatch(handle->ofmblock, handle->ifmblock, handle->desc.N, &lda_upd, &ldb_upd, &ldc_upd, &alpha, &beta, NULL, NULL);
# define LIBXSMM_DNN_FULLYCONNECTED_BWD_BF16_F32
# define LIBXSMM_DNN_FULLYCONNECTED_UPD_BF16_F32
# include "template/libxsmm_dnn_fullyconnected_st_bwdupd_custom_generic.tpl.c"
# undef LIBXSMM_DNN_FULLYCONNECTED_UPD_BF16_F32
# undef LIBXSMM_DNN_FULLYCONNECTED_BWD_BF16_F32
  } else {
    status = LIBXSMM_DNN_ERR_FC_UNSUPPORTED_FUSION;
  }
#else /* should not happen */
  LIBXSMM_UNUSED(handle); LIBXSMM_UNUSED(kind); LIBXSMM_UNUSED(start_thread); LIBXSMM_UNUSED(tid);
  status = LIBXSMM_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}


LIBXSMM_API_INTERN LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512)
libxsmm_dnn_err_t libxsmm_dnn_fullyconnected_st_bwdupd_ncnc_kcck_f32_f32(libxsmm_dnn_fullyconnected* handle, libxsmm_dnn_compute_kind kind, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
#if defined(LIBXSMM_INTRINSICS_AVX512) /*__AVX512F__*/
  typedef float element_input_type;
  typedef float element_output_type;
  typedef float element_filter_type;
  libxsmm_smmfunction_reducebatch_strd batchreduce_kernel_bwd = handle->gemm_bwd.xgemm.smrs;
  libxsmm_smmfunction_reducebatch_strd batchreduce_kernel_bwd_zerobeta = handle->gemm_bwd2.xgemm.smrs;
  libxsmm_smmfunction_reducebatch_strd batchreduce_kernel_upd = handle->gemm_upd.xgemm.smrs;
  libxsmm_smmfunction_reducebatch_strd batchreduce_kernel_upd_zerobeta = handle->gemm_upd2.xgemm.smrs;

#define LIBXSMM_DNN_FC_BWD_USE_AVX512
  if ( handle->desc.fuse_ops == LIBXSMM_DNN_FULLYCONNECTED_FUSE_NONE ) {
# include "template/libxsmm_dnn_fullyconnected_st_bwdupd_ncnc_kcck_generic.tpl.c"
  } else if ( handle->desc.fuse_ops == LIBXSMM_DNN_FULLYCONNECTED_FUSE_BIAS ) {
#define LIBXSMM_DNN_FC_BWD_FUSE_BIAS
# include "template/libxsmm_dnn_fullyconnected_st_bwdupd_ncnc_kcck_generic.tpl.c"
#undef LIBXSMM_DNN_FC_BWD_FUSE_BIAS
  } else if ( handle->desc.fuse_ops == LIBXSMM_DNN_FULLYCONNECTED_FUSE_RELU ) {
#define LIBXSMM_DNN_FC_BWD_FUSE_RELU
# include "template/libxsmm_dnn_fullyconnected_st_bwdupd_ncnc_kcck_generic.tpl.c"
#undef LIBXSMM_DNN_FC_BWD_FUSE_RELU
  } else if ( handle->desc.fuse_ops == LIBXSMM_DNN_FULLYCONNECTED_FUSE_SIGMOID ) {
#define LIBXSMM_DNN_FC_BWD_FUSE_SIGMOID
# include "template/libxsmm_dnn_fullyconnected_st_bwdupd_ncnc_kcck_generic.tpl.c"
#undef LIBXSMM_DNN_FC_BWD_FUSE_SIGMOID
  } else if ( handle->desc.fuse_ops == LIBXSMM_DNN_FULLYCONNECTED_FUSE_BIAS_RELU ) {
#define LIBXSMM_DNN_FC_BWD_FUSE_BIAS
#define LIBXSMM_DNN_FC_BWD_FUSE_RELU
# include "template/libxsmm_dnn_fullyconnected_st_bwdupd_ncnc_kcck_generic.tpl.c"
#undef LIBXSMM_DNN_FC_BWD_FUSE_RELU
#undef LIBXSMM_DNN_FC_BWD_FUSE_BIAS
  } else if ( handle->desc.fuse_ops == LIBXSMM_DNN_FULLYCONNECTED_FUSE_BIAS_SIGMOID ) {
#define LIBXSMM_DNN_FC_BWD_FUSE_BIAS
#define LIBXSMM_DNN_FC_BWD_FUSE_SIGMOID
# include "template/libxsmm_dnn_fullyconnected_st_bwdupd_ncnc_kcck_generic.tpl.c"
#undef LIBXSMM_DNN_FC_BWD_FUSE_SIGMOID
#undef LIBXSMM_DNN_FC_BWD_FUSE_BIAS
  } else {
    status = LIBXSMM_DNN_ERR_FC_UNSUPPORTED_FUSION;
  }
#undef LIBXSMM_DNN_FC_BWD_USE_AVX512
#else /* should not happen */
  LIBXSMM_UNUSED(handle); LIBXSMM_UNUSED(kind); LIBXSMM_UNUSED(start_thread); LIBXSMM_UNUSED(tid);
  status = LIBXSMM_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}

LIBXSMM_API_INTERN LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512_CORE)
libxsmm_dnn_err_t libxsmm_dnn_fullyconnected_st_bwdupd_ncnc_kcck_bf16_bf16_emu(libxsmm_dnn_fullyconnected* handle, libxsmm_dnn_compute_kind kind, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
#if defined(LIBXSMM_INTRINSICS_AVX512_CORE) /*__AVX512F__,__AVX512BW__,__AVX512DQ__*/
  typedef libxsmm_bfloat16 element_input_type;
  typedef libxsmm_bfloat16 element_output_type;
  typedef libxsmm_bfloat16 element_filter_type;
  libxsmm_bsmmfunction_reducebatch_strd batchreduce_kernel_bwd = handle->gemm_bwd.xgemm.bsmrs;
  libxsmm_bmmfunction_reducebatch_strd batchreduce_kernel_bwd_zerobeta = handle->gemm_bwd2.xgemm.bmrs;
  libxsmm_bsmmfunction_reducebatch_strd batchreduce_kernel_upd = handle->gemm_upd.xgemm.bsmrs;
  libxsmm_bmmfunction_reducebatch_strd batchreduce_kernel_upd_zerobeta = handle->gemm_upd2.xgemm.bmrs;

  /* some portable macrros fof BF16 <-> FP32 */
# include "template/libxsmm_dnn_bf16_macros_define.tpl.c"

  if ( handle->desc.fuse_ops == LIBXSMM_DNN_FULLYCONNECTED_FUSE_NONE ) {
# include "template/libxsmm_dnn_fullyconnected_st_bwdupd_ncnc_kcck_generic_bf16.tpl.c"
  } else if ( handle->desc.fuse_ops == LIBXSMM_DNN_FULLYCONNECTED_FUSE_BIAS ) {
#define LIBXSMM_DNN_FC_BWD_FUSE_BIAS
# include "template/libxsmm_dnn_fullyconnected_st_bwdupd_ncnc_kcck_generic_bf16.tpl.c"
#undef LIBXSMM_DNN_FC_BWD_FUSE_BIAS
  } else if ( handle->desc.fuse_ops == LIBXSMM_DNN_FULLYCONNECTED_FUSE_RELU ) {
#define LIBXSMM_DNN_FC_BWD_FUSE_RELU
# include "template/libxsmm_dnn_fullyconnected_st_bwdupd_ncnc_kcck_generic_bf16.tpl.c"
#undef LIBXSMM_DNN_FC_BWD_FUSE_RELU
  } else if ( handle->desc.fuse_ops == LIBXSMM_DNN_FULLYCONNECTED_FUSE_SIGMOID ) {
#define LIBXSMM_DNN_FC_BWD_FUSE_SIGMOID
# include "template/libxsmm_dnn_fullyconnected_st_bwdupd_ncnc_kcck_generic_bf16.tpl.c"
#undef LIBXSMM_DNN_FC_BWD_FUSE_SIGMOID
  } else if ( handle->desc.fuse_ops == LIBXSMM_DNN_FULLYCONNECTED_FUSE_BIAS_RELU ) {
#define LIBXSMM_DNN_FC_BWD_FUSE_BIAS
#define LIBXSMM_DNN_FC_BWD_FUSE_RELU
# include "template/libxsmm_dnn_fullyconnected_st_bwdupd_ncnc_kcck_generic_bf16.tpl.c"
#undef LIBXSMM_DNN_FC_BWD_FUSE_RELU
#undef LIBXSMM_DNN_FC_BWD_FUSE_BIAS
  } else if ( handle->desc.fuse_ops == LIBXSMM_DNN_FULLYCONNECTED_FUSE_BIAS_SIGMOID ) {
#define LIBXSMM_DNN_FC_BWD_FUSE_BIAS
#define LIBXSMM_DNN_FC_BWD_FUSE_SIGMOID
# include "template/libxsmm_dnn_fullyconnected_st_bwdupd_ncnc_kcck_generic_bf16.tpl.c"
#undef LIBXSMM_DNN_FC_BWD_FUSE_SIGMOID
#undef LIBXSMM_DNN_FC_BWD_FUSE_BIAS
  } else {
    status = LIBXSMM_DNN_ERR_FC_UNSUPPORTED_FUSION;
  }

# include "template/libxsmm_dnn_bf16_macros_undefine.tpl.c"
#else /* should not happen */
  LIBXSMM_UNUSED(handle); LIBXSMM_UNUSED(kind); LIBXSMM_UNUSED(start_thread); LIBXSMM_UNUSED(tid);
  status = LIBXSMM_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}

#if defined(LIBXSMM_INTRINSICS_AVX512_CPX)
  LIBXSMM_API_INTERN LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512_CPX)
libxsmm_dnn_err_t libxsmm_dnn_fullyconnected_st_bwdupd_ncnc_kcck_bf16_bf16(libxsmm_dnn_fullyconnected* handle, libxsmm_dnn_compute_kind kind, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
#if defined(LIBXSMM_INTRINSICS_AVX512_CPX) /*__AVX512F__,__AVX512BW__,__AVX512DQ__,__AVX512BF16__*/
  typedef libxsmm_bfloat16 element_input_type;
  typedef libxsmm_bfloat16 element_output_type;
  typedef libxsmm_bfloat16 element_filter_type;
  libxsmm_bsmmfunction_reducebatch_strd batchreduce_kernel_bwd = handle->gemm_bwd.xgemm.bsmrs;
  libxsmm_bmmfunction_reducebatch_strd batchreduce_kernel_bwd_zerobeta = handle->gemm_bwd2.xgemm.bmrs;
  libxsmm_bsmmfunction_reducebatch_strd batchreduce_kernel_upd = handle->gemm_upd.xgemm.bsmrs;
  libxsmm_bmmfunction_reducebatch_strd batchreduce_kernel_upd_zerobeta = handle->gemm_upd2.xgemm.bmrs;

#define LIBXSMM_DNN_BF16_USE_CPX_AVX512_NI
  /* some portable macrros fof BF16 <-> FP32 */
# include "template/libxsmm_dnn_bf16_macros_define.tpl.c"

  if ( handle->desc.fuse_ops == LIBXSMM_DNN_FULLYCONNECTED_FUSE_NONE ) {
# include "template/libxsmm_dnn_fullyconnected_st_bwdupd_ncnc_kcck_generic_bf16.tpl.c"
  } else if ( handle->desc.fuse_ops == LIBXSMM_DNN_FULLYCONNECTED_FUSE_BIAS ) {
#define LIBXSMM_DNN_FC_BWD_FUSE_BIAS
# include "template/libxsmm_dnn_fullyconnected_st_bwdupd_ncnc_kcck_generic_bf16.tpl.c"
#undef LIBXSMM_DNN_FC_BWD_FUSE_BIAS
  } else if ( handle->desc.fuse_ops == LIBXSMM_DNN_FULLYCONNECTED_FUSE_RELU ) {
#define LIBXSMM_DNN_FC_BWD_FUSE_RELU
# include "template/libxsmm_dnn_fullyconnected_st_bwdupd_ncnc_kcck_generic_bf16.tpl.c"
#undef LIBXSMM_DNN_FC_BWD_FUSE_RELU
  } else if ( handle->desc.fuse_ops == LIBXSMM_DNN_FULLYCONNECTED_FUSE_SIGMOID ) {
#define LIBXSMM_DNN_FC_BWD_FUSE_SIGMOID
# include "template/libxsmm_dnn_fullyconnected_st_bwdupd_ncnc_kcck_generic_bf16.tpl.c"
#undef LIBXSMM_DNN_FC_BWD_FUSE_SIGMOID
  } else if ( handle->desc.fuse_ops == LIBXSMM_DNN_FULLYCONNECTED_FUSE_BIAS_RELU ) {
#define LIBXSMM_DNN_FC_BWD_FUSE_BIAS
#define LIBXSMM_DNN_FC_BWD_FUSE_RELU
# include "template/libxsmm_dnn_fullyconnected_st_bwdupd_ncnc_kcck_generic_bf16.tpl.c"
#undef LIBXSMM_DNN_FC_BWD_FUSE_RELU
#undef LIBXSMM_DNN_FC_BWD_FUSE_BIAS
  } else if ( handle->desc.fuse_ops == LIBXSMM_DNN_FULLYCONNECTED_FUSE_BIAS_SIGMOID ) {
#define LIBXSMM_DNN_FC_BWD_FUSE_BIAS
#define LIBXSMM_DNN_FC_BWD_FUSE_SIGMOID
# include "template/libxsmm_dnn_fullyconnected_st_bwdupd_ncnc_kcck_generic_bf16.tpl.c"
#undef LIBXSMM_DNN_FC_BWD_FUSE_SIGMOID
#undef LIBXSMM_DNN_FC_BWD_FUSE_BIAS
  } else {
    status = LIBXSMM_DNN_ERR_FC_UNSUPPORTED_FUSION;
  }

# include "template/libxsmm_dnn_bf16_macros_undefine.tpl.c"
#undef LIBXSMM_DNN_BF16_USE_CPX_AVX512_NI
#else /* should not happen */
  LIBXSMM_UNUSED(handle); LIBXSMM_UNUSED(kind); LIBXSMM_UNUSED(start_thread); LIBXSMM_UNUSED(tid);
  status = LIBXSMM_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}
#else
LIBXSMM_API_INTERN LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512_CORE)
libxsmm_dnn_err_t libxsmm_dnn_fullyconnected_st_bwdupd_ncnc_kcck_bf16_bf16(libxsmm_dnn_fullyconnected* handle, libxsmm_dnn_compute_kind kind, int start_thread, int tid)
{
  return libxsmm_dnn_fullyconnected_st_bwdupd_ncnc_kcck_bf16_bf16_emu( handle, kind, start_thread, tid );
}
#endif

#if defined(LIBXSMM_INTRINSICS_AVX512_CPX)
LIBXSMM_API_INTERN LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512_CPX)
libxsmm_dnn_err_t libxsmm_dnn_fullyconnected_st_bwdupd_ncnc_kcck_bf16_bf16_amx(libxsmm_dnn_fullyconnected* handle, libxsmm_dnn_compute_kind kind, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
#if defined(LIBXSMM_INTRINSICS_AVX512_CPX) /*__AVX512F__,__AVX512BW__,__AVX512DQ__,__AVX512BF16__*/
  typedef libxsmm_bfloat16 element_input_type;
  typedef libxsmm_bfloat16 element_output_type;
  typedef libxsmm_bfloat16 element_filter_type;
  libxsmm_bsmmfunction_reducebatch_strd batchreduce_kernel_bwd = handle->gemm_bwd.xgemm.bsmrs;
  libxsmm_bmmfunction_reducebatch_strd bf16_batchreduce_kernel_bwd_zerobeta = handle->gemm_bwd3.xgemm.bmrs;
  libxsmm_bsmmfunction_reducebatch_strd batchreduce_kernel_upd = handle->gemm_upd.xgemm.bsmrs;
  libxsmm_bmmfunction_reducebatch_strd bf16_batchreduce_kernel_upd_zerobeta = handle->gemm_upd3.xgemm.bmrs;
  libxsmm_bsmmfunction bwd_tile_config_kernel = handle->bwd_config_kernel;
  /*libxsmm_bsmmfunction upd_tile_config_kernel = handle->upd_config_kernel;*/

#define LIBXSMM_DNN_BF16_USE_CPX_AVX512_NI
  /* some portable macrros fof BF16 <-> FP32 */
# include "template/libxsmm_dnn_bf16_macros_define.tpl.c"

  if ( handle->desc.fuse_ops == LIBXSMM_DNN_FULLYCONNECTED_FUSE_NONE ) {
# include "template/libxsmm_dnn_fullyconnected_st_bwdupd_ncnc_kcck_generic_bf16_amx.tpl.c"
  } else if ( handle->desc.fuse_ops == LIBXSMM_DNN_FULLYCONNECTED_FUSE_BIAS ) {
#define LIBXSMM_DNN_FC_BWD_FUSE_BIAS
# include "template/libxsmm_dnn_fullyconnected_st_bwdupd_ncnc_kcck_generic_bf16_amx.tpl.c"
#undef LIBXSMM_DNN_FC_BWD_FUSE_BIAS
  } else if ( handle->desc.fuse_ops == LIBXSMM_DNN_FULLYCONNECTED_FUSE_RELU ) {
#define LIBXSMM_DNN_FC_BWD_FUSE_RELU
# include "template/libxsmm_dnn_fullyconnected_st_bwdupd_ncnc_kcck_generic_bf16_amx.tpl.c"
#undef LIBXSMM_DNN_FC_BWD_FUSE_RELU
  } else if ( handle->desc.fuse_ops == LIBXSMM_DNN_FULLYCONNECTED_FUSE_SIGMOID ) {
#define LIBXSMM_DNN_FC_BWD_FUSE_SIGMOID
# include "template/libxsmm_dnn_fullyconnected_st_bwdupd_ncnc_kcck_generic_bf16_amx.tpl.c"
#undef LIBXSMM_DNN_FC_BWD_FUSE_SIGMOID
  } else if ( handle->desc.fuse_ops == LIBXSMM_DNN_FULLYCONNECTED_FUSE_BIAS_RELU ) {
#define LIBXSMM_DNN_FC_BWD_FUSE_BIAS
#define LIBXSMM_DNN_FC_BWD_FUSE_RELU
# include "template/libxsmm_dnn_fullyconnected_st_bwdupd_ncnc_kcck_generic_bf16_amx.tpl.c"
#undef LIBXSMM_DNN_FC_BWD_FUSE_RELU
#undef LIBXSMM_DNN_FC_BWD_FUSE_BIAS
  } else if ( handle->desc.fuse_ops == LIBXSMM_DNN_FULLYCONNECTED_FUSE_BIAS_SIGMOID ) {
#define LIBXSMM_DNN_FC_BWD_FUSE_BIAS
#define LIBXSMM_DNN_FC_BWD_FUSE_SIGMOID
# include "template/libxsmm_dnn_fullyconnected_st_bwdupd_ncnc_kcck_generic_bf16_amx.tpl.c"
#undef LIBXSMM_DNN_FC_BWD_FUSE_SIGMOID
#undef LIBXSMM_DNN_FC_BWD_FUSE_BIAS
  } else {
    status = LIBXSMM_DNN_ERR_FC_UNSUPPORTED_FUSION;
  }

# include "template/libxsmm_dnn_bf16_macros_undefine.tpl.c"
#undef LIBXSMM_DNN_BF16_USE_CPX_AVX512_NI

#else /* should not happen */
  LIBXSMM_UNUSED(handle); LIBXSMM_UNUSED(kind); LIBXSMM_UNUSED(start_thread); LIBXSMM_UNUSED(tid);
  status = LIBXSMM_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}
#else
LIBXSMM_API_INTERN LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512_CORE)
libxsmm_dnn_err_t libxsmm_dnn_fullyconnected_st_bwdupd_ncnc_kcck_bf16_bf16_amx(libxsmm_dnn_fullyconnected* handle, libxsmm_dnn_compute_kind kind, int start_thread, int tid)
{
  return libxsmm_dnn_fullyconnected_st_bwdupd_ncnc_kcck_bf16_bf16_amx_emu(handle, kind, start_thread, tid);
}
#endif

LIBXSMM_API_INTERN LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512_CORE)
libxsmm_dnn_err_t libxsmm_dnn_fullyconnected_st_bwdupd_ncnc_kcck_bf16_bf16_amx_emu(libxsmm_dnn_fullyconnected* handle, libxsmm_dnn_compute_kind kind, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
#if defined(LIBXSMM_INTRINSICS_AVX512_CORE) /*__AVX512F__,__AVX512BW__,__AVX512DQ__*/
  typedef libxsmm_bfloat16 element_input_type;
  typedef libxsmm_bfloat16 element_output_type;
  typedef libxsmm_bfloat16 element_filter_type;
  libxsmm_bsmmfunction_reducebatch_strd batchreduce_kernel_bwd = handle->gemm_bwd.xgemm.bsmrs;
  libxsmm_bmmfunction_reducebatch_strd bf16_batchreduce_kernel_bwd_zerobeta = handle->gemm_bwd3.xgemm.bmrs;
  libxsmm_bsmmfunction_reducebatch_strd batchreduce_kernel_upd = handle->gemm_upd.xgemm.bsmrs;
  libxsmm_bmmfunction_reducebatch_strd bf16_batchreduce_kernel_upd_zerobeta = handle->gemm_upd3.xgemm.bmrs;
  libxsmm_bsmmfunction bwd_tile_config_kernel = handle->bwd_config_kernel;
  /*libxsmm_bsmmfunction upd_tile_config_kernel = handle->upd_config_kernel;*/

  /* some portable macrros fof BF16 <-> FP32 */
# include "template/libxsmm_dnn_bf16_macros_define.tpl.c"

  if ( handle->desc.fuse_ops == LIBXSMM_DNN_FULLYCONNECTED_FUSE_NONE ) {
# include "template/libxsmm_dnn_fullyconnected_st_bwdupd_ncnc_kcck_generic_bf16_amx.tpl.c"
  } else if ( handle->desc.fuse_ops == LIBXSMM_DNN_FULLYCONNECTED_FUSE_BIAS ) {
#define LIBXSMM_DNN_FC_BWD_FUSE_BIAS
# include "template/libxsmm_dnn_fullyconnected_st_bwdupd_ncnc_kcck_generic_bf16_amx.tpl.c"
#undef LIBXSMM_DNN_FC_BWD_FUSE_BIAS
  } else if ( handle->desc.fuse_ops == LIBXSMM_DNN_FULLYCONNECTED_FUSE_RELU ) {
#define LIBXSMM_DNN_FC_BWD_FUSE_RELU
# include "template/libxsmm_dnn_fullyconnected_st_bwdupd_ncnc_kcck_generic_bf16_amx.tpl.c"
#undef LIBXSMM_DNN_FC_BWD_FUSE_RELU
  } else if ( handle->desc.fuse_ops == LIBXSMM_DNN_FULLYCONNECTED_FUSE_SIGMOID ) {
#define LIBXSMM_DNN_FC_BWD_FUSE_SIGMOID
# include "template/libxsmm_dnn_fullyconnected_st_bwdupd_ncnc_kcck_generic_bf16_amx.tpl.c"
#undef LIBXSMM_DNN_FC_BWD_FUSE_SIGMOID
  } else if ( handle->desc.fuse_ops == LIBXSMM_DNN_FULLYCONNECTED_FUSE_BIAS_RELU ) {
#define LIBXSMM_DNN_FC_BWD_FUSE_BIAS
#define LIBXSMM_DNN_FC_BWD_FUSE_RELU
# include "template/libxsmm_dnn_fullyconnected_st_bwdupd_ncnc_kcck_generic_bf16_amx.tpl.c"
#undef LIBXSMM_DNN_FC_BWD_FUSE_RELU
#undef LIBXSMM_DNN_FC_BWD_FUSE_BIAS
  } else if ( handle->desc.fuse_ops == LIBXSMM_DNN_FULLYCONNECTED_FUSE_BIAS_SIGMOID ) {
#define LIBXSMM_DNN_FC_BWD_FUSE_BIAS
#define LIBXSMM_DNN_FC_BWD_FUSE_SIGMOID
# include "template/libxsmm_dnn_fullyconnected_st_bwdupd_ncnc_kcck_generic_bf16_amx.tpl.c"
#undef LIBXSMM_DNN_FC_BWD_FUSE_SIGMOID
#undef LIBXSMM_DNN_FC_BWD_FUSE_BIAS
  } else {
    status = LIBXSMM_DNN_ERR_FC_UNSUPPORTED_FUSION;
  }

# include "template/libxsmm_dnn_bf16_macros_undefine.tpl.c"

#else /* should not happen */
  LIBXSMM_UNUSED(handle); LIBXSMM_UNUSED(kind); LIBXSMM_UNUSED(start_thread); LIBXSMM_UNUSED(tid);
  status = LIBXSMM_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}

LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_fullyconnected_st_bwdupd_custom(libxsmm_dnn_fullyconnected* handle, libxsmm_dnn_compute_kind kind, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  /* check if all required tensors are bound */
  if ( kind == LIBXSMM_DNN_COMPUTE_KIND_BWD ) {
    if (handle->grad_input == 0 || handle->grad_output == 0 ||
        handle->reg_filter == 0 || handle->scratch == 0         ) {
      status = LIBXSMM_DNN_ERR_DATA_NOT_BOUND;
      return status;
    }
  } else if ( kind == LIBXSMM_DNN_COMPUTE_KIND_UPD ) {
    if (handle->reg_input == 0   || handle->grad_output == 0 ||
        handle->grad_filter == 0 || handle->scratch == 0         ) {
      status = LIBXSMM_DNN_ERR_DATA_NOT_BOUND;
      return status;
    }
  } else {
    if (handle->grad_input == 0 || handle->grad_output == 0 ||
        handle->reg_input  == 0 || handle->grad_filter == 0 ||
        handle->reg_filter == 0 || handle->scratch == 0         ) {
      status = LIBXSMM_DNN_ERR_DATA_NOT_BOUND;
      return status;
    }
  }

  /* check if we are on an AVX512 platform */
#if defined(LIBXSMM_INTRINSICS_AVX512) /*__AVX512F__*/
  if ( (handle->target_archid >= LIBXSMM_X86_AVX512) && (handle->target_archid <= LIBXSMM_X86_ALLFEAT) ) {
    if (handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_F32 && handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_F32 ) {
      status = libxsmm_dnn_fullyconnected_st_bwdupd_custom_f32_f32( handle, kind, start_thread, tid);
    }
#if defined(LIBXSMM_INTRINSICS_AVX512_CORE) /*__AVX512F__*/
    else if (handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_BF16 && handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_F32 ) {
      status = libxsmm_dnn_fullyconnected_st_bwdupd_custom_bf16_f32( handle, kind, start_thread, tid);
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
      typedef libxsmm_smmfunction gemm_function;
      libxsmm_blasint lda_bwd = (libxsmm_blasint)handle->ifmblock;
      libxsmm_blasint ldb_bwd = (libxsmm_blasint)handle->desc.K;
      libxsmm_blasint ldc_bwd = (libxsmm_blasint)handle->desc.C;
      libxsmm_blasint lda_upd = (libxsmm_blasint)handle->desc.K;
      libxsmm_blasint ldb_upd = (libxsmm_blasint)handle->desc.N;
      libxsmm_blasint ldc_upd = (libxsmm_blasint)handle->ofmblock;
      element_input_type alpha = (element_input_type)1;
      element_input_type beta = (element_input_type)0;

      if ( handle->desc.fuse_ops == LIBXSMM_DNN_FULLYCONNECTED_FUSE_NONE ) {
        gemm_function gemm_kernel_bwd = libxsmm_smmdispatch(handle->ifmblock, handle->desc.N, handle->desc.K, &lda_bwd, &ldb_bwd, &ldc_bwd, &alpha, &beta, NULL, NULL);
        gemm_function gemm_kernel_upd = libxsmm_smmdispatch(handle->ofmblock, handle->ifmblock, handle->desc.N, &lda_upd, &ldb_upd, &ldc_upd, &alpha, &beta, NULL, NULL);
# include "template/libxsmm_dnn_fullyconnected_st_bwdupd_custom_generic.tpl.c"
      } else {
        status = LIBXSMM_DNN_ERR_FC_UNSUPPORTED_FUSION;
      }
    } else if (handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_BF16 && handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_F32 ) {
      typedef libxsmm_bfloat16 element_input_type;
      typedef float element_output_type;
      typedef libxsmm_bfloat16 element_filter_type;
      typedef libxsmm_smmfunction gemm_function;
      libxsmm_blasint lda_bwd = (libxsmm_blasint)handle->ifmblock;
      libxsmm_blasint ldb_bwd = (libxsmm_blasint)handle->desc.K;
      libxsmm_blasint ldc_bwd = (libxsmm_blasint)handle->desc.C;
      libxsmm_blasint lda_upd = (libxsmm_blasint)handle->desc.K;
      libxsmm_blasint ldb_upd = (libxsmm_blasint)handle->desc.N;
      libxsmm_blasint ldc_upd = (libxsmm_blasint)handle->ofmblock;
      float alpha = (element_input_type)1;
      float beta = (element_input_type)0;

      if ( handle->desc.fuse_ops == LIBXSMM_DNN_FULLYCONNECTED_FUSE_NONE ) {
        gemm_function gemm_kernel_bwd = libxsmm_smmdispatch(handle->ifmblock, handle->desc.N, handle->desc.K, &lda_bwd, &ldb_bwd, &ldc_bwd, &alpha, &beta, NULL, NULL);
        gemm_function gemm_kernel_upd = libxsmm_smmdispatch(handle->ofmblock, handle->ifmblock, handle->desc.N, &lda_upd, &ldb_upd, &ldc_upd, &alpha, &beta, NULL, NULL);
# define LIBXSMM_DNN_FULLYCONNECTED_BWD_BF16_F32
# define LIBXSMM_DNN_FULLYCONNECTED_UPD_BF16_F32
# include "template/libxsmm_dnn_fullyconnected_st_bwdupd_custom_generic.tpl.c"
# undef LIBXSMM_DNN_FULLYCONNECTED_UPD_BF16_F32
# undef LIBXSMM_DNN_FULLYCONNECTED_BWD_BF16_F32
      } else {
        status = LIBXSMM_DNN_ERR_FC_UNSUPPORTED_FUSION;
      }
    } else {
      status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  }

  return status;
}


LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_fullyconnected_st_bwdupd_ncnc_kcck(libxsmm_dnn_fullyconnected* handle, libxsmm_dnn_compute_kind kind, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
  int l_emu_amx = 0;
  const char *const l_env_emu_amx = getenv("EMULATE_AMX");
  if ( 0 == l_env_emu_amx ) {
  } else {
    l_emu_amx = atoi(l_env_emu_amx);
  }

  /* check if all required tensors are bound */
  if ( kind == LIBXSMM_DNN_COMPUTE_KIND_BWD ) {
    if (handle->grad_input == 0 || handle->grad_output == 0 ||
        handle->reg_filter == 0 || handle->scratch == 0         ) {
      status = LIBXSMM_DNN_ERR_DATA_NOT_BOUND;
      return status;
    }
  } else if ( kind == LIBXSMM_DNN_COMPUTE_KIND_UPD ) {
    if (handle->reg_input == 0   || handle->grad_output == 0 ||
        handle->grad_filter == 0 || handle->scratch == 0         ) {
      status = LIBXSMM_DNN_ERR_DATA_NOT_BOUND;
      return status;
    }
  } else {
    if (handle->grad_input == 0 || handle->grad_output == 0 ||
        handle->reg_input  == 0 || handle->grad_filter == 0 ||
        handle->reg_filter == 0 || handle->scratch == 0         ) {
      status = LIBXSMM_DNN_ERR_DATA_NOT_BOUND;
      return status;
    }
  }

  if ( ((handle->desc.fuse_ops & LIBXSMM_DNN_FULLYCONNECTED_FUSE_BIAS ) != 0) && ( handle->grad_bias == 0 ) )  {
    status = LIBXSMM_DNN_ERR_DATA_NOT_BOUND;
    return status;
  }
  if ( ((handle->desc.fuse_ops & LIBXSMM_DNN_FULLYCONNECTED_FUSE_RELU ) != 0) && ( handle->relumask == 0 ) )  {
    status = LIBXSMM_DNN_ERR_DATA_NOT_BOUND;
    return status;
  }

  /* check if we are on an AVX512 platform */
#if defined(LIBXSMM_INTRINSICS_AVX512) /*__AVX512F__*/
  if ( (handle->target_archid >= LIBXSMM_X86_AVX512) && (handle->target_archid <= LIBXSMM_X86_ALLFEAT) ) {
    if (handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_F32 && handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_F32 ) {
      status = libxsmm_dnn_fullyconnected_st_bwdupd_ncnc_kcck_f32_f32( handle, kind, start_thread, tid);
    }
#if defined(LIBXSMM_INTRINSICS_AVX512_CPX) /*__AVX512F__,__AVX512BW__,__AVX512DQ__,__AVX512BF16__*/
    else if ( handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_BF16 && handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_BF16 && handle->target_archid >= LIBXSMM_X86_AVX512_CORE && handle->target_archid < LIBXSMM_X86_AVX512_CPX) {
      status = libxsmm_dnn_fullyconnected_st_bwdupd_ncnc_kcck_bf16_bf16_emu( handle, kind, start_thread, tid);
    } else if ( handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_BF16 && handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_BF16 && handle->target_archid >= LIBXSMM_X86_AVX512_CPX  && handle->target_archid < LIBXSMM_X86_AVX512_SPR) {
      status = libxsmm_dnn_fullyconnected_st_bwdupd_ncnc_kcck_bf16_bf16( handle, kind, start_thread, tid);
    } else if ( handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_BF16 && handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_BF16 && handle->target_archid >= LIBXSMM_X86_AVX512_SPR) {
      if ( l_emu_amx == 0 ) {
        status = libxsmm_dnn_fullyconnected_st_bwdupd_ncnc_kcck_bf16_bf16_amx( handle, kind, start_thread, tid);
      } else {
        status = libxsmm_dnn_fullyconnected_st_bwdupd_ncnc_kcck_bf16_bf16_amx_emu( handle, kind, start_thread, tid);
      }
    }
#elif defined(LIBXSMM_INTRINSICS_AVX512_CORE) /*__AVX512F__,__AVX512BW__,__AVX512DQ__*/
    else if (handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_BF16 && handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_BF16 && handle->target_archid >= LIBXSMM_X86_AVX512_CORE && handle->target_archid < LIBXSMM_X86_AVX512_SPR ) {
      status = libxsmm_dnn_fullyconnected_st_bwdupd_ncnc_kcck_bf16_bf16_emu( handle, kind, start_thread, tid);
    } else if ( handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_BF16 && handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_BF16 && handle->target_archid >= LIBXSMM_X86_AVX512_SPR ) {
      if ( l_emu_amx == 0 ) {
        status = libxsmm_dnn_fullyconnected_st_bwdupd_ncnc_kcck_bf16_bf16_amx( handle, kind, start_thread, tid);
      } else {
        status = libxsmm_dnn_fullyconnected_st_bwdupd_ncnc_kcck_bf16_bf16_amx_emu( handle, kind, start_thread, tid);
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
    LIBXSMM_UNUSED( l_emu_amx );
    if (handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_F32 && handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_F32 ) {
      typedef float element_input_type;
      typedef float element_output_type;
      typedef float element_filter_type;
      libxsmm_smmfunction_reducebatch_strd batchreduce_kernel_bwd = handle->gemm_bwd.xgemm.smrs;
      libxsmm_smmfunction_reducebatch_strd batchreduce_kernel_bwd_zerobeta = handle->gemm_bwd2.xgemm.smrs;
      libxsmm_smmfunction_reducebatch_strd batchreduce_kernel_upd = handle->gemm_upd.xgemm.smrs;
      libxsmm_smmfunction_reducebatch_strd batchreduce_kernel_upd_zerobeta = handle->gemm_upd2.xgemm.smrs;

      if ( handle->desc.fuse_ops == LIBXSMM_DNN_FULLYCONNECTED_FUSE_NONE ) {
# include "template/libxsmm_dnn_fullyconnected_st_bwdupd_ncnc_kcck_generic.tpl.c"
      } else if ( handle->desc.fuse_ops == LIBXSMM_DNN_FULLYCONNECTED_FUSE_BIAS ) {
#define LIBXSMM_DNN_FC_BWD_FUSE_BIAS
# include "template/libxsmm_dnn_fullyconnected_st_bwdupd_ncnc_kcck_generic.tpl.c"
#undef LIBXSMM_DNN_FC_BWD_FUSE_BIAS
      } else if ( handle->desc.fuse_ops == LIBXSMM_DNN_FULLYCONNECTED_FUSE_RELU ) {
#define LIBXSMM_DNN_FC_BWD_FUSE_RELU
# include "template/libxsmm_dnn_fullyconnected_st_bwdupd_ncnc_kcck_generic.tpl.c"
#undef LIBXSMM_DNN_FC_BWD_FUSE_RELU
      } else if ( handle->desc.fuse_ops == LIBXSMM_DNN_FULLYCONNECTED_FUSE_SIGMOID ) {
#define LIBXSMM_DNN_FC_BWD_FUSE_SIGMOID
# include "template/libxsmm_dnn_fullyconnected_st_bwdupd_ncnc_kcck_generic.tpl.c"
#undef LIBXSMM_DNN_FC_BWD_FUSE_SIGMOID
      } else if ( handle->desc.fuse_ops == LIBXSMM_DNN_FULLYCONNECTED_FUSE_BIAS_RELU ) {
#define LIBXSMM_DNN_FC_BWD_FUSE_BIAS
#define LIBXSMM_DNN_FC_BWD_FUSE_RELU
# include "template/libxsmm_dnn_fullyconnected_st_bwdupd_ncnc_kcck_generic.tpl.c"
#undef LIBXSMM_DNN_FC_BWD_FUSE_RELU
#undef LIBXSMM_DNN_FC_BWD_FUSE_BIAS
      } else if ( handle->desc.fuse_ops == LIBXSMM_DNN_FULLYCONNECTED_FUSE_BIAS_SIGMOID ) {
#define LIBXSMM_DNN_FC_BWD_FUSE_BIAS
#define LIBXSMM_DNN_FC_BWD_FUSE_SIGMOID
# include "template/libxsmm_dnn_fullyconnected_st_bwdupd_ncnc_kcck_generic.tpl.c"
#undef LIBXSMM_DNN_FC_BWD_FUSE_SIGMOID
#undef LIBXSMM_DNN_FC_BWD_FUSE_BIAS
      } else {
        status = LIBXSMM_DNN_ERR_FC_UNSUPPORTED_FUSION;
      }
    } else {
      status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  }

  return status;
}


LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_fullyconnected_st_bwdupd_nhwc(libxsmm_dnn_fullyconnected* handle, libxsmm_dnn_compute_kind kind, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_ERR_NOT_IMPLEMENTED;
  LIBXSMM_UNUSED( handle );
  LIBXSMM_UNUSED( kind );
  LIBXSMM_UNUSED( start_thread );
  LIBXSMM_UNUSED( tid );
  return status;
}

