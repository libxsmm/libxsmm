/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Evangelos Georganas (Intel Corp.), Alexander Heinecke (Intel Corp.)
******************************************************************************/

#define NATIVE_STORECVT_F32_BF16(A,B) _mm256_storeu_si256((__m256i*)(A), (__m256i)LIBXSMM_INTRINSISCS_MM512_CVTNEPS_PBH(B))
{
  float* _dout              = &LIBXSMM_VLA_ACCESS(4, dout, inb, ikb, 0, 0, kBlocks, bn, bk);
  element_input_type* _dh   = &LIBXSMM_VLA_ACCESS(5, dh, j, inb, ikb, 0, 0, nBlocks, kBlocks, bn, bk);
  element_input_type* _o    = &LIBXSMM_VLA_ACCESS(5, o, j, inb, ikb, 0, 0, nBlocks, kBlocks, bn, bk);
  element_input_type* _co   = &LIBXSMM_VLA_ACCESS(5, co, j, inb, ikb, 0, 0, nBlocks, kBlocks, bn, bk);
  element_input_type* _dcs  = &LIBXSMM_VLA_ACCESS(4, dcs, inb, ikb, 0, 0, kBlocks, bn, bk);
  element_input_type* _ii   = &LIBXSMM_VLA_ACCESS(5, i, j, inb, ikb, 0, 0, nBlocks, kBlocks, bn, bk);
  element_input_type* _ci   = &LIBXSMM_VLA_ACCESS(5, ci, j, inb, ikb, 0, 0, nBlocks, kBlocks, bn, bk);
  element_input_type* _dci  = &LIBXSMM_VLA_ACCESS(4, dci, inb, ikb, 0, 0, kBlocks, bn, bk);
  element_input_type* _di   = &LIBXSMM_VLA_ACCESS(4, di, inb, ikb, 0, 0, kBlocks, bn, bk);
  element_input_type* _cps  = cps_ptr;
  element_input_type* _f    = &LIBXSMM_VLA_ACCESS(5, f, j, inb, ikb, 0, 0, nBlocks, kBlocks, bn, bk);
  element_input_type* _df   = &LIBXSMM_VLA_ACCESS(4, df, inb, ikb, 0, 0, kBlocks, bn, bk);
  element_input_type* _dp   = &LIBXSMM_VLA_ACCESS(4, dp, inb, ikb, 0, 0, kBlocks, bn, bk);
  element_input_type* _dcp  = &LIBXSMM_VLA_ACCESS(4, dcp, inb, ikb, 0, 0, kBlocks, bn, bk);
  element_input_type* _dciB = &LIBXSMM_VLA_ACCESS(5, dciB, ikb, inb, 0, 0, 0, nBlocks, bn_lp, bk, lpb);
  element_input_type* _diB  = &LIBXSMM_VLA_ACCESS(5, diB, ikb, inb, 0, 0, 0, nBlocks, bn_lp, bk, lpb);
  element_input_type* _dfB  = &LIBXSMM_VLA_ACCESS(5, dfB, ikb, inb, 0, 0, 0, nBlocks, bn_lp, bk, lpb);
  element_input_type* _dpB  = &LIBXSMM_VLA_ACCESS(5, dpB, ikb, inb, 0, 0, 0, nBlocks, bn_lp, bk, lpb);

  libxsmm_blasint _k, _j;
  __m512 _vdout, _vdh, _vo, _vt1, _vt2, _vco, _vdcs, _vdcp, _vii, _vci, _vdci, _vdi, _vcps, _vf, _vdf, _vdp;
  const __m512 _neg_ones = _mm512_set1_ps( (float)-1.0 );
  const __m512 _ones = _mm512_set1_ps( (float)1.0 );
  const int _lpb = 2;

  if (j == t-1) {
    for ( _j = 0; _j < bn; ++_j ) {
      for ( _k = 0; _k < bk; _k += 16 ) {
        _vdout = _mm512_loadcvt_bf16_fp32( &_dh[(_j*bk)+_k] );
        _vo = _mm512_loadcvt_bf16_fp32( &_o[(_j*bk)+_k] );
        _vt1 = _mm512_mul_ps( _vdout, _vo  );
        _vco = _mm512_loadcvt_bf16_fp32( &_co[(_j*bk)+_k] );
        _vt2 = _mm512_fnmsub_ps ( _vco, _vco, _neg_ones);
        _vt1 = _mm512_mul_ps( _vt1, _vt2 );
        _vdcs = _mm512_loadcvt_bf16_fp32( &_dcs[(_j*bk)+_k] );
        _vdcp = _mm512_add_ps( _vdcs, _vt1 );
        _vii = _mm512_loadcvt_bf16_fp32( &_ii[(_j*bk)+_k] );
        _vt1 = _mm512_mul_ps( _vii, _vdcp );
        _vci = _mm512_loadcvt_bf16_fp32( &_ci[(_j*bk)+_k] );
        _vt2 = _mm512_fnmsub_ps ( _vci, _vci, _neg_ones);
        _vdci = _mm512_mul_ps( _vt1, _vt2 );
        NATIVE_STORECVT_F32_BF16( &_dci[(_j*bk)+_k],  _vdci );
        _vt1 = _mm512_mul_ps( _vci, _vdcp );
        _vt2 = _mm512_sub_ps( _ones, _vii );
        _vdi = _mm512_mul_ps( _vii, _vt2);
        _vdi = _mm512_mul_ps( _vdi, _vt1);
        NATIVE_STORECVT_F32_BF16( &_di[(_j*bk)+_k], _vdi );
        _vcps = _mm512_loadcvt_bf16_fp32( &_cps[(_j*bk)+_k] );
        _vt1 = _mm512_mul_ps( _vcps, _vdcp );
        _vf = _mm512_loadcvt_bf16_fp32( &_f[(_j*bk)+_k] );
        _vt2 = _mm512_sub_ps( _ones, _vf );
        _vdf = _mm512_mul_ps( _vf, _vt2);
        _vdf = _mm512_mul_ps( _vdf, _vt1);
        NATIVE_STORECVT_F32_BF16( &_df[(_j*bk)+_k], _vdf );
        _vt1 = _mm512_mul_ps( _vdout, _vco);
        _vt2 = _mm512_sub_ps( _ones, _vo );
        _vt2 = _mm512_mul_ps( _vo, _vt2);
        _vdp = _mm512_mul_ps( _vt1, _vt2 );
        NATIVE_STORECVT_F32_BF16( &_dp[(_j*bk)+_k], _vdp );
        _vdcp = _mm512_mul_ps( _vdcp, _vf);
        NATIVE_STORECVT_F32_BF16( &_dcp[(_j*bk)+_k], _vdcp );
      }
    }
  } else {
    for ( _j = 0; _j < bn; ++_j ) {
      for ( _k = 0; _k < bk; _k += 16 ) {
        _vdout = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &_dout[(_j*bk)+_k] );
        _vdh = _mm512_loadcvt_bf16_fp32( &_dh[(_j*bk)+_k] );
        _vdout = _mm512_add_ps( _vdout, _vdh );
        _vo = _mm512_loadcvt_bf16_fp32( &_o[(_j*bk)+_k] );
        _vt1 = _mm512_mul_ps( _vdout, _vo  );
        _vco = _mm512_loadcvt_bf16_fp32( &_co[(_j*bk)+_k] );
        _vt2 = _mm512_fnmsub_ps ( _vco, _vco, _neg_ones);
        _vt1 = _mm512_mul_ps( _vt1, _vt2 );
        _vdcp = _mm512_loadcvt_bf16_fp32( &_dcp[(_j*bk)+_k] );
        _vdcp = _mm512_add_ps( _vdcp, _vt1 );
        _vii = _mm512_loadcvt_bf16_fp32( &_ii[(_j*bk)+_k] );
        _vt1 = _mm512_mul_ps( _vii, _vdcp );
        _vci = _mm512_loadcvt_bf16_fp32( &_ci[(_j*bk)+_k] );
        _vt2 = _mm512_fnmsub_ps ( _vci, _vci, _neg_ones);
        _vdci = _mm512_mul_ps( _vt1, _vt2 );
        NATIVE_STORECVT_F32_BF16( &_dci[(_j*bk)+_k], _vdci );
        _vt1 = _mm512_mul_ps( _vci, _vdcp );
        _vt2 = _mm512_sub_ps( _ones, _vii );
        _vdi = _mm512_mul_ps( _vii, _vt2);
        _vdi = _mm512_mul_ps( _vdi, _vt1);
        NATIVE_STORECVT_F32_BF16( &_di[(_j*bk)+_k], _vdi );
        _vcps = _mm512_loadcvt_bf16_fp32( &_cps[(_j*bk)+_k] );
        _vt1 = _mm512_mul_ps( _vcps, _vdcp );
        _vf = _mm512_loadcvt_bf16_fp32( &_f[(_j*bk)+_k] );
        _vt2 = _mm512_sub_ps( _ones, _vf );
        _vdf = _mm512_mul_ps( _vf, _vt2);
        _vdf = _mm512_mul_ps( _vdf, _vt1);
        NATIVE_STORECVT_F32_BF16( &_df[(_j*bk)+_k], _vdf );
        _vt1 = _mm512_mul_ps( _vdout, _vco);
        _vt2 = _mm512_sub_ps( _ones, _vo );
        _vt2 = _mm512_mul_ps( _vo, _vt2);
        _vdp = _mm512_mul_ps( _vt1, _vt2 );
        NATIVE_STORECVT_F32_BF16( &_dp[(_j*bk)+_k], _vdp );
        _vdcp = _mm512_mul_ps( _vdcp, _vf);
        NATIVE_STORECVT_F32_BF16( &_dcp[(_j*bk)+_k], _vdcp );
      }
    }
  }
  {
    /* Store di/dci/df/dp to diB/dciB/dfB/dpB which is CNNC AND vnni format */
    const __m512i perm_idx = LIBXSMM_INTRINSICS_MM512_SET_EPI16(31, 15, 30, 14, 29, 13, 28, 12, 27, 11, 26, 10, 25, 9, 24, 8, 23, 7, 22, 6, 21, 5, 20, 4, 19, 3, 18, 2, 17, 1, 16, 0);
    __m256i c0, c1;
    __m512i _c01;
    LIBXSMM_VLA_DECL(2, libxsmm_bfloat16, di_, _di, bk);
    LIBXSMM_VLA_DECL(2, libxsmm_bfloat16, df_, _df, bk);
    LIBXSMM_VLA_DECL(2, libxsmm_bfloat16, dp_, _dp, bk);
    LIBXSMM_VLA_DECL(2, libxsmm_bfloat16, dci_, _dci, bk);
    LIBXSMM_VLA_DECL(3, libxsmm_bfloat16, diB_, _diB, bk, _lpb);
    LIBXSMM_VLA_DECL(3, libxsmm_bfloat16, dfB_, _dfB, bk, _lpb);
    LIBXSMM_VLA_DECL(3, libxsmm_bfloat16, dpB_, _dpB, bk, _lpb);
    LIBXSMM_VLA_DECL(3, libxsmm_bfloat16, dciB_, _dciB, bk, _lpb);
    for (_j = 0; _j < bn; _j+=2) {
      for (_k = 0; _k < bk; _k+=16) {
        c0 = _mm256_loadu_si256((const __m256i*)&LIBXSMM_VLA_ACCESS(2, di_, _j, _k, bk));
        c1 = _mm256_loadu_si256((const __m256i*)&LIBXSMM_VLA_ACCESS(2, di_, _j+1, _k, bk));
        _c01 = _mm512_inserti64x4 (LIBXSMM_INTRINSICS_MM512_UNDEFINED_EPI32(), c0, 0);
        _c01 = _mm512_inserti64x4 (_c01, c1, 1);
        _mm512_store_epi32(&LIBXSMM_VLA_ACCESS(3, diB_, _j/_lpb, _k, 0, bk, _lpb), _mm512_permutexvar_epi16(perm_idx, _c01));
        c0 = _mm256_loadu_si256((const __m256i*)&LIBXSMM_VLA_ACCESS(2, df_, _j, _k, bk));
        c1 = _mm256_loadu_si256((const __m256i*)&LIBXSMM_VLA_ACCESS(2, df_, _j+1, _k, bk));
        _c01 = _mm512_inserti64x4 (LIBXSMM_INTRINSICS_MM512_UNDEFINED_EPI32(), c0, 0);
        _c01 = _mm512_inserti64x4 (_c01, c1, 1);
        _mm512_store_epi32(&LIBXSMM_VLA_ACCESS(3, dfB_, _j/_lpb, _k, 0, bk, _lpb), _mm512_permutexvar_epi16(perm_idx, _c01));
        c0 = _mm256_loadu_si256((const __m256i*)&LIBXSMM_VLA_ACCESS(2, dp_, _j, _k, bk));
        c1 = _mm256_loadu_si256((const __m256i*)&LIBXSMM_VLA_ACCESS(2, dp_, _j+1, _k, bk));
        _c01 = _mm512_inserti64x4 (LIBXSMM_INTRINSICS_MM512_UNDEFINED_EPI32(), c0, 0);
        _c01 = _mm512_inserti64x4 (_c01, c1, 1);
        _mm512_store_epi32(&LIBXSMM_VLA_ACCESS(3, dpB_, _j/_lpb, _k, 0, bk, _lpb), _mm512_permutexvar_epi16(perm_idx, _c01));
        c0 = _mm256_loadu_si256((const __m256i*)&LIBXSMM_VLA_ACCESS(2, dci_, _j, _k, bk));
        c1 = _mm256_loadu_si256((const __m256i*)&LIBXSMM_VLA_ACCESS(2, dci_, _j+1, _k, bk));
        _c01 = _mm512_inserti64x4 (LIBXSMM_INTRINSICS_MM512_UNDEFINED_EPI32(), c0, 0);
        _c01 = _mm512_inserti64x4 (_c01, c1, 1);
        _mm512_store_epi32(&LIBXSMM_VLA_ACCESS(3, dciB_, _j/_lpb, _k, 0, bk, _lpb), _mm512_permutexvar_epi16(perm_idx, _c01));
      }
    }
  }
}

#undef NATIVE_STORECVT_F32_BF16

