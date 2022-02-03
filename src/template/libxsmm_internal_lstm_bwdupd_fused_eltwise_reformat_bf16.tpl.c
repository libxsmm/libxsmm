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
{
  float* _dout              = &LIBXSMM_VLA_ACCESS(2, dout, in, ik, K);
  element_input_type* _dh   = &LIBXSMM_VLA_ACCESS(3, dh, j, in, ik, N, K);
  element_input_type* _o    = &LIBXSMM_VLA_ACCESS(3, o, j, in, ik, N, K);
  element_input_type* _co   = &LIBXSMM_VLA_ACCESS(3, co, j, in, ik, N, K);
  element_input_type* _dcs  = &LIBXSMM_VLA_ACCESS(2, dcs, in, ik, K);
  element_input_type* _ii   = &LIBXSMM_VLA_ACCESS(3, i, j, in, ik, N, K);
  element_input_type* _ci   = &LIBXSMM_VLA_ACCESS(3, ci, j, in, ik, N, K);
  element_input_type* _dci  = &LIBXSMM_VLA_ACCESS(2, dci, in, ik, K);
  element_input_type* _di   = &LIBXSMM_VLA_ACCESS(2, di, in, ik, K);
  element_input_type* _cps  = cps_ptr;
  element_input_type* _f    = &LIBXSMM_VLA_ACCESS(3, f, j, in, ik, N, K);
  element_input_type* _df   = &LIBXSMM_VLA_ACCESS(2, df, in, ik, K);
  element_input_type* _dp   = &LIBXSMM_VLA_ACCESS(2, dp, in, ik, K);
  element_input_type* _dcp  = &LIBXSMM_VLA_ACCESS(2, dcp, in, ik, K);
  element_input_type* _dciB = &LIBXSMM_VLA_ACCESS(5, dciB, ikb, inb, 0, 0, 0, nBlocks, bn_lp, bk, lpb);
  element_input_type* _diB  = &LIBXSMM_VLA_ACCESS(5, diB, ikb, inb, 0, 0, 0, nBlocks, bn_lp, bk, lpb);
  element_input_type* _dfB  = &LIBXSMM_VLA_ACCESS(5, dfB, ikb, inb, 0, 0, 0, nBlocks, bn_lp, bk, lpb);
  element_input_type* _dpB  = &LIBXSMM_VLA_ACCESS(5, dpB, ikb, inb, 0, 0, 0, nBlocks, bn_lp, bk, lpb);

  libxsmm_blasint _k, _j;
  __m512 _vdout, _vdh, _vo, _vt1, _vt2, _vco, _vdcs, _vdcp, _vii, _vci, _vdci, _vdi, _vcps, _vf, _vdf, _vdp;
  const __m512 _neg_ones = _mm512_set1_ps( (float)-1.0 );
  const __m512 _ones = _mm512_set1_ps( (float)1.0 );
  int _lpb = 2;

  if (j == t-1) {
    for ( _j = 0; _j < bn; ++_j ) {
      LIBXSMM_PRAGMA_UNROLL_N(4)
      for ( _k = 0; _k < bk; _k += 16 ) {
        _vdout = LIBXSMM_INTRINSICS_MM512_CVTPBH_PS(_mm256_loadu_si256((__m256i*)&_dh[(_j*K)+_k] ));
        _vo = LIBXSMM_INTRINSICS_MM512_CVTPBH_PS(_mm256_loadu_si256((__m256i*)&_o[(_j*K)+_k] ));
        _vt1 = _mm512_mul_ps( _vdout, _vo  );
        _vco = LIBXSMM_INTRINSICS_MM512_CVTPBH_PS(_mm256_loadu_si256((__m256i*)&_co[(_j*K)+_k] ));
        _vt2 = _mm512_fnmsub_ps ( _vco, _vco, _neg_ones);
        _vt1 = _mm512_mul_ps( _vt1, _vt2 );
        _vdcs = LIBXSMM_INTRINSICS_MM512_CVTPBH_PS(_mm256_loadu_si256((__m256i*)&_dcs[(_j*K)+_k] ));
        _vdcp = _mm512_add_ps( _vdcs, _vt1 );
        _vii = LIBXSMM_INTRINSICS_MM512_CVTPBH_PS(_mm256_loadu_si256((__m256i*)&_ii[(_j*K)+_k] ));
        _vt1 = _mm512_mul_ps( _vii, _vdcp );
        _vci = LIBXSMM_INTRINSICS_MM512_CVTPBH_PS(_mm256_loadu_si256((__m256i*)&_ci[(_j*K)+_k] ));
        _vt2 = _mm512_fnmsub_ps ( _vci, _vci, _neg_ones);
        _vdci = _mm512_mul_ps( _vt1, _vt2 );
        _mm256_stream_si256((__m256i*)&_dci[(_j*K)+_k], LIBXSMM_INTRINSISCS_MM512_CVTNEPS_PBH(_vdci) );
        _vt1 = _mm512_mul_ps( _vci, _vdcp );
        _vt2 = _mm512_sub_ps( _ones, _vii );
        _vdi = _mm512_mul_ps( _vii, _vt2);
        _vdi = _mm512_mul_ps( _vdi, _vt1);
        _mm256_stream_si256((__m256i*)&_di[(_j*K)+_k], LIBXSMM_INTRINSISCS_MM512_CVTNEPS_PBH(_vdi) );
        _vcps = LIBXSMM_INTRINSICS_MM512_CVTPBH_PS(_mm256_loadu_si256((__m256i*)&_cps[(_j*K)+_k] ));
        _vt1 = _mm512_mul_ps( _vcps, _vdcp );
        _vf = LIBXSMM_INTRINSICS_MM512_CVTPBH_PS(_mm256_loadu_si256((__m256i*)&_f[(_j*K)+_k] ));
        _vt2 = _mm512_sub_ps( _ones, _vf );
        _vdf = _mm512_mul_ps( _vf, _vt2);
        _vdf = _mm512_mul_ps( _vdf, _vt1);
        _mm256_stream_si256((__m256i*)&_df[(_j*K)+_k], LIBXSMM_INTRINSISCS_MM512_CVTNEPS_PBH(_vdf) );
        _vt1 = _mm512_mul_ps( _vdout, _vco);
        _vt2 = _mm512_sub_ps( _ones, _vo );
        _vt2 = _mm512_mul_ps( _vo, _vt2);
        _vdp = _mm512_mul_ps( _vt1, _vt2 );
        _mm256_stream_si256((__m256i*)&_dp[(_j*K)+_k], LIBXSMM_INTRINSISCS_MM512_CVTNEPS_PBH(_vdp) );
        _vdcp = _mm512_mul_ps( _vdcp, _vf);
        _mm256_stream_si256((__m256i*)&_dcp[(_j*K)+_k], LIBXSMM_INTRINSISCS_MM512_CVTNEPS_PBH(_vdcp) );
      }
    }
  } else {
    for ( _j = 0; _j < bn; ++_j ) {
      LIBXSMM_PRAGMA_UNROLL_N(4)
      for ( _k = 0; _k < bk; _k += 16 ) {
        _vdout = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &_dout[(_j*K)+_k] );
        _vdh = LIBXSMM_INTRINSICS_MM512_CVTPBH_PS(_mm256_loadu_si256((__m256i*)&_dh[(_j*K)+_k] ));
        _vdout = _mm512_add_ps( _vdout, _vdh );
        _vo = LIBXSMM_INTRINSICS_MM512_CVTPBH_PS(_mm256_loadu_si256((__m256i*)&_o[(_j*K)+_k] ));
        _vt1 = _mm512_mul_ps( _vdout, _vo  );
        _vco = LIBXSMM_INTRINSICS_MM512_CVTPBH_PS(_mm256_loadu_si256((__m256i*)&_co[(_j*K)+_k] ));
        _vt2 = _mm512_fnmsub_ps ( _vco, _vco, _neg_ones);
        _vt1 = _mm512_mul_ps( _vt1, _vt2 );
        _vdcp = LIBXSMM_INTRINSICS_MM512_CVTPBH_PS(_mm256_loadu_si256((__m256i*)&_dcp[(_j*K)+_k] ));
        _vdcp = _mm512_add_ps( _vdcp, _vt1 );
        _vii = LIBXSMM_INTRINSICS_MM512_CVTPBH_PS(_mm256_loadu_si256((__m256i*)&_ii[(_j*K)+_k] ));
        _vt1 = _mm512_mul_ps( _vii, _vdcp );
        _vci = LIBXSMM_INTRINSICS_MM512_CVTPBH_PS(_mm256_loadu_si256((__m256i*)&_ci[(_j*K)+_k] ));
        _vt2 = _mm512_fnmsub_ps ( _vci, _vci, _neg_ones);
        _vdci = _mm512_mul_ps( _vt1, _vt2 );
        _mm256_stream_si256((__m256i*)&_dci[(_j*K)+_k], LIBXSMM_INTRINSISCS_MM512_CVTNEPS_PBH(_vdci) );
        _vt1 = _mm512_mul_ps( _vci, _vdcp );
        _vt2 = _mm512_sub_ps( _ones, _vii );
        _vdi = _mm512_mul_ps( _vii, _vt2);
        _vdi = _mm512_mul_ps( _vdi, _vt1);
        _mm256_stream_si256((__m256i*)&_di[(_j*K)+_k], LIBXSMM_INTRINSISCS_MM512_CVTNEPS_PBH(_vdi) );
        _vcps = LIBXSMM_INTRINSICS_MM512_CVTPBH_PS(_mm256_loadu_si256((__m256i*)&_cps[(_j*K)+_k] ));
        _vt1 = _mm512_mul_ps( _vcps, _vdcp );
        _vf = LIBXSMM_INTRINSICS_MM512_CVTPBH_PS(_mm256_loadu_si256((__m256i*)&_f[(_j*K)+_k] ));
        _vt2 = _mm512_sub_ps( _ones, _vf );
        _vdf = _mm512_mul_ps( _vf, _vt2);
        _vdf = _mm512_mul_ps( _vdf, _vt1);
        _mm256_stream_si256((__m256i*)&_df[(_j*K)+_k], LIBXSMM_INTRINSISCS_MM512_CVTNEPS_PBH(_vdf) );
        _vt1 = _mm512_mul_ps( _vdout, _vco);
        _vt2 = _mm512_sub_ps( _ones, _vo );
        _vt2 = _mm512_mul_ps( _vo, _vt2);
        _vdp = _mm512_mul_ps( _vt1, _vt2 );
        _mm256_stream_si256((__m256i*)&_dp[(_j*K)+_k], LIBXSMM_INTRINSISCS_MM512_CVTNEPS_PBH(_vdp) );
        _vdcp = _mm512_mul_ps( _vdcp, _vf);
        _mm256_stream_si256((__m256i*)&_dcp[(_j*K)+_k], LIBXSMM_INTRINSISCS_MM512_CVTNEPS_PBH(_vdcp) );
      }
    }
  }

  { /* Store di/dci/df/dp to diB/dciB/dfB/dpB which is CNNC AND vnni format */
    LIBXSMM_VLA_DECL(2, libxsmm_bfloat16, di_, _di, K);
    LIBXSMM_VLA_DECL(2, libxsmm_bfloat16, df_, _df, K);
    LIBXSMM_VLA_DECL(2, libxsmm_bfloat16, dp_, _dp, K);
    LIBXSMM_VLA_DECL(2, libxsmm_bfloat16, dci_, _dci, K);
    LIBXSMM_VLA_DECL(3, libxsmm_bfloat16, diB_, _diB, bk, _lpb);
    LIBXSMM_VLA_DECL(3, libxsmm_bfloat16, dfB_, _dfB, bk, _lpb);
    LIBXSMM_VLA_DECL(3, libxsmm_bfloat16, dpB_, _dpB, bk, _lpb);
    LIBXSMM_VLA_DECL(3, libxsmm_bfloat16, dciB_, _dciB, bk, _lpb);
    if ( (bn % 2 == 0) && (bk % 16 == 0) ) {
      const __m512i perm_idx = LIBXSMM_INTRINSICS_MM512_SET_EPI16(31, 15, 30, 14, 29, 13, 28, 12, 27, 11, 26, 10, 25, 9, 24, 8, 23, 7, 22, 6, 21, 5, 20, 4, 19, 3, 18, 2, 17, 1, 16, 0);
      __m256i c0, c1;
      __m512i c01;
      for (_j = 0; _j < bn; _j+=2) {
        for (_k = 0; _k < bk; _k+=16) {
          c0 = _mm256_loadu_si256((const __m256i*)&LIBXSMM_VLA_ACCESS(2, di_, _j, _k, K));
          c1 = _mm256_loadu_si256((const __m256i*)&LIBXSMM_VLA_ACCESS(2, di_, _j+1, _k, K));
          c01 = _mm512_inserti64x4 (LIBXSMM_INTRINSICS_MM512_UNDEFINED_EPI32(), c0, 0);
          c01 = _mm512_inserti64x4 (c01, c1, 1);
          _mm512_storeu_si512(&LIBXSMM_VLA_ACCESS(3, diB_, _j/_lpb, _k, 0, bk, _lpb), _mm512_permutexvar_epi16(perm_idx, c01));
          c0 = _mm256_loadu_si256((const __m256i*)&LIBXSMM_VLA_ACCESS(2, df_, _j, _k, K));
          c1 = _mm256_loadu_si256((const __m256i*)&LIBXSMM_VLA_ACCESS(2, df_, _j+1, _k, K));
          c01 = _mm512_inserti64x4 (LIBXSMM_INTRINSICS_MM512_UNDEFINED_EPI32(), c0, 0);
          c01 = _mm512_inserti64x4 (c01, c1, 1);
          _mm512_storeu_si512(&LIBXSMM_VLA_ACCESS(3, dfB_, _j/_lpb, _k, 0, bk, _lpb), _mm512_permutexvar_epi16(perm_idx, c01));
          c0 = _mm256_loadu_si256((const __m256i*)&LIBXSMM_VLA_ACCESS(2, dp_, _j, _k, K));
          c1 = _mm256_loadu_si256((const __m256i*)&LIBXSMM_VLA_ACCESS(2, dp_, _j+1, _k, K));
          c01 = _mm512_inserti64x4 (LIBXSMM_INTRINSICS_MM512_UNDEFINED_EPI32(), c0, 0);
          c01 = _mm512_inserti64x4 (c01, c1, 1);
          _mm512_storeu_si512(&LIBXSMM_VLA_ACCESS(3, dpB_, _j/_lpb, _k, 0, bk, _lpb), _mm512_permutexvar_epi16(perm_idx, c01));
          c0 = _mm256_loadu_si256((const __m256i*)&LIBXSMM_VLA_ACCESS(2, dci_, _j, _k, K));
          c1 = _mm256_loadu_si256((const __m256i*)&LIBXSMM_VLA_ACCESS(2, dci_, _j+1, _k, K));
          c01 = _mm512_inserti64x4 (LIBXSMM_INTRINSICS_MM512_UNDEFINED_EPI32(), c0, 0);
          c01 = _mm512_inserti64x4 (c01, c1, 1);
          _mm512_storeu_si512(&LIBXSMM_VLA_ACCESS(3, dciB_, _j/_lpb, _k, 0, bk, _lpb), _mm512_permutexvar_epi16(perm_idx, c01));
        }
      }
    } else {
      for (_j = 0; _j < bn; _j++) {
        for (_k = 0; _k < bk; _k++) {
          LIBXSMM_VLA_ACCESS(3, diB_, _j / _lpb, _k, _j%_lpb, bk, _lpb) = LIBXSMM_VLA_ACCESS(2, di_, _j, _k, K);
          LIBXSMM_VLA_ACCESS(3, dfB_, _j / _lpb, _k, _j%_lpb, bk, _lpb) = LIBXSMM_VLA_ACCESS(2, df_, _j, _k, K);
          LIBXSMM_VLA_ACCESS(3, dpB_, _j / _lpb, _k, _j%_lpb, bk, _lpb) = LIBXSMM_VLA_ACCESS(2, dp_, _j, _k, K);
          LIBXSMM_VLA_ACCESS(3, dciB_, _j / _lpb, _k, _j%_lpb, bk, _lpb) = LIBXSMM_VLA_ACCESS(2, dci_, _j, _k, K);
        }
      }
    }
  }
}


