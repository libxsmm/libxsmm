/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Evangelos Georganas (Intel Corp.), Alexander Heinecke (Intel Corp.)
******************************************************************************/

{
  libxsmm_blasint _k, _j;
  float* _o = &LIBXSMM_VLA_ACCESS(3, o, j, in, ik, N, K);
  float* _i = &LIBXSMM_VLA_ACCESS(3, i, j, in, ik, N, K);
  float* _f = &LIBXSMM_VLA_ACCESS(3, f, j, in, ik, N, K);
  float* _ci = &LIBXSMM_VLA_ACCESS(3, ci, j, in, ik, N, K);
  float* _cps = cps_ptr;
  float* _cs = &LIBXSMM_VLA_ACCESS(3, cs, j, in, ik, N, K);
  float* _h = &LIBXSMM_VLA_ACCESS(3, h, j, in, ik, N, K);
  float* _co = &LIBXSMM_VLA_ACCESS(3, co, j, in, ik, N, K);
  __m512 _vf, _vcs, _vi, _vci, _vco, _vo, _vh;
  const __m512 _halves = _mm512_set1_ps( (LIBXSMM_DNN_ELTWISE_FTYPE)0.5 );
  for ( _j = 0; _j < bn; ++_j ) {
    LIBXSMM_PRAGMA_UNROLL_N(4)
    for ( _k = 0; _k < bk; _k += 16 ) {
      _vo = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &_o[(_j*K)+_k] );
      _vi = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &_i[(_j*K)+_k] );
      _vci = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &_ci[(_j*K)+_k] );
      _vf = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &_f[(_j*K)+_k] );
      _vcs = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &_cps[(_j*K)+_k] );
      _vo = _mm512_fmadd_ps( LIBXSMM_INTRINSICS_MM512_TANH_PS_MINIMAX2( _mm512_mul_ps( _vo, _halves ) ), _halves, _halves);
      _vi = _mm512_fmadd_ps( LIBXSMM_INTRINSICS_MM512_TANH_PS_MINIMAX2( _mm512_mul_ps( _vi, _halves ) ), _halves, _halves);
      _vci = LIBXSMM_INTRINSICS_MM512_TANH_PS_MINIMAX2( _vci );
      _vf = _mm512_fmadd_ps( LIBXSMM_INTRINSICS_MM512_TANH_PS_MINIMAX2( _mm512_mul_ps( _vf, _halves ) ), _halves, _halves);
      _vcs = _mm512_mul_ps( _vf, _vcs );
      _vcs = _mm512_fmadd_ps( _vi, _vci, _vcs );
      _vco = LIBXSMM_INTRINSICS_MM512_TANH_PS_MINIMAX2( _vcs );
      _vh = _mm512_mul_ps( _vo, _vco );
      _mm512_storeu_ps( &_o[(_j*K)+_k], _vo );
      _mm512_storeu_ps( &_i[(_j*K)+_k], _vi );
      _mm512_storeu_ps( &_ci[(_j*K)+_k], _vci );
      _mm512_storeu_ps( &_f[(_j*K)+_k], _vf );
      _mm512_storeu_ps( &_cs[(_j*K)+_k], _vcs );
      _mm512_storeu_ps( &_co[(_j*K)+_k], _vco );
      LIBXSMM_INTRINSICS_MM512_STREAM_PS( &_h[(_j*K)+_k], _vh );
    }
  }
}

