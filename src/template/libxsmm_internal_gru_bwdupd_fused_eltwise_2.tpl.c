/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Kunal Banerjee (Intel Corp.)
******************************************************************************/

{
  libxsmm_blasint _k, _j;
  __m512 _vdi, _vdo, _vi, _vhp, _vt1, _vt2;
  element_input_type* _hp;
  element_input_type* _i = &LIBXSMM_VLA_ACCESS(3, i, j, in, ik, N, K);
  element_input_type* _di = &LIBXSMM_VLA_ACCESS(2, di, in, ik, K);
  element_input_type* _do = &LIBXSMM_VLA_ACCESS(2, dp, in, ik, K);
  const __m512 _vones = _mm512_set1_ps( (float)1.0 );
  if (0 == j) {
    _hp = &LIBXSMM_VLA_ACCESS(2, hp, in, ik, K);
  } else {
    _hp = &LIBXSMM_VLA_ACCESS(3, h, j-1, in, ik, N, K);
  }
  for ( _j = 0; _j < bn; ++_j ) {
    LIBXSMM_PRAGMA_UNROLL_N(4)
    for ( _k = 0; _k < bk; _k += 16 ) {
      _vi = LIBXSMM_INTRINSICS_MM512_LOAD_PS(&_i[(_j*K)+_k]);
      _vt1 = _mm512_sub_ps(_vones, _vi);
      _vt1 = _mm512_mul_ps(_vi, _vt1);
      _vhp = LIBXSMM_INTRINSICS_MM512_LOAD_PS(&_hp[(_j*K)+_k]);
      _vdo = LIBXSMM_INTRINSICS_MM512_LOAD_PS(&_do[(_j*K)+_k]);
      _vt2 = _mm512_mul_ps(_vdo, _vhp);
      _vdi = _mm512_mul_ps(_vt1, _vt2);
      LIBXSMM_INTRINSICS_MM512_STREAM_PS(&_di[(_j*K)+_k], _vdi);
    }
  }
}
