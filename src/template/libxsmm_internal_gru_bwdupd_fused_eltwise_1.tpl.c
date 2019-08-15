/******************************************************************************
** Copyright (c) 2016-2019, Intel Corporation                                **
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
/* Kunal Banerjee (Intel Corp.)
******************************************************************************/

{
  libxsmm_blasint _k, _j;
  __m512 _vdh, _vdout, _vdf, _vdc, _vf, _vc, _vhp, _vt1, _vt2;
  element_input_type* _dout = &LIBXSMM_VLA_ACCESS(2, dout, in, ik, K);
  element_input_type* _hp;
  element_input_type* _c = &LIBXSMM_VLA_ACCESS(3, c, j, in, ik, N, K);
  element_input_type* _f = &LIBXSMM_VLA_ACCESS(3, f, j, in, ik, N, K);
  element_input_type* _dh = &LIBXSMM_VLA_ACCESS(3, dh, j, in, ik, N, K);
  element_input_type* _dc = &LIBXSMM_VLA_ACCESS(2, dc, in, ik, K);
  element_input_type* _df = &LIBXSMM_VLA_ACCESS(2, df, in, ik, K);
  const __m512 _vneg_ones = _mm512_set1_ps( (float)-1.0 );
  const __m512 _vones = _mm512_set1_ps( (float)1.0 );
  if (0 == j) {
    _hp = &LIBXSMM_VLA_ACCESS(2, hp, in, ik, K);
  } else {
    _hp = &LIBXSMM_VLA_ACCESS(3, h, j-1, in, ik, N, K);
  }
  if (j == t-1) {
    for ( _j = 0; _j < bn; ++_j ) {
      LIBXSMM_PRAGMA_UNROLL_N(4)
      for ( _k = 0; _k < bk; _k += 16 ) {
        _vdout = LIBXSMM_INTRINSICS_MM512_LOAD_PS(&_dh[(_j*K)+_k]);
        LIBXSMM_INTRINSICS_MM512_STREAM_PS(&_dout[(_j*K)+_k], _vdout);
        _vc = LIBXSMM_INTRINSICS_MM512_LOAD_PS(&_c[(_j*K)+_k]);
        _vt1 = _mm512_sub_ps(_vones, _vc);
        _vt1 = _mm512_mul_ps(_vdout, _vt1);
        _vf = LIBXSMM_INTRINSICS_MM512_LOAD_PS(&_f[(_j*K)+_k]);
        _vt2 = _mm512_fnmsub_ps(_vf, _vf, _vneg_ones);
        _vdf = _mm512_mul_ps(_vt1, _vt2);
        LIBXSMM_INTRINSICS_MM512_STREAM_PS(&_df[(_j*K)+_k], _vdf);
        _vhp = LIBXSMM_INTRINSICS_MM512_LOAD_PS(&_hp[(_j*K)+_k]);
        _vt1 = _mm512_mul_ps(_vt1, _vc);
        _vt2 = _mm512_sub_ps(_vhp, _vf);
        _vdc = _mm512_mul_ps(_vt1, _vt2);
        LIBXSMM_INTRINSICS_MM512_STREAM_PS(&_dc[(_j*K)+_k], _vdc);
      }
    }
  } else {
    for ( _j = 0; _j < bn; ++_j ) {
      LIBXSMM_PRAGMA_UNROLL_N(4)
      for ( _k = 0; _k < bk; _k += 16 ) {
        _vdout = LIBXSMM_INTRINSICS_MM512_LOAD_PS(&_dout[(_j*K)+_k]);
        _vdh = LIBXSMM_INTRINSICS_MM512_LOAD_PS(&_dh[(_j*K)+_k]);
        _vdout = _mm512_add_ps(_vdout, _vdh);
        LIBXSMM_INTRINSICS_MM512_STREAM_PS(&_dout[(_j*K)+_k], _vdout);
        _vc = LIBXSMM_INTRINSICS_MM512_LOAD_PS(&_c[(_j*K)+_k]);
        _vt1 = _mm512_sub_ps(_vones, _vc);
        _vt1 = _mm512_mul_ps(_vdout, _vt1);
        _vf = LIBXSMM_INTRINSICS_MM512_LOAD_PS(&_f[(_j*K)+_k]);
        _vt2 = _mm512_fnmsub_ps(_vf, _vf, _vneg_ones);
        _vdf = _mm512_mul_ps( _vt1, _vt2 );
        LIBXSMM_INTRINSICS_MM512_STREAM_PS(&_df[(_j*K)+_k], _vdf);
        _vhp = LIBXSMM_INTRINSICS_MM512_LOAD_PS(&_hp[(_j*K)+_k]);
        _vt1 = _mm512_mul_ps(_vt1, _vc);
        _vt2 = _mm512_sub_ps(_vhp, _vf);
        _vdc = _mm512_mul_ps( _vt1, _vt2 );
        LIBXSMM_INTRINSICS_MM512_STREAM_PS(&_dc[(_j*K)+_k], _vdc);
      }
    }
  }
}
