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
  __m512 _vdh, _vdi, _vdo, _vi, _vhp, _vt1, _vt2;
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
