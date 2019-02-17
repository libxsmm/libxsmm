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
/* Evangelos Georganas (Intel Corp.), Alexander Heinecke (Intel Corp.)
******************************************************************************/

{
  libxsmm_blasint _k, _j;
  element_input_type* _o = &LIBXSMM_VLA_ACCESS(3, o, j, in, ik, N, K);
  element_input_type* _i = &LIBXSMM_VLA_ACCESS(3, i, j, in, ik, N, K);
  element_input_type* _f = &LIBXSMM_VLA_ACCESS(3, f, j, in, ik, N, K);
  element_input_type* _ci = &LIBXSMM_VLA_ACCESS(3, ci, j, in, ik, N, K);
  element_input_type* _cps = cps_ptr;
  element_input_type* _cs = &LIBXSMM_VLA_ACCESS(3, cs, j, in, ik, N, K);
  element_input_type* _h = &LIBXSMM_VLA_ACCESS(3, h, j, in, ik, N, K);
  element_input_type* _co = &LIBXSMM_VLA_ACCESS(3, co, j, in, ik, N, K);
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
      _vo = _mm512_fmadd_ps( LIBXSMM_INTRINSICS_MM512_TANH_PS( _mm512_mul_ps( _vo, _halves ) ), _halves, _halves);
      _vi = _mm512_fmadd_ps( LIBXSMM_INTRINSICS_MM512_TANH_PS( _mm512_mul_ps( _vi, _halves ) ), _halves, _halves);
      _vci = LIBXSMM_INTRINSICS_MM512_TANH_PS( _vci );
      _vf = _mm512_fmadd_ps( LIBXSMM_INTRINSICS_MM512_TANH_PS( _mm512_mul_ps( _vf, _halves ) ), _halves, _halves);
      _vcs = _mm512_mul_ps( _vf, _vcs );
      _vcs = _mm512_fmadd_ps( _vi, _vci, _vcs );
      _vco = LIBXSMM_INTRINSICS_MM512_TANH_PS( _vcs );
      _vh = _mm512_mul_ps( _vo, _vco );
      _mm512_store_ps( &_o[(_j*K)+_k], _vo );
      _mm512_store_ps( &_i[(_j*K)+_k], _vi );
      _mm512_store_ps( &_ci[(_j*K)+_k], _vci );
      _mm512_store_ps( &_f[(_j*K)+_k], _vf );
      _mm512_store_ps( &_cs[(_j*K)+_k], _vcs );
      _mm512_store_ps( &_co[(_j*K)+_k], _vco );
      LIBXSMM_INTRINSICS_MM512_STREAM_PS( &_h[(_j*K)+_k], _vh );
    }
  }
}

