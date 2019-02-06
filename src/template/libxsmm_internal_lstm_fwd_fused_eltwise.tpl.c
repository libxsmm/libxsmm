#ifndef LSTM_FWD_ELTWISE
#define LSTM_FWD_ELTWISE

static inline __m512 _mm512_tanh_generic_ps( __m512 x ) {
  int i;
  LIBXSMM_DNN_ELTWISE_FTYPE _x[16];
  _mm512_store_ps( _x, x );
  LIBXSMM_PRAGMA_SIMD
  for (i = 0; i < 16; i++) {
    _x[i] = (LIBXSMM_DNN_ELTWISE_FTYPE) tanh((double) _x[i] );
  }
  __m512 result = _mm512_loadu_ps( _x );
  return result;
}

#if defined(LIBXSMM_INTEL_COMPILER)
#define _MM512_TANH_PS(A) _mm512_tanh_ps(A)
#else
#define _MM512_TANH_PS(A) _mm512_tanh_generic_ps(A)
#endif

static inline void libxsmm_internal_compute_o_i_f_ci_cs_co_h_ld(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld, LIBXSMM_DNN_ELTWISE_FTYPE *f, LIBXSMM_DNN_ELTWISE_FTYPE *cps, LIBXSMM_DNN_ELTWISE_FTYPE *cs, LIBXSMM_DNN_ELTWISE_FTYPE *ii, LIBXSMM_DNN_ELTWISE_FTYPE *ci,LIBXSMM_DNN_ELTWISE_FTYPE *co, LIBXSMM_DNN_ELTWISE_FTYPE *o, LIBXSMM_DNN_ELTWISE_FTYPE *h) {
  libxsmm_blasint i, j;
  __m512 _f, _cps, _cs, _ii, _ci, _co, _o, _h;
  const __m512 _halves = _mm512_set1_ps( (LIBXSMM_DNN_ELTWISE_FTYPE)0.5 );
  for ( j = 0; j < n; ++j ) {
    LIBXSMM_PRAGMA_UNROLL_N(4)
      for ( i = 0; i < m; i += 16 ) {
        _o = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &o[(j*ld)+i] );
        _ii = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &ii[(j*ld)+i] );
        _ci = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &ci[(j*ld)+i] );
        _f = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &f[(j*ld)+i] );
        _cps = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &cps[(j*ld)+i] );
        _o = _mm512_fmadd_ps( _MM512_TANH_PS( _mm512_mul_ps( _o, _halves ) ), _halves, _halves);
        _ii = _mm512_fmadd_ps( _MM512_TANH_PS( _mm512_mul_ps( _ii, _halves ) ), _halves, _halves);
        _ci = _MM512_TANH_PS( _ci );
        _f = _mm512_fmadd_ps( _MM512_TANH_PS( _mm512_mul_ps( _f, _halves ) ), _halves, _halves);
        _cs = _mm512_mul_ps( _f, _cps );
        _cs = _mm512_fmadd_ps( _ii, _ci, _cs );
        _co = _MM512_TANH_PS( _cs );
        _h = _mm512_mul_ps( _o, _co );
        _mm512_store_ps( &o[(j*ld)+i], _o );
        _mm512_store_ps( &ii[(j*ld)+i], _ii );
        _mm512_store_ps( &ci[(j*ld)+i], _ci );
        _mm512_store_ps( &f[(j*ld)+i], _f );
        _mm512_store_ps( &cs[(j*ld)+i], _cs );
        _mm512_store_ps( &co[(j*ld)+i], _co );
        LIBXSMM_INTRINSICS_MM512_STREAM_PS( &h[(j*ld)+i], _h );
      }
  }
}
#undef _MM512_TANH_PS
#endif

