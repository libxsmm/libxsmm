#ifndef LSTM_BWD_ELTWISE
#define LSTM_BWD_ELTWISE

static inline void libxsmm_internal_compute_dcp_dci_di_df_dp_ld(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld, int timestep, int t, LIBXSMM_DNN_ELTWISE_FTYPE *dout, LIBXSMM_DNN_ELTWISE_FTYPE *dh, LIBXSMM_DNN_ELTWISE_FTYPE *o, LIBXSMM_DNN_ELTWISE_FTYPE *co, LIBXSMM_DNN_ELTWISE_FTYPE *dcs, LIBXSMM_DNN_ELTWISE_FTYPE *ii, LIBXSMM_DNN_ELTWISE_FTYPE *ci, LIBXSMM_DNN_ELTWISE_FTYPE *dci, LIBXSMM_DNN_ELTWISE_FTYPE *di, LIBXSMM_DNN_ELTWISE_FTYPE *cps, LIBXSMM_DNN_ELTWISE_FTYPE *f, LIBXSMM_DNN_ELTWISE_FTYPE *df, LIBXSMM_DNN_ELTWISE_FTYPE *dp, LIBXSMM_DNN_ELTWISE_FTYPE *dcp) {
  libxsmm_blasint i, j;
  __m512 _dout, _dh, _o, _t1, _t2, _co, _dcs, _dcp, _ii, _ci, _dci, _di, _cps, _f, _df, _dp;
  const __m512 _neg_ones = _mm512_set1_ps( (float)-1.0 );
  const __m512 _ones = _mm512_set1_ps( (float)1.0 );
  if (timestep == t-1) {
    for ( j = 0; j < n; ++j ) {
      LIBXSMM_PRAGMA_UNROLL_N(4)
      for ( i = 0; i < m; i += 16 ) {
        _dout = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &dh[(j*ld)+i] );
        _o = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &o[(j*ld)+i] );
        _t1 = _mm512_mul_ps( _dout, _o  );
        _co = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &co[(j*ld)+i] );
        _t2 = _mm512_fnmsub_ps ( _co, _co, _neg_ones);
        _t1 = _mm512_mul_ps( _t1, _t2 );
        _dcs = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &dcs[(j*ld)+i] );
        _dcp = _mm512_add_ps( _dcs, _t1 );
        _ii = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &ii[(j*ld)+i] );
        _t1 = _mm512_mul_ps( _ii, _dcp );
        _ci = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &ci[(j*ld)+i] );
        _t2 = _mm512_fnmsub_ps ( _ci, _ci, _neg_ones);
        _dci = _mm512_mul_ps( _t1, _t2 );
        LIBXSMM_INTRINSICS_MM512_STREAM_PS( &dci[(j*ld)+i], _dci );
        _t1 = _mm512_mul_ps( _ci, _dcp );
        _t2 = _mm512_sub_ps( _ones, _ii );
        _di = _mm512_mul_ps( _ii, _t2);
        _di = _mm512_mul_ps( _di, _t1);
        LIBXSMM_INTRINSICS_MM512_STREAM_PS( &di[(j*ld)+i], _di );
        _cps = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &cps[(j*ld)+i] );
        _t1 = _mm512_mul_ps( _cps, _dcp );
        _f = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &f[(j*ld)+i] );
        _t2 = _mm512_sub_ps( _ones, _f );
        _df = _mm512_mul_ps( _f, _t2);
        _df = _mm512_mul_ps( _df, _t1);
        LIBXSMM_INTRINSICS_MM512_STREAM_PS( &df[(j*ld)+i], _df );
        _t1 = _mm512_mul_ps( _dout, _co);
        _t2 = _mm512_sub_ps( _ones, _o );
        _t2 = _mm512_mul_ps( _o, _t2);
        _dp = _mm512_mul_ps( _t1, _t2 );
        LIBXSMM_INTRINSICS_MM512_STREAM_PS( &dp[(j*ld)+i], _dp );
        _dcp = _mm512_mul_ps( _dcp, _f);
        LIBXSMM_INTRINSICS_MM512_STREAM_PS( &dcp[(j*ld)+i], _dcp );
      }
    }
  } else {
    for ( j = 0; j < n; ++j ) {
       LIBXSMM_PRAGMA_UNROLL_N(4)
       for ( i = 0; i < m; i += 16 ) {
        _dout = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &dout[(j*ld)+i] );
        _dh = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &dh[(j*ld)+i] );
        _dout = _mm512_add_ps( _dout, _dh );
        _o = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &o[(j*ld)+i] );
        _t1 = _mm512_mul_ps( _dout, _o  );
        _co = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &co[(j*ld)+i] );
        _t2 = _mm512_fnmsub_ps ( _co, _co, _neg_ones);
        _t1 = _mm512_mul_ps( _t1, _t2 );
        _dcp = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &dcp[(j*ld)+i] );
        _dcp = _mm512_add_ps( _dcp, _t1 );
        _ii = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &ii[(j*ld)+i] );
        _t1 = _mm512_mul_ps( _ii, _dcp );
        _ci = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &ci[(j*ld)+i] );
        _t2 = _mm512_fnmsub_ps ( _ci, _ci, _neg_ones);
        _dci = _mm512_mul_ps( _t1, _t2 );
        LIBXSMM_INTRINSICS_MM512_STREAM_PS( &dci[(j*ld)+i], _dci );
        _t1 = _mm512_mul_ps( _ci, _dcp );
        _t2 = _mm512_sub_ps( _ones, _ii );
        _di = _mm512_mul_ps( _ii, _t2);
        _di = _mm512_mul_ps( _di, _t1);
        LIBXSMM_INTRINSICS_MM512_STREAM_PS( &di[(j*ld)+i], _di );
        _cps = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &cps[(j*ld)+i] );
        _t1 = _mm512_mul_ps( _cps, _dcp );
        _f = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &f[(j*ld)+i] );
        _t2 = _mm512_sub_ps( _ones, _f );
        _df = _mm512_mul_ps( _f, _t2);
        _df = _mm512_mul_ps( _df, _t1);
        LIBXSMM_INTRINSICS_MM512_STREAM_PS( &df[(j*ld)+i], _df );
        _t1 = _mm512_mul_ps( _dout, _co);
        _t2 = _mm512_sub_ps( _ones, _o );
        _t2 = _mm512_mul_ps( _o, _t2);
        _dp = _mm512_mul_ps( _t1, _t2 );
        LIBXSMM_INTRINSICS_MM512_STREAM_PS( &dp[(j*ld)+i], _dp );
        _dcp = _mm512_mul_ps( _dcp, _f);
        LIBXSMM_INTRINSICS_MM512_STREAM_PS( &dcp[(j*ld)+i], _dcp );
      }
    }
  }
}

static inline void libxsmm_internal_compute_dcp_dci_di_df_dp_ld_and_reformat_dci_di_df_dp_ld2(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld, libxsmm_blasint ld2, int timestep, int t, LIBXSMM_DNN_ELTWISE_FTYPE *dout, LIBXSMM_DNN_ELTWISE_FTYPE *dh, LIBXSMM_DNN_ELTWISE_FTYPE *o, LIBXSMM_DNN_ELTWISE_FTYPE *co, LIBXSMM_DNN_ELTWISE_FTYPE *dcs, LIBXSMM_DNN_ELTWISE_FTYPE *ii, LIBXSMM_DNN_ELTWISE_FTYPE *ci, LIBXSMM_DNN_ELTWISE_FTYPE *dci, LIBXSMM_DNN_ELTWISE_FTYPE *di, LIBXSMM_DNN_ELTWISE_FTYPE *cps, LIBXSMM_DNN_ELTWISE_FTYPE *f, LIBXSMM_DNN_ELTWISE_FTYPE *df, LIBXSMM_DNN_ELTWISE_FTYPE *dp, LIBXSMM_DNN_ELTWISE_FTYPE *dcp, LIBXSMM_DNN_ELTWISE_FTYPE *dciB, LIBXSMM_DNN_ELTWISE_FTYPE *diB, LIBXSMM_DNN_ELTWISE_FTYPE *dfB, LIBXSMM_DNN_ELTWISE_FTYPE *dpB) {
  libxsmm_blasint i, j;
  __m512 _dout, _dh, _o, _t1, _t2, _co, _dcs, _dcp, _ii, _ci, _dci, _di, _cps, _f, _df, _dp;
  const __m512 _neg_ones = _mm512_set1_ps( (float)-1.0 );
  const __m512 _ones = _mm512_set1_ps( (float)1.0 );

  if (timestep == t-1) {
    for ( j = 0; j < n; ++j ) {
      LIBXSMM_PRAGMA_UNROLL_N(4)
      for ( i = 0; i < m; i += 16 ) {
        _dout = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &dh[(j*ld)+i] );
        _o = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &o[(j*ld)+i] );
        _t1 = _mm512_mul_ps( _dout, _o  );
        _co = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &co[(j*ld)+i] );
        _t2 = _mm512_fnmsub_ps ( _co, _co, _neg_ones);
        _t1 = _mm512_mul_ps( _t1, _t2 );
        _dcs = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &dcs[(j*ld)+i] );
        _dcp = _mm512_add_ps( _dcs, _t1 );
        _ii = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &ii[(j*ld)+i] );
        _t1 = _mm512_mul_ps( _ii, _dcp );
        _ci = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &ci[(j*ld)+i] );
        _t2 = _mm512_fnmsub_ps ( _ci, _ci, _neg_ones);
        _dci = _mm512_mul_ps( _t1, _t2 );
        LIBXSMM_INTRINSICS_MM512_STREAM_PS( &dci[(j*ld)+i], _dci );
        LIBXSMM_INTRINSICS_MM512_STREAM_PS( &dciB[(j*ld2)+i], _dci );
        _t1 = _mm512_mul_ps( _ci, _dcp );
        _t2 = _mm512_sub_ps( _ones, _ii );
        _di = _mm512_mul_ps( _ii, _t2);
        _di = _mm512_mul_ps( _di, _t1);
        LIBXSMM_INTRINSICS_MM512_STREAM_PS( &di[(j*ld)+i], _di );
        LIBXSMM_INTRINSICS_MM512_STREAM_PS( &diB[(j*ld2)+i], _di );
        _cps = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &cps[(j*ld)+i] );
        _t1 = _mm512_mul_ps( _cps, _dcp );
        _f = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &f[(j*ld)+i] );
        _t2 = _mm512_sub_ps( _ones, _f );
        _df = _mm512_mul_ps( _f, _t2);
        _df = _mm512_mul_ps( _df, _t1);
        LIBXSMM_INTRINSICS_MM512_STREAM_PS( &df[(j*ld)+i], _df );
        LIBXSMM_INTRINSICS_MM512_STREAM_PS( &dfB[(j*ld2)+i], _df );
        _t1 = _mm512_mul_ps( _dout, _co);
        _t2 = _mm512_sub_ps( _ones, _o );
        _t2 = _mm512_mul_ps( _o, _t2);
        _dp = _mm512_mul_ps( _t1, _t2 );
        LIBXSMM_INTRINSICS_MM512_STREAM_PS( &dp[(j*ld)+i], _dp );
        LIBXSMM_INTRINSICS_MM512_STREAM_PS( &dpB[(j*ld2)+i], _dp );
        _dcp = _mm512_mul_ps( _dcp, _f);
        LIBXSMM_INTRINSICS_MM512_STREAM_PS( &dcp[(j*ld)+i], _dcp );
      }
    }
  } else {
    for ( j = 0; j < n; ++j ) {
       LIBXSMM_PRAGMA_UNROLL_N(4)
       for ( i = 0; i < m; i += 16 ) {
        _dout = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &dout[(j*ld)+i] );
        _dh = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &dh[(j*ld)+i] );
        _dout = _mm512_add_ps( _dout, _dh );
        _o = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &o[(j*ld)+i] );
        _t1 = _mm512_mul_ps( _dout, _o  );
        _co = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &co[(j*ld)+i] );
        _t2 = _mm512_fnmsub_ps ( _co, _co, _neg_ones);
        _t1 = _mm512_mul_ps( _t1, _t2 );
        _dcp = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &dcp[(j*ld)+i] );
        _dcp = _mm512_add_ps( _dcp, _t1 );
        _ii = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &ii[(j*ld)+i] );
        _t1 = _mm512_mul_ps( _ii, _dcp );
        _ci = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &ci[(j*ld)+i] );
        _t2 = _mm512_fnmsub_ps ( _ci, _ci, _neg_ones);
        _dci = _mm512_mul_ps( _t1, _t2 );
        LIBXSMM_INTRINSICS_MM512_STREAM_PS( &dci[(j*ld)+i], _dci );
        LIBXSMM_INTRINSICS_MM512_STREAM_PS( &dciB[(j*ld2)+i], _dci );
        _t1 = _mm512_mul_ps( _ci, _dcp );
        _t2 = _mm512_sub_ps( _ones, _ii );
        _di = _mm512_mul_ps( _ii, _t2);
        _di = _mm512_mul_ps( _di, _t1);
        LIBXSMM_INTRINSICS_MM512_STREAM_PS( &di[(j*ld)+i], _di );
        LIBXSMM_INTRINSICS_MM512_STREAM_PS( &diB[(j*ld2)+i], _di );
        _cps = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &cps[(j*ld)+i] );
        _t1 = _mm512_mul_ps( _cps, _dcp );
        _f = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &f[(j*ld)+i] );
        _t2 = _mm512_sub_ps( _ones, _f );
        _df = _mm512_mul_ps( _f, _t2);
        _df = _mm512_mul_ps( _df, _t1);
        LIBXSMM_INTRINSICS_MM512_STREAM_PS( &df[(j*ld)+i], _df );
        LIBXSMM_INTRINSICS_MM512_STREAM_PS( &dfB[(j*ld2)+i], _df );
        _t1 = _mm512_mul_ps( _dout, _co);
        _t2 = _mm512_sub_ps( _ones, _o );
        _t2 = _mm512_mul_ps( _o, _t2);
        _dp = _mm512_mul_ps( _t1, _t2 );
        LIBXSMM_INTRINSICS_MM512_STREAM_PS( &dp[(j*ld)+i], _dp );
        LIBXSMM_INTRINSICS_MM512_STREAM_PS( &dpB[(j*ld2)+i], _dp );
        _dcp = _mm512_mul_ps( _dcp, _f);
        LIBXSMM_INTRINSICS_MM512_STREAM_PS( &dcp[(j*ld)+i], _dcp );
      }
    }
  }
}

#endif
