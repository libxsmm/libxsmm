/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Evangelos Georganas (Intel Corp.)
******************************************************************************/
#define NATIVE_MATRIX_RNE_CVT_FP32_BFP16_LD(m, n, ld, _src, _dst) \
do { \
  float *__src = _src; \
  libxsmm_bfloat16 *__dst = _dst; \
  libxsmm_blasint __i, __j; \
  __m512i __packed_result; \
  for ( __j = 0; __j < n; ++__j ) { \
    for ( __i = 0; __i < m; __i+=32 ) { \
      __packed_result = LIBXSMM_INTRINSISCS_MM512_CVTNE2PS_PBH(LIBXSMM_INTRINSICS_MM512_LOAD_PS((float*)&__src[(__j*ld)+__i+16]), LIBXSMM_INTRINSICS_MM512_LOAD_PS((float*)&__src[(__j*ld)+__i])); \
      _mm512_storeu_si512((libxsmm_bfloat16*)&__dst[(__j*ld)+__i], (__m512i) __packed_result); \
    } \
  } \
} while (0)

/* First perform the W*x part of the output  */
blocks = CB_BLOCKS;
for (j = 0; j < t; ++j) {
  /* let's run the cell in blocks for good locality */
  /* Block reduction loop if requested */
  for (CB = 0; CB < BF; CB++) {
    for (inik = thr_begin; inik < thr_end; ++inik ) {
      inb = inik % (N/bn);
      ikb = inik / (N/bn);
      ik = ikb*bk;
      /* initialize i with bi */
#ifdef PROFILE
      if (ltid == 0) gemm_start = _rdtsc();
#endif
      if (CB == 0) MATRIX_BCST_CVT_BF16_FP32_COLVECTOR_LD( bk, bn, bk, &LIBXSMM_VLA_ACCESS(5, i, j, inb, ikb, 0, 0, nBlocks, kBlocks, bn, bk), &bi[ik] );
      /* i += W.x */
      batchreduce_kernela(&LIBXSMM_VLA_ACCESS(5, wi, ikb, CB*CB_BLOCKS, 0, 0, 0, cBlocks, bc_lp, bk, lpb),
                          &LIBXSMM_VLA_ACCESS(5, x, j, inb, CB*CB_BLOCKS, 0, 0, nBlocks, cBlocks, bn, bc),
                          &LIBXSMM_VLA_ACCESS(5, i, j, inb, ikb, 0, 0, nBlocks, kBlocks, bn, bk), &blocks);

      /* initialize ci with bd */
      if (CB == 0) MATRIX_BCST_CVT_BF16_FP32_COLVECTOR_LD( bk, bn, bk, &LIBXSMM_VLA_ACCESS(5, ci, j, inb, ikb, 0, 0, nBlocks, kBlocks, bn, bk), &bd[ik] );
      /* ci += W.x */
      batchreduce_kernela(&LIBXSMM_VLA_ACCESS(5, wc, ikb, CB*CB_BLOCKS, 0, 0, 0, cBlocks, bc_lp, bk, lpb),
                          &LIBXSMM_VLA_ACCESS(5, x, j, inb, CB*CB_BLOCKS, 0, 0, nBlocks, cBlocks, bn, bc),
                          &LIBXSMM_VLA_ACCESS(5, ci, j, inb, ikb, 0, 0, nBlocks, kBlocks, bn, bk), &blocks);

      /* initialize f with (bf + forget_bias) */
      if (CB == 0)  MATRIX_BCST_CVT_BF16_FP32_COLVECTOR_CONST_LD( bk, bn, bk, &LIBXSMM_VLA_ACCESS(5, f, j, inb, ikb, 0, 0, nBlocks, kBlocks, bn, bk), &bf[ik], handle->forget_bias );
      /* f += W.x */
      batchreduce_kernela(&LIBXSMM_VLA_ACCESS(5, wf, ikb, CB*CB_BLOCKS, 0, 0, 0, cBlocks, bc_lp, bk, lpb),
                          &LIBXSMM_VLA_ACCESS(5, x, j, inb, CB*CB_BLOCKS, 0, 0, nBlocks, cBlocks, bn, bc),
                          &LIBXSMM_VLA_ACCESS(5, f, j, inb, ikb, 0, 0, nBlocks, kBlocks, bn, bk), &blocks);

      /* initialize o with bo */
      if (CB == 0) MATRIX_BCST_CVT_BF16_FP32_COLVECTOR_LD( bk, bn, bk, &LIBXSMM_VLA_ACCESS(5, o, j, inb, ikb, 0, 0, nBlocks, kBlocks, bn, bk), &bo[ik] );
      /* o += W.x */
      batchreduce_kernela(&LIBXSMM_VLA_ACCESS(5, wo, ikb, CB*CB_BLOCKS, 0, 0, 0, cBlocks, bc_lp, bk, lpb),
                          &LIBXSMM_VLA_ACCESS(5, x, j, inb, CB*CB_BLOCKS, 0, 0, nBlocks, cBlocks, bn, bc),
                          &LIBXSMM_VLA_ACCESS(5, o, j, inb, ikb, 0, 0, nBlocks, kBlocks, bn, bk), &blocks);
#ifdef PROFILE
      if (ltid == 0) {
        gemm_end = _rdtsc();
        gemm_cycles += gemm_end-gemm_start;
      }
#endif
    }
  }
  libxsmm_barrier_wait(handle->barrier, (int)ltid);
}

/* Compute the R*h part of the output  */
blocks = KB_BLOCKS;
/* Peel off the t=0 iteration to hoist the innermost if conditions  */
j = 0;
/* let's run the cell in blocks for good locality */
/* Block reduction loop if requested */
for (CB = 0; CB < BF; CB++) {
  for (inik = thr_begin; inik < thr_end; ++inik ) {
    inb = inik % (N/bn);
    ikb = inik / (N/bn);
    ik = ikb*bk;
#ifdef PROFILE
    if (ltid == 0) gemm_start = _rdtsc();
#endif
    /* i += R.h */
    batchreduce_kernelb(&LIBXSMM_VLA_ACCESS(5, ri, ikb, CB*KB_BLOCKS, 0, 0, 0, kBlocks, bk_lp, bk, lpb),
        &LIBXSMM_VLA_ACCESS(4, hp, inb, CB*KB_BLOCKS, 0, 0, kBlocks, bn, bk),
        &LIBXSMM_VLA_ACCESS(5, i, j, inb, ikb, 0, 0, nBlocks, kBlocks, bn, bk), &blocks);
#ifdef PROFILE
    if (ltid == 0) {
      gemm_end = _rdtsc();
      gemm_cycles2 += gemm_end-gemm_start;
    }
#endif
    /* Eltwise ops and downcovert for the i computed block  */
    if (CB == BF-1) {
      libxsmm_blasint _k, _j;
      float* _i = &LIBXSMM_VLA_ACCESS(5, i, j, inb, ikb, 0, 0, nBlocks, kBlocks, bn, bk);
      libxsmm_bfloat16 *dst = &LIBXSMM_VLA_ACCESS(5, i_out, j, inb, ikb, 0, 0, nBlocks, kBlocks, bn, bk);
      __m512 _vi0, _vi1;
      const __m512 _halves = _mm512_set1_ps( (LIBXSMM_DNN_ELTWISE_FTYPE)0.5 );
      for ( _j = 0; _j < bn; ++_j ) {
        for ( _k = 0; _k < bk; _k += 32 ) {
          _vi0 = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &_i[(_j*bk)+_k] );
          _vi0 = _mm512_fmadd_ps( LIBXSMM_INTRINSICS_MM512_TANH_PS_MINIMAX2( _mm512_mul_ps( _vi0, _halves ) ), _halves, _halves);
          _mm512_store_ps( &_i[(_j*bk)+_k], _vi0 );
          _vi1 = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &_i[(_j*bk)+_k+16] );
          _vi1 = _mm512_fmadd_ps( LIBXSMM_INTRINSICS_MM512_TANH_PS_MINIMAX2( _mm512_mul_ps( _vi1, _halves ) ), _halves, _halves);
          _mm512_store_ps( &_i[(_j*bk)+_k+16], _vi1 );
          _mm512_storeu_si512((libxsmm_bfloat16*)&dst[(_j*bk)+_k], (__m512i) LIBXSMM_INTRINSISCS_MM512_CVTNE2PS_PBH(_vi1, _vi0));
        }
      }
    }
#ifdef PROFILE
    if (ltid == 0) gemm_start = _rdtsc();
#endif
    /* ci += R.h */
    batchreduce_kernelb(&LIBXSMM_VLA_ACCESS(5, rc, ikb, CB*KB_BLOCKS, 0, 0, 0, kBlocks, bk_lp, bk, lpb),
        &LIBXSMM_VLA_ACCESS(4, hp, inb, CB*KB_BLOCKS, 0, 0, kBlocks, bn, bk),
        &LIBXSMM_VLA_ACCESS(5, ci, j, inb, ikb, 0, 0, nBlocks, kBlocks, bn, bk), &blocks);
#ifdef PROFILE
    if (ltid == 0) {
      gemm_end = _rdtsc();
      gemm_cycles2 += gemm_end-gemm_start;
    }
#endif
    /* Eltwise ops and downcovert for the ci computed block  */
    if (CB == BF-1) {
      libxsmm_blasint _k, _j;
      float* _ci = &LIBXSMM_VLA_ACCESS(5, ci, j, inb, ikb, 0, 0, nBlocks, kBlocks, bn, bk);
      libxsmm_bfloat16 *dst = &LIBXSMM_VLA_ACCESS(5, ci_out, j, inb, ikb, 0, 0, nBlocks, kBlocks, bn, bk);
      __m512 _vci0, _vci1;
      for ( _j = 0; _j < bn; ++_j ) {
        for ( _k = 0; _k < bk; _k += 32 ) {
          _vci0 = LIBXSMM_INTRINSICS_MM512_TANH_PS_MINIMAX2(LIBXSMM_INTRINSICS_MM512_LOAD_PS( &_ci[(_j*bk)+_k] ));
          _mm512_store_ps( &_ci[(_j*bk)+_k], _vci0 );
          _vci1 = LIBXSMM_INTRINSICS_MM512_TANH_PS_MINIMAX2(LIBXSMM_INTRINSICS_MM512_LOAD_PS( &_ci[(_j*bk)+_k+16] ));
          _mm512_store_ps( &_ci[(_j*bk)+_k+16], _vci1 );
          _mm512_storeu_si512((libxsmm_bfloat16*)&dst[(_j*bk)+_k], (__m512i) LIBXSMM_INTRINSISCS_MM512_CVTNE2PS_PBH(_vci1, _vci0));
        }
      }
    }
#ifdef PROFILE
    if (ltid == 0) gemm_start = _rdtsc();
#endif
    /* f += R.h */
    batchreduce_kernelb(&LIBXSMM_VLA_ACCESS(5, rf, ikb, CB*KB_BLOCKS, 0, 0, 0, kBlocks, bk_lp, bk, lpb),
        &LIBXSMM_VLA_ACCESS(4, hp, inb, CB*KB_BLOCKS, 0, 0, kBlocks, bn, bk),
        &LIBXSMM_VLA_ACCESS(5, f, j, inb, ikb, 0, 0, nBlocks, kBlocks, bn, bk), &blocks);
#ifdef PROFILE
    if (ltid == 0) {
      gemm_end = _rdtsc();
      gemm_cycles2 += gemm_end-gemm_start;
    }
#endif
    /* Eltwise ops and downcovert for the f computed block  */
    if (CB == BF-1) {
      libxsmm_blasint _k, _j;
      float* _f = &LIBXSMM_VLA_ACCESS(5, f, j, inb, ikb, 0, 0, nBlocks, kBlocks, bn, bk);
      libxsmm_bfloat16 *dst = &LIBXSMM_VLA_ACCESS(5, f_out, j, inb, ikb, 0, 0, nBlocks, kBlocks, bn, bk);
      __m512 _vf0, _vf1;
      const __m512 _halves = _mm512_set1_ps( (LIBXSMM_DNN_ELTWISE_FTYPE)0.5 );
      for ( _j = 0; _j < bn; ++_j ) {
        for ( _k = 0; _k < bk; _k += 32 ) {
          _vf0 = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &_f[(_j*bk)+_k] );
          _vf0 = _mm512_fmadd_ps( LIBXSMM_INTRINSICS_MM512_TANH_PS_MINIMAX2( _mm512_mul_ps( _vf0, _halves ) ), _halves, _halves);
          _mm512_store_ps( &_f[(_j*bk)+_k], _vf0 );
          _vf1 = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &_f[(_j*bk)+_k+16] );
          _vf1 = _mm512_fmadd_ps( LIBXSMM_INTRINSICS_MM512_TANH_PS_MINIMAX2( _mm512_mul_ps( _vf1, _halves ) ), _halves, _halves);
          _mm512_store_ps( &_f[(_j*bk)+_k+16], _vf1 );
          _mm512_storeu_si512((libxsmm_bfloat16*)&dst[(_j*bk)+_k], (__m512i) LIBXSMM_INTRINSISCS_MM512_CVTNE2PS_PBH(_vf1, _vf0));
        }
      }
    }
#ifdef PROFILE
    if (ltid == 0) gemm_start = _rdtsc();
#endif
    /* o += R.h */
    batchreduce_kernelb(&LIBXSMM_VLA_ACCESS(5, ro, ikb, CB*KB_BLOCKS, 0, 0, 0, kBlocks, bk_lp, bk, lpb),
        &LIBXSMM_VLA_ACCESS(4, hp, inb, CB*KB_BLOCKS, 0, 0, kBlocks, bn, bk),
        &LIBXSMM_VLA_ACCESS(5, o, j, inb, ikb, 0, 0, nBlocks, kBlocks, bn, bk), &blocks);
#ifdef PROFILE
    if (ltid == 0) {
      gemm_end = _rdtsc();
      gemm_cycles2 += gemm_end-gemm_start;
    }
#endif
    /* Eltwise ops and downcovert for the o computed block  */
    if (CB == BF-1) {
      libxsmm_blasint _k, _j;
      float* _o = &LIBXSMM_VLA_ACCESS(5, o, j, inb, ikb, 0, 0, nBlocks, kBlocks, bn, bk);
      float* _i = &LIBXSMM_VLA_ACCESS(5, i, j, inb, ikb, 0, 0, nBlocks, kBlocks, bn, bk);
      float* _f = &LIBXSMM_VLA_ACCESS(5, f, j, inb, ikb, 0, 0, nBlocks, kBlocks, bn, bk);
      float* _ci = &LIBXSMM_VLA_ACCESS(5, ci, j, inb, ikb, 0, 0, nBlocks, kBlocks, bn, bk);
      float* _cps = &LIBXSMM_VLA_ACCESS(4, cp, inb, ikb, 0, 0, kBlocks, bn, bk);
      float* _cs = &LIBXSMM_VLA_ACCESS(5, cs, j, inb, ikb, 0, 0, nBlocks, kBlocks, bn, bk);
      libxsmm_bfloat16 *dst_o = &LIBXSMM_VLA_ACCESS(5, o_out, j, inb, ikb, 0, 0, nBlocks, kBlocks, bn, bk);
      libxsmm_bfloat16 *dst_cs = &LIBXSMM_VLA_ACCESS(5, cs_out, j, inb, ikb, 0, 0, nBlocks, kBlocks, bn, bk);
      libxsmm_bfloat16 *dst_h = &LIBXSMM_VLA_ACCESS(5, h_out, j, inb, ikb, 0, 0, nBlocks, kBlocks, bn, bk);
      libxsmm_bfloat16 *dst_co = &LIBXSMM_VLA_ACCESS(5, co_out, j, inb, ikb, 0, 0, nBlocks, kBlocks, bn, bk);
      __m512 _vf, _vcs, _vi, _vci, _vco, _vo, _vh, _vf1, _vcs1, _vi1, _vci1, _vco1, _vo1, _vh1;
      const __m512 _halves = _mm512_set1_ps( (LIBXSMM_DNN_ELTWISE_FTYPE)0.5 );
      for ( _j = 0; _j < bn; ++_j ) {
        for ( _k = 0; _k < bk; _k += 32 ) {
          _vo = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &_o[(_j*bk)+_k] );
          _vi = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &_i[(_j*bk)+_k] );
          _vci = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &_ci[(_j*bk)+_k] );
          _vf = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &_f[(_j*bk)+_k] );
          _vcs = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &_cps[(_j*bk)+_k] );
          _vo = _mm512_fmadd_ps( LIBXSMM_INTRINSICS_MM512_TANH_PS_MINIMAX2( _mm512_mul_ps( _vo, _halves ) ), _halves, _halves);
          _vcs = _mm512_mul_ps( _vf, _vcs );
          _vcs = _mm512_fmadd_ps( _vi, _vci, _vcs );
          _mm512_store_ps( &_cs[(_j*bk)+_k], _vcs );
          _vco = LIBXSMM_INTRINSICS_MM512_TANH_PS_MINIMAX2( _vcs );
          _vh = _mm512_mul_ps( _vo, _vco );
          _vo1 = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &_o[(_j*bk)+_k+16] );
          _vi1 = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &_i[(_j*bk)+_k+16] );
          _vci1 = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &_ci[(_j*bk)+_k+16] );
          _vf1 = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &_f[(_j*bk)+_k+16] );
          _vcs1 = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &_cps[(_j*bk)+_k+16] );
          _vo1 = _mm512_fmadd_ps( LIBXSMM_INTRINSICS_MM512_TANH_PS_MINIMAX2( _mm512_mul_ps( _vo1, _halves ) ), _halves, _halves);
          _vcs1 = _mm512_mul_ps( _vf1, _vcs1 );
          _vcs1 = _mm512_fmadd_ps( _vi1, _vci1, _vcs1 );
          _mm512_store_ps( &_cs[(_j*bk)+_k+16], _vcs1 );
          _vco1 = LIBXSMM_INTRINSICS_MM512_TANH_PS_MINIMAX2( _vcs1 );
          _vh1 = _mm512_mul_ps( _vo1, _vco1 );
          _mm512_storeu_si512((libxsmm_bfloat16*)&dst_o[(_j*bk)+_k], (__m512i) LIBXSMM_INTRINSISCS_MM512_CVTNE2PS_PBH(_vo1, _vo));
          _mm512_storeu_si512((libxsmm_bfloat16*)&dst_cs[(_j*bk)+_k], (__m512i) LIBXSMM_INTRINSISCS_MM512_CVTNE2PS_PBH(_vcs1, _vcs));
          _mm512_storeu_si512((libxsmm_bfloat16*)&dst_h[(_j*bk)+_k], (__m512i) LIBXSMM_INTRINSISCS_MM512_CVTNE2PS_PBH(_vh1, _vh));
          _mm512_storeu_si512((libxsmm_bfloat16*)&dst_co[(_j*bk)+_k], (__m512i) LIBXSMM_INTRINSISCS_MM512_CVTNE2PS_PBH(_vco1, _vco));
        }
      }
    }
  }
}
libxsmm_barrier_wait(handle->barrier, (int)ltid);

for (j = 1; j < t; ++j) {
  /* let's run the cell in blocks for good locality */
  /* Block reduction loop if requested */
  for (CB = 0; CB < BF; CB++) {
    for (inik = thr_begin; inik < thr_end; ++inik ) {
      inb = inik % (N/bn);
      ikb = inik / (N/bn);
      ik = ikb*bk;
#ifdef PROFILE
      if (ltid == 0) gemm_start = _rdtsc();
#endif
      /* i += R.h */
      batchreduce_kernelb(&LIBXSMM_VLA_ACCESS(5, ri, ikb, CB*KB_BLOCKS, 0, 0, 0, kBlocks, bk_lp, bk, lpb),
          &LIBXSMM_VLA_ACCESS(5, h_out, j-1, inb, CB*KB_BLOCKS, 0, 0, nBlocks, kBlocks, bn, bk),
          &LIBXSMM_VLA_ACCESS(5, i, j, inb, ikb, 0, 0, nBlocks, kBlocks, bn, bk), &blocks);
#ifdef PROFILE
      if (ltid == 0) {
        gemm_end = _rdtsc();
        gemm_cycles2 += gemm_end-gemm_start;
      }
#endif
      /* Eltwise ops and downcovert for the i computed block  */
      if (CB == BF-1) {
        libxsmm_blasint _k, _j;
        float* _i = &LIBXSMM_VLA_ACCESS(5, i, j, inb, ikb, 0, 0, nBlocks, kBlocks, bn, bk);
        libxsmm_bfloat16 *dst = &LIBXSMM_VLA_ACCESS(5, i_out, j, inb, ikb, 0, 0, nBlocks, kBlocks, bn, bk);
        __m512 _vi0, _vi1;
        const __m512 _halves = _mm512_set1_ps( (LIBXSMM_DNN_ELTWISE_FTYPE)0.5 );
        for ( _j = 0; _j < bn; ++_j ) {
          for ( _k = 0; _k < bk; _k += 32 ) {
            _vi0 = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &_i[(_j*bk)+_k] );
            _vi0 = _mm512_fmadd_ps( LIBXSMM_INTRINSICS_MM512_TANH_PS_MINIMAX2( _mm512_mul_ps( _vi0, _halves ) ), _halves, _halves);
            _mm512_store_ps( &_i[(_j*bk)+_k], _vi0 );
            _vi1 = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &_i[(_j*bk)+_k+16] );
            _vi1 = _mm512_fmadd_ps( LIBXSMM_INTRINSICS_MM512_TANH_PS_MINIMAX2( _mm512_mul_ps( _vi1, _halves ) ), _halves, _halves);
            _mm512_store_ps( &_i[(_j*bk)+_k+16], _vi1 );
            _mm512_storeu_si512((libxsmm_bfloat16*)&dst[(_j*bk)+_k], (__m512i) LIBXSMM_INTRINSISCS_MM512_CVTNE2PS_PBH(_vi1, _vi0));
          }
        }
      }
#ifdef PROFILE
      if (ltid == 0) gemm_start = _rdtsc();
#endif
      /* ci += R.h */
      batchreduce_kernelb(&LIBXSMM_VLA_ACCESS(5, rc, ikb, CB*KB_BLOCKS, 0, 0, 0, kBlocks, bk_lp, bk, lpb),
          &LIBXSMM_VLA_ACCESS(5, h_out, j-1, inb, CB*KB_BLOCKS, 0, 0, nBlocks, kBlocks, bn, bk),
          &LIBXSMM_VLA_ACCESS(5, ci, j, inb, ikb, 0, 0, nBlocks, kBlocks, bn, bk), &blocks);
#ifdef PROFILE
      if (ltid == 0) {
        gemm_end = _rdtsc();
        gemm_cycles2 += gemm_end-gemm_start;
      }
#endif
      /* Eltwise ops and downcovert for the ci computed block  */
      if (CB == BF-1) {
        libxsmm_blasint _k, _j;
        float* _ci = &LIBXSMM_VLA_ACCESS(5, ci, j, inb, ikb, 0, 0, nBlocks, kBlocks, bn, bk);
        libxsmm_bfloat16 *dst = &LIBXSMM_VLA_ACCESS(5, ci_out, j, inb, ikb, 0, 0, nBlocks, kBlocks, bn, bk);
        __m512 _vci0, _vci1;
        for ( _j = 0; _j < bn; ++_j ) {
          for ( _k = 0; _k < bk; _k += 32 ) {
            _vci0 = LIBXSMM_INTRINSICS_MM512_TANH_PS_MINIMAX2(LIBXSMM_INTRINSICS_MM512_LOAD_PS( &_ci[(_j*bk)+_k] ));
            _mm512_store_ps( &_ci[(_j*bk)+_k], _vci0 );
            _vci1 = LIBXSMM_INTRINSICS_MM512_TANH_PS_MINIMAX2(LIBXSMM_INTRINSICS_MM512_LOAD_PS( &_ci[(_j*bk)+_k+16] ));
            _mm512_store_ps( &_ci[(_j*bk)+_k+16], _vci1 );
            _mm512_storeu_si512((libxsmm_bfloat16*)&dst[(_j*bk)+_k], (__m512i) LIBXSMM_INTRINSISCS_MM512_CVTNE2PS_PBH(_vci1, _vci0));
          }
        }
      }
#ifdef PROFILE
      if (ltid == 0) gemm_start = _rdtsc();
#endif
      /* f += R.h */
      batchreduce_kernelb(&LIBXSMM_VLA_ACCESS(5, rf, ikb, CB*KB_BLOCKS, 0, 0, 0, kBlocks, bk_lp, bk, lpb),
          &LIBXSMM_VLA_ACCESS(5, h_out, j-1, inb, CB*KB_BLOCKS, 0, 0, nBlocks, kBlocks, bn, bk),
          &LIBXSMM_VLA_ACCESS(5, f, j, inb, ikb, 0, 0, nBlocks, kBlocks, bn, bk), &blocks);
#ifdef PROFILE
      if (ltid == 0) {
        gemm_end = _rdtsc();
        gemm_cycles2 += gemm_end-gemm_start;
      }
#endif
      /* Eltwise ops and downcovert for the f computed block  */
      if (CB == BF-1) {
        libxsmm_blasint _k, _j;
        float* _f = &LIBXSMM_VLA_ACCESS(5, f, j, inb, ikb, 0, 0, nBlocks, kBlocks, bn, bk);
        libxsmm_bfloat16 *dst = &LIBXSMM_VLA_ACCESS(5, f_out, j, inb, ikb, 0, 0, nBlocks, kBlocks, bn, bk);
        __m512 _vf0, _vf1;
        const __m512 _halves = _mm512_set1_ps( (LIBXSMM_DNN_ELTWISE_FTYPE)0.5 );
        for ( _j = 0; _j < bn; ++_j ) {
          for ( _k = 0; _k < bk; _k += 32 ) {
            _vf0 = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &_f[(_j*bk)+_k] );
            _vf0 = _mm512_fmadd_ps( LIBXSMM_INTRINSICS_MM512_TANH_PS_MINIMAX2( _mm512_mul_ps( _vf0, _halves ) ), _halves, _halves);
            _mm512_store_ps( &_f[(_j*bk)+_k], _vf0 );
            _vf1 = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &_f[(_j*bk)+_k+16] );
            _vf1 = _mm512_fmadd_ps( LIBXSMM_INTRINSICS_MM512_TANH_PS_MINIMAX2( _mm512_mul_ps( _vf1, _halves ) ), _halves, _halves);
            _mm512_store_ps( &_f[(_j*bk)+_k+16], _vf1 );
            _mm512_storeu_si512((libxsmm_bfloat16*)&dst[(_j*bk)+_k], (__m512i) LIBXSMM_INTRINSISCS_MM512_CVTNE2PS_PBH(_vf1, _vf0));
          }
        }
      }
#ifdef PROFILE
      if (ltid == 0) gemm_start = _rdtsc();
#endif
      /* o += R.h */
      batchreduce_kernelb(&LIBXSMM_VLA_ACCESS(5, ro, ikb, CB*KB_BLOCKS, 0, 0, 0, kBlocks, bk_lp, bk, lpb),
          &LIBXSMM_VLA_ACCESS(5, h_out, j-1, inb, CB*KB_BLOCKS, 0, 0, nBlocks, kBlocks, bn, bk),
          &LIBXSMM_VLA_ACCESS(5, o, j, inb, ikb, 0, 0, nBlocks, kBlocks, bn, bk), &blocks);
#ifdef PROFILE
      if (ltid == 0) {
        gemm_end = _rdtsc();
        gemm_cycles2 += gemm_end-gemm_start;
      }
#endif
      /* Eltwise ops and downcovert for the o computed block  */
      if (CB == BF-1) {
        libxsmm_blasint _k, _j;
        float* _o = &LIBXSMM_VLA_ACCESS(5, o, j, inb, ikb, 0, 0, nBlocks, kBlocks, bn, bk);
        float* _i = &LIBXSMM_VLA_ACCESS(5, i, j, inb, ikb, 0, 0, nBlocks, kBlocks, bn, bk);
        float* _f = &LIBXSMM_VLA_ACCESS(5, f, j, inb, ikb, 0, 0, nBlocks, kBlocks, bn, bk);
        float* _ci = &LIBXSMM_VLA_ACCESS(5, ci, j, inb, ikb, 0, 0, nBlocks, kBlocks, bn, bk);
        float* _cps = &LIBXSMM_VLA_ACCESS(5, cs, j-1, inb, ikb, 0, 0, nBlocks, kBlocks, bn, bk);
        float* _cs = &LIBXSMM_VLA_ACCESS(5, cs, j, inb, ikb, 0, 0, nBlocks, kBlocks, bn, bk);
        libxsmm_bfloat16 *dst_o = &LIBXSMM_VLA_ACCESS(5, o_out, j, inb, ikb, 0, 0, nBlocks, kBlocks, bn, bk);
        libxsmm_bfloat16 *dst_cs = &LIBXSMM_VLA_ACCESS(5, cs_out, j, inb, ikb, 0, 0, nBlocks, kBlocks, bn, bk);
        libxsmm_bfloat16 *dst_h = &LIBXSMM_VLA_ACCESS(5, h_out, j, inb, ikb, 0, 0, nBlocks, kBlocks, bn, bk);
        libxsmm_bfloat16 *dst_co = &LIBXSMM_VLA_ACCESS(5, co_out, j, inb, ikb, 0, 0, nBlocks, kBlocks, bn, bk);
        __m512 _vf, _vcs, _vi, _vci, _vco, _vo, _vh, _vf1, _vcs1, _vi1, _vci1, _vco1, _vo1, _vh1;
        const __m512 _halves = _mm512_set1_ps( (LIBXSMM_DNN_ELTWISE_FTYPE)0.5 );
        for ( _j = 0; _j < bn; ++_j ) {
          for ( _k = 0; _k < bk; _k += 32 ) {
            _vo = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &_o[(_j*bk)+_k] );
            _vi = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &_i[(_j*bk)+_k] );
            _vci = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &_ci[(_j*bk)+_k] );
            _vf = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &_f[(_j*bk)+_k] );
            _vcs = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &_cps[(_j*bk)+_k] );
            _vo = _mm512_fmadd_ps( LIBXSMM_INTRINSICS_MM512_TANH_PS_MINIMAX2( _mm512_mul_ps( _vo, _halves ) ), _halves, _halves);
            _vcs = _mm512_mul_ps( _vf, _vcs );
            _vcs = _mm512_fmadd_ps( _vi, _vci, _vcs );
            _mm512_store_ps( &_cs[(_j*bk)+_k], _vcs );
            _vco = LIBXSMM_INTRINSICS_MM512_TANH_PS_MINIMAX2( _vcs );
            _vh = _mm512_mul_ps( _vo, _vco );
            _vo1 = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &_o[(_j*bk)+_k+16] );
            _vi1 = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &_i[(_j*bk)+_k+16] );
            _vci1 = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &_ci[(_j*bk)+_k+16] );
            _vf1 = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &_f[(_j*bk)+_k+16] );
            _vcs1 = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &_cps[(_j*bk)+_k+16] );
            _vo1 = _mm512_fmadd_ps( LIBXSMM_INTRINSICS_MM512_TANH_PS_MINIMAX2( _mm512_mul_ps( _vo1, _halves ) ), _halves, _halves);
            _vcs1 = _mm512_mul_ps( _vf1, _vcs1 );
            _vcs1 = _mm512_fmadd_ps( _vi1, _vci1, _vcs1 );
            _mm512_store_ps( &_cs[(_j*bk)+_k+16], _vcs1 );
            _vco1 = LIBXSMM_INTRINSICS_MM512_TANH_PS_MINIMAX2( _vcs1 );
            _vh1 = _mm512_mul_ps( _vo1, _vco1 );
            _mm512_storeu_si512((libxsmm_bfloat16*)&dst_o[(_j*bk)+_k], (__m512i) LIBXSMM_INTRINSISCS_MM512_CVTNE2PS_PBH(_vo1, _vo));
            _mm512_storeu_si512((libxsmm_bfloat16*)&dst_cs[(_j*bk)+_k], (__m512i) LIBXSMM_INTRINSISCS_MM512_CVTNE2PS_PBH(_vcs1, _vcs));
            _mm512_storeu_si512((libxsmm_bfloat16*)&dst_h[(_j*bk)+_k], (__m512i) LIBXSMM_INTRINSISCS_MM512_CVTNE2PS_PBH(_vh1, _vh));
            _mm512_storeu_si512((libxsmm_bfloat16*)&dst_co[(_j*bk)+_k], (__m512i) LIBXSMM_INTRINSISCS_MM512_CVTNE2PS_PBH(_vco1, _vco));
          }
        }
      }
    }
  }
  libxsmm_barrier_wait(handle->barrier, (int)ltid);
}

#undef NATIVE_MATRIX_RNE_CVT_FP32_BFP16_LD

