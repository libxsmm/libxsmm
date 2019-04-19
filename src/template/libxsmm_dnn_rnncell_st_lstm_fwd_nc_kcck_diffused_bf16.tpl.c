/******************************************************************************
** Copyright (c) 2017-2018, Intel Corporation                                **
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
/* Evangelos Georganas (Intel Corp.)
******************************************************************************/

/* First perform the W*x part of the output  */
for (j = 0; j < t; ++j) {
  /* let's run the cell in blocks for good locality */
  /* Block reduction loop if requested */
  for (CB = 0; CB < BF; CB++) {
    for (inik = thr_begin; inik < thr_end; ++inik ) {
      in = (inik % (N/bn))*bn;
      ikb = inik / (N/bn);
      ik = ikb*bk;
      /* initialize i with bi */
#ifdef PROFILE
      if (ltid == 0) gemm_start = _rdtsc();
#endif
      if (CB == 0) libxsmm_internal_matrix_bcst_cvt_bf16_fp32_colvector_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, i, j, in, ik, N, K), &bi[ik] );
      /* i += W.x */
      for (icb = 0, ic = 0; icb < CB_BLOCKS; ic += bc, icb++) {
        A_array[icb] = &LIBXSMM_VLA_ACCESS(5, wi, ikb, icb + CB*CB_BLOCKS, 0, 0, 0, cBlocks, bc_lp, bk, lpb);
        B_array[icb] = &LIBXSMM_VLA_ACCESS(3, x, j, in, ic + CB*CB_BLOCKS*bc, N, C);
      }
      /* Reduce batch gemm call  */
      blocks = CB_BLOCKS;
      batchreduce_kernela(A_array, B_array, &LIBXSMM_VLA_ACCESS(3, i, j, in, ik, N, K), &blocks);
#ifdef PROFILE
      if (ltid == 0) {
        gemm_end = _rdtsc();
        gemm_cycles += gemm_end-gemm_start;
      }
#endif
#ifdef PROFILE
      if (ltid == 0) gemm_start = _rdtsc();
#endif
      /* initialize ci with bd */
      if (CB == 0) libxsmm_internal_matrix_bcst_cvt_bf16_fp32_colvector_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, ci, j, in, ik, N, K), &bd[ik] );
      /* ci += W.x */
      for (icb = 0, ic = 0; icb < CB_BLOCKS; ic += bc, icb++) {
        A_array[icb] = &LIBXSMM_VLA_ACCESS(5, wc, ikb, icb + CB*CB_BLOCKS, 0, 0, 0, cBlocks, bc_lp, bk, lpb);
        B_array[icb] = &LIBXSMM_VLA_ACCESS(3, x, j, in, ic + CB*CB_BLOCKS*bc, N, C);
      }
      /* Reduce batch gemm call  */
      blocks = CB_BLOCKS;
      batchreduce_kernela(A_array, B_array, &LIBXSMM_VLA_ACCESS(3, ci, j, in, ik, N, K), &blocks);
#ifdef PROFILE
      if (ltid == 0) {
        gemm_end = _rdtsc();
        gemm_cycles += gemm_end-gemm_start;
      }
#endif
#ifdef PROFILE
      if (ltid == 0) gemm_start = _rdtsc();
#endif
      /* initialize f with (bf + forget_bias) */
      if (CB == 0)  libxsmm_internal_matrix_bcst_cvt_bf16_fp32_colvector_const_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, f, j, in, ik, N, K), &bf[ik], handle->forget_bias );
      /* f += W.x */
      for (icb = 0, ic = 0; icb < CB_BLOCKS; ic += bc, icb++) {
        A_array[icb] = &LIBXSMM_VLA_ACCESS(5, wf, ikb, icb + CB*CB_BLOCKS, 0, 0, 0, cBlocks, bc_lp, bk, lpb);
        B_array[icb] = &LIBXSMM_VLA_ACCESS(3, x, j, in, ic + CB*CB_BLOCKS*bc, N, C);
      }
      /* Reduce batch gemm call  */
      blocks = CB_BLOCKS;
      batchreduce_kernela(A_array, B_array, &LIBXSMM_VLA_ACCESS(3, f, j, in, ik, N, K), &blocks);
#ifdef PROFILE
      if (ltid == 0) {
        gemm_end = _rdtsc();
        gemm_cycles += gemm_end-gemm_start;
      }
#endif

#ifdef PROFILE
      if (ltid == 0) gemm_start = _rdtsc();
#endif
      /* initialize o with bo */
      if (CB == 0) libxsmm_internal_matrix_bcst_cvt_bf16_fp32_colvector_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, o, j, in, ik, N, K), &bo[ik] );
      /* o += W.x */
      for (icb = 0, ic = 0; icb < CB_BLOCKS; ic += bc, icb++) {
        A_array[icb] = &LIBXSMM_VLA_ACCESS(5, wo, ikb, icb + CB*CB_BLOCKS, 0, 0, 0, cBlocks, bc_lp, bk, lpb);
        B_array[icb] = &LIBXSMM_VLA_ACCESS(3, x, j, in, ic + CB*CB_BLOCKS*bc, N, C);
      }
      /* Reduce batch gemm call  */
      blocks = CB_BLOCKS;
      batchreduce_kernela(A_array, B_array, &LIBXSMM_VLA_ACCESS(3, o, j, in, ik, N, K), &blocks);
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
for (j = 0; j < t; ++j) {
  /* let's run the cell in blocks for good locality */
  /* Block reduction loop if requested */
  for (CB = 0; CB < BF; CB++) {
    for (inik = thr_begin; inik < thr_end; ++inik ) {
      in = (inik % (N/bn))*bn;
      ikb = inik / (N/bn);
      ik = ikb*bk;
#ifdef PROFILE
      if (ltid == 0) gemm_start = _rdtsc();
#endif
      /* i += R.h */
      if (0 == j) {
        for (ic = 0, icb = 0; icb < KB_BLOCKS; ic += bk, icb++) {
          A_array[icb] = &LIBXSMM_VLA_ACCESS(5, ri, ikb, icb + CB*KB_BLOCKS, 0, 0, 0, kBlocks, bk_lp, bk, lpb);
          B_array[icb] = &LIBXSMM_VLA_ACCESS(2, hp, in, ic + CB*KB_BLOCKS*bk, K);
        }
      } else {
        for (ic = 0, icb = 0; icb < KB_BLOCKS; ic += bk, icb++) {
          A_array[icb] = &LIBXSMM_VLA_ACCESS(5, ri, ikb, icb + CB*KB_BLOCKS, 0, 0, 0, kBlocks, bk_lp, bk, lpb);
          B_array[icb] = &LIBXSMM_VLA_ACCESS(3, h_out, j-1, in, ic + CB*KB_BLOCKS*bk, N, K);
        }
      }
      /* Reduce batch gemm call  */
      blocks = KB_BLOCKS;
      batchreduce_kernelb(A_array, B_array, &LIBXSMM_VLA_ACCESS(3, i, j, in, ik, N, K), &blocks);
#ifdef PROFILE
      if (ltid == 0) {
        gemm_end = _rdtsc();
        gemm_cycles2 += gemm_end-gemm_start;
      }
#endif
#ifdef PROFILE
      if (ltid == 0) gemm_start = _rdtsc();
#endif
      /* ci += R.h */
      if (0 == j) {
        for (ic = 0, icb = 0; icb < KB_BLOCKS; ic += bk, icb++) {
          A_array[icb] = &LIBXSMM_VLA_ACCESS(5, rc, ikb, icb + CB*KB_BLOCKS, 0, 0, 0, kBlocks, bk_lp, bk, lpb);
          B_array[icb] = &LIBXSMM_VLA_ACCESS(2, hp, in, ic + CB*KB_BLOCKS*bk, K);
        }
      } else {
        for (ic = 0, icb = 0; icb < KB_BLOCKS; ic += bk, icb++) {
          A_array[icb] = &LIBXSMM_VLA_ACCESS(5, rc, ikb, icb + CB*KB_BLOCKS, 0, 0, 0, kBlocks, bk_lp, bk, lpb);
          B_array[icb] = &LIBXSMM_VLA_ACCESS(3, h_out, j-1, in, ic + CB*KB_BLOCKS*bk, N, K);
        }
      }
      /* Reduce batch gemm call  */
      blocks = KB_BLOCKS;
      batchreduce_kernelb(A_array, B_array, &LIBXSMM_VLA_ACCESS(3, ci, j, in, ik, N, K), &blocks);
#ifdef PROFILE
      if (ltid == 0) {
        gemm_end = _rdtsc();
        gemm_cycles2 += gemm_end-gemm_start;
      }
#endif
#ifdef PROFILE
      if (ltid == 0) gemm_start = _rdtsc();
#endif
      /* f += R.h */
      if (0 == j) {
        for (ic = 0, icb = 0; icb < KB_BLOCKS; ic += bk, icb++) {
          A_array[icb] = &LIBXSMM_VLA_ACCESS(5, rf, ikb, icb + CB*KB_BLOCKS, 0, 0, 0, kBlocks, bk_lp, bk, lpb);
          B_array[icb] = &LIBXSMM_VLA_ACCESS(2, hp, in, ic + CB*KB_BLOCKS*bk, K);
        }
      } else {
        for (ic = 0, icb = 0; icb < KB_BLOCKS; ic += bk, icb++) {
          A_array[icb] = &LIBXSMM_VLA_ACCESS(5, rf, ikb, icb + CB*KB_BLOCKS, 0, 0, 0, kBlocks, bk_lp, bk, lpb);
          B_array[icb] = &LIBXSMM_VLA_ACCESS(3, h_out, j-1, in, ic + CB*KB_BLOCKS*bk, N, K);
        }
      }
      /* Reduce batch gemm call  */
      blocks = KB_BLOCKS;
      batchreduce_kernelb(A_array, B_array, &LIBXSMM_VLA_ACCESS(3, f, j, in, ik, N, K), &blocks);
#ifdef PROFILE
      if (ltid == 0) {
        gemm_end = _rdtsc();
        gemm_cycles2 += gemm_end-gemm_start;
      }
#endif
#ifdef PROFILE
      if (ltid == 0) gemm_start = _rdtsc();
#endif
      /* o += R.h */
      if (0 == j) {
        for (ic = 0, icb = 0; icb < KB_BLOCKS; ic += bk, icb++) {
          A_array[icb] = &LIBXSMM_VLA_ACCESS(5, ro, ikb, icb + CB*KB_BLOCKS, 0, 0, 0, kBlocks, bk_lp, bk, lpb);
          B_array[icb] = &LIBXSMM_VLA_ACCESS(2, hp, in, ic + CB*KB_BLOCKS*bk, K);
        }
      } else {
        for (ic = 0, icb = 0; icb < KB_BLOCKS; ic += bk, icb++) {
          A_array[icb] = &LIBXSMM_VLA_ACCESS(5, ro, ikb, icb + CB*KB_BLOCKS, 0, 0, 0, kBlocks, bk_lp, bk, lpb);
          B_array[icb] = &LIBXSMM_VLA_ACCESS(3, h_out, j-1, in, ic + CB*KB_BLOCKS*bk, N, K);
        }
      }
      /* Reduce batch gemm call  */
      blocks = KB_BLOCKS;
      batchreduce_kernelb(A_array, B_array, &LIBXSMM_VLA_ACCESS(3, o, j, in, ik, N, K), &blocks);
#ifdef PROFILE
      if (ltid == 0) {
        gemm_end = _rdtsc();
        gemm_cycles2 += gemm_end-gemm_start;
      }
#endif

      if (CB == BF-1) {
#ifdef PROFILE
        if (ltid == 0) {
          eltwise_start = _rdtsc();
        }
#endif
        cps_ptr = (j == 0) ? &LIBXSMM_VLA_ACCESS(2, cp, in, ik, K) : &LIBXSMM_VLA_ACCESS(3, cs, j-1, in, ik, N, K) ;
        /* Compute i, ci, f, o, cs, co and h */
#if defined(LIBXSMM_RNN_CELL_AVX512)
        if (bk % 16 == 0 && bc % 16 == 0) {
#include "libxsmm_internal_lstm_fwd_fused_eltwise_bf16.tpl.c"
        } else {
          libxsmm_internal_matrix_sigmoid_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, i, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, i, j, in, ik, N, K) );
          libxsmm_internal_matrix_sigmoid_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, f, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, f, j, in, ik, N, K) );
          libxsmm_internal_matrix_sigmoid_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, o, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, o, j, in, ik, N, K) );
          libxsmm_internal_matrix_tanh_ld(    bk, bn, K, &LIBXSMM_VLA_ACCESS(3, ci, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, ci, j, in, ik, N, K) );
          libxsmm_internal_matrix_eltwise_mult_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, f, j, in, ik, N, K), cps_ptr, &LIBXSMM_VLA_ACCESS(3, cs, j, in, ik, N, K) );
          libxsmm_internal_matrix_eltwise_fma_ld(  bk, bn, K, &LIBXSMM_VLA_ACCESS(3, i, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, ci, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, cs, j, in, ik, N, K) );
          libxsmm_internal_matrix_tanh_ld(         bk, bn, K, &LIBXSMM_VLA_ACCESS(3, cs, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, co, j, in, ik, N, K) );
          libxsmm_internal_matrix_eltwise_mult_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, o, j, in, ik, N, K),  &LIBXSMM_VLA_ACCESS(3, co, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, h, j, in, ik, N, K) );
        }
#else
        libxsmm_internal_matrix_sigmoid_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, i, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, i, j, in, ik, N, K) );
        libxsmm_internal_matrix_sigmoid_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, f, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, f, j, in, ik, N, K) );
        libxsmm_internal_matrix_sigmoid_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, o, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, o, j, in, ik, N, K) );
        libxsmm_internal_matrix_tanh_ld(    bk, bn, K, &LIBXSMM_VLA_ACCESS(3, ci, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, ci, j, in, ik, N, K) );
        libxsmm_internal_matrix_eltwise_mult_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, f, j, in, ik, N, K), cps_ptr, &LIBXSMM_VLA_ACCESS(3, cs, j, in, ik, N, K) );
        libxsmm_internal_matrix_eltwise_fma_ld(  bk, bn, K, &LIBXSMM_VLA_ACCESS(3, i, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, ci, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, cs, j, in, ik, N, K) );
        libxsmm_internal_matrix_tanh_ld(         bk, bn, K, &LIBXSMM_VLA_ACCESS(3, cs, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, co, j, in, ik, N, K) );
        libxsmm_internal_matrix_eltwise_mult_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, o, j, in, ik, N, K),  &LIBXSMM_VLA_ACCESS(3, co, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, h, j, in, ik, N, K) );
#endif
        /* Downconvert computed results to bf16 output  buffers */
        libxsmm_internal_matrix_rne_cvt_fp32_bfp16_ld(bk, bn, K, &LIBXSMM_VLA_ACCESS(3, cs, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, cs_out, j, in, ik, N, K));
        libxsmm_internal_matrix_rne_cvt_fp32_bfp16_ld(bk, bn, K, &LIBXSMM_VLA_ACCESS(3, h, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, h_out, j, in, ik, N, K));
        libxsmm_internal_matrix_rne_cvt_fp32_bfp16_ld(bk, bn, K, &LIBXSMM_VLA_ACCESS(3, i, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, i_out, j, in, ik, N, K));
        libxsmm_internal_matrix_rne_cvt_fp32_bfp16_ld(bk, bn, K, &LIBXSMM_VLA_ACCESS(3, f, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, f_out, j, in, ik, N, K));
        libxsmm_internal_matrix_rne_cvt_fp32_bfp16_ld(bk, bn, K, &LIBXSMM_VLA_ACCESS(3, o, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, o_out, j, in, ik, N, K));
        libxsmm_internal_matrix_rne_cvt_fp32_bfp16_ld(bk, bn, K, &LIBXSMM_VLA_ACCESS(3, ci, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, ci_out, j, in, ik, N, K));
        libxsmm_internal_matrix_rne_cvt_fp32_bfp16_ld(bk, bn, K, &LIBXSMM_VLA_ACCESS(3, co, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, co_out, j, in, ik, N, K));

#ifdef PROFILE
        if (ltid == 0) {
          eltwise_end = _rdtsc();
          eltwise_cycles += eltwise_end-eltwise_start;
        }
#endif
      }
    }
  }
  libxsmm_barrier_wait(handle->barrier, (int)ltid);
}

