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
  float *const src = _src; \
  libxsmm_bfloat16 *const dst = _dst; \
  libxsmm_blasint __i, __j; \
  __m512i packed_result; \
  for ( __j = 0; __j < n; ++__j ) { \
    for ( __i = 0; __i < m; __i+=32 ) { \
      packed_result = LIBXSMM_INTRINSISCS_MM512_CVTNE2PS_PBH(LIBXSMM_INTRINSICS_MM512_LOAD_PS((float*)&src[(__j*ld)+__i+16]), LIBXSMM_INTRINSICS_MM512_LOAD_PS((float*)&src[(__j*ld)+__i])); \
      _mm512_storeu_si512(&dst[(__j*ld)+__i], packed_result); \
    } \
  } \
} while (0)

for (j = t-1; j >= 0; --j) {
  /* let's run the cell in blocks for good locality */
#ifdef PROFILE
  if (ltid == 0) _start = _rdtsc();
#endif
  for (inik = thr_begin_nk; inik < thr_end_nk; ++inik ) {
    inb = inik % (N/bn);
    ikb = inik / (N/bn);
    in = (inik % (N/bn))*bn;
    ik = (inik / (N/bn))*bk;

#if defined(LIBXSMM_RNN_CELL_AVX512)
    /* Compute dcp, dci, di, df, dp */
    cps_ptr = (j == 0) ? &LIBXSMM_VLA_ACCESS(2, cp, in, ik, K) : &LIBXSMM_VLA_ACCESS(3, cs, j-1, in, ik, N, K);
    if (bcbk_multiples_of_16) {
      /* Also reformat di, dci, df and dp to be used in the UPD pass in blocked format ... */
#include "libxsmm_internal_lstm_bwdupd_fused_eltwise_reformat_bf16.tpl.c"
    } else {
      /* TODO: Add alternative  path here  */
    }
#else
    /* TODO: Add alternative path here  */
#endif
  }
#ifdef PROFILE
  if (ltid == 0) {
    _end = _rdtsc();
    eltwise_cycles += _end - _start;
  }
#endif

  if ( (LIBXSMM_DNN_COMPUTE_KIND_UPD == kind) || (LIBXSMM_DNN_COMPUTE_KIND_BWDUPD == kind) ) {
#ifdef PROFILE
    if (ltid == 0) _start = _rdtsc();
#endif
    /* transpose xt for current timestep */
    for (icin = thr_begin_nc; icin < thr_end_nc; ++icin ) {
      in = (icin / (C/bc))*bn;
      ic = (icin % (C/bc))*bc;

      for (jc = 0; jc < bc; ++jc) {
        for (jb = 0; jb < bn; ++jb) {
          en = in + jb;
          ec = ic + jc;
          LIBXSMM_VLA_ACCESS(2, xT, ec, en, N) =  LIBXSMM_VLA_ACCESS(3, x, j, en, ec, N, C);
        }
      }
    }

    /* transpose ht for current timestep */
    if (j == 0) {
      for (ikin = thr_begin_nk; ikin < thr_end_nk; ++ikin ) {
        in = (ikin / (K/bk))*bn;
        ik = (ikin % (K/bk))*bk;

        for (jk = 0; jk < bk; ++jk) {
          for (jb = 0; jb < bn; ++jb) {
            en = in + jb;
            ek = ik + jk;
            LIBXSMM_VLA_ACCESS(2, hT, ek, en, N) =  LIBXSMM_VLA_ACCESS(2, hp, en, ek, K);
          }
        }
      }
    } else {
      for (ikin = thr_begin_nk; ikin < thr_end_nk; ++ikin ) {
        in = (ikin / (K/bk))*bn;
        ik = (ikin % (K/bk))*bk;

        for (jk = 0; jk < bk; ++jk) {
          for (jb = 0; jb < bn; ++jb) {
            en = in + jb;
            ek = ik + jk;
            LIBXSMM_VLA_ACCESS(2, hT, ek, en, N) =  LIBXSMM_VLA_ACCESS(3, h, j-1, en, ek, N, K);
          }
        }
      }
    }
#ifdef PROFILE
    if (ltid == 0) {
      _end = _rdtsc();
      act_trans_cycles += _end - _start;
    }
#endif
  }

  libxsmm_barrier_wait(handle->barrier, (int)ltid);

  if ( (LIBXSMM_DNN_COMPUTE_KIND_BWD == kind) || (LIBXSMM_DNN_COMPUTE_KIND_BWDUPD == kind) ) {
#ifdef PROFILE
    if (ltid == 0) _start = _rdtsc();
#endif
    /* dx = W^T * difoc */
    blocks = KB_BLOCKS;
    for (KB = 0; KB < BF; KB++) {
      for (inic = thr_begin_nc; inic < thr_end_nc; ++inic ) {
        in = (inic % (N/bn))*bn;
        icb = inic / (N/bn);
        ic = icb*bc;

        batchreduce_kernela(&LIBXSMM_VLA_ACCESS(5, wiT, icb, KB*KB_BLOCKS, 0, 0, 0, kBlocks, bk_lp, bc, lpb),
                            &LIBXSMM_VLA_ACCESS(2, di,  in, KB*KB_BLOCKS*bk, K),
                            &LIBXSMM_VLA_ACCESS(3, dx, j, in, ic, N, C), &blocks);

        batchreduce_kernela(&LIBXSMM_VLA_ACCESS(5, wcT, icb, KB*KB_BLOCKS, 0, 0, 0, kBlocks, bk_lp, bc, lpb),
                            &LIBXSMM_VLA_ACCESS(2, dci,  in, KB*KB_BLOCKS*bk, K),
                            &LIBXSMM_VLA_ACCESS(3, dx, j, in, ic, N, C), &blocks);

        batchreduce_kernela(&LIBXSMM_VLA_ACCESS(5, wfT, icb, KB*KB_BLOCKS, 0, 0, 0, kBlocks, bk_lp, bc, lpb),
                            &LIBXSMM_VLA_ACCESS(2, df,  in, KB*KB_BLOCKS*bk, K),
                            &LIBXSMM_VLA_ACCESS(3, dx, j, in, ic, N, C), &blocks);

        batchreduce_kernela(&LIBXSMM_VLA_ACCESS(5, woT, icb, KB*KB_BLOCKS, 0, 0, 0, kBlocks, bk_lp, bc, lpb),
                            &LIBXSMM_VLA_ACCESS(2, dp,  in, KB*KB_BLOCKS*bk, K),
                            &LIBXSMM_VLA_ACCESS(3, dx, j, in, ic, N, C), &blocks);

        /* If last block, make sure we downconvert dx to bf16 */
        if (KB == BF-1) {
          NATIVE_MATRIX_RNE_CVT_FP32_BFP16_LD(bc, bn, C, &LIBXSMM_VLA_ACCESS(3, dx, j, in, ic, N, C), &LIBXSMM_VLA_ACCESS(3, dx_bf16, j, in, ic, N, C));
        }
      }
    }
#ifdef PROFILE
    if (ltid == 0) {
      _end = _rdtsc();
      dx_cycles += _end - _start;
    }
#endif
  }

#ifdef PROFILE
  if (ltid == 0) _start = _rdtsc();
#endif
  blocks = KB_BLOCKS;
  for (KB = 0; KB < BF; KB++) {
    for (inik = thr_begin_nk; inik < thr_end_nk; ++inik ) {
      in = (inik % (N/bn))*bn;
      ikb = inik / (N/bn);
      ik = ikb*bk;
      dout_ptr = (j > 0) ? (float*) &LIBXSMM_VLA_ACCESS(2, dout, in, ik, K) : (float*) &LIBXSMM_VLA_ACCESS(2, dhp_f32, in, ik, K);

      if (KB == 0) libxsmm_internal_matrix_zero_ld( bk, bn, K, dout_ptr);
      /* dout += R^T * difoc */
      batchreduce_kerneld(&LIBXSMM_VLA_ACCESS(5, riT, ikb, KB*KB_BLOCKS, 0, 0, 0, kBlocks, bk_lp, bk, lpb),
                          &LIBXSMM_VLA_ACCESS(2, di,  in, KB*KB_BLOCKS*bk, K),
                          dout_ptr, &blocks);

      batchreduce_kerneld(&LIBXSMM_VLA_ACCESS(5, rcT, ikb, KB*KB_BLOCKS, 0, 0, 0, kBlocks, bk_lp, bk, lpb),
                          &LIBXSMM_VLA_ACCESS(2, dci,  in, KB*KB_BLOCKS*bk, K),
                          dout_ptr, &blocks);

      batchreduce_kerneld(&LIBXSMM_VLA_ACCESS(5, rfT, ikb, KB*KB_BLOCKS, 0, 0, 0, kBlocks, bk_lp, bk, lpb),
                          &LIBXSMM_VLA_ACCESS(2, df,  in, KB*KB_BLOCKS*bk, K),
                          dout_ptr, &blocks);

      batchreduce_kerneld(&LIBXSMM_VLA_ACCESS(5, roT, ikb, KB*KB_BLOCKS, 0, 0, 0, kBlocks, bk_lp, bk, lpb),
                          &LIBXSMM_VLA_ACCESS(2, dp,  in, KB*KB_BLOCKS*bk, K),
                          dout_ptr, &blocks);

      /* Make sure when last and j == 0 to downconvert dhp to BF16  */
      if ((j == 0) && (KB == BF-1)) {
        NATIVE_MATRIX_RNE_CVT_FP32_BFP16_LD(bk, bn, K, dout_ptr, &LIBXSMM_VLA_ACCESS(2, dhp, in, ik, K));
      }
    }
  }
#ifdef PROFILE
  if (ltid == 0) {
    _end = _rdtsc();
    dout_cycles += _end - _start;
  }
#endif

  if ( (LIBXSMM_DNN_COMPUTE_KIND_UPD == kind) || (LIBXSMM_DNN_COMPUTE_KIND_BWDUPD == kind) ) {
#ifdef PROFILE
    if (ltid == 0) _start = _rdtsc();
#endif
    blocks = nBlocks;
    if ((C == K) && (bc == bk) && (bcbk_multiples_of_16 == 1)) {
      /* Interleave computation of dr = difoc * h^T and dw = difoc * x^T to take advantage of temporal locality */
      /* Use blocked format for di, dci, df and db */
      for (ikic = thr_begin_kk; ikic < thr_end_kk; ++ikic ) {
        icb = ikic / (K/bk);
        ic = icb*bk;
        ikb = ikic % (K/bk);
        ik = ikb*bk;
        batchreduce_kernelb(&LIBXSMM_VLA_ACCESS(5, diB, ikb, 0, 0, 0, 0, nBlocks, bn_lp, bk, lpb),
                            &LIBXSMM_VLA_ACCESS(2, hT, ic, 0, N),
                            &LIBXSMM_VLA_ACCESS(4, dri, ikb, icb, 0, 0, kBlocks, bk, bk), &blocks);

        batchreduce_kernelc(&LIBXSMM_VLA_ACCESS(5, diB, ikb, 0, 0, 0, 0, nBlocks, bn_lp, bk, lpb),
                            &LIBXSMM_VLA_ACCESS(2, xT, ic, 0, N),
                            &LIBXSMM_VLA_ACCESS(4, dwi, ikb, icb, 0, 0, cBlocks, bc, bk), &blocks);

        batchreduce_kernelb(&LIBXSMM_VLA_ACCESS(5, dciB, ikb, 0, 0, 0, 0, nBlocks, bn_lp, bk, lpb),
                            &LIBXSMM_VLA_ACCESS(2, hT, ic, 0, N),
                            &LIBXSMM_VLA_ACCESS(4, drc, ikb, icb, 0, 0, kBlocks, bk, bk), &blocks);

        batchreduce_kernelc(&LIBXSMM_VLA_ACCESS(5, dciB, ikb, 0, 0, 0, 0, nBlocks, bn_lp, bk, lpb),
                            &LIBXSMM_VLA_ACCESS(2, xT, ic, 0, N),
                            &LIBXSMM_VLA_ACCESS(4, dwc, ikb, icb, 0, 0, cBlocks, bc, bk), &blocks);

        batchreduce_kernelb(&LIBXSMM_VLA_ACCESS(5, dfB, ikb, 0, 0, 0, 0, nBlocks, bn_lp, bk, lpb),
                            &LIBXSMM_VLA_ACCESS(2, hT, ic, 0, N),
                            &LIBXSMM_VLA_ACCESS(4, drf, ikb, icb, 0, 0, kBlocks, bk, bk), &blocks);

        batchreduce_kernelc(&LIBXSMM_VLA_ACCESS(5, dfB, ikb, 0, 0, 0, 0, nBlocks, bn_lp, bk, lpb),
                            &LIBXSMM_VLA_ACCESS(2, xT, ic, 0, N),
                            &LIBXSMM_VLA_ACCESS(4, dwf, ikb, icb, 0, 0, cBlocks, bc, bk), &blocks);

        batchreduce_kernelb(&LIBXSMM_VLA_ACCESS(5, dpB, ikb, 0, 0, 0, 0, nBlocks, bn_lp, bk, lpb),
                            &LIBXSMM_VLA_ACCESS(2, hT, ic, 0, N),
                            &LIBXSMM_VLA_ACCESS(4, dro, ikb, icb, 0, 0, kBlocks, bk, bk), &blocks);

        batchreduce_kernelc(&LIBXSMM_VLA_ACCESS(5, dpB, ikb, 0, 0, 0, 0, nBlocks, bn_lp, bk, lpb),
                            &LIBXSMM_VLA_ACCESS(2, xT, ic, 0, N),
                            &LIBXSMM_VLA_ACCESS(4, dwo, ikb, icb, 0, 0, cBlocks, bc, bk), &blocks);
      }
    } else {
      /* dr = difoc * h^T */
      /* Use blocked format for di, dci, df and db */
      for (ikic = thr_begin_kk; ikic < thr_end_kk; ++ikic ) {
        icb = ikic / (K/bk);
        ic = icb*bk;
        ikb = ikic % (K/bk);
        ik = ikb*bk;
        batchreduce_kernelb(&LIBXSMM_VLA_ACCESS(5, diB, ikb, 0, 0, 0, 0, nBlocks, bn_lp, bk, lpb),
                            &LIBXSMM_VLA_ACCESS(2, hT, ic, 0, N),
                            &LIBXSMM_VLA_ACCESS(4, dri, ikb, icb, 0, 0, kBlocks, bk, bk), &blocks);

        batchreduce_kernelb(&LIBXSMM_VLA_ACCESS(5, dciB, ikb, 0, 0, 0, 0, nBlocks, bn_lp, bk, lpb),
                            &LIBXSMM_VLA_ACCESS(2, hT, ic, 0, N),
                            &LIBXSMM_VLA_ACCESS(4, drc, ikb, icb, 0, 0, kBlocks, bk, bk), &blocks);

        batchreduce_kernelb(&LIBXSMM_VLA_ACCESS(5, dfB, ikb, 0, 0, 0, 0, nBlocks, bn_lp, bk, lpb),
                            &LIBXSMM_VLA_ACCESS(2, hT, ic, 0, N),
                            &LIBXSMM_VLA_ACCESS(4, drf, ikb, icb, 0, 0, kBlocks, bk, bk), &blocks);

        batchreduce_kernelb(&LIBXSMM_VLA_ACCESS(5, dpB, ikb, 0, 0, 0, 0, nBlocks, bn_lp, bk, lpb),
                            &LIBXSMM_VLA_ACCESS(2, hT, ic, 0, N),
                            &LIBXSMM_VLA_ACCESS(4, dro, ikb, icb, 0, 0, kBlocks, bk, bk), &blocks);
      }

      /* dw = difoc * x^T */
      for (ikic = thr_begin_ck; ikic < thr_end_ck; ++ikic ) {
        icb = ikic / (K/bk);
        ic = icb*bc;
        ikb = ikic % (K/bk);
        ik = ikb*bk;
        batchreduce_kernelc(&LIBXSMM_VLA_ACCESS(5, diB, ikb, 0, 0, 0, 0, nBlocks, bn_lp, bk, lpb),
                            &LIBXSMM_VLA_ACCESS(2, xT, ic, 0, N),
                            &LIBXSMM_VLA_ACCESS(4, dwi, ikb, icb, 0, 0, cBlocks, bc, bk), &blocks);

        batchreduce_kernelc(&LIBXSMM_VLA_ACCESS(5, dciB, ikb, 0, 0, 0, 0, nBlocks, bn_lp, bk, lpb),
                            &LIBXSMM_VLA_ACCESS(2, xT, ic, 0, N),
                            &LIBXSMM_VLA_ACCESS(4, dwc, ikb, icb, 0, 0, cBlocks, bc, bk), &blocks);

        batchreduce_kernelc(&LIBXSMM_VLA_ACCESS(5, dfB, ikb, 0, 0, 0, 0, nBlocks, bn_lp, bk, lpb),
                            &LIBXSMM_VLA_ACCESS(2, xT, ic, 0, N),
                            &LIBXSMM_VLA_ACCESS(4, dwf, ikb, icb, 0, 0, cBlocks, bc, bk), &blocks);

        batchreduce_kernelc(&LIBXSMM_VLA_ACCESS(5, dpB, ikb, 0, 0, 0, 0, nBlocks, bn_lp, bk, lpb),
                            &LIBXSMM_VLA_ACCESS(2, xT, ic, 0, N),
                            &LIBXSMM_VLA_ACCESS(4, dwo, ikb, icb, 0, 0, cBlocks, bc, bk), &blocks);
      }
    }
#ifdef PROFILE
    if (ltid == 0) {
      _end = _rdtsc();
      dwdr_cycles += _end - _start;
    }
#endif

#ifdef PROFILE
    if (ltid == 0) _start = _rdtsc();
#endif
    /* gradient bias */
#if defined(LIBXSMM_RNN_CELL_AVX512)
    if (bcbk_multiples_of_16) {
      for (ik = k_thr_begin; ik < k_thr_end; ik += 16) {
        dbi_sum = LIBXSMM_INTRINSICS_MM512_LOAD_PS(&dbi[ik]);
        dbf_sum = LIBXSMM_INTRINSICS_MM512_LOAD_PS(&dbf[ik]);
        dbo_sum = LIBXSMM_INTRINSICS_MM512_LOAD_PS(&dbo[ik]);
        dbc_sum = LIBXSMM_INTRINSICS_MM512_LOAD_PS(&dbc[ik]);
        for (in = 0; in < N; in++) {
          dbi_sum = _mm512_add_ps(dbi_sum, LIBXSMM_INTRINSICS_MM512_CVTPBH_PS(_mm256_loadu_si256((__m256i*)&LIBXSMM_VLA_ACCESS(2, di,  in, ik, K))));
          dbf_sum = _mm512_add_ps(dbf_sum, LIBXSMM_INTRINSICS_MM512_CVTPBH_PS(_mm256_loadu_si256((__m256i*)&LIBXSMM_VLA_ACCESS(2, df,  in, ik, K))));
          dbo_sum = _mm512_add_ps(dbo_sum, LIBXSMM_INTRINSICS_MM512_CVTPBH_PS(_mm256_loadu_si256((__m256i*)&LIBXSMM_VLA_ACCESS(2, dp,  in, ik, K))));
          dbc_sum = _mm512_add_ps(dbc_sum, LIBXSMM_INTRINSICS_MM512_CVTPBH_PS(_mm256_loadu_si256((__m256i*)&LIBXSMM_VLA_ACCESS(2, dci,  in, ik, K))));
        }
        _mm512_storeu_ps(&dbi[ik], dbi_sum);
        _mm512_storeu_ps(&dbf[ik], dbf_sum);
        _mm512_storeu_ps(&dbo[ik], dbo_sum);
        _mm512_storeu_ps(&dbc[ik], dbc_sum);
        /* Downconvert delta bias to bf16 if done with all timesteps */
        if (j == 0) {
          _mm256_storeu_si256((__m256i*)&dbi_bf16[ik], LIBXSMM_INTRINSISCS_MM512_CVTNEPS_PBH(dbi_sum));
          _mm256_storeu_si256((__m256i*)&dbf_bf16[ik], LIBXSMM_INTRINSISCS_MM512_CVTNEPS_PBH(dbf_sum));
          _mm256_storeu_si256((__m256i*)&dbo_bf16[ik], LIBXSMM_INTRINSISCS_MM512_CVTNEPS_PBH(dbo_sum));
          _mm256_storeu_si256((__m256i*)&dbc_bf16[ik], LIBXSMM_INTRINSISCS_MM512_CVTNEPS_PBH(dbc_sum));
        }
      }
    } else {
      /* TODO: Add alternative path here  */
    }
#else
    /* TODO: Add alternative path here  */
#endif
#ifdef PROFILE
    if (ltid == 0) {
      _end = _rdtsc();
      gradient_cycles += _end - _start;
    }
#endif
  }
  libxsmm_barrier_wait(handle->barrier, (int)ltid);
}

#undef NATIVE_MATRIX_RNE_CVT_FP32_BFP16_LD

