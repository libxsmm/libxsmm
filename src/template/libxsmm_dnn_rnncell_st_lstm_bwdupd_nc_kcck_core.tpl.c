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
      if (K % 2048 != 0 || LIBXSMM_DNN_COMPUTE_KIND_BWD == kind) {
#include "libxsmm_internal_lstm_bwdupd_fused_eltwise.tpl.c"
      } else {
        /* Also reformat di, dci, df and dp to be used in the UPD pass in blocked format ... */
#include "libxsmm_internal_lstm_bwdupd_fused_eltwise_reformat.tpl.c"
      }
    } else {
      /* compute dhp */
      if (j == t-1) {
        libxsmm_internal_matrix_copy_ld(  bk, bn, K, &LIBXSMM_VLA_ACCESS(3, dh, t-1, in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, dout, in, ik, K) );
      } else {
        libxsmm_internal_matrix_add_ld(   bk, bn, K, &LIBXSMM_VLA_ACCESS(2, dout, in, ik, K), &LIBXSMM_VLA_ACCESS(3, dh, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, dout, in, ik, K) );
      }
      /* compute dcp */
      libxsmm_internal_matrix_eltwise_mult_ld(      bk, bn, K, &LIBXSMM_VLA_ACCESS(2, dout, in, ik, K), &LIBXSMM_VLA_ACCESS(3, o, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, t1, in, ik, K) );
      libxsmm_internal_matrix_complement_square_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, co, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, t2, in, ik, K) );
      libxsmm_internal_matrix_eltwise_mult_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(2, t1, in, ik, K), &LIBXSMM_VLA_ACCESS(2, t2, in, ik, K), &LIBXSMM_VLA_ACCESS(2, t1, in, ik, K) );
      if (j == t-1) {
        libxsmm_internal_matrix_add_ld(        bk, bn, K, &LIBXSMM_VLA_ACCESS(2, dcs, in, ik, K), &LIBXSMM_VLA_ACCESS(2, t1, in, ik, K), &LIBXSMM_VLA_ACCESS(2, dcp, in, ik, K) );
      } else {
        libxsmm_internal_matrix_add_ld(        bk, bn, K, &LIBXSMM_VLA_ACCESS(2, dcp, in, ik, K), &LIBXSMM_VLA_ACCESS(2, t1, in, ik, K), &LIBXSMM_VLA_ACCESS(2, dcp, in, ik, K) );
      }
      /* compute dci */
      libxsmm_internal_matrix_eltwise_mult_ld(      bk, bn, K, &LIBXSMM_VLA_ACCESS(2, dcp, in, ik, K), &LIBXSMM_VLA_ACCESS(3, i, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, t1, in, ik, K) );
      libxsmm_internal_matrix_complement_square_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, ci, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, t2, in, ik, K) );
      libxsmm_internal_matrix_eltwise_mult_ld(      bk, bn, K, &LIBXSMM_VLA_ACCESS(2, t1, in, ik, K), &LIBXSMM_VLA_ACCESS(2, t2, in, ik, K), &LIBXSMM_VLA_ACCESS(2, dci, in, ik, K) );
      /* compute di */
      libxsmm_internal_matrix_eltwise_mult_ld(      bk, bn, K, &LIBXSMM_VLA_ACCESS(2, dcp, in, ik, K), &LIBXSMM_VLA_ACCESS(3, ci, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, t1, in, ik, K) );
      libxsmm_internal_matrix_complement_ld(        bk, bn, K, &LIBXSMM_VLA_ACCESS(3, i, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, t2, in, ik, K) );
      libxsmm_internal_matrix_eltwise_mult_ld(      bk, bn, K, &LIBXSMM_VLA_ACCESS(3, i, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, t2, in, ik, K), &LIBXSMM_VLA_ACCESS(2, di, in, ik, K) );
      libxsmm_internal_matrix_eltwise_mult_ld(      bk, bn, K, &LIBXSMM_VLA_ACCESS(2, t1, in, ik, K), &LIBXSMM_VLA_ACCESS(2, di, in, ik, K), &LIBXSMM_VLA_ACCESS(2, di, in, ik, K) );
      /* compute df */
      if (j == 0) {
        libxsmm_internal_matrix_eltwise_mult_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(2, dcp, in, ik, K), &LIBXSMM_VLA_ACCESS(2, cp, in, ik, K), &LIBXSMM_VLA_ACCESS(2, t1, in, ik, K) );
      } else {
        libxsmm_internal_matrix_eltwise_mult_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(2, dcp, in, ik, K), &LIBXSMM_VLA_ACCESS(3, cs, j-1, in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, t1, in, ik, K) );
      }
      libxsmm_internal_matrix_complement_ld(   bk, bn, K, &LIBXSMM_VLA_ACCESS(3, f, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, t2, in, ik, K) );
      libxsmm_internal_matrix_eltwise_mult_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, f, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, t2, in, ik, K), &LIBXSMM_VLA_ACCESS(2, df, in, ik, K) );
      libxsmm_internal_matrix_eltwise_mult_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(2, t1, in, ik, K), &LIBXSMM_VLA_ACCESS(2, df, in, ik, K), &LIBXSMM_VLA_ACCESS(2, df, in, ik, K) );
      /* compute dp */
      libxsmm_internal_matrix_eltwise_mult_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(2, dout, in, ik, K), &LIBXSMM_VLA_ACCESS(3, co, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, t1, in, ik, K) );
      libxsmm_internal_matrix_complement_ld(   bk, bn, K, &LIBXSMM_VLA_ACCESS(3, o, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, t2, in, ik, K) );
      libxsmm_internal_matrix_eltwise_mult_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, o, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, t2, in, ik, K), &LIBXSMM_VLA_ACCESS(2, t2, in, ik, K) );
      libxsmm_internal_matrix_eltwise_mult_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(2, t1, in, ik, K), &LIBXSMM_VLA_ACCESS(2, t2, in, ik, K), &LIBXSMM_VLA_ACCESS(2, dp, in, ik, K) );
      /* update dcp */
      libxsmm_internal_matrix_eltwise_mult_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, f, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, dcp, in, ik, K), &LIBXSMM_VLA_ACCESS(2, dcp, in, ik, K) );
    }
#else
    /* compute dhp */
    if (j == t-1) {
      libxsmm_internal_matrix_copy_ld(  bk, bn, K, &LIBXSMM_VLA_ACCESS(3, dh, t-1, in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, dout, in, ik, K) );
    } else {
      libxsmm_internal_matrix_add_ld(   bk, bn, K, &LIBXSMM_VLA_ACCESS(2, dout, in, ik, K), &LIBXSMM_VLA_ACCESS(3, dh, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, dout, in, ik, K) );
    }
    /* compute dcp */
    libxsmm_internal_matrix_eltwise_mult_ld(      bk, bn, K, &LIBXSMM_VLA_ACCESS(2, dout, in, ik, K), &LIBXSMM_VLA_ACCESS(3, o, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, t1, in, ik, K) );
    libxsmm_internal_matrix_complement_square_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, co, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, t2, in, ik, K) );
    libxsmm_internal_matrix_eltwise_mult_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(2, t1, in, ik, K), &LIBXSMM_VLA_ACCESS(2, t2, in, ik, K), &LIBXSMM_VLA_ACCESS(2, t1, in, ik, K) );
    if (j == t-1) {
      libxsmm_internal_matrix_add_ld(        bk, bn, K, &LIBXSMM_VLA_ACCESS(2, dcs, in, ik, K), &LIBXSMM_VLA_ACCESS(2, t1, in, ik, K), &LIBXSMM_VLA_ACCESS(2, dcp, in, ik, K) );
    } else {
      libxsmm_internal_matrix_add_ld(        bk, bn, K, &LIBXSMM_VLA_ACCESS(2, dcp, in, ik, K), &LIBXSMM_VLA_ACCESS(2, t1, in, ik, K), &LIBXSMM_VLA_ACCESS(2, dcp, in, ik, K) );
    }
    /* compute dci */
    libxsmm_internal_matrix_eltwise_mult_ld(      bk, bn, K, &LIBXSMM_VLA_ACCESS(2, dcp, in, ik, K), &LIBXSMM_VLA_ACCESS(3, i, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, t1, in, ik, K) );
    libxsmm_internal_matrix_complement_square_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, ci, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, t2, in, ik, K) );
    libxsmm_internal_matrix_eltwise_mult_ld(      bk, bn, K, &LIBXSMM_VLA_ACCESS(2, t1, in, ik, K), &LIBXSMM_VLA_ACCESS(2, t2, in, ik, K), &LIBXSMM_VLA_ACCESS(2, dci, in, ik, K) );
    /* compute di */
    libxsmm_internal_matrix_eltwise_mult_ld(      bk, bn, K, &LIBXSMM_VLA_ACCESS(2, dcp, in, ik, K), &LIBXSMM_VLA_ACCESS(3, ci, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, t1, in, ik, K) );
    libxsmm_internal_matrix_complement_ld(        bk, bn, K, &LIBXSMM_VLA_ACCESS(3, i, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, t2, in, ik, K) );
    libxsmm_internal_matrix_eltwise_mult_ld(      bk, bn, K, &LIBXSMM_VLA_ACCESS(3, i, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, t2, in, ik, K), &LIBXSMM_VLA_ACCESS(2, di, in, ik, K) );
    libxsmm_internal_matrix_eltwise_mult_ld(      bk, bn, K, &LIBXSMM_VLA_ACCESS(2, t1, in, ik, K), &LIBXSMM_VLA_ACCESS(2, di, in, ik, K), &LIBXSMM_VLA_ACCESS(2, di, in, ik, K) );
    /* compute df */
    if (j == 0) {
      libxsmm_internal_matrix_eltwise_mult_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(2, dcp, in, ik, K), &LIBXSMM_VLA_ACCESS(2, cp, in, ik, K), &LIBXSMM_VLA_ACCESS(2, t1, in, ik, K) );
    } else {
      libxsmm_internal_matrix_eltwise_mult_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(2, dcp, in, ik, K), &LIBXSMM_VLA_ACCESS(3, cs, j-1, in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, t1, in, ik, K) );
    }
    libxsmm_internal_matrix_complement_ld(   bk, bn, K, &LIBXSMM_VLA_ACCESS(3, f, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, t2, in, ik, K) );
    libxsmm_internal_matrix_eltwise_mult_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, f, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, t2, in, ik, K), &LIBXSMM_VLA_ACCESS(2, df, in, ik, K) );
    libxsmm_internal_matrix_eltwise_mult_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(2, t1, in, ik, K), &LIBXSMM_VLA_ACCESS(2, df, in, ik, K), &LIBXSMM_VLA_ACCESS(2, df, in, ik, K) );
    /* compute dp */
    libxsmm_internal_matrix_eltwise_mult_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(2, dout, in, ik, K), &LIBXSMM_VLA_ACCESS(3, co, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, t1, in, ik, K) );
    libxsmm_internal_matrix_complement_ld(   bk, bn, K, &LIBXSMM_VLA_ACCESS(3, o, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, t2, in, ik, K) );
    libxsmm_internal_matrix_eltwise_mult_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, o, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, t2, in, ik, K), &LIBXSMM_VLA_ACCESS(2, t2, in, ik, K) );
    libxsmm_internal_matrix_eltwise_mult_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(2, t1, in, ik, K), &LIBXSMM_VLA_ACCESS(2, t2, in, ik, K), &LIBXSMM_VLA_ACCESS(2, dp, in, ik, K) );
    /* update dcp */
    libxsmm_internal_matrix_eltwise_mult_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, f, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, dcp, in, ik, K), &LIBXSMM_VLA_ACCESS(2, dcp, in, ik, K) );
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
    for (KB = 0; KB < BF; KB++) {
      for (inic = thr_begin_nc; inic < thr_end_nc; ++inic ) {
        in = (inic % (N/bn))*bn;
        icb = inic / (N/bn);
        ic = icb*bc;

        for (ik = 0, ikb = 0; ikb < KB_BLOCKS; ik += bk, ikb++) {
          A_array[ikb] = &LIBXSMM_VLA_ACCESS(4, wiT, icb, ikb + KB*KB_BLOCKS, 0, 0, kBlocks, bk, bc);
          B_array[ikb] = &LIBXSMM_VLA_ACCESS(2, di,  in, ik + KB*KB_BLOCKS*bk, K);
        }
        /* Reduce batch gemm call  */
        blocks = KB_BLOCKS;
        batchreduce_kernela(A_array, B_array, &LIBXSMM_VLA_ACCESS(3, dx, j, in, ic, N, C)  , &blocks);

        for (ik = 0, ikb = 0; ikb < KB_BLOCKS; ik += bk, ikb++) {
          A_array[ikb] = &LIBXSMM_VLA_ACCESS(4, wcT, icb, ikb + KB*KB_BLOCKS, 0, 0, kBlocks, bk, bc);
          B_array[ikb] = &LIBXSMM_VLA_ACCESS(2, dci,  in, ik + KB*KB_BLOCKS*bk, K);
        }
        /* Reduce batch gemm call  */
        batchreduce_kernela(A_array, B_array, &LIBXSMM_VLA_ACCESS(3, dx, j, in, ic, N, C)  , &blocks);

        for (ik = 0, ikb = 0; ikb < KB_BLOCKS; ik += bk, ikb++) {
          A_array[ikb] = &LIBXSMM_VLA_ACCESS(4, wfT, icb, ikb + KB*KB_BLOCKS, 0, 0, kBlocks, bk, bc);
          B_array[ikb] = &LIBXSMM_VLA_ACCESS(2, df,  in, ik + KB*KB_BLOCKS*bk, K);
        }
        /* Reduce batch gemm call  */
        batchreduce_kernela(A_array, B_array, &LIBXSMM_VLA_ACCESS(3, dx, j, in, ic, N, C)  , &blocks);

        for (ik = 0, ikb = 0; ikb < KB_BLOCKS; ik += bk, ikb++) {
          A_array[ikb] = &LIBXSMM_VLA_ACCESS(4, woT, icb, ikb + KB*KB_BLOCKS, 0, 0, kBlocks, bk, bc);
          B_array[ikb] = &LIBXSMM_VLA_ACCESS(2, dp,  in, ik + KB*KB_BLOCKS*bk, K);
        }
        /* Reduce batch gemm call  */
        batchreduce_kernela(A_array, B_array, &LIBXSMM_VLA_ACCESS(3, dx, j, in, ic, N, C)  , &blocks);
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
  for (KB = 0; KB < BF; KB++) {
    for (inik = thr_begin_nk; inik < thr_end_nk; ++inik ) {
      in = (inik % (N/bn))*bn;
      ikb = inik / (N/bn);
      ik = ikb*bk;

      dout_ptr = (j > 0) ? (element_output_type*) &LIBXSMM_VLA_ACCESS(2, dout, in, ik, K) : (element_output_type*) &LIBXSMM_VLA_ACCESS(2, dhp, in, ik, K);

      if (KB == 0) libxsmm_internal_matrix_zero_ld( bk, bn, K, dout_ptr);
      /* dout += R^T * difoc */
      for (ic = 0, icb = 0; icb < KB_BLOCKS; ic += bk, icb++) {
        A_array[icb] = &LIBXSMM_VLA_ACCESS(4, riT, ikb, icb + KB*KB_BLOCKS, 0, 0, kBlocks, bk, bk);
        B_array[icb] = &LIBXSMM_VLA_ACCESS(2, di,  in, ic + KB*KB_BLOCKS*bk, K);
      }
      /* Reduce batch gemm call  */
      blocks = KB_BLOCKS;
      batchreduce_kerneld(A_array, B_array, dout_ptr, &blocks);

      for (ic = 0, icb = 0; icb < KB_BLOCKS; ic += bk, icb++) {
        A_array[icb] = &LIBXSMM_VLA_ACCESS(4, rcT, ikb, icb + KB*KB_BLOCKS, 0, 0, kBlocks, bk, bk);
        B_array[icb] = &LIBXSMM_VLA_ACCESS(2, dci,  in, ic + KB*KB_BLOCKS*bk, K);
      }
      /* Reduce batch gemm call  */
      batchreduce_kerneld(A_array, B_array, dout_ptr, &blocks);

      for (ic = 0, icb = 0; icb < KB_BLOCKS; ic += bk, icb++) {
        A_array[icb] = &LIBXSMM_VLA_ACCESS(4, rfT, ikb, icb + KB*KB_BLOCKS, 0, 0, kBlocks, bk, bk);
        B_array[icb] = &LIBXSMM_VLA_ACCESS(2, df,  in, ic + KB*KB_BLOCKS*bk, K);
      }
      /* Reduce batch gemm call  */
      batchreduce_kerneld(A_array, B_array, dout_ptr, &blocks);

      for (ic = 0, icb = 0; icb < KB_BLOCKS; ic += bk, icb++) {
        A_array[icb] = &LIBXSMM_VLA_ACCESS(4, roT, ikb, icb + KB*KB_BLOCKS, 0, 0, kBlocks, bk, bk);
        B_array[icb] = &LIBXSMM_VLA_ACCESS(2, dp,  in, ic + KB*KB_BLOCKS*bk, K);
      }
      /* Reduce batch gemm call  */
      batchreduce_kerneld(A_array, B_array, dout_ptr, &blocks);
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
    if ((C == K) && (bc == bk) && (bcbk_multiples_of_16 == 1)) {
#if 0
      if (K % 2048 != 0) {
#endif
        /* Interleave computation of dr = difoc * h^T and dw = difoc * x^T to take advantage of temporal locality */
        for (ikic = thr_begin_kk; ikic < thr_end_kk; ++ikic ) {
          icb = ikic / (K/bk);
          ic = icb*bk;
          ikb = ikic % (K/bk);
          ik = ikb*bk;
          blocks = nBlocks;

          for (in = 0, inb = 0; in < N; in += bn, inb++) {
            A_array[inb] = &LIBXSMM_VLA_ACCESS(2, di,  in, ik, K);
            B_array[inb] = &LIBXSMM_VLA_ACCESS(2, hT, ic, in, N);
          }
          batchreduce_kernelb1(A_array, B_array, &LIBXSMM_VLA_ACCESS(4, dri, ikb, icb, 0, 0, kBlocks, bk, bk), &blocks);

          for (in = 0, inb = 0; in < N; in += bn, inb++) {
            A_array[inb] = &LIBXSMM_VLA_ACCESS(2, di,  in, ik, K);
            B_array[inb] = &LIBXSMM_VLA_ACCESS(2, xT, ic, in, N);
          }
          batchreduce_kernelc1(A_array, B_array, &LIBXSMM_VLA_ACCESS(4, dwi, ikb, icb, 0, 0, cBlocks, bc, bk), &blocks);

          for (in = 0, inb = 0; in < N; in += bn, inb++) {
            A_array[inb] = &LIBXSMM_VLA_ACCESS(2, dci, in, ik, K);
            B_array[inb] = &LIBXSMM_VLA_ACCESS(2, hT, ic, in, N);
          }
          batchreduce_kernelb1(A_array, B_array, &LIBXSMM_VLA_ACCESS(4, drc, ikb, icb, 0, 0, kBlocks, bk, bk), &blocks);

          for (in = 0, inb = 0; in < N; in += bn, inb++) {
            A_array[inb] = &LIBXSMM_VLA_ACCESS(2, dci,  in, ik, K);
            B_array[inb] = &LIBXSMM_VLA_ACCESS(2, xT, ic, in, N);
          }
          batchreduce_kernelc1(A_array, B_array, &LIBXSMM_VLA_ACCESS(4, dwc, ikb, icb, 0, 0, cBlocks, bc, bk), &blocks);

          for (in = 0, inb = 0; in < N; in += bn, inb++) {
            A_array[inb] = &LIBXSMM_VLA_ACCESS(2, df,  in, ik, K);
            B_array[inb] = &LIBXSMM_VLA_ACCESS(2, hT, ic, in, N);
          }
          batchreduce_kernelb1(A_array, B_array, &LIBXSMM_VLA_ACCESS(4, drf, ikb, icb, 0, 0, kBlocks, bk, bk), &blocks);

          for (in = 0, inb = 0; in < N; in += bn, inb++) {
            A_array[inb] = &LIBXSMM_VLA_ACCESS(2, df,  in, ik, K);
            B_array[inb] = &LIBXSMM_VLA_ACCESS(2, xT, ic, in, N);
          }
          batchreduce_kernelc1(A_array, B_array, &LIBXSMM_VLA_ACCESS(4, dwf, ikb, icb, 0, 0, cBlocks, bc, bk), &blocks);

          for (in = 0, inb = 0; in < N; in += bn, inb++) {
            A_array[inb] = &LIBXSMM_VLA_ACCESS(2, dp,  in, ik, K);
            B_array[inb] = &LIBXSMM_VLA_ACCESS(2, hT, ic, in, N);
          }
          batchreduce_kernelb1(A_array, B_array, &LIBXSMM_VLA_ACCESS(4, dro, ikb, icb, 0, 0, kBlocks, bk, bk), &blocks);

          for (in = 0, inb = 0; in < N; in += bn, inb++) {
            A_array[inb] = &LIBXSMM_VLA_ACCESS(2, dp,  in, ik, K);
            B_array[inb] = &LIBXSMM_VLA_ACCESS(2, xT, ic, in, N);
          }
          batchreduce_kernelc1(A_array, B_array, &LIBXSMM_VLA_ACCESS(4, dwo, ikb, icb, 0, 0, cBlocks, bc, bk), &blocks);
        }
        LIBXSMM_UNUSED(diB_);
        LIBXSMM_UNUSED(dciB_);
        LIBXSMM_UNUSED(dfB_);
        LIBXSMM_UNUSED(dpB_);
        LIBXSMM_UNUSED(batchreduce_kernelc);
        LIBXSMM_UNUSED(batchreduce_kernelb);
#if 0
      } else {
        /* Interleave computation of dr = difoc * h^T and dw = difoc * x^T to take advantage of temporal locality */
        /* Use blocked format for di, dci, df and dp */
        for (ikic = thr_begin_kk; ikic < thr_end_kk; ++ikic ) {
          icb = ikic / (K/bk);
          ic = icb*bk;
          ikb = ikic % (K/bk);
          ik = ikb*bk;
          blocks = nBlocks;

          for (in = 0, inb = 0; in < N; in += bn, inb++) {
            A_array[inb] = &LIBXSMM_VLA_ACCESS(4, diB, inb, ikb, 0, 0, kBlocks, bn, bk);
            B_array[inb] = &LIBXSMM_VLA_ACCESS(2, hT, ic, in, N);
          }
          batchreduce_kernelb(A_array, B_array, &LIBXSMM_VLA_ACCESS(4, dri, ikb, icb, 0, 0, kBlocks, bk, bk), &blocks);

          for (in = 0, inb = 0; in < N; in += bn, inb++) {
            A_array[inb] = &LIBXSMM_VLA_ACCESS(4, diB, inb, ikb, 0, 0, kBlocks, bn, bk);
            B_array[inb] = &LIBXSMM_VLA_ACCESS(2, xT, ic, in, N);
          }
          batchreduce_kernelc(A_array, B_array, &LIBXSMM_VLA_ACCESS(4, dwi, ikb, icb, 0, 0, cBlocks, bc, bk), &blocks);

          for (in = 0, inb = 0; in < N; in += bn, inb++) {
            A_array[inb] = &LIBXSMM_VLA_ACCESS(4, dciB, inb, ikb, 0, 0, kBlocks, bn, bk);
            B_array[inb] = &LIBXSMM_VLA_ACCESS(2, hT, ic, in, N);
          }
          batchreduce_kernelb(A_array, B_array, &LIBXSMM_VLA_ACCESS(4, drc, ikb, icb, 0, 0, kBlocks, bk, bk), &blocks);

          for (in = 0, inb = 0; in < N; in += bn, inb++) {
            A_array[inb] = &LIBXSMM_VLA_ACCESS(4, dciB, inb, ikb, 0, 0, kBlocks, bn, bk);
            B_array[inb] = &LIBXSMM_VLA_ACCESS(2, xT, ic, in, N);
          }
          batchreduce_kernelc(A_array, B_array, &LIBXSMM_VLA_ACCESS(4, dwc, ikb, icb, 0, 0, cBlocks, bc, bk), &blocks);

          for (in = 0, inb = 0; in < N; in += bn, inb++) {
            A_array[inb] = &LIBXSMM_VLA_ACCESS(4, dfB, inb, ikb, 0, 0, kBlocks, bn, bk);
            B_array[inb] = &LIBXSMM_VLA_ACCESS(2, hT, ic, in, N);
          }
          batchreduce_kernelb(A_array, B_array, &LIBXSMM_VLA_ACCESS(4, drf, ikb, icb, 0, 0, kBlocks, bk, bk), &blocks);

          for (in = 0, inb = 0; in < N; in += bn, inb++) {
            A_array[inb] = &LIBXSMM_VLA_ACCESS(4, dfB, inb, ikb, 0, 0, kBlocks, bn, bk);
            B_array[inb] = &LIBXSMM_VLA_ACCESS(2, xT, ic, in, N);
          }
          batchreduce_kernelc(A_array, B_array, &LIBXSMM_VLA_ACCESS(4, dwf, ikb, icb, 0, 0, cBlocks, bc, bk), &blocks);

          for (in = 0, inb = 0; in < N; in += bn, inb++) {
            A_array[inb] = &LIBXSMM_VLA_ACCESS(4, dpB, inb, ikb, 0, 0, kBlocks, bn, bk);
            B_array[inb] = &LIBXSMM_VLA_ACCESS(2, hT, ic, in, N);
          }
          batchreduce_kernelb(A_array, B_array, &LIBXSMM_VLA_ACCESS(4, dro, ikb, icb, 0, 0, kBlocks, bk, bk), &blocks);

          for (in = 0, inb = 0; in < N; in += bn, inb++) {
            A_array[inb] = &LIBXSMM_VLA_ACCESS(4, dpB, inb, ikb, 0, 0, kBlocks, bn, bk);
            B_array[inb] = &LIBXSMM_VLA_ACCESS(2, xT, ic, in, N);
          }
          batchreduce_kernelc(A_array, B_array, &LIBXSMM_VLA_ACCESS(4, dwo, ikb, icb, 0, 0, cBlocks, bc, bk), &blocks);
        }
      }
#endif
    } else {
      /* dr = difoc * h^T */
      for (ikic = thr_begin_kk; ikic < thr_end_kk; ++ikic ) {
        icb = ikic / (K/bk);
        ic = icb*bk;
        ikb = ikic % (K/bk);
        ik = ikb*bk;

        for (in = 0, inb = 0; in < N; in += bn, inb++) {
          A_array[inb] = &LIBXSMM_VLA_ACCESS(2, di,  in, ik, K);
          B_array[inb] = &LIBXSMM_VLA_ACCESS(2, hT, ic, in, N);
        }
        blocks = nBlocks;
        batchreduce_kernelb1(A_array, B_array, &LIBXSMM_VLA_ACCESS(4, dri, ikb, icb, 0, 0, kBlocks, bk, bk), &blocks);

        for (in = 0, inb = 0; in < N; in += bn, inb++) {
          A_array[inb] = &LIBXSMM_VLA_ACCESS(2, dci, in, ik, K);
          B_array[inb] = &LIBXSMM_VLA_ACCESS(2, hT, ic, in, N);
        }
        batchreduce_kernelb1(A_array, B_array, &LIBXSMM_VLA_ACCESS(4, drc, ikb, icb, 0, 0, kBlocks, bk, bk), &blocks);

        for (in = 0, inb = 0; in < N; in += bn, inb++) {
          A_array[inb] = &LIBXSMM_VLA_ACCESS(2, df,  in, ik, K);
          B_array[inb] = &LIBXSMM_VLA_ACCESS(2, hT, ic, in, N);
        }
        batchreduce_kernelb1(A_array, B_array, &LIBXSMM_VLA_ACCESS(4, drf, ikb, icb, 0, 0, kBlocks, bk, bk), &blocks);

        for (in = 0, inb = 0; in < N; in += bn, inb++) {
          A_array[inb] = &LIBXSMM_VLA_ACCESS(2, dp,  in, ik, K);
          B_array[inb] = &LIBXSMM_VLA_ACCESS(2, hT, ic, in, N);
        }
        batchreduce_kernelb1(A_array, B_array, &LIBXSMM_VLA_ACCESS(4, dro, ikb, icb, 0, 0, kBlocks, bk, bk), &blocks);
      }

      /* dw = difoc * x^T */
      for (ikic = thr_begin_ck; ikic < thr_end_ck; ++ikic ) {
        icb = ikic / (K/bk);
        ic = icb*bc;
        ikb = ikic % (K/bk);
        ik = ikb*bk;

        for (in = 0, inb = 0; in < N; in += bn, inb++) {
          A_array[inb] = &LIBXSMM_VLA_ACCESS(2, di,  in, ik, K);
          B_array[inb] = &LIBXSMM_VLA_ACCESS(2, xT, ic, in, N);
        }
        blocks = nBlocks;
        batchreduce_kernelc1(A_array, B_array, &LIBXSMM_VLA_ACCESS(4, dwi, ikb, icb, 0, 0, cBlocks, bc, bk), &blocks);

        for (in = 0, inb = 0; in < N; in += bn, inb++) {
          A_array[inb] = &LIBXSMM_VLA_ACCESS(2, dci,  in, ik, K);
          B_array[inb] = &LIBXSMM_VLA_ACCESS(2, xT, ic, in, N);
        }
        batchreduce_kernelc1(A_array, B_array, &LIBXSMM_VLA_ACCESS(4, dwc, ikb, icb, 0, 0, cBlocks, bc, bk), &blocks);

        for (in = 0, inb = 0; in < N; in += bn, inb++) {
          A_array[inb] = &LIBXSMM_VLA_ACCESS(2, df,  in, ik, K);
          B_array[inb] = &LIBXSMM_VLA_ACCESS(2, xT, ic, in, N);
        }
        batchreduce_kernelc1(A_array, B_array, &LIBXSMM_VLA_ACCESS(4, dwf, ikb, icb, 0, 0, cBlocks, bc, bk), &blocks);

        for (in = 0, inb = 0; in < N; in += bn, inb++) {
          A_array[inb] = &LIBXSMM_VLA_ACCESS(2, dp,  in, ik, K);
          B_array[inb] = &LIBXSMM_VLA_ACCESS(2, xT, ic, in, N);
        }
        batchreduce_kernelc1(A_array, B_array, &LIBXSMM_VLA_ACCESS(4, dwo, ikb, icb, 0, 0, cBlocks, bc, bk), &blocks);
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
          dbi_sum = _mm512_add_ps(dbi_sum, LIBXSMM_INTRINSICS_MM512_LOAD_PS(&LIBXSMM_VLA_ACCESS(2, di,  in, ik, K)));
          dbf_sum = _mm512_add_ps(dbf_sum, LIBXSMM_INTRINSICS_MM512_LOAD_PS(&LIBXSMM_VLA_ACCESS(2, df,  in, ik, K)));
          dbo_sum = _mm512_add_ps(dbo_sum, LIBXSMM_INTRINSICS_MM512_LOAD_PS(&LIBXSMM_VLA_ACCESS(2, dp,  in, ik, K)));
          dbc_sum = _mm512_add_ps(dbc_sum, LIBXSMM_INTRINSICS_MM512_LOAD_PS(&LIBXSMM_VLA_ACCESS(2, dci,  in, ik, K)));
        }
        _mm512_storeu_ps(&dbi[ik], dbi_sum);
        _mm512_storeu_ps(&dbf[ik], dbf_sum);
        _mm512_storeu_ps(&dbo[ik], dbo_sum);
        _mm512_storeu_ps(&dbc[ik], dbc_sum);
      }
    } else {
      for (ik = thr_begin_k; ik < thr_end_k; ik++) {
        for (in = 0; in < N; in++) {
          dbi[ik] += LIBXSMM_VLA_ACCESS(2, di,  in, ik, K);
          dbf[ik] += LIBXSMM_VLA_ACCESS(2, df,  in, ik, K);
          dbo[ik] += LIBXSMM_VLA_ACCESS(2, dp,  in, ik, K);
          dbc[ik] += LIBXSMM_VLA_ACCESS(2, dci, in, ik, K);
        }
      }
    }
#else
    for (ik = thr_begin_k; ik < thr_end_k; ik++) {
      for (in = 0; in < N; in++) {
        dbi[ik] += LIBXSMM_VLA_ACCESS(2, di,  in, ik, K);
        dbf[ik] += LIBXSMM_VLA_ACCESS(2, df,  in, ik, K);
        dbo[ik] += LIBXSMM_VLA_ACCESS(2, dp,  in, ik, K);
        dbc[ik] += LIBXSMM_VLA_ACCESS(2, dci, in, ik, K);
      }
    }
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

