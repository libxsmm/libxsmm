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
  float *const __src = _src; \
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

for (j = t-1; j >= 0; --j) {
  /* let's run the cell in blocks for good locality */
  for (inik = thr_begin_nk; inik < thr_end_nk; ++inik ) {
    inb = inik % (N/bn);
    ikb = inik / (N/bn);
    in = (inik % (N/bn))*bn;
    ik = (inik / (N/bn))*bk;
    /* Compute dcp, dci, di, df, dp */
    cps_ptr = (j == 0) ? &LIBXSMM_VLA_ACCESS(4, cp, inb, ikb, 0, 0, kBlocks, bn, bk) : &LIBXSMM_VLA_ACCESS(5, cs, j-1, inb, ikb, 0, 0, nBlocks, kBlocks, bn, bk);
    /* Also reformat di, dci, df and dp to be used in the UPD pass in blocked format ... */
#include "libxsmm_internal_lstm_bwdupd_fused_eltwise_ncnc_reformat_bf16.tpl.c"
  }

  if ( (LIBXSMM_DNN_COMPUTE_KIND_UPD == kind) || (LIBXSMM_DNN_COMPUTE_KIND_BWDUPD == kind) ) {
    /* transpose xt for current timestep */
    for (icin = thr_begin_nc; icin < thr_end_nc; ++icin ) {
      inb = icin / (C/bc);
      icb = icin % (C/bc);
      if (bc == 32 && bk == 32) {
        trans_act((short int*)&LIBXSMM_VLA_ACCESS(5, x, j, inb, icb, 0, 0, nBlocks, cBlocks, bn, bc), (short int*)&LIBXSMM_VLA_ACCESS(4, xT, icb, inb, 0, 0, nBlocks, bc, bn));
      } else {
        in = inb*bn;
        for (jc = 0; jc < bc; ++jc) {
          for (jb = 0; jb < bn; ++jb) {
            LIBXSMM_VLA_ACCESS(4, xT, icb, inb, jc, jb, nBlocks, bc, bn) =  LIBXSMM_VLA_ACCESS(5, x, j, inb, icb, jb, jc, nBlocks, cBlocks, bn, bc);
          }
        }
      }
    }

    /* transpose ht for current timestep */
    if (j == 0) {
      for (ikin = thr_begin_nk; ikin < thr_end_nk; ++ikin ) {
        inb = ikin / (K/bk);
        ikb = ikin % (K/bk);
        if (bc == 32 && bk == 32) {
          trans_act((short int*)&LIBXSMM_VLA_ACCESS(4, hp, inb, ikb, 0, 0, kBlocks, bn, bk), (short int*)&LIBXSMM_VLA_ACCESS(4, hT, ikb, inb, 0, 0, nBlocks, bk, bn));
        } else {
          in = inb*bn;
          ik = ikb*bk;
          for (jk = 0; jk < bk; ++jk) {
            for (jb = 0; jb < bn; ++jb) {
              LIBXSMM_VLA_ACCESS(4, hT, ikb, inb, jk, jb, nBlocks, bk, bn) =  LIBXSMM_VLA_ACCESS(4, hp, inb, ikb, jb, jk, kBlocks, bn, bk);
            }
          }
        }
      }
    } else {
      for (ikin = thr_begin_nk; ikin < thr_end_nk; ++ikin ) {
        inb = ikin / (K/bk);
        ikb = ikin % (K/bk);
        if (bc == 32 && bk == 32) {
          trans_act((short int*)&LIBXSMM_VLA_ACCESS(5, h, j-1, inb, ikb, 0, 0, nBlocks, kBlocks, bn, bk), (short int*)&LIBXSMM_VLA_ACCESS(4, hT, ikb, inb, 0, 0, nBlocks, bk, bn));
        } else {
          ik = ikb*bk;
          in = inb*bn;
          for (jk = 0; jk < bk; ++jk) {
            for (jb = 0; jb < bn; ++jb) {
              LIBXSMM_VLA_ACCESS(4, hT, ikb, inb, jk, jb, nBlocks, bk, bn) =  LIBXSMM_VLA_ACCESS(5, h, j-1, inb, ikb, jb, jk, nBlocks, kBlocks, bn, bk);
            }
          }
        }
      }
    }
  }

  libxsmm_barrier_wait(handle->barrier, (int)ltid);

  if ( (LIBXSMM_DNN_COMPUTE_KIND_BWD == kind) || (LIBXSMM_DNN_COMPUTE_KIND_BWDUPD == kind) ) {
    /* dx = W^T * difoc */
    blocks = KB_BLOCKS;
    for (KB = 0; KB < BF; KB++) {
      for (inic = thr_begin_nc; inic < thr_end_nc; ++inic ) {
        inb = inic % (N/bn);
        in = inb*bn;
        icb = inic / (N/bn);

        batchreduce_kernela(&LIBXSMM_VLA_ACCESS(5, wiT, icb, KB*KB_BLOCKS, 0, 0, 0, kBlocks, bk_lp, bc, lpb),
            &LIBXSMM_VLA_ACCESS(4, di, inb, KB*KB_BLOCKS, 0, 0, kBlocks, bn, bk),
            &LIBXSMM_VLA_ACCESS(5, dx, j, inb, icb, 0, 0, nBlocks, cBlocks, bn, bc), &blocks);

        batchreduce_kernela(&LIBXSMM_VLA_ACCESS(5, wcT, icb, KB*KB_BLOCKS, 0, 0, 0, kBlocks, bk_lp, bc, lpb),
            &LIBXSMM_VLA_ACCESS(4, dci, inb, KB*KB_BLOCKS, 0, 0, kBlocks, bn, bk),
            &LIBXSMM_VLA_ACCESS(5, dx, j, inb, icb, 0, 0, nBlocks, cBlocks, bn, bc), &blocks);

        batchreduce_kernela(&LIBXSMM_VLA_ACCESS(5, wfT, icb, KB*KB_BLOCKS, 0, 0, 0, kBlocks, bk_lp, bc, lpb),
            &LIBXSMM_VLA_ACCESS(4, df, inb, KB*KB_BLOCKS, 0, 0, kBlocks, bn, bk),
            &LIBXSMM_VLA_ACCESS(5, dx, j, inb, icb, 0, 0, nBlocks, cBlocks, bn, bc), &blocks);

        batchreduce_kernela(&LIBXSMM_VLA_ACCESS(5, woT, icb, KB*KB_BLOCKS, 0, 0, 0, kBlocks, bk_lp, bc, lpb),
            &LIBXSMM_VLA_ACCESS(4, dp, inb, KB*KB_BLOCKS, 0, 0, kBlocks, bn, bk),
            &LIBXSMM_VLA_ACCESS(5, dx, j, inb, icb, 0, 0, nBlocks, cBlocks, bn, bc), &blocks);

        /* If last block, make sure we downconvert dx to bf16 */
        if (KB == BF-1) {
          NATIVE_MATRIX_RNE_CVT_FP32_BFP16_LD(bc, bn, bc, &LIBXSMM_VLA_ACCESS(5, dx, j, inb, icb, 0, 0, nBlocks, cBlocks, bn, bc), &LIBXSMM_VLA_ACCESS(5, dx_bf16, j, inb, icb, 0, 0, nBlocks, cBlocks, bn, bc));
        }
      }
    }
  }

  blocks = KB_BLOCKS;
  for (KB = 0; KB < BF; KB++) {
    for (inik = thr_begin_nk; inik < thr_end_nk; ++inik ) {
      inb = inik % (N/bn);
      in = inb*bn;
      ikb = inik / (N/bn);
      ik = ikb*bk;
      dout_ptr = (j > 0) ? (float*) &LIBXSMM_VLA_ACCESS(4, dout, inb, ikb, 0, 0, kBlocks, bn, bk) : (float*) &LIBXSMM_VLA_ACCESS(4, dhp_f32, inb, ikb, 0, 0, kBlocks, bn, bk);

      if (KB == 0) libxsmm_internal_matrix_zero_ld( bk, bn, bk, dout_ptr);
      /* dout += R^T * difoc */
      batchreduce_kerneld(&LIBXSMM_VLA_ACCESS(5, riT, ikb, KB*KB_BLOCKS, 0, 0, 0, kBlocks, bk_lp, bk, lpb),
          &LIBXSMM_VLA_ACCESS(4, di, inb, KB*KB_BLOCKS, 0, 0, kBlocks, bn, bk),
          dout_ptr, &blocks);

      batchreduce_kerneld(&LIBXSMM_VLA_ACCESS(5, rcT, ikb, KB*KB_BLOCKS, 0, 0, 0, kBlocks, bk_lp, bk, lpb),
          &LIBXSMM_VLA_ACCESS(4, dci, inb, KB*KB_BLOCKS, 0, 0, kBlocks, bn, bk),
          dout_ptr, &blocks);

      batchreduce_kerneld(&LIBXSMM_VLA_ACCESS(5, rfT, ikb, KB*KB_BLOCKS, 0, 0, 0, kBlocks, bk_lp, bk, lpb),
          &LIBXSMM_VLA_ACCESS(4, df, inb, KB*KB_BLOCKS, 0, 0, kBlocks, bn, bk),
          dout_ptr, &blocks);

      batchreduce_kerneld(&LIBXSMM_VLA_ACCESS(5, roT, ikb, KB*KB_BLOCKS, 0, 0, 0, kBlocks, bk_lp, bk, lpb),
          &LIBXSMM_VLA_ACCESS(4, dp, inb, KB*KB_BLOCKS, 0, 0, kBlocks, bn, bk),
          dout_ptr, &blocks);

      /* Make sure when last and j == 0 to downconvert dhp to BF16  */
      if ((j == 0) && (KB == BF-1)) {
        NATIVE_MATRIX_RNE_CVT_FP32_BFP16_LD(bk, bn, bk, dout_ptr, &LIBXSMM_VLA_ACCESS(4, dhp, inb, ikb, 0, 0, kBlocks, bn, bk));
      }
    }
  }

  if ( (LIBXSMM_DNN_COMPUTE_KIND_UPD == kind) || (LIBXSMM_DNN_COMPUTE_KIND_BWDUPD == kind) ) {
    blocks = nBlocks;
    if ((C == K) && (bc == bk) && (bcbk_multiples_of_16 == 1)) {
      /* Interleave computation of dr = difoc * h^T and dw = difoc * x^T to take advantage of temporal locality */
      /* Use blocked format for di, dci, df and db */
      for (ikic = thr_begin_kk; ikic < thr_end_kk; ++ikic ) {
        icb = ikic / (K/bk);
        ikb = ikic % (K/bk);
        ik = ikb*bk;
        batchreduce_kernelb(&LIBXSMM_VLA_ACCESS(5, diB, ikb, 0, 0, 0, 0, nBlocks, bn_lp, bk, lpb),
            &LIBXSMM_VLA_ACCESS(4, hT, icb, 0, 0, 0, nBlocks, bk, bn),
            &LIBXSMM_VLA_ACCESS(4, dri, ikb, icb, 0, 0, kBlocks, bk, bk), &blocks);
        if (j == 0) {
          for (jc = 0; jc < bk; jc+=2) {
            for (jk = 0; jk < bk; jk+=16) {
              c01 = (__m512i) LIBXSMM_INTRINSISCS_MM512_CVTNE2PS_PBH( LIBXSMM_INTRINSICS_MM512_LOAD_PS(&LIBXSMM_VLA_ACCESS(4, dri, ikb, icb, jc+1, jk, kBlocks, bk, bk)), LIBXSMM_INTRINSICS_MM512_LOAD_PS(&LIBXSMM_VLA_ACCESS(4, dri, ikb, icb, jc, jk, kBlocks, bk, bk)));
              _mm512_store_epi32(&LIBXSMM_VLA_ACCESS(5, dri_bf16, ikb, icb, jc/lpb, jk, 0, kBlocks, bk_lp, bk, lpb), _mm512_permutexvar_epi16(perm_index, c01));
            }
          }
        }

        batchreduce_kernelc(&LIBXSMM_VLA_ACCESS(5, diB, ikb, 0, 0, 0, 0, nBlocks, bn_lp, bk, lpb),
            &LIBXSMM_VLA_ACCESS(4, xT, icb, 0, 0, 0, nBlocks, bc, bn),
            &LIBXSMM_VLA_ACCESS(4, dwi, ikb, icb, 0, 0, cBlocks, bc, bk), &blocks);
        if (j == 0) {
          for (jc = 0; jc < bk; jc+=2) {
            for (jk = 0; jk < bk; jk+=16) {
              c01 = (__m512i) LIBXSMM_INTRINSISCS_MM512_CVTNE2PS_PBH( LIBXSMM_INTRINSICS_MM512_LOAD_PS(&LIBXSMM_VLA_ACCESS(4, dwi, ikb, icb, jc+1, jk, cBlocks, bc, bk)), LIBXSMM_INTRINSICS_MM512_LOAD_PS(&LIBXSMM_VLA_ACCESS(4, dwi, ikb, icb, jc, jk, cBlocks, bc, bk)));
              _mm512_store_epi32(&LIBXSMM_VLA_ACCESS(5, dwi_bf16, ikb, icb, jc/lpb, jk, 0, cBlocks, bc_lp, bk, lpb), _mm512_permutexvar_epi16(perm_index, c01));
            }
          }
        }

        batchreduce_kernelb(&LIBXSMM_VLA_ACCESS(5, dciB, ikb, 0, 0, 0, 0, nBlocks, bn_lp, bk, lpb),
            &LIBXSMM_VLA_ACCESS(4, hT, icb, 0, 0, 0, nBlocks, bk, bn),
            &LIBXSMM_VLA_ACCESS(4, drc, ikb, icb, 0, 0, kBlocks, bk, bk), &blocks);
        if (j == 0) {
          for (jc = 0; jc < bk; jc+=2) {
            for (jk = 0; jk < bk; jk+=16) {
              c01 = (__m512i) LIBXSMM_INTRINSISCS_MM512_CVTNE2PS_PBH( LIBXSMM_INTRINSICS_MM512_LOAD_PS(&LIBXSMM_VLA_ACCESS(4, drc, ikb, icb, jc+1, jk, kBlocks, bk, bk)), LIBXSMM_INTRINSICS_MM512_LOAD_PS(&LIBXSMM_VLA_ACCESS(4, drc, ikb, icb, jc, jk, kBlocks, bk, bk)));
              _mm512_store_epi32(&LIBXSMM_VLA_ACCESS(5, drc_bf16, ikb, icb, jc/lpb, jk, 0, kBlocks, bk_lp, bk, lpb), _mm512_permutexvar_epi16(perm_index, c01));
            }
          }
        }

        batchreduce_kernelc(&LIBXSMM_VLA_ACCESS(5, dciB, ikb, 0, 0, 0, 0, nBlocks, bn_lp, bk, lpb),
            &LIBXSMM_VLA_ACCESS(4, xT, icb, 0, 0, 0, nBlocks, bc, bn),
            &LIBXSMM_VLA_ACCESS(4, dwc, ikb, icb, 0, 0, cBlocks, bc, bk), &blocks);
        if (j == 0) {
          for (jc = 0; jc < bk; jc+=2) {
            for (jk = 0; jk < bk; jk+=16) {
              c01 = (__m512i) LIBXSMM_INTRINSISCS_MM512_CVTNE2PS_PBH( LIBXSMM_INTRINSICS_MM512_LOAD_PS(&LIBXSMM_VLA_ACCESS(4, dwc, ikb, icb, jc+1, jk, cBlocks, bc, bk)), LIBXSMM_INTRINSICS_MM512_LOAD_PS(&LIBXSMM_VLA_ACCESS(4, dwc, ikb, icb, jc, jk, cBlocks, bc, bk)));
              _mm512_store_epi32(&LIBXSMM_VLA_ACCESS(5, dwc_bf16, ikb, icb, jc/lpb, jk, 0, cBlocks, bc_lp, bk, lpb), _mm512_permutexvar_epi16(perm_index, c01));
            }
          }
        }

        batchreduce_kernelb(&LIBXSMM_VLA_ACCESS(5, dfB, ikb, 0, 0, 0, 0, nBlocks, bn_lp, bk, lpb),
            &LIBXSMM_VLA_ACCESS(4, hT, icb, 0, 0, 0, nBlocks, bk, bn),
            &LIBXSMM_VLA_ACCESS(4, drf, ikb, icb, 0, 0, kBlocks, bk, bk), &blocks);
        if (j == 0) {
          for (jc = 0; jc < bk; jc+=2) {
            for (jk = 0; jk < bk; jk+=16) {
              c01 = (__m512i) LIBXSMM_INTRINSISCS_MM512_CVTNE2PS_PBH( LIBXSMM_INTRINSICS_MM512_LOAD_PS(&LIBXSMM_VLA_ACCESS(4, drf, ikb, icb, jc+1, jk, kBlocks, bk, bk)), LIBXSMM_INTRINSICS_MM512_LOAD_PS(&LIBXSMM_VLA_ACCESS(4, drf, ikb, icb, jc, jk, kBlocks, bk, bk)));
              _mm512_store_epi32(&LIBXSMM_VLA_ACCESS(5, drf_bf16, ikb, icb, jc/lpb, jk, 0, kBlocks, bk_lp, bk, lpb), _mm512_permutexvar_epi16(perm_index, c01));
            }
          }
        }

        batchreduce_kernelc(&LIBXSMM_VLA_ACCESS(5, dfB, ikb, 0, 0, 0, 0, nBlocks, bn_lp, bk, lpb),
            &LIBXSMM_VLA_ACCESS(4, xT, icb, 0, 0, 0, nBlocks, bc, bn),
            &LIBXSMM_VLA_ACCESS(4, dwf, ikb, icb, 0, 0, cBlocks, bc, bk), &blocks);
        if (j == 0) {
          for (jc = 0; jc < bk; jc+=2) {
            for (jk = 0; jk < bk; jk+=16) {
              c01 = (__m512i) LIBXSMM_INTRINSISCS_MM512_CVTNE2PS_PBH( LIBXSMM_INTRINSICS_MM512_LOAD_PS(&LIBXSMM_VLA_ACCESS(4, dwf, ikb, icb, jc+1, jk, cBlocks, bc, bk)), LIBXSMM_INTRINSICS_MM512_LOAD_PS(&LIBXSMM_VLA_ACCESS(4, dwf, ikb, icb, jc, jk, cBlocks, bc, bk)));
              _mm512_store_epi32(&LIBXSMM_VLA_ACCESS(5, dwf_bf16, ikb, icb, jc/lpb, jk, 0, cBlocks, bc_lp, bk, lpb), _mm512_permutexvar_epi16(perm_index, c01));
            }
          }
        }

        batchreduce_kernelb(&LIBXSMM_VLA_ACCESS(5, dpB, ikb, 0, 0, 0, 0, nBlocks, bn_lp, bk, lpb),
            &LIBXSMM_VLA_ACCESS(4, hT, icb, 0, 0, 0, nBlocks, bk, bn),
            &LIBXSMM_VLA_ACCESS(4, dro, ikb, icb, 0, 0, kBlocks, bk, bk), &blocks);
        if (j == 0) {
          for (jc = 0; jc < bk; jc+=2) {
            for (jk = 0; jk < bk; jk+=16) {
              c01 = (__m512i) LIBXSMM_INTRINSISCS_MM512_CVTNE2PS_PBH( LIBXSMM_INTRINSICS_MM512_LOAD_PS(&LIBXSMM_VLA_ACCESS(4, dro, ikb, icb, jc+1, jk, kBlocks, bk, bk)), LIBXSMM_INTRINSICS_MM512_LOAD_PS(&LIBXSMM_VLA_ACCESS(4, dro, ikb, icb, jc, jk, kBlocks, bk, bk)));
              _mm512_store_epi32(&LIBXSMM_VLA_ACCESS(5, dro_bf16, ikb, icb, jc/lpb, jk, 0, kBlocks, bk_lp, bk, lpb), _mm512_permutexvar_epi16(perm_index, c01));
            }
          }
        }


        batchreduce_kernelc(&LIBXSMM_VLA_ACCESS(5, dpB, ikb, 0, 0, 0, 0, nBlocks, bn_lp, bk, lpb),
            &LIBXSMM_VLA_ACCESS(4, xT, icb, 0, 0, 0, nBlocks, bc, bn),
            &LIBXSMM_VLA_ACCESS(4, dwo, ikb, icb, 0, 0, cBlocks, bc, bk), &blocks);
        if (j == 0) {
          for (jc = 0; jc < bk; jc+=2) {
            for (jk = 0; jk < bk; jk+=16) {
              c01 = (__m512i) LIBXSMM_INTRINSISCS_MM512_CVTNE2PS_PBH( LIBXSMM_INTRINSICS_MM512_LOAD_PS(&LIBXSMM_VLA_ACCESS(4, dwo, ikb, icb, jc+1, jk, cBlocks, bc, bk)), LIBXSMM_INTRINSICS_MM512_LOAD_PS(&LIBXSMM_VLA_ACCESS(4, dwo, ikb, icb, jc, jk, cBlocks, bc, bk)));
              _mm512_store_epi32(&LIBXSMM_VLA_ACCESS(5, dwo_bf16, ikb, icb, jc/lpb, jk, 0, cBlocks, bc_lp, bk, lpb), _mm512_permutexvar_epi16(perm_index, c01));
            }
          }
        }
      }
    } else {
      for (ikic = thr_begin_kk; ikic < thr_end_kk; ++ikic ) {
        icb = ikic / (K/bk);
        ikb = ikic % (K/bk);
        ik = ikb*bk;
        batchreduce_kernelb(&LIBXSMM_VLA_ACCESS(5, diB, ikb, 0, 0, 0, 0, nBlocks, bn_lp, bk, lpb),
            &LIBXSMM_VLA_ACCESS(4, hT, icb, 0, 0, 0, nBlocks, bk, bn),
            &LIBXSMM_VLA_ACCESS(4, dri, ikb, icb, 0, 0, kBlocks, bk, bk), &blocks);
        if (j == 0) {
          for (jc = 0; jc < bk; jc+=2) {
            for (jk = 0; jk < bk; jk+=16) {
              c01 = (__m512i) LIBXSMM_INTRINSISCS_MM512_CVTNE2PS_PBH( LIBXSMM_INTRINSICS_MM512_LOAD_PS(&LIBXSMM_VLA_ACCESS(4, dri, ikb, icb, jc+1, jk, kBlocks, bk, bk)), LIBXSMM_INTRINSICS_MM512_LOAD_PS(&LIBXSMM_VLA_ACCESS(4, dri, ikb, icb, jc, jk, kBlocks, bk, bk)));
              _mm512_store_epi32(&LIBXSMM_VLA_ACCESS(5, dri_bf16, ikb, icb, jc/lpb, jk, 0, kBlocks, bk_lp, bk, lpb), _mm512_permutexvar_epi16(perm_index, c01));
            }
          }
        }

        batchreduce_kernelb(&LIBXSMM_VLA_ACCESS(5, dciB, ikb, 0, 0, 0, 0, nBlocks, bn_lp, bk, lpb),
            &LIBXSMM_VLA_ACCESS(4, hT, icb, 0, 0, 0, nBlocks, bk, bn),
            &LIBXSMM_VLA_ACCESS(4, drc, ikb, icb, 0, 0, kBlocks, bk, bk), &blocks);
        if (j == 0) {
          for (jc = 0; jc < bk; jc+=2) {
            for (jk = 0; jk < bk; jk+=16) {
              c01 = (__m512i) LIBXSMM_INTRINSISCS_MM512_CVTNE2PS_PBH( LIBXSMM_INTRINSICS_MM512_LOAD_PS(&LIBXSMM_VLA_ACCESS(4, drc, ikb, icb, jc+1, jk, kBlocks, bk, bk)), LIBXSMM_INTRINSICS_MM512_LOAD_PS(&LIBXSMM_VLA_ACCESS(4, drc, ikb, icb, jc, jk, kBlocks, bk, bk)));
              _mm512_store_epi32(&LIBXSMM_VLA_ACCESS(5, drc_bf16, ikb, icb, jc/lpb, jk, 0, kBlocks, bk_lp, bk, lpb), _mm512_permutexvar_epi16(perm_index, c01));
            }
          }
        }

        batchreduce_kernelb(&LIBXSMM_VLA_ACCESS(5, dfB, ikb, 0, 0, 0, 0, nBlocks, bn_lp, bk, lpb),
            &LIBXSMM_VLA_ACCESS(4, hT, icb, 0, 0, 0, nBlocks, bk, bn),
            &LIBXSMM_VLA_ACCESS(4, drf, ikb, icb, 0, 0, kBlocks, bk, bk), &blocks);
        if (j == 0) {
          for (jc = 0; jc < bk; jc+=2) {
            for (jk = 0; jk < bk; jk+=16) {
              c01 = (__m512i) LIBXSMM_INTRINSISCS_MM512_CVTNE2PS_PBH( LIBXSMM_INTRINSICS_MM512_LOAD_PS(&LIBXSMM_VLA_ACCESS(4, drf, ikb, icb, jc+1, jk, kBlocks, bk, bk)), LIBXSMM_INTRINSICS_MM512_LOAD_PS(&LIBXSMM_VLA_ACCESS(4, drf, ikb, icb, jc, jk, kBlocks, bk, bk)));
              _mm512_store_epi32(&LIBXSMM_VLA_ACCESS(5, drf_bf16, ikb, icb, jc/lpb, jk, 0, kBlocks, bk_lp, bk, lpb), _mm512_permutexvar_epi16(perm_index, c01));
            }
          }
        }

        batchreduce_kernelb(&LIBXSMM_VLA_ACCESS(5, dpB, ikb, 0, 0, 0, 0, nBlocks, bn_lp, bk, lpb),
            &LIBXSMM_VLA_ACCESS(4, hT, icb, 0, 0, 0, nBlocks, bk, bn),
            &LIBXSMM_VLA_ACCESS(4, dro, ikb, icb, 0, 0, kBlocks, bk, bk), &blocks);
        if (j == 0) {
          for (jc = 0; jc < bk; jc+=2) {
            for (jk = 0; jk < bk; jk+=16) {
              c01 = (__m512i) LIBXSMM_INTRINSISCS_MM512_CVTNE2PS_PBH( LIBXSMM_INTRINSICS_MM512_LOAD_PS(&LIBXSMM_VLA_ACCESS(4, dro, ikb, icb, jc+1, jk, kBlocks, bk, bk)), LIBXSMM_INTRINSICS_MM512_LOAD_PS(&LIBXSMM_VLA_ACCESS(4, dro, ikb, icb, jc, jk, kBlocks, bk, bk)));
              _mm512_store_epi32(&LIBXSMM_VLA_ACCESS(5, dro_bf16, ikb, icb, jc/lpb, jk, 0, kBlocks, bk_lp, bk, lpb), _mm512_permutexvar_epi16(perm_index, c01));
            }
          }
        }
      }

      for (ikic = thr_begin_ck; ikic < thr_end_ck; ++ikic ) {
        icb = ikic / (K/bk);
        ikb = ikic % (K/bk);
        ik = ikb*bk;
        batchreduce_kernelc(&LIBXSMM_VLA_ACCESS(5, diB, ikb, 0, 0, 0, 0, nBlocks, bn_lp, bk, lpb),
            &LIBXSMM_VLA_ACCESS(4, xT, icb, 0, 0, 0, nBlocks, bc, bn),
            &LIBXSMM_VLA_ACCESS(4, dwi, ikb, icb, 0, 0, cBlocks, bc, bk), &blocks);
        if (j == 0) {
          for (jc = 0; jc < bc; jc+=2) {
            for (jk = 0; jk < bk; jk+=16) {
              c01 = (__m512i) LIBXSMM_INTRINSISCS_MM512_CVTNE2PS_PBH( LIBXSMM_INTRINSICS_MM512_LOAD_PS(&LIBXSMM_VLA_ACCESS(4, dwi, ikb, icb, jc+1, jk, cBlocks, bc, bk)), LIBXSMM_INTRINSICS_MM512_LOAD_PS(&LIBXSMM_VLA_ACCESS(4, dwi, ikb, icb, jc, jk, cBlocks, bc, bk)));
              _mm512_store_epi32(&LIBXSMM_VLA_ACCESS(5, dwi_bf16, ikb, icb, jc/lpb, jk, 0, cBlocks, bc_lp, bk, lpb), _mm512_permutexvar_epi16(perm_index, c01));
            }
          }
        }

        batchreduce_kernelc(&LIBXSMM_VLA_ACCESS(5, dciB, ikb, 0, 0, 0, 0, nBlocks, bn_lp, bk, lpb),
            &LIBXSMM_VLA_ACCESS(4, xT, icb, 0, 0, 0, nBlocks, bc, bn),
            &LIBXSMM_VLA_ACCESS(4, dwc, ikb, icb, 0, 0, cBlocks, bc, bk), &blocks);
        if (j == 0) {
          for (jc = 0; jc < bc; jc+=2) {
            for (jk = 0; jk < bk; jk+=16) {
              c01 = (__m512i) LIBXSMM_INTRINSISCS_MM512_CVTNE2PS_PBH( LIBXSMM_INTRINSICS_MM512_LOAD_PS(&LIBXSMM_VLA_ACCESS(4, dwc, ikb, icb, jc+1, jk, cBlocks, bc, bk)), LIBXSMM_INTRINSICS_MM512_LOAD_PS(&LIBXSMM_VLA_ACCESS(4, dwc, ikb, icb, jc, jk, cBlocks, bc, bk)));
              _mm512_store_epi32(&LIBXSMM_VLA_ACCESS(5, dwc_bf16, ikb, icb, jc/lpb, jk, 0, cBlocks, bc_lp, bk, lpb), _mm512_permutexvar_epi16(perm_index, c01));
            }
          }
        }

        batchreduce_kernelc(&LIBXSMM_VLA_ACCESS(5, dfB, ikb, 0, 0, 0, 0, nBlocks, bn_lp, bk, lpb),
            &LIBXSMM_VLA_ACCESS(4, xT, icb, 0, 0, 0, nBlocks, bc, bn),
            &LIBXSMM_VLA_ACCESS(4, dwf, ikb, icb, 0, 0, cBlocks, bc, bk), &blocks);
        if (j == 0) {
          for (jc = 0; jc < bc; jc+=2) {
            for (jk = 0; jk < bk; jk+=16) {
              c01 = (__m512i) LIBXSMM_INTRINSISCS_MM512_CVTNE2PS_PBH( LIBXSMM_INTRINSICS_MM512_LOAD_PS(&LIBXSMM_VLA_ACCESS(4, dwf, ikb, icb, jc+1, jk, cBlocks, bc, bk)), LIBXSMM_INTRINSICS_MM512_LOAD_PS(&LIBXSMM_VLA_ACCESS(4, dwf, ikb, icb, jc, jk, cBlocks, bc, bk)));
              _mm512_store_epi32(&LIBXSMM_VLA_ACCESS(5, dwf_bf16, ikb, icb, jc/lpb, jk, 0, cBlocks, bc_lp, bk, lpb), _mm512_permutexvar_epi16(perm_index, c01));
            }
          }
        }

        batchreduce_kernelc(&LIBXSMM_VLA_ACCESS(5, dpB, ikb, 0, 0, 0, 0, nBlocks, bn_lp, bk, lpb),
            &LIBXSMM_VLA_ACCESS(4, xT, icb, 0, 0, 0, nBlocks, bc, bn),
            &LIBXSMM_VLA_ACCESS(4, dwo, ikb, icb, 0, 0, cBlocks, bc, bk), &blocks);
        if (j == 0) {
          for (jc = 0; jc < bk; jc+=2) {
            for (jk = 0; jk < bk; jk+=16) {
              c01 = (__m512i) LIBXSMM_INTRINSISCS_MM512_CVTNE2PS_PBH( LIBXSMM_INTRINSICS_MM512_LOAD_PS(&LIBXSMM_VLA_ACCESS(4, dwo, ikb, icb, jc+1, jk, cBlocks, bc, bk)), LIBXSMM_INTRINSICS_MM512_LOAD_PS(&LIBXSMM_VLA_ACCESS(4, dwo, ikb, icb, jc, jk, cBlocks, bc, bk)));
              _mm512_store_epi32(&LIBXSMM_VLA_ACCESS(5, dwo_bf16, ikb, icb, jc/lpb, jk, 0, cBlocks, bc_lp, bk, lpb), _mm512_permutexvar_epi16(perm_index, c01));
            }
          }
        }
      }
    }

    /* gradient bias */
    if (bcbk_multiples_of_16) {
      for (ik = k_thr_begin; ik < k_thr_end; ik += 16) {
        dbi_sum = LIBXSMM_INTRINSICS_MM512_LOAD_PS(&dbi[ik]);
        dbf_sum = LIBXSMM_INTRINSICS_MM512_LOAD_PS(&dbf[ik]);
        dbo_sum = LIBXSMM_INTRINSICS_MM512_LOAD_PS(&dbo[ik]);
        dbc_sum = LIBXSMM_INTRINSICS_MM512_LOAD_PS(&dbc[ik]);
        for (in = 0; in < N; in++) {
          dbi_sum = _mm512_add_ps(dbi_sum, _mm512_loadcvt_bf16_fp32(&LIBXSMM_VLA_ACCESS(4, di,  in/bn, ik/bk, in%bn, ik%bk, kBlocks, bn, bk)));
          dbf_sum = _mm512_add_ps(dbf_sum, _mm512_loadcvt_bf16_fp32(&LIBXSMM_VLA_ACCESS(4, df,  in/bn, ik/bk, in%bn, ik%bk, kBlocks, bn, bk)));
          dbo_sum = _mm512_add_ps(dbo_sum, _mm512_loadcvt_bf16_fp32(&LIBXSMM_VLA_ACCESS(4, dp,  in/bn, ik/bk, in%bn, ik%bk, kBlocks, bn, bk)));
          dbc_sum = _mm512_add_ps(dbc_sum, _mm512_loadcvt_bf16_fp32(&LIBXSMM_VLA_ACCESS(4, dci,  in/bn, ik/bk, in%bn, ik%bk, kBlocks, bn, bk)));
        }
        _mm512_store_ps(&dbi[ik], dbi_sum);
        _mm512_store_ps(&dbf[ik], dbf_sum);
        _mm512_store_ps(&dbo[ik], dbo_sum);
        _mm512_store_ps(&dbc[ik], dbc_sum);
        /* Downconvert delta bias to bf16 if done with all timesteps */
        if (j == 0) {
          _mm512_storecvt_fp32_bf16(&dbi_bf16[ik], dbi_sum);
          _mm512_storecvt_fp32_bf16(&dbf_bf16[ik], dbf_sum);
          _mm512_storecvt_fp32_bf16(&dbo_bf16[ik], dbo_sum);
          _mm512_storecvt_fp32_bf16(&dbc_bf16[ik], dbc_sum);
        }
      }
    }
  }
  libxsmm_barrier_wait(handle->barrier, (int)ltid);
}

#undef NATIVE_MATRIX_RNE_CVT_FP32_BFP16_LD

