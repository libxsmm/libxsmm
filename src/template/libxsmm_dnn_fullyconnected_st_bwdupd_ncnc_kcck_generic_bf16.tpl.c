/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Evangelos Georganas, Alexander Heinecke (Intel Corp.)
******************************************************************************/

/* size variables, all const */
/* here we assume that input and output blocking is similar */
const int bn = handle->bn;
const int bk = handle->bk;
const int bc = handle->bc;
int lpb = 2;
const int bc_lp = bc/lpb;
const int bk_lp = bk/lpb;
const int bn_lp = bn/lpb;
const int nBlocksIFm = handle->desc.C / handle->bc;
const int nBlocksOFm = handle->desc.K / handle->bk;
const int nBlocksMB  = handle->desc.N / handle->bn;
int mb1ofm1 = 0, mb1 = 0, ofm1 = 0, mb2 = 0, ofm2 = 0;
#if defined(LIBXSMM_DNN_FC_BWD_FUSE_RELU) || defined(LIBXSMM_DNN_FC_BWD_FUSE_SIGMOID) || defined(LIBXSMM_DNN_FC_BWD_FUSE_BIAS)
int iteri = 0, iterj = 0;
#endif
int performed_doutput_transpose = 0;

/* computing first logical thread */
const int ltid = tid - start_thread;

#if defined(LIBXSMM_DNN_FC_BWD_FUSE_RELU) || defined(LIBXSMM_DNN_FC_BWD_FUSE_SIGMOID)
/* number of tasks for transpose that could be run in parallel */
const int eltwise_work = nBlocksOFm * nBlocksMB;
/* compute chunk size */
const int eltwise_chunksize = (eltwise_work % handle->desc.threads == 0) ? (eltwise_work / handle->desc.threads) : ((eltwise_work / handle->desc.threads) + 1);
/* compute thr_begin and thr_end */
const int eltwise_thr_begin = (ltid * eltwise_chunksize < eltwise_work) ? (ltid * eltwise_chunksize) : eltwise_work;
const int eltwise_thr_end = ((ltid + 1) * eltwise_chunksize < eltwise_work) ? ((ltid + 1) * eltwise_chunksize) : eltwise_work;
#endif

#ifdef LIBXSMM_DNN_FC_BWD_FUSE_BIAS
/* number of tasks for transpose that could be run in parallel */
const int dbias_work = nBlocksOFm;
/* compute chunk size */
const int dbias_chunksize = (dbias_work % handle->desc.threads == 0) ? (dbias_work / handle->desc.threads) : ((dbias_work / handle->desc.threads) + 1);
/* compute thr_begin and thr_end */
const int dbias_thr_begin = (ltid * dbias_chunksize < dbias_work) ? (ltid * dbias_chunksize) : dbias_work;
const int dbias_thr_end = ((ltid + 1) * dbias_chunksize < dbias_work) ? ((ltid + 1) * dbias_chunksize) : dbias_work;
#endif

#ifdef LIBXSMM_DNN_FC_BWD_FUSE_BIAS
LIBXSMM_VLA_DECL(2, libxsmm_bfloat16, dbias, (libxsmm_bfloat16*) handle->grad_bias->data, handle->bk);
#endif
#ifdef LIBXSMM_DNN_FC_BWD_FUSE_RELU
LIBXSMM_VLA_DECL(4, unsigned char,    relumask, (unsigned char*)handle->relumask->data, nBlocksOFm, handle->bn, handle->bk);
LIBXSMM_VLA_DECL(4,     __mmask32, relubitmask,     (__mmask32*)handle->relumask->data, nBlocksOFm, handle->bn, handle->bk/32);
#endif

#if defined(LIBXSMM_DNN_FC_BWD_FUSE_RELU) || defined(LIBXSMM_DNN_FC_BWD_FUSE_SIGMOID)
element_output_type *grad_output_ptr = (element_output_type*)((char*)handle->scratch + handle->doutput_scratch_mark);
element_output_type *tr_doutput_ptr = (element_output_type*)grad_output_ptr + handle->desc.N * handle->desc.K;
LIBXSMM_VLA_DECL(4, const element_output_type,   doutput_orig, (element_output_type*)handle->grad_output->data, nBlocksOFm, bn, bk);
#else
element_output_type *grad_output_ptr = (element_output_type*)handle->grad_output->data;
element_output_type *tr_doutput_ptr = (element_output_type*)handle->scratch;
#endif
LIBXSMM_VLA_DECL(4, element_output_type,   doutput, grad_output_ptr, nBlocksOFm, bn, bk);
LIBXSMM_VLA_DECL(5, element_output_type, doutput_tr, tr_doutput_ptr, nBlocksMB, bn_lp, bk, lpb);

/* lazy barrier init */
libxsmm_barrier_init(handle->barrier, ltid);

/* Apply to doutput potential fusions */
#if defined(LIBXSMM_DNN_FC_BWD_FUSE_RELU) || defined(LIBXSMM_DNN_FC_BWD_FUSE_SIGMOID)
if (bk % 32 == 0) {
  for ( mb1ofm1 = eltwise_thr_begin; mb1ofm1 < eltwise_thr_end; ++mb1ofm1 ) {
    mb1  = mb1ofm1%nBlocksMB;
    ofm1 = mb1ofm1/nBlocksMB;

    for ( iteri = 0; iteri < handle->bn; ++iteri ) {
      for ( iterj = 0; iterj < handle->bk; iterj += 32 ) {
        __m512i cur_out_reg = _mm512_loadu_si512(&LIBXSMM_VLA_ACCESS(4, doutput_orig, mb1, ofm1, iteri, iterj, nBlocksOFm, handle->bn, handle->bk));
#ifdef LIBXSMM_DNN_FC_BWD_FUSE_SIGMOID
        __m512 cur_out_reg_0, cur_out_reg_1;
        const  __m512 ones = _mm512_set1_ps(1.0f);
#endif
#ifdef LIBXSMM_DNN_FC_BWD_FUSE_RELU
        __m512i zero_reg = _mm512_setzero_si512();
        __mmask32 relumask = LIBXSMM_INTRINSICS_MM512_LOAD_MASK32 (&LIBXSMM_VLA_ACCESS(4, relubitmask, mb1, ofm1, iteri, iterj/32, nBlocksOFm, handle->bn, handle->bk/32));
        cur_out_reg = _mm512_mask_blend_epi16 (relumask, zero_reg, cur_out_reg);
#endif
#ifdef LIBXSMM_DNN_FC_BWD_FUSE_SIGMOID
        cur_out_reg_0 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepi16_epi32(LIBXSMM_INTRINSICS_MM512_EXTRACTI64X4_EPI64(cur_out_reg, 0)),16));
        cur_out_reg_1 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepi16_epi32(LIBXSMM_INTRINSICS_MM512_EXTRACTI64X4_EPI64(cur_out_reg, 1)),16));
        cur_out_reg_0 =  _mm512_mul_ps(cur_out_reg_0, _mm512_sub_ps(ones, cur_out_reg_0));
        cur_out_reg_1 =  _mm512_mul_ps(cur_out_reg_1, _mm512_sub_ps(ones, cur_out_reg_1));
        cur_out_reg = LIBXSMM_INTRINSISCS_MM512_CVTNE2PS_PBH(cur_out_reg_1, cur_out_reg_0);
#endif
        _mm512_storeu_si512(&LIBXSMM_VLA_ACCESS(4, doutput, mb1, ofm1, iteri, iterj, nBlocksOFm, handle->bn, handle->bk), cur_out_reg);
      }
    }

    /* If in UPD pass, also perform transpose of doutput  */
    if ( (kind == LIBXSMM_DNN_COMPUTE_KIND_UPD) || (kind == LIBXSMM_DNN_COMPUTE_KIND_BWDUPD) ) {
      bf16_vnni_reformat((element_output_type*)&LIBXSMM_VLA_ACCESS(4, doutput,  mb1, ofm1, 0, 0, nBlocksOFm, bn, bk), &LIBXSMM_VLA_ACCESS(5, doutput_tr, ofm1, mb1, 0, 0, 0, nBlocksMB, bn_lp, bk, lpb), bk, bn, bk, bn);
    }
  }
} else {
  for ( mb1ofm1 = eltwise_thr_begin; mb1ofm1 < eltwise_thr_end; ++mb1ofm1 ) {
    mb1  = mb1ofm1%nBlocksMB;
    ofm1 = mb1ofm1/nBlocksMB;

    for ( iteri = 0; iteri < handle->bn; ++iteri ) {
      for ( iterj = 0; iterj < handle->bk; ++iterj ) {
        element_output_type l_cur_out = LIBXSMM_VLA_ACCESS(4, doutput_orig, mb1, ofm1, iteri, iterj, nBlocksOFm, handle->bn, handle->bk);
#ifdef LIBXSMM_DNN_FC_BWD_FUSE_SIGMOID
        float l_cur_out_f32 = 0;
        libxsmm_bfloat16_hp tmp;
#endif
#ifdef LIBXSMM_DNN_FC_BWD_FUSE_RELU
        l_cur_out = (element_output_type)((LIBXSMM_VLA_ACCESS(4, relumask, mb1, ofm1, iteri, iterj, nBlocksOFm, handle->bn, handle->bk) != 0) ? l_cur_out : (element_output_type)0);
#endif
#ifdef LIBXSMM_DNN_FC_BWD_FUSE_SIGMOID
        tmp.i[0] = 0;
        tmp.i[1] = l_cur_out;
        l_cur_out_f32 = tmp.f;
        l_cur_out_f32 = l_cur_out_f32*(1.0f - l_cur_out_f32);
        libxsmm_rne_convert_fp32_bf16(&l_cur_out_f32, &l_cur_out, 1);
#endif
        LIBXSMM_VLA_ACCESS(4, doutput, mb1, ofm1, iteri, iterj, nBlocksOFm, handle->bn, handle->bk) = l_cur_out;
      }
    }

    /* If in UPD pass, also perform transpose of doutput  */
    if ( (kind == LIBXSMM_DNN_COMPUTE_KIND_UPD) || (kind == LIBXSMM_DNN_COMPUTE_KIND_BWDUPD) ) {
      for (mb2 = 0; mb2 < bn; mb2++) {
        for (ofm2 = 0; ofm2 < bk; ofm2++) {
          LIBXSMM_VLA_ACCESS(5, doutput_tr, ofm1, mb1, mb2/lpb, ofm2, mb2%lpb, nBlocksMB, bn_lp, bk, lpb) = LIBXSMM_VLA_ACCESS(4, doutput,  mb1, ofm1, mb2, ofm2, nBlocksOFm, bn, bk);
        }
      }
    }
  }
}
if ( (kind == LIBXSMM_DNN_COMPUTE_KIND_UPD) || (kind == LIBXSMM_DNN_COMPUTE_KIND_BWDUPD) ) {
  performed_doutput_transpose = 1;
}
libxsmm_barrier_wait(handle->barrier, ltid);
#endif

#if defined(LIBXSMM_DNN_FC_BWD_FUSE_BIAS)
/* Accumulation of bias happens in f32 */
{
  float *scratch_dbias = (float*) ((element_output_type*)handle->scratch + handle->desc.N * (handle->desc.K + handle->desc.C) + ltid * bk * 2);
  if (handle->bk % 16 == 0) {
    __m512 zero_reg = _mm512_setzero_ps();
    __m512 doutput_reg = _mm512_setzero_ps();
    __m512 dbias_reg = _mm512_setzero_ps();
    for ( ofm1 = dbias_thr_begin; ofm1 < dbias_thr_end; ++ofm1 ) {
      for ( iterj = 0; iterj < handle->bk; iterj += 16 ) {
        _mm512_storeu_ps(scratch_dbias+iterj, zero_reg);
      }
      for ( mb1 = 0; mb1 < nBlocksMB; ++mb1 ) {
        for ( iteri = 0; iteri < handle->bn; ++iteri ) {
          for ( iterj = 0; iterj < handle->bk; iterj += 16 ) {
            doutput_reg = LIBXSMM_INTRINSICS_MM512_CVTPBH_PS(_mm256_loadu_si256((const __m256i*)&LIBXSMM_VLA_ACCESS(4,  doutput, mb1, ofm1, iteri, iterj, nBlocksOFm, handle->bn, handle->bk)));
            dbias_reg = LIBXSMM_INTRINSICS_MM512_LOAD_PS(scratch_dbias+iterj);
            dbias_reg = _mm512_add_ps(dbias_reg, doutput_reg);
            _mm512_storeu_ps(scratch_dbias+iterj, dbias_reg);
          }
        }
      }
      for ( iterj = 0; iterj < handle->bk; iterj += 16 ) {
        _mm256_storeu_si256((__m256i*)&LIBXSMM_VLA_ACCESS( 2, dbias, ofm1, iterj, handle->bk ), LIBXSMM_INTRINSISCS_MM512_CVTNEPS_PBH( LIBXSMM_INTRINSICS_MM512_LOAD_PS(scratch_dbias+iterj)) );
      }
    }
  } else {
    for ( ofm1 = dbias_thr_begin; ofm1 < dbias_thr_end; ++ofm1 ) {
      for ( iterj = 0; iterj < handle->bk; ++iterj ) {
        scratch_dbias[iterj] = 0.0;
      }
      for ( mb1 = 0; mb1 < nBlocksMB; ++mb1 ) {
        for ( iteri = 0; iteri < handle->bn; ++iteri ) {
          for ( iterj = 0; iterj < handle->bk; ++iterj ) {
            float doutput_f32 = 0;
            libxsmm_bfloat16_hp tmp;
            tmp.i[0] = 0;
            tmp.i[1] = LIBXSMM_VLA_ACCESS(4,  doutput, mb1, ofm1, iteri, iterj, nBlocksOFm, handle->bn, handle->bk);
            doutput_f32 = tmp.f;
            scratch_dbias[iterj] += doutput_f32;
          }
        }
      }
      libxsmm_rne_convert_fp32_bf16(scratch_dbias, &LIBXSMM_VLA_ACCESS( 2, dbias, ofm1, 0, handle->bk ), handle->bk);
    }
  }
}

/* wait for eltwise to finish */
libxsmm_barrier_wait(handle->barrier, ltid);
#endif

if ( (kind == LIBXSMM_DNN_COMPUTE_KIND_BWD) || (kind == LIBXSMM_DNN_COMPUTE_KIND_BWDUPD) ){
  int use_2d_blocking = handle->bwd_2d_blocking;

  /* number of tasks that could be run in parallel */
  const int work = nBlocksIFm * nBlocksMB;
  /* compute chunk size */
  const int chunksize = (work % handle->desc.threads == 0) ? (work / handle->desc.threads) : ((work / handle->desc.threads) + 1);
  /* compute thr_begin and thr_end */
  const int thr_begin = (ltid * chunksize < work) ? (ltid * chunksize) : work;
  const int thr_end = ((ltid + 1) * chunksize < work) ? ((ltid + 1) * chunksize) : work;

  /* number of tasks for transpose that could be run in parallel */
  const int transpose_work = nBlocksIFm * nBlocksOFm;
  /* compute chunk size */
  const int transpose_chunksize = (transpose_work % handle->desc.threads == 0) ? (transpose_work / handle->desc.threads) : ((transpose_work / handle->desc.threads) + 1);
  /* compute thr_begin and thr_end */
  const int transpose_thr_begin = (ltid * transpose_chunksize < transpose_work) ? (ltid * transpose_chunksize) : transpose_work;
  const int transpose_thr_end = ((ltid + 1) * transpose_chunksize < transpose_work) ? ((ltid + 1) * transpose_chunksize) : transpose_work;

  /* loop variables */
  int ifm1 = 0, ifm2 = 0, ifm1ofm1 = 0, mb1ifm1 = 0;
  int im_tasks_per_thread = 0, in_tasks_per_thread = 0, my_in_start = 0, my_in_end = 0, my_im_start = 0, my_im_end = 0, my_row_id = 0, my_col_id = 0, row_teams = 0, column_teams = 0;

  LIBXSMM_VLA_DECL(5, const element_filter_type, filter, (element_filter_type*)handle->reg_filter->data, nBlocksIFm, bc_lp, bk, lpb);
  LIBXSMM_VLA_DECL(4,        element_input_type,    dinput, (element_input_type* )handle->grad_input->data, nBlocksIFm, bn, bc);
  LIBXSMM_VLA_DECL(5,       element_filter_type, filter_tr, (element_filter_type*)handle->scratch, nBlocksOFm, bk_lp, bc, lpb);
  float* temp_output = (float*)handle->scratch + (handle->desc.C * handle->desc.K)/2;
  LIBXSMM_VLA_DECL(4,        float,    dinput_f32, (float*) temp_output, nBlocksIFm, bn, bc);

  unsigned long long  blocks = nBlocksOFm;
  int KB_BLOCKS = nBlocksOFm, BF = 1;
  BF = handle->bwd_bf;
  KB_BLOCKS = nBlocksOFm/BF;
  blocks = KB_BLOCKS;

  if (use_2d_blocking == 1) {
    row_teams = handle->bwd_row_teams;
    column_teams = handle->bwd_column_teams;
    my_col_id = ltid % column_teams;
    my_row_id = ltid / column_teams;
    im_tasks_per_thread = LIBXSMM_UPDIV(nBlocksMB, row_teams);
    in_tasks_per_thread = LIBXSMM_UPDIV(nBlocksIFm, column_teams);
    my_im_start = LIBXSMM_MIN(my_row_id * im_tasks_per_thread, nBlocksMB);
    my_im_end = LIBXSMM_MIN((my_row_id+1) * im_tasks_per_thread, nBlocksMB);
    my_in_start = LIBXSMM_MIN(my_col_id * in_tasks_per_thread, nBlocksIFm);
    my_in_end = LIBXSMM_MIN((my_col_id+1) * in_tasks_per_thread, nBlocksIFm);
  }

  if (handle->desc.K > 1) {
    /* transpose weight */
    if ((bk % 16 == 0) && (bc % 16 == 0)) {
      for (ifm1ofm1 = transpose_thr_begin; ifm1ofm1 < transpose_thr_end; ++ifm1ofm1) {
        ofm1 = ifm1ofm1 / nBlocksIFm;
        ifm1 = ifm1ofm1 % nBlocksIFm;
        bf16_vnni_transpose((element_filter_type*)&LIBXSMM_VLA_ACCESS(5, filter,  ofm1, ifm1, 0, 0, 0, nBlocksIFm, bc_lp, bk, lpb), (element_filter_type*)&LIBXSMM_VLA_ACCESS(5, filter_tr, ifm1, ofm1, 0, 0, 0, nBlocksOFm, bk_lp, bc, lpb), bk, bc, bk, bc);
      }
    } else {
      for (ifm1ofm1 = transpose_thr_begin; ifm1ofm1 < transpose_thr_end; ++ifm1ofm1) {
        ofm1 = ifm1ofm1 / nBlocksIFm;
        ifm1 = ifm1ofm1 % nBlocksIFm;
        for (ofm2 = 0; ofm2 < bk; ++ofm2) {
          for (ifm2 = 0; ifm2 < bc; ++ifm2) {
            LIBXSMM_VLA_ACCESS(5, filter_tr, ifm1, ofm1, ofm2/lpb, ifm2, ofm2%lpb, nBlocksOFm, bk_lp, bc, lpb) = LIBXSMM_VLA_ACCESS(5, filter,  ofm1, ifm1, ifm2/lpb, ofm2, ifm2%lpb, nBlocksIFm, bc_lp, bk, lpb);
          }
        }
      }
    }

    /* wait for transpose to finish */
    libxsmm_barrier_wait(handle->barrier, ltid);

    if (use_2d_blocking == 1) {
      if (BF > 1) {
        for ( ofm1 = 0; ofm1 < BF; ++ofm1 ) {
          for (ifm1 = my_in_start; ifm1 < my_in_end; ++ifm1) {
            for (mb1 = my_im_start; mb1 < my_im_end; ++mb1) {
              /* Initialize intermediate f32 tensor */
              if ( ofm1 == 0 ) {
                memset(&LIBXSMM_VLA_ACCESS(4, dinput_f32, mb1, ifm1, 0, 0, nBlocksIFm, bn, bc), 0, bn*bc*sizeof(float));
              }
              batchreduce_kernel_bwd( &LIBXSMM_VLA_ACCESS(5, filter_tr, ifm1, ofm1*KB_BLOCKS, 0, 0, 0, nBlocksOFm, bk_lp, bc, lpb),
                  &LIBXSMM_VLA_ACCESS(4, doutput,   mb1,  ofm1*KB_BLOCKS, 0, 0, nBlocksOFm, bn, bk),
                  &LIBXSMM_VLA_ACCESS(4, dinput_f32,    mb1,  ifm1, 0, 0, nBlocksIFm, bn, bc), &blocks);
              /* downconvert intermediate f32 tensor to bf 16 and store to final C */
              if ( ofm1 == BF-1  ) {
                LIBXSMM_DNN_CONVERT_BUFFER_F32_BF16(&LIBXSMM_VLA_ACCESS(4, dinput_f32,    mb1,  ifm1, 0, 0, nBlocksIFm, bn, bc), &LIBXSMM_VLA_ACCESS(4, dinput,    mb1,  ifm1, 0, 0, nBlocksIFm, bn, bc), bn*bc);
              }
            }
          }
        }
      } else {
        for (ifm1 = my_in_start; ifm1 < my_in_end; ++ifm1) {
          for (mb1 = my_im_start; mb1 < my_im_end; ++mb1) {
            batchreduce_kernel_bwd_zerobeta( &LIBXSMM_VLA_ACCESS(5, filter_tr, ifm1, 0, 0, 0, 0, nBlocksOFm, bk_lp, bc, lpb),
                &LIBXSMM_VLA_ACCESS(4, doutput,   mb1,  0, 0, 0, nBlocksOFm, bn, bk),
                &LIBXSMM_VLA_ACCESS(4, dinput,    mb1,  ifm1, 0, 0, nBlocksIFm, bn, bc), &blocks);
          }
        }
      }
    } else {
      if (BF > 1) {
        for ( ofm1 = 0; ofm1 < BF; ++ofm1 ) {
          for ( mb1ifm1 = thr_begin; mb1ifm1 < thr_end; ++mb1ifm1 ) {
            mb1  = mb1ifm1%nBlocksMB;
            ifm1 = mb1ifm1/nBlocksMB;
            /* Initialize intermediate f32 tensor */
            if ( ofm1 == 0 ) {
              memset(&LIBXSMM_VLA_ACCESS(4, dinput_f32, mb1, ifm1, 0, 0, nBlocksIFm, bn, bc), 0, bn*bc*sizeof(float));
            }
            batchreduce_kernel_bwd( &LIBXSMM_VLA_ACCESS(5, filter_tr, ifm1, ofm1*KB_BLOCKS, 0, 0, 0, nBlocksOFm, bk_lp, bc, lpb),
                &LIBXSMM_VLA_ACCESS(4, doutput,   mb1,  ofm1*KB_BLOCKS, 0, 0, nBlocksOFm, bn, bk),
                &LIBXSMM_VLA_ACCESS(4, dinput_f32,    mb1,  ifm1, 0, 0, nBlocksIFm, bn, bc), &blocks);
            /* downconvert intermediate f32 tensor to bf 16 and store to final C */
            if ( ofm1 == BF-1  ) {
              LIBXSMM_DNN_CONVERT_BUFFER_F32_BF16(&LIBXSMM_VLA_ACCESS(4, dinput_f32,    mb1,  ifm1, 0, 0, nBlocksIFm, bn, bc), &LIBXSMM_VLA_ACCESS(4, dinput,    mb1,  ifm1, 0, 0, nBlocksIFm, bn, bc), bn*bc);
            }
          }
        }
      } else {
        for ( mb1ifm1 = thr_begin; mb1ifm1 < thr_end; ++mb1ifm1 ) {
          mb1  = mb1ifm1%nBlocksMB;
          ifm1 = mb1ifm1/nBlocksMB;
          batchreduce_kernel_bwd_zerobeta( &LIBXSMM_VLA_ACCESS(5, filter_tr, ifm1, 0, 0, 0, 0, nBlocksOFm, bk_lp, bc, lpb),
              &LIBXSMM_VLA_ACCESS(4, doutput,   mb1,  0, 0, 0, nBlocksOFm, bn, bk),
              &LIBXSMM_VLA_ACCESS(4, dinput,    mb1,  ifm1, 0, 0, nBlocksIFm, bn, bc), &blocks);
        }
      }
    }
  } else {
    /* Special case when K = 1 */
    /* number of tasks for doutput copy that could be run in parallel */
    const int copy_work_output = nBlocksMB * nBlocksOFm;
    /* compute chunk size */
    const int copy_chunksize = (copy_work_output % handle->desc.threads == 0) ? (copy_work_output / handle->desc.threads) : ((copy_work_output / handle->desc.threads) + 1);
    /* compute thr_begin and thr_end */
    const int copy_thr_begin = (ltid * copy_chunksize < copy_work_output) ? (ltid * copy_chunksize) : copy_work_output;
    const int copy_thr_end = ((ltid + 1) * copy_chunksize < copy_work_output) ? ((ltid + 1) * copy_chunksize) : copy_work_output;
    LIBXSMM_VLA_DECL(5,       element_filter_type, filter_tr_padded, (element_filter_type*)handle->scratch, nBlocksOFm, 1, bc, lpb);
    LIBXSMM_VLA_DECL(4,       element_output_type,   doutput_padded, (element_output_type*)handle->scratch + handle->desc.C * 2, nBlocksOFm, bn, lpb);

    /* Copy in weights and doutput in a padded buffer */
    for (ifm1ofm1 = transpose_thr_begin; ifm1ofm1 < transpose_thr_end; ++ifm1ofm1) {
      ofm1 = ifm1ofm1 / nBlocksIFm;
      ifm1 = ifm1ofm1 % nBlocksIFm;
      ofm2 = 0;
      for (ifm2 = 0; ifm2 < bc; ++ifm2) {
        LIBXSMM_VLA_ACCESS(5, filter_tr_padded, ifm1, ofm1, ofm2/lpb, ifm2, ofm2%lpb, nBlocksOFm, 1, bc, lpb) = LIBXSMM_VLA_ACCESS(5, filter,  ofm1, ifm1, ifm2/lpb, ofm2, ifm2%lpb, nBlocksIFm, bc_lp, bk, lpb);
        LIBXSMM_VLA_ACCESS(5, filter_tr_padded, ifm1, ofm1, ofm2/lpb, ifm2, 1, nBlocksOFm, 1, bc, lpb) = (element_filter_type)0;
      }
    }

    for (mb1ofm1 = copy_thr_begin; mb1ofm1 < copy_thr_end; ++mb1ofm1) {
      mb1 = mb1ofm1 / nBlocksOFm;
      ofm1 = mb1ofm1 % nBlocksOFm;
      ofm2 = 0;
      for (mb2 = 0; mb2 < bn; ++mb2) {
        LIBXSMM_VLA_ACCESS(4, doutput_padded,   mb1,  ofm1, mb2, 0, nBlocksOFm, bn, 2) = LIBXSMM_VLA_ACCESS(4, doutput,   mb1,  ofm1, mb2, 0, nBlocksOFm, bn, bk);
        LIBXSMM_VLA_ACCESS(4, doutput_padded,   mb1,  ofm1, mb2, 1, nBlocksOFm, bn, 2) = (element_output_type)0;
      }
    }

    libxsmm_barrier_wait(handle->barrier, ltid);

    for ( mb1ifm1 = thr_begin; mb1ifm1 < thr_end; ++mb1ifm1 ) {
      mb1  = mb1ifm1%nBlocksMB;
      ifm1 = mb1ifm1/nBlocksMB;
      batchreduce_kernel_bwd_zerobeta( &LIBXSMM_VLA_ACCESS(5, filter_tr_padded, ifm1, 0, 0, 0, 0, nBlocksOFm, 1, bc, lpb),
          &LIBXSMM_VLA_ACCESS(4, doutput_padded,   mb1,  0, 0, 0, nBlocksOFm, bn, 2),
          &LIBXSMM_VLA_ACCESS(4, dinput,    mb1,  ifm1, 0, 0, nBlocksIFm, bn, bc), &blocks);
    }
  }

  libxsmm_barrier_wait(handle->barrier, ltid);
}

if ( (kind == LIBXSMM_DNN_COMPUTE_KIND_UPD) || (kind == LIBXSMM_DNN_COMPUTE_KIND_BWDUPD) ) {
  /* number of tasks that could be run in parallel */
  const int ofm_subtasks = (handle->upd_2d_blocking == 1) ? 1 : handle->ofm_subtasks;
  const int ifm_subtasks = (handle->upd_2d_blocking == 1) ? 1 : handle->ifm_subtasks;
  const int bbk = (handle->upd_2d_blocking == 1) ? bk : bk/ofm_subtasks;
  const int bbc = (handle->upd_2d_blocking == 1) ? bc : bc/ifm_subtasks;
  const int work = nBlocksIFm * ifm_subtasks * nBlocksOFm * ofm_subtasks;
  const int Cck_work = nBlocksIFm * ifm_subtasks * ofm_subtasks;
  const int Cc_work = nBlocksIFm * ifm_subtasks;

  /* 2D blocking parameters  */
  int use_2d_blocking = handle->upd_2d_blocking;
  int im_tasks_per_thread = 0, in_tasks_per_thread = 0, my_in_start = 0, my_in_end = 0, my_im_start = 0, my_im_end = 0, my_row_id = 0, my_col_id = 0, row_teams = 0, column_teams = 0;

  /* compute chunk size */
  const int chunksize = (work % handle->desc.threads == 0) ? (work / handle->desc.threads) : ((work / handle->desc.threads) + 1);
  /* compute thr_begin and thr_end */
  const int thr_begin = (ltid * chunksize < work) ? (ltid * chunksize) : work;
  const int thr_end = ((ltid + 1) * chunksize < work) ? ((ltid + 1) * chunksize) : work;
  int BF = handle->upd_bf;

  /* loop variables */
  int ifm1ofm1 = 0, ifm1 = 0, ifm2 = 0, bfn = 0, ii = 0, jj = 0, mb1ifm1 = 0, jc = 0, jk = 0;

  /* Batch reduce related variables */
  unsigned long long  blocks = nBlocksMB/BF;

  LIBXSMM_VLA_DECL(4, const element_input_type,  input,    (element_input_type* )handle->reg_input->data, nBlocksIFm, bn, bc);
  LIBXSMM_VLA_DECL(5,       element_filter_type, dfilter,  (element_filter_type*)handle->grad_filter->data, nBlocksIFm, bc_lp, bk, lpb);

  /* Set up tensors for transposing/scratch before vnni reformatting dfilter */
  element_input_type  *tr_inp_ptr = (element_input_type*) ((element_output_type*)handle->scratch + handle->desc.N * handle->desc.K);
  float               *dfilter_f32_ptr = (float*) ((element_input_type*)tr_inp_ptr + handle->desc.N * handle->desc.C);
  element_filter_type *dfilter_scratch = (element_filter_type*) ((float*)dfilter_f32_ptr + handle->desc.C * handle->desc.K) + ltid * bc * bk;

  LIBXSMM_VLA_DECL(4, element_input_type,  input_tr,    (element_input_type*)tr_inp_ptr, nBlocksMB, bc, bn);
  LIBXSMM_VLA_DECL(4,       float, dfilter_f32,  (float*)dfilter_f32_ptr, nBlocksIFm, bc, bk);
  LIBXSMM_VLA_DECL(2, element_filter_type, dfilter_block,  (element_filter_type*)dfilter_scratch, bk);

  const int tr_out_work = nBlocksMB * nBlocksOFm;
  const int tr_out_chunksize = (tr_out_work % handle->desc.threads == 0) ? (tr_out_work / handle->desc.threads) : ((tr_out_work / handle->desc.threads) + 1);
  const int tr_out_thr_begin = (ltid * tr_out_chunksize < tr_out_work) ? (ltid * tr_out_chunksize) : tr_out_work;
  const int tr_out_thr_end = ((ltid + 1) * tr_out_chunksize < tr_out_work) ? ((ltid + 1) * tr_out_chunksize) : tr_out_work;

  const int tr_inp_work = nBlocksMB * nBlocksIFm;
  const int tr_inp_chunksize = (tr_inp_work % handle->desc.threads == 0) ? (tr_inp_work / handle->desc.threads) : ((tr_inp_work / handle->desc.threads) + 1);
  const int tr_inp_thr_begin = (ltid * tr_inp_chunksize < tr_inp_work) ? (ltid * tr_inp_chunksize) : tr_inp_work;
  const int tr_inp_thr_end = ((ltid + 1) * tr_inp_chunksize < tr_inp_work) ? ((ltid + 1) * tr_inp_chunksize) : tr_inp_work;

  /* These are used for the vnni reformatting of the f32 output  */
  __m256i c0, c1;
  __m512 a01, b01;
  __m512i c01 = LIBXSMM_INTRINSICS_MM512_UNDEFINED_EPI32();
  const __m512i perm_index = LIBXSMM_INTRINSICS_MM512_SET_EPI16(31, 15, 30, 14, 29, 13, 28, 12, 27, 11, 26, 10, 25, 9, 24, 8, 23, 7, 22, 6, 21, 5, 20, 4, 19, 3, 18, 2, 17, 1, 16, 0);

  if (use_2d_blocking == 1) {
    row_teams = handle->upd_row_teams;
    column_teams = handle->upd_column_teams;
    my_col_id = ltid % column_teams;
    my_row_id = ltid / column_teams;
    im_tasks_per_thread = LIBXSMM_UPDIV(nBlocksIFm, row_teams);
    in_tasks_per_thread = LIBXSMM_UPDIV(nBlocksOFm, column_teams);
    my_im_start = LIBXSMM_MIN(my_row_id * im_tasks_per_thread, nBlocksIFm);
    my_im_end = LIBXSMM_MIN((my_row_id+1) * im_tasks_per_thread, nBlocksIFm);
    my_in_start = LIBXSMM_MIN(my_col_id * in_tasks_per_thread, nBlocksOFm);
    my_in_end = LIBXSMM_MIN((my_col_id+1) * in_tasks_per_thread, nBlocksOFm);
  }

  /* Required upfront tranposes */
  if (bc % 32 == 0) {
    for (mb1ifm1 = tr_inp_thr_begin; mb1ifm1 < tr_inp_thr_end; mb1ifm1++) {
      mb1 = mb1ifm1%nBlocksMB;
      ifm1 = mb1ifm1/nBlocksMB;
      bf16_transpose((element_input_type*)&LIBXSMM_VLA_ACCESS(4, input, mb1, ifm1, 0, 0, nBlocksIFm, bn, bc), &LIBXSMM_VLA_ACCESS(4, input_tr, ifm1, mb1, 0, 0, nBlocksMB, bc, bn), bc, bn, bc, bn);
    }
  } else {
    for (mb1ifm1 = tr_inp_thr_begin; mb1ifm1 < tr_inp_thr_end; mb1ifm1++) {
      mb1 = mb1ifm1%nBlocksMB;
      ifm1 = mb1ifm1/nBlocksMB;
      for (mb2 = 0; mb2 < bn; mb2++) {
        for (ifm2 = 0; ifm2 < bc; ifm2++) {
          LIBXSMM_VLA_ACCESS(4, input_tr, ifm1, mb1, ifm2, mb2, nBlocksMB, bc, bn) = LIBXSMM_VLA_ACCESS(4, input, mb1, ifm1, mb2, ifm2, nBlocksIFm, bn, bc);
        }
      }
    }
  }

  if (performed_doutput_transpose == 0) {
    if (bk % 32 == 0) {
      for (mb1ofm1 = tr_out_thr_begin; mb1ofm1 < tr_out_thr_end; mb1ofm1++) {
        mb1 = mb1ofm1%nBlocksMB;
        ofm1 = mb1ofm1/nBlocksMB;
        bf16_vnni_reformat((element_output_type*)&LIBXSMM_VLA_ACCESS(4, doutput,  mb1, ofm1, 0, 0, nBlocksOFm, bn, bk), &LIBXSMM_VLA_ACCESS(5, doutput_tr, ofm1, mb1, 0, 0, 0, nBlocksMB, bn_lp, bk, lpb), bk, bn, bk, bn);
      }
    } else {
      for (mb1ofm1 = tr_out_thr_begin; mb1ofm1 < tr_out_thr_end; mb1ofm1++) {
        mb1 = mb1ofm1%nBlocksMB;
        ofm1 = mb1ofm1/nBlocksMB;
        for (mb2 = 0; mb2 < bn; mb2++) {
          for (ofm2 = 0; ofm2 < bk; ofm2++) {
            LIBXSMM_VLA_ACCESS(5, doutput_tr, ofm1, mb1, mb2/lpb, ofm2, mb2%lpb, nBlocksMB, bn_lp, bk, lpb) = LIBXSMM_VLA_ACCESS(4, doutput,  mb1, ofm1, mb2, ofm2, nBlocksOFm, bn, bk);
          }
        }
      }
    }
  }

  libxsmm_barrier_wait(handle->barrier, ltid);

  if (use_2d_blocking == 1) {
    if (BF == 1) {
      for (ofm1 = my_in_start; ofm1 < my_in_end; ++ofm1) {
        for (ifm1 = my_im_start; ifm1 < my_im_end; ++ifm1) {
          batchreduce_kernel_upd_zerobeta(&LIBXSMM_VLA_ACCESS(5, doutput_tr, ofm1, 0, 0, 0, 0, nBlocksMB, bn_lp, bk, lpb), &LIBXSMM_VLA_ACCESS(4, input_tr, ifm1, 0, 0, 0, nBlocksMB, bc, bn), &LIBXSMM_VLA_ACCESS(2, dfilter_block, 0, 0, bk), &blocks);
          /* TODO: Make this vnni reformating in the kernel...  */
          /* Copy result back to vnni format */
          if ((bc % 2 == 0) && (bk % 16 == 0)) {
            for (jc = 0; jc < bc; jc+=2) {
              for (jk = 0; jk < bk; jk+=16) {
                c1 = _mm256_loadu_si256((__m256i*)&LIBXSMM_VLA_ACCESS(2, dfilter_block, jc+1,jk, bk));
                c0 = _mm256_loadu_si256((__m256i*)&LIBXSMM_VLA_ACCESS(2, dfilter_block, jc, jk, bk));
                c01 = _mm512_inserti64x4(c01, c0, 0);
                c01 = _mm512_inserti64x4(c01, c1, 1);
                _mm512_storeu_si512(&LIBXSMM_VLA_ACCESS(5, dfilter, ofm1, ifm1, jc/lpb, jk, 0, nBlocksIFm, bc_lp, bk, lpb), _mm512_permutexvar_epi16(perm_index, c01));
              }
            }
          } else {
            for (ii = 0; ii < bc; ii++) {
              for (jj = 0; jj < bk; jj++) {
                LIBXSMM_VLA_ACCESS(5, dfilter, ofm1, ifm1, ii/lpb, jj, ii%lpb, nBlocksIFm, bc_lp, bk, lpb) = LIBXSMM_VLA_ACCESS(2, dfilter_block, ii, jj, bk);
              }
            }
          }
        }
      }
    } else {
      for (bfn = 0; bfn < BF; bfn++) {
        for (ofm1 = my_in_start; ofm1 < my_in_end; ++ofm1) {
          for (ifm1 = my_im_start; ifm1 < my_im_end; ++ifm1) {
            /* initialize current work task to zero */
            if (bfn == 0) {
              for (ii = 0; ii<bc; ii++) {
                for (jj = 0; jj<bk; jj++) {
                  LIBXSMM_VLA_ACCESS(4, dfilter_f32, ofm1, ifm1, ii, jj, nBlocksIFm, bc, bk) = 0;
                }
              }
            }
            batchreduce_kernel_upd(&LIBXSMM_VLA_ACCESS(5, doutput_tr, ofm1, bfn*blocks, 0, 0, 0, nBlocksMB, bn_lp, bk, lpb), &LIBXSMM_VLA_ACCESS(4, input_tr, ifm1, bfn*blocks, 0, 0, nBlocksMB, bc, bn), &LIBXSMM_VLA_ACCESS(4, dfilter_f32, ofm1, ifm1, 0, 0, nBlocksIFm, bc, bk), &blocks);
            /* Downconvert result to BF16 and vnni format */
            if (bfn == BF-1) {
              if ((bc % 2 == 0) && (bk % 16 == 0)) {
                for (jc = 0; jc < bc; jc+=2) {
                  for (jk = 0; jk < bk; jk+=16) {
                    a01 = LIBXSMM_INTRINSICS_MM512_LOAD_PS(&LIBXSMM_VLA_ACCESS(4, dfilter_f32, ofm1, ifm1, jc+1, jk, nBlocksIFm, bc, bk));
                    b01 = LIBXSMM_INTRINSICS_MM512_LOAD_PS(&LIBXSMM_VLA_ACCESS(4, dfilter_f32, ofm1, ifm1, jc, jk, nBlocksIFm, bc, bk));
                    c01 = LIBXSMM_INTRINSISCS_MM512_CVTNE2PS_PBH(a01, b01);
                    _mm512_storeu_si512(&LIBXSMM_VLA_ACCESS(5, dfilter, ofm1, ifm1, jc/lpb, jk, 0, nBlocksIFm, bc_lp, bk, lpb), _mm512_permutexvar_epi16(perm_index, c01));
                  }
                }
              } else {
                for (jc = 0; jc < bc; jc++) {
                  LIBXSMM_DNN_CONVERT_BUFFER_F32_BF16(&LIBXSMM_VLA_ACCESS(4, dfilter_f32, ofm1, ifm1, jc, 0, nBlocksIFm, bc, bk), &LIBXSMM_VLA_ACCESS(2, dfilter_block, jc, 0, bk), bk);
                }
                for (ii = 0; ii < bc; ii++) {
                  for (jj = 0; jj < bk; jj++) {
                    LIBXSMM_VLA_ACCESS(5, dfilter, ofm1, ifm1, ii/lpb, jj, ii%lpb, nBlocksIFm, bc_lp, bk, lpb) = LIBXSMM_VLA_ACCESS(2, dfilter_block, ii, jj, bk);
                  }
                }
              }
            }
          }
        }
      }
    }
  } else {
    if (BF == 1) {
      for ( ifm1ofm1 = thr_begin; ifm1ofm1 < thr_end; ++ifm1ofm1 ) {
        ofm1 = ifm1ofm1 / Cck_work;
        ofm2 = (ifm1ofm1 % Cck_work) / Cc_work;
        ifm1 = ((ifm1ofm1 % Cck_work) % Cc_work) / ifm_subtasks;
        ifm2 = ((ifm1ofm1 % Cck_work) % Cc_work) % ifm_subtasks;
        batchreduce_kernel_upd_zerobeta(&LIBXSMM_VLA_ACCESS(5, doutput_tr, ofm1, 0, 0, ofm2*bbk, 0, nBlocksMB, bn_lp, bk, lpb), &LIBXSMM_VLA_ACCESS(4, input_tr, ifm1, 0, ifm2*bbc, 0, nBlocksMB, bc, bn), &LIBXSMM_VLA_ACCESS(2, dfilter_block, ifm2*bbc, ofm2*bbk, bk), &blocks);
        /* TODO: Make this vnni reformating in the kernel...  */
        /* Copy result back to vnni format */
        if ((bbc % 2 == 0) && (bbk % 16 == 0)) {
          for (jc = 0; jc < bbc; jc+=2) {
            for (jk = 0; jk < bbk; jk+=16) {
              c1 = _mm256_loadu_si256((__m256i*)&LIBXSMM_VLA_ACCESS(2, dfilter_block, ifm2*bbc+jc+1, ofm2*bbk+jk, bk));
              c0 = _mm256_loadu_si256((__m256i*)&LIBXSMM_VLA_ACCESS(2, dfilter_block, ifm2*bbc+jc, ofm2*bbk+jk, bk));
              c01 = _mm512_inserti64x4(c01, c0, 0);
              c01 = _mm512_inserti64x4(c01, c1, 1);
              _mm512_storeu_si512(&LIBXSMM_VLA_ACCESS(5, dfilter, ofm1, ifm1, (ifm2*bbc+jc)/lpb, ofm2*bbk+jk, 0, nBlocksIFm, bc_lp, bk, lpb), _mm512_permutexvar_epi16(perm_index, c01));
            }
          }
        } else {
          for (ii = 0; ii < bbc; ii++) {
            for (jj = 0; jj < bbk; jj++) {
              LIBXSMM_VLA_ACCESS(5, dfilter, ofm1, ifm1, (ifm2*bbc+ii)/lpb, ofm2*bbk+jj, (ifm2*bbc+ii)%lpb, nBlocksIFm, bc_lp, bk, lpb) = LIBXSMM_VLA_ACCESS(2, dfilter_block, ifm2*bbc+ii, ofm2*bbk+jj, bk);
            }
          }
        }
      }
    } else {
      for (bfn = 0; bfn < BF; bfn++) {
        for ( ifm1ofm1 = thr_begin; ifm1ofm1 < thr_end; ++ifm1ofm1 ) {
          ofm1 = ifm1ofm1 / Cck_work;
          ofm2 = (ifm1ofm1 % Cck_work) / Cc_work;
          ifm1 = ((ifm1ofm1 % Cck_work) % Cc_work) / ifm_subtasks;
          ifm2 = ((ifm1ofm1 % Cck_work) % Cc_work) % ifm_subtasks;
          /* initialize current work task to zero */
          if (bfn == 0) {
            for (ii = 0; ii<bbc; ii++) {
              for (jj = 0; jj<bbk; jj++) {
                LIBXSMM_VLA_ACCESS(4, dfilter_f32, ofm1, ifm1, ifm2*bbc+ii, ofm2*bbk+jj, nBlocksIFm, bc, bk) = 0;
              }
            }
          }
          batchreduce_kernel_upd(&LIBXSMM_VLA_ACCESS(5, doutput_tr, ofm1, bfn*blocks, 0, ofm2*bbk, 0, nBlocksMB, bn_lp, bk, lpb), &LIBXSMM_VLA_ACCESS(4, input_tr, ifm1, bfn*blocks, ifm2*bbc, 0, nBlocksMB, bc, bn), &LIBXSMM_VLA_ACCESS(4, dfilter_f32, ofm1, ifm1, ifm2*bbc, ofm2*bbk, nBlocksIFm, bc, bk), &blocks);
          /* Downconvert result to BF16 and vnni format */
          if (bfn == BF-1) {
            if ((bbc % 2 == 0) && (bbk % 16 == 0)) {
              for (jc = 0; jc < bbc; jc+=2) {
                for (jk = 0; jk < bbk; jk+=16) {
                  a01 = LIBXSMM_INTRINSICS_MM512_LOAD_PS(&LIBXSMM_VLA_ACCESS(4, dfilter_f32, ofm1, ifm1, ifm2*bbc+jc+1, ofm2*bbk+jk, nBlocksIFm, bc, bk));
                  b01 = LIBXSMM_INTRINSICS_MM512_LOAD_PS(&LIBXSMM_VLA_ACCESS(4, dfilter_f32, ofm1, ifm1, ifm2*bbc+jc, ofm2*bbk+jk, nBlocksIFm, bc, bk));
                  c01 = LIBXSMM_INTRINSISCS_MM512_CVTNE2PS_PBH(a01, b01);
                  _mm512_storeu_si512(&LIBXSMM_VLA_ACCESS(5, dfilter, ofm1, ifm1, (ifm2*bbc+jc)/lpb, ofm2*bbk+jk, 0, nBlocksIFm, bc_lp, bk, lpb), _mm512_permutexvar_epi16(perm_index, c01));
                }
              }
            } else {
              for (jc = 0; jc < bbc; jc++) {
                LIBXSMM_DNN_CONVERT_BUFFER_F32_BF16(&LIBXSMM_VLA_ACCESS(4, dfilter_f32, ofm1, ifm1, ifm2*bbc+jc, ofm2*bbk, nBlocksIFm, bc, bk), &LIBXSMM_VLA_ACCESS(2, dfilter_block, ifm2*bbc+jc, ofm2*bbk, bk), bbk);
              }
              for (ii = 0; ii < bbc; ii++) {
                for (jj = 0; jj < bbk; jj++) {
                  LIBXSMM_VLA_ACCESS(5, dfilter, ofm1, ifm1, (ifm2*bbc+ii)/lpb, ofm2*bbk+jj, (ifm2*bbc+ii)%lpb, nBlocksIFm, bc_lp, bk, lpb) = LIBXSMM_VLA_ACCESS(2, dfilter_block, ifm2*bbc+ii, ofm2*bbk+jj, bk);
                }
              }
            }
          }
        }
      }
    }
  }
  libxsmm_barrier_wait(handle->barrier, ltid);
}

