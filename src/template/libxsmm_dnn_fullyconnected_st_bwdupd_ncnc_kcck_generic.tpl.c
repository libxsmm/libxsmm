/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Evangelos Georganas, Kunal Banerjee (Intel Corp.)
******************************************************************************/

/* here we assume that input and output blocking is similar */
const int bn = handle->bn;
const int bk = handle->bk;
const int bc = handle->bc;
const int nBlocksIFm = handle->desc.C / bc;
const int nBlocksOFm = handle->desc.K / bk;
const int nBlocksMB  = handle->desc.N / bn;

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
int mb1ofm1;
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

/* loop variables */
int ofm1 = 0, mb1 = 0, ofm2 = 0, mb2 = 0;

#if defined(LIBXSMM_DNN_FC_BWD_FUSE_RELU) || defined(LIBXSMM_DNN_FC_BWD_FUSE_SIGMOID)
element_output_type *grad_output_ptr = ((element_output_type*)handle->scratch)+(handle->desc.C*handle->desc.K);
LIBXSMM_VLA_DECL(4, const element_output_type, doutput_orig, (element_output_type*)handle->grad_output->data, nBlocksOFm, bn, bk);
#else
element_output_type *grad_output_ptr = (element_output_type*)handle->grad_output->data;
#endif
LIBXSMM_VLA_DECL(4, element_output_type, doutput, grad_output_ptr, nBlocksOFm, bn, bk);

#ifdef LIBXSMM_DNN_FC_BWD_FUSE_BIAS
LIBXSMM_VLA_DECL(2,       float,                   dbias, (float*)              handle->grad_bias->data,                          handle->bk);
#endif
#ifdef LIBXSMM_DNN_FC_BWD_FUSE_RELU
LIBXSMM_VLA_DECL(4, unsigned char,              relumask, (unsigned char*)      handle->relumask->data,   nBlocksOFm, handle->bn, handle->bk);
#endif

/* lazy barrier init */
libxsmm_barrier_init(handle->barrier, ltid);

#if defined(LIBXSMM_DNN_FC_BWD_FUSE_RELU) || defined(LIBXSMM_DNN_FC_BWD_FUSE_SIGMOID)
for ( mb1ofm1 = eltwise_thr_begin; mb1ofm1 < eltwise_thr_end; ++mb1ofm1 ) {
  mb1  = mb1ofm1%nBlocksMB;
  ofm1 = mb1ofm1/nBlocksMB;

  for ( mb2 = 0; mb2 < handle->bn; ++mb2 ) {
    for ( ofm2 = 0; ofm2 < handle->bk; ++ofm2 ) {
      float l_cur_out = LIBXSMM_VLA_ACCESS(4, doutput_orig, mb1, ofm1, mb2, ofm2, nBlocksOFm, handle->bn, handle->bk);
#ifdef LIBXSMM_DNN_FC_BWD_FUSE_RELU
      l_cur_out = (LIBXSMM_VLA_ACCESS(4, relumask, mb1, ofm1, mb2, ofm2, nBlocksOFm, handle->bn, handle->bk) != 0) ? l_cur_out : (element_output_type)0;
#endif
#ifdef LIBXSMM_DNN_FC_BWD_FUSE_SIGMOID
      l_cur_out = l_cur_out*(1.0f - l_cur_out);
#endif
      LIBXSMM_VLA_ACCESS(4, doutput, mb1, ofm1, mb2, ofm2, nBlocksOFm, handle->bn, handle->bk) = l_cur_out;
    }
  }
}

/* wait for eltwise to finish */
libxsmm_barrier_wait(handle->barrier, ltid);
#endif

#if defined(LIBXSMM_DNN_FC_BWD_FUSE_BIAS)
for ( ofm1 = dbias_thr_begin; ofm1 < dbias_thr_end; ++ofm1 ) {
  for ( ofm2 = 0; ofm2 < handle->bk; ++ofm2 ) {
    LIBXSMM_VLA_ACCESS( 2, dbias, ofm1, ofm2, handle->bk ) = 0.0f;
  }

  for ( mb1 = 0; mb1 < nBlocksMB; ++mb1 ) {
    for ( mb2 = 0; mb2 < handle->bn; ++mb2 ) {
      for ( ofm2 = 0; ofm2 < handle->bk; ++ofm2 ) {
        LIBXSMM_VLA_ACCESS( 2, dbias, ofm1, ofm2, handle->bk ) += LIBXSMM_VLA_ACCESS(4,  doutput, mb1, ofm1, mb2, ofm2, nBlocksOFm, handle->bn, handle->bk);
      }
    }
  }
}

/* wait for eltwise to finish */
libxsmm_barrier_wait(handle->barrier, ltid);
#endif

if ( (kind == LIBXSMM_DNN_COMPUTE_KIND_BWD) || (kind == LIBXSMM_DNN_COMPUTE_KIND_BWDUPD) ) {
  const int use_2d_blocking = handle->bwd_2d_blocking;

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

  LIBXSMM_VLA_DECL(4, const element_filter_type, filter, (element_filter_type*)handle->reg_filter->data, nBlocksIFm, bc, bk);
  LIBXSMM_VLA_DECL(4,        element_input_type,    dinput, (element_input_type* )handle->grad_input->data, nBlocksIFm, bn, bc);
  LIBXSMM_VLA_DECL(4,       element_filter_type, filter_tr, (element_filter_type*)handle->scratch, nBlocksOFm, bk, bc);

  unsigned long long  blocks = nBlocksOFm;
  int KB_BLOCKS = nBlocksOFm, BF = 1;
  libxsmm_meltw_unary_param trans_param;

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

  /* transpose weight */
  for (ifm1ofm1 = transpose_thr_begin; ifm1ofm1 < transpose_thr_end; ++ifm1ofm1) {
    ofm1 = ifm1ofm1 / nBlocksIFm;
    ifm1 = ifm1ofm1 % nBlocksIFm;
    trans_param.in.primary  = (void*)&LIBXSMM_VLA_ACCESS(4,    filter, ofm1, ifm1, 0, 0, nBlocksIFm, bc, bk);
    trans_param.out.primary = &LIBXSMM_VLA_ACCESS(4, filter_tr, ifm1, ofm1, 0, 0, nBlocksOFm, bk, bc);
    handle->tr_kernel( &trans_param ) ;
#if 0
    for (ofm2 = 0; ofm2 < bk; ++ofm2) {
      for (ifm2 = 0; ifm2 < bc; ++ifm2) {
        LIBXSMM_VLA_ACCESS(4, filter_tr, ifm1, ofm1, ofm2, ifm2, nBlocksOFm, bk, bc) =
          LIBXSMM_VLA_ACCESS(4, filter,  ofm1, ifm1, ifm2, ofm2, nBlocksIFm, bc, bk);
      }
    }
#endif
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
              for ( mb2 = 0; mb2 < bn; ++mb2 ) {
                for ( ifm2 = 0; ifm2 < bc; ++ifm2 ) {
                  LIBXSMM_VLA_ACCESS(4, dinput, mb1, ifm1, mb2, ifm2, nBlocksIFm, bn, bc) = (element_input_type)0;
                }
              }
            }
            batchreduce_kernel_bwd( &LIBXSMM_VLA_ACCESS(4, filter_tr, ifm1, ofm1*KB_BLOCKS, 0, 0, nBlocksOFm, bk, bc ),
                &LIBXSMM_VLA_ACCESS(4, doutput,   mb1,  ofm1*KB_BLOCKS, 0, 0, nBlocksOFm, bn, bk),
                &LIBXSMM_VLA_ACCESS(4, dinput,    mb1,  ifm1, 0, 0, nBlocksIFm, bn, bc), &blocks);
          }
        }
      }
    } else {
      for (ifm1 = my_in_start; ifm1 < my_in_end; ++ifm1) {
        for (mb1 = my_im_start; mb1 < my_im_end; ++mb1) {
          batchreduce_kernel_bwd_zerobeta( &LIBXSMM_VLA_ACCESS(4, filter_tr, ifm1, 0, 0, 0, nBlocksOFm, bk, bc),
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
            for ( mb2 = 0; mb2 < bn; ++mb2 ) {
              for ( ifm2 = 0; ifm2 < bc; ++ifm2 ) {
                LIBXSMM_VLA_ACCESS(4, dinput, mb1, ifm1, mb2, ifm2, nBlocksIFm, bn, bc) = (element_input_type)0;
              }
            }
          }
          batchreduce_kernel_bwd( &LIBXSMM_VLA_ACCESS(4, filter_tr, ifm1, ofm1*KB_BLOCKS, 0, 0, nBlocksOFm, bk, bc ),
              &LIBXSMM_VLA_ACCESS(4, doutput,   mb1,  ofm1*KB_BLOCKS, 0, 0, nBlocksOFm, bn, bk),
              &LIBXSMM_VLA_ACCESS(4, dinput,    mb1,  ifm1, 0, 0, nBlocksIFm, bn, bc), &blocks);
        }
      }
    } else {
      for ( mb1ifm1 = thr_begin; mb1ifm1 < thr_end; ++mb1ifm1 ) {
        mb1  = mb1ifm1%nBlocksMB;
        ifm1 = mb1ifm1/nBlocksMB;
        batchreduce_kernel_bwd_zerobeta( &LIBXSMM_VLA_ACCESS(4, filter_tr, ifm1, 0, 0, 0, nBlocksOFm, bk, bc ),
            &LIBXSMM_VLA_ACCESS(4, doutput,   mb1,  0, 0, 0, nBlocksOFm, bn, bk),
            &LIBXSMM_VLA_ACCESS(4, dinput,    mb1,  ifm1, 0, 0, nBlocksIFm, bn, bc), &blocks);
      }
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
  int ifm1ofm1 = 0, ifm1 = 0, ifm2 = 0, bfn = 0, ii = 0, jj = 0;

  /* Batch reduce related variables */
  unsigned long long  blocks = nBlocksMB/BF;

  LIBXSMM_VLA_DECL(4, const element_input_type,  input,    (element_input_type* )handle->reg_input->data, nBlocksIFm, bn, bc);
  LIBXSMM_VLA_DECL(4,       element_filter_type, dfilter,  (element_filter_type*)handle->grad_filter->data, nBlocksIFm, bc, bk);

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

  if (use_2d_blocking == 1) {
    if (BF == 1) {
      for (ofm1 = my_in_start; ofm1 < my_in_end; ++ofm1) {
        for (ifm1 = my_im_start; ifm1 < my_im_end; ++ifm1) {
          batchreduce_kernel_upd_zerobeta(&LIBXSMM_VLA_ACCESS(4, doutput, 0, ofm1, 0, 0, nBlocksOFm, bn, bk),
                                          &LIBXSMM_VLA_ACCESS(4, input,   0, ifm1, 0, 0, nBlocksIFm, bn, bc),
                                          &LIBXSMM_VLA_ACCESS(4, dfilter, ofm1, ifm1, 0, 0, nBlocksIFm, bc, bk), &blocks);
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
                  LIBXSMM_VLA_ACCESS(4, dfilter, ofm1, ifm1, ii, jj, nBlocksIFm, bc, bk) = (element_filter_type)0;
                }
              }
            }
            batchreduce_kernel_upd( &LIBXSMM_VLA_ACCESS(4, doutput, bfn*blocks, ofm1, 0, 0, nBlocksOFm, bn, bk),
                                    &LIBXSMM_VLA_ACCESS(4, input,   bfn*blocks, ifm1, 0, 0, nBlocksIFm, bn, bc),
                                    &LIBXSMM_VLA_ACCESS(4, dfilter, ofm1, ifm1, 0, 0, nBlocksIFm, bc, bk), &blocks);
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

        batchreduce_kernel_upd_zerobeta( &LIBXSMM_VLA_ACCESS(4, doutput, 0, ofm1, 0, ofm2*bbk, nBlocksOFm, bn, bk),
                                         &LIBXSMM_VLA_ACCESS(4, input,   0, ifm1, 0, ifm2*bbc, nBlocksIFm, bn, bc),
                                         &LIBXSMM_VLA_ACCESS(4, dfilter, ofm1, ifm1, ifm2*bbc, ofm2*bbk, nBlocksIFm, bc, bk), &blocks);
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
                LIBXSMM_VLA_ACCESS(4, dfilter, ofm1, ifm1, ifm2*bbc+ii, ofm2*bbk+jj, nBlocksIFm, bc, bk) = (element_filter_type)0;
              }
            }
          }

          batchreduce_kernel_upd( &LIBXSMM_VLA_ACCESS(4, doutput, bfn*blocks, ofm1, 0, ofm2*bbk, nBlocksOFm, bn, bk),
                                  &LIBXSMM_VLA_ACCESS(4, input,   bfn*blocks, ifm1, 0, ifm2*bbc, nBlocksIFm, bn, bc),
                                  &LIBXSMM_VLA_ACCESS(4, dfilter, ofm1, ifm1, ifm2*bbc, ofm2*bbk, nBlocksIFm, bc, bk), &blocks);
        }
      }
    }
  }

  libxsmm_barrier_wait(handle->barrier, ltid);
}

