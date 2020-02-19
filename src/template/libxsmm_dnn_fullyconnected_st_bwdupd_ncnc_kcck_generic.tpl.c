/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
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
int ofm1 = 0, mb1 = 0, iteri = 0, iterj = 0;

LIBXSMM_VLA_DECL(4, const element_output_type, doutput,   (element_output_type*)handle->grad_output->data,                        nBlocksOFm, bn, bk);
#if defined(LIBXSMM_DNN_FC_BWD_FUSE_RELU) || defined(LIBXSMM_DNN_FC_BWD_FUSE_SIGMOID)
LIBXSMM_VLA_DECL(4,       element_output_type, doutput2, ((element_output_type*)handle->scratch)+(handle->desc.C*handle->desc.K), nBlocksOFm, bn, bk);
#endif

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

  for ( iteri = 0; iteri < handle->bn; ++iteri ) {
    for ( iterj = 0; iterj < handle->bk; ++iterj ) {
      float l_cur_out = LIBXSMM_VLA_ACCESS(4, doutput, mb1, ofm1, iteri, iterj, nBlocksOFm, handle->bn, handle->bk);
#ifdef LIBXSMM_DNN_FC_BWD_FUSE_RELU
      l_cur_out = (LIBXSMM_VLA_ACCESS(4, relumask, mb1, ofm1, iteri, iterj, nBlocksOFm, handle->bn, handle->bk) != 0) ? l_cur_out : (element_output_type)0;
#endif
#ifdef LIBXSMM_DNN_FC_BWD_FUSE_SIGMOID
      l_cur_out = l_cur_out*(1.0f - l_cur_out);
#endif
      LIBXSMM_VLA_ACCESS(4, doutput2, mb1, ofm1, iteri, iterj, nBlocksOFm, handle->bn, handle->bk) = l_cur_out;
    }
  }
}

/* wait for eltwise to finish */
libxsmm_barrier_wait(handle->barrier, ltid);
#endif

#if defined(LIBXSMM_DNN_FC_BWD_FUSE_BIAS)
for ( ofm1 = dbias_thr_begin; ofm1 < dbias_thr_end; ++ofm1 ) {
  for ( iterj = 0; iterj < handle->bk; ++iterj ) {
    LIBXSMM_VLA_ACCESS( 2, dbias, ofm1, iterj, handle->bk ) = 0.0f;
  }

  for ( mb1 = 0; mb1 < nBlocksMB; ++mb1 ) {
    for ( iteri = 0; iteri < handle->bn; ++iteri ) {
      for ( iterj = 0; iterj < handle->bk; ++iterj ) {
#if defined(LIBXSMM_DNN_FC_BWD_FUSE_RELU) || defined(LIBXSMM_DNN_FC_BWD_FUSE_SIGMOID)
        LIBXSMM_VLA_ACCESS( 2, dbias, ofm1, iterj, handle->bk ) += LIBXSMM_VLA_ACCESS(4, doutput2, mb1, ofm1, iteri, iterj, nBlocksOFm, handle->bn, handle->bk);
#else
        LIBXSMM_VLA_ACCESS( 2, dbias, ofm1, iterj, handle->bk ) += LIBXSMM_VLA_ACCESS(4,  doutput, mb1, ofm1, iteri, iterj, nBlocksOFm, handle->bn, handle->bk);
#endif
      }
    }
  }
}

/* wait for eltwise to finish */
libxsmm_barrier_wait(handle->barrier, ltid);
#endif

if ( (kind == LIBXSMM_DNN_COMPUTE_KIND_BWD) || (kind == LIBXSMM_DNN_COMPUTE_KIND_BWDUPD) ) {
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
  int ofm2 = 0, ifm1 = 0, ifm2 = 0, ifm1ofm1 = 0, mb1ifm1 = 0;

  LIBXSMM_VLA_DECL(4, const element_filter_type,    filter, (element_filter_type*)handle->reg_filter->data, nBlocksIFm, bc, bk);
  LIBXSMM_VLA_DECL(4,        element_input_type,    dinput, (element_input_type* )handle->grad_input->data, nBlocksIFm, bn, bc);
  LIBXSMM_VLA_DECL(4,       element_filter_type, filter_tr, (element_filter_type*)handle->scratch, nBlocksOFm, bk, bc);
 /* Batch reduce related variables */
  unsigned long long  blocks = nBlocksOFm;
  int KB_BLOCKS = nBlocksOFm, BF = 1;

  /* Blocking reduction domain if it is too large */
  if ((handle->desc.C > 1024 && handle->desc.C <= 2048) || (handle->desc.K > 1024 && handle->desc.K <= 2048)) {
    BF = 8;
    while ( (nBlocksIFm % BF != 0) || (nBlocksOFm % BF != 0) ) {
      BF--;
    }
  }
  if (handle->desc.C > 2048 || handle->desc.K > 2048) {
    BF = 16;
    while ( (nBlocksIFm % BF != 0) || (nBlocksOFm % BF != 0) ) {
      BF--;
    }
  }
  if (handle->desc.K == 2048 && handle->desc.C == 1024) {
    BF = 2;
  }
  KB_BLOCKS = nBlocksOFm/BF;

 /* transpose weight */
  for (ifm1ofm1 = transpose_thr_begin; ifm1ofm1 < transpose_thr_end; ++ifm1ofm1) {
    ofm1 = ifm1ofm1 / nBlocksIFm;
    ifm1 = ifm1ofm1 % nBlocksIFm;
    for (ofm2 = 0; ofm2 < bk; ++ofm2) {
      for (ifm2 = 0; ifm2 < bc; ++ifm2) {
        LIBXSMM_VLA_ACCESS(4, filter_tr, ifm1, ofm1, ofm2, ifm2, nBlocksOFm, bk, bc) =
          LIBXSMM_VLA_ACCESS(4, filter,  ofm1, ifm1, ifm2, ofm2, nBlocksIFm, bc, bk);
      }
    }
  }
  /* wait for transpose to finish */
  libxsmm_barrier_wait(handle->barrier, ltid);

  for ( ofm1 = 0; ofm1 < BF; ++ofm1 ) {
    for ( mb1ifm1 = thr_begin; mb1ifm1 < thr_end; ++mb1ifm1 ) {
      mb1  = mb1ifm1%nBlocksMB;
      ifm1 = mb1ifm1/nBlocksMB;

      if ( 0 == ofm1 ) {
        for ( iteri = 0; iteri < handle->bn; ++iteri ) {
          for ( iterj = 0; iterj < handle->bc; ++iterj ) {
            LIBXSMM_VLA_ACCESS(4, dinput, mb1, ifm1, iteri, iterj, nBlocksIFm, handle->bn, handle->bc) = 0;
          }
        }
      }
      blocks = KB_BLOCKS;
      batchreduce_kernel_bwd( &LIBXSMM_VLA_ACCESS(4, filter_tr, ifm1, ofm1*KB_BLOCKS, 0, 0, nBlocksOFm, bk, bc),
#if defined(LIBXSMM_DNN_FC_BWD_FUSE_RELU) || defined(LIBXSMM_DNN_FC_BWD_FUSE_SIGMOID)
                              &LIBXSMM_VLA_ACCESS(4, doutput2,   mb1,  ofm1*KB_BLOCKS, 0, 0, nBlocksOFm, bn, bk),
#else
                              &LIBXSMM_VLA_ACCESS(4, doutput,    mb1,  ofm1*KB_BLOCKS, 0, 0, nBlocksOFm, bn, bk),
#endif
                              &LIBXSMM_VLA_ACCESS(4, dinput,     mb1,  ifm1, 0, 0, nBlocksIFm, bn, bc), &blocks);
    }
  }

  libxsmm_barrier_wait(handle->barrier, ltid);
}

if ( (kind == LIBXSMM_DNN_COMPUTE_KIND_UPD) || (kind == LIBXSMM_DNN_COMPUTE_KIND_BWDUPD) ) {
  /* number of tasks that could be run in parallel */
  const int work = nBlocksIFm * nBlocksOFm;
  /* compute chunk size */
  const int chunksize = (work % handle->desc.threads == 0) ? (work / handle->desc.threads) : ((work / handle->desc.threads) + 1);
  /* compute thr_begin and thr_end */
  const int thr_begin = (ltid * chunksize < work) ? (ltid * chunksize) : work;
  const int thr_end = ((ltid + 1) * chunksize < work) ? ((ltid + 1) * chunksize) : work;

  /* loop variables */
  int ifm1ofm1 = 0, ifm1 = 0;

  /* Batch reduce related variables */
  const element_output_type *A_array[1024];
  const element_input_type  *B_array[1024];
  unsigned long long  blocks = nBlocksMB;

  LIBXSMM_VLA_DECL(4, const element_input_type,  input,    (element_input_type* )handle->reg_input->data, nBlocksIFm, bn, bc);
  LIBXSMM_VLA_DECL(4,       element_filter_type, dfilter,  (element_filter_type*)handle->grad_filter->data, nBlocksIFm, bc, bk);

  for ( ifm1ofm1 = thr_begin; ifm1ofm1 < thr_end; ++ifm1ofm1 ) {
    ofm1 = ifm1ofm1 / nBlocksIFm;
    ifm1 = ifm1ofm1 % nBlocksIFm;
    /* prepare arguments for batch-reduce call  */
    for ( mb1 = 0; mb1 < nBlocksMB; ++mb1 ) {
#if defined(LIBXSMM_DNN_FC_BWD_FUSE_RELU) || defined(LIBXSMM_DNN_FC_BWD_FUSE_SIGMOID)
      A_array[mb1] = &LIBXSMM_VLA_ACCESS(4, doutput2,  mb1, ofm1,  0, 0, nBlocksOFm, bn, bk);
#else
      A_array[mb1] = &LIBXSMM_VLA_ACCESS(4, doutput,   mb1, ofm1,  0, 0, nBlocksOFm, bn, bk);
#endif
      B_array[mb1] = &LIBXSMM_VLA_ACCESS(4, input, mb1, ifm1, 0, 0, nBlocksIFm, bn, bc);
    }
    batchreduce_kernel_upd(A_array, B_array, &LIBXSMM_VLA_ACCESS(4, dfilter, ofm1, ifm1, 0, 0, nBlocksIFm, bc, bk), &blocks);
  }

  libxsmm_barrier_wait(handle->barrier, ltid);
}

