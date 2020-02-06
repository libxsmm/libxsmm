/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Evangelos Georganas, Alexander Heinecke (Intel Corp.)
******************************************************************************/
/* size variables, all const */
/* here we assume that input and output blocking is similar */
const int nBlocksIFm = handle->desc.C / handle->bc;
const int nBlocksOFm = handle->desc.K / handle->bk;
const int nBlocksMB  = handle->desc.N / handle->bn;

/* computing first logical thread */
const int ltid = tid - start_thread;
/* number of tasks that could be run in parallel */
const int work = nBlocksOFm * nBlocksMB;
/* compute chunk size */
const int chunksize = (work % handle->desc.threads == 0) ? (work / handle->desc.threads) : ((work / handle->desc.threads) + 1);
/* compute thr_begin and thr_end */
const int thr_begin = (ltid * chunksize < work) ? (ltid * chunksize) : work;
const int thr_end = ((ltid + 1) * chunksize < work) ? ((ltid + 1) * chunksize) : work;

LIBXSMM_VLA_DECL(4, element_output_type,       output, (element_output_type*)handle->reg_output->data, nBlocksOFm, handle->bn, handle->bk);
LIBXSMM_VLA_DECL(4, const element_input_type,  input,  (element_input_type* )handle->reg_input->data,  nBlocksIFm, handle->bn, handle->bc);
LIBXSMM_VLA_DECL(4, const element_filter_type, filter, (element_filter_type*)handle->reg_filter->data, nBlocksIFm, handle->bc, handle->bk);
#ifdef LIBXSMM_DNN_FC_FWD_FUSE_BIAS
LIBXSMM_VLA_DECL(2, const float,               bias,   (float*)              handle->reg_bias->data,                           handle->bk);
#endif
#ifdef LIBXSMM_DNN_FC_FWD_FUSE_RELU
LIBXSMM_VLA_DECL(4, unsigned char,           relumask, (unsigned char*)      handle->relumask->data,   nBlocksOFm, handle->bn, handle->bk);
#endif
unsigned long long blocks = nBlocksIFm;
int iteri = 0, iterj = 0, ifm1 = 0, ifm2 = 0, BF = 1;
int CB_BLOCKS = nBlocksIFm, perform_2d_decomp = 0;

/* The snippet below does a 2D domain decomposition of output IF the number of threads and the number of work items are compatible */
/* TODO: For now 2D decomposition targets single socket SKX */
int row_teams = 7;
int column_teams = 4;
libxsmm_blasint my_col_id = ltid % column_teams;
libxsmm_blasint my_row_id = ltid / column_teams;
int in_tasks = (int)(handle->desc.N/handle->bn);
int ik_tasks = (int)(handle->desc.K/handle->bk);
int in_tasks_per_thread = (in_tasks + row_teams-1)/row_teams;
int ik_tasks_per_thread = (ik_tasks + column_teams-1)/column_teams;
libxsmm_blasint my_in_start = LIBXSMM_MIN( my_row_id * in_tasks_per_thread, in_tasks);
libxsmm_blasint my_in_end = LIBXSMM_MIN( (my_row_id+1) * in_tasks_per_thread, in_tasks);
libxsmm_blasint my_ik_start = LIBXSMM_MIN( my_col_id * ik_tasks_per_thread, ik_tasks);
libxsmm_blasint my_ik_end = LIBXSMM_MIN( (my_col_id+1) * ik_tasks_per_thread, ik_tasks);
#ifdef STRIDE_BRGEMM
LIBXSMM_UNUSED( ifm2 );
#endif

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
if (handle->desc.C == 2048 && handle->desc.K == 1024) {
  BF = 2;
}
CB_BLOCKS = nBlocksIFm/BF;

perform_2d_decomp = (in_tasks % row_teams == 0 && ik_tasks % column_teams == 0 && row_teams*column_teams == handle->desc.threads &&
  ik_tasks_per_thread*in_tasks_per_thread*CB_BLOCKS <= 4096) ? 1 : 0;

if (perform_2d_decomp) {
  /* Auxiliary arrays for batch-reduce gemms and potential prefetch */
#ifdef ADDRESS_BRGEMM
  const element_filter_type *A_array[4096];
  const element_input_type  *B_array[4096];
#endif
#ifdef OFFSET_BRGEMM
  unsigned long long  A_offsets[4096];
  unsigned long long  B_offsets[4096];
#endif
#if defined(ADDRESS_BRGEMM) || defined(OFFSET_BRGEMM)
  int index;
#endif
  int ik, in;

  /* lazy barrier init */
  libxsmm_barrier_init(handle->barrier, (int)ltid);

  /* All data is in column-major format */
  for ( ifm1 = 0; ifm1 < BF; ++ifm1 ) {
    /* Prepare arrays for the batch-reduce calls */
    for ( ik = my_ik_start; ik < my_ik_end; ++ik ) {
      for ( in = my_in_start; in < my_in_end; ++in ) {
        if ( 0 == ifm1 ) {
          for ( iteri = 0; iteri < handle->bn; ++iteri ) {
            for ( iterj = 0; iterj < handle->bk; ++iterj ) {
              LIBXSMM_VLA_ACCESS(4, output, in, ik, iteri, iterj, nBlocksOFm, handle->bn, handle->bk) = 0;
            }
          }
        }
        /* prepare arguments for batch-reduce call */
        for ( ifm2 = 0; ifm2 < CB_BLOCKS; ++ifm2 ) {
#if defined(ADDRESS_BRGEMM) || defined(OFFSET_BRGEMM)
          index = (ik-my_ik_start)*(my_in_end-my_in_start)*CB_BLOCKS + (in-my_in_start)*CB_BLOCKS + ifm2;
#endif
#ifdef ADDRESS_BRGEMM
          A_array[index] = &LIBXSMM_VLA_ACCESS(4, filter, ik, ifm2 + ifm1*CB_BLOCKS, 0, 0, nBlocksIFm, handle->bc, handle->bk);
          B_array[index] = &LIBXSMM_VLA_ACCESS(4, input,  in, ifm2 + ifm1*CB_BLOCKS, 0, 0, nBlocksIFm, handle->bn, handle->bc);
#endif
#ifdef OFFSET_BRGEMM
          A_offsets[index] = (ifm2 + ifm1*CB_BLOCKS) * handle->bc * handle->bk * sizeof(element_filter_type);
          B_offsets[index] = (ifm2 + ifm1*CB_BLOCKS) * handle->bn * handle->bk * sizeof(element_input_type);
#endif
        }
      }
    }
    /* let's run the cell in blocks for good locality */
    for ( ik = my_ik_start; ik < my_ik_end; ++ik ) {
      for ( in = my_in_start; in < my_in_end; ++in ) {
        blocks = CB_BLOCKS;
#ifdef ADDRESS_BRGEMM
        index = (ik-my_ik_start)*(my_in_end-my_in_start)*CB_BLOCKS + (in-my_in_start)*CB_BLOCKS;
        batchreduce_kernel(&A_array[index], &B_array[index], &LIBXSMM_VLA_ACCESS(4, output, in, ik, 0, 0, nBlocksOFm, handle->bn, handle->bk), &blocks);
#endif
#ifdef OFFSET_BRGEMM
        batchreduce_kernel( &LIBXSMM_VLA_ACCESS(4, filter, ik, ifm1*CB_BLOCKS,  0, 0, nBlocksIFm, handle->bc, handle->bk),
                            &LIBXSMM_VLA_ACCESS(4, input,  in, ifm1*CB_BLOCKS,  0, 0, nBlocksIFm, handle->bn, handle->bc),
                            &LIBXSMM_VLA_ACCESS(4, output, in, ik, 0, 0, nBlocksOFm,    handle->bn, handle->bk), &blocks, A_offsets, B_offsets);
#endif
#ifdef STRIDE_BRGEMM
        batchreduce_kernel( &LIBXSMM_VLA_ACCESS(4, filter, ik, ifm1*CB_BLOCKS,  0, 0, nBlocksIFm, handle->bc, handle->bk),
                            &LIBXSMM_VLA_ACCESS(4, input,  in, ifm1*CB_BLOCKS,  0, 0, nBlocksIFm, handle->bn, handle->bc),
                            &LIBXSMM_VLA_ACCESS(4, output, in, ik, 0, 0, nBlocksOFm,    handle->bn, handle->bk), &blocks);
#endif
#ifndef LIBXSMM_DNN_FC_FWD_FUSE_NONE
        if ( ifm1 == BF-1) {
          for ( iteri = 0; iteri < handle->bn; ++iteri ) {
            for ( iterj = 0; iterj < handle->bk; ++iterj ) {
              float l_cur_out = LIBXSMM_VLA_ACCESS(4, output, in, ik, iteri, iterj, nBlocksOFm, handle->bn, handle->bk);
#ifdef LIBXSMM_DNN_FC_FWD_FUSE_BIAS
              l_cur_out += LIBXSMM_VLA_ACCESS( 2, bias, ik, iterj, handle->bk );
#endif
#ifdef LIBXSMM_DNN_FC_FWD_FUSE_RELU
              LIBXSMM_VLA_ACCESS(4, relumask, in, ik, iteri, iterj, nBlocksOFm, handle->bn, handle->bk) = ( l_cur_out > 0.0 ) ? 1 : 0;
              l_cur_out = (l_cur_out > (element_output_type)0) ? l_cur_out : (element_output_type)0;
#endif
#ifdef LIBXSMM_DNN_FC_FWD_FUSE_SIGMOID
              /* we ar using Pade 7/8 approximation */
              l_cur_out = (libxsmm_stanh_pade78( l_cur_out / 2.0f ) + 1.0f) / 2.0f;
#endif
              LIBXSMM_VLA_ACCESS(4, output, in, ik, iteri, iterj, nBlocksOFm, handle->bn, handle->bk) = l_cur_out;
            }
          }
        }
#endif
      }
    }
  }
  libxsmm_barrier_wait(handle->barrier, (int)ltid);
} else {
#ifdef ADDRESS_BRGEMM
  const element_filter_type *A_array[1024];
  const element_input_type  *B_array[1024];
#endif
#ifdef OFFSET_BRGEMM
unsigned long long  A_offsets[1024];
unsigned long long  B_offsets[1024];
#endif
  int mb1ofm1;
  /* lazy barrier init */
  libxsmm_barrier_init(handle->barrier, ltid);

  for ( ifm1 = 0; ifm1 < BF; ++ifm1 ) {
#ifdef OFFSET_BRGEMM
    /* Hoist here the offset preparation */
    for ( ifm2 = 0; ifm2 < CB_BLOCKS; ++ifm2 ) {
      A_offsets[ifm2] = (ifm2 + ifm1*CB_BLOCKS) * handle->bc * handle->bk * sizeof(element_filter_type);
      B_offsets[ifm2] = (ifm2 + ifm1*CB_BLOCKS) * handle->bn * handle->bk * sizeof(element_input_type);
    }
#endif
    for ( mb1ofm1 = thr_begin; mb1ofm1 < thr_end; ++mb1ofm1 ) {
      int mb1  = mb1ofm1%nBlocksMB;
      int ofm1 = mb1ofm1/nBlocksMB;

      if ( 0 == ifm1 ) {
        for ( iteri = 0; iteri < handle->bn; ++iteri ) {
          for ( iterj = 0; iterj < handle->bk; ++iterj ) {
            LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, iteri, iterj, nBlocksOFm, handle->bn, handle->bk) = 0;
          }
        }
      }

      blocks = CB_BLOCKS;
#ifdef ADDRESS_BRGEMM
      /* prepare arguments for batch-reduce call */
      for ( ifm2 = 0; ifm2 < CB_BLOCKS; ++ifm2 ) {
        A_array[ifm2] = &LIBXSMM_VLA_ACCESS(4, filter, ofm1, ifm2 + ifm1*CB_BLOCKS, 0, 0, nBlocksIFm, handle->bc, handle->bk);
        B_array[ifm2] = &LIBXSMM_VLA_ACCESS(4, input,  mb1,  ifm2 + ifm1*CB_BLOCKS, 0, 0, nBlocksIFm, handle->bn, handle->bc);
      }
      batchreduce_kernel(A_array, B_array, &LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, 0, 0, nBlocksOFm, handle->bn, handle->bk), &blocks);
#endif
#ifdef OFFSET_BRGEMM
      batchreduce_kernel( &LIBXSMM_VLA_ACCESS(4, filter, ofm1, ifm1*CB_BLOCKS,    0, 0, nBlocksIFm, handle->bc, handle->bk),
                          &LIBXSMM_VLA_ACCESS(4, input,  mb1,  ifm1*CB_BLOCKS,    0, 0, nBlocksIFm, handle->bn, handle->bc),
                          &LIBXSMM_VLA_ACCESS(4, output, mb1,  ofm1, 0, 0, nBlocksOFm,    handle->bn, handle->bk), &blocks, A_offsets, B_offsets);
#endif
#ifdef STRIDE_BRGEMM
      batchreduce_kernel( &LIBXSMM_VLA_ACCESS(4, filter, ofm1, ifm1*CB_BLOCKS,    0, 0, nBlocksIFm, handle->bc, handle->bk),
                          &LIBXSMM_VLA_ACCESS(4, input,  mb1,  ifm1*CB_BLOCKS,    0, 0, nBlocksIFm, handle->bn, handle->bc),
                          &LIBXSMM_VLA_ACCESS(4, output, mb1,  ofm1, 0, 0, nBlocksOFm,    handle->bn, handle->bk), &blocks);
#endif
#ifndef LIBXSMM_DNN_FC_FWD_FUSE_NONE
      if ( ifm1 == BF-1) {
        for ( iteri = 0; iteri < handle->bn; ++iteri ) {
          for ( iterj = 0; iterj < handle->bk; ++iterj ) {
            float l_cur_out = LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, iteri, iterj, nBlocksOFm, handle->bn, handle->bk);
#ifdef LIBXSMM_DNN_FC_FWD_FUSE_BIAS
            l_cur_out += LIBXSMM_VLA_ACCESS( 2, bias, ofm1, iterj, handle->bk );
#endif
#ifdef LIBXSMM_DNN_FC_FWD_FUSE_RELU
            LIBXSMM_VLA_ACCESS(4, relumask, mb1, ofm1, iteri, iterj, nBlocksOFm, handle->bn, handle->bk) = ( l_cur_out > (element_output_type)0 ) ? 1 : 0;
            l_cur_out = (l_cur_out > (element_output_type)0) ? l_cur_out : (element_output_type)0;
#endif
#ifdef LIBXSMM_DNN_FC_FWD_FUSE_SIGMOID
            /* we ar using Pade 7/8 approximation */
            l_cur_out = (libxsmm_stanh_pade78( l_cur_out / 2.0f ) + 1.0f) / 2.0f;
#endif
            LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, iteri, iterj, nBlocksOFm, handle->bn, handle->bk) = l_cur_out;
          }
        }
      }
#endif
    }
  }

  libxsmm_barrier_wait(handle->barrier, ltid);
}
