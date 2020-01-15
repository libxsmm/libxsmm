/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Evangelos Georganas (Intel Corp.)
******************************************************************************/
/* size variables, all const */
const int bn = handle->bn;
const int bk = handle->bk;
const int bc = handle->bc;
const int nBlocksIFm = handle->desc.C / bc;
const int nBlocksOFm = handle->desc.K / bk;
const int nBlocksMB  = handle->desc.N / bn;
/* computing first logical thread */
const int ltid = tid - start_thread;
/* number of tasks that could be run in parallel */
const int ofm_subtasks = handle->ofm_subtasks;//(handle->bk == 64 && !(handle->desc.C == 1024 && handle->desc.K == 1024)) ? 2 : 1;
const int ifm_subtasks = handle->ifm_subtasks;// (handle->bc == 64 && (handle->desc.C == 512 && handle->desc.K == 512)) ? 4 : 1;
const int bbk = bk/ofm_subtasks;
const int bbc = bc/ifm_subtasks;
const int work = nBlocksIFm * ifm_subtasks * nBlocksOFm * ofm_subtasks;
const int Cck_work = nBlocksIFm * ifm_subtasks * ofm_subtasks;
const int Cc_work = nBlocksIFm * ifm_subtasks;
const int work = nBlocksIFm * ifm_subtasks * nBlocksOFm * ofm_subtasks;

/* compute chunk size */
const int chunksize = (work % handle->desc.threads == 0) ? (work / handle->desc.threads) : ((work / handle->desc.threads) + 1);
/* compute thr_begin and thr_end */
const int thr_begin = (ltid * chunksize < work) ? (ltid * chunksize) : work;
const int thr_end = ((ltid + 1) * chunksize < work) ? ((ltid + 1) * chunksize) : work;
const int BF = ((handle->desc.N == 2048) && (nBlocksMB % 4 == 0)) ? 4 : 1;

/* loop variables */
int mb1 = 0, ifm1ofm1 = 0, ofm1 = 0, ifm1 = 0, ofm2 = 0, ifm2 = 0, bfn = 0, ii = 0, jj = 0;

/* Batch reduce related variables */
unsigned long long  blocks = nBlocksMB/BF;

LIBXSMM_VLA_DECL(4, const element_input_type,  input,    (element_input_type* )handle->reg_input->data, nBlocksIFm, bn, bc);
LIBXSMM_VLA_DECL(4, const element_output_type, doutput,  (element_output_type*)handle->grad_output->data, nBlocksOFm, bn, bk);
LIBXSMM_VLA_DECL(5,       element_filter_type, dfilter,  (element_filter_type*)handle->grad_filter->data, nBlocksIFm, bc/2, bk, 2);
/* Set up tensors for transposing/scratch before vnni reformatting dfilter */

/* lazy barrier init */
libxsmm_barrier_init(handle->barrier, ltid);

if (BF == 1) {
  for ( ifm1ofm1 = thr_begin; ifm1ofm1 < thr_end; ++ifm1ofm1 ) {
    ofm1 = ifm1ofm1 / Cck_work;
    ofm2 = (ifm1ofm1 % Cck_work) / Cc_work;
    ifm1 = ((ifm1ofm1 % Cck_work) % Cc_work) / ifm_subtasks;
    ifm2 = ((ifm1ofm1 % Cck_work) % Cc_work) % ifm_subtasks;
    batchreduce_kernel_zerobeta(&LIBXSMM_VLA_ACCESS(4, doutput,  0, ofm1,  0, ofm2*bbk, nBlocksOFm, bn, bk), &LIBXSMM_VLA_ACCESS(4, input, 0, ifm1, 0, ifm2*bbc, nBlocksIFm, bn, bc), &LIBXSMM_VLA_ACCESS(4, dfilter, ofm1, ifm1, ifm2*bbc, ofm2*bbk, nBlocksIFm, bc, bk), &blocks);
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
        for (ii = 0; ii<eff_c; ii++) {
          for (jj = 0; jj<eff_k; jj++) {
            LIBXSMM_VLA_ACCESS(4, dfilter, ofm1, ifm1, ifm2*bbc+ii, ofm2*bbk+jj, nBlocksIFm, bc, bk) = 0;
          }
        }
      }
      batchreduce_kernel(&LIBXSMM_VLA_ACCESS(4, doutput,  bfn*blocks, ofm1,  0, ofm2*bbk, nBlocksOFm, bn, bk), &LIBXSMM_VLA_ACCESS(4, input, bfn*blocks, ifm1, 0, ifm2*bbc, nBlocksIFm, bn, bc), &LIBXSMM_VLA_ACCESS(4, dfilter, ofm1, ifm1, ifm2*bbc, ofm2*bbk, nBlocksIFm, bc, bk), &blocks);

    }
  }
}

libxsmm_barrier_wait(handle->barrier, ltid);

