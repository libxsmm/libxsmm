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
/* size variables, all const */
/* here we assume that input and output blocking is similar */
const int bn = handle->bn;
const int bk = handle->bk;
const int bc = handle->bc;
const int nBlocksIFm = handle->desc.C / bc;
const int nBlocksOFm = handle->desc.K / bk;
const int nBlocksMB  = handle->desc.N / bn;
/* computing first logical thread */
const int ltid = tid - start_thread;
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
int ofm1 = 0, ofm2 = 0, ifm1 = 0, ifm2 = 0, ifm1ofm1 = 0, mb1ifm1 = 0, mb1 = 0;

LIBXSMM_VLA_DECL(4, const element_output_type,   doutput, (element_output_type*)handle->grad_output->data, nBlocksOFm, bn, bk);
LIBXSMM_VLA_DECL(4, const element_filter_type, filter, (element_filter_type*)handle->reg_filter->data, nBlocksIFm, bc, bk);
LIBXSMM_VLA_DECL(4,        element_input_type,    dinput, (element_input_type* )handle->grad_input->data, nBlocksIFm, bn, bc);
LIBXSMM_VLA_DECL(4,       element_filter_type, filter_tr, (element_filter_type*)handle->scratch, nBlocksOFm, bk, bc);

/* Batch reduce related variables */
#ifdef ADDRESS_BRGEMM
const element_filter_type *A_array[1024];
const element_output_type *B_array[1024];
#endif
#ifdef OFFSET_BRGEMM
unsigned long long  A_offsets[1024];
unsigned long long  B_offsets[1024];
#endif
unsigned long long  blocks = nBlocksOFm;

/* lazy barrier init */
libxsmm_barrier_init(handle->barrier, ltid);
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

for ( mb1ifm1 = thr_begin; mb1ifm1 < thr_end; ++mb1ifm1 ) {
  mb1  = mb1ifm1/nBlocksIFm;
  ifm1 = mb1ifm1%nBlocksIFm;
  /* prepare arguments for batch-reduce call  */
#ifdef ADDRESS_BRGEMM
  for ( ofm1 = 0; ofm1 < nBlocksOFm; ++ofm1 ) {
    A_array[ofm1] = &LIBXSMM_VLA_ACCESS(4, filter_tr, ifm1, ofm1, 0, 0, nBlocksOFm, bk, bc);
    B_array[ofm1] = &LIBXSMM_VLA_ACCESS(4, doutput,  mb1, ofm1,  0, 0, nBlocksOFm, bn, bk);
  }
  batchreduce_kernel(A_array, B_array, &LIBXSMM_VLA_ACCESS(4, dinput, mb1, ifm1,  0, 0, nBlocksIFm, bn, bc), &blocks);
#endif
#ifdef OFFSET_BRGEMM
  /* Hoist here the offset preparation */
  for ( ofm1 = 0; ofm1 < nBlocksOFm; ++ofm1 ) {
    A_offsets[ofm1] = ofm1 * bc * bk * sizeof(element_filter_type);
    B_offsets[ofm1] = ofm1 * bn * bk * sizeof(element_input_type);
  }
  batchreduce_kernel( &LIBXSMM_VLA_ACCESS(4, filter_tr, ifm1, ofm1, 0, 0, nBlocksOFm, bk, bc),
                      &LIBXSMM_VLA_ACCESS(4, doutput,   mb1,  ofm1, 0, 0, nBlocksOFm, bn, bk),
                      &LIBXSMM_VLA_ACCESS(4, dinput,    mb1,  ifm1, 0, 0, nBlocksIFm, bn, bc), &blocks, A_offsets, B_offsets);
#endif
#ifdef STRIDE_BRGEMM
  batchreduce_kernel( &LIBXSMM_VLA_ACCESS(4, filter_tr, ifm1, ofm1, 0, 0, nBlocksOFm, bk, bc),
                      &LIBXSMM_VLA_ACCESS(4, doutput,   mb1,  ofm1, 0, 0, nBlocksOFm, bn, bk),
                      &LIBXSMM_VLA_ACCESS(4, dinput,    mb1,  ifm1, 0, 0, nBlocksIFm, bn, bc), &blocks);
#endif
}

libxsmm_barrier_wait(handle->barrier, ltid);

