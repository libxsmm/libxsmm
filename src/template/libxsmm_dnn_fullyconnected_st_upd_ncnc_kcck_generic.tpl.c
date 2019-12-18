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
const int bn = handle->bn;
const int bk = handle->bk;
const int bc = handle->bc;
const int nBlocksIFm = handle->desc.C / bc;
const int nBlocksOFm = handle->desc.K / bk;
const int nBlocksMB  = handle->desc.N / bn;
/* computing first logical thread */
const int ltid = tid - start_thread;
/* number of tasks that could be run in parallel */
const int work = nBlocksIFm * nBlocksOFm;
/* compute chunk size */
const int chunksize = (work % handle->desc.threads == 0) ? (work / handle->desc.threads) : ((work / handle->desc.threads) + 1);
/* compute thr_begin and thr_end */
const int thr_begin = (ltid * chunksize < work) ? (ltid * chunksize) : work;
const int thr_end = ((ltid + 1) * chunksize < work) ? ((ltid + 1) * chunksize) : work;

/* loop variables */
int mb1 = 0, ifm1ofm1 = 0, ofm1 = 0, ifm1 = 0;

/* Batch reduce related variables */
#ifdef ADDRESS_BRGEMM
const element_output_type *A_array[1024];
const element_input_type  *B_array[1024];
#endif
#ifdef OFFSET_BRGEMM
unsigned long long  A_offsets[1024];
unsigned long long  B_offsets[1024];
#endif
unsigned long long  blocks = nBlocksMB;

LIBXSMM_VLA_DECL(4, const element_input_type,  input,    (element_input_type* )handle->reg_input->data, nBlocksIFm, bn, bc);
LIBXSMM_VLA_DECL(4, const element_output_type, doutput,  (element_output_type*)handle->grad_output->data, nBlocksOFm, bn, bk);
LIBXSMM_VLA_DECL(4,       element_filter_type, dfilter,  (element_filter_type*)handle->grad_filter->data, nBlocksIFm, bc, bk);

/* lazy barrier init */
libxsmm_barrier_init(handle->barrier, ltid);

for ( ifm1ofm1 = thr_begin; ifm1ofm1 < thr_end; ++ifm1ofm1 ) {
  ofm1 = ifm1ofm1 / nBlocksIFm;
  ifm1 = ifm1ofm1 % nBlocksIFm;
  /* prepare arguments for batch-reduce call  */
#ifdef ADDRESS_BRGEMM
  for ( mb1 = 0; mb1 < nBlocksMB; ++mb1 ) {
    A_array[mb1] = &LIBXSMM_VLA_ACCESS(4, doutput,  mb1, ofm1,  0, 0, nBlocksOFm, bn, bk);
    B_array[mb1] = &LIBXSMM_VLA_ACCESS(4, input, mb1, ifm1, 0, 0, nBlocksIFm, bn, bc);
  }
  batchreduce_kernel(A_array, B_array, &LIBXSMM_VLA_ACCESS(4, dfilter, ofm1, ifm1, 0, 0, nBlocksIFm, bc, bk), &blocks);
#endif
#ifdef OFFSET_BRGEMM
  /* Hoist here the offset preparation */
  for ( mb1 = 0; mb1 < nBlocksMB; ++mb1 ) {
    A_offsets[mb1] = mb1 * bn * bk * sizeof(element_filter_type);
    B_offsets[mb1] = mb1 * bn * bc * sizeof(element_input_type);
  }
  batchreduce_kernel( &LIBXSMM_VLA_ACCESS(4, doutput, mb1,  ofm1, 0, 0, nBlocksOFm, bn, bk),
                      &LIBXSMM_VLA_ACCESS(4, input,   mb1,  ifm1, 0, 0, nBlocksIFm, bn, bc),
                      &LIBXSMM_VLA_ACCESS(4, dfilter, ofm1, ifm1, 0, 0, nBlocksIFm, bc, bk), &blocks, A_offsets, B_offsets);
#endif
#ifdef STRIDE_BRGEMM
  batchreduce_kernel( &LIBXSMM_VLA_ACCESS(4, doutput, mb1,  ofm1, 0, 0, nBlocksOFm, bn, bk),
                      &LIBXSMM_VLA_ACCESS(4, input,   mb1,  ifm1, 0, 0, nBlocksIFm, bn, bc),
                      &LIBXSMM_VLA_ACCESS(4, dfilter, ofm1, ifm1, 0, 0, nBlocksIFm, bc, bk), &blocks);
#endif
}

libxsmm_barrier_wait(handle->barrier, ltid);

