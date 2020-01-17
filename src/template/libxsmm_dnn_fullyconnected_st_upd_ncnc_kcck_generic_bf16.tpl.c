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
#define _mm512_cvt2_fp32_bf16(A, B) LIBXSMM_INTRINSICS_MM512_CVT2_FP32_BF16(A, B)

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

/* compute chunk size */
const int chunksize = (work % handle->desc.threads == 0) ? (work / handle->desc.threads) : ((work / handle->desc.threads) + 1);
/* compute thr_begin and thr_end */
const int thr_begin = (ltid * chunksize < work) ? (ltid * chunksize) : work;
const int thr_end = ((ltid + 1) * chunksize < work) ? ((ltid + 1) * chunksize) : work;
const int BF = 1;//((handle->desc.N == 2048) && (nBlocksMB % 4 == 0)) ? 4 : 1;

/* loop variables */
int mb1 = 0, ifm1ofm1 = 0, ofm1 = 0, ifm1 = 0, ofm2 = 0, ifm2 = 0, bfn = 0, ii = 0, jj = 0, mb1ofm1 = 0, mb1ifm1 = 0, mb2 = 0, jc = 0, jk = 0;

/* Batch reduce related variables */
unsigned long long  blocks = nBlocksMB/BF;

LIBXSMM_VLA_DECL(4, const element_input_type,  input,    (element_input_type* )handle->reg_input->data, nBlocksIFm, bn, bc);
LIBXSMM_VLA_DECL(4, const element_output_type, doutput,  (element_output_type*)handle->grad_output->data, nBlocksOFm, bn, bk);
LIBXSMM_VLA_DECL(5,       element_filter_type, dfilter,  (element_filter_type*)handle->grad_filter->data, nBlocksIFm, bc/2, bk, 2);

/* Set up tensors for transposing/scratch before vnni reformatting dfilter */
element_output_type *tr_out_ptr = (element_output_type*)handle->scratch;
element_input_type  *tr_inp_ptr = (element_input_type*) ((element_output_type*)tr_out_ptr + handle->desc.N * handle->desc.K);
float               *dfilter_f32_ptr = (float*) ((element_input_type*)tr_inp_ptr + handle->desc.N * handle->desc.C);
element_filter_type *dfilter_scratch = (element_filter_type*) ((float*)dfilter_f32_ptr + handle->desc.C * handle->desc.K) + ltid * bc * bk;

LIBXSMM_VLA_DECL(4, element_input_type,  input_tr,    (element_input_type*)tr_inp_ptr, nBlocksMB, bc, bn);
LIBXSMM_VLA_DECL(5, element_output_type, doutput_tr,  (element_output_type*)tr_out_ptr, nBlocksMB, bn/2, bk, 2);
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

/* lazy barrier init */
libxsmm_barrier_init(handle->barrier, ltid);

/* Required upfront tranposes */

for (mb1ofm1 = tr_out_thr_begin; mb1ofm1 < tr_out_thr_end; mb1ofm1++) {
  mb1 = mb1ofm1/nBlocksOFm;
  ofm1 = mb1ofm1%nBlocksOFm;
  for (mb2 = 0; mb2 < bn; mb2++) {
    for (ofm2 = 0; ofm2 < bk; ofm2++) {
      LIBXSMM_VLA_ACCESS(5, doutput_tr, ofm1, mb1, mb2/2, ofm2, mb2%2, nBlocksMB, bn/2, bk, 2) = LIBXSMM_VLA_ACCESS(4, doutput,  mb1, ofm1, mb2, ofm2, nBlocksOFm, bn, bk);
    }
  }
}

for (mb1ifm1 = tr_inp_thr_begin; mb1ifm1 < tr_inp_thr_end; mb1ifm1++) {
  mb1 = mb1ifm1/nBlocksIFm;
  ifm1 = mb1ifm1%nBlocksIFm;
  for (mb2 = 0; mb2 < bn; mb2++) {
    for (ifm2 = 0; ifm2 < bc; ifm2++) {
      LIBXSMM_VLA_ACCESS(4, input_tr, ifm1, mb1, ifm2, mb2, nBlocksMB, bc, bn) = LIBXSMM_VLA_ACCESS(4, input, mb1, ifm1, mb2, ifm2, nBlocksIFm, bn, bc);
    }
  }
}

libxsmm_barrier_wait(handle->barrier, ltid);

if (BF == 1) {
  for ( ifm1ofm1 = thr_begin; ifm1ofm1 < thr_end; ++ifm1ofm1 ) {
    ofm1 = ifm1ofm1 / Cck_work;
    ofm2 = (ifm1ofm1 % Cck_work) / Cc_work;
    ifm1 = ((ifm1ofm1 % Cck_work) % Cc_work) / ifm_subtasks;
    ifm2 = ((ifm1ofm1 % Cck_work) % Cc_work) % ifm_subtasks;
    batchreduce_kernel_zerobeta(&LIBXSMM_VLA_ACCESS(5, doutput_tr, ofm1, 0, 0, ofm2*bbk, 0, nBlocksMB, bn/2, bk, 2), &LIBXSMM_VLA_ACCESS(4, input_tr, ifm1, 0, ifm2*bbc, 0, nBlocksMB, bc, bn), &LIBXSMM_VLA_ACCESS(2, dfilter_block, ifm2*bbc, ofm2*bbk, bk), &blocks);
    /* TODO: Make this vnni reformating in the kernel...  */
    /* Copy result back to vnni format */
    for (ii = 0; ii < bbc; ii++) {
      for (jj = 0; jj < bbk; jj++) {
        LIBXSMM_VLA_ACCESS(5, dfilter, ofm1, ifm1, (ifm2*bbc+ii)/2, ofm2*bbk+jj, (ifm2*bbc+ii)%2, nBlocksIFm, bc/2, bk, 2) = LIBXSMM_VLA_ACCESS(2, dfilter_block, ifm2*bbc+ii, ofm2*bbk+jj, bk);
      }
    }
  }
} else {
  __m512 a01, b01;
  __m512i c01;
  const __m512i perm_index = LIBXSMM_INTRINSICS_MM512_SET_EPI16(31, 15, 30, 14, 29, 13, 28, 12, 27, 11, 26, 10, 25, 9, 24, 8, 23, 7, 22, 6, 21, 5, 20, 4, 19, 3, 18, 2, 17, 1, 16, 0);
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
      batchreduce_kernel(&LIBXSMM_VLA_ACCESS(5, doutput_tr, ofm1, bfn*blocks, 0, ofm2*bbk, 0, nBlocksMB, bn/2, bk, 2), &LIBXSMM_VLA_ACCESS(4, input_tr, ifm1, bfn*blocks, ifm2*bbc, 0, nBlocksMB, bc, bn), &LIBXSMM_VLA_ACCESS(4, dfilter_f32, ofm1, ifm1, ifm2*bbc, ofm2*bbk, nBlocksIFm, bc, bk), &blocks);
      /* Downconvert result to BF16 and vnno format */
      if (bfn == BF-1) {
        for (jc = 0; jc < bbc; jc+=2) {
          for (jk = 0; jk < bbk; jk+=16) {
            a01 = LIBXSMM_INTRINSICS_MM512_LOAD_PS(&LIBXSMM_VLA_ACCESS(4, dfilter_f32, ofm1, ifm1, ifm2*bbc+jc+1, ofm2*bbk+jk, nBlocksIFm, bc, bk));
            b01 = LIBXSMM_INTRINSICS_MM512_LOAD_PS(&LIBXSMM_VLA_ACCESS(4, dfilter_f32, ofm1, ifm1, ifm2*bbc+jc, ofm2*bbk+jk, nBlocksIFm, bc, bk));
            c01 = _mm512_cvt2_fp32_bf16(a01, b01);
            _mm512_store_epi32(&LIBXSMM_VLA_ACCESS(5, dfilter, ofm1, ifm1, (ifm2*bbc+jc)/2, ofm2*bbk+jk, 0, nBlocksIFm, bc/2, bk, 2), _mm512_permutexvar_epi16(perm_index, c01));
          }
        }
      }
    }
  }
}

libxsmm_barrier_wait(handle->barrier, ltid);

#undef _mm512_cvt2_fp32_bf16
