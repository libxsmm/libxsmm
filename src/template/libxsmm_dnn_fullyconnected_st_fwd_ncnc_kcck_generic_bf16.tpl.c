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

/* loop variables */
int mb1ofm1 = 0;
int ifm1 = 0;
int img2 = 0;
int ofm2 = 0;

LIBXSMM_VLA_DECL(4, element_output_type,       output,  (element_output_type*)handle->reg_output->data, nBlocksOFm, handle->bn, handle->bk);
LIBXSMM_VLA_DECL(4, const element_input_type,  input,   (element_input_type* )handle->reg_input->data,  nBlocksIFm, handle->bn, handle->bc);
LIBXSMM_VLA_DECL(5, const element_filter_type, filter,  (element_filter_type*)handle->reg_filter->data, nBlocksIFm, handle->bc/2, handle->bk, 2);
float* temp_output = (float*)handle->scratch;
LIBXSMM_VLA_DECL(2, float,                     out_tmp, temp_output+(ltid*handle->bk*handle->bn), handle->bk );

unsigned long long  blocks = nBlocksIFm;
#ifdef ADDRESS_BRGEMM
const element_filter_type *A_array[1024];
const element_input_type  *B_array[1024];
#endif
#ifdef OFFSET_BRGEMM
unsigned long long  A_offsets[1024];
unsigned long long  B_offsets[1024];
#endif
#ifdef STRIDE_BRGEMM
LIBXSMM_UNUSED( ifm1 );
#endif

#ifdef OFFSET_BRGEMM
/* Hoist here the offset preparation */
for ( ifm1 = 0; ifm1 < nBlocksIFm; ++ifm1 ) {
  A_offsets[ifm1] = ifm1 * handle->bc/2 * handle->bk * 2 * sizeof(element_filter_type);
  B_offsets[ifm1] = ifm1 * handle->bn * handle->bk * sizeof(element_input_type);
}
#endif

/* lazy barrier init */
libxsmm_barrier_init(handle->barrier, ltid);

for ( mb1ofm1 = thr_begin; mb1ofm1 < thr_end; ++mb1ofm1 ) {
  int mb1  = mb1ofm1/nBlocksOFm;
  int ofm1 = mb1ofm1%nBlocksOFm;

#ifdef ADDRESS_BRGEMM
  /* prepare arguments for batch-reduce call  */
  for ( ifm1 = 0; ifm1 < nBlocksIFm; ++ifm1 ) {
    A_array[ifm1] = &LIBXSMM_VLA_ACCESS(5, filter, ofm1, ifm1, 0, 0, 0, nBlocksIFm, handle->bc/2, handle->bk, 2);
    B_array[ifm1] = &LIBXSMM_VLA_ACCESS(4, input,  mb1, ifm1,  0, 0, nBlocksIFm, handle->bn, handle->bc);
  }
  batchreduce_kernel(A_array, B_array, &LIBXSMM_VLA_ACCESS(2, out_tmp, 0, 0, handle->bk), &blocks);
#endif
#ifdef OFFSET_BRGEMM
  batchreduce_kernel( &LIBXSMM_VLA_ACCESS(5, filter, ofm1, 0, 0, 0, 0, nBlocksIFm, handle->bc/2, handle->bk, 2),
                      &LIBXSMM_VLA_ACCESS(4, input,  mb1, 0,  0, 0, nBlocksIFm, handle->bn, handle->bc),
                      &LIBXSMM_VLA_ACCESS(2, out_tmp, 0, 0, handle->bk), &blocks, A_offsets, B_offsets);
#endif
#ifdef STRIDE_BRGEMM
  batchreduce_kernel( &LIBXSMM_VLA_ACCESS(5, filter, ofm1, 0, 0, 0, 0, nBlocksIFm, handle->bc/2, handle->bk, 2),
                      &LIBXSMM_VLA_ACCESS(4, input,  mb1, 0,  0, 0, nBlocksIFm, handle->bn, handle->bc),
                      &LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, 0, 0, nBlocksOFm, handle->bn, handle->bk), &blocks);
#endif

#ifndef STRIDE_BRGEMM
  /* downconvert scratch to bf16 and store to final C */
  for ( img2 = 0; img2 < handle->bn; ++img2 ) {
    for ( ofm2 = 0; ofm2 < handle->bk; ofm2 += 16 ) {
      _mm256_storeu_si256( (__m256i *) &LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, img2, ofm2, nBlocksOFm, handle->bn, handle->bk),
         _mm512_cvtepi32_epi16( _mm512_srai_epi32( _mm512_castps_si512( _mm512_loadu_ps( &LIBXSMM_VLA_ACCESS(2, out_tmp, img2, ofm2, handle->bk) ) ), 16 ) ) );
    }
  }
#endif
}

libxsmm_barrier_wait(handle->barrier, ltid);

