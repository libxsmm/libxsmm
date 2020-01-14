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
const int bn = handle->bn;
const int bk = handle->bk;
const int bc = handle->bc;
const int nBlocksIFm = handle->desc.C / handle->bc;
const int nBlocksOFm = handle->desc.K / handle->bk;
const int nBlocksMB  = handle->desc.N / handle->bn;

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
int ofm1 = 0, ofm2 = 0, ifm1 = 0, ifm2 = 0, ifm1ofm1 = 0, mb1ifm1 = 0, mb1 = 0, img2 = 0;

LIBXSMM_VLA_DECL(4, const element_output_type,   doutput, (element_output_type*)handle->grad_output->data, nBlocksOFm, bn, bk);
LIBXSMM_VLA_DECL(5, const element_filter_type, filter, (element_filter_type*)handle->reg_filter->data, nBlocksIFm, bc/2, bk, 2);
LIBXSMM_VLA_DECL(4,        element_input_type,    dinput, (element_input_type* )handle->grad_input->data, nBlocksIFm, bn, bc);
LIBXSMM_VLA_DECL(5,       element_filter_type, filter_tr, (element_filter_type*)handle->scratch, nBlocksOFm, bk/2, bc, 2);
float* temp_output = (float*)handle->scratch + (handle->desc.C * handle->desc.K)/2;
LIBXSMM_VLA_DECL(4,        float,    dinput_f32, (float*) temp_output, nBlocksIFm, bn, bc);

unsigned long long  blocks = nBlocksOFm;
int KB_BLOCKS = nBlocksOFm, BF = 1, iteri = 0, iterj = 0;

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
blocks = KB_BLOCKS;

/* lazy barrier init */
libxsmm_barrier_init(handle->barrier, ltid);

/* transpose weight */
for (ifm1ofm1 = transpose_thr_begin; ifm1ofm1 < transpose_thr_end; ++ifm1ofm1) {
  ofm1 = ifm1ofm1 / nBlocksIFm;
  ifm1 = ifm1ofm1 % nBlocksIFm;
  for (ofm2 = 0; ofm2 < bk; ++ofm2) {
    for (ifm2 = 0; ifm2 < bc; ++ifm2) {
      LIBXSMM_VLA_ACCESS(5, filter_tr, ifm1, ofm1, ofm2/2, ifm2, ofm2%2, nBlocksOFm, bk/2, bc, 2) =
        LIBXSMM_VLA_ACCESS(5, filter,  ofm1, ifm1, ifm2/2, ofm2, ifm2%2, nBlocksIFm, bc/2, bk, 2);
    }
  }
}

/* wait for transpose to finish */
libxsmm_barrier_wait(handle->barrier, ltid);

if (BF > 1) {
  for ( ofm1 = 0; ofm1 < BF; ++ofm1 ) {
    for ( mb1ifm1 = thr_begin; mb1ifm1 < thr_end; ++mb1ifm1 ) {
      mb1  = mb1ifm1%nBlocksMB;
      ifm1 = mb1ifm1/nBlocksMB;
      /* Initialize intermediate f32 tensor */
      if ( 0 == ofm1 ) {
        for ( iteri = 0; iteri < bn; ++iteri ) {
          for ( iterj = 0; iterj < bc; ++iterj ) {
            LIBXSMM_VLA_ACCESS(4, dinput_f32, mb1, ifm1, iteri, iterj, nBlocksIFm, bn, bc) = 0;
          }
        }
      }
      batchreduce_kernel( &LIBXSMM_VLA_ACCESS(5, filter_tr, ifm1, ofm1*KB_BLOCKS, 0, 0, 0, nBlocksOFm, bk/2, bc, 2),
          &LIBXSMM_VLA_ACCESS(4, doutput,   mb1,  ofm1*KB_BLOCKS, 0, 0, nBlocksOFm, bn, bk),
          &LIBXSMM_VLA_ACCESS(4, dinput_f32,    mb1,  ifm1, 0, 0, nBlocksIFm, bn, bc), &blocks);
      /* downconvert intermediate f32 tensor to bf 16 and store to final C */
      if ( ofm1 == BF-1  ) {
        for ( img2 = 0; img2 < handle->bn; ++img2 ) {
          for ( ofm2 = 0; ofm2 < handle->bc; ofm2 += 16 ) {
            _mm256_storeu_si256( (__m256i *) &LIBXSMM_VLA_ACCESS(4, dinput,    mb1,  ifm1, img2, ofm2, nBlocksIFm, bn, bc),
                _mm512_cvtepi32_epi16( _mm512_srai_epi32( _mm512_castps_si512( _mm512_loadu_ps( &LIBXSMM_VLA_ACCESS(4, dinput_f32,    mb1,  ifm1, img2, ofm2, nBlocksIFm, bn, bc) ) ), 16 ) ) );
          }
        }
      }
    }
  }
} else {
  for ( mb1ifm1 = thr_begin; mb1ifm1 < thr_end; ++mb1ifm1 ) {
    mb1  = mb1ifm1%nBlocksMB;
    ifm1 = mb1ifm1/nBlocksMB;
    batchreduce_kernel_zerobeta( &LIBXSMM_VLA_ACCESS(5, filter_tr, ifm1, 0, 0, 0, 0, nBlocksOFm, bk/2, bc, 2),
        &LIBXSMM_VLA_ACCESS(4, doutput,   mb1,  0, 0, 0, nBlocksOFm, bn, bk),
        &LIBXSMM_VLA_ACCESS(4, dinput,    mb1,  ifm1, 0, 0, nBlocksIFm, bn, bc), &blocks);
  }
}

libxsmm_barrier_wait(handle->barrier, ltid);

