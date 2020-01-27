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
#if defined(LIBXSMM_DNN_FC_BWD_AVX512_CPX)
#define LIBXSMM_DNN_FC_BWD_CONVERT_F32_BF16(in, out, length) do { \
  unsigned int full_chunks = length / 32; \
  unsigned int remainder = length % 32; \
  int __i = 0; \
  if (remainder == 0) { \
    for ( __i = 0; __i < length; __i+= 32) { \
      _mm512_storeu_si512((libxsmm_bfloat16*)out+__i, (__m512i) _mm512_cvtne2ps_pbh(LIBXSMM_INTRINSICS_MM512_LOAD_PS((float*)in+__i+16), LIBXSMM_INTRINSICS_MM512_LOAD_PS((float*)in+__i))); \
    } \
  } else { \
    unsigned int chunk; \
    for ( chunk = 0; chunk < full_chunks; chunk++) { \
      __i = chunk * 32; \
      _mm512_storeu_si512((libxsmm_bfloat16*)out+__i, (__m512i) _mm512_cvtne2ps_pbh(LIBXSMM_INTRINSICS_MM512_LOAD_PS((float*)in+__i+16), LIBXSMM_INTRINSICS_MM512_LOAD_PS((float*)in+__i))); \
    } \
    libxsmm_rne_convert_fp32_bf16((float*)in+32*full_chunks, (element_output_type*)out+32*full_chunks, remainder); \
  } \
} while(0)
#else
#define LIBXSMM_DNN_FC_BWD_CONVERT_F32_BF16(in, out, length) do { \
  unsigned int full_chunks = length / 16; \
  unsigned int remainder = length % 16; \
  int __i = 0; \
  if (remainder == 0) { \
    for ( __i = 0; __i < length; __i+= 16) { \
      _mm256_storeu_si256((__m256i*)(out+__i), _mm512_cvtepi32_epi16( _mm512_srai_epi32( LIBXSMM_INTRINSICS_MM512_ROUNDNE_BF16( LIBXSMM_INTRINSICS_MM512_LOAD_PS((float*)in+__i) ),16)) ); \
    } \
  } else { \
    unsigned int chunk; \
    for ( chunk = 0; chunk < full_chunks; chunk++) { \
      __i = chunk * 16; \
      _mm256_storeu_si256((__m256i*)(out+__i), _mm512_cvtepi32_epi16( _mm512_srai_epi32( LIBXSMM_INTRINSICS_MM512_ROUNDNE_BF16( LIBXSMM_INTRINSICS_MM512_LOAD_PS((float*)in+__i) ),16)) ); \
    } \
    libxsmm_rne_convert_fp32_bf16((float*)in+16*full_chunks, (element_output_type*)out+16*full_chunks, remainder); \
  } \
} while(0)
#endif

/* size variables, all const */
/* here we assume that input and output blocking is similar */
const int bn = handle->bn;
const int bk = handle->bk;
const int bc = handle->bc;
const int nBlocksIFm = handle->desc.C / handle->bc;
const int nBlocksOFm = handle->desc.K / handle->bk;
const int nBlocksMB  = handle->desc.N / handle->bn;
int use_2d_blocking = handle->bwd_2d_blocking;

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
int ofm1 = 0, ifm1 = 0, ifm2 = 0, ifm1ofm1 = 0, mb1ifm1 = 0, mb1 = 0, ofm2 = 0;
int im_tasks_per_thread = 0, in_tasks_per_thread = 0, my_in_start = 0, my_in_end = 0, my_im_start = 0, my_im_end = 0, my_row_id = 0, my_col_id = 0, row_teams = 0, column_teams = 0;

LIBXSMM_VLA_DECL(4, const element_output_type,   doutput, (element_output_type*)handle->grad_output->data, nBlocksOFm, bn, bk);
LIBXSMM_VLA_DECL(5, const element_filter_type, filter, (element_filter_type*)handle->reg_filter->data, nBlocksIFm, bc/2, bk, 2);
LIBXSMM_VLA_DECL(4,        element_input_type,    dinput, (element_input_type* )handle->grad_input->data, nBlocksIFm, bn, bc);
LIBXSMM_VLA_DECL(5,       element_filter_type, filter_tr, (element_filter_type*)handle->scratch, nBlocksOFm, bk/2, bc, 2);
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
  im_tasks_per_thread = (nBlocksMB + row_teams-1)/row_teams;
  in_tasks_per_thread = (nBlocksIFm + column_teams-1)/column_teams;
  my_im_start = LIBXSMM_MIN( my_row_id * im_tasks_per_thread, nBlocksMB);
  my_im_end = LIBXSMM_MIN( (my_row_id+1) * im_tasks_per_thread, nBlocksMB);
  my_in_start = LIBXSMM_MIN( my_col_id * in_tasks_per_thread, nBlocksIFm);
  my_in_end = LIBXSMM_MIN( (my_col_id+1) * in_tasks_per_thread, nBlocksIFm);
}

/* lazy barrier init */
libxsmm_barrier_init(handle->barrier, ltid);
if (handle->desc.K > 1) {
  /* transpose weight */
  if ((bk % 16 == 0) && (bc % 16 == 0)) {
    for (ifm1ofm1 = transpose_thr_begin; ifm1ofm1 < transpose_thr_end; ++ifm1ofm1) {
      ofm1 = ifm1ofm1 / nBlocksIFm;
      ifm1 = ifm1ofm1 % nBlocksIFm;
      bf16_vnni_transpose((element_filter_type*)&LIBXSMM_VLA_ACCESS(5, filter,  ofm1, ifm1, 0, 0, 0, nBlocksIFm, bc/2, bk, 2), (element_filter_type*)&LIBXSMM_VLA_ACCESS(5, filter_tr, ifm1, ofm1, 0, 0, 0, nBlocksOFm, bk/2, bc, 2), bk, bc, bk, bc);
    }
  } else {
    for (ifm1ofm1 = transpose_thr_begin; ifm1ofm1 < transpose_thr_end; ++ifm1ofm1) {
      ofm1 = ifm1ofm1 / nBlocksIFm;
      ifm1 = ifm1ofm1 % nBlocksIFm;
      for (ofm2 = 0; ofm2 < bk; ++ofm2) {
        for (ifm2 = 0; ifm2 < bc; ++ifm2) {
          LIBXSMM_VLA_ACCESS(5, filter_tr, ifm1, ofm1, ofm2/2, ifm2, ofm2%2, nBlocksOFm, bk/2, bc, 2) = LIBXSMM_VLA_ACCESS(5, filter,  ofm1, ifm1, ifm2/2, ofm2, ifm2%2, nBlocksIFm, bc/2, bk, 2);
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
            batchreduce_kernel( &LIBXSMM_VLA_ACCESS(5, filter_tr, ifm1, ofm1*KB_BLOCKS, 0, 0, 0, nBlocksOFm, bk/2, bc, 2),
                &LIBXSMM_VLA_ACCESS(4, doutput,   mb1,  ofm1*KB_BLOCKS, 0, 0, nBlocksOFm, bn, bk),
                &LIBXSMM_VLA_ACCESS(4, dinput_f32,    mb1,  ifm1, 0, 0, nBlocksIFm, bn, bc), &blocks);
            /* downconvert intermediate f32 tensor to bf 16 and store to final C */
            if ( ofm1 == BF-1  ) {
              LIBXSMM_DNN_FC_BWD_CONVERT_F32_BF16(&LIBXSMM_VLA_ACCESS(4, dinput_f32,    mb1,  ifm1, 0, 0, nBlocksIFm, bn, bc), &LIBXSMM_VLA_ACCESS(4, dinput,    mb1,  ifm1, 0, 0, nBlocksIFm, bn, bc), bn*bc);
            }
          }
        }
      }
    } else {
      for (ifm1 = my_in_start; ifm1 < my_in_end; ++ifm1) {
        for (mb1 = my_im_start; mb1 < my_im_end; ++mb1) {
          batchreduce_kernel_zerobeta( &LIBXSMM_VLA_ACCESS(5, filter_tr, ifm1, 0, 0, 0, 0, nBlocksOFm, bk/2, bc, 2),
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
          batchreduce_kernel( &LIBXSMM_VLA_ACCESS(5, filter_tr, ifm1, ofm1*KB_BLOCKS, 0, 0, 0, nBlocksOFm, bk/2, bc, 2),
              &LIBXSMM_VLA_ACCESS(4, doutput,   mb1,  ofm1*KB_BLOCKS, 0, 0, nBlocksOFm, bn, bk),
              &LIBXSMM_VLA_ACCESS(4, dinput_f32,    mb1,  ifm1, 0, 0, nBlocksIFm, bn, bc), &blocks);
          /* downconvert intermediate f32 tensor to bf 16 and store to final C */
          if ( ofm1 == BF-1  ) {
            LIBXSMM_DNN_FC_BWD_CONVERT_F32_BF16(&LIBXSMM_VLA_ACCESS(4, dinput_f32,    mb1,  ifm1, 0, 0, nBlocksIFm, bn, bc), &LIBXSMM_VLA_ACCESS(4, dinput,    mb1,  ifm1, 0, 0, nBlocksIFm, bn, bc), bn*bc);
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
  LIBXSMM_VLA_DECL(5,       element_filter_type, filter_tr_padded, (element_filter_type*)handle->scratch, nBlocksOFm, 1, bc, 2);
  LIBXSMM_VLA_DECL(4,       element_output_type,   doutput_padded, (element_output_type*)handle->scratch + handle->desc.C * 2, nBlocksOFm, bn, 2);
  int mb1ofm1 = 0, mb2 = 0;

  /* Copy in weights and doutput in a padded buffer */
  for (ifm1ofm1 = transpose_thr_begin; ifm1ofm1 < transpose_thr_end; ++ifm1ofm1) {
    ofm1 = ifm1ofm1 / nBlocksIFm;
    ifm1 = ifm1ofm1 % nBlocksIFm;
    ofm2 = 0;
    for (ifm2 = 0; ifm2 < bc; ++ifm2) {
      LIBXSMM_VLA_ACCESS(5, filter_tr_padded, ifm1, ofm1, ofm2/2, ifm2, ofm2%2, nBlocksOFm, 1, bc, 2) = LIBXSMM_VLA_ACCESS(5, filter,  ofm1, ifm1, ifm2/2, ofm2, ifm2%2, nBlocksIFm, bc/2, bk, 2);
      LIBXSMM_VLA_ACCESS(5, filter_tr_padded, ifm1, ofm1, ofm2/2, ifm2, 1, nBlocksOFm, 1, bc, 2) = (element_filter_type)0;
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
    batchreduce_kernel_zerobeta( &LIBXSMM_VLA_ACCESS(5, filter_tr_padded, ifm1, 0, 0, 0, 0, nBlocksOFm, 1, bc, 2),
        &LIBXSMM_VLA_ACCESS(4, doutput_padded,   mb1,  0, 0, 0, nBlocksOFm, bn, 2),
        &LIBXSMM_VLA_ACCESS(4, dinput,    mb1,  ifm1, 0, 0, nBlocksIFm, bn, bc), &blocks);
  }
}

libxsmm_barrier_wait(handle->barrier, ltid);

#undef LIBXSMM_DNN_FC_BWD_CONVERT_F32_BF16

