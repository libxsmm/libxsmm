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
int lpb = 2;
const int bc_lp = handle->bc/lpb;
/* const int bc = handle->bc;*/
int use_2d_blocking = handle->fwd_2d_blocking;

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
int mb1ofm1 = 0, mb1 = 0, ofm1 = 0, ifm1 = 0;
int im_tasks_per_thread = 0, in_tasks_per_thread = 0, my_in_start = 0, my_in_end = 0, my_im_start = 0, my_im_end = 0, my_row_id = 0, my_col_id = 0, row_teams = 0, column_teams = 0;
#ifndef LIBXSMM_DNN_FC_FWD_FUSE_NONE
int mb2 = 0, ofm2 = 0;
#endif
LIBXSMM_VLA_DECL(4, element_output_type,       output,  (element_output_type*)handle->reg_output->data, nBlocksOFm, handle->bn, handle->bk);
LIBXSMM_VLA_DECL(4, const element_input_type,  input,   (element_input_type* )handle->reg_input->data,  nBlocksIFm, handle->bn, handle->bc);
LIBXSMM_VLA_DECL(5, const element_filter_type, filter,  (element_filter_type*)handle->reg_filter->data, nBlocksIFm, bc_lp, handle->bk, lpb);
float* temp_output = (float*)handle->scratch;
LIBXSMM_VLA_DECL(4,        float,    output_f32, (float*) temp_output, nBlocksOFm,handle->bn,handle->bk);
#ifndef LIBXSMM_DNN_FC_FWD_FUSE_NONE
#ifdef LIBXSMM_DNN_FC_FWD_FUSE_BIAS
LIBXSMM_VLA_DECL(2, const element_input_type,               bias,   (element_input_type*)              handle->reg_bias->data,                           handle->bk);
#endif
#ifdef LIBXSMM_DNN_FC_FWD_FUSE_RELU
LIBXSMM_VLA_DECL(4, unsigned char, relumask, (unsigned char*)handle->relumask->data, nBlocksOFm, handle->bn, handle->bk);
LIBXSMM_VLA_DECL(4, __mmask16,  relubitmask,     (__mmask16*)handle->relumask->data, nBlocksOFm, handle->bn, handle->bk/16);
#endif
#endif
unsigned long long  blocks = nBlocksIFm;
int CB_BLOCKS = nBlocksIFm, BF = 1;

BF = handle->fwd_bf;
CB_BLOCKS = nBlocksIFm/BF;
blocks = CB_BLOCKS;

if (use_2d_blocking == 1) {
  row_teams = handle->fwd_row_teams;
  column_teams = handle->fwd_column_teams;
  my_col_id = ltid % column_teams;
  my_row_id = ltid / column_teams;
  im_tasks_per_thread = LIBXSMM_UPDIV(nBlocksMB, row_teams);
  in_tasks_per_thread = LIBXSMM_UPDIV(nBlocksOFm, column_teams);
  my_im_start = LIBXSMM_MIN(my_row_id * im_tasks_per_thread, nBlocksMB);
  my_im_end = LIBXSMM_MIN((my_row_id+1) * im_tasks_per_thread, nBlocksMB);
  my_in_start = LIBXSMM_MIN(my_col_id * in_tasks_per_thread, nBlocksOFm);
  my_in_end = LIBXSMM_MIN((my_col_id+1) * in_tasks_per_thread, nBlocksOFm);
}

/* lazy barrier init */
libxsmm_barrier_init(handle->barrier, ltid);

if (use_2d_blocking == 1) {
  if (BF > 1) {
    for ( ifm1 = 0; ifm1 < BF; ++ifm1 ) {
      for (ofm1 = my_in_start; ofm1 < my_in_end; ++ofm1) {
        for (mb1 = my_im_start; mb1 < my_im_end; ++mb1) {
          /* Initialize intermediate f32 tensor */
          if ( ifm1 == 0 ) {
#ifdef LIBXSMM_DNN_FC_FWD_FUSE_BIAS
            for ( mb2 = 0; mb2 <handle->bn; ++mb2 ) {
              LIBXSMM_DNN_CONVERT_BUFFER_BF16_F32( &LIBXSMM_VLA_ACCESS(2, bias, ofm1, 0,handle->bk), &LIBXSMM_VLA_ACCESS(4, output_f32, mb1, ofm1, mb2, 0, nBlocksOFm,handle->bn,handle->bk), handle->bk );
            }
#else
            memset(&LIBXSMM_VLA_ACCESS(4, output_f32, mb1, ofm1, 0, 0, nBlocksOFm, handle->bn, handle->bk), 0, handle->bn*handle->bk*sizeof(float));
#endif
          }
          batchreduce_kernel( &LIBXSMM_VLA_ACCESS(5, filter, ofm1, ifm1*CB_BLOCKS, 0, 0, 0, nBlocksIFm, bc_lp, handle->bk, lpb),
              &LIBXSMM_VLA_ACCESS(4, input,  mb1, ifm1*CB_BLOCKS, 0, 0, nBlocksIFm, handle->bn, handle->bc),
              &LIBXSMM_VLA_ACCESS(4, output_f32, mb1, ofm1, 0, 0, nBlocksOFm, handle->bn, handle->bk), &blocks);
          /* downconvert intermediate f32 tensor to bf 16 and store to final C */
          if ( ifm1 == BF-1  ) {
#ifndef LIBXSMM_DNN_FC_FWD_FUSE_NONE
            if (handle->bk % 32 == 0) {
              __m512 cur_out_0 = _mm512_setzero_ps();
              __m512 cur_out_1 = _mm512_setzero_ps();
#ifdef LIBXSMM_DNN_FC_FWD_FUSE_RELU
              __mmask16 relumask0;
              __mmask16 relumask1;
#endif
#ifdef LIBXSMM_DNN_FC_FWD_FUSE_SIGMOID
              __m512 ones = _mm512_set1_ps(1.0);
              __m512 halves = _mm512_set1_ps(0.5);
#endif
              for ( mb2 = 0; mb2 < handle->bn; ++mb2 ) {
                for ( ofm2 = 0; ofm2 < handle->bk; ofm2 += 32 ) {
                  cur_out_0 = LIBXSMM_INTRINSICS_MM512_LOAD_PS(&LIBXSMM_VLA_ACCESS(4,  output_f32, mb1, ofm1, mb2, ofm2, nBlocksOFm, handle->bn, handle->bk));
                  cur_out_1 = LIBXSMM_INTRINSICS_MM512_LOAD_PS(&LIBXSMM_VLA_ACCESS(4,  output_f32, mb1, ofm1, mb2, ofm2+16, nBlocksOFm, handle->bn, handle->bk));
#ifdef LIBXSMM_DNN_FC_FWD_FUSE_RELU
                  relumask0 = _mm512_cmp_ps_mask( cur_out_0, _mm512_setzero_ps(), _CMP_GT_OQ );
                  relumask1 = _mm512_cmp_ps_mask( cur_out_1, _mm512_setzero_ps(), _CMP_GT_OQ );
                  cur_out_0 = _mm512_mask_blend_ps( relumask0, _mm512_setzero_ps(), cur_out_0 );
                  cur_out_1 = _mm512_mask_blend_ps( relumask1, _mm512_setzero_ps(), cur_out_1 );
                  LIBXSMM_INTRINSICS_MM512_STORE_MASK16( &LIBXSMM_VLA_ACCESS(4, relubitmask, mb1, ofm1, mb2, ofm2/16, nBlocksOFm, handle->bn, handle->bk/16), relumask0 );
                  LIBXSMM_INTRINSICS_MM512_STORE_MASK16( &LIBXSMM_VLA_ACCESS(4, relubitmask, mb1, ofm1, mb2, ofm2/16+1, nBlocksOFm, handle->bn, handle->bk/16), relumask1 );
#endif
#ifdef LIBXSMM_DNN_FC_FWD_FUSE_SIGMOID
                  /* we ar using Pade 7/8 approximation */
                  cur_out_0 = _mm512_mul_ps(_mm512_add_ps(LIBXSMM_INTRINSICS_MM512_TANH_PS_MINIMAX2(_mm512_mul_ps(cur_out_0, halves)), ones), halves);
                  cur_out_1 = _mm512_mul_ps(_mm512_add_ps(LIBXSMM_INTRINSICS_MM512_TANH_PS_MINIMAX2(_mm512_mul_ps(cur_out_1, halves)), ones), halves);
#endif
                  _mm512_storeu_ps(&LIBXSMM_VLA_ACCESS(4,  output_f32, mb1, ofm1, mb2, ofm2, nBlocksOFm, handle->bn, handle->bk), cur_out_0);
                  _mm512_storeu_ps(&LIBXSMM_VLA_ACCESS(4,  output_f32, mb1, ofm1, mb2, ofm2+16, nBlocksOFm, handle->bn, handle->bk), cur_out_1);
                }
              }
            } else {
              for ( mb2 = 0; mb2 < handle->bn; ++mb2 ) {
                for ( ofm2 = 0; ofm2 < handle->bk; ++ofm2 ) {
                  float l_cur_out = LIBXSMM_VLA_ACCESS(4, output_f32, mb1, ofm1, mb2, ofm2, nBlocksOFm, handle->bn, handle->bk);
#ifdef LIBXSMM_DNN_FC_FWD_FUSE_RELU
                  LIBXSMM_VLA_ACCESS(4, relumask, mb1, ofm1, mb2, ofm2, nBlocksOFm, handle->bn, handle->bk) = (unsigned char)(( l_cur_out > (float)0 ) ? 1 : 0);
                  l_cur_out = (l_cur_out > (float)0) ? l_cur_out : (float)0;
#endif
#ifdef LIBXSMM_DNN_FC_FWD_FUSE_SIGMOID
                  /* we ar using Pade 7/8 approximation */
                  l_cur_out = (libxsmm_stanh_pade78( l_cur_out / 2.0f ) + 1.0f) / 2.0f;
#endif
                  LIBXSMM_VLA_ACCESS(4, output_f32, mb1, ofm1, mb2, ofm2, nBlocksOFm, handle->bn, handle->bk) = l_cur_out;
                }
              }
            }
#endif
            LIBXSMM_DNN_CONVERT_BUFFER_F32_BF16(&LIBXSMM_VLA_ACCESS(4, output_f32,    mb1,  ofm1, 0, 0, nBlocksOFm,handle->bn,handle->bk), &LIBXSMM_VLA_ACCESS(4, output,    mb1,  ofm1, 0, 0, nBlocksOFm,handle->bn,handle->bk),handle->bn*handle->bk);
          }
        }
      }
    }
  } else {
    for (ofm1 = my_in_start; ofm1 < my_in_end; ++ofm1) {
      for (mb1 = my_im_start; mb1 < my_im_end; ++mb1) {
#ifdef LIBXSMM_DNN_FC_FWD_FUSE_BIAS
        for ( mb2 = 0; mb2 < handle->bn; ++mb2 ) {
          for ( ofm2 = 0; ofm2 < handle->bk; ++ofm2 ) {
            LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, mb2, ofm2, nBlocksOFm, handle->bn, handle->bk) = LIBXSMM_VLA_ACCESS(2, bias, ofm1, ofm2, handle->bk);
          }
        }
        batchreduce_kernel_beta( &LIBXSMM_VLA_ACCESS(5, filter, ofm1, 0, 0, 0, 0, nBlocksIFm, bc_lp, handle->bk, lpb),
            &LIBXSMM_VLA_ACCESS(4, input,  mb1, 0,  0, 0, nBlocksIFm, handle->bn, handle->bc),
            &LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, 0, 0, nBlocksOFm, handle->bn, handle->bk), &blocks);
#else
        batchreduce_kernel_zerobeta( &LIBXSMM_VLA_ACCESS(5, filter, ofm1, 0, 0, 0, 0, nBlocksIFm, bc_lp, handle->bk, lpb),
            &LIBXSMM_VLA_ACCESS(4, input,  mb1, 0,  0, 0, nBlocksIFm, handle->bn, handle->bc),
            &LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, 0, 0, nBlocksOFm, handle->bn, handle->bk), &blocks);
#endif
#ifndef LIBXSMM_DNN_FC_FWD_FUSE_NONE
        if (handle->bk % 32 == 0) {
          __m512 cur_out_0 = _mm512_setzero_ps();
          __m512 cur_out_1 = _mm512_setzero_ps();
#ifdef LIBXSMM_DNN_FC_FWD_FUSE_RELU
          __mmask16 relumask0;
          __mmask16 relumask1;
#endif
#ifdef LIBXSMM_DNN_FC_FWD_FUSE_SIGMOID
          __m512 ones = _mm512_set1_ps(1.0);
          __m512 halves = _mm512_set1_ps(0.5);
#endif
          for ( mb2 = 0; mb2 < handle->bn; ++mb2 ) {
            for ( ofm2 = 0; ofm2 < handle->bk; ofm2 += 32 ) {
              cur_out_0 = LIBXSMM_INTRINSICS_MM512_CVTPBH_PS(_mm256_loadu_si256((__m256i*)&LIBXSMM_VLA_ACCESS(4,  output, mb1, ofm1, mb2, ofm2, nBlocksOFm, handle->bn, handle->bk)));
              cur_out_1 = LIBXSMM_INTRINSICS_MM512_CVTPBH_PS(_mm256_loadu_si256((__m256i*)&LIBXSMM_VLA_ACCESS(4,  output, mb1, ofm1, mb2, ofm2+16, nBlocksOFm, handle->bn, handle->bk)));
#ifdef LIBXSMM_DNN_FC_FWD_FUSE_RELU
              relumask0 = _mm512_cmp_ps_mask( cur_out_0, _mm512_setzero_ps(), _CMP_GT_OQ );
              relumask1 = _mm512_cmp_ps_mask( cur_out_1, _mm512_setzero_ps(), _CMP_GT_OQ );
              cur_out_0 = _mm512_mask_blend_ps( relumask0, _mm512_setzero_ps(), cur_out_0 );
              cur_out_1 = _mm512_mask_blend_ps( relumask1, _mm512_setzero_ps(), cur_out_1 );
              LIBXSMM_INTRINSICS_MM512_STORE_MASK16( &LIBXSMM_VLA_ACCESS(4, relubitmask, mb1, ofm1, mb2, ofm2/16, nBlocksOFm, handle->bn, handle->bk/16), relumask0 );
              LIBXSMM_INTRINSICS_MM512_STORE_MASK16( &LIBXSMM_VLA_ACCESS(4, relubitmask, mb1, ofm1, mb2, ofm2/16+1, nBlocksOFm, handle->bn, handle->bk/16), relumask1 );
#endif
#ifdef LIBXSMM_DNN_FC_FWD_FUSE_SIGMOID
              /* we ar using Pade 7/8 approximation */
              cur_out_0 = _mm512_mul_ps(_mm512_add_ps(LIBXSMM_INTRINSICS_MM512_TANH_PS_MINIMAX2(_mm512_mul_ps(cur_out_0, halves)), ones), halves);
              cur_out_1 = _mm512_mul_ps(_mm512_add_ps(LIBXSMM_INTRINSICS_MM512_TANH_PS_MINIMAX2(_mm512_mul_ps(cur_out_1, halves)), ones), halves);
#endif
              _mm512_storeu_si512(&LIBXSMM_VLA_ACCESS(4,  output, mb1, ofm1, mb2, ofm2, nBlocksOFm, handle->bn, handle->bk), LIBXSMM_INTRINSISCS_MM512_CVTNE2PS_PBH( cur_out_1, cur_out_0 ));
            }
          }
        } else {
          for ( mb2 = 0; mb2 < handle->bn; ++mb2 ) {
            for ( ofm2 = 0; ofm2 < handle->bk; ++ofm2 ) {
#ifdef LIBXSMM_DNN_FC_FWD_FUSE_SIGMOID
              libxsmm_bfloat16_hp t;
#endif
              libxsmm_bfloat16 l_cur_out = LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, mb2, ofm2, nBlocksOFm, handle->bn, handle->bk);
#ifdef LIBXSMM_DNN_FC_FWD_FUSE_RELU
              LIBXSMM_VLA_ACCESS(4, relumask, mb1, ofm1, mb2, ofm2, nBlocksOFm, handle->bn, handle->bk) = (unsigned char)(( (l_cur_out & 0x8000) > 0 ) ? 0 : 1);
              l_cur_out = (libxsmm_bfloat16)(( (l_cur_out & 0x8000) > 0 ) ? 0 : l_cur_out);
#endif
#ifdef LIBXSMM_DNN_FC_FWD_FUSE_SIGMOID
              /* we ar using Pade 7/8 approximation */
              t.i[1] = l_cur_out;
              t.i[0] = 0;
              t.f = (libxsmm_stanh_pade78( t.f / 2.0f ) + 1.0f) / 2.0f;
              l_cur_out = t.i[1];
#endif
              LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, mb2, ofm2, nBlocksOFm, handle->bn, handle->bk) = l_cur_out;
            }
          }
        }
#endif
      }
    }
  }
} else {
  if (BF > 1) {
    for ( ifm1 = 0; ifm1 < BF; ++ifm1 ) {
      for ( mb1ofm1 = thr_begin; mb1ofm1 < thr_end; ++mb1ofm1 ) {
        mb1  = mb1ofm1%nBlocksMB;
        ofm1 = mb1ofm1/nBlocksMB;
        /* Initialize intermediate f32 tensor */
        if ( ifm1 == 0 ) {
#ifdef LIBXSMM_DNN_FC_FWD_FUSE_BIAS
          for ( mb2 = 0; mb2 <handle->bn; ++mb2 ) {
            LIBXSMM_DNN_CONVERT_BUFFER_BF16_F32( &LIBXSMM_VLA_ACCESS(2, bias, ofm1, 0,handle->bk), &LIBXSMM_VLA_ACCESS(4, output_f32, mb1, ofm1, mb2, 0, nBlocksOFm, handle->bn, handle->bk), handle->bk );
          }
#else
          memset(&LIBXSMM_VLA_ACCESS(4, output_f32, mb1, ofm1, 0, 0, nBlocksOFm, handle->bn, handle->bk), 0, handle->bn*handle->bk*sizeof(float));
#endif
        }
        batchreduce_kernel( &LIBXSMM_VLA_ACCESS(5, filter, ofm1, ifm1*CB_BLOCKS, 0, 0, 0, nBlocksIFm, bc_lp, handle->bk, lpb),
            &LIBXSMM_VLA_ACCESS(4, input,  mb1, ifm1*CB_BLOCKS, 0, 0, nBlocksIFm, handle->bn, handle->bc),
            &LIBXSMM_VLA_ACCESS(4, output_f32, mb1, ofm1, 0, 0, nBlocksOFm, handle->bn, handle->bk), &blocks);
        /* downconvert intermediate f32 tensor to bf 16 and store to final C */
        if ( ifm1 == BF-1  ) {
#ifndef LIBXSMM_DNN_FC_FWD_FUSE_NONE
          if (handle->bk % 32 == 0) {
            __m512 cur_out_0 = _mm512_setzero_ps();
            __m512 cur_out_1 = _mm512_setzero_ps();
#ifdef LIBXSMM_DNN_FC_FWD_FUSE_RELU
            __mmask16 relumask0;
            __mmask16 relumask1;
#endif
#ifdef LIBXSMM_DNN_FC_FWD_FUSE_SIGMOID
            __m512 ones = _mm512_set1_ps(1.0);
            __m512 halves = _mm512_set1_ps(0.5);
#endif
            for ( mb2 = 0; mb2 < handle->bn; ++mb2 ) {
              for ( ofm2 = 0; ofm2 < handle->bk; ofm2 += 32 ) {
                cur_out_0 = LIBXSMM_INTRINSICS_MM512_LOAD_PS(&LIBXSMM_VLA_ACCESS(4,  output_f32, mb1, ofm1, mb2, ofm2, nBlocksOFm, handle->bn, handle->bk));
                cur_out_1 = LIBXSMM_INTRINSICS_MM512_LOAD_PS(&LIBXSMM_VLA_ACCESS(4,  output_f32, mb1, ofm1, mb2, ofm2+16, nBlocksOFm, handle->bn, handle->bk));
#ifdef LIBXSMM_DNN_FC_FWD_FUSE_RELU
                relumask0 = _mm512_cmp_ps_mask( cur_out_0, _mm512_setzero_ps(), _CMP_GT_OQ );
                relumask1 = _mm512_cmp_ps_mask( cur_out_1, _mm512_setzero_ps(), _CMP_GT_OQ );
                cur_out_0 = _mm512_mask_blend_ps( relumask0, _mm512_setzero_ps(), cur_out_0 );
                cur_out_1 = _mm512_mask_blend_ps( relumask1, _mm512_setzero_ps(), cur_out_1 );
                LIBXSMM_INTRINSICS_MM512_STORE_MASK16( &LIBXSMM_VLA_ACCESS(4, relubitmask, mb1, ofm1, mb2, ofm2/16, nBlocksOFm, handle->bn, handle->bk/16), relumask0 );
                LIBXSMM_INTRINSICS_MM512_STORE_MASK16( &LIBXSMM_VLA_ACCESS(4, relubitmask, mb1, ofm1, mb2, ofm2/16+1, nBlocksOFm, handle->bn, handle->bk/16), relumask1 );
#endif
#ifdef LIBXSMM_DNN_FC_FWD_FUSE_SIGMOID
                /* we ar using Pade 7/8 approximation */
                cur_out_0 = _mm512_mul_ps(_mm512_add_ps(LIBXSMM_INTRINSICS_MM512_TANH_PS_MINIMAX2(_mm512_mul_ps(cur_out_0, halves)), ones), halves);
                cur_out_1 = _mm512_mul_ps(_mm512_add_ps(LIBXSMM_INTRINSICS_MM512_TANH_PS_MINIMAX2(_mm512_mul_ps(cur_out_1, halves)), ones), halves);
#endif
                _mm512_storeu_ps(&LIBXSMM_VLA_ACCESS(4,  output_f32, mb1, ofm1, mb2, ofm2, nBlocksOFm, handle->bn, handle->bk), cur_out_0);
                _mm512_storeu_ps(&LIBXSMM_VLA_ACCESS(4,  output_f32, mb1, ofm1, mb2, ofm2+16, nBlocksOFm, handle->bn, handle->bk), cur_out_1);
              }
            }
          } else {
            for ( mb2 = 0; mb2 < handle->bn; ++mb2 ) {
              for ( ofm2 = 0; ofm2 < handle->bk; ++ofm2 ) {
                float l_cur_out = LIBXSMM_VLA_ACCESS(4, output_f32, mb1, ofm1, mb2, ofm2, nBlocksOFm, handle->bn, handle->bk);
#ifdef LIBXSMM_DNN_FC_FWD_FUSE_RELU
                LIBXSMM_VLA_ACCESS(4, relumask, mb1, ofm1, mb2, ofm2, nBlocksOFm, handle->bn, handle->bk) = (unsigned char)(( l_cur_out > 0.0 ) ? 1 : 0);
                l_cur_out = (l_cur_out > (float)0) ? l_cur_out : (float)0;
#endif
#ifdef LIBXSMM_DNN_FC_FWD_FUSE_SIGMOID
                /* we ar using Pade 7/8 approximation */
                l_cur_out = (libxsmm_stanh_pade78( l_cur_out / 2.0f ) + 1.0f) / 2.0f;
#endif
                LIBXSMM_VLA_ACCESS(4, output_f32, mb1, ofm1, mb2, ofm2, nBlocksOFm, handle->bn, handle->bk) = l_cur_out;
              }
            }
          }
#endif
          LIBXSMM_DNN_CONVERT_BUFFER_F32_BF16(&LIBXSMM_VLA_ACCESS(4, output_f32,    mb1,  ofm1, 0, 0, nBlocksOFm, handle->bn, handle->bk), &LIBXSMM_VLA_ACCESS(4, output,    mb1,  ofm1, 0, 0, nBlocksOFm, handle->bn, handle->bk), handle->bn*handle->bk);
        }
      }
    }
  } else {
    for ( mb1ofm1 = thr_begin; mb1ofm1 < thr_end; ++mb1ofm1 ) {
      mb1  = mb1ofm1%nBlocksMB;
      ofm1 = mb1ofm1/nBlocksMB;
#ifdef LIBXSMM_DNN_FC_FWD_FUSE_BIAS
      for ( mb2 = 0; mb2 <handle->bn; ++mb2 ) {
        for ( ofm2 = 0; ofm2 <handle->bk; ++ofm2 ) {
          LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, mb2, ofm2, nBlocksOFm, handle->bn, handle->bk) = LIBXSMM_VLA_ACCESS(2, bias, ofm1, ofm2, handle->bk);
        }
      }
      batchreduce_kernel_beta( &LIBXSMM_VLA_ACCESS(5, filter, ofm1, 0, 0, 0, 0, nBlocksIFm, bc_lp, handle->bk, lpb),
          &LIBXSMM_VLA_ACCESS(4, input,  mb1, 0,  0, 0, nBlocksIFm, handle->bn, handle->bc),
          &LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, 0, 0, nBlocksOFm, handle->bn, handle->bk), &blocks);
#else
      batchreduce_kernel_zerobeta( &LIBXSMM_VLA_ACCESS(5, filter, ofm1, 0, 0, 0, 0, nBlocksIFm, bc_lp, handle->bk, lpb),
          &LIBXSMM_VLA_ACCESS(4, input,  mb1, 0,  0, 0, nBlocksIFm, handle->bn, handle->bc),
          &LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, 0, 0, nBlocksOFm, handle->bn, handle->bk), &blocks);
#endif
#ifndef LIBXSMM_DNN_FC_FWD_FUSE_NONE
      if (handle->bk % 32 == 0) {
        __m512 cur_out_0 = _mm512_setzero_ps();
        __m512 cur_out_1 = _mm512_setzero_ps();
#ifdef LIBXSMM_DNN_FC_FWD_FUSE_RELU
        __mmask16 relumask0;
        __mmask16 relumask1;
#endif
#ifdef LIBXSMM_DNN_FC_FWD_FUSE_SIGMOID
        __m512 ones = _mm512_set1_ps(1.0);
        __m512 halves = _mm512_set1_ps(0.5);
#endif
        for ( mb2 = 0; mb2 < handle->bn; ++mb2 ) {
          for ( ofm2 = 0; ofm2 < handle->bk; ofm2 += 32 ) {
            cur_out_0 = LIBXSMM_INTRINSICS_MM512_CVTPBH_PS(_mm256_loadu_si256((__m256i*)&LIBXSMM_VLA_ACCESS(4,  output, mb1, ofm1, mb2, ofm2, nBlocksOFm, handle->bn, handle->bk)));
            cur_out_1 = LIBXSMM_INTRINSICS_MM512_CVTPBH_PS(_mm256_loadu_si256((__m256i*)&LIBXSMM_VLA_ACCESS(4,  output, mb1, ofm1, mb2, ofm2+16, nBlocksOFm, handle->bn, handle->bk)));
#ifdef LIBXSMM_DNN_FC_FWD_FUSE_RELU
            relumask0 = _mm512_cmp_ps_mask( cur_out_0, _mm512_setzero_ps(), _CMP_GT_OQ );
            relumask1 = _mm512_cmp_ps_mask( cur_out_1, _mm512_setzero_ps(), _CMP_GT_OQ );
            cur_out_0 = _mm512_mask_blend_ps( relumask0, _mm512_setzero_ps(), cur_out_0 );
            cur_out_1 = _mm512_mask_blend_ps( relumask1, _mm512_setzero_ps(), cur_out_1 );
            LIBXSMM_INTRINSICS_MM512_STORE_MASK16( &LIBXSMM_VLA_ACCESS(4, relubitmask, mb1, ofm1, mb2, ofm2/16, nBlocksOFm, handle->bn, handle->bk/16), relumask0 );
            LIBXSMM_INTRINSICS_MM512_STORE_MASK16( &LIBXSMM_VLA_ACCESS(4, relubitmask, mb1, ofm1, mb2, ofm2/16+1, nBlocksOFm, handle->bn, handle->bk/16), relumask1 );
#endif
#ifdef LIBXSMM_DNN_FC_FWD_FUSE_SIGMOID
            /* we ar using Pade 7/8 approximation */
            cur_out_0 = _mm512_mul_ps(_mm512_add_ps(LIBXSMM_INTRINSICS_MM512_TANH_PS_MINIMAX2(_mm512_mul_ps(cur_out_0, halves)), ones), halves);
            cur_out_1 = _mm512_mul_ps(_mm512_add_ps(LIBXSMM_INTRINSICS_MM512_TANH_PS_MINIMAX2(_mm512_mul_ps(cur_out_1, halves)), ones), halves);
#endif
            _mm512_storeu_si512(&LIBXSMM_VLA_ACCESS(4,  output, mb1, ofm1, mb2, ofm2, nBlocksOFm, handle->bn, handle->bk), LIBXSMM_INTRINSISCS_MM512_CVTNE2PS_PBH( cur_out_1, cur_out_0 ));
          }
        }
      } else {
        for ( mb2 = 0; mb2 < handle->bn; ++mb2 ) {
          for ( ofm2 = 0; ofm2 < handle->bk; ++ofm2 ) {
#ifdef LIBXSMM_DNN_FC_FWD_FUSE_SIGMOID
            libxsmm_bfloat16_hp t;
#endif
            libxsmm_bfloat16 l_cur_out = LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, mb2, ofm2, nBlocksOFm, handle->bn, handle->bk);
#ifdef LIBXSMM_DNN_FC_FWD_FUSE_RELU
            LIBXSMM_VLA_ACCESS(4, relumask, mb1, ofm1, mb2, ofm2, nBlocksOFm, handle->bn, handle->bk) = (unsigned char)(( (l_cur_out & 0x8000) > 0 ) ? 0 : 1);
            l_cur_out = (libxsmm_bfloat16)(( (l_cur_out & 0x8000) > 0 ) ? 0 : l_cur_out);
#endif
#ifdef LIBXSMM_DNN_FC_FWD_FUSE_SIGMOID
            /* we ar using Pade 7/8 approximation */
            t.i[1] = l_cur_out;
            t.i[0] = 0;
            t.f = (libxsmm_stanh_pade78( t.f / 2.0f ) + 1.0f) / 2.0f;
            l_cur_out = t.i[1];
#endif
            LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, mb2, ofm2, nBlocksOFm, handle->bn, handle->bk) = l_cur_out;
          }
        }
      }

#endif
    }
  }
}

libxsmm_barrier_wait(handle->barrier, ltid);

