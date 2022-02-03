/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Evangelos Georganas, Alexander Heinecke (Intel Corp.)
******************************************************************************/
/* size variables, all const */
/* here we assume that input and output blocking is similar */
const int nBlocksIFm = handle->desc.C / handle->bc;
const int nBlocksOFm = handle->desc.K / handle->bk;
const int nBlocksMB  = handle->desc.N / handle->bn;
const int bn = handle->bn;
const int bk = handle->bk;
const int lpb = 2;
const int bc_lp = handle->bc/lpb;
/* const int bc = handle->bc;*/
int use_2d_blocking = handle->fwd_2d_blocking;

/* computing first logical thread */
const int ltid = tid - start_thread;

/* loop variables */
int mb1 = 0, ofm1 = 0, ifm1 = 0;
int im_tasks_per_thread = 0, in_tasks_per_thread = 0, my_in_start = 0, my_in_end = 0, my_im_start = 0, my_im_end = 0, my_row_id = 0, my_col_id = 0, row_teams = 0, column_teams = 0;
LIBXSMM_VLA_DECL(4, element_output_type,       output,  (element_output_type*)handle->reg_output->data, nBlocksOFm, handle->bn, handle->bk);
LIBXSMM_VLA_DECL(4, const element_input_type,  input,   (element_input_type* )handle->reg_input->data,  nBlocksIFm, handle->bn, handle->bc);

LIBXSMM_VLA_DECL(5, const element_filter_type, filter_compressed,  (element_filter_type*)handle->reg_filter->data, nBlocksIFm, bc_lp, handle->bk/handle->sparsity_factor_A, lpb);
LIBXSMM_VLA_DECL(5, __mmask32, idx_filter_compressed, (__mmask32*) ((element_filter_type*)handle->reg_filter->data + (handle->desc.C*handle->desc.K)/handle->sparsity_factor_A), nBlocksIFm, bc_lp, handle->bk/32, lpb);
LIBXSMM_VLA_DECL(4, element_filter_type, decompressed_filter, (element_filter_type*)handle->scratch + ltid * handle->bk * handle->desc.C, bc_lp, handle->bk, lpb);

float* temp_output = (float*)handle->scratch + (handle->desc.threads * handle->desc.C * handle->bk)/2;
LIBXSMM_VLA_DECL(4, float, output_f32, (float*) temp_output, nBlocksOFm, bn, bk);
libxsmm_meltw_gemm_param gemm_eltwise_params;

#ifndef LIBXSMM_DNN_FC_FWD_FUSE_NONE
#if defined(LIBXSMM_DNN_FC_FWD_FUSE_BIAS)
int mb2 = 0;
float* fp32_bias_scratch = (float*)handle->scratch + (handle->desc.threads * handle->desc.C * handle->bk)/2 +  ltid * handle->desc.K;
LIBXSMM_VLA_DECL(2, const element_input_type, bias, (element_input_type*) handle->reg_bias->data, handle->bk);
#endif
#if defined(LIBXSMM_DNN_FC_FWD_FUSE_RELU)
LIBXSMM_VLA_DECL(4, __mmask32,  relubitmask,     (__mmask32*)handle->relumask->data, nBlocksOFm, handle->bn, handle->bk/32);
libxsmm_meltwfunction_unary eltwise_kernel = handle->fwd_cvtfp32bf16_relu_kernel;
libxsmm_meltw_unary_param   eltwise_params;
#elif defined(LIBXSMM_DNN_FC_FWD_FUSE_SIGMOID)
libxsmm_meltwfunction_unary eltwise_kernel = handle->fwd_sigmoid_cvtfp32bf16_kernel;
libxsmm_meltw_unary_param   eltwise_params;
#else
libxsmm_meltwfunction_unary eltwise_kernel = handle->fwd_cvtfp32bf16_kernel;
libxsmm_meltw_unary_param   eltwise_params;
#endif
#else
libxsmm_meltwfunction_unary eltwise_kernel = handle->fwd_cvtfp32bf16_kernel;
libxsmm_meltw_unary_param   eltwise_params;
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
  im_tasks_per_thread = (nBlocksMB + row_teams-1)/row_teams;
  in_tasks_per_thread = (nBlocksOFm + column_teams-1)/column_teams;
  my_im_start = LIBXSMM_MIN( my_row_id * im_tasks_per_thread, nBlocksMB);
  my_im_end = LIBXSMM_MIN( (my_row_id+1) * im_tasks_per_thread, nBlocksMB);
  my_in_start = LIBXSMM_MIN( my_col_id * in_tasks_per_thread, nBlocksOFm);
  my_in_end = LIBXSMM_MIN( (my_col_id+1) * in_tasks_per_thread, nBlocksOFm);
}

/* lazy barrier init */
libxsmm_barrier_init(handle->barrier, ltid);

tile_config_kernel(NULL, NULL, NULL);

if (handle->sparsity_factor_A == 1) {
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

#ifdef WR_PREFETCH_OUTPUT
          prefetchwt_chunk((char*)&LIBXSMM_VLA_ACCESS(4, output_f32, mb1, ofm1, 0, 0, nBlocksOFm, handle->bn, handle->bk), handle->bn*handle->bk*sizeof(float));
          if ( ifm1 == BF-1  ) {
            prefetchwt_chunk((char*)&LIBXSMM_VLA_ACCESS(4,  output, mb1, ofm1, 0, 0, nBlocksOFm, handle->bn, handle->bk), handle->bn*handle->bk*sizeof(libxsmm_bfloat16));
          }
#endif

          batchreduce_kernel( &LIBXSMM_VLA_ACCESS(5, filter_compressed, ofm1, ifm1*CB_BLOCKS, 0, 0, 0, nBlocksIFm, bc_lp, handle->bk/handle->sparsity_factor_A, lpb),
              &LIBXSMM_VLA_ACCESS(4, input,  mb1, ifm1*CB_BLOCKS, 0, 0, nBlocksIFm, handle->bn, handle->bc),
              &LIBXSMM_VLA_ACCESS(4, output_f32, mb1, ofm1, 0, 0, nBlocksOFm, handle->bn, handle->bk), &blocks);

          /* downconvert intermediate f32 tensor to bf 16 and store to final C */
          if ( ifm1 == BF-1  ) {
            eltwise_params.in.primary = &LIBXSMM_VLA_ACCESS(4, output_f32, mb1, ofm1, 0, 0, nBlocksOFm, handle->bn, handle->bk);
            eltwise_params.out.primary = &LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, 0, 0, nBlocksOFm, handle->bn, handle->bk);
#if defined(LIBXSMM_DNN_FC_FWD_FUSE_RELU)
            eltwise_params.out.secondary = &LIBXSMM_VLA_ACCESS(4, relubitmask, mb1, ofm1, 0, 0, nBlocksOFm, handle->bn, handle->bk/32);
#endif
            eltwise_kernel(&eltwise_params);
          }
        }
      }
    }
  } else {
#ifdef LIBXSMM_DNN_FC_FWD_FUSE_BIAS
    LIBXSMM_DNN_CONVERT_BUFFER_BF16_F32( &LIBXSMM_VLA_ACCESS(2, bias, 0, 0,handle->bk), fp32_bias_scratch, handle->desc.K );
#endif
    for (ofm1 = my_in_start; ofm1 < my_in_end; ++ofm1) {
      for (mb1 = my_im_start; mb1 < my_im_end; ++mb1) {
#ifdef WR_PREFETCH_OUTPUT
        prefetchwt_chunk((char*)&LIBXSMM_VLA_ACCESS(4,  output, mb1, ofm1, 0, 0, nBlocksOFm, handle->bn, handle->bk), handle->bn*handle->bk*sizeof(libxsmm_bfloat16));
#endif
#ifndef LIBXSMM_DNN_FC_FWD_FUSE_NONE
#ifdef LIBXSMM_DNN_FC_FWD_FUSE_BIAS
        gemm_eltwise_params.bias_ptr  = (float*) fp32_bias_scratch + ofm1 * handle->bk;
#endif
#ifdef LIBXSMM_DNN_FC_FWD_FUSE_RELU
        gemm_eltwise_params.out_ptr   = &LIBXSMM_VLA_ACCESS(4, relubitmask, mb1, ofm1, 0, 0, nBlocksOFm, handle->bn, handle->bk/32);
#endif
        bf16_batchreduce_kernel_zerobeta_fused_eltwise( &LIBXSMM_VLA_ACCESS(5, filter_compressed, ofm1, 0, 0, 0, 0, nBlocksIFm, bc_lp, handle->bk/handle->sparsity_factor_A, lpb),
            &LIBXSMM_VLA_ACCESS(4, input,  mb1, 0,  0, 0, nBlocksIFm, handle->bn, handle->bc),
            &LIBXSMM_VLA_ACCESS(4, output, mb1,  ofm1, 0, 0, nBlocksOFm, bn, bk), &blocks, &gemm_eltwise_params);
#else
        bf16_batchreduce_kernel_zerobeta( &LIBXSMM_VLA_ACCESS(5, filter_compressed, ofm1, 0, 0, 0, 0, nBlocksIFm, bc_lp, handle->bk/handle->sparsity_factor_A, lpb),
            &LIBXSMM_VLA_ACCESS(4, input,  mb1, 0,  0, 0, nBlocksIFm, handle->bn, handle->bc),
            &LIBXSMM_VLA_ACCESS(4, output, mb1,  ofm1, 0, 0, nBlocksOFm, bn, bk), &blocks);
#endif
      }
    }
  }
} else {
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

#ifdef WR_PREFETCH_OUTPUT
          prefetchwt_chunk((char*)&LIBXSMM_VLA_ACCESS(4, output_f32, mb1, ofm1, 0, 0, nBlocksOFm, handle->bn, handle->bk), handle->bn*handle->bk*sizeof(float));
          if ( ifm1 == BF-1  ) {
            prefetchwt_chunk((char*)&LIBXSMM_VLA_ACCESS(4,  output, mb1, ofm1, 0, 0, nBlocksOFm, handle->bn, handle->bk), handle->bn*handle->bk*sizeof(libxsmm_bfloat16));
          }
#endif
          if (mb1 == my_im_start) {
            gemm_eltwise_params.sparse_bitmap     = &LIBXSMM_VLA_ACCESS(5, idx_filter_compressed, ofm1, ifm1*CB_BLOCKS, 0, 0, 0, nBlocksIFm, bc_lp, handle->bk/32, lpb);
            gemm_eltwise_params.decompress_buffer = &LIBXSMM_VLA_ACCESS(4, decompressed_filter, 0, 0, 0, 0, bc_lp, handle->bk, lpb);
            batchreduce_kernel_decompress( &LIBXSMM_VLA_ACCESS(5, filter_compressed, ofm1, ifm1*CB_BLOCKS, 0, 0, 0, nBlocksIFm, bc_lp, handle->bk/handle->sparsity_factor_A, lpb),
                &LIBXSMM_VLA_ACCESS(4, input,  mb1, ifm1*CB_BLOCKS, 0, 0, nBlocksIFm, handle->bn, handle->bc),
                &LIBXSMM_VLA_ACCESS(4, output_f32, mb1, ofm1, 0, 0, nBlocksOFm, handle->bn, handle->bk), &blocks, &gemm_eltwise_params);
          } else {
            batchreduce_kernel( &LIBXSMM_VLA_ACCESS(4, decompressed_filter, 0, 0, 0, 0, bc_lp, handle->bk, lpb),
                &LIBXSMM_VLA_ACCESS(4, input,  mb1, ifm1*CB_BLOCKS, 0, 0, nBlocksIFm, handle->bn, handle->bc),
                &LIBXSMM_VLA_ACCESS(4, output_f32, mb1, ofm1, 0, 0, nBlocksOFm, handle->bn, handle->bk), &blocks);
          }

          /* downconvert intermediate f32 tensor to bf 16 and store to final C */
          if ( ifm1 == BF-1  ) {
            eltwise_params.in.primary = &LIBXSMM_VLA_ACCESS(4, output_f32, mb1, ofm1, 0, 0, nBlocksOFm, handle->bn, handle->bk);
            eltwise_params.out.primary = &LIBXSMM_VLA_ACCESS(4, output, mb1, ofm1, 0, 0, nBlocksOFm, handle->bn, handle->bk);
#if defined(LIBXSMM_DNN_FC_FWD_FUSE_RELU)
            eltwise_params.out.secondary = &LIBXSMM_VLA_ACCESS(4, relubitmask, mb1, ofm1, 0, 0, nBlocksOFm, handle->bn, handle->bk/32);
#endif
            eltwise_kernel(&eltwise_params);
          }
        }
      }
    }
  } else {
#ifdef LIBXSMM_DNN_FC_FWD_FUSE_BIAS
    LIBXSMM_DNN_CONVERT_BUFFER_BF16_F32( &LIBXSMM_VLA_ACCESS(2, bias, 0, 0,handle->bk), fp32_bias_scratch, handle->desc.K );
#endif
    for (ofm1 = my_in_start; ofm1 < my_in_end; ++ofm1) {
      for (mb1 = my_im_start; mb1 < my_im_end; ++mb1) {
#ifdef WR_PREFETCH_OUTPUT
        prefetchwt_chunk((char*)&LIBXSMM_VLA_ACCESS(4,  output, mb1, ofm1, 0, 0, nBlocksOFm, handle->bn, handle->bk), handle->bn*handle->bk*sizeof(libxsmm_bfloat16));
#endif
#ifndef LIBXSMM_DNN_FC_FWD_FUSE_NONE
#ifdef LIBXSMM_DNN_FC_FWD_FUSE_BIAS
        gemm_eltwise_params.bias_ptr  = (float*) fp32_bias_scratch + ofm1 * handle->bk;
#endif
#ifdef LIBXSMM_DNN_FC_FWD_FUSE_RELU
        gemm_eltwise_params.out_ptr   = &LIBXSMM_VLA_ACCESS(4, relubitmask, mb1, ofm1, 0, 0, nBlocksOFm, handle->bn, handle->bk/32);
#endif
        if (mb1 == my_im_start) {
          gemm_eltwise_params.sparse_bitmap     = &LIBXSMM_VLA_ACCESS(5, idx_filter_compressed, ofm1, 0, 0, 0, 0, nBlocksIFm, bc_lp, handle->bk/32, lpb);
          gemm_eltwise_params.decompress_buffer = &LIBXSMM_VLA_ACCESS(4, decompressed_filter, 0, 0, 0, 0, bc_lp, handle->bk, lpb);
          bf16_batchreduce_kernel_zerobeta_fused_eltwise_decompress( &LIBXSMM_VLA_ACCESS(5, filter_compressed, ofm1, 0, 0, 0, 0, nBlocksIFm, bc_lp, handle->bk/handle->sparsity_factor_A, lpb),
              &LIBXSMM_VLA_ACCESS(4, input,  mb1, 0,  0, 0, nBlocksIFm, handle->bn, handle->bc),
              &LIBXSMM_VLA_ACCESS(4, output, mb1,  ofm1, 0, 0, nBlocksOFm, bn, bk), &blocks, &gemm_eltwise_params);
        } else {
          bf16_batchreduce_kernel_zerobeta_fused_eltwise( &LIBXSMM_VLA_ACCESS(4, decompressed_filter, 0, 0, 0, 0, bc_lp, handle->bk, lpb),
              &LIBXSMM_VLA_ACCESS(4, input,  mb1, 0,  0, 0, nBlocksIFm, handle->bn, handle->bc),
              &LIBXSMM_VLA_ACCESS(4, output, mb1,  ofm1, 0, 0, nBlocksOFm, bn, bk), &blocks, &gemm_eltwise_params);
        }
#else
        if (mb1 == my_im_start) {
          gemm_eltwise_params.sparse_bitmap     = &LIBXSMM_VLA_ACCESS(5, idx_filter_compressed, ofm1, 0, 0, 0, 0, nBlocksIFm, bc_lp, handle->bk/32, lpb);
          gemm_eltwise_params.decompress_buffer = &LIBXSMM_VLA_ACCESS(4, decompressed_filter, 0, 0, 0, 0, bc_lp, handle->bk, lpb);
          bf16_batchreduce_kernel_zerobeta_decompress( &LIBXSMM_VLA_ACCESS(5, filter_compressed, ofm1, 0, 0, 0, 0, nBlocksIFm, bc_lp, handle->bk/handle->sparsity_factor_A, lpb),
              &LIBXSMM_VLA_ACCESS(4, input,  mb1, 0,  0, 0, nBlocksIFm, handle->bn, handle->bc),
              &LIBXSMM_VLA_ACCESS(4, output, mb1,  ofm1, 0, 0, nBlocksOFm, bn, bk), &blocks, &gemm_eltwise_params);
        } else {
          bf16_batchreduce_kernel_zerobeta( &LIBXSMM_VLA_ACCESS(4, decompressed_filter, 0, 0, 0, 0, bc_lp, handle->bk, lpb),
              &LIBXSMM_VLA_ACCESS(4, input,  mb1, 0,  0, 0, nBlocksIFm, handle->bn, handle->bc),
              &LIBXSMM_VLA_ACCESS(4, output, mb1,  ofm1, 0, 0, nBlocksOFm, bn, bk), &blocks);
        }
#endif
      }
    }
  }
}
handle->tilerelease_kernel(NULL, NULL, NULL);
libxsmm_barrier_wait(handle->barrier, ltid);

