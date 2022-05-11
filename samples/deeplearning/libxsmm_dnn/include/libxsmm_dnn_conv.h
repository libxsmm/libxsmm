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

#ifndef LIBXSMM_DNN_CONV_H
#define LIBXSMM_DNN_CONV_H

#include <libxsmm.h>
#include <libxsmm_sync.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#if defined(_OPENMP)
# include <omp.h>
#endif

typedef enum libxsmm_dnn_conv_eltwise_fuse {
  LIBXSMM_DNN_CONV_ELTWISE_FUSE_NONE = 0,
  LIBXSMM_DNN_CONV_ELTWISE_FUSE_BIAS = 1,
  LIBXSMM_DNN_CONV_ELTWISE_FUSE_RELU = 2,
  LIBXSMM_DNN_CONV_ELTWISE_FUSE_BIAS_RELU = LIBXSMM_DNN_CONV_ELTWISE_FUSE_BIAS | LIBXSMM_DNN_CONV_ELTWISE_FUSE_RELU
} libxsmm_dnn_conv_eltwise_fuse;

typedef enum libxsmm_dnn_conv_pass {
  LIBXSMM_DNN_CONV_PASS_FWD   = 1,
  LIBXSMM_DNN_CONV_PASS_BWD_D = 2,
  LIBXSMM_DNN_CONV_PASS_BWD_W = 4,
  LIBXSMM_DNN_CONV_PASS_BWD   = 6
} libxsmm_dnn_conv_pass;

typedef struct libxsmm_dnn_conv_config {
  /* Convolution params  */
  libxsmm_blasint N;
  libxsmm_blasint H;
  libxsmm_blasint W;
  libxsmm_blasint C;
  libxsmm_blasint K;
  libxsmm_blasint R;
  libxsmm_blasint S;
  libxsmm_blasint u;
  libxsmm_blasint v;
  libxsmm_blasint pad_h;
  libxsmm_blasint pad_w;
  libxsmm_blasint pad_h_in;
  libxsmm_blasint pad_w_in;
  libxsmm_blasint pad_h_out;
  libxsmm_blasint pad_w_out;
  libxsmm_blasint threads;
  libxsmm_blasint  overwrite_output;
  libxsmm_blasint  avoid_bwd_wt_trans;
  libxsmm_blasint  zero_fwd_output_rim;
  libxsmm_dnn_conv_eltwise_fuse  fuse_type;
  libxsmm_datatype datatype_in;
  libxsmm_datatype datatype_out;
  int target_archid;

  /* additional size for internal data types */
  int bc;
  int bk;
  int ifhp;
  int ifwp;
  int ofh;
  int ofw;
  int ofhp;
  int ofwp;
  int ifmblock;
  int ofmblock;
  int blocksifm;
  int blocksofm;
  int fwd_ofw_rb;
  int fwd_ofh_rb;
  int bwd_ofw_rb;
  int bwd_ofh_rb;
  int upd_ofw_rb;
  int upd_ofh_rb;
  int fm_lp_block; /* additional blocking for low precision datatypes of feature maps */
  int blocksifm_blocking;
  int blocksofm_blocking;
  int avoid_acc_load;
  int avoid_acc_load_bwd;
  int pack_input;
  int pack_input_bwd;
  int spread_input_bwd;
  int weight_copies;
  int loop_order;
  int use_ofm_parallelization;
  int use_ifm_parallelization;
  int avoid_fmas_in_rim;
  int upd_use_batchreduce;
  int upd_pack_input;
  int upd_loop_order;
  int upd_linearized_tasklist;
  int upd_avoid_rim_fmas;
  int fwd_flags;
  int bwd_flags;
  int shuffle_filter_accesses;
  int use_fallback_fwd_loops;
  int use_fallback_bwd_loops;
  int fwd_gemm_pixels;
  int bwd_gemm_pixels;
  int input_pixels;
  int output_pixels;
  int n_used_pixels;
  int pixel_blocking;
  int use_intermediate_f32_wt_tensor;
  int upd_linearized_pixels;
  int ifwp_extended;
  int ofwp_extended;
  int batchreduce_h_pixels;
  int on_the_fly_input_packing;
  int upd_pack_input_upfront;
  int use_hybrid_imgofm_parallelization;
  int remainder_pixels;
  int pack_to_cnhw;
  int fuse_upd_transposes;
  int compute_pixels;
  int upd_trans_w_only;
  int fwd_padding_copy;
  int upd_padding_copy;
  int upd_remaining_pixels;
  int block_fwd_oj;
  int block_fwd_ifm;
  int block_fwd_ofm;
  int block_bwd_oj;
  int block_bwd_ifm;
  int block_bwd_ofm;
  int block_upd_ifm;
  int block_upd_ofm;

  /* Hoisting the compute kernels for FWD  */
  libxsmm_xmmfunction fwd_compute_kernel_strd_fused_f32;
  libxsmm_xmmfunction fwd_compute_kernel_strd_f32;
  libxsmm_xmmfunction fwd_compute_kernel2_strd_f32;
  libxsmm_xmmfunction fwd_compute_kernel_offs_fused_f32;
  libxsmm_xmmfunction fwd_compute_kernel_offs_f32;

  libxsmm_meltwfunction_unary strided_copy_kernel_f32;
  libxsmm_meltwfunction_unary ifmblock_copy_kernel_f32;
  libxsmm_meltwfunction_unary ifmblock_zero_kernel_f32;
  libxsmm_meltwfunction_unary ofmblock_zero_kernel_f32;
  libxsmm_meltwfunction_unary ofw_x_ofmblock_zero_kernel_f32;
  libxsmm_meltwfunction_unary ofh_x_ofw_x_ofmblock_zero_kernel_f32;
  libxsmm_meltwfunction_unary  relu_kernel_f32;
  libxsmm_meltwfunction_binary colbias_add_kernel_f32;

  libxsmm_xmmfunction fwd_compute_kernel_strd_fused_bf16;
  libxsmm_xmmfunction fwd_compute_kernel_strd_bf16;
  libxsmm_xmmfunction fwd_compute_kernel2_strd_bf16;
  libxsmm_xmmfunction fwd_compute_kernel_offs_fused_bf16;
  libxsmm_xmmfunction fwd_compute_kernel_offs_bf16;
  libxsmm_xmmfunction fwd_compute_kernel_strd_bf16f32;
  libxsmm_xmmfunction fwd_compute_kernel2_strd_bf16f32;
  libxsmm_meltwfunction_unary cvt_kernel_fp32bf16;

  libxsmm_meltwfunction_unary strided_copy_kernel_bf16;
  libxsmm_meltwfunction_unary ifmblock_copy_kernel_bf16;
  libxsmm_meltwfunction_unary ifmblock_zero_kernel_bf16;
  libxsmm_meltwfunction_unary ofmblock_zero_kernel_bf16;
  libxsmm_meltwfunction_unary ofw_x_ofmblock_zero_kernel_bf16;
  libxsmm_meltwfunction_unary ofh_x_ofw_x_ofmblock_zero_kernel_bf16;
  libxsmm_meltwfunction_unary  relu_kernel_bf16;
  libxsmm_meltwfunction_binary colbias_add_kernel_bf16;

  /* Hoisting the compute kernels for BWD  */
  libxsmm_xmmfunction bwd_compute_kernel_strd_f32;
  libxsmm_xmmfunction bwd_compute_kernel2_strd_f32;
  libxsmm_xmmfunction bwd_compute_kernel_offs_f32;
  libxsmm_xmmfunction bwd_compute_kernel_fallback_f32;

  libxsmm_xmmfunction bwd_compute_kernel_strd_bf16;
  libxsmm_xmmfunction bwd_compute_kernel2_strd_bf16;
  libxsmm_xmmfunction bwd_compute_kernel_strd_bf16f32;
  libxsmm_xmmfunction bwd_compute_kernel2_strd_bf16f32;
  libxsmm_xmmfunction bwd_compute_kernel_offs_bf16;
  libxsmm_xmmfunction bwd_compute_kernel_fallback_bf16;
  libxsmm_meltwfunction_unary cvt_kernel_bwd_fp32bf16;
  libxsmm_meltwfunction_unary tr_kernel;

  libxsmm_meltwfunction_unary ofh_x_ofw_x_ifmblock_zero_kernel_f32;
  libxsmm_meltwfunction_unary paddedH_x_paddedW_x_ifmblock_zero_kernel_f32;
  libxsmm_meltwfunction_unary ifhp_x_ifwp_x_ifmblock_zero_kernel_f32;

  libxsmm_meltwfunction_unary ofh_x_ofw_x_ifmblock_zero_kernel_bf16;
  libxsmm_meltwfunction_unary paddedH_x_paddedW_x_ifmblock_zero_kernel_bf16;
  libxsmm_meltwfunction_unary ifhp_x_ifwp_x_ifmblock_zero_kernel_bf16;

  /* Hoisting the compute kernels for UPD  */
  libxsmm_xmmfunction upd_compute_kernel_no_linearized_tasklist_f32;
  libxsmm_xmmfunction upd_compute_kernel_linearized_tasklist_f32;
  libxsmm_xmmfunction upd_compute_kernel_linearized_tasklist_offs_f32;
  libxsmm_xmmfunction upd_compute_kernel2_linearized_tasklist_offs_f32;
  libxsmm_xmmfunction upd_compute_kernel_flat_linearized_tasklist_offs_f32;
  libxsmm_xmmfunction upd_compute_kernel_hybrid_linearized_tasklist_offs_f32;

  libxsmm_meltwfunction_unary zero_weights_kernel_f32;
  libxsmm_meltwfunction_unary zero_ifmblock_x_ofmblock_kernel_f32;
  libxsmm_meltwfunction_unary wt_reduce_kernel0_f32;
  libxsmm_meltwfunction_unary wt_reduce_kernel1_f32;

  libxsmm_xmmfunction upd_compute_kernel1_bf16f32;
  libxsmm_xmmfunction upd_compute_kernel2_bf16f32;
  libxsmm_xmmfunction upd_compute_kernel3_bf16f32;
  libxsmm_xmmfunction upd_compute_kernel4_bf16f32;

  libxsmm_meltwfunction_unary upd_weight_cvt_f32bf16;
  libxsmm_meltwfunction_unary upd_weight_vnni_format_bf16;
  libxsmm_meltwfunction_unary zero_full_weights_f32;
  libxsmm_meltwfunction_unary zero_partial_weights_f32;
  libxsmm_meltwfunction_unary zero_ofmblock_pixels_bf16;
  libxsmm_meltwfunction_unary zero_ifmblock_input_pixels_bf16;
  libxsmm_meltwfunction_unary zero_ifmblock_input_pixels_extended_bf16;
  libxsmm_meltwfunction_unary zero_ofmblock_output_pixels_bf16;
  libxsmm_meltwfunction_unary transpose_input_pixels_bf16;
  libxsmm_meltwfunction_unary transposeNpack_input_pixels_bf16;
  libxsmm_meltwfunction_unary transpose_input_pixels_ifwp_extended_bf16;
  libxsmm_meltwfunction_unary transpose_input_pixels_ifwp_strided_extended_bf16;
  libxsmm_meltwfunction_unary transpose_input_pixels_ifwp_extended2_bf16;
  libxsmm_meltwfunction_unary vnni_output_pixels_bf16;
  libxsmm_meltwfunction_unary vnni_output_pixels_extended_bf16;
  libxsmm_meltwfunction_unary vnni_output_w_pixels_bf16;
  libxsmm_meltwfunction_unary vnni_output_w2_pixels_bf16;
  libxsmm_meltwfunction_unary vnni_output_compute_pixels_bf16;
  libxsmm_meltwfunction_unary vnni_output_zero_remaining_pixels_bf16;

  libxsmm_meltwfunction_unary wt_reduce_kernel0_bf16;
  libxsmm_meltwfunction_unary wt_reduce_kernel1_bf16;

  unsigned long long *A_offsets;
  unsigned long long *B_offsets;
  unsigned long long *A_offsets_bwd;
  unsigned long long *B_offsets_bwd;
  unsigned long long *A_offsets_upd;
  unsigned long long *B_offsets_upd;
  unsigned long long *A_offsets2_upd;
  unsigned long long *B_offsets2_upd;
  unsigned long long *A_offsets3_upd;
  unsigned long long *B_offsets3_upd;
  /* barrier */
  libxsmm_barrier* barrier;

  /* scratch */
  size_t fwd_packing_padding_scratch_size;
  size_t fwd_lp_output_full_scratch_size;
  size_t fwd_lp_output_block_scratch_size;
  size_t fwd_packing_padding_scratch_offset;
  size_t fwd_lp_output_full_scratch_offset;
  size_t fwd_lp_output_block_scratch_offset;
  size_t fwd_scratch_size;

  size_t bwd_filter_trans_scratch_size;
  size_t bwd_packing_padding_scratch_size;
  size_t bwd_lp_input_full_scratch_size;
  size_t bwd_filter_trans_scratch_offset;
  size_t bwd_packing_padding_scratch_offset;
  size_t bwd_lp_input_full_scratch_offset;
  size_t bwd_scratch_size;

  size_t upd_packing_padding_scratch_size;
  size_t upd_lp_output_full_scratch_size;
  size_t upd_lp_input_full_scratch_size;
  size_t upd_filter_scratch_size;
  size_t upd_lp_filter_full_scratch_size;
  size_t upd_packing_padding_scratch_offset;
  size_t upd_lp_output_full_scratch_offset;
  size_t upd_lp_input_full_scratch_offset;
  size_t upd_lp_filter_full_scratch_offset;
  size_t upd_filter_scratch_offset;
  size_t upd_scratch_size;

  size_t scratch_size;

} libxsmm_dnn_conv_config;

LIBXSMM_API libxsmm_dnn_conv_config setup_libxsmm_dnn_conv( libxsmm_datatype cnn_dtype_in, libxsmm_datatype cnn_dtype_out, libxsmm_blasint N, libxsmm_blasint H, libxsmm_blasint W, libxsmm_blasint C, libxsmm_blasint K, libxsmm_blasint R, libxsmm_blasint S,
    libxsmm_blasint stride_h, libxsmm_blasint stride_w,
    libxsmm_blasint pad_h, libxsmm_blasint pad_w,
    libxsmm_blasint pad_h_in, libxsmm_blasint pad_w_in,
    libxsmm_blasint pad_h_out, libxsmm_blasint pad_w_out,
    libxsmm_blasint bc, libxsmm_blasint bk, libxsmm_blasint threads, libxsmm_dnn_conv_eltwise_fuse fuse_type, libxsmm_blasint overwrite_output, libxsmm_blasint avoid_bwd_wt_trans, libxsmm_blasint zero_fwd_output_rim);

LIBXSMM_API void destroy_libxsmm_dnn_conv(libxsmm_dnn_conv_config* cfg);

LIBXSMM_API void libxsmm_dnn_conv_bwd_exec( libxsmm_dnn_conv_config cfg, const float* wt_ptr, const float* tr_wt_ptr,  const float* dout_act_ptr, float* din_act_ptr,
    unsigned char* relu_ptr, int start_tid, int my_tid, void* scratch );

LIBXSMM_API void libxsmm_dnn_conv_bwd_exec_bf16( libxsmm_dnn_conv_config cfg, const libxsmm_bfloat16* wt_ptr, const libxsmm_bfloat16* tr_wt_ptr,  const libxsmm_bfloat16* dout_act_ptr, libxsmm_bfloat16* din_act_ptr,
    unsigned char* relu_ptr, int start_tid, int my_tid, void* scratch );

LIBXSMM_API void libxsmm_dnn_conv_fwd_exec( libxsmm_dnn_conv_config cfg, const float* wt_ptr, const float* in_act_ptr, float* out_act_ptr,
    const float* bias_ptr, unsigned char* relu_ptr, int start_tid, int my_tid, void* scratch );

LIBXSMM_API void libxsmm_dnn_conv_fwd_exec_bf16( libxsmm_dnn_conv_config cfg, const libxsmm_bfloat16* wt_ptr, const libxsmm_bfloat16* in_act_ptr, libxsmm_bfloat16* out_act_ptr,
    const libxsmm_bfloat16* bias_ptr, unsigned char* relu_ptr, int start_tid, int my_tid, void* scratch );

LIBXSMM_API void libxsmm_dnn_conv_upd_exec( libxsmm_dnn_conv_config cfg, const float* in_act_ptr, const float* dout_act_ptr, float* dfilter_ptr,
    unsigned char* bias_ptr, int start_tid, int my_tid, void* scratch );

LIBXSMM_API void libxsmm_dnn_conv_upd_exec_bf16( libxsmm_dnn_conv_config cfg, const libxsmm_bfloat16* in_act_ptr, const libxsmm_bfloat16* dout_act_ptr, libxsmm_bfloat16* dfilter_ptr,
    unsigned char* bias_ptr, int start_tid, int my_tid, void* scratch );

#endif

