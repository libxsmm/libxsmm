/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Heinecke (Intel Corp.)
******************************************************************************/

#ifndef LIBXSMM_DNN_FC_H
#define LIBXSMM_DNN_FC_H

#include <libxsmm.h>
#include <libxsmm_sync.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#if defined(_OPENMP)
# include <omp.h>
#endif

typedef enum libxsmm_dnn_fc_eltw_fuse {
  LIBXSMM_DNN_FC_ELTW_FUSE_NONE = 0,
  LIBXSMM_DNN_FC_ELTW_FUSE_BIAS = 1,
  LIBXSMM_DNN_FC_ELTW_FUSE_RELU = 2,
  /* 3 is reserved for tanh */
  LIBXSMM_DNN_FC_ELTW_FUSE_BIAS_RELU = 4,
  /* 5 is reserved for tanh + bias, see naive */
  LIBXSMM_DNN_FC_ELTW_FUSE_RELU_WITH_MASK = 6,
  LIBXSMM_DNN_FC_ELTW_FUSE_BIAS_RELU_WITH_MASK = 7
} libxsmm_dnn_fc_eltw_fuse;

typedef enum libxsmm_dnn_fc_pass {
  LIBXSMM_DNN_FC_PASS_FWD   = 1,
  LIBXSMM_DNN_FC_PASS_BWD_D = 2,
  LIBXSMM_DNN_FC_PASS_BWD_W = 4,
  LIBXSMM_DNN_FC_PASS_BWD   = 6
} libxsmm_dnn_fc_pass;

typedef struct libxsmm_dnn_fc_fwd_config {
  libxsmm_blasint N;
  libxsmm_blasint C;
  libxsmm_blasint K;
  libxsmm_blasint bn;
  libxsmm_blasint bc;
  libxsmm_blasint bk;
  libxsmm_blasint threads;
  libxsmm_dnn_fc_eltw_fuse fuse_type;
  libxsmm_blasint fwd_bf;
  libxsmm_blasint fwd_2d_blocking;
  libxsmm_blasint fwd_row_teams;
  libxsmm_blasint fwd_col_teams;
  libxsmm_blasint fwd_M_hyperpartitions;
  libxsmm_blasint fwd_N_hyperpartitions;
  size_t          scratch_size;
  libxsmm_barrier* barrier;
  libxsmm_gemmfunction fwd_tileconfig_kernel;
  libxsmm_gemmfunction fwd_tilerelease_kernel;
  libxsmm_gemmfunction fwd_compute_kernel_strd;                /* beta = 1.0 */
  libxsmm_gemmfunction fwd_compute_kernel2_strd;               /* beta = 0.0 */
  libxsmm_gemmfunction_ext fwd_compute_kernel_strd_fused_f32;  /* beta = 1.0 + relu */
  libxsmm_gemmfunction_ext fwd_compute_kernel2_strd_fused_f32; /* beta = 0.0 + bias */
  libxsmm_gemmfunction_ext fwd_compute_kernel3_strd_fused_f32; /* beta = 0.0 + bias + relu */
  libxsmm_gemmfunction_ext fwd_compute_kernel4_strd_fused_f32; /* beta = 0.0 + relu */
  libxsmm_gemmfunction_ext fwd_compute_kernel5_strd_fused;     /* beta = 0.0 + bias + relu */
  libxsmm_meltwfunction_unary fwd_zero_kernel;
  libxsmm_meltwfunction_unary fwd_act_kernel;
  libxsmm_meltwfunction_unary fwd_colbcast_load_kernel;
  libxsmm_meltwfunction_unary fwd_store_kernel;
} libxsmm_dnn_fc_fwd_config;

typedef struct libxsmm_dnn_fc_bwd_config {
  libxsmm_blasint N;
  libxsmm_blasint C;
  libxsmm_blasint K;
  libxsmm_blasint bn;
  libxsmm_blasint bc;
  libxsmm_blasint bk;
  libxsmm_blasint threads;
  libxsmm_dnn_fc_eltw_fuse fuse_type;
  libxsmm_blasint bwd_bf;
  libxsmm_blasint bwd_2d_blocking;
  libxsmm_blasint bwd_col_teams;
  libxsmm_blasint bwd_row_teams;
  libxsmm_blasint bwd_M_hyperpartitions;
  libxsmm_blasint bwd_N_hyperpartitions;
  libxsmm_blasint upd_bf;
  libxsmm_blasint upd_2d_blocking;
  libxsmm_blasint upd_col_teams;
  libxsmm_blasint upd_row_teams;
  libxsmm_blasint upd_M_hyperpartitions;
  libxsmm_blasint upd_N_hyperpartitions;
  libxsmm_blasint ifm_subtasks;
  libxsmm_blasint ofm_subtasks;
  size_t  bwd_private_tr_wt_scratch_mark;
  size_t  upd_private_tr_act_scratch_mark;
  size_t  upd_private_tr_dact_scratch_mark;
  size_t  scratch_size;
  size_t  doutput_scratch_mark;
  libxsmm_barrier* barrier;
  libxsmm_gemmfunction bwd_tileconfig_kernel;
  libxsmm_gemmfunction bwd_tilerelease_kernel;
  libxsmm_gemmfunction bwd_compute_kernel_strd;   /* beta = 1.0 (bwd) */
  libxsmm_gemmfunction bwd_compute_kernel2_strd;  /* beta = 0.0 (bwd) */
  libxsmm_meltwfunction_unary bwd_relu_kernel;
  libxsmm_meltwfunction_unary delbias_reduce_kernel;
  libxsmm_meltwfunction_unary bwd_zero_kernel;
  libxsmm_meltwfunction_unary bwd_store_kernel;
  libxsmm_meltwfunction_unary vnni_to_vnniT_kernel;
  libxsmm_gemmfunction upd_tileconfig_kernel;
  libxsmm_gemmfunction upd_tilerelease_kernel;
  libxsmm_gemmfunction upd_compute_kernel_strd;   /* beta = 1.0 (upd) */
  libxsmm_gemmfunction upd_compute_kernel2_strd;  /* beta = 0.0 (upd) */
  libxsmm_meltwfunction_unary upd_store_kernel;
  libxsmm_meltwfunction_unary upd_zero_kernel;
  libxsmm_meltwfunction_unary norm_to_normT_kernel;
  libxsmm_meltwfunction_unary norm_to_vnni_kernel;
  libxsmm_meltwfunction_unary norm_to_vnni_kernel_wt;
} libxsmm_dnn_fc_bwd_config;

libxsmm_dnn_fc_fwd_config setup_libxsmm_dnn_fc_fwd(libxsmm_blasint N, libxsmm_blasint C, libxsmm_blasint K, libxsmm_blasint bn,
                                 libxsmm_blasint bc, libxsmm_blasint bk, libxsmm_blasint threads, libxsmm_dnn_fc_eltw_fuse fuse_type,
                                 libxsmm_datatype datatype_in, libxsmm_datatype datatype_out, libxsmm_datatype datatype_comp );

libxsmm_dnn_fc_bwd_config setup_libxsmm_dnn_fc_bwd(libxsmm_blasint N, libxsmm_blasint C, libxsmm_blasint K, libxsmm_blasint bn,
    libxsmm_blasint bc, libxsmm_blasint bk, libxsmm_blasint threads, libxsmm_dnn_fc_eltw_fuse fuse_type,
    libxsmm_datatype datatype_in, libxsmm_datatype datatype_out, libxsmm_datatype datatype_comp );

void libxsmm_dnn_fc_fwd_exec_f32( libxsmm_dnn_fc_fwd_config cfg, const float* wt_ptr, const float* in_act_ptr, float* out_act_ptr,
    const float* bias_ptr, unsigned char* relu_ptr, int start_tid, int my_tid, void* scratch );

void libxsmm_dnn_fc_fwd_exec_bf16( libxsmm_dnn_fc_fwd_config cfg, const libxsmm_bfloat16* wt_ptr, const libxsmm_bfloat16* in_act_ptr, libxsmm_bfloat16* out_act_ptr,
                          const libxsmm_bfloat16* bias_ptr, unsigned char* relu_ptr, int start_tid, int my_tid, void* scratch );

void libxsmm_dnn_fc_bwd_exec_f32( libxsmm_dnn_fc_bwd_config cfg, const float* wt_ptr, float* din_act_ptr,
    const float* dout_act_ptr, float* dwt_ptr, const float* in_act_ptr,
    float* dbias_ptr, const unsigned char* relu_ptr, libxsmm_dnn_fc_pass pass, int start_tid, int my_tid, void* scratch );

void libxsmm_dnn_fc_bwd_exec_bf16( libxsmm_dnn_fc_bwd_config cfg,  const libxsmm_bfloat16* wt_ptr, libxsmm_bfloat16* din_act_ptr,
                          const libxsmm_bfloat16* dout_act_ptr, libxsmm_bfloat16* dwt_ptr, const libxsmm_bfloat16* in_act_ptr,
                          libxsmm_bfloat16* dbias_ptr, const unsigned char* relu_ptr, libxsmm_dnn_fc_pass pass, int start_tid, int my_tid, void* scratch );

void destroy_libxsmm_dnn_fc_fwd(libxsmm_dnn_fc_fwd_config* cfg);

void destroy_libxsmm_dnn_fc_bwd(libxsmm_dnn_fc_bwd_config* cfg);

#endif

