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

#ifndef LIBXSMM_DNN_POOL_H
#define LIBXSMM_DNN_POOL_H

#include <libxsmm.h>
#include <libxsmm_sync.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#if defined(_OPENMP)
# include <omp.h>
#endif

typedef enum libxsmm_dnn_pooling_pass {
  LIBXSMM_DNN_POOLING_PASS_FWD = 1,
  LIBXSMM_DNN_POOLING_PASS_BWD = 2
} libxsmm_dnn_pooling_pass;

typedef enum libxsmm_dnn_pooling_type {
  LIBXSMM_DNN_POOLING_TYPE_AVG = 1,
  LIBXSMM_DNN_POOLING_TYPE_MAX = 2,
  LIBXSMM_DNN_POOLING_TYPE_MAX_NOMASK = 3
} libxsmm_dnn_pooling_type;

typedef struct libxsmm_dnn_pooling_fwd_config {
  libxsmm_blasint  N;
  libxsmm_blasint  C;
  libxsmm_blasint  H;
  libxsmm_blasint  W;
  libxsmm_blasint  R;
  libxsmm_blasint  S;
  libxsmm_blasint  bc;
  libxsmm_blasint  Bc;
  libxsmm_blasint  ofh;
  libxsmm_blasint  ofw;
  libxsmm_blasint  u;
  libxsmm_blasint  v;
  libxsmm_blasint  pad_h;
  libxsmm_blasint  pad_w;
  libxsmm_blasint  pad_h_in;
  libxsmm_blasint  pad_w_in;
  libxsmm_blasint  pad_h_out;
  libxsmm_blasint  pad_w_out;
  libxsmm_blasint  threads;
  libxsmm_datatype datatype_in;
  libxsmm_datatype datatype_out;
  libxsmm_datatype datatype_comp;
  libxsmm_dnn_pooling_type  pool_type;
  libxsmm_dnn_pooling_pass  pass_type;
  size_t           scratch_size;
  libxsmm_barrier* barrier;
  /* Aux TPP kernels */
  libxsmm_meltwfunction_unary   fwd_pool_reduce_kernel;
  libxsmm_meltwfunction_binary  fwd_scale_kernel;
} libxsmm_dnn_pooling_fwd_config;

typedef struct libxsmm_dnn_pooling_bwd_config {
  libxsmm_blasint  N;
  libxsmm_blasint  C;
  libxsmm_blasint  H;
  libxsmm_blasint  W;
  libxsmm_blasint  R;
  libxsmm_blasint  S;
  libxsmm_blasint  bc;
  libxsmm_blasint  Bc;
  libxsmm_blasint  ofh;
  libxsmm_blasint  ofw;
  libxsmm_blasint  u;
  libxsmm_blasint  v;
  libxsmm_blasint  pad_h;
  libxsmm_blasint  pad_w;
  libxsmm_blasint  pad_h_in;
  libxsmm_blasint  pad_w_in;
  libxsmm_blasint  pad_h_out;
  libxsmm_blasint  pad_w_out;
  libxsmm_blasint  threads;
  libxsmm_datatype datatype_in;
  libxsmm_datatype datatype_out;
  libxsmm_datatype datatype_comp;
  libxsmm_dnn_pooling_type  pool_type;
  libxsmm_dnn_pooling_pass  pass_type;
  size_t           scratch_size;
  libxsmm_barrier* barrier;
  /* Aux TPP kernels */
  libxsmm_matrix_eqn_function   func_bwd_max_pool;
  libxsmm_meltwfunction_unary   bwd_zero_kernel;
  libxsmm_meltwfunction_binary  func_bwd_avg_pool;
} libxsmm_dnn_pooling_bwd_config;

LIBXSMM_API libxsmm_dnn_pooling_fwd_config setup_libxsmm_dnn_pooling_fwd( const libxsmm_blasint N, const libxsmm_blasint C, const libxsmm_blasint H, const libxsmm_blasint W,
                                            const libxsmm_blasint R, const libxsmm_blasint S,
                                            const libxsmm_blasint stride_h, const libxsmm_blasint stride_w,
                                            const libxsmm_blasint pad_h, const libxsmm_blasint pad_w,
                                            const libxsmm_blasint pad_h_in, const libxsmm_blasint pad_w_in,
                                            const libxsmm_blasint pad_h_out, const libxsmm_blasint pad_w_out,
                                            const libxsmm_blasint bc, const libxsmm_blasint threads, const libxsmm_dnn_pooling_type pool_type,
                                            const libxsmm_datatype datatype_in, const libxsmm_datatype datatype_out, const libxsmm_datatype datatype_comp );

LIBXSMM_API libxsmm_dnn_pooling_bwd_config setup_libxsmm_dnn_pooling_bwd( const libxsmm_blasint N, const libxsmm_blasint C, const libxsmm_blasint H, const libxsmm_blasint W,
                                            const libxsmm_blasint R, const libxsmm_blasint S,
                                            const libxsmm_blasint stride_h, const libxsmm_blasint stride_w,
                                            const libxsmm_blasint pad_h, const libxsmm_blasint pad_w,
                                            const libxsmm_blasint pad_h_in, const libxsmm_blasint pad_w_in,
                                            const libxsmm_blasint pad_h_out, const libxsmm_blasint pad_w_out,
                                            const libxsmm_blasint bc, const libxsmm_blasint threads, const libxsmm_dnn_pooling_type pool_type,
                                            const libxsmm_datatype datatype_in, const libxsmm_datatype datatype_out, const libxsmm_datatype datatype_comp );

LIBXSMM_API void libxsmm_dnn_pooling_fwd_exec_f32( const libxsmm_dnn_pooling_fwd_config cfg, const float* in_act_ptr, float* out_act_ptr, int* mask_ptr,
                              const libxsmm_blasint start_tid, const libxsmm_blasint my_tid, void* scratch );

LIBXSMM_API void libxsmm_dnn_pooling_fwd_exec_bf16( const libxsmm_dnn_pooling_fwd_config cfg, const libxsmm_bfloat16* in_act_ptr, libxsmm_bfloat16* out_act_ptr, int* mask_ptr,
                               const libxsmm_blasint start_tid, const libxsmm_blasint my_tid, void* scratch );

LIBXSMM_API void libxsmm_dnn_pooling_bwd_exec_f32( const libxsmm_dnn_pooling_bwd_config cfg, float* din_act_ptr, const float* dout_act_ptr, const int* mask_ptr,
                              const libxsmm_blasint start_tid, const libxsmm_blasint my_tid, void* scratch );

LIBXSMM_API void libxsmm_dnn_pooling_bwd_exec_bf16( const libxsmm_dnn_pooling_bwd_config cfg, libxsmm_bfloat16* din_act_ptr, const libxsmm_bfloat16* dout_act_ptr, const int* mask_ptr,
                               const libxsmm_blasint start_tid, const libxsmm_blasint my_tid, void* scratch );

LIBXSMM_API void destroy_libxsmm_dnn_pooling_fwd(libxsmm_dnn_pooling_fwd_config* cfg);

LIBXSMM_API void destroy_libxsmm_dnn_pooling_bwd(libxsmm_dnn_pooling_bwd_config* cfg);

#endif

