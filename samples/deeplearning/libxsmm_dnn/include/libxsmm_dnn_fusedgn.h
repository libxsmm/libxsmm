/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Kirill Voronin (Intel Corp.)
******************************************************************************/

#ifndef LIBXSMM_DNN_FUSEDGN_H
#define LIBXSMM_DNN_FUSEDGN_H

#include <libxsmm.h>
#include <libxsmm_sync.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#if defined(_OPENMP)
# include <omp.h>
#endif

typedef enum libxsmm_dnn_gn_fuse {
  LIBXSMM_DNN_GN_FUSE_NONE = 0,
  LIBXSMM_DNN_GN_FUSE_RELU = 1,
  LIBXSMM_DNN_GN_FUSE_ELTWISE = 2,
  LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU = 3,
  LIBXSMM_DNN_GN_FUSE_RELU_WITH_MASK = 4,
  LIBXSMM_DNN_GN_FUSE_ELTWISE_RELU_WITH_MASK = 5
} libxsmm_dnn_gn_fuse;

typedef struct libxsmm_dnn_gn_fwd_config {
  libxsmm_blasint  N;
  libxsmm_blasint  C;
  libxsmm_blasint  G;
  libxsmm_blasint  H;
  libxsmm_blasint  W;
  libxsmm_blasint  bc;
  libxsmm_blasint  CP;
  libxsmm_blasint  num_HW_blocks;
  libxsmm_blasint  threads;
  size_t           scratch_size;

  libxsmm_datatype datatype_in;
  libxsmm_datatype datatype_out;
  libxsmm_datatype datatype_comp;

  libxsmm_barrier* barrier;

  libxsmm_matrix_eqn_function  func10;
  libxsmm_meltwfunction_unary  reduce_HW_kernel;
  libxsmm_meltwfunction_unary  reduce_rows_kernel;
  libxsmm_meltwfunction_unary  reduce_groups_kernel;
  libxsmm_meltwfunction_unary  all_zero_G_kernel;
  libxsmm_meltwfunction_unary  all_zero_kernel;
  libxsmm_meltwfunction_binary add_kernel;
  libxsmm_dnn_gn_fuse                   fuse_type;
} libxsmm_dnn_gn_fwd_config;

typedef struct libxsmm_dnn_gn_bwd_config {
  libxsmm_blasint  N;
  libxsmm_blasint  C;
  libxsmm_blasint  G;
  libxsmm_blasint  H;
  libxsmm_blasint  W;
  libxsmm_blasint  bc;
  libxsmm_blasint  CP;
  libxsmm_blasint  num_HW_blocks;
  libxsmm_blasint  threads;
  size_t           scratch_size;

  libxsmm_datatype datatype_in;
  libxsmm_datatype datatype_out;
  libxsmm_datatype datatype_comp;

  libxsmm_barrier* barrier;

  libxsmm_matrix_eqn_function  dgamma_func;
  libxsmm_matrix_eqn_function  dbeta_func;
  libxsmm_matrix_eqn_function  db_func;
  libxsmm_matrix_eqn_function  ds_func;
  libxsmm_matrix_eqn_function  din_func;
  libxsmm_meltwfunction_unary  all_zero_kernel;
  libxsmm_meltwfunction_unary  inv_relu_kernel;
  libxsmm_meltwfunction_unary  ewise_copy_kernel;
  libxsmm_dnn_gn_fuse                   fuse_type;
} libxsmm_dnn_gn_bwd_config;

LIBXSMM_API libxsmm_dnn_gn_fwd_config setup_libxsmm_dnn_gn_fwd(libxsmm_blasint N, libxsmm_blasint C, libxsmm_blasint H, libxsmm_blasint W, libxsmm_blasint G, libxsmm_blasint bc,
                                 libxsmm_blasint threads, libxsmm_dnn_gn_fuse fuse_type,
                                 libxsmm_datatype datatype_in, libxsmm_datatype datatype_out, libxsmm_datatype datatype_comp );

LIBXSMM_API libxsmm_dnn_gn_bwd_config setup_libxsmm_dnn_gn_bwd(libxsmm_blasint N, libxsmm_blasint C, libxsmm_blasint H, libxsmm_blasint W, libxsmm_blasint G, libxsmm_blasint bc,
                                 libxsmm_blasint threads, libxsmm_dnn_gn_fuse fuse_type,
                                 libxsmm_datatype datatype_in, libxsmm_datatype datatype_out, libxsmm_datatype datatype_comp );

LIBXSMM_API void destroy_libxsmm_dnn_gn_fwd(libxsmm_dnn_gn_fwd_config* cfg);

LIBXSMM_API void destroy_libxsmm_dnn_gn_bwd(libxsmm_dnn_gn_bwd_config* cfg);

LIBXSMM_API void libxsmm_dnn_gn_fwd_exec_f32( libxsmm_dnn_gn_fwd_config cfg, const float *pinp, const float *pinp_add, const float *pgamma, const float *pbeta, float *mean, float *var, float *pout, unsigned char *prelumask,
                         float eps, int start_tid, int my_tid, void *scratch );

LIBXSMM_API void libxsmm_dnn_gn_fwd_exec_bf16( libxsmm_dnn_gn_fwd_config cfg, const libxsmm_bfloat16 *pinp, const libxsmm_bfloat16 *pinp_add,
                          const float *pgamma, const float *pbeta, float *mean, float *var,
                          libxsmm_bfloat16 *pout, unsigned char *prelumask,
                          float eps, int start_tid, int my_tid, void *scratch );

LIBXSMM_API void libxsmm_dnn_gn_bwd_exec_f32( libxsmm_dnn_gn_bwd_config cfg, float *pdout, const float *pinp, const float *mean, const float *var, const float *pgamma, const unsigned char *prelumask,
                         float *pdin, float *pdin_add, float *pdgamma, float *pdbeta, float eps,
                         int start_tid, int my_tid, void *scratch);

LIBXSMM_API void libxsmm_dnn_gn_bwd_exec_bf16( libxsmm_dnn_gn_bwd_config cfg, libxsmm_bfloat16 *pdout, const libxsmm_bfloat16 *pinp, const float *mean, const float *var, const float *pgamma, const unsigned char *prelumask,
                          libxsmm_bfloat16 *pdin, libxsmm_bfloat16 *pdin_add, float *pdgamma, float *pdbeta, float eps,
                          int start_tid, int my_tid, void *scratch);

#endif

