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

#ifndef LIBXSMM_DNN_SMAX_H
#define LIBXSMM_DNN_SMAX_H

#include <libxsmm.h>
#include <libxsmm_sync.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#if defined(_OPENMP)
# include <omp.h>
#endif

typedef struct libxsmm_dnn_smax_fwd_config {
  libxsmm_blasint N;
  libxsmm_blasint C;
  libxsmm_blasint bn;
  libxsmm_blasint bc;
  libxsmm_blasint threads;
  size_t          scratch_size;
  libxsmm_barrier* barrier;
} libxsmm_dnn_smax_fwd_config;

typedef struct libxsmm_dnn_smax_bwd_config {
  libxsmm_blasint N;
  libxsmm_blasint C;
  libxsmm_blasint bn;
  libxsmm_blasint bc;
  libxsmm_blasint threads;
  size_t          scratch_size;
  float           loss_weight;
  libxsmm_barrier* barrier;
} libxsmm_dnn_smax_bwd_config;

LIBXSMM_API libxsmm_dnn_smax_fwd_config setup_libxsmm_dnn_smax_fwd(libxsmm_blasint N, libxsmm_blasint C, libxsmm_blasint bn, libxsmm_blasint bc,
                                     libxsmm_blasint threads, libxsmm_datatype datatype_in,
                                     libxsmm_datatype datatype_out, libxsmm_datatype datatype_comp);

LIBXSMM_API libxsmm_dnn_smax_bwd_config setup_libxsmm_dnn_smax_bwd(libxsmm_blasint N, libxsmm_blasint C, libxsmm_blasint bn, libxsmm_blasint bc,
                                     libxsmm_blasint threads, float loss_weight, libxsmm_datatype datatype_in,
                                     libxsmm_datatype datatype_out, libxsmm_datatype datatype_comp);

LIBXSMM_API void libxsmm_dnn_smax_fwd_exec_f32( libxsmm_dnn_smax_fwd_config cfg, const float* in_act_ptr, float* out_act_ptr, const int* label_ptr, float* loss, int start_tid, int my_tid, void* scratch );

LIBXSMM_API void libxsmm_dnn_smax_bwd_exec_f32( libxsmm_dnn_smax_bwd_config cfg, float* delin_act_ptr, const float* out_act_ptr, const int* label_ptr, int start_tid, int my_tid, void* scratch );

LIBXSMM_API void libxsmm_dnn_smax_fwd_exec_bf16( libxsmm_dnn_smax_fwd_config cfg, const libxsmm_bfloat16* in_act_ptr, libxsmm_bfloat16* out_act_ptr, const int* label_ptr, float* loss, int start_tid, int my_tid, void* scratch );

LIBXSMM_API void libxsmm_dnn_smax_bwd_exec_bf16( libxsmm_dnn_smax_bwd_config cfg, libxsmm_bfloat16* delin_act_ptr, const libxsmm_bfloat16* out_act_ptr, const int* label_ptr, int start_tid, int my_tid, void* scratch );

#endif

