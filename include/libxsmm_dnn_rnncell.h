/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Heinecke, Kunal Banerjee (Intel Corp.)
******************************************************************************/
#ifndef LIBXSMM_DNN_RNNCELL_H
#define LIBXSMM_DNN_RNNCELL_H

#include "libxsmm_dnn.h"
#include "libxsmm_dnn_tensor.h"

LIBXSMM_EXTERN_C typedef struct LIBXSMM_RETARGETABLE libxsmm_dnn_rnncell libxsmm_dnn_rnncell;

/** Type of algorithm used for convolutions. */
typedef enum libxsmm_dnn_rnncell_type {
  /** simple RNN cell with ReLU as activation function */
  LIBXSMM_DNN_RNNCELL_RNN_RELU,
  /** simple RNN cell with sigmoid as activation function */
  LIBXSMM_DNN_RNNCELL_RNN_SIGMOID,
  /** simple RNN cell with tanh as activation function */
  LIBXSMM_DNN_RNNCELL_RNN_TANH,
  /** LSTM cell */
  LIBXSMM_DNN_RNNCELL_LSTM,
  /** GRU cell */
  LIBXSMM_DNN_RNNCELL_GRU
} libxsmm_dnn_rnncell_type;

LIBXSMM_EXTERN_C typedef struct LIBXSMM_RETARGETABLE libxsmm_dnn_rnncell_desc {
  int threads;
  libxsmm_blasint K;         /* number of outputs */
  libxsmm_blasint N;         /* size of the minibatch */
  libxsmm_blasint C;         /* number of inputs */
  libxsmm_blasint max_T;     /* number of time steps */
  libxsmm_blasint bk;
  libxsmm_blasint bn;
  libxsmm_blasint bc;
  int use_fwd_fused_impl;
  int fwd_block;
  int bwdupd_block;
  libxsmm_dnn_rnncell_type cell_type;       /* cell type RNN ReLU, RNN Sigmoid, RNN Tanh, LSTM, GRU */
  libxsmm_dnn_datatype datatype_in;         /* datatypes used for all input related buffer */
  libxsmm_dnn_datatype datatype_out;        /* datatypes used for all output related buffer */
  libxsmm_dnn_tensor_format buffer_format;  /* format which is for activation buffers */
  libxsmm_dnn_tensor_format filter_format;  /* format which is for filter buffers */
} libxsmm_dnn_rnncell_desc;

LIBXSMM_API libxsmm_dnn_rnncell* libxsmm_dnn_create_rnncell(libxsmm_dnn_rnncell_desc rnncell_desc, libxsmm_dnn_err_t* status);
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_destroy_rnncell(const libxsmm_dnn_rnncell* handle);

LIBXSMM_API libxsmm_dnn_tensor_datalayout* libxsmm_dnn_rnncell_create_tensor_datalayout(const libxsmm_dnn_rnncell* handle, const libxsmm_dnn_tensor_type type, libxsmm_dnn_err_t* status);

LIBXSMM_API size_t libxsmm_dnn_rnncell_get_scratch_size(const libxsmm_dnn_rnncell* handle, const libxsmm_dnn_compute_kind kind, libxsmm_dnn_err_t* status);
LIBXSMM_API void*  libxsmm_dnn_rnncell_get_scratch_ptr (const libxsmm_dnn_rnncell* handle, libxsmm_dnn_err_t* status);
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_rnncell_bind_scratch(libxsmm_dnn_rnncell* handle, const libxsmm_dnn_compute_kind kind, const void* scratch);
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_rnncell_release_scratch(libxsmm_dnn_rnncell* handle, const libxsmm_dnn_compute_kind kind);

LIBXSMM_API size_t libxsmm_dnn_rnncell_get_internalstate_size(const libxsmm_dnn_rnncell* handle, const libxsmm_dnn_compute_kind kind, libxsmm_dnn_err_t* status);
LIBXSMM_API void*  libxsmm_dnn_rnncell_get_internalstate_ptr (const libxsmm_dnn_rnncell* handle, libxsmm_dnn_err_t* status);
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_rnncell_bind_internalstate(libxsmm_dnn_rnncell* handle, const libxsmm_dnn_compute_kind kind, const void* internalstate);
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_rnncell_release_internalstate(libxsmm_dnn_rnncell* handle, const libxsmm_dnn_compute_kind kind);

LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_rnncell_allocate_forget_bias(libxsmm_dnn_rnncell* handle, const float forget_bias);
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_rnncell_bind_tensor(libxsmm_dnn_rnncell* handle, const libxsmm_dnn_tensor* tensor, const libxsmm_dnn_tensor_type type);
LIBXSMM_API libxsmm_dnn_tensor* libxsmm_dnn_rnncell_get_tensor(libxsmm_dnn_rnncell* handle, const libxsmm_dnn_tensor_type type, libxsmm_dnn_err_t* status);
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_rnncell_release_tensor(libxsmm_dnn_rnncell* handle, const libxsmm_dnn_tensor_type type);

LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_rnncell_set_sequence_length( libxsmm_dnn_rnncell* handle, const libxsmm_blasint T );
LIBXSMM_API libxsmm_blasint libxsmm_dnn_rnncell_get_sequence_length( libxsmm_dnn_rnncell* handle, libxsmm_dnn_err_t* status );

LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_rnncell_execute_st(libxsmm_dnn_rnncell* handle, libxsmm_dnn_compute_kind kind,
  /*unsigned*/int start_thread, /*unsigned*/int tid);

#endif /*LIBXSMM_DNN_RNNCELL_H*/

