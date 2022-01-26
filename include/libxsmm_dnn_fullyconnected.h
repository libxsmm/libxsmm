/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Heinecke, Sasikanth Avancha (Intel Corp.)
******************************************************************************/
#ifndef LIBXSMM_DNN_FULLYCONNECTED_H
#define LIBXSMM_DNN_FULLYCONNECTED_H

#include "libxsmm_dnn.h"
#include "libxsmm_dnn_tensor.h"

/** Opaque handles which represents LIBXSMM fullyconnected */
LIBXSMM_EXTERN_C typedef struct LIBXSMM_RETARGETABLE libxsmm_dnn_fullyconnected libxsmm_dnn_fullyconnected;

typedef enum libxsmm_dnn_fullyconnected_fuse_op {
  /* the fuse order is: 1. BIAS, 2. Actitvation */
  LIBXSMM_DNN_FULLYCONNECTED_FUSE_NONE = 0,
  LIBXSMM_DNN_FULLYCONNECTED_FUSE_BIAS = 1,
  LIBXSMM_DNN_FULLYCONNECTED_FUSE_RELU = 2,
  LIBXSMM_DNN_FULLYCONNECTED_FUSE_SIGMOID = 4,
  LIBXSMM_DNN_FULLYCONNECTED_FUSE_BIAS_RELU = 3,
  LIBXSMM_DNN_FULLYCONNECTED_FUSE_BIAS_SIGMOID = 5
} libxsmm_dnn_fullyconnected_fuse_op;

LIBXSMM_EXTERN_C typedef struct LIBXSMM_RETARGETABLE libxsmm_dnn_fullyconnected_desc {
  int N;                                        /* number of images in mini-batch */
  int C;                                        /* number of input feature maps */
  int K;                                        /* number of output feature maps */
  int bn;
  int bk;
  int bc;
  int threads;                                  /* number of threads used */
  int compressed_A;
  int sparsity_factor_A;
  libxsmm_dnn_datatype datatype_in;             /* datatype used for all input related buffers */
  libxsmm_dnn_datatype datatype_out;            /* datatype used for all output related buffers */
  libxsmm_dnn_tensor_format buffer_format;      /* format which is for activation buffers */
  libxsmm_dnn_tensor_format filter_format;      /* format which is for filter buffers */
  libxsmm_dnn_fullyconnected_fuse_op fuse_ops;  /* fused operations */
} libxsmm_dnn_fullyconnected_desc;

LIBXSMM_API libxsmm_dnn_fullyconnected* libxsmm_dnn_create_fullyconnected(libxsmm_dnn_fullyconnected_desc fullyconnected_desc, libxsmm_dnn_err_t* status);
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_destroy_fullyconnected(const libxsmm_dnn_fullyconnected* handle);

LIBXSMM_API libxsmm_dnn_tensor_datalayout* libxsmm_dnn_fullyconnected_create_tensor_datalayout(const libxsmm_dnn_fullyconnected* handle, const libxsmm_dnn_tensor_type type, libxsmm_dnn_err_t* status);

LIBXSMM_API void*  libxsmm_dnn_fullyconnected_get_scratch_ptr (const libxsmm_dnn_fullyconnected* handle, libxsmm_dnn_err_t* status);
LIBXSMM_API size_t libxsmm_dnn_fullyconnected_get_scratch_size(const libxsmm_dnn_fullyconnected* handle, libxsmm_dnn_err_t* status);
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_fullyconnected_bind_scratch(libxsmm_dnn_fullyconnected* handle, const void* scratch);
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_fullyconnected_release_scratch(libxsmm_dnn_fullyconnected* handle);

LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_fullyconnected_bind_tensor(libxsmm_dnn_fullyconnected* handle, const libxsmm_dnn_tensor* tensor, const libxsmm_dnn_tensor_type type);
LIBXSMM_API libxsmm_dnn_tensor* libxsmm_dnn_fullyconnected_get_tensor(libxsmm_dnn_fullyconnected* handle, const libxsmm_dnn_tensor_type type, libxsmm_dnn_err_t* status);
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_fullyconnected_release_tensor(libxsmm_dnn_fullyconnected* handle, const libxsmm_dnn_tensor_type type);

LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_fullyconnected_execute_st(libxsmm_dnn_fullyconnected* handle, libxsmm_dnn_compute_kind kind,
  /*unsigned*/int start_thread, /*unsigned*/int tid);

#endif /*LIBXSMM_DNN_FULLYCONNECTED_H*/

