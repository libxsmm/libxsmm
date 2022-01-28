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
#ifndef LIBXSMM_DNN_POOLING_H
#define LIBXSMM_DNN_POOLING_H

#include "libxsmm_dnn.h"
#include "libxsmm_dnn_tensor.h"

/** Opaque handles which represents LIBXSMM pooling */
LIBXSMM_EXTERN_C typedef struct LIBXSMM_RETARGETABLE libxsmm_dnn_pooling libxsmm_dnn_pooling;

typedef enum libxsmm_dnn_pooling_type {
  LIBXSMM_DNN_POOLING_MAX = 1,
  LIBXSMM_DNN_POOLING_AVG = 2
} libxsmm_dnn_pooling_type;

LIBXSMM_EXTERN_C typedef struct LIBXSMM_RETARGETABLE libxsmm_dnn_pooling_desc {
  int N;                                     /* number of images in mini-batch */
  int C;                                     /* number of input feature maps */
  int H;                                     /* height of input image */
  int W;                                     /* width of input image */
  int R;                                     /* kernel height */
  int S;                                     /* kernel width */
  int u;                                     /* vertical stride */
  int v;                                     /* horizontal stride */
  int pad_h;                                 /* height of logical padding of input buffer */
  int pad_w;                                 /* width of logical padding of input buffer */
  int pad_h_in;                              /* height of physical zero-padding in input buffer */
  int pad_w_in;                              /* width of physical zero-padding in input buffer */
  int pad_h_out;                             /* height of physical zero-padding in output buffer */
  int pad_w_out;                             /* width of physical zero-padding in output buffer */
  int threads;                               /* number of threads used */
  libxsmm_dnn_datatype datatype_in;          /* datatypes used for all input related buffer */
  libxsmm_dnn_datatype datatype_out;         /* datatypes used for all output related buffer */
  libxsmm_dnn_datatype datatype_mask;        /* datatypes used for the masks */
  libxsmm_dnn_tensor_format buffer_format;   /* format which is for activation buffers */
  libxsmm_dnn_pooling_type pooling_type;     /* type of pooling operation */
} libxsmm_dnn_pooling_desc;

LIBXSMM_API libxsmm_dnn_pooling* libxsmm_dnn_create_pooling(libxsmm_dnn_pooling_desc pooling_desc, libxsmm_dnn_err_t* status);
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_destroy_pooling(const libxsmm_dnn_pooling* handle);

LIBXSMM_API libxsmm_dnn_tensor_datalayout* libxsmm_dnn_pooling_create_tensor_datalayout(const libxsmm_dnn_pooling* handle, const libxsmm_dnn_tensor_type type, libxsmm_dnn_err_t* status);

LIBXSMM_API size_t libxsmm_dnn_pooling_get_scratch_size(const libxsmm_dnn_pooling* handle, libxsmm_dnn_err_t* status);
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_pooling_bind_scratch(libxsmm_dnn_pooling* handle, const void* scratch);
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_pooling_release_scratch(libxsmm_dnn_pooling* handle);

LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_pooling_bind_tensor(libxsmm_dnn_pooling* handle, const libxsmm_dnn_tensor* tensor, const libxsmm_dnn_tensor_type type);
LIBXSMM_API libxsmm_dnn_tensor* libxsmm_dnn_pooling_get_tensor(libxsmm_dnn_pooling* handle, const libxsmm_dnn_tensor_type type, libxsmm_dnn_err_t* status);
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_pooling_release_tensor(libxsmm_dnn_pooling* handle, const libxsmm_dnn_tensor_type type);

LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_pooling_execute_st(libxsmm_dnn_pooling* handle, libxsmm_dnn_compute_kind kind,
  /*unsigned*/int start_thread, /*unsigned*/int tid);

#endif /*LIBXSMM_DNN_POOLING_H*/

