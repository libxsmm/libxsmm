/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Heinecke, Sasikanth Avancha (Intel Corp.)
******************************************************************************/
#ifndef LIBXSMM_DNN_SGD_H
#define LIBXSMM_DNN_SGD_H

#include "libxsmm_dnn.h"
#include "libxsmm_dnn_tensor.h"

/** Opaque handles which represents LIBXSMM optimizer */
LIBXSMM_EXTERN_C typedef struct LIBXSMM_RETARGETABLE libxsmm_dnn_optimizer libxsmm_dnn_optimizer;

typedef enum libxsmm_dnn_optimizer_type {
  LIBXSMM_DNN_OPTIMIZER_SGD = 1
} libxsmm_dnn_optimizer_type;


LIBXSMM_EXTERN_C typedef struct LIBXSMM_RETARGETABLE libxsmm_dnn_optimizer_desc {
  int C;                                        /* number of feature maps */
  int K;                                        /* number of feature maps */
  int bc;
  int bk;
  float learning_rate;                          /* learning rate */
  int threads;                                  /* number of threads used */
  libxsmm_dnn_optimizer_type opt_type;
  libxsmm_dnn_datatype datatype_master;         /* datatype used for all input related buffers */
  libxsmm_dnn_datatype datatype;                /* datatype used for all input related buffers */
  libxsmm_dnn_tensor_format filter_format;      /* format which is for filter buffers */
} libxsmm_dnn_optimizer_desc;

LIBXSMM_API libxsmm_dnn_optimizer* libxsmm_dnn_create_optimizer(libxsmm_dnn_optimizer_desc optimizer_desc, libxsmm_dnn_err_t* status);
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_destroy_optimizer(const libxsmm_dnn_optimizer* handle);

LIBXSMM_API libxsmm_dnn_tensor_datalayout* libxsmm_dnn_optimizer_create_tensor_datalayout(const libxsmm_dnn_optimizer* handle, const libxsmm_dnn_tensor_type type, libxsmm_dnn_err_t* status);

LIBXSMM_API void*  libxsmm_dnn_optimizer_get_scratch_ptr (const libxsmm_dnn_optimizer* handle, libxsmm_dnn_err_t* status);
LIBXSMM_API size_t libxsmm_dnn_optimizer_get_scratch_size(const libxsmm_dnn_optimizer* handle, libxsmm_dnn_err_t* status);
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_optimizer_bind_scratch(libxsmm_dnn_optimizer* handle, const void* scratch);
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_optimizer_release_scratch(libxsmm_dnn_optimizer* handle);

LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_optimizer_bind_tensor(libxsmm_dnn_optimizer* handle, const libxsmm_dnn_tensor* tensor, const libxsmm_dnn_tensor_type type);
LIBXSMM_API libxsmm_dnn_tensor* libxsmm_dnn_optimizer_get_tensor(libxsmm_dnn_optimizer* handle, const libxsmm_dnn_tensor_type type, libxsmm_dnn_err_t* status);
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_optimizer_release_tensor(libxsmm_dnn_optimizer* handle, const libxsmm_dnn_tensor_type type);

LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_optimizer_execute_st(libxsmm_dnn_optimizer* handle, /*unsigned*/int start_thread, /*unsigned*/int tid);

#endif /*LIBXSMM_DNN_SGD_H*/

