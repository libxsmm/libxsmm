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
#ifndef LIBXSMM_DNN_SOFTMAXLOSS_H
#define LIBXSMM_DNN_SOFTMAXLOSS_H

#include "libxsmm_dnn.h"
#include "libxsmm_dnn_tensor.h"

/** Opaque handles which represents LIBXSMM softmaxloss */
LIBXSMM_EXTERN_C typedef struct LIBXSMM_RETARGETABLE libxsmm_dnn_softmaxloss libxsmm_dnn_softmaxloss;

LIBXSMM_EXTERN_C typedef struct LIBXSMM_RETARGETABLE libxsmm_dnn_softmaxloss_desc {
  int N;                                        /* number of images in mini-batch */
  int C;                                        /* number of input feature maps */
  int bn;                                       /* requested N blocking for NCNC format */
  int bc;                                       /* requested C blocking for NCNC format */
  float loss_weight;                            /* loss weight */
  int threads;                                  /* number of threads used */
  libxsmm_dnn_datatype datatype;                /* datatype used for all buffers */
  libxsmm_dnn_tensor_format buffer_format;      /* format which is for activation buffers */
} libxsmm_dnn_softmaxloss_desc;

LIBXSMM_API libxsmm_dnn_softmaxloss* libxsmm_dnn_create_softmaxloss(libxsmm_dnn_softmaxloss_desc softmaxloss_desc, libxsmm_dnn_err_t* status);
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_destroy_softmaxloss(const libxsmm_dnn_softmaxloss* handle);

LIBXSMM_API libxsmm_dnn_tensor_datalayout* libxsmm_dnn_softmaxloss_create_tensor_datalayout(const libxsmm_dnn_softmaxloss* handle, const libxsmm_dnn_tensor_type type, libxsmm_dnn_err_t* status);

LIBXSMM_API void*  libxsmm_dnn_softmaxloss_get_scratch_ptr (const libxsmm_dnn_softmaxloss* handle, libxsmm_dnn_err_t* status);
LIBXSMM_API size_t libxsmm_dnn_softmaxloss_get_scratch_size(const libxsmm_dnn_softmaxloss* handle, libxsmm_dnn_err_t* status);
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_softmaxloss_bind_scratch(libxsmm_dnn_softmaxloss* handle, const void* scratch);
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_softmaxloss_release_scratch(libxsmm_dnn_softmaxloss* handle);

LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_softmaxloss_bind_tensor(libxsmm_dnn_softmaxloss* handle, const libxsmm_dnn_tensor* tensor, const libxsmm_dnn_tensor_type type);
LIBXSMM_API libxsmm_dnn_tensor* libxsmm_dnn_softmaxloss_get_tensor(libxsmm_dnn_softmaxloss* handle, const libxsmm_dnn_tensor_type type, libxsmm_dnn_err_t* status);
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_softmaxloss_release_tensor(libxsmm_dnn_softmaxloss* handle, const libxsmm_dnn_tensor_type type);

LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_softmaxloss_execute_st(libxsmm_dnn_softmaxloss* handle, libxsmm_dnn_compute_kind kind,
  /*unsigned*/int start_thread, /*unsigned*/int tid);

LIBXSMM_API float libxsmm_dnn_softmaxloss_get_loss(const libxsmm_dnn_softmaxloss* handle, libxsmm_dnn_err_t* status);

#endif /*LIBXSMM_DNN_SOFTMAXLOSS_H*/

