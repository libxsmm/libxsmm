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
#ifndef LIBXSMM_DNN_FUSEDBATCHNORM_H
#define LIBXSMM_DNN_FUSEDBATCHNORM_H

#include "libxsmm_dnn.h"
#include "libxsmm_dnn_tensor.h"

/** Opaque handles which represents LIBXSMM fusedbatchnorm */
LIBXSMM_EXTERN_C typedef struct LIBXSMM_RETARGETABLE libxsmm_dnn_fusedbatchnorm libxsmm_dnn_fusedbatchnorm;

LIBXSMM_API libxsmm_dnn_fusedbatchnorm* libxsmm_dnn_create_fusedbatchnorm(libxsmm_dnn_fusedbatchnorm_desc fusedbatchnorm_desc, libxsmm_dnn_err_t* status);
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_destroy_fusedbatchnorm(const libxsmm_dnn_fusedbatchnorm* handle);

LIBXSMM_API libxsmm_dnn_tensor_datalayout* libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout(const libxsmm_dnn_fusedbatchnorm* handle, const libxsmm_dnn_tensor_type type, libxsmm_dnn_err_t* status);

LIBXSMM_API size_t libxsmm_dnn_fusedbatchnorm_get_scratch_size(const libxsmm_dnn_fusedbatchnorm* handle, libxsmm_dnn_err_t* status);
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_fusedbatchnorm_bind_scratch(libxsmm_dnn_fusedbatchnorm* handle, const void* scratch);
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_fusedbatchnorm_release_scratch(libxsmm_dnn_fusedbatchnorm* handle);

LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_fusedbatchnorm_bind_tensor(libxsmm_dnn_fusedbatchnorm* handle, const libxsmm_dnn_tensor* tensor, const libxsmm_dnn_tensor_type type);
LIBXSMM_API libxsmm_dnn_tensor* libxsmm_dnn_fusedbatchnorm_get_tensor(libxsmm_dnn_fusedbatchnorm* handle, const libxsmm_dnn_tensor_type type, libxsmm_dnn_err_t* status);
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_fusedbatchnorm_release_tensor(libxsmm_dnn_fusedbatchnorm* handle, const libxsmm_dnn_tensor_type type);

LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_fusedbatchnorm_execute_st(libxsmm_dnn_fusedbatchnorm* handle, libxsmm_dnn_compute_kind kind,
  /*unsigned*/int start_thread, /*unsigned*/int tid);
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_fusedbatchnorm_reduce_stats_st(libxsmm_dnn_fusedbatchnorm** handles, int num_handles, libxsmm_dnn_compute_kind kind,
  /*unsigned*/int start_thread, /*unsigned*/int tid);

#endif /*LIBXSMM_DNN_FUSEDBATCHNORM_H*/

