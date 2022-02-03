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
#ifndef LIBXSMM_DNN_FUSEDGROUPNORM_H
#define LIBXSMM_DNN_FUSEDGROUPNORM_H

#include "libxsmm_dnn.h"
#include "libxsmm_dnn_tensor.h"

/** Opaque handles which represents LIBXSMM fusedgroupnorm */
LIBXSMM_EXTERN_C typedef struct LIBXSMM_RETARGETABLE libxsmm_dnn_fusedgroupnorm libxsmm_dnn_fusedgroupnorm;

LIBXSMM_API libxsmm_dnn_fusedgroupnorm* libxsmm_dnn_create_fusedgroupnorm(libxsmm_dnn_fusedgroupnorm_desc fusedgroupnorm_desc, libxsmm_dnn_err_t* status);
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_destroy_fusedgroupnorm(const libxsmm_dnn_fusedgroupnorm* handle);

LIBXSMM_API libxsmm_dnn_tensor_datalayout* libxsmm_dnn_fusedgroupnorm_create_tensor_datalayout(const libxsmm_dnn_fusedgroupnorm* handle, const libxsmm_dnn_tensor_type type, libxsmm_dnn_err_t* status);

LIBXSMM_API size_t libxsmm_dnn_fusedgroupnorm_get_scratch_size(const libxsmm_dnn_fusedgroupnorm* handle, libxsmm_dnn_err_t* status);
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_fusedgroupnorm_bind_scratch(libxsmm_dnn_fusedgroupnorm* handle, const void* scratch);
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_fusedgroupnorm_release_scratch(libxsmm_dnn_fusedgroupnorm* handle);

LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_fusedgroupnorm_bind_tensor(libxsmm_dnn_fusedgroupnorm* handle, const libxsmm_dnn_tensor* tensor, const libxsmm_dnn_tensor_type type);
LIBXSMM_API libxsmm_dnn_tensor* libxsmm_dnn_fusedgroupnorm_get_tensor(libxsmm_dnn_fusedgroupnorm* handle, const libxsmm_dnn_tensor_type type, libxsmm_dnn_err_t* status);
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_fusedgroupnorm_release_tensor(libxsmm_dnn_fusedgroupnorm* handle, const libxsmm_dnn_tensor_type type);

LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_fusedgroupnorm_execute_st(libxsmm_dnn_fusedgroupnorm* handle, libxsmm_dnn_compute_kind kind,
  /*unsigned*/int start_thread, /*unsigned*/int tid);
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_fusedgroupnorm_reduce_stats_st(libxsmm_dnn_fusedgroupnorm** handles, int num_handles, libxsmm_dnn_compute_kind kind,
  /*unsigned*/int start_thread, /*unsigned*/int tid);

#endif /*LIBXSMM_DNN_FUSEDGROUPNORM_H*/

