/******************************************************************************
** Copyright (c) 2017-2018, Intel Corporation                                **
** All rights reserved.                                                      **
**                                                                           **
** Redistribution and use in source and binary forms, with or without        **
** modification, are permitted provided that the following conditions        **
** are met:                                                                  **
** 1. Redistributions of source code must retain the above copyright         **
**    notice, this list of conditions and the following disclaimer.          **
** 2. Redistributions in binary form must reproduce the above copyright      **
**    notice, this list of conditions and the following disclaimer in the    **
**    documentation and/or other materials provided with the distribution.   **
** 3. Neither the name of the copyright holder nor the names of its          **
**    contributors may be used to endorse or promote products derived        **
**    from this software without specific prior written permission.          **
**                                                                           **
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS       **
** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT         **
** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR     **
** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT      **
** HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,    **
** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED  **
** TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR    **
** PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF    **
** LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING      **
** NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS        **
** SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.              **
******************************************************************************/
/* Alexander Heinecke, Sasikanth Avancha (Intel Corp.)
******************************************************************************/
#ifndef LIBXSMM_DNN_FUSEDBN_H
#define LIBXSMM_DNN_FUSEDBN_H

#include "libxsmm_macros.h"
#include "libxsmm_typedefs.h"
#include "libxsmm_dnn.h"

/** Opaque handles which represents LIBXSMM fusedbn */
LIBXSMM_EXTERN_C typedef struct LIBXSMM_RETARGETABLE libxsmm_dnn_fusedbn libxsmm_dnn_fusedbn;

typedef enum libxsmm_dnn_fusedbn_fuse_order {
  /* the fuse order is: 1. BN, 2. eltwise 3. RELU */
  LIBXSMM_DNN_FUSEDBN_ORDER_BN_ELTWISE_RELU = 0
} libxsmm_dnn_fusedbn_fuse_order;

typedef enum libxsmm_dnn_fusedbn_fuse_op {
  /* the fuse order is: 1. BN, 2. eltwise 3. RELU */
  LIBXSMM_DNN_FUSEDBN_OPS_BN = 1,
  LIBXSMM_DNN_FUSEDBN_OPS_BNSCALE = 2,
  LIBXSMM_DNN_FUSEDBN_OPS_ELTWISE = 4,
  LIBXSMM_DNN_FUSEDBN_OPS_RELU = 8,
  LIBXSMM_DNN_FUSEDBN_OPS_BN_ELTWISE = LIBXSMM_DNN_FUSEDBN_OPS_BN | LIBXSMM_DNN_FUSEDBN_OPS_ELTWISE,
  LIBXSMM_DNN_FUSEDBN_OPS_BN_RELU = LIBXSMM_DNN_FUSEDBN_OPS_BN | LIBXSMM_DNN_FUSEDBN_OPS_RELU,
  LIBXSMM_DNN_FUSEDBN_OPS_BN_ELTWISE_RELU = LIBXSMM_DNN_FUSEDBN_OPS_BN | LIBXSMM_DNN_FUSEDBN_OPS_ELTWISE | LIBXSMM_DNN_FUSEDBN_OPS_RELU,
  LIBXSMM_DNN_FUSEDBN_OPS_BNSCALE_ELTWISE = LIBXSMM_DNN_FUSEDBN_OPS_BNSCALE | LIBXSMM_DNN_FUSEDBN_OPS_ELTWISE,
  LIBXSMM_DNN_FUSEDBN_OPS_BNSCALE_RELU = LIBXSMM_DNN_FUSEDBN_OPS_BNSCALE | LIBXSMM_DNN_FUSEDBN_OPS_RELU,
  LIBXSMM_DNN_FUSEDBN_OPS_BNSCALE_ELTWISE_RELU = LIBXSMM_DNN_FUSEDBN_OPS_BNSCALE | LIBXSMM_DNN_FUSEDBN_OPS_ELTWISE | LIBXSMM_DNN_FUSEDBN_OPS_RELU
} libxsmm_dnn_fusedbn_fuse_op;

LIBXSMM_EXTERN_C typedef struct LIBXSMM_RETARGETABLE libxsmm_dnn_fusedbn_desc {
  int N;                                     /* number of images in mini-batch */
  int C;                                     /* number of input feature maps */
  int H;                                     /* height of input image */
  int W;                                     /* width of input image */
  int u;                                     /* vertical stride */
  int v;                                     /* horizontal stride */
  int threads;                               /* number of threads used */
  libxsmm_dnn_datatype datatype_in;          /* datatypes used for all input related buffer */
  libxsmm_dnn_datatype datatype_out;         /* datatypes used for all output related buffer */
  libxsmm_dnn_tensor_format buffer_format;   /* format which is for activation buffers */
  libxsmm_dnn_fusedbn_fuse_order fuse_order; /* additional options */
  libxsmm_dnn_fusedbn_fuse_op fuse_ops;      /* used ops into convolutions */
} libxsmm_dnn_fusedbn_desc;

LIBXSMM_API libxsmm_dnn_fusedbn* libxsmm_dnn_create_fusedbn(libxsmm_dnn_fusedbn_desc fusedbn_desc, libxsmm_dnn_err_t* status);
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_destroy_fusedbn(const libxsmm_dnn_fusedbn* handle);

LIBXSMM_API libxsmm_dnn_tensor_datalayout* libxsmm_dnn_fusedbn_create_tensor_datalayout(const libxsmm_dnn_fusedbn* handle, const libxsmm_dnn_tensor_type type, libxsmm_dnn_err_t* status);

LIBXSMM_API size_t libxsmm_dnn_fusedbn_get_scratch_size(const libxsmm_dnn_fusedbn* handle, libxsmm_dnn_err_t* status);
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_fusedbn_bind_scratch(libxsmm_dnn_fusedbn* handle, const void* scratch);
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_fusedbn_release_scratch(libxsmm_dnn_fusedbn* handle);

LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_fusedbn_bind_tensor(libxsmm_dnn_fusedbn* handle, const libxsmm_dnn_tensor* tensor, const libxsmm_dnn_tensor_type type);
LIBXSMM_API libxsmm_dnn_tensor* libxsmm_dnn_fusedbn_get_tensor(libxsmm_dnn_fusedbn* handle, const libxsmm_dnn_tensor_type type, libxsmm_dnn_err_t* status);
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_fusedbn_release_tensor(libxsmm_dnn_fusedbn* handle, const libxsmm_dnn_tensor_type type);

LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_fusedbn_execute_st(libxsmm_dnn_fusedbn* handle, libxsmm_dnn_compute_kind kind,
  /*unsigned*/int start_thread, /*unsigned*/int tid);

#endif /*LIBXSMM_DNN_FUSEDBN_H*/

