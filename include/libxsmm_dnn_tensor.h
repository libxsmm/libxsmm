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
#ifndef LIBXSMM_DNN_TENSOR_H
#define LIBXSMM_DNN_TENSOR_H

#include "libxsmm_typedefs.h"
#include "libxsmm_dnn.h"

/** Opaque handles which represents convolutions and LIBXSMM datatypes */
LIBXSMM_EXTERN_C typedef struct LIBXSMM_RETARGETABLE libxsmm_dnn_tensor libxsmm_dnn_tensor;

typedef enum libxsmm_dnn_tensor_dimtype {
  /** Mini-batch */
  LIBXSMM_DNN_TENSOR_DIMTYPE_N,
  /** Image Height */
  LIBXSMM_DNN_TENSOR_DIMTYPE_H,
  /** Image Width */
  LIBXSMM_DNN_TENSOR_DIMTYPE_W,
  /** channels or input channels */
  LIBXSMM_DNN_TENSOR_DIMTYPE_C,
  /** output channels */
  LIBXSMM_DNN_TENSOR_DIMTYPE_K,
  /** kernel height */
  LIBXSMM_DNN_TENSOR_DIMTYPE_R,
  /** kernel width */
  LIBXSMM_DNN_TENSOR_DIMTYPE_S,
  /** sequence lenth counter */
  LIBXSMM_DNN_TENSOR_DIMTYPE_T,
  /** channle group counter */
  LIBXSMM_DNN_TENSOR_DIMTYPE_G,
  /** general counter */
  LIBXSMM_DNN_TENSOR_DIMTYPE_X
} libxsmm_dnn_tensor_dimtype;

/** types of different buffers */
typedef enum libxsmm_dnn_tensor_type {
  /** regular input buffer */
  LIBXSMM_DNN_REGULAR_INPUT,
  /** regular input buffer */
  LIBXSMM_DNN_REGULAR_INPUT_ADD,
  /** regular input buffer, transpose */
  LIBXSMM_DNN_REGULAR_INPUT_TRANS,
  /** gradient input buffer */
  LIBXSMM_DNN_GRADIENT_INPUT,
  /** gradient input buffer */
  LIBXSMM_DNN_GRADIENT_INPUT_ADD,
  /** regular output buffer */
  LIBXSMM_DNN_REGULAR_OUTPUT,
  /** gradient output buffer */
  LIBXSMM_DNN_GRADIENT_OUTPUT,
  /** general input type */
  LIBXSMM_DNN_INPUT,
  /** general output type */
  LIBXSMM_DNN_OUTPUT,
  /** general activation type */
  LIBXSMM_DNN_ACTIVATION,
  /* regular filter */
  LIBXSMM_DNN_REGULAR_FILTER,
  /* regular filter */
  LIBXSMM_DNN_REGULAR_FILTER_TRANS,
  /* gradient filter */
  LIBXSMM_DNN_GRADIENT_FILTER,
  /* master filter */
  LIBXSMM_DNN_MASTER_FILTER,
  /** general filter type */
  LIBXSMM_DNN_FILTER,
  /* regular bias */
  LIBXSMM_DNN_REGULAR_CHANNEL_BIAS,
  /* gradient bias */
  LIBXSMM_DNN_GRADIENT_CHANNEL_BIAS,
  /* bias */
  LIBXSMM_DNN_CHANNEL_BIAS,
  /* regular beta */
  LIBXSMM_DNN_REGULAR_CHANNEL_BETA,
  /* gradient beta */
  LIBXSMM_DNN_GRADIENT_CHANNEL_BETA,
  /* beta */
  LIBXSMM_DNN_CHANNEL_BETA,
  /* regular gamma */
  LIBXSMM_DNN_REGULAR_CHANNEL_GAMMA,
  /* gradient gamma */
  LIBXSMM_DNN_GRADIENT_CHANNEL_GAMMA,
  /* Gamma */
  LIBXSMM_DNN_CHANNEL_GAMMA,
  /* regular beta */
  LIBXSMM_DNN_CHANNEL_EXPECTVAL,
  /* regular beta */
  LIBXSMM_DNN_CHANNEL_RCPSTDDEV,
  /* variance */
  LIBXSMM_DNN_CHANNEL_VARIANCE,
  /** general bias type */
  LIBXSMM_DNN_CHANNEL_SCALAR,
  /** Labels */
  LIBXSMM_DNN_LABEL,
  /** batch stats */
  LIBXSMM_DNN_BATCH_STATS,
  LIBXSMM_DNN_MAX_STATS_FWD,
  LIBXSMM_DNN_MAX_STATS_BWD,
  LIBXSMM_DNN_MAX_STATS_UPD,
  /** pooling mask */
  LIBXSMM_DNN_POOLING_MASK,
  /** ReLU mask */
  LIBXSMM_DNN_RELU_MASK,
  /** general type, if needed might cause API issues in copy in/out API */
  LIBXSMM_DNN_TENSOR,

  /** regular input buffer */
  LIBXSMM_DNN_RNN_REGULAR_INPUT,
  /** regular previous cell state buffer */
  LIBXSMM_DNN_RNN_REGULAR_CS_PREV,
  /** regular previous hidden state buffer */
  LIBXSMM_DNN_RNN_REGULAR_HIDDEN_STATE_PREV,
  /** regular weight (LSTM: wi, wc, wf, wo) */
  LIBXSMM_DNN_RNN_REGULAR_WEIGHT,
  /** regular recurrent weight (LSTM: ri, rc, rf, ro) */
  LIBXSMM_DNN_RNN_REGULAR_RECUR_WEIGHT,
  /** regular weight (LSTM: wi, wc, wf, wo) */
  LIBXSMM_DNN_RNN_REGULAR_WEIGHT_TRANS,
  /** regular recurrent weight (LSTM: ri, rc, rf, ro) */
  LIBXSMM_DNN_RNN_REGULAR_RECUR_WEIGHT_TRANS,
  /** regular bias (LSTM: bi, bc, bf, bo) */
  LIBXSMM_DNN_RNN_REGULAR_BIAS,
  /** regular output cell state buffer */
  LIBXSMM_DNN_RNN_REGULAR_CS,
  /** regular hidden state buffer */
  LIBXSMM_DNN_RNN_REGULAR_HIDDEN_STATE,
  /** gradient input buffer */
  LIBXSMM_DNN_RNN_GRADIENT_INPUT,
  /** gradient previous cell state buffer */
  LIBXSMM_DNN_RNN_GRADIENT_CS_PREV,
  /** gradient previous hidden state buffer */
  LIBXSMM_DNN_RNN_GRADIENT_HIDDEN_STATE_PREV,
  /** gradient weight */
  LIBXSMM_DNN_RNN_GRADIENT_WEIGHT,
  /** gradient recurrent weight */
  LIBXSMM_DNN_RNN_GRADIENT_RECUR_WEIGHT,
  /** gradient bias */
  LIBXSMM_DNN_RNN_GRADIENT_BIAS,
  /** gradient output cell state buffer */
  LIBXSMM_DNN_RNN_GRADIENT_CS,
  /** gradient hidden state buffer */
  LIBXSMM_DNN_RNN_GRADIENT_HIDDEN_STATE,
  /** internal i buffer */
  LIBXSMM_DNN_RNN_INTERNAL_I,
  /** internal f buffer */
  LIBXSMM_DNN_RNN_INTERNAL_F,
  /** internal o buffer */
  LIBXSMM_DNN_RNN_INTERNAL_O,
  /** internal ci buffer */
  LIBXSMM_DNN_RNN_INTERNAL_CI,
  /** internal co buffer */
  LIBXSMM_DNN_RNN_INTERNAL_CO
} libxsmm_dnn_tensor_type;

/** layout descriptor to allow external data handling
    outside of LIBXSMM */
LIBXSMM_EXTERN_C typedef struct LIBXSMM_RETARGETABLE libxsmm_dnn_tensor_datalayout {
  libxsmm_dnn_tensor_dimtype* dim_type;
  unsigned int* dim_size;
  unsigned int num_dims;
  libxsmm_dnn_tensor_format format;                /* format of activation buffer */
  libxsmm_dnn_datatype datatype;                   /* data type */
  libxsmm_dnn_tensor_type tensor_type;             /* tensor type */
} libxsmm_dnn_tensor_datalayout;

/** tensorlayout handling */
LIBXSMM_API libxsmm_dnn_tensor_datalayout* libxsmm_dnn_duplicate_tensor_datalayout(const libxsmm_dnn_tensor_datalayout* layout, libxsmm_dnn_err_t* status);
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_destroy_tensor_datalayout(libxsmm_dnn_tensor_datalayout* layout);
LIBXSMM_API unsigned int libxsmm_dnn_compare_tensor_datalayout(const libxsmm_dnn_tensor_datalayout* layout_a, const libxsmm_dnn_tensor_datalayout* layout_b, libxsmm_dnn_err_t* status);
LIBXSMM_API unsigned int libxsmm_dnn_get_tensor_size(const libxsmm_dnn_tensor_datalayout* layout, libxsmm_dnn_err_t* status);
LIBXSMM_API unsigned int libxsmm_dnn_get_tensor_elements(const libxsmm_dnn_tensor_datalayout* layout, libxsmm_dnn_err_t* status);

/** Create and manage buffers, filters and bias (non-NULL if successful) */
LIBXSMM_API libxsmm_dnn_tensor* libxsmm_dnn_link_tensor(const libxsmm_dnn_tensor_datalayout* layout, const void* data, libxsmm_dnn_err_t* status);
LIBXSMM_API libxsmm_dnn_tensor* libxsmm_dnn_link_qtensor(const libxsmm_dnn_tensor_datalayout* layout, const void* data, const unsigned char exp, libxsmm_dnn_err_t* status);
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_set_tensor_data_ptr(libxsmm_dnn_tensor* tensor, const void* data);
LIBXSMM_API void* libxsmm_dnn_get_tensor_data_ptr(const libxsmm_dnn_tensor* tensor, libxsmm_dnn_err_t* status);
LIBXSMM_API libxsmm_dnn_tensor_datalayout* libxsmm_dnn_get_tensor_datalayout(const libxsmm_dnn_tensor* tensor, libxsmm_dnn_err_t* status);
LIBXSMM_API unsigned char libxsmm_dnn_get_qtensor_scf(const libxsmm_dnn_tensor* tensor, libxsmm_dnn_err_t* status);
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_set_qtensor_scf(libxsmm_dnn_tensor* tensor, const unsigned char scf);
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_destroy_tensor(const libxsmm_dnn_tensor* tensor);
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_zero_tensor(const libxsmm_dnn_tensor* tensor);

/**
 * Copy-in/out from a plain format such [N][C][H][W] or [N][H][W][C]
 */
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_copyin_tensor(const libxsmm_dnn_tensor* tensor, const void* data, const libxsmm_dnn_tensor_format in_format);
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_copyout_tensor(const libxsmm_dnn_tensor* tensor, void* data, const libxsmm_dnn_tensor_format out_format);

#endif /*LIBXSMM_DNN_TENSOR_H*/

