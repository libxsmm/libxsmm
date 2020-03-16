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
#include "libxsmm_dnn_optimizer_sgd.h"
#include "libxsmm_main.h"

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#include <float.h>
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif


LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_optimizer_sgd_st_f32_f32(libxsmm_dnn_optimizer* handle, int start_thread, int tid);
LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_optimizer_sgd_st_bf16_bf16(libxsmm_dnn_optimizer* handle, int start_thread, int tid);


LIBXSMM_API_INTERN LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512)
libxsmm_dnn_err_t libxsmm_dnn_optimizer_sgd_st_f32_f32(libxsmm_dnn_optimizer* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
#if defined(LIBXSMM_INTRINSICS_AVX512) /*__AVX512F__*/
  typedef float element_filter_type;

# define LIBXSMM_DNN_OPTIMIZER_SGD_F32_AVX512
# include "template/libxsmm_dnn_optimizer_sgd_st_generic.tpl.c"
# undef LIBXSMM_DNN_OPTIMIZER_SGD_F32_AVX512
#else /* should not happen */
  LIBXSMM_UNUSED(handle); LIBXSMM_UNUSED(start_thread); LIBXSMM_UNUSED(tid);
  status = LIBXSMM_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}


LIBXSMM_API_INTERN LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512)
libxsmm_dnn_err_t libxsmm_dnn_optimizer_sgd_st_bf16_bf16(libxsmm_dnn_optimizer* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
#if defined(LIBXSMM_INTRINSICS_AVX512) /*__AVX512F__*/
  typedef libxsmm_bfloat16 element_filter_type;
  typedef float            element_master_type;

# define LIBXSMM_DNN_OPTIMIZER_SGD_BF16_AVX512
# include "template/libxsmm_dnn_optimizer_sgd_st_generic.tpl.c"
# undef LIBXSMM_DNN_OPTIMIZER_SGD_BF16_AVX512
#else /* should not happen */
  LIBXSMM_UNUSED(handle); LIBXSMM_UNUSED(start_thread); LIBXSMM_UNUSED(tid);
  status = LIBXSMM_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}


LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_optimizer_sgd_st(libxsmm_dnn_optimizer* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  /* check if we have filter, grad_filter */
  if ( handle->reg_filter == 0 || handle->grad_filter == 0 ) {
    status = LIBXSMM_DNN_ERR_DATA_NOT_BOUND;
    return status;
  }
  if ( (handle->master_filter == 0) && (handle->desc.datatype == LIBXSMM_DNN_DATATYPE_BF16) ) {
    status = LIBXSMM_DNN_ERR_DATA_NOT_BOUND;
    return status;
  }

  /* check if we are on an AVX512 platform */
#if defined(LIBXSMM_INTRINSICS_AVX512) /*__AVX512F__*/
  if ( libxsmm_target_archid >= LIBXSMM_X86_AVX512 ) {
    if ( handle->desc.datatype == LIBXSMM_DNN_DATATYPE_F32 ) {
      status = libxsmm_dnn_optimizer_sgd_st_f32_f32( handle, start_thread, tid);
    } else if ( handle->desc.datatype == LIBXSMM_DNN_DATATYPE_BF16 ) {
      status = libxsmm_dnn_optimizer_sgd_st_bf16_bf16( handle, start_thread, tid);
    } else {
      status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  } else
#endif
  {
    if ( handle->desc.datatype == LIBXSMM_DNN_DATATYPE_F32 ) {
      typedef float element_filter_type;

# include "template/libxsmm_dnn_optimizer_sgd_st_generic.tpl.c"
    } else if ( handle->desc.datatype == LIBXSMM_DNN_DATATYPE_BF16 ) {
      typedef libxsmm_bfloat16 element_filter_type;
      typedef float            element_master_type;

# define LIBXSMM_DNN_OPTIMIZER_SGD_BF16
# include "template/libxsmm_dnn_optimizer_sgd_st_generic.tpl.c"
# undef LIBXSMM_DNN_OPTIMIZER_SGD_BF16
    } else {
      status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  }

  return status;
}

