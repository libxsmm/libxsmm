/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Heinecke (Intel Corp.)
******************************************************************************/
#include "libxsmm_dnn_fusedgroupnorm_forward.h"
#include "libxsmm_main.h"

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#include <math.h>
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

#if 0
LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_fusedgroupnorm_st_fwd_custom_f32_f32_c16(libxsmm_dnn_fusedgroupnorm* handle, int start_thread, int tid);
LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_fusedgroupnorm_st_fwd_custom_f32_f32_c32(libxsmm_dnn_fusedgroupnorm* handle, int start_thread, int tid);
LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_fusedgroupnorm_st_fwd_custom_f32_f32_c64(libxsmm_dnn_fusedgroupnorm* handle, int start_thread, int tid);
LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_fusedgroupnorm_st_fwd_custom_bf16_bf16_c16(libxsmm_dnn_fusedgroupnorm* handle, int start_thread, int tid);
LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_fusedgroupnorm_st_fwd_custom_bf16_bf16_c32(libxsmm_dnn_fusedgroupnorm* handle, int start_thread, int tid);
LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_fusedgroupnorm_st_fwd_custom_bf16_bf16_c64(libxsmm_dnn_fusedgroupnorm* handle, int start_thread, int tid);


LIBXSMM_API_INTERN LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512)
libxsmm_dnn_err_t libxsmm_dnn_fusedgroupnorm_st_fwd_custom_f32_f32_c16(libxsmm_dnn_fusedgroupnorm* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
#if defined(LIBXSMM_INTRINSICS_AVX512) /*__AVX512F__*/
  typedef float element_input_type;
  typedef float element_output_type;
  typedef float element_stats_type;

  if ( handle->desc.fuse_order != LIBXSMM_DNN_FUSEDGN_ORDER_GN_ELTWISE_RELU ) {
    status = LIBXSMM_DNN_ERR_FUSEDGN_UNSUPPORTED_ORDER;
  } else {
    if ( (handle->desc.fuse_ops == LIBXSMM_DNN_FUSEDGN_OPS_GN) ) {
# include "template/libxsmm_dnn_fusedgroupnorm_st_fwd_custom_f32_bf16_c16_avx512.tpl.c"
    } else if ( (handle->desc.fuse_ops & LIBXSMM_DNN_FUSEDGN_OPS_ELTWISE_RELU) > 0 ) {
# define LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_ELTWISE
# define LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_RELU
# include "template/libxsmm_dnn_fusedgroupnorm_st_fwd_custom_f32_bf16_c16_avx512.tpl.c"
# undef LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_RELU
# undef LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_ELTWISE
    } else if ( (handle->desc.fuse_ops & LIBXSMM_DNN_FUSEDGN_OPS_ELTWISE_RELU_WITH_MASK) > 0 ) {
# define LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_ELTWISE
# define LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_RELU_WITH_MASK
# include "template/libxsmm_dnn_fusedgroupnorm_st_fwd_custom_f32_bf16_c16_avx512.tpl.c"
# undef LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_RELU_WITH_MASK
# undef LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_ELTWISE
    } else if ( (handle->desc.fuse_ops & LIBXSMM_DNN_FUSEDGN_OPS_ELTWISE) > 0 ) {
# define LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_ELTWISE
# include "template/libxsmm_dnn_fusedgroupnorm_st_fwd_custom_f32_bf16_c16_avx512.tpl.c"
# undef LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_ELTWISE
    } else if ( (handle->desc.fuse_ops & LIBXSMM_DNN_FUSEDGN_OPS_RELU) > 0 ) {
# define LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_RELU
# include "template/libxsmm_dnn_fusedgroupnorm_st_fwd_custom_f32_bf16_c16_avx512.tpl.c"
# undef LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_RELU
    } else if ( (handle->desc.fuse_ops & LIBXSMM_DNN_FUSEDGN_OPS_RELU_WITH_MASK) > 0 ) {
# define LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_RELU_WITH_MASK
# include "template/libxsmm_dnn_fusedgroupnorm_st_fwd_custom_f32_bf16_c16_avx512.tpl.c"
# undef LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_RELU_WITH_MASK
    } else {
      status = LIBXSMM_DNN_ERR_FUSEDGN_UNSUPPORTED_FUSION;
    }
  }
#else /* should not happen */
  LIBXSMM_UNUSED(handle); LIBXSMM_UNUSED(start_thread); LIBXSMM_UNUSED(tid);
  status = LIBXSMM_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}


LIBXSMM_API_INTERN LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512)
libxsmm_dnn_err_t libxsmm_dnn_fusedgroupnorm_st_fwd_custom_f32_f32_c32(libxsmm_dnn_fusedgroupnorm* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
#if defined(LIBXSMM_INTRINSICS_AVX512) /*__AVX512F__*/
  typedef float element_input_type;
  typedef float element_output_type;
  typedef float element_stats_type;

  if ( handle->desc.fuse_order != LIBXSMM_DNN_FUSEDGN_ORDER_GN_ELTWISE_RELU ) {
    status = LIBXSMM_DNN_ERR_FUSEDGN_UNSUPPORTED_ORDER;
  } else {
    if ( (handle->desc.fuse_ops == LIBXSMM_DNN_FUSEDGN_OPS_GN) ) {
# include "template/libxsmm_dnn_fusedgroupnorm_st_fwd_custom_f32_bf16_c32_avx512.tpl.c"
    } else if ( (handle->desc.fuse_ops & LIBXSMM_DNN_FUSEDGN_OPS_ELTWISE_RELU) > 0 ) {
# define LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_ELTWISE
# define LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_RELU
# include "template/libxsmm_dnn_fusedgroupnorm_st_fwd_custom_f32_bf16_c32_avx512.tpl.c"
# undef LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_RELU
# undef LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_ELTWISE
    } else if ( (handle->desc.fuse_ops & LIBXSMM_DNN_FUSEDGN_OPS_ELTWISE_RELU_WITH_MASK) > 0 ) {
# define LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_ELTWISE
# define LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_RELU_WITH_MASK
# include "template/libxsmm_dnn_fusedgroupnorm_st_fwd_custom_f32_bf16_c32_avx512.tpl.c"
# undef LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_RELU_WITH_MASK
# undef LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_ELTWISE
    } else if ( (handle->desc.fuse_ops & LIBXSMM_DNN_FUSEDGN_OPS_ELTWISE) > 0 ) {
# define LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_ELTWISE
# include "template/libxsmm_dnn_fusedgroupnorm_st_fwd_custom_f32_bf16_c32_avx512.tpl.c"
# undef LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_ELTWISE
    } else if ( (handle->desc.fuse_ops & LIBXSMM_DNN_FUSEDGN_OPS_RELU) > 0 ) {
# define LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_RELU
# include "template/libxsmm_dnn_fusedgroupnorm_st_fwd_custom_f32_bf16_c32_avx512.tpl.c"
# undef LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_RELU
    } else if ( (handle->desc.fuse_ops & LIBXSMM_DNN_FUSEDGN_OPS_RELU_WITH_MASK) > 0 ) {
# define LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_RELU_WITH_MASK
# include "template/libxsmm_dnn_fusedgroupnorm_st_fwd_custom_f32_bf16_c32_avx512.tpl.c"
# undef LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_RELU_WITH_MASK
    } else {
      status = LIBXSMM_DNN_ERR_FUSEDGN_UNSUPPORTED_FUSION;
    }
  }
#else /* should not happen */
  LIBXSMM_UNUSED(handle); LIBXSMM_UNUSED(start_thread); LIBXSMM_UNUSED(tid);
  status = LIBXSMM_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}


LIBXSMM_API_INTERN LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512)
libxsmm_dnn_err_t libxsmm_dnn_fusedgroupnorm_st_fwd_custom_f32_f32_c64(libxsmm_dnn_fusedgroupnorm* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
#if defined(LIBXSMM_INTRINSICS_AVX512) /*__AVX512F__*/
  typedef float element_input_type;
  typedef float element_output_type;
  typedef float element_stats_type;

  if ( handle->desc.fuse_order != LIBXSMM_DNN_FUSEDGN_ORDER_GN_ELTWISE_RELU ) {
    status = LIBXSMM_DNN_ERR_FUSEDGN_UNSUPPORTED_ORDER;
  } else {
    if ( (handle->desc.fuse_ops == LIBXSMM_DNN_FUSEDGN_OPS_GN) ) {
# include "template/libxsmm_dnn_fusedgroupnorm_st_fwd_custom_f32_bf16_c64_avx512.tpl.c"
    } else if ( (handle->desc.fuse_ops & LIBXSMM_DNN_FUSEDGN_OPS_ELTWISE_RELU) > 0 ) {
# define LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_ELTWISE
# define LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_RELU
# include "template/libxsmm_dnn_fusedgroupnorm_st_fwd_custom_f32_bf16_c64_avx512.tpl.c"
# undef LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_RELU
# undef LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_ELTWISE
    } else if ( (handle->desc.fuse_ops & LIBXSMM_DNN_FUSEDGN_OPS_ELTWISE_RELU_WITH_MASK) > 0 ) {
# define LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_ELTWISE
# define LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_RELU_WITH_MASK
# include "template/libxsmm_dnn_fusedgroupnorm_st_fwd_custom_f32_bf16_c64_avx512.tpl.c"
# undef LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_RELU_WITH_MASK
# undef LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_ELTWISE
    } else if ( (handle->desc.fuse_ops & LIBXSMM_DNN_FUSEDGN_OPS_ELTWISE) > 0 ) {
# define LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_ELTWISE
# include "template/libxsmm_dnn_fusedgroupnorm_st_fwd_custom_f32_bf16_c64_avx512.tpl.c"
# undef LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_ELTWISE
    } else if ( (handle->desc.fuse_ops & LIBXSMM_DNN_FUSEDGN_OPS_RELU) > 0 ) {
# define LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_RELU
# include "template/libxsmm_dnn_fusedgroupnorm_st_fwd_custom_f32_bf16_c64_avx512.tpl.c"
# undef LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_RELU
    } else if ( (handle->desc.fuse_ops & LIBXSMM_DNN_FUSEDGN_OPS_RELU_WITH_MASK) > 0 ) {
# define LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_RELU_WITH_MASK
# include "template/libxsmm_dnn_fusedgroupnorm_st_fwd_custom_f32_bf16_c64_avx512.tpl.c"
# undef LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_RELU_WITH_MASK
    } else {
      status = LIBXSMM_DNN_ERR_FUSEDGN_UNSUPPORTED_FUSION;
    }
  }
#else /* should not happen */
  LIBXSMM_UNUSED(handle); LIBXSMM_UNUSED(start_thread); LIBXSMM_UNUSED(tid);
  status = LIBXSMM_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}


LIBXSMM_API_INTERN LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512)
libxsmm_dnn_err_t libxsmm_dnn_fusedgroupnorm_st_fwd_custom_bf16_bf16_c16(libxsmm_dnn_fusedgroupnorm* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
#if defined(LIBXSMM_INTRINSICS_AVX512) /*__AVX512F__*/
  typedef libxsmm_bfloat16 element_input_type;
  typedef libxsmm_bfloat16 element_output_type;
  typedef float element_stats_type;

# define LIBXSMM_DNN_FUSEDGN_FWD_BF16
  if ( handle->desc.fuse_order != LIBXSMM_DNN_FUSEDGN_ORDER_GN_ELTWISE_RELU ) {
    status = LIBXSMM_DNN_ERR_FUSEDGN_UNSUPPORTED_ORDER;
  } else {
    if ( (handle->desc.fuse_ops == LIBXSMM_DNN_FUSEDGN_OPS_GN) ) {
# include "template/libxsmm_dnn_fusedgroupnorm_st_fwd_custom_f32_bf16_c16_avx512.tpl.c"
    } else if ( (handle->desc.fuse_ops & LIBXSMM_DNN_FUSEDGN_OPS_ELTWISE_RELU) > 0 ) {
# define LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_ELTWISE
# define LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_RELU
# include "template/libxsmm_dnn_fusedgroupnorm_st_fwd_custom_f32_bf16_c16_avx512.tpl.c"
# undef LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_RELU
# undef LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_ELTWISE
    } else if ( (handle->desc.fuse_ops & LIBXSMM_DNN_FUSEDGN_OPS_ELTWISE_RELU_WITH_MASK) > 0 ) {
# define LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_ELTWISE
# define LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_RELU_WITH_MASK
# include "template/libxsmm_dnn_fusedgroupnorm_st_fwd_custom_f32_bf16_c16_avx512.tpl.c"
# undef LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_RELU_WITH_MASK
# undef LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_ELTWISE
    } else if ( (handle->desc.fuse_ops & LIBXSMM_DNN_FUSEDGN_OPS_ELTWISE) > 0 ) {
# define LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_ELTWISE
# include "template/libxsmm_dnn_fusedgroupnorm_st_fwd_custom_f32_bf16_c16_avx512.tpl.c"
# undef LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_ELTWISE
    } else if ( (handle->desc.fuse_ops & LIBXSMM_DNN_FUSEDGN_OPS_RELU) > 0 ) {
# define LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_RELU
# include "template/libxsmm_dnn_fusedgroupnorm_st_fwd_custom_f32_bf16_c16_avx512.tpl.c"
# undef LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_RELU
    } else if ( (handle->desc.fuse_ops & LIBXSMM_DNN_FUSEDGN_OPS_RELU_WITH_MASK) > 0 ) {
# define LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_RELU_WITH_MASK
# include "template/libxsmm_dnn_fusedgroupnorm_st_fwd_custom_f32_bf16_c16_avx512.tpl.c"
# undef LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_RELU_WITH_MASK
    } else {
      status = LIBXSMM_DNN_ERR_FUSEDGN_UNSUPPORTED_FUSION;
    }
  }
# undef LIBXSMM_DNN_FUSEDGN_FWD_BF16
#else /* should not happen */
  LIBXSMM_UNUSED(handle); LIBXSMM_UNUSED(start_thread); LIBXSMM_UNUSED(tid);
  status = LIBXSMM_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}


LIBXSMM_API_INTERN LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512)
libxsmm_dnn_err_t libxsmm_dnn_fusedgroupnorm_st_fwd_custom_bf16_bf16_c32(libxsmm_dnn_fusedgroupnorm* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
#if defined(LIBXSMM_INTRINSICS_AVX512) /*__AVX512F__*/
  typedef libxsmm_bfloat16 element_input_type;
  typedef libxsmm_bfloat16 element_output_type;
  typedef float element_stats_type;

# define LIBXSMM_DNN_FUSEDGN_FWD_BF16
  if ( handle->desc.fuse_order != LIBXSMM_DNN_FUSEDGN_ORDER_GN_ELTWISE_RELU ) {
    status = LIBXSMM_DNN_ERR_FUSEDGN_UNSUPPORTED_ORDER;
  } else {
    if ( (handle->desc.fuse_ops == LIBXSMM_DNN_FUSEDGN_OPS_GN) ) {
# include "template/libxsmm_dnn_fusedgroupnorm_st_fwd_custom_f32_bf16_c32_avx512.tpl.c"
    } else if ( (handle->desc.fuse_ops & LIBXSMM_DNN_FUSEDGN_OPS_ELTWISE_RELU) > 0 ) {
# define LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_ELTWISE
# define LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_RELU
# include "template/libxsmm_dnn_fusedgroupnorm_st_fwd_custom_f32_bf16_c32_avx512.tpl.c"
# undef LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_RELU
# undef LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_ELTWISE
    } else if ( (handle->desc.fuse_ops & LIBXSMM_DNN_FUSEDGN_OPS_ELTWISE_RELU_WITH_MASK) > 0 ) {
# define LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_ELTWISE
# define LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_RELU_WITH_MASK
# include "template/libxsmm_dnn_fusedgroupnorm_st_fwd_custom_f32_bf16_c32_avx512.tpl.c"
# undef LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_RELU_WITH_MASK
# undef LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_ELTWISE
    } else if ( (handle->desc.fuse_ops & LIBXSMM_DNN_FUSEDGN_OPS_ELTWISE) > 0 ) {
# define LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_ELTWISE
# include "template/libxsmm_dnn_fusedgroupnorm_st_fwd_custom_f32_bf16_c32_avx512.tpl.c"
# undef LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_ELTWISE
    } else if ( (handle->desc.fuse_ops & LIBXSMM_DNN_FUSEDGN_OPS_RELU) > 0 ) {
# define LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_RELU
# include "template/libxsmm_dnn_fusedgroupnorm_st_fwd_custom_f32_bf16_c32_avx512.tpl.c"
# undef LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_RELU
    } else if ( (handle->desc.fuse_ops & LIBXSMM_DNN_FUSEDGN_OPS_RELU_WITH_MASK) > 0 ) {
# define LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_RELU_WITH_MASK
# include "template/libxsmm_dnn_fusedgroupnorm_st_fwd_custom_f32_bf16_c32_avx512.tpl.c"
# undef LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_RELU_WITH_MASK
    } else {
      status = LIBXSMM_DNN_ERR_FUSEDGN_UNSUPPORTED_FUSION;
    }
  }
# undef LIBXSMM_DNN_FUSEDGN_FWD_BF16
#else /* should not happen */
  LIBXSMM_UNUSED(handle); LIBXSMM_UNUSED(start_thread); LIBXSMM_UNUSED(tid);
  status = LIBXSMM_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}


LIBXSMM_API_INTERN LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512)
libxsmm_dnn_err_t libxsmm_dnn_fusedgroupnorm_st_fwd_custom_bf16_bf16_c64(libxsmm_dnn_fusedgroupnorm* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
#if defined(LIBXSMM_INTRINSICS_AVX512) /*__AVX512F__*/
  typedef libxsmm_bfloat16 element_input_type;
  typedef libxsmm_bfloat16 element_output_type;
  typedef float element_stats_type;

# define LIBXSMM_DNN_FUSEDGN_FWD_BF16
  if ( handle->desc.fuse_order != LIBXSMM_DNN_FUSEDGN_ORDER_GN_ELTWISE_RELU ) {
    status = LIBXSMM_DNN_ERR_FUSEDGN_UNSUPPORTED_ORDER;
  } else {
    if ( (handle->desc.fuse_ops == LIBXSMM_DNN_FUSEDGN_OPS_GN) ) {
# include "template/libxsmm_dnn_fusedgroupnorm_st_fwd_custom_f32_bf16_c64_avx512.tpl.c"
    } else if ( (handle->desc.fuse_ops & LIBXSMM_DNN_FUSEDGN_OPS_ELTWISE_RELU) > 0 ) {
# define LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_ELTWISE
# define LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_RELU
# include "template/libxsmm_dnn_fusedgroupnorm_st_fwd_custom_f32_bf16_c64_avx512.tpl.c"
# undef LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_RELU
# undef LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_ELTWISE
    } else if ( (handle->desc.fuse_ops & LIBXSMM_DNN_FUSEDGN_OPS_ELTWISE_RELU_WITH_MASK) > 0 ) {
# define LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_ELTWISE
# define LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_RELU_WITH_MASK
# include "template/libxsmm_dnn_fusedgroupnorm_st_fwd_custom_f32_bf16_c64_avx512.tpl.c"
# undef LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_RELU_WITH_MASK
# undef LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_ELTWISE
    } else if ( (handle->desc.fuse_ops & LIBXSMM_DNN_FUSEDGN_OPS_ELTWISE) > 0 ) {
# define LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_ELTWISE
# include "template/libxsmm_dnn_fusedgroupnorm_st_fwd_custom_f32_bf16_c64_avx512.tpl.c"
# undef LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_ELTWISE
    } else if ( (handle->desc.fuse_ops & LIBXSMM_DNN_FUSEDGN_OPS_RELU) > 0 ) {
# define LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_RELU
# include "template/libxsmm_dnn_fusedgroupnorm_st_fwd_custom_f32_bf16_c64_avx512.tpl.c"
# undef LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_RELU
    } else if ( (handle->desc.fuse_ops & LIBXSMM_DNN_FUSEDGN_OPS_RELU_WITH_MASK) > 0 ) {
# define LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_RELU_WITH_MASK
# include "template/libxsmm_dnn_fusedgroupnorm_st_fwd_custom_f32_bf16_c64_avx512.tpl.c"
# undef LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_RELU_WITH_MASK
    } else {
      status = LIBXSMM_DNN_ERR_FUSEDGN_UNSUPPORTED_FUSION;
    }
  }
# undef LIBXSMM_DNN_FUSEDGN_FWD_BF16
#else /* should not happen */
  LIBXSMM_UNUSED(handle); LIBXSMM_UNUSED(start_thread); LIBXSMM_UNUSED(tid);
  status = LIBXSMM_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}
#endif

LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_fusedgroupnorm_st_fwd_custom(libxsmm_dnn_fusedgroupnorm* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  /* check if all required tensors are bound */
  if ( handle->reg_input == 0 || handle->reg_output == 0 ||
       handle->reg_beta == 0  || handle->reg_gamma == 0  ||
       handle->expvalue == 0  || handle->rcpstddev == 0  || handle->variance == 0 ) {
    status = LIBXSMM_DNN_ERR_DATA_NOT_BOUND;
    return status;
  }
  if ( (handle->desc.fuse_ops & LIBXSMM_DNN_FUSEDGN_OPS_GN) > 0 ) {
    if ( handle->scratch == 0 ) {
      status = LIBXSMM_DNN_ERR_DATA_NOT_BOUND;
      return status;
    }
  }
  if ( (handle->desc.fuse_ops & LIBXSMM_DNN_FUSEDGN_OPS_ELTWISE) > 0 ) {
    if ( handle->reg_add == 0 ) {
      status = LIBXSMM_DNN_ERR_DATA_NOT_BOUND;
      return status;
    }
  }
  if ( (handle->desc.fuse_ops & LIBXSMM_DNN_FUSEDGN_OPS_RELU_WITH_MASK) > 0 ) {
    if ( handle->relumask == 0 ) {
      status = LIBXSMM_DNN_ERR_DATA_NOT_BOUND;
      return status;
    }
  }

  /* check if we are on an AVX512 platform */
#if 0
#if defined(LIBXSMM_INTRINSICS_AVX512) /*__AVX512F__*/
  if ( ( libxsmm_target_archid >= LIBXSMM_X86_AVX512 ) &&
       (handle->ofmblock == 16) ) {
    if (handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_F32 && handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_F32 ) {
      status = libxsmm_dnn_fusedgroupnorm_st_fwd_custom_f32_f32_c16( handle, start_thread, tid );
    } else if (handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_BF16 && handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_BF16 ) {
      status = libxsmm_dnn_fusedgroupnorm_st_fwd_custom_bf16_bf16_c16( handle, start_thread, tid );
    } else {
      status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  } else if ( ( libxsmm_target_archid >= LIBXSMM_X86_AVX512 ) &&
       (handle->ofmblock == 32) ) {
    if (handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_F32 && handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_F32 ) {
      status = libxsmm_dnn_fusedgroupnorm_st_fwd_custom_f32_f32_c32( handle, start_thread, tid );
    } else if (handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_BF16 && handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_BF16 ) {
      status = libxsmm_dnn_fusedgroupnorm_st_fwd_custom_bf16_bf16_c32( handle, start_thread, tid );
    } else {
      status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  } else if ( ( libxsmm_target_archid >= LIBXSMM_X86_AVX512 ) &&
       (handle->ofmblock == 64) ) {
    if (handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_F32 && handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_F32 ) {
      status = libxsmm_dnn_fusedgroupnorm_st_fwd_custom_f32_f32_c64( handle, start_thread, tid );
    } else if (handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_BF16 && handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_BF16 ) {
      status = libxsmm_dnn_fusedgroupnorm_st_fwd_custom_bf16_bf16_c64( handle, start_thread, tid );
    } else {
      status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  } else
#endif
#endif
  {
    if (handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_F32 && handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_F32 ) {
      typedef float element_input_type;
      typedef float element_output_type;
      typedef float element_stats_type;

      if ( handle->desc.fuse_order != LIBXSMM_DNN_FUSEDGN_ORDER_GN_ELTWISE_RELU ) {
        status = LIBXSMM_DNN_ERR_FUSEDGN_UNSUPPORTED_ORDER;
      } else {
        if ( handle->desc.fuse_ops == LIBXSMM_DNN_FUSEDGN_OPS_GN ) {
# include "template/libxsmm_dnn_fusedgroupnorm_st_fwd_custom_generic.tpl.c"
        } else if ( (handle->desc.fuse_ops & LIBXSMM_DNN_FUSEDGN_OPS_ELTWISE_RELU) == LIBXSMM_DNN_FUSEDGN_OPS_ELTWISE_RELU ) {
# define LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_ELTWISE
# define LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_RELU
# include "template/libxsmm_dnn_fusedgroupnorm_st_fwd_custom_generic.tpl.c"
# undef LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_RELU
# undef LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_ELTWISE
        } else if ( (handle->desc.fuse_ops & LIBXSMM_DNN_FUSEDGN_OPS_ELTWISE_RELU_WITH_MASK) == LIBXSMM_DNN_FUSEDGN_OPS_ELTWISE_RELU_WITH_MASK ) {
# define LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_ELTWISE
# define LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_RELU_WITH_MASK
# include "template/libxsmm_dnn_fusedgroupnorm_st_fwd_custom_generic.tpl.c"
# undef LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_RELU_WITH_MASK
# undef LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_ELTWISE
        } else if ( (handle->desc.fuse_ops & LIBXSMM_DNN_FUSEDGN_OPS_ELTWISE) == LIBXSMM_DNN_FUSEDGN_OPS_ELTWISE ) {
# define LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_ELTWISE
# include "template/libxsmm_dnn_fusedgroupnorm_st_fwd_custom_generic.tpl.c"
# undef LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_ELTWISE
        } else if ( (handle->desc.fuse_ops & LIBXSMM_DNN_FUSEDGN_OPS_RELU) == LIBXSMM_DNN_FUSEDGN_OPS_RELU ) {
# define LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_RELU
# include "template/libxsmm_dnn_fusedgroupnorm_st_fwd_custom_generic.tpl.c"
# undef LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_RELU
        } else if ( (handle->desc.fuse_ops & LIBXSMM_DNN_FUSEDGN_OPS_RELU_WITH_MASK) == LIBXSMM_DNN_FUSEDGN_OPS_RELU_WITH_MASK ) {
# define LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_RELU_WITH_MASK
# include "template/libxsmm_dnn_fusedgroupnorm_st_fwd_custom_generic.tpl.c"
# undef LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_RELU_WITH_MASK
        } else {
          status = LIBXSMM_DNN_ERR_FUSEDGN_UNSUPPORTED_FUSION;
        }
      }
    } else if (handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_BF16 && handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_BF16 ) {
      typedef libxsmm_bfloat16 element_input_type;
      typedef libxsmm_bfloat16 element_output_type;
      typedef float element_stats_type;

# define LIBXSMM_DNN_FUSEDGN_FWD_BF16
      if ( handle->desc.fuse_order != LIBXSMM_DNN_FUSEDGN_ORDER_GN_ELTWISE_RELU ) {
        status = LIBXSMM_DNN_ERR_FUSEDGN_UNSUPPORTED_ORDER;
      } else {
        if ( handle->desc.fuse_ops == LIBXSMM_DNN_FUSEDGN_OPS_GN ) {
# include "template/libxsmm_dnn_fusedgroupnorm_st_fwd_custom_generic.tpl.c"
        } else if ( (handle->desc.fuse_ops & LIBXSMM_DNN_FUSEDGN_OPS_ELTWISE_RELU) == LIBXSMM_DNN_FUSEDGN_OPS_ELTWISE_RELU ) {
# define LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_ELTWISE
# define LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_RELU
# include "template/libxsmm_dnn_fusedgroupnorm_st_fwd_custom_generic.tpl.c"
# undef LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_RELU
# undef LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_ELTWISE
        } else if ( (handle->desc.fuse_ops & LIBXSMM_DNN_FUSEDGN_OPS_ELTWISE_RELU_WITH_MASK) == LIBXSMM_DNN_FUSEDGN_OPS_ELTWISE_RELU_WITH_MASK ) {
# define LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_ELTWISE
# define LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_RELU_WITH_MASK
# include "template/libxsmm_dnn_fusedgroupnorm_st_fwd_custom_generic.tpl.c"
# undef LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_RELU_WITH_MASK
# undef LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_ELTWISE
        } else if ( (handle->desc.fuse_ops & LIBXSMM_DNN_FUSEDGN_OPS_ELTWISE) == LIBXSMM_DNN_FUSEDGN_OPS_ELTWISE ) {
# define LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_ELTWISE
# include "template/libxsmm_dnn_fusedgroupnorm_st_fwd_custom_generic.tpl.c"
# undef LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_ELTWISE
        } else if ( (handle->desc.fuse_ops & LIBXSMM_DNN_FUSEDGN_OPS_RELU) == LIBXSMM_DNN_FUSEDGN_OPS_RELU ) {
# define LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_RELU
# include "template/libxsmm_dnn_fusedgroupnorm_st_fwd_custom_generic.tpl.c"
# undef LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_RELU
        } else if ( (handle->desc.fuse_ops & LIBXSMM_DNN_FUSEDGN_OPS_RELU_WITH_MASK) == LIBXSMM_DNN_FUSEDGN_OPS_RELU_WITH_MASK ) {
# define LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_RELU_WITH_MASK
# include "template/libxsmm_dnn_fusedgroupnorm_st_fwd_custom_generic.tpl.c"
# undef LIBXSMM_DNN_FUSEDGN_FWD_ENABLE_RELU_WITH_MASK
        } else {
          status = LIBXSMM_DNN_ERR_FUSEDGN_UNSUPPORTED_FUSION;
        }
      }
# undef LIBXSMM_DNN_FUSEDGN_FWD_BF16
    } else {
      status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  }

  return status;
}


LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_fusedgroupnorm_st_fwd_nhwc(libxsmm_dnn_fusedgroupnorm* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_ERR_NOT_IMPLEMENTED;
  LIBXSMM_UNUSED( handle );
  LIBXSMM_UNUSED( start_thread );
  LIBXSMM_UNUSED( tid );
  return status;
}

