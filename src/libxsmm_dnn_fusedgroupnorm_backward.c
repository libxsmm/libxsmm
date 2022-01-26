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
#include "libxsmm_dnn_fusedgroupnorm_backward.h"
#include "libxsmm_main.h"

#if 0
LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_fusedgroupnorm_st_bwd_custom_f32_f32_c16(libxsmm_dnn_fusedgroupnorm* handle, int start_thread, int tid);
LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_fusedgroupnorm_st_bwd_custom_f32_f32_c32(libxsmm_dnn_fusedgroupnorm* handle, int start_thread, int tid);
LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_fusedgroupnorm_st_bwd_custom_f32_f32_c64(libxsmm_dnn_fusedgroupnorm* handle, int start_thread, int tid);
LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_fusedgroupnorm_st_bwd_custom_bf16_bf16_c16(libxsmm_dnn_fusedgroupnorm* handle, int start_thread, int tid);
LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_fusedgroupnorm_st_bwd_custom_bf16_bf16_c32(libxsmm_dnn_fusedgroupnorm* handle, int start_thread, int tid);
LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_fusedgroupnorm_st_bwd_custom_bf16_bf16_c64(libxsmm_dnn_fusedgroupnorm* handle, int start_thread, int tid);


LIBXSMM_API_INTERN LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512)
libxsmm_dnn_err_t libxsmm_dnn_fusedgroupnorm_st_bwd_custom_f32_f32_c16(libxsmm_dnn_fusedgroupnorm* handle, int start_thread, int tid)
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
# include "template/libxsmm_dnn_fusedgroupnorm_st_bwd_custom_f32_bf16_c16_avx512.tpl.c"
    } else if ( (handle->desc.fuse_ops & LIBXSMM_DNN_FUSEDGN_OPS_ELTWISE_RELU) > 0 ) {
# define LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_ELTWISE
# define LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_RELU
# include "template/libxsmm_dnn_fusedgroupnorm_st_bwd_custom_f32_bf16_c16_avx512.tpl.c"
# undef LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_RELU
# undef LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_ELTWISE
    } else if ( (handle->desc.fuse_ops & LIBXSMM_DNN_FUSEDGN_OPS_ELTWISE_RELU_WITH_MASK) > 0 ) {
# define LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_ELTWISE
# define LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_RELU_WITH_MASK
# include "template/libxsmm_dnn_fusedgroupnorm_st_bwd_custom_f32_bf16_c16_avx512.tpl.c"
# undef LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_RELU_WITH_MASK
# undef LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_ELTWISE
    } else if ( (handle->desc.fuse_ops & LIBXSMM_DNN_FUSEDGN_OPS_ELTWISE) > 0 ) {
# define LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_ELTWISE
# include "template/libxsmm_dnn_fusedgroupnorm_st_bwd_custom_f32_bf16_c16_avx512.tpl.c"
# undef LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_ELTWISE
    } else if ( (handle->desc.fuse_ops & LIBXSMM_DNN_FUSEDGN_OPS_RELU) > 0 ) {
# define LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_RELU
# include "template/libxsmm_dnn_fusedgroupnorm_st_bwd_custom_f32_bf16_c16_avx512.tpl.c"
# undef LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_RELU
    } else if ( (handle->desc.fuse_ops & LIBXSMM_DNN_FUSEDGN_OPS_RELU_WITH_MASK) > 0 ) {
# define LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_RELU_WITH_MASK
# include "template/libxsmm_dnn_fusedgroupnorm_st_bwd_custom_f32_bf16_c16_avx512.tpl.c"
# undef LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_RELU_WITH_MASK
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
libxsmm_dnn_err_t libxsmm_dnn_fusedgroupnorm_st_bwd_custom_f32_f32_c32(libxsmm_dnn_fusedgroupnorm* handle, int start_thread, int tid)
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
# include "template/libxsmm_dnn_fusedgroupnorm_st_bwd_custom_f32_bf16_c32_avx512.tpl.c"
    } else if ( (handle->desc.fuse_ops & LIBXSMM_DNN_FUSEDGN_OPS_ELTWISE_RELU) > 0 ) {
# define LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_ELTWISE
# define LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_RELU
# include "template/libxsmm_dnn_fusedgroupnorm_st_bwd_custom_f32_bf16_c32_avx512.tpl.c"
# undef LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_RELU
# undef LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_ELTWISE
    } else if ( (handle->desc.fuse_ops & LIBXSMM_DNN_FUSEDGN_OPS_ELTWISE_RELU_WITH_MASK) > 0 ) {
# define LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_ELTWISE
# define LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_RELU_WITH_MASK
# include "template/libxsmm_dnn_fusedgroupnorm_st_bwd_custom_f32_bf16_c32_avx512.tpl.c"
# undef LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_RELU_WITH_MASK
# undef LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_ELTWISE
    } else if ( (handle->desc.fuse_ops & LIBXSMM_DNN_FUSEDGN_OPS_ELTWISE) > 0 ) {
# define LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_ELTWISE
# include "template/libxsmm_dnn_fusedgroupnorm_st_bwd_custom_f32_bf16_c32_avx512.tpl.c"
# undef LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_ELTWISE
    } else if ( (handle->desc.fuse_ops & LIBXSMM_DNN_FUSEDGN_OPS_RELU) > 0 ) {
# define LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_RELU
# include "template/libxsmm_dnn_fusedgroupnorm_st_bwd_custom_f32_bf16_c32_avx512.tpl.c"
# undef LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_RELU
    } else if ( (handle->desc.fuse_ops & LIBXSMM_DNN_FUSEDGN_OPS_RELU_WITH_MASK) > 0 ) {
# define LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_RELU_WITH_MASK
# include "template/libxsmm_dnn_fusedgroupnorm_st_bwd_custom_f32_bf16_c32_avx512.tpl.c"
# undef LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_RELU_WITH_MASK
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
libxsmm_dnn_err_t libxsmm_dnn_fusedgroupnorm_st_bwd_custom_f32_f32_c64(libxsmm_dnn_fusedgroupnorm* handle, int start_thread, int tid)
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
# include "template/libxsmm_dnn_fusedgroupnorm_st_bwd_custom_f32_bf16_c64_avx512.tpl.c"
    } else if ( (handle->desc.fuse_ops & LIBXSMM_DNN_FUSEDGN_OPS_ELTWISE_RELU) > 0 ) {
# define LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_ELTWISE
# define LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_RELU
# include "template/libxsmm_dnn_fusedgroupnorm_st_bwd_custom_f32_bf16_c64_avx512.tpl.c"
# undef LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_RELU
# undef LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_ELTWISE
    } else if ( (handle->desc.fuse_ops & LIBXSMM_DNN_FUSEDGN_OPS_ELTWISE_RELU_WITH_MASK) > 0 ) {
# define LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_ELTWISE
# define LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_RELU_WITH_MASK
# include "template/libxsmm_dnn_fusedgroupnorm_st_bwd_custom_f32_bf16_c64_avx512.tpl.c"
# undef LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_RELU_WITH_MASK
# undef LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_ELTWISE
    } else if ( (handle->desc.fuse_ops & LIBXSMM_DNN_FUSEDGN_OPS_ELTWISE) > 0 ) {
# define LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_ELTWISE
# include "template/libxsmm_dnn_fusedgroupnorm_st_bwd_custom_f32_bf16_c64_avx512.tpl.c"
# undef LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_ELTWISE
    } else if ( (handle->desc.fuse_ops & LIBXSMM_DNN_FUSEDGN_OPS_RELU) > 0 ) {
# define LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_RELU
# include "template/libxsmm_dnn_fusedgroupnorm_st_bwd_custom_f32_bf16_c64_avx512.tpl.c"
# undef LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_RELU
    } else if ( (handle->desc.fuse_ops & LIBXSMM_DNN_FUSEDGN_OPS_RELU_WITH_MASK) > 0 ) {
# define LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_RELU_WITH_MASK
# include "template/libxsmm_dnn_fusedgroupnorm_st_bwd_custom_f32_bf16_c64_avx512.tpl.c"
# undef LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_RELU_WITH_MASK
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
libxsmm_dnn_err_t libxsmm_dnn_fusedgroupnorm_st_bwd_custom_bf16_bf16_c16(libxsmm_dnn_fusedgroupnorm* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
#if defined(LIBXSMM_INTRINSICS_AVX512) /*__AVX512F__*/
  typedef libxsmm_bfloat16 element_input_type;
  typedef libxsmm_bfloat16 element_output_type;
  typedef float element_stats_type;

# define LIBXSMM_DNN_FUSEDGN_BWD_BF16
  if ( handle->desc.fuse_order != LIBXSMM_DNN_FUSEDGN_ORDER_GN_ELTWISE_RELU ) {
    status = LIBXSMM_DNN_ERR_FUSEDGN_UNSUPPORTED_ORDER;
  } else {
    if ( (handle->desc.fuse_ops == LIBXSMM_DNN_FUSEDGN_OPS_GN) ) {
# include "template/libxsmm_dnn_fusedgroupnorm_st_bwd_custom_f32_bf16_c16_avx512.tpl.c"
    } else if ( (handle->desc.fuse_ops & LIBXSMM_DNN_FUSEDGN_OPS_ELTWISE_RELU) > 0 ) {
# define LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_ELTWISE
# define LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_RELU
# include "template/libxsmm_dnn_fusedgroupnorm_st_bwd_custom_f32_bf16_c16_avx512.tpl.c"
# undef LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_RELU
# undef LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_ELTWISE
    } else if ( (handle->desc.fuse_ops & LIBXSMM_DNN_FUSEDGN_OPS_ELTWISE_RELU_WITH_MASK) > 0 ) {
# define LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_ELTWISE
# define LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_RELU_WITH_MASK
# include "template/libxsmm_dnn_fusedgroupnorm_st_bwd_custom_f32_bf16_c16_avx512.tpl.c"
# undef LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_RELU_WITH_MASK
# undef LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_ELTWISE
    } else if ( (handle->desc.fuse_ops & LIBXSMM_DNN_FUSEDGN_OPS_ELTWISE) > 0 ) {
# define LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_ELTWISE
# include "template/libxsmm_dnn_fusedgroupnorm_st_bwd_custom_f32_bf16_c16_avx512.tpl.c"
# undef LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_ELTWISE
    } else if ( (handle->desc.fuse_ops & LIBXSMM_DNN_FUSEDGN_OPS_RELU) > 0 ) {
# define LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_RELU
# include "template/libxsmm_dnn_fusedgroupnorm_st_bwd_custom_f32_bf16_c16_avx512.tpl.c"
# undef LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_RELU
    } else if ( (handle->desc.fuse_ops & LIBXSMM_DNN_FUSEDGN_OPS_RELU_WITH_MASK) > 0 ) {
# define LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_RELU_WITH_MASK
# include "template/libxsmm_dnn_fusedgroupnorm_st_bwd_custom_f32_bf16_c16_avx512.tpl.c"
# undef LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_RELU_WITH_MASK
    } else {
      status = LIBXSMM_DNN_ERR_FUSEDGN_UNSUPPORTED_FUSION;
    }
  }
# undef LIBXSMM_DNN_FUSEDGN_BWD_BF16
#else /* should not happen */
  LIBXSMM_UNUSED(handle); LIBXSMM_UNUSED(start_thread); LIBXSMM_UNUSED(tid);
  status = LIBXSMM_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}


LIBXSMM_API_INTERN LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512)
libxsmm_dnn_err_t libxsmm_dnn_fusedgroupnorm_st_bwd_custom_bf16_bf16_c32(libxsmm_dnn_fusedgroupnorm* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
#if defined(LIBXSMM_INTRINSICS_AVX512) /*__AVX512F__*/
  typedef libxsmm_bfloat16 element_input_type;
  typedef libxsmm_bfloat16 element_output_type;
  typedef float element_stats_type;

# define LIBXSMM_DNN_FUSEDGN_BWD_BF16
  if ( handle->desc.fuse_order != LIBXSMM_DNN_FUSEDGN_ORDER_GN_ELTWISE_RELU ) {
    status = LIBXSMM_DNN_ERR_FUSEDGN_UNSUPPORTED_ORDER;
  } else {
    if ( (handle->desc.fuse_ops == LIBXSMM_DNN_FUSEDGN_OPS_GN) ) {
# include "template/libxsmm_dnn_fusedgroupnorm_st_bwd_custom_f32_bf16_c32_avx512.tpl.c"
    } else if ( (handle->desc.fuse_ops & LIBXSMM_DNN_FUSEDGN_OPS_ELTWISE_RELU) > 0 ) {
# define LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_ELTWISE
# define LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_RELU
# include "template/libxsmm_dnn_fusedgroupnorm_st_bwd_custom_f32_bf16_c32_avx512.tpl.c"
# undef LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_RELU
# undef LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_ELTWISE
    } else if ( (handle->desc.fuse_ops & LIBXSMM_DNN_FUSEDGN_OPS_ELTWISE_RELU_WITH_MASK) > 0 ) {
# define LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_ELTWISE
# define LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_RELU_WITH_MASK
# include "template/libxsmm_dnn_fusedgroupnorm_st_bwd_custom_f32_bf16_c32_avx512.tpl.c"
# undef LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_RELU_WITH_MASK
# undef LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_ELTWISE
    } else if ( (handle->desc.fuse_ops & LIBXSMM_DNN_FUSEDGN_OPS_ELTWISE) > 0 ) {
# define LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_ELTWISE
# include "template/libxsmm_dnn_fusedgroupnorm_st_bwd_custom_f32_bf16_c32_avx512.tpl.c"
# undef LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_ELTWISE
    } else if ( (handle->desc.fuse_ops & LIBXSMM_DNN_FUSEDGN_OPS_RELU) > 0 ) {
# define LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_RELU
# include "template/libxsmm_dnn_fusedgroupnorm_st_bwd_custom_f32_bf16_c32_avx512.tpl.c"
# undef LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_RELU
    } else if ( (handle->desc.fuse_ops & LIBXSMM_DNN_FUSEDGN_OPS_RELU_WITH_MASK) > 0 ) {
# define LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_RELU_WITH_MASK
# include "template/libxsmm_dnn_fusedgroupnorm_st_bwd_custom_f32_bf16_c32_avx512.tpl.c"
# undef LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_RELU_WITH_MASK
    } else {
      status = LIBXSMM_DNN_ERR_FUSEDGN_UNSUPPORTED_FUSION;
    }
  }
# undef LIBXSMM_DNN_FUSEDGN_BWD_BF16
#else /* should not happen */
  LIBXSMM_UNUSED(handle); LIBXSMM_UNUSED(start_thread); LIBXSMM_UNUSED(tid);
  status = LIBXSMM_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}


LIBXSMM_API_INTERN LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512)
libxsmm_dnn_err_t libxsmm_dnn_fusedgroupnorm_st_bwd_custom_bf16_bf16_c64(libxsmm_dnn_fusedgroupnorm* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
#if defined(LIBXSMM_INTRINSICS_AVX512) /*__AVX512F__*/
  typedef libxsmm_bfloat16 element_input_type;
  typedef libxsmm_bfloat16 element_output_type;
  typedef float element_stats_type;

# define LIBXSMM_DNN_FUSEDGN_BWD_BF16
  if ( handle->desc.fuse_order != LIBXSMM_DNN_FUSEDGN_ORDER_GN_ELTWISE_RELU ) {
    status = LIBXSMM_DNN_ERR_FUSEDGN_UNSUPPORTED_ORDER;
  } else {
    if ( (handle->desc.fuse_ops == LIBXSMM_DNN_FUSEDGN_OPS_GN) ) {
# include "template/libxsmm_dnn_fusedgroupnorm_st_bwd_custom_f32_bf16_c64_avx512.tpl.c"
    } else if ( (handle->desc.fuse_ops & LIBXSMM_DNN_FUSEDGN_OPS_ELTWISE_RELU) > 0 ) {
# define LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_ELTWISE
# define LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_RELU
# include "template/libxsmm_dnn_fusedgroupnorm_st_bwd_custom_f32_bf16_c64_avx512.tpl.c"
# undef LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_RELU
# undef LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_ELTWISE
    } else if ( (handle->desc.fuse_ops & LIBXSMM_DNN_FUSEDGN_OPS_ELTWISE_RELU_WITH_MASK) > 0 ) {
# define LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_ELTWISE
# define LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_RELU_WITH_MASK
# include "template/libxsmm_dnn_fusedgroupnorm_st_bwd_custom_f32_bf16_c64_avx512.tpl.c"
# undef LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_RELU_WITH_MASK
# undef LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_ELTWISE
    } else if ( (handle->desc.fuse_ops & LIBXSMM_DNN_FUSEDGN_OPS_ELTWISE) > 0 ) {
# define LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_ELTWISE
# include "template/libxsmm_dnn_fusedgroupnorm_st_bwd_custom_f32_bf16_c64_avx512.tpl.c"
# undef LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_ELTWISE
    } else if ( (handle->desc.fuse_ops & LIBXSMM_DNN_FUSEDGN_OPS_RELU) > 0 ) {
# define LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_RELU
# include "template/libxsmm_dnn_fusedgroupnorm_st_bwd_custom_f32_bf16_c64_avx512.tpl.c"
# undef LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_RELU
    } else if ( (handle->desc.fuse_ops & LIBXSMM_DNN_FUSEDGN_OPS_RELU_WITH_MASK) > 0 ) {
# define LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_RELU_WITH_MASK
# include "template/libxsmm_dnn_fusedgroupnorm_st_bwd_custom_f32_bf16_c64_avx512.tpl.c"
# undef LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_RELU_WITH_MASK
    } else {
      status = LIBXSMM_DNN_ERR_FUSEDGN_UNSUPPORTED_FUSION;
    }
  }
# undef LIBXSMM_DNN_FUSEDGN_BWD_BF16
#else /* should not happen */
  LIBXSMM_UNUSED(handle); LIBXSMM_UNUSED(start_thread); LIBXSMM_UNUSED(tid);
  status = LIBXSMM_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}
#endif

LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_fusedgroupnorm_st_bwd_custom(libxsmm_dnn_fusedgroupnorm* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  /* check if all required tensors are bound */
  if ( handle->reg_input == 0  || handle->reg_gamma == 0   ||
       handle->grad_input == 0 || handle->grad_output == 0 ||
       handle->grad_beta == 0  || handle->grad_gamma == 0  ||
       handle->expvalue == 0   || handle->rcpstddev == 0      ) {
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
    if ( handle->grad_add == 0 ) {
      status = LIBXSMM_DNN_ERR_DATA_NOT_BOUND;
      return status;
    }
  }
  if ( (handle->desc.fuse_ops & LIBXSMM_DNN_FUSEDGN_OPS_RELU) > 0 ) {
    if ( handle->reg_output == 0 ) {
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
      status = libxsmm_dnn_fusedgroupnorm_st_bwd_custom_f32_f32_c16( handle, start_thread, tid );
    } else if (handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_BF16 && handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_BF16 ) {
      status = libxsmm_dnn_fusedgroupnorm_st_bwd_custom_bf16_bf16_c16( handle, start_thread, tid );
    } else {
      status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  } else if ( ( libxsmm_target_archid >= LIBXSMM_X86_AVX512 ) &&
       (handle->ofmblock == 32) ) {
    if (handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_F32 && handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_F32 ) {
      status = libxsmm_dnn_fusedgroupnorm_st_bwd_custom_f32_f32_c32( handle, start_thread, tid );
    } else if (handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_BF16 && handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_BF16 ) {
      status = libxsmm_dnn_fusedgroupnorm_st_bwd_custom_bf16_bf16_c32( handle, start_thread, tid );
    } else {
      status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  } else if ( ( libxsmm_target_archid >= LIBXSMM_X86_AVX512 ) &&
       (handle->ofmblock == 64) ) {
    if (handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_F32 && handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_F32 ) {
      status = libxsmm_dnn_fusedgroupnorm_st_bwd_custom_f32_f32_c64( handle, start_thread, tid );
    } else if (handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_BF16 && handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_BF16 ) {
      status = libxsmm_dnn_fusedgroupnorm_st_bwd_custom_bf16_bf16_c64( handle, start_thread, tid );
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
# include "template/libxsmm_dnn_fusedgroupnorm_st_bwd_custom_generic.tpl.c"
        } else if ( (handle->desc.fuse_ops & LIBXSMM_DNN_FUSEDGN_OPS_ELTWISE_RELU) == LIBXSMM_DNN_FUSEDGN_OPS_ELTWISE_RELU ) {
# define LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_ELTWISE
# define LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_RELU
# include "template/libxsmm_dnn_fusedgroupnorm_st_bwd_custom_generic.tpl.c"
# undef LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_RELU
# undef LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_ELTWISE
        } else if ( (handle->desc.fuse_ops & LIBXSMM_DNN_FUSEDGN_OPS_ELTWISE_RELU_WITH_MASK) == LIBXSMM_DNN_FUSEDGN_OPS_ELTWISE_RELU_WITH_MASK ) {
# define LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_ELTWISE
# define LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_RELU_WITH_MASK
# include "template/libxsmm_dnn_fusedgroupnorm_st_bwd_custom_generic.tpl.c"
# undef LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_RELU_WITH_MASK
# undef LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_ELTWISE
        } else if ( (handle->desc.fuse_ops & LIBXSMM_DNN_FUSEDGN_OPS_ELTWISE) == LIBXSMM_DNN_FUSEDGN_OPS_ELTWISE ) {
# define LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_ELTWISE
# include "template/libxsmm_dnn_fusedgroupnorm_st_bwd_custom_generic.tpl.c"
# undef LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_ELTWISE
        } else if ( (handle->desc.fuse_ops & LIBXSMM_DNN_FUSEDGN_OPS_RELU) ==  LIBXSMM_DNN_FUSEDGN_OPS_RELU ) {
# define LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_RELU
# include "template/libxsmm_dnn_fusedgroupnorm_st_bwd_custom_generic.tpl.c"
# undef LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_RELU
        } else if ( (handle->desc.fuse_ops & LIBXSMM_DNN_FUSEDGN_OPS_RELU_WITH_MASK) == LIBXSMM_DNN_FUSEDGN_OPS_RELU_WITH_MASK ) {
# define LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_RELU_WITH_MASK
# include "template/libxsmm_dnn_fusedgroupnorm_st_bwd_custom_generic.tpl.c"
# undef LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_RELU_WITH_MASK
        } else {
          status = LIBXSMM_DNN_ERR_FUSEDGN_UNSUPPORTED_FUSION;
        }
      }
    } else if (handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_BF16 && handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_BF16 ) {
      typedef libxsmm_bfloat16 element_input_type;
      typedef libxsmm_bfloat16 element_output_type;
      typedef float element_stats_type;

# define LIBXSMM_DNN_FUSEDGN_BWD_BF16
      if ( handle->desc.fuse_order != LIBXSMM_DNN_FUSEDGN_ORDER_GN_ELTWISE_RELU ) {
        status = LIBXSMM_DNN_ERR_FUSEDGN_UNSUPPORTED_ORDER;
      } else {
        if ( handle->desc.fuse_ops == LIBXSMM_DNN_FUSEDGN_OPS_GN ) {
# include "template/libxsmm_dnn_fusedgroupnorm_st_bwd_custom_generic.tpl.c"
        } else if ( (handle->desc.fuse_ops & LIBXSMM_DNN_FUSEDGN_OPS_ELTWISE_RELU) == LIBXSMM_DNN_FUSEDGN_OPS_ELTWISE_RELU ) {
# define LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_ELTWISE
# define LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_RELU
# include "template/libxsmm_dnn_fusedgroupnorm_st_bwd_custom_generic.tpl.c"
# undef LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_RELU
# undef LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_ELTWISE
        } else if ( (handle->desc.fuse_ops & LIBXSMM_DNN_FUSEDGN_OPS_ELTWISE_RELU_WITH_MASK) == LIBXSMM_DNN_FUSEDGN_OPS_ELTWISE_RELU_WITH_MASK ) {
# define LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_ELTWISE
# define LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_RELU_WITH_MASK
# include "template/libxsmm_dnn_fusedgroupnorm_st_bwd_custom_generic.tpl.c"
# undef LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_RELU_WITH_MASK
# undef LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_ELTWISE
        } else if ( (handle->desc.fuse_ops & LIBXSMM_DNN_FUSEDGN_OPS_ELTWISE) == LIBXSMM_DNN_FUSEDGN_OPS_ELTWISE ) {
# define LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_ELTWISE
# include "template/libxsmm_dnn_fusedgroupnorm_st_bwd_custom_generic.tpl.c"
# undef LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_ELTWISE
        } else if ( (handle->desc.fuse_ops & LIBXSMM_DNN_FUSEDGN_OPS_RELU) == LIBXSMM_DNN_FUSEDGN_OPS_RELU ) {
# define LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_RELU
# include "template/libxsmm_dnn_fusedgroupnorm_st_bwd_custom_generic.tpl.c"
# undef LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_RELU
        } else if ( (handle->desc.fuse_ops & LIBXSMM_DNN_FUSEDGN_OPS_RELU_WITH_MASK) == LIBXSMM_DNN_FUSEDGN_OPS_RELU_WITH_MASK ) {
# define LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_RELU_WITH_MASK
# include "template/libxsmm_dnn_fusedgroupnorm_st_bwd_custom_generic.tpl.c"
# undef LIBXSMM_DNN_FUSEDGN_BWD_ENABLE_RELU_WITH_MASK
        } else {
          status = LIBXSMM_DNN_ERR_FUSEDGN_UNSUPPORTED_FUSION;
        }
      }
# undef LIBXSMM_DNN_FUSEDGN_BWD_BF16
    } else {
      status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  }

  return status;
}


LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_fusedgroupnorm_st_bwd_nhwc(libxsmm_dnn_fusedgroupnorm* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_ERR_NOT_IMPLEMENTED;
  LIBXSMM_UNUSED( handle );
  LIBXSMM_UNUSED( start_thread );
  LIBXSMM_UNUSED( tid );
  return status;
}

LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_fusedgroupnorm_reduce_stats_st_bwd_custom(libxsmm_dnn_fusedgroupnorm** handles, int num_handles, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
  int l_count;

  /* check if all required tensors are bound */
  for ( l_count = 0; l_count < num_handles; ++l_count ) {
    if ( handles[l_count]->grad_beta == 0  || handles[l_count]->grad_gamma == 0 ) {
      status = LIBXSMM_DNN_ERR_DATA_NOT_BOUND;
      return status;
    }
  }

#if 0
  /* check if we are on an AVX512 platform */
  if ( libxsmm_target_archid >= LIBXSMM_X86_AVX512 ) {
    status = libxsmm_dnn_fusedgroupnorm_reduce_stats_st_bwd_custom_avx512( handles, num_handles, start_thread, tid );
  } else
#endif
  {
    const int nBlocksFm = handles[0]->blocksifm;
    const int nFmBlock = handles[0]->ifmblock;
    /* computing first logical thread */
    const int ltid = tid - start_thread;
    /* number of tasks that could be run in parallel */
    const int work2 = nBlocksFm;
    /* compute chunk size */
    const int chunksize2 = (work2 % handles[0]->desc.threads == 0) ? (work2 / handles[0]->desc.threads) : ((work2 / handles[0]->desc.threads) + 1);
    /* compute thr_begin and thr_end */
    const int thr_begin2 = (ltid * chunksize2 < work2) ? (ltid * chunksize2) : work2;
    const int thr_end2 = ((ltid + 1) * chunksize2 < work2) ? ((ltid + 1) * chunksize2) : work2;
    int v = 0, fm;

    LIBXSMM_VLA_DECL(2, float, dgamma0,      (float*)handles[0]->grad_gamma->data, nFmBlock);
    LIBXSMM_VLA_DECL(2, float, dbeta0,       (float*)handles[0]->grad_beta->data,  nFmBlock);

    /* lazy barrier init */
    libxsmm_barrier_init(handles[0]->barrier, ltid);

    /* now we need to reduce the dgamma and dbeta  */
    for ( l_count = 1; l_count < num_handles; ++l_count ) {
      LIBXSMM_VLA_DECL(2, float, dgammar, (float*)handles[l_count]->grad_gamma->data, nFmBlock);
      LIBXSMM_VLA_DECL(2, float, dbetar,  (float*)handles[l_count]->grad_beta->data,  nFmBlock);

      for ( fm = thr_begin2; fm < thr_end2; ++fm ) {
        float* dgamma0_ptr = &LIBXSMM_VLA_ACCESS(2, dgamma0, fm, 0, nFmBlock);
        float* dbeta0_ptr  = &LIBXSMM_VLA_ACCESS(2, dbeta0,  fm, 0, nFmBlock);
        float* dgammar_ptr = &LIBXSMM_VLA_ACCESS(2, dgammar, fm, 0, nFmBlock);
        float* dbetar_ptr  = &LIBXSMM_VLA_ACCESS(2, dbetar,  fm, 0, nFmBlock);

        LIBXSMM_PRAGMA_SIMD
        for ( v=0; v < nFmBlock; v++ ) {
          dgamma0_ptr[v] += dgammar_ptr[v];
          dbeta0_ptr[v]  += dbetar_ptr[v];
        }
      }
    }

    for ( l_count = 1; l_count < num_handles; ++l_count ) {
      LIBXSMM_VLA_DECL(2, float, dgammar, (float*)handles[l_count]->grad_gamma->data, nFmBlock);
      LIBXSMM_VLA_DECL(2, float, dbetar,  (float*)handles[l_count]->grad_beta->data,  nFmBlock);

      for ( fm = thr_begin2; fm < thr_end2; ++fm ) {
        float* dgamma0_ptr = &LIBXSMM_VLA_ACCESS(2, dgamma0, fm, 0, nFmBlock);
        float* dbeta0_ptr  = &LIBXSMM_VLA_ACCESS(2, dbeta0,  fm, 0, nFmBlock);
        float* dgammar_ptr = &LIBXSMM_VLA_ACCESS(2, dgammar, fm, 0, nFmBlock);
        float* dbetar_ptr  = &LIBXSMM_VLA_ACCESS(2, dbetar,  fm, 0, nFmBlock);

        LIBXSMM_PRAGMA_SIMD
        for ( v=0; v < nFmBlock; v++ ) {
          dgammar_ptr[v] = dgamma0_ptr[v];
          dbetar_ptr[v]  = dbeta0_ptr[v];
        }
      }
    }

    libxsmm_barrier_wait(handles[0]->barrier, ltid);
  }

  return status;
}

