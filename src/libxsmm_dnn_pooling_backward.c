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
#include "libxsmm_dnn_pooling_backward.h"
#include "libxsmm_main.h"


LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_pooling_st_bwd_custom_f32_f32_c16(libxsmm_dnn_pooling* handle, int start_thread, int tid);
LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_pooling_st_bwd_custom_f32_f32_c32(libxsmm_dnn_pooling* handle, int start_thread, int tid);
LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_pooling_st_bwd_custom_f32_f32_c64(libxsmm_dnn_pooling* handle, int start_thread, int tid);
LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_pooling_st_bwd_custom_bf16_bf16_c16(libxsmm_dnn_pooling* handle, int start_thread, int tid);
LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_pooling_st_bwd_custom_bf16_bf16_c32(libxsmm_dnn_pooling* handle, int start_thread, int tid);
LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_pooling_st_bwd_custom_bf16_bf16_c64(libxsmm_dnn_pooling* handle, int start_thread, int tid);


LIBXSMM_API_INTERN LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512)
libxsmm_dnn_err_t libxsmm_dnn_pooling_st_bwd_custom_f32_f32_c16(libxsmm_dnn_pooling* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
#if defined(LIBXSMM_INTRINSICS_AVX512) /*__AVX512F__*/
  typedef float element_input_type;
  typedef float element_output_type;

  if ( handle->desc.pooling_type == LIBXSMM_DNN_POOLING_MAX ) {
# define LIBXSMM_DNN_POOLING_BWD_MAX
    typedef int element_mask_type;
# include "template/libxsmm_dnn_pooling_st_bwd_custom_f32_bf16_c16_avx512.tpl.c"
# undef LIBXSMM_DNN_POOLING_BWD_MAX
  } else if ( handle->desc.pooling_type == LIBXSMM_DNN_POOLING_AVG ) {
# define LIBXSMM_DNN_POOLING_BWD_AVG
# include "template/libxsmm_dnn_pooling_st_bwd_custom_f32_bf16_c16_avx512.tpl.c"
# undef LIBXSMM_DNN_POOLING_BWD_AVG
  } else {
    status = LIBXSMM_DNN_ERR_UNSUPPORTED_POOLING;
  }
#else /* should not happen */
  LIBXSMM_UNUSED(handle); LIBXSMM_UNUSED(start_thread); LIBXSMM_UNUSED(tid);
  status = LIBXSMM_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}


LIBXSMM_API_INTERN LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512)
libxsmm_dnn_err_t libxsmm_dnn_pooling_st_bwd_custom_f32_f32_c32(libxsmm_dnn_pooling* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
#if defined(LIBXSMM_INTRINSICS_AVX512) /*__AVX512F__*/
  typedef float element_input_type;
  typedef float element_output_type;

  if ( handle->desc.pooling_type == LIBXSMM_DNN_POOLING_MAX ) {
# define LIBXSMM_DNN_POOLING_BWD_MAX
    typedef int element_mask_type;
# include "template/libxsmm_dnn_pooling_st_bwd_custom_f32_bf16_c32_avx512.tpl.c"
# undef LIBXSMM_DNN_POOLING_BWD_MAX
  } else if ( handle->desc.pooling_type == LIBXSMM_DNN_POOLING_AVG ) {
# define LIBXSMM_DNN_POOLING_BWD_AVG
# include "template/libxsmm_dnn_pooling_st_bwd_custom_f32_bf16_c32_avx512.tpl.c"
# undef LIBXSMM_DNN_POOLING_BWD_AVG
  } else {
    status = LIBXSMM_DNN_ERR_UNSUPPORTED_POOLING;
  }
#else /* should not happen */
  LIBXSMM_UNUSED(handle); LIBXSMM_UNUSED(start_thread); LIBXSMM_UNUSED(tid);
  status = LIBXSMM_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}


LIBXSMM_API_INTERN LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512)
libxsmm_dnn_err_t libxsmm_dnn_pooling_st_bwd_custom_f32_f32_c64(libxsmm_dnn_pooling* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
#if defined(LIBXSMM_INTRINSICS_AVX512) /*__AVX512F__*/
  typedef float element_input_type;
  typedef float element_output_type;

  if ( handle->desc.pooling_type == LIBXSMM_DNN_POOLING_MAX ) {
# define LIBXSMM_DNN_POOLING_BWD_MAX
    typedef int element_mask_type;
# include "template/libxsmm_dnn_pooling_st_bwd_custom_f32_bf16_c64_avx512.tpl.c"
# undef LIBXSMM_DNN_POOLING_BWD_MAX
  } else if ( handle->desc.pooling_type == LIBXSMM_DNN_POOLING_AVG ) {
# define LIBXSMM_DNN_POOLING_BWD_AVG
# include "template/libxsmm_dnn_pooling_st_bwd_custom_f32_bf16_c64_avx512.tpl.c"
# undef LIBXSMM_DNN_POOLING_BWD_AVG
  } else {
    status = LIBXSMM_DNN_ERR_UNSUPPORTED_POOLING;
  }
#else /* should not happen */
  LIBXSMM_UNUSED(handle); LIBXSMM_UNUSED(start_thread); LIBXSMM_UNUSED(tid);
  status = LIBXSMM_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}


LIBXSMM_API_INTERN LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512)
libxsmm_dnn_err_t libxsmm_dnn_pooling_st_bwd_custom_bf16_bf16_c16(libxsmm_dnn_pooling* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
#if defined(LIBXSMM_INTRINSICS_AVX512) /*__AVX512F__*/
  typedef libxsmm_bfloat16 element_input_type;
  typedef libxsmm_bfloat16 element_output_type;

# define LIBXSMM_DNN_POOLING_BWD_BF16
  if ( handle->desc.pooling_type == LIBXSMM_DNN_POOLING_MAX ) {
# define LIBXSMM_DNN_POOLING_BWD_MAX
    typedef int element_mask_type;
# include "template/libxsmm_dnn_pooling_st_bwd_custom_f32_bf16_c16_avx512.tpl.c"
# undef LIBXSMM_DNN_POOLING_BWD_MAX
  } else if ( handle->desc.pooling_type == LIBXSMM_DNN_POOLING_AVG ) {
# define LIBXSMM_DNN_POOLING_BWD_AVG
# include "template/libxsmm_dnn_pooling_st_bwd_custom_f32_bf16_c16_avx512.tpl.c"
# undef LIBXSMM_DNN_POOLING_BWD_AVG
  } else {
    status = LIBXSMM_DNN_ERR_UNSUPPORTED_POOLING;
  }
# undef LIBXSMM_DNN_POOLING_BWD_BF16
#else /* should not happen */
  LIBXSMM_UNUSED(handle); LIBXSMM_UNUSED(start_thread); LIBXSMM_UNUSED(tid);
  status = LIBXSMM_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}


LIBXSMM_API_INTERN LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512)
libxsmm_dnn_err_t libxsmm_dnn_pooling_st_bwd_custom_bf16_bf16_c32(libxsmm_dnn_pooling* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
#if defined(LIBXSMM_INTRINSICS_AVX512) /*__AVX512F__*/
  typedef libxsmm_bfloat16 element_input_type;
  typedef libxsmm_bfloat16 element_output_type;

# define LIBXSMM_DNN_POOLING_BWD_BF16
  if ( handle->desc.pooling_type == LIBXSMM_DNN_POOLING_MAX ) {
# define LIBXSMM_DNN_POOLING_BWD_MAX
    typedef int element_mask_type;
# include "template/libxsmm_dnn_pooling_st_bwd_custom_f32_bf16_c32_avx512.tpl.c"
# undef LIBXSMM_DNN_POOLING_BWD_MAX
  } else if ( handle->desc.pooling_type == LIBXSMM_DNN_POOLING_AVG ) {
# define LIBXSMM_DNN_POOLING_BWD_AVG
# include "template/libxsmm_dnn_pooling_st_bwd_custom_f32_bf16_c32_avx512.tpl.c"
# undef LIBXSMM_DNN_POOLING_BWD_AVG
  } else {
    status = LIBXSMM_DNN_ERR_UNSUPPORTED_POOLING;
  }
# undef LIBXSMM_DNN_POOLING_BWD_BF16
#else /* should not happen */
  LIBXSMM_UNUSED(handle); LIBXSMM_UNUSED(start_thread); LIBXSMM_UNUSED(tid);
  status = LIBXSMM_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}


LIBXSMM_API_INTERN LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512)
libxsmm_dnn_err_t libxsmm_dnn_pooling_st_bwd_custom_bf16_bf16_c64(libxsmm_dnn_pooling* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
#if defined(LIBXSMM_INTRINSICS_AVX512) /*__AVX512F__*/
  typedef libxsmm_bfloat16 element_input_type;
  typedef libxsmm_bfloat16 element_output_type;

# define LIBXSMM_DNN_POOLING_BWD_BF16
  if ( handle->desc.pooling_type == LIBXSMM_DNN_POOLING_MAX ) {
# define LIBXSMM_DNN_POOLING_BWD_MAX
    typedef int element_mask_type;
# include "template/libxsmm_dnn_pooling_st_bwd_custom_f32_bf16_c64_avx512.tpl.c"
# undef LIBXSMM_DNN_POOLING_BWD_MAX
  } else if ( handle->desc.pooling_type == LIBXSMM_DNN_POOLING_AVG ) {
# define LIBXSMM_DNN_POOLING_BWD_AVG
# include "template/libxsmm_dnn_pooling_st_bwd_custom_f32_bf16_c64_avx512.tpl.c"
# undef LIBXSMM_DNN_POOLING_BWD_AVG
  } else {
    status = LIBXSMM_DNN_ERR_UNSUPPORTED_POOLING;
  }
# undef LIBXSMM_DNN_POOLING_BWD_BF16
#else /* should not happen */
  LIBXSMM_UNUSED(handle); LIBXSMM_UNUSED(start_thread); LIBXSMM_UNUSED(tid);
  status = LIBXSMM_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}


LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_pooling_st_bwd_custom(libxsmm_dnn_pooling* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  /* check if we have input, output and mask */
  if ( handle->grad_input == 0 || handle->grad_output == 0 ||
       ( (handle->mask == 0) && (handle->desc.pooling_type == LIBXSMM_DNN_POOLING_MAX) ) ) {
    status = LIBXSMM_DNN_ERR_DATA_NOT_BOUND;
    return status;
  }

  /* check if we are on an AVX512 platform */
#if defined(LIBXSMM_INTRINSICS_AVX512) /*__AVX512F__*/
  if ( ( libxsmm_target_archid >= LIBXSMM_X86_AVX512 ) &&
       (handle->ofmblock == 16) ) {
    if (handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_F32 && handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_F32 ) {
      LIBXSMM_ASSERT(NULL != handle->mask);
      status = libxsmm_dnn_pooling_st_bwd_custom_f32_f32_c16( handle, start_thread, tid);
    } else if (handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_BF16 && handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_BF16 ) {
      LIBXSMM_ASSERT(NULL != handle->mask);
      status = libxsmm_dnn_pooling_st_bwd_custom_bf16_bf16_c16( handle, start_thread, tid);
    } else {
      status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  } else if ( ( libxsmm_target_archid >= LIBXSMM_X86_AVX512 ) &&
       (handle->ofmblock == 32) ) {
    if (handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_F32 && handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_F32 ) {
      LIBXSMM_ASSERT(NULL != handle->mask);
      status = libxsmm_dnn_pooling_st_bwd_custom_f32_f32_c32( handle, start_thread, tid);
    } else if (handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_BF16 && handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_BF16 ) {
      LIBXSMM_ASSERT(NULL != handle->mask);
      status = libxsmm_dnn_pooling_st_bwd_custom_bf16_bf16_c32( handle, start_thread, tid);
    } else {
      status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  } else if ( ( libxsmm_target_archid >= LIBXSMM_X86_AVX512 ) &&
       (handle->ofmblock == 64) ) {
    if (handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_F32 && handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_F32 ) {
      LIBXSMM_ASSERT(NULL != handle->mask);
      status = libxsmm_dnn_pooling_st_bwd_custom_f32_f32_c64( handle, start_thread, tid);
    } else if (handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_BF16 && handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_BF16 ) {
      LIBXSMM_ASSERT(NULL != handle->mask);
      status = libxsmm_dnn_pooling_st_bwd_custom_bf16_bf16_c64( handle, start_thread, tid);
    } else {
      status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  } else
#endif
  {
    if (handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_F32 && handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_F32 ) {
      typedef float element_input_type;
      typedef float element_output_type;

      if ( handle->desc.pooling_type == LIBXSMM_DNN_POOLING_MAX ) {
# define LIBXSMM_DNN_POOLING_BWD_MAX
        typedef int element_mask_type;
# include "template/libxsmm_dnn_pooling_st_bwd_custom_generic.tpl.c"
# undef LIBXSMM_DNN_POOLING_BWD_MAX
      } else if ( handle->desc.pooling_type == LIBXSMM_DNN_POOLING_AVG ) {
# define LIBXSMM_DNN_POOLING_BWD_AVG
# include "template/libxsmm_dnn_pooling_st_bwd_custom_generic.tpl.c"
# undef LIBXSMM_DNN_POOLING_BWD_AVG
      } else {
        status = LIBXSMM_DNN_ERR_UNSUPPORTED_POOLING;
      }
    } else if (handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_BF16 && handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_BF16 ) {
      typedef libxsmm_bfloat16 element_input_type;
      typedef libxsmm_bfloat16 element_output_type;

# define LIBXSMM_DNN_POOLING_BWD_BF16
      if ( handle->desc.pooling_type == LIBXSMM_DNN_POOLING_MAX ) {
# define LIBXSMM_DNN_POOLING_BWD_MAX
        typedef int element_mask_type;
# include "template/libxsmm_dnn_pooling_st_bwd_custom_generic.tpl.c"
# undef LIBXSMM_DNN_POOLING_BWD_MAX
      } else if ( handle->desc.pooling_type == LIBXSMM_DNN_POOLING_AVG ) {
# define LIBXSMM_DNN_POOLING_BWD_AVG
# include "template/libxsmm_dnn_pooling_st_bwd_custom_generic.tpl.c"
# undef LIBXSMM_DNN_POOLING_BWD_AVG
      } else {
        status = LIBXSMM_DNN_ERR_UNSUPPORTED_POOLING;
      }
# undef LIBXSMM_DNN_POOLING_BWD_BF16
    } else {
      status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  }

  return status;
}


LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_pooling_st_bwd_nhwc(libxsmm_dnn_pooling* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_ERR_NOT_IMPLEMENTED;
  LIBXSMM_UNUSED( handle );
  LIBXSMM_UNUSED( start_thread );
  LIBXSMM_UNUSED( tid );
  return status;
}

