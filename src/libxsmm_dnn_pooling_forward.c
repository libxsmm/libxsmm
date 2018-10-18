/******************************************************************************
** Copyright (c) 2016-2018, Intel Corporation                                **
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
/* Alexander Heinecke (Intel Corp.)
******************************************************************************/
#include "libxsmm_dnn_pooling_forward.h"
#include <libxsmm_intrinsics_x86.h>
#include "libxsmm_main.h"
#include <libxsmm.h>

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#include <string.h>
#include <float.h>
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif


LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_pooling_st_fwd_custom_f32_f32(libxsmm_dnn_pooling* handle, int start_thread, int tid);
LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_pooling_st_fwd_custom_bf16_bf16(libxsmm_dnn_pooling* handle, int start_thread, int tid);


LIBXSMM_API_INTERN LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512)
libxsmm_dnn_err_t libxsmm_dnn_pooling_st_fwd_custom_f32_f32(libxsmm_dnn_pooling* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
#if defined(LIBXSMM_INTRINSICS_AVX512) /*__AVX512F__*/
  typedef float element_input_type;
  typedef float element_output_type;

  if ( handle->desc.pooling_type == LIBXSMM_DNN_POOLING_MAX ) {
# define LIBXSMM_DNN_POOLING_FWD_MAX
    typedef int   element_mask_type;
# include "template/libxsmm_dnn_pooling_st_fwd_custom_f32_bf16_c16_avx512.tpl.c"
# undef LIBXSMM_DNN_POOLING_FWD_MAX
  } else if ( handle->desc.pooling_type == LIBXSMM_DNN_POOLING_AVG ) {
# define LIBXSMM_DNN_POOLING_FWD_AVG
# include "template/libxsmm_dnn_pooling_st_fwd_custom_f32_bf16_c16_avx512.tpl.c"
# undef LIBXSMM_DNN_POOLING_FWD_AVG
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
libxsmm_dnn_err_t libxsmm_dnn_pooling_st_fwd_custom_bf16_bf16(libxsmm_dnn_pooling* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
#if defined(LIBXSMM_INTRINSICS_AVX512) /*__AVX512F__*/
  typedef libxsmm_bfloat16 element_input_type;
  typedef libxsmm_bfloat16 element_output_type;

# define LIBXSMM_DNN_POOLING_FWD_BF16
  if ( handle->desc.pooling_type == LIBXSMM_DNN_POOLING_MAX ) {
# define LIBXSMM_DNN_POOLING_FWD_MAX
    typedef int   element_mask_type;
# include "template/libxsmm_dnn_pooling_st_fwd_custom_f32_bf16_c16_avx512.tpl.c"
# undef LIBXSMM_DNN_POOLING_FWD_MAX
  } else if ( handle->desc.pooling_type == LIBXSMM_DNN_POOLING_AVG ) {
# define LIBXSMM_DNN_POOLING_FWD_AVG
# include "template/libxsmm_dnn_pooling_st_fwd_custom_f32_bf16_c16_avx512.tpl.c"
# undef LIBXSMM_DNN_POOLING_FWD_AVG
  } else {
    status = LIBXSMM_DNN_ERR_UNSUPPORTED_POOLING;
  }
# undef LIBXSMM_DNN_POOLING_FWD_BF16
#else /* should not happen */
  LIBXSMM_UNUSED(handle); LIBXSMM_UNUSED(start_thread); LIBXSMM_UNUSED(tid);
  status = LIBXSMM_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}


LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_pooling_st_fwd_custom(libxsmm_dnn_pooling* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  /* check if we have input, output and mask */
  if ( handle->reg_input == 0 || handle->reg_output == 0 ||
       ( (handle->mask == 0) && (handle->desc.pooling_type == LIBXSMM_DNN_POOLING_MAX) ) ) {
    status = LIBXSMM_DNN_ERR_DATA_NOT_BOUND;
    return status;
  }

  /* check if we are on an AVX512 platform */
  if ( (libxsmm_target_archid == LIBXSMM_X86_AVX512      || libxsmm_target_archid == LIBXSMM_X86_AVX512_MIC ||
        libxsmm_target_archid == LIBXSMM_X86_AVX512_CORE || libxsmm_target_archid == LIBXSMM_X86_AVX512_ICL ||
        libxsmm_target_archid == LIBXSMM_X86_AVX512_KNM  || libxsmm_target_archid == LIBXSMM_X86_AVX512_CPX    ) &&
       (handle->ofmblock == 16) ) {
    if (handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_F32 && handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_F32 ) {
      status = libxsmm_dnn_pooling_st_fwd_custom_f32_f32( handle, start_thread, tid);
    } else if (handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_BF16 && handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_BF16 ) {
      status = libxsmm_dnn_pooling_st_fwd_custom_bf16_bf16( handle, start_thread, tid);
    } else {
      status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  } else {
    if (handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_F32 && handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_F32 ) {
      typedef float element_input_type;
      typedef float element_output_type;

      if ( handle->desc.pooling_type == LIBXSMM_DNN_POOLING_MAX ) {
# define LIBXSMM_DNN_POOLING_FWD_MAX
        typedef int   element_mask_type;
# include "template/libxsmm_dnn_pooling_st_fwd_custom_generic.tpl.c"
# undef LIBXSMM_DNN_POOLING_FWD_MAX
      } else if ( handle->desc.pooling_type == LIBXSMM_DNN_POOLING_AVG ) {
# define LIBXSMM_DNN_POOLING_FWD_AVG
# include "template/libxsmm_dnn_pooling_st_fwd_custom_generic.tpl.c"
# undef LIBXSMM_DNN_POOLING_FWD_AVG
      } else {
        status = LIBXSMM_DNN_ERR_UNSUPPORTED_POOLING;
      }
    } else if (handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_BF16 && handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_BF16 ) {
      typedef libxsmm_bfloat16 element_input_type;
      typedef libxsmm_bfloat16 element_output_type;

# define LIBXSMM_DNN_POOLING_FWD_BF16
      if ( handle->desc.pooling_type == LIBXSMM_DNN_POOLING_MAX ) {
# define LIBXSMM_DNN_POOLING_FWD_MAX
        typedef int   element_mask_type;
# include "template/libxsmm_dnn_pooling_st_fwd_custom_generic.tpl.c"
# undef LIBXSMM_DNN_POOLING_FWD_MAX
      } else if ( handle->desc.pooling_type == LIBXSMM_DNN_POOLING_AVG ) {
# define LIBXSMM_DNN_POOLING_FWD_AVG
# include "template/libxsmm_dnn_pooling_st_fwd_custom_generic.tpl.c"
# undef LIBXSMM_DNN_POOLING_FWD_AVG
      } else {
        status = LIBXSMM_DNN_ERR_UNSUPPORTED_POOLING;
      }
# undef LIBXSMM_DNN_POOLING_FWD_BF16
    } else {
      status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  }

  return status;
}


LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_pooling_st_fwd_nhwc(libxsmm_dnn_pooling* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_ERR_NOT_IMPLEMENTED;
  LIBXSMM_UNUSED( handle );
  LIBXSMM_UNUSED( start_thread );
  LIBXSMM_UNUSED( tid );
  return status;
}

