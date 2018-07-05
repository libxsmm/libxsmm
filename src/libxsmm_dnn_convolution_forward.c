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
/* Alexander Heinecke, Hans Pabst (Intel Corp.)
******************************************************************************/
#include "libxsmm_dnn_convolution_forward.h"
#include <libxsmm_intrinsics_x86.h>
#include "libxsmm_main.h"
#include <libxsmm.h>

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#include <string.h>
#include <math.h>
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif


LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_convolve_st_fwd_custom_custom_f32_f32(libxsmm_dnn_layer* handle, int start_thread, int tid);
LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_convolve_st_fwd_custom_custom_bf16_bf16(libxsmm_dnn_layer* handle, int start_thread, int tid);
LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_convolve_st_fwd_custom_custom_i16_i32(libxsmm_dnn_layer* handle, int start_thread, int tid);
LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_convolve_st_fwd_custom_custom_i16_f32(libxsmm_dnn_layer* handle, int start_thread, int tid);
LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_convolve_st_fwd_custom_custom_i8_i32(libxsmm_dnn_layer* handle, int start_thread, int tid);


LIBXSMM_API_INTERN LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512)
libxsmm_dnn_err_t libxsmm_dnn_convolve_st_fwd_custom_custom_f32_f32(libxsmm_dnn_layer* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
#if defined(LIBXSMM_INTRINSICS_AVX512) /*__AVX512F__*/
  typedef float element_input_type;
  typedef float element_output_type;
  typedef float element_filter_type;
  typedef libxsmm_sconvfunction libxsmm_convfunction;
# include "template/libxsmm_dnn_convolve_st_fwd_custom_custom.tpl.c"
#else /* should not happen */
  LIBXSMM_UNUSED(handle); LIBXSMM_UNUSED(start_thread); LIBXSMM_UNUSED(tid);
  status = LIBXSMM_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}


LIBXSMM_API_INTERN LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512)
libxsmm_dnn_err_t libxsmm_dnn_convolve_st_fwd_custom_custom_bf16_bf16(libxsmm_dnn_layer* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
#if defined(LIBXSMM_INTRINSICS_AVX512) /*__AVX512F__*/
  typedef libxsmm_bfloat16 element_input_type;
  typedef libxsmm_bfloat16 element_output_type;
  typedef libxsmm_bfloat16 element_filter_type;
  typedef libxsmm_bf16convfunction libxsmm_convfunction;
# include "template/libxsmm_dnn_convolve_st_fwd_custom_custom.tpl.c"
#else /* should not happen */
  LIBXSMM_UNUSED(handle); LIBXSMM_UNUSED(start_thread); LIBXSMM_UNUSED(tid);
  status = LIBXSMM_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}


LIBXSMM_API_INTERN LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512)
libxsmm_dnn_err_t libxsmm_dnn_convolve_st_fwd_custom_custom_i16_i32(libxsmm_dnn_layer* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
#if defined(LIBXSMM_INTRINSICS_AVX512) /*__AVX512F__*/
  typedef short element_input_type;
  typedef int element_output_type;
  typedef short element_filter_type;
  typedef libxsmm_wconvfunction libxsmm_convfunction;
# include "template/libxsmm_dnn_convolve_st_fwd_custom_custom.tpl.c"
#else /* should not happen */
  LIBXSMM_UNUSED(handle); LIBXSMM_UNUSED(start_thread); LIBXSMM_UNUSED(tid);
  status = LIBXSMM_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}


LIBXSMM_API_INTERN LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512)
libxsmm_dnn_err_t libxsmm_dnn_convolve_st_fwd_custom_custom_i16_f32(libxsmm_dnn_layer* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
#if defined(LIBXSMM_INTRINSICS_AVX512) /*__AVX512F__*/
  typedef short element_input_type;
  typedef float element_output_type;
  typedef short element_filter_type;
  typedef libxsmm_wsconvfunction libxsmm_convfunction;
# include "template/libxsmm_dnn_convolve_st_fwd_custom_custom.tpl.c"
#else /* should not happen */
  LIBXSMM_UNUSED(handle); LIBXSMM_UNUSED(start_thread); LIBXSMM_UNUSED(tid);
  status = LIBXSMM_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}


LIBXSMM_API_INTERN LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512)
libxsmm_dnn_err_t libxsmm_dnn_convolve_st_fwd_custom_custom_i8_i32(libxsmm_dnn_layer* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
#if defined(LIBXSMM_INTRINSICS_AVX512) /*__AVX512F__*/
  typedef unsigned char element_input_type;
  typedef int element_output_type;
  typedef char element_filter_type;
  typedef libxsmm_budconvfunction libxsmm_convfunction;
# include "template/libxsmm_dnn_convolve_st_fwd_custom_custom.tpl.c"
#else /* should not happen */
  LIBXSMM_UNUSED(handle); LIBXSMM_UNUSED(start_thread); LIBXSMM_UNUSED(tid);
  status = LIBXSMM_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}


LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_convolve_st_fwd_custom_custom(libxsmm_dnn_layer* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  /* check if we have input, output and filter */
  if (handle->reg_input == 0 || handle->reg_output == 0 || handle->reg_filter == 0) {
    status = LIBXSMM_DNN_ERR_DATA_NOT_BOUND;
    return status;
  }

  /* check if we have a kernel JITed */
  if ( handle->use_fwd_generic != 0 ) {
    if (handle->datatype_in == LIBXSMM_DNN_DATATYPE_F32 && handle->datatype_out == LIBXSMM_DNN_DATATYPE_F32 ) {
      const libxsmm_blasint ldx = (libxsmm_blasint)(handle->desc.v*handle->ifmblock);
      typedef float element_input_type;
      typedef float element_output_type;
      typedef float element_filter_type;
      typedef libxsmm_smmfunction gemm_function;
      /* let's do a ofmblock x ofw_rb x ifmblock GEMM :-) or in other words M=nbOfm, N=ofw, K=nbIfm (col-major) */
      gemm_function gemm_kernel = libxsmm_smmdispatch(handle->ofmblock, handle->ofw, handle->ifmblock, NULL, &ldx, NULL, NULL, NULL, NULL, NULL);
# include "template/libxsmm_dnn_convolve_st_fwd_custom_custom_generic.tpl.c"
#if 0
    } else if (handle->datatype_in == LIBXSMM_DNN_DATATYPE_BF16 && handle->datatype_out == LIBXSMM_DNN_DATATYPE_BF16 ) {
      const int ldx = (int)(handle->desc.v*handle->ifmblock*handle->fm_lp_block);
      typedef libxsmm_bfloat16 element_input_type;
      typedef libxsmm_bfloat16 element_output_type;
      typedef libxsmm_bfloat16 element_filter_type;
      typedef libxsmm_smmfunction gemm_function;
      /* let's do a ofmblock x ofw_rb x ifmblock GEMM :-) or in other words M=nbOfm, N=ofw, K=nbIfm (col-major) */
      gemm_function gemm_kernel = libxsmm_smmdispatch(handle->ofmblock*handle->fm_lp_block, handle->ofw, handle->ifmblock*handle->fm_lp_block, NULL, &ldx, NULL, NULL, NULL, NULL, NULL);
# include "template/libxsmm_dnn_convolve_st_fwd_custom_custom_generic_bf16.tpl.c"
    } else if (handle->datatype_in == LIBXSMM_DNN_DATATYPE_I16 && handle->datatype_out == LIBXSMM_DNN_DATATYPE_I32 ) {
      typedef short element_input_type;
      typedef int element_output_type;
      typedef short element_filter_type;
# include "template/libxsmm_dnn_convolve_st_fwd_custom_custom_generic.tpl.c"
    } else if (handle->datatype_in ==  LIBXSMM_DNN_DATATYPE_I8 && handle->datatype_out == LIBXSMM_DNN_DATATYPE_I32 && (handle->desc.options & LIBXSMM_DNN_CONV_OPTION_ACTIVATION_UNSIGNED) > 0 ) {
      typedef unsigned char element_input_type;
      typedef int element_output_type;
      typedef char element_filter_type;
# include "template/libxsmm_dnn_convolve_st_fwd_custom_custom_generic.tpl.c"
      }
#endif
    } else {
      status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  }
  else {
    if (handle->datatype_in == LIBXSMM_DNN_DATATYPE_F32 && handle->datatype_out == LIBXSMM_DNN_DATATYPE_F32 ) {
      status = libxsmm_dnn_convolve_st_fwd_custom_custom_f32_f32( handle, start_thread, tid);
    } else if (handle->datatype_in == LIBXSMM_DNN_DATATYPE_BF16 && handle->datatype_out == LIBXSMM_DNN_DATATYPE_BF16 ) {
      status = libxsmm_dnn_convolve_st_fwd_custom_custom_bf16_bf16( handle, start_thread, tid);
    } else if (handle->datatype_in ==  LIBXSMM_DNN_DATATYPE_I16 && handle->datatype_out == LIBXSMM_DNN_DATATYPE_I32 ) {
      status = libxsmm_dnn_convolve_st_fwd_custom_custom_i16_i32( handle, start_thread, tid);
    } else if (handle->datatype_in ==  LIBXSMM_DNN_DATATYPE_I16 && handle->datatype_out == LIBXSMM_DNN_DATATYPE_F32 ) {
      status = libxsmm_dnn_convolve_st_fwd_custom_custom_i16_f32( handle, start_thread, tid);
    } else if (handle->datatype_in == LIBXSMM_DNN_DATATYPE_I8 && handle->datatype_out == LIBXSMM_DNN_DATATYPE_I32 && (handle->desc.options & LIBXSMM_DNN_CONV_OPTION_ACTIVATION_UNSIGNED) > 0 ) {
      status = libxsmm_dnn_convolve_st_fwd_custom_custom_i8_i32( handle, start_thread, tid);
    } else {
      status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  }

  return status;
}


LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_convolve_st_fwd_nhwc_custom(libxsmm_dnn_layer* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  /* check if we have input, output and filter */
  if (handle->reg_input == 0 || handle->reg_output == 0 || handle->reg_filter == 0) {
    status = LIBXSMM_DNN_ERR_DATA_NOT_BOUND;
    return status;
  }

  /* check if we have a kernel JITed */
  if ( handle->use_fwd_generic != 0 ) {
    if (handle->datatype_in == LIBXSMM_DNN_DATATYPE_F32 && handle->datatype_out == LIBXSMM_DNN_DATATYPE_F32 ) {
      const libxsmm_blasint lda = (libxsmm_blasint)(handle->ofmblock);
      const libxsmm_blasint ldb = (libxsmm_blasint)((handle->desc.pad_h == handle->desc.pad_h_in && handle->desc.pad_w == handle->desc.pad_w_in)
                        ? (handle->desc.v*handle->blocksifm*handle->ifmblock) : (handle->desc.v*handle->ifmblock));
      const libxsmm_blasint ldc = (libxsmm_blasint)(handle->blocksofm*handle->ofmblock);
      typedef float element_input_type;
      typedef float element_output_type;
      typedef float element_filter_type;
      typedef libxsmm_smmfunction gemm_function;
      /* let's do a ofmblock x ofw_rb x ifmblock GEMM :-) or in other words M=nbOfm, N=ofw, K=nbIfm (col-major) */
      gemm_function gemm_kernel = libxsmm_smmdispatch(handle->ofmblock, handle->ofw, handle->ifmblock, &lda, &ldb, &ldc, NULL, NULL, NULL, NULL);
#define LIBXSMM_DNN_TPL_FWD_DIRECT_GENERIC_NHWC_CUSTOM
# include "template/libxsmm_dnn_convolve_st_fwd_nhwc_custom-rsck_generic.tpl.c"
#undef LIBXSMM_DNN_TPL_FWD_DIRECT_GENERIC_NHWC_CUSTOM
    } else {
      status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  } else {
    /* shouldn't happen */
  }

  return status;
}


LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_convolve_st_fwd_nhwc_rsck(libxsmm_dnn_layer* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  /* check if we have input, output and filter */
  if (handle->reg_input == 0 || handle->reg_output == 0 || handle->reg_filter == 0) {
    status = LIBXSMM_DNN_ERR_DATA_NOT_BOUND;
    return status;
  }

  /* check if we have a kernel JITed */
  if ( handle->use_fwd_generic != 0 ) {
    if (handle->datatype_in == LIBXSMM_DNN_DATATYPE_F32 && handle->datatype_out == LIBXSMM_DNN_DATATYPE_F32 ) {
      const libxsmm_blasint lda = (libxsmm_blasint)(handle->blocksofm*handle->ofmblock);
      const libxsmm_blasint ldb = (libxsmm_blasint)((handle->desc.pad_h == handle->desc.pad_h_in && handle->desc.pad_w == handle->desc.pad_w_in)
                        ? (handle->desc.v*handle->blocksifm*handle->ifmblock) : (handle->desc.v*handle->ifmblock));
      const libxsmm_blasint ldc = (libxsmm_blasint)(handle->blocksofm*handle->ofmblock);
      typedef float element_input_type;
      typedef float element_output_type;
      typedef float element_filter_type;
      typedef libxsmm_smmfunction gemm_function;
      /* let's do a ofmblock x ofw_rb x ifmblock GEMM :-) or in other words M=nbOfm, N=ofw, K=nbIfm (col-major) */
      gemm_function gemm_kernel = libxsmm_smmdispatch(handle->ofmblock, handle->ofw, handle->ifmblock, &lda, &ldb, &ldc, NULL, NULL, NULL, NULL);
#define LIBXSMM_DNN_TPL_FWD_DIRECT_GENERIC_NHWC_RSCK
# include "template/libxsmm_dnn_convolve_st_fwd_nhwc_custom-rsck_generic.tpl.c"
#undef LIBXSMM_DNN_TPL_FWD_DIRECT_GENERIC_NHWC_RSCK
    } else {
      status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  } else {
    /* shouldn't happen */
  }

  return status;
}

