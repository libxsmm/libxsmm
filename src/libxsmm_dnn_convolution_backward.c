/******************************************************************************
** Copyright (c) 2016-2017, Intel Corporation                                **
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
/* Rajkishore Barik, Ankush Mandal, Alexander Heinecke (Intel Corp.)
******************************************************************************/
#include "libxsmm_dnn_convolution_backward.h"
#include <libxsmm_intrinsics_x86.h>
#include "libxsmm_main.h"

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#include <string.h>
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif


LIBXSMM_API_DEFINITION libxsmm_dnn_err_t libxsmm_dnn_convolve_st_bwd_custom_custom(libxsmm_dnn_layer* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  /* check if we have input, output and filter */
  if (handle->grad_input == 0 || handle->grad_output == 0 || handle->reg_filter == 0 || handle->scratch1 == 0 ) {
    status = LIBXSMM_DNN_ERR_DATA_NOT_BOUND;
    return status;
  }

  /* for low/mixed precision we need some scratch to be bound */
  if ( (handle->datatype != handle->datatype_itm) && (handle->scratch7 == 0) ) {
    status = LIBXSMM_DNN_ERR_DATA_NOT_BOUND;
    return status;
  }

  /* check if we have a kernel JITed */
  if (handle->code_bwd[0].xconv.sconv == 0) {
    if (handle->datatype == LIBXSMM_DNN_DATATYPE_F32 && handle->datatype_itm == LIBXSMM_DNN_DATATYPE_F32 ) {
      typedef float element_input_type;
      typedef float element_output_type;
      typedef float element_filter_type;
      if (handle->padding_flag == 1) {
#define INPUT_PADDING
#include "template/libxsmm_dnn_convolve_st_bwd_custom_custom_fallback.tpl.c"
#undef INPUT_PADDING
      } else {
#include "template/libxsmm_dnn_convolve_st_bwd_custom_custom_fallback.tpl.c"
      }
    } else if (handle->datatype ==  LIBXSMM_DNN_DATATYPE_I16 && handle->datatype_itm == LIBXSMM_DNN_DATATYPE_I32 ) {
      typedef int element_input_type;
      typedef short element_output_type;
      typedef short element_filter_type;
      if (handle->padding_flag == 1) {
#define INPUT_PADDING
#include "template/libxsmm_dnn_convolve_st_bwd_custom_custom_fallback.tpl.c"
#undef INPUT_PADDING
      } else {
#include "template/libxsmm_dnn_convolve_st_bwd_custom_custom_fallback.tpl.c"
      }
    } else if (handle->datatype == LIBXSMM_DNN_DATATYPE_I8 && handle->datatype_itm == LIBXSMM_DNN_DATATYPE_I16 && (handle->desc.options & LIBXSMM_DNN_CONV_OPTION_ACTIVATION_UNSIGNED) > 0 ) {
      typedef unsigned short element_input_type;
      typedef char element_output_type;
      typedef char element_filter_type;
      if (handle->padding_flag == 1) {
#define INPUT_PADDING
#include "template/libxsmm_dnn_convolve_st_bwd_custom_custom_fallback.tpl.c"
#undef INPUT_PADDING
      } else {
#include "template/libxsmm_dnn_convolve_st_bwd_custom_custom_fallback.tpl.c"
      }
    } else if (handle->datatype == LIBXSMM_DNN_DATATYPE_I8 && handle->datatype_itm == LIBXSMM_DNN_DATATYPE_I32 && (handle->desc.options & LIBXSMM_DNN_CONV_OPTION_ACTIVATION_UNSIGNED) > 0 ) {
      typedef unsigned int element_input_type;
      typedef char element_output_type;
      typedef char element_filter_type;
      if (handle->padding_flag == 1) {
#define INPUT_PADDING
#include "template/libxsmm_dnn_convolve_st_bwd_custom_custom_fallback.tpl.c"
#undef INPUT_PADDING
      } else {
#include "template/libxsmm_dnn_convolve_st_bwd_custom_custom_fallback.tpl.c"
      }
    } else {
      status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  }
  else {
    if (handle->datatype == LIBXSMM_DNN_DATATYPE_F32 && handle->datatype_itm == LIBXSMM_DNN_DATATYPE_F32 ) {
#if 0
      if (handle->desc.N*handle->blocksifm >= handle->desc.threads) {
#endif
      typedef float element_input_type;
      typedef float element_output_type;
      typedef float element_filter_type;
      typedef libxsmm_sconvfunction libxsmm_convfunction;
      typedef libxsmm_smmfunction libxsmm_mmfunction;
      if (handle->padding_flag == 1) {
#define INPUT_PADDING
#include "template/libxsmm_dnn_convolve_st_bwd_custom_custom.tpl.c"
#undef INPUT_PADDING
      } else {
#include "template/libxsmm_dnn_convolve_st_bwd_custom_custom.tpl.c"
      }
#if 0
      }
      else {
        typedef float element_input_type;
        typedef float element_output_type;
        typedef float element_filter_type;
        typedef libxsmm_sconvfunction libxsmm_convfunction;
#include "template/libxsmm_dnn_convolve_st_bwd_custom_custom_img_par.tpl.c"
      }
#endif
    } else if (handle->datatype ==  LIBXSMM_DNN_DATATYPE_I16 && handle->datatype_itm == LIBXSMM_DNN_DATATYPE_I32 ) {
      typedef int element_input_type;
      typedef short element_output_type;
      typedef short element_filter_type;
      typedef libxsmm_wconvfunction_bwd libxsmm_convfunction;
      if (handle->padding_flag == 1) {
#define INPUT_PADDING
#include "template/libxsmm_dnn_convolve_st_bwd_custom_custom_1.tpl.c"
#undef INPUT_PADDING
      } else {
#include "template/libxsmm_dnn_convolve_st_bwd_custom_custom_1.tpl.c"
      }
    } else if (handle->datatype == LIBXSMM_DNN_DATATYPE_I8 && handle->datatype_itm == LIBXSMM_DNN_DATATYPE_I16 && (handle->desc.options & LIBXSMM_DNN_CONV_OPTION_ACTIVATION_UNSIGNED) > 0 ) {
      typedef unsigned short element_input_type;
      typedef char element_output_type;
      typedef char element_filter_type;
      typedef libxsmm_busconvfunction_bwd libxsmm_convfunction;
      if (handle->padding_flag == 1) {
#define INPUT_PADDING
#include "template/libxsmm_dnn_convolve_st_bwd_custom_custom_1.tpl.c"
#undef INPUT_PADDING
      } else {
#include "template/libxsmm_dnn_convolve_st_bwd_custom_custom_1.tpl.c"
      }
    } else if (handle->datatype == LIBXSMM_DNN_DATATYPE_I8 && handle->datatype_itm == LIBXSMM_DNN_DATATYPE_I32 && (handle->desc.options & LIBXSMM_DNN_CONV_OPTION_ACTIVATION_UNSIGNED) > 0 ) {
      typedef unsigned int element_input_type;
      typedef char element_output_type;
      typedef char element_filter_type;
      typedef libxsmm_budconvfunction_bwd libxsmm_convfunction;
      if (handle->padding_flag == 1) {
#define INPUT_PADDING
#include "template/libxsmm_dnn_convolve_st_bwd_custom_custom_1.tpl.c"
#undef INPUT_PADDING
      } else {
#include "template/libxsmm_dnn_convolve_st_bwd_custom_custom_1.tpl.c"
      }
    } else {
      status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  }

  return status;
}

LIBXSMM_API_DEFINITION libxsmm_dnn_err_t libxsmm_dnn_convolve_st_bwd_nhwc_rsck(libxsmm_dnn_layer* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  /* check if we have input, output and filter */
  if (handle->grad_input == 0 || handle->grad_output == 0 || handle->reg_filter == 0 || handle->scratch1 == 0) {
    status = LIBXSMM_DNN_ERR_DATA_NOT_BOUND;
    return status;
  }

  /* check if we have a kernel JITed */
  if (handle->code_bwd[0].xconv.sconv == 0) {
    if (handle->datatype == LIBXSMM_DNN_DATATYPE_F32 && handle->datatype_itm == LIBXSMM_DNN_DATATYPE_F32 ) {
      typedef float element_input_type;
      typedef float element_output_type;
      typedef float element_filter_type;
      if (handle->padding_flag == 1) {
#define INPUT_PADDING
#include "template/libxsmm_dnn_convolve_st_bwd_nhwc_rsck_fallback.tpl.c"
#undef INPUT_PADDING
      } else {
# include "template/libxsmm_dnn_convolve_st_bwd_nhwc_rsck_fallback.tpl.c"
      }
    } else {
      status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  }
  else {
    if (handle->datatype == LIBXSMM_DNN_DATATYPE_F32 && handle->datatype_itm == LIBXSMM_DNN_DATATYPE_F32 ) {
#if 0
      if (handle->desc.N*handle->blocksifm >= handle->desc.threads) {
#endif
        typedef float element_input_type;
        typedef float element_output_type;
        typedef float element_filter_type;
        typedef libxsmm_sconvfunction libxsmm_convfunction;
        if (handle->padding_flag == 1) {
#define INPUT_PADDING
#include "template/libxsmm_dnn_convolve_st_bwd_nhwc_rsck.tpl.c"
#undef INPUT_PADDING
        } else {
# include "template/libxsmm_dnn_convolve_st_bwd_nhwc_rsck.tpl.c"
        }
#if 0
      }
      else {
        typedef float element_input_type;
        typedef float element_output_type;
        typedef float element_filter_type;
        typedef libxsmm_sconvfunction libxsmm_convfunction;
# include "template/libxsmm_dnn_convolve_st_bwd_nhwc_rsck_img_par.tpl.c"
      }
#endif
    } else {
      status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  }

  return status;
}

LIBXSMM_API_DEFINITION libxsmm_dnn_err_t libxsmm_dnn_convolve_st_bwd_nhwc_custom(libxsmm_dnn_layer* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  /* check if we have input, output and filter */
  if (handle->grad_input == 0 || handle->grad_output == 0 || handle->reg_filter == 0 || handle->scratch1 == 0) {
    status = LIBXSMM_DNN_ERR_DATA_NOT_BOUND;
    return status;
  }

  /* check if we have a kernel JITed */
  if (handle->code_bwd[0].xconv.sconv == 0) {
    if (handle->datatype == LIBXSMM_DNN_DATATYPE_F32 && handle->datatype_itm == LIBXSMM_DNN_DATATYPE_F32 ) {
      typedef float element_input_type;
      typedef float element_output_type;
      typedef float element_filter_type;
      if (handle->padding_flag == 1) {
#define INPUT_PADDING
# include "template/libxsmm_dnn_convolve_st_bwd_nhwc_custom_fallback.tpl.c"
#undef INPUT_PADDING
      } else {
# include "template/libxsmm_dnn_convolve_st_bwd_nhwc_custom_fallback.tpl.c"
      }
    } else {
      status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  }
  else {
    if (handle->datatype == LIBXSMM_DNN_DATATYPE_F32 && handle->datatype_itm == LIBXSMM_DNN_DATATYPE_F32 ) {
#if 0
      if (handle->desc.N*handle->blocksifm >= handle->desc.threads) {
#endif
        typedef float element_input_type;
        typedef float element_output_type;
        typedef float element_filter_type;
        typedef libxsmm_sconvfunction libxsmm_convfunction;
        if (handle->padding_flag == 1) {
#define INPUT_PADDING
# include "template/libxsmm_dnn_convolve_st_bwd_nhwc_custom.tpl.c"
#undef INPUT_PADDING
        } else {
# include "template/libxsmm_dnn_convolve_st_bwd_nhwc_custom.tpl.c"
        }
#if 0
      }
      else {
        typedef float element_input_type;
        typedef float element_output_type;
        typedef float element_filter_type;
        typedef libxsmm_sconvfunction libxsmm_convfunction;
# include "template/libxsmm_dnn_convolve_st_bwd_nhwc_custom_img_par.tpl.c"
      }
#endif
    } else {
      status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  }

  return status;
}

