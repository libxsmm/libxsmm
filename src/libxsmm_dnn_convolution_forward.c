/******************************************************************************
** Copyright (c) 2016, Intel Corporation                                     **
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
/* Alexander Heinecke (Intel Corp.), Hans Pabst (Intel Corp.)
******************************************************************************/
#include "libxsmm_dnn_convolution_forward.h"
#include <libxsmm.h>

LIBXSMM_API_DEFINITION libxsmm_dnn_err_t libxsmm_dnn_convolve_st_fwd_custom_custom(libxsmm_dnn_conv_handle* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  /* check if we have input, output and filter */
  if (handle->input == 0 || handle->output == 0 || handle->filter == 0) {
    status = LIBXSMM_DNN_ERR_DATA_NOT_BOUND;
    return status;
  }

  /* check if we have a kernel JITed */
  if (handle->code_fwd[0].xconv.sconv == 0) {
    if (handle->datatype_in == LIBXSMM_DNN_DATATYPE_F32 && handle->datatype_out == LIBXSMM_DNN_DATATYPE_F32 ) {
      typedef float element_input_type;
      typedef float element_output_type;
      typedef float element_filter_type;
# include "template/libxsmm_dnn_convolve_st_fwd_custom_custom_fallback.tpl.c"
    } else if (handle->datatype_in == LIBXSMM_DNN_DATATYPE_I16 && handle->datatype_out == LIBXSMM_DNN_DATATYPE_I32 ) {
      typedef short element_input_type;
      typedef int element_output_type;
      typedef short element_filter_type;
# include "template/libxsmm_dnn_convolve_st_fwd_custom_custom_fallback.tpl.c"
    } else if (handle->datatype_in ==  LIBXSMM_DNN_DATATYPE_I8 && handle->datatype_out == LIBXSMM_DNN_DATATYPE_I16 && (handle->desc.options & LIBXSMM_DNN_CONV_OPTION_ACTIVATION_UNSIGNED) > 0 ) {
      typedef unsigned char element_input_type;
      typedef short element_output_type;
      typedef char element_filter_type;
# include "template/libxsmm_dnn_convolve_st_fwd_custom_custom_fallback.tpl.c"
    } else if (handle->datatype_in ==  LIBXSMM_DNN_DATATYPE_I8 && handle->datatype_out == LIBXSMM_DNN_DATATYPE_I32 && (handle->desc.options & LIBXSMM_DNN_CONV_OPTION_ACTIVATION_UNSIGNED) > 0 ) {
      typedef unsigned char element_input_type;
      typedef int element_output_type;
      typedef char element_filter_type;
# include "template/libxsmm_dnn_convolve_st_fwd_custom_custom_fallback.tpl.c"
    } else {
      status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  }
  else {
    if (handle->datatype_in == LIBXSMM_DNN_DATATYPE_F32 && handle->datatype_out == LIBXSMM_DNN_DATATYPE_F32 ) {
      if (handle->desc.N*handle->blocksofm >= handle->desc.threads) {
        typedef float element_input_type;
        typedef float element_output_type;
        typedef float element_filter_type;
        typedef libxsmm_sconvfunction libxsmm_convfunction;
        if (handle->desc.u == 1 && handle->desc.v == 1) {
#define LIBXSMM_DNN_CONV_FWD_INTERNAL_STRIDE_ONE
# include "template/libxsmm_dnn_convolve_st_fwd_custom_custom.tpl.c"
#undef LIBXSMM_DNN_CONV_FWD_INTERNAL_STRIDE_ONE
        } else {
# include "template/libxsmm_dnn_convolve_st_fwd_custom_custom.tpl.c"
        }
      }
      else {
        typedef float element_input_type;
        typedef float element_output_type;
        typedef float element_filter_type;
        typedef libxsmm_sconvfunction libxsmm_convfunction;
        if (handle->desc.u == 1 && handle->desc.v == 1) {
#define LIBXSMM_DNN_CONV_FWD_INTERNAL_STRIDE_ONE
# include "template/libxsmm_dnn_convolve_st_fwd_custom_custom_img_par.tpl.c"
#undef LIBXSMM_DNN_CONV_FWD_INTERNAL_STRIDE_ONE
        } else {
# include "template/libxsmm_dnn_convolve_st_fwd_custom_custom_img_par.tpl.c"
        }
      }
    } else if (handle->datatype_in ==  LIBXSMM_DNN_DATATYPE_I16 && handle->datatype_out == LIBXSMM_DNN_DATATYPE_I32 ) {
      if (handle->desc.N*handle->blocksofm >= handle->desc.threads) {
        typedef short element_input_type;
        typedef int element_output_type;
        typedef short element_filter_type;
        typedef libxsmm_wconvfunction libxsmm_convfunction;
        if (handle->desc.u == 1 && handle->desc.v == 1) {
#define LIBXSMM_DNN_CONV_FWD_INTERNAL_STRIDE_ONE
# include "template/libxsmm_dnn_convolve_st_fwd_custom_custom.tpl.c"
#undef LIBXSMM_DNN_CONV_FWD_INTERNAL_STRIDE_ONE
        } else {
# include "template/libxsmm_dnn_convolve_st_fwd_custom_custom.tpl.c"
        }
      }
      else {
        typedef short element_input_type;
        typedef int element_output_type;
        typedef short element_filter_type;
        typedef libxsmm_wconvfunction libxsmm_convfunction;
        if (handle->desc.u == 1 && handle->desc.v == 1) {
#define LIBXSMM_DNN_CONV_FWD_INTERNAL_STRIDE_ONE
# include "template/libxsmm_dnn_convolve_st_fwd_custom_custom_img_par.tpl.c"
#undef LIBXSMM_DNN_CONV_FWD_INTERNAL_STRIDE_ONE
        } else {
# include "template/libxsmm_dnn_convolve_st_fwd_custom_custom_img_par.tpl.c"
        }
      }
    } else if (handle->datatype_in == LIBXSMM_DNN_DATATYPE_I8 && handle->datatype_out == LIBXSMM_DNN_DATATYPE_I16 && (handle->desc.options & LIBXSMM_DNN_CONV_OPTION_ACTIVATION_UNSIGNED) > 0 ) {
      if (handle->desc.N*handle->blocksofm >= handle->desc.threads) {
        typedef unsigned char element_input_type;
        typedef short element_output_type;
        typedef char element_filter_type;
        typedef libxsmm_busconvfunction libxsmm_convfunction;
        if (handle->desc.u == 1 && handle->desc.v == 1) {
#define LIBXSMM_DNN_CONV_FWD_INTERNAL_STRIDE_ONE
# include "template/libxsmm_dnn_convolve_st_fwd_custom_custom.tpl.c"
#undef LIBXSMM_DNN_CONV_FWD_INTERNAL_STRIDE_ONE
        } else {
# include "template/libxsmm_dnn_convolve_st_fwd_custom_custom.tpl.c"
        }
      }
      else {
        typedef unsigned char element_input_type;
        typedef short element_output_type;
        typedef char element_filter_type;
        typedef libxsmm_busconvfunction libxsmm_convfunction;
        if (handle->desc.u == 1 && handle->desc.v == 1) {
#define LIBXSMM_DNN_CONV_FWD_INTERNAL_STRIDE_ONE
# include "template/libxsmm_dnn_convolve_st_fwd_custom_custom_img_par.tpl.c"
#undef LIBXSMM_DNN_CONV_FWD_INTERNAL_STRIDE_ONE
        } else {
# include "template/libxsmm_dnn_convolve_st_fwd_custom_custom_img_par.tpl.c"
        }
      }
    } else if (handle->datatype_in == LIBXSMM_DNN_DATATYPE_I8 && handle->datatype_out == LIBXSMM_DNN_DATATYPE_I32 && (handle->desc.options & LIBXSMM_DNN_CONV_OPTION_ACTIVATION_UNSIGNED) > 0 ) {
      if (handle->desc.N*handle->blocksofm >= handle->desc.threads) {
        typedef unsigned char element_input_type;
        typedef int element_output_type;
        typedef char element_filter_type;
        typedef libxsmm_budconvfunction libxsmm_convfunction;
        if (handle->desc.u == 1 && handle->desc.v == 1) {
#define LIBXSMM_DNN_CONV_FWD_INTERNAL_STRIDE_ONE
# include "template/libxsmm_dnn_convolve_st_fwd_custom_custom.tpl.c"
#undef LIBXSMM_DNN_CONV_FWD_INTERNAL_STRIDE_ONE
        } else {
# include "template/libxsmm_dnn_convolve_st_fwd_custom_custom.tpl.c"
        }
      }
      else {
        typedef unsigned char element_input_type;
        typedef int element_output_type;
        typedef char element_filter_type;
        typedef libxsmm_budconvfunction libxsmm_convfunction;
        if (handle->desc.u == 1 && handle->desc.v == 1) {
#define LIBXSMM_DNN_CONV_FWD_INTERNAL_STRIDE_ONE
# include "template/libxsmm_dnn_convolve_st_fwd_custom_custom_img_par.tpl.c"
#undef LIBXSMM_DNN_CONV_FWD_INTERNAL_STRIDE_ONE
        } else {
# include "template/libxsmm_dnn_convolve_st_fwd_custom_custom_img_par.tpl.c"
        }
      }
    } else {
      status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  }

  return status;
}


LIBXSMM_API_DEFINITION libxsmm_dnn_err_t libxsmm_dnn_convolve_st_fwd_nhwc_custom(libxsmm_dnn_conv_handle* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  /* check if we have input, output and filter */
  if (handle->input == 0 || handle->output == 0 || handle->filter == 0) {
    status = LIBXSMM_DNN_ERR_DATA_NOT_BOUND;
    return status;
  }

  /* check if we have a kernel JITed */
  if (handle->code_fwd[0].xconv.sconv == 0) {
    if (handle->datatype_in == LIBXSMM_DNN_DATATYPE_F32 && handle->datatype_out == LIBXSMM_DNN_DATATYPE_F32 ) {
      typedef float element_input_type;
      typedef float element_output_type;
      typedef float element_filter_type;
# include "template/libxsmm_dnn_convolve_st_fwd_nhwc_custom_fallback.tpl.c"
    } else {
      status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  } else {
    if (handle->datatype_in == LIBXSMM_DNN_DATATYPE_F32 && handle->datatype_out == LIBXSMM_DNN_DATATYPE_F32 ) {
      if (handle->desc.N*handle->blocksofm >= handle->desc.threads) {
        typedef float element_input_type;
        typedef float element_output_type;
        typedef float element_filter_type;
        typedef libxsmm_sconvfunction libxsmm_convfunction;
# include "template/libxsmm_dnn_convolve_st_fwd_nhwc_custom.tpl.c"
      }
      else {
        typedef float element_input_type;
        typedef float element_output_type;
        typedef float element_filter_type;
        typedef libxsmm_sconvfunction libxsmm_convfunction;
# include "template/libxsmm_dnn_convolve_st_fwd_nhwc_custom_img_par.tpl.c"
      }
    } else {
      status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  }

  return status;
}


LIBXSMM_API_DEFINITION libxsmm_dnn_err_t libxsmm_dnn_convolve_st_fwd_nhwc_rsck(libxsmm_dnn_conv_handle* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  /* check if we have input, output and filter */
  if (handle->input == 0 || handle->output == 0 || handle->filter == 0) {
    status = LIBXSMM_DNN_ERR_DATA_NOT_BOUND;
    return status;
  }

  /* check if we have a kernel JITed */
  if (handle->code_fwd[0].xconv.sconv == 0) {
    if (handle->datatype_in == LIBXSMM_DNN_DATATYPE_F32 && handle->datatype_out == LIBXSMM_DNN_DATATYPE_F32 ) {
      typedef float element_input_type;
      typedef float element_output_type;
      typedef float element_filter_type;
# include "template/libxsmm_dnn_convolve_st_fwd_nhwc_rsck_fallback.tpl.c"
    } else {
      status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  } else {
    if (handle->datatype_in == LIBXSMM_DNN_DATATYPE_F32 && handle->datatype_out == LIBXSMM_DNN_DATATYPE_F32 ) {
      if (handle->desc.N*handle->blocksofm >= handle->desc.threads) {
        typedef float element_input_type;
        typedef float element_output_type;
        typedef float element_filter_type;
        typedef libxsmm_sconvfunction libxsmm_convfunction;
# include "template/libxsmm_dnn_convolve_st_fwd_nhwc_rsck.tpl.c"
      }
      else {
        typedef float element_input_type;
        typedef float element_output_type;
        typedef float element_filter_type;
        typedef libxsmm_sconvfunction libxsmm_convfunction;
# include "template/libxsmm_dnn_convolve_st_fwd_nhwc_rsck_img_par.tpl.c"
      }
    } else {
      status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  }

  return status;
}

