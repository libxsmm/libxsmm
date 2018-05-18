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
/* Alexander Heinecke, Evangelos Georganas, Hans Pabst (Intel Corp.)
******************************************************************************/
if (handle->custom_format_type == LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM_1 ) {
#if 0
  if ( 0 != handle->use_thread_private_jit )
#else
  LIBXSMM_ASSERT(0 != handle->use_thread_private_jit);
#endif
  {
    if (handle->datatype_in == LIBXSMM_DNN_DATATYPE_F32 && handle->datatype_out == LIBXSMM_DNN_DATATYPE_F32 ) {
      typedef float element_input_type;
      typedef float element_output_type;
      typedef float element_filter_type;
      typedef libxsmm_sconvfunction libxsmm_convfunction;
#     include "libxsmm_dnn_convolve_st_fwd_custom_custom_stream.tpl.c"
    }
    else if (handle->datatype_in == LIBXSMM_DNN_DATATYPE_BF16 && handle->datatype_out == LIBXSMM_DNN_DATATYPE_BF16 ) {
      typedef libxsmm_bfloat16 element_input_type;
      typedef libxsmm_bfloat16 element_output_type;
      typedef libxsmm_bfloat16 element_filter_type;
      typedef libxsmm_bf16convfunction libxsmm_convfunction;
#     include "libxsmm_dnn_convolve_st_fwd_custom_custom_stream.tpl.c"
    }
    else if (handle->datatype_in ==  LIBXSMM_DNN_DATATYPE_I16 && handle->datatype_out == LIBXSMM_DNN_DATATYPE_I32 ) {
      typedef short element_input_type;
      typedef int element_output_type;
      typedef short element_filter_type;
      typedef libxsmm_wconvfunction libxsmm_convfunction;
#     include "libxsmm_dnn_convolve_st_fwd_custom_custom_stream.tpl.c"
    }
    else if (handle->datatype_in ==  LIBXSMM_DNN_DATATYPE_I16 && handle->datatype_out == LIBXSMM_DNN_DATATYPE_F32 ) {
      typedef short element_input_type;
      typedef float element_output_type;
      typedef short element_filter_type;
      typedef libxsmm_wsconvfunction libxsmm_convfunction;
#     include "libxsmm_dnn_convolve_st_fwd_custom_custom_stream.tpl.c"
    }
    else if (handle->datatype_in == LIBXSMM_DNN_DATATYPE_I8 && handle->datatype_out == LIBXSMM_DNN_DATATYPE_I32 && (handle->desc.options & LIBXSMM_DNN_CONV_OPTION_ACTIVATION_UNSIGNED) > 0 ) {
      typedef unsigned char element_input_type;
      typedef int element_output_type;
      typedef char element_filter_type;
      typedef libxsmm_budconvfunction libxsmm_convfunction;
#     include "libxsmm_dnn_convolve_st_fwd_custom_custom_stream.tpl.c"
    }
    else {
      return LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
    }
  }
#if 0
  else {
#   include "libxsmm_dnn_convolve_st_fwd_custom_custom_1.tpl.c"
  }
#endif
}
#if 0
else if (handle->custom_format_type == LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM_2) {
# include "libxsmm_dnn_convolve_st_fwd_custom_custom_2.tpl.c"
}
#endif
else { /* more custom format code */
  LIBXSMM_ASSERT(0);
  return LIBXSMM_DNN_ERR_INVALID_FORMAT_CONVOLVE;
}
