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
#include "libxsmm_dnn_conv_fwd_custom_custom.h"

LIBXSMM_INLINE LIBXSMM_RETARGETABLE void internal_convolve_st_fwd_custom_custom_fp32_fallback(libxsmm_dnn_conv_handle* handle, int start_thread, int tid, int num_threads)
{
  typedef float element_input_type;
  typedef float element_output_type;
  typedef float element_filter_type;
# include "template/libxsmm_dnn_convolve_st_fwd_custom_custom_fallback.tpl.c"
}

LIBXSMM_INLINE LIBXSMM_RETARGETABLE void internal_convolve_st_fwd_custom_custom_int16_fallback(libxsmm_dnn_conv_handle* handle, int start_thread, int tid, int num_threads)
{
  typedef short element_input_type;
  typedef int element_output_type;
  typedef short element_filter_type;
# include "template/libxsmm_dnn_convolve_st_fwd_custom_custom_fallback.tpl.c"
}

LIBXSMM_INLINE LIBXSMM_RETARGETABLE void internal_convolve_st_fwd_custom_custom_fp32_opt(libxsmm_dnn_conv_handle* handle, int start_thread, int tid, int num_threads)
{
  typedef float element_input_type;
  typedef float element_output_type;
  typedef float element_filter_type;
  typedef libxsmm_sconvfunction libxsmm_convfunction;
# include "template/libxsmm_dnn_convolve_st_fwd_custom_custom_opt.tpl.c"
}

LIBXSMM_INLINE LIBXSMM_RETARGETABLE void internal_convolve_st_fwd_custom_custom_int16_opt(libxsmm_dnn_conv_handle* handle, int start_thread, int tid, int num_threads)
{
  typedef short element_input_type;
  typedef int element_output_type;
  typedef short element_filter_type;
  typedef libxsmm_wconvfunction libxsmm_convfunction;
# include "template/libxsmm_dnn_convolve_st_fwd_custom_custom_opt.tpl.c"
}

LIBXSMM_INLINE LIBXSMM_RETARGETABLE void internal_convolve_st_fwd_custom_custom_fp32_opt_img_par(libxsmm_dnn_conv_handle* handle, int start_thread, int tid, int num_threads)
{
  typedef float element_input_type;
  typedef float element_output_type;
  typedef float element_filter_type;
  typedef libxsmm_sconvfunction libxsmm_convfunction;
# include "template/libxsmm_dnn_convolve_st_fwd_custom_custom_opt_img_par.tpl.c"
}

LIBXSMM_INLINE LIBXSMM_RETARGETABLE void internal_convolve_st_fwd_custom_custom_int16_opt_img_par(libxsmm_dnn_conv_handle* handle, int start_thread, int tid, int num_threads)
{
  typedef short element_input_type;
  typedef int element_output_type;
  typedef short element_filter_type;
  typedef libxsmm_wconvfunction libxsmm_convfunction;
# include "template/libxsmm_dnn_convolve_st_fwd_custom_custom_opt_img_par.tpl.c"
}

LIBXSMM_API_DEFINITION libxsmm_dnn_err_t libxsmm_dnn_convolve_st_fwd_custom_custom(libxsmm_dnn_conv_handle* handle, int start_thread, int tid, int num_threads)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  /* check if we have input, output and filter */
  if (handle->input == 0 || handle->output == 0 || handle->filter == 0) {
    status = LIBXSMM_DNN_ERR_DATA_NOT_BOUND;
    return status;
  }

  /* check if we have a kernel JITed */
  if (handle->code_fwd[0].xconv.sconv == 0) {
    if (1 == handle->desc.splits) {
      switch (handle->datatype) {
        case LIBXSMM_DNN_DATATYPE_F32: {
          internal_convolve_st_fwd_custom_custom_fp32_fallback(handle, start_thread, tid, num_threads);
        } break;
        case LIBXSMM_DNN_DATATYPE_I16: {
          internal_convolve_st_fwd_custom_custom_int16_fallback(handle, start_thread, tid, num_threads);
        } break;
        default: {
          status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
          return status;
        }
      }
    } else {
      status = LIBXSMM_DNN_ERR_GENERAL;
      return status;
    }
  }
  else {
    if (1 == handle->desc.splits) {
      switch (handle->datatype) {
        case LIBXSMM_DNN_DATATYPE_F32: {
          if (handle->desc.N*handle->blocksofm >= num_threads) {
            internal_convolve_st_fwd_custom_custom_fp32_opt(handle, start_thread, tid, num_threads);
          }
          else {
            internal_convolve_st_fwd_custom_custom_fp32_opt_img_par(handle, start_thread, tid, num_threads);
          }
        } break;
        case LIBXSMM_DNN_DATATYPE_I16: {
          if (handle->desc.N*handle->blocksofm >= num_threads) {
            internal_convolve_st_fwd_custom_custom_int16_opt(handle, start_thread, tid, num_threads);
          }
          else {
            internal_convolve_st_fwd_custom_custom_int16_opt_img_par(handle, start_thread, tid, num_threads);
          }
        } break;
        default: {
          status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
          return status;
        }
      }
    } else {
      status = LIBXSMM_DNN_ERR_GENERAL;
      return status;
    }
  }

  return status;
}
