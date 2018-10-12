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
#include "libxsmm_dnn_rnncell_forward.h"
#include "libxsmm_dnn_elementwise.h"
#include "libxsmm_main.h"
#include <libxsmm_intrinsics_x86.h>
#include <libxsmm.h>

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#include <string.h>
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif


LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_rnncell_st_fwd_nc_ck_f32_f32(libxsmm_dnn_rnncell* handle, int start_thread, int tid);
LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_rnncell_st_fwd_ncnc_kcck_f32_f32(libxsmm_dnn_rnncell* handle, int start_thread, int tid);


LIBXSMM_API_INTERN LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512)
libxsmm_dnn_err_t libxsmm_dnn_rnncell_st_fwd_nc_ck_f32_f32(libxsmm_dnn_rnncell* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
#if defined(LIBXSMM_INTRINSICS_AVX512) /*__AVX512F__*/
  typedef float element_input_type;
  typedef float element_output_type;
  typedef float element_filter_type;
  if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_RNN_RELU ) {
#define LIBXSMM_DNN_RNN_RELU_FWD
# include "template/libxsmm_dnn_rnncell_st_rnn_fwd_nc_ck_generic.tpl.c"
#undef LIBXSMM_DNN_RNN_RELU_FWD
  } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_RNN_SIGMOID ) {
#define LIBXSMM_DNN_RNN_SIGMOID_FWD
# include "template/libxsmm_dnn_rnncell_st_rnn_fwd_nc_ck_generic.tpl.c"
#undef LIBXSMM_DNN_RNN_SIGMOID_FWD
  } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_RNN_TANH ) {
#define LIBXSMM_DNN_RNN_TANH_FWD
# include "template/libxsmm_dnn_rnncell_st_rnn_fwd_nc_ck_generic.tpl.c"
#undef LIBXSMM_DNN_RNN_TANH_FWD
  } else {
    /* should not happen */
  }
# include "template/libxsmm_dnn_rnncell_st_rnn_fwd_nc_ck_generic.tpl.c"
#else /* should not happen */
  LIBXSMM_UNUSED(handle); LIBXSMM_UNUSED(start_thread); LIBXSMM_UNUSED(tid);
  status = LIBXSMM_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}


LIBXSMM_API_INTERN LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512)
libxsmm_dnn_err_t libxsmm_dnn_rnncell_st_fwd_ncnc_kcck_f32_f32(libxsmm_dnn_rnncell* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
#if defined(LIBXSMM_INTRINSICS_AVX512) /*__AVX512F__*/
  LIBXSMM_UNUSED(handle); LIBXSMM_UNUSED(start_thread); LIBXSMM_UNUSED(tid);
  status = LIBXSMM_DNN_ERR_NOT_IMPLEMENTED;
#if 0
  typedef float element_input_type;
  typedef float element_output_type;
  typedef float element_filter_type;
# include "template/libxsmm_dnn_rnncell_st_rnn_fwd_ncnc_kcck_generic.tpl.c"
#endif
  LIBXSMM_UNUSED(handle); LIBXSMM_UNUSED(start_thread); LIBXSMM_UNUSED(tid);
#else /* should not happen */
  LIBXSMM_UNUSED(handle); LIBXSMM_UNUSED(start_thread); LIBXSMM_UNUSED(tid);
  status = LIBXSMM_DNN_ERR_UNSUPPORTED_ARCH;
#endif
  return status;
}


LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_rnncell_st_fwd_nc_ck(libxsmm_dnn_rnncell* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  /* check if we have input, output and filter */
#if 0
  if (handle->? == 0 ) {
    status = LIBXSMM_DNN_ERR_DATA_NOT_BOUND;
    return status;
  }
#endif

  /* check if we have a kernel JITed */
  if ( handle->fwd_generic != 0 ) {
    if (handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_F32 && handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_F32 ) {
      typedef float element_input_type;
      typedef float element_output_type;
      typedef float element_filter_type;
      if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_RNN_RELU ) {
#define LIBXSMM_DNN_RNN_RELU_FWD
# include "template/libxsmm_dnn_rnncell_st_rnn_fwd_nc_ck_generic.tpl.c"
#undef LIBXSMM_DNN_RNN_RELU_FWD
      } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_RNN_SIGMOID ) {
#define LIBXSMM_DNN_RNN_SIGMOID_FWD
# include "template/libxsmm_dnn_rnncell_st_rnn_fwd_nc_ck_generic.tpl.c"
#undef LIBXSMM_DNN_RNN_SIGMOID_FWD
      } else if ( handle->desc.cell_type == LIBXSMM_DNN_RNNCELL_RNN_TANH ) {
#define LIBXSMM_DNN_RNN_TANH_FWD
# include "template/libxsmm_dnn_rnncell_st_rnn_fwd_nc_ck_generic.tpl.c"
#undef LIBXSMM_DNN_RNN_TANH_FWD
      } else {
        /* should not happen */
      }
    } else {
      status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  }
  else {
    if (handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_F32 && handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_F32 ) {
      status = libxsmm_dnn_rnncell_st_fwd_nc_ck_f32_f32( handle, start_thread, tid);
    } else {
      status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  }

  return status;
}


LIBXSMM_API_INTERN libxsmm_dnn_err_t libxsmm_dnn_rnncell_st_fwd_ncnc_kcck(libxsmm_dnn_rnncell* handle, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  /* check if we have input, output and filter */
#if 0
  if (handle->? == 0 ) {
    status = LIBXSMM_DNN_ERR_DATA_NOT_BOUND;
    return status;
  }
#endif

  /* check if we have a kernel JITed */
  if ( handle->fwd_generic != 0 ) {
    if (handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_F32 && handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_F32 ) {
#if 0
      typedef float element_input_type;
      typedef float element_output_type;
      typedef float element_filter_type;
# include "template/libxsmm_dnn_rnncell_st_rnn_fwd_ncnc_kcck_generic.tpl.c"
#endif
    } else {
      status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  }
  else {
    if (handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_F32 && handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_F32 ) {
      status = libxsmm_dnn_rnncell_st_fwd_ncnc_kcck_f32_f32( handle, start_thread, tid);
    } else {
      status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
      return status;
    }
  }

  return status;
}

