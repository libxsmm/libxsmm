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
/* Hans Pabst, Alexander Heinecke, Rajkishore Barik (Intel Corp.)
******************************************************************************/
#include <libxsmm.h>
#include <libxsmm_sync.h>
#include "libxsmm_main.h"
#include "libxsmm_dnn_handle.h"
#include "libxsmm_dnn_convolution_forward.h"
#include "libxsmm_dnn_convolution_backward.h"
#include "libxsmm_dnn_convolution_weight_update.h"
#include "libxsmm_dnn_convolution_winograd_forward.h"
#include "libxsmm_dnn_convolution_winograd_backward.h"
#include "libxsmm_dnn_convolution_winograd_weight_update.h"

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#include <stdlib.h>
#include <string.h>
#if defined(_OPENMP)
# include <omp.h>
#endif
#if !defined(NDEBUG)
# include <stdio.h>
#endif
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif


LIBXSMM_API_DEFINITION void libxsmm_dnn_init(int target_arch)
{
  libxsmm_dnn_convolve_winograd_init(target_arch);
}


LIBXSMM_API_DEFINITION void libxsmm_dnn_finalize(void)
{
  libxsmm_dnn_convolve_winograd_finalize();
}


LIBXSMM_API_DEFINITION const char* libxsmm_dnn_get_error(libxsmm_dnn_err_t code)
{
  switch (code) {
    case LIBXSMM_DNN_SUCCESS:
      return "LIBXSMM DNN Success!";
    case LIBXSMM_DNN_WARN_FALLBACK:
      return "LIBXSMM DNN Warning: Falling back to naive code as target is currently not supported by LIBXSMM!";
    case LIBXSMM_DNN_ERR_GENERAL:
      return "LIBXSMM DNN Error: General error occurred!";
    case LIBXSMM_DNN_ERR_CREATE_HANDLE:
      return "LIBXSMM DNN Error: Handle creation failed!";
    case LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE:
      return "LIBXSMM DNN Error: Requested datatype is not available!";
    case LIBXSMM_DNN_ERR_INVALID_BLOCKING:
      return "LIBXSMM DNN Error: Requested Input/Output buffer size cannot be blocked!";
    case LIBXSMM_DNN_ERR_INVALID_HANDLE:
      return "LIBXSMM DNN Error: An invalid handle was provided!";
    case LIBXSMM_DNN_ERR_DATA_NOT_BOUND:
      return "LIBXSMM DNN Error: Not all required sources and destinations have been bound to convolution!";
    case LIBXSMM_DNN_ERR_CREATE_BUFFER:
      return "LIBXSMM DNN Error: Layer creation failed!";
    case LIBXSMM_DNN_ERR_INVALID_BUFFER:
      return "LIBXSMM DNN Error: Invalid buffer was specified!";
    case LIBXSMM_DNN_ERR_CREATE_FILTER:
      return "LIBXSMM DNN Error: Filter creation failed!";
    case LIBXSMM_DNN_ERR_INVALID_FILTER:
      return "LIBXSMM DNN Error: Invalid filter was specified!";
    case LIBXSMM_DNN_ERR_CREATE_BIAS:
      return "LIBXSMM DNN Error: Bias creation failed!";
    case LIBXSMM_DNN_ERR_INVALID_BIAS:
      return "LIBXSMM DNN Error: Invalid Bias was specified";
    case LIBXSMM_DNN_ERR_MISMATCH_BUFFER:
      return "LIBXSMM DNN Error: Layer doesn't match handle it should be bind to!";
    case LIBXSMM_DNN_ERR_INVALID_HANDLE_BUFFER:
      return "LIBXSMM DNN Error: Invalid handle or buffer!";
    case LIBXSMM_DNN_ERR_MISMATCH_FILTER:
      return "LIBXSMM DNN Error: Filter does not match handle it should be bound to!";
    case LIBXSMM_DNN_ERR_INVALID_HANDLE_FILTER:
      return "LIBXSMM DNN Error: Invalid handle or filter!";
    case LIBXSMM_DNN_ERR_INVALID_KIND:
      return "LIBXSMM DNN Error: Invalid convolution kind!";
    case LIBXSMM_DNN_ERR_INVALID_FORMAT_NCHW:
      return "LIBXSMM DNN Error: NCHW format is currently not natively supported by LIBXSMM!";
    case LIBXSMM_DNN_ERR_UNSUPPORTED_DST_FORMAT:
      return "LIBXSMM DNN Error: Unsupported destination format when copying data!";
    case LIBXSMM_DNN_ERR_UNSUPPORTED_SRC_FORMAT:
      return "LIBXSMM DNN Error: Unsupported source format when copying data!";
    case LIBXSMM_DNN_ERR_INVALID_FORMAT_CONVOLVE:
      return "LIBXSMM DNN Error: Unsupported format when requesting a convolution!";
    case LIBXSMM_DNN_ERR_INVALID_FORMAT_KCRS:
      return "LIBXSMM DNN Error: KCRS format is currently not natively supported by LIBXSMM!";
    case LIBXSMM_DNN_ERR_INVALID_FORMAT_GENERAL:
      return "LIBXSMM DNN Error: Invalid format was specified!";
    case LIBXSMM_DNN_ERR_CREATE_LAYOUT:
      return "LIBXSMM DNN Error: Layout creation failed!";
    case LIBXSMM_DNN_ERR_INVALID_LAYOUT:
      return "LIBXSMM DNN Error: Invalid layout was specified!";
    case LIBXSMM_DNN_ERR_UNSUPPORTED_ARCH:
      return "LIBXSMM DNN Error: Unsupported architecture!";
    case LIBXSMM_DNN_ERR_SCRATCH_NOT_ALLOCED:
      return "LIBXSMM DNN Error: scratch binding failed as scratch was not allocated!";
    case LIBXSMM_DNN_ERR_UNKNOWN_BUFFER_TYPE:
      return "LIBXSMM DNN Error: an unknown buffer type was provided!";
    case LIBXSMM_DNN_ERR_UNKNOWN_FILTER_TYPE:
      return "LIBXSMM DNN Error: an unknown filter type was provided!";
    case LIBXSMM_DNN_ERR_INVALID_ALGO:
      return "LIBXSMM DNN Error: Invalid algorithm was specified!";
    case LIBXSMM_DNN_ERR_INVALID_PADDING:
      return "LIBXSMM DNN Error: Invalid padding was specified!";
    default:
      return "LIBXSMM DNN Error: Unknown error or warning occurred!";
  }
}


LIBXSMM_API_DEFINITION size_t libxsmm_dnn_typesize(libxsmm_dnn_datatype datatype)
{
  switch (datatype) {
    case LIBXSMM_DNN_DATATYPE_F32: return 4;
    case LIBXSMM_DNN_DATATYPE_I32: return 4;
    case LIBXSMM_DNN_DATATYPE_I16: return 2;
    case LIBXSMM_DNN_DATATYPE_I8:  return 1;
    /* no error expected as enumeration really arrives at an enum; compiler-checked */
    default: return 1;
  }
}


LIBXSMM_API_DEFINITION size_t libxsmm_dnn_get_simd_width(libxsmm_dnn_datatype datatype)
{
  size_t l_cl_width_bytes;
  if ( libxsmm_target_archid == LIBXSMM_X86_GENERIC ) {
    l_cl_width_bytes = libxsmm_dnn_typesize(datatype);
  } else if ( libxsmm_target_archid == LIBXSMM_X86_SSE3 ||
              libxsmm_target_archid == LIBXSMM_X86_SSE4 ) {
    l_cl_width_bytes = 16;
  } else if ( libxsmm_target_archid == LIBXSMM_X86_AVX2 ||
              libxsmm_target_archid == LIBXSMM_X86_AVX ) {
    l_cl_width_bytes = 32;
  } else {
    l_cl_width_bytes = 64;
  }

  return l_cl_width_bytes/libxsmm_dnn_typesize(datatype);
}


LIBXSMM_API_DEFINITION libxsmm_dnn_layer* libxsmm_dnn_create_conv_layer(
  libxsmm_dnn_conv_desc     conv_desc,
  libxsmm_dnn_err_t*        status)
{
  libxsmm_dnn_layer* handle = 0;
  *status = LIBXSMM_DNN_SUCCESS;

  /* currently we don't support NCHW */
  if ( (conv_desc.buffer_format & LIBXSMM_DNN_TENSOR_FORMAT_NCHW) > 0 ) {
    *status = LIBXSMM_DNN_ERR_INVALID_FORMAT_NCHW;
    return 0;
  }
  /* currently we don't support KCRS */
  if ( (conv_desc.buffer_format & LIBXSMM_DNN_TENSOR_FORMAT_KCRS) > 0 ) {
    *status = LIBXSMM_DNN_ERR_INVALID_FORMAT_KCRS;
    return 0;
  }

  handle = (libxsmm_dnn_layer*)malloc(sizeof(libxsmm_dnn_layer));

  if (0 != handle) {
    /* zero entire content; not only safer but also sets data and code pointers to NULL */
    memset(handle, 0, sizeof(*handle));
    /* initialize known handle components */
    handle->desc = conv_desc;
    handle->datatype = conv_desc.datatype;
    /* select the intermediate format, only applicable for integer types */
    if ( conv_desc.datatype == LIBXSMM_DNN_DATATYPE_F32 ) {
      handle->datatype_itm = conv_desc.datatype;
    } else if ( (conv_desc.datatype == LIBXSMM_DNN_DATATYPE_I16) || (conv_desc.datatype == LIBXSMM_DNN_DATATYPE_I8) ) {
      handle->datatype_itm = LIBXSMM_DNN_DATATYPE_I32;
      if ( (conv_desc.datatype == LIBXSMM_DNN_DATATYPE_I8) && ((conv_desc.options & LIBXSMM_DNN_CONV_OPTION_16BIT_ACC) > 0) ) {
        handle->datatype_itm = LIBXSMM_DNN_DATATYPE_I16;
      }
    } else {
      /* error */
    }
    handle->buffer_format = conv_desc.buffer_format;
    handle->filter_format = conv_desc.filter_format;
    handle->fuse_ops = conv_desc.fuse_ops;
    handle->options = conv_desc.options;
    /* derive additional values */
    handle->ifhp = conv_desc.H + 2*conv_desc.pad_h_in;
    handle->ifwp = conv_desc.W + 2*conv_desc.pad_w_in;
    handle->ofh = (conv_desc.H + 2*conv_desc.pad_h - conv_desc.R) / conv_desc.u + 1;
    handle->ofw = (conv_desc.W + 2*conv_desc.pad_w - conv_desc.S) / conv_desc.v + 1;
    handle->ofhp = handle->ofh + 2*conv_desc.pad_h_out;
    handle->ofwp = handle->ofw + 2*conv_desc.pad_w_out;
    handle->avx512avx2fallback = 0;
    handle->ifmblock = 1;
    handle->ofmblock = 1;
    handle->blocksifm = conv_desc.C;
    handle->blocksofm = conv_desc.K;
    handle->fwd_ofw_rb = 1;
    handle->fwd_ofw_rb_2 = 0;
    handle->fwd_ofh_rb = 1;
    handle->bwd_ofw_rb = 1;
    handle->bwd_ofh_rb = 1;
    handle->upd_ofw_rb = 1;
    handle->upd_ofh_rb = 1;
    handle->fm_lp_block = 1;
    handle->upd_use_thread_fil = 0;
    handle->upd_use_external_reduce = 0;
    handle->filter_transposed = 0;
    /* Set algorithm to use */
    if (conv_desc.algo == LIBXSMM_DNN_CONV_ALGO_AUTO) {
      if ( (((conv_desc.buffer_format & LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM) > 0) || ((conv_desc.buffer_format & LIBXSMM_DNN_TENSOR_FORMAT_NHWC) > 0)) &&
          ((conv_desc.filter_format & LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM) > 0) &&
          (3 == conv_desc.R) && (3 == conv_desc.S) &&
          (1 == conv_desc.u) && (1 == conv_desc.v) &&
          (0 == (conv_desc.C % 16)) && (0 == (conv_desc.K % 16)) &&
          (conv_desc.datatype  == LIBXSMM_DNN_DATATYPE_F32) ) {
        handle->algo = LIBXSMM_DNN_CONV_ALGO_WINOGRAD;
      } else {
        handle->algo = LIBXSMM_DNN_CONV_ALGO_DIRECT;
      }
    } else {
      handle->algo = conv_desc.algo;
    }
    if (handle->algo != LIBXSMM_DNN_CONV_ALGO_WINOGRAD && handle->algo != LIBXSMM_DNN_CONV_ALGO_DIRECT ) {
      *status = LIBXSMM_DNN_ERR_INVALID_ALGO;
      free(handle);
      handle = 0;
      return 0;
    }
    /* @TODO we might want to fall back to direct convolution if winograd fails */
    if ( handle->algo == LIBXSMM_DNN_CONV_ALGO_WINOGRAD ) {
      *status = libxsmm_dnn_internal_create_conv_handle_winograd_check( handle );
      if ( *status == LIBXSMM_DNN_WARN_FALLBACK ) {
        handle->algo = LIBXSMM_DNN_CONV_ALGO_DIRECT;
        *status = libxsmm_dnn_internal_create_conv_handle_direct( handle );
      }
    }
    else if ( handle->algo == LIBXSMM_DNN_CONV_ALGO_DIRECT ) {
      *status = libxsmm_dnn_internal_create_conv_handle_direct( handle );
    } else {
      assert(0/*should not happen*/);
    }
  }
  else {
    *status = LIBXSMM_DNN_ERR_CREATE_HANDLE;
  }
  return handle;
}


LIBXSMM_API_DEFINITION libxsmm_dnn_err_t libxsmm_dnn_destroy_conv_layer(const libxsmm_dnn_layer* handle)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  if (0 != handle) {
    /* deallocate data components; not an error to deallocate a NULL-pointer
       deallocate code known to be not registered; no index attached
       do not use libxsmm_release_kernel here! */

    if ( (libxsmm_target_archid == LIBXSMM_X86_AVX512_MIC  ||
          libxsmm_target_archid == LIBXSMM_X86_AVX512_KNM  ||
          libxsmm_target_archid == LIBXSMM_X86_AVX512_CORE ) && (handle->avx512avx2fallback == 0) ) {
      if (handle->custom_format_type != LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM_2) {
        libxsmm_free(handle->code_fwd[0].pmm);
      }
      libxsmm_free(handle->code_fwd[1].pmm);
      libxsmm_free(handle->code_fwd[2].pmm);
      libxsmm_free(handle->code_fwd[3].pmm);
      if (handle->custom_format_type != LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM_2) {
        libxsmm_free(handle->code_bwd[0].pmm);
      }
      if ((handle->filter_format == LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM) && (handle->buffer_format == LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM)) {
        libxsmm_free(handle->code_bwd[1].pmm);
        libxsmm_free(handle->code_bwd[2].pmm);
        libxsmm_free(handle->code_bwd[3].pmm);
      }
      if (handle->custom_format_type != LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM_2) {
        libxsmm_free(handle->code_upd[0].pmm);
      }
      if ((handle->filter_format == LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM) && (handle->buffer_format == LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM)) {
        libxsmm_free(handle->code_upd[1].pmm);
        libxsmm_free(handle->code_upd[2].pmm);
        libxsmm_free(handle->code_upd[3].pmm);
        libxsmm_free(handle->code_upd[4].pmm);
        libxsmm_free(handle->code_upd[5].pmm);
      }
    } else if ( (libxsmm_target_archid == LIBXSMM_X86_AVX2) || (handle->avx512avx2fallback != 0) ) {
      if (handle->custom_format_type != LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM_2) {
        libxsmm_free(handle->code_fwd[0].pmm);
      }
      if (handle->fwd_ofw_rb_2 != 0) {
        libxsmm_free(handle->code_fwd[1].pmm);
      }
      if (handle->custom_format_type != LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM_2) {
        libxsmm_free(handle->code_bwd[0].pmm);
      }
      if (handle->custom_format_type != LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM_2) {
        libxsmm_free(handle->code_upd[0].pmm);
      }
    } else {
      /* no kernel was JITed */
    }

    /* Deallocate barrier */
    if (handle->barrier != 0 ) { libxsmm_barrier_release((const libxsmm_barrier*)handle->barrier); }

    /*Deallocate scratch in handle*/
    libxsmm_free(handle->scratch1);
    libxsmm_free(handle->scratch3);
    libxsmm_free(handle->scratch4);
    if (handle->padding_flag) libxsmm_free(handle->scratch5);

    /* deallocate handle structure */
    free(/*remove constness*/(libxsmm_dnn_layer*)handle);
  }
  else {
    status = LIBXSMM_DNN_ERR_INVALID_HANDLE;
  }

  return status;
}


LIBXSMM_API_DEFINITION libxsmm_dnn_buffer* libxsmm_dnn_link_buffer(const libxsmm_dnn_layer* handle, const libxsmm_dnn_buffer_type type, const void* data, libxsmm_dnn_tensor_format in_format, libxsmm_dnn_err_t* status)
{
  return libxsmm_dnn_link_qbuffer(handle, type, data, 0, in_format, status);
}


LIBXSMM_API_DEFINITION libxsmm_dnn_buffer* libxsmm_dnn_link_qbuffer(const libxsmm_dnn_layer* handle, const libxsmm_dnn_buffer_type type, const void* data, const char exp, libxsmm_dnn_tensor_format in_format, libxsmm_dnn_err_t* status)
{
  libxsmm_dnn_buffer* buffer = (libxsmm_dnn_buffer*)malloc(sizeof(libxsmm_dnn_buffer));
  *status = LIBXSMM_DNN_SUCCESS;

  if (handle != 0 && buffer != 0 && data != 0) {
    /* set properties of the buffer according to convolution handle */
    if ( (type == LIBXSMM_DNN_REGULAR_INPUT) || (type == LIBXSMM_DNN_GRADIENT_INPUT) || (type == LIBXSMM_DNN_INPUT) ) {
      buffer->N = handle->desc.N;
      buffer->fmb = handle->blocksifm;
      buffer->bfm = handle->ifmblock;
      buffer->bimg = handle->nbImg;
      buffer->H = handle->ifhp;
      buffer->W = handle->ifwp;
      buffer->format = in_format;
      buffer->datatype = handle->datatype;
      buffer->lpb = handle->fm_lp_block;
      buffer->exp = exp;
      buffer->custom_format_type = handle->custom_format_type;
      /* NHWC */
      if ( ((handle->buffer_format & in_format) > 0) && ((in_format & LIBXSMM_DNN_TENSOR_FORMAT_NHWC ) > 0)  && ((in_format & LIBXSMM_DNN_TENSOR_FORMAT_PTR ) > 0) ) {
        buffer->data = (void*)data;
      /* custom LIBXSMM format */
      } else if ( ((handle->buffer_format & in_format) > 0) && ((in_format & LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM ) > 0)  && ((in_format & LIBXSMM_DNN_TENSOR_FORMAT_PTR ) > 0) ) {
        buffer->data = (void*)data;
        buffer->custom_format_type = handle->custom_format_type;
      } else {
        *status = LIBXSMM_DNN_ERR_UNSUPPORTED_SRC_FORMAT;
      }
    } else if ( (type == LIBXSMM_DNN_REGULAR_OUTPUT) || (type == LIBXSMM_DNN_GRADIENT_OUTPUT) || (type == LIBXSMM_DNN_OUTPUT) ) {
      /* set properties of the buffer according to convolution handle */
      buffer->N = handle->desc.N;
      buffer->fmb = handle->blocksofm;
      buffer->bfm = handle->ofmblock;
      buffer->bimg = handle->nbImg;
      buffer->H = handle->ofhp;
      buffer->W = handle->ofwp;
      buffer->format = in_format;
      buffer->datatype = handle->datatype;
      buffer->lpb = handle->fm_lp_block;
      buffer->exp = exp;
      buffer->custom_format_type = handle->custom_format_type;
      /* NHWC */
      if ( ((handle->buffer_format & in_format) > 0) && ((in_format & LIBXSMM_DNN_TENSOR_FORMAT_NHWC ) > 0)  && ((in_format & LIBXSMM_DNN_TENSOR_FORMAT_PTR ) > 0) ) {
        buffer->data = (void*)data;
      /* custom LIBXSMM format */
      } else if ( ((handle->buffer_format & in_format) > 0) && ((in_format & LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM ) > 0)  && ((in_format & LIBXSMM_DNN_TENSOR_FORMAT_PTR ) > 0) ) {
        buffer->data = (void*)data;
        buffer->custom_format_type = handle->custom_format_type;
      } else {
        *status = LIBXSMM_DNN_ERR_UNSUPPORTED_SRC_FORMAT;
      }
    } else {
      *status = LIBXSMM_DNN_ERR_UNKNOWN_BUFFER_TYPE;
    }
  }
  else {
    *status = LIBXSMM_DNN_ERR_CREATE_BUFFER;
  }

  if (*status != LIBXSMM_DNN_SUCCESS) {
    free((libxsmm_dnn_buffer*)buffer);
    buffer = 0;
  }

  return buffer;
}


LIBXSMM_API_DEFINITION libxsmm_dnn_tensor_datalayout* libxsmm_dnn_get_buffer_datalayout(const libxsmm_dnn_layer* handle, const libxsmm_dnn_buffer_type type, libxsmm_dnn_err_t* status) {
  libxsmm_dnn_tensor_datalayout* layout;

  *status = LIBXSMM_DNN_SUCCESS;
  layout = 0;

  if (handle != 0) {
    layout = (libxsmm_dnn_tensor_datalayout*) malloc(sizeof(libxsmm_dnn_tensor_datalayout));

    if (layout != 0) {
      memset(layout, 0, sizeof(libxsmm_dnn_tensor_datalayout));
      if ((handle->buffer_format & LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM) > 0) {
        if ( handle->datatype == LIBXSMM_DNN_DATATYPE_F32 ) {
          if (handle->custom_format_type == LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM_1) {
            layout->dim_type = (libxsmm_dnn_tensor_dimtype*) malloc(5*sizeof(libxsmm_dnn_tensor_dimtype));
            layout->dim_size = (unsigned int*) malloc(5*sizeof(unsigned int));
            if (0 != layout->dim_type && 0 != layout->dim_size) { /* TODO: handle the error */
              layout->num_dims = 5;
              layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
              layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_W;
              layout->dim_type[2] = LIBXSMM_DNN_TENSOR_DIMTYPE_H;
              layout->dim_type[3] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
              layout->dim_type[4] = LIBXSMM_DNN_TENSOR_DIMTYPE_N;
              if ( (type == LIBXSMM_DNN_REGULAR_INPUT) || (type == LIBXSMM_DNN_GRADIENT_INPUT) || (type == LIBXSMM_DNN_INPUT) ) {
                layout->dim_size[0] = handle->ifmblock;
                layout->dim_size[1] = handle->ifwp;
                layout->dim_size[2] = handle->ifhp;
                layout->dim_size[3] = handle->blocksifm;
                layout->dim_size[4] = handle->desc.N;
              } else if ( (type == LIBXSMM_DNN_REGULAR_OUTPUT) || (type == LIBXSMM_DNN_GRADIENT_OUTPUT) || (type == LIBXSMM_DNN_OUTPUT) ) {
                layout->dim_size[0] = handle->ofmblock;
                layout->dim_size[1] = handle->ofwp;
                layout->dim_size[2] = handle->ofhp;
                layout->dim_size[3] = handle->blocksofm;
                layout->dim_size[4] = handle->desc.N;
              } else {
                free(layout->dim_type);
                free(layout->dim_size);
                free(layout);
                layout = 0; /* make sure a NULL is returned */
                *status = LIBXSMM_DNN_ERR_UNKNOWN_BUFFER_TYPE;
              }
            }
          } else if (handle->custom_format_type == LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM_2) {
            layout->dim_type = (libxsmm_dnn_tensor_dimtype*) malloc(6*sizeof(libxsmm_dnn_tensor_dimtype));
            layout->dim_size = (unsigned int*) malloc(6*sizeof(unsigned int));
            if (0 != layout->dim_type && 0 != layout->dim_size) { /* TODO: handle the error */
              layout->num_dims = 6;
              layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
              layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_N;
              layout->dim_type[2] = LIBXSMM_DNN_TENSOR_DIMTYPE_W;
              layout->dim_type[3] = LIBXSMM_DNN_TENSOR_DIMTYPE_H;
              layout->dim_type[4] = LIBXSMM_DNN_TENSOR_DIMTYPE_N;
              layout->dim_type[5] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
              if ( (type == LIBXSMM_DNN_REGULAR_INPUT) || (type == LIBXSMM_DNN_GRADIENT_INPUT) || (type == LIBXSMM_DNN_INPUT) ) {
                layout->dim_size[0] = handle->ifmblock;
                layout->dim_size[1] = handle->nbImg;
                layout->dim_size[2] = handle->ifwp;
                layout->dim_size[3] = handle->ifhp;
                layout->dim_size[4] = handle->desc.N/handle->nbImg;
                layout->dim_size[5] = handle->blocksifm;
              } else if ( (type == LIBXSMM_DNN_REGULAR_OUTPUT) || (type == LIBXSMM_DNN_GRADIENT_OUTPUT) || (type == LIBXSMM_DNN_OUTPUT) ) {
                layout->dim_size[0] = handle->ofmblock;
                layout->dim_size[1] = handle->nbImg;
                layout->dim_size[2] = handle->ofwp;
                layout->dim_size[3] = handle->ofhp;
                layout->dim_size[4] = handle->desc.N/handle->nbImg;
                layout->dim_size[5] = handle->blocksofm;
              } else {
                free(layout->dim_type);
                free(layout->dim_size);
                free(layout);
                layout = 0; /* make sure a NULL is returned */
                *status = LIBXSMM_DNN_ERR_UNKNOWN_BUFFER_TYPE;
              }
            }
          } else {
            free(layout);
            layout = 0; /* make sure a NULL is returned */
            *status = LIBXSMM_DNN_ERR_UNKNOWN_BUFFER_TYPE;
          }
        } else if ( (handle->datatype == LIBXSMM_DNN_DATATYPE_I16) || (handle->datatype == LIBXSMM_DNN_DATATYPE_I8) ) {
          layout->dim_type = (libxsmm_dnn_tensor_dimtype*) malloc(6*sizeof(libxsmm_dnn_tensor_dimtype));
          layout->dim_size = (unsigned int*) malloc(6*sizeof(unsigned int));
          if (0 != layout->dim_type && 0 != layout->dim_size) { /* TODO: handle the error */
            layout->num_dims = 6;
            layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
            layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
            layout->dim_type[2] = LIBXSMM_DNN_TENSOR_DIMTYPE_W;
            layout->dim_type[3] = LIBXSMM_DNN_TENSOR_DIMTYPE_H;
            layout->dim_type[4] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
            layout->dim_type[5] = LIBXSMM_DNN_TENSOR_DIMTYPE_N;
            if ( (type == LIBXSMM_DNN_REGULAR_INPUT) || (type == LIBXSMM_DNN_GRADIENT_INPUT) || (type == LIBXSMM_DNN_INPUT) )   {
              layout->dim_size[0] = handle->fm_lp_block;
              layout->dim_size[1] = handle->ifmblock;
              layout->dim_size[2] = handle->ifwp;
              layout->dim_size[3] = handle->ifhp;
              layout->dim_size[4] = handle->blocksifm;
              layout->dim_size[5] = handle->desc.N;
            } else if ( (type == LIBXSMM_DNN_REGULAR_OUTPUT) || (type == LIBXSMM_DNN_GRADIENT_OUTPUT) || (type == LIBXSMM_DNN_OUTPUT) ) {
              layout->dim_size[0] = handle->fm_lp_block;
              layout->dim_size[1] = handle->ofmblock;
              layout->dim_size[2] = handle->ofwp;
              layout->dim_size[3] = handle->ofhp;
              layout->dim_size[4] = handle->blocksofm;
              layout->dim_size[5] = handle->desc.N;
            } else {
              free(layout->dim_type);
              free(layout->dim_size);
              free(layout);
              layout = 0; /* make sure a NULL is returned */
              *status = LIBXSMM_DNN_ERR_UNKNOWN_BUFFER_TYPE;
            }
          }
        } else {
          free(layout);
          layout = 0; /* make sure a NULL is returned */
          *status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
        }
      } else if ((handle->buffer_format & LIBXSMM_DNN_TENSOR_FORMAT_NHWC) > 0) {
        layout->dim_type = (libxsmm_dnn_tensor_dimtype*) malloc(4*sizeof(libxsmm_dnn_tensor_dimtype));
        layout->dim_size = (unsigned int*) malloc(4*sizeof(unsigned int));
        if (0 != layout->dim_type && 0 != layout->dim_size) { /* TODO: handle the error */
          layout->num_dims = 4;
          layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
          layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_W;
          layout->dim_type[2] = LIBXSMM_DNN_TENSOR_DIMTYPE_H;
          layout->dim_type[3] = LIBXSMM_DNN_TENSOR_DIMTYPE_N;
          if ( (type == LIBXSMM_DNN_REGULAR_INPUT) || (type == LIBXSMM_DNN_GRADIENT_INPUT) || (type == LIBXSMM_DNN_INPUT) )   {
            layout->dim_size[0] = handle->ifmblock * handle->blocksifm;
            layout->dim_size[1] = handle->ifwp;
            layout->dim_size[2] = handle->ifhp;
            layout->dim_size[3] = handle->desc.N;
          } else if ( (type == LIBXSMM_DNN_REGULAR_OUTPUT) || (type == LIBXSMM_DNN_GRADIENT_OUTPUT) || (type == LIBXSMM_DNN_OUTPUT) ) {
            layout->dim_size[0] = handle->ofmblock * handle->blocksofm;
            layout->dim_size[1] = handle->ofwp;
            layout->dim_size[2] = handle->ofhp;
            layout->dim_size[3] = handle->desc.N;
          } else {
            free(layout->dim_type);
            free(layout->dim_size);
            free(layout);
            layout = 0; /* make sure a NULL is returned */
            *status = LIBXSMM_DNN_ERR_UNKNOWN_BUFFER_TYPE;
          }
        }
      } else {
        free(layout);
        layout = 0; /* make sure a NULL is returned */
        *status = LIBXSMM_DNN_ERR_INVALID_FORMAT_GENERAL;
      }
    } else {
      *status = LIBXSMM_DNN_ERR_CREATE_LAYOUT;
    }
  }
  else {
    *status = LIBXSMM_DNN_ERR_INVALID_HANDLE;
  }

  return layout;
}


LIBXSMM_API_DEFINITION void* libxsmm_dnn_get_buffer_data_ptr(const libxsmm_dnn_buffer* buffer, libxsmm_dnn_err_t* status)
{
  *status = LIBXSMM_DNN_SUCCESS;

  if (0 != buffer) {
    return buffer->data;
  }
  else {
    *status = LIBXSMM_DNN_ERR_INVALID_BUFFER;
  }

  return 0;
}


LIBXSMM_API_DEFINITION char libxsmm_dnn_get_qbuffer_exp(const libxsmm_dnn_buffer* buffer, libxsmm_dnn_err_t* status)
{
  *status = LIBXSMM_DNN_SUCCESS;

  if (0 != buffer) {
    return buffer->exp;
  }
  else {
    *status = LIBXSMM_DNN_ERR_INVALID_BUFFER;
  }

  return 0;
}


LIBXSMM_API_DEFINITION libxsmm_dnn_err_t libxsmm_dnn_set_qbuffer_exp(libxsmm_dnn_buffer* buffer, const char exp)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  if (0 != buffer) {
    buffer->exp = exp;
  }
  else {
    status = LIBXSMM_DNN_ERR_INVALID_BUFFER;
  }

  return status;
}


LIBXSMM_API_DEFINITION libxsmm_dnn_err_t libxsmm_dnn_destroy_buffer(const libxsmm_dnn_buffer* buffer)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  if (0 != buffer) { /* it is not an error attempting to destroy a NULL-handle */
    /* deallocate data components; not an error to deallocate a NULL-pointer, just deallocate if it's LIBXSMM private data */
    if ( (buffer->format & LIBXSMM_DNN_TENSOR_FORMAT_PTR) == 0 ) {
      libxsmm_free(buffer->data);
    }
    /* deallocate handle structure */
    free(/*remove constness*/(libxsmm_dnn_buffer*)buffer);
  }
  else {
    status = LIBXSMM_DNN_ERR_INVALID_BUFFER;
  }

  return status;
}


LIBXSMM_API_DEFINITION libxsmm_dnn_filter* libxsmm_dnn_link_filter(const libxsmm_dnn_layer* handle, const libxsmm_dnn_filter_type type, const void* data,  libxsmm_dnn_tensor_format in_format, libxsmm_dnn_err_t* status)
{
  return libxsmm_dnn_link_qfilter(handle, type, data, 0, in_format, status);
}


LIBXSMM_API_DEFINITION libxsmm_dnn_filter* libxsmm_dnn_link_qfilter(const libxsmm_dnn_layer* handle, const libxsmm_dnn_filter_type type, const void* data, const char exp,  libxsmm_dnn_tensor_format in_format, libxsmm_dnn_err_t* status)
{
  libxsmm_dnn_filter* filter = (libxsmm_dnn_filter*)malloc(sizeof(libxsmm_dnn_filter));
  *status = LIBXSMM_DNN_SUCCESS;

  if (handle != 0 && filter != 0 && data != 0) {
    /* check for filter type */
    if ( (type != LIBXSMM_DNN_REGULAR_FILTER) && (type != LIBXSMM_DNN_GRADIENT_FILTER) && (type != LIBXSMM_DNN_FILTER) ) {
      *status = LIBXSMM_DNN_ERR_UNKNOWN_FILTER_TYPE;
      free(filter);
      filter = 0;
      return filter;
    }

    /* set properties of the buffer according to convolution handle */
    filter->ifmb = handle->blocksifm;
    filter->bifm = handle->ifmblock;
    filter->ofmb = handle->blocksofm*handle->fm_lp_block; /* @TODO this is a flaky hack */
    filter->bofm = handle->ofmblock;
    filter->R = handle->desc.R;
    filter->S = handle->desc.S;
    filter->format = in_format;
    filter->datatype = handle->datatype;
    filter->lpb = handle->fm_lp_block;
    filter->exp = exp;
    filter->custom_format_type = handle->custom_format_type;
    /* RSCK */
    if ( ((handle->filter_format & in_format) > 0) && ((in_format & LIBXSMM_DNN_TENSOR_FORMAT_RSCK ) > 0)  && ((in_format & LIBXSMM_DNN_TENSOR_FORMAT_PTR ) > 0) ) {
      filter->data = (void*)data;
    /* custom LIBXSMM format */
    } else if ( ((handle->filter_format & in_format) > 0) && ((in_format & LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM ) > 0)  && ((in_format & LIBXSMM_DNN_TENSOR_FORMAT_PTR ) > 0) ) {
      filter->data = (void*)data;
      filter->custom_format_type = handle->custom_format_type;
    } else {
      *status = LIBXSMM_DNN_ERR_UNSUPPORTED_SRC_FORMAT;
    }
  }
  else {
    *status = LIBXSMM_DNN_ERR_CREATE_FILTER;
  }

  if (*status != LIBXSMM_DNN_SUCCESS) {
    *status = LIBXSMM_DNN_ERR_CREATE_FILTER;
    free((libxsmm_dnn_filter*)filter);
    filter = 0;
  }

  return filter;
}


LIBXSMM_API_DEFINITION libxsmm_dnn_tensor_datalayout* libxsmm_dnn_get_filter_datalayout(const libxsmm_dnn_layer* handle, const libxsmm_dnn_filter_type type, libxsmm_dnn_err_t* status) {
  libxsmm_dnn_tensor_datalayout* layout;

  *status = LIBXSMM_DNN_SUCCESS;
  layout = 0;

  if (handle != 0) {
    /* check for filter type */
    if ( (type != LIBXSMM_DNN_REGULAR_FILTER) && (type != LIBXSMM_DNN_GRADIENT_FILTER) && (type != LIBXSMM_DNN_FILTER) ) {
      *status = LIBXSMM_DNN_ERR_UNKNOWN_FILTER_TYPE;
      return layout;
    }

    layout = (libxsmm_dnn_tensor_datalayout*) malloc(sizeof(libxsmm_dnn_tensor_datalayout));

    if (layout != 0) {
      memset(layout, 0, sizeof(libxsmm_dnn_tensor_datalayout));
      if ((handle->filter_format & LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM) > 0) {
        if ( (handle->datatype == LIBXSMM_DNN_DATATYPE_F32) ) {
          layout->dim_type = (libxsmm_dnn_tensor_dimtype*) malloc(6*sizeof(libxsmm_dnn_tensor_dimtype));
          layout->dim_size = (unsigned int*) malloc(6*sizeof(unsigned int));
          if (0 != layout->dim_type && 0 != layout->dim_size) { /* TODO: handle the error */
            layout->num_dims = 6;
            layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
            layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
            layout->dim_type[2] = LIBXSMM_DNN_TENSOR_DIMTYPE_S;
            layout->dim_type[3] = LIBXSMM_DNN_TENSOR_DIMTYPE_R;
            layout->dim_type[4] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
            layout->dim_type[5] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
            layout->dim_size[0] = handle->ofmblock;
            layout->dim_size[1] = handle->ifmblock;
            layout->dim_size[2] = handle->desc.S;
            layout->dim_size[3] = handle->desc.R;
            layout->dim_size[4] = handle->blocksofm;
            layout->dim_size[5] = handle->blocksofm;
          }
        } else if ( (handle->datatype == LIBXSMM_DNN_DATATYPE_I16) ||
                    (handle->datatype == LIBXSMM_DNN_DATATYPE_I8) ) {
          layout->dim_type = (libxsmm_dnn_tensor_dimtype*) malloc(7*sizeof(libxsmm_dnn_tensor_dimtype));
          layout->dim_size = (unsigned int*) malloc(7*sizeof(unsigned int));
          if (0 != layout->dim_type && 0 != layout->dim_size) { /* TODO: handle the error */
            layout->num_dims = 7;
            layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
            layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
            layout->dim_type[2] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
            layout->dim_type[3] = LIBXSMM_DNN_TENSOR_DIMTYPE_S;
            layout->dim_type[4] = LIBXSMM_DNN_TENSOR_DIMTYPE_R;
            layout->dim_type[5] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
            layout->dim_type[6] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
            layout->dim_size[0] = handle->fm_lp_block;
            layout->dim_size[1] = handle->ofmblock;
            layout->dim_size[2] = handle->ifmblock;
            layout->dim_size[3] = handle->desc.S;
            layout->dim_size[4] = handle->desc.R;
            layout->dim_size[5] = handle->blocksofm;
            layout->dim_size[6] = handle->blocksofm*handle->fm_lp_block;
          }
        } else {
          free(layout);
          layout = 0; /* make sure a NULL is returned */
          *status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
        }
      } else if ((handle->filter_format & LIBXSMM_DNN_TENSOR_FORMAT_RSCK) > 0) {
        layout->dim_type = (libxsmm_dnn_tensor_dimtype*) malloc(4*sizeof(libxsmm_dnn_tensor_dimtype));
        layout->dim_size = (unsigned int*) malloc(4*sizeof(unsigned int));
        if (0 != layout->dim_type && 0 != layout->dim_size) { /* TODO: handle the error */
          layout->num_dims = 4;
          layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
          layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
          layout->dim_type[2] = LIBXSMM_DNN_TENSOR_DIMTYPE_S;
          layout->dim_type[3] = LIBXSMM_DNN_TENSOR_DIMTYPE_R;
          layout->dim_size[0] = handle->ofmblock * handle->blocksofm;
          layout->dim_size[1] = handle->ofmblock * handle->blocksofm;
          layout->dim_size[2] = handle->desc.S;
          layout->dim_size[3] = handle->desc.K;
        }
      } else {
        free(layout);
        layout = 0; /* make sure a NULL is returned */
        *status = LIBXSMM_DNN_ERR_INVALID_FORMAT_GENERAL;
      }
    } else {
      *status = LIBXSMM_DNN_ERR_CREATE_LAYOUT;
    }
  }
  else {
    *status = LIBXSMM_DNN_ERR_INVALID_HANDLE;
  }

  return layout;
}


LIBXSMM_API_DEFINITION void* libxsmm_dnn_get_filter_data_ptr(const libxsmm_dnn_filter* filter, libxsmm_dnn_err_t* status)
{
  *status = LIBXSMM_DNN_SUCCESS;

  if (0 != filter) {
    return filter->data;
  }
  else {
    *status = LIBXSMM_DNN_ERR_INVALID_FILTER;
  }

  return 0;
}


LIBXSMM_API_DEFINITION char libxsmm_dnn_get_qfilter_exp(const libxsmm_dnn_filter* filter, libxsmm_dnn_err_t* status)
{
  *status = LIBXSMM_DNN_SUCCESS;

  if (0 != filter) {
    return filter->exp;
  }
  else {
    *status = LIBXSMM_DNN_ERR_INVALID_FILTER;
  }

  return 0;
}


LIBXSMM_API_DEFINITION libxsmm_dnn_err_t libxsmm_dnn_set_qfilter_exp(libxsmm_dnn_filter* filter, const char exp)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  if (0 != filter) {
    filter->exp = exp;
  }
  else {
    status = LIBXSMM_DNN_ERR_INVALID_FILTER;
  }

  return status;
}


LIBXSMM_API_DEFINITION libxsmm_dnn_err_t libxsmm_dnn_destroy_filter(const libxsmm_dnn_filter* filter)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  if (0 != filter) { /* it is not an error attempting to destroy a NULL-handle */
    /* deallocate data components; not an error to deallocate a NULL-pointer */
    if ( (filter->format & LIBXSMM_DNN_TENSOR_FORMAT_PTR) == 0 ) {
      libxsmm_free(filter->data);
    }
    /* deallocate handle structure */
    free(/*remove constness*/(libxsmm_dnn_filter*)filter);
  }
  else {
    status = LIBXSMM_DNN_ERR_INVALID_FILTER;
  }

  return status;
}


LIBXSMM_API_DEFINITION libxsmm_dnn_err_t libxsmm_dnn_destroy_bias(const libxsmm_dnn_bias* bias)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  if (0 != bias) { /* it is not an error attempting to destroy a NULL-handle */
    /* deallocate data components; not an error to deallocate a NULL-pointer */
    libxsmm_free(bias->data);
    /* deallocate handle structure */
    free(/*remove constness*/(libxsmm_dnn_bias*)bias);
  }
  else {
    status = LIBXSMM_DNN_ERR_INVALID_BIAS;
  }

  return status;
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_destroy_datalayout(libxsmm_dnn_tensor_datalayout* layout) {
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  if (0 != layout) {
    free(layout->dim_type);
    free(layout->dim_size);
    free(layout);
  }
  else {
    status = LIBXSMM_DNN_ERR_INVALID_LAYOUT;
  }

  return status;
}


LIBXSMM_API_DEFINITION libxsmm_dnn_err_t libxsmm_dnn_copyin_buffer(const libxsmm_dnn_buffer* buffer, const void* data, libxsmm_dnn_tensor_format in_format)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  if (0 != buffer) {
    switch (in_format) {
      case LIBXSMM_DNN_TENSOR_FORMAT_NCHW: {
        if ( (buffer->format & LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM) > 0 ) {
          switch (buffer->datatype) {
            case LIBXSMM_DNN_DATATYPE_F32: {
              typedef float element_type;
#             include "template/libxsmm_dnn_buffer_copy_in_nchw.tpl.c"
            } break;
            case LIBXSMM_DNN_DATATYPE_I32: {
              typedef int element_type;
#             include "template/libxsmm_dnn_buffer_copy_in_nchw.tpl.c"
            } break;
            case LIBXSMM_DNN_DATATYPE_I16: {
              typedef short element_type;
#             include "template/libxsmm_dnn_buffer_copy_in_nchw.tpl.c"
            } break;
            case LIBXSMM_DNN_DATATYPE_I8: {
              typedef char element_type;
#             include "template/libxsmm_dnn_buffer_copy_in_nchw.tpl.c"
            } break;
            default: {
              status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
            }
          }
        } else {
          status = LIBXSMM_DNN_ERR_UNSUPPORTED_DST_FORMAT;
        }
      } break;
      default: {
        status = LIBXSMM_DNN_ERR_UNSUPPORTED_SRC_FORMAT;
      }
    }
  }
  else {
    status = LIBXSMM_DNN_ERR_INVALID_BUFFER;
  }

  return status;
}


LIBXSMM_API_DEFINITION libxsmm_dnn_err_t libxsmm_dnn_zero_buffer(const libxsmm_dnn_buffer* buffer)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  if (0 != buffer) {
    const size_t size = (size_t)buffer->N * (size_t)buffer->fmb * (size_t)buffer->lpb
                      * (size_t)buffer->bfm * (size_t)buffer->H * (size_t)buffer->W;
    size_t i;
    /* use for-loops to potentially leverage NUMA in the future */
    switch (buffer->datatype) {
      case LIBXSMM_DNN_DATATYPE_F32: {
        float* fp32_data = (float*)buffer->data;
        for (i = 0; i < size; ++i) fp32_data[i] = 0.0f;
      } break;
      case LIBXSMM_DNN_DATATYPE_I32: {
        int* int32_data = (int*)buffer->data;
        for (i = 0; i < size; ++i) int32_data[i] = 0;
      } break;
      case LIBXSMM_DNN_DATATYPE_I16: {
        short* int16_data = (short*)buffer->data;
        for (i = 0; i < size; ++i) int16_data[i] = 0;
      } break;
      case LIBXSMM_DNN_DATATYPE_I8: {
        char* int8_data = (char*)buffer->data;
        for (i = 0; i < size; ++i) int8_data[i] = 0;
      } break;
      default: {
        status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
      }
    }
  }
  else {
    status = LIBXSMM_DNN_ERR_INVALID_BUFFER;
  }

  return status;
}


LIBXSMM_API_DEFINITION libxsmm_dnn_err_t libxsmm_dnn_copyout_buffer(const libxsmm_dnn_buffer* buffer, void* data, libxsmm_dnn_tensor_format out_format)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  if (0 != buffer) {
    switch (out_format) {
      case LIBXSMM_DNN_TENSOR_FORMAT_NCHW: {
        if ( (buffer->format & LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM) > 0 ) {
          switch (buffer->datatype) {
            case LIBXSMM_DNN_DATATYPE_F32: {
              typedef float element_type;
#             include "template/libxsmm_dnn_buffer_copy_out_nchw.tpl.c"
            } break;
            case LIBXSMM_DNN_DATATYPE_I32: {
              typedef int element_type;
#             include "template/libxsmm_dnn_buffer_copy_out_nchw.tpl.c"
            } break;
            case LIBXSMM_DNN_DATATYPE_I16: {
              typedef short element_type;
#             include "template/libxsmm_dnn_buffer_copy_out_nchw.tpl.c"
            } break;
            case LIBXSMM_DNN_DATATYPE_I8: {
              typedef char element_type;
#             include "template/libxsmm_dnn_buffer_copy_out_nchw.tpl.c"
            } break;
            default: {
              status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
            }
          }
        } else {
          status = LIBXSMM_DNN_ERR_UNSUPPORTED_SRC_FORMAT;
        }
      } break;
      default: {
        status = LIBXSMM_DNN_ERR_UNSUPPORTED_DST_FORMAT;
      }
    }
  }
  else {
    status = LIBXSMM_DNN_ERR_INVALID_BUFFER;
  }

  return status;
}


LIBXSMM_API_DEFINITION libxsmm_dnn_err_t libxsmm_dnn_copyin_filter(const libxsmm_dnn_filter* filter, const void* data, libxsmm_dnn_tensor_format in_format)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  if (0 != filter) {
    switch (in_format) {
      case LIBXSMM_DNN_TENSOR_FORMAT_KCRS: {
        if ( (filter->format & LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM) > 0 ) {
          switch (filter->datatype) {
            case LIBXSMM_DNN_DATATYPE_F32: {
              typedef float element_type;
#             include "template/libxsmm_dnn_filter_copy_in_kcrs.tpl.c"
            } break;
            case LIBXSMM_DNN_DATATYPE_I16: {
              typedef short element_type;
#             include "template/libxsmm_dnn_filter_copy_in_kcrs.tpl.c"
            } break;
            case LIBXSMM_DNN_DATATYPE_I8: {
              typedef char element_type;
#             include "template/libxsmm_dnn_filter_copy_in_kcrs.tpl.c"
            } break;
            default: {
              status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
            }
          }
        } else {
          status = LIBXSMM_DNN_ERR_UNSUPPORTED_DST_FORMAT;
        }
      } break;
      default: {
        status = LIBXSMM_DNN_ERR_UNSUPPORTED_SRC_FORMAT;
      }
    }
  }
  else {
    status = LIBXSMM_DNN_ERR_INVALID_FILTER;
  }

  return status;
}


LIBXSMM_API_DEFINITION libxsmm_dnn_err_t libxsmm_dnn_copyout_filter(const libxsmm_dnn_filter* filter, void* data, libxsmm_dnn_tensor_format out_format)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  if (0 != filter) {
    switch (out_format) {
      case LIBXSMM_DNN_TENSOR_FORMAT_KCRS: {
        if ( (filter->format & LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM) > 0 ) {
          switch (filter->datatype) {
            case LIBXSMM_DNN_DATATYPE_F32: {
              typedef float element_type;
#             include "template/libxsmm_dnn_filter_copy_out_kcrs.tpl.c"
            } break;
            case LIBXSMM_DNN_DATATYPE_I32: {
              typedef int element_type;
#             include "template/libxsmm_dnn_filter_copy_out_kcrs.tpl.c"
            } break;
            case LIBXSMM_DNN_DATATYPE_I16: {
              typedef short element_type;
#             include "template/libxsmm_dnn_filter_copy_out_kcrs.tpl.c"
            } break;
            case LIBXSMM_DNN_DATATYPE_I8: {
              typedef char element_type;
#             include "template/libxsmm_dnn_filter_copy_out_kcrs.tpl.c"
            } break;
            default: {
              status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
            }
          }
        } else {
          status = LIBXSMM_DNN_ERR_UNSUPPORTED_SRC_FORMAT;
        }
      } break;
      default: {
        status = LIBXSMM_DNN_ERR_UNSUPPORTED_DST_FORMAT;
      }
    }
  }
  else {
    status = LIBXSMM_DNN_ERR_INVALID_FILTER;
  }

  return status;
}


LIBXSMM_API_DEFINITION libxsmm_dnn_err_t libxsmm_dnn_zero_filter(const libxsmm_dnn_filter* filter)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  if (0 != filter) {
    const size_t size = (size_t)filter->lpb * (size_t)filter->ifmb * (size_t)filter->bifm
                      * (size_t)filter->ofmb * (size_t)filter->bofm * (size_t)filter->R * (size_t)filter->S;
    size_t i;
    /* use for-loops to potentially leverage NUMA in the future */
    switch (filter->datatype) {
      case LIBXSMM_DNN_DATATYPE_F32: {
        float* fp32_data = (float*)filter->data;
        for (i = 0; i < size; ++i) fp32_data[i] = 0.0f;
      } break;
      case LIBXSMM_DNN_DATATYPE_I32: {
        int* int32_data = (int*)filter->data;
        for (i = 0; i < size; ++i) int32_data[i] = 0;
      } break;
      case LIBXSMM_DNN_DATATYPE_I16: {
        short* int16_data = (short*)filter->data;
        for (i = 0; i < size; ++i) int16_data[i] = 0;
      } break;
      case LIBXSMM_DNN_DATATYPE_I8: {
        char* int8_data = (char*)filter->data;
        for (i = 0; i < size; ++i) int8_data[i] = 0;
      } break;
      default: {
        status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
      }
    }
  }
  else {
    status = LIBXSMM_DNN_ERR_INVALID_BUFFER;
  }

  return status;
}


#if 0
LIBXSMM_API_DEFINITION libxsmm_dnn_err_t libxsmm_dnn_copyin_bias(const libxsmm_dnn_bias* bias, const void* data)
{
  LIBXSMM_UNUSED(bias); LIBXSMM_UNUSED(data); /* TODO: libxsmm_dnn_copyin_input */
}


LIBXSMM_API_DEFINITION libxsmm_dnn_err_t libxsmm_dnn_copyout_bias(const libxsmm_dnn_bias* bias, void* data)
{
  LIBXSMM_UNUSED(bias); LIBXSMM_UNUSED(data); /* TODO: libxsmm_dnn_copyin_input */
}
#endif


LIBXSMM_API_DEFINITION libxsmm_dnn_err_t libxsmm_dnn_bind_buffer(libxsmm_dnn_layer* handle, const libxsmm_dnn_buffer* buffer, const libxsmm_dnn_buffer_type type)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  /* check for buffer type */
  if ( (type != LIBXSMM_DNN_REGULAR_INPUT) && (type != LIBXSMM_DNN_GRADIENT_INPUT) &&
       (type != LIBXSMM_DNN_REGULAR_OUTPUT) && (type != LIBXSMM_DNN_GRADIENT_OUTPUT) ) {
    status = LIBXSMM_DNN_ERR_UNKNOWN_BUFFER_TYPE;
    return status;
  }

  if (handle != 0 && buffer != 0) {
    if ( (type == LIBXSMM_DNN_REGULAR_INPUT) || (type == LIBXSMM_DNN_GRADIENT_INPUT) ) {
      /* check if format matches */
      if ( handle->desc.N == buffer->N
        && handle->ifwp == buffer->W
        && handle->ifhp == buffer->H
        && handle->ifmblock == buffer->bfm
        && handle->blocksifm == buffer->fmb
        && handle->datatype == buffer->datatype
        && handle->fm_lp_block == buffer->lpb
        && ((handle->buffer_format & buffer->format) > 0) )
      {
        if ( type == LIBXSMM_DNN_REGULAR_INPUT ) {
          handle->reg_input = (libxsmm_dnn_buffer*)buffer;
        } else {
          handle->grad_input = (libxsmm_dnn_buffer*)buffer;
        }
      } else {
        status = LIBXSMM_DNN_ERR_MISMATCH_BUFFER;
      }
    } else if ( (type == LIBXSMM_DNN_REGULAR_OUTPUT) || (type == LIBXSMM_DNN_GRADIENT_OUTPUT) ) {
      /* check if format matches */
      if ( handle->desc.N == buffer->N
        && handle->ofwp == buffer->W
        && handle->ofhp == buffer->H
        && handle->ofmblock == buffer->bfm
        && handle->blocksofm == buffer->fmb
        && handle->datatype == buffer->datatype
        && handle->fm_lp_block == buffer->lpb
        && ((handle->buffer_format & buffer->format) > 0) )
      {
        if ( type == LIBXSMM_DNN_REGULAR_OUTPUT ) {
          handle->reg_output = (libxsmm_dnn_buffer*)buffer;
        } else {
          handle->grad_output = (libxsmm_dnn_buffer*)buffer;
        }
      } else {
        status = LIBXSMM_DNN_ERR_MISMATCH_BUFFER;
      }
    } else {
      /* cannot happen */
    }
  }
  else {
    status = LIBXSMM_DNN_ERR_INVALID_HANDLE_BUFFER;
  }

  return status;
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_release_buffer(libxsmm_dnn_layer* handle, const libxsmm_dnn_buffer_type type)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  /* check for buffer type */
  if ( (type != LIBXSMM_DNN_REGULAR_INPUT) && (type != LIBXSMM_DNN_GRADIENT_INPUT) &&
       (type != LIBXSMM_DNN_REGULAR_OUTPUT) && (type != LIBXSMM_DNN_GRADIENT_OUTPUT) ) {
    status = LIBXSMM_DNN_ERR_UNKNOWN_BUFFER_TYPE;
    return status;
  }

  if (handle != 0) {
    if ( type == LIBXSMM_DNN_REGULAR_INPUT ) {
      handle->reg_input = 0;
    } else if ( type == LIBXSMM_DNN_GRADIENT_INPUT ) {
      handle->grad_input = 0;
    } else if ( type == LIBXSMM_DNN_REGULAR_OUTPUT ) {
      handle->reg_output = 0;
    } else if ( type == LIBXSMM_DNN_GRADIENT_OUTPUT ) {
      handle->grad_output = 0;
    } else {
      /* cannot happen */
    }
  } else {
    status = LIBXSMM_DNN_ERR_INVALID_HANDLE_BUFFER;
  }

  return status;
}


LIBXSMM_API_DEFINITION libxsmm_dnn_err_t libxsmm_dnn_bind_filter(libxsmm_dnn_layer* handle, const libxsmm_dnn_filter* filter, const libxsmm_dnn_filter_type type)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  /* check for filter type */
  if ( (type != LIBXSMM_DNN_REGULAR_FILTER) && (type != LIBXSMM_DNN_GRADIENT_FILTER) ) {
    status = LIBXSMM_DNN_ERR_UNKNOWN_FILTER_TYPE;
    return status;
  }

  if (handle != 0 && filter != 0) {
    /* check if format matches */
    if ( handle->desc.R == filter->R
      && handle->desc.S == filter->S
      && handle->ifmblock == filter->bifm
      && handle->blocksifm == filter->ifmb
      && handle->ofmblock == filter->bofm
      && (handle->blocksofm*handle->fm_lp_block) == filter->ofmb /* @TODO this check is flaky */
      && handle->fm_lp_block == filter->lpb
      && ((handle->filter_format & filter->format) > 0)
      && handle->datatype == filter->datatype)
    {
      if ( type == LIBXSMM_DNN_REGULAR_FILTER ) {
        handle->reg_filter = (libxsmm_dnn_filter*)filter;
      } else {
        handle->grad_filter = (libxsmm_dnn_filter*)filter;
      }
    }
    else {
      status = LIBXSMM_DNN_ERR_MISMATCH_FILTER;
    }
  }
  else {
    status = LIBXSMM_DNN_ERR_INVALID_HANDLE_FILTER;
  }

  return status;
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_release_filter(libxsmm_dnn_layer* handle, const libxsmm_dnn_filter_type type)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  /* check for filter type */
  if ( (type != LIBXSMM_DNN_REGULAR_FILTER) && (type != LIBXSMM_DNN_GRADIENT_FILTER) ) {
    status = LIBXSMM_DNN_ERR_UNKNOWN_FILTER_TYPE;
    return status;
  }

  if (handle != 0) {
    if ( type == LIBXSMM_DNN_REGULAR_FILTER ) {
      handle->reg_filter = 0;
    } else if ( type == LIBXSMM_DNN_GRADIENT_FILTER ) {
      handle->grad_filter = 0;
    } else {
      /* cannot happen */
    }
  } else {
    status = LIBXSMM_DNN_ERR_INVALID_HANDLE_BUFFER;
  }

  return status;
}


LIBXSMM_API_DEFINITION size_t libxsmm_dnn_get_scratch_size(const libxsmm_dnn_layer* handle, const libxsmm_dnn_compute_kind kind, libxsmm_dnn_err_t* status)
{
  size_t l_scratch_size = 0;
  size_t scratch5_size = 0;
  *status = LIBXSMM_DNN_SUCCESS;

  if (0 != handle) {
    if (handle->algo == LIBXSMM_DNN_CONV_ALGO_WINOGRAD) {
      l_scratch_size = 0;
      l_scratch_size += handle->scratch1_size + 64;
      l_scratch_size += handle->scratch3_size + 64;
      l_scratch_size += handle->scratch4_size + 64;
      l_scratch_size += handle->scratchIw_size + 64;
      l_scratch_size += handle->scratchOw_size + 64;
      if (libxsmm_target_archid == LIBXSMM_X86_AVX512_KNM) {
        l_scratch_size += handle->scratchVk_size + 64;
      }
      l_scratch_size += handle->scratchInput_size + 64;
    } else {
      switch (kind) {
        case LIBXSMM_DNN_COMPUTE_KIND_FWD: {
          if (handle->padding_flag == 1) {
            scratch5_size = handle->fwdbwd_scratch_size;
            l_scratch_size = scratch5_size + 64;
          }
          /* low precision intermediate buffer */
          if ( handle->datatype != handle->datatype_itm ) {
            l_scratch_size += handle->scratch6_size + 64;
          }
        } break;
        case LIBXSMM_DNN_COMPUTE_KIND_BWD: {
          /* we need filter for transpose, + 64 to do alignment while performing bind, scratch1 */
          l_scratch_size = handle->scratch1_size + 64;
          if (handle->padding_flag == 1) {
            scratch5_size = handle->fwdbwd_scratch_size;
            l_scratch_size += scratch5_size + 64;
          }
          /* low precision intermediate buffer for input */
          if (handle->datatype != handle->datatype_itm ) {
            l_scratch_size = handle->scratch7_size + 64;
          }
        } break;
        case LIBXSMM_DNN_COMPUTE_KIND_UPD: {
          /* we need a minibatch copy for transpose of input, scratch3 */
          l_scratch_size += handle->scratch3_size + 64;
          /* potentially we need thread-local filter copies, scratch4 */
          if (handle->upd_use_thread_fil == 1) {
            l_scratch_size += handle->scratch4_size + 64;
          }
          if (handle->padding_flag == 1) {
            scratch5_size = handle->minibatch_scratch_size;
            l_scratch_size += scratch5_size + 64;
          }
        } break;
        case LIBXSMM_DNN_COMPUTE_KIND_ALL: {
          /* we need filter for transpose, + 64 to do alignment while performing bind, scratch1 */
          l_scratch_size += handle->scratch1_size + 64;
          /* we need a minibatch copy for transpose of input, scratch3 */
          l_scratch_size += handle->scratch3_size + 64;
          /* potentially we need thread-local filter copies, scratch4 */
          if (handle->upd_use_thread_fil == 1) {
            l_scratch_size += handle->scratch4_size + 64;
          }
          /* low precision intermediate buffer */
          if ( handle->datatype != handle->datatype_itm ) {
            l_scratch_size += handle->scratch6_size + 64;
            l_scratch_size += handle->scratch7_size + 64;
          }
          if (handle->padding_flag == 1) {
            scratch5_size = handle->max_scratch5_size;
            l_scratch_size += scratch5_size + 64;
          }
        } break;
        default: {
          *status = LIBXSMM_DNN_ERR_INVALID_KIND;
        }
      }
    }
  } else {
    *status = LIBXSMM_DNN_ERR_INVALID_HANDLE;
  }

  return l_scratch_size;
}


LIBXSMM_API_DEFINITION libxsmm_dnn_err_t libxsmm_dnn_bind_scratch(libxsmm_dnn_layer* handle, const libxsmm_dnn_compute_kind kind, const void* scratch)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
  size_t address = (size_t)scratch;
  size_t offset = 0;
  size_t scratch5_size = 0;

  if (scratch == 0) {
    status = LIBXSMM_DNN_ERR_SCRATCH_NOT_ALLOCED;
    if ( (kind == LIBXSMM_DNN_COMPUTE_KIND_FWD) && (handle->datatype == handle->datatype_itm) ) {
      status = LIBXSMM_DNN_SUCCESS;
    }
    return status;
  }

  if (0 != handle) {
    if (handle->algo == LIBXSMM_DNN_CONV_ALGO_WINOGRAD) {
      /* + 64 to do alignment while performing bind, scratch1 */
      if (address % 64 == 0) {
        handle->scratch1 = (void*)address;
      } else {
        offset = (64 - address % 64);
        handle->scratch1 = (void*)(address+offset);
      }
      address += handle->scratch1_size + 64;
      if (address % 64 == 0) {
        handle->scratch3 = (void*)address;
      } else {
        offset = (64 - address % 64);
        handle->scratch3 = (void*)(address+offset);
      }
      address += handle->scratch3_size + 64;
      if (address % 64 == 0) {
        handle->scratch4 = (void*)address;
      } else {
        offset = (64 - address % 64);
        handle->scratch4 = (void*)(address+offset);
      }
      address += handle->scratch4_size + 64;
      if (address % 64 == 0) {
        handle->scratchIw = (void*)address;
      } else {
        offset = (64 - address % 64);
        handle->scratchIw = (void*)(address+offset);
      }
      address += handle->scratchIw_size + 64;
      if (address % 64 == 0) {
        handle->scratchOw = (void*)address;
      } else {
        offset = (64 - address % 64);
        handle->scratchOw = (void*)(address+offset);
      }
      address += handle->scratchOw_size + 64;
      if ( libxsmm_target_archid == LIBXSMM_X86_AVX512_KNM ) {
        if (address % 64 == 0) {
          handle->scratchVk = (void*)address;
        } else {
          offset = (64 - address % 64);
          handle->scratchVk = (void*)(address+offset);
        }
        address += handle->scratchVk_size + 64;
      }
      if (address % 64 == 0) {
        handle->scratchInput = (void*)address;
      } else {
        offset = (64 - address % 64);
        handle->scratchInput = (void*)(address+offset);
      }
      handle->scratchTemp = handle->scratch3;
      address += handle->scratchInput_size + 64;
    } else {
      switch (kind) {
        case LIBXSMM_DNN_COMPUTE_KIND_FWD: {
          if (handle->padding_flag == 1) {
            scratch5_size = handle->fwdbwd_scratch_size;;
            if (address % 64 == 0) {
              handle->scratch5 = (void*)address;
            } else {
              offset = (64 - address % 64);
              handle->scratch5 = (void*)(address+offset);
            }
            /* Initialize scratch5 to zero */
            memset(handle->scratch5, 0, scratch5_size);
            address += scratch5_size + 64;
          }
          if ( handle->datatype != handle->datatype_itm ) {
            if (address % 64 == 0) {
              handle->scratch6 = (void*)address;
            } else {
              offset = (64 - address % 64);
              handle->scratch6 = (void*)(address+offset);
            }
          }
        } break;
        case LIBXSMM_DNN_COMPUTE_KIND_BWD: {
          /* we need filter for transpose, + 64 to do alignment while performing bind, scratch1 */
          if (address % 64 == 0) {
            handle->scratch1 = (void*)address;
          } else {
            offset = (64 - address % 64);
            handle->scratch1 = (void*)(address+offset);
          }
          if (handle->padding_flag == 1) {
            scratch5_size = handle->fwdbwd_scratch_size;;
            address += handle->scratch1_size + 64;
            if (address % 64 == 0) {
              handle->scratch5 = (void*)address;
            } else {
              offset = (64 - address % 64);
              handle->scratch5 = (void*)(address+offset);
            }
            /* Initialize scratch5 to zero */
            memset(handle->scratch5, 0, scratch5_size);
          }
          if ( handle->datatype != handle->datatype_itm ) {
            if (address % 64 == 0) {
              handle->scratch7 = (void*)address;
            } else {
              offset = (64 - address % 64);
              handle->scratch7 = (void*)(address+offset);
            }
          }
        } break;
        case LIBXSMM_DNN_COMPUTE_KIND_UPD: {
          /* we need a minibatch copy for transpose of input, scratch3 */
          if (handle->padding_flag == 1) {
            scratch5_size = handle->minibatch_scratch_size;
            if (address % 64 == 0) {
              handle->scratch5 = (void*)address;
            } else {
              offset = (64 - address % 64);
              handle->scratch5 = (void*)(address+offset);
            }
            /* Initialize scratch5 to zero */
            memset(handle->scratch5, 0, scratch5_size);
            address += scratch5_size + 64;
          }
          if (address % 64 == 0) {
            handle->scratch3 = (void*)address;
          } else {
            offset = (64 - address % 64);
            handle->scratch3 = (void*)(address+offset);
          }
          /* potentially we need thread-local filter copies, scratch4 */
          if (handle->upd_use_thread_fil == 1) {
            address += handle->scratch3_size + 64;
            if (address % 64 == 0) {
              handle->scratch4 = (void*)address;
            } else {
              offset = (64 - address % 64);
              handle->scratch4 = (void*)(address+offset);
            }
            /* Initialize scratch4 to zero */
            memset(handle->scratch4, 0, handle->scratch4_size);
          }
        } break;
        case LIBXSMM_DNN_COMPUTE_KIND_ALL: {
          /* we need filter for transpose, + 64 to do alignment while performing bind, scratch1 */
          if (handle->padding_flag == 1) {
            scratch5_size = handle->max_scratch5_size;
          if (address % 64 == 0) {
              handle->scratch5 = (void*)address;
            } else {
              offset = (64 - address % 64);
              handle->scratch5 = (void*)(address+offset);
            }
            /* Initialize scratch5 to zero */
            memset(handle->scratch5, 0, scratch5_size);
            address += scratch5_size + 64;
          }
          if (address % 64 == 0) {
            handle->scratch1 = (void*)address;
          } else {
            offset = (64 - address % 64);
            handle->scratch1 = (void*)(address+offset);
          }
          address += handle->scratch1_size + 64;
          /* we need a minibatch copy for transpose of input, scratch3 */
          if (address % 64 == 0) {
            handle->scratch3 = (void*)address;
          } else {
            offset = (64 - address % 64);
            handle->scratch3 = (void*)(address+offset);
          }
          address += handle->scratch3_size + 64;
          /* potentially we need thread-local filter copies, scratch4 */
          if (handle->upd_use_thread_fil == 1) {
            if (address % 64 == 0) {
              handle->scratch4 = (void*)address;
            } else {
              offset = (64 - address % 64);
              handle->scratch4 = (void*)(address+offset);
            }
            /* Initialize scratch4 to zero */
            memset(handle->scratch4, 0, handle->scratch4_size);
            address += handle->scratch4_size + 64;
          }
          /* low precision intermediate buffer */
          if ( handle->datatype != handle->datatype_itm ) {
            if (address % 64 == 0) {
              handle->scratch6 = (void*)address;
            } else {
              offset = (64 - address % 64);
              handle->scratch6 = (void*)(address+offset);
            }
            address += handle->scratch6_size + 64;
            if (address % 64 == 0) {
              handle->scratch7 = (void*)address;
            } else {
              offset = (64 - address % 64);
              handle->scratch7 = (void*)(address+offset);
            }
            address += handle->scratch7_size + 64;
          }
        } break;
        default: {
          status = LIBXSMM_DNN_ERR_INVALID_KIND;
        }
      }
    }
  } else {
    status = LIBXSMM_DNN_ERR_INVALID_HANDLE;
  }

  return status;
}


LIBXSMM_API_DEFINITION libxsmm_dnn_err_t libxsmm_dnn_release_scratch(libxsmm_dnn_layer* handle, const libxsmm_dnn_compute_kind kind)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  if (0 != handle) {
    if (handle->algo == LIBXSMM_DNN_CONV_ALGO_WINOGRAD) {
      handle->scratch1 = 0;
      handle->scratch3 = 0;
      handle->scratch4 = 0;
      handle->scratchIw = 0;
      handle->scratchOw = 0;
      handle->scratchVk = 0;
      handle->scratchInput = 0;
      handle->scratchTemp  = 0;
    } else {
      switch (kind) {
        case LIBXSMM_DNN_COMPUTE_KIND_FWD: {
          handle->scratch5 = 0;
        } break;
        case LIBXSMM_DNN_COMPUTE_KIND_BWD: {
          handle->scratch1 = 0;
          handle->scratch5 = 0;
        } break;
        case LIBXSMM_DNN_COMPUTE_KIND_UPD: {
          handle->scratch3 = 0;
          handle->scratch4 = 0;
          handle->scratch5 = 0;
        } break;
        case LIBXSMM_DNN_COMPUTE_KIND_ALL: {
          handle->scratch1 = 0;
          handle->scratch3 = 0;
          handle->scratch4 = 0;
          handle->scratch5 = 0;
        } break;
        default: {
          status = LIBXSMM_DNN_ERR_INVALID_KIND;
        }
      }
    }
  } else {
    status = LIBXSMM_DNN_ERR_INVALID_HANDLE;
  }

  return status;
}


LIBXSMM_INLINE LIBXSMM_RETARGETABLE libxsmm_dnn_err_t internal_execute_st(libxsmm_dnn_layer* handle,
  libxsmm_dnn_compute_kind kind, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  if (0 != handle) {
    switch (handle->algo) {
      case LIBXSMM_DNN_CONV_ALGO_DIRECT: {
        switch (kind) {
          case LIBXSMM_DNN_COMPUTE_KIND_FWD: {
            switch (handle->buffer_format) {
              case LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM: {
                switch (handle->filter_format) {
                  case LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM: {
                    status = libxsmm_dnn_convolve_st_fwd_custom_custom(handle, start_thread, tid);
                  } break;
                  default: {
                    status = LIBXSMM_DNN_ERR_INVALID_FORMAT_CONVOLVE;
                  }
                }
              } break;
              case LIBXSMM_DNN_TENSOR_FORMAT_NHWC: {
                switch (handle->filter_format) {
                  case LIBXSMM_DNN_TENSOR_FORMAT_RSCK: {
                    status = libxsmm_dnn_convolve_st_fwd_nhwc_rsck(handle, start_thread, tid);
                  } break;
                  case LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM: {
                    status = libxsmm_dnn_convolve_st_fwd_nhwc_custom(handle, start_thread, tid);
                  } break;
                  default: {
                    status = LIBXSMM_DNN_ERR_INVALID_FORMAT_CONVOLVE;
                  }
                }
              } break;
              default: {
                status = LIBXSMM_DNN_ERR_INVALID_FORMAT_CONVOLVE;
              }
            }
          } break;
          case LIBXSMM_DNN_COMPUTE_KIND_BWD: {
            switch (handle->buffer_format) {
              case LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM: {
                switch (handle->filter_format) {
                  case LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM: {
                    status = libxsmm_dnn_convolve_st_bwd_custom_custom(handle, start_thread, tid);
                  } break;
                  default: {
                    status = LIBXSMM_DNN_ERR_INVALID_FORMAT_CONVOLVE;
                  }
                }
              } break;
              case LIBXSMM_DNN_TENSOR_FORMAT_NHWC: {
                switch (handle->filter_format) {
                  case LIBXSMM_DNN_TENSOR_FORMAT_RSCK: {
                    status = libxsmm_dnn_convolve_st_bwd_nhwc_rsck(handle, start_thread, tid);
                  } break;
                  case LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM: {
                    status = libxsmm_dnn_convolve_st_bwd_nhwc_custom(handle, start_thread, tid);
                  } break;
                  default: {
                    status = LIBXSMM_DNN_ERR_INVALID_FORMAT_CONVOLVE;
                  }
                }
              } break;
              default: {
                status = LIBXSMM_DNN_ERR_INVALID_FORMAT_CONVOLVE;
              }
            }
          } break;
          case LIBXSMM_DNN_COMPUTE_KIND_UPD: {
            switch (handle->buffer_format) {
              case LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM: {
                 switch (handle->filter_format) {
                  case LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM: {
                    status = libxsmm_dnn_convolve_st_upd_custom_custom(handle, start_thread, tid);
                  } break;
                  default: {
                    status = LIBXSMM_DNN_ERR_INVALID_FORMAT_CONVOLVE;
                  }
                }
              } break;
              case LIBXSMM_DNN_TENSOR_FORMAT_NHWC: {
                switch (handle->filter_format) {
                  case LIBXSMM_DNN_TENSOR_FORMAT_RSCK: {
                    status = libxsmm_dnn_convolve_st_upd_nhwc_rsck(handle, start_thread, tid);
                  } break;
                  case LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM: {
                    status = libxsmm_dnn_convolve_st_upd_nhwc_custom(handle, start_thread, tid);
                  } break;
                  default: {
                    status = LIBXSMM_DNN_ERR_INVALID_FORMAT_CONVOLVE;
                  }
                }
              } break;
              default: {
                status = LIBXSMM_DNN_ERR_INVALID_FORMAT_CONVOLVE;
              }
            }
          } break;
          default: {
            status = LIBXSMM_DNN_ERR_INVALID_KIND;
          }
        }
      } break;
      case LIBXSMM_DNN_CONV_ALGO_WINOGRAD: {
        switch (kind) {
          case LIBXSMM_DNN_COMPUTE_KIND_FWD: {
            switch (handle->buffer_format) {
              case LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM: {
                switch (handle->filter_format) {
                  case LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM: {
                    status = libxsmm_dnn_convolve_winograd_st_fwd_custom_custom(handle, start_thread, tid);
                  } break;
                  default: {
                    status = LIBXSMM_DNN_ERR_INVALID_FORMAT_CONVOLVE;
                  }
                }
              } break;
              case LIBXSMM_DNN_TENSOR_FORMAT_NHWC: {
                switch (handle->filter_format) {
                  case LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM: {
                    status = libxsmm_dnn_convolve_winograd_st_fwd_nhwc_custom(handle, start_thread, tid);
                  } break;
                  default: {
                    status = LIBXSMM_DNN_ERR_INVALID_FORMAT_CONVOLVE;
                  }
                }
              } break;
              default: {
                status = LIBXSMM_DNN_ERR_INVALID_FORMAT_CONVOLVE;
              }
            }
          } break;
          case LIBXSMM_DNN_COMPUTE_KIND_BWD: {
            switch (handle->buffer_format) {
              case LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM: {
                switch (handle->filter_format) {
                  case LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM: {
                    status = libxsmm_dnn_convolve_winograd_st_bwd_custom_custom(handle, start_thread, tid);
                  } break;
                  default: {
                    status = LIBXSMM_DNN_ERR_INVALID_FORMAT_CONVOLVE;
                  }
                }
              } break;
              case LIBXSMM_DNN_TENSOR_FORMAT_NHWC: {
                switch (handle->filter_format) {
                  case LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM: {
                    status = libxsmm_dnn_convolve_winograd_st_bwd_nhwc_custom(handle, start_thread, tid);
                  } break;
                  default: {
                    status = LIBXSMM_DNN_ERR_INVALID_FORMAT_CONVOLVE;
                  }
                }
              } break;
              default: {
                status = LIBXSMM_DNN_ERR_INVALID_FORMAT_CONVOLVE;
              }
            }
          } break;
          case LIBXSMM_DNN_COMPUTE_KIND_UPD: {
            switch (handle->buffer_format) {
              case LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM: {
                 switch (handle->filter_format) {
                  case LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM: {
                    status = libxsmm_dnn_convolve_winograd_st_upd_custom_custom(handle, start_thread, tid);
                  } break;
                  default: {
                    status = LIBXSMM_DNN_ERR_INVALID_FORMAT_CONVOLVE;
                  }
                }
              } break;
              case LIBXSMM_DNN_TENSOR_FORMAT_NHWC: {
                 switch (handle->filter_format) {
                  case LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM: {
                    status = libxsmm_dnn_convolve_winograd_st_upd_nhwc_custom(handle, start_thread, tid);
                  } break;
                  default: {
                    status = LIBXSMM_DNN_ERR_INVALID_FORMAT_CONVOLVE;
                  }
                }
              } break;
              default: {
                status = LIBXSMM_DNN_ERR_INVALID_FORMAT_CONVOLVE;
              }
            }
          } break;
          default: {
            status = LIBXSMM_DNN_ERR_INVALID_KIND;
          }
        }
      } break;
      default: {
        status = LIBXSMM_DNN_ERR_INVALID_ALGO;
      }
    }
  }
  else {
    status = LIBXSMM_DNN_ERR_INVALID_HANDLE;
  }

  return status;
}


LIBXSMM_API_DEFINITION libxsmm_dnn_err_t libxsmm_dnn_execute_st(libxsmm_dnn_layer* handle,
  libxsmm_dnn_compute_kind kind, /*unsigned*/int start_thread, /*unsigned*/int tid)
{
  return internal_execute_st(handle, kind, start_thread, tid);
}


LIBXSMM_API_DEFINITION void libxsmm_dnn_execute(libxsmm_dnn_layer* handle, libxsmm_dnn_compute_kind kind)
{
#if defined(_OPENMP)
# pragma omp parallel num_threads(handle->desc.threads)
  {
    const int tid = omp_get_thread_num();
    internal_execute_st(handle, kind, 0, tid);
  }
#else
  internal_execute_st(handle, kind, 0/*start_thread*/, 0/*tid*/);
#endif
}


LIBXSMM_API_DEFINITION libxsmm_dnn_err_t libxsmm_dnn_transpose_filter(libxsmm_dnn_layer* handle, const libxsmm_dnn_filter_type type) {
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
  int ofm1, ifm1, kj, ki, ifm2, ofm2;

  /* check for filter type */
  if ( (type != LIBXSMM_DNN_REGULAR_FILTER) ) {
    status = LIBXSMM_DNN_ERR_UNKNOWN_FILTER_TYPE;
    return status;
  }

  /* check if we have input, output and filter */
  if (handle->reg_filter == 0) {
    status = LIBXSMM_DNN_ERR_DATA_NOT_BOUND;
    return status;
  }

  /* check if we have scratch */
  if (handle->scratch1 == 0) {
    status = LIBXSMM_DNN_ERR_SCRATCH_NOT_ALLOCED;
    return status;
  }

  /* check that filter is in RSCK storage */
  if ( (handle->filter_format & LIBXSMM_DNN_TENSOR_FORMAT_RSCK) == 0 ) {
    status = LIBXSMM_DNN_ERR_MISMATCH_FILTER;
    return status;
  }

  /* check that we are in FP32 */
  if ( handle->datatype == LIBXSMM_DNN_DATATYPE_F32 ) {
    LIBXSMM_VLA_DECL(6, float, wt, (float*)handle->reg_filter->data, handle->desc.S, handle->blocksifm, handle->ifmblock, handle->blocksofm, handle->ofmblock);
    LIBXSMM_VLA_DECL(6, float, tr_wt, (float*)handle->scratch1, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock);

    for (ofm1 = 0; ofm1 < handle->blocksofm; ++ofm1) {
      for (ifm1 = 0; ifm1 < handle->blocksifm; ++ifm1) {
        for (kj=0; kj < handle->desc.R; ++kj) {
          for (ki=0; ki < handle->desc.S; ++ki) {
            for (ofm2 = 0; ofm2 < handle->ofmblock; ++ofm2) {
              for (ifm2 = 0; ifm2 < handle->ifmblock; ++ifm2) {
                LIBXSMM_VLA_ACCESS(6, tr_wt, ofm1, ifm1, kj, ki, ofm2, ifm2, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock) =
                  LIBXSMM_VLA_ACCESS(6, wt,  kj, ki, ifm1, ifm2, ofm1, ofm2, handle->desc.S, handle->blocksifm, handle->ifmblock, handle->blocksofm, handle->ofmblock);
              }
            }
          }
        }
      }
    }
    handle->filter_transposed = 1;
    return status;
  } else {
    status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
    return status;
  }
}


LIBXSMM_API_DEFINITION libxsmm_dnn_err_t libxsmm_dnn_reduce_wu_filters(libxsmm_dnn_layer* handle, const libxsmm_dnn_filter_type type) {
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
  int i, j, filter_size;

  /* check for filter type */
  if ( (type != LIBXSMM_DNN_GRADIENT_FILTER) ) {
    status = LIBXSMM_DNN_ERR_UNKNOWN_FILTER_TYPE;
    return status;
  }

  /* check if we have input, output and filter */
  if (handle->grad_filter == 0) {
    status = LIBXSMM_DNN_ERR_DATA_NOT_BOUND;
    return status;
  }

  /* check that we are in FP32 */
  if (handle->datatype == LIBXSMM_DNN_DATATYPE_F32 ) {
    if (handle->upd_use_external_reduce != 0) {
      float* filter_ptr = (float*)handle->grad_filter->data;
      /* calculate filter size */
      filter_size = handle->blocksofm * handle->blocksifm * handle->desc.R * handle->desc.S * handle->ofmblock * handle->ifmblock;

      for ( i = 0; i < handle->desc.threads; i++ ) {
        float* tmp_filter_ptr = ((float*)handle->scratch4) + (i*filter_size);
        for ( j = 0; j < filter_size; j++) {
          filter_ptr[j] += tmp_filter_ptr[j];
        }
      }
    }
  } else {
    status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
  }

  return status;
}


LIBXSMM_API_DEFINITION libxsmm_dnn_err_t libxsmm_dnn_get_codegen_success(libxsmm_dnn_layer* handle, libxsmm_dnn_compute_kind kind) {
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  if (0 != handle) {
    switch (kind) {
      case LIBXSMM_DNN_COMPUTE_KIND_FWD: {
        if (handle->code_fwd[0].xconv.sconv == 0) {
          status = LIBXSMM_DNN_WARN_FALLBACK;
        }
      } break;
      case LIBXSMM_DNN_COMPUTE_KIND_BWD: {
        if (handle->code_bwd[0].xconv.sconv == 0) {
          status = LIBXSMM_DNN_WARN_FALLBACK;
        }
      } break;
      case LIBXSMM_DNN_COMPUTE_KIND_UPD: {
        if (handle->code_upd[0].xconv.sconv == 0) {
          status = LIBXSMM_DNN_WARN_FALLBACK;
        }
      } break;
      default: {
        status = LIBXSMM_DNN_ERR_INVALID_KIND;
      }
    }
  } else {
    status = LIBXSMM_DNN_ERR_INVALID_HANDLE;
  }

  return status;
}


LIBXSMM_API_DEFINITION libxsmm_dnn_err_t libxsmm_dnn_get_parallel_tasks(libxsmm_dnn_layer* handle, libxsmm_dnn_compute_kind kind, unsigned int* num_tasks) {
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  if (0 != handle) {
    switch (kind) {
      case LIBXSMM_DNN_COMPUTE_KIND_FWD: {
        *num_tasks = handle->desc.N * handle->blocksofm;
      } break;
      case LIBXSMM_DNN_COMPUTE_KIND_BWD: {
        *num_tasks = handle->desc.N * handle->blocksifm;
      } break;
      case LIBXSMM_DNN_COMPUTE_KIND_UPD: {
        if (handle->upd_use_thread_fil > 0) {
          *num_tasks = handle->desc.N * handle->blocksifm * handle->blocksofm;
        } else {
          *num_tasks = handle->blocksifm * handle->blocksofm;
        }
      } break;
      default: {
        status = LIBXSMM_DNN_ERR_INVALID_KIND;
      }
    }
  } else {
    status = LIBXSMM_DNN_ERR_INVALID_HANDLE;
  }

  return status;
}


#if defined(LIBXSMM_BUILD) || defined(LIBXSMM_DNN_INTERNAL_API)

LIBXSMM_API_DEFINITION libxsmm_sconvfunction libxsmm_create_sconv_forward(
  const libxsmm_convolution_forward_descriptor* descriptor)
{
  libxsmm_code_pointer code = { 0 };
  LIBXSMM_INIT
  if (0 != descriptor) {
    libxsmm_build_request request;
    request.descriptor.cfwd = descriptor;
    request.kind = LIBXSMM_BUILD_KIND_CFWD;
    libxsmm_build(&request, LIBXSMM_CAPACITY_REGISTRY/*not managed*/, &code);
  }
#if !defined(NDEBUG) /* library code is expected to be mute */
  else {
    static int error_once = 0;
    if (1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED)) {
      fprintf(stderr, "LIBXSMM: invalid descriptor (forward convolution)!\n");
    }
  }
#endif
  return code.xconv.sconv;
}


LIBXSMM_API_DEFINITION libxsmm_sconvfunction libxsmm_create_sconv_backward(
  const libxsmm_convolution_backward_descriptor* descriptor)
{
  libxsmm_code_pointer code = { 0 };
  LIBXSMM_INIT
  if (0 != descriptor) {
    libxsmm_build_request request;
    request.descriptor.cbwd = descriptor;
    request.kind = LIBXSMM_BUILD_KIND_CBWD;
    libxsmm_build(&request, LIBXSMM_CAPACITY_REGISTRY/*not managed*/, &code);
  }
#if !defined(NDEBUG) /* library code is expected to be mute */
  else {
    static int error_once = 0;
    if (1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED)) {
      fprintf(stderr, "LIBXSMM: invalid descriptor (backward convolution)!\n");
    }
  }
#endif
  return code.xconv.sconv;
}


LIBXSMM_API_DEFINITION libxsmm_sconvfunction libxsmm_create_sconv_update_weights(
  const libxsmm_convolution_weight_update_descriptor* descriptor)
{
  libxsmm_code_pointer code = { 0 };
  LIBXSMM_INIT
  if (0 != descriptor) {
    libxsmm_build_request request;
    request.descriptor.cupd = descriptor;
    request.kind = LIBXSMM_BUILD_KIND_CUPD;
    libxsmm_build(&request, LIBXSMM_CAPACITY_REGISTRY/*not managed*/, &code);
  }
#if !defined(NDEBUG) /* library code is expected to be mute */
  else {
    static int error_once = 0;
    if (1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED)) {
      fprintf(stderr, "LIBXSMM: invalid convolution descriptor (weight update)!\n");
    }
  }
#endif
  return code.xconv.sconv;
}

LIBXSMM_API_DEFINITION void* libxsmm_create_xconv_forward(
  const libxsmm_convolution_forward_descriptor* descriptor)
{
  libxsmm_code_pointer code = { 0 };
  LIBXSMM_INIT
  if (0 != descriptor) {
    libxsmm_build_request request;
    request.descriptor.cfwd = descriptor;
    request.kind = LIBXSMM_BUILD_KIND_CFWD;
    libxsmm_build(&request, LIBXSMM_CAPACITY_REGISTRY/*not managed*/, &code);
  }
#if !defined(NDEBUG) /* library code is expected to be mute */
  else {
    static int error_once = 0;
    if (1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED)) {
      fprintf(stderr, "LIBXSMM: invalid descriptor (forward convolution)!\n");
    }
  }
#endif
  return code.pmm;
}


LIBXSMM_API_DEFINITION void* libxsmm_create_xconv_backward(
  const libxsmm_convolution_backward_descriptor* descriptor)
{
  libxsmm_code_pointer code = { 0 };
  LIBXSMM_INIT
  if (0 != descriptor) {
    libxsmm_build_request request;
    request.descriptor.cbwd = descriptor;
    request.kind = LIBXSMM_BUILD_KIND_CBWD;
    libxsmm_build(&request, LIBXSMM_CAPACITY_REGISTRY/*not managed*/, &code);
  }
#if !defined(NDEBUG) /* library code is expected to be mute */
  else {
    static int error_once = 0;
    if (1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED)) {
      fprintf(stderr, "LIBXSMM: invalid descriptor (backward convolution)!\n");
    }
  }
#endif
  return code.pmm;
}


LIBXSMM_API_DEFINITION void* libxsmm_create_xconv_update_weights(
  const libxsmm_convolution_weight_update_descriptor* descriptor)
{
  libxsmm_code_pointer code = { 0 };
  LIBXSMM_INIT
  if (0 != descriptor) {
    libxsmm_build_request request;
    request.descriptor.cupd = descriptor;
    request.kind = LIBXSMM_BUILD_KIND_CUPD;
    libxsmm_build(&request, LIBXSMM_CAPACITY_REGISTRY/*not managed*/, &code);
  }
#if !defined(NDEBUG) /* library code is expected to be mute */
  else {
    static int error_once = 0;
    if (1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED)) {
      fprintf(stderr, "LIBXSMM: invalid convolution descriptor (weight update)!\n");
    }
  }
#endif
  return code.pmm;
}


LIBXSMM_API_DEFINITION void* libxsmm_create_xconv_wino_forward(
  const libxsmm_convolution_winograd_descriptor* descriptor)
{
  libxsmm_code_pointer code = { 0 };
  LIBXSMM_INIT
  if (0 != descriptor) {
    libxsmm_build_request request;
    request.descriptor.cwino = descriptor;
    request.kind = LIBXSMM_BUILD_KIND_CWFWD;
    libxsmm_build(&request, LIBXSMM_CAPACITY_REGISTRY/*not managed*/, &code);
  }
#if !defined(NDEBUG) /* library code is expected to be mute */
  else {
    static int error_once = 0;
    if (1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED)) {
      fprintf(stderr, "LIBXSMM: invalid descriptor (forward convolution)!\n");
    }
  }
#endif
  return code.pmm;
}


LIBXSMM_API_DEFINITION void* libxsmm_create_xconv_wino_backward(
  const libxsmm_convolution_winograd_descriptor* descriptor)
{
  libxsmm_code_pointer code = { 0 };
  LIBXSMM_INIT
  if (0 != descriptor) {
    libxsmm_build_request request;
    request.descriptor.cwino = descriptor;
    request.kind = LIBXSMM_BUILD_KIND_CWBWD;
    libxsmm_build(&request, LIBXSMM_CAPACITY_REGISTRY/*not managed*/, &code);
  }
#if !defined(NDEBUG) /* library code is expected to be mute */
  else {
    static int error_once = 0;
    if (1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED)) {
      fprintf(stderr, "LIBXSMM: invalid descriptor (backward convolution)!\n");
    }
  }
#endif
  return code.pmm;
}


LIBXSMM_API_DEFINITION void* libxsmm_create_xconv_wino_update_weights(
  const libxsmm_convolution_winograd_descriptor* descriptor)
{
  libxsmm_code_pointer code = { 0 };
  LIBXSMM_INIT
  if (0 != descriptor) {
    libxsmm_build_request request;
    request.descriptor.cwino = descriptor;
    request.kind = LIBXSMM_BUILD_KIND_CWUPD;
    libxsmm_build(&request, LIBXSMM_CAPACITY_REGISTRY/*not managed*/, &code);
  }
#if !defined(NDEBUG) /* library code is expected to be mute */
  else {
    static int error_once = 0;
    if (1 == LIBXSMM_ATOMIC_ADD_FETCH(&error_once, 1, LIBXSMM_ATOMIC_RELAXED)) {
      fprintf(stderr, "LIBXSMM: invalid convolution descriptor (weight update)!\n");
    }
  }
#endif
  return code.pmm;
}


/* TODO: fix this hack and provide a proper API */
LIBXSMM_API void libxsmm_set_flag_reuseInput(libxsmm_dnn_layer* /*handle*/, char /*type*/);
LIBXSMM_API_DEFINITION void libxsmm_set_flag_reuseInput(libxsmm_dnn_layer* handle, char type)
{
  if (type == 'A') {
    handle->flag_reuseInput = 1;
  } else {
    handle->flag_reuseInput = 0;
  }
}

#endif /*defined(LIBXSMM_BUILD) || defined(LIBXSMM_DNN_INTERNAL_API)*/

