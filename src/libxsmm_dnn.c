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
/* Hans Pabst (Intel Corp.), Alexander Heinecke (Intel Corp.),
   Rajkishore Barik (Intel Corp.)
******************************************************************************/
#include <libxsmm.h>
#include <libxsmm_sync.h>
#include "libxsmm_main.h"
#include "libxsmm_dnn_handle.h"
#include "libxsmm_dnn_convolution_forward.h"
#include "libxsmm_dnn_convolution_backward.h"
#include "libxsmm_dnn_convolution_weight_update.h"

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


LIBXSMM_API_DEFINITION const char* libxsmm_dnn_get_error(libxsmm_dnn_err_t code)
{
  switch (code) {
    case LIBXSMM_DNN_SUCCESS:
      return "LIBXSMM DNN Success!";
    case LIBXSMM_DNN_WARN_FALLBACK:
      return "LIBXSMM DNN Warning: Falling back to naive code as target is currently not supported by LIBXSMM!";
    case LIBXSMM_DNN_ERR_GENERAL:
      return "LIBXSMM DNN Error: General error occured!";
    case LIBXSMM_DNN_ERR_CREATE_HANDLE:
      return "LIBXSMM DNN Error: Handle creation failed!";
    case LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE:
      return "LIBXSMM DNN Error: Requested datatype is not available!";
    case LIBXSMM_DNN_ERR_INVALID_BLOCKING:
      return "LIBXSMM DNN Error: Requested Input/Output buffer size cannot be blocked!";
    case LIBXSMM_DNN_ERR_INVALID_HANDLE:
      return "LIBXSMM DNN Error: An invalid handle was proivded!";
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
      return "LIBXSMM DNN Error: Invalid hanlde or buffer!";
    case LIBXSMM_DNN_ERR_MISMATCH_FILTER:
      return "LIBXSMM DNN Error: Filter doens't match handle it should be bind to!";
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
    default:
      return "LIBXSMM DNN Error: Unknown error or warning occured!";
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


LIBXSMM_API_DEFINITION libxsmm_dnn_conv_handle* libxsmm_dnn_create_conv_handle(
  libxsmm_dnn_conv_desc     conv_desc)
{
  libxsmm_dnn_err_t status;
  return libxsmm_dnn_create_conv_handle_check( conv_desc, &status);
}


LIBXSMM_API_DEFINITION libxsmm_dnn_conv_handle* libxsmm_dnn_create_conv_handle_check(
  libxsmm_dnn_conv_desc     conv_desc,
  libxsmm_dnn_err_t*        status)
{
  libxsmm_dnn_conv_handle* handle = 0;
  *status = LIBXSMM_DNN_SUCCESS;

  /* currently we don't support NCHW */
  if ( (conv_desc.buffer_format & LIBXSMM_DNN_CONV_FORMAT_NCHW) > 0 ) {
    *status = LIBXSMM_DNN_ERR_INVALID_FORMAT_NCHW;
    return 0;
  }
  /* currently we don't support KCRS */
  if ( (conv_desc.buffer_format & LIBXSMM_DNN_CONV_FORMAT_KCRS) > 0 ) {
    *status = LIBXSMM_DNN_ERR_INVALID_FORMAT_KCRS;
    return 0;
  }

  handle = (libxsmm_dnn_conv_handle*)malloc(sizeof(libxsmm_dnn_conv_handle));

  if (0 != handle) {
    /* zero entire content; not only safer but also sets data and code pointers to NULL */
    memset(handle, 0, sizeof(*handle));
    /* initialize known handle components */
    handle->desc = conv_desc;
    handle->datatype_in = conv_desc.datatype_in;
    handle->datatype_out = conv_desc.datatype_out;
    handle->buffer_format = conv_desc.buffer_format;
    handle->filter_format = conv_desc.filter_format;
    handle->fuse_ops = conv_desc.fuse_ops;
    handle->options = conv_desc.options;
    /* derive additional values */
    handle->ifhp = conv_desc.H;
    handle->ifwp = conv_desc.W;
    handle->ofh = (conv_desc.H - conv_desc.R) / conv_desc.u + 1;
    handle->ofw = (conv_desc.W - conv_desc.S) / conv_desc.v + 1;
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

    /* Set algorithm to use */
    if (conv_desc.algo == LIBXSMM_DNN_CONV_ALGO_AUTO) {
      handle->algo = LIBXSMM_DNN_CONV_ALGO_DIRECT;
    } else {
      handle->algo = conv_desc.algo;
    }

    if ( handle->algo == LIBXSMM_DNN_CONV_ALGO_DIRECT ) {
       *status = libxsmm_dnn_internal_create_conv_handle_direct_check( handle );
    } else {
      /* shouldn't happen */
    }
  }
  else {
    *status = LIBXSMM_DNN_ERR_CREATE_HANDLE;
  }
  return handle;
}


LIBXSMM_API_DEFINITION libxsmm_dnn_err_t libxsmm_dnn_destroy_conv_handle(const libxsmm_dnn_conv_handle* handle)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  if (0 != handle) { /* it is not an error attempting to destroy a NULL-handle */
    /* deallocate data components; not an error to deallocate a NULL-pointer
       deallocate code known to be not registered; no index attached
       do not use libxsmm_release_kernel here! */
    if ( (libxsmm_get_target_archid() == LIBXSMM_X86_AVX512_MIC  ||
          libxsmm_get_target_archid() == LIBXSMM_X86_AVX512_CORE    ) && (handle->avx512avx2fallback == 0) ) {
      libxsmm_xfree(handle->code_fwd[0].pmm);
      libxsmm_xfree(handle->code_fwd[1].pmm);
      libxsmm_xfree(handle->code_fwd[2].pmm);
      libxsmm_xfree(handle->code_fwd[3].pmm);
      libxsmm_xfree(handle->code_bwd[0].pmm);
      if ((handle->filter_format == LIBXSMM_DNN_CONV_FORMAT_LIBXSMM) && (handle->buffer_format == LIBXSMM_DNN_CONV_FORMAT_LIBXSMM)) {
        libxsmm_xfree(handle->code_bwd[1].pmm);
        libxsmm_xfree(handle->code_bwd[2].pmm);
        libxsmm_xfree(handle->code_bwd[3].pmm);
      }
      libxsmm_xfree(handle->code_upd[0].pmm);
      if ((handle->filter_format == LIBXSMM_DNN_CONV_FORMAT_LIBXSMM) && (handle->buffer_format == LIBXSMM_DNN_CONV_FORMAT_LIBXSMM)) {
        libxsmm_xfree(handle->code_upd[1].pmm);
        libxsmm_xfree(handle->code_upd[2].pmm);
        libxsmm_xfree(handle->code_upd[3].pmm);
        libxsmm_xfree(handle->code_upd[4].pmm);
        libxsmm_xfree(handle->code_upd[5].pmm);
      }
    } else if ( (libxsmm_get_target_archid() == LIBXSMM_X86_AVX2) || (handle->avx512avx2fallback != 0) ) {
      libxsmm_xfree(handle->code_fwd[0].pmm);
      if (handle->fwd_ofw_rb_2 != 0) {
        libxsmm_xfree(handle->code_fwd[1].pmm);
      }
      libxsmm_xfree(handle->code_bwd[0].pmm);
      libxsmm_xfree(handle->code_upd[0].pmm);
    } else {
      /* no kernel was JITed */
    }

    /*Deallocate scratch in handle*/
    if(handle->scratch1 != 0) libxsmm_xfree(handle->scratch1);
    if(handle->scratch2 != 0) libxsmm_barrier_release((const libxsmm_barrier*)handle->scratch2);
    if(handle->scratch3 != 0) libxsmm_xfree(handle->scratch3);
    if(handle->scratch4 != 0) libxsmm_xfree(handle->scratch4);

    /* deallocate handle structure */
    free(/*remove constness*/(libxsmm_dnn_conv_handle*)handle);
  }
  else {
    status = LIBXSMM_DNN_ERR_INVALID_HANDLE;
  }

  return status;
}

LIBXSMM_API_DEFINITION libxsmm_dnn_buffer* libxsmm_dnn_create_input_buffer(const libxsmm_dnn_conv_handle* handle)
{
  libxsmm_dnn_err_t status;
  return libxsmm_dnn_create_input_buffer_check(handle, &status);
}


LIBXSMM_API_DEFINITION libxsmm_dnn_buffer* libxsmm_dnn_create_input_buffer_check(const libxsmm_dnn_conv_handle* handle, libxsmm_dnn_err_t* status)
{
  libxsmm_dnn_buffer* buffer = 0;

  if (0 != handle) {
    buffer = (libxsmm_dnn_buffer*)malloc(sizeof(libxsmm_dnn_buffer));

    if (0 != buffer) { /* set properties of the buffer according to convolution handle */
      buffer->N = handle->desc.N;
      buffer->fmb = handle->blocksifm;
      buffer->bfm = handle->ifmblock;
      buffer->H = handle->ifhp;
      buffer->W = handle->ifwp;
      buffer->format = handle->buffer_format;
      buffer->datatype = handle->datatype_in;
      buffer->lpb = handle->fm_lp_block;
      buffer->data = libxsmm_aligned_malloc( /* allocate raw data */
        buffer->N * buffer->fmb * buffer->bfm * buffer->H * buffer->W * buffer->lpb * libxsmm_dnn_typesize(buffer->datatype),
        LIBXSMM_ALIGNMENT);
      if (0 == buffer->data) {
        free(buffer);
        buffer = 0;
      }
    }
  }

  assert(0 != status);
  *status = 0 != buffer ? LIBXSMM_DNN_SUCCESS : LIBXSMM_DNN_ERR_CREATE_BUFFER;

  return buffer;
}


LIBXSMM_API_DEFINITION libxsmm_dnn_buffer* libxsmm_dnn_link_input_buffer(const libxsmm_dnn_conv_handle* handle, const void* data, libxsmm_dnn_conv_format in_format)
{
  libxsmm_dnn_err_t status;
  return libxsmm_dnn_link_input_buffer_check( handle, data, in_format, &status );
}


LIBXSMM_API_DEFINITION libxsmm_dnn_buffer* libxsmm_dnn_link_input_buffer_check(const libxsmm_dnn_conv_handle* handle, const void* data, libxsmm_dnn_conv_format in_format, libxsmm_dnn_err_t* status)
{
  libxsmm_dnn_buffer* buffer = (libxsmm_dnn_buffer*)malloc(sizeof(libxsmm_dnn_buffer));
  *status = LIBXSMM_DNN_SUCCESS;

  if (handle != 0 && buffer != 0 && data != 0) {
    /* set properties of the buffer according to convolution handle */
    buffer->N = handle->desc.N;
    buffer->fmb = handle->blocksifm;
    buffer->bfm = handle->ifmblock;
    buffer->H = handle->ifhp;
    buffer->W = handle->ifwp;
    buffer->format = in_format;
    buffer->datatype = handle->datatype_in;
    buffer->lpb = handle->fm_lp_block;
    /* NHWC */
    if ( ((handle->buffer_format & in_format) > 0) && ((in_format & LIBXSMM_DNN_CONV_FORMAT_NHWC ) > 0)  && ((in_format & LIBXSMM_DNN_CONV_FORMAT_PTR ) > 0) ) {
      buffer->data = (void*)data;
    /* custom LIBXSMM format */
    } else if ( ((handle->buffer_format & in_format) > 0) && ((in_format & LIBXSMM_DNN_CONV_FORMAT_LIBXSMM ) > 0)  && ((in_format & LIBXSMM_DNN_CONV_FORMAT_PTR ) > 0) ) {
      buffer->data = (void*)data;
    } else {
      *status = LIBXSMM_DNN_ERR_UNSUPPORTED_SRC_FORMAT;
    }
  }
  else {
    *status = LIBXSMM_DNN_ERR_CREATE_BUFFER;
    buffer = 0;
  }

  if (*status != LIBXSMM_DNN_SUCCESS) {
    free((libxsmm_dnn_buffer*)buffer);
    buffer = 0;
  }

  return buffer;
}


LIBXSMM_API_DEFINITION libxsmm_dnn_conv_datalayout* libxsmm_dnn_get_input_buffer_datalayout(const libxsmm_dnn_conv_handle* handle) {
  libxsmm_dnn_err_t status;
  return libxsmm_dnn_get_input_buffer_datalayout_check( handle, &status );
}


LIBXSMM_API_DEFINITION libxsmm_dnn_conv_datalayout* libxsmm_dnn_get_input_buffer_datalayout_check(const libxsmm_dnn_conv_handle* handle, libxsmm_dnn_err_t* status) {
  libxsmm_dnn_conv_datalayout* layout;

  *status = LIBXSMM_DNN_SUCCESS;
  layout = 0;

  if (handle != 0) {
    layout = (libxsmm_dnn_conv_datalayout*) malloc(sizeof(libxsmm_dnn_conv_datalayout));
    memset( layout, 0, sizeof(libxsmm_dnn_conv_datalayout) );

    if (layout != 0) {
      if ((handle->buffer_format & LIBXSMM_DNN_CONV_FORMAT_LIBXSMM) > 0) {
        if ( handle->datatype_in == LIBXSMM_DNN_DATATYPE_F32 ) {
          layout->dim_type = (libxsmm_dnn_conv_dimtype*) malloc(5*sizeof(libxsmm_dnn_conv_dimtype));
          layout->dim_size = (unsigned int*) malloc(5*sizeof(unsigned int));

          layout->num_dims = 5;
          layout->dim_size[0] = handle->ifmblock;
          layout->dim_size[1] = handle->ifwp;
          layout->dim_size[2] = handle->ifhp;
          layout->dim_size[3] = handle->blocksifm;
          layout->dim_size[4] = handle->desc.N;
          layout->dim_type[0] = LIBXSMM_DNN_CONV_DIMTYPE_C;
          layout->dim_type[1] = LIBXSMM_DNN_CONV_DIMTYPE_W;
          layout->dim_type[2] = LIBXSMM_DNN_CONV_DIMTYPE_H;
          layout->dim_type[3] = LIBXSMM_DNN_CONV_DIMTYPE_C;
          layout->dim_type[4] = LIBXSMM_DNN_CONV_DIMTYPE_N;
        } else if ( (handle->datatype_in == LIBXSMM_DNN_DATATYPE_I16) || (handle->datatype_in == LIBXSMM_DNN_DATATYPE_I8) ) {
          layout->dim_type = (libxsmm_dnn_conv_dimtype*) malloc(6*sizeof(libxsmm_dnn_conv_dimtype));
          layout->dim_size = (unsigned int*) malloc(6*sizeof(unsigned int));

          layout->num_dims = 6;
          layout->dim_size[0] = handle->fm_lp_block;
          layout->dim_size[1] = handle->ifmblock;
          layout->dim_size[2] = handle->ifwp;
          layout->dim_size[3] = handle->ifhp;
          layout->dim_size[4] = handle->blocksifm;
          layout->dim_size[5] = handle->desc.N;
          layout->dim_type[0] = LIBXSMM_DNN_CONV_DIMTYPE_C;
          layout->dim_type[1] = LIBXSMM_DNN_CONV_DIMTYPE_C;
          layout->dim_type[2] = LIBXSMM_DNN_CONV_DIMTYPE_W;
          layout->dim_type[3] = LIBXSMM_DNN_CONV_DIMTYPE_H;
          layout->dim_type[4] = LIBXSMM_DNN_CONV_DIMTYPE_C;
          layout->dim_type[5] = LIBXSMM_DNN_CONV_DIMTYPE_N;
        } else {
          free(layout);
          *status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
        }
      } else if ((handle->buffer_format & LIBXSMM_DNN_CONV_FORMAT_NHWC) > 0) {
        layout->dim_type = (libxsmm_dnn_conv_dimtype*) malloc(4*sizeof(libxsmm_dnn_conv_dimtype));
        layout->dim_size = (unsigned int*) malloc(4*sizeof(unsigned int));

        layout->num_dims = 4;
        layout->dim_size[0] = handle->ifmblock * handle->blocksifm;
        layout->dim_size[1] = handle->ifwp;
        layout->dim_size[2] = handle->ifhp;
        layout->dim_size[3] = handle->desc.N;
        layout->dim_type[0] = LIBXSMM_DNN_CONV_DIMTYPE_C;
        layout->dim_type[1] = LIBXSMM_DNN_CONV_DIMTYPE_W;
        layout->dim_type[2] = LIBXSMM_DNN_CONV_DIMTYPE_H;
        layout->dim_type[3] = LIBXSMM_DNN_CONV_DIMTYPE_N;
      } else {
        free(layout);
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


LIBXSMM_API_DEFINITION libxsmm_dnn_buffer* libxsmm_dnn_create_output_buffer(const libxsmm_dnn_conv_handle* handle)
{
  libxsmm_dnn_err_t status;
  return libxsmm_dnn_create_output_buffer_check(handle, &status);
}


LIBXSMM_API_DEFINITION libxsmm_dnn_buffer* libxsmm_dnn_create_output_buffer_check(const libxsmm_dnn_conv_handle* handle, libxsmm_dnn_err_t* status)
{
  libxsmm_dnn_buffer* buffer = 0;

  if (0 != handle) {
    buffer = (libxsmm_dnn_buffer*)malloc(sizeof(libxsmm_dnn_buffer));

    if (0 != buffer) { /* set properties of the buffer according to convolution handle */
      buffer->N = handle->desc.N;
      buffer->fmb = handle->blocksofm;
      buffer->bfm = handle->ofmblock;
      buffer->H = handle->ofhp;
      buffer->W = handle->ofwp;
      buffer->format = handle->buffer_format;
      buffer->lpb = 1;
      buffer->datatype = handle->datatype_out;
      buffer->data = libxsmm_aligned_malloc( /* allocate raw data, we always have a 4 byte wide type */
        buffer->N * buffer->fmb * buffer->bfm * buffer->H * buffer->W * buffer->lpb * libxsmm_dnn_typesize(buffer->datatype),
        LIBXSMM_ALIGNMENT);
      if (0 == buffer->data) {
        free(buffer);
        buffer = 0;
      }
    }
  }

  assert(0 != status);
  *status = 0 != buffer ? LIBXSMM_DNN_SUCCESS : LIBXSMM_DNN_ERR_CREATE_BUFFER;

  return buffer;
}


LIBXSMM_API_DEFINITION libxsmm_dnn_buffer* libxsmm_dnn_link_output_buffer(const libxsmm_dnn_conv_handle* handle, const void* data, libxsmm_dnn_conv_format in_format)
{
  libxsmm_dnn_err_t status;
  return libxsmm_dnn_link_output_buffer_check( handle, data, in_format, &status );
}


LIBXSMM_API_DEFINITION libxsmm_dnn_buffer* libxsmm_dnn_link_output_buffer_check(const libxsmm_dnn_conv_handle* handle, const void* data, libxsmm_dnn_conv_format in_format, libxsmm_dnn_err_t* status)
{
  libxsmm_dnn_buffer* buffer = (libxsmm_dnn_buffer*)malloc(sizeof(libxsmm_dnn_buffer));
  *status = LIBXSMM_DNN_SUCCESS;

  if (handle != 0 && buffer != 0 && data != 0) {
    /* set properties of the buffer according to convolution handle */
    buffer->N = handle->desc.N;
    buffer->fmb = handle->blocksofm;
    buffer->bfm = handle->ofmblock;
    buffer->H = handle->ofhp;
    buffer->W = handle->ofwp;
    buffer->format = in_format;
    buffer->datatype = handle->datatype_out;
    buffer->lpb = 1;
    /* NHWC */
    if ( ((handle->buffer_format & in_format) > 0) && ((in_format & LIBXSMM_DNN_CONV_FORMAT_NHWC ) > 0)  && ((in_format & LIBXSMM_DNN_CONV_FORMAT_PTR ) > 0) ) {
      buffer->data = (void*)data;
    /* custom LIBXSMM format */
    } else if ( ((handle->buffer_format & in_format) > 0) && ((in_format & LIBXSMM_DNN_CONV_FORMAT_LIBXSMM ) > 0)  && ((in_format & LIBXSMM_DNN_CONV_FORMAT_PTR ) > 0) ) {
      buffer->data = (void*)data;
    } else {
      *status = LIBXSMM_DNN_ERR_UNSUPPORTED_SRC_FORMAT;
    }
  }
  else {
    *status = LIBXSMM_DNN_ERR_CREATE_BUFFER;
    buffer = 0;
  }

  if (*status != LIBXSMM_DNN_SUCCESS) {
    free((libxsmm_dnn_buffer*)buffer);
    buffer = 0;
  }

  return buffer;
}


LIBXSMM_API_DEFINITION libxsmm_dnn_conv_datalayout* libxsmm_dnn_get_output_buffer_datalayout(const libxsmm_dnn_conv_handle* handle) {
  libxsmm_dnn_err_t status;
  return libxsmm_dnn_get_output_buffer_datalayout_check( handle, &status );
}


LIBXSMM_API_DEFINITION libxsmm_dnn_conv_datalayout* libxsmm_dnn_get_output_buffer_datalayout_check(const libxsmm_dnn_conv_handle* handle, libxsmm_dnn_err_t* status) {
  libxsmm_dnn_conv_datalayout* layout;

  *status = LIBXSMM_DNN_SUCCESS;
  layout = 0;

  if (handle != 0) {
    layout = (libxsmm_dnn_conv_datalayout*) malloc(sizeof(libxsmm_dnn_conv_datalayout));
    memset( layout, 0, sizeof(libxsmm_dnn_conv_datalayout) );

    if (layout != 0) {
      if ((handle->buffer_format & LIBXSMM_DNN_CONV_FORMAT_LIBXSMM) > 0) {
        if ( (handle->datatype_out == LIBXSMM_DNN_DATATYPE_F32) || (handle->datatype_out == LIBXSMM_DNN_DATATYPE_I32) ) {
          layout->dim_type = (libxsmm_dnn_conv_dimtype*) malloc(5*sizeof(libxsmm_dnn_conv_dimtype));
          layout->dim_size = (unsigned int*) malloc(5*sizeof(unsigned int));

          layout->num_dims = 5;
          layout->dim_size[0] = handle->ofmblock;
          layout->dim_size[1] = handle->ifwp;
          layout->dim_size[2] = handle->ifhp;
          layout->dim_size[3] = handle->blocksofm;
          layout->dim_size[4] = handle->desc.N;
          layout->dim_type[0] = LIBXSMM_DNN_CONV_DIMTYPE_C;
          layout->dim_type[1] = LIBXSMM_DNN_CONV_DIMTYPE_W;
          layout->dim_type[2] = LIBXSMM_DNN_CONV_DIMTYPE_H;
          layout->dim_type[3] = LIBXSMM_DNN_CONV_DIMTYPE_C;
          layout->dim_type[4] = LIBXSMM_DNN_CONV_DIMTYPE_N;
        } else {
          free(layout);
          *status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
        }
      } else if ((handle->buffer_format & LIBXSMM_DNN_CONV_FORMAT_NHWC) > 0) {
        layout->dim_type = (libxsmm_dnn_conv_dimtype*) malloc(4*sizeof(libxsmm_dnn_conv_dimtype));
        layout->dim_size = (unsigned int*) malloc(4*sizeof(unsigned int));

        layout->num_dims = 4;
        layout->dim_size[0] = handle->ofmblock * handle->blocksofm;
        layout->dim_size[1] = handle->ifwp;
        layout->dim_size[2] = handle->ifhp;
        layout->dim_size[3] = handle->desc.N;
        layout->dim_type[0] = LIBXSMM_DNN_CONV_DIMTYPE_C;
        layout->dim_type[1] = LIBXSMM_DNN_CONV_DIMTYPE_W;
        layout->dim_type[2] = LIBXSMM_DNN_CONV_DIMTYPE_H;
        layout->dim_type[3] = LIBXSMM_DNN_CONV_DIMTYPE_N;
      } else {
        free(layout);
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


LIBXSMM_API_DEFINITION libxsmm_dnn_err_t libxsmm_dnn_destroy_buffer(const libxsmm_dnn_buffer* buffer)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  if (0 != buffer) { /* it is not an error attempting to destroy a NULL-handle */
    /* deallocate data components; not an error to deallocate a NULL-pointer, just deallocate if it's LIBXSMM private data */
    if ( (buffer->format & LIBXSMM_DNN_CONV_FORMAT_PTR) == 0 ) {
      libxsmm_xfree(buffer->data);
    }
    /* deallocate handle structure */
    free(/*remove constness*/(libxsmm_dnn_buffer*)buffer);
  }
  else {
    status = LIBXSMM_DNN_ERR_INVALID_BUFFER;
  }

  return status;
}


LIBXSMM_API_DEFINITION libxsmm_dnn_filter* libxsmm_dnn_create_filter(const libxsmm_dnn_conv_handle* handle)
{
  libxsmm_dnn_err_t status;
  return libxsmm_dnn_create_filter_check(handle, &status);
}


LIBXSMM_API_DEFINITION libxsmm_dnn_filter* libxsmm_dnn_create_filter_check(const libxsmm_dnn_conv_handle* handle, libxsmm_dnn_err_t* status)
{
  libxsmm_dnn_filter* filter = 0;

  if (0 != handle) {
    filter = (libxsmm_dnn_filter*)malloc(sizeof(libxsmm_dnn_filter));

    if (0 != filter) { /* set properties of the buffer according to convolution handle */
      filter->ifmb = handle->blocksifm;
      filter->bifm = handle->ifmblock;
      filter->ofmb = handle->blocksofm;
      filter->bofm = handle->ofmblock;
      filter->R = handle->desc.R;
      filter->S = handle->desc.S;
      filter->format = handle->filter_format;
      filter->datatype = handle->datatype_in;
      filter->lpb = handle->fm_lp_block;
      filter->data = libxsmm_aligned_malloc( /* allocate raw data */
        filter->ifmb * filter->bifm * filter->ofmb * filter->bofm * filter->R * filter->S * filter->lpb * libxsmm_dnn_typesize(filter->datatype),
        LIBXSMM_ALIGNMENT);
      if (0 == filter->data) {
        free(filter);
        filter = 0;
      }
    }
  }

  assert(0 != status);
  *status = 0 != filter ? LIBXSMM_DNN_SUCCESS : LIBXSMM_DNN_ERR_CREATE_FILTER;

  return filter;
}


LIBXSMM_API_DEFINITION libxsmm_dnn_filter* libxsmm_dnn_link_filter(const libxsmm_dnn_conv_handle* handle, const void* data, libxsmm_dnn_conv_format in_format)
{
  libxsmm_dnn_err_t status;
  return libxsmm_dnn_link_filter_check(handle, data, in_format, &status);
}


LIBXSMM_API_DEFINITION libxsmm_dnn_filter* libxsmm_dnn_link_filter_check(const libxsmm_dnn_conv_handle* handle, const void* data, libxsmm_dnn_conv_format in_format, libxsmm_dnn_err_t* status)
{
  libxsmm_dnn_filter* filter = (libxsmm_dnn_filter*)malloc(sizeof(libxsmm_dnn_filter));
  *status = LIBXSMM_DNN_SUCCESS;

  if (handle != 0 && filter != 0 && data != 0) {
    /* set properties of the buffer according to convolution handle */
    filter->ifmb = handle->blocksifm;
    filter->bifm = handle->ifmblock;
    filter->ofmb = handle->blocksofm;
    filter->bofm = handle->ofmblock;
    filter->R = handle->desc.R;
    filter->S = handle->desc.S;
    filter->format = in_format;
    filter->datatype = handle->datatype_in;
    filter->lpb = handle->fm_lp_block;
    /* RSCK */
    if ( ((handle->filter_format & in_format) > 0) && ((in_format & LIBXSMM_DNN_CONV_FORMAT_RSCK ) > 0)  && ((in_format & LIBXSMM_DNN_CONV_FORMAT_PTR ) > 0) ) {
      filter->data = (void*)data;
    /* custom LIBXSMM format */
    } else if ( ((handle->filter_format & in_format) > 0) && ((in_format & LIBXSMM_DNN_CONV_FORMAT_LIBXSMM ) > 0)  && ((in_format & LIBXSMM_DNN_CONV_FORMAT_PTR ) > 0) ) {
      filter->data = (void*)data;
    } else {
      *status = LIBXSMM_DNN_ERR_UNSUPPORTED_SRC_FORMAT;
    }
  }
  else {
    *status = LIBXSMM_DNN_ERR_CREATE_FILTER;
    filter = 0;
  }

  if (*status != LIBXSMM_DNN_SUCCESS) {
    *status = LIBXSMM_DNN_ERR_CREATE_FILTER;
    free((libxsmm_dnn_filter*)filter);
    filter = 0;
  }

  return filter;
}


LIBXSMM_API_DEFINITION libxsmm_dnn_conv_datalayout* libxsmm_dnn_get_filter_datalayout(const libxsmm_dnn_conv_handle* handle) {
  libxsmm_dnn_err_t status;
  return libxsmm_dnn_get_filter_datalayout_check( handle, &status );
}


LIBXSMM_API_DEFINITION libxsmm_dnn_conv_datalayout* libxsmm_dnn_get_filter_datalayout_check(const libxsmm_dnn_conv_handle* handle, libxsmm_dnn_err_t* status) {
  libxsmm_dnn_conv_datalayout* layout;

  *status = LIBXSMM_DNN_SUCCESS;
  layout = 0;

  if (handle != 0) {
    layout = (libxsmm_dnn_conv_datalayout*) malloc(sizeof(libxsmm_dnn_conv_datalayout));
    memset( layout, 0, sizeof(libxsmm_dnn_conv_datalayout) );

    if (layout != 0) {
      if ((handle->filter_format & LIBXSMM_DNN_CONV_FORMAT_LIBXSMM) > 0) {
        if ( (handle->datatype_in == LIBXSMM_DNN_DATATYPE_F32) && (handle->datatype_out == LIBXSMM_DNN_DATATYPE_F32) ) {
          layout->dim_type = (libxsmm_dnn_conv_dimtype*) malloc(6*sizeof(libxsmm_dnn_conv_dimtype));
          layout->dim_size = (unsigned int*) malloc(6*sizeof(unsigned int));

          layout->num_dims = 6;
          layout->dim_size[0] = handle->ofmblock;
          layout->dim_size[1] = handle->ifmblock;
          layout->dim_size[2] = handle->desc.S;
          layout->dim_size[3] = handle->desc.R;
          layout->dim_size[4] = handle->blocksofm;
          layout->dim_size[5] = handle->blocksofm;
          layout->dim_type[0] = LIBXSMM_DNN_CONV_DIMTYPE_K;
          layout->dim_type[1] = LIBXSMM_DNN_CONV_DIMTYPE_C;
          layout->dim_type[2] = LIBXSMM_DNN_CONV_DIMTYPE_S;
          layout->dim_type[3] = LIBXSMM_DNN_CONV_DIMTYPE_R;
          layout->dim_type[4] = LIBXSMM_DNN_CONV_DIMTYPE_C;
          layout->dim_type[5] = LIBXSMM_DNN_CONV_DIMTYPE_K;
        } else if ( ((handle->datatype_in == LIBXSMM_DNN_DATATYPE_I16) && (handle->datatype_out == LIBXSMM_DNN_DATATYPE_I32)) ||
                    ((handle->datatype_in == LIBXSMM_DNN_DATATYPE_I8)  && (handle->datatype_out == LIBXSMM_DNN_DATATYPE_I16)) ||
                    ((handle->datatype_in == LIBXSMM_DNN_DATATYPE_I8)  && (handle->datatype_out == LIBXSMM_DNN_DATATYPE_I32))    ) {
          layout->dim_type = (libxsmm_dnn_conv_dimtype*) malloc(7*sizeof(libxsmm_dnn_conv_dimtype));
          layout->dim_size = (unsigned int*) malloc(7*sizeof(unsigned int));

          layout->num_dims = 7;
          layout->dim_size[0] = handle->fm_lp_block;
          layout->dim_size[1] = handle->ofmblock;
          layout->dim_size[2] = handle->ifmblock;
          layout->dim_size[3] = handle->desc.S;
          layout->dim_size[4] = handle->desc.R;
          layout->dim_size[5] = handle->blocksofm;
          layout->dim_size[6] = handle->blocksofm;
          layout->dim_type[0] = LIBXSMM_DNN_CONV_DIMTYPE_C;
          layout->dim_type[1] = LIBXSMM_DNN_CONV_DIMTYPE_K;
          layout->dim_type[2] = LIBXSMM_DNN_CONV_DIMTYPE_C;
          layout->dim_type[3] = LIBXSMM_DNN_CONV_DIMTYPE_S;
          layout->dim_type[4] = LIBXSMM_DNN_CONV_DIMTYPE_R;
          layout->dim_type[5] = LIBXSMM_DNN_CONV_DIMTYPE_C;
          layout->dim_type[6] = LIBXSMM_DNN_CONV_DIMTYPE_K;
        } else {
          free(layout);
          *status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
        }
      } else if ((handle->filter_format & LIBXSMM_DNN_CONV_FORMAT_RSCK) > 0) {
        layout->dim_type = (libxsmm_dnn_conv_dimtype*) malloc(4*sizeof(libxsmm_dnn_conv_dimtype));
        layout->dim_size = (unsigned int*) malloc(4*sizeof(unsigned int));

        layout->num_dims = 4;
        layout->dim_size[0] = handle->ofmblock * handle->blocksofm;
        layout->dim_size[1] = handle->ofmblock * handle->blocksofm;
        layout->dim_size[2] = handle->desc.S;
        layout->dim_size[3] = handle->desc.K;
        layout->dim_type[0] = LIBXSMM_DNN_CONV_DIMTYPE_K;
        layout->dim_type[1] = LIBXSMM_DNN_CONV_DIMTYPE_C;
        layout->dim_type[2] = LIBXSMM_DNN_CONV_DIMTYPE_S;
        layout->dim_type[3] = LIBXSMM_DNN_CONV_DIMTYPE_R;
      } else {
        free(layout);
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


LIBXSMM_API_DEFINITION libxsmm_dnn_err_t libxsmm_dnn_destroy_filter(const libxsmm_dnn_filter* filter)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  if (0 != filter) { /* it is not an error attempting to destroy a NULL-handle */
    /* deallocate data components; not an error to deallocate a NULL-pointer */
    if ( (filter->format & LIBXSMM_DNN_CONV_FORMAT_PTR) == 0 ) {
      libxsmm_xfree(filter->data);
    }
    /* deallocate handle structure */
    free(/*remove constness*/(libxsmm_dnn_filter*)filter);
  }
  else {
    status = LIBXSMM_DNN_ERR_INVALID_FILTER;
  }

  return status;
}


LIBXSMM_API_DEFINITION libxsmm_dnn_bias* libxsmm_dnn_create_bias(const libxsmm_dnn_conv_handle* handle)
{
  libxsmm_dnn_err_t status;
  return libxsmm_dnn_create_bias_check(handle, &status);
}


LIBXSMM_API_DEFINITION libxsmm_dnn_bias* libxsmm_dnn_create_bias_check(const libxsmm_dnn_conv_handle* handle, libxsmm_dnn_err_t* status)
{
  libxsmm_dnn_bias* bias = 0;

  if (0 != handle) {
    libxsmm_dnn_bias* bias = (libxsmm_dnn_bias*)malloc(sizeof(libxsmm_dnn_bias));

    if (0 != bias) { /* set properties of the buffer according to convolution handle */
      bias->fmb = handle->blocksifm;
      bias->bfm = handle->ifmblock;
      bias->datatype = handle->datatype_out;
      bias->lpb = handle->fm_lp_block;
      bias->data = libxsmm_aligned_malloc( /* allocate raw data, we always have a 4 byte wide type */
        bias->fmb * bias->bfm * bias->lpb * libxsmm_dnn_typesize(bias->datatype),
        LIBXSMM_ALIGNMENT);
      if (0 == bias->data) {
        free(bias);
        bias = 0;
      }
    }
  }

  assert(0 != status);
  *status = 0 != bias ? LIBXSMM_DNN_SUCCESS : LIBXSMM_DNN_ERR_CREATE_BIAS;

  return bias;
}


LIBXSMM_API_DEFINITION libxsmm_dnn_err_t libxsmm_dnn_destroy_bias(const libxsmm_dnn_bias* bias)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  if (0 != bias) { /* it is not an error attempting to destroy a NULL-handle */
    /* deallocate data components; not an error to deallocate a NULL-pointer */
    libxsmm_xfree(bias->data);
    /* deallocate handle structure */
    free(/*remove constness*/(libxsmm_dnn_bias*)bias);
  }
  else {
    status = LIBXSMM_DNN_ERR_INVALID_BIAS;
  }

  return status;
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_destroy_datalayout(libxsmm_dnn_conv_datalayout* layout) {
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


LIBXSMM_API_DEFINITION libxsmm_dnn_err_t libxsmm_dnn_copyin_buffer(const libxsmm_dnn_buffer* buffer, const void* data, libxsmm_dnn_conv_format in_format)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  if (0 != buffer) {
    switch (in_format) {
      case LIBXSMM_DNN_CONV_FORMAT_NCHW: {
        switch (buffer->format) {
          case LIBXSMM_DNN_CONV_FORMAT_LIBXSMM: {
            switch (buffer->datatype) {
              case LIBXSMM_DNN_DATATYPE_F32: {
                typedef float element_type;
#               include "template/libxsmm_dnn_buffer_copy_in_nchw.tpl.c"
              } break;
              case LIBXSMM_DNN_DATATYPE_I32: {
                typedef int element_type;
#               include "template/libxsmm_dnn_buffer_copy_in_nchw.tpl.c"
              } break;
              case LIBXSMM_DNN_DATATYPE_I16: {
                typedef short element_type;
#               include "template/libxsmm_dnn_buffer_copy_in_nchw.tpl.c"
              } break;
              case LIBXSMM_DNN_DATATYPE_I8: {
                typedef char element_type;
#               include "template/libxsmm_dnn_buffer_copy_in_nchw.tpl.c"
              } break;
              default: {
                status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
              }
            }
          } break;
          default: {
            status = LIBXSMM_DNN_ERR_UNSUPPORTED_DST_FORMAT;
          }
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
  const size_t size = (size_t)buffer->N * (size_t)buffer->fmb
                    * (size_t)buffer->bfm * (size_t)buffer->H * (size_t)buffer->W;
  size_t i;

  if (0 != buffer) {
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


LIBXSMM_API_DEFINITION libxsmm_dnn_err_t libxsmm_dnn_copyout_buffer(const libxsmm_dnn_buffer* buffer, void* data, libxsmm_dnn_conv_format out_format)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  if (0 != buffer) {
    switch (out_format) {
      case LIBXSMM_DNN_CONV_FORMAT_NCHW: {
        switch (buffer->format) {
          case LIBXSMM_DNN_CONV_FORMAT_LIBXSMM: {
            switch (buffer->datatype) {
              case LIBXSMM_DNN_DATATYPE_F32: {
                typedef float element_type;
#               include "template/libxsmm_dnn_buffer_copy_out_nchw.tpl.c"
              } break;
              case LIBXSMM_DNN_DATATYPE_I32: {
                typedef int element_type;
#               include "template/libxsmm_dnn_buffer_copy_out_nchw.tpl.c"
              } break;
              case LIBXSMM_DNN_DATATYPE_I16: {
                typedef short element_type;
#               include "template/libxsmm_dnn_buffer_copy_out_nchw.tpl.c"
              } break;
              case LIBXSMM_DNN_DATATYPE_I8: {
                typedef char element_type;
#               include "template/libxsmm_dnn_buffer_copy_out_nchw.tpl.c"
              } break;
              default: {
                status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
              }
            }
          } break;
          default: {
            status = LIBXSMM_DNN_ERR_UNSUPPORTED_SRC_FORMAT;
          }
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


LIBXSMM_API_DEFINITION libxsmm_dnn_err_t libxsmm_dnn_copyin_filter(const libxsmm_dnn_filter* filter, const void* data, libxsmm_dnn_conv_format in_format)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  if (0 != filter) {
    switch (in_format) {
      case LIBXSMM_DNN_CONV_FORMAT_KCRS: {
        switch (filter->format) {
          case LIBXSMM_DNN_CONV_FORMAT_LIBXSMM: {
            switch (filter->datatype) {
              case LIBXSMM_DNN_DATATYPE_F32: {
                typedef float element_type;
#               include "template/libxsmm_dnn_filter_copy_in_kcrs.tpl.c"
              } break;
              case LIBXSMM_DNN_DATATYPE_I16: {
                typedef short element_type;
#               include "template/libxsmm_dnn_filter_copy_in_kcrs.tpl.c"
              } break;
              case LIBXSMM_DNN_DATATYPE_I8: {
                typedef char element_type;
#               include "template/libxsmm_dnn_filter_copy_in_kcrs.tpl.c"
              } break;
              default: {
                status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
              }
            }
          } break;
          default: {
            status = LIBXSMM_DNN_ERR_UNSUPPORTED_DST_FORMAT;
          }
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


LIBXSMM_API_DEFINITION libxsmm_dnn_err_t libxsmm_dnn_copyout_filter(const libxsmm_dnn_filter* filter, void* data, libxsmm_dnn_conv_format out_format)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  if (0 != filter) {
    switch (out_format) {
      case LIBXSMM_DNN_CONV_FORMAT_KCRS: {
        switch (filter->format) {
          case LIBXSMM_DNN_CONV_FORMAT_LIBXSMM: {
            switch (filter->datatype) {
              case LIBXSMM_DNN_DATATYPE_F32: {
                typedef float element_type;
#               include "template/libxsmm_dnn_filter_copy_out_kcrs.tpl.c"
              } break;
              case LIBXSMM_DNN_DATATYPE_I32: {
                typedef int element_type;
#               include "template/libxsmm_dnn_filter_copy_out_kcrs.tpl.c"
              } break;
              case LIBXSMM_DNN_DATATYPE_I16: {
                typedef short element_type;
#               include "template/libxsmm_dnn_filter_copy_out_kcrs.tpl.c"
              } break;
              case LIBXSMM_DNN_DATATYPE_I8: {
                typedef char element_type;
#               include "template/libxsmm_dnn_filter_copy_out_kcrs.tpl.c"
              } break;
              default: {
                status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
              }
            }
          } break;
          default: {
            status = LIBXSMM_DNN_ERR_UNSUPPORTED_SRC_FORMAT;
          }
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


LIBXSMM_API_DEFINITION libxsmm_dnn_err_t libxsmm_dnn_bind_input_buffer(libxsmm_dnn_conv_handle* handle, const libxsmm_dnn_buffer* buffer)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  if (handle != 0 && buffer != 0) {
    /* check if format matches */
    if ( handle->desc.N == buffer->N
      && handle->ifwp == buffer->W
      && handle->ifhp == buffer->H
      && handle->ifmblock == buffer->bfm
      && handle->blocksifm == buffer->fmb
      && handle->datatype_in == buffer->datatype
      && handle->fm_lp_block == buffer->lpb
      && ((handle->buffer_format & buffer->format) > 0) )
    {
      handle->input = (libxsmm_dnn_buffer*)buffer;
    }
    else {
      status = LIBXSMM_DNN_ERR_MISMATCH_BUFFER;
    }
  }
  else {
    status = LIBXSMM_DNN_ERR_INVALID_HANDLE_BUFFER;
  }

  return status;
}


LIBXSMM_API_DEFINITION libxsmm_dnn_err_t libxsmm_dnn_bind_output_buffer(libxsmm_dnn_conv_handle* handle, const libxsmm_dnn_buffer* buffer)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  if (handle != 0 && buffer != 0) {
    /* check if format matches */
    if ( handle->desc.N == buffer->N
      && handle->ofwp == buffer->W
      && handle->ofhp == buffer->H
      && handle->ofmblock == buffer->bfm
      && handle->blocksofm == buffer->fmb
      && buffer->lpb == 1
      && ((handle->buffer_format & buffer->format) > 0)
      && handle->datatype_out == buffer->datatype )
    {
      handle->output = (libxsmm_dnn_buffer*)buffer;
    }
    else {
      status = LIBXSMM_DNN_ERR_MISMATCH_BUFFER;
    }
  }
  else {
    status = LIBXSMM_DNN_ERR_INVALID_HANDLE_BUFFER;
  }

  return status;
}


LIBXSMM_API_DEFINITION libxsmm_dnn_err_t libxsmm_dnn_bind_filter(libxsmm_dnn_conv_handle* handle, const libxsmm_dnn_filter* filter)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  if (handle != 0 && filter != 0) {
    /* check if format matches */
    if ( handle->desc.R == filter->R
      && handle->desc.S == filter->S
      && handle->ifmblock == filter->bifm
      && handle->blocksifm == filter->ifmb
      && handle->ofmblock == filter->bofm
      && handle->blocksofm == filter->ofmb
      && handle->fm_lp_block == filter->lpb
      && ((handle->filter_format & filter->format) > 0)
      && handle->datatype_in == filter->datatype)
    {
      handle->filter = (libxsmm_dnn_filter*)filter;
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


LIBXSMM_INLINE LIBXSMM_RETARGETABLE libxsmm_dnn_err_t internal_convolve_st(libxsmm_dnn_conv_handle* handle,
  libxsmm_dnn_conv_kind kind, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  if (0 != handle) {
    switch (kind) {
      case LIBXSMM_DNN_CONV_KIND_FWD: {
        switch (handle->buffer_format) {
          case LIBXSMM_DNN_CONV_FORMAT_LIBXSMM: {
            switch (handle->filter_format) {
              case LIBXSMM_DNN_CONV_FORMAT_LIBXSMM: {
                status = libxsmm_dnn_convolve_st_fwd_custom_custom(handle, start_thread, tid);
              } break;
              default: {
                status = LIBXSMM_DNN_ERR_INVALID_FORMAT_CONVOLVE;
              }
            }
          } break;
          case LIBXSMM_DNN_CONV_FORMAT_NHWC: {
            switch (handle->filter_format) {
              case LIBXSMM_DNN_CONV_FORMAT_RSCK: {
                status = libxsmm_dnn_convolve_st_fwd_nhwc_rsck(handle, start_thread, tid);
              } break;
              case LIBXSMM_DNN_CONV_FORMAT_LIBXSMM: {
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
      case LIBXSMM_DNN_CONV_KIND_BWD: {
        switch (handle->buffer_format) {
          case LIBXSMM_DNN_CONV_FORMAT_LIBXSMM: {
            switch (handle->filter_format) {
              case LIBXSMM_DNN_CONV_FORMAT_LIBXSMM: {
                status = libxsmm_dnn_convolve_st_bwd_custom_custom(handle, start_thread, tid);
              } break;
              default: {
                status = LIBXSMM_DNN_ERR_INVALID_FORMAT_CONVOLVE;
              }
            }
          } break;
          case LIBXSMM_DNN_CONV_FORMAT_NHWC: {
            switch (handle->filter_format) {
              case LIBXSMM_DNN_CONV_FORMAT_RSCK: {
                status = libxsmm_dnn_convolve_st_bwd_nhwc_rsck(handle, start_thread, tid);
              } break;
              case LIBXSMM_DNN_CONV_FORMAT_LIBXSMM: {
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
      case LIBXSMM_DNN_CONV_KIND_UPD: {
        switch (handle->buffer_format) {
          case LIBXSMM_DNN_CONV_FORMAT_LIBXSMM: {
            switch (handle->filter_format) {
              case LIBXSMM_DNN_CONV_FORMAT_LIBXSMM: {
                status = libxsmm_dnn_convolve_st_upd_custom_custom(handle, start_thread, tid);
              } break;
              default: {
                status = LIBXSMM_DNN_ERR_INVALID_FORMAT_CONVOLVE;
              }
            }
          } break;
          case LIBXSMM_DNN_CONV_FORMAT_NHWC: {
            switch (handle->filter_format) {
              case LIBXSMM_DNN_CONV_FORMAT_RSCK: {
                status = libxsmm_dnn_convolve_st_upd_nhwc_rsck(handle, start_thread, tid);
              } break;
              case LIBXSMM_DNN_CONV_FORMAT_LIBXSMM: {
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
  }
  else {
    status = LIBXSMM_DNN_ERR_INVALID_HANDLE;
  }

  return status;
}


LIBXSMM_API_DEFINITION void libxsmm_dnn_convolve(libxsmm_dnn_conv_handle* handle, libxsmm_dnn_conv_kind kind)
{
#if defined(_OPENMP)
# pragma omp parallel num_threads(handle->desc.threads)
  {
    const int tid = omp_get_thread_num();
    internal_convolve_st(handle, kind, 0, tid);
  }
#else
  internal_convolve_st(handle, kind, 0/*start_thread*/, 0/*tid*/);
#endif
}


LIBXSMM_API_DEFINITION libxsmm_dnn_err_t libxsmm_dnn_convolve_st(libxsmm_dnn_conv_handle* handle,
  libxsmm_dnn_conv_kind kind, /*unsigned*/int start_thread, /*unsigned*/int tid)
{
  return internal_convolve_st(handle, kind, start_thread, tid);
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
    libxsmm_build(&request, LIBXSMM_REGSIZE/*not managed*/, &code);
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
    libxsmm_build(&request, LIBXSMM_REGSIZE/*not managed*/, &code);
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
    libxsmm_build(&request, LIBXSMM_REGSIZE/*not managed*/, &code);
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
    libxsmm_build(&request, LIBXSMM_REGSIZE/*not managed*/, &code);
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
    libxsmm_build(&request, LIBXSMM_REGSIZE/*not managed*/, &code);
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
    libxsmm_build(&request, LIBXSMM_REGSIZE/*not managed*/, &code);
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

#endif /*defined(LIBXSMM_BUILD) || defined(LIBXSMM_DNN_INTERNAL_API)*/

