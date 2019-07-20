/******************************************************************************
** Copyright (c) 2016-2019, Intel Corporation                                **
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
#include "libxsmm_dnn_convolution_forward.h"
#include "libxsmm_dnn_convolution_backward.h"
#include "libxsmm_dnn_convolution_weight_update.h"

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#include <stdlib.h>
#include <string.h>
#include <math.h>
#if defined(_OPENMP)
# include <omp.h>
#endif
#if !defined(NDEBUG)
# include <stdio.h>
#endif
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif


LIBXSMM_API libxsmm_dnn_layer* libxsmm_dnn_create_conv_layer(
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
    handle->datatype_in = conv_desc.datatype_in;
    handle->datatype_out = conv_desc.datatype_out;
    /* select the intermediate format, only applicable for integer types */
    if ( (conv_desc.datatype_in == LIBXSMM_DNN_DATATYPE_F32) && (conv_desc.datatype_out != LIBXSMM_DNN_DATATYPE_F32) ) {
      /* error */
    } else if ( (conv_desc.datatype_in == LIBXSMM_DNN_DATATYPE_BF16) && (conv_desc.datatype_out != LIBXSMM_DNN_DATATYPE_BF16) ) {
      /* error */
    } else if ( (conv_desc.datatype_in == LIBXSMM_DNN_DATATYPE_I16) && (conv_desc.datatype_out != LIBXSMM_DNN_DATATYPE_F32) ) {
      /* error */
    } else if ( (conv_desc.datatype_in == LIBXSMM_DNN_DATATYPE_I8) && (conv_desc.datatype_out != LIBXSMM_DNN_DATATYPE_I32) ) {
      /* error */
    } else if ( (conv_desc.datatype_in == LIBXSMM_DNN_DATATYPE_I8) && (conv_desc.datatype_out != LIBXSMM_DNN_DATATYPE_F32) ) {
      /* error */
    } else {
      /* fine, no error */
    }
    handle->buffer_format = conv_desc.buffer_format;
    handle->filter_format = conv_desc.filter_format;
    handle->fuse_ops = conv_desc.fuse_ops;
    handle->post_bn = handle->desc.post_bn;
    handle->pre_bn = handle->desc.pre_bn;
    handle->fuse_batchstats_fwd = 0;
    handle->fuse_batchstats_bwd = 0;
    handle->fuse_eltwise_bwd = 0;
    handle->fuse_relu_bwd = 0;

    /* TODO: This check should be removed when this fuse flag is deprecated */
    if (handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_BATCH_STATS_FWD) {
      handle->fuse_batchstats_fwd = 1;
    }

    if (handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_RELU_BWD) {
      handle->fuse_relu_bwd = 1;
    }

    /* Enable batchnorm fusion depending on the input */
    if (handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_BATCHNORM_STATS) {
      if (handle->desc.post_bn != NULL) {
        handle->fuse_batchstats_fwd = 1;
      }
      if (handle->desc.pre_bn != NULL) {
        handle->fuse_batchstats_bwd = 1;
     }
    }

    handle->options = conv_desc.options;
    /* derive additional values */
    handle->ifhp = conv_desc.H + 2*conv_desc.pad_h_in;
    handle->ifwp = conv_desc.W + 2*conv_desc.pad_w_in;
    handle->ofh = (conv_desc.H + 2*conv_desc.pad_h - conv_desc.R) / conv_desc.u + 1;
    handle->ofw = (conv_desc.W + 2*conv_desc.pad_w - conv_desc.S) / conv_desc.v + 1;
    handle->ofhp = handle->ofh + 2*conv_desc.pad_h_out;
    handle->ofwp = handle->ofw + 2*conv_desc.pad_w_out;
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
    handle->blocksifm_blocking = 1;
    handle->blocksofm_blocking = 1;
    handle->upd_use_thread_fil = 0;
    handle->upd_use_external_reduce = 0;
    handle->filter_transposed = 0;
    /* Set algorithm to use */
    if (conv_desc.algo == LIBXSMM_DNN_CONV_ALGO_AUTO) {
      handle->algo = LIBXSMM_DNN_CONV_ALGO_DIRECT;
    } else {
      handle->algo = conv_desc.algo;
    }
    if ( handle->algo != LIBXSMM_DNN_CONV_ALGO_DIRECT ) {
      *status = LIBXSMM_DNN_ERR_INVALID_ALGO;
      free(handle);
      handle = 0;
      return 0;
    }
    *status = libxsmm_dnn_internal_create_conv_handle_direct( handle );
  }
  else {
    *status = LIBXSMM_DNN_ERR_CREATE_HANDLE;
  }
  /* account for eventually deallocated handle */
  if ( LIBXSMM_DNN_SUCCESS != *status ) {
    handle = 0;
  }
  return handle;
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_destroy_conv_layer(const libxsmm_dnn_layer* handle)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  if (0 != handle) {
    /* free internal structures and code of the handle */
    libxsmm_dnn_internal_free_structs_code_conv_handle( handle );

    /* Deallocate barrier */
    if (handle->barrier != 0 ) { libxsmm_barrier_release((const libxsmm_barrier*)handle->barrier); }

    /* deallocate handle structure itself */
    free(/*remove constness*/(libxsmm_dnn_layer*)handle);
  }

  return status;
}


LIBXSMM_API libxsmm_dnn_tensor_datalayout* libxsmm_dnn_create_tensor_datalayout(const libxsmm_dnn_layer* handle, const libxsmm_dnn_tensor_type type, libxsmm_dnn_err_t* status) {
  libxsmm_dnn_tensor_datalayout* layout;

  *status = LIBXSMM_DNN_SUCCESS;
  layout = 0;

  if (handle != 0) {
    layout = (libxsmm_dnn_tensor_datalayout*) malloc(sizeof(libxsmm_dnn_tensor_datalayout));

    if (layout != 0) {
      memset(layout, 0, sizeof(libxsmm_dnn_tensor_datalayout));
      if ( (type == LIBXSMM_DNN_REGULAR_INPUT)  || (type == LIBXSMM_DNN_GRADIENT_INPUT)  || (type == LIBXSMM_DNN_INPUT)  ||
           (type == LIBXSMM_DNN_REGULAR_OUTPUT) || (type == LIBXSMM_DNN_GRADIENT_OUTPUT) || (type == LIBXSMM_DNN_OUTPUT)    ) {
        layout->format = handle->buffer_format;
        layout->tensor_type = LIBXSMM_DNN_ACTIVATION;

        if ((handle->buffer_format & LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM) > 0) {
          if ( ((handle->datatype_in == LIBXSMM_DNN_DATATYPE_F32) && (handle->datatype_out == LIBXSMM_DNN_DATATYPE_F32) ) ) {
            layout->datatype = LIBXSMM_DNN_DATATYPE_F32;
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
                *status = LIBXSMM_DNN_ERR_UNKNOWN_TENSOR_TYPE;
              }
            } else {
              free(layout);
              layout = 0; /* make sure a NULL is returned */
              *status = LIBXSMM_DNN_ERR_UNKNOWN_TENSOR_TYPE;
            }
          /* @TODO this need to change */
          } else if ( (handle->datatype_in == LIBXSMM_DNN_DATATYPE_I16) && (handle->datatype_out == LIBXSMM_DNN_DATATYPE_I32) ) {
            if ( ( (type == LIBXSMM_DNN_REGULAR_INPUT) || (type == LIBXSMM_DNN_INPUT) )  ) {
              layout->datatype = handle->datatype_in;
            } else if ( (type == LIBXSMM_DNN_REGULAR_OUTPUT) || (type == LIBXSMM_DNN_OUTPUT) ) {
              layout->datatype = handle->datatype_out;
            }
            layout->dim_type = (libxsmm_dnn_tensor_dimtype*) malloc(5*sizeof(libxsmm_dnn_tensor_dimtype));
            layout->dim_size = (unsigned int*) malloc(5*sizeof(unsigned int));
            if (0 != layout->dim_type && 0 != layout->dim_size) { /* TODO: handle the error */
              layout->num_dims = 5;
              layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
              layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_W;
              layout->dim_type[2] = LIBXSMM_DNN_TENSOR_DIMTYPE_H;
              layout->dim_type[3] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
              layout->dim_type[4] = LIBXSMM_DNN_TENSOR_DIMTYPE_N;
              if ( (type == LIBXSMM_DNN_REGULAR_INPUT) || (type == LIBXSMM_DNN_GRADIENT_INPUT) || (type == LIBXSMM_DNN_INPUT) )   {
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
                *status = LIBXSMM_DNN_ERR_UNKNOWN_TENSOR_TYPE;
              }
            }
          } else if ( (handle->datatype_in == LIBXSMM_DNN_DATATYPE_BF16) && (handle->datatype_out == LIBXSMM_DNN_DATATYPE_BF16) ) {
            layout->datatype = LIBXSMM_DNN_DATATYPE_BF16;
            layout->dim_type = (libxsmm_dnn_tensor_dimtype*) malloc(6*sizeof(libxsmm_dnn_tensor_dimtype));
            layout->dim_size = (unsigned int*) malloc(6*sizeof(unsigned int));
            if (0 != layout->dim_type && 0 != layout->dim_size) { /* TODO: handle the error */
              layout->num_dims = 5;
              layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
              layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_W;
              layout->dim_type[2] = LIBXSMM_DNN_TENSOR_DIMTYPE_H;
              layout->dim_type[3] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
              layout->dim_type[4] = LIBXSMM_DNN_TENSOR_DIMTYPE_N;
              if ( (type == LIBXSMM_DNN_REGULAR_INPUT) || (type == LIBXSMM_DNN_GRADIENT_INPUT) || (type == LIBXSMM_DNN_INPUT) )   {
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
                *status = LIBXSMM_DNN_ERR_UNKNOWN_TENSOR_TYPE;
              }
            }
          } else if ( ((handle->datatype_in == LIBXSMM_DNN_DATATYPE_I16) && (handle->datatype_out == LIBXSMM_DNN_DATATYPE_F32)) ||  ((handle->datatype_in == LIBXSMM_DNN_DATATYPE_I8) && (handle->datatype_out == LIBXSMM_DNN_DATATYPE_I32))  ) {
            if ( ( (type == LIBXSMM_DNN_REGULAR_INPUT) || (type == LIBXSMM_DNN_INPUT) || (type == LIBXSMM_DNN_GRADIENT_OUTPUT)  )  ) {
              layout->datatype = handle->datatype_in;
            } else if ( (type == LIBXSMM_DNN_REGULAR_OUTPUT) || (type == LIBXSMM_DNN_OUTPUT) || (type == LIBXSMM_DNN_GRADIENT_INPUT) ) {
              layout->datatype = handle->datatype_out;
            }
            layout->dim_type = (libxsmm_dnn_tensor_dimtype*) malloc(5*sizeof(libxsmm_dnn_tensor_dimtype));
            layout->dim_size = (unsigned int*) malloc(5*sizeof(unsigned int));
            if (0 != layout->dim_type && 0 != layout->dim_size) { /* TODO: handle the error */
              if ( (type == LIBXSMM_DNN_REGULAR_INPUT) || (type == LIBXSMM_DNN_INPUT) )   {
                layout->num_dims = 5;
                layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
                layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_W;
                layout->dim_type[2] = LIBXSMM_DNN_TENSOR_DIMTYPE_H;
                layout->dim_type[3] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
                layout->dim_type[4] = LIBXSMM_DNN_TENSOR_DIMTYPE_N;
                layout->dim_size[0] = handle->ifmblock;
                layout->dim_size[1] = handle->ifwp;
                layout->dim_size[2] = handle->ifhp;
                layout->dim_size[3] = handle->blocksifm;
                layout->dim_size[4] = handle->desc.N;
              } else if ( type == LIBXSMM_DNN_GRADIENT_OUTPUT )   {
                layout->num_dims = 5;
                layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
                layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_W;
                layout->dim_type[2] = LIBXSMM_DNN_TENSOR_DIMTYPE_H;
                layout->dim_type[3] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
                layout->dim_type[4] = LIBXSMM_DNN_TENSOR_DIMTYPE_N;
                layout->dim_size[0] = handle->ofmblock;
                layout->dim_size[1] = handle->ofwp;
                layout->dim_size[2] = handle->ofhp;
                layout->dim_size[3] = handle->blocksofm;
                layout->dim_size[4] = handle->desc.N;
              } else if ( (type == LIBXSMM_DNN_REGULAR_OUTPUT) || (type == LIBXSMM_DNN_OUTPUT) ) {
                layout->num_dims = 5;
                layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
                layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_W;
                layout->dim_type[2] = LIBXSMM_DNN_TENSOR_DIMTYPE_H;
                layout->dim_type[3] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
                layout->dim_type[4] = LIBXSMM_DNN_TENSOR_DIMTYPE_N;
                layout->dim_size[0] = handle->ofmblock;
                layout->dim_size[1] = handle->ofwp;
                layout->dim_size[2] = handle->ofhp;
                layout->dim_size[3] = handle->blocksofm;
                layout->dim_size[4] = handle->desc.N;
              } else if ( type == LIBXSMM_DNN_GRADIENT_INPUT ) {
                layout->num_dims = 5;
                layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
                layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_W;
                layout->dim_type[2] = LIBXSMM_DNN_TENSOR_DIMTYPE_H;
                layout->dim_type[3] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
                layout->dim_type[4] = LIBXSMM_DNN_TENSOR_DIMTYPE_N;
                layout->dim_size[0] = handle->ifmblock;
                layout->dim_size[1] = handle->ifwp;
                layout->dim_size[2] = handle->ifhp;
                layout->dim_size[3] = handle->blocksifm;
                layout->dim_size[4] = handle->desc.N;
              } else {
                free(layout->dim_type);
                free(layout->dim_size);
                free(layout);
                layout = 0; /* make sure a NULL is returned */
                *status = LIBXSMM_DNN_ERR_UNKNOWN_TENSOR_TYPE;
              }
            }
          } else {
            free(layout);
            layout = 0; /* make sure a NULL is returned */
            *status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
          }
        } else if ((handle->buffer_format & LIBXSMM_DNN_TENSOR_FORMAT_NHWC) > 0) {
          if ( ((handle->datatype_in == LIBXSMM_DNN_DATATYPE_F32) && (handle->datatype_out == LIBXSMM_DNN_DATATYPE_F32) ) ) {
            layout->datatype = LIBXSMM_DNN_DATATYPE_F32;
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
                *status = LIBXSMM_DNN_ERR_UNKNOWN_TENSOR_TYPE;
              }
            }
          } else {
            free(layout);
            layout = 0; /* make sure a NULL is returned */
            *status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
          }
        } else {
          free(layout);
          layout = 0; /* make sure a NULL is returned */
          *status = LIBXSMM_DNN_ERR_INVALID_FORMAT_GENERAL;
        }
      } else if ( (type == LIBXSMM_DNN_REGULAR_FILTER) || (type == LIBXSMM_DNN_GRADIENT_FILTER) || (type == LIBXSMM_DNN_FILTER) ) {
        layout->format = handle->filter_format;
        layout->tensor_type = LIBXSMM_DNN_FILTER;

        if ((handle->filter_format & LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM) > 0) {
          if ( (handle->datatype_in == LIBXSMM_DNN_DATATYPE_F32) && (handle->datatype_out == LIBXSMM_DNN_DATATYPE_F32) ) {
            layout->datatype = LIBXSMM_DNN_DATATYPE_F32;
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
              layout->dim_size[4] = handle->blocksifm;
              layout->dim_size[5] = handle->blocksofm;
            }
          } else if ( (handle->datatype_in == LIBXSMM_DNN_DATATYPE_BF16) && (handle->datatype_out == LIBXSMM_DNN_DATATYPE_BF16) ) {
            layout->datatype = LIBXSMM_DNN_DATATYPE_BF16;
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
              layout->dim_size[2] = handle->ifmblock/handle->fm_lp_block;
              layout->dim_size[3] = handle->desc.S;
              layout->dim_size[4] = handle->desc.R;
              layout->dim_size[5] = handle->blocksifm;
              layout->dim_size[6] = handle->blocksofm;
            }
          } else if ( ((handle->datatype_in == LIBXSMM_DNN_DATATYPE_I16) && (handle->datatype_out == LIBXSMM_DNN_DATATYPE_F32)) || ((handle->datatype_in == LIBXSMM_DNN_DATATYPE_I8) && (handle->datatype_out == LIBXSMM_DNN_DATATYPE_I32)) ) {
            if ( (type == LIBXSMM_DNN_REGULAR_FILTER) || (type == LIBXSMM_DNN_FILTER) ) {
              layout->datatype = handle->datatype_in;
            } else if (type == LIBXSMM_DNN_GRADIENT_FILTER) {
              layout->datatype = handle->datatype_out;
            }
            layout->dim_type = (libxsmm_dnn_tensor_dimtype*) malloc(7*sizeof(libxsmm_dnn_tensor_dimtype));
            layout->dim_size = (unsigned int*) malloc(7*sizeof(unsigned int));
            if (0 != layout->dim_type && 0 != layout->dim_size) { /* TODO: handle the error */
              if ((type == LIBXSMM_DNN_REGULAR_FILTER) || (type == LIBXSMM_DNN_FILTER)) {
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
                layout->dim_size[2] = handle->ifmblock/handle->fm_lp_block;
                layout->dim_size[3] = handle->desc.S;
                layout->dim_size[4] = handle->desc.R;
                layout->dim_size[5] = handle->blocksifm;
                layout->dim_size[6] = handle->blocksofm;
              }
            }
          } else {
            free(layout);
            layout = 0; /* make sure a NULL is returned */
            *status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
          }
        } else if ((handle->filter_format & LIBXSMM_DNN_TENSOR_FORMAT_RSCK) > 0) {
          if ( (handle->datatype_in == LIBXSMM_DNN_DATATYPE_F32) && (handle->datatype_out == LIBXSMM_DNN_DATATYPE_F32) ) {
            layout->datatype = LIBXSMM_DNN_DATATYPE_F32;
            layout->dim_type = (libxsmm_dnn_tensor_dimtype*) malloc(4*sizeof(libxsmm_dnn_tensor_dimtype));
            layout->dim_size = (unsigned int*) malloc(4*sizeof(unsigned int));
            if (0 != layout->dim_type && 0 != layout->dim_size) { /* TODO: handle the error */
              layout->num_dims = 4;
              layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
              layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
              layout->dim_type[2] = LIBXSMM_DNN_TENSOR_DIMTYPE_S;
              layout->dim_type[3] = LIBXSMM_DNN_TENSOR_DIMTYPE_R;
              layout->dim_size[0] = handle->ofmblock * handle->blocksofm;
              layout->dim_size[1] = handle->ifmblock * handle->blocksifm;
              layout->dim_size[2] = handle->desc.S;
              layout->dim_size[3] = handle->desc.R;
            }
          } else {
            free(layout);
            layout = 0; /* make sure a NULL is returned */
            *status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
          }
        } else {
          free(layout);
          layout = 0; /* make sure a NULL is returned */
          *status = LIBXSMM_DNN_ERR_INVALID_FORMAT_GENERAL;
        }
      } else if ( type == LIBXSMM_DNN_REGULAR_FILTER_TRANS ) {
        layout->format = handle->filter_format;
        layout->tensor_type = LIBXSMM_DNN_REGULAR_FILTER_TRANS;

        if ((handle->filter_format & LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM) > 0) {
          if ( (handle->datatype_in == LIBXSMM_DNN_DATATYPE_F32) && (handle->datatype_out == LIBXSMM_DNN_DATATYPE_F32) ) {
            layout->datatype = LIBXSMM_DNN_DATATYPE_F32;
            layout->dim_type = (libxsmm_dnn_tensor_dimtype*) malloc(6*sizeof(libxsmm_dnn_tensor_dimtype));
            layout->dim_size = (unsigned int*) malloc(6*sizeof(unsigned int));
            if (0 != layout->dim_type && 0 != layout->dim_size) { /* TODO: handle the error */
              layout->num_dims = 6;
              layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
              layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
              layout->dim_type[2] = LIBXSMM_DNN_TENSOR_DIMTYPE_S;
              layout->dim_type[3] = LIBXSMM_DNN_TENSOR_DIMTYPE_R;
              layout->dim_type[4] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
              layout->dim_type[5] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
              layout->dim_size[0] = handle->ifmblock;
              layout->dim_size[1] = handle->ofmblock;
              layout->dim_size[2] = handle->desc.S;
              layout->dim_size[3] = handle->desc.R;
              layout->dim_size[4] = handle->blocksofm;
              layout->dim_size[5] = handle->blocksifm;
            }
          } else if ( (handle->datatype_in == LIBXSMM_DNN_DATATYPE_BF16) && (handle->datatype_out == LIBXSMM_DNN_DATATYPE_BF16) ) {
            layout->datatype = LIBXSMM_DNN_DATATYPE_BF16;
            layout->dim_type = (libxsmm_dnn_tensor_dimtype*) malloc(7*sizeof(libxsmm_dnn_tensor_dimtype));
            layout->dim_size = (unsigned int*) malloc(7*sizeof(unsigned int));
            if (0 != layout->dim_type && 0 != layout->dim_size) { /* TODO: handle the error */
              layout->num_dims = 7;
              layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
              layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
              layout->dim_type[2] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
              layout->dim_type[3] = LIBXSMM_DNN_TENSOR_DIMTYPE_S;
              layout->dim_type[4] = LIBXSMM_DNN_TENSOR_DIMTYPE_R;
              layout->dim_type[5] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
              layout->dim_type[6] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
              layout->dim_size[0] = handle->fm_lp_block;
              layout->dim_size[1] = handle->ifmblock;
              layout->dim_size[2] = handle->ofmblock/handle->fm_lp_block;
              layout->dim_size[3] = handle->desc.S;
              layout->dim_size[4] = handle->desc.R;
              layout->dim_size[5] = handle->blocksofm;
              layout->dim_size[6] = handle->blocksifm;
            }
          } else {
            free(layout);
            layout = 0; /* make sure a NULL is returned */
            *status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
          }
#if 0
        } else if ((handle->filter_format & LIBXSMM_DNN_TENSOR_FORMAT_RSCK) > 0) {
          if ( (handle->datatype_in == LIBXSMM_DNN_DATATYPE_F32) && (handle->datatype_out == LIBXSMM_DNN_DATATYPE_F32) ) {
            layout->datatype = LIBXSMM_DNN_DATATYPE_F32;
            layout->dim_type = (libxsmm_dnn_tensor_dimtype*) malloc(4*sizeof(libxsmm_dnn_tensor_dimtype));
            layout->dim_size = (unsigned int*) malloc(4*sizeof(unsigned int));
            if (0 != layout->dim_type && 0 != layout->dim_size) { /* TODO: handle the error */
              layout->num_dims = 4;
              layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
              layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
              layout->dim_type[2] = LIBXSMM_DNN_TENSOR_DIMTYPE_S;
              layout->dim_type[3] = LIBXSMM_DNN_TENSOR_DIMTYPE_R;
              layout->dim_size[0] = handle->ofmblock * handle->blocksofm;
              layout->dim_size[1] = handle->ifmblock * handle->blocksifm;
              layout->dim_size[2] = handle->desc.S;
              layout->dim_size[3] = handle->desc.K;
            }
          } else {
            free(layout);
            layout = 0; /* make sure a NULL is returned */
            *status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
          }
#endif
        } else {
          free(layout);
          layout = 0; /* make sure a NULL is returned */
          *status = LIBXSMM_DNN_ERR_INVALID_FORMAT_GENERAL;
        }
      } else if ( (type == LIBXSMM_DNN_REGULAR_CHANNEL_BIAS) || (type == LIBXSMM_DNN_GRADIENT_CHANNEL_BIAS) || (type == LIBXSMM_DNN_CHANNEL_BIAS) ) {
        layout->format = handle->buffer_format;
        layout->tensor_type = LIBXSMM_DNN_CHANNEL_SCALAR;

        if ((handle->buffer_format & LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM) > 0) {
          if ( handle->datatype_out == LIBXSMM_DNN_DATATYPE_F32 ) {
            layout->datatype = handle->datatype_out;
            layout->dim_type = (libxsmm_dnn_tensor_dimtype*) malloc(2*sizeof(libxsmm_dnn_tensor_dimtype));
            layout->dim_size = (unsigned int*) malloc(2*sizeof(unsigned int));

            if (0 != layout->dim_type && 0 != layout->dim_size) { /* TODO: handle the error */
              layout->num_dims = 2;
              layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
              layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
              layout->dim_size[0] = handle->ofmblock;
              layout->dim_size[1] = handle->blocksofm;
            }
#if 0
          } else if ( (handle->datatype_in == LIBXSMM_DNN_DATATYPE_I16) || (handle->datatype_in == LIBXSMM_DNN_DATATYPE_I8) ) {
            layout->dim_type = (libxsmm_dnn_tensor_dimtype*) malloc(3*sizeof(libxsmm_dnn_tensor_dimtype));
            layout->dim_size = (unsigned int*) malloc(3*sizeof(unsigned int));

            if (0 != layout->dim_type && 0 != layout->dim_size) { /* TODO: handle the error */
              layout->num_dims = 3;
              layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
              layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
              layout->dim_type[2] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
              layout->dim_size[0] = handle->fm_lp_block;
              layout->dim_size[1] = handle->ofmblock;
              layout->dim_size[2] = handle->blocksofm;
            }
#endif
          } else {
            free(layout);
            layout = 0; /* make sure a NULL is returned */
            *status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
          }
        } else if ((handle->buffer_format & LIBXSMM_DNN_TENSOR_FORMAT_NHWC) > 0) {
          layout->datatype = handle->datatype_out;
          if ( handle->datatype_in == LIBXSMM_DNN_DATATYPE_F32 ) {
            layout->dim_type = (libxsmm_dnn_tensor_dimtype*) malloc(1*sizeof(libxsmm_dnn_tensor_dimtype));
            layout->dim_size = (unsigned int*) malloc(1*sizeof(unsigned int));

            if (0 != layout->dim_type && 0 != layout->dim_size) { /* TODO: handle the error */
              layout->num_dims = 1;
              layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
              layout->dim_size[0] = handle->ofmblock*handle->blocksofm;
            }
          } else {
            free(layout);
            layout = 0; /* make sure a NULL is returned */
            *status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
          }
        } else {
          free(layout);
          layout = 0; /* make sure a NULL is returned */
          *status = LIBXSMM_DNN_ERR_INVALID_FORMAT_GENERAL;
        }
      } else if ( (type == LIBXSMM_DNN_BATCH_STATS) ) {
        layout->format = handle->buffer_format;
        layout->tensor_type = LIBXSMM_DNN_BATCH_STATS;

        if ((handle->buffer_format & LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM) > 0) {
          if ( (handle->datatype_out == LIBXSMM_DNN_DATATYPE_F32) || (handle->datatype_out == LIBXSMM_DNN_DATATYPE_BF16) ) {
            layout->datatype = LIBXSMM_DNN_DATATYPE_F32;
            layout->dim_type = (libxsmm_dnn_tensor_dimtype*) malloc(4*sizeof(libxsmm_dnn_tensor_dimtype));
            layout->dim_size = (unsigned int*) malloc(4*sizeof(unsigned int));

            if (0 != layout->dim_type && 0 != layout->dim_size) { /* TODO: handle the error */
              layout->num_dims = 2;
              layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
              layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_N;
              layout->dim_type[2] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
              layout->dim_type[3] = LIBXSMM_DNN_TENSOR_DIMTYPE_X;
              layout->dim_size[0] = handle->ofmblock;
              layout->dim_size[1] = handle->desc.N;
              layout->dim_size[2] = handle->blocksofm;
              layout->dim_size[3] = 2;
            }
          } else {
            free(layout);
            layout = 0; /* make sure a NULL is returned */
            *status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
          }
        } else {
          free(layout);
          layout = 0; /* make sure a NULL is returned */
          *status = LIBXSMM_DNN_ERR_INVALID_FORMAT_GENERAL;
        }
      } else if (type == LIBXSMM_DNN_MAX_STATS_FWD) {
        layout->format = handle->buffer_format;
        layout->tensor_type = LIBXSMM_DNN_MAX_STATS_FWD;
        layout->datatype = LIBXSMM_DNN_DATATYPE_F32;
        layout->dim_type = (libxsmm_dnn_tensor_dimtype*) malloc(2*sizeof(libxsmm_dnn_tensor_dimtype));
        layout->dim_size = (unsigned int*) malloc(2*sizeof(unsigned int));
        if (0 != layout->dim_type && 0 != layout->dim_size) { /* TODO: handle the error */
          layout->num_dims = 2;
          layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
          layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_N;
          layout->dim_size[0] = handle->ifmblock;
          layout->dim_size[1] = handle->desc.N;
        }
      } else if (type == LIBXSMM_DNN_MAX_STATS_BWD) {
        layout->format = handle->buffer_format;
        layout->tensor_type = LIBXSMM_DNN_MAX_STATS_BWD;
        layout->datatype = LIBXSMM_DNN_DATATYPE_F32;
        layout->dim_type = (libxsmm_dnn_tensor_dimtype*) malloc(2*sizeof(libxsmm_dnn_tensor_dimtype));
        layout->dim_size = (unsigned int*) malloc(2*sizeof(unsigned int));
        if (0 != layout->dim_type && 0 != layout->dim_size) { /* TODO: handle the error */
          layout->num_dims = 2;
          layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
          layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_N;
          layout->dim_size[0] = handle->ifmblock;
          layout->dim_size[1] = handle->desc.N;
        }
      } else if (type == LIBXSMM_DNN_MAX_STATS_UPD) {
        layout->format = handle->buffer_format;
        layout->tensor_type = LIBXSMM_DNN_MAX_STATS_UPD;
        layout->datatype = LIBXSMM_DNN_DATATYPE_F32;
        layout->dim_type = (libxsmm_dnn_tensor_dimtype*) malloc(2*sizeof(libxsmm_dnn_tensor_dimtype));
        layout->dim_size = (unsigned int*) malloc(2*sizeof(unsigned int));
        if (0 != layout->dim_type && 0 != layout->dim_size) { /* TODO: handle the error */
          layout->num_dims = 2;
          layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
          layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_N;
          layout->dim_size[0] = handle->ifmblock;
          layout->dim_size[1] = handle->desc.N;
        }
      } else {
        free(layout);
        layout = 0; /* make sure a NULL is returned */
        *status = LIBXSMM_DNN_ERR_UNKNOWN_TENSOR_TYPE;
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



LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_trans_reg_filter(const libxsmm_dnn_layer* handle) {
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  if (handle != 0) {
    if ( (handle->reg_filter != 0) && (handle->reg_filter_tr != 0) ) {
      /* TODO handle more datatypes */
      int ifm1, ifm2, kj, ki, ofm1, ofm2;
      LIBXSMM_VLA_DECL(6, float, wt, (float*)handle->reg_filter->data, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock);
      LIBXSMM_VLA_DECL(6, float, tr_wt, (float*)handle->reg_filter_tr->data, handle->blocksofm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock);

      /* TODO we might want to do this in parallel.... */
      for ( ifm1 = 0; ifm1 < handle->blocksifm; ++ifm1 ) {
        for ( ofm1 = 0; ofm1 < handle->blocksofm; ++ofm1 ) {
          for (kj=0; kj < handle->desc.R; ++kj) {
            for (ki=0; ki < handle->desc.S; ++ki) {
              for ( ofm2 = 0; ofm2 < handle->ofmblock; ++ofm2 ) {
                for ( ifm2 = 0; ifm2 < handle->ifmblock; ++ifm2 ) {
                  LIBXSMM_VLA_ACCESS(6, tr_wt, ifm1, ofm1, handle->desc.R-1-kj, handle->desc.S-1-ki, ofm2, ifm2, handle->blocksofm, handle->desc.R, handle->desc.S, handle->ofmblock, handle->ifmblock) =
                    LIBXSMM_VLA_ACCESS(6, wt, ofm1, ifm1, kj, ki, ifm2, ofm2, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock);
                }
              }
            }
          }
        }
      }
    } else {
      status = LIBXSMM_DNN_ERR_INVALID_TENSOR;
    }
  } else {
    status = LIBXSMM_DNN_ERR_INVALID_HANDLE;
  }

  return status;
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_bind_tensor(libxsmm_dnn_layer* handle, const libxsmm_dnn_tensor* tensor, const libxsmm_dnn_tensor_type type)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  /* check for tensor type */
  if ( (type != LIBXSMM_DNN_REGULAR_INPUT)        && (type != LIBXSMM_DNN_GRADIENT_INPUT)  &&
      (type != LIBXSMM_DNN_REGULAR_OUTPUT)       && (type != LIBXSMM_DNN_GRADIENT_OUTPUT) &&
      (type != LIBXSMM_DNN_REGULAR_FILTER)       && (type != LIBXSMM_DNN_GRADIENT_FILTER) &&
      (type != LIBXSMM_DNN_REGULAR_CHANNEL_BIAS)         && (type != LIBXSMM_DNN_GRADIENT_CHANNEL_BIAS)   &&
      (type != LIBXSMM_DNN_REGULAR_FILTER_TRANS) && (type != LIBXSMM_DNN_BATCH_STATS) && (type != LIBXSMM_DNN_MAX_STATS_FWD) && (type != LIBXSMM_DNN_MAX_STATS_BWD)  && (type != LIBXSMM_DNN_MAX_STATS_UPD)  ) {
    status = LIBXSMM_DNN_ERR_UNKNOWN_TENSOR_TYPE;
    return status;
  }

  if (handle != 0 && tensor != 0) {
    libxsmm_dnn_tensor_datalayout* handle_layout = libxsmm_dnn_create_tensor_datalayout(handle, type, &status);

    if ( libxsmm_dnn_compare_tensor_datalayout(handle_layout, tensor->layout, &status) == 0 ) {
      if ( type == LIBXSMM_DNN_REGULAR_INPUT ) {
        handle->reg_input = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_GRADIENT_INPUT ) {
        handle->grad_input = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_REGULAR_OUTPUT ) {
        handle->reg_output = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_GRADIENT_OUTPUT ) {
        handle->grad_output = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_REGULAR_FILTER ) {
        handle->reg_filter = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_GRADIENT_FILTER ) {
        handle->grad_filter = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_REGULAR_CHANNEL_BIAS ) {
        handle->reg_bias = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_GRADIENT_CHANNEL_BIAS ) {
        handle->grad_bias = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_REGULAR_FILTER_TRANS ) {
        handle->reg_filter_tr = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_BATCH_STATS ) {
        handle->batch_stats = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_MAX_STATS_FWD ) {
        handle->maxstats_fwd = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_MAX_STATS_BWD ) {
        handle->maxstats_bwd = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_MAX_STATS_UPD ) {
        handle->maxstats_upd = (libxsmm_dnn_tensor*)tensor;
      } else {
        /* cannot happen */
      }
    } else {
      status = LIBXSMM_DNN_ERR_MISMATCH_TENSOR;
    }

    libxsmm_dnn_destroy_tensor_datalayout( handle_layout );
  }
  else {
    status = LIBXSMM_DNN_ERR_INVALID_HANDLE_TENSOR;
  }

  return status;
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_release_tensor(libxsmm_dnn_layer* handle, const libxsmm_dnn_tensor_type type)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  /* check for tensor type */
  if ( (type != LIBXSMM_DNN_REGULAR_INPUT)        && (type != LIBXSMM_DNN_GRADIENT_INPUT)  &&
      (type != LIBXSMM_DNN_REGULAR_OUTPUT)       && (type != LIBXSMM_DNN_GRADIENT_OUTPUT) &&
      (type != LIBXSMM_DNN_REGULAR_FILTER)       && (type != LIBXSMM_DNN_GRADIENT_FILTER) &&
      (type != LIBXSMM_DNN_REGULAR_CHANNEL_BIAS)         && (type != LIBXSMM_DNN_GRADIENT_CHANNEL_BIAS)   &&
      (type != LIBXSMM_DNN_REGULAR_FILTER_TRANS) && (type != LIBXSMM_DNN_BATCH_STATS) && (type != LIBXSMM_DNN_MAX_STATS_FWD) && (type != LIBXSMM_DNN_MAX_STATS_BWD)  && (type != LIBXSMM_DNN_MAX_STATS_UPD)  ) {
    status = LIBXSMM_DNN_ERR_UNKNOWN_TENSOR_TYPE;
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
    } else if ( type == LIBXSMM_DNN_REGULAR_FILTER ) {
      handle->reg_filter = 0;
    } else if ( type == LIBXSMM_DNN_GRADIENT_FILTER ) {
      handle->grad_filter = 0;
    } else if ( type == LIBXSMM_DNN_REGULAR_CHANNEL_BIAS ) {
      handle->reg_bias = 0;
    } else if ( type == LIBXSMM_DNN_GRADIENT_CHANNEL_BIAS ) {
      handle->grad_bias = 0;
    } else if ( type == LIBXSMM_DNN_REGULAR_FILTER_TRANS ) {
      handle->reg_filter_tr = 0;
    } else if ( type == LIBXSMM_DNN_BATCH_STATS ) {
      handle->batch_stats = 0;
    } else if ( type == LIBXSMM_DNN_MAX_STATS_FWD ) {
      handle->maxstats_fwd = 0;
    } else if ( type == LIBXSMM_DNN_MAX_STATS_BWD ) {
      handle->maxstats_bwd = 0;
    } else if ( type == LIBXSMM_DNN_MAX_STATS_UPD ) {
      handle->maxstats_upd = 0;
    } else {
      /* cannot happen */
    }
  } else {
    status = LIBXSMM_DNN_ERR_INVALID_HANDLE_TENSOR;
  }

  return status;
}


LIBXSMM_API size_t libxsmm_dnn_get_scratch_size(const libxsmm_dnn_layer* handle, const libxsmm_dnn_compute_kind kind, libxsmm_dnn_err_t* status)
{
  size_t l_scratch_size = 0;
  *status = LIBXSMM_DNN_SUCCESS;

  if (0 != handle) {
      switch (kind) {
        case LIBXSMM_DNN_COMPUTE_KIND_FWD: {
                                             if (handle->padding_flag == 1) {
                                               l_scratch_size = handle->fwdbwd_scratch_size + 64;
                                             }
                                             l_scratch_size += handle->max_scratch5_size + 64;
                                             l_scratch_size += handle->scratch6_size + 64;
                                             l_scratch_size += handle->scratch7_size + 64;
                                           } break;
        case LIBXSMM_DNN_COMPUTE_KIND_BWD: {
                                             /* we need filter for transpose, + 64 to do alignment while performing bind, scratch1 */
                                             l_scratch_size = handle->scratch1_size + 64;
                                             l_scratch_size += handle->fwdbwd_scratch_size + 64;
                                             l_scratch_size += handle->max_scratch5_size + 64;
                                             l_scratch_size += handle->scratch7_size + 64;
                                           } break;
        case LIBXSMM_DNN_COMPUTE_KIND_UPD: {
                                             if (handle->use_lp_kernel == 1) {
                                               l_scratch_size += handle->scratch2_size + 64;
                                             }
                                             /* we need a minibatch copy for transpose of input, scratch3 */
                                             l_scratch_size += handle->scratch3_size + 64;
                                             /* potentially we need thread-local filter copies, scratch4 */
                                             if (handle->upd_use_thread_fil == 1) {
                                               l_scratch_size += handle->scratch4_size + 64;
                                             }
                                             l_scratch_size += handle->max_scratch5_size + 64;
                                             l_scratch_size += handle->minibatch_scratch_size + 64;
                                             l_scratch_size += handle->scratch6_size + 64;
                                             if (handle->scratch7_size != 0) {
                                               l_scratch_size += handle->scratch7_size + 64;
                                             }
                                           } break;
        case LIBXSMM_DNN_COMPUTE_KIND_ALL: {
                                             /* we need filter for transpose, + 64 to do alignment while performing bind, scratch1 */
                                             l_scratch_size += handle->scratch1_size + 64;
                                             if (handle->use_lp_kernel == 1) {
                                               l_scratch_size += handle->scratch2_size + 64;
                                             }
                                             /* we need a minibatch copy for transpose of input, scratch3 */
                                             l_scratch_size += handle->scratch3_size + 64;
                                             /* potentially we need thread-local filter copies, scratch4 */
                                             if (handle->upd_use_thread_fil == 1) {
                                               l_scratch_size += handle->scratch4_size + 64;
                                             }
                                             l_scratch_size += handle->max_scratch5_size + 64;
                                             if (handle->scratch6_size != 0) {
                                               l_scratch_size += handle->scratch6_size + 64;
                                             }
                                             if (handle->scratch7_size != 0) {
                                               l_scratch_size += handle->scratch7_size + 64;
                                             }
                                           } break;
        default: {
          *status = LIBXSMM_DNN_ERR_INVALID_KIND;
        }
      }
  } else {
    *status = LIBXSMM_DNN_ERR_INVALID_HANDLE;
  }

  return l_scratch_size;
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_bind_scratch(libxsmm_dnn_layer* handle, const libxsmm_dnn_compute_kind kind, const void* scratch)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
  uintptr_t address = (uintptr_t)scratch;
  size_t offset = 0;

  if (scratch == 0) {
    status = LIBXSMM_DNN_ERR_SCRATCH_NOT_ALLOCED;
    /* check this if, this is bogus, not sure why there */
#if 0
    if ( (kind == LIBXSMM_DNN_COMPUTE_KIND_FWD) && (handle->datatype_in == handle->datatype_out) ) {
      status = LIBXSMM_DNN_SUCCESS;
    }
#endif
    return status;
  }

  if (0 != handle) {
      switch (kind) {
        case LIBXSMM_DNN_COMPUTE_KIND_FWD: {
                                             if (address % 64 == 0) {
                                               handle->scratch1 = (void*)address;
                                             } else {
                                               offset = (64 - address % 64);
                                               handle->scratch1 = (void*)(address+offset);
                                             }
                                             address += handle->scratch1_size + 64;
                                             if (address % 64 == 0) {
                                               handle->scratch5 = (void*)address;
                                             } else {
                                               offset = (64 - address % 64);
                                               handle->scratch5 = (void*)(address+offset);
                                             }
                                             address += handle->max_scratch5_size + 64;
                                             if (handle->scratch6_size != 0) {
                                               if (address % 64 == 0) {
                                                 handle->scratch6 = (void*)address;
                                               }
                                               else {
                                                 offset = (64 - address % 64);
                                                 handle->scratch6 = (void*)(address + offset);
                                               }
                                               address += handle->scratch6_size + 64;
                                             }
                                             if (handle->scratch7_size != 0) {
                                               if (address % 64 == 0) {
                                                 handle->scratch7 = (void*)address;
                                               } else {
                                                 offset = (64 - address % 64);
                                                 handle->scratch7 = (void*)(address+offset);
                                               }
                                               address += handle->scratch7_size + 64;
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
                                             if (address % 64 == 0) {
                                               handle->scratch5 = (void*)address;
                                             }
                                             else {
                                               offset = (64 - address % 64);
                                               handle->scratch5 = (void*)(address + offset);
                                             }
                                             address += handle->max_scratch5_size + 64;
                                             if (handle->scratch7_size != 0) {
                                               if (address % 64 == 0) {
                                                 handle->scratch7 = (void*)address;
                                               } else {
                                                 offset = (64 - address % 64);
                                                 handle->scratch7 = (void*)(address+offset);
                                               }
                                               address += handle->scratch7_size + 64;
                                             }
                                           } break;
        case LIBXSMM_DNN_COMPUTE_KIND_UPD: {
                                             /* we need a minibatch copy for transpose of input, scratch3 */
                                             if (handle->use_lp_kernel == 1) {
                                               if (address % 64 == 0) {
                                                 handle->scratch2 = (void*)address;
                                               }
                                               else {
                                                 offset = (64 - address % 64);
                                                 handle->scratch2 = (void*)(address + offset);
                                               }
                                               address += handle->scratch2_size + 64;
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
                                               address += handle->scratch4_size + 64;
                                             }
                                             if (address % 64 == 0) {
                                               handle->scratch5 = (void*)address;
                                             }
                                             else {
                                               offset = (64 - address % 64);
                                               handle->scratch5 = (void*)(address + offset);
                                             }
                                             address += handle->max_scratch5_size + 64;
                                             if (handle->scratch6_size != 0) {
                                               if (address % 64 == 0) {
                                                 handle->scratch6 = (void*)address;
                                               }
                                               else {
                                                 offset = (64 - address % 64);
                                                 handle->scratch6 = (void*)(address + offset);
                                               }
                                               address += handle->scratch6_size + 64;
                                             }
                                             if (handle->scratch7_size != 0) {
                                               if (address % 64 == 0) {
                                                 handle->scratch7 = (void*)address;
                                               } else {
                                                 offset = (64 - address % 64);
                                                 handle->scratch7 = (void*)(address+offset);
                                               }
                                               address += handle->scratch7_size + 64;
                                             }
                                           } break;
        case LIBXSMM_DNN_COMPUTE_KIND_ALL: {
                                           /* we need filter for transpose, + 64 to do alignment while performing bind, scratch1 */
                                             if (address % 64 == 0) {
                                               handle->scratch1 = (void*)address;
                                             } else {
                                               offset = (64 - address % 64);
                                               handle->scratch1 = (void*)(address+offset);
                                             }
                                             address += handle->scratch1_size + 64;
                                             if (handle->use_lp_kernel == 1) {
                                               if (address % 64 == 0) {
                                                 handle->scratch2 = (void*)address;
                                               }
                                               else {
                                                 offset = (64 - address % 64);
                                                 handle->scratch2 = (void*)(address + offset);
                                               }
                                               address += handle->scratch2_size + 64;
                                             }
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
                                               address += handle->scratch4_size + 64;
                                             }
                                             if (address % 64 == 0) {
                                               handle->scratch5 = (void*)address;
                                             }
                                             else {
                                               offset = (64 - address % 64);
                                               handle->scratch5 = (void*)(address + offset);
                                             }
                                             address += handle->max_scratch5_size + 64;
                                             if (handle->scratch6_size != 0) {
                                               if (address % 64 == 0) {
                                                 handle->scratch6 = (void*)address;
                                               }
                                               else {
                                                 offset = (64 - address % 64);
                                                 handle->scratch6 = (void*)(address + offset);
                                               }
                                               address += handle->scratch6_size + 64;
                                             }
                                             if (handle->scratch7_size != 0) {
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
  } else {
    status = LIBXSMM_DNN_ERR_INVALID_HANDLE;
  }

  return status;
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_release_scratch(libxsmm_dnn_layer* handle, const libxsmm_dnn_compute_kind kind)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  if (0 != handle) {
      switch (kind) {
        case LIBXSMM_DNN_COMPUTE_KIND_FWD: {
                                             handle->scratch5 = 0;
                                             handle->scratch6 = 0;
                                             handle->scratch7 = 0;
        } break;
        case LIBXSMM_DNN_COMPUTE_KIND_BWD: {
                                             handle->scratch1 = 0;
                                             handle->scratch5 = 0;
                                             handle->scratch7 = 0;
        } break;
        case LIBXSMM_DNN_COMPUTE_KIND_UPD: {
                                             handle->scratch2 = 0;
                                             handle->scratch3 = 0;
                                             handle->scratch4 = 0;
                                             handle->scratch5 = 0;
                                             handle->scratch6 = 0;
                                             handle->scratch7 = 0;
        } break;
        case LIBXSMM_DNN_COMPUTE_KIND_ALL: {
                                             handle->scratch1 = 0;
                                             handle->scratch2 = 0;
                                             handle->scratch3 = 0;
                                             handle->scratch4 = 0;
                                             handle->scratch5 = 0;
                                             handle->scratch6 = 0;
                                             handle->scratch7 = 0;
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


LIBXSMM_API_INLINE libxsmm_dnn_err_t internal_execute_st(libxsmm_dnn_layer* handle,
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


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_execute_st(libxsmm_dnn_layer* handle,
    libxsmm_dnn_compute_kind kind, /*unsigned*/int start_thread, /*unsigned*/int tid)
{
  return internal_execute_st(handle, kind, start_thread, tid);
}


LIBXSMM_API void libxsmm_dnn_execute(libxsmm_dnn_layer* handle, libxsmm_dnn_compute_kind kind)
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


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_transpose_filter(libxsmm_dnn_layer* handle, const libxsmm_dnn_tensor_type type) {
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
  int ofm1, ifm1, kj, ki, ifm2, ofm2;

  /* check for filter type */
  if ( (type != LIBXSMM_DNN_REGULAR_FILTER) ) {
    status = LIBXSMM_DNN_ERR_UNKNOWN_TENSOR_TYPE;
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
    status = LIBXSMM_DNN_ERR_MISMATCH_TENSOR;
    return status;
  }

  /* check that we are in FP32 */
  if ( handle->datatype_in == LIBXSMM_DNN_DATATYPE_F32 ) {
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


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_reduce_wu_filters(libxsmm_dnn_layer* handle, const libxsmm_dnn_tensor_type type) {
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
  int i, j, filter_size;

  /* check for filter type */
  if ( (type != LIBXSMM_DNN_GRADIENT_FILTER) ) {
    status = LIBXSMM_DNN_ERR_UNKNOWN_TENSOR_TYPE;
    return status;
  }

  /* check if we have input, output and filter */
  if (handle->grad_filter == 0) {
    status = LIBXSMM_DNN_ERR_DATA_NOT_BOUND;
    return status;
  }

  /* check that we are in FP32 */
  if (handle->datatype_out == LIBXSMM_DNN_DATATYPE_F32 ) {
    if (handle->upd_use_external_reduce != 0) {
      float* filter_ptr = (float*)handle->grad_filter->data;
      /* calculate filter size */
      filter_size = handle->blocksofm * handle->blocksifm * handle->desc.R * handle->desc.S * handle->ofmblock * handle->ifmblock;

      for ( i = 0; i < handle->desc.threads; ++i ) {
        float* tmp_filter_ptr = ((float*)handle->scratch4) + ((size_t)i * filter_size);
        for ( j = 0; j < filter_size; ++j ) {
          filter_ptr[j] += tmp_filter_ptr[j];
        }
      }
    }
  } else {
    status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
  }

  return status;
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_get_codegen_success(libxsmm_dnn_layer* handle, libxsmm_dnn_compute_kind kind) {
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  if (0 != handle) {
    switch (kind) {
      case LIBXSMM_DNN_COMPUTE_KIND_FWD: {
                                           if (handle->code_fwd[0].pmm == 0) {
                                             status = LIBXSMM_DNN_WARN_FALLBACK;
                                           }
                                         } break;
      case LIBXSMM_DNN_COMPUTE_KIND_BWD: {
                                           if (handle->code_bwd[0].pmm == 0) {
                                             status = LIBXSMM_DNN_WARN_FALLBACK;
                                           }
                                         } break;
      case LIBXSMM_DNN_COMPUTE_KIND_UPD: {
                                           if (handle->code_upd[0].pmm == 0) {
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


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_get_parallel_tasks(libxsmm_dnn_layer* handle, libxsmm_dnn_compute_kind kind, unsigned int* num_tasks) {
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

