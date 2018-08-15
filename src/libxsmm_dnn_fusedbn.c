/******************************************************************************
** Copyright (c) 2017-2018, Intel Corporation                                **
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
/* Alexander Heinecke, Sasikanth Avancha (Intel Corp.)
******************************************************************************/

#include "libxsmm_main.h"
#include "libxsmm_dnn_setup.h"

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#include <string.h>
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif


LIBXSMM_API libxsmm_dnn_fusedbn* libxsmm_dnn_create_fusedbn(libxsmm_dnn_fusedbn_desc fusedbn_desc, libxsmm_dnn_err_t* status) {
  libxsmm_dnn_fusedbn* handle = 0;
  int noarch;

  if ( ((handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_BF16) && (handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_BF16)) ||
       ((handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_F32) && (handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_F32))    ) {
    handle = (libxsmm_dnn_fusedbn*)malloc(sizeof(libxsmm_dnn_fusedbn));

    if (0 != handle) {
      *status = LIBXSMM_DNN_SUCCESS;
      /* zero entire content; not only safer but also sets data and code pointers to NULL */
      memset(handle, 0, sizeof(*handle));
      /* let's make the desciption presitent */
      handle->desc = fusedbn_desc;
      /* we need to compute the memory layout given the */
      *status = libxsmm_dnn_get_feature_map_blocks( handle->desc.C, handle->desc.C,
                                                    &(handle->ifmblock), &(handle->ifmblock_hp),
                                                    &(handle->ofmblock), &(handle->ofmblock_lp),
                                                    &(handle->fm_lp_block), handle->desc.datatype_in, handle->desc.datatype_out, &noarch );
      /* compute the outer blocks */
      if ( (handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_BF16) && (handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_BF16) ) {
        handle->blocksifm = handle->desc.C / handle->ifmblock_hp;
        handle->blocksofm = handle->desc.C / handle->ofmblock;
        handle->blocksifm_lp = handle->desc.C / handle->ifmblock_hp;
        handle->blocksofm_lp = handle->desc.C / handle->ofmblock;
      } else {
        /* this is FP32 */
        handle->blocksifm = handle->desc.C / handle->ifmblock;
        handle->blocksofm = handle->desc.C / handle->ofmblock;
        handle->blocksifm_lp = handle->blocksifm;
        handle->blocksofm_lp = handle->blocksofm;
      }
      /* create barrier */
      handle->barrier = libxsmm_barrier_create(handle->desc.threads, 1);
    } else {
      *status = LIBXSMM_DNN_ERR_CREATE_HANDLE;
    }
  } else {
    *status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
  }

  return handle;
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_destroy_fusedbn(const libxsmm_dnn_fusedbn* handle){
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  if (0 != handle) {
    /* Deallocate barrier */
    if (handle->barrier != 0 ) { libxsmm_barrier_release((const libxsmm_barrier*)handle->barrier); }
    /* deallocate handle structure */
    free(/*remove constness*/(libxsmm_dnn_fusedbn*)handle);
  } else {
    status = LIBXSMM_DNN_ERR_INVALID_HANDLE;
  }

  return status;
}


LIBXSMM_API libxsmm_dnn_tensor_datalayout* libxsmm_dnn_fusedbn_create_tensor_datalayout(const libxsmm_dnn_fusedbn* handle, const libxsmm_dnn_tensor_type type, libxsmm_dnn_err_t* status) {
  libxsmm_dnn_tensor_datalayout* layout;

  *status = LIBXSMM_DNN_SUCCESS;
  layout = 0;

  if (handle != 0) {
    layout = (libxsmm_dnn_tensor_datalayout*) malloc(sizeof(libxsmm_dnn_tensor_datalayout));

    if (layout != 0) {
      memset(layout, 0, sizeof(libxsmm_dnn_tensor_datalayout));
      layout->custom_format = LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM_1;

      if ( (type == LIBXSMM_DNN_REGULAR_INPUT)  || (type == LIBXSMM_DNN_GRADIENT_INPUT)  || (type == LIBXSMM_DNN_INPUT)  ||
           (type == LIBXSMM_DNN_REGULAR_OUTPUT) || (type == LIBXSMM_DNN_GRADIENT_OUTPUT) || (type == LIBXSMM_DNN_OUTPUT)    ) {
        if ((handle->desc.buffer_format & LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM) > 0) {
          if ( ((handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_F32) && (handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_F32) ) ) {
            layout->datatype = LIBXSMM_DNN_DATATYPE_F32;
            layout->dim_type = (libxsmm_dnn_tensor_dimtype*) malloc(5*sizeof(libxsmm_dnn_tensor_dimtype));
            layout->dim_size = (unsigned int*) malloc(5*sizeof(unsigned int));

            if (0 != layout->dim_type && 0 != layout->dim_size) {
              layout->num_dims = 5;
              layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
              layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_W;
              layout->dim_type[2] = LIBXSMM_DNN_TENSOR_DIMTYPE_H;
              layout->dim_type[3] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
              layout->dim_type[4] = LIBXSMM_DNN_TENSOR_DIMTYPE_N;
              if ( (type == LIBXSMM_DNN_REGULAR_INPUT) || (type == LIBXSMM_DNN_GRADIENT_INPUT) || (type == LIBXSMM_DNN_INPUT) ) {
                layout->dim_size[0] = handle->ifmblock;
                layout->dim_size[1] = handle->desc.W;
                layout->dim_size[2] = handle->desc.H;
                layout->dim_size[3] = handle->blocksifm;
                layout->dim_size[4] = handle->desc.N;
              } else if ( (type == LIBXSMM_DNN_REGULAR_OUTPUT) || (type == LIBXSMM_DNN_GRADIENT_OUTPUT) || (type == LIBXSMM_DNN_OUTPUT) ) {
                layout->dim_size[0] = handle->ofmblock;
                layout->dim_size[1] = handle->desc.W;
                layout->dim_size[2] = handle->desc.H;
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
              *status = LIBXSMM_DNN_ERR_CREATE_LAYOUT_ARRAYS;
            }
          } else if ( (handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_BF16) && (handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_BF16) ) {
            layout->datatype = LIBXSMM_DNN_DATATYPE_BF16;
            layout->dim_type = (libxsmm_dnn_tensor_dimtype*) malloc(6*sizeof(libxsmm_dnn_tensor_dimtype));
            layout->dim_size = (unsigned int*) malloc(6*sizeof(unsigned int));
            if (0 != layout->dim_type && 0 != layout->dim_size) {
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
                layout->dim_size[2] = handle->desc.W;
                layout->dim_size[3] = handle->desc.H;
                layout->dim_size[4] = handle->blocksifm;
                layout->dim_size[5] = handle->desc.N;
              } else if ( (type == LIBXSMM_DNN_REGULAR_OUTPUT) || (type == LIBXSMM_DNN_GRADIENT_OUTPUT) || (type == LIBXSMM_DNN_OUTPUT) ) {
                layout->dim_size[0] = handle->fm_lp_block;
                layout->dim_size[1] = handle->ofmblock_lp;
                layout->dim_size[2] = handle->desc.W;
                layout->dim_size[3] = handle->desc.H;
                layout->dim_size[4] = handle->blocksofm;
                layout->dim_size[5] = handle->desc.N;
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
              *status = LIBXSMM_DNN_ERR_CREATE_LAYOUT_ARRAYS;
            }
          } else {
            free(layout);
            layout = 0; /* make sure a NULL is returned */
            *status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
          }
        } else if ((handle->desc.buffer_format & LIBXSMM_DNN_TENSOR_FORMAT_NHWC) > 0) {
          if ( ((handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_F32) && (handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_F32)) ||
               ((handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_BF16) && (handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_BF16))    ) {
            layout->datatype = handle->desc.datatype_in;
            layout->dim_type = (libxsmm_dnn_tensor_dimtype*) malloc(4*sizeof(libxsmm_dnn_tensor_dimtype));
            layout->dim_size = (unsigned int*) malloc(4*sizeof(unsigned int));
            if (0 != layout->dim_type && 0 != layout->dim_size) { /* TODO: handle the error */
              layout->num_dims = 4;
              layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
              layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_W;
              layout->dim_type[2] = LIBXSMM_DNN_TENSOR_DIMTYPE_H;
              layout->dim_type[3] = LIBXSMM_DNN_TENSOR_DIMTYPE_N;
              if ( (type == LIBXSMM_DNN_REGULAR_INPUT) || (type == LIBXSMM_DNN_GRADIENT_INPUT) || (type == LIBXSMM_DNN_INPUT)   || 
                   (type == LIBXSMM_DNN_REGULAR_OUTPUT) || (type == LIBXSMM_DNN_GRADIENT_OUTPUT) || (type == LIBXSMM_DNN_OUTPUT)   )   {
                layout->dim_size[0] = handle->desc.C;
                layout->dim_size[1] = handle->desc.W;
                layout->dim_size[2] = handle->desc.H;
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
      } else if ( (type == LIBXSMM_DNN_REGULAR_CHANNEL_SCALAR) || (type == LIBXSMM_DNN_GRADIENT_CHANNEL_SCALAR) || (type == LIBXSMM_DNN_CHANNEL_SCALAR) ) {
        layout->format = handle->desc.buffer_format;
        layout->tensor_type = LIBXSMM_DNN_CHANNEL_SCALAR;

        if ((handle->desc.buffer_format & LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM) > 0) {
          if ( handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_F32 ) {
            layout->datatype = handle->desc.datatype_in;
            layout->dim_type = (libxsmm_dnn_tensor_dimtype*) malloc(2*sizeof(libxsmm_dnn_tensor_dimtype));
            layout->dim_size = (unsigned int*) malloc(2*sizeof(unsigned int));

            if (0 != layout->dim_type && 0 != layout->dim_size) {
              layout->num_dims = 2;
              layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
              layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
              layout->dim_size[0] = handle->ifmblock*handle->fm_lp_block;
              layout->dim_size[1] = handle->blocksifm;
            } else {
              free(layout);
              layout = 0; /* make sure a NULL is returned */
              *status = LIBXSMM_DNN_ERR_CREATE_LAYOUT_ARRAYS;
            }
          } else {
            free(layout);
            layout = 0; /* make sure a NULL is returned */
            *status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
          }
        } else if ((handle->desc.buffer_format & LIBXSMM_DNN_TENSOR_FORMAT_NHWC) > 0) {
          if ( handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_F32 ) {
            layout->datatype = handle->desc.datatype_out;
            layout->dim_type = (libxsmm_dnn_tensor_dimtype*) malloc(1*sizeof(libxsmm_dnn_tensor_dimtype));
            layout->dim_size = (unsigned int*) malloc(1*sizeof(unsigned int));

            if (0 != layout->dim_type && 0 != layout->dim_size) {
              layout->num_dims = 1;
              layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
              layout->dim_size[0] = handle->desc.C;
            } else {
              free(layout);
              layout = 0; /* make sure a NULL is returned */
              *status = LIBXSMM_DNN_ERR_CREATE_LAYOUT_ARRAYS;
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

  return 0;
}

LIBXSMM_API size_t libxsmm_dnn_fusedbn_get_scratch_size(const libxsmm_dnn_fusedbn* handle, const libxsmm_dnn_compute_kind kind, libxsmm_dnn_err_t* status) {
  LIBXSMM_UNUSED(handle);
  LIBXSMM_UNUSED(kind);
  LIBXSMM_UNUSED(status);
  return 0;
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_fusedbn_bind_scratch(libxsmm_dnn_fusedbn* handle, const libxsmm_dnn_compute_kind kind, const void* scratch) {
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
  LIBXSMM_UNUSED(handle);
  LIBXSMM_UNUSED(kind);
  LIBXSMM_UNUSED(scratch);
  return status;
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_fusedbn_release_scratch(libxsmm_dnn_fusedbn* handle, const libxsmm_dnn_compute_kind kind) {
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
  LIBXSMM_UNUSED(handle);
  LIBXSMM_UNUSED(kind);
  return status;
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_fusedbn_bind_tensor(libxsmm_dnn_fusedbn* handle, const libxsmm_dnn_tensor* tensor, const libxsmm_dnn_tensor_type type) {
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
  LIBXSMM_UNUSED(handle);
  LIBXSMM_UNUSED(tensor);
  LIBXSMM_UNUSED(type);
  return status;
}


LIBXSMM_API libxsmm_dnn_tensor* libxsmm_dnn_fusedbn_get_tensor(libxsmm_dnn_fusedbn* handle, const libxsmm_dnn_tensor_type type, libxsmm_dnn_err_t* status) {
  LIBXSMM_UNUSED(handle);
  LIBXSMM_UNUSED(type);
  LIBXSMM_UNUSED(status);
  return 0;
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_fusedbn_release_tensor(libxsmm_dnn_fusedbn* handle, const libxsmm_dnn_tensor_type type) {
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
  LIBXSMM_UNUSED(handle);
  LIBXSMM_UNUSED(type);
  return status;
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_fusedbn_execute_st(libxsmm_dnn_fusedbn* handle, libxsmm_dnn_compute_kind kind,
  /*unsigned*/int start_thread, /*unsigned*/int tid) {
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
  LIBXSMM_UNUSED(handle);
  LIBXSMM_UNUSED(kind);
  LIBXSMM_UNUSED(start_thread);
  LIBXSMM_UNUSED(tid);
  return status;
}

