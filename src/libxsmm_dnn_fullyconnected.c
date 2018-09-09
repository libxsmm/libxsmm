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
#include "libxsmm_dnn_fullyconnected_weight_update.h"
#include "libxsmm_dnn_fullyconnected_backward.h"
#include "libxsmm_dnn_fullyconnected_forward.h"
#include "libxsmm_dnn_setup.h"
#include "libxsmm_main.h"

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#include <string.h>
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif


LIBXSMM_API libxsmm_dnn_fullyconnected* libxsmm_dnn_create_fullyconnected(libxsmm_dnn_fullyconnected_desc fullyconnected_desc, libxsmm_dnn_err_t* status) {
  libxsmm_dnn_fullyconnected* handle = 0;
  int noarch;

  if ( ((fullyconnected_desc.datatype_in == LIBXSMM_DNN_DATATYPE_BF16) && (fullyconnected_desc.datatype_out == LIBXSMM_DNN_DATATYPE_BF16)) ||
       ((fullyconnected_desc.datatype_in == LIBXSMM_DNN_DATATYPE_F32) && (fullyconnected_desc.datatype_out == LIBXSMM_DNN_DATATYPE_F32))    ) {
    handle = (libxsmm_dnn_fullyconnected*)malloc(sizeof(libxsmm_dnn_fullyconnected));

    if (0 != handle) {
      *status = LIBXSMM_DNN_SUCCESS;
      /* zero entire content; not only safer but also sets data and code pointers to NULL */
      memset(handle, 0, sizeof(*handle));
      /* let's make the description persistent */
      handle->desc = fullyconnected_desc;
      /* we need to compute the memory layout given the */
      *status = libxsmm_dnn_get_feature_map_blocks( handle->desc.C, handle->desc.K,
                                                    &(handle->ifmblock), &(handle->ifmblock_hp),
                                                    &(handle->ofmblock), &(handle->ofmblock_lp),
                                                    &(handle->fm_lp_block), handle->desc.datatype_in, handle->desc.datatype_out, &noarch );
      /* compute the outer blocks */
      if ( (handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_BF16) && (handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_BF16) ) {
        handle->blocksifm = handle->desc.C / handle->ifmblock_hp;
        handle->blocksofm = handle->desc.K / handle->ofmblock;
        handle->blocksifm_lp = handle->desc.C / handle->ifmblock_hp;
        handle->blocksofm_lp = handle->desc.K / handle->ofmblock;
      } else {
        /* this is FP32 */
        handle->blocksifm = handle->desc.C / handle->ifmblock;
        handle->blocksofm = handle->desc.K / handle->ofmblock;
        handle->blocksifm_lp = handle->blocksifm;
        handle->blocksofm_lp = handle->blocksofm;
      }
      /* create barrier */
      handle->barrier = libxsmm_barrier_create(handle->desc.threads, 1);
      /* calculate scratch size for batchstats */
      handle->scratch_size = sizeof(float) * LIBXSMM_MAX( ((size_t)handle->desc.C + (size_t)handle->desc.K) * (size_t)handle->desc.N, 
                                                           (size_t)handle->desc.C * (size_t)handle->desc.K                            ) ;
    } else {
      *status = LIBXSMM_DNN_ERR_CREATE_HANDLE;
    }
  } else {
    *status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
  }

  return handle;
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_destroy_fullyconnected(const libxsmm_dnn_fullyconnected* handle) {
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  if (0 != handle) {
    /* Deallocate barrier */
    if (handle->barrier != 0 ) { libxsmm_barrier_release((const libxsmm_barrier*)handle->barrier); }
    /* deallocate handle structure */
    free(/*remove constness*/(libxsmm_dnn_fullyconnected*)handle);
  } else {
    status = LIBXSMM_DNN_ERR_INVALID_HANDLE;
  }

  return status;
}


LIBXSMM_API libxsmm_dnn_tensor_datalayout* libxsmm_dnn_fullyconnected_create_tensor_datalayout(const libxsmm_dnn_fullyconnected* handle, const libxsmm_dnn_tensor_type type, libxsmm_dnn_err_t* status) {
  libxsmm_dnn_tensor_datalayout* layout;

  *status = LIBXSMM_DNN_SUCCESS;
  layout = 0;

  if (handle != 0) {
    layout = (libxsmm_dnn_tensor_datalayout*) malloc(sizeof(libxsmm_dnn_tensor_datalayout));

    if (layout != 0) {
      memset(layout, 0, sizeof(libxsmm_dnn_tensor_datalayout));
      layout->custom_format = LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM_1;

      if ( (type == LIBXSMM_DNN_REGULAR_INPUT)     || (type == LIBXSMM_DNN_GRADIENT_INPUT)  || (type == LIBXSMM_DNN_INPUT)  ||
           (type == LIBXSMM_DNN_REGULAR_OUTPUT)    || (type == LIBXSMM_DNN_GRADIENT_OUTPUT) || (type == LIBXSMM_DNN_OUTPUT)    ) {
        layout->format = handle->desc.buffer_format;
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
              if ( (type == LIBXSMM_DNN_REGULAR_INPUT)     || (type == LIBXSMM_DNN_GRADIENT_INPUT)     || (type == LIBXSMM_DNN_INPUT)  ) {
                layout->dim_size[0] = handle->ifmblock;
                layout->dim_size[1] = 1;
                layout->dim_size[2] = 1;
                layout->dim_size[3] = handle->blocksifm;
                layout->dim_size[4] = handle->desc.N;
              } else if ( (type == LIBXSMM_DNN_REGULAR_OUTPUT) || (type == LIBXSMM_DNN_GRADIENT_OUTPUT) || (type == LIBXSMM_DNN_OUTPUT) ) {
                layout->dim_size[0] = handle->ofmblock;
                layout->dim_size[1] = 1;
                layout->dim_size[2] = 1;
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
              if ( (type == LIBXSMM_DNN_REGULAR_INPUT)     || (type == LIBXSMM_DNN_GRADIENT_INPUT)     || (type == LIBXSMM_DNN_INPUT)  ) {
                layout->dim_size[0] = handle->fm_lp_block;
                layout->dim_size[1] = handle->ifmblock;
                layout->dim_size[2] = 1;
                layout->dim_size[3] = 1;
                layout->dim_size[4] = handle->blocksifm;
                layout->dim_size[5] = handle->desc.N;
              } else if ( (type == LIBXSMM_DNN_REGULAR_OUTPUT) || (type == LIBXSMM_DNN_GRADIENT_OUTPUT) || (type == LIBXSMM_DNN_OUTPUT) ) {
                layout->dim_size[0] = handle->fm_lp_block;
                layout->dim_size[1] = handle->ofmblock_lp;
                layout->dim_size[2] = 1;
                layout->dim_size[3] = 1;
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
              if ( (type == LIBXSMM_DNN_REGULAR_INPUT)     || (type == LIBXSMM_DNN_GRADIENT_INPUT)     || (type == LIBXSMM_DNN_INPUT)  )   {
                layout->dim_size[0] = handle->desc.C;
                layout->dim_size[1] = 1;
                layout->dim_size[2] = 1;
                layout->dim_size[3] = handle->desc.N;
              } else if ( (type == LIBXSMM_DNN_REGULAR_OUTPUT) || (type == LIBXSMM_DNN_GRADIENT_OUTPUT) || (type == LIBXSMM_DNN_OUTPUT) )   {
                layout->dim_size[0] = handle->desc.C;
                layout->dim_size[1] = 1;
                layout->dim_size[2] = 1;
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
      } else if ( (type == LIBXSMM_DNN_REGULAR_FILTER)  || (type == LIBXSMM_DNN_GRADIENT_FILTER)  || (type == LIBXSMM_DNN_FILTER)  ) {
        layout->format = handle->desc.filter_format;
        layout->tensor_type = LIBXSMM_DNN_FILTER;

        if ((handle->desc.filter_format & LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM) > 0) {
          if ( (handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_F32) && (handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_F32) ) {
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
              layout->dim_size[2] = 1;
              layout->dim_size[3] = 1;
              layout->dim_size[4] = handle->blocksifm;
              layout->dim_size[5] = handle->blocksofm;
            } else {
              free(layout);
              layout = 0; /* make sure a NULL is returned */
              *status = LIBXSMM_DNN_ERR_CREATE_LAYOUT_ARRAYS;
            }
          } else if ( (handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_BF16) && (handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_BF16) ) {
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
              layout->dim_size[2] = handle->ifmblock;
              layout->dim_size[3] = 1;
              layout->dim_size[4] = 1;
              layout->dim_size[5] = handle->blocksifm;
              layout->dim_size[6] = handle->blocksofm;
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
        } else if ((handle->desc.filter_format & LIBXSMM_DNN_TENSOR_FORMAT_RSCK) > 0) {
          if ( (handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_F32) && (handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_F32) ) {
            layout->datatype = handle->desc.datatype_in;
            layout->dim_type = (libxsmm_dnn_tensor_dimtype*) malloc(4*sizeof(libxsmm_dnn_tensor_dimtype));
            layout->dim_size = (unsigned int*) malloc(4*sizeof(unsigned int));
            if (0 != layout->dim_type && 0 != layout->dim_size) {
              layout->num_dims = 4;
              layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
              layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
              layout->dim_type[2] = LIBXSMM_DNN_TENSOR_DIMTYPE_S;
              layout->dim_type[3] = LIBXSMM_DNN_TENSOR_DIMTYPE_R;
              layout->dim_size[0] = handle->ofmblock * handle->blocksofm;
              layout->dim_size[1] = handle->ifmblock * handle->blocksifm;
              layout->dim_size[2] = 1;
              layout->dim_size[3] = 1;
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
}

LIBXSMM_API size_t libxsmm_dnn_fullyconnected_get_scratch_size(const libxsmm_dnn_fullyconnected* handle, libxsmm_dnn_err_t* status) {
  size_t l_scratch_size = 0;
  *status = LIBXSMM_DNN_SUCCESS;

  if (0 != handle) {
    l_scratch_size = handle->scratch_size + 64; /* 64 byte extra in case the user code does not care about alignment */
  } else {
    *status = LIBXSMM_DNN_ERR_INVALID_HANDLE;
  }

  return l_scratch_size;
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_fullyconnected_bind_scratch(libxsmm_dnn_fullyconnected* handle, const void* scratch) {
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
  uintptr_t address = (uintptr_t)scratch;
  size_t offset = 0;

  if (scratch == 0) {
    status = LIBXSMM_DNN_ERR_SCRATCH_NOT_ALLOCED;
    return status;
  }

  if (0 != handle) {
    /* align the internal scratch buffer if needed */
    if (address % 64 == 0) {
      handle->scratch = (void*)address;
    } else {
      offset = (64 - address % 64);
      handle->scratch = (void*)(address+offset);
    }
  } else {
    status = LIBXSMM_DNN_ERR_INVALID_HANDLE;
  }

  return status;
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_fullyconnected_release_scratch(libxsmm_dnn_fullyconnected* handle) {
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  if (0 != handle) {
    handle->scratch = 0;
  } else {
    status = LIBXSMM_DNN_ERR_INVALID_HANDLE;
  }

  return status;
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_fullyconnected_bind_tensor(libxsmm_dnn_fullyconnected* handle, const libxsmm_dnn_tensor* tensor, const libxsmm_dnn_tensor_type type) {
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  /* check for tensor type */
  if ( (type != LIBXSMM_DNN_REGULAR_INPUT)  && (type != LIBXSMM_DNN_GRADIENT_INPUT)  &&
       (type != LIBXSMM_DNN_REGULAR_OUTPUT) && (type != LIBXSMM_DNN_GRADIENT_OUTPUT) &&
       (type != LIBXSMM_DNN_REGULAR_FILTER) && (type != LIBXSMM_DNN_GRADIENT_FILTER)    ) {
    status = LIBXSMM_DNN_ERR_UNKNOWN_TENSOR_TYPE;
    return status;
  }

  if (handle != 0 && tensor != 0) {
    libxsmm_dnn_tensor_datalayout* handle_layout = libxsmm_dnn_fullyconnected_create_tensor_datalayout(handle, type, &status);

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


LIBXSMM_API libxsmm_dnn_tensor* libxsmm_dnn_fullyconnected_get_tensor(libxsmm_dnn_fullyconnected* handle, const libxsmm_dnn_tensor_type type, libxsmm_dnn_err_t* status) {
  libxsmm_dnn_tensor* return_tensor = 0;

  *status = LIBXSMM_DNN_SUCCESS;

  /* check for tensor type */
  if ( (type != LIBXSMM_DNN_REGULAR_INPUT)  && (type != LIBXSMM_DNN_GRADIENT_INPUT)  &&
       (type != LIBXSMM_DNN_REGULAR_OUTPUT) && (type != LIBXSMM_DNN_GRADIENT_OUTPUT) &&
       (type != LIBXSMM_DNN_REGULAR_FILTER) && (type != LIBXSMM_DNN_GRADIENT_FILTER)    ) {
    *status = LIBXSMM_DNN_ERR_UNKNOWN_TENSOR_TYPE;
    return return_tensor;
  }

  if (handle != 0) {
    if ( type == LIBXSMM_DNN_REGULAR_INPUT ) {
      return_tensor = handle->reg_input;
    } else if ( type == LIBXSMM_DNN_GRADIENT_INPUT ) {
      return_tensor = handle->grad_input;
    } else if ( type == LIBXSMM_DNN_REGULAR_OUTPUT ) {
      return_tensor = handle->reg_output;
    } else if ( type == LIBXSMM_DNN_GRADIENT_OUTPUT ) {
      return_tensor = handle->grad_output;
    } else if ( type == LIBXSMM_DNN_REGULAR_FILTER ) {
      return_tensor = handle->reg_filter;
    } else if ( type == LIBXSMM_DNN_GRADIENT_FILTER ) {
      return_tensor = handle->grad_filter;
    } else {
      /* cannot happen */
    }
  } else {
    *status = LIBXSMM_DNN_ERR_INVALID_HANDLE;
  }

  return return_tensor;
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_fullyconnected_release_tensor(libxsmm_dnn_fullyconnected* handle, const libxsmm_dnn_tensor_type type) {
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  /* check for tensor type */
  if ( (type != LIBXSMM_DNN_REGULAR_INPUT)  && (type != LIBXSMM_DNN_GRADIENT_INPUT)  &&
       (type != LIBXSMM_DNN_REGULAR_OUTPUT) && (type != LIBXSMM_DNN_GRADIENT_OUTPUT) &&
       (type != LIBXSMM_DNN_REGULAR_FILTER) && (type != LIBXSMM_DNN_GRADIENT_FILTER)    ) {
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
    } else {
      /* cannot happen */
    }
  } else {
    status = LIBXSMM_DNN_ERR_INVALID_HANDLE;
  }

  return status;
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_fullyconnected_execute_st(libxsmm_dnn_fullyconnected* handle, libxsmm_dnn_compute_kind kind,
  /*unsigned*/int start_thread, /*unsigned*/int tid) {
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
  LIBXSMM_UNUSED( start_thread );
  LIBXSMM_UNUSED( tid );

  if (0 != handle) {
    switch (kind) {
      case LIBXSMM_DNN_COMPUTE_KIND_FWD: {
        if ( (handle->desc.buffer_format == LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM) && (handle->desc.filter_format == LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM) ) {
          status = libxsmm_dnn_fullyconnected_st_fwd_custom( handle, start_thread, tid );
        } else {
          status = LIBXSMM_DNN_ERR_INVALID_FORMAT_FC;
        }
      } break;
      case LIBXSMM_DNN_COMPUTE_KIND_BWD: {
        if ( (handle->desc.buffer_format == LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM) && (handle->desc.filter_format == LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM) ) {
          status = libxsmm_dnn_fullyconnected_st_bwd_custom( handle, start_thread, tid );
        } else {
          status = LIBXSMM_DNN_ERR_INVALID_FORMAT_FC;
        }
      } break;
      case LIBXSMM_DNN_COMPUTE_KIND_UPD: {
        if ( (handle->desc.buffer_format == LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM) && (handle->desc.filter_format == LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM) ) {
          status = libxsmm_dnn_fullyconnected_st_upd_custom( handle, start_thread, tid );
        } else {
          status = LIBXSMM_DNN_ERR_INVALID_FORMAT_FC;
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

