/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Heinecke, Sasikanth Avancha (Intel Corp.)
******************************************************************************/
#include "libxsmm_dnn_fusedbatchnorm_backward.h"
#include "libxsmm_dnn_fusedbatchnorm_forward.h"
#include "libxsmm_main.h"


LIBXSMM_API libxsmm_dnn_fusedbatchnorm* libxsmm_dnn_create_fusedbatchnorm(libxsmm_dnn_fusedbatchnorm_desc fusedbatchnorm_desc, libxsmm_dnn_err_t* status) {
  libxsmm_dnn_fusedbatchnorm* handle = 0;
  int lpb;

  /* init libxsmm */
  LIBXSMM_INIT

  if ( fusedbatchnorm_desc.partN > fusedbatchnorm_desc.fullN ) {
    *status = LIBXSMM_DNN_ERR_CREATE_HANDLE;
    return handle;
  } else if ( (fusedbatchnorm_desc.partN != fusedbatchnorm_desc.fullN) && ((fusedbatchnorm_desc.fuse_ops & LIBXSMM_DNN_FUSEDBN_OPS_BNSTATS_NORED) == 0 ) && ((fusedbatchnorm_desc.fuse_ops & LIBXSMM_DNN_FUSEDBN_OPS_BNSCALE) == 0 ) ) {
    *status = LIBXSMM_DNN_ERR_CREATE_HANDLE;
    return handle;
  } else {
  }

  if ( ((fusedbatchnorm_desc.datatype_in == LIBXSMM_DNN_DATATYPE_BF16) && (fusedbatchnorm_desc.datatype_out == LIBXSMM_DNN_DATATYPE_BF16)) ||
       ((fusedbatchnorm_desc.datatype_in == LIBXSMM_DNN_DATATYPE_F32) && (fusedbatchnorm_desc.datatype_out == LIBXSMM_DNN_DATATYPE_F32))    ) {
    /* zero entire content; not only safer but also sets data and code pointers to NULL */
    handle = (libxsmm_dnn_fusedbatchnorm*)calloc(1, sizeof(libxsmm_dnn_fusedbatchnorm));

    if (0 != handle) {
      *status = LIBXSMM_DNN_SUCCESS;
      /* let's make the description persistent */
      handle->desc = fusedbatchnorm_desc;
      /* we need to compute the memory layout given the */
      *status = libxsmm_dnn_get_feature_map_blocks( handle->desc.C, handle->desc.C,
                                                    &(handle->ifmblock), &(handle->ofmblock), &lpb,
                                                    handle->desc.datatype_in, handle->desc.datatype_out );
      /* compute the outer blocks */
      handle->blocksifm = handle->desc.C / handle->ifmblock;
      handle->blocksofm = handle->desc.C / handle->ofmblock;
      /* create barrier */
      handle->barrier = libxsmm_barrier_create(handle->desc.threads, 1);
      /* calculate scratch size for batchstats */
      handle->scratch_size = (sizeof(float) * 2 * handle->desc.C * handle->desc.partN);
    } else {
      *status = LIBXSMM_DNN_ERR_CREATE_HANDLE;
    }
  } else {
    *status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
  }

  return handle;
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_destroy_fusedbatchnorm(const libxsmm_dnn_fusedbatchnorm* handle) {
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  if (0 != handle) {
    /* Deallocate barrier */
    if (handle->barrier != 0 ) { libxsmm_barrier_release((const libxsmm_barrier*)handle->barrier); }
    /* deallocate handle structure */
    free(/*remove constness*/(libxsmm_dnn_fusedbatchnorm*)handle);
  } else {
    status = LIBXSMM_DNN_ERR_INVALID_HANDLE;
  }

  return status;
}


LIBXSMM_API libxsmm_dnn_tensor_datalayout* libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout(const libxsmm_dnn_fusedbatchnorm* handle, const libxsmm_dnn_tensor_type type, libxsmm_dnn_err_t* status) {
  libxsmm_dnn_tensor_datalayout* layout;

  *status = LIBXSMM_DNN_SUCCESS;
  layout = 0;

  if (handle != 0) {
    /* zero entire content; not only safer but also sets data and code pointers to NULL */
    layout = (libxsmm_dnn_tensor_datalayout*)calloc(1, sizeof(libxsmm_dnn_tensor_datalayout));

    if (layout != 0) {
      layout->format = handle->desc.buffer_format;

      if ( (type == LIBXSMM_DNN_REGULAR_INPUT)     || (type == LIBXSMM_DNN_GRADIENT_INPUT)  || (type == LIBXSMM_DNN_INPUT)  ||
           (type == LIBXSMM_DNN_REGULAR_OUTPUT)    || (type == LIBXSMM_DNN_GRADIENT_OUTPUT) || (type == LIBXSMM_DNN_OUTPUT) ||
           (type == LIBXSMM_DNN_REGULAR_INPUT_ADD) || (type == LIBXSMM_DNN_GRADIENT_INPUT_ADD)                                  ) {
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
              if ( (type == LIBXSMM_DNN_REGULAR_INPUT)     || (type == LIBXSMM_DNN_GRADIENT_INPUT)     || (type == LIBXSMM_DNN_INPUT) ||
                   (type == LIBXSMM_DNN_REGULAR_INPUT_ADD) || (type == LIBXSMM_DNN_GRADIENT_INPUT_ADD)                                   ) {
                layout->dim_size[0] = handle->ifmblock;
                layout->dim_size[1] = handle->desc.W + (2*handle->desc.pad_w_in);
                layout->dim_size[2] = handle->desc.H + (2*handle->desc.pad_h_in);
                layout->dim_size[3] = handle->blocksifm;
                layout->dim_size[4] = handle->desc.partN;
              } else if ( (type == LIBXSMM_DNN_REGULAR_OUTPUT) || (type == LIBXSMM_DNN_GRADIENT_OUTPUT) || (type == LIBXSMM_DNN_OUTPUT) ) {
                layout->dim_size[0] = handle->ofmblock;
                layout->dim_size[1] = (handle->desc.W/handle->desc.v) + (2*handle->desc.pad_w_out);
                layout->dim_size[2] = (handle->desc.H/handle->desc.u) + (2*handle->desc.pad_h_out);
                layout->dim_size[3] = handle->blocksofm;
                layout->dim_size[4] = handle->desc.partN;
              } else { /* coverity[dead_error_begin] */
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
            layout->dim_type = (libxsmm_dnn_tensor_dimtype*) malloc(5*sizeof(libxsmm_dnn_tensor_dimtype));
            layout->dim_size = (unsigned int*) malloc(5*sizeof(unsigned int));
            if (0 != layout->dim_type && 0 != layout->dim_size) {
              layout->num_dims = 5;
              layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
              layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_W;
              layout->dim_type[2] = LIBXSMM_DNN_TENSOR_DIMTYPE_H;
              layout->dim_type[3] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
              layout->dim_type[4] = LIBXSMM_DNN_TENSOR_DIMTYPE_N;
              if ( (type == LIBXSMM_DNN_REGULAR_INPUT)     || (type == LIBXSMM_DNN_GRADIENT_INPUT)     || (type == LIBXSMM_DNN_INPUT) ||
                   (type == LIBXSMM_DNN_REGULAR_INPUT_ADD) || (type == LIBXSMM_DNN_GRADIENT_INPUT_ADD)                                   ) {
                layout->dim_size[0] = handle->ifmblock;
                layout->dim_size[1] = handle->desc.W + (2*handle->desc.pad_w_in);
                layout->dim_size[2] = handle->desc.H + (2*handle->desc.pad_h_in);
                layout->dim_size[3] = handle->blocksifm;
                layout->dim_size[4] = handle->desc.partN;
              } else if ( (type == LIBXSMM_DNN_REGULAR_OUTPUT) || (type == LIBXSMM_DNN_GRADIENT_OUTPUT) || (type == LIBXSMM_DNN_OUTPUT) ) {
                layout->dim_size[0] = handle->ofmblock;
                layout->dim_size[1] = (handle->desc.W/handle->desc.v) + (2*handle->desc.pad_w_out);
                layout->dim_size[2] = (handle->desc.H/handle->desc.u) + (2*handle->desc.pad_h_out);
                layout->dim_size[3] = handle->blocksofm;
                layout->dim_size[4] = handle->desc.partN;
              } else { /* coverity[dead_error_begin] */
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
              if ( (type == LIBXSMM_DNN_REGULAR_INPUT)     || (type == LIBXSMM_DNN_GRADIENT_INPUT)     || (type == LIBXSMM_DNN_INPUT) ||
                   (type == LIBXSMM_DNN_REGULAR_INPUT_ADD) || (type == LIBXSMM_DNN_GRADIENT_INPUT_ADD)                                      )   {
                layout->dim_size[0] = handle->desc.C;
                layout->dim_size[1] = handle->desc.W + (2*handle->desc.pad_w_in);
                layout->dim_size[2] = handle->desc.H + (2*handle->desc.pad_h_in);
                layout->dim_size[3] = handle->desc.partN;
              } else if ( (type == LIBXSMM_DNN_REGULAR_OUTPUT) || (type == LIBXSMM_DNN_GRADIENT_OUTPUT) || (type == LIBXSMM_DNN_OUTPUT) )   {
                layout->dim_size[0] = handle->desc.C;
                layout->dim_size[1] = (handle->desc.W/handle->desc.v) + (2*handle->desc.pad_w_out);
                layout->dim_size[2] = (handle->desc.H/handle->desc.u) + (2*handle->desc.pad_h_out);
                layout->dim_size[3] = handle->desc.partN;
              } else { /* coverity[dead_error_begin] */
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
      } else if ( (type == LIBXSMM_DNN_REGULAR_CHANNEL_BETA)  || (type == LIBXSMM_DNN_GRADIENT_CHANNEL_BETA)  || (type == LIBXSMM_DNN_CHANNEL_BETA)     ||
                  (type == LIBXSMM_DNN_REGULAR_CHANNEL_GAMMA) || (type == LIBXSMM_DNN_GRADIENT_CHANNEL_GAMMA) || (type == LIBXSMM_DNN_CHANNEL_GAMMA)    ||
                  (type == LIBXSMM_DNN_CHANNEL_EXPECTVAL)     || (type == LIBXSMM_DNN_CHANNEL_RCPSTDDEV)      || (type == LIBXSMM_DNN_CHANNEL_VARIANCE)    ) {
        layout->tensor_type = LIBXSMM_DNN_CHANNEL_SCALAR;

        if ((handle->desc.buffer_format & LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM) > 0) {
          if ( handle->desc.datatype_stats == LIBXSMM_DNN_DATATYPE_F32 ) {
            layout->datatype = handle->desc.datatype_stats;
            layout->dim_type = (libxsmm_dnn_tensor_dimtype*) malloc(2*sizeof(libxsmm_dnn_tensor_dimtype));
            layout->dim_size = (unsigned int*) malloc(2*sizeof(unsigned int));

            if (0 != layout->dim_type && 0 != layout->dim_size) {
              layout->num_dims = 2;
              layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
              layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
              layout->dim_size[0] = handle->ifmblock;
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
          if ( handle->desc.datatype_stats == LIBXSMM_DNN_DATATYPE_F32 ) {
            layout->datatype = handle->desc.datatype_stats;
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
      } else if ( (type == LIBXSMM_DNN_RELU_MASK) ) {
        layout->tensor_type = LIBXSMM_DNN_RELU_MASK;

        if ((handle->desc.buffer_format & LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM) > 0) {
          layout->datatype = LIBXSMM_DNN_DATATYPE_I8;
          layout->dim_type = (libxsmm_dnn_tensor_dimtype*) malloc(5*sizeof(libxsmm_dnn_tensor_dimtype));
          layout->dim_size = (unsigned int*) malloc(5*sizeof(unsigned int));

          if (0 != layout->dim_type && 0 != layout->dim_size) {
            layout->num_dims = 5;
            layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
            layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_W;
            layout->dim_type[2] = LIBXSMM_DNN_TENSOR_DIMTYPE_H;
            layout->dim_type[3] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
            layout->dim_type[4] = LIBXSMM_DNN_TENSOR_DIMTYPE_N;
            layout->dim_size[0] = handle->ofmblock;
            layout->dim_size[1] = (handle->desc.W/handle->desc.v) + (2*handle->desc.pad_w_out);
            layout->dim_size[2] = (handle->desc.H/handle->desc.u) + (2*handle->desc.pad_h_out);
            layout->dim_size[3] = handle->blocksofm;
            layout->dim_size[4] = handle->desc.partN;
          } else {
            free(layout);
            layout = 0; /* make sure a NULL is returned */
            *status = LIBXSMM_DNN_ERR_CREATE_LAYOUT_ARRAYS;
          }
        } else if ((handle->desc.buffer_format & LIBXSMM_DNN_TENSOR_FORMAT_NHWC) > 0) {
          layout->datatype = LIBXSMM_DNN_DATATYPE_I8;
          layout->dim_type = (libxsmm_dnn_tensor_dimtype*) malloc(4*sizeof(libxsmm_dnn_tensor_dimtype));
          layout->dim_size = (unsigned int*) malloc(4*sizeof(unsigned int));

          if (0 != layout->dim_type && 0 != layout->dim_size) {
            layout->num_dims = 6;
            layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
            layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_W;
            layout->dim_type[2] = LIBXSMM_DNN_TENSOR_DIMTYPE_H;
            layout->dim_type[3] = LIBXSMM_DNN_TENSOR_DIMTYPE_N;
            layout->dim_size[0] = handle->ofmblock*handle->blocksofm;
            layout->dim_size[1] = (handle->desc.W/handle->desc.v) + (2*handle->desc.pad_w_out);
            layout->dim_size[2] = (handle->desc.H/handle->desc.u) + (2*handle->desc.pad_h_out);
            layout->dim_size[3] = handle->desc.partN;
          } else {
            free(layout);
            layout = 0; /* make sure a NULL is returned */
            *status = LIBXSMM_DNN_ERR_CREATE_LAYOUT_ARRAYS;
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

LIBXSMM_API size_t libxsmm_dnn_fusedbatchnorm_get_scratch_size(const libxsmm_dnn_fusedbatchnorm* handle, libxsmm_dnn_err_t* status) {
  size_t l_scratch_size = 0;
  *status = LIBXSMM_DNN_SUCCESS;

  if (0 != handle) {
    l_scratch_size = handle->scratch_size + 64; /* 64 byte extra in case the user code does not care about alignment */
  } else {
    *status = LIBXSMM_DNN_ERR_INVALID_HANDLE;
  }

  return l_scratch_size;
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_fusedbatchnorm_bind_scratch(libxsmm_dnn_fusedbatchnorm* handle, const void* scratch) {
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


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_fusedbatchnorm_release_scratch(libxsmm_dnn_fusedbatchnorm* handle) {
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  if (0 != handle) {
    handle->scratch = 0;
  } else {
    status = LIBXSMM_DNN_ERR_INVALID_HANDLE;
  }

  return status;
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_fusedbatchnorm_bind_tensor(libxsmm_dnn_fusedbatchnorm* handle, const libxsmm_dnn_tensor* tensor, const libxsmm_dnn_tensor_type type) {
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  /* check for tensor type */
  if ( (type != LIBXSMM_DNN_REGULAR_INPUT)         && (type != LIBXSMM_DNN_GRADIENT_INPUT)         &&
       (type != LIBXSMM_DNN_REGULAR_OUTPUT)        && (type != LIBXSMM_DNN_GRADIENT_OUTPUT)        &&
       (type != LIBXSMM_DNN_REGULAR_INPUT_ADD)     && (type != LIBXSMM_DNN_GRADIENT_INPUT_ADD)     &&
       (type != LIBXSMM_DNN_REGULAR_CHANNEL_BETA)  && (type != LIBXSMM_DNN_GRADIENT_CHANNEL_BETA)  &&
       (type != LIBXSMM_DNN_REGULAR_CHANNEL_GAMMA) && (type != LIBXSMM_DNN_GRADIENT_CHANNEL_GAMMA) &&
       (type != LIBXSMM_DNN_CHANNEL_EXPECTVAL)     && (type != LIBXSMM_DNN_CHANNEL_RCPSTDDEV)      &&
       (type != LIBXSMM_DNN_CHANNEL_VARIANCE)      && (type != LIBXSMM_DNN_RELU_MASK)                  ) {
    status = LIBXSMM_DNN_ERR_UNKNOWN_TENSOR_TYPE;
    return status;
  }

  if (handle != 0 && tensor != 0) {
    libxsmm_dnn_tensor_datalayout* handle_layout = libxsmm_dnn_fusedbatchnorm_create_tensor_datalayout(handle, type, &status);

    if ( libxsmm_dnn_compare_tensor_datalayout(handle_layout, tensor->layout, &status) == 0 ) {
      if ( type == LIBXSMM_DNN_REGULAR_INPUT ) {
        handle->reg_input = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_GRADIENT_INPUT ) {
        handle->grad_input = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_REGULAR_OUTPUT ) {
        handle->reg_output = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_GRADIENT_OUTPUT ) {
        handle->grad_output = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_REGULAR_INPUT_ADD ) {
        handle->reg_add = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_GRADIENT_INPUT_ADD ) {
        handle->grad_add = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_REGULAR_CHANNEL_BETA ) {
        handle->reg_beta = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_GRADIENT_CHANNEL_BETA ) {
        handle->grad_beta = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_REGULAR_CHANNEL_GAMMA ) {
        handle->reg_gamma = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_GRADIENT_CHANNEL_GAMMA ) {
        handle->grad_gamma = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_CHANNEL_EXPECTVAL ) {
        handle->expvalue = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_CHANNEL_RCPSTDDEV ) {
        handle->rcpstddev = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_CHANNEL_VARIANCE ) {
        handle->variance = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_RELU_MASK ) {
        handle->relumask = (libxsmm_dnn_tensor*)tensor;
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


LIBXSMM_API libxsmm_dnn_tensor* libxsmm_dnn_fusedbatchnorm_get_tensor(libxsmm_dnn_fusedbatchnorm* handle, const libxsmm_dnn_tensor_type type, libxsmm_dnn_err_t* status) {
  libxsmm_dnn_tensor* return_tensor = 0;

  *status = LIBXSMM_DNN_SUCCESS;

  /* check for tensor type */
  if ( (type != LIBXSMM_DNN_REGULAR_INPUT)         && (type != LIBXSMM_DNN_GRADIENT_INPUT)         &&
       (type != LIBXSMM_DNN_REGULAR_OUTPUT)        && (type != LIBXSMM_DNN_GRADIENT_OUTPUT)        &&
       (type != LIBXSMM_DNN_REGULAR_INPUT_ADD)     && (type != LIBXSMM_DNN_GRADIENT_INPUT_ADD)     &&
       (type != LIBXSMM_DNN_REGULAR_CHANNEL_BETA)  && (type != LIBXSMM_DNN_GRADIENT_CHANNEL_BETA)  &&
       (type != LIBXSMM_DNN_REGULAR_CHANNEL_GAMMA) && (type != LIBXSMM_DNN_GRADIENT_CHANNEL_GAMMA) &&
       (type != LIBXSMM_DNN_CHANNEL_EXPECTVAL)     && (type != LIBXSMM_DNN_CHANNEL_RCPSTDDEV)      &&
       (type != LIBXSMM_DNN_CHANNEL_VARIANCE)      && (type != LIBXSMM_DNN_RELU_MASK)                 ) {
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
    } else if ( type == LIBXSMM_DNN_REGULAR_INPUT_ADD ) {
      return_tensor = handle->reg_add;
    } else if ( type == LIBXSMM_DNN_GRADIENT_INPUT_ADD ) {
      return_tensor = handle->grad_add;
    } else if ( type == LIBXSMM_DNN_REGULAR_CHANNEL_BETA ) {
      return_tensor = handle->reg_beta;
    } else if ( type == LIBXSMM_DNN_GRADIENT_CHANNEL_BETA ) {
      return_tensor = handle->grad_beta;
    } else if ( type == LIBXSMM_DNN_REGULAR_CHANNEL_GAMMA ) {
      return_tensor = handle->reg_gamma;
    } else if ( type == LIBXSMM_DNN_GRADIENT_CHANNEL_GAMMA ) {
      return_tensor = handle->grad_gamma;
    } else if ( type == LIBXSMM_DNN_CHANNEL_EXPECTVAL ) {
      return_tensor = handle->expvalue;
    } else if ( type == LIBXSMM_DNN_CHANNEL_RCPSTDDEV ) {
      return_tensor = handle->rcpstddev;
    } else if ( type == LIBXSMM_DNN_CHANNEL_VARIANCE ) {
      return_tensor = handle->variance;
    } else if ( type == LIBXSMM_DNN_RELU_MASK ) {
      return_tensor = handle->relumask;
    } else {
      /* cannot happen */
    }
  } else {
    *status = LIBXSMM_DNN_ERR_INVALID_HANDLE;
  }

  return return_tensor;
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_fusedbatchnorm_release_tensor(libxsmm_dnn_fusedbatchnorm* handle, const libxsmm_dnn_tensor_type type) {
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  /* check for tensor type */
  if ( (type != LIBXSMM_DNN_REGULAR_INPUT)         && (type != LIBXSMM_DNN_GRADIENT_INPUT)         &&
       (type != LIBXSMM_DNN_REGULAR_OUTPUT)        && (type != LIBXSMM_DNN_GRADIENT_OUTPUT)        &&
       (type != LIBXSMM_DNN_REGULAR_INPUT_ADD)     && (type != LIBXSMM_DNN_GRADIENT_INPUT_ADD)     &&
       (type != LIBXSMM_DNN_REGULAR_CHANNEL_BETA)  && (type != LIBXSMM_DNN_GRADIENT_CHANNEL_BETA)  &&
       (type != LIBXSMM_DNN_REGULAR_CHANNEL_GAMMA) && (type != LIBXSMM_DNN_GRADIENT_CHANNEL_GAMMA) &&
       (type != LIBXSMM_DNN_CHANNEL_EXPECTVAL)     && (type != LIBXSMM_DNN_CHANNEL_RCPSTDDEV)      &&
       (type != LIBXSMM_DNN_CHANNEL_VARIANCE)      && (type != LIBXSMM_DNN_RELU_MASK)                 ) {
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
    } else if ( type == LIBXSMM_DNN_REGULAR_INPUT_ADD ) {
      handle->reg_add = 0;
    } else if ( type == LIBXSMM_DNN_GRADIENT_INPUT_ADD ) {
      handle->grad_add = 0;
    } else if ( type == LIBXSMM_DNN_REGULAR_CHANNEL_BETA ) {
      handle->reg_beta = 0;
    } else if ( type == LIBXSMM_DNN_GRADIENT_CHANNEL_BETA ) {
      handle->grad_beta = 0;
    } else if ( type == LIBXSMM_DNN_REGULAR_CHANNEL_GAMMA ) {
      handle->reg_gamma = 0;
    } else if ( type == LIBXSMM_DNN_GRADIENT_CHANNEL_GAMMA ) {
      handle->grad_gamma = 0;
    } else if ( type == LIBXSMM_DNN_CHANNEL_EXPECTVAL ) {
      handle->expvalue = 0;
    } else if ( type == LIBXSMM_DNN_CHANNEL_RCPSTDDEV ) {
      handle->rcpstddev = 0;
    } else if ( type == LIBXSMM_DNN_CHANNEL_VARIANCE ) {
      handle->variance = 0;
    } else if ( type == LIBXSMM_DNN_RELU_MASK ) {
      handle->relumask = 0;
    } else {
      /* cannot happen */
    }
  } else {
    status = LIBXSMM_DNN_ERR_INVALID_HANDLE;
  }

  return status;
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_fusedbatchnorm_execute_st(libxsmm_dnn_fusedbatchnorm* handle, libxsmm_dnn_compute_kind kind,
  /*unsigned*/int start_thread, /*unsigned*/int tid) {
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  if (0 != handle) {
    switch (kind) {
      case LIBXSMM_DNN_COMPUTE_KIND_FWD: {
        switch (handle->desc.buffer_format) {
          case LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM: {
            status = libxsmm_dnn_fusedbatchnorm_st_fwd_custom( handle, start_thread, tid );
          } break;
          default: {
            status = LIBXSMM_DNN_ERR_INVALID_FORMAT_FUSEDBN;
          }
        }
      } break;
      case LIBXSMM_DNN_COMPUTE_KIND_BWD: {
        switch (handle->desc.buffer_format) {
          case LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM: {
            status = libxsmm_dnn_fusedbatchnorm_st_bwd_custom( handle, start_thread, tid );
          } break;
          default: {
            status = LIBXSMM_DNN_ERR_INVALID_FORMAT_FUSEDBN;
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


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_fusedbatchnorm_reduce_stats_st(libxsmm_dnn_fusedbatchnorm** handles, int num_handles, libxsmm_dnn_compute_kind kind,
  /*unsigned*/int start_thread, /*unsigned*/int tid) {
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  if (0 != handles && num_handles > 0) {
    switch (kind) {
      case LIBXSMM_DNN_COMPUTE_KIND_FWD: {
        switch (handles[0]->desc.buffer_format) {
          case LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM: {
            status = libxsmm_dnn_fusedbatchnorm_reduce_stats_st_fwd_custom( handles, num_handles, start_thread, tid );
          } break;
          default: {
            status = LIBXSMM_DNN_ERR_INVALID_FORMAT_FUSEDBN;
          }
        }
      } break;
      case LIBXSMM_DNN_COMPUTE_KIND_BWD: {
        switch (handles[0]->desc.buffer_format) {
          case LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM: {
            status = libxsmm_dnn_fusedbatchnorm_reduce_stats_st_bwd_custom( handles, num_handles, start_thread, tid );
          } break;
          default: {
            status = LIBXSMM_DNN_ERR_INVALID_FORMAT_FUSEDBN;
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
