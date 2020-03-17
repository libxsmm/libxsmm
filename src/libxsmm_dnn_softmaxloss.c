/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Alexander Heinecke, Sasikanth Avancha (Intel Corp.)
******************************************************************************/
#include "libxsmm_dnn_softmaxloss_backward.h"
#include "libxsmm_dnn_softmaxloss_forward.h"
#include "libxsmm_main.h"


LIBXSMM_API libxsmm_dnn_softmaxloss* libxsmm_dnn_create_softmaxloss(libxsmm_dnn_softmaxloss_desc softmaxloss_desc, libxsmm_dnn_err_t* status) {
  libxsmm_dnn_softmaxloss* handle = 0;
  int lpb;

  if ( (softmaxloss_desc.datatype == LIBXSMM_DNN_DATATYPE_F32) || (softmaxloss_desc.datatype == LIBXSMM_DNN_DATATYPE_BF16) ) {
    handle = (libxsmm_dnn_softmaxloss*)malloc(sizeof(libxsmm_dnn_softmaxloss));

    if (0 != handle) {
      *status = LIBXSMM_DNN_SUCCESS;
      /* zero entire content; not only safer but also sets data and code pointers to NULL */
      memset(handle, 0, sizeof(*handle));
      /* let's make the description persistent */
      handle->desc = softmaxloss_desc;

      /* cnn */
      if ( (handle->desc.buffer_format & LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM) > 0 ) {
        int bk;
        /* we need to compute the memory layout given the */
        *status = libxsmm_dnn_get_feature_map_blocks( handle->desc.C, handle->desc.C,
                                                      &(handle->bc), &bk, &lpb,
                                                      handle->desc.datatype, handle->desc.datatype );
        /* compute the outer blocks */
        handle->Bc = handle->desc.C / handle->bc;
        handle->bn = 1;
        handle->Bn = handle->desc.N;
      } else if ( (handle->desc.buffer_format & LIBXSMM_DNN_TENSOR_FORMAT_NCPACKED) > 0 ) {
        handle->bc = handle->desc.bc;
        handle->bn = handle->desc.bn;
        handle->Bc = handle->desc.C / handle->bc;
        handle->Bn = handle->desc.N / handle->bn;
      } else {
        *status = LIBXSMM_DNN_ERR_CREATE_HANDLE;
        free( handle );
        handle = 0;
        return handle;
      }
      /* create barrier */
      handle->barrier = libxsmm_barrier_create(handle->desc.threads, 1);
      /* calculate scratch size for local softmaxloss copies of one feature map block per thread */
      if ( softmaxloss_desc.datatype == LIBXSMM_DNN_DATATYPE_BF16 ) {
        handle->scratch_size = (sizeof(float)*handle->desc.C*handle->desc.N*2);
      } else {
        handle->scratch_size = 1;
      }
    } else {
      *status = LIBXSMM_DNN_ERR_CREATE_HANDLE;
    }
  } else {
    *status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
  }

  return handle;
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_destroy_softmaxloss(const libxsmm_dnn_softmaxloss* handle) {
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  if (0 != handle) {
    /* Deallocate barrier */
    if (handle->barrier != 0 ) { libxsmm_barrier_release((const libxsmm_barrier*)handle->barrier); }
    /* deallocate handle structure */
    free(/*remove constness*/(libxsmm_dnn_softmaxloss*)handle);
  } else {
    status = LIBXSMM_DNN_ERR_INVALID_HANDLE;
  }

  return status;
}


LIBXSMM_API libxsmm_dnn_tensor_datalayout* libxsmm_dnn_softmaxloss_create_tensor_datalayout(const libxsmm_dnn_softmaxloss* handle, const libxsmm_dnn_tensor_type type, libxsmm_dnn_err_t* status) {
  libxsmm_dnn_tensor_datalayout* layout;

  *status = LIBXSMM_DNN_SUCCESS;
  layout = 0;

  if (handle != 0) {
    layout = (libxsmm_dnn_tensor_datalayout*) malloc(sizeof(libxsmm_dnn_tensor_datalayout));

    if (layout != 0) {
      memset(layout, 0, sizeof(libxsmm_dnn_tensor_datalayout));
      layout->format = handle->desc.buffer_format;

      if ( (type == LIBXSMM_DNN_REGULAR_INPUT)   || (type == LIBXSMM_DNN_GRADIENT_INPUT)  || (type == LIBXSMM_DNN_INPUT) ||
           (type == LIBXSMM_DNN_REGULAR_OUTPUT)  || (type == LIBXSMM_DNN_OUTPUT)                                            ) {
        if ((handle->desc.buffer_format & LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM) > 0) {
          layout->datatype = handle->desc.datatype;
          layout->dim_type = (libxsmm_dnn_tensor_dimtype*) malloc(3*sizeof(libxsmm_dnn_tensor_dimtype));
          layout->dim_size = (unsigned int*) malloc(3*sizeof(unsigned int));

          if (0 != layout->dim_type && 0 != layout->dim_size) {
            layout->num_dims = 3;
            layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
            layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
            layout->dim_type[2] = LIBXSMM_DNN_TENSOR_DIMTYPE_N;
            layout->dim_size[0] = handle->bc;
            layout->dim_size[1] = handle->Bc;
            layout->dim_size[2] = handle->desc.N;
          } else {
            free(layout);
            layout = 0; /* make sure a NULL is returned */
            *status = LIBXSMM_DNN_ERR_CREATE_LAYOUT_ARRAYS;
          }
        } else if ((handle->desc.buffer_format & LIBXSMM_DNN_TENSOR_FORMAT_NCPACKED) > 0) {
          layout->datatype = handle->desc.datatype;
          layout->dim_type = (libxsmm_dnn_tensor_dimtype*) malloc(4*sizeof(libxsmm_dnn_tensor_dimtype));
          layout->dim_size = (unsigned int*) malloc(4*sizeof(unsigned int));

          if (0 != layout->dim_type && 0 != layout->dim_size) {
            layout->num_dims = 4;
            layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
            layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_N;
            layout->dim_type[2] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
            layout->dim_type[3] = LIBXSMM_DNN_TENSOR_DIMTYPE_N;
            layout->dim_size[0] = handle->bc;
            layout->dim_size[1] = handle->bn;
            layout->dim_size[2] = handle->Bc;
            layout->dim_size[3] = handle->Bn;
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
      } else if ( type == LIBXSMM_DNN_LABEL ) {
        layout->datatype = LIBXSMM_DNN_DATATYPE_I32;
        layout->dim_type = (libxsmm_dnn_tensor_dimtype*) malloc(1*sizeof(libxsmm_dnn_tensor_dimtype));
        layout->dim_size = (unsigned int*) malloc(1*sizeof(unsigned int));

        if (0 != layout->dim_type && 0 != layout->dim_size) {
          layout->num_dims = 1;
          layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_N;
          layout->dim_size[0] = handle->desc.N;
        } else {
          free(layout);
          layout = 0; /* make sure a NULL is returned */
          *status = LIBXSMM_DNN_ERR_CREATE_LAYOUT_ARRAYS;
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


LIBXSMM_API size_t libxsmm_dnn_softmaxloss_get_scratch_size(const libxsmm_dnn_softmaxloss* handle, libxsmm_dnn_err_t* status) {
  size_t l_scratch_size = 0;
  *status = LIBXSMM_DNN_SUCCESS;

  if (0 != handle) {
    l_scratch_size = handle->scratch_size + 64; /* 64 byte extra in case the user code does not care about alignment */
  } else {
    *status = LIBXSMM_DNN_ERR_INVALID_HANDLE;
  }

  return l_scratch_size;
}


LIBXSMM_API void* libxsmm_dnn_softmaxloss_get_scratch_ptr(const libxsmm_dnn_softmaxloss* handle, libxsmm_dnn_err_t* status)
{
  *status = LIBXSMM_DNN_SUCCESS;

  if (0 != handle) {
    return handle->scratch;
  } else {
    *status = LIBXSMM_DNN_ERR_INVALID_HANDLE;
  }

  return 0;
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_softmaxloss_bind_scratch(libxsmm_dnn_softmaxloss* handle, const void* scratch) {
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


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_softmaxloss_release_scratch(libxsmm_dnn_softmaxloss* handle) {
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  if (0 != handle) {
    handle->scratch = 0;
  } else {
    status = LIBXSMM_DNN_ERR_INVALID_HANDLE;
  }

  return status;
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_softmaxloss_bind_tensor(libxsmm_dnn_softmaxloss* handle, const libxsmm_dnn_tensor* tensor, const libxsmm_dnn_tensor_type type) {
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  /* check for tensor type */
  if ( (type != LIBXSMM_DNN_REGULAR_INPUT)  && (type != LIBXSMM_DNN_GRADIENT_INPUT) &&
       (type != LIBXSMM_DNN_REGULAR_OUTPUT) && (type != LIBXSMM_DNN_LABEL)             ) {
    status = LIBXSMM_DNN_ERR_UNKNOWN_TENSOR_TYPE;
    return status;
  }

  if (handle != 0 && tensor != 0) {
    libxsmm_dnn_tensor_datalayout* handle_layout = libxsmm_dnn_softmaxloss_create_tensor_datalayout(handle, type, &status);

    if ( libxsmm_dnn_compare_tensor_datalayout(handle_layout, tensor->layout, &status) == 0 ) {
      if ( type == LIBXSMM_DNN_REGULAR_INPUT ) {
        handle->reg_input = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_GRADIENT_INPUT ) {
        handle->grad_input = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_REGULAR_OUTPUT ) {
        handle->reg_output = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_LABEL ) {
        handle->label = (libxsmm_dnn_tensor*)tensor;
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


LIBXSMM_API libxsmm_dnn_tensor* libxsmm_dnn_softmaxloss_get_tensor(libxsmm_dnn_softmaxloss* handle, const libxsmm_dnn_tensor_type type, libxsmm_dnn_err_t* status) {
  libxsmm_dnn_tensor* return_tensor = 0;

  *status = LIBXSMM_DNN_SUCCESS;

  /* check for tensor type */
  if ( (type != LIBXSMM_DNN_REGULAR_INPUT)  && (type != LIBXSMM_DNN_GRADIENT_INPUT) &&
       (type != LIBXSMM_DNN_REGULAR_OUTPUT) && (type != LIBXSMM_DNN_LABEL)             ) {
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
    } else if ( type == LIBXSMM_DNN_LABEL ) {
      return_tensor = handle->label;
    } else {
      /* cannot happen */
    }
  } else {
    *status = LIBXSMM_DNN_ERR_INVALID_HANDLE;
  }

  return return_tensor;
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_softmaxloss_release_tensor(libxsmm_dnn_softmaxloss* handle, const libxsmm_dnn_tensor_type type) {
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  /* check for tensor type */
  if ( (type != LIBXSMM_DNN_REGULAR_INPUT)  && (type != LIBXSMM_DNN_GRADIENT_INPUT) &&
       (type != LIBXSMM_DNN_REGULAR_OUTPUT) && (type != LIBXSMM_DNN_LABEL)             ) {
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
    } else if ( type == LIBXSMM_DNN_LABEL ) {
      handle->label = 0;
    } else {
      /* cannot happen */
    }
  } else {
    status = LIBXSMM_DNN_ERR_INVALID_HANDLE;
  }

  return status;
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_softmaxloss_execute_st(libxsmm_dnn_softmaxloss* handle, libxsmm_dnn_compute_kind kind,
  /*unsigned*/int start_thread, /*unsigned*/int tid) {
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  if (0 != handle) {
    switch (kind) {
      case LIBXSMM_DNN_COMPUTE_KIND_FWD: {
        status = libxsmm_dnn_softmaxloss_st_fwd_ncnc( handle, start_thread, tid );
      } break;
      case LIBXSMM_DNN_COMPUTE_KIND_BWD: {
        status = libxsmm_dnn_softmaxloss_st_bwd_ncnc( handle, start_thread, tid );
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

LIBXSMM_API float libxsmm_dnn_softmaxloss_get_loss(const libxsmm_dnn_softmaxloss* handle, libxsmm_dnn_err_t* status) {
  float l_loss = 0.0f;
  *status = LIBXSMM_DNN_SUCCESS;

  if (0 != handle) {
    l_loss = handle->loss;
  } else {
    *status = LIBXSMM_DNN_ERR_INVALID_HANDLE;
  }

  return l_loss;
}

