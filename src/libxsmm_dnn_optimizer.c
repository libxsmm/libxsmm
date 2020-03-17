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
#include "libxsmm_dnn_optimizer_sgd.h"
#include "libxsmm_main.h"


LIBXSMM_API libxsmm_dnn_optimizer* libxsmm_dnn_create_optimizer(libxsmm_dnn_optimizer_desc optimizer_desc, libxsmm_dnn_err_t* status) {
  libxsmm_dnn_optimizer* handle = 0;

  if ( (optimizer_desc.datatype == LIBXSMM_DNN_DATATYPE_F32) || (optimizer_desc.datatype == LIBXSMM_DNN_DATATYPE_BF16) ) {
    handle = (libxsmm_dnn_optimizer*)malloc(sizeof(libxsmm_dnn_optimizer));

    if (0 != handle) {
      *status = LIBXSMM_DNN_SUCCESS;
      /* zero entire content; not only safer but also sets data and code pointers to NULL */
      memset(handle, 0, sizeof(*handle));
      /* let's make the description persistent */
      handle->desc = optimizer_desc;

      if ( (handle->desc.filter_format & LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM) > 0 ) {
        /* we need to compute the memory layout given the */
        *status = libxsmm_dnn_get_feature_map_blocks( handle->desc.C, handle->desc.K,
                                                      &(handle->bc), &(handle->bk), &(handle->fm_lp_block),
                                                      handle->desc.datatype, handle->desc.datatype );
        /* compute the outer blocks */
        handle->Bc = handle->desc.C / handle->bc;
        handle->Bk = handle->desc.K / handle->bk;
      } else if ( (handle->desc.filter_format & LIBXSMM_DNN_TENSOR_FORMAT_CKPACKED) > 0 ) {
        if ( optimizer_desc.datatype == LIBXSMM_DNN_DATATYPE_F32 ) {
          handle->fm_lp_block = 1;
        } else if ( optimizer_desc.datatype == LIBXSMM_DNN_DATATYPE_BF16 ) {
          handle->fm_lp_block = 2;
        } else {
        }
        handle->bc = handle->desc.bc;
        handle->bk = handle->desc.bk;
        handle->Bc = handle->desc.C / handle->bc;
        handle->Bk = handle->desc.K / handle->bk;
      } else {
        *status = LIBXSMM_DNN_ERR_CREATE_HANDLE;
        free( handle );
        handle = 0;
        return handle;
      }
      /* create barrier */
      handle->barrier = libxsmm_barrier_create(handle->desc.threads, 1);
      /* calculate scratch size for local optimizer copies of one feature map block per thread */
      handle->scratch_size = 1;
    } else {
      *status = LIBXSMM_DNN_ERR_CREATE_HANDLE;
    }
  } else {
    *status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
  }

  return handle;
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_destroy_optimizer(const libxsmm_dnn_optimizer* handle) {
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  if (0 != handle) {
    /* Deallocate barrier */
    if (handle->barrier != 0 ) { libxsmm_barrier_release((const libxsmm_barrier*)handle->barrier); }
    /* deallocate handle structure */
    free(/*remove constness*/(libxsmm_dnn_optimizer*)handle);
  } else {
    status = LIBXSMM_DNN_ERR_INVALID_HANDLE;
  }

  return status;
}


LIBXSMM_API libxsmm_dnn_tensor_datalayout* libxsmm_dnn_optimizer_create_tensor_datalayout(const libxsmm_dnn_optimizer* handle, const libxsmm_dnn_tensor_type type, libxsmm_dnn_err_t* status) {
  libxsmm_dnn_tensor_datalayout* layout;

  *status = LIBXSMM_DNN_SUCCESS;
  layout = 0;

  if (handle != 0) {
    layout = (libxsmm_dnn_tensor_datalayout*) malloc(sizeof(libxsmm_dnn_tensor_datalayout));

    if (layout != 0) {
      memset(layout, 0, sizeof(libxsmm_dnn_tensor_datalayout));
      layout->format = handle->desc.filter_format;

      if ( (type == LIBXSMM_DNN_REGULAR_FILTER) || (type == LIBXSMM_DNN_GRADIENT_FILTER) || (type == LIBXSMM_DNN_MASTER_FILTER) ) {
        if ( ((handle->desc.filter_format & LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM) > 0) || ((handle->desc.filter_format & LIBXSMM_DNN_TENSOR_FORMAT_CKPACKED) > 0) ) {
          if ( handle->desc.datatype == LIBXSMM_DNN_DATATYPE_F32 ) {
            layout->datatype = handle->desc.datatype;
            layout->dim_type = (libxsmm_dnn_tensor_dimtype*) malloc(4*sizeof(libxsmm_dnn_tensor_dimtype));
            layout->dim_size = (unsigned int*) malloc(4*sizeof(unsigned int));

            if (0 != layout->dim_type && 0 != layout->dim_size) {
              layout->num_dims = 4;
              layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
              layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
              layout->dim_type[2] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
              layout->dim_type[3] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
              layout->dim_size[0] = handle->bk;
              layout->dim_size[1] = handle->bc;
              layout->dim_size[2] = handle->Bc;
              layout->dim_size[3] = handle->Bk;
            } else {
              free(layout);
              layout = 0; /* make sure a NULL is returned */
              *status = LIBXSMM_DNN_ERR_CREATE_LAYOUT_ARRAYS;
            }
          } else if ( handle->desc.datatype == LIBXSMM_DNN_DATATYPE_BF16 ) {
            layout->datatype = handle->desc.datatype;
            layout->dim_type = (libxsmm_dnn_tensor_dimtype*) malloc(5*sizeof(libxsmm_dnn_tensor_dimtype));
            layout->dim_size = (unsigned int*) malloc(5*sizeof(unsigned int));

            if (0 != layout->dim_type && 0 != layout->dim_size) {
              layout->num_dims = 5;
              layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
              layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
              layout->dim_type[2] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
              layout->dim_type[3] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
              layout->dim_type[4] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
              layout->dim_size[0] = handle->fm_lp_block;
              layout->dim_size[1] = handle->bk;
              layout->dim_size[2] = handle->bc/handle->fm_lp_block;
              layout->dim_size[3] = handle->Bc;
              layout->dim_size[4] = handle->Bk;
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


LIBXSMM_API size_t libxsmm_dnn_optimizer_get_scratch_size(const libxsmm_dnn_optimizer* handle, libxsmm_dnn_err_t* status) {
  size_t l_scratch_size = 0;
  *status = LIBXSMM_DNN_SUCCESS;

  if (0 != handle) {
    l_scratch_size = handle->scratch_size + 64; /* 64 byte extra in case the user code does not care about alignment */
  } else {
    *status = LIBXSMM_DNN_ERR_INVALID_HANDLE;
  }

  return l_scratch_size;
}


LIBXSMM_API void* libxsmm_dnn_optimizer_get_scratch_ptr(const libxsmm_dnn_optimizer* handle, libxsmm_dnn_err_t* status)
{
  *status = LIBXSMM_DNN_SUCCESS;

  if (0 != handle) {
    return handle->scratch;
  } else {
    *status = LIBXSMM_DNN_ERR_INVALID_HANDLE;
  }

  return 0;
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_optimizer_bind_scratch(libxsmm_dnn_optimizer* handle, const void* scratch) {
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


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_optimizer_release_scratch(libxsmm_dnn_optimizer* handle) {
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  if (0 != handle) {
    handle->scratch = 0;
  } else {
    status = LIBXSMM_DNN_ERR_INVALID_HANDLE;
  }

  return status;
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_optimizer_bind_tensor(libxsmm_dnn_optimizer* handle, const libxsmm_dnn_tensor* tensor, const libxsmm_dnn_tensor_type type) {
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  /* check for tensor type */
  if ( (type != LIBXSMM_DNN_REGULAR_FILTER) && (type != LIBXSMM_DNN_GRADIENT_FILTER) && (type != LIBXSMM_DNN_MASTER_FILTER) ) {
    status = LIBXSMM_DNN_ERR_UNKNOWN_TENSOR_TYPE;
    return status;
  }

  if (handle != 0 && tensor != 0) {
    libxsmm_dnn_tensor_datalayout* handle_layout = libxsmm_dnn_optimizer_create_tensor_datalayout(handle, type, &status);

    if ( libxsmm_dnn_compare_tensor_datalayout(handle_layout, tensor->layout, &status) == 0 ) {
      if ( type == LIBXSMM_DNN_REGULAR_FILTER ) {
        handle->reg_filter = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_GRADIENT_FILTER ) {
        handle->grad_filter = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_MASTER_FILTER ) {
        handle->master_filter = (libxsmm_dnn_tensor*)tensor;
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


LIBXSMM_API libxsmm_dnn_tensor* libxsmm_dnn_optimizer_get_tensor(libxsmm_dnn_optimizer* handle, const libxsmm_dnn_tensor_type type, libxsmm_dnn_err_t* status) {
  libxsmm_dnn_tensor* return_tensor = 0;

  *status = LIBXSMM_DNN_SUCCESS;

  /* check for tensor type */
  if ( (type != LIBXSMM_DNN_REGULAR_FILTER) && (type != LIBXSMM_DNN_GRADIENT_FILTER) && (type != LIBXSMM_DNN_MASTER_FILTER) ) {
    *status = LIBXSMM_DNN_ERR_UNKNOWN_TENSOR_TYPE;
    return return_tensor;
  }

  if (handle != 0) {
    if ( type == LIBXSMM_DNN_REGULAR_FILTER ) {
      return_tensor = handle->reg_filter;
    } else if ( type == LIBXSMM_DNN_GRADIENT_FILTER ) {
      return_tensor = handle->grad_filter;
    } else if ( type == LIBXSMM_DNN_MASTER_FILTER ) {
      return_tensor = handle->master_filter;
    } else {
      /* cannot happen */
    }
  } else {
    *status = LIBXSMM_DNN_ERR_INVALID_HANDLE;
  }

  return return_tensor;
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_optimizer_release_tensor(libxsmm_dnn_optimizer* handle, const libxsmm_dnn_tensor_type type) {
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  /* check for tensor type */
  if ( (type != LIBXSMM_DNN_REGULAR_FILTER) && (type != LIBXSMM_DNN_GRADIENT_FILTER) && (type != LIBXSMM_DNN_MASTER_FILTER) ) {
    status = LIBXSMM_DNN_ERR_UNKNOWN_TENSOR_TYPE;
    return status;
  }

  if (handle != 0) {
    if ( type == LIBXSMM_DNN_REGULAR_FILTER ) {
      handle->reg_filter = 0;
    } else if ( type == LIBXSMM_DNN_GRADIENT_FILTER ) {
      handle->grad_filter = 0;
    } else if ( type == LIBXSMM_DNN_MASTER_FILTER ) {
      handle->master_filter = 0;
    } else {
      /* cannot happen */
    }
  } else {
    status = LIBXSMM_DNN_ERR_INVALID_HANDLE;
  }

  return status;
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_optimizer_execute_st(libxsmm_dnn_optimizer* handle, /*unsigned*/int start_thread, /*unsigned*/int tid) {
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  if (0 != handle) {
    LIBXSMM_UNUSED(start_thread);
    LIBXSMM_UNUSED(tid);
  }
  else {
    status = LIBXSMM_DNN_ERR_INVALID_HANDLE;
  }

  return status;
}

