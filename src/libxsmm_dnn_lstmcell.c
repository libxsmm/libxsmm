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
/* Alexander Heinecke, Kunal Banerjee (Intel Corp.)
******************************************************************************/
#include "libxsmm_dnn_elementwise.h"
#include "libxsmm_main.h"

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#include <string.h>
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif


LIBXSMM_API libxsmm_dnn_lstmcell* libxsmm_dnn_create_lstmcell(libxsmm_dnn_lstmcell_desc lstmcell_desc, libxsmm_dnn_err_t* status)
{
  libxsmm_dnn_lstmcell* handle = 0;
  handle = (libxsmm_dnn_lstmcell*)malloc(sizeof(libxsmm_dnn_lstmcell));
  if (0 != handle) {
    *status = LIBXSMM_DNN_SUCCESS;
    /* zero entire content; not only safer but also sets data and code pointers to NULL */
    memset(handle, 0, sizeof(*handle));
    /* initialize known handle components */
    handle->desc = lstmcell_desc;
    if ( (lstmcell_desc.datatype_in != LIBXSMM_DNN_DATATYPE_F32) || (lstmcell_desc.datatype_out != LIBXSMM_DNN_DATATYPE_F32) ) {
      /* error */
      *status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
      return handle;
    }
    handle->bk = 64;
    handle->bn = 64;
    handle->bc = 64;
    if (lstmcell_desc.t < 1) {
      *status = LIBXSMM_DNN_ERR_TIME_STEPS_TOO_SMALL;
    }
    /* Need to allocate space for scratch and internalstate libxsmm_dnn_tensor's */
    handle->dit  = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->dft  = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->dot  = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->dct  = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->deltat = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->doutt  = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->t1 = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->t2 = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->barrier = libxsmm_barrier_create(handle->desc.nThreads, 1);
    if (NULL == handle->dit  || NULL == handle->dft    || NULL == handle->dot  || 
        NULL == handle->dct  || NULL == handle->deltat || NULL == handle->doutt ||
        NULL == handle->t1  || NULL == handle->t2      || NULL == handle->barrier)
    {
      free(handle->dit);    free(handle->dft);   free(handle->dot); free(handle->dct);
      free(handle->deltat); free(handle->doutt); free(handle->t1);  free(handle->t2);
      *status = LIBXSMM_DNN_ERR_CREATE_HANDLE;
    }
  } else {
    *status = LIBXSMM_DNN_ERR_CREATE_HANDLE;
  }
  return handle;
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_destroy_lstmcell(const libxsmm_dnn_lstmcell* handle)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
  if (0 != handle) {
    free(handle->dit);    free(handle->dft);   free(handle->dot); free(handle->dct);
    free(handle->deltat); free(handle->doutt); free(handle->t1);  free(handle->t2);
    /* Deallocate barrier */
    if (handle->barrier != 0 ) { libxsmm_barrier_release((const libxsmm_barrier*)handle->barrier); }
    /* deallocate handle structure */
    free(/*remove constness*/(libxsmm_dnn_lstmcell*)handle);
  }
  return status;
}


LIBXSMM_API libxsmm_dnn_tensor_datalayout* libxsmm_dnn_lstmcell_create_tensor_datalayout(const libxsmm_dnn_lstmcell* handle, const libxsmm_dnn_tensor_type type, libxsmm_dnn_err_t* status)
{
  libxsmm_dnn_tensor_datalayout* layout = 0;
  *status = LIBXSMM_DNN_SUCCESS;
  layout = 0;
  if (handle != 0) {
    layout = (libxsmm_dnn_tensor_datalayout*) malloc(sizeof(libxsmm_dnn_tensor_datalayout));
    if (layout != 0) {
      memset(layout, 0, sizeof(libxsmm_dnn_tensor_datalayout));
      /*layout->custom_format = handle->custom_format_type;*/
      if ( (type == LIBXSMM_DNN_LSTM_REGULAR_INPUT)             || (type == LIBXSMM_DNN_LSTM_GRADIENT_INPUT)             ||
           (type == LIBXSMM_DNN_LSTM_REGULAR_CS_PREV)           || (type == LIBXSMM_DNN_LSTM_GRADIENT_CS_PREV)           ||
           (type == LIBXSMM_DNN_LSTM_REGULAR_HIDDEN_STATE_PREV) || (type == LIBXSMM_DNN_LSTM_GRADIENT_HIDDEN_STATE_PREV) ||
           (type == LIBXSMM_DNN_LSTM_REGULAR_WEIGHT)            || (type == LIBXSMM_DNN_LSTM_GRADIENT_WEIGHT)            ||
           (type == LIBXSMM_DNN_LSTM_REGULAR_BIAS)              || (type == LIBXSMM_DNN_LSTM_GRADIENT_BIAS)              ||
           (type == LIBXSMM_DNN_LSTM_REGULAR_CS)                || (type == LIBXSMM_DNN_LSTM_GRADIENT_CS)                ||
           (type == LIBXSMM_DNN_LSTM_REGULAR_HIDDEN_STATE)      || (type == LIBXSMM_DNN_LSTM_GRADIENT_HIDDEN_STATE)      ||
           (type == LIBXSMM_DNN_LSTM_INTERNAL_I)                || (type == LIBXSMM_DNN_LSTM_INTERNAL_F)                 ||
           (type == LIBXSMM_DNN_LSTM_INTERNAL_O)                || (type == LIBXSMM_DNN_LSTM_INTERNAL_C) ) {
        layout->format = handle->desc.buffer_format;
        layout->tensor_type = LIBXSMM_DNN_ACTIVATION;

        if ((handle->desc.buffer_format & LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM) > 0) {
          if ( ((handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_F32) && (handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_F32) ) ) {
            layout->datatype = LIBXSMM_DNN_DATATYPE_F32;
            if (1 /*handle->custom_format_type == LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM_1*/) {
              layout->dim_type = (libxsmm_dnn_tensor_dimtype*) malloc(4*sizeof(libxsmm_dnn_tensor_dimtype));
              layout->dim_size = (unsigned int*) malloc(4*sizeof(unsigned int));

              if (0 != layout->dim_type && 0 != layout->dim_size) { /* TODO: handle the error */
                layout->num_dims = 4;
                if ( (type == LIBXSMM_DNN_LSTM_REGULAR_INPUT) || (type == LIBXSMM_DNN_LSTM_GRADIENT_INPUT) ) {
                  layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_RLK;
                  layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_RLN;
                  layout->dim_type[2] = LIBXSMM_DNN_TENSOR_DIMTYPE_RLK;
                  layout->dim_type[3] = LIBXSMM_DNN_TENSOR_DIMTYPE_RLN;
                  layout->dim_size[0] = (unsigned int)handle->bc;
                  layout->dim_size[1] = (unsigned int)handle->bn;
                  layout->dim_size[2] = (unsigned int)(handle->desc.C / handle->bc);
                  layout->dim_size[3] = (unsigned int)(handle->desc.N / handle->bn);
                } else if ( (type == LIBXSMM_DNN_LSTM_REGULAR_CS_PREV)           || (type == LIBXSMM_DNN_LSTM_GRADIENT_CS_PREV)           ||
                            (type == LIBXSMM_DNN_LSTM_REGULAR_HIDDEN_STATE_PREV) || (type == LIBXSMM_DNN_LSTM_GRADIENT_HIDDEN_STATE_PREV) ||
                            (type == LIBXSMM_DNN_LSTM_REGULAR_HIDDEN_STATE)      || (type == LIBXSMM_DNN_LSTM_GRADIENT_HIDDEN_STATE)      ||
                            (type == LIBXSMM_DNN_LSTM_REGULAR_CS)                || (type == LIBXSMM_DNN_LSTM_GRADIENT_CS)                ||
                            (type == LIBXSMM_DNN_LSTM_INTERNAL_I)                || (type == LIBXSMM_DNN_LSTM_INTERNAL_F)                 ||
                            (type == LIBXSMM_DNN_LSTM_INTERNAL_O)                || (type == LIBXSMM_DNN_LSTM_INTERNAL_C) ) {
                  layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_RLM;
                  layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_RLN;
                  layout->dim_type[2] = LIBXSMM_DNN_TENSOR_DIMTYPE_RLM;
                  layout->dim_type[3] = LIBXSMM_DNN_TENSOR_DIMTYPE_RLN;
                  layout->dim_size[0] = (unsigned int)handle->bk;
                  layout->dim_size[1] = (unsigned int)handle->bn;
                  layout->dim_size[2] = (unsigned int)(handle->desc.K / handle->bk);
                  layout->dim_size[3] = (unsigned int)(handle->desc.N / handle->bn);
                } else if ( (type == LIBXSMM_DNN_LSTM_REGULAR_WEIGHT) || (type == LIBXSMM_DNN_LSTM_GRADIENT_WEIGHT) ) {
                  layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_RLM;
                  layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_RLK;
                  layout->dim_type[2] = LIBXSMM_DNN_TENSOR_DIMTYPE_RLK;
                  layout->dim_type[3] = LIBXSMM_DNN_TENSOR_DIMTYPE_RLM;
                  layout->dim_size[0] = (unsigned int)handle->bk;
                  layout->dim_size[1] = (unsigned int)handle->bc;
                  layout->dim_size[2] = (unsigned int)(handle->desc.C / handle->bc);
                  layout->dim_size[3] = (unsigned int)(handle->desc.K / handle->bk);
                } else if ( (type == LIBXSMM_DNN_LSTM_REGULAR_BIAS) || (type == LIBXSMM_DNN_LSTM_GRADIENT_BIAS) ) {
                  layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_RLM;
                  layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_RLN;
                  layout->dim_type[2] = LIBXSMM_DNN_TENSOR_DIMTYPE_RLM;
                  layout->dim_type[3] = LIBXSMM_DNN_TENSOR_DIMTYPE_RLN;
                  layout->dim_size[0] = (unsigned int)handle->bk;
                  layout->dim_size[1] = (unsigned int)handle->bn;
                  layout->dim_size[2] = (unsigned int)(handle->desc.K / handle->bk);
                  layout->dim_size[3] = (unsigned int)(handle->desc.N / handle->bn);
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
              *status = LIBXSMM_DNN_ERR_UNKNOWN_TENSOR_TYPE;
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
  } else {
    *status = LIBXSMM_DNN_ERR_INVALID_HANDLE;
  }
  return layout;
}


LIBXSMM_API size_t libxsmm_dnn_lstmcell_get_scratch_size(const libxsmm_dnn_lstmcell* handle, const libxsmm_dnn_compute_kind kind, libxsmm_dnn_err_t* status)
{
  const size_t sizeof_datatype = sizeof(float);
  size_t size = 0;
  *status = LIBXSMM_DNN_SUCCESS;

  if (0 != handle) {
    switch (kind) {
      case LIBXSMM_DNN_COMPUTE_KIND_FWD: {
                                           size += 0;
                                         } break;
      case LIBXSMM_DNN_COMPUTE_KIND_BWD:
      case LIBXSMM_DNN_COMPUTE_KIND_UPD:
      case LIBXSMM_DNN_COMPUTE_KIND_ALL: {
                                           size += (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype * (size_t)handle->desc.t; /* dit */
                                           size += 64;
                                           size += (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype * (size_t)handle->desc.t; /* dft */
                                           size += 64;
                                           size += (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype * (size_t)handle->desc.t; /* dot */
                                           size += 64;
                                           size += (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype * (size_t)handle->desc.t; /* dct */
                                           size += 64;
                                           size += (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype * (size_t)handle->desc.t; /* deltat */
                                           size += 64;
                                           size += (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype * (size_t)handle->desc.t; /* doutt */
                                           size += 64;
                                           size += (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype; /* t1 */
                                           size += 64;
                                           size += (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype; /* t2 */
                                           size += 64;
                                         } break;
      default: {
                 *status = LIBXSMM_DNN_ERR_INVALID_KIND;
               }
    }
  } else {
    *status = LIBXSMM_DNN_ERR_INVALID_HANDLE;
  }

  return size;
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_lstmcell_bind_scratch(libxsmm_dnn_lstmcell* handle, const libxsmm_dnn_compute_kind kind, const void* scratch)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
  uintptr_t address = (uintptr_t)scratch;
  size_t offset = 0;
  size_t scratch_size = 0;
  const size_t sizeof_datatype = sizeof(float);

  if (0 != handle) {
    switch (kind) {
      case LIBXSMM_DNN_COMPUTE_KIND_FWD: {
        /* forward only has no scratch need */
      } break;
      case LIBXSMM_DNN_COMPUTE_KIND_BWD:
      case LIBXSMM_DNN_COMPUTE_KIND_UPD:
      case LIBXSMM_DNN_COMPUTE_KIND_ALL: {
        if (scratch == 0) {
          status = LIBXSMM_DNN_ERR_SCRATCH_NOT_ALLOCED;
          return status;
        }

        if (address % 64 == 0) {
          handle->dit->data = (void*)address;
        } else {
          offset = (64 - address % 64);
          handle->dit->data = (void*)(address+offset);
        }
        scratch_size = (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype * (size_t)handle->desc.t;
        address += scratch_size + 64;
        if (address % 64 == 0) {
          handle->dft->data = (void*)address;
        } else {
          offset = (64 - address % 64);
          handle->dft->data = (void*)(address+offset);
        }
        scratch_size = (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype * (size_t)handle->desc.t;
        address += scratch_size + 64;
        if (address % 64 == 0) {
          handle->dot->data = (void*)address;
        } else {
          offset = (64 - address % 64);
          handle->dot->data = (void*)(address+offset);
        }
        scratch_size = (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype * (size_t)handle->desc.t;
        address += scratch_size + 64;
        if (address % 64 == 0) {
          handle->dct->data = (void*)address;
        } else {
          offset = (64 - address % 64);
          handle->dct->data = (void*)(address+offset);
        }
        scratch_size = (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype * (size_t)handle->desc.t;
        address += scratch_size + 64;
        if (address % 64 == 0) {
          handle->deltat->data = (void*)address;
        } else {
          offset = (64 - address % 64);
          handle->deltat->data = (void*)(address+offset);
        }
        scratch_size = (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype * (size_t)handle->desc.t;
        address += scratch_size + 64;
        if (address % 64 == 0) {
          handle->doutt->data = (void*)address;
        } else {
          offset = (64 - address % 64);
          handle->doutt->data = (void*)(address+offset);
        }
        scratch_size = (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype * (size_t)handle->desc.t;
        address += scratch_size + 64;
        if (address % 64 == 0) {
          handle->t1->data = (void*)address;
        } else {
          offset = (64 - address % 64);
          handle->t1->data = (void*)(address+offset);
        }
        scratch_size = (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype;
        address += scratch_size + 64;
        if (address % 64 == 0) {
          handle->t2->data = (void*)address;
        } else {
          offset = (64 - address % 64);
          handle->t2->data = (void*)(address+offset);
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


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_lstmcell_release_scratch(libxsmm_dnn_lstmcell* handle, const libxsmm_dnn_compute_kind kind)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  if (0 != handle) {
    switch (kind) {
      case LIBXSMM_DNN_COMPUTE_KIND_FWD: {
                                         } break;
      case LIBXSMM_DNN_COMPUTE_KIND_BWD:
      case LIBXSMM_DNN_COMPUTE_KIND_UPD:
      case LIBXSMM_DNN_COMPUTE_KIND_ALL: {
                                           handle->dit->data = 0;
                                           handle->dft->data = 0;
                                           handle->dot->data = 0;
                                           handle->dct->data = 0;
                                           handle->deltat->data = 0;
                                           handle->doutt->data = 0;
                                           handle->t1->data = 0;
                                           handle->t2->data = 0;
                                           handle->dit = 0;
                                           handle->dft = 0;
                                           handle->dot = 0;
                                           handle->dct = 0;
                                           handle->deltat = 0;
                                           handle->doutt = 0;
                                           handle->t1 = 0;
                                           handle->t2 = 0;
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


LIBXSMM_API size_t libxsmm_dnn_lstmcell_get_internalstate_size(const libxsmm_dnn_lstmcell* handle, const libxsmm_dnn_compute_kind kind, libxsmm_dnn_err_t* status)
{
  const size_t sizeof_datatype = sizeof(float);
  size_t size = 0;
  *status = LIBXSMM_DNN_SUCCESS;
  LIBXSMM_UNUSED(sizeof_datatype); LIBXSMM_UNUSED(size);

  if (0 != handle) {
    switch (kind) {
      case LIBXSMM_DNN_COMPUTE_KIND_FWD: {
                                           /* with i, f, o, c, d exposed as i/o, there is currently no need for internal state */
                                         } break;
      case LIBXSMM_DNN_COMPUTE_KIND_BWD:
      case LIBXSMM_DNN_COMPUTE_KIND_UPD:
      case LIBXSMM_DNN_COMPUTE_KIND_ALL: {
                                           /* with i, f, o, c, d exposed as i/o, there is currently no need for internal state */
                                         } break;
      default: {
                 *status = LIBXSMM_DNN_ERR_INVALID_KIND;
               }
    }
  } else {
    *status = LIBXSMM_DNN_ERR_INVALID_HANDLE;
  }

  return size;
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_lstmcell_bind_internalstate(libxsmm_dnn_lstmcell* handle, const libxsmm_dnn_compute_kind kind, const void* internalstate)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
  uintptr_t address = (uintptr_t)internalstate;
  size_t offset = 0;
  size_t scratch_size = 0;
  const size_t sizeof_datatype = sizeof(float);
  LIBXSMM_UNUSED(sizeof_datatype); LIBXSMM_UNUSED(address); LIBXSMM_UNUSED(offset); LIBXSMM_UNUSED(scratch_size);

  /*
  if (internalstate == 0) {
    status = LIBXSMM_DNN_ERR_SCRATCH_NOT_ALLOCED;
    return status;
  }
  */

  if (0 != handle) {
    switch (kind) {
      case LIBXSMM_DNN_COMPUTE_KIND_FWD: {
                                         } break;
      case LIBXSMM_DNN_COMPUTE_KIND_BWD:
      case LIBXSMM_DNN_COMPUTE_KIND_UPD:
      case LIBXSMM_DNN_COMPUTE_KIND_ALL: {
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


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_lstmcell_release_internalstate(libxsmm_dnn_lstmcell* handle, const libxsmm_dnn_compute_kind kind)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  if (0 != handle) {
    switch (kind) {
      case LIBXSMM_DNN_COMPUTE_KIND_FWD: {
                                         } break;
      case LIBXSMM_DNN_COMPUTE_KIND_BWD:
      case LIBXSMM_DNN_COMPUTE_KIND_UPD:
      case LIBXSMM_DNN_COMPUTE_KIND_ALL: {
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


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_lstmcell_assign_internalstate(libxsmm_dnn_lstmcell* handle, const void* igoldtb, const void* fgoldtb, const void* ogoldtb, const void* cgoldtb, const void* dgoldtb)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  if (handle != 0 && igoldtb != 0 && fgoldtb != 0 && ogoldtb != 0 && cgoldtb != 0 && dgoldtb != 0) {
    const libxsmm_blasint K = handle->desc.K, N = handle->desc.N, t = handle->desc.t;
    LIBXSMM_VLA_DECL(2, /*const*/ LIBXSMM_DNN_ELTWISE_FTYPE, igold, (/*const*/ LIBXSMM_DNN_ELTWISE_FTYPE*)igoldtb, K * N);
    LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, i, (LIBXSMM_DNN_ELTWISE_FTYPE*)handle->it->data, K * N);
    LIBXSMM_VLA_DECL(2, /*const*/ LIBXSMM_DNN_ELTWISE_FTYPE, fgold, (/*const*/ LIBXSMM_DNN_ELTWISE_FTYPE*)fgoldtb, K * N);
    LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, f, (LIBXSMM_DNN_ELTWISE_FTYPE*)handle->ft->data, K * N);
    LIBXSMM_VLA_DECL(2, /*const*/ LIBXSMM_DNN_ELTWISE_FTYPE, ogold, (/*const*/ LIBXSMM_DNN_ELTWISE_FTYPE*)ogoldtb, K * N);
    LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, o, (LIBXSMM_DNN_ELTWISE_FTYPE*)handle->ot->data, K * N);
    LIBXSMM_VLA_DECL(2, /*const*/ LIBXSMM_DNN_ELTWISE_FTYPE, cgold, (/*const*/ LIBXSMM_DNN_ELTWISE_FTYPE*)cgoldtb, K * N);
    LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, c, (LIBXSMM_DNN_ELTWISE_FTYPE*)handle->ct->data, K * N);
    LIBXSMM_VLA_DECL(2, /*const*/ LIBXSMM_DNN_ELTWISE_FTYPE, dgold, (/*const*/ LIBXSMM_DNN_ELTWISE_FTYPE*)dgoldtb, K * N);
    LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, cs, (LIBXSMM_DNN_ELTWISE_FTYPE*)handle->cst->data, K * N);
    libxsmm_blasint it;
    for (it = 0; it < t; ++it) {
      libxsmm_internal_matrix_copy(K*N, &LIBXSMM_VLA_ACCESS(2, igold, it, 0, K * N), &LIBXSMM_VLA_ACCESS(2, i, it, 0, K * N), 0, 0, 1);
      libxsmm_internal_matrix_copy(K*N, &LIBXSMM_VLA_ACCESS(2, fgold, it, 0, K * N), &LIBXSMM_VLA_ACCESS(2, f, it, 0, K * N), 0, 0, 1);
      libxsmm_internal_matrix_copy(K*N, &LIBXSMM_VLA_ACCESS(2, ogold, it, 0, K * N), &LIBXSMM_VLA_ACCESS(2, o, it, 0, K * N), 0, 0, 1);
      libxsmm_internal_matrix_copy(K*N, &LIBXSMM_VLA_ACCESS(2, cgold, it, 0, K * N), &LIBXSMM_VLA_ACCESS(2, c, it, 0, K * N), 0, 0, 1);
      libxsmm_internal_matrix_copy(K*N, &LIBXSMM_VLA_ACCESS(2, dgold, it, 0, K * N), &LIBXSMM_VLA_ACCESS(2, cs, it, 0, K * N), 0, 0, 1);
    }
  } else {
    status = LIBXSMM_DNN_ERR_INVALID_HANDLE_TENSOR;
  }

  return status;
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_lstmcell_bind_tensor(libxsmm_dnn_lstmcell* handle, const libxsmm_dnn_tensor* tensor, const libxsmm_dnn_tensor_type type)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  /* check for tensor type */
  if ( (type != LIBXSMM_DNN_LSTM_REGULAR_INPUT)             && (type != LIBXSMM_DNN_LSTM_GRADIENT_INPUT)             &&
       (type != LIBXSMM_DNN_LSTM_REGULAR_CS_PREV)           && (type != LIBXSMM_DNN_LSTM_GRADIENT_CS_PREV)           &&
       (type != LIBXSMM_DNN_LSTM_REGULAR_HIDDEN_STATE_PREV) && (type != LIBXSMM_DNN_LSTM_GRADIENT_HIDDEN_STATE_PREV) &&
       (type != LIBXSMM_DNN_LSTM_REGULAR_WEIGHT)            && (type != LIBXSMM_DNN_LSTM_GRADIENT_WEIGHT)            &&
       (type != LIBXSMM_DNN_LSTM_REGULAR_BIAS)              && (type != LIBXSMM_DNN_LSTM_GRADIENT_BIAS)              &&
       (type != LIBXSMM_DNN_LSTM_REGULAR_CS)                && (type != LIBXSMM_DNN_LSTM_GRADIENT_CS)                &&
       (type != LIBXSMM_DNN_LSTM_REGULAR_HIDDEN_STATE)      && (type != LIBXSMM_DNN_LSTM_GRADIENT_HIDDEN_STATE)      &&
       (type != LIBXSMM_DNN_LSTM_INTERNAL_I)                && (type != LIBXSMM_DNN_LSTM_INTERNAL_F)                 &&
       (type != LIBXSMM_DNN_LSTM_INTERNAL_O)                && (type != LIBXSMM_DNN_LSTM_INTERNAL_O) ) {
    status = LIBXSMM_DNN_ERR_UNKNOWN_TENSOR_TYPE;
    return status;
  }

  if (handle != 0 && tensor != 0) {
    libxsmm_dnn_tensor_datalayout* handle_layout = libxsmm_dnn_lstmcell_create_tensor_datalayout(handle, type, &status);

    if ( libxsmm_dnn_compare_tensor_datalayout(handle_layout, tensor->layout, &status) == 0 ) {
      if ( type == LIBXSMM_DNN_LSTM_REGULAR_INPUT ) {
        handle->xt = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_LSTM_GRADIENT_INPUT ) {
        handle->dxt = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_LSTM_REGULAR_CS_PREV ) {
        handle->csp = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_LSTM_GRADIENT_CS_PREV ) {
        handle->dcsp = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_LSTM_REGULAR_HIDDEN_STATE_PREV ) {
        handle->hp = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_LSTM_GRADIENT_HIDDEN_STATE_PREV ) {
        handle->dhp = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_LSTM_REGULAR_WEIGHT ) {
        handle->w = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_LSTM_GRADIENT_WEIGHT ) {
        handle->dw = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_LSTM_REGULAR_BIAS ) {
        handle->b = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_LSTM_GRADIENT_BIAS ) {
        handle->db = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_LSTM_REGULAR_CS ) {
        handle->cst = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_LSTM_GRADIENT_CS ) {
        handle->dcst = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_LSTM_REGULAR_HIDDEN_STATE ) {
        handle->ht = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_LSTM_GRADIENT_HIDDEN_STATE ) {
        handle->dht = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_LSTM_INTERNAL_I ) {
        handle->it = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_LSTM_INTERNAL_F ) {
        handle->ft = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_LSTM_INTERNAL_O ) {
        handle->ot = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_LSTM_INTERNAL_C ) {
        handle->ct = (libxsmm_dnn_tensor*)tensor;
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


LIBXSMM_API libxsmm_dnn_tensor* libxsmm_dnn_lstmcell_get_tensor(libxsmm_dnn_lstmcell* handle, const libxsmm_dnn_tensor_type type, libxsmm_dnn_err_t* status)
{
  libxsmm_dnn_tensor* tensor = 0;
  LIBXSMM_UNUSED(status/*TODO*/);

  /* check for tensor type */
  if ( (type != LIBXSMM_DNN_LSTM_REGULAR_INPUT)             && (type != LIBXSMM_DNN_LSTM_GRADIENT_INPUT)             &&
       (type != LIBXSMM_DNN_LSTM_REGULAR_CS_PREV)           && (type != LIBXSMM_DNN_LSTM_GRADIENT_CS_PREV)           &&
       (type != LIBXSMM_DNN_LSTM_REGULAR_HIDDEN_STATE_PREV) && (type != LIBXSMM_DNN_LSTM_GRADIENT_HIDDEN_STATE_PREV) &&
       (type != LIBXSMM_DNN_LSTM_REGULAR_WEIGHT)            && (type != LIBXSMM_DNN_LSTM_GRADIENT_WEIGHT)            &&
       (type != LIBXSMM_DNN_LSTM_REGULAR_BIAS)              && (type != LIBXSMM_DNN_LSTM_GRADIENT_BIAS)              &&
       (type != LIBXSMM_DNN_LSTM_REGULAR_CS)                && (type != LIBXSMM_DNN_LSTM_GRADIENT_CS)                &&
       (type != LIBXSMM_DNN_LSTM_REGULAR_HIDDEN_STATE)      && (type != LIBXSMM_DNN_LSTM_GRADIENT_HIDDEN_STATE)      &&
       (type != LIBXSMM_DNN_LSTM_INTERNAL_I)                && (type != LIBXSMM_DNN_LSTM_INTERNAL_F)                 &&
       (type != LIBXSMM_DNN_LSTM_INTERNAL_O)                && (type != LIBXSMM_DNN_LSTM_INTERNAL_O) ) {
    return tensor;
  }

  if (handle != 0) {
    if ( type == LIBXSMM_DNN_LSTM_REGULAR_INPUT ) {
      tensor = handle->xt;
    } else if ( type == LIBXSMM_DNN_LSTM_GRADIENT_INPUT ) {
      tensor = handle->dxt;
    } else if ( type == LIBXSMM_DNN_LSTM_REGULAR_CS_PREV ) {
      tensor = handle->csp;
    } else if ( type == LIBXSMM_DNN_LSTM_GRADIENT_CS_PREV ) {
      tensor = handle->dcsp;
    } else if ( type == LIBXSMM_DNN_LSTM_REGULAR_HIDDEN_STATE_PREV ) {
      tensor = handle->hp;
    } else if ( type == LIBXSMM_DNN_LSTM_GRADIENT_HIDDEN_STATE_PREV ) {
      tensor = handle->dhp;
    } else if ( type == LIBXSMM_DNN_LSTM_REGULAR_WEIGHT ) {
      tensor = handle->w;
    } else if ( type == LIBXSMM_DNN_LSTM_GRADIENT_WEIGHT ) {
      tensor = handle->dw;
    } else if ( type == LIBXSMM_DNN_LSTM_REGULAR_BIAS ) {
      tensor = handle->b;
    } else if ( type == LIBXSMM_DNN_LSTM_GRADIENT_BIAS ) {
      tensor = handle->db;
    } else if ( type == LIBXSMM_DNN_LSTM_REGULAR_CS ) {
      tensor = handle->cst;
    } else if ( type == LIBXSMM_DNN_LSTM_GRADIENT_CS ) {
      tensor = handle->dcst;
    } else if ( type == LIBXSMM_DNN_LSTM_REGULAR_HIDDEN_STATE ) {
      tensor = handle->ht;
    } else if ( type == LIBXSMM_DNN_LSTM_GRADIENT_HIDDEN_STATE ) {
      tensor = handle->dht;
    } else if ( type == LIBXSMM_DNN_LSTM_INTERNAL_I ) {
      tensor = handle->it;
    } else if ( type == LIBXSMM_DNN_LSTM_INTERNAL_F ) {
      tensor = handle->ft;
    } else if ( type == LIBXSMM_DNN_LSTM_INTERNAL_O ) {
      tensor = handle->ot;
    } else if ( type == LIBXSMM_DNN_LSTM_INTERNAL_C ) {
      tensor = handle->ct;
    } else {
      /* cannot happen */
    }
  }

  return tensor;
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_lstmcell_release_tensor(libxsmm_dnn_lstmcell* handle, const libxsmm_dnn_tensor_type type)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  /* check for tensor type */
  if ( (type != LIBXSMM_DNN_LSTM_REGULAR_INPUT)             && (type != LIBXSMM_DNN_LSTM_GRADIENT_INPUT)             &&
       (type != LIBXSMM_DNN_LSTM_REGULAR_CS_PREV)           && (type != LIBXSMM_DNN_LSTM_GRADIENT_CS_PREV)           &&
       (type != LIBXSMM_DNN_LSTM_REGULAR_HIDDEN_STATE_PREV) && (type != LIBXSMM_DNN_LSTM_GRADIENT_HIDDEN_STATE_PREV) &&
       (type != LIBXSMM_DNN_LSTM_REGULAR_WEIGHT)            && (type != LIBXSMM_DNN_LSTM_GRADIENT_WEIGHT)            &&
       (type != LIBXSMM_DNN_LSTM_REGULAR_BIAS)              && (type != LIBXSMM_DNN_LSTM_GRADIENT_BIAS)              &&
       (type != LIBXSMM_DNN_LSTM_REGULAR_CS)                && (type != LIBXSMM_DNN_LSTM_GRADIENT_CS)                &&
       (type != LIBXSMM_DNN_LSTM_REGULAR_HIDDEN_STATE)      && (type != LIBXSMM_DNN_LSTM_GRADIENT_HIDDEN_STATE)      &&
       (type != LIBXSMM_DNN_LSTM_INTERNAL_I)                && (type != LIBXSMM_DNN_LSTM_INTERNAL_F)                 &&
       (type != LIBXSMM_DNN_LSTM_INTERNAL_O)                && (type != LIBXSMM_DNN_LSTM_INTERNAL_O) ) {
    status = LIBXSMM_DNN_ERR_UNKNOWN_TENSOR_TYPE;
    return status;
  }

  if (handle != 0) {
    if ( type == LIBXSMM_DNN_LSTM_REGULAR_INPUT ) {
      handle->xt = 0;
    } else if ( type == LIBXSMM_DNN_LSTM_GRADIENT_INPUT ) {
      handle->dxt = 0;
    } else if ( type == LIBXSMM_DNN_LSTM_REGULAR_CS_PREV ) {
      handle->csp = 0;
    } else if ( type == LIBXSMM_DNN_LSTM_GRADIENT_CS_PREV ) {
      handle->dcsp = 0;
    } else if ( type == LIBXSMM_DNN_LSTM_REGULAR_HIDDEN_STATE_PREV ) {
      handle->hp = 0;
    } else if ( type == LIBXSMM_DNN_LSTM_GRADIENT_HIDDEN_STATE_PREV ) {
      handle->dhp = 0;
    } else if ( type == LIBXSMM_DNN_LSTM_REGULAR_WEIGHT ) {
      handle->w = 0;
    } else if ( type == LIBXSMM_DNN_LSTM_GRADIENT_WEIGHT ) {
      handle->dw = 0;
    } else if ( type == LIBXSMM_DNN_LSTM_REGULAR_BIAS ) {
      handle->b = 0;
    } else if ( type == LIBXSMM_DNN_LSTM_GRADIENT_BIAS ) {
      handle->db = 0;
    } else if ( type == LIBXSMM_DNN_LSTM_REGULAR_CS ) {
      handle->cst = 0;
    } else if ( type == LIBXSMM_DNN_LSTM_GRADIENT_CS ) {
      handle->dcst = 0;
    } else if ( type == LIBXSMM_DNN_LSTM_REGULAR_HIDDEN_STATE ) {
      handle->ht = 0;
    } else if ( type == LIBXSMM_DNN_LSTM_GRADIENT_HIDDEN_STATE ) {
      handle->dht = 0;
    } else if ( type == LIBXSMM_DNN_LSTM_INTERNAL_I ) {
      handle->it = 0;
    } else if ( type == LIBXSMM_DNN_LSTM_INTERNAL_F ) {
      handle->ft = 0;
    } else if ( type == LIBXSMM_DNN_LSTM_INTERNAL_O ) {
      handle->ot = 0;
    } else if ( type == LIBXSMM_DNN_LSTM_INTERNAL_C ) {
      handle->ct = 0;
    } else {
      /* cannot happen */
    }
  }
  else {
    status = LIBXSMM_DNN_ERR_INVALID_HANDLE_TENSOR;
  }

  return status;
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_lstmcell_fwd(libxsmm_dnn_lstmcell* lstm, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
  libxsmm_blasint K = lstm->desc.K;
  libxsmm_blasint N = lstm->desc.N;
  libxsmm_blasint C = lstm->desc.C;
  libxsmm_blasint t = lstm->desc.t;
  libxsmm_blasint bk = lstm->bk;
  libxsmm_blasint bn = lstm->bn;
  libxsmm_blasint bc = lstm->bc;
  LIBXSMM_DNN_ELTWISE_FTYPE *xt  = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->xt->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *csp = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->csp->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *hpD = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->hp->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *w   = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->w->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *wiD = &(w[0]);
  LIBXSMM_DNN_ELTWISE_FTYPE *wcD = &(w[K*N]);
  LIBXSMM_DNN_ELTWISE_FTYPE *wfD = &(w[2*K*N]);
  LIBXSMM_DNN_ELTWISE_FTYPE *woD = &(w[3*K*N]);
  LIBXSMM_DNN_ELTWISE_FTYPE *riD = &(w[4*K*N]);
  LIBXSMM_DNN_ELTWISE_FTYPE *rcD = &(w[4*K*N + K*K]);
  LIBXSMM_DNN_ELTWISE_FTYPE *rfD = &(w[4*K*N + 2*K*K]);
  LIBXSMM_DNN_ELTWISE_FTYPE *roD = &(w[4*K*N + 3*K*K]);
  LIBXSMM_DNN_ELTWISE_FTYPE *b   = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->b->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *bi  = &(b[0]);
  LIBXSMM_DNN_ELTWISE_FTYPE *bd  = &(b[K]);
  LIBXSMM_DNN_ELTWISE_FTYPE *bf  = &(b[2*K]);
  LIBXSMM_DNN_ELTWISE_FTYPE *bo  = &(b[3*K]);
  LIBXSMM_DNN_ELTWISE_FTYPE *cst = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->cst->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *ht  = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->ht->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *it  = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->it->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *ft  = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->ft->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *ot  = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->ot->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *ct  = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->ct->data;
  LIBXSMM_VLA_DECL(3, LIBXSMM_DNN_ELTWISE_FTYPE, x, xt, N, C);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, cp, csp, K);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, hp, hpD, K);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, wi, wiD, K);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, wf, wfD, K);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, wo, woD, K);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, wc, wcD, K);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, ri, riD, K);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, rf, rfD, K);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, ro, roD, K);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, rc, rcD, K);
  LIBXSMM_VLA_DECL(3, LIBXSMM_DNN_ELTWISE_FTYPE, cs, cst, N, K);
  LIBXSMM_VLA_DECL(3, LIBXSMM_DNN_ELTWISE_FTYPE, h, ht, N, K);
  LIBXSMM_VLA_DECL(3, LIBXSMM_DNN_ELTWISE_FTYPE, i, it, N, K);
  LIBXSMM_VLA_DECL(3, LIBXSMM_DNN_ELTWISE_FTYPE, f, ft, N, K);
  LIBXSMM_VLA_DECL(3, LIBXSMM_DNN_ELTWISE_FTYPE, o, ot, N, K);
  LIBXSMM_VLA_DECL(3, LIBXSMM_DNN_ELTWISE_FTYPE, c, ct, N, K);
  libxsmm_blasint j, ik, in, ic;
  libxsmm_smmfunction gemmkernela = libxsmm_smmdispatch( bk, bn, bc, &K, &C, &K, NULL, NULL, NULL, NULL );
  libxsmm_smmfunction gemmkernelb = libxsmm_smmdispatch( bk, bn, bk, &K, &K, &K, NULL, NULL, NULL, NULL );
  LIBXSMM_UNUSED(tid); LIBXSMM_UNUSED(start_thread); /* TODO: remove */

  /* All data is in column-major format */
  for (j = 0; j < t; ++j) {
    /* let's run the cell in blocks for good locality */
    for (in = 0; in < N; in += bn) {
      for (ik = 0; ik < K; ik += bk) {
        /* initialize with bias */
        libxsmm_internal_matrix_bcst_colvector_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, i, j, in, ik, N, K), &bi[ik] );
        /* i += W.x */
        for (ic = 0; ic < C; ic += bc) {
          gemmkernela( &LIBXSMM_VLA_ACCESS(2, wi, ic, ik, K), &LIBXSMM_VLA_ACCESS(3, x, j, in, ic, N, C), &LIBXSMM_VLA_ACCESS(3, i, j, in, ik, N, K) );
        }
        /* i += U.h */
        for (ic = 0; ic < K; ic += bk) {
          gemmkernelb( &LIBXSMM_VLA_ACCESS(2, ri, ic, ik, K), &LIBXSMM_VLA_ACCESS(3, h, j, in, ic, N, K), &LIBXSMM_VLA_ACCESS(3, i, j, in, ik, N, K) );
        }
        libxsmm_internal_matrix_sigmoid_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, i, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, i, j, in, ik, N, K) );

        /* initialize with bias */
        libxsmm_internal_matrix_bcst_colvector_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, f, j, in, ik, N, K), &bf[ik] );
        /* f += W.x */
        for (ic = 0; ic < C; ic += bc) {
          gemmkernela( &LIBXSMM_VLA_ACCESS(2, wf, ic, ik, K), &LIBXSMM_VLA_ACCESS(3, x, j, in, ic, N, C), &LIBXSMM_VLA_ACCESS(3, f, j, in, ik, N, K) );
        }
        /* f += U.h */
        for (ic = 0; ic < K; ic += bk) {
          gemmkernelb( &LIBXSMM_VLA_ACCESS(2, rf, ic, ik, K), &LIBXSMM_VLA_ACCESS(3, h, j, in, ic, N, K), &LIBXSMM_VLA_ACCESS(3, f, j, in, ik, N, K) );
        }
        libxsmm_internal_matrix_sigmoid_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, f, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, f, j, in, ik, N, K) );

        /* initialize with bias */
        libxsmm_internal_matrix_bcst_colvector_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, o, j, in, ik, N, K), &bo[ik] );
        /* o += W.x */
        for (ic = 0; ic < C; ic += bc) {
          gemmkernela( &LIBXSMM_VLA_ACCESS(2, wo, ic, ik, K), &LIBXSMM_VLA_ACCESS(3, x, j, in, ic, N, C), &LIBXSMM_VLA_ACCESS(3, o, j, in, ik, N, K) );
        }
        /* o += U.h */
        for (ic = 0; ic < K; ic += bk) {
          gemmkernelb( &LIBXSMM_VLA_ACCESS(2, ro, ic, ik, K), &LIBXSMM_VLA_ACCESS(3, h, j, in, ic, N, K), &LIBXSMM_VLA_ACCESS(3, o, j, in, ik, N, K) );
        }
        libxsmm_internal_matrix_sigmoid_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, o, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, o, j, in, ik, N, K) );

        /* initialize with bias */
        libxsmm_internal_matrix_bcst_colvector_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, c, j, in, ik, N, K), &bd[ik] );
        /* c += W.x */
        for (ic = 0; ic < C; ic += bc) {
          gemmkernela( &LIBXSMM_VLA_ACCESS(2, wc, ic, ik, K), &LIBXSMM_VLA_ACCESS(3, x, j, in, ic, N, C), &LIBXSMM_VLA_ACCESS(3, c, j, in, ik, N, K) );
        }
        /* c += U.h */
        for (ic = 0; ic < K; ic += bk) {
          gemmkernelb( &LIBXSMM_VLA_ACCESS(2, rc, ic, ik, K), &LIBXSMM_VLA_ACCESS(3, h, j, in, ic, N, K), &LIBXSMM_VLA_ACCESS(3, c, j, in, ik, N, K) );
        }
        libxsmm_internal_matrix_tanh_ld(    bk, bn, K, &LIBXSMM_VLA_ACCESS(3, c, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, c, j, in, ik, N, K) );

        /* cs = f.cs */
        if (0 == j) {
          libxsmm_internal_matrix_eltwise_mult_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, f, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, cp, in, ik, K), &LIBXSMM_VLA_ACCESS(3, cs, j, in, ik, N, K) );
        } else {
          libxsmm_internal_matrix_eltwise_mult_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, f, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, cs, j-1, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, cs, j, in, ik, N, K) );
        }
        /* cs += i.c */
        libxsmm_internal_matrix_eltwise_fma_ld(  bk, bn, K, &LIBXSMM_VLA_ACCESS(3, i, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, c, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, cs, j, in, ik, N, K) );
        /* h = o.tanh(d) */
        if (0 == j) {
          libxsmm_internal_matrix_elt_mult_tanh_ld(  bk, bn, K, &LIBXSMM_VLA_ACCESS(3, o, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, cs, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, hp, in, ik, K) );
        } else {
          libxsmm_internal_matrix_elt_mult_tanh_ld(  bk, bn, K, &LIBXSMM_VLA_ACCESS(3, o, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, cs, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, h, j, in, ik, N, K) );
        }
      }
    }
  }

  return status;
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_lstmcell_bwd_upd_bu(libxsmm_dnn_lstmcell* lstm, int start_thread, int tid, int pass)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
  libxsmm_blasint K = lstm->desc.K;
  libxsmm_blasint N = lstm->desc.N;
  libxsmm_blasint C = lstm->desc.C;
  libxsmm_blasint t = lstm->desc.t;
  libxsmm_blasint bk = lstm->bk;
  libxsmm_blasint bn = lstm->bn;
  libxsmm_blasint bc = lstm->bc;
  int nThreads = lstm->desc.nThreads;
  LIBXSMM_DNN_ELTWISE_FTYPE *xt   = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->xt->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *csp  = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->csp->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *hpD  = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->hp->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *w    = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->w->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *wiD  = &(w[0]);
  LIBXSMM_DNN_ELTWISE_FTYPE *wcD  = &(w[K*N]);
  LIBXSMM_DNN_ELTWISE_FTYPE *wfD  = &(w[2*K*N]);
  LIBXSMM_DNN_ELTWISE_FTYPE *woD  = &(w[3*K*N]);
  LIBXSMM_DNN_ELTWISE_FTYPE *riD  = &(w[4*K*N]);
  LIBXSMM_DNN_ELTWISE_FTYPE *rcD  = &(w[4*K*N + K*K]);
  LIBXSMM_DNN_ELTWISE_FTYPE *rfD  = &(w[4*K*N + 2*K*K]);
  LIBXSMM_DNN_ELTWISE_FTYPE *roD  = &(w[4*K*N + 3*K*K]);
  LIBXSMM_DNN_ELTWISE_FTYPE *cst  = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->cst->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *ht   = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->ht->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *it   = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->it->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *ft   = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->ft->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *ot   = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->ot->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *ct   = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->ct->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *dxt  = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->dxt->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *dcsp = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->dcsp->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *dhpD = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->dhp->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *dw   = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->dw->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *dwiD = &(dw[0]);
  LIBXSMM_DNN_ELTWISE_FTYPE *dwcD = &(dw[K*N]);
  LIBXSMM_DNN_ELTWISE_FTYPE *dwfD = &(dw[2*K*N]);
  LIBXSMM_DNN_ELTWISE_FTYPE *dwoD = &(dw[3*K*N]);
  LIBXSMM_DNN_ELTWISE_FTYPE *driD = &(dw[4*K*N]);
  LIBXSMM_DNN_ELTWISE_FTYPE *drcD = &(dw[4*K*N + K*K]);
  LIBXSMM_DNN_ELTWISE_FTYPE *drfD = &(dw[4*K*N + 2*K*K]);
  LIBXSMM_DNN_ELTWISE_FTYPE *droD = &(dw[4*K*N + 3*K*K]);
  LIBXSMM_DNN_ELTWISE_FTYPE *db   = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->db->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *dbi  = &(db[0]);
  LIBXSMM_DNN_ELTWISE_FTYPE *dbc  = &(db[K]);
  LIBXSMM_DNN_ELTWISE_FTYPE *dbf  = &(db[2*K]);
  LIBXSMM_DNN_ELTWISE_FTYPE *dbo  = &(db[3*K]);
  LIBXSMM_DNN_ELTWISE_FTYPE *dcst = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->dcst->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *dht  = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->dht->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *dit  = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->dit->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *dft  = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->dft->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *dct  = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->dct->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *dot  = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->dot->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *deltat = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->deltat->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *doutt  = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->doutt->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *t1D = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->t1->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *t2D = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->t2->data;
  LIBXSMM_VLA_DECL(3, LIBXSMM_DNN_ELTWISE_FTYPE, x, xt, N, C);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, cp, csp, K);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, hp, hpD, K);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, wi, wiD, K);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, wf, wfD, K);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, wo, woD, K);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, wc, wcD, K);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, ri, riD, K);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, rf, rfD, K);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, ro, roD, K);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, rc, rcD, K);
  LIBXSMM_VLA_DECL(3, LIBXSMM_DNN_ELTWISE_FTYPE, cs, cst, N, K);
  LIBXSMM_VLA_DECL(3, LIBXSMM_DNN_ELTWISE_FTYPE, h, ht, N, K);
  LIBXSMM_VLA_DECL(3, LIBXSMM_DNN_ELTWISE_FTYPE, i, it, N, K);
  LIBXSMM_VLA_DECL(3, LIBXSMM_DNN_ELTWISE_FTYPE, f, ft, N, K);
  LIBXSMM_VLA_DECL(3, LIBXSMM_DNN_ELTWISE_FTYPE, o, ot, N, K);
  LIBXSMM_VLA_DECL(3, LIBXSMM_DNN_ELTWISE_FTYPE, c, ct, N, K);
  LIBXSMM_VLA_DECL(3, LIBXSMM_DNN_ELTWISE_FTYPE, dx, dxt, N, C);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, dcp, dcsp, K);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, dhp, dhpD, K);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, dwi, dwiD, K);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, dwf, dwfD, K);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, dwo, dwoD, K);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, dwc, dwcD, K);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, dri, driD, K);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, drf, drfD, K);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, dro, droD, K);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, drc, drcD, K);
  LIBXSMM_VLA_DECL(3, LIBXSMM_DNN_ELTWISE_FTYPE, dcs, dcst, N, K);
  LIBXSMM_VLA_DECL(3, LIBXSMM_DNN_ELTWISE_FTYPE, dh, dht, N, K);
  LIBXSMM_VLA_DECL(3, LIBXSMM_DNN_ELTWISE_FTYPE, di, dit, N, K);
  LIBXSMM_VLA_DECL(3, LIBXSMM_DNN_ELTWISE_FTYPE, df, dft, N, K);
  LIBXSMM_VLA_DECL(3, LIBXSMM_DNN_ELTWISE_FTYPE, dp, dot, N, K);
  LIBXSMM_VLA_DECL(3, LIBXSMM_DNN_ELTWISE_FTYPE, dc, dct, N, K);
  LIBXSMM_VLA_DECL(3, LIBXSMM_DNN_ELTWISE_FTYPE, delta, deltat, N, K);
  LIBXSMM_VLA_DECL(3, LIBXSMM_DNN_ELTWISE_FTYPE, dout, doutt, N, K);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, t1, t1D, K);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, t2, t2D, K);
  libxsmm_blasint j, ik, in, ic, jk, jn, jc, ek, en, ec;
  /* const int ltid = tid - start_thread; */

  /* initialization is done at the beginning */
  if (1 == pass || 3 == pass) {
    libxsmm_internal_matrix_zero(N*C*t, dxt,  start_thread, tid, nThreads);
  }
  if (2 == pass || 3 == pass) {
    libxsmm_internal_matrix_zero(C*K,   dwiD, start_thread, tid, nThreads);
    libxsmm_internal_matrix_zero(C*K,   dwfD, start_thread, tid, nThreads);
    libxsmm_internal_matrix_zero(C*K,   dwoD, start_thread, tid, nThreads);
    libxsmm_internal_matrix_zero(C*K,   dwcD, start_thread, tid, nThreads);
    libxsmm_internal_matrix_zero(K*K,   driD, start_thread, tid, nThreads);
    libxsmm_internal_matrix_zero(K*K,   drfD, start_thread, tid, nThreads);
    libxsmm_internal_matrix_zero(K*K,   droD, start_thread, tid, nThreads);
    libxsmm_internal_matrix_zero(K*K,   drcD, start_thread, tid, nThreads);
    libxsmm_internal_matrix_zero(K,     dbi,  start_thread, tid, nThreads);
    libxsmm_internal_matrix_zero(K,     dbf,  start_thread, tid, nThreads);
    libxsmm_internal_matrix_zero(K,     dbo,  start_thread, tid, nThreads);
    libxsmm_internal_matrix_zero(K,     dbc,  start_thread, tid, nThreads);
  }
  libxsmm_internal_matrix_zero(N*K*t, doutt,  start_thread, tid, nThreads);
  for (j = t-1; j >= 0; --j) {
    /* let's run the cell in blocks for good locality */
    for (in = 0; in < N; in += bn) {
      for (ik = 0; ik < K; ik += bk) {
        /* compute delta */
        if (j == t-1) {
          libxsmm_internal_matrix_copy( bk*bn, &LIBXSMM_VLA_ACCESS(3, dh, t-1, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, delta, t-1, in, ik, N, K), start_thread, tid, nThreads );
        } else {
          libxsmm_internal_matrix_add_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, dout, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, dh, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, delta, j, in, ik, N, K) );
        }
        /* compute dcs */
        libxsmm_internal_matrix_eltwise_mult_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, delta, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, o, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, t1, in, ik, K) );
        libxsmm_internal_matrix_tanh_inverse_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, cs, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, t2, in, ik, K) );
        if (j == t-1) {
          libxsmm_internal_matrix_eltwise_mult_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(2, t1, in, ik, K), &LIBXSMM_VLA_ACCESS(2, t2, in, ik, K), &LIBXSMM_VLA_ACCESS(3, dcs, j, in, ik, N, K) );
        } else {
          libxsmm_internal_matrix_eltwise_mult_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(2, t1, in, ik, K), &LIBXSMM_VLA_ACCESS(2, t2, in, ik, K), &LIBXSMM_VLA_ACCESS(3, dcs, j, in, ik, N, K) );
          libxsmm_internal_matrix_eltwise_fma_ld(  bk, bn, K, &LIBXSMM_VLA_ACCESS(3, delta, j+1, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, f, j+1, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, dcs, j, in, ik, N, K) );
        }
        /* compute dc */
        libxsmm_internal_matrix_eltwise_mult_ld(      bk, bn, K, &LIBXSMM_VLA_ACCESS(3, dcs, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, i, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, t1, in, ik, K) );
        libxsmm_internal_matrix_complement_square_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, c, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, t2, in, ik, K) );
        libxsmm_internal_matrix_eltwise_mult_ld(      bk, bn, K, &LIBXSMM_VLA_ACCESS(2, t1, in, ik, K), &LIBXSMM_VLA_ACCESS(2, t2, in, ik, K), &LIBXSMM_VLA_ACCESS(3, dc, j, in, ik, N, K) );
        /* compute di */
        libxsmm_internal_matrix_eltwise_mult_ld(      bk, bn, K, &LIBXSMM_VLA_ACCESS(3, dcs, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, c, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, t1, in, ik, K) );
        libxsmm_internal_matrix_complement_ld(        bk, bn, K, &LIBXSMM_VLA_ACCESS(3, i, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, t2, in, ik, K) );
        libxsmm_internal_matrix_eltwise_mult_ld(      bk, bn, K, &LIBXSMM_VLA_ACCESS(3, i, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, t2, in, ik, K), &LIBXSMM_VLA_ACCESS(3, di, j, in, ik, N, K) );
        libxsmm_internal_matrix_eltwise_mult_ld(      bk, bn, K, &LIBXSMM_VLA_ACCESS(2, t1, in, ik, K), &LIBXSMM_VLA_ACCESS(3, di, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, di, j, in, ik, N, K) );
        /* compute df */
        if (j >= 1) {
          libxsmm_internal_matrix_eltwise_mult_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, dcs, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, cs, j-1, in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, t1, in, ik, K) );
          libxsmm_internal_matrix_complement_ld(   bk, bn, K, &LIBXSMM_VLA_ACCESS(3, f, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, t2, in, ik, K) );
          libxsmm_internal_matrix_eltwise_mult_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, f, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, t2, in, ik, K), &LIBXSMM_VLA_ACCESS(3, df, j, in, ik, N, K) );
          libxsmm_internal_matrix_eltwise_mult_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(2, t1, in, ik, K), &LIBXSMM_VLA_ACCESS(3, df, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, df, j, in, ik, N, K) );
        } else {
          /* df is zero for j == 0 */
          libxsmm_internal_matrix_zero( bk*bn, &LIBXSMM_VLA_ACCESS(3, df, j, in, ik, N, K), start_thread, tid, nThreads );
        }
        /* compute do */
        libxsmm_internal_matrix_tanh_ld(         bk, bn, K, &LIBXSMM_VLA_ACCESS(3, cs, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, t1, in, ik, K) );
        libxsmm_internal_matrix_complement_ld(   bk, bn, K, &LIBXSMM_VLA_ACCESS(3, o, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, t2, in, ik, K) );
        libxsmm_internal_matrix_eltwise_mult_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, delta, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, t1, in, ik, K), &LIBXSMM_VLA_ACCESS(2, t1, in, ik, K) );
        libxsmm_internal_matrix_eltwise_mult_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, o, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, t2, in, ik, K), &LIBXSMM_VLA_ACCESS(2, t2, in, ik, K) );
        libxsmm_internal_matrix_eltwise_mult_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(2, t1, in, ik, K), &LIBXSMM_VLA_ACCESS(2, t2, in, ik, K), &LIBXSMM_VLA_ACCESS(3, dp, j, in, ik, N, K) );
        for (jn = 0; jn < bn; jn++) {
          for (jk = 0; jk < bk; jk++) {
            en = in + jn;
            ek = ik + jk;
            /* compute dout */
            if (j > 0) {
              for (ic = 0; ic < K; ic += bk) {
                for (jc = 0; jc < bk; jc++) {
                  ec = ic + jc;
                  LIBXSMM_VLA_ACCESS(3, dout, j-1, en, ec, N, K) += LIBXSMM_VLA_ACCESS(3, di, j, en, ek, N, K) * LIBXSMM_VLA_ACCESS(2, ri, ec, ek, K);
                  LIBXSMM_VLA_ACCESS(3, dout, j-1, en, ec, N, K) += LIBXSMM_VLA_ACCESS(3, df, j, en, ek, N, K) * LIBXSMM_VLA_ACCESS(2, rf, ec, ek, K);
                  LIBXSMM_VLA_ACCESS(3, dout, j-1, en, ec, N, K) += LIBXSMM_VLA_ACCESS(3, dp, j, en, ek, N, K) * LIBXSMM_VLA_ACCESS(2, ro, ec, ek, K);
                  LIBXSMM_VLA_ACCESS(3, dout, j-1, en, ec, N, K) += LIBXSMM_VLA_ACCESS(3, dc, j, en, ek, N, K) * LIBXSMM_VLA_ACCESS(2, rc, ec, ek, K);
                }
              }
            }
            /* compute dx */
            if (1 == pass || 3 == pass) {
              for (ic = 0; ic < C; ic += bc) {
                for (jc = 0; jc < bc; jc++) {
                  ec = ic + jc;
                  LIBXSMM_VLA_ACCESS(3, dx, j, en, ec, N, C) += LIBXSMM_VLA_ACCESS(3, di, j, en, ek, N, K) * LIBXSMM_VLA_ACCESS(2, wi, ec, ek, K);
                  LIBXSMM_VLA_ACCESS(3, dx, j, en, ec, N, C) += LIBXSMM_VLA_ACCESS(3, df, j, en, ek, N, K) * LIBXSMM_VLA_ACCESS(2, wf, ec, ek, K);
                  LIBXSMM_VLA_ACCESS(3, dx, j, en, ec, N, C) += LIBXSMM_VLA_ACCESS(3, dp, j, en, ek, N, K) * LIBXSMM_VLA_ACCESS(2, wo, ec, ek, K);
                  LIBXSMM_VLA_ACCESS(3, dx, j, en, ec, N, C) += LIBXSMM_VLA_ACCESS(3, dc, j, en, ek, N, K) * LIBXSMM_VLA_ACCESS(2, wc, ec, ek, K);
                }
              }
            }
            if (2 == pass || 3 == pass) {
              /* dr = delta * h^T */
              if (j > 0) {
                for (ic = 0; ic < K; ic += bk) {
                  for (jc = 0; jc < bk; jc++) {
                    ec = ic + jc;
                    LIBXSMM_VLA_ACCESS(2, dri, ec, ek, K) += LIBXSMM_VLA_ACCESS(3, h, j-1, en, ec, N, K) * LIBXSMM_VLA_ACCESS(3, di, j, en, ek, N, K);
                    LIBXSMM_VLA_ACCESS(2, drf, ec, ek, K) += LIBXSMM_VLA_ACCESS(3, h, j-1, en, ec, N, K) * LIBXSMM_VLA_ACCESS(3, df, j, en, ek, N, K);
                    LIBXSMM_VLA_ACCESS(2, dro, ec, ek, K) += LIBXSMM_VLA_ACCESS(3, h, j-1, en, ec, N, K) * LIBXSMM_VLA_ACCESS(3, dp, j, en, ek, N, K);
                    LIBXSMM_VLA_ACCESS(2, drc, ec, ek, K) += LIBXSMM_VLA_ACCESS(3, h, j-1, en, ec, N, K) * LIBXSMM_VLA_ACCESS(3, dc, j, en, ek, N, K);
                  }
                }
              }
              /* dw = delta * x^T */
              for (ic = 0; ic < C; ic += bc) {
                for (jc = 0; jc < bc; jc++) {
                  ec = ic + jc;
                  LIBXSMM_VLA_ACCESS(2, dwi, ec, ek, K) += LIBXSMM_VLA_ACCESS(3, x, j, en, ec, N, C) * LIBXSMM_VLA_ACCESS(3, di, j, en, ek, N, K);
                  LIBXSMM_VLA_ACCESS(2, dwf, ec, ek, K) += LIBXSMM_VLA_ACCESS(3, x, j, en, ec, N, C) * LIBXSMM_VLA_ACCESS(3, df, j, en, ek, N, K);
                  LIBXSMM_VLA_ACCESS(2, dwo, ec, ek, K) += LIBXSMM_VLA_ACCESS(3, x, j, en, ec, N, C) * LIBXSMM_VLA_ACCESS(3, dp, j, en, ek, N, K);
                  LIBXSMM_VLA_ACCESS(2, dwc, ec, ek, K) += LIBXSMM_VLA_ACCESS(3, x, j, en, ec, N, C) * LIBXSMM_VLA_ACCESS(3, dc, j, en, ek, N, K);
                }
              }
              if (j > 0) {
                dbi[ek] += LIBXSMM_VLA_ACCESS(3, di, j, en, ek, N, K);
                dbf[ek] += LIBXSMM_VLA_ACCESS(3, df, j, en, ek, N, K);
                dbo[ek] += LIBXSMM_VLA_ACCESS(3, dp, j, en, ek, N, K);
                dbc[ek] += LIBXSMM_VLA_ACCESS(3, dc, j, en, ek, N, K);
              }
            }
          }
        }
      }
    }
  }

  return status;
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_lstmcell_execute_st(libxsmm_dnn_lstmcell* handle, libxsmm_dnn_compute_kind kind,
  /*unsigned*/int start_thread, /*unsigned*/int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  if (0 != handle) {
    switch (kind) {
      case LIBXSMM_DNN_COMPUTE_KIND_FWD: {
                                           status = libxsmm_dnn_lstmcell_fwd(handle, start_thread, tid);
                                         } break;
      case LIBXSMM_DNN_COMPUTE_KIND_BWD: {
                                           status = libxsmm_dnn_lstmcell_bwd_upd_bu(handle, start_thread, tid, 1);
                                         } break;
      case LIBXSMM_DNN_COMPUTE_KIND_UPD: {
                                           status = libxsmm_dnn_lstmcell_bwd_upd_bu(handle, start_thread, tid, 2);
                                         } break;
      case LIBXSMM_DNN_COMPUTE_KIND_ALL: {
                                           status = libxsmm_dnn_lstmcell_bwd_upd_bu(handle, start_thread, tid, 3);
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

