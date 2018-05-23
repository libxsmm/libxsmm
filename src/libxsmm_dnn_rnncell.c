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
/* Alexander Heinecke, Kunal Banerjee (Intel Corp.)
******************************************************************************/

#include <libxsmm.h>
#include <math.h>
#include "libxsmm_main.h"

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#include <string.h>
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

#if defined(LSTM_TIMING)
#include <stdio.h>
double Gbl_t_input_total = 0., Gbl_t_recur_total = 0., Gbl_t_eltwise_total = 0., Gbl_t_nonlin_total = 0.;
unsigned long long Gbl_t_input = 0, Gbl_t_recur = 0, Gbl_t_eltwise = 0, Gbl_t_nonlin = 0;
double Gbl_duration_input = 0., Gbl_duration_recur = 0., Gbl_duration_eltwise = 0., Gbl_duration_nonlin = 0.;
#endif


LIBXSMM_API libxsmm_dnn_rnncell* libxsmm_dnn_create_rnncell(libxsmm_dnn_rnncell_desc rnncell_desc, libxsmm_dnn_err_t* status) {
  libxsmm_dnn_rnncell* handle = 0;
  *status = LIBXSMM_DNN_SUCCESS;

  handle = (libxsmm_dnn_rnncell*)malloc(sizeof(libxsmm_dnn_rnncell));
  if (0 != handle) {
    /* zero entire content; not only safer but also sets data and code pointers to NULL */
    memset(handle, 0, sizeof(*handle));
    /* initialize known handle components */
    handle->desc = rnncell_desc;
    handle->datatype_in = rnncell_desc.datatype_in;
    handle->datatype_out = rnncell_desc.datatype_out;
    if ( (rnncell_desc.datatype_in != LIBXSMM_DNN_DATATYPE_F32) || (rnncell_desc.datatype_out != LIBXSMM_DNN_DATATYPE_F32) ) {
      /* error */
      *status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
      return handle;
    }
    handle->buffer_format = rnncell_desc.buffer_format;
    handle->m = rnncell_desc.m;
    handle->n = rnncell_desc.n;
    handle->k = rnncell_desc.k;
    handle->t = rnncell_desc.t;
    if (rnncell_desc.t < 2) {
      *status = LIBXSMM_DNN_ERR_TIME_STEPS_TOO_SMALL;
    }
    handle->bm = rnncell_desc.bm;
    handle->bn = rnncell_desc.bn;
    handle->bk = rnncell_desc.bk;
    handle->b_m1 = rnncell_desc.b_m1;
    handle->b_n1 = rnncell_desc.b_n1;
    handle->b_k1 = rnncell_desc.b_k1;
    handle->b_m2 = rnncell_desc.b_m2;
    handle->b_n2 = rnncell_desc.b_n2;
    handle->b_k2 = rnncell_desc.b_k2;
  } else {
    *status = LIBXSMM_DNN_ERR_CREATE_HANDLE;
  }
  return handle;
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_destroy_rnncell(const libxsmm_dnn_rnncell* handle) {
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
  if (0 != handle) {
    /* deallocate handle structure */
    free(/*remove constness*/(libxsmm_dnn_rnncell*)handle);
  }
  return status;
}


LIBXSMM_API libxsmm_dnn_tensor_datalayout* libxsmm_dnn_rnncell_create_tensor_datalayout(const libxsmm_dnn_rnncell* handle, const libxsmm_dnn_tensor_type type, libxsmm_dnn_err_t* status) {
  libxsmm_dnn_tensor_datalayout* layout = 0;
  *status = LIBXSMM_DNN_SUCCESS;
  layout = 0;
  if (handle != 0) {
    layout = (libxsmm_dnn_tensor_datalayout*) malloc(sizeof(libxsmm_dnn_tensor_datalayout));
    if (layout != 0) {
      memset(layout, 0, sizeof(libxsmm_dnn_tensor_datalayout));
      /*layout->custom_format = handle->custom_format_type;*/
      if ( (type == LIBXSMM_DNN_REGULAR_INPUT)  || (type == LIBXSMM_DNN_GRADIENT_INPUT)  || (type == LIBXSMM_DNN_INPUT)  ||
           (type == LIBXSMM_DNN_REGULAR_OUTPUT) || (type == LIBXSMM_DNN_GRADIENT_OUTPUT) || (type == LIBXSMM_DNN_OUTPUT)    ) {
        layout->format = handle->buffer_format;
        layout->tensor_type = LIBXSMM_DNN_ACTIVATION;

        if ((handle->buffer_format & LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM) > 0) {
          if ( ((handle->datatype_in == LIBXSMM_DNN_DATATYPE_F32) && (handle->datatype_out == LIBXSMM_DNN_DATATYPE_F32) ) ) {
            layout->datatype = LIBXSMM_DNN_DATATYPE_F32;
            if (1 /*handle->custom_format_type == LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM_1*/) {
              layout->dim_type = (libxsmm_dnn_tensor_dimtype*) malloc(4*sizeof(libxsmm_dnn_tensor_dimtype));
              layout->dim_size = (unsigned int*) malloc(4*sizeof(unsigned int));

              if (0 != layout->dim_type && 0 != layout->dim_size) { /* TODO: handle the error */
                layout->num_dims = 4;
                layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
                layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
                layout->dim_type[2] = LIBXSMM_DNN_TENSOR_DIMTYPE_W;
                layout->dim_type[3] = LIBXSMM_DNN_TENSOR_DIMTYPE_H;
                if ( (type == LIBXSMM_DNN_REGULAR_INPUT) || (type == LIBXSMM_DNN_GRADIENT_INPUT) || (type == LIBXSMM_DNN_INPUT) ) {
                  layout->dim_size[0] = handle->bm;
                  layout->dim_size[1] = handle->bk;
                  layout->dim_size[2] = handle->k / handle->bk;
                  layout->dim_size[3] = handle->m / handle->bm;
                } else if ( (type == LIBXSMM_DNN_REGULAR_OUTPUT) || (type == LIBXSMM_DNN_GRADIENT_OUTPUT) || (type == LIBXSMM_DNN_OUTPUT) ) {
                  layout->dim_size[0] = handle->bm;
                  layout->dim_size[1] = handle->bn;
                  layout->dim_size[2] = handle->m / handle->bm;
                  layout->dim_size[3] = handle->n / handle->bn;
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


LIBXSMM_API size_t libxsmm_dnn_rnncell_get_scratch_size(const libxsmm_dnn_rnncell* handle, const libxsmm_dnn_compute_kind kind, libxsmm_dnn_err_t* status) {
  size_t sizeof_datatype = sizeof(float);
  size_t size = 0;
  *status = LIBXSMM_DNN_SUCCESS;

  if (0 != handle) {
    switch (kind) {
      case LIBXSMM_DNN_COMPUTE_KIND_FWD: {
                                           size += handle->m * handle->n * sizeof_datatype * handle->t; /* z1t */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype; /* z2 */
                                           size += 64;
                                         } break;
      case LIBXSMM_DNN_COMPUTE_KIND_BWD:
      case LIBXSMM_DNN_COMPUTE_KIND_UPD:
      case LIBXSMM_DNN_COMPUTE_KIND_ALL: {
                                           size += handle->m * handle->n * sizeof_datatype * handle->t; /* z1t */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype; /* z2i, zi */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype * handle->t; /* deltat */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype; /* di1 */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype; /* di2 */
                                           size += 64;
                                           size += handle->m * handle->m * sizeof_datatype; /* dj1 */
                                           size += 64;
                                           size += handle->m * handle->k * sizeof_datatype; /* dw1 */
                                           size += 64;
                                           size += handle->m * handle->m * sizeof_datatype; /* uTp */
                                           size += 64;
                                           size += handle->m * handle->k * sizeof_datatype; /* wTp */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype; /* hTp */
                                           size += 64;
                                           size += handle->k * handle->n * sizeof_datatype; /* xTp */
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


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_rnncell_bind_scratch(libxsmm_dnn_rnncell* handle, const libxsmm_dnn_compute_kind kind, const void* scratch) {
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
  size_t address = (size_t)scratch;
  size_t offset = 0;
  size_t scratch_size = 0;
  size_t sizeof_datatype = sizeof(float);

  if (scratch == 0) {
    status = LIBXSMM_DNN_ERR_SCRATCH_NOT_ALLOCED;
    return status;
  }

  if (0 != handle) {
    switch (kind) {
      case LIBXSMM_DNN_COMPUTE_KIND_FWD: {
                                           if (address % 64 == 0) {
                                             handle->z1t->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->z1t->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype * handle->t;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->z2->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->z2->data = (void*)(address+offset);
                                           }
                                         } break;
      case LIBXSMM_DNN_COMPUTE_KIND_BWD:
      case LIBXSMM_DNN_COMPUTE_KIND_UPD:
      case LIBXSMM_DNN_COMPUTE_KIND_ALL: {
                                           if (address % 64 == 0) {
                                             handle->z1t->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->z1t->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype * handle->t;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->z2->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->z2->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->deltat->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->deltat->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype * handle->t;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->di1->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->di1->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->di2->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->di2->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->dj1->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->dj1->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->m * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->dw1->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->dw1->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->k * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->uTp->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->uTp->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->m * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->wTp->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->wTp->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->k * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->hTp->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->hTp->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->xTp->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->xTp->data = (void*)(address+offset);
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


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_rnncell_release_scratch(libxsmm_dnn_rnncell* handle, const libxsmm_dnn_compute_kind kind) {
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  if (0 != handle) {
    switch (kind) {
      case LIBXSMM_DNN_COMPUTE_KIND_FWD: {
                                           handle->z1t->data = 0;
                                           handle->z2->data = 0;
                                           handle->z1t = 0;
                                           handle->z2 = 0;
                                         } break;
      case LIBXSMM_DNN_COMPUTE_KIND_BWD:
      case LIBXSMM_DNN_COMPUTE_KIND_UPD:
      case LIBXSMM_DNN_COMPUTE_KIND_ALL: {
                                           handle->z1t->data = 0;
                                           handle->z2->data = 0;
                                           handle->deltat->data = 0;
                                           handle->di1->data = 0;
                                           handle->di2->data = 0;
                                           handle->dj1->data = 0;
                                           handle->dw1->data = 0;
                                           handle->uTp->data = 0;
                                           handle->wTp->data = 0;
                                           handle->hTp->data = 0;
                                           handle->xTp->data = 0;
                                           handle->z1t = 0;
                                           handle->z2 = 0;
                                           handle->deltat = 0;
                                           handle->di1 = 0;
                                           handle->di2 = 0;
                                           handle->dj1 = 0;
                                           handle->dw1 = 0;
                                           handle->uTp = 0;
                                           handle->wTp = 0;
                                           handle->hTp = 0;
                                           handle->xTp = 0;
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


LIBXSMM_API size_t libxsmm_dnn_rnncell_get_internalstate_size(const libxsmm_dnn_rnncell* handle, const libxsmm_dnn_compute_kind kind, libxsmm_dnn_err_t* status) {
  size_t sizeof_datatype = sizeof(float);
  size_t size = 0;
  *status = LIBXSMM_DNN_SUCCESS;

  if (0 != handle) {
    switch (kind) {
      case LIBXSMM_DNN_COMPUTE_KIND_FWD: {
                                           size += handle->m * handle->k * sizeof_datatype; /* w */
                                           size += 64;
                                           size += handle->k * handle->n * sizeof_datatype * handle->t; /* xt */
                                           size += 64;
                                           size += handle->m * handle->m * sizeof_datatype; /* u */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype; /* h */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype; /* z */
                                           size += 64;
                                         } break;
      case LIBXSMM_DNN_COMPUTE_KIND_BWD:
      case LIBXSMM_DNN_COMPUTE_KIND_UPD:
      case LIBXSMM_DNN_COMPUTE_KIND_ALL: {
                                           size += handle->m * handle->n * sizeof_datatype * handle->t; /* zt */
                                           size += 64;
                                           size += handle->m * handle->m * sizeof_datatype; /* u */
                                           size += 64;
                                           size += handle->k * handle->n * sizeof_datatype * handle->t; /* xt */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype * handle->t; /* ht */
                                           size += 64;
                                           size += handle->m * handle->m * sizeof_datatype; /* djdu */
                                           size += 64;
                                           size += handle->m * handle->k * sizeof_datatype; /* djdw */
                                           size += 64;
                                           size += handle->k * handle->n * sizeof_datatype * handle->t; /* djdxt */
                                           size += 64;
                                           size += handle->m * handle->k * sizeof_datatype; /* w */
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


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_rnncell_bind_internalstate(libxsmm_dnn_rnncell* handle, const libxsmm_dnn_compute_kind kind, const void* internalstate) {
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
  size_t address = (size_t)internalstate;
  size_t offset = 0;
  size_t scratch_size = 0;
  size_t sizeof_datatype = sizeof(float);

  if (internalstate == 0) {
    status = LIBXSMM_DNN_ERR_SCRATCH_NOT_ALLOCED;
    return status;
  }

  if (0 != handle) {
    switch (kind) {
      case LIBXSMM_DNN_COMPUTE_KIND_FWD: {
                                           if (address % 64 == 0) {
                                             handle->w->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->w->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->k * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->xt->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->xt->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->k * handle->n * sizeof_datatype * handle->t;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->u->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->u->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->m * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->h->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->h->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->z->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->z->data = (void*)(address+offset);
                                           }
                                         } break;
      case LIBXSMM_DNN_COMPUTE_KIND_BWD:
      case LIBXSMM_DNN_COMPUTE_KIND_UPD:
      case LIBXSMM_DNN_COMPUTE_KIND_ALL: {
                                           if (address % 64 == 0) {
                                             handle->z->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->z->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype * handle->t;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->u->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->u->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->m * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->xt->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->xt->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->k * handle->n * sizeof_datatype * handle->t;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->h->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->h->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype * handle->t;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->djdu->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->djdu->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->m * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->djdw->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->djdw->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->k * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->djdxt->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->djdxt->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->k * handle->n * sizeof_datatype * handle->t;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->w->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->w->data = (void*)(address+offset);
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


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_rnncell_release_internalstate(libxsmm_dnn_rnncell* handle, const libxsmm_dnn_compute_kind kind) {
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  if (0 != handle) {
    switch (kind) {
      case LIBXSMM_DNN_COMPUTE_KIND_FWD: {
                                           handle->w->data = 0;
                                           handle->xt->data = 0;
                                           handle->u->data = 0;
                                           handle->h->data = 0;
                                           handle->z->data = 0;
                                           handle->w = 0;
                                           handle->xt = 0;
                                           handle->u = 0;
                                           handle->h = 0;
                                           handle->z = 0;
                                         } break;
      case LIBXSMM_DNN_COMPUTE_KIND_BWD:
      case LIBXSMM_DNN_COMPUTE_KIND_UPD:
      case LIBXSMM_DNN_COMPUTE_KIND_ALL: {
                                           handle->z->data = 0;
                                           handle->u->data = 0;
                                           handle->xt->data = 0;
                                           handle->h->data = 0;
                                           handle->djdu->data = 0;
                                           handle->djdw->data = 0;
                                           handle->djdxt->data = 0;
                                           handle->w->data = 0;
                                           handle->z = 0;
                                           handle->u = 0;
                                           handle->xt = 0;
                                           handle->h = 0;
                                           handle->djdu = 0;
                                           handle->djdw = 0;
                                           handle->djdxt = 0;
                                           handle->w = 0;
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


/* TODO: May be we don't need the followuing three functions */
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_rnncell_bind_tensor(libxsmm_dnn_rnncell* handle, const libxsmm_dnn_tensor* tensor, const libxsmm_dnn_tensor_type type) {
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  if (handle != 0 && tensor != 0) {
    libxsmm_dnn_tensor_datalayout* handle_layout = libxsmm_dnn_rnncell_create_tensor_datalayout(handle, type, &status);
    if ( libxsmm_dnn_compare_tensor_datalayout(handle_layout, tensor->layout, &status) == 0 ) {
      /* Need to populate this code */
    } else {
      status = LIBXSMM_DNN_ERR_MISMATCH_TENSOR;
    }
    libxsmm_dnn_destroy_tensor_datalayout( handle_layout );
  } else {
    status = LIBXSMM_DNN_ERR_INVALID_HANDLE_TENSOR;
  }

  return status;
}


LIBXSMM_API libxsmm_dnn_tensor* libxsmm_dnn_rnncell_get_tensor(libxsmm_dnn_rnncell* handle, const libxsmm_dnn_tensor_type type, libxsmm_dnn_err_t* status) {
  libxsmm_dnn_tensor* tensor = 0;

  LIBXSMM_UNUSED( handle );
  LIBXSMM_UNUSED( type );
  *status = LIBXSMM_DNN_SUCCESS;

  return tensor;
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_rnncell_release_tensor(libxsmm_dnn_rnncell* handle, const libxsmm_dnn_tensor_type type) {
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  if (handle != 0) {
    LIBXSMM_UNUSED(type/* Need to populate this code */);
  } else {
    status = LIBXSMM_DNN_ERR_INVALID_HANDLE_TENSOR;
  }

  return status;
}


# define FTYPE float /* TODO: undefine/remove generic symbol names as header-only interfers with user's code */
/* TODO: avoid generic function names without prefix e.g., libxsmm_internal_ or libxsmm_ */
LIBXSMM_API_INLINE void matinit(int seed, FTYPE * dst,
  libxsmm_blasint nrows, libxsmm_blasint ncols, libxsmm_blasint ld, double scale)
{
  const double seed1 = scale * (seed + 1);
  libxsmm_blasint i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < ncols; ++i) {
    libxsmm_blasint j = 0;
    for (; j < nrows; ++j) {
      const libxsmm_blasint k = i * ld + j;
      dst[k] = (FTYPE)(seed1 / (k + 1));
    }
    for (; j < ld; ++j) {
      const libxsmm_blasint k = i * ld + j;
      dst[k] = (FTYPE)seed;
    }
  }
}


LIBXSMM_API_INLINE void matrix_add(libxsmm_blasint size, FTYPE *a, FTYPE *b, FTYPE *c)
{
  libxsmm_blasint i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < size; i++) {
    c[i] = a[i] + b[i];
  }
}


LIBXSMM_API_INLINE void matrix_eltwise_mult(libxsmm_blasint size, FTYPE *a, FTYPE *b, FTYPE *c)
{
  libxsmm_blasint i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < size; i++) {
    c[i] = a[i] * b[i];
  }
}


LIBXSMM_API_INLINE void matrix_sigmoid(libxsmm_blasint size, FTYPE *src, FTYPE *dst)
{
  libxsmm_blasint i;
  FTYPE exp_value;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < size; i++) {
    exp_value = (FTYPE)exp((double) -src[i]);
    dst[i] = 1 / (1 + exp_value);
  }
}


LIBXSMM_API_INLINE void matrix_tanh(libxsmm_blasint size, FTYPE *src, FTYPE *dst)
{
  libxsmm_blasint i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < size; i++) {
    dst[i] = (FTYPE)tanh((double)src[i]);
  }
}


LIBXSMM_API_INLINE void matrix_relu(libxsmm_blasint size, FTYPE *src, FTYPE *dst)
{
  libxsmm_blasint i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < size; i++) {
    dst[i] = (src[i] >= 0) ? src[i] : 0;
  }
}


LIBXSMM_API_INLINE void matrix_sigmoid_inverse(libxsmm_blasint size, FTYPE *src, FTYPE *dst)
{
  libxsmm_blasint i;
  FTYPE exp_value;
  FTYPE sig_exp;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < size; i++) {
    exp_value = (FTYPE)exp((double) -src[i]);
    sig_exp = 1 / (1 + exp_value);
    dst[i] = (1 - sig_exp)*sig_exp;
  }
}


LIBXSMM_API_INLINE void matrix_tanh_inverse(libxsmm_blasint size, FTYPE *src, FTYPE *dst)
{
  libxsmm_blasint i;
  FTYPE tanh_value;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < size; i++) {
    tanh_value = (FTYPE)tanh((double)src[i]);
    dst[i] = 1 - (tanh_value * tanh_value);
  }
}


LIBXSMM_API_INLINE void matrix_relu_inverse(libxsmm_blasint size, FTYPE *src, FTYPE *dst, FTYPE *input)
{
  libxsmm_blasint i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < size; i++) {
    dst[i] = (input[i] >= 0) ? src[i] : 0;
  }
}


LIBXSMM_API_INLINE void matrix_transpose(libxsmm_blasint rows, libxsmm_blasint cols, FTYPE *src, FTYPE *dst)
{
  libxsmm_blasint i, j;
  LIBXSMM_VLA_DECL(2, FTYPE, src2D, src, cols);
  LIBXSMM_VLA_DECL(2, FTYPE, dst2D, dst, rows);
#if defined(_OPENMP)
# pragma omp parallel for private(i, j) LIBXSMM_OPENMP_COLLAPSE(2)
#endif
  for (i = 0; i < rows; i++) {
    for (j = 0; j < cols; j++) {
      LIBXSMM_VLA_ACCESS(2, dst2D, j, i, rows) = LIBXSMM_VLA_ACCESS(2, src2D, i, j, cols);
    }
  }
}


LIBXSMM_API_INLINE void matrix_copy(libxsmm_blasint size, FTYPE *src, FTYPE *dst)
{
  libxsmm_blasint i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < size; i++) {
    dst[i] = src[i];
  }
}


LIBXSMM_API_INLINE void matrix_complement(libxsmm_blasint size, FTYPE *src, FTYPE *dst)
{
  libxsmm_blasint i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < size; i++) {
    dst[i] = 1 - src[i];
  }
}


LIBXSMM_API_INLINE void matrix_complement_square(libxsmm_blasint size, FTYPE *src, FTYPE *dst)
{
  libxsmm_blasint i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < size; i++) {
    dst[i] = 1 - (src[i] * src[i]);
  }
}


LIBXSMM_API_INLINE void recursive_step(libxsmm_bgemm_handle* handle, FTYPE* u, FTYPE* h, FTYPE* op1, FTYPE *op2,
  FTYPE *temp, FTYPE *dst, int act, libxsmm_blasint size, int tid, int nthreads)
{
#if defined(LSTM_TIMING)
  Gbl_t_recur = libxsmm_timer_tick();
#endif
  libxsmm_bgemm(handle, u, h, op1, tid, nthreads);
#if defined(LSTM_TIMING)
  Gbl_duration_recur = libxsmm_timer_duration(Gbl_t_recur, libxsmm_timer_tick());
  Gbl_t_recur_total += Gbl_duration_recur;
  Gbl_t_eltwise = libxsmm_timer_tick();
#endif
  matrix_add(size, op1, op2, temp);
#if defined(LSTM_TIMING)
  Gbl_duration_eltwise = libxsmm_timer_duration(Gbl_t_eltwise, libxsmm_timer_tick());
  Gbl_t_eltwise_total += Gbl_duration_eltwise;
  Gbl_t_nonlin = libxsmm_timer_tick();
#endif
  switch (act) {
    case 0:
      /* do nothing -- this is required for the last time step */
      dst = temp;
      break;
    case 1:
      matrix_relu(size, temp, dst);
      break;
    case 2:
      matrix_sigmoid(size, temp, dst);
      break;
    case 3:
      matrix_tanh(size, temp, dst);
      break;
    default:
      /* fprintf(stdout, "Unsupported activation function: %d\n", act); */
      dst = temp;
  }
#if defined(LSTM_TIMING)
  Gbl_duration_nonlin = libxsmm_timer_duration(Gbl_t_nonlin, libxsmm_timer_tick());
  Gbl_t_nonlin_total += Gbl_duration_nonlin;
#endif
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_rnncell_fwd(libxsmm_dnn_rnncell* rnn, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
  libxsmm_blasint m = rnn->m;
  libxsmm_blasint n = rnn->n;
  libxsmm_blasint k = rnn->k;
  libxsmm_blasint t = rnn->t;
#if defined(LSTM_TIMING)
  const double gflops = ((2.0 * m * n * k) + (2.0 * m * n * m) + (2.0 * m * n)) * t * 1E-9;
#endif
  int reuse = 1;
  /* The following code should be in template */
  FTYPE *w = (FTYPE*)rnn->w->data;
  FTYPE *xt = (FTYPE*)rnn->xt->data;
  FTYPE *u = (FTYPE*)rnn->u->data;
  FTYPE *h = (FTYPE*)rnn->h->data;
  FTYPE *z1t = (FTYPE*)rnn->z1t->data;
  FTYPE *z2 = (FTYPE*)rnn->z2->data;
  FTYPE *z = (FTYPE*)rnn->z->data;
  /* libxsmm_bgemm_handle *handlewx = rnn->handlewx; */
  libxsmm_bgemm_handle *handleuh = rnn->handleuh;
  libxsmm_bgemm_handle *handlett = rnn->handlett;
  LIBXSMM_VLA_DECL(2, FTYPE, x, xt, k * n);
  LIBXSMM_VLA_DECL(2, FTYPE, z1, z1t, m * n);
  LIBXSMM_VLA_DECL(2, FTYPE, hnr, h, m * n);
  LIBXSMM_VLA_DECL(2, FTYPE, znr, z, m * n);
#if defined(LSTM_TIMING)
  unsigned long long start;
  double duration;
  Gbl_t_input_total = 0.; Gbl_t_recur_total = 0.; Gbl_t_eltwise_total = 0.; Gbl_t_nonlin_total = 0.;
  Gbl_t_input = 0; Gbl_t_recur = 0; Gbl_t_eltwise = 0; Gbl_t_nonlin = 0;
  Gbl_duration_input = 0.; Gbl_duration_recur = 0.; Gbl_duration_eltwise = 0.; Gbl_duration_nonlin = 0.;
#endif
  /* int s; */
  int i;

  LIBXSMM_UNUSED(start_thread/* Need to populate this code */);
#if defined(LSTM_TIMING)
  start = libxsmm_timer_tick();
#endif
  /* for (s = 0; s < nrepeat; ++s) { */
#if defined(LSTM_TIMING)
    Gbl_t_input = libxsmm_timer_tick();
#endif
    /* The following loop may be absorbed into libxsmm_lstm_omp */
    libxsmm_bgemm(handlett, w, &LIBXSMM_VLA_ACCESS(2, x, 0, 0, k * n), &LIBXSMM_VLA_ACCESS(2, z1, 0, 0, m * n), tid, rnn->nThreads);
#if defined(LSTM_TIMING)
    Gbl_duration_input = libxsmm_timer_duration(Gbl_t_input, libxsmm_timer_tick());
    Gbl_t_input_total += Gbl_duration_input;
#endif
    if (reuse) {
      for (i = 0; i < t-1; ++i) {
        recursive_step(handleuh, u, h, z2, &LIBXSMM_VLA_ACCESS(2, z1, i, 0, m * n), z, h, 1, m * n, tid, rnn->nThreads); /*sigmoid*/
      }
      recursive_step(handleuh, u, h, z2, &LIBXSMM_VLA_ACCESS(2, z1, t-1, 0, m * n), z, z, 0, m * n, tid, rnn->nThreads); /*nop*/
    } else {
      for (i = 0; i < t-1; ++i) {
        recursive_step(handleuh, u, &LIBXSMM_VLA_ACCESS(2, hnr, i, 0, m * n), z2, &LIBXSMM_VLA_ACCESS(2, z1, i, 0, m * n),
          &LIBXSMM_VLA_ACCESS(2, znr, i, 0, m * n), &LIBXSMM_VLA_ACCESS(2, hnr, i+1, 0, m * n), 1, m * n, tid, rnn->nThreads); /*sigmoid*/
      }
      recursive_step(handleuh, u, &LIBXSMM_VLA_ACCESS(2, hnr, t-1, 0, m * n), z2, &LIBXSMM_VLA_ACCESS(2, z1, t-1, 0, m * n),
        &LIBXSMM_VLA_ACCESS(2, znr, t-1, 0, m * n), &LIBXSMM_VLA_ACCESS(2, znr, t-1, 0, m * n), 0, m * n, tid, rnn->nThreads); /*nop*/
    }
  /* } */
#if defined(LSTM_TIMING)
  duration = libxsmm_timer_duration(start, libxsmm_timer_tick());
  if (0 < duration) {
    fprintf(stdout, "\tLIBXSMM: %.1f GFLOPS/s\n", gflops / duration); /* *nrepeat */
  }
  double t_total = Gbl_t_input_total + Gbl_t_recur_total + Gbl_t_eltwise_total + Gbl_t_nonlin_total;
  fprintf(stdout, "Percentage of time spent in input matrix multiplication: %lf\n", Gbl_t_input_total*100.0/t_total);
  fprintf(stdout, "Percentage of time spent in recurrence matrix multiplication: %lf\n", Gbl_t_recur_total*100.0/t_total);
  fprintf(stdout, "Percentage of time spent in element-wise operations: %lf\n", Gbl_t_eltwise_total*100.0/t_total);
  fprintf(stdout, "Percentage of time spent in non-linear operations: %lf\n", Gbl_t_nonlin_total*100.0/t_total);
#endif
  return status;
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_rnncell_bwd_upd_bu(libxsmm_dnn_rnncell* rnn, int start_thread, int tid, int pass)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
  libxsmm_blasint m = rnn->m;
  libxsmm_blasint n = rnn->n;
  libxsmm_blasint k = rnn->k;
  libxsmm_blasint t = rnn->t;
#ifdef LSTM_TIMING
  const double tflops = 12; /* transcendental flops */
  double gflops = m * m; /* U^T */
  gflops += (2.0 * m * n * m); /* U^T * delta */
  gflops += (m * n); /* dJdh + (U^T * delta) */
  gflops += (tflops * m * n); /* sigma'(Z) */
  gflops += (m * n); /* sigma'(Z) * (dJdh + (U^T * delta)) */
  gflops *= t; /* for t time steps */
  double tempflops;
  if (pass == 2 || pass == 3) {
    tempflops = m * n; /* h^T */
    tempflops += (2.0 * m * n * m); /* delta * h^T */
    tempflops *= t; /* for t time steps */
    tempflops += (m * m * (t-1)); /* for summation of dJdU */
    gflops += tempflops;
    tempflops = k * n; /* x^T */
    tempflops += (2.0 * m * n * k); /* delta * x^T */
    tempflops *= t; /* for t time steps */
    tempflops += (m * k * (t-1)); /* for summation of dJdW */
    gflops += tempflops;
  }
  if (pass == 1 || pass == 3) {
    tempflops = m * k; /* W^T */
    tempflops += (2.0 * m * n * k); /* W^T * delta */
    tempflops *= t; /* for t time steps of input */
    gflops += tempflops;
  }
  gflops *= 1E-9; /* to convert flops to Gflops */
#endif
  FTYPE *djdht = (FTYPE*)rnn->djdht->data;
  FTYPE *zt = (FTYPE*)rnn->z->data;
  FTYPE *deltat = (FTYPE*)rnn->deltat->data;
  FTYPE *u = (FTYPE*)rnn->u->data;
  FTYPE *xt = (FTYPE*)rnn->xt->data;
  FTYPE *ht = (FTYPE*)rnn->h->data;
  FTYPE *w = (FTYPE*)rnn->w->data;
  FTYPE *djdu = (FTYPE*)rnn->djdu->data;
  FTYPE *djdw = (FTYPE*)rnn->djdw->data;
  FTYPE *djdxt = (FTYPE*)rnn->djdxt->data;
  FTYPE* zi = (FTYPE*)rnn->z1t->data;
  FTYPE* di1 = (FTYPE*)rnn->di1->data;
  FTYPE* di2 = (FTYPE*)rnn->di2->data;
  FTYPE* dj1 = (FTYPE*)rnn->dj1->data;
  FTYPE* dw1 = (FTYPE*)rnn->dw1->data;
  /*
  FTYPE* uTp = (FTYPE*)rnn->uTp->data;
  FTYPE* wTp = (FTYPE*)rnn->wTp->data;
  FTYPE* hTp = (FTYPE*)rnn->hTp->data;
  FTYPE* xTp = (FTYPE*)rnn->xTp->data;
  */
  libxsmm_bgemm_handle *handleud = rnn->handlewx;
  libxsmm_bgemm_handle *handledh = rnn->handleuh;
  libxsmm_bgemm_handle *handledx = rnn->handlett;
  libxsmm_bgemm_handle *handlewd = rnn->handlewd;
  LIBXSMM_VLA_DECL(2, FTYPE, djdh, djdht, m * n);
  LIBXSMM_VLA_DECL(2, FTYPE, z, zt, m * n);
  LIBXSMM_VLA_DECL(2, FTYPE, delta, deltat, m * n);
  LIBXSMM_VLA_DECL(2, FTYPE, x, xt, k * n);
  LIBXSMM_VLA_DECL(2, FTYPE, h, ht, m * n);
  LIBXSMM_VLA_DECL(2, FTYPE, djdx, djdxt, k * n);

  /* int s; */
  int i;
#ifdef LSTM_TIMING
  unsigned long long start;
  double duration;
  start = libxsmm_timer_tick();
#endif

  LIBXSMM_UNUSED(start_thread/* Need to populate this code */);
  /* for (s = 0; s < nrepeat; ++s) { */
    LIBXSMM_MATINIT(FTYPE, 0, &LIBXSMM_VLA_ACCESS(2, delta, t-1, 0, m * n), m, n, m, 0.0);
    /* matrix_transpose(m, m, u, uTp); - already taken care of in init */
    for (i = t-2; i >= 0; --i) {
      matrix_sigmoid_inverse(m * n, &LIBXSMM_VLA_ACCESS(2, z, i+1, 0, m * n), zi);
      /* libxsmm_bgemm(handleud, uTp, &LIBXSMM_VLA_ACCESS(2, delta, i+1, 0, m * n), di1, tid, rnn->nThreads); */
      libxsmm_bgemm(handleud, u, &LIBXSMM_VLA_ACCESS(2, delta, i+1, 0, m * n), di1, tid, rnn->nThreads);
      matrix_add(m * n, &LIBXSMM_VLA_ACCESS(2, djdh, i+1, 0, m * n), di1, di2);
      matrix_eltwise_mult(m * n, zi, di2, &LIBXSMM_VLA_ACCESS(2, delta, i, 0, m * n));
    }
    if (pass == 1 || pass == 3) {
      /* matrix_transpose(m, k, w, wTp); - already taken care of in init */
      for (i = 0; i < t; ++i) {
        /* libxsmm_bgemm(handlewd, wTp, &LIBXSMM_VLA_ACCESS(2, delta, i, 0, m * n), &LIBXSMM_VLA_ACCESS(2, djdx, i, 0, k * n), tid, rnn->nThreads); */
        libxsmm_bgemm(handlewd, w, &LIBXSMM_VLA_ACCESS(2, delta, i, 0, m * n), &LIBXSMM_VLA_ACCESS(2, djdx, i, 0, k * n), tid, rnn->nThreads);
      }
    }
    if (pass == 2 || pass == 3) {
      for (i = 0; i < t; ++i) {
        /* matrix_transpose(m, n, &LIBXSMM_VLA_ACCESS(2, h, i, 0, m * n), hTp); - already taken care of in init */
        /* libxsmm_bgemm(handledh, &LIBXSMM_VLA_ACCESS(2, delta, i, 0, m * n), hTp, dj1, tid, rnn->nThreads); */
        libxsmm_bgemm(handledh, &LIBXSMM_VLA_ACCESS(2, delta, i, 0, m * n), h, dj1, tid, rnn->nThreads);
        matrix_add(m*m, dj1, djdu, djdu);
        /* matrix_transpose(k, n, &LIBXSMM_VLA_ACCESS(2, x, i, 0, k * n), xTp); - already taken care of in init */
        /* libxsmm_bgemm(handledx, &LIBXSMM_VLA_ACCESS(2, delta, i, 0, m * n), xTp, dw1, tid, rnn->nThreads); */
        libxsmm_bgemm(handledx, &LIBXSMM_VLA_ACCESS(2, delta, i, 0, m * n), x, dw1, tid, rnn->nThreads);
        matrix_add(m*k, dw1, djdw, djdw);
      }
    }
  /* } */
#ifdef LSTM_TIMING
  duration = libxsmm_timer_duration(start, libxsmm_timer_tick());
  if (0 < duration) {
    fprintf(stdout, "\tLIBXSMM: %.1f GFLOPS/s\n", gflops / duration); /* *nrepeat */
  }
#endif

  return status;
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_rnncell_execute_st(libxsmm_dnn_rnncell* handle, libxsmm_dnn_compute_kind kind,
  /*unsigned*/int start_thread, /*unsigned*/int tid) {
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  if (0 != handle) {
    switch (kind) {
      case LIBXSMM_DNN_COMPUTE_KIND_FWD: {
                                           status = libxsmm_dnn_rnncell_fwd(handle, start_thread, tid);
                                         } break;
      case LIBXSMM_DNN_COMPUTE_KIND_BWD: {
                                           status = libxsmm_dnn_rnncell_bwd_upd_bu(handle, start_thread, tid, 1);
                                         } break;
      case LIBXSMM_DNN_COMPUTE_KIND_UPD: {
                                           status = libxsmm_dnn_rnncell_bwd_upd_bu(handle, start_thread, tid, 2);
                                         } break;
      case LIBXSMM_DNN_COMPUTE_KIND_ALL: {
                                           status = libxsmm_dnn_rnncell_bwd_upd_bu(handle, start_thread, tid, 3);
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

