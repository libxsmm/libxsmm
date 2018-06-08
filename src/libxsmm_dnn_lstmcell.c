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


#if defined(LSTM_TIMING)
#include <stdio.h>
double Gbl_t_input_total = 0., Gbl_t_recur_total = 0., Gbl_t_eltwise_total = 0., Gbl_t_nonlin_total = 0.;
unsigned long long Gbl_t_input = 0, Gbl_t_recur = 0, Gbl_t_eltwise = 0, Gbl_t_nonlin = 0;
double Gbl_duration_input = 0., Gbl_duration_recur = 0., Gbl_duration_eltwise = 0., Gbl_duration_nonlin = 0.;
#endif


LIBXSMM_API libxsmm_dnn_lstmcell* libxsmm_dnn_create_lstmcell(libxsmm_dnn_lstmcell_desc lstmcell_desc, libxsmm_dnn_err_t* status)
{
  libxsmm_dnn_lstmcell* handle = 0;
  *status = LIBXSMM_DNN_SUCCESS;

  handle = (libxsmm_dnn_lstmcell*)malloc(sizeof(libxsmm_dnn_lstmcell));
  if (0 != handle) {
    /* zero entire content; not only safer but also sets data and code pointers to NULL */
    memset(handle, 0, sizeof(*handle));
    /* initialize known handle components */
    handle->desc = lstmcell_desc;
    handle->datatype_in = lstmcell_desc.datatype_in;
    handle->datatype_out = lstmcell_desc.datatype_out;
    if ( (lstmcell_desc.datatype_in != LIBXSMM_DNN_DATATYPE_F32) || (lstmcell_desc.datatype_out != LIBXSMM_DNN_DATATYPE_F32) ) {
      /* error */
      *status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
      return handle;
    }
    handle->buffer_format = lstmcell_desc.buffer_format;
    handle->m = lstmcell_desc.m;
    handle->n = lstmcell_desc.n;
    handle->k = lstmcell_desc.k;
    handle->t = lstmcell_desc.t;
    if (lstmcell_desc.t < 2) {
      *status = LIBXSMM_DNN_ERR_TIME_STEPS_TOO_SMALL;
    }
    handle->bm = lstmcell_desc.bm;
    handle->bn = lstmcell_desc.bn;
    handle->bk = lstmcell_desc.bk;
    handle->b_m1 = lstmcell_desc.b_m1;
    handle->b_n1 = lstmcell_desc.b_n1;
    handle->b_k1 = lstmcell_desc.b_k1;
    handle->b_m2 = lstmcell_desc.b_m2;
    handle->b_n2 = lstmcell_desc.b_n2;
    handle->b_k2 = lstmcell_desc.b_k2;
  } else {
    *status = LIBXSMM_DNN_ERR_CREATE_HANDLE;
  }
  return handle;
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_destroy_lstmcell(const libxsmm_dnn_lstmcell* handle)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
  if (0 != handle) {
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
      if ( (type == LIBXSMM_DNN_LSTM_REGULAR_INPUT)          || (type == LIBXSMM_DNN_LSTM_GRADIENT_INPUT)  ||
           (type == LIBXSMM_DNN_LSTM_REGULAR_HIDDEN_STATE)   || (type == LIBXSMM_DNN_LSTM_GRADIENT_HIDDEN_STATE) ||
           (type == LIBXSMM_DNN_LSTM_REGULAR_WEIGHT_I)       || (type == LIBXSMM_DNN_LSTM_GRADIENT_WEIGHT_I) ||
           (type == LIBXSMM_DNN_LSTM_REGULAR_WEIGHT_F)       || (type == LIBXSMM_DNN_LSTM_GRADIENT_WEIGHT_F) ||
           (type == LIBXSMM_DNN_LSTM_REGULAR_WEIGHT_O)       || (type == LIBXSMM_DNN_LSTM_GRADIENT_WEIGHT_O) ||
           (type == LIBXSMM_DNN_LSTM_REGULAR_WEIGHT_C)       || (type == LIBXSMM_DNN_LSTM_GRADIENT_WEIGHT_C) ||
           (type == LIBXSMM_DNN_LSTM_REGULAR_RECUR_WEIGHT_I) || (type == LIBXSMM_DNN_LSTM_GRADIENT_RECUR_WEIGHT_I) ||
           (type == LIBXSMM_DNN_LSTM_REGULAR_RECUR_WEIGHT_F) || (type == LIBXSMM_DNN_LSTM_GRADIENT_RECUR_WEIGHT_F) ||
           (type == LIBXSMM_DNN_LSTM_REGULAR_RECUR_WEIGHT_O) || (type == LIBXSMM_DNN_LSTM_GRADIENT_RECUR_WEIGHT_O) ||
           (type == LIBXSMM_DNN_LSTM_REGULAR_RECUR_WEIGHT_C) || (type == LIBXSMM_DNN_LSTM_GRADIENT_RECUR_WEIGHT_C) ||
           (type == LIBXSMM_DNN_LSTM_REGULAR_BIAS_I)         || (type == LIBXSMM_DNN_LSTM_GRADIENT_BIAS_I)   ||
           (type == LIBXSMM_DNN_LSTM_REGULAR_BIAS_F)         || (type == LIBXSMM_DNN_LSTM_GRADIENT_BIAS_F)   ||
           (type == LIBXSMM_DNN_LSTM_REGULAR_BIAS_O)         || (type == LIBXSMM_DNN_LSTM_GRADIENT_BIAS_O)   ||
           (type == LIBXSMM_DNN_LSTM_REGULAR_BIAS_C)         || (type == LIBXSMM_DNN_LSTM_GRADIENT_BIAS_C) ) {
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
                /* TODO: Check if the following layout works for bwd and upd passes */
                if ( (type == LIBXSMM_DNN_LSTM_REGULAR_INPUT) || (type == LIBXSMM_DNN_LSTM_GRADIENT_INPUT) ) {
                  layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_RLK;
                  layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_RLN;
                  layout->dim_type[2] = LIBXSMM_DNN_TENSOR_DIMTYPE_RLK;
                  layout->dim_type[3] = LIBXSMM_DNN_TENSOR_DIMTYPE_RLN;
                  layout->dim_size[0] = handle->bk;
                  layout->dim_size[1] = handle->bn;
                  layout->dim_size[2] = handle->k / handle->bk;
                  layout->dim_size[3] = handle->n / handle->bn;
                } else if ( (type == LIBXSMM_DNN_LSTM_REGULAR_HIDDEN_STATE) || (type == LIBXSMM_DNN_LSTM_GRADIENT_HIDDEN_STATE) ) {
                  layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_RLM;
                  layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_RLN;
                  layout->dim_type[2] = LIBXSMM_DNN_TENSOR_DIMTYPE_RLM;
                  layout->dim_type[3] = LIBXSMM_DNN_TENSOR_DIMTYPE_RLN;
                  layout->dim_size[0] = handle->bm;
                  layout->dim_size[1] = handle->bn;
                  layout->dim_size[2] = handle->m / handle->bm;
                  layout->dim_size[3] = handle->n / handle->bn;
                } else if ( (type == LIBXSMM_DNN_LSTM_REGULAR_WEIGHT_I) || (type == LIBXSMM_DNN_LSTM_GRADIENT_WEIGHT_I) ||
                            (type == LIBXSMM_DNN_LSTM_REGULAR_WEIGHT_F) || (type == LIBXSMM_DNN_LSTM_GRADIENT_WEIGHT_F) ||
                            (type == LIBXSMM_DNN_LSTM_REGULAR_WEIGHT_O) || (type == LIBXSMM_DNN_LSTM_GRADIENT_WEIGHT_O) ||
                            (type == LIBXSMM_DNN_LSTM_REGULAR_WEIGHT_C) || (type == LIBXSMM_DNN_LSTM_GRADIENT_WEIGHT_C) ) {
                  layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_RLM;
                  layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_RLK;
                  layout->dim_type[2] = LIBXSMM_DNN_TENSOR_DIMTYPE_RLK;
                  layout->dim_type[3] = LIBXSMM_DNN_TENSOR_DIMTYPE_RLM;
                  layout->dim_size[0] = handle->bm;
                  layout->dim_size[1] = handle->bk;
                  layout->dim_size[2] = handle->k / handle->bk;
                  layout->dim_size[3] = handle->m / handle->bm;
                } else if ( (type == LIBXSMM_DNN_LSTM_REGULAR_RECUR_WEIGHT_I) || (type == LIBXSMM_DNN_LSTM_GRADIENT_RECUR_WEIGHT_I) ||
                            (type == LIBXSMM_DNN_LSTM_REGULAR_RECUR_WEIGHT_F) || (type == LIBXSMM_DNN_LSTM_GRADIENT_RECUR_WEIGHT_F) ||
                            (type == LIBXSMM_DNN_LSTM_REGULAR_RECUR_WEIGHT_O) || (type == LIBXSMM_DNN_LSTM_GRADIENT_RECUR_WEIGHT_O) ||
                            (type == LIBXSMM_DNN_LSTM_REGULAR_RECUR_WEIGHT_C) || (type == LIBXSMM_DNN_LSTM_GRADIENT_RECUR_WEIGHT_C) ) {
                  layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_RLM;
                  layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_RLM;
                  layout->dim_type[2] = LIBXSMM_DNN_TENSOR_DIMTYPE_RLM;
                  layout->dim_type[3] = LIBXSMM_DNN_TENSOR_DIMTYPE_RLM;
                  layout->dim_size[0] = handle->bm;
                  layout->dim_size[1] = handle->bm;
                  layout->dim_size[2] = handle->m / handle->bm;
                  layout->dim_size[3] = handle->m / handle->bm;
                } else if ( (type == LIBXSMM_DNN_LSTM_REGULAR_BIAS_I) || (type == LIBXSMM_DNN_LSTM_GRADIENT_BIAS_I) ||
                            (type == LIBXSMM_DNN_LSTM_REGULAR_BIAS_F) || (type == LIBXSMM_DNN_LSTM_GRADIENT_BIAS_F) ||
                            (type == LIBXSMM_DNN_LSTM_REGULAR_BIAS_O) || (type == LIBXSMM_DNN_LSTM_GRADIENT_BIAS_O) ||
                            (type == LIBXSMM_DNN_LSTM_REGULAR_BIAS_C) || (type == LIBXSMM_DNN_LSTM_GRADIENT_BIAS_C) ) {
                  layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_RLM;
                  layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_RLN;
                  layout->dim_type[2] = LIBXSMM_DNN_TENSOR_DIMTYPE_RLM;
                  layout->dim_type[3] = LIBXSMM_DNN_TENSOR_DIMTYPE_RLN;
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


LIBXSMM_API size_t libxsmm_dnn_lstmcell_get_scratch_size(const libxsmm_dnn_lstmcell* handle, const libxsmm_dnn_compute_kind kind, libxsmm_dnn_err_t* status)
{
  size_t sizeof_datatype = sizeof(float);
  size_t size = 0;
  *status = LIBXSMM_DNN_SUCCESS;

  if (0 != handle) {
    switch (kind) {
      case LIBXSMM_DNN_COMPUTE_KIND_FWD: {
#if defined(NON_FUSED_INPUT_GEMM)
                                           size += handle->m * handle->n * sizeof_datatype * handle->t; /* i1t */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype * handle->t; /* f1t */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype * handle->t; /* o1t */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype * handle->t; /* c1t */
                                           size += 64;
#else
                                           size += handle->m * 4 * handle->n * sizeof_datatype * handle->t; /* i1t */
                                           size += 64;
                                           size += 0; /* f1t */
                                           size += 0; /* o1t */
                                           size += 0; /* c1t */
#endif
                                           size += handle->m * handle->n * sizeof_datatype; /* i2 */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype; /* f2 */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype; /* o2 */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype; /* c2 */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype; /* d1 */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype; /* d2 */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype; /* dh */
                                           size += 64;
                                         } break;
      case LIBXSMM_DNN_COMPUTE_KIND_BWD:
      case LIBXSMM_DNN_COMPUTE_KIND_UPD:
      case LIBXSMM_DNN_COMPUTE_KIND_ALL: {
#if defined(NON_FUSED_INPUT_GEMM)
                                           size += handle->m * handle->n * sizeof_datatype * handle->t; /* i1t */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype * handle->t; /* f1t */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype * handle->t; /* o1t */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype * handle->t; /* c1t */
                                           size += 64;
#else
                                           size += handle->m * 4 * handle->n * sizeof_datatype * handle->t; /* i1t */
                                           size += 64;
                                           size += 0; /* f1t */
                                           size += 0; /* o1t */
                                           size += 0; /* c1t */
#endif
                                           size += handle->m * handle->n * sizeof_datatype; /* i1b */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype; /* i2 */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype; /* i3 */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype; /* f1b */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype; /* f2 */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype; /* f3 */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype; /* o1b */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype; /* o2 */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype; /* c1b */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype; /* c2 */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype; /* d1 */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype; /* d2 */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype; /* d3 (dh) */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype; /* d4 */
                                           size += 64;
                                           size += handle->m * handle->k * sizeof_datatype; /* wTp */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype; /* rTp */
                                           size += 64;
                                           size += handle->k * handle->n * sizeof_datatype; /* xTp */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype; /* deltaTp */
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
#if defined(NON_FUSED_INPUT_GEMM)
                                           if (address % 64 == 0) {
                                             handle->i1t->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->i1t->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype * handle->t;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->f1t->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->f1t->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype * handle->t;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->o1t->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->o1t->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype * handle->t;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->c1t->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->c1t->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype * handle->t;
                                           address += scratch_size + 64;
#else
                                           if (address % 64 == 0) {
                                             handle->i1t->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->i1t->data = (void*)(address+offset);
                                           }
                                           handle->f1t->data = handle->i1t->data; /* not used */
                                           handle->o1t->data = handle->i1t->data; /* not used */
                                           handle->c1t->data = handle->i1t->data; /* not used */
                                           scratch_size = handle->m * 4 * handle->n * sizeof_datatype * handle->t;
                                           address += scratch_size + 64;
#endif
                                           if (address % 64 == 0) {
                                             handle->i2->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->i2->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->f2->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->f2->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->o2->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->o2->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->c2->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->c2->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->d1->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->d1->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->d2->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->d2->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->dh->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->dh->data = (void*)(address+offset);
                                           }
                                         } break;
      case LIBXSMM_DNN_COMPUTE_KIND_BWD:
      case LIBXSMM_DNN_COMPUTE_KIND_UPD:
      case LIBXSMM_DNN_COMPUTE_KIND_ALL: {
#if defined(NON_FUSED_INPUT_GEMM)
                                           if (address % 64 == 0) {
                                             handle->i1t->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->i1t->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype * handle->t;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->f1t->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->f1t->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype * handle->t;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->o1t->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->o1t->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype * handle->t;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->c1t->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->c1t->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype * handle->t;
                                           address += scratch_size + 64;
#else
                                           if (address % 64 == 0) {
                                             handle->i1t->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->i1t->data = (void*)(address+offset);
                                           }
                                           handle->f1t->data = handle->i1t->data; /* not used */
                                           handle->o1t->data = handle->i1t->data; /* not used */
                                           handle->c1t->data = handle->i1t->data; /* not used */
                                           scratch_size = handle->m * 4 * handle->n * sizeof_datatype * handle->t;
                                           address += scratch_size + 64;
#endif
                                           if (address % 64 == 0) {
                                             handle->i1b->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->i1b->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->i2->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->i2->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->i3->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->i3->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->f1b->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->f1b->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->f2->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->f2->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->f3->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->f3->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->o1b->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->o1b->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->o2->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->o2->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->c1b->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->c1b->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->c2->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->c2->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->d1->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->d1->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->d2->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->d2->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->dh->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->dh->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->d4->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->d4->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype;
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
                                             handle->rTp->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->rTp->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->xTp->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->xTp->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->k * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->deltaTp->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->deltaTp->data = (void*)(address+offset);
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
                                           handle->i1t->data = 0;
                                           handle->i2->data = 0;
                                           handle->f1t->data = 0;
                                           handle->f2->data = 0;
                                           handle->o1t->data = 0;
                                           handle->o2->data = 0;
                                           handle->c1t->data = 0;
                                           handle->c2->data = 0;
                                           handle->d1->data = 0;
                                           handle->d2->data = 0;
                                           handle->dh->data = 0;
                                           handle->i1t = 0;
                                           handle->i2 = 0;
                                           handle->f1t = 0;
                                           handle->f2 = 0;
                                           handle->o1t = 0;
                                           handle->o2 = 0;
                                           handle->c1t = 0;
                                           handle->c2 = 0;
                                           handle->d1 = 0;
                                           handle->d2 = 0;
                                           handle->dh = 0;
                                         } break;
      case LIBXSMM_DNN_COMPUTE_KIND_BWD:
      case LIBXSMM_DNN_COMPUTE_KIND_UPD:
      case LIBXSMM_DNN_COMPUTE_KIND_ALL: {
                                           handle->i1t->data = 0;
                                           handle->i1b->data = 0;
                                           handle->i2->data = 0;
                                           handle->f1t->data = 0;
                                           handle->f1b->data = 0;
                                           handle->f2->data = 0;
                                           handle->o1t->data = 0;
                                           handle->o1b->data = 0;
                                           handle->o2->data = 0;
                                           handle->c1t->data = 0;
                                           handle->c1b->data = 0;
                                           handle->c2->data = 0;
                                           handle->d1->data = 0;
                                           handle->d2->data = 0;
                                           handle->dh->data = 0;
                                           handle->i3->data = 0;
                                           handle->f3->data = 0;
                                           handle->d4->data = 0;
                                           handle->rTp->data = 0;
                                           handle->wTp->data = 0;
                                           handle->xTp->data = 0;
                                           handle->deltaTp->data = 0;
                                           handle->i1t = 0;
                                           handle->i1b = 0;
                                           handle->i2 = 0;
                                           handle->f1t = 0;
                                           handle->f1b = 0;
                                           handle->f2 = 0;
                                           handle->o1t = 0;
                                           handle->o1b = 0;
                                           handle->o2 = 0;
                                           handle->c1t = 0;
                                           handle->c1b = 0;
                                           handle->c2 = 0;
                                           handle->d1 = 0;
                                           handle->d2 = 0;
                                           handle->dh = 0;
                                           handle->i3 = 0;
                                           handle->f3 = 0;
                                           handle->d4 = 0;
                                           handle->rTp = 0;
                                           handle->wTp = 0;
                                           handle->xTp = 0;
                                           handle->deltaTp = 0;
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
  size_t sizeof_datatype = sizeof(float);
  size_t size = 0;
  *status = LIBXSMM_DNN_SUCCESS;

  if (0 != handle) {
    switch (kind) {
      case LIBXSMM_DNN_COMPUTE_KIND_FWD: {
                                           size += handle->m * handle->n * sizeof_datatype; /* i */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype; /* f */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype; /* o */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype; /* c */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype; /* d */
                                           size += 64;
                                         } break;
      case LIBXSMM_DNN_COMPUTE_KIND_BWD:
      case LIBXSMM_DNN_COMPUTE_KIND_UPD:
      case LIBXSMM_DNN_COMPUTE_KIND_ALL: {
                                           size += handle->m * handle->n * sizeof_datatype * handle->t; /* i */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype * handle->t; /* f */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype * handle->t; /* o */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype * handle->t; /* c */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype * handle->t; /* d */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype * handle->t; /* djddt */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype * handle->t; /* djdit */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype * handle->t; /* djdft */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype * handle->t; /* djdct */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype * handle->t; /* djdot */
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


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_lstmcell_bind_internalstate(libxsmm_dnn_lstmcell* handle, const libxsmm_dnn_compute_kind kind, const void* internalstate)
{
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
                                             handle->i->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->i->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->f->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->f->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->o->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->o->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->c->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->c->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->d->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->d->data = (void*)(address+offset);
                                           }
                                         } break;
      case LIBXSMM_DNN_COMPUTE_KIND_BWD:
      case LIBXSMM_DNN_COMPUTE_KIND_UPD:
      case LIBXSMM_DNN_COMPUTE_KIND_ALL: {
                                           if (address % 64 == 0) {
                                             handle->i->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->i->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype * handle->t;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->f->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->f->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype * handle->t;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->o->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->o->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype * handle->t;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->c->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->c->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype * handle->t;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->d->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->d->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype * handle->t;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->djddt->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->djddt->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype * handle->t;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->djdit->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->djdit->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype * handle->t;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->djdft->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->djdft->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype * handle->t;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->djdot->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->djdot->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype * handle->t;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->djdct->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->djdct->data = (void*)(address+offset);
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


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_lstmcell_release_internalstate(libxsmm_dnn_lstmcell* handle, const libxsmm_dnn_compute_kind kind)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  if (0 != handle) {
    switch (kind) {
      case LIBXSMM_DNN_COMPUTE_KIND_FWD: {
                                           handle->i->data = 0;
                                           handle->f->data = 0;
                                           handle->o->data = 0;
                                           handle->c->data = 0;
                                           handle->d->data = 0;
                                           handle->i = 0;
                                           handle->f = 0;
                                           handle->o = 0;
                                           handle->c = 0;
                                           handle->d = 0;
                                         } break;
      case LIBXSMM_DNN_COMPUTE_KIND_BWD:
      case LIBXSMM_DNN_COMPUTE_KIND_UPD:
      case LIBXSMM_DNN_COMPUTE_KIND_ALL: {
                                           handle->i->data = 0;
                                           handle->f->data = 0;
                                           handle->o->data = 0;
                                           handle->c->data = 0;
                                           handle->d->data = 0;
                                           handle->djddt->data = 0;
                                           handle->djdit->data = 0;
                                           handle->djdft->data = 0;
                                           handle->djdot->data = 0;
                                           handle->djdct->data = 0;
                                           handle->i = 0;
                                           handle->f = 0;
                                           handle->o = 0;
                                           handle->c = 0;
                                           handle->d = 0;
                                           handle->djddt = 0;
                                           handle->djdit = 0;
                                           handle->djdft = 0;
                                           handle->djdot = 0;
                                           handle->djdct = 0;
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


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_lstmcell_bind_tensor(libxsmm_dnn_lstmcell* handle, const libxsmm_dnn_tensor* tensor, const libxsmm_dnn_tensor_type type)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  /* check for tensor type */
  if ( (type != LIBXSMM_DNN_LSTM_REGULAR_INPUT)         && (type != LIBXSMM_DNN_LSTM_GRADIENT_INPUT)  &&
      (type != LIBXSMM_DNN_LSTM_REGULAR_HIDDEN_STATE)   && (type != LIBXSMM_DNN_LSTM_GRADIENT_HIDDEN_STATE) &&
      (type != LIBXSMM_DNN_LSTM_REGULAR_WEIGHT_I)       && (type != LIBXSMM_DNN_LSTM_GRADIENT_WEIGHT_I) &&
      (type != LIBXSMM_DNN_LSTM_REGULAR_WEIGHT_F)       && (type != LIBXSMM_DNN_LSTM_GRADIENT_WEIGHT_F) &&
      (type != LIBXSMM_DNN_LSTM_REGULAR_WEIGHT_O)       && (type != LIBXSMM_DNN_LSTM_GRADIENT_WEIGHT_O) &&
      (type != LIBXSMM_DNN_LSTM_REGULAR_WEIGHT_C)       && (type != LIBXSMM_DNN_LSTM_GRADIENT_WEIGHT_C) &&
      (type != LIBXSMM_DNN_LSTM_REGULAR_RECUR_WEIGHT_I) && (type != LIBXSMM_DNN_LSTM_GRADIENT_RECUR_WEIGHT_I) &&
      (type != LIBXSMM_DNN_LSTM_REGULAR_RECUR_WEIGHT_F) && (type != LIBXSMM_DNN_LSTM_GRADIENT_RECUR_WEIGHT_F) &&
      (type != LIBXSMM_DNN_LSTM_REGULAR_RECUR_WEIGHT_O) && (type != LIBXSMM_DNN_LSTM_GRADIENT_RECUR_WEIGHT_O) &&
      (type != LIBXSMM_DNN_LSTM_REGULAR_RECUR_WEIGHT_C) && (type != LIBXSMM_DNN_LSTM_GRADIENT_RECUR_WEIGHT_C) &&
      (type != LIBXSMM_DNN_LSTM_REGULAR_BIAS_I)         && (type != LIBXSMM_DNN_LSTM_GRADIENT_BIAS_I)   &&
      (type != LIBXSMM_DNN_LSTM_REGULAR_BIAS_F)         && (type != LIBXSMM_DNN_LSTM_GRADIENT_BIAS_F)   &&
      (type != LIBXSMM_DNN_LSTM_REGULAR_BIAS_O)         && (type != LIBXSMM_DNN_LSTM_GRADIENT_BIAS_O)   &&
      (type != LIBXSMM_DNN_LSTM_REGULAR_BIAS_C)         && (type != LIBXSMM_DNN_LSTM_GRADIENT_BIAS_C) ) {
    status = LIBXSMM_DNN_ERR_UNKNOWN_TENSOR_TYPE;
    return status;
  }

  if (handle != 0 && tensor != 0) {
    libxsmm_dnn_tensor_datalayout* handle_layout = libxsmm_dnn_lstmcell_create_tensor_datalayout(handle, type, &status);

    if ( libxsmm_dnn_compare_tensor_datalayout(handle_layout, tensor->layout, &status) == 0 ) {
      if ( type == LIBXSMM_DNN_LSTM_REGULAR_INPUT ) {
        handle->xt = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_LSTM_GRADIENT_INPUT ) {
        handle->djdxt = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_LSTM_REGULAR_HIDDEN_STATE ) {
        handle->h = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_LSTM_GRADIENT_HIDDEN_STATE ) {
        handle->djdht = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_LSTM_REGULAR_WEIGHT_I ) {
        handle->wi = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_LSTM_GRADIENT_WEIGHT_I ) {
        handle->djdwi = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_LSTM_REGULAR_WEIGHT_F ) {
        handle->wf = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_LSTM_GRADIENT_WEIGHT_F ) {
        handle->djdwf = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_LSTM_REGULAR_WEIGHT_O ) {
        handle->wo = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_LSTM_GRADIENT_WEIGHT_O ) {
        handle->djdwo = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_LSTM_REGULAR_WEIGHT_C ) {
        handle->wc = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_LSTM_GRADIENT_WEIGHT_C ) {
        handle->djdwc = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_LSTM_REGULAR_RECUR_WEIGHT_I ) {
        handle->ri = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_LSTM_GRADIENT_RECUR_WEIGHT_I ) {
        handle->djdri = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_LSTM_REGULAR_RECUR_WEIGHT_F ) {
        handle->rf = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_LSTM_GRADIENT_RECUR_WEIGHT_F ) {
        handle->djdrf = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_LSTM_REGULAR_RECUR_WEIGHT_O ) {
        handle->ro = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_LSTM_GRADIENT_RECUR_WEIGHT_O ) {
        handle->djdro = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_LSTM_REGULAR_RECUR_WEIGHT_C ) {
        handle->rc = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_LSTM_GRADIENT_RECUR_WEIGHT_C ) {
        handle->djdrc = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_LSTM_REGULAR_BIAS_I ) {
        handle->bi = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_LSTM_GRADIENT_BIAS_I ) {
        handle->djdbi = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_LSTM_REGULAR_BIAS_F ) {
        handle->bf = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_LSTM_GRADIENT_BIAS_F ) {
        handle->djdbf = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_LSTM_REGULAR_BIAS_O ) {
        handle->bo = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_LSTM_GRADIENT_BIAS_O ) {
        handle->djdbo = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_LSTM_REGULAR_BIAS_C ) {
        handle->bc = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_LSTM_GRADIENT_BIAS_C ) {
        handle->djdbc = (libxsmm_dnn_tensor*)tensor;
      } else {
        /* cannot happen */
      }
    } else {
      status = LIBXSMM_DNN_ERR_MISMATCH_TENSOR;
    }

    /* libxsmm_dnn_destroy_tensor_datalayout( handle_layout ); */
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
  if ( (type != LIBXSMM_DNN_LSTM_REGULAR_INPUT)         && (type != LIBXSMM_DNN_LSTM_GRADIENT_INPUT)  &&
      (type != LIBXSMM_DNN_LSTM_REGULAR_HIDDEN_STATE)   && (type != LIBXSMM_DNN_LSTM_GRADIENT_HIDDEN_STATE) &&
      (type != LIBXSMM_DNN_LSTM_REGULAR_WEIGHT_I)       && (type != LIBXSMM_DNN_LSTM_GRADIENT_WEIGHT_I) &&
      (type != LIBXSMM_DNN_LSTM_REGULAR_WEIGHT_F)       && (type != LIBXSMM_DNN_LSTM_GRADIENT_WEIGHT_F) &&
      (type != LIBXSMM_DNN_LSTM_REGULAR_WEIGHT_O)       && (type != LIBXSMM_DNN_LSTM_GRADIENT_WEIGHT_O) &&
      (type != LIBXSMM_DNN_LSTM_REGULAR_WEIGHT_C)       && (type != LIBXSMM_DNN_LSTM_GRADIENT_WEIGHT_C) &&
      (type != LIBXSMM_DNN_LSTM_REGULAR_RECUR_WEIGHT_I) && (type != LIBXSMM_DNN_LSTM_GRADIENT_RECUR_WEIGHT_I) &&
      (type != LIBXSMM_DNN_LSTM_REGULAR_RECUR_WEIGHT_F) && (type != LIBXSMM_DNN_LSTM_GRADIENT_RECUR_WEIGHT_F) &&
      (type != LIBXSMM_DNN_LSTM_REGULAR_RECUR_WEIGHT_O) && (type != LIBXSMM_DNN_LSTM_GRADIENT_RECUR_WEIGHT_O) &&
      (type != LIBXSMM_DNN_LSTM_REGULAR_RECUR_WEIGHT_C) && (type != LIBXSMM_DNN_LSTM_GRADIENT_RECUR_WEIGHT_C) &&
      (type != LIBXSMM_DNN_LSTM_REGULAR_BIAS_I)         && (type != LIBXSMM_DNN_LSTM_GRADIENT_BIAS_I)   &&
      (type != LIBXSMM_DNN_LSTM_REGULAR_BIAS_F)         && (type != LIBXSMM_DNN_LSTM_GRADIENT_BIAS_F)   &&
      (type != LIBXSMM_DNN_LSTM_REGULAR_BIAS_O)         && (type != LIBXSMM_DNN_LSTM_GRADIENT_BIAS_O)   &&
      (type != LIBXSMM_DNN_LSTM_REGULAR_BIAS_C)         && (type != LIBXSMM_DNN_LSTM_GRADIENT_BIAS_C) ) {
    return tensor;
  }

  if (handle != 0) {
    if ( type == LIBXSMM_DNN_LSTM_REGULAR_INPUT ) {
      tensor = handle->xt;
    } else if ( type == LIBXSMM_DNN_LSTM_GRADIENT_INPUT ) {
      tensor = handle->djdxt;
    } else if ( type == LIBXSMM_DNN_LSTM_REGULAR_HIDDEN_STATE ) {
      tensor = handle->h;
    } else if ( type == LIBXSMM_DNN_LSTM_GRADIENT_HIDDEN_STATE ) {
      tensor = handle->djdht;
    } else if ( type == LIBXSMM_DNN_LSTM_REGULAR_WEIGHT_I ) {
      tensor = handle->wi;
    } else if ( type == LIBXSMM_DNN_LSTM_GRADIENT_WEIGHT_I ) {
      tensor = handle->djdwi;
    } else if ( type == LIBXSMM_DNN_LSTM_REGULAR_WEIGHT_F ) {
      tensor = handle->wf;
    } else if ( type == LIBXSMM_DNN_LSTM_GRADIENT_WEIGHT_F ) {
      tensor = handle->djdwf;
    } else if ( type == LIBXSMM_DNN_LSTM_REGULAR_WEIGHT_O ) {
      tensor = handle->wo;
    } else if ( type == LIBXSMM_DNN_LSTM_GRADIENT_WEIGHT_O ) {
      tensor = handle->djdwo;
    } else if ( type == LIBXSMM_DNN_LSTM_REGULAR_WEIGHT_C ) {
      tensor = handle->wc;
    } else if ( type == LIBXSMM_DNN_LSTM_GRADIENT_WEIGHT_C ) {
      tensor = handle->djdwc;
    } else if ( type == LIBXSMM_DNN_LSTM_REGULAR_RECUR_WEIGHT_I ) {
      tensor = handle->ri;
    } else if ( type == LIBXSMM_DNN_LSTM_GRADIENT_RECUR_WEIGHT_I ) {
      tensor = handle->djdri;
    } else if ( type == LIBXSMM_DNN_LSTM_REGULAR_RECUR_WEIGHT_F ) {
      tensor = handle->rf;
    } else if ( type == LIBXSMM_DNN_LSTM_GRADIENT_RECUR_WEIGHT_F ) {
      tensor = handle->djdrf;
    } else if ( type == LIBXSMM_DNN_LSTM_REGULAR_RECUR_WEIGHT_O ) {
      tensor = handle->ro;
    } else if ( type == LIBXSMM_DNN_LSTM_GRADIENT_RECUR_WEIGHT_O ) {
      tensor = handle->djdro;
    } else if ( type == LIBXSMM_DNN_LSTM_REGULAR_RECUR_WEIGHT_C ) {
      tensor = handle->rc;
    } else if ( type == LIBXSMM_DNN_LSTM_GRADIENT_RECUR_WEIGHT_C ) {
      tensor = handle->djdrc;
    } else if ( type == LIBXSMM_DNN_LSTM_REGULAR_BIAS_I ) {
      tensor = handle->bi;
    } else if ( type == LIBXSMM_DNN_LSTM_GRADIENT_BIAS_I ) {
      tensor = handle->djdbi;
    } else if ( type == LIBXSMM_DNN_LSTM_REGULAR_BIAS_F ) {
      tensor = handle->bf;
    } else if ( type == LIBXSMM_DNN_LSTM_GRADIENT_BIAS_F ) {
      tensor = handle->djdbf;
    } else if ( type == LIBXSMM_DNN_LSTM_REGULAR_BIAS_O ) {
      tensor = handle->bo;
    } else if ( type == LIBXSMM_DNN_LSTM_GRADIENT_BIAS_O ) {
      tensor = handle->djdbo;
    } else if ( type == LIBXSMM_DNN_LSTM_REGULAR_BIAS_C ) {
      tensor = handle->bc;
    } else if ( type == LIBXSMM_DNN_LSTM_GRADIENT_BIAS_C ) {
      tensor = handle->djdbc;
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
  if ( (type != LIBXSMM_DNN_LSTM_REGULAR_INPUT)         && (type != LIBXSMM_DNN_LSTM_GRADIENT_INPUT)  &&
      (type != LIBXSMM_DNN_LSTM_REGULAR_HIDDEN_STATE)   && (type != LIBXSMM_DNN_LSTM_GRADIENT_HIDDEN_STATE) &&
      (type != LIBXSMM_DNN_LSTM_REGULAR_WEIGHT_I)       && (type != LIBXSMM_DNN_LSTM_GRADIENT_WEIGHT_I) &&
      (type != LIBXSMM_DNN_LSTM_REGULAR_WEIGHT_F)       && (type != LIBXSMM_DNN_LSTM_GRADIENT_WEIGHT_F) &&
      (type != LIBXSMM_DNN_LSTM_REGULAR_WEIGHT_O)       && (type != LIBXSMM_DNN_LSTM_GRADIENT_WEIGHT_O) &&
      (type != LIBXSMM_DNN_LSTM_REGULAR_WEIGHT_C)       && (type != LIBXSMM_DNN_LSTM_GRADIENT_WEIGHT_C) &&
      (type != LIBXSMM_DNN_LSTM_REGULAR_RECUR_WEIGHT_I) && (type != LIBXSMM_DNN_LSTM_GRADIENT_RECUR_WEIGHT_I) &&
      (type != LIBXSMM_DNN_LSTM_REGULAR_RECUR_WEIGHT_F) && (type != LIBXSMM_DNN_LSTM_GRADIENT_RECUR_WEIGHT_F) &&
      (type != LIBXSMM_DNN_LSTM_REGULAR_RECUR_WEIGHT_O) && (type != LIBXSMM_DNN_LSTM_GRADIENT_RECUR_WEIGHT_O) &&
      (type != LIBXSMM_DNN_LSTM_REGULAR_RECUR_WEIGHT_C) && (type != LIBXSMM_DNN_LSTM_GRADIENT_RECUR_WEIGHT_C) &&
      (type != LIBXSMM_DNN_LSTM_REGULAR_BIAS_I)         && (type != LIBXSMM_DNN_LSTM_GRADIENT_BIAS_I)   &&
      (type != LIBXSMM_DNN_LSTM_REGULAR_BIAS_F)         && (type != LIBXSMM_DNN_LSTM_GRADIENT_BIAS_F)   &&
      (type != LIBXSMM_DNN_LSTM_REGULAR_BIAS_O)         && (type != LIBXSMM_DNN_LSTM_GRADIENT_BIAS_O)   &&
      (type != LIBXSMM_DNN_LSTM_REGULAR_BIAS_C)         && (type != LIBXSMM_DNN_LSTM_GRADIENT_BIAS_C) ) {
    status = LIBXSMM_DNN_ERR_UNKNOWN_TENSOR_TYPE;
    return status;
  }

  if (handle != 0) {
    if ( type == LIBXSMM_DNN_LSTM_REGULAR_INPUT ) {
      handle->xt = 0;
    } else if ( type == LIBXSMM_DNN_LSTM_GRADIENT_INPUT ) {
      handle->djdxt = 0;
    } else if ( type == LIBXSMM_DNN_LSTM_REGULAR_HIDDEN_STATE ) {
      handle->h = 0;
    } else if ( type == LIBXSMM_DNN_LSTM_GRADIENT_HIDDEN_STATE ) {
      handle->djdht = 0;
    } else if ( type == LIBXSMM_DNN_LSTM_REGULAR_WEIGHT_I ) {
      handle->wi = 0;
    } else if ( type == LIBXSMM_DNN_LSTM_GRADIENT_WEIGHT_I ) {
      handle->djdwi = 0;
    } else if ( type == LIBXSMM_DNN_LSTM_REGULAR_WEIGHT_F ) {
      handle->wf = 0;
    } else if ( type == LIBXSMM_DNN_LSTM_GRADIENT_WEIGHT_F ) {
      handle->djdwf = 0;
    } else if ( type == LIBXSMM_DNN_LSTM_REGULAR_WEIGHT_O ) {
      handle->wo = 0;
    } else if ( type == LIBXSMM_DNN_LSTM_GRADIENT_WEIGHT_O ) {
      handle->djdwo = 0;
    } else if ( type == LIBXSMM_DNN_LSTM_REGULAR_WEIGHT_C ) {
      handle->wc = 0;
    } else if ( type == LIBXSMM_DNN_LSTM_GRADIENT_WEIGHT_C ) {
      handle->djdwc = 0;
    } else if ( type == LIBXSMM_DNN_LSTM_REGULAR_RECUR_WEIGHT_I ) {
      handle->ri = 0;
    } else if ( type == LIBXSMM_DNN_LSTM_GRADIENT_RECUR_WEIGHT_I ) {
      handle->djdri = 0;
    } else if ( type == LIBXSMM_DNN_LSTM_REGULAR_RECUR_WEIGHT_F ) {
      handle->rf = 0;
    } else if ( type == LIBXSMM_DNN_LSTM_GRADIENT_RECUR_WEIGHT_F ) {
      handle->djdrf = 0;
    } else if ( type == LIBXSMM_DNN_LSTM_REGULAR_RECUR_WEIGHT_O ) {
      handle->ro = 0;
    } else if ( type == LIBXSMM_DNN_LSTM_GRADIENT_RECUR_WEIGHT_O ) {
      handle->djdro = 0;
    } else if ( type == LIBXSMM_DNN_LSTM_REGULAR_RECUR_WEIGHT_C ) {
      handle->rc = 0;
    } else if ( type == LIBXSMM_DNN_LSTM_GRADIENT_RECUR_WEIGHT_C ) {
      handle->djdrc = 0;
    } else if ( type == LIBXSMM_DNN_LSTM_REGULAR_BIAS_I ) {
      handle->bi = 0;
    } else if ( type == LIBXSMM_DNN_LSTM_GRADIENT_BIAS_I ) {
      handle->djdbi = 0;
    } else if ( type == LIBXSMM_DNN_LSTM_REGULAR_BIAS_F ) {
      handle->bf = 0;
    } else if ( type == LIBXSMM_DNN_LSTM_GRADIENT_BIAS_F ) {
      handle->djdbf = 0;
    } else if ( type == LIBXSMM_DNN_LSTM_REGULAR_BIAS_O ) {
      handle->bo = 0;
    } else if ( type == LIBXSMM_DNN_LSTM_GRADIENT_BIAS_O ) {
      handle->djdbo = 0;
    } else if ( type == LIBXSMM_DNN_LSTM_REGULAR_BIAS_C ) {
      handle->bc = 0;
    } else if ( type == LIBXSMM_DNN_LSTM_GRADIENT_BIAS_C ) {
      handle->djdbc = 0;
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
  libxsmm_blasint m = lstm->m;
  libxsmm_blasint n = lstm->n;
  libxsmm_blasint k = lstm->k;
  libxsmm_blasint t = lstm->t;
#if defined(LSTM_TIMING)
  const double gflops = (((2.0 * m * n * k) + (2.0 * m * n * m) + (2.0 * m * n)) * 4.0 + (5.0 * m * n)) * t * 1E-9;
#endif
  int reuse = 1;
  LIBXSMM_DNN_ELTWISE_FTYPE *wi = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->wi->data;
#if defined(NON_FUSED_INPUT_GEMM)
  LIBXSMM_DNN_ELTWISE_FTYPE *wf = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->wf->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *wo = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->wo->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *wc = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->wc->data;
#endif
  LIBXSMM_DNN_ELTWISE_FTYPE *xt = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->xt->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *ri = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->ri->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *rf = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->rf->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *ro = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->ro->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *rc = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->rc->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *h = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->h->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *i2 = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->i2->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *f2 = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->f2->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *o2 = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->o2->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *c2 = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->c2->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *i = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->i->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *f = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->f->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *o = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->o->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *c = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->c->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *dh = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->dh->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *d1 = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->d1->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *d2 = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->d2->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *d = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->d->data;
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, x, xt, k * n);
#if defined(NON_FUSED_INPUT_GEMM)
  LIBXSMM_DNN_ELTWISE_FTYPE *i1t = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->i1t->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *f1t = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->f1t->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *o1t = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->o1t->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *c1t = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->c1t->data;
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, i1, i1t, m * n);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, f1, f1t, m * n);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, o1, o1t, m * n);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, c1, c1t, m * n);
#else
  LIBXSMM_VLA_DECL(3, LIBXSMM_DNN_ELTWISE_FTYPE, i4,
    (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->i1t->data, t, m * n);
  LIBXSMM_DNN_ELTWISE_FTYPE *i1t = &LIBXSMM_VLA_ACCESS(3, i4, 0, 0, 0, t, m * n);
  LIBXSMM_DNN_ELTWISE_FTYPE *f1t = &LIBXSMM_VLA_ACCESS(3, i4, 1, 0, 0, t, m * n);
  LIBXSMM_DNN_ELTWISE_FTYPE *o1t = &LIBXSMM_VLA_ACCESS(3, i4, 2, 0, 0, t, m * n);
  LIBXSMM_DNN_ELTWISE_FTYPE *c1t = &LIBXSMM_VLA_ACCESS(3, i4, 3, 0, 0, t, m * n);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, i1, i1t, m * n);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, f1, f1t, m * n);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, o1, o1t, m * n);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, c1, c1t, m * n);
#endif
  /* libxsmm_bgemm_handle *handlewx = lstm->handlewx; */
  libxsmm_bgemm_handle *handleuh = lstm->handleuh;
  libxsmm_bgemm_handle *handlett = lstm->handlett;
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, hnr, h, m * n);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, inr, i, m * n);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, fnr, f, m * n);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, onr, o, m * n);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, cnr, c, m * n);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, dnr, d, m * n);
#if defined(LSTM_TIMING)
  unsigned long long start;
  double duration;
  Gbl_t_input_total = 0.; Gbl_t_recur_total = 0.; Gbl_t_eltwise_total = 0.; Gbl_t_nonlin_total = 0.;
  Gbl_t_input = 0; Gbl_t_recur = 0; Gbl_t_eltwise = 0; Gbl_t_nonlin = 0;
  Gbl_duration_input = 0.; Gbl_duration_recur = 0.; Gbl_duration_eltwise = 0.; Gbl_duration_nonlin = 0.;
#endif

  /* int s; */
  int j;

  LIBXSMM_UNUSED(start_thread/* Need to populate this code */);
#if defined(LSTM_TIMING)
  start = libxsmm_timer_tick();
#endif
  if (reuse) {
    /* for (s = 0; s < nrepeat; ++s) { */
#if defined(LSTM_TIMING)
      Gbl_t_input = libxsmm_timer_tick();
#endif
#if defined(NON_FUSED_INPUT_GEMM)
      libxsmm_bgemm(handlett, wi, &LIBXSMM_VLA_ACCESS(2, x, 0, 0, k * n), &LIBXSMM_VLA_ACCESS(2, i1, 0, 0, m * n), tid, lstm->nThreads);
      libxsmm_bgemm(handlett, wf, &LIBXSMM_VLA_ACCESS(2, x, 0, 0, k * n), &LIBXSMM_VLA_ACCESS(2, f1, 0, 0, m * n), tid, lstm->nThreads);
      libxsmm_bgemm(handlett, wo, &LIBXSMM_VLA_ACCESS(2, x, 0, 0, k * n), &LIBXSMM_VLA_ACCESS(2, o1, 0, 0, m * n), tid, lstm->nThreads);
      libxsmm_bgemm(handlett, wc, &LIBXSMM_VLA_ACCESS(2, x, 0, 0, k * n), &LIBXSMM_VLA_ACCESS(2, c1, 0, 0, m * n), tid, lstm->nThreads);
#else
      libxsmm_bgemm(handlett, wi, &LIBXSMM_VLA_ACCESS(2, x, 0, 0, k * n), &LIBXSMM_VLA_ACCESS(3, i4, 0, 0, 0, t, m * n), tid, lstm->nThreads);
#endif
#if defined(LSTM_TIMING)
      Gbl_duration_input = libxsmm_timer_duration(Gbl_t_input, libxsmm_timer_tick());
      Gbl_t_input_total += Gbl_duration_input;
#endif
      for (j = 0; j < t-1; ++j) {
        libxsmm_internal_recursive_step(handleuh, ri, h, i2, &LIBXSMM_VLA_ACCESS(2, i1, j, 0, m * n), i, i, 1, m * n, tid, lstm->nThreads); /*sigmoid*/
        libxsmm_internal_recursive_step(handleuh, rf, h, f2, &LIBXSMM_VLA_ACCESS(2, f1, j, 0, m * n), f, f, 1, m * n, tid, lstm->nThreads); /*sigmoid*/
        libxsmm_internal_recursive_step(handleuh, ro, h, o2, &LIBXSMM_VLA_ACCESS(2, o1, j, 0, m * n), o, o, 1, m * n, tid, lstm->nThreads); /*sigmoid*/
        libxsmm_internal_recursive_step(handleuh, rc, h, c2, &LIBXSMM_VLA_ACCESS(2, c1, j, 0, m * n), c, c, 1, m * n, tid, lstm->nThreads); /*tanh*/
#if defined(LSTM_TIMING)
        Gbl_t_eltwise = libxsmm_timer_tick();
#endif
        libxsmm_internal_matrix_eltwise_mult(m*n, f, d, d1);
        libxsmm_internal_matrix_eltwise_mult(m*n, i, c, d2);
        libxsmm_internal_matrix_add(m*n, d1, d2, d);
#if defined(LSTM_TIMING)
        Gbl_duration_eltwise = libxsmm_timer_duration(Gbl_t_eltwise, libxsmm_timer_tick());
        Gbl_t_eltwise_total += Gbl_duration_eltwise;
        Gbl_t_nonlin = libxsmm_timer_tick();
#endif
        libxsmm_internal_matrix_relu(m*n, d, dh); /*tanh*/
#if defined(LSTM_TIMING)
        Gbl_duration_nonlin = libxsmm_timer_duration(Gbl_t_nonlin, libxsmm_timer_tick());
        Gbl_t_nonlin_total += Gbl_duration_nonlin;
        Gbl_t_eltwise = libxsmm_timer_tick();
#endif
        libxsmm_internal_matrix_eltwise_mult(m*n, o, dh, h);
#if defined(LSTM_TIMING)
        Gbl_duration_eltwise = libxsmm_timer_duration(Gbl_t_eltwise, libxsmm_timer_tick());
        Gbl_t_eltwise_total += Gbl_duration_eltwise;
#endif
      }
      libxsmm_internal_recursive_step(handleuh, ri, h, i2, &LIBXSMM_VLA_ACCESS(2, i1, t-1, 0, m * n), i, i, 1, m * n, tid, lstm->nThreads); /*sigmoid*/
      libxsmm_internal_recursive_step(handleuh, rf, h, f2, &LIBXSMM_VLA_ACCESS(2, f1, t-1, 0, m * n), f, f, 1, m * n, tid, lstm->nThreads); /*sigmoid*/
      libxsmm_internal_recursive_step(handleuh, ro, h, o2, &LIBXSMM_VLA_ACCESS(2, o1, t-1, 0, m * n), o, o, 1, m * n, tid, lstm->nThreads); /*sigmoid*/
      libxsmm_internal_recursive_step(handleuh, rc, h, c2, &LIBXSMM_VLA_ACCESS(2, c1, t-1, 0, m * n), c, c, 1, m * n, tid, lstm->nThreads); /*tanh*/
#if defined(LSTM_TIMING)
      Gbl_t_eltwise = libxsmm_timer_tick();
#endif
      libxsmm_internal_matrix_eltwise_mult(m*n, f, d, d1);
      libxsmm_internal_matrix_eltwise_mult(m*n, i, c, d2);
      libxsmm_internal_matrix_add(m*n, d1, d2, d);
#if defined(LSTM_TIMING)
      Gbl_duration_eltwise = libxsmm_timer_duration(Gbl_t_eltwise, libxsmm_timer_tick());
      Gbl_t_eltwise_total += Gbl_duration_eltwise;
#endif
    /* } */ /* end for nrepeat */
  } else {
    /* for (s = 0; s < nrepeat; ++s) { */
#if defined(LSTM_TIMING)
      Gbl_t_input = libxsmm_timer_tick();
#endif
#if defined(NON_FUSED_INPUT_GEMM)
      libxsmm_bgemm(handlett, wi, &LIBXSMM_VLA_ACCESS(2, x, 0, 0, k * n), &LIBXSMM_VLA_ACCESS(2, i1, 0, 0, m * n), tid, lstm->nThreads);
      libxsmm_bgemm(handlett, wf, &LIBXSMM_VLA_ACCESS(2, x, 0, 0, k * n), &LIBXSMM_VLA_ACCESS(2, f1, 0, 0, m * n), tid, lstm->nThreads);
      libxsmm_bgemm(handlett, wo, &LIBXSMM_VLA_ACCESS(2, x, 0, 0, k * n), &LIBXSMM_VLA_ACCESS(2, o1, 0, 0, m * n), tid, lstm->nThreads);
      libxsmm_bgemm(handlett, wc, &LIBXSMM_VLA_ACCESS(2, x, 0, 0, k * n), &LIBXSMM_VLA_ACCESS(2, c1, 0, 0, m * n), tid, lstm->nThreads);
#else
      libxsmm_bgemm(handlett, wi, &LIBXSMM_VLA_ACCESS(2, x, 0, 0, k * n), &LIBXSMM_VLA_ACCESS(3, i4, 0, 0, 0, t, m * n), tid, lstm->nThreads);
#endif
#if defined(LSTM_TIMING)
      Gbl_duration_input = libxsmm_timer_duration(Gbl_t_input, libxsmm_timer_tick());
      Gbl_t_input_total += Gbl_duration_input;
#endif
      for (j = 0; j < t-1; ++j) {
        libxsmm_internal_recursive_step(handleuh, ri, &LIBXSMM_VLA_ACCESS(2, hnr, j, 0, m * n), i2, &LIBXSMM_VLA_ACCESS(2, i1, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, inr, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, inr, j, 0, m * n), 1, m * n, tid, lstm->nThreads); /*sigmoid*/
        libxsmm_internal_recursive_step(handleuh, rf, &LIBXSMM_VLA_ACCESS(2, hnr, j, 0, m * n), f2, &LIBXSMM_VLA_ACCESS(2, f1, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, fnr, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, fnr, j, 0, m * n), 1, m * n, tid, lstm->nThreads); /*sigmoid*/
        libxsmm_internal_recursive_step(handleuh, ro, &LIBXSMM_VLA_ACCESS(2, hnr, j, 0, m * n), o2, &LIBXSMM_VLA_ACCESS(2, o1, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, onr, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, onr, j, 0, m * n), 1, m * n, tid, lstm->nThreads); /*sigmoid*/
        libxsmm_internal_recursive_step(handleuh, rc, &LIBXSMM_VLA_ACCESS(2, hnr, j, 0, m * n), c2, &LIBXSMM_VLA_ACCESS(2, c1, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, cnr, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, cnr, j, 0, m * n), 1, m * n, tid, lstm->nThreads); /*tanh*/
#if defined(LSTM_TIMING)
        Gbl_t_eltwise = libxsmm_timer_tick();
#endif
        libxsmm_internal_matrix_eltwise_mult(m*n, &LIBXSMM_VLA_ACCESS(2, fnr, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, dnr, j, 0, m * n), d1);
        libxsmm_internal_matrix_eltwise_mult(m*n, &LIBXSMM_VLA_ACCESS(2, inr, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, cnr, j, 0, m * n), d2);
        libxsmm_internal_matrix_add(m*n, d1, d2, &LIBXSMM_VLA_ACCESS(2, dnr, j+1, 0, m * n));
#if defined(LSTM_TIMING)
        Gbl_duration_eltwise = libxsmm_timer_duration(Gbl_t_eltwise, libxsmm_timer_tick());
        Gbl_t_eltwise_total += Gbl_duration_eltwise;
        Gbl_t_nonlin = libxsmm_timer_tick();
#endif
        libxsmm_internal_matrix_relu(m*n, &LIBXSMM_VLA_ACCESS(2, dnr, j+1, 0, m * n), dh); /*tanh*/
#if defined(LSTM_TIMING)
        Gbl_duration_nonlin = libxsmm_timer_duration(Gbl_t_nonlin, libxsmm_timer_tick());
        Gbl_t_nonlin_total += Gbl_duration_nonlin;
        Gbl_t_eltwise = libxsmm_timer_tick();
#endif
        libxsmm_internal_matrix_eltwise_mult(m*n, &LIBXSMM_VLA_ACCESS(2, onr, j, 0, m * n), dh, &LIBXSMM_VLA_ACCESS(2, hnr, j+1, 0, m * n));
#if defined(LSTM_TIMING)
        Gbl_duration_eltwise = libxsmm_timer_duration(Gbl_t_eltwise, libxsmm_timer_tick());
        Gbl_t_eltwise_total += Gbl_duration_eltwise;
#endif
      }
      libxsmm_internal_recursive_step(handleuh, ri, &LIBXSMM_VLA_ACCESS(2, hnr, t-1, 0, m * n), i2, &LIBXSMM_VLA_ACCESS(2, i1, t-1, 0, m * n), &LIBXSMM_VLA_ACCESS(2, inr, t-2, 0, m * n), &LIBXSMM_VLA_ACCESS(2, inr, t-2, 0, m * n), 1, m * n, tid, lstm->nThreads); /*sigmoid*/
      libxsmm_internal_recursive_step(handleuh, rf, &LIBXSMM_VLA_ACCESS(2, hnr, t-1, 0, m * n), f2, &LIBXSMM_VLA_ACCESS(2, f1, t-1, 0, m * n), &LIBXSMM_VLA_ACCESS(2, fnr, t-2, 0, m * n), &LIBXSMM_VLA_ACCESS(2, fnr, t-2, 0, m * n), 1, m * n, tid, lstm->nThreads); /*sigmoid*/
      libxsmm_internal_recursive_step(handleuh, ro, &LIBXSMM_VLA_ACCESS(2, hnr, t-1, 0, m * n), o2, &LIBXSMM_VLA_ACCESS(2, o1, t-1, 0, m * n), &LIBXSMM_VLA_ACCESS(2, onr, t-2, 0, m * n), &LIBXSMM_VLA_ACCESS(2, onr, t-2, 0, m * n), 1, m * n, tid, lstm->nThreads); /*sigmoid*/
      libxsmm_internal_recursive_step(handleuh, rc, &LIBXSMM_VLA_ACCESS(2, hnr, t-1, 0, m * n), c2, &LIBXSMM_VLA_ACCESS(2, c1, t-1, 0, m * n), &LIBXSMM_VLA_ACCESS(2, cnr, t-2, 0, m * n), &LIBXSMM_VLA_ACCESS(2, cnr, t-2, 0, m * n), 1, m * n, tid, lstm->nThreads); /*tanh*/
#if defined(LSTM_TIMING)
      Gbl_t_eltwise = libxsmm_timer_tick();
#endif
      libxsmm_internal_matrix_eltwise_mult(m*n, &LIBXSMM_VLA_ACCESS(2, fnr, t-2, 0, m * n), &LIBXSMM_VLA_ACCESS(2, dnr, t-1, 0, m * n), d1);
      libxsmm_internal_matrix_eltwise_mult(m*n, &LIBXSMM_VLA_ACCESS(2, inr, t-2, 0, m * n), &LIBXSMM_VLA_ACCESS(2, cnr, t-2, 0, m * n), d2);
      libxsmm_internal_matrix_add(m*n, d1, d2, &LIBXSMM_VLA_ACCESS(2, dnr, t-1, 0, m * n));
#if defined(LSTM_TIMING)
      Gbl_duration_eltwise = libxsmm_timer_duration(Gbl_t_eltwise, libxsmm_timer_tick());
      Gbl_t_eltwise_total += Gbl_duration_eltwise;
#endif
    /* } */
  }
#if defined(LSTM_TIMING)
  duration = libxsmm_timer_duration(start, libxsmm_timer_tick());
  if (0 < duration) {
    fprintf(stdout, "\tLIBXSMM: %.1f GFLOPS/s\n", gflops * nrepeat / duration);
  }
  double t_total = Gbl_t_input_total + Gbl_t_recur_total + Gbl_t_eltwise_total + Gbl_t_nonlin_total;
  fprintf(stdout, "Percentage of time spent in input matrix multiplication: %lf\n", Gbl_t_input_total*100.0/t_total);
  fprintf(stdout, "Percentage of time spent in recurrence matrix multiplication: %lf\n", Gbl_t_recur_total*100.0/t_total);
  fprintf(stdout, "Percentage of time spent in element-wise operations: %lf\n", Gbl_t_eltwise_total*100.0/t_total);
  fprintf(stdout, "Percentage of time spent in non-linear operations: %lf\n", Gbl_t_nonlin_total*100.0/t_total);
#endif

  return status;
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_lstmcell_bwd_upd_bu(libxsmm_dnn_lstmcell* lstm, int start_thread, int tid, int pass)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
  libxsmm_blasint m = lstm->m;
  libxsmm_blasint n = lstm->n;
  libxsmm_blasint k = lstm->k;
  libxsmm_blasint t = lstm->t;
#if defined(LSTM_TIMING)
  const double tflops = 12; /* transcendental flops */
  double gflops = m * n; /* delta + delta_out */
  gflops += (6.0 * m * n + tflops * m * n); /* dJdd */
  gflops += (4.0 * m * n); /* dJdc */
  gflops += (4.0 * m * n); /* dJdi */
  gflops += (4.0 * m * n); /* dJdf */
  gflops += (4.0 * m * n + tflops * m * n); /* dJdo */
  double tempflops;
  if (pass == 1 || pass == 3) {
    tempflops += (4.0 * m * k); /* W^T */
    tempflops += (8.0 * m * n * k); /* W^T * dJd{c, i, f, o} */
    tempflops += (3.0 * m * k); /* summation */
    gflops += tempflops;
  }
  tempflops += (4.0 * m * m); /* R^T */
  tempflops += (8.0 * m * n * m); /* R^T * dJd{c, i, f, o} */
  gflops += tempflops;
  gflops *= t; /* for t time steps */
  if (pass == 2 || pass == 3) {
    tempflops = k * n; /* x^T */
    tempflops += (8.0 * m * n * k); /* delta{c, i, f, o} * x^T */
    tempflops *= t; /* for t time steps */
    tempflops += (4.0 * m * k * (t-1)); /* for summation of dJdW{c, i, f, o} */
    gflops += tempflops;
    tempflops = 4.0 * m * n; /* delta^T */
    tempflops += (8.0 * m * n * m); /* delta{c, i, f, o} * delta^T */
    tempflops *= (t - 1); /* for (t - 1) time steps */
    tempflops += (4.0 * m * n * (t-2)); /* for summation of dJdR{c, i, f, o} */
    gflops += tempflops;
    gflops += (4.0 * m * n * (t - 1)); /* delbias */
  }
  gflops *= 1E-9; /* to convert flops to Gflops */
#endif
  LIBXSMM_DNN_ELTWISE_FTYPE *wi = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->wi->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *wf = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->wf->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *wo = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->wo->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *wc = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->wc->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *xt = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->xt->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *ri = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->ri->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *rf = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->rf->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *ro = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->ro->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *rc = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->rc->data;
  /* LIBXSMM_DNN_ELTWISE_FTYPE *ht = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->h->data; */
  LIBXSMM_DNN_ELTWISE_FTYPE *i1 = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->i1t->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *i2 = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->i2->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *i3 = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->i3->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *f1 = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->f1t->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *f2 = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->f2->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *f3 = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->f3->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *o1 = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->o1t->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *o2 = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->o2->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *c1 = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->c1t->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *c2 = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->c2->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *it = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->i->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *ft = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->f->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *ot = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->o->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *ct = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->c->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *d1 = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->d1->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *d2 = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->d2->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *d3 = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->dh->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *d4 = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->d4->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *dt = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->d->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *djdht = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->djdht->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *deltat = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->deltat->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *djddt = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->djddt->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *djdit = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->djdit->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *djdft = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->djdft->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *djdct = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->djdct->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *djdot = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->djdot->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *djdxt = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->djdxt->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *djdwi = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->djdwi->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *djdwf = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->djdwf->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *djdwo = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->djdwo->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *djdwc = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->djdwc->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *djdri = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->djdri->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *djdrf = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->djdrf->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *djdro = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->djdro->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *djdrc = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->djdrc->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *djdbi = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->djdbi->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *djdbf = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->djdbf->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *djdbo = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->djdbo->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *djdbc = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->djdbc->data;
  /*
  LIBXSMM_DNN_ELTWISE_FTYPE *rTp = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->rTp->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *wTp = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->wTp->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *deltaTp = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->deltaTp->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *xTp = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->xTp->data;
  */
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, x, xt, k * n);
  /* LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, h, ht, m * n); */
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, i, it, m * n);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, f, ft, m * n);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, o, ot, m * n);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, c, ct, m * n);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, d, dt, m * n);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, djdh, djdht, m * n);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, delta, deltat, m * n);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, djdd, djddt, m * n);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, djdi, djdit, m * n);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, djdf, djdft, m * n);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, djdo, djdot, m * n);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, djdc, djdct, m * n);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, djdx, djdxt, k * n);
  libxsmm_bgemm_handle *handleud = lstm->handlewx;
  libxsmm_bgemm_handle *handledh = lstm->handleuh;
  libxsmm_bgemm_handle *handledx = lstm->handlett;
  libxsmm_bgemm_handle *handlewd = lstm->handlewd;
#if defined(LSTM_TIMING)
  unsigned long long start;
  double duration;
#endif
  /* int s; */
  int j;

  LIBXSMM_UNUSED(start_thread/* Need to populate this code */);
#if defined(LSTM_TIMING)
  start = libxsmm_timer_tick();
#endif
  /* for (s = 0; s < nrepeat; ++s) { */
    /* compute delta */
    libxsmm_internal_matrix_copy(m * n, &LIBXSMM_VLA_ACCESS(2, djdh, t-1, 0, m * n), &LIBXSMM_VLA_ACCESS(2, delta, t-1, 0, m * n));
    /* compute djdd */
    libxsmm_internal_matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, djdh, t-1, 0, m * n), &LIBXSMM_VLA_ACCESS(2, o, t-1, 0, m * n), d1);
    libxsmm_internal_matrix_tanh_inverse(m * n, &LIBXSMM_VLA_ACCESS(2, d, t-1, 0, m * n), d2);
    libxsmm_internal_matrix_eltwise_mult(m * n, d1, d2, &LIBXSMM_VLA_ACCESS(2, djdd, t-1, 0, m * n));
    /* compute djdc */
    libxsmm_internal_matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, djdd, t-1, 0, m * n), &LIBXSMM_VLA_ACCESS(2, i, t-1, 0, m * n), c1);
    libxsmm_internal_matrix_complement_square(m * n, &LIBXSMM_VLA_ACCESS(2, c, t-1, 0, m * n), c2);
    libxsmm_internal_matrix_eltwise_mult(m * n, c1, c2, &LIBXSMM_VLA_ACCESS(2, djdc, t-1, 0, m * n));
    /* compute djdi */
    libxsmm_internal_matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, djdd, t-1, 0, m * n), &LIBXSMM_VLA_ACCESS(2, c, t-1, 0, m * n), i1);
    libxsmm_internal_matrix_complement(m * n, &LIBXSMM_VLA_ACCESS(2, i, t-1, 0, m * n), i2);
    libxsmm_internal_matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, i, t-1, 0, m * n), i2, i3);
    libxsmm_internal_matrix_eltwise_mult(m * n, i1, i3, &LIBXSMM_VLA_ACCESS(2, djdi, t-1, 0, m * n));
    /* compute djdf */
    libxsmm_internal_matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, djdd, t-1, 0, m * n), &LIBXSMM_VLA_ACCESS(2, d, t-2, 0, m * n), f1);
    libxsmm_internal_matrix_complement(m * n, &LIBXSMM_VLA_ACCESS(2, f, t-1, 0, m * n), f2);
    libxsmm_internal_matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, f, t-1, 0, m * n), f2, f3);
    libxsmm_internal_matrix_eltwise_mult(m * n, f1, f3, &LIBXSMM_VLA_ACCESS(2, djdf, t-1, 0, m * n));
    /* compute djdo */
    libxsmm_internal_matrix_tanh(m * n, &LIBXSMM_VLA_ACCESS(2, d, t-1, 0, m * n), o1);
    libxsmm_internal_matrix_complement(m * n, &LIBXSMM_VLA_ACCESS(2, o, t-1, 0, m * n), o2);
    libxsmm_internal_matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, delta, t-1, 0, m * n), o1, o1);
    libxsmm_internal_matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, o, t-1, 0, m * n), o2, o2);
    libxsmm_internal_matrix_eltwise_mult(m * n, o1, o2, &LIBXSMM_VLA_ACCESS(2, djdo, t-1, 0, m * n));
    if (pass == 1 || pass == 3) {
      /* compute djdx */
      /* libxsmm_internal_matrix_transpose(m, k, wi, wTp); - already taken care of in init */
      /* libxsmm_bgemm(handlewd, wTp, &LIBXSMM_VLA_ACCESS(2, djdi, t-1, 0, m * n), &LIBXSMM_VLA_ACCESS(2, djdx, t-1, 0, k * n), tid, lstm->nThreads); */
      libxsmm_bgemm(handlewd, wi, &LIBXSMM_VLA_ACCESS(2, djdi, t-1, 0, m * n), &LIBXSMM_VLA_ACCESS(2, djdx, t-1, 0, k * n), tid, lstm->nThreads);
      /* libxsmm_internal_matrix_transpose(m, k, wf, wTp); - already taken care of in init */
      /* libxsmm_bgemm(handlewd, wTp, &LIBXSMM_VLA_ACCESS(2, djdf, t-1, 0, m * n), &LIBXSMM_VLA_ACCESS(2, djdx, t-1, 0, k * n), tid, lstm->nThreads); */
      libxsmm_bgemm(handlewd, wf, &LIBXSMM_VLA_ACCESS(2, djdf, t-1, 0, m * n), &LIBXSMM_VLA_ACCESS(2, djdx, t-1, 0, k * n), tid, lstm->nThreads);
      /* libxsmm_internal_matrix_transpose(m, k, wo, wTp); - already taken care of in init */
      /* libxsmm_bgemm(handlewd, wTp, &LIBXSMM_VLA_ACCESS(2, djdo, t-1, 0, m * n), &LIBXSMM_VLA_ACCESS(2, djdx, t-1, 0, k * n), tid, lstm->nThreads); */
      libxsmm_bgemm(handlewd, wo, &LIBXSMM_VLA_ACCESS(2, djdo, t-1, 0, m * n), &LIBXSMM_VLA_ACCESS(2, djdx, t-1, 0, k * n), tid, lstm->nThreads);
      /* libxsmm_internal_matrix_transpose(m, k, wc, wTp); - already taken care of in init */
      /* libxsmm_bgemm(handlewd, wTp, &LIBXSMM_VLA_ACCESS(2, djdc, t-1, 0, m * n), &LIBXSMM_VLA_ACCESS(2, djdx, t-1, 0, k * n), tid, lstm->nThreads); */
      libxsmm_bgemm(handlewd, wc, &LIBXSMM_VLA_ACCESS(2, djdc, t-1, 0, m * n), &LIBXSMM_VLA_ACCESS(2, djdx, t-1, 0, k * n), tid, lstm->nThreads);
    }
    for (j = t-2; j >= 0; --j) {
      /* compute delta */
      /* libxsmm_internal_matrix_transpose(m, m, ri, rTp); - already taken care of in init */
      /* libxsmm_bgemm(handleud, rTp, &LIBXSMM_VLA_ACCESS(2, djdi, j, 0, m * n),  &LIBXSMM_VLA_ACCESS(2, delta, j+1, 0, m * n), tid, lstm->nThreads); */
      libxsmm_bgemm(handleud, ri, &LIBXSMM_VLA_ACCESS(2, djdi, j, 0, m * n),  &LIBXSMM_VLA_ACCESS(2, delta, j+1, 0, m * n), tid, lstm->nThreads);
      /* libxsmm_internal_matrix_transpose(m, m, rf, rTp); - already taken care of in init */
      /* libxsmm_bgemm(handleud, rTp, &LIBXSMM_VLA_ACCESS(2, djdf, j, 0, m * n),  &LIBXSMM_VLA_ACCESS(2, delta, j+1, 0, m * n), tid, lstm->nThreads); */
      libxsmm_bgemm(handleud, rf, &LIBXSMM_VLA_ACCESS(2, djdf, j, 0, m * n),  &LIBXSMM_VLA_ACCESS(2, delta, j+1, 0, m * n), tid, lstm->nThreads);
      /* libxsmm_internal_matrix_transpose(m, m, ro, rTp); - already taken care of in init */
      /* libxsmm_bgemm(handleud, rTp, &LIBXSMM_VLA_ACCESS(2, djdo, j, 0, m * n),  &LIBXSMM_VLA_ACCESS(2, delta, j+1, 0, m * n), tid, lstm->nThreads); */
      libxsmm_bgemm(handleud, ro, &LIBXSMM_VLA_ACCESS(2, djdo, j, 0, m * n),  &LIBXSMM_VLA_ACCESS(2, delta, j+1, 0, m * n), tid, lstm->nThreads);
      /* libxsmm_internal_matrix_transpose(m, m, rc, rTp); - already taken care of in init */
      /* libxsmm_bgemm(handleud, rTp, &LIBXSMM_VLA_ACCESS(2, djdc, j, 0, m * n),  &LIBXSMM_VLA_ACCESS(2, delta, j+1, 0, m * n), tid, lstm->nThreads); */
      libxsmm_bgemm(handleud, rc, &LIBXSMM_VLA_ACCESS(2, djdc, j, 0, m * n),  &LIBXSMM_VLA_ACCESS(2, delta, j+1, 0, m * n), tid, lstm->nThreads);
      libxsmm_internal_matrix_add(m * n, &LIBXSMM_VLA_ACCESS(2, djdh, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, delta, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, delta, j, 0, m * n));
      /* compute djdd */
      libxsmm_internal_matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, djdh, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, o, j, 0, m * n), d1);
      libxsmm_internal_matrix_tanh_inverse(m * n, &LIBXSMM_VLA_ACCESS(2, d, j, 0, m * n), d2);
      libxsmm_internal_matrix_eltwise_mult(m * n, d1, d2, d3);
      libxsmm_internal_matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, delta, j+1, 0, m * n), &LIBXSMM_VLA_ACCESS(2, f, j+1, 0, m * n), d4);
      libxsmm_internal_matrix_add(m * n, d3, d4, &LIBXSMM_VLA_ACCESS(2, djdd, j, 0, m * n));
      /* compute djdc */
      libxsmm_internal_matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, djdd, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, i, j, 0, m * n), c1);
      libxsmm_internal_matrix_complement_square(m * n, &LIBXSMM_VLA_ACCESS(2, c, j, 0, m * n), c2);
      libxsmm_internal_matrix_eltwise_mult(m * n, c1, c2, &LIBXSMM_VLA_ACCESS(2, djdc, j, 0, m * n));
      /* compute djdi */
      libxsmm_internal_matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, djdd, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, c, j, 0, m * n), i1);
      libxsmm_internal_matrix_complement(m * n, &LIBXSMM_VLA_ACCESS(2, i, j, 0, m * n), i2);
      libxsmm_internal_matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, i, j, 0, m * n), i2, i3);
      libxsmm_internal_matrix_eltwise_mult(m * n, i1, i3, &LIBXSMM_VLA_ACCESS(2, djdi, j, 0, m * n));
      /* compute djdf */
      if (j >= 1) {
        libxsmm_internal_matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, djdd, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, d, j-1, 0, m * n), f1);
        libxsmm_internal_matrix_complement(m * n, &LIBXSMM_VLA_ACCESS(2, f, j, 0, m * n), f2);
        libxsmm_internal_matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, f, j, 0, m * n), f2, f3);
        libxsmm_internal_matrix_eltwise_mult(m * n, f1, f3, &LIBXSMM_VLA_ACCESS(2, djdf, j, 0, m * n));
      } else {
        /* djdf is zero for j == 0 */
        libxsmm_internal_matinit( 0, &LIBXSMM_VLA_ACCESS(2, djdf, j, 0, m * n), m, n, m, 0.0);
      }
      /* compute djdo */
      libxsmm_internal_matrix_tanh(m * n, &LIBXSMM_VLA_ACCESS(2, d, j, 0, m * n), o1);
      libxsmm_internal_matrix_complement(m * n, &LIBXSMM_VLA_ACCESS(2, o, j, 0, m * n), o2);
      libxsmm_internal_matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, delta, j, 0, m * n), o1, o1);
      libxsmm_internal_matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, o, j, 0, m * n), o2, o2);
      libxsmm_internal_matrix_eltwise_mult(m * n, o1, o2, &LIBXSMM_VLA_ACCESS(2, djdo, j, 0, m * n));
      if (pass == 1 || pass == 3) {
        /* compute djdx */
        /* libxsmm_internal_matrix_transpose(m, k, wi, wTp); - already taken care of in init */
        /* libxsmm_bgemm(handlewd, wTp, &LIBXSMM_VLA_ACCESS(2, djdi, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, djdx, j, 0, k * n), tid, lstm->nThreads); */
        libxsmm_bgemm(handlewd, wi, &LIBXSMM_VLA_ACCESS(2, djdi, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, djdx, j, 0, k * n), tid, lstm->nThreads);
        /* libxsmm_internal_matrix_transpose(m, k, wf, wTp); - already taken care of in init */
        /* libxsmm_bgemm(handlewd, wTp, &LIBXSMM_VLA_ACCESS(2, djdf, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, djdx, j, 0, k * n), tid, lstm->nThreads); */
        libxsmm_bgemm(handlewd, wf, &LIBXSMM_VLA_ACCESS(2, djdf, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, djdx, j, 0, k * n), tid, lstm->nThreads);
        /* libxsmm_internal_matrix_transpose(m, k, wo, wTp); - already taken care of in init */
        /* libxsmm_bgemm(handlewd, wTp, &LIBXSMM_VLA_ACCESS(2, djdo, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, djdx, j, 0, k * n), tid, lstm->nThreads); */
        libxsmm_bgemm(handlewd, wo, &LIBXSMM_VLA_ACCESS(2, djdo, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, djdx, j, 0, k * n), tid, lstm->nThreads);
        /* libxsmm_internal_matrix_transpose(m, k, wc, wTp); - already taken care of in init */
        /* libxsmm_bgemm(handlewd, wTp, &LIBXSMM_VLA_ACCESS(2, djdc, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, djdx, j, 0, k * n), tid, lstm->nThreads); */
        libxsmm_bgemm(handlewd, wc, &LIBXSMM_VLA_ACCESS(2, djdc, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, djdx, j, 0, k * n), tid, lstm->nThreads);
      }
    }
    if (pass == 2 || pass == 3) {
      /* compute djdw */
      for (j = 0; j < t; ++j) {
        /* libxsmm_internal_matrix_transpose(k, n, &LIBXSMM_VLA_ACCESS(2, x, j, 0, k * n), xTp); - already taken care of in init */
        /*
        libxsmm_bgemm(handledx, &LIBXSMM_VLA_ACCESS(2, djdi, j, 0, m * n), xTp, djdwi, tid, lstm->nThreads);
        libxsmm_bgemm(handledx, &LIBXSMM_VLA_ACCESS(2, djdf, j, 0, m * n), xTp, djdwf, tid, lstm->nThreads);
        libxsmm_bgemm(handledx, &LIBXSMM_VLA_ACCESS(2, djdo, j, 0, m * n), xTp, djdwo, tid, lstm->nThreads);
        libxsmm_bgemm(handledx, &LIBXSMM_VLA_ACCESS(2, djdc, j, 0, m * n), xTp, djdwc, tid, lstm->nThreads);
        */
        libxsmm_bgemm(handledx, &LIBXSMM_VLA_ACCESS(2, djdi, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, x, j, 0, k * n), djdwi, tid, lstm->nThreads);
        libxsmm_bgemm(handledx, &LIBXSMM_VLA_ACCESS(2, djdf, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, x, j, 0, k * n), djdwf, tid, lstm->nThreads);
        libxsmm_bgemm(handledx, &LIBXSMM_VLA_ACCESS(2, djdo, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, x, j, 0, k * n), djdwo, tid, lstm->nThreads);
        libxsmm_bgemm(handledx, &LIBXSMM_VLA_ACCESS(2, djdc, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, x, j, 0, k * n), djdwc, tid, lstm->nThreads);
      }
      /* compute djdr */
      for (j = 0; j < t-1; ++j) {
        /* libxsmm_internal_matrix_transpose(m, n, &LIBXSMM_VLA_ACCESS(2, delta, j, 0, m * n), deltaTp); - already taken care of in init */
        libxsmm_bgemm(handledh, &LIBXSMM_VLA_ACCESS(2, djdi, j+1, 0, m * n), &LIBXSMM_VLA_ACCESS(2, delta, j, 0, m * n), djdri, tid, lstm->nThreads);
        libxsmm_bgemm(handledh, &LIBXSMM_VLA_ACCESS(2, djdf, j+1, 0, m * n), &LIBXSMM_VLA_ACCESS(2, delta, j, 0, m * n), djdrf, tid, lstm->nThreads);
        libxsmm_bgemm(handledh, &LIBXSMM_VLA_ACCESS(2, djdo, j+1, 0, m * n), &LIBXSMM_VLA_ACCESS(2, delta, j, 0, m * n), djdro, tid, lstm->nThreads);
        libxsmm_bgemm(handledh, &LIBXSMM_VLA_ACCESS(2, djdc, j+1, 0, m * n), &LIBXSMM_VLA_ACCESS(2, delta, j, 0, m * n), djdrc, tid, lstm->nThreads);
      }
      /* compute djdb */
      for (j = 0; j < t-1; j++) {
        libxsmm_internal_matrix_add(m * n, &LIBXSMM_VLA_ACCESS(2, djdi, j, 0, m * n), djdbi, djdbi);
        libxsmm_internal_matrix_add(m * n, &LIBXSMM_VLA_ACCESS(2, djdf, j, 0, m * n), djdbf, djdbf);
        libxsmm_internal_matrix_add(m * n, &LIBXSMM_VLA_ACCESS(2, djdo, j, 0, m * n), djdbo, djdbo);
        libxsmm_internal_matrix_add(m * n, &LIBXSMM_VLA_ACCESS(2, djdc, j, 0, m * n), djdbc, djdbc);
      }
    }
  /* } */
#if defined(LSTM_TIMING)
  duration = libxsmm_timer_duration(start, libxsmm_timer_tick());
  if (0 < duration) {
    fprintf(stdout, "\tLIBXSMM: %.1f GFLOPS/s\n", gflops * nrepeat / duration);
  }
#endif

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

