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

/* #define LSTM_TIMING */
#if defined(LSTM_TIMING)
#include <stdio.h>
double Gbl_t_input_total = 0., Gbl_t_recur_total = 0., Gbl_t_eltwise_total = 0., Gbl_t_nonlin_total = 0.;
unsigned long long Gbl_t_input = 0, Gbl_t_recur = 0, Gbl_t_eltwise = 0, Gbl_t_nonlin = 0;
double Gbl_duration_input = 0., Gbl_duration_recur = 0., Gbl_duration_eltwise = 0., Gbl_duration_nonlin = 0.;
#endif


LIBXSMM_API libxsmm_dnn_rnncell* libxsmm_dnn_create_rnncell(libxsmm_dnn_rnncell_desc rnncell_desc, libxsmm_dnn_err_t* status)
{
  libxsmm_dnn_rnncell* handle = 0;
  const char *const env_b_m1 = getenv("LIBXSMM_BGEMM_M1");
  const int b_m1 = (0 == env_b_m1) ? 1 : atoi(env_b_m1);
  const char *const env_b_n1 = getenv("LIBXSMM_BGEMM_N1");
  const int b_n1 = (0 == env_b_n1) ? 1 : atoi(env_b_n1);
  const char *const env_b_k1 = getenv("LIBXSMM_BGEMM_K1");
  const int b_k1 = (0 == env_b_k1) ? 1 : atoi(env_b_k1);
  const char *const env_b_m2 = getenv("LIBXSMM_BGEMM_M2");
  const int b_m2 = (0 == env_b_m2) ? 1 : atoi(env_b_m2);
  const char *const env_b_n2 = getenv("LIBXSMM_BGEMM_N2");
  const int b_n2 = (0 == env_b_n2) ? 1 : atoi(env_b_n2);
  const char *const env_b_k2 = getenv("LIBXSMM_BGEMM_K2");
  const int b_k2 = (0 == env_b_k2) ? 1 : atoi(env_b_k2);
  const char transa = 'N', transb = 'N'; /* no transposes */
  const int gemm_flags = LIBXSMM_GEMM_FLAGS(transa, transb);
  const float alpha = 1, beta = 1;
  const libxsmm_bgemm_order order = (libxsmm_bgemm_order)0; /* denotes order of execution for bgemm */
  const libxsmm_gemm_prefetch_type strategy = (libxsmm_gemm_prefetch_type)LIBXSMM_PREFETCH_AUTO;

  handle = (libxsmm_dnn_rnncell*)malloc(sizeof(libxsmm_dnn_rnncell));
  if (0 != handle) {
    *status = LIBXSMM_DNN_SUCCESS;
    /* zero entire content; not only safer but also sets data and code pointers to NULL */
    memset(handle, 0, sizeof(*handle));
    /* initialize known handle components */
    handle->nThreads = rnncell_desc.nThreads;
    handle->desc = rnncell_desc;
    handle->datatype_in = rnncell_desc.datatype_in;
    handle->datatype_out = rnncell_desc.datatype_out;
    handle->reuse = rnncell_desc.reuse;
    handle->pass = rnncell_desc.pass;
    if ( (rnncell_desc.datatype_in != LIBXSMM_DNN_DATATYPE_F32) || (rnncell_desc.datatype_out != LIBXSMM_DNN_DATATYPE_F32) ) {
      /* error */
      *status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
      return handle;
    }
    handle->buffer_format = rnncell_desc.buffer_format;
    handle->custom_format_type = LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM_1; /* required only for comparing layouts */
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
    handle->b_m1 = b_m1;
    handle->b_n1 = b_n1;
    handle->b_k1 = b_k1;
    handle->b_m2 = b_m2;
    handle->b_n2 = b_n2;
    handle->b_k2 = b_k2;
    if (handle->pass == 0) {
      handle->handlewx = libxsmm_bgemm_handle_create(handle->nThreads, LIBXSMM_GEMM_PRECISION(float), LIBXSMM_GEMM_PRECISION(float),
        handle->m, handle->n, handle->k, &(handle->bm), &(handle->bn), &(handle->bk), &(handle->b_m1), &(handle->b_n1), &(handle->b_k1), &(handle->b_k2),
        &alpha, &beta, &gemm_flags, &strategy, &order);
      handle->handleuh = libxsmm_bgemm_handle_create(handle->nThreads, LIBXSMM_GEMM_PRECISION(float), LIBXSMM_GEMM_PRECISION(float),
        handle->m, handle->n, handle->m, &(handle->bm), &(handle->bn), &(handle->bm), &(handle->b_m1), &(handle->b_n1), &(handle->b_m1), &(handle->b_m2),
        &alpha, &beta, &gemm_flags, &strategy, &order);
      handle->handlett = libxsmm_bgemm_handle_create(handle->nThreads, LIBXSMM_GEMM_PRECISION(float), LIBXSMM_GEMM_PRECISION(float),
        handle->m, handle->n*handle->t, handle->k, &(handle->bm), &(handle->bn), &(handle->bk), &(handle->b_m1), &(handle->b_n1), &(handle->b_k1), &(handle->b_k2),
        &alpha, &beta, &gemm_flags, &strategy, &order);
    } else {
      handle->handlewx = libxsmm_bgemm_handle_create(handle->nThreads, LIBXSMM_GEMM_PRECISION(float), LIBXSMM_GEMM_PRECISION(float),
        handle->m, handle->n, handle->m, &(handle->bm), &(handle->bn), &(handle->bm), &(handle->b_m1), &(handle->b_n1), &(handle->b_m1), &(handle->b_m2),
        &alpha, &beta, &gemm_flags, &strategy, &order); /* U^T*delta */
      handle->handleuh = libxsmm_bgemm_handle_create(handle->nThreads, LIBXSMM_GEMM_PRECISION(float), LIBXSMM_GEMM_PRECISION(float),
        handle->m, handle->m, handle->n, &(handle->bm), &(handle->bm), &(handle->bn), &(handle->b_m1), &(handle->b_m1), &(handle->b_n1), &(handle->b_n2),
        &alpha, &beta, &gemm_flags, &strategy, &order); /* delta*h^T */
      handle->handlett = libxsmm_bgemm_handle_create(handle->nThreads, LIBXSMM_GEMM_PRECISION(float), LIBXSMM_GEMM_PRECISION(float),
        handle->m, handle->k, handle->n, &(handle->bm), &(handle->bk), &(handle->bn), &(handle->b_m1), &(handle->b_k1), &(handle->b_n1), &(handle->b_n2),
        &alpha, &beta, &gemm_flags, &strategy, &order); /* delta*x^T */
      handle->handlewd = libxsmm_bgemm_handle_create(handle->nThreads, LIBXSMM_GEMM_PRECISION(float), LIBXSMM_GEMM_PRECISION(float),
        handle->k, handle->n, handle->m, &(handle->bk), &(handle->bn), &(handle->bm), &(handle->b_k1), &(handle->b_n1), &(handle->b_m1), &(handle->b_m2),
        &alpha, &beta, &gemm_flags, &strategy, &order); /* W^T*delta */
    }
    /* Need to allocate space for scratch libxsmm_dnn_tensor's */
    handle->z   = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->deltat = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->z1t = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->z2  = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->di1 = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->di2 = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->deltaMt = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->barrier = libxsmm_barrier_create(handle->nThreads, 1);
    if (NULL == handle->deltat || NULL == handle->deltaMt ||
        NULL == handle->z || NULL == handle->z1t || NULL == handle->z2 ||
        NULL == handle->di1 || NULL == handle->di2 || NULL == handle->barrier)
    {
      free(handle->deltat); free(handle->deltaMt);
      free(handle->z); free(handle->z1t); free(handle->z2);
      free(handle->di1); free(handle->di2);
      *status = LIBXSMM_DNN_ERR_CREATE_HANDLE;
    }
  } else {
    *status = LIBXSMM_DNN_ERR_CREATE_HANDLE;
  }
  return handle;
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_destroy_rnncell(const libxsmm_dnn_rnncell* handle)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
  if (0 != handle) {
    free(handle->deltat); free(handle->deltaMt);
    free(handle->z); free(handle->z1t); free(handle->z2);
    free(handle->di1); free(handle->di2);
    /* Deallocate barrier */
    if (handle->barrier != 0 ) { libxsmm_barrier_release((const libxsmm_barrier*)handle->barrier); }
    /* deallocate handle structure */
    free(/*remove constness*/(libxsmm_dnn_rnncell*)handle);
  } else {
    status = LIBXSMM_DNN_ERR_INVALID_HANDLE;
  }
  return status;
}


LIBXSMM_API libxsmm_dnn_tensor_datalayout* libxsmm_dnn_rnncell_create_tensor_datalayout(const libxsmm_dnn_rnncell* handle, const libxsmm_dnn_tensor_type type, libxsmm_dnn_err_t* status)
{
  libxsmm_dnn_tensor_datalayout* layout;
  *status = LIBXSMM_DNN_SUCCESS;
  layout = 0;
  if (handle != 0) {
    layout = (libxsmm_dnn_tensor_datalayout*) malloc(sizeof(libxsmm_dnn_tensor_datalayout));
    if (layout != 0) {
      memset(layout, 0, sizeof(libxsmm_dnn_tensor_datalayout));
      layout->custom_format = handle->custom_format_type;
      if ( (type == LIBXSMM_DNN_RNN_REGULAR_INPUT)        || (type == LIBXSMM_DNN_RNN_GRADIENT_INPUT)  ||
           (type == LIBXSMM_DNN_RNN_REGULAR_HIDDEN_STATE) || (type == LIBXSMM_DNN_RNN_GRADIENT_HIDDEN_STATE) ||
           (type == LIBXSMM_DNN_RNN_REGULAR_WEIGHT)       || (type == LIBXSMM_DNN_RNN_GRADIENT_WEIGHT) ||
           (type == LIBXSMM_DNN_RNN_REGULAR_RECUR_WEIGHT) || (type == LIBXSMM_DNN_RNN_GRADIENT_RECUR_WEIGHT) ) {
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
                if ( (type == LIBXSMM_DNN_RNN_REGULAR_INPUT) || (type == LIBXSMM_DNN_RNN_GRADIENT_INPUT) ) {
                  layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_RLK;
                  layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_RLN;
                  layout->dim_type[2] = LIBXSMM_DNN_TENSOR_DIMTYPE_RLK;
                  layout->dim_type[3] = LIBXSMM_DNN_TENSOR_DIMTYPE_RLN;
                  layout->dim_size[0] = handle->bk;
                  layout->dim_size[1] = handle->bn;
                  layout->dim_size[2] = handle->k / handle->bk;
                  layout->dim_size[3] = handle->n / handle->bn;
                } else if ( (type == LIBXSMM_DNN_RNN_REGULAR_HIDDEN_STATE) || (type == LIBXSMM_DNN_RNN_GRADIENT_HIDDEN_STATE) ) {
                  layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_RLM;
                  layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_RLN;
                  layout->dim_type[2] = LIBXSMM_DNN_TENSOR_DIMTYPE_RLM;
                  layout->dim_type[3] = LIBXSMM_DNN_TENSOR_DIMTYPE_RLN;
                  layout->dim_size[0] = handle->bm;
                  layout->dim_size[1] = handle->bn;
                  layout->dim_size[2] = handle->m / handle->bm;
                  layout->dim_size[3] = handle->n / handle->bn;
                } else if ( (type == LIBXSMM_DNN_RNN_REGULAR_WEIGHT) || (type == LIBXSMM_DNN_RNN_GRADIENT_WEIGHT) ) {
                  layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_RLM;
                  layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_RLK;
                  layout->dim_type[2] = LIBXSMM_DNN_TENSOR_DIMTYPE_RLK;
                  layout->dim_type[3] = LIBXSMM_DNN_TENSOR_DIMTYPE_RLM;
                  layout->dim_size[0] = handle->bm;
                  layout->dim_size[1] = handle->bk;
                  layout->dim_size[2] = handle->k / handle->bk;
                  layout->dim_size[3] = handle->m / handle->bm;
                } else if ( (type == LIBXSMM_DNN_RNN_REGULAR_RECUR_WEIGHT) || (type == LIBXSMM_DNN_RNN_GRADIENT_RECUR_WEIGHT) ) {
                  layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_RLM;
                  layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_RLM;
                  layout->dim_type[2] = LIBXSMM_DNN_TENSOR_DIMTYPE_RLM;
                  layout->dim_type[3] = LIBXSMM_DNN_TENSOR_DIMTYPE_RLM;
                  layout->dim_size[0] = handle->bm;
                  layout->dim_size[1] = handle->bm;
                  layout->dim_size[2] = handle->m / handle->bm;
                  layout->dim_size[3] = handle->m / handle->bm;
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


LIBXSMM_API size_t libxsmm_dnn_rnncell_get_scratch_size(const libxsmm_dnn_rnncell* handle, const libxsmm_dnn_compute_kind kind, libxsmm_dnn_err_t* status)
{
  size_t sizeof_datatype = sizeof(float);
  size_t size = 0;
  *status = LIBXSMM_DNN_SUCCESS;

  if (0 != handle) {
    switch (kind) {
      case LIBXSMM_DNN_COMPUTE_KIND_FWD: {
                                           size += handle->m * handle->n * sizeof_datatype * handle->t; /* z1t */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype * handle->t; /* z2 */
                                           size += 64;
                                         } break;
      case LIBXSMM_DNN_COMPUTE_KIND_BWD:
      case LIBXSMM_DNN_COMPUTE_KIND_UPD:
      case LIBXSMM_DNN_COMPUTE_KIND_ALL: {
                                           size += handle->m * handle->n * sizeof_datatype * handle->t; /* z1t */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype; /* z2, zi */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype * handle->t; /* deltat */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype * handle->t; /* di1 */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype; /* di2 */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype * handle->t; /* deltaMt */
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


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_rnncell_bind_scratch(libxsmm_dnn_rnncell* handle, const libxsmm_dnn_compute_kind kind, const void* scratch)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
  uintptr_t address = (uintptr_t)scratch;
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
                                           scratch_size = handle->m * handle->n * sizeof_datatype * handle->t;
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
                                             handle->deltaMt->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->deltaMt->data = (void*)(address+offset);
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


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_rnncell_release_scratch(libxsmm_dnn_rnncell* handle, const libxsmm_dnn_compute_kind kind)
{
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
                                           handle->deltaMt->data = 0;
                                           handle->z1t = 0;
                                           handle->z2 = 0;
                                           handle->deltat = 0;
                                           handle->di1 = 0;
                                           handle->di2 = 0;
                                           handle->deltaMt = 0;
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

LIBXSMM_API size_t libxsmm_dnn_rnncell_get_internalstate_size(const libxsmm_dnn_rnncell* handle, const libxsmm_dnn_compute_kind kind, libxsmm_dnn_err_t* status)
{
  size_t sizeof_datatype = sizeof(float);
  size_t size = 0;
  *status = LIBXSMM_DNN_SUCCESS;

  if (0 != handle) {
    switch (kind) {
      case LIBXSMM_DNN_COMPUTE_KIND_FWD: {
                                           size += handle->m * handle->n * sizeof_datatype * handle->t; /* zt */
                                           size += 64;
                                         } break;
      case LIBXSMM_DNN_COMPUTE_KIND_BWD:
      case LIBXSMM_DNN_COMPUTE_KIND_UPD:
      case LIBXSMM_DNN_COMPUTE_KIND_ALL: {
                                           size += handle->m * handle->n * sizeof_datatype * handle->t; /* zt */
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


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_rnncell_bind_internalstate(libxsmm_dnn_rnncell* handle, const libxsmm_dnn_compute_kind kind, const void* internalstate)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
  uintptr_t address = (uintptr_t)internalstate;
  size_t offset = 0;

  if (internalstate == 0) {
    status = LIBXSMM_DNN_ERR_SCRATCH_NOT_ALLOCED;
    return status;
  }

  if (0 != handle) {
    switch (kind) {
      case LIBXSMM_DNN_COMPUTE_KIND_FWD: {
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


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_rnncell_release_internalstate(libxsmm_dnn_rnncell* handle, const libxsmm_dnn_compute_kind kind)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  if (0 != handle) {
    switch (kind) {
      case LIBXSMM_DNN_COMPUTE_KIND_FWD: {
                                           handle->z->data = 0;
                                           handle->z = 0;
                                         } break;
      case LIBXSMM_DNN_COMPUTE_KIND_BWD:
      case LIBXSMM_DNN_COMPUTE_KIND_UPD:
      case LIBXSMM_DNN_COMPUTE_KIND_ALL: {
                                           handle->z->data = 0;
                                           handle->z = 0;
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


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_rnncell_assign_internalstate(libxsmm_dnn_rnncell* handle, const void* zgoldtb)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  if (handle != 0 && zgoldtb != 0) {
    const libxsmm_blasint m = handle->m, n = handle->n, t = handle->t;
    LIBXSMM_VLA_DECL(2, const LIBXSMM_DNN_ELTWISE_FTYPE, zgold, (const LIBXSMM_DNN_ELTWISE_FTYPE*)zgoldtb, m * n);
    LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, z, (LIBXSMM_DNN_ELTWISE_FTYPE*)handle->z->data, m * n);
    libxsmm_blasint it;
    for (it = 0; it < t; ++it) {
      libxsmm_bgemm_copyin_b(handle->handlewx, &LIBXSMM_VLA_ACCESS(2, zgold, it, 0, m * n), &m, &LIBXSMM_VLA_ACCESS(2, z, it, 0, m * n));
    }
  } else {
    status = LIBXSMM_DNN_ERR_INVALID_HANDLE_TENSOR;
  }

  return status;
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_rnncell_bind_tensor(libxsmm_dnn_rnncell* handle, const libxsmm_dnn_tensor* tensor, const libxsmm_dnn_tensor_type type)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  /* check for tensor type */
  if ( (type != LIBXSMM_DNN_RNN_REGULAR_INPUT)       && (type != LIBXSMM_DNN_RNN_GRADIENT_INPUT)  &&
      (type != LIBXSMM_DNN_RNN_REGULAR_HIDDEN_STATE) && (type != LIBXSMM_DNN_RNN_GRADIENT_HIDDEN_STATE) &&
      (type != LIBXSMM_DNN_RNN_REGULAR_WEIGHT)       && (type != LIBXSMM_DNN_RNN_GRADIENT_WEIGHT) &&
      (type != LIBXSMM_DNN_RNN_REGULAR_RECUR_WEIGHT) && (type != LIBXSMM_DNN_RNN_GRADIENT_RECUR_WEIGHT) ) {
    status = LIBXSMM_DNN_ERR_UNKNOWN_TENSOR_TYPE;
    return status;
  }

  if (handle != 0 && tensor != 0) {
    libxsmm_dnn_tensor_datalayout* handle_layout = libxsmm_dnn_rnncell_create_tensor_datalayout(handle, type, &status);

    if ( libxsmm_dnn_compare_tensor_datalayout(handle_layout, tensor->layout, &status) == 0 ) {
      if ( type == LIBXSMM_DNN_RNN_REGULAR_INPUT ) {
        handle->xt = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_RNN_GRADIENT_INPUT ) {
        handle->djdxt = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_RNN_REGULAR_HIDDEN_STATE ) {
        handle->h = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_RNN_GRADIENT_HIDDEN_STATE ) {
        handle->djdht = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_RNN_REGULAR_WEIGHT ) {
        handle->w = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_RNN_GRADIENT_WEIGHT ) {
        handle->djdw = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_RNN_REGULAR_RECUR_WEIGHT ) {
        handle->u = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_RNN_GRADIENT_RECUR_WEIGHT ) {
        handle->djdu = (libxsmm_dnn_tensor*)tensor;
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


LIBXSMM_API libxsmm_dnn_tensor* libxsmm_dnn_rnncell_get_tensor(libxsmm_dnn_rnncell* handle, const libxsmm_dnn_tensor_type type, libxsmm_dnn_err_t* status)
{
  libxsmm_dnn_tensor* tensor = 0;
  LIBXSMM_UNUSED(status/*TODO*/);

  /* check for tensor type */
  if ( (type != LIBXSMM_DNN_RNN_REGULAR_INPUT)       && (type != LIBXSMM_DNN_RNN_GRADIENT_INPUT)  &&
      (type != LIBXSMM_DNN_RNN_REGULAR_HIDDEN_STATE) && (type != LIBXSMM_DNN_RNN_GRADIENT_HIDDEN_STATE) &&
      (type != LIBXSMM_DNN_RNN_REGULAR_WEIGHT)       && (type != LIBXSMM_DNN_RNN_GRADIENT_WEIGHT) &&
      (type != LIBXSMM_DNN_RNN_REGULAR_RECUR_WEIGHT) && (type != LIBXSMM_DNN_RNN_GRADIENT_RECUR_WEIGHT) ) {
    return tensor;
  }

  if (handle != 0) {
    if ( type == LIBXSMM_DNN_RNN_REGULAR_INPUT ) {
      tensor = handle->xt;
    } else if ( type == LIBXSMM_DNN_RNN_GRADIENT_INPUT ) {
      tensor = handle->djdxt;
    } else if ( type == LIBXSMM_DNN_RNN_REGULAR_HIDDEN_STATE ) {
      tensor = handle->h;
    } else if ( type == LIBXSMM_DNN_RNN_GRADIENT_HIDDEN_STATE ) {
      tensor = handle->djdht;
    } else if ( type == LIBXSMM_DNN_RNN_REGULAR_WEIGHT ) {
      tensor = handle->w;
    } else if ( type == LIBXSMM_DNN_RNN_GRADIENT_WEIGHT ) {
      tensor = handle->djdw;
    } else if ( type == LIBXSMM_DNN_RNN_REGULAR_RECUR_WEIGHT ) {
      tensor = handle->u;
    } else if ( type == LIBXSMM_DNN_RNN_GRADIENT_RECUR_WEIGHT ) {
      tensor = handle->djdu;
    } else {
      /* cannot happen */
    }
  }

  return tensor;
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_rnncell_release_tensor(libxsmm_dnn_rnncell* handle, const libxsmm_dnn_tensor_type type)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  /* check for tensor type */
  if ( (type != LIBXSMM_DNN_RNN_REGULAR_INPUT)       && (type != LIBXSMM_DNN_RNN_GRADIENT_INPUT)  &&
      (type != LIBXSMM_DNN_RNN_REGULAR_HIDDEN_STATE) && (type != LIBXSMM_DNN_RNN_GRADIENT_HIDDEN_STATE) &&
      (type != LIBXSMM_DNN_RNN_REGULAR_WEIGHT)       && (type != LIBXSMM_DNN_RNN_GRADIENT_WEIGHT) &&
      (type != LIBXSMM_DNN_RNN_REGULAR_RECUR_WEIGHT) && (type != LIBXSMM_DNN_RNN_GRADIENT_RECUR_WEIGHT) ) {
    status = LIBXSMM_DNN_ERR_UNKNOWN_TENSOR_TYPE;
    return status;
  }

  if (handle != 0) {
    if ( type == LIBXSMM_DNN_RNN_REGULAR_INPUT ) {
      handle->xt = 0;
    } else if ( type == LIBXSMM_DNN_RNN_GRADIENT_INPUT ) {
      handle->djdxt = 0;
    } else if ( type == LIBXSMM_DNN_RNN_REGULAR_HIDDEN_STATE ) {
      handle->h = 0;
    } else if ( type == LIBXSMM_DNN_RNN_GRADIENT_HIDDEN_STATE ) {
      handle->djdht = 0;
    } else if ( type == LIBXSMM_DNN_RNN_REGULAR_WEIGHT ) {
      handle->w = 0;
    } else if ( type == LIBXSMM_DNN_RNN_GRADIENT_WEIGHT ) {
      handle->djdw = 0;
    } else if ( type == LIBXSMM_DNN_RNN_REGULAR_RECUR_WEIGHT ) {
      handle->u = 0;
    } else if ( type == LIBXSMM_DNN_RNN_GRADIENT_RECUR_WEIGHT ) {
      handle->djdu = 0;
    } else {
      /* cannot happen */
    }
  }
  else {
    status = LIBXSMM_DNN_ERR_INVALID_HANDLE_TENSOR;
  }

  return status;
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_rnncell_fwd(libxsmm_dnn_rnncell* rnn, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
  libxsmm_blasint m = rnn->m;
  libxsmm_blasint n = rnn->n;
  libxsmm_blasint k = rnn->k;
  libxsmm_blasint t = rnn->t;
#if defined(LSTM_TIMING)
  const double tflops = 12;
  const double gflops = ((2.0 * m * n * k) + (2.0 * m * n * m) + (m * n) + (tflops * m * n)) * (double)t * 1E-9;
#endif
  int reuse = rnn->reuse;
  /* The following code should be in template */
  LIBXSMM_DNN_ELTWISE_FTYPE *w = (LIBXSMM_DNN_ELTWISE_FTYPE*)rnn->w->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *xt = (LIBXSMM_DNN_ELTWISE_FTYPE*)rnn->xt->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *u = (LIBXSMM_DNN_ELTWISE_FTYPE*)rnn->u->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *h = (LIBXSMM_DNN_ELTWISE_FTYPE*)rnn->h->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *z1t = (LIBXSMM_DNN_ELTWISE_FTYPE*)rnn->z1t->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *z2t = (LIBXSMM_DNN_ELTWISE_FTYPE*)rnn->z2->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *z = (LIBXSMM_DNN_ELTWISE_FTYPE*)rnn->z->data;
  /* libxsmm_bgemm_handle *handlewx = rnn->handlewx; */
  libxsmm_bgemm_handle *handleuh = rnn->handleuh;
  libxsmm_bgemm_handle *handlett = rnn->handlett;
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, x, xt, k * n);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, z1, z1t, m * n);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, z2, z2t, m * n);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, hnr, h, m * n);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, znr, z, m * n);
#if defined(LSTM_TIMING)
  unsigned long long start;
  double duration;
  Gbl_t_input_total = 0.; Gbl_t_recur_total = 0.; Gbl_t_eltwise_total = 0.; Gbl_t_nonlin_total = 0.;
  Gbl_t_input = 0; Gbl_t_recur = 0; Gbl_t_eltwise = 0; Gbl_t_nonlin = 0;
  Gbl_duration_input = 0.; Gbl_duration_recur = 0.; Gbl_duration_eltwise = 0.; Gbl_duration_nonlin = 0.;
#endif
  int i;
  const int ltid = tid - start_thread;

  libxsmm_barrier_init(rnn->barrier, ltid);
#if defined(LSTM_TIMING)
  if (ltid == 0) {
    start = libxsmm_timer_tick();
    Gbl_t_input = libxsmm_timer_tick();
  }
#endif
  libxsmm_bgemm_st(handlett, w, &LIBXSMM_VLA_ACCESS(2, x, 0, 0, k * n), &LIBXSMM_VLA_ACCESS(2, z1, 0, 0, m * n), start_thread, tid);
#if defined(LSTM_TIMING)
  if (ltid == 0) {
    Gbl_duration_input = libxsmm_timer_duration(Gbl_t_input, libxsmm_timer_tick());
    Gbl_t_input_total += Gbl_duration_input;
  }
#endif

  if (reuse) {
    for (i = 0; i < t; ++i) {
      libxsmm_internal_recursive_step(handleuh, u, h, &LIBXSMM_VLA_ACCESS(2, z2, i, 0, m * n), &LIBXSMM_VLA_ACCESS(2, z1, i, 0, m * n), z, h, 2, m * n, start_thread, tid); /*sigmoid*/
      libxsmm_barrier_wait(rnn->barrier, ltid);
    }
  } else {
    for (i = 0; i < t; ++i) {
      libxsmm_internal_recursive_step(handleuh, u, &LIBXSMM_VLA_ACCESS(2, hnr, i, 0, m * n), &LIBXSMM_VLA_ACCESS(2, z2, i, 0, m * n), &LIBXSMM_VLA_ACCESS(2, z1, i, 0, m * n),
        &LIBXSMM_VLA_ACCESS(2, znr, i, 0, m * n), &LIBXSMM_VLA_ACCESS(2, hnr, i+1, 0, m * n), 2, m * n, start_thread, tid); /*sigmoid*/
      libxsmm_barrier_wait(rnn->barrier, ltid);
    }
  }
#if defined(LSTM_TIMING)
  if (ltid == 0) {
    duration = libxsmm_timer_duration(start, libxsmm_timer_tick());
    if (0 < duration) {
      fprintf(stdout, "\tLIBXSMM: %.1f GFLOPS/s\n", gflops / duration); /* *nrepeat */
      double t_total = Gbl_t_input_total + Gbl_t_recur_total + Gbl_t_eltwise_total + Gbl_t_nonlin_total;
      fprintf(stdout, "Percentage of time spent in input matrix multiplication: %lf\n", Gbl_t_input_total*100.0/t_total);
      fprintf(stdout, "Percentage of time spent in recurrence matrix multiplication: %lf\n", Gbl_t_recur_total*100.0/t_total);
      fprintf(stdout, "Percentage of time spent in element-wise operations: %lf\n", Gbl_t_eltwise_total*100.0/t_total);
      fprintf(stdout, "Percentage of time spent in non-linear operations: %lf\n", Gbl_t_nonlin_total*100.0/t_total);
    }
  }
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
  LIBXSMM_DNN_ELTWISE_FTYPE *djdht = (LIBXSMM_DNN_ELTWISE_FTYPE*)rnn->djdht->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *zt = (LIBXSMM_DNN_ELTWISE_FTYPE*)rnn->z->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *deltat = (LIBXSMM_DNN_ELTWISE_FTYPE*)rnn->deltat->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *u = (LIBXSMM_DNN_ELTWISE_FTYPE*)rnn->u->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *xt = (LIBXSMM_DNN_ELTWISE_FTYPE*)rnn->xt->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *ht = (LIBXSMM_DNN_ELTWISE_FTYPE*)rnn->h->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *w = (LIBXSMM_DNN_ELTWISE_FTYPE*)rnn->w->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *djdu = (LIBXSMM_DNN_ELTWISE_FTYPE*)rnn->djdu->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *djdw = (LIBXSMM_DNN_ELTWISE_FTYPE*)rnn->djdw->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *djdxt = (LIBXSMM_DNN_ELTWISE_FTYPE*)rnn->djdxt->data;
  LIBXSMM_DNN_ELTWISE_FTYPE* zi = (LIBXSMM_DNN_ELTWISE_FTYPE*)rnn->z1t->data;
  LIBXSMM_DNN_ELTWISE_FTYPE* di1t = (LIBXSMM_DNN_ELTWISE_FTYPE*)rnn->di1->data;
  LIBXSMM_DNN_ELTWISE_FTYPE* di2 = (LIBXSMM_DNN_ELTWISE_FTYPE*)rnn->di2->data;
  LIBXSMM_DNN_ELTWISE_FTYPE* deltaMt = (LIBXSMM_DNN_ELTWISE_FTYPE*)rnn->deltaMt->data;
  libxsmm_bgemm_handle *handleud = rnn->handlewx;
  libxsmm_bgemm_handle *handledh = rnn->handleuh;
  libxsmm_bgemm_handle *handledx = rnn->handlett;
  libxsmm_bgemm_handle *handlewd = rnn->handlewd;
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, djdh, djdht, m * n);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, z, zt, m * n);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, delta, deltat, m * n);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, x, xt, k * n);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, h, ht, m * n);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, djdx, djdxt, k * n);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, di1, di1t, m * n);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, deltaM, deltaMt, m * n);

  int i;
  const int ltid = tid - start_thread;

  libxsmm_barrier_init(rnn->barrier, ltid);
  libxsmm_internal_matrix_zero(m * n, &LIBXSMM_VLA_ACCESS(2, delta, t-1, 0, m * n), start_thread, tid, rnn->nThreads);
  if (rnn->nThreads > 1) { libxsmm_barrier_wait(rnn->barrier, ltid); }
  for (i = t-2; i >= 0; --i) {
    libxsmm_internal_matrix_sigmoid_inverse(m * n, &LIBXSMM_VLA_ACCESS(2, z, i+1, 0, m * n), zi, start_thread, tid, rnn->nThreads);
    if (rnn->nThreads > 1) { libxsmm_barrier_wait(rnn->barrier, ltid); }
    libxsmm_bgemm_st(handleud, u, &LIBXSMM_VLA_ACCESS(2, delta, i+1, 0, m * n), &LIBXSMM_VLA_ACCESS(2, di1, i, 0, m * n), start_thread, tid);
    libxsmm_internal_matrix_add(m * n, &LIBXSMM_VLA_ACCESS(2, djdh, i+1, 0, m * n), &LIBXSMM_VLA_ACCESS(2, di1, i, 0, m * n), di2, start_thread, tid, rnn->nThreads);
    libxsmm_internal_matrix_eltwise_mult(m * n, zi, di2, &LIBXSMM_VLA_ACCESS(2, delta, i, 0, m * n), start_thread, tid, rnn->nThreads);
    if (rnn->nThreads > 1) { libxsmm_barrier_wait(rnn->barrier, ltid); }
  }
  if (pass == 1 || pass == 3) {
    for (i = 0; i < t; ++i) {
      libxsmm_bgemm_st(handlewd, w, &LIBXSMM_VLA_ACCESS(2, delta, i, 0, m * n), &LIBXSMM_VLA_ACCESS(2, djdx, i, 0, k * n), start_thread, tid);
    }
  }
  if (pass == 2 || pass == 3) {
    for (i = 0; i < t; ++i) {
      libxsmm_bgemm_convert_b_to_a(handleud, &LIBXSMM_VLA_ACCESS(2, delta, i, 0, m * n), &m, &LIBXSMM_VLA_ACCESS(2, deltaM, i, 0, m * n));
    }
    if (rnn->nThreads > 1) { libxsmm_barrier_wait(rnn->barrier, ltid); }
    for (i = 0; i < t; ++i) {
      libxsmm_bgemm_st(handledh, &LIBXSMM_VLA_ACCESS(2, deltaM, i, 0, m * n), &LIBXSMM_VLA_ACCESS(2, h, i, 0, m * n), djdu, start_thread, tid);
      libxsmm_bgemm_st(handledx, &LIBXSMM_VLA_ACCESS(2, deltaM, i, 0, m * n), &LIBXSMM_VLA_ACCESS(2, x, i, 0, k * m), djdw, start_thread, tid);
    }
  }

  return status;
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_rnncell_execute_st(libxsmm_dnn_rnncell* handle, libxsmm_dnn_compute_kind kind,
  /*unsigned*/int start_thread, /*unsigned*/int tid)
{
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
