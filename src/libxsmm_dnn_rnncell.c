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

#include <libxsmm.h>

#include "libxsmm_dnn_elementwise.h"
#include "libxsmm_main.h"

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#include <math.h>
#include <string.h>
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

LIBXSMM_API libxsmm_dnn_rnncell* libxsmm_dnn_create_rnncell(libxsmm_dnn_rnncell_desc rnncell_desc, libxsmm_dnn_err_t* status)
{
  libxsmm_dnn_rnncell* handle = 0;
# if 0
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
#endif

  handle = (libxsmm_dnn_rnncell*)malloc(sizeof(libxsmm_dnn_rnncell));
  if (0 != handle) {
    *status = LIBXSMM_DNN_SUCCESS;
    /* zero entire content; not only safer but also sets data and code pointers to NULL */
    memset(handle, 0, sizeof(*handle));
    /* initialize known handle components */
    handle->desc = rnncell_desc;
    if ( (rnncell_desc.datatype_in != LIBXSMM_DNN_DATATYPE_F32) || (rnncell_desc.datatype_out != LIBXSMM_DNN_DATATYPE_F32) ) {
      /* error */
      *status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
      return handle;
    }
    if (rnncell_desc.t < 1) {
      *status = LIBXSMM_DNN_ERR_TIME_STEPS_TOO_SMALL;
    }
    handle->bk = 64; /* rnncell_desc.bk; */
    handle->bn = 64; /* rnncell_desc.bn; */
    handle->bc = 64; /* rnncell_desc.bc; */
#if 0
    handle->b_m1 = b_m1;
    handle->b_n1 = b_n1;
    handle->b_k1 = b_k1;
    handle->b_m2 = b_m2;
    handle->b_n2 = b_n2;
    handle->b_k2 = b_k2;
    if (handle->pass == 0) {
      handle->handlewx = libxsmm_bgemm_handle_create(handle->desc.NThreads, LIBXSMM_GEMM_PRECISION(float), LIBXSMM_GEMM_PRECISION(float),
        handle->m, handle->desc.N, handle->desc.K, &(handle->bm), &(handle->bn), &(handle->bk), &(handle->b_m1), &(handle->b_n1), &(handle->b_k1), &(handle->b_k2),
        &alpha, &beta, &gemm_flags, &strategy, &order);
      handle->handleuh = libxsmm_bgemm_handle_create(handle->desc.NThreads, LIBXSMM_GEMM_PRECISION(float), LIBXSMM_GEMM_PRECISION(float),
        handle->m, handle->desc.N, handle->m, &(handle->bm), &(handle->bn), &(handle->bm), &(handle->b_m1), &(handle->b_n1), &(handle->b_m1), &(handle->b_m2),
        &alpha, &beta, &gemm_flags, &strategy, &order);
      handle->handlett = libxsmm_bgemm_handle_create(handle->desc.NThreads, LIBXSMM_GEMM_PRECISION(float), LIBXSMM_GEMM_PRECISION(float),
        handle->m, handle->desc.N*handle->t, handle->desc.K, &(handle->bm), &(handle->bn), &(handle->bk), &(handle->b_m1), &(handle->b_n1), &(handle->b_k1), &(handle->b_k2),
        &alpha, &beta, &gemm_flags, &strategy, &order);
    } else {
      handle->handlewx = libxsmm_bgemm_handle_create(handle->desc.NThreads, LIBXSMM_GEMM_PRECISION(float), LIBXSMM_GEMM_PRECISION(float),
        handle->m, handle->desc.N, handle->m, &(handle->bm), &(handle->bn), &(handle->bm), &(handle->b_m1), &(handle->b_n1), &(handle->b_m1), &(handle->b_m2),
        &alpha, &beta, &gemm_flags, &strategy, &order); /* U^T*delta */
      handle->handleuh = libxsmm_bgemm_handle_create(handle->desc.NThreads, LIBXSMM_GEMM_PRECISION(float), LIBXSMM_GEMM_PRECISION(float),
        handle->m, handle->m, handle->desc.N, &(handle->bm), &(handle->bm), &(handle->bn), &(handle->b_m1), &(handle->b_m1), &(handle->b_n1), &(handle->b_n2),
        &alpha, &beta, &gemm_flags, &strategy, &order); /* delta*h^T */
      handle->handlett = libxsmm_bgemm_handle_create(handle->desc.NThreads, LIBXSMM_GEMM_PRECISION(float), LIBXSMM_GEMM_PRECISION(float),
        handle->m, handle->desc.K, handle->desc.N, &(handle->bm), &(handle->bk), &(handle->bn), &(handle->b_m1), &(handle->b_k1), &(handle->b_n1), &(handle->b_n2),
        &alpha, &beta, &gemm_flags, &strategy, &order); /* delta*x^T */
      handle->handlewd = libxsmm_bgemm_handle_create(handle->desc.NThreads, LIBXSMM_GEMM_PRECISION(float), LIBXSMM_GEMM_PRECISION(float),
        handle->desc.K, handle->desc.N, handle->m, &(handle->bk), &(handle->bn), &(handle->bm), &(handle->b_k1), &(handle->b_n1), &(handle->b_m1), &(handle->b_m2),
        &alpha, &beta, &gemm_flags, &strategy, &order); /* W^T*delta */
    }
#endif
    /* Need to allocate space for scratch libxsmm_dnn_tensor's */
    handle->z   = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->bM  = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->deltat = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->z1  = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->z2  = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->di1 = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->di2 = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->deltaMt = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->barrier = libxsmm_barrier_create(handle->desc.nThreads, 1);
    if (NULL == handle->deltat || NULL == handle->deltaMt || NULL == handle->bM ||
        NULL == handle->z || NULL == handle->z1 || NULL == handle->z2 ||
        NULL == handle->di1 || NULL == handle->di2 || NULL == handle->barrier)
    {
      free(handle->deltat); free(handle->deltaMt); free(handle->bM);
      free(handle->z); free(handle->z1); free(handle->z2);
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
    free(handle->deltat); free(handle->deltaMt); free(handle->bM);
    free(handle->z); free(handle->z1); free(handle->z2);
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
      if ( (type == LIBXSMM_DNN_RNN_REGULAR_INPUT) || (type == LIBXSMM_DNN_RNN_GRADIENT_INPUT) ||
           (type == LIBXSMM_DNN_RNN_REGULAR_HIDDEN_STATE) || (type == LIBXSMM_DNN_RNN_GRADIENT_HIDDEN_STATE) ) {
        layout->format = handle->desc.buffer_format;
        layout->tensor_type = LIBXSMM_DNN_ACTIVATION;
        if ((handle->desc.buffer_format & LIBXSMM_DNN_TENSOR_FORMAT_NCNC) > 0) {
          if ( ((handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_F32) && (handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_F32) ) ) {
            layout->datatype = LIBXSMM_DNN_DATATYPE_F32;
            layout->dim_type = (libxsmm_dnn_tensor_dimtype*) malloc(4*sizeof(libxsmm_dnn_tensor_dimtype));
            layout->dim_size = (unsigned int*) malloc(4*sizeof(unsigned int));

            if (0 != layout->dim_type && 0 != layout->dim_size) { /* TODO: handle the error */
              layout->num_dims = 4;

              if ( (type == LIBXSMM_DNN_RNN_REGULAR_INPUT) || (type == LIBXSMM_DNN_RNN_GRADIENT_INPUT) ) {
                layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
                layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_N;
                layout->dim_type[2] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
                layout->dim_type[3] = LIBXSMM_DNN_TENSOR_DIMTYPE_N;
                layout->dim_size[0] = (unsigned int)handle->bc;
                layout->dim_size[1] = (unsigned int)handle->bn;
                layout->dim_size[2] = (unsigned int)(handle->desc.C / handle->bc);
                layout->dim_size[3] = (unsigned int)(handle->desc.N / handle->bn);
              } else if ( (type == LIBXSMM_DNN_RNN_REGULAR_HIDDEN_STATE) || (type == LIBXSMM_DNN_RNN_GRADIENT_HIDDEN_STATE) ) {
                layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
                layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_N;
                layout->dim_type[2] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
                layout->dim_type[3] = LIBXSMM_DNN_TENSOR_DIMTYPE_N;
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
        } else if ((handle->desc.buffer_format & LIBXSMM_DNN_TENSOR_FORMAT_NC) > 0) {
          if ( ((handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_F32) && (handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_F32) ) ) {
            layout->datatype = LIBXSMM_DNN_DATATYPE_F32;
            layout->dim_type = (libxsmm_dnn_tensor_dimtype*) malloc(4*sizeof(libxsmm_dnn_tensor_dimtype));
            layout->dim_size = (unsigned int*) malloc(4*sizeof(unsigned int));

            if (0 != layout->dim_type && 0 != layout->dim_size) { /* TODO: handle the error */
              layout->num_dims = 4;

              if ( (type == LIBXSMM_DNN_RNN_REGULAR_INPUT) || (type == LIBXSMM_DNN_RNN_GRADIENT_INPUT) ) {
                layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
                layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
                layout->dim_type[2] = LIBXSMM_DNN_TENSOR_DIMTYPE_N;
                layout->dim_type[3] = LIBXSMM_DNN_TENSOR_DIMTYPE_N;
                layout->dim_size[0] = (unsigned int)handle->bc;
                layout->dim_size[1] = (unsigned int)(handle->desc.C / handle->bc);
                layout->dim_size[2] = (unsigned int)handle->bn;
                layout->dim_size[3] = (unsigned int)(handle->desc.N / handle->bn);
              } else if ( (type == LIBXSMM_DNN_RNN_REGULAR_HIDDEN_STATE) || (type == LIBXSMM_DNN_RNN_GRADIENT_HIDDEN_STATE) ) {
                layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
                layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
                layout->dim_type[2] = LIBXSMM_DNN_TENSOR_DIMTYPE_N;
                layout->dim_type[3] = LIBXSMM_DNN_TENSOR_DIMTYPE_N;
                layout->dim_size[0] = (unsigned int)handle->bk;
                layout->dim_size[1] = (unsigned int)(handle->desc.K / handle->bk);
                layout->dim_size[2] = (unsigned int)handle->bn;
                layout->dim_size[3] = (unsigned int)(handle->desc.N / handle->bn);
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
      } else if ( (type == LIBXSMM_DNN_RNN_REGULAR_WEIGHT)       || (type == LIBXSMM_DNN_RNN_GRADIENT_WEIGHT) ||
                  (type == LIBXSMM_DNN_RNN_REGULAR_RECUR_WEIGHT) || (type == LIBXSMM_DNN_RNN_GRADIENT_RECUR_WEIGHT) ) {
        layout->format = handle->desc.filter_format;
        layout->tensor_type = LIBXSMM_DNN_FILTER;
        if ((handle->desc.filter_format & LIBXSMM_DNN_TENSOR_FORMAT_KCCK) > 0) {
          if ( ((handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_F32) && (handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_F32) ) ) {
            layout->datatype = LIBXSMM_DNN_DATATYPE_F32;
            layout->dim_type = (libxsmm_dnn_tensor_dimtype*) malloc(4*sizeof(libxsmm_dnn_tensor_dimtype));
            layout->dim_size = (unsigned int*) malloc(4*sizeof(unsigned int));

            if (0 != layout->dim_type && 0 != layout->dim_size) { /* TODO: handle the error */
              layout->num_dims = 4;

              if ( (type == LIBXSMM_DNN_RNN_REGULAR_WEIGHT) || (type == LIBXSMM_DNN_RNN_GRADIENT_WEIGHT) ) {
                layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
                layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
                layout->dim_type[2] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
                layout->dim_type[3] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
                layout->dim_size[0] = (unsigned int)handle->bk;
                layout->dim_size[1] = (unsigned int)handle->bc;
                layout->dim_size[2] = (unsigned int)(handle->desc.C / handle->bc);
                layout->dim_size[3] = (unsigned int)(handle->desc.K / handle->bk);
              } else if ( (type == LIBXSMM_DNN_RNN_REGULAR_RECUR_WEIGHT) || (type == LIBXSMM_DNN_RNN_GRADIENT_RECUR_WEIGHT) ) {
                layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
                layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
                layout->dim_type[2] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
                layout->dim_type[3] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
                layout->dim_size[0] = (unsigned int)handle->bk;
                layout->dim_size[1] = (unsigned int)handle->bk;
                layout->dim_size[2] = (unsigned int)(handle->desc.K / handle->bk);
                layout->dim_size[3] = (unsigned int)(handle->desc.K / handle->bk);
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
          } else {
            free(layout);
            layout = 0; /* make sure a NULL is returned */
            *status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
          }
        } else if ((handle->desc.filter_format & LIBXSMM_DNN_TENSOR_FORMAT_CK) > 0) {
          if ( ((handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_F32) && (handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_F32) ) ) {
            layout->datatype = LIBXSMM_DNN_DATATYPE_F32;
            layout->dim_type = (libxsmm_dnn_tensor_dimtype*) malloc(4*sizeof(libxsmm_dnn_tensor_dimtype));
            layout->dim_size = (unsigned int*) malloc(4*sizeof(unsigned int));

            if (0 != layout->dim_type && 0 != layout->dim_size) { /* TODO: handle the error */
              layout->num_dims = 4;

              if ( (type == LIBXSMM_DNN_RNN_REGULAR_WEIGHT) || (type == LIBXSMM_DNN_RNN_GRADIENT_WEIGHT) ) {
                layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
                layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
                layout->dim_type[2] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
                layout->dim_type[3] = LIBXSMM_DNN_TENSOR_DIMTYPE_C;
                layout->dim_size[0] = (unsigned int)handle->bk;
                layout->dim_size[1] = (unsigned int)(handle->desc.K / handle->bk);
                layout->dim_size[2] = (unsigned int)handle->bc;
                layout->dim_size[3] = (unsigned int)(handle->desc.C / handle->bc);
              } else if ( (type == LIBXSMM_DNN_RNN_REGULAR_RECUR_WEIGHT) || (type == LIBXSMM_DNN_RNN_GRADIENT_RECUR_WEIGHT) ) {
                layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
                layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
                layout->dim_type[2] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
                layout->dim_type[3] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
                layout->dim_size[0] = (unsigned int)handle->bk;
                layout->dim_size[1] = (unsigned int)(handle->desc.K / handle->bk);
                layout->dim_size[2] = (unsigned int)handle->bk;
                layout->dim_size[3] = (unsigned int)(handle->desc.K / handle->bk);
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
      } else if ( (type == LIBXSMM_DNN_RNN_REGULAR_BIAS) || (type == LIBXSMM_DNN_RNN_GRADIENT_BIAS) ) {
        layout->format = handle->desc.buffer_format;
        layout->tensor_type = LIBXSMM_DNN_CHANNEL_SCALAR;

        if ( ((handle->desc.buffer_format & LIBXSMM_DNN_TENSOR_FORMAT_NC) > 0) || ((handle->desc.buffer_format & LIBXSMM_DNN_TENSOR_FORMAT_NCNC) > 0) ) {
          if ( ((handle->desc.datatype_in == LIBXSMM_DNN_DATATYPE_F32) && (handle->desc.datatype_out == LIBXSMM_DNN_DATATYPE_F32) ) ) {
            layout->datatype = LIBXSMM_DNN_DATATYPE_F32;
            layout->dim_type = (libxsmm_dnn_tensor_dimtype*) malloc(2*sizeof(libxsmm_dnn_tensor_dimtype));
            layout->dim_size = (unsigned int*) malloc(2*sizeof(unsigned int));

            if (0 != layout->dim_type && 0 != layout->dim_size) { /* TODO: handle the error */
              layout->num_dims = 2;

              if ( (type == LIBXSMM_DNN_RNN_REGULAR_BIAS) || (type == LIBXSMM_DNN_RNN_GRADIENT_BIAS) ) {
                layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
                layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_K;
                layout->dim_size[0] = (unsigned int)handle->bk;
                layout->dim_size[1] = (unsigned int)(handle->desc.K / handle->bk);
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
  const size_t sizeof_datatype = sizeof(float);
  size_t size = 0;
  *status = LIBXSMM_DNN_SUCCESS;

  if (0 != handle) {
    switch (kind) {
      case LIBXSMM_DNN_COMPUTE_KIND_FWD: {
                                           size += (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype; /* z1 */
                                           size += 64;
                                           size += (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype; /* z2 */
                                           size += 64;
                                           size += (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype; /* bM */
                                           size += 64;
                                         } break;
      case LIBXSMM_DNN_COMPUTE_KIND_BWD:
      case LIBXSMM_DNN_COMPUTE_KIND_UPD:
      case LIBXSMM_DNN_COMPUTE_KIND_ALL: {
                                           size += (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype; /* z1 */
                                           size += 64;
                                           size += (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype; /* z2, zi */
                                           size += 64;
                                           size += (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype * (size_t)handle->desc.t; /* deltat */
                                           size += 64;
                                           size += (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype; /* di1 */
                                           size += 64;
                                           size += (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype; /* di2 */
                                           size += 64;
                                           size += (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype * (size_t)handle->desc.t; /* deltaMt */
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
  const size_t sizeof_datatype = sizeof(float);

  if (scratch == 0) {
    status = LIBXSMM_DNN_ERR_SCRATCH_NOT_ALLOCED;
    return status;
  }

  if (0 != handle) {
    switch (kind) {
      case LIBXSMM_DNN_COMPUTE_KIND_FWD: {
                                           if (address % 64 == 0) {
                                             handle->z1->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->z1->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->z2->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->z2->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->bM->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->bM->data = (void*)(address+offset);
                                           }
                                         } break;
      case LIBXSMM_DNN_COMPUTE_KIND_BWD:
      case LIBXSMM_DNN_COMPUTE_KIND_UPD:
      case LIBXSMM_DNN_COMPUTE_KIND_ALL: {
                                           if (address % 64 == 0) {
                                             handle->z1->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->z1->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->z2->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->z2->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype;
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
                                             handle->di1->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->di1->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->di2->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->di2->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype;
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
                                           handle->z1->data = 0;
                                           handle->z2->data = 0;
                                           handle->bM->data = 0;
                                           handle->z1 = 0;
                                           handle->z2 = 0;
                                           handle->bM = 0;
                                         } break;
      case LIBXSMM_DNN_COMPUTE_KIND_BWD:
      case LIBXSMM_DNN_COMPUTE_KIND_UPD:
      case LIBXSMM_DNN_COMPUTE_KIND_ALL: {
                                           handle->z1->data = 0;
                                           handle->z2->data = 0;
                                           handle->deltat->data = 0;
                                           handle->di1->data = 0;
                                           handle->di2->data = 0;
                                           handle->deltaMt->data = 0;
                                           handle->z1 = 0;
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
  const size_t sizeof_datatype = sizeof(float);
  size_t size = 0;
  *status = LIBXSMM_DNN_SUCCESS;

  if (0 != handle) {
    switch (kind) {
      case LIBXSMM_DNN_COMPUTE_KIND_FWD: {
                                           size += (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype * (size_t)handle->desc.t; /* zt */
                                           size += 64;
                                         } break;
      case LIBXSMM_DNN_COMPUTE_KIND_BWD:
      case LIBXSMM_DNN_COMPUTE_KIND_UPD:
      case LIBXSMM_DNN_COMPUTE_KIND_ALL: {
                                           size += (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype * (size_t)handle->desc.t; /* zt */
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
    const libxsmm_blasint K = handle->desc.K, N = handle->desc.N, t = handle->desc.t;
    LIBXSMM_VLA_DECL(2, const LIBXSMM_DNN_ELTWISE_FTYPE, zgold, (const LIBXSMM_DNN_ELTWISE_FTYPE*)zgoldtb, K * N);
    LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, z, (LIBXSMM_DNN_ELTWISE_FTYPE*)handle->z->data, K * N);
    libxsmm_blasint it;
    for (it = 0; it < t; ++it) {
      libxsmm_internal_matrix_copy(K*N*t, (LIBXSMM_DNN_ELTWISE_FTYPE*)zgoldtb, (LIBXSMM_DNN_ELTWISE_FTYPE*)handle->z->data, 0, 0, 1);
      /* libxsmm_bgemm_copyin_b(handle->handlewx, &LIBXSMM_VLA_ACCESS(2, zgold, it, 0, K * N), &K, &LIBXSMM_VLA_ACCESS(2, z, it, 0, K * N)); */
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
      (type != LIBXSMM_DNN_RNN_REGULAR_RECUR_WEIGHT) && (type != LIBXSMM_DNN_RNN_GRADIENT_RECUR_WEIGHT) &&
      (type != LIBXSMM_DNN_RNN_REGULAR_BIAS)         && (type != LIBXSMM_DNN_RNN_GRADIENT_BIAS) ) {
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
      } else if ( type == LIBXSMM_DNN_RNN_REGULAR_BIAS ) {
        handle->b = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_RNN_GRADIENT_BIAS ) {
        handle->djdb = (libxsmm_dnn_tensor*)tensor;
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


LIBXSMM_API libxsmm_dnn_tensor* libxsmm_dnn_rnncell_get_tensor(libxsmm_dnn_rnncell* handle, const libxsmm_dnn_tensor_type type, libxsmm_dnn_err_t* status)
{
  libxsmm_dnn_tensor* tensor = 0;
  LIBXSMM_UNUSED(status/*TODO*/);

  /* check for tensor type */
  if ( (type != LIBXSMM_DNN_RNN_REGULAR_INPUT)       && (type != LIBXSMM_DNN_RNN_GRADIENT_INPUT)  &&
      (type != LIBXSMM_DNN_RNN_REGULAR_HIDDEN_STATE) && (type != LIBXSMM_DNN_RNN_GRADIENT_HIDDEN_STATE) &&
      (type != LIBXSMM_DNN_RNN_REGULAR_WEIGHT)       && (type != LIBXSMM_DNN_RNN_GRADIENT_WEIGHT) &&
      (type != LIBXSMM_DNN_RNN_REGULAR_RECUR_WEIGHT) && (type != LIBXSMM_DNN_RNN_GRADIENT_RECUR_WEIGHT) &&
      (type != LIBXSMM_DNN_RNN_REGULAR_BIAS)         && (type != LIBXSMM_DNN_RNN_GRADIENT_BIAS) ) {
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
    } else if ( type == LIBXSMM_DNN_RNN_REGULAR_BIAS ) {
      tensor = handle->b;
    } else if ( type == LIBXSMM_DNN_RNN_GRADIENT_BIAS ) {
      tensor = handle->djdb;
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
      (type != LIBXSMM_DNN_RNN_REGULAR_RECUR_WEIGHT) && (type != LIBXSMM_DNN_RNN_GRADIENT_RECUR_WEIGHT) &&
      (type != LIBXSMM_DNN_RNN_REGULAR_BIAS)         && (type != LIBXSMM_DNN_RNN_GRADIENT_BIAS) ) {
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
    } else if ( type == LIBXSMM_DNN_RNN_REGULAR_BIAS ) {
      handle->b = 0;
    } else if ( type == LIBXSMM_DNN_RNN_GRADIENT_BIAS ) {
      handle->djdb = 0;
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
  libxsmm_blasint K = rnn->desc.K;
  libxsmm_blasint N = rnn->desc.N;
  libxsmm_blasint C = rnn->desc.C;
  libxsmm_blasint t = rnn->desc.t;
  libxsmm_blasint bk = rnn->bk;
  libxsmm_blasint bn = rnn->bn;
  libxsmm_blasint bc = rnn->bc;
  int nonlin = rnn->desc.nonlin;
  /* The following code should be in template */
  LIBXSMM_DNN_ELTWISE_FTYPE *wD = (LIBXSMM_DNN_ELTWISE_FTYPE*)rnn->w->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *xt = (LIBXSMM_DNN_ELTWISE_FTYPE*)rnn->xt->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *uD = (LIBXSMM_DNN_ELTWISE_FTYPE*)rnn->u->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *b = (LIBXSMM_DNN_ELTWISE_FTYPE*)rnn->b->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *ht = (LIBXSMM_DNN_ELTWISE_FTYPE*)rnn->h->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *z1D = (LIBXSMM_DNN_ELTWISE_FTYPE*)rnn->z1->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *z2D = (LIBXSMM_DNN_ELTWISE_FTYPE*)rnn->z2->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *zt = (LIBXSMM_DNN_ELTWISE_FTYPE*)rnn->z->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *bM = (LIBXSMM_DNN_ELTWISE_FTYPE*)rnn->bM->data;

  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, w, wD, K);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, u, uD, K);
  LIBXSMM_VLA_DECL(3, LIBXSMM_DNN_ELTWISE_FTYPE, x, xt, N, C);
  LIBXSMM_VLA_DECL(3, LIBXSMM_DNN_ELTWISE_FTYPE, h, ht, N, K);
  LIBXSMM_VLA_DECL(3, LIBXSMM_DNN_ELTWISE_FTYPE, z, zt, N, K);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, z1, z1D, K);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, z2, z2D, K);
  libxsmm_blasint i, ik, in, ic;
  libxsmm_smmfunction gemmkernela = libxsmm_smmdispatch( bk, bn, bc, &K, &C, &K, NULL, NULL, NULL, NULL );
  libxsmm_smmfunction gemmkernelb = libxsmm_smmdispatch( bk, bn, bk, &K, &K, &K, NULL, NULL, NULL, NULL );
  LIBXSMM_UNUSED(tid); LIBXSMM_UNUSED(start_thread); LIBXSMM_UNUSED(bM); /* TODO: remove */

  /* All data is in column-major format */
  for (i = 0; i < t; ++i) {
    /* let's run the cell in blocks for good locality */
    for (in = 0; in < N; in += bn) {
      for (ik = 0; ik < K; ik += bk) {
        /* we nee to set z1 to zero */
        libxsmm_internal_matrix_zero_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(2, z1, in, ik, K));


        /* z += W.x */
        for (ic = 0; ic < C; ic += bc) {
          /* this is a small matmul */
          gemmkernela( &LIBXSMM_VLA_ACCESS(2, w, ic, ik, K), &LIBXSMM_VLA_ACCESS(3, x, i, in, ic, N, C), &LIBXSMM_VLA_ACCESS(2, z1, in, ik, K) );
        }

        /* we nee to set z2 to zero */
        libxsmm_internal_matrix_zero_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(2, z2, in, ik, K));

        /* z2 = U.h */
        for (ic = 0; ic < K; ic += bk) {
          /* this is a small matmul */
          gemmkernelb( &LIBXSMM_VLA_ACCESS(2, u, ic, ik, K), &LIBXSMM_VLA_ACCESS(3, h, i, in, ic, N, K), &LIBXSMM_VLA_ACCESS(2, z2, in, ik, K) );
        }

        /* now let's run the elementwise kernels */
        libxsmm_internal_matrix_add_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(2, z1, in, ik, K),
                                                   &LIBXSMM_VLA_ACCESS(2, z2, in, ik, K),
                                                   &LIBXSMM_VLA_ACCESS(3, z, i, in, ik, N, K) );

        libxsmm_internal_matrix_add_colvector_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, z, i, in, ik, N, K), &b[ik] );

        if (1 == nonlin) {
          libxsmm_internal_matrix_relu_ld(    bk, bn, K, &LIBXSMM_VLA_ACCESS(3, z, i, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, h, i+1, in, ik, N, K) );
        } else if (2 == nonlin) {
          libxsmm_internal_matrix_sigmoid_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, z, i, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, h, i+1, in, ik, N, K) );
        } else {
          libxsmm_internal_matrix_tanh_ld(    bk, bn, K, &LIBXSMM_VLA_ACCESS(3, z, i, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, h, i+1, in, ik, N, K) );
        }
      }
    }
  }

  return status;
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_rnncell_bwd_upd_bu(libxsmm_dnn_rnncell* rnn, int start_thread, int tid, int pass)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
  libxsmm_blasint K = rnn->desc.K;
  libxsmm_blasint N = rnn->desc.N;
  libxsmm_blasint C = rnn->desc.C;
  libxsmm_blasint t = rnn->desc.t;
  libxsmm_blasint bk = rnn->bk;
  libxsmm_blasint bn = rnn->bn;
  libxsmm_blasint bc = rnn->bc;
  int nonlin = rnn->desc.nonlin;
  int nThreads = rnn->desc.nThreads;
  LIBXSMM_DNN_ELTWISE_FTYPE *djdht = (LIBXSMM_DNN_ELTWISE_FTYPE*)rnn->djdht->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *zt = (LIBXSMM_DNN_ELTWISE_FTYPE*)rnn->z->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *deltat = (LIBXSMM_DNN_ELTWISE_FTYPE*)rnn->deltat->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *uD = (LIBXSMM_DNN_ELTWISE_FTYPE*)rnn->u->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *xt = (LIBXSMM_DNN_ELTWISE_FTYPE*)rnn->xt->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *ht = (LIBXSMM_DNN_ELTWISE_FTYPE*)rnn->h->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *wD = (LIBXSMM_DNN_ELTWISE_FTYPE*)rnn->w->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *djduD = (LIBXSMM_DNN_ELTWISE_FTYPE*)rnn->djdu->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *djdwD = (LIBXSMM_DNN_ELTWISE_FTYPE*)rnn->djdw->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *djdb = (LIBXSMM_DNN_ELTWISE_FTYPE*)rnn->djdb->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *djdxt = (LIBXSMM_DNN_ELTWISE_FTYPE*)rnn->djdxt->data;
  LIBXSMM_DNN_ELTWISE_FTYPE* ziD = (LIBXSMM_DNN_ELTWISE_FTYPE*)rnn->z1->data;
  LIBXSMM_DNN_ELTWISE_FTYPE* di1D = (LIBXSMM_DNN_ELTWISE_FTYPE*)rnn->di1->data;
  LIBXSMM_DNN_ELTWISE_FTYPE* di2D = (LIBXSMM_DNN_ELTWISE_FTYPE*)rnn->di2->data;
  /* LIBXSMM_DNN_ELTWISE_FTYPE* deltaMt = (LIBXSMM_DNN_ELTWISE_FTYPE*)rnn->deltaMt->data; */
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, u, uD, K);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, w, wD, K);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, djdu, djduD, K);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, djdw, djdwD, K);
  LIBXSMM_VLA_DECL(3, LIBXSMM_DNN_ELTWISE_FTYPE, djdh, djdht, N, K);
  LIBXSMM_VLA_DECL(3, LIBXSMM_DNN_ELTWISE_FTYPE, z, zt, N, K);
  LIBXSMM_VLA_DECL(3, LIBXSMM_DNN_ELTWISE_FTYPE, delta, deltat, N, K);
  LIBXSMM_VLA_DECL(3, LIBXSMM_DNN_ELTWISE_FTYPE, x, xt, N, C);
  LIBXSMM_VLA_DECL(3, LIBXSMM_DNN_ELTWISE_FTYPE, h, ht, N, K);
  LIBXSMM_VLA_DECL(3, LIBXSMM_DNN_ELTWISE_FTYPE, djdx, djdxt, N, C);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, di1, di1D, K);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, di2, di2D, K);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, zi, ziD, K);

  libxsmm_blasint i, ik, in, ic, jk, jn, jc, ek, en, ec;
  const int ltid = tid - start_thread;

  /* initialization is done at the beginning */
  libxsmm_internal_matrix_zero(N*C*t, djdxt, start_thread, tid, nThreads);
  libxsmm_internal_matrix_zero(C*K*t, djdwD, start_thread, tid, nThreads);
  libxsmm_internal_matrix_zero(K*K*t, djduD, start_thread, tid, nThreads);
  /* The following code is for time step t-1 */
  for (in = 0; in < N; in += bn) {
    for (ik = 0; ik < K; ik += bk) {
      if (1 == nonlin) {
        libxsmm_internal_matrix_relu_inverse_ld(    bk, bn, K, &LIBXSMM_VLA_ACCESS(3, z, t-1, in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, zi, in, ik, K) );
      } else if (2 == nonlin) {
        libxsmm_internal_matrix_sigmoid_inverse_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, z, t-1, in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, zi, in, ik, K) );
      } else {
        libxsmm_internal_matrix_tanh_inverse_ld(    bk, bn, K, &LIBXSMM_VLA_ACCESS(3, z, t-1, in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, zi, in, ik, K) );
      }
      libxsmm_internal_matrix_eltwise_mult_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(2, zi, in, ik, K),
                                                          &LIBXSMM_VLA_ACCESS(3, djdh,  t-1, in, ik, N, K),
                                                          &LIBXSMM_VLA_ACCESS(3, delta, t-1, in, ik, N, K) );
    }
    for (jn = 0; jn < bn; jn++) {
      for (jk = 0; jk < bk; jk++) {
        en = in + jn;
        ek = ik + jk;
        if (1 == pass || 3 == pass) {
          /* djdx = W^T * delta */
          for (ic = 0; ic < C; ic += bc) {
            for (jc = 0; jc < bc; jc++) {
              ec = ic + jc;
              LIBXSMM_VLA_ACCESS(3, djdx, t-1, en, ec, N, C) += LIBXSMM_VLA_ACCESS(3, delta, t-1, en, ek, N, K) * LIBXSMM_VLA_ACCESS(2, w, ec, ek, K);
            }
          }
        }
        if (2 == pass || 3 == pass) {
          /* djdu = delta * h^T */
          for (ic = 0; ic < K; ic += bk) {
            for (jc = 0; jc < bk; jc++) {
              ec = ic + jc;
              LIBXSMM_VLA_ACCESS(2, djdu, ec, ek, K) += LIBXSMM_VLA_ACCESS(3, h, t-1, en, ec, N, K) * LIBXSMM_VLA_ACCESS(3, delta, t-1, en, ek, N, K);
            }
          }
          /* djdw = delta * x^T */
          for (ic = 0; ic < C; ic += bc) {
            for (jc = 0; jc < bc; jc++) {
              ec = ic + jc;
              LIBXSMM_VLA_ACCESS(2, djdw, ec, ek, K) += LIBXSMM_VLA_ACCESS(3, x, t-1, en, ec, N, C) * LIBXSMM_VLA_ACCESS(3, delta, t-1, en, ek, N, K);
            }
          }
          djdb[ek] += LIBXSMM_VLA_ACCESS(3, delta, t-1, en, ek, N, K);
        }
      }
    }
  }
  for (i = t-2; i >= 0; --i) {
    /* let's run the cell in blocks for good locality */
    for (in = 0; in < N; in += bn) {
      for (ik = 0; ik < K; ik += bk) {
        if (1 == nonlin) {
          libxsmm_internal_matrix_relu_inverse_ld(    bk, bn, K, &LIBXSMM_VLA_ACCESS(3, z, i, in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, zi, in, ik, K) );
        } else if (2 == nonlin) {
          libxsmm_internal_matrix_sigmoid_inverse_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, z, i, in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, zi, in, ik, K) );
        } else {
          libxsmm_internal_matrix_tanh_inverse_ld(    bk, bn, K, &LIBXSMM_VLA_ACCESS(3, z, i, in, ik, N, K), &LIBXSMM_VLA_ACCESS(2, zi, in, ik, K) );
        }
        /* di1 = U^T * delta */
        for (jn = 0; jn < bn; jn++) {
          for (jk = 0; jk < bk; jk++) {
            en = in + jn;
            ek = ik + jk;
            LIBXSMM_VLA_ACCESS(2, di1, en, ek, K) = (LIBXSMM_DNN_ELTWISE_FTYPE)0;
            for (ic = 0; ic < K; ic += bk) {
              for (jc = 0; jc < bk; jc++) {
                ec = ic + jc;
                LIBXSMM_VLA_ACCESS(2, di1, en, ek, K) += LIBXSMM_VLA_ACCESS(3, delta, i+1, en, ec, N, K) * LIBXSMM_VLA_ACCESS(2, u, ek, ec, K);
              }
            }
          }
        }
        libxsmm_internal_matrix_add_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(2, di1, in, ik, K),
                                                   &LIBXSMM_VLA_ACCESS(3, djdh, i, in, ik, N, K),
                                                   &LIBXSMM_VLA_ACCESS(2, di2, in, ik, K) );
        libxsmm_internal_matrix_eltwise_mult_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(2, zi,  in, ik, K),
                                                            &LIBXSMM_VLA_ACCESS(2, di2, in, ik, K),
                                                            &LIBXSMM_VLA_ACCESS(3, delta, i, in, ik, N, K) );
        for (jn = 0; jn < bn; jn++) {
          for (jk = 0; jk < bk; jk++) {
            en = in + jn;
            ek = ik + jk;
            if (1 == pass || 3 == pass) {
              /* djdx = W^T * delta */
              for (ic = 0; ic < C; ic += bc) {
                for (jc = 0; jc < bc; jc++) {
                  ec = ic + jc;
                  LIBXSMM_VLA_ACCESS(3, djdx, i, en, ec, N, C) += LIBXSMM_VLA_ACCESS(3, delta, i, en, ek, N, K) * LIBXSMM_VLA_ACCESS(2, w, ec, ek, K);
                }
              }
            }
            if (2 == pass || 3 == pass) {
              /* djdu = delta * h^T */
              for (ic = 0; ic < K; ic += bk) {
                for (jc = 0; jc < bk; jc++) {
                  ec = ic + jc;
                  LIBXSMM_VLA_ACCESS(2, djdu, ec, ek, K) += LIBXSMM_VLA_ACCESS(3, h, i, en, ec, N, K) * LIBXSMM_VLA_ACCESS(3, delta, i, en, ek, N, K);
                }
              }
              /* djdw = delta * x^T */
              for (ic = 0; ic < C; ic += bc) {
                for (jc = 0; jc < bc; jc++) {
                  ec = ic + jc;
                  LIBXSMM_VLA_ACCESS(2, djdw, ec, ek, K) += LIBXSMM_VLA_ACCESS(3, x, i, en, ec, N, C) * LIBXSMM_VLA_ACCESS(3, delta, i, en, ek, N, K);
                }
              }
              djdb[ek] += LIBXSMM_VLA_ACCESS(3, delta, i, en, ek, N, K);
            }
          }
        }
      }
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
