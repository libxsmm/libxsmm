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
    handle->i1t = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->i1b = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->i2  = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->f1t = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->f1b = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->f2  = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->o1t = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->o1b = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->o2  = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->c1t = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->c1b = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->c2  = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->i   = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->f   = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->o   = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->c   = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->dh  = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->d1  = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->d2  = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->d   = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->i3  = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->f3  = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->d4  = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->djdht  = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->deltat = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->djddt  = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->djdit  = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->djdft  = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->djdct  = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->djdot  = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->djdxt  = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->djdwi  = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->djdwf  = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->djdwo  = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->djdwc  = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->djdri  = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->djdrf  = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->djdro  = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->djdrc  = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->djdbi  = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->djdbf  = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->djdbo  = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->djdbc  = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->doutt  = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->barrier = libxsmm_barrier_create(handle->desc.nThreads, 1);
    if (NULL == handle->doutt || NULL == handle->i1t || NULL == handle->i1b || NULL == handle->i2 || NULL == handle->f1t || NULL == handle->f1b ||
        NULL == handle->f2 || NULL == handle->o1t || NULL == handle->o1b || NULL == handle->o2 || NULL == handle->c1t ||
        NULL == handle->c1b || NULL == handle->c2 || NULL == handle->i || NULL == handle->f || NULL == handle->o ||
        NULL == handle->c || NULL == handle->dh || NULL == handle->d1 || NULL == handle->d2 || NULL == handle->d ||
        NULL == handle->i3 || NULL == handle->f3 || NULL == handle->d4 || NULL == handle->djdht || NULL == handle->deltat ||
        NULL == handle->djddt || NULL == handle->djdit || NULL == handle->djdft || NULL == handle->djdct ||
        NULL == handle->djdot || NULL == handle->djdxt || NULL == handle->djdwi || NULL == handle->djdwf ||
        NULL == handle->djdwo || NULL == handle->djdwc || NULL == handle->djdri || NULL == handle->djdrf ||
        NULL == handle->djdro || NULL == handle->djdrc || NULL == handle->djdbi || NULL == handle->djdbf ||
        NULL == handle->djdbo || NULL == handle->djdbc || NULL == handle->barrier)
    {
      free(handle->doutt); free(handle->i1t); free(handle->i1b); free(handle->i2); free(handle->f1t); free(handle->f1b);
      free(handle->f2); free(handle->o1t); free(handle->o1b); free(handle->o2); free(handle->c1t);
      free(handle->c1b); free(handle->c2); free(handle->i); free(handle->f); free(handle->o);
      free(handle->c); free(handle->dh); free(handle->d1); free(handle->d2); free(handle->d);
      free(handle->i3); free(handle->f3); free(handle->d4); free(handle->djdht); free(handle->deltat);
      free(handle->djddt); free(handle->djdit); free(handle->djdft); free(handle->djdct);
      free(handle->djdot); free(handle->djdxt); free(handle->djdwi); free(handle->djdwf);
      free(handle->djdwo); free(handle->djdwc); free(handle->djdri); free(handle->djdrf);
      free(handle->djdro); free(handle->djdrc); free(handle->djdbi); free(handle->djdbf);
      free(handle->djdbo); free(handle->djdbc);
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
    free(handle->doutt); free(handle->i1t); free(handle->i1b); free(handle->i2); free(handle->f1t); free(handle->f1b);
    free(handle->f2); free(handle->o1t); free(handle->o1b); free(handle->o2); free(handle->c1t);
    free(handle->c1b); free(handle->c2); free(handle->i); free(handle->f); free(handle->o);
    free(handle->c); free(handle->dh); free(handle->d1); free(handle->d2); free(handle->d);
    free(handle->i3); free(handle->f3); free(handle->d4); free(handle->djdht); free(handle->deltat);
    free(handle->djddt); free(handle->djdit); free(handle->djdft); free(handle->djdct);
    free(handle->djdot); free(handle->djdxt); free(handle->djdwi); free(handle->djdwf);
    free(handle->djdwo); free(handle->djdwc); free(handle->djdri); free(handle->djdrf);
    free(handle->djdro); free(handle->djdrc); free(handle->djdbi); free(handle->djdbf);
    free(handle->djdbo); free(handle->djdbc);
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
                } else if ( (type == LIBXSMM_DNN_LSTM_REGULAR_HIDDEN_STATE) || (type == LIBXSMM_DNN_LSTM_GRADIENT_HIDDEN_STATE) ) {
                  layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_RLM;
                  layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_RLN;
                  layout->dim_type[2] = LIBXSMM_DNN_TENSOR_DIMTYPE_RLM;
                  layout->dim_type[3] = LIBXSMM_DNN_TENSOR_DIMTYPE_RLN;
                  layout->dim_size[0] = (unsigned int)handle->bk;
                  layout->dim_size[1] = (unsigned int)handle->bn;
                  layout->dim_size[2] = (unsigned int)(handle->desc.K / handle->bk);
                  layout->dim_size[3] = (unsigned int)(handle->desc.N / handle->bn);
                } else if ( (type == LIBXSMM_DNN_LSTM_REGULAR_WEIGHT_I) || (type == LIBXSMM_DNN_LSTM_GRADIENT_WEIGHT_I) ||
                            (type == LIBXSMM_DNN_LSTM_REGULAR_WEIGHT_F) || (type == LIBXSMM_DNN_LSTM_GRADIENT_WEIGHT_F) ||
                            (type == LIBXSMM_DNN_LSTM_REGULAR_WEIGHT_O) || (type == LIBXSMM_DNN_LSTM_GRADIENT_WEIGHT_O) ||
                            (type == LIBXSMM_DNN_LSTM_REGULAR_WEIGHT_C) || (type == LIBXSMM_DNN_LSTM_GRADIENT_WEIGHT_C) ) {
                  layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_RLM;
                  layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_RLK;
                  layout->dim_type[2] = LIBXSMM_DNN_TENSOR_DIMTYPE_RLK;
                  layout->dim_type[3] = LIBXSMM_DNN_TENSOR_DIMTYPE_RLM;
                  layout->dim_size[0] = (unsigned int)handle->bk;
                  layout->dim_size[1] = (unsigned int)handle->bc;
                  layout->dim_size[2] = (unsigned int)(handle->desc.C / handle->bc);
                  layout->dim_size[3] = (unsigned int)(handle->desc.K / handle->bk);
                } else if ( (type == LIBXSMM_DNN_LSTM_REGULAR_RECUR_WEIGHT_I) || (type == LIBXSMM_DNN_LSTM_GRADIENT_RECUR_WEIGHT_I) ||
                            (type == LIBXSMM_DNN_LSTM_REGULAR_RECUR_WEIGHT_F) || (type == LIBXSMM_DNN_LSTM_GRADIENT_RECUR_WEIGHT_F) ||
                            (type == LIBXSMM_DNN_LSTM_REGULAR_RECUR_WEIGHT_O) || (type == LIBXSMM_DNN_LSTM_GRADIENT_RECUR_WEIGHT_O) ||
                            (type == LIBXSMM_DNN_LSTM_REGULAR_RECUR_WEIGHT_C) || (type == LIBXSMM_DNN_LSTM_GRADIENT_RECUR_WEIGHT_C) ) {
                  layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_RLM;
                  layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_RLM;
                  layout->dim_type[2] = LIBXSMM_DNN_TENSOR_DIMTYPE_RLM;
                  layout->dim_type[3] = LIBXSMM_DNN_TENSOR_DIMTYPE_RLM;
                  layout->dim_size[0] = (unsigned int)handle->bk;
                  layout->dim_size[1] = (unsigned int)handle->bk;
                  layout->dim_size[2] = (unsigned int)(handle->desc.K / handle->bk);
                  layout->dim_size[3] = (unsigned int)(handle->desc.K / handle->bk);
                } else if ( (type == LIBXSMM_DNN_LSTM_REGULAR_BIAS_I) || (type == LIBXSMM_DNN_LSTM_GRADIENT_BIAS_I) ||
                            (type == LIBXSMM_DNN_LSTM_REGULAR_BIAS_F) || (type == LIBXSMM_DNN_LSTM_GRADIENT_BIAS_F) ||
                            (type == LIBXSMM_DNN_LSTM_REGULAR_BIAS_O) || (type == LIBXSMM_DNN_LSTM_GRADIENT_BIAS_O) ||
                            (type == LIBXSMM_DNN_LSTM_REGULAR_BIAS_C) || (type == LIBXSMM_DNN_LSTM_GRADIENT_BIAS_C) ) {
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
                                           size += (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype * (size_t)handle->desc.t; /* i1t */
                                           size += 64;
                                           size += (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype * (size_t)handle->desc.t; /* f1t */
                                           size += 64;
                                           size += (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype * (size_t)handle->desc.t; /* o1t */
                                           size += 64;
                                           size += (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype * (size_t)handle->desc.t; /* c1t */
                                           size += 64;
                                           size += (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype; /* i1b */
                                           size += 64;
                                           size += (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype; /* i2 */
                                           size += 64;
                                           size += (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype; /* i3 */
                                           size += 64;
                                           size += (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype; /* f1b */
                                           size += 64;
                                           size += (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype; /* f2 */
                                           size += 64;
                                           size += (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype; /* f3 */
                                           size += 64;
                                           size += (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype; /* o1b */
                                           size += 64;
                                           size += (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype; /* o2 */
                                           size += 64;
                                           size += (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype; /* c1b */
                                           size += 64;
                                           size += (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype; /* c2 */
                                           size += 64;
                                           size += (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype; /* d1 */
                                           size += 64;
                                           size += (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype; /* d2 */
                                           size += 64;
                                           size += (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype; /* d3 (dh) */
                                           size += 64;
                                           size += (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype; /* d4 */
                                           size += 64;
                                           size += (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype * (size_t)handle->desc.t; /* delta */
                                           size += 64;
                                           size += (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype * (size_t)handle->desc.t; /* doutt */
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

  if (scratch == 0) {
    status = LIBXSMM_DNN_ERR_SCRATCH_NOT_ALLOCED;
    return status;
  }

  if (0 != handle) {
    switch (kind) {
      case LIBXSMM_DNN_COMPUTE_KIND_FWD: {
                                         } break;
      case LIBXSMM_DNN_COMPUTE_KIND_BWD:
      case LIBXSMM_DNN_COMPUTE_KIND_UPD:
      case LIBXSMM_DNN_COMPUTE_KIND_ALL: {
                                           if (address % 64 == 0) {
                                             handle->i1t->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->i1t->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype * (size_t)handle->desc.t;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->f1t->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->f1t->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype * (size_t)handle->desc.t;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->o1t->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->o1t->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype * (size_t)handle->desc.t;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->c1t->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->c1t->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype * (size_t)handle->desc.t;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->i1b->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->i1b->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->i2->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->i2->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->i3->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->i3->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->f1b->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->f1b->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->f2->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->f2->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->f3->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->f3->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->o1b->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->o1b->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->o2->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->o2->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->c1b->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->c1b->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->c2->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->c2->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->d1->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->d1->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->d2->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->d2->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->dh->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->dh->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->d4->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->d4->data = (void*)(address+offset);
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
                                             handle->doutt->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->doutt->data = (void*)(address+offset);
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
                                           handle->deltat->data = 0;
                                           handle->doutt->data = 0;
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
                                           handle->deltat = 0;
                                           handle->doutt = 0;
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

  if (0 != handle) {
    switch (kind) {
      case LIBXSMM_DNN_COMPUTE_KIND_FWD: {
                                           size += (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype * (size_t)handle->desc.t; /* i */
                                           size += 64;
                                           size += (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype * (size_t)handle->desc.t; /* f */
                                           size += 64;
                                           size += (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype * (size_t)handle->desc.t; /* o */
                                           size += 64;
                                           size += (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype * (size_t)handle->desc.t; /* c */
                                           size += 64;
                                           size += (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype * ((size_t)handle->desc.t + 1); /* d */
                                           size += 64;
                                         } break;
      case LIBXSMM_DNN_COMPUTE_KIND_BWD:
      case LIBXSMM_DNN_COMPUTE_KIND_UPD:
      case LIBXSMM_DNN_COMPUTE_KIND_ALL: {
                                           size += (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype * (size_t)handle->desc.t; /* i */
                                           size += 64;
                                           size += (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype * (size_t)handle->desc.t; /* f */
                                           size += 64;
                                           size += (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype * (size_t)handle->desc.t; /* o */
                                           size += 64;
                                           size += (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype * (size_t)handle->desc.t; /* c */
                                           size += 64;
                                           size += (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype * (size_t)handle->desc.t; /* d */
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
  uintptr_t address = (uintptr_t)internalstate;
  size_t offset = 0;
  size_t scratch_size = 0;
  const size_t sizeof_datatype = sizeof(float);

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
                                           scratch_size = (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype * (size_t)handle->desc.t;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->f->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->f->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype * (size_t)handle->desc.t;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->o->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->o->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype * (size_t)handle->desc.t;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->c->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->c->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype * (size_t)handle->desc.t;
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
                                           scratch_size = (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype * (size_t)handle->desc.t;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->f->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->f->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype * (size_t)handle->desc.t;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->o->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->o->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype * (size_t)handle->desc.t;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->c->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->c->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->desc.K * (size_t)handle->desc.N * sizeof_datatype * (size_t)handle->desc.t;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->d->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->d->data = (void*)(address+offset);
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
                                           handle->i = 0;
                                           handle->f = 0;
                                           handle->o = 0;
                                           handle->c = 0;
                                           handle->d = 0;
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
    LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, i, (LIBXSMM_DNN_ELTWISE_FTYPE*)handle->i->data, K * N);
    LIBXSMM_VLA_DECL(2, /*const*/ LIBXSMM_DNN_ELTWISE_FTYPE, fgold, (/*const*/ LIBXSMM_DNN_ELTWISE_FTYPE*)fgoldtb, K * N);
    LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, f, (LIBXSMM_DNN_ELTWISE_FTYPE*)handle->f->data, K * N);
    LIBXSMM_VLA_DECL(2, /*const*/ LIBXSMM_DNN_ELTWISE_FTYPE, ogold, (/*const*/ LIBXSMM_DNN_ELTWISE_FTYPE*)ogoldtb, K * N);
    LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, o, (LIBXSMM_DNN_ELTWISE_FTYPE*)handle->o->data, K * N);
    LIBXSMM_VLA_DECL(2, /*const*/ LIBXSMM_DNN_ELTWISE_FTYPE, cgold, (/*const*/ LIBXSMM_DNN_ELTWISE_FTYPE*)cgoldtb, K * N);
    LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, c, (LIBXSMM_DNN_ELTWISE_FTYPE*)handle->c->data, K * N);
    LIBXSMM_VLA_DECL(2, /*const*/ LIBXSMM_DNN_ELTWISE_FTYPE, dgold, (/*const*/ LIBXSMM_DNN_ELTWISE_FTYPE*)dgoldtb, K * N);
    LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, d, (LIBXSMM_DNN_ELTWISE_FTYPE*)handle->d->data, K * N);
    libxsmm_blasint it;
    for (it = 0; it < t; ++it) {
      libxsmm_internal_matrix_copy(K*N, &LIBXSMM_VLA_ACCESS(2, igold, it, 0, K * N), &LIBXSMM_VLA_ACCESS(2, i, it, 0, K * N), 0, 0, 1);
      libxsmm_internal_matrix_copy(K*N, &LIBXSMM_VLA_ACCESS(2, fgold, it, 0, K * N), &LIBXSMM_VLA_ACCESS(2, f, it, 0, K * N), 0, 0, 1);
      libxsmm_internal_matrix_copy(K*N, &LIBXSMM_VLA_ACCESS(2, ogold, it, 0, K * N), &LIBXSMM_VLA_ACCESS(2, o, it, 0, K * N), 0, 0, 1);
      libxsmm_internal_matrix_copy(K*N, &LIBXSMM_VLA_ACCESS(2, cgold, it, 0, K * N), &LIBXSMM_VLA_ACCESS(2, c, it, 0, K * N), 0, 0, 1);
      libxsmm_internal_matrix_copy(K*N, &LIBXSMM_VLA_ACCESS(2, dgold, it, 0, K * N), &LIBXSMM_VLA_ACCESS(2, d, it, 0, K * N), 0, 0, 1);
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
        handle->bd = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_LSTM_GRADIENT_BIAS_C ) {
        handle->djdbc = (libxsmm_dnn_tensor*)tensor;
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
      tensor = handle->bd;
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
      handle->bd = 0;
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
  libxsmm_blasint K = lstm->desc.K;
  libxsmm_blasint N = lstm->desc.N;
  libxsmm_blasint C = lstm->desc.C;
  libxsmm_blasint t = lstm->desc.t;
  libxsmm_blasint bk = lstm->bk;
  libxsmm_blasint bn = lstm->bn;
  libxsmm_blasint bc = lstm->bc;
  LIBXSMM_DNN_ELTWISE_FTYPE *wiD = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->wi->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *wfD = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->wf->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *woD = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->wo->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *wcD = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->wc->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *xt  = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->xt->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *riD = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->ri->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *rfD = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->rf->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *roD = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->ro->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *rcD = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->rc->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *ht  = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->h->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *bi  = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->bi->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *bf  = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->bf->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *bo  = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->bo->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *bd  = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->bd->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *iD  = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->i->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *fD  = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->f->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *oD  = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->o->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *cD  = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->c->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *dD  = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->d->data;
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, wi, wiD, K);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, wf, wfD, K);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, wo, woD, K);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, wc, wcD, K);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, ri, riD, K);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, rf, rfD, K);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, ro, roD, K);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, rc, rcD, K);
  LIBXSMM_VLA_DECL(3, LIBXSMM_DNN_ELTWISE_FTYPE, x, xt, N, C);
  LIBXSMM_VLA_DECL(3, LIBXSMM_DNN_ELTWISE_FTYPE, h, ht, N, K);
  LIBXSMM_VLA_DECL(3, LIBXSMM_DNN_ELTWISE_FTYPE, i, iD, N, K);
  LIBXSMM_VLA_DECL(3, LIBXSMM_DNN_ELTWISE_FTYPE, f, fD, N, K);
  LIBXSMM_VLA_DECL(3, LIBXSMM_DNN_ELTWISE_FTYPE, o, oD, N, K);
  LIBXSMM_VLA_DECL(3, LIBXSMM_DNN_ELTWISE_FTYPE, c, cD, N, K);
  LIBXSMM_VLA_DECL(3, LIBXSMM_DNN_ELTWISE_FTYPE, d, dD, N, K);
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

        /* d = f.d */
        libxsmm_internal_matrix_eltwise_mult_ld( bk, bn, K, &LIBXSMM_VLA_ACCESS(3, f, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, d, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, d, j+1, in, ik, N, K) );
        /* d += i.c */
        libxsmm_internal_matrix_eltwise_fma_ld(  bk, bn, K, &LIBXSMM_VLA_ACCESS(3, i, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, c, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, d, j+1, in, ik, N, K) );
        /* h = o.tanh(d) */
        libxsmm_internal_matrix_elt_mult_tanh_ld(  bk, bn, K, &LIBXSMM_VLA_ACCESS(3, o, j, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, d, j+1, in, ik, N, K), &LIBXSMM_VLA_ACCESS(3, h, j+1, in, ik, N, K) );
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
  LIBXSMM_DNN_ELTWISE_FTYPE *wi = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->wi->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *wf = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->wf->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *wo = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->wo->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *wc = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->wc->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *xt = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->xt->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *ri = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->ri->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *rf = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->rf->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *ro = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->ro->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *rc = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->rc->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *ht = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->h->data;
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
  LIBXSMM_DNN_ELTWISE_FTYPE *doutt = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->doutt->data;
#if 0
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, x, xt, k * n);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, h, ht, m * n);
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
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, djdiM, djdiMt, m * n);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, djdfM, djdfMt, m * n);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, djdoM, djdoMt, m * n);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, djdcM, djdcMt, m * n);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, dout, doutt, m * n);
  libxsmm_bgemm_handle *handleud = lstm->handlewx;
  libxsmm_bgemm_handle *handledh = lstm->handleuh;
  libxsmm_bgemm_handle *handledx = lstm->handlett;
  libxsmm_bgemm_handle *handlewd = lstm->handlewd;
  libxsmm_blasint j, s, q, l, p;
  const int ltid = tid - start_thread;

  libxsmm_barrier_init(lstm->barrier, ltid);
  for (j = t-1; j >= 0; --j) {
    /* compute deltagold */
    if (j == t-1) {
      libxsmm_internal_matrix_copy(m * n, &LIBXSMM_VLA_ACCESS(2, djdh, t-1, 0, m * n), &LIBXSMM_VLA_ACCESS(2, delta, t-1, 0, m * n), start_thread, tid, lstm->nThreads);
      libxsmm_barrier_wait(lstm->barrier, ltid);
    } else {
      libxsmm_internal_matrix_add(m * n, &LIBXSMM_VLA_ACCESS(2, dout, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, djdh, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, delta, j, 0, m * n), start_thread, tid, lstm->nThreads);
      libxsmm_barrier_wait(lstm->barrier, ltid);
    }
    /* compute djdd */
    libxsmm_internal_matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, delta, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, o, j, 0, m * n), d1, start_thread, tid, lstm->nThreads);
    libxsmm_barrier_wait(lstm->barrier, ltid);
    libxsmm_internal_matrix_tanh_inverse(m * n, &LIBXSMM_VLA_ACCESS(2, d, j, 0, m * n), d2, start_thread, tid, lstm->nThreads);
    libxsmm_barrier_wait(lstm->barrier, ltid);
    if (j == t-1) {
      libxsmm_internal_matrix_eltwise_mult(m * n, d1, d2, &LIBXSMM_VLA_ACCESS(2, djdd, j, 0, m * n), start_thread, tid, lstm->nThreads);
      libxsmm_barrier_wait(lstm->barrier, ltid);
    } else {
      libxsmm_internal_matrix_eltwise_mult(m * n, d1, d2, d3, start_thread, tid, lstm->nThreads);
      libxsmm_barrier_wait(lstm->barrier, ltid);
      libxsmm_internal_matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, delta, j+1, 0, m * n), &LIBXSMM_VLA_ACCESS(2, f, j+1, 0, m * n), d4, start_thread, tid, lstm->nThreads);
      libxsmm_barrier_wait(lstm->barrier, ltid);
      libxsmm_internal_matrix_add(m * n, d3, d4, &LIBXSMM_VLA_ACCESS(2, djdd, j, 0, m * n), start_thread, tid, lstm->nThreads);
      libxsmm_barrier_wait(lstm->barrier, ltid);
    }
    /* compute djdc */
    libxsmm_internal_matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, djdd, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, i, j, 0, m * n), c1, start_thread, tid, lstm->nThreads);
    libxsmm_barrier_wait(lstm->barrier, ltid);
    libxsmm_internal_matrix_complement_square(m * n, &LIBXSMM_VLA_ACCESS(2, c, j, 0, m * n), c2, start_thread, tid, lstm->nThreads);
    libxsmm_barrier_wait(lstm->barrier, ltid);
    libxsmm_internal_matrix_eltwise_mult(m * n, c1, c2, &LIBXSMM_VLA_ACCESS(2, djdc, j, 0, m * n), start_thread, tid, lstm->nThreads);
    libxsmm_barrier_wait(lstm->barrier, ltid);
    /* compute djdi */
    libxsmm_internal_matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, djdd, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, c, j, 0, m * n), i1, start_thread, tid, lstm->nThreads);
    libxsmm_barrier_wait(lstm->barrier, ltid);
    libxsmm_internal_matrix_complement(m * n, &LIBXSMM_VLA_ACCESS(2, i, j, 0, m * n), i2, start_thread, tid, lstm->nThreads);
    libxsmm_barrier_wait(lstm->barrier, ltid);
    libxsmm_internal_matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, i, j, 0, m * n), i2, i3, start_thread, tid, lstm->nThreads);
    libxsmm_barrier_wait(lstm->barrier, ltid);
    libxsmm_internal_matrix_eltwise_mult(m * n, i1, i3, &LIBXSMM_VLA_ACCESS(2, djdi, j, 0, m * n), start_thread, tid, lstm->nThreads);
    libxsmm_barrier_wait(lstm->barrier, ltid);
    /* compute djdf */
    if (j >= 1) {
      libxsmm_internal_matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, djdd, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, d, j-1, 0, m * n), f1, start_thread, tid, lstm->nThreads);
      libxsmm_barrier_wait(lstm->barrier, ltid);
      libxsmm_internal_matrix_complement(m * n, &LIBXSMM_VLA_ACCESS(2, f, j, 0, m * n), f2, start_thread, tid, lstm->nThreads);
      libxsmm_barrier_wait(lstm->barrier, ltid);
      libxsmm_internal_matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, f, j, 0, m * n), f2, f3, start_thread, tid, lstm->nThreads);
      libxsmm_barrier_wait(lstm->barrier, ltid);
      libxsmm_internal_matrix_eltwise_mult(m * n, f1, f3, &LIBXSMM_VLA_ACCESS(2, djdf, j, 0, m * n), start_thread, tid, lstm->nThreads);
      libxsmm_barrier_wait(lstm->barrier, ltid);
    } else {
      /* djdf is zero for j == 0 */
      libxsmm_internal_matrix_zero(m * n, &LIBXSMM_VLA_ACCESS(2, djdf, j, 0, m * n), start_thread, tid, lstm->nThreads);
      libxsmm_barrier_wait(lstm->barrier, ltid);
    }
    /* compute djdo */
    libxsmm_internal_matrix_tanh(m * n, &LIBXSMM_VLA_ACCESS(2, d, j, 0, m * n), o1, start_thread, tid, lstm->nThreads);
    libxsmm_barrier_wait(lstm->barrier, ltid);
    libxsmm_internal_matrix_complement(m * n, &LIBXSMM_VLA_ACCESS(2, o, j, 0, m * n), o2, start_thread, tid, lstm->nThreads);
    libxsmm_barrier_wait(lstm->barrier, ltid);
    libxsmm_internal_matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, delta, j, 0, m * n), o1, o1, start_thread, tid, lstm->nThreads);
    libxsmm_barrier_wait(lstm->barrier, ltid);
    libxsmm_internal_matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, o, j, 0, m * n), o2, o2, start_thread, tid, lstm->nThreads);
    libxsmm_barrier_wait(lstm->barrier, ltid);
    libxsmm_internal_matrix_eltwise_mult(m * n, o1, o2, &LIBXSMM_VLA_ACCESS(2, djdo, j, 0, m * n), start_thread, tid, lstm->nThreads);
    libxsmm_barrier_wait(lstm->barrier, ltid);
    /* compute dout */
    if (j >= 1) {
      libxsmm_bgemm_st(handleud, ri, &LIBXSMM_VLA_ACCESS(2, djdi, j, 0, m * n),  &LIBXSMM_VLA_ACCESS(2, dout, j-1, 0, m * n), start_thread, tid);
      libxsmm_barrier_wait(lstm->barrier, ltid);
      libxsmm_bgemm_st(handleud, rf, &LIBXSMM_VLA_ACCESS(2, djdf, j, 0, m * n),  &LIBXSMM_VLA_ACCESS(2, dout, j-1, 0, m * n), start_thread, tid);
      libxsmm_barrier_wait(lstm->barrier, ltid);
      libxsmm_bgemm_st(handleud, ro, &LIBXSMM_VLA_ACCESS(2, djdo, j, 0, m * n),  &LIBXSMM_VLA_ACCESS(2, dout, j-1, 0, m * n), start_thread, tid);
      libxsmm_barrier_wait(lstm->barrier, ltid);
      libxsmm_bgemm_st(handleud, rc, &LIBXSMM_VLA_ACCESS(2, djdc, j, 0, m * n),  &LIBXSMM_VLA_ACCESS(2, dout, j-1, 0, m * n), start_thread, tid);
      libxsmm_barrier_wait(lstm->barrier, ltid);
    }
    if (pass == 1 || pass == 3) {
      /* compute djdx */
      libxsmm_bgemm_st(handlewd, wi, &LIBXSMM_VLA_ACCESS(2, djdi, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, djdx, j, 0, k * n), start_thread, tid);
      libxsmm_barrier_wait(lstm->barrier, ltid);
      libxsmm_bgemm_st(handlewd, wf, &LIBXSMM_VLA_ACCESS(2, djdf, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, djdx, j, 0, k * n), start_thread, tid);
      libxsmm_barrier_wait(lstm->barrier, ltid);
      libxsmm_bgemm_st(handlewd, wo, &LIBXSMM_VLA_ACCESS(2, djdo, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, djdx, j, 0, k * n), start_thread, tid);
      libxsmm_barrier_wait(lstm->barrier, ltid);
      libxsmm_bgemm_st(handlewd, wc, &LIBXSMM_VLA_ACCESS(2, djdc, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, djdx, j, 0, k * n), start_thread, tid);
      libxsmm_barrier_wait(lstm->barrier, ltid);
    }
  }
  if (pass == 2 || pass == 3) {
    /* Reorganizing djdi, djdf, dfdo, djdc */
    for (j = 0; j < t; ++j) {
      libxsmm_bgemm_convert_b_to_a(handleud, &LIBXSMM_VLA_ACCESS(2, djdi, j, 0, m * n), &m, &LIBXSMM_VLA_ACCESS(2, djdiM, j, 0, m * n));
      libxsmm_barrier_wait(lstm->barrier, ltid);
      libxsmm_bgemm_convert_b_to_a(handleud, &LIBXSMM_VLA_ACCESS(2, djdf, j, 0, m * n), &m, &LIBXSMM_VLA_ACCESS(2, djdfM, j, 0, m * n));
      libxsmm_barrier_wait(lstm->barrier, ltid);
      libxsmm_bgemm_convert_b_to_a(handleud, &LIBXSMM_VLA_ACCESS(2, djdo, j, 0, m * n), &m, &LIBXSMM_VLA_ACCESS(2, djdoM, j, 0, m * n));
      libxsmm_barrier_wait(lstm->barrier, ltid);
      libxsmm_bgemm_convert_b_to_a(handleud, &LIBXSMM_VLA_ACCESS(2, djdc, j, 0, m * n), &m, &LIBXSMM_VLA_ACCESS(2, djdcM, j, 0, m * n));
      libxsmm_barrier_wait(lstm->barrier, ltid);
    }
    /* compute djdw */
    for (j = 0; j < t; ++j) {
      libxsmm_bgemm_st(handledx, &LIBXSMM_VLA_ACCESS(2, djdiM, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, x, j, 0, k * n), djdwi, start_thread, tid);
      libxsmm_barrier_wait(lstm->barrier, ltid);
      libxsmm_bgemm_st(handledx, &LIBXSMM_VLA_ACCESS(2, djdfM, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, x, j, 0, k * n), djdwf, start_thread, tid);
      libxsmm_barrier_wait(lstm->barrier, ltid);
      libxsmm_bgemm_st(handledx, &LIBXSMM_VLA_ACCESS(2, djdoM, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, x, j, 0, k * n), djdwo, start_thread, tid);
      libxsmm_barrier_wait(lstm->barrier, ltid);
      libxsmm_bgemm_st(handledx, &LIBXSMM_VLA_ACCESS(2, djdcM, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, x, j, 0, k * n), djdwc, start_thread, tid);
      libxsmm_barrier_wait(lstm->barrier, ltid);
    }
    /* compute djdr */
    for (j = 0; j < t-1; ++j) {
      libxsmm_bgemm_st(handledh, &LIBXSMM_VLA_ACCESS(2, djdiM, j+1, 0, m * n), &LIBXSMM_VLA_ACCESS(2, h, j, 0, m * n), djdri, start_thread, tid);
      libxsmm_barrier_wait(lstm->barrier, ltid);
      libxsmm_bgemm_st(handledh, &LIBXSMM_VLA_ACCESS(2, djdfM, j+1, 0, m * n), &LIBXSMM_VLA_ACCESS(2, h, j, 0, m * n), djdrf, start_thread, tid);
      libxsmm_barrier_wait(lstm->barrier, ltid);
      libxsmm_bgemm_st(handledh, &LIBXSMM_VLA_ACCESS(2, djdoM, j+1, 0, m * n), &LIBXSMM_VLA_ACCESS(2, h, j, 0, m * n), djdro, start_thread, tid);
      libxsmm_barrier_wait(lstm->barrier, ltid);
      libxsmm_bgemm_st(handledh, &LIBXSMM_VLA_ACCESS(2, djdcM, j+1, 0, m * n), &LIBXSMM_VLA_ACCESS(2, h, j, 0, m * n), djdrc, start_thread, tid);
      libxsmm_barrier_wait(lstm->barrier, ltid);
    }
    /* compute djdb */
    if ((tid - start_thread) == 0) {
      for (j = 0; j < t-1; j++) {
        for (s = 0; s < n/bn; s++) {
          for (q = 0; q < m/bm; q++) {
            for (l = 0; l < bn; l++) {
              for (p = 0; p < bm; p++) {
                djdbi[q*bm+p] += LIBXSMM_VLA_ACCESS(2, djdi, j+1, (size_t)s*m*bn + (size_t)q*bm*bn + (size_t)l*bm+p, m * n);
                djdbf[q*bm+p] += LIBXSMM_VLA_ACCESS(2, djdf, j+1, (size_t)s*m*bn + (size_t)q*bm*bn + (size_t)l*bm+p, m * n);
                djdbo[q*bm+p] += LIBXSMM_VLA_ACCESS(2, djdo, j+1, (size_t)s*m*bn + (size_t)q*bm*bn + (size_t)l*bm+p, m * n);
                djdbc[q*bm+p] += LIBXSMM_VLA_ACCESS(2, djdc, j+1, (size_t)s*m*bn + (size_t)q*bm*bn + (size_t)l*bm+p, m * n);
              }
            }
          }
        }
      }
    }
    libxsmm_barrier_wait(lstm->barrier, ltid);
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

