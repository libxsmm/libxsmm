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

/* #define NON_FUSED_INPUT_GEMM */

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


LIBXSMM_API libxsmm_dnn_lstmcell* libxsmm_dnn_create_lstmcell(libxsmm_dnn_lstmcell_desc lstmcell_desc, libxsmm_dnn_err_t* status)
{
  libxsmm_dnn_lstmcell* handle = 0;
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

  handle = (libxsmm_dnn_lstmcell*)malloc(sizeof(libxsmm_dnn_lstmcell));
  if (0 != handle) {
    *status = LIBXSMM_DNN_SUCCESS;
    /* zero entire content; not only safer but also sets data and code pointers to NULL */
    memset(handle, 0, sizeof(*handle));
    /* initialize known handle components */
    handle->nThreads = lstmcell_desc.nThreads;
    handle->desc = lstmcell_desc;
    handle->datatype_in = lstmcell_desc.datatype_in;
    handle->datatype_out = lstmcell_desc.datatype_out;
    handle->reuse = lstmcell_desc.reuse;
    handle->pass = lstmcell_desc.pass;
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
#if defined(NON_FUSED_INPUT_GEMM)
      handle->handlett = libxsmm_bgemm_handle_create(handle->nThreads, LIBXSMM_GEMM_PRECISION(float), LIBXSMM_GEMM_PRECISION(float),
        handle->m, handle->n*handle->t, handle->k, &(handle->bm), &(handle->bn), &(handle->bk), &(handle->b_m1), &(handle->b_n1), &(handle->b_k1), &(handle->b_k2),
        &alpha, &beta, &gemm_flags, &strategy, &order);
#else
      handle->handlett = libxsmm_bgemm_handle_create(handle->nThreads, LIBXSMM_GEMM_PRECISION(float), LIBXSMM_GEMM_PRECISION(float),
        handle->m*4, handle->n*handle->t, handle->k, &(handle->bm), &(handle->bn), &(handle->bk), &(handle->b_m1), &(handle->b_n1), &(handle->b_k1), &(handle->b_k2),
        &alpha, &beta, &gemm_flags, &strategy, &order);
#endif
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
    handle->i4t    = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->djdiMt = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->djdfMt = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->djdcMt = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->djdoMt = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->barrier = libxsmm_barrier_create(handle->nThreads, 1);
    if (NULL == handle->i1t || NULL == handle->i1b || NULL == handle->i2 || NULL == handle->f1t || NULL == handle->f1b ||
        NULL == handle->f2 || NULL == handle->o1t || NULL == handle->o1b || NULL == handle->o2 || NULL == handle->c1t ||
        NULL == handle->c1b || NULL == handle->c2 || NULL == handle->i || NULL == handle->f || NULL == handle->o ||
        NULL == handle->c || NULL == handle->dh || NULL == handle->d1 || NULL == handle->d2 || NULL == handle->d ||
        NULL == handle->i3 || NULL == handle->f3 || NULL == handle->d4 || NULL == handle->djdht || NULL == handle->deltat ||
        NULL == handle->djddt || NULL == handle->djdit || NULL == handle->djdft || NULL == handle->djdct ||
        NULL == handle->djdot || NULL == handle->djdxt || NULL == handle->djdwi || NULL == handle->djdwf ||
        NULL == handle->djdwo || NULL == handle->djdwc || NULL == handle->djdri || NULL == handle->djdrf ||
        NULL == handle->djdro || NULL == handle->djdrc || NULL == handle->djdbi || NULL == handle->djdbf ||
        NULL == handle->djdbo || NULL == handle->djdbc || NULL == handle->i4t   || NULL == handle->djdiMt||
        NULL == handle->djdfMt|| NULL == handle->djdoMt|| NULL == handle->djdcMt|| NULL == handle->barrier)
    {
      free(handle->i1t); free(handle->i1b); free(handle->i2); free(handle->f1t); free(handle->f1b);
      free(handle->f2); free(handle->o1t); free(handle->o1b); free(handle->o2); free(handle->c1t);
      free(handle->c1b); free(handle->c2); free(handle->i); free(handle->f); free(handle->o);
      free(handle->c); free(handle->dh); free(handle->d1); free(handle->d2); free(handle->d);
      free(handle->i3); free(handle->f3); free(handle->d4); free(handle->djdht); free(handle->deltat);
      free(handle->djddt); free(handle->djdit); free(handle->djdft); free(handle->djdct);
      free(handle->djdot); free(handle->djdxt); free(handle->djdwi); free(handle->djdwf);
      free(handle->djdwo); free(handle->djdwc); free(handle->djdri); free(handle->djdrf);
      free(handle->djdro); free(handle->djdrc); free(handle->djdbi); free(handle->djdbf);
      free(handle->djdbo); free(handle->djdbc); free(handle->i4t);
      free(handle->djdiMt);free(handle->djdfMt);free(handle->djdoMt);free(handle->djdcMt);
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
    free(handle->i1t); free(handle->i1b); free(handle->i2); free(handle->f1t); free(handle->f1b);
    free(handle->f2); free(handle->o1t); free(handle->o1b); free(handle->o2); free(handle->c1t);
    free(handle->c1b); free(handle->c2); free(handle->i); free(handle->f); free(handle->o);
    free(handle->c); free(handle->dh); free(handle->d1); free(handle->d2); free(handle->d);
    free(handle->i3); free(handle->f3); free(handle->d4); free(handle->djdht); free(handle->deltat);
    free(handle->djddt); free(handle->djdit); free(handle->djdft); free(handle->djdct);
    free(handle->djdot); free(handle->djdxt); free(handle->djdwi); free(handle->djdwf);
    free(handle->djdwo); free(handle->djdwc); free(handle->djdri); free(handle->djdrf);
    free(handle->djdro); free(handle->djdrc); free(handle->djdbi); free(handle->djdbf);
    free(handle->djdbo); free(handle->djdbc); free(handle->i4t);
    free(handle->djdiMt);free(handle->djdfMt);free(handle->djdoMt);free(handle->djdcMt);
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
#if defined(NON_FUSED_INPUT_GEMM)
                  layout->dim_size[3] = handle->m / handle->bm;
#else
                  layout->dim_size[3] = (handle->m * 4) / handle->bm;
#endif
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
                                           size += handle->m * handle->n * sizeof_datatype * handle->t; /* i1t */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype * handle->t; /* f1t */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype * handle->t; /* o1t */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype * handle->t; /* c1t */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype * handle->t; /* i2 */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype * handle->t; /* f2 */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype * handle->t; /* o2 */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype * handle->t; /* c2 */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype; /* d1 */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype; /* d2 */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype; /* dh */
                                           size += 64;
#if !defined(NON_FUSED_INPUT_GEMM)
                                           size += handle->m * 4 * handle->n * sizeof_datatype * handle->t; /* i4t */
                                           size += 64;
#endif
                                         } break;
      case LIBXSMM_DNN_COMPUTE_KIND_BWD:
      case LIBXSMM_DNN_COMPUTE_KIND_UPD:
      case LIBXSMM_DNN_COMPUTE_KIND_ALL: {
                                           size += handle->m * handle->n * sizeof_datatype * handle->t; /* i1t */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype * handle->t; /* f1t */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype * handle->t; /* o1t */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype * handle->t; /* c1t */
                                           size += 64;
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
                                           size += handle->m * handle->n * sizeof_datatype * handle->t; /* delta */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype * handle->t; /* djdiMt */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype * handle->t; /* djdfMt */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype * handle->t; /* djdcMt */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype * handle->t; /* djdoMt */
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
  size_t sizeof_datatype = sizeof(float);

  if (scratch == 0) {
    status = LIBXSMM_DNN_ERR_SCRATCH_NOT_ALLOCED;
    return status;
  }

  if (0 != handle) {
    switch (kind) {
      case LIBXSMM_DNN_COMPUTE_KIND_FWD: {
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
                                           if (address % 64 == 0) {
                                             handle->i2->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->i2->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype * handle->t;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->f2->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->f2->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype * handle->t;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->o2->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->o2->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype * handle->t;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->c2->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->c2->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype * handle->t;
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
#if !defined(NON_FUSED_INPUT_GEMM)
                                           scratch_size = handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->i4t->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->i4t->data = (void*)(address+offset);
                                           }
#endif
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
                                             handle->deltat->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->deltat->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype * handle->t;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->djdiMt->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->djdiMt->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype * handle->t;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->djdfMt->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->djdfMt->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype * handle->t;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->djdoMt->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->djdoMt->data = (void*)(address+offset);
                                           }
                                           scratch_size = handle->m * handle->n * sizeof_datatype * handle->t;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->djdcMt->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->djdcMt->data = (void*)(address+offset);
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
                                           handle->i4t->data = 0;
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
                                           handle->i4t = 0;
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
                                           handle->djdiMt->data = 0;
                                           handle->djdfMt->data = 0;
                                           handle->djdoMt->data = 0;
                                           handle->djdcMt->data = 0;
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
                                           handle->djdiMt = 0;
                                           handle->djdfMt = 0;
                                           handle->djdoMt = 0;
                                           handle->djdcMt = 0;
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
                                           size += handle->m * handle->n * sizeof_datatype * handle->t; /* i */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype * handle->t; /* f */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype * handle->t; /* o */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype * handle->t; /* c */
                                           size += 64;
                                           size += handle->m * handle->n * sizeof_datatype * (handle->t + 1); /* d */
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
  uintptr_t address = (uintptr_t)internalstate;
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


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_lstmcell_assign_internalstate(libxsmm_dnn_lstmcell* handle, const void* igoldtb, const void* fgoldtb, const void* ogoldtb, const void* cgoldtb, const void* dgoldtb)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  if (handle != 0 && igoldtb != 0 && fgoldtb != 0 && ogoldtb != 0 && cgoldtb != 0 && dgoldtb != 0) {
    const libxsmm_blasint m = handle->m, n = handle->n, t = handle->t;
    LIBXSMM_VLA_DECL(2, const LIBXSMM_DNN_ELTWISE_FTYPE, igold, (const LIBXSMM_DNN_ELTWISE_FTYPE*)igoldtb, m * n);
    LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, i, (LIBXSMM_DNN_ELTWISE_FTYPE*)handle->i->data, m * n);
    LIBXSMM_VLA_DECL(2, const LIBXSMM_DNN_ELTWISE_FTYPE, fgold, (const LIBXSMM_DNN_ELTWISE_FTYPE*)fgoldtb, m * n);
    LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, f, (LIBXSMM_DNN_ELTWISE_FTYPE*)handle->f->data, m * n);
    LIBXSMM_VLA_DECL(2, const LIBXSMM_DNN_ELTWISE_FTYPE, ogold, (const LIBXSMM_DNN_ELTWISE_FTYPE*)ogoldtb, m * n);
    LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, o, (LIBXSMM_DNN_ELTWISE_FTYPE*)handle->o->data, m * n);
    LIBXSMM_VLA_DECL(2, const LIBXSMM_DNN_ELTWISE_FTYPE, cgold, (const LIBXSMM_DNN_ELTWISE_FTYPE*)cgoldtb, m * n);
    LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, c, (LIBXSMM_DNN_ELTWISE_FTYPE*)handle->c->data, m * n);
    LIBXSMM_VLA_DECL(2, const LIBXSMM_DNN_ELTWISE_FTYPE, dgold, (const LIBXSMM_DNN_ELTWISE_FTYPE*)dgoldtb, m * n);
    LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, d, (LIBXSMM_DNN_ELTWISE_FTYPE*)handle->d->data, m * n);
    libxsmm_blasint it;
    for (it = 0; it < t; ++it) {
      libxsmm_bgemm_copyin_b(handle->handlewd, &LIBXSMM_VLA_ACCESS(2, igold, it, 0, m * n), &m, &LIBXSMM_VLA_ACCESS(2, i, it, 0, m * n));
      libxsmm_bgemm_copyin_b(handle->handlewd, &LIBXSMM_VLA_ACCESS(2, fgold, it, 0, m * n), &m, &LIBXSMM_VLA_ACCESS(2, f, it, 0, m * n));
      libxsmm_bgemm_copyin_b(handle->handlewd, &LIBXSMM_VLA_ACCESS(2, ogold, it, 0, m * n), &m, &LIBXSMM_VLA_ACCESS(2, o, it, 0, m * n));
      libxsmm_bgemm_copyin_b(handle->handlewd, &LIBXSMM_VLA_ACCESS(2, cgold, it, 0, m * n), &m, &LIBXSMM_VLA_ACCESS(2, c, it, 0, m * n));
      libxsmm_bgemm_copyin_b(handle->handlewd, &LIBXSMM_VLA_ACCESS(2, dgold, it, 0, m * n), &m, &LIBXSMM_VLA_ACCESS(2, d, it, 0, m * n));
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


LIBXSMM_API void libxsmm_dnn_lstmcell_split_wx(libxsmm_dnn_lstmcell* lstm, libxsmm_blasint offset, void* src, void* dst, int start_thread, int tid, int nthreads)
{
  int ltid;
  int chunksize;
  int thr_begin;
  int thr_end;
  int size;
  libxsmm_blasint job, i, j, k, l, p;
  libxsmm_blasint m = lstm->m;
  libxsmm_blasint n = lstm->n;
  libxsmm_blasint t = lstm->t;
  libxsmm_blasint bm = lstm->bm;
  libxsmm_blasint bn = lstm->bn;
  libxsmm_blasint mb = m / bm;
  libxsmm_blasint nb = n / bn;
  LIBXSMM_VLA_DECL(5, LIBXSMM_DNN_ELTWISE_FTYPE, real_src, (LIBXSMM_DNN_ELTWISE_FTYPE*)src, nb, mb*4, bn, bm);
  LIBXSMM_VLA_DECL(5, LIBXSMM_DNN_ELTWISE_FTYPE, real_dst, (LIBXSMM_DNN_ELTWISE_FTYPE*)dst, nb, mb, bn, bm);
  ltid = tid - start_thread;
  /* number of tasks that could be run in parallel */
  size = t*n*m;
  /* compute chunk size */
  chunksize = (size % nthreads == 0) ? (size / nthreads) : (size / nthreads) + 1;
  /* compute thr_begin and thr_end */
  thr_begin = (ltid * chunksize < size) ? (ltid * chunksize) : size;
  thr_end = ((ltid + 1) * chunksize < size) ? ((ltid + 1) * chunksize) : size;
  for (job = thr_begin; job < thr_end; job++) {
    i = job / (n * m);
    j = (job % (n * m)) / (m * bn);
    k = ((job % (n * m)) % (m * bn)) / (bn * bm);
    l = (((job % (n * m)) % (m * bn)) % (bn * bm)) / bm;
    p = (((job % (n * m)) % (m * bn)) % (bn * bm)) % bm;
    LIBXSMM_VLA_ACCESS(5, real_dst, i, j, k, l, p, nb, mb, bn, bm) =
      LIBXSMM_VLA_ACCESS(5, real_src, i, j, mb*offset + k, l, p, nb, mb*4, bn, bm);
  }
}

LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_lstmcell_fwd(libxsmm_dnn_lstmcell* lstm, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
  libxsmm_blasint m = lstm->m;
  libxsmm_blasint n = lstm->n;
  /*libxsmm_blasint k = lstm->k;*/
  libxsmm_blasint t = lstm->t;
#if defined(LSTM_TIMING)
  libxsmm_blasint k = lstm->k;
  const double tflops = 12;
  const double gflops = (((2.0 * m * n * k) + (2.0 * m * n * m) + (2.0 * m * n) + (tflops * m * n)) * 4.0 + (4.0 * m * n) + (tflops * m * n)) * (double)t * 1E-9;
#endif
  int reuse = lstm->reuse;
  LIBXSMM_DNN_ELTWISE_FTYPE *wi  = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->wi->data;
#if defined(NON_FUSED_INPUT_GEMM)
  LIBXSMM_DNN_ELTWISE_FTYPE *wf  = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->wf->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *wo  = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->wo->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *wc  = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->wc->data;
#endif
  LIBXSMM_DNN_ELTWISE_FTYPE *xt  = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->xt->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *ri  = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->ri->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *rf  = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->rf->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *ro  = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->ro->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *rc  = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->rc->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *h   = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->h->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *bi  = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->bi->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *bf  = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->bf->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *bo  = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->bo->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *bc  = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->bc->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *i2t = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->i2->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *f2t = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->f2->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *o2t = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->o2->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *c2t = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->c2->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *i   = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->i->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *f   = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->f->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *o   = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->o->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *c   = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->c->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *dh  = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->dh->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *d1  = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->d1->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *d2  = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->d2->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *d   = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->d->data;
  /* LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, x, xt, k * n); */
  LIBXSMM_DNN_ELTWISE_FTYPE *i1t = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->i1t->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *f1t = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->f1t->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *o1t = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->o1t->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *c1t = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->c1t->data;
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, i1, i1t, m * n);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, f1, f1t, m * n);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, o1, o1t, m * n);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, c1, c1t, m * n);
  /* libxsmm_bgemm_handle *handlewx = lstm->handlewx; */
  libxsmm_bgemm_handle *handleuh = lstm->handleuh;
  libxsmm_bgemm_handle *handlett = lstm->handlett;
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, hnr, h, m * n);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, inr, i, m * n);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, fnr, f, m * n);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, onr, o, m * n);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, cnr, c, m * n);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, dnr, d, m * n);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, i2, i2t, m * n);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, f2, f2t, m * n);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, o2, o2t, m * n);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, c2, c2t, m * n);
#if defined(LSTM_TIMING)
  unsigned long long start;
  double duration;
  Gbl_t_input_total = 0.; Gbl_t_recur_total = 0.; Gbl_t_eltwise_total = 0.; Gbl_t_nonlin_total = 0.;
  Gbl_t_input = 0; Gbl_t_recur = 0; Gbl_t_eltwise = 0; Gbl_t_nonlin = 0;
  Gbl_duration_input = 0.; Gbl_duration_recur = 0.; Gbl_duration_eltwise = 0.; Gbl_duration_nonlin = 0.;
#endif
  int j;
  const int ltid = tid - start_thread;

  libxsmm_barrier_init(lstm->barrier, ltid);
#if defined(LSTM_TIMING)
  if (ltid == 0) { start = libxsmm_timer_tick(); }
#endif

  if (reuse) {
#if defined(LSTM_TIMING)
    if (ltid == 0) { Gbl_t_input = libxsmm_timer_tick(); }
#endif
#if defined(NON_FUSED_INPUT_GEMM)
    libxsmm_bgemm_st(handlett, wi, &LIBXSMM_VLA_ACCESS(2, x, 0, 0, k * n), &LIBXSMM_VLA_ACCESS(2, i1, 0, 0, m * n), start_thread, tid);
    libxsmm_bgemm_st(handlett, wf, &LIBXSMM_VLA_ACCESS(2, x, 0, 0, k * n), &LIBXSMM_VLA_ACCESS(2, f1, 0, 0, m * n), start_thread, tid);
    libxsmm_bgemm_st(handlett, wo, &LIBXSMM_VLA_ACCESS(2, x, 0, 0, k * n), &LIBXSMM_VLA_ACCESS(2, o1, 0, 0, m * n), start_thread, tid);
    libxsmm_bgemm_st(handlett, wc, &LIBXSMM_VLA_ACCESS(2, x, 0, 0, k * n), &LIBXSMM_VLA_ACCESS(2, c1, 0, 0, m * n), start_thread, tid);
#else
    libxsmm_bgemm_st(handlett, wi, xt, (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->i4t->data, start_thread, tid);
    libxsmm_dnn_lstmcell_split_wx(lstm, 0, lstm->i4t->data, lstm->i1t->data, start_thread, tid, lstm->nThreads);
    libxsmm_dnn_lstmcell_split_wx(lstm, 1, lstm->i4t->data, lstm->f1t->data, start_thread, tid, lstm->nThreads);
    libxsmm_dnn_lstmcell_split_wx(lstm, 2, lstm->i4t->data, lstm->o1t->data, start_thread, tid, lstm->nThreads);
    libxsmm_dnn_lstmcell_split_wx(lstm, 3, lstm->i4t->data, lstm->c1t->data, start_thread, tid, lstm->nThreads);
    libxsmm_barrier_wait(lstm->barrier, ltid);
#endif
#if defined(LSTM_TIMING)
    if (ltid == 0) {
      Gbl_duration_input = libxsmm_timer_duration(Gbl_t_input, libxsmm_timer_tick());
      Gbl_t_input_total += Gbl_duration_input;
    }
#endif
    for (j = 0; j < t; ++j) {
#if defined(LSTM_TIMING)
      if (ltid == 0) { Gbl_t_eltwise = libxsmm_timer_tick(); }
#endif
      libxsmm_internal_matrix_add(m * n, &LIBXSMM_VLA_ACCESS(2, i1, j, 0, m * n), bi, &LIBXSMM_VLA_ACCESS(2, i1, j, 0, m * n), start_thread, tid, lstm->nThreads);
      libxsmm_internal_matrix_add(m * n, &LIBXSMM_VLA_ACCESS(2, f1, j, 0, m * n), bf, &LIBXSMM_VLA_ACCESS(2, f1, j, 0, m * n), start_thread, tid, lstm->nThreads);
      libxsmm_internal_matrix_add(m * n, &LIBXSMM_VLA_ACCESS(2, o1, j, 0, m * n), bo, &LIBXSMM_VLA_ACCESS(2, o1, j, 0, m * n), start_thread, tid, lstm->nThreads);
      libxsmm_internal_matrix_add(m * n, &LIBXSMM_VLA_ACCESS(2, c1, j, 0, m * n), bc, &LIBXSMM_VLA_ACCESS(2, c1, j, 0, m * n), start_thread, tid, lstm->nThreads);
      libxsmm_barrier_wait(lstm->barrier, ltid);
#if defined(LSTM_TIMING)
      if (ltid == 0) {
        Gbl_duration_eltwise = libxsmm_timer_duration(Gbl_t_eltwise, libxsmm_timer_tick());
        Gbl_t_eltwise_total += Gbl_duration_eltwise;
      }
#endif
      libxsmm_internal_recursive_step(handleuh, ri, h, &LIBXSMM_VLA_ACCESS(2, i2, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, i1, j, 0, m * n), i, i, 2, m * n, start_thread, tid); /*sigmoid*/
      libxsmm_internal_recursive_step(handleuh, rf, h, &LIBXSMM_VLA_ACCESS(2, f2, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, f1, j, 0, m * n), f, f, 2, m * n, start_thread, tid); /*sigmoid*/
      libxsmm_internal_recursive_step(handleuh, ro, h, &LIBXSMM_VLA_ACCESS(2, o2, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, o1, j, 0, m * n), o, o, 2, m * n, start_thread, tid); /*sigmoid*/
      libxsmm_internal_recursive_step(handleuh, rc, h, &LIBXSMM_VLA_ACCESS(2, c2, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, c1, j, 0, m * n), c, c, 3, m * n, start_thread, tid); /*tanh*/
      libxsmm_barrier_wait(lstm->barrier, ltid);
#if defined(LSTM_TIMING)
      if (ltid == 0) { Gbl_t_eltwise = libxsmm_timer_tick(); }
#endif
      libxsmm_internal_matrix_eltwise_mult(m*n, f, d, d1, start_thread, tid, lstm->nThreads);
      libxsmm_internal_matrix_eltwise_mult(m*n, i, c, d2, start_thread, tid, lstm->nThreads);
      libxsmm_barrier_wait(lstm->barrier, ltid);
      libxsmm_internal_matrix_add(m*n, d1, d2, d, start_thread, tid, lstm->nThreads);
#if defined(LSTM_TIMING)
      libxsmm_barrier_wait(lstm->barrier, ltid); /* Additional barrier introduced to measure time */
      if (ltid == 0) {
        Gbl_duration_eltwise = libxsmm_timer_duration(Gbl_t_eltwise, libxsmm_timer_tick());
        Gbl_t_eltwise_total += Gbl_duration_eltwise;
        Gbl_t_nonlin = libxsmm_timer_tick();
      }
#endif
      libxsmm_internal_matrix_tanh(m*n, d, dh, start_thread, tid, lstm->nThreads); /*tanh*/
#if defined(LSTM_TIMING)
      libxsmm_barrier_wait(lstm->barrier, ltid); /* Additional barrier introduced to measure time */
      if (ltid == 0) {
        Gbl_duration_nonlin = libxsmm_timer_duration(Gbl_t_nonlin, libxsmm_timer_tick());
        Gbl_t_nonlin_total += Gbl_duration_nonlin;
        Gbl_t_eltwise = libxsmm_timer_tick();
      }
#endif
      libxsmm_internal_matrix_eltwise_mult(m*n, o, dh, h, start_thread, tid, lstm->nThreads);
      libxsmm_barrier_wait(lstm->barrier, ltid);
#if defined(LSTM_TIMING)
      if (ltid == 0) {
        Gbl_duration_eltwise = libxsmm_timer_duration(Gbl_t_eltwise, libxsmm_timer_tick());
        Gbl_t_eltwise_total += Gbl_duration_eltwise;
      }
#endif
    }
  } else {
#if defined(LSTM_TIMING)
    if (ltid == 0) { Gbl_t_input = libxsmm_timer_tick(); }
#endif
#if defined(NON_FUSED_INPUT_GEMM)
    libxsmm_bgemm_st(handlett, wi, &LIBXSMM_VLA_ACCESS(2, x, 0, 0, k * n), &LIBXSMM_VLA_ACCESS(2, i1, 0, 0, m * n), start_thread, tid);
    libxsmm_bgemm_st(handlett, wf, &LIBXSMM_VLA_ACCESS(2, x, 0, 0, k * n), &LIBXSMM_VLA_ACCESS(2, f1, 0, 0, m * n), start_thread, tid);
    libxsmm_bgemm_st(handlett, wo, &LIBXSMM_VLA_ACCESS(2, x, 0, 0, k * n), &LIBXSMM_VLA_ACCESS(2, o1, 0, 0, m * n), start_thread, tid);
    libxsmm_bgemm_st(handlett, wc, &LIBXSMM_VLA_ACCESS(2, x, 0, 0, k * n), &LIBXSMM_VLA_ACCESS(2, c1, 0, 0, m * n), start_thread, tid);
#else
    libxsmm_bgemm_st(handlett, wi, xt, (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->i4t->data, start_thread, tid);
    libxsmm_dnn_lstmcell_split_wx(lstm, 0, lstm->i4t->data, lstm->i1t->data, start_thread, tid, lstm->nThreads);
    libxsmm_dnn_lstmcell_split_wx(lstm, 1, lstm->i4t->data, lstm->f1t->data, start_thread, tid, lstm->nThreads);
    libxsmm_dnn_lstmcell_split_wx(lstm, 2, lstm->i4t->data, lstm->o1t->data, start_thread, tid, lstm->nThreads);
    libxsmm_dnn_lstmcell_split_wx(lstm, 3, lstm->i4t->data, lstm->c1t->data, start_thread, tid, lstm->nThreads);
    libxsmm_barrier_wait(lstm->barrier, ltid);
#endif
#if defined(LSTM_TIMING)
    if (ltid == 0) {
      Gbl_duration_input = libxsmm_timer_duration(Gbl_t_input, libxsmm_timer_tick());
      Gbl_t_input_total += Gbl_duration_input;
    }
#endif
    for (j = 0; j < t; ++j) {
#if defined(LSTM_TIMING)
      if (ltid == 0) { Gbl_t_eltwise = libxsmm_timer_tick(); }
#endif
      libxsmm_internal_matrix_add(m * n, &LIBXSMM_VLA_ACCESS(2, i1, j, 0, m * n), bi, &LIBXSMM_VLA_ACCESS(2, i1, j, 0, m * n), start_thread, tid, lstm->nThreads);
      libxsmm_internal_matrix_add(m * n, &LIBXSMM_VLA_ACCESS(2, f1, j, 0, m * n), bf, &LIBXSMM_VLA_ACCESS(2, f1, j, 0, m * n), start_thread, tid, lstm->nThreads);
      libxsmm_internal_matrix_add(m * n, &LIBXSMM_VLA_ACCESS(2, o1, j, 0, m * n), bo, &LIBXSMM_VLA_ACCESS(2, o1, j, 0, m * n), start_thread, tid, lstm->nThreads);
      libxsmm_internal_matrix_add(m * n, &LIBXSMM_VLA_ACCESS(2, c1, j, 0, m * n), bc, &LIBXSMM_VLA_ACCESS(2, c1, j, 0, m * n), start_thread, tid, lstm->nThreads);
      libxsmm_barrier_wait(lstm->barrier, ltid);
#if defined(LSTM_TIMING)
      if (ltid == 0) {
        Gbl_duration_eltwise = libxsmm_timer_duration(Gbl_t_eltwise, libxsmm_timer_tick());
        Gbl_t_eltwise_total += Gbl_duration_eltwise;
      }
#endif
      libxsmm_internal_recursive_step(handleuh, ri, &LIBXSMM_VLA_ACCESS(2, hnr, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, i2, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, i1, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, inr, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, inr, j, 0, m * n), 2, m * n, start_thread, tid); /*sigmoid*/
      libxsmm_internal_recursive_step(handleuh, rf, &LIBXSMM_VLA_ACCESS(2, hnr, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, f2, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, f1, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, fnr, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, fnr, j, 0, m * n), 2, m * n, start_thread, tid); /*sigmoid*/
      libxsmm_internal_recursive_step(handleuh, ro, &LIBXSMM_VLA_ACCESS(2, hnr, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, o2, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, o1, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, onr, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, onr, j, 0, m * n), 2, m * n, start_thread, tid); /*sigmoid*/
      libxsmm_internal_recursive_step(handleuh, rc, &LIBXSMM_VLA_ACCESS(2, hnr, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, c2, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, c1, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, cnr, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, cnr, j, 0, m * n), 3, m * n, start_thread, tid); /*tanh*/
      libxsmm_barrier_wait(lstm->barrier, ltid);
#if defined(LSTM_TIMING)
      if (ltid == 0) { Gbl_t_eltwise = libxsmm_timer_tick(); }
#endif
      libxsmm_internal_matrix_eltwise_mult(m*n, &LIBXSMM_VLA_ACCESS(2, fnr, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, dnr, j, 0, m * n), d1, start_thread, tid, lstm->nThreads);
      libxsmm_internal_matrix_eltwise_mult(m*n, &LIBXSMM_VLA_ACCESS(2, inr, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, cnr, j, 0, m * n), d2, start_thread, tid, lstm->nThreads);
      libxsmm_barrier_wait(lstm->barrier, ltid);
      libxsmm_internal_matrix_add(m*n, d1, d2, &LIBXSMM_VLA_ACCESS(2, dnr, j+1, 0, m * n), start_thread, tid, lstm->nThreads);
#if defined(LSTM_TIMING)
      libxsmm_barrier_wait(lstm->barrier, ltid); /* Additional barrier introduced to measure time */
      if (ltid == 0) {
        Gbl_duration_eltwise = libxsmm_timer_duration(Gbl_t_eltwise, libxsmm_timer_tick());
        Gbl_t_eltwise_total += Gbl_duration_eltwise;
        Gbl_t_nonlin = libxsmm_timer_tick();
      }
#endif
      libxsmm_internal_matrix_tanh(m*n, &LIBXSMM_VLA_ACCESS(2, dnr, j+1, 0, m * n), dh, start_thread, tid, lstm->nThreads); /*tanh*/
#if defined(LSTM_TIMING)
      libxsmm_barrier_wait(lstm->barrier, ltid); /* Additional barrier introduced to measure time */
      if (ltid == 0) {
        Gbl_duration_nonlin = libxsmm_timer_duration(Gbl_t_nonlin, libxsmm_timer_tick());
        Gbl_t_nonlin_total += Gbl_duration_nonlin;
        Gbl_t_eltwise = libxsmm_timer_tick();
      }
#endif
      libxsmm_internal_matrix_eltwise_mult(m*n, &LIBXSMM_VLA_ACCESS(2, onr, j, 0, m * n), dh, &LIBXSMM_VLA_ACCESS(2, hnr, j+1, 0, m * n), start_thread, tid, lstm->nThreads);
      libxsmm_barrier_wait(lstm->barrier, ltid);
#if defined(LSTM_TIMING)
      if (ltid == 0) {
        Gbl_duration_eltwise = libxsmm_timer_duration(Gbl_t_eltwise, libxsmm_timer_tick());
        Gbl_t_eltwise_total += Gbl_duration_eltwise;
      }
#endif
    }
  }
#if defined(LSTM_TIMING)
  if (ltid == 0) {
    duration = libxsmm_timer_duration(start, libxsmm_timer_tick());
    if (0 < duration) {
      fprintf(stdout, "\tLIBXSMM: %.1f GFLOPS/s\n", gflops / duration);
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


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_lstmcell_bwd_upd_bu(libxsmm_dnn_lstmcell* lstm, int start_thread, int tid, int pass)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
  libxsmm_blasint m = lstm->m;
  libxsmm_blasint n = lstm->n;
  libxsmm_blasint k = lstm->k;
  libxsmm_blasint t = lstm->t;
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
  LIBXSMM_DNN_ELTWISE_FTYPE *djdiMt = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->djdiMt->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *djdfMt = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->djdfMt->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *djdcMt = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->djdcMt->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *djdoMt = (LIBXSMM_DNN_ELTWISE_FTYPE*)lstm->djdoMt->data;
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
  libxsmm_bgemm_handle *handleud = lstm->handlewx;
  libxsmm_bgemm_handle *handledh = lstm->handleuh;
  libxsmm_bgemm_handle *handledx = lstm->handlett;
  libxsmm_bgemm_handle *handlewd = lstm->handlewd;
  int j;
  const int ltid = tid - start_thread;

  libxsmm_barrier_init(lstm->barrier, ltid);
  /* compute delta */
  libxsmm_internal_matrix_copy(m * n, &LIBXSMM_VLA_ACCESS(2, djdh, t-1, 0, m * n), &LIBXSMM_VLA_ACCESS(2, delta, t-1, 0, m * n), start_thread, tid, lstm->nThreads);
  /* compute djdd */
  libxsmm_internal_matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, djdh, t-1, 0, m * n), &LIBXSMM_VLA_ACCESS(2, o, t-1, 0, m * n), d1, start_thread, tid, lstm->nThreads);
  libxsmm_internal_matrix_tanh_inverse(m * n, &LIBXSMM_VLA_ACCESS(2, d, t-1, 0, m * n), d2, start_thread, tid, lstm->nThreads);
  libxsmm_barrier_wait(lstm->barrier, ltid);
  libxsmm_internal_matrix_eltwise_mult(m * n, d1, d2, &LIBXSMM_VLA_ACCESS(2, djdd, t-1, 0, m * n), start_thread, tid, lstm->nThreads);
  /* compute djdc */
  libxsmm_internal_matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, djdd, t-1, 0, m * n), &LIBXSMM_VLA_ACCESS(2, i, t-1, 0, m * n), c1, start_thread, tid, lstm->nThreads);
  libxsmm_internal_matrix_complement_square(m * n, &LIBXSMM_VLA_ACCESS(2, c, t-1, 0, m * n), c2, start_thread, tid, lstm->nThreads);
  libxsmm_barrier_wait(lstm->barrier, ltid);
  libxsmm_internal_matrix_eltwise_mult(m * n, c1, c2, &LIBXSMM_VLA_ACCESS(2, djdc, t-1, 0, m * n), start_thread, tid, lstm->nThreads);
  /* compute djdi */
  libxsmm_internal_matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, djdd, t-1, 0, m * n), &LIBXSMM_VLA_ACCESS(2, c, t-1, 0, m * n), i1, start_thread, tid, lstm->nThreads);
  libxsmm_internal_matrix_complement(m * n, &LIBXSMM_VLA_ACCESS(2, i, t-1, 0, m * n), i2, start_thread, tid, lstm->nThreads);
  libxsmm_barrier_wait(lstm->barrier, ltid);
  libxsmm_internal_matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, i, t-1, 0, m * n), i2, i3, start_thread, tid, lstm->nThreads);
  libxsmm_barrier_wait(lstm->barrier, ltid);
  libxsmm_internal_matrix_eltwise_mult(m * n, i1, i3, &LIBXSMM_VLA_ACCESS(2, djdi, t-1, 0, m * n), start_thread, tid, lstm->nThreads);
  /* compute djdf */
  libxsmm_internal_matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, djdd, t-1, 0, m * n), &LIBXSMM_VLA_ACCESS(2, d, t-2, 0, m * n), f1, start_thread, tid, lstm->nThreads);
  libxsmm_internal_matrix_complement(m * n, &LIBXSMM_VLA_ACCESS(2, f, t-1, 0, m * n), f2, start_thread, tid, lstm->nThreads);
  libxsmm_barrier_wait(lstm->barrier, ltid);
  libxsmm_internal_matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, f, t-1, 0, m * n), f2, f3, start_thread, tid, lstm->nThreads);
  libxsmm_barrier_wait(lstm->barrier, ltid);
  libxsmm_internal_matrix_eltwise_mult(m * n, f1, f3, &LIBXSMM_VLA_ACCESS(2, djdf, t-1, 0, m * n), start_thread, tid, lstm->nThreads);
  /* compute djdo */
  libxsmm_internal_matrix_tanh(m * n, &LIBXSMM_VLA_ACCESS(2, d, t-1, 0, m * n), o1, start_thread, tid, lstm->nThreads);
  libxsmm_internal_matrix_complement(m * n, &LIBXSMM_VLA_ACCESS(2, o, t-1, 0, m * n), o2, start_thread, tid, lstm->nThreads);
  libxsmm_barrier_wait(lstm->barrier, ltid);
  libxsmm_internal_matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, delta, t-1, 0, m * n), o1, o1, start_thread, tid, lstm->nThreads);
  libxsmm_internal_matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, o, t-1, 0, m * n), o2, o2, start_thread, tid, lstm->nThreads);
  libxsmm_barrier_wait(lstm->barrier, ltid);
  libxsmm_internal_matrix_eltwise_mult(m * n, o1, o2, &LIBXSMM_VLA_ACCESS(2, djdo, t-1, 0, m * n), start_thread, tid, lstm->nThreads);
  libxsmm_barrier_wait(lstm->barrier, ltid);
  if (pass == 1 || pass == 3) {
    /* compute djdx */
    libxsmm_bgemm_st(handlewd, wi, &LIBXSMM_VLA_ACCESS(2, djdi, t-1, 0, m * n), &LIBXSMM_VLA_ACCESS(2, djdx, t-1, 0, k * n), start_thread, tid);
    libxsmm_bgemm_st(handlewd, wf, &LIBXSMM_VLA_ACCESS(2, djdf, t-1, 0, m * n), &LIBXSMM_VLA_ACCESS(2, djdx, t-1, 0, k * n), start_thread, tid);
    libxsmm_bgemm_st(handlewd, wo, &LIBXSMM_VLA_ACCESS(2, djdo, t-1, 0, m * n), &LIBXSMM_VLA_ACCESS(2, djdx, t-1, 0, k * n), start_thread, tid);
    libxsmm_bgemm_st(handlewd, wc, &LIBXSMM_VLA_ACCESS(2, djdc, t-1, 0, m * n), &LIBXSMM_VLA_ACCESS(2, djdx, t-1, 0, k * n), start_thread, tid);
  }
  for (j = t-2; j >= 0; --j) {
    /* compute delta */
    libxsmm_bgemm_st(handleud, ri, &LIBXSMM_VLA_ACCESS(2, djdi, j, 0, m * n),  &LIBXSMM_VLA_ACCESS(2, delta, j+1, 0, m * n), start_thread, tid);
    libxsmm_bgemm_st(handleud, rf, &LIBXSMM_VLA_ACCESS(2, djdf, j, 0, m * n),  &LIBXSMM_VLA_ACCESS(2, delta, j+1, 0, m * n), start_thread, tid);
    libxsmm_bgemm_st(handleud, ro, &LIBXSMM_VLA_ACCESS(2, djdo, j, 0, m * n),  &LIBXSMM_VLA_ACCESS(2, delta, j+1, 0, m * n), start_thread, tid);
    libxsmm_bgemm_st(handleud, rc, &LIBXSMM_VLA_ACCESS(2, djdc, j, 0, m * n),  &LIBXSMM_VLA_ACCESS(2, delta, j+1, 0, m * n), start_thread, tid);
    libxsmm_internal_matrix_add(m * n, &LIBXSMM_VLA_ACCESS(2, djdh, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, delta, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, delta, j, 0, m * n), start_thread, tid, lstm->nThreads);
    /* compute djdd */
    libxsmm_internal_matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, djdh, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, o, j, 0, m * n), d1, start_thread, tid, lstm->nThreads);
    libxsmm_internal_matrix_tanh_inverse(m * n, &LIBXSMM_VLA_ACCESS(2, d, j, 0, m * n), d2, start_thread, tid, lstm->nThreads);
    libxsmm_barrier_wait(lstm->barrier, ltid);
    libxsmm_internal_matrix_eltwise_mult(m * n, d1, d2, d3, start_thread, tid, lstm->nThreads);
    libxsmm_internal_matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, delta, j+1, 0, m * n), &LIBXSMM_VLA_ACCESS(2, f, j+1, 0, m * n), d4, start_thread, tid, lstm->nThreads);
    libxsmm_barrier_wait(lstm->barrier, ltid);
    libxsmm_internal_matrix_add(m * n, d3, d4, &LIBXSMM_VLA_ACCESS(2, djdd, j, 0, m * n), start_thread, tid, lstm->nThreads);
    /* compute djdc */
    libxsmm_internal_matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, djdd, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, i, j, 0, m * n), c1, start_thread, tid, lstm->nThreads);
    libxsmm_internal_matrix_complement_square(m * n, &LIBXSMM_VLA_ACCESS(2, c, j, 0, m * n), c2, start_thread, tid, lstm->nThreads);
    libxsmm_barrier_wait(lstm->barrier, ltid);
    libxsmm_internal_matrix_eltwise_mult(m * n, c1, c2, &LIBXSMM_VLA_ACCESS(2, djdc, j, 0, m * n), start_thread, tid, lstm->nThreads);
    /* compute djdi */
    libxsmm_internal_matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, djdd, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, c, j, 0, m * n), i1, start_thread, tid, lstm->nThreads);
    libxsmm_internal_matrix_complement(m * n, &LIBXSMM_VLA_ACCESS(2, i, j, 0, m * n), i2, start_thread, tid, lstm->nThreads);
    libxsmm_barrier_wait(lstm->barrier, ltid);
    libxsmm_internal_matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, i, j, 0, m * n), i2, i3, start_thread, tid, lstm->nThreads);
    libxsmm_barrier_wait(lstm->barrier, ltid);
    libxsmm_internal_matrix_eltwise_mult(m * n, i1, i3, &LIBXSMM_VLA_ACCESS(2, djdi, j, 0, m * n), start_thread, tid, lstm->nThreads);
    /* compute djdf */
    if (j >= 1) {
      libxsmm_internal_matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, djdd, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, d, j-1, 0, m * n), f1, start_thread, tid, lstm->nThreads);
      libxsmm_internal_matrix_complement(m * n, &LIBXSMM_VLA_ACCESS(2, f, j, 0, m * n), f2, start_thread, tid, lstm->nThreads);
      libxsmm_barrier_wait(lstm->barrier, ltid);
      libxsmm_internal_matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, f, j, 0, m * n), f2, f3, start_thread, tid, lstm->nThreads);
      libxsmm_barrier_wait(lstm->barrier, ltid);
      libxsmm_internal_matrix_eltwise_mult(m * n, f1, f3, &LIBXSMM_VLA_ACCESS(2, djdf, j, 0, m * n), start_thread, tid, lstm->nThreads);
    } else {
      /* djdf is zero for j == 0 */
      libxsmm_internal_matrix_zero(m * n, &LIBXSMM_VLA_ACCESS(2, djdf, j, 0, m * n), start_thread, tid, lstm->nThreads);
    }
    /* compute djdo */
    libxsmm_internal_matrix_tanh(m * n, &LIBXSMM_VLA_ACCESS(2, d, j, 0, m * n), o1, start_thread, tid, lstm->nThreads);
    libxsmm_internal_matrix_complement(m * n, &LIBXSMM_VLA_ACCESS(2, o, j, 0, m * n), o2, start_thread, tid, lstm->nThreads);
    libxsmm_barrier_wait(lstm->barrier, ltid);
    libxsmm_internal_matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, delta, j, 0, m * n), o1, o1, start_thread, tid, lstm->nThreads);
    libxsmm_internal_matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, o, j, 0, m * n), o2, o2, start_thread, tid, lstm->nThreads);
    libxsmm_barrier_wait(lstm->barrier, ltid);
    libxsmm_internal_matrix_eltwise_mult(m * n, o1, o2, &LIBXSMM_VLA_ACCESS(2, djdo, j, 0, m * n), start_thread, tid, lstm->nThreads);
    libxsmm_barrier_wait(lstm->barrier, ltid);
    if (pass == 1 || pass == 3) {
      /* compute djdx */
      libxsmm_bgemm_st(handlewd, wi, &LIBXSMM_VLA_ACCESS(2, djdi, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, djdx, j, 0, k * n), start_thread, tid);
      libxsmm_bgemm_st(handlewd, wf, &LIBXSMM_VLA_ACCESS(2, djdf, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, djdx, j, 0, k * n), start_thread, tid);
      libxsmm_bgemm_st(handlewd, wo, &LIBXSMM_VLA_ACCESS(2, djdo, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, djdx, j, 0, k * n), start_thread, tid);
      libxsmm_bgemm_st(handlewd, wc, &LIBXSMM_VLA_ACCESS(2, djdc, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, djdx, j, 0, k * n), start_thread, tid);
    }
  }
  if (pass == 2 || pass == 3) {
    /* Reorganizing djdi, djdf, dfdo, djdc */
    for (j = 0; j < t; ++j) {
      libxsmm_bgemm_convert_b_to_a(handleud, &LIBXSMM_VLA_ACCESS(2, djdi, j, 0, m * n), &m, &LIBXSMM_VLA_ACCESS(2, djdiM, j, 0, m * n));
      libxsmm_bgemm_convert_b_to_a(handleud, &LIBXSMM_VLA_ACCESS(2, djdf, j, 0, m * n), &m, &LIBXSMM_VLA_ACCESS(2, djdfM, j, 0, m * n));
      libxsmm_bgemm_convert_b_to_a(handleud, &LIBXSMM_VLA_ACCESS(2, djdo, j, 0, m * n), &m, &LIBXSMM_VLA_ACCESS(2, djdoM, j, 0, m * n));
      libxsmm_bgemm_convert_b_to_a(handleud, &LIBXSMM_VLA_ACCESS(2, djdc, j, 0, m * n), &m, &LIBXSMM_VLA_ACCESS(2, djdcM, j, 0, m * n));
      libxsmm_barrier_wait(lstm->barrier, ltid);
    }
    /* compute djdw */
    for (j = 0; j < t; ++j) {
      libxsmm_bgemm_st(handledx, &LIBXSMM_VLA_ACCESS(2, djdiM, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, x, j, 0, k * n), djdwi, start_thread, tid);
      libxsmm_bgemm_st(handledx, &LIBXSMM_VLA_ACCESS(2, djdfM, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, x, j, 0, k * n), djdwf, start_thread, tid);
      libxsmm_bgemm_st(handledx, &LIBXSMM_VLA_ACCESS(2, djdoM, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, x, j, 0, k * n), djdwo, start_thread, tid);
      libxsmm_bgemm_st(handledx, &LIBXSMM_VLA_ACCESS(2, djdcM, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, x, j, 0, k * n), djdwc, start_thread, tid);
    }
    /* compute djdr */
    for (j = 0; j < t-1; ++j) {
      libxsmm_bgemm_st(handledh, &LIBXSMM_VLA_ACCESS(2, djdiM, j+1, 0, m * n), &LIBXSMM_VLA_ACCESS(2, h, j, 0, m * n), djdri, start_thread, tid);
      libxsmm_bgemm_st(handledh, &LIBXSMM_VLA_ACCESS(2, djdfM, j+1, 0, m * n), &LIBXSMM_VLA_ACCESS(2, h, j, 0, m * n), djdrf, start_thread, tid);
      libxsmm_bgemm_st(handledh, &LIBXSMM_VLA_ACCESS(2, djdoM, j+1, 0, m * n), &LIBXSMM_VLA_ACCESS(2, h, j, 0, m * n), djdro, start_thread, tid);
      libxsmm_bgemm_st(handledh, &LIBXSMM_VLA_ACCESS(2, djdcM, j+1, 0, m * n), &LIBXSMM_VLA_ACCESS(2, h, j, 0, m * n), djdrc, start_thread, tid);
    }
    /* compute djdb */
    for (j = 0; j < t-1; j++) {
      libxsmm_internal_matrix_add(m * n, &LIBXSMM_VLA_ACCESS(2, djdi, j, 0, m * n), djdbi, djdbi, start_thread, tid, lstm->nThreads);
      libxsmm_internal_matrix_add(m * n, &LIBXSMM_VLA_ACCESS(2, djdf, j, 0, m * n), djdbf, djdbf, start_thread, tid, lstm->nThreads);
      libxsmm_internal_matrix_add(m * n, &LIBXSMM_VLA_ACCESS(2, djdo, j, 0, m * n), djdbo, djdbo, start_thread, tid, lstm->nThreads);
      libxsmm_internal_matrix_add(m * n, &LIBXSMM_VLA_ACCESS(2, djdc, j, 0, m * n), djdbc, djdbc, start_thread, tid, lstm->nThreads);
    }
    libxsmm_barrier_wait(lstm->barrier, ltid);
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

