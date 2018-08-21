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
extern double Gbl_t_input_total, Gbl_t_recur_total, Gbl_t_eltwise_total, Gbl_t_nonlin_total;
extern unsigned long long Gbl_t_input, Gbl_t_recur, Gbl_t_eltwise, Gbl_t_nonlin;
extern double Gbl_duration_input, Gbl_duration_recur, Gbl_duration_eltwise, Gbl_duration_nonlin;
#endif


LIBXSMM_API libxsmm_dnn_grucell* libxsmm_dnn_create_grucell(libxsmm_dnn_grucell_desc grucell_desc, libxsmm_dnn_err_t* status)
{
  libxsmm_dnn_grucell* handle = 0;
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

  handle = (libxsmm_dnn_grucell*)malloc(sizeof(libxsmm_dnn_grucell));
  if (0 != handle) {
    *status = LIBXSMM_DNN_SUCCESS;
    /* zero entire content; not only safer but also sets data and code pointers to NULL */
    memset(handle, 0, sizeof(*handle));
    /* initialize known handle components */
    handle->nThreads = grucell_desc.nThreads;
    handle->desc = grucell_desc;
    handle->datatype_in = grucell_desc.datatype_in;
    handle->datatype_out = grucell_desc.datatype_out;
    handle->reuse = grucell_desc.reuse;
    handle->pass = grucell_desc.pass;
    if ( (grucell_desc.datatype_in != LIBXSMM_DNN_DATATYPE_F32) || (grucell_desc.datatype_out != LIBXSMM_DNN_DATATYPE_F32) ) {
      /* error */
      *status = LIBXSMM_DNN_ERR_UNSUPPORTED_DATATYPE;
      return handle;
    }
    handle->buffer_format = grucell_desc.buffer_format;
    handle->m = grucell_desc.m;
    handle->n = grucell_desc.n;
    handle->k = grucell_desc.k;
    handle->t = grucell_desc.t;
    if (grucell_desc.t < 2) {
      *status = LIBXSMM_DNN_ERR_TIME_STEPS_TOO_SMALL;
    }
    handle->bm = grucell_desc.bm;
    handle->bn = grucell_desc.bn;
    handle->bk = grucell_desc.bk;
    handle->b_m1 = b_m1;
    handle->b_n1 = b_n1;
    handle->b_k1 = b_k1;
    handle->b_m2 = b_m2;
    handle->b_n2 = b_n2;
    handle->b_k2 = b_k2;
    if (handle->pass == 0) {
      handle->handleux = libxsmm_bgemm_handle_create(handle->nThreads, LIBXSMM_GEMM_PRECISION(float), LIBXSMM_GEMM_PRECISION(float),
        handle->m, handle->n, handle->k, &(handle->bm), &(handle->bn), &(handle->bk), &(handle->b_m1), &(handle->b_n1), &(handle->b_k1), &(handle->b_k2),
        &alpha, &beta, &gemm_flags, &strategy, &order);
      handle->handlewh = libxsmm_bgemm_handle_create(handle->nThreads, LIBXSMM_GEMM_PRECISION(float), LIBXSMM_GEMM_PRECISION(float),
        handle->m, handle->n, handle->m, &(handle->bm), &(handle->bn), &(handle->bm), &(handle->b_m1), &(handle->b_n1), &(handle->b_m1), &(handle->b_m2),
        &alpha, &beta, &gemm_flags, &strategy, &order);
      handle->handlett = libxsmm_bgemm_handle_create(handle->nThreads, LIBXSMM_GEMM_PRECISION(float), LIBXSMM_GEMM_PRECISION(float),
        handle->m, handle->n*handle->t, handle->k, &(handle->bm), &(handle->bn), &(handle->bk), &(handle->b_m1), &(handle->b_n1), &(handle->b_k1), &(handle->b_k2),
        &alpha, &beta, &gemm_flags, &strategy, &order);
    } else {
      handle->handleux = libxsmm_bgemm_handle_create(handle->nThreads, LIBXSMM_GEMM_PRECISION(float), LIBXSMM_GEMM_PRECISION(float),
        handle->m, handle->n, handle->m, &(handle->bm), &(handle->bn), &(handle->bm), &(handle->b_m1), &(handle->b_n1), &(handle->b_m1), &(handle->b_m2),
        &alpha, &beta, &gemm_flags, &strategy, &order); /* U^T*delta */
      handle->handlewh = libxsmm_bgemm_handle_create(handle->nThreads, LIBXSMM_GEMM_PRECISION(float), LIBXSMM_GEMM_PRECISION(float),
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
    handle->r1t = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->r2t = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->z1t = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->z2t = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->g1t = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->g2t = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->g3  = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->h1  = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->h2  = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->h3  = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->r   = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->z   = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->g   = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->d4  = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->d5  = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->d6  = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->d7  = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->d8  = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->d9  = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->d10 = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->d11 = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->d12 = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->d13 = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->d14 = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->d15 = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->d16 = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->d17 = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->d18 = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->d19 = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->d20 = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->d21 = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->d22 = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->d23 = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->d10M  = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->d11M  = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->d18M  = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->djdwr = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->djdwz = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->djdwg = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->djdxt = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->djdur = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->djduz = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->djdug = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->djdht = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->djdbr = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->djdbz = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->djdbg = (libxsmm_dnn_tensor*)malloc(sizeof(libxsmm_dnn_tensor));
    handle->barrier = libxsmm_barrier_create(handle->nThreads, 1);
    if (NULL == handle->r1t || NULL == handle->r2t || NULL == handle->z1t || NULL == handle->z2t || NULL == handle->g1t ||
        NULL == handle->g2t || NULL == handle->g3 || NULL == handle->h1 || NULL == handle->h2 || NULL == handle->h3 ||
        NULL == handle->r || NULL == handle->z || NULL == handle->g || NULL == handle->barrier ||
        NULL == handle->djdwr || NULL == handle->djdwz ||NULL == handle->djdwg || NULL == handle->djdxt ||
        NULL == handle->djdur || NULL == handle->djduz ||NULL == handle->djdug || NULL == handle->djdht ||
        NULL == handle->djdbr || NULL == handle->djdbz ||NULL == handle->djdbg ||
        NULL == handle->d4 || NULL == handle->d5 || NULL == handle->d6 || NULL == handle->d7 || NULL == handle->d8 ||
        NULL == handle->d9 || NULL == handle->d10 || NULL == handle->d11 || NULL == handle->d12 || NULL == handle->d13 ||
        NULL == handle->d14 || NULL == handle->d15 || NULL == handle->d16 || NULL == handle->d17 || NULL == handle->d18 ||
        NULL == handle->d19 || NULL == handle->d20 || NULL == handle->d21 || NULL == handle->d22 || NULL == handle->d23 ||
        NULL == handle->d10M || NULL == handle->d11M || NULL == handle->d18M)
    {
      free(handle->r1t); free(handle->r2t); free(handle->z1t); free(handle->z2t); free(handle->g1t);
      free(handle->g2t); free(handle->g3); free(handle->h1); free(handle->h2); free(handle->h3);
      free(handle->r); free(handle->z); free(handle->g);
      free(handle->djdwr); free(handle->djdwz); free(handle->djdwg); free(handle->djdxt);
      free(handle->djdur); free(handle->djduz); free(handle->djdug); free(handle->djdht);
      free(handle->djdbr); free(handle->djdbz); free(handle->djdbg);
      free(handle->d4); free(handle->d5); free(handle->d6); free(handle->d7); free(handle->d8); free(handle->d9);
      free(handle->d10); free(handle->d11); free(handle->d12); free(handle->d13); free(handle->d14); free(handle->d15);
      free(handle->d16); free(handle->d17); free(handle->d18); free(handle->d19); free(handle->d20); free(handle->d21);
      free(handle->d22); free(handle->d23); free(handle->d10M); free(handle->d11M); free(handle->d18M);
      *status = LIBXSMM_DNN_ERR_CREATE_HANDLE;
    }
  } else {
    *status = LIBXSMM_DNN_ERR_CREATE_HANDLE;
  }
  return handle;
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_destroy_grucell(const libxsmm_dnn_grucell* handle)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
  if (0 != handle) {
    free(handle->r1t); free(handle->r2t); free(handle->z1t); free(handle->z2t); free(handle->g1t);
    free(handle->g2t); free(handle->g3); free(handle->h1); free(handle->h2); free(handle->h3);
    free(handle->r); free(handle->z); free(handle->g);
    free(handle->djdwr); free(handle->djdwz); free(handle->djdwg); free(handle->djdxt);
    free(handle->djdur); free(handle->djduz); free(handle->djdug); free(handle->djdht);
    free(handle->djdbr); free(handle->djdbz); free(handle->djdbg);
    free(handle->d4); free(handle->d5); free(handle->d6); free(handle->d7); free(handle->d8); free(handle->d9);
    free(handle->d10); free(handle->d11); free(handle->d12); free(handle->d13); free(handle->d14); free(handle->d15);
    free(handle->d16); free(handle->d17); free(handle->d18); free(handle->d19); free(handle->d20); free(handle->d21);
    free(handle->d22); free(handle->d23); free(handle->d10M); free(handle->d11M); free(handle->d18M);
    /* Deallocate barrier */
    if (handle->barrier != 0 ) { libxsmm_barrier_release((const libxsmm_barrier*)handle->barrier); }
    /* deallocate handle structure */
    free(/*remove constness*/(libxsmm_dnn_grucell*)handle);
  }
  return status;
}


LIBXSMM_API libxsmm_dnn_tensor_datalayout* libxsmm_dnn_grucell_create_tensor_datalayout(const libxsmm_dnn_grucell* handle, const libxsmm_dnn_tensor_type type, libxsmm_dnn_err_t* status)
{
  libxsmm_dnn_tensor_datalayout* layout = 0;
  *status = LIBXSMM_DNN_SUCCESS;
  layout = 0;
  if (handle != 0) {
    layout = (libxsmm_dnn_tensor_datalayout*) malloc(sizeof(libxsmm_dnn_tensor_datalayout));
    if (layout != 0) {
      memset(layout, 0, sizeof(libxsmm_dnn_tensor_datalayout));
      /*layout->custom_format = handle->custom_format_type;*/
      if ( (type == LIBXSMM_DNN_GRU_REGULAR_INPUT)          || (type == LIBXSMM_DNN_GRU_GRADIENT_INPUT)  ||
           (type == LIBXSMM_DNN_GRU_REGULAR_HIDDEN_STATE)   || (type == LIBXSMM_DNN_GRU_GRADIENT_HIDDEN_STATE) ||
           (type == LIBXSMM_DNN_GRU_REGULAR_WEIGHT_R)       || (type == LIBXSMM_DNN_GRU_GRADIENT_WEIGHT_R) ||
           (type == LIBXSMM_DNN_GRU_REGULAR_WEIGHT_Z)       || (type == LIBXSMM_DNN_GRU_GRADIENT_WEIGHT_Z) ||
           (type == LIBXSMM_DNN_GRU_REGULAR_WEIGHT_G)       || (type == LIBXSMM_DNN_GRU_GRADIENT_WEIGHT_G) ||
           (type == LIBXSMM_DNN_GRU_REGULAR_RECUR_WEIGHT_R) || (type == LIBXSMM_DNN_GRU_GRADIENT_RECUR_WEIGHT_R) ||
           (type == LIBXSMM_DNN_GRU_REGULAR_RECUR_WEIGHT_Z) || (type == LIBXSMM_DNN_GRU_GRADIENT_RECUR_WEIGHT_Z) ||
           (type == LIBXSMM_DNN_GRU_REGULAR_RECUR_WEIGHT_G) || (type == LIBXSMM_DNN_GRU_GRADIENT_RECUR_WEIGHT_G) ||
           (type == LIBXSMM_DNN_GRU_REGULAR_BIAS_R)         || (type == LIBXSMM_DNN_GRU_GRADIENT_BIAS_R)   ||
           (type == LIBXSMM_DNN_GRU_REGULAR_BIAS_Z)         || (type == LIBXSMM_DNN_GRU_GRADIENT_BIAS_Z)   ||
           (type == LIBXSMM_DNN_GRU_REGULAR_BIAS_G)         || (type == LIBXSMM_DNN_GRU_GRADIENT_BIAS_G) ) {
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
                if ( (type == LIBXSMM_DNN_GRU_REGULAR_INPUT) || (type == LIBXSMM_DNN_GRU_GRADIENT_INPUT) ) {
                  layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_RLK;
                  layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_RLN;
                  layout->dim_type[2] = LIBXSMM_DNN_TENSOR_DIMTYPE_RLK;
                  layout->dim_type[3] = LIBXSMM_DNN_TENSOR_DIMTYPE_RLN;
                  layout->dim_size[0] = handle->bk;
                  layout->dim_size[1] = handle->bn;
                  layout->dim_size[2] = handle->k / handle->bk;
                  layout->dim_size[3] = handle->n / handle->bn;
                } else if ( (type == LIBXSMM_DNN_GRU_REGULAR_HIDDEN_STATE) || (type == LIBXSMM_DNN_GRU_GRADIENT_HIDDEN_STATE) ) {
                  layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_RLM;
                  layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_RLN;
                  layout->dim_type[2] = LIBXSMM_DNN_TENSOR_DIMTYPE_RLM;
                  layout->dim_type[3] = LIBXSMM_DNN_TENSOR_DIMTYPE_RLN;
                  layout->dim_size[0] = handle->bm;
                  layout->dim_size[1] = handle->bn;
                  layout->dim_size[2] = handle->m / handle->bm;
                  layout->dim_size[3] = handle->n / handle->bn;
                } else if ( (type == LIBXSMM_DNN_GRU_REGULAR_WEIGHT_R) || (type == LIBXSMM_DNN_GRU_GRADIENT_WEIGHT_R) ||
                            (type == LIBXSMM_DNN_GRU_REGULAR_WEIGHT_Z) || (type == LIBXSMM_DNN_GRU_GRADIENT_WEIGHT_Z) ||
                            (type == LIBXSMM_DNN_GRU_REGULAR_WEIGHT_G) || (type == LIBXSMM_DNN_GRU_GRADIENT_WEIGHT_G) ) {
                  layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_RLM;
                  layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_RLK;
                  layout->dim_type[2] = LIBXSMM_DNN_TENSOR_DIMTYPE_RLK;
                  layout->dim_type[3] = LIBXSMM_DNN_TENSOR_DIMTYPE_RLM;
                  layout->dim_size[0] = handle->bm;
                  layout->dim_size[1] = handle->bk;
                  layout->dim_size[2] = handle->k / handle->bk;
                  layout->dim_size[3] = handle->m / handle->bm;
                } else if ( (type == LIBXSMM_DNN_GRU_REGULAR_RECUR_WEIGHT_R) || (type == LIBXSMM_DNN_GRU_GRADIENT_RECUR_WEIGHT_R) ||
                            (type == LIBXSMM_DNN_GRU_REGULAR_RECUR_WEIGHT_Z) || (type == LIBXSMM_DNN_GRU_GRADIENT_RECUR_WEIGHT_Z) ||
                            (type == LIBXSMM_DNN_GRU_REGULAR_RECUR_WEIGHT_G) || (type == LIBXSMM_DNN_GRU_GRADIENT_RECUR_WEIGHT_G) ) {
                  layout->dim_type[0] = LIBXSMM_DNN_TENSOR_DIMTYPE_RLM;
                  layout->dim_type[1] = LIBXSMM_DNN_TENSOR_DIMTYPE_RLM;
                  layout->dim_type[2] = LIBXSMM_DNN_TENSOR_DIMTYPE_RLM;
                  layout->dim_type[3] = LIBXSMM_DNN_TENSOR_DIMTYPE_RLM;
                  layout->dim_size[0] = handle->bm;
                  layout->dim_size[1] = handle->bm;
                  layout->dim_size[2] = handle->m / handle->bm;
                  layout->dim_size[3] = handle->m / handle->bm;
                } else if ( (type == LIBXSMM_DNN_GRU_REGULAR_BIAS_R) || (type == LIBXSMM_DNN_GRU_GRADIENT_BIAS_R) ||
                            (type == LIBXSMM_DNN_GRU_REGULAR_BIAS_Z) || (type == LIBXSMM_DNN_GRU_GRADIENT_BIAS_Z) ||
                            (type == LIBXSMM_DNN_GRU_REGULAR_BIAS_G) || (type == LIBXSMM_DNN_GRU_GRADIENT_BIAS_G) ) {
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


LIBXSMM_API size_t libxsmm_dnn_grucell_get_scratch_size(const libxsmm_dnn_grucell* handle, const libxsmm_dnn_compute_kind kind, libxsmm_dnn_err_t* status)
{
  size_t sizeof_datatype = sizeof(float);
  size_t size = 0;
  *status = LIBXSMM_DNN_SUCCESS;

  if (0 != handle) {
    switch (kind) {
      case LIBXSMM_DNN_COMPUTE_KIND_FWD: {
                                           size += (size_t)handle->m * handle->n * sizeof_datatype * handle->t; /* r1t */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype * handle->t; /* r2t */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype * handle->t; /* z1t */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype * handle->t; /* z2t */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype * handle->t; /* g1t */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype * handle->t; /* g2t */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype; /* g3 */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype; /* h1 */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype; /* h2 */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype; /* h3 */
                                           size += 64;
                                         } break;
      case LIBXSMM_DNN_COMPUTE_KIND_BWD:
      case LIBXSMM_DNN_COMPUTE_KIND_UPD:
      case LIBXSMM_DNN_COMPUTE_KIND_ALL: {
                                           size += (size_t)handle->m * handle->n * sizeof_datatype; /* d4 */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype; /* d5 */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype; /* d6 */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype; /* d7 */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype; /* d8 */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype; /* d9 */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype; /* d10 */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype; /* d11 */
                                           size += 64;
                                           size += (size_t)handle->k * handle->n * sizeof_datatype; /* d12 */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype; /* d13 */
                                           size += 64;
                                           size += (size_t)handle->k * handle->n * sizeof_datatype; /* d14 */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype; /* d15 */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype; /* d16 */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype; /* d17 */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype; /* d18 */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype; /* d19 */
                                           size += 64;
                                           size += (size_t)handle->k * handle->n * sizeof_datatype; /* d20 */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype; /* d21 */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype; /* d22 */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype; /* d23 */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype; /* d10M */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype; /* d11M */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype; /* d18M */
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


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_grucell_bind_scratch(libxsmm_dnn_grucell* handle, const libxsmm_dnn_compute_kind kind, const void* scratch)
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
                                             handle->r1t->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->r1t->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->m * handle->n * sizeof_datatype * handle->t;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->r2t->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->r2t->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->m * handle->n * sizeof_datatype * handle->t;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->z1t->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->z1t->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->m * handle->n * sizeof_datatype * handle->t;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->z2t->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->z2t->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->m * handle->n * sizeof_datatype * handle->t;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->g1t->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->g1t->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->m * handle->n * sizeof_datatype * handle->t;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->g2t->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->g2t->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->m * handle->n * sizeof_datatype * handle->t;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->g3->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->g3->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->h1->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->h1->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->h2->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->h2->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->h3->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->h3->data = (void*)(address+offset);
                                           }
                                         } break;
      case LIBXSMM_DNN_COMPUTE_KIND_BWD:
      case LIBXSMM_DNN_COMPUTE_KIND_UPD:
      case LIBXSMM_DNN_COMPUTE_KIND_ALL: {
                                           if (address % 64 == 0) {
                                             handle->d4->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->d4->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->d5->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->d5->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->d6->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->d6->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->d7->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->d7->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->d8->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->d8->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->d9->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->d9->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->d10->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->d10->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->d11->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->d11->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->d12->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->d12->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->k * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->d13->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->d13->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->d14->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->d14->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->k * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->d15->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->d15->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->d16->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->d16->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->d17->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->d17->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->d18->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->d18->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->d19->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->d19->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->d20->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->d20->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->k * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->d21->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->d21->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->d22->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->d22->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->d23->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->d23->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->d10M->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->d10M->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->d11M->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->d11M->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->m * handle->n * sizeof_datatype;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->d18M->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->d18M->data = (void*)(address+offset);
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


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_grucell_release_scratch(libxsmm_dnn_grucell* handle, const libxsmm_dnn_compute_kind kind)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  if (0 != handle) {
    switch (kind) {
      case LIBXSMM_DNN_COMPUTE_KIND_FWD: {
                                           handle->r1t->data = 0;
                                           handle->r2t->data = 0;
                                           handle->z1t->data = 0;
                                           handle->z2t->data = 0;
                                           handle->g1t->data = 0;
                                           handle->g2t->data = 0;
                                           handle->g3->data = 0;
                                           handle->h1->data = 0;
                                           handle->h2->data = 0;
                                           handle->h3->data = 0;
                                           handle->r1t = 0;
                                           handle->r2t = 0;
                                           handle->z1t = 0;
                                           handle->z2t = 0;
                                           handle->g1t = 0;
                                           handle->g2t = 0;
                                           handle->g3 = 0;
                                           handle->h1 = 0;
                                           handle->h2 = 0;
                                           handle->h3 = 0;
                                         } break;
      case LIBXSMM_DNN_COMPUTE_KIND_BWD:
      case LIBXSMM_DNN_COMPUTE_KIND_UPD:
      case LIBXSMM_DNN_COMPUTE_KIND_ALL: {
                                           handle->r1t->data = 0;
                                           handle->r2t->data = 0;
                                           handle->z1t->data = 0;
                                           handle->z2t->data = 0;
                                           handle->g1t->data = 0;
                                           handle->g2t->data = 0;
                                           handle->g3->data = 0;
                                           handle->h1->data = 0;
                                           handle->h2->data = 0;
                                           handle->h3->data = 0;
                                           handle->d4->data = 0;
                                           handle->d5->data = 0;
                                           handle->d6->data = 0;
                                           handle->d7->data = 0;
                                           handle->d8->data = 0;
                                           handle->d9->data = 0;
                                           handle->d10->data = 0;
                                           handle->d11->data = 0;
                                           handle->d12->data = 0;
                                           handle->d13->data = 0;
                                           handle->d14->data = 0;
                                           handle->d15->data = 0;
                                           handle->d16->data = 0;
                                           handle->d17->data = 0;
                                           handle->d18->data = 0;
                                           handle->d19->data = 0;
                                           handle->d20->data = 0;
                                           handle->d21->data = 0;
                                           handle->d22->data = 0;
                                           handle->d23->data = 0;
                                           handle->d10M->data = 0;
                                           handle->d11M->data = 0;
                                           handle->d18M->data = 0;
                                           handle->r1t = 0;
                                           handle->r2t = 0;
                                           handle->z1t = 0;
                                           handle->z2t = 0;
                                           handle->g1t = 0;
                                           handle->g2t = 0;
                                           handle->g3 = 0;
                                           handle->h1 = 0;
                                           handle->h2 = 0;
                                           handle->h3 = 0;
                                           handle->d4 = 0;
                                           handle->d5 = 0;
                                           handle->d6 = 0;
                                           handle->d7 = 0;
                                           handle->d8 = 0;
                                           handle->d9 = 0;
                                           handle->d10 = 0;
                                           handle->d11 = 0;
                                           handle->d12 = 0;
                                           handle->d13 = 0;
                                           handle->d14 = 0;
                                           handle->d15 = 0;
                                           handle->d16 = 0;
                                           handle->d17 = 0;
                                           handle->d18 = 0;
                                           handle->d19 = 0;
                                           handle->d20 = 0;
                                           handle->d21 = 0;
                                           handle->d22 = 0;
                                           handle->d23 = 0;
                                           handle->d10M = 0;
                                           handle->d11M = 0;
                                           handle->d18M = 0;
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


LIBXSMM_API size_t libxsmm_dnn_grucell_get_internalstate_size(const libxsmm_dnn_grucell* handle, const libxsmm_dnn_compute_kind kind, libxsmm_dnn_err_t* status)
{
  size_t sizeof_datatype = sizeof(float);
  size_t size = 0;
  *status = LIBXSMM_DNN_SUCCESS;

  if (0 != handle) {
    switch (kind) {
      case LIBXSMM_DNN_COMPUTE_KIND_FWD: {
                                           size += (size_t)handle->m * handle->n * sizeof_datatype * handle->t; /* r */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype * handle->t; /* z */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype * handle->t; /* g */
                                           size += 64;
                                         } break;
      case LIBXSMM_DNN_COMPUTE_KIND_BWD:
      case LIBXSMM_DNN_COMPUTE_KIND_UPD:
      case LIBXSMM_DNN_COMPUTE_KIND_ALL: {
                                           size += (size_t)handle->m * handle->n * sizeof_datatype * handle->t; /* r */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype * handle->t; /* z */
                                           size += 64;
                                           size += (size_t)handle->m * handle->n * sizeof_datatype * handle->t; /* g */
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


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_grucell_bind_internalstate(libxsmm_dnn_grucell* handle, const libxsmm_dnn_compute_kind kind, const void* internalstate)
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
                                             handle->r->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->r->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->m * handle->n * sizeof_datatype * handle->t;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->z->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->z->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->m * handle->n * sizeof_datatype * handle->t;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->g->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->g->data = (void*)(address+offset);
                                           }
                                         } break;
      case LIBXSMM_DNN_COMPUTE_KIND_BWD:
      case LIBXSMM_DNN_COMPUTE_KIND_UPD:
      case LIBXSMM_DNN_COMPUTE_KIND_ALL: {
                                           if (address % 64 == 0) {
                                             handle->r->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->r->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->m * handle->n * sizeof_datatype * handle->t;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->z->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->z->data = (void*)(address+offset);
                                           }
                                           scratch_size = (size_t)handle->m * handle->n * sizeof_datatype * handle->t;
                                           address += scratch_size + 64;
                                           if (address % 64 == 0) {
                                             handle->g->data = (void*)address;
                                           } else {
                                             offset = (64 - address % 64);
                                             handle->g->data = (void*)(address+offset);
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


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_grucell_release_internalstate(libxsmm_dnn_grucell* handle, const libxsmm_dnn_compute_kind kind)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  if (0 != handle) {
    switch (kind) {
      case LIBXSMM_DNN_COMPUTE_KIND_FWD: {
                                           handle->r->data = 0;
                                           handle->z->data = 0;
                                           handle->g->data = 0;
                                           handle->r = 0;
                                           handle->z = 0;
                                           handle->g = 0;
                                         } break;
      case LIBXSMM_DNN_COMPUTE_KIND_BWD:
      case LIBXSMM_DNN_COMPUTE_KIND_UPD:
      case LIBXSMM_DNN_COMPUTE_KIND_ALL: {
                                           handle->r->data = 0;
                                           handle->z->data = 0;
                                           handle->g->data = 0;
                                           handle->r = 0;
                                           handle->z = 0;
                                           handle->g = 0;
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


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_grucell_assign_internalstate(libxsmm_dnn_grucell* handle, const void* rgoldtb, const void* zgoldtb, const void* ggoldtb)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  if (handle != 0 && rgoldtb != 0 && zgoldtb != 0 && ggoldtb != 0) {
    const libxsmm_blasint m = handle->m, n = handle->n, t = handle->t;
    LIBXSMM_VLA_DECL(2, const LIBXSMM_DNN_ELTWISE_FTYPE, rgold, (const LIBXSMM_DNN_ELTWISE_FTYPE*)rgoldtb, m * n);
    LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, r, (LIBXSMM_DNN_ELTWISE_FTYPE*)handle->r->data, m * n);
    LIBXSMM_VLA_DECL(2, const LIBXSMM_DNN_ELTWISE_FTYPE, zgold, (const LIBXSMM_DNN_ELTWISE_FTYPE*)zgoldtb, m * n);
    LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, z, (LIBXSMM_DNN_ELTWISE_FTYPE*)handle->z->data, m * n);
    LIBXSMM_VLA_DECL(2, const LIBXSMM_DNN_ELTWISE_FTYPE, ggold, (const LIBXSMM_DNN_ELTWISE_FTYPE*)ggoldtb, m * n);
    LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, g, (LIBXSMM_DNN_ELTWISE_FTYPE*)handle->g->data, m * n);
    libxsmm_blasint it;
    for (it = 0; it < t; ++it) {
      libxsmm_bgemm_copyin_b(handle->handlewd, &LIBXSMM_VLA_ACCESS(2, rgold, it, 0, m * n), &m, &LIBXSMM_VLA_ACCESS(2, r, it, 0, m * n));
      libxsmm_bgemm_copyin_b(handle->handlewd, &LIBXSMM_VLA_ACCESS(2, zgold, it, 0, m * n), &m, &LIBXSMM_VLA_ACCESS(2, z, it, 0, m * n));
      libxsmm_bgemm_copyin_b(handle->handlewd, &LIBXSMM_VLA_ACCESS(2, ggold, it, 0, m * n), &m, &LIBXSMM_VLA_ACCESS(2, g, it, 0, m * n));
    }
  } else {
    status = LIBXSMM_DNN_ERR_INVALID_HANDLE_TENSOR;
  }

  return status;
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_grucell_bind_tensor(libxsmm_dnn_grucell* handle, const libxsmm_dnn_tensor* tensor, const libxsmm_dnn_tensor_type type)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  /* check for tensor type */
  if ( (type != LIBXSMM_DNN_GRU_REGULAR_INPUT)         && (type != LIBXSMM_DNN_GRU_GRADIENT_INPUT)  &&
      (type != LIBXSMM_DNN_GRU_REGULAR_HIDDEN_STATE)   && (type != LIBXSMM_DNN_GRU_GRADIENT_HIDDEN_STATE) &&
      (type != LIBXSMM_DNN_GRU_REGULAR_WEIGHT_R)       && (type != LIBXSMM_DNN_GRU_GRADIENT_WEIGHT_R) &&
      (type != LIBXSMM_DNN_GRU_REGULAR_WEIGHT_Z)       && (type != LIBXSMM_DNN_GRU_GRADIENT_WEIGHT_Z) &&
      (type != LIBXSMM_DNN_GRU_REGULAR_WEIGHT_G)       && (type != LIBXSMM_DNN_GRU_GRADIENT_WEIGHT_G) &&
      (type != LIBXSMM_DNN_GRU_REGULAR_RECUR_WEIGHT_R) && (type != LIBXSMM_DNN_GRU_GRADIENT_RECUR_WEIGHT_R) &&
      (type != LIBXSMM_DNN_GRU_REGULAR_RECUR_WEIGHT_Z) && (type != LIBXSMM_DNN_GRU_GRADIENT_RECUR_WEIGHT_Z) &&
      (type != LIBXSMM_DNN_GRU_REGULAR_RECUR_WEIGHT_G) && (type != LIBXSMM_DNN_GRU_GRADIENT_RECUR_WEIGHT_G) &&
      (type != LIBXSMM_DNN_GRU_REGULAR_BIAS_R)         && (type != LIBXSMM_DNN_GRU_GRADIENT_BIAS_R)   &&
      (type != LIBXSMM_DNN_GRU_REGULAR_BIAS_Z)         && (type != LIBXSMM_DNN_GRU_GRADIENT_BIAS_Z)   &&
      (type != LIBXSMM_DNN_GRU_REGULAR_BIAS_G)         && (type != LIBXSMM_DNN_GRU_GRADIENT_BIAS_G) ) {
    status = LIBXSMM_DNN_ERR_UNKNOWN_TENSOR_TYPE;
    return status;
  }

  if (handle != 0 && tensor != 0) {
    libxsmm_dnn_tensor_datalayout* handle_layout = libxsmm_dnn_grucell_create_tensor_datalayout(handle, type, &status);

    if ( libxsmm_dnn_compare_tensor_datalayout(handle_layout, tensor->layout, &status) == 0 ) {
      if ( type == LIBXSMM_DNN_GRU_REGULAR_INPUT ) {
        handle->xt = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_GRU_GRADIENT_INPUT ) {
        handle->djdxt = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_GRU_REGULAR_HIDDEN_STATE ) {
        handle->h = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_GRU_GRADIENT_HIDDEN_STATE ) {
        handle->djdht = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_GRU_REGULAR_WEIGHT_R ) {
        handle->ur = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_GRU_GRADIENT_WEIGHT_R ) {
        handle->djdur = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_GRU_REGULAR_WEIGHT_Z ) {
        handle->uz = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_GRU_GRADIENT_WEIGHT_Z ) {
        handle->djduz = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_GRU_REGULAR_WEIGHT_G ) {
        handle->ug = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_GRU_GRADIENT_WEIGHT_G ) {
        handle->djdug = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_GRU_REGULAR_RECUR_WEIGHT_R ) {
        handle->wr = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_GRU_GRADIENT_RECUR_WEIGHT_R ) {
        handle->djdwr = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_GRU_REGULAR_RECUR_WEIGHT_Z ) {
        handle->wz = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_GRU_GRADIENT_RECUR_WEIGHT_Z ) {
        handle->djdwz = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_GRU_REGULAR_RECUR_WEIGHT_G ) {
        handle->wg = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_GRU_GRADIENT_RECUR_WEIGHT_G ) {
        handle->djdwg = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_GRU_REGULAR_BIAS_R ) {
        handle->br = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_GRU_GRADIENT_BIAS_R ) {
        handle->djdbr = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_GRU_REGULAR_BIAS_Z ) {
        handle->bz = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_GRU_GRADIENT_BIAS_Z ) {
        handle->djdbz = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_GRU_REGULAR_BIAS_G ) {
        handle->bg = (libxsmm_dnn_tensor*)tensor;
      } else if ( type == LIBXSMM_DNN_GRU_GRADIENT_BIAS_G ) {
        handle->djdbg = (libxsmm_dnn_tensor*)tensor;
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


LIBXSMM_API libxsmm_dnn_tensor* libxsmm_dnn_grucell_get_tensor(libxsmm_dnn_grucell* handle, const libxsmm_dnn_tensor_type type, libxsmm_dnn_err_t* status)
{
  libxsmm_dnn_tensor* tensor = 0;
  LIBXSMM_UNUSED(status/*TODO*/);

  /* check for tensor type */
  if ( (type != LIBXSMM_DNN_GRU_REGULAR_INPUT)         && (type != LIBXSMM_DNN_GRU_GRADIENT_INPUT)  &&
      (type != LIBXSMM_DNN_GRU_REGULAR_HIDDEN_STATE)   && (type != LIBXSMM_DNN_GRU_GRADIENT_HIDDEN_STATE) &&
      (type != LIBXSMM_DNN_GRU_REGULAR_WEIGHT_R)       && (type != LIBXSMM_DNN_GRU_GRADIENT_WEIGHT_R) &&
      (type != LIBXSMM_DNN_GRU_REGULAR_WEIGHT_Z)       && (type != LIBXSMM_DNN_GRU_GRADIENT_WEIGHT_Z) &&
      (type != LIBXSMM_DNN_GRU_REGULAR_WEIGHT_G)       && (type != LIBXSMM_DNN_GRU_GRADIENT_WEIGHT_G) &&
      (type != LIBXSMM_DNN_GRU_REGULAR_RECUR_WEIGHT_R) && (type != LIBXSMM_DNN_GRU_GRADIENT_RECUR_WEIGHT_R) &&
      (type != LIBXSMM_DNN_GRU_REGULAR_RECUR_WEIGHT_Z) && (type != LIBXSMM_DNN_GRU_GRADIENT_RECUR_WEIGHT_Z) &&
      (type != LIBXSMM_DNN_GRU_REGULAR_RECUR_WEIGHT_G) && (type != LIBXSMM_DNN_GRU_GRADIENT_RECUR_WEIGHT_G) &&
      (type != LIBXSMM_DNN_GRU_REGULAR_BIAS_R)         && (type != LIBXSMM_DNN_GRU_GRADIENT_BIAS_R)   &&
      (type != LIBXSMM_DNN_GRU_REGULAR_BIAS_Z)         && (type != LIBXSMM_DNN_GRU_GRADIENT_BIAS_Z)   &&
      (type != LIBXSMM_DNN_GRU_REGULAR_BIAS_G)         && (type != LIBXSMM_DNN_GRU_GRADIENT_BIAS_G) ) {
    return tensor;
  }

  if (handle != 0) {
    if ( type == LIBXSMM_DNN_GRU_REGULAR_INPUT ) {
      tensor = handle->xt;
    } else if ( type == LIBXSMM_DNN_GRU_GRADIENT_INPUT ) {
      tensor = handle->djdxt;
    } else if ( type == LIBXSMM_DNN_GRU_REGULAR_HIDDEN_STATE ) {
      tensor = handle->h;
    } else if ( type == LIBXSMM_DNN_GRU_GRADIENT_HIDDEN_STATE ) {
      tensor = handle->djdht;
    } else if ( type == LIBXSMM_DNN_GRU_REGULAR_WEIGHT_R ) {
      tensor = handle->ur;
    } else if ( type == LIBXSMM_DNN_GRU_GRADIENT_WEIGHT_R ) {
      tensor = handle->djdur;
    } else if ( type == LIBXSMM_DNN_GRU_REGULAR_WEIGHT_Z ) {
      tensor = handle->uz;
    } else if ( type == LIBXSMM_DNN_GRU_GRADIENT_WEIGHT_Z ) {
      tensor = handle->djduz;
    } else if ( type == LIBXSMM_DNN_GRU_REGULAR_WEIGHT_G ) {
      tensor = handle->ug;
    } else if ( type == LIBXSMM_DNN_GRU_GRADIENT_WEIGHT_G ) {
      tensor = handle->djdug;
    } else if ( type == LIBXSMM_DNN_GRU_REGULAR_RECUR_WEIGHT_R ) {
      tensor = handle->wr;
    } else if ( type == LIBXSMM_DNN_GRU_GRADIENT_RECUR_WEIGHT_R ) {
      tensor = handle->djdwr;
    } else if ( type == LIBXSMM_DNN_GRU_REGULAR_RECUR_WEIGHT_Z ) {
      tensor = handle->wz;
    } else if ( type == LIBXSMM_DNN_GRU_GRADIENT_RECUR_WEIGHT_Z ) {
      tensor = handle->djdwz;
    } else if ( type == LIBXSMM_DNN_GRU_REGULAR_RECUR_WEIGHT_G ) {
      tensor = handle->wg;
    } else if ( type == LIBXSMM_DNN_GRU_GRADIENT_RECUR_WEIGHT_G ) {
      tensor = handle->djdwg;
    } else if ( type == LIBXSMM_DNN_GRU_REGULAR_BIAS_R ) {
      tensor = handle->br;
    } else if ( type == LIBXSMM_DNN_GRU_GRADIENT_BIAS_R ) {
      tensor = handle->djdbr;
    } else if ( type == LIBXSMM_DNN_GRU_REGULAR_BIAS_Z ) {
      tensor = handle->bz;
    } else if ( type == LIBXSMM_DNN_GRU_GRADIENT_BIAS_Z ) {
      tensor = handle->djdbz;
    } else if ( type == LIBXSMM_DNN_GRU_REGULAR_BIAS_G ) {
      tensor = handle->bg;
    } else if ( type == LIBXSMM_DNN_GRU_GRADIENT_BIAS_G ) {
      tensor = handle->djdbg;
    } else {
      /* cannot happen */
    }
  }

  return tensor;
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_grucell_release_tensor(libxsmm_dnn_grucell* handle, const libxsmm_dnn_tensor_type type)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  /* check for tensor type */
  if ( (type != LIBXSMM_DNN_GRU_REGULAR_INPUT)         && (type != LIBXSMM_DNN_GRU_GRADIENT_INPUT)  &&
      (type != LIBXSMM_DNN_GRU_REGULAR_HIDDEN_STATE)   && (type != LIBXSMM_DNN_GRU_GRADIENT_HIDDEN_STATE) &&
      (type != LIBXSMM_DNN_GRU_REGULAR_WEIGHT_R)       && (type != LIBXSMM_DNN_GRU_GRADIENT_WEIGHT_R) &&
      (type != LIBXSMM_DNN_GRU_REGULAR_WEIGHT_Z)       && (type != LIBXSMM_DNN_GRU_GRADIENT_WEIGHT_Z) &&
      (type != LIBXSMM_DNN_GRU_REGULAR_WEIGHT_G)       && (type != LIBXSMM_DNN_GRU_GRADIENT_WEIGHT_G) &&
      (type != LIBXSMM_DNN_GRU_REGULAR_RECUR_WEIGHT_R) && (type != LIBXSMM_DNN_GRU_GRADIENT_RECUR_WEIGHT_R) &&
      (type != LIBXSMM_DNN_GRU_REGULAR_RECUR_WEIGHT_Z) && (type != LIBXSMM_DNN_GRU_GRADIENT_RECUR_WEIGHT_Z) &&
      (type != LIBXSMM_DNN_GRU_REGULAR_RECUR_WEIGHT_G) && (type != LIBXSMM_DNN_GRU_GRADIENT_RECUR_WEIGHT_G) &&
      (type != LIBXSMM_DNN_GRU_REGULAR_BIAS_R)         && (type != LIBXSMM_DNN_GRU_GRADIENT_BIAS_R)   &&
      (type != LIBXSMM_DNN_GRU_REGULAR_BIAS_Z)         && (type != LIBXSMM_DNN_GRU_GRADIENT_BIAS_Z)   &&
      (type != LIBXSMM_DNN_GRU_REGULAR_BIAS_G)         && (type != LIBXSMM_DNN_GRU_GRADIENT_BIAS_G) ) {
    status = LIBXSMM_DNN_ERR_UNKNOWN_TENSOR_TYPE;
    return status;
  }

  if (handle != 0) {
    if ( type == LIBXSMM_DNN_GRU_REGULAR_INPUT ) {
      handle->xt = 0;
    } else if ( type == LIBXSMM_DNN_GRU_GRADIENT_INPUT ) {
      handle->djdxt = 0;
    } else if ( type == LIBXSMM_DNN_GRU_REGULAR_HIDDEN_STATE ) {
      handle->h = 0;
    } else if ( type == LIBXSMM_DNN_GRU_GRADIENT_HIDDEN_STATE ) {
      handle->djdht = 0;
    } else if ( type == LIBXSMM_DNN_GRU_REGULAR_WEIGHT_R ) {
      handle->ur = 0;
    } else if ( type == LIBXSMM_DNN_GRU_GRADIENT_WEIGHT_R ) {
      handle->djdur = 0;
    } else if ( type == LIBXSMM_DNN_GRU_REGULAR_WEIGHT_Z ) {
      handle->uz = 0;
    } else if ( type == LIBXSMM_DNN_GRU_GRADIENT_WEIGHT_Z ) {
      handle->djduz = 0;
    } else if ( type == LIBXSMM_DNN_GRU_REGULAR_WEIGHT_G ) {
      handle->ug = 0;
    } else if ( type == LIBXSMM_DNN_GRU_GRADIENT_WEIGHT_G ) {
      handle->djdug = 0;
    } else if ( type == LIBXSMM_DNN_GRU_REGULAR_RECUR_WEIGHT_R ) {
      handle->wr = 0;
    } else if ( type == LIBXSMM_DNN_GRU_GRADIENT_RECUR_WEIGHT_R ) {
      handle->djdwr = 0;
    } else if ( type == LIBXSMM_DNN_GRU_REGULAR_RECUR_WEIGHT_Z ) {
      handle->wz = 0;
    } else if ( type == LIBXSMM_DNN_GRU_GRADIENT_RECUR_WEIGHT_Z ) {
      handle->djdwz = 0;
    } else if ( type == LIBXSMM_DNN_GRU_REGULAR_RECUR_WEIGHT_G ) {
      handle->wg = 0;
    } else if ( type == LIBXSMM_DNN_GRU_GRADIENT_RECUR_WEIGHT_G ) {
      handle->djdwg = 0;
    } else if ( type == LIBXSMM_DNN_GRU_REGULAR_BIAS_R ) {
      handle->br = 0;
    } else if ( type == LIBXSMM_DNN_GRU_GRADIENT_BIAS_R ) {
      handle->djdbr = 0;
    } else if ( type == LIBXSMM_DNN_GRU_REGULAR_BIAS_Z ) {
      handle->bz = 0;
    } else if ( type == LIBXSMM_DNN_GRU_GRADIENT_BIAS_Z ) {
      handle->djdbz = 0;
    } else if ( type == LIBXSMM_DNN_GRU_REGULAR_BIAS_G ) {
      handle->bg = 0;
    } else if ( type == LIBXSMM_DNN_GRU_GRADIENT_BIAS_G ) {
      handle->djdbg = 0;
    } else {
      /* cannot happen */
    }
  }
  else {
    status = LIBXSMM_DNN_ERR_INVALID_HANDLE_TENSOR;
  }

  return status;
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_grucell_fwd(libxsmm_dnn_grucell* gru, int start_thread, int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
  libxsmm_blasint m = gru->m;
  libxsmm_blasint n = gru->n;
  libxsmm_blasint k = gru->k;
  libxsmm_blasint t = gru->t;
#if defined(LSTM_TIMING)
  libxsmm_blasint k = gru->k;
  const double tflops = 12;
  const double gflops = ((2.0 * m * n * k) + (2.0 * m * n * m) + (2.0 * m * n) + (tflops * m * n)) * 2.0; /* r and z */
  gflops += (m * n) + (2.0 * m * n * k) + (2.0 * m * n * m) + (tflops * m * n); /* g */
  gflops += 4.0 * (m * n); /* h */
  gflops *= (double)t * 1E-9; /* t time steps */
#endif
  int reuse = gru->reuse;
  LIBXSMM_DNN_ELTWISE_FTYPE *wr  = (LIBXSMM_DNN_ELTWISE_FTYPE*)gru->wr->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *wz  = (LIBXSMM_DNN_ELTWISE_FTYPE*)gru->wz->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *wg  = (LIBXSMM_DNN_ELTWISE_FTYPE*)gru->wg->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *xt  = (LIBXSMM_DNN_ELTWISE_FTYPE*)gru->xt->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *ur  = (LIBXSMM_DNN_ELTWISE_FTYPE*)gru->ur->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *uz  = (LIBXSMM_DNN_ELTWISE_FTYPE*)gru->uz->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *ug  = (LIBXSMM_DNN_ELTWISE_FTYPE*)gru->ug->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *h   = (LIBXSMM_DNN_ELTWISE_FTYPE*)gru->h->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *br  = (LIBXSMM_DNN_ELTWISE_FTYPE*)gru->br->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *bz  = (LIBXSMM_DNN_ELTWISE_FTYPE*)gru->bz->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *bg  = (LIBXSMM_DNN_ELTWISE_FTYPE*)gru->bg->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *r1t = (LIBXSMM_DNN_ELTWISE_FTYPE*)gru->r1t->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *r2t = (LIBXSMM_DNN_ELTWISE_FTYPE*)gru->r2t->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *z1t = (LIBXSMM_DNN_ELTWISE_FTYPE*)gru->z1t->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *z2t = (LIBXSMM_DNN_ELTWISE_FTYPE*)gru->z2t->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *g1t = (LIBXSMM_DNN_ELTWISE_FTYPE*)gru->g1t->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *g2t = (LIBXSMM_DNN_ELTWISE_FTYPE*)gru->g2t->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *g3  = (LIBXSMM_DNN_ELTWISE_FTYPE*)gru->g3->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *h1  = (LIBXSMM_DNN_ELTWISE_FTYPE*)gru->h1->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *h2  = (LIBXSMM_DNN_ELTWISE_FTYPE*)gru->h2->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *h3  = (LIBXSMM_DNN_ELTWISE_FTYPE*)gru->h3->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *r   = (LIBXSMM_DNN_ELTWISE_FTYPE*)gru->r->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *z   = (LIBXSMM_DNN_ELTWISE_FTYPE*)gru->z->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *g   = (LIBXSMM_DNN_ELTWISE_FTYPE*)gru->g->data;
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, x, xt, k * n);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, r1, r1t, m * n);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, z1, z1t, m * n);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, g1, g1t, m * n);
  /*libxsmm_bgemm_handle *handleux = gru->handleux;*/
  libxsmm_bgemm_handle *handlewh = gru->handlewh;
  libxsmm_bgemm_handle *handlett = gru->handlett;
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, hnr, h, m * n);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, rnr, r, m * n);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, znr, z, m * n);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, gnr, g, m * n);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, r2, r2t, m * n);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, z2, z2t, m * n);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, g2, g2t, m * n);
#if defined(LSTM_TIMING)
  unsigned long long start;
  double duration;
  Gbl_t_input_total = 0.; Gbl_t_recur_total = 0.; Gbl_t_eltwise_total = 0.; Gbl_t_nonlin_total = 0.;
  Gbl_t_input = 0; Gbl_t_recur = 0; Gbl_t_eltwise = 0; Gbl_t_nonlin = 0;
  Gbl_duration_input = 0.; Gbl_duration_recur = 0.; Gbl_duration_eltwise = 0.; Gbl_duration_nonlin = 0.;
#endif
  int j;
  const int ltid = tid - start_thread;

  libxsmm_barrier_init(gru->barrier, ltid);
#if defined(LSTM_TIMING)
  if (ltid == 0) { start = libxsmm_timer_tick(); }
#endif

  if (reuse) {
#if defined(LSTM_TIMING)
    if (ltid == 0) { Gbl_t_input = libxsmm_timer_tick(); }
#endif
    libxsmm_bgemm_st(handlett, ur, &LIBXSMM_VLA_ACCESS(2, x, 0, 0, k * n), &LIBXSMM_VLA_ACCESS(2, r1, 0, 0, m * n), start_thread, tid);
    libxsmm_bgemm_st(handlett, uz, &LIBXSMM_VLA_ACCESS(2, x, 0, 0, k * n), &LIBXSMM_VLA_ACCESS(2, z1, 0, 0, m * n), start_thread, tid);
    libxsmm_bgemm_st(handlett, ug, &LIBXSMM_VLA_ACCESS(2, x, 0, 0, k * n), &LIBXSMM_VLA_ACCESS(2, g1, 0, 0, m * n), start_thread, tid);
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
      libxsmm_internal_matrix_add(m * n, &LIBXSMM_VLA_ACCESS(2, r1, j, 0, m * n), br, &LIBXSMM_VLA_ACCESS(2, r1, j, 0, m * n), start_thread, tid, gru->nThreads);
      libxsmm_internal_matrix_add(m * n, &LIBXSMM_VLA_ACCESS(2, z1, j, 0, m * n), bz, &LIBXSMM_VLA_ACCESS(2, z1, j, 0, m * n), start_thread, tid, gru->nThreads);
      libxsmm_internal_matrix_add(m * n, &LIBXSMM_VLA_ACCESS(2, g1, j, 0, m * n), bg, &LIBXSMM_VLA_ACCESS(2, g1, j, 0, m * n), start_thread, tid, gru->nThreads);
      libxsmm_barrier_wait(gru->barrier, ltid);
#if defined(LSTM_TIMING)
      if (ltid == 0) {
        Gbl_duration_eltwise = libxsmm_timer_duration(Gbl_t_eltwise, libxsmm_timer_tick());
        Gbl_t_eltwise_total += Gbl_duration_eltwise;
      }
#endif
      libxsmm_internal_recursive_step(handlewh, wr, h, &LIBXSMM_VLA_ACCESS(2, r2, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, r1, j, 0, m * n), r, r, 2, m * n, start_thread, tid); /*sigmoid*/
      libxsmm_internal_recursive_step(handlewh, wz, h, &LIBXSMM_VLA_ACCESS(2, z2, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, z1, j, 0, m * n), z, z, 2, m * n, start_thread, tid); /*sigmoid*/
      libxsmm_barrier_wait(gru->barrier, ltid);
#if defined(LSTM_TIMING)
      if (ltid == 0) { Gbl_t_eltwise = libxsmm_timer_tick(); }
#endif
      libxsmm_internal_matrix_eltwise_mult(m*n, h, r, g3, start_thread, tid, gru->nThreads);
      libxsmm_barrier_wait(gru->barrier, ltid);
#if defined(LSTM_TIMING)
      if (ltid == 0) {
        Gbl_duration_eltwise = libxsmm_timer_duration(Gbl_t_eltwise, libxsmm_timer_tick());
        Gbl_t_eltwise_total += Gbl_duration_eltwise;
      }
#endif
      libxsmm_internal_recursive_step(handlewh, wg, g3, &LIBXSMM_VLA_ACCESS(2, g2, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, g1, j, 0, m * n), g, g, 3, m * n, start_thread, tid); /*tanh*/
      libxsmm_barrier_wait(gru->barrier, ltid);
#if defined(LSTM_TIMING)
      if (ltid == 0) { Gbl_t_eltwise = libxsmm_timer_tick(); }
#endif
      libxsmm_internal_matrix_eltwise_mult(m*n, z, g, h3, start_thread, tid, gru->nThreads);
      libxsmm_internal_matrix_complement(m*n, z, h2, start_thread, tid, gru->nThreads);
      libxsmm_barrier_wait(gru->barrier, ltid);
      libxsmm_internal_matrix_eltwise_mult(m*n, h, h2, h1, start_thread, tid, gru->nThreads);
      libxsmm_barrier_wait(gru->barrier, ltid);
      libxsmm_internal_matrix_add(m*n, h1, h3, h, start_thread, tid, gru->nThreads);
#if defined(LSTM_TIMING)
      libxsmm_barrier_wait(gru->barrier, ltid); /* Additional barrier introduced to measure time */
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
    libxsmm_bgemm_st(handlett, ur, &LIBXSMM_VLA_ACCESS(2, x, 0, 0, k * n), &LIBXSMM_VLA_ACCESS(2, r1, 0, 0, m * n), start_thread, tid);
    libxsmm_bgemm_st(handlett, uz, &LIBXSMM_VLA_ACCESS(2, x, 0, 0, k * n), &LIBXSMM_VLA_ACCESS(2, z1, 0, 0, m * n), start_thread, tid);
    libxsmm_bgemm_st(handlett, ug, &LIBXSMM_VLA_ACCESS(2, x, 0, 0, k * n), &LIBXSMM_VLA_ACCESS(2, g1, 0, 0, m * n), start_thread, tid);
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
      libxsmm_internal_matrix_add(m * n, &LIBXSMM_VLA_ACCESS(2, r1, j, 0, m * n), br, &LIBXSMM_VLA_ACCESS(2, r1, j, 0, m * n), start_thread, tid, gru->nThreads);
      libxsmm_internal_matrix_add(m * n, &LIBXSMM_VLA_ACCESS(2, z1, j, 0, m * n), bz, &LIBXSMM_VLA_ACCESS(2, z1, j, 0, m * n), start_thread, tid, gru->nThreads);
      libxsmm_internal_matrix_add(m * n, &LIBXSMM_VLA_ACCESS(2, g1, j, 0, m * n), bg, &LIBXSMM_VLA_ACCESS(2, g1, j, 0, m * n), start_thread, tid, gru->nThreads);
      libxsmm_barrier_wait(gru->barrier, ltid);
#if defined(LSTM_TIMING)
      if (ltid == 0) {
        Gbl_duration_eltwise = libxsmm_timer_duration(Gbl_t_eltwise, libxsmm_timer_tick());
        Gbl_t_eltwise_total += Gbl_duration_eltwise;
      }
#endif
      libxsmm_internal_recursive_step(handlewh, wr, &LIBXSMM_VLA_ACCESS(2, hnr, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, r2, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, r1, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, rnr, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, rnr, j, 0, m * n), 2, m * n, start_thread, tid); /*sigmoid*/
      libxsmm_internal_recursive_step(handlewh, wz, &LIBXSMM_VLA_ACCESS(2, hnr, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, z2, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, z1, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, znr, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, znr, j, 0, m * n), 2, m * n, start_thread, tid); /*sigmoid*/
      libxsmm_barrier_wait(gru->barrier, ltid);
#if defined(LSTM_TIMING)
      if (ltid == 0) { Gbl_t_eltwise = libxsmm_timer_tick(); }
#endif
      libxsmm_internal_matrix_eltwise_mult(m*n, &LIBXSMM_VLA_ACCESS(2, hnr, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, rnr, j, 0, m * n), g3, start_thread, tid, gru->nThreads);
      libxsmm_barrier_wait(gru->barrier, ltid);
#if defined(LSTM_TIMING)
      if (ltid == 0) {
        Gbl_duration_eltwise = libxsmm_timer_duration(Gbl_t_eltwise, libxsmm_timer_tick());
        Gbl_t_eltwise_total += Gbl_duration_eltwise;
      }
#endif
      libxsmm_internal_recursive_step(handlewh, wg, g3, &LIBXSMM_VLA_ACCESS(2, g2, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, g1, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, gnr, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, gnr, j, 0, m * n), 3, m * n, start_thread, tid); /*tanh*/
      libxsmm_barrier_wait(gru->barrier, ltid);
#if defined(LSTM_TIMING)
      if (ltid == 0) { Gbl_t_eltwise = libxsmm_timer_tick(); }
#endif
      libxsmm_internal_matrix_eltwise_mult(m*n, &LIBXSMM_VLA_ACCESS(2, znr, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, gnr, j, 0, m * n), h3, start_thread, tid, gru->nThreads);
      libxsmm_internal_matrix_complement(m*n, &LIBXSMM_VLA_ACCESS(2, znr, j, 0, m * n), h2, start_thread, tid, gru->nThreads);
      libxsmm_barrier_wait(gru->barrier, ltid);
      libxsmm_internal_matrix_eltwise_mult(m*n, &LIBXSMM_VLA_ACCESS(2, hnr, j, 0, m *n), h2, h1, start_thread, tid, gru->nThreads);
      libxsmm_barrier_wait(gru->barrier, ltid);
      libxsmm_internal_matrix_add(m*n, h1, h3, &LIBXSMM_VLA_ACCESS(2, hnr, j+1, 0, m * n), start_thread, tid, gru->nThreads);
#if defined(LSTM_TIMING)
      libxsmm_barrier_wait(gru->barrier, ltid); /* Additional barrier introduced to measure time */
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


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_grucell_bwd_upd_bu(libxsmm_dnn_grucell* gru, int start_thread, int tid, int pass)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;
  libxsmm_blasint m = gru->m;
  libxsmm_blasint n = gru->n;
  libxsmm_blasint k = gru->k;
  libxsmm_blasint t = gru->t;
  LIBXSMM_DNN_ELTWISE_FTYPE *wr = (LIBXSMM_DNN_ELTWISE_FTYPE*)gru->wr->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *wz = (LIBXSMM_DNN_ELTWISE_FTYPE*)gru->wz->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *wg = (LIBXSMM_DNN_ELTWISE_FTYPE*)gru->wg->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *xt = (LIBXSMM_DNN_ELTWISE_FTYPE*)gru->xt->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *ur = (LIBXSMM_DNN_ELTWISE_FTYPE*)gru->ur->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *uz = (LIBXSMM_DNN_ELTWISE_FTYPE*)gru->uz->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *ug = (LIBXSMM_DNN_ELTWISE_FTYPE*)gru->ug->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *ht = (LIBXSMM_DNN_ELTWISE_FTYPE*)gru->h->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *rt = (LIBXSMM_DNN_ELTWISE_FTYPE*)gru->r->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *zt = (LIBXSMM_DNN_ELTWISE_FTYPE*)gru->z->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *gt = (LIBXSMM_DNN_ELTWISE_FTYPE*)gru->g->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *d4 = (LIBXSMM_DNN_ELTWISE_FTYPE*)gru->d4->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *d5 = (LIBXSMM_DNN_ELTWISE_FTYPE*)gru->d5->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *d6 = (LIBXSMM_DNN_ELTWISE_FTYPE*)gru->d6->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *d7 = (LIBXSMM_DNN_ELTWISE_FTYPE*)gru->d7->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *d8 = (LIBXSMM_DNN_ELTWISE_FTYPE*)gru->d8->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *d9 = (LIBXSMM_DNN_ELTWISE_FTYPE*)gru->d9->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *d10 = (LIBXSMM_DNN_ELTWISE_FTYPE*)gru->d10->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *d11 = (LIBXSMM_DNN_ELTWISE_FTYPE*)gru->d11->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *d12 = (LIBXSMM_DNN_ELTWISE_FTYPE*)gru->d12->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *d13 = (LIBXSMM_DNN_ELTWISE_FTYPE*)gru->d13->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *d14 = (LIBXSMM_DNN_ELTWISE_FTYPE*)gru->d14->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *d15 = (LIBXSMM_DNN_ELTWISE_FTYPE*)gru->d15->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *d16 = (LIBXSMM_DNN_ELTWISE_FTYPE*)gru->d16->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *d17 = (LIBXSMM_DNN_ELTWISE_FTYPE*)gru->d17->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *d18 = (LIBXSMM_DNN_ELTWISE_FTYPE*)gru->d18->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *d19 = (LIBXSMM_DNN_ELTWISE_FTYPE*)gru->d19->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *d20 = (LIBXSMM_DNN_ELTWISE_FTYPE*)gru->d20->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *d21 = (LIBXSMM_DNN_ELTWISE_FTYPE*)gru->d21->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *d22 = (LIBXSMM_DNN_ELTWISE_FTYPE*)gru->d22->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *d23 = (LIBXSMM_DNN_ELTWISE_FTYPE*)gru->d23->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *djdht = (LIBXSMM_DNN_ELTWISE_FTYPE*)gru->djdht->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *djdxt = (LIBXSMM_DNN_ELTWISE_FTYPE*)gru->djdxt->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *djdwr = (LIBXSMM_DNN_ELTWISE_FTYPE*)gru->djdwr->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *djdwz = (LIBXSMM_DNN_ELTWISE_FTYPE*)gru->djdwz->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *djdwg = (LIBXSMM_DNN_ELTWISE_FTYPE*)gru->djdwg->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *djdur = (LIBXSMM_DNN_ELTWISE_FTYPE*)gru->djdur->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *djduz = (LIBXSMM_DNN_ELTWISE_FTYPE*)gru->djduz->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *djdug = (LIBXSMM_DNN_ELTWISE_FTYPE*)gru->djdug->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *djdbr = (LIBXSMM_DNN_ELTWISE_FTYPE*)gru->djdbr->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *djdbz = (LIBXSMM_DNN_ELTWISE_FTYPE*)gru->djdbz->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *djdbg = (LIBXSMM_DNN_ELTWISE_FTYPE*)gru->djdbg->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *d10M = (LIBXSMM_DNN_ELTWISE_FTYPE*)gru->d10M->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *d11M = (LIBXSMM_DNN_ELTWISE_FTYPE*)gru->d11M->data;
  LIBXSMM_DNN_ELTWISE_FTYPE *d18M = (LIBXSMM_DNN_ELTWISE_FTYPE*)gru->d18M->data;
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, x, xt, k * n);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, h, ht, m * n);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, r, rt, m * n);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, z, zt, m * n);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, g, gt, m * n);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, djdh, djdht, m * n);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, djdx, djdxt, k * n);
  libxsmm_bgemm_handle *handleud = gru->handleux;
  libxsmm_bgemm_handle *handledh = gru->handlewh;
  libxsmm_bgemm_handle *handledx = gru->handlett;
  libxsmm_bgemm_handle *handlewd = gru->handlewd;
  int j;
  const int ltid = tid - start_thread;

  libxsmm_barrier_init(gru->barrier, ltid);
#if 0
  /* compute delta */
  libxsmm_internal_matrix_copy(m * n, &LIBXSMM_VLA_ACCESS(2, djdh, t-1, 0, m * n), &LIBXSMM_VLA_ACCESS(2, delta, t-1, 0, m * n), start_thread, tid, gru->nThreads);
  /* compute djdd */
  libxsmm_internal_matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, djdh, t-1, 0, m * n), &LIBXSMM_VLA_ACCESS(2, o, t-1, 0, m * n), d1, start_thread, tid, gru->nThreads);
  libxsmm_internal_matrix_tanh_inverse(m * n, &LIBXSMM_VLA_ACCESS(2, d, t-1, 0, m * n), d2, start_thread, tid, gru->nThreads);
  libxsmm_barrier_wait(gru->barrier, ltid);
  libxsmm_internal_matrix_eltwise_mult(m * n, d1, d2, &LIBXSMM_VLA_ACCESS(2, djdd, t-1, 0, m * n), start_thread, tid, gru->nThreads);
  /* compute djdc */
  libxsmm_internal_matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, djdd, t-1, 0, m * n), &LIBXSMM_VLA_ACCESS(2, i, t-1, 0, m * n), c1, start_thread, tid, gru->nThreads);
  libxsmm_internal_matrix_complement_square(m * n, &LIBXSMM_VLA_ACCESS(2, c, t-1, 0, m * n), c2, start_thread, tid, gru->nThreads);
  libxsmm_barrier_wait(gru->barrier, ltid);
  libxsmm_internal_matrix_eltwise_mult(m * n, c1, c2, &LIBXSMM_VLA_ACCESS(2, djdc, t-1, 0, m * n), start_thread, tid, gru->nThreads);
  /* compute djdi */
  libxsmm_internal_matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, djdd, t-1, 0, m * n), &LIBXSMM_VLA_ACCESS(2, c, t-1, 0, m * n), i1, start_thread, tid, gru->nThreads);
  libxsmm_internal_matrix_complement(m * n, &LIBXSMM_VLA_ACCESS(2, i, t-1, 0, m * n), i2, start_thread, tid, gru->nThreads);
  libxsmm_barrier_wait(gru->barrier, ltid);
  libxsmm_internal_matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, i, t-1, 0, m * n), i2, i3, start_thread, tid, gru->nThreads);
  libxsmm_barrier_wait(gru->barrier, ltid);
  libxsmm_internal_matrix_eltwise_mult(m * n, i1, i3, &LIBXSMM_VLA_ACCESS(2, djdi, t-1, 0, m * n), start_thread, tid, gru->nThreads);
  /* compute djdf */
  libxsmm_internal_matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, djdd, t-1, 0, m * n), &LIBXSMM_VLA_ACCESS(2, d, t-2, 0, m * n), f1, start_thread, tid, gru->nThreads);
  libxsmm_internal_matrix_complement(m * n, &LIBXSMM_VLA_ACCESS(2, f, t-1, 0, m * n), f2, start_thread, tid, gru->nThreads);
  libxsmm_barrier_wait(gru->barrier, ltid);
  libxsmm_internal_matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, f, t-1, 0, m * n), f2, f3, start_thread, tid, gru->nThreads);
  libxsmm_barrier_wait(gru->barrier, ltid);
  libxsmm_internal_matrix_eltwise_mult(m * n, f1, f3, &LIBXSMM_VLA_ACCESS(2, djdf, t-1, 0, m * n), start_thread, tid, gru->nThreads);
  /* compute djdo */
  libxsmm_internal_matrix_tanh(m * n, &LIBXSMM_VLA_ACCESS(2, d, t-1, 0, m * n), o1, start_thread, tid, gru->nThreads);
  libxsmm_internal_matrix_complement(m * n, &LIBXSMM_VLA_ACCESS(2, o, t-1, 0, m * n), o2, start_thread, tid, gru->nThreads);
  libxsmm_barrier_wait(gru->barrier, ltid);
  libxsmm_internal_matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, delta, t-1, 0, m * n), o1, o1, start_thread, tid, gru->nThreads);
  libxsmm_internal_matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, o, t-1, 0, m * n), o2, o2, start_thread, tid, gru->nThreads);
  libxsmm_barrier_wait(gru->barrier, ltid);
  libxsmm_internal_matrix_eltwise_mult(m * n, o1, o2, &LIBXSMM_VLA_ACCESS(2, djdo, t-1, 0, m * n), start_thread, tid, gru->nThreads);
  libxsmm_barrier_wait(gru->barrier, ltid);
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
    libxsmm_internal_matrix_add(m * n, &LIBXSMM_VLA_ACCESS(2, djdh, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, delta, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, delta, j, 0, m * n), start_thread, tid, gru->nThreads);
    /* compute djdd */
    libxsmm_internal_matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, djdh, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, o, j, 0, m * n), d1, start_thread, tid, gru->nThreads);
    libxsmm_internal_matrix_tanh_inverse(m * n, &LIBXSMM_VLA_ACCESS(2, d, j, 0, m * n), d2, start_thread, tid, gru->nThreads);
    libxsmm_barrier_wait(gru->barrier, ltid);
    libxsmm_internal_matrix_eltwise_mult(m * n, d1, d2, d3, start_thread, tid, gru->nThreads);
    libxsmm_internal_matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, delta, j+1, 0, m * n), &LIBXSMM_VLA_ACCESS(2, f, j+1, 0, m * n), d4, start_thread, tid, gru->nThreads);
    libxsmm_barrier_wait(gru->barrier, ltid);
    libxsmm_internal_matrix_add(m * n, d3, d4, &LIBXSMM_VLA_ACCESS(2, djdd, j, 0, m * n), start_thread, tid, gru->nThreads);
    /* compute djdc */
    libxsmm_internal_matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, djdd, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, i, j, 0, m * n), c1, start_thread, tid, gru->nThreads);
    libxsmm_internal_matrix_complement_square(m * n, &LIBXSMM_VLA_ACCESS(2, c, j, 0, m * n), c2, start_thread, tid, gru->nThreads);
    libxsmm_barrier_wait(gru->barrier, ltid);
    libxsmm_internal_matrix_eltwise_mult(m * n, c1, c2, &LIBXSMM_VLA_ACCESS(2, djdc, j, 0, m * n), start_thread, tid, gru->nThreads);
    /* compute djdi */
    libxsmm_internal_matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, djdd, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, c, j, 0, m * n), i1, start_thread, tid, gru->nThreads);
    libxsmm_internal_matrix_complement(m * n, &LIBXSMM_VLA_ACCESS(2, i, j, 0, m * n), i2, start_thread, tid, gru->nThreads);
    libxsmm_barrier_wait(gru->barrier, ltid);
    libxsmm_internal_matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, i, j, 0, m * n), i2, i3, start_thread, tid, gru->nThreads);
    libxsmm_barrier_wait(gru->barrier, ltid);
    libxsmm_internal_matrix_eltwise_mult(m * n, i1, i3, &LIBXSMM_VLA_ACCESS(2, djdi, j, 0, m * n), start_thread, tid, gru->nThreads);
    /* compute djdf */
    if (j >= 1) {
      libxsmm_internal_matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, djdd, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, d, j-1, 0, m * n), f1, start_thread, tid, gru->nThreads);
      libxsmm_internal_matrix_complement(m * n, &LIBXSMM_VLA_ACCESS(2, f, j, 0, m * n), f2, start_thread, tid, gru->nThreads);
      libxsmm_barrier_wait(gru->barrier, ltid);
      libxsmm_internal_matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, f, j, 0, m * n), f2, f3, start_thread, tid, gru->nThreads);
      libxsmm_barrier_wait(gru->barrier, ltid);
      libxsmm_internal_matrix_eltwise_mult(m * n, f1, f3, &LIBXSMM_VLA_ACCESS(2, djdf, j, 0, m * n), start_thread, tid, gru->nThreads);
    } else {
      /* djdf is zero for j == 0 */
      libxsmm_internal_matrix_zero(m * n, &LIBXSMM_VLA_ACCESS(2, djdf, j, 0, m * n), start_thread, tid, gru->nThreads);
    }
    /* compute djdo */
    libxsmm_internal_matrix_tanh(m * n, &LIBXSMM_VLA_ACCESS(2, d, j, 0, m * n), o1, start_thread, tid, gru->nThreads);
    libxsmm_internal_matrix_complement(m * n, &LIBXSMM_VLA_ACCESS(2, o, j, 0, m * n), o2, start_thread, tid, gru->nThreads);
    libxsmm_barrier_wait(gru->barrier, ltid);
    libxsmm_internal_matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, delta, j, 0, m * n), o1, o1, start_thread, tid, gru->nThreads);
    libxsmm_internal_matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, o, j, 0, m * n), o2, o2, start_thread, tid, gru->nThreads);
    libxsmm_barrier_wait(gru->barrier, ltid);
    libxsmm_internal_matrix_eltwise_mult(m * n, o1, o2, &LIBXSMM_VLA_ACCESS(2, djdo, j, 0, m * n), start_thread, tid, gru->nThreads);
    libxsmm_barrier_wait(gru->barrier, ltid);
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
      libxsmm_barrier_wait(gru->barrier, ltid);
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
      libxsmm_internal_matrix_add(m * n, &LIBXSMM_VLA_ACCESS(2, djdi, j, 0, m * n), djdbi, djdbi, start_thread, tid, gru->nThreads);
      libxsmm_internal_matrix_add(m * n, &LIBXSMM_VLA_ACCESS(2, djdf, j, 0, m * n), djdbf, djdbf, start_thread, tid, gru->nThreads);
      libxsmm_internal_matrix_add(m * n, &LIBXSMM_VLA_ACCESS(2, djdo, j, 0, m * n), djdbo, djdbo, start_thread, tid, gru->nThreads);
      libxsmm_internal_matrix_add(m * n, &LIBXSMM_VLA_ACCESS(2, djdc, j, 0, m * n), djdbc, djdbc, start_thread, tid, gru->nThreads);
    }
    libxsmm_barrier_wait(gru->barrier, ltid);
  }
#endif

  return status;
}


LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_grucell_execute_st(libxsmm_dnn_grucell* handle, libxsmm_dnn_compute_kind kind,
  /*unsigned*/int start_thread, /*unsigned*/int tid)
{
  libxsmm_dnn_err_t status = LIBXSMM_DNN_SUCCESS;

  if (0 != handle) {
    switch (kind) {
      case LIBXSMM_DNN_COMPUTE_KIND_FWD: {
                                           status = libxsmm_dnn_grucell_fwd(handle, start_thread, tid);
                                         } break;
      case LIBXSMM_DNN_COMPUTE_KIND_BWD: {
                                           status = libxsmm_dnn_grucell_bwd_upd_bu(handle, start_thread, tid, 1);
                                         } break;
      case LIBXSMM_DNN_COMPUTE_KIND_UPD: {
                                           status = libxsmm_dnn_grucell_bwd_upd_bu(handle, start_thread, tid, 2);
                                         } break;
      case LIBXSMM_DNN_COMPUTE_KIND_ALL: {
                                           status = libxsmm_dnn_grucell_bwd_upd_bu(handle, start_thread, tid, 3);
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

