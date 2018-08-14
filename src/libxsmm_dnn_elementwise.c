/******************************************************************************
** Copyright (c) 2018, Intel Corporation                                     **
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
/* Kunal Banerjee (Intel Corp.)
******************************************************************************/
#include "libxsmm_dnn_elementwise.h"
#include "libxsmm_bgemm_types.h"

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#include <math.h>
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif


LIBXSMM_API_INTERN void libxsmm_internal_matrix_zero(libxsmm_blasint size, LIBXSMM_DNN_ELTWISE_FTYPE *src, int start_thread, int tid, int nthreads)
{
  int ltid;
  int chunksize;
  int thr_begin;
  int thr_end;
  libxsmm_blasint i;
  ltid = tid - start_thread;
  /* compute chunk size */
  chunksize = (size % nthreads == 0) ? (size / nthreads) : (size / nthreads) + 1;
  /* compute thr_begin and thr_end */
  thr_begin = (ltid * chunksize < size) ? (ltid * chunksize) : size;
  thr_end = ((ltid + 1) * chunksize < size) ? ((ltid + 1) * chunksize) : size;

  for (i = thr_begin; i < thr_end; i++) {
    src[i] = (LIBXSMM_DNN_ELTWISE_FTYPE)0;
  }
}


LIBXSMM_API_INTERN void libxsmm_internal_matrix_add(libxsmm_blasint size, LIBXSMM_DNN_ELTWISE_FTYPE *a, LIBXSMM_DNN_ELTWISE_FTYPE *b, LIBXSMM_DNN_ELTWISE_FTYPE *c, int start_thread, int tid, int nthreads)
{
  int ltid;
  int chunksize;
  int thr_begin;
  int thr_end;
  libxsmm_blasint i;
  ltid = tid - start_thread;
  /* compute chunk size */
  chunksize = (size % nthreads == 0) ? (size / nthreads) : (size / nthreads) + 1;
  /* compute thr_begin and thr_end */
  thr_begin = (ltid * chunksize < size) ? (ltid * chunksize) : size;
  thr_end = ((ltid + 1) * chunksize < size) ? ((ltid + 1) * chunksize) : size;

  for (i = thr_begin; i < thr_end; i++) {
    c[i] = a[i] + b[i];
  }
}


LIBXSMM_API_INTERN void libxsmm_internal_matrix_eltwise_mult(libxsmm_blasint size, LIBXSMM_DNN_ELTWISE_FTYPE *a, LIBXSMM_DNN_ELTWISE_FTYPE *b, LIBXSMM_DNN_ELTWISE_FTYPE *c, int start_thread, int tid, int nthreads)
{
  int ltid;
  int chunksize;
  int thr_begin;
  int thr_end;
  libxsmm_blasint i;
  ltid = tid - start_thread;
  /* compute chunk size */
  chunksize = (size % nthreads == 0) ? (size / nthreads) : (size / nthreads) + 1;
  /* compute thr_begin and thr_end */
  thr_begin = (ltid * chunksize < size) ? (ltid * chunksize) : size;
  thr_end = ((ltid + 1) * chunksize < size) ? ((ltid + 1) * chunksize) : size;

  for (i = thr_begin; i < thr_end; i++) {
    c[i] = a[i] * b[i];
  }
}


LIBXSMM_API_INTERN void libxsmm_internal_matrix_sigmoid(libxsmm_blasint size, LIBXSMM_DNN_ELTWISE_FTYPE *src, LIBXSMM_DNN_ELTWISE_FTYPE *dst, int start_thread, int tid, int nthreads)
{
  int ltid;
  int chunksize;
  int thr_begin;
  int thr_end;
  libxsmm_blasint i;
  LIBXSMM_DNN_ELTWISE_FTYPE exp_value;
  ltid = tid - start_thread;
  /* compute chunk size */
  chunksize = (size % nthreads == 0) ? (size / nthreads) : (size / nthreads) + 1;
  /* compute thr_begin and thr_end */
  thr_begin = (ltid * chunksize < size) ? (ltid * chunksize) : size;
  thr_end = ((ltid + 1) * chunksize < size) ? ((ltid + 1) * chunksize) : size;

  for (i = thr_begin; i < thr_end; i++) {
    exp_value = (LIBXSMM_DNN_ELTWISE_FTYPE)exp((double) -src[i]);
    dst[i] = 1 / (1 + exp_value);
  }
}


LIBXSMM_API_INTERN void libxsmm_internal_matrix_tanh(libxsmm_blasint size, LIBXSMM_DNN_ELTWISE_FTYPE *src, LIBXSMM_DNN_ELTWISE_FTYPE *dst, int start_thread, int tid, int nthreads)
{
  int ltid;
  int chunksize;
  int thr_begin;
  int thr_end;
  libxsmm_blasint i;
  ltid = tid - start_thread;
  /* compute chunk size */
  chunksize = (size % nthreads == 0) ? (size / nthreads) : (size / nthreads) + 1;
  /* compute thr_begin and thr_end */
  thr_begin = (ltid * chunksize < size) ? (ltid * chunksize) : size;
  thr_end = ((ltid + 1) * chunksize < size) ? ((ltid + 1) * chunksize) : size;

  for (i = thr_begin; i < thr_end; i++) {
    dst[i] = (LIBXSMM_DNN_ELTWISE_FTYPE)tanh((double)src[i]);
  }
}


LIBXSMM_API_INTERN void libxsmm_internal_matrix_relu(libxsmm_blasint size, LIBXSMM_DNN_ELTWISE_FTYPE *src, LIBXSMM_DNN_ELTWISE_FTYPE *dst, int start_thread, int tid, int nthreads)
{
  int ltid;
  int chunksize;
  int thr_begin;
  int thr_end;
  libxsmm_blasint i;
  ltid = tid - start_thread;
  /* compute chunk size */
  chunksize = (size % nthreads == 0) ? (size / nthreads) : (size / nthreads) + 1;
  /* compute thr_begin and thr_end */
  thr_begin = (ltid * chunksize < size) ? (ltid * chunksize) : size;
  thr_end = ((ltid + 1) * chunksize < size) ? ((ltid + 1) * chunksize) : size;

  for (i = thr_begin; i < thr_end; i++) {
    dst[i] = (src[i] >= 0) ? src[i] : 0;
  }
}


LIBXSMM_API_INTERN void libxsmm_internal_matrix_sigmoid_inverse(libxsmm_blasint size, LIBXSMM_DNN_ELTWISE_FTYPE *src, LIBXSMM_DNN_ELTWISE_FTYPE *dst, int start_thread, int tid, int nthreads)
{
  int ltid;
  int chunksize;
  int thr_begin;
  int thr_end;
  libxsmm_blasint i;
  LIBXSMM_DNN_ELTWISE_FTYPE exp_value;
  LIBXSMM_DNN_ELTWISE_FTYPE sig_exp;
  ltid = tid - start_thread;
  /* compute chunk size */
  chunksize = (size % nthreads == 0) ? (size / nthreads) : (size / nthreads) + 1;
  /* compute thr_begin and thr_end */
  thr_begin = (ltid * chunksize < size) ? (ltid * chunksize) : size;
  thr_end = ((ltid + 1) * chunksize < size) ? ((ltid + 1) * chunksize) : size;

  for (i = thr_begin; i < thr_end; i++) {
    exp_value = (LIBXSMM_DNN_ELTWISE_FTYPE)exp((double) -src[i]);
    sig_exp = 1 / (1 + exp_value);
    dst[i] = (1 - sig_exp)*sig_exp;
  }
}


LIBXSMM_API_INTERN void libxsmm_internal_matrix_tanh_inverse(libxsmm_blasint size, LIBXSMM_DNN_ELTWISE_FTYPE *src, LIBXSMM_DNN_ELTWISE_FTYPE *dst, int start_thread, int tid, int nthreads)
{
  int ltid;
  int chunksize;
  int thr_begin;
  int thr_end;
  libxsmm_blasint i;
  LIBXSMM_DNN_ELTWISE_FTYPE tanh_value;
  ltid = tid - start_thread;
  /* compute chunk size */
  chunksize = (size % nthreads == 0) ? (size / nthreads) : (size / nthreads) + 1;
  /* compute thr_begin and thr_end */
  thr_begin = (ltid * chunksize < size) ? (ltid * chunksize) : size;
  thr_end = ((ltid + 1) * chunksize < size) ? ((ltid + 1) * chunksize) : size;

  for (i = thr_begin; i < thr_end; i++) {
    tanh_value = (LIBXSMM_DNN_ELTWISE_FTYPE)tanh((double)src[i]);
    dst[i] = 1 - (tanh_value * tanh_value);
  }
}


LIBXSMM_API_INTERN void libxsmm_internal_matrix_relu_inverse(libxsmm_blasint size, LIBXSMM_DNN_ELTWISE_FTYPE *src, LIBXSMM_DNN_ELTWISE_FTYPE *dst, LIBXSMM_DNN_ELTWISE_FTYPE *input, int start_thread, int tid, int nthreads)
{
  int ltid;
  int chunksize;
  int thr_begin;
  int thr_end;
  libxsmm_blasint i;
  ltid = tid - start_thread;
  /* compute chunk size */
  chunksize = (size % nthreads == 0) ? (size / nthreads) : (size / nthreads) + 1;
  /* compute thr_begin and thr_end */
  thr_begin = (ltid * chunksize < size) ? (ltid * chunksize) : size;
  thr_end = ((ltid + 1) * chunksize < size) ? ((ltid + 1) * chunksize) : size;

  for (i = thr_begin; i < thr_end; i++) {
    dst[i] = (input[i] >= 0) ? src[i] : 0;
  }
}


LIBXSMM_API_INTERN void libxsmm_internal_matrix_transpose(libxsmm_blasint rows, libxsmm_blasint cols, LIBXSMM_DNN_ELTWISE_FTYPE *src, LIBXSMM_DNN_ELTWISE_FTYPE *dst, int start_thread, int tid, int nthreads)
{
  int ltid;
  int chunksize;
  int thr_begin;
  int thr_end;
  int size;
  libxsmm_blasint job, i, j;
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, src2D, src, cols);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, dst2D, dst, rows);
  ltid = tid - start_thread;
  /* number of tasks that could be run in parallel */
  size = rows*cols;
  /* compute chunk size */
  chunksize = (size % nthreads == 0) ? (size / nthreads) : (size / nthreads) + 1;
  /* compute thr_begin and thr_end */
  thr_begin = (ltid * chunksize < size) ? (ltid * chunksize) : size;
  thr_end = ((ltid + 1) * chunksize < size) ? ((ltid + 1) * chunksize) : size;

  for (job = thr_begin; job < thr_end; job++) {
    i = job / cols;
    j = job % cols;
    LIBXSMM_VLA_ACCESS(2, dst2D, j, i, rows) = LIBXSMM_VLA_ACCESS(2, src2D, i, j, cols);
  }
}


LIBXSMM_API_INTERN void libxsmm_internal_matrix_copy(libxsmm_blasint size, LIBXSMM_DNN_ELTWISE_FTYPE *src, LIBXSMM_DNN_ELTWISE_FTYPE *dst, int start_thread, int tid, int nthreads)
{
  int ltid;
  int chunksize;
  int thr_begin;
  int thr_end;
  libxsmm_blasint i;
  ltid = tid - start_thread;
  /* compute chunk size */
  chunksize = (size % nthreads == 0) ? (size / nthreads) : (size / nthreads) + 1;
  /* compute thr_begin and thr_end */
  thr_begin = (ltid * chunksize < size) ? (ltid * chunksize) : size;
  thr_end = ((ltid + 1) * chunksize < size) ? ((ltid + 1) * chunksize) : size;

  for (i = thr_begin; i < thr_end; i++) {
    dst[i] = src[i];
  }
}


LIBXSMM_API_INTERN void libxsmm_internal_matrix_complement(libxsmm_blasint size, LIBXSMM_DNN_ELTWISE_FTYPE *src, LIBXSMM_DNN_ELTWISE_FTYPE *dst, int start_thread, int tid, int nthreads)
{
  int ltid;
  int chunksize;
  int thr_begin;
  int thr_end;
  libxsmm_blasint i;
  ltid = tid - start_thread;
  /* compute chunk size */
  chunksize = (size % nthreads == 0) ? (size / nthreads) : (size / nthreads) + 1;
  /* compute thr_begin and thr_end */
  thr_begin = (ltid * chunksize < size) ? (ltid * chunksize) : size;
  thr_end = ((ltid + 1) * chunksize < size) ? ((ltid + 1) * chunksize) : size;

  for (i = thr_begin; i < thr_end; i++) {
    dst[i] = 1 - src[i];
  }
}


LIBXSMM_API_INTERN void libxsmm_internal_matrix_complement_square(libxsmm_blasint size, LIBXSMM_DNN_ELTWISE_FTYPE *src, LIBXSMM_DNN_ELTWISE_FTYPE *dst, int start_thread, int tid, int nthreads)
{
  int ltid;
  int chunksize;
  int thr_begin;
  int thr_end;
  libxsmm_blasint i;
  ltid = tid - start_thread;
  /* compute chunk size */
  chunksize = (size % nthreads == 0) ? (size / nthreads) : (size / nthreads) + 1;
  /* compute thr_begin and thr_end */
  thr_begin = (ltid * chunksize < size) ? (ltid * chunksize) : size;
  thr_end = ((ltid + 1) * chunksize < size) ? ((ltid + 1) * chunksize) : size;

  for (i = thr_begin; i < thr_end; i++) {
    dst[i] = 1 - (src[i] * src[i]);
  }
}

/* #define LSTM_TIMING */
#if defined(LSTM_TIMING)
extern double Gbl_t_input_total, Gbl_t_recur_total, Gbl_t_eltwise_total, Gbl_t_nonlin_total;
extern unsigned long long Gbl_t_input, Gbl_t_recur, Gbl_t_eltwise, Gbl_t_nonlin;
extern double Gbl_duration_input, Gbl_duration_recur, Gbl_duration_eltwise, Gbl_duration_nonlin;
#endif

LIBXSMM_API_INTERN void libxsmm_internal_recursive_step(libxsmm_bgemm_handle* handle, LIBXSMM_DNN_ELTWISE_FTYPE* u, LIBXSMM_DNN_ELTWISE_FTYPE* h, LIBXSMM_DNN_ELTWISE_FTYPE* op1, LIBXSMM_DNN_ELTWISE_FTYPE *op2,
  LIBXSMM_DNN_ELTWISE_FTYPE *temp, LIBXSMM_DNN_ELTWISE_FTYPE *dst, int act, libxsmm_blasint size, int start_thread, int tid)
{
  const int ltid = tid - start_thread;
#if defined(LSTM_TIMING)
  if (ltid == 0) { Gbl_t_recur = libxsmm_timer_tick(); }
#endif
  libxsmm_bgemm_st(handle, u, h, op1, start_thread, ltid);
#if defined(LSTM_TIMING)
  if (ltid == 0) {
    Gbl_duration_recur = libxsmm_timer_duration(Gbl_t_recur, libxsmm_timer_tick());
    Gbl_t_recur_total += Gbl_duration_recur;
    Gbl_t_eltwise = libxsmm_timer_tick();
  }
#endif
  libxsmm_internal_matrix_add(size, op1, op2, temp, start_thread, ltid, handle->nthreads);
#if defined(LSTM_TIMING)
  libxsmm_barrier_wait(handle->barrier, ltid); /* Additional barrier introduced to measure time */
  if (ltid == 0) {
    Gbl_duration_eltwise = libxsmm_timer_duration(Gbl_t_eltwise, libxsmm_timer_tick());
    Gbl_t_eltwise_total += Gbl_duration_eltwise;
    Gbl_t_nonlin = libxsmm_timer_tick();
  }
#endif
  switch (act) {
    case 0:
      /* do nothing */
      dst = temp;
      break;
    case 1:
      libxsmm_internal_matrix_relu(size, temp, dst, start_thread, ltid, handle->nthreads);
      break;
    case 2:
      libxsmm_internal_matrix_sigmoid(size, temp, dst, start_thread, ltid, handle->nthreads);
      break;
    case 3:
      libxsmm_internal_matrix_tanh(size, temp, dst, start_thread, ltid, handle->nthreads);
      break;
    default:
      /* fprintf(stdout, "Unsupported activation function: %d\n", act); */
      dst = temp;
  }
#if defined(LSTM_TIMING)
  libxsmm_barrier_wait(handle->barrier, ltid); /* Additional barrier introduced to measure time */
  if (ltid == 0) {
    Gbl_duration_nonlin = libxsmm_timer_duration(Gbl_t_nonlin, libxsmm_timer_tick());
    Gbl_t_nonlin_total += Gbl_duration_nonlin;
  }
#endif
}

