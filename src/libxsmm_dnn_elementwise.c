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
  const int ltid = tid - start_thread;
  /* compute chunk size */
  const libxsmm_blasint chunksize = (size % nthreads == 0) ? (size / nthreads) : (size / nthreads) + 1;
  /* compute thr_begin and thr_end */
  const libxsmm_blasint thr_begin = (ltid * chunksize < size) ? (ltid * chunksize) : size;
  const libxsmm_blasint thr_end = ((ltid + 1) * chunksize < size) ? ((ltid + 1) * chunksize) : size;
  libxsmm_blasint i;

  for (i = thr_begin; i < thr_end; i++) {
    src[i] = (LIBXSMM_DNN_ELTWISE_FTYPE)0;
  }
}


LIBXSMM_API_INTERN void libxsmm_internal_matrix_add(libxsmm_blasint size, LIBXSMM_DNN_ELTWISE_FTYPE *a, LIBXSMM_DNN_ELTWISE_FTYPE *b, LIBXSMM_DNN_ELTWISE_FTYPE *c, int start_thread, int tid, int nthreads)
{
  const int ltid = tid - start_thread;
  /* compute chunk size */
  const libxsmm_blasint chunksize = (size % nthreads == 0) ? (size / nthreads) : (size / nthreads) + 1;
  /* compute thr_begin and thr_end */
  const libxsmm_blasint thr_begin = (ltid * chunksize < size) ? (ltid * chunksize) : size;
  const libxsmm_blasint thr_end = ((ltid + 1) * chunksize < size) ? ((ltid + 1) * chunksize) : size;
  libxsmm_blasint i;

  for (i = thr_begin; i < thr_end; i++) {
    c[i] = a[i] + b[i];
  }
}


LIBXSMM_API_INTERN void libxsmm_internal_matrix_eltwise_mult(libxsmm_blasint size, LIBXSMM_DNN_ELTWISE_FTYPE *a, LIBXSMM_DNN_ELTWISE_FTYPE *b, LIBXSMM_DNN_ELTWISE_FTYPE *c, int start_thread, int tid, int nthreads)
{
  const int ltid = tid - start_thread;
  /* compute chunk size */
  const libxsmm_blasint chunksize = (size % nthreads == 0) ? (size / nthreads) : (size / nthreads) + 1;
  /* compute thr_begin and thr_end */
  const libxsmm_blasint thr_begin = (ltid * chunksize < size) ? (ltid * chunksize) : size;
  const libxsmm_blasint thr_end = ((ltid + 1) * chunksize < size) ? ((ltid + 1) * chunksize) : size;
  libxsmm_blasint i;

  for (i = thr_begin; i < thr_end; i++) {
    c[i] = a[i] * b[i];
  }
}


LIBXSMM_API_INTERN void libxsmm_internal_matrix_sigmoid(libxsmm_blasint size, LIBXSMM_DNN_ELTWISE_FTYPE *src, LIBXSMM_DNN_ELTWISE_FTYPE *dst, int start_thread, int tid, int nthreads)
{
  const int ltid = tid - start_thread;
  /* compute chunk size */
  const libxsmm_blasint chunksize = (size % nthreads == 0) ? (size / nthreads) : (size / nthreads) + 1;
  /* compute thr_begin and thr_end */
  const libxsmm_blasint thr_begin = (ltid * chunksize < size) ? (ltid * chunksize) : size;
  const libxsmm_blasint thr_end = ((ltid + 1) * chunksize < size) ? ((ltid + 1) * chunksize) : size;
  libxsmm_blasint i;

  for (i = thr_begin; i < thr_end; i++) {
    const LIBXSMM_DNN_ELTWISE_FTYPE exp_value = (LIBXSMM_DNN_ELTWISE_FTYPE)exp((double) -src[i]);
    dst[i] = 1 / (1 + exp_value);
  }
}


LIBXSMM_API_INTERN void libxsmm_internal_matrix_tanh(libxsmm_blasint size, LIBXSMM_DNN_ELTWISE_FTYPE *src, LIBXSMM_DNN_ELTWISE_FTYPE *dst, int start_thread, int tid, int nthreads)
{
  const int ltid = tid - start_thread;
  /* compute chunk size */
  const libxsmm_blasint chunksize = (size % nthreads == 0) ? (size / nthreads) : (size / nthreads) + 1;
  /* compute thr_begin and thr_end */
  const libxsmm_blasint thr_begin = (ltid * chunksize < size) ? (ltid * chunksize) : size;
  const libxsmm_blasint thr_end = ((ltid + 1) * chunksize < size) ? ((ltid + 1) * chunksize) : size;
  libxsmm_blasint i;

  for (i = thr_begin; i < thr_end; i++) {
    dst[i] = (LIBXSMM_DNN_ELTWISE_FTYPE)tanh((double)src[i]);
  }
}


LIBXSMM_API_INTERN void libxsmm_internal_matrix_relu(libxsmm_blasint size, LIBXSMM_DNN_ELTWISE_FTYPE *src, LIBXSMM_DNN_ELTWISE_FTYPE *dst, int start_thread, int tid, int nthreads)
{
  const int ltid = tid - start_thread;
  /* compute chunk size */
  const libxsmm_blasint chunksize = (size % nthreads == 0) ? (size / nthreads) : (size / nthreads) + 1;
  /* compute thr_begin and thr_end */
  const libxsmm_blasint thr_begin = (ltid * chunksize < size) ? (ltid * chunksize) : size;
  const libxsmm_blasint thr_end = ((ltid + 1) * chunksize < size) ? ((ltid + 1) * chunksize) : size;
  libxsmm_blasint i;

  for (i = thr_begin; i < thr_end; i++) {
    dst[i] = (src[i] > 0.0f) ? src[i] : 0.0f;
  }
}


LIBXSMM_API_INTERN void libxsmm_internal_matrix_sigmoid_inverse(libxsmm_blasint size, LIBXSMM_DNN_ELTWISE_FTYPE *src, LIBXSMM_DNN_ELTWISE_FTYPE *dst, int start_thread, int tid, int nthreads)
{
  const int ltid = tid - start_thread;
  /* compute chunk size */
  const libxsmm_blasint chunksize = (size % nthreads == 0) ? (size / nthreads) : (size / nthreads) + 1;
  /* compute thr_begin and thr_end */
  const libxsmm_blasint thr_begin = (ltid * chunksize < size) ? (ltid * chunksize) : size;
  const libxsmm_blasint thr_end = ((ltid + 1) * chunksize < size) ? ((ltid + 1) * chunksize) : size;
  libxsmm_blasint i;

  for (i = thr_begin; i < thr_end; i++) {
    const LIBXSMM_DNN_ELTWISE_FTYPE exp_value = (LIBXSMM_DNN_ELTWISE_FTYPE)exp((double) -src[i]);
    const LIBXSMM_DNN_ELTWISE_FTYPE sig_exp = 1 / (1 + exp_value);
    dst[i] = (1 - sig_exp)*sig_exp;
  }
}


LIBXSMM_API_INTERN void libxsmm_internal_matrix_tanh_inverse(libxsmm_blasint size, LIBXSMM_DNN_ELTWISE_FTYPE *src, LIBXSMM_DNN_ELTWISE_FTYPE *dst, int start_thread, int tid, int nthreads)
{
  const int ltid = tid - start_thread;
  /* compute chunk size */
  const libxsmm_blasint chunksize = (size % nthreads == 0) ? (size / nthreads) : (size / nthreads) + 1;
  /* compute thr_begin and thr_end */
  const libxsmm_blasint thr_begin = (ltid * chunksize < size) ? (ltid * chunksize) : size;
  const libxsmm_blasint thr_end = ((ltid + 1) * chunksize < size) ? ((ltid + 1) * chunksize) : size;
  libxsmm_blasint i;

  for (i = thr_begin; i < thr_end; i++) {
    const LIBXSMM_DNN_ELTWISE_FTYPE tanh_value = (LIBXSMM_DNN_ELTWISE_FTYPE)tanh((double)src[i]);
    dst[i] = 1 - (tanh_value * tanh_value);
  }
}


LIBXSMM_API_INTERN void libxsmm_internal_matrix_relu_inverse(libxsmm_blasint size, LIBXSMM_DNN_ELTWISE_FTYPE *src, LIBXSMM_DNN_ELTWISE_FTYPE *dst, int start_thread, int tid, int nthreads)
{
  const int ltid = tid - start_thread;
  /* compute chunk size */
  const libxsmm_blasint chunksize = (size % nthreads == 0) ? (size / nthreads) : (size / nthreads) + 1;
  /* compute thr_begin and thr_end */
  const libxsmm_blasint thr_begin = (ltid * chunksize < size) ? (ltid * chunksize) : size;
  const libxsmm_blasint thr_end = ((ltid + 1) * chunksize < size) ? ((ltid + 1) * chunksize) : size;
  libxsmm_blasint i;

  for (i = thr_begin; i < thr_end; i++) {
    dst[i] = (LIBXSMM_DNN_ELTWISE_FTYPE)(src[i] > 0.0f ? 1.0f : 0.0f);
  }
}


LIBXSMM_API_INTERN void libxsmm_internal_matrix_transpose(libxsmm_blasint rows, libxsmm_blasint cols, LIBXSMM_DNN_ELTWISE_FTYPE *src, LIBXSMM_DNN_ELTWISE_FTYPE *dst, int start_thread, int tid, int nthreads)
{
  const int ltid = tid - start_thread;
  /* number of tasks that could be run in parallel */
  const libxsmm_blasint size = rows * cols;
  /* compute chunk size */
  const libxsmm_blasint chunksize = (size % nthreads == 0) ? (size / nthreads) : (size / nthreads) + 1;
  /* compute thr_begin and thr_end */
  const libxsmm_blasint thr_begin = (ltid * chunksize < size) ? (ltid * chunksize) : size;
  const libxsmm_blasint thr_end = ((ltid + 1) * chunksize < size) ? ((ltid + 1) * chunksize) : size;
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, src2D, src, cols);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, dst2D, dst, rows);
  libxsmm_blasint job;

  for (job = thr_begin; job < thr_end; ++job) {
    const libxsmm_blasint i = job / cols;
    const libxsmm_blasint j = job % cols;
    LIBXSMM_VLA_ACCESS(2, dst2D, j, i, rows) = LIBXSMM_VLA_ACCESS(2, src2D, i, j, cols);
  }
}


LIBXSMM_API_INTERN void libxsmm_internal_matrix_copy(libxsmm_blasint size, LIBXSMM_DNN_ELTWISE_FTYPE *src, LIBXSMM_DNN_ELTWISE_FTYPE *dst, int start_thread, int tid, int nthreads)
{
  const int ltid = tid - start_thread;
  /* compute chunk size */
  const libxsmm_blasint chunksize = (size % nthreads == 0) ? (size / nthreads) : (size / nthreads) + 1;
  /* compute thr_begin and thr_end */
  const libxsmm_blasint thr_begin = (ltid * chunksize < size) ? (ltid * chunksize) : size;
  const libxsmm_blasint thr_end = ((ltid + 1) * chunksize < size) ? ((ltid + 1) * chunksize) : size;
  libxsmm_blasint i;

  for (i = thr_begin; i < thr_end; i++) {
    dst[i] = src[i];
  }
}


LIBXSMM_API_INTERN void libxsmm_internal_matrix_complement(libxsmm_blasint size, LIBXSMM_DNN_ELTWISE_FTYPE *src, LIBXSMM_DNN_ELTWISE_FTYPE *dst, int start_thread, int tid, int nthreads)
{
  const int ltid = tid - start_thread;
  /* compute chunk size */
  const libxsmm_blasint chunksize = (size % nthreads == 0) ? (size / nthreads) : (size / nthreads) + 1;
  /* compute thr_begin and thr_end */
  const libxsmm_blasint thr_begin = (ltid * chunksize < size) ? (ltid * chunksize) : size;
  const libxsmm_blasint thr_end = ((ltid + 1) * chunksize < size) ? ((ltid + 1) * chunksize) : size;
  libxsmm_blasint i;

  for (i = thr_begin; i < thr_end; i++) {
    dst[i] = 1 - src[i];
  }
}


LIBXSMM_API_INTERN void libxsmm_internal_matrix_complement_square(libxsmm_blasint size, LIBXSMM_DNN_ELTWISE_FTYPE *src, LIBXSMM_DNN_ELTWISE_FTYPE *dst, int start_thread, int tid, int nthreads)
{
  const int ltid = tid - start_thread;
  /* compute chunk size */
  const libxsmm_blasint chunksize = (size % nthreads == 0) ? (size / nthreads) : (size / nthreads) + 1;
  /* compute thr_begin and thr_end */
  const libxsmm_blasint thr_begin = (ltid * chunksize < size) ? (ltid * chunksize) : size;
  const libxsmm_blasint thr_end = ((ltid + 1) * chunksize < size) ? ((ltid + 1) * chunksize) : size;
  libxsmm_blasint i;

  for (i = thr_begin; i < thr_end; i++) {
    dst[i] = 1 - (src[i] * src[i]);
  }
}


LIBXSMM_API_INTERN void libxsmm_internal_matrix_inverse(libxsmm_blasint size, LIBXSMM_DNN_ELTWISE_FTYPE *src, LIBXSMM_DNN_ELTWISE_FTYPE *dst, int start_thread, int tid, int nthreads)
{
  const int ltid = tid - start_thread;
  /* compute chunk size */
  const libxsmm_blasint chunksize = (size % nthreads == 0) ? (size / nthreads) : (size / nthreads) + 1;
  /* compute thr_begin and thr_end */
  const libxsmm_blasint thr_begin = (ltid * chunksize < size) ? (ltid * chunksize) : size;
  const libxsmm_blasint thr_end = ((ltid + 1) * chunksize < size) ? ((ltid + 1) * chunksize) : size;
  libxsmm_blasint i;

  for (i = thr_begin; i < thr_end; i++) {
    dst[i] = -src[i];
  }
}


LIBXSMM_API_INTERN void libxsmm_internal_matrix_1D_2D(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint bm, libxsmm_blasint bn, LIBXSMM_DNN_ELTWISE_FTYPE *src, LIBXSMM_DNN_ELTWISE_FTYPE *dst, int start_thread, int tid, int nthreads)
{
  const int ltid = tid - start_thread;
  /* compute chunk size */
  const libxsmm_blasint chunksize = (m % nthreads == 0) ? (m / nthreads) : (m / nthreads) + 1;
  /* compute thr_begin and thr_end */
  const libxsmm_blasint thr_begin = (ltid * chunksize < m) ? (ltid * chunksize) : m;
  const libxsmm_blasint thr_end = ((ltid + 1) * chunksize < m) ? ((ltid + 1) * chunksize) : m;
  libxsmm_blasint i, j;
  LIBXSMM_VLA_DECL(4, LIBXSMM_DNN_ELTWISE_FTYPE, real_dst, (LIBXSMM_DNN_ELTWISE_FTYPE*)dst, m/bm, bn, bm);

  for (i = thr_begin; i < thr_end; i++) {
    const libxsmm_blasint mb = i/bm;
    const libxsmm_blasint ibm = i%bm;
    for (j = 0; j < n; j++) {
      const libxsmm_blasint nb = j/bn;
      const libxsmm_blasint ibn = j%bn;
      LIBXSMM_VLA_ACCESS(4, real_dst, nb, mb, ibn, ibm, m/bm, bn, bm) = src[i];
    }
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
      libxsmm_internal_matrix_relu(size, temp, dst, start_thread, tid, handle->nthreads);
      break;
    case 2:
      libxsmm_internal_matrix_sigmoid(size, temp, dst, start_thread, tid, handle->nthreads);
      break;
    case 3:
      libxsmm_internal_matrix_tanh(size, temp, dst, start_thread, tid, handle->nthreads);
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

LIBXSMM_API_INTERN void libxsmm_internal_matrix_zero_ld(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld, LIBXSMM_DNN_ELTWISE_FTYPE *srcdst) {
  libxsmm_blasint i, j;

  for ( j = 0; j < n; ++j ) {
    LIBXSMM_PRAGMA_SIMD
    for ( i = 0; i < m; ++i ) {
      srcdst[(j*ld)+i] = (LIBXSMM_DNN_ELTWISE_FTYPE)0;
    }
  }
}

LIBXSMM_API_INTERN void libxsmm_internal_matrix_add_ld(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld, LIBXSMM_DNN_ELTWISE_FTYPE *src0, LIBXSMM_DNN_ELTWISE_FTYPE *src1, LIBXSMM_DNN_ELTWISE_FTYPE *dst) {
  libxsmm_blasint i, j;

  for ( j = 0; j < n; ++j ) {
    LIBXSMM_PRAGMA_SIMD
    for ( i = 0; i < m; ++i ) {
      dst[(j*ld)+i] = src0[(j*ld)+i] + src1[(j*ld)+i];
    }
  }
}

LIBXSMM_API_INTERN void libxsmm_internal_matrix_eltwise_mult_ld(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld, LIBXSMM_DNN_ELTWISE_FTYPE *src0, LIBXSMM_DNN_ELTWISE_FTYPE *src1, LIBXSMM_DNN_ELTWISE_FTYPE *dst) {
  libxsmm_blasint i, j;

  for ( j = 0; j < n; ++j ) {
    LIBXSMM_PRAGMA_SIMD
    for ( i = 0; i < m; ++i ) {
      dst[(j*ld)+i] = src0[(j*ld)+i] * src1[(j*ld)+i];
    }
  }
}

LIBXSMM_API_INTERN void libxsmm_internal_matrix_add_colvector_ld(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld, LIBXSMM_DNN_ELTWISE_FTYPE *srcdst, LIBXSMM_DNN_ELTWISE_FTYPE *colv) {
  libxsmm_blasint i, j;

  for ( j = 0; j < n; ++j ) {
    LIBXSMM_PRAGMA_SIMD
    for ( i = 0; i < m; ++i ) {
      srcdst[(j*ld)+i] += colv[i];
    }
  }
}

LIBXSMM_API_INTERN void libxsmm_internal_matrix_bcst_colvector_ld(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld, LIBXSMM_DNN_ELTWISE_FTYPE *srcdst, LIBXSMM_DNN_ELTWISE_FTYPE *colv) {
  libxsmm_blasint i, j;

  for ( j = 0; j < n; ++j ) {
    LIBXSMM_PRAGMA_SIMD
    for ( i = 0; i < m; ++i ) {
      srcdst[(j*ld)+i] = colv[i];
    }
  }
}

LIBXSMM_API_INTERN void libxsmm_internal_matrix_sigmoid_ld(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld, LIBXSMM_DNN_ELTWISE_FTYPE *src, LIBXSMM_DNN_ELTWISE_FTYPE *dst) {
  libxsmm_blasint i, j;

  for ( j = 0; j < n; ++j ) {
    LIBXSMM_PRAGMA_SIMD
    for ( i = 0; i < m; ++i ) {
      LIBXSMM_DNN_ELTWISE_FTYPE mid_value = (LIBXSMM_DNN_ELTWISE_FTYPE)exp(((LIBXSMM_DNN_ELTWISE_FTYPE)-1)*src[(j*ld)+i]);
      dst[(j*ld)+i] = (LIBXSMM_DNN_ELTWISE_FTYPE)1 / ((LIBXSMM_DNN_ELTWISE_FTYPE)1 + mid_value);
    }
  }
}

LIBXSMM_API_INTERN void libxsmm_internal_matrix_tanh_ld(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld, LIBXSMM_DNN_ELTWISE_FTYPE *src, LIBXSMM_DNN_ELTWISE_FTYPE *dst) {
  libxsmm_blasint i, j;

  for ( j = 0; j < n; ++j ) {
    LIBXSMM_PRAGMA_SIMD
    for ( i = 0; i < m; ++i ) {
     dst[(j*ld)+i] = (LIBXSMM_DNN_ELTWISE_FTYPE)tanh(src[(j*ld)+i]);
    }
  }
}

LIBXSMM_API_INTERN void libxsmm_internal_matrix_relu_ld(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld, LIBXSMM_DNN_ELTWISE_FTYPE *src, LIBXSMM_DNN_ELTWISE_FTYPE *dst) {
  libxsmm_blasint i, j;

  for ( j = 0; j < n; ++j ) {
    LIBXSMM_PRAGMA_SIMD
    for ( i = 0; i < m; ++i ) {
      dst[(j*ld)+i] = (src[(j*ld)+i] < 0) ? (LIBXSMM_DNN_ELTWISE_FTYPE)0 : src[(j*ld)+i];
    }
  }
}

LIBXSMM_API_INTERN void libxsmm_internal_matrix_sigmoid_inverse_ld(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld, LIBXSMM_DNN_ELTWISE_FTYPE *src, LIBXSMM_DNN_ELTWISE_FTYPE *dst) {
  libxsmm_blasint i, j;

  for ( j = 0; j < n; ++j ) {
    LIBXSMM_PRAGMA_SIMD
    for ( i = 0; i < m; ++i ) {
      LIBXSMM_DNN_ELTWISE_FTYPE exp_value = (LIBXSMM_DNN_ELTWISE_FTYPE)exp(((LIBXSMM_DNN_ELTWISE_FTYPE)-1)*src[(j*ld)+i]);
      LIBXSMM_DNN_ELTWISE_FTYPE mid_value = (LIBXSMM_DNN_ELTWISE_FTYPE)1 / ((LIBXSMM_DNN_ELTWISE_FTYPE)1 + exp_value);
      dst[(j*ld)+i] = ((LIBXSMM_DNN_ELTWISE_FTYPE)1 - mid_value) * mid_value;
    }
  }
}

LIBXSMM_API_INTERN void libxsmm_internal_matrix_tanh_inverse_ld(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld, LIBXSMM_DNN_ELTWISE_FTYPE *src, LIBXSMM_DNN_ELTWISE_FTYPE *dst) {
  libxsmm_blasint i, j;

  for ( j = 0; j < n; ++j ) {
    LIBXSMM_PRAGMA_SIMD
    for ( i = 0; i < m; ++i ) {
     LIBXSMM_DNN_ELTWISE_FTYPE tanh_value = (LIBXSMM_DNN_ELTWISE_FTYPE)tanh(src[i]);
     dst[(j*ld)+i] = (LIBXSMM_DNN_ELTWISE_FTYPE)1 - (tanh_value * tanh_value);
    }
  }
}

LIBXSMM_API_INTERN void libxsmm_internal_matrix_relu_inverse_ld(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld, LIBXSMM_DNN_ELTWISE_FTYPE *src, LIBXSMM_DNN_ELTWISE_FTYPE *dst) {
  libxsmm_blasint i, j;

  for ( j = 0; j < n; ++j ) {
    LIBXSMM_PRAGMA_SIMD
    for ( i = 0; i < m; ++i ) {
      dst[(j*ld)+i] = (src[(j*ld)+i] < 0) ? (LIBXSMM_DNN_ELTWISE_FTYPE)0 : (LIBXSMM_DNN_ELTWISE_FTYPE)1;
    }
  }
}

