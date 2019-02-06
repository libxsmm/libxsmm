/******************************************************************************
** Copyright (c) 2018-2019, Intel Corporation                                **
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
/* Kunal Banerjee, Evangelos Georganas (Intel Corp.)
******************************************************************************/
#include "libxsmm_dnn_elementwise.h"
#include "libxsmm_blocked_gemm_types.h"

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
  const libxsmm_blasint thr_end = LIBXSMM_MIN(ltid * chunksize + chunksize, size);
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
  const libxsmm_blasint thr_end = LIBXSMM_MIN(ltid * chunksize + chunksize, size);
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
  const libxsmm_blasint thr_end = LIBXSMM_MIN(ltid * chunksize + chunksize, size);
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
  const libxsmm_blasint thr_end = LIBXSMM_MIN(ltid * chunksize + chunksize, size);
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
  const libxsmm_blasint thr_end = LIBXSMM_MIN(ltid * chunksize + chunksize, size);
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
  const libxsmm_blasint thr_end = LIBXSMM_MIN(ltid * chunksize + chunksize, size);
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
  const libxsmm_blasint thr_end = LIBXSMM_MIN(ltid * chunksize + chunksize, size);
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
  const libxsmm_blasint thr_end = LIBXSMM_MIN(ltid * chunksize + chunksize, size);
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
  const libxsmm_blasint thr_end = LIBXSMM_MIN(ltid * chunksize + chunksize, size);
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
  const libxsmm_blasint thr_end = LIBXSMM_MIN(ltid * chunksize + chunksize, size);
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
  const libxsmm_blasint thr_end = LIBXSMM_MIN(ltid * chunksize + chunksize, size);
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
  const libxsmm_blasint thr_end = LIBXSMM_MIN(ltid * chunksize + chunksize, size);
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
  const libxsmm_blasint thr_end = LIBXSMM_MIN(ltid * chunksize + chunksize, size);
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
  const libxsmm_blasint thr_end = LIBXSMM_MIN(ltid * chunksize + chunksize, size);
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
  const libxsmm_blasint thr_end = LIBXSMM_MIN(ltid * chunksize + chunksize, m);
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

LIBXSMM_API_INTERN void libxsmm_internal_recursive_step(libxsmm_blocked_gemm_handle* handle, LIBXSMM_DNN_ELTWISE_FTYPE* u, LIBXSMM_DNN_ELTWISE_FTYPE* h, LIBXSMM_DNN_ELTWISE_FTYPE* op1, LIBXSMM_DNN_ELTWISE_FTYPE *op2,
  LIBXSMM_DNN_ELTWISE_FTYPE *temp, LIBXSMM_DNN_ELTWISE_FTYPE *dst, int act, libxsmm_blasint size, int start_thread, int tid)
{
  const int ltid = tid - start_thread;
#if defined(LSTM_TIMING)
  if (ltid == 0) { Gbl_t_recur = libxsmm_timer_tick(); }
#endif
  libxsmm_blocked_gemm_st(handle, u, h, op1, start_thread, ltid);
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

LIBXSMM_API_INTERN void libxsmm_internal_matrix_copy_ld(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld, LIBXSMM_DNN_ELTWISE_FTYPE *src, LIBXSMM_DNN_ELTWISE_FTYPE *dst) {
  libxsmm_blasint i, j;

  for ( j = 0; j < n; ++j ) {
    LIBXSMM_PRAGMA_SIMD
    for ( i = 0; i < m; ++i ) {
      dst[(j*ld)+i] = src[(j*ld)+i];
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

LIBXSMM_API_INTERN void libxsmm_internal_matrix_inplace_eltwise_mult_ld(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld, LIBXSMM_DNN_ELTWISE_FTYPE *src0, LIBXSMM_DNN_ELTWISE_FTYPE *srcdst) {
  libxsmm_blasint i, j;

  for ( j = 0; j < n; ++j ) {
    LIBXSMM_PRAGMA_SIMD
    for ( i = 0; i < m; ++i ) {
      srcdst[(j*ld)+i] *= src0[(j*ld)+i];
    }
  }
}

LIBXSMM_API_INTERN void libxsmm_internal_matrix_eltwise_fma_ld(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld, LIBXSMM_DNN_ELTWISE_FTYPE *src0, LIBXSMM_DNN_ELTWISE_FTYPE *src1, LIBXSMM_DNN_ELTWISE_FTYPE *dst) {
  libxsmm_blasint i, j;

  for ( j = 0; j < n; ++j ) {
    LIBXSMM_PRAGMA_SIMD
    for ( i = 0; i < m; ++i ) {
      dst[(j*ld)+i] += src0[(j*ld)+i] * src1[(j*ld)+i];
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

LIBXSMM_API_INTERN void libxsmm_internal_matrix_bcst_colvector_const_ld(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld, LIBXSMM_DNN_ELTWISE_FTYPE *srcdst, LIBXSMM_DNN_ELTWISE_FTYPE *colv, LIBXSMM_DNN_ELTWISE_FTYPE const_bias) {
  libxsmm_blasint i, j;

  for ( j = 0; j < n; ++j ) {
    LIBXSMM_PRAGMA_SIMD
    for ( i = 0; i < m; ++i ) {
      srcdst[(j*ld)+i] = colv[i] + const_bias;
    }
  }
}

LIBXSMM_API_INTERN void libxsmm_internal_matrix_sigmoid_ld(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld, LIBXSMM_DNN_ELTWISE_FTYPE *src, LIBXSMM_DNN_ELTWISE_FTYPE *dst) {
  libxsmm_blasint i, j;

  for ( j = 0; j < n; ++j ) {
    LIBXSMM_PRAGMA_SIMD
    for ( i = 0; i < m; ++i ) {
      const LIBXSMM_DNN_ELTWISE_FTYPE mid_value = (LIBXSMM_DNN_ELTWISE_FTYPE)exp((double) -src[(j*ld)+i]);
      dst[(j*ld)+i] = (LIBXSMM_DNN_ELTWISE_FTYPE)1 / ((LIBXSMM_DNN_ELTWISE_FTYPE)1 + mid_value);
    }
  }
}

LIBXSMM_API_INTERN void libxsmm_internal_matrix_tanh_ld(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld, LIBXSMM_DNN_ELTWISE_FTYPE *src, LIBXSMM_DNN_ELTWISE_FTYPE *dst) {
  libxsmm_blasint i, j;

  for ( j = 0; j < n; ++j ) {
    LIBXSMM_PRAGMA_SIMD
    for ( i = 0; i < m; ++i ) {
      dst[(j*ld)+i] = (LIBXSMM_DNN_ELTWISE_FTYPE)tanh((double) src[(j*ld)+i]);
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
      LIBXSMM_DNN_ELTWISE_FTYPE exp_value = (LIBXSMM_DNN_ELTWISE_FTYPE)exp((double) -src[(j*ld)+i]);
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
     LIBXSMM_DNN_ELTWISE_FTYPE tanh_value = (LIBXSMM_DNN_ELTWISE_FTYPE)tanh((double) src[(j*ld)+i]);
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

LIBXSMM_API_INTERN void libxsmm_internal_matrix_sigmoid_inverse_inplace_eltwise_mult_ld(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld, LIBXSMM_DNN_ELTWISE_FTYPE *src, LIBXSMM_DNN_ELTWISE_FTYPE *dst) {
  libxsmm_blasint i, j;

  for ( j = 0; j < n; ++j ) {
    LIBXSMM_PRAGMA_SIMD
    for ( i = 0; i < m; ++i ) {
      LIBXSMM_DNN_ELTWISE_FTYPE exp_value = (LIBXSMM_DNN_ELTWISE_FTYPE)exp((double) -src[(j*ld)+i]);
      LIBXSMM_DNN_ELTWISE_FTYPE mid_value = (LIBXSMM_DNN_ELTWISE_FTYPE)1 / ((LIBXSMM_DNN_ELTWISE_FTYPE)1 + exp_value);
      dst[(j*ld)+i] *= ((LIBXSMM_DNN_ELTWISE_FTYPE)1 - mid_value) * mid_value;
    }
  }
}

LIBXSMM_API_INTERN void libxsmm_internal_matrix_tanh_inverse_inplace_eltwise_mult_ld(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld, LIBXSMM_DNN_ELTWISE_FTYPE *src, LIBXSMM_DNN_ELTWISE_FTYPE *dst) {
  libxsmm_blasint i, j;

  for ( j = 0; j < n; ++j ) {
    LIBXSMM_PRAGMA_SIMD
    for ( i = 0; i < m; ++i ) {
     LIBXSMM_DNN_ELTWISE_FTYPE tanh_value = (LIBXSMM_DNN_ELTWISE_FTYPE)tanh((double) src[(j*ld)+i]);
     dst[(j*ld)+i] *= (LIBXSMM_DNN_ELTWISE_FTYPE)1 - (tanh_value * tanh_value);
    }
  }
}

LIBXSMM_API_INTERN void libxsmm_internal_matrix_relu_inverse_inplace_eltwise_mult_ld(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld, LIBXSMM_DNN_ELTWISE_FTYPE *src, LIBXSMM_DNN_ELTWISE_FTYPE *dst) {
  libxsmm_blasint i, j;

  for ( j = 0; j < n; ++j ) {
    LIBXSMM_PRAGMA_SIMD
    for ( i = 0; i < m; ++i ) {
      dst[(j*ld)+i] *= (src[(j*ld)+i] < 0) ? (LIBXSMM_DNN_ELTWISE_FTYPE)0 : (LIBXSMM_DNN_ELTWISE_FTYPE)1;
    }
  }
}

LIBXSMM_API_INTERN void libxsmm_internal_matrix_complement_ld(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld, LIBXSMM_DNN_ELTWISE_FTYPE *src, LIBXSMM_DNN_ELTWISE_FTYPE *dst) {
  libxsmm_blasint i, j;

  for ( j = 0; j < n; ++j ) {
    LIBXSMM_PRAGMA_SIMD
    for ( i = 0; i < m; ++i ) {
     dst[(j*ld)+i] = (LIBXSMM_DNN_ELTWISE_FTYPE)1 - src[(j*ld)+i];
    }
  }
}

LIBXSMM_API_INTERN void libxsmm_internal_matrix_complement_square_ld(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld, LIBXSMM_DNN_ELTWISE_FTYPE *src, LIBXSMM_DNN_ELTWISE_FTYPE *dst) {
  libxsmm_blasint i, j;

  for ( j = 0; j < n; ++j ) {
    LIBXSMM_PRAGMA_SIMD
    for ( i = 0; i < m; ++i ) {
     dst[(j*ld)+i] = (LIBXSMM_DNN_ELTWISE_FTYPE)1 - (src[(j*ld)+i] * src[(j*ld)+i]);
    }
  }
}


LIBXSMM_API_INTERN void libxsmm_internal_matrix_rne_mask_fp32_bfp16_ld(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld, float* src, float* dst) {
  libxsmm_blasint i,j;

  /* rnaz buffer to bfp16 */
  for ( j = 0; j < n; ++j ) {
    for ( i = 0; i < m; ++i ) {
      unsigned int int_round = 0;
      unsigned int do_round = 1;
      const void *const ptr = &int_round;

      int_round = *((unsigned int*)&(src[(j*ld)+i]));

      /* we don't round NaN and inf */
      if ( (int_round & 0x7f800000) == 0x7f800000 ) {
        do_round = 0;
      }

      /* perform round nearest tie even */
      if ( do_round != 0 ) {
        unsigned int fixup = (int_round >> 16) & 1;
        int_round = int_round + 0x00007fff + fixup;
      }

      /* chop bits to create BFP16 in FP32 */
      int_round = int_round & 0xffff0000;

      dst[(j*ld)+i] = *((float*)ptr);
    }
  }
}


LIBXSMM_API_INTERN void libxsmm_internal_compute_dcp_dci_di_df_dp_ld(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld, int timestep, int t, LIBXSMM_DNN_ELTWISE_FTYPE *dout, LIBXSMM_DNN_ELTWISE_FTYPE *dh, LIBXSMM_DNN_ELTWISE_FTYPE *o, LIBXSMM_DNN_ELTWISE_FTYPE *co, LIBXSMM_DNN_ELTWISE_FTYPE *dcs, LIBXSMM_DNN_ELTWISE_FTYPE *ii, LIBXSMM_DNN_ELTWISE_FTYPE *ci, LIBXSMM_DNN_ELTWISE_FTYPE *dci, LIBXSMM_DNN_ELTWISE_FTYPE *di, LIBXSMM_DNN_ELTWISE_FTYPE *cps, LIBXSMM_DNN_ELTWISE_FTYPE *f, LIBXSMM_DNN_ELTWISE_FTYPE *df, LIBXSMM_DNN_ELTWISE_FTYPE *dp, LIBXSMM_DNN_ELTWISE_FTYPE *dcp) {
#if defined(LIBXSMM_INTRINSICS_AVX512)
  libxsmm_blasint i, j;
  __m512 _dout, _dh, _o, _t1, _t2, _co, _dcs, _dcp, _ii, _ci, _dci, _di, _cps, _f, _df, _dp;
  const __m512 _neg_ones = _mm512_set1_ps( (float)-1.0 );
  const __m512 _ones = _mm512_set1_ps( (float)1.0 );
  if (timestep == t-1) {
    for ( j = 0; j < n; ++j ) {
      LIBXSMM_PRAGMA_UNROLL_N(4)
      for ( i = 0; i < m; i += 16 ) {
        _dout = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &dh[(j*ld)+i] );
        _o = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &o[(j*ld)+i] );
        _t1 = _mm512_mul_ps( _dout, _o  );
        _co = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &co[(j*ld)+i] );
        _t2 = _mm512_fnmsub_ps ( _co, _co, _neg_ones);
        _t1 = _mm512_mul_ps( _t1, _t2 );
        _dcs = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &dcs[(j*ld)+i] );
        _dcp = _mm512_add_ps( _dcs, _t1 );
        _ii = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &ii[(j*ld)+i] );
        _t1 = _mm512_mul_ps( _ii, _dcp );
        _ci = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &ci[(j*ld)+i] );
        _t2 = _mm512_fnmsub_ps ( _ci, _ci, _neg_ones);
        _dci = _mm512_mul_ps( _t1, _t2 );
        LIBXSMM_INTRINSICS_MM512_STREAM_PS( &dci[(j*ld)+i], _dci );
        _t1 = _mm512_mul_ps( _ci, _dcp );
        _t2 = _mm512_sub_ps( _ones, _ii );
        _di = _mm512_mul_ps( _ii, _t2);
        _di = _mm512_mul_ps( _di, _t1);
        LIBXSMM_INTRINSICS_MM512_STREAM_PS( &di[(j*ld)+i], _di );
        _cps = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &cps[(j*ld)+i] );
        _t1 = _mm512_mul_ps( _cps, _dcp );
        _f = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &f[(j*ld)+i] );
        _t2 = _mm512_sub_ps( _ones, _f );
        _df = _mm512_mul_ps( _f, _t2);
        _df = _mm512_mul_ps( _df, _t1);
        LIBXSMM_INTRINSICS_MM512_STREAM_PS( &df[(j*ld)+i], _df );
        _t1 = _mm512_mul_ps( _dout, _co);
        _t2 = _mm512_sub_ps( _ones, _o );
        _t2 = _mm512_mul_ps( _o, _t2);
        _dp = _mm512_mul_ps( _t1, _t2 );
        LIBXSMM_INTRINSICS_MM512_STREAM_PS( &dp[(j*ld)+i], _dp );
        _dcp = _mm512_mul_ps( _dcp, _f);
        LIBXSMM_INTRINSICS_MM512_STREAM_PS( &dcp[(j*ld)+i], _dcp );
      }
    }
  } else {
    for ( j = 0; j < n; ++j ) {
       LIBXSMM_PRAGMA_UNROLL_N(4)
       for ( i = 0; i < m; i += 16 ) {
        _dout = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &dout[(j*ld)+i] );
        _dh = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &dh[(j*ld)+i] );
        _dout = _mm512_add_ps( _dout, _dh );
        _o = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &o[(j*ld)+i] );
        _t1 = _mm512_mul_ps( _dout, _o  );
        _co = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &co[(j*ld)+i] );
        _t2 = _mm512_fnmsub_ps ( _co, _co, _neg_ones);
        _t1 = _mm512_mul_ps( _t1, _t2 );
        _dcp = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &dcp[(j*ld)+i] );
        _dcp = _mm512_add_ps( _dcp, _t1 );
        _ii = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &ii[(j*ld)+i] );
        _t1 = _mm512_mul_ps( _ii, _dcp );
        _ci = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &ci[(j*ld)+i] );
        _t2 = _mm512_fnmsub_ps ( _ci, _ci, _neg_ones);
        _dci = _mm512_mul_ps( _t1, _t2 );
        LIBXSMM_INTRINSICS_MM512_STREAM_PS( &dci[(j*ld)+i], _dci );
        _t1 = _mm512_mul_ps( _ci, _dcp );
        _t2 = _mm512_sub_ps( _ones, _ii );
        _di = _mm512_mul_ps( _ii, _t2);
        _di = _mm512_mul_ps( _di, _t1);
        LIBXSMM_INTRINSICS_MM512_STREAM_PS( &di[(j*ld)+i], _di );
        _cps = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &cps[(j*ld)+i] );
        _t1 = _mm512_mul_ps( _cps, _dcp );
        _f = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &f[(j*ld)+i] );
        _t2 = _mm512_sub_ps( _ones, _f );
        _df = _mm512_mul_ps( _f, _t2);
        _df = _mm512_mul_ps( _df, _t1);
        LIBXSMM_INTRINSICS_MM512_STREAM_PS( &df[(j*ld)+i], _df );
        _t1 = _mm512_mul_ps( _dout, _co);
        _t2 = _mm512_sub_ps( _ones, _o );
        _t2 = _mm512_mul_ps( _o, _t2);
        _dp = _mm512_mul_ps( _t1, _t2 );
        LIBXSMM_INTRINSICS_MM512_STREAM_PS( &dp[(j*ld)+i], _dp );
        _dcp = _mm512_mul_ps( _dcp, _f);
        LIBXSMM_INTRINSICS_MM512_STREAM_PS( &dcp[(j*ld)+i], _dcp );
      }
    }
  }
#else
LIBXSMM_UNUSED(m);LIBXSMM_UNUSED(n);LIBXSMM_UNUSED(ld);LIBXSMM_UNUSED(timestep);
LIBXSMM_UNUSED(t);LIBXSMM_UNUSED(dout);LIBXSMM_UNUSED(dh);LIBXSMM_UNUSED(o);
LIBXSMM_UNUSED(co);LIBXSMM_UNUSED(dcs);LIBXSMM_UNUSED(ii);LIBXSMM_UNUSED(ci);
LIBXSMM_UNUSED(dci);LIBXSMM_UNUSED(di);LIBXSMM_UNUSED(cps);LIBXSMM_UNUSED(f);
LIBXSMM_UNUSED(df);LIBXSMM_UNUSED(dp);LIBXSMM_UNUSED(dcp);
#endif
}

LIBXSMM_API_INTERN void libxsmm_internal_compute_dcp_dci_di_df_dp_ld_and_reformat_dci_di_df_dp_ld2(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld, libxsmm_blasint ld2, int timestep, int t, LIBXSMM_DNN_ELTWISE_FTYPE *dout, LIBXSMM_DNN_ELTWISE_FTYPE *dh, LIBXSMM_DNN_ELTWISE_FTYPE *o, LIBXSMM_DNN_ELTWISE_FTYPE *co, LIBXSMM_DNN_ELTWISE_FTYPE *dcs, LIBXSMM_DNN_ELTWISE_FTYPE *ii, LIBXSMM_DNN_ELTWISE_FTYPE *ci, LIBXSMM_DNN_ELTWISE_FTYPE *dci, LIBXSMM_DNN_ELTWISE_FTYPE *di, LIBXSMM_DNN_ELTWISE_FTYPE *cps, LIBXSMM_DNN_ELTWISE_FTYPE *f, LIBXSMM_DNN_ELTWISE_FTYPE *df, LIBXSMM_DNN_ELTWISE_FTYPE *dp, LIBXSMM_DNN_ELTWISE_FTYPE *dcp, LIBXSMM_DNN_ELTWISE_FTYPE *dciB, LIBXSMM_DNN_ELTWISE_FTYPE *diB, LIBXSMM_DNN_ELTWISE_FTYPE *dfB, LIBXSMM_DNN_ELTWISE_FTYPE *dpB) {
#if defined(LIBXSMM_INTRINSICS_AVX512)
  libxsmm_blasint i, j;
  __m512 _dout, _dh, _o, _t1, _t2, _co, _dcs, _dcp, _ii, _ci, _dci, _di, _cps, _f, _df, _dp;
  const __m512 _neg_ones = _mm512_set1_ps( (float)-1.0 );
  const __m512 _ones = _mm512_set1_ps( (float)1.0 );

  if (timestep == t-1) {
    for ( j = 0; j < n; ++j ) {
      LIBXSMM_PRAGMA_UNROLL_N(4)
      for ( i = 0; i < m; i += 16 ) {
        _dout = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &dh[(j*ld)+i] );
        _o = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &o[(j*ld)+i] );
        _t1 = _mm512_mul_ps( _dout, _o  );
        _co = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &co[(j*ld)+i] );
        _t2 = _mm512_fnmsub_ps ( _co, _co, _neg_ones);
        _t1 = _mm512_mul_ps( _t1, _t2 );
        _dcs = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &dcs[(j*ld)+i] );
        _dcp = _mm512_add_ps( _dcs, _t1 );
        _ii = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &ii[(j*ld)+i] );
        _t1 = _mm512_mul_ps( _ii, _dcp );
        _ci = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &ci[(j*ld)+i] );
        _t2 = _mm512_fnmsub_ps ( _ci, _ci, _neg_ones);
        _dci = _mm512_mul_ps( _t1, _t2 );
        LIBXSMM_INTRINSICS_MM512_STREAM_PS( &dci[(j*ld)+i], _dci );
        LIBXSMM_INTRINSICS_MM512_STREAM_PS( &dciB[(j*ld2)+i], _dci );
        _t1 = _mm512_mul_ps( _ci, _dcp );
        _t2 = _mm512_sub_ps( _ones, _ii );
        _di = _mm512_mul_ps( _ii, _t2);
        _di = _mm512_mul_ps( _di, _t1);
        LIBXSMM_INTRINSICS_MM512_STREAM_PS( &di[(j*ld)+i], _di );
        LIBXSMM_INTRINSICS_MM512_STREAM_PS( &diB[(j*ld2)+i], _di );
        _cps = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &cps[(j*ld)+i] );
        _t1 = _mm512_mul_ps( _cps, _dcp );
        _f = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &f[(j*ld)+i] );
        _t2 = _mm512_sub_ps( _ones, _f );
        _df = _mm512_mul_ps( _f, _t2);
        _df = _mm512_mul_ps( _df, _t1);
        LIBXSMM_INTRINSICS_MM512_STREAM_PS( &df[(j*ld)+i], _df );
        LIBXSMM_INTRINSICS_MM512_STREAM_PS( &dfB[(j*ld2)+i], _df );
        _t1 = _mm512_mul_ps( _dout, _co);
        _t2 = _mm512_sub_ps( _ones, _o );
        _t2 = _mm512_mul_ps( _o, _t2);
        _dp = _mm512_mul_ps( _t1, _t2 );
        LIBXSMM_INTRINSICS_MM512_STREAM_PS( &dp[(j*ld)+i], _dp );
        LIBXSMM_INTRINSICS_MM512_STREAM_PS( &dpB[(j*ld2)+i], _dp );
        _dcp = _mm512_mul_ps( _dcp, _f);
        LIBXSMM_INTRINSICS_MM512_STREAM_PS( &dcp[(j*ld)+i], _dcp );
      }
    }
  } else {
    for ( j = 0; j < n; ++j ) {
       LIBXSMM_PRAGMA_UNROLL_N(4)
       for ( i = 0; i < m; i += 16 ) {
        _dout = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &dout[(j*ld)+i] );
        _dh = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &dh[(j*ld)+i] );
        _dout = _mm512_add_ps( _dout, _dh );
        _o = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &o[(j*ld)+i] );
        _t1 = _mm512_mul_ps( _dout, _o  );
        _co = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &co[(j*ld)+i] );
        _t2 = _mm512_fnmsub_ps ( _co, _co, _neg_ones);
        _t1 = _mm512_mul_ps( _t1, _t2 );
        _dcp = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &dcp[(j*ld)+i] );
        _dcp = _mm512_add_ps( _dcp, _t1 );
        _ii = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &ii[(j*ld)+i] );
        _t1 = _mm512_mul_ps( _ii, _dcp );
        _ci = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &ci[(j*ld)+i] );
        _t2 = _mm512_fnmsub_ps ( _ci, _ci, _neg_ones);
        _dci = _mm512_mul_ps( _t1, _t2 );
        LIBXSMM_INTRINSICS_MM512_STREAM_PS( &dci[(j*ld)+i], _dci );
        LIBXSMM_INTRINSICS_MM512_STREAM_PS( &dciB[(j*ld2)+i], _dci );
        _t1 = _mm512_mul_ps( _ci, _dcp );
        _t2 = _mm512_sub_ps( _ones, _ii );
        _di = _mm512_mul_ps( _ii, _t2);
        _di = _mm512_mul_ps( _di, _t1);
        LIBXSMM_INTRINSICS_MM512_STREAM_PS( &di[(j*ld)+i], _di );
        LIBXSMM_INTRINSICS_MM512_STREAM_PS( &diB[(j*ld2)+i], _di );
        _cps = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &cps[(j*ld)+i] );
        _t1 = _mm512_mul_ps( _cps, _dcp );
        _f = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &f[(j*ld)+i] );
        _t2 = _mm512_sub_ps( _ones, _f );
        _df = _mm512_mul_ps( _f, _t2);
        _df = _mm512_mul_ps( _df, _t1);
        LIBXSMM_INTRINSICS_MM512_STREAM_PS( &df[(j*ld)+i], _df );
        LIBXSMM_INTRINSICS_MM512_STREAM_PS( &dfB[(j*ld2)+i], _df );
        _t1 = _mm512_mul_ps( _dout, _co);
        _t2 = _mm512_sub_ps( _ones, _o );
        _t2 = _mm512_mul_ps( _o, _t2);
        _dp = _mm512_mul_ps( _t1, _t2 );
        LIBXSMM_INTRINSICS_MM512_STREAM_PS( &dp[(j*ld)+i], _dp );
        LIBXSMM_INTRINSICS_MM512_STREAM_PS( &dpB[(j*ld2)+i], _dp );
        _dcp = _mm512_mul_ps( _dcp, _f);
        LIBXSMM_INTRINSICS_MM512_STREAM_PS( &dcp[(j*ld)+i], _dcp );
      }
    }
  }
#else
LIBXSMM_UNUSED(m);LIBXSMM_UNUSED(n);LIBXSMM_UNUSED(ld);LIBXSMM_UNUSED(timestep);
LIBXSMM_UNUSED(t);LIBXSMM_UNUSED(dout);LIBXSMM_UNUSED(dh);LIBXSMM_UNUSED(o);
LIBXSMM_UNUSED(co);LIBXSMM_UNUSED(dcs);LIBXSMM_UNUSED(ii);LIBXSMM_UNUSED(ci);
LIBXSMM_UNUSED(dci);LIBXSMM_UNUSED(di);LIBXSMM_UNUSED(cps);LIBXSMM_UNUSED(f);
LIBXSMM_UNUSED(df);LIBXSMM_UNUSED(dp);LIBXSMM_UNUSED(dcp);
#endif
}

#if defined(LIBXSMM_INTRINSICS_AVX512)
LIBXSMM_API_INLINE LIBXSMM_INTRINSICS(LIBXSMM_X86_AVX512) __m512 _mm512_tanh_generic_ps( __m512 x ) {
  int i;
  LIBXSMM_DNN_ELTWISE_FTYPE _x[16];
  _mm512_store_ps( _x, x );
  LIBXSMM_PRAGMA_SIMD
  for (i = 0; i < 16; i++) {
    _x[i] = (LIBXSMM_DNN_ELTWISE_FTYPE) tanh((double) _x[i] );
  }
  __m512 result = _mm512_loadu_ps( _x );
  return result;
}
#endif

LIBXSMM_API_INTERN void libxsmm_internal_compute_o_i_f_ci_cs_co_h_ld(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld, LIBXSMM_DNN_ELTWISE_FTYPE *f, LIBXSMM_DNN_ELTWISE_FTYPE *cps, LIBXSMM_DNN_ELTWISE_FTYPE *cs, LIBXSMM_DNN_ELTWISE_FTYPE *ii, LIBXSMM_DNN_ELTWISE_FTYPE *ci,LIBXSMM_DNN_ELTWISE_FTYPE *co, LIBXSMM_DNN_ELTWISE_FTYPE *o, LIBXSMM_DNN_ELTWISE_FTYPE *h) {
#if defined(LIBXSMM_INTRINSICS_AVX512)
#if defined(LIBXSMM_INTEL_COMPILER)
#define _MM512_TANH_PS(A) _mm512_tanh_ps(A)
#else
#define _MM512_TANH_PS(A) _mm512_tanh_generic_ps(A)
#endif
  libxsmm_blasint i, j;
  __m512 _f, _cps, _cs, _ii, _ci, _co, _o, _h;
  const __m512 _halves = _mm512_set1_ps( (float)0.5 );
    for ( j = 0; j < n; ++j ) {
       LIBXSMM_PRAGMA_UNROLL_N(4)
       for ( i = 0; i < m; i += 16 ) {
        _o = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &o[(j*ld)+i] );
        _ii = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &ii[(j*ld)+i] );
        _ci = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &ci[(j*ld)+i] );
        _f = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &f[(j*ld)+i] );
        _cps = LIBXSMM_INTRINSICS_MM512_LOAD_PS( &cps[(j*ld)+i] );
        _o = _mm512_fmadd_ps( _MM512_TANH_PS( _mm512_mul_ps( _o, _halves ) ), _halves, _halves);
        _ii = _mm512_fmadd_ps( _MM512_TANH_PS( _mm512_mul_ps( _ii, _halves ) ), _halves, _halves);
        _ci = _MM512_TANH_PS( _ci );
        _f = _mm512_fmadd_ps( _MM512_TANH_PS( _mm512_mul_ps( _f, _halves ) ), _halves, _halves);
        _cs = _mm512_mul_ps( _f, _cps );
        _cs = _mm512_fmadd_ps( _ii, _ci, _cs );
        _co = _MM512_TANH_PS( _cs );
        _h = _mm512_mul_ps( _o, _co );
        _mm512_store_ps( &o[(j*ld)+i], _o );
        _mm512_store_ps( &ii[(j*ld)+i], _ii );
        _mm512_store_ps( &ci[(j*ld)+i], _ci );
        _mm512_store_ps( &f[(j*ld)+i], _f );
        _mm512_store_ps( &cs[(j*ld)+i], _cs );
        _mm512_store_ps( &co[(j*ld)+i], _co );
        LIBXSMM_INTRINSICS_MM512_STREAM_PS( &h[(j*ld)+i], _h );
      }
  }
#undef _MM512_TANH_PS
#else
LIBXSMM_UNUSED(m);LIBXSMM_UNUSED(n);LIBXSMM_UNUSED(ld);LIBXSMM_UNUSED(f);
LIBXSMM_UNUSED(cps);LIBXSMM_UNUSED(cs);LIBXSMM_UNUSED(ii);LIBXSMM_UNUSED(ci);
LIBXSMM_UNUSED(co);LIBXSMM_UNUSED(o);LIBXSMM_UNUSED(h);
#endif
}
