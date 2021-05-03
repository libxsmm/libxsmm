/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Kunal Banerjee, Evangelos Georganas (Intel Corp.)
******************************************************************************/
#include "libxsmm_dnn_elementwise.h"

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

LIBXSMM_API_INTERN void libxsmm_internal_matrix_zero_ld(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld, LIBXSMM_DNN_ELTWISE_FTYPE *srcdst) {
  libxsmm_blasint i = 0, j;

  for ( j = 0; j < n; ++j ) {
    LIBXSMM_PRAGMA_SIMD
    for ( i = 0; i < m; ++i ) {
      srcdst[(j*ld)+i] = (LIBXSMM_DNN_ELTWISE_FTYPE)0;
    }
  }
}

LIBXSMM_API_INTERN void libxsmm_internal_matrix_copy_ld(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld, LIBXSMM_DNN_ELTWISE_FTYPE *src, LIBXSMM_DNN_ELTWISE_FTYPE *dst) {
  libxsmm_blasint i = 0, j;

  for ( j = 0; j < n; ++j ) {
    LIBXSMM_PRAGMA_SIMD
    for ( i = 0; i < m; ++i ) {
      dst[(j*ld)+i] = src[(j*ld)+i];
    }
  }
}

LIBXSMM_API_INTERN void libxsmm_internal_matrix_add_ld(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld, LIBXSMM_DNN_ELTWISE_FTYPE *src0, LIBXSMM_DNN_ELTWISE_FTYPE *src1, LIBXSMM_DNN_ELTWISE_FTYPE *dst) {
  libxsmm_blasint i = 0, j;

  for ( j = 0; j < n; ++j ) {
    LIBXSMM_PRAGMA_SIMD
    for ( i = 0; i < m; ++i ) {
      dst[(j*ld)+i] = src0[(j*ld)+i] + src1[(j*ld)+i];
    }
  }
}

LIBXSMM_API_INTERN void libxsmm_internal_matrix_sub_ld(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld, LIBXSMM_DNN_ELTWISE_FTYPE *src0, LIBXSMM_DNN_ELTWISE_FTYPE *src1, LIBXSMM_DNN_ELTWISE_FTYPE *dst) {
  libxsmm_blasint i = 0, j;

  for ( j = 0; j < n; ++j ) {
    LIBXSMM_PRAGMA_SIMD
    for ( i = 0; i < m; ++i ) {
      dst[(j*ld)+i] = src0[(j*ld)+i] - src1[(j*ld)+i];
    }
  }
}

LIBXSMM_API_INTERN void libxsmm_internal_matrix_eltwise_mult_ld(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld, LIBXSMM_DNN_ELTWISE_FTYPE *src0, LIBXSMM_DNN_ELTWISE_FTYPE *src1, LIBXSMM_DNN_ELTWISE_FTYPE *dst) {
  libxsmm_blasint i = 0, j;

  for ( j = 0; j < n; ++j ) {
    LIBXSMM_PRAGMA_SIMD
    for ( i = 0; i < m; ++i ) {
      dst[(j*ld)+i] = src0[(j*ld)+i] * src1[(j*ld)+i];
    }
  }
}

LIBXSMM_API_INTERN void libxsmm_internal_matrix_inplace_eltwise_mult_ld(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld, LIBXSMM_DNN_ELTWISE_FTYPE *src0, LIBXSMM_DNN_ELTWISE_FTYPE *srcdst) {
  libxsmm_blasint i = 0, j;

  for ( j = 0; j < n; ++j ) {
    LIBXSMM_PRAGMA_SIMD
    for ( i = 0; i < m; ++i ) {
      srcdst[(j*ld)+i] *= src0[(j*ld)+i];
    }
  }
}

LIBXSMM_API_INTERN void libxsmm_internal_matrix_eltwise_fma_ld(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld, LIBXSMM_DNN_ELTWISE_FTYPE *src0, LIBXSMM_DNN_ELTWISE_FTYPE *src1, LIBXSMM_DNN_ELTWISE_FTYPE *dst) {
  libxsmm_blasint i = 0, j;

  for ( j = 0; j < n; ++j ) {
    LIBXSMM_PRAGMA_SIMD
    for ( i = 0; i < m; ++i ) {
      dst[(j*ld)+i] += src0[(j*ld)+i] * src1[(j*ld)+i];
    }
  }
}

LIBXSMM_API_INTERN void libxsmm_internal_matrix_add_colvector_ld(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld, LIBXSMM_DNN_ELTWISE_FTYPE *srcdst, LIBXSMM_DNN_ELTWISE_FTYPE *colv) {
  libxsmm_blasint i = 0, j;

  for ( j = 0; j < n; ++j ) {
    LIBXSMM_PRAGMA_SIMD
    for ( i = 0; i < m; ++i ) {
      srcdst[(j*ld)+i] += colv[i];
    }
  }
}

LIBXSMM_API_INTERN void libxsmm_internal_matrix_bcst_colvector_ld(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld, LIBXSMM_DNN_ELTWISE_FTYPE *srcdst, LIBXSMM_DNN_ELTWISE_FTYPE *colv) {
  libxsmm_blasint i = 0, j;

  for ( j = 0; j < n; ++j ) {
    LIBXSMM_PRAGMA_SIMD
    for ( i = 0; i < m; ++i ) {
      srcdst[(j*ld)+i] = colv[i];
    }
  }
}

LIBXSMM_API_INTERN void libxsmm_internal_matrix_bcst_cvt_bf16_fp32_colvector_ld(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld, LIBXSMM_DNN_ELTWISE_FTYPE *srcdst, libxsmm_bfloat16 *colv) {
  libxsmm_blasint i, j;
  libxsmm_bfloat16_hp t;

  t.i[0] = 0;
  for ( j = 0; j < n; ++j ) {
    for ( i = 0; i < m; ++i ) {
      t.i[1] = colv[i];
      srcdst[(j*ld)+i] = t.f;
    }
  }
}

LIBXSMM_API_INTERN void libxsmm_internal_matrix_bcst_colvector_const_ld(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld, LIBXSMM_DNN_ELTWISE_FTYPE *srcdst, LIBXSMM_DNN_ELTWISE_FTYPE *colv, LIBXSMM_DNN_ELTWISE_FTYPE const_bias) {
  libxsmm_blasint i = 0, j;

  for ( j = 0; j < n; ++j ) {
    LIBXSMM_PRAGMA_SIMD
    for ( i = 0; i < m; ++i ) {
      srcdst[(j*ld)+i] = colv[i] + const_bias;
    }
  }
}

LIBXSMM_API_INTERN void libxsmm_internal_matrix_bcst_cvt_bf16_fp32_colvector_const_ld(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld, LIBXSMM_DNN_ELTWISE_FTYPE *srcdst, libxsmm_bfloat16 *colv, LIBXSMM_DNN_ELTWISE_FTYPE const_bias) {
  libxsmm_blasint i, j;
  libxsmm_bfloat16_hp t;

  t.i[0] = 0;
  for ( j = 0; j < n; ++j ) {
    for ( i = 0; i < m; ++i ) {
      t.i[1] = colv[i];
      srcdst[(j*ld)+i] = t.f + const_bias;
    }
  }
}

LIBXSMM_API_INTERN void libxsmm_internal_matrix_sigmoid_ld(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld, LIBXSMM_DNN_ELTWISE_FTYPE *src, LIBXSMM_DNN_ELTWISE_FTYPE *dst) {
  libxsmm_blasint i = 0, j;

  for ( j = 0; j < n; ++j ) {
    LIBXSMM_PRAGMA_SIMD
    for ( i = 0; i < m; ++i ) {
      const LIBXSMM_DNN_ELTWISE_FTYPE mid_value = (LIBXSMM_DNN_ELTWISE_FTYPE)exp((double) -src[(j*ld)+i]);
      dst[(j*ld)+i] = (LIBXSMM_DNN_ELTWISE_FTYPE)1 / ((LIBXSMM_DNN_ELTWISE_FTYPE)1 + mid_value);
    }
  }
}

LIBXSMM_API_INTERN void libxsmm_internal_matrix_tanh_ld(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld, LIBXSMM_DNN_ELTWISE_FTYPE *src, LIBXSMM_DNN_ELTWISE_FTYPE *dst) {
  libxsmm_blasint i = 0, j;

  for ( j = 0; j < n; ++j ) {
    LIBXSMM_PRAGMA_SIMD
    for ( i = 0; i < m; ++i ) {
      dst[(j*ld)+i] = (LIBXSMM_DNN_ELTWISE_FTYPE)tanh((double) src[(j*ld)+i]);
    }
  }
}

LIBXSMM_API_INTERN void libxsmm_internal_matrix_relu_ld(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld, LIBXSMM_DNN_ELTWISE_FTYPE *src, LIBXSMM_DNN_ELTWISE_FTYPE *dst) {
  libxsmm_blasint i = 0, j;

  for ( j = 0; j < n; ++j ) {
    LIBXSMM_PRAGMA_SIMD
    for ( i = 0; i < m; ++i ) {
      dst[(j*ld)+i] = (src[(j*ld)+i] < 0) ? (LIBXSMM_DNN_ELTWISE_FTYPE)0 : src[(j*ld)+i];
    }
  }
}

LIBXSMM_API_INTERN void libxsmm_internal_matrix_sigmoid_inverse_ld(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld, LIBXSMM_DNN_ELTWISE_FTYPE *src, LIBXSMM_DNN_ELTWISE_FTYPE *dst) {
  libxsmm_blasint i = 0, j;

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
  libxsmm_blasint i = 0, j;

  for ( j = 0; j < n; ++j ) {
    LIBXSMM_PRAGMA_SIMD
    for ( i = 0; i < m; ++i ) {
      LIBXSMM_DNN_ELTWISE_FTYPE tanh_value = (LIBXSMM_DNN_ELTWISE_FTYPE)tanh((double) src[(j*ld)+i]);
      dst[(j*ld)+i] = (LIBXSMM_DNN_ELTWISE_FTYPE)1 - (tanh_value * tanh_value);
    }
  }
}

LIBXSMM_API_INTERN void libxsmm_internal_matrix_relu_inverse_ld(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld, LIBXSMM_DNN_ELTWISE_FTYPE *src, LIBXSMM_DNN_ELTWISE_FTYPE *dst) {
  libxsmm_blasint i = 0, j;

  for ( j = 0; j < n; ++j ) {
    LIBXSMM_PRAGMA_SIMD
    for ( i = 0; i < m; ++i ) {
      dst[(j*ld)+i] = (src[(j*ld)+i] < 0) ? (LIBXSMM_DNN_ELTWISE_FTYPE)0 : (LIBXSMM_DNN_ELTWISE_FTYPE)1;
    }
  }
}

LIBXSMM_API_INTERN void libxsmm_internal_matrix_sigmoid_inverse_inplace_eltwise_mult_ld(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld, LIBXSMM_DNN_ELTWISE_FTYPE *src, LIBXSMM_DNN_ELTWISE_FTYPE *dst) {
  libxsmm_blasint i = 0, j;

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
  libxsmm_blasint i = 0, j;

  for ( j = 0; j < n; ++j ) {
    LIBXSMM_PRAGMA_SIMD
    for ( i = 0; i < m; ++i ) {
      LIBXSMM_DNN_ELTWISE_FTYPE tanh_value = (LIBXSMM_DNN_ELTWISE_FTYPE)tanh((double) src[(j*ld)+i]);
      dst[(j*ld)+i] *= (LIBXSMM_DNN_ELTWISE_FTYPE)1 - (tanh_value * tanh_value);
    }
  }
}

LIBXSMM_API_INTERN void libxsmm_internal_matrix_relu_inverse_inplace_eltwise_mult_ld(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld, LIBXSMM_DNN_ELTWISE_FTYPE *src, LIBXSMM_DNN_ELTWISE_FTYPE *dst) {
  libxsmm_blasint i = 0, j;

  for ( j = 0; j < n; ++j ) {
    LIBXSMM_PRAGMA_SIMD
    for ( i = 0; i < m; ++i ) {
      dst[(j*ld)+i] *= (src[(j*ld)+i] < 0) ? (LIBXSMM_DNN_ELTWISE_FTYPE)0 : (LIBXSMM_DNN_ELTWISE_FTYPE)1;
    }
  }
}

LIBXSMM_API_INTERN void libxsmm_internal_matrix_complement_ld(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld, LIBXSMM_DNN_ELTWISE_FTYPE *src, LIBXSMM_DNN_ELTWISE_FTYPE *dst) {
  libxsmm_blasint i = 0, j;

  for ( j = 0; j < n; ++j ) {
    LIBXSMM_PRAGMA_SIMD
    for ( i = 0; i < m; ++i ) {
      dst[(j*ld)+i] = (LIBXSMM_DNN_ELTWISE_FTYPE)1 - src[(j*ld)+i];
    }
  }
}

LIBXSMM_API_INTERN void libxsmm_internal_matrix_complement_square_ld(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld, LIBXSMM_DNN_ELTWISE_FTYPE *src, LIBXSMM_DNN_ELTWISE_FTYPE *dst) {
  libxsmm_blasint i = 0, j;

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

LIBXSMM_API_INTERN void libxsmm_internal_matrix_rne_cvt_fp32_bfp16_ld(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld, float* src, libxsmm_bfloat16* dst) {
  libxsmm_blasint i,j;

  /* truncate buffer to bfp16 */
  for ( j = 0; j < n; ++j ) {
    for ( i = 0; i < m; ++i ) {
      unsigned int int_round = 0;
      unsigned int do_round = 1;
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
      /* create the bfp16 value by shifting out the lower 16bits */
      int_round = int_round >> 16;
      dst[(j*ld)+i] = (unsigned short)int_round;
    }
  }
}

LIBXSMM_API_INTERN void libxsmm_internal_matrix_cvt_bf16_fp32_ld(libxsmm_blasint m, libxsmm_blasint n, libxsmm_blasint ld, libxsmm_bfloat16 *src, LIBXSMM_DNN_ELTWISE_FTYPE *dst) {
  libxsmm_blasint i, j;
  libxsmm_bfloat16_hp t;

  t.i[0] = 0;
  for ( j = 0; j < n; ++j ) {
    for ( i = 0; i < m; ++i ) {
      t.i[1] = src[(j*ld)+i];
      dst[(j*ld)+i] = t.f;
    }
  }
}

