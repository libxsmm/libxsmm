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
#include <math.h>

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif


LIBXSMM_API_INTERN void libxsmm_internal_matinit(int seed, LIBXSMM_DNN_ELTWISE_FTYPE *dst,
  libxsmm_blasint nrows, libxsmm_blasint ncols, libxsmm_blasint ld, double scale)
{
  const double seed1 = scale * (seed + 1);
  libxsmm_blasint i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < ncols; ++i) {
    libxsmm_blasint j = 0;
    for (; j < nrows; ++j) {
      const libxsmm_blasint k = i * ld + j;
      dst[k] = (LIBXSMM_DNN_ELTWISE_FTYPE)(seed1 / (k + 1));
    }
    for (; j < ld; ++j) {
      const libxsmm_blasint k = i * ld + j;
      dst[k] = (LIBXSMM_DNN_ELTWISE_FTYPE)seed;
    }
  }
}


LIBXSMM_API_INTERN void libxsmm_internal_matrix_add(libxsmm_blasint size, LIBXSMM_DNN_ELTWISE_FTYPE *a, LIBXSMM_DNN_ELTWISE_FTYPE *b, LIBXSMM_DNN_ELTWISE_FTYPE *c)
{
  libxsmm_blasint i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < size; i++) {
    c[i] = a[i] + b[i];
  }
}


LIBXSMM_API_INTERN void libxsmm_internal_matrix_eltwise_mult(libxsmm_blasint size, LIBXSMM_DNN_ELTWISE_FTYPE *a, LIBXSMM_DNN_ELTWISE_FTYPE *b, LIBXSMM_DNN_ELTWISE_FTYPE *c)
{
  libxsmm_blasint i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < size; i++) {
    c[i] = a[i] * b[i];
  }
}


LIBXSMM_API_INTERN void libxsmm_internal_matrix_sigmoid(libxsmm_blasint size, LIBXSMM_DNN_ELTWISE_FTYPE *src, LIBXSMM_DNN_ELTWISE_FTYPE *dst)
{
  libxsmm_blasint i;
  LIBXSMM_DNN_ELTWISE_FTYPE exp_value;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < size; i++) {
    exp_value = (LIBXSMM_DNN_ELTWISE_FTYPE)exp((double) -src[i]);
    dst[i] = 1 / (1 + exp_value);
  }
}


LIBXSMM_API_INTERN void libxsmm_internal_matrix_tanh(libxsmm_blasint size, LIBXSMM_DNN_ELTWISE_FTYPE *src, LIBXSMM_DNN_ELTWISE_FTYPE *dst)
{
  libxsmm_blasint i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < size; i++) {
    dst[i] = (LIBXSMM_DNN_ELTWISE_FTYPE)tanh((double)src[i]);
  }
}


LIBXSMM_API_INTERN void libxsmm_internal_matrix_relu(libxsmm_blasint size, LIBXSMM_DNN_ELTWISE_FTYPE *src, LIBXSMM_DNN_ELTWISE_FTYPE *dst)
{
  libxsmm_blasint i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < size; i++) {
    dst[i] = (src[i] >= 0) ? src[i] : 0;
  }
}


LIBXSMM_API_INTERN void libxsmm_internal_matrix_sigmoid_inverse(libxsmm_blasint size, LIBXSMM_DNN_ELTWISE_FTYPE *src, LIBXSMM_DNN_ELTWISE_FTYPE *dst)
{
  libxsmm_blasint i;
  LIBXSMM_DNN_ELTWISE_FTYPE exp_value;
  LIBXSMM_DNN_ELTWISE_FTYPE sig_exp;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < size; i++) {
    exp_value = (LIBXSMM_DNN_ELTWISE_FTYPE)exp((double) -src[i]);
    sig_exp = 1 / (1 + exp_value);
    dst[i] = (1 - sig_exp)*sig_exp;
  }
}


LIBXSMM_API_INTERN void libxsmm_internal_matrix_tanh_inverse(libxsmm_blasint size, LIBXSMM_DNN_ELTWISE_FTYPE *src, LIBXSMM_DNN_ELTWISE_FTYPE *dst)
{
  libxsmm_blasint i;
  LIBXSMM_DNN_ELTWISE_FTYPE tanh_value;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < size; i++) {
    tanh_value = (LIBXSMM_DNN_ELTWISE_FTYPE)tanh((double)src[i]);
    dst[i] = 1 - (tanh_value * tanh_value);
  }
}


LIBXSMM_API_INTERN void libxsmm_internal_matrix_relu_inverse(libxsmm_blasint size, LIBXSMM_DNN_ELTWISE_FTYPE *src, LIBXSMM_DNN_ELTWISE_FTYPE *dst, LIBXSMM_DNN_ELTWISE_FTYPE *input)
{
  libxsmm_blasint i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < size; i++) {
    dst[i] = (input[i] >= 0) ? src[i] : 0;
  }
}


LIBXSMM_API_INTERN void libxsmm_internal_matrix_transpose(libxsmm_blasint rows, libxsmm_blasint cols, LIBXSMM_DNN_ELTWISE_FTYPE *src, LIBXSMM_DNN_ELTWISE_FTYPE *dst)
{
  libxsmm_blasint i, j;
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, src2D, src, cols);
  LIBXSMM_VLA_DECL(2, LIBXSMM_DNN_ELTWISE_FTYPE, dst2D, dst, rows);
#if defined(_OPENMP)
# pragma omp parallel for private(i, j) LIBXSMM_OPENMP_COLLAPSE(2)
#endif
  for (i = 0; i < rows; i++) {
    for (j = 0; j < cols; j++) {
      LIBXSMM_VLA_ACCESS(2, dst2D, j, i, rows) = LIBXSMM_VLA_ACCESS(2, src2D, i, j, cols);
    }
  }
}


LIBXSMM_API_INTERN void libxsmm_internal_matrix_copy(libxsmm_blasint size, LIBXSMM_DNN_ELTWISE_FTYPE *src, LIBXSMM_DNN_ELTWISE_FTYPE *dst)
{
  libxsmm_blasint i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < size; i++) {
    dst[i] = src[i];
  }
}


LIBXSMM_API_INTERN void libxsmm_internal_matrix_complement(libxsmm_blasint size, LIBXSMM_DNN_ELTWISE_FTYPE *src, LIBXSMM_DNN_ELTWISE_FTYPE *dst)
{
  libxsmm_blasint i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < size; i++) {
    dst[i] = 1 - src[i];
  }
}


LIBXSMM_API_INTERN void libxsmm_internal_matrix_complement_square(libxsmm_blasint size, LIBXSMM_DNN_ELTWISE_FTYPE *src, LIBXSMM_DNN_ELTWISE_FTYPE *dst)
{
  libxsmm_blasint i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < size; i++) {
    dst[i] = 1 - (src[i] * src[i]);
  }
}


LIBXSMM_API_INTERN void libxsmm_internal_recursive_step(libxsmm_bgemm_handle* handle, LIBXSMM_DNN_ELTWISE_FTYPE* u, LIBXSMM_DNN_ELTWISE_FTYPE* h, LIBXSMM_DNN_ELTWISE_FTYPE* op1, LIBXSMM_DNN_ELTWISE_FTYPE *op2,
  LIBXSMM_DNN_ELTWISE_FTYPE *temp, LIBXSMM_DNN_ELTWISE_FTYPE *dst, int act, libxsmm_blasint size, int tid, int nthreads)
{
#if defined(LSTM_TIMING)
  Gbl_t_recur = libxsmm_timer_tick();
#endif
  libxsmm_bgemm(handle, u, h, op1, tid, nthreads);
#if defined(LSTM_TIMING)
  Gbl_duration_recur = libxsmm_timer_duration(Gbl_t_recur, libxsmm_timer_tick());
  Gbl_t_recur_total += Gbl_duration_recur;
  Gbl_t_eltwise = libxsmm_timer_tick();
#endif
  libxsmm_internal_matrix_add(size, op1, op2, temp);
#if defined(LSTM_TIMING)
  Gbl_duration_eltwise = libxsmm_timer_duration(Gbl_t_eltwise, libxsmm_timer_tick());
  Gbl_t_eltwise_total += Gbl_duration_eltwise;
  Gbl_t_nonlin = libxsmm_timer_tick();
#endif
  switch (act) {
    case 0:
      /* do nothing -- this is required for the last time step */
      dst = temp;
      break;
    case 1:
      libxsmm_internal_matrix_relu(size, temp, dst);
      break;
    case 2:
      libxsmm_internal_matrix_sigmoid(size, temp, dst);
      break;
    case 3:
      libxsmm_internal_matrix_tanh(size, temp, dst);
      break;
    default:
      /* fprintf(stdout, "Unsupported activation function: %d\n", act); */
      dst = temp;
  }
#if defined(LSTM_TIMING)
  Gbl_duration_nonlin = libxsmm_timer_duration(Gbl_t_nonlin, libxsmm_timer_tick());
  Gbl_t_nonlin_total += Gbl_duration_nonlin;
#endif
}

