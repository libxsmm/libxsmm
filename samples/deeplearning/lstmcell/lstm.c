/******************************************************************************
** Copyright (c) 2016-2018, Intel Corporation                                **
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
/* Kunal Banerjee (Intel Corp.), Dheevatsa Mudigere (Intel Corp.)
   Alexander Heinecke (Intel Corp.), Hans Pabst (Intel Corp.)
******************************************************************************/
#include <libxsmm.h>
#include <math.h>

#define CHKERR_LIBXSMM_DNN(A) if ( A != LIBXSMM_DNN_SUCCESS ) fprintf(stderr, "%s\n", libxsmm_dnn_get_error(A) );

#define LSTM_TIMING

/* #define NON_FUSED_INPUT_GEMM */

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#if defined(__MKL) || defined(MKL_DIRECT_CALL_SEQ) || defined(MKL_DIRECT_CALL)
# include <mkl_service.h>
#endif
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

#if !defined(ITYPE)
# define ITYPE float
#endif

#if !defined(CHECK) && \
  (!defined(__BLAS) || (0 != __BLAS)) && /* BLAS available */ \
  (LIBXSMM_EQUAL(ITYPE, float) || LIBXSMM_EQUAL(ITYPE, double))
# define CHECK
#endif

#if defined(LSTM_TIMING)
double Gbl_t_input_total = 0., Gbl_t_recur_total = 0., Gbl_t_eltwise_total = 0., Gbl_t_nonlin_total = 0.;
unsigned long long Gbl_t_input = 0, Gbl_t_recur = 0, Gbl_t_eltwise = 0, Gbl_t_nonlin = 0;
double Gbl_duration_input = 0., Gbl_duration_recur = 0., Gbl_duration_eltwise = 0., Gbl_duration_nonlin = 0.;
#endif

struct rnn_handle {
  libxsmm_blasint m;
  libxsmm_blasint n;
  libxsmm_blasint k;
  libxsmm_blasint t;
  ITYPE *w;
  ITYPE *xt;
  ITYPE *u;
  ITYPE *h;
  ITYPE *z1t;
  ITYPE *z2;
  ITYPE *z;
  /* UPD */
  ITYPE* djdht;
  ITYPE* deltat;
  ITYPE* djdu;
  ITYPE* djdw;
  ITYPE* djdxt;
  ITYPE* di1;
  ITYPE* di2;
  ITYPE* dj1;
  ITYPE* dw1;
  ITYPE* uTp;
  ITYPE* wTp;
  ITYPE* hTp;
  ITYPE* xTp;
  libxsmm_bgemm_handle *handlewx;
  libxsmm_bgemm_handle *handleuh;
  libxsmm_bgemm_handle *handlett;
  libxsmm_bgemm_handle *handlewd;
};

struct lstm_handle {
  libxsmm_blasint m;
  libxsmm_blasint n;
  libxsmm_blasint k;
  libxsmm_blasint t;
  ITYPE *wi;
  ITYPE *wf;
  ITYPE *wo;
  ITYPE *wc;
  ITYPE *xt;
  ITYPE *ri;
  ITYPE *rf;
  ITYPE *ro;
  ITYPE *rc;
  ITYPE *h;
  ITYPE *i1t;
  ITYPE *i2;
  ITYPE *f1t;
  ITYPE *f2;
  ITYPE *o1t;
  ITYPE *o2;
  ITYPE *c1t;
  ITYPE *c2;
  ITYPE *i;
  ITYPE *f;
  ITYPE *o;
  ITYPE *c;
  ITYPE *dh;
  ITYPE *d1;
  ITYPE *d2;
  ITYPE *d;
  /* UPD */
  ITYPE *i3;
  ITYPE *f3;
  ITYPE *d4;
  ITYPE *djdht;
  ITYPE *deltat;
  ITYPE *djddt;
  ITYPE *djdit;
  ITYPE *djdft;
  ITYPE *djdct;
  ITYPE *djdot;
  ITYPE *djdxt;
  ITYPE *djdwi;
  ITYPE *djdwf;
  ITYPE *djdwo;
  ITYPE *djdwc;
  ITYPE *djdri;
  ITYPE *djdrf;
  ITYPE *djdro;
  ITYPE *djdrc;
  ITYPE *djdbi;
  ITYPE *djdbf;
  ITYPE *djdbo;
  ITYPE *djdbc;
  ITYPE *rTp;
  ITYPE *wTp;
  ITYPE *deltaTp;
  ITYPE *xTp;
  libxsmm_bgemm_handle *handlewx;
  libxsmm_bgemm_handle *handleuh;
  libxsmm_bgemm_handle *handlett;
  libxsmm_bgemm_handle *handlewd;
};


void matinit(int seed, ITYPE * dst,
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
      dst[k] = (ITYPE)(seed1 / (k + 1));
    }
    for (; j < ld; ++j) {
      const libxsmm_blasint k = i * ld + j;
      dst[k] = (ITYPE)seed;
    }
  }
}


void matrix_add(libxsmm_blasint size, ITYPE *a, ITYPE *b, ITYPE *c)
{
  libxsmm_blasint i;
#if defined(_OPENMP)
# pragma omp parallel for private(i, size)
#endif
  for (i = 0; i < size; i++) {
    c[i] = a[i] + b[i];
  }
}


void matrix_eltwise_mult(libxsmm_blasint size, ITYPE *a, ITYPE *b, ITYPE *c)
{
  libxsmm_blasint i;
#if defined(_OPENMP)
# pragma omp parallel for private(i, size)
#endif
  for (i = 0; i < size; i++) {
    c[i] = a[i] * b[i];
  }
}


void matrix_sigmoid(libxsmm_blasint size, ITYPE *src, ITYPE *dst)
{
  libxsmm_blasint i;
  ITYPE exp_value;
#if defined(_OPENMP)
# pragma omp parallel for private(i, size)
#endif
  for (i = 0; i < size; i++) {
    exp_value = (ITYPE)exp( -src[i]);
    dst[i] = 1 / (1 + exp_value);
  }
}


void matrix_tanh(libxsmm_blasint size, ITYPE *src, ITYPE *dst)
{
  libxsmm_blasint i;
#if defined(_OPENMP)
# pragma omp parallel for private(i, size)
#endif
  for (i = 0; i < size; i++) {
    dst[i] = tanh(src[i]);
  }
}


void matrix_relu(libxsmm_blasint size, ITYPE *src, ITYPE *dst)
{
  libxsmm_blasint i;
#if defined(_OPENMP)
# pragma omp parallel for private(i, size)
#endif
  for (i = 0; i < size; i++) {
    dst[i] = (src[i] >= 0) ? src[i] : 0;
  }
}


void matrix_sigmoid_inverse(libxsmm_blasint size, ITYPE *src, ITYPE *dst)
{
  libxsmm_blasint i;
  ITYPE exp_value;
  ITYPE sig_exp;
#if defined(_OPENMP)
# pragma omp parallel for private(i, size)
#endif
  for (i = 0; i < size; i++) {
    exp_value = (ITYPE)exp( -src[i]);
    sig_exp = 1 / (1 + exp_value);
    dst[i] = (1 - sig_exp)*sig_exp;
  }
}


void matrix_tanh_inverse(libxsmm_blasint size, ITYPE *src, ITYPE *dst)
{
  libxsmm_blasint i;
  ITYPE sech_value;
#if defined(_OPENMP)
# pragma omp parallel for private(i, size)
#endif
  for (i = 0; i < size; i++) {
    sech_value = sech(src[i]);
    dst[i] = sech_value * sech_value;
  }
}


void matrix_relu_inverse(libxsmm_blasint size, ITYPE *src, ITYPE *dst, ITYPE *input)
{
  libxsmm_blasint i;
#if defined(_OPENMP)
# pragma omp parallel for private(i, size)
#endif
  for (i = 0; i < size; i++) {
    dst[i] = (input[i] >= 0) ? src[i] : 0;
  }
}


void matrix_transpose(libxsmm_blasint rows, libxsmm_blasint cols, ITYPE *src, ITYPE *dst)
{
  libxsmm_blasint i, j;
  LIBXSMM_VLA_DECL(2, ITYPE, src2D, src, cols);
  LIBXSMM_VLA_DECL(2, ITYPE, dst2D, dst, rows);
#if defined(_OPENMP)
# pragma omp parallel for private(i, j, rows, cols) collapse(2)
#endif
  for (i = 0; i < rows; i++) {
    for (j = 0; j < cols; j++) {
      LIBXSMM_VLA_ACCESS(2, dst2D, j, i, rows) = LIBXSMM_VLA_ACCESS(2, src2D, i, j, cols);
    }
  }
}


void matrix_copy(libxsmm_blasint size, ITYPE *src, ITYPE *dst)
{
  libxsmm_blasint i;
#if defined(_OPENMP)
# pragma omp parallel for private(i, size)
#endif
  for (i = 0; i < size; i++) {
    dst[i] = src[i];
  }
}


void matrix_complement(libxsmm_blasint size, ITYPE *src, ITYPE *dst)
{
  libxsmm_blasint i;
#if defined(_OPENMP)
# pragma omp parallel for private(i, size)
#endif
  for (i = 0; i < size; i++) {
    dst[i] = 1 - src[i];
  }
}


void matrix_complement_square(libxsmm_blasint size, ITYPE *src, ITYPE *dst)
{
  libxsmm_blasint i;
#if defined(_OPENMP)
# pragma omp parallel for private(i, size)
#endif
  for (i = 0; i < size; i++) {
    dst[i] = 1 - (src[i] * src[i]);
  }
}


void recursive_step(libxsmm_bgemm_handle* handle, ITYPE* u, ITYPE* h, ITYPE* op1, ITYPE *op2,
  ITYPE *temp, ITYPE *dst, int act, libxsmm_blasint size)
{
#if defined(LSTM_TIMING)
  Gbl_t_recur = libxsmm_timer_tick();
#endif
  libxsmm_bgemm_omp(handle, u, h, op1, 1);
#if defined(LSTM_TIMING)
  Gbl_duration_recur = libxsmm_timer_duration(Gbl_t_recur, libxsmm_timer_tick());
  Gbl_t_recur_total += Gbl_duration_recur;
  Gbl_t_eltwise = libxsmm_timer_tick();
#endif
  matrix_add(size, op1, op2, temp);
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
      matrix_relu(size, temp, dst);
      break;
    case 2:
      matrix_sigmoid(size, temp, dst);
      break;
    case 3:
      matrix_tanh(size, temp, dst);
      break;
    default:
      fprintf(stdout, "Unsupported activation function: %d\n", act);
      dst = temp;
  }
#if defined(LSTM_TIMING)
  Gbl_duration_nonlin = libxsmm_timer_duration(Gbl_t_nonlin, libxsmm_timer_tick());
  Gbl_t_nonlin_total += Gbl_duration_nonlin;
#endif
}


libxsmm_dnn_tensor* libxsmm_create_dnn_tensor_rnn( ITYPE *data )
{
  libxsmm_dnn_err_t status;
  /*libxsmm_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_INPUT, &status ); CHKERR_LIBXSMM_DNN( status );*/
  libxsmm_dnn_tensor* tensor  = libxsmm_dnn_link_tensor( NULL, data, &status ); /*CHKERR_LIBXSMM_DNN( status );*/
  return tensor;
}


void rnn_init(struct rnn_handle *rnn, ITYPE *wgold, ITYPE *xgoldt, ITYPE *ugold,
  ITYPE *hgold, ITYPE *z1gold, ITYPE *z2gold, ITYPE *zgold,
  const libxsmm_blasint ldw, const libxsmm_blasint ldx, const libxsmm_blasint ldz,
  const libxsmm_blasint ldu, const libxsmm_blasint ldh, const libxsmm_blasint reuse)
{
#if defined(CHECK)
  const char *const env_check = getenv("CHECK");
  const double check = LIBXSMM_ABS(0 == env_check ? 0 : atof(env_check));
#endif
  const char transa = 'N', transb = 'N'; /* no transposes */
  const int gemm_flags = LIBXSMM_GEMM_FLAGS(transa, transb);
  const ITYPE alpha = 1, beta = 1;
  libxsmm_blasint m = rnn->m;
  libxsmm_blasint n = rnn->n;
  libxsmm_blasint k = rnn->k;
  libxsmm_blasint t = rnn->t;
  ITYPE *w = (ITYPE*)rnn->w;
  ITYPE *xt = (ITYPE*)rnn->xt;
  ITYPE *u = (ITYPE*)rnn->u;
  ITYPE *h = (ITYPE*)rnn->h;
  ITYPE *z1t = (ITYPE*)rnn->z1t;
  ITYPE *z2 = (ITYPE*)rnn->z2;
  ITYPE *z = (ITYPE*)rnn->z;
  libxsmm_bgemm_handle *handlewx = rnn->handlewx;
  libxsmm_bgemm_handle *handleuh = rnn->handleuh;
  LIBXSMM_VLA_DECL(2, ITYPE, xgold, xgoldt, ldx * n);
  LIBXSMM_VLA_DECL(2, ITYPE, x, xt, k * n);
  LIBXSMM_VLA_DECL(2, ITYPE, z1, z1t, m * n);
  LIBXSMM_VLA_DECL(2, ITYPE, hnr, h, m * n);
  LIBXSMM_VLA_DECL(2, ITYPE, znr, z, m * n);

  LIBXSMM_MATINIT(ITYPE, 42, wgold, m, k, ldw, 1.0);
  int it;
  for (it = 0; it < t; ++it) {
    matinit(24, &LIBXSMM_VLA_ACCESS(2, xgold, it, 0, ldx * n), k, n, ldx, 1.0);
  }
  matinit(42, ugold, m, m, ldu, 1.0);
  matinit(24, hgold, m, n, ldh, 1.0);
  matinit( 0, z1gold, m, n, ldz, 1.0);
  matinit( 0, z2gold, m, n, ldz, 1.0);
  matinit( 0, zgold, m, n, ldz, 1.0);
  libxsmm_bgemm_copyin_a(handlewx, wgold, &ldw, w);
  for (it = 0; it < t; ++it) {
    libxsmm_bgemm_copyin_b(handlewx, &LIBXSMM_VLA_ACCESS(2, xgold, it, 0, ldx * n), &ldx, &LIBXSMM_VLA_ACCESS(2, x, it, 0, k * n));
  }
  libxsmm_bgemm_copyin_a(handleuh, ugold, &ldu, u);
  if (reuse) {
    libxsmm_bgemm_copyin_b(handleuh, hgold, &ldh, h);
  } else {
    for (it = 0; it < t; ++it) {
      libxsmm_bgemm_copyin_b(handleuh, hgold, &ldh, &LIBXSMM_VLA_ACCESS(2, hnr, it, 0, m * n));
    }
  }
  for (it = 0; it < t; ++it) {
    libxsmm_bgemm_copyin_c(handlewx, z1gold, &ldz, &LIBXSMM_VLA_ACCESS(2, z1, it, 0, m * n));
  }
  libxsmm_bgemm_copyin_c(handleuh, z2gold, &ldz, z2);
  if (reuse) {
    libxsmm_bgemm_copyin_c(handlewx, zgold, &ldz, z);
  } else {
    for (it = 0; it < t; ++it) {
      libxsmm_bgemm_copyin_c(handlewx, zgold, &ldz, &LIBXSMM_VLA_ACCESS(2, znr, it, 0, m * n));
    }
  }
#if defined(MKL_ENABLE_AVX512)
  mkl_enable_instructions(MKL_ENABLE_AVX512);
#endif
  /* warmup OpenMP (populate thread pool) */
  libxsmm_bgemm_omp(handlewx, w, x, &LIBXSMM_VLA_ACCESS(2, z1, 0, 0, m * n), 1);
#if defined(CHECK)
  if (!LIBXSMM_FEQ(0, check)) {
    LIBXSMM_XBLAS_SYMBOL(ITYPE)(&transa, &transb, &m, &n, &k, &alpha, wgold, &ldw, &LIBXSMM_VLA_ACCESS(2, xgold, 0, 0, ldx * n), &ldx, &beta, z1gold, &ldz);
  }
#endif
  libxsmm_gemm_print(stdout, LIBXSMM_GEMM_PRECISION(ITYPE),
    &transa, &transb, &m, &n, &k, &alpha, w, &ldw, x, &ldx, &beta, &LIBXSMM_VLA_ACCESS(2, z1, 0, 0, m * n), &ldz);
  fprintf(stdout, "\n\n");
  /* warmup OpenMP (populate thread pool) */
  libxsmm_bgemm_omp(handleuh, u, h, z2, 1);
#if defined(CHECK)
  if (!LIBXSMM_FEQ(0, check)) {
    LIBXSMM_XBLAS_SYMBOL(ITYPE)(&transa, &transb, &m, &n, &m, &alpha, ugold, &ldu, hgold, &ldh, &beta, z2gold, &ldz);
  }
#endif
  libxsmm_gemm_print(stdout, LIBXSMM_GEMM_PRECISION(ITYPE),
    &transa, &transb, &m, &n, &m, &alpha, u, &ldu, h, &ldh, &beta, z2, &ldz);
  fprintf(stdout, "\n\n");
}


void rnn_execute(struct rnn_handle *rnn, const int nrepeat, const libxsmm_blasint reuse)
{
  const char transa = 'N', transb = 'N'; /* no transposes */
  const int gemm_flags = LIBXSMM_GEMM_FLAGS(transa, transb);
  const ITYPE alpha = 1, beta = 1;
  libxsmm_blasint m = rnn->m;
  libxsmm_blasint n = rnn->n;
  libxsmm_blasint k = rnn->k;
  libxsmm_blasint t = rnn->t;
  const double gflops = ((2.0 * m * n * k) + (2.0 * m * n * m) + (2.0 * m * n)) * t * 1E-9;
  ITYPE *w = (ITYPE*)rnn->w;
  ITYPE *xt = (ITYPE*)rnn->xt;
  ITYPE *u = (ITYPE*)rnn->u;
  ITYPE *h = (ITYPE*)rnn->h;
  ITYPE *z1t = (ITYPE*)rnn->z1t;
  ITYPE *z2 = (ITYPE*)rnn->z2;
  ITYPE *z = (ITYPE*)rnn->z;
  libxsmm_bgemm_handle *handlewx = rnn->handlewx;
  libxsmm_bgemm_handle *handleuh = rnn->handleuh;
  libxsmm_bgemm_handle *handlett = rnn->handlett;
  LIBXSMM_VLA_DECL(2, ITYPE, x, xt, k * n);
  LIBXSMM_VLA_DECL(2, ITYPE, z1, z1t, m * n);
  LIBXSMM_VLA_DECL(2, ITYPE, hnr, h, m * n);
  LIBXSMM_VLA_DECL(2, ITYPE, znr, z, m * n);
  unsigned long long start;
  double duration;
#if defined(LSTM_TIMING)
  Gbl_t_input_total = 0.; Gbl_t_recur_total = 0.; Gbl_t_eltwise_total = 0.; Gbl_t_nonlin_total = 0.;
  Gbl_t_input = 0; Gbl_t_recur = 0; Gbl_t_eltwise = 0; Gbl_t_nonlin = 0;
  Gbl_duration_input = 0.; Gbl_duration_recur = 0.; Gbl_duration_eltwise = 0.; Gbl_duration_nonlin = 0.;
#endif

  int s;
  int i;
  libxsmm_blasint nt = n*t;
  start = libxsmm_timer_tick();
  for (s = 0; s < nrepeat; ++s) {
#if defined(LSTM_TIMING)
    Gbl_t_input = libxsmm_timer_tick();
#endif
    /* The following loop may be absorbed into libxsmm_lstm_omp */
    libxsmm_bgemm_omp(handlett, w, &LIBXSMM_VLA_ACCESS(2, x, 0, 0, k * n), &LIBXSMM_VLA_ACCESS(2, z1, 0, 0, m * n), 1/*nrepeat*/);
    /*LIBXSMM_XBLAS_SYMBOL(ITYPE)(&transa, &transb, &m, &nt, &k, &alpha, w, m, &LIBXSMM_VLA_ACCESS(2, x, 0, 0, k * n), k, &beta, z1, m);*/
#if defined(LSTM_TIMING)
    Gbl_duration_input = libxsmm_timer_duration(Gbl_t_input, libxsmm_timer_tick());
    Gbl_t_input_total += Gbl_duration_input;
#endif
    if (reuse) {
      for (i = 0; i < t-1; ++i) {
        recursive_step(handleuh, u, h, z2, &LIBXSMM_VLA_ACCESS(2, z1, i, 0, m * n), z, h, 1, m * n); /*sigmoid*/
      }
      recursive_step(handleuh, u, h, z2, &LIBXSMM_VLA_ACCESS(2, z1, t-1, 0, m * n), z, z, 0, m * n); /*nop*/
    } else {
      for (i = 0; i < t-1; ++i) {
        recursive_step(handleuh, u, &LIBXSMM_VLA_ACCESS(2, hnr, i, 0, m * n), z2, &LIBXSMM_VLA_ACCESS(2, z1, i, 0, m * n),
          &LIBXSMM_VLA_ACCESS(2, znr, i, 0, m * n), &LIBXSMM_VLA_ACCESS(2, hnr, i+1, 0, m * n), 1, m * n); /*sigmoid*/
      }
      recursive_step(handleuh, u, &LIBXSMM_VLA_ACCESS(2, hnr, t-1, 0, m * n), z2, &LIBXSMM_VLA_ACCESS(2, z1, t-1, 0, m * n),
        &LIBXSMM_VLA_ACCESS(2, znr, t-1, 0, m * n), &LIBXSMM_VLA_ACCESS(2, znr, t-1, 0, m * n), 0, m * n); /*nop*/
    }
  }
  duration = libxsmm_timer_duration(start, libxsmm_timer_tick());
  if (0 < duration) {
    fprintf(stdout, "\tLIBXSMM: %.1f GFLOPS/s\n", gflops * nrepeat / duration);
  }
#if defined(LSTM_TIMING)
  double t_total = Gbl_t_input_total + Gbl_t_recur_total + Gbl_t_eltwise_total + Gbl_t_nonlin_total;
  fprintf(stdout, "Percentage of time spent in input matrix multiplication: %lf\n", Gbl_t_input_total*100.0/t_total);
  fprintf(stdout, "Percentage of time spent in recurrence matrix multiplication: %lf\n", Gbl_t_recur_total*100.0/t_total);
  fprintf(stdout, "Percentage of time spent in element-wise operations: %lf\n", Gbl_t_eltwise_total*100.0/t_total);
  fprintf(stdout, "Percentage of time spent in non-linear operations: %lf\n", Gbl_t_nonlin_total*100.0/t_total);
#endif
}


void rnn_destroy(struct rnn_handle *rnn)
{
#if 0
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( rnn->w ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( rnn->xt ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( rnn->u ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( rnn->h ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( rnn->z1t ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( rnn->z2 ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( rnn->z ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( rnn->djdht ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( rnn->deltat ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( rnn->djdu ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( rnn->djdw ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( rnn->djdxt ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( rnn->di1 ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( rnn->di2 ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( rnn->dj1 ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( rnn->dw1 ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( rnn->uTp ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( rnn->wTp ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( rnn->hTp ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( rnn->xTp ) );
#endif
}


int rnn (const libxsmm_blasint m, const libxsmm_blasint n, const libxsmm_blasint k, const libxsmm_blasint t,
         const libxsmm_blasint bm, const libxsmm_blasint bn, const libxsmm_blasint bk, const libxsmm_bgemm_order order, const int nrepeat,
         const libxsmm_blasint b_m1, const libxsmm_blasint b_n1, const libxsmm_blasint b_k1, const libxsmm_blasint b_k2, const libxsmm_blasint b_m2,
         const libxsmm_blasint ldw, const libxsmm_blasint ldx, const libxsmm_blasint ldz, const libxsmm_blasint ldu, const libxsmm_blasint ldh,
         const libxsmm_blasint reuse)
{
#if defined(CHECK)
  const char *const env_check = getenv("CHECK");
  const double check = LIBXSMM_ABS(0 == env_check ? 0 : atof(env_check));
#endif
  int result = EXIT_SUCCESS;
  const double gflops = ((2.0 * m * n * k) + (2.0 * m * n * m) + (2.0 * m * n)) * t * 1E-9;
  const char transa = 'N', transb = 'N'; /* no transposes */
  const int gemm_flags = LIBXSMM_GEMM_FLAGS(transa, transb);
  const ITYPE alpha = 1, beta = 1;

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload target(LIBXSMM_OFFLOAD_TARGET)
#endif
  {
    ITYPE* wgold = (ITYPE*)libxsmm_malloc(ldw * k * sizeof(ITYPE));
    ITYPE* xgoldt = (ITYPE*)libxsmm_malloc(ldx * n * sizeof(ITYPE) * t);
    ITYPE* ugold = (ITYPE*)libxsmm_malloc(ldu * m * sizeof(ITYPE));
    ITYPE* hgold = (ITYPE*)libxsmm_malloc(ldh * n * sizeof(ITYPE));
    ITYPE* z1gold = (ITYPE*)libxsmm_malloc(ldz * n * sizeof(ITYPE));
    ITYPE* z2gold = (ITYPE*)libxsmm_malloc(ldz * n * sizeof(ITYPE));
    ITYPE* zgold = (ITYPE*)libxsmm_malloc(ldz * n * sizeof(ITYPE));

    ITYPE* w = (ITYPE*)libxsmm_malloc(m * k * sizeof(ITYPE));
    ITYPE* xt = (ITYPE*)libxsmm_malloc(k * n * sizeof(ITYPE) * t);
    ITYPE* u = (ITYPE*)libxsmm_malloc(m * m * sizeof(ITYPE));
    ITYPE* h;
    ITYPE* z;
    if (reuse) {
      h = (ITYPE*)libxsmm_malloc(m * n * sizeof(ITYPE));
      z = (ITYPE*)libxsmm_malloc(m * n * sizeof(ITYPE));
    } else {
      h = (ITYPE*)libxsmm_malloc(m * n * sizeof(ITYPE) * t);
      z = (ITYPE*)libxsmm_malloc(m * n * sizeof(ITYPE) * t);
    }
    ITYPE* z1t = (ITYPE*)libxsmm_malloc(m * n * sizeof(ITYPE) * t);
    ITYPE* z2 = (ITYPE*)libxsmm_malloc(m * n * sizeof(ITYPE));
    LIBXSMM_VLA_DECL(2, ITYPE, xgold, xgoldt, ldx * n);
    libxsmm_bgemm_handle* handlewx = 0;
    libxsmm_bgemm_handle* handleuh = 0;
    libxsmm_bgemm_handle* handlett = 0;
    const libxsmm_gemm_prefetch_type strategy = LIBXSMM_PREFETCH_AUTO;
    handlewx = libxsmm_bgemm_handle_create(LIBXSMM_GEMM_PRECISION(ITYPE), LIBXSMM_GEMM_PRECISION(ITYPE),
      m, n, k, &bm, &bn, &bk, &b_m1, &b_n1, &b_k1, &b_k2,
      &alpha, &beta, &gemm_flags, &strategy, &order);
    handleuh = libxsmm_bgemm_handle_create(LIBXSMM_GEMM_PRECISION(ITYPE), LIBXSMM_GEMM_PRECISION(ITYPE),
      m, n, m, &bm, &bn, &bm, &b_m1, &b_n1, &b_m1, &b_m2,
      &alpha, &beta, &gemm_flags, &strategy, &order);
    handlett = libxsmm_bgemm_handle_create(LIBXSMM_GEMM_PRECISION(ITYPE), LIBXSMM_GEMM_PRECISION(ITYPE),
      m, n*t, k, &bm, &bn, &bk, &b_m1, &b_n1, &b_k1, &b_k2,
      &alpha, &beta, &gemm_flags, &strategy, &order);

    struct rnn_handle rnn;
    rnn.m = m;
    rnn.n = n;
    rnn.k = k;
    rnn.t = t;
    rnn.w = w;
    rnn.xt = xt;
    rnn.u = u;
    rnn.h = h;
    rnn.z1t = z1t;
    rnn.z2 = z2;
    rnn.z = z;
    rnn.handlewx = handlewx;
    rnn.handleuh = handleuh;
    rnn.handlett = handlett;

    if (0 != handlewx && 0 != handleuh && 0 != handlett) {
      rnn_init(&rnn, wgold, xgoldt, ugold, hgold, z1gold, z2gold, zgold, ldw, ldx, ldz, ldu, ldh, reuse);
      rnn_execute(&rnn, nrepeat, reuse);
#if defined(CHECK)
      unsigned long long start;
      double duration;
      int s;
      if (!LIBXSMM_FEQ(0, check)) { /* validate result against LAPACK/BLAS xGEMM */
        ITYPE* ztest = 0;
        int i;
        start = libxsmm_timer_tick();
        for (s = 0; s < nrepeat; ++s) {
          for (i = 0; i < t-1; ++i) {
            LIBXSMM_XBLAS_SYMBOL(ITYPE)(&transa, &transb, &m, &n, &k, &alpha, wgold, &ldw, &LIBXSMM_VLA_ACCESS(2, xgold, i, 0, k * n), &ldx, &beta, z1gold, &ldz);
            LIBXSMM_XBLAS_SYMBOL(ITYPE)(&transa, &transb, &m, &n, &m, &alpha, ugold, &ldu, hgold, &ldh, &beta, z2gold, &ldz);
            matrix_add(m*n, z1gold, z2gold, zgold);
            matrix_relu(m*n, zgold, hgold); /*sigmoid*/
          }
          LIBXSMM_XBLAS_SYMBOL(ITYPE)(&transa, &transb, &m, &n, &k, &alpha, wgold, &ldw, &LIBXSMM_VLA_ACCESS(2, xgold, t-1, 0, k * n), &ldx, &beta, z1gold, &ldz);
          LIBXSMM_XBLAS_SYMBOL(ITYPE)(&transa, &transb, &m, &n, &m, &alpha, ugold, &ldu, hgold, &ldh, &beta, z2gold, &ldz);
          matrix_add(m*n, z1gold, z2gold, zgold);
        }
        duration = libxsmm_timer_duration(start, libxsmm_timer_tick());
        if (0 < duration) {
          fprintf(stdout, "\tBLAS: %.1f GFLOPS/s\n", gflops * nrepeat / duration);
        }
        /* free memory not needed further; avoid double-free later on */
        libxsmm_free(wgold); wgold = 0;
        libxsmm_free(xgoldt); xgoldt = 0;
        libxsmm_free(ugold); ugold = 0;
        libxsmm_free(hgold); hgold = 0;
        libxsmm_free(z1gold); z1gold = 0;
        libxsmm_free(z2gold); z2gold = 0;
        /* allocate C-matrix in regular format, and perform copy-out */
        ztest = (ITYPE*)libxsmm_malloc(ldz * n * sizeof(ITYPE));
        LIBXSMM_VLA_DECL(2, ITYPE, znr, z, m * n);
        if (0 != ztest) {
          libxsmm_matdiff_info diff;
          if (reuse) {
            libxsmm_bgemm_copyout_c(handleuh, z, &ldz, ztest);
          } else {
            libxsmm_bgemm_copyout_c(handleuh, &LIBXSMM_VLA_ACCESS(2, znr, t-1, 0, m * n), &ldz, ztest);
          }
          if (EXIT_SUCCESS == libxsmm_matdiff(LIBXSMM_DATATYPE(ITYPE), m, n, zgold, ztest, &ldz, &ldz, &diff)) {
            fprintf(stdout, "\tdiff: L2abs=%f L2rel=%f\n", diff.l2_abs, diff.linf_abs);
            if (check < 100.0 * diff.normf_rel) {
              fprintf(stderr, "FAILED with an error of %f%%!\n", 100.0 * diff.normf_rel);
              result = EXIT_FAILURE;
            }
          }
          libxsmm_free(ztest);
        }
      }
#endif
      libxsmm_bgemm_handle_destroy(handlewx);
      libxsmm_bgemm_handle_destroy(handleuh);
      libxsmm_bgemm_handle_destroy(handlett);
    }
    else {
      fprintf(stderr, "FAILED to create BGEMM-handle! For details retry with LIBXSMM_VERBOSE=1.\n");
      result = EXIT_FAILURE;
    }
    libxsmm_free(wgold);
    libxsmm_free(xgoldt);
    libxsmm_free(ugold);
    libxsmm_free(hgold);
    libxsmm_free(z1gold);
    libxsmm_free(z2gold);
    libxsmm_free(zgold);
    rnn_destroy(&rnn);
  }
  fprintf(stdout, "Finished\n");

  return result;
}


void rnn_bwd_upd_init(struct rnn_handle *rnn, ITYPE *djdhgoldt, ITYPE *zgoldt, ITYPE *deltagoldt,
  ITYPE *ugold, ITYPE *xgoldt, ITYPE *hgoldt, ITYPE *wgold,
  ITYPE *djdugold, ITYPE *djdwgold, ITYPE *djdxgoldt,
  const libxsmm_blasint ldw, const libxsmm_blasint ldx, const libxsmm_blasint ldz,
  const libxsmm_blasint ldu, const libxsmm_blasint ldh)
{
#if defined(CHECK)
  const char *const env_check = getenv("CHECK");
  const double check = LIBXSMM_ABS(0 == env_check ? 0 : atof(env_check));
#endif
  const char transa = 'N', transb = 'N'; /* no transposes */
  const int gemm_flags = LIBXSMM_GEMM_FLAGS(transa, transb);
  const ITYPE alpha = 1, beta = 1;
  libxsmm_blasint m = rnn->m;
  libxsmm_blasint n = rnn->n;
  libxsmm_blasint k = rnn->k;
  libxsmm_blasint t = rnn->t;
  const libxsmm_blasint ldn = n;
  ITYPE *djdht = (ITYPE*)rnn->djdht;
  ITYPE *zt = (ITYPE*)rnn->z;
  ITYPE *deltat = (ITYPE*)rnn->deltat;
  ITYPE *u = (ITYPE*)rnn->u;
  ITYPE *xt = (ITYPE*)rnn->xt;
  ITYPE *ht = (ITYPE*)rnn->h;
  ITYPE *w = (ITYPE*)rnn->w;
  ITYPE *djdu = (ITYPE*)rnn->djdu;
  ITYPE *djdw = (ITYPE*)rnn->djdw;
  ITYPE *djdxt = (ITYPE*)rnn->djdxt;
  libxsmm_bgemm_handle *handleud = rnn->handlewx;
  libxsmm_bgemm_handle *handledh = rnn->handleuh;
  libxsmm_bgemm_handle *handledx = rnn->handlett;
  libxsmm_bgemm_handle *handlewd = rnn->handlewd;
  ITYPE* ugoldTp = (ITYPE*)libxsmm_malloc(ldu * m * sizeof(ITYPE));
  ITYPE* wgoldTp = (ITYPE*)libxsmm_malloc(ldw * k * sizeof(ITYPE));
  ITYPE* hgoldTp = (ITYPE*)libxsmm_malloc(ldh * n * sizeof(ITYPE));
  ITYPE* xgoldTp = (ITYPE*)libxsmm_malloc(ldx * n * sizeof(ITYPE));
  LIBXSMM_VLA_DECL(2, ITYPE, djdhgold, djdhgoldt, ldh * n);
  LIBXSMM_VLA_DECL(2, ITYPE, djdxgold, djdxgoldt, ldx * n);
  LIBXSMM_VLA_DECL(2, ITYPE, zgold, zgoldt, ldz * n);
  LIBXSMM_VLA_DECL(2, ITYPE, deltagold, deltagoldt, ldz * n);
  LIBXSMM_VLA_DECL(2, ITYPE, xgold, xgoldt, ldx * n);
  LIBXSMM_VLA_DECL(2, ITYPE, hgold, hgoldt, ldh * n);
  LIBXSMM_VLA_DECL(2, ITYPE, djdh, djdht, m * n);
  LIBXSMM_VLA_DECL(2, ITYPE, z, zt, m * n);
  LIBXSMM_VLA_DECL(2, ITYPE, delta, deltat, m * n);
  LIBXSMM_VLA_DECL(2, ITYPE, x, xt, k * n);
  LIBXSMM_VLA_DECL(2, ITYPE, h, ht, m * n);
  LIBXSMM_VLA_DECL(2, ITYPE, djdx, djdxt, k * n);

  LIBXSMM_MATINIT(ITYPE, 42, ugold, m, m, ldu, 1.0);
  LIBXSMM_MATINIT(ITYPE, 42, wgold, m, k, ldw, 1.0);
  LIBXSMM_MATINIT(ITYPE,  0, djdu, m, m, m, 0.0);
  LIBXSMM_MATINIT(ITYPE,  0, djdw, m, k, m, 0.0);
  int it;
  for (it = 0; it < t; ++it) {
    matinit(24, &LIBXSMM_VLA_ACCESS(2, djdhgold, it, 0, ldh * n), m, n, ldh, 1.0);
    matinit(24, &LIBXSMM_VLA_ACCESS(2, zgold, it, 0, ldz * n), m, n, ldz, 1.0);
    matinit(24, &LIBXSMM_VLA_ACCESS(2, xgold, it, 0, ldx * n), k, n, ldx, 1.0);
    matinit(24, &LIBXSMM_VLA_ACCESS(2, hgold, it, 0, ldh * n), m, n, ldh, 1.0);
    matinit( 0, &LIBXSMM_VLA_ACCESS(2, deltagold, it, 0, ldz * n), m, n, ldz, 0.0);
    matinit( 0, &LIBXSMM_VLA_ACCESS(2, djdxgold, it, 0, ldx * n), k, n, k, 0.0);
    matinit( 0, &LIBXSMM_VLA_ACCESS(2, delta, it, 0, m * n), m, n, m, 0.0);
    matinit( 0, &LIBXSMM_VLA_ACCESS(2, djdx, it, 0, k * n), k, n, k, 0.0);
  }
  matrix_transpose(m, m, ugold, ugoldTp);
  libxsmm_bgemm_copyin_a(handleud, ugoldTp, &ldu, u);
  for (it = 0; it < t; ++it) {
    matrix_transpose(m, n, &LIBXSMM_VLA_ACCESS(2, hgold, it, 0, m * n), hgoldTp);
    libxsmm_bgemm_copyin_b(handledh, hgoldTp, &ldn, &LIBXSMM_VLA_ACCESS(2, h, it, 0, m * n));
    matrix_transpose(k, n, &LIBXSMM_VLA_ACCESS(2, xgold, it, 0, k * n), xgoldTp);
    libxsmm_bgemm_copyin_b(handledx, xgoldTp, &ldn, &LIBXSMM_VLA_ACCESS(2, x, it, 0, k * n));
  }
  matrix_transpose(m, k, wgold, wgoldTp);
  libxsmm_bgemm_copyin_a(handlewd, wgoldTp, &ldx, w);
  for (it = 0; it < t; ++it) {
    libxsmm_bgemm_copyin_b(handleud, &LIBXSMM_VLA_ACCESS(2, djdhgold, it, 0, ldh * n), &ldh, &LIBXSMM_VLA_ACCESS(2, djdh, it, 0, m * n));
    libxsmm_bgemm_copyin_b(handleud, &LIBXSMM_VLA_ACCESS(2, zgold, it, 0, ldz * n), &ldz, &LIBXSMM_VLA_ACCESS(2, z, it, 0, m * n));
  }
  fprintf(stdout, "\n\n");
}


void rnn_bwd_upd_execute(struct rnn_handle *rnn, const int nrepeat, const libxsmm_blasint pass)
{
  const char transa = 'N', transb = 'N'; /* no transposes */
  const int gemm_flags = LIBXSMM_GEMM_FLAGS(transa, transb);
  const ITYPE alpha = 1, beta = 1;
  libxsmm_blasint m = rnn->m;
  libxsmm_blasint n = rnn->n;
  libxsmm_blasint k = rnn->k;
  libxsmm_blasint t = rnn->t;
  const double tflops = 12; /* transcendental flops */
  double gflops = m * m; /* U^T */
  gflops += (2.0 * m * n * m); /* U^T * delta */
  gflops += (m * n); /* dJdh + (U^T * delta) */
  gflops += (tflops * m * n); /* sigma'(Z) */
  gflops += (m * n); /* sigma'(Z) * (dJdh + (U^T * delta)) */
  gflops *= t; /* for t time steps */
  double tempflops;
  if (pass == 2 || pass == 3) {
    tempflops = m * n; /* h^T */
    tempflops += (2.0 * m * n * m); /* delta * h^T */
    tempflops *= t; /* for t time steps */
    tempflops += (m * m * (t-1)); /* for summation of dJdU */
    gflops += tempflops;
    tempflops = k * n; /* x^T */
    tempflops += (2.0 * m * n * k); /* delta * x^T */
    tempflops *= t; /* for t time steps */
    tempflops += (m * k * (t-1)); /* for summation of dJdW */
    gflops += tempflops;
  }
  if (pass == 1 || pass == 3) {
    tempflops = m * k; /* W^T */
    tempflops += (2.0 * m * n * k); /* W^T * delta */
    tempflops *= t; /* for t time steps of input */
    gflops += tempflops;
  }
  gflops *= 1E-9; /* to convert flops to Gflops */
  ITYPE *djdht = (ITYPE*)rnn->djdht;
  ITYPE *zt = (ITYPE*)rnn->z;
  ITYPE *deltat = (ITYPE*)rnn->deltat;
  ITYPE *u = (ITYPE*)rnn->u;
  ITYPE *xt = (ITYPE*)rnn->xt;
  ITYPE *ht = (ITYPE*)rnn->h;
  ITYPE *w = (ITYPE*)rnn->w;
  ITYPE *djdu = (ITYPE*)rnn->djdu;
  ITYPE *djdw = (ITYPE*)rnn->djdw;
  ITYPE *djdxt = (ITYPE*)rnn->djdxt;
  ITYPE* zi = (ITYPE*)rnn->z1t;
  ITYPE* di1 = (ITYPE*)rnn->di1;
  ITYPE* di2 = (ITYPE*)rnn->di2;
  ITYPE* dj1 = (ITYPE*)rnn->dj1;
  ITYPE* dw1 = (ITYPE*)rnn->dw1;
  /*
  ITYPE* uTp = (ITYPE*)rnn->uTp;
  ITYPE* wTp = (ITYPE*)rnn->wTp;
  ITYPE* hTp = (ITYPE*)rnn->hTp;
  ITYPE* xTp = (ITYPE*)rnn->xTp;
  */
  libxsmm_bgemm_handle *handleud = rnn->handlewx;
  libxsmm_bgemm_handle *handledh = rnn->handleuh;
  libxsmm_bgemm_handle *handledx = rnn->handlett;
  libxsmm_bgemm_handle *handlewd = rnn->handlewd;
  LIBXSMM_VLA_DECL(2, ITYPE, djdh, djdht, m * n);
  LIBXSMM_VLA_DECL(2, ITYPE, z, zt, m * n);
  LIBXSMM_VLA_DECL(2, ITYPE, delta, deltat, m * n);
  LIBXSMM_VLA_DECL(2, ITYPE, x, xt, k * n);
  LIBXSMM_VLA_DECL(2, ITYPE, h, ht, m * n);
  LIBXSMM_VLA_DECL(2, ITYPE, djdx, djdxt, k * n);
  unsigned long long start;
  double duration;
  int s;
  int i;
  start = libxsmm_timer_tick();
  for (s = 0; s < nrepeat; ++s) {
    LIBXSMM_MATINIT(ITYPE, 0, &LIBXSMM_VLA_ACCESS(2, delta, t-1, 0, m * n), m, n, m, 0.0);
    /* matrix_transpose(m, m, u, uTp); - already taken care of in init */
    for (i = t-2; i >= 0; --i) {
      matrix_sigmoid_inverse(m * n, &LIBXSMM_VLA_ACCESS(2, z, i+1, 0, m * n), zi);
      /* libxsmm_bgemm_omp(handleud, uTp, &LIBXSMM_VLA_ACCESS(2, delta, i+1, 0, m * n), di1, 1); */
      libxsmm_bgemm_omp(handleud, u, &LIBXSMM_VLA_ACCESS(2, delta, i+1, 0, m * n), di1, 1);
      matrix_add(m * n, &LIBXSMM_VLA_ACCESS(2, djdh, i+1, 0, m * n), di1, di2);
      matrix_eltwise_mult(m * n, zi, di2, &LIBXSMM_VLA_ACCESS(2, delta, i, 0, m * n));
    }
    if (pass == 1 || pass == 3) {
      /* matrix_transpose(m, k, w, wTp); - already taken care of in init */
      for (i = 0; i < t; ++i) {
        /* libxsmm_bgemm_omp(handlewd, wTp, &LIBXSMM_VLA_ACCESS(2, delta, i, 0, m * n), &LIBXSMM_VLA_ACCESS(2, djdx, i, 0, k * n), 1); */
        libxsmm_bgemm_omp(handlewd, w, &LIBXSMM_VLA_ACCESS(2, delta, i, 0, m * n), &LIBXSMM_VLA_ACCESS(2, djdx, i, 0, k * n), 1);
      }
    }
    if (pass == 2 || pass == 3) {
      for (i = 0; i < t; ++i) {
        /* matrix_transpose(m, n, &LIBXSMM_VLA_ACCESS(2, h, i, 0, m * n), hTp); - already taken care of in init */
        /* libxsmm_bgemm_omp(handledh, &LIBXSMM_VLA_ACCESS(2, delta, i, 0, m * n), hTp, dj1, 1); */
        libxsmm_bgemm_omp(handledh, &LIBXSMM_VLA_ACCESS(2, delta, i, 0, m * n), h, dj1, 1);
        matrix_add(m*m, dj1, djdu, djdu);
        /* matrix_transpose(k, n, &LIBXSMM_VLA_ACCESS(2, x, i, 0, k * n), xTp); - already taken care of in init */
        /* libxsmm_bgemm_omp(handledx, &LIBXSMM_VLA_ACCESS(2, delta, i, 0, m * n), xTp, dw1, 1); */
        libxsmm_bgemm_omp(handledx, &LIBXSMM_VLA_ACCESS(2, delta, i, 0, m * n), x, dw1, 1);
        matrix_add(m*k, dw1, djdw, djdw);
      }
    }
  }
  duration = libxsmm_timer_duration(start, libxsmm_timer_tick());
  if (0 < duration) {
    fprintf(stdout, "\tLIBXSMM: %.1f GFLOPS/s\n", gflops * nrepeat / duration);
  }
}


int rnn_bwd_upd(const libxsmm_blasint m, const libxsmm_blasint n, const libxsmm_blasint k, const libxsmm_blasint t,
         const libxsmm_blasint bm, const libxsmm_blasint bn, const libxsmm_blasint bk, const libxsmm_bgemm_order order, const int nrepeat,
         const libxsmm_blasint b_m1, const libxsmm_blasint b_n1, const libxsmm_blasint b_k1, const libxsmm_blasint b_m2, const libxsmm_blasint b_n2,
         const libxsmm_blasint b_k2, const libxsmm_blasint ldw, const libxsmm_blasint ldx, const libxsmm_blasint ldz, const libxsmm_blasint ldu,
         const libxsmm_blasint ldh, const libxsmm_blasint pass)
{
#if defined(CHECK)
  const char *const env_check = getenv("CHECK");
  const double check = LIBXSMM_ABS(0 == env_check ? 0 : atof(env_check));
#endif
  int result = EXIT_SUCCESS;
  /* TODO: Check the computation of gflops */
  const char transa = 'N', transb = 'N'; /* no transposes */
  const int gemm_flags = LIBXSMM_GEMM_FLAGS(transa, transb);
  const ITYPE alpha = 1, beta = 1, beta0 = 0;
  const libxsmm_blasint ldn = n;
  const double tflops = 12; /* transcendental flops */
  double gflops = m * m; /* U^T */
  gflops += (2.0 * m * n * m); /* U^T * delta */
  gflops += (m * n); /* dJdh + (U^T * delta) */
  gflops += (tflops * m * n); /* sigma'(Z) */
  gflops += (m * n); /* sigma'(Z) * (dJdh + (U^T * delta)) */
  gflops *= t; /* for t time steps */
  double tempflops;
  if (pass == 2 || pass == 3) {
    tempflops = m * n; /* h^T */
    tempflops += (2.0 * m * n * m); /* delta * h^T */
    tempflops *= t; /* for t time steps */
    tempflops += (m * m * (t-1)); /* for summation of dJdU */
    gflops += tempflops;
    tempflops = k * n; /* x^T */
    tempflops += (2.0 * m * n * k); /* delta * x^T */
    tempflops *= t; /* for t time steps */
    tempflops += (m * k * (t-1)); /* for summation of dJdW */
    gflops += tempflops;
  }
  if (pass == 1 || pass == 3) {
    tempflops = m * k; /* W^T */
    tempflops += (2.0 * m * n * k); /* W^T * delta */
    tempflops *= t; /* for t time steps of input */
    gflops += tempflops;
  }
  gflops *= 1E-9; /* to convert flops to Gflops */

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload target(LIBXSMM_OFFLOAD_TARGET)
#endif
  {
    ITYPE* djdhgoldt = (ITYPE*)libxsmm_malloc(ldh * n * sizeof(ITYPE) * t);
    ITYPE* zgoldt = (ITYPE*)libxsmm_malloc(ldz * n * sizeof(ITYPE) * t);
    ITYPE* deltagoldt = (ITYPE*)libxsmm_malloc(ldz * n * sizeof(ITYPE) * t);
    ITYPE* ugold = (ITYPE*)libxsmm_malloc(ldu * m * sizeof(ITYPE));
    ITYPE* xgoldt = (ITYPE*)libxsmm_malloc(ldx * n * sizeof(ITYPE) * t);
    ITYPE* hgoldt = (ITYPE*)libxsmm_malloc(ldh * n * sizeof(ITYPE) * t);
    ITYPE* djdugold = (ITYPE*)libxsmm_malloc(ldu * m * sizeof(ITYPE));
    ITYPE* djdwgold = (ITYPE*)libxsmm_malloc(ldw * k * sizeof(ITYPE));
    ITYPE* djdxgoldt = (ITYPE*)libxsmm_malloc(ldx * n * sizeof(ITYPE) * t);
    ITYPE* wgold = (ITYPE*)libxsmm_malloc(ldw * k * sizeof(ITYPE));
    ITYPE* zigold = (ITYPE*)libxsmm_malloc(ldz * n * sizeof(ITYPE));
    ITYPE* di1gold = (ITYPE*)libxsmm_malloc(ldz * n * sizeof(ITYPE));
    ITYPE* di2gold = (ITYPE*)libxsmm_malloc(ldz * n * sizeof(ITYPE));
    ITYPE* dj1gold = (ITYPE*)libxsmm_malloc(ldu * m * sizeof(ITYPE));
    ITYPE* dw1gold = (ITYPE*)libxsmm_malloc(ldw * k * sizeof(ITYPE));
    ITYPE* ugoldTp = (ITYPE*)libxsmm_malloc(ldu * m * sizeof(ITYPE));
    ITYPE* wgoldTp = (ITYPE*)libxsmm_malloc(ldw * k * sizeof(ITYPE));
    ITYPE* hgoldTp = (ITYPE*)libxsmm_malloc(ldh * n * sizeof(ITYPE));
    ITYPE* xgoldTp = (ITYPE*)libxsmm_malloc(ldx * n * sizeof(ITYPE));
    ITYPE* djdht = (ITYPE*)libxsmm_malloc(m * n * sizeof(ITYPE) * t);
    ITYPE* zt = (ITYPE*)libxsmm_malloc(m * n * sizeof(ITYPE) * t);
    ITYPE* deltat = (ITYPE*)libxsmm_malloc(m * n * sizeof(ITYPE) * t);
    ITYPE* u = (ITYPE*)libxsmm_malloc(m * m * sizeof(ITYPE));
    ITYPE* xt = (ITYPE*)libxsmm_malloc(k * n * sizeof(ITYPE) * t);
    ITYPE* ht = (ITYPE*)libxsmm_malloc(m * n * sizeof(ITYPE) * t);
    ITYPE* djdu = (ITYPE*)libxsmm_malloc(m * m * sizeof(ITYPE));
    ITYPE* djdw = (ITYPE*)libxsmm_malloc(m * k * sizeof(ITYPE));
    ITYPE* djdxt = (ITYPE*)libxsmm_malloc(k * n * sizeof(ITYPE) * t);
    ITYPE* w = (ITYPE*)libxsmm_malloc(m * k * sizeof(ITYPE));
    ITYPE* zi = (ITYPE*)libxsmm_malloc(m * n * sizeof(ITYPE));
    ITYPE* di1 = (ITYPE*)libxsmm_malloc(m * n * sizeof(ITYPE));
    ITYPE* di2 = (ITYPE*)libxsmm_malloc(m * n * sizeof(ITYPE));
    ITYPE* dj1 = (ITYPE*)libxsmm_malloc(m * m * sizeof(ITYPE));
    ITYPE* dw1 = (ITYPE*)libxsmm_malloc(m * k * sizeof(ITYPE));
    ITYPE* uTp = (ITYPE*)libxsmm_malloc(m * m * sizeof(ITYPE));
    ITYPE* wTp = (ITYPE*)libxsmm_malloc(m * k * sizeof(ITYPE));
    ITYPE* hTp = (ITYPE*)libxsmm_malloc(m * n * sizeof(ITYPE));
    ITYPE* xTp = (ITYPE*)libxsmm_malloc(k * n * sizeof(ITYPE));
    LIBXSMM_VLA_DECL(2, ITYPE, djdhgold, djdhgoldt, ldh * n);
    LIBXSMM_VLA_DECL(2, ITYPE, zgold, zgoldt, ldz * n);
    LIBXSMM_VLA_DECL(2, ITYPE, deltagold, deltagoldt, ldz * n);
    LIBXSMM_VLA_DECL(2, ITYPE, xgold, xgoldt, ldx * n);
    LIBXSMM_VLA_DECL(2, ITYPE, hgold, hgoldt, ldh * n);
    LIBXSMM_VLA_DECL(2, ITYPE, djdxgold, djdxgoldt, ldx * n);
    LIBXSMM_VLA_DECL(2, ITYPE, djdx, djdxt, k * n);
    libxsmm_bgemm_handle* handleud = 0;
    libxsmm_bgemm_handle* handledh = 0;
    libxsmm_bgemm_handle* handledx = 0;
    libxsmm_bgemm_handle* handlewd = 0;
    const libxsmm_gemm_prefetch_type strategy = LIBXSMM_PREFETCH_AUTO;
    handleud = libxsmm_bgemm_handle_create(LIBXSMM_GEMM_PRECISION(ITYPE), LIBXSMM_GEMM_PRECISION(ITYPE),
      m, n, m, &bm, &bn, &bm, &b_m1, &b_n1, &b_m1, &b_m2,
      &alpha, &beta, &gemm_flags, &strategy, &order); /* U^T*delta */
    handledh = libxsmm_bgemm_handle_create(LIBXSMM_GEMM_PRECISION(ITYPE), LIBXSMM_GEMM_PRECISION(ITYPE),
      m, m, n, &bm, &bm, &bn, &b_m1, &b_m1, &b_n1, &b_n2,
      &alpha, &beta, &gemm_flags, &strategy, &order); /* delta*h^T */
    handledx = libxsmm_bgemm_handle_create(LIBXSMM_GEMM_PRECISION(ITYPE), LIBXSMM_GEMM_PRECISION(ITYPE),
      m, k, n, &bm, &bk, &bn, &b_m1, &b_k1, &b_n1, &b_n2,
      &alpha, &beta, &gemm_flags, &strategy, &order); /* delta*x^T */
    handlewd = libxsmm_bgemm_handle_create(LIBXSMM_GEMM_PRECISION(ITYPE), LIBXSMM_GEMM_PRECISION(ITYPE),
      k, n, m, &bk, &bn, &bm, &b_k1, &b_n1, &b_m1, &b_m2,
      &alpha, &beta, &gemm_flags, &strategy, &order); /* W^T*delta */

    struct rnn_handle rnn;
    rnn.m = m;
    rnn.n = n;
    rnn.k = k;
    rnn.t = t;
    rnn.w = w;
    rnn.xt = xt;
    rnn.u = u;
    rnn.h = ht;
    rnn.z = zt;
    rnn.djdht = djdht;
    rnn.deltat = deltat;
    rnn.djdu = djdu;
    rnn.djdw = djdw;
    rnn.djdxt = djdxt;
    rnn.z1t = zi;
    rnn.di1 = di1;
    rnn.di2 = di2;
    rnn.dj1 = dj1;
    rnn.dw1 = dw1;
    rnn.uTp = uTp;
    rnn.wTp = wTp;
    rnn.hTp = hTp;
    rnn.xTp = xTp;
    rnn.handlewx = handleud;
    rnn.handleuh = handledh;
    rnn.handlett = handledx;
    rnn.handlewd = handlewd;

    if (0 != handleud && 0 != handledh && 0 != handledx && 0 != handlewd) {
      rnn_bwd_upd_init(&rnn, djdhgoldt, zgoldt, deltagoldt, ugold, xgoldt, hgoldt, wgold, djdugold, djdwgold, djdxgoldt, ldw, ldx, ldz, ldu, ldh);
      rnn_bwd_upd_execute(&rnn, nrepeat, pass);
#if defined(CHECK)
      unsigned long long start;
      double duration;
      int s;
      if (!LIBXSMM_FEQ(0, check)) { /* validate result against LAPACK/BLAS xGEMM */
        ITYPE* djtest = 0;
        int i;
        start = libxsmm_timer_tick();
        for (s = 0; s < nrepeat; ++s) {
          LIBXSMM_MATINIT(ITYPE, 0, &LIBXSMM_VLA_ACCESS(2, deltagold, t-1, 0, m * n), m, n, ldz, 0.0);
          matrix_transpose(m, m, ugold, ugoldTp);
          for (i = t-2; i >= 0; --i) {
            matrix_sigmoid_inverse(m * n, &LIBXSMM_VLA_ACCESS(2, zgold, i+1, 0, m * n), zigold);
            LIBXSMM_XBLAS_SYMBOL(ITYPE)(&transa, &transb, &m, &n, &m, &alpha, ugoldTp, &ldu, &LIBXSMM_VLA_ACCESS(2, deltagold, i+1, 0, m * n), &ldz, &beta0, di1gold, &ldz);
            matrix_add(m * n, &LIBXSMM_VLA_ACCESS(2, djdhgold, i+1, 0, m * n), di1gold, di2gold);
            matrix_eltwise_mult(m * n, zigold, di2gold, &LIBXSMM_VLA_ACCESS(2, deltagold, i, 0, m * n));
          }
          if (pass == 1 || pass == 3) {
            matrix_transpose(m, k, wgold, wgoldTp);
            for (i = 0; i < t; ++i) {
              LIBXSMM_XBLAS_SYMBOL(ITYPE)(&transa, &transb, &k, &n, &m, &alpha, wgoldTp, &ldx, &LIBXSMM_VLA_ACCESS(2, deltagold, i, 0, m * n), &ldz, &beta0, &LIBXSMM_VLA_ACCESS(2, djdxgold, i, 0, k * n), &ldx);
            }
          }
          if (pass == 2 || pass == 3) {
            for (i = 0; i < t; ++i) {
              matrix_transpose(m, n, &LIBXSMM_VLA_ACCESS(2, hgold, i, 0, m * n), hgoldTp);
              LIBXSMM_XBLAS_SYMBOL(ITYPE)(&transa, &transb, &m, &m, &n, &alpha, &LIBXSMM_VLA_ACCESS(2, deltagold, i, 0, m * n), &ldz, hgoldTp, &ldn, &beta0, dj1gold, &ldz);
              matrix_add(m*m, dj1gold, djdugold, djdugold);
              matrix_transpose(k, n, &LIBXSMM_VLA_ACCESS(2, xgold, i, 0, k * n), xgoldTp);
              LIBXSMM_XBLAS_SYMBOL(ITYPE)(&transa, &transb, &m, &k, &n, &alpha, &LIBXSMM_VLA_ACCESS(2, deltagold, i, 0, m * n), &ldz, xgoldTp, &ldn, &beta0, dw1gold, &ldw);
              matrix_add(m*k, dw1gold, djdwgold, djdwgold);
            }
          }
        }
        duration = libxsmm_timer_duration(start, libxsmm_timer_tick());
        if (0 < duration) {
          fprintf(stdout, "\tBLAS: %.1f GFLOPS/s\n", gflops * nrepeat / duration);
        }
        /* free memory not needed further; avoid double-free later on */
        libxsmm_free(djdhgoldt); djdhgoldt = 0;
        libxsmm_free(zgoldt); zgoldt = 0;
        libxsmm_free(deltagoldt); deltagoldt = 0;
        libxsmm_free(ugold); ugold = 0;
        libxsmm_free(xgoldt); xgoldt = 0;
        libxsmm_free(hgoldt); hgoldt = 0;
        libxsmm_free(wgold); wgold = 0;
        libxsmm_free(zigold); zigold = 0;
        libxsmm_free(di1gold); di1gold = 0;
        libxsmm_free(di2gold); di2gold = 0;
        libxsmm_free(dj1gold); dj1gold = 0;
        libxsmm_free(ugoldTp); ugoldTp = 0;
        libxsmm_free(wgoldTp); wgoldTp = 0;
        libxsmm_free(hgoldTp); hgoldTp = 0;
        libxsmm_free(xgoldTp); xgoldTp = 0;
        if (pass == 1 || pass == 3) {
          /* allocate C-matrix in regular format, and perform copy-out */
          djtest = (ITYPE*)libxsmm_malloc(ldx * n * sizeof(ITYPE));
          if (0 != djtest) {
            libxsmm_matdiff_info diff;
            for (i = 0; i < t; ++i) {
              libxsmm_bgemm_copyout_c(handlewd, &LIBXSMM_VLA_ACCESS(2, djdx, i, 0, k * n), &ldx, djtest);
              if (EXIT_SUCCESS == libxsmm_matdiff(LIBXSMM_DATATYPE(ITYPE), k, n, &LIBXSMM_VLA_ACCESS(2, djdxgold, i, 0, k * n), djtest, &ldx, &ldx, &diff)) {
                fprintf(stdout, "dJ/dX_%d::\tdiff: L2abs=%f L2rel=%f\n", i, diff.l2_abs, diff.linf_abs);
                if (check < 100.0 * diff.normf_rel) {
                  fprintf(stderr, "FAILED with an error of %f%%!\n", 100.0 * diff.normf_rel);
                  result = EXIT_FAILURE;
                }
              }
            }
          }
          libxsmm_free(djtest);
        }
        if (pass == 2 || pass == 3) {
          /* allocate C-matrix in regular format, and perform copy-out */
          djtest = (ITYPE*)libxsmm_malloc(ldu * m * sizeof(ITYPE));
          if (0 != djtest) {
            libxsmm_matdiff_info diff;
            libxsmm_bgemm_copyout_c(handledh, djdu, &ldu, djtest);
            if (EXIT_SUCCESS == libxsmm_matdiff(LIBXSMM_DATATYPE(ITYPE), m, m, djdugold, djtest, &ldu, &ldu, &diff)) {
              fprintf(stdout, "dJ/dU::\tdiff: L2abs=%f L2rel=%f\n", diff.l2_abs, diff.linf_abs);
              if (check < 100.0 * diff.normf_rel) {
                fprintf(stderr, "FAILED with an error of %f%%!\n", 100.0 * diff.normf_rel);
                result = EXIT_FAILURE;
              }
            }
          }
          libxsmm_free(djtest);
          /* allocate C-matrix in regular format, and perform copy-out */
          djtest = (ITYPE*)libxsmm_malloc(ldw * k * sizeof(ITYPE));
          if (0 != djtest) {
            libxsmm_matdiff_info diff;
            libxsmm_bgemm_copyout_c(handledx, djdw, &ldw, djtest);
            if (EXIT_SUCCESS == libxsmm_matdiff(LIBXSMM_DATATYPE(ITYPE), m, k, djdwgold, djtest, &ldw, &ldw, &diff)) {
              fprintf(stdout, "dJ/dW::\tdiff: L2abs=%f L2rel=%f\n", diff.l2_abs, diff.linf_abs);
              if (check < 100.0 * diff.normf_rel) {
                fprintf(stderr, "FAILED with an error of %f%%!\n", 100.0 * diff.normf_rel);
                result = EXIT_FAILURE;
              }
            }
          }
          libxsmm_free(djtest);
        }
      }
#endif
      libxsmm_bgemm_handle_destroy(handleud);
      libxsmm_bgemm_handle_destroy(handledh);
      libxsmm_bgemm_handle_destroy(handledx);
      libxsmm_bgemm_handle_destroy(handlewd);
    }
    else {
      fprintf(stderr, "FAILED to create BGEMM-handle! For details retry with LIBXSMM_VERBOSE=1.\n");
      result = EXIT_FAILURE;
    }
    libxsmm_free(djdht);
    libxsmm_free(zt);
    libxsmm_free(deltat);
    libxsmm_free(u);
    libxsmm_free(xt);
    libxsmm_free(ht);
    rnn_destroy(&rnn);
  }
  fprintf(stdout, "Finished\n");

  return result;
}


void lstm_init(struct lstm_handle *lstm, ITYPE *wigold, ITYPE *wfgold, ITYPE *wogold, ITYPE *wcgold,
  ITYPE *xgoldt, ITYPE *rigold, ITYPE *rfgold, ITYPE *rogold, ITYPE *rcgold, ITYPE *hgold,
  ITYPE *i1gold, ITYPE *i2gold, ITYPE *f1gold, ITYPE *f2gold, ITYPE *o1gold, ITYPE *o2gold,
  ITYPE *c1gold, ITYPE *c2gold, ITYPE *igold, ITYPE *fgold, ITYPE *ogold, ITYPE *cgold,
  ITYPE *dhgold, ITYPE *d1gold, ITYPE *d2gold, ITYPE *dgold,
  const libxsmm_blasint ldw, const libxsmm_blasint ldx, const libxsmm_blasint ldz,
  const libxsmm_blasint ldu, const libxsmm_blasint ldh, const libxsmm_blasint reuse)
{
#if defined(CHECK)
  const char *const env_check = getenv("CHECK");
  const double check = LIBXSMM_ABS(0 == env_check ? 0 : atof(env_check));
#endif
  const char transa = 'N', transb = 'N'; /* no transposes */
  const int gemm_flags = LIBXSMM_GEMM_FLAGS(transa, transb);
  const ITYPE alpha = 1, beta = 1;
  libxsmm_blasint m = lstm->m;
  libxsmm_blasint n = lstm->n;
  libxsmm_blasint k = lstm->k;
  libxsmm_blasint t = lstm->t;
  ITYPE *wi = (ITYPE*)lstm->wi;
  ITYPE *wf = (ITYPE*)lstm->wf;
  ITYPE *wo = (ITYPE*)lstm->wo;
  ITYPE *wc = (ITYPE*)lstm->wc;
  ITYPE *xt = (ITYPE*)lstm->xt;
  ITYPE *ri = (ITYPE*)lstm->ri;
  ITYPE *rf = (ITYPE*)lstm->rf;
  ITYPE *ro = (ITYPE*)lstm->ro;
  ITYPE *rc = (ITYPE*)lstm->rc;
  ITYPE *h = (ITYPE*)lstm->h;
  ITYPE *i1t = (ITYPE*)lstm->i1t;
  ITYPE *i2 = (ITYPE*)lstm->i2;
  ITYPE *f1t = (ITYPE*)lstm->f1t;
  ITYPE *f2 = (ITYPE*)lstm->f2;
  ITYPE *o1t = (ITYPE*)lstm->o1t;
  ITYPE *o2 = (ITYPE*)lstm->o2;
  ITYPE *c1t = (ITYPE*)lstm->c1t;
  ITYPE *c2 = (ITYPE*)lstm->c2;
  ITYPE *i = (ITYPE*)lstm->i;
  ITYPE *f = (ITYPE*)lstm->f;
  ITYPE *o = (ITYPE*)lstm->o;
  ITYPE *c = (ITYPE*)lstm->c;
  ITYPE *dh = (ITYPE*)lstm->dh;
  ITYPE *d1 = (ITYPE*)lstm->d1;
  ITYPE *d2 = (ITYPE*)lstm->d2;
  ITYPE *d = (ITYPE*)lstm->d;
  LIBXSMM_VLA_DECL(2, ITYPE, xgold, xgoldt, ldx * n);
  LIBXSMM_VLA_DECL(2, ITYPE, x, xt, k * n);
#if defined(NON_FUSED_INPUT_GEMM)
  LIBXSMM_VLA_DECL(2, ITYPE, i1, i1t, m * n);
  LIBXSMM_VLA_DECL(2, ITYPE, f1, f1t, m * n);
  LIBXSMM_VLA_DECL(2, ITYPE, o1, o1t, m * n);
  LIBXSMM_VLA_DECL(2, ITYPE, c1, c1t, m * n);
#else
  LIBXSMM_VLA_DECL(3, ITYPE, i1, i1t, t, m * n);
#endif
  libxsmm_bgemm_handle *handlewx = lstm->handlewx;
  libxsmm_bgemm_handle *handleuh = lstm->handleuh;
  LIBXSMM_VLA_DECL(2, ITYPE, hnr, h, m * n);
  LIBXSMM_VLA_DECL(2, ITYPE, inr, i, m * n);
  LIBXSMM_VLA_DECL(2, ITYPE, fnr, f, m * n);
  LIBXSMM_VLA_DECL(2, ITYPE, onr, o, m * n);
  LIBXSMM_VLA_DECL(2, ITYPE, cnr, c, m * n);
  LIBXSMM_VLA_DECL(2, ITYPE, dnr, d, m * n);

  LIBXSMM_MATINIT(ITYPE, 42, wigold, m, k, ldw, 1.0);
  LIBXSMM_MATINIT(ITYPE, 42, wfgold, m, k, ldw, 1.0);
  LIBXSMM_MATINIT(ITYPE, 42, wogold, m, k, ldw, 1.0);
  LIBXSMM_MATINIT(ITYPE, 42, wcgold, m, k, ldw, 1.0);
  int it;
  for (it = 0; it < t; ++it) {
    LIBXSMM_MATINIT(ITYPE, 24, &LIBXSMM_VLA_ACCESS(2, xgold, it, 0, ldx * n), k, n, ldx, 1.0);
  }
  LIBXSMM_MATINIT(ITYPE, 42, rigold, m, m, ldu, 1.0);
  LIBXSMM_MATINIT(ITYPE, 42, rfgold, m, m, ldu, 1.0);
  LIBXSMM_MATINIT(ITYPE, 42, rogold, m, m, ldu, 1.0);
  LIBXSMM_MATINIT(ITYPE, 42, rcgold, m, m, ldu, 1.0);
  LIBXSMM_MATINIT(ITYPE, 24, hgold, m, n, ldh, 1.0);
  LIBXSMM_MATINIT(ITYPE,  0, dhgold, m, n, ldh, 1.0);
  LIBXSMM_MATINIT(ITYPE,  0, i1gold, m, n, ldz, 1.0);
  LIBXSMM_MATINIT(ITYPE,  0, i2gold, m, n, ldz, 1.0);
  LIBXSMM_MATINIT(ITYPE,  0, f1gold, m, n, ldz, 1.0);
  LIBXSMM_MATINIT(ITYPE,  0, f2gold, m, n, ldz, 1.0);
  LIBXSMM_MATINIT(ITYPE,  0, o1gold, m, n, ldz, 1.0);
  LIBXSMM_MATINIT(ITYPE,  0, o2gold, m, n, ldz, 1.0);
  LIBXSMM_MATINIT(ITYPE,  0, c1gold, m, n, ldz, 1.0);
  LIBXSMM_MATINIT(ITYPE,  0, c2gold, m, n, ldz, 1.0);
  LIBXSMM_MATINIT(ITYPE,  0, igold, m, n, ldz, 1.0);
  LIBXSMM_MATINIT(ITYPE,  0, fgold, m, n, ldz, 1.0);
  LIBXSMM_MATINIT(ITYPE,  0, ogold, m, n, ldz, 1.0);
  LIBXSMM_MATINIT(ITYPE,  0, cgold, m, n, ldz, 1.0);
  LIBXSMM_MATINIT(ITYPE,  0, d1gold, m, n, ldz, 1.0);
  LIBXSMM_MATINIT(ITYPE,  0, d2gold, m, n, ldz, 1.0);
  LIBXSMM_MATINIT(ITYPE, 24, dgold, m, n, ldz, 1.0);
#if defined(NON_FUSED_INPUT_GEMM)
  libxsmm_bgemm_copyin_a(handlewx, wigold, &ldw, wi);
  libxsmm_bgemm_copyin_a(handlewx, wfgold, &ldw, wf);
  libxsmm_bgemm_copyin_a(handlewx, wogold, &ldw, wo);
  libxsmm_bgemm_copyin_a(handlewx, wcgold, &ldw, wc);
#else
  LIBXSMM_VLA_DECL(2, ITYPE, wi4, wi, m * k);
  libxsmm_bgemm_copyin_a(handlewx, wigold, &ldw, &LIBXSMM_VLA_ACCESS(2, wi4, 0, 0, m * k));
  libxsmm_bgemm_copyin_a(handlewx, wfgold, &ldw, &LIBXSMM_VLA_ACCESS(2, wi4, 1, 0, m * k));
  libxsmm_bgemm_copyin_a(handlewx, wogold, &ldw, &LIBXSMM_VLA_ACCESS(2, wi4, 2, 0, m * k));
  libxsmm_bgemm_copyin_a(handlewx, wcgold, &ldw, &LIBXSMM_VLA_ACCESS(2, wi4, 3, 0, m * k));
#endif
  for (it = 0; it < t; ++it) {
    libxsmm_bgemm_copyin_b(handlewx, &LIBXSMM_VLA_ACCESS(2, xgold, it, 0, ldx * n), &ldx, &LIBXSMM_VLA_ACCESS(2, x, it, 0, k * n));
  }
  libxsmm_bgemm_copyin_a(handleuh, rigold, &ldu, ri);
  libxsmm_bgemm_copyin_a(handleuh, rfgold, &ldu, rf);
  libxsmm_bgemm_copyin_a(handleuh, rogold, &ldu, ro);
  libxsmm_bgemm_copyin_a(handleuh, rcgold, &ldu, rc);
  if (reuse) {
    libxsmm_bgemm_copyin_b(handleuh, hgold, &ldh, h);
  } else {
    for (it = 0; it < t; ++it) {
      libxsmm_bgemm_copyin_b(handleuh, hgold, &ldh, &LIBXSMM_VLA_ACCESS(2, hnr, it, 0, m * n));
    }
  }
#if defined(NON_FUSED_INPUT_GEMM)
  for (it = 0; it < t; ++it) {
    libxsmm_bgemm_copyin_c(handlewx, i1gold, &ldz, &LIBXSMM_VLA_ACCESS(2, i1, it, 0, m * n));
    libxsmm_bgemm_copyin_c(handlewx, f1gold, &ldz, &LIBXSMM_VLA_ACCESS(2, f1, it, 0, m * n));
    libxsmm_bgemm_copyin_c(handlewx, o1gold, &ldz, &LIBXSMM_VLA_ACCESS(2, o1, it, 0, m * n));
    libxsmm_bgemm_copyin_c(handlewx, c1gold, &ldz, &LIBXSMM_VLA_ACCESS(2, c1, it, 0, m * n));
  }
#else
  for (it = 0; it < t; ++it) {
    libxsmm_bgemm_copyin_c(handlewx, i1gold, &ldz, &LIBXSMM_VLA_ACCESS(3, i1, 0, it, 0, t, m * n));
  }
  for (it = 0; it < t; ++it) {
    libxsmm_bgemm_copyin_c(handlewx, f1gold, &ldz, &LIBXSMM_VLA_ACCESS(3, i1, 1, it, 0, t, m * n));
  }
  for (it = 0; it < t; ++it) {
    libxsmm_bgemm_copyin_c(handlewx, o1gold, &ldz, &LIBXSMM_VLA_ACCESS(3, i1, 2, it, 0, t, m * n));
  }
  for (it = 0; it < t; ++it) {
    libxsmm_bgemm_copyin_c(handlewx, c1gold, &ldz, &LIBXSMM_VLA_ACCESS(3, i1, 3, it, 0, t, m * n));
  }
#endif
  if (reuse) {
    libxsmm_bgemm_copyin_c(handleuh, igold, &ldh, i);
    libxsmm_bgemm_copyin_c(handleuh, fgold, &ldh, f);
    libxsmm_bgemm_copyin_c(handleuh, ogold, &ldh, o);
    libxsmm_bgemm_copyin_c(handleuh, cgold, &ldh, c);
    libxsmm_bgemm_copyin_c(handleuh, dgold, &ldh, d);
  } else {
    for (it = 0; it < t; ++it) {
      libxsmm_bgemm_copyin_c(handleuh, igold, &ldh, &LIBXSMM_VLA_ACCESS(2, inr, it, 0, m * n));
      libxsmm_bgemm_copyin_c(handleuh, fgold, &ldh, &LIBXSMM_VLA_ACCESS(2, fnr, it, 0, m * n));
      libxsmm_bgemm_copyin_c(handleuh, ogold, &ldh, &LIBXSMM_VLA_ACCESS(2, onr, it, 0, m * n));
      libxsmm_bgemm_copyin_c(handleuh, cgold, &ldh, &LIBXSMM_VLA_ACCESS(2, cnr, it, 0, m * n));
      libxsmm_bgemm_copyin_c(handleuh, dgold, &ldh, &LIBXSMM_VLA_ACCESS(2, dnr, it, 0, m * n));
    }
  }
  libxsmm_bgemm_copyin_c(handleuh, d1gold, &ldh, d1);
  libxsmm_bgemm_copyin_c(handleuh, d2gold, &ldh, d2);
  libxsmm_bgemm_copyin_c(handleuh, dhgold, &ldh, dh);
#if defined(MKL_ENABLE_AVX512)
  mkl_enable_instructions(MKL_ENABLE_AVX512);
#endif
  /* warmup OpenMP (populate thread pool) */
#if defined(NON_FUSED_INPUT_GEMM)
  libxsmm_bgemm_omp(handlewx, wi, x, &LIBXSMM_VLA_ACCESS(2, i1, 0, 0, m * n), 1);
  libxsmm_bgemm_omp(handlewx, wf, x, &LIBXSMM_VLA_ACCESS(2, f1, 0, 0, m * n), 1);
  libxsmm_bgemm_omp(handlewx, wo, x, &LIBXSMM_VLA_ACCESS(2, o1, 0, 0, m * n), 1);
  libxsmm_bgemm_omp(handlewx, wc, x, &LIBXSMM_VLA_ACCESS(2, c1, 0, 0, m * n), 1);
#else
  libxsmm_bgemm_omp(handlewx, wi, x, &LIBXSMM_VLA_ACCESS(3, i1, 0, 0, 0, t, m * n), 1);
#endif
#if defined(CHECK)
  if (!LIBXSMM_FEQ(0, check)) {
    LIBXSMM_XBLAS_SYMBOL(ITYPE)(&transa, &transb, &m, &n, &k, &alpha, wigold, &ldw, &LIBXSMM_VLA_ACCESS(2, xgold, 0, 0, ldx * n), &ldx, &beta, i1gold, &ldz);
    LIBXSMM_XBLAS_SYMBOL(ITYPE)(&transa, &transb, &m, &n, &k, &alpha, wfgold, &ldw, &LIBXSMM_VLA_ACCESS(2, xgold, 0, 0, ldx * n), &ldx, &beta, f1gold, &ldz);
    LIBXSMM_XBLAS_SYMBOL(ITYPE)(&transa, &transb, &m, &n, &k, &alpha, wogold, &ldw, &LIBXSMM_VLA_ACCESS(2, xgold, 0, 0, ldx * n), &ldx, &beta, o1gold, &ldz);
    LIBXSMM_XBLAS_SYMBOL(ITYPE)(&transa, &transb, &m, &n, &k, &alpha, wcgold, &ldw, &LIBXSMM_VLA_ACCESS(2, xgold, 0, 0, ldx * n), &ldx, &beta, c1gold, &ldz);
  }
#endif
  libxsmm_gemm_print(stdout, LIBXSMM_GEMM_PRECISION(ITYPE),
    &transa, &transb, &m, &n, &k, &alpha, wi, &ldw, x, &ldx, &beta, &LIBXSMM_VLA_ACCESS(2, i1, 0, 0, m * n), &ldz);
  fprintf(stdout, "\n\n");
  /* warmup OpenMP (populate thread pool) */
  libxsmm_bgemm_omp(handleuh, ri, h, i2, 1);
  libxsmm_bgemm_omp(handleuh, rf, h, f2, 1);
  libxsmm_bgemm_omp(handleuh, ro, h, o2, 1);
  libxsmm_bgemm_omp(handleuh, rc, h, c2, 1);
#if defined(CHECK)
  if (!LIBXSMM_FEQ(0, check)) {
    LIBXSMM_XBLAS_SYMBOL(ITYPE)(&transa, &transb, &m, &n, &m, &alpha, rigold, &ldu, hgold, &ldh, &beta, i2gold, &ldz);
    LIBXSMM_XBLAS_SYMBOL(ITYPE)(&transa, &transb, &m, &n, &m, &alpha, rfgold, &ldu, hgold, &ldh, &beta, f2gold, &ldz);
    LIBXSMM_XBLAS_SYMBOL(ITYPE)(&transa, &transb, &m, &n, &m, &alpha, rogold, &ldu, hgold, &ldh, &beta, o2gold, &ldz);
    LIBXSMM_XBLAS_SYMBOL(ITYPE)(&transa, &transb, &m, &n, &m, &alpha, rcgold, &ldu, hgold, &ldh, &beta, c2gold, &ldz);
  }
#endif
  libxsmm_gemm_print(stdout, LIBXSMM_GEMM_PRECISION(ITYPE),
    &transa, &transb, &m, &n, &m, &alpha, ri, &ldu, h, &ldh, &beta, i2, &ldz);
  fprintf(stdout, "\n\n");
}


void lstm_execute(struct lstm_handle *lstm, const int nrepeat, const libxsmm_blasint reuse)
{
  const char transa = 'N', transb = 'N'; /* no transposes */
  const int gemm_flags = LIBXSMM_GEMM_FLAGS(transa, transb);
  const ITYPE alpha = 1, beta = 1;
  libxsmm_blasint m = lstm->m;
  libxsmm_blasint n = lstm->n;
  libxsmm_blasint k = lstm->k;
  libxsmm_blasint t = lstm->t;
  const double gflops = (((2.0 * m * n * k) + (2.0 * m * n * m) + (2.0 * m * n)) * 4.0 + (5.0 * m * n)) * t * 1E-9;
  ITYPE *wi = (ITYPE*)lstm->wi;
  ITYPE *wf = (ITYPE*)lstm->wf;
  ITYPE *wo = (ITYPE*)lstm->wo;
  ITYPE *wc = (ITYPE*)lstm->wc;
  ITYPE *xt = (ITYPE*)lstm->xt;
  ITYPE *ri = (ITYPE*)lstm->ri;
  ITYPE *rf = (ITYPE*)lstm->rf;
  ITYPE *ro = (ITYPE*)lstm->ro;
  ITYPE *rc = (ITYPE*)lstm->rc;
  ITYPE *h = (ITYPE*)lstm->h;
  ITYPE *i1t = (ITYPE*)lstm->i1t;
  ITYPE *i2 = (ITYPE*)lstm->i2;
  ITYPE *f1t = (ITYPE*)lstm->f1t;
  ITYPE *f2 = (ITYPE*)lstm->f2;
  ITYPE *o1t = (ITYPE*)lstm->o1t;
  ITYPE *o2 = (ITYPE*)lstm->o2;
  ITYPE *c1t = (ITYPE*)lstm->c1t;
  ITYPE *c2 = (ITYPE*)lstm->c2;
  ITYPE *i = (ITYPE*)lstm->i;
  ITYPE *f = (ITYPE*)lstm->f;
  ITYPE *o = (ITYPE*)lstm->o;
  ITYPE *c = (ITYPE*)lstm->c;
  ITYPE *dh = (ITYPE*)lstm->dh;
  ITYPE *d1 = (ITYPE*)lstm->d1;
  ITYPE *d2 = (ITYPE*)lstm->d2;
  ITYPE *d = (ITYPE*)lstm->d;
  LIBXSMM_VLA_DECL(2, ITYPE, x, xt, k * n);
#if defined(NON_FUSED_INPUT_GEMM)
  LIBXSMM_VLA_DECL(2, ITYPE, i1, i1t, m * n);
  LIBXSMM_VLA_DECL(2, ITYPE, f1, f1t, m * n);
  LIBXSMM_VLA_DECL(2, ITYPE, o1, o1t, m * n);
  LIBXSMM_VLA_DECL(2, ITYPE, c1, c1t, m * n);
#else
  LIBXSMM_VLA_DECL(3, ITYPE, i4, i1t, t, m * n);
  i1t = &LIBXSMM_VLA_ACCESS(3, i4, 0, 0, 0, t, m * n);
  f1t = &LIBXSMM_VLA_ACCESS(3, i4, 1, 0, 0, t, m * n);
  o1t = &LIBXSMM_VLA_ACCESS(3, i4, 2, 0, 0, t, m * n);
  c1t = &LIBXSMM_VLA_ACCESS(3, i4, 3, 0, 0, t, m * n);
  LIBXSMM_VLA_DECL(2, ITYPE, i1, i1t, m * n);
  LIBXSMM_VLA_DECL(2, ITYPE, f1, f1t, m * n);
  LIBXSMM_VLA_DECL(2, ITYPE, o1, o1t, m * n);
  LIBXSMM_VLA_DECL(2, ITYPE, c1, c1t, m * n);
#endif
  libxsmm_bgemm_handle *handlewx = lstm->handlewx;
  libxsmm_bgemm_handle *handleuh = lstm->handleuh;
  libxsmm_bgemm_handle *handlett = lstm->handlett;
  LIBXSMM_VLA_DECL(2, ITYPE, hnr, h, m * n);
  LIBXSMM_VLA_DECL(2, ITYPE, inr, i, m * n);
  LIBXSMM_VLA_DECL(2, ITYPE, fnr, f, m * n);
  LIBXSMM_VLA_DECL(2, ITYPE, onr, o, m * n);
  LIBXSMM_VLA_DECL(2, ITYPE, cnr, c, m * n);
  LIBXSMM_VLA_DECL(2, ITYPE, dnr, d, m * n);
  unsigned long long start;
  double duration;
#if defined(LSTM_TIMING)
  Gbl_t_input_total = 0.; Gbl_t_recur_total = 0.; Gbl_t_eltwise_total = 0.; Gbl_t_nonlin_total = 0.;
  Gbl_t_input = 0; Gbl_t_recur = 0; Gbl_t_eltwise = 0; Gbl_t_nonlin = 0;
  Gbl_duration_input = 0.; Gbl_duration_recur = 0.; Gbl_duration_eltwise = 0.; Gbl_duration_nonlin = 0.;
#endif

  int s;
  int j;
  libxsmm_blasint nt = n*t;
  start = libxsmm_timer_tick();
  if (reuse) {
    for (s = 0; s < nrepeat; ++s) {
#if defined(LSTM_TIMING)
      Gbl_t_input = libxsmm_timer_tick();
#endif
      /* The following loop may be absorbed into libxsmm_lstm_omp */
#if defined(NON_FUSED_INPUT_GEMM)
      libxsmm_bgemm_omp(handlett, wi, &LIBXSMM_VLA_ACCESS(2, x, 0, 0, k * n), &LIBXSMM_VLA_ACCESS(2, i1, 0, 0, m * n), 1/*nrepeat*/);
      libxsmm_bgemm_omp(handlett, wf, &LIBXSMM_VLA_ACCESS(2, x, 0, 0, k * n), &LIBXSMM_VLA_ACCESS(2, f1, 0, 0, m * n), 1/*nrepeat*/);
      libxsmm_bgemm_omp(handlett, wo, &LIBXSMM_VLA_ACCESS(2, x, 0, 0, k * n), &LIBXSMM_VLA_ACCESS(2, o1, 0, 0, m * n), 1/*nrepeat*/);
      libxsmm_bgemm_omp(handlett, wc, &LIBXSMM_VLA_ACCESS(2, x, 0, 0, k * n), &LIBXSMM_VLA_ACCESS(2, c1, 0, 0, m * n), 1/*nrepeat*/);
#else
      libxsmm_bgemm_omp(handlett, wi, &LIBXSMM_VLA_ACCESS(2, x, 0, 0, k * n), &LIBXSMM_VLA_ACCESS(3, i4, 0, 0, 0, t, m * n), 1/*nrepeat*/);
#endif
#if defined(LSTM_TIMING)
      Gbl_duration_input = libxsmm_timer_duration(Gbl_t_input, libxsmm_timer_tick());
      Gbl_t_input_total += Gbl_duration_input;
#endif
      for (j = 0; j < t-1; ++j) {
        recursive_step(handleuh, ri, h, i2, &LIBXSMM_VLA_ACCESS(2, i1, j, 0, m * n), i, i, 1, m * n); /*sigmoid*/
        recursive_step(handleuh, rf, h, f2, &LIBXSMM_VLA_ACCESS(2, f1, j, 0, m * n), f, f, 1, m * n); /*sigmoid*/
        recursive_step(handleuh, ro, h, o2, &LIBXSMM_VLA_ACCESS(2, o1, j, 0, m * n), o, o, 1, m * n); /*sigmoid*/
        recursive_step(handleuh, rc, h, c2, &LIBXSMM_VLA_ACCESS(2, c1, j, 0, m * n), c, c, 1, m * n); /*tanh*/
#if defined(LSTM_TIMING)
        Gbl_t_eltwise = libxsmm_timer_tick();
#endif
        matrix_eltwise_mult(m*n, f, d, d1);
        matrix_eltwise_mult(m*n, i, c, d2);
        matrix_add(m*n, d1, d2, d);
#if defined(LSTM_TIMING)
        Gbl_duration_eltwise = libxsmm_timer_duration(Gbl_t_eltwise, libxsmm_timer_tick());
        Gbl_t_eltwise_total += Gbl_duration_eltwise;
        Gbl_t_nonlin = libxsmm_timer_tick();
#endif
        matrix_relu(m*n, d, dh); /*tanh*/
#if defined(LSTM_TIMING)
        Gbl_duration_nonlin = libxsmm_timer_duration(Gbl_t_nonlin, libxsmm_timer_tick());
        Gbl_t_nonlin_total += Gbl_duration_nonlin;
        Gbl_t_eltwise = libxsmm_timer_tick();
#endif
        matrix_eltwise_mult(m*n, o, dh, h);
#if defined(LSTM_TIMING)
        Gbl_duration_eltwise = libxsmm_timer_duration(Gbl_t_eltwise, libxsmm_timer_tick());
        Gbl_t_eltwise_total += Gbl_duration_eltwise;
#endif
      }
      recursive_step(handleuh, ri, h, i2, &LIBXSMM_VLA_ACCESS(2, i1, t-1, 0, m * n), i, i, 1, m * n); /*sigmoid*/
      recursive_step(handleuh, rf, h, f2, &LIBXSMM_VLA_ACCESS(2, f1, t-1, 0, m * n), f, f, 1, m * n); /*sigmoid*/
      recursive_step(handleuh, ro, h, o2, &LIBXSMM_VLA_ACCESS(2, o1, t-1, 0, m * n), o, o, 1, m * n); /*sigmoid*/
      recursive_step(handleuh, rc, h, c2, &LIBXSMM_VLA_ACCESS(2, c1, t-1, 0, m * n), c, c, 1, m * n); /*tanh*/
#if defined(LSTM_TIMING)
      Gbl_t_eltwise = libxsmm_timer_tick();
#endif
      matrix_eltwise_mult(m*n, f, d, d1);
      matrix_eltwise_mult(m*n, i, c, d2);
      matrix_add(m*n, d1, d2, d);
#if defined(LSTM_TIMING)
      Gbl_duration_eltwise = libxsmm_timer_duration(Gbl_t_eltwise, libxsmm_timer_tick());
      Gbl_t_eltwise_total += Gbl_duration_eltwise;
#endif
    }/* end for nrepeat */
  } else {
    for (s = 0; s < nrepeat; ++s) {
#if defined(LSTM_TIMING)
      Gbl_t_input = libxsmm_timer_tick();
#endif
      /* The following loop may be absorbed into libxsmm_lstm_omp */
#if defined(NON_FUSED_INPUT_GEMM)
      libxsmm_bgemm_omp(handlett, wi, &LIBXSMM_VLA_ACCESS(2, x, 0, 0, k * n), &LIBXSMM_VLA_ACCESS(2, i1, 0, 0, m * n), 1/*nrepeat*/);
      libxsmm_bgemm_omp(handlett, wf, &LIBXSMM_VLA_ACCESS(2, x, 0, 0, k * n), &LIBXSMM_VLA_ACCESS(2, f1, 0, 0, m * n), 1/*nrepeat*/);
      libxsmm_bgemm_omp(handlett, wo, &LIBXSMM_VLA_ACCESS(2, x, 0, 0, k * n), &LIBXSMM_VLA_ACCESS(2, o1, 0, 0, m * n), 1/*nrepeat*/);
      libxsmm_bgemm_omp(handlett, wc, &LIBXSMM_VLA_ACCESS(2, x, 0, 0, k * n), &LIBXSMM_VLA_ACCESS(2, c1, 0, 0, m * n), 1/*nrepeat*/);
#else
      libxsmm_bgemm_omp(handlett, wi, &LIBXSMM_VLA_ACCESS(2, x, 0, 0, k * n), &LIBXSMM_VLA_ACCESS(3, i4, 0, 0, 0, t, m * n), 1/*nrepeat*/);
#endif
#if defined(LSTM_TIMING)
      Gbl_duration_input = libxsmm_timer_duration(Gbl_t_input, libxsmm_timer_tick());
      Gbl_t_input_total += Gbl_duration_input;
#endif
      for (j = 0; j < t-1; ++j) {
        recursive_step(handleuh, ri, &LIBXSMM_VLA_ACCESS(2, hnr, j, 0, m * n), i2, &LIBXSMM_VLA_ACCESS(2, i1, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, inr, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, inr, j, 0, m * n), 1, m * n); /*sigmoid*/
        recursive_step(handleuh, rf, &LIBXSMM_VLA_ACCESS(2, hnr, j, 0, m * n), f2, &LIBXSMM_VLA_ACCESS(2, f1, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, fnr, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, fnr, j, 0, m * n), 1, m * n); /*sigmoid*/
        recursive_step(handleuh, ro, &LIBXSMM_VLA_ACCESS(2, hnr, j, 0, m * n), o2, &LIBXSMM_VLA_ACCESS(2, o1, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, onr, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, onr, j, 0, m * n), 1, m * n); /*sigmoid*/
        recursive_step(handleuh, rc, &LIBXSMM_VLA_ACCESS(2, hnr, j, 0, m * n), c2, &LIBXSMM_VLA_ACCESS(2, c1, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, cnr, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, cnr, j, 0, m * n), 1, m * n); /*tanh*/
#if defined(LSTM_TIMING)
        Gbl_t_eltwise = libxsmm_timer_tick();
#endif
        matrix_eltwise_mult(m*n, &LIBXSMM_VLA_ACCESS(2, fnr, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, dnr, j, 0, m * n), d1);
        matrix_eltwise_mult(m*n, &LIBXSMM_VLA_ACCESS(2, inr, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, cnr, j, 0, m * n), d2);
        matrix_add(m*n, d1, d2, &LIBXSMM_VLA_ACCESS(2, dnr, j+1, 0, m * n));
#if defined(LSTM_TIMING)
        Gbl_duration_eltwise = libxsmm_timer_duration(Gbl_t_eltwise, libxsmm_timer_tick());
        Gbl_t_eltwise_total += Gbl_duration_eltwise;
        Gbl_t_nonlin = libxsmm_timer_tick();
#endif
        matrix_relu(m*n, &LIBXSMM_VLA_ACCESS(2, dnr, j+1, 0, m * n), dh); /*tanh*/
#if defined(LSTM_TIMING)
        Gbl_duration_nonlin = libxsmm_timer_duration(Gbl_t_nonlin, libxsmm_timer_tick());
        Gbl_t_nonlin_total += Gbl_duration_nonlin;
        Gbl_t_eltwise = libxsmm_timer_tick();
#endif
        matrix_eltwise_mult(m*n, &LIBXSMM_VLA_ACCESS(2, onr, j, 0, m * n), dh, &LIBXSMM_VLA_ACCESS(2, hnr, j+1, 0, m * n));
#if defined(LSTM_TIMING)
        Gbl_duration_eltwise = libxsmm_timer_duration(Gbl_t_eltwise, libxsmm_timer_tick());
        Gbl_t_eltwise_total += Gbl_duration_eltwise;
#endif
      }
      recursive_step(handleuh, ri, &LIBXSMM_VLA_ACCESS(2, hnr, t-1, 0, m * n), i2, &LIBXSMM_VLA_ACCESS(2, i1, t-1, 0, m * n), &LIBXSMM_VLA_ACCESS(2, inr, t-2, 0, m * n), &LIBXSMM_VLA_ACCESS(2, inr, t-2, 0, m * n), 1, m * n); /*sigmoid*/
      recursive_step(handleuh, rf, &LIBXSMM_VLA_ACCESS(2, hnr, t-1, 0, m * n), f2, &LIBXSMM_VLA_ACCESS(2, f1, t-1, 0, m * n), &LIBXSMM_VLA_ACCESS(2, fnr, t-2, 0, m * n), &LIBXSMM_VLA_ACCESS(2, fnr, t-2, 0, m * n), 1, m * n); /*sigmoid*/
      recursive_step(handleuh, ro, &LIBXSMM_VLA_ACCESS(2, hnr, t-1, 0, m * n), o2, &LIBXSMM_VLA_ACCESS(2, o1, t-1, 0, m * n), &LIBXSMM_VLA_ACCESS(2, onr, t-2, 0, m * n), &LIBXSMM_VLA_ACCESS(2, onr, t-2, 0, m * n), 1, m * n); /*sigmoid*/
      recursive_step(handleuh, rc, &LIBXSMM_VLA_ACCESS(2, hnr, t-1, 0, m * n), c2, &LIBXSMM_VLA_ACCESS(2, c1, t-1, 0, m * n), &LIBXSMM_VLA_ACCESS(2, cnr, t-2, 0, m * n), &LIBXSMM_VLA_ACCESS(2, cnr, t-2, 0, m * n), 1, m * n); /*tanh*/
#if defined(LSTM_TIMING)
      Gbl_t_eltwise = libxsmm_timer_tick();
#endif
      matrix_eltwise_mult(m*n, &LIBXSMM_VLA_ACCESS(2, fnr, t-2, 0, m * n), &LIBXSMM_VLA_ACCESS(2, dnr, t-1, 0, m * n), d1);
      matrix_eltwise_mult(m*n, &LIBXSMM_VLA_ACCESS(2, inr, t-2, 0, m * n), &LIBXSMM_VLA_ACCESS(2, cnr, t-2, 0, m * n), d2);
      matrix_add(m*n, d1, d2, &LIBXSMM_VLA_ACCESS(2, dnr, t-1, 0, m * n));
#if defined(LSTM_TIMING)
      Gbl_duration_eltwise = libxsmm_timer_duration(Gbl_t_eltwise, libxsmm_timer_tick());
      Gbl_t_eltwise_total += Gbl_duration_eltwise;
#endif
    }
  }
  duration = libxsmm_timer_duration(start, libxsmm_timer_tick());
  if (0 < duration) {
    fprintf(stdout, "\tLIBXSMM: %.1f GFLOPS/s\n", gflops * nrepeat / duration);
  }
#if defined(LSTM_TIMING)
  double t_total = Gbl_t_input_total + Gbl_t_recur_total + Gbl_t_eltwise_total + Gbl_t_nonlin_total;
  fprintf(stdout, "Percentage of time spent in input matrix multiplication: %lf\n", Gbl_t_input_total*100.0/t_total);
  fprintf(stdout, "Percentage of time spent in recurrence matrix multiplication: %lf\n", Gbl_t_recur_total*100.0/t_total);
  fprintf(stdout, "Percentage of time spent in element-wise operations: %lf\n", Gbl_t_eltwise_total*100.0/t_total);
  fprintf(stdout, "Percentage of time spent in non-linear operations: %lf\n", Gbl_t_nonlin_total*100.0/t_total);
#endif
}


void lstm_destroy(struct lstm_handle *lstm)
{
#if 0
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( lstm->wi ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( lstm->wf ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( lstm->wo ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( lstm->wc ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( lstm->xt ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( lstm->ri ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( lstm->rf ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( lstm->ro ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( lstm->rc ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( lstm->h ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( lstm->i1t ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( lstm->i2 ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( lstm->f1t ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( lstm->f2 ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( lstm->o1t ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( lstm->o2 ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( lstm->c1t ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( lstm->c2 ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( lstm->i ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( lstm->f ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( lstm->o ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( lstm->c ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( lstm->dh ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( lstm->d1 ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( lstm->d2 ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( lstm->d ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( lstm->i3 ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( lstm->f3 ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( lstm->d4 ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( lstm->djdht ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( lstm->deltat ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( lstm->djddt ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( lstm->djdit ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( lstm->djdft ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( lstm->djdct ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( lstm->djdot ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( lstm->djdxt ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( lstm->djdwi ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( lstm->djdwf ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( lstm->djdwo ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( lstm->djdwc ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( lstm->djdri ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( lstm->djdrf ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( lstm->djdro ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( lstm->djdrc ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( lstm->djdbi ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( lstm->djdbf ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( lstm->djdbo ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( lstm->djdbc ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( lstm->rTp ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( lstm->wTp ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( lstm->deltaTp ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( lstm->xTp ) );
#endif
}


int lstm (const libxsmm_blasint m, const libxsmm_blasint n, const libxsmm_blasint k, const libxsmm_blasint t,
          const libxsmm_blasint bm, const libxsmm_blasint bn, const libxsmm_blasint bk, const libxsmm_bgemm_order order, const int nrepeat,
          const libxsmm_blasint b_m1, const libxsmm_blasint b_n1, const libxsmm_blasint b_k1, const libxsmm_blasint b_k2, const libxsmm_blasint b_m2,
          const libxsmm_blasint ldw, const libxsmm_blasint ldx, const libxsmm_blasint ldz, const libxsmm_blasint ldu, const libxsmm_blasint ldh,
          const libxsmm_blasint reuse)
{
#if defined(CHECK)
  const char *const env_check = getenv("CHECK");
  const double check = LIBXSMM_ABS(0 == env_check ? 0 : atof(env_check));
#endif
  int result = EXIT_SUCCESS;
  const double gflops = (((2.0 * m * n * k) + (2.0 * m * n * m) + (2.0 * m * n)) * 4.0 + (5.0 * m * n)) * t * 1E-9;
  const char transa = 'N', transb = 'N'; /* no transposes */
  const int gemm_flags = LIBXSMM_GEMM_FLAGS(transa, transb);
  const ITYPE alpha = 1, beta = 1;

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload target(LIBXSMM_OFFLOAD_TARGET)
#endif
  {
    ITYPE* wigold = (ITYPE*)libxsmm_malloc(ldw * k * sizeof(ITYPE));
    ITYPE* wfgold = (ITYPE*)libxsmm_malloc(ldw * k * sizeof(ITYPE));
    ITYPE* wogold = (ITYPE*)libxsmm_malloc(ldw * k * sizeof(ITYPE));
    ITYPE* wcgold = (ITYPE*)libxsmm_malloc(ldw * k * sizeof(ITYPE));
    ITYPE* xgoldt = (ITYPE*)libxsmm_malloc(ldx * n * sizeof(ITYPE) * t);
    ITYPE* rigold = (ITYPE*)libxsmm_malloc(ldu * m * sizeof(ITYPE));
    ITYPE* rfgold = (ITYPE*)libxsmm_malloc(ldu * m * sizeof(ITYPE));
    ITYPE* rogold = (ITYPE*)libxsmm_malloc(ldu * m * sizeof(ITYPE));
    ITYPE* rcgold = (ITYPE*)libxsmm_malloc(ldu * m * sizeof(ITYPE));
    ITYPE* hgold = (ITYPE*)libxsmm_malloc(ldh * n * sizeof(ITYPE));
    ITYPE* i1gold = (ITYPE*)libxsmm_malloc(ldz * n * sizeof(ITYPE));
    ITYPE* i2gold = (ITYPE*)libxsmm_malloc(ldz * n * sizeof(ITYPE));
    ITYPE* f1gold = (ITYPE*)libxsmm_malloc(ldz * n * sizeof(ITYPE));
    ITYPE* f2gold = (ITYPE*)libxsmm_malloc(ldz * n * sizeof(ITYPE));
    ITYPE* o1gold = (ITYPE*)libxsmm_malloc(ldz * n * sizeof(ITYPE));
    ITYPE* o2gold = (ITYPE*)libxsmm_malloc(ldz * n * sizeof(ITYPE));
    ITYPE* c1gold = (ITYPE*)libxsmm_malloc(ldz * n * sizeof(ITYPE));
    ITYPE* c2gold = (ITYPE*)libxsmm_malloc(ldz * n * sizeof(ITYPE));
    ITYPE* igold = (ITYPE*)libxsmm_malloc(ldz * n * sizeof(ITYPE));
    ITYPE* fgold = (ITYPE*)libxsmm_malloc(ldz * n * sizeof(ITYPE));
    ITYPE* ogold = (ITYPE*)libxsmm_malloc(ldz * n * sizeof(ITYPE));
    ITYPE* cgold = (ITYPE*)libxsmm_malloc(ldz * n * sizeof(ITYPE));
    ITYPE* d1gold = (ITYPE*)libxsmm_malloc(ldz * n * sizeof(ITYPE));
    ITYPE* d2gold = (ITYPE*)libxsmm_malloc(ldz * n * sizeof(ITYPE));
    ITYPE* dhgold = (ITYPE*)libxsmm_malloc(ldz * n * sizeof(ITYPE));
    ITYPE* dgold = (ITYPE*)libxsmm_malloc(ldz * n * sizeof(ITYPE));
#if defined(NON_FUSED_INPUT_GEMM)
    ITYPE* wi = (ITYPE*)libxsmm_malloc(m * k * sizeof(ITYPE));
    ITYPE* wf = (ITYPE*)libxsmm_malloc(m * k * sizeof(ITYPE));
    ITYPE* wo = (ITYPE*)libxsmm_malloc(m * k * sizeof(ITYPE));
    ITYPE* wc = (ITYPE*)libxsmm_malloc(m * k * sizeof(ITYPE));
#else
    ITYPE* wi = (ITYPE*)libxsmm_malloc(m * 4 * k * sizeof(ITYPE));
    ITYPE* wf = 0;
    ITYPE* wo = 0;
    ITYPE* wc = 0;
#endif
    ITYPE* xt = (ITYPE*)libxsmm_malloc(m * n * sizeof(ITYPE) * t);
    ITYPE* ri = (ITYPE*)libxsmm_malloc(m * m * sizeof(ITYPE));
    ITYPE* rf = (ITYPE*)libxsmm_malloc(m * m * sizeof(ITYPE));
    ITYPE* ro = (ITYPE*)libxsmm_malloc(m * m * sizeof(ITYPE));
    ITYPE* rc = (ITYPE*)libxsmm_malloc(m * m * sizeof(ITYPE));
    ITYPE* h;
    if (reuse) {
      h = (ITYPE*)libxsmm_malloc(m * n * sizeof(ITYPE));
    } else {
      h = (ITYPE*)libxsmm_malloc(m * n * sizeof(ITYPE) * t);
    }
#if defined(NON_FUSED_INPUT_GEMM)
    ITYPE* i1t = (ITYPE*)libxsmm_malloc(m * n * sizeof(ITYPE) * t);
    ITYPE* f1t = (ITYPE*)libxsmm_malloc(m * n * sizeof(ITYPE) * t);
    ITYPE* o1t = (ITYPE*)libxsmm_malloc(m * n * sizeof(ITYPE) * t);
    ITYPE* c1t = (ITYPE*)libxsmm_malloc(m * n * sizeof(ITYPE) * t);
#else
    ITYPE* i1t = (ITYPE*)libxsmm_malloc(m * 4 * n * sizeof(ITYPE) * t);
    ITYPE* f1t = 0;
    ITYPE* o1t = 0;
    ITYPE* c1t = 0;
#endif
    ITYPE* i2 = (ITYPE*)libxsmm_malloc(m * n * sizeof(ITYPE));
    ITYPE* f2 = (ITYPE*)libxsmm_malloc(m * n * sizeof(ITYPE));
    ITYPE* o2 = (ITYPE*)libxsmm_malloc(m * n * sizeof(ITYPE));
    ITYPE* c2 = (ITYPE*)libxsmm_malloc(m * n * sizeof(ITYPE));
    ITYPE* d1 = (ITYPE*)libxsmm_malloc(m * n * sizeof(ITYPE));
    ITYPE* d2 = (ITYPE*)libxsmm_malloc(m * n * sizeof(ITYPE));
    ITYPE* dh = (ITYPE*)libxsmm_malloc(m * n * sizeof(ITYPE));
    ITYPE* i;
    ITYPE* f;
    ITYPE* o;
    ITYPE* c;
    ITYPE* d;
    if (reuse) {
      i = (ITYPE*)libxsmm_malloc(m * n * sizeof(ITYPE));
      f = (ITYPE*)libxsmm_malloc(m * n * sizeof(ITYPE));
      o = (ITYPE*)libxsmm_malloc(m * n * sizeof(ITYPE));
      c = (ITYPE*)libxsmm_malloc(m * n * sizeof(ITYPE));
      d = (ITYPE*)libxsmm_malloc(m * n * sizeof(ITYPE));
    } else {
      i = (ITYPE*)libxsmm_malloc(m * n * sizeof(ITYPE) * t);
      f = (ITYPE*)libxsmm_malloc(m * n * sizeof(ITYPE) * t);
      o = (ITYPE*)libxsmm_malloc(m * n * sizeof(ITYPE) * t);
      c = (ITYPE*)libxsmm_malloc(m * n * sizeof(ITYPE) * t);
      d = (ITYPE*)libxsmm_malloc(m * n * sizeof(ITYPE) * t);
    }
    LIBXSMM_VLA_DECL(2, ITYPE, xgold, xgoldt, ldx * n);
    libxsmm_bgemm_handle* handlewx = 0;
    libxsmm_bgemm_handle* handleuh = 0;
    libxsmm_bgemm_handle* handlett = 0;
    const libxsmm_gemm_prefetch_type strategy = LIBXSMM_PREFETCH_AUTO;
    handlewx = libxsmm_bgemm_handle_create(LIBXSMM_GEMM_PRECISION(ITYPE), LIBXSMM_GEMM_PRECISION(ITYPE),
      m, n, k, &bm, &bn, &bk, &b_m1, &b_n1, &b_k1, &b_k2,
      &alpha, &beta, &gemm_flags, &strategy, &order);
    handleuh = libxsmm_bgemm_handle_create(LIBXSMM_GEMM_PRECISION(ITYPE), LIBXSMM_GEMM_PRECISION(ITYPE),
      m, n, m, &bm, &bn, &bm, &b_m1, &b_n1, &b_m1, &b_m2,
      &alpha, &beta, &gemm_flags, &strategy, &order);
#if defined(NON_FUSED_INPUT_GEMM)
    handlett = libxsmm_bgemm_handle_create(LIBXSMM_GEMM_PRECISION(ITYPE), LIBXSMM_GEMM_PRECISION(ITYPE),
      m, n*t, k, &bm, &bn, &bk, &b_m1, &b_n1, &b_k1, &b_k2,
      &alpha, &beta, &gemm_flags, &strategy, &order);
#else
    handlett = libxsmm_bgemm_handle_create(LIBXSMM_GEMM_PRECISION(ITYPE), LIBXSMM_GEMM_PRECISION(ITYPE),
      m*4, n*t, k, &bm, &bn, &bk, &b_m1, &b_n1, &b_k1, &b_k2,
      &alpha, &beta, &gemm_flags, &strategy, &order);
#endif

    struct lstm_handle lstm;
    lstm.m = m;
    lstm.n = n;
    lstm.k = k;
    lstm.t = t;
    lstm.wi = wi;
    lstm.wf = wf;
    lstm.wo = wo;
    lstm.wc = wc;
    lstm.xt = xt;
    lstm.ri = ri;
    lstm.rf = rf;
    lstm.ro = ro;
    lstm.rc = rc;
    lstm.h = h;
    lstm.i1t = i1t;
    lstm.i2 = i2;
    lstm.f1t = f1t;
    lstm.f2 = f2;
    lstm.o1t = o1t;
    lstm.o2 = o2;
    lstm.c1t = c1t;
    lstm.c2 = c2;
    lstm.i = i;
    lstm.f = f;
    lstm.o = o;
    lstm.c = c;
    lstm.dh = dh;
    lstm.d1 = d1;
    lstm.d2 = d2;
    lstm.d = d;
    lstm.handlewx = handlewx;
    lstm.handleuh = handleuh;
    lstm.handlett = handlett;

    if (0 != handlewx && 0 != handleuh && 0 != handlett) {
      lstm_init(&lstm, wigold, wfgold, wogold, wcgold, xgoldt, rigold, rfgold, rogold, rcgold, hgold,
        i1gold, i2gold, f1gold, f2gold, o1gold, o2gold, c1gold, c2gold, igold, fgold, ogold, cgold,
        dhgold, d1gold, d2gold, dgold, ldw, ldx, ldz, ldu, ldh, reuse);
      lstm_execute(&lstm, nrepeat, reuse);

#if defined(CHECK)
      if (!LIBXSMM_FEQ(0, check)) { /* validate result against LAPACK/BLAS xGEMM */
        unsigned long long start;
        double duration;
        ITYPE* dtest = 0;
        int j;
        int s;
        start = libxsmm_timer_tick();
        for (s = 0; s < nrepeat; ++s) {
          for (j = 0; j < t; ++j) {
            LIBXSMM_XBLAS_SYMBOL(ITYPE)(&transa, &transb, &m, &n, &k, &alpha, wigold, &ldw, &LIBXSMM_VLA_ACCESS(2, xgold, j, 0, k * n), &ldx, &beta, i1gold, &ldz);
            LIBXSMM_XBLAS_SYMBOL(ITYPE)(&transa, &transb, &m, &n, &k, &alpha, wfgold, &ldw, &LIBXSMM_VLA_ACCESS(2, xgold, j, 0, k * n), &ldx, &beta, f1gold, &ldz);
            LIBXSMM_XBLAS_SYMBOL(ITYPE)(&transa, &transb, &m, &n, &k, &alpha, wogold, &ldw, &LIBXSMM_VLA_ACCESS(2, xgold, j, 0, k * n), &ldx, &beta, o1gold, &ldz);
            LIBXSMM_XBLAS_SYMBOL(ITYPE)(&transa, &transb, &m, &n, &k, &alpha, wcgold, &ldw, &LIBXSMM_VLA_ACCESS(2, xgold, j, 0, k * n), &ldx, &beta, c1gold, &ldz);
            LIBXSMM_XBLAS_SYMBOL(ITYPE)(&transa, &transb, &m, &n, &m, &alpha, rigold, &ldu, hgold, &ldh, &beta, i2gold, &ldz);
            LIBXSMM_XBLAS_SYMBOL(ITYPE)(&transa, &transb, &m, &n, &m, &alpha, rfgold, &ldu, hgold, &ldh, &beta, f2gold, &ldz);
            LIBXSMM_XBLAS_SYMBOL(ITYPE)(&transa, &transb, &m, &n, &m, &alpha, rogold, &ldu, hgold, &ldh, &beta, o2gold, &ldz);
            LIBXSMM_XBLAS_SYMBOL(ITYPE)(&transa, &transb, &m, &n, &m, &alpha, rcgold, &ldu, hgold, &ldh, &beta, c2gold, &ldz);
            matrix_add(m*n, i1gold, i2gold, igold);
            matrix_add(m*n, f1gold, f2gold, fgold);
            matrix_add(m*n, o1gold, o2gold, ogold);
            matrix_add(m*n, c1gold, c2gold, cgold);
            matrix_relu(m*n, igold, igold); /*sigmoid*/
            matrix_relu(m*n, fgold, fgold); /*sigmoid*/
            matrix_relu(m*n, ogold, ogold); /*sigmoid*/
            matrix_relu(m*n, cgold, cgold); /*tanh*/
            matrix_eltwise_mult(m*n, fgold, dgold, d1gold);
            matrix_eltwise_mult(m*n, igold, cgold, d2gold);
            matrix_add(m*n, d1gold, d2gold, dgold);
            if (j < t-1) {
              matrix_relu(m*n, dgold, dhgold); /*tanh*/
              matrix_eltwise_mult(m*n, ogold, dhgold, hgold);
            }
          }
        }
        duration = libxsmm_timer_duration(start, libxsmm_timer_tick());
        if (0 < duration) {
          fprintf(stdout, "\tBLAS: %.1f GFLOPS/s\n", gflops * nrepeat / duration);
        }
        /* free memory not needed further; avoid double-free later on */
        libxsmm_free(wigold); wigold = 0;
        libxsmm_free(wfgold); wfgold = 0;
        libxsmm_free(wogold); wogold = 0;
        libxsmm_free(wcgold); wcgold = 0;
        libxsmm_free(xgoldt); xgoldt = 0;
        libxsmm_free(rigold); rigold = 0;
        libxsmm_free(rfgold); rfgold = 0;
        libxsmm_free(rogold); rogold = 0;
        libxsmm_free(rcgold); rcgold = 0;
        libxsmm_free(hgold); hgold = 0;
        libxsmm_free(i1gold); i1gold = 0;
        libxsmm_free(i2gold); i2gold = 0;
        libxsmm_free(f1gold); f1gold = 0;
        libxsmm_free(f2gold); f2gold = 0;
        libxsmm_free(o1gold); o1gold = 0;
        libxsmm_free(o2gold); o2gold = 0;
        libxsmm_free(c1gold); c1gold = 0;
        libxsmm_free(c2gold); c2gold = 0;
        libxsmm_free(igold); igold = 0;
        libxsmm_free(fgold); fgold = 0;
        libxsmm_free(ogold); ogold = 0;
        libxsmm_free(cgold); cgold = 0;
        libxsmm_free(dhgold); dhgold = 0;
        libxsmm_free(d1gold); d1gold = 0;
        libxsmm_free(d2gold); d2gold = 0;
        /* allocate C-matrix in regular format, and perform copy-out */
        dtest = (ITYPE*)libxsmm_malloc(ldz * n * sizeof(ITYPE));
        LIBXSMM_VLA_DECL(2, ITYPE, dnr, d, m * n);
        if (0 != dtest) {
          libxsmm_matdiff_info diff;
          if (reuse) {
            libxsmm_bgemm_copyout_c(handleuh, d, &ldz, dtest);
          } else {
            libxsmm_bgemm_copyout_c(handleuh, &LIBXSMM_VLA_ACCESS(2, dnr, t-1, 0, m * n), &ldz, dtest);
          }
          if (EXIT_SUCCESS == libxsmm_matdiff(LIBXSMM_DATATYPE(ITYPE), m, n, dgold, dtest, &ldz, &ldz, &diff)) {
            fprintf(stdout, "\tdiff: L2abs=%f L2rel=%f\n", diff.l2_abs, diff.linf_abs);
            if (check < 100.0 * diff.normf_rel) {
              fprintf(stderr, "FAILED with an error of %f%%!\n", 100.0 * diff.normf_rel);
              result = EXIT_FAILURE;
            }
          }
          libxsmm_free(dtest);
        }
      }
#endif
      libxsmm_bgemm_handle_destroy(handlewx);
      libxsmm_bgemm_handle_destroy(handleuh);
      libxsmm_bgemm_handle_destroy(handlett);
    }
    else {
      fprintf(stderr, "FAILED to create BGEMM-handle! For details retry with LIBXSMM_VERBOSE=1.\n");
      result = EXIT_FAILURE;
    }
    libxsmm_free(wigold);
    libxsmm_free(wfgold);
    libxsmm_free(wogold);
    libxsmm_free(wcgold);
    libxsmm_free(xgoldt);
    libxsmm_free(rigold);
    libxsmm_free(rfgold);
    libxsmm_free(rogold);
    libxsmm_free(rcgold);
    libxsmm_free(hgold);
    libxsmm_free(i1gold);
    libxsmm_free(i2gold);
    libxsmm_free(f1gold);
    libxsmm_free(f2gold);
    libxsmm_free(o1gold);
    libxsmm_free(o2gold);
    libxsmm_free(c1gold);
    libxsmm_free(c2gold);
    libxsmm_free(igold);
    libxsmm_free(fgold);
    libxsmm_free(ogold);
    libxsmm_free(cgold);
    libxsmm_free(dhgold);
    libxsmm_free(d1gold);
    libxsmm_free(d2gold);
    libxsmm_free(dgold);
    lstm_destroy(&lstm);
  }
  fprintf(stdout, "Finished\n");

  return result;
}


void lstm_bwd_upd_init(struct lstm_handle *lstm, ITYPE *wigold, ITYPE *wfgold, ITYPE *wogold, ITYPE *wcgold,
  ITYPE *xgoldt, ITYPE *rigold, ITYPE *rfgold, ITYPE *rogold, ITYPE *rcgold, ITYPE *hgoldt,
  ITYPE *i1gold, ITYPE *i2gold, ITYPE *i3gold, ITYPE *f1gold, ITYPE *f2gold, ITYPE *f3gold,
  ITYPE *o1gold, ITYPE *o2gold, ITYPE *c1gold, ITYPE *c2gold, ITYPE *igoldt, ITYPE *fgoldt,
  ITYPE *ogoldt, ITYPE *cgoldt, ITYPE *d1gold, ITYPE *d2gold, ITYPE *d3gold, ITYPE *d4gold,
  ITYPE *dgoldt, ITYPE *djdhgoldt, ITYPE *deltagoldt, ITYPE *djddgoldt, ITYPE *djdigoldt,
  ITYPE *djdfgoldt, ITYPE *djdcgoldt, ITYPE *djdogoldt, ITYPE *djdxgoldt, ITYPE *djdwigold,
  ITYPE *djdwfgold, ITYPE *djdwogold, ITYPE *djdwcgold, ITYPE *djdrigold, ITYPE *djdrfgold,
  ITYPE *djdrogold, ITYPE *djdrcgold, ITYPE *djdbigold, ITYPE *djdbfgold, ITYPE *djdbogold,
  ITYPE *djdbcgold, const libxsmm_blasint ldw, const libxsmm_blasint ldx, const libxsmm_blasint ldz,
  const libxsmm_blasint ldu, const libxsmm_blasint ldh, const libxsmm_blasint pass)
{
#if defined(CHECK)
  const char *const env_check = getenv("CHECK");
  const double check = LIBXSMM_ABS(0 == env_check ? 0 : atof(env_check));
#endif
  const char transa = 'N', transb = 'N'; /* no transposes */
  const int gemm_flags = LIBXSMM_GEMM_FLAGS(transa, transb);
  const ITYPE alpha = 1, beta = 1;
  libxsmm_blasint m = lstm->m;
  libxsmm_blasint n = lstm->n;
  libxsmm_blasint k = lstm->k;
  libxsmm_blasint t = lstm->t;
  ITYPE *wi = (ITYPE*)lstm->wi;
  ITYPE *wf = (ITYPE*)lstm->wf;
  ITYPE *wo = (ITYPE*)lstm->wo;
  ITYPE *wc = (ITYPE*)lstm->wc;
  ITYPE *xt = (ITYPE*)lstm->xt;
  ITYPE *ri = (ITYPE*)lstm->ri;
  ITYPE *rf = (ITYPE*)lstm->rf;
  ITYPE *ro = (ITYPE*)lstm->ro;
  ITYPE *rc = (ITYPE*)lstm->rc;
  ITYPE *ht = (ITYPE*)lstm->h;
  ITYPE *i1 = (ITYPE*)lstm->i1t;
  ITYPE *i2 = (ITYPE*)lstm->i2;
  ITYPE *i3 = (ITYPE*)lstm->i3;
  ITYPE *f1 = (ITYPE*)lstm->f1t;
  ITYPE *f2 = (ITYPE*)lstm->f2;
  ITYPE *f3 = (ITYPE*)lstm->f3;
  ITYPE *o1 = (ITYPE*)lstm->o1t;
  ITYPE *o2 = (ITYPE*)lstm->o2;
  ITYPE *c1 = (ITYPE*)lstm->c1t;
  ITYPE *c2 = (ITYPE*)lstm->c2;
  ITYPE *it = (ITYPE*)lstm->i;
  ITYPE *ft = (ITYPE*)lstm->f;
  ITYPE *ot = (ITYPE*)lstm->o;
  ITYPE *ct = (ITYPE*)lstm->c;
  ITYPE *d1 = (ITYPE*)lstm->d1;
  ITYPE *d2 = (ITYPE*)lstm->d2;
  ITYPE *d3 = (ITYPE*)lstm->dh;
  ITYPE *d4 = (ITYPE*)lstm->d4;
  ITYPE *dt = (ITYPE*)lstm->d;
  ITYPE *djdht = (ITYPE*)lstm->djdht;
  ITYPE *deltat = (ITYPE*)lstm->deltat;
  ITYPE *djddt = (ITYPE*)lstm->djddt;
  ITYPE *djdit = (ITYPE*)lstm->djdit;
  ITYPE *djdft = (ITYPE*)lstm->djdft;
  ITYPE *djdct = (ITYPE*)lstm->djdct;
  ITYPE *djdot = (ITYPE*)lstm->djdot;
  ITYPE *djdxt = (ITYPE*)lstm->djdxt;
  ITYPE *djdwi = (ITYPE*)lstm->djdwi;
  ITYPE *djdwf = (ITYPE*)lstm->djdwf;
  ITYPE *djdwo = (ITYPE*)lstm->djdwo;
  ITYPE *djdwc = (ITYPE*)lstm->djdwc;
  ITYPE *djdri = (ITYPE*)lstm->djdri;
  ITYPE *djdrf = (ITYPE*)lstm->djdrf;
  ITYPE *djdro = (ITYPE*)lstm->djdro;
  ITYPE *djdrc = (ITYPE*)lstm->djdrc;
  ITYPE *djdbi = (ITYPE*)lstm->djdbi;
  ITYPE *djdbf = (ITYPE*)lstm->djdbf;
  ITYPE *djdbo = (ITYPE*)lstm->djdbo;
  ITYPE *djdbc = (ITYPE*)lstm->djdbc;
  ITYPE* wgoldTp = (ITYPE*)libxsmm_malloc(ldw * k * sizeof(ITYPE));
  ITYPE* rgoldTp = (ITYPE*)libxsmm_malloc(ldu * n * sizeof(ITYPE));
  ITYPE* xgoldTp = (ITYPE*)libxsmm_malloc(ldx * n * sizeof(ITYPE));
  ITYPE* deltagoldTp = (ITYPE*)libxsmm_malloc(ldz * n * sizeof(ITYPE));
  LIBXSMM_VLA_DECL(2, ITYPE, xgold, xgoldt, ldx * n);
  LIBXSMM_VLA_DECL(2, ITYPE, hgold, hgoldt, ldh * n);
  LIBXSMM_VLA_DECL(2, ITYPE, igold, igoldt, ldz * n);
  LIBXSMM_VLA_DECL(2, ITYPE, fgold, fgoldt, ldz * n);
  LIBXSMM_VLA_DECL(2, ITYPE, ogold, ogoldt, ldz * n);
  LIBXSMM_VLA_DECL(2, ITYPE, cgold, cgoldt, ldz * n);
  LIBXSMM_VLA_DECL(2, ITYPE, dgold, dgoldt, ldz * n);
  LIBXSMM_VLA_DECL(2, ITYPE, djdhgold, djdhgoldt, ldh * n);
  LIBXSMM_VLA_DECL(2, ITYPE, deltagold, deltagoldt, ldz * n);
  LIBXSMM_VLA_DECL(2, ITYPE, djddgold, djddgoldt, ldz * n);
  LIBXSMM_VLA_DECL(2, ITYPE, djdigold, djdigoldt, ldz * n);
  LIBXSMM_VLA_DECL(2, ITYPE, djdfgold, djdfgoldt, ldz * n);
  LIBXSMM_VLA_DECL(2, ITYPE, djdogold, djdogoldt, ldz * n);
  LIBXSMM_VLA_DECL(2, ITYPE, djdcgold, djdcgoldt, ldz * n);
  LIBXSMM_VLA_DECL(2, ITYPE, djdxgold, djdxgoldt, ldx * n);
  LIBXSMM_VLA_DECL(2, ITYPE, x, xt, k * n);
  LIBXSMM_VLA_DECL(2, ITYPE, h, ht, m * n);
  LIBXSMM_VLA_DECL(2, ITYPE, i, it, m * n);
  LIBXSMM_VLA_DECL(2, ITYPE, f, ft, m * n);
  LIBXSMM_VLA_DECL(2, ITYPE, o, ot, m * n);
  LIBXSMM_VLA_DECL(2, ITYPE, c, ct, m * n);
  LIBXSMM_VLA_DECL(2, ITYPE, d, dt, m * n);
  LIBXSMM_VLA_DECL(2, ITYPE, djdh, djdht, m * n);
  LIBXSMM_VLA_DECL(2, ITYPE, delta, deltat, m * n);
  LIBXSMM_VLA_DECL(2, ITYPE, djdd, djddt, m * n);
  LIBXSMM_VLA_DECL(2, ITYPE, djdi, djdit, m * n);
  LIBXSMM_VLA_DECL(2, ITYPE, djdf, djdft, m * n);
  LIBXSMM_VLA_DECL(2, ITYPE, djdo, djdot, m * n);
  LIBXSMM_VLA_DECL(2, ITYPE, djdc, djdct, m * n);
  LIBXSMM_VLA_DECL(2, ITYPE, djdx, djdxt, k * n);
  libxsmm_bgemm_handle *handleud = lstm->handlewx;
  libxsmm_bgemm_handle *handledh = lstm->handleuh;
  libxsmm_bgemm_handle *handledx = lstm->handlett;
  libxsmm_bgemm_handle *handlewd = lstm->handlewd;

  LIBXSMM_MATINIT(ITYPE, 42, wigold, m, k, ldw, 1.0);
  LIBXSMM_MATINIT(ITYPE, 42, wfgold, m, k, ldw, 1.0);
  LIBXSMM_MATINIT(ITYPE, 42, wogold, m, k, ldw, 1.0);
  LIBXSMM_MATINIT(ITYPE, 42, wcgold, m, k, ldw, 1.0);
  int iter;
  for (iter = 0; iter < t; ++iter) {
    LIBXSMM_MATINIT(ITYPE, 24, &LIBXSMM_VLA_ACCESS(2, xgold, iter, 0, ldx * n), k, n, ldx, 1.0);
    LIBXSMM_MATINIT(ITYPE, 24, &LIBXSMM_VLA_ACCESS(2, hgold, iter, 0, ldh * n), m, n, ldh, 1.0);
    LIBXSMM_MATINIT(ITYPE, 24, &LIBXSMM_VLA_ACCESS(2, igold, iter, 0, ldz * n), m, n, ldz, 1.0);
    LIBXSMM_MATINIT(ITYPE, 24, &LIBXSMM_VLA_ACCESS(2, fgold, iter, 0, ldz * n), m, n, ldz, 1.0);
    LIBXSMM_MATINIT(ITYPE, 24, &LIBXSMM_VLA_ACCESS(2, ogold, iter, 0, ldz * n), m, n, ldz, 1.0);
    LIBXSMM_MATINIT(ITYPE, 24, &LIBXSMM_VLA_ACCESS(2, cgold, iter, 0, ldz * n), m, n, ldz, 1.0);
    LIBXSMM_MATINIT(ITYPE, 24, &LIBXSMM_VLA_ACCESS(2, dgold, iter, 0, ldz * n), m, n, ldz, 1.0);
    LIBXSMM_MATINIT(ITYPE, 24, &LIBXSMM_VLA_ACCESS(2, djdhgold, iter, 0, ldh * n), m, n, ldh, 1.0);
    LIBXSMM_MATINIT(ITYPE,  0, &LIBXSMM_VLA_ACCESS(2, deltagold, iter, 0, ldz * n), m, n, ldz, 0.0);
    LIBXSMM_MATINIT(ITYPE,  0, &LIBXSMM_VLA_ACCESS(2, djddgold, iter, 0, ldz * n), m, n, ldz, 0.0);
    LIBXSMM_MATINIT(ITYPE,  0, &LIBXSMM_VLA_ACCESS(2, djdigold, iter, 0, ldz * n), m, n, ldz, 0.0);
    LIBXSMM_MATINIT(ITYPE,  0, &LIBXSMM_VLA_ACCESS(2, djdfgold, iter, 0, ldz * n), m, n, ldz, 0.0);
    LIBXSMM_MATINIT(ITYPE,  0, &LIBXSMM_VLA_ACCESS(2, djdogold, iter, 0, ldz * n), m, n, ldz, 0.0);
    LIBXSMM_MATINIT(ITYPE,  0, &LIBXSMM_VLA_ACCESS(2, djdcgold, iter, 0, ldz * n), m, n, ldz, 0.0);
  }
  LIBXSMM_MATINIT(ITYPE, 42, rigold, m, m, ldu, 1.0);
  LIBXSMM_MATINIT(ITYPE, 42, rfgold, m, m, ldu, 1.0);
  LIBXSMM_MATINIT(ITYPE, 42, rogold, m, m, ldu, 1.0);
  LIBXSMM_MATINIT(ITYPE, 42, rcgold, m, m, ldu, 1.0);
  LIBXSMM_MATINIT(ITYPE,  0, i1gold, m, n, ldz, 0.0);
  LIBXSMM_MATINIT(ITYPE,  0, i2gold, m, n, ldz, 0.0);
  LIBXSMM_MATINIT(ITYPE,  0, i3gold, m, n, ldz, 0.0);
  LIBXSMM_MATINIT(ITYPE,  0, f1gold, m, n, ldz, 0.0);
  LIBXSMM_MATINIT(ITYPE,  0, f2gold, m, n, ldz, 0.0);
  LIBXSMM_MATINIT(ITYPE,  0, f3gold, m, n, ldz, 0.0);
  LIBXSMM_MATINIT(ITYPE,  0, o1gold, m, n, ldz, 0.0);
  LIBXSMM_MATINIT(ITYPE,  0, o2gold, m, n, ldz, 0.0);
  LIBXSMM_MATINIT(ITYPE,  0, c1gold, m, n, ldz, 0.0);
  LIBXSMM_MATINIT(ITYPE,  0, c2gold, m, n, ldz, 0.0);
  LIBXSMM_MATINIT(ITYPE,  0, d1gold, m, n, ldz, 0.0);
  LIBXSMM_MATINIT(ITYPE,  0, d2gold, m, n, ldz, 0.0);
  LIBXSMM_MATINIT(ITYPE,  0, d3gold, m, n, ldz, 0.0);
  LIBXSMM_MATINIT(ITYPE,  0, d4gold, m, n, ldz, 0.0);
  LIBXSMM_MATINIT(ITYPE,  0, djdrigold, m, m, ldu, 0.0);
  LIBXSMM_MATINIT(ITYPE,  0, djdrfgold, m, m, ldu, 0.0);
  LIBXSMM_MATINIT(ITYPE,  0, djdrogold, m, m, ldu, 0.0);
  LIBXSMM_MATINIT(ITYPE,  0, djdrcgold, m, m, ldu, 0.0);
  LIBXSMM_MATINIT(ITYPE,  0, djdwigold, m, k, ldw, 0.0);
  LIBXSMM_MATINIT(ITYPE,  0, djdwfgold, m, k, ldw, 0.0);
  LIBXSMM_MATINIT(ITYPE,  0, djdwogold, m, k, ldw, 0.0);
  LIBXSMM_MATINIT(ITYPE,  0, djdwcgold, m, k, ldw, 0.0);
  LIBXSMM_MATINIT(ITYPE,  0, djdbigold, m, n, ldz, 0.0);
  LIBXSMM_MATINIT(ITYPE,  0, djdbfgold, m, n, ldz, 0.0);
  LIBXSMM_MATINIT(ITYPE,  0, djdbogold, m, n, ldz, 0.0);
  LIBXSMM_MATINIT(ITYPE,  0, djdbcgold, m, n, ldz, 0.0);
  matrix_transpose(m, k, wigold, wgoldTp);
  libxsmm_bgemm_copyin_a(handlewd, wgoldTp, &ldx, wi);
  matrix_transpose(m, k, wfgold, wgoldTp);
  libxsmm_bgemm_copyin_a(handlewd, wgoldTp, &ldx, wf);
  matrix_transpose(m, k, wogold, wgoldTp);
  libxsmm_bgemm_copyin_a(handlewd, wgoldTp, &ldx, wo);
  matrix_transpose(m, k, wcgold, wgoldTp);
  libxsmm_bgemm_copyin_a(handlewd, wgoldTp, &ldx, wc);
  for (iter = 0; iter < t; ++iter) {
    libxsmm_bgemm_copyin_b(handlewd, &LIBXSMM_VLA_ACCESS(2, hgold, iter, 0, ldh * n), &ldh, &LIBXSMM_VLA_ACCESS(2, h, iter, 0, m * n));
    matrix_transpose(m, n, &LIBXSMM_VLA_ACCESS(2, xgold, iter, 0, ldx * n), xgoldTp);
    libxsmm_bgemm_copyin_b(handledx, xgoldTp, &ldx, &LIBXSMM_VLA_ACCESS(2, x, iter, 0, k * n));
    libxsmm_bgemm_copyin_b(handlewd, &LIBXSMM_VLA_ACCESS(2, igold, iter, 0, ldz * n), &ldz, &LIBXSMM_VLA_ACCESS(2, i, iter, 0, m * n));
    libxsmm_bgemm_copyin_b(handlewd, &LIBXSMM_VLA_ACCESS(2, fgold, iter, 0, ldz * n), &ldz, &LIBXSMM_VLA_ACCESS(2, f, iter, 0, m * n));
    libxsmm_bgemm_copyin_b(handlewd, &LIBXSMM_VLA_ACCESS(2, ogold, iter, 0, ldz * n), &ldz, &LIBXSMM_VLA_ACCESS(2, o, iter, 0, m * n));
    libxsmm_bgemm_copyin_b(handlewd, &LIBXSMM_VLA_ACCESS(2, cgold, iter, 0, ldz * n), &ldz, &LIBXSMM_VLA_ACCESS(2, c, iter, 0, m * n));
    libxsmm_bgemm_copyin_b(handlewd, &LIBXSMM_VLA_ACCESS(2, dgold, iter, 0, ldz * n), &ldz, &LIBXSMM_VLA_ACCESS(2, d, iter, 0, m * n));
    libxsmm_bgemm_copyin_b(handlewd, &LIBXSMM_VLA_ACCESS(2, djdhgold, iter, 0, ldz * n), &ldz, &LIBXSMM_VLA_ACCESS(2, djdh, iter, 0, m * n));
    LIBXSMM_MATINIT(ITYPE,  0, &LIBXSMM_VLA_ACCESS(2, delta, iter, 0, ldz * n), m, n, ldz, 0.0);
    LIBXSMM_MATINIT(ITYPE,  0, &LIBXSMM_VLA_ACCESS(2, djdd, iter, 0, ldz * n), m, n, ldz, 0.0);
    LIBXSMM_MATINIT(ITYPE,  0, &LIBXSMM_VLA_ACCESS(2, djdi, iter, 0, ldz * n), m, n, ldz, 0.0);
    LIBXSMM_MATINIT(ITYPE,  0, &LIBXSMM_VLA_ACCESS(2, djdf, iter, 0, ldz * n), m, n, ldz, 0.0);
    LIBXSMM_MATINIT(ITYPE,  0, &LIBXSMM_VLA_ACCESS(2, djdo, iter, 0, ldz * n), m, n, ldz, 0.0);
    LIBXSMM_MATINIT(ITYPE,  0, &LIBXSMM_VLA_ACCESS(2, djdc, iter, 0, ldz * n), m, n, ldz, 0.0);
  }
  matrix_transpose(m, m, rigold, rgoldTp);
  libxsmm_bgemm_copyin_a(handleud, rgoldTp, &ldu, ri);
  matrix_transpose(m, m, rfgold, rgoldTp);
  libxsmm_bgemm_copyin_a(handleud, rgoldTp, &ldu, rf);
  matrix_transpose(m, m, rogold, rgoldTp);
  libxsmm_bgemm_copyin_a(handleud, rgoldTp, &ldu, ro);
  matrix_transpose(m, m, rcgold, rgoldTp);
  libxsmm_bgemm_copyin_a(handleud, rgoldTp, &ldu, rc);
  LIBXSMM_MATINIT(ITYPE,  0, i1, m, n, m, 0.0);
  LIBXSMM_MATINIT(ITYPE,  0, i2, m, n, m, 0.0);
  LIBXSMM_MATINIT(ITYPE,  0, i3, m, n, m, 0.0);
  LIBXSMM_MATINIT(ITYPE,  0, f1, m, n, m, 0.0);
  LIBXSMM_MATINIT(ITYPE,  0, f2, m, n, m, 0.0);
  LIBXSMM_MATINIT(ITYPE,  0, f3, m, n, m, 0.0);
  LIBXSMM_MATINIT(ITYPE,  0, o1, m, n, m, 0.0);
  LIBXSMM_MATINIT(ITYPE,  0, o2, m, n, m, 0.0);
  LIBXSMM_MATINIT(ITYPE,  0, c1, m, n, m, 0.0);
  LIBXSMM_MATINIT(ITYPE,  0, c2, m, n, m, 0.0);
  LIBXSMM_MATINIT(ITYPE,  0, d1, m, n, m, 0.0);
  LIBXSMM_MATINIT(ITYPE,  0, d2, m, n, m, 0.0);
  LIBXSMM_MATINIT(ITYPE,  0, d3, m, n, m, 0.0);
  LIBXSMM_MATINIT(ITYPE,  0, d4, m, n, m, 0.0);
  LIBXSMM_MATINIT(ITYPE,  0, djdri, m, m, m, 0.0);
  LIBXSMM_MATINIT(ITYPE,  0, djdrf, m, m, m, 0.0);
  LIBXSMM_MATINIT(ITYPE,  0, djdro, m, m, m, 0.0);
  LIBXSMM_MATINIT(ITYPE,  0, djdrc, m, m, m, 0.0);
  LIBXSMM_MATINIT(ITYPE,  0, djdwi, m, k, m, 0.0);
  LIBXSMM_MATINIT(ITYPE,  0, djdwf, m, k, m, 0.0);
  LIBXSMM_MATINIT(ITYPE,  0, djdwo, m, k, m, 0.0);
  LIBXSMM_MATINIT(ITYPE,  0, djdwc, m, k, m, 0.0);
  LIBXSMM_MATINIT(ITYPE,  0, djdbi, m, n, m, 0.0);
  LIBXSMM_MATINIT(ITYPE,  0, djdbf, m, n, m, 0.0);
  LIBXSMM_MATINIT(ITYPE,  0, djdbo, m, n, m, 0.0);
  LIBXSMM_MATINIT(ITYPE,  0, djdbc, m, n, m, 0.0);
}


void lstm_bwd_upd_execute(struct lstm_handle *lstm, const int nrepeat, const libxsmm_blasint pass)
{
  const char transa = 'N', transb = 'N'; /* no transposes */
  const int gemm_flags = LIBXSMM_GEMM_FLAGS(transa, transb);
  const ITYPE alpha = 1, beta = 1;
  libxsmm_blasint m = lstm->m;
  libxsmm_blasint n = lstm->n;
  libxsmm_blasint k = lstm->k;
  libxsmm_blasint t = lstm->t;
  const double tflops = 12; /* transcendental flops */
  double gflops = m * n; /* delta + delta_out */
  gflops += (6.0 * m * n + tflops * m * n); /* dJdd */
  gflops += (4.0 * m * n); /* dJdc */
  gflops += (4.0 * m * n); /* dJdi */
  gflops += (4.0 * m * n); /* dJdf */
  gflops += (4.0 * m * n + tflops * m * n); /* dJdo */
  double tempflops;
  if (pass == 1 || pass == 3) {
    tempflops += (4.0 * m * k); /* W^T */
    tempflops += (8.0 * m * n * k); /* W^T * dJd{c, i, f, o} */
    tempflops += (3.0 * m * k); /* summation */
    gflops += tempflops;
  }
  tempflops += (4.0 * m * m); /* R^T */
  tempflops += (8.0 * m * n * m); /* R^T * dJd{c, i, f, o} */
  gflops += tempflops;
  gflops *= t; /* for t time steps */
  if (pass == 2 || pass == 3) {
    tempflops = k * n; /* x^T */
    tempflops += (8.0 * m * n * k); /* delta{c, i, f, o} * x^T */
    tempflops *= t; /* for t time steps */
    tempflops += (4.0 * m * k * (t-1)); /* for summation of dJdW{c, i, f, o} */
    gflops += tempflops;
    tempflops = 4.0 * m * n; /* delta^T */
    tempflops += (8.0 * m * n * m); /* delta{c, i, f, o} * delta^T */
    tempflops *= (t - 1); /* for (t - 1) time steps */
    tempflops += (4.0 * m * n * (t-2)); /* for summation of dJdR{c, i, f, o} */
    gflops += tempflops;
    gflops += (4.0 * m * n * (t - 1)); /* delbias */
  }
  gflops *= 1E-9; /* to convert flops to Gflops */
  ITYPE *wi = (ITYPE*)lstm->wi;
  ITYPE *wf = (ITYPE*)lstm->wf;
  ITYPE *wo = (ITYPE*)lstm->wo;
  ITYPE *wc = (ITYPE*)lstm->wc;
  ITYPE *xt = (ITYPE*)lstm->xt;
  ITYPE *ri = (ITYPE*)lstm->ri;
  ITYPE *rf = (ITYPE*)lstm->rf;
  ITYPE *ro = (ITYPE*)lstm->ro;
  ITYPE *rc = (ITYPE*)lstm->rc;
  ITYPE *ht = (ITYPE*)lstm->h;
  ITYPE *i1 = (ITYPE*)lstm->i1t;
  ITYPE *i2 = (ITYPE*)lstm->i2;
  ITYPE *i3 = (ITYPE*)lstm->i3;
  ITYPE *f1 = (ITYPE*)lstm->f1t;
  ITYPE *f2 = (ITYPE*)lstm->f2;
  ITYPE *f3 = (ITYPE*)lstm->f3;
  ITYPE *o1 = (ITYPE*)lstm->o1t;
  ITYPE *o2 = (ITYPE*)lstm->o2;
  ITYPE *c1 = (ITYPE*)lstm->c1t;
  ITYPE *c2 = (ITYPE*)lstm->c2;
  ITYPE *it = (ITYPE*)lstm->i;
  ITYPE *ft = (ITYPE*)lstm->f;
  ITYPE *ot = (ITYPE*)lstm->o;
  ITYPE *ct = (ITYPE*)lstm->c;
  ITYPE *d1 = (ITYPE*)lstm->d1;
  ITYPE *d2 = (ITYPE*)lstm->d2;
  ITYPE *d3 = (ITYPE*)lstm->dh;
  ITYPE *d4 = (ITYPE*)lstm->d4;
  ITYPE *dt = (ITYPE*)lstm->d;
  ITYPE *djdht = (ITYPE*)lstm->djdht;
  ITYPE *deltat = (ITYPE*)lstm->deltat;
  ITYPE *djddt = (ITYPE*)lstm->djddt;
  ITYPE *djdit = (ITYPE*)lstm->djdit;
  ITYPE *djdft = (ITYPE*)lstm->djdft;
  ITYPE *djdct = (ITYPE*)lstm->djdct;
  ITYPE *djdot = (ITYPE*)lstm->djdot;
  ITYPE *djdxt = (ITYPE*)lstm->djdxt;
  ITYPE *djdwi = (ITYPE*)lstm->djdwi;
  ITYPE *djdwf = (ITYPE*)lstm->djdwf;
  ITYPE *djdwo = (ITYPE*)lstm->djdwo;
  ITYPE *djdwc = (ITYPE*)lstm->djdwc;
  ITYPE *djdri = (ITYPE*)lstm->djdri;
  ITYPE *djdrf = (ITYPE*)lstm->djdrf;
  ITYPE *djdro = (ITYPE*)lstm->djdro;
  ITYPE *djdrc = (ITYPE*)lstm->djdrc;
  ITYPE *djdbi = (ITYPE*)lstm->djdbi;
  ITYPE *djdbf = (ITYPE*)lstm->djdbf;
  ITYPE *djdbo = (ITYPE*)lstm->djdbo;
  ITYPE *djdbc = (ITYPE*)lstm->djdbc;
  /*
  ITYPE *rTp = (ITYPE*)lstm->rTp;
  ITYPE *wTp = (ITYPE*)lstm->wTp;
  ITYPE *deltaTp = (ITYPE*)lstm->deltaTp;
  ITYPE *xTp = (ITYPE*)lstm->xTp;
  */
  LIBXSMM_VLA_DECL(2, ITYPE, x, xt, k * n);
  LIBXSMM_VLA_DECL(2, ITYPE, h, ht, m * n);
  LIBXSMM_VLA_DECL(2, ITYPE, i, it, m * n);
  LIBXSMM_VLA_DECL(2, ITYPE, f, ft, m * n);
  LIBXSMM_VLA_DECL(2, ITYPE, o, ot, m * n);
  LIBXSMM_VLA_DECL(2, ITYPE, c, ct, m * n);
  LIBXSMM_VLA_DECL(2, ITYPE, d, dt, m * n);
  LIBXSMM_VLA_DECL(2, ITYPE, djdh, djdht, m * n);
  LIBXSMM_VLA_DECL(2, ITYPE, delta, deltat, m * n);
  LIBXSMM_VLA_DECL(2, ITYPE, djdd, djddt, m * n);
  LIBXSMM_VLA_DECL(2, ITYPE, djdi, djdit, m * n);
  LIBXSMM_VLA_DECL(2, ITYPE, djdf, djdft, m * n);
  LIBXSMM_VLA_DECL(2, ITYPE, djdo, djdot, m * n);
  LIBXSMM_VLA_DECL(2, ITYPE, djdc, djdct, m * n);
  LIBXSMM_VLA_DECL(2, ITYPE, djdx, djdxt, k * n);
  libxsmm_bgemm_handle *handleud = lstm->handlewx;
  libxsmm_bgemm_handle *handledh = lstm->handleuh;
  libxsmm_bgemm_handle *handledx = lstm->handlett;
  libxsmm_bgemm_handle *handlewd = lstm->handlewd;
  unsigned long long start;
  double duration;
  int s;
  int j;
  start = libxsmm_timer_tick();
  for (s = 0; s < nrepeat; ++s) {
    /* compute delta */
    matrix_copy(m * n, &LIBXSMM_VLA_ACCESS(2, djdh, t-1, 0, m * n), &LIBXSMM_VLA_ACCESS(2, delta, t-1, 0, m * n));
    /* compute djdd */
    matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, djdh, t-1, 0, m * n), &LIBXSMM_VLA_ACCESS(2, o, t-1, 0, m * n), d1);
    matrix_tanh_inverse(m * n, &LIBXSMM_VLA_ACCESS(2, d, t-1, 0, m * n), d2);
    matrix_eltwise_mult(m * n, d1, d2, &LIBXSMM_VLA_ACCESS(2, djdd, t-1, 0, m * n));
    /* compute djdc */
    matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, djdd, t-1, 0, m * n), &LIBXSMM_VLA_ACCESS(2, i, t-1, 0, m * n), c1);
    matrix_complement_square(m * n, &LIBXSMM_VLA_ACCESS(2, c, t-1, 0, m * n), c2);
    matrix_eltwise_mult(m * n, c1, c2, &LIBXSMM_VLA_ACCESS(2, djdc, t-1, 0, m * n));
    /* compute djdi */
    matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, djdd, t-1, 0, m * n), &LIBXSMM_VLA_ACCESS(2, c, t-1, 0, m * n), i1);
    matrix_complement(m * n, &LIBXSMM_VLA_ACCESS(2, i, t-1, 0, m * n), i2);
    matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, i, t-1, 0, m * n), i2, i3);
    matrix_eltwise_mult(m * n, i1, i3, &LIBXSMM_VLA_ACCESS(2, djdi, t-1, 0, m * n));
    /* compute djdf */
    matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, djdd, t-1, 0, m * n), &LIBXSMM_VLA_ACCESS(2, d, t-2, 0, m * n), f1);
    matrix_complement(m * n, &LIBXSMM_VLA_ACCESS(2, f, t-1, 0, m * n), f2);
    matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, f, t-1, 0, m * n), f2, f3);
    matrix_eltwise_mult(m * n, f1, f3, &LIBXSMM_VLA_ACCESS(2, djdf, t-1, 0, m * n));
    /* compute djdo */
    matrix_tanh(m * n, &LIBXSMM_VLA_ACCESS(2, d, t-1, 0, m * n), o1);
    matrix_complement(m * n, &LIBXSMM_VLA_ACCESS(2, o, t-1, 0, m * n), o2);
    matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, delta, t-1, 0, m * n), o1, o1);
    matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, o, t-1, 0, m * n), o2, o2);
    matrix_eltwise_mult(m * n, o1, o2, &LIBXSMM_VLA_ACCESS(2, djdo, t-1, 0, m * n));
    if (pass == 1 || pass == 3) {
      /* compute djdx */
      /* matrix_transpose(m, k, wi, wTp); - already taken care of in init */
      /* libxsmm_bgemm_omp(handlewd, wTp, &LIBXSMM_VLA_ACCESS(2, djdi, t-1, 0, m * n), &LIBXSMM_VLA_ACCESS(2, djdx, t-1, 0, k * n), 1); */
      libxsmm_bgemm_omp(handlewd, wi, &LIBXSMM_VLA_ACCESS(2, djdi, t-1, 0, m * n), &LIBXSMM_VLA_ACCESS(2, djdx, t-1, 0, k * n), 1);
      /* matrix_transpose(m, k, wf, wTp); - already taken care of in init */
      /* libxsmm_bgemm_omp(handlewd, wTp, &LIBXSMM_VLA_ACCESS(2, djdf, t-1, 0, m * n), &LIBXSMM_VLA_ACCESS(2, djdx, t-1, 0, k * n), 1); */
      libxsmm_bgemm_omp(handlewd, wf, &LIBXSMM_VLA_ACCESS(2, djdf, t-1, 0, m * n), &LIBXSMM_VLA_ACCESS(2, djdx, t-1, 0, k * n), 1);
      /* matrix_transpose(m, k, wo, wTp); - already taken care of in init */
      /* libxsmm_bgemm_omp(handlewd, wTp, &LIBXSMM_VLA_ACCESS(2, djdo, t-1, 0, m * n), &LIBXSMM_VLA_ACCESS(2, djdx, t-1, 0, k * n), 1); */
      libxsmm_bgemm_omp(handlewd, wo, &LIBXSMM_VLA_ACCESS(2, djdo, t-1, 0, m * n), &LIBXSMM_VLA_ACCESS(2, djdx, t-1, 0, k * n), 1);
      /* matrix_transpose(m, k, wc, wTp); - already taken care of in init */
      /* libxsmm_bgemm_omp(handlewd, wTp, &LIBXSMM_VLA_ACCESS(2, djdc, t-1, 0, m * n), &LIBXSMM_VLA_ACCESS(2, djdx, t-1, 0, k * n), 1); */
      libxsmm_bgemm_omp(handlewd, wc, &LIBXSMM_VLA_ACCESS(2, djdc, t-1, 0, m * n), &LIBXSMM_VLA_ACCESS(2, djdx, t-1, 0, k * n), 1);
    }
    for (j = t-2; j >= 0; --j) {
      /* compute delta */
      /* matrix_transpose(m, m, ri, rTp); - already taken care of in init */
      /* libxsmm_bgemm_omp(handleud, rTp, &LIBXSMM_VLA_ACCESS(2, djdi, j, 0, m * n),  &LIBXSMM_VLA_ACCESS(2, delta, j+1, 0, m * n), 1); */
      libxsmm_bgemm_omp(handleud, ri, &LIBXSMM_VLA_ACCESS(2, djdi, j, 0, m * n),  &LIBXSMM_VLA_ACCESS(2, delta, j+1, 0, m * n), 1);
      /* matrix_transpose(m, m, rf, rTp); - already taken care of in init */
      /* libxsmm_bgemm_omp(handleud, rTp, &LIBXSMM_VLA_ACCESS(2, djdf, j, 0, m * n),  &LIBXSMM_VLA_ACCESS(2, delta, j+1, 0, m * n), 1); */
      libxsmm_bgemm_omp(handleud, rf, &LIBXSMM_VLA_ACCESS(2, djdf, j, 0, m * n),  &LIBXSMM_VLA_ACCESS(2, delta, j+1, 0, m * n), 1);
      /* matrix_transpose(m, m, ro, rTp); - already taken care of in init */
      /* libxsmm_bgemm_omp(handleud, rTp, &LIBXSMM_VLA_ACCESS(2, djdo, j, 0, m * n),  &LIBXSMM_VLA_ACCESS(2, delta, j+1, 0, m * n), 1); */
      libxsmm_bgemm_omp(handleud, ro, &LIBXSMM_VLA_ACCESS(2, djdo, j, 0, m * n),  &LIBXSMM_VLA_ACCESS(2, delta, j+1, 0, m * n), 1);
      /* matrix_transpose(m, m, rc, rTp); - already taken care of in init */
      /* libxsmm_bgemm_omp(handleud, rTp, &LIBXSMM_VLA_ACCESS(2, djdc, j, 0, m * n),  &LIBXSMM_VLA_ACCESS(2, delta, j+1, 0, m * n), 1); */
      libxsmm_bgemm_omp(handleud, rc, &LIBXSMM_VLA_ACCESS(2, djdc, j, 0, m * n),  &LIBXSMM_VLA_ACCESS(2, delta, j+1, 0, m * n), 1);
      matrix_add(m * n, &LIBXSMM_VLA_ACCESS(2, djdh, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, delta, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, delta, j, 0, m * n));
      /* compute djdd */
      matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, djdh, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, o, j, 0, m * n), d1);
      matrix_tanh_inverse(m * n, &LIBXSMM_VLA_ACCESS(2, d, j, 0, m * n), d2);
      matrix_eltwise_mult(m * n, d1, d2, d3);
      matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, delta, j+1, 0, m * n), &LIBXSMM_VLA_ACCESS(2, f, j+1, 0, m * n), d4);
      matrix_add(m * n, d3, d4, &LIBXSMM_VLA_ACCESS(2, djdd, j, 0, m * n));
      /* compute djdc */
      matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, djdd, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, i, j, 0, m * n), c1);
      matrix_complement_square(m * n, &LIBXSMM_VLA_ACCESS(2, c, j, 0, m * n), c2);
      matrix_eltwise_mult(m * n, c1, c2, &LIBXSMM_VLA_ACCESS(2, djdc, j, 0, m * n));
      /* compute djdi */
      matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, djdd, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, c, j, 0, m * n), i1);
      matrix_complement(m * n, &LIBXSMM_VLA_ACCESS(2, i, j, 0, m * n), i2);
      matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, i, j, 0, m * n), i2, i3);
      matrix_eltwise_mult(m * n, i1, i3, &LIBXSMM_VLA_ACCESS(2, djdi, j, 0, m * n));
      /* compute djdf */
      if (j >= 1) {
        matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, djdd, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, d, j-1, 0, m * n), f1);
        matrix_complement(m * n, &LIBXSMM_VLA_ACCESS(2, f, j, 0, m * n), f2);
        matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, f, j, 0, m * n), f2, f3);
        matrix_eltwise_mult(m * n, f1, f3, &LIBXSMM_VLA_ACCESS(2, djdf, j, 0, m * n));
      } else {
        /* djdf is zero for j == 0 */
        matinit( 0, &LIBXSMM_VLA_ACCESS(2, djdf, j, 0, m * n), m, n, m, 0.0);
      }
      /* compute djdo */
      matrix_tanh(m * n, &LIBXSMM_VLA_ACCESS(2, d, j, 0, m * n), o1);
      matrix_complement(m * n, &LIBXSMM_VLA_ACCESS(2, o, j, 0, m * n), o2);
      matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, delta, j, 0, m * n), o1, o1);
      matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, o, j, 0, m * n), o2, o2);
      matrix_eltwise_mult(m * n, o1, o2, &LIBXSMM_VLA_ACCESS(2, djdo, j, 0, m * n));
      if (pass == 1 || pass == 3) {
        /* compute djdx */
        /* matrix_transpose(m, k, wi, wTp); - already taken care of in init */
        /* libxsmm_bgemm_omp(handlewd, wTp, &LIBXSMM_VLA_ACCESS(2, djdi, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, djdx, j, 0, k * n), 1); */
        libxsmm_bgemm_omp(handlewd, wi, &LIBXSMM_VLA_ACCESS(2, djdi, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, djdx, j, 0, k * n), 1);
        /* matrix_transpose(m, k, wf, wTp); - already taken care of in init */
        /* libxsmm_bgemm_omp(handlewd, wTp, &LIBXSMM_VLA_ACCESS(2, djdf, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, djdx, j, 0, k * n), 1); */
        libxsmm_bgemm_omp(handlewd, wf, &LIBXSMM_VLA_ACCESS(2, djdf, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, djdx, j, 0, k * n), 1);
        /* matrix_transpose(m, k, wo, wTp); - already taken care of in init */
        /* libxsmm_bgemm_omp(handlewd, wTp, &LIBXSMM_VLA_ACCESS(2, djdo, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, djdx, j, 0, k * n), 1); */
        libxsmm_bgemm_omp(handlewd, wo, &LIBXSMM_VLA_ACCESS(2, djdo, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, djdx, j, 0, k * n), 1);
        /* matrix_transpose(m, k, wc, wTp); - already taken care of in init */
        /* libxsmm_bgemm_omp(handlewd, wTp, &LIBXSMM_VLA_ACCESS(2, djdc, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, djdx, j, 0, k * n), 1); */
        libxsmm_bgemm_omp(handlewd, wc, &LIBXSMM_VLA_ACCESS(2, djdc, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, djdx, j, 0, k * n), 1);
      }
    }
    if (pass == 2 || pass == 3) {
      /* compute djdw */
      for (j = 0; j < t; ++j) {
        /* matrix_transpose(k, n, &LIBXSMM_VLA_ACCESS(2, x, j, 0, k * n), xTp); - already taken care of in init */
        /*
        libxsmm_bgemm_omp(handledx, &LIBXSMM_VLA_ACCESS(2, djdi, j, 0, m * n), xTp, djdwi, 1);
        libxsmm_bgemm_omp(handledx, &LIBXSMM_VLA_ACCESS(2, djdf, j, 0, m * n), xTp, djdwf, 1);
        libxsmm_bgemm_omp(handledx, &LIBXSMM_VLA_ACCESS(2, djdo, j, 0, m * n), xTp, djdwo, 1);
        libxsmm_bgemm_omp(handledx, &LIBXSMM_VLA_ACCESS(2, djdc, j, 0, m * n), xTp, djdwc, 1);
        */
        libxsmm_bgemm_omp(handledx, &LIBXSMM_VLA_ACCESS(2, djdi, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, x, j, 0, k * n), djdwi, 1);
        libxsmm_bgemm_omp(handledx, &LIBXSMM_VLA_ACCESS(2, djdf, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, x, j, 0, k * n), djdwf, 1);
        libxsmm_bgemm_omp(handledx, &LIBXSMM_VLA_ACCESS(2, djdo, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, x, j, 0, k * n), djdwo, 1);
        libxsmm_bgemm_omp(handledx, &LIBXSMM_VLA_ACCESS(2, djdc, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, x, j, 0, k * n), djdwc, 1);
      }
      /* compute djdr */
      for (j = 0; j < t-1; ++j) {
        /* matrix_transpose(m, n, &LIBXSMM_VLA_ACCESS(2, delta, j, 0, m * n), deltaTp); - already taken care of in init */
        libxsmm_bgemm_omp(handledh, &LIBXSMM_VLA_ACCESS(2, djdi, j+1, 0, m * n), &LIBXSMM_VLA_ACCESS(2, delta, j, 0, m * n), djdri, 1);
        libxsmm_bgemm_omp(handledh, &LIBXSMM_VLA_ACCESS(2, djdf, j+1, 0, m * n), &LIBXSMM_VLA_ACCESS(2, delta, j, 0, m * n), djdrf, 1);
        libxsmm_bgemm_omp(handledh, &LIBXSMM_VLA_ACCESS(2, djdo, j+1, 0, m * n), &LIBXSMM_VLA_ACCESS(2, delta, j, 0, m * n), djdro, 1);
        libxsmm_bgemm_omp(handledh, &LIBXSMM_VLA_ACCESS(2, djdc, j+1, 0, m * n), &LIBXSMM_VLA_ACCESS(2, delta, j, 0, m * n), djdrc, 1);
      }
      /* compute djdb */
      for (j = 0; j < t-1; j++) {
        matrix_add(m * n, &LIBXSMM_VLA_ACCESS(2, djdi, j, 0, m * n), djdbi, djdbi);
        matrix_add(m * n, &LIBXSMM_VLA_ACCESS(2, djdf, j, 0, m * n), djdbf, djdbf);
        matrix_add(m * n, &LIBXSMM_VLA_ACCESS(2, djdo, j, 0, m * n), djdbo, djdbo);
        matrix_add(m * n, &LIBXSMM_VLA_ACCESS(2, djdc, j, 0, m * n), djdbc, djdbc);
      }
    }
  }
  duration = libxsmm_timer_duration(start, libxsmm_timer_tick());
  if (0 < duration) {
    fprintf(stdout, "\tLIBXSMM: %.1f GFLOPS/s\n", gflops * nrepeat / duration);
  }
}


int lstm_bwd_upd(const libxsmm_blasint m, const libxsmm_blasint n, const libxsmm_blasint k, const libxsmm_blasint t,
          const libxsmm_blasint bm, const libxsmm_blasint bn, const libxsmm_blasint bk, const libxsmm_bgemm_order order, const int nrepeat,
          const libxsmm_blasint b_m1, const libxsmm_blasint b_n1, const libxsmm_blasint b_k1, const libxsmm_blasint b_m2, const libxsmm_blasint b_n2,
          const libxsmm_blasint b_k2, const libxsmm_blasint ldw, const libxsmm_blasint ldx, const libxsmm_blasint ldz, const libxsmm_blasint ldu,
          const libxsmm_blasint ldh, const libxsmm_blasint pass)
{
#if defined(CHECK)
  const char *const env_check = getenv("CHECK");
  const double check = LIBXSMM_ABS(0 == env_check ? 0 : atof(env_check));
#endif
  int result = EXIT_SUCCESS;
  /* TODO: Check the computation of gflops */
  const double tflops = 12; /* transcendental flops */
  double gflops = m * n; /* delta + delta_out */
  gflops += (6.0 * m * n + tflops * m * n); /* dJdd */
  gflops += (4.0 * m * n); /* dJdc */
  gflops += (4.0 * m * n); /* dJdi */
  gflops += (4.0 * m * n); /* dJdf */
  gflops += (4.0 * m * n + tflops * m * n); /* dJdo */
  double tempflops;
  if (pass == 1 || pass == 3) {
    tempflops += (4.0 * m * k); /* W^T */
    tempflops += (8.0 * m * n * k); /* W^T * dJd{c, i, f, o} */
    tempflops += (3.0 * m * k); /* summation */
    gflops += tempflops;
  }
  tempflops += (4.0 * m * m); /* R^T */
  tempflops += (8.0 * m * n * m); /* R^T * dJd{c, i, f, o} */
  gflops += tempflops;
  gflops *= t; /* for t time steps */
  if (pass == 2 || pass == 3) {
    tempflops = k * n; /* x^T */
    tempflops += (8.0 * m * n * k); /* delta{c, i, f, o} * x^T */
    tempflops *= t; /* for t time steps */
    tempflops += (4.0 * m * k * (t-1)); /* for summation of dJdW{c, i, f, o} */
    gflops += tempflops;
    tempflops = 4.0 * m * n; /* delta^T */
    tempflops += (8.0 * m * n * m); /* delta{c, i, f, o} * delta^T */
    tempflops *= (t - 1); /* for (t - 1) time steps */
    tempflops += (4.0 * m * n * (t-2)); /* for summation of dJdR{c, i, f, o} */
    gflops += tempflops;
    gflops += (4.0 * m * n * (t - 1)); /* delbias */
  }
  gflops *= 1E-9; /* to convert flops to Gflops */
  const char transa = 'N', transb = 'N'; /* no transposes */
  const int gemm_flags = LIBXSMM_GEMM_FLAGS(transa, transb);
  const ITYPE alpha = 1, beta = 1, beta0 = 0;
  const libxsmm_blasint ldn = n;

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload target(LIBXSMM_OFFLOAD_TARGET)
#endif
  {
    ITYPE* wigold = (ITYPE*)libxsmm_malloc(ldw * k * sizeof(ITYPE));
    ITYPE* wfgold = (ITYPE*)libxsmm_malloc(ldw * k * sizeof(ITYPE));
    ITYPE* wogold = (ITYPE*)libxsmm_malloc(ldw * k * sizeof(ITYPE));
    ITYPE* wcgold = (ITYPE*)libxsmm_malloc(ldw * k * sizeof(ITYPE));
    ITYPE* xgoldt = (ITYPE*)libxsmm_malloc(ldx * n * sizeof(ITYPE) * t);
    ITYPE* rigold = (ITYPE*)libxsmm_malloc(ldu * m * sizeof(ITYPE));
    ITYPE* rfgold = (ITYPE*)libxsmm_malloc(ldu * m * sizeof(ITYPE));
    ITYPE* rogold = (ITYPE*)libxsmm_malloc(ldu * m * sizeof(ITYPE));
    ITYPE* rcgold = (ITYPE*)libxsmm_malloc(ldu * m * sizeof(ITYPE));
    ITYPE* hgoldt = (ITYPE*)libxsmm_malloc(ldh * n * sizeof(ITYPE) * t);
    ITYPE* i1gold = (ITYPE*)libxsmm_malloc(ldz * n * sizeof(ITYPE));
    ITYPE* i2gold = (ITYPE*)libxsmm_malloc(ldz * n * sizeof(ITYPE));
    ITYPE* i3gold = (ITYPE*)libxsmm_malloc(ldz * n * sizeof(ITYPE));
    ITYPE* f1gold = (ITYPE*)libxsmm_malloc(ldz * n * sizeof(ITYPE));
    ITYPE* f2gold = (ITYPE*)libxsmm_malloc(ldz * n * sizeof(ITYPE));
    ITYPE* f3gold = (ITYPE*)libxsmm_malloc(ldz * n * sizeof(ITYPE));
    ITYPE* o1gold = (ITYPE*)libxsmm_malloc(ldz * n * sizeof(ITYPE));
    ITYPE* o2gold = (ITYPE*)libxsmm_malloc(ldz * n * sizeof(ITYPE));
    ITYPE* c1gold = (ITYPE*)libxsmm_malloc(ldz * n * sizeof(ITYPE));
    ITYPE* c2gold = (ITYPE*)libxsmm_malloc(ldz * n * sizeof(ITYPE));
    ITYPE* igoldt = (ITYPE*)libxsmm_malloc(ldz * n * sizeof(ITYPE) * t);
    ITYPE* fgoldt = (ITYPE*)libxsmm_malloc(ldz * n * sizeof(ITYPE) * t);
    ITYPE* ogoldt = (ITYPE*)libxsmm_malloc(ldz * n * sizeof(ITYPE) * t);
    ITYPE* cgoldt = (ITYPE*)libxsmm_malloc(ldz * n * sizeof(ITYPE) * t);
    ITYPE* d1gold = (ITYPE*)libxsmm_malloc(ldz * n * sizeof(ITYPE));
    ITYPE* d2gold = (ITYPE*)libxsmm_malloc(ldz * n * sizeof(ITYPE));
    ITYPE* d3gold = (ITYPE*)libxsmm_malloc(ldz * n * sizeof(ITYPE));
    ITYPE* d4gold = (ITYPE*)libxsmm_malloc(ldz * n * sizeof(ITYPE));
    ITYPE* dgoldt = (ITYPE*)libxsmm_malloc(ldz * n * sizeof(ITYPE) * t);
    ITYPE* djdhgoldt = (ITYPE*)libxsmm_malloc(ldh * n * sizeof(ITYPE) * t);
    ITYPE* deltagoldt = (ITYPE*)libxsmm_malloc(ldz * n * sizeof(ITYPE) * t);
    ITYPE* djddgoldt = (ITYPE*)libxsmm_malloc(ldz * n * sizeof(ITYPE) * t);
    ITYPE* djdigoldt = (ITYPE*)libxsmm_malloc(ldz * n * sizeof(ITYPE) * t);
    ITYPE* djdfgoldt = (ITYPE*)libxsmm_malloc(ldz * n * sizeof(ITYPE) * t);
    ITYPE* djdcgoldt = (ITYPE*)libxsmm_malloc(ldz * n * sizeof(ITYPE) * t);
    ITYPE* djdogoldt = (ITYPE*)libxsmm_malloc(ldz * n * sizeof(ITYPE) * t);
    ITYPE* djdxgoldt = (ITYPE*)libxsmm_malloc(ldx * n * sizeof(ITYPE) * t);
    ITYPE* djdwigold = (ITYPE*)libxsmm_malloc(ldw * k * sizeof(ITYPE));
    ITYPE* djdwfgold = (ITYPE*)libxsmm_malloc(ldw * k * sizeof(ITYPE));
    ITYPE* djdwogold = (ITYPE*)libxsmm_malloc(ldw * k * sizeof(ITYPE));
    ITYPE* djdwcgold = (ITYPE*)libxsmm_malloc(ldw * k * sizeof(ITYPE));
    ITYPE* djdrigold = (ITYPE*)libxsmm_malloc(ldu * m * sizeof(ITYPE));
    ITYPE* djdrfgold = (ITYPE*)libxsmm_malloc(ldu * m * sizeof(ITYPE));
    ITYPE* djdrogold = (ITYPE*)libxsmm_malloc(ldu * m * sizeof(ITYPE));
    ITYPE* djdrcgold = (ITYPE*)libxsmm_malloc(ldu * m * sizeof(ITYPE));
    ITYPE* djdbigold = (ITYPE*)libxsmm_malloc(ldz * n * sizeof(ITYPE));
    ITYPE* djdbfgold = (ITYPE*)libxsmm_malloc(ldz * n * sizeof(ITYPE));
    ITYPE* djdbogold = (ITYPE*)libxsmm_malloc(ldz * n * sizeof(ITYPE));
    ITYPE* djdbcgold = (ITYPE*)libxsmm_malloc(ldz * n * sizeof(ITYPE));
    ITYPE* wgoldTp = (ITYPE*)libxsmm_malloc(ldw * k * sizeof(ITYPE));
    ITYPE* rgoldTp = (ITYPE*)libxsmm_malloc(ldz * n * sizeof(ITYPE));
    ITYPE* xgoldTp = (ITYPE*)libxsmm_malloc(ldx * n * sizeof(ITYPE));
    ITYPE* deltagoldTp = (ITYPE*)libxsmm_malloc(ldz * n * sizeof(ITYPE));
    ITYPE* wi = (ITYPE*)libxsmm_malloc(m * k * sizeof(ITYPE));
    ITYPE* wf = (ITYPE*)libxsmm_malloc(m * k * sizeof(ITYPE));
    ITYPE* wo = (ITYPE*)libxsmm_malloc(m * k * sizeof(ITYPE));
    ITYPE* wc = (ITYPE*)libxsmm_malloc(m * k * sizeof(ITYPE));
    ITYPE* xt = (ITYPE*)libxsmm_malloc(m * n * sizeof(ITYPE) * t);
    ITYPE* ri = (ITYPE*)libxsmm_malloc(m * m * sizeof(ITYPE));
    ITYPE* rf = (ITYPE*)libxsmm_malloc(m * m * sizeof(ITYPE));
    ITYPE* ro = (ITYPE*)libxsmm_malloc(m * m * sizeof(ITYPE));
    ITYPE* rc = (ITYPE*)libxsmm_malloc(m * m * sizeof(ITYPE));
    ITYPE* h = (ITYPE*)libxsmm_malloc(m * n * sizeof(ITYPE) * t);
    ITYPE* djdht = (ITYPE*)libxsmm_malloc(m * n * sizeof(ITYPE) * t);
    ITYPE* deltat = (ITYPE*)libxsmm_malloc(m * n * sizeof(ITYPE) * t);
    ITYPE* djddt = (ITYPE*)libxsmm_malloc(m * n * sizeof(ITYPE) * t);
    ITYPE* djdit = (ITYPE*)libxsmm_malloc(m * n * sizeof(ITYPE) * t);
    ITYPE* djdft = (ITYPE*)libxsmm_malloc(m * n * sizeof(ITYPE) * t);
    ITYPE* djdct = (ITYPE*)libxsmm_malloc(m * n * sizeof(ITYPE) * t);
    ITYPE* djdot = (ITYPE*)libxsmm_malloc(m * n * sizeof(ITYPE) * t);
    ITYPE* djdxt = (ITYPE*)libxsmm_malloc(k * n * sizeof(ITYPE) * t);
    ITYPE* djdwi = (ITYPE*)libxsmm_malloc(ldw * k * sizeof(ITYPE));
    ITYPE* djdwf = (ITYPE*)libxsmm_malloc(ldw * k * sizeof(ITYPE));
    ITYPE* djdwo = (ITYPE*)libxsmm_malloc(ldw * k * sizeof(ITYPE));
    ITYPE* djdwc = (ITYPE*)libxsmm_malloc(ldw * k * sizeof(ITYPE));
    ITYPE* djdri = (ITYPE*)libxsmm_malloc(ldu * m * sizeof(ITYPE));
    ITYPE* djdrf = (ITYPE*)libxsmm_malloc(ldu * m * sizeof(ITYPE));
    ITYPE* djdro = (ITYPE*)libxsmm_malloc(ldu * m * sizeof(ITYPE));
    ITYPE* djdrc = (ITYPE*)libxsmm_malloc(ldu * m * sizeof(ITYPE));
    ITYPE* djdbi = (ITYPE*)libxsmm_malloc(ldz * n * sizeof(ITYPE));
    ITYPE* djdbf = (ITYPE*)libxsmm_malloc(ldz * n * sizeof(ITYPE));
    ITYPE* djdbo = (ITYPE*)libxsmm_malloc(ldz * n * sizeof(ITYPE));
    ITYPE* djdbc = (ITYPE*)libxsmm_malloc(ldz * n * sizeof(ITYPE));
    ITYPE* i1 = (ITYPE*)libxsmm_malloc(m * n * sizeof(ITYPE));
    ITYPE* i2 = (ITYPE*)libxsmm_malloc(m * n * sizeof(ITYPE));
    ITYPE* i3 = (ITYPE*)libxsmm_malloc(m * n * sizeof(ITYPE));
    ITYPE* f1 = (ITYPE*)libxsmm_malloc(m * n * sizeof(ITYPE));
    ITYPE* f2 = (ITYPE*)libxsmm_malloc(m * n * sizeof(ITYPE));
    ITYPE* f3 = (ITYPE*)libxsmm_malloc(m * n * sizeof(ITYPE));
    ITYPE* o1 = (ITYPE*)libxsmm_malloc(m * n * sizeof(ITYPE));
    ITYPE* o2 = (ITYPE*)libxsmm_malloc(m * n * sizeof(ITYPE));
    ITYPE* c1 = (ITYPE*)libxsmm_malloc(m * n * sizeof(ITYPE));
    ITYPE* c2 = (ITYPE*)libxsmm_malloc(m * n * sizeof(ITYPE));
    ITYPE* d1 = (ITYPE*)libxsmm_malloc(m * n * sizeof(ITYPE));
    ITYPE* d2 = (ITYPE*)libxsmm_malloc(m * n * sizeof(ITYPE));
    ITYPE* d3 = (ITYPE*)libxsmm_malloc(m * n * sizeof(ITYPE));
    ITYPE* d4 = (ITYPE*)libxsmm_malloc(m * n * sizeof(ITYPE));
    ITYPE* i = (ITYPE*)libxsmm_malloc(m * n * sizeof(ITYPE) * t);
    ITYPE* f = (ITYPE*)libxsmm_malloc(m * n * sizeof(ITYPE) * t);
    ITYPE* o = (ITYPE*)libxsmm_malloc(m * n * sizeof(ITYPE) * t);
    ITYPE* c = (ITYPE*)libxsmm_malloc(m * n * sizeof(ITYPE) * t);
    ITYPE* d = (ITYPE*)libxsmm_malloc(m * n * sizeof(ITYPE) * t);
    ITYPE* wTp = (ITYPE*)libxsmm_malloc(m * k * sizeof(ITYPE));
    ITYPE* rTp = (ITYPE*)libxsmm_malloc(m * n * sizeof(ITYPE));
    ITYPE* xTp = (ITYPE*)libxsmm_malloc(k * n * sizeof(ITYPE));
    ITYPE* deltaTp = (ITYPE*)libxsmm_malloc(m * n * sizeof(ITYPE));
    LIBXSMM_VLA_DECL(2, ITYPE, xgold, xgoldt, ldx * n);
    LIBXSMM_VLA_DECL(2, ITYPE, igold, igoldt, ldz * n);
    LIBXSMM_VLA_DECL(2, ITYPE, fgold, fgoldt, ldz * n);
    LIBXSMM_VLA_DECL(2, ITYPE, ogold, ogoldt, ldz * n);
    LIBXSMM_VLA_DECL(2, ITYPE, cgold, cgoldt, ldz * n);
    LIBXSMM_VLA_DECL(2, ITYPE, dgold, dgoldt, ldz * n);
    LIBXSMM_VLA_DECL(2, ITYPE, djdhgold, djdhgoldt, ldh * n);
    LIBXSMM_VLA_DECL(2, ITYPE, deltagold, deltagoldt, ldz * n);
    LIBXSMM_VLA_DECL(2, ITYPE, djddgold, djddgoldt, ldz * n);
    LIBXSMM_VLA_DECL(2, ITYPE, djdigold, djdigoldt, ldz * n);
    LIBXSMM_VLA_DECL(2, ITYPE, djdfgold, djdfgoldt, ldz * n);
    LIBXSMM_VLA_DECL(2, ITYPE, djdogold, djdogoldt, ldz * n);
    LIBXSMM_VLA_DECL(2, ITYPE, djdcgold, djdcgoldt, ldz * n);
    LIBXSMM_VLA_DECL(2, ITYPE, djdxgold, djdxgoldt, ldx * n);
    LIBXSMM_VLA_DECL(2, ITYPE, djdx, djdxt, ldx * n);
    libxsmm_bgemm_handle* handleud = 0;
    libxsmm_bgemm_handle* handledh = 0;
    libxsmm_bgemm_handle* handledx = 0;
    libxsmm_bgemm_handle* handlewd = 0;
    const libxsmm_gemm_prefetch_type strategy = LIBXSMM_PREFETCH_AUTO;
    handleud = libxsmm_bgemm_handle_create(LIBXSMM_GEMM_PRECISION(ITYPE), LIBXSMM_GEMM_PRECISION(ITYPE),
      m, n, m, &bm, &bn, &bm, &b_m1, &b_n1, &b_m1, &b_m2,
      &alpha, &beta, &gemm_flags, &strategy, &order); /* U^T*delta */
    handledh = libxsmm_bgemm_handle_create(LIBXSMM_GEMM_PRECISION(ITYPE), LIBXSMM_GEMM_PRECISION(ITYPE),
      m, m, n, &bm, &bm, &bn, &b_m1, &b_m1, &b_n1, &b_n2,
      &alpha, &beta, &gemm_flags, &strategy, &order); /* delta*h^T */
    handledx = libxsmm_bgemm_handle_create(LIBXSMM_GEMM_PRECISION(ITYPE), LIBXSMM_GEMM_PRECISION(ITYPE),
      m, k, n, &bm, &bk, &bn, &b_m1, &b_k1, &b_n1, &b_n2,
      &alpha, &beta, &gemm_flags, &strategy, &order); /* delta*x^T */
    handlewd = libxsmm_bgemm_handle_create(LIBXSMM_GEMM_PRECISION(ITYPE), LIBXSMM_GEMM_PRECISION(ITYPE),
      k, n, m, &bk, &bn, &bm, &b_k1, &b_n1, &b_m1, &b_m2,
      &alpha, &beta, &gemm_flags, &strategy, &order); /* W^T*delta */

    struct lstm_handle lstm;
    lstm.m = m;
    lstm.n = n;
    lstm.k = k;
    lstm.t = t;
    lstm.wi = wi;
    lstm.wf = wf;
    lstm.wo = wo;
    lstm.wc = wc;
    lstm.xt = xt;
    lstm.ri = ri;
    lstm.rf = rf;
    lstm.ro = ro;
    lstm.rc = rc;
    lstm.h = h;
    lstm.i1t = i1;
    lstm.i2 = i2;
    lstm.i3 = i3;
    lstm.f1t = f1;
    lstm.f2 = f2;
    lstm.f3 = f3;
    lstm.o1t = o1;
    lstm.o2 = o2;
    lstm.c1t = c1;
    lstm.c2 = c2;
    lstm.i = i;
    lstm.f = f;
    lstm.o = o;
    lstm.c = c;
    lstm.d1 = d1;
    lstm.d2 = d2;
    lstm.dh = d3;
    lstm.d4 = d4;
    lstm.d = d;
    lstm.djdht = djdht;
    lstm.deltat = deltat;
    lstm.djddt = djddt;
    lstm.djdit = djdit;
    lstm.djdft = djdft;
    lstm.djdct = djdct;
    lstm.djdot = djdot;
    lstm.djdxt = djdxt;
    lstm.djdwi = djdwi;
    lstm.djdwf = djdwf;
    lstm.djdwo = djdwo;
    lstm.djdwc = djdwc;
    lstm.djdri = djdri;
    lstm.djdrf = djdrf;
    lstm.djdro = djdro;
    lstm.djdrc = djdrc;
    lstm.djdbi = djdbi;
    lstm.djdbf = djdbf;
    lstm.djdbo = djdbo;
    lstm.djdbc = djdbc;
    lstm.rTp = rTp;
    lstm.wTp = wTp;
    lstm.deltaTp = deltaTp;
    lstm.xTp = xTp;
    lstm.handlewx = handleud;
    lstm.handleuh = handledh;
    lstm.handlett = handledx;
    lstm.handlewd = handlewd;

    if (0 != handleud && 0 != handledh && 0 != handledx && 0 != handlewd) {
      lstm_bwd_upd_init(&lstm, wigold, wfgold, wogold, wcgold, xgoldt, rigold, rfgold, rogold, rcgold, hgoldt,
        i1gold, i2gold, i3gold, f1gold, f2gold, f3gold, o1gold, o2gold, c1gold, c2gold, igoldt, fgoldt, ogoldt,
        cgoldt, d1gold, d2gold, d3gold, d4gold, dgoldt, djdhgoldt, deltagoldt, djddgoldt, djdigoldt,
        djdfgoldt, djdcgoldt, djdogoldt, djdxgoldt, djdwigold, djdwfgold, djdwogold, djdwcgold,
        djdrigold, djdrfgold, djdrogold, djdrcgold, djdbigold, djdbfgold, djdbogold, djdbcgold,
        ldw, ldx, ldz, ldu, ldh, pass);
      lstm_bwd_upd_execute(&lstm, nrepeat, pass);

#if defined(CHECK)
      if (!LIBXSMM_FEQ(0, check)) { /* validate result against LAPACK/BLAS xGEMM */
        unsigned long long start;
        double duration;
        ITYPE* djtest = 0;
        int j;
        int s;
        start = libxsmm_timer_tick();
        for (s = 0; s < nrepeat; ++s) {
          /* compute deltagold */
          matrix_copy(m * n, &LIBXSMM_VLA_ACCESS(2, djdhgold, t-1, 0, m * n), &LIBXSMM_VLA_ACCESS(2, deltagold, t-1, 0, m * n));
          /* compute djddgold */
          matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, djdhgold, t-1, 0, m * n), &LIBXSMM_VLA_ACCESS(2, ogold, t-1, 0, m * n), d1gold);
          matrix_tanh_inverse(m * n, &LIBXSMM_VLA_ACCESS(2, dgold, t-1, 0, m * n), d2gold);
          matrix_eltwise_mult(m * n, d1gold, d2gold, &LIBXSMM_VLA_ACCESS(2, djddgold, t-1, 0, m * n));
          /* compute djdcgold */
          matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, djddgold, t-1, 0, m * n), &LIBXSMM_VLA_ACCESS(2, igold, t-1, 0, m * n), c1gold);
          matrix_complement_square(m * n, &LIBXSMM_VLA_ACCESS(2, cgold, t-1, 0, m * n), c2gold);
          matrix_eltwise_mult(m * n, c1gold, c2gold, &LIBXSMM_VLA_ACCESS(2, djdcgold, t-1, 0, m * n));
          /* compute djdigold */
          matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, djddgold, t-1, 0, m * n), &LIBXSMM_VLA_ACCESS(2, cgold, t-1, 0, m * n), i1gold);
          matrix_complement(m * n, &LIBXSMM_VLA_ACCESS(2, igold, t-1, 0, m * n), i2gold);
          matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, igold, t-1, 0, m * n), i2gold, i3gold);
          matrix_eltwise_mult(m * n, i1gold, i3gold, &LIBXSMM_VLA_ACCESS(2, djdigold, t-1, 0, m * n));
          /* compute djdfgold */
          matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, djddgold, t-1, 0, m * n), &LIBXSMM_VLA_ACCESS(2, dgold, t-2, 0, m * n), f1gold);
          matrix_complement(m * n, &LIBXSMM_VLA_ACCESS(2, fgold, t-1, 0, m * n), f2gold);
          matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, fgold, t-1, 0, m * n), f2gold, f3gold);
          matrix_eltwise_mult(m * n, f1gold, f3gold, &LIBXSMM_VLA_ACCESS(2, djdfgold, t-1, 0, m * n));
          /* compute djdogold */
          matrix_tanh(m * n, &LIBXSMM_VLA_ACCESS(2, dgold, t-1, 0, m * n), o1gold);
          matrix_complement(m * n, &LIBXSMM_VLA_ACCESS(2, ogold, t-1, 0, m * n), o2gold);
          matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, deltagold, t-1, 0, m * n), o1gold, o1gold);
          matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, ogold, t-1, 0, m * n), o2gold, o2gold);
          matrix_eltwise_mult(m * n, o1gold, o2gold, &LIBXSMM_VLA_ACCESS(2, djdogold, t-1, 0, m * n));
          if (pass == 1 || pass == 3) {
            /* compute djdxgold */
            matrix_transpose(m, k, wigold, wgoldTp);
            LIBXSMM_XBLAS_SYMBOL(ITYPE)(&transa, &transb, &k, &n, &m, &alpha, wgoldTp, &ldx, &LIBXSMM_VLA_ACCESS(2, djdigold, t-1, 0, m * n), &ldz, &beta0, &LIBXSMM_VLA_ACCESS(2, djdxgold, t-1, 0, k * n), &ldx);
            matrix_transpose(m, k, wfgold, wgoldTp);
            LIBXSMM_XBLAS_SYMBOL(ITYPE)(&transa, &transb, &k, &n, &m, &alpha, wgoldTp, &ldx, &LIBXSMM_VLA_ACCESS(2, djdfgold, t-1, 0, m * n), &ldz, &beta0, &LIBXSMM_VLA_ACCESS(2, djdxgold, t-1, 0, k * n), &ldx);
            matrix_transpose(m, k, wogold, wgoldTp);
            LIBXSMM_XBLAS_SYMBOL(ITYPE)(&transa, &transb, &k, &n, &m, &alpha, wgoldTp, &ldx, &LIBXSMM_VLA_ACCESS(2, djdogold, t-1, 0, m * n), &ldz, &beta0, &LIBXSMM_VLA_ACCESS(2, djdxgold, t-1, 0, k * n), &ldx);
            matrix_transpose(m, k, wcgold, wgoldTp);
            LIBXSMM_XBLAS_SYMBOL(ITYPE)(&transa, &transb, &k, &n, &m, &alpha, wgoldTp, &ldx, &LIBXSMM_VLA_ACCESS(2, djdcgold, t-1, 0, m * n), &ldz, &beta0, &LIBXSMM_VLA_ACCESS(2, djdxgold, t-1, 0, k * n), &ldx);
          }
          for (j = t-2; j >= 0; --j) {
            /* compute deltagold */
            matrix_transpose(m, m, rigold, rgoldTp);
            LIBXSMM_XBLAS_SYMBOL(ITYPE)(&transa, &transb, &m, &n, &m, &alpha, rgoldTp, &ldu, &LIBXSMM_VLA_ACCESS(2, djdigold, j, 0, m * n), &ldz, &beta0, &LIBXSMM_VLA_ACCESS(2, deltagold, j+1, 0, m * n), &ldz);
            matrix_transpose(m, m, rfgold, rgoldTp);
            LIBXSMM_XBLAS_SYMBOL(ITYPE)(&transa, &transb, &m, &n, &m, &alpha, rgoldTp, &ldu, &LIBXSMM_VLA_ACCESS(2, djdfgold, j, 0, m * n), &ldz, &beta0, &LIBXSMM_VLA_ACCESS(2, deltagold, j+1, 0, m * n), &ldz);
            matrix_transpose(m, m, rogold, rgoldTp);
            LIBXSMM_XBLAS_SYMBOL(ITYPE)(&transa, &transb, &m, &n, &m, &alpha, rgoldTp, &ldu, &LIBXSMM_VLA_ACCESS(2, djdogold, j, 0, m * n), &ldz, &beta0, &LIBXSMM_VLA_ACCESS(2, deltagold, j+1, 0, m * n), &ldz);
            matrix_transpose(m, m, rcgold, rgoldTp);
            LIBXSMM_XBLAS_SYMBOL(ITYPE)(&transa, &transb, &m, &n, &m, &alpha, rgoldTp, &ldu, &LIBXSMM_VLA_ACCESS(2, djdcgold, j, 0, m * n), &ldz, &beta0, &LIBXSMM_VLA_ACCESS(2, deltagold, j+1, 0, m * n), &ldz);
            matrix_add(m * n, &LIBXSMM_VLA_ACCESS(2, djdhgold, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, deltagold, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, deltagold, j, 0, m * n));
            /* compute djddgold */
            matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, djdhgold, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, ogold, j, 0, m * n), d1gold);
            matrix_tanh_inverse(m * n, &LIBXSMM_VLA_ACCESS(2, dgold, j, 0, m * n), d2gold);
            matrix_eltwise_mult(m * n, d1gold, d2gold, d3gold);
            matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, deltagold, j+1, 0, m * n), &LIBXSMM_VLA_ACCESS(2, fgold, j+1, 0, m * n), d4gold);
            matrix_add(m * n, d3gold, d4gold, &LIBXSMM_VLA_ACCESS(2, djddgold, j, 0, m * n));
            /* compute djdcgold */
            matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, djddgold, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, igold, j, 0, m * n), c1gold);
          matrix_complement_square(m * n, &LIBXSMM_VLA_ACCESS(2, cgold, j, 0, m * n), c2gold);
          matrix_eltwise_mult(m * n, c1gold, c2gold, &LIBXSMM_VLA_ACCESS(2, djdcgold, j, 0, m * n));
            /* compute djdigold */
            matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, djddgold, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, cgold, j, 0, m * n), i1gold);
            matrix_complement(m * n, &LIBXSMM_VLA_ACCESS(2, igold, j, 0, m * n), i2gold);
            matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, igold, j, 0, m * n), i2gold, i3gold);
            matrix_eltwise_mult(m * n, i1gold, i3gold, &LIBXSMM_VLA_ACCESS(2, djdigold, j, 0, m * n));
            /* compute djdfgold */
            if (j >= 1) {
              matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, djddgold, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, dgold, j-1, 0, m * n), f1gold);
              matrix_complement(m * n, &LIBXSMM_VLA_ACCESS(2, fgold, j, 0, m * n), f2gold);
              matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, fgold, j, 0, m * n), f2gold, f3gold);
              matrix_eltwise_mult(m * n, f1gold, f3gold, &LIBXSMM_VLA_ACCESS(2, djdfgold, j, 0, m * n));
            } else {
              /* djdf is zero for j == 0 */
              matinit( 0, &LIBXSMM_VLA_ACCESS(2, djdfgold, j, 0, m * n), m, n, ldz, 0.0);
            }
            /* compute djdogold */
            matrix_tanh(m * n, &LIBXSMM_VLA_ACCESS(2, dgold, j, 0, m * n), o1gold);
            matrix_complement(m * n, &LIBXSMM_VLA_ACCESS(2, ogold, j, 0, m * n), o2gold);
            matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, deltagold, j, 0, m * n), o1gold, o1gold);
            matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, ogold, j, 0, m * n), o2gold, o2gold);
            matrix_eltwise_mult(m * n, o1gold, o2gold, &LIBXSMM_VLA_ACCESS(2, djdogold, j, 0, m * n));
            if (pass == 1 || pass == 3) {
              /* compute djdxgold */
              matrix_transpose(m, k, wigold, wgoldTp);
              LIBXSMM_XBLAS_SYMBOL(ITYPE)(&transa, &transb, &k, &n, &m, &alpha, wgoldTp, &ldx, &LIBXSMM_VLA_ACCESS(2, djdigold, j, 0, m * n), &ldz, &beta0, &LIBXSMM_VLA_ACCESS(2, djdxgold, j, 0, k * n), &ldx);
              matrix_transpose(m, k, wfgold, wgoldTp);
              LIBXSMM_XBLAS_SYMBOL(ITYPE)(&transa, &transb, &k, &n, &m, &alpha, wgoldTp, &ldx, &LIBXSMM_VLA_ACCESS(2, djdfgold, j, 0, m * n), &ldz, &beta0, &LIBXSMM_VLA_ACCESS(2, djdxgold, j, 0, k * n), &ldx);
              matrix_transpose(m, k, wogold, wgoldTp);
              LIBXSMM_XBLAS_SYMBOL(ITYPE)(&transa, &transb, &k, &n, &m, &alpha, wgoldTp, &ldx, &LIBXSMM_VLA_ACCESS(2, djdogold, j, 0, m * n), &ldz, &beta0, &LIBXSMM_VLA_ACCESS(2, djdxgold, j, 0, k * n), &ldx);
              matrix_transpose(m, k, wcgold, wgoldTp);
              LIBXSMM_XBLAS_SYMBOL(ITYPE)(&transa, &transb, &k, &n, &m, &alpha, wgoldTp, &ldx, &LIBXSMM_VLA_ACCESS(2, djdcgold, j, 0, m * n), &ldz, &beta0, &LIBXSMM_VLA_ACCESS(2, djdxgold, j, 0, k * n), &ldx);
            }
          }
          if (pass == 2 || pass == 3) {
            /* compute djdwgold */
            for (j = 0; j < t; ++j) {
              matrix_transpose(k, n, &LIBXSMM_VLA_ACCESS(2, xgold, j, 0, k * n), xgoldTp);
              LIBXSMM_XBLAS_SYMBOL(ITYPE)(&transa, &transb, &m, &k, &n, &alpha, &LIBXSMM_VLA_ACCESS(2, djdigold, j, 0, m * n), &ldz, xgoldTp, &ldn, &beta, djdwigold, &ldw);
              LIBXSMM_XBLAS_SYMBOL(ITYPE)(&transa, &transb, &m, &k, &n, &alpha, &LIBXSMM_VLA_ACCESS(2, djdfgold, j, 0, m * n), &ldz, xgoldTp, &ldn, &beta, djdwfgold, &ldw);
              LIBXSMM_XBLAS_SYMBOL(ITYPE)(&transa, &transb, &m, &k, &n, &alpha, &LIBXSMM_VLA_ACCESS(2, djdogold, j, 0, m * n), &ldz, xgoldTp, &ldn, &beta, djdwogold, &ldw);
              LIBXSMM_XBLAS_SYMBOL(ITYPE)(&transa, &transb, &m, &k, &n, &alpha, &LIBXSMM_VLA_ACCESS(2, djdcgold, j, 0, m * n), &ldz, xgoldTp, &ldn, &beta, djdwcgold, &ldw);

            }
            /* compute djdrgold */
            for (j = 0; j < t-1; ++j) {
              matrix_transpose(m, n, &LIBXSMM_VLA_ACCESS(2, deltagold, j, 0, m * n), deltagoldTp);
              LIBXSMM_XBLAS_SYMBOL(ITYPE)(&transa, &transb, &m, &m, &n, &alpha, &LIBXSMM_VLA_ACCESS(2, djdigold, j+1, 0, m * n), &ldz, deltagoldTp, &ldn, &beta, djdrigold, &ldu);
              LIBXSMM_XBLAS_SYMBOL(ITYPE)(&transa, &transb, &m, &m, &n, &alpha, &LIBXSMM_VLA_ACCESS(2, djdfgold, j+1, 0, m * n), &ldz, deltagoldTp, &ldn, &beta, djdrfgold, &ldu);
              LIBXSMM_XBLAS_SYMBOL(ITYPE)(&transa, &transb, &m, &m, &n, &alpha, &LIBXSMM_VLA_ACCESS(2, djdogold, j+1, 0, m * n), &ldz, deltagoldTp, &ldn, &beta, djdrogold, &ldu);
              LIBXSMM_XBLAS_SYMBOL(ITYPE)(&transa, &transb, &m, &m, &n, &alpha, &LIBXSMM_VLA_ACCESS(2, djdcgold, j+1, 0, m * n), &ldz, deltagoldTp, &ldn, &beta, djdrcgold, &ldu);
            }
            /* compute djdbgold */
            for (j = 0; j < t-1; j++) {
              matrix_add(m * n, &LIBXSMM_VLA_ACCESS(2, djdigold, j, 0, m * n), djdbigold, djdbigold);
              matrix_add(m * n, &LIBXSMM_VLA_ACCESS(2, djdfgold, j, 0, m * n), djdbfgold, djdbfgold);
              matrix_add(m * n, &LIBXSMM_VLA_ACCESS(2, djdogold, j, 0, m * n), djdbogold, djdbogold);
              matrix_add(m * n, &LIBXSMM_VLA_ACCESS(2, djdcgold, j, 0, m * n), djdbcgold, djdbcgold);
            }
          }
        }
        duration = libxsmm_timer_duration(start, libxsmm_timer_tick());
        if (0 < duration) {
          fprintf(stdout, "\tBLAS: %.1f GFLOPS/s\n", gflops * nrepeat / duration);
        }
        /* free memory not needed further; avoid double-free later on */
        libxsmm_free(wigold); wigold = 0;
        libxsmm_free(wfgold); wfgold = 0;
        libxsmm_free(wogold); wogold = 0;
        libxsmm_free(wcgold); wcgold = 0;
        libxsmm_free(xgoldt); xgoldt = 0;
        libxsmm_free(rigold); rigold = 0;
        libxsmm_free(rfgold); rfgold = 0;
        libxsmm_free(rogold); rogold = 0;
        libxsmm_free(rcgold); rcgold = 0;
        libxsmm_free(hgoldt); hgoldt = 0;
        libxsmm_free(i1gold); i1gold = 0;
        libxsmm_free(i2gold); i2gold = 0;
        libxsmm_free(i3gold); i3gold = 0;
        libxsmm_free(f1gold); f1gold = 0;
        libxsmm_free(f2gold); f2gold = 0;
        libxsmm_free(f3gold); f3gold = 0;
        libxsmm_free(o1gold); o1gold = 0;
        libxsmm_free(o2gold); o2gold = 0;
        libxsmm_free(c1gold); c1gold = 0;
        libxsmm_free(c2gold); c2gold = 0;
        libxsmm_free(igoldt); igoldt = 0;
        libxsmm_free(fgoldt); fgoldt = 0;
        libxsmm_free(ogoldt); ogoldt = 0;
        libxsmm_free(cgoldt); cgoldt = 0;
        libxsmm_free(d1gold); d1gold = 0;
        libxsmm_free(d2gold); d2gold = 0;
        libxsmm_free(d3gold); d3gold = 0;
        libxsmm_free(d4gold); d4gold = 0;
        libxsmm_free(dgoldt); dgoldt = 0;
        libxsmm_free(wgoldTp); wgoldTp = 0;
        libxsmm_free(rgoldTp); rgoldTp = 0;
        libxsmm_free(xgoldTp); xgoldTp = 0;
        libxsmm_free(deltagoldTp); deltagoldTp = 0;
        if (pass == 1 || pass == 3) {
          /* allocate C-matrix in regular format, and perform copy-out */
          djtest = (ITYPE*)libxsmm_malloc(ldx * n * sizeof(ITYPE));
          if (0 != djtest) {
            libxsmm_matdiff_info diff;
            for (j = 0; j < t; ++j) {
              libxsmm_bgemm_copyout_c(handlewd, &LIBXSMM_VLA_ACCESS(2, djdx, j, 0, k * n), &ldx, djtest);
              if (EXIT_SUCCESS == libxsmm_matdiff(LIBXSMM_DATATYPE(ITYPE), k, n, &LIBXSMM_VLA_ACCESS(2, djdxgold, j, 0, k * n), djtest, &ldx, &ldx, &diff)) {
                fprintf(stdout, "dJ/dX_%d::\tdiff: L2abs=%f L2rel=%f\n", j, diff.l2_abs, diff.linf_abs);
                if (check < 100.0 * diff.normf_rel) {
                  fprintf(stderr, "FAILED with an error of %f%%!\n", 100.0 * diff.normf_rel);
                  result = EXIT_FAILURE;
                }
              }
            }
          }
          libxsmm_free(djtest);
        }
        if (pass == 2 || pass == 3) {
          /* allocate C-matrix in regular format, and perform copy-out */
          djtest = (ITYPE*)libxsmm_malloc(ldu * m * sizeof(ITYPE));
          if (0 != djtest) {
            libxsmm_matdiff_info diff;
            libxsmm_bgemm_copyout_c(handledh, djdri, &ldu, djtest);
            if (EXIT_SUCCESS == libxsmm_matdiff(LIBXSMM_DATATYPE(ITYPE), m, m, djdrigold, djtest, &ldu, &ldu, &diff)) {
              fprintf(stdout, "dJ/dRi::\tdiff: L2abs=%f L2rel=%f\n", diff.l2_abs, diff.linf_abs);
              if (check < 100.0 * diff.normf_rel) {
                fprintf(stderr, "FAILED with an error of %f%%!\n", 100.0 * diff.normf_rel);
                result = EXIT_FAILURE;
              }
            }
          }
          libxsmm_free(djtest);
          /* allocate C-matrix in regular format, and perform copy-out */
          djtest = (ITYPE*)libxsmm_malloc(ldu * m * sizeof(ITYPE));
          if (0 != djtest) {
            libxsmm_matdiff_info diff;
            libxsmm_bgemm_copyout_c(handledh, djdrf, &ldu, djtest);
            if (EXIT_SUCCESS == libxsmm_matdiff(LIBXSMM_DATATYPE(ITYPE), m, m, djdrfgold, djtest, &ldu, &ldu, &diff)) {
              fprintf(stdout, "dJ/dRf::\tdiff: L2abs=%f L2rel=%f\n", diff.l2_abs, diff.linf_abs);
              if (check < 100.0 * diff.normf_rel) {
                fprintf(stderr, "FAILED with an error of %f%%!\n", 100.0 * diff.normf_rel);
                result = EXIT_FAILURE;
              }
            }
          }
          libxsmm_free(djtest);
          /* allocate C-matrix in regular format, and perform copy-out */
          djtest = (ITYPE*)libxsmm_malloc(ldu * m * sizeof(ITYPE));
          if (0 != djtest) {
            libxsmm_matdiff_info diff;
            libxsmm_bgemm_copyout_c(handledh, djdro, &ldu, djtest);
            if (EXIT_SUCCESS == libxsmm_matdiff(LIBXSMM_DATATYPE(ITYPE), m, m, djdrogold, djtest, &ldu, &ldu, &diff)) {
              fprintf(stdout, "dJ/dRo::\tdiff: L2abs=%f L2rel=%f\n", diff.l2_abs, diff.linf_abs);
              if (check < 100.0 * diff.normf_rel) {
                fprintf(stderr, "FAILED with an error of %f%%!\n", 100.0 * diff.normf_rel);
                result = EXIT_FAILURE;
              }
            }
          }
          libxsmm_free(djtest);
          /* allocate C-matrix in regular format, and perform copy-out */
          djtest = (ITYPE*)libxsmm_malloc(ldu * m * sizeof(ITYPE));
          if (0 != djtest) {
            libxsmm_matdiff_info diff;
            libxsmm_bgemm_copyout_c(handledh, djdrc, &ldu, djtest);
            if (EXIT_SUCCESS == libxsmm_matdiff(LIBXSMM_DATATYPE(ITYPE), m, m, djdrcgold, djtest, &ldu, &ldu, &diff)) {
              fprintf(stdout, "dJ/dRc::\tdiff: L2abs=%f L2rel=%f\n", diff.l2_abs, diff.linf_abs);
              if (check < 100.0 * diff.normf_rel) {
                fprintf(stderr, "FAILED with an error of %f%%!\n", 100.0 * diff.normf_rel);
                result = EXIT_FAILURE;
              }
            }
          }
          libxsmm_free(djtest);
          /* allocate C-matrix in regular format, and perform copy-out */
          djtest = (ITYPE*)libxsmm_malloc(ldw * k * sizeof(ITYPE));
          if (0 != djtest) {
            libxsmm_matdiff_info diff;
            libxsmm_bgemm_copyout_c(handledx, djdwi, &ldw, djtest);
            if (EXIT_SUCCESS == libxsmm_matdiff(LIBXSMM_DATATYPE(ITYPE), m, k, djdwigold, djtest, &ldw, &ldw, &diff)) {
              fprintf(stdout, "dJ/dWi::\tdiff: L2abs=%f L2rel=%f\n", diff.l2_abs, diff.linf_abs);
              if (check < 100.0 * diff.normf_rel) {
                fprintf(stderr, "FAILED with an error of %f%%!\n", 100.0 * diff.normf_rel);
                result = EXIT_FAILURE;
              }
            }
          }
          libxsmm_free(djtest);
          /* allocate C-matrix in regular format, and perform copy-out */
          djtest = (ITYPE*)libxsmm_malloc(ldw * k * sizeof(ITYPE));
          if (0 != djtest) {
            libxsmm_matdiff_info diff;
            libxsmm_bgemm_copyout_c(handledx, djdwf, &ldw, djtest);
            if (EXIT_SUCCESS == libxsmm_matdiff(LIBXSMM_DATATYPE(ITYPE), m, k, djdwfgold, djtest, &ldw, &ldw, &diff)) {
              fprintf(stdout, "dJ/dWf::\tdiff: L2abs=%f L2rel=%f\n", diff.l2_abs, diff.linf_abs);
              if (check < 100.0 * diff.normf_rel) {
                fprintf(stderr, "FAILED with an error of %f%%!\n", 100.0 * diff.normf_rel);
                result = EXIT_FAILURE;
              }
            }
          }
          libxsmm_free(djtest);
          /* allocate C-matrix in regular format, and perform copy-out */
          djtest = (ITYPE*)libxsmm_malloc(ldw * k * sizeof(ITYPE));
          if (0 != djtest) {
            libxsmm_matdiff_info diff;
            libxsmm_bgemm_copyout_c(handledx, djdwo, &ldw, djtest);
            if (EXIT_SUCCESS == libxsmm_matdiff(LIBXSMM_DATATYPE(ITYPE), m, k, djdwogold, djtest, &ldw, &ldw, &diff)) {
              fprintf(stdout, "dJ/dWo::\tdiff: L2abs=%f L2rel=%f\n", diff.l2_abs, diff.linf_abs);
              if (check < 100.0 * diff.normf_rel) {
                fprintf(stderr, "FAILED with an error of %f%%!\n", 100.0 * diff.normf_rel);
                result = EXIT_FAILURE;
              }
            }
          }
          libxsmm_free(djtest);
          /* allocate C-matrix in regular format, and perform copy-out */
          djtest = (ITYPE*)libxsmm_malloc(ldw * k * sizeof(ITYPE));
          if (0 != djtest) {
            libxsmm_matdiff_info diff;
            libxsmm_bgemm_copyout_c(handledx, djdwc, &ldw, djtest);
            if (EXIT_SUCCESS == libxsmm_matdiff(LIBXSMM_DATATYPE(ITYPE), m, k, djdwcgold, djtest, &ldw, &ldw, &diff)) {
              fprintf(stdout, "dJ/dWc::\tdiff: L2abs=%f L2rel=%f\n", diff.l2_abs, diff.linf_abs);
              if (check < 100.0 * diff.normf_rel) {
                fprintf(stderr, "FAILED with an error of %f%%!\n", 100.0 * diff.normf_rel);
                result = EXIT_FAILURE;
              }
            }
          }
          libxsmm_free(djtest);
          /* allocate C-matrix in regular format, and perform copy-out */
          djtest = (ITYPE*)libxsmm_malloc(ldz * n * sizeof(ITYPE));
          if (0 != djtest) {
            libxsmm_matdiff_info diff;
            libxsmm_bgemm_copyout_c(handleud, djdbi, &ldz, djtest);
            if (EXIT_SUCCESS == libxsmm_matdiff(LIBXSMM_DATATYPE(ITYPE), m, n, djdbigold, djtest, &ldz, &ldz, &diff)) {
              fprintf(stdout, "dJ/dBi::\tdiff: L2abs=%f L2rel=%f\n", diff.l2_abs, diff.linf_abs);
              if (check < 100.0 * diff.normf_rel) {
                fprintf(stderr, "FAILED with an error of %f%%!\n", 100.0 * diff.normf_rel);
                result = EXIT_FAILURE;
              }
            }
          }
          libxsmm_free(djtest);
          /* allocate C-matrix in regular format, and perform copy-out */
          djtest = (ITYPE*)libxsmm_malloc(ldz * n * sizeof(ITYPE));
          if (0 != djtest) {
            libxsmm_matdiff_info diff;
            libxsmm_bgemm_copyout_c(handleud, djdbf, &ldz, djtest);
            if (EXIT_SUCCESS == libxsmm_matdiff(LIBXSMM_DATATYPE(ITYPE), m, n, djdbfgold, djtest, &ldz, &ldz, &diff)) {
              fprintf(stdout, "dJ/dBf::\tdiff: L2abs=%f L2rel=%f\n", diff.l2_abs, diff.linf_abs);
              if (check < 100.0 * diff.normf_rel) {
                fprintf(stderr, "FAILED with an error of %f%%!\n", 100.0 * diff.normf_rel);
                result = EXIT_FAILURE;
              }
            }
          }
          libxsmm_free(djtest);
          /* allocate C-matrix in regular format, and perform copy-out */
          djtest = (ITYPE*)libxsmm_malloc(ldz * n * sizeof(ITYPE));
          if (0 != djtest) {
            libxsmm_matdiff_info diff;
            libxsmm_bgemm_copyout_c(handleud, djdbo, &ldz, djtest);
            if (EXIT_SUCCESS == libxsmm_matdiff(LIBXSMM_DATATYPE(ITYPE), m, n, djdbogold, djtest, &ldz, &ldz, &diff)) {
              fprintf(stdout, "dJ/dBo::\tdiff: L2abs=%f L2rel=%f\n", diff.l2_abs, diff.linf_abs);
              if (check < 100.0 * diff.normf_rel) {
                fprintf(stderr, "FAILED with an error of %f%%!\n", 100.0 * diff.normf_rel);
                result = EXIT_FAILURE;
              }
            }
          }
          libxsmm_free(djtest);
          /* allocate C-matrix in regular format, and perform copy-out */
          djtest = (ITYPE*)libxsmm_malloc(ldz * n * sizeof(ITYPE));
          if (0 != djtest) {
            libxsmm_matdiff_info diff;
            libxsmm_bgemm_copyout_c(handleud, djdbc, &ldz, djtest);
            if (EXIT_SUCCESS == libxsmm_matdiff(LIBXSMM_DATATYPE(ITYPE), m, n, djdbcgold, djtest, &ldz, &ldz, &diff)) {
              fprintf(stdout, "dJ/dBi::\tdiff: L2abs=%f L2rel=%f\n", diff.l2_abs, diff.linf_abs);
              if (check < 100.0 * diff.normf_rel) {
                fprintf(stderr, "FAILED with an error of %f%%!\n", 100.0 * diff.normf_rel);
                result = EXIT_FAILURE;
              }
            }
          }
          libxsmm_free(djtest);
        }
      }
#endif
      libxsmm_bgemm_handle_destroy(handleud);
      libxsmm_bgemm_handle_destroy(handledh);
      libxsmm_bgemm_handle_destroy(handledx);
      libxsmm_bgemm_handle_destroy(handlewd);
    }
    else {
      fprintf(stderr, "FAILED to create BGEMM-handle! For details retry with LIBXSMM_VERBOSE=1.\n");
      result = EXIT_FAILURE;
    }
    libxsmm_free(wigold);
    libxsmm_free(wfgold);
    libxsmm_free(wogold);
    libxsmm_free(wcgold);
    libxsmm_free(xgoldt);
    libxsmm_free(rigold);
    libxsmm_free(rfgold);
    libxsmm_free(rogold);
    libxsmm_free(rcgold);
    libxsmm_free(hgoldt);
    libxsmm_free(i1gold);
    libxsmm_free(i2gold);
    libxsmm_free(i3gold);
    libxsmm_free(f1gold);
    libxsmm_free(f2gold);
    libxsmm_free(f3gold);
    libxsmm_free(o1gold);
    libxsmm_free(o2gold);
    libxsmm_free(c1gold);
    libxsmm_free(c2gold);
    libxsmm_free(igoldt);
    libxsmm_free(fgoldt);
    libxsmm_free(ogoldt);
    libxsmm_free(cgoldt);
    libxsmm_free(d1gold);
    libxsmm_free(d2gold);
    libxsmm_free(d3gold);
    libxsmm_free(d4gold);
    libxsmm_free(dgoldt);
    libxsmm_free(wgoldTp);
    libxsmm_free(rgoldTp);
    libxsmm_free(xgoldTp);
    libxsmm_free(deltagoldTp);
    libxsmm_free(djdhgoldt);
    libxsmm_free(deltagoldt);
    libxsmm_free(djddgoldt);
    libxsmm_free(djdigoldt);
    libxsmm_free(djdfgoldt);
    libxsmm_free(djdogoldt);
    libxsmm_free(djdcgoldt);
    libxsmm_free(djdxgoldt);
    libxsmm_free(djdwigold);
    libxsmm_free(djdwfgold);
    libxsmm_free(djdwogold);
    libxsmm_free(djdwcgold);
    libxsmm_free(djdrigold);
    libxsmm_free(djdrfgold);
    libxsmm_free(djdrogold);
    libxsmm_free(djdrcgold);
    libxsmm_free(djdbigold);
    libxsmm_free(djdbfgold);
    libxsmm_free(djdbogold);
    libxsmm_free(djdbcgold);
    lstm_destroy(&lstm);
  }
  fprintf(stdout, "Finished\n");

  return result;
}


int main(int argc, char* argv[])
{
  const int type = (1 < argc ? atoi(argv[1]) : 0);
  const libxsmm_blasint pass = (2 < argc ? atoi(argv[2]) : 0);
  const libxsmm_blasint m = (3 < argc ? atoi(argv[3]) : 1024);
  const libxsmm_blasint k = (5 < argc ? atoi(argv[5]) : m);
  const libxsmm_blasint n = (4 < argc ? atoi(argv[4]) : k);
  const libxsmm_blasint t = (6 < argc ? atoi(argv[6]) : 4);
  const int nrepeat = (7 < argc ? atoi(argv[7]) : 100);
  const libxsmm_blasint reuse = (8 < argc ? atoi(argv[8]) : 1);
  const libxsmm_blasint bm = (9 < argc ? atoi(argv[9]) : 32);
  const libxsmm_blasint bk = (11 < argc ? atoi(argv[11]) : bm);
  const libxsmm_blasint bn = (10 < argc ? atoi(argv[10]) : bk);
  const libxsmm_bgemm_order order = (libxsmm_bgemm_order)(12 < argc ? atoi(argv[12]) : 0);
  const libxsmm_blasint b_m1 = (13 < argc ? atoi(argv[13]) : 1);
  const libxsmm_blasint b_n1  = (14 < argc ? atoi(argv[14]) : 1);
  const libxsmm_blasint b_k1 = (15 < argc ? atoi(argv[15]) : 1);
  const libxsmm_blasint b_m2 = (16 < argc ? atoi(argv[16]) : 1);
  const libxsmm_blasint b_n2 = (17 < argc ? atoi(argv[17]) : 1);
  const libxsmm_blasint b_k2 = (18 < argc ? atoi(argv[18]) : 1);
  const libxsmm_blasint ldw = (19 < argc ? atoi(argv[19]) : m);
  const libxsmm_blasint ldx = (20 < argc ? atoi(argv[20]) : k);
  const libxsmm_blasint ldz = (21 < argc ? atoi(argv[21]) : m);
  const libxsmm_blasint ldu = (22 < argc ? atoi(argv[22]) : m);
  const libxsmm_blasint ldh = (23 < argc ? atoi(argv[23]) : m);
  int result = EXIT_SUCCESS;
  if (argc > 1 && !strncmp(argv[1], "-h", 3)) { /* check command line */
    printf("\nUsage: ./lstmcell [type: 0--RNN, 1--LSTM] [pass: 0--FWD, 1--BWD, 2--UPD, 3--BWD+UPD] [M] [N] [K] [time_steps > 1] [reps] [reuse (for FWD): 0/1] [bm] [bn] [bk] [order] [b_m1] [b_n1] [b_k1] [b_m2] [b_n2] [b_k2]\n\n");
    return result;
  }
  if (t <= 1) {
    printf("time_steps %d should be greater than 1\n\n", t);
    return result;
  }
  if ( (pass == 1 || pass == 2 || pass == 3) &&
      !(bm == bn && bn == bk && b_m1 == b_n1 && b_n1 == b_k1 && b_m2 == b_n2 && b_n2 == b_k2) ) {
    printf("Required condition for BWD/UPD, bm == bn == bk and bm1 == bn1 == bk1 and bm2 == bn2 == bk2\n\n");
    return result;
  }
  if (type == 0) {
    fprintf(stdout, "Running RNN ...\n");
    if (pass == 0) {
      return rnn (m, n, k, t, bm, bn, bk, order, nrepeat, b_m1, b_n1, b_k1, b_k2, b_m2, ldw, ldx, ldz, ldu, ldh, reuse);
    } else if (pass == 1 || pass == 2 || pass == 3) {
      return rnn_bwd_upd(m, n, k, t, bm, bn, bk, order, nrepeat, b_m1, b_n1, b_k1, b_m2, b_n2, b_k2, ldw, ldx, ldz, ldu, ldh, pass);
    } else {
      printf("Unknown pass: %d, valid arguments for pass = {0(FWD), 1(BWD), 2(UPD), 3(BWD+UPD)\n\n", pass);
      return result;
    }
  } else if (type == 1) {
    fprintf(stdout, "Running LSTM ...\n");
    if (pass == 0) {
      return lstm(m, n, k, t, bm, bn, bk, order, nrepeat, b_m1, b_n1, b_k1, b_k2, b_m2, ldw, ldx, ldz, ldu, ldh, reuse);
    } else if (pass == 1 || pass == 2 || pass == 3) {
      return lstm_bwd_upd(m, n, k, t, bm, bn, bk, order, nrepeat, b_m1, b_n1, b_k1, b_m2, b_n2, b_k2, ldw, ldx, ldz, ldu, ldh, pass);
    } else {
      printf("Unknown pass: %d, valid arguments for pass = {0(FWD), 1(BWD), 2(UPD), 3(BWD+UPD)\n\n", pass);
      return result;
    }
  } else {
    fprintf(stdout, "Type %d currently not implemented!\n", type);
    return result;
  }
}

