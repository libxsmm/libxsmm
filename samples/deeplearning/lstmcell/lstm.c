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
  libxsmm_bgemm_handle *handlewx;
  libxsmm_bgemm_handle *handleuh;
  libxsmm_bgemm_handle *handlett;
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
  ITYPE *d0;
  ITYPE *d1;
  ITYPE *d2;
  ITYPE *d;
  libxsmm_bgemm_handle *handlewx;
  libxsmm_bgemm_handle *handleuh;
  libxsmm_bgemm_handle *handlett;
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
    dst[i] = (src[i] >= 0) ? src[i] : -src[i];
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
  libxsmm_bgemm_copyin_b(handleuh, hgold, &ldh, h);
  for (it = 0; it < t; ++it) {
    libxsmm_bgemm_copyin_c(handlewx, z1gold, &ldz, &LIBXSMM_VLA_ACCESS(2, z1, it, 0, m * n));
  }
  libxsmm_bgemm_copyin_c(handleuh, z2gold, &ldz, z2);
  libxsmm_bgemm_copyin_c(handlewx, zgold, &ldz, z);
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


void rnn_execute(struct rnn_handle *rnn, const int nrepeat)
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
    for (i = 0; i < t-1; ++i) {
      recursive_step(handleuh, u, h, z2, &LIBXSMM_VLA_ACCESS(2, z1, i, 0, m * n), z, h, 1, m * n); /*sigmoid*/
    }
    recursive_step(handleuh, u, h, z2, &LIBXSMM_VLA_ACCESS(2, z1, t-1, 0, m * n), z, z, 0, m * n); /*nop*/
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
#endif
}


int rnn (const libxsmm_blasint m, const libxsmm_blasint n, const libxsmm_blasint k, const libxsmm_blasint t,
         const libxsmm_blasint bm, const libxsmm_blasint bn, const libxsmm_blasint bk, const libxsmm_bgemm_order order, const int nrepeat,
         const libxsmm_blasint b_m1, const libxsmm_blasint b_n1, const libxsmm_blasint b_k1, const libxsmm_blasint b_k2, const libxsmm_blasint b_m2,
         const libxsmm_blasint ldw, const libxsmm_blasint ldx, const libxsmm_blasint ldz, const libxsmm_blasint ldu, const libxsmm_blasint ldh)
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
    ITYPE* h = (ITYPE*)libxsmm_malloc(m * n * sizeof(ITYPE));
    ITYPE* z1t = (ITYPE*)libxsmm_malloc(m * n * sizeof(ITYPE) * t);
    ITYPE* z2 = (ITYPE*)libxsmm_malloc(m * n * sizeof(ITYPE));
    ITYPE* z = (ITYPE*)libxsmm_malloc(m * n * sizeof(ITYPE));
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
      rnn_init(&rnn, wgold, xgoldt, ugold, hgold, z1gold, z2gold, zgold, ldw, ldx, ldz, ldu, ldh);
      rnn_execute(&rnn, nrepeat);
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
        if (0 != ztest) {
          libxsmm_matdiff_info diff;
          libxsmm_bgemm_copyout_c(handleuh, z, &ldz, ztest);
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


void lstm_init(struct lstm_handle *lstm, ITYPE *wigold, ITYPE *wfgold, ITYPE *wogold, ITYPE *wcgold,
  ITYPE *xgoldt, ITYPE *rigold, ITYPE *rfgold, ITYPE *rogold, ITYPE *rcgold, ITYPE *hgold,
  ITYPE *i1gold, ITYPE *i2gold, ITYPE *f1gold, ITYPE *f2gold, ITYPE *o1gold, ITYPE *o2gold,
  ITYPE *c1gold, ITYPE *c2gold, ITYPE *igold, ITYPE *fgold, ITYPE *ogold, ITYPE *cgold,
  ITYPE *d0gold, ITYPE *d1gold, ITYPE *d2gold, ITYPE *dgold,
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
  ITYPE *d0 = (ITYPE*)lstm->d0;
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
  LIBXSMM_MATINIT(ITYPE, 24, d0gold, m, n, ldh, 1.0);
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
  LIBXSMM_MATINIT(ITYPE,  0, dgold, m, n, ldz, 1.0);
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
  libxsmm_bgemm_copyin_b(handleuh, hgold, &ldh, h);
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
  libxsmm_bgemm_copyin_c(handleuh, igold, &ldh, i);
  libxsmm_bgemm_copyin_c(handleuh, fgold, &ldh, f);
  libxsmm_bgemm_copyin_c(handleuh, ogold, &ldh, o);
  libxsmm_bgemm_copyin_c(handleuh, cgold, &ldh, c);
  libxsmm_bgemm_copyin_c(handleuh, d1gold, &ldh, d1);
  libxsmm_bgemm_copyin_c(handleuh, d2gold, &ldh, d2);
  libxsmm_bgemm_copyin_c(handleuh, d0gold, &ldh, d0);
  libxsmm_bgemm_copyin_c(handleuh, dgold, &ldh, d);
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


void lstm_execute(struct lstm_handle *lstm, const int nrepeat)
{
  const char transa = 'N', transb = 'N'; /* no transposes */
  const int gemm_flags = LIBXSMM_GEMM_FLAGS(transa, transb);
  const ITYPE alpha = 1, beta = 1;
  libxsmm_blasint m = lstm->m;
  libxsmm_blasint n = lstm->n;
  libxsmm_blasint k = lstm->k;
  libxsmm_blasint t = lstm->t;
  const double gflops = (((2.0 * m * n * k) + (2.0 * m * n * m) + (2.0 * m * n)) * 4.0 + (4.0 * m * n)) * t * 1E-9;
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
  ITYPE *d0 = (ITYPE*)lstm->d0;
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
    recursive_step(handleuh, ri, h, i2, &LIBXSMM_VLA_ACCESS(2, i1, 0, 0, m * n), i, i, 1, m * n); /*sigmoid*/
    recursive_step(handleuh, rf, h, f2, &LIBXSMM_VLA_ACCESS(2, f1, 0, 0, m * n), f, f, 1, m * n); /*sigmoid*/
    recursive_step(handleuh, ro, h, o2, &LIBXSMM_VLA_ACCESS(2, o1, 0, 0, m * n), o, o, 1, m * n); /*sigmoid*/
    recursive_step(handleuh, rc, h, c2, &LIBXSMM_VLA_ACCESS(2, c1, 0, 0, m * n), c, c, 1, m * n); /*tanh*/
#if defined(LSTM_TIMING)
    Gbl_t_eltwise = libxsmm_timer_tick();
#endif
    matrix_eltwise_mult(m*n, f, d0, d1);
    matrix_eltwise_mult(m*n, i, c, d2);
    matrix_add(m*n, d1, d2, d);
#if defined(LSTM_TIMING)
    Gbl_duration_eltwise = libxsmm_timer_duration(Gbl_t_eltwise, libxsmm_timer_tick());
    Gbl_t_eltwise_total += Gbl_duration_eltwise;
#endif
    if (t > 1) {
#if defined(LSTM_TIMING)
      Gbl_t_nonlin = libxsmm_timer_tick();
#endif
      matrix_relu(m*n, d, d); /*tanh*/
#if defined(LSTM_TIMING)
      Gbl_duration_nonlin = libxsmm_timer_duration(Gbl_t_nonlin, libxsmm_timer_tick());
      Gbl_t_nonlin_total += Gbl_duration_nonlin;
      Gbl_t_eltwise = libxsmm_timer_tick();
#endif
      matrix_eltwise_mult(m*n, o, d, h);
#if defined(LSTM_TIMING)
      Gbl_duration_eltwise = libxsmm_timer_duration(Gbl_t_eltwise, libxsmm_timer_tick());
      Gbl_t_eltwise_total += Gbl_duration_eltwise;
#endif
    }
    for (j = 1; j < t-1; ++j) {
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
      matrix_relu(m*n, d, d); /*tanh*/
#if defined(LSTM_TIMING)
      Gbl_duration_nonlin = libxsmm_timer_duration(Gbl_t_nonlin, libxsmm_timer_tick());
      Gbl_t_nonlin_total += Gbl_duration_nonlin;
      Gbl_t_eltwise = libxsmm_timer_tick();
#endif
      matrix_eltwise_mult(m*n, o, d, h);
#if defined(LSTM_TIMING)
      Gbl_duration_eltwise = libxsmm_timer_duration(Gbl_t_eltwise, libxsmm_timer_tick());
      Gbl_t_eltwise_total += Gbl_duration_eltwise;
#endif
    }
    if (t > 1) {
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
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( lstm->d0 ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( lstm->d1 ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( lstm->d2 ) );
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( lstm->d ) );
#endif
}


int lstm (const libxsmm_blasint m, const libxsmm_blasint n, const libxsmm_blasint k, const libxsmm_blasint t,
          const libxsmm_blasint bm, const libxsmm_blasint bn, const libxsmm_blasint bk, const libxsmm_bgemm_order order, const int nrepeat,
          const libxsmm_blasint b_m1, const libxsmm_blasint b_n1, const libxsmm_blasint b_k1, const libxsmm_blasint b_k2, const libxsmm_blasint b_m2,
          const libxsmm_blasint ldw, const libxsmm_blasint ldx, const libxsmm_blasint ldz, const libxsmm_blasint ldu, const libxsmm_blasint ldh)
{
#if defined(CHECK)
  const char *const env_check = getenv("CHECK");
  const double check = LIBXSMM_ABS(0 == env_check ? 0 : atof(env_check));
#endif
  int result = EXIT_SUCCESS;
  const double gflops = (((2.0 * m * n * k) + (2.0 * m * n * m) + (2.0 * m * n)) * 4.0 + (4.0 * m * n)) * t * 1E-9;
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
    ITYPE* d0gold = (ITYPE*)libxsmm_malloc(ldz * n * sizeof(ITYPE));
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
    ITYPE* h = (ITYPE*)libxsmm_malloc(m * n * sizeof(ITYPE));
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
    ITYPE* i = (ITYPE*)libxsmm_malloc(m * n * sizeof(ITYPE));
    ITYPE* f = (ITYPE*)libxsmm_malloc(m * n * sizeof(ITYPE));
    ITYPE* o = (ITYPE*)libxsmm_malloc(m * n * sizeof(ITYPE));
    ITYPE* c = (ITYPE*)libxsmm_malloc(m * n * sizeof(ITYPE));
    ITYPE* d1 = (ITYPE*)libxsmm_malloc(m * n * sizeof(ITYPE));
    ITYPE* d2 = (ITYPE*)libxsmm_malloc(m * n * sizeof(ITYPE));
    ITYPE* d0 = (ITYPE*)libxsmm_malloc(m * n * sizeof(ITYPE));
    ITYPE* d = (ITYPE*)libxsmm_malloc(m * n * sizeof(ITYPE));
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
    lstm.d0 = d0;
    lstm.d1 = d1;
    lstm.d2 = d2;
    lstm.d = d;
    lstm.handlewx = handlewx;
    lstm.handleuh = handleuh;
    lstm.handlett = handlett;

    if (0 != handlewx && 0 != handleuh && 0 != handlett) {
      lstm_init(&lstm, wigold, wfgold, wogold, wcgold, xgoldt, rigold, rfgold, rogold, rcgold, hgold,
        i1gold, i2gold, f1gold, f2gold, o1gold, o2gold, c1gold, c2gold, igold, fgold, ogold, cgold,
        d0gold, d1gold, d2gold, dgold, ldw, ldx, ldz, ldu, ldh);
      lstm_execute(&lstm, nrepeat);

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
            if (j == 0) {
              matrix_eltwise_mult(m*n, fgold, d0gold, d1gold);
            } else {
              matrix_eltwise_mult(m*n, fgold, dgold, d1gold);
            }
            matrix_eltwise_mult(m*n, igold, cgold, d2gold);
            matrix_add(m*n, d1gold, d2gold, dgold);
            if (j < t-1) {
              matrix_relu(m*n, dgold, dgold); /*tanh*/
              matrix_eltwise_mult(m*n, ogold, dgold, hgold);
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
        libxsmm_free(d0gold); d0gold = 0;
        libxsmm_free(d1gold); d1gold = 0;
        libxsmm_free(d2gold); d2gold = 0;
        /* allocate C-matrix in regular format, and perform copy-out */
        dtest = (ITYPE*)libxsmm_malloc(ldz * n * sizeof(ITYPE));
        if (0 != dtest) {
          libxsmm_matdiff_info diff;
          libxsmm_bgemm_copyout_c(handleuh, d, &ldz, dtest);
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
    libxsmm_free(d0gold);
    libxsmm_free(d1gold);
    libxsmm_free(d2gold);
    libxsmm_free(dgold);
    lstm_destroy(&lstm);
  }
  fprintf(stdout, "Finished\n");

  return result;
}


int main(int argc, char* argv[])
{
  const int type = (1 < argc ? atoi(argv[1]) : 0);
  const libxsmm_blasint m = (2 < argc ? atoi(argv[2]) : 1024);
  const libxsmm_blasint k = (4 < argc ? atoi(argv[4]) : m);
  const libxsmm_blasint n = (3 < argc ? atoi(argv[3]) : k);
  const libxsmm_blasint t = (5 < argc ? atoi(argv[5]) : 3);
  const libxsmm_blasint bm = (6 < argc ? atoi(argv[6]) : 32);
  const libxsmm_blasint bk = (8 < argc ? atoi(argv[8]) : bm);
  const libxsmm_blasint bn = (7 < argc ? atoi(argv[7]) : bk);
  const libxsmm_bgemm_order order = (libxsmm_bgemm_order)(9 < argc ? atoi(argv[9]) : 0);
  const int nrepeat = (10 < argc ? atoi(argv[10]) : 100);
  const libxsmm_blasint b_m1 = (11 < argc ? atoi(argv[11]) : 1);
  const libxsmm_blasint b_n1  = (12 < argc ? atoi(argv[12]) : 1);
  const libxsmm_blasint b_k1 = (13 < argc ? atoi(argv[13]) : 1);
  const libxsmm_blasint b_k2 = (14 < argc ? atoi(argv[14]) : 1);
  const libxsmm_blasint b_m2 = (15 < argc ? atoi(argv[15]) : 1);
  const libxsmm_blasint ldw = (16 < argc ? atoi(argv[16]) : m);
  const libxsmm_blasint ldx = (17 < argc ? atoi(argv[17]) : k);
  const libxsmm_blasint ldz = (18 < argc ? atoi(argv[18]) : m);
  const libxsmm_blasint ldu = (19 < argc ? atoi(argv[19]) : m);
  const libxsmm_blasint ldh = (20 < argc ? atoi(argv[20]) : m);
  int result = EXIT_SUCCESS;
  if (argc > 1 && !strncmp(argv[1], "-h", 3)) { /* check command line */
    printf("\nUsage: ./lstm [type: 0--RNN, 1--LSTM] [M] [N] [K] [time_steps] [bm] [bn] [bk] [order] [reps] [b_m1] [b_n1] [b_k1] [b_k2] [b_m2]\n\n");
    return result;
  }
  if (type == 0) {
    fprintf(stdout, "Running RNN ...\n");
    return rnn (m, n, k, t, bm, bn, bk, order, nrepeat, b_m1, b_n1, b_k1, b_k2, b_m2, ldw, ldx, ldz, ldu, ldh);
  } else if (type == 1) {
    fprintf(stdout, "Running LSTM ...\n");
    return lstm(m, n, k, t, bm, bn, bk, order, nrepeat, b_m1, b_n1, b_k1, b_k2, b_m2, ldw, ldx, ldz, ldu, ldh);
  } else {
    fprintf(stdout, "Type %d currently not implemented!\n", type);
    return result;
  }
}

