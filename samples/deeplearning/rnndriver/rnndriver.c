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
/* Kunal Banerjee (Intel Corp.)
******************************************************************************/
#include <libxsmm.h>
#include <math.h>
#include <libxsmm_intrinsics_x86.h>

#define CHKERR_LIBXSMM_DNN(A) if ( A != LIBXSMM_DNN_SUCCESS ) fprintf(stderr, "%s\n", libxsmm_dnn_get_error(A) );

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
#if defined(_OPENMP)
# include <omp.h>
#endif

LIBXSMM_INLINE void zero_buf(float* buf, size_t size) {
  int i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < (int)size; ++i) {
    buf[i] = 0.0f;
  }
}


LIBXSMM_INLINE void init_buf(float* buf, size_t size, int initPos, int initOne)
{
  int i;
  zero_buf(buf, size);
  for (i = 0; i < (int)size; ++i) {
    buf[i] = (float)((initOne != 0) ? 1.0 : ((initPos != 0) ? libxsmm_rand_f64() : (0.05 - libxsmm_rand_f64()/10.0)));
  }
}


LIBXSMM_INLINE void matinit(int seed, float * dst,
  int nrows, int ncols, int ld, double scale)
{
  const double seed1 = scale * (seed + 1);
  int i;
#if 0
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < nrows*ncols; ++i) {
    dst[i] = (float)1;
  }
#else
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < ncols; ++i) {
    int j = 0;
    for (; j < nrows; ++j) {
      const int k = i * ld + j;
      dst[k] = (float)(seed1 / (k + 1));
    }
    for (; j < ld; ++j) {
      const int k = i * ld + j;
      dst[k] = (float)seed;
    }
  }
#endif
}


LIBXSMM_INLINE void matrix_add(int size, float *a, float *b, float *c)
{
  int i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < size; i++) {
    c[i] = a[i] + b[i];
  }
}


LIBXSMM_INLINE void matrix_eltwise_mult(int size, float *a, float *b, float *c)
{
  int i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < size; i++) {
    c[i] = a[i] * b[i];
  }
}


LIBXSMM_INLINE void matrix_sigmoid(int size, float *src, float *dst)
{
  int i;
  float exp_value;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < size; i++) {
    exp_value = (float)exp((double) -src[i]);
    dst[i] = 1 / (1 + exp_value);
  }
}


LIBXSMM_INLINE void matrix_tanh(int size, float *src, float *dst)
{
  int i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < size; i++) {
    dst[i] = (float)tanh((double)src[i]);
  }
}


LIBXSMM_INLINE void matrix_relu(int size, float *src, float *dst)
{
  int i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < size; i++) {
    dst[i] = (src[i] >= 0) ? src[i] : 0;
  }
}


LIBXSMM_INLINE void matrix_sigmoid_inverse(int size, float *src, float *dst)
{
  int i;
  float exp_value;
  float sig_exp;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < size; i++) {
    exp_value = (float)exp((double) -src[i]);
    sig_exp = 1 / (1 + exp_value);
    dst[i] = (1 - sig_exp)*sig_exp;
  }
}


LIBXSMM_INLINE void matrix_tanh_inverse(int size, float *src, float *dst)
{
  int i;
  float tanh_value;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < size; i++) {
    tanh_value = (float)tanh((double)src[i]);
    dst[i] = 1 - (tanh_value * tanh_value);
  }
}


LIBXSMM_INLINE void matrix_relu_inverse(int size, float *src, float *dst, float *input)
{
  int i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < size; i++) {
    dst[i] = (input[i] >= 0) ? src[i] : 0;
  }
}


LIBXSMM_INLINE void matrix_transpose(int rows, int cols, float *src, float *dst)
{
  int i, j;
  LIBXSMM_VLA_DECL(2, float, src2D, src, cols);
  LIBXSMM_VLA_DECL(2, float, dst2D, dst, rows);
#if defined(_OPENMP)
# pragma omp parallel for private(i, j)
#endif
  for (i = 0; i < rows; i++) {
    for (j = 0; j < cols; j++) {
      LIBXSMM_VLA_ACCESS(2, dst2D, j, i, rows) = LIBXSMM_VLA_ACCESS(2, src2D, i, j, cols);
    }
  }
}


LIBXSMM_INLINE void matrix_copy(int size, float *src, float *dst)
{
  int i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < size; i++) {
    dst[i] = src[i];
  }
}


LIBXSMM_INLINE void matrix_complement(int size, float *src, float *dst)
{
  int i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < size; i++) {
    dst[i] = 1 - src[i];
  }
}


void matrix_complement_square(int size, float *src, float *dst)
{
  int i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < size; i++) {
    dst[i] = 1 - (src[i] * src[i]);
  }
}


void libxsmm_bgemm_copyout_b(int k, int n, int blk_k, int blk_n, float *src, float *dst)
{
  LIBXSMM_VLA_DECL(4, float, real_src, src, k/blk_k, blk_n, blk_k);
  LIBXSMM_VLA_DECL(2, float, real_dst, dst, k);
  int kb, nb, bk, bn;

  for (nb = 0; nb < (n/blk_n); ++nb) {
    for (kb = 0; kb < (k/blk_k); ++kb) {
      for (bn = 0; bn < blk_n; ++bn) {
        for (bk = 0; bk < blk_k; ++bk) {
          LIBXSMM_VLA_ACCESS(2, real_dst, nb * blk_n + bn, kb * blk_k + bk, k) =
            LIBXSMM_VLA_ACCESS(4, real_src, nb, kb, bn, bk, k/blk_k, blk_n, blk_k);
        }
      }
    }
  }
}


int main(int argc, char* argv[])
{
  float *wgold, *xgoldt, *ugold, *hgold, *z1gold, *z2gold, *zgold;
  float *w, *xt, *u, *h, *htest, *hgold_temp;

  const char transa = 'N', transb = 'N'; /* no transposes */
  const int gemm_flags = LIBXSMM_GEMM_FLAGS(transa, transb);
  const float alpha = 1, beta = 1, beta0 = 0;
  void *scratch, *internalstate;
  size_t scratch_size = 0, internalstate_size = 0;

  int iters = 10;                /* repetitions of benchmark */
  int type = 0;                  /* type: 0--RNN, 1--LSTM */
  int pass = 0;                  /* pass: 0--FWD, 1--BWD, 2--UPD, 3--BWD+UPD */
  int m = 1024;                  /* number of outputs */
  int n = 512;                   /* size of mini-batch */
  int k = 256;                   /* number of inputs */
  int t = 5;                     /* number of time steps (> 1) */
  int reuse = 1;                 /* reuse=1 for FWD overwrites the same memory
                                  * for intermediate values during inference;
                                  * reuse value is immaterial for BWD and UPD */
  int bm = 32;                   /* first blocking factor for m */
  int bn = 32;                   /* first blocking factor for n */
  int bk = 32;                   /* first blocking factor for k */
  libxsmm_bgemm_order order = 0; /* denotes order of execution for bgemm */
  int b_m1 = 1;                  /* second blocking factor for m */
  int b_n1 = 1;                  /* second blocking factor for n */
  int b_k1 = 1;                  /* second blocking factor for k */
  int b_m2 = 1;                  /* third blocking factor for m */
  int b_n2 = 1;                  /* third blocking factor for n */
  int b_k2 = 1;                  /* third blocking factor for k */
  libxsmm_bgemm_handle* handlewx = 0;
  libxsmm_bgemm_handle* handleuh = 0;
  libxsmm_bgemm_handle* handlett = 0;
  libxsmm_bgemm_handle* handlewd = 0;
  const libxsmm_gemm_prefetch_type strategy = LIBXSMM_PREFETCH_AUTO;

  const char *const env_check = getenv("CHECK");
  const double check = LIBXSMM_ABS(0 == env_check ? 1 : atof(env_check));

#if defined(_OPENMP)
  int nThreads = omp_get_max_threads(); /* number of threads */
#else
  int nThreads = 1; /* number of threads */
#endif

  /*
  unsigned long long l_start, l_end;
  double l_total = 0.0;
  double flops = 0.0;
  */
  int i, it;

  libxsmm_dnn_rnncell_desc rnncell_desc;
  libxsmm_dnn_rnncell* libxsmm_handle;
  libxsmm_dnn_tensor* libxsmm_input;
  libxsmm_dnn_tensor* libxsmm_hidden_state;
  libxsmm_dnn_tensor* libxsmm_weight;
  libxsmm_dnn_tensor* libxsmm_recur_weight;
  /*
  libxsmm_dnn_tensor* libxsmm_dinput;
  libxsmm_dnn_tensor* libxsmm_dhidden_state;
  libxsmm_dnn_tensor* libxsmm_dweight;
  libxsmm_dnn_tensor* libxsmm_drecur_weight;
  */

  libxsmm_dnn_tensor_datalayout* libxsmm_layout;
  libxsmm_dnn_err_t status;
  libxsmm_dnn_err_t global_status = LIBXSMM_DNN_SUCCESS;

  libxsmm_matdiff_info norms_fwd, norms_bwd, norms_upd, diff;
  memset(&norms_fwd, 0, sizeof(norms_fwd));
  memset(&norms_bwd, 0, sizeof(norms_bwd));
  memset(&norms_upd, 0, sizeof(norms_upd));
  memset(&diff, 0, sizeof(diff));

  if (argc > 1 && !strncmp(argv[1], "-h", 3)) {
    printf("\nUsage: ./lstmdriver [reps] [type: 0--RNN, 1--LSTM] [pass: 0--FWD, 1--BWD, 2--UPD, 3--BWD+UPD] [M] [N] [K] [time_steps > 1] [reuse (for FWD): 0/1] [bm] [bn] [bk] [order] [b_m1] [b_n1] [b_k1] [b_m2] [b_n2] [b_k2]\n\n");
    return 0;
  }
  libxsmm_srand(1);

  /* reading new values from cli */
  i = 1;
  if (argc > i) iters = atoi(argv[i++]);
  if (argc > i) type  = atoi(argv[i++]);
  if (argc > i) pass  = atoi(argv[i++]);
  if (argc > i) m     = atoi(argv[i++]);
  if (argc > i) n     = atoi(argv[i++]);
  if (argc > i) k     = atoi(argv[i++]);
  if (argc > i) t     = atoi(argv[i++]);
  if (argc > i) reuse = atoi(argv[i++]);
  if (argc > i) bm    = atoi(argv[i++]);
  if (argc > i) bn    = atoi(argv[i++]);
  if (argc > i) bk    = atoi(argv[i++]);
  if (argc > i) order = (libxsmm_bgemm_order)(atoi(argv[i++]));
  if (argc > i) b_m1  = atoi(argv[i++]);
  if (argc > i) b_n1  = atoi(argv[i++]);
  if (argc > i) b_k1  = atoi(argv[i++]);
  if (argc > i) b_m2  = atoi(argv[i++]);
  if (argc > i) b_n2  = atoi(argv[i++]);
  if (argc > i) b_k2  = atoi(argv[i++]);

  if (t <= 1) {
    printf("time_steps %d should be greater than 1\n\n", t);
    return 0;
  }
  if (!(pass == 0 || pass == 1 || pass == 2 || pass == 3)) {
    printf("Unknown pass: %d, valid arguments for pass = {0(FWD), 1(BWD), 2(UPD), 3(BWD+UPD)\n\n", pass);
    return 0;
  }
  if ( (pass == 1 || pass == 2 || pass == 3) &&
      !(bm == bn && bn == bk && b_m1 == b_n1 && b_n1 == b_k1 && b_m2 == b_n2 && b_n2 == b_k2) ) {
    printf("Required condition for BWD/UPD, bm == bn == bk and bm1 == bn1 == bk1 and bm2 == bn2 == bk2\n\n");
    return 0;
  }

#if defined(__SSE3__)
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_ROUNDING_MODE(_MM_ROUND_NEAREST);
#endif

  /* print some summary */
  printf("##########################################\n");
  printf("#          Setting Up (Common)           #\n");
  printf("##########################################\n");
  printf("PARAMS: M:%d  N:%d  K:%d  T:%d\n", m, n, k, t);
  printf("PARAMS: ITERS:%d", iters); if (LIBXSMM_FEQ(0, check)) printf("  Threads:%d\n", nThreads); else printf("\n");
  printf("SIZE Weight (MB): %10.2f MiB\n", (double)(m*k*sizeof(float))/(1024.0*1024.0) );
  printf("SIZE Input (MB): %10.2f MiB\n", (double)(k*n*sizeof(float))/(1024.0*1024.0) );
  printf("SIZE Hidden State: %10.2f MiB\n", (double)(m*n*sizeof(float))/(1024.0*1024.0) );

  /* allocate data */
  wgold  = (float*)libxsmm_aligned_malloc( m*k*sizeof(float), 2097152);
  xgoldt = (float*)libxsmm_aligned_malloc( k*n*t*sizeof(float), 2097152);
  ugold  = (float*)libxsmm_aligned_malloc( m*m*sizeof(float), 2097152);
  hgold  = (float*)libxsmm_aligned_malloc( m*n*sizeof(float), 2097152);
  z1gold = (float*)libxsmm_aligned_malloc( m*n*sizeof(float), 2097152);
  z2gold = (float*)libxsmm_aligned_malloc( m*n*sizeof(float), 2097152);
  zgold  = (float*)libxsmm_aligned_malloc( m*n*sizeof(float), 2097152);
  w      = (float*)libxsmm_aligned_malloc( m*k*sizeof(float), 2097152);
  xt     = (float*)libxsmm_aligned_malloc( k*n*t*sizeof(float), 2097152);
  u      = (float*)libxsmm_aligned_malloc( m*m*sizeof(float), 2097152);
  if (reuse) {
    h      = (float*)libxsmm_aligned_malloc( m*n*sizeof(float), 2097152);
  } else {
    h      = (float*)libxsmm_aligned_malloc( m*n*(t+1)*sizeof(float), 2097152);
  }
  htest  = (float*)libxsmm_aligned_malloc( m*n*sizeof(float), 2097152);
  hgold_temp = (float*)libxsmm_aligned_malloc( m*n*sizeof(float), 2097152);

  /* initialize data */
  LIBXSMM_VLA_DECL(2, float, xgold, xgoldt, k * n);
  matinit(42, wgold, m, k, m, 1.0);
  for (it = 0; it < t; ++it) {
    matinit(24, &LIBXSMM_VLA_ACCESS(2, xgold, it, 0, k * n), k, n, k, 1.0);
  }
  matinit(42, ugold, m, m, m, 1.0);
  matinit(24, hgold, m, n, m, 1.0);
  matrix_copy(m*n, hgold, hgold_temp); /* Required because hgold may get overwritten */
  zero_buf(z1gold, m*n);
  zero_buf(z2gold, m*n);
  zero_buf(zgold, m*n);

  /* first touch LIBXSMM */
  zero_buf( w,  m*k );
  zero_buf( xt, k*n*t );
  LIBXSMM_VLA_DECL(2, float, x, xt, k * n);
  zero_buf( u,  m*m );
  if (reuse) {
    zero_buf( h,  m*n );
  } else {
    zero_buf( h,  m*n*(t+1) );
  }
  LIBXSMM_VLA_DECL(2, float, hnr, h, m * n);

  handlewx = libxsmm_bgemm_handle_create(LIBXSMM_GEMM_PRECISION(float), LIBXSMM_GEMM_PRECISION(float),
      m, n, k, &bm, &bn, &bk, &b_m1, &b_n1, &b_k1, &b_k2,
      &alpha, &beta, &gemm_flags, &strategy, &order);
  handleuh = libxsmm_bgemm_handle_create(LIBXSMM_GEMM_PRECISION(float), LIBXSMM_GEMM_PRECISION(float),
      m, n, m, &bm, &bn, &bm, &b_m1, &b_n1, &b_m1, &b_m2,
      &alpha, &beta, &gemm_flags, &strategy, &order);
  handlett = libxsmm_bgemm_handle_create(LIBXSMM_GEMM_PRECISION(float), LIBXSMM_GEMM_PRECISION(float),
      m, n*t, k, &bm, &bn, &bk, &b_m1, &b_n1, &b_k1, &b_k2,
      &alpha, &beta, &gemm_flags, &strategy, &order);

  if (LIBXSMM_NEQ(0, check)) {
    printf("##########################################\n");
    printf("#         Computing Reference ...        #\n");
    printf("##########################################\n");
    if (pass == 0) {
      for (i = 0; i < t; ++i) {
        LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &m, &n, &k, &alpha, wgold, &m, &LIBXSMM_VLA_ACCESS(2, xgold, i, 0, k * n), &k, &beta0, z1gold, &m);
        LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &m, &n, &m, &alpha, ugold, &m, hgold, &m, &beta0, z2gold, &m);
        matrix_add(m*n, z1gold, z2gold, zgold);
        matrix_sigmoid(m*n, zgold, hgold);
      }
    }
    /*
    if (pass == 1 || pass == 2 || pass == 3) {
      zero_buf(&LIBXSMM_VLA_ACCESS(2, deltagold, t-1, 0, m * n), m*n);
      matrix_transpose(m, m, ugold, ugoldTp);
      for (i = t-2; i >= 0; --i) {
        matrix_sigmoid_inverse(m * n, &LIBXSMM_VLA_ACCESS(2, zgold, i+1, 0, m * n), zigold);
        LIBXSMM_XBLAS_SYMBOL(ITYPE)(&transa, &transb, &m, &n, &m, &alpha, ugoldTp, &m, &LIBXSMM_VLA_ACCESS(2, deltagold, i+1, 0, m * n), &m, &beta, di1gold, &m);
        matrix_add(m * n, &LIBXSMM_VLA_ACCESS(2, djdhgold, i+1, 0, m * n), di1gold, di2gold);
        matrix_eltwise_mult(m * n, zigold, di2gold, &LIBXSMM_VLA_ACCESS(2, deltagold, i, 0, m * n));
      }
      if (pass == 1 || pass == 3) {
        matrix_transpose(m, k, wgold, wgoldTp);
        for (i = 0; i < t; ++i) {
          LIBXSMM_XBLAS_SYMBOL(ITYPE)(&transa, &transb, &k, &n, &m, &alpha, wgoldTp, &k, &LIBXSMM_VLA_ACCESS(2, deltagold, i, 0, m * n), &m, &beta, &LIBXSMM_VLA_ACCESS(2, djdxgold, i, 0, k * n), &k);
        }
      }
      if (pass == 2 || pass == 3) {
        for (i = 0; i < t; ++i) {
          matrix_transpose(m, n, &LIBXSMM_VLA_ACCESS(2, hgold, i, 0, m * n), hgoldTp);
          LIBXSMM_XBLAS_SYMBOL(ITYPE)(&transa, &transb, &m, &m, &n, &alpha, &LIBXSMM_VLA_ACCESS(2, deltagold, i, 0, m * n), &m, hgoldTp, &n, &beta, dj1gold, &m);
          matrix_add(m*m, dj1gold, djdugold, djdugold);
          matrix_transpose(k, n, &LIBXSMM_VLA_ACCESS(2, xgold, i, 0, k * n), xgoldTp);
          LIBXSMM_XBLAS_SYMBOL(ITYPE)(&transa, &transb, &m, &k, &n, &alpha, &LIBXSMM_VLA_ACCESS(2, deltagold, i, 0, m * n), &m, xgoldTp, &n, &beta, dw1gold, &m);
          matrix_add(m*k, dw1gold, djdwgold, djdwgold);
        }
      }
    }
    */
    printf("##########################################\n");
    printf("#      Computing Reference ... done      #\n");
    printf("##########################################\n");
  }

  if (1 /* format == 'A' || format == 'L' */) {
    printf("\n");
    printf("##########################################\n");
    printf("#      Setting Up  (custom-Storage)      #\n");
    printf("##########################################\n");

    /* setup LIBXSMM handle */
    rnncell_desc.nThreads = nThreads;
    rnncell_desc.m = m;
    rnncell_desc.n = n;
    rnncell_desc.k = k;
    rnncell_desc.t = t;
    rnncell_desc.bm = bm;
    rnncell_desc.bn = bn;
    rnncell_desc.bk = bk;
    rnncell_desc.b_m1 = b_m1;
    rnncell_desc.b_n1 = b_n1;
    rnncell_desc.b_k1 = b_k1;
    rnncell_desc.b_m2 = b_m2;
    rnncell_desc.b_n2 = b_n2;
    rnncell_desc.b_k2 = b_k2;
    rnncell_desc.reuse = reuse;
    rnncell_desc.datatype_in = LIBXSMM_DNN_DATATYPE_F32;
    rnncell_desc.datatype_out = LIBXSMM_DNN_DATATYPE_F32;
    rnncell_desc.buffer_format = LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM;
    rnncell_desc.handlewx = handlewx;
    rnncell_desc.handleuh = handleuh;
    rnncell_desc.handlett = handlett;
    rnncell_desc.handlewd = handlewd;

    libxsmm_handle = libxsmm_dnn_create_rnncell( rnncell_desc, &status );
    CHKERR_LIBXSMM_DNN( status );

    /* setup LIBXSMM buffers and filter */
    libxsmm_layout = libxsmm_dnn_rnncell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_RNN_REGULAR_INPUT, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_input = libxsmm_dnn_link_tensor( libxsmm_layout, xt, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_rnncell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_RNN_REGULAR_HIDDEN_STATE, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_hidden_state = libxsmm_dnn_link_tensor( libxsmm_layout, h, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_rnncell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_RNN_REGULAR_WEIGHT, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_weight = libxsmm_dnn_link_tensor( libxsmm_layout, w, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_rnncell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_RNN_REGULAR_RECUR_WEIGHT, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_recur_weight = libxsmm_dnn_link_tensor( libxsmm_layout, u, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    /* copy in data to LIBXSMM format */
    CHKERR_LIBXSMM_DNN( libxsmm_bgemm_copyin_a(handlewx, wgold, &m, w) );
    for (it = 0; it < t; ++it) {
      CHKERR_LIBXSMM_DNN( libxsmm_bgemm_copyin_b(handlewx, &LIBXSMM_VLA_ACCESS(2, xgold, it, 0, k * n), &k, &LIBXSMM_VLA_ACCESS(2, x, it, 0, k * n)) );
    }
    CHKERR_LIBXSMM_DNN( libxsmm_bgemm_copyin_a(handleuh, ugold, &m, u) );
    if (reuse) {
      CHKERR_LIBXSMM_DNN( libxsmm_bgemm_copyin_b(handleuh, hgold_temp, &m, h) );
    } else {
      CHKERR_LIBXSMM_DNN( libxsmm_bgemm_copyin_b(handleuh, hgold_temp, &m, &LIBXSMM_VLA_ACCESS(2, hnr, 0, 0, m * n)) );
      zero_buf(&LIBXSMM_VLA_ACCESS(2, hnr, 1, 0, m * n), m * n * t);
    }

    /* bind buffers and filter to handle */
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_bind_tensor( libxsmm_handle, libxsmm_input, LIBXSMM_DNN_RNN_REGULAR_INPUT ) );
    /* CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_bind_tensor( libxsmm_handle, libxsmm_dinput, LIBXSMM_DNN_RNN_GRADIENT_INPUT ) ); */
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_bind_tensor( libxsmm_handle, libxsmm_hidden_state, LIBXSMM_DNN_RNN_REGULAR_HIDDEN_STATE ) );
    /* CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_bind_tensor( libxsmm_handle, libxsmm_dhidden_state, LIBXSMM_DNN_RNN_GRADIENT_HIDDEN_STATE ) ); */
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_bind_tensor( libxsmm_handle, libxsmm_weight, LIBXSMM_DNN_RNN_REGULAR_WEIGHT ) );
    /* CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_bind_tensor( libxsmm_handle, libxsmm_dweight, LIBXSMM_DNN_RNN_GRADIENT_WEIGHT ) ); */
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_bind_tensor( libxsmm_handle, libxsmm_recur_weight, LIBXSMM_DNN_RNN_REGULAR_RECUR_WEIGHT ) );
    /* CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_bind_tensor( libxsmm_handle, libxsmm_drecur_weight, LIBXSMM_DNN_RNN_GRADIENT_RECUR_WEIGHT ) ); */

    /* let's allocate and bind scratch */
    scratch_size = libxsmm_dnn_rnncell_get_scratch_size( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_FWD, &status ); //KB //ALL
    CHKERR_LIBXSMM_DNN( status );
    scratch = libxsmm_aligned_malloc( scratch_size, 2097152 );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_bind_scratch( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_FWD, scratch ) ); //KB //ALL
    /* set scratch to bogus to make sure that libxsmm takes care of zeroing internally */
    zero_buf( (float*)scratch, scratch_size/4 );
    
    /* let's allocate and bind scratch */
    internalstate_size = libxsmm_dnn_rnncell_get_internalstate_size( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_FWD, &status ); //KB //ALL
    CHKERR_LIBXSMM_DNN( status );
    internalstate = libxsmm_aligned_malloc( internalstate_size, 2097152 );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_bind_internalstate( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_FWD, internalstate ) ); //KB //ALL
    zero_buf( (float*)internalstate, internalstate_size/4 );

    if ((pass == 0) && LIBXSMM_NEQ(0, check)) {
      printf("##########################################\n");
      printf("#   Correctness - FWD (custom-Storage)   #\n");
      printf("##########################################\n");
      /* run LIBXSMM RNN */
#if defined(_OPENMP)
#     pragma omp parallel
#endif
      {
#if defined(_OPENMP)
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_FWD, 0, tid ) );
      }
      /* copy out data */
      if (reuse) {
        /* CHKERR_LIBXSMM_DNN( libxsmm_bgemm_copyout_b( k, n, bk, bn, h, htest ) ); */
        libxsmm_bgemm_copyout_b( m, n, bm, bn, h, htest );
      } else {
        /* CHKERR_LIBXSMM_DNN( libxsmm_bgemm_copyout_b( k, n, bk, bn, &LIBXSMM_VLA_ACCESS(2, hnr, t, 0, m * n), htest ) ); */
        libxsmm_bgemm_copyout_b( m, n, bm, bn, &LIBXSMM_VLA_ACCESS(2, hnr, t, 0, m * n), htest );
      }

      /* compare */
      libxsmm_matdiff(LIBXSMM_DATATYPE_F32, m*n, 1, hgold, htest, 0, 0, &norms_fwd);
      printf("L1 reference  : %.25g\n", norms_fwd.l1_ref);
      printf("L1 test       : %.25g\n", norms_fwd.l1_tst);
      printf("L2 abs.error  : %.24f\n", norms_fwd.l2_abs);
      printf("L2 rel.error  : %.24f\n", norms_fwd.l2_rel);
      printf("Linf abs.error: %.24f\n", norms_fwd.linf_abs);
      printf("Linf rel.error: %.24f\n", norms_fwd.linf_rel);
      printf("Check-norm    : %.24f\n", norms_fwd.normf_rel);
      libxsmm_matdiff_reduce(&diff, &norms_fwd);
    }
#if 0
    if ( (pass == 1 || pass == 3) && LIBXSMM_NEQ(0, check) ) {
      printf("##########################################\n");
      printf("#   Correctness - BWD (custom-Storage)   #\n");
      printf("##########################################\n");
      /* let's do some additional init such that we can run passes standalone */
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyin_tensor(    libxsmm_doutput, (void*)naive_output_bp, LIBXSMM_DNN_TENSOR_FORMAT_NCHW ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyin_tensor(    libxsmm_dinput, (void*)naive_input_save, LIBXSMM_DNN_TENSOR_FORMAT_NCHW ) );
#if defined(USE_BWD_NO_FILTER_TRANSPOSE_OVERWRITE)
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_trans_reg_filter( libxsmm_handle ) );
#endif

      /* run LIBXSMM convolutions */
#if defined(_OPENMP)
#     pragma omp parallel
#endif
      {
#if defined(_OPENMP)
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        CHKERR_LIBXSMM_DNN( libxsmm_dnn_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_BWD, 0, tid ) );
      }

      /* copy out data */
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyout_tensor( libxsmm_dinput, (void*)naive_libxsmm_input, LIBXSMM_DNN_TENSOR_FORMAT_NCHW ) );

      /* compare */
      libxsmm_matdiff(LIBXSMM_DATATYPE_F32, nImg*nIfm*ifhp*ifwp, 1, naive_input, naive_libxsmm_input, 0, 0, &norms_bwd);
      printf("L1 reference  : %.25g\n", norms_bwd.l1_ref);
      printf("L1 test       : %.25g\n", norms_bwd.l1_tst);
      printf("L2 abs.error  : %.24f\n", norms_bwd.l2_abs);
      printf("L2 rel.error  : %.24f\n", norms_bwd.l2_rel);
      printf("Linf abs.error: %.24f\n", norms_bwd.linf_abs);
      printf("Linf rel.error: %.24f\n", norms_bwd.linf_rel);
      printf("Check-norm    : %.24f\n", norms_bwd.normf_rel);
      libxsmm_matdiff_reduce(&diff, &norms_bwd);
    }

    if ((type == 'A' || type == 'U') && LIBXSMM_NEQ(0, check)) {
      printf("##########################################\n");
      printf("#   Correctness - UPD (custom-Storage)   #\n");
      printf("##########################################\n");
      /* let's do some additional init such that we can run passes standalone */
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyin_tensor( libxsmm_input, (void*)naive_input_save, LIBXSMM_DNN_TENSOR_FORMAT_NCHW ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyin_tensor( libxsmm_doutput, (void*)naive_output_wu, LIBXSMM_DNN_TENSOR_FORMAT_NCHW ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyin_tensor( libxsmm_dfilter, (void*)naive_filter, LIBXSMM_DNN_TENSOR_FORMAT_KCRS ) );
      /* run LIBXSMM convolutions */
#if defined(_OPENMP)
#     pragma omp parallel
#endif
      {
#if defined(_OPENMP)
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        CHKERR_LIBXSMM_DNN( libxsmm_dnn_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_UPD, 0, tid ) );
      }
      if (conv_desc.options == LIBXSMM_DNN_CONV_OPTION_UPD_NO_FILTER_REDUCE) {
        CHKERR_LIBXSMM_DNN( libxsmm_dnn_reduce_wu_filters( libxsmm_handle, LIBXSMM_DNN_GRADIENT_FILTER ) );
      }
      /* copy out data */
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyout_tensor( libxsmm_dfilter, (void*)naive_libxsmm_filter, LIBXSMM_DNN_TENSOR_FORMAT_KCRS ) );

      /* compare */
      libxsmm_matdiff(LIBXSMM_DATATYPE_F32, nOfm*nIfm*kh*kw, 1, naive_filter_wu, naive_libxsmm_filter, 0, 0, &norms_upd);
      printf("L1 reference  : %.25g\n", norms_upd.l1_ref);
      printf("L1 test       : %.25g\n", norms_upd.l1_tst);
      printf("L2 abs.error  : %.24f\n", norms_upd.l2_abs);
      printf("L2 rel.error  : %.24f\n", norms_upd.l2_rel);
      printf("Linf abs.error: %.24f\n", norms_upd.linf_abs);
      printf("Linf rel.error: %.24f\n", norms_upd.linf_rel);
      printf("Check-norm    : %.24f\n", norms_upd.normf_rel);
      libxsmm_matdiff_reduce(&diff, &norms_upd);
    }

    if (pass == 2 || pass == 3) {
      printf("##########################################\n");
      printf("#   Performance - FWD (custom-Storage)   #\n");
      printf("##########################################\n");
      /* run LIBXSMM convolution for performance */
      l_start = libxsmm_timer_tick();
#if defined(_OPENMP)
#     pragma omp parallel private(i)
#endif
      {
#if defined(_OPENMP)
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        for (i = 0; i < iters; ++i) {
          libxsmm_dnn_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_FWD, 0, tid );
        }
      }
      l_end = libxsmm_timer_tick();
      l_total = libxsmm_timer_duration(l_start, l_end);
      flops = (double)nImg * (double)nIfm * (double)nOfm * (double)ofh * (double)ofw * (double)(2 * kh * kw) * (double)iters;

      printf("GFLOP  = %.5g\n", flops*1e-9/(double)iters);
      printf("fp time = %.5g\n", ((double)(l_total/iters)));
      printf("GFLOPS  = %.5g\n", (flops*1e-9)/l_total);

      printf("PERFDUMP,FP,%s,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%.5g,%.5g,%f,%f,%f,%f,%f,%f,%f\n", LIBXSMM_VERSION, nThreads, nImg, nIfm, nOfm,
        ifw, ifh, kw, kh, stride, padw, padh, ((double)(l_total/iters)), (flops*1e-9)/l_total, norms_fwd.l1_ref, norms_fwd.l1_tst,
        norms_fwd.l2_abs, norms_fwd.l2_rel, norms_fwd.linf_abs, norms_fwd.linf_rel, norms_fwd.normf_rel);
    }

    if ( (type == 'A' || type == 'B') && (nIfm > 3) ) {
      printf("##########################################\n");
      printf("#   Performance - BWD (custom-Storage)   #\n");
      printf("##########################################\n");
      /* run LIBXSMM convolution for performance */
      l_start = libxsmm_timer_tick();

#if defined(_OPENMP)
#     pragma omp parallel  private(i)
#endif
      {
#if defined(_OPENMP)
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        for (i = 0; i < iters; ++i) {
          libxsmm_dnn_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_BWD, 0, tid );
        }
      }
      l_end = libxsmm_timer_tick();
      l_total = libxsmm_timer_duration(l_start, l_end);
      flops = (double)nImg * (double)nIfm * (double)nOfm * (double)ofh * (double)ofw * (double)(2 * kh * kw) * (double)iters;

      printf("GFLOP  = %.5g\n", flops*1e-9/(double)iters);
      printf("bp time = %.5g\n", ((double)(l_total/iters)));
      printf("GFLOPS  = %.5g\n", (flops*1e-9)/l_total);

      printf("PERFDUMP,BP,%s,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%.5g,%.5g,%f,%f,%f,%f,%f,%f,%f\n", LIBXSMM_VERSION, nThreads, nImg, nIfm, nOfm,
        ifw, ifh, kw, kh, stride, padw, padh, ((double)(l_total/iters)), (flops*1e-9)/l_total, norms_bwd.l1_ref, norms_bwd.l1_tst,
        norms_bwd.l2_abs, norms_bwd.l2_rel, norms_bwd.linf_abs, norms_bwd.linf_rel, norms_bwd.normf_rel);
    }

    if (type == 'A' || type == 'U') {
      printf("##########################################\n");
      printf("#   Performance - UPD (custom-Storage)   #\n");
      printf("##########################################\n");
      /* run LIBXSMM convolution for performance */
      l_start = libxsmm_timer_tick();

#if defined(_OPENMP)
#     pragma omp parallel private(i)
#endif
      {
#if defined(_OPENMP)
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        for (i = 0; i < iters; ++i) {
          libxsmm_dnn_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_UPD, 0, tid );
          if (conv_desc.options == LIBXSMM_DNN_CONV_OPTION_UPD_NO_FILTER_REDUCE) {
            CHKERR_LIBXSMM_DNN( libxsmm_dnn_reduce_wu_filters( libxsmm_handle, LIBXSMM_DNN_GRADIENT_FILTER ) );
          }
        }
      }
      l_end = libxsmm_timer_tick();
      l_total = libxsmm_timer_duration(l_start, l_end);
      flops = (double)nImg * (double)nIfm * (double)nOfm * (double)ofh * (double)ofw * (double)(2 * kh * kw) * (double)iters;

      printf("GFLOP  = %.5g\n", flops*1e-9/(double)iters);
      printf("wu time = %.5g\n", ((double)(l_total/iters)));
      printf("GFLOPS  = %.5g\n", (flops*1e-9)/l_total);

      printf("PERFDUMP,WU,%s,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%.5g,%.5g,%f,%f,%f,%f,%f,%f,%f\n", LIBXSMM_VERSION, nThreads, nImg, nIfm, nOfm,
        ifw, ifh, kw, kh, stride, padw, padh, ((double)(l_total/iters)), (flops*1e-9)/l_total, norms_upd.l1_ref, norms_upd.l1_tst,
        norms_upd.l2_abs, norms_upd.l2_rel, norms_upd.linf_abs, norms_upd.linf_rel, norms_upd.normf_rel);
    }

    /* clean-up */
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_release_scratch( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_ALL ) );
    libxsmm_free(scratch);
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_release_tensor( libxsmm_handle, LIBXSMM_DNN_REGULAR_INPUT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_release_tensor( libxsmm_handle, LIBXSMM_DNN_GRADIENT_INPUT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_release_tensor( libxsmm_handle, LIBXSMM_DNN_REGULAR_OUTPUT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_release_tensor( libxsmm_handle, LIBXSMM_DNN_GRADIENT_OUTPUT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_release_tensor( libxsmm_handle, LIBXSMM_DNN_REGULAR_FILTER ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_release_tensor( libxsmm_handle, LIBXSMM_DNN_GRADIENT_FILTER ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_release_tensor( libxsmm_handle, LIBXSMM_DNN_REGULAR_BIAS ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_release_tensor( libxsmm_handle, LIBXSMM_DNN_GRADIENT_BIAS ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_release_tensor( libxsmm_handle, LIBXSMM_DNN_REGULAR_FILTER_TRANS ) );
#ifdef USE_FUSED_BATCH_STATS
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_release_tensor( libxsmm_handle, LIBXSMM_DNN_BATCH_STATS ) );
#endif
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_input ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_output ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_filter ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_dinput ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_doutput ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_dfilter ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_bias ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_dbias ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_filter_tr ) );
#ifdef USE_FUSED_BATCH_STATS
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_batchstats ) );
#endif
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_conv_layer( libxsmm_handle ) );
#endif //KB
  }

  /* deallocate data */
  libxsmm_free(wgold);
  libxsmm_free(xgoldt);
  libxsmm_free(ugold);
  libxsmm_free(hgold);
  libxsmm_free(z1gold);
  libxsmm_free(z2gold);
  libxsmm_free(zgold);
  libxsmm_free(w);
  libxsmm_free(xt);
  libxsmm_free(u);
  libxsmm_free(h);
  libxsmm_free(htest);

  { const char *const env_check_scale = getenv("CHECK_SCALE");
    const double check_scale = LIBXSMM_ABS(0 == env_check_scale ? 1.0 : atof(env_check_scale));
    if (LIBXSMM_NEQ(0, check) && (check < 100.0 * check_scale * diff.normf_rel) && (global_status == LIBXSMM_DNN_SUCCESS)) {
      fprintf(stderr, "FAILED with an error of %f%%!\n", 100.0 * diff.normf_rel);
      exit(EXIT_FAILURE);
    }
  }

  /* some empty lines at the end */
  printf("\n\n\n");

  return EXIT_SUCCESS;
}

