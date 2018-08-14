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
#include <libxsmm.h>
#include <libxsmm_intrinsics_x86.h>

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#if defined(_OPENMP)
# include <omp.h>
#endif
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

#define CHKERR_LIBXSMM_DNN(A) if ( A != LIBXSMM_DNN_SUCCESS ) fprintf(stderr, "%s\n", libxsmm_dnn_get_error(A) );


LIBXSMM_INLINE void zero_buf(float* buf, size_t size) {
  int i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < (int)size; ++i) {
    buf[i] = 0.0f;
  }
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
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < size; i++) {
    const float exp_value = (float)exp((double) -src[i]);
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
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < size; i++) {
    const float exp_value = (float)exp((double) -src[i]);
    const float sig_exp = 1 / (1 + exp_value);
    dst[i] = (1 - sig_exp)*sig_exp;
  }
}


LIBXSMM_INLINE void matrix_tanh_inverse(int size, float *src, float *dst)
{
  int i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < size; i++) {
    const float tanh_value = (float)tanh((double)src[i]);
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


LIBXSMM_INLINE void matrix_complement_square(int size, float *src, float *dst)
{
  int i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < size; i++) {
    dst[i] = 1 - (src[i] * src[i]);
  }
}


LIBXSMM_INLINE void libxsmm_bgemm_copyout_b(int k, int n, int blk_k, int blk_n, float *src, float *dst)
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
  /* Arrays related to FWD pass */
  float *wgold, *xgoldt, *ugold, *hgold = NULL, *z1gold = NULL, *z2gold = NULL, *zgold = NULL;
  float *w, *xt, *u, *h = NULL, *htest = NULL, *hgold_temp = NULL;
  /* Arrays related to BWD and UPD pass */
  float *djdhgoldt = NULL, *zgoldt = NULL, *deltagoldt = NULL, *hgoldt = NULL, *djdugold = NULL, *djdwgold = NULL, *djdxgoldt = NULL;
  float *zigold = NULL, *di1gold = NULL, *di2gold = NULL, *dj1gold = NULL, *dw1gold = NULL, *ugoldTp = NULL, *wgoldTp = NULL, *hgoldTp = NULL, *xgoldTp = NULL;
  float *djdht = NULL, *ht = NULL, *djdu = NULL, *djdw = NULL, *djdxt = NULL, *djdxtestt = NULL, *djdwtest = NULL, *djdutest = NULL;

  const char transa = 'N', transb = 'N'; /* no transposes */
  const int gemm_flags = LIBXSMM_GEMM_FLAGS(transa, transb);
  const float alpha = 1, beta = 1, beta0 = 0;
  void *scratch, *internalstate;
  size_t scratch_size = 0, internalstate_size = 0;

  int iters = 10; /* repetitions of benchmark */
  int pass = 3;   /* pass: 0--FWD, 1--BWD, 2--UPD, 3--BWD+UPD */
  int m = 1024;   /* number of outputs */
  int n = 512;    /* size of mini-batch */
  int k = 256;    /* number of inputs */
  int t = 5;      /* number of time steps (> 1) */
  int reuse = 1;  /* reuse=1 for FWD overwrites the same memory
                   * for intermediate values during inference;
                   * reuse value is immaterial for BWD and UPD */
  int bm = 32;    /* first blocking factor for m */
  int bn = 32;    /* first blocking factor for n */
  int bk = 32;    /* first blocking factor for k */
  /* denotes order of execution for bgemm */
  libxsmm_bgemm_order order = LIBXSMM_BGEMM_ORDER_JIK;
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
  libxsmm_bgemm_handle* handlewx = 0;
  libxsmm_bgemm_handle* handleuh = 0;
  libxsmm_bgemm_handle* handlett = 0;
  libxsmm_bgemm_handle* handlewd = 0;
  const libxsmm_gemm_prefetch_type strategy = LIBXSMM_PREFETCH_AUTO;

  const char *const env_check = getenv("CHECK");
  const double check = LIBXSMM_ABS(0 == env_check ? 0/*disabled by default*/ : atof(env_check));

#if defined(_OPENMP)
  int nThreads = omp_get_max_threads(); /* number of threads */
#else
  int nThreads = 1; /* number of threads */
#endif

  unsigned long long l_start, l_end;
  double l_total = 0.0;
  double flops = 0.0, tempflops = 0.0;
  const double tflops = 12; /* transcendental flops */
  int i, it;

  libxsmm_dnn_rnncell_desc rnncell_desc;
  libxsmm_dnn_rnncell* libxsmm_handle;
  libxsmm_dnn_tensor* libxsmm_input;
  libxsmm_dnn_tensor* libxsmm_hidden_state;
  libxsmm_dnn_tensor* libxsmm_weight;
  libxsmm_dnn_tensor* libxsmm_recur_weight;
  libxsmm_dnn_tensor* libxsmm_dinput = NULL;
  libxsmm_dnn_tensor* libxsmm_dhidden_state = NULL;
  libxsmm_dnn_tensor* libxsmm_dweight = NULL;
  libxsmm_dnn_tensor* libxsmm_drecur_weight = NULL;

  libxsmm_dnn_tensor_datalayout* libxsmm_layout;
  libxsmm_dnn_err_t status;
  libxsmm_dnn_err_t global_status = LIBXSMM_DNN_SUCCESS;

  libxsmm_matdiff_info norms_fwd, norms_bwd, norms_upd_w, norms_upd_u, diff;
  memset(&norms_fwd, 0, sizeof(norms_fwd));
  memset(&norms_bwd, 0, sizeof(norms_bwd));
  memset(&norms_upd_w, 0, sizeof(norms_upd_w));
  memset(&norms_upd_u, 0, sizeof(norms_upd_u));
  memset(&diff, 0, sizeof(diff));

  if (argc > 1 && !strncmp(argv[1], "-h", 3)) {
    printf("\nUsage: ./rnndriver [reps] [pass: 0--FWD, 1--BWD, 2--UPD, 3--BWD+UPD] [M] [N] [K] [time_steps > 1] [reuse (for FWD): 0/1] [bm] [bn] [bk]\n\n");
    return 0;
  }
  libxsmm_srand(1);

  /* reading new values from cli */
  i = 1;
  if (argc > i) iters = atoi(argv[i++]);
  if (argc > i) pass  = atoi(argv[i++]);
  if (argc > i) m     = atoi(argv[i++]);
  if (argc > i) n     = atoi(argv[i++]);
  if (argc > i) k     = atoi(argv[i++]);
  if (argc > i) t     = atoi(argv[i++]);
  if (argc > i) reuse = atoi(argv[i++]);
  if (argc > i) bm    = atoi(argv[i++]);
  if (argc > i) bn    = atoi(argv[i++]);
  if (argc > i) bk    = atoi(argv[i++]);

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
  if (pass == 0) {
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
  } else {
    djdhgoldt  = (float*)libxsmm_aligned_malloc(m * n * sizeof(float) * t, 2097152);
    zgoldt     = (float*)libxsmm_aligned_malloc(m * n * sizeof(float) * t, 2097152);
    deltagoldt = (float*)libxsmm_aligned_malloc(m * n * sizeof(float) * t, 2097152);
    ugold      = (float*)libxsmm_aligned_malloc(m * m * sizeof(float), 2097152);
    xgoldt     = (float*)libxsmm_aligned_malloc(k * n * sizeof(float) * t, 2097152);
    hgoldt     = (float*)libxsmm_aligned_malloc(m * n * sizeof(float) * t, 2097152);
    djdugold   = (float*)libxsmm_aligned_malloc(m * m * sizeof(float), 2097152);
    djdwgold   = (float*)libxsmm_aligned_malloc(m * k * sizeof(float), 2097152);
    djdxgoldt  = (float*)libxsmm_aligned_malloc(k * n * sizeof(float) * t, 2097152);
    wgold      = (float*)libxsmm_aligned_malloc(m * k * sizeof(float), 2097152);
    zigold     = (float*)libxsmm_aligned_malloc(m * n * sizeof(float), 2097152);
    di1gold    = (float*)libxsmm_aligned_malloc(m * n * sizeof(float), 2097152);
    di2gold    = (float*)libxsmm_aligned_malloc(m * n * sizeof(float), 2097152);
    dj1gold    = (float*)libxsmm_aligned_malloc(m * m * sizeof(float), 2097152);
    dw1gold    = (float*)libxsmm_aligned_malloc(m * k * sizeof(float), 2097152);
    ugoldTp    = (float*)libxsmm_aligned_malloc(m * m * sizeof(float), 2097152);
    wgoldTp    = (float*)libxsmm_aligned_malloc(m * k * sizeof(float), 2097152);
    hgoldTp    = (float*)libxsmm_aligned_malloc(m * n * sizeof(float), 2097152);
    xgoldTp    = (float*)libxsmm_aligned_malloc(k * n * sizeof(float), 2097152);
    w          = (float*)libxsmm_aligned_malloc(m * k * sizeof(float), 2097152);
    xt         = (float*)libxsmm_aligned_malloc(k * n * sizeof(float) * t, 2097152);
    u          = (float*)libxsmm_aligned_malloc(m * m * sizeof(float), 2097152);
    ht         = (float*)libxsmm_aligned_malloc(m * n * sizeof(float) * t, 2097152);
    djdw       = (float*)libxsmm_aligned_malloc(m * k * sizeof(float), 2097152);
    djdxt      = (float*)libxsmm_aligned_malloc(k * n * sizeof(float) * t, 2097152);
    djdu       = (float*)libxsmm_aligned_malloc(m * m * sizeof(float), 2097152);
    djdht      = (float*)libxsmm_aligned_malloc(m * n * sizeof(float) * t, 2097152);
    djdwtest   = (float*)libxsmm_aligned_malloc(m * k * sizeof(float), 2097152);
    djdxtestt  = (float*)libxsmm_aligned_malloc(k * n * sizeof(float) * t, 2097152);
    djdutest   = (float*)libxsmm_aligned_malloc(m * m * sizeof(float), 2097152);
  }
  LIBXSMM_VLA_DECL(2, float, djdhgold, djdhgoldt, m * n);
  LIBXSMM_VLA_DECL(2, float, djdxgold, djdxgoldt, k * n);
  LIBXSMM_VLA_DECL(2, float, zgoldb, zgoldt, m * n);
  LIBXSMM_VLA_DECL(2, float, deltagold, deltagoldt, m * n);
  LIBXSMM_VLA_DECL(2, float, xgold, xgoldt, k * n);
  LIBXSMM_VLA_DECL(2, float, hgoldb, hgoldt, m * n);
  LIBXSMM_VLA_DECL(2, float, djdh, djdht, m * n);
  /*LIBXSMM_VLA_DECL(2, float, xb, xt, k * n);*/
  LIBXSMM_VLA_DECL(2, float, hb, ht, m * n);
  LIBXSMM_VLA_DECL(2, float, djdx, djdxt, k * n);

  /* initialize data */
  if (pass == 0) {
    LIBXSMM_MATINIT(float, 42, wgold, m, k, m, 1.0);
    for (it = 0; it < t; ++it) {
      LIBXSMM_MATINIT(float, 24, &LIBXSMM_VLA_ACCESS(2, xgold, it, 0, k * n), k, n, k, 1.0);
    }
    LIBXSMM_MATINIT(float, 42, ugold, m, m, m, 1.0);
    LIBXSMM_MATINIT(float, 24, hgold, m, n, m, 1.0);
    matrix_copy(m*n, hgold, hgold_temp); /* Required because hgold may get overwritten */
    zero_buf(z1gold, m*n);
    zero_buf(z2gold, m*n);
    zero_buf(zgold, m*n);
  } else {
    LIBXSMM_MATINIT(float, 42, ugold, m, m, m, 1.0);
    LIBXSMM_MATINIT(float, 42, wgold, m, k, m, 1.0);
    for (it = 0; it < t; ++it) {
      LIBXSMM_MATINIT(float, 24, &LIBXSMM_VLA_ACCESS(2, djdhgold, it, 0, m * n), m, n, m, 1.0);
      LIBXSMM_MATINIT(float, 24, &LIBXSMM_VLA_ACCESS(2, zgoldb, it, 0, m * n), m, n, m, 1.0);
      LIBXSMM_MATINIT(float, 24, &LIBXSMM_VLA_ACCESS(2, xgold, it, 0, k * n), k, n, k, 1.0);
      LIBXSMM_MATINIT(float, 24, &LIBXSMM_VLA_ACCESS(2, hgoldb, it, 0, m * n), m, n, m, 1.0);
    }
    zero_buf(deltagoldt, m*n*t);
    zero_buf(djdugold, m*m);
    zero_buf(djdwgold, m*k);
    zero_buf(djdxgoldt, k*n*t);
    zero_buf(zigold, m*n);
    zero_buf(di1gold, m*n);
    zero_buf(di2gold, m*n);
    zero_buf(dj1gold, m*m);
    zero_buf(dw1gold, m*k);
    zero_buf(ugoldTp, m*m);
    zero_buf(wgoldTp, m*k);
    zero_buf(hgoldTp, m*n);
    zero_buf(xgoldTp, k*n);
  }

  /* first touch LIBXSMM */
  if (pass == 0) {
    zero_buf(w,  m*k);
    zero_buf(xt, k*n*t);
    zero_buf(u,  m*m);
    if (reuse) {
      zero_buf(h, m*n);
    } else {
      zero_buf(h, m*n*(t+1));
    }
  } else {
    zero_buf(w, m*k);
    zero_buf(xt, k*n*t);
    zero_buf(u, m*m);
    zero_buf(ht, m*n*t);
    zero_buf(djdw, m*k);
    zero_buf(djdxt, k*n*t);
    zero_buf(djdu, m*m);
    zero_buf(djdht, m*n*t);
  }
  LIBXSMM_VLA_DECL(2, float, x, xt, k * n);
  LIBXSMM_VLA_DECL(2, float, hnr, h, m * n);

  if (pass == 0) {
    handlewx = libxsmm_bgemm_handle_create(nThreads, LIBXSMM_GEMM_PRECISION(float), LIBXSMM_GEMM_PRECISION(float),
      m, n, k, &bm, &bn, &bk, &b_m1, &b_n1, &b_k1, &b_k2,
      &alpha, &beta, &gemm_flags, &strategy, &order);
    handleuh = libxsmm_bgemm_handle_create(nThreads, LIBXSMM_GEMM_PRECISION(float), LIBXSMM_GEMM_PRECISION(float),
      m, n, m, &bm, &bn, &bm, &b_m1, &b_n1, &b_m1, &b_m2,
      &alpha, &beta, &gemm_flags, &strategy, &order);
    handlett = libxsmm_bgemm_handle_create(nThreads, LIBXSMM_GEMM_PRECISION(float), LIBXSMM_GEMM_PRECISION(float),
      m, n*t, k, &bm, &bn, &bk, &b_m1, &b_n1, &b_k1, &b_k2,
      &alpha, &beta, &gemm_flags, &strategy, &order);
  } else {
    handlewx = libxsmm_bgemm_handle_create(nThreads, LIBXSMM_GEMM_PRECISION(float), LIBXSMM_GEMM_PRECISION(float),
      m, n, m, &bm, &bn, &bm, &b_m1, &b_n1, &b_m1, &b_m2,
      &alpha, &beta, &gemm_flags, &strategy, &order); /* U^T*delta */
    handleuh = libxsmm_bgemm_handle_create(nThreads, LIBXSMM_GEMM_PRECISION(float), LIBXSMM_GEMM_PRECISION(float),
      m, m, n, &bm, &bm, &bn, &b_m1, &b_m1, &b_n1, &b_n2,
      &alpha, &beta, &gemm_flags, &strategy, &order); /* delta*h^T */
    handlett = libxsmm_bgemm_handle_create(nThreads, LIBXSMM_GEMM_PRECISION(float), LIBXSMM_GEMM_PRECISION(float),
      m, k, n, &bm, &bk, &bn, &b_m1, &b_k1, &b_n1, &b_n2,
      &alpha, &beta, &gemm_flags, &strategy, &order); /* delta*x^T */
    handlewd = libxsmm_bgemm_handle_create(nThreads, LIBXSMM_GEMM_PRECISION(float), LIBXSMM_GEMM_PRECISION(float),
      k, n, m, &bk, &bn, &bm, &b_k1, &b_n1, &b_m1, &b_m2,
      &alpha, &beta, &gemm_flags, &strategy, &order); /* W^T*delta */
  }

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
    } else {
      zero_buf(&LIBXSMM_VLA_ACCESS(2, deltagold, t-1, 0, m * n), m*n);
      matrix_transpose(m, m, ugold, ugoldTp);
      for (i = t-2; i >= 0; --i) {
        matrix_sigmoid_inverse(m * n, &LIBXSMM_VLA_ACCESS(2, zgoldb, i+1, 0, m * n), zigold);
        LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &m, &n, &m, &alpha, ugoldTp, &m, &LIBXSMM_VLA_ACCESS(2, deltagold, i+1, 0, m * n), &m, &beta0, di1gold, &m);
        matrix_add(m * n, &LIBXSMM_VLA_ACCESS(2, djdhgold, i+1, 0, m * n), di1gold, di2gold);
        matrix_eltwise_mult(m * n, zigold, di2gold, &LIBXSMM_VLA_ACCESS(2, deltagold, i, 0, m * n));
      }
      if (pass == 1 || pass == 3) {
        matrix_transpose(m, k, wgold, wgoldTp);
        for (i = 0; i < t; ++i) {
          LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &k, &n, &m, &alpha, wgoldTp, &k, &LIBXSMM_VLA_ACCESS(2, deltagold, i, 0, m * n), &m, &beta0, &LIBXSMM_VLA_ACCESS(2, djdxgold, i, 0, k * n), &k);
        }
      }
      if (pass == 2 || pass == 3) {
        for (i = 0; i < t; ++i) {
          matrix_transpose(m, n, &LIBXSMM_VLA_ACCESS(2, hgoldb, i, 0, m * n), hgoldTp);
          LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &m, &m, &n, &alpha, &LIBXSMM_VLA_ACCESS(2, deltagold, i, 0, m * n), &m, hgoldTp, &n, &beta0, dj1gold, &m);
          matrix_add(m*m, dj1gold, djdugold, djdugold);
          matrix_transpose(k, n, &LIBXSMM_VLA_ACCESS(2, xgold, i, 0, k * n), xgoldTp);
          LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &m, &k, &n, &alpha, &LIBXSMM_VLA_ACCESS(2, deltagold, i, 0, m * n), &m, xgoldTp, &n, &beta0, dw1gold, &m);
          matrix_add(m*k, dw1gold, djdwgold, djdwgold);
        }
      }
    }
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
    rnncell_desc.reuse = reuse;
    rnncell_desc.pass = pass;
    rnncell_desc.datatype_in = LIBXSMM_DNN_DATATYPE_F32;
    rnncell_desc.datatype_out = LIBXSMM_DNN_DATATYPE_F32;
    rnncell_desc.buffer_format = LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM;

    libxsmm_handle = libxsmm_dnn_create_rnncell( rnncell_desc, &status );
    CHKERR_LIBXSMM_DNN( status );

    /* setup LIBXSMM buffers and filter */
    libxsmm_layout = libxsmm_dnn_rnncell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_RNN_REGULAR_INPUT, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_input = libxsmm_dnn_link_tensor( libxsmm_layout, xt, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_rnncell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_RNN_REGULAR_WEIGHT, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_weight = libxsmm_dnn_link_tensor( libxsmm_layout, w, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_rnncell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_RNN_REGULAR_RECUR_WEIGHT, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_recur_weight = libxsmm_dnn_link_tensor( libxsmm_layout, u, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    if (pass == 0) {
      libxsmm_layout = libxsmm_dnn_rnncell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_RNN_REGULAR_HIDDEN_STATE, &status ); CHKERR_LIBXSMM_DNN( status );
      libxsmm_hidden_state = libxsmm_dnn_link_tensor( libxsmm_layout, h, &status ); CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
    } else {
      libxsmm_layout = libxsmm_dnn_rnncell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_RNN_REGULAR_HIDDEN_STATE, &status ); CHKERR_LIBXSMM_DNN( status );
      libxsmm_hidden_state = libxsmm_dnn_link_tensor( libxsmm_layout, ht, &status ); CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

      libxsmm_layout = libxsmm_dnn_rnncell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_RNN_GRADIENT_INPUT, &status ); CHKERR_LIBXSMM_DNN( status );
      libxsmm_dinput = libxsmm_dnn_link_tensor( libxsmm_layout, djdxt, &status ); CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

      libxsmm_layout = libxsmm_dnn_rnncell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_RNN_GRADIENT_WEIGHT, &status ); CHKERR_LIBXSMM_DNN( status );
      libxsmm_dweight = libxsmm_dnn_link_tensor( libxsmm_layout, djdw, &status ); CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

      libxsmm_layout = libxsmm_dnn_rnncell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_RNN_GRADIENT_RECUR_WEIGHT, &status ); CHKERR_LIBXSMM_DNN( status );
      libxsmm_drecur_weight = libxsmm_dnn_link_tensor( libxsmm_layout, djdu, &status ); CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

      libxsmm_layout = libxsmm_dnn_rnncell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_RNN_GRADIENT_HIDDEN_STATE, &status ); CHKERR_LIBXSMM_DNN( status );
      libxsmm_dhidden_state = libxsmm_dnn_link_tensor( libxsmm_layout, djdht, &status ); CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
    }

    /* copy in data to LIBXSMM format */
    if (pass == 0) {
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
    } else {
      matrix_transpose(m, m, ugold, ugoldTp);
      CHKERR_LIBXSMM_DNN( libxsmm_bgemm_copyin_a(handlewx, ugoldTp, &m, u) );
      for (it = 0; it < t; ++it) {
        matrix_transpose(m, n, &LIBXSMM_VLA_ACCESS(2, hgoldb, it, 0, m * n), hgoldTp);
        CHKERR_LIBXSMM_DNN( libxsmm_bgemm_copyin_b(handleuh, hgoldTp, &n, &LIBXSMM_VLA_ACCESS(2, hb, it, 0, m * n)) );
        matrix_transpose(k, n, &LIBXSMM_VLA_ACCESS(2, xgold, it, 0, k * n), xgoldTp);
        CHKERR_LIBXSMM_DNN( libxsmm_bgemm_copyin_b(handlett, xgoldTp, &n, &LIBXSMM_VLA_ACCESS(2, x, it, 0, k * n)) );
      }
      matrix_transpose(m, k, wgold, wgoldTp);
      CHKERR_LIBXSMM_DNN( libxsmm_bgemm_copyin_a(handlewd, wgoldTp, &k, w) );
      for (it = 0; it < t; ++it) {
        CHKERR_LIBXSMM_DNN( libxsmm_bgemm_copyin_b(handlewx, &LIBXSMM_VLA_ACCESS(2, djdhgold, it, 0, m * n), &m, &LIBXSMM_VLA_ACCESS(2, djdh, it, 0, m * n)) );
      }
    }

    /* bind buffers and filter to handle */
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_bind_tensor( libxsmm_handle, libxsmm_input, LIBXSMM_DNN_RNN_REGULAR_INPUT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_bind_tensor( libxsmm_handle, libxsmm_hidden_state, LIBXSMM_DNN_RNN_REGULAR_HIDDEN_STATE ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_bind_tensor( libxsmm_handle, libxsmm_weight, LIBXSMM_DNN_RNN_REGULAR_WEIGHT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_bind_tensor( libxsmm_handle, libxsmm_recur_weight, LIBXSMM_DNN_RNN_REGULAR_RECUR_WEIGHT ) );
    if (pass != 0) {
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_bind_tensor( libxsmm_handle, libxsmm_dinput, LIBXSMM_DNN_RNN_GRADIENT_INPUT ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_bind_tensor( libxsmm_handle, libxsmm_dhidden_state, LIBXSMM_DNN_RNN_GRADIENT_HIDDEN_STATE ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_bind_tensor( libxsmm_handle, libxsmm_dweight, LIBXSMM_DNN_RNN_GRADIENT_WEIGHT ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_bind_tensor( libxsmm_handle, libxsmm_drecur_weight, LIBXSMM_DNN_RNN_GRADIENT_RECUR_WEIGHT ) );
    }

    /* let's allocate and bind scratch */
    if (pass == 0) {
      scratch_size = libxsmm_dnn_rnncell_get_scratch_size( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_FWD, &status );
      CHKERR_LIBXSMM_DNN( status );
      scratch = libxsmm_aligned_malloc( scratch_size, 2097152 );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_bind_scratch( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_FWD, scratch ) );
    } else {
      scratch_size = libxsmm_dnn_rnncell_get_scratch_size( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_ALL, &status );
      CHKERR_LIBXSMM_DNN( status );
      scratch = libxsmm_aligned_malloc( scratch_size, 2097152 );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_bind_scratch( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_ALL, scratch ) );
    }
    zero_buf( (float*)scratch, scratch_size/4 );

    /* let's allocate and bind internalstate */
    if (pass == 0) {
      internalstate_size = libxsmm_dnn_rnncell_get_internalstate_size( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_FWD, &status );
      CHKERR_LIBXSMM_DNN( status );
      internalstate = libxsmm_aligned_malloc( internalstate_size, 2097152 );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_bind_internalstate( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_FWD, internalstate ) );
    } else {
      internalstate_size = libxsmm_dnn_rnncell_get_internalstate_size( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_ALL, &status );
      CHKERR_LIBXSMM_DNN( status );
      internalstate = libxsmm_aligned_malloc( internalstate_size, 2097152 );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_bind_internalstate( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_ALL, internalstate ) );
    }
    zero_buf( (float*)internalstate, internalstate_size/4 );
    if (pass != 0) {
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_assign_internalstate( libxsmm_handle, zgoldt ) );
    }

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
        libxsmm_bgemm_copyout_b( m, n, bm, bn, h, htest );
      } else {
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

    if ( (pass == 1) && LIBXSMM_NEQ(0, check) ) {
      printf("##########################################\n");
      printf("#   Correctness - BWD (custom-Storage)   #\n");
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
        CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_BWD, 0, tid ) );
      }

      LIBXSMM_VLA_DECL(2, float, djdxtest, djdxtestt, k * n);
      /* copy out data */
      for (i = 0; i < t; ++i) {
        CHKERR_LIBXSMM_DNN( libxsmm_bgemm_copyout_c(handlewd, &LIBXSMM_VLA_ACCESS(2, djdx, i, 0, k * n), &k, &LIBXSMM_VLA_ACCESS(2, djdxtest, i, 0, k * n)) );
      }

      /* compare */
      libxsmm_matdiff(LIBXSMM_DATATYPE_F32, k*n*t, 1, djdxgoldt, djdxtestt, 0, 0, &norms_bwd);
      printf("L1 reference  : %.25g\n", norms_bwd.l1_ref);
      printf("L1 test       : %.25g\n", norms_bwd.l1_tst);
      printf("L2 abs.error  : %.24f\n", norms_bwd.l2_abs);
      printf("L2 rel.error  : %.24f\n", norms_bwd.l2_rel);
      printf("Linf abs.error: %.24f\n", norms_bwd.linf_abs);
      printf("Linf rel.error: %.24f\n", norms_bwd.linf_rel);
      printf("Check-norm    : %.24f\n", norms_bwd.normf_rel);
      libxsmm_matdiff_reduce(&diff, &norms_bwd);
    }

    if ( (pass == 2) && LIBXSMM_NEQ(0, check) ) {
      printf("##########################################\n");
      printf("#   Correctness - UPD (custom-Storage)   #\n");
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
        CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_UPD, 0, tid ) );
      }

      /* copy out data */
      CHKERR_LIBXSMM_DNN( libxsmm_bgemm_copyout_c(handlett, djdw, &m, djdwtest) );
      CHKERR_LIBXSMM_DNN( libxsmm_bgemm_copyout_c(handleuh, djdu, &m, djdutest) );

      /* compare */
      libxsmm_matdiff(LIBXSMM_DATATYPE_F32, m*k, 1, djdwgold, djdwtest, 0, 0, &norms_upd_w);
      printf("Delta weight\n");
      printf("L1 reference  : %.25g\n", norms_upd_w.l1_ref);
      printf("L1 test       : %.25g\n", norms_upd_w.l1_tst);
      printf("L2 abs.error  : %.24f\n", norms_upd_w.l2_abs);
      printf("L2 rel.error  : %.24f\n", norms_upd_w.l2_rel);
      printf("Linf abs.error: %.24f\n", norms_upd_w.linf_abs);
      printf("Linf rel.error: %.24f\n", norms_upd_w.linf_rel);
      printf("Check-norm    : %.24f\n", norms_upd_w.normf_rel);
      libxsmm_matdiff_reduce(&diff, &norms_upd_w);

      libxsmm_matdiff(LIBXSMM_DATATYPE_F32, m*m, 1, djdugold, djdutest, 0, 0, &norms_upd_u);
      printf("Delta recurrent weight\n");
      printf("L1 reference  : %.25g\n", norms_upd_u.l1_ref);
      printf("L1 test       : %.25g\n", norms_upd_u.l1_tst);
      printf("L2 abs.error  : %.24f\n", norms_upd_u.l2_abs);
      printf("L2 rel.error  : %.24f\n", norms_upd_u.l2_rel);
      printf("Linf abs.error: %.24f\n", norms_upd_u.linf_abs);
      printf("Linf rel.error: %.24f\n", norms_upd_u.linf_rel);
      printf("Check-norm    : %.24f\n", norms_upd_u.normf_rel);
      libxsmm_matdiff_reduce(&diff, &norms_upd_u);
    }

    if ( (pass == 3) && LIBXSMM_NEQ(0, check) ) {
      printf("##########################################\n");
      printf("# Correctness - BWD+UPD (custom-Storage) #\n");
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
        CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_ALL, 0, tid ) );
      }

      LIBXSMM_VLA_DECL(2, float, djdxtest, djdxtestt, k * n);
      /* copy out data */
      for (i = 0; i < t; ++i) {
        CHKERR_LIBXSMM_DNN( libxsmm_bgemm_copyout_c(handlewd, &LIBXSMM_VLA_ACCESS(2, djdx, i, 0, k * n), &k, &LIBXSMM_VLA_ACCESS(2, djdxtest, i, 0, k * n)) );
      }
      CHKERR_LIBXSMM_DNN( libxsmm_bgemm_copyout_c(handlett, djdw, &m, djdwtest) );
      CHKERR_LIBXSMM_DNN( libxsmm_bgemm_copyout_c(handleuh, djdu, &m, djdutest) );

      /* compare */
      libxsmm_matdiff(LIBXSMM_DATATYPE_F32, k*n*t, 1, djdxgoldt, djdxtestt, 0, 0, &norms_bwd);
      printf("Delta input\n");
      printf("L1 reference  : %.25g\n", norms_bwd.l1_ref);
      printf("L1 test       : %.25g\n", norms_bwd.l1_tst);
      printf("L2 abs.error  : %.24f\n", norms_bwd.l2_abs);
      printf("L2 rel.error  : %.24f\n", norms_bwd.l2_rel);
      printf("Linf abs.error: %.24f\n", norms_bwd.linf_abs);
      printf("Linf rel.error: %.24f\n", norms_bwd.linf_rel);
      printf("Check-norm    : %.24f\n", norms_bwd.normf_rel);
      libxsmm_matdiff_reduce(&diff, &norms_bwd);

      libxsmm_matdiff(LIBXSMM_DATATYPE_F32, m*k, 1, djdwgold, djdwtest, 0, 0, &norms_upd_w);
      printf("Delta weight\n");
      printf("L1 reference  : %.25g\n", norms_upd_w.l1_ref);
      printf("L1 test       : %.25g\n", norms_upd_w.l1_tst);
      printf("L2 abs.error  : %.24f\n", norms_upd_w.l2_abs);
      printf("L2 rel.error  : %.24f\n", norms_upd_w.l2_rel);
      printf("Linf abs.error: %.24f\n", norms_upd_w.linf_abs);
      printf("Linf rel.error: %.24f\n", norms_upd_w.linf_rel);
      printf("Check-norm    : %.24f\n", norms_upd_w.normf_rel);
      libxsmm_matdiff_reduce(&diff, &norms_upd_w);

      libxsmm_matdiff(LIBXSMM_DATATYPE_F32, m*m, 1, djdugold, djdutest, 0, 0, &norms_upd_u);
      printf("Delta recurrent weight\n");
      printf("L1 reference  : %.25g\n", norms_upd_u.l1_ref);
      printf("L1 test       : %.25g\n", norms_upd_u.l1_tst);
      printf("L2 abs.error  : %.24f\n", norms_upd_u.l2_abs);
      printf("L2 rel.error  : %.24f\n", norms_upd_u.l2_rel);
      printf("Linf abs.error: %.24f\n", norms_upd_u.linf_abs);
      printf("Linf rel.error: %.24f\n", norms_upd_u.linf_rel);
      printf("Check-norm    : %.24f\n", norms_upd_u.normf_rel);
      libxsmm_matdiff_reduce(&diff, &norms_upd_u);
    }

    if ( pass == 0 ) {
      printf("##########################################\n");
      printf("#   Performance - FWD (custom-Storage)   #\n");
      printf("##########################################\n");
      /* run LIBXSMM RNN for performance */
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
          libxsmm_dnn_rnncell_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_FWD, 0, tid );
        }
      }
      l_end = libxsmm_timer_tick();
      l_total = libxsmm_timer_duration(l_start, l_end);
      flops = ((2.0 * m * n * k) + (2.0 * m * n * m) + (m * n) + (tflops * m * n)) * (double)t * (double)iters;

      printf("GFLOP  = %.5g\n", flops*1e-9/(double)iters);
      printf("fp time = %.5g\n", ((double)(l_total/iters)));
      printf("GFLOPS  = %.5g\n", (flops*1e-9)/l_total);

      printf("PERFDUMP,FP,%s,%i,%i,%i,%i,%i,%i,%i,%i,%.5g,%.5g\n", LIBXSMM_VERSION, nThreads, m, n, k, t, bm, bn, bk, ((double)(l_total/iters)), (flops*1e-9)/l_total);
    }

    if ( pass == 1 ) {
      printf("##########################################\n");
      printf("#   Performance - BWD (custom-Storage)   #\n");
      printf("##########################################\n");
      /* run LIBXSMM RNN for performance */
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
          libxsmm_dnn_rnncell_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_BWD, 0, tid );
        }
      }
      l_end = libxsmm_timer_tick();
      l_total = libxsmm_timer_duration(l_start, l_end);
      flops = m * m; /* U^T */
      flops += (2.0 * m * n * m); /* U^T * delta */
      flops += (m * n); /* dJdh + (U^T * delta) */
      flops += (tflops * m * n); /* sigma'(Z) */
      flops += (m * n); /* sigma'(Z) * (dJdh + (U^T * delta)) */
      flops *= t; /* for t time steps */
      tempflops = m * k; /* W^T */
      tempflops += (2.0 * m * n * k); /* W^T * delta */
      tempflops *= t; /* for t time steps of input */
      flops += tempflops;
      flops *= iters;

      printf("GFLOP  = %.5g\n", flops*1e-9/(double)iters);
      printf("bp time = %.5g\n", ((double)(l_total/iters)));
      printf("GFLOPS  = %.5g\n", (flops*1e-9)/l_total);

      printf("PERFDUMP,BP,%s,%i,%i,%i,%i,%i,%i,%i,%i,%.5g,%.5g\n", LIBXSMM_VERSION, nThreads, m, n, k, t, bm, bn, bk, ((double)(l_total/iters)), (flops*1e-9)/l_total);
    }

    if ( pass == 2 ) {
      printf("##########################################\n");
      printf("#   Performance - UPD (custom-Storage)   #\n");
      printf("##########################################\n");
      /* run LIBXSMM RNN for performance */
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
          libxsmm_dnn_rnncell_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_UPD, 0, tid );
        }
      }
      l_end = libxsmm_timer_tick();
      l_total = libxsmm_timer_duration(l_start, l_end);
      flops = m * m; /* U^T */
      flops += (2.0 * m * n * m); /* U^T * delta */
      flops += (m * n); /* dJdh + (U^T * delta) */
      flops += (tflops * m * n); /* sigma'(Z) */
      flops += (m * n); /* sigma'(Z) * (dJdh + (U^T * delta)) */
      flops *= t; /* for t time steps */
      tempflops = m * n; /* h^T */
      tempflops += (2.0 * m * n * m); /* delta * h^T */
      tempflops *= t; /* for t time steps */
      tempflops += (m * m * (t-1)); /* for summation of dJdU */
      flops += tempflops;
      tempflops = k * n; /* x^T */
      tempflops += (2.0 * m * n * k); /* delta * x^T */
      tempflops *= t; /* for t time steps */
      tempflops += (m * k * (t-1)); /* for summation of dJdW */
      flops += tempflops;
      flops *= iters;

      printf("GFLOP  = %.5g\n", flops*1e-9/(double)iters);
      printf("wu time = %.5g\n", ((double)(l_total/iters)));
      printf("GFLOPS  = %.5g\n", (flops*1e-9)/l_total);

      printf("PERFDUMP,WU,%s,%i,%i,%i,%i,%i,%i,%i,%i,%.5g,%.5g\n", LIBXSMM_VERSION, nThreads, m, n, k, t, bm, bn, bk, ((double)(l_total/iters)), (flops*1e-9)/l_total);
    }

    if ( pass == 3 ) {
      printf("##########################################\n");
      printf("# Performance - BWD+UPD (custom-Storage) #\n");
      printf("##########################################\n");
      /* run LIBXSMM RNN for performance */
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
          libxsmm_dnn_rnncell_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_ALL, 0, tid );
        }
      }
      l_end = libxsmm_timer_tick();
      l_total = libxsmm_timer_duration(l_start, l_end);
      flops = m * m; /* U^T */
      flops += (2.0 * m * n * m); /* U^T * delta */
      flops += (m * n); /* dJdh + (U^T * delta) */
      flops += (tflops * m * n); /* sigma'(Z) */
      flops += (m * n); /* sigma'(Z) * (dJdh + (U^T * delta)) */
      flops *= t; /* for t time steps */
      tempflops = m * n; /* h^T */
      tempflops += (2.0 * m * n * m); /* delta * h^T */
      tempflops *= t; /* for t time steps */
      tempflops += (m * m * (t-1)); /* for summation of dJdU */
      flops += tempflops;
      tempflops = k * n; /* x^T */
      tempflops += (2.0 * m * n * k); /* delta * x^T */
      tempflops *= t; /* for t time steps */
      tempflops += (m * k * (t-1)); /* for summation of dJdW */
      flops += tempflops;
      tempflops = m * k; /* W^T */
      tempflops += (2.0 * m * n * k); /* W^T * delta */
      tempflops *= t; /* for t time steps of input */
      flops += tempflops;
      flops *= iters;

      printf("GFLOP  = %.5g\n", flops*1e-9/(double)iters);
      printf("bp+wu time = %.5g\n", ((double)(l_total/iters)));
      printf("GFLOPS  = %.5g\n", (flops*1e-9)/l_total);

      printf("PERFDUMP,BP+WU,%s,%i,%i,%i,%i,%i,%i,%i,%i,%.5g,%.5g\n", LIBXSMM_VERSION, nThreads, m, n, k, t, bm, bn, bk, ((double)(l_total/iters)), (flops*1e-9)/l_total);
    }

    /* clean-up */
    if (pass == 0) {
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_release_scratch( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_FWD ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_release_internalstate( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_FWD ) );
    } else {
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_release_scratch( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_ALL ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_release_internalstate( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_ALL ) );
    }
    libxsmm_free(scratch);
    libxsmm_free(internalstate);
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_release_tensor( libxsmm_handle, LIBXSMM_DNN_RNN_REGULAR_INPUT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_release_tensor( libxsmm_handle, LIBXSMM_DNN_RNN_REGULAR_HIDDEN_STATE ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_release_tensor( libxsmm_handle, LIBXSMM_DNN_RNN_REGULAR_WEIGHT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_release_tensor( libxsmm_handle, LIBXSMM_DNN_RNN_REGULAR_RECUR_WEIGHT ) );
    if (pass != 0) {
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_release_tensor( libxsmm_handle, LIBXSMM_DNN_RNN_GRADIENT_INPUT ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_release_tensor( libxsmm_handle, LIBXSMM_DNN_RNN_GRADIENT_HIDDEN_STATE ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_release_tensor( libxsmm_handle, LIBXSMM_DNN_RNN_GRADIENT_WEIGHT ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_release_tensor( libxsmm_handle, LIBXSMM_DNN_RNN_GRADIENT_RECUR_WEIGHT ) );
    }
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_input ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_hidden_state ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_weight ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_recur_weight ) );
    if (pass != 0) {
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_dinput ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_dhidden_state ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_dweight ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_drecur_weight ) );
    }
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_rnncell( libxsmm_handle ) );
  }

  /* deallocate data */
  if (pass == 0) {
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
    libxsmm_free(hgold_temp);
  } else {
    libxsmm_free(wgold);
    libxsmm_free(xgoldt);
    libxsmm_free(ugold);
    libxsmm_free(hgoldt);
    libxsmm_free(djdwgold);
    libxsmm_free(djdxgoldt);
    libxsmm_free(djdugold);
    libxsmm_free(djdhgoldt);
    libxsmm_free(zgoldt);
    libxsmm_free(deltagoldt);
    libxsmm_free(zigold);
    libxsmm_free(di1gold);
    libxsmm_free(di2gold);
    libxsmm_free(dj1gold);
    libxsmm_free(dw1gold);
    libxsmm_free(wgoldTp);
    libxsmm_free(xgoldTp);
    libxsmm_free(ugoldTp);
    libxsmm_free(hgoldTp);
    libxsmm_free(w);
    libxsmm_free(xt);
    libxsmm_free(u);
    libxsmm_free(ht);
    libxsmm_free(djdw);
    libxsmm_free(djdxt);
    libxsmm_free(djdu);
    libxsmm_free(djdht);
    libxsmm_free(djdwtest);
    libxsmm_free(djdxtestt);
    libxsmm_free(djdutest);
  }

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

