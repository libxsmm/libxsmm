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
  /* Arrays related to FWD pass */
  float *urgold, *uzgold, *uggold, *xgoldt, *wrgold, *wzgold, *wggold, *hgold = NULL, *brgold = NULL, *bzgold = NULL, *bggold = NULL;
  float *rgold = NULL, *zgold = NULL, *ggold = NULL;
  float *r1gold, *r2gold, *z1gold, *z2gold, *g1gold, *g2gold, *g3gold, *h1gold, *h2gold, *h3gold;
  float *ur, *uz, *ug, *xt, *wr, *wz, *wg, *h = NULL, *br, *bz, *bg, *htest = NULL, *hgold_temp = NULL;
  /* Arrays related to BWD and UPD pass */
#if 0
  float *hgoldt = NULL, *rgoldt = NULL, *zgoldt = NULL, *ggoldt = NULL, *cgoldt = NULL, *i3gold = NULL, *f3gold = NULL, *d3gold = NULL, *d4gold = NULL, *dgoldt = NULL, *deltagoldt = NULL;
  float *djdhgoldt = NULL, *djddgoldt = NULL, *djdigoldt = NULL, *djdfgoldt = NULL, *djdcgoldt = NULL, *djdogoldt = NULL, *djdxgoldt = NULL;
  float *djdwigold = NULL, *djdwfgold = NULL, *djdwogold = NULL, *djdwcgold = NULL, *djdrigold = NULL, *djdrfgold = NULL, *djdrogold = NULL, *djdrcgold = NULL;
  float *djdbigold = NULL, *djdbfgold = NULL, *djdbogold = NULL, *djdbcgold = NULL, *wgoldTp = NULL, *rgoldTp = NULL, *xgoldTp = NULL, *hgoldTp = NULL;
  float *ht = NULL, *djdht = NULL, *djdxt = NULL, *djdwi = NULL, *djdwf = NULL, *djdwo = NULL, *djdwc = NULL, *djdri = NULL, *djdrf = NULL, *djdro = NULL, *djdrc = NULL, *djdbi = NULL, *djdbf = NULL, *djdbo = NULL, *djdbc = NULL;
  float *djdxtestt = NULL, *djdwtest = NULL, *djdrtest = NULL, *djdbtest = NULL, *djdwgold4 = NULL, *djdrgold4 = NULL, *djdbgold4 = NULL;
#endif

  const char transa = 'N', transb = 'N'; /* no transposes */
  const int gemm_flags = LIBXSMM_GEMM_FLAGS(transa, transb);
  const float alpha = 1, beta = 1, beta0 = 0;
  void *scratch, *internalstate;
  size_t scratch_size = 0, internalstate_size = 0;

  int iters = 10;   /* repetitions of benchmark */
  int pass = 0;     /* pass: 0--FWD, 1--BWD, 2--UPD, 3--BWD+UPD */
  int m = 1024;     /* number of outputs */
  int n = 512;      /* size of mini-batch */
  int k = 256;      /* number of inputs */
  int t = 5;        /* number of time steps (> 1) */
  int reuse = 1;    /* reuse=1 for FWD overwrites the same memory
                     * for intermediate values during inference;
                     * reuse value is immaterial for BWD and UPD */
  int bm = 32;      /* first blocking factor for m */
  int bn = 32;      /* first blocking factor for n */
  int bk = 32;      /* first blocking factor for k */
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
  libxsmm_bgemm_handle* handleux = 0;
  libxsmm_bgemm_handle* handlewh = 0;
  libxsmm_bgemm_handle* handlett = 0;
  /*libxsmm_bgemm_handle* handlewd = 0;*/
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
  int i, j, it;

  libxsmm_dnn_grucell_desc grucell_desc;
  libxsmm_dnn_grucell* libxsmm_handle;
  libxsmm_dnn_tensor* libxsmm_input;
  libxsmm_dnn_tensor* libxsmm_hidden_state;
  libxsmm_dnn_tensor* libxsmm_weight_r;
  libxsmm_dnn_tensor* libxsmm_weight_z;
  libxsmm_dnn_tensor* libxsmm_weight_g;
  libxsmm_dnn_tensor* libxsmm_recur_weight_r;
  libxsmm_dnn_tensor* libxsmm_recur_weight_z;
  libxsmm_dnn_tensor* libxsmm_recur_weight_g;
  libxsmm_dnn_tensor* libxsmm_bias_r;
  libxsmm_dnn_tensor* libxsmm_bias_z;
  libxsmm_dnn_tensor* libxsmm_bias_g;
  libxsmm_dnn_tensor* libxsmm_dinput = NULL;
  libxsmm_dnn_tensor* libxsmm_dhidden_state = NULL;
  libxsmm_dnn_tensor* libxsmm_dweight_r = NULL;
  libxsmm_dnn_tensor* libxsmm_dweight_z = NULL;
  libxsmm_dnn_tensor* libxsmm_dweight_g = NULL;
  libxsmm_dnn_tensor* libxsmm_drecur_weight_r = NULL;
  libxsmm_dnn_tensor* libxsmm_drecur_weight_z = NULL;
  libxsmm_dnn_tensor* libxsmm_drecur_weight_g = NULL;
  libxsmm_dnn_tensor* libxsmm_dbias_r = NULL;
  libxsmm_dnn_tensor* libxsmm_dbias_z = NULL;
  libxsmm_dnn_tensor* libxsmm_dbias_g = NULL;

  libxsmm_dnn_tensor_datalayout* libxsmm_layout;
  libxsmm_dnn_err_t status;
  libxsmm_dnn_err_t global_status = LIBXSMM_DNN_SUCCESS;

  libxsmm_matdiff_info norms_fwd, norms_bwd, norms_upd_w, norms_upd_r, norms_upd_b, diff;
  memset(&norms_fwd, 0, sizeof(norms_fwd));
  memset(&norms_bwd, 0, sizeof(norms_bwd));
  memset(&norms_upd_w, 0, sizeof(norms_upd_w));
  memset(&norms_upd_r, 0, sizeof(norms_upd_r));
  memset(&norms_upd_b, 0, sizeof(norms_upd_b));
  memset(&diff, 0, sizeof(diff));

  if (argc > 1 && !strncmp(argv[1], "-h", 3)) {
    printf("\nUsage: ./grudriver [reps] [pass: 0--FWD, 1--BWD, 2--UPD, 3--BWD+UPD] [M] [N] [K] [time_steps > 1] [reuse (for FWD): 0/1] [bm] [bn] [bk]\n\n");
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
    urgold = (float*)libxsmm_aligned_malloc( m*k*sizeof(float), 2097152);
    uzgold = (float*)libxsmm_aligned_malloc( m*k*sizeof(float), 2097152);
    uggold = (float*)libxsmm_aligned_malloc( m*k*sizeof(float), 2097152);
    xgoldt = (float*)libxsmm_aligned_malloc( k*n*t*sizeof(float), 2097152);
    wrgold = (float*)libxsmm_aligned_malloc( m*m*sizeof(float), 2097152);
    wzgold = (float*)libxsmm_aligned_malloc( m*m*sizeof(float), 2097152);
    wggold = (float*)libxsmm_aligned_malloc( m*m*sizeof(float), 2097152);
    hgold  = (float*)libxsmm_aligned_malloc( m*n*sizeof(float), 2097152);
    brgold = (float*)libxsmm_aligned_malloc( m*n*sizeof(float), 2097152);
    bzgold = (float*)libxsmm_aligned_malloc( m*n*sizeof(float), 2097152);
    bggold = (float*)libxsmm_aligned_malloc( m*n*sizeof(float), 2097152);
    rgold  = (float*)libxsmm_aligned_malloc( m*n*sizeof(float), 2097152);
    zgold  = (float*)libxsmm_aligned_malloc( m*n*sizeof(float), 2097152);
    ggold  = (float*)libxsmm_aligned_malloc( m*n*sizeof(float), 2097152);
    r1gold = (float*)libxsmm_aligned_malloc( m*n*sizeof(float), 2097152);
    r2gold = (float*)libxsmm_aligned_malloc( m*n*sizeof(float), 2097152);
    z1gold = (float*)libxsmm_aligned_malloc( m*n*sizeof(float), 2097152);
    z2gold = (float*)libxsmm_aligned_malloc( m*n*sizeof(float), 2097152);
    g1gold = (float*)libxsmm_aligned_malloc( m*n*sizeof(float), 2097152);
    g2gold = (float*)libxsmm_aligned_malloc( m*n*sizeof(float), 2097152);
    g3gold = (float*)libxsmm_aligned_malloc( m*n*sizeof(float), 2097152);
    h1gold = (float*)libxsmm_aligned_malloc( m*n*sizeof(float), 2097152);
    h2gold = (float*)libxsmm_aligned_malloc( m*n*sizeof(float), 2097152);
    h3gold = (float*)libxsmm_aligned_malloc( m*n*sizeof(float), 2097152);
    ur     = (float*)libxsmm_aligned_malloc( m*k*sizeof(float), 2097152);
    uz     = (float*)libxsmm_aligned_malloc( m*k*sizeof(float), 2097152);
    ug     = (float*)libxsmm_aligned_malloc( m*k*sizeof(float), 2097152);
    xt     = (float*)libxsmm_aligned_malloc( k*n*t*sizeof(float), 2097152);
    wr     = (float*)libxsmm_aligned_malloc( m*m*sizeof(float), 2097152);
    wz     = (float*)libxsmm_aligned_malloc( m*m*sizeof(float), 2097152);
    wg     = (float*)libxsmm_aligned_malloc( m*m*sizeof(float), 2097152);
    if (reuse) {
      h      = (float*)libxsmm_aligned_malloc( m*n*sizeof(float), 2097152);
    } else {
      h      = (float*)libxsmm_aligned_malloc( m*n*(t+1)*sizeof(float), 2097152);
    }
    br     = (float*)libxsmm_aligned_malloc( m*n*sizeof(float), 2097152);
    bz     = (float*)libxsmm_aligned_malloc( m*n*sizeof(float), 2097152);
    bg     = (float*)libxsmm_aligned_malloc( m*n*sizeof(float), 2097152);
    htest  = (float*)libxsmm_aligned_malloc( m*n*sizeof(float), 2097152);
    hgold_temp = (float*)libxsmm_aligned_malloc( m*n*sizeof(float), 2097152);
  } else {
#if 0
    wigold = (float*)libxsmm_aligned_malloc(m * k * sizeof(float), 2097152);
    wfgold = (float*)libxsmm_aligned_malloc(m * k * sizeof(float), 2097152);
    wogold = (float*)libxsmm_aligned_malloc(m * k * sizeof(float), 2097152);
    wcgold = (float*)libxsmm_aligned_malloc(m * k * sizeof(float), 2097152);
    xgoldt = (float*)libxsmm_aligned_malloc(k * n * sizeof(float) * t, 2097152);
    rigold = (float*)libxsmm_aligned_malloc(m * m * sizeof(float), 2097152);
    rfgold = (float*)libxsmm_aligned_malloc(m * m * sizeof(float), 2097152);
    rogold = (float*)libxsmm_aligned_malloc(m * m * sizeof(float), 2097152);
    rcgold = (float*)libxsmm_aligned_malloc(m * m * sizeof(float), 2097152);
    hgoldt = (float*)libxsmm_aligned_malloc(m * n * sizeof(float) * t, 2097152);
    i1gold = (float*)libxsmm_aligned_malloc(m * n * sizeof(float), 2097152);
    i2gold = (float*)libxsmm_aligned_malloc(m * n * sizeof(float), 2097152);
    i3gold = (float*)libxsmm_aligned_malloc(m * n * sizeof(float), 2097152);
    f1gold = (float*)libxsmm_aligned_malloc(m * n * sizeof(float), 2097152);
    f2gold = (float*)libxsmm_aligned_malloc(m * n * sizeof(float), 2097152);
    f3gold = (float*)libxsmm_aligned_malloc(m * n * sizeof(float), 2097152);
    o1gold = (float*)libxsmm_aligned_malloc(m * n * sizeof(float), 2097152);
    o2gold = (float*)libxsmm_aligned_malloc(m * n * sizeof(float), 2097152);
    c1gold = (float*)libxsmm_aligned_malloc(m * n * sizeof(float), 2097152);
    c2gold = (float*)libxsmm_aligned_malloc(m * n * sizeof(float), 2097152);
    igoldt = (float*)libxsmm_aligned_malloc(m * n * sizeof(float) * t, 2097152);
    fgoldt = (float*)libxsmm_aligned_malloc(m * n * sizeof(float) * t, 2097152);
    ogoldt = (float*)libxsmm_aligned_malloc(m * n * sizeof(float) * t, 2097152);
    cgoldt = (float*)libxsmm_aligned_malloc(m * n * sizeof(float) * t, 2097152);
    d1gold = (float*)libxsmm_aligned_malloc(m * n * sizeof(float), 2097152);
    d2gold = (float*)libxsmm_aligned_malloc(m * n * sizeof(float), 2097152);
    d3gold = (float*)libxsmm_aligned_malloc(m * n * sizeof(float), 2097152);
    d4gold = (float*)libxsmm_aligned_malloc(m * n * sizeof(float), 2097152);
    dgoldt = (float*)libxsmm_aligned_malloc(m * n * sizeof(float) * t, 2097152);
    djdhgoldt = (float*)libxsmm_aligned_malloc(m * n * sizeof(float) * t, 2097152);
    deltagoldt = (float*)libxsmm_aligned_malloc(m * n * sizeof(float) * t, 2097152);
    djddgoldt = (float*)libxsmm_aligned_malloc(m * n * sizeof(float) * t, 2097152);
    djdigoldt = (float*)libxsmm_aligned_malloc(m * n * sizeof(float) * t, 2097152);
    djdfgoldt = (float*)libxsmm_aligned_malloc(m * n * sizeof(float) * t, 2097152);
    djdcgoldt = (float*)libxsmm_aligned_malloc(m * n * sizeof(float) * t, 2097152);
    djdogoldt = (float*)libxsmm_aligned_malloc(m * n * sizeof(float) * t, 2097152);
    djdxgoldt = (float*)libxsmm_aligned_malloc(k * n * sizeof(float) * t, 2097152);
    djdwigold = (float*)libxsmm_aligned_malloc(m * k * sizeof(float), 2097152);
    djdwfgold = (float*)libxsmm_aligned_malloc(m * k * sizeof(float), 2097152);
    djdwogold = (float*)libxsmm_aligned_malloc(m * k * sizeof(float), 2097152);
    djdwcgold = (float*)libxsmm_aligned_malloc(m * k * sizeof(float), 2097152);
    djdrigold = (float*)libxsmm_aligned_malloc(m * m * sizeof(float), 2097152);
    djdrfgold = (float*)libxsmm_aligned_malloc(m * m * sizeof(float), 2097152);
    djdrogold = (float*)libxsmm_aligned_malloc(m * m * sizeof(float), 2097152);
    djdrcgold = (float*)libxsmm_aligned_malloc(m * m * sizeof(float), 2097152);
    djdbigold = (float*)libxsmm_aligned_malloc(m * n * sizeof(float), 2097152);
    djdbfgold = (float*)libxsmm_aligned_malloc(m * n * sizeof(float), 2097152);
    djdbogold = (float*)libxsmm_aligned_malloc(m * n * sizeof(float), 2097152);
    djdbcgold = (float*)libxsmm_aligned_malloc(m * n * sizeof(float), 2097152);
    wgoldTp = (float*)libxsmm_aligned_malloc(m * k * sizeof(float), 2097152);
    rgoldTp = (float*)libxsmm_aligned_malloc(m * m * sizeof(float), 2097152);
    xgoldTp = (float*)libxsmm_aligned_malloc(k * n * sizeof(float), 2097152);
    hgoldTp = (float*)libxsmm_aligned_malloc(m * n * sizeof(float), 2097152);
    wi = (float*)libxsmm_aligned_malloc(m * k * sizeof(float), 2097152);
    wf = (float*)libxsmm_aligned_malloc(m * k * sizeof(float), 2097152);
    wo = (float*)libxsmm_aligned_malloc(m * k * sizeof(float), 2097152);
    wc = (float*)libxsmm_aligned_malloc(m * k * sizeof(float), 2097152);
    xt = (float*)libxsmm_aligned_malloc(m * n * sizeof(float) * t, 2097152);
    ri = (float*)libxsmm_aligned_malloc(m * m * sizeof(float), 2097152);
    rf = (float*)libxsmm_aligned_malloc(m * m * sizeof(float), 2097152);
    ro = (float*)libxsmm_aligned_malloc(m * m * sizeof(float), 2097152);
    rc = (float*)libxsmm_aligned_malloc(m * m * sizeof(float), 2097152);
    ht = (float*)libxsmm_aligned_malloc(m * n * sizeof(float) * t, 2097152);
    bi = (float*)libxsmm_aligned_malloc( m*n*sizeof(float), 2097152);
    bf = (float*)libxsmm_aligned_malloc( m*n*sizeof(float), 2097152);
    bo = (float*)libxsmm_aligned_malloc( m*n*sizeof(float), 2097152);
    bc = (float*)libxsmm_aligned_malloc( m*n*sizeof(float), 2097152);
    djdht = (float*)libxsmm_aligned_malloc(m * n * sizeof(float) * t, 2097152);
    djdxt = (float*)libxsmm_aligned_malloc(k * n * sizeof(float) * t, 2097152);
    djdwi = (float*)libxsmm_aligned_malloc(m * k * sizeof(float), 2097152);
    djdwf = (float*)libxsmm_aligned_malloc(m * k * sizeof(float), 2097152);
    djdwo = (float*)libxsmm_aligned_malloc(m * k * sizeof(float), 2097152);
    djdwc = (float*)libxsmm_aligned_malloc(m * k * sizeof(float), 2097152);
    djdri = (float*)libxsmm_aligned_malloc(m * m * sizeof(float), 2097152);
    djdrf = (float*)libxsmm_aligned_malloc(m * m * sizeof(float), 2097152);
    djdro = (float*)libxsmm_aligned_malloc(m * m * sizeof(float), 2097152);
    djdrc = (float*)libxsmm_aligned_malloc(m * m * sizeof(float), 2097152);
    djdbi = (float*)libxsmm_aligned_malloc(m * n * sizeof(float), 2097152);
    djdbf = (float*)libxsmm_aligned_malloc(m * n * sizeof(float), 2097152);
    djdbo = (float*)libxsmm_aligned_malloc(m * n * sizeof(float), 2097152);
    djdbc = (float*)libxsmm_aligned_malloc(m * n * sizeof(float), 2097152);
    djdxtestt  = (float*)libxsmm_aligned_malloc(k * n * sizeof(float) * t, 2097152);
    djdwtest   = (float*)libxsmm_aligned_malloc(m * k * sizeof(float) * 4, 2097152);
    djdrtest   = (float*)libxsmm_aligned_malloc(m * m * sizeof(float) * 4, 2097152);
    djdbtest   = (float*)libxsmm_aligned_malloc(m * n * sizeof(float) * 4, 2097152);
    djdwgold4  = (float*)libxsmm_aligned_malloc(m * k * sizeof(float) * 4, 2097152);
    djdrgold4  = (float*)libxsmm_aligned_malloc(m * m * sizeof(float) * 4, 2097152);
    djdbgold4  = (float*)libxsmm_aligned_malloc(m * n * sizeof(float) * 4, 2097152);
#endif
  }
  LIBXSMM_VLA_DECL(2, float, xgold, xgoldt, k * n);
#if 0
  LIBXSMM_VLA_DECL(2, float, igoldb, igoldt, m * n);
  LIBXSMM_VLA_DECL(2, float, fgoldb, fgoldt, m * n);
  LIBXSMM_VLA_DECL(2, float, ogoldb, ogoldt, m * n);
  LIBXSMM_VLA_DECL(2, float, cgoldb, cgoldt, m * n);
  LIBXSMM_VLA_DECL(2, float, dgoldb, dgoldt, m * n);
  LIBXSMM_VLA_DECL(2, float, hgoldb, hgoldt, m * n);
  LIBXSMM_VLA_DECL(2, float, djdhgold, djdhgoldt, m * n);
  LIBXSMM_VLA_DECL(2, float, deltagold, deltagoldt, m * n);
  LIBXSMM_VLA_DECL(2, float, djddgold, djddgoldt, m * n);
  LIBXSMM_VLA_DECL(2, float, djdigold, djdigoldt, m * n);
  LIBXSMM_VLA_DECL(2, float, djdfgold, djdfgoldt, m * n);
  LIBXSMM_VLA_DECL(2, float, djdogold, djdogoldt, m * n);
  LIBXSMM_VLA_DECL(2, float, djdcgold, djdcgoldt, m * n);
  LIBXSMM_VLA_DECL(2, float, djdxgold, djdxgoldt, k * n);
#endif

  /* initialize data */
  if (pass == 0) {
    LIBXSMM_MATINIT(float, 42, urgold, m, k, m, 1.0);
    LIBXSMM_MATINIT(float, 42, uzgold, m, k, m, 1.0);
    LIBXSMM_MATINIT(float, 42, uggold, m, k, m, 1.0);
    for (it = 0; it < t; ++it) {
      LIBXSMM_MATINIT(float, 24, &LIBXSMM_VLA_ACCESS(2, xgold, it, 0, k * n), k, n, k, 1.0);
    }
    LIBXSMM_MATINIT(float, 42, wrgold, m, m, m, 1.0);
    LIBXSMM_MATINIT(float, 42, wzgold, m, m, m, 1.0);
    LIBXSMM_MATINIT(float, 42, wggold, m, m, m, 1.0);
    LIBXSMM_MATINIT(float, 24, hgold, m, n, m, 1.0);
    matrix_copy(m*n, hgold, hgold_temp); /* Required because hgold may get overwritten */
    LIBXSMM_MATINIT(float, 24, brgold, m, n, m, 1.0);
    LIBXSMM_MATINIT(float, 24, bzgold, m, n, m, 1.0);
    LIBXSMM_MATINIT(float, 24, bggold, m, n, m, 1.0);
    zero_buf(rgold, m*n);
    zero_buf(zgold, m*n);
    zero_buf(ggold, m*n);
    zero_buf(r1gold, m*n);
    zero_buf(r2gold, m*n);
    zero_buf(z1gold, m*n);
    zero_buf(z2gold, m*n);
    zero_buf(g1gold, m*n);
    zero_buf(g2gold, m*n);
    zero_buf(g3gold, m*n);
    zero_buf(h1gold, m*n);
    zero_buf(h2gold, m*n);
    zero_buf(h3gold, m*n);
  } else {
#if 0
    LIBXSMM_MATINIT(float, 42, wigold, m, k, m, 1.0);
    LIBXSMM_MATINIT(float, 42, wfgold, m, k, m, 1.0);
    LIBXSMM_MATINIT(float, 42, wogold, m, k, m, 1.0);
    LIBXSMM_MATINIT(float, 42, wcgold, m, k, m, 1.0);
    LIBXSMM_MATINIT(float, 42, rigold, m, m, m, 1.0);
    LIBXSMM_MATINIT(float, 42, rfgold, m, m, m, 1.0);
    LIBXSMM_MATINIT(float, 42, rogold, m, m, m, 1.0);
    LIBXSMM_MATINIT(float, 42, rcgold, m, m, m, 1.0);
    for (it = 0; it < t; ++it) {
      LIBXSMM_MATINIT(float, 24, &LIBXSMM_VLA_ACCESS(2, xgold, it, 0, k * n), k, n, k, 1.0);
      LIBXSMM_MATINIT(float, 24, &LIBXSMM_VLA_ACCESS(2, hgoldb, it, 0, m * n), m, n, m, 1.0);
      LIBXSMM_MATINIT(float, 24, &LIBXSMM_VLA_ACCESS(2, igoldb, it, 0, m * n), m, n, m, 1.0);
      LIBXSMM_MATINIT(float, 24, &LIBXSMM_VLA_ACCESS(2, fgoldb, it, 0, m * n), m, n, m, 1.0);
      LIBXSMM_MATINIT(float, 24, &LIBXSMM_VLA_ACCESS(2, ogoldb, it, 0, m * n), m, n, m, 1.0);
      LIBXSMM_MATINIT(float, 24, &LIBXSMM_VLA_ACCESS(2, cgoldb, it, 0, m * n), m, n, m, 1.0);
      LIBXSMM_MATINIT(float, 24, &LIBXSMM_VLA_ACCESS(2, dgoldb, it, 0, m * n), m, n, m, 1.0);
      LIBXSMM_MATINIT(float, 24, &LIBXSMM_VLA_ACCESS(2, djdhgold, it, 0, m * n), m, n, m, 1.0);
    }
    zero_buf(i1gold, m*n);
    zero_buf(i2gold, m*n);
    zero_buf(i3gold, m*n);
    zero_buf(f1gold, m*n);
    zero_buf(f2gold, m*n);
    zero_buf(f3gold, m*n);
    zero_buf(o1gold, m*n);
    zero_buf(o2gold, m*n);
    zero_buf(c1gold, m*n);
    zero_buf(c2gold, m*n);
    zero_buf(d1gold, m*n);
    zero_buf(d2gold, m*n);
    zero_buf(d3gold, m*n);
    zero_buf(d4gold, m*n);
    zero_buf(deltagoldt, m*n*t);
    zero_buf(djddgoldt, m*n*t);
    zero_buf(djdigoldt, m*n*t);
    zero_buf(djdfgoldt, m*n*t);
    zero_buf(djdogoldt, m*n*t);
    zero_buf(djdcgoldt, m*n*t);
    zero_buf(djdxgoldt, k*n*t);
    zero_buf(djdwigold, m*k);
    zero_buf(djdwfgold, m*k);
    zero_buf(djdwogold, m*k);
    zero_buf(djdwcgold, m*k);
    zero_buf(djdrigold, m*m);
    zero_buf(djdrfgold, m*m);
    zero_buf(djdrogold, m*m);
    zero_buf(djdrcgold, m*m);
    zero_buf(djdbigold, m*n);
    zero_buf(djdbfgold, m*n);
    zero_buf(djdbogold, m*n);
    zero_buf(djdbcgold, m*n);
    zero_buf(wgoldTp, m*k);
    zero_buf(rgoldTp, m*m);
    zero_buf(xgoldTp, k*n);
    zero_buf(hgoldTp, m*n);
#endif
  }

  /* first touch LIBXSMM */
  if (pass == 0) {
    zero_buf(ur, m*k);
    zero_buf(uz, m*k);
    zero_buf(ug, m*k);
    zero_buf(xt, k*n*t);
    zero_buf(wr, m*m);
    zero_buf(wz, m*m);
    zero_buf(wg, m*m);
    if (reuse) {
      zero_buf(h, m*n);
    } else {
      zero_buf(h, m*n*(t+1));
    }
    zero_buf(br, m*n);
    zero_buf(bz, m*n);
    zero_buf(bg, m*n);
  }
  else {
#if 0
    zero_buf(wi, m*k);
    zero_buf(wf, m*k);
    zero_buf(wo, m*k);
    zero_buf(wc, m*k);
    zero_buf(xt, k*n*t);
    zero_buf(ri, m*m);
    zero_buf(rf, m*m);
    zero_buf(ro, m*m);
    zero_buf(rc, m*m);
    zero_buf(ht, m*n*t);
    zero_buf(bi, m*n);
    zero_buf(bf, m*n);
    zero_buf(bo, m*n);
    zero_buf(bc, m*n);
    zero_buf(djdwi, m*k);
    zero_buf(djdwf, m*k);
    zero_buf(djdwo, m*k);
    zero_buf(djdwc, m*k);
    zero_buf(djdxt, k*n*t);
    zero_buf(djdri, m*m);
    zero_buf(djdrf, m*m);
    zero_buf(djdro, m*m);
    zero_buf(djdrc, m*m);
    zero_buf(djdht, m*n*t);
    zero_buf(djdbi, m*n);
    zero_buf(djdbf, m*n);
    zero_buf(djdbo, m*n);
    zero_buf(djdbc, m*n);
#endif
  }
  LIBXSMM_VLA_DECL(2, float, x, xt, k * n);
  LIBXSMM_VLA_DECL(2, float, hnr, h, m * n);
#if 0
  LIBXSMM_VLA_DECL(2, float, djdx, djdxt, k * n);
  LIBXSMM_VLA_DECL(2, float, djdh, djdht, m * n);
  LIBXSMM_VLA_DECL(2, float, hb, ht, m * n);
#endif

  if (pass == 0) {
    handleux = libxsmm_bgemm_handle_create(nThreads, LIBXSMM_GEMM_PRECISION(float), LIBXSMM_GEMM_PRECISION(float),
      m, n, k, &bm, &bn, &bk, &b_m1, &b_n1, &b_k1, &b_k2,
      &alpha, &beta, &gemm_flags, &strategy, &order);
    handlewh = libxsmm_bgemm_handle_create(nThreads, LIBXSMM_GEMM_PRECISION(float), LIBXSMM_GEMM_PRECISION(float),
      m, n, m, &bm, &bn, &bm, &b_m1, &b_n1, &b_m1, &b_m2,
      &alpha, &beta, &gemm_flags, &strategy, &order);
    handlett = libxsmm_bgemm_handle_create(nThreads, LIBXSMM_GEMM_PRECISION(float), LIBXSMM_GEMM_PRECISION(float),
      m, n*t, k, &bm, &bn, &bk, &b_m1, &b_n1, &b_k1, &b_k2,
      &alpha, &beta, &gemm_flags, &strategy, &order);
  } else {
#if 0
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
#endif
  }

  if (LIBXSMM_NEQ(0, check)) {
    printf("##########################################\n");
    printf("#         Computing Reference ...        #\n");
    printf("##########################################\n");
    if (pass == 0) {
      for (j = 0; j < t; ++j) {
        LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &m, &n, &k, &alpha, urgold, &m, &LIBXSMM_VLA_ACCESS(2, xgold, j, 0, k * n), &k, &beta0, r1gold, &m);
        LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &m, &n, &m, &alpha, wrgold, &m, hgold, &m, &beta0, r2gold, &m);
        matrix_add(m*n, r1gold, r2gold, rgold);
        matrix_add(m*n, rgold, brgold, rgold);
        matrix_sigmoid(m*n, rgold, rgold); /*sigmoid*/
        LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &m, &n, &k, &alpha, uzgold, &m, &LIBXSMM_VLA_ACCESS(2, xgold, j, 0, k * n), &k, &beta0, z1gold, &m);
        LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &m, &n, &m, &alpha, wzgold, &m, hgold, &m, &beta0, z2gold, &m);
        matrix_add(m*n, z1gold, z2gold, zgold);
        matrix_add(m*n, zgold, bzgold, zgold);
        matrix_sigmoid(m*n, zgold, zgold); /*sigmoid*/
        LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &m, &n, &k, &alpha, uggold, &m, &LIBXSMM_VLA_ACCESS(2, xgold, j, 0, k * n), &k, &beta0, g1gold, &m);
        matrix_eltwise_mult(m*n, rgold, hgold, g3gold); 
        LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &m, &n, &m, &alpha, wggold, &m, g3gold, &m, &beta0, g2gold, &m);
        matrix_add(m*n, g1gold, g2gold, ggold);
        matrix_add(m*n, ggold, bggold, ggold);
        matrix_tanh(m*n, ggold, ggold); /*tanh*/
        matrix_eltwise_mult(m*n, zgold, ggold, h1gold);
        matrix_complement(m*n, zgold, h3gold);
        matrix_eltwise_mult(m*n, hgold, h3gold, h2gold);
        matrix_add(m*n, h1gold, h2gold, hgold);
      }
    } else {
#if 0
      /* compute deltagold */
      matrix_copy(m * n, &LIBXSMM_VLA_ACCESS(2, djdhgold, t-1, 0, m * n), &LIBXSMM_VLA_ACCESS(2, deltagold, t-1, 0, m * n));
      /* compute djddgold */
      matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, djdhgold, t-1, 0, m * n), &LIBXSMM_VLA_ACCESS(2, ogoldb, t-1, 0, m * n), d1gold);
      matrix_tanh_inverse(m * n, &LIBXSMM_VLA_ACCESS(2, dgoldb, t-1, 0, m * n), d2gold);
      matrix_eltwise_mult(m * n, d1gold, d2gold, &LIBXSMM_VLA_ACCESS(2, djddgold, t-1, 0, m * n));
      /* compute djdcgold */
      matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, djddgold, t-1, 0, m * n), &LIBXSMM_VLA_ACCESS(2, igoldb, t-1, 0, m * n), c1gold);
      matrix_complement_square(m * n, &LIBXSMM_VLA_ACCESS(2, cgoldb, t-1, 0, m * n), c2gold);
      matrix_eltwise_mult(m * n, c1gold, c2gold, &LIBXSMM_VLA_ACCESS(2, djdcgold, t-1, 0, m * n));
      /* compute djdigold */
      matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, djddgold, t-1, 0, m * n), &LIBXSMM_VLA_ACCESS(2, cgoldb, t-1, 0, m * n), i1gold);
      matrix_complement(m * n, &LIBXSMM_VLA_ACCESS(2, igoldb, t-1, 0, m * n), i2gold);
      matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, igoldb, t-1, 0, m * n), i2gold, i3gold);
      matrix_eltwise_mult(m * n, i1gold, i3gold, &LIBXSMM_VLA_ACCESS(2, djdigold, t-1, 0, m * n));
      /* compute djdfgold */
      matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, djddgold, t-1, 0, m * n), &LIBXSMM_VLA_ACCESS(2, dgoldb, t-2, 0, m * n), f1gold);
      matrix_complement(m * n, &LIBXSMM_VLA_ACCESS(2, fgoldb, t-1, 0, m * n), f2gold);
      matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, fgoldb, t-1, 0, m * n), f2gold, f3gold);
      matrix_eltwise_mult(m * n, f1gold, f3gold, &LIBXSMM_VLA_ACCESS(2, djdfgold, t-1, 0, m * n));
      /* compute djdogold */
      matrix_tanh(m * n, &LIBXSMM_VLA_ACCESS(2, dgoldb, t-1, 0, m * n), o1gold);
      matrix_complement(m * n, &LIBXSMM_VLA_ACCESS(2, ogoldb, t-1, 0, m * n), o2gold);
      matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, deltagold, t-1, 0, m * n), o1gold, o1gold);
      matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, ogoldb, t-1, 0, m * n), o2gold, o2gold);
      matrix_eltwise_mult(m * n, o1gold, o2gold, &LIBXSMM_VLA_ACCESS(2, djdogold, t-1, 0, m * n));
      if (pass == 1 || pass == 3) {
        /* compute djdxgold */
        matrix_transpose(m, k, wigold, wgoldTp);
        LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &k, &n, &m, &alpha, wgoldTp, &k, &LIBXSMM_VLA_ACCESS(2, djdigold, t-1, 0, m * n), &m, &beta, &LIBXSMM_VLA_ACCESS(2, djdxgold, t-1, 0, k * n), &k);
        matrix_transpose(m, k, wfgold, wgoldTp);
        LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &k, &n, &m, &alpha, wgoldTp, &k, &LIBXSMM_VLA_ACCESS(2, djdfgold, t-1, 0, m * n), &m, &beta, &LIBXSMM_VLA_ACCESS(2, djdxgold, t-1, 0, k * n), &k);
        matrix_transpose(m, k, wogold, wgoldTp);
        LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &k, &n, &m, &alpha, wgoldTp, &k, &LIBXSMM_VLA_ACCESS(2, djdogold, t-1, 0, m * n), &m, &beta, &LIBXSMM_VLA_ACCESS(2, djdxgold, t-1, 0, k * n), &k);
        matrix_transpose(m, k, wcgold, wgoldTp);
        LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &k, &n, &m, &alpha, wgoldTp, &k, &LIBXSMM_VLA_ACCESS(2, djdcgold, t-1, 0, m * n), &m, &beta, &LIBXSMM_VLA_ACCESS(2, djdxgold, t-1, 0, k * n), &k);
      }
      for (j = t-2; j >= 0; --j) {
        /* compute deltagold */
        matrix_transpose(m, m, rigold, rgoldTp);
        LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &m, &n, &m, &alpha, rgoldTp, &m, &LIBXSMM_VLA_ACCESS(2, djdigold, j, 0, m * n), &m, &beta, &LIBXSMM_VLA_ACCESS(2, deltagold, j+1, 0, m * n), &m);
        matrix_transpose(m, m, rfgold, rgoldTp);
        LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &m, &n, &m, &alpha, rgoldTp, &m, &LIBXSMM_VLA_ACCESS(2, djdfgold, j, 0, m * n), &m, &beta, &LIBXSMM_VLA_ACCESS(2, deltagold, j+1, 0, m * n), &m);
        matrix_transpose(m, m, rogold, rgoldTp);
        LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &m, &n, &m, &alpha, rgoldTp, &m, &LIBXSMM_VLA_ACCESS(2, djdogold, j, 0, m * n), &m, &beta, &LIBXSMM_VLA_ACCESS(2, deltagold, j+1, 0, m * n), &m);
        matrix_transpose(m, m, rcgold, rgoldTp);
        LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &m, &n, &m, &alpha, rgoldTp, &m, &LIBXSMM_VLA_ACCESS(2, djdcgold, j, 0, m * n), &m, &beta, &LIBXSMM_VLA_ACCESS(2, deltagold, j+1, 0, m * n), &m);
        matrix_add(m * n, &LIBXSMM_VLA_ACCESS(2, djdhgold, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, deltagold, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, deltagold, j, 0, m * n));
        /* compute djddgold */
        matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, djdhgold, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, ogoldb, j, 0, m * n), d1gold);
        matrix_tanh_inverse(m * n, &LIBXSMM_VLA_ACCESS(2, dgoldb, j, 0, m * n), d2gold);
        matrix_eltwise_mult(m * n, d1gold, d2gold, d3gold);
        matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, deltagold, j+1, 0, m * n), &LIBXSMM_VLA_ACCESS(2, fgoldb, j+1, 0, m * n), d4gold);
        matrix_add(m * n, d3gold, d4gold, &LIBXSMM_VLA_ACCESS(2, djddgold, j, 0, m * n));
        /* compute djdcgold */
        matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, djddgold, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, igoldb, j, 0, m * n), c1gold);
      matrix_complement_square(m * n, &LIBXSMM_VLA_ACCESS(2, cgoldb, j, 0, m * n), c2gold);
      matrix_eltwise_mult(m * n, c1gold, c2gold, &LIBXSMM_VLA_ACCESS(2, djdcgold, j, 0, m * n));
        /* compute djdigold */
        matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, djddgold, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, cgoldb, j, 0, m * n), i1gold);
        matrix_complement(m * n, &LIBXSMM_VLA_ACCESS(2, igoldb, j, 0, m * n), i2gold);
        matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, igoldb, j, 0, m * n), i2gold, i3gold);
        matrix_eltwise_mult(m * n, i1gold, i3gold, &LIBXSMM_VLA_ACCESS(2, djdigold, j, 0, m * n));
        /* compute djdfgold */
        if (j >= 1) {
          matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, djddgold, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, dgoldb, j-1, 0, m * n), f1gold);
          matrix_complement(m * n, &LIBXSMM_VLA_ACCESS(2, fgoldb, j, 0, m * n), f2gold);
          matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, fgoldb, j, 0, m * n), f2gold, f3gold);
          matrix_eltwise_mult(m * n, f1gold, f3gold, &LIBXSMM_VLA_ACCESS(2, djdfgold, j, 0, m * n));
        } else {
          /* djdf is zero for j == 0 */
          /* init_buf( 0, &LIBXSMM_VLA_ACCESS(2, djdfgold, j, 0, m * n), m, n, ldz, 0.0); */
          zero_buf(&LIBXSMM_VLA_ACCESS(2, djdfgold, j, 0, m * n), m*n);
        }
        /* compute djdogold */
        matrix_tanh(m * n, &LIBXSMM_VLA_ACCESS(2, dgoldb, j, 0, m * n), o1gold);
        matrix_complement(m * n, &LIBXSMM_VLA_ACCESS(2, ogoldb, j, 0, m * n), o2gold);
        matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, deltagold, j, 0, m * n), o1gold, o1gold);
        matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, ogoldb, j, 0, m * n), o2gold, o2gold);
        matrix_eltwise_mult(m * n, o1gold, o2gold, &LIBXSMM_VLA_ACCESS(2, djdogold, j, 0, m * n));
        if (pass == 1 || pass == 3) {
          /* compute djdxgold */
          matrix_transpose(m, k, wigold, wgoldTp);
          LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &k, &n, &m, &alpha, wgoldTp, &k, &LIBXSMM_VLA_ACCESS(2, djdigold, j, 0, m * n), &m, &beta, &LIBXSMM_VLA_ACCESS(2, djdxgold, j, 0, k * n), &k);
          matrix_transpose(m, k, wfgold, wgoldTp);
          LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &k, &n, &m, &alpha, wgoldTp, &k, &LIBXSMM_VLA_ACCESS(2, djdfgold, j, 0, m * n), &m, &beta, &LIBXSMM_VLA_ACCESS(2, djdxgold, j, 0, k * n), &k);
          matrix_transpose(m, k, wogold, wgoldTp);
          LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &k, &n, &m, &alpha, wgoldTp, &k, &LIBXSMM_VLA_ACCESS(2, djdogold, j, 0, m * n), &m, &beta, &LIBXSMM_VLA_ACCESS(2, djdxgold, j, 0, k * n), &k);
          matrix_transpose(m, k, wcgold, wgoldTp);
          LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &k, &n, &m, &alpha, wgoldTp, &k, &LIBXSMM_VLA_ACCESS(2, djdcgold, j, 0, m * n), &m, &beta, &LIBXSMM_VLA_ACCESS(2, djdxgold, j, 0, k * n), &k);
        }
      }
      if (pass == 2 || pass == 3) {
        /* compute djdwgold */
        for (j = 0; j < t; ++j) {
          matrix_transpose(k, n, &LIBXSMM_VLA_ACCESS(2, xgold, j, 0, k * n), xgoldTp);
          LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &m, &k, &n, &alpha, &LIBXSMM_VLA_ACCESS(2, djdigold, j, 0, m * n), &m, xgoldTp, &n, &beta, djdwigold, &m);
          LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &m, &k, &n, &alpha, &LIBXSMM_VLA_ACCESS(2, djdfgold, j, 0, m * n), &m, xgoldTp, &n, &beta, djdwfgold, &m);
          LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &m, &k, &n, &alpha, &LIBXSMM_VLA_ACCESS(2, djdogold, j, 0, m * n), &m, xgoldTp, &n, &beta, djdwogold, &m);
          LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &m, &k, &n, &alpha, &LIBXSMM_VLA_ACCESS(2, djdcgold, j, 0, m * n), &m, xgoldTp, &n, &beta, djdwcgold, &m);
        }
        /* compute djdrgold */
        for (j = 0; j < t-1; ++j) {
          matrix_transpose(m, n, &LIBXSMM_VLA_ACCESS(2, hgoldb, j, 0, m * n), hgoldTp);
          LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &m, &m, &n, &alpha, &LIBXSMM_VLA_ACCESS(2, djdigold, j+1, 0, m * n), &m, hgoldTp, &n, &beta, djdrigold, &m);
          LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &m, &m, &n, &alpha, &LIBXSMM_VLA_ACCESS(2, djdfgold, j+1, 0, m * n), &m, hgoldTp, &n, &beta, djdrfgold, &m);
          LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &m, &m, &n, &alpha, &LIBXSMM_VLA_ACCESS(2, djdogold, j+1, 0, m * n), &m, hgoldTp, &n, &beta, djdrogold, &m);
          LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &m, &m, &n, &alpha, &LIBXSMM_VLA_ACCESS(2, djdcgold, j+1, 0, m * n), &m, hgoldTp, &n, &beta, djdrcgold, &m);
        }
        /* compute djdbgold */
        for (j = 0; j < t-1; j++) {
          matrix_add(m * n, &LIBXSMM_VLA_ACCESS(2, djdigold, j, 0, m * n), djdbigold, djdbigold);
          matrix_add(m * n, &LIBXSMM_VLA_ACCESS(2, djdfgold, j, 0, m * n), djdbfgold, djdbfgold);
          matrix_add(m * n, &LIBXSMM_VLA_ACCESS(2, djdogold, j, 0, m * n), djdbogold, djdbogold);
          matrix_add(m * n, &LIBXSMM_VLA_ACCESS(2, djdcgold, j, 0, m * n), djdbcgold, djdbcgold);
        }
      }
#endif
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
    grucell_desc.nThreads = nThreads;
    grucell_desc.m = m;
    grucell_desc.n = n;
    grucell_desc.k = k;
    grucell_desc.t = t;
    grucell_desc.bm = bm;
    grucell_desc.bn = bn;
    grucell_desc.bk = bk;
    grucell_desc.reuse = reuse;
    grucell_desc.pass = pass;
    grucell_desc.datatype_in = LIBXSMM_DNN_DATATYPE_F32;
    grucell_desc.datatype_out = LIBXSMM_DNN_DATATYPE_F32;
    grucell_desc.buffer_format = LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM;

    libxsmm_handle = libxsmm_dnn_create_grucell( grucell_desc, &status );
    CHKERR_LIBXSMM_DNN( status );

    /* setup LIBXSMM buffers and filter */
    libxsmm_layout = libxsmm_dnn_grucell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_GRU_REGULAR_INPUT, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_input = libxsmm_dnn_link_tensor( libxsmm_layout, xt, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_grucell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_GRU_REGULAR_WEIGHT_R, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_weight_r = libxsmm_dnn_link_tensor( libxsmm_layout, ur, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_grucell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_GRU_REGULAR_WEIGHT_Z, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_weight_z = libxsmm_dnn_link_tensor( libxsmm_layout, uz, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_grucell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_GRU_REGULAR_WEIGHT_G, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_weight_g = libxsmm_dnn_link_tensor( libxsmm_layout, ug, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_grucell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_GRU_REGULAR_RECUR_WEIGHT_R, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_recur_weight_r = libxsmm_dnn_link_tensor( libxsmm_layout, wr, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_grucell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_GRU_REGULAR_RECUR_WEIGHT_Z, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_recur_weight_z = libxsmm_dnn_link_tensor( libxsmm_layout, wz, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_grucell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_GRU_REGULAR_RECUR_WEIGHT_G, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_recur_weight_g = libxsmm_dnn_link_tensor( libxsmm_layout, wg, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_grucell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_GRU_REGULAR_BIAS_R, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_bias_r = libxsmm_dnn_link_tensor( libxsmm_layout, br, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_grucell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_GRU_REGULAR_BIAS_Z, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_bias_z = libxsmm_dnn_link_tensor( libxsmm_layout, bz, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_grucell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_GRU_REGULAR_BIAS_G, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_bias_g = libxsmm_dnn_link_tensor( libxsmm_layout, bg, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    if (pass == 0) {
      libxsmm_layout = libxsmm_dnn_grucell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_GRU_REGULAR_HIDDEN_STATE, &status ); CHKERR_LIBXSMM_DNN( status );
      libxsmm_hidden_state = libxsmm_dnn_link_tensor( libxsmm_layout, h, &status ); CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
    } else {
#if 0
      libxsmm_layout = libxsmm_dnn_grucell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_GRU_REGULAR_HIDDEN_STATE, &status ); CHKERR_LIBXSMM_DNN( status );
      libxsmm_hidden_state = libxsmm_dnn_link_tensor( libxsmm_layout, ht, &status ); CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

      libxsmm_layout = libxsmm_dnn_grucell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_GRU_GRADIENT_INPUT, &status ); CHKERR_LIBXSMM_DNN( status );
      libxsmm_dinput = libxsmm_dnn_link_tensor( libxsmm_layout, djdxt, &status ); CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

      libxsmm_layout = libxsmm_dnn_grucell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_GRU_GRADIENT_WEIGHT_I, &status ); CHKERR_LIBXSMM_DNN( status );
      libxsmm_dweight_i = libxsmm_dnn_link_tensor( libxsmm_layout, djdwi, &status ); CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

      libxsmm_layout = libxsmm_dnn_grucell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_GRU_GRADIENT_WEIGHT_F, &status ); CHKERR_LIBXSMM_DNN( status );
      libxsmm_dweight_f = libxsmm_dnn_link_tensor( libxsmm_layout, djdwf, &status ); CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

      libxsmm_layout = libxsmm_dnn_grucell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_GRU_GRADIENT_WEIGHT_O, &status ); CHKERR_LIBXSMM_DNN( status );
      libxsmm_dweight_o = libxsmm_dnn_link_tensor( libxsmm_layout, djdwo, &status ); CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

      libxsmm_layout = libxsmm_dnn_grucell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_GRU_GRADIENT_WEIGHT_C, &status ); CHKERR_LIBXSMM_DNN( status );
      libxsmm_dweight_c = libxsmm_dnn_link_tensor( libxsmm_layout, djdwc, &status ); CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

      libxsmm_layout = libxsmm_dnn_grucell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_GRU_GRADIENT_RECUR_WEIGHT_I, &status ); CHKERR_LIBXSMM_DNN( status );
      libxsmm_drecur_weight_i = libxsmm_dnn_link_tensor( libxsmm_layout, djdri, &status ); CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

      libxsmm_layout = libxsmm_dnn_grucell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_GRU_GRADIENT_RECUR_WEIGHT_F, &status ); CHKERR_LIBXSMM_DNN( status );
      libxsmm_drecur_weight_f = libxsmm_dnn_link_tensor( libxsmm_layout, djdrf, &status ); CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

      libxsmm_layout = libxsmm_dnn_grucell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_GRU_GRADIENT_RECUR_WEIGHT_O, &status ); CHKERR_LIBXSMM_DNN( status );
      libxsmm_drecur_weight_o = libxsmm_dnn_link_tensor( libxsmm_layout, djdro, &status ); CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

      libxsmm_layout = libxsmm_dnn_grucell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_GRU_GRADIENT_RECUR_WEIGHT_C, &status ); CHKERR_LIBXSMM_DNN( status );
      libxsmm_drecur_weight_c = libxsmm_dnn_link_tensor( libxsmm_layout, djdrc, &status ); CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

      libxsmm_layout = libxsmm_dnn_grucell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_GRU_GRADIENT_HIDDEN_STATE, &status ); CHKERR_LIBXSMM_DNN( status );
      libxsmm_dhidden_state = libxsmm_dnn_link_tensor( libxsmm_layout, djdht, &status ); CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

      libxsmm_layout = libxsmm_dnn_grucell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_GRU_GRADIENT_BIAS_I, &status ); CHKERR_LIBXSMM_DNN( status );
      libxsmm_dbias_i = libxsmm_dnn_link_tensor( libxsmm_layout, djdbi, &status ); CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

      libxsmm_layout = libxsmm_dnn_grucell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_GRU_GRADIENT_BIAS_F, &status ); CHKERR_LIBXSMM_DNN( status );
      libxsmm_dbias_f = libxsmm_dnn_link_tensor( libxsmm_layout, djdbf, &status ); CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

      libxsmm_layout = libxsmm_dnn_grucell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_GRU_GRADIENT_BIAS_O, &status ); CHKERR_LIBXSMM_DNN( status );
      libxsmm_dbias_o = libxsmm_dnn_link_tensor( libxsmm_layout, djdbo, &status ); CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

      libxsmm_layout = libxsmm_dnn_grucell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_GRU_GRADIENT_BIAS_C, &status ); CHKERR_LIBXSMM_DNN( status );
      libxsmm_dbias_c = libxsmm_dnn_link_tensor( libxsmm_layout, djdbc, &status ); CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
#endif
    }

    /* copy in data to LIBXSMM format */
    if (pass == 0) {
      CHKERR_LIBXSMM_DNN( libxsmm_bgemm_copyin_a(handleux, urgold, &m, ur) );
      CHKERR_LIBXSMM_DNN( libxsmm_bgemm_copyin_a(handleux, uzgold, &m, uz) );
      CHKERR_LIBXSMM_DNN( libxsmm_bgemm_copyin_a(handleux, uggold, &m, ug) );
      for (it = 0; it < t; ++it) {
        CHKERR_LIBXSMM_DNN( libxsmm_bgemm_copyin_b(handleux, &LIBXSMM_VLA_ACCESS(2, xgold, it, 0, m * n), &m, &LIBXSMM_VLA_ACCESS(2, x, it, 0, k * n)) );
      }
      CHKERR_LIBXSMM_DNN( libxsmm_bgemm_copyin_a(handlewh, wrgold, &m, wr) );
      CHKERR_LIBXSMM_DNN( libxsmm_bgemm_copyin_a(handlewh, wzgold, &m, wz) );
      CHKERR_LIBXSMM_DNN( libxsmm_bgemm_copyin_a(handlewh, wggold, &m, wg) );
      if (reuse) {
        CHKERR_LIBXSMM_DNN( libxsmm_bgemm_copyin_b(handlewh, hgold_temp, &m, h) );
      } else {
        CHKERR_LIBXSMM_DNN( libxsmm_bgemm_copyin_b(handlewh, hgold_temp, &m, &LIBXSMM_VLA_ACCESS(2, hnr, 0, 0, m * n)) );
        zero_buf(&LIBXSMM_VLA_ACCESS(2, hnr, 1, 0, m * n), m*n*t);
      }
      CHKERR_LIBXSMM_DNN( libxsmm_bgemm_copyin_b(handlewh, brgold, &m, br) );
      CHKERR_LIBXSMM_DNN( libxsmm_bgemm_copyin_b(handlewh, bzgold, &m, bz) );
      CHKERR_LIBXSMM_DNN( libxsmm_bgemm_copyin_b(handlewh, bggold, &m, bg) );
    } else {
#if 0
      matrix_transpose(m, k, wigold, wgoldTp);
      CHKERR_LIBXSMM_DNN( libxsmm_bgemm_copyin_a(handlewd, wgoldTp, &k, wi) );
      matrix_transpose(m, k, wfgold, wgoldTp);
      CHKERR_LIBXSMM_DNN( libxsmm_bgemm_copyin_a(handlewd, wgoldTp, &k, wf) );
      matrix_transpose(m, k, wogold, wgoldTp);
      CHKERR_LIBXSMM_DNN( libxsmm_bgemm_copyin_a(handlewd, wgoldTp, &k, wo) );
      matrix_transpose(m, k, wcgold, wgoldTp);
      CHKERR_LIBXSMM_DNN( libxsmm_bgemm_copyin_a(handlewd, wgoldTp, &k, wc) );
      for (it = 0; it < t; ++it) {
        matrix_transpose(m, n, &LIBXSMM_VLA_ACCESS(2, hgoldb, it, 0, m * n), hgoldTp);
        CHKERR_LIBXSMM_DNN( libxsmm_bgemm_copyin_b(handleuh, hgoldTp, &m, &LIBXSMM_VLA_ACCESS(2, hb, it, 0, m * n)) );
        matrix_transpose(k, n, &LIBXSMM_VLA_ACCESS(2, xgold, it, 0, k * n), xgoldTp);
        CHKERR_LIBXSMM_DNN( libxsmm_bgemm_copyin_b(handlett, xgoldTp, &k, &LIBXSMM_VLA_ACCESS(2, x, it, 0, k * n)) );
        CHKERR_LIBXSMM_DNN( libxsmm_bgemm_copyin_b(handlewd, &LIBXSMM_VLA_ACCESS(2, djdhgold, it, 0, m * n), &m, &LIBXSMM_VLA_ACCESS(2, djdh, it, 0, m * n)) );
      }
      matrix_transpose(m, m, rigold, rgoldTp);
      CHKERR_LIBXSMM_DNN( libxsmm_bgemm_copyin_a(handlewx, rgoldTp, &m, ri) );
      matrix_transpose(m, m, rfgold, rgoldTp);
      CHKERR_LIBXSMM_DNN( libxsmm_bgemm_copyin_a(handlewx, rgoldTp, &m, rf) );
      matrix_transpose(m, m, rogold, rgoldTp);
      CHKERR_LIBXSMM_DNN( libxsmm_bgemm_copyin_a(handlewx, rgoldTp, &m, ro) );
      matrix_transpose(m, m, rcgold, rgoldTp);
      CHKERR_LIBXSMM_DNN( libxsmm_bgemm_copyin_a(handlewx, rgoldTp, &m, rc) );
#endif
    }

    /* bind buffers and filter to handle */
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_grucell_bind_tensor( libxsmm_handle, libxsmm_input, LIBXSMM_DNN_GRU_REGULAR_INPUT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_grucell_bind_tensor( libxsmm_handle, libxsmm_hidden_state, LIBXSMM_DNN_GRU_REGULAR_HIDDEN_STATE ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_grucell_bind_tensor( libxsmm_handle, libxsmm_weight_r, LIBXSMM_DNN_GRU_REGULAR_WEIGHT_R ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_grucell_bind_tensor( libxsmm_handle, libxsmm_weight_z, LIBXSMM_DNN_GRU_REGULAR_WEIGHT_Z ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_grucell_bind_tensor( libxsmm_handle, libxsmm_weight_g, LIBXSMM_DNN_GRU_REGULAR_WEIGHT_G ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_grucell_bind_tensor( libxsmm_handle, libxsmm_recur_weight_r, LIBXSMM_DNN_GRU_REGULAR_RECUR_WEIGHT_R ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_grucell_bind_tensor( libxsmm_handle, libxsmm_recur_weight_z, LIBXSMM_DNN_GRU_REGULAR_RECUR_WEIGHT_Z ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_grucell_bind_tensor( libxsmm_handle, libxsmm_recur_weight_g, LIBXSMM_DNN_GRU_REGULAR_RECUR_WEIGHT_G ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_grucell_bind_tensor( libxsmm_handle, libxsmm_bias_r, LIBXSMM_DNN_GRU_REGULAR_BIAS_R ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_grucell_bind_tensor( libxsmm_handle, libxsmm_bias_z, LIBXSMM_DNN_GRU_REGULAR_BIAS_Z ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_grucell_bind_tensor( libxsmm_handle, libxsmm_bias_g, LIBXSMM_DNN_GRU_REGULAR_BIAS_G ) );

#if 0
    if (pass != 0) {
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_grucell_bind_tensor( libxsmm_handle, libxsmm_dinput, LIBXSMM_DNN_GRU_GRADIENT_INPUT ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_grucell_bind_tensor( libxsmm_handle, libxsmm_dhidden_state, LIBXSMM_DNN_GRU_GRADIENT_HIDDEN_STATE ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_grucell_bind_tensor( libxsmm_handle, libxsmm_dweight_i, LIBXSMM_DNN_GRU_GRADIENT_WEIGHT_I ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_grucell_bind_tensor( libxsmm_handle, libxsmm_dweight_f, LIBXSMM_DNN_GRU_GRADIENT_WEIGHT_F ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_grucell_bind_tensor( libxsmm_handle, libxsmm_dweight_o, LIBXSMM_DNN_GRU_GRADIENT_WEIGHT_O ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_grucell_bind_tensor( libxsmm_handle, libxsmm_dweight_c, LIBXSMM_DNN_GRU_GRADIENT_WEIGHT_C ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_grucell_bind_tensor( libxsmm_handle, libxsmm_drecur_weight_i, LIBXSMM_DNN_GRU_GRADIENT_RECUR_WEIGHT_I ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_grucell_bind_tensor( libxsmm_handle, libxsmm_drecur_weight_f, LIBXSMM_DNN_GRU_GRADIENT_RECUR_WEIGHT_F ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_grucell_bind_tensor( libxsmm_handle, libxsmm_drecur_weight_o, LIBXSMM_DNN_GRU_GRADIENT_RECUR_WEIGHT_O ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_grucell_bind_tensor( libxsmm_handle, libxsmm_drecur_weight_c, LIBXSMM_DNN_GRU_GRADIENT_RECUR_WEIGHT_C ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_grucell_bind_tensor( libxsmm_handle, libxsmm_dbias_i, LIBXSMM_DNN_GRU_GRADIENT_BIAS_I ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_grucell_bind_tensor( libxsmm_handle, libxsmm_dbias_f, LIBXSMM_DNN_GRU_GRADIENT_BIAS_F ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_grucell_bind_tensor( libxsmm_handle, libxsmm_dbias_o, LIBXSMM_DNN_GRU_GRADIENT_BIAS_O ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_grucell_bind_tensor( libxsmm_handle, libxsmm_dbias_c, LIBXSMM_DNN_GRU_GRADIENT_BIAS_C ) );
    }
#endif

    /* let's allocate and bind scratch */
    if (pass == 0) {
      scratch_size = libxsmm_dnn_grucell_get_scratch_size( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_FWD, &status );
      CHKERR_LIBXSMM_DNN( status );
      scratch = libxsmm_aligned_malloc( scratch_size, 2097152 );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_grucell_bind_scratch( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_FWD, scratch ) );
    } else {
#if 0
      scratch_size = libxsmm_dnn_grucell_get_scratch_size( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_ALL, &status );
      CHKERR_LIBXSMM_DNN( status );
      scratch = libxsmm_aligned_malloc( scratch_size, 2097152 );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_grucell_bind_scratch( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_ALL, scratch ) );
#endif
    }
    zero_buf( (float*)scratch, scratch_size/4 );

    /* let's allocate and bind internalstate */
    if (pass == 0) {
      internalstate_size = libxsmm_dnn_grucell_get_internalstate_size( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_FWD, &status );
      CHKERR_LIBXSMM_DNN( status );
      internalstate = libxsmm_aligned_malloc( internalstate_size, 2097152 );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_grucell_bind_internalstate( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_FWD, internalstate ) );
    } else {
#if 0
      internalstate_size = libxsmm_dnn_grucell_get_internalstate_size( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_ALL, &status );
      CHKERR_LIBXSMM_DNN( status );
      internalstate = libxsmm_aligned_malloc( internalstate_size, 2097152 );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_grucell_bind_internalstate( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_ALL, internalstate ) );
#endif
    }
    zero_buf( (float*)internalstate, internalstate_size/4 );
#if 0
    if (pass != 0) {
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_grucell_assign_internalstate( libxsmm_handle, igoldt, fgoldt, ogoldt, cgoldt, dgoldt ) );
    }
#endif

    if ((pass == 0) && LIBXSMM_NEQ(0, check)) {
      printf("##########################################\n");
      printf("#   Correctness - FWD (custom-Storage)   #\n");
      printf("##########################################\n");
      /* run LIBXSMM LSTM */
#if defined(_OPENMP)
#     pragma omp parallel
#endif
      {
#if defined(_OPENMP)
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        CHKERR_LIBXSMM_DNN( libxsmm_dnn_grucell_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_FWD, 0, tid ) );
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

#if 0
    if ( (pass == 1) && LIBXSMM_NEQ(0, check) ) {
      printf("##########################################\n");
      printf("#   Correctness - BWD (custom-Storage)   #\n");
      printf("##########################################\n");
      /* run LIBXSMM LSTM */
#if defined(_OPENMP)
#     pragma omp parallel
#endif
      {
#if defined(_OPENMP)
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        CHKERR_LIBXSMM_DNN( libxsmm_dnn_grucell_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_BWD, 0, tid ) );
      }

      /* copy out data */
      LIBXSMM_VLA_DECL(2, float, djdxtest, djdxtestt, k * n);
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
      /* run LIBXSMM LSTM */
#if defined(_OPENMP)
#     pragma omp parallel
#endif
      {
#if defined(_OPENMP)
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        CHKERR_LIBXSMM_DNN( libxsmm_dnn_grucell_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_UPD, 0, tid ) );
      }

      /* copy out data */
      LIBXSMM_VLA_DECL(2, float, djdw4test, djdwtest, m * k);
      LIBXSMM_VLA_DECL(2, float, djdr4test, djdrtest, m * m);
      LIBXSMM_VLA_DECL(2, float, djdb4test, djdbtest, m * n);
      CHKERR_LIBXSMM_DNN( libxsmm_bgemm_copyout_c(handlett, djdwi, &m, &LIBXSMM_VLA_ACCESS(2, djdw4test, 0, 0, m * k)) );
      CHKERR_LIBXSMM_DNN( libxsmm_bgemm_copyout_c(handlett, djdwf, &m, &LIBXSMM_VLA_ACCESS(2, djdw4test, 1, 0, m * k)) );
      CHKERR_LIBXSMM_DNN( libxsmm_bgemm_copyout_c(handlett, djdwo, &m, &LIBXSMM_VLA_ACCESS(2, djdw4test, 2, 0, m * k)) );
      CHKERR_LIBXSMM_DNN( libxsmm_bgemm_copyout_c(handlett, djdwc, &m, &LIBXSMM_VLA_ACCESS(2, djdw4test, 3, 0, m * k)) );
      CHKERR_LIBXSMM_DNN( libxsmm_bgemm_copyout_c(handleuh, djdri, &m, &LIBXSMM_VLA_ACCESS(2, djdr4test, 0, 0, m * m)) );
      CHKERR_LIBXSMM_DNN( libxsmm_bgemm_copyout_c(handleuh, djdrf, &m, &LIBXSMM_VLA_ACCESS(2, djdr4test, 1, 0, m * m)) );
      CHKERR_LIBXSMM_DNN( libxsmm_bgemm_copyout_c(handleuh, djdro, &m, &LIBXSMM_VLA_ACCESS(2, djdr4test, 2, 0, m * m)) );
      CHKERR_LIBXSMM_DNN( libxsmm_bgemm_copyout_c(handleuh, djdrc, &m, &LIBXSMM_VLA_ACCESS(2, djdr4test, 3, 0, m * m)) );
      CHKERR_LIBXSMM_DNN( libxsmm_bgemm_copyout_c(handleuh, djdbi, &m, &LIBXSMM_VLA_ACCESS(2, djdb4test, 0, 0, m * n)) );
      CHKERR_LIBXSMM_DNN( libxsmm_bgemm_copyout_c(handleuh, djdbf, &m, &LIBXSMM_VLA_ACCESS(2, djdb4test, 1, 0, m * n)) );
      CHKERR_LIBXSMM_DNN( libxsmm_bgemm_copyout_c(handleuh, djdbo, &m, &LIBXSMM_VLA_ACCESS(2, djdb4test, 2, 0, m * n)) );
      CHKERR_LIBXSMM_DNN( libxsmm_bgemm_copyout_c(handleuh, djdbc, &m, &LIBXSMM_VLA_ACCESS(2, djdb4test, 3, 0, m * n)) );
      LIBXSMM_VLA_DECL(2, float, djdw4, djdwgold4, m * k);
      LIBXSMM_VLA_DECL(2, float, djdr4, djdrgold4, m * m);
      LIBXSMM_VLA_DECL(2, float, djdb4, djdbgold4, m * n);
      matrix_copy(m * k, djdwigold, &LIBXSMM_VLA_ACCESS(2, djdw4, 0, 0, m * k));
      matrix_copy(m * k, djdwfgold, &LIBXSMM_VLA_ACCESS(2, djdw4, 1, 0, m * k));
      matrix_copy(m * k, djdwogold, &LIBXSMM_VLA_ACCESS(2, djdw4, 2, 0, m * k));
      matrix_copy(m * k, djdwcgold, &LIBXSMM_VLA_ACCESS(2, djdw4, 3, 0, m * k));
      matrix_copy(m * m, djdrigold, &LIBXSMM_VLA_ACCESS(2, djdr4, 0, 0, m * m));
      matrix_copy(m * m, djdrfgold, &LIBXSMM_VLA_ACCESS(2, djdr4, 1, 0, m * m));
      matrix_copy(m * m, djdrogold, &LIBXSMM_VLA_ACCESS(2, djdr4, 2, 0, m * m));
      matrix_copy(m * m, djdrcgold, &LIBXSMM_VLA_ACCESS(2, djdr4, 3, 0, m * m));
      matrix_copy(m * n, djdbigold, &LIBXSMM_VLA_ACCESS(2, djdb4, 0, 0, m * n));
      matrix_copy(m * n, djdbfgold, &LIBXSMM_VLA_ACCESS(2, djdb4, 1, 0, m * n));
      matrix_copy(m * n, djdbogold, &LIBXSMM_VLA_ACCESS(2, djdb4, 2, 0, m * n));
      matrix_copy(m * n, djdbcgold, &LIBXSMM_VLA_ACCESS(2, djdb4, 3, 0, m * n));

      /* compare */
      libxsmm_matdiff(LIBXSMM_DATATYPE_F32, m*k*4, 1, djdwgold4, djdwtest, 0, 0, &norms_upd_w);
      printf("Delta weight\n");
      printf("L1 reference  : %.25g\n", norms_upd_w.l1_ref);
      printf("L1 test       : %.25g\n", norms_upd_w.l1_tst);
      printf("L2 abs.error  : %.24f\n", norms_upd_w.l2_abs);
      printf("L2 rel.error  : %.24f\n", norms_upd_w.l2_rel);
      printf("Linf abs.error: %.24f\n", norms_upd_w.linf_abs);
      printf("Linf rel.error: %.24f\n", norms_upd_w.linf_rel);
      printf("Check-norm    : %.24f\n", norms_upd_w.normf_rel);
      libxsmm_matdiff_reduce(&diff, &norms_upd_w);

      libxsmm_matdiff(LIBXSMM_DATATYPE_F32, m*m*4, 1, djdrgold4, djdrtest, 0, 0, &norms_upd_r);
      printf("Delta recurrent weight\n");
      printf("L1 reference  : %.25g\n", norms_upd_r.l1_ref);
      printf("L1 test       : %.25g\n", norms_upd_r.l1_tst);
      printf("L2 abs.error  : %.24f\n", norms_upd_r.l2_abs);
      printf("L2 rel.error  : %.24f\n", norms_upd_r.l2_rel);
      printf("Linf abs.error: %.24f\n", norms_upd_r.linf_abs);
      printf("Linf rel.error: %.24f\n", norms_upd_r.linf_rel);
      printf("Check-norm    : %.24f\n", norms_upd_r.normf_rel);
      libxsmm_matdiff_reduce(&diff, &norms_upd_r);

      libxsmm_matdiff(LIBXSMM_DATATYPE_F32, m*n*4, 1, djdbgold4, djdbtest, 0, 0, &norms_upd_b);
      printf("Delta bias\n");
      printf("L1 reference  : %.25g\n", norms_upd_b.l1_ref);
      printf("L1 test       : %.25g\n", norms_upd_b.l1_tst);
      printf("L2 abs.error  : %.24f\n", norms_upd_b.l2_abs);
      printf("L2 rel.error  : %.24f\n", norms_upd_b.l2_rel);
      printf("Linf abs.error: %.24f\n", norms_upd_b.linf_abs);
      printf("Linf rel.error: %.24f\n", norms_upd_b.linf_rel);
      printf("Check-norm    : %.24f\n", norms_upd_b.normf_rel);
      libxsmm_matdiff_reduce(&diff, &norms_upd_b);
    }

    if ( (pass == 3) && LIBXSMM_NEQ(0, check) ) {
      printf("##########################################\n");
      printf("# Correctness - BWD+UPD (custom-Storage) #\n");
      printf("##########################################\n");
      /* run LIBXSMM LSTM */
#if defined(_OPENMP)
#     pragma omp parallel
#endif
      {
#if defined(_OPENMP)
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        CHKERR_LIBXSMM_DNN( libxsmm_dnn_grucell_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_ALL, 0, tid ) );
      }

      /* copy out data */
      LIBXSMM_VLA_DECL(2, float, djdxtest, djdxtestt, k * n);
      for (i = 0; i < t; ++i) {
        CHKERR_LIBXSMM_DNN( libxsmm_bgemm_copyout_c(handlewd, &LIBXSMM_VLA_ACCESS(2, djdx, i, 0, k * n), &k, &LIBXSMM_VLA_ACCESS(2, djdxtest, i, 0, k * n)) );
      }
      LIBXSMM_VLA_DECL(2, float, djdw4test, djdwtest, m * k);
      LIBXSMM_VLA_DECL(2, float, djdr4test, djdrtest, m * m);
      LIBXSMM_VLA_DECL(2, float, djdb4test, djdbtest, m * n);
      CHKERR_LIBXSMM_DNN( libxsmm_bgemm_copyout_c(handlett, djdwi, &m, &LIBXSMM_VLA_ACCESS(2, djdw4test, 0, 0, m * k)) );
      CHKERR_LIBXSMM_DNN( libxsmm_bgemm_copyout_c(handlett, djdwf, &m, &LIBXSMM_VLA_ACCESS(2, djdw4test, 1, 0, m * k)) );
      CHKERR_LIBXSMM_DNN( libxsmm_bgemm_copyout_c(handlett, djdwo, &m, &LIBXSMM_VLA_ACCESS(2, djdw4test, 2, 0, m * k)) );
      CHKERR_LIBXSMM_DNN( libxsmm_bgemm_copyout_c(handlett, djdwc, &m, &LIBXSMM_VLA_ACCESS(2, djdw4test, 3, 0, m * k)) );
      CHKERR_LIBXSMM_DNN( libxsmm_bgemm_copyout_c(handleuh, djdri, &m, &LIBXSMM_VLA_ACCESS(2, djdr4test, 0, 0, m * m)) );
      CHKERR_LIBXSMM_DNN( libxsmm_bgemm_copyout_c(handleuh, djdrf, &m, &LIBXSMM_VLA_ACCESS(2, djdr4test, 1, 0, m * m)) );
      CHKERR_LIBXSMM_DNN( libxsmm_bgemm_copyout_c(handleuh, djdro, &m, &LIBXSMM_VLA_ACCESS(2, djdr4test, 2, 0, m * m)) );
      CHKERR_LIBXSMM_DNN( libxsmm_bgemm_copyout_c(handleuh, djdrc, &m, &LIBXSMM_VLA_ACCESS(2, djdr4test, 3, 0, m * m)) );
      CHKERR_LIBXSMM_DNN( libxsmm_bgemm_copyout_c(handleuh, djdbi, &m, &LIBXSMM_VLA_ACCESS(2, djdb4test, 0, 0, m * n)) );
      CHKERR_LIBXSMM_DNN( libxsmm_bgemm_copyout_c(handleuh, djdbf, &m, &LIBXSMM_VLA_ACCESS(2, djdb4test, 1, 0, m * n)) );
      CHKERR_LIBXSMM_DNN( libxsmm_bgemm_copyout_c(handleuh, djdbo, &m, &LIBXSMM_VLA_ACCESS(2, djdb4test, 2, 0, m * n)) );
      CHKERR_LIBXSMM_DNN( libxsmm_bgemm_copyout_c(handleuh, djdbc, &m, &LIBXSMM_VLA_ACCESS(2, djdb4test, 3, 0, m * n)) );
      LIBXSMM_VLA_DECL(2, float, djdw4, djdwgold4, m * k);
      LIBXSMM_VLA_DECL(2, float, djdr4, djdrgold4, m * m);
      LIBXSMM_VLA_DECL(2, float, djdb4, djdbgold4, m * n);
      matrix_copy(m * k, djdwigold, &LIBXSMM_VLA_ACCESS(2, djdw4, 0, 0, m * k));
      matrix_copy(m * k, djdwfgold, &LIBXSMM_VLA_ACCESS(2, djdw4, 1, 0, m * k));
      matrix_copy(m * k, djdwogold, &LIBXSMM_VLA_ACCESS(2, djdw4, 2, 0, m * k));
      matrix_copy(m * k, djdwcgold, &LIBXSMM_VLA_ACCESS(2, djdw4, 3, 0, m * k));
      matrix_copy(m * m, djdrigold, &LIBXSMM_VLA_ACCESS(2, djdr4, 0, 0, m * m));
      matrix_copy(m * m, djdrfgold, &LIBXSMM_VLA_ACCESS(2, djdr4, 1, 0, m * m));
      matrix_copy(m * m, djdrogold, &LIBXSMM_VLA_ACCESS(2, djdr4, 2, 0, m * m));
      matrix_copy(m * m, djdrcgold, &LIBXSMM_VLA_ACCESS(2, djdr4, 3, 0, m * m));
      matrix_copy(m * n, djdbigold, &LIBXSMM_VLA_ACCESS(2, djdb4, 0, 0, m * n));
      matrix_copy(m * n, djdbfgold, &LIBXSMM_VLA_ACCESS(2, djdb4, 1, 0, m * n));
      matrix_copy(m * n, djdbogold, &LIBXSMM_VLA_ACCESS(2, djdb4, 2, 0, m * n));
      matrix_copy(m * n, djdbcgold, &LIBXSMM_VLA_ACCESS(2, djdb4, 3, 0, m * n));

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

      libxsmm_matdiff(LIBXSMM_DATATYPE_F32, m*k*4, 1, djdwgold4, djdwtest, 0, 0, &norms_upd_w);
      printf("Delta weight\n");
      printf("L1 reference  : %.25g\n", norms_upd_w.l1_ref);
      printf("L1 test       : %.25g\n", norms_upd_w.l1_tst);
      printf("L2 abs.error  : %.24f\n", norms_upd_w.l2_abs);
      printf("L2 rel.error  : %.24f\n", norms_upd_w.l2_rel);
      printf("Linf abs.error: %.24f\n", norms_upd_w.linf_abs);
      printf("Linf rel.error: %.24f\n", norms_upd_w.linf_rel);
      printf("Check-norm    : %.24f\n", norms_upd_w.normf_rel);
      libxsmm_matdiff_reduce(&diff, &norms_upd_w);

      libxsmm_matdiff(LIBXSMM_DATATYPE_F32, m*m*4, 1, djdrgold4, djdrtest, 0, 0, &norms_upd_r);
      printf("Delta recurrent weight\n");
      printf("L1 reference  : %.25g\n", norms_upd_r.l1_ref);
      printf("L1 test       : %.25g\n", norms_upd_r.l1_tst);
      printf("L2 abs.error  : %.24f\n", norms_upd_r.l2_abs);
      printf("L2 rel.error  : %.24f\n", norms_upd_r.l2_rel);
      printf("Linf abs.error: %.24f\n", norms_upd_r.linf_abs);
      printf("Linf rel.error: %.24f\n", norms_upd_r.linf_rel);
      printf("Check-norm    : %.24f\n", norms_upd_r.normf_rel);
      libxsmm_matdiff_reduce(&diff, &norms_upd_r);

      libxsmm_matdiff(LIBXSMM_DATATYPE_F32, m*n*4, 1, djdbgold4, djdbtest, 0, 0, &norms_upd_b);
      printf("Delta bias\n");
      printf("L1 reference  : %.25g\n", norms_upd_b.l1_ref);
      printf("L1 test       : %.25g\n", norms_upd_b.l1_tst);
      printf("L2 abs.error  : %.24f\n", norms_upd_b.l2_abs);
      printf("L2 rel.error  : %.24f\n", norms_upd_b.l2_rel);
      printf("Linf abs.error: %.24f\n", norms_upd_b.linf_abs);
      printf("Linf rel.error: %.24f\n", norms_upd_b.linf_rel);
      printf("Check-norm    : %.24f\n", norms_upd_b.normf_rel);
      libxsmm_matdiff_reduce(&diff, &norms_upd_b);
    }
#endif

    if ( pass == 0 ) {
      printf("##########################################\n");
      printf("#   Performance - FWD (custom-Storage)   #\n");
      printf("##########################################\n");
      /* run LIBXSMM LSTM for performance */
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
          libxsmm_dnn_grucell_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_FWD, 0, tid );
        }
      }
      l_end = libxsmm_timer_tick();
      l_total = libxsmm_timer_duration(l_start, l_end);
      flops = (((2.0 * m * n * k) + (2.0 * m * n * m) + (2.0 * m * n) + (tflops * m * n)) * 2.0 + (m * n) + (2.0 * m * n * k) + (2.0 * m * n * m) + (tflops * m * n) + 4.0 * (m * n)) * (double)t * (double)iters;

      printf("GFLOP  = %.5g\n", flops*1e-9/(double)iters);
      printf("fp time = %.5g\n", ((double)(l_total/iters)));
      printf("GFLOPS  = %.5g\n", (flops*1e-9)/l_total);

      printf("PERFDUMP,FP,%s,%i,%i,%i,%i,%i,%i,%i,%i,%.5g,%.5g\n", LIBXSMM_VERSION, nThreads, m, n, k, t, bm, bn, bk, ((double)(l_total/iters)), (flops*1e-9)/l_total);
    }

#if 0
    if ( pass == 1 ) {
      printf("##########################################\n");
      printf("#   Performance - BWD (custom-Storage)   #\n");
      printf("##########################################\n");
      /* run LIBXSMM LSTM for performance */
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
          libxsmm_dnn_grucell_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_BWD, 0, tid );
        }
      }
      l_end = libxsmm_timer_tick();
      l_total = libxsmm_timer_duration(l_start, l_end);
      flops = m * n; /* delta + delta_out */
      flops += (6.0 * m * n + tflops * m * n); /* dJdd */
      flops += (4.0 * m * n); /* dJdc */
      flops += (4.0 * m * n); /* dJdi */
      flops += (4.0 * m * n); /* dJdf */
      flops += (4.0 * m * n + tflops * m * n); /* dJdo */
      tempflops = (4.0 * m * k); /* W^T */
      tempflops += (8.0 * m * n * k); /* W^T * dJd{c, i, f, o} */
      tempflops += (3.0 * m * k); /* summation */
      flops += tempflops;
      tempflops = (4.0 * m * m); /* R^T */
      tempflops += (8.0 * m * n * m); /* R^T * dJd{c, i, f, o} */
      flops += tempflops;
      flops *= t; /* for t time steps */
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
      /* run LIBXSMM LSTM for performance */
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
          libxsmm_dnn_grucell_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_UPD, 0, tid );
        }
      }
      l_end = libxsmm_timer_tick();
      l_total = libxsmm_timer_duration(l_start, l_end);
      flops = m * n; /* delta + delta_out */
      flops += (6.0 * m * n + tflops * m * n); /* dJdd */
      flops += (4.0 * m * n); /* dJdc */
      flops += (4.0 * m * n); /* dJdi */
      flops += (4.0 * m * n); /* dJdf */
      flops += (4.0 * m * n + tflops * m * n); /* dJdo */
      tempflops = (4.0 * m * m); /* R^T */
      tempflops += (8.0 * m * n * m); /* R^T * dJd{c, i, f, o} */
      flops += tempflops;
      flops *= t; /* for t time steps */
      tempflops = k * n; /* x^T */
      tempflops += (8.0 * m * n * k); /* delta{c, i, f, o} * x^T */
      tempflops *= t; /* for t time steps */
      tempflops += (4.0 * m * k * (t-1)); /* for summation of dJdW{c, i, f, o} */
      flops += tempflops;
      tempflops = 4.0 * m * n; /* delta^T */
      tempflops += (8.0 * m * n * m); /* delta{c, i, f, o} * delta^T */
      tempflops *= (t - 1); /* for (t - 1) time steps */
      tempflops += (4.0 * m * n * (t-2)); /* for summation of dJdR{c, i, f, o} */
      flops += tempflops;
      flops += (4.0 * m * n * (t - 1)); /* delbias */
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
      /* run LIBXSMM LSTM for performance */
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
          libxsmm_dnn_grucell_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_ALL, 0, tid );
        }
      }
      l_end = libxsmm_timer_tick();
      l_total = libxsmm_timer_duration(l_start, l_end);
      flops = m * n; /* delta + delta_out */
      flops += (6.0 * m * n + tflops * m * n); /* dJdd */
      flops += (4.0 * m * n); /* dJdc */
      flops += (4.0 * m * n); /* dJdi */
      flops += (4.0 * m * n); /* dJdf */
      flops += (4.0 * m * n + tflops * m * n); /* dJdo */
      tempflops = (4.0 * m * k); /* W^T */
      tempflops += (8.0 * m * n * k); /* W^T * dJd{c, i, f, o} */
      tempflops += (3.0 * m * k); /* summation */
      flops += tempflops;
      tempflops = (4.0 * m * m); /* R^T */
      tempflops += (8.0 * m * n * m); /* R^T * dJd{c, i, f, o} */
      flops += tempflops;
      flops *= t; /* for t time steps */
      tempflops = k * n; /* x^T */
      tempflops += (8.0 * m * n * k); /* delta{c, i, f, o} * x^T */
      tempflops *= t; /* for t time steps */
      tempflops += (4.0 * m * k * (t-1)); /* for summation of dJdW{c, i, f, o} */
      flops += tempflops;
      tempflops = 4.0 * m * n; /* delta^T */
      tempflops += (8.0 * m * n * m); /* delta{c, i, f, o} * delta^T */
      tempflops *= (t - 1); /* for (t - 1) time steps */
      tempflops += (4.0 * m * n * (t-2)); /* for summation of dJdR{c, i, f, o} */
      flops += tempflops;
      flops += (4.0 * m * n * (t - 1)); /* delbias */
      flops *= iters;

      printf("GFLOP  = %.5g\n", flops*1e-9/(double)iters);
      printf("bp+wu time = %.5g\n", ((double)(l_total/iters)));
      printf("GFLOPS  = %.5g\n", (flops*1e-9)/l_total);

      printf("PERFDUMP,BP+WU,%s,%i,%i,%i,%i,%i,%i,%i,%i,%.5g,%.5g\n", LIBXSMM_VERSION, nThreads, m, n, k, t, bm, bn, bk, ((double)(l_total/iters)), (flops*1e-9)/l_total);
    }
#endif

    /* clean-up */
    if (pass == 0) {
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_grucell_release_scratch( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_FWD ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_grucell_release_internalstate( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_FWD ) );
    } else {
#if 0
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_grucell_release_scratch( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_ALL ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_grucell_release_internalstate( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_ALL ) );
#endif
    }
    libxsmm_free(scratch);
    libxsmm_free(internalstate);
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_grucell_release_tensor( libxsmm_handle, LIBXSMM_DNN_GRU_REGULAR_INPUT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_grucell_release_tensor( libxsmm_handle, LIBXSMM_DNN_GRU_REGULAR_HIDDEN_STATE ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_grucell_release_tensor( libxsmm_handle, LIBXSMM_DNN_GRU_REGULAR_WEIGHT_R ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_grucell_release_tensor( libxsmm_handle, LIBXSMM_DNN_GRU_REGULAR_WEIGHT_Z ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_grucell_release_tensor( libxsmm_handle, LIBXSMM_DNN_GRU_REGULAR_WEIGHT_G ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_grucell_release_tensor( libxsmm_handle, LIBXSMM_DNN_GRU_REGULAR_RECUR_WEIGHT_R ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_grucell_release_tensor( libxsmm_handle, LIBXSMM_DNN_GRU_REGULAR_RECUR_WEIGHT_Z ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_grucell_release_tensor( libxsmm_handle, LIBXSMM_DNN_GRU_REGULAR_RECUR_WEIGHT_G ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_grucell_release_tensor( libxsmm_handle, LIBXSMM_DNN_GRU_REGULAR_BIAS_R ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_grucell_release_tensor( libxsmm_handle, LIBXSMM_DNN_GRU_REGULAR_BIAS_Z ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_grucell_release_tensor( libxsmm_handle, LIBXSMM_DNN_GRU_REGULAR_BIAS_G ) );
#if 0
    if (pass != 0) {
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_grucell_release_tensor( libxsmm_handle, LIBXSMM_DNN_GRU_GRADIENT_INPUT ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_grucell_release_tensor( libxsmm_handle, LIBXSMM_DNN_GRU_GRADIENT_HIDDEN_STATE ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_grucell_release_tensor( libxsmm_handle, LIBXSMM_DNN_GRU_GRADIENT_WEIGHT_I ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_grucell_release_tensor( libxsmm_handle, LIBXSMM_DNN_GRU_GRADIENT_WEIGHT_F ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_grucell_release_tensor( libxsmm_handle, LIBXSMM_DNN_GRU_GRADIENT_WEIGHT_O ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_grucell_release_tensor( libxsmm_handle, LIBXSMM_DNN_GRU_GRADIENT_WEIGHT_C ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_grucell_release_tensor( libxsmm_handle, LIBXSMM_DNN_GRU_GRADIENT_RECUR_WEIGHT_I ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_grucell_release_tensor( libxsmm_handle, LIBXSMM_DNN_GRU_GRADIENT_RECUR_WEIGHT_F ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_grucell_release_tensor( libxsmm_handle, LIBXSMM_DNN_GRU_GRADIENT_RECUR_WEIGHT_O ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_grucell_release_tensor( libxsmm_handle, LIBXSMM_DNN_GRU_GRADIENT_RECUR_WEIGHT_C ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_grucell_release_tensor( libxsmm_handle, LIBXSMM_DNN_GRU_GRADIENT_BIAS_I ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_grucell_release_tensor( libxsmm_handle, LIBXSMM_DNN_GRU_GRADIENT_BIAS_F ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_grucell_release_tensor( libxsmm_handle, LIBXSMM_DNN_GRU_GRADIENT_BIAS_O ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_grucell_release_tensor( libxsmm_handle, LIBXSMM_DNN_GRU_GRADIENT_BIAS_C ) );
    }
#endif
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_input ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_hidden_state ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_weight_r ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_weight_z ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_weight_g ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_recur_weight_r ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_recur_weight_z ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_recur_weight_g ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_bias_r ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_bias_z ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_bias_g ) );
#if 0
    if (pass != 0) {
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_dinput ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_dhidden_state ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_dweight_i ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_dweight_f ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_dweight_o ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_dweight_c ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_drecur_weight_i ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_drecur_weight_f ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_drecur_weight_o ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_drecur_weight_c ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_dbias_i ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_dbias_f ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_dbias_o ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_dbias_c ) );
    }
#endif
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_grucell( libxsmm_handle ) );
  }

  /* deallocate data */
  if (pass == 0) {
    libxsmm_free(urgold);
    libxsmm_free(uzgold);
    libxsmm_free(uggold);
    libxsmm_free(xgoldt);
    libxsmm_free(wrgold);
    libxsmm_free(wzgold);
    libxsmm_free(wggold);
    libxsmm_free(hgold);
    libxsmm_free(brgold);
    libxsmm_free(bzgold);
    libxsmm_free(bggold);
    libxsmm_free(rgold);
    libxsmm_free(zgold);
    libxsmm_free(ggold);
    libxsmm_free(r1gold);
    libxsmm_free(r2gold);
    libxsmm_free(z1gold);
    libxsmm_free(z2gold);
    libxsmm_free(g1gold);
    libxsmm_free(g2gold);
    libxsmm_free(g3gold);
    libxsmm_free(h1gold);
    libxsmm_free(h2gold);
    libxsmm_free(h3gold);
    libxsmm_free(ur);
    libxsmm_free(uz);
    libxsmm_free(ug);
    libxsmm_free(xt);
    libxsmm_free(wr);
    libxsmm_free(wz);
    libxsmm_free(wg);
    libxsmm_free(h);
    libxsmm_free(br);
    libxsmm_free(bz);
    libxsmm_free(bg);
    libxsmm_free(htest);
    libxsmm_free(hgold_temp);
  } else {
#if 0
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
    libxsmm_free(igoldt);
    libxsmm_free(fgoldt);
    libxsmm_free(ogoldt);
    libxsmm_free(cgoldt);
    libxsmm_free(dgoldt);
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
    libxsmm_free(d1gold);
    libxsmm_free(d2gold);
    libxsmm_free(d3gold);
    libxsmm_free(d4gold);
    libxsmm_free(deltagoldt);
    libxsmm_free(djdhgoldt);
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
    libxsmm_free(wgoldTp);
    libxsmm_free(rgoldTp);
    libxsmm_free(xgoldTp);
    libxsmm_free(hgoldTp);
    libxsmm_free(wi);
    libxsmm_free(wf);
    libxsmm_free(wo);
    libxsmm_free(wc);
    libxsmm_free(xt);
    libxsmm_free(ri);
    libxsmm_free(rf);
    libxsmm_free(ro);
    libxsmm_free(rc);
    libxsmm_free(ht);
    libxsmm_free(bi);
    libxsmm_free(bf);
    libxsmm_free(bo);
    libxsmm_free(bc);
    libxsmm_free(djdht);
    libxsmm_free(djdxt);
    libxsmm_free(djdwi);
    libxsmm_free(djdwf);
    libxsmm_free(djdwo);
    libxsmm_free(djdwc);
    libxsmm_free(djdri);
    libxsmm_free(djdrf);
    libxsmm_free(djdro);
    libxsmm_free(djdrc);
    libxsmm_free(djdbi);
    libxsmm_free(djdbf);
    libxsmm_free(djdbo);
    libxsmm_free(djdbc);
    libxsmm_free(djdxtestt);
    libxsmm_free(djdwtest);
    libxsmm_free(djdrtest);
    libxsmm_free(djdbtest);
    libxsmm_free(djdrgold4);
    libxsmm_free(djdbgold4);
#endif
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

