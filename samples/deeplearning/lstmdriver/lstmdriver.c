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
/* #define TWO_GEMMS */

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


LIBXSMM_INLINE void matrix_eltwise_mult_ld_a(int m, int n, int ld, float *a, float *b, float *c)
{
  int i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < m*n; i++) {
    int row = i / m;
    int col = i % m;
    c[i] = a[row*ld + col] * b[i];
  }
}


LIBXSMM_INLINE void matrix_eltwise_mult_ld_ab(int m, int n, int ld, float *a, float *b, float *c)
{
  int i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < m*n; i++) {
    int row = i / m;
    int col = i % m;
    c[i] = a[row*ld + col] * b[row*ld + col];
  }
}


LIBXSMM_INLINE void matrix_eltwise_mult_ld_c(int m, int n, int ld, float *a, float *b, float *c)
{
  int i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < m*n; i++) {
    int row = i / m;
    int col = i % m;
    c[row*ld + col] = a[i] * b[i];
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
    dst[i] = 1.0f / (1.0f + exp_value);
  }
}


LIBXSMM_INLINE void matrix_sigmoid_ld(int m, int n, int ld, float *src, float *dst)
{
  int i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < m*n; i++) {
    int row = i / m;
    int col = i % m;
    const float exp_value = (float)exp((double) -src[row*ld + col]);
    dst[row*ld + col] = 1.0f / (1.0f + exp_value);
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


LIBXSMM_INLINE void matrix_tanh_ld(int m, int n, int ld, float *src, float *dst)
{
  int i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < m*n; i++) {
    int row = i / m;
    int col = i % m;
    dst[row*ld + col] = (float)tanh((double)src[row*ld + col]);
  }
}


LIBXSMM_INLINE void matrix_relu(int size, float *src, float *dst)
{
  int i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < size; i++) {
    dst[i] = (src[i] > 0.0f) ? src[i] : 0.0f;
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
    const float sig_exp = 1.0f / (1.0f + exp_value);
    dst[i] = (1.0f - sig_exp)*sig_exp;
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
    dst[i] = 1.0f - (tanh_value * tanh_value);
  }
}


LIBXSMM_INLINE void matrix_relu_inverse(int size, float *src, float *dst)
{
  int i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < size; i++) {
    dst[i] = (src[i] > 0.0f) ? 1.0f : 0.0f;
  }
}

LIBXSMM_INLINE void matrix_transpose(int rows, int cols, float *src, float *dst)
{
  libxsmm_otrans_omp(dst, src, sizeof(float), cols, rows, cols/*ldi*/, rows/*ldo*/);
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


LIBXSMM_INLINE void matrix_copy_bias(int m, int n, int ld, float *src, float *dst)
{
  int i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < m*n; i++) {
    int row = i / m;
    int col = i % m;
    dst[row*ld + col] = src[col];
  }
}


LIBXSMM_INLINE void matrix_complement(int size, float *src, float *dst)
{
  int i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < size; i++) {
    dst[i] = 1.0f - src[i];
  }
}


LIBXSMM_INLINE void matrix_complement_ld(int m, int n, int ld, float *src, float *dst)
{
  int i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < m*n; i++) {
    int row = i / m;
    int col = i % m;
    dst[i] = 1.0f - src[row*ld + col];
  }
}


LIBXSMM_INLINE void matrix_complement_square(int size, float *src, float *dst)
{
  int i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < size; i++) {
    dst[i] = 1.0f - (src[i] * src[i]);
  }
}


LIBXSMM_INLINE void matrix_complement_square_ld(int m, int n, int ld, float *src, float *dst)
{
  int i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < m*n; i++) {
    int row = i / m;
    int col = i % m;
    dst[i] = 1.0f - (src[row*ld + col] * src[row*ld + col]);
  }
}


LIBXSMM_INLINE void convert_ck_c4k(int C, int K, float *src, float *dst)
{
  int x, y;
#if defined(_OPENMP)
# pragma omp parallel for private(x, y)
#endif
  for (y = 0; y < C; y++) {
    for (x = 0; x < K; x++) {
      dst[y*4*K + x] = src[y*K + x];
    }
  }
}


LIBXSMM_INLINE void convert_c4k_4ck(int C, int K, float *src, float *dst)
{
  /* offsets: i--0, c--1, f--2, o--3 */
  int x, y, offset;
#if defined(_OPENMP)
# pragma omp parallel for private(x, y, offset)
#endif
  for (offset = 0; offset < 4; offset++) {
    for (y = 0; y < C; y++) {
      for (x = 0; x < K; x++) {
        dst[offset*C*K + y*K + x] = src[y*4*K + offset*K + x];
      }
    }
  }
}


LIBXSMM_INLINE void convert_nk_nck(int N, int K, int CK, float *src, float *dst)
{
  int x, y;
#if defined(_OPENMP)
# pragma omp parallel for private(x, y)
#endif
  for (y = 0; y < N; y++) {
    for (x = 0; x < K; x++) {
      dst[y*CK + x] = src[y*K + x];
    }
  }
}


int main(int argc, char* argv[])
{
  float *wigold, *wfgold, *wogold, *wcgold, *xgoldt, *rigold, *rfgold, *rogold, *rcgold, *bigold, *bfgold, *bogold, *bcgold, *bfgold_fb;
  float *icfogoldt, *hgoldt;
#if defined(TWO_GEMMS)
  float *w4gold, *r4gold;
  float *djdw4gold, *djdr4gold;
  float *xgoldTp, *hgoldTp;
#else
  float *wr8gold, *xhgold;
  float *djdwr8gold;
  float *xhgoldTp;
#endif
  float *cspgold,*hpgold, *djdcspgold, *djdhpgold, *dgoldt, *doutgoldt;
  float *i1gold, *i2gold, *f1gold, *f2gold, *o1gold, *o2gold, *c1gold, *c2gold, *d1gold, *d2gold, *dhgold;
  float *xt, *csp, *hp, *w, *r, *b, *cst, *ht;
  float *it, *ft, *ot, *cit, *cot;
  float *dxt, *dcsp, *dhp, *dw, *dr, *db, *dcs, *dht;
  float *i3gold, *f3gold, *d3gold, *d4gold, *deltagoldt;
  float *djdhgoldt, *djdigoldt, *djdfgoldt, *djdcgoldt, *djdogoldt, *djdxgoldt;
  float *djdb4gold, *dicfogoldt, *djdcsgold;
  float *wigoldTp, *wfgoldTp, *wogoldTp, *wcgoldTp, *rigoldTp, *rfgoldTp, *rogoldTp, *rcgoldTp;
  float forget_bias = 1.0f;

  const char transa = 'N', transb = 'N'; /* no transposes */
  const float alpha = 1, beta = 1, beta0 = 0;
  void *scratch, *internalstate;
  size_t scratch_size = 0, internalstate_size = 0;

  int iters = 10;   /* repetitions of benchmark */
  int pass = 3;     /* pass: 0--FWD, 1--BWD, 2--UPD, 3--BWD+UPD */
  int N = 128;      /* size of mini-batch */
  int C = 512;      /* number of inputs */
  int K = 64;       /* number of outputs */
  int t = 5;        /* number of time steps (>= 1) */
  int K4 = K * 4;
  int CK = C + K;

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
  int j, l, p;

  libxsmm_dnn_rnncell_desc lstmcell_desc;
  libxsmm_dnn_rnncell* libxsmm_handle;
  libxsmm_dnn_tensor* libxsmm_input;
  libxsmm_dnn_tensor* libxsmm_cs_prev;
  libxsmm_dnn_tensor* libxsmm_hidden_state_prev;
  libxsmm_dnn_tensor* libxsmm_weight;
  libxsmm_dnn_tensor* libxsmm_recur_weight;
  libxsmm_dnn_tensor* libxsmm_bias;
  libxsmm_dnn_tensor* libxsmm_cs;
  libxsmm_dnn_tensor* libxsmm_hidden_state;
  libxsmm_dnn_tensor* libxsmm_i;
  libxsmm_dnn_tensor* libxsmm_f;
  libxsmm_dnn_tensor* libxsmm_o;
  libxsmm_dnn_tensor* libxsmm_ci;
  libxsmm_dnn_tensor* libxsmm_co;
  libxsmm_dnn_tensor* libxsmm_dinput;
  libxsmm_dnn_tensor* libxsmm_dcs_prev;
  libxsmm_dnn_tensor* libxsmm_dhidden_state_prev;
  libxsmm_dnn_tensor* libxsmm_dweight;
  libxsmm_dnn_tensor* libxsmm_drecur_weight;
  libxsmm_dnn_tensor* libxsmm_dbias;
  libxsmm_dnn_tensor* libxsmm_dcs;
  libxsmm_dnn_tensor* libxsmm_dhidden_state;

  libxsmm_dnn_tensor_datalayout* libxsmm_layout;
  libxsmm_dnn_err_t status;
  libxsmm_dnn_err_t global_status = LIBXSMM_DNN_SUCCESS;

  libxsmm_matdiff_info norms_fwd, norms_bwd, norms_upd_w, norms_upd_r, norms_upd_b, diff;
  libxsmm_matdiff_clear(&norms_fwd);
  libxsmm_matdiff_clear(&norms_bwd);
  libxsmm_matdiff_clear(&norms_upd_w);
  libxsmm_matdiff_clear(&norms_upd_r);
  libxsmm_matdiff_clear(&norms_upd_b);
  libxsmm_matdiff_clear(&diff);

  if (argc > 1 && !strncmp(argv[1], "-h", 3)) {
    printf("\nUsage: ./lstmdriver [reps] [pass: 0--FWD, 1--BWD, 2--UPD, 3--BWD+UPD] [N] [C] [K] [time_steps > 0]\n\n");
    return 0;
  }
  libxsmm_srand(1);

  /* reading new values from cli */
  j = 1;
  if (argc > j) iters = atoi(argv[j++]);
  if (argc > j) pass  = atoi(argv[j++]);
  if (argc > j) N     = atoi(argv[j++]);
  if (argc > j) C     = atoi(argv[j++]);
  if (argc > j) K     = atoi(argv[j++]);
  if (argc > j) t     = atoi(argv[j++]);
  K4 = K * 4;
  CK = C + K;

  if (t <= 0) {
    printf("time_steps %d should be greater than 1\n\n", t);
    return 0;
  }
  if (!(pass == 0 || pass == 1 || pass == 2 || pass == 3)) {
    printf("Unknown pass: %d, valid arguments for pass = {0(FWD), 1(BWD), 2(UPD), 3(BWD+UPD)\n\n", pass);
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
  printf("PARAMS: N:%d  C:%d  K:%d  T:%d\n", N, C, K, t);
  printf("PARAMS: ITERS:%d", iters); if (LIBXSMM_FEQ(0, check)) printf("  Threads:%d\n", nThreads); else printf("\n");
  printf("SIZE Weight (MB): %10.2f MiB\n", (double)(C*K*sizeof(float))/(1024.0*1024.0) );
  printf("SIZE Input (MB): %10.2f MiB\n", (double)(N*C*sizeof(float))/(1024.0*1024.0) );
  printf("SIZE Hidden State: %10.2f MiB\n", (double)(K*N*sizeof(float))/(1024.0*1024.0) );

  /* allocate data */
  xgoldt = (float*)libxsmm_aligned_malloc(N*C*t*sizeof(float), 2097152);
  cspgold= (float*)libxsmm_aligned_malloc(K*N*sizeof(float), 2097152);
  hpgold = (float*)libxsmm_aligned_malloc(K*N*sizeof(float), 2097152);
  wigold = (float*)libxsmm_aligned_malloc(C*K*sizeof(float), 2097152);
  wfgold = (float*)libxsmm_aligned_malloc(C*K*sizeof(float), 2097152);
  wogold = (float*)libxsmm_aligned_malloc(C*K*sizeof(float), 2097152);
  wcgold = (float*)libxsmm_aligned_malloc(C*K*sizeof(float), 2097152);
  rigold = (float*)libxsmm_aligned_malloc(K*K*sizeof(float), 2097152);
  rfgold = (float*)libxsmm_aligned_malloc(K*K*sizeof(float), 2097152);
  rogold = (float*)libxsmm_aligned_malloc(K*K*sizeof(float), 2097152);
  rcgold = (float*)libxsmm_aligned_malloc(K*K*sizeof(float), 2097152);
#if defined(TWO_GEMMS)
  w4gold = (float*)libxsmm_aligned_malloc(C*K*4*sizeof(float), 2097152);
  r4gold = (float*)libxsmm_aligned_malloc(K*K*4*sizeof(float), 2097152);
#else
  wr8gold= (float*)libxsmm_aligned_malloc((C+K)*K*4*sizeof(float), 2097152);
  xhgold = (float*)libxsmm_aligned_malloc((C+K)*N*sizeof(float), 2097152);
#endif
  bigold = (float*)libxsmm_aligned_malloc(K*sizeof(float), 2097152);
  bfgold = (float*)libxsmm_aligned_malloc(K*sizeof(float), 2097152);
  bogold = (float*)libxsmm_aligned_malloc(K*sizeof(float), 2097152);
  bcgold = (float*)libxsmm_aligned_malloc(K*sizeof(float), 2097152);
  hgoldt = (float*)libxsmm_aligned_malloc(K*N*t*sizeof(float), 2097152);
  bfgold_fb = (float*)libxsmm_aligned_malloc(K*sizeof(float), 2097152);
  dgoldt = (float*)libxsmm_aligned_malloc(K*N*t*sizeof(float), 2097152);
  icfogoldt = (float*)libxsmm_aligned_malloc(4*K*N*t*sizeof(float), 2097152);
  i1gold = (float*)libxsmm_aligned_malloc(K*N*sizeof(float), 2097152);
  i2gold = (float*)libxsmm_aligned_malloc(K*N*sizeof(float), 2097152);
  i3gold = (float*)libxsmm_aligned_malloc(K*N*sizeof(float), 2097152);
  f1gold = (float*)libxsmm_aligned_malloc(K*N*sizeof(float), 2097152);
  f2gold = (float*)libxsmm_aligned_malloc(K*N*sizeof(float), 2097152);
  f3gold = (float*)libxsmm_aligned_malloc(K*N*sizeof(float), 2097152);
  o1gold = (float*)libxsmm_aligned_malloc(K*N*sizeof(float), 2097152);
  o2gold = (float*)libxsmm_aligned_malloc(K*N*sizeof(float), 2097152);
  c1gold = (float*)libxsmm_aligned_malloc(K*N*sizeof(float), 2097152);
  c2gold = (float*)libxsmm_aligned_malloc(K*N*sizeof(float), 2097152);
  d1gold = (float*)libxsmm_aligned_malloc(K*N*sizeof(float), 2097152);
  d2gold = (float*)libxsmm_aligned_malloc(K*N*sizeof(float), 2097152);
  d3gold = (float*)libxsmm_aligned_malloc(K*N*sizeof(float), 2097152);
  d4gold = (float*)libxsmm_aligned_malloc(K*N*sizeof(float), 2097152);
  dhgold = (float*)libxsmm_aligned_malloc(K*N*sizeof(float), 2097152);
  djdhgoldt  = (float*)libxsmm_aligned_malloc(K*N*t*sizeof(float), 2097152);
  deltagoldt = (float*)libxsmm_aligned_malloc(K*N*t*sizeof(float), 2097152);
  djdcspgold = (float*)libxsmm_aligned_malloc(K*N*sizeof(float), 2097152);
  djdigoldt  = (float*)libxsmm_aligned_malloc(K*N*t*sizeof(float), 2097152);
  djdfgoldt  = (float*)libxsmm_aligned_malloc(K*N*t*sizeof(float), 2097152);
  djdcgoldt  = (float*)libxsmm_aligned_malloc(K*N*t*sizeof(float), 2097152);
  djdogoldt  = (float*)libxsmm_aligned_malloc(K*N*t*sizeof(float), 2097152);
  dicfogoldt = (float*)libxsmm_aligned_malloc(4*K*N*t*sizeof(float), 2097152);
  djdxgoldt  = (float*)libxsmm_aligned_malloc(N*C*t*sizeof(float), 2097152);
#if defined(TWO_GEMMS)
  djdw4gold  = (float*)libxsmm_aligned_malloc(C*K*4*sizeof(float), 2097152);
  djdr4gold  = (float*)libxsmm_aligned_malloc(K*K*4*sizeof(float), 2097152);
  xgoldTp    = (float*)libxsmm_aligned_malloc(N*C*sizeof(float), 2097152);
  hgoldTp    = (float*)libxsmm_aligned_malloc(K*N*sizeof(float), 2097152);
#else
  djdwr8gold = (float*)libxsmm_aligned_malloc((C+K)*K*4*sizeof(float), 2097152);
  xhgoldTp   = (float*)libxsmm_aligned_malloc(N*(C+K)*sizeof(float), 2097152);
#endif
  djdb4gold  = (float*)libxsmm_aligned_malloc(K*4*sizeof(float), 2097152);
  djdcsgold  = (float*)libxsmm_aligned_malloc(K*N*sizeof(float), 2097152);
  djdhpgold  = (float*)libxsmm_aligned_malloc(K*N*sizeof(float), 2097152);
  wigoldTp   = (float*)libxsmm_aligned_malloc(C*K*sizeof(float), 2097152);
  wfgoldTp   = (float*)libxsmm_aligned_malloc(C*K*sizeof(float), 2097152);
  wogoldTp   = (float*)libxsmm_aligned_malloc(C*K*sizeof(float), 2097152);
  wcgoldTp   = (float*)libxsmm_aligned_malloc(C*K*sizeof(float), 2097152);
  rigoldTp   = (float*)libxsmm_aligned_malloc(K*K*sizeof(float), 2097152);
  rfgoldTp   = (float*)libxsmm_aligned_malloc(K*K*sizeof(float), 2097152);
  rogoldTp   = (float*)libxsmm_aligned_malloc(K*K*sizeof(float), 2097152);
  rcgoldTp   = (float*)libxsmm_aligned_malloc(K*K*sizeof(float), 2097152);
  doutgoldt  = (float*)libxsmm_aligned_malloc(K*N*t*sizeof(float), 2097152);
  xt    = (float*)libxsmm_aligned_malloc(N*C*t*sizeof(float), 2097152);
  csp   = (float*)libxsmm_aligned_malloc(K*N*sizeof(float), 2097152);
  hp    = (float*)libxsmm_aligned_malloc(K*N*sizeof(float), 2097152);
  w     = (float*)libxsmm_aligned_malloc(C*K*4*sizeof(float), 2097152);
  r     = (float*)libxsmm_aligned_malloc(K*K*4*sizeof(float), 2097152);
  b     = (float*)libxsmm_aligned_malloc(K*4*sizeof(float), 2097152);
  cst   = (float*)libxsmm_aligned_malloc(K*N*t*sizeof(float), 2097152);
  ht    = (float*)libxsmm_aligned_malloc(K*N*t*sizeof(float), 2097152);
  it    = (float*)libxsmm_aligned_malloc(K*N*t*sizeof(float), 2097152);
  ft    = (float*)libxsmm_aligned_malloc(K*N*t*sizeof(float), 2097152);
  ot    = (float*)libxsmm_aligned_malloc(K*N*t*sizeof(float), 2097152);
  cit   = (float*)libxsmm_aligned_malloc(K*N*t*sizeof(float), 2097152);
  cot   = (float*)libxsmm_aligned_malloc(K*N*t*sizeof(float), 2097152);
  dxt   = (float*)libxsmm_aligned_malloc(N*C*t*sizeof(float), 2097152);
  dcsp  = (float*)libxsmm_aligned_malloc(K*N*sizeof(float), 2097152);
  dhp   = (float*)libxsmm_aligned_malloc(K*N*sizeof(float), 2097152);
  dw    = (float*)libxsmm_aligned_malloc(C*K*4*sizeof(float), 2097152);
  dr    = (float*)libxsmm_aligned_malloc(K*K*4*sizeof(float), 2097152);
  db    = (float*)libxsmm_aligned_malloc(K*4*sizeof(float), 2097152);
  dcs   = (float*)libxsmm_aligned_malloc(K*N*sizeof(float), 2097152);
  dht   = (float*)libxsmm_aligned_malloc(K*N*t*sizeof(float), 2097152);
  LIBXSMM_VLA_DECL(2, float, xgold, xgoldt, N * C);
  LIBXSMM_VLA_DECL(2, float, dgold, dgoldt, K * N);
  LIBXSMM_VLA_DECL(2, float, hgold, hgoldt, K * N);
  LIBXSMM_VLA_DECL(3, float, icfogold, icfogoldt, N, 4 * K);
  LIBXSMM_VLA_DECL(2, float, djdhgold, djdhgoldt, K * N);
  LIBXSMM_VLA_DECL(2, float, deltagold, deltagoldt, K * N);
  LIBXSMM_VLA_DECL(2, float, doutgold, doutgoldt, K * N);
  LIBXSMM_VLA_DECL(3, float, dicfogold, dicfogoldt, N, 4 * K);
  LIBXSMM_VLA_DECL(2, float, djdxgold, djdxgoldt, N * C);
  LIBXSMM_VLA_DECL(2, float, h, ht, K * N);

  /* initialize data */
  /* FWD */
  LIBXSMM_MATINIT_OMP(float, 24, cspgold,N, K, N, 1.0);
  LIBXSMM_MATINIT_OMP(float, 24, hpgold, N, K, N, 1.0);
  LIBXSMM_MATINIT_OMP(float, 42, wigold, C, K, C, 1.0);
  LIBXSMM_MATINIT_OMP(float, 42, wfgold, C, K, C, 1.0);
  LIBXSMM_MATINIT_OMP(float, 42, wogold, C, K, C, 1.0);
  LIBXSMM_MATINIT_OMP(float, 42, wcgold, C, K, C, 1.0);
  for (j = 0; j < t; ++j) {
    LIBXSMM_MATINIT_OMP(float, 24, &LIBXSMM_VLA_ACCESS(2, xgold, j, 0, N * C), N, C, N, 1.0);
  }
  LIBXSMM_MATINIT_OMP(float, 42, rigold, K, K, K, 1.0);
  LIBXSMM_MATINIT_OMP(float, 42, rfgold, K, K, K, 1.0);
  LIBXSMM_MATINIT_OMP(float, 42, rogold, K, K, K, 1.0);
  LIBXSMM_MATINIT_OMP(float, 42, rcgold, K, K, K, 1.0);
  LIBXSMM_MATINIT_OMP(float, 24, bigold, 1, K, 1, 1.0);
  LIBXSMM_MATINIT_OMP(float, 24, bfgold, 1, K, 1, 1.0);
  LIBXSMM_MATINIT_OMP(float, 24, bogold, 1, K, 1, 1.0);
  LIBXSMM_MATINIT_OMP(float, 24, bcgold, 1, K, 1, 1.0);
  for (j = 0; j < K; j++) {
    bfgold_fb[j] = bfgold[j] + forget_bias;
  }
  zero_buf(dgoldt, K*N*t);
  zero_buf(icfogoldt, 4*K*N*t);
  zero_buf(i1gold, K*N);
  zero_buf(i2gold, K*N);
  zero_buf(f1gold, K*N);
  zero_buf(f2gold, K*N);
  zero_buf(o1gold, K*N);
  zero_buf(o2gold, K*N);
  zero_buf(c1gold, K*N);
  zero_buf(c2gold, K*N);
  zero_buf(d1gold, K*N);
  zero_buf(d2gold, K*N);
  zero_buf(dhgold, K*N);
#if defined(TWO_GEMMS)
  zero_buf(w4gold, C*K*4);
  zero_buf(r4gold, K*K*4);
#else
  zero_buf(wr8gold, (C+K)*K*4);
  zero_buf(xhgold,  (C+K)*N);
#endif
  /* BWD/UPD */
  for (j = 0; j < t; ++j) {
    LIBXSMM_MATINIT_OMP(float, 24, &LIBXSMM_VLA_ACCESS(2, djdhgold, j, 0, K * N), N, K, N, 1.0);
  }
  LIBXSMM_MATINIT_OMP(float, 24, djdcsgold, N, K, N, 1.0);
  zero_buf(i3gold, K*N);
  zero_buf(f3gold, K*N);
  zero_buf(d3gold, K*N);
  zero_buf(d4gold, K*N);
  zero_buf(deltagoldt, K*N*t);
  zero_buf(djdcspgold, K*N);
  zero_buf(djdigoldt, K*N*t);
  zero_buf(djdfgoldt, K*N*t);
  zero_buf(djdogoldt, K*N*t);
  zero_buf(djdcgoldt, K*N*t);
  zero_buf(djdxgoldt, N*C*t);
  zero_buf(djdhpgold, K*N);
  zero_buf(wigoldTp, C*K);
  zero_buf(wfgoldTp, C*K);
  zero_buf(wogoldTp, C*K);
  zero_buf(wcgoldTp, C*K);
  zero_buf(rigoldTp, K*K);
  zero_buf(rfgoldTp, K*K);
  zero_buf(rogoldTp, K*K);
  zero_buf(rcgoldTp, K*K);
  zero_buf(doutgoldt, K*N*t);
  zero_buf(dicfogoldt, 4*K*N*t);
#if defined(TWO_GEMMS)
  zero_buf(djdw4gold, C*K*4);
  zero_buf(djdr4gold, K*K*4);
  zero_buf(xgoldTp, N*C);
  zero_buf(hgoldTp, K*N);
#else
  zero_buf(djdwr8gold, (C+K)*K*4);
  zero_buf(xhgoldTp, N*(C+K));
#endif
  zero_buf(djdb4gold, K*4);

  /* first touch LIBXSMM */
  zero_buf(xt,  N*C*t);
  zero_buf(csp, K*N);
  zero_buf(hp,  K*N);
  zero_buf(w,   C*K*4);
  zero_buf(r,   K*K*4);
  zero_buf(b,   K*4);
  zero_buf(cst, K*N*t);
  zero_buf(ht,  K*N*t);
  zero_buf(it,  K*N*t);
  zero_buf(ft,  K*N*t);
  zero_buf(ot,  K*N*t);
  zero_buf(cit, K*N*t);
  zero_buf(cot, K*N*t);
  zero_buf(dxt,  N*C*t);
  zero_buf(dcsp, K*N);
  zero_buf(dhp,  K*N);
  zero_buf(dw,   C*K*4);
  zero_buf(dr,   K*K*4);
  zero_buf(db,   K*4);
  zero_buf(dcs,  K*N);
  zero_buf(dht,  K*N*t);

  if (LIBXSMM_NEQ(0, check)) {
    printf("##########################################\n");
    printf("#         Computing Reference ...        #\n");
    printf("##########################################\n");
    /* FWD */
#if defined(TWO_GEMMS)
    convert_ck_c4k(C, K, wigold, w4gold);
    convert_ck_c4k(C, K, wcgold, &(w4gold[K]));
    convert_ck_c4k(C, K, wfgold, &(w4gold[2*K]));
    convert_ck_c4k(C, K, wogold, &(w4gold[3*K]));
    convert_ck_c4k(K, K, rigold, r4gold);
    convert_ck_c4k(K, K, rcgold, &(r4gold[K]));
    convert_ck_c4k(K, K, rfgold, &(r4gold[2*K]));
    convert_ck_c4k(K, K, rogold, &(r4gold[3*K]));
#else
    convert_ck_c4k(C, K, wigold, wr8gold);
    convert_ck_c4k(C, K, wcgold, &(wr8gold[K]));
    convert_ck_c4k(C, K, wfgold, &(wr8gold[2*K]));
    convert_ck_c4k(C, K, wogold, &(wr8gold[3*K]));
    convert_ck_c4k(K, K, rigold, &(wr8gold[C*K*4]));
    convert_ck_c4k(K, K, rcgold, &(wr8gold[C*K*4 + K]));
    convert_ck_c4k(K, K, rfgold, &(wr8gold[C*K*4 + 2*K]));
    convert_ck_c4k(K, K, rogold, &(wr8gold[C*K*4 + 3*K]));
#endif
    for (j = 0; j < t; ++j) {
      /* Initialization with bias */
      matrix_copy_bias(K, N, 4*K, bigold,    &LIBXSMM_VLA_ACCESS(3, icfogold, j, 0, 0,   N, 4 * K));
      matrix_copy_bias(K, N, 4*K, bcgold,    &LIBXSMM_VLA_ACCESS(3, icfogold, j, 0, K,   N, 4 * K));
      matrix_copy_bias(K, N, 4*K, bfgold_fb, &LIBXSMM_VLA_ACCESS(3, icfogold, j, 0, 2*K, N, 4 * K));
      matrix_copy_bias(K, N, 4*K, bogold,    &LIBXSMM_VLA_ACCESS(3, icfogold, j, 0, 3*K, N, 4 * K));
#if defined(TWO_GEMMS)
      /* icfo += W * x */
      LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &K4, &N, &C, &alpha, w4gold, &K4, &LIBXSMM_VLA_ACCESS(2, xgold, j, 0, N * C), &C, &beta, &LIBXSMM_VLA_ACCESS(3, icfogold, j, 0, 0, N, 4 * K), &K4);
      /* icfo += R * h */
      if (j == 0) {
        LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &K4, &N, &K, &alpha, r4gold, &K4, hpgold, &K, &beta, &LIBXSMM_VLA_ACCESS(3, icfogold, 0, 0, 0, N, 4 * K), &K4);
      } else {
        LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &K4, &N, &K, &alpha, r4gold, &K4, &LIBXSMM_VLA_ACCESS(2, hgold, j-1, 0, K * N), &K, &beta, &LIBXSMM_VLA_ACCESS(3, icfogold, j, 0, 0, N, 4 * K), &K4);
      }
#else
      /* Concatenate x and h */
      convert_nk_nck(N, C, C+K, &LIBXSMM_VLA_ACCESS(2, xgold, j, 0, N * C),  xhgold);
      if (j == 0) {
        convert_nk_nck(N, K, C+K, hpgold, &(xhgold[C]));
      } else {
        convert_nk_nck(N, K, C+K, &LIBXSMM_VLA_ACCESS(2, hgold, j-1, 0, K * N), &(xhgold[C]));
      }
      /* icfo += (W * x) + (R * h) */
      LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &K4, &N, &CK, &alpha, wr8gold, &K4, xhgold, &CK, &beta, &LIBXSMM_VLA_ACCESS(3, icfogold, j, 0, 0, N, 4 * K), &K4);
#endif
      /* icfo = non-lin(icfo) */
      matrix_sigmoid_ld(K, N, 4*K, &LIBXSMM_VLA_ACCESS(3, icfogold, j, 0, 0,   N, 4 * K), &LIBXSMM_VLA_ACCESS(3, icfogold, j, 0, 0,   N, 4 * K));
      matrix_tanh_ld   (K, N, 4*K, &LIBXSMM_VLA_ACCESS(3, icfogold, j, 0, K,   N, 4 * K), &LIBXSMM_VLA_ACCESS(3, icfogold, j, 0, K,   N, 4 * K));
      matrix_sigmoid_ld(K, N, 4*K, &LIBXSMM_VLA_ACCESS(3, icfogold, j, 0, 2*K, N, 4 * K), &LIBXSMM_VLA_ACCESS(3, icfogold, j, 0, 2*K, N, 4 * K));
      matrix_sigmoid_ld(K, N, 4*K, &LIBXSMM_VLA_ACCESS(3, icfogold, j, 0, 3*K, N, 4 * K), &LIBXSMM_VLA_ACCESS(3, icfogold, j, 0, 3*K, N, 4 * K));
      /* d1 = f.d */
      if (j == 0) {
        matrix_eltwise_mult_ld_a(K, N, 4*K, &LIBXSMM_VLA_ACCESS(3, icfogold, 0, 0, 2*K, N, 4 * K), cspgold, d1gold);
      } else {
        matrix_eltwise_mult_ld_a(K, N, 4*K, &LIBXSMM_VLA_ACCESS(3, icfogold, j, 0, 2*K, N, 4 * K), &LIBXSMM_VLA_ACCESS(2, dgold, j-1, 0, K * N), d1gold);
      }
      /* d2 = i.c */
      matrix_eltwise_mult_ld_ab(K, N, 4*K, &LIBXSMM_VLA_ACCESS(3, icfogold, j, 0, 0,   N, 4 * K), &LIBXSMM_VLA_ACCESS(3, icfogold, j, 0, K, N, 4 * K), d2gold);
      /* d = d1 + d2 */
      matrix_add(K*N, d1gold, d2gold, &LIBXSMM_VLA_ACCESS(2, dgold, j, 0, K * N));
      /* dh = tanh(d) */
      matrix_tanh(K*N, &LIBXSMM_VLA_ACCESS(2, dgold, j, 0, K * N), dhgold);
      /* h = o.dh */
      matrix_eltwise_mult_ld_a (K, N, 4*K, &LIBXSMM_VLA_ACCESS(3, icfogold, j, 0, 3*K, N, 4 * K), dhgold, &LIBXSMM_VLA_ACCESS(2, hgold, j, 0, K * N));
    }
    /* BWD/UPD */
    matrix_transpose(C, K, wigold, wigoldTp);
    matrix_transpose(C, K, wfgold, wfgoldTp);
    matrix_transpose(C, K, wogold, wogoldTp);
    matrix_transpose(C, K, wcgold, wcgoldTp);
    matrix_transpose(K, K, rigold, rigoldTp);
    matrix_transpose(K, K, rfgold, rfgoldTp);
    matrix_transpose(K, K, rogold, rogoldTp);
    matrix_transpose(K, K, rcgold, rcgoldTp);
    for (j = t-1; j >= 0; --j) {
      /* compute deltagold */
      if (j == t-1) {
        matrix_copy(K * N, &LIBXSMM_VLA_ACCESS(2, djdhgold, t-1, 0, K * N), &LIBXSMM_VLA_ACCESS(2, deltagold, t-1, 0, K * N));
      } else {
        matrix_add(K * N, &LIBXSMM_VLA_ACCESS(2, doutgold, j, 0, K * N), &LIBXSMM_VLA_ACCESS(2, djdhgold, j, 0, K * N), &LIBXSMM_VLA_ACCESS(2, deltagold, j, 0, K * N));
      }
      /* compute djdcspgold */
      matrix_eltwise_mult_ld_a(K, N, 4*K, &LIBXSMM_VLA_ACCESS(3, icfogold, j, 0, 3*K, N, 4 * K), &LIBXSMM_VLA_ACCESS(2, deltagold, j, 0, K * N), d1gold);
      matrix_tanh_inverse(K * N, &LIBXSMM_VLA_ACCESS(2, dgold, j, 0, K * N), d2gold);
      matrix_eltwise_mult(K * N, d1gold, d2gold, d3gold);
      if (j == t-1) {
        matrix_add(K * N, d3gold, djdcsgold, djdcspgold);
      } else {
        matrix_add(K * N, d3gold, djdcspgold, djdcspgold);
      }
      /* compute djdcgold */
      matrix_eltwise_mult_ld_a   (K, N, 4*K, &LIBXSMM_VLA_ACCESS(3, icfogold, j, 0, 0, N, 4 * K), djdcspgold, c1gold);
      matrix_complement_square_ld(K, N, 4*K, &LIBXSMM_VLA_ACCESS(3, icfogold, j, 0, K, N, 4 * K), c2gold);
      matrix_eltwise_mult_ld_c   (K, N, 4*K, c1gold, c2gold, &LIBXSMM_VLA_ACCESS(3, dicfogold, j, 0, K, N, 4 * K));
      /* compute djdigold */
      matrix_eltwise_mult_ld_a   (K, N, 4*K, &LIBXSMM_VLA_ACCESS(3, icfogold, j, 0, K, N, 4 * K), djdcspgold, i1gold);
      matrix_complement_ld       (K, N, 4*K, &LIBXSMM_VLA_ACCESS(3, icfogold, j, 0, 0, N, 4 * K), i2gold);
      matrix_eltwise_mult_ld_a   (K, N, 4*K, &LIBXSMM_VLA_ACCESS(3, icfogold, j, 0, 0, N, 4 * K), i2gold, i3gold);
      matrix_eltwise_mult_ld_c   (K, N, 4*K, i1gold, i3gold, &LIBXSMM_VLA_ACCESS(3, dicfogold, j, 0, 0, N, 4 * K));
      /* compute djdfgold */
      if (j == 0) {
        matrix_eltwise_mult(K * N, djdcspgold, cspgold, f1gold);
      } else {
        matrix_eltwise_mult(K * N, djdcspgold, &LIBXSMM_VLA_ACCESS(2, dgold, j-1, 0, K * N), f1gold);
      }
      matrix_complement_ld       (K, N, 4*K, &LIBXSMM_VLA_ACCESS(3, icfogold, j, 0, 2*K, N, 4 * K), f2gold);
      matrix_eltwise_mult_ld_a   (K, N, 4*K, &LIBXSMM_VLA_ACCESS(3, icfogold, j, 0, 2*K, N, 4 * K), f2gold, f3gold);
      matrix_eltwise_mult_ld_c   (K, N, 4*K, f1gold, f3gold, &LIBXSMM_VLA_ACCESS(3, dicfogold, j, 0, 2*K, N, 4 * K));
      /* compute djdogold */
      matrix_tanh(K * N, &LIBXSMM_VLA_ACCESS(2, dgold, j, 0, K * N), o1gold);
      matrix_eltwise_mult(K * N, &LIBXSMM_VLA_ACCESS(2, deltagold, j, 0, K * N), o1gold, o1gold);
      matrix_complement_ld       (K, N, 4*K, &LIBXSMM_VLA_ACCESS(3, icfogold, j, 0, 3*K, N, 4 * K), o2gold);
      matrix_eltwise_mult_ld_a   (K, N, 4*K, &LIBXSMM_VLA_ACCESS(3, icfogold, j, 0, 3*K, N, 4 * K), o2gold, o2gold);
      matrix_eltwise_mult_ld_c   (K, N, 4*K, o1gold, o2gold, &LIBXSMM_VLA_ACCESS(3, dicfogold, j, 0, 3*K, N, 4 * K));
      /* update djdcspgold */
      matrix_eltwise_mult_ld_a   (K, N, 4*K, &LIBXSMM_VLA_ACCESS(3, icfogold, j, 0, 2*K, N, 4 * K), djdcspgold, djdcspgold);
      if (j > 0) {
        /* compute doutgold */
        LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &K, &N, &K, &alpha, rigoldTp, &K, &LIBXSMM_VLA_ACCESS(3, dicfogold, j, 0, 0,   N, 4 * K), &K4, &beta, &LIBXSMM_VLA_ACCESS(2, doutgold, j-1, 0, K * N), &K);
        LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &K, &N, &K, &alpha, rcgoldTp, &K, &LIBXSMM_VLA_ACCESS(3, dicfogold, j, 0, K,   N, 4 * K), &K4, &beta, &LIBXSMM_VLA_ACCESS(2, doutgold, j-1, 0, K * N), &K);
        LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &K, &N, &K, &alpha, rfgoldTp, &K, &LIBXSMM_VLA_ACCESS(3, dicfogold, j, 0, 2*K, N, 4 * K), &K4, &beta, &LIBXSMM_VLA_ACCESS(2, doutgold, j-1, 0, K * N), &K);
        LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &K, &N, &K, &alpha, rogoldTp, &K, &LIBXSMM_VLA_ACCESS(3, dicfogold, j, 0, 3*K, N, 4 * K), &K4, &beta, &LIBXSMM_VLA_ACCESS(2, doutgold, j-1, 0, K * N), &K);
      } else {
        /* compute djdhpgold */
        LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &K, &N, &K, &alpha, rigoldTp, &K, &LIBXSMM_VLA_ACCESS(3, dicfogold, 0, 0,   0, N, 4 * K), &K4, &beta, djdhpgold, &K);
        LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &K, &N, &K, &alpha, rcgoldTp, &K, &LIBXSMM_VLA_ACCESS(3, dicfogold, 0, 0,   K, N, 4 * K), &K4, &beta, djdhpgold, &K);
        LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &K, &N, &K, &alpha, rfgoldTp, &K, &LIBXSMM_VLA_ACCESS(3, dicfogold, 0, 0, 2*K, N, 4 * K), &K4, &beta, djdhpgold, &K);
        LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &K, &N, &K, &alpha, rogoldTp, &K, &LIBXSMM_VLA_ACCESS(3, dicfogold, 0, 0, 4*K, N, 4 * K), &K4, &beta, djdhpgold, &K);
      }
      if (pass == 1 || pass == 3) {
        /* compute djdxgold */
        LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &C, &N, &K, &alpha, wigoldTp, &C, &LIBXSMM_VLA_ACCESS(3, dicfogold, j, 0,   0, N, 4 * K), &K4, &beta, &LIBXSMM_VLA_ACCESS(2, djdxgold, j, 0, N * C), &C);
        LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &C, &N, &K, &alpha, wcgoldTp, &C, &LIBXSMM_VLA_ACCESS(3, dicfogold, j, 0,   K, N, 4 * K), &K4, &beta, &LIBXSMM_VLA_ACCESS(2, djdxgold, j, 0, N * C), &C);
        LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &C, &N, &K, &alpha, wfgoldTp, &C, &LIBXSMM_VLA_ACCESS(3, dicfogold, j, 0, 2*K, N, 4 * K), &K4, &beta, &LIBXSMM_VLA_ACCESS(2, djdxgold, j, 0, N * C), &C);
        LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &C, &N, &K, &alpha, wogoldTp, &C, &LIBXSMM_VLA_ACCESS(3, dicfogold, j, 0, 3*K, N, 4 * K), &K4, &beta, &LIBXSMM_VLA_ACCESS(2, djdxgold, j, 0, N * C), &C);
      }
      if (pass == 2 || pass == 3) {
#if defined(TWO_GEMMS)
        /* compute djdwgold */
        matrix_transpose(N, C, &LIBXSMM_VLA_ACCESS(2, xgold, j, 0, N * C), xgoldTp);
        LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &K4, &C, &N, &alpha, &LIBXSMM_VLA_ACCESS(3, dicfogold, j, 0, 0, N, 4 * K), &K4, xgoldTp, &N, &beta, djdw4gold, &K4);

        /* compute djdrgold */
        if (j == 0) {
          matrix_transpose(N, K, hpgold, hgoldTp);
        } else {
          matrix_transpose(N, K, &LIBXSMM_VLA_ACCESS(2, hgold, j-1, 0, K * N), hgoldTp);
        }
        LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &K4, &K, &N, &alpha, &LIBXSMM_VLA_ACCESS(3, dicfogold, j, 0, 0, N, 4 * K), &K4, hgoldTp, &N, &beta, djdr4gold, &K4);
#else
        matrix_transpose(N, C, &LIBXSMM_VLA_ACCESS(2, xgold, j, 0, N * C), xhgoldTp);
        if (j == 0) {
          matrix_transpose(N, K, hpgold, &(xhgoldTp[N*C]));
        } else {
          matrix_transpose(N, K, &LIBXSMM_VLA_ACCESS(2, hgold, j-1, 0, K * N), &(xhgoldTp[N*C]));
        }
        LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &K4, &CK, &N, &alpha, &LIBXSMM_VLA_ACCESS(3, dicfogold, j, 0, 0, N, 4 * K), &K4, xhgoldTp, &N, &beta, djdwr8gold, &K4);
#endif

        /* compute djdbgold */
#if defined(_OPENMP)
# pragma omp parallel for private(l, p)
#endif
        for (l = 0; l < K; l++) {
          for (p = 0; p < N; p++) {
            djdb4gold[l]       += LIBXSMM_VLA_ACCESS(3, dicfogold, j, p, l,       N, 4 * K);
            djdb4gold[l + K]   += LIBXSMM_VLA_ACCESS(3, dicfogold, j, p, l + K,   N, 4 * K);
            djdb4gold[l + 2*K] += LIBXSMM_VLA_ACCESS(3, dicfogold, j, p, l + 2*K, N, 4 * K);
            djdb4gold[l + 3*K] += LIBXSMM_VLA_ACCESS(3, dicfogold, j, p, l + 3*K, N, 4 * K);
          }
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
    lstmcell_desc.threads = nThreads;
    lstmcell_desc.N = N;
    lstmcell_desc.C = C;
    lstmcell_desc.K = K;
    lstmcell_desc.t = t;
    lstmcell_desc.cell_type = LIBXSMM_DNN_RNNCELL_LSTM;
    lstmcell_desc.datatype_in = LIBXSMM_DNN_DATATYPE_F32;
    lstmcell_desc.datatype_out = LIBXSMM_DNN_DATATYPE_F32;
    lstmcell_desc.buffer_format = LIBXSMM_DNN_TENSOR_FORMAT_NC;
    lstmcell_desc.filter_format = LIBXSMM_DNN_TENSOR_FORMAT_CK;

    libxsmm_handle = libxsmm_dnn_create_rnncell( lstmcell_desc, &status );
    CHKERR_LIBXSMM_DNN( status );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_allocate_forget_bias(libxsmm_handle, forget_bias) );

    /* setup LIBXSMM buffers and filter */
    libxsmm_layout = libxsmm_dnn_rnncell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_RNN_REGULAR_INPUT, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_input = libxsmm_dnn_link_tensor( libxsmm_layout, xt, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_rnncell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_RNN_REGULAR_CS_PREV, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_cs_prev = libxsmm_dnn_link_tensor( libxsmm_layout, csp, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_rnncell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_RNN_REGULAR_HIDDEN_STATE_PREV, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_hidden_state_prev = libxsmm_dnn_link_tensor( libxsmm_layout, hp, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_rnncell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_RNN_REGULAR_WEIGHT, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_weight = libxsmm_dnn_link_tensor( libxsmm_layout, w, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_rnncell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_RNN_REGULAR_RECUR_WEIGHT, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_recur_weight = libxsmm_dnn_link_tensor( libxsmm_layout, r, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_rnncell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_RNN_REGULAR_BIAS, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_bias = libxsmm_dnn_link_tensor( libxsmm_layout, b, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_rnncell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_RNN_REGULAR_CS, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_cs = libxsmm_dnn_link_tensor( libxsmm_layout, cst, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_rnncell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_RNN_REGULAR_HIDDEN_STATE, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_hidden_state = libxsmm_dnn_link_tensor( libxsmm_layout, ht, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_rnncell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_RNN_INTERNAL_I, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_i = libxsmm_dnn_link_tensor( libxsmm_layout, it, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_rnncell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_RNN_INTERNAL_F, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_f = libxsmm_dnn_link_tensor( libxsmm_layout, ft, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_rnncell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_RNN_INTERNAL_O, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_o = libxsmm_dnn_link_tensor( libxsmm_layout, ot, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_rnncell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_RNN_INTERNAL_CI, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_ci = libxsmm_dnn_link_tensor( libxsmm_layout, cit, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_rnncell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_RNN_INTERNAL_CO, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_co = libxsmm_dnn_link_tensor( libxsmm_layout, cot, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_rnncell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_RNN_GRADIENT_INPUT, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dinput = libxsmm_dnn_link_tensor( libxsmm_layout, dxt, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_rnncell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_RNN_GRADIENT_CS_PREV, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dcs_prev = libxsmm_dnn_link_tensor( libxsmm_layout, dcsp, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_rnncell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_RNN_GRADIENT_HIDDEN_STATE_PREV, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dhidden_state_prev = libxsmm_dnn_link_tensor( libxsmm_layout, dhp, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_rnncell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_RNN_GRADIENT_WEIGHT, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dweight = libxsmm_dnn_link_tensor( libxsmm_layout, dw, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_rnncell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_RNN_GRADIENT_RECUR_WEIGHT, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_drecur_weight = libxsmm_dnn_link_tensor( libxsmm_layout, dr, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_rnncell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_RNN_GRADIENT_BIAS, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dbias = libxsmm_dnn_link_tensor( libxsmm_layout, db, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_rnncell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_RNN_GRADIENT_CS, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dcs = libxsmm_dnn_link_tensor( libxsmm_layout, dcs, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_rnncell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_RNN_GRADIENT_HIDDEN_STATE, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dhidden_state = libxsmm_dnn_link_tensor( libxsmm_layout, dht, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    /* copy in data to LIBXSMM format */
    matrix_copy(N*C*t, xgoldt, xt);
    matrix_copy(K*N, cspgold, csp);
    matrix_copy(K*N, hpgold, hp);
    convert_ck_c4k(C, K, wigold, w);
    convert_ck_c4k(C, K, wcgold, &(w[K]));
    convert_ck_c4k(C, K, wfgold, &(w[2*K]));
    convert_ck_c4k(C, K, wogold, &(w[3*K]));
    convert_ck_c4k(K, K, rigold, r);
    convert_ck_c4k(K, K, rcgold, &(r[K]));
    convert_ck_c4k(K, K, rfgold, &(r[2*K]));
    convert_ck_c4k(K, K, rogold, &(r[3*K]));
    matrix_copy(K, bigold, &(b[0]));
    matrix_copy(K, bcgold, &(b[K]));
    matrix_copy(K, bfgold, &(b[2*K]));
    matrix_copy(K, bogold, &(b[3*K]));
    matrix_copy(K*N*t, djdhgoldt, dht);
    matrix_copy(K*N, djdcsgold, dcs);

    /* bind buffers and filter to handle */
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_bind_tensor( libxsmm_handle, libxsmm_input, LIBXSMM_DNN_RNN_REGULAR_INPUT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_bind_tensor( libxsmm_handle, libxsmm_cs_prev, LIBXSMM_DNN_RNN_REGULAR_CS_PREV ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_bind_tensor( libxsmm_handle, libxsmm_hidden_state_prev, LIBXSMM_DNN_RNN_REGULAR_HIDDEN_STATE_PREV ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_bind_tensor( libxsmm_handle, libxsmm_weight, LIBXSMM_DNN_RNN_REGULAR_WEIGHT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_bind_tensor( libxsmm_handle, libxsmm_recur_weight, LIBXSMM_DNN_RNN_REGULAR_RECUR_WEIGHT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_bind_tensor( libxsmm_handle, libxsmm_bias, LIBXSMM_DNN_RNN_REGULAR_BIAS ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_bind_tensor( libxsmm_handle, libxsmm_cs, LIBXSMM_DNN_RNN_REGULAR_CS ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_bind_tensor( libxsmm_handle, libxsmm_hidden_state, LIBXSMM_DNN_RNN_REGULAR_HIDDEN_STATE ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_bind_tensor( libxsmm_handle, libxsmm_i, LIBXSMM_DNN_RNN_INTERNAL_I ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_bind_tensor( libxsmm_handle, libxsmm_f, LIBXSMM_DNN_RNN_INTERNAL_F ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_bind_tensor( libxsmm_handle, libxsmm_o, LIBXSMM_DNN_RNN_INTERNAL_O ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_bind_tensor( libxsmm_handle, libxsmm_ci, LIBXSMM_DNN_RNN_INTERNAL_CI ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_bind_tensor( libxsmm_handle, libxsmm_co, LIBXSMM_DNN_RNN_INTERNAL_CO ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_bind_tensor( libxsmm_handle, libxsmm_dinput, LIBXSMM_DNN_RNN_GRADIENT_INPUT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_bind_tensor( libxsmm_handle, libxsmm_dcs_prev, LIBXSMM_DNN_RNN_GRADIENT_CS_PREV ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_bind_tensor( libxsmm_handle, libxsmm_dhidden_state_prev, LIBXSMM_DNN_RNN_GRADIENT_HIDDEN_STATE_PREV ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_bind_tensor( libxsmm_handle, libxsmm_dweight, LIBXSMM_DNN_RNN_GRADIENT_WEIGHT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_bind_tensor( libxsmm_handle, libxsmm_drecur_weight, LIBXSMM_DNN_RNN_GRADIENT_RECUR_WEIGHT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_bind_tensor( libxsmm_handle, libxsmm_dbias, LIBXSMM_DNN_RNN_GRADIENT_BIAS ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_bind_tensor( libxsmm_handle, libxsmm_dcs, LIBXSMM_DNN_RNN_GRADIENT_CS ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_bind_tensor( libxsmm_handle, libxsmm_dhidden_state, LIBXSMM_DNN_RNN_GRADIENT_HIDDEN_STATE ) );

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
      internalstate = (0 != internalstate_size ? libxsmm_aligned_malloc( internalstate_size, 2097152 ) : NULL);
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_bind_internalstate( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_FWD, internalstate ) );
    } else {
      internalstate_size = libxsmm_dnn_rnncell_get_internalstate_size( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_ALL, &status );
      CHKERR_LIBXSMM_DNN( status );
      internalstate = (0 != internalstate_size ? libxsmm_aligned_malloc( internalstate_size, 2097152 ) : NULL);
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_bind_internalstate( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_ALL, internalstate ) );
    }
    zero_buf( (float*)internalstate, internalstate_size/4 );

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
        CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_FWD, 0, tid ) );
      }

      /* compare */
      libxsmm_matdiff(&norms_fwd, LIBXSMM_DATATYPE_F32, K*N, 1, &LIBXSMM_VLA_ACCESS(2, hgold, t-1, 0, K * N), &LIBXSMM_VLA_ACCESS(2, h, t-1, 0, K * N), 0, 0);
      printf("L1 reference  : %.25g\n", norms_fwd.l1_ref);
      printf("L1 test       : %.25g\n", norms_fwd.l1_tst);
      printf("L2 abs.error  : %.24f\n", norms_fwd.l2_abs);
      printf("L2 rel.error  : %.24f\n", norms_fwd.l2_rel);
      printf("Linf abs.error: %.24f\n", norms_fwd.linf_abs);
      printf("Linf rel.error: %.24f\n", norms_fwd.linf_rel);
      printf("Check-norm    : %.24f\n", norms_fwd.normf_rel);
      libxsmm_matdiff_reduce(&diff, &norms_fwd);
    } else {
      /* We need to always run FWD pass once to populate i, f, o, ci, co, cs, h */
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
    }

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
        CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_BWD, 0, tid ) );
      }

      /* compare */
      libxsmm_matdiff(&norms_bwd, LIBXSMM_DATATYPE_F32, N*C*t, 1, djdxgoldt, dxt, 0, 0);
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
        CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_UPD, 0, tid ) );
      }

      /* compare */
#if defined(TWO_GEMMS)
      libxsmm_matdiff(&norms_upd_w, LIBXSMM_DATATYPE_F32, C*K*4, 1, djdw4gold, dw, 0, 0);
#else
      libxsmm_matdiff(&norms_upd_w, LIBXSMM_DATATYPE_F32, C*K*4, 1, djdwr8gold, dw, 0, 0);
#endif
      printf("Delta weight\n");
      printf("L1 reference  : %.25g\n", norms_upd_w.l1_ref);
      printf("L1 test       : %.25g\n", norms_upd_w.l1_tst);
      printf("L2 abs.error  : %.24f\n", norms_upd_w.l2_abs);
      printf("L2 rel.error  : %.24f\n", norms_upd_w.l2_rel);
      printf("Linf abs.error: %.24f\n", norms_upd_w.linf_abs);
      printf("Linf rel.error: %.24f\n", norms_upd_w.linf_rel);
      printf("Check-norm    : %.24f\n", norms_upd_w.normf_rel);
      libxsmm_matdiff_reduce(&diff, &norms_upd_w);

#if defined(TWO_GEMMS)
      libxsmm_matdiff(&norms_upd_r, LIBXSMM_DATATYPE_F32, K*K*4, 1, djdr4gold, dr, 0, 0);
#else
      libxsmm_matdiff(&norms_upd_r, LIBXSMM_DATATYPE_F32, K*K*4, 1, &(djdwr8gold[C*K*4]), dr, 0, 0);
#endif
      printf("Delta recurrent weight\n");
      printf("L1 reference  : %.25g\n", norms_upd_r.l1_ref);
      printf("L1 test       : %.25g\n", norms_upd_r.l1_tst);
      printf("L2 abs.error  : %.24f\n", norms_upd_r.l2_abs);
      printf("L2 rel.error  : %.24f\n", norms_upd_r.l2_rel);
      printf("Linf abs.error: %.24f\n", norms_upd_r.linf_abs);
      printf("Linf rel.error: %.24f\n", norms_upd_r.linf_rel);
      printf("Check-norm    : %.24f\n", norms_upd_r.normf_rel);
      libxsmm_matdiff_reduce(&diff, &norms_upd_r);

      libxsmm_matdiff(&norms_upd_b, LIBXSMM_DATATYPE_F32, K*4, 1, djdb4gold, db, 0, 0);
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
        CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_BWDUPD, 0, tid ) );
      }

      /* compare */
      libxsmm_matdiff(&norms_bwd, LIBXSMM_DATATYPE_F32, N*C*t, 1, djdxgoldt, dxt, 0, 0);
      printf("Delta input\n");
      printf("L1 reference  : %.25g\n", norms_bwd.l1_ref);
      printf("L1 test       : %.25g\n", norms_bwd.l1_tst);
      printf("L2 abs.error  : %.24f\n", norms_bwd.l2_abs);
      printf("L2 rel.error  : %.24f\n", norms_bwd.l2_rel);
      printf("Linf abs.error: %.24f\n", norms_bwd.linf_abs);
      printf("Linf rel.error: %.24f\n", norms_bwd.linf_rel);
      printf("Check-norm    : %.24f\n", norms_bwd.normf_rel);
      libxsmm_matdiff_reduce(&diff, &norms_bwd);

#if defined(TWO_GEMMS)
      libxsmm_matdiff(&norms_upd_w, LIBXSMM_DATATYPE_F32, C*K*4, 1, djdw4gold, dw, 0, 0);
#else
      libxsmm_matdiff(&norms_upd_w, LIBXSMM_DATATYPE_F32, C*K*4, 1, djdwr8gold, dw, 0, 0);
#endif
      printf("Delta weight\n");
      printf("L1 reference  : %.25g\n", norms_upd_w.l1_ref);
      printf("L1 test       : %.25g\n", norms_upd_w.l1_tst);
      printf("L2 abs.error  : %.24f\n", norms_upd_w.l2_abs);
      printf("L2 rel.error  : %.24f\n", norms_upd_w.l2_rel);
      printf("Linf abs.error: %.24f\n", norms_upd_w.linf_abs);
      printf("Linf rel.error: %.24f\n", norms_upd_w.linf_rel);
      printf("Check-norm    : %.24f\n", norms_upd_w.normf_rel);
      libxsmm_matdiff_reduce(&diff, &norms_upd_w);

#if defined(TWO_GEMMS)
      libxsmm_matdiff(&norms_upd_r, LIBXSMM_DATATYPE_F32, K*K*4, 1, djdr4gold, dr, 0, 0);
#else
      libxsmm_matdiff(&norms_upd_r, LIBXSMM_DATATYPE_F32, K*K*4, 1, &(djdwr8gold[C*K*4]), dr, 0, 0);
#endif
      printf("Delta recurrent weight\n");
      printf("L1 reference  : %.25g\n", norms_upd_r.l1_ref);
      printf("L1 test       : %.25g\n", norms_upd_r.l1_tst);
      printf("L2 abs.error  : %.24f\n", norms_upd_r.l2_abs);
      printf("L2 rel.error  : %.24f\n", norms_upd_r.l2_rel);
      printf("Linf abs.error: %.24f\n", norms_upd_r.linf_abs);
      printf("Linf rel.error: %.24f\n", norms_upd_r.linf_rel);
      printf("Check-norm    : %.24f\n", norms_upd_r.normf_rel);
      libxsmm_matdiff_reduce(&diff, &norms_upd_r);

      libxsmm_matdiff(&norms_upd_b, LIBXSMM_DATATYPE_F32, K*4, 1, djdb4gold, db, 0, 0);
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

    if ( pass == 0 ) {
      printf("##########################################\n");
      printf("#   Performance - FWD (custom-Storage)   #\n");
      printf("##########################################\n");
      /* run LIBXSMM LSTM for performance */
      l_start = libxsmm_timer_tick();

#if defined(_OPENMP)
#     pragma omp parallel private(j)
#endif
      {
#if defined(_OPENMP)
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        for (j = 0; j < iters; ++j) {
          libxsmm_dnn_rnncell_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_FWD, 0, tid );
        }
      }
      l_end = libxsmm_timer_tick();
      l_total = libxsmm_timer_duration(l_start, l_end);
      flops = (((2.0 * K * N * C) + (2.0 * K * N * K) + (2.0 * K * N) + (tflops * K * N)) * 4.0 + (4.0 * K * N) + (tflops * K * N)) * (double)t * (double)iters;

      printf("GFLOP  = %.5g\n", flops*1e-9/(double)iters);
      printf("fp time = %.5g\n", ((double)(l_total/iters)));
      printf("GFLOPS  = %.5g\n", (flops*1e-9)/l_total);

      printf("PERFDUMP,FP,%s,%i,%i,%i,%i,%i,%.5g,%.5g\n", LIBXSMM_VERSION, nThreads, N, C, K, t, ((double)(l_total/iters)), (flops*1e-9)/l_total);
    }

    if ( pass == 1 ) {
      printf("##########################################\n");
      printf("#   Performance - BWD (custom-Storage)   #\n");
      printf("##########################################\n");
      /* run LIBXSMM LSTM for performance */
      l_start = libxsmm_timer_tick();

#if defined(_OPENMP)
#     pragma omp parallel private(j)
#endif
      {
#if defined(_OPENMP)
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        for (j = 0; j < iters; ++j) {
          libxsmm_dnn_rnncell_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_BWD, 0, tid );
        }
      }
      l_end = libxsmm_timer_tick();
      l_total = libxsmm_timer_duration(l_start, l_end);
      flops = K * N; /* delta + delta_out */
      flops += (6.0 * K * N + tflops * K * N); /* dJdd */
      flops += (4.0 * K * N); /* dJdc */
      flops += (4.0 * K * N); /* dJdi */
      flops += (4.0 * K * N); /* dJdf */
      flops += (4.0 * K * N + tflops * K * N); /* dJdo */
      tempflops = (4.0 * K * C); /* W^T */
      tempflops += (8.0 * K * N * C); /* W^T * dJd{c, i, f, o} */
      tempflops += (3.0 * K * C); /* summation */
      flops += tempflops;
      tempflops = (4.0 * K * K); /* R^T */
      tempflops += (8.0 * K * N * K); /* R^T * dJd{c, i, f, o} */
      flops += tempflops;
      flops *= t; /* for t time steps */
      flops *= iters;

      printf("GFLOP  = %.5g\n", flops*1e-9/(double)iters);
      printf("bp time = %.5g\n", ((double)(l_total/iters)));
      printf("GFLOPS  = %.5g\n", (flops*1e-9)/l_total);

      printf("PERFDUMP,BP,%s,%i,%i,%i,%i,%i,%.5g,%.5g\n", LIBXSMM_VERSION, nThreads, N, C, K, t, ((double)(l_total/iters)), (flops*1e-9)/l_total);
    }

    if ( pass == 2 ) {
      printf("##########################################\n");
      printf("#   Performance - UPD (custom-Storage)   #\n");
      printf("##########################################\n");
      /* run LIBXSMM LSTM for performance */
      l_start = libxsmm_timer_tick();

#if defined(_OPENMP)
#     pragma omp parallel private(j)
#endif
      {
#if defined(_OPENMP)
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        for (j = 0; j < iters; ++j) {
          libxsmm_dnn_rnncell_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_UPD, 0, tid );
        }
      }
      l_end = libxsmm_timer_tick();
      l_total = libxsmm_timer_duration(l_start, l_end);
      flops = K * N; /* delta + delta_out */
      flops += (6.0 * K * N + tflops * K * N); /* dJdd */
      flops += (4.0 * K * N); /* dJdc */
      flops += (4.0 * K * N); /* dJdi */
      flops += (4.0 * K * N); /* dJdf */
      flops += (4.0 * K * N + tflops * K * N); /* dJdo */
      tempflops = (4.0 * K * K); /* R^T */
      tempflops += (8.0 * K * N * K); /* R^T * dJd{c, i, f, o} */
      flops += tempflops;
      flops *= t; /* for t time steps */
      tempflops = C * N; /* x^T */
      tempflops += (8.0 * K * N * C); /* delta{c, i, f, o} * x^T */
      tempflops *= t; /* for t time steps */
      tempflops += (4.0 * K * C * (t-1)); /* for summation of dJdW{c, i, f, o} */
      flops += tempflops;
      tempflops = 4.0 * K * N; /* delta^T */
      tempflops += (8.0 * K * N * K); /* delta{c, i, f, o} * delta^T */
      tempflops *= (t - 1); /* for (t - 1) time steps */
      tempflops += (4.0 * K * N * (t-2)); /* for summation of dJdR{c, i, f, o} */
      flops += tempflops;
      flops += (4.0 * K * N * (t - 1)); /* delbias */
      flops *= iters;

      printf("GFLOP  = %.5g\n", flops*1e-9/(double)iters);
      printf("wu time = %.5g\n", ((double)(l_total/iters)));
      printf("GFLOPS  = %.5g\n", (flops*1e-9)/l_total);

      printf("PERFDUMP,WU,%s,%i,%i,%i,%i,%i,%.5g,%.5g\n", LIBXSMM_VERSION, nThreads, N, C, K, t, ((double)(l_total/iters)), (flops*1e-9)/l_total);
    }

    if ( pass == 3 ) {
      printf("##########################################\n");
      printf("# Performance - BWD+UPD (custom-Storage) #\n");
      printf("##########################################\n");
      /* run LIBXSMM LSTM for performance */
      l_start = libxsmm_timer_tick();

#if defined(_OPENMP)
#     pragma omp parallel private(j)
#endif
      {
#if defined(_OPENMP)
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        for (j = 0; j < iters; ++j) {
          libxsmm_dnn_rnncell_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_BWDUPD, 0, tid );
        }
      }
      l_end = libxsmm_timer_tick();
      l_total = libxsmm_timer_duration(l_start, l_end);
      flops = K * N; /* delta + delta_out */
      flops += (6.0 * K * N + tflops * K * N); /* dJdd */
      flops += (4.0 * K * N); /* dJdc */
      flops += (4.0 * K * N); /* dJdi */
      flops += (4.0 * K * N); /* dJdf */
      flops += (4.0 * K * N + tflops * K * N); /* dJdo */
      tempflops = (4.0 * K * C); /* W^T */
      tempflops += (8.0 * K * N * C); /* W^T * dJd{c, i, f, o} */
      tempflops += (3.0 * K * C); /* summation */
      flops += tempflops;
      tempflops = (4.0 * K * K); /* R^T */
      tempflops += (8.0 * K * N * K); /* R^T * dJd{c, i, f, o} */
      flops += tempflops;
      flops *= t; /* for t time steps */
      tempflops = C * N; /* x^T */
      tempflops += (8.0 * K * N * C); /* delta{c, i, f, o} * x^T */
      tempflops *= t; /* for t time steps */
      tempflops += (4.0 * K * C * (t-1)); /* for summation of dJdW{c, i, f, o} */
      flops += tempflops;
      tempflops = 4.0 * K * N; /* delta^T */
      tempflops += (8.0 * K * N * K); /* delta{c, i, f, o} * delta^T */
      tempflops *= (t - 1); /* for (t - 1) time steps */
      tempflops += (4.0 * K * N * (t-2)); /* for summation of dJdR{c, i, f, o} */
      flops += tempflops;
      flops += (4.0 * K * N * (t - 1)); /* delbias */
      flops *= iters;

      printf("GFLOP  = %.5g\n", flops*1e-9/(double)iters);
      printf("bp+wu time = %.5g\n", ((double)(l_total/iters)));
      printf("GFLOPS  = %.5g\n", (flops*1e-9)/l_total);

      printf("PERFDUMP,BP+WU,%s,%i,%i,%i,%i,%i,%.5g,%.5g\n", LIBXSMM_VERSION, nThreads, N, C, K, t, ((double)(l_total/iters)), (flops*1e-9)/l_total);
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
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_release_tensor( libxsmm_handle, LIBXSMM_DNN_RNN_REGULAR_CS_PREV ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_release_tensor( libxsmm_handle, LIBXSMM_DNN_RNN_REGULAR_HIDDEN_STATE_PREV ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_release_tensor( libxsmm_handle, LIBXSMM_DNN_RNN_REGULAR_WEIGHT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_release_tensor( libxsmm_handle, LIBXSMM_DNN_RNN_REGULAR_RECUR_WEIGHT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_release_tensor( libxsmm_handle, LIBXSMM_DNN_RNN_REGULAR_BIAS ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_release_tensor( libxsmm_handle, LIBXSMM_DNN_RNN_REGULAR_CS ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_release_tensor( libxsmm_handle, LIBXSMM_DNN_RNN_REGULAR_HIDDEN_STATE ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_release_tensor( libxsmm_handle, LIBXSMM_DNN_RNN_INTERNAL_I ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_release_tensor( libxsmm_handle, LIBXSMM_DNN_RNN_INTERNAL_F ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_release_tensor( libxsmm_handle, LIBXSMM_DNN_RNN_INTERNAL_O ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_release_tensor( libxsmm_handle, LIBXSMM_DNN_RNN_INTERNAL_CI ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_release_tensor( libxsmm_handle, LIBXSMM_DNN_RNN_INTERNAL_CO ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_release_tensor( libxsmm_handle, LIBXSMM_DNN_RNN_GRADIENT_INPUT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_release_tensor( libxsmm_handle, LIBXSMM_DNN_RNN_GRADIENT_CS_PREV ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_release_tensor( libxsmm_handle, LIBXSMM_DNN_RNN_GRADIENT_HIDDEN_STATE_PREV ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_release_tensor( libxsmm_handle, LIBXSMM_DNN_RNN_GRADIENT_WEIGHT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_release_tensor( libxsmm_handle, LIBXSMM_DNN_RNN_GRADIENT_RECUR_WEIGHT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_release_tensor( libxsmm_handle, LIBXSMM_DNN_RNN_GRADIENT_BIAS ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_release_tensor( libxsmm_handle, LIBXSMM_DNN_RNN_GRADIENT_CS ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_release_tensor( libxsmm_handle, LIBXSMM_DNN_RNN_GRADIENT_HIDDEN_STATE ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_input ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_cs_prev ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_hidden_state_prev ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_weight ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_recur_weight ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_bias ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_cs ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_hidden_state ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_i ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_f ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_o ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_ci ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_co ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_dinput ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_dcs_prev ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_dhidden_state_prev ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_dweight ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_drecur_weight ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_dbias ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_dcs ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_dhidden_state ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_rnncell( libxsmm_handle ) );
  }

  /* deallocate data */
  libxsmm_free(xgoldt);
  libxsmm_free(cspgold);
  libxsmm_free(hpgold);
  libxsmm_free(wigold);
  libxsmm_free(wfgold);
  libxsmm_free(wogold);
  libxsmm_free(wcgold);
  libxsmm_free(rigold);
  libxsmm_free(rfgold);
  libxsmm_free(rogold);
  libxsmm_free(rcgold);
  libxsmm_free(bigold);
  libxsmm_free(bfgold);
  libxsmm_free(bogold);
  libxsmm_free(bcgold);
  libxsmm_free(bfgold_fb);
  libxsmm_free(dgoldt);
  libxsmm_free(hgoldt);
  libxsmm_free(icfogoldt);
  libxsmm_free(i1gold);
  libxsmm_free(i2gold);
  libxsmm_free(f1gold);
  libxsmm_free(f2gold);
  libxsmm_free(o1gold);
  libxsmm_free(o2gold);
  libxsmm_free(c1gold);
  libxsmm_free(c2gold);
  libxsmm_free(d1gold);
  libxsmm_free(d2gold);
  libxsmm_free(dhgold);
  libxsmm_free(i3gold);
  libxsmm_free(f3gold);
  libxsmm_free(d3gold);
  libxsmm_free(d4gold);
  libxsmm_free(deltagoldt);
  libxsmm_free(djdhgoldt);
  libxsmm_free(djdcspgold);
  libxsmm_free(djdigoldt);
  libxsmm_free(djdfgoldt);
  libxsmm_free(djdogoldt);
  libxsmm_free(djdcgoldt);
  libxsmm_free(djdxgoldt);
  libxsmm_free(djdhpgold);
  libxsmm_free(djdb4gold);
  libxsmm_free(wigoldTp);
  libxsmm_free(wcgoldTp);
  libxsmm_free(wfgoldTp);
  libxsmm_free(wogoldTp);
  libxsmm_free(rigoldTp);
  libxsmm_free(rcgoldTp);
  libxsmm_free(rfgoldTp);
  libxsmm_free(rogoldTp);
  libxsmm_free(doutgoldt);
#if defined(TWO_GEMMS)
  libxsmm_free(w4gold);
  libxsmm_free(r4gold);
  libxsmm_free(djdw4gold);
  libxsmm_free(djdr4gold);
  libxsmm_free(xgoldTp);
  libxsmm_free(hgoldTp);
#else
  libxsmm_free(wr8gold);
  libxsmm_free(xhgold);
  libxsmm_free(djdwr8gold);
  libxsmm_free(xhgoldTp);
#endif
  libxsmm_free(xt);
  libxsmm_free(csp);
  libxsmm_free(hp);
  libxsmm_free(w);
  libxsmm_free(r);
  libxsmm_free(b);
  libxsmm_free(cst);
  libxsmm_free(ht);
  libxsmm_free(dxt);
  libxsmm_free(dcsp);
  libxsmm_free(dhp);
  libxsmm_free(dw);
  libxsmm_free(dr);
  libxsmm_free(db);
  libxsmm_free(dcs);
  libxsmm_free(dht);
  libxsmm_free(it);
  libxsmm_free(ft);
  libxsmm_free(ot);
  libxsmm_free(cit);
  libxsmm_free(cot);
  libxsmm_free(djdcsgold);

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

