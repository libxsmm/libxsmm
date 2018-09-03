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
  libxsmm_otrans_omp(dst, src, sizeof(float), rows, cols, rows/*ldi*/, rows/*ldo*/);
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


LIBXSMM_INLINE void matrix_inverse(int size, float *src, float *dst)
{
  int i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < size; i++) {
    dst[i] = -src[i];
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
  float *djdhgoldt = NULL, *rgoldt = NULL, *zgoldt = NULL, *ggoldt = NULL, *hgoldt = NULL;
  float *djdxgoldt = NULL, *djdwrgold = NULL, *djdwzgold = NULL, *djdwggold = NULL, *djdurgold = NULL, *djduzgold = NULL, *djduggold = NULL;
  float *djdbrgold = NULL, *djdbzgold = NULL, *djdbggold = NULL, *wrgoldTp = NULL, *wzgoldTp = NULL, *wggoldTp = NULL, *xgoldTp = NULL, *hgoldTp = NULL;
  float *urgoldTp = NULL, *uzgoldTp = NULL, *uggoldTp = NULL;
  float *d3gold = NULL, *d4gold = NULL, *d5gold = NULL, *d6gold = NULL, *d7gold = NULL, *d8gold = NULL, *d9gold = NULL, *d10gold = NULL, *d11gold = NULL, *d12gold = NULL, *d13gold = NULL;
  float *d14gold = NULL, *d15gold = NULL, *d16gold = NULL, *d17gold = NULL, *d18gold = NULL, *d19gold = NULL, *d20gold = NULL, *d21gold = NULL, *d22gold = NULL, *d23gold = NULL;
  float *ht = NULL, *djdht = NULL, *djdxt = NULL, *djdwr = NULL, *djdwz = NULL, *djdwg = NULL, *djdur = NULL, *djduz = NULL, *djdug = NULL, *djdbr = NULL, *djdbz = NULL, *djdbg = NULL;
  float *djdxtestt = NULL, *djdwtest = NULL, *djdutest = NULL, *djdbtest = NULL, *djdwgold3 = NULL, *djdugold3 = NULL, *djdbgold3 = NULL;

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
  int t = 10;       /* number of time steps (> 1) */
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
  double flops = 0.0;
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
    urgold = (float*)libxsmm_aligned_malloc(m * k * sizeof(float), 2097152);
    uzgold = (float*)libxsmm_aligned_malloc(m * k * sizeof(float), 2097152);
    uggold = (float*)libxsmm_aligned_malloc(m * k * sizeof(float), 2097152);
    xgoldt = (float*)libxsmm_aligned_malloc(k * n * sizeof(float) * t, 2097152);
    wrgold = (float*)libxsmm_aligned_malloc(m * m * sizeof(float), 2097152);
    wzgold = (float*)libxsmm_aligned_malloc(m * m * sizeof(float), 2097152);
    wggold = (float*)libxsmm_aligned_malloc(m * m * sizeof(float), 2097152);
    hgoldt = (float*)libxsmm_aligned_malloc(m * n * sizeof(float) * t, 2097152);
    rgoldt = (float*)libxsmm_aligned_malloc(m * n * sizeof(float) * t, 2097152);
    zgoldt = (float*)libxsmm_aligned_malloc(m * n * sizeof(float) * t, 2097152);
    ggoldt = (float*)libxsmm_aligned_malloc(m * n * sizeof(float) * t, 2097152);
    d3gold = (float*)libxsmm_aligned_malloc(m * n * sizeof(float), 2097152);
    d4gold = (float*)libxsmm_aligned_malloc(m * n * sizeof(float), 2097152);
    d5gold = (float*)libxsmm_aligned_malloc(m * n * sizeof(float), 2097152);
    d6gold = (float*)libxsmm_aligned_malloc(m * n * sizeof(float), 2097152);
    d7gold = (float*)libxsmm_aligned_malloc(m * n * sizeof(float), 2097152);
    d8gold = (float*)libxsmm_aligned_malloc(m * n * sizeof(float), 2097152);
    d9gold = (float*)libxsmm_aligned_malloc(m * n * sizeof(float), 2097152);
    d10gold= (float*)libxsmm_aligned_malloc(m * n * sizeof(float), 2097152);
    d11gold= (float*)libxsmm_aligned_malloc(m * n * sizeof(float), 2097152);
    d12gold= (float*)libxsmm_aligned_malloc(k * n * sizeof(float), 2097152);
    d13gold= (float*)libxsmm_aligned_malloc(m * n * sizeof(float), 2097152);
    d14gold= (float*)libxsmm_aligned_malloc(k * n * sizeof(float), 2097152);
    d15gold= (float*)libxsmm_aligned_malloc(m * n * sizeof(float), 2097152);
    d16gold= (float*)libxsmm_aligned_malloc(m * n * sizeof(float), 2097152);
    d17gold= (float*)libxsmm_aligned_malloc(m * n * sizeof(float), 2097152);
    d18gold= (float*)libxsmm_aligned_malloc(m * n * sizeof(float), 2097152);
    d19gold= (float*)libxsmm_aligned_malloc(m * n * sizeof(float), 2097152);
    d20gold= (float*)libxsmm_aligned_malloc(k * n * sizeof(float), 2097152);
    d21gold= (float*)libxsmm_aligned_malloc(m * n * sizeof(float), 2097152);
    d22gold= (float*)libxsmm_aligned_malloc(m * n * sizeof(float), 2097152);
    d23gold= (float*)libxsmm_aligned_malloc(m * n * sizeof(float), 2097152);
    djdhgoldt = (float*)libxsmm_aligned_malloc(m * n * sizeof(float) * t, 2097152);
    djdxgoldt = (float*)libxsmm_aligned_malloc(k * n * sizeof(float) * t, 2097152);
    djdurgold = (float*)libxsmm_aligned_malloc(m * k * sizeof(float), 2097152);
    djduzgold = (float*)libxsmm_aligned_malloc(m * k * sizeof(float), 2097152);
    djduggold = (float*)libxsmm_aligned_malloc(m * k * sizeof(float), 2097152);
    djdwrgold = (float*)libxsmm_aligned_malloc(m * m * sizeof(float), 2097152);
    djdwzgold = (float*)libxsmm_aligned_malloc(m * m * sizeof(float), 2097152);
    djdwggold = (float*)libxsmm_aligned_malloc(m * m * sizeof(float), 2097152);
    djdbrgold = (float*)libxsmm_aligned_malloc(m * n * sizeof(float), 2097152);
    djdbzgold = (float*)libxsmm_aligned_malloc(m * n * sizeof(float), 2097152);
    djdbggold = (float*)libxsmm_aligned_malloc(m * n * sizeof(float), 2097152);
    urgoldTp = (float*)libxsmm_aligned_malloc(m * k * sizeof(float), 2097152);
    uzgoldTp = (float*)libxsmm_aligned_malloc(m * k * sizeof(float), 2097152);
    uggoldTp = (float*)libxsmm_aligned_malloc(m * k * sizeof(float), 2097152);
    wrgoldTp = (float*)libxsmm_aligned_malloc(m * m * sizeof(float), 2097152);
    wzgoldTp = (float*)libxsmm_aligned_malloc(m * m * sizeof(float), 2097152);
    wggoldTp = (float*)libxsmm_aligned_malloc(m * m * sizeof(float), 2097152);
    xgoldTp = (float*)libxsmm_aligned_malloc(k * n * sizeof(float), 2097152);
    hgoldTp = (float*)libxsmm_aligned_malloc(m * n * sizeof(float), 2097152);
    ur = (float*)libxsmm_aligned_malloc(m * k * sizeof(float), 2097152);
    uz = (float*)libxsmm_aligned_malloc(m * k * sizeof(float), 2097152);
    ug = (float*)libxsmm_aligned_malloc(m * k * sizeof(float), 2097152);
    xt = (float*)libxsmm_aligned_malloc(k * n * sizeof(float) * t, 2097152);
    wr = (float*)libxsmm_aligned_malloc(m * m * sizeof(float), 2097152);
    wz = (float*)libxsmm_aligned_malloc(m * m * sizeof(float), 2097152);
    wg = (float*)libxsmm_aligned_malloc(m * m * sizeof(float), 2097152);
    ht = (float*)libxsmm_aligned_malloc(m * n * sizeof(float) * t, 2097152);
    br = (float*)libxsmm_aligned_malloc(m * n * sizeof(float), 2097152);
    bz = (float*)libxsmm_aligned_malloc(m * n * sizeof(float), 2097152);
    bg = (float*)libxsmm_aligned_malloc(m * n * sizeof(float), 2097152);
    djdht = (float*)libxsmm_aligned_malloc(m * n * sizeof(float) * t, 2097152);
    djdxt = (float*)libxsmm_aligned_malloc(k * n * sizeof(float) * t, 2097152);
    djdur = (float*)libxsmm_aligned_malloc(m * k * sizeof(float), 2097152);
    djduz = (float*)libxsmm_aligned_malloc(m * k * sizeof(float), 2097152);
    djdug = (float*)libxsmm_aligned_malloc(m * k * sizeof(float), 2097152);
    djdwr = (float*)libxsmm_aligned_malloc(m * m * sizeof(float), 2097152);
    djdwz = (float*)libxsmm_aligned_malloc(m * m * sizeof(float), 2097152);
    djdwg = (float*)libxsmm_aligned_malloc(m * m * sizeof(float), 2097152);
    djdbr = (float*)libxsmm_aligned_malloc(m * n * sizeof(float), 2097152);
    djdbz = (float*)libxsmm_aligned_malloc(m * n * sizeof(float), 2097152);
    djdbg = (float*)libxsmm_aligned_malloc(m * n * sizeof(float), 2097152);
    djdxtestt  = (float*)libxsmm_aligned_malloc(k * n * sizeof(float) * t, 2097152);
    djdutest   = (float*)libxsmm_aligned_malloc(m * k * sizeof(float) * 3, 2097152);
    djdwtest   = (float*)libxsmm_aligned_malloc(m * m * sizeof(float) * 3, 2097152);
    djdbtest   = (float*)libxsmm_aligned_malloc(m * n * sizeof(float) * 3, 2097152);
    djdugold3  = (float*)libxsmm_aligned_malloc(m * k * sizeof(float) * 3, 2097152);
    djdwgold3  = (float*)libxsmm_aligned_malloc(m * m * sizeof(float) * 3, 2097152);
    djdbgold3  = (float*)libxsmm_aligned_malloc(m * n * sizeof(float) * 3, 2097152);
  }
  LIBXSMM_VLA_DECL(2, float, xgold, xgoldt, k * n);
  LIBXSMM_VLA_DECL(2, float, rgoldb, rgoldt, m * n);
  LIBXSMM_VLA_DECL(2, float, zgoldb, zgoldt, m * n);
  LIBXSMM_VLA_DECL(2, float, ggoldb, ggoldt, m * n);
  LIBXSMM_VLA_DECL(2, float, hgoldb, hgoldt, m * n);
  LIBXSMM_VLA_DECL(2, float, djdhgold, djdhgoldt, m * n);
  LIBXSMM_VLA_DECL(2, float, djdxgold, djdxgoldt, k * n);

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
    LIBXSMM_MATINIT(float, 42, urgold, m, k, m, 0.01);
    LIBXSMM_MATINIT(float, 42, uzgold, m, k, m, 0.01);
    LIBXSMM_MATINIT(float, 42, uggold, m, k, m, 0.01);
    LIBXSMM_MATINIT(float, 42, wrgold, m, m, m, 0.01);
    LIBXSMM_MATINIT(float, 42, wzgold, m, m, m, 0.01);
    LIBXSMM_MATINIT(float, 42, wggold, m, m, m, 0.01);
    for (it = 0; it < t; ++it) {
      LIBXSMM_MATINIT(float, 24, &LIBXSMM_VLA_ACCESS(2, xgold, it, 0, k * n), k, n, k, 0.01);
      LIBXSMM_MATINIT(float, 24, &LIBXSMM_VLA_ACCESS(2, hgoldb, it, 0, m * n), m, n, m, 0.01);
      LIBXSMM_MATINIT(float, 24, &LIBXSMM_VLA_ACCESS(2, rgoldb, it, 0, m * n), m, n, m, 0.01);
      LIBXSMM_MATINIT(float, 24, &LIBXSMM_VLA_ACCESS(2, zgoldb, it, 0, m * n), m, n, m, 0.01);
      LIBXSMM_MATINIT(float, 24, &LIBXSMM_VLA_ACCESS(2, ggoldb, it, 0, m * n), m, n, m, 0.01);
      LIBXSMM_MATINIT(float, 24, &LIBXSMM_VLA_ACCESS(2, djdhgold, it, 0, m * n), m, n, m, 0.01);
    }
    zero_buf(d3gold, m*n);
    zero_buf(d4gold, m*n);
    zero_buf(d5gold, m*n);
    zero_buf(d6gold, m*n);
    zero_buf(d7gold, m*n);
    zero_buf(d8gold, m*n);
    zero_buf(d9gold, m*n);
    zero_buf(d10gold, m*n);
    zero_buf(d11gold, m*n);
    zero_buf(d12gold, k*n);
    zero_buf(d13gold, m*n);
    zero_buf(d14gold, k*n);
    zero_buf(d15gold, m*n);
    zero_buf(d16gold, m*n);
    zero_buf(d17gold, m*n);
    zero_buf(d18gold, m*n);
    zero_buf(d19gold, m*n);
    zero_buf(d20gold, k*n);
    zero_buf(d21gold, m*n);
    zero_buf(d22gold, m*n);
    zero_buf(d23gold, m*n);
    zero_buf(djdxgoldt, k*n*t);
    zero_buf(djdurgold, m*k);
    zero_buf(djduzgold, m*k);
    zero_buf(djduggold, m*k);
    zero_buf(djdwrgold, m*m);
    zero_buf(djdwzgold, m*m);
    zero_buf(djdwggold, m*m);
    zero_buf(djdbrgold, m*n);
    zero_buf(djdbzgold, m*n);
    zero_buf(djdbggold, m*n);
    zero_buf(urgoldTp, m*k);
    zero_buf(uzgoldTp, m*k);
    zero_buf(uggoldTp, m*k);
    zero_buf(wrgoldTp, m*m);
    zero_buf(wzgoldTp, m*m);
    zero_buf(wggoldTp, m*m);
    zero_buf(xgoldTp, k*n);
    zero_buf(hgoldTp, m*n);
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
    zero_buf(ur, m*k);
    zero_buf(uz, m*k);
    zero_buf(ug, m*k);
    zero_buf(xt, k*n*t);
    zero_buf(wr, m*m);
    zero_buf(wz, m*m);
    zero_buf(wg, m*m);
    zero_buf(ht, m*n*t);
    zero_buf(br, m*n);
    zero_buf(bz, m*n);
    zero_buf(bg, m*n);
    zero_buf(djdur, m*k);
    zero_buf(djduz, m*k);
    zero_buf(djdug, m*k);
    zero_buf(djdxt, k*n*t);
    zero_buf(djdwr, m*m);
    zero_buf(djdwz, m*m);
    zero_buf(djdwg, m*m);
    zero_buf(djdht, m*n*t);
    zero_buf(djdbr, m*n);
    zero_buf(djdbz, m*n);
    zero_buf(djdbg, m*n);
  }
  LIBXSMM_VLA_DECL(2, float, x, xt, k * n);
  LIBXSMM_VLA_DECL(2, float, hnr, h, m * n);
  LIBXSMM_VLA_DECL(2, float, djdx, djdxt, k * n);
  LIBXSMM_VLA_DECL(2, float, djdh, djdht, m * n);
  LIBXSMM_VLA_DECL(2, float, hb, ht, m * n);

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
    handlewd = libxsmm_bgemm_handle_create(nThreads, LIBXSMM_GEMM_PRECISION(float), LIBXSMM_GEMM_PRECISION(float),
      m, n, m, &bm, &bn, &bm, &b_m1, &b_n1, &b_m1, &b_m2,
      &alpha, &beta, &gemm_flags, &strategy, &order); /* W^T*delta */
    handlewh = libxsmm_bgemm_handle_create(nThreads, LIBXSMM_GEMM_PRECISION(float), LIBXSMM_GEMM_PRECISION(float),
      m, m, n, &bm, &bm, &bn, &b_m1, &b_m1, &b_n1, &b_n2,
      &alpha, &beta, &gemm_flags, &strategy, &order); /* delta*h^T */
    handlett = libxsmm_bgemm_handle_create(nThreads, LIBXSMM_GEMM_PRECISION(float), LIBXSMM_GEMM_PRECISION(float),
      m, k, n, &bm, &bk, &bn, &b_m1, &b_k1, &b_n1, &b_n2,
      &alpha, &beta, &gemm_flags, &strategy, &order); /* delta*x^T */
    handleux = libxsmm_bgemm_handle_create(nThreads, LIBXSMM_GEMM_PRECISION(float), LIBXSMM_GEMM_PRECISION(float),
      k, n, m, &bk, &bn, &bm, &b_k1, &b_n1, &b_m1, &b_m2,
      &alpha, &beta, &gemm_flags, &strategy, &order); /* U^T*delta */
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
      matrix_transpose(m, k, urgold, urgoldTp);
      matrix_transpose(m, k, uzgold, uzgoldTp);
      matrix_transpose(m, k, uggold, uggoldTp);
      matrix_transpose(m, m, wrgold, wrgoldTp);
      matrix_transpose(m, m, wzgold, wzgoldTp);
      matrix_transpose(m, m, wggold, wggoldTp);
      for (j = t-1; j >= 0; j--) {
        /* d3 = djdh + d23 (delta) */
        matrix_add(m * n, &LIBXSMM_VLA_ACCESS(2, djdhgold, j, 0, m * n), d23gold, d3gold);
        /* d4 = (1 - z).d3 */
        matrix_complement(m * n, &LIBXSMM_VLA_ACCESS(2, zgoldb, j, 0, m * n), d4gold);
        matrix_eltwise_mult(m * n, d4gold, d3gold, d4gold);
        /* d5 = d3.h */
        matrix_eltwise_mult(m * n, d3gold, &LIBXSMM_VLA_ACCESS(2, hgoldb, j, 0, m * n), d5gold);
        /* d6 = -d5 */
        matrix_inverse(m * n, d5gold, d6gold);
        /* d7 = d3.g */
        matrix_eltwise_mult(m * n, d3gold, &LIBXSMM_VLA_ACCESS(2, ggoldb, j, 0, m * n), d7gold);
        /* d8 = d3.z */
        matrix_eltwise_mult(m * n, d3gold, &LIBXSMM_VLA_ACCESS(2, zgoldb, j, 0, m * n), d8gold);
        /* d9 = d7 + d8 */
        matrix_add(m * n, d7gold, d8gold, d9gold);
        /* d10 = d8.tanh'(g) */
        matrix_complement_square(m * n, &LIBXSMM_VLA_ACCESS(2, ggoldb, j, 0, m * n), d10gold);
        matrix_eltwise_mult(m * n, d8gold, d10gold, d10gold);
        /* d11 = d9.sig'(z) */
        matrix_complement(m * n, &LIBXSMM_VLA_ACCESS(2, zgoldb, j, 0, m * n), d11gold);
        matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, zgoldb, j, 0, m * n), d11gold, d11gold);
        matrix_eltwise_mult(m * n, d9gold, d11gold, d11gold);
        /* d13 = Wg^T * d10 */
        LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &m, &n, &m, &alpha, wggoldTp, &m, d10gold, &m, &beta0, d13gold, &m);
        /* d15 = Wz^T * d11 */
        LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &m, &n, &m, &alpha, wzgoldTp, &m, d11gold, &m, &beta0, d15gold, &m);
        /* d16 = d13.h */
        matrix_eltwise_mult(m * n, d13gold, &LIBXSMM_VLA_ACCESS(2, hgoldb, j, 0, m * n), d16gold);
        /* d17 = d13.r */
        matrix_eltwise_mult(m * n, d13gold, &LIBXSMM_VLA_ACCESS(2, rgoldb, j, 0, m * n), d17gold);
        /* d18 = d16.sig'(r) */
        matrix_complement(m * n, &LIBXSMM_VLA_ACCESS(2, rgoldb, j, 0, m * n), d18gold);
        matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, rgoldb, j, 0, m * n), d18gold, d18gold);
        matrix_eltwise_mult(m * n, d16gold, d18gold, d18gold);
        /* d19 = d17 + d4 */
        matrix_add(m * n, d17gold, d4gold, d19gold);
        /* d21 = Wr^T * d18 */
        LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &m, &n, &m, &alpha, wrgoldTp, &m, d18gold, &m, &beta0, d21gold, &m);
        /* d22 = d21 + d15 */
        matrix_add(m * n, d21gold, d15gold, d22gold);
        /* d23 = d19 + d22 */
        matrix_add(m * n, d19gold, d22gold, d23gold);
        if (1 == pass || 3 == pass) {
          /* d12 = Ug^T * d10 */
          LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &k, &n, &m, &alpha, uggoldTp, &k, d10gold, &m, &beta0, d12gold, &k);
          /* d14 = Uz^T * d11 */
          LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &k, &n, &m, &alpha, uzgoldTp, &k, d11gold, &m, &beta0, d14gold, &k);
          /* d20 = Ur^T * d18 */
          LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &k, &n, &m, &alpha, urgoldTp, &k, d18gold, &m, &beta0, d20gold, &k);
          /* djdx = d12 + d14 + d20 */
          matrix_add(k * n, d12gold, d14gold, &LIBXSMM_VLA_ACCESS(2, djdxgold, j, 0, k * n));
          matrix_add(k * n, &LIBXSMM_VLA_ACCESS(2, djdxgold, j, 0, k * n), d20gold, &LIBXSMM_VLA_ACCESS(2, djdxgold, j, 0, k * n));
        }
        if (2 == pass || 3 == pass) {
          /* djdwr = djdwr + d18 * h^T */
          matrix_transpose(m, n, &LIBXSMM_VLA_ACCESS(2, hgoldb, j, 0, m * n), hgoldTp);
          LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &m, &m, &n, &alpha, d18gold, &m, hgoldTp, &n, &beta, djdwrgold, &m);
          /* djdwz = djdwz + d11 * h^T */
          LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &m, &m, &n, &alpha, d11gold, &m, hgoldTp, &n, &beta, djdwzgold, &m);
          /* djdwg = djdwg + d10 * (h.r)^T */
          matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, hgoldb, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, rgoldb, j, 0, m * n), d4gold);
          matrix_transpose(m, n, d4gold, hgoldTp);
          LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &m, &m, &n, &alpha, d10gold, &m, hgoldTp, &n, &beta, djdwggold, &m);
          /* djdur = djdur + d18 * x^T */
          matrix_transpose(k, n, &LIBXSMM_VLA_ACCESS(2, xgold, j, 0, k * n), xgoldTp);
          LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &m, &k, &n, &alpha, d18gold, &m, xgoldTp, &n, &beta, djdurgold, &m);
          /* djduz = djduz + d11 * x^T */
          LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &m, &k, &n, &alpha, d11gold, &m, xgoldTp, &n, &beta, djduzgold, &m);
          /* djdug = djdug + d10 * x^T */
          LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &m, &k, &n, &alpha, d10gold, &m, xgoldTp, &n, &beta, djduggold, &m);
          /* djdbr = djdbr + d18 */
          matrix_add(m * n, djdbrgold, d18gold, djdbrgold);
          /* djdbz = djdbz + d11 */
          matrix_add(m * n, djdbzgold, d11gold, djdbzgold);
          /* djdbg = djdbg + d10 */
          matrix_add(m * n, djdbggold, d10gold, djdbggold);
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
      libxsmm_layout = libxsmm_dnn_grucell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_GRU_REGULAR_HIDDEN_STATE, &status ); CHKERR_LIBXSMM_DNN( status );
      libxsmm_hidden_state = libxsmm_dnn_link_tensor( libxsmm_layout, ht, &status ); CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

      libxsmm_layout = libxsmm_dnn_grucell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_GRU_GRADIENT_INPUT, &status ); CHKERR_LIBXSMM_DNN( status );
      libxsmm_dinput = libxsmm_dnn_link_tensor( libxsmm_layout, djdxt, &status ); CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

      libxsmm_layout = libxsmm_dnn_grucell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_GRU_GRADIENT_WEIGHT_R, &status ); CHKERR_LIBXSMM_DNN( status );
      libxsmm_dweight_r = libxsmm_dnn_link_tensor( libxsmm_layout, djdur, &status ); CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

      libxsmm_layout = libxsmm_dnn_grucell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_GRU_GRADIENT_WEIGHT_Z, &status ); CHKERR_LIBXSMM_DNN( status );
      libxsmm_dweight_z = libxsmm_dnn_link_tensor( libxsmm_layout, djduz, &status ); CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

      libxsmm_layout = libxsmm_dnn_grucell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_GRU_GRADIENT_WEIGHT_G, &status ); CHKERR_LIBXSMM_DNN( status );
      libxsmm_dweight_g = libxsmm_dnn_link_tensor( libxsmm_layout, djdug, &status ); CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

      libxsmm_layout = libxsmm_dnn_grucell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_GRU_GRADIENT_RECUR_WEIGHT_R, &status ); CHKERR_LIBXSMM_DNN( status );
      libxsmm_drecur_weight_r = libxsmm_dnn_link_tensor( libxsmm_layout, djdwr, &status ); CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

      libxsmm_layout = libxsmm_dnn_grucell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_GRU_GRADIENT_RECUR_WEIGHT_Z, &status ); CHKERR_LIBXSMM_DNN( status );
      libxsmm_drecur_weight_z = libxsmm_dnn_link_tensor( libxsmm_layout, djdwz, &status ); CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

      libxsmm_layout = libxsmm_dnn_grucell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_GRU_GRADIENT_RECUR_WEIGHT_G, &status ); CHKERR_LIBXSMM_DNN( status );
      libxsmm_drecur_weight_g = libxsmm_dnn_link_tensor( libxsmm_layout, djdwg, &status ); CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

      libxsmm_layout = libxsmm_dnn_grucell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_GRU_GRADIENT_HIDDEN_STATE, &status ); CHKERR_LIBXSMM_DNN( status );
      libxsmm_dhidden_state = libxsmm_dnn_link_tensor( libxsmm_layout, djdht, &status ); CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

      libxsmm_layout = libxsmm_dnn_grucell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_GRU_GRADIENT_BIAS_R, &status ); CHKERR_LIBXSMM_DNN( status );
      libxsmm_dbias_r = libxsmm_dnn_link_tensor( libxsmm_layout, djdbr, &status ); CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

      libxsmm_layout = libxsmm_dnn_grucell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_GRU_GRADIENT_BIAS_Z, &status ); CHKERR_LIBXSMM_DNN( status );
      libxsmm_dbias_z = libxsmm_dnn_link_tensor( libxsmm_layout, djdbz, &status ); CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

      libxsmm_layout = libxsmm_dnn_grucell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_GRU_GRADIENT_BIAS_G, &status ); CHKERR_LIBXSMM_DNN( status );
      libxsmm_dbias_g = libxsmm_dnn_link_tensor( libxsmm_layout, djdbg, &status ); CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
    }

    /* copy in data to LIBXSMM format */
    if (pass == 0) {
      CHKERR_LIBXSMM_DNN( libxsmm_bgemm_copyin_a(handleux, urgold, &m, ur) );
      CHKERR_LIBXSMM_DNN( libxsmm_bgemm_copyin_a(handleux, uzgold, &m, uz) );
      CHKERR_LIBXSMM_DNN( libxsmm_bgemm_copyin_a(handleux, uggold, &m, ug) );
      for (it = 0; it < t; ++it) {
        CHKERR_LIBXSMM_DNN( libxsmm_bgemm_copyin_b(handleux, &LIBXSMM_VLA_ACCESS(2, xgold, it, 0, k * n), &k, &LIBXSMM_VLA_ACCESS(2, x, it, 0, k * n)) );
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
      matrix_transpose(m, k, urgold, urgoldTp);
      CHKERR_LIBXSMM_DNN( libxsmm_bgemm_copyin_a(handleux, urgoldTp, &k, ur) );
      matrix_transpose(m, k, uzgold, uzgoldTp);
      CHKERR_LIBXSMM_DNN( libxsmm_bgemm_copyin_a(handleux, uzgoldTp, &k, uz) );
      matrix_transpose(m, k, uggold, uggoldTp);
      CHKERR_LIBXSMM_DNN( libxsmm_bgemm_copyin_a(handleux, uggoldTp, &k, ug) );
      for (it = 0; it < t; ++it) {
        CHKERR_LIBXSMM_DNN( libxsmm_bgemm_copyin_b(handlewd, &LIBXSMM_VLA_ACCESS(2, hgoldb, it, 0, m * n), &m, &LIBXSMM_VLA_ACCESS(2, hb, it, 0, m * n)) );
        matrix_transpose(k, n, &LIBXSMM_VLA_ACCESS(2, xgold, it, 0, k * n), xgoldTp);
        CHKERR_LIBXSMM_DNN( libxsmm_bgemm_copyin_b(handlett, xgoldTp, &k, &LIBXSMM_VLA_ACCESS(2, x, it, 0, k * n)) );
        CHKERR_LIBXSMM_DNN( libxsmm_bgemm_copyin_b(handlewd, &LIBXSMM_VLA_ACCESS(2, djdhgold, it, 0, m * n), &m, &LIBXSMM_VLA_ACCESS(2, djdh, it, 0, m * n)) );
      }
      matrix_transpose(m, m, wrgold, wrgoldTp);
      CHKERR_LIBXSMM_DNN( libxsmm_bgemm_copyin_a(handlewd, wrgoldTp, &m, wr) );
      matrix_transpose(m, m, wzgold, wzgoldTp);
      CHKERR_LIBXSMM_DNN( libxsmm_bgemm_copyin_a(handlewd, wzgoldTp, &m, wz) );
      matrix_transpose(m, m, wggold, wggoldTp);
      CHKERR_LIBXSMM_DNN( libxsmm_bgemm_copyin_a(handlewd, wggoldTp, &m, wg) );
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

    if (pass != 0) {
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_grucell_bind_tensor( libxsmm_handle, libxsmm_dinput, LIBXSMM_DNN_GRU_GRADIENT_INPUT ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_grucell_bind_tensor( libxsmm_handle, libxsmm_dhidden_state, LIBXSMM_DNN_GRU_GRADIENT_HIDDEN_STATE ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_grucell_bind_tensor( libxsmm_handle, libxsmm_dweight_r, LIBXSMM_DNN_GRU_GRADIENT_WEIGHT_R ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_grucell_bind_tensor( libxsmm_handle, libxsmm_dweight_z, LIBXSMM_DNN_GRU_GRADIENT_WEIGHT_Z ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_grucell_bind_tensor( libxsmm_handle, libxsmm_dweight_g, LIBXSMM_DNN_GRU_GRADIENT_WEIGHT_G ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_grucell_bind_tensor( libxsmm_handle, libxsmm_drecur_weight_r, LIBXSMM_DNN_GRU_GRADIENT_RECUR_WEIGHT_R ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_grucell_bind_tensor( libxsmm_handle, libxsmm_drecur_weight_z, LIBXSMM_DNN_GRU_GRADIENT_RECUR_WEIGHT_Z ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_grucell_bind_tensor( libxsmm_handle, libxsmm_drecur_weight_g, LIBXSMM_DNN_GRU_GRADIENT_RECUR_WEIGHT_G ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_grucell_bind_tensor( libxsmm_handle, libxsmm_dbias_r, LIBXSMM_DNN_GRU_GRADIENT_BIAS_R ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_grucell_bind_tensor( libxsmm_handle, libxsmm_dbias_z, LIBXSMM_DNN_GRU_GRADIENT_BIAS_Z ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_grucell_bind_tensor( libxsmm_handle, libxsmm_dbias_g, LIBXSMM_DNN_GRU_GRADIENT_BIAS_G ) );
    }

    /* let's allocate and bind scratch */
    if (pass == 0) {
      scratch_size = libxsmm_dnn_grucell_get_scratch_size( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_FWD, &status );
      CHKERR_LIBXSMM_DNN( status );
      scratch = libxsmm_aligned_malloc( scratch_size, 2097152 );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_grucell_bind_scratch( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_FWD, scratch ) );
    } else {
      scratch_size = libxsmm_dnn_grucell_get_scratch_size( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_ALL, &status );
      CHKERR_LIBXSMM_DNN( status );
      scratch = libxsmm_aligned_malloc( scratch_size, 2097152 );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_grucell_bind_scratch( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_ALL, scratch ) );
    }
    zero_buf( (float*)scratch, scratch_size/4 );

    /* let's allocate and bind internalstate */
    if (pass == 0) {
      internalstate_size = libxsmm_dnn_grucell_get_internalstate_size( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_FWD, &status );
      CHKERR_LIBXSMM_DNN( status );
      internalstate = libxsmm_aligned_malloc( internalstate_size, 2097152 );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_grucell_bind_internalstate( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_FWD, internalstate ) );
    } else {
      internalstate_size = libxsmm_dnn_grucell_get_internalstate_size( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_ALL, &status );
      CHKERR_LIBXSMM_DNN( status );
      internalstate = libxsmm_aligned_malloc( internalstate_size, 2097152 );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_grucell_bind_internalstate( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_ALL, internalstate ) );
    }
    zero_buf( (float*)internalstate, internalstate_size/4 );
    if (pass != 0) {
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_grucell_assign_internalstate( libxsmm_handle, rgoldt, zgoldt, ggoldt ) );
    }

    if ((pass == 0) && LIBXSMM_NEQ(0, check)) {
      printf("##########################################\n");
      printf("#   Correctness - FWD (custom-Storage)   #\n");
      printf("##########################################\n");
      /* run LIBXSMM GRU */
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

    if ( (pass == 1) && LIBXSMM_NEQ(0, check) ) {
      printf("##########################################\n");
      printf("#   Correctness - BWD (custom-Storage)   #\n");
      printf("##########################################\n");
      /* run LIBXSMM GRU */
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
        CHKERR_LIBXSMM_DNN( libxsmm_bgemm_copyout_c(handleux, &LIBXSMM_VLA_ACCESS(2, djdx, i, 0, k * n), &k, &LIBXSMM_VLA_ACCESS(2, djdxtest, i, 0, k * n)) );
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
      /* run LIBXSMM GRU */
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
      LIBXSMM_VLA_DECL(2, float, djdw3test, djdwtest, m * m);
      LIBXSMM_VLA_DECL(2, float, djdu3test, djdutest, m * k);
      LIBXSMM_VLA_DECL(2, float, djdb3test, djdbtest, m * n);
      CHKERR_LIBXSMM_DNN( libxsmm_bgemm_copyout_c(handlett, djdur, &m, &LIBXSMM_VLA_ACCESS(2, djdu3test, 0, 0, m * k)) );
      CHKERR_LIBXSMM_DNN( libxsmm_bgemm_copyout_c(handlett, djduz, &m, &LIBXSMM_VLA_ACCESS(2, djdu3test, 1, 0, m * k)) );
      CHKERR_LIBXSMM_DNN( libxsmm_bgemm_copyout_c(handlett, djdug, &m, &LIBXSMM_VLA_ACCESS(2, djdu3test, 2, 0, m * k)) );
      CHKERR_LIBXSMM_DNN( libxsmm_bgemm_copyout_c(handlewh, djdwr, &m, &LIBXSMM_VLA_ACCESS(2, djdw3test, 0, 0, m * m)) );
      CHKERR_LIBXSMM_DNN( libxsmm_bgemm_copyout_c(handlewh, djdwz, &m, &LIBXSMM_VLA_ACCESS(2, djdw3test, 1, 0, m * m)) );
      CHKERR_LIBXSMM_DNN( libxsmm_bgemm_copyout_c(handlewh, djdwg, &m, &LIBXSMM_VLA_ACCESS(2, djdw3test, 2, 0, m * m)) );
      CHKERR_LIBXSMM_DNN( libxsmm_bgemm_copyout_c(handlewd, djdbr, &m, &LIBXSMM_VLA_ACCESS(2, djdb3test, 0, 0, m * n)) );
      CHKERR_LIBXSMM_DNN( libxsmm_bgemm_copyout_c(handlewd, djdbz, &m, &LIBXSMM_VLA_ACCESS(2, djdb3test, 1, 0, m * n)) );
      CHKERR_LIBXSMM_DNN( libxsmm_bgemm_copyout_c(handlewd, djdbg, &m, &LIBXSMM_VLA_ACCESS(2, djdb3test, 2, 0, m * n)) );
      LIBXSMM_VLA_DECL(2, float, djdu3, djdugold3, m * k);
      LIBXSMM_VLA_DECL(2, float, djdw3, djdwgold3, m * m);
      LIBXSMM_VLA_DECL(2, float, djdb3, djdbgold3, m * n);
      matrix_copy(m * k, djdurgold, &LIBXSMM_VLA_ACCESS(2, djdu3, 0, 0, m * k));
      matrix_copy(m * k, djduzgold, &LIBXSMM_VLA_ACCESS(2, djdu3, 1, 0, m * k));
      matrix_copy(m * k, djduggold, &LIBXSMM_VLA_ACCESS(2, djdu3, 2, 0, m * k));
      matrix_copy(m * m, djdwrgold, &LIBXSMM_VLA_ACCESS(2, djdw3, 0, 0, m * m));
      matrix_copy(m * m, djdwzgold, &LIBXSMM_VLA_ACCESS(2, djdw3, 1, 0, m * m));
      matrix_copy(m * m, djdwggold, &LIBXSMM_VLA_ACCESS(2, djdw3, 2, 0, m * m));
      matrix_copy(m * n, djdbrgold, &LIBXSMM_VLA_ACCESS(2, djdb3, 0, 0, m * n));
      matrix_copy(m * n, djdbzgold, &LIBXSMM_VLA_ACCESS(2, djdb3, 1, 0, m * n));
      matrix_copy(m * n, djdbggold, &LIBXSMM_VLA_ACCESS(2, djdb3, 2, 0, m * n));

      /* compare */
      libxsmm_matdiff(LIBXSMM_DATATYPE_F32, m*k*3, 1, djdugold3, djdutest, 0, 0, &norms_upd_w);
      printf("Delta weight\n");
      printf("L1 reference  : %.25g\n", norms_upd_w.l1_ref);
      printf("L1 test       : %.25g\n", norms_upd_w.l1_tst);
      printf("L2 abs.error  : %.24f\n", norms_upd_w.l2_abs);
      printf("L2 rel.error  : %.24f\n", norms_upd_w.l2_rel);
      printf("Linf abs.error: %.24f\n", norms_upd_w.linf_abs);
      printf("Linf rel.error: %.24f\n", norms_upd_w.linf_rel);
      printf("Check-norm    : %.24f\n", norms_upd_w.normf_rel);
      libxsmm_matdiff_reduce(&diff, &norms_upd_w);

      libxsmm_matdiff(LIBXSMM_DATATYPE_F32, m*m*3, 1, djdwgold3, djdwtest, 0, 0, &norms_upd_r);
      printf("Delta recurrent weight\n");
      printf("L1 reference  : %.25g\n", norms_upd_r.l1_ref);
      printf("L1 test       : %.25g\n", norms_upd_r.l1_tst);
      printf("L2 abs.error  : %.24f\n", norms_upd_r.l2_abs);
      printf("L2 rel.error  : %.24f\n", norms_upd_r.l2_rel);
      printf("Linf abs.error: %.24f\n", norms_upd_r.linf_abs);
      printf("Linf rel.error: %.24f\n", norms_upd_r.linf_rel);
      printf("Check-norm    : %.24f\n", norms_upd_r.normf_rel);
      libxsmm_matdiff_reduce(&diff, &norms_upd_r);

      libxsmm_matdiff(LIBXSMM_DATATYPE_F32, m*n*3, 1, djdbgold3, djdbtest, 0, 0, &norms_upd_b);
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
      /* run LIBXSMM GRU */
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
        CHKERR_LIBXSMM_DNN( libxsmm_bgemm_copyout_c(handleux, &LIBXSMM_VLA_ACCESS(2, djdx, i, 0, k * n), &k, &LIBXSMM_VLA_ACCESS(2, djdxtest, i, 0, k * n)) );
      }
      LIBXSMM_VLA_DECL(2, float, djdu3test, djdutest, m * k);
      LIBXSMM_VLA_DECL(2, float, djdw3test, djdwtest, m * m);
      LIBXSMM_VLA_DECL(2, float, djdb3test, djdbtest, m * n);
      CHKERR_LIBXSMM_DNN( libxsmm_bgemm_copyout_c(handlett, djdur, &m, &LIBXSMM_VLA_ACCESS(2, djdu3test, 0, 0, m * k)) );
      CHKERR_LIBXSMM_DNN( libxsmm_bgemm_copyout_c(handlett, djduz, &m, &LIBXSMM_VLA_ACCESS(2, djdu3test, 1, 0, m * k)) );
      CHKERR_LIBXSMM_DNN( libxsmm_bgemm_copyout_c(handlett, djdug, &m, &LIBXSMM_VLA_ACCESS(2, djdu3test, 2, 0, m * k)) );
      CHKERR_LIBXSMM_DNN( libxsmm_bgemm_copyout_c(handlewh, djdwr, &m, &LIBXSMM_VLA_ACCESS(2, djdw3test, 0, 0, m * m)) );
      CHKERR_LIBXSMM_DNN( libxsmm_bgemm_copyout_c(handlewh, djdwz, &m, &LIBXSMM_VLA_ACCESS(2, djdw3test, 1, 0, m * m)) );
      CHKERR_LIBXSMM_DNN( libxsmm_bgemm_copyout_c(handlewh, djdwg, &m, &LIBXSMM_VLA_ACCESS(2, djdw3test, 2, 0, m * m)) );
      CHKERR_LIBXSMM_DNN( libxsmm_bgemm_copyout_c(handlewd, djdbr, &m, &LIBXSMM_VLA_ACCESS(2, djdb3test, 0, 0, m * n)) );
      CHKERR_LIBXSMM_DNN( libxsmm_bgemm_copyout_c(handlewd, djdbz, &m, &LIBXSMM_VLA_ACCESS(2, djdb3test, 1, 0, m * n)) );
      CHKERR_LIBXSMM_DNN( libxsmm_bgemm_copyout_c(handlewd, djdbg, &m, &LIBXSMM_VLA_ACCESS(2, djdb3test, 2, 0, m * n)) );
      LIBXSMM_VLA_DECL(2, float, djdu3, djdugold3, m * k);
      LIBXSMM_VLA_DECL(2, float, djdw3, djdwgold3, m * m);
      LIBXSMM_VLA_DECL(2, float, djdb3, djdbgold3, m * n);
      matrix_copy(m * k, djdurgold, &LIBXSMM_VLA_ACCESS(2, djdu3, 0, 0, m * k));
      matrix_copy(m * k, djduzgold, &LIBXSMM_VLA_ACCESS(2, djdu3, 1, 0, m * k));
      matrix_copy(m * k, djduggold, &LIBXSMM_VLA_ACCESS(2, djdu3, 2, 0, m * k));
      matrix_copy(m * m, djdwrgold, &LIBXSMM_VLA_ACCESS(2, djdw3, 0, 0, m * m));
      matrix_copy(m * m, djdwzgold, &LIBXSMM_VLA_ACCESS(2, djdw3, 1, 0, m * m));
      matrix_copy(m * m, djdwggold, &LIBXSMM_VLA_ACCESS(2, djdw3, 2, 0, m * m));
      matrix_copy(m * n, djdbrgold, &LIBXSMM_VLA_ACCESS(2, djdb3, 0, 0, m * n));
      matrix_copy(m * n, djdbzgold, &LIBXSMM_VLA_ACCESS(2, djdb3, 1, 0, m * n));
      matrix_copy(m * n, djdbggold, &LIBXSMM_VLA_ACCESS(2, djdb3, 2, 0, m * n));

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

      libxsmm_matdiff(LIBXSMM_DATATYPE_F32, m*k*3, 1, djdugold3, djdutest, 0, 0, &norms_upd_w);
      printf("Delta weight\n");
      printf("L1 reference  : %.25g\n", norms_upd_w.l1_ref);
      printf("L1 test       : %.25g\n", norms_upd_w.l1_tst);
      printf("L2 abs.error  : %.24f\n", norms_upd_w.l2_abs);
      printf("L2 rel.error  : %.24f\n", norms_upd_w.l2_rel);
      printf("Linf abs.error: %.24f\n", norms_upd_w.linf_abs);
      printf("Linf rel.error: %.24f\n", norms_upd_w.linf_rel);
      printf("Check-norm    : %.24f\n", norms_upd_w.normf_rel);
      libxsmm_matdiff_reduce(&diff, &norms_upd_w);

      libxsmm_matdiff(LIBXSMM_DATATYPE_F32, m*m*3, 1, djdwgold3, djdwtest, 0, 0, &norms_upd_r);
      printf("Delta recurrent weight\n");
      printf("L1 reference  : %.25g\n", norms_upd_r.l1_ref);
      printf("L1 test       : %.25g\n", norms_upd_r.l1_tst);
      printf("L2 abs.error  : %.24f\n", norms_upd_r.l2_abs);
      printf("L2 rel.error  : %.24f\n", norms_upd_r.l2_rel);
      printf("Linf abs.error: %.24f\n", norms_upd_r.linf_abs);
      printf("Linf rel.error: %.24f\n", norms_upd_r.linf_rel);
      printf("Check-norm    : %.24f\n", norms_upd_r.normf_rel);
      libxsmm_matdiff_reduce(&diff, &norms_upd_r);

      libxsmm_matdiff(LIBXSMM_DATATYPE_F32, m*n*3, 1, djdbgold3, djdbtest, 0, 0, &norms_upd_b);
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
      /* run LIBXSMM GRU for performance */
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

    if ( pass == 1 ) {
      printf("##########################################\n");
      printf("#   Performance - BWD (custom-Storage)   #\n");
      printf("##########################################\n");
      /* run LIBXSMM GRU for performance */
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
      flops = m * n; /* d3 = djdh + d23 (delta) */
      flops += 2.0 *m * n; /* d4 = (1 - z).d3 */
      flops += m * n; /* d5 = d3.h */
      flops += m * n; /* d6 = -d5 */
      flops += m * n; /* d7 = d3.g */
      flops += m * n; /* d8 = d3.z */
      flops += m * n; /* d9 = d7 + d8 */
      flops += 3.0 * m * n; /* d10 = d8.tanh'(g) */
      flops += 3.0 * m * n; /* d11 = d9.sig'(z) */
      flops += (2.0 * m * m * n + m * m) ; /* d13 = Wg^T * d10 (including transpose) */
      flops += (2.0 * m * m * n + m * m) ; /* d15 = Wz^T * d11 (including transpose) */
      flops += m * n; /* d16 = d13.z */
      flops += m * n; /* d17 = d13.r */
      flops += 3.0 * m * n; /* d18 = d16.sig'(r) */
      flops += m * n; /* d19 = d17 + d4 */
      flops += (2.0 * m * m * n + m * m) ; /* d21 = Wr^T * d18 (including transpose) */
      flops += m * n; /* d22 = d21 + d15 */
      flops += m * n; /* d23 = d19 + d22 */
      flops += (2.0 * m * k * n + m * k) ; /* d12 = Ug^T * d10 (including transpose) */
      flops += (2.0 * m * k * n + m * k) ; /* d14 = Uz^T * d11 (including transpose) */
      flops += (2.0 * m * k * n + m * k) ; /* d20 = Ur^T * d18 (including transpose) */
      flops += 2.0 * m * n; /* djdx = d12 + d14 + d20 */
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
      /* run LIBXSMM GRU for performance */
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
      flops = m * n; /* d3 = djdh + d23 (delta) */
      flops += 2.0 *m * n; /* d4 = (1 - z).d3 */
      flops += m * n; /* d5 = d3.h */
      flops += m * n; /* d6 = -d5 */
      flops += m * n; /* d7 = d3.g */
      flops += m * n; /* d8 = d3.z */
      flops += m * n; /* d9 = d7 + d8 */
      flops += 3.0 * m * n; /* d10 = d8.tanh'(g) */
      flops += 3.0 * m * n; /* d11 = d9.sig'(z) */
      flops += (2.0 * m * m * n + m * m) ; /* d13 = Wg^T * d10 (including transpose) */
      flops += (2.0 * m * m * n + m * m) ; /* d15 = Wz^T * d11 (including transpose) */
      flops += m * n; /* d16 = d13.z */
      flops += m * n; /* d17 = d13.r */
      flops += 3.0 * m * n; /* d18 = d16.sig'(r) */
      flops += m * n; /* d19 = d17 + d4 */
      flops += (2.0 * m * m * n + m * m) ; /* d21 = Wr^T * d18 (including transpose) */
      flops += m * n; /* d22 = d21 + d15 */
      flops += m * n; /* d23 = d19 + d22 */
      flops += (2.0 * m * n * m + m * n + m * m) ; /* djdwr = djdwr + d18 * h^T */
      flops += (2.0 * m * n * m + m * n + m * m) ; /* djdwz = djdwz + d11 * h^T */
      flops += (2.0 * m * n * m + 2.0 * m * n + m * m) ; /* djdwg = djdwg + d10 * (h.r)^T */
      flops += (2.0 * m * n * k + k * n + m * k) ; /* djdur = djdur + d18 * x^T */
      flops += (2.0 * m * n * k + k * n + m * k) ; /* djduz = djduz + d11 * x^T */
      flops += (2.0 * m * n * k + k * n + m * k) ; /* djdug = djdug + d10 * x^T */
      flops += m * n; /* djdbr = djdbr + d18 */
      flops += m * n; /* djdbz = djdbz + d11 */
      flops += m * n; /* djdbg = djdbg + d10 */
      flops *= t; /* for t time steps */
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
      /* run LIBXSMM GRU for performance */
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
      flops = m * n; /* d3 = djdh + d23 (delta) */
      flops += 2.0 *m * n; /* d4 = (1 - z).d3 */
      flops += m * n; /* d5 = d3.h */
      flops += m * n; /* d6 = -d5 */
      flops += m * n; /* d7 = d3.g */
      flops += m * n; /* d8 = d3.z */
      flops += m * n; /* d9 = d7 + d8 */
      flops += 3.0 * m * n; /* d10 = d8.tanh'(g) */
      flops += 3.0 * m * n; /* d11 = d9.sig'(z) */
      flops += (2.0 * m * m * n + m * m) ; /* d13 = Wg^T * d10 (including transpose) */
      flops += (2.0 * m * m * n + m * m) ; /* d15 = Wz^T * d11 (including transpose) */
      flops += m * n; /* d16 = d13.z */
      flops += m * n; /* d17 = d13.r */
      flops += 3.0 * m * n; /* d18 = d16.sig'(r) */
      flops += m * n; /* d19 = d17 + d4 */
      flops += (2.0 * m * m * n + m * m) ; /* d21 = Wr^T * d18 (including transpose) */
      flops += m * n; /* d22 = d21 + d15 */
      flops += m * n; /* d23 = d19 + d22 */
      flops += (2.0 * m * k * n + m * k) ; /* d12 = Ug^T * d10 (including transpose) */
      flops += (2.0 * m * k * n + m * k) ; /* d14 = Uz^T * d11 (including transpose) */
      flops += (2.0 * m * k * n + m * k) ; /* d20 = Ur^T * d18 (including transpose) */
      flops += 2.0 * m * n; /* djdx = d12 + d14 + d20 */
      flops += (2.0 * m * n * m + m * n + m * m) ; /* djdwr = djdwr + d18 * h^T */
      flops += (2.0 * m * n * m + m * n + m * m) ; /* djdwz = djdwz + d11 * h^T */
      flops += (2.0 * m * n * m + 2.0 * m * n + m * m) ; /* djdwg = djdwg + d10 * (h.r)^T */
      flops += (2.0 * m * n * k + k * n + m * k) ; /* djdur = djdur + d18 * x^T */
      flops += (2.0 * m * n * k + k * n + m * k) ; /* djduz = djduz + d11 * x^T */
      flops += (2.0 * m * n * k + k * n + m * k) ; /* djdug = djdug + d10 * x^T */
      flops += m * n; /* djdbr = djdbr + d18 */
      flops += m * n; /* djdbz = djdbz + d11 */
      flops += m * n; /* djdbg = djdbg + d10 */
      flops *= t; /* for t time steps */
      flops *= iters;

      printf("GFLOP  = %.5g\n", flops*1e-9/(double)iters);
      printf("bp+wu time = %.5g\n", ((double)(l_total/iters)));
      printf("GFLOPS  = %.5g\n", (flops*1e-9)/l_total);

      printf("PERFDUMP,BP+WU,%s,%i,%i,%i,%i,%i,%i,%i,%i,%.5g,%.5g\n", LIBXSMM_VERSION, nThreads, m, n, k, t, bm, bn, bk, ((double)(l_total/iters)), (flops*1e-9)/l_total);
    }

    /* clean-up */
    if (pass == 0) {
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_grucell_release_scratch( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_FWD ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_grucell_release_internalstate( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_FWD ) );
    } else {
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_grucell_release_scratch( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_ALL ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_grucell_release_internalstate( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_ALL ) );
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
    if (pass != 0) {
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_grucell_release_tensor( libxsmm_handle, LIBXSMM_DNN_GRU_GRADIENT_INPUT ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_grucell_release_tensor( libxsmm_handle, LIBXSMM_DNN_GRU_GRADIENT_HIDDEN_STATE ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_grucell_release_tensor( libxsmm_handle, LIBXSMM_DNN_GRU_GRADIENT_WEIGHT_R ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_grucell_release_tensor( libxsmm_handle, LIBXSMM_DNN_GRU_GRADIENT_WEIGHT_Z ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_grucell_release_tensor( libxsmm_handle, LIBXSMM_DNN_GRU_GRADIENT_WEIGHT_G ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_grucell_release_tensor( libxsmm_handle, LIBXSMM_DNN_GRU_GRADIENT_RECUR_WEIGHT_R ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_grucell_release_tensor( libxsmm_handle, LIBXSMM_DNN_GRU_GRADIENT_RECUR_WEIGHT_Z ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_grucell_release_tensor( libxsmm_handle, LIBXSMM_DNN_GRU_GRADIENT_RECUR_WEIGHT_G ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_grucell_release_tensor( libxsmm_handle, LIBXSMM_DNN_GRU_GRADIENT_BIAS_R ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_grucell_release_tensor( libxsmm_handle, LIBXSMM_DNN_GRU_GRADIENT_BIAS_Z ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_grucell_release_tensor( libxsmm_handle, LIBXSMM_DNN_GRU_GRADIENT_BIAS_G ) );
    }
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
    if (pass != 0) {
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_dinput ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_dhidden_state ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_dweight_r ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_dweight_z ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_dweight_g ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_drecur_weight_r ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_drecur_weight_z ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_drecur_weight_g ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_dbias_r ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_dbias_z ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_dbias_g ) );
    }
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
    libxsmm_free(wrgold);
    libxsmm_free(wzgold);
    libxsmm_free(wggold);
    libxsmm_free(xgoldt);
    libxsmm_free(urgold);
    libxsmm_free(uzgold);
    libxsmm_free(uggold);
    libxsmm_free(hgoldt);
    libxsmm_free(rgoldt);
    libxsmm_free(zgoldt);
    libxsmm_free(ggoldt);
    libxsmm_free(d3gold);
    libxsmm_free(d4gold);
    libxsmm_free(d5gold);
    libxsmm_free(d6gold);
    libxsmm_free(d7gold);
    libxsmm_free(d8gold);
    libxsmm_free(d9gold);
    libxsmm_free(d10gold);
    libxsmm_free(d11gold);
    libxsmm_free(d12gold);
    libxsmm_free(d13gold);
    libxsmm_free(d14gold);
    libxsmm_free(d15gold);
    libxsmm_free(d16gold);
    libxsmm_free(d17gold);
    libxsmm_free(d18gold);
    libxsmm_free(d19gold);
    libxsmm_free(d20gold);
    libxsmm_free(d21gold);
    libxsmm_free(d22gold);
    libxsmm_free(d23gold);
    libxsmm_free(djdhgoldt);
    libxsmm_free(djdxgoldt);
    libxsmm_free(djdwrgold);
    libxsmm_free(djdwzgold);
    libxsmm_free(djdwggold);
    libxsmm_free(djdurgold);
    libxsmm_free(djduzgold);
    libxsmm_free(djduggold);
    libxsmm_free(djdbrgold);
    libxsmm_free(djdbzgold);
    libxsmm_free(djdbggold);
    libxsmm_free(wrgoldTp);
    libxsmm_free(wzgoldTp);
    libxsmm_free(wggoldTp);
    libxsmm_free(urgoldTp);
    libxsmm_free(uzgoldTp);
    libxsmm_free(uggoldTp);
    libxsmm_free(xgoldTp);
    libxsmm_free(hgoldTp);
    libxsmm_free(wr);
    libxsmm_free(wz);
    libxsmm_free(wg);
    libxsmm_free(xt);
    libxsmm_free(ur);
    libxsmm_free(uz);
    libxsmm_free(ug);
    libxsmm_free(ht);
    libxsmm_free(br);
    libxsmm_free(bz);
    libxsmm_free(bg);
    libxsmm_free(djdht);
    libxsmm_free(djdxt);
    libxsmm_free(djdwr);
    libxsmm_free(djdwz);
    libxsmm_free(djdwg);
    libxsmm_free(djdur);
    libxsmm_free(djduz);
    libxsmm_free(djdug);
    libxsmm_free(djdbr);
    libxsmm_free(djdbz);
    libxsmm_free(djdbg);
    libxsmm_free(djdxtestt);
    libxsmm_free(djdwtest);
    libxsmm_free(djdutest);
    libxsmm_free(djdbtest);
    libxsmm_free(djdugold3);
    libxsmm_free(djdbgold3);
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

