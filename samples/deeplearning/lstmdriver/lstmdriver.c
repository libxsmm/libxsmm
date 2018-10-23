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
    dst[i] = 1.0f / (1.0f + exp_value);
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
  float *wigold, *wfgold, *wogold, *wcgold, *xgoldt, *rigold, *rfgold, *rogold, *rcgold, *hgoldt, *bigold, *bfgold, *bogold, *bcgold;
  float *cspgold, *hpgold/*, *dcspgold, *dhpgold*/;
  float *igoldt, *fgoldt, *ogoldt, *cgoldt, *dgoldt, *bimgold, *bfmgold, *bomgold, *bcmgold, *doutgoldt;
  float *i1gold, *i2gold, *f1gold, *f2gold, *o1gold, *o2gold, *c1gold, *c2gold, *d1gold, *d2gold, *dhgold;
  float *xt, *csp, *hp, *w, *r, *b, *cst, *ht;
  float *it, *ft, *ot, *cit, *cot;
  float *dxt, *dcspt, *dhpt, *dw, *dr, *db, *dcs, *dht;
  float *i3gold, *f3gold, *d3gold, *d4gold, *deltagoldt;
  float *djdhgoldt, *djddgoldt, *djdigoldt, *djdfgoldt, *djdcgoldt, *djdogoldt, *djdxgoldt;
  float *djdwigold, *djdwfgold, *djdwogold, *djdwcgold, *djdrigold, *djdrfgold, *djdrogold, *djdrcgold;
  float *djdbigold, *djdbfgold, *djdbogold, *djdbcgold, *wgoldTp, *rgoldTp, *xgoldTp, *hgoldTp;
  float *htest, *djdxtestt, *djdwtest, *djdrtest, *djdbtest, *djdwgold4, *djdrgold4, *djdbgold4;

  const char transa = 'N', transb = 'N'; /* no transposes */
  const float alpha = 1, beta = 1, beta0 = 0;
  void *scratch, *internalstate;
  size_t scratch_size = 0, internalstate_size = 0;

  int iters = 10;   /* repetitions of benchmark */
  int pass = 3;     /* pass: 0--FWD, 1--BWD, 2--UPD, 3--BWD+UPD */
  int m = 64;       /* number of outputs */
  int n = 128;      /* size of mini-batch */
  int k = 512;      /* number of inputs */
  int t = 5;        /* number of time steps (> 1) */
  int bm = 64;      /* first blocking factor for m */
  int bn = 64;      /* first blocking factor for n */
  int bk = 64;      /* first blocking factor for k */

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
  int j, l;

  libxsmm_dnn_lstmcell_desc lstmcell_desc;
  libxsmm_dnn_lstmcell* libxsmm_handle;
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
  memset(&norms_fwd, 0, sizeof(norms_fwd));
  memset(&norms_bwd, 0, sizeof(norms_bwd));
  memset(&norms_upd_w, 0, sizeof(norms_upd_w));
  memset(&norms_upd_r, 0, sizeof(norms_upd_r));
  memset(&norms_upd_b, 0, sizeof(norms_upd_b));
  memset(&diff, 0, sizeof(diff));

  if (argc > 1 && !strncmp(argv[1], "-h", 3)) {
    printf("\nUsage: ./lstmdriver [reps] [pass: 0--FWD, 1--BWD, 2--UPD, 3--BWD+UPD] [M] [N] [K] [time_steps > 0]\n\n");
    return 0;
  }
  libxsmm_srand(1);

  /* reading new values from cli */
  j = 1;
  if (argc > j) iters = atoi(argv[j++]);
  if (argc > j) pass  = atoi(argv[j++]);
  if (argc > j) m     = atoi(argv[j++]);
  if (argc > j) n     = atoi(argv[j++]);
  if (argc > j) k     = atoi(argv[j++]);
  if (argc > j) t     = atoi(argv[j++]);
  if (argc > j) bm    = atoi(argv[j++]);
  if (argc > j) bn    = atoi(argv[j++]);
  if (argc > j) bk    = atoi(argv[j++]);

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
  printf("PARAMS: M:%d  N:%d  K:%d  T:%d\n", m, n, k, t);
  printf("PARAMS: ITERS:%d", iters); if (LIBXSMM_FEQ(0, check)) printf("  Threads:%d\n", nThreads); else printf("\n");
  printf("SIZE Weight (MB): %10.2f MiB\n", (double)(m*k*sizeof(float))/(1024.0*1024.0) );
  printf("SIZE Input (MB): %10.2f MiB\n", (double)(k*n*sizeof(float))/(1024.0*1024.0) );
  printf("SIZE Hidden State: %10.2f MiB\n", (double)(m*n*sizeof(float))/(1024.0*1024.0) );

  /* allocate data */
  xgoldt = (float*)libxsmm_aligned_malloc(k*n*t*sizeof(float), 2097152);
  cspgold= (float*)libxsmm_aligned_malloc(m*n*sizeof(float), 2097152);
  hpgold = (float*)libxsmm_aligned_malloc(m*n*sizeof(float), 2097152);
  wigold = (float*)libxsmm_aligned_malloc(m*k*sizeof(float), 2097152);
  wfgold = (float*)libxsmm_aligned_malloc(m*k*sizeof(float), 2097152);
  wogold = (float*)libxsmm_aligned_malloc(m*k*sizeof(float), 2097152);
  wcgold = (float*)libxsmm_aligned_malloc(m*k*sizeof(float), 2097152);
  rigold = (float*)libxsmm_aligned_malloc(m*m*sizeof(float), 2097152);
  rfgold = (float*)libxsmm_aligned_malloc(m*m*sizeof(float), 2097152);
  rogold = (float*)libxsmm_aligned_malloc(m*m*sizeof(float), 2097152);
  rcgold = (float*)libxsmm_aligned_malloc(m*m*sizeof(float), 2097152);
  bigold = (float*)libxsmm_aligned_malloc(m*sizeof(float), 2097152);
  bfgold = (float*)libxsmm_aligned_malloc(m*sizeof(float), 2097152);
  bogold = (float*)libxsmm_aligned_malloc(m*sizeof(float), 2097152);
  bcgold = (float*)libxsmm_aligned_malloc(m*sizeof(float), 2097152);
  hgoldt = (float*)libxsmm_aligned_malloc(m*n*sizeof(float), 2097152);
  bimgold= (float*)libxsmm_aligned_malloc(m*n*sizeof(float), 2097152);
  bfmgold= (float*)libxsmm_aligned_malloc(m*n*sizeof(float), 2097152);
  bomgold= (float*)libxsmm_aligned_malloc(m*n*sizeof(float), 2097152);
  bcmgold= (float*)libxsmm_aligned_malloc(m*n*sizeof(float), 2097152);
  igoldt = (float*)libxsmm_aligned_malloc(m*n*t*sizeof(float), 2097152);
  fgoldt = (float*)libxsmm_aligned_malloc(m*n*t*sizeof(float), 2097152);
  ogoldt = (float*)libxsmm_aligned_malloc(m*n*t*sizeof(float), 2097152);
  cgoldt = (float*)libxsmm_aligned_malloc(m*n*t*sizeof(float), 2097152);
  dgoldt = (float*)libxsmm_aligned_malloc(m*n*t*sizeof(float), 2097152);
  i1gold = (float*)libxsmm_aligned_malloc(m*n*sizeof(float), 2097152);
  i2gold = (float*)libxsmm_aligned_malloc(m*n*sizeof(float), 2097152);
  i3gold = (float*)libxsmm_aligned_malloc(m*n*sizeof(float), 2097152);
  f1gold = (float*)libxsmm_aligned_malloc(m*n*sizeof(float), 2097152);
  f2gold = (float*)libxsmm_aligned_malloc(m*n*sizeof(float), 2097152);
  f3gold = (float*)libxsmm_aligned_malloc(m*n*sizeof(float), 2097152);
  o1gold = (float*)libxsmm_aligned_malloc(m*n*sizeof(float), 2097152);
  o2gold = (float*)libxsmm_aligned_malloc(m*n*sizeof(float), 2097152);
  c1gold = (float*)libxsmm_aligned_malloc(m*n*sizeof(float), 2097152);
  c2gold = (float*)libxsmm_aligned_malloc(m*n*sizeof(float), 2097152);
  d1gold = (float*)libxsmm_aligned_malloc(m*n*sizeof(float), 2097152);
  d2gold = (float*)libxsmm_aligned_malloc(m*n*sizeof(float), 2097152);
  d3gold = (float*)libxsmm_aligned_malloc(m*n*sizeof(float), 2097152);
  d4gold = (float*)libxsmm_aligned_malloc(m*n*sizeof(float), 2097152);
  dhgold = (float*)libxsmm_aligned_malloc(m*n*sizeof(float), 2097152);
  djdhgoldt = (float*)libxsmm_aligned_malloc(m*n*t*sizeof(float), 2097152);
  deltagoldt= (float*)libxsmm_aligned_malloc(m*n*t*sizeof(float), 2097152);
  djddgoldt = (float*)libxsmm_aligned_malloc(m*n*t*sizeof(float), 2097152);
  djdigoldt = (float*)libxsmm_aligned_malloc(m*n*t*sizeof(float), 2097152);
  djdfgoldt = (float*)libxsmm_aligned_malloc(m*n*t*sizeof(float), 2097152);
  djdcgoldt = (float*)libxsmm_aligned_malloc(m*n*t*sizeof(float), 2097152);
  djdogoldt = (float*)libxsmm_aligned_malloc(m*n*t*sizeof(float), 2097152);
  djdxgoldt = (float*)libxsmm_aligned_malloc(k*n*t*sizeof(float), 2097152);
  djdwigold = (float*)libxsmm_aligned_malloc(m*k*sizeof(float), 2097152);
  djdwfgold = (float*)libxsmm_aligned_malloc(m*k*sizeof(float), 2097152);
  djdwogold = (float*)libxsmm_aligned_malloc(m*k*sizeof(float), 2097152);
  djdwcgold = (float*)libxsmm_aligned_malloc(m*k*sizeof(float), 2097152);
  djdrigold = (float*)libxsmm_aligned_malloc(m*m*sizeof(float), 2097152);
  djdrfgold = (float*)libxsmm_aligned_malloc(m*m*sizeof(float), 2097152);
  djdrogold = (float*)libxsmm_aligned_malloc(m*m*sizeof(float), 2097152);
  djdrcgold = (float*)libxsmm_aligned_malloc(m*m*sizeof(float), 2097152);
  djdbigold = (float*)libxsmm_aligned_malloc(m*sizeof(float), 2097152);
  djdbfgold = (float*)libxsmm_aligned_malloc(m*sizeof(float), 2097152);
  djdbogold = (float*)libxsmm_aligned_malloc(m*sizeof(float), 2097152);
  djdbcgold = (float*)libxsmm_aligned_malloc(m*sizeof(float), 2097152);
  wgoldTp = (float*)libxsmm_aligned_malloc(m*k*sizeof(float), 2097152);
  rgoldTp = (float*)libxsmm_aligned_malloc(m*m*sizeof(float), 2097152);
  xgoldTp = (float*)libxsmm_aligned_malloc(k*n*sizeof(float), 2097152);
  hgoldTp = (float*)libxsmm_aligned_malloc(m*n*sizeof(float), 2097152);
  doutgoldt = (float*)libxsmm_aligned_malloc(m*n*t*sizeof(float), 2097152);
  xt     = (float*)libxsmm_aligned_malloc(k*n*t*sizeof(float), 2097152);
  csp    = (float*)libxsmm_aligned_malloc(m*n*sizeof(float), 2097152);
  hp     = (float*)libxsmm_aligned_malloc(m*n*sizeof(float), 2097152);
  w      = (float*)libxsmm_aligned_malloc(m*k*4*sizeof(float), 2097152);
  r      = (float*)libxsmm_aligned_malloc(m*m*4*sizeof(float), 2097152);
  b      = (float*)libxsmm_aligned_malloc(m*4*sizeof(float), 2097152);
  cst    = (float*)libxsmm_aligned_malloc(m*n*t*sizeof(float), 2097152);
  ht     = (float*)libxsmm_aligned_malloc(m*n*t*sizeof(float), 2097152);
  it     = (float*)libxsmm_aligned_malloc(m*n*t*sizeof(float), 2097152);
  ft     = (float*)libxsmm_aligned_malloc(m*n*t*sizeof(float), 2097152);
  ot     = (float*)libxsmm_aligned_malloc(m*n*t*sizeof(float), 2097152);
  cit    = (float*)libxsmm_aligned_malloc(m*n*t*sizeof(float), 2097152);
  cot    = (float*)libxsmm_aligned_malloc(m*n*t*sizeof(float), 2097152);
  dxt  = (float*)libxsmm_aligned_malloc(k*n*t*sizeof(float), 2097152);
  dcspt  = (float*)libxsmm_aligned_malloc(m*n*t*sizeof(float), 2097152);
  dhpt   = (float*)libxsmm_aligned_malloc(m*n*t*sizeof(float), 2097152);
  dw   = (float*)libxsmm_aligned_malloc(m*k*4*sizeof(float), 2097152);
  dr   = (float*)libxsmm_aligned_malloc(m*m*4*sizeof(float), 2097152);
  db   = (float*)libxsmm_aligned_malloc(m*4*sizeof(float), 2097152);
  dcs    = (float*)libxsmm_aligned_malloc(m*n*sizeof(float), 2097152);
  dht  = (float*)libxsmm_aligned_malloc(m*n*t*sizeof(float), 2097152);
  htest  = (float*)libxsmm_aligned_malloc(m*n*t*sizeof(float), 2097152);
  djdxtestt  = (float*)libxsmm_aligned_malloc(k*n*sizeof(float)*t, 2097152);
  djdwtest   = (float*)libxsmm_aligned_malloc(m*k*sizeof(float)*4, 2097152);
  djdrtest   = (float*)libxsmm_aligned_malloc(m*m*sizeof(float)*4, 2097152);
  djdbtest   = (float*)libxsmm_aligned_malloc(m*sizeof(float)*4, 2097152);
  djdwgold4  = (float*)libxsmm_aligned_malloc(m*k*sizeof(float)*4, 2097152);
  djdrgold4  = (float*)libxsmm_aligned_malloc(m*m*sizeof(float)*4, 2097152);
  djdbgold4  = (float*)libxsmm_aligned_malloc(m*sizeof(float)*4, 2097152);
  LIBXSMM_VLA_DECL(2, float, xgold, xgoldt, k * n);
  LIBXSMM_VLA_DECL(2, float, igold, igoldt, m * n);
  LIBXSMM_VLA_DECL(2, float, fgold, fgoldt, m * n);
  LIBXSMM_VLA_DECL(2, float, ogold, ogoldt, m * n);
  LIBXSMM_VLA_DECL(2, float, cgold, cgoldt, m * n);
  LIBXSMM_VLA_DECL(2, float, dgold, dgoldt, m * n);
  LIBXSMM_VLA_DECL(2, float, hgold, hgoldt, m * n);
  LIBXSMM_VLA_DECL(2, float, djdhgold, djdhgoldt, m * n);
  LIBXSMM_VLA_DECL(2, float, deltagold, deltagoldt, m * n);
  LIBXSMM_VLA_DECL(2, float, doutgold, doutgoldt, m * n);
  LIBXSMM_VLA_DECL(2, float, djddgold, djddgoldt, m * n);
  LIBXSMM_VLA_DECL(2, float, djdigold, djdigoldt, m * n);
  LIBXSMM_VLA_DECL(2, float, djdfgold, djdfgoldt, m * n);
  LIBXSMM_VLA_DECL(2, float, djdogold, djdogoldt, m * n);
  LIBXSMM_VLA_DECL(2, float, djdcgold, djdcgoldt, m * n);
  LIBXSMM_VLA_DECL(2, float, djdxgold, djdxgoldt, k * n);

  /*LIBXSMM_VLA_DECL(2, float, x, xt, k * n);*/
  LIBXSMM_VLA_DECL(2, float, h, ht, m * n);
  /*LIBXSMM_VLA_DECL(2, float, dx, dxt, k * n);*/
  /*LIBXSMM_VLA_DECL(2, float, dh, dht, m * n);*/

  /* initialize data */
  /* FWD */
  LIBXSMM_MATINIT_OMP(float, 24, cspgold,n, m, n, 1.0);
  LIBXSMM_MATINIT_OMP(float, 24, hpgold, n, m, n, 1.0);
  LIBXSMM_MATINIT_OMP(float, 42, wigold, k, m, k, 1.0);
  LIBXSMM_MATINIT_OMP(float, 42, wfgold, k, m, k, 1.0);
  LIBXSMM_MATINIT_OMP(float, 42, wogold, k, m, k, 1.0);
  LIBXSMM_MATINIT_OMP(float, 42, wcgold, k, m, k, 1.0);
  for (j = 0; j < t; ++j) {
    LIBXSMM_MATINIT_OMP(float, 24, &LIBXSMM_VLA_ACCESS(2, xgold, j, 0, k * n), n, k, n, 1.0);
  }
  LIBXSMM_MATINIT_OMP(float, 42, rigold, m, m, m, 1.0);
  LIBXSMM_MATINIT_OMP(float, 42, rfgold, m, m, m, 1.0);
  LIBXSMM_MATINIT_OMP(float, 42, rogold, m, m, m, 1.0);
  LIBXSMM_MATINIT_OMP(float, 42, rcgold, m, m, m, 1.0);
  LIBXSMM_MATINIT_OMP(float, 24, bigold, 1, m, 1, 1.0);
  LIBXSMM_MATINIT_OMP(float, 24, bfgold, 1, m, 1, 1.0);
  LIBXSMM_MATINIT_OMP(float, 24, bogold, 1, m, 1, 1.0);
  LIBXSMM_MATINIT_OMP(float, 24, bcgold, 1, m, 1, 1.0);
  for (j = 0; j < n; j++) {
    matrix_copy(m, bigold, &(bimgold[j*m]));
    matrix_copy(m, bfgold, &(bfmgold[j*m]));
    matrix_copy(m, bogold, &(bomgold[j*m]));
    matrix_copy(m, bcgold, &(bcmgold[j*m]));
  }
  for (j = 0; j < t; ++j) {
    zero_buf(&LIBXSMM_VLA_ACCESS(2, hgold, j, 0, m * n), m*n);
    zero_buf(&LIBXSMM_VLA_ACCESS(2, igold, j, 0, m * n), m*n);
    zero_buf(&LIBXSMM_VLA_ACCESS(2, fgold, j, 0, m * n), m*n);
    zero_buf(&LIBXSMM_VLA_ACCESS(2, ogold, j, 0, m * n), m*n);
    zero_buf(&LIBXSMM_VLA_ACCESS(2, cgold, j, 0, m * n), m*n);
    zero_buf(&LIBXSMM_VLA_ACCESS(2, dgold, j, 0, m * n), m*n);
  }
  zero_buf(i1gold, m*n);
  zero_buf(i2gold, m*n);
  zero_buf(f1gold, m*n);
  zero_buf(f2gold, m*n);
  zero_buf(o1gold, m*n);
  zero_buf(o2gold, m*n);
  zero_buf(c1gold, m*n);
  zero_buf(c2gold, m*n);
  zero_buf(d1gold, m*n);
  zero_buf(d2gold, m*n);
  zero_buf(dhgold, m*n);
  /* BWD/UPD */
  for (j = 0; j < t; ++j) {
    LIBXSMM_MATINIT_OMP(float, 24, &LIBXSMM_VLA_ACCESS(2, djdhgold, j, 0, m * n), n, m, n, 1.0);
  }
  zero_buf(i3gold, m*n);
  zero_buf(f3gold, m*n);
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
  zero_buf(djdbigold, m);
  zero_buf(djdbfgold, m);
  zero_buf(djdbogold, m);
  zero_buf(djdbcgold, m);
  zero_buf(wgoldTp, m*k);
  zero_buf(rgoldTp, m*m);
  zero_buf(xgoldTp, k*n);
  zero_buf(hgoldTp, m*n);
  zero_buf(doutgoldt, m*n*t);

  /* first touch LIBXSMM */
  zero_buf(xt,  k*n*t);
  zero_buf(csp, m*n);
  zero_buf(hp,  m*n);
  zero_buf(w,   m*k*4);
  zero_buf(r,   m*m*4);
  zero_buf(b,   m*4);
  zero_buf(cst, m*n*t);
  zero_buf(ht,  m*n*t);
  zero_buf(it,  m*n*t);
  zero_buf(ft,  m*n*t);
  zero_buf(ot,  m*n*t);
  zero_buf(cit, m*n*t);
  zero_buf(cot, m*n*t);
  zero_buf(dxt,   k*n*t);
  zero_buf(dcspt, m*n*t);
  zero_buf(dhpt,  m*n*t);
  zero_buf(dw,    m*k*4);
  zero_buf(dr,    m*m*4);
  zero_buf(db,    m*4);
  zero_buf(dcs,   m*n);
  zero_buf(dht,   m*n*t);

  if (LIBXSMM_NEQ(0, check)) {
    printf("##########################################\n");
    printf("#         Computing Reference ...        #\n");
    printf("##########################################\n");
    /* FWD */
    for (j = 0; j < t; ++j) {
      LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &m, &n, &k, &alpha, wigold, &m, &LIBXSMM_VLA_ACCESS(2, xgold, j, 0, k * n), &k, &beta0, i1gold, &m);
      LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &m, &n, &k, &alpha, wfgold, &m, &LIBXSMM_VLA_ACCESS(2, xgold, j, 0, k * n), &k, &beta0, f1gold, &m);
      LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &m, &n, &k, &alpha, wogold, &m, &LIBXSMM_VLA_ACCESS(2, xgold, j, 0, k * n), &k, &beta0, o1gold, &m);
      LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &m, &n, &k, &alpha, wcgold, &m, &LIBXSMM_VLA_ACCESS(2, xgold, j, 0, k * n), &k, &beta0, c1gold, &m);
      if (j == 0) {
        LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &m, &n, &m, &alpha, rigold, &m, hpgold, &m, &beta0, i2gold, &m);
        LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &m, &n, &m, &alpha, rfgold, &m, hpgold, &m, &beta0, f2gold, &m);
        LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &m, &n, &m, &alpha, rogold, &m, hpgold, &m, &beta0, o2gold, &m);
        LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &m, &n, &m, &alpha, rcgold, &m, hpgold, &m, &beta0, c2gold, &m);
      } else {
        LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &m, &n, &m, &alpha, rigold, &m, &LIBXSMM_VLA_ACCESS(2, hgold, j-1, 0, m * n), &m, &beta0, i2gold, &m);
        LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &m, &n, &m, &alpha, rfgold, &m, &LIBXSMM_VLA_ACCESS(2, hgold, j-1, 0, m * n), &m, &beta0, f2gold, &m);
        LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &m, &n, &m, &alpha, rogold, &m, &LIBXSMM_VLA_ACCESS(2, hgold, j-1, 0, m * n), &m, &beta0, o2gold, &m);
        LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &m, &n, &m, &alpha, rcgold, &m, &LIBXSMM_VLA_ACCESS(2, hgold, j-1, 0, m * n), &m, &beta0, c2gold, &m);
      }
      matrix_add(m*n, i1gold, i2gold, &LIBXSMM_VLA_ACCESS(2, igold, j, 0, m * n));
      matrix_add(m*n, &LIBXSMM_VLA_ACCESS(2, igold, j, 0, m * n), bimgold, &LIBXSMM_VLA_ACCESS(2, igold, j, 0, m * n));
      matrix_add(m*n, f1gold, f2gold, &LIBXSMM_VLA_ACCESS(2, fgold, j, 0, m * n));
      matrix_add(m*n, &LIBXSMM_VLA_ACCESS(2, fgold, j, 0, m * n), bfmgold, &LIBXSMM_VLA_ACCESS(2, fgold, j, 0, m * n));
      matrix_add(m*n, o1gold, o2gold, &LIBXSMM_VLA_ACCESS(2, ogold, j, 0, m * n));
      matrix_add(m*n, &LIBXSMM_VLA_ACCESS(2, ogold, j, 0, m * n), bomgold, &LIBXSMM_VLA_ACCESS(2, ogold, j, 0, m * n));
      matrix_add(m*n, c1gold, c2gold, &LIBXSMM_VLA_ACCESS(2, cgold, j, 0, m * n));
      matrix_add(m*n, &LIBXSMM_VLA_ACCESS(2, cgold, j, 0, m * n), bcmgold, &LIBXSMM_VLA_ACCESS(2, cgold, j, 0, m * n));
      matrix_sigmoid(m*n, &LIBXSMM_VLA_ACCESS(2, igold, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, igold, j, 0, m * n)); /*sigmoid*/
      matrix_sigmoid(m*n, &LIBXSMM_VLA_ACCESS(2, fgold, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, fgold, j, 0, m * n)); /*sigmoid*/
      matrix_sigmoid(m*n, &LIBXSMM_VLA_ACCESS(2, ogold, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, ogold, j, 0, m * n)); /*sigmoid*/
      matrix_tanh(m*n, &LIBXSMM_VLA_ACCESS(2, cgold, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, cgold, j, 0, m * n)); /*tanh*/
      if (j == 0) {
        matrix_eltwise_mult(m*n, &LIBXSMM_VLA_ACCESS(2, fgold, j, 0, m * n), cspgold, d1gold);
      } else {
        matrix_eltwise_mult(m*n, &LIBXSMM_VLA_ACCESS(2, fgold, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, dgold, j-1, 0, m * n), d1gold);
      }
      matrix_eltwise_mult(m*n, &LIBXSMM_VLA_ACCESS(2, igold, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, cgold, j, 0, m * n), d2gold);
      matrix_add(m*n, d1gold, d2gold, &LIBXSMM_VLA_ACCESS(2, dgold, j, 0, m * n));
      matrix_tanh(m*n, &LIBXSMM_VLA_ACCESS(2, dgold, j, 0, m * n), dhgold); /*tanh*/
      matrix_eltwise_mult(m*n, &LIBXSMM_VLA_ACCESS(2, ogold, j, 0, m * n), dhgold, &LIBXSMM_VLA_ACCESS(2, hgold, j, 0, m * n));
    }
    /* BWD/UPD */
    for (j = t-1; j >= 0; --j) {
      /* compute deltagold */
      if (j == t-1) {
        matrix_copy(m * n, &LIBXSMM_VLA_ACCESS(2, djdhgold, t-1, 0, m * n), &LIBXSMM_VLA_ACCESS(2, deltagold, t-1, 0, m * n));
      } else {
        matrix_add(m * n, &LIBXSMM_VLA_ACCESS(2, doutgold, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, djdhgold, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, deltagold, j, 0, m * n));
      }
      /* compute djddgold */
      matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, deltagold, j, 0, m * n), &LIBXSMM_VLA_ACCESS(2, ogold, j, 0, m * n), d1gold);
      matrix_tanh_inverse(m * n, &LIBXSMM_VLA_ACCESS(2, dgold, j, 0, m * n), d2gold);
      if (j == t-1) {
        matrix_eltwise_mult(m * n, d1gold, d2gold, &LIBXSMM_VLA_ACCESS(2, djddgold, j, 0, m * n));
      } else {
        matrix_eltwise_mult(m * n, d1gold, d2gold, d3gold);
        matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, djddgold, j+1, 0, m * n), &LIBXSMM_VLA_ACCESS(2, fgold, j+1, 0, m * n), d4gold);
        matrix_add(m * n, d3gold, d4gold, &LIBXSMM_VLA_ACCESS(2, djddgold, j, 0, m * n));
      }
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
        zero_buf(&LIBXSMM_VLA_ACCESS(2, djdfgold, j, 0, m * n), m*n);
      }
      /* compute djdogold */
      matrix_tanh(m * n, &LIBXSMM_VLA_ACCESS(2, dgold, j, 0, m * n), o1gold);
      matrix_complement(m * n, &LIBXSMM_VLA_ACCESS(2, ogold, j, 0, m * n), o2gold);
      matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, deltagold, j, 0, m * n), o1gold, o1gold);
      matrix_eltwise_mult(m * n, &LIBXSMM_VLA_ACCESS(2, ogold, j, 0, m * n), o2gold, o2gold);
      matrix_eltwise_mult(m * n, o1gold, o2gold, &LIBXSMM_VLA_ACCESS(2, djdogold, j, 0, m * n));
      if (j >= 1) {
        /* compute doutgold */
        matrix_transpose(m, m, rigold, rgoldTp);
        LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &m, &n, &m, &alpha, rgoldTp, &m, &LIBXSMM_VLA_ACCESS(2, djdigold, j, 0, m * n), &m, &beta, &LIBXSMM_VLA_ACCESS(2, doutgold, j-1, 0, m * n), &m);
        matrix_transpose(m, m, rfgold, rgoldTp);
        LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &m, &n, &m, &alpha, rgoldTp, &m, &LIBXSMM_VLA_ACCESS(2, djdfgold, j, 0, m * n), &m, &beta, &LIBXSMM_VLA_ACCESS(2, doutgold, j-1, 0, m * n), &m);
        matrix_transpose(m, m, rogold, rgoldTp);
        LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &m, &n, &m, &alpha, rgoldTp, &m, &LIBXSMM_VLA_ACCESS(2, djdogold, j, 0, m * n), &m, &beta, &LIBXSMM_VLA_ACCESS(2, doutgold, j-1, 0, m * n), &m);
        matrix_transpose(m, m, rcgold, rgoldTp);
        LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &m, &n, &m, &alpha, rgoldTp, &m, &LIBXSMM_VLA_ACCESS(2, djdcgold, j, 0, m * n), &m, &beta, &LIBXSMM_VLA_ACCESS(2, doutgold, j-1, 0, m * n), &m);
      }
      if (pass == 1 || pass == 3) {
        /* compute djdxgold */
        matrix_transpose(k, m, wigold, wgoldTp);
        LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &k, &n, &m, &alpha, wgoldTp, &k, &LIBXSMM_VLA_ACCESS(2, djdigold, j, 0, m * n), &m, &beta, &LIBXSMM_VLA_ACCESS(2, djdxgold, j, 0, k * n), &k);
        matrix_transpose(k, m, wfgold, wgoldTp);
        LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &k, &n, &m, &alpha, wgoldTp, &k, &LIBXSMM_VLA_ACCESS(2, djdfgold, j, 0, m * n), &m, &beta, &LIBXSMM_VLA_ACCESS(2, djdxgold, j, 0, k * n), &k);
        matrix_transpose(k, m, wogold, wgoldTp);
        LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &k, &n, &m, &alpha, wgoldTp, &k, &LIBXSMM_VLA_ACCESS(2, djdogold, j, 0, m * n), &m, &beta, &LIBXSMM_VLA_ACCESS(2, djdxgold, j, 0, k * n), &k);
        matrix_transpose(k, m, wcgold, wgoldTp);
        LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &k, &n, &m, &alpha, wgoldTp, &k, &LIBXSMM_VLA_ACCESS(2, djdcgold, j, 0, m * n), &m, &beta, &LIBXSMM_VLA_ACCESS(2, djdxgold, j, 0, k * n), &k);
      }
    }
    if (pass == 2 || pass == 3) {
      /* compute djdwgold */
      for (j = 0; j < t; ++j) {
        matrix_transpose(n, k, &LIBXSMM_VLA_ACCESS(2, xgold, j, 0, k * n), xgoldTp);
        LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &m, &k, &n, &alpha, &LIBXSMM_VLA_ACCESS(2, djdigold, j, 0, m * n), &m, xgoldTp, &n, &beta, djdwigold, &m);
        LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &m, &k, &n, &alpha, &LIBXSMM_VLA_ACCESS(2, djdfgold, j, 0, m * n), &m, xgoldTp, &n, &beta, djdwfgold, &m);
        LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &m, &k, &n, &alpha, &LIBXSMM_VLA_ACCESS(2, djdogold, j, 0, m * n), &m, xgoldTp, &n, &beta, djdwogold, &m);
        LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &m, &k, &n, &alpha, &LIBXSMM_VLA_ACCESS(2, djdcgold, j, 0, m * n), &m, xgoldTp, &n, &beta, djdwcgold, &m);
      }
      /* compute djdrgold */
      for (j = 0; j < t-1; ++j) {
        matrix_transpose(n, m, &LIBXSMM_VLA_ACCESS(2, hgold, j, 0, m * n), hgoldTp);
        LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &m, &m, &n, &alpha, &LIBXSMM_VLA_ACCESS(2, djdigold, j+1, 0, m * n), &m, hgoldTp, &n, &beta, djdrigold, &m);
        LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &m, &m, &n, &alpha, &LIBXSMM_VLA_ACCESS(2, djdfgold, j+1, 0, m * n), &m, hgoldTp, &n, &beta, djdrfgold, &m);
        LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &m, &m, &n, &alpha, &LIBXSMM_VLA_ACCESS(2, djdogold, j+1, 0, m * n), &m, hgoldTp, &n, &beta, djdrogold, &m);
        LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &m, &m, &n, &alpha, &LIBXSMM_VLA_ACCESS(2, djdcgold, j+1, 0, m * n), &m, hgoldTp, &n, &beta, djdrcgold, &m);
      }
      /* compute djdbgold */
      for (j = 0; j < t-1; j++) {
        for (l = 0; l < m*n; l++) {
          djdbigold[l%m] += LIBXSMM_VLA_ACCESS(2, djdigold, j+1, l, m * n);
          djdbfgold[l%m] += LIBXSMM_VLA_ACCESS(2, djdfgold, j+1, l, m * n);
          djdbogold[l%m] += LIBXSMM_VLA_ACCESS(2, djdogold, j+1, l, m * n);
          djdbcgold[l%m] += LIBXSMM_VLA_ACCESS(2, djdcgold, j+1, l, m * n);
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
    lstmcell_desc.nThreads = nThreads;
    lstmcell_desc.K = m;
    lstmcell_desc.N = n;
    lstmcell_desc.C = k;
    lstmcell_desc.t = t;
    lstmcell_desc.pass = pass;
    lstmcell_desc.datatype_in = LIBXSMM_DNN_DATATYPE_F32;
    lstmcell_desc.datatype_out = LIBXSMM_DNN_DATATYPE_F32;
    lstmcell_desc.buffer_format = LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM;

    libxsmm_handle = libxsmm_dnn_create_lstmcell( lstmcell_desc, &status );
    CHKERR_LIBXSMM_DNN( status );

    /* setup LIBXSMM buffers and filter */
    libxsmm_layout = libxsmm_dnn_lstmcell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_LSTM_REGULAR_INPUT, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_input = libxsmm_dnn_link_tensor( libxsmm_layout, xt, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_lstmcell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_LSTM_REGULAR_CS_PREV, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_cs_prev = libxsmm_dnn_link_tensor( libxsmm_layout, csp, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_lstmcell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_LSTM_REGULAR_HIDDEN_STATE_PREV, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_hidden_state_prev = libxsmm_dnn_link_tensor( libxsmm_layout, hp, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_lstmcell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_LSTM_REGULAR_WEIGHT, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_weight = libxsmm_dnn_link_tensor( libxsmm_layout, w, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_lstmcell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_LSTM_REGULAR_RECUR_WEIGHT, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_recur_weight = libxsmm_dnn_link_tensor( libxsmm_layout, r, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_lstmcell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_LSTM_REGULAR_BIAS, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_bias = libxsmm_dnn_link_tensor( libxsmm_layout, b, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_lstmcell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_LSTM_REGULAR_CS, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_cs = libxsmm_dnn_link_tensor( libxsmm_layout, cst, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_lstmcell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_LSTM_REGULAR_HIDDEN_STATE, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_hidden_state = libxsmm_dnn_link_tensor( libxsmm_layout, ht, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_lstmcell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_LSTM_INTERNAL_I, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_i = libxsmm_dnn_link_tensor( libxsmm_layout, it, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_lstmcell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_LSTM_INTERNAL_F, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_f = libxsmm_dnn_link_tensor( libxsmm_layout, ft, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_lstmcell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_LSTM_INTERNAL_O, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_o = libxsmm_dnn_link_tensor( libxsmm_layout, ot, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_lstmcell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_LSTM_INTERNAL_CI, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_ci = libxsmm_dnn_link_tensor( libxsmm_layout, cit, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_lstmcell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_LSTM_INTERNAL_CO, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_co = libxsmm_dnn_link_tensor( libxsmm_layout, cot, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_lstmcell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_LSTM_GRADIENT_INPUT, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dinput = libxsmm_dnn_link_tensor( libxsmm_layout, dxt, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_lstmcell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_LSTM_GRADIENT_CS_PREV, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dcs_prev = libxsmm_dnn_link_tensor( libxsmm_layout, dcspt, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_lstmcell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_LSTM_GRADIENT_HIDDEN_STATE_PREV, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dhidden_state_prev = libxsmm_dnn_link_tensor( libxsmm_layout, dhpt, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_lstmcell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_LSTM_GRADIENT_WEIGHT, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dweight = libxsmm_dnn_link_tensor( libxsmm_layout, dw, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_lstmcell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_LSTM_GRADIENT_RECUR_WEIGHT, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_drecur_weight = libxsmm_dnn_link_tensor( libxsmm_layout, dr, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_lstmcell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_LSTM_GRADIENT_BIAS, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dbias = libxsmm_dnn_link_tensor( libxsmm_layout, db, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_lstmcell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_LSTM_GRADIENT_CS, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dcs = libxsmm_dnn_link_tensor( libxsmm_layout, dcs, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_lstmcell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_LSTM_GRADIENT_HIDDEN_STATE, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dhidden_state = libxsmm_dnn_link_tensor( libxsmm_layout, dht, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    /* copy in data to LIBXSMM format */
    matrix_copy(k*n*t, xgoldt, xt);
    matrix_copy(m*n, cspgold, csp);
    matrix_copy(m*n, hpgold, hp);
    matrix_copy(m*k, wigold, &(w[0]));
    matrix_copy(m*k, wcgold, &(w[m*k]));
    matrix_copy(m*k, wfgold, &(w[2*m*k]));
    matrix_copy(m*k, wogold, &(w[3*m*k]));
    matrix_copy(m*m, rigold, &(r[0]));
    matrix_copy(m*m, rcgold, &(r[m*m]));
    matrix_copy(m*m, rfgold, &(r[2*m*m]));
    matrix_copy(m*m, rogold, &(r[3*m*m]));
    matrix_copy(m, bigold, &(b[0]));
    matrix_copy(m, bcgold, &(b[m]));
    matrix_copy(m, bfgold, &(b[2*m]));
    matrix_copy(m, bogold, &(b[3*m]));
    matrix_copy(m*n*t, djdhgoldt, dht);

    /* bind buffers and filter to handle */
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_lstmcell_bind_tensor( libxsmm_handle, libxsmm_input, LIBXSMM_DNN_LSTM_REGULAR_INPUT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_lstmcell_bind_tensor( libxsmm_handle, libxsmm_cs_prev, LIBXSMM_DNN_LSTM_REGULAR_CS_PREV ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_lstmcell_bind_tensor( libxsmm_handle, libxsmm_hidden_state_prev, LIBXSMM_DNN_LSTM_REGULAR_HIDDEN_STATE_PREV ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_lstmcell_bind_tensor( libxsmm_handle, libxsmm_weight, LIBXSMM_DNN_LSTM_REGULAR_WEIGHT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_lstmcell_bind_tensor( libxsmm_handle, libxsmm_recur_weight, LIBXSMM_DNN_LSTM_REGULAR_RECUR_WEIGHT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_lstmcell_bind_tensor( libxsmm_handle, libxsmm_bias, LIBXSMM_DNN_LSTM_REGULAR_BIAS ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_lstmcell_bind_tensor( libxsmm_handle, libxsmm_cs, LIBXSMM_DNN_LSTM_REGULAR_CS ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_lstmcell_bind_tensor( libxsmm_handle, libxsmm_hidden_state, LIBXSMM_DNN_LSTM_REGULAR_HIDDEN_STATE ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_lstmcell_bind_tensor( libxsmm_handle, libxsmm_i, LIBXSMM_DNN_LSTM_INTERNAL_I ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_lstmcell_bind_tensor( libxsmm_handle, libxsmm_f, LIBXSMM_DNN_LSTM_INTERNAL_F ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_lstmcell_bind_tensor( libxsmm_handle, libxsmm_o, LIBXSMM_DNN_LSTM_INTERNAL_O ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_lstmcell_bind_tensor( libxsmm_handle, libxsmm_ci, LIBXSMM_DNN_LSTM_INTERNAL_CI ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_lstmcell_bind_tensor( libxsmm_handle, libxsmm_co, LIBXSMM_DNN_LSTM_INTERNAL_CO ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_lstmcell_bind_tensor( libxsmm_handle, libxsmm_dinput, LIBXSMM_DNN_LSTM_GRADIENT_INPUT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_lstmcell_bind_tensor( libxsmm_handle, libxsmm_dcs_prev, LIBXSMM_DNN_LSTM_GRADIENT_CS_PREV ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_lstmcell_bind_tensor( libxsmm_handle, libxsmm_dhidden_state_prev, LIBXSMM_DNN_LSTM_GRADIENT_HIDDEN_STATE_PREV ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_lstmcell_bind_tensor( libxsmm_handle, libxsmm_dweight, LIBXSMM_DNN_LSTM_GRADIENT_WEIGHT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_lstmcell_bind_tensor( libxsmm_handle, libxsmm_drecur_weight, LIBXSMM_DNN_LSTM_GRADIENT_RECUR_WEIGHT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_lstmcell_bind_tensor( libxsmm_handle, libxsmm_dbias, LIBXSMM_DNN_LSTM_GRADIENT_BIAS ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_lstmcell_bind_tensor( libxsmm_handle, libxsmm_dcs, LIBXSMM_DNN_LSTM_GRADIENT_CS ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_lstmcell_bind_tensor( libxsmm_handle, libxsmm_dhidden_state, LIBXSMM_DNN_LSTM_GRADIENT_HIDDEN_STATE ) );

    /* let's allocate and bind scratch */
    if (pass == 0) {
      scratch_size = libxsmm_dnn_lstmcell_get_scratch_size( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_FWD, &status );
      CHKERR_LIBXSMM_DNN( status );
      scratch = libxsmm_aligned_malloc( scratch_size, 2097152 );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_lstmcell_bind_scratch( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_FWD, scratch ) );
    } else {
      scratch_size = libxsmm_dnn_lstmcell_get_scratch_size( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_ALL, &status );
      CHKERR_LIBXSMM_DNN( status );
      scratch = libxsmm_aligned_malloc( scratch_size, 2097152 );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_lstmcell_bind_scratch( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_ALL, scratch ) );
    }
    zero_buf( (float*)scratch, scratch_size/4 );

    /* let's allocate and bind internalstate */
    if (pass == 0) {
      internalstate_size = libxsmm_dnn_lstmcell_get_internalstate_size( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_FWD, &status );
      CHKERR_LIBXSMM_DNN( status );
      internalstate = libxsmm_aligned_malloc( internalstate_size, 2097152 );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_lstmcell_bind_internalstate( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_FWD, internalstate ) );
    } else {
      internalstate_size = libxsmm_dnn_lstmcell_get_internalstate_size( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_ALL, &status );
      CHKERR_LIBXSMM_DNN( status );
      internalstate = libxsmm_aligned_malloc( internalstate_size, 2097152 );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_lstmcell_bind_internalstate( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_ALL, internalstate ) );
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
        CHKERR_LIBXSMM_DNN( libxsmm_dnn_lstmcell_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_FWD, 0, tid ) );
      }
      /* copy out data */
      matrix_copy(m*n, &LIBXSMM_VLA_ACCESS(2, h, t-1, 0, m * n), htest);

      /* compare */
      libxsmm_matdiff(LIBXSMM_DATATYPE_F32, m*n, 1, &LIBXSMM_VLA_ACCESS(2, hgold, t-1, 0, m * n), htest, 0, 0, &norms_fwd);
      printf("L1 reference  : %.25g\n", norms_fwd.l1_ref);
      printf("L1 test       : %.25g\n", norms_fwd.l1_tst);
      printf("L2 abs.error  : %.24f\n", norms_fwd.l2_abs);
      printf("L2 rel.error  : %.24f\n", norms_fwd.l2_rel);
      printf("Linf abs.error: %.24f\n", norms_fwd.linf_abs);
      printf("Linf rel.error: %.24f\n", norms_fwd.linf_rel);
      printf("Check-norm    : %.24f\n", norms_fwd.normf_rel);
      libxsmm_matdiff_reduce(&diff, &norms_fwd);
    } else {
      /* We need to always run FWD pass once to populate i, f, o, ci, co, cs */
#if defined(_OPENMP)
#     pragma omp parallel
#endif
      {
#if defined(_OPENMP)
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        CHKERR_LIBXSMM_DNN( libxsmm_dnn_lstmcell_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_FWD, 0, tid ) );
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
        CHKERR_LIBXSMM_DNN( libxsmm_dnn_lstmcell_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_BWD, 0, tid ) );
      }

      /* copy out data */
      matrix_copy(k*n*t, dxt, djdxtestt);

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
        CHKERR_LIBXSMM_DNN( libxsmm_dnn_lstmcell_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_UPD, 0, tid ) );
      }

      /* copy out data */
      LIBXSMM_VLA_DECL(2, float, djdw4test, djdwtest, m * k);
      LIBXSMM_VLA_DECL(2, float, djdr4test, djdrtest, m * m);
      LIBXSMM_VLA_DECL(2, float, djdb4test, djdbtest, m);
      matrix_copy(m*k, &(dw[0]),     &LIBXSMM_VLA_ACCESS(2, djdw4test, 0, 0, m * k));
      matrix_copy(m*k, &(dw[m*k]),   &LIBXSMM_VLA_ACCESS(2, djdw4test, 1, 0, m * k));
      matrix_copy(m*k, &(dw[2*m*k]), &LIBXSMM_VLA_ACCESS(2, djdw4test, 2, 0, m * k));
      matrix_copy(m*k, &(dw[3*m*k]), &LIBXSMM_VLA_ACCESS(2, djdw4test, 3, 0, m * k));
      matrix_copy(m*m, &(dr[0]),     &LIBXSMM_VLA_ACCESS(2, djdr4test, 0, 0, m * m));
      matrix_copy(m*m, &(dr[m*m]),   &LIBXSMM_VLA_ACCESS(2, djdr4test, 1, 0, m * m));
      matrix_copy(m*m, &(dr[2*m*m]), &LIBXSMM_VLA_ACCESS(2, djdr4test, 2, 0, m * m));
      matrix_copy(m*m, &(dr[3*m*m]), &LIBXSMM_VLA_ACCESS(2, djdr4test, 3, 0, m * m));
      matrix_copy(m, &(db[0]),   &LIBXSMM_VLA_ACCESS(2, djdb4test, 0, 0, m));
      matrix_copy(m, &(db[m]),   &LIBXSMM_VLA_ACCESS(2, djdb4test, 1, 0, m));
      matrix_copy(m, &(db[2*m]), &LIBXSMM_VLA_ACCESS(2, djdb4test, 2, 0, m));
      matrix_copy(m, &(db[3*m]), &LIBXSMM_VLA_ACCESS(2, djdb4test, 3, 0, m));
      LIBXSMM_VLA_DECL(2, float, djdw4, djdwgold4, m * k);
      LIBXSMM_VLA_DECL(2, float, djdr4, djdrgold4, m * m);
      LIBXSMM_VLA_DECL(2, float, djdb4, djdbgold4, m);
      matrix_copy(m * k, djdwigold, &LIBXSMM_VLA_ACCESS(2, djdw4, 0, 0, m * k));
      matrix_copy(m * k, djdwcgold, &LIBXSMM_VLA_ACCESS(2, djdw4, 1, 0, m * k));
      matrix_copy(m * k, djdwfgold, &LIBXSMM_VLA_ACCESS(2, djdw4, 2, 0, m * k));
      matrix_copy(m * k, djdwogold, &LIBXSMM_VLA_ACCESS(2, djdw4, 3, 0, m * k));
      matrix_copy(m * m, djdrigold, &LIBXSMM_VLA_ACCESS(2, djdr4, 0, 0, m * m));
      matrix_copy(m * m, djdrcgold, &LIBXSMM_VLA_ACCESS(2, djdr4, 1, 0, m * m));
      matrix_copy(m * m, djdrfgold, &LIBXSMM_VLA_ACCESS(2, djdr4, 2, 0, m * m));
      matrix_copy(m * m, djdrogold, &LIBXSMM_VLA_ACCESS(2, djdr4, 3, 0, m * m));
      matrix_copy(m, djdbigold, &LIBXSMM_VLA_ACCESS(2, djdb4, 0, 0, m));
      matrix_copy(m, djdbcgold, &LIBXSMM_VLA_ACCESS(2, djdb4, 1, 0, m));
      matrix_copy(m, djdbfgold, &LIBXSMM_VLA_ACCESS(2, djdb4, 2, 0, m));
      matrix_copy(m, djdbogold, &LIBXSMM_VLA_ACCESS(2, djdb4, 3, 0, m));

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

      libxsmm_matdiff(LIBXSMM_DATATYPE_F32, m*4, 1, djdbgold4, djdbtest, 0, 0, &norms_upd_b);
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
        CHKERR_LIBXSMM_DNN( libxsmm_dnn_lstmcell_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_ALL, 0, tid ) );
      }

      /* copy out data */
      matrix_copy(k*n*t, dxt, djdxtestt);
      LIBXSMM_VLA_DECL(2, float, djdw4test, djdwtest, m * k);
      LIBXSMM_VLA_DECL(2, float, djdr4test, djdrtest, m * m);
      LIBXSMM_VLA_DECL(2, float, djdb4test, djdbtest, m);
      matrix_copy(m*k, &(dw[0]),     &LIBXSMM_VLA_ACCESS(2, djdw4test, 0, 0, m * k));
      matrix_copy(m*k, &(dw[m*k]),   &LIBXSMM_VLA_ACCESS(2, djdw4test, 1, 0, m * k));
      matrix_copy(m*k, &(dw[2*m*k]), &LIBXSMM_VLA_ACCESS(2, djdw4test, 2, 0, m * k));
      matrix_copy(m*k, &(dw[3*m*k]), &LIBXSMM_VLA_ACCESS(2, djdw4test, 3, 0, m * k));
      matrix_copy(m*m, &(dr[0]),     &LIBXSMM_VLA_ACCESS(2, djdr4test, 0, 0, m * m));
      matrix_copy(m*m, &(dr[m*m]),   &LIBXSMM_VLA_ACCESS(2, djdr4test, 1, 0, m * m));
      matrix_copy(m*m, &(dr[2*m*m]), &LIBXSMM_VLA_ACCESS(2, djdr4test, 2, 0, m * m));
      matrix_copy(m*m, &(dr[3*m*m]), &LIBXSMM_VLA_ACCESS(2, djdr4test, 3, 0, m * m));
      matrix_copy(m, &(db[0]),   &LIBXSMM_VLA_ACCESS(2, djdb4test, 0, 0, m));
      matrix_copy(m, &(db[m]),   &LIBXSMM_VLA_ACCESS(2, djdb4test, 1, 0, m));
      matrix_copy(m, &(db[2*m]), &LIBXSMM_VLA_ACCESS(2, djdb4test, 2, 0, m));
      matrix_copy(m, &(db[3*m]), &LIBXSMM_VLA_ACCESS(2, djdb4test, 3, 0, m));
      LIBXSMM_VLA_DECL(2, float, djdw4, djdwgold4, m * k);
      LIBXSMM_VLA_DECL(2, float, djdr4, djdrgold4, m * m);
      LIBXSMM_VLA_DECL(2, float, djdb4, djdbgold4, m);
      matrix_copy(m * k, djdwigold, &LIBXSMM_VLA_ACCESS(2, djdw4, 0, 0, m * k));
      matrix_copy(m * k, djdwcgold, &LIBXSMM_VLA_ACCESS(2, djdw4, 1, 0, m * k));
      matrix_copy(m * k, djdwfgold, &LIBXSMM_VLA_ACCESS(2, djdw4, 2, 0, m * k));
      matrix_copy(m * k, djdwogold, &LIBXSMM_VLA_ACCESS(2, djdw4, 3, 0, m * k));
      matrix_copy(m * m, djdrigold, &LIBXSMM_VLA_ACCESS(2, djdr4, 0, 0, m * m));
      matrix_copy(m * m, djdrcgold, &LIBXSMM_VLA_ACCESS(2, djdr4, 1, 0, m * m));
      matrix_copy(m * m, djdrfgold, &LIBXSMM_VLA_ACCESS(2, djdr4, 2, 0, m * m));
      matrix_copy(m * m, djdrogold, &LIBXSMM_VLA_ACCESS(2, djdr4, 3, 0, m * m));
      matrix_copy(m, djdbigold, &LIBXSMM_VLA_ACCESS(2, djdb4, 0, 0, m));
      matrix_copy(m, djdbcgold, &LIBXSMM_VLA_ACCESS(2, djdb4, 1, 0, m));
      matrix_copy(m, djdbfgold, &LIBXSMM_VLA_ACCESS(2, djdb4, 2, 0, m));
      matrix_copy(m, djdbogold, &LIBXSMM_VLA_ACCESS(2, djdb4, 3, 0, m));

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

      libxsmm_matdiff(LIBXSMM_DATATYPE_F32, m*4, 1, djdbgold4, djdbtest, 0, 0, &norms_upd_b);
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
          libxsmm_dnn_lstmcell_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_FWD, 0, tid );
        }
      }
      l_end = libxsmm_timer_tick();
      l_total = libxsmm_timer_duration(l_start, l_end);
      flops = (((2.0 * m * n * k) + (2.0 * m * n * m) + (2.0 * m * n) + (tflops * m * n)) * 4.0 + (4.0 * m * n) + (tflops * m * n)) * (double)t * (double)iters;

      printf("GFLOP  = %.5g\n", flops*1e-9/(double)iters);
      printf("fp time = %.5g\n", ((double)(l_total/iters)));
      printf("GFLOPS  = %.5g\n", (flops*1e-9)/l_total);

      printf("PERFDUMP,FP,%s,%i,%i,%i,%i,%i,%i,%i,%i,%.5g,%.5g\n", LIBXSMM_VERSION, nThreads, m, n, k, t, bm, bn, bk, ((double)(l_total/iters)), (flops*1e-9)/l_total);
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
          libxsmm_dnn_lstmcell_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_BWD, 0, tid );
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
#     pragma omp parallel private(j)
#endif
      {
#if defined(_OPENMP)
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        for (j = 0; j < iters; ++j) {
          libxsmm_dnn_lstmcell_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_UPD, 0, tid );
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
#     pragma omp parallel private(j)
#endif
      {
#if defined(_OPENMP)
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        for (j = 0; j < iters; ++j) {
          libxsmm_dnn_lstmcell_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_ALL, 0, tid );
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

    /* clean-up */
    if (pass == 0) {
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_lstmcell_release_scratch( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_FWD ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_lstmcell_release_internalstate( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_FWD ) );
    } else {
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_lstmcell_release_scratch( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_ALL ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_lstmcell_release_internalstate( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_ALL ) );
    }
    libxsmm_free(scratch);
    libxsmm_free(internalstate);
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_lstmcell_release_tensor( libxsmm_handle, LIBXSMM_DNN_LSTM_REGULAR_INPUT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_lstmcell_release_tensor( libxsmm_handle, LIBXSMM_DNN_LSTM_REGULAR_CS_PREV ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_lstmcell_release_tensor( libxsmm_handle, LIBXSMM_DNN_LSTM_REGULAR_HIDDEN_STATE_PREV ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_lstmcell_release_tensor( libxsmm_handle, LIBXSMM_DNN_LSTM_REGULAR_WEIGHT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_lstmcell_release_tensor( libxsmm_handle, LIBXSMM_DNN_LSTM_REGULAR_RECUR_WEIGHT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_lstmcell_release_tensor( libxsmm_handle, LIBXSMM_DNN_LSTM_REGULAR_BIAS ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_lstmcell_release_tensor( libxsmm_handle, LIBXSMM_DNN_LSTM_REGULAR_CS ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_lstmcell_release_tensor( libxsmm_handle, LIBXSMM_DNN_LSTM_REGULAR_HIDDEN_STATE ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_lstmcell_release_tensor( libxsmm_handle, LIBXSMM_DNN_LSTM_INTERNAL_I ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_lstmcell_release_tensor( libxsmm_handle, LIBXSMM_DNN_LSTM_INTERNAL_F ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_lstmcell_release_tensor( libxsmm_handle, LIBXSMM_DNN_LSTM_INTERNAL_O ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_lstmcell_release_tensor( libxsmm_handle, LIBXSMM_DNN_LSTM_INTERNAL_CI ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_lstmcell_release_tensor( libxsmm_handle, LIBXSMM_DNN_LSTM_INTERNAL_CO ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_lstmcell_release_tensor( libxsmm_handle, LIBXSMM_DNN_LSTM_GRADIENT_INPUT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_lstmcell_release_tensor( libxsmm_handle, LIBXSMM_DNN_LSTM_GRADIENT_CS_PREV ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_lstmcell_release_tensor( libxsmm_handle, LIBXSMM_DNN_LSTM_GRADIENT_HIDDEN_STATE_PREV ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_lstmcell_release_tensor( libxsmm_handle, LIBXSMM_DNN_LSTM_GRADIENT_WEIGHT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_lstmcell_release_tensor( libxsmm_handle, LIBXSMM_DNN_LSTM_GRADIENT_RECUR_WEIGHT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_lstmcell_release_tensor( libxsmm_handle, LIBXSMM_DNN_LSTM_GRADIENT_BIAS ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_lstmcell_release_tensor( libxsmm_handle, LIBXSMM_DNN_LSTM_GRADIENT_CS ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_lstmcell_release_tensor( libxsmm_handle, LIBXSMM_DNN_LSTM_GRADIENT_HIDDEN_STATE ) );
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
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_lstmcell( libxsmm_handle ) );
  }

  /* deallocate data */
  /*
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
  libxsmm_free(hgoldt);
  libxsmm_free(bimgold);
  libxsmm_free(bfmgold);
  libxsmm_free(bomgold);
  libxsmm_free(bcmgold);
  libxsmm_free(igoldt);
  libxsmm_free(fgoldt);
  libxsmm_free(ogoldt);
  libxsmm_free(cgoldt);
  libxsmm_free(dgoldt);
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
  libxsmm_free(doutgoldt);
  libxsmm_free(xt);
  libxsmm_free(csp);
  libxsmm_free(hp);
  libxsmm_free(w);
  libxsmm_free(r);
  libxsmm_free(b);
  libxsmm_free(cst);
  libxsmm_free(ht);
  libxsmm_free(dxt);
  libxsmm_free(dcspt);
  libxsmm_free(dhpt);
  libxsmm_free(dw);
  libxsmm_free(dr);
  libxsmm_free(db);
  libxsmm_free(dcs);
  libxsmm_free(dht);
  libxsmm_free(htest);
  libxsmm_free(djdxtestt);
  libxsmm_free(djdwtest);
  libxsmm_free(djdrtest);
  libxsmm_free(djdbtest);
  libxsmm_free(djdrgold4);
  libxsmm_free(djdbgold4);
  */

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

