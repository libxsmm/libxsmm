/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Evangelos Georganas, Kunal Banerjee (Intel Corp.)
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

/* include c-based dnn library */
#include "../common/dnn_common.h"

#define CHKERR_LIBXSMM_DNN(A) { const int chkerr_libxsmm_dnn_ = A; if (LIBXSMM_DNN_SUCCESS != chkerr_libxsmm_dnn_) { \
  fprintf(stderr, "%s\n", libxsmm_dnn_get_error(chkerr_libxsmm_dnn_)); global_status = chkerr_libxsmm_dnn_; } \
}

int main(int argc, char* argv[])
{
  float *wigold, *wfgold, *wogold, *wcgold, *xgoldt, *rigold, *rfgold, *rogold, *rcgold, *hgoldt, *bigold, *bfgold, *bogold, *bcgold, *bfgold_fb;
  float *cspgold, *hpgold, *djdcspgold, *djdhpgold;
  float *igoldt, *fgoldt, *ogoldt, *cgoldt, *dgoldt, *bimgold, *bfmgold, *bomgold, *bcmgold, *doutgoldt;
  float *i1gold, *i2gold, *f1gold, *f2gold, *o1gold, *o2gold, *c1gold, *c2gold, *d1gold, *d2gold, *dhgold;
  float *xt, *csp, *hp, *w, *wt, *w_tmp, *r, *rt, *r_tmp, *b, *cst, *ht;
  float *it, *ft, *ot, *cit, *cot;
  float *dxt, *dcsp, *dhp, *dw, *dr, *db, *dcs, *dht;
  float *i3gold, *f3gold, *d3gold, *d4gold, *deltagoldt;
  float *djdhgoldt, *djdigoldt, *djdfgoldt, *djdcgoldt, *djdogoldt, *djdxgoldt;
  float *djdwigold, *djdwfgold, *djdwogold, *djdwcgold, *djdrigold, *djdrfgold, *djdrogold, *djdrcgold;
  float *djdbigold, *djdbfgold, *djdbogold, *djdbcgold, *djdcsgold, *wgoldTp, *rgoldTp, *xgoldTp, *hgoldTp;
  float *htest, *djdxtestt, *djdwtest, *djdrtest, *djdbtest, *djdwgold4, *djdrgold4, *djdbgold4;
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
  int t = 5;        /* number of time steps (> 1) */
  int bk = 64;
  int bn = 64;
  int bc = 64;

  const char *const env_check = getenv("CHECK");
  const double check = LIBXSMM_ABS(0 == env_check ? 1/*enable by default*/ : atof(env_check));

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

  libxsmm_dnn_rnncell_desc lstmcell_desc;
  libxsmm_dnn_rnncell* libxsmm_handle;
  libxsmm_dnn_tensor* libxsmm_input;
  libxsmm_dnn_tensor* libxsmm_cs_prev;
  libxsmm_dnn_tensor* libxsmm_hidden_state_prev;
  libxsmm_dnn_tensor* libxsmm_weight;
  libxsmm_dnn_tensor* libxsmm_recur_weight;
  libxsmm_dnn_tensor* libxsmm_weight_t;
  libxsmm_dnn_tensor* libxsmm_recur_weight_t;
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
  libxsmm_rng_set_seed(1);

  /* reading new values from cli */
  j = 1;
  if (argc > j) iters = atoi(argv[j++]);
  if (argc > j) pass  = atoi(argv[j++]);
  if (argc > j) N     = atoi(argv[j++]);
  if (argc > j) C     = atoi(argv[j++]);
  if (argc > j) K     = atoi(argv[j++]);
  if (argc > j) t     = atoi(argv[j++]);
  if (argc > j) bn     = atoi(argv[j++]);
  if (argc > j) bk     = atoi(argv[j++]);
  if (argc > j) bc     = atoi(argv[j++]);

  if (t <= 0) {
    printf("time_steps %d should be greater than 1\n\n", t);
    return 0;
  }
  if (!(pass == 0 || pass == 1 || pass == 2 || pass == 3 || pass == 4)) {
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
  bigold = (float*)libxsmm_aligned_malloc(K*sizeof(float), 2097152);
  bfgold = (float*)libxsmm_aligned_malloc(K*sizeof(float), 2097152);
  bogold = (float*)libxsmm_aligned_malloc(K*sizeof(float), 2097152);
  bcgold = (float*)libxsmm_aligned_malloc(K*sizeof(float), 2097152);
  hgoldt = (float*)libxsmm_aligned_malloc(K*N*t*sizeof(float), 2097152);
  bfgold_fb = (float*)libxsmm_aligned_malloc(K*sizeof(float), 2097152);
  bimgold= (float*)libxsmm_aligned_malloc(K*N*sizeof(float), 2097152);
  bfmgold= (float*)libxsmm_aligned_malloc(K*N*sizeof(float), 2097152);
  bomgold= (float*)libxsmm_aligned_malloc(K*N*sizeof(float), 2097152);
  bcmgold= (float*)libxsmm_aligned_malloc(K*N*sizeof(float), 2097152);
  igoldt = (float*)libxsmm_aligned_malloc(K*N*t*sizeof(float), 2097152);
  fgoldt = (float*)libxsmm_aligned_malloc(K*N*t*sizeof(float), 2097152);
  ogoldt = (float*)libxsmm_aligned_malloc(K*N*t*sizeof(float), 2097152);
  cgoldt = (float*)libxsmm_aligned_malloc(K*N*t*sizeof(float), 2097152);
  dgoldt = (float*)libxsmm_aligned_malloc(K*N*t*sizeof(float), 2097152);
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
  djdxgoldt  = (float*)libxsmm_aligned_malloc(N*C*t*sizeof(float), 2097152);
  djdwigold  = (float*)libxsmm_aligned_malloc(C*K*sizeof(float), 2097152);
  djdwfgold  = (float*)libxsmm_aligned_malloc(C*K*sizeof(float), 2097152);
  djdwogold  = (float*)libxsmm_aligned_malloc(C*K*sizeof(float), 2097152);
  djdwcgold  = (float*)libxsmm_aligned_malloc(C*K*sizeof(float), 2097152);
  djdrigold  = (float*)libxsmm_aligned_malloc(K*K*sizeof(float), 2097152);
  djdrfgold  = (float*)libxsmm_aligned_malloc(K*K*sizeof(float), 2097152);
  djdrogold  = (float*)libxsmm_aligned_malloc(K*K*sizeof(float), 2097152);
  djdrcgold  = (float*)libxsmm_aligned_malloc(K*K*sizeof(float), 2097152);
  djdbigold  = (float*)libxsmm_aligned_malloc(K*sizeof(float), 2097152);
  djdbfgold  = (float*)libxsmm_aligned_malloc(K*sizeof(float), 2097152);
  djdbogold  = (float*)libxsmm_aligned_malloc(K*sizeof(float), 2097152);
  djdbcgold  = (float*)libxsmm_aligned_malloc(K*sizeof(float), 2097152);
  djdcsgold  = (float*)libxsmm_aligned_malloc(K*N*sizeof(float), 2097152);
  djdhpgold  = (float*)libxsmm_aligned_malloc(K*N*sizeof(float), 2097152);
  wgoldTp    = (float*)libxsmm_aligned_malloc(C*K*sizeof(float), 2097152);
  rgoldTp    = (float*)libxsmm_aligned_malloc(K*K*sizeof(float), 2097152);
  xgoldTp    = (float*)libxsmm_aligned_malloc(N*C*sizeof(float), 2097152);
  hgoldTp    = (float*)libxsmm_aligned_malloc(K*N*sizeof(float), 2097152);
  doutgoldt  = (float*)libxsmm_aligned_malloc(K*N*t*sizeof(float), 2097152);
  xt    = (float*)libxsmm_aligned_malloc(N*C*t*sizeof(float), 2097152);
  csp   = (float*)libxsmm_aligned_malloc(K*N*sizeof(float), 2097152);
  hp    = (float*)libxsmm_aligned_malloc(K*N*sizeof(float), 2097152);
  w     = (float*)libxsmm_aligned_malloc(C*K*4*sizeof(float), 2097152);
  r     = (float*)libxsmm_aligned_malloc(K*K*4*sizeof(float), 2097152);
  wt    = (float*)libxsmm_aligned_malloc(C*K*4*sizeof(float), 2097152);
  rt    = (float*)libxsmm_aligned_malloc(K*K*4*sizeof(float), 2097152);
  w_tmp = (float*)libxsmm_aligned_malloc(C*K*4*sizeof(float), 2097152);
  r_tmp = (float*)libxsmm_aligned_malloc(K*K*4*sizeof(float), 2097152);
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
  htest = (float*)libxsmm_aligned_malloc(K*N*sizeof(float), 2097152);
  djdxtestt  = (float*)libxsmm_aligned_malloc(C*N*t*sizeof(float), 2097152);
  djdwtest   = (float*)libxsmm_aligned_malloc(C*K*4*sizeof(float), 2097152);
  djdrtest   = (float*)libxsmm_aligned_malloc(K*K*4*sizeof(float), 2097152);
  djdbtest   = (float*)libxsmm_aligned_malloc(K*4*sizeof(float), 2097152);
  djdwgold4  = (float*)libxsmm_aligned_malloc(C*K*4*sizeof(float), 2097152);
  djdrgold4  = (float*)libxsmm_aligned_malloc(K*K*4*sizeof(float), 2097152);
  djdbgold4  = (float*)libxsmm_aligned_malloc(K*4*sizeof(float), 2097152);
  LIBXSMM_VLA_DECL(2, float, xgold, xgoldt, N * C);
  LIBXSMM_VLA_DECL(2, float, igold, igoldt, K * N);
  LIBXSMM_VLA_DECL(2, float, fgold, fgoldt, K * N);
  LIBXSMM_VLA_DECL(2, float, ogold, ogoldt, K * N);
  LIBXSMM_VLA_DECL(2, float, cgold, cgoldt, K * N);
  LIBXSMM_VLA_DECL(2, float, dgold, dgoldt, K * N);
  LIBXSMM_VLA_DECL(2, float, hgold, hgoldt, K * N);
  LIBXSMM_VLA_DECL(2, float, djdhgold, djdhgoldt, K * N);
  LIBXSMM_VLA_DECL(2, float, deltagold, deltagoldt, K * N);
  LIBXSMM_VLA_DECL(2, float, doutgold, doutgoldt, K * N);
  LIBXSMM_VLA_DECL(2, float, djdigold, djdigoldt, K * N);
  LIBXSMM_VLA_DECL(2, float, djdfgold, djdfgoldt, K * N);
  LIBXSMM_VLA_DECL(2, float, djdogold, djdogoldt, K * N);
  LIBXSMM_VLA_DECL(2, float, djdcgold, djdcgoldt, K * N);
  LIBXSMM_VLA_DECL(2, float, djdxgold, djdxgoldt, N * C);
  LIBXSMM_VLA_DECL(2, float, h, ht, K * N);

  /* initialize data */
  /* FWD */
  init_buf(cspgold, N*K, 0, 0);
  init_buf(hpgold, N*K, 0, 0);
  init_buf(wigold, C*K, 0, 0);
  init_buf(wfgold, C*K, 0, 0);
  init_buf(wogold, C*K, 0, 0);
  init_buf(wcgold, K*C, 0, 0);
  for (j = 0; j < t; ++j) {
    init_buf(&LIBXSMM_VLA_ACCESS(2, xgold, j, 0, N * C), N*C, 0, 0);
  }
  init_buf(rigold, K*K, 0, 0);
  init_buf(rfgold, K*K, 0, 0);
  init_buf(rogold, K*K, 0, 0);
  init_buf(rcgold, K*K, 0, 0);
  init_buf(bigold, K, 0, 0);
  init_buf(bfgold, K, 0, 0);
  init_buf(bogold, K, 0, 0);
  init_buf(bcgold, K, 0, 0);

  for (j = 0; j < K; j++) {
    bfgold_fb[j] = bfgold[j] + forget_bias;
  }
  for (j = 0; j < N; j++) {
    matrix_copy(K, bigold, &(bimgold[j*K]));
    matrix_copy(K, bfgold_fb, &(bfmgold[j*K]));
    matrix_copy(K, bogold, &(bomgold[j*K]));
    matrix_copy(K, bcgold, &(bcmgold[j*K]));
  }
  for (j = 0; j < t; ++j) {
    zero_buf(&LIBXSMM_VLA_ACCESS(2, hgold, j, 0, K * N), K*N);
    zero_buf(&LIBXSMM_VLA_ACCESS(2, igold, j, 0, K * N), K*N);
    zero_buf(&LIBXSMM_VLA_ACCESS(2, fgold, j, 0, K * N), K*N);
    zero_buf(&LIBXSMM_VLA_ACCESS(2, ogold, j, 0, K * N), K*N);
    zero_buf(&LIBXSMM_VLA_ACCESS(2, cgold, j, 0, K * N), K*N);
    zero_buf(&LIBXSMM_VLA_ACCESS(2, dgold, j, 0, K * N), K*N);
  }
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
  /* BWD/UPD */
  for (j = 0; j < t; ++j) {
    init_buf(&LIBXSMM_VLA_ACCESS(2, djdhgold, j, 0, K * N), K*N, 0, 0);
  }
  init_buf(djdcsgold, K*N, 0, 0);
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
  zero_buf(djdwigold, C*K);
  zero_buf(djdwfgold, C*K);
  zero_buf(djdwogold, C*K);
  zero_buf(djdwcgold, C*K);
  zero_buf(djdrigold, K*K);
  zero_buf(djdrfgold, K*K);
  zero_buf(djdrogold, K*K);
  zero_buf(djdrcgold, K*K);
  zero_buf(djdbigold, K);
  zero_buf(djdbfgold, K);
  zero_buf(djdbogold, K);
  zero_buf(djdbcgold, K);
  zero_buf(djdhpgold, K*N);
  zero_buf(wgoldTp, C*K);
  zero_buf(rgoldTp, K*K);
  zero_buf(xgoldTp, N*C);
  zero_buf(hgoldTp, K*N);
  zero_buf(doutgoldt, K*N*t);

  /* first touch LIBXSMM */
  zero_buf(xt,  N*C*t);
  zero_buf(csp, K*N);
  zero_buf(hp,  K*N);
  zero_buf(w,   C*K*4);
  zero_buf(r,   K*K*4);
  zero_buf(wt,  C*K*4);
  zero_buf(rt,  K*K*4);
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
    for (j = 0; j < t; ++j) {
      LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &K, &N, &C, &alpha, wigold, &K, &LIBXSMM_VLA_ACCESS(2, xgold, j, 0, N * C), &C, &beta0, i1gold, &K);
      LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &K, &N, &C, &alpha, wfgold, &K, &LIBXSMM_VLA_ACCESS(2, xgold, j, 0, N * C), &C, &beta0, f1gold, &K);
      LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &K, &N, &C, &alpha, wogold, &K, &LIBXSMM_VLA_ACCESS(2, xgold, j, 0, N * C), &C, &beta0, o1gold, &K);
      LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &K, &N, &C, &alpha, wcgold, &K, &LIBXSMM_VLA_ACCESS(2, xgold, j, 0, N * C), &C, &beta0, c1gold, &K);
      if (j == 0) {
        LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &K, &N, &K, &alpha, rigold, &K, hpgold, &K, &beta0, i2gold, &K);
        LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &K, &N, &K, &alpha, rfgold, &K, hpgold, &K, &beta0, f2gold, &K);
        LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &K, &N, &K, &alpha, rogold, &K, hpgold, &K, &beta0, o2gold, &K);
        LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &K, &N, &K, &alpha, rcgold, &K, hpgold, &K, &beta0, c2gold, &K);
      } else {
        LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &K, &N, &K, &alpha, rigold, &K, &LIBXSMM_VLA_ACCESS(2, hgold, j-1, 0, K * N), &K, &beta0, i2gold, &K);
        LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &K, &N, &K, &alpha, rfgold, &K, &LIBXSMM_VLA_ACCESS(2, hgold, j-1, 0, K * N), &K, &beta0, f2gold, &K);
        LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &K, &N, &K, &alpha, rogold, &K, &LIBXSMM_VLA_ACCESS(2, hgold, j-1, 0, K * N), &K, &beta0, o2gold, &K);
        LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &K, &N, &K, &alpha, rcgold, &K, &LIBXSMM_VLA_ACCESS(2, hgold, j-1, 0, K * N), &K, &beta0, c2gold, &K);
      }
      matrix_add(K*N, i1gold, i2gold, &LIBXSMM_VLA_ACCESS(2, igold, j, 0, K * N));
      matrix_add(K*N, &LIBXSMM_VLA_ACCESS(2, igold, j, 0, K * N), bimgold, &LIBXSMM_VLA_ACCESS(2, igold, j, 0, K * N));
      matrix_add(K*N, f1gold, f2gold, &LIBXSMM_VLA_ACCESS(2, fgold, j, 0, K * N));
      matrix_add(K*N, &LIBXSMM_VLA_ACCESS(2, fgold, j, 0, K * N), bfmgold, &LIBXSMM_VLA_ACCESS(2, fgold, j, 0, K * N));
      matrix_add(K*N, o1gold, o2gold, &LIBXSMM_VLA_ACCESS(2, ogold, j, 0, K * N));
      matrix_add(K*N, &LIBXSMM_VLA_ACCESS(2, ogold, j, 0, K * N), bomgold, &LIBXSMM_VLA_ACCESS(2, ogold, j, 0, K * N));
      matrix_add(K*N, c1gold, c2gold, &LIBXSMM_VLA_ACCESS(2, cgold, j, 0, K * N));
      matrix_add(K*N, &LIBXSMM_VLA_ACCESS(2, cgold, j, 0, K * N), bcmgold, &LIBXSMM_VLA_ACCESS(2, cgold, j, 0, K * N));
      matrix_sigmoid(K*N, &LIBXSMM_VLA_ACCESS(2, igold, j, 0, K * N), &LIBXSMM_VLA_ACCESS(2, igold, j, 0, K * N));
      matrix_sigmoid(K*N, &LIBXSMM_VLA_ACCESS(2, fgold, j, 0, K * N), &LIBXSMM_VLA_ACCESS(2, fgold, j, 0, K * N));
      matrix_sigmoid(K*N, &LIBXSMM_VLA_ACCESS(2, ogold, j, 0, K * N), &LIBXSMM_VLA_ACCESS(2, ogold, j, 0, K * N));
      matrix_tanh(K*N, &LIBXSMM_VLA_ACCESS(2, cgold, j, 0, K * N), &LIBXSMM_VLA_ACCESS(2, cgold, j, 0, K * N));
      if (j == 0) {
        matrix_eltwise_mult(K*N, &LIBXSMM_VLA_ACCESS(2, fgold, j, 0, K * N), cspgold, d1gold);
      } else {
        matrix_eltwise_mult(K*N, &LIBXSMM_VLA_ACCESS(2, fgold, j, 0, K * N), &LIBXSMM_VLA_ACCESS(2, dgold, j-1, 0, K * N), d1gold);
      }
      matrix_eltwise_mult(K*N, &LIBXSMM_VLA_ACCESS(2, igold, j, 0, K * N), &LIBXSMM_VLA_ACCESS(2, cgold, j, 0, K * N), d2gold);
      matrix_add(K*N, d1gold, d2gold, &LIBXSMM_VLA_ACCESS(2, dgold, j, 0, K * N));
      matrix_tanh(K*N, &LIBXSMM_VLA_ACCESS(2, dgold, j, 0, K * N), dhgold);
      matrix_eltwise_mult(K*N, &LIBXSMM_VLA_ACCESS(2, ogold, j, 0, K * N), dhgold, &LIBXSMM_VLA_ACCESS(2, hgold, j, 0, K * N));
    }
    /* BWD/UPD */
    for (j = t-1; j >= 0; --j) {
      /* compute deltagold */
      if (j == t-1) {
        matrix_copy(K * N, &LIBXSMM_VLA_ACCESS(2, djdhgold, t-1, 0, K * N), &LIBXSMM_VLA_ACCESS(2, deltagold, t-1, 0, K * N));
      } else {
        matrix_add(K * N, &LIBXSMM_VLA_ACCESS(2, doutgold, j, 0, K * N), &LIBXSMM_VLA_ACCESS(2, djdhgold, j, 0, K * N), &LIBXSMM_VLA_ACCESS(2, deltagold, j, 0, K * N));
      }
      /* compute djdcspgold */
      matrix_eltwise_mult(K * N, &LIBXSMM_VLA_ACCESS(2, deltagold, j, 0, K * N), &LIBXSMM_VLA_ACCESS(2, ogold, j, 0, K * N), d1gold);
      matrix_tanh_inverse(K * N, &LIBXSMM_VLA_ACCESS(2, dgold, j, 0, K * N), d2gold);
      matrix_eltwise_mult(K * N, d1gold, d2gold, d3gold);
      if (j == t-1) {
        matrix_add(K * N, d3gold, djdcsgold, djdcspgold);
      } else {
        matrix_add(K * N, d3gold, djdcspgold, djdcspgold);
      }
      /* compute djdcgold */
      matrix_eltwise_mult(K * N, djdcspgold, &LIBXSMM_VLA_ACCESS(2, igold, j, 0, K * N), c1gold);
      matrix_complement_square(K * N, &LIBXSMM_VLA_ACCESS(2, cgold, j, 0, K * N), c2gold);
      matrix_eltwise_mult(K * N, c1gold, c2gold, &LIBXSMM_VLA_ACCESS(2, djdcgold, j, 0, K * N));
      /* compute djdigold */
      matrix_eltwise_mult(K * N, djdcspgold, &LIBXSMM_VLA_ACCESS(2, cgold, j, 0, K * N), i1gold);
      matrix_complement(K * N, &LIBXSMM_VLA_ACCESS(2, igold, j, 0, K * N), i2gold);
      matrix_eltwise_mult(K * N, &LIBXSMM_VLA_ACCESS(2, igold, j, 0, K * N), i2gold, i3gold);
      matrix_eltwise_mult(K * N, i1gold, i3gold, &LIBXSMM_VLA_ACCESS(2, djdigold, j, 0, K * N));
      /* compute djdfgold */
      if (j == 0) {
        matrix_eltwise_mult(K * N, djdcspgold, cspgold, f1gold);
      } else {
        matrix_eltwise_mult(K * N, djdcspgold, &LIBXSMM_VLA_ACCESS(2, dgold, j-1, 0, K * N), f1gold);
      }
      matrix_complement(K * N, &LIBXSMM_VLA_ACCESS(2, fgold, j, 0, K * N), f2gold);
      matrix_eltwise_mult(K * N, &LIBXSMM_VLA_ACCESS(2, fgold, j, 0, K * N), f2gold, f3gold);
      matrix_eltwise_mult(K * N, f1gold, f3gold, &LIBXSMM_VLA_ACCESS(2, djdfgold, j, 0, K * N));
      /* compute djdogold */
      matrix_tanh(K * N, &LIBXSMM_VLA_ACCESS(2, dgold, j, 0, K * N), o1gold);
      matrix_eltwise_mult(K * N, &LIBXSMM_VLA_ACCESS(2, deltagold, j, 0, K * N), o1gold, o1gold);
      matrix_complement(K * N, &LIBXSMM_VLA_ACCESS(2, ogold, j, 0, K * N), o2gold);
      matrix_eltwise_mult(K * N, &LIBXSMM_VLA_ACCESS(2, ogold, j, 0, K * N), o2gold, o2gold);
      matrix_eltwise_mult(K * N, o1gold, o2gold, &LIBXSMM_VLA_ACCESS(2, djdogold, j, 0, K * N));
      /* update djdcspgold */
      matrix_eltwise_mult(K * N, djdcspgold, &LIBXSMM_VLA_ACCESS(2, fgold, j, 0, K * N), djdcspgold);
      if (j > 0) {
        /* compute doutgold */
        matrix_transpose(K, K, rigold, rgoldTp);
        LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &K, &N, &K, &alpha, rgoldTp, &K, &LIBXSMM_VLA_ACCESS(2, djdigold, j, 0, K * N), &K, &beta, &LIBXSMM_VLA_ACCESS(2, doutgold, j-1, 0, K * N), &K);
        matrix_transpose(K, K, rfgold, rgoldTp);
        LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &K, &N, &K, &alpha, rgoldTp, &K, &LIBXSMM_VLA_ACCESS(2, djdfgold, j, 0, K * N), &K, &beta, &LIBXSMM_VLA_ACCESS(2, doutgold, j-1, 0, K * N), &K);
        matrix_transpose(K, K, rogold, rgoldTp);
        LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &K, &N, &K, &alpha, rgoldTp, &K, &LIBXSMM_VLA_ACCESS(2, djdogold, j, 0, K * N), &K, &beta, &LIBXSMM_VLA_ACCESS(2, doutgold, j-1, 0, K * N), &K);
        matrix_transpose(K, K, rcgold, rgoldTp);
        LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &K, &N, &K, &alpha, rgoldTp, &K, &LIBXSMM_VLA_ACCESS(2, djdcgold, j, 0, K * N), &K, &beta, &LIBXSMM_VLA_ACCESS(2, doutgold, j-1, 0, K * N), &K);
      } else {
        /* compute djdhpgold */
        matrix_transpose(K, K, rigold, rgoldTp);
        LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &K, &N, &K, &alpha, rgoldTp, &K, &LIBXSMM_VLA_ACCESS(2, djdigold, 0, 0, K * N), &K, &beta, djdhpgold, &K);
        matrix_transpose(K, K, rfgold, rgoldTp);
        LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &K, &N, &K, &alpha, rgoldTp, &K, &LIBXSMM_VLA_ACCESS(2, djdfgold, 0, 0, K * N), &K, &beta, djdhpgold, &K);
        matrix_transpose(K, K, rogold, rgoldTp);
        LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &K, &N, &K, &alpha, rgoldTp, &K, &LIBXSMM_VLA_ACCESS(2, djdogold, 0, 0, K * N), &K, &beta, djdhpgold, &K);
        matrix_transpose(K, K, rcgold, rgoldTp);
        LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &K, &N, &K, &alpha, rgoldTp, &K, &LIBXSMM_VLA_ACCESS(2, djdcgold, 0, 0, K * N), &K, &beta, djdhpgold, &K);
      }
      if (pass == 1 || pass == 3) {
        /* compute djdxgold */
        matrix_transpose(C, K, wigold, wgoldTp);
        LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &C, &N, &K, &alpha, wgoldTp, &C, &LIBXSMM_VLA_ACCESS(2, djdigold, j, 0, K * N), &K, &beta, &LIBXSMM_VLA_ACCESS(2, djdxgold, j, 0, N * C), &C);
        matrix_transpose(C, K, wfgold, wgoldTp);
        LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &C, &N, &K, &alpha, wgoldTp, &C, &LIBXSMM_VLA_ACCESS(2, djdfgold, j, 0, K * N), &K, &beta, &LIBXSMM_VLA_ACCESS(2, djdxgold, j, 0, N * C), &C);
        matrix_transpose(C, K, wogold, wgoldTp);
        LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &C, &N, &K, &alpha, wgoldTp, &C, &LIBXSMM_VLA_ACCESS(2, djdogold, j, 0, K * N), &K, &beta, &LIBXSMM_VLA_ACCESS(2, djdxgold, j, 0, N * C), &C);
        matrix_transpose(C, K, wcgold, wgoldTp);
        LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &C, &N, &K, &alpha, wgoldTp, &C, &LIBXSMM_VLA_ACCESS(2, djdcgold, j, 0, K * N), &K, &beta, &LIBXSMM_VLA_ACCESS(2, djdxgold, j, 0, N * C), &C);
      }
      if (pass == 2 || pass == 3) {
        /* compute djdwgold */
        matrix_transpose(N, C, &LIBXSMM_VLA_ACCESS(2, xgold, j, 0, N * C), xgoldTp);
        LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &K, &C, &N, &alpha, &LIBXSMM_VLA_ACCESS(2, djdigold, j, 0, K * N), &K, xgoldTp, &N, &beta, djdwigold, &K);
        LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &K, &C, &N, &alpha, &LIBXSMM_VLA_ACCESS(2, djdfgold, j, 0, K * N), &K, xgoldTp, &N, &beta, djdwfgold, &K);
        LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &K, &C, &N, &alpha, &LIBXSMM_VLA_ACCESS(2, djdogold, j, 0, K * N), &K, xgoldTp, &N, &beta, djdwogold, &K);
        LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &K, &C, &N, &alpha, &LIBXSMM_VLA_ACCESS(2, djdcgold, j, 0, K * N), &K, xgoldTp, &N, &beta, djdwcgold, &K);

        /* compute djdrgold */
        if (j == 0) {
          matrix_transpose(N, K, hpgold, hgoldTp);
        } else {
          matrix_transpose(N, K, &LIBXSMM_VLA_ACCESS(2, hgold, j-1, 0, K * N), hgoldTp);
        }
        LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &K, &K, &N, &alpha, &LIBXSMM_VLA_ACCESS(2, djdigold, j, 0, K * N), &K, hgoldTp, &N, &beta, djdrigold, &K);
        LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &K, &K, &N, &alpha, &LIBXSMM_VLA_ACCESS(2, djdfgold, j, 0, K * N), &K, hgoldTp, &N, &beta, djdrfgold, &K);
        LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &K, &K, &N, &alpha, &LIBXSMM_VLA_ACCESS(2, djdogold, j, 0, K * N), &K, hgoldTp, &N, &beta, djdrogold, &K);
        LIBXSMM_XBLAS_SYMBOL(float)(&transa, &transb, &K, &K, &N, &alpha, &LIBXSMM_VLA_ACCESS(2, djdcgold, j, 0, K * N), &K, hgoldTp, &N, &beta, djdrcgold, &K);

        /* compute djdbgold */
        for (l = 0; l < K*N; l++) {
          djdbigold[l%K] += LIBXSMM_VLA_ACCESS(2, djdigold, j, l, K * N);
          djdbfgold[l%K] += LIBXSMM_VLA_ACCESS(2, djdfgold, j, l, K * N);
          djdbogold[l%K] += LIBXSMM_VLA_ACCESS(2, djdogold, j, l, K * N);
          djdbcgold[l%K] += LIBXSMM_VLA_ACCESS(2, djdcgold, j, l, K * N);
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

    if ( N % bn != 0 ) {
      bn = N;
    }
    if ( C % bc != 0 ) {
      bc = C;
    }
    if ( K % bk != 0 ) {
      bk = K;
    }

    /* setup LIBXSMM handle */
    lstmcell_desc.threads = nThreads;
    lstmcell_desc.N = N;
    lstmcell_desc.C = C;
    lstmcell_desc.K = K;
    lstmcell_desc.max_T = t;
    lstmcell_desc.bn = bn;
    lstmcell_desc.bk = bk;
    lstmcell_desc.bc = bc;
    lstmcell_desc.cell_type = LIBXSMM_DNN_RNNCELL_LSTM;
    lstmcell_desc.datatype_in = LIBXSMM_DNN_DATATYPE_F32;
    lstmcell_desc.datatype_out = LIBXSMM_DNN_DATATYPE_F32;
    lstmcell_desc.buffer_format = LIBXSMM_DNN_TENSOR_FORMAT_NC;
    lstmcell_desc.filter_format = LIBXSMM_DNN_TENSOR_FORMAT_CKPACKED;
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

    libxsmm_layout = libxsmm_dnn_rnncell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_RNN_REGULAR_WEIGHT_TRANS, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_weight_t = libxsmm_dnn_link_tensor( libxsmm_layout, wt, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_rnncell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_RNN_REGULAR_RECUR_WEIGHT, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_recur_weight = libxsmm_dnn_link_tensor( libxsmm_layout, r, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_rnncell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_RNN_REGULAR_RECUR_WEIGHT_TRANS, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_recur_weight_t = libxsmm_dnn_link_tensor( libxsmm_layout, rt, &status ); CHKERR_LIBXSMM_DNN( status );
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
    convert_ck_c4k_offset(C, K, 0, wigold, w_tmp);
    convert_ck_c4k_offset(C, K, 1, wcgold, w_tmp);
    convert_ck_c4k_offset(C, K, 2, wfgold, w_tmp);
    convert_ck_c4k_offset(C, K, 3, wogold, w_tmp);
    convert_ck_c4k_offset(K, K, 0, rigold, r_tmp);
    convert_ck_c4k_offset(K, K, 1, rcgold, r_tmp);
    convert_ck_c4k_offset(K, K, 2, rfgold, r_tmp);
    convert_ck_c4k_offset(K, K, 3, rogold, r_tmp);
    matrix_copy_CK_to_KCCK(w_tmp, w,  C, 4*K, bc, bk);
    matrix_copy_CK_to_KCCK(r_tmp, r,  K, 4*K, bk, bk);
    matrix_copy_CK_to_CKKC(wigold, wt,         C, K, bc, bk);
    matrix_copy_CK_to_CKKC(wcgold, wt+(C*K)  , C, K, bc, bk);
    matrix_copy_CK_to_CKKC(wfgold, wt+(2*C*K), C, K, bc, bk);
    matrix_copy_CK_to_CKKC(wogold, wt+(3*C*K), C, K, bc, bk);
    matrix_copy_CK_to_CKKC(rigold, rt,         K, K, bk, bk);
    matrix_copy_CK_to_CKKC(rcgold, rt+(K*K),   K, K, bk, bk);
    matrix_copy_CK_to_CKKC(rfgold, rt+(2*K*K), K, K, bk, bk);
    matrix_copy_CK_to_CKKC(rogold, rt+(3*K*K), K, K, bk, bk);
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
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_bind_tensor( libxsmm_handle, libxsmm_weight_t, LIBXSMM_DNN_RNN_REGULAR_WEIGHT_TRANS ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_bind_tensor( libxsmm_handle, libxsmm_recur_weight_t, LIBXSMM_DNN_RNN_REGULAR_RECUR_WEIGHT_TRANS ) );
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
      /* copy out data */
      matrix_copy(K*N, &LIBXSMM_VLA_ACCESS(2, h, t-1, 0, K * N), htest);

      /* compare */
      libxsmm_matdiff(&norms_fwd, LIBXSMM_DATATYPE_F32, K*N, 1, &LIBXSMM_VLA_ACCESS(2, hgold, t-1, 0, K * N), htest, 0, 0);
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

      /* copy out data */
      matrix_copy(N*C*t, dxt, djdxtestt);

      /* compare */
      libxsmm_matdiff(&norms_bwd, LIBXSMM_DATATYPE_F32, N*C*t, 1, djdxgoldt, djdxtestt, 0, 0);
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

      /* copy out data */
      matrix_copy_KCCK_to_CK(dw, w_tmp, C, 4*K, bc, bk);
      matrix_copy_KCCK_to_CK(dr, r_tmp, K, 4*K, bk, bk);
      convert_c4k_4ck(C, K, w_tmp, djdwtest);
      convert_c4k_4ck(K, K, r_tmp, djdrtest);
      LIBXSMM_VLA_DECL(2, float, djdb4test, djdbtest, K);
      matrix_copy(K, &(db[0]),   &LIBXSMM_VLA_ACCESS(2, djdb4test, 0, 0, K));
      matrix_copy(K, &(db[K]),   &LIBXSMM_VLA_ACCESS(2, djdb4test, 1, 0, K));
      matrix_copy(K, &(db[2*K]), &LIBXSMM_VLA_ACCESS(2, djdb4test, 2, 0, K));
      matrix_copy(K, &(db[3*K]), &LIBXSMM_VLA_ACCESS(2, djdb4test, 3, 0, K));
      LIBXSMM_VLA_DECL(2, float, djdw4, djdwgold4, C*K);
      LIBXSMM_VLA_DECL(2, float, djdr4, djdrgold4, K*K);
      LIBXSMM_VLA_DECL(2, float, djdb4, djdbgold4, K);
      matrix_copy(C*K, djdwigold, &LIBXSMM_VLA_ACCESS(2, djdw4, 0, 0, C*K));
      matrix_copy(C*K, djdwcgold, &LIBXSMM_VLA_ACCESS(2, djdw4, 1, 0, C*K));
      matrix_copy(C*K, djdwfgold, &LIBXSMM_VLA_ACCESS(2, djdw4, 2, 0, C*K));
      matrix_copy(C*K, djdwogold, &LIBXSMM_VLA_ACCESS(2, djdw4, 3, 0, C*K));
      matrix_copy(K*K, djdrigold, &LIBXSMM_VLA_ACCESS(2, djdr4, 0, 0, K*K));
      matrix_copy(K*K, djdrcgold, &LIBXSMM_VLA_ACCESS(2, djdr4, 1, 0, K*K));
      matrix_copy(K*K, djdrfgold, &LIBXSMM_VLA_ACCESS(2, djdr4, 2, 0, K*K));
      matrix_copy(K*K, djdrogold, &LIBXSMM_VLA_ACCESS(2, djdr4, 3, 0, K*K));
      matrix_copy(K, djdbigold, &LIBXSMM_VLA_ACCESS(2, djdb4, 0, 0, K));
      matrix_copy(K, djdbcgold, &LIBXSMM_VLA_ACCESS(2, djdb4, 1, 0, K));
      matrix_copy(K, djdbfgold, &LIBXSMM_VLA_ACCESS(2, djdb4, 2, 0, K));
      matrix_copy(K, djdbogold, &LIBXSMM_VLA_ACCESS(2, djdb4, 3, 0, K));

      /* compare */
      libxsmm_matdiff(&norms_upd_w, LIBXSMM_DATATYPE_F32, C*K*4, 1, djdwgold4, djdwtest, 0, 0);
      printf("Delta weight\n");
      printf("L1 reference  : %.25g\n", norms_upd_w.l1_ref);
      printf("L1 test       : %.25g\n", norms_upd_w.l1_tst);
      printf("L2 abs.error  : %.24f\n", norms_upd_w.l2_abs);
      printf("L2 rel.error  : %.24f\n", norms_upd_w.l2_rel);
      printf("Linf abs.error: %.24f\n", norms_upd_w.linf_abs);
      printf("Linf rel.error: %.24f\n", norms_upd_w.linf_rel);
      printf("Check-norm    : %.24f\n", norms_upd_w.normf_rel);
      libxsmm_matdiff_reduce(&diff, &norms_upd_w);

      libxsmm_matdiff(&norms_upd_r, LIBXSMM_DATATYPE_F32, K*K*4, 1, djdrgold4, djdrtest, 0, 0);
      printf("Delta recurrent weight\n");
      printf("L1 reference  : %.25g\n", norms_upd_r.l1_ref);
      printf("L1 test       : %.25g\n", norms_upd_r.l1_tst);
      printf("L2 abs.error  : %.24f\n", norms_upd_r.l2_abs);
      printf("L2 rel.error  : %.24f\n", norms_upd_r.l2_rel);
      printf("Linf abs.error: %.24f\n", norms_upd_r.linf_abs);
      printf("Linf rel.error: %.24f\n", norms_upd_r.linf_rel);
      printf("Check-norm    : %.24f\n", norms_upd_r.normf_rel);
      libxsmm_matdiff_reduce(&diff, &norms_upd_r);

      libxsmm_matdiff(&norms_upd_b, LIBXSMM_DATATYPE_F32, K*4, 1, djdbgold4, djdbtest, 0, 0);
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

      /* copy out data */
      matrix_copy(N*C*t, dxt, djdxtestt);
      matrix_copy_KCCK_to_CK(dw, w_tmp, C, 4*K, bc, bk);
      matrix_copy_KCCK_to_CK(dr, r_tmp, K, 4*K, bk, bk);
      convert_c4k_4ck(C, K, w_tmp, djdwtest);
      convert_c4k_4ck(K, K, r_tmp, djdrtest);
      /*LIBXSMM_VLA_DECL(2, float, djdw4test, djdwtest, C*K);*/
      /*LIBXSMM_VLA_DECL(2, float, djdr4test, djdrtest, K*K);*/
      LIBXSMM_VLA_DECL(2, float, djdb4test, djdbtest, K);
      matrix_copy(K, &(db[0]),   &LIBXSMM_VLA_ACCESS(2, djdb4test, 0, 0, K));
      matrix_copy(K, &(db[K]),   &LIBXSMM_VLA_ACCESS(2, djdb4test, 1, 0, K));
      matrix_copy(K, &(db[2*K]), &LIBXSMM_VLA_ACCESS(2, djdb4test, 2, 0, K));
      matrix_copy(K, &(db[3*K]), &LIBXSMM_VLA_ACCESS(2, djdb4test, 3, 0, K));
      LIBXSMM_VLA_DECL(2, float, djdw4, djdwgold4, C*K);
      LIBXSMM_VLA_DECL(2, float, djdr4, djdrgold4, K*K);
      LIBXSMM_VLA_DECL(2, float, djdb4, djdbgold4, K);
      matrix_copy(C*K, djdwigold, &LIBXSMM_VLA_ACCESS(2, djdw4, 0, 0, C*K));
      matrix_copy(C*K, djdwcgold, &LIBXSMM_VLA_ACCESS(2, djdw4, 1, 0, C*K));
      matrix_copy(C*K, djdwfgold, &LIBXSMM_VLA_ACCESS(2, djdw4, 2, 0, C*K));
      matrix_copy(C*K, djdwogold, &LIBXSMM_VLA_ACCESS(2, djdw4, 3, 0, C*K));
      matrix_copy(K*K, djdrigold, &LIBXSMM_VLA_ACCESS(2, djdr4, 0, 0, K*K));
      matrix_copy(K*K, djdrcgold, &LIBXSMM_VLA_ACCESS(2, djdr4, 1, 0, K*K));
      matrix_copy(K*K, djdrfgold, &LIBXSMM_VLA_ACCESS(2, djdr4, 2, 0, K*K));
      matrix_copy(K*K, djdrogold, &LIBXSMM_VLA_ACCESS(2, djdr4, 3, 0, K*K));
      matrix_copy(K, djdbigold, &LIBXSMM_VLA_ACCESS(2, djdb4, 0, 0, K));
      matrix_copy(K, djdbcgold, &LIBXSMM_VLA_ACCESS(2, djdb4, 1, 0, K));
      matrix_copy(K, djdbfgold, &LIBXSMM_VLA_ACCESS(2, djdb4, 2, 0, K));
      matrix_copy(K, djdbogold, &LIBXSMM_VLA_ACCESS(2, djdb4, 3, 0, K));

      /* compare */
      libxsmm_matdiff(&norms_bwd, LIBXSMM_DATATYPE_F32, N*C*t, 1, djdxgoldt, djdxtestt, 0, 0);
      printf("Delta input\n");
      printf("L1 reference  : %.25g\n", norms_bwd.l1_ref);
      printf("L1 test       : %.25g\n", norms_bwd.l1_tst);
      printf("L2 abs.error  : %.24f\n", norms_bwd.l2_abs);
      printf("L2 rel.error  : %.24f\n", norms_bwd.l2_rel);
      printf("Linf abs.error: %.24f\n", norms_bwd.linf_abs);
      printf("Linf rel.error: %.24f\n", norms_bwd.linf_rel);
      printf("Check-norm    : %.24f\n", norms_bwd.normf_rel);
      libxsmm_matdiff_reduce(&diff, &norms_bwd);

      libxsmm_matdiff(&norms_upd_w, LIBXSMM_DATATYPE_F32, C*K*4, 1, djdwgold4, djdwtest, 0, 0);
      printf("Delta weight\n");
      printf("L1 reference  : %.25g\n", norms_upd_w.l1_ref);
      printf("L1 test       : %.25g\n", norms_upd_w.l1_tst);
      printf("L2 abs.error  : %.24f\n", norms_upd_w.l2_abs);
      printf("L2 rel.error  : %.24f\n", norms_upd_w.l2_rel);
      printf("Linf abs.error: %.24f\n", norms_upd_w.linf_abs);
      printf("Linf rel.error: %.24f\n", norms_upd_w.linf_rel);
      printf("Check-norm    : %.24f\n", norms_upd_w.normf_rel);
      libxsmm_matdiff_reduce(&diff, &norms_upd_w);

      libxsmm_matdiff(&norms_upd_r, LIBXSMM_DATATYPE_F32, K*K*4, 1, djdrgold4, djdrtest, 0, 0);
      printf("Delta recurrent weight\n");
      printf("L1 reference  : %.25g\n", norms_upd_r.l1_ref);
      printf("L1 test       : %.25g\n", norms_upd_r.l1_tst);
      printf("L2 abs.error  : %.24f\n", norms_upd_r.l2_abs);
      printf("L2 rel.error  : %.24f\n", norms_upd_r.l2_rel);
      printf("Linf abs.error: %.24f\n", norms_upd_r.linf_abs);
      printf("Linf rel.error: %.24f\n", norms_upd_r.linf_rel);
      printf("Check-norm    : %.24f\n", norms_upd_r.normf_rel);
      libxsmm_matdiff_reduce(&diff, &norms_upd_r);

      libxsmm_matdiff(&norms_upd_b, LIBXSMM_DATATYPE_F32, K*4, 1, djdbgold4, djdbtest, 0, 0);
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
      tempflops = (8.0 * K * N * C); /* W^T * dJd{c, i, f, o} */
      tempflops += (3.0 * K * C); /* summation */
      flops += tempflops;
      tempflops = (8.0 * K * N * K); /* R^T * dJd{c, i, f, o} */
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
      tempflops = (8.0 * K * N * K); /* R^T * dJd{c, i, f, o} */
      flops += tempflops;
      flops *= t; /* for t time steps */
      tempflops = (8.0 * K * N * C); /* delta{c, i, f, o} * x^T */
      tempflops *= t; /* for t time steps */
      flops += tempflops;
      tempflops = (8.0 * K * N * K); /* delta{c, i, f, o} * delta^T */
      tempflops *= t; /* for t time steps */
      flops += tempflops;
      flops += (4.0 * K * N * t); /* delbias */
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
      tempflops = (8.0 * K * N * C); /* W^T * dJd{c, i, f, o} */
      tempflops += (3.0 * K * C); /* summation */
      flops += tempflops;
      tempflops = (8.0 * K * N * K); /* R^T * dJd{c, i, f, o} */
      flops += tempflops;
      flops *= t; /* for t time steps */
      tempflops = (8.0 * K * N * C); /* delta{c, i, f, o} * x^T */
      tempflops *= t; /* for t time steps */
      flops += tempflops;
      tempflops = (8.0 * K * N * K); /* delta{c, i, f, o} * delta^T */
      tempflops *= t; /* for t time steps */
      flops += tempflops;
      flops += (4.0 * K * N * t); /* delbias */
      flops *= iters;

      printf("GFLOP  = %.5g\n", flops*1e-9/(double)iters);
      printf("bp+wu time = %.5g\n", ((double)(l_total/iters)));
      printf("GFLOPS  = %.5g\n", (flops*1e-9)/l_total);

      printf("PERFDUMP,BP+WU,%s,%i,%i,%i,%i,%i,%.5g,%.5g\n", LIBXSMM_VERSION, nThreads, N, C, K, t, ((double)(l_total/iters)), (flops*1e-9)/l_total);
    }

    if ( pass == 4 ) {
      printf("###############################################\n");
      printf("# Performance - FWD+BWD+UPD (nc-kcck Storage) #\n");
      printf("###############################################\n");
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
      tempflops = (8.0 * K * N * C); /* W^T * dJd{c, i, f, o} */
      tempflops += (3.0 * K * C); /* summation */
      flops += tempflops;
      tempflops = (8.0 * K * N * K); /* R^T * dJd{c, i, f, o} */
      flops += tempflops;
      flops *= t; /* for t time steps */
      tempflops = (8.0 * K * N * C); /* delta{c, i, f, o} * x^T */
      tempflops *= t; /* for t time steps */
      flops += tempflops;
      tempflops = (8.0 * K * N * K); /* delta{c, i, f, o} * delta^T */
      tempflops *= t; /* for t time steps */
      flops += tempflops;
      flops += (4.0 * K * N * t); /* delbias */
      flops += (((2.0 * K * N * C) + (2.0 * K * N * K) + (2.0 * K * N) + (tflops * K * N)) * 4.0 + (4.0 * K * N) + (tflops * K * N)) * (double)t;
      flops *= iters;

      printf("GFLOP  = %.5g\n", flops*1e-9/(double)iters);
      printf("fp+bp+wu time = %.5g\n", ((double)(l_total/iters)));
      printf("GFLOPS  = %.5g\n", (flops*1e-9)/l_total);

      printf("PERFDUMP,FP+BP+WU,%s,%i,%i,%i,%i,%i,%.5g,%.5g\n", LIBXSMM_VERSION, nThreads, N, C, K, t, ((double)(l_total/iters)), (flops*1e-9)/l_total);
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
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_release_tensor( libxsmm_handle, LIBXSMM_DNN_RNN_REGULAR_WEIGHT_TRANS ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_release_tensor( libxsmm_handle, LIBXSMM_DNN_RNN_REGULAR_RECUR_WEIGHT_TRANS ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_release_tensor( libxsmm_handle, LIBXSMM_DNN_RNN_REGULAR_BIAS ) );
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
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_weight_t ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_recur_weight_t ) );
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
  libxsmm_free(hgoldt);
  libxsmm_free(bimgold);
  libxsmm_free(bfmgold);
  libxsmm_free(bomgold);
  libxsmm_free(bcmgold);
  libxsmm_free(bfgold_fb);
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
  libxsmm_free(djdcspgold);
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
  libxsmm_free(djdhpgold);
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
  libxsmm_free(wt);
  libxsmm_free(rt);
  libxsmm_free(w_tmp);
  libxsmm_free(r_tmp);
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
  libxsmm_free(htest);
  libxsmm_free(djdxtestt);
  libxsmm_free(djdwtest);
  libxsmm_free(djdrtest);
  libxsmm_free(djdbtest);
  libxsmm_free(djdwgold4);
  libxsmm_free(djdrgold4);
  libxsmm_free(djdbgold4);
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

  return global_status;
}

