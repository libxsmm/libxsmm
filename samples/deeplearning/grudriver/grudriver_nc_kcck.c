/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
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

/* include c-based dnn library */
#include "../common/dnn_common.h"

#define CHKERR_LIBXSMM_DNN(A) { const int chkerr_libxsmm_dnn_ = A; if (LIBXSMM_DNN_SUCCESS != chkerr_libxsmm_dnn_) { \
  fprintf(stderr, "%s\n", libxsmm_dnn_get_error(chkerr_libxsmm_dnn_)); global_status = chkerr_libxsmm_dnn_; } \
}

int main(int argc, char* argv[])
{
  float *wigold, *wcgold, *wfgold, *rigold, *rcgold, *rfgold, *bigold, *bcgold, *bfgold;
  float *xgoldt, *hpgold, *hgoldt;
  float *dwgold, *drgold, *dbgold;
  float *dxgoldt, *dhpgold, *dhgoldt;
  float *igoldt, *cgoldt, *fgoldt, *ogoldt;
  float *xt, *hp, *w, *r, *b, *ht;
  float *it, *ct, *ft, *ot;
  float *dxt, *dhp, *dw, *dr, *db, *dht;
  float *scratch_bu, *dwtest, *drtest, *w_tmp, *r_tmp;

  void *scratch, *internalstate;
  size_t scratch_size = 0, internalstate_size = 0;

  int iters = 10;   /* repetitions of benchmark */
  int pass = 0;     /* pass: 0--FWD, 1--BWD, 2--UPD, 3--BWD+UPD */
  int N = 168;      /* size of mini-batch */
  int C = 512;      /* number of inputs */
  int K = 256;      /* number of outputs */
  int t = 50;       /* number of time steps (>= 1) */
  int bn = 24;
  int bc = 64;
  int bk = 64;

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
  int j;

  libxsmm_dnn_rnncell_desc grucell_desc;
  libxsmm_dnn_rnncell* libxsmm_handle;
  libxsmm_dnn_tensor* libxsmm_input;
  libxsmm_dnn_tensor* libxsmm_hidden_state_prev;
  libxsmm_dnn_tensor* libxsmm_weight;
  libxsmm_dnn_tensor* libxsmm_recur_weight;
  libxsmm_dnn_tensor* libxsmm_bias;
  libxsmm_dnn_tensor* libxsmm_hidden_state;
  libxsmm_dnn_tensor* libxsmm_i;
  libxsmm_dnn_tensor* libxsmm_c;
  libxsmm_dnn_tensor* libxsmm_f;
  libxsmm_dnn_tensor* libxsmm_o;
  libxsmm_dnn_tensor* libxsmm_dinput;
  libxsmm_dnn_tensor* libxsmm_dhidden_state_prev;
  libxsmm_dnn_tensor* libxsmm_dweight;
  libxsmm_dnn_tensor* libxsmm_drecur_weight;
  libxsmm_dnn_tensor* libxsmm_dbias;
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
    printf("\nUsage: ./grudriver [reps] [pass: 0--FWD, 1--BWD, 2--UPD, 3--BWD+UPD] [N] [C] [K] [time_steps > 0]\n\n");
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
  if (argc > j) bn    = atoi(argv[j++]);
  if (argc > j) bc    = atoi(argv[j++]);
  if (argc > j) bk    = atoi(argv[j++]);

  if (t <= 0) {
    printf("time_steps %d should be greater than or equal to 1\n\n", t);
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
  xgoldt     = (float*)libxsmm_aligned_malloc(N*C*t*sizeof(float), 2097152);
  hpgold     = (float*)libxsmm_aligned_malloc(K*N*sizeof(float),   2097152);
  wigold     = (float*)libxsmm_aligned_malloc(C*K*sizeof(float),   2097152);
  wcgold     = (float*)libxsmm_aligned_malloc(C*K*sizeof(float),   2097152);
  wfgold     = (float*)libxsmm_aligned_malloc(C*K*sizeof(float),   2097152);
  rigold     = (float*)libxsmm_aligned_malloc(K*K*sizeof(float),   2097152);
  rcgold     = (float*)libxsmm_aligned_malloc(K*K*sizeof(float),   2097152);
  rfgold     = (float*)libxsmm_aligned_malloc(K*K*sizeof(float),   2097152);
  bigold     = (float*)libxsmm_aligned_malloc(K*sizeof(float),     2097152);
  bcgold     = (float*)libxsmm_aligned_malloc(K*sizeof(float),     2097152);
  bfgold     = (float*)libxsmm_aligned_malloc(K*sizeof(float),     2097152);
  hgoldt     = (float*)libxsmm_aligned_malloc(K*N*t*sizeof(float), 2097152);
  igoldt     = (float*)libxsmm_aligned_malloc(K*N*t*sizeof(float), 2097152);
  cgoldt     = (float*)libxsmm_aligned_malloc(K*N*t*sizeof(float), 2097152);
  fgoldt     = (float*)libxsmm_aligned_malloc(K*N*t*sizeof(float), 2097152);
  ogoldt     = (float*)libxsmm_aligned_malloc(K*N*t*sizeof(float), 2097152);
  dxgoldt    = (float*)libxsmm_aligned_malloc(N*C*t*sizeof(float), 2097152);
  dhpgold    = (float*)libxsmm_aligned_malloc(K*N*sizeof(float),   2097152);
  dwgold     = (float*)libxsmm_aligned_malloc(C*K*3*sizeof(float), 2097152);
  drgold     = (float*)libxsmm_aligned_malloc(K*K*3*sizeof(float), 2097152);
  dbgold     = (float*)libxsmm_aligned_malloc(K*3*sizeof(float),   2097152);
  dhgoldt    = (float*)libxsmm_aligned_malloc(K*N*t*sizeof(float), 2097152);
  scratch_bu = (float*)libxsmm_aligned_malloc(K*N*6*sizeof(float), 2097152);
  xt         = (float*)libxsmm_aligned_malloc(N*C*t*sizeof(float), 2097152);
  hp         = (float*)libxsmm_aligned_malloc(K*N*sizeof(float),   2097152);
  w          = (float*)libxsmm_aligned_malloc(C*K*3*sizeof(float), 2097152);
  r          = (float*)libxsmm_aligned_malloc(K*K*3*sizeof(float), 2097152);
  b          = (float*)libxsmm_aligned_malloc(K*3*sizeof(float),   2097152);
  ht         = (float*)libxsmm_aligned_malloc(K*N*t*sizeof(float), 2097152);
  it         = (float*)libxsmm_aligned_malloc(K*N*t*sizeof(float), 2097152);
  ct         = (float*)libxsmm_aligned_malloc(K*N*t*sizeof(float), 2097152);
  ft         = (float*)libxsmm_aligned_malloc(K*N*t*sizeof(float), 2097152);
  ot         = (float*)libxsmm_aligned_malloc(K*N*t*sizeof(float), 2097152);
  dxt        = (float*)libxsmm_aligned_malloc(N*C*t*sizeof(float), 2097152);
  dhp        = (float*)libxsmm_aligned_malloc(K*N*sizeof(float),   2097152);
  dw         = (float*)libxsmm_aligned_malloc(C*K*3*sizeof(float), 2097152);
  dr         = (float*)libxsmm_aligned_malloc(K*K*3*sizeof(float), 2097152);
  db         = (float*)libxsmm_aligned_malloc(K*3*sizeof(float),   2097152);
  dht        = (float*)libxsmm_aligned_malloc(K*N*t*sizeof(float), 2097152);
  dwtest     = (float*)libxsmm_aligned_malloc(C*K*3*sizeof(float), 2097152);
  drtest     = (float*)libxsmm_aligned_malloc(K*K*3*sizeof(float), 2097152);
  w_tmp      = (float*)libxsmm_aligned_malloc(C*K*3*sizeof(float), 2097152);
  r_tmp      = (float*)libxsmm_aligned_malloc(K*K*3*sizeof(float), 2097152);
  LIBXSMM_VLA_DECL(2, float, xgold, xgoldt, N * C);
  LIBXSMM_VLA_DECL(2, float, hgold, hgoldt, N * K);
  /*LIBXSMM_VLA_DECL(2, float, igold, igoldt, N * K);*/
  /*LIBXSMM_VLA_DECL(2, float, cgold, cgoldt, N * K);*/
  /*LIBXSMM_VLA_DECL(2, float, fgold, fgoldt, N * K);*/
  /*LIBXSMM_VLA_DECL(2, float, ogold, ogoldt, N * K);*/
  /*LIBXSMM_VLA_DECL(2, float, dxgold, dxgoldt, N * C);*/
  LIBXSMM_VLA_DECL(2, float, dhgold, dhgoldt, N * K);
  LIBXSMM_VLA_DECL(2, float, h, ht, N * K);

  /* initialize data */
  /* FWD */
  for (j = 0; j < t; ++j) {
    LIBXSMM_MATINIT_OMP(float, 24, &LIBXSMM_VLA_ACCESS(2, xgold, j, 0, N * C), N, C, N, 1.0);
  }
  LIBXSMM_MATINIT_OMP(float, 24, hpgold, N, K, N, 1.0);
  LIBXSMM_MATINIT_OMP(float, 42, wigold, C, K, C, 1.0);
  LIBXSMM_MATINIT_OMP(float, 42, wcgold, C, K, C, 1.0);
  LIBXSMM_MATINIT_OMP(float, 42, wfgold, C, K, C, 1.0);
  LIBXSMM_MATINIT_OMP(float, 42, rigold, K, K, K, 1.0);
  LIBXSMM_MATINIT_OMP(float, 42, rcgold, K, K, K, 1.0);
  LIBXSMM_MATINIT_OMP(float, 42, rfgold, K, K, K, 1.0);
  LIBXSMM_MATINIT_OMP(float, 24, bigold, 1, K, 1, 1.0);
  LIBXSMM_MATINIT_OMP(float, 24, bcgold, 1, K, 1, 1.0);
  LIBXSMM_MATINIT_OMP(float, 24, bfgold, 1, K, 1, 1.0);
  zero_buf(hgoldt, N*K*t);
  /* BWD/UPD */
  for (j = 0; j < t; ++j) {
    LIBXSMM_MATINIT_OMP(float, 24, &LIBXSMM_VLA_ACCESS(2, dhgold, j, 0, K * N), N, K, N, 1.0);
  }
  zero_buf(dxgoldt, N*C*t);
  zero_buf(dhpgold, K*N);
  zero_buf(dwgold,  C*K*3);
  zero_buf(drgold,  K*K*3);
  zero_buf(dbgold,  K*3);

  /* first touch LIBXSMM */
  zero_buf(xt,  N*C*t);
  zero_buf(hp,  K*N);
  zero_buf(w,   C*K*3);
  zero_buf(r,   K*K*3);
  zero_buf(b,   K*3);
  zero_buf(ht,  N*K*t);
  zero_buf(it,  K*N*t);
  zero_buf(ct,  K*N*t);
  zero_buf(ft,  K*N*t);
  zero_buf(ot,  K*N*t);
  zero_buf(dxt, N*C*t);
  zero_buf(dhp, K*N);
  zero_buf(dw,  C*K*3);
  zero_buf(dr,  K*K*3);
  zero_buf(db,  K*3);
  zero_buf(dht, K*N*t);
  zero_buf(w_tmp, C*K*3);
  zero_buf(r_tmp, K*K*3);

  if (LIBXSMM_NEQ(0, check)) {
    printf("##########################################\n");
    printf("#         Computing Reference ...        #\n");
    printf("##########################################\n");

    gru_ref_fwd( N, C, K, t,
                 wigold, wcgold, wfgold,
                 rigold, rcgold, rfgold,
                 bigold, bcgold, bfgold,
                 xgoldt, hpgold, hgoldt,
                 igoldt, cgoldt, fgoldt, ogoldt );

    gru_ref_bwd_upd( N, C, K, t,
                     xgoldt, hpgold, hgoldt,
                     igoldt, cgoldt, fgoldt, ogoldt,
                     wigold, wcgold, wfgold,
                     rigold, rcgold, rfgold,
                     dhgoldt, dwgold, drgold, dbgold,
                     dxgoldt, dhpgold, scratch_bu );

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
    grucell_desc.threads = nThreads;
    grucell_desc.N = N;
    grucell_desc.C = C;
    grucell_desc.K = K;
    grucell_desc.max_T = t;
    grucell_desc.bn = bn;
    grucell_desc.bc = bc;
    grucell_desc.bk = bk;
    grucell_desc.cell_type = LIBXSMM_DNN_RNNCELL_GRU;
    grucell_desc.datatype_in = LIBXSMM_DNN_DATATYPE_F32;
    grucell_desc.datatype_out = LIBXSMM_DNN_DATATYPE_F32;
    grucell_desc.buffer_format = LIBXSMM_DNN_TENSOR_FORMAT_NC;
    grucell_desc.filter_format = LIBXSMM_DNN_TENSOR_FORMAT_CKPACKED;

    libxsmm_handle = libxsmm_dnn_create_rnncell( grucell_desc, &status );
    CHKERR_LIBXSMM_DNN( status );

    /* setup LIBXSMM buffers and filter */
    libxsmm_layout = libxsmm_dnn_rnncell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_RNN_REGULAR_INPUT, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_input = libxsmm_dnn_link_tensor( libxsmm_layout, xt, &status ); CHKERR_LIBXSMM_DNN( status );
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

    libxsmm_layout = libxsmm_dnn_rnncell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_RNN_REGULAR_HIDDEN_STATE, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_hidden_state = libxsmm_dnn_link_tensor( libxsmm_layout, ht, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_rnncell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_RNN_INTERNAL_I, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_i = libxsmm_dnn_link_tensor( libxsmm_layout, it, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_rnncell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_RNN_INTERNAL_CI, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_c = libxsmm_dnn_link_tensor( libxsmm_layout, ct, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_rnncell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_RNN_INTERNAL_F, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_f = libxsmm_dnn_link_tensor( libxsmm_layout, ft, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_rnncell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_RNN_INTERNAL_O, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_o = libxsmm_dnn_link_tensor( libxsmm_layout, ot, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_rnncell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_RNN_GRADIENT_INPUT, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dinput = libxsmm_dnn_link_tensor( libxsmm_layout, dxt, &status ); CHKERR_LIBXSMM_DNN( status );
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

    libxsmm_layout = libxsmm_dnn_rnncell_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_RNN_GRADIENT_HIDDEN_STATE, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dhidden_state = libxsmm_dnn_link_tensor( libxsmm_layout, dht, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    /* copy in data to LIBXSMM format */
    matrix_copy(N*C*t, xgoldt, xt);
    matrix_copy(K*N, hpgold, hp);
    convert_ck_c3k(C, K, wigold, w_tmp);
    convert_ck_c3k(C, K, wcgold, &(w_tmp[K]));
    convert_ck_c3k(C, K, wfgold, &(w_tmp[2*K]));
    convert_ck_c3k(K, K, rigold, r_tmp);
    convert_ck_c3k(K, K, rcgold, &(r_tmp[K]));
    convert_ck_c3k(K, K, rfgold, &(r_tmp[2*K]));
    matrix_copy_CK_to_KCCK(w_tmp, w, C, 3*K, bc, bk);
    matrix_copy_CK_to_KCCK(r_tmp, r, K, 3*K, bk, bk);
    matrix_copy(K, bigold, b);
    matrix_copy(K, bcgold, &(b[K]));
    matrix_copy(K, bfgold, &(b[2*K]));
    matrix_copy(K*N*t, dhgoldt, dht);

    /* bind buffers and filter to handle */
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_bind_tensor( libxsmm_handle, libxsmm_input, LIBXSMM_DNN_RNN_REGULAR_INPUT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_bind_tensor( libxsmm_handle, libxsmm_hidden_state_prev, LIBXSMM_DNN_RNN_REGULAR_HIDDEN_STATE_PREV ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_bind_tensor( libxsmm_handle, libxsmm_weight, LIBXSMM_DNN_RNN_REGULAR_WEIGHT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_bind_tensor( libxsmm_handle, libxsmm_recur_weight, LIBXSMM_DNN_RNN_REGULAR_RECUR_WEIGHT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_bind_tensor( libxsmm_handle, libxsmm_bias, LIBXSMM_DNN_RNN_REGULAR_BIAS ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_bind_tensor( libxsmm_handle, libxsmm_hidden_state, LIBXSMM_DNN_RNN_REGULAR_HIDDEN_STATE ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_bind_tensor( libxsmm_handle, libxsmm_i, LIBXSMM_DNN_RNN_INTERNAL_I ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_bind_tensor( libxsmm_handle, libxsmm_c, LIBXSMM_DNN_RNN_INTERNAL_CI ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_bind_tensor( libxsmm_handle, libxsmm_f, LIBXSMM_DNN_RNN_INTERNAL_F ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_bind_tensor( libxsmm_handle, libxsmm_o, LIBXSMM_DNN_RNN_INTERNAL_O ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_bind_tensor( libxsmm_handle, libxsmm_dinput, LIBXSMM_DNN_RNN_GRADIENT_INPUT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_bind_tensor( libxsmm_handle, libxsmm_dhidden_state_prev, LIBXSMM_DNN_RNN_GRADIENT_HIDDEN_STATE_PREV ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_bind_tensor( libxsmm_handle, libxsmm_dweight, LIBXSMM_DNN_RNN_GRADIENT_WEIGHT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_bind_tensor( libxsmm_handle, libxsmm_drecur_weight, LIBXSMM_DNN_RNN_GRADIENT_RECUR_WEIGHT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_bind_tensor( libxsmm_handle, libxsmm_dbias, LIBXSMM_DNN_RNN_GRADIENT_BIAS ) );
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
      internalstate = libxsmm_aligned_malloc( internalstate_size, 2097152 );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_bind_internalstate( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_FWD, internalstate ) );
    } else {
      internalstate_size = libxsmm_dnn_rnncell_get_internalstate_size( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_ALL, &status );
      CHKERR_LIBXSMM_DNN( status );
      internalstate = libxsmm_aligned_malloc( internalstate_size, 2097152 );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_bind_internalstate( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_ALL, internalstate ) );
    }
    zero_buf( (float*)internalstate, internalstate_size/4 );

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
      /* We need to always run FWD pass once to populate i, c, f, o, h */
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
        CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_BWD, 0, tid ) );
      }

      /* compare */
      libxsmm_matdiff(&norms_bwd, LIBXSMM_DATATYPE_F32, N*C*t, 1, dxgoldt, dxt, 0, 0);
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
        CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_UPD, 0, tid ) );
      }

      matrix_copy_KCCK_to_CK(dw, w_tmp, C, 3*K, bc, bk);
      matrix_copy_KCCK_to_CK(dr, r_tmp, K, 3*K, bk, bk);
      /*
      convert_c3k_3ck(C, K, w_tmp, dwtest);
      convert_c3k_3ck(K, K, r_tmp, drtest);
      */
      convert_ck_c3k(C, K, &(dwgold[0]),     &(dwtest[0]));
      convert_ck_c3k(C, K, &(dwgold[C*K]),   &(dwtest[K]));
      convert_ck_c3k(C, K, &(dwgold[2*C*K]), &(dwtest[2*K]));
      convert_ck_c3k(K, K, &(drgold[0]),     &(drtest[0]));
      convert_ck_c3k(K, K, &(drgold[K*K]),   &(drtest[K]));
      convert_ck_c3k(K, K, &(drgold[2*K*K]), &(drtest[2*K]));
      /* compare */
      /*libxsmm_matdiff(&norms_upd_w, LIBXSMM_DATATYPE_F32, C*K*3, 1, dwtest, dw, 0, 0);*/
      libxsmm_matdiff(&norms_upd_w, LIBXSMM_DATATYPE_F32, C*K*3, 1, dwtest, w_tmp, 0, 0);
      printf("Delta weight\n");
      printf("L1 reference  : %.25g\n", norms_upd_w.l1_ref);
      printf("L1 test       : %.25g\n", norms_upd_w.l1_tst);
      printf("L2 abs.error  : %.24f\n", norms_upd_w.l2_abs);
      printf("L2 rel.error  : %.24f\n", norms_upd_w.l2_rel);
      printf("Linf abs.error: %.24f\n", norms_upd_w.linf_abs);
      printf("Linf rel.error: %.24f\n", norms_upd_w.linf_rel);
      printf("Check-norm    : %.24f\n", norms_upd_w.normf_rel);
      libxsmm_matdiff_reduce(&diff, &norms_upd_w);

      /*libxsmm_matdiff(&norms_upd_r, LIBXSMM_DATATYPE_F32, K*K*3, 1, drtest, dr, 0, 0);*/
      libxsmm_matdiff(&norms_upd_r, LIBXSMM_DATATYPE_F32, K*K*3, 1, drtest, r_tmp, 0, 0);
      printf("Delta recurrent weight\n");
      printf("L1 reference  : %.25g\n", norms_upd_r.l1_ref);
      printf("L1 test       : %.25g\n", norms_upd_r.l1_tst);
      printf("L2 abs.error  : %.24f\n", norms_upd_r.l2_abs);
      printf("L2 rel.error  : %.24f\n", norms_upd_r.l2_rel);
      printf("Linf abs.error: %.24f\n", norms_upd_r.linf_abs);
      printf("Linf rel.error: %.24f\n", norms_upd_r.linf_rel);
      printf("Check-norm    : %.24f\n", norms_upd_r.normf_rel);
      libxsmm_matdiff_reduce(&diff, &norms_upd_r);

      libxsmm_matdiff(&norms_upd_b, LIBXSMM_DATATYPE_F32, K*3, 1, dbgold, db, 0, 0);
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
        CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_BWDUPD, 0, tid ) );
      }

      matrix_copy_KCCK_to_CK(dw, w_tmp, C, 3*K, bc, bk);
      matrix_copy_KCCK_to_CK(dr, r_tmp, K, 3*K, bk, bk);
      convert_ck_c3k(C, K, &(dwgold[0]),     &(dwtest[0]));
      convert_ck_c3k(C, K, &(dwgold[C*K]),   &(dwtest[K]));
      convert_ck_c3k(C, K, &(dwgold[2*C*K]), &(dwtest[2*K]));
      convert_ck_c3k(K, K, &(drgold[0]),     &(drtest[0]));
      convert_ck_c3k(K, K, &(drgold[K*K]),   &(drtest[K]));
      convert_ck_c3k(K, K, &(drgold[2*K*K]), &(drtest[2*K]));
      /* compare */
      libxsmm_matdiff(&norms_bwd, LIBXSMM_DATATYPE_F32, N*C*t, 1, dxgoldt, dxt, 0, 0);
      printf("Delta input\n");
      printf("L1 reference  : %.25g\n", norms_bwd.l1_ref);
      printf("L1 test       : %.25g\n", norms_bwd.l1_tst);
      printf("L2 abs.error  : %.24f\n", norms_bwd.l2_abs);
      printf("L2 rel.error  : %.24f\n", norms_bwd.l2_rel);
      printf("Linf abs.error: %.24f\n", norms_bwd.linf_abs);
      printf("Linf rel.error: %.24f\n", norms_bwd.linf_rel);
      printf("Check-norm    : %.24f\n", norms_bwd.normf_rel);
      libxsmm_matdiff_reduce(&diff, &norms_bwd);

      /*libxsmm_matdiff(&norms_upd_w, LIBXSMM_DATATYPE_F32, C*K*3, 1, dwtest, dw, 0, 0);*/
      libxsmm_matdiff(&norms_upd_w, LIBXSMM_DATATYPE_F32, C*K*3, 1, dwtest, w_tmp, 0, 0);
      printf("Delta weight\n");
      printf("L1 reference  : %.25g\n", norms_upd_w.l1_ref);
      printf("L1 test       : %.25g\n", norms_upd_w.l1_tst);
      printf("L2 abs.error  : %.24f\n", norms_upd_w.l2_abs);
      printf("L2 rel.error  : %.24f\n", norms_upd_w.l2_rel);
      printf("Linf abs.error: %.24f\n", norms_upd_w.linf_abs);
      printf("Linf rel.error: %.24f\n", norms_upd_w.linf_rel);
      printf("Check-norm    : %.24f\n", norms_upd_w.normf_rel);
      libxsmm_matdiff_reduce(&diff, &norms_upd_w);

      /*libxsmm_matdiff(&norms_upd_r, LIBXSMM_DATATYPE_F32, K*K*3, 1, drtest, dr, 0, 0);*/
      libxsmm_matdiff(&norms_upd_r, LIBXSMM_DATATYPE_F32, K*K*3, 1, drtest, r_tmp, 0, 0);
      printf("Delta recurrent weight\n");
      printf("L1 reference  : %.25g\n", norms_upd_r.l1_ref);
      printf("L1 test       : %.25g\n", norms_upd_r.l1_tst);
      printf("L2 abs.error  : %.24f\n", norms_upd_r.l2_abs);
      printf("L2 rel.error  : %.24f\n", norms_upd_r.l2_rel);
      printf("Linf abs.error: %.24f\n", norms_upd_r.linf_abs);
      printf("Linf rel.error: %.24f\n", norms_upd_r.linf_rel);
      printf("Check-norm    : %.24f\n", norms_upd_r.normf_rel);
      libxsmm_matdiff_reduce(&diff, &norms_upd_r);

      libxsmm_matdiff(&norms_upd_b, LIBXSMM_DATATYPE_F32, K*3, 1, dbgold, db, 0, 0);
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
      flops = (((2.0 * K * N * C) + (2.0 * K * N * K) + (2.0 * K * N) + (tflops * K * N)) * 2.0 + (K * N) + (2.0 * K * N * C) + (2.0 * K * N * K) + (tflops * K * N) + 4.0 * (K * N)) * (double)t * (double)iters;

      printf("GFLOP  = %.5g\n", flops*1e-9/(double)iters);
      printf("fp time = %.5g\n", ((double)(l_total/iters)));
      printf("GFLOPS  = %.5g\n", (flops*1e-9)/l_total);

      printf("PERFDUMP,FP,%s,%i,%i,%i,%i,%i,%i,%i,%i,%.5g,%.5g\n", LIBXSMM_VERSION, nThreads, N, C, K, t, bn, bc, bk, ((double)(l_total/iters)), (flops*1e-9)/l_total);
    }

    if ( pass == 1 ) {
      printf("##########################################\n");
      printf("#   Performance - BWD (custom-Storage)   #\n");
      printf("##########################################\n");
      /* run LIBXSMM GRU for performance */
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
      flops = K * N; /* d3 = djdh + d23 (delta) */
      flops += 2.0 * K * N; /* d4 = (1 - z).d3 */
      flops += K * N; /* d5 = d3.h */
      flops += K * N; /* d6 = -d5 */
      flops += K * N; /* d7 = d3.g */
      flops += K * N; /* d8 = d3.z */
      flops += K * N; /* d9 = d7 + d8 */
      flops += 3.0 * K * N; /* d10 = d8.tanh'(g) */
      flops += 3.0 * K * N; /* d11 = d9.sig'(z) */
      flops += (2.0 * K * K * N + K * K); /* d13 = Wg^T * d10 (including transpose) */
      flops += (2.0 * K * K * N + K * K); /* d15 = Wz^T * d11 (including transpose) */
      flops += K * N; /* d16 = d13.z */
      flops += K * N; /* d17 = d13.r */
      flops += 3.0 * K * N; /* d18 = d16.sig'(r) */
      flops += K * N; /* d19 = d17 + d4 */
      flops += (2.0 * K * K * N + K * K); /* d21 = Wr^T * d18 (including transpose) */
      flops += K * N; /* d22 = d21 + d15 */
      flops += K * N; /* d23 = d19 + d22 */
      flops += (2.0 * K * C * N + K * C); /* d12 = Ug^T * d10 (including transpose) */
      flops += (2.0 * K * C * N + K * C); /* d14 = Uz^T * d11 (including transpose) */
      flops += (2.0 * K * C * N + K * C); /* d20 = Ur^T * d18 (including transpose) */
      flops += 2.0 * K * N; /* djdx = d12 + d14 + d20 */
      flops *= t; /* for t time steps */
      flops *= iters;

      printf("GFLOP  = %.5g\n", flops*1e-9/(double)iters);
      printf("bp time = %.5g\n", ((double)(l_total/iters)));
      printf("GFLOPS  = %.5g\n", (flops*1e-9)/l_total);

      printf("PERFDUMP,BP,%s,%i,%i,%i,%i,%i,%i,%i,%i,%.5g,%.5g\n", LIBXSMM_VERSION, nThreads, N, C, K, t, bn, bc, bk, ((double)(l_total/iters)), (flops*1e-9)/l_total);
    }

    if ( pass == 2 ) {
      printf("##########################################\n");
      printf("#   Performance - UPD (custom-Storage)   #\n");
      printf("##########################################\n");
      /* run LIBXSMM GRU for performance */
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
      flops = K * N; /* d3 = djdh + d23 (delta) */
      flops += 2.0 * K * N; /* d4 = (1 - z).d3 */
      flops += K * N; /* d5 = d3.h */
      flops += K * N; /* d6 = -d5 */
      flops += K * N; /* d7 = d3.g */
      flops += K * N; /* d8 = d3.z */
      flops += K * N; /* d9 = d7 + d8 */
      flops += 3.0 * K * N; /* d10 = d8.tanh'(g) */
      flops += 3.0 * K * N; /* d11 = d9.sig'(z) */
      flops += (2.0 * K * K * N + K * K); /* d13 = Wg^T * d10 (including transpose) */
      flops += (2.0 * K * K * N + K * K); /* d15 = Wz^T * d11 (including transpose) */
      flops += K * N; /* d16 = d13.z */
      flops += K * N; /* d17 = d13.r */
      flops += 3.0 * K * N; /* d18 = d16.sig'(r) */
      flops += K * N; /* d19 = d17 + d4 */
      flops += (2.0 * K * K * N + K * K); /* d21 = Wr^T * d18 (including transpose) */
      flops += K * N; /* d22 = d21 + d15 */
      flops += K * N; /* d23 = d19 + d22 */
      flops += (2.0 * K * N * K + K * N + K * K); /* djdwr = djdwr + d18 * h^T */
      flops += (2.0 * K * N * K + K * N + K * K); /* djdwz = djdwz + d11 * h^T */
      flops += (2.0 * K * N * K + 2.0 * K * N + K * K); /* djdwg = djdwg + d10 * (h.r)^T */
      flops += (2.0 * K * N * C + C * N + K * C); /* djdur = djdur + d18 * x^T */
      flops += (2.0 * K * N * C + C * N + K * C); /* djduz = djduz + d11 * x^T */
      flops += (2.0 * K * N * C + C * N + K * C); /* djdug = djdug + d10 * x^T */
      flops += K * N; /* djdbr = djdbr + d18 */
      flops += K * N; /* djdbz = djdbz + d11 */
      flops += K * N; /* djdbg = djdbg + d10 */
      flops *= t; /* for t time steps */
      flops *= iters;

      printf("GFLOP  = %.5g\n", flops*1e-9/(double)iters);
      printf("wu time = %.5g\n", ((double)(l_total/iters)));
      printf("GFLOPS  = %.5g\n", (flops*1e-9)/l_total);

      printf("PERFDUMP,WU,%s,%i,%i,%i,%i,%i,%i,%i,%i,%.5g,%.5g\n", LIBXSMM_VERSION, nThreads, N, C, K, t, bn, bc, bk, ((double)(l_total/iters)), (flops*1e-9)/l_total);
    }

    if ( pass == 3 ) {
      printf("##########################################\n");
      printf("# Performance - BWD+UPD (custom-Storage) #\n");
      printf("##########################################\n");
      /* run LIBXSMM GRU for performance */
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
      flops = K * N; /* d3 = djdh + d23 (delta) */
      flops += 2.0 * K * N; /* d4 = (1 - z).d3 */
      flops += K * N; /* d5 = d3.h */
      flops += K * N; /* d6 = -d5 */
      flops += K * N; /* d7 = d3.g */
      flops += K * N; /* d8 = d3.z */
      flops += K * N; /* d9 = d7 + d8 */
      flops += 3.0 * K * N; /* d10 = d8.tanh'(g) */
      flops += 3.0 * K * N; /* d11 = d9.sig'(z) */
      flops += (2.0 * K * K * N + K * K); /* d13 = Wg^T * d10 (including transpose) */
      flops += (2.0 * K * K * N + K * K); /* d15 = Wz^T * d11 (including transpose) */
      flops += K * N; /* d16 = d13.z */
      flops += K * N; /* d17 = d13.r */
      flops += 3.0 * K * N; /* d18 = d16.sig'(r) */
      flops += K * N; /* d19 = d17 + d4 */
      flops += (2.0 * K * K * N + K * K); /* d21 = Wr^T * d18 (including transpose) */
      flops += K * N; /* d22 = d21 + d15 */
      flops += K * N; /* d23 = d19 + d22 */
      flops += (2.0 * K * C * N + K * C); /* d12 = Ug^T * d10 (including transpose) */
      flops += (2.0 * K * C * N + K * C); /* d14 = Uz^T * d11 (including transpose) */
      flops += (2.0 * K * C * N + K * C); /* d20 = Ur^T * d18 (including transpose) */
      flops += 2.0 * K * N; /* djdx = d12 + d14 + d20 */
      flops += (2.0 * K * N * K + K * N + K * K); /* djdwr = djdwr + d18 * h^T */
      flops += (2.0 * K * N * K + K * N + K * K); /* djdwz = djdwz + d11 * h^T */
      flops += (2.0 * K * N * K + 2.0 * K * N + K * K); /* djdwg = djdwg + d10 * (h.r)^T */
      flops += (2.0 * K * N * C + C * N + K * C); /* djdur = djdur + d18 * x^T */
      flops += (2.0 * K * N * C + C * N + K * C); /* djduz = djduz + d11 * x^T */
      flops += (2.0 * K * N * C + C * N + K * C); /* djdug = djdug + d10 * x^T */
      flops += K * N; /* djdbr = djdbr + d18 */
      flops += K * N; /* djdbz = djdbz + d11 */
      flops += K * N; /* djdbg = djdbg + d10 */
      flops *= t; /* for t time steps */
      flops *= iters;

      printf("GFLOP  = %.5g\n", flops*1e-9/(double)iters);
      printf("bp+wu time = %.5g\n", ((double)(l_total/iters)));
      printf("GFLOPS  = %.5g\n", (flops*1e-9)/l_total);

      printf("PERFDUMP,BP+WU,%s,%i,%i,%i,%i,%i,%i,%i,%i,%.5g,%.5g\n", LIBXSMM_VERSION, nThreads, N, C, K, t, bn, bc, bk, ((double)(l_total/iters)), (flops*1e-9)/l_total);
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
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_release_tensor( libxsmm_handle, LIBXSMM_DNN_RNN_REGULAR_HIDDEN_STATE_PREV ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_release_tensor( libxsmm_handle, LIBXSMM_DNN_RNN_REGULAR_WEIGHT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_release_tensor( libxsmm_handle, LIBXSMM_DNN_RNN_REGULAR_RECUR_WEIGHT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_release_tensor( libxsmm_handle, LIBXSMM_DNN_RNN_REGULAR_BIAS ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_release_tensor( libxsmm_handle, LIBXSMM_DNN_RNN_REGULAR_HIDDEN_STATE ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_release_tensor( libxsmm_handle, LIBXSMM_DNN_RNN_INTERNAL_I ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_release_tensor( libxsmm_handle, LIBXSMM_DNN_RNN_INTERNAL_CI ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_release_tensor( libxsmm_handle, LIBXSMM_DNN_RNN_INTERNAL_F ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_release_tensor( libxsmm_handle, LIBXSMM_DNN_RNN_INTERNAL_O ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_release_tensor( libxsmm_handle, LIBXSMM_DNN_RNN_GRADIENT_INPUT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_release_tensor( libxsmm_handle, LIBXSMM_DNN_RNN_GRADIENT_HIDDEN_STATE_PREV ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_release_tensor( libxsmm_handle, LIBXSMM_DNN_RNN_GRADIENT_WEIGHT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_release_tensor( libxsmm_handle, LIBXSMM_DNN_RNN_GRADIENT_RECUR_WEIGHT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_release_tensor( libxsmm_handle, LIBXSMM_DNN_RNN_GRADIENT_BIAS ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_rnncell_release_tensor( libxsmm_handle, LIBXSMM_DNN_RNN_GRADIENT_HIDDEN_STATE ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_input ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_hidden_state_prev ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_weight ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_recur_weight ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_bias ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_hidden_state ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_i ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_c ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_f ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_o ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_dinput ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_dhidden_state_prev ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_dweight ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_drecur_weight ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_dbias ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_dhidden_state ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_rnncell( libxsmm_handle ) );
  }

  /* deallocate data */
  libxsmm_free(xgoldt);
  libxsmm_free(hpgold);
  libxsmm_free(wigold);
  libxsmm_free(wcgold);
  libxsmm_free(wfgold);
  libxsmm_free(rigold);
  libxsmm_free(rcgold);
  libxsmm_free(rfgold);
  libxsmm_free(bigold);
  libxsmm_free(bcgold);
  libxsmm_free(bfgold);
  libxsmm_free(hgoldt);
  libxsmm_free(igoldt);
  libxsmm_free(cgoldt);
  libxsmm_free(fgoldt);
  libxsmm_free(ogoldt);
  libxsmm_free(dxgoldt);
  libxsmm_free(dhpgold);
  libxsmm_free(dwgold);
  libxsmm_free(drgold);
  libxsmm_free(dbgold);
  libxsmm_free(dhgoldt);
  libxsmm_free(xt);
  libxsmm_free(hp);
  libxsmm_free(w);
  libxsmm_free(r);
  libxsmm_free(b);
  libxsmm_free(ht);
  libxsmm_free(it);
  libxsmm_free(ct);
  libxsmm_free(ft);
  libxsmm_free(ot);
  libxsmm_free(dxt);
  libxsmm_free(dhp);
  libxsmm_free(dw);
  libxsmm_free(dr);
  libxsmm_free(db);
  libxsmm_free(dht);
  libxsmm_free(dwtest);
  libxsmm_free(drtest);

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

