/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Evangelos Georganas (Intel Corp.)
******************************************************************************/
#include <libxsmm.h>
#include <libxsmm_intrinsics_x86.h>
#include "../common/dnn_common.h"

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

LIBXSMM_INLINE void matrix_copy_CK_to_KCCK_bfp16(libxsmm_bfloat16 *src, libxsmm_bfloat16 *dst, int C, int K, int bc, int bk)
{
  int k1, k2, c1, c2;
  int kBlocks = K/bk;
  int cBlocks = C/bc;
  LIBXSMM_VLA_DECL(2, libxsmm_bfloat16, real_src, src, K);
  LIBXSMM_VLA_DECL(5, libxsmm_bfloat16, real_dst, dst, cBlocks, bc/2, bk, 2);

#if defined(_OPENMP)
# pragma omp parallel for private(k1)
#endif
  for (k1 = 0; k1 < kBlocks; k1++) {
    for (c1 = 0; c1 < cBlocks; c1++) {
      for (c2 = 0; c2 < bc; c2++) {
        for (k2 = 0; k2 < bk; k2++) {
          LIBXSMM_VLA_ACCESS(5, real_dst, k1, c1, c2/2, k2, c2%2, cBlocks, bc/2, bk, 2) =
            LIBXSMM_VLA_ACCESS(2, real_src, c1*bc+c2, k1*bk+k2, K);
        }
      }
    }
  }
}
LIBXSMM_INLINE void matrix_copy_bfp16_f32(int size, libxsmm_bfloat16 *src, float *dst)
{
  int i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < size; i++) {
    libxsmm_bfloat16_hp t;
    t.i[1] = src[i];
    t.i[0] = 0;
    dst[i] = t.f;
  }
}

LIBXSMM_INLINE void matrix_copy_KCCK_to_CKKC_bfp16(libxsmm_bfloat16 *src, libxsmm_bfloat16 *dst, int C, int K, int bc, int bk)
{
  int k1, k2, c1, c2;
  int kBlocks = K/bk;
  int cBlocks = C/bc;
  LIBXSMM_VLA_DECL(5, libxsmm_bfloat16, real_dst, dst, kBlocks, bk/2, bc, 2);
  LIBXSMM_VLA_DECL(5, libxsmm_bfloat16, real_src, src, cBlocks, bc/2, bk, 2);

#if defined(_OPENMP)
# pragma omp parallel for private(k1)
#endif
  for (k1 = 0; k1 < kBlocks; k1++) {
    for (c1 = 0; c1 < cBlocks; c1++) {
      for (c2 = 0; c2 < bc; c2++) {
        for (k2 = 0; k2 < bk; k2++) {
          LIBXSMM_VLA_ACCESS(5, real_dst, c1, k1, k2/2, c2, k2%2, kBlocks, bk/2, bc, 2) =
          LIBXSMM_VLA_ACCESS(5, real_src, k1, c1, c2/2, k2, c2%2, cBlocks, bc/2, bk, 2);
        }
      }
    }
  }
}

LIBXSMM_INLINE void convert_ck_f32_to_c4k_bfp16(int C, int K, float *src, libxsmm_bfloat16 *dst)
{
  int x, y;
#if defined(_OPENMP)
  LIBXSMM_OMP_VAR(x);
# pragma omp parallel for private(x, y)
#endif
  for (y = 0; y < C; y++) {
    for (x = 0; x < K; x++) {
      libxsmm_bfloat16_hp t;
      t.f = src[y*K + x];
      dst[y*4*K + x] = t.i[1];
    }
  }
}

LIBXSMM_INLINE void zero_buf_f32(float* buf, size_t size) {
  int i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < (int)size; ++i) {
    buf[i] = 0.0f;
  }
}

LIBXSMM_INLINE void zero_buf_bfp16(libxsmm_bfloat16* buf, size_t size) {
  int i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < (int)size; ++i) {
    buf[i] = 0;
  }
}

#define CHKERR_LIBXSMM_DNN(A) if ( A != LIBXSMM_DNN_SUCCESS ) fprintf(stderr, "%s\n", libxsmm_dnn_get_error(A) );

#if 0
#define TWO_GEMMS
#endif
#if 0
#define PROFILE
#endif
#if defined(PROFILE)
unsigned long long Gbl_blas_start, Gbl_blas_end, Gbl_eltwise_start, Gbl_eltwise_end, Gbl_conv_start, Gbl_conv_end;
double Gbl_blas_total, Gbl_eltwise_total, Gbl_conv_total;
#endif

LIBXSMM_INLINE void matrix_copy_reformat_f32_bfp16(int T, int N, int C, int bn, int bc, float *src, libxsmm_bfloat16 *dst)
{
  int t, n, c, i;
  int nBlocks = N/bn, cBlocks = C/bc;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (t = 0; t < T; t++) {
    for (n = 0; n < N; n++) {
      for (c = 0; c < C; c++) {
        i = t*N*C+n*C+c;
        libxsmm_bfloat16_hp tmp;
        tmp.f = src[i];
        dst[t*N*C+(n/bn)*C*bn+(c/bc)*bc*bn+(n%bn)*bc+c%bc] = tmp.i[1];
      }
    }
  }
}

LIBXSMM_INLINE void matrix_copy_reformat_bfp16_f32(int T, int N, int C, int bn, int bc, libxsmm_bfloat16 *src, float *dst)
{
  int t, n, c, i;
  int nBlocks = N/bn, cBlocks = C/bc;
  for (t = 0; t < T; t++) {
    for (n = 0; n < N; n++) {
      for (c = 0; c < C; c++) {
        i = t*N*C+n*C+c;
        libxsmm_bfloat16_hp tmp;
        tmp.i[1] = src[t*N*C+(n/bn)*C*bn+(c/bc)*bc*bn+(n%bn)*bc+c%bc];
        tmp.i[0] = 0;
        dst[i] = tmp.f;

      }
    }
  }
}

int main(int argc, char* argv[])
{
  float *wigold, *wfgold, *wogold, *wcgold, *rigold, *rfgold, *rogold, *rcgold, *bigold, *bfgold, *bogold, *bcgold;
  float *xgoldt, *cspgold,*hpgold, *csgoldt, *cogoldt, *hgoldt;
  float *dwgold, *drgold, *dbgold;
  float *dxgoldt, *dcspgold, *dhpgold, *dcsgold, *dhgoldt;
  float *icfogoldt, *wgold, *rgold;
  float *scratch_fwd, *scratch_bu;
  libxsmm_bfloat16 *xt, *csp, *hp, *w, *wt, *r, *rt, *b, *cst, *ht, *w_tmp, *r_tmp;
  libxsmm_bfloat16 *it, *ft, *ot, *cit, *cot;
  libxsmm_bfloat16 *dxt, *dcsp, *dhp, *dw, *dr, *db, *dcs, *dht;
  float forget_bias = 1.0f;
  float *h_test, *dxt_test, *dw_test, *dr_test, *db_test;

  void *scratch, *internalstate;
  size_t scratch_size = 0, internalstate_size = 0;

  int iters = 10;   /* repetitions of benchmark */
  int pass = 0;     /* pass: 0--FWD, 1--BWD, 2--UPD, 3--BWD+UPD */
  int N = 168;      /* size of mini-batch */
  int C = 512;      /* number of inputs */
  int K = 512;      /* number of outputs */
  int t = 50;       /* number of time steps (>= 1) */
  int bn = 32;
  int bk = 32;
  int bc = 32;
  int use_fwd_fused_impl = 0;
  int bwdupd_block = 1;
  int fwd_block = 1;

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
  int j;

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
  if (argc > j) bn    = atoi(argv[j++]);
  if (argc > j) bc    = atoi(argv[j++]);
  if (argc > j) bk    = atoi(argv[j++]);
  if (argc > j) use_fwd_fused_impl    = atoi(argv[j++]);
  if (argc > j) fwd_block    = atoi(argv[j++]);

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
  xgoldt    = (float*)libxsmm_aligned_malloc(N*C*t*sizeof(float), 2097152);
  cspgold   = (float*)libxsmm_aligned_malloc(K*N*sizeof(float), 2097152);
  hpgold    = (float*)libxsmm_aligned_malloc(K*N*sizeof(float), 2097152);
  wigold    = (float*)libxsmm_aligned_malloc(C*K*sizeof(float), 2097152);
  wfgold    = (float*)libxsmm_aligned_malloc(C*K*sizeof(float), 2097152);
  wogold    = (float*)libxsmm_aligned_malloc(C*K*sizeof(float), 2097152);
  wcgold    = (float*)libxsmm_aligned_malloc(C*K*sizeof(float), 2097152);
  rigold    = (float*)libxsmm_aligned_malloc(K*K*sizeof(float), 2097152);
  rfgold    = (float*)libxsmm_aligned_malloc(K*K*sizeof(float), 2097152);
  rogold    = (float*)libxsmm_aligned_malloc(K*K*sizeof(float), 2097152);
  rcgold    = (float*)libxsmm_aligned_malloc(K*K*sizeof(float), 2097152);
  bigold    = (float*)libxsmm_aligned_malloc(K*sizeof(float), 2097152);
  bfgold    = (float*)libxsmm_aligned_malloc(K*sizeof(float), 2097152);
  bogold    = (float*)libxsmm_aligned_malloc(K*sizeof(float), 2097152);
  bcgold    = (float*)libxsmm_aligned_malloc(K*sizeof(float), 2097152);
  csgoldt   = (float*)libxsmm_aligned_malloc(K*N*t*sizeof(float), 2097152);
  cogoldt   = (float*)libxsmm_aligned_malloc(K*N*t*sizeof(float), 2097152);
  hgoldt    = (float*)libxsmm_aligned_malloc(K*N*t*sizeof(float), 2097152);
  dxgoldt   = (float*)libxsmm_aligned_malloc(N*C*t*sizeof(float), 2097152);
  dcspgold  = (float*)libxsmm_aligned_malloc(K*N*sizeof(float), 2097152);
  dhpgold   = (float*)libxsmm_aligned_malloc(K*N*sizeof(float), 2097152);
#if defined(TWO_GEMMS)
  wgold     = (float*)libxsmm_aligned_malloc(C*K*4*sizeof(float), 2097152);
  rgold     = (float*)libxsmm_aligned_malloc(K*K*4*sizeof(float), 2097152);
  dwgold    = (float*)libxsmm_aligned_malloc(C*K*4*sizeof(float), 2097152);
  drgold    = (float*)libxsmm_aligned_malloc(K*K*4*sizeof(float), 2097152);
#else
  wgold     = (float*)libxsmm_aligned_malloc((C+K)*K*4*sizeof(float), 2097152);
  rgold     = NULL;
  dwgold    = (float*)libxsmm_aligned_malloc((C+K)*K*4*sizeof(float), 2097152);
  drgold    = NULL;
#endif
  scratch_fwd = (float*)libxsmm_aligned_malloc((C+K)*N*sizeof(float), 2097152);
  scratch_bu  = (float*)libxsmm_aligned_malloc(((C+K)*N*2 + K*N*t*5)*sizeof(float), 2097152);
  dbgold    = (float*)libxsmm_aligned_malloc(K*4*sizeof(float), 2097152);
  dcsgold   = (float*)libxsmm_aligned_malloc(K*N*sizeof(float), 2097152);
  dhgoldt   = (float*)libxsmm_aligned_malloc(K*N*t*sizeof(float), 2097152);
  icfogoldt = (float*)libxsmm_aligned_malloc(K*N*t*4*sizeof(float), 2097152);
  xt        = (libxsmm_bfloat16*)libxsmm_aligned_malloc(N*C*t*sizeof(libxsmm_bfloat16), 2097152);
  csp       = (libxsmm_bfloat16*)libxsmm_aligned_malloc(K*N*sizeof(libxsmm_bfloat16), 2097152);
  hp        = (libxsmm_bfloat16*)libxsmm_aligned_malloc(K*N*sizeof(libxsmm_bfloat16), 2097152);
  w         = (libxsmm_bfloat16*)libxsmm_aligned_malloc(C*K*4*sizeof(libxsmm_bfloat16), 2097152);
  r         = (libxsmm_bfloat16*)libxsmm_aligned_malloc(K*K*4*sizeof(libxsmm_bfloat16), 2097152);
  wt        = (libxsmm_bfloat16*)libxsmm_aligned_malloc(C*K*4*sizeof(libxsmm_bfloat16), 2097152);
  rt        = (libxsmm_bfloat16*)libxsmm_aligned_malloc(K*K*4*sizeof(libxsmm_bfloat16), 2097152);
  w_tmp     = (libxsmm_bfloat16*)libxsmm_aligned_malloc(C*K*4*sizeof(libxsmm_bfloat16), 2097152);
  r_tmp     = (libxsmm_bfloat16*)libxsmm_aligned_malloc(K*K*4*sizeof(libxsmm_bfloat16), 2097152);
  b         = (libxsmm_bfloat16*)libxsmm_aligned_malloc(K*4*sizeof(libxsmm_bfloat16), 2097152);
  cst       = (libxsmm_bfloat16*)libxsmm_aligned_malloc(K*N*t*sizeof(libxsmm_bfloat16), 2097152);
  ht        = (libxsmm_bfloat16*)libxsmm_aligned_malloc(K*N*t*sizeof(libxsmm_bfloat16), 2097152);
  it        = (libxsmm_bfloat16*)libxsmm_aligned_malloc(K*N*t*sizeof(libxsmm_bfloat16), 2097152);
  ft        = (libxsmm_bfloat16*)libxsmm_aligned_malloc(K*N*t*sizeof(libxsmm_bfloat16), 2097152);
  ot        = (libxsmm_bfloat16*)libxsmm_aligned_malloc(K*N*t*sizeof(libxsmm_bfloat16), 2097152);
  cit       = (libxsmm_bfloat16*)libxsmm_aligned_malloc(K*N*t*sizeof(libxsmm_bfloat16), 2097152);
  cot       = (libxsmm_bfloat16*)libxsmm_aligned_malloc(K*N*t*sizeof(libxsmm_bfloat16), 2097152);
  dxt       = (libxsmm_bfloat16*)libxsmm_aligned_malloc(N*C*t*sizeof(libxsmm_bfloat16), 2097152);
  dcsp      = (libxsmm_bfloat16*)libxsmm_aligned_malloc(K*N*sizeof(libxsmm_bfloat16), 2097152);
  dhp       = (libxsmm_bfloat16*)libxsmm_aligned_malloc(K*N*sizeof(libxsmm_bfloat16), 2097152);
  dw        = (libxsmm_bfloat16*)libxsmm_aligned_malloc(C*K*4*sizeof(libxsmm_bfloat16), 2097152);
  dr        = (libxsmm_bfloat16*)libxsmm_aligned_malloc(K*K*4*sizeof(libxsmm_bfloat16), 2097152);
  db        = (libxsmm_bfloat16*)libxsmm_aligned_malloc(K*4*sizeof(libxsmm_bfloat16), 2097152);
  dcs       = (libxsmm_bfloat16*)libxsmm_aligned_malloc(K*N*sizeof(libxsmm_bfloat16), 2097152);
  dht       = (libxsmm_bfloat16*)libxsmm_aligned_malloc(K*N*t*sizeof(libxsmm_bfloat16), 2097152);
  h_test    = (float*)libxsmm_aligned_malloc(K*N*t*sizeof(float), 2097152);
  dxt_test  = (float*)libxsmm_aligned_malloc(C*N*t*sizeof(float), 2097152);
  dw_test   = (float*)libxsmm_aligned_malloc(K*C*4*sizeof(float), 2097152);
  dr_test   = (float*)libxsmm_aligned_malloc(K*K*4*sizeof(float), 2097152);
  db_test   = (float*)libxsmm_aligned_malloc(K*4*sizeof(float), 2097152);

  LIBXSMM_VLA_DECL(2, float, xgold, xgoldt, N * C);
  LIBXSMM_VLA_DECL(2, float, hgold, hgoldt, K * N);
  LIBXSMM_VLA_DECL(2, float, dhgold, dhgoldt, K * N);
  /*LIBXSMM_VLA_DECL(2, float, h, h_test, K * N);*/

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

  zero_buf_f32(csgoldt, K*N*t);
  zero_buf_f32(cogoldt, K*N*t);
  zero_buf_f32(hgoldt,  K*N*t);
  /* BWD/UPD */
  for (j = 0; j < t; ++j) {
    init_buf(&LIBXSMM_VLA_ACCESS(2, dhgold, j, 0, K * N), K*N, 0, 0);
  }
  init_buf(dcsgold, K*N, 0, 0);

  zero_buf_f32(dxgoldt,  N*C*t);
  zero_buf_f32(dcspgold, K*N);
  zero_buf_f32(dhpgold,  K*N);
#if defined(TWO_GEMMS)
  zero_buf_f32(dwgold, C*K*4);
  zero_buf_f32(drgold, K*K*4);
#else
  zero_buf_f32(dwgold, (C+K)*K*4);
#endif
  zero_buf_f32(dbgold, K*4);

  /* first touch LIBXSMM */
  zero_buf_bfp16(xt,  N*C*t);
  zero_buf_bfp16(csp, K*N);
  zero_buf_bfp16(hp,  K*N);
  zero_buf_bfp16(w,   C*K*4);
  zero_buf_bfp16(r,   K*K*4);
  zero_buf_bfp16(b,   K*4);
  zero_buf_bfp16(cst, K*N*t);
  zero_buf_bfp16(ht,  K*N*t);
  zero_buf_bfp16(it,  K*N*t);
  zero_buf_bfp16(ft,  K*N*t);
  zero_buf_bfp16(ot,  K*N*t);
  zero_buf_bfp16(cit, K*N*t);
  zero_buf_bfp16(cot, K*N*t);
  zero_buf_bfp16(dxt,  N*C*t);
  zero_buf_bfp16(dcsp, K*N);
  zero_buf_bfp16(dhp,  K*N);
  zero_buf_bfp16(dw,   C*K*4);
  zero_buf_bfp16(dr,   K*K*4);
  zero_buf_bfp16(db,   K*4);
  zero_buf_bfp16(dcs,  K*N);
  zero_buf_bfp16(dht,  K*N*t);

  /* Make things bf16 */
  rne_mask_fp32_bfp16( wigold, wigold, C*K );
  rne_mask_fp32_bfp16( wcgold, wcgold, C*K );
  rne_mask_fp32_bfp16( wfgold, wfgold, C*K );
  rne_mask_fp32_bfp16( wogold, wogold, C*K );
  rne_mask_fp32_bfp16( rigold, rigold, K*K );
  rne_mask_fp32_bfp16( rcgold, rcgold, K*K );
  rne_mask_fp32_bfp16( rfgold, rfgold, K*K );
  rne_mask_fp32_bfp16( rogold, rogold, K*K );
  rne_mask_fp32_bfp16( bigold, bigold, K );
  rne_mask_fp32_bfp16( bcgold, bcgold, K );
  rne_mask_fp32_bfp16( bfgold, bfgold, K );
  rne_mask_fp32_bfp16( bogold, bogold, K );
  rne_mask_fp32_bfp16( xgoldt, xgoldt, N*C*t );
  rne_mask_fp32_bfp16( cspgold, cspgold, N*K );
  rne_mask_fp32_bfp16( hpgold, hpgold, N*K );
  rne_mask_fp32_bfp16( csgoldt, csgoldt, N*K*t );
  rne_mask_fp32_bfp16( cogoldt, cogoldt, N*K*t );
  rne_mask_fp32_bfp16( hgoldt, hgoldt, N*K*t );
  rne_mask_fp32_bfp16( icfogoldt, icfogoldt, N*K*t*4 );
#if defined(TWO_GEMMS)
  rne_mask_fp32_bfp16( wgold, wgold, C*K*4 );
  rne_mask_fp32_bfp16( rgold, rgold, K*K*4 );
#else
  rne_mask_fp32_bfp16( wgold, wgold, (C+K)*K*4 );
#endif
  rne_mask_fp32_bfp16( dcsgold, dcsgold, K*N );
  rne_mask_fp32_bfp16( dhgoldt, dhgoldt, K*N*t );
#if defined(TWO_GEMMS)
  rne_mask_fp32_bfp16( dwgold, dwgold, 4*C*K );
  rne_mask_fp32_bfp16( drgold, drgold, 4*K*K );
#else
  rne_mask_fp32_bfp16( dwgold, dwgold, 4*(C+K)*K );
#endif
  rne_mask_fp32_bfp16( dbgold, dbgold, 4*K );
  rne_mask_fp32_bfp16( dxgoldt, dxgoldt, N*C*t );
  rne_mask_fp32_bfp16( dcspgold, dcspgold, N*K );
  rne_mask_fp32_bfp16( dhpgold, dhpgold, K*N );

  if (LIBXSMM_NEQ(0, check)) {
    printf("##########################################\n");
    printf("#         Computing Reference ...        #\n");
    printf("##########################################\n");

    lstm_ref_fwd( N, C, K, t, forget_bias,
                  wigold, wcgold, wfgold, wogold,
                  rigold, rcgold, rfgold, rogold,
                  bigold, bcgold, bfgold, bogold,
                  xgoldt, cspgold, hpgold,
                  csgoldt, cogoldt, hgoldt,
                  icfogoldt, wgold, rgold, scratch_fwd );

    /* Make things BF16 for bwd/upd pass refernce computation */
    rne_mask_fp32_bfp16( icfogoldt, icfogoldt, 4*K*N*t );
    rne_mask_fp32_bfp16( hgoldt, hgoldt, K*N*t );
    rne_mask_fp32_bfp16( cogoldt, cogoldt, K*N*t );
    rne_mask_fp32_bfp16( csgoldt, csgoldt, K*N*t );
    rne_mask_fp32_bfp16( cspgold, cspgold, K*N );
    rne_mask_fp32_bfp16( hpgold, hpgold, K*N );
    rne_mask_fp32_bfp16( dhgoldt, dhgoldt, t*K*N );
    rne_mask_fp32_bfp16( dcsgold, dcsgold, K*N );

    lstm_ref_bwd_upd( N, C, K, t,
                      xgoldt, cspgold, hpgold,
                      csgoldt, cogoldt, hgoldt,
                      icfogoldt, wgold, rgold,
                      dcsgold, dhgoldt,
                      dwgold, drgold, dbgold,
                      dxgoldt, dcspgold, dhpgold, scratch_bu );

    rne_mask_fp32_bfp16( dxgoldt, dxgoldt, C*N*t );
#if defined(TWO_GEMMS)
    rne_mask_fp32_bfp16( dwgold, dwgold, 4*C*K );
    rne_mask_fp32_bfp16( drgold, drgold, 4*K*K );
#else
    rne_mask_fp32_bfp16( dwgold, dwgold, 4*(C+K)*K );
#endif
    rne_mask_fp32_bfp16( dbgold, dbgold, 4*K );
    rne_mask_fp32_bfp16( dcspgold, dcspgold, N*K );
    rne_mask_fp32_bfp16( dhpgold, dhpgold, K*N );

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
    lstmcell_desc.datatype_in = LIBXSMM_DNN_DATATYPE_BF16;
    lstmcell_desc.datatype_out = LIBXSMM_DNN_DATATYPE_BF16;
    lstmcell_desc.buffer_format = LIBXSMM_DNN_TENSOR_FORMAT_NCPACKED;
    lstmcell_desc.filter_format = LIBXSMM_DNN_TENSOR_FORMAT_CKPACKED;
    lstmcell_desc.fwd_block = fwd_block;
    lstmcell_desc.bwdupd_block = bwdupd_block;
    lstmcell_desc.use_fwd_fused_impl = use_fwd_fused_impl;

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

    /* copy in data to LIBXSMM bf16 format */
    matrix_copy_reformat_f32_bfp16(t, N, C, bn, bc,  xgoldt, xt);
    matrix_copy_reformat_f32_bfp16(1, N, K, bn, bk, cspgold, csp);
    matrix_copy_reformat_f32_bfp16(1, N, K, bn, bk, hpgold, hp);
    convert_ck_f32_to_c4k_bfp16(C, K, wigold, w_tmp);
    convert_ck_f32_to_c4k_bfp16(C, K, wcgold, &(w_tmp[K]));
    convert_ck_f32_to_c4k_bfp16(C, K, wfgold, &(w_tmp[2*K]));
    convert_ck_f32_to_c4k_bfp16(C, K, wogold, &(w_tmp[3*K]));
    convert_ck_f32_to_c4k_bfp16(K, K, rigold, r_tmp);
    convert_ck_f32_to_c4k_bfp16(K, K, rcgold, &(r_tmp[K]));
    convert_ck_f32_to_c4k_bfp16(K, K, rfgold, &(r_tmp[2*K]));
    convert_ck_f32_to_c4k_bfp16(K, K, rogold, &(r_tmp[3*K]));
    matrix_copy_CK_to_KCCK_bfp16(w_tmp, w,  C, 4*K, bc, bk);
    matrix_copy_CK_to_KCCK_bfp16(r_tmp, r,  K, 4*K, bk, bk);
    matrix_copy_KCCK_to_CKKC_bfp16(&w[0], &wt[0], C, K, bc, bk);
    matrix_copy_KCCK_to_CKKC_bfp16(&w[C*K], &wt[C*K], C, K, bc, bk);
    matrix_copy_KCCK_to_CKKC_bfp16(&w[2*C*K], &wt[2*C*K], C, K, bc, bk);
    matrix_copy_KCCK_to_CKKC_bfp16(&w[3*C*K], &wt[3*C*K], C, K, bc, bk);
    matrix_copy_KCCK_to_CKKC_bfp16(&r[0], &rt[0], K, K, bk, bk);
    matrix_copy_KCCK_to_CKKC_bfp16(&r[K*K], &rt[K*K], K, K, bk, bk);
    matrix_copy_KCCK_to_CKKC_bfp16(&r[2*K*K], &rt[2*K*K], K, K, bk, bk);
    matrix_copy_KCCK_to_CKKC_bfp16(&r[3*K*K], &rt[3*K*K], K, K, bk, bk);
#if 0
    matrix_copy_f32_bfp16(K, bigold, &(b[0]));
    matrix_copy_f32_bfp16(K, bcgold, &(b[K]));
    matrix_copy_f32_bfp16(K, bfgold, &(b[2*K]));
    matrix_copy_f32_bfp16(K, bogold, &(b[3*K]));
    matrix_copy_f32_bfp16(K*N*t, dhgoldt, dht);
    matrix_copy_f32_bfp16(K*N, dcsgold, dcs);
#endif
    matrix_copy_reformat_f32_bfp16(1, 1, K, 1, K, bigold, &(b[0]));
    matrix_copy_reformat_f32_bfp16(1, 1, K, 1, K, bcgold, &(b[K]));
    matrix_copy_reformat_f32_bfp16(1, 1, K, 1, K, bfgold, &(b[2*K]));
    matrix_copy_reformat_f32_bfp16(1, 1, K, 1, K, bogold, &(b[3*K]));
    matrix_copy_reformat_f32_bfp16(t, N, K, bn, bk, dhgoldt, dht);
    matrix_copy_reformat_f32_bfp16(1, N, K, bn, bk, dcsgold, dcs);


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

      /* Upconvert libxsmm bf16 buffer to fp32 for correctness check */
      matrix_copy_reformat_bfp16_f32(t, N, K, bn, bk, ht, h_test);

      /* compare */
      libxsmm_matdiff(&norms_fwd, LIBXSMM_DATATYPE_F32, t*K*N, 1, &LIBXSMM_VLA_ACCESS(2, hgold, 0, 0, K * N), h_test, 0, 0);
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

      /* Upconvert libxsmm bf16 buffer to fp32 for correctness check */
      matrix_copy_bfp16_f32(N*C*t, dxt, dxt_test);

      /* compare */
      libxsmm_matdiff(&norms_bwd, LIBXSMM_DATATYPE_F32, N*C*t, 1, dxgoldt, dxt_test, 0, 0);
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

      /* Copy out dw matrices from KCCK bf16 format to CK bf16 format */
      matrix_copy_KCCK_to_CK_bf16(dw, w_tmp, C, 4*K, bc, bk);

      /* Upconvert libxsmm bf16 buffer to fp32 for correctness check */
      matrix_copy_bfp16_f32(4*C*K, w_tmp, dw_test);

      /* compare */
      libxsmm_matdiff(&norms_upd_w, LIBXSMM_DATATYPE_F32, C*K*4, 1, dwgold, dw_test, 0, 0);
      printf("Delta weight\n");
      printf("L1 reference  : %.25g\n", norms_upd_w.l1_ref);
      printf("L1 test       : %.25g\n", norms_upd_w.l1_tst);
      printf("L2 abs.error  : %.24f\n", norms_upd_w.l2_abs);
      printf("L2 rel.error  : %.24f\n", norms_upd_w.l2_rel);
      printf("Linf abs.error: %.24f\n", norms_upd_w.linf_abs);
      printf("Linf rel.error: %.24f\n", norms_upd_w.linf_rel);
      printf("Check-norm    : %.24f\n", norms_upd_w.normf_rel);
      libxsmm_matdiff_reduce(&diff, &norms_upd_w);


      /* Copy out dr matrices from KCCK bf16 format to CK bf16 format */
      matrix_copy_KCCK_to_CK_bf16(dr, r_tmp, K, 4*K, bk, bk);

      /* Upconvert libxsmm bf16 buffer to fp32 for correctness check */
      matrix_copy_bfp16_f32(4*K*K, r_tmp, dr_test);

#if defined(TWO_GEMMS)
      libxsmm_matdiff(&norms_upd_r, LIBXSMM_DATATYPE_F32, K*K*4, 1, drgold, dr_test, 0, 0);
#else
      libxsmm_matdiff(&norms_upd_r, LIBXSMM_DATATYPE_F32, K*K*4, 1, &(dwgold[C*K*4]), dr_test, 0, 0);
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

      /* Upconvert libxsmm bf16 buffer to fp32 for correctness check */
      matrix_copy_bfp16_f32(4*K, db, db_test);

      libxsmm_matdiff(&norms_upd_b, LIBXSMM_DATATYPE_F32, K*4, 1, dbgold, db_test, 0, 0);
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

      /* Upconvert libxsmm bf16 buffer to fp32 for correctness check */
      matrix_copy_reformat_bfp16_f32(t, N, C, bn, bc, dxt, dxt_test);

      /* compare */
      libxsmm_matdiff(&norms_bwd, LIBXSMM_DATATYPE_F32, N*C*t, 1, dxgoldt, dxt_test, 0, 0);
      printf("Delta input\n");
      printf("L1 reference  : %.25g\n", norms_bwd.l1_ref);
      printf("L1 test       : %.25g\n", norms_bwd.l1_tst);
      printf("L2 abs.error  : %.24f\n", norms_bwd.l2_abs);
      printf("L2 rel.error  : %.24f\n", norms_bwd.l2_rel);
      printf("Linf abs.error: %.24f\n", norms_bwd.linf_abs);
      printf("Linf rel.error: %.24f\n", norms_bwd.linf_rel);
      printf("Check-norm    : %.24f\n", norms_bwd.normf_rel);
      libxsmm_matdiff_reduce(&diff, &norms_bwd);

      /* Copy out dw matrices from KCCK bf16 format to CK bf16 format */
      matrix_copy_KCCK_to_CK_bf16(dw, w_tmp, C, 4*K, bc, bk);
      /* Upconvert libxsmm bf16 buffer to fp32 for correctness check */
      matrix_copy_bfp16_f32(4*C*K, w_tmp, dw_test);

      libxsmm_matdiff(&norms_upd_w, LIBXSMM_DATATYPE_F32, C*K*4, 1, dwgold, dw_test, 0, 0);
      printf("Delta weight\n");
      printf("L1 reference  : %.25g\n", norms_upd_w.l1_ref);
      printf("L1 test       : %.25g\n", norms_upd_w.l1_tst);
      printf("L2 abs.error  : %.24f\n", norms_upd_w.l2_abs);
      printf("L2 rel.error  : %.24f\n", norms_upd_w.l2_rel);
      printf("Linf abs.error: %.24f\n", norms_upd_w.linf_abs);
      printf("Linf rel.error: %.24f\n", norms_upd_w.linf_rel);
      printf("Check-norm    : %.24f\n", norms_upd_w.normf_rel);
      libxsmm_matdiff_reduce(&diff, &norms_upd_w);


      /* Copy out dr matrices from KCCK bf16 format to CK bf16 format */
      matrix_copy_KCCK_to_CK_bf16(dr, r_tmp, K, 4*K, bk, bk);

      /* Upconvert libxsmm bf16 buffer to fp32 for correctness check */
      matrix_copy_bfp16_f32(4*K*K, r_tmp, dr_test);

#if defined(TWO_GEMMS)
      libxsmm_matdiff(&norms_upd_r, LIBXSMM_DATATYPE_F32, K*K*4, 1, drgold, dr_test, 0, 0);
#else
      libxsmm_matdiff(&norms_upd_r, LIBXSMM_DATATYPE_F32, K*K*4, 1, &(dwgold[C*K*4]), dr_test, 0, 0);
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

      /* Upconvert libxsmm bf16 buffer to fp32 for correctness check */
      matrix_copy_bfp16_f32(4*K, db, db_test);

      libxsmm_matdiff(&norms_upd_b, LIBXSMM_DATATYPE_F32, K*4, 1, dbgold, db_test, 0, 0);
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

      printf("##########################################\n");
      printf("#   Performance - FWD (BLAS)             #\n");
      printf("##########################################\n");
      /* run LIBXSMM LSTM for performance */
#if defined(PROFILE)
      Gbl_blas_total = 0.0;
      Gbl_eltwise_total = 0.0;
      Gbl_conv_total = 0.0;
#endif
      l_start = libxsmm_timer_tick();
      for (j = 0; j < iters; ++j) {
        lstm_ref_fwd( N, C, K, t, forget_bias,
            wigold, wcgold, wfgold, wogold,
            rigold, rcgold, rfgold, rogold,
            bigold, bcgold, bfgold, bogold,
            xgoldt, cspgold, hpgold,
            csgoldt, cogoldt, hgoldt,
            icfogoldt, wgold, rgold, scratch_fwd );
      }
      l_end = libxsmm_timer_tick();
      l_total = libxsmm_timer_duration(l_start, l_end);

      printf("GFLOP  = %.5g\n", flops*1e-9/(double)iters);
      printf("fp time = %.5g\n", ((double)(l_total/iters)));
      printf("GFLOPS  = %.5g\n", (flops*1e-9)/l_total);
#if defined(PROFILE)
      flops = (((2.0 * K * N * C) + (2.0 * K * N * K)) * 4.0) * (double)t * (double)iters;
      printf("BLAS GFLOP  = %.5g\n", flops*1e-9/(double)iters);
      printf("BLAS time = %.5g\n", ((double)(Gbl_blas_total/iters)));
      printf("BLAS GFLOPS  = %.5g\n", (flops*1e-9)/Gbl_blas_total);

      flops = ((tflops * K * N) * 4.0 + (4.0 * K * N) + (tflops * K * N)) * (double)t * (double)iters;
      printf("ELTWISE GFLOP  = %.5g\n", flops*1e-9/(double)iters);
      printf("ELTWISE time = %.5g\n", ((double)(Gbl_eltwise_total/iters)));
      printf("ELTWISE GFLOPS  = %.5g\n", (flops*1e-9)/Gbl_eltwise_total);

      flops = ((C * K  + K * K) * 4.0) * (double)iters;
      printf("CONVERSION GFLOP  = %.5g\n", flops*1e-9/(double)iters);
      printf("CONVERSION time = %.5g\n", ((double)(Gbl_conv_total/iters)));
      printf("CONVERSION GFLOPS  = %.5g\n", (flops*1e-9)/Gbl_conv_total);
#endif

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

      printf("##########################################\n");
      printf("# Performance - BWD+UPD (BLAS)           #\n");
      printf("##########################################\n");
      /* run LIBXSMM LSTM for performance */
#if defined(PROFILE)
      Gbl_blas_total = 0.0;
      Gbl_eltwise_total = 0.0;
#endif
      l_start = libxsmm_timer_tick();
      for (j = 0; j < iters; ++j) {
        lstm_ref_bwd_upd( N, C, K, t,
            xgoldt, cspgold, hpgold,
            csgoldt, cogoldt, hgoldt,
            icfogoldt, wgold, rgold,
            dcsgold, dhgoldt,
            dwgold, drgold, dbgold,
            dxgoldt, dcspgold, dhpgold, scratch_bu );
      }
      l_end = libxsmm_timer_tick();
      l_total = libxsmm_timer_duration(l_start, l_end);

      printf("GFLOP  = %.5g\n", flops*1e-9/(double)iters);
      printf("bp+wu time = %.5g\n", ((double)(l_total/iters)));
      printf("GFLOPS  = %.5g\n", (flops*1e-9)/l_total);
#if defined(PROFILE)
      flops = (((4.0 * K * N * C) + (4.0 * K * N * K)) * 4.0) * (double)t * (double)iters;
      printf("BLAS GFLOP  = %.5g\n", flops*1e-9/(double)iters);
      printf("BLAS time = %.5g\n", ((double)(Gbl_blas_total/iters)));
      printf("BLAS GFLOPS  = %.5g\n", (flops*1e-9)/Gbl_blas_total);

      flops = (19.0 * K * N) * (double)t * (double)iters;
      printf("ELTWISE GFLOP  = %.5g\n", flops*1e-9/(double)iters);
      printf("ELTWISE time = %.5g\n", ((double)(Gbl_eltwise_total/iters)));
      printf("ELTWISE GFLOPS  = %.5g\n", (flops*1e-9)/Gbl_eltwise_total);
#endif

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
  libxsmm_free(csgoldt);
  libxsmm_free(cogoldt);
  libxsmm_free(hgoldt);
  libxsmm_free(wgold);
  libxsmm_free(icfogoldt);
  libxsmm_free(dxgoldt);
  libxsmm_free(dcspgold);
  libxsmm_free(dhpgold);
  libxsmm_free(dwgold);
#if defined(TWO_GEMMS)
  libxsmm_free(rgold);
  libxsmm_free(drgold);
#endif
  libxsmm_free(scratch_fwd);
  libxsmm_free(scratch_bu);
  libxsmm_free(dbgold);
  libxsmm_free(dcsgold);
  libxsmm_free(dhgoldt);
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
  libxsmm_free(h_test);
  libxsmm_free(dxt_test);
  libxsmm_free(dw_test);
  libxsmm_free(dr_test);
  libxsmm_free(db_test);

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

