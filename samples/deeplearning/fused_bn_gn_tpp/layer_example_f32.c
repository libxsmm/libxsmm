/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Kirill Voronin (Intel Corp.)
******************************************************************************/
#include <libxsmm.h>
#include <libxsmm_sync.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>

/* include c-based dnn library */
#include "../common/dnn_common.h"
#include "bn_tpp_common.h"
#include "bn_tpp_fwd_custom_f32.h"
#include "bn_tpp_bwd_custom_f32.h"

#define COMPUTE_FP64_REFERENCE

int main( int argc, char* argv[] ) {

  my_bn_fwd_config my_bn_fwd;
  my_bn_bwd_config my_bn_bwd;

  naive_fusedbatchnorm_t naive_param;
  void *scratch;

  const float eps = FLT_EPSILON;
  libxsmm_blasint i, it;
  float *inp, *inp_add, *out, *dinp, *dout, *dinp_add, *eqn_dinp, *eqn_dout, *eqn_dinp_add, *dbeta, *eqn_dbeta, *dgamma, *eqn_dgamma, *eqn_out, *gamma, *beta, *cache_fl, *mean, *var, sum = 0.0;
  unsigned char *relumask_uncompressed, *relumask, *eqn_relumask;
  float *naive_inp, *naive_inp_add, *naive_out, *naive_rcpstdev, *naive_zeros, *naive_dinp, *naive_dout, *naive_dbeta, *naive_dgamma, *naive_dinp_add;
  unsigned char *naive_relumask;

#ifdef COMPUTE_FP64_REFERENCE
  double *naive_inp_fp64, *naive_inp_add_fp64, *naive_out_fp64, *naive_rcpstdev_fp64, *naive_zeros_fp64, *naive_dinp_fp64, *naive_dout_fp64, *naive_dbeta_fp64, *naive_dgamma_fp64, *naive_dinp_add_fp64;
  double *beta_fp64, *gamma_fp64, *mean_fp64, *var_fp64;
  double *dbeta_fp64, *dgamma_fp64;
  float *naive_out_fp64_downscaled_to_fp32, *out_fp64_downscaled_to_fp32;
  float *naive_dinp_fp64_downscaled_to_fp32, *dinp_fp64_downscaled_to_fp32;
  float *naive_dinp_add_fp64_downscaled_to_fp32, *dinp_add_fp64_downscaled_to_fp32;
  float *dgamma_fp64_downscaled_to_fp32;
  float *dbeta_fp64_downscaled_to_fp32;
#endif

  int iters     = 100;
  int N         = 28;
  int C         = 2 * 64;
  int H         = 0; /* defined later */
  int W         = 0; /* defined later */
  int HW        = 784;
  int bc        = 64;
  int CP        = 0; /* defined later */
  int stride    = 1; /* stride when accessing inputs */
  int pad_h_in  = 0; /* padding mode */
  int pad_w_in  = 0; /* padding mode */
  int pad_h_out = 0; /* padding mode */
  int pad_w_out = 0; /* padding mode */
  int norm_type = 0; /* 0: full batchnorm, 1: batch scaling only */
  int fuse_type = 5; /* 0: nothing fused, 1: relu fused, 2: ewise fused, 3: relu and ewise fused, 4: relu with mask, 5: relu and ewise with mask  */

  int stride_h = 0;  /* defined later */
  int stride_w = 0;  /* defined later */

  const char *const env_check = getenv("CHECK");
  const double check = LIBXSMM_ABS(0 == env_check ? 1 : atof(env_check));

#if defined(_OPENMP)
  int nThreads = omp_get_max_threads(); /* number of threads */
#else
  int nThreads = 1; /* number of threads */
#endif

  unsigned long long l_start, l_end;
  double l_total = 0, l_total2 = 0;
  double t_vec = 0, t_tpp = 0;

  libxsmm_matdiff_info norms_fwd, norms_bwd_d, norms_bwd_beta, norms_bwd_gamma;

  libxsmm_matdiff_clear(&norms_fwd);
  libxsmm_matdiff_clear(&norms_bwd_d);
  libxsmm_matdiff_clear(&norms_bwd_beta);
  libxsmm_matdiff_clear(&norms_bwd_gamma);

  if (argc > 1 && !strncmp(argv[1], "-h", 3)) {
    printf("Usage: %s iters N CP H W bc pad_w_in pad_h_in pad_w_out pad_h_out stride norm_type fuse_type (tail is optional) \n", argv[0]);
    return 0;
  }

  libxsmm_rng_set_seed(1);

  /* reading new values from cli */
  i = 1;
  if ( argc > i ) iters = atoi(argv[i++]);
  if ( argc > i ) N = atoi(argv[i++]);
  if ( argc > i ) C = atoi(argv[i++]);
  if ( argc > i ) H  = atoi(argv[i++]);
  if ( argc > i ) W  = atoi(argv[i++]);
  if ( argc > i ) bc = atoi(argv[i++]);
  if ( argc > i ) pad_w_in   = atoi(argv[i++]);
  if ( argc > i ) pad_h_in   = atoi(argv[i++]);
  if ( argc > i ) pad_w_out  = atoi(argv[i++]);
  if ( argc > i ) pad_h_out  = atoi(argv[i++]);
  if ( argc > i ) stride     = atoi(argv[i++]);
  if ( argc > i ) norm_type  = atoi(argv[i++]);
  if ( argc > i ) fuse_type  = atoi(argv[i++]);

  CP = C / bc;

  /* if H and W are read from cli, redefine HW */
  if (H && W)
    HW = H*W;
  else { /* else, set formally H and W from the previously set HW */
    H = HW;
    W = 1;
  }

  if (pad_w_in || pad_h_in || pad_w_out || pad_h_out) {
    printf("Padding is not supported (must be all 0)\n");
    return -1;
  }

  if ( stride != 1 ) {
    printf("Non-unit stride is not supported \n");
    return -1;
  }

/*
  if (fuse_type != 0 && fuse_type != 4 && fuse_type != 5) {
    printf("Unsupported fuse_type %d was provided (0, 4 and 5 are supported only)\n", fuse_type);
    return -1;
  }
*/

  stride_w = stride;
  stride_h = stride;

  /* set struct for naive batch normalization */
  naive_param.N = N;
  naive_param.C = CP*bc;
  naive_param.H = H;
  naive_param.W = W;
  naive_param.stride_h = stride_h;
  naive_param.stride_w = stride_w;
  naive_param.norm_type = norm_type; /* 0: full batchnorm, 1: batch scaling only */
  naive_param.fuse_type = fuse_type; /* 0: nothing fused, 1: relu fused, 2: elementwise fused, 3: relu and elementwise fused */

#if defined(__SSE3__)
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_ROUNDING_MODE(_MM_ROUND_NEAREST);
#endif

  /* print some summary */
  printf("##########################################\n");
  printf("#          Setting Up (Common)           #\n");
  printf("##########################################\n");
  printf("PARAMS: N:%d  C:%d  CP:%d bc:%d H:%d W:%d STRIDE:%d\n", N, CP*bc, CP, bc, H, W, stride);
  printf("PARAMS: FUSE TYPE:%d\n", fuse_type);
  printf("PARAMS: ITERS:%d", iters); if (LIBXSMM_FEQ(0, check)) printf("  Threads:%d\n", nThreads); else printf("\n");
  printf("SIZE Input  (MB): %10.2f MiB\n", (double)(N*CP*HW*bc*sizeof(float))/(1024.0*1024.0) );
  printf("SIZE Output (MB): %10.2f MiB\n", (double)(N*CP*HW*bc*sizeof(float))/(1024.0*1024.0) );

  /* allocate data */
  inp        = (float*) libxsmm_aligned_malloc( sizeof(float)*N*CP*HW*bc,   2097152);
  out        = (float*) libxsmm_aligned_malloc( sizeof(float)*N*CP*HW*bc,   2097152);
  inp_add    = (float*) libxsmm_aligned_malloc( sizeof(float)*N*CP*HW*bc,   2097152);
  dinp       = (float*) libxsmm_aligned_malloc( sizeof(float)*N*CP*HW*bc,   2097152);
  dout       = (float*) libxsmm_aligned_malloc( sizeof(float)*N*CP*HW*bc,   2097152);
  dinp_add   = (float*) libxsmm_aligned_malloc( sizeof(float)*N*CP*HW*bc,   2097152);
  dgamma     = (float*) libxsmm_aligned_malloc( sizeof(float)*CP*bc,   2097152);
  dbeta      = (float*) libxsmm_aligned_malloc( sizeof(float)*CP*bc,   2097152);
  eqn_dinp   = (float*) libxsmm_aligned_malloc( sizeof(float)*N*CP*HW*bc,   2097152);
  eqn_dout   = (float*) libxsmm_aligned_malloc( sizeof(float)*N*CP*HW*bc,   2097152);
  eqn_dgamma = (float*) libxsmm_aligned_malloc( sizeof(float)*CP*bc,   2097152);
  eqn_dbeta  = (float*) libxsmm_aligned_malloc( sizeof(float)*CP*bc,   2097152);
  eqn_dinp_add = (float*) libxsmm_aligned_malloc( sizeof(float)*N*CP*HW*bc,   2097152);
  gamma      = (float*) libxsmm_aligned_malloc( sizeof(float)*CP*bc,   2097152);
  beta       = (float*) libxsmm_aligned_malloc( sizeof(float)*CP*bc,   2097152);
  mean       = (float*) libxsmm_aligned_malloc( sizeof(float)*CP*bc,   2097152);
  var        = (float*) libxsmm_aligned_malloc( sizeof(float)*CP*bc,   2097152);
  eqn_out    = (float*) libxsmm_aligned_malloc( sizeof(float)*N*CP*HW*bc,   2097152);
  cache_fl   = (float*) libxsmm_aligned_malloc( sizeof(float)*1024*1024,   2097152);

  relumask     = (unsigned char*) libxsmm_aligned_malloc( sizeof(unsigned char)*N*CP*HW*bc, 2097152);
  relumask_uncompressed = (unsigned char*) libxsmm_aligned_malloc( sizeof(unsigned char)*N*CP*HW*bc, 2097152);
  eqn_relumask = (unsigned char*) libxsmm_aligned_malloc( sizeof(unsigned char)*N*CP*HW*bc, 2097152);

  naive_inp      = (float*) libxsmm_aligned_malloc( sizeof(float)*N*C*H*W, 2097152);
  naive_out      = (float*) libxsmm_aligned_malloc( sizeof(float)*N*C*H*W, 2097152);
  naive_inp_add  = (float*) libxsmm_aligned_malloc( sizeof(float)*N*C*H*W, 2097152);
  naive_dinp     = (float*) libxsmm_aligned_malloc( sizeof(float)*N*C*H*W, 2097152);
  naive_dout     = (float*) libxsmm_aligned_malloc( sizeof(float)*N*C*H*W, 2097152);
  naive_dinp_add = (float*) libxsmm_aligned_malloc( sizeof(float)*N*C*H*W, 2097152);
  naive_dgamma   = (float*) libxsmm_aligned_malloc( sizeof(float)*C,       2097152);
  naive_dbeta    = (float*) libxsmm_aligned_malloc( sizeof(float)*C,       2097152);
  naive_rcpstdev = (float*) libxsmm_aligned_malloc( sizeof(float)*C,       2097152);
  naive_zeros    = (float*) libxsmm_aligned_malloc( sizeof(float)*N*C*H*W, 2097152);

  naive_relumask = (unsigned char*) libxsmm_aligned_malloc( sizeof(unsigned char)*N*C*H*W, 2097152);

#ifdef COMPUTE_FP64_REFERENCE
  naive_inp_fp64      = (double*) libxsmm_aligned_malloc( sizeof(double)*N*C*H*W, 2097152);
  naive_inp_add_fp64  = (double*) libxsmm_aligned_malloc( sizeof(double)*N*C*H*W, 2097152);
  naive_out_fp64      = (double*) libxsmm_aligned_malloc( sizeof(double)*N*C*H*W, 2097152);
  naive_dinp_fp64     = (double*) libxsmm_aligned_malloc( sizeof(double)*N*C*H*W,   2097152);
  naive_dout_fp64     = (double*) libxsmm_aligned_malloc( sizeof(double)*N*C*H*W,   2097152);
  naive_dinp_add_fp64 = (double*) libxsmm_aligned_malloc( sizeof(double)*N*C*H*W,   2097152);
  naive_dgamma_fp64   = (double*) libxsmm_aligned_malloc( sizeof(double)*C,   2097152);
  naive_dbeta_fp64    = (double*) libxsmm_aligned_malloc( sizeof(double)*C,   2097152);
  naive_rcpstdev_fp64 = (double*) libxsmm_aligned_malloc( sizeof(double)*C,   2097152);
  naive_zeros_fp64    = (double*) libxsmm_aligned_malloc( sizeof(double)*N*C*H*W, 2097152);

  gamma_fp64  = (double*) libxsmm_aligned_malloc( sizeof(double)*CP*bc,   2097152);
  beta_fp64   = (double*) libxsmm_aligned_malloc( sizeof(double)*CP*bc,   2097152);
  mean_fp64   = (double*) libxsmm_aligned_malloc( sizeof(double)*CP*bc,   2097152);
  var_fp64    = (double*) libxsmm_aligned_malloc( sizeof(double)*CP*bc,   2097152);

  dgamma_fp64 = (double*) libxsmm_aligned_malloc( sizeof(double)*CP*bc,   2097152);
  dbeta_fp64  = (double*) libxsmm_aligned_malloc( sizeof(double)*CP*bc,   2097152);
  dgamma_fp64_downscaled_to_fp32     = (float*) libxsmm_aligned_malloc( sizeof(float)*(CP*bc)*1, 2097152);
  dbeta_fp64_downscaled_to_fp32      = (float*) libxsmm_aligned_malloc( sizeof(float)*(CP*bc)*1, 2097152);

  naive_out_fp64_downscaled_to_fp32  = (float*) libxsmm_aligned_malloc( sizeof(float)*N*C*H*W, 2097152);
  naive_dinp_fp64_downscaled_to_fp32 = (float*) libxsmm_aligned_malloc( sizeof(float)*N*C*H*W, 2097152);
  naive_dinp_add_fp64_downscaled_to_fp32 = (float*) libxsmm_aligned_malloc( sizeof(float)*N*C*H*W, 2097152);

  out_fp64_downscaled_to_fp32        = (float*) libxsmm_aligned_malloc( sizeof(float)*N*(CP*bc)*HW*1, 2097152);
  dinp_fp64_downscaled_to_fp32       = (float*) libxsmm_aligned_malloc( sizeof(float)*N*(CP*bc)*HW*1, 2097152);
  dinp_add_fp64_downscaled_to_fp32   = (float*) libxsmm_aligned_malloc( sizeof(float)*N*(CP*bc)*HW*1, 2097152);

#endif

  /* initialize data */
  init_buf(inp,      N*CP*HW*bc, 1, 0);
  //init_buf(out,      N*CP*HW*bc, 1, 0);
  init_buf(inp_add,  N*CP*HW*bc, 1, 0);
  init_buf(dinp,     N*CP*HW*bc, 1, 0);
  init_buf(dout,     N*CP*HW*bc, 1, 0);
  //init_buf(dinp_add, N*CP*HW*bc, 1, 0);

  //copy_buf(out,      eqn_out,      N*CP*HW*bc);
  //copy_buf(dinp,     eqn_dinp,     N*CP*HW*bc);
  copy_buf(dout,     eqn_dout,     N*CP*HW*bc);

  zero_buf(naive_zeros, N*C*H*W);
#ifdef COMPUTE_FP64_REFERENCE
  zero_buf_fp64(naive_zeros_fp64, N*C*H*W);
#endif

  init_buf(gamma,  CP*bc, 1, 0);
  init_buf(beta,   CP*bc, 1, 0);
  init_buf(dgamma, CP*bc, 1, 0);
  init_buf(dbeta,  CP*bc, 1, 0);
  copy_buf(dgamma, eqn_dgamma, CP*bc);
  copy_buf(dbeta,  eqn_dbeta,  CP*bc);
#ifdef COMPUTE_FP64_REFERENCE
  extend_buf_fp32_to_fp64(gamma, gamma_fp64, CP*bc);
  extend_buf_fp32_to_fp64(beta,  beta_fp64,  CP*bc);
#endif

  zero_buf_uint8(relumask, N*CP*HW*bc);
  zero_buf_uint8(relumask_uncompressed, N*CP*HW*bc);

  init_buf(cache_fl,  1024*1024, 1, 0);

  /* setup TPPs (standalone or through the configs) */

  my_bn_fwd = setup_my_bn_fwd(N, C, H, W, bc, nThreads, (my_normalization_fuse)fuse_type );
  my_bn_bwd = setup_my_bn_bwd(N, C, H, W, bc, nThreads, (my_normalization_fuse)fuse_type );

  /* allocate and bind scratch */
  if ( my_bn_fwd.scratch_size > 0 || my_bn_bwd.scratch_size > 0 ) {
    size_t alloc_size = LIBXSMM_MAX( my_bn_fwd.scratch_size, my_bn_bwd.scratch_size);
    scratch = libxsmm_aligned_malloc( alloc_size, 2097152 );
    init_buf( (float*)(scratch), (alloc_size)/4, 0, 0 );
  }

  /* Check correctness */
  if (LIBXSMM_NEQ(0, check)) {
#if defined(_OPENMP)
#   pragma omp parallel
#endif
    {
#if defined(_OPENMP)
      const int tid = omp_get_thread_num();
#else
      const int tid = 0;
#endif
      my_bn_fwd_exec( my_bn_fwd, inp, inp_add, gamma, beta, mean, var, eqn_out, eqn_relumask, eps, 0, tid, scratch);
    }

    tensor_copy_NCHWc_to_NCHW (inp,     naive_inp,     N, C, H, W, bc);
    tensor_copy_NCHWc_to_NCHW (inp_add, naive_inp_add, N, C, H, W, bc);

    naive_fusedbatchnorm_fp(&naive_param, naive_inp, naive_out, naive_inp_add,
                                        beta, gamma, eps, mean, naive_rcpstdev, var, naive_relumask);

    tensor_copy_NCHW_to_NCHWc       (naive_out     , out,      N, C, H, W, bc);
    tensor_copy_NCHW_to_NCHWc_uint8 (naive_relumask, relumask_uncompressed, N, C, H, W, bc);
    mask_compress_uint8 (relumask_uncompressed, relumask, N*CP*H*W*bc);

#ifdef COMPUTE_FP64_REFERENCE
    extend_buf_fp32_to_fp64 (naive_inp,     naive_inp_fp64,     N*C*H*W);
    extend_buf_fp32_to_fp64 (naive_inp_add, naive_inp_add_fp64, N*C*H*W);

    naive_fusedbatchnorm_fp_fp64(&naive_param, naive_inp_fp64, naive_out_fp64, naive_inp_add_fp64,
                                        beta_fp64, gamma_fp64, eps, mean_fp64, naive_rcpstdev_fp64, var_fp64, naive_relumask);

    truncate_buf_fp64_to_fp32 (naive_out_fp64, naive_out_fp64_downscaled_to_fp32, N*C*H*W);

    tensor_copy_NCHW_to_NCHWc (naive_out_fp64_downscaled_to_fp32, out_fp64_downscaled_to_fp32, N, C, H, W, bc);

    tensor_copy_NCHW_to_NCHWc_uint8 (naive_relumask, relumask_uncompressed, N, C, H, W, bc);
    mask_compress_uint8 (relumask_uncompressed, relumask, N*CP*H*W*bc);
#endif

    /* compare */
    printf("############################################\n");
    printf("# Correctness FP32 FWD Batchnorm - Output  #\n");
    printf("############################################\n");
    libxsmm_matdiff(&norms_fwd, LIBXSMM_DATATYPE_F32, N*CP*HW*bc, 1, out, eqn_out, 0, 0);
    printf("L1 reference  : %.25g\n", norms_fwd.l1_ref);
    printf("L1 test       : %.25g\n", norms_fwd.l1_tst);
    printf("L2 abs.error  : %.24f\n", norms_fwd.l2_abs);
    printf("L2 rel.error  : %.24f\n", norms_fwd.l2_rel);
    printf("Linf abs.error: %.24f\n", norms_fwd.linf_abs);
    printf("Linf rel.error: %.24f\n", norms_fwd.linf_rel);
    printf("Check-norm    : %.24f\n\n", norms_fwd.normf_rel);

#ifdef COMPUTE_FP64_REFERENCE
    printf("##################################################\n");
    printf("# Correctness FP32 FWD Batchnorm - Output (fp64) #\n");
    printf("##################################################\n");
    libxsmm_matdiff(&norms_fwd, LIBXSMM_DATATYPE_F32, N*CP*HW*bc, 1, out_fp64_downscaled_to_fp32, eqn_out, 0, 0);
    printf("L1 reference  : %.25g\n", norms_fwd.l1_ref);
    printf("L1 test       : %.25g\n", norms_fwd.l1_tst);
    printf("L2 abs.error  : %.24f\n", norms_fwd.l2_abs);
    printf("L2 rel.error  : %.24f\n", norms_fwd.l2_rel);
    printf("Linf abs.error: %.24f\n", norms_fwd.linf_abs);
    printf("Linf rel.error: %.24f\n", norms_fwd.linf_rel);
    printf("Check-norm    : %.24f\n\n", norms_fwd.normf_rel);
#endif

    if (fuse_type == 4 || fuse_type == 5) {
      printf("############################################\n");
      printf("# Correctness FP32 FWD Batchnorm - Relumask  #\n");
      printf("############################################\n");
      libxsmm_matdiff(&norms_fwd, LIBXSMM_DATATYPE_I8, N*CP*HW*bc, 1, relumask, eqn_relumask, 0, 0);
      printf("L1 reference  : %.25g\n", norms_fwd.l1_ref);
      printf("L1 test       : %.25g\n", norms_fwd.l1_tst);
      printf("L2 abs.error  : %.24f\n", norms_fwd.l2_abs);
      printf("L2 rel.error  : %.24f\n", norms_fwd.l2_rel);
      printf("Linf abs.error: %.24f\n", norms_fwd.linf_abs);
      printf("Linf rel.error: %.24f\n", norms_fwd.linf_rel);
      printf("Check-norm    : %.24f\n\n", norms_fwd.normf_rel);
    }
  } /* checking correctness for FWD */

  for (i = 0; i < 1024 * 1024; i++ ) {
    sum += cache_fl[i];
  }
  naive_fusedbatchnorm_fp(&naive_param, naive_inp, naive_out, naive_inp_add,
                                        beta, gamma, eps, mean, naive_rcpstdev, var, naive_relumask);
  l_start = libxsmm_timer_tick();
  for (it = 0; it < iters; it++) {
    naive_fusedbatchnorm_fp(&naive_param, naive_inp, naive_out, naive_inp_add,
                                        beta, gamma, eps, mean, naive_rcpstdev, var, naive_relumask);
  }
  l_end = libxsmm_timer_tick();
  l_total = libxsmm_timer_duration(l_start, l_end);
  printf("Scaler batchnorm time FWD  = %.5g\n", ((double)(l_total)));
  for (i = 0; i < 1024 * 1024; i++ ) {
    sum += cache_fl[i] + (float)l_total;
  }
#if defined(_OPENMP)
#   pragma omp parallel
#endif
    {
#if defined(_OPENMP)
      const int tid = omp_get_thread_num();
#else
      const int tid = 0;
#endif
      my_bn_fwd_exec( my_bn_fwd, inp, inp_add, gamma, beta, mean, var, eqn_out, eqn_relumask, eps, 0, tid, scratch);
    }
  l_start = libxsmm_timer_tick();
  for (it = 0; it < iters; it++) {
#if defined(_OPENMP)
#   pragma omp parallel
#endif
    {
#if defined(_OPENMP)
      const int tid = omp_get_thread_num();
#else
      const int tid = 0;
#endif
      my_bn_fwd_exec( my_bn_fwd, inp, inp_add, gamma, beta, mean, var, eqn_out, eqn_relumask, eps, 0, tid, scratch );
    }
  }
  l_end = libxsmm_timer_tick();
  l_total2 = libxsmm_timer_duration(l_start, l_end);
  printf("TPP batchnorm time FWD  = %.5g\n", ((double)(l_total2)));
  printf("Speedup FWD is %.5g\n", l_total/l_total2);

  if (LIBXSMM_NEQ(0, check)) {
#if defined(_OPENMP)
#   pragma omp parallel
#endif
    {
#if defined(_OPENMP)
      const int tid = omp_get_thread_num();
#else
      const int tid = 0;
#endif

      my_bn_bwd_exec( my_bn_bwd, eqn_dout, inp, mean, var, gamma, relumask, eqn_dinp, eqn_dinp_add, eqn_dgamma, eqn_dbeta, eps, 0, tid, scratch );
    }

    tensor_copy_NCHWc_to_NCHW (inp,  naive_inp,   N, C, H, W, bc);
    tensor_copy_NCHWc_to_NCHW (out,  naive_out,   N, C, H, W, bc);
    tensor_copy_NCHWc_to_NCHW (dout, naive_dout,  N, C, H, W, bc);

    naive_fusedbatchnorm_bp(&naive_param, naive_inp, naive_dinp, naive_out, naive_dout, naive_dinp_add,
                                       beta, dbeta, gamma, dgamma, mean, naive_rcpstdev);

    tensor_copy_NCHW_to_NCHWc (naive_dinp,     dinp,     N, C, H, W, bc);
    tensor_copy_NCHW_to_NCHWc (naive_dinp_add, dinp_add, N, C, H, W, bc);

#ifdef COMPUTE_FP64_REFERENCE
    extend_buf_fp32_to_fp64 (naive_inp,  naive_inp_fp64,  N*C*H*W);
    extend_buf_fp32_to_fp64 (naive_out,  naive_out_fp64,  N*C*H*W);
    extend_buf_fp32_to_fp64 (naive_dout, naive_dout_fp64, N*C*H*W);

    naive_fusedbatchnorm_bp_fp64(&naive_param, naive_inp_fp64, naive_dinp_fp64, naive_out_fp64, naive_dout_fp64, naive_dinp_add_fp64,
                                       beta_fp64, dbeta_fp64, gamma_fp64, dgamma_fp64, mean_fp64, naive_rcpstdev_fp64);

    truncate_buf_fp64_to_fp32 (naive_dinp_fp64,     naive_dinp_fp64_downscaled_to_fp32,     N*C*H*W);
    truncate_buf_fp64_to_fp32 (naive_dinp_add_fp64, naive_dinp_add_fp64_downscaled_to_fp32, N*C*H*W);
    truncate_buf_fp64_to_fp32 (dgamma_fp64, dgamma_fp64_downscaled_to_fp32, CP*bc);
    truncate_buf_fp64_to_fp32 (dbeta_fp64,  dbeta_fp64_downscaled_to_fp32, CP*bc);

    tensor_copy_NCHW_to_NCHWc (naive_dinp_fp64_downscaled_to_fp32,     dinp_fp64_downscaled_to_fp32,     N, C, H, W, bc);
    tensor_copy_NCHW_to_NCHWc (naive_dinp_add_fp64_downscaled_to_fp32, dinp_add_fp64_downscaled_to_fp32, N, C, H, W, bc);
#endif

    /* compare */
    printf("############################################\n");
    printf("# Correctness FP32 BWD Batchnorm - Dinput  #\n");
    printf("############################################\n");
    libxsmm_matdiff(&norms_bwd_d, LIBXSMM_DATATYPE_F32, N*CP*HW*bc, 1, dinp, eqn_dinp, 0, 0);
    printf("L1 reference  : %.25g\n", norms_bwd_d.l1_ref);
    printf("L1 test       : %.25g\n", norms_bwd_d.l1_tst);
    printf("L2 abs.error  : %.24f\n", norms_bwd_d.l2_abs);
    printf("L2 rel.error  : %.24f\n", norms_bwd_d.l2_rel);
    printf("Linf abs.error: %.24f\n", norms_bwd_d.linf_abs);
    printf("Linf rel.error: %.24f\n", norms_bwd_d.linf_rel);
    printf("Check-norm    : %.24f\n\n", norms_bwd_d.normf_rel);

#ifdef COMPUTE_FP64_REFERENCE
    printf("##################################################\n");
    printf("# Correctness FP32 BWD Batchnorm - Dinput (fp64) #\n");
    printf("##################################################\n");
    libxsmm_matdiff(&norms_bwd_d, LIBXSMM_DATATYPE_F32, N*CP*HW*bc, 1, dinp_fp64_downscaled_to_fp32, eqn_dinp, 0, 0);
    printf("L1 reference  : %.25g\n", norms_bwd_d.l1_ref);
    printf("L1 test       : %.25g\n", norms_bwd_d.l1_tst);
    printf("L2 abs.error  : %.24f\n", norms_bwd_d.l2_abs);
    printf("L2 rel.error  : %.24f\n", norms_bwd_d.l2_rel);
    printf("Linf abs.error: %.24f\n", norms_bwd_d.linf_abs);
    printf("Linf rel.error: %.24f\n", norms_bwd_d.linf_rel);
    printf("Check-norm    : %.24f\n\n", norms_bwd_d.normf_rel);
#endif

    if (fuse_type == 2 || fuse_type == 5) {
      printf("################################################\n");
      printf("# Correctness FP32 BWD Batchnorm - Dinput add  #\n");
      printf("################################################\n");
      libxsmm_matdiff(&norms_bwd_d, LIBXSMM_DATATYPE_F32, N*CP*HW*bc, 1, dinp_add, eqn_dinp_add, 0, 0);
      printf("L1 reference  : %.25g\n", norms_bwd_d.l1_ref);
      printf("L1 test       : %.25g\n", norms_bwd_d.l1_tst);
      printf("L2 abs.error  : %.24f\n", norms_bwd_d.l2_abs);
      printf("L2 rel.error  : %.24f\n", norms_bwd_d.l2_rel);
      printf("Linf abs.error: %.24f\n", norms_bwd_d.linf_abs);
      printf("Linf rel.error: %.24f\n", norms_bwd_d.linf_rel);
      printf("Check-norm    : %.24f\n\n", norms_bwd_d.normf_rel);

#ifdef COMPUTE_FP64_REFERENCE
      printf("##################################################\n");
      printf("# Correctness FP32 BWD Batchnorm - Dinput add (fp64) #\n");
      printf("##################################################\n");
      libxsmm_matdiff(&norms_bwd_d, LIBXSMM_DATATYPE_F32, N*CP*HW*bc, 1, dinp_add_fp64_downscaled_to_fp32, eqn_dinp_add, 0, 0);
      printf("L1 reference  : %.25g\n", norms_bwd_d.l1_ref);
      printf("L1 test       : %.25g\n", norms_bwd_d.l1_tst);
      printf("L2 abs.error  : %.24f\n", norms_bwd_d.l2_abs);
      printf("L2 rel.error  : %.24f\n", norms_bwd_d.l2_rel);
      printf("Linf abs.error: %.24f\n", norms_bwd_d.linf_abs);
      printf("Linf rel.error: %.24f\n", norms_bwd_d.linf_rel);
      printf("Check-norm    : %.24f\n\n", norms_bwd_d.normf_rel);
#endif
    }

    printf("###########################################\n");
    printf("# Correctness FP32 BWD Batchnorm - Dbeta  #\n");
    printf("###########################################\n");
    libxsmm_matdiff(&norms_bwd_beta, LIBXSMM_DATATYPE_F32, CP*bc, 1, dbeta, eqn_dbeta, 0, 0);
    printf("L1 reference  : %.25g\n", norms_bwd_beta.l1_ref);
    printf("L1 test       : %.25g\n", norms_bwd_beta.l1_tst);
    printf("L2 abs.error  : %.24f\n", norms_bwd_beta.l2_abs);
    printf("L2 rel.error  : %.24f\n", norms_bwd_beta.l2_rel);
    printf("Linf abs.error: %.24f\n", norms_bwd_beta.linf_abs);
    printf("Linf rel.error: %.24f\n", norms_bwd_beta.linf_rel);
    printf("Check-norm    : %.24f\n\n", norms_bwd_beta.normf_rel);

#ifdef COMPUTE_FP64_REFERENCE
    printf("##################################################\n");
    printf("# Correctness FP32 BWD Batchnorm - Dbeta (fp64)  #\n");
    printf("##################################################\n");
    libxsmm_matdiff(&norms_bwd_beta, LIBXSMM_DATATYPE_F32, CP*bc, 1, dbeta_fp64_downscaled_to_fp32, eqn_dbeta, 0, 0);
    printf("L1 reference  : %.25g\n", norms_bwd_beta.l1_ref);
    printf("L1 test       : %.25g\n", norms_bwd_beta.l1_tst);
    printf("L2 abs.error  : %.24f\n", norms_bwd_beta.l2_abs);
    printf("L2 rel.error  : %.24f\n", norms_bwd_beta.l2_rel);
    printf("Linf abs.error: %.24f\n", norms_bwd_beta.linf_abs);
    printf("Linf rel.error: %.24f\n", norms_bwd_beta.linf_rel);
    printf("Check-norm    : %.24f\n\n", norms_bwd_beta.normf_rel);
#endif

    printf("############################################\n");
    printf("# Correctness FP32 BWD Batchnorm - Dgamma  #\n");
    printf("############################################\n");
    libxsmm_matdiff(&norms_bwd_gamma, LIBXSMM_DATATYPE_F32, CP*bc, 1, dgamma, eqn_dgamma, 0, 0);
    printf("L1 reference  : %.25g\n", norms_bwd_gamma.l1_ref);
    printf("L1 test       : %.25g\n", norms_bwd_gamma.l1_tst);
    printf("L2 abs.error  : %.24f\n", norms_bwd_gamma.l2_abs);
    printf("L2 rel.error  : %.24f\n", norms_bwd_gamma.l2_rel);
    printf("Linf abs.error: %.24f\n", norms_bwd_gamma.linf_abs);
    printf("Linf rel.error: %.24f\n", norms_bwd_gamma.linf_rel);
    printf("Check-norm    : %.24f\n\n", norms_bwd_gamma.normf_rel);

#ifdef COMPUTE_FP64_REFERENCE
    printf("##################################################\n");
    printf("# Correctness FP32 BWD Batchnorm - Dgamma (fp64) #\n");
    printf("##################################################\n");
    libxsmm_matdiff(&norms_bwd_gamma, LIBXSMM_DATATYPE_F32, CP*bc, 1, dgamma_fp64_downscaled_to_fp32, eqn_dgamma, 0, 0);
    printf("L1 reference  : %.25g\n", norms_bwd_gamma.l1_ref);
    printf("L1 test       : %.25g\n", norms_bwd_gamma.l1_tst);
    printf("L2 abs.error  : %.24f\n", norms_bwd_gamma.l2_abs);
    printf("L2 rel.error  : %.24f\n", norms_bwd_gamma.l2_rel);
    printf("Linf abs.error: %.24f\n", norms_bwd_gamma.linf_abs);
    printf("Linf rel.error: %.24f\n", norms_bwd_gamma.linf_rel);
    printf("Check-norm    : %.24f\n\n", norms_bwd_gamma.normf_rel);
#endif
  } /* correctness for BWD */

  for (i = 0; i < 1024 * 1024; i++ ) {
    sum += cache_fl[i];
  }
  naive_fusedbatchnorm_bp(&naive_param, naive_inp, naive_dinp, naive_out, naive_dout, naive_dinp_add,
                                       beta, dbeta, gamma, dgamma, mean, naive_rcpstdev);
  l_start = libxsmm_timer_tick();
  for (it = 0; it < iters; it++) {
    naive_fusedbatchnorm_bp(&naive_param, naive_inp, naive_dinp, naive_out, naive_dout, naive_dinp_add,
                                       beta, dbeta, gamma, dgamma, mean, naive_rcpstdev);
  }
  l_end = libxsmm_timer_tick();
  l_total = libxsmm_timer_duration(l_start, l_end);
  printf("Scaler batchnorm time BWD = %.5g\n", ((double)(l_total)));
  for (i = 0; i < 1024 * 1024; i++ ) {
    sum += cache_fl[i] + (float)l_total;
  }
#if defined(_OPENMP)
#   pragma omp parallel
#endif
    {
#if defined(_OPENMP)
      const int tid = omp_get_thread_num();
#else
      const int tid = 0;
#endif
      my_bn_bwd_exec( my_bn_bwd, eqn_dout, inp, mean, var, gamma, relumask, eqn_dinp, eqn_dinp_add, eqn_dgamma, eqn_dbeta, eps, 0, tid, scratch );
    }
  l_start = libxsmm_timer_tick();

  for (it = 0; it < iters; it++) {
#if defined(_OPENMP)
#   pragma omp parallel
#endif
    {
#if defined(_OPENMP)
      const int tid = omp_get_thread_num();
#else
      const int tid = 0;
#endif
      my_bn_bwd_exec( my_bn_bwd, eqn_dout, inp, mean, var, gamma, relumask, eqn_dinp, eqn_dinp_add, eqn_dgamma, eqn_dbeta, eps, 0, tid, scratch );
    }
  }

  l_end = libxsmm_timer_tick();
  l_total2 = libxsmm_timer_duration(l_start, l_end);
  printf("TPP batchnorm time BWD = %.5g\n", ((double)(l_total2)));
  printf("Speedup BWD is %.5g\n", l_total/l_total2);
  /* printf("Running sum is %.5f\n", sum); */

  t_tpp += l_total2;
  t_vec += l_total;

  printf("\n\n=================================\n");
  printf("Total Speedup via TPP Matrix equation is %.5g\n", t_vec/t_tpp);
  printf("=================================\n");

  /* deallocate data */
  if ( scratch != NULL ) {
    libxsmm_free(scratch);
  }
  libxsmm_free(inp);
  libxsmm_free(out);
  libxsmm_free(inp_add);
  libxsmm_free(dinp);
  libxsmm_free(dout);
  libxsmm_free(dinp_add);
  libxsmm_free(eqn_dinp);
  libxsmm_free(eqn_dout);
  libxsmm_free(eqn_dinp_add);
  libxsmm_free(dgamma);
  libxsmm_free(dbeta);
  libxsmm_free(eqn_dgamma);
  libxsmm_free(eqn_dbeta);
  libxsmm_free(mean);
  libxsmm_free(var);
  libxsmm_free(gamma);
  libxsmm_free(beta);
  libxsmm_free(eqn_out);
  libxsmm_free(cache_fl);

  libxsmm_free(relumask);
  libxsmm_free(relumask_uncompressed);
  libxsmm_free(eqn_relumask);

  libxsmm_free(naive_inp);
  libxsmm_free(naive_out);
  libxsmm_free(naive_inp_add);
  libxsmm_free(naive_dinp);
  libxsmm_free(naive_dout);
  libxsmm_free(naive_dgamma);
  libxsmm_free(naive_dbeta);
  libxsmm_free(naive_rcpstdev);
  libxsmm_free(naive_zeros);

  libxsmm_free(naive_relumask);

#ifdef COMPUTE_FP64_REFERENCE
  libxsmm_free(naive_inp_fp64);
  libxsmm_free(naive_out_fp64);
  libxsmm_free(naive_inp_add_fp64);
  libxsmm_free(naive_rcpstdev_fp64);
  libxsmm_free(naive_zeros_fp64);
  libxsmm_free(naive_dinp_fp64);
  libxsmm_free(naive_dout_fp64);
  libxsmm_free(naive_dinp_add_fp64);
  libxsmm_free(naive_dbeta_fp64);
  libxsmm_free(naive_dgamma_fp64);

  libxsmm_free(beta_fp64);
  libxsmm_free(gamma_fp64);
  libxsmm_free(mean_fp64);
  libxsmm_free(var_fp64);

  libxsmm_free(dbeta_fp64);
  libxsmm_free(dgamma_fp64);

  libxsmm_free(naive_out_fp64_downscaled_to_fp32);
  libxsmm_free(out_fp64_downscaled_to_fp32);
  libxsmm_free(naive_dinp_fp64_downscaled_to_fp32);
  libxsmm_free(dinp_fp64_downscaled_to_fp32);
  libxsmm_free(naive_dinp_add_fp64_downscaled_to_fp32);
  libxsmm_free(dinp_add_fp64_downscaled_to_fp32);
  libxsmm_free(dgamma_fp64_downscaled_to_fp32);
  libxsmm_free(dbeta_fp64_downscaled_to_fp32);
#endif

  return 0;
}

