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
#include <libxsmm_dnn.h>
#include <dnn_common.h>

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#if defined(_OPENMP)
# include <omp.h>
#endif

int main( int argc, char* argv[] ) {

  libxsmm_dnn_bn_fwd_config libxsmm_dnn_bn_fwd;
  libxsmm_dnn_bn_bwd_config libxsmm_dnn_bn_bwd;

  naive_fusedbatchnorm_t naive_param;
  void *scratch = NULL;

  const float eps = FLT_EPSILON;
  libxsmm_blasint i, it;
  float            *eqn_inp,      *eqn_inp_add,      *eqn_dinp,      *eqn_dout,      *eqn_dinp_add,      *eqn_out;
  libxsmm_bfloat16 *eqn_inp_bf16, *eqn_inp_add_bf16, *eqn_dinp_bf16, *eqn_dout_bf16, *eqn_dinp_add_bf16, *eqn_out_bf16;

  float            *naive_eqn_out, *naive_eqn_dinp, *naive_eqn_dinp_add, *naive_eqn_dout; /* LIBXSMM output tensors (NCHWc) reordered into NCHW (bf16) */

  float *eqn_dbeta, *eqn_dgamma, *gamma, *beta, *naive_mean, *naive_var, *eqn_mean, *eqn_var, sum = 0.0;
  float *cache_fl;
  unsigned char *relumask_uncompressed, *relumask, *eqn_relumask;
  float            *naive_inp,      *naive_inp_add,      *naive_out,      *naive_dinp,      *naive_dout,      *naive_dinp_add;
  libxsmm_bfloat16 *naive_inp_bf16, *naive_inp_add_bf16, *naive_out_bf16, *naive_dinp_bf16, *naive_dout_bf16, *naive_dinp_add_bf16;
  float *naive_rcpstdev, *naive_dbeta, *naive_dgamma;
  unsigned char *naive_relumask;

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
  int norm_type = 0; /* 0: full batchnorm, 1: batch scaling only for fwd / only input gradient for bwd */
  int fuse_type = 5; /* 0: nothing fused, 1: relu fused, 2: ewise fused, 3: relu and ewise fused, 4: relu with mask, 5: relu and ewise with mask  */

  int stride_h = 0;  /* defined later */
  int stride_w = 0;  /* defined later */

  int prec_bf16 = 0;

  libxsmm_datatype bn_dtype = LIBXSMM_DATATYPE_F32;

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

  libxsmm_matdiff_info norms_fwd_out, norms_fwd_mean, norms_fwd_var, norms_fwd_mask, norms_bwd_din, norms_bwd_dout, norms_bwd_beta, norms_bwd_gamma;

  libxsmm_matdiff_clear(&norms_fwd_out);
  libxsmm_matdiff_clear(&norms_fwd_mean);
  libxsmm_matdiff_clear(&norms_fwd_var);
  libxsmm_matdiff_clear(&norms_fwd_mask);
  libxsmm_matdiff_clear(&norms_bwd_din);
  libxsmm_matdiff_clear(&norms_bwd_dout);
  libxsmm_matdiff_clear(&norms_bwd_beta);
  libxsmm_matdiff_clear(&norms_bwd_gamma);

  if (argc > 1 && !strncmp(argv[1], "-h", 3)) {
    printf("Usage: %s iters N CP H W bc pad_w_in pad_h_in pad_w_out pad_h_out stride norm_type fuse_type prec_bf16 (tail is optional) \n", argv[0]);
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
  if ( argc > i ) prec_bf16  = atoi(argv[i++]);

  CP = C / bc;

  if ( C % bc != 0 || CP == 0 ) {
    printf("Bad input channel blocking: C = %d bc = %d \n", C, bc);
    return -1;
  }

  /* if H and W are read from cli, redefine HW */
  if (H && W)
    HW = H*W;
  else { /* else, set formally H and W from the value of HW hardcoded above */
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

  if ( norm_type != 0 && norm_type != 1) {
    printf("Only full batchnorm (norm_type = 0) and scale-only (normalize-only for fwd, only input gradient for bwd) norm types are supported \n");
    return -1;
  }

  if ((fuse_type == 4 || fuse_type == 5) && bc % 16 != 0) {
    fprintf( stderr, "Fused ReLU with a mask will not work for sizes which are not a multiple of 16 (2BYTE limitation). Bailing...!\n");
    return -1;
  }

  if (fuse_type != 0 && fuse_type != 2 && fuse_type != 4 && fuse_type != 5) {
    printf("Unsupported fuse_type %d was provided (0, 2, 4 and 5 are supported only)\n", fuse_type);
    return -1;
  }

  if (prec_bf16 > 0) {
    bn_dtype = LIBXSMM_DATATYPE_BF16;
  }

  stride_w = stride;
  stride_h = stride;

  /* set struct for naive batch normalization */
  naive_param.N = N;
  naive_param.C = CP*bc;
  naive_param.H = H;
  naive_param.W = W;
  naive_param.stride_h  = stride_h;
  naive_param.stride_w  = stride_w;
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
  printf("PARAMS: N:%d  C:%d  CP:%d bc:%d H:%d W:%d STRIDE:%d (PADDING: must be 0s)\n", N, CP*bc, CP, bc, H, W, stride);
  printf("PARAMS: NORM TYPE:%d\n", norm_type);
  printf("PARAMS: FUSE TYPE:%d\n", fuse_type);
  printf("PARAMS: PREC     :%d\n", prec_bf16);
  printf("PARAMS: ITERS:%d", iters); if (LIBXSMM_FEQ(0, check)) printf("  Threads:%d\n", nThreads); else printf("\n");
  printf("SIZE Input  (MB): %10.2f MiB\n", (double)(N*CP*HW*bc*sizeof(float))/(1024.0*1024.0) );
  printf("SIZE Output (MB): %10.2f MiB\n", (double)(N*CP*HW*bc*sizeof(float))/(1024.0*1024.0) );

  /* allocate data */
  eqn_inp        = (float*) libxsmm_aligned_malloc( sizeof(float)*N*CP*HW*bc,   2097152);
  eqn_inp_add    = (float*) libxsmm_aligned_malloc( sizeof(float)*N*CP*HW*bc,   2097152);
  eqn_dinp       = (float*) libxsmm_aligned_malloc( sizeof(float)*N*CP*HW*bc,   2097152);
  eqn_dout       = (float*) libxsmm_aligned_malloc( sizeof(float)*N*CP*HW*bc,   2097152);
  eqn_dinp_add   = (float*) libxsmm_aligned_malloc( sizeof(float)*N*CP*HW*bc,   2097152);
  eqn_out        = (float*) libxsmm_aligned_malloc( sizeof(float)*N*CP*HW*bc,   2097152);

  eqn_inp_bf16      = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*CP*HW*bc,   2097152);
  eqn_inp_add_bf16  = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*CP*HW*bc,   2097152);
  eqn_dinp_bf16     = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*CP*HW*bc,   2097152);
  eqn_dout_bf16     = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*CP*HW*bc,   2097152);
  eqn_dinp_add_bf16 = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*CP*HW*bc,   2097152);
  eqn_out_bf16      = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*CP*HW*bc,   2097152);

  naive_eqn_dinp     = (float*) libxsmm_aligned_malloc( sizeof(float)*N*CP*HW*bc,   2097152);
  naive_eqn_dout     = (float*) libxsmm_aligned_malloc( sizeof(float)*N*CP*HW*bc,   2097152);
  naive_eqn_dinp_add = (float*) libxsmm_aligned_malloc( sizeof(float)*N*CP*HW*bc,   2097152);
  naive_eqn_out      = (float*) libxsmm_aligned_malloc( sizeof(float)*N*CP*HW*bc,   2097152);

  eqn_dgamma = (float*) libxsmm_aligned_malloc( sizeof(float)*CP*bc,   2097152);
  eqn_dbeta  = (float*) libxsmm_aligned_malloc( sizeof(float)*CP*bc,   2097152);

  gamma      = (float*) libxsmm_aligned_malloc( sizeof(float)*CP*bc,   2097152);
  beta       = (float*) libxsmm_aligned_malloc( sizeof(float)*CP*bc,   2097152);
  naive_mean       = (float*) libxsmm_aligned_malloc( sizeof(float)*CP*bc,   2097152);
  naive_var        = (float*) libxsmm_aligned_malloc( sizeof(float)*CP*bc,   2097152);
  eqn_mean   = (float*) libxsmm_aligned_malloc( sizeof(float)*CP*bc,   2097152);
  eqn_var    = (float*) libxsmm_aligned_malloc( sizeof(float)*CP*bc,   2097152);

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
  naive_relumask = (unsigned char*) libxsmm_aligned_malloc( sizeof(unsigned char)*N*C*H*W, 2097152);

  /* Allocate bf16 counterparts */
  naive_inp_bf16      = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*C*H*W, 2097152);
  naive_out_bf16      = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*C*H*W, 2097152);
  naive_inp_add_bf16  = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*C*H*W, 2097152);
  naive_dinp_bf16     = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*C*H*W, 2097152);
  naive_dout_bf16     = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*C*H*W, 2097152);
  naive_dinp_add_bf16 = (libxsmm_bfloat16*) libxsmm_aligned_malloc( sizeof(libxsmm_bfloat16)*N*C*H*W, 2097152);

  /* initialize data */
  init_buf(naive_inp,      N*CP*HW*bc, 1, 0);
  init_buf(naive_inp_add,  N*CP*HW*bc, 1, 0);
  init_buf(naive_dinp,     N*CP*HW*bc, 1, 0);
  init_buf(naive_dout,     N*CP*HW*bc, 1, 0);
  init_buf(gamma,          CP*bc,      1, 0);
  init_buf(beta,           CP*bc,      1, 0);
  init_buf(naive_dgamma,   CP*bc,      1, 0);
  init_buf(naive_dbeta,    CP*bc,      1, 0);

  libxsmm_rne_convert_fp32_bf16(naive_inp,     naive_inp_bf16,     N*C*H*W);
  libxsmm_rne_convert_fp32_bf16(naive_inp_add, naive_inp_add_bf16, N*C*H*W);
  libxsmm_rne_convert_fp32_bf16(naive_dinp,    naive_dinp_bf16,    N*C*H*W);
  libxsmm_rne_convert_fp32_bf16(naive_dout,    naive_dout_bf16,    N*C*H*W);

  if (norm_type == 1) { /* normalize-only batchnorm, hence initializing mean and variance */
    int i;
    init_buf(naive_mean,   CP*bc,      1, 0);
    init_buf(naive_var,    CP*bc,      1, 0);

    copy_buf(naive_mean,  eqn_mean, CP*bc);
    copy_buf(naive_var,   eqn_var,  CP*bc);

    for (i = 0; i < C; i++) {
      naive_rcpstdev[i] = (float)(1.0/sqrt(naive_var[i] + eps));
    }
  }

  copy_buf(naive_dgamma, eqn_dgamma, CP*bc);
  copy_buf(naive_dbeta,  eqn_dbeta,  CP*bc);

  zero_buf_uint8(relumask,              N*CP*HW*bc);
  zero_buf_uint8(relumask_uncompressed, N*CP*HW*bc);

  init_buf(cache_fl, 1024*1024, 1, 0);

  /* first touch LIBXSMM */
  zero_buf(eqn_inp,      N*CP*HW*bc);
  zero_buf(eqn_inp_add,  N*CP*HW*bc);
  zero_buf(eqn_out,      N*CP*HW*bc);
  zero_buf(eqn_dinp,     N*CP*HW*bc);
  zero_buf(eqn_dout,     N*CP*HW*bc);

  if (prec_bf16 > 0) {
    libxsmm_rne_convert_fp32_bf16( naive_inp,          naive_inp_bf16,     N*C*H*W );
    libxsmm_convert_bf16_f32     ( naive_inp_bf16,     naive_inp,          N*C*H*W );
    libxsmm_rne_convert_fp32_bf16( naive_inp_add,      naive_inp_add_bf16, N*C*H*W );
    libxsmm_convert_bf16_f32     ( naive_inp_add_bf16, naive_inp_add,      N*C*H*W );
    libxsmm_rne_convert_fp32_bf16( naive_dinp,         naive_dinp_bf16,    N*C*H*W );
    libxsmm_convert_bf16_f32     ( naive_dinp_bf16,    naive_dinp,         N*C*H*W );
    libxsmm_rne_convert_fp32_bf16( naive_dout,         naive_dout_bf16,    N*C*H*W );
    libxsmm_convert_bf16_f32     ( naive_dout_bf16,    naive_dout,         N*C*H*W );
  }

  if (LIBXSMM_NEQ(0, check)) {
    printf("##########################################\n");
    printf("#         Computing Reference ...        #\n");
    printf("##########################################\n");

    naive_fusedbatchnorm_fp(&naive_param, naive_inp, naive_out, naive_inp_add,
                                        beta, gamma, eps, naive_mean, naive_rcpstdev, naive_var, naive_relumask);

    naive_fusedbatchnorm_bp(&naive_param, naive_inp, naive_dinp, naive_out, naive_dout, naive_dinp_add,
                                       beta, naive_dbeta, gamma, naive_dgamma, naive_mean, naive_rcpstdev);

    tensor_copy_NCHW_to_NCHWc_uint8 (naive_relumask, relumask_uncompressed, N, C, H, W, bc);
    /* since naive implementation returnes the mask with 1 char per entry, after changing layout, a compression into bitmask is needed */
    mask_compress_uint8 (relumask_uncompressed, relumask, N*CP*H*W*bc);

    printf("##########################################\n");
    printf("#      Computing Reference ... done      #\n");
    printf("##########################################\n");
  }

  printf("##########################################\n");
  printf("#          Setting Up (TPP)              #\n");
  printf("##########################################\n");
  /* setup TPPs (standalone or through the configs) */

  libxsmm_dnn_bn_fwd = setup_libxsmm_dnn_bn_fwd(N, C, H, W, bc, nThreads, (libxsmm_dnn_bn_fuse)fuse_type, bn_dtype, bn_dtype, LIBXSMM_DATATYPE_F32);
  libxsmm_dnn_bn_bwd = setup_libxsmm_dnn_bn_bwd(N, C, H, W, bc, nThreads, (libxsmm_dnn_bn_fuse)fuse_type, bn_dtype, bn_dtype, LIBXSMM_DATATYPE_F32);

  /* allocate and bind scratch */
  if ( libxsmm_dnn_bn_fwd.scratch_size > 0 || libxsmm_dnn_bn_bwd.scratch_size > 0 ) {
    size_t alloc_size = LIBXSMM_MAX( libxsmm_dnn_bn_fwd.scratch_size, libxsmm_dnn_bn_bwd.scratch_size);
    scratch = libxsmm_aligned_malloc( alloc_size, 2097152 );
    init_buf( (float*)(scratch), (alloc_size)/4, 0, 0 );
  }

  /* copy tensors into the right format */
  if (prec_bf16 > 0) {
    tensor_copy_NCHW_to_NCHWc_bf16( naive_inp_bf16,     eqn_inp_bf16,     N, C, H, W, bc );
    tensor_copy_NCHW_to_NCHWc_bf16( naive_inp_add_bf16, eqn_inp_add_bf16, N, C, H, W, bc );
  } else {
    tensor_copy_NCHW_to_NCHWc( naive_inp,     eqn_inp,     N, C, H, W, bc );
    tensor_copy_NCHW_to_NCHWc( naive_inp_add, eqn_inp_add, N, C, H, W, bc );
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
      if (prec_bf16 > 0)
        libxsmm_dnn_bn_fwd_exec_bf16( libxsmm_dnn_bn_fwd, eqn_inp_bf16, eqn_inp_add_bf16, gamma, beta, eqn_mean, eqn_var, eqn_out_bf16, eqn_relumask, eps, 0, tid, scratch, (libxsmm_dnn_bn_norm_type)norm_type );
      else
        libxsmm_dnn_bn_fwd_exec_f32 ( libxsmm_dnn_bn_fwd, eqn_inp, eqn_inp_add, gamma, beta, eqn_mean, eqn_var, eqn_out, eqn_relumask, eps, 0, tid, scratch, (libxsmm_dnn_bn_norm_type)norm_type );
    }

    /* copy out data */
    if (prec_bf16 > 0)
      libxsmm_convert_bf16_f32( eqn_out_bf16, eqn_out, N*C*H*W );

    tensor_copy_NCHWc_to_NCHW( eqn_out, naive_eqn_out, N, C, H, W, bc );

    /* compare */
    printf("############################################\n");
    printf("# Correctness FWD Batchnorm - Output       #\n");
    printf("############################################\n");
    libxsmm_matdiff(&norms_fwd_out, LIBXSMM_DATATYPE_F32, N*CP*HW*bc, 1, naive_out, naive_eqn_out, 0, 0);
    printf("L1 reference  : %.25g\n", norms_fwd_out.l1_ref);
    printf("L1 test       : %.25g\n", norms_fwd_out.l1_tst);
    printf("L2 abs.error  : %.24f\n", norms_fwd_out.l2_abs);
    printf("L2 rel.error  : %.24f\n", norms_fwd_out.l2_rel);
    printf("Linf abs.error: %.24f\n", norms_fwd_out.linf_abs);
    printf("Linf rel.error: %.24f\n", norms_fwd_out.linf_rel);
    printf("Check-norm    : %.24f\n\n", norms_fwd_out.normf_rel);

    printf("############################################\n");
    printf("# Correctness FWD Batchnorm - Mean         #\n");
    printf("############################################\n");
    libxsmm_matdiff(&norms_fwd_mean, LIBXSMM_DATATYPE_F32, C, 1, naive_mean, eqn_mean, 0, 0);
    printf("L1 reference  : %.25g\n", norms_fwd_mean.l1_ref);
    printf("L1 test       : %.25g\n", norms_fwd_mean.l1_tst);
    printf("L2 abs.error  : %.24f\n", norms_fwd_mean.l2_abs);
    printf("L2 rel.error  : %.24f\n", norms_fwd_mean.l2_rel);
    printf("Linf abs.error: %.24f\n", norms_fwd_mean.linf_abs);
    printf("Linf rel.error: %.24f\n", norms_fwd_mean.linf_rel);
    printf("Check-norm    : %.24f\n\n", norms_fwd_mean.normf_rel);

    printf("############################################\n");
    printf("# Correctness FWD Batchnorm - Var          #\n");
    printf("############################################\n");
    libxsmm_matdiff(&norms_fwd_var, LIBXSMM_DATATYPE_F32, C, 1, naive_var, eqn_var, 0, 0);
    printf("L1 reference  : %.25g\n", norms_fwd_var.l1_ref);
    printf("L1 test       : %.25g\n", norms_fwd_var.l1_tst);
    printf("L2 abs.error  : %.24f\n", norms_fwd_var.l2_abs);
    printf("L2 rel.error  : %.24f\n", norms_fwd_var.l2_rel);
    printf("Linf abs.error: %.24f\n", norms_fwd_var.linf_abs);
    printf("Linf rel.error: %.24f\n", norms_fwd_var.linf_rel);
    printf("Check-norm    : %.24f\n\n", norms_fwd_var.normf_rel);

    if (fuse_type == 4 || fuse_type == 5) {
      printf("############################################\n");
      printf("# Correctness FWD Batchnorm - Relumask     #\n");
      printf("############################################\n");
      libxsmm_matdiff(&norms_fwd_mask, LIBXSMM_DATATYPE_I8, N*CP*HW*bc, 1, relumask, eqn_relumask, 0, 0);
      printf("L1 reference  : %.25g\n", norms_fwd_mask.l1_ref);
      printf("L1 test       : %.25g\n", norms_fwd_mask.l1_tst);
      printf("L2 abs.error  : %.24f\n", norms_fwd_mask.l2_abs);
      printf("L2 rel.error  : %.24f\n", norms_fwd_mask.l2_rel);
      printf("Linf abs.error: %.24f\n", norms_fwd_mask.linf_abs);
      printf("Linf rel.error: %.24f\n", norms_fwd_mask.linf_rel);
      printf("Check-norm    : %.24f\n\n", norms_fwd_mask.normf_rel);
    }
  } /* checking correctness for FWD */

  for (i = 0; i < 1024 * 1024; i++ ) {
    sum += cache_fl[i];
  }
  naive_fusedbatchnorm_fp(&naive_param, naive_inp, naive_out, naive_inp_add,
                                        beta, gamma, eps, naive_mean, naive_rcpstdev, naive_var, naive_relumask);
  l_start = libxsmm_timer_tick();
  for (it = 0; it < iters; it++) {
    naive_fusedbatchnorm_fp(&naive_param, naive_inp, naive_out, naive_inp_add,
                                        beta, gamma, eps, naive_mean, naive_rcpstdev, naive_var, naive_relumask);
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
      if (prec_bf16 > 0)
        libxsmm_dnn_bn_fwd_exec_bf16( libxsmm_dnn_bn_fwd, eqn_inp_bf16, eqn_inp_add_bf16, gamma, beta, eqn_mean, eqn_var, eqn_out_bf16, eqn_relumask, eps, 0, tid, scratch, (libxsmm_dnn_bn_norm_type)norm_type );
      else
        libxsmm_dnn_bn_fwd_exec_f32 ( libxsmm_dnn_bn_fwd, eqn_inp, eqn_inp_add, gamma, beta, eqn_mean, eqn_var, eqn_out, eqn_relumask, eps, 0, tid, scratch, (libxsmm_dnn_bn_norm_type)norm_type );
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
      if (prec_bf16 > 0)
        libxsmm_dnn_bn_fwd_exec_bf16( libxsmm_dnn_bn_fwd, eqn_inp_bf16, eqn_inp_add_bf16, gamma, beta, eqn_mean, eqn_var, eqn_out_bf16, eqn_relumask, eps, 0, tid, scratch, (libxsmm_dnn_bn_norm_type)norm_type );
      else
        libxsmm_dnn_bn_fwd_exec_f32 ( libxsmm_dnn_bn_fwd, eqn_inp, eqn_inp_add, gamma, beta, eqn_mean, eqn_var, eqn_out, eqn_relumask, eps, 0, tid, scratch, (libxsmm_dnn_bn_norm_type)norm_type );
    }
  }
  l_end = libxsmm_timer_tick();
  l_total2 = libxsmm_timer_duration(l_start, l_end);
  printf("TPP batchnorm time FWD  = %.5g\n", ((double)(l_total2)));
  printf("Speedup FWD is %.5g\n", l_total/l_total2);

  /* copy tensors into the right format */
  if (prec_bf16 > 0) {
    tensor_copy_NCHW_to_NCHWc_bf16( naive_inp_bf16,     eqn_inp_bf16,     N, C, H, W, bc );
    tensor_copy_NCHW_to_NCHWc_bf16( naive_dout_bf16, eqn_dout_bf16, N, C, H, W, bc );
  } else {
    tensor_copy_NCHW_to_NCHWc( naive_inp,     eqn_inp,     N, C, H, W, bc );
    tensor_copy_NCHW_to_NCHWc( naive_dout, eqn_dout, N, C, H, W, bc );
  }

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
      if (prec_bf16 > 0)
        libxsmm_dnn_bn_bwd_exec_bf16( libxsmm_dnn_bn_bwd, eqn_dout_bf16, eqn_inp_bf16, naive_mean, naive_var, gamma, relumask, eqn_dinp_bf16, eqn_dinp_add_bf16, eqn_dgamma, eqn_dbeta, eps, 0, tid, scratch, (libxsmm_dnn_bn_norm_type)norm_type );
      else
        libxsmm_dnn_bn_bwd_exec_f32 ( libxsmm_dnn_bn_bwd, eqn_dout, eqn_inp, naive_mean, naive_var, gamma, relumask, eqn_dinp, eqn_dinp_add, eqn_dgamma, eqn_dbeta, eps, 0, tid, scratch, (libxsmm_dnn_bn_norm_type)norm_type );
    }

    /* copy out data */
    if (prec_bf16 > 0) {
      libxsmm_convert_bf16_f32( eqn_dinp_bf16, eqn_dinp, N*C*H*W );
      libxsmm_convert_bf16_f32( eqn_dinp_add_bf16, eqn_dinp_add, N*C*H*W );
      libxsmm_convert_bf16_f32( eqn_dout_bf16, eqn_dout, N*C*H*W );
    }

    tensor_copy_NCHWc_to_NCHW( eqn_dinp, naive_eqn_dinp, N, C, H, W, bc );
    tensor_copy_NCHWc_to_NCHW( eqn_dinp_add, naive_eqn_dinp_add, N, C, H, W, bc );
    tensor_copy_NCHWc_to_NCHW( eqn_dout, naive_eqn_dout, N, C, H, W, bc );

    /* compare */
    printf("############################################\n");
    printf("# Correctness BWD Batchnorm - Dinput       #\n");
    printf("############################################\n");
    libxsmm_matdiff(&norms_bwd_din, LIBXSMM_DATATYPE_F32, N*CP*HW*bc, 1, naive_dinp, naive_eqn_dinp, 0, 0);
    printf("L1 reference  : %.25g\n", norms_bwd_din.l1_ref);
    printf("L1 test       : %.25g\n", norms_bwd_din.l1_tst);
    printf("L2 abs.error  : %.24f\n", norms_bwd_din.l2_abs);
    printf("L2 rel.error  : %.24f\n", norms_bwd_din.l2_rel);
    printf("Linf abs.error: %.24f\n", norms_bwd_din.linf_abs);
    printf("Linf rel.error: %.24f\n", norms_bwd_din.linf_rel);
    printf("Check-norm    : %.24f\n\n", norms_bwd_din.normf_rel);

    printf("############################################\n");
    printf("# Correctness BWD Batchnorm - Dout         #\n");
    printf("############################################\n");
    libxsmm_matdiff(&norms_bwd_dout, LIBXSMM_DATATYPE_F32, N*CP*HW*bc, 1, naive_dout, naive_eqn_dout, 0, 0);
    printf("L1 reference  : %.25g\n", norms_bwd_dout.l1_ref);
    printf("L1 test       : %.25g\n", norms_bwd_dout.l1_tst);
    printf("L2 abs.error  : %.24f\n", norms_bwd_dout.l2_abs);
    printf("L2 rel.error  : %.24f\n", norms_bwd_dout.l2_rel);
    printf("Linf abs.error: %.24f\n", norms_bwd_dout.linf_abs);
    printf("Linf rel.error: %.24f\n", norms_bwd_dout.linf_rel);
    printf("Check-norm    : %.24f\n\n", norms_bwd_dout.normf_rel);

    if (fuse_type == 2 || fuse_type == 5) {
      printf("################################################\n");
      printf("# Correctness BWD Batchnorm - Dinput add       #\n");
      printf("################################################\n");
      libxsmm_matdiff(&norms_bwd_din, LIBXSMM_DATATYPE_F32, N*CP*HW*bc, 1, naive_dinp_add, naive_eqn_dinp_add, 0, 0);
      printf("L1 reference  : %.25g\n", norms_bwd_din.l1_ref);
      printf("L1 test       : %.25g\n", norms_bwd_din.l1_tst);
      printf("L2 abs.error  : %.24f\n", norms_bwd_din.l2_abs);
      printf("L2 rel.error  : %.24f\n", norms_bwd_din.l2_rel);
      printf("Linf abs.error: %.24f\n", norms_bwd_din.linf_abs);
      printf("Linf rel.error: %.24f\n", norms_bwd_din.linf_rel);
      printf("Check-norm    : %.24f\n\n", norms_bwd_din.normf_rel);
    }

    printf("###########################################\n");
    printf("# Correctness BWD Batchnorm - Dbeta       #\n");
    printf("###########################################\n");
    libxsmm_matdiff(&norms_bwd_beta, LIBXSMM_DATATYPE_F32, CP*bc, 1, naive_dbeta, eqn_dbeta, 0, 0);
    printf("L1 reference  : %.25g\n", norms_bwd_beta.l1_ref);
    printf("L1 test       : %.25g\n", norms_bwd_beta.l1_tst);
    printf("L2 abs.error  : %.24f\n", norms_bwd_beta.l2_abs);
    printf("L2 rel.error  : %.24f\n", norms_bwd_beta.l2_rel);
    printf("Linf abs.error: %.24f\n", norms_bwd_beta.linf_abs);
    printf("Linf rel.error: %.24f\n", norms_bwd_beta.linf_rel);
    printf("Check-norm    : %.24f\n\n", norms_bwd_beta.normf_rel);

    printf("############################################\n");
    printf("# Correctness BWD Batchnorm - Dgamma       #\n");
    printf("############################################\n");
    libxsmm_matdiff(&norms_bwd_gamma, LIBXSMM_DATATYPE_F32, CP*bc, 1, naive_dgamma, eqn_dgamma, 0, 0);
    printf("L1 reference  : %.25g\n", norms_bwd_gamma.l1_ref);
    printf("L1 test       : %.25g\n", norms_bwd_gamma.l1_tst);
    printf("L2 abs.error  : %.24f\n", norms_bwd_gamma.l2_abs);
    printf("L2 rel.error  : %.24f\n", norms_bwd_gamma.l2_rel);
    printf("Linf abs.error: %.24f\n", norms_bwd_gamma.linf_abs);
    printf("Linf rel.error: %.24f\n", norms_bwd_gamma.linf_rel);
    printf("Check-norm    : %.24f\n\n", norms_bwd_gamma.normf_rel);
  } /* correctness for BWD */

  for (i = 0; i < 1024 * 1024; i++ ) {
    sum += cache_fl[i];
  }
  naive_fusedbatchnorm_bp(&naive_param, naive_inp, naive_dinp, naive_out, naive_dout, naive_dinp_add,
                                       beta, naive_dbeta, gamma, naive_dgamma, naive_mean, naive_rcpstdev);
  l_start = libxsmm_timer_tick();
  for (it = 0; it < iters; it++) {
    naive_fusedbatchnorm_bp(&naive_param, naive_inp, naive_dinp, naive_out, naive_dout, naive_dinp_add,
                                       beta, naive_dbeta, gamma, naive_dgamma, naive_mean, naive_rcpstdev);
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
      if (prec_bf16 > 0)
        libxsmm_dnn_bn_bwd_exec_bf16( libxsmm_dnn_bn_bwd, eqn_dout_bf16, eqn_inp_bf16, naive_mean, naive_var, gamma, relumask, eqn_dinp_bf16, eqn_dinp_add_bf16, eqn_dgamma, eqn_dbeta, eps, 0, tid, scratch, (libxsmm_dnn_bn_norm_type)norm_type );
      else
        libxsmm_dnn_bn_bwd_exec_f32 ( libxsmm_dnn_bn_bwd, eqn_dout, eqn_inp, naive_mean, naive_var, gamma, relumask, eqn_dinp, eqn_dinp_add, eqn_dgamma, eqn_dbeta, eps, 0, tid, scratch, (libxsmm_dnn_bn_norm_type)norm_type );
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
      if (prec_bf16 > 0)
        libxsmm_dnn_bn_bwd_exec_bf16( libxsmm_dnn_bn_bwd, eqn_dout_bf16, eqn_inp_bf16, naive_mean, naive_var, gamma, relumask, eqn_dinp_bf16, eqn_dinp_add_bf16, eqn_dgamma, eqn_dbeta, eps, 0, tid, scratch, (libxsmm_dnn_bn_norm_type)norm_type );
      else
        libxsmm_dnn_bn_bwd_exec_f32 ( libxsmm_dnn_bn_bwd, eqn_dout, eqn_inp, naive_mean, naive_var, gamma, relumask, eqn_dinp, eqn_dinp_add, eqn_dgamma, eqn_dbeta, eps, 0, tid, scratch, (libxsmm_dnn_bn_norm_type)norm_type );
    }
  }

  l_end = libxsmm_timer_tick();
  l_total2 = libxsmm_timer_duration(l_start, l_end);
  printf("TPP batchnorm time BWD = %.5g\n", ((double)(l_total2)));
  printf("Speedup BWD is %.5g\n", l_total/l_total2);

  t_tpp += l_total2;
  t_vec += l_total;

  printf("\n\n=================================\n");
  printf("Total Speedup via TPP Matrix equation is %.5g\n", t_vec/t_tpp);
  printf("=================================\n");

  /* deallocate data */

  destroy_libxsmm_dnn_bn_fwd(&libxsmm_dnn_bn_fwd);
  destroy_libxsmm_dnn_bn_bwd(&libxsmm_dnn_bn_bwd);

  if ( scratch != NULL ) {
    libxsmm_free(scratch);
  }

  libxsmm_free(eqn_inp);
  libxsmm_free(eqn_inp_add);
  libxsmm_free(eqn_dinp);
  libxsmm_free(eqn_dout);
  libxsmm_free(eqn_dinp_add);
  libxsmm_free(eqn_out);
  libxsmm_free(eqn_dgamma);
  libxsmm_free(eqn_dbeta);
  libxsmm_free(eqn_mean);
  libxsmm_free(eqn_var);

  libxsmm_free(eqn_inp_bf16);
  libxsmm_free(eqn_inp_add_bf16);
  libxsmm_free(eqn_dinp_bf16);
  libxsmm_free(eqn_dinp_add_bf16);
  libxsmm_free(eqn_dout_bf16);
  libxsmm_free(eqn_out_bf16);

  libxsmm_free(naive_mean);
  libxsmm_free(naive_var);
  libxsmm_free(gamma);
  libxsmm_free(beta);

  libxsmm_free(cache_fl);

  libxsmm_free(relumask);
  libxsmm_free(relumask_uncompressed);
  libxsmm_free(eqn_relumask);

  libxsmm_free(naive_inp);
  libxsmm_free(naive_out);
  libxsmm_free(naive_inp_add);
  libxsmm_free(naive_dinp);
  libxsmm_free(naive_dout);
  libxsmm_free(naive_dinp_add);
  libxsmm_free(naive_dgamma);
  libxsmm_free(naive_dbeta);
  libxsmm_free(naive_rcpstdev);
  libxsmm_free(naive_relumask);

  libxsmm_free(naive_inp_bf16);
  libxsmm_free(naive_out_bf16);
  libxsmm_free(naive_inp_add_bf16);
  libxsmm_free(naive_dinp_bf16);
  libxsmm_free(naive_dout_bf16);
  libxsmm_free(naive_dinp_add_bf16);

  libxsmm_free(naive_eqn_dinp);
  libxsmm_free(naive_eqn_dout);
  libxsmm_free(naive_eqn_dinp_add);
  libxsmm_free(naive_eqn_out);

  return 0;
}

